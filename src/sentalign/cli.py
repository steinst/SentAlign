"""Command line of SentAlign: ``sentalign`` and ``python -m sentalign``.

Two forms, both with the version 1 flags::

    sentalign -dir /path/to/files -sl eng -tl isl [-f file1.txt file2.txt] [-out output]
    sentalign SRC_FILE TRG_FILE -o OUT_DIR

Exit code: 0 if every file produced output, 1 if any file failed, 2 for usage errors.
"""
import argparse
import faulthandler
import logging
import os
import shutil
import sys
import tempfile

import sentalign
from sentalign import files as filetools
from sentalign.config import COMPAT_PRESETS, add_alignment_arguments, config_from_args

log = logging.getLogger('sentalign')


def build_parser():
    parser = argparse.ArgumentParser(
        prog='sentalign', allow_abbrev=False,
        description='SentAlign: sentence alignment with LaBSE embeddings. Align the files of a corpus folder '
                    '(-dir, with one sub-folder per language) or one pair of files (SRC_FILE TRG_FILE -o OUT_DIR).')
    parser.add_argument('pair', nargs='*', metavar='SRC_FILE TRG_FILE', help='one source and one target file to align (needs -o)')
    parser.add_argument('--output', '-o', metavar='OUT_DIR', help='output folder for the SRC_FILE TRG_FILE form')
    add_alignment_arguments(parser)
    parser.add_argument('--compat', choices=COMPAT_PRESETS, default='none',
                        help='none: SentAlign 2 (default).  paper-2023: reproduce SentAlign 1.0 and the 2023 paper '
                             '(its arithmetic, post-processing and large-file anchors, faults included)')
    parser.add_argument('--overwrite', action='store_true', help='align files again that already have output')
    parser.add_argument('--list-files', action='store_true', help='print the files that would be aligned and exit')
    parser.add_argument('--quiet', '-q', action='store_true', help='only log warnings and errors')
    parser.add_argument('--verbose', '-v', action='store_true', help='log debug information (stages inside the anchoring)')
    parser.add_argument('--bibtex', action='store_true',
                        help='after the run, also print the BibTeX entries of the publications to cite for it (the '
                             'list of what to cite is always printed)')
    parser.add_argument('--version', action='version', version='sentalign ' + sentalign.__version__)
    return parser


def setup_logging(quiet=False, verbose=False):
    level = logging.WARNING if quiet else (logging.DEBUG if verbose else logging.INFO)
    root = logging.getLogger('sentalign')
    root.setLevel(level)
    if not root.handlers:
        handler = logging.StreamHandler(sys.stderr)
        handler.setFormatter(logging.Formatter('%(asctime)s %(levelname)-7s %(message)s', '%Y-%m-%d %H:%M:%S'))
        root.addHandler(handler)
        root.propagate = False


def main(argv=None):
    parser = build_parser()
    try:
        args = parser.parse_args(argv)
    except SystemExit as e:  # argparse: 0 for --help/--version, 2 for usage errors
        return e.code if isinstance(e.code, int) else 2
    try:
        return _run(parser, args)
    except SystemExit as e:
        return e.code if isinstance(e.code, int) else 2


def _run(parser, args):
    setup_logging(args.quiet, args.verbose)
    try:
        faulthandler.enable()  # a segmentation fault at least leaves a Python traceback on stderr
    except (ValueError, OSError):   # stderr is not a real file (captured output, some notebooks)
        pass
    try:
        config = config_from_args(args)
    except ValueError as e:
        parser.error(str(e))
    code = _run_config(parser, args, config)
    if not args.list_files:
        # what to cite for this run: always shown (also with --quiet); the methods of other authors are named
        from sentalign.citations import citations_for, format_citations
        print(format_citations(citations_for(config), bibtex=args.bibtex), file=sys.stderr)
    return code


def _run_config(parser, args, config):
    dev = config.deviations_from_paper()
    if dev:
        log.warning('--compat paper-2023 with settings that differ from SentAlign 1.0 (%s): the results will not be '
                    'those of the paper', ', '.join('%s=%s' % kv for kv in sorted(dev.items())))
    pair_mode = len(args.pair) > 0

    if pair_mode:
        if len(args.pair) != 2:
            parser.error('give exactly two files: SRC_FILE TRG_FILE')
        if args.corpus_folder or args.filename:
            parser.error('SRC_FILE TRG_FILE cannot be combined with -dir or -f')
        if not args.output:
            parser.error('the SRC_FILE TRG_FILE form needs -o OUT_DIR')
        for p in args.pair:
            if not os.path.isfile(p):
                parser.error('file not found: ' + p)
        _pair_languages(parser, config, args)
        return _run_pair(parser, config, args)

    if args.output:
        parser.error('-o OUT_DIR is only for the SRC_FILE TRG_FILE form; use -out with -dir')
    if not args.corpus_folder:
        parser.error('give -dir CORPUS_FOLDER or SRC_FILE TRG_FILE -o OUT_DIR')
    corpus = config.corpus_folder
    src_folder = os.path.join(corpus, config.source_language)
    trg_folder = os.path.join(corpus, config.target_language)
    for folder in (corpus, src_folder, trg_folder):
        if not os.path.isdir(folder):
            parser.error('folder not found: ' + folder)
    output_folder = os.path.join(corpus, config.output_folder)

    names = config.filename if config.filename else filetools.list_source_files(src_folder)
    todo, done = filetools.select_files(names, output_folder, config.overwrite)
    if args.list_files:
        for name in todo:
            print(name)
        return 0
    return _align(parser, config, todo, done)


def _align(parser, config, todo, done, **folders):
    from sentalign.aligner import Aligner  # heavy imports (torch, transformers) only when there is work
    if done:
        log.info('%d file(s) already aligned (use --overwrite to align them again)', len(done))
    if not todo:
        log.info('Nothing to align.')
        return 0
    _settle_device(parser, config)
    aligner = Aligner(config, **folders)
    return _finish(aligner, todo)


def _settle_device(parser, config):
    """The device of the run, before any work: without a CUDA device the default (auto) is the CPU; a CUDA device
    that was asked for by name and is not there ends the run at once instead of failing every file."""
    asked = config.proc_device
    try:
        config.resolve_device()
    except ValueError as e:
        parser.error(str(e))
    if asked == 'auto' and config.proc_device == 'cpu':
        log.warning('no CUDA device found: running on the CPU, which is much slower (an NVIDIA GPU needs a CUDA '
                    'build of PyTorch)')
    else:
        log.info('device: %s', config.proc_device)


def _finish(aligner, todo):
    aligned, failed = aligner.run(todo)
    if failed:
        log.error('%d file(s) failed: %s', len(failed), ', '.join(failed))
        return 1
    return 0


def _pair_languages(parser, config, args):
    """The languages of the pair form are what -sl / -tl say and nothing else: there are no language folders, and
    the defaults of the -dir form (eng / isl) would be recorded, and given to the encoder, for any text.  Not given:
    None, and an encoder that is told the language of the texts (LASER) is refused."""
    from sentalign.scoring import get_encoder
    config.source_language, config.target_language = args.source_language, args.target_language
    if args.source_language is not None and args.target_language is not None:
        return
    try:
        needs = getattr(get_encoder(config.encoder_kind), 'needs_language', False)
    except Exception:       # an unknown encoder is reported when the model is built
        return
    if needs:
        parser.error('encoder %r needs the language of the texts: give -sl and -tl with SRC_FILE TRG_FILE'
                     % config.encoder_kind)


def _run_pair(parser, config, args):
    src, trg = args.pair
    name = os.path.basename(src)
    out_dir = os.path.abspath(args.output)
    todo, done = filetools.select_files([name], out_dir, config.overwrite)
    if args.list_files:
        for n in todo:
            print(n)
        return 0
    if not todo:
        log.info('%s is already aligned in %s (use --overwrite to align it again)', name, out_dir)
        return 0
    _settle_device(parser, config)
    # one pair of files: build a temporary corpus folder with the layout the pipeline expects
    work = tempfile.mkdtemp(prefix='sentalign_')
    try:
        os.makedirs(os.path.join(work, 'src'))
        os.makedirs(os.path.join(work, 'trg'))
        shutil.copyfile(src, os.path.join(work, 'src', name))
        shutil.copyfile(trg, os.path.join(work, 'trg', name))
        os.makedirs(out_dir, exist_ok=True)
        config.corpus_folder = work
        from sentalign.aligner import Aligner
        # the folders are named, so the configured languages stay what the user gave (see `_pair_languages`)
        aligner = Aligner(config, source_folder=os.path.join(work, 'src'), target_folder=os.path.join(work, 'trg'),
                          output_folder=out_dir, info_folder=out_dir)
        aligner.input_paths = {'source': os.path.abspath(src), 'target': os.path.abspath(trg)}
        return _finish(aligner, todo)
    finally:
        shutil.rmtree(work, ignore_errors=True)


if __name__ == '__main__':
    sys.exit(main())
