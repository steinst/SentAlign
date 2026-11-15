#!/usr/bin/env python3
"""Precision / recall / F1 against gold alignments that cover only *windows* of a document.

Book-length documents cannot be aligned by hand completely.  The gold standard of the SentAlign book test set
therefore covers windows: stretches of the document (say 40 source sentences each) spread over the whole book.  The
aligner is run on the *whole* document - so that mistakes made far away, e.g. a wrong anchor, show up - and only the
links that touch a window are scored.

Gold file format (one file per document; the alignment lines are those of the BleuAlign / ParIce gold files)::

    # window 1
    [120]:[131]
    [121, 122]:[132]
    [123]:[]
    # window 2
    ...

Every sentence of a window is in exactly one gold line (a link, or a null line like ``[123]:[]`` / ``[]:[140]``), and
the sentences of a window are consecutive on both sides.  A window is the set of source and target sentence numbers
of its gold lines.

A system link is scored with a window when it contains at least one source or one target sentence of that window.
A system link that reaches across the edge of a window therefore counts (wrong in the strict measure; the lax
measure accepts it when it overlaps a gold link on both sides).  Precision, recall and F1 are computed by the
functions of ``evaluate.py`` over all windows of all documents together (micro-average), with the same treatment of
null links (excluded unless ``--nulls``).

    python evaluation/evaluate_windows.py -t out/doc1.path out/doc2.path -g gold/doc1.txt gold/doc2.txt [--per-window]
"""
import argparse
import os
import sys
from ast import literal_eval

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from evaluate import LAX_RECALL, log_final_scores, read_alignments, score_multiple  # noqa: E402


class GoldError(ValueError):
    """The gold file does not follow the rules above."""


def _parse_line(line):
    fields = [x.strip() for x in line.split(':') if len(x.strip())]
    if len(fields) < 2:
        raise GoldError('line %r does not have two ":"-separated fields' % line)
    try:
        return tuple(literal_eval(fields[0])), tuple(literal_eval(fields[1]))
    except (ValueError, SyntaxError) as e:
        raise GoldError('cannot read line %r' % line) from e


def read_windows(path):
    """[(name, [(src ids, trg ids), ...]), ...] in file order.  Lines before the first '# window' form a window
    called 'all' (a gold file without window lines is one window: the whole annotated part)."""
    windows = []
    name, links = None, []
    with open(path, encoding='utf-8') as f:
        for raw in f:
            line = raw.strip()
            if not line:
                continue
            if line.startswith('#'):
                words = line.lstrip('#').split()
                if words and words[0].lower() == 'window':
                    if links:
                        windows.append((name or 'all', links))
                    name, links = ' '.join(words[1:]) or str(len(windows) + 1), []
                continue                                   # other comments are ignored
            links.append(_parse_line(line))
    if links:
        windows.append((name or 'all', links))
    return windows


def check_window(links, where='', allow_gaps=False):
    """Every sentence once; consecutive on both sides.  `allow_gaps` skips both checks: the BleuAlign gold files
    leave out a few sentences and list one twice (test1, source sentence 218), see evaluate.py.
    Returns (source ids, target ids) as sets."""
    src, trg = [], []
    for s, t in links:
        src.extend(s)
        trg.extend(t)
    for side, ids in (('source', src), ('target', trg)):
        if allow_gaps:
            continue
        if len(ids) != len(set(ids)):
            dup = sorted({i for i in ids if ids.count(i) > 1})
            raise GoldError('%s: %s sentence(s) %s in more than one gold line' % (where, side, dup[:10]))
        if ids and sorted(ids) != list(range(min(ids), max(ids) + 1)):
            missing = sorted(set(range(min(ids), max(ids) + 1)) - set(ids))
            raise GoldError('%s: %s sentence(s) %s inside the window have no gold line' % (where, side, missing[:10]))
    return set(src), set(trg)


def links_in_window(test, src_ids, trg_ids):
    """The system links that contain a source or a target sentence of the window."""
    return [(s, t) for s, t in test if src_ids.intersection(s) or trg_ids.intersection(t)]


def score_documents(gold_files, test_files, keep_nulls=True, per_window=False, out=None, allow_gaps=False,
                    lax_recall='standard'):
    out = out or sys.stdout
    if len(gold_files) != len(test_files):
        raise ValueError('%d gold files but %d test files' % (len(gold_files), len(test_files)))
    gold_list, test_list, rows = [], [], []
    for gf, tf in zip(gold_files, test_files):
        test = [(tuple(s), tuple(t)) for s, t in read_alignments(tf)]
        for name, links in read_windows(gf):
            src_ids, trg_ids = check_window(links, '%s, window %s' % (gf, name), allow_gaps)
            sys_links = links_in_window(test, src_ids, trg_ids)
            gold_list.append(links)
            test_list.append(sys_links)
            if per_window:
                res = score_multiple([links], [sys_links], keep_nulls, lax_recall=lax_recall)
                crossing = sum(1 for s, t in sys_links if not (set(s) <= src_ids and set(t) <= trg_ids))
                rows.append((os.path.basename(gf), name, len(src_ids), len(trg_ids), crossing, res))
    if per_window:
        print('file\twindow\tsrc_sentences\ttrg_sentences\tlinks_crossing_edge\tP_strict\tR_strict\tF1_strict\tF1_lax',
              file=out)
        for f, name, ns, nt, crossing, r in rows:
            print('%s\t%s\t%d\t%d\t%d\t%.3f\t%.3f\t%.3f\t%.3f' % (f, name, ns, nt, crossing, r['precision_strict'],
                                                                 r['recall_strict'], r['f1_strict'], r['f1_lax']),
                  file=out)
    return score_multiple(gold_list, test_list, keep_nulls, lax_recall=lax_recall)


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.split('\n\n')[0],
                                 formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    ap.add_argument('-t', '--test', nargs='+', required=True, help='system output (.path files), one per document')
    ap.add_argument('-g', '--gold', nargs='+', required=True, help='windowed gold files, in the same order')
    ap.add_argument('-n', '--nulls', action='store_false', help='score null links too (as evaluate.py)')
    ap.add_argument('--per-window', action='store_true', help='print a table with one line per window')
    ap.add_argument('--allow-gaps', action='store_true',
                    help='do not check that every sentence of a window has exactly one gold line (old gold files '
                         'such as BleuAlign break that rule)')
    ap.add_argument('--lax-recall', choices=LAX_RECALL, default='standard',
                    help='paper-2023: lax recall as computed by the SentAlign 1.0 script (see evaluate.py)')
    args = ap.parse_args(argv)
    try:
        res = score_documents(args.gold, args.test, args.nulls, args.per_window, allow_gaps=args.allow_gaps,
                              lax_recall=args.lax_recall)
    except (GoldError, ValueError) as e:
        print('error: %s' % e, file=sys.stderr)
        return 2
    log_final_scores(res)
    return 0


if __name__ == '__main__':
    sys.exit(main())
