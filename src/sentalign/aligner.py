"""The SentAlign 2023 pipeline as an object: no globals, no work at import time.

This is a port of ``sentAlign.py`` (tag v1.0-paper).  The algorithms, their order and their results are unchanged
(``--compat paper-2023``); functions that carry known faults keep them and say so in a comment (``KNOWN FAULT``).
"""
import datetime
import glob
import inspect
import logging
import os
import random
import shutil
import string
import threading
import time
from contextlib import contextmanager

import numpy as np
import torch

from sentalign.document import Side
from sentalign.scoring import get_encoder, get_scorer
from sentalign.scoring.cache import EmbeddingCache
from sentalign.scoring.embedding import legacy_views
from sentalign import anchors as anchoring
from sentalign.scoring.encoders import ModelLoadError
from sentalign.pathfinding import LINK, PathParams, get_pathfinder
from sentalign.pathfinding.segments import PaperLookup, make_segment, steps_to_paper_text
from sentalign.postprocess import reevaluate as post
from sentalign.legacy import align_anchors as legacy
from sentalign.legacy.align_anchors import align_anchors_multi
from sentalign import files as filetools
from sentalign import runrecord

log = logging.getLogger('sentalign')

PROGRESS_INTERVAL = 30  # seconds between "still aligning" lines


# outputs of a file are written under this suffix and renamed when both are complete; the process id keeps two
# instances that work on the same file out of each other's temporary files
NEW_SUFFIX = '.new.%d' % os.getpid()


class RunInfo:
    """What the 2023 ReportInfo kept for the .info text file and, now, for the JSON record.

    ``add_nodes`` is the interface the path search calls.  In the worker processes it updates a copy that is lost
    (a fault of 1.0), so the node counter is not reported.
    """

    def __init__(self):
        self.input_file = ''
        self.stage = ''
        self.source_file_length = 0
        self.target_file_length = 0
        self.total_path_knots = 0
        self.current_path_knot = 1
        self.anchors = []
        self.file_elapsed_time = 0.0
        self.align_elapsed_time = 0.0
        self.gale_church_elapsed_time = 0.0
        self.calc_anchors_elapsed_time = 0.0
        self.calc_elapsed_labse = 0.0
        self.greedy_algorithm_elapsed_time = 0.0

    def set_total_calculations(self, total_calculations):
        self.total_path_knots = total_calculations

    def set_status(self, status):
        self.stage = status
        log.debug('%s: %s', self.input_file, status)

    def add_nodes(self, calculated_nodes):
        self.current_path_knot += calculated_nodes

    def print_info(self):
        """The text of the old .info file (same keys and order as version 1)."""
        return ('File: ' + str(self.input_file) + '\n' + 'File Elapsed Time: ' + str(self.file_elapsed_time) + '\n' +
                'Source file length: ' + str(self.source_file_length) + '\n' +
                'Target file length: ' + str(self.target_file_length) + '\n' + 'Nodes: ' + str(self.total_path_knots) + '\n' +
                'Gale-Church Elapsed Time: ' + str(self.gale_church_elapsed_time) + '\n' + 'Align Elapsed Time: ' + str(self.align_elapsed_time) + '\n' +
                'Calc Anchors Elapsed Time: ' + str(self.calc_anchors_elapsed_time) + '\n' + 'Greedy Algorithm Elapsed Time: ' + str(self.greedy_algorithm_elapsed_time) + '\n' +
                'Anchors: ' + str(self.anchors) + '\n')


class Aligner:
    """Owns the configuration, the model and the per-run state of one SentAlign run."""

    def __init__(self, config, source_folder=None, target_folder=None, output_folder=None, info_folder=None):
        self.cfg = config.resolve_device()      # 'auto' -> cuda or cpu, whichever is there
        corpus = config.corpus_folder
        self.source_language_folder = source_folder or os.path.join(corpus, config.source_language)
        self.target_language_folder = target_folder or os.path.join(corpus, config.target_language)
        self.output_folder = output_folder or os.path.join(corpus, config.output_folder)
        self.align_info_folder = info_folder or config.info_folder or os.path.join(corpus, 'align_info')
        self.tmp_parent = os.path.join(corpus, 'tmp')
        self.temporary_folder = None
        self.scorer = None      # built on first use (loads the encoder model)
        self.info = RunInfo()
        self.timings = {}
        self.input_paths = None  # set by the CLI for one pair of files
        self._warned = set()
        self._prepared = {}      # file name -> sides and vectors of the group being aligned (see _next_group)
        self._group_end = 0      # position in the list of documents where that group ends
        self.sides = None

    @classmethod
    def for_memory(cls, config):
        """An aligner for documents in memory (no corpus folder): see ``sentalign.api``."""
        self = cls.__new__(cls)
        self.cfg = config.resolve_device()
        self.source_language_folder = self.target_language_folder = self.output_folder = self.align_info_folder = None
        self.tmp_parent = self.temporary_folder = None
        self.scorer = None
        self.info = RunInfo()
        self.timings = {}
        self.input_paths = None
        self._warned = set()
        self._prepared = {}
        self._group_end = 0
        self.sides = None
        return self

    # ------------------------------------------------------------------ housekeeping
    def _warn(self, place, message):
        """Log the exception being handled at WARNING level, once per distinct place per file."""
        if place not in self._warned:
            self._warned.add(place)
            log.warning('%s: %s (logged once per file; the run continues as in version 1)', self.info.input_file, message, exc_info=True)

    def _status(self, status):
        self.info.set_status(status)

    @contextmanager
    def _stage(self, name):
        start = time.perf_counter()
        log.debug('%s: %s ...', self.info.input_file, name)
        try:
            yield
        finally:
            dt = time.perf_counter() - start
            self.timings[name] = self.timings.get(name, 0.0) + dt
            log.info('%s: %s done in %.1f s', self.info.input_file, name, dt)

    def _empty_cache(self):
        if str(self.cfg.proc_device).startswith('cuda') and torch.cuda.is_available():
            torch.cuda.empty_cache()

    def ensure_scorer(self):
        """Build the scorer from the configuration (once per run; the model is loaded here)."""
        if self.scorer is None:
            cfg = self.cfg
            scorer_class = get_scorer(cfg.scorer)
            encoder = None
            if getattr(scorer_class, 'needs_encoder', True):
                encoder_class = get_encoder(cfg.encoder_kind)
                half = {}
                if cfg.half:
                    params = inspect.signature(encoder_class).parameters
                    if 'half' in params or any(p.kind is p.VAR_KEYWORD for p in params.values()):
                        half = {'half': True}
                    else:
                        log.warning('encoder %r has no half-precision option; encoding in float32', cfg.encoder_kind)
                encoder = encoder_class(model_name=cfg.model_name, device=cfg.proc_device,
                                        pooling=cfg.pooling, batch_size=cfg.batch_size,
                                        order=cfg.batch_order, max_tokens=cfg.max_tokens,
                                        revision=cfg.model_revision, dedupe=cfg.dedupe_texts, **half,
                                        **({'trust_remote_code': True} if cfg.trust_remote_code else {}))
                encoder.load()
            cache = EmbeddingCache(cfg.embedding_cache) if cfg.embedding_cache else None
            self.scorer = scorer_class(encoder, cache=cache)
        return self.scorer

    @property
    def encoder(self):
        return self.ensure_scorer().encoder

    def _make_temporary_folder(self):
        os.makedirs(self.tmp_parent, exist_ok=True)
        folder = os.path.join(self.tmp_parent, ''.join(random.choices(string.ascii_uppercase, k=8)))
        os.mkdir(folder)
        return folder

    # ------------------------------------------------------------------ running
    def run(self, file_names):
        """Align the given files (already selected and sorted).  Returns (aligned, failed) lists of names."""
        aligned, failed = [], []
        start = datetime.datetime.now()
        self.temporary_folder = self._make_temporary_folder()
        self._prepared, self._group_end = {}, 0
        try:
            for n, name in enumerate(file_names):
                # another instance may have finished this file in the meantime
                if not self.cfg.overwrite and filetools.is_done(self.output_folder, name):
                    log.info('%s: already aligned by another run, skipped', name)
                    continue
                log.info('%s: starting (%d of %d)', name, n + 1, len(file_names))
                try:
                    if n >= self._group_end:
                        self._next_group(file_names, n, self._read_sides)
                    self.process_file(name)
                    aligned.append(name)
                except ModelLoadError as e:
                    log.error('%s: %s', name, e)
                    failed.append(name)
                    failed.extend(file_names[n + 1:])
                    break
                except Exception as e:
                    log.error('%s: FAILED (%s: %s)', name, type(e).__name__, e, exc_info=True)
                    self._remove_partial_output(name)
                    failed.append(name)
                self._prepared.pop(name, None)      # a file that failed before its vectors were used
        finally:
            self._prepared = {}
            shutil.rmtree(self.temporary_folder, ignore_errors=True)
            self.temporary_folder = None
        log.info('Aligned %d files (%d failed) in %.1f seconds.', len(aligned), len(failed),
                 (datetime.datetime.now() - start).total_seconds())
        return aligned, failed

    def _make_sides(self, source, target):
        cfg = self.cfg
        kw = dict(max_k=cfg.max_concatenations, minimum_length_words=cfg.minimum_length_words,
                  maximum_length_words=cfg.maximum_length_words)
        return Side(list(source), lang=cfg.source_language, **kw), Side(list(target), lang=cfg.target_language, **kw)

    def _read_sides(self, name):
        """The two sides of a file for `_next_group`; None for a file that is left to `process_file` (it cannot be
        read or is empty, which is reported there) or that is already aligned."""
        try:
            src = self.read_sentences(self.source_language_folder, name)
            trg = self.read_sentences(self.target_language_folder, name)
        except (OSError, UnicodeDecodeError):
            return None
        if not src or not trg or not (self.cfg.overwrite or not filetools.is_done(self.output_folder, name)):
            return None
        return self._make_sides([src[i] for i in range(len(src))], [trg[i] for i in range(len(trg))])

    def _can_batch(self):
        """Batching is an optional capability of a scorer: `encode_sides` and `use_vectors` (see `Scorer`)."""
        scorer = self.scorer if self.scorer is not None else get_scorer(self.cfg.scorer)
        return all(callable(getattr(scorer, m, None)) for m in ('encode_sides', 'use_vectors'))

    def _next_group(self, names, start, load):
        """Encode the span texts of the next group of small documents together: names[start:], up to `--batch-texts`
        texts.  `load(name)` gives the two sides of a document, or None for one that is left alone.

        The encoder then works on a few full batches instead of many nearly empty ones, and texts shared between
        documents are encoded once.  Only one group is kept at a time (memory: 3 KB per text): the caller asks for
        the next one when its loop reaches `_group_end`, and `align_document` releases the vectors of each document
        as it takes them.  A document above the limit is encoded on its own.  Not used by paper-2023 (batch
        composition changes the last digits of the vectors), nor with a scorer that cannot batch.
        """
        self._prepared = {}
        self._group_end = end = len(names)
        limit = self.cfg.batch_texts
        if not limit or len(names) < 2 or not self._can_batch():
            return
        group, texts = [], 0
        for pos in range(start, len(names)):
            sides = load(names[pos])
            if sides is None:
                continue
            rows = sides[0].n_rows + sides[1].n_rows
            if rows > limit:                        # a big document is encoded on its own
                continue
            if texts + rows > limit:
                end = pos                           # the next group starts here
                break
            group.append((names[pos],) + tuple(sides))
            texts += rows
        self._group_end = end                       # set before encoding: a failure is not repeated for every file
        if len(group) > 1:
            scorer = self.ensure_scorer()
            started = time.perf_counter()
            try:
                vectors = scorer.encode_sides([x for _, ss, ts in group for x in (ss, ts)])
            except ModelLoadError:
                raise
            except Exception as e:
                # the group failed, not its first file: every file is prepared on its own, where a file that really
                # cannot be encoded fails alone
                log.warning('could not encode %d files together (%s: %s); they are encoded one by one',
                            len(group), type(e).__name__, e)
                return
            log.info('encoded %d files (%d texts) together in %.1f s', len(group), texts, time.perf_counter() - started)
            for k, (name, ss, ts) in enumerate(group):
                self._prepared[name] = (ss, ts, vectors[2 * k], vectors[2 * k + 1])

    def _remove_partial_output(self, name):
        """Remove half-written temporary outputs of a failed file (finished outputs are never touched).  Exact names:
        a pattern such as `name.*` would also take the files of `name.2020.txt`.  A `.new` without process id is what
        earlier revisions wrote."""
        base = os.path.join(self.output_folder, name)
        ends = (NEW_SUFFIX, NEW_SUFFIX + '.tmp', '.new', '.new.tmp', '.tmp')
        for p in [base + ext + end for ext in ('.path', '.aligned') for end in ends] + [base + '.json.tmp']:
            try:
                os.remove(p)
            except OSError:
                pass

    # ------------------------------------------------------------------ one file
    def process_file(self, filename):
        file_start = time.perf_counter()
        file_name = filename.split('/')[-1]
        self.info = RunInfo()
        self.info.input_file = file_name
        self.timings = {}
        self._warned = set()
        legacy.reset_warnings()

        try:
            ## Read file dictionaries ##
            with self._stage('read'):
                source_dict = self.read_sentences(self.source_language_folder, file_name)
                target_dict = self.read_sentences(self.target_language_folder, file_name)
            source_len = len(source_dict.keys())
            target_len = len(target_dict.keys())

            self.info.source_file_length = source_len
            self.info.target_file_length = target_len
            log.info('%s: %d source and %d target lines', file_name, source_len, target_len)

            if not (source_len > 0 and target_len > 0):
                raise ValueError('empty input (%d source lines, %d target lines): no output is written' % (source_len, target_len))

            start_align = time.process_time()
            total_path, total_score = self.align_document(source_dict, target_dict, file_name)

            finalpairs = self.get_pairs(total_path, source_dict, target_dict, total_score)

            with self._stage('write'):
                # both outputs first under a temporary name, then the old .aligned is removed and the new files are
                # renamed: a failure before the renames leaves the old pair, one in between leaves a file that is
                # not "done" (no .aligned), never a new .path with old pairs
                self.write_path_to_file(total_path, file_name, total_score, suffix=NEW_SUFFIX)
                n_pairs = self.write_pairs_to_file(finalpairs, file_name, total_score, suffix=NEW_SUFFIX)
                self._commit_outputs(file_name)
            end_align = time.process_time()
            self.info.align_elapsed_time = end_align - start_align
            self.info.file_elapsed_time = time.perf_counter() - file_start

            n_links = len([p for p in total_path.split('\n') if p.strip()])
            self._write_records(file_name, n_links, n_pairs, file_start)
            log.info('%s: done in %.1f s (%d links, %d aligned pairs)', file_name, time.perf_counter() - file_start, n_links, n_pairs)
        finally:
            # the temporary files of this file (the folder itself is removed at the end of the run)
            for f in glob.glob(self.temporary_folder + '/*'):
                try:
                    os.remove(f)
                except OSError:
                    log.warning('could not remove temporary file %s', f)

    # ------------------------------------------------------------------ one document (files or memory)
    def align_document(self, source_dict, target_dict, name=''):
        """Align one document pair given as {position: sentence} dictionaries.  Returns the path and the score column
        as text (the formats of the .path file).  Used for files (`process_file`) and from memory (`sentalign.api`)."""
        cfg = self.cfg
        source_len, target_len = len(source_dict), len(target_dict)
        prepared = self._prepared.pop(name, None) if name else None

        ## spans (version 1: the overlap files) ##
        if prepared is not None:
            src_side, trg_side, src_vec, trg_vec = prepared
        else:
            with self._stage('overlaps'):
                src_side, trg_side = self._make_sides([source_dict[i] for i in range(source_len)],
                                                      [target_dict[i] for i in range(target_len)])

        ## encode the spans ##
        with self._stage('embeddings'):
            scorer = self.ensure_scorer()
            if prepared is not None:
                scorer.use_vectors(src_side, trg_side, src_vec, trg_vec)    # encoded together with other files
            else:
                scorer.prepare(src_side, trg_side)
        self.sides = (src_side, trg_side)

        self._empty_cache()
        with self._stage('anchoring'):
            # anchors as (i, j) "sentences i and j begin a new segment"; the end of the document closes the list
            anchor_list = self.find_anchors(src_side, trg_side) + [(source_len, target_len)]

        self.info.anchors = [list(a) for a in anchor_list]
        log.info('%s: %d segment(s)', name, len(anchor_list))

        ## text-keyed dictionaries for the 2023 post-processing (and the 2023 path search, if selected) ##
        self._status("Reading LaBSE embeddings")
        with self._stage('read-embeddings'):
            if cfg.compat == 'paper-2023':
                src_emb_dict, trg_emb_dict = legacy_views(self.scorer)
            else:
                src_emb_dict = trg_emb_dict = None

        ## Run aligner ##
        self._status("Creating anchor chunks")
        start_anchor = (0, 0)
        matrix_anchors = ()
        for curr_anchor in anchor_list:
            this_anchor = (int(curr_anchor[0]), int(curr_anchor[1]))
            matrix_anchors += (((start_anchor[0] - 1, start_anchor[1] - 1), (this_anchor[0], this_anchor[1])),)
            start_anchor = this_anchor

        self._empty_cache()
        self._status("Aligning...")
        with self._stage('align'):
            if cfg.path_backend == 'legacy':
                return self._align(matrix_anchors, source_dict, target_dict, src_emb_dict, trg_emb_dict)
            return self._align_segments(matrix_anchors, source_dict, target_dict, src_emb_dict, trg_emb_dict)

    def _align_segments(self, matrix_anchors, source_dict, target_dict, src_emb_dict, trg_emb_dict):
        """Path search and post-processing of every segment between two anchors (threads; nothing is copied).

        paper-2023: the new kernel with the arithmetic and the embedding lookups of the 2023 code, followed by the
        2023 post-processing (unchanged code, faults included), so the output is that of version 1.
        Otherwise: the version-2 path search and post-processing.
        """
        cfg = self.cfg
        paper = cfg.compat == 'paper-2023'
        src_side, trg_side = self.sides
        params = PathParams(score_cutoff=cfg.score_cutoff, free_concatenations=cfg.free_concatenations,
                            minimum_length_words=cfg.minimum_length_words, maximum_length_words=cfg.maximum_length_words,
                            penalty_after_words=cfg.penalty_after_words, penalty_per_word=cfg.penalty_per_word,
                            paper_arithmetic=paper, backend=cfg.path_backend, reward=cfg.reward,
                            length_prior=cfg.length_prior)
        workers = max(1, int(cfg.num_proc) if cfg.num_proc else 1)
        # the rust kernel can use several cores inside one segment; share the cores between the segments
        params.threads = max(1, workers // len(matrix_anchors))
        finder_class = get_pathfinder(cfg.pathfinder)
        lookup = PaperLookup(self.scorer) if paper else None
        # the word limits belong to the sentalign-dp search and its post-processing; a path finder without
        # post-processing does not apply them, so its links are scored as they are
        limits = (cfg.minimum_length_words, cfg.maximum_length_words) if finder_class.postprocess else ()
        span_score = None if paper else post.SpanScores(self.scorer, src_side, trg_side, *limits)
        self.info.total_path_knots = sum((a[1][0] - a[0][0] + 1) * (a[1][1] - a[0][1] + 1) for a in matrix_anchors)

        def one(anchor):
            (start_source, start_target), (end_source, end_target) = anchor
            src_first, trg_first = start_source + 1, start_target + 1
            n_src, n_trg = end_source - src_first, end_target - trg_first
            finder = finder_class(params)
            if hasattr(finder, 'configure'):
                finder.configure(cfg.pathfinder_options)
            if paper:
                steps = finder.find(lookup.segment(src_first, n_src, trg_first, n_trg), stop_node=(start_source, start_target))
                text = steps_to_paper_text(steps, src_first, trg_first)
                return legacy.postprocess_path(text, {}, end_source, end_target, source_dict, target_dict,
                                               src_emb_dict, trg_emb_dict, cfg.score_cutoff)
            steps = finder.find(make_segment(self.scorer, src_side, trg_side, src_first, n_src, trg_first, n_trg))
            links = [post.Link(src_first + p, src_first + p + ks, trg_first + q, trg_first + q + kt)
                     for kind, p, ks, q, kt in steps if kind == LINK]
            if finder.postprocess:
                links = post.postprocess(links, span_score, cfg.score_cutoff, src_first, n_src, trg_first, n_trg)
            else:
                # the path finder's result stands; the links only get their scores (NaN = no score: 0)
                for ln in links:
                    v = span_score(ln.s0, ln.s1, ln.t0, ln.t1)
                    ln.score = 0.0 if v != v else v
                post.check(links, src_first, n_src, trg_first, n_trg)
            path, scores = [], []
            for s_ids, t_ids, score in post.rows(links, src_first, n_src, trg_first, n_trg):
                path.append('[%s:%s]\n' % (','.join(map(str, s_ids)), ','.join(map(str, t_ids))))
                scores.append('0\n' if score is None else str(np.float32(score)) + '\n')
            return ''.join(path), ''.join(scores)

        if len(matrix_anchors) == 1 or workers == 1:
            results = [one(a) for a in matrix_anchors]
        else:
            from concurrent.futures import ThreadPoolExecutor
            with ThreadPoolExecutor(max_workers=min(workers, len(matrix_anchors))) as pool:
                results = list(pool.map(one, matrix_anchors))
        return ''.join(r[0] for r in results), ''.join(r[1] for r in results)

    def _align(self, matrix_anchors, source_dict, target_dict, src_emb_dict, trg_emb_dict):
        """The 2023 path search itself (--path-backend legacy): worker processes, slow, bit-exact."""
        cfg = self.cfg
        stop = threading.Event()
        started = time.perf_counter()

        def report():
            # a plain "still running" line from the parent; the node counter of the workers is lost
            while not stop.wait(PROGRESS_INTERVAL):
                log.info('%s: still aligning (%d chunk(s), %.0f s)', self.info.input_file, len(matrix_anchors), time.perf_counter() - started)

        total_calculations = 0
        for anchor_pair in matrix_anchors:
            total_calculations += (anchor_pair[1][0] - anchor_pair[0][0] + 1) * (anchor_pair[1][1] - anchor_pair[0][1] + 1)
        self.info.total_path_knots = total_calculations

        t = threading.Thread(target=report, name='sentalign-progress', daemon=True)
        t.start()
        try:
            return align_anchors_multi(matrix_anchors, source_dict, target_dict, src_emb_dict, trg_emb_dict, cfg.num_proc, cfg.score_cutoff,
                                       cfg.max_concatenations, self.info, cfg.minimum_length_words, cfg.maximum_length_words,
                                       cfg.penalty_after_words, cfg.penalty_per_word, cfg.free_concatenations)
        finally:
            stop.set()
            t.join()

    # ------------------------------------------------------------------ output
    def _write_records(self, file_name, n_links, n_pairs, file_start):
        os.makedirs(self.align_info_folder, exist_ok=True)
        filetools.atomic_write(os.path.join(self.align_info_folder, file_name + '.info'), self.info.print_info())
        record = runrecord.build_record(self.cfg, file_name, self.info, self.timings, time.perf_counter() - file_start,
                                        n_links, n_pairs, self.input_paths,
                                        scoring=self.scorer.describe() if self.scorer is not None else None,
                                        anchoring=getattr(self, 'anchor_report', None))
        runrecord.write_record(os.path.join(self.output_folder, file_name + '.json'), record)

    def get_pairs(self, path, source_dict, target_dict, total_score):
        pairs = []
        path_pairs = path.strip().split('\n')
        score_list = total_score.strip().split('\n')
        ctr = 0
        for p in path_pairs:
            s, t = p.strip().replace('[', '').replace(']', '').split(':')
            s_nums = s.split(',')
            t_nums = t.split(',')
            source_out = ''
            target_out = ''
            is_null = '' in s_nums or '' in t_nums          # a link with an empty side (skipped sentences)
            if '' not in s_nums:
                for s_num in s_nums:
                    source_out += ' ' + source_dict[int(s_num)]
                source_out = source_out.strip().lstrip()
            else:
                source_out = 'NULLALIGN'
            if '' not in t_nums:
                for t_num in t_nums:
                    target_out += ' ' + target_dict[int(t_num)]
                target_out = target_out.strip().lstrip()
            else:
                target_out = 'NULLALIGN'
            pairs.append([source_out, target_out, score_list[ctr], is_null])
            ctr += 1
        return pairs

    def _commit_outputs(self, file_name):
        """Rename the two finished outputs; .aligned last, so the file is done only when both are in place."""
        base = os.path.join(self.output_folder, file_name)
        # An old .aligned must not be left beside a new .path if the second rename fails.  It is removed when the
        # file is aligned again (--overwrite) or is a leftover without a .path.  A complete pair without --overwrite
        # was finished by another instance while this one was working: it is replaced, never removed first.
        if self.cfg.overwrite or not os.path.exists(base + '.path'):
            try:
                os.remove(base + '.aligned')
            except FileNotFoundError:
                pass
        os.replace(base + '.path' + NEW_SUFFIX, base + '.path')
        os.replace(base + '.aligned' + NEW_SUFFIX, base + '.aligned')

    def write_path_to_file(self, path, file_name, total_score, suffix=''):
        path = path.replace(':', ']:[').strip()
        out_path = ''
        pathlist = path.split('\n')
        scorelist = total_score.split('\n')
        ctr = 0
        for i in pathlist:
            out_path += i + ':' + scorelist[ctr] + '\n'
            ctr += 1
        os.makedirs(self.output_folder, exist_ok=True)
        filetools.atomic_write(self.output_folder + '/' + file_name + '.path' + suffix, out_path)

    def write_pairs_to_file(self, pairs, file_name, total_score, suffix=''):
        out = []
        paper = self.cfg.compat == 'paper-2023'
        for p in pairs:
            outstring = p[0] + '\t' + p[1] + '\t' + p[2] + '\n'
            # version 2 drops the rows marked as null by get_pairs; version 1 dropped every row whose text contained
            # the word NULLALIGN, a real sentence too (kept under paper-2023)
            if (outstring.find('NULLALIGN') == -1) if paper else not p[3]:
                out.append(outstring)
        filetools.atomic_write(self.output_folder + '/' + file_name + '.aligned' + suffix, ''.join(out))
        return len(out)

    ##### ALIGNMENT FUNCTIONS #####
    @staticmethod
    def read_sentences(file_folder, file_name):
        sentence_dict = {}
        ctr = 0
        with open(file_folder + '/' + file_name, 'r', encoding='utf-8') as f:
            for sentence in f:
                sentence_dict[ctr] = sentence.strip()
                ctr += 1
        return sentence_dict

    # ------------------------------------------------------------------ anchors
    def find_anchors(self, src_side, trg_side):
        """Anchors [(i, j), ...] without the end of the document.

        ``--anchors auto`` (default): version 2 uses the mutual-best strategy when the document has more nodes than
        the limit (``-anchor``); paper-2023 does what version 1 did for files above ``-maxgc`` sentences (its greedy
        anchors, faults included) and solves smaller files whole.  For those, version 1 tried Gale-Church anchors
        when the graph was above the node limit, but that code could not work and ended in a crash or in solving the
        whole graph; the whole graph is what is done here.
        """
        cfg = self.cfg
        n, m = len(src_side), len(trg_side)
        name = cfg.anchors
        self.anchor_report = {'strategy': name, 'node_limit': cfg.anchoring_delimiter, 'nodes': n * m}
        if name == 'auto':
            if cfg.compat == 'paper-2023':
                if n > cfg.maximum_length_gale_church or m > cfg.maximum_length_gale_church:
                    name = 'legacy-greedy'
                else:
                    if n * m > cfg.anchoring_delimiter:
                        log.info('%s: %d nodes, above the limit of %d: solved whole (version 1 failed here)',
                                 self.info.input_file, n * m, cfg.anchoring_delimiter)
                    name = 'none'
            elif not getattr(get_pathfinder(cfg.pathfinder), 'uses_anchors', True):
                name = 'none'                # the path finder handles documents of any length itself
            else:
                name = 'mutual-best'
            self.anchor_report['strategy'] = name
        if name == 'none' or (n * m <= cfg.anchoring_delimiter and not name.startswith('file:')):
            # no cutting: the whole document is one segment, so the limit for a segment applies to it too (not for
            # path finders that search a corridor and take whole documents; version 1 had no such limit)
            if (cfg.compat != 'paper-2023' and n * m > cfg.max_segment_nodes
                    and getattr(get_pathfinder(cfg.pathfinder), 'uses_anchors', True)):
                raise MemoryError('the document has %d nodes and is solved whole; the limit for one segment is %d '
                                  '(--max-segment-nodes).  Give anchors with --anchors file:PATH, or raise the limit '
                                  'if there is memory for one byte per node.' % (n * m, cfg.max_segment_nodes))
            return []
        if name == 'legacy-greedy':
            try:
                from sentalign.legacy.greedy_anchors import legacy_greedy_anchors
            except ImportError as e:
                raise RuntimeError('--anchors legacy-greedy needs the compiled module sentalign.legacy.greedy, which '
                                   'was not built at installation (no C compiler?): %s' % e) from e
            start = time.process_time()
            found = legacy_greedy_anchors(src_side, trg_side, self.scorer, cfg.anchoring_delimiter,
                                          cfg.minimum_labse_anchor, cfg.labse_subtraction, cfg.cutoff_penalty)
            self.info.greedy_algorithm_elapsed_time = time.process_time() - start
            return found[:-1]                      # version 1 ends the list with the end of the document
        options = dict(min_score=cfg.minimum_labse_anchor, floor_score=cfg.abs_minimum_labse_anchor,
                       margin=cfg.anchor_margin, min_words=cfg.anchor_min_words)
        if name.startswith('file:'):
            strategy = anchoring.get_anchorer('file')(path=name[5:], **options)
        else:
            strategy = anchoring.get_anchorer(name)(**options)
        found = anchoring.validate(strategy.find(src_side, trg_side, self.scorer, cfg.anchoring_delimiter), n, m)
        self.anchor_report.update(strategy.report)
        largest = max((i1 - i0 + 1) * (j1 - j0 + 1) for i0, i1, j0, j1 in anchoring.segments(found, n, m))
        self.anchor_report['largest_segment_nodes'] = largest
        if largest > cfg.max_segment_nodes:
            raise MemoryError('a segment of %d nodes is left after anchoring; the limit for one segment is %d '
                              '(--max-segment-nodes).  Give anchors with --anchors file:PATH, or raise the limit '
                              'if there is memory for one byte per node.' % (largest, cfg.max_segment_nodes))
        return found
