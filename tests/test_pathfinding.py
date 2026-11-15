"""The path-search kernels and the version-2 post-processing.

* paper-2023: both kernels give the path the 2023 code found for every saved segment (tests/fixtures).
* the compiled kernel and the Python reference give identical back-pointers and sums, in both arithmetic modes,
  on random segments (banded and unbanded).
* version 2: every sentence exactly once, links in order, consecutive sentences only; the defects of the 2023
  post-processing do not occur.
"""
import glob
import json
import os

import numpy as np
import pytest

from sentalign.document import Side
from sentalign.pathfinding import LINK, PathParams, Segment, get_pathfinder
from sentalign.pathfinding.base import span_words
from sentalign.pathfinding.segments import PaperLookup, make_segment, steps_to_paper_text
from sentalign.pathfinding.sentalign_dp import SentAlignDP, get_kernel_class
from sentalign.postprocess import reevaluate as post
from sentalign.scoring.embedding import EmbeddingScorer

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SETS = {'bleualign': ('deu', 'fra'), 'parice': ('eng', 'isl')}

try:
    get_kernel_class('cython')
    BACKENDS = ['cython', 'python']
except ImportError:                      # not built (source checkout without installation)
    BACKENDS = ['python']
try:
    get_kernel_class('rust')
    BACKENDS.append('rust')
except ImportError:                      # the optional package sentalign-rust is not installed
    pass


def _fixture_root(name):
    for sub in ('eval', 'mini'):
        root = os.path.join(ROOT, 'tests', 'fixtures', sub, name)
        if glob.glob(os.path.join(root, 'stages', '*.json')):
            return root
    return None


def _saved_scorer(name, fn, root):
    sl, tl = SETS[name]
    sc = EmbeddingScorer(encoder=None)
    sc.src = Side.from_file(os.path.join(ROOT, 'eval_data', name, sl, fn))
    sc.trg = Side.from_file(os.path.join(ROOT, 'eval_data', name, tl, fn))
    sc.src_emb = np.load(os.path.join(root, 'embeddings', fn + '.src.labse_emb'), allow_pickle=True)
    sc.trg_emb = np.load(os.path.join(root, 'embeddings', fn + '.trg.labse_emb'), allow_pickle=True)
    return sc


def _saved_cases():
    for name in sorted(SETS):
        root = _fixture_root(name)
        if root:
            for sf in sorted(glob.glob(os.path.join(root, 'stages', '*.json'))):
                yield name, os.path.basename(sf).split('__')[0], root, sf


CASES = list(_saved_cases())


# ---------------------------------------------------------------------------- paper-2023
@pytest.mark.skipif(not CASES, reason='no fixtures')
@pytest.mark.parametrize('backend', BACKENDS)
def test_kernel_finds_the_2023_path(backend):
    for name, fn, root, sf in CASES:
        with open(sf, encoding='utf-8') as f:
            st = json.load(f)
        (ss, st_), (es, et) = st['anchor']
        p = st['params']
        seg = PaperLookup(_saved_scorer(name, fn, root)).segment(ss + 1, es - ss - 1, st_ + 1, et - st_ - 1)
        finder = SentAlignDP(PathParams(score_cutoff=p['score_cutoff'], free_concatenations=p['free_concats'],
                                        minimum_length_words=p['minimum_length_words'],
                                        maximum_length_words=p['maximum_length_words'],
                                        penalty_after_words=p['start_penalty_word_number'],
                                        penalty_per_word=p['penalty_per_word'], paper_arithmetic=True, backend=backend))
        steps = finder.find(seg, stop_node=(ss, st_))
        assert steps_to_paper_text(steps, ss + 1, st_ + 1) == st['dp_path'], (fn, backend)
        assert finder.last_score == pytest.approx(st['dp_total'], rel=1e-4), (fn, backend)


# ---------------------------------------------------------------------------- kernels agree
def _random_segment(rng, n_src, n_trg, K=4, nan_share=0.02, long_words=False):
    blocks = {}
    for ks in range(1, K + 1):
        for kt in range(1, K + 1):
            if ks <= n_src and kt <= n_trg:
                b = rng.uniform(0.1, 1.0, (n_src - ks + 1, n_trg - kt + 1)).astype(np.float32)
                b[rng.random(b.shape) < nan_share] = np.nan
                # a diagonal of good 1-1 scores so that links are on the path
                if ks == 1 and kt == 1:
                    for i in range(min(n_src, n_trg)):
                        b[i, i] = np.float32(0.6 + 0.4 * rng.random())
                blocks[ks, kt] = b
    hi = 60 if long_words else 12
    return Segment(n_src, n_trg, K, lambda ks, kt, p0, p1: blocks[ks, kt][p0:p1],
                   span_words(rng.integers(1, hi, n_src), K), span_words(rng.integers(1, hi, n_trg), K))


@pytest.mark.skipif(len(BACKENDS) < 2, reason='compiled kernel not built')
@pytest.mark.parametrize('paper', [True, False])
def test_compiled_kernel_equals_python_reference(paper):
    rng = np.random.default_rng(7)
    for trial in range(12):
        n_src, n_trg = int(rng.integers(1, 45)), int(rng.integers(1, 45))
        seg = _random_segment(rng, n_src, n_trg, long_words=trial % 3 == 0)
        for cutoff in (0.4, 0.0 if trial % 4 == 0 else 0.55):
            res = {}
            for backend, band in (('python', 1 << 30), ('cython', 1 << 30), ('cython', 4 * 16 * (n_trg + 1) * 3)):
                f = SentAlignDP(PathParams(score_cutoff=cutoff, paper_arithmetic=paper, backend=backend, band_bytes=band,
                                           penalty_after_words=80 if trial % 3 else 20))
                bp = f.back_pointers(seg)
                res[backend, band] = (bp.copy(), f.last_score)
            ref = res['python', 1 << 30]
            for key, (bp, total) in res.items():
                assert np.array_equal(bp, ref[0]), (trial, cutoff, key)
                assert total == ref[1], (trial, cutoff, key)      # bit-identical sums


@pytest.mark.skipif('rust' not in BACKENDS, reason='sentalign-rust not installed')
@pytest.mark.parametrize('paper', [True, False])
def test_rust_kernel_equals_python_reference_on_one_and_several_cores(paper):
    """Same back-pointers and bit-identical sums, serial and with tiles in parallel, banded and unbanded."""
    rng = np.random.default_rng(11)
    for trial in range(10):
        n_src, n_trg = int(rng.integers(1, 300)), int(rng.integers(1, 300))
        seg = _random_segment(rng, n_src, n_trg, long_words=trial % 3 == 0)
        cutoff = 0.0 if trial == 4 else 0.4
        ref = SentAlignDP(PathParams(score_cutoff=cutoff, paper_arithmetic=paper, backend='python'))
        ref_bp = ref.back_pointers(seg).copy()
        for threads in (1, 3, 8):
            for band in (1 << 30, 4 * 16 * (n_trg + 1) * 7):
                f = SentAlignDP(PathParams(score_cutoff=cutoff, paper_arithmetic=paper, backend='rust',
                                           threads=threads, band_bytes=band))
                assert np.array_equal(f.back_pointers(seg), ref_bp), (trial, threads, band)
                assert f.last_score == ref.last_score, (trial, threads, band)


@pytest.mark.parametrize('backend', BACKENDS)
def test_path_covers_every_sentence_once(backend):
    rng = np.random.default_rng(3)
    for n_src, n_trg in ((1, 1), (1, 7), (9, 1), (30, 26), (0, 5), (5, 0)):
        seg = _random_segment(rng, n_src, n_trg)
        steps = SentAlignDP(PathParams(backend=backend)).find(seg)
        s = t = 0
        for kind, p, ks, q, kt in steps:
            assert (p, q) == (s, t)
            s, t = s + ks, t + kt
        assert (s, t) == (n_src, n_trg)


def test_registry():
    assert get_pathfinder('sentalign-dp') is SentAlignDP
    with pytest.raises(KeyError):
        get_pathfinder('no-such-search')


# ---------------------------------------------------------------------------- version-2 post-processing
class TableScore:
    """Scores from a table {(s0, s1, t0, t1): value}; everything else scores `default`."""

    def __init__(self, table, default=0.1):
        self.table, self.default = table, default

    def __call__(self, s0, s1, t0, t1):
        return self.table.get((s0, s1, t0, t1), self.default)


def test_reevaluation_splits_a_merged_link_into_its_best_pairs():
    score = TableScore({(0, 1, 0, 1): 0.9, (1, 2, 1, 2): 0.8, (0, 2, 0, 2): 0.7})
    links = post.postprocess([post.Link(0, 2, 0, 2)], score, 0.4, 0, 2, 0, 2)
    assert [(l.s0, l.s1, l.t0, l.t1) for l in links] == [(0, 1, 0, 1), (1, 2, 1, 2)]


def test_rounding_noise_does_not_merge_two_perfect_pairs():
    """Seen in a book: two identical sentence pairs, each 0.99999994, against the merged 2-2 link at 1.0."""
    score = TableScore({(0, 1, 0, 1): 0.99999994, (1, 2, 1, 2): 0.99999994, (0, 2, 0, 2): 1.0})
    links = post.postprocess([post.Link(0, 2, 0, 2)], score, 0.4, 0, 2, 0, 2)
    assert [(l.s0, l.s1, l.t0, l.t1) for l in links] == [(0, 1, 0, 1), (1, 2, 1, 2)]


def test_a_free_sentence_is_absorbed_once_by_the_neighbour_that_gains_most():
    """2023 fault: target sentence 1 was added to the link before it and to the link after it."""
    score = TableScore({(0, 1, 0, 1): 0.6, (1, 2, 2, 3): 0.6, (0, 1, 0, 2): 0.7, (1, 2, 1, 3): 0.9})
    links = post.postprocess([post.Link(0, 1, 0, 1), post.Link(1, 2, 2, 3)], score, 0.4, 0, 2, 0, 3)
    assert [(l.s0, l.s1, l.t0, l.t1) for l in links] == [(0, 1, 0, 1), (1, 2, 1, 3)]
    assert links[1].score == pytest.approx(0.9)


def test_a_linked_sentence_is_not_treated_as_free():
    """2023 fault: after [0,1]:[0,1] was split into two links, sentence 1 was listed as unaligned and the next
    link took it as well."""
    score = TableScore({(0, 1, 0, 1): 0.9, (1, 2, 1, 2): 0.85, (2, 3, 2, 3): 0.6, (1, 3, 1, 3): 0.95})
    links = post.postprocess([post.Link(0, 2, 0, 2), post.Link(2, 3, 2, 3)], score, 0.4, 0, 3, 0, 3)
    assert [(l.s0, l.s1, l.t0, l.t1) for l in links] == [(0, 1, 0, 1), (1, 2, 1, 2), (2, 3, 2, 3)]


def test_rows_lay_out_unaligned_sentences_before_each_link():
    r = post.rows([post.Link(1, 2, 2, 3, 0.5)], 0, 3, 0, 4)
    assert r == [((0,), (), None), ((), (0,), None), ((), (1,), None), ((1,), (2,), 0.5), ((2,), (), None), ((), (3,), None)]


def test_check_rejects_overlapping_links():
    with pytest.raises(AssertionError):
        post.check([post.Link(0, 2, 0, 1), post.Link(1, 3, 1, 2)], 0, 3, 0, 2)


@pytest.mark.skipif(not CASES, reason='no fixtures')
def test_version2_on_the_saved_gold_files_aligns_no_sentence_twice():
    seen = set()
    for name, fn, root, sf in CASES:
        if (name, fn) in seen:
            continue
        seen.add((name, fn))
        sc = _saved_scorer(name, fn, root)
        n_src, n_trg = len(sc.src), len(sc.trg)
        steps = SentAlignDP(PathParams()).find(make_segment(sc, sc.src, sc.trg, 0, n_src, 0, n_trg))
        links = [post.Link(p, p + ks, q, q + kt) for kind, p, ks, q, kt in steps if kind == LINK]
        links = post.postprocess(links, post.SpanScores(sc, sc.src, sc.trg), 0.4, 0, n_src, 0, n_trg)
        rows = post.rows(links, 0, n_src, 0, n_trg)
        assert sorted(i for s, t, _ in rows for i in s) == list(range(n_src)), fn
        assert sorted(j for s, t, _ in rows for j in t) == list(range(n_trg)), fn
        assert len(links) > 0.5 * min(n_src, n_trg), fn


def test_blank_lines_are_never_linked_in_version2():
    src = Side(['a b c', '', 'd e'], max_k=2)
    trg = Side(['x y', 'z', ''], max_k=2)
    sc = EmbeddingScorer(encoder=None)
    sc.src, sc.trg = src, trg
    rng = np.random.default_rng(0)
    sc.src_emb = rng.normal(size=(src.n_rows, 8)).astype(np.float32)
    sc.trg_emb = rng.normal(size=(trg.n_rows, 8)).astype(np.float32)
    seg = make_segment(sc, src, trg, 0, 3, 0, 3)
    assert np.isnan(seg.score_block(1, 1, 0, 3)[1]).all() and np.isnan(seg.score_block(1, 1, 0, 3)[:, 2]).all()
    assert np.isnan(seg.score_block(2, 1, 0, 2)).all()            # both 2-sentence source spans contain the blank
    assert np.isnan(post.SpanScores(sc, src, trg)(1, 2, 0, 1))


# ---------------------------------------------------------------------------- whole pipeline, version 2
@pytest.mark.slow
@pytest.mark.gpu_or_cpu
def test_version2_end_to_end_on_parice(tmp_path, device, labse_available, repo_root):
    """--compat none through the command line: every sentence exactly once, accuracy as the paper's."""
    import re
    import shutil
    import subprocess
    import sys
    from conftest import run_cli
    corpus = tmp_path / 'parice'
    for lang in ('eng', 'isl'):
        shutil.copytree(os.path.join(repo_root, 'eval_data', 'parice', lang), corpus / lang)
    r = run_cli(['-dir', corpus, '-sl', 'eng', '-tl', 'isl', '-device', device, '--compat', 'none', '-proc', '2'])
    assert r.returncode == 0, r.stderr[-3000:]
    paths = sorted(glob.glob(str(corpus / 'output' / '*.path')))
    assert len(paths) == 10
    for f in paths:
        fn = os.path.basename(f)[:-5]
        seen_s, seen_t = [], []
        for line in open(f, encoding='utf-8'):
            m = re.match(r'\[(.*?)\]:\[(.*?)\]:', line)
            seen_s += [int(x) for x in m.group(1).split(',') if x]
            seen_t += [int(x) for x in m.group(2).split(',') if x]
        assert seen_s == list(range(sum(1 for _ in open(corpus / 'eng' / fn, encoding='utf-8')))), fn
        assert seen_t == list(range(sum(1 for _ in open(corpus / 'isl' / fn, encoding='utf-8')))), fn
    gold = sorted(glob.glob(os.path.join(repo_root, 'eval_data', 'parice', 'gold', '*.txt')))
    out = subprocess.run([sys.executable, os.path.join(repo_root, 'evaluation', 'evaluate.py'), '-t'] + paths + ['-g'] + gold,
                         capture_output=True, text=True)
    f1 = re.search(r'F1\s*\|\s*([\d.]+)\s*\|\s*([\d.]+)', out.stdout + out.stderr)
    assert f1 and float(f1.group(1)) >= 0.945 and float(f1.group(2)) >= 0.965, out.stdout + out.stderr
