"""Anchoring.

Synthetic documents with a known alignment and controlled similarity scores check that the strategies return only
sure anchors, that contradicting candidates fall out, that a segment without a sure anchor is left whole, and that
the faults of the 2023 anchoring are gone from the version-2 path.
"""
import numpy as np
import pytest

from sentalign import anchors as anchoring
from sentalign.anchors.base import AnchorError, choose, longest_chain, nodes, segments, supported, validate
from sentalign.anchors.strategies import FromFile, GaleChurchConfirmed, MutualBest, NoAnchors
from sentalign.config import AlignConfig
from sentalign.document import Side
from sentalign.scoring.base import Scorer


class MatrixScorer(Scorer):
    """1-1 scores from a matrix (spans of more than one sentence are not needed by the anchor strategies)."""

    def __init__(self, matrix):
        self.m = np.asarray(matrix, dtype=np.float32)

    def block(self, ks, kt, i0, i1, j0, j1, out=None):
        assert ks == kt == 1
        return self.m[i0:i1, j0:j1]


def _doc(n, words=6):
    return Side(['w%d ' % i + ' '.join('x' * 3 for _ in range(words - 1)) for i in range(n)], max_k=1)


def _diagonal(n, m, on=0.97, off=0.3, seed=0):
    """Target sentence j translates source sentence round(j * n / m); everything else scores low."""
    rng = np.random.default_rng(seed)
    mat = rng.uniform(0.05, off, (n, m)).astype(np.float32)
    pairs = []
    for i in range(min(n, m)):
        mat[i, i] = on
        pairs.append((i, i))
    return mat, pairs


# ---------------------------------------------------------------------------- helpers
def test_validate_and_segments():
    assert validate([(3, 4), (7, 9)], 10, 12) == [(3, 4), (7, 9)]
    assert segments([(3, 4), (7, 9)], 10, 12) == [(0, 3, 0, 4), (3, 7, 4, 9), (7, 10, 9, 12)]
    for bad in ([(3, 4), (3, 9)], [(5, 4), (4, 9)], [(0, 3)], [(3, 12)], [(10, 5)]):
        with pytest.raises(AnchorError):
            validate(bad, 10, 12)


def test_longest_chain_drops_contradicting_candidates():
    cands = [(1, 1, .9), (2, 2, .9), (3, 40, .99), (4, 4, .9), (5, 5, .9), (30, 6, .99), (31, 31, .9)]
    # (3, 40) contradicts the others; (30, 6) is wrong too, but it is in order with them
    assert [c[:2] for c in longest_chain(cands)] == [(1, 1), (2, 2), (4, 4), (5, 5), (30, 6), (31, 31)]
    assert longest_chain([]) == []


def test_supported_drops_isolated_candidates():
    chain = [(i, i, .9) for i in range(1, 6)] + [(30, 6, .99)] + [(i, i, .9) for i in range(31, 36)] + [(90, 80, .99)]
    kept = supported(chain, window=20, support=2)
    assert (30, 6, .99) not in kept and (90, 80, .99) not in kept and len(kept) == 10
    # a real jump (a passage missing on one side) keeps its candidates: they have neighbours after the jump
    jump = [(i, i, .9) for i in range(1, 6)] + [(500 + i, 6 + i, .9) for i in range(5)]
    assert supported(jump, window=20, support=2) == jump


def test_choose_uses_few_anchors_and_prefers_confident_ones():
    chain = [(i, i, 0.96 if i % 10 == 0 else 0.85) for i in range(1, 100)]
    anchors, left = choose(chain, 100, 100, 3000, [0.95, 0.9, 0.85, 0.8])
    assert not left and all(a[0] % 10 == 0 for a in anchors)                    # only the 0.96 candidates were needed
    assert all(nodes(*s) <= 3000 for s in segments(anchors, 100, 100))
    assert len(anchors) <= 7
    # no candidate inside a large segment: it is reported, not split
    anchors, left = choose([(2, 2, 0.99)], 100, 100, 3000, [0.95, 0.8])
    assert anchors == [(2, 2)] and left == [(2, 100, 2, 100)]


# ---------------------------------------------------------------------------- mutual best
def test_mutual_best_finds_anchors_on_the_true_alignment():
    mat, pairs = _diagonal(120, 120)
    strat = MutualBest()
    found = strat.find(_doc(120), _doc(120), MatrixScorer(mat), max_nodes=2500)
    assert found and all(i == j for i, j in found)
    assert all(nodes(*s) <= 2500 for s in segments(found, 120, 120))
    assert strat.report['contradicted'] == 0 and strat.report['segments_left_too_large'] == []


def test_mutual_best_computes_in_bands():
    mat, _ = _diagonal(90, 70, seed=3)
    a, b = MutualBest(), MutualBest()
    b.chunk_rows = 7
    da, db = a.best_two(_doc(90), _doc(70), MatrixScorer(mat)), b.best_two(_doc(90), _doc(70), MatrixScorer(mat))
    for x, y in zip(da, db):
        assert np.array_equal(x, y)
    assert np.array_equal(da[0], mat.max(1)) and np.array_equal(da[3], mat.max(0))
    assert np.allclose(da[1], np.sort(mat, 1)[:, -2]) and np.allclose(da[4], np.sort(mat, 0)[-2])


def test_a_repeated_sentence_is_no_anchor():
    """The 2023 fault behind the skipped chapters: a short or repeated sentence ("Yes.") scores 0.95+ against the
    wrong partner.  Two equally good partners -> no margin -> not a candidate."""
    mat, _ = _diagonal(60, 60)
    mat[10, 50] = 0.99          # sentence 10 also matches target 50 perfectly, e.g. the same sentence again
    mat[10, 10] = 0.99
    cands = MutualBest().candidates(_doc(60), _doc(60), MatrixScorer(mat))
    assert all(c[0] != 10 for c in cands)


def test_short_sentences_and_blank_lines_are_no_anchors():
    mat, _ = _diagonal(40, 40)
    src, trg = _doc(40), _doc(40)
    src.word_counts[5] = 2
    trg.word_counts[7] = 1
    src.sentences[9] = ''
    cands = MutualBest().candidates(src, trg, MatrixScorer(mat))
    got = {c[0] for c in cands}
    assert 5 not in got and 7 not in got and 9 not in got and len(got) == 37


def test_an_isolated_false_match_is_contradicted_by_the_rest():
    mat, _ = _diagonal(80, 80, on=0.9)
    mat[20, 70] = 0.99          # a unique, confident, wrong match far off the diagonal
    mat[20, 20] = 0.3
    mat[70, 70] = 0.3
    strat = MutualBest()
    found = strat.find(_doc(80), _doc(80), MatrixScorer(mat), max_nodes=800)
    assert (20, 70) not in found and all(i == j for i, j in found)
    assert strat.report['contradicted'] + strat.report['isolated'] >= 1


def test_a_false_match_that_is_in_order_is_still_not_used():
    mat, _ = _diagonal(80, 80, on=0.9)
    for k in range(6, 30):               # sentences 6..29 on both sides have no good match
        mat[k, k] = 0.3
    mat[28, 7] = 0.99                    # unique, confident, in order with 5-5 and 30-30, and wrong
    strat = MutualBest()
    found = strat.find(_doc(80), _doc(80), MatrixScorer(mat), max_nodes=500)
    assert (28, 7) not in found and all(i == j for i, j in found) and strat.report['isolated'] >= 1


def test_no_sure_anchor_means_no_anchor():
    """Scores below the floor everywhere: the document is left whole (and the caller is told)."""
    mat, _ = _diagonal(50, 50, on=0.7)
    strat = MutualBest()
    assert strat.find(_doc(50), _doc(50), MatrixScorer(mat), max_nodes=400) == []
    assert strat.report['segments_left_too_large'] == [[0, 50, 0, 50]]


def test_abridged_translation_anchors_follow_the_real_position():
    """The target has only every second source sentence: anchors lie on i = 2j, far from the i = j diagonal that the
    2023 band assumed."""
    n, m = 200, 100
    rng = np.random.default_rng(1)
    mat = rng.uniform(0.05, 0.3, (n, m)).astype(np.float32)
    for j in range(m):
        mat[2 * j, j] = 0.97
    found = MutualBest().find(_doc(n), _doc(m), MatrixScorer(mat), max_nodes=2000)
    assert found and all(i == 2 * j for i, j in found)


# ---------------------------------------------------------------------------- other strategies
def test_gale_church_confirmed_uses_only_confirmed_one_to_one_beads():
    src = Side(['Sentence number %d has some words in it, %s.' % (i, 'x' * (5 + 3 * (i % 7))) for i in range(60)], max_k=1)
    trg = Side(['Satz Nummer %d hat einige Woerter, %s.' % (i, 'y' * (5 + 3 * (i % 7))) for i in range(60)], max_k=1)
    mat, _ = _diagonal(60, 60)
    mat[30, 30] = 0.5                        # Gale-Church links 30-30, the scorer does not confirm it
    strat = GaleChurchConfirmed()
    cands = strat.candidates(src, trg, MatrixScorer(mat))
    assert cands and all(i == j for i, j, _ in cands) and all(i != 30 for i, j, _ in cands)
    found = strat.find(src, trg, MatrixScorer(mat), max_nodes=900)
    assert found and all(nodes(*s) <= 900 for s in segments(found, 60, 60))


def test_anchors_from_a_file(tmp_path):
    f = tmp_path / 'cuts.tsv'
    f.write_text('# chapter starts\n10\t12\n0\t0\n30,33\n', encoding='utf-8')
    assert FromFile(path=str(f)).find(_doc(40), _doc(45), None, 10) == [(10, 12), (30, 33)]
    f.write_text('10\t12\n9\t20\n', encoding='utf-8')
    with pytest.raises(AnchorError):
        FromFile(path=str(f)).find(_doc(40), _doc(45), None, 10)


def test_registry_and_presets():
    assert anchoring.get_anchorer('mutual-best') is MutualBest and anchoring.get_anchorer('none') is NoAnchors
    assert NoAnchors().find(_doc(5), _doc(5), None, 1) == []
    with pytest.raises(KeyError):
        anchoring.get_anchorer('no-such-strategy')
    assert AlignConfig(compat='paper-2023').anchoring_delimiter == 4000000
    assert AlignConfig(compat='none').anchoring_delimiter == 100000000
    assert AlignConfig(compat='none', anchoring_delimiter=5).anchoring_delimiter == 5


def test_the_faulty_version1_modules_are_gone():
    """utilities.pyx (TypeError on every call, out-of-bounds read) and anchoring.pyx were only used by
    the mid-size anchoring of version 1, which could not work; they are not part of version 2."""
    import importlib
    for name in ('sentalign.utilities', 'sentalign.anchoring', 'sentalign.galechurch'):
        with pytest.raises(ImportError):
            importlib.import_module(name)
