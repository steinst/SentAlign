"""The chain path finder (seed, chain, extend; after ChainAlign)."""
import itertools

import numpy as np
import pytest

from sentalign.pathfinding import LINK, Segment, get_pathfinder
from sentalign.pathfinding import chain_finder as cf


def _unit(a):
    return (a / np.linalg.norm(a, axis=-1, keepdims=True)).astype(np.float32)


def _segment(src_vec, trg_vec, src_sent=None, trg_sent=None, with_vectors=False):
    """Segment over span vectors [k-1] -> (n - k + 1, d); scores are their dot products."""
    n, m = len(src_vec[0]), len(trg_vec[0])

    def score_block(ks, kt, p0, p1, out=None):
        return np.matmul(src_vec[ks - 1][p0:p1], trg_vec[kt - 1].T)

    return Segment(n, m, 4, score_block, None, None, src_sent or ['x' * 20] * n, trg_sent or ['y' * 20] * m,
                   src_vectors=(lambda k: src_vec[k - 1]) if with_vectors else None,
                   trg_vectors=(lambda k: trg_vec[k - 1]) if with_vectors else None)


def _spans(one):
    return [_unit(np.stack([one[p:p + k].sum(0) for p in range(len(one) - k + 1)])) for k in range(1, 5)]


def test_registered_with_the_papers_settings():
    f = get_pathfinder('chain')()
    assert f.options['threshold'] == 0.3 and f.options['band'] == 0.15 and f.options['top_k'] == 10
    assert f.options['skip_cost'] == -0.5 and f.options['max_size'] == 5 and f.uses_anchors is False
    assert cf.link_types(5, 4) == cf.TYPES and len(cf.TYPES) == 10
    assert cf.link_types(3, 4) == [(1, 1), (2, 1), (1, 2)]
    assert f.configure({'length_prior': 'false', 'band': '0'}).options['length_prior'] is False


def test_length_prior():
    assert cf.length_prior(np.array([100.0]), np.array([100.0]))[0] == 1.0
    assert 0 < cf.length_prior(np.array([100.0]), np.array([300.0]))[0] < 1.0
    assert cf.length_prior(np.array([0.0]), np.array([0.0]))[0] == 0.0


def _brute_force(i, k, j, l, w):
    """Best total weight of a non-overlapping, non-crossing subset: quadratic dynamic programme."""
    E = len(w)
    dp = np.array(w, dtype=float)
    for e in range(E):
        for f in range(E):
            if i[f] + k[f] <= i[e] and j[f] + l[f] <= j[e]:
                dp[e] = max(dp[e], w[e] + dp[f])        # f before e in source order, so dp[f] is final
    return dp.max()


@pytest.mark.parametrize('seed', range(8))
def test_chain_is_the_maximum_weight_non_crossing_set(seed):
    rng = np.random.default_rng(seed)
    E, N, M = 300, 40, 45
    k, l = rng.integers(1, 4, E), rng.integers(1, 4, E)
    i, j = rng.integers(0, N - 3, E), rng.integers(0, M - 3, E)
    w = rng.random(E) + 0.01
    by = np.lexsort((j, i))
    i, k, j, l, w = i[by], k[by], j[by], l[by], w[by]
    picked = cf.chain(i, k, j, l, w, M)
    for a, b in zip(picked, picked[1:]):
        assert i[a] + k[a] <= i[b] and j[a] + l[a] <= j[b]            # in order, no overlap, no crossing
    assert abs(w[picked].sum() - _brute_force(i, k, j, l, w)) < 1e-9
    assert cf.chain(i[:0], k[:0], j[:0], l[:0], w[:0], M) == []


def test_gap_alignment_loops_and_diagonals_agree():
    rng = np.random.default_rng(3)
    types = cf.TYPES
    for g, h in [(1, 1), (3, 2), (7, 9), (25, 30)]:
        scores = [rng.normal(0, 0.3, size=(g - k + 1, h - l + 1)) if g >= k and h >= l else None for k, l in types]
        small = cf.align_gap(scores, types, g, h, -0.5)
        # force the diagonal version by embedding the same problem above the size limit of the loops
        big = _diagonal(scores, types, g, h, -0.5)
        assert small == big, (g, h)
        for (a, k, b, l), (a2, k2, b2, l2) in zip(small, small[1:]):
            assert a + k <= a2 and b + l <= b2


def _diagonal(scores, types, g, h, skip):
    """align_gap with the loop version switched off (pad the grid with unreachable cells is not needed: call the
    function on a copy whose size test fails)."""
    import types as pytypes
    code = cf.align_gap.__code__
    consts = tuple(-1 if c == 400 else c for c in code.co_consts)
    fn = pytypes.FunctionType(code.replace(co_consts=consts), cf.align_gap.__globals__)
    return fn(scores, types, g, h, skip)


def test_one_by_one_gap_is_linked_whatever_its_score():
    """With the means taken over the gap, a 1 x 1 gap has score 0, which beats two skips (a property of the method)."""
    d = cf.gap_scores(np.array([[0.05]]), True)
    assert d[0, 0] == 0.0
    assert cf.align_gap([d] + [None] * 9, cf.TYPES, 1, 1, -0.5) == [(0, 1, 0, 1)]
    assert cf.align_gap([cf.gap_scores(np.array([[np.nan]]), True)] + [None] * 9, cf.TYPES, 1, 1, -0.5) == []


@pytest.mark.parametrize('with_vectors', [False, True])
def test_parallel_documents_with_a_merge_and_an_insertion(with_vectors):
    rng = np.random.default_rng(1)
    n, d = 80, 48
    base = _unit(rng.normal(size=(n, d)))
    src = base.copy()
    # target: sentences 10 and 11 of the source are one target sentence; 5 unrelated sentences after sentence 40
    trg = np.concatenate([base[:10], _unit(base[10:11] + base[11:12]), base[12:41], _unit(rng.normal(size=(5, d))), base[41:]])
    seg = _segment(_spans(src), _spans(_unit(trg + 0.02 * rng.normal(size=trg.shape))), with_vectors=with_vectors)
    f = get_pathfinder('chain')()
    steps = f.find(seg)
    assert all(kind == LINK for kind, *_ in steps)
    links = {(s, ks, t, kt) for _, s, ks, t, kt in steps}
    assert (10, 2, 10, 1) in links                                   # the merged pair
    assert (0, 1, 0, 1) in links and (79, 1, 83, 1) in links
    linked_t = {t + x for _, _, t, kt in links for x in range(kt)}
    assert not (linked_t & set(range(40, 45)))                       # the inserted sentences stay unaligned
    assert f.report['chain_links'] > 70
    for (s, ks, t, kt), (s2, _, t2, _) in zip(sorted(links), sorted(links)[1:]):
        assert s + ks <= s2 and t + kt <= t2
