"""--reward squared-cosine: the link weight of ChainAlign in the SentAlign path search."""
import numpy as np
import pytest

from sentalign.config import AlignConfig
from sentalign.pathfinding import LINK, PathParams, Segment, get_pathfinder
from sentalign.pathfinding import chain_finder as cf
from sentalign.pathfinding.base import span_words


def _segment(n, m, seed, sentences=False):
    rng = np.random.default_rng(seed)
    blocks = {(ks, kt): rng.uniform(0.0, 1.0, size=(n - ks + 1, m - kt + 1)).astype(np.float32)
              for ks in range(1, 5) for kt in range(1, 5)}
    blocks[(1, 1)][2, :] = np.nan                                   # a blank line

    def score_block(ks, kt, p0, p1, out=None):
        return blocks[(ks, kt)][p0:p1].copy()

    words = np.full(max(n, m), 5)
    seg = Segment(n, m, 4, score_block, span_words(words[:n], 4), span_words(words[:m], 4),
                  ['x' * int(k) for k in rng.integers(5, 80, n)] if sentences else None,
                  ['y' * int(k) for k in rng.integers(5, 80, m)] if sentences else None)
    return seg, blocks


def _reward(steps, blocks):
    return sum(float(blocks[(ks, kt)][p, q]) ** 2 for kind, p, ks, q, kt in steps if kind == LINK)


@pytest.mark.parametrize('seed', range(4))
def test_squared_cosine_path_is_optimal_and_the_kernels_agree(seed):
    seg, blocks = _segment(23, 19, seed)
    params = PathParams(score_cutoff=0.3, reward='squared-cosine', backend='python')
    steps = get_pathfinder('sentalign-dp')(params).find(seg)
    for kind, p, ks, q, kt in steps:
        if kind == LINK:
            assert blocks[(ks, kt)][p, q] >= 0.3                    # no link below the cutoff, none on the blank line
    # the best total of score ** 2 over all monotone paths, by an independent search
    types = [(k, l) for k in range(1, 5) for l in range(1, 5)]
    scores = [np.where(np.nan_to_num(blocks[t], nan=-1.0) >= 0.3, blocks[t].astype(np.float64) ** 2, -np.inf) for t in types]
    best = sum(float(blocks[(k, l)][a, b]) ** 2 for a, k, b, l in cf.align_gap(scores, types, 23, 19, 0.0))
    assert abs(_reward(steps, blocks) - best) < 1e-4
    try:
        import sentalign.pathfinding._dp_cython  # noqa: F401
    except ImportError:
        return
    assert get_pathfinder('sentalign-dp')(PathParams(score_cutoff=0.3, reward='squared-cosine', backend='cython')).find(seg) == steps


def test_length_prior_changes_the_path_and_needs_the_texts():
    seg, _ = _segment(23, 19, 1, sentences=True)
    plain = get_pathfinder('sentalign-dp')(PathParams(score_cutoff=0.3, reward='squared-cosine', backend='python')).find(seg)
    prior = get_pathfinder('sentalign-dp')(PathParams(score_cutoff=0.3, reward='squared-cosine', length_prior=True,
                                                       backend='python')).find(seg)
    assert plain != prior
    with pytest.raises(ValueError, match='sentence texts'):
        get_pathfinder('sentalign-dp')(PathParams(reward='squared-cosine', length_prior=True, backend='python')).find(_segment(9, 9, 0)[0])


def test_reward_in_the_configuration():
    assert AlignConfig().reward == 'sentalign' and AlignConfig().score_cutoff == 0.4
    assert AlignConfig(reward='squared-cosine').score_cutoff == 0.3
    assert AlignConfig(reward='squared-cosine', score_cutoff=0.5).score_cutoff == 0.5
    with pytest.raises(ValueError, match='compat none'):
        AlignConfig(reward='squared-cosine', compat='paper-2023')
    with pytest.raises(ValueError, match='unknown --reward'):
        AlignConfig(reward='cubic')
    with pytest.raises(ValueError, match='length-prior'):
        AlignConfig(length_prior=True)
    import argparse
    from sentalign.config import add_alignment_arguments, config_from_args
    p = argparse.ArgumentParser(); add_alignment_arguments(p)
    assert config_from_args(p.parse_args([])).score_cutoff == 0.4
    c = config_from_args(p.parse_args(['--reward', 'squared-cosine', '--length-prior']))
    assert c.score_cutoff == 0.3 and c.length_prior
