"""evaluation/evaluate.py: strict and lax scores, the two definitions of lax recall, empty inputs."""
import glob
import os
import sys

import pytest

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, 'evaluation'))
import evaluate  # noqa: E402

KEYS = ('precision_strict', 'recall_strict', 'f1_strict', 'precision_lax', 'recall_lax', 'f1_lax')


def _scores(gold, test, **kw):
    res = evaluate.score_multiple([gold], [test], True, **kw)
    return tuple(res[k] for k in KEYS)


def test_a_split_link_is_found_under_the_lax_condition():
    """Gold 2-2, system two 1-1 links: both system links are lax-correct and the gold link is lax-found."""
    gold = [((0, 1), (0, 1))]
    test = [((0,), (0,)), ((1,), (1,))]
    assert _scores(gold, test) == pytest.approx((0, 0, 0, 1, 1, 1))
    # the 2023 script: 2 lax true positives / (2 + 1 gold link without an exact match)
    assert _scores(gold, test, lax_recall='paper-2023') == pytest.approx((0, 0, 0, 1, 2 / 3, 0.8))


def test_a_merged_link_finds_both_gold_links():
    gold = [((0,), (0,)), ((1,), (1,)), ((2,), (2,))]
    test = [((0, 1), (0, 1)), ((2,), (2,))]
    p_s, r_s, _, p_l, r_l, _ = _scores(gold, test)
    assert (p_s, r_s, p_l, r_l) == pytest.approx((0.5, 1 / 3, 1, 1))


def test_lax_needs_overlap_on_both_sides():
    gold = [((0,), (0,)), ((1,), (1,))]
    test = [((0,), (1,)), ((1,), (5,))]             # source overlaps, target of the same gold link does not
    assert _scores(gold, test) == pytest.approx((0, 0, 0, 0, 0, 0))


def test_exact_output_scores_one_in_both_modes():
    gold = [((0,), (0,)), ((1, 2), (1,)), ((3,), ())]
    for mode in evaluate.LAX_RECALL:
        assert _scores(gold, list(gold), lax_recall=mode) == pytest.approx((1,) * 6)


@pytest.mark.parametrize('mode', evaluate.LAX_RECALL)
def test_empty_sides_give_zero_and_no_error(mode):
    link = [((0,), (0,))]
    only_nulls = [((0,), ()), ((), (0,))]           # nothing is left after the null links are removed
    assert _scores(link, only_nulls, lax_recall=mode) == (0,) * 6
    assert _scores(link, [], lax_recall=mode) == (0,) * 6
    assert _scores([], link, lax_recall=mode) == (0,) * 6
    assert _scores([], [], lax_recall=mode) == (0,) * 6


def test_null_links_are_scored_when_asked():
    gold = [((0,), (0,)), ((1,), ())]
    test = [((0,), (0,)), ((1,), ()), ((), (1,))]
    res = evaluate.score_multiple([gold], [test], False)
    assert res['precision_strict'] == pytest.approx(2 / 3) and res['recall_strict'] == 1
    assert res['precision_lax'] == pytest.approx(2 / 3) and res['recall_lax'] == 1


def test_unknown_lax_recall_is_refused():
    with pytest.raises(ValueError):
        evaluate.score_multiple([[]], [[]], True, lax_recall='other')


# (strict P, R, F1, lax P, R, F1) of the SentAlign 1.0 output; paper-2023: the README of version 1 and the paper
EXPECTED = {
    ('bleualign', 'paper-2023'): (0.935, 0.929, 0.932, 0.998, 0.933, 0.964),
    ('bleualign', 'standard'): (0.935, 0.929, 0.932, 0.998, 0.980, 0.989),
    ('parice', 'paper-2023'): (0.948, 0.953, 0.951, 0.985, 0.955, 0.970),
    ('parice', 'standard'): (0.948, 0.953, 0.951, 0.985, 0.988, 0.986),
}


@pytest.mark.parametrize('name,mode', sorted(EXPECTED))
def test_numbers_on_the_evaluation_sets(name, mode):
    gold = sorted(glob.glob(os.path.join(ROOT, 'eval_data', name, 'gold', '*')))
    pairs = [(g, os.path.join(ROOT, 'tests', 'golden', 'v1_gpu', name, os.path.basename(g) + '.path')) for g in gold]
    pairs = [(g, t) for g, t in pairs if os.path.exists(t)]
    assert pairs
    res = evaluate.score_multiple([evaluate.read_alignments(g) for g, _ in pairs],
                                  [evaluate.read_alignments(t) for _, t in pairs], True, lax_recall=mode)
    assert tuple(res[k] for k in KEYS) == pytest.approx(EXPECTED[name, mode], abs=5e-4)
