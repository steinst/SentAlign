"""Short segments, back-pointer range, word limits in post-processing, a sentence that contains the word NULLALIGN."""
import math
import types

import numpy as np
import pytest

from sentalign.aligner import Aligner
from sentalign.config import AlignConfig
from sentalign.document import Side
from sentalign.pathfinding.segments import _blank_spans
from sentalign.pathfinding._dp_python import DPKernelPython
from sentalign.postprocess.reevaluate import SpanScores, absorb

from test_api import _fake_aligner


# ---------------------------------------------------------------------------- short segments
def test_blank_spans_of_a_short_segment():
    side = Side(['a', 'b'], max_k=4)
    assert [len(_blank_spans(side, 0, 2, k)) for k in (1, 2, 3, 4)] == [2, 1, 0, 0]
    assert _blank_spans(side, 0, 2, 4).dtype == bool


@pytest.mark.parametrize('pathfinder', ['sentalign-dp', 'chain', 'vecalign', 'galechurch'])
@pytest.mark.parametrize('n', [1, 2, 3])
def test_documents_of_one_to_three_sentences(pathfinder, n):
    src = ['The cat sat on mat number %d.' % i for i in range(n)]
    trg = ['Kottur sat a mottu nr %d.' % i for i in range(n)]
    res = _fake_aligner(pathfinder=pathfinder).align(src, trg)
    assert sorted(i for ln in res.links for i in ln.src) == list(range(n))
    assert sorted(j for ln in res.links for j in ln.trg) == list(range(n))


def test_two_sentence_segment_made_by_anchors_in_a_longer_document(tmp_path):
    cuts = tmp_path / 'cuts.tsv'
    cuts.write_text('3\t3\n5\t5\n', encoding='utf-8')          # segments of 3, 2 and 3 sentences
    src = ['The cat sat on mat number %d.' % i for i in range(8)]
    trg = ['Kottur sat a mottu nr %d.' % i for i in range(8)]
    res = _fake_aligner(anchors='file:%s' % cuts).align(src, trg)
    assert sorted(i for ln in res.links for i in ln.src) == list(range(8))


# ---------------------------------------------------------------------------- back-pointers
def test_max_concatenations_above_15_is_refused():
    with pytest.raises(ValueError, match='at most 15'):
        AlignConfig(max_concatenations=16).resolve()
    assert AlignConfig(max_concatenations=15).resolve().max_concatenations == 15
    with pytest.raises(ValueError, match='at most 15'):
        DPKernelPython(5, 5, 16, 0.0, 1, 110, 100, 0.0, 3, False)
    try:
        from sentalign.pathfinding._dp_cython import DPKernelCython
    except ImportError:
        return
    with pytest.raises(ValueError, match='at most 15'):
        DPKernelCython(5, 5, 16, 0.0, 1, 110, 100, 0.0, 3, False)


# ---------------------------------------------------------------------------- word limits after the search
class _Scorer:
    def pair(self, s0, ks, t0, kt):
        return 0.9


def test_span_scores_respect_the_word_limits():
    src = Side(['one two three', 'four five'], max_k=4)
    trg = Side(['a b c', 'd e'], max_k=4)
    free = SpanScores(_Scorer(), src, trg)
    assert free(0, 2, 0, 2) == pytest.approx(0.9)
    limited = SpanScores(_Scorer(), src, trg, 1, 4)           # at most 4 words per span
    assert limited(0, 1, 0, 1) == pytest.approx(0.9)
    assert math.isnan(limited(0, 2, 0, 1))                    # 5 source words
    assert math.isnan(limited(0, 1, 0, 2))                    # 5 target words
    assert math.isnan(SpanScores(_Scorer(), src, trg, 4, 110)(1, 2, 1, 2))   # 2 words, minimum 4


def test_absorption_does_not_build_a_link_over_the_word_limit():
    src = Side(['one two three', 'four five'], max_k=4)
    trg = Side(['a b c', 'd e'], max_k=4)

    class Rising:
        def pair(self, s0, ks, t0, kt):
            return 0.5 + 0.1 * (ks + kt)

    from sentalign.postprocess.reevaluate import Link
    links = [Link(0, 1, 0, 1, 0.7)]
    absorb(links, {1}, {1}, SpanScores(Rising(), src, trg, 1, 4))
    assert (links[0].s1, links[0].t1) == (1, 1)               # unlimited, it would take sentence 1 on both sides
    links = [Link(0, 1, 0, 1, 0.7)]
    absorb(links, {1}, {1}, SpanScores(Rising(), src, trg))
    assert (links[0].s1, links[0].t1) == (2, 2)


# ---------------------------------------------------------------------------- literal NULLALIGN
def _write(compat, tmp_path, src, trg, path='[0]:[0]\n[1]:[]\n[]:[1]\n'):
    me = types.SimpleNamespace(cfg=types.SimpleNamespace(compat=compat), output_folder=str(tmp_path))
    scores = '0.9\n0.0\n0.0\n'
    pairs = Aligner.get_pairs(me, path, dict(enumerate(src)), dict(enumerate(trg)), scores)
    Aligner.write_pairs_to_file(me, pairs, 'f', scores)
    return (tmp_path / 'f.aligned').read_text(encoding='utf-8').splitlines()


def test_a_sentence_with_the_word_nullalign_is_kept(tmp_path):
    src = ['The NULLALIGN marker.', 'lonely source']
    trg = ['Merkið NULLALIGN.', 'lonely target']
    assert _write('none', tmp_path, src, trg) == ['The NULLALIGN marker.\tMerkið NULLALIGN.\t0.9']
    # version 1 dropped it; that stays under paper-2023
    assert _write('paper-2023', tmp_path, src, trg) == []
