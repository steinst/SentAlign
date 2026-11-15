"""The code examples of docs/extending.md run (so the documentation does not rot)."""
import os
import re


from sentalign.api import SentAlign
from sentalign.document import Side

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SRC = ['The cat sat on the mat.', 'It was warm.', 'Outside the rain fell.', 'Nobody minded the weather at all.']
TRG = ['Kötturinn sat á mottunni.', 'Það var hlýtt.', 'Úti rigndi.', 'Engum var sama um veðrið.']


def _blocks():
    text = open(os.path.join(ROOT, 'docs', 'extending.md'), encoding='utf-8').read()
    return re.findall(r'```python\n(.*?)```', text, flags=re.S)


def _block(marker):
    found = [b for b in _blocks() if marker in b]
    assert len(found) == 1, marker
    return found[0]


def test_scorer_example_runs_end_to_end_without_a_model():
    ns = {}
    exec(_block('class LengthScorer'), ns)
    res = SentAlign(scorer='length', device='cpu').align(SRC, TRG)
    assert sorted(i for ln in res.links for i in ln.src) == [0, 1, 2, 3]
    assert len(res.aligned()) >= 3


def test_python_api_example_with_the_registered_scorer():
    exec(_block('class LengthScorer'), {})
    result = SentAlign(scorer='length', device='cpu').align(SRC, TRG)
    assert result.pairs() and all(isinstance(p[2], float) for p in result.pairs())


def test_pathfinder_example_runs_through_the_pipeline():
    exec(_block('class LengthScorer'), {})
    exec(_block('class GreedyOneToOne'), {})
    res = SentAlign(scorer='length', pathfinder='greedy-1-1', compat='none', device='cpu').align(SRC, TRG)
    assert res.aligned() and all(len(ln.src) == 1 and len(ln.trg) == 1 for ln in res.aligned())


def test_anchor_example_returns_valid_anchors():
    ns = {}
    exec(_block('class EveryThousand'), ns)
    from sentalign.anchors import get_anchorer, validate
    strategy = get_anchorer('every-thousand')()
    n = 2500
    found = strategy.find(Side(['a b c d'] * n, max_k=1), Side(['a b c d'] * n, max_k=1), None, 10)
    assert validate(found, n, n) == [(833, 833), (1667, 1667)]
