"""The Python API, aligning several files in one process, and encoding small files together."""
import glob
import os
import re
import shutil

import numpy as np
import pytest

import sentalign
from sentalign.api import Alignment, Link, SentAlign, parse_path
from sentalign.document import Side
from sentalign.scoring.embedding import EmbeddingScorer

from test_scoring import FakeEncoder

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def _fake_aligner(**options):
    """A SentAlign whose encoder is a fake (vectors from the text): plumbing tests without a model."""
    options.setdefault('device', 'cpu')
    sa = SentAlign(**options)
    cfg = sa.config
    enc = FakeEncoder(order=cfg.batch_order, dedupe=cfg.dedupe_texts)
    sa._aligner.scorer = EmbeddingScorer(enc)
    return sa


def _lines(path):
    with open(path, encoding='utf-8') as f:
        return [line.strip() for line in f]


def test_lazy_exports_and_import_without_torch():
    import subprocess
    import sys
    out = subprocess.run([sys.executable, '-c', 'import sys, sentalign; print("torch" in sys.modules); '
                          'print(sentalign.SentAlign.__name__, sentalign.Alignment.__name__)'],
                         capture_output=True, text=True, env=dict(os.environ, PYTHONPATH=os.path.join(ROOT, 'src')))
    assert out.stdout.split() == ['False', 'SentAlign', 'Alignment'], out.stderr
    with pytest.raises(AttributeError):
        sentalign.nothing_here


def test_options_are_checked():
    with pytest.raises(TypeError):
        SentAlign(no_such_option=1)
    with pytest.raises(TypeError):
        SentAlign(sentalign.api.AlignConfig(), device='cpu')
    assert SentAlign(encoder='x/y', device='cpu', cutoff=0.5).config.model_name == 'x/y'


def test_align_returns_every_sentence_once_and_formats():
    src = ['The cat sat on the mat.', 'It was warm.', 'Outside the rain fell.', 'Nobody minded the weather at all.']
    trg = ['Kötturinn sat á mottunni.', 'Það var hlýtt.', 'Úti rigndi.', 'Engum var sama um veðrið.']
    res = _fake_aligner().align(src, trg)
    assert isinstance(res, Alignment) and len(res) == len(res.links)
    assert sorted(i for ln in res.links for i in ln.src) == list(range(4))
    assert sorted(j for ln in res.links for j in ln.trg) == list(range(4))
    for line, ln in zip(res.to_path_lines(), res.links):
        assert re.fullmatch(r'\[[\d,]*\]:\[[\d,]*\]:[\d.e-]+', line), line
    for s, t, score in res.pairs():
        assert s and t
    d = res.to_dict()
    assert len(d['links']) == len(res.links) and 'timings_seconds' in d['info']


def test_empty_input_is_an_error():
    with pytest.raises(ValueError):
        _fake_aligner().align([], ['x'])


def test_parse_path_round_trip():
    links = parse_path('[0,1:0]\n[2:]\n[:1]\n', '0.7565561\n0\n0\n')
    assert [(l.src, l.trg) for l in links] == [((0, 1), (0,)), ((2,), ()), ((), (1,))]
    assert links[0].score_text == '0.7565561' and not links[1].is_aligned and links[0].shape == '2:1'
    assert Link((1,), (2,), 0.5).score_text == '0.5'


def test_align_many_equals_one_by_one_with_batched_encoding():
    """Several documents encoded together give the same alignments as one at a time (the fake encoder is exact)."""
    docs = []
    for k in range(5):
        s = ['Sentence %d of document %d, with some words in it.' % (i, k) for i in range(12 + k)]
        t = ['Setning %d í skjali %d, með nokkrum orðum.' % (i, k) for i in range(12 + k)]
        docs.append((s, t))
    one = _fake_aligner(compat='none', batch_texts=0)
    many = _fake_aligner(compat='none', batch_texts=100000)
    a = [one.align(s, t).to_path_lines() for s, t in docs]
    b = [r.to_path_lines() for r in many.align_many(docs)]
    assert a == b
    # fewer, fuller encoder batches: 10 half-empty ones against ceil(500 / 128)
    assert len(many._aligner.scorer.encoder.batches) < len(one._aligner.scorer.encoder.batches)
    assert len(many._aligner.scorer.encoder.batches) == 4


def test_encode_sides_equals_encoding_each_side():
    sides = [Side(['alpha beta gamma', 'delta epsilon', 'alpha beta gamma'], max_k=2), Side(['delta epsilon', 'zeta'], max_k=2)]
    sc = EmbeddingScorer(FakeEncoder(order='length', dedupe=True))
    together = sc.encode_sides(sides)
    for side, vec in zip(sides, together):
        assert np.array_equal(vec, FakeEncoder(order='file').encode(side.texts))


def test_batched_cache(tmp_path):
    from sentalign.scoring.cache import EmbeddingCache
    sides = [Side(['one two', 'three four'], max_k=1), Side(['five six'], max_k=1)]
    sc = EmbeddingScorer(FakeEncoder(), cache=EmbeddingCache(tmp_path))
    first = sc.encode_sides(sides)
    sc2 = EmbeddingScorer(FakeEncoder(), cache=EmbeddingCache(tmp_path))
    second = sc2.encode_sides(sides)
    assert sc2.cache_hits == 2 and all(np.array_equal(a, b) for a, b in zip(first, second))


# ---------------------------------------------------------------------------- with the model
@pytest.mark.slow
@pytest.mark.gpu_or_cpu
def test_api_equals_the_command_line(tmp_path, device, labse_available, repo_root):
    """The same sentences through sentalign.SentAlign and through the command line give the same alignment."""
    from conftest import run_cli
    corpus = tmp_path / 'c'
    for lang in ('eng', 'isl'):
        shutil.copytree(os.path.join(repo_root, 'eval_data', 'parice', lang), corpus / lang)
    r = run_cli(['-dir', corpus, '-sl', 'eng', '-tl', 'isl', '-device', device, '-proc', '1'])
    assert r.returncode == 0, r.stderr[-2000:]
    sa = SentAlign(device=device)
    names = sorted(os.listdir(corpus / 'eng'))
    results = sa.align_many([(_lines(corpus / 'eng' / n), _lines(corpus / 'isl' / n)) for n in names])
    for n, res in zip(names, results):
        cli = [l.rstrip('\n') for l in open(corpus / 'output' / (n + '.path'), encoding='utf-8')]
        got = res.to_path_lines()
        assert [g.rsplit(':', 1)[0] for g in got] == [c.rsplit(':', 1)[0] for c in cli], n      # same links
        assert max(abs(float(g.rsplit(':', 1)[1]) - float(c.rsplit(':', 1)[1])) for g, c in zip(got, cli)) < 1e-4, n


@pytest.mark.slow
@pytest.mark.gpu_or_cpu
def test_many_files_in_one_run_equal_files_one_by_one(tmp_path, device, labse_available, repo_root):
    """--batch-texts (small files encoded together) does not change the alignments."""
    from conftest import run_cli
    outs = {}
    for label, extra in (('one', ['--batch-texts', '0']), ('together', ['--batch-texts', '200000'])):
        corpus = tmp_path / label
        for lang in ('eng', 'isl'):
            shutil.copytree(os.path.join(repo_root, 'eval_data', 'parice', lang), corpus / lang)
        r = run_cli(['-dir', corpus, '-sl', 'eng', '-tl', 'isl', '-device', device, '-proc', '1'] + extra)
        assert r.returncode == 0, r.stderr[-2000:]
        outs[label] = corpus
    for f in sorted(glob.glob(str(outs['one'] / 'output' / '*.path'))):
        a = [l.rsplit(':', 1)[0] for l in open(f, encoding='utf-8')]
        b = [l.rsplit(':', 1)[0] for l in open(os.path.join(outs['together'], 'output', os.path.basename(f)), encoding='utf-8')]
        assert a == b, f


def test_without_the_compiled_kernel_the_python_reference_is_used(caplog):
    """An installation without a C compiler still works (slowly): the same alignment from the reference kernel."""
    src = ['The cat sat on the mat.', 'It was warm.', 'Outside the rain fell.', 'Nobody minded the weather at all.']
    trg = ['Kötturinn sat á mottunni.', 'Það var hlýtt.', 'Úti rigndi.', 'Engum var sama um veðrið.']
    fast = _fake_aligner().align(src, trg).to_path_lines()
    slow = _fake_aligner(path_backend='python').align(src, trg).to_path_lines()
    assert fast == slow
