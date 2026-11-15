"""The vecalign path finder (Vecalign's search, included in the package) and the path-finder options."""
import os

import numpy as np
import pytest

from sentalign.pathfinding import LINK, Segment, get_pathfinder
from sentalign.pathfinding.vecalign_finder import VecalignFinder, end_indexed

REPO = os.environ.get('SENTALIGN_VECALIGN_REPO')        # a clone of github.com/thompsonb/vecalign (Bleualign data)


def _segment(src, trg, max_k=4, pad=None):
    """Segment over span vectors given as lists [k-1] -> (n - k + 1, d)."""
    return Segment(len(src[0]), len(trg[0]), max_k, None, None, None,
                   src_vectors=lambda k: src[k - 1], trg_vectors=lambda k: trg[k - 1],
                   text_vector=(lambda side, text: pad) if pad is not None else None)


def test_options_are_checked_and_converted():
    f = VecalignFinder().configure({'alignment_max_size': '5', 'del_percentile_frac': '0.3'})
    assert f.options['alignment_max_size'] == 5 and f.options['del_percentile_frac'] == 0.3
    assert f.options['seed'] == 42 and VecalignFinder.uses_anchors is False
    with pytest.raises(ValueError, match='no option'):
        VecalignFinder().configure({'cutoff': 1})
    assert get_pathfinder('vecalign') is VecalignFinder
    assert get_pathfinder('sentalign-dp')().configure({}).options == {}


def test_command_line_options_reach_the_config():
    import argparse
    from sentalign.config import add_alignment_arguments, config_from_args
    p = argparse.ArgumentParser()
    add_alignment_arguments(p)
    cfg = config_from_args(p.parse_args(['--pathfinder', 'vecalign', '--pathfinder-opt', 'alignment-max-size=5', 'seed=1']))
    assert cfg.pathfinder_options == {'alignment_max_size': '5', 'seed': '1'}


def test_end_indexed_array_is_vecaligns_layout():
    n, d = 5, 3
    spans = [np.full((n - k + 1, d), 0.0, dtype=np.float32) for k in range(1, 4)]
    for k, a in enumerate(spans, start=1):
        a[:, 0] = np.arange(n - k + 1) + 10 * k                # value = 10 k + start
    pad = np.full(d, -1, dtype=np.float32)
    out = end_indexed(lambda k: spans[k - 1], n, 3, pad)
    assert out.shape == (3, n, d)
    assert list(out[0, :, 0]) == [10, 11, 12, 13, 14]            # one sentence ending at j starts at j
    assert list(out[1, :, 0]) == [-1, 20, 21, 22, 23]            # two sentences ending at j start at j - 1
    assert list(out[2, :, 0]) == [-1, -1, 30, 31, 32]
    short = end_indexed(lambda k: spans[k - 1][:max(0, 2 - k + 1)], 2, 3, pad)   # a document of two sentences
    assert list(short[2, :, 0]) == [-1, -1]
    with pytest.raises(ValueError, match='scorer with vectors'):
        end_indexed(lambda k: None, n, 3, pad)


def _unit(a):
    return (a / np.linalg.norm(a, axis=-1, keepdims=True)).astype(np.float32)


def test_parallel_documents_align_one_to_one():
    rng = np.random.default_rng(0)
    n, d = 60, 32
    base = _unit(rng.normal(size=(n, d)))

    def spans(noise):
        one = _unit(base + noise * rng.normal(size=(n, d)))
        return [_unit(np.stack([one[p:p + k].sum(0) for p in range(n - k + 1)])) for k in range(1, 5)]

    steps = VecalignFinder().find(_segment(spans(0.02), spans(0.02)))
    assert steps == [(LINK, i, 1, i, 1) for i in range(n)]
    assert VecalignFinder().find(_segment(spans(0.02), spans(0.02))) == steps        # seeded: repeatable
    with pytest.raises(ValueError, match='max-concatenations'):
        VecalignFinder().configure({'alignment_max_size': 8}).find(_segment(spans(0.02), spans(0.02)))
    assert VecalignFinder().find(_segment([np.zeros((0, d), np.float32)] * 4, spans(0.02))) == []


def _random_documents(seed, n, m, d=24):
    """Two documents with a common part, extra sentences on both sides, as end-indexed Vecalign arrays."""
    rng = np.random.default_rng(seed)
    base = rng.normal(size=(max(n, m), d))
    def doc(size):
        one = _unit(base[:size] + 0.3 * rng.normal(size=(size, d)))
        return end_indexed(lambda k: _unit(np.stack([one[p:p + k].sum(0) for p in range(size - k + 1)])), size, 4, one[0])
    return doc(n), doc(m)


@pytest.mark.parametrize('n,m', [(40, 37), (350, 420)])
def test_python_version_of_the_compiled_functions_gives_the_same_alignment(n, m, monkeypatch):
    """Without a C compiler the Python functions are used: same beads (also through the coarse-to-fine levels)."""
    core = pytest.importorskip('sentalign.pathfinding._vecalign_core')
    from sentalign.pathfinding import _vecalign as va, _vecalign_core_py as py
    names = ('make_dense_costs', 'score_path', 'sparse_dp', 'make_sparse_costs', 'dense_dp')
    results = []
    for module in (core, py):
        for name in names:
            monkeypatch.setattr(va, name, getattr(module, name))
        v0, v1 = _random_documents(n, n, m)
        np.random.seed(42)
        stack = va.vecalign(v0, v1, va.make_alignment_types(4), 0.2, 7, 300, 20000, 100)
        results.append((stack[0]['final_alignments'], stack[0]['alignment_scores']))
    assert results[0][0] == results[1][0]
    assert np.allclose(results[0][1], results[1][1], atol=1e-4)


@pytest.mark.skipif(not REPO, reason='set SENTALIGN_VECALIGN_REPO to a clone of the Vecalign repository')
@pytest.mark.parametrize('size', [4, 8])
def test_same_beads_as_the_original_on_its_own_vectors(size):
    """With the LASER vectors shipped with Vecalign, the finder returns what the original returns for each document
    (each one seeded like the first document of a run)."""
    dp = pytest.importorskip('vecalign.dp_utils')
    data = os.path.join(REPO, 'bleualign_data')
    tables = {lang: dp.read_in_embeddings(os.path.join(data, 'overlaps.' + lang), os.path.join(data, 'overlaps.%s.emb' % lang))
              for lang in ('de', 'fr')}
    for t in range(7):
        lines = {lang: open(os.path.join(data, 'test%d.%s' % (t, lang)), encoding='utf-8').readlines() for lang in ('de', 'fr')}
        # the original
        vecs = [dp.make_doc_embedding(*tables[lang], lines[lang], size) for lang in ('de', 'fr')]
        np.random.seed(42)
        stack = dp.vecalign(vecs0=vecs[0], vecs1=vecs[1], final_alignment_types=dp.make_alignment_types(size),
                            del_percentile_frac=0.2, width_over2=int(np.ceil(size / 2.0)) + 5, max_size_full_dp=300,
                            costs_sample_size=20000, num_samps_for_norm=100)
        expected = [(LINK, x[0], len(x), y[0], len(y)) for x, y in stack[0]['final_alignments'] if len(x) and len(y)]
        # the finder, fed with spans that start at each sentence (the layout of the SentAlign scorer)
        sides = []
        for lang in ('de', 'fr'):
            sent2line, emb = tables[lang]
            clean = [dp.preprocess_line(x) for x in lines[lang]]
            sides.append([np.stack([emb[sent2line[' '.join(clean[p:p + k])[:dp.MAX_LINE_CHARS]]]
                                    for p in range(len(clean) - k + 1)]) for k in range(1, size + 1)])
        pads = {'src': tables['de'][1][tables['de'][0]['PAD']], 'trg': tables['fr'][1][tables['fr'][0]['PAD']]}
        seg = Segment(len(lines['de']), len(lines['fr']), size, None, None, None,
                      src_vectors=lambda k: sides[0][k - 1], trg_vectors=lambda k: sides[1][k - 1],
                      text_vector=lambda side, text: pads[side])
        got = VecalignFinder().configure({'alignment_max_size': size}).find(seg)
        assert got == expected, 'test%d' % t
