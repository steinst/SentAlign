"""Documents, encoders, the embedding scorer, the cache and the version-1 views.

Fast tests use the saved version-1 overlap files and vectors in tests/fixtures (mini or eval) and fake encoders;
the slow ones load LaBSE (and a second encoder) and run on cuda or cpu.
"""
import glob
import os

import numpy as np
import pytest

from conftest import run_cli
from sentalign.document import BLANK_LINE, MAX_TEXT_CHARS, Side
from sentalign.scoring import Scorer, get_encoder, get_scorer, register_scorer
from sentalign.scoring.cache import EmbeddingCache
from sentalign.scoring.embedding import EmbeddingScorer, anchor_lists, legacy_views, text_dict
from sentalign.scoring.encoders import EncodingError, HFEncoder, prepare_text

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SETS = {'bleualign': ('deu', 'fra'), 'parice': ('eng', 'isl')}


def fixture_files():
    """(set, file, side, sentence file, fixture base) for every saved file and side."""
    out = []
    for name, (sl, tl) in sorted(SETS.items()):
        for sub in ('eval', 'mini'):
            emb = os.path.join(ROOT, 'tests', 'fixtures', sub, name, 'embeddings')
            if glob.glob(os.path.join(emb, '*.labse_emb')):
                break
        else:
            continue
        for f in sorted(glob.glob(os.path.join(emb, '*.src.overlaps'))):
            fn = os.path.basename(f)[:-len('.src.overlaps')]
            for side, lang in (('src', sl), ('trg', tl)):
                out.append((name, fn, side, os.path.join(ROOT, 'eval_data', name, lang, fn), os.path.join(emb, fn + '.' + side)))
    return out


FIXTURES = fixture_files()
pytestmark_fixtures = pytest.mark.skipif(not FIXTURES, reason='no fixtures (tests/fixtures/mini or eval)')


def _target_file(name, src_file):
    """The target-language file of a source-language sentence file (works with Windows paths too)."""
    return os.path.join(ROOT, 'eval_data', name, SETS[name][1], os.path.basename(src_file))


def _lines(path):
    with open(path, encoding='utf-8') as f:
        return f.readlines()


def _scorer_from_fixture(src_file, trg_file, src_base, trg_base):
    src, trg = Side.from_file(src_file), Side.from_file(trg_file)
    sc = EmbeddingScorer(encoder=None)
    sc.src, sc.trg = src, trg
    sc.src_emb = np.load(src_base + '.labse_emb', allow_pickle=True)
    sc.trg_emb = np.load(trg_base + '.labse_emb', allow_pickle=True)
    return sc


# ---------------------------------------------------------------------------- documents
@pytestmark_fixtures
@pytest.mark.parametrize('name,fn,side,sent_file,base', FIXTURES, ids=lambda x: x if isinstance(x, str) and '/' not in x else None)
def test_side_reproduces_version1_overlap_files(name, fn, side, sent_file, base):
    s = Side.from_file(sent_file)
    assert s.texts == [l.rstrip('\n') for l in _lines(base + '.overlaps')]
    assert [s.line_numbers(r) for r in range(s.n_rows)] == [l.rstrip('\n') for l in _lines(base + '.linenumbers')]
    assert s.n_rows == len(np.load(base + '.labse_emb', allow_pickle=True))


def test_side_rows_blank_and_long_lines():
    long = 'x ' * (MAX_TEXT_CHARS // 2 + 10)
    s = Side(['a b', '', 'c', long.strip()], max_k=2)
    assert s.texts[:4] == ['a b', BLANK_LINE, 'c', long.strip()[:MAX_TEXT_CHARS]]
    assert s.texts[4:] == ['a b ' + BLANK_LINE, BLANK_LINE + ' c', ('c ' + long.strip())[:MAX_TEXT_CHARS]]
    assert s.row(2, 1) == 5 and s.rows_of_size(2) == slice(4, 7)
    assert list(s.word_counts) == [2, 0, 1, len(long.split())]
    with pytest.raises(IndexError):
        s.row(2, 3)


def test_blank_and_long_lines_keep_their_version1_behaviour():
    """Version 1 looked up embeddings by the sentence text: a blank line ('' in the sentence table, 'BLANK_LINE'
    in the overlap file) and a sentence over 10,000 characters (cut in the overlap file only) never find their
    vector, so they score 0.  paper-2023 keeps that; version 2 aligns them explicitly."""
    long = ('word ' * 2500).strip()
    s = Side(['one', '', long], max_k=1)
    d = text_dict(s, np.eye(3, dtype=np.float32))
    assert '' not in d and BLANK_LINE in d
    assert long not in d and long[:MAX_TEXT_CHARS].strip() in d    # key = cut text, stripped


# ---------------------------------------------------------------------------- version-1 views and blocks
@pytestmark_fixtures
def test_legacy_views_equal_version1_dictionaries():
    for name, fn, side, sent_file, base in FIXTURES:
        if side != 'src':
            continue
        trg_file = _target_file(name, sent_file)
        sc = _scorer_from_fixture(sent_file, trg_file, base, base[:-3] + 'trg')
        for which, b, d in zip(('src', 'trg'), (base, base[:-3] + 'trg'), legacy_views(sc)):
            v1 = {}
            embs = np.load(b + '.labse_emb', allow_pickle=True)
            for i, line in enumerate(_lines(b + '.overlaps')):   # version 1 open_emb_file (.used = .overlaps)
                v1[line.strip()] = embs[i]
            assert d.keys() == v1.keys(), (fn, which)
            assert all(d[k] is v1[k] or np.array_equal(d[k], v1[k]) for k in v1), (fn, which)


@pytestmark_fixtures
@pytest.mark.parametrize('overlap', [1, 2])
def test_anchor_lists_equal_version1_create_anchor_files(overlap):
    for name, fn, side, sent_file, base in FIXTURES:
        s = Side.from_file(sent_file)
        texts, lines = anchor_lists(s, overlap)
        v1_texts, v1_lines = [], []
        for text, ln in zip(_lines(base + '.overlaps'), _lines(base + '.linenumbers')):
            if len(ln.split(',')) <= overlap:     # every row of the size, the length flag being ineffective
                v1_texts.append(text)
                v1_lines.append(ln)
        assert texts == v1_texts and lines == v1_lines, (fn, side)


@pytestmark_fixtures
def test_block_equals_version1_dot_products():
    name, fn, side, sent_file, base = FIXTURES[0]
    trg_file = _target_file(name, sent_file)
    sc = _scorer_from_fixture(sent_file, trg_file, base, base[:-3] + 'trg')
    n, m = len(sc.src), len(sc.trg)
    for ks in (1, 2, 4):
        for kt in (1, 3):
            i1, j1 = n - ks + 1, m - kt + 1
            blk = sc.block(ks, kt, 0, i1, 0, j1, max_rows=7)    # several row chunks
            assert blk.dtype == np.float32 and blk.shape == (i1, j1)
            for i in range(0, i1, max(1, i1 // 5)):
                for j in range(0, j1, max(1, j1 // 5)):
                    a = sc.src_emb[sc.src.row(ks, i)]
                    b = sc.trg_emb[sc.trg.row(kt, j)]
                    assert abs(blk[i, j] - b.dot(a.transpose())) <= 1e-6   # version 1: trg.dot(src.T)
                    assert abs(sc.pair(i, ks, j, kt) - blk[i, j]) <= 1e-6
    with pytest.raises(IndexError):
        sc.block(2, 1, 0, n, 0, m)


# ---------------------------------------------------------------------------- encoders without a model
class FakeEncoder(HFEncoder):
    """Vectors that depend only on the text; optional out-of-memory above a batch size."""

    def __init__(self, oom_above=None, always_oom=False, **kw):
        super().__init__(model_name='fake', device='cpu', **kw)
        self.oom_above = oom_above
        self.always_oom = always_oom
        self.batches = []

    def load(self):
        self.model = object()

    def _encode_batch(self, batch):
        import torch
        if self.always_oom or (self.oom_above and len(batch) > self.oom_above):
            raise torch.OutOfMemoryError('CUDA out of memory (fake)')
        self.batches.append(list(batch))
        v = np.asarray([[len(t), sum(map(ord, t)) % 97, 1.0] for t in batch], dtype=np.float32)
        return v / np.linalg.norm(v, axis=1, keepdims=True)


def test_out_of_memory_halves_the_batch_and_keeps_the_order():
    texts = ['t%d %s' % (i, 'x' * (i % 7)) for i in range(50)]
    ref = FakeEncoder(batch_size=1).encode(texts)
    enc = FakeEncoder(oom_above=6, batch_size=32)
    got = enc.encode(texts)
    assert np.array_equal(got, ref)
    assert enc.oom_retries == 3 and max(len(b) for b in enc.batches) == 4


def test_out_of_memory_at_batch_size_one_raises():
    with pytest.raises(EncodingError):
        FakeEncoder(always_oom=True, batch_size=4).encode(['a', 'b'])


def test_file_order_batches_like_version1_and_length_order_restores_rows():
    texts = ['__a__', 'bb', 'ccc ccc', 'bb', 'd'] * 60          # 300 rows: 128 + 128 + 44 in file order
    f = FakeEncoder(order='file')
    vf = f.encode(texts)
    assert [len(b) for b in f.batches] == [128, 128, 44]
    assert f.batches[0][0] == prepare_text('__a__') == '_a_'    # version 1 collapsed runs of '_'
    l = FakeEncoder(order='length', dedupe=True)
    vl = l.encode(texts)
    assert np.array_equal(vf, vl) and sum(len(b) for b in l.batches) == 4   # 4 distinct texts


def test_cache_round_trip_and_fingerprint(tmp_path):
    texts = ['one', 'two', 'three']
    cache = EmbeddingCache(tmp_path)
    enc = FakeEncoder()
    sc = EmbeddingScorer(enc, cache=cache)
    side = Side(texts, max_k=1)
    sc.prepare(side, side)
    assert sc.cache_hits == 1          # the target side is the same text: read back from the cache
    again = EmbeddingScorer(FakeEncoder(), cache=cache)
    again.prepare(side, side)
    assert again.cache_hits == 2 and np.array_equal(again.src_emb, sc.src_emb)
    assert FakeEncoder(pooling='mean').fingerprint() != enc.fingerprint()
    assert FakeEncoder(order='length').fingerprint() != enc.fingerprint()


def test_registry_and_custom_scorer():
    assert get_encoder('hf') is HFEncoder and get_scorer('embedding') is EmbeddingScorer
    with pytest.raises(KeyError):
        get_scorer('no-such-scorer')

    class LengthScorer(Scorer):
        kind = 'length'

        def prepare(self, src, trg):
            self.src, self.trg = src, trg

        def block(self, ks, kt, i0, i1, j0, j1, out=None):
            a = np.asarray([sum(self.src.word_counts[i:i + ks]) for i in range(i0, i1)], dtype=np.float32)
            b = np.asarray([sum(self.trg.word_counts[j:j + kt]) for j in range(j0, j1)], dtype=np.float32)
            return (np.minimum.outer(a, b) / np.maximum(np.maximum.outer(a, b), 1)).astype(np.float32)

    register_scorer('length-test', LengthScorer)
    sc = get_scorer('length-test')()
    sc.prepare(Side(['a b', 'c']), Side(['a', 'b c d']))
    assert sc.pair(0, 1, 1, 1) == pytest.approx(2 / 3)


# ---------------------------------------------------------------------------- with the model
@pytest.mark.slow
@pytest.mark.gpu_or_cpu
@pytestmark_fixtures
def test_hf_encoder_reproduces_version1_vectors(device, labse_available):
    """paper-2023 settings: the vectors of the saved files again (byte-identical on the reference GPU with
    SENTALIGN_STRICT_BYTES=1, otherwise within 1e-5)."""
    enc = HFEncoder(device=device)               # defaults = paper-2023: LaBSE, pooler, 128, file order
    for name, fn, side, sent_file, base in FIXTURES[:4]:
        got = enc.encode(Side.from_file(sent_file).texts)
        ref = np.load(base + '.labse_emb', allow_pickle=True)
        if os.environ.get('SENTALIGN_STRICT_BYTES') == '1' and device.startswith('cuda'):
            assert np.array_equal(got, ref), (fn, side)
        else:
            assert np.abs(got - ref).max() <= 1e-5, (fn, side)


SECOND_ENCODER = 'sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2'


@pytest.mark.slow
@pytest.mark.gpu_or_cpu
def test_second_encoder_end_to_end(tmp_path, device, repo_root):
    """Another encoder (mean pooling, 384 dimensions) through the whole command line."""
    cached = glob.glob(os.path.join(os.path.expanduser('~'), '.cache', 'huggingface', 'hub',
                                    'models--' + SECOND_ENCODER.replace('/', '--'), 'snapshots', '*'))
    if not cached and not os.environ.get('SENTALIGN_ALLOW_DOWNLOAD'):
        pytest.skip('%s not in the Hugging Face cache (set SENTALIGN_ALLOW_DOWNLOAD=1)' % SECOND_ENCODER)
    src = os.path.join(repo_root, 'eval_data', 'parice', 'eng', 'n_1.txt')
    trg = os.path.join(repo_root, 'eval_data', 'parice', 'isl', 'n_1.txt')
    out = tmp_path / 'out'
    r = run_cli([src, trg, '-o', out, '--encoder', SECOND_ENCODER, '--pooling', 'mean', '--compat', 'none',
                 '-device', device, '-proc', '2'])
    assert r.returncode == 0, r.stderr[-3000:]
    path = (out / 'n_1.txt.path').read_text(encoding='utf-8').splitlines()
    n_src = sum(1 for _ in open(src, encoding='utf-8'))
    aligned = [l for l in path if not l.startswith('[]') and ':[]:' not in l]
    # plumbing, not quality: the scoring constants (cutoff 0.4, ...) were tuned for LaBSE, and this model's
    # similarities are lower, so fewer links pass (12 of 25 sentences on this file); accuracy is measured separately
    assert aligned and all(-1.0001 <= float(l.rsplit(':', 1)[1]) <= 1.0001 for l in aligned)
    assert n_src == 25
    import json
    rec = json.loads((out / 'n_1.txt.json').read_text(encoding='utf-8'))
    assert rec['scoring']['encoder']['model'] == SECOND_ENCODER and rec['scoring']['encoder']['pooling'] == 'mean'


# ---------------------------------------------------------------------------- half precision (--half)
def test_half_option_changes_the_fingerprint_only_when_on_and_needs_cuda():
    from sentalign.scoring.encoders import ModelLoadError
    assert HFEncoder(device='cpu').fingerprint() == HFEncoder(device='cpu', half=False).fingerprint()
    assert HFEncoder(device='cuda', half=True).fingerprint() != HFEncoder(device='cuda').fingerprint()
    assert HFEncoder(device='cuda', half=True).describe()['half'] is True
    with pytest.raises(ModelLoadError, match='half precision needs a CUDA device'):
        HFEncoder(device='cpu', half=True).load()


@pytest.mark.slow
@pytest.mark.gpu_or_cpu
def test_half_precision_vectors_are_float32_and_close(device, labse_available):
    if not device.startswith('cuda'):
        pytest.skip('half precision needs a GPU')
    texts = ['The cat sat on the mat.', 'Kötturinn sat á mottunni.', 'Das ist ein Satz.', 'Hvað er klukkan?', 'Yes.'] * 4
    full = HFEncoder(device=device).encode(texts)
    half = HFEncoder(device=device, half=True).encode(texts)
    assert half.dtype == np.float32 and half.shape == full.shape
    assert np.allclose(np.linalg.norm(half, axis=1), 1.0, atol=1e-5)
    cos = (full * half).sum(axis=1)
    assert cos.min() > 0.999, cos.min()


def test_half_is_on_by_default_only_for_cuda_and_v2():
    from sentalign.config import AlignConfig
    assert AlignConfig(proc_device='cuda').half is True
    assert AlignConfig(proc_device='cuda:1').half is True
    assert AlignConfig(proc_device='cuda', half=False).half is False
    assert AlignConfig(proc_device='cpu').half is False
    assert AlignConfig(proc_device='cuda', compat='paper-2023').half is False
    assert AlignConfig(proc_device='cuda', compat='paper-2023', half=True).deviations_from_paper() == {'half': True}
