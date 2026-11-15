"""Short model names, language-aware encoders (LASER) and their plumbing; no model is loaded."""
import numpy as np
import pytest

from sentalign.config import AlignConfig
from sentalign.document import Side
from sentalign.scoring.embedding import EmbeddingScorer
from sentalign.scoring.laser import LaserEncoder, flores_code


def test_short_model_names():
    c = AlignConfig(model_name='bge-m3')
    assert (c.encoder_kind, c.model_name, c.pooling) == ('hf', 'BAAI/bge-m3', 'cls')
    c = AlignConfig(model_name='MiniLM')
    assert c.model_name.endswith('paraphrase-multilingual-MiniLM-L12-v2') and c.pooling == 'mean'
    c = AlignConfig(model_name='laser2', proc_device='cuda')
    assert (c.encoder_kind, c.model_name, c.half) == ('laser', 'laser2', False)
    c = AlignConfig(model_name='labse')
    assert (c.encoder_kind, c.model_name, c.pooling) == ('hf', 'setu4993/LaBSE', 'pooler')
    assert AlignConfig(model_name='some/other-model').model_name == 'some/other-model'


def test_flores_codes():
    assert flores_code('isl') == 'isl_Latn' and flores_code('is') == 'isl_Latn'
    assert flores_code('eng_Latn') == 'eng_Latn' and flores_code('no') == 'nob_Latn'
    assert flores_code('xyz', known=['xyz_Cyrl']) == 'xyz_Cyrl'
    with pytest.raises(ValueError):
        flores_code(None)


class FakeLaser:
    """Encoder that depends on the language: the vector is the language number repeated."""
    kind = 'fake'
    needs_language = True
    calls = []

    def __init__(self):
        self.seen = []

    def fingerprint(self, lang=None):
        return 'fake-' + str(lang)

    def describe(self):
        return {'kind': 'fake'}

    def encode(self, texts, lang=None):
        self.seen.append((lang, len(texts)))
        v = np.zeros((len(texts), 4), dtype=np.float32)
        v[:, 0 if lang == 'eng' else 1] = 1.0
        return v


def test_scorer_passes_the_language_of_each_side():
    enc = FakeLaser()
    sc = EmbeddingScorer(enc)
    src = Side(['a b', 'c d'], lang='eng')
    trg = Side(['e f', 'g h'], lang='isl')
    sc.prepare(src, trg)
    assert [l for l, _ in enc.seen] == ['eng', 'isl']
    enc.seen.clear()
    out = sc.encode_sides([src, trg, Side(['x y'], lang='eng')])
    assert sorted(enc.seen) == [('eng', 4), ('isl', 3)]
    assert out[0][0, 0] == 1.0 and out[1][0, 1] == 1.0 and out[2][0, 0] == 1.0


def test_laser_needs_its_package_and_a_valid_variant():
    from sentalign.scoring.encoders import ModelLoadError
    with pytest.raises(ValueError):
        LaserEncoder('laser9')
    enc = LaserEncoder('laser2')
    assert enc.fingerprint('isl') == enc.fingerprint('eng')          # one model for all languages
    assert LaserEncoder('laser3').fingerprint('isl') != LaserEncoder('laser3').fingerprint('eng')
    try:
        import laser_encoders  # noqa: F401
    except Exception:
        with pytest.raises(ModelLoadError):
            enc.encode(['hello'], lang='eng')


def test_remote_code_is_only_run_on_request():
    from sentalign.scoring.encoders import HFEncoder
    assert HFEncoder().trust_remote_code is False and AlignConfig().trust_remote_code is False
    assert AlignConfig(model_name='some/model', trust_remote_code=True).trust_remote_code is True
