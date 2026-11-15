"""LASER encoders (Facebook/Meta): LASER 2 (one BiLSTM for about 100 languages), LASER 3 (a Transformer per
language, for languages LASER 2 covers badly).

    sentalign ... --encoder laser2          # or laser3

LASER 2 and 3 need the package ``laser_encoders`` (``pip install laser_encoders``; it needs ``fairseq`` and downloads
the model files on first use).  The encoder needs the language of each text: it is taken from
``--source-language`` / ``--target-language`` (ISO 639-3 such as ``isl``, two-letter codes or the FLORES-200 codes
used by LASER, ``isl_Latn``).  With ``laser3`` a language without a LASER 3 model uses LASER 2, as in the package.
Vectors are L2-normalised so that the block score is the cosine.  The score cutoff of the paper (0.4) was tuned for
LaBSE: tune it for LASER on development data.
"""
import logging

import numpy as np

from sentalign.scoring.encoders import ModelLoadError, prepare_text

log = logging.getLogger('sentalign.encoders')

VARIANTS = ('laser2', 'laser3')

# two-letter and 639-3 codes whose FLORES code is not simply <code>_Latn
_SPECIAL = {'no': 'nob_Latn', 'nor': 'nob_Latn', 'nb': 'nob_Latn', 'nob': 'nob_Latn', 'nn': 'nno_Latn',
            'zh': 'zho_Hans', 'zho': 'zho_Hans', 'ru': 'rus_Cyrl', 'rus': 'rus_Cyrl', 'uk': 'ukr_Cyrl',
            'ukr': 'ukr_Cyrl', 'ar': 'arb_Arab', 'ara': 'arb_Arab', 'el': 'ell_Grek', 'ell': 'ell_Grek',
            'he': 'heb_Hebr', 'heb': 'heb_Hebr', 'ja': 'jpn_Jpan', 'jpn': 'jpn_Jpan', 'ko': 'kor_Hang',
            'kor': 'kor_Hang', 'hi': 'hin_Deva', 'hin': 'hin_Deva', 'bg': 'bul_Cyrl', 'bul': 'bul_Cyrl'}
_TWO = {'en': 'eng', 'is': 'isl', 'de': 'deu', 'fr': 'fra', 'es': 'spa', 'it': 'ita', 'pt': 'por', 'nl': 'nld',
        'sv': 'swe', 'da': 'dan', 'fi': 'fin', 'pl': 'pol', 'cs': 'ces', 'hu': 'hun', 'tr': 'tur', 'la': 'lat',
        'et': 'est', 'lv': 'lvs', 'lt': 'lit', 'ro': 'ron', 'ga': 'gle', 'cy': 'cym'}


def flores_code(lang, known=None):
    """FLORES-200 code of a language code: ``isl`` / ``is`` / ``isl_Latn`` -> ``isl_Latn``.  ``known`` (a collection of
    codes) is used to find the script when the code is not in the small table."""
    if lang is None:
        raise ValueError('LASER needs the language of the texts (--source-language / --target-language)')
    lang = str(lang).strip()
    if '_' in lang:
        return lang
    low = lang.lower()
    if low in _SPECIAL:
        return _SPECIAL[low]
    code = _TWO.get(low, low)
    if known:
        hits = sorted(k for k in known if k.startswith(code + '_'))
        if hits:
            return next((h for h in hits if h.endswith('_Latn')), hits[0])
    return code + '_Latn'


class LaserEncoder:
    """texts -> L2-normalised float32 vectors with LASER; one model per language (LASER 3) or one for all (LASER 2)."""

    kind = 'laser'
    needs_language = True

    def __init__(self, model_name='laser2', device='cpu', batch_size=128, dedupe=False, **ignored):
        if model_name not in VARIANTS:
            raise ValueError('LASER variant must be one of %s (got %r)' % (VARIANTS, model_name))
        self.model_name = model_name
        self.device = device
        self.batch_size = int(batch_size)
        self.dedupe = bool(dedupe)
        self._pipelines = {}
        self.oom_retries = 0

    def load(self):
        pass        # models are loaded per language on first use

    def describe(self):
        return {'kind': self.kind, 'model': self.model_name, 'device': str(self.device), 'dedupe': self.dedupe}

    def fingerprint(self, lang=None):
        import hashlib
        import json
        d = {'kind': self.kind, 'model': self.model_name, 'lang': None if self.model_name == 'laser2' else lang}
        return hashlib.sha256(json.dumps(d, sort_keys=True).encode()).hexdigest()[:20]

    # ------------------------------------------------------------------ models
    def _pipeline(self, lang):
        if self.model_name == 'laser2':
            key = 'laser2'
        else:
            key = flores_code(lang)
        if key in self._pipelines:
            return self._pipelines[key]
        try:
            from laser_encoders import LaserEncoderPipeline
            if self.model_name == 'laser2':
                pipe = LaserEncoderPipeline(laser='laser2')
            else:
                pipe = LaserEncoderPipeline(lang=flores_code(lang))
        except ImportError as e:
            raise ModelLoadError('%s needs the package %s: %s' % (
                self.model_name, 'laser_encoders (and fairseq)', e)) from e
        except Exception as e:
            raise ModelLoadError('could not load %s for %r: %s' % (self.model_name, lang, e)) from e
        self._pipelines[key] = pipe
        return pipe

    # ------------------------------------------------------------------ encoding
    def encode(self, texts, lang=None):
        texts = [prepare_text(t) for t in texts]
        if not texts:
            return np.zeros((0, 0), dtype=np.float32)
        pipe = self._pipeline(lang)
        out = None
        for start in range(0, len(texts), 4096):
            vec = np.asarray(pipe.encode_sentences(texts[start:start + 4096]), dtype=np.float32)
            if out is None:
                out = np.empty((len(texts), vec.shape[1]), dtype=np.float32)
            out[start:start + len(vec)] = vec
        norm = np.linalg.norm(out, axis=1, keepdims=True)
        return out / np.maximum(norm, 1e-12)

