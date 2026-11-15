"""Embedding scorer: cosine similarity of encoder vectors, by blocks of spans.

Also builds the text-keyed dictionaries and line lists the version-1 path search and anchoring still need
(``legacy_views``), with exactly the content of the version-1 temporary files.
"""
import logging
import threading

import numpy as np

from sentalign.scoring.base import Scorer

log = logging.getLogger('sentalign.scoring')


class EmbeddingScorer(Scorer):
    kind = 'embedding'

    def __init__(self, encoder, cache=None):
        self.encoder = encoder
        self.cache = cache
        self.src = self.trg = None
        self.src_emb = self.trg_emb = None
        self.cache_hits = 0
        self._extra = {}                     # vectors of single texts asked for by a path finder (text_vector)
        self._extra_lock = threading.Lock()

    # ------------------------------------------------------------------ preparation
    # An encoder with ``needs_language = True`` (LASER) is told the language of the texts: ``encode(texts, lang=...)``
    # and ``fingerprint(lang)``; the other encoders are called as before.
    def _lang_args(self, lang):
        return (lang,) if getattr(self.encoder, 'needs_language', False) else ()

    def _fingerprint(self, lang):
        return self.encoder.fingerprint(*self._lang_args(lang))

    def _encode(self, texts, lang):
        if getattr(self.encoder, 'needs_language', False):
            return self.encoder.encode(texts, lang=lang)
        return self.encoder.encode(texts)

    def _vectors(self, side):
        if self.cache is not None:
            if hasattr(self.encoder, 'load'):
                self.encoder.load()      # the fingerprint includes the resolved model revision
            hit = self.cache.load(self._fingerprint(side.lang), side.texts)
            if hit is not None:
                self.cache_hits += 1
                return hit
        vec = self._encode(side.texts, side.lang)
        if self.cache is not None:
            self.cache.save(self._fingerprint(side.lang), side.texts, vec)
        return vec

    def prepare(self, src, trg):
        """Encode both sides (source first, then target, as version 1)."""
        self.src, self.trg = src, trg
        self.src_emb = self._vectors(src)
        self.trg_emb = self._vectors(trg)
        for side, emb in (('source', self.src_emb), ('target', self.trg_emb)):
            if emb.shape[0] != (src if side == 'source' else trg).n_rows:
                raise RuntimeError('%s: %d vectors for %d spans' % (side, emb.shape[0],
                                                                    (src if side == 'source' else trg).n_rows))

    def encode_sides(self, sides):
        """Vectors for several documents' sides at once (small files: far fewer, fuller encoder batches).  Texts that
        occur more than once are encoded once; the cache is used per side.  -> list of (n_rows, d) arrays."""
        out = [None] * len(sides)
        todo = []
        for k, side in enumerate(sides):
            if self.cache is not None:
                if hasattr(self.encoder, 'load'):
                    self.encoder.load()
                hit = self.cache.load(self._fingerprint(side.lang), side.texts)
                if hit is not None:
                    self.cache_hits += 1
                    out[k] = hit
                    continue
            todo.append(k)
        # sides of different languages are encoded separately when the encoder depends on the language
        by_lang = {}
        for k in todo:
            by_lang.setdefault(sides[k].lang if getattr(self.encoder, 'needs_language', False) else None, []).append(k)
        for lang, group in by_lang.items():
            texts = [t for k in group for t in sides[k].texts]
            first = {}
            index = np.asarray([first.setdefault(t, len(first)) for t in texts], dtype=np.int64)
            vectors = self._encode(list(first), lang)
            pos = 0
            for k in group:
                n = sides[k].n_rows
                out[k] = vectors[index[pos:pos + n]]       # an array of its own: released with its document
                pos += n
                if self.cache is not None:
                    self.cache.save(self._fingerprint(sides[k].lang), sides[k].texts, out[k])
        return out

    def use_vectors(self, src, trg, src_vec, trg_vec):
        """Like ``prepare``, with vectors that were encoded together with other documents."""
        self.src, self.trg, self.src_emb, self.trg_emb = src, trg, src_vec, trg_vec

    # ------------------------------------------------------------------ scores
    def block(self, ks, kt, i0, i1, j0, j1, out=None, max_rows=4096):
        """Scores of all ks-sentence source spans starting in [i0, i1) against all kt-sentence target spans
        starting in [j0, j1), as float32 (rows = source).  Computed in row chunks of `max_rows`."""
        if not (0 <= i0 <= i1 <= len(self.src) - ks + 1) or not (0 <= j0 <= j1 <= len(self.trg) - kt + 1):
            raise IndexError('block out of range: source %d..%d (k=%d, n=%d), target %d..%d (k=%d, n=%d)'
                             % (i0, i1, ks, len(self.src), j0, j1, kt, len(self.trg)))
        s = self.src_emb[self.src.offsets[ks - 1] + i0: self.src.offsets[ks - 1] + i1]
        t = self.trg_emb[self.trg.offsets[kt - 1] + j0: self.trg.offsets[kt - 1] + j1]
        if out is None:
            out = np.empty((i1 - i0, j1 - j0), dtype=np.float32)
        for r in range(0, i1 - i0, max_rows):
            np.matmul(s[r:r + max_rows], t.T, out=out[r:r + max_rows])
        return out

    def pair(self, src_start, src_size, trg_start, trg_size):
        a = self.src_emb[self.src.row(src_size, src_start)]
        b = self.trg_emb[self.trg.row(trg_size, trg_start)]
        return float(np.dot(b, a))

    def span_vectors(self, side, k, first, n):
        doc, emb = (self.src, self.src_emb) if side == 'src' else (self.trg, self.trg_emb)
        if emb is None or not (1 <= k <= doc.max_k):
            return None
        start = doc.offsets[k - 1] + first
        return emb[start:start + max(0, n - k + 1)]

    def text_vector(self, side, text):
        doc = self.src if side == 'src' else self.trg
        lang = getattr(doc, 'lang', None)
        key = (lang if getattr(self.encoder, 'needs_language', False) else None, text)
        with self._extra_lock:
            if key not in self._extra:
                self._extra[key] = np.asarray(self._encode([text], lang)[0], dtype=np.float32)
            return self._extra[key]

    def describe(self):
        return {'scorer': self.kind, 'encoder': self.encoder.describe(),
                'cache': None if self.cache is None else str(self.cache.root), 'cache_hits': self.cache_hits}


# ---------------------------------------------------------------------- version-1 views
def text_dict(side, emb, max_size=None):
    """{text row stripped: vector}, later rows overwrite earlier ones (version 1 ``open_emb_file`` for
    max_size=None, the dictionaries of ``create_anchor_files`` for max_size=1 or 2)."""
    d = {}
    for r, text in enumerate(side.texts):
        if max_size is None or side.sizes[r] <= max_size:
            d[text.strip()] = emb[r]
    return d


def anchor_lists(side, max_size):
    """The text and line-number lists of version 1 ``create_anchor_files`` (strings with the trailing newline).

    KNOWN FAULT (length filter, SentAlign 1.0), fixed in version 2: version 1 read the length flag as the string
    'True'/'False', and bool('False') is True, so every row of the requested sizes is used.
    """
    rows = [r for r in range(side.n_rows) if side.sizes[r] <= max_size]
    texts = [side.texts[r] + '\n' for r in rows]
    lines = [side.line_numbers(r) + '\n' for r in rows]
    return texts, lines


def legacy_views(scorer, max_size=None):
    """(source dict, target dict) for the version-1 code."""
    return text_dict(scorer.src, scorer.src_emb, max_size), text_dict(scorer.trg, scorer.trg_emb, max_size)
