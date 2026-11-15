"""Scorer interface and registry.

A scorer tells the aligner how similar a span of source sentences is to a span of target sentences.  The path
search and the anchoring only call ``block`` and ``pair``; they never see a model or a vector.  To add a scorer
(another embedding model type, a cross-encoder, a length model ...), subclass ``Scorer`` and register it, either with
``register_scorer`` or through the entry-point group ``sentalign.scorers`` of an installed package; encoders likewise
with ``register_encoder`` / ``sentalign.encoders``.
"""
from importlib import metadata


class Scorer:
    """``prepare`` and ``block`` are all a scorer must have.

    Optional capability, batching: a scorer that also has ``encode_sides(sides) -> list of per-side data`` and
    ``use_vectors(src, trg, src_data, trg_data)`` (as ``prepare``, with the data ``encode_sides`` returned for these
    two sides) gets the sides of many small documents at once (``--batch-texts``).  Without these two methods every
    document is prepared on its own.  They are not defined here on purpose: their presence is the capability.
    """
    kind = 'abstract'
    needs_encoder = True      # False: the pipeline does not load an encoder model for this scorer

    def prepare(self, src, trg):
        """Called once per document with the two ``sentalign.document.Side`` objects."""
        raise NotImplementedError

    def block(self, ks, kt, i0, i1, j0, j1, out=None):
        """float32 array (i1-i0, j1-j0): score of the ks-sentence source span starting at i against the
        kt-sentence target span starting at j, for i in [i0, i1) and j in [j0, j1)."""
        raise NotImplementedError

    def pair(self, src_start, src_size, trg_start, trg_size):
        return float(self.block(src_size, trg_size, src_start, src_start + 1, trg_start, trg_start + 1)[0, 0])

    def span_vectors(self, side, k, first, n):
        """(n - k + 1, d) float32: vectors of the k-sentence spans starting at sentences first .. first+n-k of the
        'src' or 'trg' side, for path finders that work on vectors.  None: this scorer has no vectors."""
        return None

    def text_vector(self, side, text):
        """Vector of an arbitrary text, encoded like the texts of that side; None if not available."""
        return None

    def describe(self):
        return {'scorer': self.kind}


_SCORERS = {}
_ENCODERS = {}


def register_scorer(name, cls):
    _SCORERS[name] = cls


def register_encoder(name, cls):
    _ENCODERS[name] = cls


def _from_entry_points(group, table):
    try:
        eps = metadata.entry_points(group=group)
    except TypeError:  # Python < 3.10 style API
        eps = metadata.entry_points().get(group, [])
    for ep in eps:
        table.setdefault(ep.name, ep)


def _resolve(table, group, name, what):
    if name not in table:
        _from_entry_points(group, table)
    if name not in table:
        raise KeyError('unknown %s %r (known: %s)' % (what, name, ', '.join(sorted(table)) or 'none'))
    obj = table[name]
    if isinstance(obj, metadata.EntryPoint):
        obj = table[name] = obj.load()
    return obj


def get_scorer(name):
    return _resolve(_SCORERS, 'sentalign.scorers', name, 'scorer')


def get_encoder(name):
    return _resolve(_ENCODERS, 'sentalign.encoders', name, 'encoder')
