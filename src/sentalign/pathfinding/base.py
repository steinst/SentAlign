"""Path search interface and registry.

A path finder gets one *segment* (the part of a document between two anchors) and returns the best path through it
as a list of steps.  It asks the segment for scores and word counts and never sees a model or a file.  A finder that
works on the vectors themselves (Vecalign) sets ``needs_vectors`` and asks the segment for them; such a finder only
works with a scorer that has vectors.
To add an algorithm, subclass ``PathFinder`` and register it with ``register_pathfinder`` or through the entry-point
group ``sentalign.pathfinders``; select it with ``--pathfinder NAME``.
"""
from dataclasses import dataclass
from importlib import metadata
from typing import Callable

import numpy as np

# one step of a path: (kind, src_start, src_size, trg_start, trg_size), positions relative to the segment
SKIP_SOURCE, SKIP_TARGET, LINK = 'skip-source', 'skip-target', 'link'


@dataclass
class Segment:
    """What a path finder may know about a segment.

    n_src, n_trg     number of sentences on each side
    max_k            longest span (number of sentences) that can be linked
    score_block(ks, kt, p0, p1) -> float32 array (p1 - p0, n_trg - kt + 1): score of the ks-sentence source span
                     starting at sentence p (p0 <= p < p1) against the kt-sentence target span starting at q, for
                     every q; NaN where no score exists (e.g. a blank line)
    src_words[ks-1][p], trg_words[kt-1][q]   words of the span of ks (kt) sentences starting at p (q)
    src_sentences, trg_sentences             the sentence texts (optional; Gale-Church needs them)
    src_vectors(k), trg_vectors(k)           (n - k + 1, d) float32: the vectors of the k-sentence spans starting at
                     each sentence of the segment, or None when the scorer has no vectors (optional)
    text_vector(side, text)                  the vector of an arbitrary text encoded like the 'src' or 'trg' side, or
                     None (optional; Vecalign pads with the vector of the text 'PAD')
    """
    n_src: int
    n_trg: int
    max_k: int
    score_block: Callable
    src_words: list
    trg_words: list
    src_sentences: list = None    # the sentence texts of the segment (only needed by length-based path finders)
    trg_sentences: list = None
    takes_out: bool = False       # score_block accepts out=<array> and writes the scores there (saves a copy)
    src_vectors: Callable = None
    trg_vectors: Callable = None
    text_vector: Callable = None


@dataclass
class PathParams:
    """The settings of the SentAlign path search (same meaning and defaults as the 2023 command line)."""
    score_cutoff: float = 0.4
    free_concatenations: int = 2
    minimum_length_words: int = 1
    maximum_length_words: int = 110
    penalty_after_words: int = 80
    penalty_per_word: float = 0.01
    reward: str = 'sentalign'           # sentalign | squared-cosine
    length_prior: bool = False          # squared-cosine: times sqrt(log2(1 + shorter / longer)) of the character lengths
    paper_arithmetic: bool = False      # float32 accumulation and the other 2023 details (paper-2023 mode)
    backend: str = 'auto'               # auto | cython | rust | python
    threads: int = 1                    # cores for one segment (only the rust kernel can use more than one)
    band_bytes: int = 256 * 1024 * 1024  # memory for the scores of one band of rows


class PathFinder:
    name = 'abstract'
    postprocess = True     # False: the path is the result (links are only scored), no re-evaluation or absorption
    uses_anchors = True    # False: `--anchors auto` gives the finder the whole document (it needs no anchors)
    needs_vectors = False  # True: the finder reads segment.src_vectors / trg_vectors
    option_defaults = {}   # the finder's own settings and their defaults (--pathfinder-opt NAME=VALUE)

    def __init__(self, params=None):
        self.params = params or PathParams()
        self.options = dict(self.option_defaults)

    def configure(self, options):
        """Set the finder's own options; values given as text are converted to the type of the default."""
        if not hasattr(self, 'options'):                 # a subclass with its own __init__
            self.options = dict(self.option_defaults)
        for key, value in (options or {}).items():
            if key not in self.option_defaults:
                raise ValueError('path finder %r has no option %r (options: %s)'
                                 % (self.name, key, ', '.join(sorted(self.option_defaults)) or 'none'))
            default = self.option_defaults[key]
            if isinstance(value, str) and not isinstance(default, str) and default is not None:
                if isinstance(default, bool):
                    value = value.lower() in ('1', 'true', 'yes', 'on')
                else:
                    value = type(default)(value)
            self.options[key] = value
        return self

    def find(self, segment):
        """-> list of steps (kind, src_start, src_size, trg_start, trg_size) covering every sentence once, in order."""
        raise NotImplementedError


def span_words(counts, max_k):
    """[ks-1][p] = sum of counts[p:p+ks] for every span that fits (int32 arrays)."""
    counts = np.asarray(counts, dtype=np.int64)
    csum = np.concatenate(([0], np.cumsum(counts)))
    return [(csum[k:] - csum[:len(csum) - k]).astype(np.int32) if k <= len(counts) else np.zeros(0, dtype=np.int32)
            for k in range(1, max_k + 1)]


_PATHFINDERS = {}


def register_pathfinder(name, cls):
    _PATHFINDERS[name] = cls


def get_pathfinder(name):
    if name not in _PATHFINDERS:
        try:
            eps = metadata.entry_points(group='sentalign.pathfinders')
        except TypeError:
            eps = metadata.entry_points().get('sentalign.pathfinders', [])
        for ep in eps:
            _PATHFINDERS.setdefault(ep.name, ep)
    if name not in _PATHFINDERS:
        raise KeyError('unknown path finder %r (known: %s)' % (name, ', '.join(sorted(_PATHFINDERS)) or 'none'))
    obj = _PATHFINDERS[name]
    if isinstance(obj, metadata.EntryPoint):
        obj = _PATHFINDERS[name] = obj.load()
    return obj
