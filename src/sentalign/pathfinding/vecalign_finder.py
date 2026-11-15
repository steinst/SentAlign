"""Vecalign (Thompson and Koehn 2019) as a path finder.

    sentalign ... --pathfinder vecalign [--pathfinder-opt alignment_max_size=5 del_percentile_frac=0.2]

The search is Vecalign's own code, included in the package (``_vecalign.py`` and the compiled ``_vecalign_core``,
taken from the Vecalign repository; without a C compiler at installation a slower Python version of the compiled
part is used): cost of a bead
``2 * nx * ny * (1 - cos) / (norm_x + norm_y)`` with norms from random samples of the other document, a deletion cost
set to a percentile of the costs of random sentence pairs, a full 1-1 search on down-sampled vectors and a search in a
corridor around that path at each finer level.  It runs on the vectors of the SentAlign scorer (LaBSE by default;
Vecalign's settings were chosen for LASER), so it needs a scorer with vectors.  Its output is the alignment: beads
with sentences on both sides become links, the rest is unaligned; the score cutoff, the re-evaluation and the
absorption of SentAlign are not applied.  The score column is the scorer's cosine of the link, not Vecalign's cost.

What is the same as the original command line and what is not:

* Same: with the same vectors, the first (or only) document of a run gets the alignment of ``vecalign`` for that
  document.  The padding vector is that of the text ``PAD``, as in the original (it only enters the random samples).
* The random generator is seeded (``seed``, 42) before **every** document.  The original seeds once per process, so
  there the result of a document depends on the documents aligned before it.
* ``alignment_max_size`` A means links of x + y <= A sentences, and the original then builds vectors for spans of up
  to A sentences, which all enter the random samples.  Here spans go up to ``--max-concatenations`` (4): with A <= 4
  nothing differs; with A = 5 the samples are drawn from spans of 1-4 instead of 1-5 sentences.  A > 5 needs a larger
  ``--max-concatenations``.
* Anchors are not used (``--anchors auto``): Vecalign needs time and memory linear in the document length.

Vecalign is Copyright 2019 Brian Thompson, Apache License 2.0.
"""
import threading
from math import ceil

import numpy as np

from sentalign.pathfinding.base import LINK, PathFinder

_RANDOM_LOCK = threading.Lock()         # the original draws from numpy's global random generator


def end_indexed(vectors, n, overlaps, pad):
    """The array Vecalign expects: ``out[k-1, j]`` is the vector of the k sentences *ending* at sentence j, and the
    padding vector where fewer than k sentences end there.  ``vectors(k)`` gives the spans *starting* at each
    sentence.  A copy: Vecalign normalises its input in place."""
    first = vectors(1)
    if first is None:
        raise ValueError('the vecalign path finder needs a scorer with vectors (an embedding scorer)')
    out = np.empty((overlaps, n, first.shape[1]), dtype=np.float32)
    for k in range(1, overlaps + 1):
        head = min(k - 1, n)
        out[k - 1, :head] = pad if pad is not None else first[:head]
        if n >= k:
            out[k - 1, k - 1:] = vectors(k)
    return out


class VecalignFinder(PathFinder):
    name = 'vecalign'
    postprocess = False
    uses_anchors = False
    needs_vectors = True
    option_defaults = {
        'alignment_max_size': 4,        # links of x + y <= this many sentences (the paper used 6, the README 8)
        'del_percentile_frac': 0.2,     # deletion cost: this percentile of the costs of random sentence pairs
        'search_buffer_size': 5,        # half width of the corridor, on top of half the largest link
        'max_size_full_dp': 300,        # documents are halved until n * m <= this squared
        'costs_sample_size': 20000,
        'num_samps_for_norm': 100,
        'seed': 42,
    }

    def find(self, segment):
        from sentalign.pathfinding._vecalign import make_alignment_types, vecalign
        if segment.n_src == 0 or segment.n_trg == 0:
            return []
        if segment.src_vectors is None or segment.trg_vectors is None:
            raise ValueError('the vecalign path finder needs a scorer with vectors (an embedding scorer)')
        opt = self.options
        size = max(2, int(opt['alignment_max_size']))
        if size - 1 > segment.max_k:
            raise ValueError('alignment_max_size %d needs spans of %d sentences: set --max-concatenations %d'
                             % (size, size - 1, size - 1))
        overlaps = min(size, segment.max_k)
        pads = [segment.text_vector(side, 'PAD') if segment.text_vector is not None else None
                for side in ('src', 'trg')]
        vecs0 = end_indexed(segment.src_vectors, segment.n_src, overlaps, pads[0])
        vecs1 = end_indexed(segment.trg_vectors, segment.n_trg, overlaps, pads[1])
        with _RANDOM_LOCK:
            np.random.seed(int(opt['seed']))
            stack = vecalign(vecs0=vecs0, vecs1=vecs1,
                             final_alignment_types=make_alignment_types(size),
                             del_percentile_frac=float(opt['del_percentile_frac']),
                             width_over2=ceil(size / 2.0) + int(opt['search_buffer_size']),
                             max_size_full_dp=int(opt['max_size_full_dp']),
                             costs_sample_size=int(opt['costs_sample_size']),
                             num_samps_for_norm=int(opt['num_samps_for_norm']))
        return [(LINK, x[0], len(x), y[0], len(y)) for x, y in stack[0]['final_alignments'] if len(x) and len(y)]
