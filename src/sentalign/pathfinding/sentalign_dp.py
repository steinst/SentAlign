"""The SentAlign path search (the algorithm of the 2023 paper) as a ``PathFinder``.

The work is done by a kernel (``_dp_cython`` compiled, ``_dp_python`` reference; same results).  This module feeds
the kernel band by band with score blocks from the segment and reads the path back from the back-pointers.
"""
import logging

import numpy as np

from sentalign.pathfinding.base import LINK, SKIP_SOURCE, SKIP_TARGET, PathFinder
from sentalign.pathfinding import _dp_python

log = logging.getLogger('sentalign.pathfinding')


def get_kernel_class(backend='auto'):
    if backend in ('auto', 'cython'):
        try:
            from sentalign.pathfinding._dp_cython import DPKernelCython
            return DPKernelCython
        except ImportError:
            if backend == 'cython':
                raise
            log.warning('the compiled path-search kernel is not available, using the slow Python reference')
    elif backend == 'rust':
        try:
            from sentalign_rust import DPKernelRust
        except ImportError as e:
            raise ImportError('--path-backend rust needs the package sentalign-rust (pip install sentalign-rust, or '
                              '"maturin develop --release" in the rust/ folder of the repository): %s' % e) from e
        return DPKernelRust
    elif backend != 'python':
        raise ValueError('unknown path-search backend %r (auto, cython, rust, python)' % backend)
    return _dp_python.DPKernelPython


class SentAlignDP(PathFinder):
    name = 'sentalign-dp'

    def find(self, segment, stop_node=None):
        """Best path through the segment.  `stop_node` (paper-2023 only) reproduces a fault of the 2023 code:
        its trace-back stopped early when it came to the node whose segment-relative position happened to equal
        the absolute position of the segment start."""
        bp = self.back_pointers(segment)
        return trace_back(bp, segment.max_k, stop_node)

    def back_pointers(self, segment):
        p = self.params
        K = segment.max_k
        Sn, Tn = segment.n_src + 1, segment.n_trg + 1          # nodes: 0 .. number of sentences
        extra = {'threads': max(1, p.threads)} if p.backend == 'rust' else {}
        squared = p.reward == 'squared-cosine'
        if squared:
            # Reward of a link: score ** 2 (times the length prior); a skipped sentence earns nothing; no length
            # penalty.  The kernels are used as they are: with more free concatenations than a link can have they
            # pay (ks + kt) * value for a link and the "cutoff" for a skip, so they get the cutoff 0 and the value
            # score ** 2 / (ks + kt), with NaN (no link) where the score is below the real cutoff.
            if p.paper_arithmetic:
                raise ValueError('the squared-cosine reward is not available with the 2023 arithmetic')
            kernel = get_kernel_class(p.backend)(Sn, Tn, K, 0.0, p.minimum_length_words, p.maximum_length_words,
                                                 2 ** 30, 0.0, 2 * K + 1, False, **extra)
            prior = _LengthPrior(segment) if p.length_prior else None
        else:
            kernel = get_kernel_class(p.backend)(Sn, Tn, K, p.score_cutoff, p.minimum_length_words,
                                                 p.maximum_length_words, p.penalty_after_words, p.penalty_per_word,
                                                 p.free_concatenations, p.paper_arithmetic, **extra)
        # words of the span of ks sentences ENDING in node a (0 where there is no such span)
        slen = np.zeros((K, Sn), dtype=np.int32)
        tlen = np.zeros((K, Tn), dtype=np.int32)
        for k in range(1, K + 1):
            if k <= segment.n_src:
                slen[k - 1, k:] = segment.src_words[k - 1]
            if k <= segment.n_trg:
                tlen[k - 1, k:] = segment.trg_words[k - 1]
        rows = max(1, int(p.band_bytes // (K * K * Tn * 4)))
        for a0 in range(0, Sn, rows):
            a1 = min(Sn, a0 + rows)
            # NaN where no link (ks, kt) can end: the first ks rows and the first kt columns of the segment
            band = np.empty((K, K, a1 - a0, Tn), dtype=np.float32)
            for ks in range(1, K + 1):
                lo = max(a0, ks)                                # first node of the band that a ks-span can end in
                for kt in range(1, K + 1):
                    plane = band[ks - 1, kt - 1]
                    if lo >= a1 or kt > segment.n_trg:
                        plane[...] = np.nan
                        continue
                    plane[:lo - a0, :] = np.nan
                    plane[:, :kt] = np.nan
                    if segment.takes_out:
                        segment.score_block(ks, kt, lo - ks, a1 - ks, out=plane[lo - a0:, kt:])
                    else:
                        plane[lo - a0:, kt:] = segment.score_block(ks, kt, lo - ks, a1 - ks)
                    if squared:
                        part = plane[lo - a0:, kt:]
                        with np.errstate(invalid='ignore'):
                            below = ~(part >= p.score_cutoff)             # also True where there is no score
                        np.multiply(part, part, out=part)
                        if prior is not None:
                            part *= prior.block(ks, kt, lo - ks, a1 - ks)
                        part /= ks + kt
                        part[below] = np.nan
            kernel.run_band(a0, a1, band, slen, tlen)
        self.last_score = kernel.final_score()
        return kernel.bp_array


class _LengthPrior:
    """The length-ratio prior of ChainAlign for blocks of span pairs: sqrt(log2(1 + shorter / longer)), lengths in
    characters, the target length scaled by the character ratio of the two texts of the segment."""

    def __init__(self, segment):
        if segment.src_sentences is None or segment.trg_sentences is None:
            raise ValueError('the length prior needs the sentence texts in the segment')
        self.src = np.concatenate(([0], np.cumsum([len(s.strip()) for s in segment.src_sentences]))).astype(np.float64)
        self.trg = np.concatenate(([0], np.cumsum([len(s.strip()) for s in segment.trg_sentences]))).astype(np.float64)
        self.ratio = self.src[-1] / self.trg[-1] if self.trg[-1] else 1.0

    def block(self, ks, kt, p0, p1):
        """float32 (p1 - p0, n_trg - kt + 1) for the ks-sentence source spans starting at p0 .. p1-1."""
        cs = (self.src[p0 + ks:p1 + ks] - self.src[p0:p1])[:, None]
        ct = ((self.trg[kt:] - self.trg[:len(self.trg) - kt]) * self.ratio)[None, :]
        hi = np.maximum(cs, ct)
        ratio = np.divide(np.minimum(cs, ct), hi, out=np.zeros_like(hi), where=hi > 0)
        return np.sqrt(np.log2(1.0 + ratio)).astype(np.float32)


def trace_back(bp, K, stop_node=None):
    """Steps (kind, src_start, src_size, trg_start, trg_size) from the back-pointer array, first step first."""
    a, b = bp.shape[0] - 1, bp.shape[1] - 1
    steps = []
    while True:
        code = int(bp[a, b])
        stop = stop_node is not None and (a, b) == tuple(stop_node)
        if code == 0:
            break
        if code == 1:
            steps.append((SKIP_SOURCE, a - 1, 1, b, 0))
            a -= 1
        elif code == 2:
            steps.append((SKIP_TARGET, a, 0, b - 1, 1))
            b -= 1
        else:
            ks, kt = (code - 3) // K + 1, (code - 3) % K + 1
            steps.append((LINK, a - ks, ks, b - kt, kt))
            a -= ks
            b -= kt
        if stop:
            break
    steps.reverse()
    return steps
