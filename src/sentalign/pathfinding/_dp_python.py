"""The SentAlign path search, reference implementation (readable, slow).

The alignment graph
-------------------
Nodes are pairs (a, b): "the first a source sentences and the first b target sentences of this segment are done".
From a node the path can

* skip one source sentence   (a-1, b) -> (a, b)      reward: the score cutoff (0.4)
* skip one target sentence   (a, b-1) -> (a, b)      reward: the score cutoff
* link ks source sentences to kt target sentences  (a-ks, b-kt) -> (a, b),  1 <= ks, kt <= K (4)
      reward:  (ks + kt) * (score - length penalty) ** (ks + kt - free + 1)        (free = 2: a 1-1 link gets
      2 * score, a 1-2 link 3 * score**2, ...; the power is what keeps long merges from winning)
      only if score >= cutoff and both sides have between `minw` and `maxw` words.

The best path from (0, 0) to the last node maximises the summed reward.  This is a dynamic programme over the grid,
row by row; every node looks at up to K*K links.  Nodes are processed in bands of rows so that the scores of only a
part of the segment are in memory at a time.

Two arithmetic modes
--------------------
``acc32=False`` (version 2): plain double precision; a link whose score is unknown (NaN) is never taken.
``acc32=True``  (paper-2023): exactly the arithmetic of the 2023 code under NumPy 2, where the scores are
float32 numbers and a sum becomes float32 as soon as one link is on the path, while a path of skips only stays a
Python float.  This module then simply uses NumPy float32 scalars and Python floats as the 2023 code did; the
compiled kernel (_dp_cython.pyx) emulates the same and is tested against this one.
"""
import math

import numpy as np

START, SKIP_SOURCE, SKIP_TARGET, LINK = 0, 1, 2, 3     # back-pointer codes; a link (ks, kt) is LINK + (ks-1)*K + (kt-1)


class DPKernelPython:
    def __init__(self, n_rows, n_cols, K, cutoff, minw, maxw, penw, ppw, free, acc32):
        if K > 15:                           # a back-pointer is one uint8: 3 + (ks-1)*K + (kt-1) must stay below 256
            raise ValueError('max_concatenations %d is too large: the path search supports at most 15' % K)
        self.Sn, self.Tn, self.K = n_rows, n_cols, K
        self.cutoff, self.minw, self.maxw, self.penw, self.ppw, self.free = cutoff, minw, maxw, penw, ppw, free
        self.acc32 = acc32
        self.bp_array = np.zeros((n_rows, n_cols), dtype=np.uint8)
        self.best = {}                       # row number -> list of path sums (the last K+1 rows are kept)

    # -- reward of one link ------------------------------------------------------------------------------------
    def _link(self, score, ks, kt, src_words, trg_words, before):
        """Path sum after the link, or None if the link is not allowed."""
        if not (self.minw <= src_words <= self.maxw and self.minw <= trg_words <= self.maxw):
            return None
        if self.acc32:
            if math.isnan(score):
                score = 0                   # 2023: no embedding found for the text -> the integer 0
            if not (self.cutoff <= score):
                return None
            penalty = 0
            if src_words > self.penw:
                penalty += (src_words - self.penw) * self.ppw
            if trg_words > self.penw:
                penalty += (trg_words - self.penw) * self.ppw
            n = ks + kt
            if n - self.free >= 0:
                return (((score - penalty) ** (n - self.free + 1)) * n) + before
            return ((score - penalty) * n) + before
        if math.isnan(score) or not (self.cutoff <= float(score)):
            return None
        penalty = 0.0
        if src_words > self.penw:
            penalty += (src_words - self.penw) * self.ppw
        if trg_words > self.penw:
            penalty += (trg_words - self.penw) * self.ppw
        n = ks + kt
        x = float(score) - penalty
        value = x
        if n - self.free >= 0:
            for _ in range(n - self.free):        # x ** (n - free + 1) by repeated multiplication
                value *= x
        return value * n + before

    # -- one band of rows ---------------------------------------------------------------------------------------
    def run_band(self, a0, a1, band, slen, tlen):
        """Rows a0 <= a < a1.  band[ks-1, kt-1, a-a0, b]: score of the link (ks, kt) ending in node (a, b);
        slen[ks-1, a] / tlen[kt-1, b]: words of the ks source / kt target sentences ending there."""
        K, Tn, cutoff = self.K, self.Tn, self.cutoff
        for a in range(a0, a1):
            row = [0] * Tn
            self.best[a] = row
            self.best.pop(a - K - 1, None)
            up_row = self.best.get(a - 1)
            for b in range(Tn):
                if a == 0 and b == 0:
                    value, code = 0, START
                elif a == 0:
                    value, code = row[b - 1] + cutoff, SKIP_TARGET
                elif b == 0:
                    value, code = up_row[0] + cutoff, SKIP_SOURCE
                elif row[b - 1] < up_row[b]:
                    value, code = up_row[b] + cutoff, SKIP_SOURCE
                else:
                    value, code = row[b - 1] + cutoff, SKIP_TARGET
                best = value
                for ks in range(min(K, a), 0, -1):                 # longest first, as the 2023 code
                    for kt in range(min(K, b), 0, -1):
                        cand = self._link(band[ks - 1, kt - 1, a - a0, b], ks, kt, int(slen[ks - 1, a]),
                                          int(tlen[kt - 1, b]), self.best[a - ks][b - kt])
                        if cand is None:
                            if not self.acc32:
                                continue
                            cand = 0          # 2023: a link that is not allowed counts as the path sum 0
                        if cand >= best:      # ties go to the later (shorter) link, as the 2023 code
                            best = value = cand
                            code = LINK + (ks - 1) * K + (kt - 1)
                row[b] = value
                self.bp_array[a, b] = code

    def final_score(self):
        return float(self.best[self.Sn - 1][self.Tn - 1])
