"""Rust path-search kernel for SentAlign: ``sentalign ... --path-backend rust``.

``DPKernelRust`` has the interface of the other kernels (``sentalign.pathfinding._dp_python`` is the readable
reference) and gives bit-for-bit the same results; with ``threads > 1`` it computes the tiles of each anti-diagonal
in parallel.  The arrays live here, in NumPy; the Rust function gets their addresses.
"""
import numpy as np

from sentalign_rust import _core

__version__ = "2.0.0"


class DPKernelRust:
    def __init__(self, n_rows, n_cols, K, cutoff, minw, maxw, penw, ppw, free, acc32, threads=1):
        self.Sn, self.Tn, self.K = int(n_rows), int(n_cols), int(K)
        self._params = (float(cutoff), int(minw), int(maxw), int(penw), float(ppw), int(free), bool(acc32))
        self.threads = max(1, int(threads))
        self.bp_array = np.zeros((self.Sn, self.Tn), dtype=np.uint8)
        # the last K rows of path sums of the band before (zeros before the first band)
        self._carry_best = np.zeros((self.K, self.Tn), dtype=np.float64)
        self._carry_is32 = np.zeros((self.K, self.Tn), dtype=np.uint8)

    def run_band(self, a0, a1, band, slen, tlen):
        K, Sn, Tn = self.K, self.Sn, self.Tn
        rows = a1 - a0
        band = np.ascontiguousarray(band, dtype=np.float32)
        slen = np.ascontiguousarray(slen, dtype=np.int32)
        tlen = np.ascontiguousarray(tlen, dtype=np.int32)
        if not (0 <= a0 <= a1 <= Sn) or band.shape != (K, K, rows, Tn) or slen.shape != (K, Sn) or tlen.shape != (K, Tn):
            raise ValueError('run_band: array shapes do not fit the kernel')
        best = np.empty((K + rows, Tn), dtype=np.float64)
        is32 = np.empty((K + rows, Tn), dtype=np.uint8)
        best[:K] = self._carry_best
        is32[:K] = self._carry_is32
        _core.run_band(a0, a1, Sn, Tn, K, band.ctypes.data, slen.ctypes.data, tlen.ctypes.data,
                       self.bp_array.ctypes.data, best.ctypes.data, is32.ctypes.data, *self._params, self.threads)
        # keep the last K rows for the next band (a band shorter than K rows keeps some of the old ones)
        self._carry_best = np.ascontiguousarray(best[-K:])
        self._carry_is32 = np.ascontiguousarray(is32[-K:])

    def final_score(self):
        return float(self._carry_best[-1, self.Tn - 1])
