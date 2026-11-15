# cython: language_level=3, boundscheck=False, wraparound=False, cdivision=True, initializedcheck=False
"""The SentAlign path search, compiled kernel.  Same algorithm and same results as _dp_python.py (the tests
compare the two); see that module for the description.

acc32 mode reproduces the arithmetic of the 2023 code under NumPy 2: a path sum is either a Python float (double;
a path that so far only skipped sentences) or a NumPy float32 (as soon as a link is on the path).  Mixed operations
are done in float32 with the double cast to float32 (NumPy's rule for Python scalars), float32 ** int is libm powf
(checked: identical to NumPy for every exponent used).  A sum is therefore kept as a double plus a flag "is float32".
"""
import numpy as np
cimport numpy as cnp
from libc.math cimport powf, isnan

cnp.import_array()

cdef enum:
    START = 0
    SKIP_SOURCE = 1
    SKIP_TARGET = 2
    LINK = 3


cdef class DPKernelCython:
    cdef public int Sn, Tn, K
    cdef double cutoff, ppw
    cdef int minw, maxw, penw, free
    cdef bint acc32
    cdef public object bp_array
    cdef cnp.uint8_t[:, ::1] bp
    cdef double[:, ::1] best          # ring buffer of the last K+1 rows
    cdef cnp.uint8_t[:, ::1] is32

    def __init__(self, int n_rows, int n_cols, int K, double cutoff, int minw, int maxw, int penw, double ppw,
                 int free, bint acc32):
        if K > 15:                           # a back-pointer is one uint8: 3 + (ks-1)*K + (kt-1) must stay below 256
            raise ValueError('max_concatenations %d is too large: the path search supports at most 15' % K)
        self.Sn, self.Tn, self.K = n_rows, n_cols, K
        self.cutoff, self.minw, self.maxw, self.penw, self.ppw, self.free = cutoff, minw, maxw, penw, ppw, free
        self.acc32 = acc32
        self.bp_array = np.zeros((n_rows, n_cols), dtype=np.uint8)
        self.bp = self.bp_array
        self.best = np.zeros((K + 1, n_cols), dtype=np.float64)
        self.is32 = np.zeros((K + 1, n_cols), dtype=np.uint8)

    def run_band(self, int a0, int a1, float[:, :, :, ::1] band, int[:, ::1] slen, int[:, ::1] tlen):
        if not (0 <= a0 <= a1 <= self.Sn) or band.shape[0] != self.K or band.shape[1] != self.K \
                or band.shape[2] < a1 - a0 or band.shape[3] != self.Tn or slen.shape[0] != self.K \
                or slen.shape[1] != self.Sn or tlen.shape[0] != self.K or tlen.shape[1] != self.Tn:
            raise ValueError('run_band: array shapes do not fit the kernel')
        with nogil:
            self._run(a0, a1, band, slen, tlen)

    cdef void _run(self, int a0, int a1, float[:, :, :, ::1] band, int[:, ::1] slen, int[:, ::1] tlen) noexcept nogil:
        cdef int K = self.K, Tn = self.Tn, R = self.K + 1
        cdef int minw = self.minw, maxw = self.maxw, penw = self.penw, free = self.free
        cdef double cutoff = self.cutoff, ppw = self.ppw
        cdef float cutoff_f = <float> cutoff
        cdef bint acc32 = self.acc32
        cdef int a, b, ks, kt, ra, rup, rn, sl, tl, n, e, ksmax, ktmax
        cdef double v, maxv, cv, lv, uv, nv, pen, x64, p64
        cdef bint f, maxf, cf, lf, uf, nf, less, ge, valid
        cdef float l, x, c
        cdef cnp.uint8_t code

        for a in range(a0, a1):
            ra = a % R
            rup = (a - 1 + R) % R
            for b in range(Tn):
                # ---- skip a sentence ----
                if a == 0 and b == 0:
                    v = 0.0; f = 0; code = START
                elif a == 0 or (b != 0 and not self._less(self.best[ra, b - 1], self.is32[ra, b - 1],
                                                           self.best[rup, b], self.is32[rup, b])):
                    lv = self.best[ra, b - 1]; lf = self.is32[ra, b - 1]
                    if lf:
                        v = <double> (<float> lv + cutoff_f); f = 1
                    else:
                        v = lv + cutoff; f = 0
                    code = SKIP_TARGET
                else:
                    uv = self.best[rup, b]; uf = self.is32[rup, b]
                    if uf:
                        v = <double> (<float> uv + cutoff_f); f = 1
                    else:
                        v = uv + cutoff; f = 0
                    code = SKIP_SOURCE
                maxv = v; maxf = f
                # ---- links, longest first ----
                ksmax = K if a > K else a
                ktmax = K if b > K else b
                for ks in range(ksmax, 0, -1):
                    sl = slen[ks - 1, a]
                    rn = (a - ks) % R
                    for kt in range(ktmax, 0, -1):
                        tl = tlen[kt - 1, b]
                        l = band[ks - 1, kt - 1, a - a0, b]
                        nv = self.best[rn, b - kt]; nf = self.is32[rn, b - kt]
                        n = ks + kt
                        valid = minw <= sl and sl <= maxw and minw <= tl and tl <= maxw
                        if acc32:
                            if isnan(l):
                                l = 0.0
                                valid = valid and cutoff <= 0.0
                            else:
                                valid = valid and cutoff_f <= l
                            if valid:
                                pen = 0.0
                                if sl > penw:
                                    pen += (sl - penw) * ppw
                                if tl > penw:
                                    pen += (tl - penw) * ppw
                                x = l - <float> pen
                                if n - free >= 0:
                                    c = powf(x, <float> (n - free + 1)) * <float> n + <float> nv
                                else:
                                    c = x * <float> n + <float> nv
                                cv = <double> c; cf = 1
                            else:
                                cv = 0.0; cf = 0        # 2023: a link that is not allowed counts as the sum 0
                            if cf or maxf:
                                ge = <float> cv >= <float> maxv
                            else:
                                ge = cv >= maxv
                        else:
                            if isnan(l) or not valid or not (cutoff <= <double> l):
                                continue
                            pen = 0.0
                            if sl > penw:
                                pen += (sl - penw) * ppw
                            if tl > penw:
                                pen += (tl - penw) * ppw
                            x64 = <double> l - pen
                            p64 = x64
                            if n - free >= 0:
                                for e in range(n - free):
                                    p64 = p64 * x64
                            cv = p64 * n + nv; cf = 0
                            ge = cv >= maxv
                        if ge:
                            maxv = cv; maxf = cf
                            v = cv; f = cf
                            code = LINK + (ks - 1) * K + (kt - 1)
                self.best[ra, b] = v
                self.is32[ra, b] = f
                self.bp[a, b] = code

    cdef inline bint _less(self, double lv, bint lf, double uv, bint uf) noexcept nogil:
        """left < up, with NumPy's mixed float32 / Python float comparison."""
        if lf or uf:
            return <float> lv < <float> uv
        return lv < uv

    def final_score(self):
        return float(self.best[(self.Sn - 1) % (self.K + 1), self.Tn - 1])
