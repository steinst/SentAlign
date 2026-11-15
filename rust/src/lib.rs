//! The SentAlign path search in Rust.
//!
//! Same algorithm and bit-for-bit the same results as `sentalign/pathfinding/_dp_python.py` (the readable
//! reference, where the graph and the link reward are described) and `_dp_cython.pyx`.  What this kernel adds is
//! the option to work on several cores: the band of rows is cut into tiles, and the tiles on one anti-diagonal do
//! not depend on each other (a node only looks at nodes above and to the left, at most K rows and columns away),
//! so they are computed in parallel, diagonal after diagonal.
//!
//! All arrays are owned by the Python side (`sentalign_rust.DPKernelRust`), which passes their addresses and
//! guarantees their sizes, element types and C layout for the duration of the call:
//!   band  f32  [K][K][rows][tn]   score of the link (ks, kt) ending in node (a0 + r, b); NaN = no score
//!   slen  i32  [K][sn]            words of the ks source sentences ending in node a (same for tlen, [K][tn])
//!   bp    u8   [sn][tn]           back-pointers (output)
//!   best  f64  [K + rows][tn]     path sums: K rows carried over from the band before, then this band (in/out)
//!   is32  u8   [K + rows][tn]     paper-2023 arithmetic: the sum is a float32 (see _dp_cython.pyx)

use pyo3::prelude::*;
use rayon::prelude::*;

const START: u8 = 0;
const SKIP_SOURCE: u8 = 1;
const SKIP_TARGET: u8 = 2;
const LINK: u8 = 3;

#[derive(Clone, Copy)]
struct Kernel {
    a0: usize,
    sn: usize,
    tn: usize,
    k: usize,
    rows: usize,
    cutoff: f64,
    cutoff_f: f32,
    minw: i32,
    maxw: i32,
    penw: i32,
    ppw: f64,
    free: i32,
    acc32: bool,
    band: *const f32,
    slen: *const i32,
    tlen: *const i32,
    bp: *mut u8,
    best: *mut f64,
    is32: *mut u8,
}

// The tiles of one anti-diagonal write disjoint parts of bp / best / is32 and only read parts that were finished on
// earlier diagonals, so sharing the raw pointers between threads is sound.
unsafe impl Send for Kernel {}
unsafe impl Sync for Kernel {}

impl Kernel {
    #[inline(always)]
    fn at(&self, a: usize, b: usize) -> usize {
        (a + self.k - self.a0) * self.tn + b
    }

    /// left < up with NumPy's rule for float32 against a Python float: compare as float32.
    #[inline(always)]
    fn less(lv: f64, lf: bool, uv: f64, uf: bool) -> bool {
        if lf || uf {
            (lv as f32) < (uv as f32)
        } else {
            lv < uv
        }
    }

    #[inline(always)]
    unsafe fn cell(&self, a: usize, b: usize) {
        let k = self.k;
        let tn = self.tn;
        let (mut v, mut f, mut code);
        // ---- skip a sentence ----
        if a == 0 && b == 0 {
            v = 0.0f64;
            f = false;
            code = START;
        } else {
            let from_left = if a == 0 {
                true
            } else if b == 0 {
                false
            } else {
                let l = self.at(a, b - 1);
                let u = self.at(a - 1, b);
                !Self::less(*self.best.add(l), *self.is32.add(l) != 0, *self.best.add(u), *self.is32.add(u) != 0)
            };
            let p = if from_left { self.at(a, b - 1) } else { self.at(a - 1, b) };
            let pv = *self.best.add(p);
            if *self.is32.add(p) != 0 {
                v = ((pv as f32) + self.cutoff_f) as f64;
                f = true;
            } else {
                v = pv + self.cutoff;
                f = false;
            }
            code = if from_left { SKIP_TARGET } else { SKIP_SOURCE };
        }
        let mut maxv = v;
        let mut maxf = f;
        // ---- links, longest first ----
        let ksmax = if a > k { k } else { a };
        let ktmax = if b > k { k } else { b };
        let band_row = (a - self.a0) * tn + b;
        let plane = self.rows * tn;
        let mut ks = ksmax;
        while ks >= 1 {
            let sl = *self.slen.add((ks - 1) * self.sn + a);
            let mut kt = ktmax;
            while kt >= 1 {
                let tl = *self.tlen.add((kt - 1) * tn + b);
                let mut l = *self.band.add(((ks - 1) * k + (kt - 1)) * plane + band_row);
                let q = self.at(a - ks, b - kt);
                let nv = *self.best.add(q);
                let n = (ks + kt) as i32;
                let mut valid = self.minw <= sl && sl <= self.maxw && self.minw <= tl && tl <= self.maxw;
                let (cv, cf, ge);
                if self.acc32 {
                    if l.is_nan() {
                        l = 0.0;
                        valid = valid && self.cutoff <= 0.0;
                    } else {
                        valid = valid && self.cutoff_f <= l;
                    }
                    if valid {
                        let mut pen = 0.0f64;
                        if sl > self.penw {
                            pen += (sl - self.penw) as f64 * self.ppw;
                        }
                        if tl > self.penw {
                            pen += (tl - self.penw) as f64 * self.ppw;
                        }
                        let x: f32 = l - (pen as f32);
                        let c: f32 = if n - self.free >= 0 {
                            x.powf((n - self.free + 1) as f32) * (n as f32) + (nv as f32)
                        } else {
                            x * (n as f32) + (nv as f32)
                        };
                        cv = c as f64;
                        cf = true;
                    } else {
                        cv = 0.0; // 2023: a link that is not allowed counts as the sum 0
                        cf = false;
                    }
                    ge = if cf || maxf { (cv as f32) >= (maxv as f32) } else { cv >= maxv };
                } else {
                    if l.is_nan() || !valid || !(self.cutoff <= l as f64) {
                        kt -= 1;
                        continue;
                    }
                    let mut pen = 0.0f64;
                    if sl > self.penw {
                        pen += (sl - self.penw) as f64 * self.ppw;
                    }
                    if tl > self.penw {
                        pen += (tl - self.penw) as f64 * self.ppw;
                    }
                    let x = l as f64 - pen;
                    let mut p = x;
                    if n - self.free >= 0 {
                        for _ in 0..(n - self.free) {
                            p *= x;
                        }
                    }
                    cv = p * (n as f64) + nv;
                    cf = false;
                    ge = cv >= maxv;
                }
                if ge {
                    maxv = cv;
                    maxf = cf;
                    v = cv;
                    f = cf;
                    code = LINK + ((ks - 1) * k + (kt - 1)) as u8;
                }
                kt -= 1;
            }
            ks -= 1;
        }
        let o = self.at(a, b);
        *self.best.add(o) = v;
        *self.is32.add(o) = f as u8;
        *self.bp.add(a * tn + b) = code;
    }

    unsafe fn tile(&self, r0: usize, r1: usize, c0: usize, c1: usize) {
        for a in r0..r1 {
            for b in c0..c1 {
                self.cell(a, b);
            }
        }
    }

    fn run(&self, a1: usize, threads: usize) {
        if threads <= 1 {
            unsafe { self.tile(self.a0, a1, 0, self.tn) };
            return;
        }
        // tiles at least K wide and high, so that a node only reaches into the neighbouring tiles
        // (a band of a very wide document has few rows: the tiles are made lower so that a diagonal still has
        // about as many tiles as there are threads)
        let tile_rows = ((a1 - self.a0) / threads).clamp(self.k, 48);
        let tile_cols = self.k.max(self.tn.div_ceil(4 * threads)).max(64);
        let nr = (a1 - self.a0).div_ceil(tile_rows);
        let nc = self.tn.div_ceil(tile_cols);
        let pool = rayon::ThreadPoolBuilder::new().num_threads(threads).build().expect("thread pool");
        pool.install(|| {
            for d in 0..(nr + nc - 1) {
                let lo = d.saturating_sub(nc - 1);
                let hi = d.min(nr - 1);
                (lo..=hi).into_par_iter().for_each(|r| {
                    let c = d - r;
                    let r0 = self.a0 + r * tile_rows;
                    let c0 = c * tile_cols;
                    unsafe { self.tile(r0, (r0 + tile_rows).min(a1), c0, (c0 + tile_cols).min(self.tn)) };
                });
            }
        });
    }
}

/// Rows a0 <= a < a1 of the path search.  See the module documentation for the arrays; the Python wrapper is the
/// only intended caller.
#[pyfunction]
#[pyo3(signature = (a0, a1, sn, tn, k, band, slen, tlen, bp, best, is32, cutoff, minw, maxw, penw, ppw, free, acc32, threads))]
#[allow(clippy::too_many_arguments)]
fn run_band(
    py: Python<'_>,
    a0: usize,
    a1: usize,
    sn: usize,
    tn: usize,
    k: usize,
    band: usize,
    slen: usize,
    tlen: usize,
    bp: usize,
    best: usize,
    is32: usize,
    cutoff: f64,
    minw: i32,
    maxw: i32,
    penw: i32,
    ppw: f64,
    free: i32,
    acc32: bool,
    threads: usize,
) -> PyResult<()> {
    if !(a0 <= a1 && a1 <= sn) || k == 0 || k > 15 || tn == 0 {
        return Err(pyo3::exceptions::PyValueError::new_err("run_band: sizes do not fit"));
    }
    let kernel = Kernel {
        a0,
        sn,
        tn,
        k,
        rows: a1 - a0,
        cutoff,
        cutoff_f: cutoff as f32,
        minw,
        maxw,
        penw,
        ppw,
        free,
        acc32,
        band: band as *const f32,
        slen: slen as *const i32,
        tlen: tlen as *const i32,
        bp: bp as *mut u8,
        best: best as *mut f64,
        is32: is32 as *mut u8,
    };
    py.detach(|| kernel.run(a1, threads));
    Ok(())
}

#[pymodule]
fn _core(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_function(wrap_pyfunction!(run_band, m)?)?;
    Ok(())
}
