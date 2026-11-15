"""``chain``: seed, chain, extend.  A re-implementation after ChainAlign (Goldner and Gramazio, WMT 2026,
"ChainAlign: Cross-Lingual Sentence Alignment via Sparse Co-linear Chaining"), written from the paper's algorithms
2-4.  The authors' code is not public, so this cannot be shown to be identical to theirs; where the paper leaves a
detail open the choice made here is listed below.

    sentalign ... --pathfinder chain [--pathfinder-opt band=0 skip_cost=-0.1]

1. **Seed.**  For every link type (k source sentences, l target sentences, k + l <= 5: ten types) the scores of all
   span pairs; per source span the ``top_k`` (10) best cells with score >= ``threshold`` (0.3) inside the diagonal
   band ``|i / N_k - j / M_l| < band`` (0.15) become candidate edges.
2. **Chain.**  An edge weighs ``score ** 2 * sqrt(log2(1 + min(c_s, c_t r) / max(c_s, c_t r)))``, where c_s and c_t
   are the character lengths of the two spans and r the character ratio of the two documents.  The set of edges
   with the largest total weight in which no two edges overlap or cross is selected (weighted longest increasing
   subsequence).  Nothing is paid for sentences left out, and nothing forces an edge in.
3. **Extend.**  Each gap between two consecutive chain edges that has sentences on both sides (also before the
   first and after the last edge) is aligned by a small dynamic programme with the same link types, link score
   ``score - (row mean + column mean) / 2`` (means over the gap's score matrix of that type) and ``skip_cost``
   (-0.5) per skipped sentence.  Chain edges are never changed.

The result is the alignment: no score cutoff of SentAlign, no re-evaluation, no absorption.  Anchors are not used
(the band and the character ratio are defined on the whole document).  For text that is not parallel throughout
(abridged or partial translations) the paper uses ``band=0`` (no band) together with ``skip_cost=-0.1``; the length
prior also assumes a complete translation (``length_prior=false`` switches it off).

Choices where the paper is silent or ambiguous:
* ties: among equal scores the cell that numpy's partition keeps; in the chain the earlier edge (source start, then
  target start, then the order of the types) and the predecessor with the smallest target end; in the gaps the first
  link type in the paper's order, then skipping a source sentence, then a target sentence;
* the means in the gap score are taken over the gap (algorithm 4), not over the whole document;
* character lengths are those of the stripped sentences, without the joining spaces;
* top-k is taken per source span, whichever document is longer;
* a gap of more than ``max_gap_cells`` (10 million) node pairs is left unaligned, with a warning (the paper's gaps
  are a few sentences; a book with a long omitted part has much larger ones).
"""
import logging

import numpy as np

from sentalign.pathfinding.base import LINK, PathFinder

log = logging.getLogger('sentalign.chain')

# the paper's order of the link types (section 3.1)
TYPES = [(1, 1), (2, 1), (1, 2), (3, 1), (1, 3), (2, 2), (3, 2), (2, 3), (4, 1), (1, 4)]
SKIP_S, SKIP_T = 100, 101


def link_types(max_size, max_k):
    return [(k, l) for k, l in TYPES if k + l <= max_size and k <= max_k and l <= max_k]


def char_sums(sentences):
    """Cumulative character lengths: csum[b] - csum[a] = characters of sentences a .. b-1."""
    return np.concatenate(([0], np.cumsum([len(s.strip()) for s in sentences]))).astype(np.int64)


def length_prior(cs, ct_scaled):
    """sqrt(log2(1 + min / max)): 1 when the two lengths agree, towards 0 when they do not."""
    hi = np.maximum(cs, ct_scaled)
    ratio = np.divide(np.minimum(cs, ct_scaled), hi, out=np.zeros_like(hi, dtype=np.float64), where=hi > 0)
    return np.sqrt(np.log2(1.0 + ratio))


class SpanScores:
    """Scores of a rectangle of span pairs: straight from the vectors when the segment has them (only the rectangle
    is computed), otherwise through ``score_block`` (whole rows).  NaN where a span contains a blank line."""

    def __init__(self, segment):
        self.segment = segment
        self.vectors = None
        if segment.src_vectors is not None and segment.trg_vectors is not None and segment.src_vectors(1) is not None:
            self.vectors = {}
            self.blank = {}
            for side, sentences in (('s', segment.src_sentences), ('t', segment.trg_sentences)):
                csum = np.concatenate(([0], np.cumsum([0 if x.strip() else 1 for x in sentences])))
                for k in range(1, segment.max_k + 1):
                    b = (csum[k:] - csum[:len(csum) - k]) > 0 if k <= len(sentences) else np.zeros(0, dtype=bool)
                    self.blank[(side, k)] = b if b.any() else None

    def block(self, k, l, p0, p1, q0, q1):
        """float32 (p1 - p0, q1 - q0): k-sentence source spans starting at p0 .. p1-1 against l-sentence target
        spans starting at q0 .. q1-1.  The array may be changed by the caller."""
        if self.vectors is None:
            return np.array(self.segment.score_block(k, l, p0, p1)[:, q0:q1], dtype=np.float32)
        for side, size, get in (('s', k, self.segment.src_vectors), ('t', l, self.segment.trg_vectors)):
            if (side, size) not in self.vectors:
                self.vectors[(side, size)] = get(size)
        out = np.matmul(self.vectors[('s', k)][p0:p1], self.vectors[('t', l)][q0:q1].T)
        if self.blank[('s', k)] is not None:
            out[self.blank[('s', k)][p0:p1], :] = np.nan
        if self.blank[('t', l)] is not None:
            out[:, self.blank[('t', l)][q0:q1]] = np.nan
        return out


# ---------------------------------------------------------------------------------------------------- seed
def seed(segment, types, threshold, band, top_k, cos_power, use_prior, band_bytes=256 * 1024 * 1024, scores=None):
    """Candidate edges, sorted by source start, then target start: arrays (i, k, j, l, score, weight)."""
    N, M = segment.n_src, segment.n_trg
    scores = scores or SpanScores(segment)
    src_c, trg_c = char_sums(segment.src_sentences), char_sums(segment.trg_sentences)
    r = src_c[-1] / trg_c[-1] if trg_c[-1] else 1.0
    out = []
    for order, (k, l) in enumerate(types):
        Nk, Ml = N - k + 1, M - l + 1
        if Nk <= 0 or Ml <= 0:
            continue
        cols = np.arange(Ml) / Ml
        rows_per = max(1, band_bytes // (4 * Ml))
        if band and band > 0:
            rows_per = min(rows_per, max(32, Nk // 64))          # short blocks keep the window of columns narrow
        for p0 in range(0, Nk, rows_per):
            p1 = min(Nk, p0 + rows_per)
            q0, q1 = 0, Ml
            if band and band > 0:                                 # only the columns the band can reach for these rows
                q0 = max(0, int(np.floor((p0 / Nk - band) * Ml)))
                q1 = min(Ml, int(np.ceil(((p1 - 1) / Nk + band) * Ml)) + 1)
            block = scores.block(k, l, p0, p1, q0, q1)
            np.nan_to_num(block, copy=False, nan=-np.inf)
            if band and band > 0:
                block[np.abs((np.arange(p0, p1) / Nk)[:, None] - cols[None, q0:q1]) >= band] = -np.inf
            keep_n = min(top_k, q1 - q0)
            idx = np.argpartition(block, (q1 - q0) - keep_n, axis=1)[:, (q1 - q0) - keep_n:]
            val = np.take_along_axis(block, idx, axis=1)
            rr, cc = np.nonzero(val >= threshold)
            i, j, cos = p0 + rr, q0 + idx[rr, cc], val[rr, cc].astype(np.float64)
            weight = cos ** cos_power
            if use_prior:
                weight = weight * length_prior((src_c[i + k] - src_c[i]).astype(np.float64),
                                               (trg_c[j + l] - trg_c[j]) * r)
            out.append((i, np.full(len(i), k), j, np.full(len(i), l), cos, weight, np.full(len(i), order)))
    if not out:
        return tuple(np.zeros(0, dtype=np.int64) for _ in range(4)) + (np.zeros(0), np.zeros(0))
    i, k, j, l, cos, weight, order = (np.concatenate(x) for x in zip(*out))
    by = np.lexsort((order, j, i))                       # source start, target start, type
    return (i[by].astype(np.int64), k[by].astype(np.int64), j[by].astype(np.int64), l[by].astype(np.int64),
            cos[by], weight[by])


# ---------------------------------------------------------------------------------------------------- chain
def chain(i, k, j, l, weight, n_trg):
    """Indices of the maximum-weight set of edges that neither overlap nor cross, in source order.

    Edges must be sorted by source start.  An edge may follow another if that one ends, on both sides, at or before
    the place where this one starts.  Sweep over the source starts: before the edges starting at position p are
    scored, every edge whose source end is <= p is entered into ``best[target end]``; the best predecessor of an
    edge starting at target j is then the prefix maximum of ``best`` up to j (computed with one cumulative maximum
    per source position instead of the paper's Fenwick tree: same result)."""
    E = len(weight)
    if E == 0:
        return []
    src_end, trg_end = i + k, j + l
    dp = np.zeros(E)
    parent = np.full(E, -1, dtype=np.int64)
    best = np.zeros(n_trg + 1)                           # best chain weight of the entered edges ending at this target
    arg = np.full(n_trg + 1, -1, dtype=np.int64)
    positions = np.arange(n_trg + 1)
    by_end = np.argsort(src_end, kind='stable').tolist()
    ends, tends = src_end.tolist(), trg_end.tolist()
    entered = 0
    acc = pos = None
    bounds = np.concatenate(([0], np.flatnonzero(np.diff(i)) + 1, [E]))
    for a, b in zip(bounds[:-1].tolist(), bounds[1:].tolist()):
        p = int(i[a])
        changed = acc is None
        while entered < E and ends[by_end[entered]] <= p:
            e = by_end[entered]
            t = tends[e]
            if dp[e] > best[t]:
                best[t], arg[t] = dp[e], e
                changed = True
            entered += 1
        if changed:
            acc = np.maximum.accumulate(best)
            rise = np.empty(n_trg + 1, dtype=bool)
            rise[0] = True
            rise[1:] = acc[1:] > acc[:-1]
            pos = np.maximum.accumulate(np.where(rise, positions, 0))     # where the prefix maximum was first reached
        q = j[a:b]
        dp[a:b] = weight[a:b] + acc[q]
        parent[a:b] = arg[pos[q]]
    e = int(np.argmax(dp))
    out = []
    while e >= 0:
        out.append(e)
        e = int(parent[e])
    return out[::-1]


# ---------------------------------------------------------------------------------------------------- extend
def gap_scores(block, margin):
    """Link scores of one type inside a gap: the scores minus half the row mean and half the column mean."""
    d = np.array(block, dtype=np.float64)
    if margin and d.size:
        ok = ~np.isnan(d)
        filled = np.where(ok, d, 0.0)
        with np.errstate(invalid='ignore', divide='ignore'):
            row = filled.sum(1) / ok.sum(1)
            col = filled.sum(0) / ok.sum(0)
        d = d - 0.5 * (np.nan_to_num(row)[:, None] + np.nan_to_num(col)[None, :])
    return np.where(np.isnan(d), -np.inf, d)


def align_gap(scores, types, g, h, skip_cost):
    """Best path through a gap of g source and h target sentences -> [(source offset, k, target offset, l)].
    ``scores[t][a, b]``: score of a link of type t starting at (a, b), -inf if not allowed."""
    F = np.full((g + 1, h + 1), -np.inf)
    F[0, 0] = 0.0
    bp = np.full((g + 1, h + 1), -1, dtype=np.int16)
    if (g + 1) * (h + 1) <= 400:                          # small gaps (nearly all): plain loops
        for a in range(g + 1):
            for b in range(h + 1):
                if a == 0 and b == 0:
                    continue
                best, how = -np.inf, -1
                for t, (k, l) in enumerate(types):
                    if a >= k and b >= l and scores[t] is not None:
                        v = F[a - k, b - l] + scores[t][a - k, b - l]
                        if v > best:
                            best, how = v, t
                if a >= 1 and F[a - 1, b] + skip_cost > best:
                    best, how = F[a - 1, b] + skip_cost, SKIP_S
                if b >= 1 and F[a, b - 1] + skip_cost > best:
                    best, how = F[a, b - 1] + skip_cost, SKIP_T
                F[a, b], bp[a, b] = best, how
    else:                                                 # the same recurrence, one anti-diagonal at a time
        for s in range(1, g + h + 1):
            a = np.arange(max(0, s - h), min(g, s) + 1)
            b = s - a
            best = np.full(len(a), -np.inf)
            how = np.full(len(a), -1, dtype=np.int16)
            for t, (k, l) in enumerate(types):
                if scores[t] is None:
                    continue
                ok = (a >= k) & (b >= l)
                if ok.any():
                    v = F[a[ok] - k, b[ok] - l] + scores[t][a[ok] - k, b[ok] - l]
                    better = v > best[ok]
                    sel = np.flatnonzero(ok)[better]
                    best[sel], how[sel] = v[better], t
            for code, da, db in ((SKIP_S, 1, 0), (SKIP_T, 0, 1)):
                ok = (a >= da) & (b >= db)
                v = F[a[ok] - da, b[ok] - db] + skip_cost
                better = v > best[ok]
                sel = np.flatnonzero(ok)[better]
                best[sel], how[sel] = v[better], code
            F[a, b], bp[a, b] = best, how
    links = []
    a, b = g, h
    while a > 0 or b > 0:
        how = int(bp[a, b])
        if how == SKIP_S:
            a -= 1
        elif how == SKIP_T:
            b -= 1
        else:
            k, l = types[how]
            a, b = a - k, b - l
            links.append((a, k, b, l))
    return links[::-1]


class ChainFinder(PathFinder):
    name = 'chain'
    postprocess = False
    uses_anchors = False
    option_defaults = {
        'threshold': 0.3,          # lowest score of a candidate edge (tau)
        'band': 0.15,              # diagonal band as a fraction of the document; 0 = no band
        'top_k': 10,               # candidates kept per source span and link type
        'max_size': 5,             # link types with k + l <= this in the gaps ...
        'seed_max_size': 5,        # ... and among the candidate edges (3 = the paper's base configuration)
        'cos_power': 2.0,          # exponent of the score in the edge weight
        'length_prior': True,      # multiply the edge weight by the length-ratio prior
        'skip_cost': -0.5,         # per skipped sentence in the gaps (-0.1 for text that is not parallel throughout)
        'margin': True,            # gap score = score - (row mean + column mean) / 2; False: the plain score
        'extend': True,            # align the gaps; False: the chain only
        'max_gap_cells': 10000000,
    }

    def find(self, segment):
        if segment.src_sentences is None or segment.trg_sentences is None:
            raise ValueError('the chain path finder needs the sentence texts in the segment')
        N, M = segment.n_src, segment.n_trg
        if N == 0 or M == 0:
            return []
        opt = self.options
        seed_types = link_types(int(opt['seed_max_size']), segment.max_k)
        self.scores = SpanScores(segment)
        i, k, j, l, cos, weight = seed(segment, seed_types, float(opt['threshold']), float(opt['band'] or 0),
                                       int(opt['top_k']), float(opt['cos_power']), bool(opt['length_prior']),
                                       self.params.band_bytes, self.scores)
        picked = chain(i, k, j, l, weight, M)
        links = [(int(i[e]), int(k[e]), int(j[e]), int(l[e])) for e in picked]
        self.report = {'candidates': int(len(weight)), 'chain_links': len(links), 'gap_links': 0, 'gaps_skipped': 0}
        if opt['extend']:
            links = self.extend(segment, links)
        return [(LINK, s, ks, t, kt) for s, ks, t, kt in links]

    # ------------------------------------------------------------------ gaps
    def extend(self, segment, links):
        opt = self.options
        types = link_types(int(opt['max_size']), segment.max_k)
        skip, margin, limit = float(opt['skip_cost']), bool(opt['margin']), int(opt['max_gap_cells'])
        N, M = segment.n_src, segment.n_trg
        scores_of = getattr(self, 'scores', None) or SpanScores(segment)

        out = []
        edges = links + [(N, 0, M, 0)]
        s0 = t0 = 0
        for s, ks, t, kt in edges:
            g, h = s - s0, t - t0
            if g > 0 and h > 0:
                if (g + 1) * (h + 1) > limit:
                    self.report['gaps_skipped'] += 1
                    log.warning('chain: a gap of %d x %d sentences is left unaligned (larger than max_gap_cells)', g, h)
                else:
                    scores = [gap_scores(scores_of.block(kk, ll, s0, s - kk + 1, t0, t - ll + 1), margin)
                              if (g >= kk and h >= ll) else None
                              for kk, ll in types]
                    found = align_gap(scores, types, g, h, skip)
                    self.report['gap_links'] += len(found)
                    out.extend((s0 + a, kk, t0 + b, ll) for a, kk, b, ll in found)
            if ks:
                out.append((s, ks, t, kt))
            s0, t0 = s + ks, t + kt
        return out
