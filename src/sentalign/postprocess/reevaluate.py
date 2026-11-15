"""Post-processing of a path (version 2): re-evaluate the links, let links absorb neighbouring unaligned
sentences, and lay out the result.

The path search favours merging: a 2-2 link of two good pairs can beat two 1-1 links.  As in the 2023 paper, every
link with more than one sentence on a side is therefore looked at again:

1. *Re-evaluation.*  A link with several sentences on both sides is replaced by its best-scoring pair of
   sub-spans, and the sentences before and after that pair are treated the same way; a link with several sentences on
   one side keeps the best-scoring part of that side.  What is left over becomes unaligned.
2. *Absorption.*  An unaligned sentence directly before or after a link is added to it if that raises the link's
   score.

Differences from the 2023 code (``sentalign.legacy``), which stays available as ``--compat paper-2023``:

* only spans of consecutive sentences are considered (2023: any subset, which could produce a link that skips a
  sentence);
* a sentence that is part of a link is never also treated as unaligned, and an absorbed sentence is absorbed
  once; so no sentence can end up in two links;
* absorption is applied best gain first instead of in file order, which makes the result independent of the order
  of the links;
* the result is checked: every sentence exactly once, links in order.
"""
import heapq
import math
from dataclasses import dataclass


@dataclass
class Link:
    s0: int      # first source sentence
    s1: int      # one past the last source sentence
    t0: int
    t1: int
    score: float = 0.0


class SpanScores:
    """Score of a source span against a target span, NaN if there is none (blank line, span longer than max_k, word
    count outside the limits)."""

    def __init__(self, scorer, src, trg, minimum_length_words=0, maximum_length_words=None):
        self.scorer, self.src, self.trg = scorer, src, trg
        self.max_k = src.max_k
        # the same word limits as the path search: a link it would refuse must not come out of re-evaluation or
        # absorption either
        self.minw, self.maxw = minimum_length_words, maximum_length_words

    def _has_blank(self, side, start, size):
        return any(not side.sentences[i] for i in range(start, start + size))

    def __call__(self, s0, s1, t0, t1):
        ks, kt = s1 - s0, t1 - t0
        if not (1 <= ks <= self.max_k and 1 <= kt <= self.max_k):
            return math.nan
        if self._has_blank(self.src, s0, ks) or self._has_blank(self.trg, t0, kt):
            return math.nan
        for side, a, k in ((self.src, s0, ks), (self.trg, t0, kt)):
            words = int(side.word_counts[a:a + k].sum())
            if words < self.minw or (self.maxw is not None and words > self.maxw):
                return math.nan
        return float(self.scorer.pair(s0, ks, t0, kt))


TIE = 1e-6      # a span must beat an earlier (shorter) one by more than this; float32 scores differ by less by chance


def _best_pair(score, s0, s1, t0, t1):
    """Best-scoring (sub-span of s0..s1, sub-span of t0..t1); shorter spans first, the first maximum wins, and a
    later span only replaces it when it is better by more than rounding noise (two identical sentence pairs, each
    scoring 0.99999994, are two links, not one merged link scoring 1.0)."""
    best, arg = 0.0, None
    for ls in range(1, s1 - s0 + 1):
        for a in range(s0, s1 - ls + 1):
            for lt in range(1, t1 - t0 + 1):
                for b in range(t0, t1 - lt + 1):
                    v = score(a, a + ls, b, b + lt)
                    if v > best + (TIE if arg is not None else 0.0):      # NaN never wins
                        best, arg = v, (a, a + ls, b, b + lt)
    return best, arg


def _split(score, cutoff, s0, s1, t0, t1):
    """Links for the block s0..s1 x t0..t1: the best pair if it reaches the cutoff, then the same before and after."""
    if s0 >= s1 or t0 >= t1:
        return []
    best, arg = _best_pair(score, s0, s1, t0, t1)
    if arg is None or best < cutoff:
        return []
    a0, a1, b0, b1 = arg
    return _split(score, cutoff, s0, a0, t0, b0) + [Link(a0, a1, b0, b1, best)] + _split(score, cutoff, a1, s1, b1, t1)


def reevaluate(links, score, cutoff):
    out = []
    for ln in links:
        ns, nt = ln.s1 - ln.s0, ln.t1 - ln.t0
        if ns == 1 and nt == 1:
            out.append(Link(ln.s0, ln.s1, ln.t0, ln.t1, score(ln.s0, ln.s1, ln.t0, ln.t1)))
        elif ns > 1 and nt > 1:
            out.extend(_split(score, cutoff, ln.s0, ln.s1, ln.t0, ln.t1))
        else:
            # several sentences on one side only: keep the best part of that side (as 2023: no cutoff test here,
            # the link as a whole passed the cutoff in the path search)
            best, arg = _best_pair(score, ln.s0, ln.s1, ln.t0, ln.t1)
            if arg is not None:
                out.append(Link(*arg, best))
    return out


def absorb(links, src_free, trg_free, score):
    """Let links take in free (unaligned) neighbouring sentences while that raises their score; best gain first.
    `src_free` / `trg_free` are sets of sentence numbers and are updated."""
    version = [0] * len(links)
    heap = []

    def push(i):
        ln = links[i]
        for kind, cand in (('s-', (ln.s0 - 1, ln.s1, ln.t0, ln.t1)), ('s+', (ln.s0, ln.s1 + 1, ln.t0, ln.t1)),
                           ('t-', (ln.s0, ln.s1, ln.t0 - 1, ln.t1)), ('t+', (ln.s0, ln.s1, ln.t0, ln.t1 + 1))):
            new = {'s-': ln.s0 - 1, 's+': ln.s1, 't-': ln.t0 - 1, 't+': ln.t1}[kind]
            if new not in (src_free if kind[0] == 's' else trg_free):
                continue
            v = score(*cand)
            if v > ln.score:
                heapq.heappush(heap, (-(v - ln.score), i, kind, version[i], cand, v))

    for i in range(len(links)):
        push(i)
    while heap:
        _, i, kind, ver, cand, v = heapq.heappop(heap)
        if ver != version[i]:
            continue                                   # the link has changed since this candidate was computed
        new = {'s-': cand[0], 's+': cand[1] - 1, 't-': cand[2], 't+': cand[3] - 1}[kind]
        free = src_free if kind[0] == 's' else trg_free
        if new not in free:
            continue                                   # a neighbour took the sentence first
        free.discard(new)
        links[i] = Link(*cand, v)
        version[i] += 1
        push(i)
    return links


def postprocess(links, score, cutoff, src_first, n_src, trg_first, n_trg):
    """links: the links of the path of one segment (absolute sentence numbers).  -> final links, in order."""
    links = reevaluate(links, score, cutoff)
    src_free = set(range(src_first, src_first + n_src))
    trg_free = set(range(trg_first, trg_first + n_trg))
    for ln in links:
        src_free.difference_update(range(ln.s0, ln.s1))
        trg_free.difference_update(range(ln.t0, ln.t1))
    links = absorb(links, src_free, trg_free, score)
    check(links, src_first, n_src, trg_first, n_trg)
    return links


def check(links, src_first, n_src, trg_first, n_trg):
    """Every sentence in at most one link, links in order and inside the segment."""
    s, t = src_first, trg_first
    for ln in links:
        if not (s <= ln.s0 < ln.s1 and t <= ln.t0 < ln.t1):
            raise AssertionError('links overlap or are out of order at %r' % (ln,))
        s, t = ln.s1, ln.t1
    if s > src_first + n_src or t > trg_first + n_trg:
        raise AssertionError('a link reaches beyond the segment')


def rows(links, src_first, n_src, trg_first, n_trg):
    """Output rows (source numbers, target numbers, score) in the order of the 2023 files: before each link the
    unaligned source sentences, then the unaligned target sentences; unaligned rows have score None."""
    out = []
    s, t = src_first, trg_first
    for ln in list(links) + [None]:
        s_to = ln.s0 if ln else src_first + n_src
        t_to = ln.t0 if ln else trg_first + n_trg
        out.extend(((i,), (), None) for i in range(s, s_to))
        out.extend(((), (j,), None) for j in range(t, t_to))
        if ln:
            out.append((tuple(range(ln.s0, ln.s1)), tuple(range(ln.t0, ln.t1)), ln.score))
            s, t = ln.s1, ln.t1
    return out
