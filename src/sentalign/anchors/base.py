"""Anchors: where a long document is cut so that the path search works on smaller segments.

An anchor (i, j) says "source sentence i and target sentence j begin a new segment": the path must pass through the
node between sentences i-1 / i and j-1 / j.  A wrong anchor is the worst error the aligner can make (everything
between the true and the false position is lost), so a strategy should return few anchors and only sure ones; if it
cannot split a segment, the segment is simply solved as a larger graph.

To add a strategy, subclass ``AnchorStrategy`` and register it with ``register_anchorer`` or through the entry-point
group ``sentalign.anchorers``; select it with ``--anchors NAME``.
"""
import bisect
import logging
from importlib import metadata

log = logging.getLogger('sentalign.anchors')


class AnchorError(ValueError):
    """The anchors are not usable (not increasing, outside the document)."""


class AnchorStrategy:
    name = 'abstract'

    def __init__(self, min_score=0.95, floor_score=0.8, margin=0.05, min_words=4):
        self.min_score, self.floor_score, self.margin, self.min_words = min_score, floor_score, margin, min_words
        self.report = {}          # what the strategy did, for the JSON run record

    def find(self, src, trg, scorer, max_nodes):
        """-> anchors [(i, j), ...] strictly increasing on both sides, 0 < i < len(src), 0 < j < len(trg)."""
        raise NotImplementedError


def validate(anchors, n_src, n_trg):
    """Strictly increasing on both sides and inside the document.  Returns the anchors as int tuples."""
    out = []
    pi = pj = 0
    for a in anchors:
        i, j = int(a[0]), int(a[1])
        if not (pi < i < n_src and pj < j < n_trg):
            raise AnchorError('anchor (%d, %d) after (%d, %d) in a document of %d x %d sentences'
                              % (i, j, pi, pj, n_src, n_trg))
        out.append((i, j))
        pi, pj = i, j
    return out


def segments(anchors, n_src, n_trg):
    """[(i0, i1, j0, j1), ...]: the segments between the anchors."""
    out = []
    i0 = j0 = 0
    for i, j in list(anchors) + [(n_src, n_trg)]:
        out.append((i0, i, j0, j))
        i0, j0 = i, j
    return out


def nodes(i0, i1, j0, j1):
    return (i1 - i0 + 1) * (j1 - j0 + 1)


def longest_chain(pairs):
    """Largest subset of the (i, j, ...) tuples that increases strictly in i and in j (longest increasing
    subsequence).  A false candidate is one that the true ones contradict, so it falls out here."""
    pairs = sorted(pairs)
    tails, tail_idx, back = [], [], [-1] * len(pairs)
    for k, p in enumerate(pairs):
        if k and pairs[k - 1][0] == p[0]:
            continue                      # one candidate per source sentence
        pos = bisect.bisect_left(tails, p[1])
        if pos == len(tails):
            tails.append(p[1])
            tail_idx.append(k)
        else:
            tails[pos] = p[1]
            tail_idx[pos] = k
        back[k] = tail_idx[pos - 1] if pos else -1
    chain = []
    k = tail_idx[-1] if tail_idx else -1
    while k >= 0:
        chain.append(pairs[k])
        k = back[k]
    return chain[::-1]


def supported(chain, window=20, support=2):
    """The members of the chain that have at least `support` other members within `window` sentences on both sides
    of the text.  Being in order with the rest does not make a candidate right: a false match far off the diagonal
    (source 30 against target 6, between the true pairs 5-5 and 31-31) is in order too.  A true match sits among
    other matches that continue the same stretch of text; an isolated one is not used as an anchor."""
    out = []
    lo = 0
    for k, c in enumerate(chain):
        while chain[lo][0] < c[0] - window:
            lo += 1
        near = 0
        h = lo
        while h < len(chain) and chain[h][0] <= c[0] + window:
            if h != k and abs(chain[h][1] - c[1]) <= window:
                near += 1
            h += 1
        if near >= support:
            out.append(c)
    return out


def choose(chain, n_src, n_trg, max_nodes, levels):
    """Pick as few anchors from the chain [(i, j, score), ...] as are needed to bring every segment under
    `max_nodes`: in a segment that is too large, the candidate that splits it most evenly, taken from the most
    confident score level that has one.  -> (anchors, segments that could not be split)."""
    anchors, too_large = [], []
    starts = [c[0] for c in chain]

    def split(i0, i1, j0, j1):
        if nodes(i0, i1, j0, j1) <= max_nodes:
            return
        lo, hi = bisect.bisect_right(starts, i0), bisect.bisect_left(starts, i1)
        inside = [c for c in chain[lo:hi] if j0 < c[1] < j1]
        for level in levels:
            cands = [c for c in inside if c[2] >= level]
            if cands:
                i, j, _ = min(cands, key=lambda c: max(nodes(i0, c[0], j0, c[1]), nodes(c[0], i1, c[1], j1)))
                split(i0, i, j0, j)
                anchors.append((i, j))
                split(i, i1, j, j1)
                return
        too_large.append((i0, i1, j0, j1))

    split(0, n_src, 0, n_trg)
    return anchors, too_large


_ANCHORERS = {}


def register_anchorer(name, cls):
    _ANCHORERS[name] = cls


def get_anchorer(name):
    if name not in _ANCHORERS:
        try:
            eps = metadata.entry_points(group='sentalign.anchorers')
        except TypeError:
            eps = metadata.entry_points().get('sentalign.anchorers', [])
        for ep in eps:
            _ANCHORERS.setdefault(ep.name, ep)
    if name not in _ANCHORERS:
        raise KeyError('unknown anchor strategy %r (known: %s)' % (name, ', '.join(sorted(_ANCHORERS)) or 'none'))
    obj = _ANCHORERS[name]
    if isinstance(obj, metadata.EntryPoint):
        obj = _ANCHORERS[name] = obj.load()
    return obj
