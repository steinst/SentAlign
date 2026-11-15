"""The anchor strategies of SentAlign 2.

``mutual-best``   (default) a pair of sentences is a candidate only if each is the other's best match in the whole
                  document, clearly ahead of the second best on both sides, and neither is very short.  Candidates
                  that are out of order with the rest, or that stand alone (no other candidates near them on both
                  sides of the text), are not used.
``gale-church``   the candidates are the 1-1 links of the Gale-Church alignment that the scorer confirms (the
                  anchoring described in the 2023 paper).
``file``          anchors given by the user (e.g. confirmed chapter starts), one "i<TAB>j" per line.
``none``          no anchors.

For the first two, the candidates that contradict the others are removed (longest increasing chain), and from the
rest only as many are used as are needed, the most confident and most evenly splitting first (``base.choose``).
"""
import logging

import numpy as np

from sentalign.anchors.base import AnchorStrategy, choose, longest_chain, supported, validate
from sentalign.gale_church import gale_church_beads

log = logging.getLogger('sentalign.anchors')


def _levels(top, floor, step=0.05):
    out, v = [], top
    while v > floor + 1e-9:
        out.append(round(v, 6))
        v -= step
    return out + [floor]


class NoAnchors(AnchorStrategy):
    name = 'none'

    def find(self, src, trg, scorer, max_nodes):
        return []


class MutualBest(AnchorStrategy):
    name = 'mutual-best'
    chunk_rows = 2048
    window = 20        # an anchor needs `support` other consistent candidates within this many sentences
    support = 2

    def best_two(self, src, trg, scorer):
        """For every source sentence its best and second-best target sentence (1-1 scores), and the same for every
        target sentence, computed in bands of rows."""
        n, m = len(src), len(trg)
        blank_s = np.asarray([not s for s in src.sentences])
        blank_t = np.asarray([not s for s in trg.sentences])
        row_best = np.full(n, -np.inf, dtype=np.float32)
        row_second = np.full(n, -np.inf, dtype=np.float32)
        row_arg = np.zeros(n, dtype=np.int64)
        col_best = np.full(m, -np.inf, dtype=np.float32)
        col_second = np.full(m, -np.inf, dtype=np.float32)
        col_arg = np.zeros(m, dtype=np.int64)
        for i0 in range(0, n, self.chunk_rows):
            i1 = min(n, i0 + self.chunk_rows)
            blk = np.array(scorer.block(1, 1, i0, i1, 0, m), dtype=np.float32, copy=True)
            blk[blank_s[i0:i1], :] = -np.inf
            blk[:, blank_t] = -np.inf
            # rows
            arg = blk.argmax(axis=1)
            row_arg[i0:i1] = arg
            row_best[i0:i1] = blk[np.arange(i1 - i0), arg]
            if m > 1:
                row_second[i0:i1] = np.partition(blk, m - 2, axis=1)[:, m - 2]
            # columns: merge this band into the running best and second best
            carg = blk.argmax(axis=0)
            cbest = blk[carg, np.arange(m)]
            csecond = np.partition(blk, i1 - i0 - 2, axis=0)[i1 - i0 - 2] if i1 - i0 > 1 else np.full(m, -np.inf, dtype=np.float32)
            better = cbest > col_best
            col_second = np.maximum(np.maximum(col_second, csecond), np.minimum(col_best, cbest))
            col_arg = np.where(better, carg + i0, col_arg)
            col_best = np.maximum(col_best, cbest)
        return row_best, row_second, row_arg, col_best, col_second, col_arg

    def candidates(self, src, trg, scorer):
        row_best, row_second, row_arg, col_best, col_second, col_arg = self.best_two(src, trg, scorer)
        sw, tw = src.word_counts, trg.word_counts
        out = []
        for i in range(len(src)):
            j = int(row_arg[i])
            v = float(row_best[i])
            if (col_arg[j] == i and v >= self.floor_score and v - row_second[i] >= self.margin
                    and v - col_second[j] >= self.margin and sw[i] >= self.min_words and tw[j] >= self.min_words):
                out.append((i, j, v))
        return out

    def find(self, src, trg, scorer, max_nodes):
        n, m = len(src), len(trg)
        cands = self.candidates(src, trg, scorer)
        in_order = longest_chain(cands)
        near = supported(in_order, self.window, self.support)
        chain = [c for c in near if 0 < c[0] < n and 0 < c[1] < m]
        anchors, too_large = choose(chain, n, m, max_nodes, _levels(self.min_score, self.floor_score))
        self.report = {'strategy': self.name, 'candidates': len(cands), 'contradicted': len(cands) - len(in_order),
                       'isolated': len(in_order) - len(near), 'usable': len(chain), 'used': len(anchors),
                       'segments_left_too_large': [list(s) for s in too_large]}
        if too_large:
            log.warning('%s: no sure anchor inside %d segment(s) larger than %d nodes; they are solved whole '
                        '(largest %d x %d sentences)', self.name, len(too_large), max_nodes,
                        max(s[1] - s[0] for s in too_large), max(s[3] - s[2] for s in too_large))
        return validate(sorted(anchors), n, m)


class GaleChurchConfirmed(MutualBest):
    name = 'gale-church'

    def candidates(self, src, trg, scorer):
        sw, tw = src.word_counts, trg.word_counts
        out = []
        for s0, s1, t0, t1 in gale_church_beads(src.sentences, trg.sentences):
            if s1 - s0 == 1 and t1 - t0 == 1 and sw[s0] >= self.min_words and tw[t0] >= self.min_words:
                v = float(scorer.pair(s0, 1, t0, 1))
                if v >= self.floor_score:
                    out.append((s0, t0, v))
        return out


class FromFile(AnchorStrategy):
    """Anchors from a text file: one "source_sentence<TAB>target_sentence" per line (0-based numbers of the sentences
    that begin a new segment); lines starting with # are ignored.  All of them are used, whatever the node limit."""
    name = 'file'

    def __init__(self, path=None, **kw):
        super().__init__(**kw)
        self.path = path

    def find(self, src, trg, scorer, max_nodes):
        anchors = []
        with open(self.path, encoding='utf-8') as f:
            for line in f:
                line = line.strip()
                if line and not line.startswith('#'):
                    a, b = line.replace(',', '\t').split()[:2]
                    anchors.append((int(a), int(b)))
        n, m = len(src), len(trg)
        anchors = [a for a in sorted(set(anchors)) if a not in ((0, 0), (n, m))]
        self.report = {'strategy': self.name, 'file': self.path, 'used': len(anchors)}
        return validate(anchors, n, m)
