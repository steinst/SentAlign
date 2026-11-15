"""Segments for the path search, built from the two document sides and a scorer.

``make_segment``            version 2: scores straight from the scorer; a span with a blank line has no score.
``PaperLookup.segment``     paper-2023: the scores the 2023 code found by looking up texts in its dictionaries.
"""
import numpy as np

from sentalign.pathfinding.base import LINK, SKIP_SOURCE, SKIP_TARGET, Segment, span_words


def _blank_spans(side, first, n, k):
    """bool array over the spans of k sentences starting at first .. first+n-k: contains a blank sentence."""
    if k > n:                      # no span of k sentences (a short segment): empty mask, as span_words does
        return np.zeros(0, dtype=bool)
    blank = np.asarray([0 if s else 1 for s in side.sentences[first:first + n]], dtype=np.int64)
    csum = np.concatenate(([0], np.cumsum(blank)))
    return (csum[k:] - csum[:len(csum) - k]) > 0


def make_segment(scorer, src, trg, src_first, n_src, trg_first, n_trg):
    """Segment of n_src source sentences from `src_first` and n_trg target sentences from `trg_first` (version 2)."""
    K = src.max_k
    # spans that contain a blank line, per span length (None: there is none, the usual case)
    src_blank = [_blank_spans(src, src_first, n_src, k) for k in range(1, K + 1)]
    trg_blank = [_blank_spans(trg, trg_first, n_trg, k) for k in range(1, K + 1)]
    src_blank = [b if b.any() else None for b in src_blank]
    trg_blank = [b if b.any() else None for b in trg_blank]

    def score_block(ks, kt, p0, p1, out=None):
        """Scores of the block; written into `out` (a float32 array of the block's shape) if given."""
        blk = scorer.block(ks, kt, src_first + p0, src_first + p1, trg_first, trg_first + n_trg - kt + 1, out=out)
        if out is not None and blk is not out:           # a scorer that ignores `out`
            out[...] = blk
            blk = out
        if src_blank[ks - 1] is not None:
            blk[src_blank[ks - 1][p0:p1], :] = np.nan
        if trg_blank[kt - 1] is not None:
            blk[:, trg_blank[kt - 1]] = np.nan
        return blk

    return Segment(n_src, n_trg, K, score_block,
                   span_words(src.word_counts[src_first:src_first + n_src], K),
                   span_words(trg.word_counts[trg_first:trg_first + n_trg], K),
                   src.sentences[src_first:src_first + n_src], trg.sentences[trg_first:trg_first + n_trg],
                   takes_out=True,
                   src_vectors=lambda k: scorer.span_vectors('src', k, src_first, n_src),
                   trg_vectors=lambda k: scorer.span_vectors('trg', k, trg_first, n_trg),
                   text_vector=scorer.text_vector)


class PaperLookup:
    """The embedding lookups of the 2023 code, as arrays (paper-2023 mode).

    Version 1 found the vector of a span by looking up its text, ``' '.join(sentences)``, in a dictionary keyed by
    the stripped texts of the overlap file.  Consequences that are reproduced here:
    * two spans with the same text share the vector of the last one in the file;
    * a span with a blank line, or longer than 10,000 characters, finds nothing: score "0" (NaN in the blocks);
    * the word count of a span is the number of spaces in that text plus one.
    """

    def __init__(self, scorer):
        self.scorer = scorer
        self.src, self.trg = scorer.src, scorer.trg
        self.src_vec, self.src_spaces = self._side(self.src, scorer.src_emb)
        self.trg_vec, self.trg_spaces = self._side(self.trg, scorer.trg_emb)

    @staticmethod
    def _side(side, emb):
        key_row = {}
        for r, text in enumerate(side.texts):
            key_row[text.strip()] = r                       # later rows overwrite earlier ones
        found = np.full(side.n_rows, -1, dtype=np.int64)
        for r in range(side.n_rows):
            s, k = int(side.starts[r]), int(side.sizes[r])
            found[r] = key_row.get(' '.join(side.sentences[s:s + k]), -1)
        vec = emb[np.where(found >= 0, found, 0)].astype(np.float32, copy=True)
        vec[found < 0] = np.nan                             # a product with NaN stays NaN: "nothing found"
        spaces = np.asarray([s.count(' ') for s in side.sentences], dtype=np.int64)
        return vec, spaces

    def segment(self, src_first, n_src, trg_first, n_trg):
        K = self.src.max_k
        src, trg = self.src, self.trg

        def score_block(ks, kt, p0, p1, out=None):
            s = self.src_vec[src.offsets[ks - 1] + src_first + p0: src.offsets[ks - 1] + src_first + p1]
            t = self.trg_vec[trg.offsets[kt - 1] + trg_first: trg.offsets[kt - 1] + trg_first + n_trg - kt + 1]
            return np.matmul(s, t.T, out=out)

        def words(spaces, first, n):
            # text.count(' ') + 1 of the joined span: spaces inside the sentences + (k - 1) joints + 1
            return [w + k for k, w in enumerate(span_words(spaces[first:first + n], K), start=1)]

        return Segment(n_src, n_trg, K, score_block, words(self.src_spaces, src_first, n_src),
                       words(self.trg_spaces, trg_first, n_trg), takes_out=True)


def steps_to_paper_text(steps, src_first, trg_first):
    """The path as the text the 2023 code passed from its path search to its post-processing."""
    out = []
    for kind, p, ks, q, kt in steps:
        if kind == SKIP_SOURCE:
            out.append('[%d:]\n' % (src_first + p))
        elif kind == SKIP_TARGET:
            out.append('[:%d]\n' % (trg_first + q))
        elif kind == LINK:
            out.append('[%s:%s]\n' % (','.join(str(src_first + p + i) for i in range(ks)),
                                      ','.join(str(trg_first + q + j) for j in range(kt))))
    return ''.join(out)
