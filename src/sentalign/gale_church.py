"""Gale-Church length-based alignment, from the ``galechurch`` package (Rust core).

Replaces the Cython implementation and the backward file reader that SentAlign 1.0 carried (about 540 lines).

Blank lines.  Gale-Church gives a pair of empty sides the cost minus infinity (the original program did the same), which
turns the whole alignment into nonsense when a blank line is passed in as a zero-length sentence.  Version 1 avoided
that by cutting the file at blank lines and aligning the blocks one by one, but it numbered the sentences inside each
block from 0, so for a file with blank lines the anchors pointed at the wrong sentences.  Here the blank
lines are left out, the rest is aligned as one block, and the beads are given in the numbers of the original file.
A bead that would span a blank line is dropped (the sentences are simply not linked by Gale-Church).
"""
import galechurch


def gale_church_beads(src_sentences, trg_sentences):
    """Linking beads (s0, s1, t0, t1) of the Gale-Church alignment, in order, numbered as in the input lists.
    Only beads with at least one sentence on each side and no blank line inside are returned; every other sentence is
    unaligned."""
    src_keep = [i for i, s in enumerate(src_sentences) if s.strip()]
    trg_keep = [j for j, s in enumerate(trg_sentences) if s.strip()]
    n, m = len(src_keep), len(trg_keep)
    if n == 0 or m == 0:
        return []
    # the package refuses blocks above 50 M cells unless told otherwise; it needs one byte per cell
    block = galechurch.align([src_sentences[i] for i in src_keep], [trg_sentences[j] for j in trg_keep],
                             max_block_size=(n + 1) * (m + 1) + 1)
    beads = []
    for b in block.beads:
        if b.source_end > b.source_start and b.target_end > b.target_start:
            s0, s1 = src_keep[b.source_start], src_keep[b.source_end - 1] + 1
            t0, t1 = trg_keep[b.target_start], trg_keep[b.target_end - 1] + 1
            if s1 - s0 == b.source_end - b.source_start and t1 - t0 == b.target_end - b.target_start:
                beads.append((s0, s1, t0, t1))
    return beads


def anchor_candidates(src_sentences, trg_sentences):
    """The Gale-Church beads as the pairs of comma-joined sentence numbers ['0,1', '2'] that the version-1 anchoring
    expects."""
    return [[','.join(map(str, range(s0, s1))), ','.join(map(str, range(t0, t1)))]
            for s0, s1, t0, t1 in gale_church_beads(src_sentences, trg_sentences)]
