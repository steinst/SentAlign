"""Gale-Church (sentence lengths only) as a path finder: a baseline that needs no embeddings for the path.

The links are the Gale-Church beads (1-1, 1-2, 2-1, 2-2).  Sentences that no bead links are unaligned (insertions,
deletions, blank lines).  The links are scored with the scorer so that the output has the usual score column, but
the scores play no part in the choice, and there is no post-processing: the result is Gale-Church's.
"""
from sentalign.gale_church import gale_church_beads
from sentalign.pathfinding.base import LINK, PathFinder


class GaleChurchFinder(PathFinder):
    name = 'galechurch'
    postprocess = False

    def find(self, segment, stop_node=None):
        """Link steps only; unaligned sentences are implied by the gaps between them."""
        if segment.src_sentences is None or segment.trg_sentences is None:
            raise ValueError('the galechurch path finder needs the sentence texts in the segment')
        return [(LINK, s0, s1 - s0, t0, t1 - t0)
                for s0, s1, t0, t1 in gale_church_beads(segment.src_sentences, segment.trg_sentences)]
