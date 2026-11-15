"""Scoring: encoders (texts -> vectors) and scorers (span similarity by blocks)."""
from sentalign.scoring.base import Scorer, get_encoder, get_scorer, register_encoder, register_scorer


def _register_builtins():
    from sentalign.scoring.embedding import EmbeddingScorer
    from sentalign.scoring.encoders import HFEncoder
    from sentalign.scoring.laser import LaserEncoder
    register_encoder('hf', HFEncoder)
    register_encoder('laser', LaserEncoder)
    register_scorer('embedding', EmbeddingScorer)


_register_builtins()

__all__ = ['Scorer', 'get_encoder', 'get_scorer', 'register_encoder', 'register_scorer']
