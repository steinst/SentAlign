"""Anchoring of long documents.  See ``base`` (interface) and ``strategies``."""
from sentalign.anchors.base import AnchorError, AnchorStrategy, get_anchorer, register_anchorer, segments, validate


def _register_builtins():
    from sentalign.anchors import strategies as s
    for cls in (s.NoAnchors, s.MutualBest, s.GaleChurchConfirmed, s.FromFile):
        register_anchorer(cls.name, cls)


_register_builtins()

__all__ = ['AnchorError', 'AnchorStrategy', 'get_anchorer', 'register_anchorer', 'segments', 'validate']
