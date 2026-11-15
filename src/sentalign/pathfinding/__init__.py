"""Path search: the SentAlign dynamic programme and the interface for other algorithms."""
from sentalign.pathfinding.base import (LINK, SKIP_SOURCE, SKIP_TARGET, PathFinder, PathParams, Segment,
                                        get_pathfinder, register_pathfinder)


def _register_builtins():
    from sentalign.pathfinding.sentalign_dp import SentAlignDP
    from sentalign.pathfinding.galechurch_finder import GaleChurchFinder
    register_pathfinder('sentalign-dp', SentAlignDP)
    from sentalign.pathfinding.vecalign_finder import VecalignFinder
    register_pathfinder('galechurch', GaleChurchFinder)
    from sentalign.pathfinding.chain_finder import ChainFinder
    register_pathfinder('vecalign', VecalignFinder)
    register_pathfinder('chain', ChainFinder)


_register_builtins()

__all__ = ['LINK', 'SKIP_SOURCE', 'SKIP_TARGET', 'PathFinder', 'PathParams', 'Segment', 'get_pathfinder',
           'register_pathfinder']
