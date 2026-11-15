"""SentAlign: sentence alignment with LaBSE embeddings.

Importing this package has no side effects: nothing is compiled, no file is written, no process is started and the
heavy libraries (torch, transformers) are not imported until they are needed.  Use the ``sentalign`` command, or from
Python ``sentalign.align(source_sentences, target_sentences)`` and ``sentalign.SentAlign`` (see ``sentalign.api``).
"""

__version__ = "2.0.0"
__all__ = ["__version__", "align", "SentAlign", "Alignment", "Link"]


def __getattr__(name):          # lazy: importing sentalign does not import torch
    if name in ("align", "SentAlign", "Alignment", "Link"):
        from sentalign import api
        return getattr(api, name)
    raise AttributeError("module 'sentalign' has no attribute %r" % name)
