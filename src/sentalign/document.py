"""One language side of a document: sentences and all concatenation spans, as arrays.

Replaces the overlap files of version 1 (``overlaps.<file>.src`` with ``.linenumbers``, ``.overlaps``,
``.lengthok``).  The order of the span rows is the order of those files, which matters for ``paper-2023`` mode:
the encoder batches rows in this order, and the legacy dictionaries let later rows overwrite earlier ones.
"""
from dataclasses import dataclass, field
from typing import List, Optional

import numpy as np

BLANK_LINE = 'BLANK_LINE'      # version 1 replaced an empty line by this text before encoding
MAX_TEXT_CHARS = 10000         # version 1 cut every span text at this many characters


def read_sentences(path):
    """Lines of a sentence file, stripped (as version 1 ``read_sentences``)."""
    with open(path, encoding='utf-8') as f:
        return [line.strip() for line in f]


@dataclass
class Side:
    sentences: List[str]
    max_k: int = 4
    minimum_length_words: int = 1
    maximum_length_words: int = 110
    lang: Optional[str] = None            # language code of the text (used by language-specific encoders)
    texts: List[str] = field(init=False)
    starts: np.ndarray = field(init=False)
    sizes: np.ndarray = field(init=False)
    length_ok: np.ndarray = field(init=False)
    word_counts: np.ndarray = field(init=False)
    offsets: List[int] = field(init=False)

    def __post_init__(self):
        if self.max_k < 1:
            raise ValueError('max_k must be >= 1')
        n = len(self.sentences)
        # as preprocess_line of Vecalign (Copyright 2019 Brian Thompson, Apache-2.0): strip, empty line -> BLANK_LINE
        prepared = [s.strip() or BLANK_LINE for s in self.sentences]
        texts, starts, sizes, offsets = [], [], [], [0]
        for k in range(1, self.max_k + 1):
            for start in range(n - k + 1):
                texts.append(' '.join(prepared[start:start + k])[:MAX_TEXT_CHARS])
                starts.append(start)
                sizes.append(k)
            offsets.append(len(texts))
        self.texts = texts
        self.starts = np.asarray(starts, dtype=np.int64)
        self.sizes = np.asarray(sizes, dtype=np.int64)
        self.offsets = offsets                      # rows of size k: offsets[k-1] .. offsets[k]-1
        words = np.asarray([len(t.split()) for t in texts], dtype=np.int64)
        self.length_ok = (words >= self.minimum_length_words) & (words <= self.maximum_length_words)
        self.word_counts = np.asarray([len(s.split()) for s in self.sentences], dtype=np.int64)

    @classmethod
    def from_file(cls, path, **kw):
        return cls(read_sentences(path), **kw)

    def __len__(self):
        return len(self.sentences)

    @property
    def n_rows(self):
        return len(self.texts)

    def row(self, k, start):
        """Row of the span of k sentences starting at `start`."""
        if not (1 <= k <= self.max_k) or not (0 <= start <= len(self.sentences) - k):
            raise IndexError('no span of %d sentence(s) starting at %d (document has %d)' % (k, start, len(self)))
        return self.offsets[k - 1] + start

    def rows_of_size(self, k):
        """slice of the rows of all spans of k sentences (contiguous, ordered by start)."""
        return slice(self.offsets[k - 1], self.offsets[k])

    def line_numbers(self, row):
        """'3,4,5' for the row of the span 3..5 (the version-1 .linenumbers format)."""
        s, k = int(self.starts[row]), int(self.sizes[row])
        return ','.join(str(i) for i in range(s, s + k))
