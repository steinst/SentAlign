"""Python API: align documents that are in memory.

    >>> import sentalign
    >>> aligner = sentalign.SentAlign(device='cpu')                     # loads the model once
    >>> result = aligner.align(english_sentences, icelandic_sentences)
    >>> for link in result.aligned():
    ...     print(link.src, link.trg, round(link.score, 2))
    >>> result.pairs()                                                    # [(source text, target text, score), ...]

``sentalign.align(src, trg, **options)`` does the same with a shared, cached aligner (the model is loaded on the first
call and reused while the options stay the same).

The options are those of the command line, in their long Python names (``score_cutoff=0.4``, ``max_concatenations=4``,
``encoder='setu4993/LaBSE'`` ...); ``compat='none'`` is SentAlign 2, ``compat='paper-2023'`` reproduces SentAlign 1.0.
"""
import json
import dataclasses
import re
from dataclasses import dataclass, field
from typing import List, Optional, Sequence, Tuple

from sentalign.config import AlignConfig

_ALIASES = {'encoder': 'model_name', 'device': 'proc_device', 'workers': 'num_proc', 'cutoff': 'score_cutoff',
            'max_concats': 'max_concatenations', 'anchor_limit': 'anchoring_delimiter'}


@dataclass
class Link:
    """One row of the alignment.  `src` / `trg` are tuples of sentence numbers (0-based); an empty tuple means the
    sentences on the other side have no counterpart.  `score` is 0 for such rows."""
    src: Tuple[int, ...]
    trg: Tuple[int, ...]
    score: float
    text: Optional[str] = field(default=None, repr=False, compare=False)   # the score as written in the .path file

    @property
    def score_text(self):
        return self.text if self.text is not None else str(self.score)

    @property
    def is_aligned(self):
        return bool(self.src and self.trg)

    @property
    def shape(self):
        return '%d:%d' % (len(self.src), len(self.trg))


@dataclass
class Alignment:
    links: List[Link]
    source: Sequence[str]
    target: Sequence[str]
    anchors: list = field(default_factory=list)       # [(i, j), ...] segment starts used (empty: solved whole)
    info: dict = field(default_factory=dict)          # timings and anchoring details

    def aligned(self):
        """The links that have sentences on both sides."""
        return [ln for ln in self.links if ln.is_aligned]

    def pairs(self, sep=' '):
        """[(source text, target text, score), ...] of the aligned links (the .aligned file)."""
        return [(sep.join(self.source[i] for i in ln.src), sep.join(self.target[j] for j in ln.trg), ln.score)
                for ln in self.aligned()]

    def to_path_lines(self):
        """The rows of the .path file: '[0,1]:[2]:0.7565561'."""
        return ['[%s]:[%s]:%s' % (','.join(map(str, ln.src)), ','.join(map(str, ln.trg)), ln.score_text)
                for ln in self.links]

    def to_dict(self):
        return {'links': [{'src': list(ln.src), 'trg': list(ln.trg), 'score': ln.score} for ln in self.links],
                'anchors': [list(a) for a in self.anchors], 'info': self.info}

    def __len__(self):
        return len(self.links)


_ROW = re.compile(r'\[([\d,]*):([\d,]*)\]')       # the form inside the aligner: '[0,1:2]'


def parse_path(path_text, score_text):
    """Links from the text the aligner produces: rows '[0,1:2]' (the .path file writes '[0,1]:[2]') and a score per row."""
    links = []
    for row, score in zip((x for x in path_text.split('\n') if x.strip()), (x for x in score_text.split('\n') if x.strip())):
        m = _ROW.match(row)
        s = tuple(int(x) for x in m.group(1).split(',') if x)
        t = tuple(int(x) for x in m.group(2).split(',') if x)
        links.append(Link(s, t, float(score), score.strip()))
    return links


class SentAlign:
    """A sentence aligner with the model loaded once.  Options as for the command line (see the module docstring)."""

    def __init__(self, config: Optional[AlignConfig] = None, **options):
        if config is None:
            names = {f.name for f in dataclasses.fields(AlignConfig)}
            kw = {}
            for k, v in options.items():
                k = _ALIASES.get(k, k)
                if k not in names:
                    raise TypeError('unknown option %r' % k)
                kw[k] = v
            kw.setdefault('num_proc', 1)
            config = AlignConfig(**kw)
        elif options:
            raise TypeError('give either a config or options, not both')
        config.resolve()
        self.config = config
        from sentalign.aligner import Aligner
        self._aligner = Aligner.for_memory(config)

    def citations(self, bibtex=False):
        """The publications to cite for alignments made with this configuration, as text (``bibtex=True`` adds the
        BibTeX entries).  Methods of other authors (encoder, Vecalign, chain ...) are listed with their papers."""
        from sentalign.citations import citations_for, format_citations
        return format_citations(citations_for(self.config), bibtex=bibtex)

    def align(self, source: Sequence[str], target: Sequence[str]) -> Alignment:
        """Align two lists of sentences."""
        return self.align_many([(source, target)])[0]

    def align_many(self, pairs) -> List[Alignment]:
        """Align several (source, target) pairs; with ``compat='none'`` their texts are encoded together, in groups
        of up to ``batch_texts`` texts (one group in memory at a time)."""
        from sentalign.legacy import align_anchors as legacy
        al = self._aligner
        cfg = self.config
        pairs = [(list(s), list(t)) for s, t in pairs]
        for k, (s, t) in enumerate(pairs):
            if not s or not t:
                raise ValueError('pair %d: empty source or target (%d, %d sentences)' % (k, len(s), len(t)))
        names = ['#%d' % k for k in range(len(pairs))]
        sides = dict(zip(names, pairs))
        al._prepared, al._group_end = {}, 0
        out = []
        for k, (s, t) in enumerate(pairs):
            al.info = type(al.info)()
            al.info.input_file = '#%d' % k
            al.timings = {}
            al._warned = set()
            legacy.reset_warnings()
            if k >= al._group_end:              # the next group of small documents, encoded together
                al._next_group(names, k, lambda name: al._make_sides(*sides[name]))
            path, scores = al.align_document(dict(enumerate(s)), dict(enumerate(t)), names[k])
            out.append(Alignment(parse_path(path, scores), s, t, [tuple(a) for a in al.info.anchors[:-1]],
                                 {'timings_seconds': dict(al.timings), 'anchoring': getattr(al, 'anchor_report', None)}))
        al._prepared = {}
        return out


_shared = {}


def align(source, target, **options):
    """Align two lists of sentences with a shared aligner (see the module docstring)."""
    # options may hold dicts (pathfinder_options): key from a canonical text form
    key = json.dumps(options, sort_keys=True, default=repr)
    if key not in _shared:
        _shared.clear()                     # keep only one model in memory
        _shared[key] = SentAlign(**options)
    return _shared[key].align(source, target)
