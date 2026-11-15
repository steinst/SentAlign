# Extending SentAlign

SentAlign 2 has four parts that can be exchanged without touching the rest. Each is a small class; register it with a
function call or, to ship it in your own package, through an *entry point*.

| Part | Question it answers | Interface | Command line |
|---|---|---|---|
| Encoder | texts → vectors | `encode(texts) -> (n, d) float32` | `--encoder`, `--encoder-kind` |
| Scorer | how similar are these spans? | `prepare`, `block`, `pair` | `--scorer` |
| Path finder | which links are best? | `find(segment) -> steps` | `--pathfinder` |
| Anchor strategy | where can a long document be cut? | `find(src, trg, scorer, max_nodes) -> anchors` | `--anchors` |

Entry-point groups: `sentalign.encoders`, `sentalign.scorers`, `sentalign.pathfinders`, `sentalign.anchorers`. In the
`pyproject.toml` of your package:

```toml
[project.entry-points."sentalign.scorers"]
length = "mypackage.scorers:LengthScorer"
```

## A different embedding model

For a fine-tuned LaBSE or any Hugging Face encoder no code is needed:

```bash
sentalign -dir corpus -sl eng -tl isl --encoder /models/labse-finetuned
sentalign -dir corpus -sl eng -tl isl --encoder sentence-transformers/paraphrase-multilingual-mpnet-base-v2 --pooling mean
```

The score cutoff (`-cutoff`, 0.4) was tuned for LaBSE; other models have other score ranges and usually need their own.

## A scorer

A scorer answers for blocks of spans: `block(ks, kt, i0, i1, j0, j1)` returns a float32 array of shape
`(i1 - i0, j1 - j0)`, the score of every `ks`-sentence source span starting in `[i0, i1)` against every `kt`-sentence
target span starting in `[j0, j1)`. `NaN` means "no score". This one scores by length ratio, no model needed:

```python
import numpy as np
from sentalign.scoring import Scorer, register_scorer

class LengthScorer(Scorer):
    kind = "length"
    needs_encoder = False                                  # no model is loaded for this scorer

    def __init__(self, encoder=None, cache=None):          # the pipeline passes these two
        pass

    def prepare(self, src, trg):                           # src and trg are sentalign.document.Side objects
        self.src, self.trg = src, trg

    def block(self, ks, kt, i0, i1, j0, j1, out=None):
        a = np.array([self.src.word_counts[i:i + ks].sum() for i in range(i0, i1)], dtype=np.float32)
        b = np.array([self.trg.word_counts[j:j + kt].sum() for j in range(j0, j1)], dtype=np.float32)
        return (np.minimum.outer(a, b) / np.maximum(np.maximum.outer(a, b), 1)).astype(np.float32)

register_scorer("length", LengthScorer)
```

```python
import sentalign
result = sentalign.SentAlign(scorer="length", device="cpu").align(src_sentences, trg_sentences)
```

`prepare` and `block` are all a scorer needs. Optional: a scorer that also has `encode_sides(sides)` (a list of
`Side` objects of many small documents; returns one piece of data per side, in the same order) and
`use_vectors(src, trg, src_data, trg_data)` (does what `prepare` does, with the data `encode_sides` returned for these
two sides) is given the small documents of a run in groups of up to `--batch-texts` texts, as the embedding scorer
is (fuller encoder batches, repeated texts encoded once). A scorer without these two methods gets every document
through `prepare`.

(In your own package use the entry point instead of calling `register_scorer`; from the command line the scorer must
be importable and registered before `sentalign` runs, which the entry point takes care of.)

## A path finder

A path finder gets a `Segment` (the part of a document between two anchors) and returns the links. The segment gives
it `n_src`, `n_trg`, `max_k`, `score_block(ks, kt, p0, p1)` (the scores, rows = source start, columns = target start),
`src_words` / `trg_words` and the sentence texts. Steps are `(kind, src_start, src_size, trg_start, trg_size)`; only
`LINK` steps are used, sentences without a link are unaligned. With `postprocess = True` (the default) SentAlign
re-evaluates and extends the links afterwards; set it to `False` if your finder's result should stand.

```python
import numpy as np
from sentalign.pathfinding import LINK, PathFinder, register_pathfinder

class GreedyOneToOne(PathFinder):
    """Link source sentence i to the best later target sentence, never going back."""
    name = "greedy-1-1"
    postprocess = False

    def find(self, segment, stop_node=None):
        scores = segment.score_block(1, 1, 0, segment.n_src)      # (n_src, n_trg)
        steps, j0 = [], 0
        for i in range(segment.n_src):
            if j0 >= segment.n_trg:
                break
            j = j0 + int(np.argmax(np.nan_to_num(scores[i, j0:], nan=-1.0)))
            if scores[i, j] >= self.params.score_cutoff:
                steps.append((LINK, i, 1, j, 1))
                j0 = j + 1
        return steps

register_pathfinder("greedy-1-1", GreedyOneToOne)
```
`sentalign ... --compat none --pathfinder greedy-1-1`. A finder can have settings of its own: list them with their
defaults in the class attribute `option_defaults`, read them from `self.options`, and set them with
`--pathfinder-opt NAME=VALUE`. A finder that works on vectors instead of scores reads `segment.src_vectors(k)` and
`segment.trg_vectors(k)` (the vectors of the k-sentence spans starting at each sentence; `None` with a scorer that
has none); one that needs no anchors sets `uses_anchors = False`. `sentalign/pathfinding/vecalign_finder.py` is an
example of all three. Other algorithms are only available in the default
(`--compat none`) mode, because `paper-2023` is by definition the 2023 path search.

## An anchor strategy

An anchor `(i, j)` says that source sentence `i` and target sentence `j` begin a new segment. The strategy gets the two
`Side` objects, the scorer and the node limit, and returns strictly increasing anchors (or `[]`). A wrong anchor is the
worst mistake the aligner can make, so return only anchors you are sure of; `sentalign.anchors.base` has the helpers
SentAlign's own strategies use (`longest_chain`, `supported`, `choose`).

```python
from sentalign.anchors import AnchorStrategy, register_anchorer

class EveryThousand(AnchorStrategy):
    name = "every-thousand"
    def find(self, src, trg, scorer, max_nodes):
        n, m = len(src), len(trg)
        k = min(n, m) // 1000
        return [(round(n * c / (k + 1)), round(m * c / (k + 1))) for c in range(1, k + 1)]   # only for equal-length texts!

register_anchorer("every-thousand", EveryThousand)
```

## Reproducing the 2023 paper

`--compat paper-2023` (`compat='paper-2023'` in Python) gives the alignments of SentAlign 1.0 (tag `v1.0-paper`): the
same arithmetic (float32 sums as under NumPy 2), the same embedding lookups, the 2023 re-evaluation of links with its
faults (three sentences of the German-French evaluation set are aligned twice), and the greedy anchoring of version 1
for files above `-maxgc` sentences. On the evaluation sets the `.path` and `.aligned` files are byte-identical on the
GPU the golden files were made on, and identical in alignments with scores within 1e-5 elsewhere. It is a reference,
not a recommendation; only a few options can be changed in this mode (the encoder, `--pathfinder` and `--anchors` other
than the 2023 ones are refused or reported as deviations).
