"""Small documents are encoded group by group (not all before the first is aligned), and a scorer without the batching
methods still aligns several documents."""
import numpy as np

from sentalign.aligner import Aligner
from sentalign.api import SentAlign
from sentalign.config import AlignConfig
from sentalign.document import Side
from sentalign.scoring import Scorer, register_scorer
from sentalign.scoring.embedding import EmbeddingScorer

from test_api import _fake_aligner
from test_scoring import FakeEncoder


def _docs(n):
    docs = []
    for k in range(n):
        s = ['Sentence %d of document %d, with some words in it.' % (i, k) for i in range(6 + k % 3)]
        t = ['Setning %d í skjali %d, með nokkrum orðum.' % (i, k) for i in range(6 + k % 3)]
        docs.append((s, t))
    return docs


class _Watched(EmbeddingScorer):
    """Records the group sizes and, when a document takes its vectors, the rows held for it and for later ones."""

    def __init__(self, encoder, aligner):
        super().__init__(encoder)
        self.aligner = aligner
        self.groups, self.retained = [], []

    def encode_sides(self, sides):
        self.groups.append(sum(s.n_rows for s in sides))
        return super().encode_sides(sides)

    def use_vectors(self, src, trg, src_vec, trg_vec):
        waiting = sum(len(sv) + len(tv) for _, _, sv, tv in self.aligner._prepared.values())
        self.retained.append(len(src_vec) + len(trg_vec) + waiting)
        super().use_vectors(src, trg, src_vec, trg_vec)


def _rows(docs, sa):
    k = sa.config.max_concatenations
    return [Side(s, max_k=k).n_rows + Side(t, max_k=k).n_rows for s, t in docs]


# ---------------------------------------------------------------------------- one group at a time
def test_align_many_keeps_one_group_of_vectors_at_a_time():
    docs = _docs(9)
    limit = 150
    sa = SentAlign(device='cpu', batch_texts=limit)
    sc = sa._aligner.scorer = _Watched(FakeEncoder(order='length', dedupe=True), sa._aligner)
    got = [r.to_path_lines() for r in sa.align_many(docs)]
    assert max(_rows(docs, sa)) * 2 <= limit < sum(_rows(docs, sa)) / 2      # several groups of several documents
    assert len(sc.groups) > 2 and len(sc.retained) == len(docs)
    assert max(sc.groups) <= limit and max(sc.retained) <= limit
    assert sa._aligner._prepared == {}
    one = _fake_aligner(batch_texts=0)
    assert got == [one.align(s, t).to_path_lines() for s, t in docs]


def test_a_document_above_the_limit_is_encoded_alone_and_does_not_end_the_group():
    docs = _docs(4)
    docs.insert(1, (['Long document, sentence %d here.' % i for i in range(60)],
                    ['Langt skjal, setning %d hér.' % i for i in range(60)]))
    sa = SentAlign(device='cpu', batch_texts=200)
    sc = sa._aligner.scorer = _Watched(FakeEncoder(order='length', dedupe=True), sa._aligner)
    rows = _rows(docs, sa)
    assert rows[1] > 200 and sum(rows) - rows[1] <= 200
    got = [r.to_path_lines() for r in sa.align_many(docs)]
    assert sc.groups == [sum(rows) - rows[1]] and len(sc.retained) == 4       # as before the fix: one group around it
    one = _fake_aligner(batch_texts=0)
    assert got == [one.align(s, t).to_path_lines() for s, t in docs]


def _corpus(tmp_path, docs, **options):
    for lang, column in (('eng', 0), ('isl', 1)):
        (tmp_path / lang).mkdir(parents=True)
        for k, doc in enumerate(docs):
            (tmp_path / lang / ('doc%02d.txt' % k)).write_text('\n'.join(doc[column]) + '\n', encoding='utf-8')
    cfg = AlignConfig(corpus_folder=str(tmp_path), source_language='eng', target_language='isl', proc_device='cpu',
                      num_proc=1, **options)
    cfg.resolve()
    return Aligner(cfg), ['doc%02d.txt' % k for k in range(len(docs))]


def _output(tmp_path, names):
    return [(tmp_path / 'output' / (n + '.path')).read_text(encoding='utf-8') for n in names]


def test_a_run_over_files_keeps_one_group_of_vectors_at_a_time(tmp_path):
    docs = _docs(9)
    limit = 150
    al, names = _corpus(tmp_path / 'together', docs, batch_texts=limit)
    sc = al.scorer = _Watched(FakeEncoder(order='length', dedupe=True), al)
    (tmp_path / 'together' / 'eng' / names[4]).write_text('', encoding='utf-8')       # an empty file fails on its own
    aligned, failed = al.run(names)
    assert failed == [names[4]] and len(aligned) == 8
    assert len(sc.groups) > 2 and len(sc.retained) == 8
    assert max(sc.groups) <= limit and max(sc.retained) <= limit
    assert al._prepared == {}

    one, _ = _corpus(tmp_path / 'one', docs, batch_texts=0)
    one.scorer = EmbeddingScorer(FakeEncoder(order='length', dedupe=True))
    one.run(aligned)
    assert _output(tmp_path / 'together', aligned) == _output(tmp_path / 'one', aligned)


def test_vectors_of_a_group_are_separate_arrays():
    """A view into one array for the whole group would keep the group in memory until its last document is done."""
    sides = [Side(['alpha beta', 'gamma'], max_k=2), Side(['delta', 'alpha beta'], max_k=2)]
    for vec in EmbeddingScorer(FakeEncoder(order='length', dedupe=True)).encode_sides(sides):
        assert vec.base is None


# ---------------------------------------------------------------------------- a scorer without batching
class _LengthScorer(Scorer):
    """Only what `Scorer` requires: `prepare` and `block`."""
    kind = 'length-scorer'
    needs_encoder = False

    def __init__(self, encoder=None, cache=None):
        pass

    def prepare(self, src, trg):
        self.src, self.trg = src, trg

    def block(self, ks, kt, i0, i1, j0, j1, out=None):
        a = np.array([self.src.word_counts[i:i + ks].sum() for i in range(i0, i1)], dtype=np.float32)
        b = np.array([self.trg.word_counts[j:j + kt].sum() for j in range(j0, j1)], dtype=np.float32)
        return (np.minimum.outer(a, b) / np.maximum(np.maximum.outer(a, b), 1)).astype(np.float32)


register_scorer('length-scorer', _LengthScorer)


def test_a_scorer_with_only_prepare_and_block_aligns_two_pairs():
    docs = _docs(2)
    sa = SentAlign(scorer='length-scorer', device='cpu')
    assert sa.config.batch_texts                                   # batching is on; the scorer just cannot do it
    results = sa.align_many(docs)
    for (s, t), res in zip(docs, results):
        assert sorted(i for ln in res.links for i in ln.src) == list(range(len(s)))
        assert sorted(j for ln in res.links for j in ln.trg) == list(range(len(t)))
    assert [r.to_path_lines() for r in results] == [sa.align(s, t).to_path_lines() for s, t in docs]


def test_a_scorer_with_only_prepare_and_block_aligns_two_files(tmp_path):
    al, names = _corpus(tmp_path, _docs(2), scorer='length-scorer')
    aligned, failed = al.run(names)
    assert aligned == names and failed == []
    assert all(_output(tmp_path, names))
