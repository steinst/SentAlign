"""Gale-Church from the galechurch package.

* The beads equal those of the Cython implementation that SentAlign 1.0 carried: saved as golden files
  (tests/golden/galechurch_v1; the four small books through SENTALIGN_BOOK_FIXTURES).
* Blank lines: version 1 cut the file at blank lines and numbered each block from 0.
* Gale-Church as a path finder (--pathfinder galechurch, --compat none).
"""
import glob
import os

import numpy as np
import pytest

from sentalign.gale_church import anchor_candidates, gale_church_beads
from sentalign.pathfinding import LINK, PathParams, get_pathfinder
from sentalign.pathfinding.segments import make_segment
from sentalign.document import Side
from sentalign.scoring.embedding import EmbeddingScorer

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SETS = {'bleualign': ('deu', 'fra'), 'parice': ('eng', 'isl')}


def _lines(path):
    with open(path, encoding='utf-8') as f:
        return [line.strip() for line in f]


def _pharaoh(beads):
    return ['[%s]:[%s]' % (','.join(map(str, range(s0, s1))), ','.join(map(str, range(t0, t1)))) for s0, s1, t0, t1 in beads]


def _golden(path):
    """The beads of the 2023 implementation that link sentences on both sides (it also listed insertions and
    deletions, '[3]:[]' and '[]:[4]')."""
    with open(path, encoding='utf-8') as f:
        return [line.strip() for line in f if line.strip() and not line.startswith('[]') and not line.strip().endswith(':[]')]


def test_beads_equal_the_version1_implementation_on_the_gold_sets():
    n = 0
    for name, (sl, tl) in SETS.items():
        for src in sorted(glob.glob(os.path.join(ROOT, 'eval_data', name, sl, '*.txt'))):
            fn = os.path.basename(src)
            trg = os.path.join(ROOT, 'eval_data', name, tl, fn)
            got = _pharaoh(gale_church_beads(_lines(src), _lines(trg)))
            assert got == _golden(os.path.join(ROOT, 'tests', 'golden', 'galechurch_v1', name, fn + '.beads')), fn
            n += 1
    assert n == 17


def test_beads_equal_the_version1_implementation_on_small_books():
    base = os.environ.get('SENTALIGN_BOOK_FIXTURES')
    if not base:
        pytest.skip('needs_fixtures: SENTALIGN_BOOK_FIXTURES not set (book texts are outside the repository)')
    files = sorted(glob.glob(os.path.join(base, 'cap_small', 'eng', '*.txt')))
    assert len(files) == 4
    for src in files:
        fn = os.path.basename(src)
        got = _pharaoh(gale_church_beads(_lines(src), _lines(os.path.join(base, 'cap_small', 'isl', fn))))
        assert got == _golden(os.path.join(base, 'gc_golden', fn + '.beads')), fn


def test_beads_cover_every_sentence_once_in_order():
    src = ['Short.', 'A somewhat longer sentence here.', 'x', 'Another one follows right after that.', 'End.']
    trg = ['Kurz.', 'Ein etwas laengerer Satz hier.', 'Noch einer folgt direkt danach.', 'Ende.', 'Extra']
    beads = gale_church_beads(src, trg)
    s = t = 0
    for s0, s1, t0, t1 in beads:
        assert (s0, t0) == (s, t)
        s, t = s1, t1
    assert (s, t) == (len(src), len(trg))


def test_anchor_candidates_format_and_only_two_sided_beads():
    src = ['aaaa bbbb', 'cccc', 'dddd eeee ffff']
    trg = ['aaaa bbbb', 'cccc dddd', 'eeee ffff gggg']
    for s, t in anchor_candidates(src, trg):
        assert s and t and all(x.isdigit() for x in (s + ',' + t).split(','))


def test_blank_lines_are_left_out_and_numbers_stay_global():
    """A blank line is a zero-length sentence and Gale-Church gives two empty sides the cost minus infinity, which
    wrecks the alignment; version 1 cut the file at blank lines and numbered every block from 0.
    Now: blank lines left out, one block, numbers as in the file."""
    src = ['Eins zwei drei vier.', 'Fuenf sechs sieben.', '', 'Acht neun zehn elf zwoelf.', 'Dreizehn vierzehn.']
    trg = ['One two three four.', 'Five six seven.', '', 'Eight nine ten eleven twelve.', 'Thirteen fourteen.']
    assert gale_church_beads(src, trg) == [(0, 1, 0, 1), (1, 2, 1, 2), (3, 4, 3, 4), (4, 5, 4, 5)]
    # without the blank lines the same beads, shifted
    s2 = [x for x in src if x]
    t2 = [x for x in trg if x]
    assert gale_church_beads(s2, t2) == [(0, 1, 0, 1), (1, 2, 1, 2), (2, 3, 2, 3), (3, 4, 3, 4)]
    assert anchor_candidates(src, trg)[2] == ['3', '3']


def test_a_bead_over_a_blank_line_is_dropped_and_empty_sides_give_no_beads():
    src = ['aaaa bbbb cccc dddd', '', 'eeee ffff']
    trg = ['aaaa bbbb cccc dddd eeee ffff']
    for s0, s1, t0, t1 in gale_church_beads(src, trg):
        assert all(src[i] for i in range(s0, s1))
    assert gale_church_beads([], ['x']) == [] and gale_church_beads(['x'], ['']) == []


# ---------------------------------------------------------------------------- as a path finder
def _scorer(src, trg):
    sc = EmbeddingScorer(encoder=None)
    sc.src, sc.trg = Side(src), Side(trg)
    rng = np.random.default_rng(1)
    sc.src_emb = rng.normal(size=(sc.src.n_rows, 8)).astype(np.float32)
    sc.trg_emb = rng.normal(size=(sc.trg.n_rows, 8)).astype(np.float32)
    return sc


def test_galechurch_path_finder_returns_the_beads_as_steps():
    src = ['Short.', 'A somewhat longer sentence here.', 'Third one, which is quite long as well.', 'End.']
    trg = ['Kurz.', 'Ein etwas laengerer Satz hier.', 'Der dritte, der auch ziemlich lang ist.', 'Ende.']
    sc = _scorer(src, trg)
    finder = get_pathfinder('galechurch')(PathParams())
    assert finder.postprocess is False
    steps = finder.find(make_segment(sc, sc.src, sc.trg, 0, 4, 0, 4))
    assert steps == [(LINK, i, 1, i, 1) for i in range(4)]
    with pytest.raises(ValueError):
        finder.find(type('S', (), {'src_sentences': None, 'trg_sentences': None})())


def test_paper_mode_refuses_another_pathfinder():
    from sentalign.config import AlignConfig
    with pytest.raises(ValueError):
        AlignConfig(compat='paper-2023', pathfinder='galechurch')
    assert AlignConfig(compat='none', pathfinder='galechurch').pathfinder == 'galechurch'


@pytest.mark.slow
@pytest.mark.gpu_or_cpu
def test_galechurch_baseline_end_to_end(tmp_path, device, labse_available, repo_root):
    import re
    import shutil
    from conftest import run_cli
    corpus = tmp_path / 'bleualign'
    for lang in ('deu', 'fra'):
        shutil.copytree(os.path.join(repo_root, 'eval_data', 'bleualign', lang), corpus / lang)
    r = run_cli(['-dir', corpus, '-sl', 'deu', '-tl', 'fra', '-device', device, '--compat', 'none',
                 '--pathfinder', 'galechurch', '-proc', '2'])
    assert r.returncode == 0, r.stderr[-3000:]
    for fn in sorted(os.listdir(corpus / 'deu')):
        gold = _golden(os.path.join(repo_root, 'tests', 'golden', 'galechurch_v1', 'bleualign', fn + '.beads'))
        got = [line.rsplit(':', 1)[0] for line in open(corpus / 'output' / (fn + '.path'), encoding='utf-8')]
        # same links as Gale-Church; unaligned sentences are written one per row
        assert [g for g in got if '[]' not in g] == gold, fn
