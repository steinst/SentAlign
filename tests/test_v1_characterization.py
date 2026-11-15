"""Characterisation tests for the 2023 code (tag v1.0-paper).

They replay the path search and the post-processing of the unchanged `align_anchors.py` from saved sentences and
saved LaBSE embeddings (no GPU, no model) and compare the result with the golden `.path` files that the unchanged
program wrote on the GPU.  When the new implementation exists, the same fixtures and the same expectations are used
for its `--compat paper-2023` mode.

The module under test is now the moved legacy module `sentalign.legacy.align_anchors` (logic unchanged).

Fixtures (captured from the unchanged 2023 program):
  tests/fixtures/eval/<set>/embeddings/<file>.{src,trg}.overlaps    concatenation layers, one string per line
  tests/fixtures/eval/<set>/embeddings/<file>.{src,trg}.labse_emb   embeddings of those lines (float32, normalised)
  tests/fixtures/eval/<set>/stages/<file>__<anchors>.json            intermediate results of every stage
  tests/golden/v1_gpu/<set>/<file>.path                              output of the unchanged program
The full set (tests/fixtures/eval/, 40 MB, not in git) is downloaded with tests/fetch_fixtures.py; without it the
three smallest files per set in tests/fixtures/mini/ (in git) are replayed.
Book fixtures are large and live outside the repository: set SENTALIGN_BOOK_FIXTURES to
  the folder that holds them (uses capture/books_small, cap_small/{eng,isl,output}).

Run:  python tests/test_v1_characterization.py        (or: python -m pytest tests/test_v1_characterization.py)
"""
import glob
import json
import os
import sys

import numpy as np

try:
    import pytest
except ImportError:  # run as a plain script in an environment without pytest
    pytest = None

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "src"))
from sentalign.legacy import align_anchors as legacy  # noqa: E402

EVAL_SETS = {'bleualign': ('deu', 'fra'), 'parice': ('eng', 'isl')}
DEFAULTS = dict(score_cutoff=0.4, max_concats=4, minimum_length_words=1, maximum_length_words=110,
                start_penalty_word_number=80, penalty_per_word=0.01, free_concats=2)


def _skip(message):
    """Skip under pytest; return the message when run as a script."""
    if pytest is not None and 'pytest' in sys.modules and os.environ.get('PYTEST_CURRENT_TEST'):
        pytest.skip(message)
    return 'skipped (' + message + ')'


def _note(message):
    """Information line for the script runner; None under pytest (tests must not return values)."""
    return None if os.environ.get('PYTEST_CURRENT_TEST') else message


def fixture_root(name):
    """tests/fixtures/eval if present (full set), else tests/fixtures/mini (three files per set), else None."""
    for sub_folder in ('eval', 'mini'):
        root = os.path.join(ROOT, 'tests', 'fixtures', sub_folder, name)
        if glob.glob(os.path.join(root, 'stages', '*.json')):
            return root
    return None


class _NoProgress:
    def add_nodes(self, n):
        pass


def read_sentences(path):
    """As sentAlign.read_sentences: position -> stripped line."""
    with open(path, encoding='utf-8') as f:
        return {i: line.strip() for i, line in enumerate(f)}


def read_embeddings(base):
    """As sentAlign.open_emb_file: stripped text of an overlap line -> its embedding."""
    lines = open(base + '.overlaps', encoding='utf-8').readlines()
    embs = np.load(base + '.labse_emb', allow_pickle=True)
    assert len(lines) == len(embs), base
    return {lines[i].strip(): embs[i] for i in range(len(lines))}


def to_path_file(path, scores):
    """As sentAlign.write_path_to_file."""
    path = path.replace(':', ']:[').strip()
    return ''.join(p + ':' + s + '\n' for p, s in zip(path.split('\n'), scores.split('\n')))


def replay(src_file, trg_file, emb_base, stage_files):
    src, trg = read_sentences(src_file), read_sentences(trg_file)
    src_emb, trg_emb = read_embeddings(emb_base + '.src'), read_embeddings(emb_base + '.trg')
    total_path = total_scores = ''
    mismatches = []
    for sf in stage_files:
        st = json.load(open(sf, encoding='utf-8'))
        anchor = (tuple(st['anchor'][0]), tuple(st['anchor'][1]))
        p = st['params']
        _, path, scores = legacy.align_anchors(anchor, src, trg, src_emb, trg_emb, p['score_cutoff'], p['max_concats'],
                                               p['minimum_length_words'], p['maximum_length_words'],
                                               p['start_penalty_word_number'], p['penalty_per_word'], p['free_concats'],
                                               _NoProgress())
        if path != st['final_path']:
            mismatches.append(os.path.basename(sf))
        total_path += path
        total_scores += scores
    return to_path_file(total_path, total_scores), mismatches


def _stage_files(folder, name):
    files = glob.glob(os.path.join(folder, name + '__*.json'))
    return sorted(files, key=lambda f: int(f.rsplit('__', 2)[1].split('_')[0]))


def _alignments(text):
    return [line.rsplit(':', 1)[0] for line in text.splitlines()]


def _scores(text):
    return np.array([float(line.rsplit(':', 1)[1]) for line in text.splitlines()])


def check_eval_set(name):
    sl, tl = EVAL_SETS[name]
    fx = fixture_root(name)
    if fx is None:
        return None
    files = sorted(os.path.basename(f) for f in glob.glob(os.path.join(ROOT, 'eval_data', name, sl, '*.txt')))
    assert files, 'no evaluation files for ' + name
    files = [fn for fn in files if _stage_files(os.path.join(fx, 'stages'), fn)]
    for fn in files:
        got, mism = replay(os.path.join(ROOT, 'eval_data', name, sl, fn), os.path.join(ROOT, 'eval_data', name, tl, fn),
                           os.path.join(fx, 'embeddings', fn), _stage_files(os.path.join(fx, 'stages'), fn))
        gold = open(os.path.join(ROOT, 'tests', 'golden', 'v1_gpu', name, fn + '.path'), encoding='utf-8').read()
        assert not mism, (fn, mism)
        assert _alignments(got) == _alignments(gold), fn + ': alignments differ from the golden file'
        assert np.allclose(_scores(got), _scores(gold), atol=1e-6), fn + ': scores differ from the golden file'
        # Byte identity also depends on the BLAS library behind numpy (last digit of float32 dot products), so it
        # is only required on the reference machine, where the golden files were made: SENTALIGN_STRICT_BYTES=1.
        if os.environ.get('SENTALIGN_STRICT_BYTES') == '1':
            assert got == gold, fn + ': not byte-identical to the golden file'
    return len(files)


def _eval_test(name):
    n = check_eval_set(name)
    if n is None:
        return _skip('no fixtures: run tests/fetch_fixtures.py or restore tests/fixtures/mini')
    kind = 'full set' if fixture_root(name).split(os.sep)[-2] == 'eval' else 'mini set'
    return _note('%d files (%s)' % (n, kind))


def test_bleualign_replay_matches_golden():
    return _eval_test('bleualign')


def test_parice_replay_matches_golden():
    return _eval_test('parice')


def test_small_books_replay_matches_golden():
    base = os.environ.get('SENTALIGN_BOOK_FIXTURES')
    if not base:
        return _skip('needs_fixtures: SENTALIGN_BOOK_FIXTURES not set (large book fixtures, outside the repository)')
    n = 0
    for src_file in sorted(glob.glob(os.path.join(base, 'cap_small', 'eng', '*.txt'))):
        fn = os.path.basename(src_file)
        got, mism = replay(src_file, os.path.join(base, 'cap_small', 'isl', fn),
                           os.path.join(base, 'capture', 'books_small', 'embeddings', fn),
                           _stage_files(os.path.join(base, 'capture', 'books_small', 'stages'), fn))
        gold = open(os.path.join(base, 'cap_small', 'output', fn + '.path'), encoding='utf-8').read()
        assert not mism, (fn, mism)
        assert got == gold, fn
        n += 1
    return _note('%d books' % n)


def test_known_defects_are_in_the_golden_files():
    """The 2023 output aligns three sentences of the German-French set twice.  The compat mode has to
    keep that; the default mode of version 2 must not."""
    def twice(path_file):
        seen = {}
        for line in open(path_file, encoding='utf-8'):
            s, t, _ = line.strip().replace('[', '').replace(']', '').split(':')
            if s and t:
                for side, ids in (('s', s), ('t', t)):
                    for i in ids.split(','):
                        seen[side, int(i)] = seen.get((side, int(i)), 0) + 1
        return sorted(k for k, v in seen.items() if v > 1)
    g = os.path.join(ROOT, 'tests', 'golden', 'v1_gpu', 'bleualign')
    assert twice(os.path.join(g, 'test0.txt.path')) == [('t', 99)]
    assert twice(os.path.join(g, 'test6.txt.path')) == [('s', 4), ('t', 4)]


if __name__ == '__main__':
    failed = 0
    for name, fn in sorted(globals().items()):
        if name.startswith('test_'):
            try:
                r = fn()
                print('ok    ', name, r or '')
            except AssertionError as e:
                failed += 1
                print('FAILED', name, e)
    sys.exit(1 if failed else 0)
