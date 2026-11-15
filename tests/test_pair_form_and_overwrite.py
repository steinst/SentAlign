"""The pair form keeps the languages, `sentalign.align` with dict options, a failed overwrite, --max-segment-nodes for
documents that are not cut."""
import os
import sys
import types

import numpy as np
import pytest

from sentalign import api, cli
from sentalign.aligner import Aligner
from sentalign.config import AlignConfig, add_alignment_arguments
from sentalign.scoring.embedding import EmbeddingScorer

from test_group_encoding import _corpus, _docs
from test_scoring import FakeEncoder


# ---------------------------------------------------------------------------- pair form
def _pair_args(tmp_path, *extra):
    import argparse
    parser = argparse.ArgumentParser()
    add_alignment_arguments(parser)
    parser.add_argument('pair', nargs='*')
    parser.add_argument('-o', '--output')
    src, trg = tmp_path / 's.txt', tmp_path / 't.txt'
    src.write_text('One.\nTwo.\n', encoding='utf-8')
    trg.write_text('Eitt.\nTvö.\n', encoding='utf-8')
    args = parser.parse_args(list(extra) + [str(src), str(trg), '-o', str(tmp_path / 'out')])
    args.list_files = False
    from sentalign.config import config_from_args
    return config_from_args(args), args


def test_pair_form_keeps_the_languages_and_names_the_folders(tmp_path, monkeypatch):
    seen = {}

    class Fake:
        def __init__(self, config, **kw):
            seen['cfg'], seen['kw'] = config, kw

        def run(self, names):
            return names, []

    import sentalign.aligner
    monkeypatch.setattr(sentalign.aligner, 'Aligner', Fake)
    config, args = _pair_args(tmp_path, '-sl', 'deu', '-tl', 'fra')
    assert cli._run_pair(cli.build_parser(), config, args) == 0
    assert (seen['cfg'].source_language, seen['cfg'].target_language) == ('deu', 'fra')
    assert os.path.basename(seen['kw']['source_folder']) == 'src' and os.path.basename(seen['kw']['target_folder']) == 'trg'


# (an encoder that needs the language, -sl / -tl not given: refused, see test_output_files_and_options.py)


# ---------------------------------------------------------------------------- dict options
def test_align_with_pathfinder_options_dict(monkeypatch):
    made = []

    class Stub:
        def __init__(self, **options):
            made.append(options)

        def align(self, s, t):
            return 'ok'

    monkeypatch.setattr(api, 'SentAlign', Stub)
    monkeypatch.setattr(api, '_shared', {})
    for _ in range(2):
        assert api.align(['a'], ['b'], pathfinder_options={'x': 1, 'y': 2}) == 'ok'
    assert len(made) == 1                                       # the same options: the aligner is shared
    api.align(['a'], ['b'], pathfinder_options={'x': 2})
    assert len(made) == 2


# ---------------------------------------------------------------------------- failed overwrite
def test_failed_overwrite_leaves_no_done_file_with_mixed_content(tmp_path, monkeypatch):
    al, names = _corpus(tmp_path, _docs(1), overwrite=True)
    al.scorer = EmbeddingScorer(FakeEncoder(order='length', dedupe=True))
    assert al.run(names) == (names, [])
    out = tmp_path / 'output'
    old_path = (out / (names[0] + '.path')).read_text(encoding='utf-8')
    old_pairs = (out / (names[0] + '.aligned')).read_text(encoding='utf-8')

    def boom(*a, **k):
        raise OSError('disk full')

    # 1. the failure comes while the new files are written: the old pair is untouched
    monkeypatch.setattr(Aligner, 'write_pairs_to_file', boom)
    aligned, failed = al.run(names)
    assert failed == names
    assert (out / (names[0] + '.path')).read_text(encoding='utf-8') == old_path
    assert (out / (names[0] + '.aligned')).read_text(encoding='utf-8') == old_pairs
    assert not [f for f in os.listdir(out) if '.new' in f or f.endswith('.tmp')]

    # 2. the failure comes between the renames: not done (no .aligned), never a new .path with old pairs
    monkeypatch.undo()
    real = os.replace
    n = {'k': 0}

    def replace(a, b):
        n['k'] += 1
        if str(b).endswith('.aligned'):
            raise OSError('gone')
        return real(a, b)

    monkeypatch.setattr(os, 'replace', replace)
    aligned, failed = al.run(names)
    monkeypatch.undo()
    assert failed == names
    from sentalign import files
    assert not files.is_done(str(out), names[0])


# ---------------------------------------------------------------------------- segment limit, not cut
@pytest.mark.parametrize('anchors', ['none', 'auto'])
def test_max_segment_nodes_applies_to_documents_that_are_not_cut(anchors):
    src = ['The cat sat on mat number %d.' % i for i in range(6)]
    trg = ['Kottur sat a mottu nr %d.' % i for i in range(6)]
    sa = _fake(anchors=anchors, max_segment_nodes=10)
    with pytest.raises(MemoryError, match='max-segment-nodes'):
        sa.align(src, trg)
    assert _fake(anchors=anchors, max_segment_nodes=10, pathfinder='vecalign').align(src, trg).links


def _fake(**options):
    from test_api import _fake_aligner
    return _fake_aligner(**options)


def test_max_segment_nodes_is_not_applied_under_paper_2023():
    src = ['The cat sat on mat number %d.' % i for i in range(6)]
    trg = ['Kottur sat a mottu nr %d.' % i for i in range(6)]
    assert _fake(compat='paper-2023', max_segment_nodes=10).align(src, trg).links
