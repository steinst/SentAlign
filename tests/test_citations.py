"""What to cite for a run: SentAlign always, and the paper of every method of other authors that is used."""
from sentalign.citations import ENTRIES, citations_for, format_citations
from sentalign.config import AlignConfig


def _keys(**kw):
    return [k for _, k, _ in citations_for(AlignConfig(**kw))]


def test_default_run_cites_sentalign_and_labse():
    assert _keys() == ['sentalign2', 'sentalign', 'labse']
    assert 'placeholder' in ENTRIES['sentalign2']['text'].lower()


def test_methods_of_others_are_named():
    assert 'vecalign' in _keys(pathfinder='vecalign')
    assert 'chainalign' in _keys(pathfinder='chain')
    assert 'chainalign' in _keys(reward='squared-cosine')
    assert 'chainalign' not in _keys()
    assert _keys(pathfinder='galechurch', anchors='gale-church').count('galechurch') == 1       # once
    assert _keys(model_name='laser2')[2:] == ['laser', 'laser3']
    assert 'bge-m3' in _keys(model_name='bge-m3') and 'sbert-multilingual' in _keys(model_name='minilm')
    rows = citations_for(AlignConfig(model_name='someone/their-model'))
    assert rows[2][1] is None and 'someone/their-model' in rows[2][0]


def test_notice_text_and_bibtex():
    rows = citations_for(AlignConfig(pathfinder='vecalign'))
    plain = format_citations(rows)
    assert 'Thompson' in plain and '@inproceedings' not in plain and '--bibtex' in plain
    full = format_citations(rows, bibtex=True)
    assert full.count('@') == 4 and 'thompson-koehn-2019-vecalign' in full
    for entry in ENTRIES.values():
        assert entry['bibtex'].count('{') == entry['bibtex'].count('}')


def test_command_line_prints_the_notice(capsys, tmp_path):
    from sentalign import cli
    (tmp_path / 'eng').mkdir(); (tmp_path / 'isl').mkdir()
    assert cli.main(['-dir', str(tmp_path), '--pathfinder', 'chain', '--bibtex', '-q']) == 0     # nothing to align
    err = capsys.readouterr().err
    assert 'please cite' in err and 'ChainAlign' in err and '@inproceedings{goldner-gramazio-2026-chainalign' in err
    assert cli.main(['-dir', str(tmp_path), '--list-files']) == 0
    assert 'please cite' not in capsys.readouterr().err


def test_python_api_gives_the_notice():
    import sentalign
    assert hasattr(sentalign.SentAlign, 'citations')
