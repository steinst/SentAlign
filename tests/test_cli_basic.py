"""Command line behaviour: help, --list-files, exit codes, -f, skipping, --overwrite, cleanup, no list files."""
import json
import os

import pytest

import sentalign

SRC_LINES = ['Das ist der erste Satz.', 'Hier kommt der zweite Satz.', 'Und noch ein dritter Satz.']
TRG_LINES = ['Ceci est la première phrase.', 'Voici la deuxième phrase.', 'Et encore une troisième phrase.',
             'Une quatrième phrase de plus.']


def make_corpus(root, names=('a.txt', 'b.txt')):
    for lang, lines in (('deu', SRC_LINES), ('fra', TRG_LINES)):
        os.makedirs(root / lang, exist_ok=True)
        for n in names:
            (root / lang / n).write_text('\n'.join(lines) + '\n', encoding='utf-8')
    return root


def test_help(cli):
    out = cli(['--help'])
    assert out.returncode == 0
    for flag in ('--corpus-folder', '-dir', '-sl', '-tl', '--filename', '-f', '-out', '-concats', '-freejoins', '-cutoff',
                 '-minwords', '-maxwords', '-penwords', '-wordpen', '-anchor', '-maxgc', '-device', '-proc',
                 '--compat', '--list-files', '--overwrite', '--quiet', '--verbose', 'SRC_FILE'):
        assert flag in out.stdout, flag
    assert '--num_overlaps' not in out.stdout and "'-n'" not in out.stdout


def test_version(cli):
    out = cli(['--version'])
    assert out.returncode == 0 and sentalign.__version__ in out.stdout


def test_usage_errors_exit_2(cli, tmp_path):
    assert cli([]).returncode == 2
    assert cli(['-dir', tmp_path / 'missing']).returncode == 2
    corpus = make_corpus(tmp_path / 'c')
    assert cli(['-dir', corpus, '-sl', 'deu', '-tl', 'nope']).returncode == 2       # target folder missing
    assert cli(['-dir', corpus, '-sl', 'deu', '-tl', 'fra', '-proc', 'many']).returncode == 2
    assert cli(['-dir', corpus, '-sl', 'deu', '-tl', 'fra', '--compat', 'other']).returncode == 2
    assert cli([str(corpus / 'deu' / 'a.txt')]).returncode == 2                     # one file only
    assert cli([str(corpus / 'deu' / 'a.txt'), str(corpus / 'fra' / 'a.txt')]).returncode == 2  # no -o
    assert cli([str(corpus / 'deu' / 'nope.txt'), str(corpus / 'fra' / 'a.txt'), '-o', tmp_path / 'o']).returncode == 2


def test_list_files_sorted_and_f_selects(cli, tmp_path):
    corpus = make_corpus(tmp_path / 'c', names=('c.txt', 'a.txt', 'b.txt'))
    base = ['-dir', corpus, '-sl', 'deu', '-tl', 'fra']
    out = cli(base + ['--list-files'])
    assert out.returncode == 0 and out.stdout.split() == ['a.txt', 'b.txt', 'c.txt']
    out = cli(base + ['--list-files', '-f', 'c.txt', 'a.txt'])
    assert out.stdout.split() == ['a.txt', 'c.txt']
    # --list-files creates nothing
    assert sorted(os.listdir(corpus)) == ['deu', 'fra']


def test_finished_files_are_skipped_and_overwrite_lists_them(cli, tmp_path):
    corpus = make_corpus(tmp_path / 'c')
    out_dir = corpus / 'output'
    out_dir.mkdir()
    for n in ('a.txt', 'b.txt'):
        (out_dir / (n + '.path')).write_text('x')
        (out_dir / (n + '.aligned')).write_text('x')
    (out_dir / 'c.txt.path').write_text('x')   # only half done: counts as not done
    base = ['-dir', corpus, '-sl', 'deu', '-tl', 'fra']
    assert cli(base + ['--list-files']).stdout.split() == []
    assert cli(base + ['--list-files', '--overwrite']).stdout.split() == ['a.txt', 'b.txt']
    out = cli(base)  # nothing to do: exits 0 without loading the model, creates no tmp folder or list files
    assert out.returncode == 0
    assert 'Nothing to align' in out.stderr
    assert sorted(os.listdir(corpus)) == ['deu', 'fra', 'output']
    assert (out_dir / 'a.txt.path').read_text() == 'x'


def test_failing_file_gives_exit_1_and_is_not_marked_done(cli, tmp_path):
    corpus = make_corpus(tmp_path / 'c', names=('a.txt',))
    out = cli(['-dir', corpus, '-sl', 'deu', '-tl', 'fra', '-f', 'missing.txt', '-device', 'cpu'])
    assert out.returncode == 1, out.stderr
    assert 'missing.txt' in out.stderr and 'FAILED' in out.stderr
    assert not (corpus / 'output' / 'missing.txt.path').exists()
    assert not (corpus / 'output' / 'missing.txt.aligned').exists()
    assert [d for d in os.listdir(corpus / 'tmp')] == []          # temporary folder removed
    assert not (corpus / 'files2align.txt').exists() and not (corpus / 'filesdone.txt').exists()


@pytest.mark.gpu_or_cpu
@pytest.mark.slow
def test_run_skip_overwrite_cleanup(cli, tmp_path, device, labse_available):
    # a.txt has 3 source and 4 target lines (different line counts are fine), b.txt is never selected
    corpus = make_corpus(tmp_path / 'c')
    base = ['-dir', corpus, '-sl', 'deu', '-tl', 'fra', '-device', device, '-proc', '2']
    out = cli(base + ['-f', 'a.txt'])
    assert out.returncode == 0, out.stderr
    out_dir = corpus / 'output'
    assert sorted(os.listdir(out_dir)) == ['a.txt.aligned', 'a.txt.json', 'a.txt.path']   # -f selects, no .tmp left
    assert (corpus / 'align_info' / 'a.txt.info').exists()
    assert os.listdir(corpus / 'tmp') == []
    assert not (corpus / 'files2align.txt').exists() and not (corpus / 'filesdone.txt').exists()
    record = json.loads((out_dir / 'a.txt.json').read_text())
    assert record['compat'] == 'none' and record['model'] == 'setu4993/LaBSE'
    assert record['counts']['source_lines'] == 3 and record['counts']['target_lines'] == 4
    for key in ('settings', 'anchors', 'timings_seconds', 'versions'):
        assert key in record
    for key in ('python', 'numpy', 'torch', 'transformers', 'sentalign'):
        assert record['versions'][key]
    # every sentence occurs in the path
    path = (out_dir / 'a.txt.path').read_text().splitlines()
    assert len(path) >= 4

    # second run over the whole folder: a.txt is skipped, b.txt is aligned
    stamp = (out_dir / 'a.txt.path').stat().st_mtime_ns
    out = cli(base)
    assert out.returncode == 0, out.stderr
    assert (out_dir / 'a.txt.path').stat().st_mtime_ns == stamp
    assert (out_dir / 'b.txt.path').exists() and (out_dir / 'b.txt.aligned').exists()
    assert (out_dir / 'a.txt.path').read_text() == (out_dir / 'b.txt.path').read_text()

    # --overwrite aligns a.txt again
    out = cli(base + ['--overwrite', '-f', 'a.txt'])
    assert out.returncode == 0, out.stderr
    assert (out_dir / 'a.txt.path').stat().st_mtime_ns > stamp
    assert os.listdir(corpus / 'tmp') == []


@pytest.mark.gpu_or_cpu
@pytest.mark.slow
def test_single_pair_form(cli, tmp_path, device, labse_available):
    make_corpus(tmp_path / 'c', names=('a.txt',))
    scratch = tmp_path / 'scratch'
    scratch.mkdir()
    out = cli([tmp_path / 'c' / 'deu' / 'a.txt', tmp_path / 'c' / 'fra' / 'a.txt', '-o', tmp_path / 'out', '-device', device],
              cwd=scratch)
    assert out.returncode == 0, out.stderr
    assert sorted(os.listdir(tmp_path / 'out')) == ['a.txt.aligned', 'a.txt.info', 'a.txt.json', 'a.txt.path']
    assert os.listdir(scratch) == []
