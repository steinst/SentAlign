"""The scripts of the wheel check in CI: check_golden.py (equal alignments, scores within 1e-5), check_install.py."""
import os
import shutil
import subprocess
import sys

import pytest

import check_golden
import check_install

HERE = os.path.dirname(os.path.abspath(__file__))
GOLDEN = os.path.join(HERE, 'golden', 'v1_cpu', 'bleualign')
SCRIPT = os.path.join(HERE, 'check_golden.py')


def _copy(tmp_path, name='test2.txt'):
    for ext in ('.path', '.aligned'):
        shutil.copy(os.path.join(GOLDEN, name + ext), tmp_path / (name + ext))


def test_golden_matches_itself(tmp_path):
    _copy(tmp_path)
    assert check_golden.check(str(tmp_path), GOLDEN, ['test2.txt']) == []
    out = subprocess.run([sys.executable, SCRIPT, str(tmp_path), GOLDEN, 'test2.txt'], capture_output=True, text=True)
    assert out.returncode == 0, out.stderr


def test_small_score_difference_is_accepted(tmp_path):
    _copy(tmp_path)
    p = tmp_path / 'test2.txt.path'
    lines = p.read_text(encoding='utf-8').splitlines()
    head, score = lines[0].rsplit(':', 1)
    lines[0] = '%s:%.6f' % (head, float(score) + 5e-6)
    p.write_text('\n'.join(lines) + '\n', encoding='utf-8')
    assert check_golden.check(str(tmp_path), GOLDEN, ['test2.txt']) == []


def test_other_alignment_or_score_or_missing_file_fails(tmp_path):
    _copy(tmp_path)
    p = tmp_path / 'test2.txt.path'
    lines = p.read_text(encoding='utf-8').splitlines()
    head, score = lines[0].rsplit(':', 1)
    lines[0] = '%s:%.6f' % (head, float(score) + 1e-3)
    p.write_text('\n'.join(lines) + '\n', encoding='utf-8')
    assert any('scores differ' in x for x in check_golden.check(str(tmp_path), GOLDEN, ['test2.txt']))

    p.write_text('\n'.join(lines[1:]) + '\n', encoding='utf-8')     # a link fewer
    assert any('number of links' in x for x in check_golden.check(str(tmp_path), GOLDEN, ['test2.txt']))

    os.remove(tmp_path / 'test2.txt.aligned')
    problems = check_golden.check(str(tmp_path), GOLDEN, ['test2.txt'])
    assert any('missing' in x for x in problems)
    out = subprocess.run([sys.executable, SCRIPT, str(tmp_path), GOLDEN, 'test2.txt'], capture_output=True, text=True)
    assert out.returncode == 1


# tests/check_install.py, the check that an installed wheel or source package has its compiled parts

def test_check_install_finds_the_compiled_parts_and_reports_missing_ones(monkeypatch):
    assert check_install.missing_parts() == []       # the tests run with the extensions built in place
    monkeypatch.setattr(check_install, 'COMPILED', check_install.COMPILED + ['sentalign.no_such_module'])
    assert [m.split()[0] for m in check_install.missing_parts()] == ['sentalign.no_such_module']


def test_check_install_rejects_the_checkout():
    env = dict(os.environ, PYTHONPATH=os.path.join(os.path.dirname(HERE), 'src'))
    out = subprocess.run([sys.executable, os.path.join(HERE, 'check_install.py')], capture_output=True, text=True,
                         env=env)
    assert out.returncode == 1 and 'from the checkout' in out.stderr


def test_intel_macs_get_the_numpy_and_transformers_that_work_with_pytorch_2_2():
    # PyTorch 2.2 is the last for Intel Macs; transformers 5 refuses it and NumPy 2 breaks it
    tomllib = pytest.importorskip('tomllib')
    with open(os.path.join(os.path.dirname(HERE), 'pyproject.toml'), 'rb') as f:
        deps = tomllib.load(f)['project']['dependencies']
    intel_mac = "sys_platform == 'darwin' and platform_machine == 'x86_64'"
    assert 'numpy<2; ' + intel_mac in deps and 'transformers<5; ' + intel_mac in deps
    assert 'numpy' in deps and 'transformers' in deps          # no limit anywhere else
