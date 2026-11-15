"""End to end: the new command line on both evaluation sets against the golden files of the 2023 code.

On cuda the .path and .aligned files must be byte-identical.  On cpu the alignments must be identical and the scores
equal within 1e-5 (embeddings differ in the last digits between devices).  The numbers of the README follow from that.
Needs the LaBSE model in the Hugging Face cache (or SENTALIGN_ALLOW_DOWNLOAD=1).
"""
import glob
import os
import re
import shutil
import subprocess
import sys

import pytest

from check_golden import compare_aligned as _compare_aligned, compare_path as _compare_path

SETS = {'bleualign': ('deu', 'fra'), 'parice': ('eng', 'isl')}
# (strict P, R, F1, lax P, R, F1) as printed by evaluation/evaluate.py --lax-recall paper-2023 (the numbers of the
# 2023 paper; ParIce measured on the unchanged 2023 code)
EXPECTED = {'bleualign': (0.935, 0.929, 0.932, 0.998, 0.933, 0.964),
            'parice': (0.948, 0.953, 0.951, 0.985, 0.955, 0.970)}

pytestmark = [pytest.mark.slow, pytest.mark.gpu_or_cpu]


@pytest.mark.parametrize('name', sorted(SETS))
def test_eval_set_matches_golden(name, tmp_path, device, labse_available, cli, repo_root):
    sl, tl = SETS[name]
    corpus = tmp_path / name
    shutil.copytree(os.path.join(repo_root, 'eval_data', name), corpus)   # the repository stays clean
    out = cli(['-dir', corpus, '-sl', sl, '-tl', tl, '-device', device, '--compat', 'paper-2023'], timeout=3600)
    assert out.returncode == 0, out.stderr[-3000:]
    golden = os.path.join(repo_root, 'tests', 'golden', 'v1_gpu', name)
    gold_files = sorted(glob.glob(os.path.join(golden, '*.path')))
    assert gold_files
    for gold_path in gold_files:
        base = os.path.basename(gold_path)[:-len('.path')]
        for ext, compare in (('.path', _compare_path), ('.aligned', _compare_aligned)):
            got_file = corpus / 'output' / (base + ext)
            gold_file = os.path.join(golden, base + ext)
            if device.startswith('cuda'):
                assert got_file.read_bytes() == open(gold_file, 'rb').read(), base + ext + ' is not byte-identical'
            else:
                compare(got_file.read_text(encoding='utf-8'), open(gold_file, encoding='utf-8').read())
    # no leftovers
    assert os.listdir(corpus / 'tmp') == []
    assert not (corpus / 'files2align.txt').exists() and not (corpus / 'filesdone.txt').exists()

    # evaluation numbers (README)
    gold_dir = os.path.join(repo_root, 'eval_data', name, 'gold')
    tests = sorted(glob.glob(str(corpus / 'output' / '*.path')))
    tests = [t for t in tests if os.path.basename(t).startswith(('test', 'es_', 'n_', 's_', 't_', 'u_'))]
    golds = [os.path.join(gold_dir, os.path.basename(t)[:-len('.path')]) for t in tests]
    ev = subprocess.run([sys.executable, os.path.join(repo_root, 'evaluation', 'evaluate.py'), '--lax-recall', 'paper-2023', '-t'] + tests + ['-g'] + golds,
                        capture_output=True, text=True, timeout=600)
    assert ev.returncode == 0, ev.stderr  # the table is printed to stderr
    numbers = [float(x) for line in ev.stderr.splitlines() if line.strip().startswith(('| Precision', '| Recall', '| F1', '|  Precision'))
               for x in re.findall(r'\d\.\d+', line)]
    expected = EXPECTED[name]
    # printed as Precision(strict, lax), Recall(strict, lax), F1(strict, lax)
    assert len(numbers) == 6, ev.stderr
    got = (numbers[0], numbers[2], numbers[4], numbers[1], numbers[3], numbers[5])
    assert got == pytest.approx(expected, abs=5e-4), ev.stderr
