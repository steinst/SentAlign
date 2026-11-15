"""The windowed scorer of the book test set (evaluation/evaluate_windows.py)."""
import glob
import os
import sys

import pytest

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, 'evaluation'))
import evaluate  # noqa: E402
import evaluate_windows as ew  # noqa: E402


def _write(path, text):
    path.write_text(text, encoding='utf-8')
    return str(path)


def test_whole_document_windows_give_the_numbers_of_evaluate_py():
    """A gold file without window lines is one window; on the German-French set the result must be exactly what
    evaluate.py gives (README: strict 0.935 / 0.929 / 0.932)."""
    gold = sorted(glob.glob(os.path.join(ROOT, 'eval_data', 'bleualign', 'gold', 'test*.txt')))
    test = [os.path.join(ROOT, 'tests', 'golden', 'v1_gpu', 'bleualign', os.path.basename(g) + '.path') for g in gold]
    ref = evaluate.score_multiple([evaluate.read_alignments(g) for g in gold],
                                  [evaluate.read_alignments(t) for t in test], True)
    got = ew.score_documents(gold, test, allow_gaps=True)     # these gold files leave out a few sentences
    with pytest.raises(ew.GoldError):
        ew.score_documents(gold, test)
    assert got == pytest.approx(ref)
    assert round(got['f1_strict'], 3) == 0.932 and round(got['f1_lax'], 3) == 0.989
    old = ew.score_documents(gold, test, allow_gaps=True, lax_recall='paper-2023')
    assert round(old['f1_strict'], 3) == 0.932 and round(old['f1_lax'], 3) == 0.964      # the 2023 paper


def test_only_links_touching_a_window_are_scored(tmp_path):
    gold = _write(tmp_path / 'g.txt', '# window a\n[10]:[11]\n[11, 12]:[12]\n[13]:[]\n# window b\n[50]:[52]\n[51]:[53]\n')
    # perfect inside the windows, nonsense everywhere else
    test = _write(tmp_path / 't.path', '[0]:[5]:0.5\n[1, 2]:[0]:0.4\n[10]:[11]:0.9\n[11,12]:[12]:0.8\n[13]:[]:0\n'
                                         '[30]:[30]:0.7\n[50]:[52]:0.9\n[51]:[53]:0.9\n[52]:[54]:0.9\n')
    res = ew.score_documents([gold], [test])
    assert res['precision_strict'] == 1.0 and res['recall_strict'] == 1.0


def test_a_link_across_the_window_edge_counts():
    links = [((10,), (11,)), ((11,), (12,))]
    src, trg = ew.check_window(links)
    sys_links = [((9, 10), (10, 11)), ((11,), (12,)), ((20,), (20,))]
    assert ew.links_in_window(sys_links, src, trg) == [((9, 10), (10, 11)), ((11,), (12,))]
    res = evaluate.score_multiple([links], [ew.links_in_window(sys_links, src, trg)], True)
    assert res['precision_strict'] == 0.5 and res['recall_strict'] == 0.5      # the crossing link is wrong (strict)
    assert res['precision_lax'] == 1.0                                         # but overlaps the gold link (lax)


def test_window_rules_are_checked(tmp_path):
    with pytest.raises(ew.GoldError):        # source sentence 11 twice
        ew.check_window([((10, 11), (1,)), ((11,), (2,))])
    with pytest.raises(ew.GoldError):        # target sentence 2 missing inside the window
        ew.check_window([((10,), (1,)), ((11,), (3,))])
    bad = _write(tmp_path / 'bad.txt', '# window 1\n[1]:[1]\n[3]:[2]\n')
    test = _write(tmp_path / 't.path', '[1]:[1]:1\n[2]:[]:0\n[3]:[2]:1\n')
    assert ew.main(['-t', test, '-g', bad]) == 2
    assert ew.main(['-t', test, '-g', bad, '--allow-gaps']) == 0


def test_reading_windows(tmp_path):
    g = _write(tmp_path / 'g.txt', '# a comment\n[0]:[0]\n# window two\n\n[5]:[6]\n[]:[7]\n')
    w = ew.read_windows(g)
    assert [n for n, _ in w] == ['all', 'two'] and w[1][1] == [((5,), (6,)), ((), (7,))]


def test_per_window_table(tmp_path, capsys):
    gold = _write(tmp_path / 'g.txt', '# window 1\n[0]:[0]\n[1]:[1]\n# window 2\n[9]:[9]\n')
    test = _write(tmp_path / 't.path', '[0]:[0]:1\n[1]:[2]:1\n[9]:[9]:1\n')
    ew.main(['-t', test, '-g', gold, '--per-window'])
    lines = [l for l in capsys.readouterr().out.strip().splitlines() if '\t' in l]
    assert lines[0].startswith('file\twindow') and len(lines) == 3
    assert lines[1].split('\t')[5] == '0.500' and lines[2].split('\t')[5] == '1.000'
