"""Compare SentAlign output (.path and .aligned files) with golden files.

The alignments must be identical; the scores may differ by TOL, since embeddings computed on another device or
system differ in the last digits.  Used by tests/test_cli_e2e.py and by the wheel check in .github/workflows/wheels.yml:

    python tests/check_golden.py OUT_DIR GOLDEN_DIR NAME [NAME ...]

compares OUT_DIR/NAME.path and OUT_DIR/NAME.aligned with the files of the same name in GOLDEN_DIR.  Exit code 0 when
all match, 1 otherwise.
"""
import os
import sys

TOL = 1e-5


def _split_path_line(line):
    head, score = line.rsplit(':', 1)
    return head, float(score)


def compare_path(got, gold):
    g, h = got.splitlines(), gold.splitlines()
    assert len(g) == len(h), 'different number of links: %d vs %d' % (len(g), len(h))
    for a, b in zip(g, h):
        (ha, sa), (hb, sb) = _split_path_line(a), _split_path_line(b)
        assert ha == hb, 'alignments differ: %s vs %s' % (a, b)
        assert abs(sa - sb) <= TOL, 'scores differ: %s vs %s' % (a, b)


def compare_aligned(got, gold):
    g, h = got.splitlines(), gold.splitlines()
    assert len(g) == len(h), 'different number of lines: %d vs %d' % (len(g), len(h))
    for a, b in zip(g, h):
        # the sentences may contain tabs themselves: the score is the text after the last tab
        (ta, sa), (tb, sb) = a.rsplit('\t', 1), b.rsplit('\t', 1)
        assert ta == tb, 'aligned sentences differ'
        assert abs(float(sa) - float(sb)) <= TOL, 'scores differ: %s vs %s' % (sa, sb)


def check(out_dir, golden_dir, names):
    """Return a list of problems (empty when every file matches)."""
    problems = []
    for name in names:
        for ext, compare in (('.path', compare_path), ('.aligned', compare_aligned)):
            got_file, gold_file = os.path.join(out_dir, name + ext), os.path.join(golden_dir, name + ext)
            if not os.path.exists(got_file):
                problems.append('%s: missing' % got_file)
                continue
            with open(got_file, encoding='utf-8') as f, open(gold_file, encoding='utf-8') as g:
                try:
                    compare(f.read(), g.read())
                except AssertionError as e:
                    problems.append('%s: %s' % (got_file, e))
    return problems


def main(argv=None):
    argv = sys.argv[1:] if argv is None else argv
    if len(argv) < 3:
        print(__doc__, file=sys.stderr)
        return 2
    problems = check(argv[0], argv[1], argv[2:])
    for p in problems:
        print(p, file=sys.stderr)
    if not problems:
        print('output matches the golden files: %s' % ', '.join(argv[2:]))
    return 1 if problems else 0


if __name__ == '__main__':
    sys.exit(main())
