"""Check an installed sentalign (a wheel or the source package): the compiled parts must be there.

The C extensions are optional at build time (setup.py), so a failed compilation still gives a working but slow
installation; this script makes that an error.  Used by .github/workflows/wheels.yml:

    python tests/check_install.py [--rust]

--rust also checks the kernel of the package sentalign-rust.  Exit code 0 when everything is there.
"""
import importlib
import os
import sys

COMPILED = ['sentalign.pathfinding._dp_cython',     # the path-search kernel
            'sentalign.pathfinding._vecalign_core',  # Vecalign's search, --pathfinder vecalign
            'sentalign.legacy.greedy']               # --anchors legacy-greedy


def missing_parts(rust=False):
    """The compiled modules that cannot be imported, with the reason."""
    missing = []
    for name in COMPILED + (['sentalign_rust._core'] if rust else []):
        try:
            importlib.import_module(name)
        except ImportError as e:
            missing.append('%s (%s)' % (name, e))
    return missing


def main(argv=None):
    argv = sys.argv[1:] if argv is None else argv
    import sentalign
    here = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    if os.path.abspath(sentalign.__file__).startswith(here + os.sep):
        print('sentalign is imported from the checkout (%s), not from the installation' % sentalign.__file__,
              file=sys.stderr)
        return 1
    missing = missing_parts('--rust' in argv)
    if missing:
        print('not compiled or not installed: ' + ', '.join(missing), file=sys.stderr)
        return 1
    print('installation ok: sentalign %s from %s' % (sentalign.__version__, os.path.dirname(sentalign.__file__)))
    return 0


if __name__ == '__main__':
    sys.exit(main())
