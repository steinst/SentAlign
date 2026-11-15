# -*- coding: UTF-8 -*-
"""Compatibility shim: ``python sentAlign.py -dir ... -sl ... -tl ...`` still works (needs ``pip install -e .``).

Version 2 is a package: use the ``sentalign`` command or ``python -m sentalign`` instead.
"""
import sys

try:
    from sentalign.cli import main
except ImportError:
    sys.exit('The sentalign package is not installed. Run "pip install -e ." in the repository first.')

if __name__ == '__main__':
    sys.exit(main())
