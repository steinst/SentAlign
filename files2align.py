# -*- coding: UTF-8 -*-
"""Compatibility shim for the old ``files2align.py`` (now ``sentalign-files``; the main command no longer needs it)."""
import sys

try:
    from sentalign.files import main
except ImportError:
    sys.exit('The sentalign package is not installed. Run "pip install -e ." in the repository first.')

if __name__ == '__main__':
    sys.exit(main())
