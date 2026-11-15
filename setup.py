"""Build the Cython extension modules at build/install time (never at import time)."""
import numpy as np
from Cython.Build import cythonize
from setuptools import Extension, setup

# the only Cython left from version 1: the greedy anchor selection, kept for --anchors legacy-greedy
# Both extensions are optional: without a C compiler SentAlign still installs; the path search then uses the Python
# reference kernel (much slower) and --anchors legacy-greedy is not available.
extensions = [Extension("sentalign.legacy.greedy", ["src/sentalign/legacy/greedy.pyx"], include_dirs=[np.get_include()],
                        optional=True)]
# The path-search kernel must give the same sums as the Python reference bit for bit, so the compiler may not
# fuse a multiplication and an addition into one instruction (FMA contraction; GCC and clang do that by default
# where the processor has it, e.g. on Apple Silicon).  MSVC does not contract by default.
import sys
NO_CONTRACTION = [] if sys.platform == "win32" else ["-ffp-contract=off"]
extensions.append(Extension("sentalign.pathfinding._dp_cython", ["src/sentalign/pathfinding/_dp_cython.pyx"],
                            include_dirs=[np.get_include()], extra_compile_args=NO_CONTRACTION,
                            define_macros=[("NPY_NO_DEPRECATED_API", "NPY_1_7_API_VERSION")], optional=True))
# Vecalign's search functions (taken from the Vecalign repository, Apache-2.0), for --pathfinder vecalign
extensions.append(Extension("sentalign.pathfinding._vecalign_core", ["src/sentalign/pathfinding/_vecalign_core.pyx"],
                            include_dirs=[np.get_include()],
                            define_macros=[("NPY_NO_DEPRECATED_API", "NPY_1_7_API_VERSION")], optional=True))

setup(ext_modules=cythonize(extensions, language_level=3, compiler_directives={"language_level": 3}))
