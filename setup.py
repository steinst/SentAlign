from setuptools import setup, Extension
from Cython.Build import cythonize
import numpy as np
import sys

# Enable optimizations across all Cython modules
common_macros = [("NPY_NO_DEPRECATED_API", "NPY_1_7_API_VERSION")]

extra_compile_args = []
extra_link_args = []

if sys.platform == "win32":
    # MSVC flags: optimize for speed, fast FP
    extra_compile_args = ["/O2", "/fp:fast"]
else:
    # GCC/Clang flags
    extra_compile_args = ["-O3", "-ffast-math", "-march=native"]

extensions = [
    Extension(
        "*",
        ["*.pyx"],
        include_dirs=[np.get_include()],
        define_macros=common_macros,
        extra_compile_args=extra_compile_args,
        extra_link_args=extra_link_args,
    )
]

setup(
    ext_modules=cythonize(
        extensions,
        language_level=3,
        compiler_directives={
            "boundscheck": False,
            "wraparound": False,
            "cdivision": True,
            "initializedcheck": False,
        },
    ),
)