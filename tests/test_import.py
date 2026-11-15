"""`import sentalign` has no side effects: no files, no processes or threads, no pyximport, no output."""
import os
import subprocess
import sys

import sentalign

CODE = r'''
import importlib.util, multiprocessing, os, sys, threading
spec = importlib.util.find_spec("sentalign")
pkg = spec.submodule_search_locations[0]
before = sorted(os.listdir(pkg))
cwd_before = sorted(os.listdir("."))
import sentalign
import sentalign.config, sentalign.cli, sentalign.files          # the light modules
assert sorted(os.listdir(pkg)) == before, "import changed the package folder"
assert sorted(os.listdir(".")) == cwd_before, "import created files in the working directory"
assert "pyximport" not in sys.modules, "pyximport was imported"
assert "torch" not in sys.modules and "transformers" not in sys.modules, "heavy libraries imported at import time"
assert threading.active_count() == 1 and not multiprocessing.active_children(), "import started threads or processes"
assert not [f for f in os.listdir(pkg) if f.endswith(".reload1")], "reload files"
print("ok", sentalign.__version__)
'''


def test_import_has_no_side_effects(tmp_path, cli_env):
    out = subprocess.run([sys.executable, '-c', CODE], cwd=tmp_path, env=cli_env, capture_output=True, text=True, timeout=120)
    assert out.returncode == 0, out.stderr
    assert out.stdout.strip() == 'ok %s' % sentalign.__version__
    assert out.stderr == ''
    assert os.listdir(tmp_path) == []


def test_importing_the_aligner_module_compiles_nothing(tmp_path, cli_env):
    code = ('import importlib.util, os, sys\n'
            'pkg = importlib.util.find_spec("sentalign").submodule_search_locations[0]\n'
            'before = sorted(os.listdir(pkg))\n'
            'import sentalign.aligner\n'
            'assert sorted(os.listdir(pkg)) == before\n'
            'assert "pyximport" not in sys.modules\n'
            'import threading; assert threading.active_count() == 1\n')
    out = subprocess.run([sys.executable, '-c', code], cwd=tmp_path, env=cli_env, capture_output=True, text=True, timeout=300)
    assert out.returncode == 0, out.stderr
    assert os.listdir(tmp_path) == []
