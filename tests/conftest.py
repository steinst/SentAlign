"""pytest configuration: import path, device choice, fixtures and skip messages for the SentAlign tests."""
import glob
import os
import shutil
import subprocess
import sys

import pytest

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SRC = os.path.join(ROOT, 'src')
if SRC not in sys.path:  # the tests also run from a checkout that has not been installed (compiled parts excepted)
    sys.path.insert(0, SRC)

MIN_FREE_GPU_MB = 6 * 1024


def pytest_configure(config):
    for line in ('slow: slow test (runs the aligner end to end)',
                 'gpu_or_cpu: runs the LaBSE model (cuda when enough memory is free, else cpu)',
                 'needs_fixtures: needs the large fixtures (tests/fetch_fixtures.py) or SENTALIGN_BOOK_FIXTURES'):
        config.addinivalue_line('markers', line)


def free_gpu_mb():
    """Free memory of the first GPU in MB, or 0 when there is no usable GPU."""
    if not shutil.which('nvidia-smi'):
        return 0
    try:
        out = subprocess.run(['nvidia-smi', '--query-gpu=memory.free', '--format=csv,noheader,nounits'],
                             capture_output=True, text=True, timeout=30)
        return int(out.stdout.strip().splitlines()[0])
    except (OSError, ValueError, IndexError, subprocess.SubprocessError):
        return 0


@pytest.fixture(scope='session')
def device():
    """cuda when at least 6 GB are free, else cpu; SENTALIGN_DEVICE overrides."""
    forced = os.environ.get('SENTALIGN_DEVICE')
    if forced:
        return forced
    return 'cuda' if free_gpu_mb() >= MIN_FREE_GPU_MB else 'cpu'


def _hf_hub_cache():
    env = os.environ
    if env.get('HF_HUB_CACHE'):
        return env['HF_HUB_CACHE']
    if env.get('HUGGINGFACE_HUB_CACHE'):
        return env['HUGGINGFACE_HUB_CACHE']
    return os.path.join(env.get('HF_HOME') or os.path.join(os.path.expanduser('~'), '.cache', 'huggingface'), 'hub')


def _model_cached():
    return bool(glob.glob(os.path.join(_hf_hub_cache(), 'models--setu4993--LaBSE', 'snapshots', '*')))


@pytest.fixture(scope='session')
def labse_available():
    if not _model_cached() and not os.environ.get('SENTALIGN_ALLOW_DOWNLOAD'):
        pytest.skip('the LaBSE model (setu4993/LaBSE) is not in the Hugging Face cache; run once with network access '
                    'or set SENTALIGN_ALLOW_DOWNLOAD=1 to let the test download it')


@pytest.fixture(scope='session')
def repo_root():
    return ROOT


@pytest.fixture(scope='session')
def cli_env():
    env = dict(os.environ)
    env['PYTHONPATH'] = SRC + os.pathsep + env.get('PYTHONPATH', '')
    return env


def run_cli(args, cwd=None, env=None, timeout=1800):
    """Run `python -m sentalign ...` in a subprocess."""
    return subprocess.run([sys.executable, '-m', 'sentalign'] + [str(a) for a in args], capture_output=True, text=True,
                          cwd=cwd, env=env, timeout=timeout)


@pytest.fixture(scope='session')
def cli():
    env = dict(os.environ)
    env['PYTHONPATH'] = SRC + os.pathsep + env.get('PYTHONPATH', '')
    return lambda args, **kw: run_cli(args, env=env, **kw)
