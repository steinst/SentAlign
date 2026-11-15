"""The device of a run: cuda when there is one, else the CPU; ``--half`` follows the device that is used."""
import json
import logging

import pytest

import sentalign.aligner
import sentalign.config
from sentalign import cli
from sentalign.aligner import Aligner
from sentalign.config import AlignConfig

from test_scoring import FakeEncoder


def _cuda(monkeypatch, there):
    """A machine with or without CUDA, and the recording encoder in place of the Hugging Face one."""
    monkeypatch.setattr(sentalign.config, '_cuda_available', lambda: there)
    monkeypatch.setattr(sentalign.aligner, 'get_encoder', lambda kind: _Recording)
    _Recording.asked.clear()


class _Recording(FakeEncoder):
    """Keeps the device and precision the aligner asks for."""
    asked = []

    def __init__(self, model_name=None, device=None, half=False, **kw):
        super().__init__(**kw)
        _Recording.asked.append((device, half))


@pytest.fixture
def logged():
    """The messages of the sentalign logger (it does not propagate once the command line has set it up)."""
    lines = []
    handler = logging.Handler()
    handler.emit = lambda record: lines.append(record.getMessage())
    logger = logging.getLogger('sentalign')
    logger.addHandler(handler)
    yield lines
    logger.removeHandler(handler)


def _pair(tmp_path, *extra):
    src, trg = tmp_path / 's.txt', tmp_path / 't.txt'
    src.write_text('One.\nTwo.\n', encoding='utf-8')
    trg.write_text('Eitt.\nTvö.\n', encoding='utf-8')
    return [str(src), str(trg), '-o', str(tmp_path / 'out')] + list(extra)


# ---------------------------------------------------------------------------- the configuration
def test_the_default_device_is_decided_when_the_aligner_is_built(monkeypatch):
    cfg = AlignConfig()
    assert (cfg.proc_device, cfg.half) == ('auto', None)       # no torch import for --help or --list-files
    _cuda(monkeypatch, False)
    cfg.resolve_device()
    assert (cfg.proc_device, cfg.half) == ('cpu', False)
    _cuda(monkeypatch, True)                                   # settled: a second call changes nothing
    assert cfg.resolve_device().proc_device == 'cpu'
    cfg = AlignConfig().resolve_device()
    assert (cfg.proc_device, cfg.half) == ('cuda', True)


def test_half_follows_the_device_that_is_used(monkeypatch):
    _cuda(monkeypatch, False)
    assert AlignConfig(half=False).resolve_device().half is False
    assert AlignConfig(half=True).resolve_device().half is True         # asked for: the encoder refuses it on the CPU
    _cuda(monkeypatch, True)
    assert AlignConfig(half=False).resolve_device().half is False
    assert AlignConfig(proc_device='cpu').resolve_device().proc_device == 'cpu'
    assert AlignConfig(model_name='laser2').resolve_device().half is False      # not a Hugging Face encoder


@pytest.mark.parametrize('there', [False, True])
def test_paper_2023_is_float32_on_every_device(monkeypatch, there):
    _cuda(monkeypatch, there)
    cfg = AlignConfig(compat='paper-2023')
    assert cfg.half is False and cfg.deviations_from_paper() == {}
    cfg.resolve_device()
    assert cfg.proc_device == ('cuda' if there else 'cpu')
    assert cfg.half is False and cfg.deviations_from_paper() == {}


def test_a_cuda_device_asked_for_by_name_must_be_there(monkeypatch):
    _cuda(monkeypatch, False)
    for device in ('cuda', 'cuda:1'):
        with pytest.raises(ValueError, match='no CUDA device'):
            AlignConfig(proc_device=device).resolve_device()
        with pytest.raises(ValueError, match='no CUDA device'):
            Aligner(AlignConfig(corpus_folder='x', proc_device=device))
    _cuda(monkeypatch, True)
    assert AlignConfig(proc_device='cuda:1').resolve_device().proc_device == 'cuda:1'


# ---------------------------------------------------------------------------- the command line and the API
def test_without_cuda_the_command_line_runs_on_the_cpu(tmp_path, monkeypatch, logged):
    """Before: the default was cuda, the model could not be loaded and every file failed (exit code 1)."""
    _cuda(monkeypatch, False)
    assert cli.main(_pair(tmp_path)) == 0
    assert _Recording.asked == [('cpu', False)]
    assert any('running on the CPU' in line for line in logged)
    record = json.loads((tmp_path / 'out' / 's.txt.json').read_text(encoding='utf-8'))
    assert record['settings']['proc_device'] == 'cpu' and record['settings']['half'] is False
    assert (tmp_path / 'out' / 's.txt.path').read_text(encoding='utf-8').count('\n') == 2


def test_with_cuda_the_default_is_cuda_in_half_precision(tmp_path, monkeypatch, logged):
    _cuda(monkeypatch, True)
    assert cli.main(_pair(tmp_path)) == 0
    assert cli.main(_pair(tmp_path, '--compat', 'paper-2023', '--overwrite')) == 0
    assert cli.main(_pair(tmp_path, '-device', 'cpu', '--overwrite')) == 0
    assert cli.main(_pair(tmp_path, '--no-half', '--overwrite')) == 0
    assert _Recording.asked == [('cuda', True), ('cuda', False), ('cpu', False), ('cuda', False)]
    assert 'device: cuda' in logged and not any('running on the CPU' in line for line in logged)


def test_cuda_by_name_without_cuda_is_refused_before_any_work(tmp_path, monkeypatch, capsys):
    _cuda(monkeypatch, False)
    assert cli.main(_pair(tmp_path, '-device', 'cuda')) == 2
    assert 'no CUDA device' in capsys.readouterr().err
    assert _Recording.asked == [] and not (tmp_path / 'out').exists()
    # nothing to do: no complaint (the device is not needed)
    assert cli.main(_pair(tmp_path, '-device', 'cuda', '--list-files')) == 0


def test_the_api_picks_the_device_too(monkeypatch):
    from sentalign.api import SentAlign
    for there, device in ((False, 'cpu'), (True, 'cuda')):
        _cuda(monkeypatch, there)
        config = SentAlign().config
        assert (config.proc_device, config.half) == (device, there)
    _cuda(monkeypatch, False)
    with pytest.raises(ValueError, match='no CUDA device'):
        SentAlign(device='cuda')
