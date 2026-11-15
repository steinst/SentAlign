"""Word limits only where the search applies them, cleanup and commit of the output files, a failed group, Finish and
the page unload, the languages of the pair form, option names in the texts."""
import logging
import os
import re
import shutil
import subprocess
import types

import pytest

from sentalign import cli, files
from sentalign.aligner import Aligner
from sentalign.api import SentAlign
from sentalign.config import AlignConfig
from sentalign.scoring import register_encoder
from sentalign.scoring.embedding import EmbeddingScorer
from sentalign.scoring.encoders import ModelLoadError

from test_api import _fake_aligner
from test_group_encoding import _corpus, _docs, _output
from test_scoring import FakeEncoder

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
HTML = os.path.join(ROOT, 'tools', 'annotation', 'templates', 'annotate.html')


# ---------------------------------------------------------------------------- word limits
@pytest.mark.parametrize('pathfinder', ['vecalign', 'chain', 'galechurch'])
def test_word_limits_do_not_touch_path_finders_that_do_not_apply_them(pathfinder):
    """These path finders do not use -minwords / -maxwords in the search, so the limits must not change the score
    of a link they made (a pair above -maxwords came out with score 0)."""
    long = ' '.join('word%d' % i for i in range(131))
    src = ['The cat sat on mat number %d.' % i for i in range(3)] + [long] + ['The last one.']
    trg = ['Kottur sat a mottu nr %d.' % i for i in range(3)] + [long] + ['Sidasta.']
    limited = _fake_aligner(pathfinder=pathfinder).align(src, trg)                      # default -maxwords 110
    free = _fake_aligner(pathfinder=pathfinder, maximum_length_words=100000).align(src, trg)
    assert limited.to_path_lines() == free.to_path_lines()
    assert [ln.score for ln in limited.links if 3 in ln.src and 3 in ln.trg] == [pytest.approx(1.0, abs=1e-3)]


# ---------------------------------------------------------------------------- cleanup of a failed file
def test_cleanup_of_a_failed_file_leaves_the_output_of_other_files(tmp_path):
    """`news` fails; the finished output of `news.2020.newswire.txt` begins with the same name and must stay."""
    s, t = _docs(1)[0]
    names = ['news.2020.newswire.txt', 'news']
    for lang, text in (('eng', s), ('isl', t)):
        (tmp_path / lang).mkdir()
        (tmp_path / lang / names[0]).write_text('\n'.join(text) + '\n', encoding='utf-8')
        (tmp_path / lang / names[1]).write_text('', encoding='utf-8')         # empty: fails
    cfg = AlignConfig(corpus_folder=str(tmp_path), proc_device='cpu', num_proc=1)
    al = Aligner(cfg)
    al.scorer = EmbeddingScorer(FakeEncoder(order='length', dedupe=True))
    out = tmp_path / 'output'
    out.mkdir()
    (out / 'news.path.new').write_text('x', encoding='utf-8')       # left by a run that was killed while writing
    assert al.run(names) == ([names[0]], [names[1]])
    left = sorted(os.listdir(out))
    assert left == [names[0] + e for e in ('.aligned', '.json', '.path')], left


# ---------------------------------------------------------------------------- a group that cannot be encoded
class _GroupFails(EmbeddingScorer):
    error = RuntimeError('CUDA error (fake)')

    def encode_sides(self, sides):
        raise self.error


def test_a_group_that_cannot_be_encoded_is_aligned_file_by_file(tmp_path, caplog):
    docs = _docs(3)
    al, names = _corpus(tmp_path / 'group', docs)
    al.scorer = _GroupFails(FakeEncoder(order='length', dedupe=True))
    with caplog.at_level(logging.WARNING, logger='sentalign'):
        assert al.run(names) == (names, [])            # the first file was charged with the failure of the group
    assert 'one by one' in caplog.text
    one, _ = _corpus(tmp_path / 'one', docs, batch_texts=0)
    one.scorer = EmbeddingScorer(FakeEncoder(order='length', dedupe=True))
    one.run(names)
    assert _output(tmp_path / 'group', names) == _output(tmp_path / 'one', names)


def test_align_many_survives_a_group_that_cannot_be_encoded():
    docs = _docs(3)
    sa = SentAlign(device='cpu')
    sa._aligner.scorer = _GroupFails(FakeEncoder(order='length', dedupe=True))
    got = [r.to_path_lines() for r in sa.align_many(docs)]
    one = _fake_aligner(batch_texts=0)
    assert got == [one.align(s, t).to_path_lines() for s, t in docs]


def test_a_model_that_cannot_be_loaded_still_ends_the_run(tmp_path):
    al, names = _corpus(tmp_path, _docs(3))
    al.scorer = _GroupFails(FakeEncoder(order='length', dedupe=True))
    al.scorer.error = ModelLoadError('no model')
    assert al.run(names) == ([], names)


# ---------------------------------------------------------------------------- two instances on one file
def test_temporary_output_names_belong_to_the_process(tmp_path, monkeypatch):
    al, names = _corpus(tmp_path, _docs(1))
    al.scorer = EmbeddingScorer(FakeEncoder(order='length', dedupe=True))
    seen = []
    real = Aligner._commit_outputs

    def commit(self, file_name):
        seen.extend(os.listdir(self.output_folder))
        return real(self, file_name)

    monkeypatch.setattr(Aligner, '_commit_outputs', commit)
    assert al.run(names) == (names, [])
    assert len(seen) == 2 and all(('.%d' % os.getpid()) in f for f in seen), seen
    assert sorted(os.listdir(tmp_path / 'output')) == [names[0] + e for e in ('.aligned', '.json', '.path')]


def test_a_failed_commit_does_not_remove_the_output_another_instance_finished(tmp_path, monkeypatch):
    al, names = _corpus(tmp_path, _docs(1))
    al.scorer = EmbeddingScorer(FakeEncoder(order='length', dedupe=True))
    out = tmp_path / 'output'
    real_write = Aligner.write_pairs_to_file

    def write(self, *a, **k):
        n = real_write(self, *a, **k)
        # another instance finishes the file while this one is writing
        for e in ('.path', '.aligned'):
            (out / (names[0] + e)).write_text('OTHER', encoding='utf-8')
        return n

    real_replace = os.replace

    def replace(a, b):
        if str(b).endswith('.path'):
            raise OSError('gone')
        return real_replace(a, b)

    monkeypatch.setattr(Aligner, 'write_pairs_to_file', write)
    monkeypatch.setattr(os, 'replace', replace)
    aligned, failed = al.run(names)
    monkeypatch.undo()
    assert failed == names
    assert files.is_done(str(out), names[0])
    assert (out / (names[0] + '.aligned')).read_text(encoding='utf-8') == 'OTHER'
    assert sorted(os.listdir(out)) == [names[0] + e for e in ('.aligned', '.path')]


def test_a_stale_aligned_file_without_a_path_is_still_replaced_safely(tmp_path, monkeypatch):
    """Not done (no .path), no --overwrite: a failure between the renames must not leave new .path + old .aligned."""
    al, names = _corpus(tmp_path, _docs(1))
    al.scorer = EmbeddingScorer(FakeEncoder(order='length', dedupe=True))
    out = tmp_path / 'output'
    out.mkdir()
    (out / (names[0] + '.aligned')).write_text('STALE', encoding='utf-8')
    real_replace = os.replace

    def replace(a, b):
        if str(b).endswith('.aligned'):
            raise OSError('gone')
        return real_replace(a, b)

    monkeypatch.setattr(os, 'replace', replace)
    aligned, failed = al.run(names)
    monkeypatch.undo()
    assert failed == names and not files.is_done(str(out), names[0])


# ---------------------------------------------------------------------------- Finish and leaving the page
NODE = r'''
const fs = require('fs');
const html = fs.readFileSync(process.argv[2], 'utf8');
const grab = (a, b) => { const i = html.indexOf(a); const j = html.indexOf(b, i); return html.slice(i, j); };
const line = a => grab(a, '\n');
const code = grab('function changed()', 'function link()') + grab('// saves run one after another', 'async function finish()')
           + line('async function finish()') + line("window.addEventListener('beforeunload'");
const fake = `
  let saveTimer = null, dirty = false, lastAction = Date.now(), activeSince = Date.now(), readonly = false;
  let state = {}, PID = 'p', N = 1, unload = null;
  const document = {getElementById: () => ({textContent: ''})};
  const window = {addEventListener: (name, f) => { unload = f; }};
  const leave = () => unload({preventDefault() {}});
  const say = () => {};
  const setTimeout = (f, ms) => { timers.push(f); return timers.length; };
  const clearTimeout = (id) => { if (id) timers[id - 1] = null; };
  const fire = () => { for (let k = 0; k < timers.length; k++) { const f = timers[k]; timers[k] = null; if (f) f(); } };
  const fetch = async (url, o) => { const st = JSON.parse(o.body).status; await new Promise(r => global.setImmediate(r)); log.push(st);
                                    return {json: async () => ({ok: true})}; };
` + code + `
  // 1. the page is left while Finish is on its way, and the annotator stays on it
  changed(); finish(); leave(); await saveChain;
  log.push('|');
  // 2. left again later, nothing changed since Finish
  leave(); await saveChain;
  log.push('|');
  // 3. a change after Finish is a new draft
  changed(); fire(); await saveChain;
  log.push('|');
  // 4. a change made while Finish is on its way (the request is out) is not lost either
  finish(); await null; await null; changed(); leave(); await saveChain;
`;
(async () => { const timers = [], log = [];
  await (new (Object.getPrototypeOf(async function(){}).constructor)('timers', 'log', fake))(timers, log);
  console.log(log.join(' ')); })();
'''


@pytest.mark.skipif(shutil.which('node') is None, reason='node is not installed')
def test_leaving_the_page_does_not_undo_finish(tmp_path):
    js = tmp_path / 'check.cjs'
    js.write_text(NODE, encoding='utf-8')
    out = subprocess.run(['node', str(js), HTML], capture_output=True, text=True, timeout=60)
    assert out.returncode == 0, out.stderr
    assert out.stdout.strip() == 'done | | draft | done draft'


# ---------------------------------------------------------------------------- languages of the pair form
def _pair(tmp_path, *extra):
    src, trg = tmp_path / 's.txt', tmp_path / 't.txt'
    src.write_text('One.\nTwo.\n', encoding='utf-8')
    trg.write_text('Eitt.\nTvö.\n', encoding='utf-8')
    return list(extra) + [str(src), str(trg), '-o', str(tmp_path / 'out'), '-device', 'cpu']


class _NeedsLang(FakeEncoder):
    needs_language = True


register_encoder('needs-lang', _NeedsLang)


def _capture_aligner(monkeypatch):
    seen = {}

    def fake(config, **kw):
        seen['cfg'] = config
        return types.SimpleNamespace(run=lambda names: (names, []))

    import sentalign.aligner
    monkeypatch.setattr(sentalign.aligner, 'Aligner', fake)
    return seen


def test_pair_form_without_languages_records_none(tmp_path, monkeypatch):
    seen = _capture_aligner(monkeypatch)
    assert cli.main(_pair(tmp_path)) == 0
    assert (seen['cfg'].source_language, seen['cfg'].target_language) == (None, None)
    assert cli.main(_pair(tmp_path, '-sl', 'deu', '-tl', 'fra')) == 0
    assert (seen['cfg'].source_language, seen['cfg'].target_language) == ('deu', 'fra')


def test_pair_form_with_an_encoder_that_needs_the_language(tmp_path, monkeypatch, capsys, caplog):
    seen = _capture_aligner(monkeypatch)
    # not given (or only one of them): refused before any model is loaded
    for extra in ([], ['-sl', 'eng']):
        assert cli.main(_pair(tmp_path, '--encoder-kind', 'needs-lang', *extra)) == 2
        assert '-sl' in capsys.readouterr().err and not seen
    # given, also when they are eng / isl: no complaint
    with caplog.at_level(logging.WARNING, logger='sentalign'):
        assert cli.main(_pair(tmp_path, '--encoder-kind', 'needs-lang', '-sl', 'eng', '-tl', 'isl')) == 0
    assert 'language' not in caplog.text
    assert (seen['cfg'].source_language, seen['cfg'].target_language) == ('eng', 'isl')


def test_folder_form_keeps_the_default_languages(tmp_path):
    for lang in ('eng', 'isl'):
        (tmp_path / lang).mkdir()
        (tmp_path / lang / 'a.txt').write_text('One.\n', encoding='utf-8')
    args = cli.build_parser().parse_args(['-dir', str(tmp_path)])
    from sentalign.config import config_from_args
    cfg = config_from_args(args)
    assert (cfg.source_language, cfg.target_language) == ('eng', 'isl')
    assert cli.main(['-dir', str(tmp_path), '--list-files']) == 0


# ---------------------------------------------------------------------------- -concats 16 under paper-2023
def test_concats_above_15_under_paper_2023_names_the_legacy_backend():
    with pytest.raises(ValueError, match='--path-backend legacy'):
        AlignConfig(compat='paper-2023', max_concatenations=16)
    with pytest.raises(ValueError) as e:
        AlignConfig(max_concatenations=16)
    assert 'legacy' not in str(e.value)                 # version 2 has no backend that can
    assert AlignConfig(compat='paper-2023', max_concatenations=16, path_backend='legacy').max_concatenations == 16


# ---------------------------------------------------------------------------- option names in the texts
def _known_options():
    return set(cli.build_parser()._option_string_actions) | {'--no-half'}


def test_help_names_only_options_that_exist():
    text = re.sub(r'-\n\s+', '-', cli.build_parser().format_help())      # argparse breaks lines at hyphens
    unknown = {o for o in re.findall(r'(?<![\w-])--[a-z][a-z0-9-]+', text)} - _known_options()
    assert not unknown, unknown
    assert 'whole document that is not cut' in ' '.join(text.split())       # --max-segment-nodes


@pytest.mark.parametrize('name', ['README.md', 'CHANGELOG.md'])
def test_documents_name_only_options_that_exist(name):
    with open(os.path.join(ROOT, name), encoding='utf-8') as f:
        text = f.read()
    other_tools = {'--lax-recall', '--index-url'}       # evaluation/evaluate.py; pip (installing PyTorch)
    long = set(re.findall(r'(?<![\w-])--[a-z][a-z0-9-]+', text)) - _known_options() - other_tools
    assert not long, long
    for wrong in ('-minl', '-maxl'):                    # the word limits are -minwords / -maxwords
        assert not re.search(r'%s\b' % wrong, text), wrong


def test_readme_gives_the_limit_of_max_concatenations():
    with open(os.path.join(ROOT, 'README.md'), encoding='utf-8') as f:
        lines = [ln for ln in f if '--max-concatenations' in ln]
    assert any('15' in ln for ln in lines), lines
