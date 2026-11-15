"""Annotation tool: autosave must not undo Finish; sentence numbers are checked."""
import os
import re
import shutil
import subprocess
import sys

import pytest

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, 'tools', 'annotation'))
import store  # noqa: E402

HTML = os.path.join(ROOT, 'tools', 'annotation', 'templates', 'annotate.html')


# ---------------------------------------------------------------------------- sentence numbers
WIN = {'start': 2, 'end': 4}


def _state(links, src_null=(), trg_null=()):
    return {'links': links, 'src_null': list(src_null), 'trg_null': list(trg_null)}


def _ok():
    return _state([{'src': [2], 'trg': [5]}, {'src': [3], 'trg': [6]}])


def test_valid_annotation_passes():
    assert store.check(_ok(), WIN, 'src', 10, 8) == []


@pytest.mark.parametrize('bad', [
    _state([{'src': [2], 'trg': [10]}, {'src': [3], 'trg': [6]}]),                 # past the other document
    _state([{'src': [2], 'trg': [-1]}, {'src': [3], 'trg': [6]}]),                 # negative
    _state([{'src': [2], 'trg': ['5']}, {'src': [3], 'trg': [6]}]),                # a string
    _state([{'src': [2], 'trg': [5.5]}, {'src': [3], 'trg': [6]}]),                # a float
    _state([{'src': [2], 'trg': [True]}, {'src': [3], 'trg': [6]}]),               # a bool
    _state([{'src': [2], 'trg': [5]}, {'src': [3], 'trg': [6]}], trg_null=[99]),    # null list, other side
    _state([{'src': [2], 'trg': [5]}, {'src': [3], 'trg': [6]}], src_null=[1000]),  # null list, window side
    _state([{'src': [2], 'trg': [5]}, {'src': [8], 'trg': [6]}]),                  # past the window-side document
])
def test_invalid_sentence_numbers_are_refused(bad):
    assert store.check(bad, WIN, 'src', 10, 8)


def test_window_side_bound_is_the_document_of_that_side():
    # side 'trg': the window numbers are target numbers, the other side is the source
    assert any('does not exist' in p for p in store.check(_state([{'src': [10], 'trg': [2]}, {'src': [1], 'trg': [3]}]),
                                                          WIN, 'trg', 10, 8))


# ---------------------------------------------------------------------------- autosave and Finish
NODE = r'''
const fs = require('fs');
const html = fs.readFileSync(process.argv[2], 'utf8');
const grab = (a, b) => { const i = html.indexOf(a); const j = html.indexOf(b, i); return html.slice(i, j); };
const code = grab('function changed()', 'function link()') + grab('// saves run one after another', 'async function finish()')
           + html.slice(html.indexOf('async function finish()'), html.indexOf('\n', html.indexOf('async function finish()')));
const log = [];
const timers = [];
const fake = `
  let saveTimer = null, dirty = false, lastAction = Date.now(), activeSince = Date.now(), readonly = false;
  let state = {}, PID = 'p', N = 1;
  const document = {getElementById: () => ({textContent: ''})};
  const say = () => {};
  const setTimeout = (f, ms) => { timers.push(f); return timers.length; };
  const clearTimeout = (id) => { if (id) timers[id - 1] = null; };
  const fetch = async (url, o) => { const st = JSON.parse(o.body).status; await new Promise(r => global.setImmediate(r)); log.push(st);
                                    return {json: async () => ({ok: true})}; };
` + code + `
  changed();                 // an edit: autosave is scheduled
  finish();                  // Finish before the timer fires
  for (const f of timers) if (f) f();   // the timer would fire now, if it were not cancelled
  await saveChain;
`;
(async () => { await (new (Object.getPrototypeOf(async function(){}).constructor)('timers', 'log', fake))(timers, log);
  console.log(JSON.stringify(log)); })();
'''


@pytest.mark.skipif(shutil.which('node') is None, reason='node is not installed')
def test_pending_autosave_does_not_arrive_after_finish(tmp_path):
    js = tmp_path / 'check.cjs'
    js.write_text(NODE, encoding='utf-8')
    out = subprocess.run(['node', str(js), HTML], capture_output=True, text=True, timeout=60)
    assert out.returncode == 0, out.stderr
    assert out.stdout.strip() == '["done"]'
