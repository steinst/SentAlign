"""The annotation tool of the book test set (tools/annotation): windows, checks, gold export, web app."""
import json
import os
import sys

import pytest

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, 'tools', 'annotation'))
sys.path.insert(0, os.path.join(ROOT, 'evaluation'))
flask = pytest.importorskip('flask')

import prepare  # noqa: E402
import store  # noqa: E402
import evaluate_windows as ew  # noqa: E402


def _xml(path, paragraphs):
    """A document in the format of the book pipeline: paragraphs of sentences."""
    body = []
    k = 0
    for p, sents in enumerate(paragraphs):
        ss = []
        for s in sents:
            k += 1
            ss.append('<s id="s%d">%s</s>' % (k, s))
        body.append('<p id="p%d">%s</p>' % (p + 1, ''.join(ss)))
    path.write_text('<book><header/><text><chapter n="1"><head>I</head>%s</chapter></text></book>' % ''.join(body),
                    encoding='utf-8')
    return str(path)


def _doc(n_par, per_par, tag):
    return [['%s %d.%d' % (tag, p, s) for s in range(per_par[p % len(per_par)])] for p in range(n_par)]


@pytest.fixture
def data(tmp_path):
    src = _xml(tmp_path / 'src.xml', _doc(60, [3, 5, 8, 2], 'Source'))
    trg1 = _xml(tmp_path / 'trg1.xml', _doc(30, [4, 6, 3], 'Target'))
    trg2 = _xml(tmp_path / 'trg2.xml', _doc(30, [4, 6, 3], 'More'))
    d = str(tmp_path / 'data')
    prepare.main(['add-pair', d, 'demo', '--src', src, '--trg', trg1, trg2, '--src-lang', 'en', '--trg-lang', 'is',
                  '--windows', '6', '--min', '20', '--max', '40', '--meta', 'author=Nobody'])
    return d


def test_read_xml_numbers_sentences_like_the_aligner(tmp_path):
    p = _xml(tmp_path / 'a.xml', [['One  two.', 'Three.'], ['Four.']])
    q = _xml(tmp_path / 'b.xml', [['Five.']])
    sents, par = prepare.read_xml([p, q])
    assert sents == ['One two.', 'Three.', 'Four.', 'Five.'] and par == [0, 0, 1, 2]


def test_windows_vary_follow_paragraphs_and_spread(data):
    st = store.Store(data)
    w = st.windows('demo')
    src = st.side('demo', 'source')
    starts = set(prepare.paragraph_starts(src['paragraph']))
    assert 4 <= len(w) <= 6
    for x in w:
        assert x['start'] in starts and 20 <= x['end'] - x['start'] <= 40 or x['end'] == len(src['sentences'])
    assert all(a['end'] <= b['start'] for a, b in zip(w, w[1:]))           # no overlap, in order
    assert w[0]['start'] == 0 and w[-1]['start'] > 0.7 * len(src['sentences'])
    assert len({x['end'] - x['start'] for x in w}) > 1                     # lengths vary with the paragraphs
    meta = st.meta('demo')
    assert meta['n_trg'] == 2 * sum(4 + 6 + 3 for _ in range(10)) and meta['author'] == 'Nobody'


def test_full_windows_cover_the_document():
    par = [k // 3 for k in range(100)]
    w = prepare.make_windows(par, 0, 10, 15, full=True)
    assert w[0][0] == 0 and w[-1][1] == 100 and all(a[1] == b[0] for a, b in zip(w, w[1:]))
    assert all(b - a <= 15 for a, b in w)


def _complete_state(win):
    """A finished annotation: 1-1 links, the last window sentence without counterpart, one 2-1 link."""
    a, b = win['start'], win['end']
    links = [{'src': [i], 'trg': [100 + i - a], 'loose': False} for i in range(a, b - 3)]
    links.append({'src': [b - 3, b - 2], 'trg': [100 + b - 3 - a + 1], 'loose': True})   # skips one target sentence
    return {'links': links, 'src_null': [b - 1], 'trg_null': [], 'flags': [], 'comment': ''}


def test_check_and_gold_export_round_trip(data):
    st = store.Store(data)
    win = st.windows('demo')[1]
    state = _complete_state(win)
    assert store.check(state, win, 'src', 10 ** 6) == []
    lines = store.gold_lines(state)
    # the target sentence between the last two links is null; every window sentence appears once
    tmp = os.path.join(data, 'g.txt')
    with open(tmp, 'w', encoding='utf-8') as f:
        f.write('# window x\n' + '\n'.join(lines) + '\n')
    (name, links), = ew.read_windows(tmp)
    src_ids, trg_ids = ew.check_window(links)                     # passes the strict rules of the scorer
    assert set(range(win['start'], win['end'])) == src_ids
    assert ('[]:[%d]' % (100 + win['end'] - 3 - win['start'])) in lines


def test_check_finds_the_problems(data):
    st = store.Store(data)
    win = st.windows('demo')[0]
    s = _complete_state(win)
    s['src_null'] = []
    assert any('neither linked' in p for p in store.check(s, win, 'src', 10 ** 6))
    s = _complete_state(win)
    s['links'][1]['trg'] = [s['links'][0]['trg'][0]]
    assert any('two links' in p for p in store.check(s, win, 'src', 10 ** 6))
    s = _complete_state(win)
    s['links'][0]['trg'], s['links'][1]['trg'] = s['links'][1]['trg'], s['links'][0]['trg']
    assert any('cross' in p for p in store.check(s, win, 'src', 10 ** 6))
    s = _complete_state(win)
    s['links'][0]['src'] = [win['start'], win['start'] + 2]
    assert any('not consecutive' in p for p in store.check(s, win, 'src', 10 ** 6))


def test_web_app_flow(data):
    import app as webapp
    webapp.main(['add-user', data, 'anna', '--password', 'pw1'])
    webapp.main(['add-user', data, 'boss', '--password', 'pw2', '--admin'])
    prepare.main(['assign', data, '--primary', 'anna', '--second', 'boss', '--overlap', '2'])
    application = webapp.create_app(data, secret=b'test')
    c = application.test_client()
    assert c.get('/api/assignments').status_code == 401
    assert c.post('/login', data={'username': 'anna', 'password': 'wrong'}).headers['Location'].endswith('wrong=1')
    c.post('/login', data={'username': 'anna', 'password': 'pw1'})
    rows = c.get('/api/assignments').get_json()
    assert len(rows) == len(store.Store(data).windows('demo')) and rows[0]['status'] == 'open'
    n = rows[1]['window']
    d = c.get('/api/window/demo/%d' % n).get_json()
    assert d['window_sentences'] and d['other_sentences'] and d['side'] == 'src'
    assert 'score' not in json.dumps(d)                          # no system output reaches the annotator
    win = d['window']
    bad = {'links': [], 'src_null': [], 'trg_null': [], 'flags': []}
    r = c.post('/api/window/demo/%d' % n, json={'state': bad, 'status': 'done', 'seconds': 5})
    assert r.status_code == 400 and r.get_json()['problems']
    r = c.post('/api/window/demo/%d' % n, json={'state': _complete_state(win), 'status': 'done', 'seconds': 30})
    assert r.get_json()['ok']
    assert c.get('/api/search/demo/target?q=more 3.').get_json()
    assert c.get('/api/text/demo/source?from=0&to=5').get_json()[0]['i'] == 0
    assert c.get('/admin').status_code == 403
    # export: one finished window
    out = os.path.join(data, 'export')
    prepare.main(['export', data, out])
    gold = open(os.path.join(out, 'demo', 'gold.txt'), encoding='utf-8').read()
    assert gold.startswith('# window %d' % n)
    meta = json.load(open(os.path.join(out, 'demo', 'meta.json'), encoding='utf-8'))
    assert meta['windows_done'] == 1
    # admin and agreement
    c2 = application.test_client()
    c2.post('/login', data={'username': 'boss', 'password': 'pw2'})
    assert c2.get('/admin').status_code == 200
    second = [r for r in c2.get('/api/assignments').get_json() if r['role'] == 'second']
    if any(r['window'] == n for r in second):
        c2.post('/api/window/demo/%d' % n, json={'state': _complete_state(win), 'status': 'done', 'seconds': 1})
        agr = c2.get('/api/admin/agreement').get_json()
        assert agr['total']['f1_strict'] == 1.0
    assert c2.get('/api/admin/agreement').status_code == 200


def test_assigning_again_replaces_the_old_assignment(data):
    prepare.main(['assign', data, '--primary', 'trial'])
    prepare.main(['assign', data, '--primary', 'anna', '--second', 'trial', '--overlap', '2'])
    rows = store.Store(data).assignments(None)
    assert {r['annotator'] for r in rows if r['role'] == 'primary'} == {'anna'}
    assert sum(1 for r in rows if r['annotator'] == 'trial') == 2


def test_prefilled_windows_are_served_checked_and_counted_on_export(data, tmp_path):
    st = store.Store(data)
    w = st.windows('demo')[0]
    side = st.meta('demo')['window_side']
    assert st.prefill_state('demo', w['n']) is None
    # a system alignment: every window-side sentence linked 1-1 to the target sentence with the same number
    links = [{'src': [i], 'trg': [i], 'score': 0.9 if i % 5 else 0.2} for i in range(len(st.side('demo', 'source')['sentences']))
             if i < len(st.side('demo', 'target')['sentences'])]
    with open(os.path.join(data, 'pairs', 'demo', 'prefill.json'), 'w') as f:
        json.dump({'system': 'test', 'links': links}, f)
    st = store.Store(data)
    state = st.prefill_state('demo', w['n'])
    assert state['prefilled'] and state['links'] and all('score' in l for l in state['links'])
    other = 'target' if side == 'src' else 'source'
    window_side = 'source' if side == 'src' else 'target'
    assert store.check(state, w, side, len(st.side('demo', other)['sentences']),
                       len(st.side('demo', window_side)['sentences'])) == []
    # the annotator finishes after changing one link: the export counts the unchanged ones
    state['links'][0]['trg'] = [state['links'][0]['trg'][0] + 1]
    st.assign('anna', 'demo', w['n'], 'primary')
    st.save('anna', 'demo', w['n'], state, 'done', 5)
    out = str(tmp_path / 'out')
    prepare.main(['export', data, out, '--annotator', 'anna'])
    meta = json.load(open(os.path.join(out, 'demo', 'meta.json')))
    assert meta['prefill_links'] == len(state['links']) and meta['prefill_links_unchanged'] == len(state['links']) - 1
