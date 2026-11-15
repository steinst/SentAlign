#!/usr/bin/env python3
"""Web app for the sentence-alignment gold standard of the SentAlign book test set.

    python app.py add-user DATA NAME --password PW [--admin]
    python app.py serve DATA [--host 0.0.0.0] [--port 5080]

The annotator sees one window at a time: the window's sentences on one side and the other text around the matching
position (more can be loaded, and the text can be searched).  If `prepare.py prefill` has been run, the window starts
with the system's links (with their similarity scores; doubtful ones are marked) and the annotator corrects them;
otherwise it starts empty.  Work is saved continuously; "Finish" checks the window first.
"""
import argparse
import json
import os
import sys

from flask import Flask, abort, jsonify, redirect, render_template, request, session, url_for
from werkzeug.security import check_password_hash, generate_password_hash

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from store import Store, check, gold_lines  # noqa: E402

CONTEXT = 5           # sentences shown before and after the window on the window side
OTHER_SPAN = 120      # sentences of the other side loaded around the proportional position


def users_path(data):
    return os.path.join(data, 'users.json')


def load_users(data):
    try:
        with open(users_path(data), encoding='utf-8') as f:
            return json.load(f)
    except FileNotFoundError:
        return {}


def create_app(data, secret=None):
    app = Flask(__name__, template_folder=os.path.join(os.path.dirname(os.path.abspath(__file__)), 'templates'))
    app.secret_key = secret or _secret(data)
    app.config['DATA'] = data
    store = Store(data)

    def user():
        name = session.get('user')
        if not name or name not in load_users(data):
            abort(401)
        return name

    def is_admin(name):
        return bool(load_users(data).get(name, {}).get('admin'))

    def allowed(name, pid, n):
        if not (store.is_assigned(name, pid, n) or is_admin(name)):
            abort(403)

    @app.get('/')
    def index():
        if not session.get('user') or session['user'] not in load_users(data):
            return render_template('login.html', wrong=request.args.get('wrong'))
        return render_template('index.html', user=session['user'], admin=is_admin(session['user']))

    @app.post('/login')
    def login():
        name = request.form.get('username', '').strip()
        u = load_users(data).get(name)
        if not u or not check_password_hash(u['password'], request.form.get('password', '')):
            return redirect(url_for('index', wrong=1))
        session['user'] = name
        return redirect(url_for('index'))

    @app.get('/logout')
    def logout():
        session.clear()
        return redirect(url_for('index'))

    @app.get('/annotate/<pid>/<int:n>')
    def annotate(pid, n):
        name = user()
        allowed(name, pid, n)
        return render_template('annotate.html', user=name, pid=pid, n=n)

    @app.get('/api/assignments')
    def assignments():
        name = user()
        rows = store.assignments(None if (is_admin(name) and request.args.get('all')) else name)
        for r in rows:
            w = store.window(r['pair'], r['window'])
            r['sentences'] = w['end'] - w['start']
            r['side'] = store.meta(r['pair'])['window_side']
            r['langs'] = '%s-%s' % (store.meta(r['pair'])['src_lang'], store.meta(r['pair'])['trg_lang'])
        return jsonify(rows)

    def slice_side(pid, which, a, b):
        s = store.side(pid, which)
        a, b = max(0, a), min(len(s['sentences']), b)
        return [{'i': i, 'text': s['sentences'][i], 'p': s['paragraph'][i]} for i in range(a, b)]

    @app.get('/api/window/<pid>/<int:n>')
    def window(pid, n):
        name = user()
        allowed(name, pid, n)
        owner = request.args.get('of', name) if is_admin(name) else name
        meta, w = store.meta(pid), store.window(pid, n)
        side = meta['window_side']
        wside, oside = ('source', 'target') if side == 'src' else ('target', 'source')
        n_w, n_o = len(store.side(pid, wside)['sentences']), len(store.side(pid, oside)['sentences'])
        centre = int((w['start'] + w['end']) / 2 / max(1, n_w) * n_o)
        state, status, seconds = store.load(owner, pid, n)
        if state is None:
            pre = store.prefill_state(pid, n)
            if pre is not None:
                state, status = pre, 'prefilled'
        return jsonify({'meta': meta, 'window': w, 'side': side, 'n_window_side': n_w, 'n_other_side': n_o,
                        'window_sentences': slice_side(pid, wside, w['start'] - CONTEXT, w['end'] + CONTEXT),
                        'other_sentences': slice_side(pid, oside, centre - OTHER_SPAN, centre + OTHER_SPAN),
                        'state': state or {'links': [], 'src_null': [], 'trg_null': [], 'flags': [], 'comment': ''},
                        'status': status, 'seconds': seconds, 'owner': owner})

    @app.get('/api/text/<pid>/<which>')
    def text(pid, which):
        user()
        if which not in ('source', 'target'):
            abort(404)
        a, b = int(request.args.get('from', 0)), int(request.args.get('to', 0))
        if b - a > 2000:
            abort(400)
        return jsonify(slice_side(pid, which, a, b))

    @app.get('/api/search/<pid>/<which>')
    def search(pid, which):
        user()
        q = request.args.get('q', '').strip().lower()
        if which not in ('source', 'target') or len(q) < 2:
            return jsonify([])
        s = store.side(pid, which)['sentences']
        hits = [i for i, t in enumerate(s) if q in t.lower()][:50]
        return jsonify([{'i': i, 'text': s[i]} for i in hits])

    @app.post('/api/window/<pid>/<int:n>')
    def save(pid, n):
        name = user()
        if not store.is_assigned(name, pid, n):
            abort(403)
        body = request.get_json(force=True)
        state, status = body.get('state') or {}, body.get('status', 'draft')
        seconds = float(body.get('seconds', 0))
        if status == 'done':
            meta = store.meta(pid)
            wside, other = ('source', 'target') if meta['window_side'] == 'src' else ('target', 'source')
            problems = check(state, store.window(pid, n), meta['window_side'], len(store.side(pid, other)['sentences']),
                             len(store.side(pid, wside)['sentences']))
            if problems:
                store.save(name, pid, n, state, 'draft', seconds)
                return jsonify({'ok': False, 'problems': problems}), 400
        store.save(name, pid, n, state, status if status in ('draft', 'done') else 'draft', seconds)
        return jsonify({'ok': True})

    @app.get('/admin')
    def admin():
        name = user()
        if not is_admin(name):
            abort(403)
        return render_template('admin.html', user=name)

    @app.get('/api/admin/agreement')
    def agreement():
        name = user()
        if not is_admin(name):
            abort(403)
        return jsonify(compute_agreement(store))

    return app


def compute_agreement(store):
    """Strict / lax F1 between the two annotators of every doubly annotated, finished window."""
    here = os.path.dirname(os.path.abspath(__file__))
    sys.path.insert(0, os.path.join(here, '..', '..', 'evaluation'))
    from evaluate import score_multiple
    from ast import literal_eval
    rows = store.assignments(None)
    second = [r for r in rows if r['role'] == 'second']
    out, gold_all, test_all = [], [], []
    for r in second:
        a = store.final_annotation(r['pair'], r['window'])
        b, status, _ = store.load(r['annotator'], r['pair'], r['window'])
        if a is None or b is None or status != 'done':
            continue

        def parse(state):
            res = []
            for line in gold_lines(state):
                s, t = line.split(':')
                res.append((tuple(literal_eval(s)), tuple(literal_eval(t))))
            return res
        ga, gb = parse(a), parse(b)
        res = score_multiple([ga], [gb], True)
        out.append({'pair': r['pair'], 'window': r['window'], 'f1_strict': res['f1_strict'], 'f1_lax': res['f1_lax']})
        gold_all.append(ga)
        test_all.append(gb)
    total = score_multiple(gold_all, test_all, True) if gold_all else None
    return {'windows': out, 'total': total}


def _secret(data):
    path = os.path.join(data, 'secret.key')
    if not os.path.exists(path):
        with open(path, 'wb') as f:
            f.write(os.urandom(32))
        os.chmod(path, 0o600)
    with open(path, 'rb') as f:
        return f.read()


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.split('\n\n')[0])
    sub = ap.add_subparsers(dest='cmd', required=True)
    a = sub.add_parser('add-user')
    a.add_argument('data')
    a.add_argument('name')
    a.add_argument('--password', required=True)
    a.add_argument('--admin', action='store_true')
    s = sub.add_parser('serve')
    s.add_argument('data')
    s.add_argument('--host', default='0.0.0.0')
    s.add_argument('--port', type=int, default=5080)
    args = ap.parse_args(argv)
    if args.cmd == 'add-user':
        os.makedirs(args.data, exist_ok=True)
        users = load_users(args.data)
        users[args.name] = {'password': generate_password_hash(args.password), 'admin': args.admin}
        with open(users_path(args.data), 'w', encoding='utf-8') as f:
            json.dump(users, f, indent=1)
        os.chmod(users_path(args.data), 0o600)
        print('user %s %s' % (args.name, '(admin)' if args.admin else ''))
    else:
        create_app(args.data).run(host=args.host, port=args.port, threaded=True)


if __name__ == '__main__':
    main()
