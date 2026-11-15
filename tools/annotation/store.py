"""Storage of the annotation app: the prepared pairs (JSON files) and the annotations (SQLite).

An annotation of one window is a JSON object
    {"links": [{"src": [12, 13], "trg": [15], "loose": false}, ...],
     "src_null": [14, ...], "trg_null": [...], "flags": [{"side": "src", "i": 13, "note": "..."}], "comment": "..."}
with sentence numbers of the whole document.  The window side's sentences must each be in one link or in the null
list; on the other side, the sentences between the first and the last linked one that are in no link are null too
(they are added on export).
"""
import json
import os
import sqlite3
import threading
import time

SCHEMA = """
create table if not exists assignment (annotator text, pair text, window integer, role text,
                                        primary key (annotator, pair, window));
create table if not exists annotation (annotator text, pair text, window integer, state text, status text,
                                        seconds real, updated real, primary key (annotator, pair, window));
create table if not exists event (annotator text, pair text, window integer, at real, what text);
"""


class Store:
    def __init__(self, data):
        self.data = data
        self.db_path = os.path.join(data, 'annotations.db')
        self._local = threading.local()
        self._cache = {}
        with self.db() as db:
            db.executescript(SCHEMA)

    def db(self):
        conn = getattr(self._local, 'conn', None)
        if conn is None:
            conn = sqlite3.connect(self.db_path, timeout=30)
            conn.execute('pragma journal_mode=wal')
            self._local.conn = conn
        return conn

    # ------------------------------------------------------------------ prepared pairs
    def pair_ids(self):
        root = os.path.join(self.data, 'pairs')
        return sorted(d for d in os.listdir(root) if os.path.isdir(os.path.join(root, d))) if os.path.isdir(root) else []

    def _json(self, pid, name):
        key = (pid, name)
        if key not in self._cache:
            with open(os.path.join(self.data, 'pairs', pid, name + '.json'), encoding='utf-8') as f:
                self._cache[key] = json.load(f)
        return self._cache[key]

    def meta(self, pid):
        return self._json(pid, 'meta')

    def side(self, pid, which):
        return self._json(pid, which)

    def windows(self, pid):
        return self._json(pid, 'windows')

    def window(self, pid, n):
        for w in self.windows(pid):
            if w['n'] == n:
                return w
        raise KeyError((pid, n))

    # ------------------------------------------------------------------ pre-filled alignment
    def prefill(self, pid):
        """The system alignment written by `prepare.py prefill` ({'system': ..., 'links': [{src, trg, score}]}) or None."""
        key = (pid, 'prefill')
        if key not in self._cache:
            path = os.path.join(self.data, 'pairs', pid, 'prefill.json')
            if os.path.exists(path):
                with open(path, encoding='utf-8') as f:
                    self._cache[key] = json.load(f)
            else:
                self._cache[key] = None
        return self._cache[key]

    def prefill_state(self, pid, n):
        """Initial state of a window from the pre-filled alignment, or None: the links that touch the window on the
        window side, and the window sentences that are in no link as 'no counterpart'."""
        pre = self.prefill(pid)
        if pre is None:
            return None
        side = self.meta(pid)['window_side']
        w = self.window(pid, n)
        links, covered = [], set()
        for ln in pre['links']:
            ids = ln[side]
            if ids and ids[-1] >= w['start'] and ids[0] < w['end']:
                links.append({'src': list(ln['src']), 'trg': list(ln['trg']), 'loose': False,
                              'score': round(ln['score'], 3)})
                covered.update(ids)
        null = [i for i in range(w['start'], w['end']) if i not in covered]
        state = {'links': links, 'src_null': [], 'trg_null': [], 'flags': [], 'comment': '', 'prefilled': True}
        state[side + '_null'] = null
        return state

    # ------------------------------------------------------------------ assignments and annotations
    def clear_assignments(self, pid):
        """Remove the assignments of a pair (the annotations stay)."""
        with self.db() as db:
            db.execute('delete from assignment where pair = ?', (pid,))

    def assign(self, annotator, pid, n, role):
        with self.db() as db:
            db.execute('insert or replace into assignment values (?, ?, ?, ?)', (annotator, pid, n, role))

    def assignments(self, annotator=None):
        q = 'select a.annotator, a.pair, a.window, a.role, coalesce(n.status, "open"), coalesce(n.seconds, 0) ' \
            'from assignment a left join annotation n on a.annotator = n.annotator and a.pair = n.pair ' \
            'and a.window = n.window'
        args = ()
        if annotator:
            q += ' where a.annotator = ?'
            args = (annotator,)
        return [dict(zip(('annotator', 'pair', 'window', 'role', 'status', 'seconds'), r))
                for r in self.db().execute(q + ' order by a.pair, a.window', args)]

    def is_assigned(self, annotator, pid, n):
        return self.db().execute('select 1 from assignment where annotator = ? and pair = ? and window = ?',
                                 (annotator, pid, n)).fetchone() is not None

    def load(self, annotator, pid, n):
        r = self.db().execute('select state, status, seconds from annotation where annotator = ? and pair = ? and '
                              'window = ?', (annotator, pid, n)).fetchone()
        if r is None:
            return None, 'open', 0.0
        return json.loads(r[0]), r[1], r[2]

    def save(self, annotator, pid, n, state, status, add_seconds):
        _, _, seconds = self.load(annotator, pid, n)
        with self.db() as db:
            db.execute('insert or replace into annotation values (?, ?, ?, ?, ?, ?, ?)',
                       (annotator, pid, n, json.dumps(state, ensure_ascii=False), status,
                        seconds + max(0.0, add_seconds), time.time()))
            db.execute('insert into event values (?, ?, ?, ?, ?)', (annotator, pid, n, time.time(), status))

    def final_annotation(self, pid, n, annotator=None):
        """The finished annotation of a window: of `annotator`, or of whoever has the primary assignment."""
        if annotator is None:
            r = self.db().execute('select annotator from assignment where pair = ? and window = ? and role = "primary"',
                                  (pid, n)).fetchone()
            if r is None:
                return None
            annotator = r[0]
        state, status, _ = self.load(annotator, pid, n)
        return state if status == 'done' else None


# ---------------------------------------------------------------------- checks and export
def _bad_numbers(state, limits):
    """Sentence numbers that are not integers or lie outside [0, limit) of their side (`limits`: side -> count or
    None for no limit), in the links and the null lists."""
    problems = []
    lists = [('link %d' % (k + 1), s, ln.get(s, [])) for k, ln in enumerate(state.get('links', [])) for s in ('src', 'trg')]
    lists += [('the %s list of unaligned sentences' % ('source' if s == 'src' else 'target'), s, state.get(s + '_null', []))
              for s in ('src', 'trg')]
    for where, s, ids in lists:
        name = 'source' if s == 'src' else 'target'
        if not isinstance(ids, list):
            problems.append('%s: the %s sentences are not a list' % (where, name))
            continue
        for i in ids:
            if not isinstance(i, int) or isinstance(i, bool):
                problems.append('%s: %r is not a sentence number' % (where, i))
            elif i < 0 or (limits[s] is not None and i >= limits[s]):
                problems.append('%s: %s sentence number %d does not exist' % (where, name, i + 1))
    return problems


def check(state, window, side, n_other, n_window=None):
    """Problems of an annotation, as a list of messages (empty: the window can be finished).
    `side` is 'src' or 'trg' (the window side); `n_window` / `n_other` are the sentence counts of the two
    documents (`n_window` None: not checked)."""
    other = 'trg' if side == 'src' else 'src'
    bad = _bad_numbers(state, {side: n_window, other: n_other})
    if bad:
        return bad          # the other checks need valid numbers
    problems = []
    seen = {'src': {}, 'trg': {}}
    for k, ln in enumerate(state.get('links', [])):
        for s in ('src', 'trg'):
            ids = ln.get(s, [])
            if not ids:
                problems.append('link %d has no %s sentence' % (k + 1, 'source' if s == 'src' else 'target'))
            if ids != list(range(min(ids), max(ids) + 1)) if ids else False:
                problems.append('link %d: the %s sentences are not consecutive' % (k + 1, 'source' if s == 'src' else 'target'))
            for i in ids:
                if i in seen[s]:
                    problems.append('%s sentence %d is in two links' % ('source' if s == 'src' else 'target', i + 1))
                seen[s][i] = k
    for s in ('src', 'trg'):
        for i in state.get(s + '_null', []):
            if i in seen[s]:
                problems.append('%s sentence %d is both linked and unaligned' % ('source' if s == 'src' else 'target', i + 1))
    nulls = set(state.get(side + '_null', []))
    missing = [i for i in range(window['start'], window['end']) if i not in seen[side] and i not in nulls]
    if missing:
        problems.append('%d sentence(s) of the window are neither linked nor marked unaligned (first: %d)'
                        % (len(missing), missing[0] + 1))
    # links in order on both sides (sentence alignment is monotone)
    links = sorted(state.get('links', []), key=lambda l: min(l.get(side) or [10 ** 9]))
    last = -1
    for ln in links:
        if ln.get(other):
            if min(ln[other]) <= last:
                problems.append('links cross: the link with %s sentence %d comes before an earlier link on the other '
                                'side' % ('source' if side == 'src' else 'target', min(ln[side]) + 1))
                break
            last = max(ln[other])
    return problems


def gold_lines(state):
    """The window as gold lines '[12, 13]:[15]' (BleuAlign / ParIce format), in document order, with the
    null lines of both sides; unlinked sentences of the other side between the first and the last link are null."""
    links = [(sorted(l['src']), sorted(l['trg'])) for l in state.get('links', [])]
    src_null = set(state.get('src_null', []))
    trg_null = set(state.get('trg_null', []))
    linked_t = {j for _, t in links for j in t}
    linked_s = {i for s, _ in links for i in s}
    if linked_t:
        trg_null |= {j for j in range(min(linked_t), max(linked_t) + 1) if j not in linked_t}
    if linked_s:
        src_null |= {i for i in range(min(linked_s), max(linked_s) + 1) if i not in linked_s}
    rows = [((s[0], 0), '%s:%s' % (s, t)) for s, t in links]
    rows += [((i, 1), '[%d]:[]' % i) for i in src_null]
    # a target null goes before the first link whose target starts after it
    for j in trg_null:
        after = [s[0] for s, t in links if t and t[0] > j]
        key = (min(after), -1) if after else (max([s[-1] for s, _ in links] + list(src_null) + [-1]) + 0.5, 2)
        rows.append((key, '[]:[%d]' % j))
    rows.sort(key=lambda r: r[0])
    return [r[1] for r in rows]
