#!/usr/bin/env python3
"""Prepare document pairs for the annotation app, assign windows to annotators, and export the gold standard.

    prepare.py add-pair DATA ID --src SRC.xml --trg TRG.xml [TRG2.xml ...] --src-lang fr --trg-lang is
                        [--window-side src|trg] [--windows 10] [--min 30] [--max 60] [--full] [--seed 1]
                        [--meta key=value ...]
    prepare.py assign   DATA --primary NAME [--second NAME --overlap 2] [--pairs ID ...]   (replaces earlier assignments)
    prepare.py prefill  DATA [--pairs ID ...] [--option NAME=VALUE ...]   (SentAlign alignment to be corrected)
    prepare.py export   DATA OUT_DIR [--annotator NAME]     -> OUT_DIR/<pair>/{source.txt,target.txt,gold.txt,meta.json}

The XML files are those of the SentAlign book pipeline (chapter / p / s).  The sentences are numbered exactly as in
the alignment input (every <s> in document order, whitespace collapsed), so the gold refers to the same sentence
numbers as the aligner's output.  Several target files (an Icelandic book in volumes) are joined in the order given.

Windows are stretches of consecutive paragraphs on one side (`--window-side`, default the source; use the target
side for abridged translations, where most of the source has no counterpart).  Their length varies between --min
and --max sentences because they start and end at paragraph boundaries (a paragraph longer than --max is cut).  They
are spread evenly over the document: one near the beginning, one near the end.  `--full` cuts the whole document
into consecutive windows instead (complete annotation of a short document).
"""
import argparse
import json
import os
import random
import sys
import xml.etree.ElementTree as ET


def read_xml(paths):
    """-> (sentences, paragraph index of each sentence) for one or more XML files joined in order."""
    sentences, paragraph = [], []
    p = -1
    for path in paths:
        root = ET.parse(path).getroot()
        for ch in root.iter('chapter'):
            for par in ch.findall('p'):
                p += 1
                for s in par.findall('s'):
                    sentences.append(' '.join((s.text or '').split()))
                    paragraph.append(p)
    return sentences, paragraph


def paragraph_starts(paragraph):
    return [i for i in range(len(paragraph)) if i == 0 or paragraph[i] != paragraph[i - 1]]


def make_windows(paragraph, n_windows, min_len, max_len, full=False, seed=1):
    """[(start, end)] (end exclusive) on the window side; consecutive paragraphs, length between min and max where the
    paragraphs allow it, evenly spread (or covering everything with --full)."""
    n = len(paragraph)
    starts = paragraph_starts(paragraph) + [n]
    if n == 0:
        return []

    def window_from(a):
        # extend paragraph by paragraph until at least min_len; never beyond max_len (cut inside a long paragraph)
        b = a
        for s in starts:
            if s <= a:
                continue
            b = s
            if b - a >= min_len:
                break
        return a, min(b, a + max_len, n)

    if full:
        out, a = [], 0
        while a < n:
            a, b = window_from(a)
            out.append((a, b))
            a = b
        return out
    rng = random.Random(seed)
    span = max(0, n - min_len)
    out = []
    for k in range(n_windows):
        # evenly spread centres, jittered by up to a quarter of the spacing; snapped back to a paragraph start
        target = span * k / max(1, n_windows - 1)
        jitter = rng.uniform(-0.25, 0.25) * span / max(1, n_windows) if 0 < k < n_windows - 1 else 0
        pos = int(max(0, min(span, target + jitter)))
        a = max(s for s in starts if s <= pos) if pos > 0 else 0
        if out and a < out[-1][1]:
            a = out[-1][1]                                   # never overlap the previous window
            if a >= n:
                break
        out.append(window_from(a))
    return [w for w in out if w[1] > w[0]]


def add_pair(args):
    src, src_par = read_xml([args.src])
    trg, trg_par = read_xml(args.trg)
    side_par = src_par if args.window_side == 'src' else trg_par
    windows = make_windows(side_par, args.windows, args.min, args.max, args.full, args.seed)
    folder = os.path.join(args.data, 'pairs', args.id)
    os.makedirs(folder, exist_ok=True)
    meta = {'id': args.id, 'src_lang': args.src_lang, 'trg_lang': args.trg_lang, 'window_side': args.window_side,
            'src_files': [os.path.basename(args.src)], 'trg_files': [os.path.basename(t) for t in args.trg],
            'n_src': len(src), 'n_trg': len(trg), 'full': args.full, 'seed': args.seed}
    for kv in args.meta or []:
        k, _, v = kv.partition('=')
        meta[k] = v
    with open(os.path.join(folder, 'meta.json'), 'w', encoding='utf-8') as f:
        json.dump(meta, f, ensure_ascii=False, indent=1)
    with open(os.path.join(folder, 'source.json'), 'w', encoding='utf-8') as f:
        json.dump({'sentences': src, 'paragraph': src_par}, f, ensure_ascii=False)
    with open(os.path.join(folder, 'target.json'), 'w', encoding='utf-8') as f:
        json.dump({'sentences': trg, 'paragraph': trg_par}, f, ensure_ascii=False)
    with open(os.path.join(folder, 'windows.json'), 'w', encoding='utf-8') as f:
        json.dump([{'n': k + 1, 'start': a, 'end': b} for k, (a, b) in enumerate(windows)], f)
    total = sum(b - a for a, b in windows)
    print('%s: %d source, %d target sentences; %d windows on the %s side, %d sentences (%d-%d per window)'
          % (args.id, len(src), len(trg), len(windows), args.window_side, total,
             min(b - a for a, b in windows), max(b - a for a, b in windows)))


def assign(args):
    sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
    from store import Store
    st = Store(args.data)
    pairs = args.pairs or st.pair_ids()
    rng = random.Random(7)
    n = 0
    for pid in pairs:
        windows = st.windows(pid)
        st.clear_assignments(pid)          # a new assignment replaces the old one (annotations are kept)
        for w in windows:
            st.assign(args.primary, pid, w['n'], 'primary')
            n += 1
        if args.second:
            for w in rng.sample(windows, min(args.overlap, len(windows))):
                st.assign(args.second, pid, w['n'], 'second')
                n += 1
    print('%d assignments' % n)


def prefill(args):
    """Align each prepared pair with SentAlign and store the links as pairs/<id>/prefill.json, so that the annotator
    corrects an alignment instead of making one.  The sentences are those of the pair (same numbering as the gold)."""
    sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
    from store import Store
    import sentalign
    st = Store(args.data)
    opts = {}
    for kv in args.option or []:
        k, v = kv.split('=', 1)
        for conv in (int, float, str):
            try:
                opts[k] = conv(v)
                break
            except ValueError:
                pass
    for pid in args.pairs or st.pair_ids():
        src, trg = st.side(pid, 'source')['sentences'], st.side(pid, 'target')['sentences']
        res = sentalign.align(src, trg, **opts)
        links = [{'src': list(ln.src), 'trg': list(ln.trg), 'score': float(ln.score)} for ln in res.aligned()]
        info = {'system': 'SentAlign %s' % getattr(sentalign, '__version__', ''), 'options': opts, 'links': links}
        with open(os.path.join(args.data, 'pairs', pid, 'prefill.json'), 'w', encoding='utf-8') as f:
            json.dump(info, f)
        print('%s: %d links (%.0f %% of the source and %.0f %% of the target sentences)'
              % (pid, len(links), 100 * sum(len(l['src']) for l in links) / max(1, len(src)),
                 100 * sum(len(l['trg']) for l in links) / max(1, len(trg))))


def prefill_changes(st, pid, ann, w):
    """(links of the prefill in this window, how many of them are unchanged in the final annotation)."""
    pre = st.prefill_state(pid, w['n'])
    if pre is None:
        return None
    key = lambda l: (tuple(sorted(l['src'])), tuple(sorted(l['trg'])))
    final = {key(l) for l in ann.get('links', [])}
    return len(pre['links']), sum(1 for l in pre['links'] if key(l) in final)


def export(args):
    sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
    from store import Store, gold_lines
    st = Store(args.data)
    for pid in st.pair_ids():
        meta, src, trg = st.meta(pid), st.side(pid, 'source'), st.side(pid, 'target')
        out = os.path.join(args.out, pid)
        lines, done, todo = [], 0, 0
        pf_total = pf_same = 0
        for w in st.windows(pid):
            ann = st.final_annotation(pid, w['n'], args.annotator)
            if ann is None:
                todo += 1
                continue
            done += 1
            ch = prefill_changes(st, pid, ann, w)
            if ch:
                pf_total, pf_same = pf_total + ch[0], pf_same + ch[1]
            lines.append('# window %d' % w['n'])
            lines.extend(gold_lines(ann))
        os.makedirs(out, exist_ok=True)
        with open(os.path.join(out, 'source.txt'), 'w', encoding='utf-8') as f:
            f.write('\n'.join(src['sentences']) + '\n')
        with open(os.path.join(out, 'target.txt'), 'w', encoding='utf-8') as f:
            f.write('\n'.join(trg['sentences']) + '\n')
        with open(os.path.join(out, 'gold.txt'), 'w', encoding='utf-8') as f:
            f.write('\n'.join(lines) + ('\n' if lines else ''))
        with open(os.path.join(out, 'meta.json'), 'w', encoding='utf-8') as f:
            extra = {'prefill_links': pf_total, 'prefill_links_unchanged': pf_same,
                     'prefill_system': (st.prefill(pid) or {}).get('system')} if st.prefill(pid) else {}
            json.dump(dict(meta, windows_done=done, windows_open=todo, **extra), f, ensure_ascii=False, indent=1)
        print('%s: %d windows exported, %d not finished' % (pid, done, todo))


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.split('\n\n')[0])
    sub = ap.add_subparsers(dest='cmd', required=True)
    a = sub.add_parser('add-pair')
    a.add_argument('data')
    a.add_argument('id')
    a.add_argument('--src', required=True)
    a.add_argument('--trg', required=True, nargs='+')
    a.add_argument('--src-lang', required=True)
    a.add_argument('--trg-lang', required=True)
    a.add_argument('--window-side', choices=('src', 'trg'), default='src')
    a.add_argument('--windows', type=int, default=10)
    a.add_argument('--min', type=int, default=30)
    a.add_argument('--max', type=int, default=60)
    a.add_argument('--full', action='store_true')
    a.add_argument('--seed', type=int, default=1)
    a.add_argument('--meta', nargs='*')
    b = sub.add_parser('assign')
    b.add_argument('data')
    b.add_argument('--primary', required=True)
    b.add_argument('--second')
    b.add_argument('--overlap', type=int, default=2)
    b.add_argument('--pairs', nargs='*')
    pf = sub.add_parser('prefill', help='align the pairs with SentAlign (needs the sentalign package) for correction')
    pf.add_argument('data')
    pf.add_argument('--pairs', nargs='*')
    pf.add_argument('--option', nargs='*', metavar='NAME=VALUE', help='SentAlign options, e.g. encoder=BAAI/bge-m3')
    c = sub.add_parser('export')
    c.add_argument('data')
    c.add_argument('out')
    c.add_argument('--annotator', help='export this annotator\'s version (default: the primary annotator)')
    args = ap.parse_args(argv)
    {'add-pair': add_pair, 'assign': assign, 'export': export, 'prefill': prefill}[args.cmd](args)


if __name__ == '__main__':
    main()
