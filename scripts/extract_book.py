"""Regenerate data/book.yaml (parts, chapters, sections) from the Quarto book project.

Usage:  python3 scripts/extract_book.py ["/path/to/ML Personal Book"]
Default book path: ~/Desktop/ML Personal Book. No third-party packages needed.
"""
import re, os, sys, json
HERE = os.path.dirname(os.path.abspath(__file__))
BOOK = os.path.expanduser(sys.argv[1] if len(sys.argv) > 1 else '~/Desktop/ML Personal Book')
OUT  = os.path.join(HERE, '..', 'data', 'book.yaml')
# Newer drafts that supersede the root .qmd (root file is what Quarto builds today).
OVERRIDE = {'Chapter_04_Probability.qmd': '_draft/Chapter_04_Probability_v3.qmd',
            'Chapter_24_Attention_And_The_Transformer.qmd': '_draft/Chapter_24_Attention_And_The_Transformer_v1.qmd'}

def clean(h):
    h = re.sub(r'\s*\{[^}]*\}\s*$', '', h).strip()
    return h

def headings(path):
    out, fence = [], False
    for line in open(path, encoding='utf-8'):
        if line.lstrip().startswith(('```', '~~~')): fence = not fence; continue
        if fence: continue
        m = re.match(r'^(#{1,3}) (.+)$', line.rstrip('\n'))
        if m: out.append((len(m.group(1)), clean(m.group(2)), m.group(2)))
    return out

def is_stub(path):
    t = open(path, encoding='utf-8').read()
    return len(t) < 2000 or 'Draft pending' in t

def stub_summary(path):
    t = open(path, encoding='utf-8').read()
    t = re.sub(r':::.*?:::', '', t, flags=re.S)
    paras = [p.strip() for p in t.split('\n\n') if p.strip() and not p.strip().startswith('#')]
    return paras[0].strip('*') if paras else ''

def key(title):
    head = re.split(r'\s[—:\-]\s|:|—', title)[0]
    w = re.findall(r'[a-z]+', head.lower())
    return ' '.join(w[:2])

# Planned sections from the master outline (used only for chapters not drafted yet).
outline = {}
cur = None
for lvl, h, raw in headings(os.path.join(BOOK, 'manuscript/combined/ml_book_full_manuscript_by_outline.md')):
    if lvl == 2:
        m = re.match(r'^(Ch \d+: |Interlude — |Epilogue — )?(.*)$', h)
        name = 'interlude' if h.startswith('Interlude') else 'epilogue' if h.startswith('Epilogue') else key(m.group(2))
        cur = outline.setdefault(name, [])
    elif lvl == 3 and cur is not None:
        t = re.sub(r'^([0-9]+|I|E)\.\d+\s+', '', h)
        if 'Closing notes' in t: continue
        cur.append(t)

def read_quarto(path):
    # Minimal reader for book.chapters in _quarto.yml (plain files and `- part:` blocks).
    items, part = [], None
    in_book = in_ch = False
    for line in open(path, encoding='utf-8'):
        if re.match(r'^book:', line): in_book = True; continue
        if in_book and re.match(r'^\S', line): break
        if in_book and re.match(r'^  chapters:', line): in_ch = True; continue
        if not in_ch: continue
        m = re.match(r'^    - part:\s*"?(.*?)"?\s*$', line)
        if m: part = {'part': m.group(1), 'chapters': []}; items.append(part); continue
        m = re.match(r'^        - (\S.*?)\s*$', line)
        if m and part is not None: part['chapters'].append(m.group(1)); continue
        m = re.match(r'^    - (\S.*?)\s*$', line)
        if m: part = None; items.append(m.group(1))
    return {'chapters': items}

q = read_quarto(os.path.join(BOOK, '_quarto.yml'))

def chapter(fname, number, label=None):
    path = os.path.join(BOOK, OVERRIDE.get(fname, fname))
    hs = headings(path)
    title = next((h for l, h, _ in hs if l == 1), fname)
    title = re.sub(r'^(Epilogue|Interlude)\s*[—-]\s*', '', title)
    prefix = str(number) if number else ('I' if label == 'Interlude' else 'E')
    ch = {'number': number, 'label': label, 'title': title, 'source': OVERRIDE.get(fname, fname)}
    if is_stub(path):
        k = 'interlude' if label == 'Interlude' else 'epilogue' if label == 'Epilogue' else key(title)
        ch['status'] = 'planned'
        s = stub_summary(path)
        if s and not s.startswith('This chapter'): ch['summary'] = s
        ch['sections'] = [{'num': f'{prefix}.{i}', 'title': t} for i, t in enumerate(outline.get(k, []), 1)]
    else:
        ch['status'] = 'drafted'
        secs = []
        for l, h, raw in hs:
            if l == 2:
                secs.append({'num': f'{prefix}.{len(secs)+1}', 'title': h, 'subs': []})
            elif l == 3 and secs and not h.startswith(('Under the Hood', 'Figure')):
                secs[-1]['subs'].append(h)
        for s in secs:
            if not s['subs']: del s['subs']
        ch['sections'] = secs
    return ch

parts, epilogue = [], None
for item in q['chapters']:
    if isinstance(item, dict):
        chs = []
        for f in item['chapters']:
            m = re.match(r'Chapter_(\d+)_', f)
            chs.append(chapter(f, int(m.group(1)) if m else None, None if m else 'Interlude'))
        parts.append({'title': item['part'], 'chapters': chs})
    elif item.startswith('Epilogue'):
        epilogue = chapter(item, None, 'Epilogue')

# keep the hand-written header (title, subtitle, about...) from the existing file
old = open(OUT, encoding='utf-8').read()
header = old[:old.index('parts:')]
header = re.sub(r'^#.*\n', '', header, flags=re.M)

def dump(o, ind=0):
    sp = '  ' * ind
    lines = []
    if isinstance(o, dict):
        for k, v in o.items():
            if v is None: continue
            if isinstance(v, (dict, list)):
                if not v: continue
                lines.append(f'{sp}{k}:'); lines += dump(v, ind + 1)
            else:
                lines.append(f'{sp}{k}: {json.dumps(v, ensure_ascii=False)}')
    else:
        for v in o:
            if isinstance(v, dict):
                sub = dump(v, ind + 1)
                sub[0] = sp + '- ' + sub[0].lstrip()
                lines += sub
            else:
                lines.append(f'{sp}- {json.dumps(v, ensure_ascii=False)}')
    return lines

body = dump({'parts': parts, 'epilogue': epilogue})
banner = ('# The Book — data for the /book/ contents page.\n'
          '# Parts, chapters and sections are GENERATED from the Quarto project by\n'
          '#   scripts/extract_book.py   (re-run it after restructuring the book).\n'
          '# Drafted chapters use their real headings; planned ones use the master outline.\n'
          '# status: published (has blog posts) | drafted | planned\n')
open(OUT, 'w', encoding='utf-8').write(banner + header + '\n'.join(body) + '\n')
n = sum(1 for p in parts for c in p['chapters'] if c['number'])
print('parts', len(parts), 'numbered chapters', n, 'drafted', sum(1 for p in parts for c in p['chapters'] if c['status']=='drafted'))
for p in parts:
    print('#', p['title'])
    for c in p['chapters']:
        print(f"  {c['number'] or c['label']:>9} {c['status']:8} {len(c['sections']):2} secs  {c['title'][:60]}")
print('  Epilogue', epilogue['status'], len(epilogue['sections']), epilogue['title'])
