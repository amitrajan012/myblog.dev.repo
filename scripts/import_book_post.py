"""Turn a plain Markdown post (e.g. from the book folder's blog/) into a book-blog post.

Usage:
  python3 scripts/import_book_post.py SOURCE.md [--kind concept|excerpt] [--chapter N]
                                      [--section 6.5] [--subtitle "..."] [--tags "A, B"]

What it does:
  * takes the title from the first "# " heading (and removes that heading from the body)
  * writes content/book-blog/<file-name>.md with front matter
  * copies local images (![alt](../figures/x.svg)) into static/img/book-blog/<file-name>/
    and points the post at the copies
  * keeps LaTeX intact: the site's Markdown passes $...$ and $$...$$ through untouched
    (Goldmark passthrough), so the TeX is copied as written. The one thing Markdown can
    still break is a maths line that starts with "+", "-", "=", "*" or "1." (read as a
    list item or heading underline), so such lines are joined onto the line above.
Re-running it overwrites the blog copy; the source file is never changed.
"""
import argparse, datetime, json, os, re, shutil, string, sys

HERE = os.path.dirname(os.path.abspath(__file__))
PUNCT = set(string.punctuation) - {'$'}

BAD_LINE = re.compile(r'^\s*([+\-=*]|\d+[.)])(\s|$)')

def protect(tex):
    lines = tex.split('\n'); out = [lines[0]]
    for ln in lines[1:]:
        if BAD_LINE.match(ln): out[-1] = out[-1].rstrip() + ' ' + ln.strip()
        else: out.append(ln)
    return '\n'.join(out)

def protect_math(md):
    out, i, n = [], 0, len(md)
    in_code = False
    while i < n:
        if md.startswith('```', i) and (i == 0 or md[i-1] == '\n'):
            j = md.find('\n```', i + 3)
            j = n if j < 0 else md.find('\n', j + 4) if md.find('\n', j + 4) >= 0 else n
            out.append(md[i:j]); i = j; continue
        if md[i] == '`':
            j = md.find('`', i + 1)
            if j > 0: out.append(md[i:j+1]); i = j + 1; continue
        if md.startswith('$$', i):
            j = md.find('$$', i + 2)
            if j < 0: sys.exit('Unclosed $$ block')
            out.append('$$' + protect(md[i+2:j]) + '$$'); i = j + 2; continue
        if md[i] == '$' and (i == 0 or md[i-1] != '\\'):
            j = i + 1
            while j < n and not (md[j] == '$' and md[j-1] != '\\'):
                if md[j] == '\n' and md[j+1:j+2] == '\n': j = -1; break
                j += 1
            if j > 0 and j < n:
                out.append('$' + protect(md[i+1:j]) + '$'); i = j + 1; continue
        out.append(md[i]); i += 1
    return ''.join(out)

def copy_images(body, src_dir, slug):
    def repl(m):
        alt, url = m.group(1), m.group(2).strip()
        if re.match(r'^(https?:|/|data:)', url): return m.group(0)
        src = os.path.normpath(os.path.join(src_dir, url))
        if not os.path.exists(src): sys.exit(f'Image not found: {src}')
        dest_dir = os.path.join(HERE, '..', 'static', 'img', 'book-blog', slug)
        os.makedirs(dest_dir, exist_ok=True)
        shutil.copy2(src, os.path.join(dest_dir, os.path.basename(src)))
        print('copied image', os.path.basename(src))
        return f'![{alt}](/img/book-blog/{slug}/{os.path.basename(src)})'
    return re.sub(r'!\[([^\]]*)\]\(([^)\s]+)\)', repl, body)

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('source'); ap.add_argument('--kind', default='concept', choices=['concept', 'excerpt'])
    ap.add_argument('--chapter', type=int); ap.add_argument('--section'); ap.add_argument('--subtitle')
    ap.add_argument('--tags', default=''); ap.add_argument('--date')
    a = ap.parse_args()
    text = open(a.source, encoding='utf-8').read()
    m = re.search(r'^# (.+)\n', text, re.M)
    if not m: sys.exit('No "# Title" heading found')
    title = m.group(1).strip()
    body = (text[:m.start()] + text[m.end():]).lstrip('\n')
    slug = os.path.splitext(os.path.basename(a.source))[0]
    body = copy_images(body, os.path.dirname(os.path.abspath(a.source)), slug)
    dest = os.path.join(HERE, '..', 'content', 'book-blog', slug + '.md')
    date = a.date
    if not date and os.path.exists(dest):   # keep the original publish date on re-import
        dm = re.search(r'^date = (\S+)', open(dest, encoding='utf-8').read(), re.M)
        date = dm.group(1) if dm else None
    date = date or datetime.datetime.now().astimezone().replace(microsecond=0).isoformat()
    fm = ['+++', f'title = {json.dumps(title, ensure_ascii=False)}', f'date = {date}',
          'draft = false', f'postkind = "{a.kind}"']
    if a.subtitle: fm.append(f'subtitle = {json.dumps(a.subtitle, ensure_ascii=False)}')
    if a.chapter: fm.append(f'chapter = {a.chapter}')
    if a.section: fm.append(f'section = "{a.section}"')
    tags = [t.strip() for t in a.tags.split(',') if t.strip()]
    book = re.search(r'^title:\s*"(.*)"', open(os.path.join(HERE, '..', 'data', 'book.yaml'), encoding='utf-8').read(), re.M)
    if book and book.group(1) not in tags: tags.insert(0, book.group(1))   # every book post carries the book's tag
    if tags: fm.append('tags = ' + json.dumps(tags, ensure_ascii=False))
    fm.append('+++')
    open(dest, 'w', encoding='utf-8').write('\n'.join(fm) + '\n\n' + protect_math(body))
    print('wrote', os.path.relpath(dest))

if __name__ == '__main__':
    main()
