// Shared search logic for the /search/ page and the home-page quick search.
window.NB = (function () {
  var cache = null;
  var norm = function (s) { return (s || '').toLowerCase().normalize('NFKD').replace(/[̀-ͯ]/g, ''); };
  var esc = function (s) { return String(s).replace(/[&<>"]/g, function (c) { return { '&': '&amp;', '<': '&lt;', '>': '&gt;', '"': '&quot;' }[c]; }); };
  var terms = function (q) { return norm(q).split(/\s+/).filter(Boolean); };
  var mark = function (text, ts) {
    var out = esc(text);
    ts.forEach(function (t) {
      if (t.length < 2) return;
      out = out.replace(new RegExp('(' + t.replace(/[.*+?^${}()|[\]\\]/g, '\\$&') + ')', 'ig'), '<mark>$1</mark>');
    });
    return out;
  };
  var snippet = function (text, ts, before, after) {
    var low = norm(text), at = -1;
    for (var i = 0; i < ts.length && at < 0; i++) at = low.indexOf(ts[i]);
    if (at < 0) return '';
    var start = Math.max(0, at - (before || 70)), end = Math.min(text.length, at + (after || 130));
    return (start ? '…' : '') + mark(text.slice(start, end), ts) + (end < text.length ? '…' : '');
  };
  var score = function (p, ts) {
    var title = norm(p.t), tags = norm(p.g.join(' ')), meta = norm(p.s + ' ' + p.l), text = norm(p.x), total = 0;
    for (var i = 0; i < ts.length; i++) {
      var t = ts[i], s = 0;
      if (title.indexOf(t) >= 0) s += 6;
      if (tags.indexOf(t) >= 0) s += 5;
      if (meta.indexOf(t) >= 0) s += 3;
      if (text.indexOf(t) >= 0) s += 1;
      if (!s) return 0;            // every word must match somewhere
      total += s;
    }
    return total;
  };
  // filter: { k: series key, tag: topic name }
  var search = function (data, ts, filter) {
    filter = filter || {};
    var hits = [];
    data.forEach(function (p) {
      if (filter.k && p.k !== filter.k) return;
      if (filter.tag && p.g.map(norm).indexOf(norm(filter.tag)) < 0) return;
      var s = ts.length ? score(p, ts) : 1;
      if (s) hits.push({ p: p, s: s });
    });
    return hits.sort(function (a, b) { return b.s - a.s; });   // stable: ties stay newest-first
  };
  var load = function (url) {
    if (!cache) cache = fetch(url).then(function (r) { if (!r.ok) throw new Error(r.status); return r.json(); });
    return cache;
  };
  return { norm: norm, esc: esc, terms: terms, mark: mark, snippet: snippet, search: search, load: load };
})();
