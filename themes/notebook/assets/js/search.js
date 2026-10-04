(function () {
  var script = document.currentScript || document.querySelector('script[data-index]');
  var input = document.getElementById('q');
  var results = document.getElementById('search-results');
  var status = document.getElementById('search-status');
  var browse = document.getElementById('search-browse');
  var seriesBox = document.getElementById('search-series');
  var tagBox = document.getElementById('search-tag');
  var params = new URLSearchParams(location.search);
  var state = { q: params.get('q') || '', k: params.get('s') || '', tag: params.get('tag') || '' };
  var data = null;

  var norm = function (s) { return (s || '').toLowerCase().normalize('NFKD').replace(/[̀-ͯ]/g, ''); };
  var esc = function (s) { return s.replace(/[&<>"]/g, function (c) { return { '&': '&amp;', '<': '&lt;', '>': '&gt;', '"': '&quot;' }[c]; }); };
  var mark = function (text, terms) {
    var out = esc(text);
    terms.forEach(function (t) {
      if (t.length < 2) return;
      out = out.replace(new RegExp('(' + t.replace(/[.*+?^${}()|[\]\\]/g, '\\$&') + ')', 'ig'), '<mark>$1</mark>');
    });
    return out;
  };

  function snippet(text, terms) {
    var low = norm(text), at = -1;
    for (var i = 0; i < terms.length && at < 0; i++) at = low.indexOf(terms[i]);
    if (at < 0) return '';
    var start = Math.max(0, at - 70), end = Math.min(text.length, at + 130);
    return (start ? '…' : '') + mark(text.slice(start, end), terms) + (end < text.length ? '…' : '');
  }

  function score(p, terms) {
    var title = norm(p.t), tags = norm(p.g.join(' ')), meta = norm(p.s + ' ' + p.l), text = norm(p.x);
    var total = 0;
    for (var i = 0; i < terms.length; i++) {
      var t = terms[i], s = 0;
      if (title.indexOf(t) >= 0) s += 6;
      if (tags.indexOf(t) >= 0) s += 5;
      if (meta.indexOf(t) >= 0) s += 3;
      if (text.indexOf(t) >= 0) s += 1;
      if (!s) return 0;              // every word must match somewhere
      total += s;
    }
    return total;
  }

  function sync() {
    var u = new URLSearchParams();
    if (state.q) u.set('q', state.q);
    if (state.k) u.set('s', state.k);
    if (state.tag) u.set('tag', state.tag);
    history.replaceState(null, '', location.pathname + (u.toString() ? '?' + u : ''));
  }

  function render() {
    seriesBox.querySelectorAll('button').forEach(function (b) {
      var on = b.getAttribute('data-k') === state.k;
      b.classList.toggle('is-on', on); b.setAttribute('aria-pressed', on ? 'true' : 'false');
    });
    tagBox.hidden = !state.tag;
    if (state.tag) tagBox.querySelector('.tag').textContent = state.tag;
    if (!data) return;
    var terms = norm(state.q).split(/\s+/).filter(Boolean);
    var active = terms.length || state.tag || state.k;
    browse.hidden = !!active;
    if (!active) { results.innerHTML = ''; status.textContent = data.length + ' notes searchable'; return; }
    var hits = [];
    data.forEach(function (p) {
      if (state.k && p.k !== state.k) return;
      if (state.tag && p.g.map(norm).indexOf(norm(state.tag)) < 0) return;
      var s = terms.length ? score(p, terms) : 1;
      if (s) hits.push({ p: p, s: s });
    });
    hits.sort(function (a, b) { return b.s - a.s; });   // stable: ties keep newest-first order
    status.textContent = hits.length + (hits.length === 1 ? ' result' : ' results');
    results.innerHTML = hits.slice(0, 200).map(function (h) {
      var p = h.p;
      return '<li class="row"><a href="' + p.u + '"><time>' + esc(p.d) + '</time><span class="row-main"><span class="row-title">' +
        mark(p.t, terms) + '</span><span class="row-sub">' + esc(p.l || '') + (p.g.length ? ' · ' + mark(p.g.slice(0, 4).join(', '), terms) : '') +
        '</span>' + (terms.length ? '<span class="row-snip">' + snippet(p.x, terms) + '</span>' : '') + '</span><span class="chip' + (p.k === 'book' ? ' chip-book' : '') + '">' + esc(p.s) + '</span></a></li>';
    }).join('') || '<li class="empty">Nothing matches. Try fewer words, or browse <a href="/tags/">all topics</a>.</li>';
  }

  var timer;
  input.value = state.q;
  input.addEventListener('input', function () {
    clearTimeout(timer);
    timer = setTimeout(function () { state.q = input.value.trim(); sync(); render(); }, 120);
  });
  input.form.addEventListener('submit', function (e) { e.preventDefault(); state.q = input.value.trim(); sync(); render(); });
  seriesBox.addEventListener('click', function (e) {
    var b = e.target.closest('button'); if (!b) return;
    state.k = b.getAttribute('data-k'); sync(); render();
  });
  browse.addEventListener('click', function (e) {
    var a = e.target.closest('a[data-tag]'); if (!a) return;
    e.preventDefault(); state.tag = a.getAttribute('data-tag'); sync(); render(); window.scrollTo(0, 0);
  });
  document.getElementById('clear-tag').addEventListener('click', function () { state.tag = ''; sync(); render(); input.focus(); });

  status.textContent = 'Loading…';
  fetch(script.getAttribute('data-index')).then(function (r) { return r.json(); }).then(function (d) { data = d; render(); })
    .catch(function () { status.textContent = 'Search index could not be loaded.'; });
  render();
})();
