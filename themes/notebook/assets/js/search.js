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
    var ts = NB.terms(state.q);
    var active = ts.length || state.tag || state.k;
    browse.hidden = !!active;
    if (!active) { results.innerHTML = ''; status.textContent = data.length + ' notes searchable'; return; }
    var hits = NB.search(data, ts, { k: state.k, tag: state.tag });
    status.textContent = hits.length + (hits.length === 1 ? ' result' : ' results');
    results.innerHTML = hits.slice(0, 200).map(function (h) {
      var p = h.p;
      return '<li class="row"><a href="' + p.u + '"><time>' + NB.esc(p.d) + '</time><span class="row-main"><span class="row-title">' +
        NB.mark(p.t, ts) + '</span><span class="row-sub">' + NB.esc(p.l || '') + (p.g.length ? ' · ' + NB.mark(p.g.slice(0, 4).join(', '), ts) : '') +
        '</span>' + (ts.length ? '<span class="row-snip">' + NB.snippet(p.x, ts) + '</span>' : '') +
        '</span><span class="chip' + (p.k === 'book' ? ' chip-book' : '') + '">' + NB.esc(p.s) + '</span></a></li>';
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
  NB.load(script.getAttribute('data-index')).then(function (d) { data = d; render(); })
    .catch(function () { status.textContent = 'Search index could not be loaded.'; });
  render();
})();
