// Home page: search box above the Latest card with an instant results pane.
(function () {
  var box = document.querySelector('[data-quick-search]');
  if (!box) return;
  var input = box.querySelector('input');
  var pane = box.querySelector('.qs-pane');
  var list = box.querySelector('.qs-list');
  var foot = box.querySelector('.qs-foot');
  var url = box.getAttribute('data-index');
  var data = null, active = -1, timer, lastQ = null;

  var open = function (on) { pane.hidden = !on; input.setAttribute('aria-expanded', on ? 'true' : 'false'); };
  var options = function () { return Array.prototype.slice.call(list.querySelectorAll('[role="option"]')); };
  var setActive = function (i) {
    var opts = options(); if (!opts.length) return;
    active = (i + opts.length) % opts.length;
    opts.forEach(function (o, j) { o.setAttribute('aria-selected', j === active ? 'true' : 'false'); });
    input.setAttribute('aria-activedescendant', opts[active].id);
    opts[active].scrollIntoView({ block: 'nearest' });
  };
  var full = function () { return '/search/?q=' + encodeURIComponent(input.value.trim()); };

  function render() {
    var q = input.value.trim();
    active = -1; input.removeAttribute('aria-activedescendant');
    if (!q) { open(false); return; }
    if (!data) { list.innerHTML = '<li class="qs-msg">Loading…</li>'; foot.hidden = true; open(true); return; }
    var ts = NB.terms(q), hits = NB.search(data, ts);
    var changed = q !== lastQ; lastQ = q;
    list.innerHTML = hits.slice(0, 30).map(function (h, i) {
      var p = h.p;
      return '<li role="option" id="qs-opt-' + i + '" aria-selected="false"><a href="' + p.u + '" tabindex="-1">' +
        '<span class="qs-top"><span class="qs-title">' + NB.mark(p.t, ts) + '</span><span class="chip' + (p.k === 'book' ? ' chip-book' : '') + '">' + NB.esc(p.s) + '</span></span>' +
        '<span class="qs-snip">' + (NB.snippet(p.x, ts, 50, 110) || NB.esc(p.l || '')) + '</span></a></li>';
    }).join('') || '<li class="qs-msg">No notes match “' + NB.esc(q) + '”.</li>';
    if (changed) list.scrollTop = 0;   // new query: start at the top of the results
    foot.hidden = !hits.length;
    foot.querySelector('a').href = full();
    foot.querySelector('a').textContent = 'See all ' + hits.length + ' result' + (hits.length === 1 ? '' : 's') + ' →';
    open(true);
  }

  var ensure = function () { if (!data) NB.load(url).then(function (d) { data = d; render(); }).catch(function () { list.innerHTML = '<li class="qs-msg">Search is unavailable right now.</li>'; }); };
  input.addEventListener('focus', function () { ensure(); if (input.value.trim()) render(); });
  input.addEventListener('input', function () { ensure(); clearTimeout(timer); timer = setTimeout(render, 100); });
  input.addEventListener('keydown', function (e) {
    if (e.key === 'ArrowDown') { e.preventDefault(); if (pane.hidden) render(); setActive(active + 1); }
    else if (e.key === 'ArrowUp') { e.preventDefault(); setActive(active - 1); }
    else if (e.key === 'Enter') {
      e.preventDefault();
      var opts = options();
      if (active >= 0 && opts[active]) location.href = opts[active].querySelector('a').href;
      else if (input.value.trim()) location.href = full();
    } else if (e.key === 'Escape') { open(false); }
  });
  document.addEventListener('click', function (e) { if (!box.contains(e.target)) open(false); });
})();
