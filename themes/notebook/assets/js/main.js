(function () {
  // Lists with filter pills (home page, book blog). Shows the first N matching rows.
  // On the home page, when the sidebar (topics + about) is taller than the list,
  // more rows are revealed until the list reaches the bottom of the sidebar.
  var filters = document.querySelector('.filters');
  var list = document.getElementById('note-list');
  if (filters && list) {
    var minRows = parseInt(filters.getAttribute('data-count'), 10) || 12;
    var rows = Array.prototype.slice.call(list.querySelectorAll('.row'));
    var main = list.closest('.notes-main');
    var side = document.querySelector('.notes-side');
    var current = 'all';
    var empty = document.createElement('li');
    empty.className = 'empty';
    empty.textContent = 'No notes in this series yet.';
    var sideBySide = function () {
      return main && side && Math.abs(main.getBoundingClientRect().top - side.getBoundingClientRect().top) < 40;
    };
    var render = function () {
      var matches = rows.filter(function (r) { return current === 'all' || r.getAttribute('data-series') === current; });
      rows.forEach(function (r) { r.classList.remove('is-shown'); r.classList.add('is-hidden'); });
      var shown = 0;
      matches.forEach(function (r) { if (shown < minRows) { r.classList.remove('is-hidden'); r.classList.add('is-shown'); shown++; } });
      if (sideBySide()) {
        var target = side.getBoundingClientRect().bottom;
        while (shown < matches.length && main.getBoundingClientRect().bottom < target) {
          matches[shown].classList.remove('is-hidden'); matches[shown].classList.add('is-shown'); shown++;
        }
      }
      if (shown === 0) list.appendChild(empty); else if (empty.parentNode) empty.remove();
    };
    filters.addEventListener('click', function (e) {
      var btn = e.target.closest('button[data-filter]');
      if (!btn) return;
      current = btn.getAttribute('data-filter');
      filters.querySelectorAll('button').forEach(function (b) {
        var on = b === btn;
        b.classList.toggle('is-on', on);
        b.setAttribute('aria-pressed', on ? 'true' : 'false');
      });
      render();
    });
    if (main && side) {
      render();
      var t;
      window.addEventListener('resize', function () { clearTimeout(t); t = setTimeout(render, 150); });
      if (document.fonts && document.fonts.ready) document.fonts.ready.then(render);
    }
  }
  // Note pages: reading progress bar.
  var bar = document.getElementById('progress-bar');
  if (bar) {
    var update = function () {
      var h = document.documentElement;
      var max = h.scrollHeight - h.clientHeight;
      bar.style.width = (max > 0 ? Math.min(100, (h.scrollTop / max) * 100) : 0) + '%';
    };
    document.addEventListener('scroll', update, { passive: true });
    update();
  }
})();
