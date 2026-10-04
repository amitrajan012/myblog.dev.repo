(function () {
  // Home: filter the recent-notes list by series, showing the first N matches.
  var filters = document.querySelector('.filters');
  var list = document.getElementById('note-list');
  if (filters && list) {
    var n = parseInt(filters.getAttribute('data-count'), 10) || 12;
    var rows = Array.prototype.slice.call(list.querySelectorAll('.row'));
    var empty = document.createElement('li');
    empty.className = 'empty';
    empty.textContent = 'No notes in this series yet.';
    filters.addEventListener('click', function (e) {
      var btn = e.target.closest('button[data-filter]');
      if (!btn) return;
      var f = btn.getAttribute('data-filter');
      filters.querySelectorAll('button').forEach(function (b) {
        var on = b === btn;
        b.classList.toggle('is-on', on);
        b.setAttribute('aria-pressed', on ? 'true' : 'false');
      });
      var shown = 0;
      rows.forEach(function (r) {
        var match = f === 'all' || r.getAttribute('data-series') === f;
        var show = match && shown < n;
        if (show) shown++;
        r.classList.toggle('is-shown', show);
        r.classList.toggle('is-hidden', !show);
      });
      if (shown === 0) list.appendChild(empty); else if (empty.parentNode) empty.remove();
    });
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
