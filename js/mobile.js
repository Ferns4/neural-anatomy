/* mobile.js — the phone layout: one screen at a time behind a bottom tab bar, training settings
   in a bottom sheet, and a peek card over the diagram for a tapped neuron or the backprop walkthrough. */
(function (NA) {
  'use strict';
  const { el, fmt } = NA;
  const App = NA.app;

  const M = { mobile: false, screen: 'network', peek: null };
  // Landscape phones keep the side-by-side layout: one screen at a time would leave the diagram too short.
  const mq = window.matchMedia('(max-width: 760px)');

  M.init = function () {
    M.nav = document.getElementById('mnav');
    M.peekEl = document.getElementById('mpeek');
    M.scrim = document.getElementById('mscrim');
    M.nav.addEventListener('click', (e) => { const b = e.target.closest('[data-screen]'); if (b) M.go(b.dataset.screen); });
    document.getElementById('btnSettings').addEventListener('click', () => M.sheet(true));
    document.getElementById('btnSheetDone').addEventListener('click', () => M.sheet(false));
    M.scrim.addEventListener('click', () => M.sheet(false));
    document.addEventListener('keydown', (e) => { if (e.key === 'Escape' && document.body.dataset.sheet) M.sheet(false); });
    App.on('selection', (o) => { if (M.mobile && o && o.fromGraph) { M.peek = 'neuron'; renderPeek(); } });
    App.on('walk', () => { if (M.mobile) renderPeek(); });
    App.on('weights', () => { if (M.mobile && M.peek === 'neuron' && !App.training) renderPeek(); });
    App.on('probe', () => { if (M.mobile && M.peek === 'neuron') renderPeek(); });
    App.on('model', () => { M.peek = null; syncLabels(); renderPeek(); });
    if (mq.addEventListener) mq.addEventListener('change', apply); else mq.addListener(apply);
    apply();
  };

  function apply() {
    const was = M.mobile;
    M.mobile = mq.matches;
    const b = document.body;
    b.dataset.layout = M.mobile ? 'mobile' : 'desktop';
    if (M.mobile) M.go(M.screen);
    else { delete b.dataset.screen; M.sheet(false); M.peekEl.hidden = true; }
    syncLabels();
    if (was !== M.mobile) App.emit('lens');
  }

  M.go = function (s) {
    M.screen = s;
    document.body.dataset.screen = s;
    NA.$$('[data-screen]', M.nav).forEach((b) => b.setAttribute('aria-current', b.dataset.screen === s ? 'page' : 'false'));
    NA.tooltip.hide();
    if (NA.inspector) NA.inspector.dirty = true;
    if (NA.dataView) NA.dataView.dirty = true;
    if (NA.netView) NA.netView.dirty = true;
    renderPeek();
  };

  M.sheet = function (open) {
    const b = document.body;
    if (open && M.mobile) {
      b.dataset.sheet = 'settings';
      M.scrim.hidden = false;
      document.getElementById('btnSheetDone').focus();
    } else {
      const wasOpen = !!b.dataset.sheet;
      delete b.dataset.sheet;
      M.scrim.hidden = true;
      if (wasOpen) document.getElementById('btnSettings').focus();
    }
  };

  function syncLabels() {
    const lab = M.nav && M.nav.querySelector('.mnav-data');
    if (lab && App.ds) lab.textContent = App.isImage() ? 'Draw' : 'Output';
  }

  // ------------------------------------------------------------------ peek card
  function renderPeek() {
    const P = M.peekEl;
    if (!P) return;
    if (!M.mobile || M.screen !== 'network') { P.hidden = true; return; }
    const walking = App.walk && App.walk.active && NA.walk && NA.walk.summary;
    if (walking) renderWalk(P);
    else if (M.peek === 'neuron' && App.validNeuron(App.selected)) renderNeuron(P);
    else { P.hidden = true; return; }
    P.hidden = false;
  }

  function closeBtn(onclick) {
    return el('button', { class: 'icon-btn mp-close', type: 'button', 'aria-label': 'Close', onclick }, '×');
  }

  function renderNeuron(P) {
    const { l, j } = App.selected, net = App.net, tr = App.probeTrace();
    const kind = l === 0 ? 'Input' : l === net.L ? 'Output' : `Hidden ${l} · ${net.act.label}`;
    const stat = (k, v) => el('span', null, el('b', { text: k }), v);
    const stats = l === 0
      ? [stat('value', fmt(tr.a[0][j], 3))]
      : [stat('z', fmt(tr.z[l][j], 3)), stat(l === net.L && App.ds.task !== 'regression' ? 'p' : 'a', fmt(tr.a[l][j], 3)), stat('bias', fmt(net.b[l - 1][j], 2))];
    const ablatable = l > 0 && l < net.L;
    NA.clear(P).append(
      el('div', { class: 'mp-head' },
        el('div', null, el('div', { class: 'eyebrow', text: kind }), el('div', { class: 'mp-title', text: l === 0 ? App.neuronName(0, j) : App.neuronName(l, j) })),
        closeBtn(() => { M.peek = null; renderPeek(); })),
      el('div', { class: 'mp-stats' }, ...stats),
      el('div', { class: 'btn-row' },
        el('button', { class: 'btn btn-primary', type: 'button', onclick: () => { M.go('insp'); App.emit('tab', 'neuron'); } }, 'Inspect neuron'),
        ablatable ? el('button', { class: 'btn', type: 'button', onclick: () => App.toggleAblate(l, j) }, net.mask[l][j] ? 'Ablate' : 'Restore') : null));
  }

  function renderWalk(P) {
    const s = NA.walk.summary();
    NA.clear(P).append(
      el('div', { class: 'mp-head' },
        el('div', null, el('div', { class: 'eyebrow', text: `Backprop · step ${s.phase + 1} of ${s.total}` }), el('div', { class: 'mp-title', text: s.title }))),
      el('div', { class: 'btn-row' },
        el('button', { class: 'btn', type: 'button', disabled: s.phase === 0, onclick: () => NA.walk.step(-1) }, '← Back'),
        el('button', { class: 'btn btn-primary', type: 'button', disabled: s.phase >= s.total - 1, onclick: () => NA.walk.step(1) }, 'Next →'),
        el('button', { class: 'btn btn-quiet', type: 'button', onclick: () => M.go('insp') }, 'Read the maths')));
  }

  NA.mobile = M;
})(window.NA);
