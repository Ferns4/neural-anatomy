/* chrome.js — header, model zoo, training controls, lens bar, history timeline, tooltip,
   keyboard shortcuts and start-up. */
(function (NA) {
  'use strict';
  const { theme, clamp, fmt, fmtPct, fmtInt, el, $, $$ } = NA;
  const App = NA.app;
  const E = NA.engine;

  // ------------------------------------------------------------------ tooltip
  NA.tooltip = {
    init() { this.el = document.getElementById('tooltip'); },
    show(children, x, y) {
      const t = this.el;
      NA.clear(t).append(...[].concat(children).filter(Boolean));
      t.hidden = false;
      const r = t.getBoundingClientRect(), vw = window.innerWidth, vh = window.innerHeight;
      let left = x + 16, top = y + 14;
      if (left + r.width > vw - 8) left = x - r.width - 16;
      if (top + r.height > vh - 8) top = y - r.height - 14;
      t.style.left = Math.max(8, left) + 'px'; t.style.top = Math.max(8, top) + 'px';
    },
    hide() { if (this.el) this.el.hidden = true; },
  };

  const CH = { lastRead: 0 };

  // ------------------------------------------------------------------ controls
  const OPTIONS = {
    lr: [0.0003, 0.001, 0.003, 0.005, 0.01, 0.015, 0.02, 0.03, 0.05, 0.1, 0.3, 1, 3],
    batch: [1, 4, 8, 10, 16, 32, 64, 128],
    regRate: [0, 0.0001, 0.001, 0.003, 0.01, 0.03, 0.1],
  };
  function fillSelect(sel, values, label) {
    NA.clear(sel);
    for (const v of values) sel.appendChild(el('option', { value: v }, label ? label(v) : String(v)));
  }
  function setSelect(sel, v, values, label) {
    if (values && !values.includes(v)) { values.push(v); values.sort((a, b) => a - b); fillSelect(sel, values, label); }
    sel.value = String(v);
  }

  CH.init = function () {
    CH.play = $('#btnPlay'); CH.playLabel = $('#btnPlay .pl-label'); CH.ff = $('#ffProgress');
    $('#btnPlay').addEventListener('click', () => App.toggle());
    $('#btnStep').addEventListener('click', () => App.stepOnce());
    $('#btnReset').addEventListener('click', () => App.reset());
    $('#btnFF').addEventListener('click', () => App.fastForward(500));
    fillSelect($('#hpLr'), OPTIONS.lr);
    fillSelect($('#hpBatch'), OPTIONS.batch);
    fillSelect($('#hpRegRate'), OPTIONS.regRate);
    const on = (id, fn) => $(id).addEventListener('change', (e) => fn(e.target.value));
    on('#hpLr', (v) => App.applyPatch({ lr: +v }));
    on('#hpOpt', (v) => App.applyPatch({ optimizer: v }));
    on('#hpBatch', (v) => App.applyPatch({ batch: +v }));
    on('#hpAct', (v) => App.applyPatch({ act: v }));
    on('#hpReg', (v) => App.applyPatch({ reg: v, regRate: v !== 'none' && !App.hyper.regRate ? 0.001 : v === 'none' ? 0 : App.hyper.regRate }));
    on('#hpRegRate', (v) => App.applyPatch({ regRate: +v, reg: +v > 0 && App.hyper.reg === 'none' ? 'l2' : +v === 0 ? 'none' : App.hyper.reg }));
    on('#hpInit', (v) => App.applyPatch({ init: v }));
    on('#hpSpeed', (v) => { App.speed = v; NA.store.set('speed2', v); });
    on('#hpSeed', (v) => App.applyPatch({ seed: Math.max(1, Math.round(+v) || 1) }));
    $('#btnSeed').addEventListener('click', () => App.applyPatch({ seed: (App.hyper.seed % 999) + 1 }));
    App.speed = NA.store.get('speed2', 'normal');
    if (!['slow', 'normal', 'fast', 'max'].includes(App.speed)) App.speed = 'normal';
    $('#hpSpeed').value = App.speed;

    // lens bar
    $$('.seg[data-lens]').forEach((seg) => seg.addEventListener('click', (e) => { const b = e.target.closest('button'); if (b) App.setLens(seg.dataset.lens, b.dataset.v); }));
    $$('[data-toggle]').forEach((b) => b.addEventListener('click', () => App.setLens(b.dataset.toggle, !App.lenses[b.dataset.toggle])));

    // model zoo
    $('#modelPicker').addEventListener('click', () => CH.openZoo());
    $('#zoo').addEventListener('click', (e) => { if (e.target.id === 'zoo' || e.target.closest('[data-close]')) CH.closeZoo(); });
    $('#helpBtn').addEventListener('click', () => CH.toggleHelp());
    $('#help').addEventListener('click', (e) => { if (e.target.id === 'help' || e.target.closest('[data-close]')) CH.toggleHelp(false); });
    buildZoo();

    // timeline
    CH.tl = $('#tlCanvas');
    const scrub = (e) => {
      const r = CH.tl.getBoundingClientRect(), H = App.history, snaps = H.snaps, f = CH.tlFrame;
      if (!snaps.length || !f) return;
      const step = f.xmin + clamp((e.clientX - r.left - f.padL) / f.w, 0, 1) * (f.xmax - f.xmin);
      let bi = 0;
      for (let i = 0; i < snaps.length; i++) if (Math.abs(snaps[i].step - step) < Math.abs(snaps[bi].step - step)) bi = i;
      if (bi === snaps.length - 1 && App.viewing === null) return;
      App.viewSnapshot(bi);
    };
    let dragging = false;
    CH.tl.addEventListener('pointerdown', (e) => { dragging = true; CH.tl.setPointerCapture(e.pointerId); scrub(e); });
    CH.tl.addEventListener('pointermove', (e) => { if (dragging) scrub(e); });
    CH.tl.addEventListener('pointerup', () => { dragging = false; });
    CH.tl.addEventListener('keydown', (e) => {
      const H = App.history, cur = App.viewing === null ? H.snaps.length - 1 : App.viewing;
      if (e.key === 'ArrowLeft') { e.preventDefault(); App.viewSnapshot(cur - 1); }
      if (e.key === 'ArrowRight') { e.preventDefault(); if (cur + 1 >= H.snaps.length - 1 && App.viewing !== null) App.backToLive(); else App.viewSnapshot(cur + 1); }
      if (e.key === 'End') App.backToLive();
    });
    $('#btnLive').addEventListener('click', () => App.backToLive());
    $('#btnResume').addEventListener('click', () => App.play());
    new ResizeObserver(() => { CH.tlDirty = true; }).observe(CH.tl.parentElement);

    // events
    App.on('training', syncTransport);
    App.on('hyper', syncControls);
    App.on('model', () => { syncControls(); syncHeader(); });
    App.on('preset', syncHeader);
    App.on('lens', syncLenses);
    App.on('history', () => { CH.tlDirty = true; });
    App.on('weights', () => { CH.roDirty = true; CH.tlDirty = true; });
    App.on('probe', () => { CH.roDirty = true; });
    document.addEventListener('keydown', onKey);
    App.register(CH);
  };

  function syncControls() {
    const h = App.hyper;
    setSelect($('#hpLr'), h.lr, OPTIONS.lr);
    $('#hpOpt').value = h.optimizer;
    setSelect($('#hpBatch'), h.batch, OPTIONS.batch);
    $('#hpAct').value = App.arch.act;
    $('#hpReg').value = h.reg;
    setSelect($('#hpRegRate'), h.regRate, OPTIONS.regRate);
    $('#hpInit').value = h.init;
    $('#hpSeed').value = h.seed;
    $('#hpAct').disabled = App.arch.hidden.length === 0;
    $('#hpAct').title = App.arch.hidden.length === 0 ? 'No hidden layers: the only neuron is the output' : '';
    CH.roDirty = true;
  }
  function syncHeader() {
    const p = App.preset, net = App.net;
    $('#mpName').textContent = p.name;
    $('#mpArch').textContent = net.sizes.join(' → ');
    try { history.replaceState(null, '', '#' + p.id); } catch (e) { /* not allowed in some frames */ }
    document.title = 'Neural Anatomy';
  }
  function syncLenses() {
    $$('.seg[data-lens]').forEach((seg) => $$('button', seg).forEach((b) => b.setAttribute('aria-pressed', String(App.lenses[seg.dataset.lens] === b.dataset.v))));
    $$('[data-toggle]').forEach((b) => b.setAttribute('aria-pressed', String(!!App.lenses[b.dataset.toggle])));
  }
  function syncTransport() {
    const t = App.training, ff = App.ffTarget !== null;
    CH.play.setAttribute('aria-pressed', String(t));
    CH.play.classList.toggle('is-running', t);
    CH.playLabel.textContent = t ? (ff ? 'Training…' : 'Pause') : 'Train';
    $('#icoPlay').hidden = t; $('#icoPause').hidden = !t;
    CH.ff.hidden = !ff;
    CH.ffStart = ff ? App.step : null;
    CH.roDirty = true;
  }

  // ------------------------------------------------------------------ per-frame updates
  CH.frame = function (now) {
    if (App.ffTarget !== null) {
      const total = App.ffTarget - (CH.ffStart ?? 0) || 1;
      $('#ffBar').style.width = clamp((App.step - (CH.ffStart ?? 0)) / total, 0, 1) * 100 + '%';
    }
    if (CH.roDirty && now - CH.lastRead > 90) { CH.lastRead = now; CH.roDirty = false; readouts(); }
    if (CH.tlDirty && (!App.training || now - (CH.tlTime || 0) > 250)) { CH.tlTime = now; CH.tlDirty = false; drawTimeline(); }
  };

  function readouts() {
    const m = App.metrics(), cls = App.ds.task !== 'regression', step = App.displayStep();
    $('#roStep').textContent = fmtInt(step);
    $('#roEpoch').textContent = fmt(App.epoch(step), 1);
    $('#roTrain').textContent = fmt(m.trainLoss, 3);
    $('#roTest').textContent = fmt(m.testLoss, 3);
    $('#roAccLabel').textContent = cls ? 'Test accuracy' : 'Test R²';
    $('#roAcc').textContent = cls ? fmtPct(m.testAcc, 1) : fmt(m.testAcc, 3);
    const state = $('#roState');
    if (App.diverged) { state.textContent = 'Diverged'; state.className = 'state-pill pill-critical'; }
    else if (App.viewing !== null) { state.textContent = 'History'; state.className = 'state-pill pill-info'; }
    else if (App.training) { state.textContent = App.ffTarget !== null ? 'Pre-training' : 'Training'; state.className = 'state-pill pill-good'; }
    else { state.textContent = step === 0 ? 'Untrained' : 'Paused'; state.className = 'state-pill pill-quiet'; }
  }

  function drawTimeline() {
    const c = CH.tl, wrap = c.parentElement, W = Math.max(120, wrap.clientWidth), H = 46;
    const ctx = NA.fitCanvas(c, W, H), pts = App.history.points, snaps = App.history.snaps;
    ctx.clearRect(0, 0, W, H);
    const padL = 4, padR = 4, w = W - padL - padR, h = H - 10;
    const xmin = 0, xmax = Math.max(1, pts.length ? pts[pts.length - 1].step : 1);
    CH.tlFrame = { padL, w, xmin, xmax };
    ctx.fillStyle = theme.c('panel-2'); NA.rr(ctx, 0, 0, W, H, 6); ctx.fill();
    let lo = Infinity, hi = -Infinity;
    for (const p of pts) for (const v of [p.trainLoss, p.testLoss]) if (v > 0 && isFinite(v)) { lo = Math.min(lo, v); hi = Math.max(hi, v); }
    if (isFinite(lo)) {
      const llo = Math.log10(lo), lhi = Math.log10(hi) + 1e-9;
      const X = (s) => padL + ((s - xmin) / (xmax - xmin)) * w, Y = (v) => 5 + h - ((Math.log10(Math.max(v, lo)) - llo) / (lhi - llo || 1)) * h;
      for (const [key, color] of [['trainLoss', theme.c('ink-2')], ['testLoss', theme.c('pos')]]) {
        ctx.strokeStyle = color; ctx.lineWidth = 1.5; ctx.beginPath();
        pts.forEach((p, i) => { if (!isFinite(p[key]) || p[key] <= 0) return; i ? ctx.lineTo(X(p.step), Y(p[key])) : ctx.moveTo(X(p.step), Y(p[key])); });
        ctx.stroke();
      }
      // snapshot ticks
      ctx.fillStyle = theme.c('muted', 0.6);
      for (const s of snaps) ctx.fillRect(Math.round(X(s.step)), H - 3, 1, 3);
      const cur = App.displayStep(), x = Math.round(X(cur)) + 0.5;
      ctx.strokeStyle = theme.c('ink'); ctx.lineWidth = 1.5;
      ctx.beginPath(); ctx.moveTo(x, 2); ctx.lineTo(x, H - 2); ctx.stroke();
      ctx.fillStyle = theme.c('ink'); NA.rr(ctx, x - 4, 1, 8, 8, 2); ctx.fill();
    }
    const viewing = App.viewing !== null;
    $('#tlStatus').textContent = viewing ? `Viewing step ${fmtInt(App.displayStep())} of ${fmtInt(App.step)}` : `${fmtInt(App.step)} steps · ${snaps.length} snapshots`;
    $('#btnLive').hidden = !viewing;
    $('#btnResume').hidden = !viewing;
    c.setAttribute('aria-valuenow', String(App.displayStep()));
    c.setAttribute('aria-valuemax', String(App.step));
    c.setAttribute('aria-valuetext', `Step ${App.displayStep()}`);
  }

  // ------------------------------------------------------------------ model zoo
  function buildZoo() {
    const body = $('#zooBody');
    NA.clear(body);
    for (const g of NA.presets.GROUPS) {
      const items = NA.presets.PRESETS.filter((p) => p.group === g);
      body.appendChild(el('h3', { class: 'zoo-group', text: g }));
      const grid = el('div', { class: 'zoo-grid' });
      for (const p of items) {
        const sizes = [p.data.dataset === 'digits' ? 196 : p.data.dataset in NA.data.CURVES ? 1 : (p.data.features || ['x1', 'x2']).length, ...p.arch.hidden, NA.data.DATASETS[p.data.dataset].task === 'multiclass' ? (p.data.dataset === 'digits' ? 10 : 3) : 1];
        const glyph = el('canvas', { class: 'zoo-glyph', 'aria-hidden': 'true' });
        const card = el('button', { class: 'zoo-card', type: 'button', dataset: { id: p.id }, onclick: () => { CH.closeZoo(); App.loadPreset(p.id); } },
          glyph,
          el('span', { class: 'zc-text' },
            el('span', { class: 'zc-name', text: p.name }),
            el('span', { class: 'zc-tag', text: p.tagline }),
            el('span', { class: 'zc-meta', text: `${sizes.join('–')} · ${p.arch.hidden.length ? E.ACTIVATIONS[p.arch.act].label : 'no hidden layer'} · ${NA.data.DATASETS[p.data.dataset].label}` })));
        grid.appendChild(card);
        card._sizes = sizes;
      }
      body.appendChild(grid);
    }
  }
  function drawGlyphs() {
    $$('.zoo-card').forEach((card) => {
      const c = $('.zoo-glyph', card), sizes = card._sizes, W = 64, H = 44, ctx = NA.fitCanvas(c, W, H);
      ctx.clearRect(0, 0, W, H);
      const xs = sizes.map((_, i) => 6 + (i * (W - 12)) / Math.max(1, sizes.length - 1));
      const ys = sizes.map((n) => { const m = Math.min(n, 6), out = []; for (let k = 0; k < m; k++) out.push(H / 2 + (k - (m - 1) / 2) * 6.5); return out; });
      ctx.strokeStyle = theme.c('line-2'); ctx.lineWidth = 0.6;
      for (let i = 0; i < sizes.length - 1; i++) for (const a of ys[i]) for (const b of ys[i + 1]) { ctx.beginPath(); ctx.moveTo(xs[i], a); ctx.lineTo(xs[i + 1], b); ctx.stroke(); }
      sizes.forEach((n, i) => ys[i].forEach((y) => { ctx.beginPath(); ctx.arc(xs[i], y, 2.2, 0, Math.PI * 2); ctx.fillStyle = i === sizes.length - 1 ? theme.c('ink') : theme.c('ink-2'); ctx.fill(); }));
    });
  }
  CH.openZoo = function () {
    const z = $('#zoo');
    z.hidden = false;
    drawGlyphs();
    $$('.zoo-card').forEach((c) => c.setAttribute('aria-current', String(App.preset && c.dataset.id === App.preset.id)));
    const cur = $('.zoo-card[aria-current="true"]') || $('.zoo-card');
    if (cur) cur.focus();
  };
  CH.closeZoo = function () { $('#zoo').hidden = true; $('#modelPicker').focus(); };
  CH.toggleHelp = function (force) {
    const h = $('#help'), show = force === undefined ? h.hidden : force;
    h.hidden = !show;
    if (show) $('[data-close]', h).focus();
  };

  // ------------------------------------------------------------------ keyboard
  function onKey(e) {
    if (e.key === 'Escape') { if (!$('#zoo').hidden) CH.closeZoo(); if (!$('#help').hidden) CH.toggleHelp(false); return; }
    const t = e.target;
    if (t && (t.tagName === 'INPUT' || t.tagName === 'SELECT' || t.tagName === 'TEXTAREA' || t.isContentEditable)) return;
    if (e.metaKey || e.ctrlKey || e.altKey) return;
    if (!$('#zoo').hidden) return;
    const k = e.key;
    if (k === ' ' && !(t && t.tagName === 'BUTTON')) { e.preventDefault(); App.toggle(); }
    else if (k === 's' || k === 'S') App.stepOnce();
    else if (k === 'r' || k === 'R') App.reset();
    else if (k === '1') App.setLens('edges', 'weight');
    else if (k === '2') App.setLens('edges', 'signal');
    else if (k === '3') App.setLens('edges', 'gradient');
    else if (k === '4') App.setLens('nodes', 'activation');
    else if (k === '5') App.setLens('nodes', 'preact');
    else if (k === '6') App.setLens('nodes', 'gradient');
    else if (k === 'm' || k === 'M') App.setLens('maps', !App.lenses.maps);
    else if (k === 'v' || k === 'V') App.setLens('values', !App.lenses.values);
    else if (k === 'f' || k === 'F') App.setLens('flow', !App.lenses.flow);
    else if (k === 'b' || k === 'B') App.setLens('bias', !App.lenses.bias);
    else if (k === 'h' || k === 'H') App.setLens('health', !App.lenses.health);
    else if (k === '?') CH.toggleHelp();
  }

  // ------------------------------------------------------------------ theme toggle (standalone site only;
  // inside the artifact viewer the host owns the theme)
  const SUN = '<svg width="15" height="15" viewBox="0 0 16 16" aria-hidden="true"><circle cx="8" cy="8" r="3" fill="none" stroke="currentColor" stroke-width="1.5"/><path d="M8 1.5v1.6M8 12.9v1.6M1.5 8h1.6M12.9 8h1.6M3.4 3.4l1.1 1.1M11.5 11.5l1.1 1.1M3.4 12.6l1.1-1.1M11.5 4.5l1.1-1.1" stroke="currentColor" stroke-width="1.5" stroke-linecap="round"/></svg>';
  const MOON = '<svg width="15" height="15" viewBox="0 0 16 16" aria-hidden="true"><path d="M13.2 10.1A5.6 5.6 0 0 1 5.9 2.8a5.6 5.6 0 1 0 7.3 7.3z" fill="none" stroke="currentColor" stroke-width="1.5" stroke-linejoin="round"/></svg>';
  function initThemeToggle() {
    const btn = $('#themeBtn');
    let embedded = true;
    try { embedded = window.top !== window; } catch (e) { embedded = true; }
    if (embedded || !btn) return;
    const saved = NA.store.get('theme', null);
    if (saved === 'light' || saved === 'dark') document.documentElement.setAttribute('data-theme', saved);
    btn.hidden = false;
    const sync = () => {
      btn.innerHTML = theme.dark ? SUN : MOON;
      btn.setAttribute('aria-label', theme.dark ? 'Switch to light theme' : 'Switch to dark theme');
      btn.title = btn.getAttribute('aria-label');
    };
    btn.addEventListener('click', () => {
      const next = theme.dark ? 'light' : 'dark';
      document.documentElement.setAttribute('data-theme', next);
      NA.store.set('theme', next);
    });
    theme.onChange(sync);
    sync();
  }

  // ------------------------------------------------------------------ start-up
  function boot() {
    initThemeToggle();
    theme.read();
    NA.tooltip.init();
    NA.netView.init();
    NA.dataView.init();
    NA.inspector.init();
    CH.init();
    const refreshTheme = () => { theme.read(); App.emit('weights'); CH.tlDirty = true; CH.roDirty = true; };
    if (window.matchMedia) window.matchMedia('(prefers-color-scheme: dark)').addEventListener('change', refreshTheme);
    new MutationObserver(refreshTheme).observe(document.documentElement, { attributes: true, attributeFilter: ['data-theme', 'class'] });
    if (document.fonts && document.fonts.ready) document.fonts.ready.then(() => { App.emit('lens'); App.emit('weights'); });
    const hash = (location.hash || '').slice(1);
    let start = NA.presets.byId[hash] ? hash : NA.store.get('preset', 'spiral');
    if (!NA.presets.byId[start]) start = 'spiral';
    App.loadPreset(start, { sync: true });
    requestAnimationFrame((t) => App.frame(t));
  }
  NA.chrome = CH;
  if (document.readyState === 'loading') document.addEventListener('DOMContentLoaded', boot); else boot();
})(window.NA);
