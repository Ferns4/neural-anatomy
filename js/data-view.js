/* data-view.js — the specimen panel: decision map, curve fit or drawing pad, plus the hidden-space view. */
(function (NA) {
  'use strict';
  const { theme, clamp, fmt, fmtPct, el, rr } = NA;
  const App = NA.app;
  const D = NA.data;

  const DV = { dirty: true, view: 'output', lastDraw: 0 };

  DV.init = function () {
    DV.root = document.getElementById('dataPanel');
    DV.body = document.getElementById('dataBody');
    DV.tabs = document.getElementById('dataTabs');
    DV.controls = document.getElementById('dataControls');
    DV.tabs.addEventListener('click', (e) => { const b = e.target.closest('[data-view]'); if (b) DV.setView(b.dataset.view); });
    const mark = () => { DV.dirty = true; };
    ['weights', 'probe', 'hover', 'selection', 'overlay', 'lens'].forEach((t) => App.on(t, mark));
    App.on('model', () => { DV.build(); });
    App.on('preset', (p) => { DV.setView(p.view || 'output'); });
    App.on('view', (v) => DV.setView(v));
    App.on('sampleDigit', (d) => { const k = Array.from(App.ds.valIdx).find((i) => App.ds.y[i] === d); if (k !== undefined) App.setProbeSample(k); });
    App.on('test', () => { DV.dirty = true; DV.galleryKey = ''; });
    theme.onChange(mark);
    let lastW = 0;
    new ResizeObserver(() => {
      const w = DV.body.clientWidth;
      if (Math.abs(w - lastW) < 2) return;
      lastW = w;
      requestAnimationFrame(() => DV.layout());
    }).observe(DV.body);
    App.register(DV);
  };

  DV.setView = function (v) {
    if (v === 'hidden' && App.isLine()) v = 'output';
    DV.view = v;
    NA.$$('[data-view]', DV.tabs).forEach((b) => b.setAttribute('aria-selected', String(b.dataset.view === v)));
    DV.outEl.hidden = v !== 'output';
    DV.hsEl.hidden = v !== 'hidden';
    DV.layout();
    DV.dirty = true;
    if (v === 'hidden') NA.hiddenSpace.enter();
  };

  // ------------------------------------------------------------------ construction per model
  DV.build = function () {
    const ds = App.ds;
    NA.clear(DV.body);
    const outTab = DV.tabs.querySelector('[data-view="output"]'), hsTab = DV.tabs.querySelector('[data-view="hidden"]');
    outTab.textContent = ds.kind === 'image' ? 'Draw' : 'Output';
    hsTab.hidden = ds.kind === 'line';
    DV.tabs.hidden = ds.kind === 'line';
    let title = DV.tabs.parentElement.querySelector('.panel-title');
    if (!title) { title = el('span', { class: 'panel-title', text: 'Network output vs. x' }); DV.tabs.parentElement.appendChild(title); }
    title.hidden = ds.kind !== 'line';
    DV.outEl = el('div', { class: 'dv-out' });
    DV.hsEl = el('div', { class: 'dv-hs', hidden: true });
    DV.body.append(DV.outEl, DV.hsEl);
    if (ds.kind === 'plane') buildPlane();
    else if (ds.kind === 'line') buildLine();
    else buildImage();
    NA.hiddenSpace.build(DV.hsEl);
    buildDataControls();
    DV.setView(DV.view);
  };

  function chip(label, key, title) {
    const b = el('button', { class: 'chip', type: 'button', 'aria-pressed': String(!!App.overlays[key]), title, onclick: () => { App.setOverlay(key, !App.overlays[key]); b.setAttribute('aria-pressed', String(!!App.overlays[key])); } }, label);
    return b;
  }

  DV.layout = function () {
    if (!DV.canvas || !DV.outEl || DV.outEl.hidden) { if (NA.hiddenSpace && DV.view === 'hidden') NA.hiddenSpace.layout(); return; }
    const w = Math.max(160, DV.outEl.clientWidth || DV.body.clientWidth);
    if (App.isPlane()) { DV.size = Math.min(w, 520); DV.ctx = NA.fitCanvas(DV.canvas, DV.size, DV.size); }
    else if (App.isLine()) { DV.cw = w; DV.ch = Math.round(clamp(w * 0.82, 220, 360)); DV.ctx = NA.fitCanvas(DV.canvas, DV.cw, DV.ch); }
    else if (App.isImage()) layoutImage(w);
    DV.dirty = true;
  };

  DV.frame = function (now) {
    if (!DV.root.offsetParent) return; // skip while its screen is hidden
    if (DV.view === 'hidden') { NA.hiddenSpace.frame(now); return; }
    if (!DV.dirty || !DV.ctx) return;
    if (App.training && now - DV.lastDraw < 50) return;
    DV.lastDraw = now; DV.dirty = false;
    if (App.isPlane()) drawPlane();
    else if (App.isLine()) drawLine();
    else drawImage();
    updateReadout();
  };

  // ================================================================== 2-D plane
  function buildPlane() {
    const tools = el('div', { class: 'chip-row', role: 'group', 'aria-label': 'Map overlays' },
      chip('Train', 'train', 'Show training points (filled dots)'),
      chip('Validation', 'val', 'Show validation points (rings): tracked live while training'),
      chip('Test', 'test', 'Show the held-out test points (squares). Run the test from the Training tab.'),
      chip('Errors', 'errors', 'Ring the points the network gets wrong'),
      chip('Neuron lines', 'neuronLines', 'Draw where each first-layer neuron’s pre-activation crosses zero'));
    DV.canvas = el('canvas', { class: 'plane-canvas', 'aria-label': 'Decision map of the network over the input plane' });
    DV.off = document.createElement('canvas');
    DV.readout = el('div', { class: 'readout-line', 'aria-live': 'polite' });
    DV.legendEl = el('div', { class: 'mini-legend' });
    const host = el('div', { class: 'zoom-host' }, DV.canvas);
    DV.outEl.append(tools, el('div', { class: 'canvas-wrap' }, host), DV.readout, DV.legendEl);
    // one finger (or the mouse) moves the probe; two fingers, double-tap or Ctrl + scroll zoom
    DV.zoom = new NA.ui.Zoom(DV.canvas, { host, onChange: () => { DV.dirty = true; NA.tooltip.hide(); } });
    let dragging = false;
    const toRaw = (ev) => {
      const r = DV.canvas.getBoundingClientRect(), [wx, wy] = DV.zoom.toWorld(ev.clientX - r.left, ev.clientY - r.top);
      return [clamp((wx / r.width) * 2 - 1, -1, 1), clamp(1 - (wy / r.height) * 2, -1, 1)];
    };
    DV.canvas.addEventListener('pointerdown', (ev) => {
      if (DV.zoom.ptrs.size > 1) { dragging = false; return; }
      dragging = true; DV.canvas.setPointerCapture(ev.pointerId);
      const raw = toRaw(ev), k = nearestPoint(ev);
      if (k !== null) App.setProbeSample(k); else App.setProbe(raw, null);
    });
    DV.canvas.addEventListener('pointermove', (ev) => {
      if (DV.zoom.busy) { dragging = false; return; }
      const raw = toRaw(ev);
      if (dragging) { App.setProbe(raw, null); return; }
      const k = nearestPoint(ev);
      DV.hoverPoint = k;
      App.previewProbe(k !== null ? Array.from(App.ds.raw.subarray(2 * k, 2 * k + 2)) : raw);
      if (k !== null) NA.tooltip.show(sampleTip(k), ev.clientX, ev.clientY); else NA.tooltip.hide();
    });
    const end = () => { dragging = false; };
    DV.canvas.addEventListener('pointerup', end);
    DV.canvas.addEventListener('pointercancel', end);
    DV.canvas.addEventListener('pointerleave', () => { DV.hoverPoint = null; App.previewProbe(null); NA.tooltip.hide(); });
    renderPlaneLegend();
  }

  const SPLIT_KEY = ['train', 'val', 'test'], SPLIT_NAME = ['Training', 'Validation', 'Test'];
  function visibleSample(k) { return !!App.overlays[SPLIT_KEY[App.ds.split[k]]]; }
  // Training points are filled dots, validation points rings, held-out test points squares.
  function drawSample(ctx, x, y, r, split, col, sc = 1) {
    ctx.beginPath();
    if (split === 2) { const s = r * 0.95; ctx.rect(x - s, y - s, 2 * s, 2 * s); } else ctx.arc(x, y, r, 0, Math.PI * 2);
    if (split === 0) { ctx.fillStyle = col; ctx.fill(); }
    else { ctx.lineWidth = 1.7 * sc; ctx.strokeStyle = col; ctx.stroke(); }
  }
  function nearestPoint(ev) {
    const ds = App.ds, r = DV.canvas.getBoundingClientRect(), [mx, my] = DV.zoom ? DV.zoom.toWorld(ev.clientX - r.left, ev.clientY - r.top) : [ev.clientX - r.left, ev.clientY - r.top];
    let best = null, bd = 8 / (DV.zoom ? DV.zoom.k : 1);
    for (let k = 0; k < ds.n; k++) {
      if (!visibleSample(k)) continue;
      const x = ((ds.raw[2 * k] + 1) / 2) * r.width, y = ((1 - ds.raw[2 * k + 1]) / 2) * r.height;
      const d = Math.hypot(x - mx, y - my);
      if (d < bd) { bd = d; best = k; }
    }
    return best;
  }
  function sampleTip(k) {
    const ds = App.ds, pred = App.predictionsAll()[k];
    const row = (a, b) => el('div', { class: 'tt-row' }, el('span', { class: 'tt-k', text: a }), el('span', { class: 'tt-v', text: b }));
    const kids = [el('div', { class: 'tt-title', text: `${SPLIT_NAME[ds.split[k]]} sample #${k}` })];
    if (ds.kind === 'plane') kids.push(row('position', `(${fmt(ds.raw[2 * k], 2)}, ${fmt(ds.raw[2 * k + 1], 2)})`));
    if (ds.task === 'regression') { kids.push(row('x', fmt(ds.raw[k], 3)), row('target y', fmt(ds.y[k], 3)), row('prediction ŷ', fmt(pred, 3))); }
    else {
      kids.push(row('true class', App.className(ds.y[k])));
      kids.push(row('predicted', App.className(pred)));
      if (pred !== ds.y[k]) kids.push(el('div', { class: 'tt-note', text: 'Misclassified' }));
    }
    kids.push(el('div', { class: 'tt-hint', text: 'Click to make this the probe' }));
    return kids;
  }

  function renderPlaneLegend() {
    const ds = App.ds, items = [];
    for (let k = 0; k < ds.classes; k++) items.push(el('span', { class: 'lg-item' }, el('i', { class: 'dot', style: { background: `var(--${['neg', 'pos', 'c3'][k]})` } }), App.className(k)));
    items.push(el('span', { class: 'lg-item' }, el('i', { class: 'dot dot-filled' }), 'train'), el('span', { class: 'lg-item' }, el('i', { class: 'dot dot-ring' }), 'validation'), el('span', { class: 'lg-item' }, el('i', { class: 'dot dot-square' }), 'test (held out)'));
    NA.clear(DV.legendEl).append(...items);
  }

  function drawPlane() {
    const ctx = DV.ctx, S = DV.size, ds = App.ds, net = App.net, L = net.L;
    const g = App.grid(), G = g.G, n = g.n;
    const hov = App.hover && App.hover.l > 0 && App.hover.l < L ? App.hover : null;
    // background map
    if (DV.off.width !== G) { DV.off.width = G; DV.off.height = G; }
    const octx = DV.off.getContext('2d'), im = octx.createImageData(G, G), d = im.data;
    if (hov) {
      const A = g.F.a[hov.l], off = hov.j * n, lut = theme.divSoft;
      let m = 1e-9; for (let p = 0; p < n; p++) m = Math.max(m, Math.abs(A[off + p]));
      for (let p = 0; p < n; p++) { const k = Math.round((clamp(A[off + p] / m, -1, 1) + 1) * 128) * 3; d[4 * p] = lut[k]; d[4 * p + 1] = lut[k + 1]; d[4 * p + 2] = lut[k + 2]; d[4 * p + 3] = 255; }
    } else if (ds.task === 'binary') {
      const A = g.F.a[L], lut = theme.divSoft;
      for (let p = 0; p < n; p++) { const k = Math.round(clamp(A[p], 0, 1) * 256) * 3; d[4 * p] = lut[k]; d[4 * p + 1] = lut[k + 1]; d[4 * p + 2] = lut[k + 2]; d[4 * p + 3] = 255; }
    } else {
      const A = g.F.a[L], K = net.sizes[L], panel = theme.tokens.panel;
      const soft = theme.classes.slice(0, K).map((c) => NA.mixLab(c, panel, 0.42));
      for (let p = 0; p < n; p++) {
        let r = 0, gg = 0, b = 0;
        for (let k = 0; k < K; k++) { const pk = A[k * n + p]; r += pk * soft[k][0]; gg += pk * soft[k][1]; b += pk * soft[k][2]; }
        d[4 * p] = r; d[4 * p + 1] = gg; d[4 * p + 2] = b; d[4 * p + 3] = 255;
      }
    }
    octx.putImageData(im, 0, 0);
    ctx.clearRect(0, 0, S, S);
    ctx.save();
    rr(ctx, 0, 0, S, S, 8); ctx.clip();
    DV.zoom.apply(ctx, DV.canvas._dpr || 1);
    const sc = 1 / DV.zoom.k; // markers and lines keep their on-screen size while the map magnifies
    ctx.imageSmoothingEnabled = true;
    ctx.drawImage(DV.off, 0, 0, S, S);
    const cell = S / G;
    const drawSegs = (segs, color, width) => {
      ctx.strokeStyle = color; ctx.lineWidth = width * sc; ctx.lineCap = 'round';
      ctx.beginPath();
      for (let i = 0; i < segs.length; i += 4) { ctx.moveTo((segs[i] + 0.5) * cell, (segs[i + 1] + 0.5) * cell); ctx.lineTo((segs[i + 2] + 0.5) * cell, (segs[i + 3] + 0.5) * cell); }
      ctx.stroke();
    };
    // axes
    ctx.strokeStyle = theme.c('ink', 0.1); ctx.lineWidth = sc;
    ctx.beginPath(); ctx.moveTo(S / 2, 0); ctx.lineTo(S / 2, S); ctx.moveTo(0, S / 2); ctx.lineTo(S, S / 2); ctx.stroke();
    // neuron lines (first hidden layer zero crossings)
    if (App.overlays.neuronLines && L > 1) {
      const Z = g.F.z[1];
      for (let j = 0; j < net.sizes[1]; j++) {
        if (!net.mask[1][j]) continue;
        const sel = App.selected && App.selected.l === 1 && App.selected.j === j;
        drawSegs(NA.engine.contour(Z.subarray(j * n, (j + 1) * n), G, G, 0), sel ? theme.c('focus', 0.95) : theme.c('ink', 0.38), sel ? 2.2 : 1.1);
      }
    }
    if (hov) drawSegs(NA.engine.contour(g.F.z[hov.l].subarray(hov.j * n, (hov.j + 1) * n), G, G, 0), theme.c('focus', 0.9), 2);
    // decision boundary
    if (!hov) {
      const A = g.F.a[L];
      if (ds.task === 'binary') drawSegs(NA.engine.contour(A.subarray(0, n), G, G, 0.5), theme.c('ink', 0.85), 1.6);
      else {
        const K = net.sizes[L], margin = new Float32Array(n);
        for (let k = 0; k < K; k++) {
          for (let p = 0; p < n; p++) { let other = 0; for (let q = 0; q < K; q++) if (q !== k) other = Math.max(other, A[q * n + p]); margin[p] = A[k * n + p] - other; }
          drawSegs(NA.engine.contour(margin, G, G, 0), theme.c('ink', 0.75), 1.4);
        }
      }
    }
    // data points: test squares first, then validation rings, training dots on top
    const preds = App.overlays.errors ? App.predictionsAll() : null;
    const rad = (ds.n > 450 ? 3 : ds.n > 200 ? 3.6 : 4.4) * sc;
    for (const split of [2, 1, 0]) {
      if (!App.overlays[SPLIT_KEY[split]]) continue;
      for (let k = 0; k < ds.n; k++) {
        if (ds.split[k] !== split) continue;
        const x = ((ds.raw[2 * k] + 1) / 2) * S, y = ((1 - ds.raw[2 * k + 1]) / 2) * S;
        const r = k === DV.hoverPoint ? rad + 2 * sc : rad;
        ctx.beginPath(); ctx.arc(x, y, r + 1.4 * sc, 0, Math.PI * 2); ctx.fillStyle = theme.c('panel'); ctx.fill();
        drawSample(ctx, x, y, r, split, theme.classColor(ds.y[k]), sc);
        if (preds && preds[k] !== ds.y[k]) {
          ctx.beginPath(); ctx.arc(x, y, r + 3.4 * sc, 0, Math.PI * 2);
          ctx.lineWidth = 1.6 * sc; ctx.strokeStyle = theme.c('critical'); ctx.stroke();
        }
      }
    }
    // probe
    const p = App.probe;
    if (p && p.raw.length === 2) {
      const x = ((p.raw[0] + 1) / 2) * S, y = ((1 - p.raw[1]) / 2) * S;
      ctx.strokeStyle = theme.c('ink', 0.28); ctx.lineWidth = sc;
      ctx.beginPath(); ctx.moveTo(x, 0); ctx.lineTo(x, S); ctx.moveTo(0, y); ctx.lineTo(S, y); ctx.stroke();
      ctx.beginPath(); ctx.arc(x, y, 7 * sc, 0, Math.PI * 2); ctx.lineWidth = 3.5 * sc; ctx.strokeStyle = theme.c('panel'); ctx.stroke();
      ctx.lineWidth = 2 * sc; ctx.strokeStyle = theme.c('ink'); ctx.stroke();
    }
    ctx.restore();
    // frame & ticks
    ctx.strokeStyle = theme.c('line-2'); ctx.lineWidth = 1;
    rr(ctx, 0.5, 0.5, S - 1, S - 1, 8); ctx.stroke();
    if (DV.zoom.k > 1.01) return; // axis names only make sense unzoomed
    // axis names at the ends of the zero lines, on a small backing so they read over any colour
    ctx.font = `600 11px ${NA.getFont('mono')}`;
    const label = (text, x, y, align) => {
      const w = ctx.measureText(text).width + 8, x0 = align === 'right' ? x - w : x;
      ctx.fillStyle = theme.c('panel', 0.82); rr(ctx, x0, y - 8, w, 16, 4); ctx.fill();
      ctx.fillStyle = theme.c('ink'); ctx.textAlign = 'left'; ctx.textBaseline = 'middle'; ctx.fillText(text, x0 + 4, y);
    };
    label('x₁', S - 6, S / 2 + 12, 'right');
    label('x₂', S / 2 + 6, 14, 'left');
    if (hov) {
      ctx.textAlign = 'right'; ctx.textBaseline = 'top'; ctx.font = `600 11px ${NA.getFont('ui')}`;
      const label = `Showing ${App.neuronName(hov.l, hov.j)}`;
      const w = ctx.measureText(label).width + 14;
      ctx.fillStyle = theme.c('panel', 0.9); rr(ctx, S - w - 6, 6, w, 20, 5); ctx.fill();
      ctx.fillStyle = theme.c('ink'); ctx.fillText(label, S - 13, 10);
    }
  }

  // ================================================================== 1-D curve
  function buildLine() {
    const tools = el('div', { class: 'chip-row', role: 'group', 'aria-label': 'Plot overlays' },
      chip('Train', 'train', 'Show training points (filled dots)'),
      chip('Validation', 'val', 'Show validation points (rings)'),
      chip('Test', 'test', 'Show the held-out test points (squares)'),
      chip('Contributions', 'contributions', 'Show each last-layer neuron’s contribution w·a(x)'));
    DV.canvas = el('canvas', { class: 'line-canvas', 'aria-label': 'Network output as a function of x' });
    DV.readout = el('div', { class: 'readout-line', 'aria-live': 'polite' });
    DV.legendEl = el('div', { class: 'mini-legend' },
      el('span', { class: 'lg-item' }, el('i', { class: 'key-line key-fit' }), 'network ŷ(x)'),
      el('span', { class: 'lg-item' }, el('i', { class: 'key-line key-target' }), 'true function'),
      el('span', { class: 'lg-item' }, el('i', { class: 'key-line key-contrib' }), 'w·a per neuron'),
      el('span', { class: 'lg-item' }, el('i', { class: 'key-tri' }), 'kink'));
    DV.outEl.append(tools, el('div', { class: 'canvas-wrap' }, DV.canvas), DV.readout, DV.legendEl);
    const ds = App.ds;
    let lo = Infinity, hi = -Infinity;
    for (let k = 0; k < ds.n; k++) { lo = Math.min(lo, ds.y[k]); hi = Math.max(hi, ds.y[k]); }
    const span = hi - lo || 1;
    DV.yRange = [lo - span * 0.28, hi + span * 0.28];
    let dragging = false;
    const xOf = (ev) => { const r = DV.canvas.getBoundingClientRect(), f = DV.lineFrame; return clamp(((ev.clientX - r.left - f.x0) / f.w) * 2 - 1, -1, 1); };
    DV.canvas.addEventListener('pointerdown', (ev) => { dragging = true; DV.canvas.setPointerCapture(ev.pointerId); const k = nearestLinePoint(ev); if (k !== null) App.setProbeSample(k); else App.setProbe([xOf(ev)], null); });
    DV.canvas.addEventListener('pointermove', (ev) => {
      if (!DV.lineFrame) return;
      if (dragging) { App.setProbe([xOf(ev)], null); return; }
      const k = nearestLinePoint(ev);
      DV.hoverPoint = k;
      App.previewProbe([k !== null ? App.ds.raw[k] : xOf(ev)]);
      if (k !== null) NA.tooltip.show(sampleTip(k), ev.clientX, ev.clientY); else NA.tooltip.hide();
    });
    DV.canvas.addEventListener('pointerup', () => { dragging = false; });
    DV.canvas.addEventListener('pointerleave', () => { DV.hoverPoint = null; App.previewProbe(null); NA.tooltip.hide(); });
  }
  function nearestLinePoint(ev) {
    const f = DV.lineFrame, ds = App.ds, r = DV.canvas.getBoundingClientRect(), mx = ev.clientX - r.left, my = ev.clientY - r.top;
    let best = null, bd = 8;
    for (let k = 0; k < ds.n; k++) {
      if (!visibleSample(k)) continue;
      const d = Math.hypot(f.X(ds.raw[k]) - mx, f.Y(ds.y[k]) - my);
      if (d < bd) { bd = d; best = k; }
    }
    return best;
  }

  function drawLine() {
    const ctx = DV.ctx, W = DV.cw, H = DV.ch, ds = App.ds, net = App.net, L = net.L, g = App.grid(), n = g.n;
    const padL = 34, padR = 10, padT = 10, padB = 30;
    const x0 = padL, w = W - padL - padR, y0 = padT, h = H - padT - padB;
    const [ylo, yhi] = DV.yRange;
    const X = (x) => x0 + ((x + 1) / 2) * w, Y = (y) => y0 + h - ((y - ylo) / (yhi - ylo)) * h;
    DV.lineFrame = { x0, w, X, Y };
    ctx.clearRect(0, 0, W, H);
    ctx.fillStyle = theme.c('panel-2'); rr(ctx, x0, y0, w, h, 6); ctx.fill();
    // grid
    ctx.strokeStyle = theme.c('line'); ctx.lineWidth = 1;
    ctx.font = `500 10px ${NA.getFont('mono')}`; ctx.fillStyle = theme.c('muted');
    const yt = niceTicks(ylo, yhi, 5);
    ctx.textAlign = 'right'; ctx.textBaseline = 'middle';
    for (const t of yt) { const yy = Math.round(Y(t)) + 0.5; ctx.beginPath(); ctx.moveTo(x0, yy); ctx.lineTo(x0 + w, yy); ctx.stroke(); ctx.fillText(fmt(t, 1), x0 - 6, yy); }
    ctx.textAlign = 'center'; ctx.textBaseline = 'top';
    for (const t of [-1, -0.5, 0, 0.5, 1]) { const xx = Math.round(X(t)) + 0.5; ctx.beginPath(); ctx.moveTo(xx, y0); ctx.lineTo(xx, y0 + h); ctx.stroke(); ctx.fillText(fmt(t, 1), xx, y0 + h + 13); }
    ctx.save(); rr(ctx, x0, y0, w, h, 6); ctx.clip();
    // target function
    const f = D.CURVES[ds.id].f;
    ctx.strokeStyle = theme.c('ink-2', 0.55); ctx.lineWidth = 1.3;
    ctx.beginPath(); for (let i = 0; i <= 200; i++) { const x = -1 + i / 100; i ? ctx.lineTo(X(x), Y(f(x))) : ctx.moveTo(X(x), Y(f(x))); } ctx.stroke();
    // contributions of the last hidden layer
    if (App.overlays.contributions && L > 1) {
      const A = g.F.a[L - 1], W2 = net.W[L - 1], m = net.sizes[L - 1];
      for (let j = 0; j < m; j++) {
        const wj = W2[j], focus = App.hover && App.hover.l === L - 1 && App.hover.j === j || App.selected && App.selected.l === L - 1 && App.selected.j === j;
        ctx.strokeStyle = theme.divColor(Math.sign(wj || 1) * 0.75, focus ? 1 : 0.5);
        ctx.lineWidth = focus ? 2.4 : 1.1;
        ctx.beginPath();
        for (let p = 0; p < n; p++) { const yy = Y(wj * A[j * n + p]); p ? ctx.lineTo(X(g.raw[p]), yy) : ctx.moveTo(X(g.raw[p]), yy); }
        ctx.stroke();
      }
      const b = net.b[L - 1][0];
      ctx.strokeStyle = theme.c('muted', 0.8); ctx.lineWidth = 1;
      ctx.beginPath(); ctx.moveTo(x0, Y(b)); ctx.lineTo(x0 + w, Y(b)); ctx.stroke();
      ctx.fillStyle = theme.c('muted'); ctx.textAlign = 'left'; ctx.textBaseline = 'bottom'; ctx.font = `500 10px ${NA.getFont('mono')}`;
      ctx.fillText('bias ' + fmt(b, 2), x0 + 6, Y(b) - 2);
    }
    // data points
    for (const split of [2, 1, 0]) {
      if (!App.overlays[SPLIT_KEY[split]]) continue;
      for (let k = 0; k < ds.n; k++) {
        if (ds.split[k] !== split) continue;
        drawSample(ctx, X(ds.raw[k]), Y(ds.y[k]), k === DV.hoverPoint ? 4.6 : 3, split, theme.c('ink-2', split ? 0.9 : 0.75));
      }
    }
    // network output
    const out = g.F.a[L];
    ctx.strokeStyle = theme.c('pos'); ctx.lineWidth = 2.5; ctx.lineJoin = 'round';
    ctx.beginPath(); for (let p = 0; p < n; p++) { const yy = Y(out[p]); p ? ctx.lineTo(X(g.raw[p]), yy) : ctx.moveTo(X(g.raw[p]), yy); } ctx.stroke();
    // probe
    const pr = App.probe;
    if (pr) {
      const tr = App.probeTrace(), px = X(pr.raw[0]);
      ctx.strokeStyle = theme.c('ink', 0.35); ctx.lineWidth = 1;
      ctx.beginPath(); ctx.moveTo(px, y0); ctx.lineTo(px, y0 + h); ctx.stroke();
      ctx.beginPath(); ctx.arc(px, Y(tr.a[L][0]), 5.5, 0, Math.PI * 2);
      ctx.fillStyle = theme.c('panel'); ctx.fill(); ctx.lineWidth = 2; ctx.strokeStyle = theme.c('ink'); ctx.stroke();
    }
    ctx.restore();
    // kinks of first-layer rectifier neurons
    if (L > 1) {
      const Z = g.F.z[1];
      for (let j = 0; j < net.sizes[1]; j++) {
        if (net.actOf(1, j).family !== 'rectifier') continue;
        for (let p = 1; p < n; p++) {
          const a = Z[j * n + p - 1], b = Z[j * n + p];
          if ((a > 0) !== (b > 0)) {
            const xk = g.raw[p - 1] + (g.raw[p] - g.raw[p - 1]) * (a / (a - b));
            const focus = App.hover && App.hover.l === 1 && App.hover.j === j || App.selected && App.selected.l === 1 && App.selected.j === j;
            ctx.fillStyle = focus ? theme.c('focus') : theme.c('ink-2', 0.7);
            const xx = X(xk), yy = y0 + h + 2;
            ctx.beginPath(); ctx.moveTo(xx, yy); ctx.lineTo(xx - 4, yy + 7); ctx.lineTo(xx + 4, yy + 7); ctx.closePath(); ctx.fill();
          }
        }
      }
    }
  }

  function niceTicks(lo, hi, count) {
    const span = hi - lo, step0 = span / count, mag = Math.pow(10, Math.floor(Math.log10(step0))), r = step0 / mag;
    const step = (r < 1.5 ? 1 : r < 3 ? 2 : r < 7 ? 5 : 10) * mag, out = [];
    for (let t = Math.ceil(lo / step) * step; t <= hi + 1e-9; t += step) out.push(Math.abs(t) < 1e-9 ? 0 : t);
    return out;
  }
  NA.niceTicks = niceTicks;

  // ================================================================== digits
  function buildImage() {
    const pad = el('canvas', { class: 'pad-canvas', 'aria-label': 'Drawing pad: draw a digit with the mouse or a finger' });
    const preview = el('canvas', { class: 'preview-canvas', 'aria-label': 'The 14 by 14 pixel input the network sees' });
    DV.pad = pad; DV.preview = preview; DV.canvas = pad;
    DV.strokes = []; DV.penWidth = D.PEN;
    const bClear = el('button', { class: 'btn', type: 'button', onclick: () => { DV.strokes = []; DV.penWidth = D.PEN; commitDrawing(); } }, 'Clear');
    const bUndo = el('button', { class: 'btn', type: 'button', onclick: () => { DV.strokes.pop(); commitDrawing(); } }, 'Undo');
    const bRand = el('button', { class: 'btn', type: 'button', onclick: () => { const t = App.ds.valIdx; App.setProbeSample(t[Math.floor(Math.random() * t.length)]); } }, 'Random sample');
    const sal = chip('Saliency', 'saliency', 'Colour each pixel by how much inking it would raise (blue) or lower (orange) the predicted digit’s score');
    DV.predEl = el('div', { class: 'digit-pred' });
    DV.galleryEl = el('div', { class: 'gallery', role: 'list' });
    DV.galleryMode = 'all';
    const gTabs = el('div', { class: 'seg seg-sm', role: 'group', 'aria-label': 'Sample filter' },
      el('button', { type: 'button', 'aria-pressed': 'true', dataset: { g: 'all' } }, 'Validation samples'),
      el('button', { type: 'button', 'aria-pressed': 'false', dataset: { g: 'errors' } }, 'Mistakes'));
    gTabs.addEventListener('click', (e) => { const b = e.target.closest('button'); if (!b) return; DV.galleryMode = b.dataset.g; NA.$$('button', gTabs).forEach((x) => x.setAttribute('aria-pressed', String(x === b))); DV.galleryKey = ''; DV.dirty = true; });
    DV.readout = el('div', { class: 'readout-line', 'aria-live': 'polite' });
    DV.outEl.append(
      el('div', { class: 'pad-row' },
        el('div', { class: 'pad-col' }, el('div', { class: 'eyebrow', text: 'Draw here' }), pad),
        el('div', { class: 'pad-col' }, el('div', { class: 'eyebrow', text: 'Network input' }), preview, sal)),
      el('div', { class: 'btn-row' }, bClear, bUndo, bRand),
      DV.readout,
      DV.predEl,
      el('div', { class: 'gallery-head' }, gTabs),
      DV.galleryEl);
    let drawing = false;
    const pt = (ev) => { const r = pad.getBoundingClientRect(); return [clamp((ev.clientX - r.left) / r.width, 0, 1), clamp((ev.clientY - r.top) / r.height, 0, 1)]; };
    pad.addEventListener('pointerdown', (ev) => {
      ev.preventDefault();
      drawing = true; pad.setPointerCapture(ev.pointerId);
      if (DV.fromSample) { DV.fromSample = false; }
      DV.strokes.push([pt(ev)]);
      commitDrawing(true);
    });
    pad.addEventListener('pointermove', (ev) => {
      if (!drawing) return;
      const s = DV.strokes[DV.strokes.length - 1], p = pt(ev), q = s[s.length - 1];
      if (Math.hypot(p[0] - q[0], p[1] - q[1]) > 0.008) { s.push(p); commitDrawing(true); }
    });
    const end = () => { if (drawing) { drawing = false; commitDrawing(); } };
    pad.addEventListener('pointerup', end); pad.addEventListener('pointercancel', end);
    if (DV._unsubProbe) DV._unsubProbe();
    const syncFromProbe = () => {
      const p = App.pinned;
      if (!App.isImage()) return;
      if (p && p.strokes && p !== DV._lastPinned && !DV._committing) { DV.strokes = p.strokes.map((s) => s.map((q) => q.slice())); DV.penWidth = p.width || D.PEN; }
      DV._lastPinned = p;
    };
    DV._lastPinned = null;
    DV._unsubProbe = App.on('probe', syncFromProbe);
    syncFromProbe();
  }

  // Strokes from the pad live in pad space; loaded samples are already in frame space.
  let rafPending = false;
  function commitDrawing(live) {
    if (live) { if (rafPending) return; rafPending = true; requestAnimationFrame(() => { rafPending = false; commitDrawing(false); }); return; }
    const strokes = DV.strokes.filter((s) => s.length);
    const norm = strokes.length ? D.normalizeStrokes(strokes) : [];
    const img = D.rasterize(norm, D.IMG, DV.penWidth);
    DV._committing = true;
    App.setProbe(img, null, { strokes: DV.strokes.map((s) => s.map((q) => q.slice())), width: DV.penWidth });
    DV._committing = false;
    DV.dirty = true;
  }

  function layoutImage(w) {
    const padSide = Math.round(clamp(w * 0.56, 140, 230)), prevSide = Math.round(clamp(w - padSide - 16, 84, 150));
    DV.padSide = padSide; DV.prevSide = prevSide;
    DV.ctx = NA.fitCanvas(DV.pad, padSide, padSide);
    DV.pctx = NA.fitCanvas(DV.preview, prevSide, prevSide);
  }

  function drawImage() {
    const ctx = DV.ctx, S = DV.padSide, ds = App.ds, G = ds.imgSize, p = App.probe;
    // pad
    ctx.clearRect(0, 0, S, S);
    ctx.fillStyle = theme.c('panel-2'); rr(ctx, 0, 0, S, S, 8); ctx.fill();
    ctx.strokeStyle = theme.c('line'); ctx.lineWidth = 1;
    for (let i = 1; i < 4; i++) { const t = Math.round((i * S) / 4) + 0.5; ctx.beginPath(); ctx.moveTo(t, 6); ctx.lineTo(t, S - 6); ctx.moveTo(6, t); ctx.lineTo(S - 6, t); ctx.stroke(); }
    ctx.strokeStyle = theme.c('ink'); ctx.lineCap = 'round'; ctx.lineJoin = 'round';
    ctx.lineWidth = Math.max(3, DV.penWidth * S * 0.95);
    for (const s of DV.strokes) {
      ctx.beginPath();
      s.forEach((q, i) => (i ? ctx.lineTo(q[0] * S, q[1] * S) : ctx.moveTo(q[0] * S, q[1] * S)));
      if (s.length === 1) ctx.lineTo(s[0][0] * S + 0.1, s[0][1] * S);
      ctx.stroke();
    }
    if (!DV.strokes.length) {
      ctx.fillStyle = theme.c('muted'); ctx.font = `500 13px ${NA.getFont('ui')}`; ctx.textAlign = 'center'; ctx.textBaseline = 'middle';
      ctx.fillText('Draw a digit 0–9', S / 2, S / 2);
    }
    ctx.strokeStyle = theme.c('line-2'); rr(ctx, 0.5, 0.5, S - 1, S - 1, 8); ctx.lineWidth = 1; ctx.stroke();
    // preview with optional saliency
    const pc = DV.pctx, P = DV.prevSide, cell = P / G, ink = theme.tokens.ink, base = theme.tokens['panel-2'];
    pc.clearRect(0, 0, P, P);
    pc.fillStyle = NA.css(base); pc.fillRect(0, 0, P, P);
    for (let i = 0; i < G * G; i++) { const t = clamp(p.raw[i], 0, 1); if (t > 0.01) { pc.fillStyle = NA.css(NA.mixRgb(base, ink, t)); pc.fillRect((i % G) * cell, Math.floor(i / G) * cell, cell + 0.5, cell + 0.5); } }
    if (App.overlays.saliency) {
      const tr = App.probeTrace(), K = App.net.sizes[App.net.L];
      let k = 0; for (let q = 1; q < K; q++) if (tr.a[App.net.L][q] > tr.a[App.net.L][k]) k = q;
      const gIn = App.net.inputGradient(p.x, 0, k), m = NA.maxAbs(gIn) || 1;
      for (let i = 0; i < G * G; i++) { const t = gIn[i] / m; if (Math.abs(t) < 0.05) continue; pc.fillStyle = theme.divColor(t, Math.min(0.85, Math.abs(t) * 1.1)); pc.fillRect((i % G) * cell, Math.floor(i / G) * cell, cell + 0.5, cell + 0.5); }
    }
    pc.strokeStyle = theme.c('line'); pc.lineWidth = 1;
    for (let i = 1; i < G; i++) { const t = Math.round(i * cell) + 0.5; pc.beginPath(); pc.moveTo(t, 0); pc.lineTo(t, P); pc.moveTo(0, t); pc.lineTo(P, t); pc.stroke(); }
    pc.strokeStyle = theme.c('line-2'); pc.strokeRect(0.5, 0.5, P - 1, P - 1);
    drawDigitPred();
    drawGallery();
  }

  function drawDigitPred() {
    const tr = App.probeTrace(), L = App.net.L, a = tr.a[L];
    let k = 0; for (let q = 1; q < a.length; q++) if (a[q] > a[k]) k = q;
    const cols = [];
    for (let q = 0; q < a.length; q++) {
      const h = Math.max(2, a[q] * 44);
      cols.push(el('div', { class: 'dp-col' + (q === k ? ' is-top' : ''), title: `Digit ${q}: ${fmtPct(a[q], 1)}` },
        el('div', { class: 'dp-bar-wrap' }, el('div', { class: 'dp-bar', style: { height: h + 'px' } })),
        el('div', { class: 'dp-lab', text: String(q) })));
    }
    const head = el('div', { class: 'dp-head' },
      el('span', { class: 'dp-big', text: String(k) }),
      el('span', { class: 'dp-conf', text: `${fmtPct(a[k], a[k] > 0.995 ? 1 : 0)} confident` }),
      App.probe.label !== null ? el('span', { class: 'dp-true', text: `true digit ${App.probe.label}` }) : el('span', { class: 'dp-true', text: 'your drawing' }));
    NA.clear(DV.predEl).append(head, el('div', { class: 'dp-cols' }, ...cols));
  }

  function drawGallery() {
    const ds = App.ds, key = App.version + DV.galleryMode + (App.training ? Math.floor(performance.now() / 1500) : '');
    if (DV.galleryKey === key) return;
    DV.galleryKey = key;
    const preds = App.predictionsAll(), G = ds.imgSize;
    let list = Array.from(ds.valIdx);
    if (DV.galleryMode === 'errors') list = list.filter((k) => preds[k] !== ds.y[k]);
    list = list.slice(0, 40);
    const items = list.map((k) => {
      const c = document.createElement('canvas'); c.width = G; c.height = G;
      const cx = c.getContext('2d'), im = cx.createImageData(G, G), ink = theme.tokens.ink, base = theme.tokens['panel-2'];
      for (let i = 0; i < G * G; i++) { const t = clamp(ds.raw[k * G * G + i], 0, 1); im.data[4 * i] = base[0] + (ink[0] - base[0]) * t; im.data[4 * i + 1] = base[1] + (ink[1] - base[1]) * t; im.data[4 * i + 2] = base[2] + (ink[2] - base[2]) * t; im.data[4 * i + 3] = 255; }
      cx.putImageData(im, 0, 0);
      const wrong = preds[k] !== ds.y[k];
      return el('button', { class: 'g-item' + (wrong ? ' is-wrong' : ''), type: 'button', role: 'listitem', title: `True ${ds.y[k]}, predicted ${preds[k]}`, onclick: () => App.setProbeSample(k) },
        c, el('span', { class: 'g-lab', text: wrong ? `${ds.y[k]}→${preds[k]}` : String(ds.y[k]) }));
    });
    if (!items.length) items.push(el('div', { class: 'empty-note', text: 'No mistakes on the validation set.' }));
    NA.clear(DV.galleryEl).append(...items);
  }

  // ================================================================== readout + data controls
  function updateReadout() {
    if (!DV.readout) return;
    const p = App.probe, tr = App.probeTrace(), ds = App.ds, L = App.net.L;
    const parts = [];
    if (ds.kind === 'plane') {
      parts.push(el('span', { class: 'ro-k', text: 'Probe' }), el('span', { class: 'mono', text: `(${fmt(p.raw[0], 2)}, ${fmt(p.raw[1], 2)})` }));
      if (ds.task === 'binary') parts.push(el('span', { class: 'ro-k', text: 'P(blue)' }), el('span', { class: 'mono', text: fmt(tr.a[L][0], 3) }));
      else { let k = 0; for (let q = 1; q < tr.a[L].length; q++) if (tr.a[L][q] > tr.a[L][k]) k = q; parts.push(el('span', { class: 'ro-k', text: 'Predicts' }), el('span', { text: `${['orange', 'blue', 'green'][k]} ${fmtPct(tr.a[L][k])}` })); }
      if (p.label !== null) parts.push(el('span', { class: 'ro-k', text: 'True' }), el('span', { text: ['orange', 'blue', 'green'][p.label] }));
      else parts.push(el('span', { class: 'ro-hint', text: 'Click to pin the probe' }));
    } else if (ds.kind === 'line') {
      parts.push(el('span', { class: 'ro-k', text: 'x' }), el('span', { class: 'mono', text: fmt(p.raw[0], 2) }), el('span', { class: 'ro-k', text: 'ŷ' }), el('span', { class: 'mono', text: fmt(tr.a[L][0], 3) }));
      if (p.label !== null) parts.push(el('span', { class: 'ro-k', text: 'target' }), el('span', { class: 'mono', text: fmt(p.label, 3) }));
    } else {
      parts.push(el('span', { class: 'ro-hint', text: p.label !== null ? `Test sample · true digit ${p.label}. Draw on top of it to change it.` : 'Your drawing is centred and scaled to fit 14×14 pixels, just like the training digits.' }));
    }
    NA.clear(DV.readout).append(...parts);
  }

  function buildDataControls() {
    const ds = App.ds, sp = App.spec;
    NA.clear(DV.controls);
    const kindSets = Object.entries(D.DATASETS).filter(([id, d]) => d.kind === ds.kind);
    if (kindSets.length > 1) {
      const tiles = el('div', { class: 'ds-tiles', role: 'radiogroup', 'aria-label': 'Dataset' });
      for (const [id, d] of kindSets) {
        const c = document.createElement('canvas');
        const b = el('button', { class: 'ds-tile', type: 'button', role: 'radio', 'aria-checked': String(id === ds.id), title: `${d.label}: ${d.hint}`, onclick: () => App.applyPatch({ dataset: id }) }, c, el('span', { text: d.label }));
        tiles.appendChild(b);
        drawDatasetThumb(c, id);
      }
      DV.controls.append(el('div', { class: 'ctl-block' }, el('div', { class: 'eyebrow', text: 'Dataset' }), tiles));
    } else {
      DV.controls.append(el('div', { class: 'ctl-block' }, el('div', { class: 'eyebrow', text: 'Dataset' }), el('p', { class: 'ctl-note', text: `${ds.def.label}: ${ds.n.toLocaleString()} images. ${ds.trainIdx.length.toLocaleString()} for training, ${ds.valIdx.length.toLocaleString()} for validation while it trains, and ${ds.testIdx.length.toLocaleString()} held out for the final test. Generated from stroke templates with random slant, jitter and pen width.` })));
    }
    if (ds.kind !== 'image') {
      const sliders = el('div', { class: 'ctl-grid' },
        slider('Noise', 'noise', 0, 1, 0.05, sp.noise, (v) => fmtPct(v)),
        slider('Training share (rest: half validation, half test)', 'trainRatio', 0.2, 0.9, 0.05, sp.trainRatio, (v) => fmtPct(v)),
        slider('Points', 'n', 40, ds.kind === 'line' ? 400 : 800, 20, sp.n, (v) => String(v)));
      DV.controls.append(el('div', { class: 'ctl-block' }, sliders,
        el('button', { class: 'btn btn-quiet', type: 'button', onclick: () => App.applyPatch({ dataSeed: (sp.seed || 1) + 1 }) }, 'New sample of points')));
    }
    if (ds.kind === 'plane') {
      const feats = el('div', { class: 'chip-row', role: 'group', 'aria-label': 'Input features' });
      for (const f of D.FEATURES) {
        const on = ds.featureIds.includes(f.id);
        feats.appendChild(el('button', { class: 'chip chip-mono', type: 'button', 'aria-pressed': String(on), onclick: () => {
          let ids = ds.featureIds.slice();
          if (on) ids = ids.filter((x) => x !== f.id); else ids = D.FEATURES.map((q) => q.id).filter((id) => id === f.id || ids.includes(id));
          if (ids.length) App.applyPatch({ features: ids });
        } }, f.label));
      }
      DV.controls.append(el('div', { class: 'ctl-block' }, el('div', { class: 'eyebrow', text: 'Input features' }), feats,
        el('p', { class: 'ctl-note', text: 'Engineered features give the first layer curved building blocks. Changing them resets the weights.' })));
    }
  }

  function slider(label, key, min, max, step, value, show) {
    const id = 'ds-' + key;
    const out = el('output', { class: 'mono', for: id, text: show(value) });
    const input = el('input', { id, type: 'range', min, max, step, value });
    input.addEventListener('input', () => { out.textContent = show(+input.value); });
    input.addEventListener('change', () => App.applyPatch({ [key]: +input.value }));
    return el('label', { class: 'slider', for: id }, el('span', { class: 'sl-top' }, el('span', { text: label }), out), input);
  }

  function drawDatasetThumb(c, id) {
    const S = 44, ctx = NA.fitCanvas(c, S, S), d = D.DATASETS[id];
    ctx.fillStyle = theme.c('panel-2'); rr(ctx, 0, 0, S, S, 5); ctx.fill();
    const ds = D.buildDataset({ dataset: id, n: d.kind === 'line' ? 60 : 140, noise: 0.1, trainRatio: 1, seed: 3 });
    if (d.kind === 'plane') {
      for (let k = 0; k < ds.n; k++) { ctx.fillStyle = theme.classColor(ds.y[k]); ctx.beginPath(); ctx.arc(((ds.raw[2 * k] + 1) / 2) * S, ((1 - ds.raw[2 * k + 1]) / 2) * S, 1.6, 0, Math.PI * 2); ctx.fill(); }
    } else {
      const f = D.CURVES[id].f;
      ctx.strokeStyle = theme.c('pos'); ctx.lineWidth = 1.6;
      ctx.beginPath(); for (let i = 0; i <= 40; i++) { const x = -1 + i / 20, y = f(x); const px = 4 + ((x + 1) / 2) * (S - 8), py = S / 2 - y * S * 0.32; i ? ctx.lineTo(px, py) : ctx.moveTo(px, py); } ctx.stroke();
    }
  }

  NA.dataView = DV;
})(window.NA);
