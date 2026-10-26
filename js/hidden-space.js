/* hidden-space.js — how each layer re-arranges the data: direct plots for 1–2 neurons,
   PCA projections for wider layers, with animated morphs between layers. */
(function (NA) {
  'use strict';
  const { theme, clamp, fmt, fmtPct, el, rr } = NA;
  const App = NA.app;
  const E = NA.engine;

  const HS = { stage: 1, data: null, dataKey: '', dataTime: 0, anim: null, dirty: true, focusClass: null, playing: null, lastDraw: 0 };

  HS.build = function (root) {
    HS.root = root;
    NA.clear(root);
    const L = App.net.L;
    HS.stage = clamp(HS.stage, 0, L);
    if (HS.stage === 0 && L > 1) HS.stage = 1;
    HS.anim = null; HS.data = null; HS.dataKey = '';
    HS.chips = el('div', { class: 'seg seg-sm', role: 'group', 'aria-label': 'Layer to show' });
    for (let s = 0; s <= L; s++) {
      const label = s === 0 ? 'Input' : s === L ? (App.ds.task === 'regression' ? 'Output' : 'Logits') : 'H' + s;
      HS.chips.appendChild(el('button', { type: 'button', 'aria-pressed': String(s === HS.stage), dataset: { s }, title: s === 0 ? 'Input space' : s === L ? 'Output pre-activations (logits)' : `Hidden layer ${s}` }, label));
    }
    HS.chips.addEventListener('click', (e) => { const b = e.target.closest('button'); if (b) { stopPlay(); go(+b.dataset.s); } });
    HS.playBtn = el('button', { class: 'btn btn-quiet', type: 'button', onclick: () => (HS.playing ? stopPlay() : playThrough()) }, 'Play through layers');
    HS.canvas = el('canvas', { class: 'hs-canvas', 'aria-label': 'Data points as represented inside the selected layer' });
    HS.info = el('p', { class: 'hs-info', 'aria-live': 'polite' });
    HS.legend = el('div', { class: 'mini-legend' });
    root.append(el('div', { class: 'hs-top' }, HS.chips, HS.playBtn), el('div', { class: 'canvas-wrap' }, HS.canvas), HS.info, HS.legend);
    if (App.isImage()) {
      const fc = el('div', { class: 'seg seg-sm seg-digits', role: 'group', 'aria-label': 'Highlight a digit' });
      for (let k = 0; k < 10; k++) fc.appendChild(el('button', { type: 'button', 'aria-pressed': 'false', dataset: { k } }, String(k)));
      fc.addEventListener('click', (e) => { const b = e.target.closest('button'); if (!b) return; const k = +b.dataset.k; HS.focusClass = HS.focusClass === k ? null : k; NA.$$('button', fc).forEach((x) => x.setAttribute('aria-pressed', String(+x.dataset.k === HS.focusClass))); HS.dirty = true; });
      root.append(el('div', { class: 'hs-digits' }, el('span', { class: 'eyebrow', text: 'Highlight' }), fc));
    }
    HS.canvas.addEventListener('pointermove', onMove);
    HS.canvas.addEventListener('pointerleave', () => { HS.hoverK = null; NA.tooltip.hide(); HS.dirty = true; });
    HS.canvas.addEventListener('click', () => { if (HS.hoverK !== null && HS.hoverK !== undefined) App.setProbeSample(HS.hoverK); });
    renderLegend();
  };

  HS.enter = function () { HS.layout(); HS.dirty = true; };
  HS.layout = function () {
    if (!HS.root || HS.root.hidden) return;
    const w = Math.max(160, HS.root.clientWidth);
    HS.size = Math.min(w, 520);
    HS.ctx = NA.fitCanvas(HS.canvas, HS.size, HS.size);
    HS.dirty = true;
  };
  ['weights', 'probe', 'model', 'selection'].forEach((t) => App.on(t, () => { HS.dirty = true; }));
  theme.onChange(() => { HS.dirty = true; });

  function go(s) {
    const L = App.net.L;
    s = clamp(s, 0, L);
    if (s === HS.stage) return;
    const data = ensureData(true);
    if (data && !NA.prefersReducedMotion()) HS.anim = { from: HS.stage, to: s, t0: performance.now(), dur: 750 };
    HS.stage = s;
    NA.$$('button', HS.chips).forEach((b) => b.setAttribute('aria-pressed', String(+b.dataset.s === s)));
    HS.dirty = true;
  }
  function playThrough() {
    const L = App.net.L;
    HS.playBtn.textContent = 'Stop';
    go(0);
    let s = 0;
    const tick = () => { s++; if (s > L) { stopPlay(); return; } go(s); HS.playing = setTimeout(tick, 1300); };
    HS.playing = setTimeout(tick, 900);
  }
  function stopPlay() { if (HS.playing) clearTimeout(HS.playing); HS.playing = null; if (HS.playBtn) HS.playBtn.textContent = 'Play through layers'; }

  // ------------------------------------------------------------------ data per stage
  function ensureData(force) {
    const key = App.version + ':' + App.net.sizes.join('-');
    if (HS.data && HS.dataKey === key) return HS.data;
    if (HS.data && !force && App.training && performance.now() - HS.dataTime < 600) return HS.data;
    HS.data = computeStages();
    HS.dataKey = key; HS.dataTime = performance.now();
    return HS.data;
  }

  function computeStages() {
    const ds = App.ds, net = App.net, L = net.L, F = App.dataPass(), N = F.n;
    const idx = ds.kind === 'image' ? Array.from(ds.testIdx) : NA.range(ds.n);
    const m = idx.length;
    // grid lines for the input plane
    let gridF = null, gridRaw = null;
    const LINES = 13, PTS = 49;
    if (ds.kind === 'plane') {
      gridRaw = new Float64Array(LINES * 2 * PTS * 2);
      let q = 0;
      for (let o = 0; o < 2; o++) for (let li = 0; li < LINES; li++) for (let t = 0; t < PTS; t++) {
        const c = -1 + (2 * li) / (LINES - 1), u = -1 + (2 * t) / (PTS - 1);
        gridRaw[q++] = o ? u : c; gridRaw[q++] = o ? c : u;
      }
      gridF = net.forwardMany(NA.data.featurizePlane(gridRaw, LINES * 2 * PTS, ds.featureIds), LINES * 2 * PTS);
    }
    const gN = gridF ? gridF.n : 0;
    const stages = [];
    let prev = null;
    for (let s = 0; s <= L; s++) {
      const usesRaw = s === 0 && ds.kind === 'plane';
      const d = usesRaw ? 2 : net.sizes[s];
      const src = s === L ? F.z[L] : F.a[s];
      const gsrc = gridF ? (s === L ? gridF.z[L] : gridF.a[s]) : null;
      const vec = (q) => { const v = new Float64Array(d); for (let j = 0; j < d; j++) v[j] = usesRaw ? ds.raw[2 * idx[q] + j] : src[j * N + idx[q]]; return v; };
      const coords = new Float32Array(m * 2);
      let grid = gsrc || usesRaw ? new Float32Array(gN * 2) : null;
      let method, explained = null, project;
      if (d === 1) {
        method = 'strip';
        for (let q = 0; q < m; q++) { coords[2 * q] = src[idx[q]]; coords[2 * q + 1] = jitter(q, ds.y[idx[q]], ds); }
        grid = null;
        project = (v) => [v[0], 0];
      } else if (d === 2) {
        method = 'direct';
        for (let q = 0; q < m; q++) { const v = vec(q); coords[2 * q] = v[0]; coords[2 * q + 1] = v[1]; }
        if (grid) for (let p = 0; p < gN; p++) { grid[2 * p] = usesRaw ? gridRaw[2 * p] : gsrc[p]; grid[2 * p + 1] = usesRaw ? gridRaw[2 * p + 1] : gsrc[gN + p]; }
        project = (v) => [v[0], v[1]];
      } else {
        method = 'pca';
        const arr = new Float32Array(d * m);
        for (let j = 0; j < d; j++) for (let q = 0; q < m; q++) arr[j * m + q] = src[j * N + idx[q]];
        const pca = E.pca2(arr, m, d);
        coords.set(pca.proj);
        explained = pca.explained;
        if (grid) { const v = new Float64Array(d); for (let p = 0; p < gN; p++) { for (let j = 0; j < d; j++) v[j] = gsrc[j * gN + p]; const r = pca.project(v); grid[2 * p] = r[0]; grid[2 * p + 1] = r[1]; } }
        project = (v) => pca.project(v);
      }
      // normalise into a unit box (uniform scale keeps angles honest)
      let cx = 0, cy = 0;
      for (let q = 0; q < m; q++) { cx += coords[2 * q]; cy += coords[2 * q + 1]; }
      cx /= m; cy /= m;
      const rad = [];
      for (let q = 0; q < m; q++) rad.push(Math.max(Math.abs(coords[2 * q] - cx), Math.abs(coords[2 * q + 1] - cy)));
      rad.sort((a, b) => a - b);
      const R = Math.max(1e-6, NA.quantile(rad, 0.985));
      let rot = null;
      const norm = (x, y) => { let nx = (x - cx) / R, ny = (y - cy) / R; if (rot) { const r = rot.apply(nx, ny); nx = r[0]; ny = r[1]; } return [nx, ny]; };
      const ncoords = new Float32Array(m * 2);
      for (let q = 0; q < m; q++) { ncoords[2 * q] = (coords[2 * q] - cx) / R; ncoords[2 * q + 1] = (coords[2 * q + 1] - cy) / R; }
      if (method === 'pca' && prev) {
        rot = E.procrustes2(ncoords, prev.coords, m);
        for (let q = 0; q < m; q++) { const r = rot.apply(ncoords[2 * q], ncoords[2 * q + 1]); ncoords[2 * q] = r[0]; ncoords[2 * q + 1] = r[1]; }
      }
      let ngrid = null;
      if (grid) { ngrid = new Float32Array(gN * 2); for (let p = 0; p < gN; p++) { const r = norm(grid[2 * p], grid[2 * p + 1]); ngrid[2 * p] = r[0]; ngrid[2 * p + 1] = r[1]; } }
      const st = { s, d, method, explained, coords: ncoords, grid: ngrid, toScreen: (v) => { const r = project(v); return norm(r[0], method === 'strip' ? 0 : r[1]); }, cx, cy, R, rot };
      // decision line for binary output in 2-D or 1-D stages
      if (ds.task === 'binary' && s === L - 1 && d === 2 && L > 1) {
        const w = net.W[L - 1], b = net.b[L - 1][0];
        st.line = { w0: w[0] * R, w1: w[1] * R, c: w[0] * cx + w[1] * cy + b };
      } else if (ds.task === 'binary' && s === L) st.vline = (0 - cx) / R;
      stages.push(st);
      prev = st;
    }
    return { stages, idx, m, lines: LINES * 2, pts: PTS };
  }
  function jitter(q, y, ds) {
    const h = Math.sin(q * 12.9898 + 78.233) * 43758.5453;
    const u = h - Math.floor(h);
    return ds.task === 'regression' ? (u - 0.5) * 0.6 : (u - 0.5) * 0.5 + (ds.classes ? (y - (ds.classes - 1) / 2) * 0.55 : 0);
  }

  // ------------------------------------------------------------------ drawing
  HS.frame = function (now) {
    if (!HS.ctx || !HS.root || HS.root.hidden) return;
    const animating = HS.anim && now - HS.anim.t0 < HS.anim.dur;
    if (!HS.dirty && !animating) return;
    if (!animating && App.training && now - HS.lastDraw < 120) return;
    HS.lastDraw = now; HS.dirty = false;
    const data = ensureData(false);
    if (!data) return;
    draw(data, now);
  };

  function draw(data, now) {
    const ctx = HS.ctx, S = HS.size, ds = App.ds, L = App.net.L;
    const st = data.stages[HS.stage];
    let t = 1, from = null;
    if (HS.anim) {
      t = clamp((now - HS.anim.t0) / HS.anim.dur, 0, 1);
      if (t >= 1) HS.anim = null; else { from = data.stages[HS.anim.from]; t = t < 0.5 ? 4 * t * t * t : 1 - Math.pow(-2 * t + 2, 3) / 2; }
    }
    const k = S * 0.43, cxs = S / 2, cys = S / 2;
    const P = (arr, i) => {
      let x = arr === null ? 0 : arr[2 * i], y = arr === null ? 0 : arr[2 * i + 1];
      return [cxs + x * k, cys - y * k];
    };
    const mix = (a, b, i) => {
      if (!from) return P(b, i);
      const p = P(a, i), q = P(b, i);
      return [p[0] + (q[0] - p[0]) * t, p[1] + (q[1] - p[1]) * t];
    };
    ctx.clearRect(0, 0, S, S);
    ctx.fillStyle = theme.c('panel-2'); rr(ctx, 0, 0, S, S, 8); ctx.fill();
    ctx.save(); rr(ctx, 0, 0, S, S, 8); ctx.clip();
    // axes through the origin of the unit box
    ctx.strokeStyle = theme.c('line'); ctx.lineWidth = 1;
    ctx.beginPath(); ctx.moveTo(cxs, 0); ctx.lineTo(cxs, S); ctx.moveTo(0, cys); ctx.lineTo(S, cys); ctx.stroke();
    // warped grid
    const gA = from ? from.grid : null, gB = st.grid;
    if (gB && (!from || gA)) {
      ctx.strokeStyle = theme.c('ink', theme.dark ? 0.2 : 0.16); ctx.lineWidth = 1;
      for (let li = 0; li < data.lines; li++) {
        ctx.beginPath();
        for (let tt = 0; tt < data.pts; tt++) {
          const i = li * data.pts + tt, p = from ? mix(gA, gB, i) : P(gB, i);
          tt ? ctx.lineTo(p[0], p[1]) : ctx.moveTo(p[0], p[1]);
        }
        ctx.stroke();
      }
    }
    // decision boundary in this space
    if (!from) {
      if (st.line) {
        const { w0, w1, c } = st.line;
        ctx.strokeStyle = theme.c('ink', 0.85); ctx.lineWidth = 1.8;
        ctx.beginPath();
        if (Math.abs(w1) > Math.abs(w0)) { const y1 = (-c - w0 * -2) / w1, y2 = (-c - w0 * 2) / w1; ctx.moveTo(cxs - 2 * k, cys - y1 * k); ctx.lineTo(cxs + 2 * k, cys - y2 * k); }
        else { const x1 = (-c - w1 * -2) / w0, x2 = (-c - w1 * 2) / w0; ctx.moveTo(cxs + x1 * k, cys + 2 * k); ctx.lineTo(cxs + x2 * k, cys - 2 * k); }
        ctx.stroke();
      } else if (st.vline !== undefined) {
        ctx.strokeStyle = theme.c('ink', 0.85); ctx.lineWidth = 1.8;
        ctx.beginPath(); ctx.moveTo(cxs + st.vline * k, 0); ctx.lineTo(cxs + st.vline * k, S); ctx.stroke();
      }
    }
    // points
    const isImg = ds.kind === 'image';
    const r = data.m > 450 ? 2.6 : 3.2;
    if (isImg) { ctx.font = `600 10px ${NA.getFont('mono')}`; ctx.textAlign = 'center'; ctx.textBaseline = 'middle'; }
    for (let pass = 0; pass < 2; pass++) {
      for (let q = 0; q < data.m; q++) {
        const kk = data.idx[q], y = ds.y[kk];
        const focus = isImg && HS.focusClass !== null && y === HS.focusClass;
        if (isImg && (pass === 1) !== focus) continue;
        if (!isImg && pass === 1) continue;
        const p = from ? mix(from.coords, st.coords, q) : P(st.coords, q);
        if (isImg) {
          ctx.fillStyle = focus ? theme.c('pos') : HS.focusClass !== null ? theme.c('muted', 0.35) : theme.c('ink-2', 0.75);
          ctx.fillText(String(y), p[0], p[1]);
        } else {
          ctx.fillStyle = ds.task === 'regression' ? theme.divColor(clamp(y, -1, 1)) : theme.classColor(y);
          ctx.beginPath(); ctx.arc(p[0], p[1], q === HS.hoverQ ? r + 2 : r, 0, Math.PI * 2); ctx.fill();
        }
      }
    }
    // probe
    if (!from) {
      const tr = App.probeTrace();
      const v = HS.stage === 0 && ds.kind === 'plane' ? App.probe.raw : HS.stage === L ? tr.z[L] : tr.a[HS.stage];
      const pp = st.toScreen(v);
      const px = cxs + pp[0] * k, py = cys - (st.method === 'strip' ? 0 : pp[1]) * k;
      ctx.beginPath(); ctx.arc(px, py, 7, 0, Math.PI * 2);
      ctx.lineWidth = 3.5; ctx.strokeStyle = theme.c('panel'); ctx.stroke();
      ctx.lineWidth = 2; ctx.strokeStyle = theme.c('ink'); ctx.stroke();
    }
    ctx.restore();
    ctx.strokeStyle = theme.c('line-2'); ctx.lineWidth = 1; rr(ctx, 0.5, 0.5, S - 1, S - 1, 8); ctx.stroke();
    // axis captions
    ctx.font = `500 10px ${NA.getFont('mono')}`; ctx.fillStyle = theme.c('ink-2');
    const [ax, ay] = axisNames(st);
    ctx.textAlign = 'right'; ctx.textBaseline = 'bottom'; ctx.fillText(ax + ' →', S - 8, S - 6);
    if (ay) { ctx.save(); ctx.translate(14, 10); ctx.rotate(Math.PI / 2); ctx.textAlign = 'left'; ctx.textBaseline = 'middle'; ctx.fillText('← ' + ay, 0, 0); ctx.restore(); }
    if (from) {
      ctx.textAlign = 'left'; ctx.textBaseline = 'bottom'; ctx.font = `600 11px ${NA.getFont('ui')}`; ctx.fillStyle = theme.c('ink');
      ctx.fillText(`${stageName(HS.anim ? HS.anim.from : 0)} → ${stageName(HS.stage)}`, 10, S - 7);
    }
    HS.info.textContent = describe(st);
  }

  function stageName(s) { const L = App.net.L; return s === 0 ? 'Input' : s === L ? 'Logits' : 'Hidden ' + s; }
  function axisNames(st) {
    const L = App.net.L;
    if (st.s === 0 && App.isPlane()) return ['x₁', 'x₂'];
    if (st.method === 'direct') return [App.neuronName(st.s, 0, true), App.neuronName(st.s, 1, true)];
    if (st.method === 'strip') return [st.s === L ? 'logit z' : App.neuronName(st.s, 0, true), null];
    return [`PC1 · ${fmtPct(st.explained[0])} of variance`, `PC2 · ${fmtPct(st.explained[1])}`];
  }
  function describe(st) {
    const L = App.net.L, ds = App.ds, name = stageName(st.s);
    if (st.s === 0 && ds.kind === 'plane') return 'The raw input plane. The grid lines are carried through the network so you can see how each layer bends, folds and stretches space.';
    if (st.s === 0) return `${ds.dim} input pixels, projected onto the two directions of greatest variance (PCA). Digits overlap heavily in raw pixel space.`;
    if (st.method === 'strip') return st.s === L ? 'The output neuron’s raw score z. Points right of the line z = 0 are classified blue. A good network pushes the two classes apart along this axis.' : `${name} has a single neuron, so every point is squeezed onto one axis.`;
    if (st.method === 'direct') return `${name} has exactly 2 neurons, so this is its true output, not a projection.${st.line ? ' The line is where the output neuron’s z = 0: if the classes fall on opposite sides, this layer made the problem linearly separable.' : ''}`;
    const ex = st.explained[0] + st.explained[1];
    return `${name} has ${st.d} ${st.s === L ? 'outputs' : 'neurons'}. This PCA view keeps the two directions of greatest variance (${fmtPct(ex)} of the total), rotated to line up with the previous layer.${ex < 0.5 ? ' Much of the structure lives in directions not shown.' : ''}`;
  }

  function renderLegend() {
    const ds = App.ds, items = [];
    if (ds.kind === 'plane') for (let k = 0; k < ds.classes; k++) items.push(el('span', { class: 'lg-item' }, el('i', { class: 'dot', style: { background: `var(--${['neg', 'pos', 'c3'][k]})` } }), App.className(k)));
    if (ds.kind === 'image') items.push(el('span', { class: 'lg-item', text: 'Each test image is drawn as its true digit.' }));
    items.push(el('span', { class: 'lg-item' }, el('i', { class: 'dot dot-probe' }), 'probe'));
    NA.clear(HS.legend).append(...items);
  }

  function onMove(ev) {
    const data = HS.data;
    if (!data || HS.anim) return;
    const r = HS.canvas.getBoundingClientRect(), mx = ev.clientX - r.left, my = ev.clientY - r.top, S = HS.size, k = S * 0.43;
    const st = data.stages[HS.stage];
    let best = null, bd = 9;
    for (let q = 0; q < data.m; q++) {
      const x = S / 2 + st.coords[2 * q] * k, y = S / 2 - st.coords[2 * q + 1] * k;
      const d = Math.hypot(x - mx, y - my);
      if (d < bd) { bd = d; best = q; }
    }
    HS.hoverQ = best;
    HS.hoverK = best === null ? null : data.idx[best];
    HS.dirty = true;
    if (best === null) { NA.tooltip.hide(); return; }
    const kk = data.idx[best], ds = App.ds, pred = App.predictionsAll()[kk];
    const kids = [el('div', { class: 'tt-title', text: `Sample #${kk}` })];
    if (ds.task === 'regression') kids.push(el('div', { class: 'tt-row' }, el('span', { class: 'tt-k', text: 'x → ŷ' }), el('span', { class: 'tt-v', text: `${fmt(ds.raw[kk], 2)} → ${fmt(pred, 2)}` })));
    else {
      kids.push(el('div', { class: 'tt-row' }, el('span', { class: 'tt-k', text: 'true' }), el('span', { class: 'tt-v', text: App.className(ds.y[kk]) })));
      kids.push(el('div', { class: 'tt-row' }, el('span', { class: 'tt-k', text: 'predicted' }), el('span', { class: 'tt-v', text: App.className(pred) })));
    }
    kids.push(el('div', { class: 'tt-hint', text: 'Click to make this the probe' }));
    NA.tooltip.show(kids, ev.clientX, ev.clientY);
  }

  NA.hiddenSpace = HS;
})(window.NA);
