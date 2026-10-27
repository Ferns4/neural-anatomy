/* landscape.js — the loss around the network's weights, as a map. Two directions d₁ and d₂ through weight space
   span a plane through the current weights w; the map shows the training loss at w + a·d₁ + b·d₂ over a grid of
   (a, b), and the path training took, projected onto the plane. Two choices of plane:
   - along the training path: the two directions the weights moved along most (principal components of the
     snapshots in the history), so the path lies almost flat in the map;
   - random: two random directions, each scaled neuron by neuron to the size of that neuron's incoming weights
     ("filter normalisation", Li et al. 2018), so a step of 1 means the same amount of change everywhere.
   The grid is computed a slice at a time while training is paused, on a sample of the training points. */
(function (NA) {
  'use strict';
  const { theme, clamp, fmtSci: fmt, fmtPct, el } = NA; // fmtSci: tiny losses as 4.1×10⁻⁴, not 4.1e−4
  const App = NA.app;

  const G = 29; // grid points along each direction
  const LS = { mode: 'path', job: null, map: null, key: '', hover: null, dirty: true };

  // ------------------------------------------------------------------ the plane
  const dot = (a, b) => { let s = 0; for (let i = 0; i < a.length; i++) s += a[i] * b[i]; return s; };
  const norm = (a) => Math.sqrt(dot(a, a));

  // Random directions, scaled per neuron to the size of its incoming weights; biases are left out (direction 0).
  function randomPlane(net, w, seed) {
    const rng = new NA.RNG(seed), dirs = [new Float64Array(w.length), new Float64Array(w.length)];
    for (const d of dirs) {
      let o = 0;
      for (let l = 0; l < net.L; l++) {
        const nIn = net.sizes[l], nOut = net.sizes[l + 1];
        for (let j = 0; j < nOut; j++) {
          let ss = 0, sw = 0;
          for (let i = 0; i < nIn; i++) { const v = rng.gauss(); d[o + j * nIn + i] = v; ss += v * v; sw += w[o + j * nIn + i] ** 2; }
          const k = Math.sqrt(sw) / (Math.sqrt(ss) || 1);
          for (let i = 0; i < nIn; i++) d[o + j * nIn + i] *= k;
        }
        o += nIn * nOut + nOut; // the biases stay 0
      }
    }
    return { d1: dirs[0], d2: dirs[1], range: [[-1, 1], [-1, 1]], explained: null };
  }

  /* The plane the training path moved in most: the top two principal components of the snapshots' offsets from w,
     found through the small Gram matrix of the snapshots (power iteration with deflation). */
  function pathPlane(w, snaps) {
    const S = snaps.length, M = snaps.map((s) => { const v = new Float64Array(w.length); for (let i = 0; i < w.length; i++) v[i] = s.params[i] - w[i]; return v; });
    const Gm = M.map((a) => M.map((b) => dot(a, b))), trace = Gm.reduce((s, row, i) => s + row[i], 0);
    if (!(trace > 1e-12)) return null;
    const comps = [];
    for (let c = 0; c < 2; c++) {
      let u = new Float64Array(S).map((_, i) => 1 + 0.01 * i), lam = 0;
      for (let it = 0; it < 200; it++) {
        const v = new Float64Array(S);
        for (let i = 0; i < S; i++) for (let j = 0; j < S; j++) v[i] += Gm[i][j] * u[j];
        for (const p of comps) { const k = dot(v, p.u); for (let i = 0; i < S; i++) v[i] -= k * p.u[i]; }
        lam = norm(v); if (!(lam > 1e-15)) break;
        for (let i = 0; i < S; i++) v[i] /= lam;
        u = v;
      }
      if (!(lam > trace * 1e-9)) break;
      comps.push({ u, lam });
    }
    if (comps.length < 2) return null;
    const dirs = comps.map(({ u }) => {
      const d = new Float64Array(w.length);
      for (let s = 0; s < S; s++) { const k = u[s]; if (k) { const m = M[s]; for (let i = 0; i < d.length; i++) d[i] += k * m[i]; } }
      const n = norm(d); for (let i = 0; i < d.length; i++) d[i] /= n || 1;
      return d;
    });
    // the second direction made exactly perpendicular to the first
    const k = dot(dirs[1], dirs[0]); for (let i = 0; i < w.length; i++) dirs[1][i] -= k * dirs[0][i];
    const n2 = norm(dirs[1]); for (let i = 0; i < w.length; i++) dirs[1][i] /= n2 || 1;
    const coords = M.map((m) => [dot(m, dirs[0]), dot(m, dirs[1])]);
    const span = (c) => { const lo = Math.min(0, ...coords.map((p) => p[c])), hi = Math.max(0, ...coords.map((p) => p[c])), pad = (hi - lo) * 0.18 || 0.5; return [lo - pad, hi + pad]; };
    return { d1: dirs[0], d2: dirs[1], range: [span(0), span(1)], explained: (comps[0].lam + comps[1].lam) / trace };
  }

  // ------------------------------------------------------------------ computing the map
  function startJob() {
    const net = App.net, w = Float64Array.from(net.getParams()), snaps = App.history.snaps;
    const upTo = App.viewing !== null ? App.viewing : snaps.length - 1, path = snaps.slice(0, upTo + 1);
    let plane = LS.mode === 'path' && path.length >= 3 ? pathPlane(w, path) : null;
    const fellBack = LS.mode === 'path' && !plane;
    if (!plane) plane = randomPlane(net, w, 1234);
    // the path in the plane: each snapshot's offset from w, by least squares on the two directions
    const a11 = dot(plane.d1, plane.d1), a12 = dot(plane.d1, plane.d2), a22 = dot(plane.d2, plane.d2), det = a11 * a22 - a12 * a12 || 1;
    const trail = path.map((s) => {
      let b1 = 0, b2 = 0;
      for (let i = 0; i < w.length; i++) { const v = s.params[i] - w[i]; b1 += v * plane.d1[i]; b2 += v * plane.d2[i]; }
      return { step: s.step, a: (a22 * b1 - a12 * b2) / det, b: (a11 * b2 - a12 * b1) / det };
    });
    const ds = App.ds, idx = App.evalTrain.slice(0, ds.kind === 'image' ? 120 : 200);
    const probe = net.clone(), base = probe.evaluate(ds.X, ds.y, idx, ds.task).loss;
    LS.job = { w, plane, trail, fellBack, probe, idx, base, i: 0, loss: new Float64Array(G * G).fill(NaN), key: LS.wantKey() };
    LS.map = null;
    LS.dirty = true;
  }
  function runJob(budget) {
    const J = LS.job, t0 = performance.now(), P = J.w.length, theta = new Float32Array(P), ds = App.ds;
    const [[a0, a1], [b0, b1]] = J.plane.range;
    while (J.i < G * G && performance.now() - t0 < budget) {
      const r = Math.floor(J.i / G), c = J.i % G, a = a0 + ((a1 - a0) * c) / (G - 1), b = b1 - ((b1 - b0) * r) / (G - 1);
      for (let k = 0; k < P; k++) theta[k] = J.w[k] + a * J.plane.d1[k] + b * J.plane.d2[k];
      J.probe.setParams(theta);
      const v = J.probe.evaluate(ds.X, ds.y, J.idx, ds.task).loss;
      J.loss[J.i] = Number.isFinite(v) ? v : 1e6;
      J.i++;
    }
    if (J.i >= G * G) { LS.map = J; LS.job = null; LS.key = J.key; NA.a11y && LS.canvas && NA.a11y.refresh(); }
    LS.dirty = true;
  }
  // what the map depends on: the weights shown, the plane chosen and the size of the history
  LS.wantKey = () => JSON.stringify([App.version, App.viewing, LS.mode, App.history.snaps.length, App.preset && App.preset.id]);

  // ------------------------------------------------------------------ UI
  LS.build = function (root) {
    LS.root = root;
    NA.clear(root);
    LS.job = null; LS.map = null; LS.key = '';
    LS.modeSeg = el('div', { class: 'seg seg-sm', role: 'group', 'aria-label': 'Directions of the plane' },
      ...[['path', 'Along the training path'], ['random', 'Random directions']].map(([v, label]) => el('button', { type: 'button', 'aria-pressed': String(LS.mode === v), dataset: { v } }, label)));
    LS.modeSeg.addEventListener('click', (e) => { const b = e.target.closest('button'); if (!b || b.dataset.v === LS.mode) return; LS.mode = b.dataset.v; NA.$$('button', LS.modeSeg).forEach((x) => x.setAttribute('aria-pressed', String(x.dataset.v === LS.mode))); LS.key = ''; LS.job = null; LS.dirty = true; });
    LS.canvas = el('canvas', { class: 'ls-canvas' });
    LS.status = el('p', { class: 'hs-info ls-status', 'aria-live': 'polite' });
    LS.info = el('p', { class: 'hs-info' });
    LS.legend = el('div', { class: 'mini-legend' },
      el('span', null, el('span', { class: 'ramp ramp-grad', 'aria-hidden': 'true' }), 'training loss, low to high (log scale)'),
      el('span', null, el('i', { class: 'ls-key ls-key-start', 'aria-hidden': 'true' }), 'start'),
      el('span', null, el('i', { class: 'ls-key ls-key-now', 'aria-hidden': 'true' }), 'the weights shown'));
    root.append(el('div', { class: 'hs-top' }, LS.modeSeg), el('div', { class: 'canvas-wrap' }, LS.canvas), LS.status, LS.legend, LS.info);
    LS.canvas.addEventListener('pointermove', (e) => { const r = LS.canvas.getBoundingClientRect(); LS.hover = { x: e.clientX - r.left, y: e.clientY - r.top }; LS.dirty = true; });
    LS.canvas.addEventListener('pointerleave', () => { LS.hover = null; LS.dirty = true; });
    if (NA.a11y) NA.a11y.describe(LS.canvas, describe);
    LS.layout();
  };
  LS.enter = function () { LS.dirty = true; LS.layout(); };
  LS.layout = function () {
    if (!LS.canvas || !LS.root || LS.root.hidden) return;
    const w = Math.max(160, Math.min(LS.root.clientWidth || 300, 520));
    LS.size = w; LS.ctx = NA.fitCanvas(LS.canvas, w, w);
    LS.dirty = true;
  };
  LS.frame = function (now) {
    if (!LS.ctx || !LS.root || LS.root.hidden) return;
    // a new map once training pauses (a map of weights that keep moving would be out of date at once)
    if (App.training !== LS.wasTraining) { LS.wasTraining = App.training; LS.dirty = true; }
    if (!App.training) {
      const want = LS.wantKey();
      if (LS.job && LS.job.key !== want) LS.job = null;
      if (!LS.job && want !== LS.key && now - (LS.lastStart || 0) > 250) { LS.lastStart = now; startJob(); }
      if (LS.job) runJob(12);
    }
    if (!LS.dirty) return;
    LS.dirty = false;
    draw();
  };

  function draw() {
    const ctx = LS.ctx, W = LS.size, J = LS.job || LS.map;
    ctx.clearRect(0, 0, W, W);
    const pad = { l: 30, r: 8, t: 8, b: 26 }, q = { x: pad.l, y: pad.t, w: W - pad.l - pad.r, h: W - pad.t - pad.b };
    ctx.fillStyle = theme.c('panel-2'); ctx.fillRect(q.x, q.y, q.w, q.h);
    if (!J) { status(App.training ? 'The map is drawn while training is paused.' : 'Working out the map…'); return; }
    const L = J.loss, [[a0, a1], [b0, b1]] = J.plane.range;
    let lo = Infinity, hi = -Infinity;
    for (const v of L) if (Number.isFinite(v)) { lo = Math.min(lo, v); hi = Math.max(hi, v); }
    const llo = Math.log10(Math.max(lo, 1e-6)), lhi = Math.max(llo + 0.3, Math.min(Math.log10(Math.max(hi, 1e-6)), llo + 3));
    // the grid as an image: the log of the loss interpolated between grid points, in twelve bands so the levels read
    // like contour lines (points not computed yet stay blank)
    const U = 4, R = (G - 1) * U + 1, lg = Array.from(L, (v) => (Number.isFinite(v) ? Math.log10(Math.max(v, 1e-6)) : NaN));
    const c = document.createElement('canvas'); c.width = c.height = R;
    const g = c.getContext('2d'), img = g.createImageData(R, R), lut = theme.gradLUT, bg = theme.tokens['panel-2'];
    for (let y = 0; y < R; y++) {
      const r0 = Math.min(G - 2, Math.floor(y / U)), fy = y / U - r0;
      for (let x = 0; x < R; x++) {
        const c0 = Math.min(G - 2, Math.floor(x / U)), fx = x / U - c0, k = r0 * G + c0;
        const v = (lg[k] * (1 - fx) + lg[k + 1] * fx) * (1 - fy) + (lg[k + G] * (1 - fx) + lg[k + G + 1] * fx) * fy, p = (y * R + x) * 4;
        if (!Number.isFinite(v)) { img.data[p] = bg[0]; img.data[p + 1] = bg[1]; img.data[p + 2] = bg[2]; img.data[p + 3] = 255; continue; }
        const t = Math.floor(clamp((v - llo) / (lhi - llo), 0, 0.999) * 12) / 11, i = Math.round(t * 255) * 3;
        img.data[p] = lut[i]; img.data[p + 1] = lut[i + 1]; img.data[p + 2] = lut[i + 2]; img.data[p + 3] = 255;
      }
    }
    g.putImageData(img, 0, 0);
    ctx.imageSmoothingEnabled = true;
    ctx.drawImage(c, q.x, q.y, q.w, q.h);
    ctx.strokeStyle = theme.c('line-2'); ctx.lineWidth = 1; ctx.strokeRect(q.x + 0.5, q.y + 0.5, q.w - 1, q.h - 1);
    const X = (a) => q.x + ((a - a0) / (a1 - a0)) * q.w, Y = (b) => q.y + ((b1 - b) / (b1 - b0)) * q.h;
    // axes through the weights shown
    ctx.strokeStyle = theme.c('ink', 0.18); ctx.setLineDash([3, 4]);
    ctx.beginPath(); ctx.moveTo(X(0), q.y); ctx.lineTo(X(0), q.y + q.h); ctx.moveTo(q.x, Y(0)); ctx.lineTo(q.x + q.w, Y(0)); ctx.stroke(); ctx.setLineDash([]);
    // the path, from the first snapshot to the weights shown
    const T = J.trail;
    if (T.length > 1) {
      ctx.strokeStyle = theme.c('ink', 0.85); ctx.lineWidth = 1.6; ctx.lineJoin = 'round'; ctx.beginPath();
      T.forEach((p, i) => { const x = X(clamp(p.a, a0, a1)), y = Y(clamp(p.b, b0, b1)); if (i) ctx.lineTo(x, y); else ctx.moveTo(x, y); });
      ctx.stroke();
      // the start: an open ring, so it reads the same on a light or a dark map
      const s = T[0];
      ctx.strokeStyle = theme.c('ink'); ctx.lineWidth = 2;
      ctx.beginPath(); ctx.arc(X(clamp(s.a, a0, a1)), Y(clamp(s.b, b0, b1)), 5, 0, Math.PI * 2); ctx.stroke();
    }
    ctx.fillStyle = theme.c('ink'); ctx.strokeStyle = theme.c('panel'); ctx.lineWidth = 2;
    ctx.beginPath(); ctx.arc(X(0), Y(0), 5, 0, Math.PI * 2); ctx.fill(); ctx.stroke();
    // axis labels
    const mono = (px) => `500 ${px}px ${getComputedStyle(document.body).getPropertyValue('--font-mono') || 'monospace'}`;
    ctx.font = mono(10); ctx.fillStyle = theme.c('muted'); ctx.textBaseline = 'top';
    ctx.textAlign = 'left'; ctx.fillText(fmt(a0, 2), q.x, q.y + q.h + 5);
    ctx.textAlign = 'right'; ctx.fillText(fmt(a1, 2), q.x + q.w, q.y + q.h + 5);
    ctx.textAlign = 'center'; ctx.fillText('direction 1', q.x + q.w / 2, q.y + q.h + 5);
    ctx.save(); ctx.translate(10, q.y + q.h / 2); ctx.rotate(-Math.PI / 2); ctx.textBaseline = 'middle'; ctx.fillText('direction 2', 0, 0); ctx.restore();
    // what is under the pointer
    let hoverText = '';
    if (LS.hover && LS.hover.x >= q.x && LS.hover.x <= q.x + q.w && LS.hover.y >= q.y && LS.hover.y <= q.y + q.h) {
      const c2 = Math.round(((LS.hover.x - q.x) / q.w) * (G - 1)), r2 = Math.round(((LS.hover.y - q.y) / q.h) * (G - 1)), v = L[r2 * G + c2];
      const a = a0 + ((a1 - a0) * c2) / (G - 1), b = b1 - ((b1 - b0) * r2) / (G - 1);
      ctx.strokeStyle = theme.c('focus'); ctx.lineWidth = 1.5; ctx.strokeRect(X(a) - q.w / G / 2, Y(b) - q.h / G / 2, q.w / G, q.h / G);
      hoverText = Number.isFinite(v) ? `Here (${fmt(a, 2)}, ${fmt(b, 2)}): training loss ${fmt(v, 3)}.` : '';
    }
    if (LS.job) status(`Working out the map: ${fmtPct(LS.job.i / (G * G), 0)}…`);
    else status(hoverText || `Training loss from ${fmt(lo, 3)} to ${fmt(hi, 3)} on this map; ${fmt(J.base, 3)} at the weights shown (the dot).${App.training ? ' Training: the map updates when it pauses.' : ''}`);
    LS.info.textContent = J.plane.explained !== null
      ? `The two directions are the ones the weights moved along most during training: together they hold ${fmtPct(J.plane.explained, 0)} of the path’s movement, so the line is training as it really went, seen from above. Each map uses ${J.idx.length} training points.`
      : `${J.fellBack ? 'The history is too short for the path’s directions, so these are random ones. ' : ''}Two random directions, each scaled neuron by neuron to the size of that neuron’s weights: a step of 1 changes every neuron by about as much as its weights are big. In random directions, training’s path barely shows: it moves almost at right angles to both. Each map uses ${J.idx.length} training points.`;
  }
  function status(t) { if (LS.status.textContent !== t) LS.status.textContent = t; }
  function describe() {
    const J = LS.map;
    if (!J) return 'Loss landscape: not computed yet.';
    let lo = Infinity, hi = -Infinity; for (const v of J.loss) { lo = Math.min(lo, v); hi = Math.max(hi, v); }
    return `Loss landscape around the weights shown, ${J.plane.explained !== null ? 'in the plane of the training path' : 'in two random directions'}: training loss from ${fmt(lo, 3)} to ${fmt(hi, 3)}, and ${fmt(J.base, 3)} at the weights shown. The path of training is drawn from its start to the weights shown.`;
  }

  NA.landscape = LS;
})(window.NA);
