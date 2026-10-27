/* lab-diffusion.js — the diffusion lab: watch noise turn into data step by step, see the forward process
   dissolve the data, and look at the learned denoising field; inspector tabs cover sample quality, training,
   the noise schedule and the network. */
(function (NA) {
  'use strict';
  const { el, clear, clamp, fmt, fmtPct, fmtInt, theme } = NA;
  const rr = (...a) => NA.rr(...a);
  const App = NA.app, H = NA.labHost, UI = NA.labUI, DM = NA.diffusion;

  const DATASETS = [{ id: 'pinwheel', label: 'Pinwheel' }, { id: 'spiral', label: 'Spiral' }, { id: 'moons', label: 'Two moons' }];
  const RANGE = 3; // plotted square: ±3 standard deviations

  const DF = H.transport({
    id: 'diffusion', screens: { network: 'Samples', data: 'Process', insp: 'Inspect' },
    budget: 12, lr: 2e-3, batch: 128, dataset: 'pinwheel', nSamples: 600, sampler: 'ddpm', steps: 100,
    view: 'generate', k: null, tView: 50, show: { data: true, trails: true, x0: false }, probes: [], version: 0,
  });

  // ------------------------------------------------------------------ data and model
  // Standardised points of one of the app's plane datasets: the same generator and seed as training.
  function points(dataset, noise) {
    const ds = NA.data.buildDataset({ dataset, n: 1500, noise, trainRatio: 0.8, seed: 1, features: ['x1', 'x2'] });
    const raw = ds.raw, out = new Float32Array(2 * ds.n);
    let mx = 0, my = 0;
    for (let i = 0; i < ds.n; i++) { mx += raw[2 * i]; my += raw[2 * i + 1]; }
    mx /= ds.n; my /= ds.n;
    let v = 0;
    for (let i = 0; i < ds.n; i++) v += (raw[2 * i] - mx) ** 2 + (raw[2 * i + 1] - my) ** 2;
    const sd = Math.sqrt(v / (2 * ds.n));
    for (let i = 0; i < ds.n; i++) { out[2 * i] = (raw[2 * i] - mx) / sd; out[2 * i + 1] = (raw[2 * i + 1] - my) / sd; }
    return { pts: out, n: ds.n };
  }

  DF.load = function (dataset, scratch) {
    const W = NA.diffusionWeights[dataset];
    this.dataset = dataset;
    this.D = points(dataset, W ? W.noise : 0.03);
    this.model = new DM.Diffusion({ T: W ? W.T : 100, hidden: W ? W.hidden : [64, 64, 64], seed: scratch ? 2 : 1 });
    this.history = []; this.step = 0; this.pretrained = false;
    if (W && !scratch) {
      this.model.net.setParams(NA.tf.decodeWeights(W.weights));
      this.step = W.steps; this.pretrained = true;
      this.history = W.curve.map(([step, loss]) => ({ step, loss }));
    }
    this.runLoss = null; this.quality = null; this.lossByT = null;
    // a fixed noise draw for the forward-process view
    const rng = new NA.RNG(17); this.eps = new Float32Array(2 * this.D.n);
    for (let i = 0; i < this.eps.length; i++) this.eps[i] = rng.gauss();
    this.probes = [];
    this.bump();
    this.generate(false); // open on the reverse process playing: noise turning into data
  };
  DF.bump = function () { this.version++; this.dirty = true; this.inspDirty = true; this.leftDirty = true; };

  DF.trainChunk = function (budget, maxSteps) {
    const t0 = performance.now();
    let k = 0;
    do {
      const r = this.model.trainStep(this.D.pts, this.D.n, this.batch, this.lr);
      this.runLoss = this.runLoss === null ? r.loss : 0.98 * this.runLoss + 0.02 * r.loss;
      this.step++; k++;
      if (this.step % 50 === 0) { this.history.push({ step: this.step, loss: this.runLoss }); if (this.history.length > 500) this.history = this.history.filter((q, i) => i % 2 === 0); }
      // refresh the samples now and then so progress is visible
      if (this.step % 600 === 0) this.generate(true, 25);
    } while (performance.now() - t0 < budget && k < maxSteps);
    this.quality = null; this.lossByT = null;
    this.inspDirty = true; this.leftDirty = true;
    return k;
  };
  DF.onPause = function () { this.generate(true); this.bump(); };
  DF.stepOnce = function () { if (this.training) this.pause(); this.trainChunk(0, 1); this.bump(); };
  DF.reset = function () { if (this.training) this.pause(); this.load(this.dataset, true); NA.ui.toast.show('Random weights: the samples are noise. Press Train; shapes appear after a few thousand steps.'); };

  // The step sequence for sampling: T, …, 0 in `steps` equal jumps.
  DF.schedule = function (steps) { const T = this.model.T, ts = []; for (let k = steps; k >= 0; k--) ts.push(Math.round((k * T) / steps)); return ts; };
  /* Starts a reverse process from fresh noise and keeps every intermediate state, so the slider can scrub
     through it. The steps are computed a few per frame (advanceGen) so the page never stalls.
     instant: jump to the result once it is ready; otherwise animate from noise. */
  DF.generate = function (instant, steps) {
    const n = this.nSamples, rng = new NA.RNG((Math.random() * 1e9) | 0), xy = new Float32Array(2 * n);
    for (let i = 0; i < 2 * n; i++) xy[i] = rng.gauss();
    const ts = this.schedule(steps || this.steps);
    this.gen = { ts, xs: [Float32Array.from(xy)], x0s: [null], n, xy, rng, ddim: this.sampler === 'ddim', done: false, instant };
    this.k = 0;
    this.animating = !instant;
    this.quality = null;
    this.dirty = true; this.inspDirty = true; this.leftDirty = true;
  };
  DF.advanceGen = function (budget) {
    const g = this.gen;
    if (!g || g.done) return;
    const t0 = performance.now();
    while (g.xs.length < g.ts.length && performance.now() - t0 < budget) {
      const i = g.xs.length - 1;
      g.x0s.push(this.model.reverseStep(g.xy, g.n, g.ts[i], g.ddim, g.rng, g.ts[i + 1]));
      g.xs.push(Float32Array.from(g.xy));
    }
    if (g.xs.length === g.ts.length) { g.done = true; if (g.instant) { this.k = g.ts.length - 1; this.dirty = true; } this.inspDirty = true; }
  };
  DF.measure = function () {
    if (!this.gen.done) return null;
    if (this.quality && this.quality.version === this.version + ':' + this.gen.xs.length) return this.quality;
    const g = this.gen, last = g.xs[g.xs.length - 1], half = this.D.n >> 1;
    const q = DM.chamfer(last, g.n, this.D.pts, this.D.n);
    // reference: how far one half of the real data sits from the other half
    const ref = DM.chamfer(this.D.pts.subarray(0, 2 * half), half, this.D.pts.subarray(2 * half), this.D.n - half);
    this.quality = Object.assign(q, { ref, version: this.version + ':' + g.xs.length });
    return this.quality;
  };
  // Average loss at each noise level (10 bands of t), on fresh noise.
  DF.lossBands = function () {
    if (this.lossByT) return this.lossByT;
    const bands = new Float64Array(10), cnt = new Float64Array(10), rng = new NA.RNG(99);
    for (let r = 0; r < 6; r++) {
      const res = this.model.evalBatch(this.D.pts, this.D.n, 256, rng);
      for (let p = 0; p < res.ts.length; p++) { const b = Math.min(9, Math.floor(((res.ts[p] - 1) / this.model.T) * 10)); bands[b] += res.per[p]; cnt[b]++; }
    }
    this.lossByT = Array.from(bands, (s, i) => s / Math.max(1, cnt[i]));
    return this.lossByT;
  };

  // ------------------------------------------------------------------ lifecycle
  DF.activate = function (preset, ui) {
    this.ui = ui; this.preset = preset;
    if (!this.model) this.load(this.dataset);
    this.ro = UI.readouts(ui.readouts, [{ key: 'step', label: 'Step', keep: true }, { key: 'loss', label: 'Loss' }, { key: 't', label: 'Noise step', keep: true }]);
    this.buildHyper(ui.hyper);
    this.buildLeft(ui.left);
    this.buildCenter(ui.center);
    this.tabs = UI.tabs(ui.right, [{ id: 'overview', label: 'Overview' }, { id: 'train', label: 'Training' }, { id: 'sched', label: 'Noise schedule' }, { id: 'net', label: 'Network' }], (id, body) => this.buildTab(id, body));
    this.offTheme = theme.onChange(() => this.bump());
    this.bump();
  };
  DF.deactivate = function () { if (this.offTheme) this.offTheme(); if (this.resizeObs) this.resizeObs.disconnect(); };
  DF.archText = function () { return `(x, y, t) → 3 × 64 SiLU → predicted noise ε̂ · ${this.model.T} noise steps`; };

  DF.frame = function (now) {
    this.runFrame();
    this.advanceGen(this.training ? 4 : 10);
    if (this.animating && now - (this.lastAnim || 0) > (this.gen.ts.length > 40 ? 16 : 70) && this.k < this.gen.xs.length - 1) {
      this.lastAnim = now;
      this.k = Math.min(this.gen.xs.length - 1, this.k + 1);
      if (this.k >= this.gen.ts.length - 1) { this.animating = false; this.inspDirty = true; }
      this.dirty = true; this.syncSlider();
    }
    if (now - (this.lastRo || 0) > 120) {
      this.lastRo = now;
      this.ro.set('step', fmtInt(this.step));
      const last = this.history[this.history.length - 1];
      this.ro.set('loss', fmt(this.runLoss ?? (last && last.loss), 4));
      this.ro.set('t', `${this.currentT()} / ${this.model.T}`);
    }
    if (this.dirty && this.canvas.offsetParent) { this.dirty = false; this.draw(); }
    if (this.leftDirty && this.ui.left.offsetParent && (!this.training || now - (this.lastLeft || 0) > 500)) { this.lastLeft = now; this.leftDirty = false; this.renderLeft(); }
    if (this.inspDirty && this.tabs.body.offsetParent && !this.animating && (!this.training || now - (this.lastInsp || 0) > 600)) { this.lastInsp = now; this.inspDirty = false; this.updateTab(); }
  };
  DF.currentT = function () { return this.view === 'generate' ? this.gen.ts[this.k] : this.tView; };

  DF.buildHyper = function (root) {
    root.append(
      UI.field('Learning rate', UI.select([0.0005, 0.001, 0.002, 0.003].map((v) => ({ value: v, label: String(v) })), this.lr, (v) => { this.lr = +v; })),
      UI.field('Batch', UI.select([64, 128, 256].map((v) => ({ value: v, label: v + ' points' })), this.batch, (v) => { this.batch = +v; })),
      el('div', { class: 'sheet-actions' },
        el('button', { class: 'btn', type: 'button', onclick: () => { NA.mobile && NA.mobile.sheet(false); this.fastForward(1000); } }, 'Train 1,000 steps'),
        el('button', { class: 'btn', type: 'button', onclick: () => { NA.mobile && NA.mobile.sheet(false); if (NA.mobile && NA.mobile.mobile) NA.mobile.go('network'); this.view = 'generate'; this.viewSeg.set('generate'); this.generate(false); } }, 'Generate')));
  };

  // ------------------------------------------------------------------ left: the process
  DF.buildLeft = function (root) {
    const body = UI.panel(root, 'Process');
    this.dsTiles = el('div', { class: 'ds-tiles', role: 'radiogroup', 'aria-label': 'Dataset' });
    for (const d of DATASETS) {
      const c = el('canvas', { 'aria-hidden': 'true' });
      const b = el('button', { class: 'ds-tile', type: 'button', role: 'radio', 'aria-checked': String(d.id === this.dataset), onclick: () => { this.pause(); this.load(d.id); NA.$$('.ds-tile', this.dsTiles).forEach((x) => x.setAttribute('aria-checked', String(x === b))); NA.ui.toast.show(`${d.label}: loaded a denoiser trained on it.`); } }, c, d.label);
      b._c = c; b._id = d.id;
      this.dsTiles.append(b);
    }
    this.samplerSeg = UI.seg([{ value: 'ddpm', label: 'DDPM (random)', title: 'Adds fresh noise at every step' }, { value: 'ddim', label: 'DDIM (deterministic)', title: 'No added noise: the same start always lands in the same place' }], this.sampler, (v) => { this.sampler = v; this.generate(false); }, 'Sampler');
    const steps = UI.select([100, 50, 25, 10].map((v) => ({ value: v, label: `${v} steps` })), this.steps, (v) => { this.steps = +v; this.view = 'generate'; this.viewSeg.set('generate'); this.generate(false); }, { 'aria-label': 'Sampling steps' });
    const count = UI.select([300, 600, 1000].map((v) => ({ value: v, label: `${v} points` })), this.nSamples, (v) => { this.nSamples = +v; this.generate(false); }, { 'aria-label': 'Number of samples' });
    this.schedCanvas = el('canvas');
    body.append(
      UI.section('Training data', UI.note('The model learns this distribution of 2-D points, then has to produce new points that look like it.'), this.dsTiles),
      UI.section('Generate', el('div', { class: 'btn-row' }, el('button', { class: 'btn btn-primary', type: 'button', onclick: () => { this.view = 'generate'; this.viewSeg.set('generate'); if (NA.mobile && NA.mobile.mobile) NA.mobile.go('network'); this.generate(false); } }, 'Generate new points'), steps, count), this.samplerSeg),
      UI.section('How it works',
        el('div', { class: 'fx' },
          el('div', { class: 'fx-line fx-strong', text: 'Forward:  x_t = √ᾱ_t · x₀ + √(1 − ᾱ_t) · ε' }),
          el('div', { class: 'fx-line', text: 'Train:  loss = ‖ε − ε̂(x_t, t)‖²' }),
          el('div', { class: 'fx-line', text: 'Reverse:  x̂₀ = (x_t − √(1 − ᾱ_t) · ε̂) / √ᾱ_t, then step to t − 1' })),
        UI.note('Adding noise is easy and needs no learning. The network learns the reverse: given a noisy point and how noisy it is (t), guess the noise ε that was added. Subtracting a little of that guess at a time walks pure noise back to data.')),
      UI.section('Noise schedule', el('div', { class: 'chart-box' }, this.schedCanvas), el('div', { class: 'mini-legend' },
        el('span', { class: 'lg-item' }, el('i', { class: 'key-line', style: { background: 'var(--pos)' } }), 'signal kept √ᾱ_t'), el('span', { class: 'lg-item' }, el('i', { class: 'key-line', style: { background: 'var(--neg)' } }), 'noise √(1 − ᾱ_t)'))));
  };
  DF.renderLeft = function () {
    for (const b of NA.$$('.ds-tile', this.dsTiles)) {
      const ctx = NA.fitCanvas(b._c, 46, 46), P = b._id === this.dataset ? this.D : (b._pts || (b._pts = points(b._id, NA.diffusionWeights[b._id].noise)));
      ctx.clearRect(0, 0, 46, 46); ctx.fillStyle = theme.c('ink-2');
      for (let i = 0; i < P.n; i += 3) ctx.fillRect(23 + P.pts[2 * i] * 7.5, 23 - P.pts[2 * i + 1] * 7.5, 1.2, 1.2);
    }
    const W = Math.max(200, this.schedCanvas.parentElement.clientWidth), Hh = 110, ctx = NA.fitCanvas(this.schedCanvas, W, Hh), S = this.model.sched, T = this.model.T;
    ctx.clearRect(0, 0, W, Hh);
    const X = (t) => 28 + (t / T) * (W - 36), Y = (v) => 8 + (1 - v) * (Hh - 26);
    ctx.strokeStyle = theme.c('line'); ctx.strokeRect(28.5, 8.5, W - 36, Hh - 26);
    ctx.font = '500 10px ' + (getComputedStyle(document.body).getPropertyValue('--font-mono') || 'monospace'); ctx.fillStyle = theme.c('muted'); ctx.textAlign = 'right'; ctx.textBaseline = 'middle';
    ctx.fillText('1', 24, Y(1)); ctx.fillText('0', 24, Y(0)); ctx.textAlign = 'center'; ctx.textBaseline = 'top'; ctx.fillText('t = 0', X(0) + 12, Hh - 14); ctx.fillText(`t = ${T}`, X(T) - 14, Hh - 14);
    const curve = (fn, col) => { ctx.strokeStyle = col; ctx.lineWidth = 2; ctx.beginPath(); for (let t = 0; t <= T; t++) { const x = X(t), y = Y(fn(t)); t ? ctx.lineTo(x, y) : ctx.moveTo(x, y); } ctx.stroke(); };
    curve((t) => Math.sqrt(S.abar[t]), theme.c('pos'));
    curve((t) => Math.sqrt(1 - S.abar[t]), theme.c('neg'));
    const tc = this.currentT();
    ctx.strokeStyle = theme.c('ink', 0.5); ctx.setLineDash([2, 3]); ctx.beginPath(); ctx.moveTo(X(tc), 8); ctx.lineTo(X(tc), Hh - 18); ctx.stroke(); ctx.setLineDash([]);
  };

  // ------------------------------------------------------------------ centre: samples
  DF.buildCenter = function (root) {
    this.viewSeg = UI.seg([{ value: 'generate', label: 'Generate', title: 'Noise turning into data' }, { value: 'forward', label: 'Add noise', title: 'The forward process on the training data' }, { value: 'field', label: 'Denoiser', title: 'Where the network thinks each point’s clean version is' }], this.view, (v) => { this.view = v; this.syncSlider(); this.dirty = true; this.leftDirty = true; }, 'View');
    const chips = el('div', { class: 'chip-row' },
      UI.chip('Data', this.show.data, (on) => { this.show.data = on; this.dirty = true; }, 'Show the training data faintly'),
      UI.chip('Trails', this.show.trails, (on) => { this.show.trails = on; this.dirty = true; }, 'Paths of 40 points: the last 10 steps with DDPM, the whole path with DDIM'),
      UI.chip('Predicted x̂₀', this.show.x0, (on) => { this.show.x0 = on; this.dirty = true; }, 'At each step, where the network thinks each point will end up'));
    root.append(el('div', { class: 'lab-toolbar' }, this.viewSeg, chips));
    this.stage = el('div', { class: 'lab-stage' });
    this.canvas = el('canvas', { 'aria-label': 'Points in the plane: generated samples, the noising process or the denoising field' });
    this.stage.append(this.canvas);
    this.playBtn = el('button', { class: 'icon-btn', type: 'button', 'aria-label': 'Play the reverse process', onclick: () => this.playScrub() }, '▶');
    this.slider = el('input', { type: 'range', min: 0, max: 100, step: 1, value: 100, 'aria-label': 'Noise step' });
    this.slider.addEventListener('input', () => this.onSlider());
    this.sliderLab = el('span', { class: 'mono df-t' });
    root.append(this.stage, el('div', { class: 'df-scrub' }, this.playBtn, this.slider, this.sliderLab), el('div', { class: 'lab-foot' },
      el('span', { class: 'lg-item' }, el('i', { class: 'dot', style: { background: 'var(--pos)' } }), 'sample'),
      el('span', { class: 'lg-item' }, el('i', { class: 'dot', style: { background: 'var(--muted)' } }), 'training data'),
      el('span', { class: 'lg-item', text: 'Denoiser view: click to drop a point and follow it to the data' })));
    this.canvas.addEventListener('click', (e) => this.click(e));
    this.resizeObs = new ResizeObserver(() => { this.dirty = true; });
    this.resizeObs.observe(this.stage);
    this.syncSlider();
  };
  DF.syncSlider = function () {
    const T = this.model.T;
    if (this.view === 'generate') { this.slider.max = this.gen.ts.length - 1; this.slider.value = this.k; this.sliderLab.textContent = `step ${this.k}/${this.gen.ts.length - 1} · t = ${this.gen.ts[this.k]}`; }
    else { this.slider.max = T; this.slider.value = T - this.tView; this.sliderLab.textContent = `t = ${this.tView} · noise ${fmt(Math.sqrt(1 - this.model.sched.abar[this.tView]), 2)}`; }
    this.playBtn.textContent = this.animating ? '❚❚' : '▶';
  };
  DF.onSlider = function () {
    const v = +this.slider.value;
    this.animating = false;
    if (this.view === 'generate') this.k = Math.min(v, this.gen.xs.length - 1); else { this.tView = this.model.T - v; this.probes = this.probes.map((p) => this.followProbe(p.x, p.y)); }
    this.syncSlider(); this.dirty = true; this.leftDirty = true;
  };
  DF.playScrub = function () {
    if (this.view !== 'generate') { this.view = 'generate'; this.viewSeg.set('generate'); }
    if (this.animating) { this.animating = false; this.syncSlider(); return; }
    if (this.k >= this.gen.ts.length - 1) this.k = 0;
    this.animating = true; this.syncSlider();
  };
  // A point dropped in the denoiser view follows deterministic (DDIM) steps from the shown t down to 0.
  DF.followProbe = function (x, y) {
    const path = [[x, y]], xy = new Float32Array([x, y]), ts = [];
    for (let t = this.tView; t >= 0; t -= Math.max(1, Math.round(this.tView / 40))) ts.push(t);
    if (ts[ts.length - 1] !== 0) ts.push(0);
    for (let i = 0; i + 1 < ts.length; i++) { this.model.reverseStep(xy, 1, ts[i], true, null, ts[i + 1]); path.push([xy[0], xy[1]]); }
    return { x, y, path };
  };
  DF.click = function (e) {
    if (this.view !== 'field' || this.tView === 0) return;
    const r = this.canvas.getBoundingClientRect(), L = this.lay, px = e.clientX - r.left, py = e.clientY - r.top;
    const x = ((px - L.cx) / L.s), y = -((py - L.cy) / L.s);
    this.probes.push(this.followProbe(x, y));
    if (this.probes.length > 8) this.probes.shift();
    this.dirty = true;
  };

  DF.draw = function () {
    const W = Math.max(200, this.stage.clientWidth), Hh = Math.max(200, this.stage.clientHeight), ctx = NA.fitCanvas(this.canvas, W, Hh);
    const side = Math.min(W, Hh) - 16, s = side / (2 * RANGE), cx = W / 2, cy = Hh / 2;
    this.lay = { cx, cy, s };
    const X = (x) => cx + x * s, Y = (y) => cy - y * s;
    ctx.clearRect(0, 0, W, Hh);
    ctx.fillStyle = theme.c('panel-2'); rr(ctx, cx - side / 2, cy - side / 2, side, side, 10); ctx.fill();
    ctx.strokeStyle = theme.c('line'); ctx.lineWidth = 1;
    for (let g = -RANGE + 1; g < RANGE; g++) { ctx.beginPath(); ctx.moveTo(X(g), cy - side / 2); ctx.lineTo(X(g), cy + side / 2); ctx.moveTo(cx - side / 2, Y(g)); ctx.lineTo(cx + side / 2, Y(g)); ctx.stroke(); }
    ctx.save(); ctx.beginPath(); rr(ctx, cx - side / 2, cy - side / 2, side, side, 10); ctx.clip();
    const D = this.D;
    if (this.show.data || this.view === 'forward') {
      ctx.fillStyle = theme.c('muted', this.view === 'forward' ? 0.22 : 0.4);
      for (let i = 0; i < D.n; i++) ctx.fillRect(X(D.pts[2 * i]) - 0.8, Y(D.pts[2 * i + 1]) - 0.8, 1.6, 1.6);
    }
    const dots = (xy, n, color, r) => { ctx.fillStyle = color; for (let i = 0; i < n; i++) { ctx.beginPath(); ctx.arc(X(xy[2 * i]), Y(xy[2 * i + 1]), r, 0, Math.PI * 2); ctx.fill(); } };
    if (this.view === 'generate') {
      const g = this.gen, k = this.k;
      if (this.show.trails && k > 0) {
        // DDPM paths are jittery, so they get short comet tails; deterministic DDIM paths are drawn whole
        const from = this.sampler === 'ddim' ? 0 : Math.max(0, k - 10);
        ctx.lineWidth = 1.2; ctx.lineJoin = 'round';
        for (let i = 0; i < Math.min(40, g.n); i++) {
          ctx.strokeStyle = theme.series[i % 8]; ctx.globalAlpha = 0.55;
          ctx.beginPath();
          for (let j = from; j <= k; j++) { const x = X(g.xs[j][2 * i]), y = Y(g.xs[j][2 * i + 1]); j > from ? ctx.lineTo(x, y) : ctx.moveTo(x, y); }
          ctx.stroke();
        }
        ctx.globalAlpha = 1;
      }
      if (this.show.x0 && g.x0s[k]) dots(g.x0s[k], g.n, theme.c('grad', 0.45), 1.6);
      dots(g.xs[k], g.n, theme.c('pos', 0.8), 2);
    } else if (this.view === 'forward') {
      const t = this.tView, xt = this.model.noised(D.pts, this.eps, D.n, t);
      dots(xt, D.n, theme.c('pos', 0.7), 1.7);
    } else {
      this.drawField(ctx, X, Y, side);
    }
    ctx.restore();
    // corner label
    ctx.font = '600 11px ' + (getComputedStyle(document.body).getPropertyValue('--font-ui') || 'sans-serif'); ctx.fillStyle = theme.c('muted'); ctx.textAlign = 'left'; ctx.textBaseline = 'top';
    const t = this.currentT(), lab = this.view === 'generate' ? (t === this.model.T ? 'Pure noise' : t === 0 ? 'Generated samples' : `Denoising… t = ${t}`) : this.view === 'forward' ? `Training data with noise, t = ${t}` : `Denoising field at t = ${t}`;
    ctx.fillText(lab.toUpperCase(), cx - side / 2 + 10, cy - side / 2 + 10);
    this.syncSlider();
  };
  // Arrows from a grid of points toward the network's guess of their clean version x̂₀.
  DF.drawField = function (ctx, X, Y, side) {
    const t = Math.max(1, this.tView), G = side > 420 ? 21 : 15, n = G * G, xy = new Float32Array(2 * n);
    for (let i = 0; i < G; i++) for (let j = 0; j < G; j++) { const p = i * G + j; xy[2 * p] = -RANGE + ((j + 0.5) / G) * 2 * RANGE; xy[2 * p + 1] = -RANGE + ((i + 0.5) / G) * 2 * RANGE; }
    const eps = this.model.predict(xy, n, t), ab = this.model.sched.abar[t], cell = (2 * RANGE) / G;
    ctx.lineWidth = 1.3;
    for (let p = 0; p < n; p++) {
      const x = xy[2 * p], y = xy[2 * p + 1];
      const x0 = (x - Math.sqrt(1 - ab) * eps[2 * p]) / Math.sqrt(ab), y0 = (y - Math.sqrt(1 - ab) * eps[2 * p + 1]) / Math.sqrt(ab);
      let dx = x0 - x, dy = y0 - y;
      const len = Math.hypot(dx, dy), max = cell * 0.85, k = len > max ? max / len : 1;
      dx *= k; dy *= k;
      const a = clamp(len / 2.5, 0.25, 1);
      ctx.strokeStyle = theme.c('grad', a); ctx.fillStyle = theme.c('grad', a);
      const x1 = X(x), y1 = Y(y), x2 = X(x + dx), y2 = Y(y + dy);
      ctx.beginPath(); ctx.moveTo(x1, y1); ctx.lineTo(x2, y2); ctx.stroke();
      const ang = Math.atan2(y2 - y1, x2 - x1);
      ctx.beginPath(); ctx.moveTo(x2, y2); ctx.lineTo(x2 - 4 * Math.cos(ang - 0.45), y2 - 4 * Math.sin(ang - 0.45)); ctx.lineTo(x2 - 4 * Math.cos(ang + 0.45), y2 - 4 * Math.sin(ang + 0.45)); ctx.fill();
    }
    for (const pr of this.probes) {
      ctx.strokeStyle = theme.c('ink'); ctx.lineWidth = 2; ctx.beginPath();
      pr.path.forEach(([x, y], i) => (i ? ctx.lineTo(X(x), Y(y)) : ctx.moveTo(X(x), Y(y))));
      ctx.stroke();
      ctx.fillStyle = theme.c('ink'); ctx.beginPath(); ctx.arc(X(pr.x), Y(pr.y), 4, 0, Math.PI * 2); ctx.fill();
      const [ex, ey] = pr.path[pr.path.length - 1];
      ctx.fillStyle = theme.c('pos'); ctx.beginPath(); ctx.arc(X(ex), Y(ey), 4.5, 0, Math.PI * 2); ctx.fill();
    }
  };

  // ------------------------------------------------------------------ right: inspector
  DF.buildTab = function (id, body) {
    this.tabId = id; const P = (this.parts = {});
    if (id === 'overview') {
      body.append(UI.modelCard(this.preset, (t) => this.runTry(t)));
      P.findings = el('div', { class: 'diag-list' });
      body.append(UI.section('How good are the samples?', UI.note('Distances are in standard deviations of the data. As a yardstick, two halves of the real data sit this far from each other too.'), P.findings));
    } else if (id === 'train') {
      P.tiles = el('div', { class: 'tiles' });
      const c = el('canvas'); P.loss = new NA.charts.LineChart(c, { height: 140, format: (v) => fmt(v, 4) });
      P.bands = el('canvas');
      body.append(P.tiles,
        UI.section('Loss', el('div', { class: 'chart-box' }, c), UI.note('Mean squared error between the true noise and the predicted noise, smoothed. It never reaches 0: when only a little noise was added, nobody can tell exactly which part of a point is noise and which is data.')),
        UI.section('Loss by noise level', el('div', { class: 'chart-box' }, P.bands), UI.note('Average loss in ten bands of t. At high noise a point is almost all noise, so ε is easy to guess and the loss is near 0. It is hardest at low-to-middle noise, where data and noise are mixed about evenly.')),
        UI.section('Experiment', UI.note('Training draws a fresh batch of data points, noise levels and noise every step.'), el('div', { class: 'btn-row' },
          el('button', { class: 'btn btn-primary', type: 'button', onclick: () => { this.reset(); this.play(); } }, 'Train from scratch'),
          el('button', { class: 'btn', type: 'button', onclick: () => this.fastForward(1000) }, 'Train 1,000 more steps'))));
    } else if (id === 'sched') {
      P.canvas = el('canvas');
      body.append(UI.section('Cosine noise schedule', el('div', { class: 'chart-box' }, P.canvas), el('div', { class: 'mini-legend' },
        el('span', { class: 'lg-item' }, el('i', { class: 'key-line', style: { background: 'var(--pos)' } }), 'ᾱ_t: share of the signal’s variance left'),
        el('span', { class: 'lg-item' }, el('i', { class: 'key-line', style: { background: 'var(--grad)' } }), 'β_t: noise added at step t')),
        UI.note(`${this.model.T} steps take the data to almost pure Gaussian noise (ᾱ = ${fmt(this.model.sched.abar[this.model.T], 4)} at the end). The cosine schedule adds noise slowly at first, so early steps keep the shape recognisable, and faster later.`)));
    } else if (id === 'net') {
      P.arch = el('div', { class: 'kv-grid' });
      P.canvas = el('canvas');
      body.append(UI.section('The denoiser', P.arch), UI.section('How it knows the time', UI.note('The network gets t as the number itself and as sines and cosines at four frequencies, so it can behave very differently at high and low noise.'), el('div', { class: 'chart-box' }, P.canvas)));
    }
    this.updateTab();
  };
  DF.updateTab = function () {
    const id = this.tabId, P = this.parts;
    if (!P) return;
    if (id === 'overview') {
      const q = this.measure(), items = [];
      if (!q) { clear(P.findings).append(UI.note('Generating samples…')); this.inspDirty = true; return; }
      const f = (level, title, text) => items.push(el('div', { class: 'diag diag-' + level }, el('div', { class: 'diag-head' }, UI.pill(level, { good: 'Good', info: 'Note', warn: 'Watch' }[level]), el('h4', { class: 'diag-title', text: title })), el('p', { class: 'diag-body', text })));
      const ok = q.precision < q.ref.precision * 2.5;
      f(ok ? 'good' : 'warn', `Precision: ${fmt(q.precision, 3)} from the data`, `Each generated point is on average ${fmt(q.precision, 3)} from the nearest real point (real data: ${fmt(q.ref.precision, 3)}). ${ok ? 'The samples sit on the shape.' : 'Many samples fall between the arms: train longer or sample with more steps.'}`);
      f(q.coverage < q.ref.coverage * 2.5 ? 'good' : 'warn', `Coverage: ${fmt(q.coverage, 3)}`, `Each real point is on average ${fmt(q.coverage, 3)} from the nearest sample (real data: ${fmt(q.ref.coverage, 3)}). Low coverage would mean the model ignores part of the shape.`);
      f('info', `${this.gen.ts.length - 1} steps, ${this.sampler === 'ddim' ? 'DDIM' : 'DDPM'}`, this.sampler === 'ddim' ? 'Deterministic sampling: no noise is added on the way back, so each starting point always lands in the same place.' : 'Stochastic sampling: a little fresh noise is added at every step, which helps the samples spread over the whole shape.');
      clear(P.findings).append(...items);
    } else if (id === 'train') {
      const last = this.history[this.history.length - 1];
      clear(P.tiles).append(UI.tile('Step', fmtInt(this.step), this.pretrained ? 'trained offline, same code' : null), UI.tile('Loss', fmt(this.runLoss ?? (last && last.loss), 4)), UI.tile('Parameters', fmtInt(this.model.net.paramCount)), UI.tile('Batch', `${this.batch} points`));
      P.loss.frame(); P.loss.set({ series: [{ name: 'loss', color: theme.c('pos'), points: this.history.map((q) => [q.step, q.loss]) }] });
      this.drawBands(P.bands);
    } else if (id === 'sched') this.drawSchedule(P.canvas);
    else if (id === 'net') this.renderNet(P);
  };
  DF.drawBands = function (canvas) {
    const b = this.lossBands(), W = Math.max(220, canvas.parentElement.clientWidth), Hh = 130, ctx = NA.fitCanvas(canvas, W, Hh), m = Math.max(...b, 1e-6);
    ctx.clearRect(0, 0, W, Hh);
    const bw = (W - 40) / 10, mono = getComputedStyle(document.body).getPropertyValue('--font-mono') || 'monospace';
    ctx.font = `500 10px ${mono}`; ctx.textAlign = 'center';
    b.forEach((v, i) => {
      const h = (v / m) * (Hh - 34), x = 30 + i * bw;
      ctx.fillStyle = theme.c('pos', 0.75); NA.rr(ctx, x + 2, Hh - 18 - h, bw - 4, h, 3); ctx.fill();
      ctx.fillStyle = theme.c('muted'); ctx.textBaseline = 'top'; ctx.fillText(`${i * 10}`, x + bw / 2, Hh - 14);
      ctx.textBaseline = 'bottom'; ctx.fillStyle = theme.c('ink-2'); ctx.fillText(fmt(v, 2), x + bw / 2, Hh - 20 - h);
    });
    ctx.textAlign = 'left'; ctx.textBaseline = 'top'; ctx.fillStyle = theme.c('muted'); ctx.fillText('t →', 0, Hh - 14);
  };
  DF.drawSchedule = function (canvas) {
    const S = this.model.sched, T = this.model.T, W = Math.max(220, canvas.parentElement.clientWidth), Hh = 150, ctx = NA.fitCanvas(canvas, W, Hh);
    ctx.clearRect(0, 0, W, Hh);
    const X = (t) => 30 + (t / T) * (W - 40), Y = (v) => 8 + (1 - v) * (Hh - 28);
    ctx.strokeStyle = theme.c('line'); ctx.strokeRect(30.5, 8.5, W - 40, Hh - 28);
    const curve = (fn, col) => { ctx.strokeStyle = col; ctx.lineWidth = 2; ctx.beginPath(); for (let t = 1; t <= T; t++) { const x = X(t), y = Y(fn(t)); t > 1 ? ctx.lineTo(x, y) : ctx.moveTo(x, y); } ctx.stroke(); };
    curve((t) => S.abar[t], theme.c('pos'));
    curve((t) => S.beta[t] / 0.5, theme.c('grad'));
    const mono = getComputedStyle(document.body).getPropertyValue('--font-mono') || 'monospace';
    ctx.font = `500 10px ${mono}`; ctx.fillStyle = theme.c('muted'); ctx.textAlign = 'right'; ctx.textBaseline = 'middle';
    ctx.fillText('1', 26, Y(1)); ctx.fillText('0', 26, Y(0));
    ctx.textAlign = 'center'; ctx.textBaseline = 'top'; ctx.fillText('t', X(T / 2), Hh - 14);
    ctx.textAlign = 'right'; ctx.fillText('β scaled: 0.5 at the top', W - 12, 12);
  };
  DF.renderNet = function (P) {
    const net = this.model.net;
    clear(P.arch).append(...[
      ['Inputs', `${DM.IN_DIM}: x, y, t/T and 8 sine and cosine features of t`], ['Hidden', `${net.sizes.slice(1, -1).join(' → ')} SiLU`], ['Output', '2: the predicted noise ε̂ for x and y'], ['Parameters', fmtInt(net.paramCount)],
    ].map(([k, v]) => el('div', { class: 'kv' }, el('span', { class: 'kv-k', text: k }), el('span', { class: 'kv-v', text: v }))));
    const W = Math.max(220, P.canvas.parentElement.clientWidth), Hh = 120, ctx = NA.fitCanvas(P.canvas, W, Hh), T = this.model.T, X = new Float32Array(DM.IN_DIM * (T + 1));
    ctx.clearRect(0, 0, W, Hh);
    ctx.strokeStyle = theme.c('line'); ctx.beginPath(); ctx.moveTo(0, Hh / 2); ctx.lineTo(W, Hh / 2); ctx.stroke();
    for (let f = 2; f < DM.IN_DIM; f++) {
      ctx.strokeStyle = theme.series[(f - 2) % 8]; ctx.lineWidth = 1.4; ctx.beginPath();
      for (let t = 0; t <= T; t++) {
        const tau = t / T, v = f === 2 ? tau * 2 - 1 : (f - 3) % 2 === 0 ? Math.sin(Math.PI * Math.pow(2, (f - 3) >> 1) * tau) : Math.cos(Math.PI * Math.pow(2, (f - 3) >> 1) * tau);
        const x = (t / T) * W, y = Hh / 2 - v * (Hh / 2 - 6); t ? ctx.lineTo(x, y) : ctx.moveTo(x, y);
      }
      ctx.stroke();
    }
    void X;
  };

  DF.runTry = function (t) {
    if (t.action === 'ddim') { this.sampler = 'ddim'; this.samplerSeg.set('ddim'); this.view = 'generate'; this.viewSeg.set('generate'); this.generate(false); NA.ui.toast.show('DDIM: no fresh noise on the way back. Trails become smooth curves.'); }
    else if (t.action === 'fewSteps') { this.steps = 10; this.view = 'generate'; this.viewSeg.set('generate'); this.generate(false); NA.ui.toast.show('10 big steps instead of 100: faster, but blurrier.'); }
    else if (t.action === 'field') { this.view = 'field'; this.tView = 60; this.viewSeg.set('field'); this.probes = [this.followProbe(2.4, 2.4), this.followProbe(-2.5, 0.3), this.followProbe(0.4, -2.6)]; this.syncSlider(); NA.ui.toast.show('Three points dropped at t = 60. Drag the slider to change the noise level; click to drop more.'); }
    else if (t.action === 'moons') { this.pause(); this.load('moons', true); NA.$$('.ds-tile', this.dsTiles).forEach((x) => x.setAttribute('aria-checked', String(x._id === 'moons'))); this.tabs.select('train'); this.play(); NA.ui.toast.show('Training a fresh denoiser on two moons. Samples refresh every 600 steps.'); return; }
    if (NA.mobile && NA.mobile.mobile) NA.mobile.go('network');
    this.dirty = true; this.inspDirty = true;
  };

  // Zoo glyph: noise on the left turning into a curve on the right.
  DF.glyph = function (ctx, W, Hh) {
    const rng = new NA.RNG(3);
    for (let i = 0; i < 70; i++) {
      const u = i / 70, x = 6 + u * (W - 12), tgt = Hh / 2 + Math.sin(u * 7) * (Hh / 3.2), y = tgt + (1 - u) * rng.gauss(0, Hh / 4);
      ctx.fillStyle = theme.c(u > 0.55 ? 'pos' : 'ink-2', 0.4 + 0.6 * u); ctx.fillRect(x, y, 1.8, 1.8);
    }
  };

  H.register(DF);
  NA.diffusionLab = DF;
})(window.NA);
