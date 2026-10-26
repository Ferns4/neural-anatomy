/* app.js — application state, the training loop, training history and shared analysis caches.
   Views read everything through App and re-render on its events. */
(function (NA) {
  'use strict';
  const { MLP } = NA.engine;
  const D = NA.data;

  const DEFAULT_LENSES = { edges: 'weight', nodes: 'activation', maps: true, values: false, bias: false, flow: false, health: false, acts: true };
  const DEFAULT_OVERLAYS = { train: true, val: true, test: false, errors: false, neuronLines: false, contributions: false, saliency: false };

  const App = {
    ev: NA.Emitter(),
    on(type, fn) { return this.ev.on(type, fn); },
    emit(type, payload) { this.ev.emit(type, payload); },

    preset: null, spec: null, arch: null, hyper: null,
    lenses: Object.assign({}, DEFAULT_LENSES),
    overlays: Object.assign({}, DEFAULT_OVERLAYS),
    speed: 'normal',
    ds: null, net: null,
    step: 0, training: false, ffTarget: null, diverged: false,
    version: 0, probeVersion: 0,
    probe: null, pinned: null,
    selected: null, hover: null,
    history: null, viewing: null, liveParams: null,
    caches: {},
    components: [],
  };

  App.register = function (c) { this.components.push(c); return c; };

  // ------------------------------------------------------------------ naming helpers
  App.isPlane = () => App.ds && App.ds.kind === 'plane';
  App.isLine = () => App.ds && App.ds.kind === 'line';
  App.isImage = () => App.ds && App.ds.kind === 'image';
  App.isRegression = () => App.ds && App.ds.task === 'regression';
  App.outLayer = () => App.net.L;

  App.layerName = function (l, short) {
    if (l === 0) return 'Input';
    if (l === this.net.L) return 'Output';
    return short ? 'H' + l : 'Hidden ' + l;
  };
  App.className = function (k) {
    if (this.isImage()) return 'Digit ' + k;
    if (this.ds.classes === 2) return k ? 'Blue (class 1)' : 'Orange (class 0)';
    return ['Orange', 'Blue', 'Green'][k] + ' (class ' + k + ')';
  };
  App.neuronName = function (l, j, short) {
    const ds = this.ds;
    if (l === 0) {
      if (ds.kind === 'image') return `pixel (${j % ds.imgSize}, ${Math.floor(j / ds.imgSize)})`;
      return ds.featureLabels[j];
    }
    if (l === this.net.L) {
      if (ds.task === 'regression') return 'ŷ';
      if (ds.task === 'binary') return short ? 'Out' : 'Output';
      return this.isImage() ? (short ? String(j) : 'Digit ' + j) : (short ? 'C' + j : 'Class ' + j);
    }
    return short ? `H${l}·${j + 1}` : `Hidden ${l} · neuron ${j + 1}`;
  };
  App.validNeuron = function (s) { return s && this.net && s.l >= 0 && s.l <= this.net.L && s.j >= 0 && s.j < this.net.sizes[s.l]; };
  App.defaultSelection = function () { return { l: 1, j: 0 }; };
  App.epoch = function (step) { return ((step === undefined ? this.step : step) * Math.min(this.hyper.batch, this.ds.trainIdx.length)) / this.ds.trainIdx.length; };

  // ------------------------------------------------------------------ model construction
  App.outputSpec = function (ds) {
    if (ds.task === 'regression') return { n: 1, out: 'identity' };
    if (ds.task === 'binary') return { n: 1, out: 'sigmoid' };
    return { n: ds.classes, out: 'softmax' };
  };

  // Per-neuron activation ids for hidden layer h, padded with the default where none was chosen.
  App.layerActs = function (h) {
    const n = this.arch.hidden[h], src = (this.arch.acts && this.arch.acts[h]) || [];
    const out = [];
    for (let j = 0; j < n; j++) out.push(src[j] || this.arch.act);
    return out;
  };

  App.rebuild = function (opts = {}) {
    const s = this.spec;
    if (opts.data || !this.ds) this.ds = D.buildDataset(s);
    else D.setFeatures(this.ds, s.features);
    const ds = this.ds, o = this.outputSpec(ds);
    const acts = [null, ...this.arch.hidden.map((n, h) => this.layerActs(h)), null];
    this.net = new MLP({ sizes: [ds.dim, ...this.arch.hidden, o.n], act: this.arch.act, acts, out: o.out, loss: this.hyper.loss, init: this.hyper.init, seed: this.hyper.seed });
    this.hyper.loss = this.net.lossId;
    this.rng = new NA.RNG((this.hyper.seed * 31 + 7) >>> 0);
    this.order = Int32Array.from(ds.trainIdx);
    this.rng.shuffle(this.order);
    this.batchPos = 0;
    const cap = ds.kind === 'image' ? 400 : 600;
    this.evalTrain = ds.trainIdx.length > cap ? ds.trainIdx.slice(0, cap) : ds.trainIdx;
    this.evalVal = ds.valIdx.length > cap ? ds.valIdx.slice(0, cap) : ds.valIdx;
    this.testResult = null;
    this.step = 0; this.version++; this.caches = {}; this._pt = null;
    this.training = false; this.ffTarget = null; this.diverged = false;
    this.history = { points: [], snaps: [], pointEvery: ds.kind === 'image' ? 8 : 2, snapEvery: ds.kind === 'image' ? 10 : 5, lastPointStep: 0, lastSnapStep: 0 };
    this.viewing = null; this.liveParams = null;
    this.recordPoint();
    this.snapshot();
    if (!this.validNeuron(this.selected) || this.selected.l === 0 && !opts.keepSelection) this.selected = this.defaultSelection();
    if (!this.validNeuron(this.selected)) this.selected = this.defaultSelection();
    this.hover = null;
    this.resetProbe(!opts.data);
    this.emit('model');
    this.emit('training');
    this.emit('history');
  };

  // sync: train before the first paint (used at start-up so the page opens on a trained model)
  App.loadPreset = function (id, opts = {}) {
    const p = NA.presets.byId[id] || NA.presets.PRESETS[0];
    const copy = (o) => JSON.parse(JSON.stringify(o));
    this.preset = p;
    this.spec = Object.assign({ features: ['x1', 'x2'] }, copy(p.data));
    this.arch = copy(p.arch);
    this.hyper = copy(p.hyper);
    this.lenses = Object.assign({}, DEFAULT_LENSES, p.lenses);
    if (NA.prefersReducedMotion()) this.lenses.flow = false;
    this.overlays = Object.assign({}, DEFAULT_OVERLAYS, p.overlays);
    this.selected = null;
    this.rebuild({ data: true });
    this.emit('preset', p);
    this.emit('lens');
    this.emit('hyper');
    if (opts.sync) this.trainNow(p.pretrain);
    else this.fastForward(p.pretrain);
    NA.store.set('preset', p.id);
  };
  App.trainNow = function (n) {
    this.trainSteps(n);
    this.recordPoint();
    this.snapshot();
    this.emit('weights'); this.emit('history'); this.emit('training');
  };

  /* Apply a "try this" change or a control change. Structural changes rebuild the network. */
  App.applyPatch = function (p) {
    let data = false, model = false;
    const sp = this.spec, h = this.hyper;
    if (p.dataset !== undefined && p.dataset !== sp.dataset) {
      const kind = D.DATASETS[p.dataset].kind;
      if (kind !== this.ds.kind) sp.features = kind === 'plane' ? ['x1', 'x2'] : [];
      sp.dataset = p.dataset; data = true;
    }
    for (const k of ['noise', 'n', 'trainRatio']) if (p[k] !== undefined && p[k] !== sp[k]) { sp[k] = p[k]; data = true; }
    if (p.dataSeed !== undefined) { sp.seed = p.dataSeed; data = true; }
    if (p.features) { sp.features = p.features.slice(); model = true; }
    if (p.hidden) {
      // keep each surviving neuron's activation when layer sizes change
      const old = this.arch.hidden.map((n, h) => this.layerActs(h));
      this.arch.hidden = p.hidden.slice();
      this.arch.acts = this.arch.hidden.map((n, h) => (old[h] ? old[h].slice(0, n) : null));
      model = true;
    }
    if (p.acts) this.arch.acts = p.acts.map((a) => (a ? a.slice() : null));
    if (p.act && (p.act !== this.arch.act || (this.net && !this.net.isUniform()))) { this.arch.act = p.act; this.arch.acts = null; model = true; }
    if (p.loss) { h.loss = p.loss; if (this.net) { this.net.setLoss(p.loss); h.loss = this.net.lossId; this.version++; this.caches = {}; this._pt = null; } }
    if (p.seed !== undefined) { h.seed = p.seed; model = true; }
    if (p.init && p.init !== h.init) { h.init = p.init; model = true; }
    for (const k of ['lr', 'optimizer', 'batch', 'reg', 'regRate']) if (p[k] !== undefined) h[k] = p[k];
    if (p.optimizer !== undefined && this.net) this.net.resetOptimizer(p.optimizer);
    if (data || model || p.reset) this.rebuild({ data, keepSelection: true });
    else if (p.loss) { this.recordPoint(); this.emit('weights'); this.emit('history'); }
    this.emit('hyper');
    if (p.train) this.fastForward(p.train);
    if (p.tab) this.emit('tab', p.tab);
    if (p.view) this.emit('view', p.view);
    if (p.sampleDigit !== undefined) this.emit('sampleDigit', p.sampleDigit);
  };

  // ------------------------------------------------------------------ architecture edits
  const actLabel = (id) => (NA.engine.ACTIVATIONS[id] || {}).label || id;
  App.toast = function (msg) { this.emit('toast', msg); };

  // Change one neuron's activation in place: weights are kept so the effect shows immediately.
  App.setNeuronAct = function (l, j, id) {
    if (l <= 0 || l >= this.net.L) return;
    const h = l - 1;
    if (!this.arch.acts) this.arch.acts = [];
    const acts = this.layerActs(h);
    acts[j] = id;
    this.arch.acts[h] = acts;
    this.net.setAct(l, j, id);
    this.afterActChange();
    this.toast(`${this.neuronName(l, j)} now uses ${actLabel(id)}. Weights kept.`);
  };
  App.setLayerAct = function (l, id) {
    if (l <= 0 || l >= this.net.L) return;
    const h = l - 1, n = this.arch.hidden[h];
    if (!this.arch.acts) this.arch.acts = [];
    this.arch.acts[h] = new Array(n).fill(id);
    for (let j = 0; j < n; j++) this.net.setAct(l, j, id);
    this.afterActChange();
    this.toast(`All ${n} neurons in ${this.layerName(l)} now use ${actLabel(id)}. Weights kept.`);
  };
  // Several neurons ({l, j} list) switched to one activation, e.g. dead ReLUs to Leaky ReLU.
  App.setActs = function (list, id) {
    if (!list.length) return;
    if (!this.arch.acts) this.arch.acts = [];
    for (const { l, j } of list) {
      if (l <= 0 || l >= this.net.L) continue;
      const acts = this.layerActs(l - 1);
      acts[j] = id;
      this.arch.acts[l - 1] = acts;
      this.net.setAct(l, j, id);
    }
    this.afterActChange();
    this.toast(`${list.length === 1 ? this.neuronName(list[0].l, list[0].j) + ' now uses' : list.length + ' neurons now use'} ${actLabel(id)}. Weights kept.`);
  };
  App.afterActChange = function () {
    if (this.training) this.pause();
    this.version++; this.caches = {}; this._pt = null;
    this.recordPoint();
    this.emit('weights'); this.emit('arch'); this.emit('history'); this.emit('hyper');
  };
  App.setLoss = function (id) {
    this.applyPatch({ loss: id });
    const L = NA.engine.LOSSES[this.net.lossId];
    this.toast(`Loss is now ${L.label.toLowerCase()}. Training continues from the current weights.`);
  };
  // Insert a hidden layer after hidden index h (−1 = right after the input; default: before the output).
  App.addLayer = function (h) {
    const hidden = this.arch.hidden.slice();
    if (hidden.length >= 6) { this.toast('Six hidden layers is the maximum here.'); return; }
    const at = h === undefined ? hidden.length : h + 1;
    const size = hidden.length ? hidden[Math.max(0, Math.min(hidden.length - 1, at - 1))] : 4;
    const acts = hidden.map((n, k) => this.layerActs(k));
    const neighbour = acts[Math.max(0, Math.min(acts.length - 1, at - 1))];
    hidden.splice(at, 0, size);
    acts.splice(at, 0, new Array(size).fill(neighbour ? mostCommon(neighbour) : this.arch.act));
    this.applyPatch({ hidden, acts });
    this.toast(`Added Hidden ${at + 1} with ${size} neurons. Weights re-initialized.`);
  };
  App.removeLayer = function (h) {
    const hidden = this.arch.hidden.slice();
    if (!hidden.length) return;
    const at = h === undefined ? hidden.length - 1 : h;
    const acts = hidden.map((n, k) => this.layerActs(k));
    hidden.splice(at, 1); acts.splice(at, 1);
    this.applyPatch({ hidden, acts });
    this.toast(hidden.length ? `Removed Hidden ${at + 1}. Weights re-initialized.` : 'No hidden layers left: the network is now a single layer.');
  };
  App.resizeLayer = function (h, d) {
    const hidden = this.arch.hidden.slice(), max = this.isImage() ? 48 : 32;
    const next = Math.max(1, Math.min(max, hidden[h] + d));
    if (next === hidden[h]) return;
    const acts = hidden.map((n, k) => this.layerActs(k));
    if (d > 0) acts[h].push(mostCommon(acts[h])); else acts[h].pop();
    hidden[h] = next;
    this.applyPatch({ hidden, acts });
  };
  function mostCommon(ids) {
    const c = {};
    let best = ids[0];
    for (const id of ids) { c[id] = (c[id] || 0) + 1; if (c[id] > c[best]) best = id; }
    return best;
  }

  // ------------------------------------------------------------------ training
  App.trainSteps = function (k) {
    const ds = this.ds, net = this.net, h = this.hyper, H = this.history;
    const bs = Math.max(1, Math.min(h.batch, this.order.length));
    for (let s = 0; s < k; s++) {
      if (this.batchPos + bs > this.order.length) { this.rng.shuffle(this.order); this.batchPos = 0; }
      const loss = net.trainBatch(ds.X, ds.y, this.order, this.batchPos, this.batchPos + bs, h);
      this.batchPos += bs;
      this.step++;
      this.batchLoss = loss;
      if (!isFinite(loss) || !isFinite(net.gradRms[0])) { this.diverged = true; this.recordPoint(); this.pause(); break; }
      if (this.step - H.lastPointStep >= H.pointEvery) this.recordPoint();
      if (this.step - H.lastSnapStep >= H.snapEvery) this.snapshot();
      if (this.ffTarget !== null && this.step >= this.ffTarget) break;
    }
    this.version++;
  };

  App.recordPoint = function () {
    const net = this.net, ds = this.ds, H = this.history;
    const tr = net.evaluate(ds.X, ds.y, this.evalTrain, ds.task);
    const va = net.evaluate(ds.X, ds.y, this.evalVal, ds.task);
    H.points.push({ step: this.step, epoch: this.epoch(), trainLoss: tr.loss, valLoss: va.loss, trainAcc: tr.acc, valAcc: va.acc, grad: Array.from(net.gradRms), ratio: Array.from(net.updateRatio), reg: net.regPenalty(this.hyper.reg, this.hyper.regRate) });
    if (H.points.length > 360) {
      const last = H.points.length - 1;
      H.points = H.points.filter((pt, i) => i % 2 === 0 || i === last);
      H.pointEvery *= 2;
    }
    H.lastPointStep = this.step;
  };

  App.snapshot = function () {
    const H = this.history;
    H.snaps.push({ step: this.step, params: this.net.getParams() });
    if (H.snaps.length > 160) {
      const last = H.snaps.length - 1;
      H.snaps = H.snaps.filter((s, i) => i % 2 === 0 || i === last);
      H.snapEvery *= 2;
    }
    H.lastSnapStep = this.step;
  };

  App.metrics = function () {
    const pts = this.history.points;
    if (this.viewing !== null) {
      const st = this.history.snaps[this.viewing].step;
      let best = pts[0];
      for (const p of pts) if (p.step <= st) best = p;
      return best;
    }
    return pts[pts.length - 1];
  };
  App.displayStep = function () { return this.viewing !== null ? this.history.snaps[this.viewing].step : this.step; };

  App.play = function () {
    if (this.viewing !== null) this.resumeFromView();
    if (this.diverged) return;
    this.training = true; this.emit('training');
  };
  App.pause = function () {
    const was = this.training;
    this.training = false; this.ffTarget = null;
    if (was) { this.recordPoint(); this.snapshot(); this.emit('history'); }
    this.emit('training');
    this.emit('weights');
  };
  App.toggle = function () { if (this.training) this.pause(); else this.play(); };
  App.stepOnce = function () {
    if (this.viewing !== null) this.resumeFromView();
    if (this.training) this.pause();
    this.trainSteps(1);
    this.recordPoint();
    this.emit('weights'); this.emit('history');
  };
  App.fastForward = function (n) {
    if (!n) return;
    if (this.viewing !== null) this.resumeFromView();
    this.diverged = false;
    this.ffTarget = this.step + n;
    this.training = true;
    this.emit('training');
  };
  App.reset = function () { this.rebuild({ keepSelection: true }); };

  // Steps per animation frame for each speed; 'max' and fast-forward fill a time budget instead.
  const SPEED_STEPS = { normal: 2, fast: 16 };
  App.frame = function (now) {
    if (this.training) {
      const ff = this.ffTarget !== null;
      if (!ff && this.speed === 'slow') {
        if (now - (this._lastSlow || 0) > 160) { this.trainSteps(1); this._lastSlow = now; }
      } else if (!ff && SPEED_STEPS[this.speed]) {
        const t0 = performance.now();
        for (let k = 0; k < SPEED_STEPS[this.speed] && this.training && performance.now() - t0 < 12; k++) this.trainSteps(1);
      } else {
        const budget = ff ? 24 : 11, t0 = performance.now(), chunk = this.ds.kind === 'image' ? 2 : 4;
        let guard = 0;
        do { this.trainSteps(chunk); guard++; }
        while (this.training && performance.now() - t0 < budget && guard < 400 && !(ff && this.step >= this.ffTarget));
      }
      if (this.ffTarget !== null && this.step >= this.ffTarget) { this.pause(); this.emit('ffdone'); }
      this.emit('weights');
    }
    for (const c of this.components) if (c.frame) c.frame(now);
    requestAnimationFrame((t) => this.frame(t));
  };

  // ------------------------------------------------------------------ history (time travel)
  App.viewSnapshot = function (i) {
    const H = this.history;
    if (!H.snaps.length) return;
    if (this.training) this.pause();
    i = Math.max(0, Math.min(H.snaps.length - 1, i));
    if (i === H.snaps.length - 1 && this.viewing === null) return;
    if (this.viewing === null) this.liveParams = this.net.getParams();
    this.viewing = i;
    this.net.setParams(H.snaps[i].params);
    this.version++;
    this.emit('weights'); this.emit('history');
  };
  App.backToLive = function () {
    if (this.viewing === null) return;
    this.net.setParams(this.liveParams);
    this.viewing = null; this.liveParams = null;
    this.version++;
    this.emit('weights'); this.emit('history');
  };
  App.resumeFromView = function () {
    if (this.viewing === null) return;
    const H = this.history, snap = H.snaps[this.viewing];
    H.snaps = H.snaps.slice(0, this.viewing + 1);
    H.points = H.points.filter((p) => p.step <= snap.step);
    this.step = snap.step;
    this.viewing = null; this.liveParams = null;
    this.net.resetOptimizer(this.hyper.optimizer);
    H.lastPointStep = H.lastSnapStep = snap.step;
    this.diverged = false;
    this.emit('history');
  };

  // ------------------------------------------------------------------ direct manipulation
  App.weightsEdited = function () {
    if (this.training) this.pause();
    this.version++;
    this.emit('weights');
  };
  App.setWeight = function (l, j, i, v) { this.net.W[l - 1][j * this.net.sizes[l - 1] + i] = v; this.weightsEdited(); };
  App.setBias = function (l, j, v) { this.net.b[l - 1][j] = v; this.weightsEdited(); };
  App.toggleAblate = function (l, j) {
    if (l >= this.net.L) return;
    const m = this.net.mask[l];
    m[j] = m[j] ? 0 : 1;
    this.version++;
    this.emit('weights'); this.emit('ablation');
  };
  App.ablatedCount = function () { let c = 0; for (let l = 0; l < this.net.L; l++) for (const v of this.net.mask[l]) if (!v) c++; return c; };
  App.restoreAll = function () { for (let l = 0; l < this.net.L; l++) this.net.mask[l].fill(1); this.version++; this.emit('weights'); this.emit('ablation'); };

  App.select = function (s, opts = {}) {
    if (!this.validNeuron(s)) return;
    this.selected = { l: s.l, j: s.j };
    this.emit('selection', opts);
  };
  App.setHover = function (s) {
    const same = (a, b) => (a === b) || (a && b && a.l === b.l && a.j === b.j);
    if (same(this.hover, s)) return;
    this.hover = s ? { l: s.l, j: s.j } : null;
    this.emit('hover');
  };
  App.setLens = function (key, value) { this.lenses[key] = value; this.emit('lens'); };
  App.setOverlay = function (key, value) { this.overlays[key] = value; this.emit('overlay'); };

  // ------------------------------------------------------------------ probe input
  App.makeProbe = function (raw, sample) {
    const ds = this.ds, r = Float64Array.from(raw);
    const x = ds.kind === 'plane' ? D.featurizePlane(r, 1, ds.featureIds) : r;
    return { raw: r, x, sample: sample === undefined ? null : sample, label: sample !== null && sample !== undefined ? ds.y[sample] : null };
  };
  App.setProbe = function (raw, sample = null, extra) {
    const p = this.makeProbe(raw, sample);
    if (extra) Object.assign(p, extra);
    this.pinned = p; this.probe = p;
    this.probeVersion++;
    this.emit('probe');
  };
  App.previewProbe = function (raw) {
    if (raw === null) { if (this.probe === this.pinned) return; this.probe = this.pinned; }
    else this.probe = this.makeProbe(raw, null);
    this.probeVersion++;
    this.emit('probe');
  };
  App.setProbeSample = function (k) {
    const d = this.ds.rawDim;
    this.setProbe(this.ds.raw.subarray(k * d, (k + 1) * d), k, this.ds.strokes ? { strokes: this.ds.strokes[k].strokes, width: this.ds.strokes[k].width } : null);
  };
  App.resetProbe = function (keep) {
    const ds = this.ds;
    if (keep && this.pinned && this.pinned.raw.length === ds.rawDim) {
      const p = this.pinned;
      if (p.sample !== null && p.sample < ds.n) this.setProbeSample(p.sample);
      else this.setProbe(p.raw, null, p.strokes ? { strokes: p.strokes, width: p.width } : null);
      return;
    }
    if (ds.kind === 'image') { this.setProbeSample(ds.valIdx[0]); return; }
    // pick the training sample nearest a fixed, typical location
    const target = ds.kind === 'plane' ? [0.42, 0.3] : [0.3];
    let best = ds.trainIdx[0], bd = Infinity;
    for (const k of ds.trainIdx) {
      let d = 0; for (let i = 0; i < ds.rawDim; i++) d += (ds.raw[k * ds.rawDim + i] - target[i]) ** 2;
      if (d < bd) { bd = d; best = k; }
    }
    this.setProbeSample(best);
  };
  App.probeTrace = function () {
    const key = this.version + ':' + this.probeVersion;
    if (this._pt && this._pt.key === key) return this._pt.value;
    const p = this.probe;
    const value = this.net.trace(p.x, 0, p.label === null ? undefined : p.label);
    this._pt = { key, value };
    return value;
  };

  // ------------------------------------------------------------------ analysis caches
  /* Recomputes when weights changed; while training, a stale value younger than maxAge ms is reused. */
  App.cached = function (name, maxAge, compute) {
    const c = this.caches[name], now = performance.now();
    if (c && c.version === this.version) return c.value;
    if (c && this.training && now - c.time < maxAge) return c.value;
    const value = compute(c ? c.value : null);
    this.caches[name] = { version: this.version, time: now, value };
    return value;
  };

  App.GRID = 50;
  App.grid = function () {
    return this.cached('grid', 45, (prev) => {
      const ds = this.ds;
      if (ds.kind === 'plane') {
        const G = this.GRID, n = G * G;
        let raw = prev && prev.raw && prev.raw.length === n * 2 ? prev.raw : null;
        if (!raw) {
          raw = new Float64Array(n * 2);
          for (let gy = 0; gy < G; gy++) for (let gx = 0; gx < G; gx++) { const p = gy * G + gx; raw[2 * p] = -1 + (2 * gx + 1) / G; raw[2 * p + 1] = 1 - (2 * gy + 1) / G; }
        }
        const X = D.featurizePlane(raw, n, ds.featureIds);
        return { kind: 'plane', G, n, raw, F: this.net.forwardMany(X, n, prev && prev.F) };
      }
      if (ds.kind === 'line') {
        const n = 241, raw = new Float64Array(n);
        for (let i = 0; i < n; i++) raw[i] = -1 + (2 * i) / (n - 1);
        return { kind: 'line', n, raw, F: this.net.forwardMany(raw, n, prev && prev.F) };
      }
      return null;
    });
  };
  App.dataPass = function () {
    return this.cached('dataPass', 220, (prev) => this.net.forwardMany(this.ds.X, this.ds.n, prev));
  };
  App.stats = function () {
    return this.cached('stats', 400, () => NA.engine.neuronStats(this.net, this.dataPass(), this.ds.trainIdx));
  };
  App.gradReport = function () {
    return this.cached('grad', 160, () => this.net.gradientReport(this.ds.X, this.ds.y, this.evalTrain, this.hyper.reg, this.hyper.regRate));
  };
  App.importance = function () {
    return this.cached('importance', 2500, () => this.net.importance(this.ds.X, this.ds.y, this.evalVal.length ? this.evalVal : this.evalTrain, this.ds.task));
  };

  /* The held-out test: evaluated only on request, after training. Records the step it was run at so
     the report can tell when the model has changed since. */
  App.runTest = function () {
    const ds = this.ds, net = this.net, idx = ds.testIdx, n = idx.length, d0 = net.sizes[0];
    if (!n) return null;
    const cls = ds.task !== 'regression', K = cls ? (ds.task === 'binary' ? 2 : ds.classes) : 0;
    const confusion = cls ? Array.from({ length: K }, () => new Array(K).fill(0)) : null;
    let loss = 0, correct = 0, sse = 0, sae = 0, sy = 0, syy = 0;
    const errors = [];
    for (const k of idx) {
      net.forward(ds.X, k * d0);
      loss += net.lossOf(ds.y[k]);
      const out = net.a[net.L];
      if (cls) {
        const pred = net.predicted(), conf = ds.task === 'binary' ? (pred ? out[0] : 1 - out[0]) : out[pred];
        confusion[ds.y[k]][pred]++;
        if (pred === ds.y[k]) correct++; else errors.push({ k, truth: ds.y[k], pred, conf });
      } else {
        const r = out[0] - ds.y[k];
        sse += r * r; sae += Math.abs(r); sy += ds.y[k]; syy += ds.y[k] * ds.y[k];
        errors.push({ k, truth: ds.y[k], pred: out[0], err: Math.abs(r) });
      }
    }
    const m = this.metrics();
    const res = { step: this.displayStep(), version: this.version, n, loss: loss / n, task: ds.task, lossId: net.lossId, valLoss: m.valLoss, valAcc: m.valAcc, trainAcc: m.trainAcc };
    if (cls) { res.acc = correct / n; res.confusion = confusion; res.errors = errors.sort((a, b) => b.conf - a.conf); }
    else {
      const sst = syy - (sy * sy) / n;
      res.acc = sst > 1e-12 ? 1 - sse / sst : 0;
      res.rmse = Math.sqrt(sse / n); res.mae = sae / n;
      res.errors = errors.sort((a, b) => b.err - a.err).slice(0, 12);
    }
    this.testResult = res;
    this.emit('test', res);
    return res;
  };
  // Max |activation| per layer over the dataset, used to normalise colours.
  App.layerScale = function () {
    return this.cached('scale', 400, () => {
      const F = this.dataPass();
      return this.net.sizes.map((m, l) => {
        let a = 1e-9, z = 1e-9;
        const A = F.a[l], Z = F.z[l];
        for (let i = 0; i < A.length; i++) { const va = Math.abs(A[i]), vz = Math.abs(Z[i]); if (va > a) a = va; if (vz > z) z = vz; }
        return { a, z };
      });
    });
  };
  App.predictionsAll = function () {
    return this.cached('pred', 300, () => {
      const F = this.dataPass(), ds = this.ds, L = this.net.L, n = ds.n, out = new Float64Array(n), A = F.a[L], K = this.net.sizes[L];
      for (let p = 0; p < n; p++) {
        if (ds.task === 'regression') out[p] = A[p];
        else if (ds.task === 'binary') out[p] = A[p] >= 0.5 ? 1 : 0;
        else { let best = 0; for (let k = 1; k < K; k++) if (A[k * n + p] > A[best * n + p]) best = k; out[p] = best; }
      }
      return out;
    });
  };

  NA.app = App;
  NA.DEFAULT_LENSES = DEFAULT_LENSES;
  NA.DEFAULT_OVERLAYS = DEFAULT_OVERLAYS;
})(window.NA);
