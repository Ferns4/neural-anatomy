/* classic-ml.js — the standard models of machine learning that are not neural networks, written from scratch for
   the model builder: a linear model (least squares or ridge regression for numbers, logistic regression for
   classes), k-nearest neighbours, a decision tree (CART), a random forest (trees on bootstrap samples and random
   features) and gradient-boosted trees in the manner of XGBoost (second-order gradients, L2-regularised leaf
   values, a split penalty γ, shrinkage, row and column sampling, histogram split search).
   Each model trains on the whole training set a round at a time, so the page can show it growing: one solve or
   gradient step, one level of a tree, one tree of a forest, one boosting round. The interface matches the
   neural engine's where the builder needs it: step() trains one round, predict(X, n) gives scores per class
   (log-probabilities, so a softmax recovers the probabilities) or values, evaluate(X, Y, n) gives the loss
   and accuracy counts. Inputs are flat rows of numbers: a picture or a sequence of numbers is read pixel by
   pixel, position by position. */
(function (NA) {
  'use strict';

  const KINDS = {
    linear: { label: 'Linear model', defaults: { lambda: 0 } },
    knn: { label: 'k-nearest neighbours', defaults: { k: 5, weights: 'uniform' } },
    tree: { label: 'Decision tree', defaults: { depth: 6, minLeaf: 2 } },
    forest: { label: 'Random forest', defaults: { trees: 50, depth: 8, features: 'sqrt', minLeaf: 1 } },
    boost: { label: 'Gradient boosting', defaults: { rounds: 100, eta: 0.3, depth: 4, lambda: 1, gamma: 0, subsample: 1, colsample: 1 } },
  };
  const isClassic = (type) => Object.prototype.hasOwnProperty.call(KINDS, type);
  const withDefaults = (spec) => Object.assign({}, KINDS[spec.type].defaults, ...Object.keys(spec).filter((k) => spec[k] !== undefined && spec[k] !== null).map((k) => ({ [k]: spec[k] })));
  const MAX_BINS = 48;

  // ------------------------------------------------------------------ checking a setup
  /* Whether blocks (with a classic model among them) fit the data d: { ok, shapes, outShape, problem, params }, in
     the form of the neural engine's check(). A classic model is the whole model, so it must be the only block. */
  function check(blocks, d) {
    const res = { ok: true, shapes: [], outShape: null, problem: null, params: 0 }, inp = d.input || {}, out = d.output || {};
    const F = inp.kind === 'vector' ? inp.dim : inp.kind === 'image' ? inp.c * inp.h * inp.w : inp.kind === 'series' ? inp.len * inp.dim : 0;
    const K = out.kind === 'classes' ? out.k : out.kind === 'value' ? out.k || 1 : 0;
    const classic = blocks.filter((b) => isClassic(b.type));
    for (const b of blocks) {
      const s = isClassic(b.type) ? withDefaults(b) : null;
      let problem = null;
      if (!s) problem = { reason: `A ${KINDS[classic[0].type].label.toLowerCase()} is a whole model on its own, not a layer: it cannot be combined with other blocks. Remove this block, or the ${KINDS[classic[0].type].label.toLowerCase()}.`, fix: null };
      else if (classic.length > 1) problem = { reason: 'Only one classic model at a time: each is a whole model on its own.', fix: null };
      else if (!F) problem = { reason: 'Classic models read rows of numbers. Words and tokens are not numbers to them: pick a problem with points, a curve, a table, pictures or a series.', fix: null };
      else if (!K) problem = { reason: 'Classic models predict a class or a value for each example, not a token at every position.', fix: null };
      else problem = settingsProblem(s, d);
      const params = s && !problem ? estimate(s, F, K, d.nTr) : 0;
      res.shapes.push({ in: F ? [F] : null, out: K ? [K] : null, params, problem });
      if (problem) res.ok = false;
      res.params += params;
    }
    res.outShape = K ? [K] : null;
    return res;
  }
  function settingsProblem(s, d) {
    const int = (v, lo, hi) => Number.isInteger(v) && v >= lo && v <= hi;
    if (s.type === 'linear' && !(s.lambda >= 0)) return { reason: 'The ridge penalty λ must be 0 or more.', fix: null };
    if (s.type === 'knn' && !int(s.k, 1, Math.max(1, d.nTr || 1))) return { reason: `k must be a whole number from 1 to the number of training examples (${d.nTr}).`, fix: null };
    if ((s.type === 'tree' || s.type === 'forest' || s.type === 'boost') && !int(s.depth, 1, 16)) return { reason: 'Tree depth must be a whole number from 1 to 16.', fix: null };
    if (s.type === 'forest' && !int(s.trees, 1, 500)) return { reason: 'A forest needs 1 to 500 trees.', fix: null };
    if (s.type === 'boost' && !int(s.rounds, 1, 2000)) return { reason: 'Boosting needs 1 to 2,000 rounds.', fix: null };
    if (s.type === 'boost' && !(s.eta > 0 && s.eta <= 1)) return { reason: 'The learning rate η must be above 0 and at most 1.', fix: null };
    return null;
  }
  // a rough count of the numbers the model will learn, before it has
  function estimate(s, F, K, n) {
    const leaves = (depth) => Math.min(2 ** depth, Math.max(1, n || 1));
    const tree = (depth, outs) => (leaves(depth) - 1) * 2 + leaves(depth) * outs;
    if (s.type === 'linear') return K * (F + 1);
    if (s.type === 'knn') return (n || 0) * (F + 1);
    if (s.type === 'tree') return tree(s.depth, K);
    if (s.type === 'forest') return s.trees * tree(s.depth, K);
    return s.rounds * (K > 2 ? K : 1) * tree(s.depth, 1);
  }

  // ------------------------------------------------------------------ helpers
  // Up to MAX_BINS thresholds per feature from the training values (quantiles), and each training value's bin, stored
  // feature by feature (bins[f·n + i]) so that a split search reads one contiguous column.
  function binData(X, n, F) {
    const thr = [], bins = new Uint8Array(n * F), col = new Float64Array(n);
    for (let f = 0; f < F; f++) {
      for (let i = 0; i < n; i++) col[i] = X[i * F + f];
      const sorted = Float64Array.from(col).sort(), t = [];
      for (let q = 1; q < MAX_BINS; q++) {
        const v = sorted[Math.min(n - 1, Math.floor((q * n) / MAX_BINS))];
        if (!t.length || v > t[t.length - 1]) t.push(v);
      }
      if (t.length && t[t.length - 1] >= sorted[n - 1]) t.pop(); // a threshold at the maximum splits nothing off
      thr.push(Float64Array.from(t));
      for (let i = 0; i < n; i++) { let lo = 0, hi = t.length; const v = col[i]; while (lo < hi) { const m = (lo + hi) >> 1; if (v <= t[m]) hi = m; else lo = m + 1; } bins[f * n + i] = lo; }
    }
    return { thr, bins, n };
  }
  const logsumexp = (a, o, K) => { let m = -Infinity; for (let k = 0; k < K; k++) m = Math.max(m, a[o + k]); let s = 0; for (let k = 0; k < K; k++) s += Math.exp(a[o + k] - m); return m + Math.log(s); };

  /* One regression tree grown on gradients g and hessians h of the rows in idx (XGBoost's exact form: leaf value
     −G/(H + λ), split gain ½[G_L²/(H_L+λ) + G_R²/(H_R+λ) − G²/(H+λ)] − γ), or for a plain tree on class counts or
     values: the same machinery with g, h set so that the leaf value is the mean (squared error) and the gain is
     the drop in squared error, which for one-hot targets is the Gini decrease. Grown level by level; feats: the
     features a split may use (all, or a random subset per split for a forest). */
  class Tree {
    constructor(outs) { this.outs = outs; this.feat = []; this.thr = []; this.bin = []; this.left = []; this.right = []; this.value = []; this.n = []; this.gain = []; this.depth = []; }
    leafCount() { return this.feat.filter((f) => f < 0).length; }
    get size() { return this.feat.length; }
    // the leaf index for row r of X (F numbers per row)
    leafOf(X, o) { let i = 0; while (this.feat[i] >= 0) i = X[o + this.feat[i]] <= this.thr[i] ? this.left[i] : this.right[i]; return i; }
  }
  /* Grows tree t over the given rows. G[r·O + c], Hs[r·O + c]: per row, per output c (O outputs, e.g. one-hot class
     residuals), so one tree can hold class probabilities in its leaves. Options: depth, minLeaf, lambda, gamma,
     pickFeatures(), eta (scales the leaf values). Returns the leaves' rows for the caller (boosting updates its
     predictions from them). levels(n) grows n more levels; the tree is done when no leaf can split. */
  class Grower {
    constructor(t, rows, G, Hs, O, data, o) {
      Object.assign(this, { t, G, Hs, O, data, o });
      this.frontier = [this.addNode(rows, 0)];
    }
    stats(rows) {
      const { O, G, Hs } = this, g = new Float64Array(O), h = new Float64Array(O);
      for (const r of rows) for (let c = 0; c < O; c++) { g[c] += G[r * O + c]; h[c] += Hs[r * O + c]; }
      return { g, h };
    }
    leafValue(st) { const { O, o } = this, v = new Float64Array(O); for (let c = 0; c < O; c++) v[c] = (-st.g[c] / (st.h[c] + o.lambda)) * (o.eta || 1); return v; }
    addNode(rows, depth) {
      const t = this.t, i = t.feat.length, st = this.stats(rows);
      t.feat.push(-1); t.thr.push(0); t.bin.push(0); t.left.push(-1); t.right.push(-1); t.value.push(this.leafValue(st)); t.n.push(rows.length); t.gain.push(0); t.depth.push(depth);
      return { i, rows, st, depth };
    }
    score(g, h) { let s = 0; for (let c = 0; c < this.O; c++) s += (g[c] * g[c]) / (h[c] + this.o.lambda); return s; }
    // the best split of one node: { f, b, gain } over the binned features (histograms of G and H per bin)
    best(node) {
      const { O, G, Hs, data, o } = this, n = data.n, { bins, thr } = data, base = this.score(node.st.g, node.st.h), rows = node.rows, R = rows.length;
      if (!this.hg) { this.hg = new Float64Array(MAX_BINS * O); this.hh = new Float64Array(MAX_BINS * O); this.cnt = new Int32Array(MAX_BINS); for (const k of ['gl', 'hl', 'gr', 'hr']) this[k] = new Float64Array(O); }
      const { hg, hh, cnt, gl, hl, gr, hr } = this;
      let best = null;
      for (const f of o.pickFeatures()) {
        const nb = thr[f].length + 1;
        if (nb < 2) continue;
        hg.fill(0, 0, nb * O); hh.fill(0, 0, nb * O); cnt.fill(0, 0, nb);
        const col = f * n;
        if (O === 1) for (let q = 0; q < R; q++) { const r = rows[q], b = bins[col + r]; cnt[b]++; hg[b] += G[r]; hh[b] += Hs[r]; }
        else for (let q = 0; q < R; q++) { const r = rows[q], b = bins[col + r]; cnt[b]++; for (let c = 0; c < O; c++) { hg[b * O + c] += G[r * O + c]; hh[b * O + c] += Hs[r * O + c]; } }
        gl.fill(0); hl.fill(0);
        let nl = 0;
        for (let b = 0; b < nb - 1; b++) {
          nl += cnt[b];
          for (let c = 0; c < O; c++) { gl[c] += hg[b * O + c]; hl[c] += hh[b * O + c]; }
          const nr = node.rows.length - nl;
          if (nl < o.minLeaf || nr < o.minLeaf) continue;
          let okH = true;
          for (let c = 0; c < O; c++) { gr[c] = node.st.g[c] - gl[c]; hr[c] = node.st.h[c] - hl[c]; }
          if (o.minChild) { let sl = 0, sr = 0; for (let c = 0; c < O; c++) { sl += hl[c]; sr += hr[c]; } okH = sl >= o.minChild && sr >= o.minChild; }
          if (!okH) continue;
          const gain = 0.5 * (this.score(gl, hl) + this.score(gr, hr) - base) - o.gamma;
          if (gain > 1e-12 && (!best || gain > best.gain)) best = { f, b, gain };
        }
      }
      return best;
    }
    // grows one more level; false when nothing could split
    level() {
      const t = this.t, next = [], n = this.data.n;
      for (const node of this.frontier) {
        if (node.depth >= this.o.depth || node.rows.length < 2 * this.o.minLeaf) continue;
        const s = this.best(node);
        if (!s) continue;
        const L = [], R = [];
        const col = s.f * n;
        for (const r of node.rows) (this.data.bins[col + r] <= s.b ? L : R).push(r);
        const i = node.i;
        t.feat[i] = s.f; t.bin[i] = s.b; t.thr[i] = this.data.thr[s.f][s.b]; t.gain[i] = s.gain;
        const l = this.addNode(L, node.depth + 1), r = this.addNode(R, node.depth + 1);
        t.left[i] = l.i; t.right[i] = r.i;
        next.push(l, r);
      }
      this.frontier = next;
      return next.length > 0;
    }
    grow() { while (this.level()); return this.t; }
  }

  // ------------------------------------------------------------------ the models
  class Classic {
    constructor(spec, d, opts) {
      const s = withDefaults(spec), inp = d.input;
      const F = inp.kind === 'vector' ? inp.dim : inp.kind === 'image' ? inp.c * inp.h * inp.w : inp.len * inp.dim;
      const cls = d.output.kind === 'classes', K = cls ? d.output.k : d.output.k || 1;
      Object.assign(this, { classic: true, type: s.type, s, F, K, cls, n: d.nTr, X: d.Xtr, Y: d.Ytr, rounds: 0, layers: [], params: [], embed: null });
      this.rng = new NA.RNG(((opts && opts.seed) || 1) * 7907 + 13);
      this.inSize = F; this.outSize = K; this.outK = K; this.kind = cls ? 'classes' : 'value'; this.ySize = cls ? 1 : K;
      this.trees = [];
      if (s.type === 'tree' || s.type === 'forest' || s.type === 'boost') this.data = Object.assign(binData(d.Xtr, d.nTr, F), { F });
      if (s.type === 'boost') this.initBoost();
      if (s.type === 'linear') this.initLinear();
      this.target = s.type === 'forest' ? s.trees : s.type === 'boost' ? s.rounds : s.type === 'tree' ? s.depth : s.type === 'linear' ? (cls ? 300 : 1) : 1;
    }
    // ---- the interface the builder uses
    step() { if (!this.done()) this.round(); return { loss: this.trainLoss(), correct: 0, count: 0, gradNorm: 0 }; }
    done() { return this.rounds >= this.target || !!this.finished; }
    canExtend() { return this.type === 'forest' || this.type === 'boost' || (this.type === 'linear' && this.cls); }
    extend() { this.finished = false; this.target += this.type === 'forest' ? this.s.trees : this.type === 'boost' ? this.s.rounds : 300; }
    gradNorms() { return []; }
    zeroGrad() {}
    get paramCount() {
      if (this.type === 'linear') return this.K * (this.F + 1);
      if (this.type === 'knn') return this.n * (this.F + 1);
      let s = 0;
      for (const t of this.trees) for (let i = 0; i < t.feat.length; i++) s += t.feat[i] >= 0 ? 2 : t.outs;
      return s;
    }
    // what one round is called, for the page
    unit() { return { linear: this.cls ? 'gradient step' : 'solve', knn: 'look-up', tree: 'level', forest: 'tree', boost: 'round' }[this.type]; }

    round() {
      const t = this.type;
      if (t === 'linear') this.cls ? this.logisticSteps(1) : this.solveRidge();
      else if (t === 'knn') this.finished = true; // nothing to learn: the training examples are the model
      else if (t === 'tree') this.treeLevel();
      else if (t === 'forest') this.addForestTree();
      else this.boostRound();
      this.rounds++;
      this._trainLoss = null;
    }

    // ---- linear model
    initLinear() { this.W = new Float64Array(this.K * (this.F + 1)); this.mW = new Float64Array(this.W.length); this.vW = new Float64Array(this.W.length); }
    // ridge regression in closed form: (XᵀX + λI) w = Xᵀy with a bias column (not penalised), by Cholesky
    solveRidge() {
      const { F, K, n, X, Y } = this, P = F + 1, A = new Float64Array(P * P), lam = this.s.lambda;
      const row = new Float64Array(P);
      for (let i = 0; i < n; i++) {
        for (let f = 0; f < F; f++) row[f] = X[i * F + f];
        row[F] = 1;
        for (let a = 0; a < P; a++) { const ra = row[a]; if (!ra) continue; for (let b = 0; b <= a; b++) A[a * P + b] += ra * row[b]; }
      }
      for (let a = 0; a < P; a++) for (let b = a + 1; b < P; b++) A[a * P + b] = A[b * P + a];
      for (let f = 0; f < F; f++) A[f * P + f] += lam + 1e-9 * n; // a hair of ridge keeps a singular XᵀX solvable
      A[F * P + F] += 1e-9;
      const L = cholesky(A, P);
      for (let k = 0; k < K; k++) {
        const b = new Float64Array(P);
        for (let i = 0; i < n; i++) { const y = Y[i * K + k]; for (let f = 0; f < F; f++) b[f] += X[i * F + f] * y; b[F] += y; }
        const w = cholSolve(L, b, P);
        for (let p = 0; p < P; p++) this.W[k * P + p] = w[p];
      }
      this.finished = true;
    }
    // logistic regression: full-batch gradient steps (Adam) on the mean cross-entropy plus ½λ|w|²
    logisticSteps(k) {
      const { F, K, n, X, Y, W } = this, P = F + 1, lam = this.s.lambda, g = new Float64Array(W.length), z = new Float64Array(K);
      for (let it = 0; it < k; it++) {
        g.fill(0);
        for (let i = 0; i < n; i++) {
          for (let c = 0; c < K; c++) { let s = W[c * P + F]; for (let f = 0; f < F; f++) s += W[c * P + f] * X[i * F + f]; z[c] = s; }
          const lse = logsumexp(z, 0, K);
          for (let c = 0; c < K; c++) {
            const e = (Math.exp(z[c] - lse) - (c === Y[i] ? 1 : 0)) / n;
            for (let f = 0; f < F; f++) g[c * P + f] += e * X[i * F + f];
            g[c * P + F] += e;
          }
        }
        for (let c = 0; c < K; c++) for (let f = 0; f < F; f++) g[c * P + f] += lam * W[c * P + f];
        this.adamT = (this.adamT || 0) + 1;
        const lr = 0.05, b1 = 0.9, b2 = 0.999, c1 = 1 - b1 ** this.adamT, c2 = 1 - b2 ** this.adamT;
        for (let p = 0; p < W.length; p++) { this.mW[p] = b1 * this.mW[p] + (1 - b1) * g[p]; this.vW[p] = b2 * this.vW[p] + (1 - b2) * g[p] * g[p]; W[p] -= (lr * this.mW[p] / c1) / (Math.sqrt(this.vW[p] / c2) + 1e-8); }
      }
    }

    // ---- trees
    // a plain tree or a forest's tree: gradients that make each leaf the mean of its rows' targets (class one-hots or values)
    treeTargets(rows) {
      const { K, cls, Y } = this, O = K, G = new Float64Array(this.n * O), Hs = new Float64Array(this.n * O);
      for (const r of rows) for (let c = 0; c < O; c++) { const y = cls ? (Y[r] === c ? 1 : 0) : Y[r * K + c]; G[r * O + c] += -y; Hs[r * O + c] += 1; }
      return { G, Hs, O };
    }
    treeLevel() {
      if (!this.grower) {
        const rows = Array.from({ length: this.n }, (_, i) => i), { G, Hs, O } = this.treeTargets(rows), all = Array.from({ length: this.F }, (_, i) => i);
        this.trees = [new Tree(O)];
        this.grower = new Grower(this.trees[0], rows, G, Hs, O, this.data, { depth: this.s.depth, minLeaf: this.s.minLeaf, lambda: 0, gamma: 0, pickFeatures: () => all });
      }
      if (!this.grower.level()) this.finished = true;
    }
    addForestTree() {
      const n = this.n, F = this.F, rows = [], counts = new Int32Array(n);
      for (let i = 0; i < n; i++) counts[this.rng.int(n)]++; // a bootstrap sample: n rows drawn with replacement
      for (let i = 0; i < n; i++) for (let c = 0; c < counts[i]; c++) rows.push(i);
      const { G, Hs, O } = this.treeTargets(rows);
      const m = this.s.features === 'all' ? F : this.s.features === 'third' ? Math.max(1, Math.round(F / 3)) : Math.max(1, Math.round(Math.sqrt(F)));
      const pick = () => { if (m >= F) return Array.from({ length: F }, (_, i) => i); const all = Array.from({ length: F }, (_, i) => i); for (let i = 0; i < m; i++) { const j = i + this.rng.int(F - i); const tmp = all[i]; all[i] = all[j]; all[j] = tmp; } return all.slice(0, m); };
      const t = new Tree(O);
      // a bootstrap row drawn twice counts twice, through G and H
      new Grower(t, [...new Set(rows)], G, Hs, O, this.data, { depth: this.s.depth, minLeaf: this.s.minLeaf, lambda: 0, gamma: 0, pickFeatures: pick }).grow();
      this.trees.push(t);
    }

    // ---- gradient boosting
    initBoost() {
      const { K, cls, n, Y } = this;
      this.T = cls && K === 2 ? 1 : K; // one tree per round for two classes (a logistic score), one per class otherwise
      this.base = new Float64Array(this.T);
      if (!cls) for (let c = 0; c < K; c++) { let s = 0; for (let i = 0; i < n; i++) s += Y[i * K + c]; this.base[c] = s / n; }
      this.Ftr = new Float64Array(n * this.T);
      for (let i = 0; i < n; i++) for (let c = 0; c < this.T; c++) this.Ftr[i * this.T + c] = this.base[c];
    }
    boostRound() {
      const { n, T, K, cls, Y, Ftr, s } = this, G = new Float64Array(n), Hs = new Float64Array(n);
      const rows = [];
      for (let i = 0; i < n; i++) if (s.subsample >= 1 || this.rng.next() < s.subsample) rows.push(i);
      const F = this.F, m = Math.max(1, Math.round(F * Math.min(1, s.colsample))), cols = Array.from({ length: F }, (_, i) => i);
      for (let i = 0; i < m && m < F; i++) { const j = i + this.rng.int(F - i); const tmp = cols[i]; cols[i] = cols[j]; cols[j] = tmp; }
      const feats = cols.slice(0, m), p = new Float64Array(K);
      const round = [];
      for (let c = 0; c < T; c++) {
        for (const i of rows) {
          if (!cls) { G[i] = Ftr[i * T + c] - Y[i * K + c]; Hs[i] = 1; }
          else if (T === 1) { const q = 1 / (1 + Math.exp(-Ftr[i])); G[i] = q - Y[i]; Hs[i] = Math.max(q * (1 - q), 1e-6); }
          else {
            const lse = logsumexp(Ftr, i * T, T);
            for (let k = 0; k < T; k++) p[k] = Math.exp(Ftr[i * T + k] - lse);
            G[i] = p[c] - (Y[i] === c ? 1 : 0); Hs[i] = Math.max(2 * p[c] * (1 - p[c]), 1e-6);
          }
        }
        const t = new Tree(1);
        new Grower(t, rows, G, Hs, 1, this.data, { depth: s.depth, minLeaf: 1, minChild: 1, lambda: s.lambda, gamma: s.gamma, eta: s.eta, pickFeatures: () => feats }).grow();
        round.push(t);
      }
      // the new trees' outputs added to every training row's scores (all rows, not just the sampled ones)
      for (let c = 0; c < T; c++) { const t = round[c]; for (let i = 0; i < n; i++) Ftr[i * T + c] += t.value[t.leafOf(this.X, i * this.F)][0]; }
      this.trees.push(...round);
    }

    boostLoss() {
      const { n, T, K, cls, Y, Ftr } = this;
      let sum = 0;
      for (let i = 0; i < n; i++) {
        if (!cls) for (let c = 0; c < K; c++) { const r = Ftr[i * T + c] - Y[i * K + c]; sum += 0.5 * r * r; }
        else if (T === 1) { const z = Ftr[i]; sum += Math.max(z, 0) - z * Y[i] + Math.log1p(Math.exp(-Math.abs(z))); }
        else sum += logsumexp(Ftr, i * T, T) - Ftr[i * T + Y[i]];
      }
      return sum / n;
    }

    // ---- k-nearest neighbours
    knnPredict(X, o, out, oo) {
      const { F, n, K, cls, s } = this, k = Math.min(s.k, n), best = new Float64Array(k).fill(Infinity), who = new Int32Array(k).fill(-1);
      for (let j = 0; j < n; j++) {
        let d = 0;
        for (let f = 0; f < F; f++) { const v = X[o + f] - this.X[j * F + f]; d += v * v; }
        if (d >= best[k - 1]) continue;
        let p = k - 1;
        while (p > 0 && best[p - 1] > d) { best[p] = best[p - 1]; who[p] = who[p - 1]; p--; }
        best[p] = d; who[p] = j;
      }
      const acc = new Float64Array(K);
      let wsum = 0;
      for (let q = 0; q < k; q++) {
        const j = who[q];
        if (j < 0) continue;
        const w = s.weights === 'distance' ? 1 / (Math.sqrt(best[q]) + 1e-6) : 1;
        wsum += w;
        if (cls) acc[this.Y[j]] += w; else for (let c = 0; c < K; c++) acc[c] += w * this.Y[j * K + c];
      }
      for (let c = 0; c < K; c++) out[oo + c] = cls ? Math.log(acc[c] / wsum + 1e-6) : acc[c] / wsum;
      this.lastNeighbours = who;
    }

    // ---- predictions: scores per class (log-probabilities) or values, n rows
    predict(X, n) {
      const { F, K, cls, type } = this, out = new Float32Array(n * K);
      if (type === 'knn') { for (let i = 0; i < n; i++) this.knnPredict(X, i * F, out, i * K); return out; }
      if (type === 'linear') {
        const P = F + 1, W = this.W, z = new Float64Array(K);
        for (let i = 0; i < n; i++) {
          for (let c = 0; c < K; c++) { let s = W[c * P + F]; for (let f = 0; f < F; f++) s += W[c * P + f] * X[i * F + f]; z[c] = s; }
          for (let c = 0; c < K; c++) out[i * K + c] = z[c];
        }
        return out;
      }
      if (type === 'boost') {
        const T = this.T, sc = new Float64Array(T);
        for (let i = 0; i < n; i++) {
          sc.set(this.base);
          for (let r = 0; r < this.trees.length; r++) { const t = this.trees[r]; sc[r % T] += t.value[t.leafOf(X, i * F)][0]; }
          if (!cls) for (let c = 0; c < K; c++) out[i * K + c] = sc[c];
          else if (T === 1) { out[i * 2] = 0; out[i * 2 + 1] = sc[0]; } // softmax of (0, s) is the logistic σ(s)
          else for (let c = 0; c < K; c++) out[i * K + c] = sc[c];
        }
        return out;
      }
      // a tree or a forest: the mean of the trees' leaf values (class shares or values)
      const acc = new Float64Array(K), nt = Math.max(1, this.trees.length);
      for (let i = 0; i < n; i++) {
        acc.fill(0);
        for (const t of this.trees) { const v = t.value[t.leafOf(X, i * F)]; for (let c = 0; c < K; c++) acc[c] += v[c]; }
        for (let c = 0; c < K; c++) out[i * K + c] = cls ? Math.log(Math.max(acc[c] / nt, 0) + 1e-6) : acc[c] / nt;
      }
      return out;
    }
    /* Loss and accuracy counts over n rows: the mean cross-entropy (log loss) for classes, half the squared error for
       values, as the neural models report them. */
    evaluate(X, Y, n) {
      const out = this.predict(X, n), K = this.K;
      let sum = 0, correct = 0;
      for (let i = 0; i < n; i++) {
        if (this.cls) {
          const lse = logsumexp(out, i * K, K), y = Y[i];
          sum += lse - out[i * K + y];
          let b = 0; for (let c = 1; c < K; c++) if (out[i * K + c] > out[i * K + b]) b = c;
          if (b === y) correct++;
        } else for (let c = 0; c < K; c++) { const r = out[i * K + c] - Y[i * K + c]; sum += 0.5 * r * r; }
      }
      return { loss: n ? sum / n : 0, sum, correct, count: n };
    }
    trainLoss() {
      if (this.type === 'boost') return this.boostLoss(); // from the running scores: no need to walk every tree again
      if (this._trainLoss === null || this._trainLoss === undefined) {
        const n = Math.min(this.n, this.type === 'knn' ? 300 : 1500);
        this._trainLoss = this.evaluate(this.X, this.Y, n).loss;
      }
      return this._trainLoss;
    }
  }

  // ------------------------------------------------------------------ linear algebra
  function cholesky(A, n) {
    const L = new Float64Array(n * n);
    for (let i = 0; i < n; i++) for (let j = 0; j <= i; j++) {
      let s = A[i * n + j];
      for (let k = 0; k < j; k++) s -= L[i * n + k] * L[j * n + k];
      L[i * n + j] = i === j ? Math.sqrt(Math.max(s, 1e-12)) : s / L[j * n + j];
    }
    return L;
  }
  function cholSolve(L, b, n) {
    const y = new Float64Array(n), x = new Float64Array(n);
    for (let i = 0; i < n; i++) { let s = b[i]; for (let k = 0; k < i; k++) s -= L[i * n + k] * y[k]; y[i] = s / L[i * n + i]; }
    for (let i = n - 1; i >= 0; i--) { let s = y[i]; for (let k = i + 1; k < n; k++) s -= L[k * n + i] * x[k]; x[i] = s / L[i * n + i]; }
    return x;
  }

  function build(blocks, d, opts) {
    const chk = check(blocks, d);
    if (!chk.ok) { const p = chk.shapes.find((s) => s.problem); throw new Error((p ? p.problem : chk.problem || { reason: 'This setup does not fit.' }).reason); }
    return new Classic(blocks.find((b) => isClassic(b.type)), d, opts);
  }

  NA.cml = { KINDS, isClassic, check, build, withDefaults, Tree };
})(typeof globalThis !== 'undefined' ? (globalThis.NA = globalThis.NA || {}) : (window.NA = window.NA || {}));
