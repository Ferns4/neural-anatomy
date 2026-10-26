/* engine.js — a small multilayer perceptron whose every number can be inspected:
   pre-activations, activations, error signals, gradients, optimizer state. */
(function (NA) {
  'use strict';

  const LEAK = 0.05;
  const SQ2PI = Math.sqrt(2 / Math.PI);
  const sigmoid = (z) => 1 / (1 + Math.exp(-z));
  const gelu = (z) => 0.5 * z * (1 + Math.tanh(SQ2PI * (z + 0.044715 * z * z * z)));
  function geluD(z) {
    const u = SQ2PI * (z + 0.044715 * z * z * z), t = Math.tanh(u);
    return 0.5 * (1 + t) + 0.5 * z * (1 - t * t) * SQ2PI * (1 + 3 * 0.044715 * z * z);
  }

  const softplus = (z) => (z > 30 ? z : Math.log1p(Math.exp(z)));

  // f: activation, d: derivative given (z, a). he: use He initialisation. sat: is this output on a flat tail?
  // formula / dformula: the function and its slope as text; apply(z): the formula with a number put in.
  const ACTIVATIONS = {
    relu: { id: 'relu', label: 'ReLU', short: 'ReLU', formula: 'max(0, z)', dformula: '1 if z > 0, else 0', apply: (z) => `max(0, ${z})`,
      f: (z) => (z > 0 ? z : 0), d: (z) => (z > 0 ? 1 : 0), family: 'rectifier', he: true, canDie: true },
    leaky: { id: 'leaky', label: 'Leaky ReLU', short: 'Leaky', formula: 'max(0.05z, z)', dformula: '1 if z > 0, else 0.05', apply: (z) => `max(0.05·${z}, ${z})`,
      f: (z) => (z > 0 ? z : LEAK * z), d: (z) => (z > 0 ? 1 : LEAK), family: 'rectifier', he: true },
    elu: { id: 'elu', label: 'ELU', short: 'ELU', formula: 'z if z > 0, else eᶻ − 1', dformula: '1 if z > 0, else eᶻ', apply: (z) => (z[0] === '−' ? `e^(${z}) − 1` : z),
      f: (z) => (z > 0 ? z : Math.expm1(z)), d: (z, a) => (z > 0 ? 1 : a + 1), family: 'smooth', he: true },
    gelu: { id: 'gelu', label: 'GELU', short: 'GELU', formula: 'z · Φ(z)', dformula: 'Φ(z) + z · φ(z)', apply: (z) => `${z} · Φ(${z})`,
      f: gelu, d: geluD, family: 'smooth', he: true, note: 'Φ is the standard normal cumulative distribution; computed with the tanh approximation.' },
    swish: { id: 'swish', label: 'Swish', short: 'Swish', formula: 'z · σ(z)', dformula: 'σ(z) + z · σ(z)(1 − σ(z))', apply: (z) => `${z} · σ(${z})`,
      f: (z) => z * sigmoid(z), d: (z) => { const s = sigmoid(z); return s + z * s * (1 - s); }, family: 'smooth', he: true },
    softplus: { id: 'softplus', label: 'Softplus', short: 'Softplus', formula: 'ln(1 + eᶻ)', dformula: 'σ(z) = 1 / (1 + e⁻ᶻ)', apply: (z) => `ln(1 + e^(${z}))`,
      f: softplus, d: (z) => sigmoid(z), family: 'smooth', he: true },
    tanh: { id: 'tanh', label: 'Tanh', short: 'tanh', formula: 'tanh(z) = (eᶻ − e⁻ᶻ) / (eᶻ + e⁻ᶻ)', dformula: '1 − tanh²(z)', apply: (z) => `tanh(${z})`,
      f: Math.tanh, d: (z, a) => 1 - a * a, family: 'squash', lo: -1, hi: 1, sat: (a) => Math.abs(a) > 0.97 },
    sigmoid: { id: 'sigmoid', label: 'Sigmoid', short: 'σ', formula: 'σ(z) = 1 / (1 + e⁻ᶻ)', dformula: 'σ(z) · (1 − σ(z))', apply: (z) => `1 / (1 + e^(${z[0] === '−' ? z.slice(1) : '−' + z}))`,
      f: sigmoid, d: (z, a) => a * (1 - a), family: 'squash', lo: 0, hi: 1, sat: (a) => a < 0.03 || a > 0.97 },
    sine: { id: 'sine', label: 'Sine', short: 'sin', formula: 'sin(z)', dformula: 'cos(z)', apply: (z) => `sin(${z})`,
      f: Math.sin, d: (z) => Math.cos(z), family: 'periodic', lo: -1, hi: 1 },
    gaussian: { id: 'gaussian', label: 'Gaussian', short: 'Gauss', formula: 'e^(−z²)', dformula: '−2z · e^(−z²)', apply: (z) => `e^(−(${z})²)`,
      f: (z) => Math.exp(-z * z), d: (z, a) => -2 * z * a, family: 'bump', lo: 0, hi: 1, sat: (a) => a < 0.03 },
    linear: { id: 'linear', label: 'Linear', short: 'linear', formula: 'z', dformula: '1', apply: (z) => z,
      f: (z) => z, d: () => 1, family: 'linear' },
  };
  const ACTIVATION_ORDER = ['relu', 'leaky', 'elu', 'gelu', 'swish', 'softplus', 'tanh', 'sigmoid', 'sine', 'gaussian', 'linear'];

  // Losses available for each output layer, first one is the default.
  const HUBER = 0.2;
  // formula[out]: the loss written for that output layer; delta[out]: its error signal ∂L/∂z.
  const LOSSES = {
    xent: { id: 'xent', label: 'Cross-entropy', short: 'Cross-ent.',
      formula: { sigmoid: 'L = −[y·ln p + (1 − y)·ln(1 − p)]', softmax: 'L = −ln p_y' },
      delta: { sigmoid: 'δ = p − y', softmax: 'δₖ = pₖ − yₖ' },
      hint: 'Punishes confident mistakes hardest; with sigmoid or softmax the error signal is simply p − y.' },
    mse: { id: 'mse', label: 'Squared error', short: 'Squared',
      formula: { sigmoid: 'L = ½(p − y)²', softmax: 'L = ½ Σₖ (pₖ − yₖ)²', identity: 'L = ½(ŷ − y)²' },
      delta: { sigmoid: 'δ = (p − y) · p(1 − p)', softmax: 'δⱼ = pⱼ[(pⱼ − yⱼ) − Σₖ pₖ(pₖ − yₖ)]', identity: 'δ = ŷ − y' },
      hint: 'Penalises the square of the miss. On a probability output it learns slowly when confidently wrong, because the slope p(1 − p) is tiny there.' },
    hinge: { id: 'hinge', label: 'Hinge', short: 'Hinge',
      formula: { sigmoid: 'L = max(0, 1 − t·z), t = ±1' },
      delta: { sigmoid: 'δ = −t if t·z < 1, else 0' },
      hint: 'The support-vector-machine loss on the raw score z: points beyond the margin (t·z ≥ 1) stop contributing at all.' },
    mae: { id: 'mae', label: 'Absolute error', short: 'Absolute',
      formula: { identity: 'L = |ŷ − y|' },
      delta: { identity: 'δ = sign(ŷ − y)' },
      hint: 'Every miss pulls with the same strength, so outliers matter less and the fit tracks the median.' },
    huber: { id: 'huber', label: 'Huber', short: 'Huber',
      formula: { identity: `L = ½r² if |r| ≤ ${HUBER}, else ${HUBER}(|r| − ${HUBER / 2})` },
      delta: { identity: `δ = r if |r| ≤ ${HUBER}, else ${HUBER}·sign(r)` },
      hint: `Squared error for small misses, absolute error beyond ${HUBER}: smooth near the target and robust to outliers.` },
  };
  const LOSS_FOR = { sigmoid: ['xent', 'mse', 'hinge'], softmax: ['xent', 'mse'], identity: ['mse', 'mae', 'huber'] };

  const OPTIMIZERS = {
    sgd: { id: 'sgd', label: 'SGD' },
    momentum: { id: 'momentum', label: 'Momentum' },
    adam: { id: 'adam', label: 'Adam' },
  };

  class MLP {
    /* cfg: { sizes:[n0..nL], act, acts:[per layer: array of activation ids], out:'sigmoid'|'softmax'|'identity',
              loss, init, seed } — every hidden neuron can have its own activation function. */
    constructor(cfg) {
      this.sizes = cfg.sizes.slice();
      this.L = this.sizes.length - 1;
      this.actId = cfg.act || 'relu';
      this.act = ACTIVATIONS[this.actId];
      this.out = cfg.out || 'sigmoid';
      this.setLoss(cfg.loss);
      const S = this.sizes;
      this.actIds = S.map((n, l) => (l > 0 && l < this.L ? new Array(n).fill(this.actId) : null));
      if (cfg.acts) {
        for (let l = 1; l < this.L; l++) {
          const src = cfg.acts[l];
          if (src) for (let j = 0; j < S[l]; j++) if (src[j] && ACTIVATIONS[src[j]]) this.actIds[l][j] = src[j];
        }
      }
      this.actFns = this.actIds.map((ids) => (ids ? ids.map((id) => ACTIVATIONS[id]) : null));
      this.W = []; this.b = []; this.gW = []; this.gb = [];
      for (let l = 0; l < this.L; l++) {
        this.W.push(new Float64Array(S[l + 1] * S[l]));
        this.b.push(new Float64Array(S[l + 1]));
        this.gW.push(new Float64Array(S[l + 1] * S[l]));
        this.gb.push(new Float64Array(S[l + 1]));
      }
      this.z = S.map((n) => new Float64Array(n));
      this.a = S.map((n) => new Float64Array(n));
      this.delta = S.map((n) => new Float64Array(n));
      this.mask = S.map((n) => new Uint8Array(n).fill(1)); // 0 = ablated (forced to output 0)
      this.gradRms = new Float64Array(this.L);
      this.updateRatio = new Float64Array(this.L);
      this.opt = null;
      this.t = 0;
      this.initMode = cfg.init || 'auto';
      this.seed = cfg.seed || 1;
      this.init(this.initMode, this.seed);
    }

    get paramCount() { let n = 0; for (let l = 0; l < this.L; l++) n += this.W[l].length + this.b[l].length; return n; }
    isOutput(l) { return l === this.L; }

    // Activation object of neuron (l, j); the output layer reports its own squashing function.
    actOf(l, j) {
      if (this.actFns[l]) return this.actFns[l][j];
      if (l === this.L) return this.out === 'sigmoid' ? ACTIVATIONS.sigmoid : ACTIVATIONS.linear;
      return ACTIVATIONS.linear;
    }
    setAct(l, j, id) { if (this.actIds[l] && ACTIVATIONS[id]) { this.actIds[l][j] = id; this.actFns[l][j] = ACTIVATIONS[id]; } }
    // The id shared by every neuron in layer l, or 'mixed'.
    layerActId(l) { const ids = this.actIds[l]; if (!ids) return null; return ids.every((x) => x === ids[0]) ? ids[0] : 'mixed'; }
    isUniform() { for (let l = 1; l < this.L; l++) if (this.layerActId(l) !== this.actId) return false; return true; }
    setLoss(id) { const ok = LOSS_FOR[this.out]; this.lossId = ok.includes(id) ? id : ok[0]; }

    init(mode, seed) {
      const rng = new NA.RNG((seed * 7919 + 17) >>> 0);
      for (let l = 0; l < this.L; l++) {
        const nIn = this.sizes[l], nOut = this.sizes[l + 1], last = l === this.L - 1;
        const fns = this.actFns ? this.actFns[l + 1] : null;
        const rect = fns ? fns.filter((a) => a.he).length * 2 > fns.length : !!this.act.he;
        let sd;
        if (mode === 'small') sd = 0.01;
        else if (mode === 'large') sd = 1.6;
        else if (mode === 'he' || (mode === 'auto' && rect && !last)) sd = Math.sqrt(2 / nIn);
        else sd = Math.sqrt(2 / (nIn + nOut));
        const W = this.W[l], b = this.b[l];
        for (let i = 0; i < W.length; i++) W[i] = rng.gauss(0, sd);
        for (let j = 0; j < b.length; j++) b[j] = last ? 0 : rng.gauss(0, mode === 'small' ? 0.01 : 0.1);
      }
      this.resetOptimizer();
      this.t = 0;
    }

    resetOptimizer(type) {
      const mk = (arrs) => arrs.map((a) => new Float64Array(a.length));
      this.opt = { type: type || null, mW: mk(this.W), vW: mk(this.W), mb: mk(this.b), vb: mk(this.b) };
      this.t = 0;
    }

    // ------------------------------------------------------------ forward
    forward(x, off = 0) {
      const S = this.sizes, L = this.L, a0 = this.a[0], m0 = this.mask[0];
      for (let i = 0; i < S[0]; i++) a0[i] = m0[i] ? x[off + i] : 0;
      for (let l = 0; l < L; l++) {
        const nIn = S[l], nOut = S[l + 1], W = this.W[l], b = this.b[l], aIn = this.a[l], z = this.z[l + 1], a = this.a[l + 1];
        for (let j = 0; j < nOut; j++) {
          let s = b[j];
          const o = j * nIn;
          for (let i = 0; i < nIn; i++) s += W[o + i] * aIn[i];
          z[j] = s;
        }
        if (l < L - 1) {
          const fns = this.actFns[l + 1], m = this.mask[l + 1];
          for (let j = 0; j < nOut; j++) a[j] = m[j] ? fns[j].f(z[j]) : 0;
        } else this._outAct(z, a);
      }
      return this.a[L];
    }

    _outAct(z, a) {
      const n = z.length;
      if (this.out === 'sigmoid') { for (let k = 0; k < n; k++) a[k] = sigmoid(z[k]); }
      else if (this.out === 'softmax') {
        let m = -Infinity; for (let k = 0; k < n; k++) if (z[k] > m) m = z[k];
        let s = 0; for (let k = 0; k < n; k++) { a[k] = Math.exp(z[k] - m); s += a[k]; }
        for (let k = 0; k < n; k++) a[k] /= s;
      } else { for (let k = 0; k < n; k++) a[k] = z[k]; }
    }

    // Loss of the most recent forward pass against target y, under the selected loss function.
    lossOf(y) {
      const zL = this.z[this.L], aL = this.a[this.L], id = this.lossId;
      if (this.out === 'sigmoid') {
        const z = zL[0];
        if (id === 'mse') { const d = aL[0] - y; return 0.5 * d * d; }
        if (id === 'hinge') { const t = y > 0.5 ? 1 : -1; return Math.max(0, 1 - t * z); }
        return Math.max(z, 0) - z * y + Math.log1p(Math.exp(-Math.abs(z)));
      }
      if (this.out === 'softmax') {
        if (id === 'mse') { let s = 0; for (let k = 0; k < aL.length; k++) { const d = aL[k] - (k === (y | 0) ? 1 : 0); s += d * d; } return 0.5 * s; }
        let m = -Infinity; for (let k = 0; k < zL.length; k++) if (zL[k] > m) m = zL[k];
        let s = 0; for (let k = 0; k < zL.length; k++) s += Math.exp(zL[k] - m);
        return m + Math.log(s) - zL[y | 0];
      }
      const r = aL[0] - y, ar = Math.abs(r);
      if (id === 'mae') return ar;
      if (id === 'huber') return ar <= HUBER ? 0.5 * r * r : HUBER * (ar - 0.5 * HUBER);
      return 0.5 * r * r;
    }

    predicted() {
      const aL = this.a[this.L];
      if (this.out === 'sigmoid') return aL[0] >= 0.5 ? 1 : 0;
      if (this.out === 'softmax') { let k = 0; for (let i = 1; i < aL.length; i++) if (aL[i] > aL[k]) k = i; return k; }
      return aL[0];
    }

    // ------------------------------------------------------------ backward
    // Seeds the output error δ = ∂L/∂z for target y. Cross-entropy with sigmoid or softmax, and
    // squared error with a linear output, reduce to ŷ − y; the other pairings keep their extra factors.
    _seed(y) {
      const dL = this.delta[this.L], aL = this.a[this.L], zL = this.z[this.L], id = this.lossId;
      if (this.out === 'softmax') {
        const c = y | 0;
        if (id === 'mse') {
          let s = 0; for (let k = 0; k < dL.length; k++) s += aL[k] * (aL[k] - (k === c ? 1 : 0));
          for (let j = 0; j < dL.length; j++) dL[j] = aL[j] * ((aL[j] - (j === c ? 1 : 0)) - s);
        } else for (let k = 0; k < dL.length; k++) dL[k] = aL[k] - (k === c ? 1 : 0);
      } else if (this.out === 'sigmoid') {
        const p = aL[0];
        if (id === 'mse') dL[0] = (p - y) * p * (1 - p);
        else if (id === 'hinge') { const t = y > 0.5 ? 1 : -1; dL[0] = t * zL[0] < 1 ? -t : 0; }
        else dL[0] = p - y;
      } else {
        const r = aL[0] - y;
        dL[0] = id === 'mae' ? Math.sign(r) : id === 'huber' ? (Math.abs(r) <= HUBER ? r : HUBER * Math.sign(r)) : r;
      }
    }

    // δ[l] = (W[l]ᵀ δ[l+1]) ⊙ f'(z[l]), using each neuron's own activation; for l = 0 this is the
    // gradient with respect to the inputs.
    _propagate(toInput) {
      const S = this.sizes;
      for (let l = this.L - 1; l >= (toInput ? 0 : 1); l--) {
        const nIn = S[l], nOut = S[l + 1], W = this.W[l], dOut = this.delta[l + 1], dIn = this.delta[l], m = this.mask[l];
        const zIn = this.z[l], aIn = this.a[l], fns = this.actFns[l];
        for (let i = 0; i < nIn; i++) {
          if (!m[i]) { dIn[i] = 0; continue; }
          let s = 0;
          for (let j = 0; j < nOut; j++) s += W[j * nIn + i] * dOut[j];
          dIn[i] = l === 0 ? s : s * fns[i].d(zIn[i], aIn[i]);
        }
      }
    }

    backward(y) {
      this._seed(y);
      this._propagate(false);
      const S = this.sizes;
      for (let l = this.L - 1; l >= 0; l--) {
        const nIn = S[l], nOut = S[l + 1], gW = this.gW[l], gb = this.gb[l], dOut = this.delta[l + 1], aIn = this.a[l];
        for (let j = 0; j < nOut; j++) {
          const d = dOut[j];
          if (d === 0) continue;
          gb[j] += d;
          const o = j * nIn;
          for (let i = 0; i < nIn; i++) gW[o + i] += d * aIn[i];
        }
      }
    }

    // Averaged gradient over samples idx[start..end) (plus regulariser). Returns mean data loss.
    computeGrad(X, Y, idx, start, end, reg, lambda) {
      const L = this.L, d0 = this.sizes[0];
      for (let l = 0; l < L; l++) { this.gW[l].fill(0); this.gb[l].fill(0); }
      let loss = 0;
      const n = Math.max(1, end - start);
      for (let s = start; s < end; s++) {
        const k = idx ? idx[s] : s;
        this.forward(X, k * d0);
        loss += this.lossOf(Y[k]);
        this.backward(Y[k]);
      }
      const inv = 1 / n;
      for (let l = 0; l < L; l++) {
        const gW = this.gW[l], gb = this.gb[l], W = this.W[l];
        for (let i = 0; i < gW.length; i++) gW[i] *= inv;
        for (let j = 0; j < gb.length; j++) gb[j] *= inv;
        if (lambda > 0 && reg === 'l2') for (let i = 0; i < gW.length; i++) gW[i] += lambda * W[i];
        else if (lambda > 0 && reg === 'l1') for (let i = 0; i < gW.length; i++) gW[i] += lambda * (W[i] > 0 ? 1 : W[i] < 0 ? -1 : 0);
      }
      return loss * inv;
    }

    regPenalty(reg, lambda) {
      if (!(lambda > 0) || reg === 'none') return 0;
      let s = 0;
      for (let l = 0; l < this.L; l++) { const W = this.W[l]; for (let i = 0; i < W.length; i++) s += reg === 'l2' ? 0.5 * W[i] * W[i] : Math.abs(W[i]); }
      return lambda * s;
    }

    // Apply the gradients currently in gW/gb with the chosen optimizer.
    step(h) {
      const type = h.optimizer || 'adam', lr = h.lr;
      if (!this.opt || this.opt.type !== type) this.resetOptimizer(type);
      const o = this.opt;
      this.t++;
      const b1 = 0.9, b2 = 0.999, eps = 1e-8, mu = 0.9;
      const c1 = 1 - Math.pow(b1, this.t), c2 = 1 - Math.pow(b2, this.t);
      for (let l = 0; l < this.L; l++) {
        let du2 = 0, w2 = 0, g2 = 0;
        for (let pass = 0; pass < 2; pass++) {
          const P = pass ? this.b[l] : this.W[l], G = pass ? this.gb[l] : this.gW[l];
          const M = pass ? o.mb[l] : o.mW[l], V = pass ? o.vb[l] : o.vW[l];
          for (let i = 0; i < P.length; i++) {
            const g = G[i];
            let u;
            if (type === 'sgd') u = -lr * g;
            else if (type === 'momentum') { M[i] = mu * M[i] - lr * g; u = M[i]; }
            else { M[i] = b1 * M[i] + (1 - b1) * g; V[i] = b2 * V[i] + (1 - b2) * g * g; u = (-lr * (M[i] / c1)) / (Math.sqrt(V[i] / c2) + eps); }
            if (!pass) { du2 += u * u; w2 += P[i] * P[i]; g2 += g * g; }
            P[i] += u;
          }
        }
        this.gradRms[l] = Math.sqrt(g2 / this.W[l].length);
        this.updateRatio[l] = Math.sqrt(du2) / (Math.sqrt(w2) + 1e-12);
      }
    }

    trainBatch(X, Y, idx, start, end, h) {
      const loss = this.computeGrad(X, Y, idx, start, end, h.reg, h.regRate);
      this.step(h);
      return loss;
    }

    /* Full-batch gradient without updating anything: averaged ∂L/∂W, ∂L/∂b and the
       root-mean-square error signal δ reaching every neuron. */
    gradientReport(X, Y, idx, reg, lambda) {
      const S = this.sizes, L = this.L, d0 = S[0], n = idx.length;
      const gW = this.W.map((w) => new Float64Array(w.length)), gb = this.b.map((b) => new Float64Array(b.length));
      const d2 = S.map((m) => new Float64Array(m));
      let loss = 0;
      for (let s = 0; s < n; s++) {
        const k = idx[s];
        this.forward(X, k * d0);
        loss += this.lossOf(Y[k]);
        this._seed(Y[k]);
        this._propagate(true);
        for (let l = 0; l <= L; l++) { const d = this.delta[l], acc = d2[l]; for (let j = 0; j < S[l]; j++) acc[j] += d[j] * d[j]; }
        for (let l = 0; l < L; l++) {
          const nIn = S[l], nOut = S[l + 1], G = gW[l], B = gb[l], dOut = this.delta[l + 1], aIn = this.a[l];
          for (let j = 0; j < nOut; j++) {
            const d = dOut[j]; if (d === 0) continue;
            B[j] += d; const o = j * nIn;
            for (let i = 0; i < nIn; i++) G[o + i] += d * aIn[i];
          }
        }
      }
      const inv = 1 / Math.max(1, n);
      let gMax = 0;
      for (let l = 0; l < L; l++) {
        const G = gW[l], B = gb[l], W = this.W[l];
        for (let i = 0; i < G.length; i++) {
          G[i] *= inv;
          if (lambda > 0 && reg === 'l2') G[i] += lambda * W[i];
          else if (lambda > 0 && reg === 'l1') G[i] += lambda * Math.sign(W[i]);
          const v = Math.abs(G[i]); if (v > gMax) gMax = v;
        }
        for (let j = 0; j < B.length; j++) B[j] *= inv;
      }
      const dRms = d2.map((acc) => acc.map((v) => Math.sqrt(v * inv)));
      let dMax = 0; for (let l = 1; l <= L; l++) for (const v of dRms[l]) if (v > dMax) dMax = v;
      const layerRms = gW.map((G) => { let s = 0; for (let i = 0; i < G.length; i++) s += G[i] * G[i]; return Math.sqrt(s / Math.max(1, G.length)); });
      return { gW, gb, dRms, gMax, dMax, layerRms, loss: loss * inv };
    }

    /* Knock out each non-output neuron in turn and measure the damage on a sample set. */
    importance(X, Y, idx, task) {
      const base = this.evaluate(X, Y, idx, task), out = [];
      for (let l = 0; l < this.L; l++) {
        const row = [];
        for (let j = 0; j < this.sizes[l]; j++) {
          if (!this.mask[l][j]) { row.push({ dLoss: 0, dAcc: 0, off: true }); continue; }
          this.mask[l][j] = 0;
          const r = this.evaluate(X, Y, idx, task);
          this.mask[l][j] = 1;
          row.push({ dLoss: r.loss - base.loss, dAcc: r.acc - base.acc, off: false });
        }
        out.push(row);
      }
      return { base, layers: out };
    }

    // Full trace of one sample: copies of z, a and δ for every layer, plus loss.
    trace(x, off, y) {
      this.forward(x, off);
      const hasY = y !== undefined && y !== null && !Number.isNaN(y);
      let loss = NaN;
      if (hasY) { loss = this.lossOf(y); this._seed(y); this._propagate(true); }
      return {
        z: this.z.map((v) => Float64Array.from(v)),
        a: this.a.map((v) => Float64Array.from(v)),
        delta: hasY ? this.delta.map((v) => Float64Array.from(v)) : null,
        loss, y: hasY ? y : null,
      };
    }

    // Gradient of output pre-activation k with respect to the input vector.
    inputGradient(x, off, k) {
      this.forward(x, off);
      const dL = this.delta[this.L];
      dL.fill(0); dL[k] = 1;
      this._propagate(true);
      return Float64Array.from(this.delta[0]);
    }

    evaluate(X, Y, idx, task) {
      const d0 = this.sizes[0], n = idx.length;
      if (!n) return { loss: NaN, acc: NaN };
      let loss = 0, correct = 0, sy = 0, syy = 0, sse = 0;
      for (let s = 0; s < n; s++) {
        const k = idx[s];
        this.forward(X, k * d0);
        loss += this.lossOf(Y[k]);
        if (task === 'regression') { const d = this.a[this.L][0] - Y[k]; sse += d * d; sy += Y[k]; syy += Y[k] * Y[k]; }
        else if (this.predicted() === Y[k]) correct++;
      }
      if (task === 'regression') { const sst = syy - (sy * sy) / n; return { loss: loss / n, acc: sst > 1e-12 ? 1 - sse / sst : 0 }; }
      return { loss: loss / n, acc: correct / n };
    }

    /* Forward many samples, storing every layer neuron-major: out.a[l][j*n + p]. */
    forwardMany(X, n, out) {
      const S = this.sizes, L = this.L, d0 = S[0];
      if (!out || out.n !== n || !out.sizes || out.sizes.join() !== S.join()) {
        out = { n, sizes: S.slice(), a: S.map((m) => new Float32Array(m * n)), z: S.map((m) => new Float32Array(m * n)) };
      }
      for (let p = 0; p < n; p++) {
        this.forward(X, p * d0);
        for (let l = 0; l <= L; l++) {
          const al = this.a[l], zl = this.z[l], A = out.a[l], Z = out.z[l], m = S[l];
          for (let j = 0; j < m; j++) { A[j * n + p] = al[j]; Z[j * n + p] = l === 0 ? al[j] : zl[j]; }
        }
      }
      return out;
    }

    getParams() {
      const out = new Float32Array(this.paramCount);
      let o = 0;
      for (let l = 0; l < this.L; l++) { out.set(this.W[l], o); o += this.W[l].length; out.set(this.b[l], o); o += this.b[l].length; }
      return out;
    }
    setParams(p) {
      let o = 0;
      for (let l = 0; l < this.L; l++) {
        const W = this.W[l], b = this.b[l];
        for (let i = 0; i < W.length; i++) W[i] = p[o++];
        for (let j = 0; j < b.length; j++) b[j] = p[o++];
      }
    }
    clone() {
      const c = new MLP({ sizes: this.sizes, act: this.actId, acts: this.actIds, out: this.out, loss: this.lossId, init: this.initMode, seed: this.seed });
      c.setParams(this.getParams());
      c.mask = this.mask.map((m) => Uint8Array.from(m));
      return c;
    }
  }

  // ------------------------------------------------------------ statistics
  /* Per-neuron statistics over a forwardMany() result, optionally restricted to sample indices idx.
     activeFrac: share of inputs with z > 0. satFrac: share on the flat tails of the neuron's activation. */
  function neuronStats(net, F, idx) {
    const N = F.n, n = idx ? idx.length : N;
    return net.sizes.map((m, l) => {
      const A = F.a[l], Z = F.z[l], hidden = l > 0 && l < net.L;
      const out = [];
      const buf = new Float32Array(n);
      for (let j = 0; j < m; j++) {
        const act = hidden ? net.actFns[l][j] : null;
        let s = 0, ss = 0, mn = Infinity, mx = -Infinity, active = 0, sat = 0;
        for (let q = 0; q < n; q++) {
          const p = idx ? idx[q] : q;
          const v = A[j * N + p];
          buf[q] = v; s += v; ss += v * v;
          if (v < mn) mn = v; if (v > mx) mx = v;
          if (hidden) {
            if (Z[j * N + p] > 0) active++;
            if (act.sat && act.sat(v)) sat++;
          }
        }
        buf.sort();
        const mean = s / n;
        out.push({
          mean, std: Math.sqrt(Math.max(0, ss / n - mean * mean)), min: mn, max: mx,
          activeFrac: hidden ? active / n : 1, satFrac: hidden ? sat / n : 0,
          q: [0.05, 0.25, 0.5, 0.75, 0.95].map((q) => NA.quantile(buf, q)),
        });
      }
      return out;
    });
  }

  // Pearson correlation between two neurons' activations (neuron-major arrays).
  function correlation(A, n, j1, j2) {
    let s1 = 0, s2 = 0, s11 = 0, s22 = 0, s12 = 0;
    for (let p = 0; p < n; p++) { const x = A[j1 * n + p], y = A[j2 * n + p]; s1 += x; s2 += y; s11 += x * x; s22 += y * y; s12 += x * y; }
    const c = s12 / n - (s1 / n) * (s2 / n), v1 = s11 / n - (s1 / n) ** 2, v2 = s22 / n - (s2 / n) ** 2;
    return v1 > 1e-12 && v2 > 1e-12 ? c / Math.sqrt(v1 * v2) : 0;
  }

  // ------------------------------------------------------------ PCA (top two components)
  /* arr is neuron-major: arr[j*n + p]. Returns projection coords (Float32Array n*2) and explained variance. */
  function pca2(arr, n, d) {
    const mean = new Float64Array(d);
    for (let j = 0; j < d; j++) { let s = 0; for (let p = 0; p < n; p++) s += arr[j * n + p]; mean[j] = s / n; }
    const C = new Float64Array(d * d);
    for (let i = 0; i < d; i++) {
      for (let j = i; j < d; j++) {
        let s = 0; const mi = mean[i], mj = mean[j];
        for (let p = 0; p < n; p++) s += (arr[i * n + p] - mi) * (arr[j * n + p] - mj);
        C[i * d + j] = C[j * d + i] = s / n;
      }
    }
    let total = 0; for (let i = 0; i < d; i++) total += C[i * d + i];
    const comps = [], vals = [];
    const M = Float64Array.from(C);
    for (let c = 0; c < Math.min(2, d); c++) {
      let v = new Float64Array(d);
      for (let i = 0; i < d; i++) v[i] = 1 + 0.37 * Math.sin(1.7 * i + c * 2.3);
      let lambda = 0;
      for (let it = 0; it < 120; it++) {
        const w = new Float64Array(d);
        for (let i = 0; i < d; i++) { let s = 0; for (let j = 0; j < d; j++) s += M[i * d + j] * v[j]; w[i] = s; }
        let norm = 0; for (let i = 0; i < d; i++) norm += w[i] * w[i];
        norm = Math.sqrt(norm);
        if (norm < 1e-14) break;
        for (let i = 0; i < d; i++) w[i] /= norm;
        lambda = norm; v = w;
      }
      comps.push(v); vals.push(lambda);
      for (let i = 0; i < d; i++) for (let j = 0; j < d; j++) M[i * d + j] -= lambda * v[i] * v[j];
    }
    while (comps.length < 2) { comps.push(new Float64Array(d)); vals.push(0); }
    const proj = new Float32Array(n * 2);
    for (let p = 0; p < n; p++) {
      let x = 0, y = 0;
      for (let j = 0; j < d; j++) { const v = arr[j * n + p] - mean[j]; x += v * comps[0][j]; y += v * comps[1][j]; }
      proj[p * 2] = x; proj[p * 2 + 1] = y;
    }
    return { proj, mean, comps, explained: vals.map((v) => (total > 0 ? v / total : 0)), project(vec) {
      let x = 0, y = 0; for (let j = 0; j < d; j++) { const v = vec[j] - mean[j]; x += v * comps[0][j]; y += v * comps[1][j]; } return [x, y];
    } };
  }

  /* Orthogonal Procrustes in 2-D: rotate (and maybe reflect) points A to best match B. Both n×2. */
  function procrustes2(A, B, n) {
    let ax = 0, ay = 0, bx = 0, by = 0;
    for (let p = 0; p < n; p++) { ax += A[2 * p]; ay += A[2 * p + 1]; bx += B[2 * p]; by += B[2 * p + 1]; }
    ax /= n; ay /= n; bx /= n; by /= n;
    let sxx = 0, sxy = 0, syx = 0, syy = 0;
    for (let p = 0; p < n; p++) {
      const x1 = A[2 * p] - ax, y1 = A[2 * p + 1] - ay, x2 = B[2 * p] - bx, y2 = B[2 * p + 1] - by;
      sxx += x1 * x2; sxy += x1 * y2; syx += y1 * x2; syy += y1 * y2;
    }
    const rot = Math.hypot(sxx + syy, sxy - syx), ref = Math.hypot(sxx - syy, sxy + syx);
    const reflect = ref > rot;
    const th = reflect ? Math.atan2(sxy + syx, sxx - syy) : Math.atan2(sxy - syx, sxx + syy);
    const c = Math.cos(th), s = Math.sin(th);
    return {
      reflect, theta: th,
      apply(x, y) { if (reflect) y = -y; return [x * c - y * s, x * s + y * c]; },
    };
  }

  // ------------------------------------------------------------ contours (marching squares)
  /* values: gw×gh row-major grid. Returns flat array of segments [x1,y1,x2,y2,...] in grid units. */
  function contour(values, gw, gh, level) {
    const segs = [];
    for (let j = 0; j < gh - 1; j++) {
      for (let i = 0; i < gw - 1; i++) {
        const tl = values[j * gw + i], tr = values[j * gw + i + 1], br = values[(j + 1) * gw + i + 1], bl = values[(j + 1) * gw + i];
        const c = (tl > level ? 8 : 0) | (tr > level ? 4 : 0) | (br > level ? 2 : 0) | (bl > level ? 1 : 0);
        if (c === 0 || c === 15) continue;
        const top = () => [i + (level - tl) / (tr - tl), j];
        const right = () => [i + 1, j + (level - tr) / (br - tr)];
        const bottom = () => [i + (level - bl) / (br - bl), j + 1];
        const left = () => [i, j + (level - tl) / (bl - tl)];
        const add = (p, q) => segs.push(p[0], p[1], q[0], q[1]);
        const center = (tl + tr + br + bl) / 4 > level;
        switch (c) {
          case 1: case 14: add(left(), bottom()); break;
          case 2: case 13: add(bottom(), right()); break;
          case 3: case 12: add(left(), right()); break;
          case 4: case 11: add(top(), right()); break;
          case 6: case 9: add(top(), bottom()); break;
          case 7: case 8: add(left(), top()); break;
          case 5: if (center) { add(left(), top()); add(bottom(), right()); } else { add(top(), right()); add(left(), bottom()); } break;
          case 10: if (center) { add(top(), right()); add(left(), bottom()); } else { add(left(), top()); add(bottom(), right()); } break;
        }
      }
    }
    return segs;
  }

  // Is neuron (l, j) dead or flat over the statistics? ReLU dies when z ≤ 0 everywhere; anything else is
  // reported as flat when its output barely varies.
  function deadKind(net, stats, l, j) {
    if (l <= 0 || l >= net.L || !net.mask[l][j]) return null;
    const s = stats[l][j], act = net.actFns[l][j];
    if (act.canDie) return s.activeFrac === 0 ? 'dead' : null;
    return s.std < 1e-4 ? 'flat' : null;
  }

  NA.engine = { MLP, ACTIVATIONS, ACTIVATION_ORDER, LOSSES, LOSS_FOR, HUBER, OPTIMIZERS, neuronStats, deadKind, correlation, pca2, procrustes2, contour, sigmoid };
})(typeof globalThis !== 'undefined' ? (globalThis.NA = globalThis.NA || {}) : (window.NA = window.NA || {}));
