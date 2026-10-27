/* diffusion.js — a denoising diffusion model (DDPM) for 2-D point clouds.
   Forward process: x_t = √ᾱ_t · x₀ + √(1 − ᾱ_t) · ε, with a cosine noise schedule over T steps.
   A small SiLU MLP sees (x_t, features of t) and predicts ε; training minimises ‖ε − ε̂‖².
   Sampling runs the process backwards from pure noise, either stochastically (DDPM) or
   deterministically (DDIM). Gradients are checked in tools/diffusion-gradcheck.js. */
(function (NA) {
  'use strict';

  const silu = (z) => z / (1 + Math.exp(-z));
  const siluGrad = (z) => { const s = 1 / (1 + Math.exp(-z)); return s * (1 + z * (1 - s)); };

  /* Fully connected network on a batch: SiLU hidden layers, linear output. Row-major activations. */
  class DenseNet {
    constructor(sizes, seed = 1, f64 = false) {
      const A = f64 ? Float64Array : Float32Array;
      this.sizes = sizes; this.L = sizes.length - 1; this.A = A;
      this.W = []; this.b = []; this.gW = []; this.gb = [];
      for (let l = 0; l < this.L; l++) {
        this.W.push(new A(sizes[l] * sizes[l + 1])); this.b.push(new A(sizes[l + 1]));
        this.gW.push(new A(sizes[l] * sizes[l + 1])); this.gb.push(new A(sizes[l + 1]));
      }
      this.cap = 0;
      this.paramCount = this.W.reduce((s, w) => s + w.length, 0) + this.b.reduce((s, b) => s + b.length, 0);
      const rng = new NA.RNG(seed * 6151 + 3);
      for (let l = 0; l < this.L; l++) {
        const sd = Math.sqrt((l === this.L - 1 ? 1 : 2) / sizes[l]);
        for (let i = 0; i < this.W[l].length; i++) this.W[l][i] = rng.gauss(0, sd * (l === this.L - 1 ? 0.5 : 1));
      }
      this.t = 0; this.opt = null;
    }
    ensure(n) {
      if (n <= this.cap) return;
      this.cap = n;
      this.z = this.sizes.map((m) => new this.A(n * m)); this.a = this.sizes.map((m) => new this.A(n * m));
      this.ga = this.sizes.map((m) => new this.A(n * m));
    }
    // X: n×in (row-major). Returns the n×out output buffer (valid until the next call).
    forward(X, n) {
      this.ensure(n);
      this.a[0].set(X.subarray(0, n * this.sizes[0]));
      for (let l = 0; l < this.L; l++) {
        const nIn = this.sizes[l], nOut = this.sizes[l + 1], W = this.W[l], b = this.b[l], x = this.a[l], z = this.z[l + 1], y = this.a[l + 1], last = l === this.L - 1;
        for (let p = 0; p < n; p++) {
          const xo = p * nIn, zo = p * nOut;
          for (let j = 0; j < nOut; j++) z[zo + j] = b[j];
          for (let i = 0; i < nIn; i++) {
            const xv = x[xo + i];
            if (xv === 0) continue;
            const wo = i * nOut;
            for (let j = 0; j < nOut; j++) z[zo + j] += xv * W[wo + j];
          }
          if (last) for (let j = 0; j < nOut; j++) y[zo + j] = z[zo + j];
          else for (let j = 0; j < nOut; j++) y[zo + j] = silu(z[zo + j]);
        }
      }
      return this.a[this.L];
    }
    // dOut: n×out gradient of the loss with respect to the output; adds into gW, gb.
    backward(dOut, n) {
      this.ga[this.L].set(dOut.subarray(0, n * this.sizes[this.L]));
      for (let l = this.L - 1; l >= 0; l--) {
        const nIn = this.sizes[l], nOut = this.sizes[l + 1], W = this.W[l], gW = this.gW[l], gb = this.gb[l], x = this.a[l], z = this.z[l + 1], gy = this.ga[l + 1], gx = this.ga[l], last = l === this.L - 1;
        if (!last) for (let k = 0; k < n * nOut; k++) gy[k] *= siluGrad(z[k]);
        for (let p = 0; p < n; p++) {
          const xo = p * nIn, yo = p * nOut;
          for (let j = 0; j < nOut; j++) gb[j] += gy[yo + j];
          for (let i = 0; i < nIn; i++) {
            const wo = i * nOut, xv = x[xo + i];
            let s = 0;
            for (let j = 0; j < nOut; j++) { const g = gy[yo + j]; gW[wo + j] += xv * g; s += g * W[wo + j]; }
            gx[xo + i] = s;
          }
        }
      }
    }
    zeroGrad() { for (let l = 0; l < this.L; l++) { this.gW[l].fill(0); this.gb[l].fill(0); } }
    adam(lr) {
      const b1 = 0.9, b2 = 0.999, eps = 1e-8;
      if (!this.opt) this.opt = [...this.W, ...this.b].map((w) => ({ m: new this.A(w.length), v: new this.A(w.length) }));
      this.t++;
      lr *= Math.min(1, this.t / 200); // warm-up, so continuing from pretrained weights does not jolt them
      const c1 = 1 - Math.pow(b1, this.t), c2 = 1 - Math.pow(b2, this.t);
      [...this.W, ...this.b].forEach((w, k) => {
        const g = k < this.L ? this.gW[k] : this.gb[k - this.L], { m, v } = this.opt[k];
        for (let i = 0; i < w.length; i++) {
          m[i] = b1 * m[i] + (1 - b1) * g[i]; v[i] = b2 * v[i] + (1 - b2) * g[i] * g[i];
          w[i] -= (lr * (m[i] / c1)) / (Math.sqrt(v[i] / c2) + eps);
        }
      });
    }
    getParams() { const out = []; for (let l = 0; l < this.L; l++) out.push(...this.W[l], ...this.b[l]); return Float32Array.from(out); }
    setParams(flat) { let o = 0; for (let l = 0; l < this.L; l++) { this.W[l].set(flat.subarray(o, o + this.W[l].length)); o += this.W[l].length; this.b[l].set(flat.subarray(o, o + this.b[l].length)); o += this.b[l].length; } }
  }

  // Cosine schedule (Nichol & Dhariwal): ᾱ(t) = cos²(((t/T + s)/(1 + s))·π/2), β clipped to 0.5.
  function cosineSchedule(T, s = 0.008) {
    const f = (t) => Math.cos((((t / T) + s) / (1 + s)) * Math.PI / 2) ** 2;
    const abar = new Float64Array(T + 1), beta = new Float64Array(T + 1), alpha = new Float64Array(T + 1);
    abar[0] = 1; alpha[0] = 1;
    for (let t = 1; t <= T; t++) {
      beta[t] = Math.min(0.5, 1 - f(t) / f(t - 1));
      alpha[t] = 1 - beta[t];
      abar[t] = abar[t - 1] * alpha[t];
    }
    return { T, beta, alpha, abar };
  }

  const K_FREQ = 4;
  const IN_DIM = 2 + 1 + 2 * K_FREQ;
  // Network input for a point (x, y) at step t: the point, t/T, and sines and cosines of t at four frequencies.
  function writeInput(X, o, x, y, t, T) {
    const tau = t / T;
    X[o] = x; X[o + 1] = y; X[o + 2] = tau * 2 - 1;
    for (let k = 0; k < K_FREQ; k++) { const w = Math.PI * Math.pow(2, k) * tau; X[o + 3 + 2 * k] = Math.sin(w); X[o + 4 + 2 * k] = Math.cos(w); }
  }

  class Diffusion {
    /* o: { T steps, hidden [widths], seed } */
    constructor(o = {}) {
      this.o = Object.assign({ T: 100, hidden: [64, 64, 64], seed: 1 }, o);
      this.sched = cosineSchedule(this.o.T);
      this.net = new DenseNet([IN_DIM, ...this.o.hidden, 2], this.o.seed, !!this.o.f64);
      this.rng = new NA.RNG(this.o.seed * 97 + 5);
      this.X = new Float32Array(0); this.target = new Float32Array(0);
    }
    get T() { return this.o.T; }
    buf(n) { if (this.X.length < n * IN_DIM) { this.X = new Float32Array(n * IN_DIM); this.target = new Float32Array(n * 2); this.dOut = new Float32Array(n * 2); } }

    /* One step on a batch drawn from data (Float32Array of n points, already standardised). Returns the loss
       and, per sample, its t and squared error (for the loss-by-noise-level chart). */
    trainStep(data, nData, batch = 128, lr = 2e-3) {
      const r = this.rng, T = this.o.T, ab = this.sched.abar;
      this.buf(batch);
      const ts = new Int32Array(batch);
      for (let p = 0; p < batch; p++) {
        const k = r.int(nData), t = 1 + r.int(T), a = Math.sqrt(ab[t]), s = Math.sqrt(1 - ab[t]);
        const e1 = r.gauss(), e2 = r.gauss();
        writeInput(this.X, p * IN_DIM, a * data[2 * k] + s * e1, a * data[2 * k + 1] + s * e2, t, T);
        this.target[2 * p] = e1; this.target[2 * p + 1] = e2; ts[p] = t;
      }
      const out = this.net.forward(this.X, batch);
      let loss = 0;
      const per = new Float32Array(batch);
      for (let p = 0; p < batch; p++) {
        const d1 = out[2 * p] - this.target[2 * p], d2 = out[2 * p + 1] - this.target[2 * p + 1];
        per[p] = (d1 * d1 + d2 * d2) / 2; loss += per[p];
        this.dOut[2 * p] = d1 / batch; this.dOut[2 * p + 1] = d2 / batch;
      }
      this.net.zeroGrad();
      this.net.backward(this.dOut, batch);
      this.net.adam(lr);
      return { loss: loss / batch, ts, per };
    }

    // Per-sample loss on a fresh batch, without training (for the loss-by-noise-level chart).
    evalBatch(data, nData, batch, rng) {
      const T = this.o.T, ab = this.sched.abar, ts = new Int32Array(batch), per = new Float32Array(batch);
      this.buf(batch);
      for (let p = 0; p < batch; p++) {
        const k = rng.int(nData), t = 1 + rng.int(T), a = Math.sqrt(ab[t]), s = Math.sqrt(1 - ab[t]), e1 = rng.gauss(), e2 = rng.gauss();
        writeInput(this.X, p * IN_DIM, a * data[2 * k] + s * e1, a * data[2 * k + 1] + s * e2, t, T);
        this.target[2 * p] = e1; this.target[2 * p + 1] = e2; ts[p] = t;
      }
      const out = this.net.forward(this.X, batch);
      for (let p = 0; p < batch; p++) { const d1 = out[2 * p] - this.target[2 * p], d2 = out[2 * p + 1] - this.target[2 * p + 1]; per[p] = (d1 * d1 + d2 * d2) / 2; }
      return { ts, per };
    }

    // ε̂ for n points (xy: 2n) at step t. Returns a Float32Array of 2n.
    predict(xy, n, t) {
      this.buf(n);
      for (let p = 0; p < n; p++) writeInput(this.X, p * IN_DIM, xy[2 * p], xy[2 * p + 1], t, this.o.T);
      return Float32Array.from(this.net.forward(this.X, n).subarray(0, 2 * n));
    }

    /* One reverse step for every point, in place, from t to tPrev (default t − 1; larger jumps sample in
       fewer steps). ddim: deterministic update. Returns the predicted clean points x̂₀ (useful for drawing). */
    reverseStep(xy, n, t, ddim, rng, tPrev = t - 1) {
      const S = this.sched, eps = this.predict(xy, n, t), ab = S.abar[t], abPrev = S.abar[tPrev], x0 = new Float32Array(2 * n);
      const alphaEff = ab / abPrev, betaEff = 1 - alphaEff;
      for (let p = 0; p < 2 * n; p++) {
        let x0p = (xy[p] - Math.sqrt(1 - ab) * eps[p]) / Math.sqrt(ab);
        x0p = Math.max(-4, Math.min(4, x0p));
        x0[p] = x0p;
        const epsC = (xy[p] - Math.sqrt(ab) * x0p) / Math.sqrt(1 - ab); // ε consistent with the clipped x̂₀
        if (ddim || tPrev === 0) xy[p] = Math.sqrt(abPrev) * x0p + Math.sqrt(1 - abPrev) * epsC;
        else {
          // posterior mean of q(x_tPrev | x_t, x̂₀) plus noise with its variance
          const c0 = (Math.sqrt(abPrev) * betaEff) / (1 - ab), ct = (Math.sqrt(alphaEff) * (1 - abPrev)) / (1 - ab);
          const varT = (betaEff * (1 - abPrev)) / (1 - ab);
          xy[p] = c0 * x0p + ct * xy[p] + Math.sqrt(varT) * rng.gauss();
        }
      }
      return x0;
    }

    // Forward process applied to clean points: x_t for a fixed noise draw (eps: 2n).
    noised(x0, eps, n, t) {
      const ab = this.sched.abar[t], a = Math.sqrt(ab), s = Math.sqrt(1 - ab), out = new Float32Array(2 * n);
      for (let p = 0; p < 2 * n; p++) out[p] = a * x0[p] + s * eps[p];
      return out;
    }
  }

  /* Sample quality: how far generated points sit from the data (precision) and how well they cover it
     (coverage), as mean nearest-neighbour distances in standardised units. */
  function chamfer(gen, nGen, data, nData) {
    const nn = (A, nA, B, nB) => {
      let s = 0;
      for (let i = 0; i < nA; i++) {
        let best = Infinity;
        const x = A[2 * i], y = A[2 * i + 1];
        for (let j = 0; j < nB; j++) { const dx = B[2 * j] - x, dy = B[2 * j + 1] - y, d = dx * dx + dy * dy; if (d < best) best = d; }
        s += Math.sqrt(best);
      }
      return s / Math.max(1, nA);
    };
    return { precision: nn(gen, nGen, data, nData), coverage: nn(data, nData, gen, nGen) };
  }

  NA.diffusion = { Diffusion, DenseNet, cosineSchedule, chamfer, IN_DIM };
})(typeof globalThis !== 'undefined' ? (globalThis.NA = globalThis.NA || {}) : (window.NA = window.NA || {}));
