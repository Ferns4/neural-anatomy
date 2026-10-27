/* pinn.js — a physics-informed neural network for the damped harmonic oscillator
       u″ + 2ζω u′ + ω² u = 0,   u(0) = 1,  u′(0) = 0,   t ∈ [0, 1].
   The network u(t) is a small tanh MLP. Its first and second derivatives with respect to t are carried
   through every layer alongside the value (forward-mode, exact), so the ODE residual can be part of the loss;
   the backward pass then differentiates through all three streams. Checked in tools/pinn-gradcheck.js. */
(function (NA) {
  'use strict';

  // Exact solution of the underdamped oscillator, and its derivative.
  function exact(t, zeta, omega) {
    const wd = omega * Math.sqrt(1 - zeta * zeta), a = zeta * omega, e = Math.exp(-a * t);
    return e * (Math.cos(wd * t) + (a / wd) * Math.sin(wd * t));
  }
  function exactD(t, zeta, omega) {
    const wd = omega * Math.sqrt(1 - zeta * zeta), a = zeta * omega;
    return -Math.exp(-a * t) * ((a * a) / wd + wd) * Math.sin(wd * t);
  }

  class PINN {
    /* sizes: [1, h, h, h, 1]; tanh hidden layers, linear output. The input is t mapped to [-1, 1]. */
    constructor(sizes = [1, 32, 32, 32, 1], seed = 1, f64 = false) {
      const A = f64 ? Float64Array : Float32Array;
      this.sizes = sizes; this.L = sizes.length - 1; this.A = A;
      this.W = []; this.b = []; this.gW = []; this.gb = [];
      for (let l = 0; l < this.L; l++) {
        this.W.push(new A(sizes[l + 1] * sizes[l])); this.b.push(new A(sizes[l + 1]));
        this.gW.push(new A(sizes[l + 1] * sizes[l])); this.gb.push(new A(sizes[l + 1]));
      }
      // per-layer value / first / second derivative caches (one point at a time)
      this.a = sizes.map((n) => [new A(n), new A(n), new A(n)]);
      this.ga = sizes.map((n) => [new A(n), new A(n), new A(n)]);
      this.zd = sizes.map((n) => [new A(n), new A(n)]); // dz/dt and d²z/dt² of each hidden pre-activation
      this.paramCount = this.W.reduce((s, w) => s + w.length, 0) + this.b.reduce((s, b) => s + b.length, 0);
      this.init(seed);
    }
    init(seed) {
      const rng = new NA.RNG(seed * 104729 + 7);
      for (let l = 0; l < this.L; l++) {
        const nIn = this.sizes[l], nOut = this.sizes[l + 1], sd = Math.sqrt(2 / (nIn + nOut)); // Xavier for tanh
        for (let i = 0; i < this.W[l].length; i++) this.W[l][i] = rng.gauss(0, sd);
        this.b[l].fill(0);
      }
      this.t = 0; this.opt = null;
    }

    /* Value, du/dt and d²u/dt² at time t. With order 0 only the value is computed. */
    forward(t, order = 2) {
      const s = 2; // input x = 2t − 1, so dx/dt = 2
      const a0 = this.a[0];
      a0[0][0] = 2 * t - 1; a0[1][0] = s; a0[2][0] = 0;
      for (let l = 0; l < this.L; l++) {
        const W = this.W[l], b = this.b[l], nIn = this.sizes[l], nOut = this.sizes[l + 1], last = l === this.L - 1;
        const [x, x1, x2] = this.a[l], [y, y1, y2] = this.a[l + 1];
        for (let j = 0; j < nOut; j++) {
          let z = b[j], z1 = 0, z2 = 0;
          const o = j * nIn;
          for (let i = 0; i < nIn; i++) { const w = W[o + i]; z += w * x[i]; if (order) { z1 += w * x1[i]; z2 += w * x2[i]; } }
          if (last) { y[j] = z; y1[j] = z1; y2[j] = z2; continue; }
          const v = Math.tanh(z), d = 1 - v * v;
          y[j] = v; this.zd[l + 1][0][j] = z1; this.zd[l + 1][1][j] = z2;
          if (order) { y1[j] = d * z1; y2[j] = -2 * v * d * z1 * z1 + d * z2; }
        }
      }
      const out = this.a[this.L];
      return [out[0][0], out[1][0], out[2][0]];
    }

    /* After forward(t): adds gU·∂u/∂θ + g1·∂u′/∂θ + g2·∂u″/∂θ to the parameter gradients. */
    backward(gU, g1, g2) {
      const L = this.L, top = this.ga[L];
      top[0][0] = gU; top[1][0] = g1; top[2][0] = g2;
      for (let l = L - 1; l >= 0; l--) {
        const W = this.W[l], nIn = this.sizes[l], nOut = this.sizes[l + 1], last = l === L - 1;
        const [x, x1, x2] = this.a[l], [y, y1, y2] = this.a[l + 1], [gy, gy1, gy2] = this.ga[l + 1], [gx, gx1, gx2] = this.ga[l];
        gx.fill(0); gx1.fill(0); gx2.fill(0);
        const gW = this.gW[l], gb = this.gb[l];
        for (let j = 0; j < nOut; j++) {
          let gz, gz1, gz2;
          if (last) { gz = gy[j]; gz1 = gy1[j]; gz2 = gy2[j]; }
          else {
            // y = tanh(z), y1 = σ′ z1, y2 = σ″ z1² + σ′ z2, with σ′ = 1 − v², σ″ = −2vσ′, σ‴ = σ′(4v² − 2σ′)
            const v = y[j], d1 = 1 - v * v, d2 = -2 * v * d1, d3 = d1 * (4 * v * v - 2 * d1);
            const z1 = this.zd[l + 1][0][j], z2 = this.zd[l + 1][1][j];
            gz2 = gy2[j] * d1;
            gz1 = gy1[j] * d1 + gy2[j] * 2 * d2 * z1;
            gz = gy[j] * d1 + gy1[j] * d2 * z1 + gy2[j] * (d3 * z1 * z1 + d2 * z2);
          }
          gb[j] += gz;
          const o = j * nIn;
          for (let i = 0; i < nIn; i++) {
            const w = W[o + i];
            gW[o + i] += gz * x[i] + gz1 * x1[i] + gz2 * x2[i];
            gx[i] += w * gz; gx1[i] += w * gz1; gx2[i] += w * gz2;
          }
        }
      }
    }

    zeroGrad() { for (let l = 0; l < this.L; l++) { this.gW[l].fill(0); this.gb[l].fill(0); } }

    /* One Adam step. problem: { zeta, omega }, data: [{t, u}], colloc: [t...], w: { data, phys, ic }.
       Physics residual is scaled by 1/ω² so every term is of order 1. Returns the loss terms. */
    trainStep(problem, data, colloc, w, lr = 2e-3) {
      const { zeta, omega } = problem, k = omega * omega, c = 2 * zeta * omega;
      this.zeroGrad();
      let ld = 0, lp = 0, li = 0;
      if (w.data && data.length) {
        for (const p of data) { const [u] = this.forward(p.t, 0); const r = u - p.u; ld += r * r; this.backward((2 * w.data * r) / data.length, 0, 0); }
        ld /= data.length;
      }
      if (w.phys && colloc.length) {
        for (const t of colloc) {
          const [u, u1, u2] = this.forward(t, 2), r = (u2 + c * u1 + k * u) / k;
          lp += r * r;
          const g = (2 * w.phys * r) / colloc.length / k;
          this.backward(g * k, g * c, g);
        }
        lp /= colloc.length;
      }
      if (w.ic) {
        const [u, u1] = this.forward(0, 2), r0 = u - 1, r1 = u1 / omega;
        li = r0 * r0 + r1 * r1;
        this.backward(2 * w.ic * r0, (2 * w.ic * r1) / omega, 0);
      }
      this.adam(lr);
      return { data: ld, phys: lp, ic: li, total: w.data * ld + w.phys * lp + w.ic * li };
    }
    adam(lr) {
      const b1 = 0.9, b2 = 0.999, eps = 1e-8;
      if (!this.opt) this.opt = { mW: this.W.map((w) => new this.A(w.length)), vW: this.W.map((w) => new this.A(w.length)), mb: this.b.map((b) => new this.A(b.length)), vb: this.b.map((b) => new this.A(b.length)) };
      this.t++;
      // a short warm-up: a fresh Adam takes full-size steps at first, which would jolt pretrained weights
      lr *= Math.min(1, this.t / 200);
      const c1 = 1 - Math.pow(b1, this.t), c2 = 1 - Math.pow(b2, this.t), O = this.opt;
      const upd = (w, g, m, v) => {
        for (let i = 0; i < w.length; i++) {
          m[i] = b1 * m[i] + (1 - b1) * g[i]; v[i] = b2 * v[i] + (1 - b2) * g[i] * g[i];
          w[i] -= (lr * (m[i] / c1)) / (Math.sqrt(v[i] / c2) + eps);
        }
      };
      for (let l = 0; l < this.L; l++) { upd(this.W[l], this.gW[l], O.mW[l], O.vW[l]); upd(this.b[l], this.gb[l], O.mb[l], O.vb[l]); }
    }

    getParams() { const out = []; for (let l = 0; l < this.L; l++) out.push(...this.W[l], ...this.b[l]); return Float32Array.from(out); }
    setParams(flat) { let o = 0; for (let l = 0; l < this.L; l++) { this.W[l].set(flat.subarray(o, o + this.W[l].length)); o += this.W[l].length; this.b[l].set(flat.subarray(o, o + this.b[l].length)); o += this.b[l].length; } }

    // Relative L2 error against the exact solution on a fine grid.
    error(problem, n = 200) {
      let num = 0, den = 0;
      for (let i = 0; i <= n; i++) { const t = i / n, e = exact(t, problem.zeta, problem.omega), r = this.forward(t, 0)[0] - e; num += r * r; den += e * e; }
      return Math.sqrt(num / den);
    }
  }

  /* The training set: a few noisy observations early in time, and evenly spaced collocation points where
     only the physics is enforced. */
  function makeProblem(o = {}) {
    const p = Object.assign({ zeta: 0.1, omega: 18, nData: 10, tMax: 0.36, noise: 0.03, nColloc: 40, seed: 3 }, o);
    const rng = new NA.RNG(p.seed * 31 + 1), data = [];
    for (let i = 0; i < p.nData; i++) { const t = p.nData === 1 ? 0 : (p.tMax * i) / (p.nData - 1); data.push({ t, u: exact(t, p.zeta, p.omega) + rng.gauss(0, p.noise) }); }
    const colloc = [];
    for (let i = 0; i < p.nColloc; i++) colloc.push((i + 0.5) / p.nColloc);
    return { ...p, data, colloc };
  }

  NA.pinn = { PINN, exact, exactD, makeProblem };
})(typeof globalThis !== 'undefined' ? (globalThis.NA = globalThis.NA || {}) : (window.NA = window.NA || {}));
