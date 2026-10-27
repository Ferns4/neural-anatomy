/* transformer.js — a tiny GPT-style language model.
   tokens → token embedding + position embedding → L × [LayerNorm → causal multi-head self-attention → +residual
   → LayerNorm → GELU MLP → +residual] → final LayerNorm → unembedding → next-token logits.
   Forward and backward passes are written out by hand and checked against finite differences in
   tools/lm-gradcheck.js. Sequences are processed one at a time; gradients accumulate over a batch. */
(function (NA) {
  'use strict';

  const GELU_C = Math.sqrt(2 / Math.PI);
  function gelu(u) { return 0.5 * u * (1 + Math.tanh(GELU_C * (u + 0.044715 * u * u * u))); }
  function geluGrad(u) {
    const t = Math.tanh(GELU_C * (u + 0.044715 * u * u * u));
    return 0.5 * (1 + t) + 0.5 * u * (1 - t * t) * GELU_C * (1 + 3 * 0.044715 * u * u);
  }

  // y[n×m] = x[n×k] · W[k×m] + b   (row-major, W stored input-major)
  function matmul(y, x, W, b, n, k, m) {
    for (let i = 0; i < n; i++) {
      const yo = i * m, xo = i * k;
      if (b) for (let j = 0; j < m; j++) y[yo + j] = b[j]; else y.fill(0, yo, yo + m);
      for (let p = 0; p < k; p++) {
        const xv = x[xo + p];
        if (xv === 0) continue;
        const wo = p * m;
        for (let j = 0; j < m; j++) y[yo + j] += xv * W[wo + j];
      }
    }
  }
  // Backward of y = x·W + b: dx = dy·Wᵀ (written, or added when acc), dW += xᵀ·dy, db += Σ dy.
  function matmulBack(dx, dW, db, dy, x, W, n, k, m, acc) {
    for (let i = 0; i < n; i++) {
      const yo = i * m, xo = i * k;
      for (let p = 0; p < k; p++) {
        const wo = p * m, xv = x[xo + p];
        let s = 0;
        for (let j = 0; j < m; j++) { const g = dy[yo + j]; s += g * W[wo + j]; dW[wo + j] += xv * g; }
        if (dx) dx[xo + p] = acc ? dx[xo + p] + s : s;
      }
      if (db) for (let j = 0; j < m; j++) db[j] += dy[yo + j];
    }
  }

  function layerNorm(out, xhat, rstd, x, g, b, n, d) {
    for (let i = 0; i < n; i++) {
      const o = i * d;
      let mean = 0;
      for (let j = 0; j < d; j++) mean += x[o + j];
      mean /= d;
      let v = 0;
      for (let j = 0; j < d; j++) { const c = x[o + j] - mean; v += c * c; }
      const r = 1 / Math.sqrt(v / d + 1e-5);
      rstd[i] = r;
      for (let j = 0; j < d; j++) { const h = (x[o + j] - mean) * r; xhat[o + j] = h; out[o + j] = h * g[j] + b[j]; }
    }
  }
  // Adds the input gradient into dx.
  function layerNormBack(dx, dg, db, dout, xhat, rstd, g, n, d) {
    for (let i = 0; i < n; i++) {
      const o = i * d;
      let s1 = 0, s2 = 0;
      for (let j = 0; j < d; j++) {
        const dy = dout[o + j], dh = dy * g[j];
        s1 += dh; s2 += dh * xhat[o + j];
        dg[j] += dy * xhat[o + j]; db[j] += dy;
      }
      s1 /= d; s2 /= d;
      const r = rstd[i];
      for (let j = 0; j < d; j++) dx[o + j] += r * (dout[o + j] * g[j] - s1 - xhat[o + j] * s2);
    }
  }

  function softmaxRow(out, logits, o, V) {
    let max = -Infinity;
    for (let c = 0; c < V; c++) if (logits[o + c] > max) max = logits[o + c];
    let s = 0;
    for (let c = 0; c < V; c++) { const e = Math.exp(logits[o + c] - max); out[o + c] = e; s += e; }
    for (let c = 0; c < V; c++) out[o + c] /= s;
  }

  /* Causal attention for one layer. qkv rows are [q | k | v]; att[h] holds n×n probabilities (row i: query i);
     cat receives each head's output in its dh-wide slot. A head with mask 0 outputs zeros. */
  function attention(cat, att, qkv, n, d, H, mask) {
    const dh = d / H, scale = 1 / Math.sqrt(dh), d3 = 3 * d;
    cat.fill(0, 0, n * d);
    for (let h = 0; h < H; h++) {
      const A = att[h], on = !mask || mask[h];
      for (let i = 0; i < n; i++) {
        const qo = i * d3 + h * dh, ro = i * n;
        let max = -Infinity;
        for (let j = 0; j <= i; j++) {
          const ko = j * d3 + d + h * dh;
          let s = 0;
          for (let c = 0; c < dh; c++) s += qkv[qo + c] * qkv[ko + c];
          s *= scale; A[ro + j] = s;
          if (s > max) max = s;
        }
        let sum = 0;
        for (let j = 0; j <= i; j++) { const e = Math.exp(A[ro + j] - max); A[ro + j] = e; sum += e; }
        for (let j = 0; j <= i; j++) A[ro + j] /= sum;
        for (let j = i + 1; j < n; j++) A[ro + j] = 0;
        if (!on) continue;
        const oo = i * d + h * dh;
        for (let j = 0; j <= i; j++) {
          const a = A[ro + j], vo = j * d3 + 2 * d + h * dh;
          for (let c = 0; c < dh; c++) cat[oo + c] += a * qkv[vo + c];
        }
      }
    }
  }
  // Adds gradients for q, k and v into dqkv (zeroed by the caller).
  function attentionBack(dqkv, dcat, att, qkv, n, d, H, mask, dArow) {
    const dh = d / H, scale = 1 / Math.sqrt(dh), d3 = 3 * d;
    for (let h = 0; h < H; h++) {
      if (mask && !mask[h]) continue;
      const A = att[h];
      for (let i = 0; i < n; i++) {
        const oo = i * d + h * dh, ro = i * n;
        let dot = 0;
        for (let j = 0; j <= i; j++) {
          const vo = j * d3 + 2 * d + h * dh, a = A[ro + j];
          let s = 0;
          for (let c = 0; c < dh; c++) { const g = dcat[oo + c]; s += g * qkv[vo + c]; dqkv[vo + c] += a * g; }
          dArow[j] = s; dot += a * s;
        }
        const qo = i * d3 + h * dh;
        for (let j = 0; j <= i; j++) {
          const ds = A[ro + j] * (dArow[j] - dot) * scale;
          if (ds === 0) continue;
          const ko = j * d3 + d + h * dh;
          for (let c = 0; c < dh; c++) { dqkv[qo + c] += ds * qkv[ko + c]; dqkv[ko + c] += ds * qkv[qo + c]; }
        }
      }
    }
  }

  class TinyGPT {
    /* cfg: { V vocabulary, T context length, d width, H heads, L layers, F MLP width, seed, f64 } */
    constructor(cfg) {
      this.cfg = Object.assign({ V: 36, T: 20, d: 32, H: 4, L: 2, F: 128, seed: 1, init: 0.08 }, cfg);
      const { V, T, d, H, L, F } = this.cfg;
      if (d % H) throw new Error('width must be divisible by the number of heads');
      const Arr = this.cfg.f64 ? Float64Array : Float32Array;
      this.Arr = Arr;
      this.params = [];
      const P = (name, size, shape, matrix) => { const p = { name, w: new Arr(size), g: new Arr(size), shape, matrix: !!matrix }; this.params.push(p); return p; };
      this.E = P('token embedding', V * d, [V, d], true);
      this.Pos = P('position embedding', T * d, [T, d], true);
      this.layers = [];
      for (let l = 0; l < L; l++) {
        const n = `block ${l + 1} `;
        this.layers.push({
          ln1g: P(n + 'LN1 gain', d, [d]), ln1b: P(n + 'LN1 bias', d, [d]),
          Wqkv: P(n + 'attention Wq Wk Wv', d * 3 * d, [d, 3 * d], true), bqkv: P(n + 'attention qkv bias', 3 * d, [3 * d]),
          Wo: P(n + 'attention Wo', d * d, [d, d], true), bo: P(n + 'attention output bias', d, [d]),
          ln2g: P(n + 'LN2 gain', d, [d]), ln2b: P(n + 'LN2 bias', d, [d]),
          W1: P(n + 'MLP W1', d * F, [d, F], true), b1: P(n + 'MLP b1', F, [F]),
          W2: P(n + 'MLP W2', F * d, [F, d], true), b2: P(n + 'MLP b2', d, [d]),
        });
      }
      this.lnfg = P('final LN gain', d, [d]); this.lnfb = P('final LN bias', d, [d]);
      this.U = P('unembedding', d * V, [d, V], true); this.bu = P('unembedding bias', V, [V]);
      this.headMask = this.layers.map(() => new Uint8Array(H).fill(1));
      this.mlpMask = new Uint8Array(L).fill(1);
      this.paramCount = this.params.reduce((s, p) => s + p.w.length, 0);
      this.t = 0; this.opt = null;
      this.init(this.cfg.seed);
      this.alloc();
    }

    init(seed) {
      const rng = new NA.RNG(seed * 7919 + 13), s = this.cfg.init, L = this.cfg.L;
      for (const p of this.params) {
        if (p.name.endsWith('gain')) p.w.fill(1);
        else if (!p.matrix) p.w.fill(0);
        else {
          // residual-stream writers start smaller so deep stacks begin close to the identity
          const sd = /Wo|W2/.test(p.name) ? s / Math.sqrt(2 * L) : s;
          for (let i = 0; i < p.w.length; i++) p.w[i] = rng.gauss(0, sd);
        }
      }
      this.t = 0; this.opt = null;
    }

    alloc() {
      const { V, T, d, H, L, F } = this.cfg, A = this.Arr, td = () => new A(T * d);
      const per = (fn) => this.layers.map(fn);
      this.ws = {
        x: [td(), ...per(() => td())], mid: per(td), ao: per(td), mo: per(td),
        ln1: per(td), ln1h: per(td), ln1r: per(() => new A(T)), ln2: per(td), ln2h: per(td), ln2r: per(() => new A(T)),
        qkv: per(() => new A(T * 3 * d)), att: per(() => Array.from({ length: H }, () => new A(T * T))), cat: per(td),
        u: per(() => new A(T * F)), g: per(() => new A(T * F)),
        lnf: td(), lnfh: td(), lnfr: new A(T), logits: new A(T * V), probs: new A(T * V),
        dlogits: new A(T * V), dlnf: td(), dx: td(), dmid: td(), dg: new A(T * F), du: new A(T * F), dln: td(),
        dcat: td(), dqkv: new A(T * 3 * d), dArow: new A(T),
      };
    }

    // ------------------------------------------------------------ forward
    forward(ids) {
      const n = Math.min(ids.length, this.cfg.T), { d, H, L, F, V } = this.cfg, ws = this.ws, E = this.E.w, Pw = this.Pos.w;
      const x0 = ws.x[0];
      for (let i = 0; i < n; i++) { const eo = ids[i] * d, po = i * d; for (let j = 0; j < d; j++) x0[po + j] = E[eo + j] + Pw[po + j]; }
      for (let l = 0; l < L; l++) {
        const P = this.layers[l], x = ws.x[l], mid = ws.mid[l], ao = ws.ao[l], mo = ws.mo[l], xn = ws.x[l + 1];
        layerNorm(ws.ln1[l], ws.ln1h[l], ws.ln1r[l], x, P.ln1g.w, P.ln1b.w, n, d);
        matmul(ws.qkv[l], ws.ln1[l], P.Wqkv.w, P.bqkv.w, n, d, 3 * d);
        attention(ws.cat[l], ws.att[l], ws.qkv[l], n, d, H, this.headMask[l]);
        matmul(ao, ws.cat[l], P.Wo.w, P.bo.w, n, d, d);
        for (let k = 0; k < n * d; k++) mid[k] = x[k] + ao[k];
        layerNorm(ws.ln2[l], ws.ln2h[l], ws.ln2r[l], mid, P.ln2g.w, P.ln2b.w, n, d);
        matmul(ws.u[l], ws.ln2[l], P.W1.w, P.b1.w, n, d, F);
        const u = ws.u[l], g = ws.g[l];
        for (let k = 0; k < n * F; k++) g[k] = gelu(u[k]);
        matmul(mo, g, P.W2.w, P.b2.w, n, F, d);
        if (this.mlpMask[l]) for (let k = 0; k < n * d; k++) xn[k] = mid[k] + mo[k];
        else for (let k = 0; k < n * d; k++) xn[k] = mid[k];
      }
      layerNorm(ws.lnf, ws.lnfh, ws.lnfr, ws.x[L], this.lnfg.w, this.lnfb.w, n, d);
      matmul(ws.logits, ws.lnf, this.U.w, this.bu.w, n, d, V);
      for (let i = 0; i < n; i++) softmaxRow(ws.probs, ws.logits, i * V, V);
      return n;
    }

    // ------------------------------------------------------------ backward
    /* After forward(ids): adds wt × the gradient of the summed next-token loss to every parameter's .g and
       returns the summed loss. Position i predicts ids[i + 1]. */
    backward(ids, wt) {
      const n = Math.min(ids.length, this.cfg.T), { d, H, L, F, V } = this.cfg, ws = this.ws;
      const dl = ws.dlogits, probs = ws.probs;
      let loss = 0;
      for (let i = 0; i < n; i++) {
        const o = i * V;
        if (i + 1 >= ids.length) { dl.fill(0, o, o + V); continue; }
        const t = ids[i + 1];
        loss -= Math.log(Math.max(probs[o + t], 1e-12));
        for (let c = 0; c < V; c++) dl[o + c] = probs[o + c] * wt;
        dl[o + t] -= wt;
      }
      matmulBack(ws.dlnf, this.U.g, this.bu.g, dl, ws.lnf, this.U.w, n, d, V, false);
      const dx = ws.dx, dmid = ws.dmid;
      dx.fill(0, 0, n * d);
      layerNormBack(dx, this.lnfg.g, this.lnfb.g, ws.dlnf, ws.lnfh, ws.lnfr, this.lnfg.w, n, d);
      for (let l = L - 1; l >= 0; l--) {
        const P = this.layers[l];
        for (let k = 0; k < n * d; k++) dmid[k] = dx[k];
        if (this.mlpMask[l]) {
          matmulBack(ws.dg, P.W2.g, P.b2.g, dx, ws.g[l], P.W2.w, n, F, d, false);
          const u = ws.u[l];
          for (let k = 0; k < n * F; k++) ws.du[k] = ws.dg[k] * geluGrad(u[k]);
          matmulBack(ws.dln, P.W1.g, P.b1.g, ws.du, ws.ln2[l], P.W1.w, n, d, F, false);
          layerNormBack(dmid, P.ln2g.g, P.ln2b.g, ws.dln, ws.ln2h[l], ws.ln2r[l], P.ln2g.w, n, d);
        }
        matmulBack(ws.dcat, P.Wo.g, P.bo.g, dmid, ws.cat[l], P.Wo.w, n, d, d, false);
        ws.dqkv.fill(0, 0, n * 3 * d);
        attentionBack(ws.dqkv, ws.dcat, ws.att[l], ws.qkv[l], n, d, H, this.headMask[l], ws.dArow);
        matmulBack(ws.dln, P.Wqkv.g, P.bqkv.g, ws.dqkv, ws.ln1[l], P.Wqkv.w, n, d, 3 * d, false);
        for (let k = 0; k < n * d; k++) dx[k] = dmid[k];
        layerNormBack(dx, P.ln1g.g, P.ln1b.g, ws.dln, ws.ln1h[l], ws.ln1r[l], P.ln1g.w, n, d);
      }
      const Eg = this.E.g, Pg = this.Pos.g;
      for (let i = 0; i < n; i++) { const eo = ids[i] * d, po = i * d; for (let j = 0; j < d; j++) { Eg[eo + j] += dx[po + j]; Pg[po + j] += dx[po + j]; } }
      return loss;
    }

    zeroGrad() { for (const p of this.params) p.g.fill(0); }

    /* One optimisation step on a batch of id sequences. Returns the mean next-token loss and the gradient norm. */
    trainStep(batch, opt = {}) {
      this.zeroGrad();
      let count = 0;
      for (const s of batch) count += Math.min(s.length, this.cfg.T) - 1;
      const wt = 1 / Math.max(1, count);
      let loss = 0;
      for (const s of batch) { this.forward(s); loss += this.backward(s, wt); }
      const gnorm = this.gradNorm();
      this.adam(opt, gnorm);
      return { loss: loss / Math.max(1, count), gnorm };
    }
    gradNorm() { let s = 0; for (const p of this.params) for (let i = 0; i < p.g.length; i++) s += p.g[i] * p.g[i]; return Math.sqrt(s); }

    // Adam with linear warm-up, gradient clipping and decoupled weight decay on matrices.
    adam(o, gnorm) {
      const lr0 = o.lr || 3e-3, b1 = 0.9, b2 = o.beta2 || 0.99, eps = 1e-8, wd = o.wd ?? 0.01, clip = o.clip || 1;
      if (!this.opt) this.opt = this.params.map((p) => ({ m: new this.Arr(p.w.length), v: new this.Arr(p.w.length) }));
      this.t++;
      const lr = lr0 * Math.min(1, this.t / (o.warmup || 100));
      const c1 = 1 - Math.pow(b1, this.t), c2 = 1 - Math.pow(b2, this.t), cs = gnorm > clip ? clip / gnorm : 1;
      this.params.forEach((p, k) => {
        const { m, v } = this.opt[k], w = p.w, g = p.g, decay = p.matrix ? lr * wd : 0;
        for (let i = 0; i < w.length; i++) {
          const gi = g[i] * cs;
          m[i] = b1 * m[i] + (1 - b1) * gi;
          v[i] = b2 * v[i] + (1 - b2) * gi * gi;
          w[i] -= lr * (m[i] / c1) / (Math.sqrt(v[i] / c2) + eps) + decay * w[i];
        }
      });
    }

    // ------------------------------------------------------------ parameters
    getParams() {
      const out = new Float32Array(this.paramCount);
      let o = 0;
      for (const p of this.params) { out.set(p.w, o); o += p.w.length; }
      return out;
    }
    setParams(flat) {
      let o = 0;
      for (const p of this.params) { p.w.set(flat.subarray(o, o + p.w.length)); o += p.w.length; }
    }
    resetOptimizer() { this.opt = null; this.t = 0; }

    // ------------------------------------------------------------ analysis
    // Final LayerNorm and unembedding applied to any residual-stream state: the "logit lens".
    decode(x, n) {
      const { d, V } = this.cfg, A = this.Arr, h = new A(n * d), xh = new A(n * d), r = new A(n), out = new Float32Array(n * V);
      layerNorm(h, xh, r, x, this.lnfg.w, this.lnfb.w, n, d);
      matmul(out, h, this.U.w, this.bu.w, n, d, V);
      for (let i = 0; i < n; i++) softmaxRow(out, out, i * V, V);
      return out;
    }

    /* Everything the views need for one prompt: residual stream after each stage, attention patterns, what
       each head and MLP writes, MLP activations, predictions and the logit lens at every stage. */
    trace(ids) {
      const n = this.forward(ids), { d, H, L, F, V } = this.cfg, ws = this.ws, dh = d / H;
      const cp = (a, len) => Float32Array.from(a.subarray(0, len));
      const tr = { n, ids: ids.slice(0, n), resid: [cp(ws.x[0], n * d)], stages: ['embed'], attn: [], heads: [], mlp: [], mlpAct: [], attnBias: [],
        qkv: [], ln1: [], ln2: [], pre: [],
        probs: cp(ws.probs, n * V), logits: cp(ws.logits, n * V), lnfScale: cp(ws.lnfr, n), lnf: cp(ws.lnf, n * d) };
      for (let l = 0; l < L; l++) {
        const P = this.layers[l], cat = ws.cat[l], Wo = P.Wo.w;
        tr.attn.push(ws.att[l].map((A) => cp(A, n * n)));
        // internals for the architecture view: q, k, v, both LayerNorm outputs and the MLP before GELU
        tr.qkv.push(cp(ws.qkv[l], n * 3 * d)); tr.ln1.push(cp(ws.ln1[l], n * d)); tr.ln2.push(cp(ws.ln2[l], n * d)); tr.pre.push(cp(ws.u[l], n * F));
        // each head's write into the residual stream: its slice of cat times its rows of Wo
        const writes = [];
        for (let h = 0; h < H; h++) {
          const out = new Float32Array(n * d);
          for (let i = 0; i < n; i++) for (let c = 0; c < dh; c++) {
            const a = cat[i * d + h * dh + c];
            if (a === 0) continue;
            const wo = (h * dh + c) * d;
            for (let j = 0; j < d; j++) out[i * d + j] += a * Wo[wo + j];
          }
          writes.push(out);
        }
        tr.heads.push(writes);
        tr.attnBias.push(Float32Array.from(P.bo.w));
        tr.resid.push(cp(ws.mid[l], n * d)); tr.stages.push('attn' + l);
        tr.mlp.push(this.mlpMask[l] ? cp(ws.mo[l], n * d) : new Float32Array(n * d));
        tr.mlpAct.push(cp(ws.g[l], n * F));
        tr.resid.push(cp(ws.x[l + 1], n * d)); tr.stages.push('mlp' + l);
      }
      tr.lens = tr.resid.map((x) => this.decode(x, n));
      return tr;
    }

    /* Direct logit attribution at position i: how much the embedding, each head, each attention bias and
       each MLP add to logit(a) − logit(b), with the final LayerNorm's scale frozen. The parts sum exactly to
       the logit difference. */
    attribution(tr, i, a, b) {
      const { d, H, L, V } = this.cfg, g = this.lnfg.w, U = this.U.w, r = tr.lnfScale[i];
      const dir = new Float32Array(d);
      for (let j = 0; j < d; j++) dir[j] = g[j] * (U[j * V + a] - (b === undefined || b === null ? 0 : U[j * V + b])) * r;
      const score = (vec, o) => {
        let mean = 0; for (let j = 0; j < d; j++) mean += vec[o + j]; mean /= d;
        let s = 0; for (let j = 0; j < d; j++) s += (vec[o + j] - mean) * dir[j];
        return s;
      };
      const parts = [{ id: 'embed', label: 'Embedding', value: score(tr.resid[0], i * d) }];
      for (let l = 0; l < L; l++) {
        for (let h = 0; h < H; h++) parts.push({ id: `h${l}.${h}`, label: `L${l + 1} H${h + 1}`, layer: l, head: h, value: score(tr.heads[l][h], i * d) });
        parts.push({ id: `bo${l}`, label: `L${l + 1} attention bias`, value: score(tr.attnBias[l], 0) });
        parts.push({ id: `mlp${l}`, label: `MLP ${l + 1}`, layer: l, value: score(tr.mlp[l], i * d) });
      }
      let constant = 0;
      for (let j = 0; j < d; j++) constant += this.lnfb.w[j] * (U[j * V + a] - (b === undefined || b === null ? 0 : U[j * V + b]));
      constant += this.bu.w[a] - (b === undefined || b === null ? 0 : this.bu.w[b]);
      parts.push({ id: 'const', label: 'Biases after the stream', value: constant });
      return parts;
    }
  }

  // ------------------------------------------------------------------ float16 weights in base64
  const f32 = new Float32Array(1), u32 = new Uint32Array(f32.buffer);
  function toHalf(val) {
    f32[0] = val;
    const x = u32[0], sign = (x >>> 16) & 0x8000, exp = ((x >>> 23) & 0xff) - 112, mant = x & 0x7fffff;
    if (exp <= 0) {
      if (exp < -10) return sign;
      const m = (mant | 0x800000) >> (1 - exp);
      return sign | ((m + 0x1000) >> 13);
    }
    if (exp >= 31) return sign | 0x7c00;
    return (sign | (exp << 10) | (mant >> 13)) + ((mant >> 12) & 1);
  }
  function fromHalf(h) {
    const s = h & 0x8000 ? -1 : 1, e = (h >> 10) & 0x1f, m = h & 0x3ff;
    if (e === 0) return s * m * 5.960464477539063e-8;
    if (e === 31) return m ? NaN : s * Infinity;
    return s * (1 + m / 1024) * Math.pow(2, e - 15);
  }
  function encodeWeights(flat) {
    const h = new Uint16Array(flat.length);
    for (let i = 0; i < flat.length; i++) h[i] = toHalf(flat[i]);
    const bytes = new Uint8Array(h.buffer);
    if (typeof Buffer !== 'undefined') return Buffer.from(bytes).toString('base64');
    let s = '';
    for (let i = 0; i < bytes.length; i += 0x8000) s += String.fromCharCode.apply(null, bytes.subarray(i, i + 0x8000));
    return btoa(s);
  }
  function decodeWeights(b64) {
    let bytes;
    if (typeof Buffer !== 'undefined' && typeof window === 'undefined') bytes = new Uint8Array(Buffer.from(b64, 'base64'));
    else { const s = atob(b64); bytes = new Uint8Array(s.length); for (let i = 0; i < s.length; i++) bytes[i] = s.charCodeAt(i); }
    const h = new Uint16Array(bytes.buffer, bytes.byteOffset, bytes.byteLength >> 1), out = new Float32Array(h.length);
    for (let i = 0; i < h.length; i++) out[i] = fromHalf(h[i]);
    return out;
  }

  // ------------------------------------------------------------------ evaluation and head roles
  /* Loss, perplexity and skill accuracies over a list of stories:
     lookup = the right drink after "N drinks"; agree = verb of the right number (probability mass);
     conflict = agreement when the nearest noun has the other number. */
  function evaluate(model, stories, W) {
    const V = model.cfg.V;
    let loss = 0, count = 0, correct = 0, lk = 0, lkOk = 0, ag = 0, agOk = 0, cf = 0, cfOk = 0;
    for (const s of stories) {
      const ids = s.ids, n = model.forward(ids), P = model.ws.probs;
      for (let i = 0; i + 1 < n; i++) {
        const o = i * V, t = ids[i + 1];
        loss -= Math.log(Math.max(P[o + t], 1e-12)); count++;
        let best = 0; for (let c = 1; c < V; c++) if (P[o + c] > P[o + best]) best = c;
        if (best === t) correct++;
      }
      for (const r of s.roles) {
        if (r.q + 1 >= n) continue;
        const o = r.q * V;
        if (r.type === 'lookup') {
          lk++;
          let best = 0; for (let c = 1; c < V; c++) if (P[o + c] > P[o + best]) best = c;
          if (best === ids[r.q + 1]) lkOk++;
        } else {
          let right = 0, wrong = 0;
          for (const c of r.plural ? W.VPL : W.VSG) right += P[o + c];
          for (const c of r.plural ? W.VSG : W.VPL) wrong += P[o + c];
          const ok = right > wrong;
          ag++; if (ok) agOk++;
          if (r.conflict) { cf++; if (ok) cfOk++; }
        }
      }
    }
    const l = loss / Math.max(1, count);
    return { loss: l, ppl: Math.exp(l), acc: correct / Math.max(1, count), lookup: lk ? lkOk / lk : null, agree: ag ? agOk / ag : null, conflict: cf ? cfOk / cf : null, n: stories.length };
  }

  /* Average attention patterns per head over stories, scored against behaviours we can name:
     prev (to the token just before), bos (to the first token), self, lookup (from "drinks" to the right drink),
     subject (from the noun before the verb back to the sentence's subject, when they differ). */
  function headStats(model, stories) {
    const { L, H } = model.cfg;
    const z = () => ({ prev: 0, prevN: 0, bos: 0, self: 0, allN: 0, lookup: 0, lookupWrong: 0, lookupN: 0, subj: 0, subjN: 0, name: 0, ent: 0 });
    const st = Array.from({ length: L }, () => Array.from({ length: H }, z));
    for (const s of stories) {
      const n = model.forward(s.ids);
      for (let l = 0; l < L; l++) for (let h = 0; h < H; h++) {
        const A = model.ws.att[l][h], S = st[l][h];
        for (let i = 1; i < n; i++) {
          const ro = i * n;
          S.prev += A[ro + i - 1]; S.prevN++;
          S.bos += A[ro]; S.self += A[ro + i]; S.allN++;
          let e = 0; for (let j = 0; j <= i; j++) { const a = A[ro + j]; if (a > 1e-9) e -= a * Math.log(a); }
          S.ent += e / Math.log(i + 1);
        }
        for (const r of s.roles) {
          if (r.type === 'lookup') {
            const ro = r.q * n;
            S.lookup += A[ro + r.src]; S.name += A[ro + r.name];
            for (const o of r.others) S.lookupWrong += A[ro + o];
            S.lookupN++;
          } else if (r.distractor >= 0) {
            S.subj += A[r.q * n + r.subj]; S.subjN++;
          }
        }
      }
    }
    return st.map((row) => row.map((S) => ({
      prev: S.prev / Math.max(1, S.prevN), bos: S.bos / Math.max(1, S.allN), self: S.self / Math.max(1, S.allN), spread: S.ent / Math.max(1, S.allN),
      lookup: S.lookupN ? S.lookup / S.lookupN : 0, lookupWrong: S.lookupN ? S.lookupWrong / S.lookupN : 0, name: S.lookupN ? S.name / S.lookupN : 0,
      subject: S.subjN ? S.subj / S.subjN : 0,
    })));
  }
  // A short role for a head from its statistics.
  function headRole(s) {
    if (s.lookup > 0.35 && s.lookup > 2 * s.lookupWrong) return { id: 'lookup', label: 'Lookup head', text: `From "drinks" it attends to the drink that person likes (${Math.round(s.lookup * 100)}% of its attention there).` };
    if (s.subject > 0.35) return { id: 'subject', label: 'Subject head', text: `Before a verb it looks back past the nearer noun to the subject (${Math.round(s.subject * 100)}%).` };
    if (s.prev > 0.45) return { id: 'prev', label: 'Previous-token head', text: `Mostly attends to the token just before (${Math.round(s.prev * 100)}%), copying it forward one place.` };
    if (s.bos > 0.5) return { id: 'bos', label: 'Resting head', text: `Parks ${Math.round(s.bos * 100)}% of its attention on ⟨bos⟩, which acts as "nothing to do here".` };
    if (s.self > 0.45) return { id: 'self', label: 'Self head', text: `Mostly attends to its own position (${Math.round(s.self * 100)}%).` };
    if (s.lookup + s.lookupWrong > 0.4) return { id: 'drinks', label: 'Drink finder', text: `From "drinks" it attends to drinks in general (${Math.round((s.lookup + s.lookupWrong) * 100)}%), not yet the right one.` };
    return { id: 'mixed', label: 'Mixed head', text: `Spreads its attention (${Math.round(s.spread * 100)}% of the maximum spread); no single job stands out.` };
  }

  NA.tf = { TinyGPT, evaluate, headStats, headRole, encodeWeights, decodeWeights, gelu, geluGrad };
})(typeof globalThis !== 'undefined' ? (globalThis.NA = globalThis.NA || {}) : (window.NA = window.NA || {}));
