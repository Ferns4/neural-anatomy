/* builder-engine.js — the engine behind "build a model from scratch". A model is a list of blocks (dense, gated,
   activation, dropout, noise, LayerNorm, batch norm, Fourier features, self-attention, transformer, recurrent
   (RNN, LSTM, GRU), 1-D convolution, positions, sequence pooling, image patches, convolution, max and average
   pool, flatten, global average pool, parallel branches; and separable convolution, squeeze-and-excitation,
   coordinate channels, random shifts, global max pool, attention pooling, a state-space layer, radial basis units,
   maxout and a mixture of experts) between an input (a vector, an image, a sequence of token ids or of numbers)
   and an output head (classes, a value, or a token at every position). Shapes are plain arrays per example: [F] a vector, [C, H, W] an image, [T, D] a
   sequence. Every buffer holds the whole batch, one example after another, so a layer is a few large matrix
   products rather than many small ones. Forward and backward passes are written out by hand and checked against
   finite differences in tools/builder-gradcheck.js. Needs core.js (NA.RNG) and engine.js (the activations). */
(function (NA) {
  'use strict';

  // ------------------------------------------------------------------ kernels
  /* Y[n×m] += A[n×k] · W[m×k]ᵀ, each matrix starting at its own offset. Rows of A and W are both contiguous, so
     every output is a dot product of two rows; two rows of A against four rows of W keep eight sums in
     registers and load six numbers for every eight multiply-adds. Every product in the engine comes through
     here, with a transposed copy wherever a layout does not fit. */
  function gemmNT(Y, oy, A, oa, W, ow, n, m, k) {
    let i = 0;
    for (; i + 1 < n; i += 2) {
      const a0 = oa + i * k, a1 = a0 + k, y0 = oy + i * m, y1 = y0 + m;
      let j = 0;
      for (; j + 3 < m; j += 4) {
        const w0 = ow + j * k, w1 = w0 + k, w2 = w1 + k, w3 = w2 + k;
        let s00 = 0, s01 = 0, s02 = 0, s03 = 0, s10 = 0, s11 = 0, s12 = 0, s13 = 0;
        for (let p = 0; p < k; p++) {
          const x0 = A[a0 + p], x1 = A[a1 + p], v0 = W[w0 + p], v1 = W[w1 + p], v2 = W[w2 + p], v3 = W[w3 + p];
          s00 += x0 * v0; s01 += x0 * v1; s02 += x0 * v2; s03 += x0 * v3;
          s10 += x1 * v0; s11 += x1 * v1; s12 += x1 * v2; s13 += x1 * v3;
        }
        Y[y0 + j] += s00; Y[y0 + j + 1] += s01; Y[y0 + j + 2] += s02; Y[y0 + j + 3] += s03;
        Y[y1 + j] += s10; Y[y1 + j + 1] += s11; Y[y1 + j + 2] += s12; Y[y1 + j + 3] += s13;
      }
      for (; j < m; j++) {
        const w0 = ow + j * k;
        let s0 = 0, s1 = 0;
        for (let p = 0; p < k; p++) { const v = W[w0 + p]; s0 += A[a0 + p] * v; s1 += A[a1 + p] * v; }
        Y[y0 + j] += s0; Y[y1 + j] += s1;
      }
    }
    if (i < n) {
      const a0 = oa + i * k, y0 = oy + i * m;
      for (let j = 0; j < m; j++) {
        const w0 = ow + j * k;
        let s = 0;
        for (let p = 0; p < k; p++) s += A[a0 + p] * W[w0 + p];
        Y[y0 + j] += s;
      }
    }
  }

  // out[m×n] = x[n×m]ᵀ, x starting at offset xo
  function transpose(out, x, xo, n, m) {
    for (let i = 0; i < n; i++) {
      const r = xo + i * m;
      for (let j = 0, o = i; j < m; j++, o += n) out[o] = x[r + j];
    }
  }

  /* Named buffers that grow to the largest size asked for. view() hands out exactly n numbers (a view when the
     buffer is larger), so switching between a training batch and a bigger evaluation batch reuses memory;
     get() is for scratch space, where only the first n numbers matter. */
  class Mem {
    constructor(Arr) { this.Arr = Arr; this.bufs = Object.create(null); }
    get(name, n, Arr) {
      let a = this.bufs[name];
      if (!a || a.length < n) a = this.bufs[name] = new (Arr || this.Arr)(Math.max(1, n));
      return a;
    }
    view(name, n, Arr) { const a = this.get(name, n, Arr); return a.length === n ? a : a.subarray(0, n); }
  }

  // y[N×M] = x[N×K] · Wᵀ + b, with W stored one row per output unit
  function linear(y, x, W, b, N, K, M) {
    for (let i = 0, o = 0; i < N; i++) for (let j = 0; j < M; j++, o++) y[o] = b[j];
    gemmNT(y, 0, x, 0, W, 0, N, M, K);
  }
  /* Backward of linear(): dW += dyᵀ·x and db += Σ dy over the rows, then dx = dy·W (added into dx when acc,
     skipped when dx is null). Transposed copies in the shared scratch space put both products in gemmNT's form;
     xT, when given, is x already transposed (attention's three projections share one input). */
  function linearBack(scr, dx, acc, dW, db, dy, x, W, N, K, M, xT) {
    const dyT = scr.get('dyT', M * N);
    transpose(dyT, dy, 0, N, M);
    if (!xT) { xT = scr.get('xT', K * N); transpose(xT, x, 0, N, K); }
    gemmNT(dW, 0, dyT, 0, xT, 0, M, K, N);
    for (let j = 0; j < M; j++) { let s = 0; for (let i = j * N, e = i + N; i < e; i++) s += dyT[i]; db[j] += s; }
    if (!dx) return;
    const WT = scr.get('WT', K * M);
    transpose(WT, W, 0, M, K);
    if (!acc) dx.fill(0, 0, N * K);
    gemmNT(dx, 0, dy, 0, WT, 0, N, K, M);
  }

  // Each of n rows of width d → zero mean and unit variance (eps 1e-5), then gain and bias.
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
  // Adds the input gradient into dx (unless dx is null) and accumulates the gain and bias gradients.
  function layerNormBack(dx, dg, db, dout, xhat, rstd, g, n, d) {
    for (let i = 0; i < n; i++) {
      const o = i * d;
      let s1 = 0, s2 = 0;
      for (let j = 0; j < d; j++) {
        const dy = dout[o + j], dh = dy * g[j];
        s1 += dh; s2 += dh * xhat[o + j];
        dg[j] += dy * xhat[o + j]; db[j] += dy;
      }
      if (!dx) continue;
      s1 /= d; s2 /= d;
      const r = rstd[i];
      for (let j = 0; j < d; j++) dx[o + j] += r * (dout[o + j] * g[j] - s1 - xhat[o + j] * s2);
    }
  }

  /* Multi-head attention over B sequences of T positions, D features split into H heads of D/H: per head, the
     scores q·kᵀ/√(D/H) (causal: only positions up to the query's own), a softmax, and the weighted sum of v
     written into that head's slot of o. att keeps every head's probabilities, [B, H, T, T], masked entries 0;
     row is float64 scratch of length T. Keys are taken four at a time so each number of q is loaded once. */
  function attend(o, att, q, k, v, B, T, D, H, causal, row) {
    const dh = D / H, scale = 1 / Math.sqrt(dh);
    for (let b = 0; b < B; b++) {
      const base = b * T * D;
      for (let h = 0; h < H; h++) {
        const a0 = (b * H + h) * T * T, ho = base + h * dh;
        for (let i = 0; i < T; i++) {
          const qo = ho + i * D, ro = a0 + i * T, jn = causal ? i + 1 : T;
          let j = 0;
          for (; j + 3 < jn; j += 4) {
            const k0 = ho + j * D, k1 = k0 + D, k2 = k1 + D, k3 = k2 + D;
            let s0 = 0, s1 = 0, s2 = 0, s3 = 0;
            for (let c = 0; c < dh; c++) { const x = q[qo + c]; s0 += x * k[k0 + c]; s1 += x * k[k1 + c]; s2 += x * k[k2 + c]; s3 += x * k[k3 + c]; }
            row[j] = s0 * scale; row[j + 1] = s1 * scale; row[j + 2] = s2 * scale; row[j + 3] = s3 * scale;
          }
          for (; j < jn; j++) {
            const ko = ho + j * D;
            let s = 0;
            for (let c = 0; c < dh; c++) s += q[qo + c] * k[ko + c];
            row[j] = s * scale;
          }
          let max = -Infinity, sum = 0;
          for (j = 0; j < jn; j++) if (row[j] > max) max = row[j];
          for (j = 0; j < jn; j++) { const e = Math.exp(row[j] - max); row[j] = e; sum += e; }
          for (j = 0; j < jn; j++) att[ro + j] = row[j] /= sum;
          for (j = jn; j < T; j++) att[ro + j] = 0;
          for (let c = 0; c < dh; c++) o[qo + c] = 0;
          for (j = 0; j + 3 < jn; j += 4) {
            const p0 = row[j], p1 = row[j + 1], p2 = row[j + 2], p3 = row[j + 3], v0 = ho + j * D, v1 = v0 + D, v2 = v1 + D, v3 = v2 + D;
            for (let c = 0; c < dh; c++) o[qo + c] += p0 * v[v0 + c] + p1 * v[v1 + c] + p2 * v[v2 + c] + p3 * v[v3 + c];
          }
          for (; j < jn; j++) {
            const p = row[j], vo = ho + j * D;
            for (let c = 0; c < dh; c++) o[qo + c] += p * v[vo + c];
          }
        }
      }
    }
  }
  // Adds the gradients for q, k and v (zeroed by the caller) given the gradient for o; row is float64 scratch.
  function attendBack(dq, dk, dv, dO, att, q, k, v, B, T, D, H, causal, row) {
    const dh = D / H, scale = 1 / Math.sqrt(dh);
    for (let b = 0; b < B; b++) {
      const base = b * T * D;
      for (let h = 0; h < H; h++) {
        const a0 = (b * H + h) * T * T, ho = base + h * dh;
        for (let i = 0; i < T; i++) {
          const qo = ho + i * D, ro = a0 + i * T, jn = causal ? i + 1 : T;
          // da_j = dO_i · v_j, and v_j collects a_j · dO_i
          let dot = 0, j = 0;
          for (; j + 3 < jn; j += 4) {
            const v0 = ho + j * D, v1 = v0 + D, v2 = v1 + D, v3 = v2 + D, p0 = att[ro + j], p1 = att[ro + j + 1], p2 = att[ro + j + 2], p3 = att[ro + j + 3];
            let s0 = 0, s1 = 0, s2 = 0, s3 = 0;
            for (let c = 0; c < dh; c++) {
              const g = dO[qo + c];
              s0 += g * v[v0 + c]; s1 += g * v[v1 + c]; s2 += g * v[v2 + c]; s3 += g * v[v3 + c];
              dv[v0 + c] += p0 * g; dv[v1 + c] += p1 * g; dv[v2 + c] += p2 * g; dv[v3 + c] += p3 * g;
            }
            row[j] = s0; row[j + 1] = s1; row[j + 2] = s2; row[j + 3] = s3;
            dot += p0 * s0 + p1 * s1 + p2 * s2 + p3 * s3;
          }
          for (; j < jn; j++) {
            const vo = ho + j * D, p = att[ro + j];
            let s = 0;
            for (let c = 0; c < dh; c++) { const g = dO[qo + c]; s += g * v[vo + c]; dv[vo + c] += p * g; }
            row[j] = s; dot += p * s;
          }
          // through the softmax, ds_j = a_j (da_j − Σ a·da), then through the scaled dot product
          for (j = 0; j < jn; j++) row[j] = att[ro + j] * (row[j] - dot) * scale;
          for (j = 0; j + 3 < jn; j += 4) {
            const d0 = row[j], d1 = row[j + 1], d2 = row[j + 2], d3 = row[j + 3], k0 = ho + j * D, k1 = k0 + D, k2 = k1 + D, k3 = k2 + D;
            for (let c = 0; c < dh; c++) {
              const x = q[qo + c];
              dq[qo + c] += d0 * k[k0 + c] + d1 * k[k1 + c] + d2 * k[k2 + c] + d3 * k[k3 + c];
              dk[k0 + c] += d0 * x; dk[k1 + c] += d1 * x; dk[k2 + c] += d2 * x; dk[k3 + c] += d3 * x;
            }
          }
          for (; j < jn; j++) {
            const ds = row[j], ko = ho + j * D;
            for (let c = 0; c < dh; c++) { dq[qo + c] += ds * k[ko + c]; dk[ko + c] += ds * q[qo + c]; }
          }
        }
      }
    }
  }

  /* Activations come from NA.engine.ACTIVATIONS (f(z) and the slope d(z, a)). Float64 models, as in the gradient
     checks, call them directly. Float32 models read the smooth ones from tables filled from those same functions:
     linear interpolation on steps of 1/512 over [−8, 8], error under 2e-6, the exact function outside. Math.tanh
     and Math.exp cost about 20 ns each, as much as twenty multiply-adds, and GELU alone took a quarter of a
     transformer's training step. ReLU, leaky ReLU and linear are exact and cheap as they are. */
  const TAB_LO = -8, TAB_HI = 8, TAB_N = 8192, TAB_INV = TAB_N / (TAB_HI - TAB_LO);
  const PIECEWISE = { relu: true, leaky: true, linear: true };
  const tables = Object.create(null);
  function activation(id, f64) {
    const fn = NA.engine.ACTIVATIONS[id];
    if (f64 || PIECEWISE[id]) return fn;
    let t = tables[id];
    if (!t || t.src !== fn) {
      const F = new Float32Array(TAB_N + 2), G = new Float32Array(TAB_N + 2);
      for (let i = 0; i < TAB_N + 2; i++) { const z = TAB_LO + i / TAB_INV, a = fn.f(z); F[i] = a; G[i] = fn.d(z, a); }
      const look = (T, z) => { const u = (z - TAB_LO) * TAB_INV, j = u | 0; return T[j] + (T[j + 1] - T[j]) * (u - j); };
      t = tables[id] = Object.assign({}, fn, {
        src: fn, F, G,
        f: (z) => (z > TAB_LO && z < TAB_HI ? look(F, z) : fn.f(z)),
        d: (z, a) => (z > TAB_LO && z < TAB_HI ? look(G, z) : fn.d(z, a)),
      });
    }
    return t;
  }

  // a = f(z) and its backward with one activation for all numbers (tables read inline) ...
  function activate(a, z, n, fn) {
    const F = fn.F;
    if (!F) { const f = fn.f; for (let i = 0; i < n; i++) a[i] = f(z[i]); return; }
    const f = fn.src.f;
    for (let i = 0; i < n; i++) {
      const v = z[i];
      if (v > TAB_LO && v < TAB_HI) { const u = (v - TAB_LO) * TAB_INV, j = u | 0; a[i] = F[j] + (F[j + 1] - F[j]) * (u - j); }
      else a[i] = f(v);
    }
  }
  function activateBack(dz, dy, z, a, n, fn) {
    const G = fn.G;
    if (!G) { const d = fn.d; for (let i = 0; i < n; i++) dz[i] = dy[i] * d(z[i], a[i]); return; }
    const d = fn.src.d;
    for (let i = 0; i < n; i++) {
      const v = z[i];
      if (v > TAB_LO && v < TAB_HI) { const u = (v - TAB_LO) * TAB_INV, j = u | 0; dz[i] = dy[i] * (G[j] + (G[j + 1] - G[j]) * (u - j)); }
      else dz[i] = dy[i] * d(v, a[i]);
    }
  }
  // ... or one per unit, rows of fns.length numbers
  function activateUnits(a, z, rows, fns) {
    const U = fns.length;
    for (let r = 0, i = 0; r < rows; r++) for (let j = 0; j < U; j++, i++) a[i] = fns[j].f(z[i]);
  }
  function activateUnitsBack(dz, dy, z, a, rows, fns) {
    const U = fns.length;
    for (let r = 0, i = 0; r < rows; r++) for (let j = 0; j < U; j++, i++) dz[i] = dy[i] * fns[j].d(z[i], a[i]);
  }

  // ------------------------------------------------------------------ catalogue
  const BLOCKS = {
    dense: { label: 'Dense', defaults: { units: 16, act: 'relu', residual: false } },
    activation: { label: 'Activation', defaults: { act: 'relu' } },
    dropout: { label: 'Dropout', defaults: { p: 0.2 } },
    layernorm: { label: 'LayerNorm', defaults: {} },
    attention: { label: 'Self-attention', defaults: { heads: 2, causal: true, residual: true } },
    transformer: { label: 'Transformer block', defaults: { heads: 2, mlp: 4, causal: true } },
    conv: { label: 'Convolution', defaults: { filters: 8, k: 3, stride: 1, act: 'relu' } },
    pool: { label: 'Max pool', defaults: { size: 2 } },
    flatten: { label: 'Flatten', defaults: {} },
    gap: { label: 'Global average pool', defaults: {} },
    gated: { label: 'Gated (GLU)', defaults: { units: 16, gate: 'swish' } },
    batchnorm: { label: 'Batch norm', defaults: {} },
    noise: { label: 'Gaussian noise', defaults: { sigma: 0.1 } },
    fourier: { label: 'Fourier features', defaults: { features: 16, scale: 1 } },
    rnn: { label: 'Recurrent', defaults: { cell: 'lstm', units: 32 } },
    conv1d: { label: '1-D convolution', defaults: { filters: 16, k: 3, dilation: 1, causal: false, act: 'relu', residual: false } },
    position: { label: 'Positions', defaults: { mode: 'learned' } },
    seqpool: { label: 'Sequence pool', defaults: { mode: 'last' } },
    patches: { label: 'Patches', defaults: { size: 2, dim: 32 } },
    avgpool: { label: 'Average pool', defaults: { size: 2 } },
    parallel: { label: 'Parallel branches', defaults: { merge: 'concat' } },
    sepconv: { label: 'Separable convolution', defaults: { filters: 16, k: 3, dilation: 1, stride: 1, act: 'relu' } },
    se: { label: 'Squeeze-and-excitation', defaults: { ratio: 4 } },
    coords: { label: 'Coordinate channels', defaults: { radius: false } },
    augment: { label: 'Random shifts', defaults: { shift: 2, flip: false } },
    gmp: { label: 'Global max pool', defaults: {} },
    attnpool: { label: 'Attention pool', defaults: {} },
    ssm: { label: 'State-space layer', defaults: { state: 16, units: 16, act: 'gelu' } },
    rbf: { label: 'Radial basis', defaults: { units: 16, width: 0.5 } },
    maxout: { label: 'Maxout', defaults: { units: 16, pieces: 2 } },
    moe: { label: 'Mixture of experts', defaults: { experts: 4, units: 16, top: 2, act: 'relu' } },
  };
  // squeeze-and-excitation's bottleneck: the channels divided by the ratio, rounded half up, at least 1 (in whole
  // numbers, so Python's (2C + r) // 2r gives the same)
  const seHidden = (C, ratio) => Math.max(1, Math.floor((2 * C + ratio) / (2 * ratio)));
  // how parallel branches are joined: side by side (their features, or an image's channels), or number by number
  const MERGES = { concat: 'join', add: 'add', mul: 'multiply', mean: 'average' };
  const LETTERS = 'ABCD';
  const LOSSES = {
    xent: { id: 'xent', label: 'Cross-entropy', outputs: ['classes', 'tokens'] },
    mse: { id: 'mse', label: 'Squared error', outputs: ['classes', 'value'] },
    hinge: { id: 'hinge', label: 'Hinge', outputs: ['classes'] },
    mae: { id: 'mae', label: 'Absolute error', outputs: ['value'] },
    huber: { id: 'huber', label: 'Huber', outputs: ['value'] },
  };
  const OPTIMIZERS = {
    sgd: { id: 'sgd', label: 'SGD' },
    momentum: { id: 'momentum', label: 'Momentum' },
    adam: { id: 'adam', label: 'Adam' },
    adamw: { id: 'adamw', label: 'AdamW' },
  };
  const INITS = ['auto', 'he', 'xavier', 'small'];
  const OUTPUT_NAME = { classes: 'a class output', value: 'a value output', tokens: 'a token output' };

  // ------------------------------------------------------------------ shapes
  const own = (o, key) => typeof key === 'string' && Object.prototype.hasOwnProperty.call(o, key); // not "toString"
  const size = (s) => s.reduce((p, v) => p * v, 1);
  const shapeText = (s) => (s ? s.join(' × ') : '');
  const isInt = (v, min = 1) => Number.isInteger(v) && v >= min;
  const sameShape = (a, b) => a.length === b.length && a.every((v, i) => v === b[i]);
  const hidden = (mlp, D) => Math.max(1, Math.round(mlp * D));
  function describe(s) {
    if (s.length === 3) return `an image (${shapeText(s)})`;
    if (s.length === 2) return `a sequence (${shapeText(s)})`;
    return `a flat vector (${shapeText(s)})`;
  }
  function divisors(D) {
    const out = [];
    for (let h = 1; h <= Math.min(D, 16); h++) if (D % h === 0) out.push(h);
    return out.length > 1 ? out.slice(0, -1).join(', ') + ' or ' + out[out.length - 1] : String(out[0]);
  }
  const unknownAct = (id) => `"${id}" is not an activation function this builder knows.`;

  // The spec with its block's defaults filled in; numbers typed into text boxes may arrive as strings.
  function withDefaults(spec) {
    const out = Object.assign({}, BLOCKS[spec.type].defaults);
    for (const key in spec) {
      let v = spec[key];
      if (v === undefined || v === null) continue;
      if (typeof v === 'string' && typeof out[key] === 'number' && v.trim() !== '' && !isNaN(+v)) v = +v;
      out[key] = v;
    }
    return out;
  }

  function normInput(input) {
    if (!input) return null;
    if (input.kind === 'vector') return { kind: 'vector', dim: +input.dim };
    if (input.kind === 'image') return { kind: 'image', c: +input.c, h: +input.h, w: +input.w };
    if (input.kind === 'tokens') return { kind: 'tokens', vocab: +input.vocab, len: +input.len, embed: +input.embed };
    if (input.kind === 'series') return { kind: 'series', len: +input.len, dim: +input.dim };
    return { kind: input.kind };
  }
  // The shape the first block receives, the embedding's parameter count, or a problem with the input itself.
  function inputShape(input) {
    const inp = normInput(input), bad = (reason) => ({ shape: null, params: 0, problem: { reason, fix: null } });
    if (!inp) return bad('The model needs an input.');
    if (inp.kind === 'vector') return isInt(inp.dim) ? { shape: [inp.dim], params: 0, problem: null } : bad('A vector input needs at least one number per example.');
    if (inp.kind === 'image') {
      return isInt(inp.c) && isInt(inp.h) && isInt(inp.w) ? { shape: [inp.c, inp.h, inp.w], params: 0, problem: null }
        : bad('An image input needs at least one channel and a height and width of at least one pixel.');
    }
    if (inp.kind === 'series') return isInt(inp.len) && isInt(inp.dim) ? { shape: [inp.len, inp.dim], params: 0, problem: null } : bad('A sequence of numbers needs a length and at least one number per position.');
    if (inp.kind === 'tokens') {
      if (!isInt(inp.vocab) || !isInt(inp.len) || !isInt(inp.embed)) return bad('A token input needs a vocabulary, a length and an embedding width of at least 1.');
      return { shape: [inp.len, inp.embed], params: (inp.vocab + inp.len) * inp.embed, problem: null };
    }
    return bad(`"${inp.kind}" is not a kind of input this builder knows.`);
  }

  /* What one block makes of the shape it receives: { out, params, problem }. When the block does not fit, out
     and params are what they would be once the suggested fix is in (or the input passed through unchanged when
     there is no fix), so the blocks after it can still be checked. */
  function blockShape(spec, inS) {
    const ok = (out, params) => ({ out: out.slice(), params, problem: null });
    const fail = (reason, fix = null, out = inS, params = 0) => ({ out: out.slice(), params, problem: { reason, fix } });
    if (!spec || !own(BLOCKS, spec.type)) return fail(`"${spec && spec.type}" is not a kind of block this builder knows.`);
    const s = withDefaults(spec), A = NA.engine.ACTIVATIONS, kind = inS.length; // 1 vector, 2 sequence, 3 image
    switch (spec.type) {
      case 'dense': {
        if (!isInt(s.units)) return fail('A Dense layer needs a whole number of units, at least 1.');
        if (!own(A, s.act)) return fail(unknownAct(s.act));
        const p = (F) => F * s.units + s.units;
        if (kind === 3) return fail(`A Dense layer needs a flat vector, but this is ${describe(inS)}.`, [{ type: 'flatten' }], [s.units], p(size(inS)));
        return ok(kind === 1 ? [s.units] : [inS[0], s.units], p(inS[kind - 1]));
      }
      case 'activation':
        return own(A, s.act) ? ok(inS, 0) : fail(unknownAct(s.act));
      case 'dropout':
        return typeof s.p === 'number' && s.p >= 0 && s.p < 1 ? ok(inS, 0) : fail('Dropout needs a rate of at least 0 and below 1.');
      case 'layernorm':
        if (kind === 3) return fail(`LayerNorm normalises a vector, or each position of a sequence, but this is ${describe(inS)}.`, [{ type: 'flatten' }], [size(inS)], 2 * size(inS));
        return ok(inS, 2 * inS[kind - 1]);
      case 'attention':
      case 'transformer': {
        const name = spec.type === 'attention' ? 'Self-attention' : 'A Transformer block';
        if (kind !== 2) return fail(`${name} needs a sequence (positions × features), but this is ${describe(inS)}.`);
        const D = inS[1];
        if (!isInt(s.heads)) return fail(`${name} needs a whole number of heads, at least 1.`);
        if (D % s.heads) return fail(`The ${D} features cannot be split evenly into ${s.heads} heads; choose a head count that divides ${D}, such as ${divisors(D)}.`);
        const attn = 4 * (D * D + D);
        if (spec.type === 'attention') return ok(inS, attn);
        if (!(typeof s.mlp === 'number' && s.mlp > 0)) return fail('A Transformer block needs an MLP width multiplier above 0.');
        const Hd = hidden(s.mlp, D);
        return ok(inS, attn + 4 * D + D * Hd + Hd + Hd * D + D);
      }
      case 'conv': {
        if (kind !== 3) return fail(`A convolution needs an image (channels × height × width), but this is ${describe(inS)}.`);
        if (!isInt(s.filters)) return fail('A convolution needs a whole number of filters, at least 1.');
        if (!isInt(s.k)) return fail('A convolution needs a kernel size of at least 1.');
        if (!isInt(s.stride)) return fail('A convolution needs a stride of at least 1.');
        if (!own(A, s.act)) return fail(unknownAct(s.act));
        const [C, H, W] = inS;
        return ok([s.filters, Math.ceil(H / s.stride), Math.ceil(W / s.stride)], s.filters * C * s.k * s.k + s.filters);
      }
      case 'gated': {
        if (!isInt(s.units)) return fail('A gated layer needs a whole number of units, at least 1.');
        if (!own(A, s.gate)) return fail(unknownAct(s.gate));
        const p = (F) => 2 * (F * s.units + s.units);
        if (kind === 3) return fail(`A gated layer needs a flat vector or a sequence, but this is ${describe(inS)}.`, [{ type: 'flatten' }], [s.units], p(size(inS)));
        return ok(kind === 1 ? [s.units] : [inS[0], s.units], p(inS[kind - 1]));
      }
      case 'batchnorm':
        return ok(inS, 2 * (kind === 3 ? inS[0] : inS[kind - 1]));
      case 'noise':
        return typeof s.sigma === 'number' && s.sigma >= 0 ? ok(inS, 0) : fail('Gaussian noise needs a spread σ of at least 0.');
      case 'fourier':
        if (kind !== 1) return fail(`Fourier features work on a flat vector of numbers, but this is ${describe(inS)}.`, kind === 3 ? [{ type: 'flatten' }] : null);
        if (!isInt(s.features)) return fail('Fourier features need a whole number of frequencies, at least 1.');
        if (!(typeof s.scale === 'number' && s.scale > 0)) return fail('Fourier features need a frequency spread σ above 0.');
        return ok([inS[0] + 2 * s.features], 0);
      case 'rnn': {
        if (kind !== 2) return fail(`A recurrent layer reads a sequence (positions × features), but this is ${describe(inS)}.`);
        if (!isInt(s.units)) return fail('A recurrent layer needs a whole number of units, at least 1.');
        if (!own(GATES, s.cell)) return fail(`"${s.cell}" is not a recurrent cell this builder knows (RNN, LSTM or GRU).`);
        const G = GATES[s.cell], H = s.units;
        return ok([inS[0], H], G * H * (inS[1] + H + 2));
      }
      case 'conv1d': {
        if (kind !== 2) return fail(`A 1-D convolution slides along a sequence (positions × features), but this is ${describe(inS)}; for a picture, use Conv.`);
        if (!isInt(s.filters)) return fail('A 1-D convolution needs a whole number of filters, at least 1.');
        if (!isInt(s.k)) return fail('A 1-D convolution needs a window of at least 1 position.');
        if (!isInt(s.dilation)) return fail('A 1-D convolution needs a dilation of at least 1.');
        if (!own(A, s.act)) return fail(unknownAct(s.act));
        return ok([inS[0], s.filters], s.filters * s.k * inS[1] + s.filters);
      }
      case 'position':
        if (kind !== 2) return fail(`Positions are added to a sequence (positions × features), but this is ${describe(inS)}.`);
        return ok(inS, s.mode === 'sine' ? 0 : size(inS));
      case 'seqpool':
        if (kind !== 2) return fail(`Sequence pooling squeezes a sequence (positions × features) into one vector, but this is ${describe(inS)}.`, kind === 3 ? [{ type: 'gap' }] : null, kind === 3 ? [inS[0]] : inS);
        return ok([inS[1]], 0);
      case 'patches': {
        if (kind !== 3) return fail(`Patches cut up an image (channels × height × width), but this is ${describe(inS)}.`);
        if (!isInt(s.size)) return fail('Patches need a size of at least 1 pixel.');
        if (!isInt(s.dim)) return fail('Patches need a width (numbers per patch) of at least 1.');
        const [C, H, W] = inS, N = Math.floor(H / s.size) * Math.floor(W / s.size);
        if (!N) return fail(`A ${s.size} × ${s.size} patch is bigger than this image (${shapeText(inS)}).`);
        return ok([N, s.dim], s.dim * C * s.size * s.size + s.dim + N * s.dim);
      }
      case 'avgpool': {
        if (kind !== 3) return fail(`Average pooling needs an image (channels × height × width), but this is ${describe(inS)}; for a sequence, use sequence pooling.`);
        if (!isInt(s.size)) return fail('Average pooling needs a window size of at least 1.');
        const [C, H, W] = inS, Ho = Math.floor(H / s.size), Wo = Math.floor(W / s.size);
        if (!Ho || !Wo) return fail(`The ${s.size} × ${s.size} pooling window is bigger than this image (${shapeText(inS)}).`);
        return ok([C, Ho, Wo], 0);
      }
      case 'pool': {
        if (kind !== 3) return fail(`Max pooling needs an image (channels × height × width), but this is ${describe(inS)}.`);
        if (!isInt(s.size)) return fail('Max pooling needs a window size of at least 1.');
        const [C, H, W] = inS, Ho = Math.floor(H / s.size), Wo = Math.floor(W / s.size);
        if (!Ho || !Wo) return fail(`The ${s.size} × ${s.size} pooling window is bigger than this image (${shapeText(inS)}).`);
        return ok([C, Ho, Wo], 0);
      }
      case 'sepconv': {
        if (kind !== 3) return fail(`A separable convolution needs an image (channels × height × width), but this is ${describe(inS)}; for a sequence, use the 1-D convolution.`);
        if (!isInt(s.filters)) return fail('A separable convolution needs a whole number of filters, at least 1.');
        if (!isInt(s.k) || s.k % 2 === 0) return fail('A separable convolution needs an odd filter size (1, 3, 5 or 7), so that every window has a centre pixel.');
        if (!isInt(s.dilation)) return fail('A separable convolution needs a dilation of at least 1.');
        if (!isInt(s.stride)) return fail('A separable convolution needs a stride of at least 1.');
        if (!own(A, s.act)) return fail(unknownAct(s.act));
        const [C, H, W] = inS;
        return ok([s.filters, Math.ceil(H / s.stride), Math.ceil(W / s.stride)], C * s.k * s.k + C + s.filters * C + s.filters);
      }
      case 'se': {
        if (!isInt(s.ratio)) return fail('Squeeze-and-excitation needs a whole-number squeeze ratio, at least 1.');
        const C = kind === 3 ? inS[0] : inS[kind - 1], h = seHidden(C, s.ratio);
        return ok(inS, 2 * C * h + h + C);
      }
      case 'coords':
        if (kind !== 3) return fail(`Coordinate channels are added to an image (channels × height × width), but this is ${describe(inS)}; a sequence gets its positions from the Positions block.`);
        return ok([inS[0] + (s.radius ? 3 : 2), inS[1], inS[2]], 0);
      case 'augment':
        if (kind !== 3) return fail(`Random shifts move pictures about, but this is ${describe(inS)}; for numbers, Gaussian noise does a similar job.`);
        if (!isInt(s.shift, 0)) return fail('Random shifts need a whole number of pixels, 0 or more.');
        return ok(inS, 0);
      case 'gmp':
        if (kind === 1) return fail(`Global max pooling keeps the largest value over the positions of an image or a sequence, but this is already ${describe(inS)}.`);
        return ok(kind === 3 ? [inS[0]] : [inS[1]], 0);
      case 'attnpool':
        if (kind === 1) return fail(`Attention pooling weighs the positions of a sequence (or the pixels of an image) into one vector, but this is already ${describe(inS)}.`);
        return ok(kind === 3 ? [inS[0]] : [inS[1]], kind === 3 ? inS[0] : inS[1]);
      case 'ssm': {
        if (kind !== 2) return fail(`A state-space layer reads a sequence (positions × features) one step at a time, but this is ${describe(inS)}.`);
        if (!isInt(s.state)) return fail('A state-space layer needs a state of at least 1 number.');
        if (!isInt(s.units)) return fail('A state-space layer needs a whole number of output units, at least 1.');
        if (!own(A, s.act)) return fail(unknownAct(s.act));
        const N = s.state;
        return ok([inS[0], s.units], N + N * inS[1] + s.units * N + s.units);
      }
      case 'rbf': {
        if (!isInt(s.units)) return fail('A radial basis layer needs a whole number of units, at least 1.');
        if (!(typeof s.width === 'number' && s.width > 0)) return fail('A radial basis layer needs a starting width above 0.');
        const p = (F) => s.units * F + s.units;
        if (kind === 3) return fail(`A radial basis layer measures distances between flat vectors, but this is ${describe(inS)}.`, [{ type: 'flatten' }], [s.units], p(size(inS)));
        return ok(kind === 1 ? [s.units] : [inS[0], s.units], p(inS[kind - 1]));
      }
      case 'maxout': {
        if (!isInt(s.units)) return fail('A maxout layer needs a whole number of units, at least 1.');
        if (!isInt(s.pieces)) return fail('A maxout layer needs at least 1 piece per unit (2 or more make it bend).');
        const p = (F) => s.pieces * s.units * (F + 1);
        if (kind === 3) return fail(`A maxout layer needs a flat vector or a sequence, but this is ${describe(inS)}.`, [{ type: 'flatten' }], [s.units], p(size(inS)));
        return ok(kind === 1 ? [s.units] : [inS[0], s.units], p(inS[kind - 1]));
      }
      case 'moe': {
        if (!isInt(s.experts, 2)) return fail('A mixture of experts needs at least 2 experts.');
        if (!isInt(s.units)) return fail('A mixture of experts needs a whole number of units in each expert, at least 1.');
        if (!isInt(s.top, 0) || s.top > s.experts) return fail(`A mixture of experts sends each example to its top 1 to ${s.experts} experts (or 0: to all of them, weighted).`);
        if (!own(A, s.act)) return fail(unknownAct(s.act));
        const E = s.experts, p = (F) => E * (F * s.units + s.units) + F * E + E;
        if (kind === 3) return fail(`A mixture of experts needs a flat vector or a sequence, but this is ${describe(inS)}.`, [{ type: 'flatten' }], [s.units], p(size(inS)));
        return ok(kind === 1 ? [s.units] : [inS[0], s.units], p(inS[kind - 1]));
      }
      case 'flatten':
        return ok([size(inS)], 0);
      case 'gap':
        if (kind === 1) return fail(`Global average pooling averages over the positions of an image or a sequence, but this is already ${describe(inS)}.`);
        return ok(kind === 3 ? [inS[0]] : [inS[1]], 0);
      case 'parallel': {
        const list = Array.isArray(spec.branches) ? spec.branches : [];
        if (list.length < 2 || list.length > 4 || !list.every(Array.isArray)) return fail('Parallel branches need 2 to 4 branches, each a list of blocks (an empty one passes the input through).');
        if (!own(MERGES, s.merge)) return fail(`"${s.merge}" is not a way to join branches this builder knows (join, add, multiply or average).`);
        const lanes = list.map((bl) => stackShape(bl, inS)), outs = lanes.map((q) => q.out), params = lanes.reduce((t, q) => t + q.params, 0);
        const r = (res) => Object.assign(res, { branches: lanes.map((q) => q.shapes) });
        const bad = lanes.findIndex((q) => q.problem);
        if (bad >= 0) {
          const k = lanes[bad].shapes.findIndex((q) => q.problem);
          return r(fail(`Branch ${LETTERS[bad]}, block ${k + 1}: ${lanes[bad].problem.reason}`, null, inS, params));
        }
        const R = outs[0].length;
        if (s.merge !== 'concat') {
          if (!outs.every((o) => sameShape(o, outs[0]))) return r(fail(`To ${MERGES[s.merge]} the branches number by number, each must give the same shape, but they give ${outs.map((o, b) => `${LETTERS[b]} ${shapeText(o)}`).join(', ')}. Join them side by side instead, or change a branch.`, null, outs[0], params));
          return r(ok(outs[0], params));
        }
        // joined side by side: a vector's numbers, each position's features, or an image's channels
        if (!outs.every((o) => o.length === R)) return r(fail(`Joined branches must give the same kind of data, but they give ${outs.map((o, b) => `${LETTERS[b]}: ${describe(o)}`).join(', ')}. Flatten or pool the ones that differ.`, null, outs[0], params));
        if (R === 2 && !outs.every((o) => o[0] === outs[0][0])) return r(fail(`Joined sequences must have the same positions, but they have ${outs.map((o) => o[0]).join(', ')}.`, null, outs[0], params));
        if (R === 3 && !outs.every((o) => o[1] === outs[0][1] && o[2] === outs[0][2])) return r(fail(`Joined images must have the same height and width, but they are ${outs.map((o) => `${o[1]} × ${o[2]}`).join(', ')}. Pool or stride the larger ones the same way.`, null, outs[0], params));
        const sum = (j) => outs.reduce((t, o) => t + o[j], 0);
        return r(ok(R === 1 ? [sum(0)] : R === 2 ? [outs[0][0], sum(1)] : [sum(0), outs[0][1], outs[0][2]], params));
      }
    }
    return fail(`"${spec.type}" is not a kind of block this builder knows.`);
  }
  // A stack of blocks (one parallel branch) on a shape: each block's shape, and the first problem among them.
  function stackShape(list, inS) {
    let shape = inS, params = 0, problem = null;
    const shapes = [];
    for (const spec of list) {
      const r = spec && spec.type === 'parallel' ? { out: shape.slice(), params: 0, problem: { reason: 'A branch cannot hold more parallel branches: put the second set after this one instead.', fix: null } } : blockShape(spec, shape);
      shapes.push({ in: shape.slice(), out: r.out, params: r.params, problem: r.problem });
      if (r.problem && !problem) problem = r.problem;
      params += r.params; shape = r.out;
    }
    return { shapes, out: shape, params, problem };
  }

  // The output head on the last shape: { out, params, problem }, out and params again as if any fix were in.
  function outputShape(output, inS, input, nBlocks) {
    const fail = (reason, fix = null, out = null, params = 0) => ({ out, params, problem: { reason, fix } });
    if (!output) return fail('The model needs an output.');
    const where = nBlocks ? 'the last block gives' : 'the input is', kind = inS.length;
    if (output.kind === 'classes' || output.kind === 'value') {
      const cls = output.kind === 'classes', K = output.k === undefined && !cls ? 1 : +output.k;
      if (cls && !isInt(K, 2)) return fail('A class output needs at least two classes.');
      if (!isInt(K)) return fail('A value output needs at least one number to predict.');
      const what = cls ? 'A class output' : 'A value output';
      if (kind === 3) return fail(`${what} needs a flat vector, but ${where} ${describe(inS)}; flatten it, or average each channel with global average pooling.`, [{ type: 'flatten' }], [K], size(inS) * K + K);
      if (kind === 2) return fail(`${what} needs a flat vector, but ${where} ${describe(inS)}; average over the positions with global average pooling, or flatten it.`, [{ type: 'gap' }], [K], inS[1] * K + K);
      return { out: [K], params: inS[0] * K + K, problem: null };
    }
    if (output.kind === 'tokens') {
      const V = output.vocab !== undefined ? +output.vocab : input && input.kind === 'tokens' ? +input.vocab : NaN;
      if (!isInt(V, 2)) return fail('A token output needs a vocabulary of at least two tokens.');
      if (kind !== 2) return fail(`A token output predicts a token at every position, so it needs a sequence (positions × features), but ${where} ${describe(inS)}.`);
      return { out: [inS[0], V], params: inS[1] * V + V, problem: null };
    }
    return fail(`"${output.kind}" is not a kind of output this builder knows.`);
  }

  // Validates a model without allocating it.
  function check(blocks, input, output) {
    blocks = Array.isArray(blocks) ? blocks : [];
    const res = { ok: true, shapes: [], outShape: null, problem: null, params: 0 }, inp = inputShape(input);
    if (inp.problem) {
      res.ok = false; res.problem = inp.problem;
      res.shapes = blocks.map(() => ({ in: null, out: null, params: 0, problem: null }));
      return res;
    }
    let shape = inp.shape, params = inp.params;
    for (const spec of blocks) {
      const r = blockShape(spec, shape), e = { in: shape.slice(), out: r.out, params: r.params, problem: r.problem };
      if (r.branches) e.branches = r.branches; // parallel branches: the shapes of the blocks in each
      res.shapes.push(e);
      if (r.problem) res.ok = false;
      params += r.params; shape = r.out;
    }
    const o = outputShape(output, shape, input, blocks.length);
    res.outShape = o.out; res.problem = o.problem; res.params = params + o.params;
    if (o.problem) res.ok = false;
    return res;
  }

  // ------------------------------------------------------------------ parameters
  function param(ctx, name, shape, decay) {
    return { name, shape, w: new ctx.Arr(size(shape)), g: new ctx.Arr(size(shape)), decay };
  }
  function gauss(ctx, p, sd, from = 0, to = p.w.length) { for (let i = from; i < to; i++) p.w[i] = ctx.rng.gauss(0, sd); }
  // Standard deviation for a weight matrix with fanIn inputs: He when the following activation is a rectifier.
  function initSd(mode, fanIn, he) {
    if (mode === 'small') return 0.02;
    return Math.sqrt((mode === 'he' || (mode === 'auto' && he) ? 2 : 1) / fanIn);
  }
  // attention projections and the output layer are always scaled for a linear output
  const linearInit = (ctx) => (ctx.init === 'small' ? 'small' : 'xavier');

  // ------------------------------------------------------------------ layers
  /* Every layer: alloc(B) sizes its buffers for a batch of B; forward(x, B, train) returns its output buffer;
     backward(dy, B, needDx) accumulates its parameter gradients and returns the gradient for its input (or null
     when nothing before it learns). Layers keep a reference to their input for the backward pass and never write
     into it. */

  // Token embedding plus learned position embedding: ids [T] → [T, D].
  class Embed {
    constructor(ctx, V, T, D) {
      Object.assign(this, { type: 'embed', spec: null, inShape: [T], outShape: [T, D], V, T, D, mem: new Mem(ctx.Arr) });
      this.tok = param(ctx, 'token embedding', [V, D], false);
      this.pos = param(ctx, 'position embedding', [T, D], false);
      gauss(ctx, this.tok, 0.1); gauss(ctx, this.pos, 0.1);
      this.params = [this.tok, this.pos];
    }
    alloc(B) { this.out = this.mem.view('out', B * this.T * this.D); this.ids = this.mem.view('ids', B * this.T, Int32Array); }
    forward(X, xo, B) {
      const { V, T, D, out, ids } = this, E = this.tok.w, P = this.pos.w;
      for (let i = 0, n = B * T; i < n; i++) {
        const v = X[xo + i], id = v | 0;
        if (id !== v || id < 0 || id >= V) throw new Error(`Token id ${v} is outside the vocabulary (0 to ${V - 1}).`);
        ids[i] = id;
        const eo = id * D, po = (i % T) * D, oo = i * D;
        for (let c = 0; c < D; c++) out[oo + c] = E[eo + c] + P[po + c];
      }
      return out;
    }
    backward(dy, B) {
      const { T, D, ids } = this, gE = this.tok.g, gP = this.pos.g;
      for (let i = 0, n = B * T; i < n; i++) {
        const eo = ids[i] * D, po = (i % T) * D, oo = i * D;
        for (let c = 0; c < D; c++) { const g = dy[oo + c]; gE[eo + c] += g; gP[po + c] += g; }
      }
    }
  }

  // z = x·Wᵀ + b then the activation (each unit its own with acts); on a sequence, the same at every position.
  class Dense {
    constructor(spec, cfg, inS, outS, ctx, isOutput) {
      const A = NA.engine.ACTIVATIONS, nIn = inS[inS.length - 1], U = outS[outS.length - 1];
      Object.assign(this, { type: 'dense', spec, inShape: inS, outShape: outS, nIn, U, rows: inS.length === 2 ? inS[0] : 1, scr: ctx.scr, mem: new Mem(ctx.Arr) });
      this.isOutput = !!isOutput;
      this.W = param(ctx, 'weights', [U, nIn], true);
      this.b = param(ctx, 'bias', [U], false);
      this.params = [this.W, this.b];
      const base = activation(own(A, cfg.act) ? cfg.act : 'linear', ctx.f64);
      let fns = Array.isArray(cfg.acts) ? Array.from({ length: U }, (_, j) => (own(A, cfg.acts[j]) ? activation(cfg.acts[j], ctx.f64) : base)) : null;
      if (fns && fns.every((f) => f === fns[0])) { this.fn = fns[0]; fns = null; } else this.fn = base;
      this.fns = fns;
      this.linear = !fns && this.fn.id === 'linear';
      this.residual = !!cfg.residual && sameShape(inS, outS);
      const mode = isOutput ? linearInit(ctx) : ctx.init;
      for (let j = 0; j < U; j++) gauss(ctx, this.W, initSd(mode, nIn, (fns ? fns[j] : this.fn).he), j * nIn, (j + 1) * nIn);
    }
    alloc(B) {
      const m = this.mem, N = B * this.rows, n = N * this.U;
      this.z = m.view('z', n);
      this.a = this.linear ? this.z : m.view('a', n);
      this.out = this.residual ? m.view('out', n) : this.a;
      this.dz = this.linear ? null : m.view('dz', n);
      this.dx = m.view('dx', N * this.nIn);
    }
    forward(x, B) {
      const N = B * this.rows, n = N * this.U;
      this.x = x;
      linear(this.z, x, this.W.w, this.b.w, N, this.nIn, this.U);
      if (this.fns) activateUnits(this.a, this.z, N, this.fns);
      else if (!this.linear) activate(this.a, this.z, n, this.fn);
      if (this.residual) { const out = this.out, a = this.a; for (let i = 0; i < n; i++) out[i] = x[i] + a[i]; }
      return this.out;
    }
    backward(dy, B, needDx) {
      const N = B * this.rows, n = N * this.U, dx = needDx ? this.dx : null;
      let dz = dy;
      if (this.fns) { dz = this.dz; activateUnitsBack(dz, dy, this.z, this.a, N, this.fns); }
      else if (!this.linear) { dz = this.dz; activateBack(dz, dy, this.z, this.a, n, this.fn); }
      linearBack(this.scr, dx, false, this.W.g, this.b.g, dz, this.x, this.W.w, N, this.nIn, this.U);
      if (dx && this.residual) for (let i = 0; i < n; i++) dx[i] += dy[i];
      return dx;
    }
  }

  class Act {
    constructor(spec, cfg, inS, outS, ctx) {
      Object.assign(this, { type: 'activation', spec, inShape: inS, outShape: outS, params: [], n: size(inS), fn: activation(cfg.act, ctx.f64), mem: new Mem(ctx.Arr) });
    }
    alloc(B) { this.out = this.mem.view('out', B * this.n); this.dx = this.mem.view('dx', B * this.n); }
    forward(x, B) { this.x = x; activate(this.out, x, B * this.n, this.fn); return this.out; }
    backward(dy, B, needDx) {
      if (!needDx) return null;
      activateBack(this.dx, dy, this.x, this.out, B * this.n, this.fn);
      return this.dx;
    }
  }

  // Inverted dropout: kept numbers are scaled by 1/(1 − p) while training, so evaluation is the plain identity.
  class Dropout {
    constructor(spec, cfg, inS, outS, ctx) {
      Object.assign(this, { type: 'dropout', spec, inShape: inS, outShape: outS, params: [], p: cfg.p, n: size(inS), model: ctx.model, mem: new Mem(ctx.Arr) });
    }
    alloc(B) {
      const n = B * this.n;
      this.buf = this.mem.view('out', n); this.mask = this.mem.view('mask', n); this.dx = this.mem.view('dx', n);
      this.out = this.buf;
    }
    forward(x, B, train) {
      this.train = train;
      if (!train) return (this.out = x);
      const n = B * this.n, p = this.p, keep = 1 / (1 - p), rng = this.model.rng, out = this.buf, mask = this.mask;
      for (let i = 0; i < n; i++) { const m = rng.next() < p ? 0 : keep; mask[i] = m; out[i] = x[i] * m; }
      return (this.out = out);
    }
    backward(dy, B, needDx) {
      if (!needDx) return null;
      if (!this.train) return dy;
      const n = B * this.n, dx = this.dx, mask = this.mask;
      for (let i = 0; i < n; i++) dx[i] = dy[i] * mask[i];
      return dx;
    }
  }

  // LayerNorm over the last dimension: the whole vector, or each position of a sequence.
  class LayerNormBlock {
    constructor(spec, cfg, inS, outS, ctx) {
      const d = inS[inS.length - 1];
      Object.assign(this, { type: 'layernorm', spec, inShape: inS, outShape: outS, d, rows: inS.length === 2 ? inS[0] : 1, mem: new Mem(ctx.Arr) });
      this.g = param(ctx, 'gain', [d], false); this.g.w.fill(1);
      this.b = param(ctx, 'bias', [d], false);
      this.params = [this.g, this.b];
    }
    alloc(B) {
      const m = this.mem, n = B * this.rows, nd = n * this.d;
      this.out = m.view('out', nd); this.xhat = m.view('xhat', nd); this.rstd = m.view('rstd', n); this.dx = m.view('dx', nd);
    }
    forward(x, B) { layerNorm(this.out, this.xhat, this.rstd, x, this.g.w, this.b.w, B * this.rows, this.d); return this.out; }
    backward(dy, B, needDx) {
      const dx = needDx ? this.dx : null;
      if (dx) dx.fill(0);
      layerNormBack(dx, this.g.g, this.b.g, dy, this.xhat, this.rstd, this.g.w, B * this.rows, this.d);
      return dx;
    }
  }

  // The projections and per-head attention shared by the attention and transformer blocks, without a residual.
  class AttnCore {
    constructor(ctx, T, D, H, causal) {
      Object.assign(this, { T, D, H, causal, scr: ctx.scr, mem: new Mem(ctx.Arr) });
      const sd = initSd(linearInit(ctx), D);
      const W = (name) => { const p = param(ctx, name, [D, D], true); gauss(ctx, p, sd); return p; };
      const b = (name) => param(ctx, name, [D], false);
      this.Wq = W('Wq'); this.bq = b('bq'); this.Wk = W('Wk'); this.bk = b('bk');
      this.Wv = W('Wv'); this.bv = b('bv'); this.Wo = W('Wo'); this.bo = b('bo');
      this.params = [this.Wq, this.bq, this.Wk, this.bk, this.Wv, this.bv, this.Wo, this.bo];
    }
    alloc(B) {
      const m = this.mem, n = B * this.T * this.D;
      this.q = m.view('q', n); this.k = m.view('k', n); this.v = m.view('v', n); this.o = m.view('o', n); this.y = m.view('y', n);
      this.dq = m.view('dq', n); this.dk = m.view('dk', n); this.dv = m.view('dv', n); this.dO = m.view('dO', n);
      this.att = m.view('att', B * this.H * this.T * this.T);
      this.row = m.get('row', this.T, Float64Array);
    }
    forward(x, B) {
      const N = B * this.T, D = this.D;
      this.x = x;
      linear(this.q, x, this.Wq.w, this.bq.w, N, D, D);
      linear(this.k, x, this.Wk.w, this.bk.w, N, D, D);
      linear(this.v, x, this.Wv.w, this.bv.w, N, D, D);
      attend(this.o, this.att, this.q, this.k, this.v, B, this.T, D, this.H, this.causal, this.row);
      linear(this.y, this.o, this.Wo.w, this.bo.w, N, D, D);
      return this.y;
    }
    // Writes the gradient for x into dx (added when acc; skipped when dx is null).
    backward(dy, B, dx, acc) {
      const N = B * this.T, D = this.D, s = this.scr, xT = s.get('attnXT', N * D);
      linearBack(s, this.dO, false, this.Wo.g, this.bo.g, dy, this.o, this.Wo.w, N, D, D);
      this.dq.fill(0); this.dk.fill(0); this.dv.fill(0);
      attendBack(this.dq, this.dk, this.dv, this.dO, this.att, this.q, this.k, this.v, B, this.T, D, this.H, this.causal, this.row);
      transpose(xT, this.x, 0, N, D);
      linearBack(s, dx, acc, this.Wq.g, this.bq.g, this.dq, this.x, this.Wq.w, N, D, D, xT);
      linearBack(s, dx, true, this.Wk.g, this.bk.g, this.dk, this.x, this.Wk.w, N, D, D, xT);
      linearBack(s, dx, true, this.Wv.g, this.bv.g, this.dv, this.x, this.Wv.w, N, D, D, xT);
    }
  }

  class Attention {
    constructor(spec, cfg, inS, outS, ctx) {
      const [T, D] = inS;
      Object.assign(this, { type: 'attention', spec, inShape: inS, outShape: outS, T, D, heads: cfg.heads, causal: cfg.causal !== false, residual: cfg.residual !== false, mem: new Mem(ctx.Arr) });
      this.core = new AttnCore(ctx, T, D, this.heads, this.causal);
      this.params = this.core.params;
    }
    alloc(B) {
      const n = B * this.T * this.D;
      this.core.alloc(B);
      this.out = this.residual ? this.mem.view('out', n) : this.core.y;
      this.dx = this.mem.view('dx', n);
      this.attn = this.core.att;
    }
    forward(x, B) {
      const y = this.core.forward(x, B);
      if (this.residual) { const out = this.out, n = B * this.T * this.D; for (let i = 0; i < n; i++) out[i] = x[i] + y[i]; }
      return this.out;
    }
    backward(dy, B, needDx) {
      const dx = needDx ? this.dx : null;
      this.core.backward(dy, B, dx, false);
      if (dx && this.residual) { const n = B * this.T * this.D; for (let i = 0; i < n; i++) dx[i] += dy[i]; }
      return dx;
    }
  }

  // Pre-LayerNorm block: x + Attn(LN1(x)), then that + W2·GELU(W1·LN2(·) + b1) + b2.
  class Transformer {
    constructor(spec, cfg, inS, outS, ctx) {
      const [T, D] = inS, Hd = hidden(cfg.mlp, D);
      Object.assign(this, { type: 'transformer', spec, inShape: inS, outShape: outS, T, D, Hd, heads: cfg.heads, causal: cfg.causal !== false, scr: ctx.scr, mem: new Mem(ctx.Arr) });
      this.gelu = activation('gelu', ctx.f64);
      this.ln1g = param(ctx, 'LN1 gain', [D], false); this.ln1g.w.fill(1);
      this.ln1b = param(ctx, 'LN1 bias', [D], false);
      this.core = new AttnCore(ctx, T, D, this.heads, this.causal);
      this.ln2g = param(ctx, 'LN2 gain', [D], false); this.ln2g.w.fill(1);
      this.ln2b = param(ctx, 'LN2 bias', [D], false);
      this.W1 = param(ctx, 'W1', [Hd, D], true); this.b1 = param(ctx, 'b1', [Hd], false);
      this.W2 = param(ctx, 'W2', [D, Hd], true); this.b2 = param(ctx, 'b2', [D], false);
      gauss(ctx, this.W1, initSd(ctx.init, D, this.gelu.he));
      gauss(ctx, this.W2, initSd(ctx.init, Hd, false));
      this.params = [this.ln1g, this.ln1b, ...this.core.params, this.ln2g, this.ln2b, this.W1, this.b1, this.W2, this.b2];
    }
    alloc(B) {
      const m = this.mem, N = B * this.T, n = N * this.D, nh = N * this.Hd;
      this.core.alloc(B);
      for (const k of ['h1', 'xh1', 'mid', 'h2', 'xh2', 'mo', 'out', 'dmid', 'dh', 'dx']) this[k] = m.view(k, n);
      this.r1 = m.view('r1', N); this.r2 = m.view('r2', N);
      this.u = m.view('u', nh); this.g = m.view('g', nh); this.du = m.view('du', nh); this.dg = m.view('dg', nh);
      this.attn = this.core.att;
    }
    forward(x, B) {
      const N = B * this.T, D = this.D, Hd = this.Hd, n = N * D, { mid, out, mo } = this;
      layerNorm(this.h1, this.xh1, this.r1, x, this.ln1g.w, this.ln1b.w, N, D);
      const y = this.core.forward(this.h1, B);
      for (let i = 0; i < n; i++) mid[i] = x[i] + y[i];
      layerNorm(this.h2, this.xh2, this.r2, mid, this.ln2g.w, this.ln2b.w, N, D);
      linear(this.u, this.h2, this.W1.w, this.b1.w, N, D, Hd);
      activate(this.g, this.u, N * Hd, this.gelu);
      linear(mo, this.g, this.W2.w, this.b2.w, N, Hd, D);
      for (let i = 0; i < n; i++) out[i] = mid[i] + mo[i];
      return out;
    }
    backward(dy, B, needDx) {
      const N = B * this.T, D = this.D, Hd = this.Hd, n = N * D, s = this.scr, { dmid, dh } = this;
      // out = mid + MLP(LN2(mid))
      linearBack(s, this.dg, false, this.W2.g, this.b2.g, dy, this.g, this.W2.w, N, Hd, D);
      activateBack(this.du, this.dg, this.u, this.g, N * Hd, this.gelu);
      linearBack(s, dh, false, this.W1.g, this.b1.g, this.du, this.h2, this.W1.w, N, D, Hd);
      for (let i = 0; i < n; i++) dmid[i] = dy[i];
      layerNormBack(dmid, this.ln2g.g, this.ln2b.g, dh, this.xh2, this.r2, this.ln2g.w, N, D);
      // mid = x + Attn(LN1(x)); dh is reused for the gradient reaching LN1's output
      this.core.backward(dmid, B, dh, false);
      const dx = needDx ? this.dx : null;
      if (dx) for (let i = 0; i < n; i++) dx[i] = dmid[i];
      layerNormBack(dx, this.ln1g.g, this.ln1b.g, dh, this.xh1, this.r1, this.ln1g.w, N, D);
      return dx;
    }
  }

  /* 2-D convolution with 'same' zero padding (pad = floor(k/2) on the top and left), output ceil(H/stride) ×
     ceil(W/stride). Each example's input is unrolled into patches (im2col) so the convolution becomes one
     matrix product: forward with a patch per row, backward with the patches transposed. */
  class Conv {
    constructor(spec, cfg, inS, outS, ctx) {
      const [C, H, W] = inS, F = outS[0], k = cfg.k, A = NA.engine.ACTIVATIONS;
      Object.assign(this, { type: 'conv', spec, inShape: inS, outShape: outS, C, H, W, F, k, s: cfg.stride, pad: k >> 1, Ho: outS[1], Wo: outS[2], scr: ctx.scr, mem: new Mem(ctx.Arr) });
      this.J = C * k * k; this.Q = this.Ho * this.Wo;
      this.Wt = param(ctx, 'kernels', [F, C, k, k], true);
      this.b = param(ctx, 'bias', [F], false);
      this.params = [this.Wt, this.b];
      this.fn = activation(own(A, cfg.act) ? cfg.act : 'relu', ctx.f64);
      this.linear = this.fn.id === 'linear';
      this.residual = !!cfg.residual && F === C && cfg.stride === 1; // ResNet-style: the input added back when the shape is kept
      gauss(ctx, this.Wt, initSd(ctx.init, this.J, this.fn.he));
    }
    alloc(B) {
      const m = this.mem, n = B * this.F * this.Q;
      this.z = m.view('z', n);
      this.a = this.linear ? this.z : m.view('a', n);
      this.out = this.residual ? m.view('out', n) : this.a;
      this.dz = this.linear ? null : m.view('dz', n);
      this.dx = m.view('dx', B * this.C * this.H * this.W);
    }
    // Range of output columns whose window column kx lands inside the image.
    span(kx) {
      const { W, s, pad, Wo } = this;
      return [Math.max(0, Math.ceil((pad - kx) / s)), Math.min(Wo, Math.floor((W - 1 + pad - kx) / s) + 1)];
    }
    // P[q, j]: row q is output pixel q, column j is (channel, ky, kx); zero where the window hangs over the edge.
    im2row(P, x, xo) {
      const { C, H, W, k, s, pad, Ho, Wo, J } = this;
      P.fill(0, 0, Ho * Wo * J);
      for (let c = 0, j = 0; c < C; c++) for (let ky = 0; ky < k; ky++) for (let kx = 0; kx < k; kx++, j++) {
        const [ox0, ox1] = this.span(kx);
        for (let oy = 0; oy < Ho; oy++) {
          const iy = oy * s + ky - pad;
          if (iy < 0 || iy >= H) continue;
          const row = xo + (c * H + iy) * W + kx - pad;
          for (let ox = ox0, q = (oy * Wo + ox0) * J + j; ox < ox1; ox++, q += J) P[q] = x[row + ox * s];
        }
      }
    }
    // The same patches transposed, Pc[j, q], so the backward products read contiguous rows.
    im2col(Pc, x, xo) {
      const { C, H, W, k, s, pad, Ho, Wo, Q } = this;
      Pc.fill(0, 0, this.J * Q);
      for (let c = 0, j = 0; c < C; c++) for (let ky = 0; ky < k; ky++) for (let kx = 0; kx < k; kx++, j++) {
        const [ox0, ox1] = this.span(kx);
        for (let oy = 0; oy < Ho; oy++) {
          const iy = oy * s + ky - pad;
          if (iy < 0 || iy >= H) continue;
          const row = xo + (c * H + iy) * W + kx - pad, po = j * Q + oy * Wo;
          for (let ox = ox0; ox < ox1; ox++) Pc[po + ox] = x[row + ox * s];
        }
      }
    }
    // Adds patch gradients back onto the input pixels they were read from (overlapping windows add up).
    col2im(dx, xo, dPc) {
      const { C, H, W, k, s, pad, Ho, Wo, Q } = this;
      for (let c = 0, j = 0; c < C; c++) for (let ky = 0; ky < k; ky++) for (let kx = 0; kx < k; kx++, j++) {
        const [ox0, ox1] = this.span(kx);
        for (let oy = 0; oy < Ho; oy++) {
          const iy = oy * s + ky - pad;
          if (iy < 0 || iy >= H) continue;
          const row = xo + (c * H + iy) * W + kx - pad, po = j * Q + oy * Wo;
          for (let ox = ox0; ox < ox1; ox++) dx[row + ox * s] += dPc[po + ox];
        }
      }
    }
    forward(x, B) {
      const { F, Q, J, z } = this, CHW = this.C * this.H * this.W, P = this.scr.get('patches', Q * J), w = this.Wt.w, bias = this.b.w;
      this.x = x;
      for (let e = 0; e < B; e++) {
        this.im2row(P, x, e * CHW);
        const zo = e * F * Q;
        for (let f = 0; f < F; f++) z.fill(bias[f], zo + f * Q, zo + (f + 1) * Q);
        gemmNT(z, zo, w, 0, P, 0, F, Q, J);
      }
      if (!this.linear) activate(this.a, z, B * F * Q, this.fn);
      if (this.residual) { const out = this.out, a = this.a; for (let i = 0, n = B * F * Q; i < n; i++) out[i] = x[i] + a[i]; }
      return this.out;
    }
    backward(dy, B, needDx) {
      const { F, Q, J } = this, CHW = this.C * this.H * this.W, s = this.scr, gW = this.Wt.g, gb = this.b.g, dx = needDx ? this.dx : null;
      let dz = dy;
      if (!this.linear) { dz = this.dz; activateBack(dz, dy, this.z, this.a, B * F * Q, this.fn); }
      const Pc = s.get('cols', J * Q);
      let WT = null, dzT = null, dPc = null;
      if (dx) {
        WT = s.get('convWT', J * F); dzT = s.get('dzT', Q * F); dPc = s.get('dcols', J * Q);
        transpose(WT, this.Wt.w, 0, F, J);
      }
      for (let e = 0; e < B; e++) {
        const zo = e * F * Q;
        for (let f = 0; f < F; f++) { let t = 0; for (let i = zo + f * Q, end = i + Q; i < end; i++) t += dz[i]; gb[f] += t; }
        this.im2col(Pc, this.x, e * CHW);
        gemmNT(gW, 0, dz, zo, Pc, 0, F, J, Q);           // dW[f, j] += Σ_q dz[f, q] · patch[j, q]
        if (!dx) continue;
        transpose(dzT, dz, zo, F, Q);
        dPc.fill(0, 0, J * Q);
        gemmNT(dPc, 0, WT, 0, dzT, 0, J, Q, F);          // dpatch[j, q] = Σ_f W[f, j] · dz[f, q]
        dx.fill(0, e * CHW, (e + 1) * CHW);
        this.col2im(dx, e * CHW, dPc);
      }
      if (dx && this.residual) for (let i = 0, n = B * F * Q; i < n; i++) dx[i] += dy[i];
      return dx;
    }
  }

  // Max pool with stride = size; rows and columns beyond the last whole window are dropped.
  class Pool {
    constructor(spec, cfg, inS, outS, ctx) {
      const [C, H, W] = inS;
      Object.assign(this, { type: 'pool', spec, inShape: inS, outShape: outS, params: [], C, H, W, S: cfg.size, Ho: outS[1], Wo: outS[2], mem: new Mem(ctx.Arr) });
    }
    alloc(B) {
      const n = B * this.C * this.Ho * this.Wo;
      this.out = this.mem.view('out', n); this.arg = this.mem.view('arg', n, Int32Array); this.dx = this.mem.view('dx', B * this.C * this.H * this.W);
    }
    forward(x, B) {
      const { C, H, W, S, Ho, Wo, out, arg } = this;
      for (let e = 0, o = 0; e < B; e++) for (let c = 0; c < C; c++) {
        const base = (e * C + c) * H * W;
        for (let oy = 0; oy < Ho; oy++) for (let ox = 0; ox < Wo; ox++, o++) {
          let bi = base + oy * S * W + ox * S, bv = x[bi];
          for (let dy = 0; dy < S; dy++) {
            const row = base + (oy * S + dy) * W + ox * S;
            for (let dx = 0; dx < S; dx++) { const v = x[row + dx]; if (v > bv) { bv = v; bi = row + dx; } }
          }
          out[o] = bv; arg[o] = bi;
        }
      }
      return out;
    }
    // the gradient goes only to the pixel that won each window
    backward(dy, B, needDx) {
      if (!needDx) return null;
      const dx = this.dx, arg = this.arg, n = B * this.C * this.Ho * this.Wo;
      dx.fill(0);
      for (let o = 0; o < n; o++) dx[arg[o]] += dy[o];
      return dx;
    }
  }

  // [C, H, W] or [T, D] → one flat vector: the numbers are already in that order, so nothing moves.
  class Flatten {
    constructor(spec, cfg, inS, outS) { Object.assign(this, { type: 'flatten', spec, inShape: inS, outShape: outS, params: [], out: null }); }
    alloc() {}
    forward(x) { return (this.out = x); }
    backward(dy) { return dy; }
  }

  // Global average: each channel of an image over its pixels, or each feature of a sequence over its positions.
  class Gap {
    constructor(spec, cfg, inS, outS, ctx) {
      Object.assign(this, { type: 'gap', spec, inShape: inS, outShape: outS, params: [], image: inS.length === 3, nIn: size(inS), nOut: size(outS), mem: new Mem(ctx.Arr) });
      if (this.image) { this.C = inS[0]; this.P = inS[1] * inS[2]; } else { this.T = inS[0]; this.D = inS[1]; }
    }
    alloc(B) { this.out = this.mem.view('out', B * this.nOut); this.dx = this.mem.view('dx', B * this.nIn); }
    forward(x, B) {
      const out = this.out;
      if (this.image) {
        const P = this.P, inv = 1 / P;
        for (let r = 0, n = B * this.C; r < n; r++) { let s = 0; for (let i = r * P, e = i + P; i < e; i++) s += x[i]; out[r] = s * inv; }
      } else {
        const { T, D } = this, inv = 1 / T;
        out.fill(0);
        for (let b = 0; b < B; b++) {
          const o = b * D;
          for (let t = 0; t < T; t++) { const xo = (b * T + t) * D; for (let c = 0; c < D; c++) out[o + c] += x[xo + c]; }
          for (let c = 0; c < D; c++) out[o + c] *= inv;
        }
      }
      return out;
    }
    backward(dy, B, needDx) {
      if (!needDx) return null;
      const dx = this.dx;
      if (this.image) {
        const P = this.P, inv = 1 / P;
        for (let r = 0, n = B * this.C; r < n; r++) { const g = dy[r] * inv; dx.fill(g, r * P, (r + 1) * P); }
      } else {
        const { T, D } = this, inv = 1 / T;
        for (let b = 0; b < B; b++) for (let t = 0; t < T; t++) {
          const xo = (b * T + t) * D, o = b * D;
          for (let c = 0; c < D; c++) dx[xo + c] = dy[o + c] * inv;
        }
      }
      return dx;
    }
  }

  // ------------------------------------------------------------------ more layers
  const sig = (v) => 1 / (1 + Math.exp(-v));

  /* Recurrent layers over a sequence [T, D] → [T, H], one position at a time with a state carried forward: a plain
     RNN, h = tanh(W·x + U·h + b); an LSTM, whose input, forget and output gates decide what a separate memory c
     takes in, keeps and shows; or a GRU, with reset and update gates. The input projections of every position are
     one matrix product up front; the backward pass runs the positions in reverse (backpropagation through time).
     Weights follow PyTorch: W_ih [G·H, D], W_hh [G·H, H] and two biases, gates in its order (LSTM i, f, g, o;
     GRU r, z, n). */
  const GATES = { rnn: 1, lstm: 4, gru: 3 };
  class Recurrent {
    constructor(spec, cfg, inS, outS, ctx) {
      const [T, D] = inS, cell = own(GATES, cfg.cell) ? cfg.cell : 'lstm', H = cfg.units, G = GATES[cell], GH = G * H;
      Object.assign(this, { type: 'rnn', spec, inShape: inS, outShape: outS, T, D, H, G, GH, cell, scr: ctx.scr, mem: new Mem(ctx.Arr) });
      this.Wih = param(ctx, 'W_ih', [GH, D], true); this.Whh = param(ctx, 'W_hh', [GH, H], true);
      this.bih = param(ctx, 'b_ih', [GH], false); this.bhh = param(ctx, 'b_hh', [GH], false);
      this.params = [this.Wih, this.Whh, this.bih, this.bhh];
      gauss(ctx, this.Wih, initSd(linearInit(ctx), D)); gauss(ctx, this.Whh, initSd(linearInit(ctx), H));
      if (cell === 'lstm') this.bih.w.fill(1, H, 2 * H); // forget gates start open, so the memory lasts from the start
    }
    alloc(B) {
      const m = this.mem, { T, H, GH, D } = this, N = B * T;
      this.xw = m.view('xw', N * GH); this.gates = m.view('gates', N * GH); this.out = m.view('h', N * H);
      this.c = this.cell === 'lstm' ? m.view('c', N * H) : null;
      this.hn = this.cell === 'gru' ? m.view('hn', N * H) : null;
      this.hp = m.view('hp', B * H); this.hw = m.view('hw', B * GH);
      this.dxw = m.view('dxw', N * GH); this.dhw = m.view('dhw', B * GH); this.dh = m.view('dh', B * H); this.dc = m.view('dc', B * H); this.dhp = m.view('dhp', B * H);
      this.dx = m.view('dx', N * D);
    }
    // the states h of position t − 1 for the whole batch, one row each (zeros before the first position)
    prev(t, B) {
      const { T, H, hp, out } = this;
      for (let b = 0; b < B; b++) { const o = b * H; if (t) { const src = (b * T + t - 1) * H; for (let j = 0; j < H; j++) hp[o + j] = out[src + j]; } else hp.fill(0, o, o + H); }
    }
    forward(x, B) {
      const { T, D, H, GH, cell, xw, gates, out, c, hn, hp, hw } = this, Whh = this.Whh.w, bhh = this.bhh.w;
      this.x = x;
      linear(xw, x, this.Wih.w, this.bih.w, B * T, D, GH);
      for (let t = 0; t < T; t++) {
        this.prev(t, B);
        for (let b = 0; b < B; b++) hw.set(bhh, b * GH);
        gemmNT(hw, 0, hp, 0, Whh, 0, B, GH, H);
        for (let b = 0; b < B; b++) {
          const r = b * T + t, gi = r * GH, ho = r * H, wo = b * GH, po = b * H;
          if (cell === 'rnn') {
            for (let j = 0; j < H; j++) { const h = Math.tanh(xw[gi + j] + hw[wo + j]); gates[gi + j] = h; out[ho + j] = h; }
          } else if (cell === 'lstm') {
            const cp = t ? (r - 1) * H : -1;
            for (let j = 0; j < H; j++) {
              const i = sig(xw[gi + j] + hw[wo + j]), f = sig(xw[gi + H + j] + hw[wo + H + j]);
              const g = Math.tanh(xw[gi + 2 * H + j] + hw[wo + 2 * H + j]), o = sig(xw[gi + 3 * H + j] + hw[wo + 3 * H + j]);
              gates[gi + j] = i; gates[gi + H + j] = f; gates[gi + 2 * H + j] = g; gates[gi + 3 * H + j] = o;
              const cv = f * (cp >= 0 ? c[cp + j] : 0) + i * g;
              c[ho + j] = cv; out[ho + j] = o * Math.tanh(cv);
            }
          } else {
            for (let j = 0; j < H; j++) {
              const rr = sig(xw[gi + j] + hw[wo + j]), z = sig(xw[gi + H + j] + hw[wo + H + j]), hv = hw[wo + 2 * H + j];
              const n = Math.tanh(xw[gi + 2 * H + j] + rr * hv);
              gates[gi + j] = rr; gates[gi + H + j] = z; gates[gi + 2 * H + j] = n; hn[ho + j] = hv;
              out[ho + j] = (1 - z) * n + z * hp[po + j];
            }
          }
        }
      }
      return out;
    }
    backward(dy, B, needDx) {
      const { T, D, H, GH, cell, gates, c, hn, hp, dxw, dhw, dh, dc, dhp } = this, s = this.scr;
      const WT = s.get('rnnWT', H * GH), dhwT = s.get('rnnDhwT', GH * B), hpT = s.get('rnnHpT', H * B);
      transpose(WT, this.Whh.w, 0, GH, H);
      dh.fill(0); dc.fill(0); // what flows back from position t + 1 into h_t (and the LSTM's c_t)
      for (let t = T - 1; t >= 0; t--) {
        this.prev(t, B);
        for (let b = 0; b < B; b++) {
          const r = b * T + t, gi = r * GH, ho = r * H, wo = b * GH, po = b * H;
          if (cell === 'rnn') {
            for (let j = 0; j < H; j++) { const h = gates[gi + j], g = (dy[ho + j] + dh[po + j]) * (1 - h * h); dxw[gi + j] = g; dhw[wo + j] = g; }
          } else if (cell === 'lstm') {
            const cp = t ? (r - 1) * H : -1;
            for (let j = 0; j < H; j++) {
              const i = gates[gi + j], f = gates[gi + H + j], g = gates[gi + 2 * H + j], o = gates[gi + 3 * H + j], tc = Math.tanh(c[ho + j]);
              const dht = dy[ho + j] + dh[po + j], dct = dc[po + j] + dht * o * (1 - tc * tc), cprev = cp >= 0 ? c[cp + j] : 0;
              const di = dct * g * i * (1 - i), df = dct * cprev * f * (1 - f), dg = dct * i * (1 - g * g), dO = dht * tc * o * (1 - o);
              dxw[gi + j] = dhw[wo + j] = di; dxw[gi + H + j] = dhw[wo + H + j] = df;
              dxw[gi + 2 * H + j] = dhw[wo + 2 * H + j] = dg; dxw[gi + 3 * H + j] = dhw[wo + 3 * H + j] = dO;
              dc[po + j] = dct * f;
            }
          } else {
            for (let j = 0; j < H; j++) {
              const rr = gates[gi + j], z = gates[gi + H + j], n = gates[gi + 2 * H + j], hv = hn[ho + j];
              const dht = dy[ho + j] + dh[po + j], dan = dht * (1 - z) * (1 - n * n);
              const dar = dan * hv * rr * (1 - rr), daz = dht * (hp[po + j] - n) * z * (1 - z);
              dxw[gi + j] = dhw[wo + j] = dar; dxw[gi + H + j] = dhw[wo + H + j] = daz;
              dxw[gi + 2 * H + j] = dan; dhw[wo + 2 * H + j] = dan * rr;
              dhp[po + j] = dht * z; // the direct path from h_(t−1) to h_t
            }
          }
        }
        // this position's share of W_hh and b_hh, and the gradient reaching h_(t−1)
        transpose(dhwT, dhw, 0, B, GH); transpose(hpT, hp, 0, B, H);
        gemmNT(this.Whh.g, 0, dhwT, 0, hpT, 0, GH, H, B);
        const gb = this.bhh.g;
        for (let k = 0; k < GH; k++) { let sum = 0; for (let b = 0; b < B; b++) sum += dhw[b * GH + k]; gb[k] += sum; }
        if (cell !== 'gru') dhp.fill(0);
        gemmNT(dhp, 0, dhw, 0, WT, 0, B, H, GH);
        dh.set(dhp);
      }
      const dx = needDx ? this.dx : null;
      linearBack(s, dx, false, this.Wih.g, this.bih.g, dxw, this.x, this.Wih.w, B * T, D, GH);
      return dx;
    }
  }

  /* 1-D convolution along a sequence [T, D] → [T, F]: each output position mixes a window of k neighbouring positions
     (spaced d apart when dilated), zero-padded so the length stays T. A causal window only reaches back. */
  class Conv1d {
    constructor(spec, cfg, inS, outS, ctx) {
      const [T, D] = inS, F = cfg.filters, k = cfg.k, dil = cfg.dilation || 1, A = NA.engine.ACTIVATIONS, reach = (k - 1) * dil;
      Object.assign(this, { type: 'conv1d', spec, inShape: inS, outShape: outS, T, D, F, k, dil, left: cfg.causal ? reach : Math.floor(reach / 2), J: k * D, scr: ctx.scr, mem: new Mem(ctx.Arr) });
      this.Wt = param(ctx, 'kernels', [F, k, D], true); this.b = param(ctx, 'bias', [F], false);
      this.params = [this.Wt, this.b];
      this.fn = activation(own(A, cfg.act) ? cfg.act : 'relu', ctx.f64); this.linear = this.fn.id === 'linear';
      this.residual = !!cfg.residual && F === D;
      gauss(ctx, this.Wt, initSd(ctx.init, this.J, this.fn.he));
    }
    alloc(B) {
      const m = this.mem, N = B * this.T, n = N * this.F;
      this.P = m.view('patches', N * this.J); this.z = m.view('z', n);
      this.a = this.linear ? this.z : m.view('a', n);
      this.out = this.residual ? m.view('out', n) : this.a;
      this.dz = this.linear ? null : m.view('dz', n);
      this.dP = m.view('dP', N * this.J); this.dx = m.view('dx', N * this.D);
    }
    // calls fn(row, column offset, input offset) for every window position that lands inside the sequence
    windows(B, fn) {
      const { T, D, k, dil, left, J } = this;
      for (let b = 0; b < B; b++) for (let t = 0; t < T; t++) {
        const ro = (b * T + t) * J;
        for (let j = 0; j < k; j++) { const s = t - left + j * dil; if (s >= 0 && s < T) fn(ro + j * D, (b * T + s) * D); }
      }
    }
    forward(x, B) {
      const N = B * this.T, n = N * this.F, D = this.D, P = this.P;
      this.x = x;
      P.fill(0);
      this.windows(B, (po, xo) => { for (let c = 0; c < D; c++) P[po + c] = x[xo + c]; });
      linear(this.z, P, this.Wt.w, this.b.w, N, this.J, this.F);
      if (!this.linear) activate(this.a, this.z, n, this.fn);
      if (this.residual) { const out = this.out, a = this.a; for (let i = 0; i < n; i++) out[i] = x[i] + a[i]; }
      return this.out;
    }
    backward(dy, B, needDx) {
      const N = B * this.T, n = N * this.F, D = this.D, dx = needDx ? this.dx : null;
      let dz = dy;
      if (!this.linear) { dz = this.dz; activateBack(dz, dy, this.z, this.a, n, this.fn); }
      linearBack(this.scr, dx ? this.dP : null, false, this.Wt.g, this.b.g, dz, this.P, this.Wt.w, N, this.J, this.F);
      if (!dx) return null;
      const dP = this.dP;
      dx.fill(0);
      this.windows(B, (po, xo) => { for (let c = 0; c < D; c++) dx[xo + c] += dP[po + c]; });
      if (this.residual) for (let i = 0; i < n; i++) dx[i] += dy[i];
      return dx;
    }
  }

  // A sequence [T, D] pooled into one vector [D]: its last position, each feature's largest value, or the mean.
  class SeqPool {
    constructor(spec, cfg, inS, outS, ctx) {
      const [T, D] = inS;
      Object.assign(this, { type: 'seqpool', spec, inShape: inS, outShape: outS, params: [], T, D, mode: ['last', 'max', 'mean'].includes(cfg.mode) ? cfg.mode : 'last', mem: new Mem(ctx.Arr) });
    }
    alloc(B) { const m = this.mem; this.out = m.view('out', B * this.D); this.arg = m.view('arg', B * this.D, Int32Array); this.dx = m.view('dx', B * this.T * this.D); }
    forward(x, B) {
      const { T, D, out, arg, mode } = this;
      for (let b = 0; b < B; b++) for (let c = 0; c < D; c++) {
        const o = b * D + c, base = b * T * D + c;
        if (mode === 'mean') { let s = 0; for (let t = 0; t < T; t++) s += x[base + t * D]; out[o] = s / T; continue; }
        let bi = base + (T - 1) * D;
        if (mode === 'max') { bi = base; for (let t = 1; t < T; t++) if (x[base + t * D] > x[bi]) bi = base + t * D; }
        out[o] = x[bi]; arg[o] = bi;
      }
      return out;
    }
    backward(dy, B, needDx) {
      if (!needDx) return null;
      const { T, D, dx, arg, mode } = this;
      dx.fill(0);
      for (let b = 0; b < B; b++) for (let c = 0; c < D; c++) {
        const o = b * D + c;
        if (mode === 'mean') { const g = dy[o] / T, base = b * T * D + c; for (let t = 0; t < T; t++) dx[base + t * D] += g; } else dx[arg[o]] += dy[o];
      }
      return dx;
    }
  }

  /* Cuts an image [C, H, W] into p × p patches, left to right and top to bottom, and maps each to D numbers with a
     learned projection, plus a learned vector for its place: a sequence [N, D] for attention to work on, which is
     how a vision transformer starts. Rows and columns past the last whole patch are left out. */
  class Patches {
    constructor(spec, cfg, inS, outS, ctx) {
      const [C, H, W] = inS, p = cfg.size, D = cfg.dim, nw = Math.floor(W / p), N = outS[0], J = C * p * p;
      Object.assign(this, { type: 'patches', spec, inShape: inS, outShape: outS, C, H, W, p, D, nw, N, J, scr: ctx.scr, mem: new Mem(ctx.Arr) });
      this.Wp = param(ctx, 'projection', [D, J], true); this.bp = param(ctx, 'bias', [D], false); this.pos = param(ctx, 'position', [N, D], false);
      gauss(ctx, this.Wp, initSd(linearInit(ctx), J)); gauss(ctx, this.pos, 0.02);
      this.params = [this.Wp, this.bp, this.pos];
    }
    alloc(B) { const m = this.mem, R = B * this.N; this.P = m.view('patches', R * this.J); this.out = m.view('out', R * this.D); this.dP = m.view('dP', R * this.J); this.dx = m.view('dx', B * this.C * this.H * this.W); }
    // fn(patch-matrix index, image index) for every pixel of every patch
    pixels(B, fn) {
      const { C, H, W, p, nw, N, J } = this;
      for (let b = 0; b < B; b++) for (let n = 0; n < N; n++) {
        const py = Math.floor(n / nw) * p, px = (n % nw) * p, ro = (b * N + n) * J;
        let j = 0;
        for (let c = 0; c < C; c++) for (let yy = 0; yy < p; yy++) { const xo = ((b * C + c) * H + py + yy) * W + px; for (let xx = 0; xx < p; xx++) fn(ro + j++, xo + xx); }
      }
    }
    forward(x, B) {
      const { N, D, J, out, P } = this, pos = this.pos.w;
      this.pixels(B, (i, xi) => { P[i] = x[xi]; });
      linear(out, P, this.Wp.w, this.bp.w, B * N, J, D);
      for (let b = 0; b < B; b++) for (let i = 0, o = b * N * D; i < N * D; i++) out[o + i] += pos[i];
      return out;
    }
    backward(dy, B, needDx) {
      const { N, D, J } = this, gp = this.pos.g;
      for (let b = 0; b < B; b++) for (let i = 0, o = b * N * D; i < N * D; i++) gp[i] += dy[o + i];
      linearBack(this.scr, needDx ? this.dP : null, false, this.Wp.g, this.bp.g, dy, this.P, this.Wp.w, B * N, J, D);
      if (!needDx) return null;
      const dx = this.dx, dP = this.dP;
      dx.fill(0);
      this.pixels(B, (i, xi) => { dx[xi] += dP[i]; });
      return dx;
    }
  }

  /* Batch normalisation: while training, each feature (each channel of an image) is shifted and scaled to mean 0 and
     variance 1 over the batch (and over the pixels or positions), then given a learned scale and shift. Running
     averages of the batches' means and variances stand in when the model is used. */
  class BatchNorm {
    constructor(spec, cfg, inS, outS, ctx) {
      const kind = inS.length, ch = kind === 3 ? inS[0] : inS[inS.length - 1], per = size(inS);
      Object.assign(this, { type: 'batchnorm', spec, inShape: inS, outShape: outS, kind, ch, per, momentum: 0.1, mem: new Mem(ctx.Arr) });
      // the channel of each number of one example
      this.chan = new Int32Array(per);
      for (let i = 0; i < per; i++) this.chan[i] = kind === 3 ? Math.floor(i / (inS[1] * inS[2])) : i % ch;
      this.g = param(ctx, 'gain', [ch], false); this.g.w.fill(1); this.b = param(ctx, 'bias', [ch], false);
      this.params = [this.g, this.b];
      this.rmean = new Float64Array(ch); this.rvar = new Float64Array(ch).fill(1);
    }
    alloc(B) {
      const m = this.mem, n = B * this.per;
      this.out = m.view('out', n); this.xhat = m.view('xhat', n); this.dx = m.view('dx', n);
      this.mean = m.get('mean', this.ch, Float64Array); this.rstd = m.get('rstd', this.ch, Float64Array); this.s1 = m.get('s1', this.ch, Float64Array); this.s2 = m.get('s2', this.ch, Float64Array);
    }
    forward(x, B, train) {
      const { ch, per, chan, mean, rstd, xhat, out } = this, g = this.g.w, bb = this.b.w, cnt = B * per / ch;
      this.train = train;
      if (train) {
        const v = this.s1;
        mean.fill(0); v.fill(0);
        for (let e = 0, i = 0; e < B; e++) for (let k = 0; k < per; k++, i++) mean[chan[k]] += x[i];
        for (let c = 0; c < ch; c++) mean[c] /= cnt;
        for (let e = 0, i = 0; e < B; e++) for (let k = 0; k < per; k++, i++) { const d = x[i] - mean[chan[k]]; v[chan[k]] += d * d; }
        for (let c = 0; c < ch; c++) {
          const varc = v[c] / cnt;
          rstd[c] = 1 / Math.sqrt(varc + 1e-5);
          this.rmean[c] += this.momentum * (mean[c] - this.rmean[c]);
          this.rvar[c] += this.momentum * ((cnt > 1 ? varc * cnt / (cnt - 1) : varc) - this.rvar[c]);
        }
      } else for (let c = 0; c < ch; c++) { mean[c] = this.rmean[c]; rstd[c] = 1 / Math.sqrt(this.rvar[c] + 1e-5); }
      for (let e = 0, i = 0; e < B; e++) for (let k = 0; k < per; k++, i++) { const c = chan[k], h = (x[i] - mean[c]) * rstd[c]; xhat[i] = h; out[i] = h * g[c] + bb[c]; }
      return out;
    }
    backward(dy, B, needDx) {
      const { ch, per, chan, rstd, xhat, s1, s2 } = this, g = this.g.w, gg = this.g.g, gb = this.b.g, cnt = B * per / ch;
      s1.fill(0); s2.fill(0);
      for (let e = 0, i = 0; e < B; e++) for (let k = 0; k < per; k++, i++) { const c = chan[k]; s1[c] += dy[i]; s2[c] += dy[i] * xhat[i]; }
      for (let c = 0; c < ch; c++) { gb[c] += s1[c]; gg[c] += s2[c]; }
      if (!needDx) return null;
      const dx = this.dx;
      if (!this.train) { for (let e = 0, i = 0; e < B; e++) for (let k = 0; k < per; k++, i++) { const c = chan[k]; dx[i] = dy[i] * g[c] * rstd[c]; } return dx; }
      // through the batch statistics: dx = γ·rstd·(dy − mean(dy) − x̂·mean(dy·x̂)), per channel
      for (let e = 0, i = 0; e < B; e++) for (let k = 0; k < per; k++, i++) { const c = chan[k]; dx[i] = g[c] * rstd[c] * (dy[i] - s1[c] / cnt - xhat[i] * s2[c] / cnt); }
      return dx;
    }
  }

  // Average pool with stride = size: each window's mean; rows and columns beyond the last whole window are dropped.
  class AvgPool {
    constructor(spec, cfg, inS, outS, ctx) {
      const [C, H, W] = inS;
      Object.assign(this, { type: 'avgpool', spec, inShape: inS, outShape: outS, params: [], C, H, W, S: cfg.size, Ho: outS[1], Wo: outS[2], mem: new Mem(ctx.Arr) });
    }
    alloc(B) { this.out = this.mem.view('out', B * this.C * this.Ho * this.Wo); this.dx = this.mem.view('dx', B * this.C * this.H * this.W); }
    // fn(output index, input index) for every pixel of every window
    cells(B, fn) {
      const { C, H, W, S, Ho, Wo } = this;
      for (let e = 0, o = 0; e < B; e++) for (let c = 0; c < C; c++) {
        const base = (e * C + c) * H * W;
        for (let oy = 0; oy < Ho; oy++) for (let ox = 0; ox < Wo; ox++, o++) for (let dy = 0; dy < S; dy++) { const row = base + (oy * S + dy) * W + ox * S; for (let dx = 0; dx < S; dx++) fn(o, row + dx); }
      }
    }
    forward(x, B) { const out = this.out, inv = 1 / (this.S * this.S); out.fill(0); this.cells(B, (o, i) => { out[o] += x[i] * inv; }); return out; }
    backward(dy, B, needDx) {
      if (!needDx) return null;
      const dx = this.dx, inv = 1 / (this.S * this.S);
      dx.fill(0); this.cells(B, (o, i) => { dx[i] += dy[o] * inv; });
      return dx;
    }
  }

  // Gaussian noise added to every number while training (σ times a standard normal draw); nothing when the model is used.
  class Noise {
    constructor(spec, cfg, inS, outS, ctx) { Object.assign(this, { type: 'noise', spec, inShape: inS, outShape: outS, params: [], n: size(inS), sigma: Math.max(0, +cfg.sigma || 0), model: ctx.model, mem: new Mem(ctx.Arr) }); }
    alloc(B) { this.buf = this.mem.view('out', B * this.n); this.out = this.buf; }
    forward(x, B, train) {
      if (!train || !this.sigma) return (this.out = x);
      const out = this.buf, rng = this.model.rng, s = this.sigma;
      for (let i = 0, n = B * this.n; i < n; i++) out[i] = x[i] + s * rng.gauss();
      return (this.out = out);
    }
    backward(dy, B, needDx) { return needDx ? dy : null; }
  }

  /* Fourier features: x [F] becomes [x, sin(2π·Bx), cos(2π·Bx)] for m fixed random frequency vectors (the rows of
     B, drawn once with spread σ). With them the next layers can build sharp, wiggly functions of x that a plain
     network learns only slowly. Nothing here is learned. */
  class Fourier {
    constructor(spec, cfg, inS, outS, ctx) {
      const F = inS[0], m = cfg.features;
      Object.assign(this, { type: 'fourier', spec, inShape: inS, outShape: outS, params: [], F, m, nOut: F + 2 * m, mem: new Mem(ctx.Arr) });
      this.freq = new Float64Array(m * F);
      for (let i = 0; i < m * F; i++) this.freq[i] = ctx.rng.gauss(0, cfg.scale);
    }
    alloc(B) { const m = this.mem; this.out = m.view('out', B * this.nOut); this.ang = m.get('ang', B * this.m, Float64Array); this.dx = m.view('dx', B * this.F); }
    forward(x, B) {
      const { F, m, nOut, out, ang, freq } = this, TAU = 2 * Math.PI;
      this.ang = this.mem.get('ang', B * m, Float64Array);
      for (let b = 0; b < B; b++) {
        const xo = b * F, o = b * nOut;
        for (let i = 0; i < F; i++) out[o + i] = x[xo + i];
        for (let j = 0; j < m; j++) {
          let s = 0;
          for (let i = 0; i < F; i++) s += freq[j * F + i] * x[xo + i];
          const a = TAU * s;
          this.ang[b * m + j] = a; out[o + F + j] = Math.sin(a); out[o + F + m + j] = Math.cos(a);
        }
      }
      void ang;
      return out;
    }
    backward(dy, B, needDx) {
      if (!needDx) return null;
      const { F, m, nOut, dx, freq, ang } = this, TAU = 2 * Math.PI;
      for (let b = 0; b < B; b++) {
        const xo = b * F, o = b * nOut;
        for (let i = 0; i < F; i++) dx[xo + i] = dy[o + i];
        for (let j = 0; j < m; j++) {
          const a = ang[b * m + j], g = TAU * (Math.cos(a) * dy[o + F + j] - Math.sin(a) * dy[o + F + m + j]);
          for (let i = 0; i < F; i++) dx[xo + i] += g * freq[j * F + i];
        }
      }
      return dx;
    }
  }

  /* Gated linear unit: two projections of the input, the first multiplied by a gate made from the second,
     out = (x·W₁ + b₁) ⊙ g(x·W₂ + b₂), with g a sigmoid (GLU), Swish (SwiGLU, used in many large language models) or
     GELU (GEGLU). On a sequence, the same at every position. */
  class Gated {
    constructor(spec, cfg, inS, outS, ctx) {
      const nIn = inS[inS.length - 1], U = outS[outS.length - 1], A = NA.engine.ACTIVATIONS;
      Object.assign(this, { type: 'gated', spec, inShape: inS, outShape: outS, nIn, U, rows: inS.length === 2 ? inS[0] : 1, scr: ctx.scr, mem: new Mem(ctx.Arr) });
      this.fn = activation(own(A, cfg.gate) ? cfg.gate : 'swish', ctx.f64);
      this.W1 = param(ctx, 'value weights', [U, nIn], true); this.b1 = param(ctx, 'value bias', [U], false);
      this.W2 = param(ctx, 'gate weights', [U, nIn], true); this.b2 = param(ctx, 'gate bias', [U], false);
      this.params = [this.W1, this.b1, this.W2, this.b2];
      gauss(ctx, this.W1, initSd(linearInit(ctx), nIn)); gauss(ctx, this.W2, initSd(ctx.init, nIn, this.fn.he));
    }
    alloc(B) {
      const m = this.mem, N = B * this.rows, n = N * this.U;
      for (const k of ['z1', 'z2', 'a2', 'out', 'dz1', 'da2', 'dz2']) this[k] = m.view(k, n);
      this.dx = m.view('dx', N * this.nIn);
    }
    forward(x, B) {
      const N = B * this.rows, n = N * this.U, { z1, z2, a2, out } = this;
      this.x = x;
      linear(z1, x, this.W1.w, this.b1.w, N, this.nIn, this.U);
      linear(z2, x, this.W2.w, this.b2.w, N, this.nIn, this.U);
      activate(a2, z2, n, this.fn);
      for (let i = 0; i < n; i++) out[i] = z1[i] * a2[i];
      return out;
    }
    backward(dy, B, needDx) {
      const N = B * this.rows, n = N * this.U, { z1, z2, a2, dz1, da2, dz2 } = this, dx = needDx ? this.dx : null;
      for (let i = 0; i < n; i++) { dz1[i] = dy[i] * a2[i]; da2[i] = dy[i] * z1[i]; }
      activateBack(dz2, da2, z2, a2, n, this.fn);
      linearBack(this.scr, dx, false, this.W1.g, this.b1.g, dz1, this.x, this.W1.w, N, this.nIn, this.U);
      linearBack(this.scr, dx, true, this.W2.g, this.b2.g, dz2, this.x, this.W2.w, N, this.nIn, this.U);
      return dx;
    }
  }

  /* Adds a vector for each position of a sequence, so that what comes next can tell positions apart: learned, or the
     fixed sine and cosine waves of the original transformer (PE[t, 2i] = sin(t / 10000^(2i/D)), and cos for 2i + 1). */
  class Position {
    constructor(spec, cfg, inS, outS, ctx) {
      const [T, D] = inS;
      Object.assign(this, { type: 'position', spec, inShape: inS, outShape: outS, T, D, learned: cfg.mode !== 'sine', mem: new Mem(ctx.Arr) });
      if (this.learned) { this.pos = param(ctx, 'position', [T, D], false); gauss(ctx, this.pos, 0.02); this.params = [this.pos]; }
      else {
        this.params = [];
        this.table = new ctx.Arr(T * D);
        for (let t = 0; t < T; t++) for (let i = 0; i < D; i++) { const a = t / Math.pow(10000, (2 * Math.floor(i / 2)) / D); this.table[t * D + i] = i % 2 ? Math.cos(a) : Math.sin(a); }
      }
    }
    alloc(B) { this.out = this.mem.view('out', B * this.T * this.D); }
    forward(x, B) {
      const P = this.learned ? this.pos.w : this.table, n = this.T * this.D, out = this.out;
      for (let b = 0; b < B; b++) for (let i = 0, o = b * n; i < n; i++) out[o + i] = x[o + i] + P[i];
      return out;
    }
    backward(dy, B, needDx) {
      if (this.learned) { const g = this.pos.g, n = this.T * this.D; for (let b = 0; b < B; b++) for (let i = 0, o = b * n; i < n; i++) g[i] += dy[o + i]; }
      return needDx ? dy : null;
    }
  }

  /* Parallel branches: each branch is its own stack of layers on the same input, and their outputs are joined side by
     side (a vector's numbers, each position's features, an image's channels), added, multiplied or averaged, number
     by number. An empty branch passes the input through, so adding it to the others is a residual connection.
     Backward: each branch gets its share of the gradient (its slice, all of it, all of it times the other branches'
     outputs, or a k-th of it), runs it back through its own layers, and the input's gradient is the sum. */
  class Parallel {
    constructor(spec, cfg, inS, outS, ctx) {
      Object.assign(this, { type: 'parallel', spec, inShape: inS, outShape: outS, merge: cfg.merge, inSize: size(inS), outSize: size(outS), mem: new Mem(ctx.Arr) });
      this.lanes = spec.branches.map((list) => { const sh = stackShape(list, inS).shapes; return list.map((s, k) => new LAYERS[s.type](s, withDefaults(s), sh[k].in, sh[k].out, ctx)); });
      this.k = this.lanes.length;
      this.laneOut = spec.branches.map((list) => stackShape(list, inS).out);
      this.sizes = this.laneOut.map(size);
      this.rows = outS.length === 2 ? outS[0] : 1; // joined side by side: one row per position of a sequence
      this.inner = this.lanes.flat();
      this.params = [].concat(...this.inner.map((L) => L.params));
    }
    alloc(B) {
      for (const L of this.inner) L.alloc(B);
      this.out = this.mem.view('out', B * this.outSize);
      this.dx = this.mem.view('dx', B * this.inSize);
      this.dh = this.sizes.map((n, b) => this.mem.view('dh' + b, B * n));
    }
    forward(x, B, train) {
      this.x = x;
      const ys = (this.ys = this.lanes.map((lane) => { let h = x; for (const L of lane) h = L.forward(h, B, train); return h; }));
      const out = this.out, n = B * this.outSize;
      if (this.merge === 'concat') {
        const R = this.rows, row = this.outSize / R;
        let off = 0;
        ys.forEach((y, b) => {
          const c = this.sizes[b] / R;
          for (let e = 0, rows = B * R; e < rows; e++) { const src = e * c, dst = e * row + off; for (let j = 0; j < c; j++) out[dst + j] = y[src + j]; }
          off += c;
        });
      } else {
        const y0 = ys[0];
        for (let i = 0; i < n; i++) out[i] = y0[i];
        for (let b = 1; b < this.k; b++) { const y = ys[b]; if (this.merge === 'mul') for (let i = 0; i < n; i++) out[i] *= y[i]; else for (let i = 0; i < n; i++) out[i] += y[i]; }
        if (this.merge === 'mean') { const s = 1 / this.k; for (let i = 0; i < n; i++) out[i] *= s; }
      }
      return out;
    }
    backward(dy, B, needDx) {
      const n = B * this.outSize, ys = this.ys, dh = this.dh, k = this.k;
      // each branch's share of the gradient, all of them before any branch runs back
      if (this.merge === 'concat') {
        const R = this.rows, row = this.outSize / R;
        let off = 0;
        dh.forEach((g, b) => {
          const c = this.sizes[b] / R;
          for (let e = 0, rows = B * R; e < rows; e++) { const dst = e * c, src = e * row + off; for (let j = 0; j < c; j++) g[dst + j] = dy[src + j]; }
          off += c;
        });
      } else if (this.merge === 'mul') {
        for (let b = 0; b < k; b++) { const g = dh[b]; for (let i = 0; i < n; i++) { let p = dy[i]; for (let c = 0; c < k; c++) if (c !== b) p *= ys[c][i]; g[i] = p; } }
      } else {
        const s = this.merge === 'mean' ? 1 / k : 1;
        for (let b = 0; b < k; b++) { const g = dh[b]; for (let i = 0; i < n; i++) g[i] = dy[i] * s; }
      }
      // back through each branch, last layer first; a layer passes a gradient back when anything before it learns
      const dx = needDx ? this.dx : null, m = B * this.inSize;
      if (dx) dx.fill(0);
      this.lanes.forEach((lane, b) => {
        const flags = [];
        let learns = needDx;
        for (const L of lane) { flags.push(learns); if (L.params.length) learns = true; }
        let g = dh[b];
        for (let j = lane.length - 1; j >= 0 && g; j--) g = lane[j].backward(g, B, flags[j]);
        if (dx && g) for (let i = 0; i < m; i++) dx[i] += g[i];
      });
      return dx;
    }
  }

  // ------------------------------------------------------------------ the newest layers
  /* Depthwise-separable convolution: a k × k filter for each input channel on its own (depthwise; its taps spaced d
     apart when dilated), then a 1 × 1 convolution that mixes the channels into F outputs (pointwise), then the
     activation. C·k² + C·F weights instead of a full convolution's C·k²·F: the idea behind Xception and MobileNet.
     'Same' zero padding of d·(k − 1)/2 on every side (k is odd); the output is ceil(H/stride) × ceil(W/stride). */
  class SepConv {
    constructor(spec, cfg, inS, outS, ctx) {
      const [C, H, W] = inS, F = outS[0], k = cfg.k, dil = cfg.dilation || 1, A = NA.engine.ACTIVATIONS;
      Object.assign(this, { type: 'sepconv', spec, inShape: inS, outShape: outS, C, H, W, F, k, dil, s: cfg.stride || 1, pad: (dil * (k - 1)) >> 1, Ho: outS[1], Wo: outS[2], scr: ctx.scr, mem: new Mem(ctx.Arr) });
      this.Q = this.Ho * this.Wo;
      this.dw = param(ctx, 'depthwise kernels', [C, k, k], true); this.db = param(ctx, 'depthwise bias', [C], false);
      this.pw = param(ctx, 'pointwise weights', [F, C], true); this.pb = param(ctx, 'bias', [F], false);
      this.params = [this.dw, this.db, this.pw, this.pb];
      this.fn = activation(own(A, cfg.act) ? cfg.act : 'relu', ctx.f64); this.linear = this.fn.id === 'linear';
      gauss(ctx, this.dw, initSd(linearInit(ctx), k * k)); gauss(ctx, this.pw, initSd(ctx.init, C, this.fn.he));
    }
    alloc(B) {
      const m = this.mem, n = B * this.F * this.Q;
      this.mid = m.view('mid', B * this.C * this.Q); this.z = m.view('z', n);
      this.a = this.linear ? this.z : m.view('a', n); this.out = this.a;
      this.dz = this.linear ? null : m.view('dz', n);
      this.dx = m.view('dx', B * this.C * this.H * this.W);
    }
    // the output columns whose tap kx lands inside the picture
    span(kx) { const { W, s, Wo } = this, o = kx * this.dil - this.pad; return [Math.max(0, Math.ceil(-o / s)), Math.min(Wo, Math.floor((W - 1 - o) / s) + 1)]; }
    forward(x, B) {
      const { C, H, W, F, k, dil, s, pad, Wo, Q, mid, z } = this, w = this.dw.w, db = this.db.w, pb = this.pb.w, HW = H * W, T = this.scr.get('sepT', Q * C);
      this.x = x;
      for (let e = 0; e < B; e++) {
        const mo = e * C * Q, xo = e * C * HW;
        // depthwise: each channel through its own k × k filter
        for (let c = 0; c < C; c++) {
          const mc = mo + c * Q, xc = xo + c * HW;
          mid.fill(db[c], mc, mc + Q);
          for (let ky = 0; ky < k; ky++) for (let kx = 0; kx < k; kx++) {
            const wv = w[(c * k + ky) * k + kx], [ox0, ox1] = this.span(kx), off = kx * dil - pad;
            for (let oy = 0; oy < this.Ho; oy++) {
              const iy = oy * s + ky * dil - pad;
              if (iy < 0 || iy >= H) continue;
              const row = xc + iy * W + off, mr = mc + oy * Wo;
              for (let ox = ox0; ox < ox1; ox++) mid[mr + ox] += wv * x[row + ox * s];
            }
          }
        }
        // pointwise: z[f, q] = b[f] + Σ_c pw[f, c] · mid[c, q], one matrix product
        transpose(T, mid, mo, C, Q);
        const zo = e * F * Q;
        for (let f = 0; f < F; f++) z.fill(pb[f], zo + f * Q, zo + (f + 1) * Q);
        gemmNT(z, zo, this.pw.w, 0, T, 0, F, Q, C);
      }
      if (!this.linear) activate(this.a, z, B * F * Q, this.fn);
      return this.out;
    }
    backward(dy, B, needDx) {
      const { C, H, W, F, k, dil, s, pad, Wo, Q, mid, x } = this, HW = H * W, sc = this.scr, w = this.dw.w, gw = this.dw.g, gdb = this.db.g, gpb = this.pb.g;
      let dz = dy;
      if (!this.linear) { dz = this.dz; activateBack(dz, dy, this.z, this.a, B * F * Q, this.fn); }
      const pwT = sc.get('sepPwT', C * F), dzT = sc.get('sepDzT', Q * F), dm = sc.get('sepDm', C * Q), dx = needDx ? this.dx : null;
      transpose(pwT, this.pw.w, 0, F, C);
      if (dx) dx.fill(0, 0, B * C * HW);
      for (let e = 0; e < B; e++) {
        const zo = e * F * Q, mo = e * C * Q, xo = e * C * HW;
        for (let f = 0; f < F; f++) { let t = 0; for (let i = zo + f * Q, end = i + Q; i < end; i++) t += dz[i]; gpb[f] += t; }
        gemmNT(this.pw.g, 0, dz, zo, mid, mo, F, C, Q);          // dpw[f, c] += Σ_q dz[f, q] · mid[c, q]
        transpose(dzT, dz, zo, F, Q);
        dm.fill(0, 0, C * Q);
        gemmNT(dm, 0, pwT, 0, dzT, 0, C, Q, F);                   // dmid[c, q] = Σ_f pw[f, c] · dz[f, q]
        for (let c = 0; c < C; c++) {
          const mc = c * Q, xc = xo + c * HW;
          let t = 0;
          for (let q = 0; q < Q; q++) t += dm[mc + q];
          gdb[c] += t;
          for (let ky = 0; ky < k; ky++) for (let kx = 0; kx < k; kx++) {
            const ki = (c * k + ky) * k + kx, wv = w[ki], [ox0, ox1] = this.span(kx), off = kx * dil - pad;
            let g = 0;
            for (let oy = 0; oy < this.Ho; oy++) {
              const iy = oy * s + ky * dil - pad;
              if (iy < 0 || iy >= H) continue;
              const row = xc + iy * W + off, mr = mc + oy * Wo;
              for (let ox = ox0; ox < ox1; ox++) { const gm = dm[mr + ox], xi = row + ox * s; g += gm * x[xi]; if (dx) dx[xi] += gm * wv; }
            }
            gw[ki] += g;
          }
        }
      }
      return dx;
    }
  }

  /* Squeeze-and-excitation: each channel of an image (each feature of a sequence or a vector) is averaged over its
     positions (the squeeze), a small network turns those C averages into C gates between 0 and 1 (the excitation:
     a dense layer of C/r ReLU units, then one of C sigmoids), and every channel is multiplied by its gate. Attention
     over channels: the network learns which of its features matter for this example. */
  class SE {
    constructor(spec, cfg, inS, outS, ctx) {
      const kind = inS.length, C = kind === 3 ? inS[0] : inS[kind - 1], P = kind === 3 ? inS[1] * inS[2] : kind === 2 ? inS[0] : 1, h = seHidden(C, cfg.ratio);
      // a channel's numbers sit one after another in an image (channels first), C apart in a sequence
      Object.assign(this, { type: 'se', spec, inShape: inS, outShape: outS, C, P, h, cs: kind === 3 ? P : 1, ps: kind === 3 ? 1 : C, scr: ctx.scr, mem: new Mem(ctx.Arr) });
      this.W1 = param(ctx, 'squeeze weights', [h, C], true); this.b1 = param(ctx, 'squeeze bias', [h], false);
      this.W2 = param(ctx, 'excite weights', [C, h], true); this.b2 = param(ctx, 'excite bias', [C], false);
      this.params = [this.W1, this.b1, this.W2, this.b2];
      gauss(ctx, this.W1, initSd(ctx.init, C, true)); gauss(ctx, this.W2, initSd(linearInit(ctx), h));
    }
    alloc(B) {
      const m = this.mem, C = this.C, h = this.h, n = B * C * this.P;
      for (const key of ['s', 'z2', 'g', 'dg', 'ds']) this[key] = m.view(key, B * C);
      for (const key of ['z1', 'a1', 'da1']) this[key] = m.view(key, B * h);
      this.out = m.view('out', n); this.dx = m.view('dx', n);
    }
    forward(x, B) {
      const { C, P, h, cs, ps, s, z1, a1, z2, g, out } = this;
      this.x = x;
      for (let b = 0; b < B; b++) for (let c = 0; c < C; c++) { const o = b * C * P + c * cs; let t = 0; for (let p = 0; p < P; p++) t += x[o + p * ps]; s[b * C + c] = t / P; }
      linear(z1, s, this.W1.w, this.b1.w, B, C, h);
      for (let i = 0, n = B * h; i < n; i++) a1[i] = z1[i] > 0 ? z1[i] : 0;
      linear(z2, a1, this.W2.w, this.b2.w, B, h, C);
      for (let i = 0, n = B * C; i < n; i++) g[i] = sig(z2[i]);
      for (let b = 0; b < B; b++) for (let c = 0; c < C; c++) { const o = b * C * P + c * cs, gv = g[b * C + c]; for (let p = 0; p < P; p++) out[o + p * ps] = x[o + p * ps] * gv; }
      return out;
    }
    backward(dy, B, needDx) {
      const { C, P, h, cs, ps, s, z1, a1, g, dg, da1, ds, x } = this;
      // through the gates (each one scales all of its channel's numbers), the sigmoids, the ReLUs and the squeeze
      for (let b = 0; b < B; b++) for (let c = 0; c < C; c++) { const o = b * C * P + c * cs, gv = g[b * C + c]; let t = 0; for (let p = 0; p < P; p++) t += dy[o + p * ps] * x[o + p * ps]; dg[b * C + c] = t * gv * (1 - gv); }
      linearBack(this.scr, da1, false, this.W2.g, this.b2.g, dg, a1, this.W2.w, B, h, C);
      for (let i = 0, n = B * h; i < n; i++) if (z1[i] <= 0) da1[i] = 0;
      linearBack(this.scr, needDx ? ds : null, false, this.W1.g, this.b1.g, da1, s, this.W1.w, B, C, h);
      if (!needDx) return null;
      const dx = this.dx, inv = 1 / P;
      for (let b = 0; b < B; b++) for (let c = 0; c < C; c++) { const o = b * C * P + c * cs, gv = g[b * C + c], d = ds[b * C + c] * inv; for (let p = 0; p < P; p++) dx[o + p * ps] = dy[o + p * ps] * gv + d; }
      return dx;
    }
  }

  /* Coordinate channels (CoordConv): two more channels holding each pixel's column and row, from −1 to 1 (with
     radius, a third: its distance from the centre, 0 to 1). A convolution slides the same filter everywhere, so on
     its own it cannot tell where it is; with these it can. Nothing is learned. */
  class Coords {
    constructor(spec, cfg, inS, outS, ctx) {
      const [C, H, W] = inS, K = outS[0] - C, lin = (n, j) => (n > 1 ? -1 + (2 * j) / (n - 1) : -1); // as torch.linspace(-1, 1, n)
      Object.assign(this, { type: 'coords', spec, inShape: inS, outShape: outS, params: [], C, H, W, K, mem: new Mem(ctx.Arr) });
      this.grid = new ctx.Arr(K * H * W);
      for (let y = 0; y < H; y++) for (let x = 0; x < W; x++) {
        const u = lin(W, x), v = lin(H, y), p = y * W + x;
        this.grid[p] = u; this.grid[H * W + p] = v;
        if (K > 2) this.grid[2 * H * W + p] = Math.sqrt(u * u + v * v) / Math.SQRT2;
      }
    }
    alloc(B) { const HW = this.H * this.W; this.out = this.mem.view('out', B * (this.C + this.K) * HW); this.dx = this.mem.view('dx', B * this.C * HW); }
    forward(x, B) {
      const n = this.C * this.H * this.W, m = n + this.grid.length, out = this.out;
      for (let b = 0; b < B; b++) { for (let i = 0; i < n; i++) out[b * m + i] = x[b * n + i]; out.set(this.grid, b * m + n); }
      return out;
    }
    backward(dy, B, needDx) {
      if (!needDx) return null;
      const n = this.C * this.H * this.W, m = n + this.grid.length, dx = this.dx;
      for (let b = 0; b < B; b++) for (let i = 0; i < n; i++) dx[b * n + i] = dy[b * m + i];
      return dx;
    }
  }

  /* Random shifts while training: each picture is moved by up to s pixels left or right and up or down (the pixels
     that come in are 0), and with flip it is mirrored left to right half of the time: out[c, y, x] = in[c, y + oy,
     x′ + ox] with x′ = W − 1 − x when mirrored. A new move for every example at every step, so the model never sees
     quite the same picture twice (data augmentation). When the model is used, nothing changes. */
  class Augment {
    constructor(spec, cfg, inS, outS, ctx) {
      const [C, H, W] = inS;
      Object.assign(this, { type: 'augment', spec, inShape: inS, outShape: outS, params: [], C, H, W, shift: Math.max(0, cfg.shift | 0), flip: !!cfg.flip, model: ctx.model, mem: new Mem(ctx.Arr) });
    }
    alloc(B) { const n = B * this.C * this.H * this.W; this.buf = this.mem.view('out', n); this.dx = this.mem.view('dx', n); this.moves = this.mem.view('moves', 3 * B, Int32Array); this.out = this.buf; }
    // example b, moved: copies src into dst (or, going back, adds dst's gradient onto src's pixels)
    move(b, src, dst, back) {
      const { C, H, W, moves } = this, HW = H * W, oy = moves[3 * b], ox = moves[3 * b + 1], fl = moves[3 * b + 2];
      for (let c = 0; c < C; c++) for (let y = 0; y < H; y++) {
        const iy = y + oy;
        if (iy < 0 || iy >= H) continue;
        const ro = (b * C + c) * HW + y * W, ri = (b * C + c) * HW + iy * W;
        for (let x = 0; x < W; x++) {
          const ix = (fl ? W - 1 - x : x) + ox;
          if (ix < 0 || ix >= W) continue;
          if (back) dst[ri + ix] += src[ro + x]; else dst[ro + x] = src[ri + ix];
        }
      }
    }
    forward(x, B, train) {
      this.moved = !!train && (this.shift > 0 || this.flip);
      if (!this.moved) return (this.out = x);
      const { shift, moves, buf } = this, rng = this.model.rng;
      buf.fill(0, 0, B * this.C * this.H * this.W);
      for (let b = 0; b < B; b++) {
        moves[3 * b] = rng.int(2 * shift + 1) - shift; moves[3 * b + 1] = rng.int(2 * shift + 1) - shift; moves[3 * b + 2] = this.flip && rng.next() < 0.5 ? 1 : 0;
        this.move(b, x, buf, false);
      }
      return (this.out = buf);
    }
    backward(dy, B, needDx) {
      if (!needDx) return null;
      if (!this.moved) return dy;
      const dx = this.dx;
      dx.fill(0, 0, B * this.C * this.H * this.W);
      for (let b = 0; b < B; b++) this.move(b, dy, dx, true);
      return dx;
    }
  }

  // Global max pool: each channel of an image (each feature of a sequence) keeps its largest value over the positions.
  class Gmp {
    constructor(spec, cfg, inS, outS, ctx) {
      const image = inS.length === 3, C = image ? inS[0] : inS[1], P = image ? inS[1] * inS[2] : inS[0];
      Object.assign(this, { type: 'gmp', spec, inShape: inS, outShape: outS, params: [], C, P, cs: image ? P : 1, ps: image ? 1 : C, mem: new Mem(ctx.Arr) });
    }
    alloc(B) { const m = this.mem; this.out = m.view('out', B * this.C); this.arg = m.view('arg', B * this.C, Int32Array); this.dx = m.view('dx', B * this.C * this.P); }
    forward(x, B) {
      const { C, P, cs, ps, out, arg } = this;
      for (let b = 0; b < B; b++) for (let c = 0; c < C; c++) {
        const o = b * C * P + c * cs;
        let bi = o;
        for (let p = 1; p < P; p++) if (x[o + p * ps] > x[bi]) bi = o + p * ps;
        out[b * C + c] = x[bi]; arg[b * C + c] = bi;
      }
      return out;
    }
    // the gradient goes only to the number that won
    backward(dy, B, needDx) {
      if (!needDx) return null;
      const dx = this.dx, arg = this.arg;
      dx.fill(0);
      for (let i = 0, n = B * this.C; i < n; i++) dx[arg[i]] += dy[i];
      return dx;
    }
  }

  /* Attention pooling: one learned query vector w scores every position of a sequence (every pixel of an image),
     s_t = w · x_t; a softmax turns the scores into weights that add up to 1, and the output is the weighted average
     of the positions' vectors. A learned, example-by-example choice of where to look, in place of an average or the
     last step: the simplest attention there is. */
  class AttnPool {
    constructor(spec, cfg, inS, outS, ctx) {
      const image = inS.length === 3, D = image ? inS[0] : inS[1], T = image ? inS[1] * inS[2] : inS[0];
      // a feature's numbers sit T apart in an image's layout (channels first), next to each other in a sequence
      Object.assign(this, { type: 'attnpool', spec, inShape: inS, outShape: outS, D, T, fs: image ? T : 1, ts: image ? 1 : D, mem: new Mem(ctx.Arr) });
      this.w = param(ctx, 'query', [D], false); gauss(ctx, this.w, initSd(linearInit(ctx), D));
      this.params = [this.w];
    }
    alloc(B) { const m = this.mem; this.out = m.view('out', B * this.D); this.att = m.view('att', B * this.T); this.dx = m.view('dx', B * this.T * this.D); this.row = m.get('row', this.T, Float64Array); }
    forward(x, B) {
      const { D, T, fs, ts, out, att } = this, w = this.w.w;
      this.x = x;
      for (let b = 0; b < B; b++) {
        const o = b * T * D, ao = b * T;
        let max = -Infinity, sum = 0;
        for (let t = 0; t < T; t++) { let s = 0; for (let c = 0; c < D; c++) s += w[c] * x[o + t * ts + c * fs]; att[ao + t] = s; if (s > max) max = s; }
        for (let t = 0; t < T; t++) { const e = Math.exp(att[ao + t] - max); att[ao + t] = e; sum += e; }
        for (let t = 0; t < T; t++) att[ao + t] /= sum;
        for (let c = 0; c < D; c++) { let s = 0; for (let t = 0; t < T; t++) s += att[ao + t] * x[o + t * ts + c * fs]; out[b * D + c] = s; }
      }
      return out;
    }
    backward(dy, B, needDx) {
      const { D, T, fs, ts, att, x, row } = this, w = this.w.w, gw = this.w.g, dx = needDx ? this.dx : null;
      for (let b = 0; b < B; b++) {
        const o = b * T * D, ao = b * T, go = b * D;
        // da_t = dy · x_t, then through the softmax: ds_t = a_t (da_t − Σ a·da)
        let dot = 0;
        for (let t = 0; t < T; t++) { let s = 0; for (let c = 0; c < D; c++) s += dy[go + c] * x[o + t * ts + c * fs]; row[t] = s; dot += att[ao + t] * s; }
        for (let t = 0; t < T; t++) {
          const a = att[ao + t], d = a * (row[t] - dot);
          for (let c = 0; c < D; c++) { const i = o + t * ts + c * fs; gw[c] += d * x[i]; if (dx) dx[i] = a * dy[go + c] + d * w[c]; }
        }
      }
      return dx;
    }
  }

  /* A state-space layer: a linear recurrence, the core of S4, LRU and Mamba without their extras. N memory cells, each
     a moving average of a learned mix of the input with its own learned memory a between 0 and 1:
     h_t = a ⊙ h_(t−1) + (1 − a) ⊙ B·x_t, then y_t = f(C·h_t + c). A cell with a near 1 remembers for many steps, one
     near 0 follows the input. With no squashing inside the loop, the gradient through time neither fades nor
     explodes, and the whole recurrence could be run for all steps at once (a parallel scan); here it runs step by step.
     The memories start spread from about 2 to 32 steps (a = e^(−1/τ)), learned as logits: a = σ(θ). */
  class SSM {
    constructor(spec, cfg, inS, outS, ctx) {
      const [T, D] = inS, N = cfg.state, U = cfg.units, A = NA.engine.ACTIVATIONS;
      Object.assign(this, { type: 'ssm', spec, inShape: inS, outShape: outS, T, D, N, U, scr: ctx.scr, mem: new Mem(ctx.Arr) });
      this.theta = param(ctx, 'memory logits', [N], false);
      this.Bm = param(ctx, 'input weights', [N, D], true);
      this.Cm = param(ctx, 'output weights', [U, N], true); this.cb = param(ctx, 'bias', [U], false);
      this.params = [this.theta, this.Bm, this.Cm, this.cb];
      this.fn = activation(own(A, cfg.act) ? cfg.act : 'gelu', ctx.f64); this.linear = this.fn.id === 'linear';
      for (let j = 0; j < N; j++) { const a = Math.exp(-1 / Math.pow(2, 1 + (4 * j) / Math.max(1, N - 1))); this.theta.w[j] = Math.log(a / (1 - a)); }
      gauss(ctx, this.Bm, initSd(linearInit(ctx), D)); gauss(ctx, this.Cm, initSd(ctx.init, N, this.fn.he));
      this.noBias = new ctx.Arr(N);
    }
    alloc(B) {
      const m = this.mem, R = B * this.T;
      this.u = m.view('u', R * this.N); this.h = m.view('h', R * this.N); this.z = m.view('z', R * this.U);
      this.a = this.linear ? this.z : m.view('a', R * this.U); this.out = this.a;
      this.dz = this.linear ? null : m.view('dz', R * this.U);
      this.dh = m.view('dh', R * this.N); this.du = m.view('du', R * this.N); this.dx = m.view('dx', R * this.D);
      this.dec = m.get('dec', this.N, Float64Array); this.gA = m.get('gA', this.N, Float64Array); this.carry = m.get('carry', this.N, Float64Array);
    }
    forward(x, B) {
      const { T, D, N, U, u, h, dec } = this, R = B * T, th = this.theta.w;
      this.x = x;
      for (let j = 0; j < N; j++) dec[j] = sig(th[j]);
      linear(u, x, this.Bm.w, this.noBias, R, D, N);
      for (let b = 0; b < B; b++) for (let t = 0; t < T; t++) {
        const o = (b * T + t) * N, p = o - N;
        for (let j = 0; j < N; j++) h[o + j] = (t ? dec[j] * h[p + j] : 0) + (1 - dec[j]) * u[o + j];
      }
      linear(this.z, h, this.Cm.w, this.cb.w, R, N, U);
      if (!this.linear) activate(this.a, this.z, R * U, this.fn);
      return this.out;
    }
    backward(dy, B, needDx) {
      const { T, D, N, U, u, h, dh, du, dec, gA, carry } = this, R = B * T, sc = this.scr;
      let dz = dy;
      if (!this.linear) { dz = this.dz; activateBack(dz, dy, this.z, this.a, R * U, this.fn); }
      linearBack(sc, dh, false, this.Cm.g, this.cb.g, dz, h, this.Cm.w, R, N, U);
      // back through time: what reaches h_t from its output and from h_(t+1), shared between a, h_(t−1) and B·x_t
      gA.fill(0);
      for (let b = 0; b < B; b++) {
        carry.fill(0);
        for (let t = T - 1; t >= 0; t--) {
          const o = (b * T + t) * N, p = o - N;
          for (let j = 0; j < N; j++) {
            const g = dh[o + j] + carry[j];
            gA[j] += g * ((t ? h[p + j] : 0) - u[o + j]);
            du[o + j] = (1 - dec[j]) * g;
            carry[j] = dec[j] * g;
          }
        }
      }
      const gth = this.theta.g, noGrad = sc.get('ssmNoBias', N);
      for (let j = 0; j < N; j++) gth[j] += gA[j] * dec[j] * (1 - dec[j]);
      linearBack(sc, needDx ? this.dx : null, false, this.Bm.g, noGrad, du, this.x, this.Bm.w, R, D, N);
      return needDx ? this.dx : null;
    }
  }

  /* Radial basis units: each unit has a centre c and a width σ, both learned, and answers how close the input is to
     its centre, φ = exp(−‖x − c‖² / 2σ²): 1 at the centre, fading to 0 away from it. A layer of them covers the
     input space with bumps that the next layer adds up (an RBF network). The centres start spread over −1…1 and the
     widths at the width you choose; widths are learned as log σ, so they stay positive. On a sequence, the same at
     every position. */
  class RBF {
    constructor(spec, cfg, inS, outS, ctx) {
      const nIn = inS[inS.length - 1], U = outS[outS.length - 1];
      Object.assign(this, { type: 'rbf', spec, inShape: inS, outShape: outS, nIn, U, rows: inS.length === 2 ? inS[0] : 1, mem: new Mem(ctx.Arr) });
      this.c = param(ctx, 'centres', [U, nIn], false); this.ls = param(ctx, 'log widths', [U], false);
      this.params = [this.c, this.ls];
      for (let i = 0; i < this.c.w.length; i++) this.c.w[i] = ctx.rng.uniform(-1, 1);
      this.ls.w.fill(Math.log(cfg.width > 0 ? cfg.width : 0.5));
    }
    alloc(B) { const m = this.mem, n = B * this.rows * this.U; this.d2 = m.view('d2', n); this.out = m.view('out', n); this.dx = m.view('dx', B * this.rows * this.nIn); }
    forward(x, B) {
      const { nIn: F, U, d2, out } = this, c = this.c.w, ls = this.ls.w;
      this.x = x;
      for (let r = 0, R = B * this.rows; r < R; r++) for (let j = 0; j < U; j++) {
        let s = 0;
        for (let i = 0; i < F; i++) { const d = x[r * F + i] - c[j * F + i]; s += d * d; }
        d2[r * U + j] = s; out[r * U + j] = Math.exp(-0.5 * s * Math.exp(-2 * ls[j]));
      }
      return out;
    }
    backward(dy, B, needDx) {
      const { nIn: F, U, d2, out, x } = this, c = this.c.w, ls = this.ls.w, gc = this.c.g, gl = this.ls.g, R = B * this.rows, dx = needDx ? this.dx : null;
      if (dx) dx.fill(0);
      for (let j = 0; j < U; j++) {
        const iv = Math.exp(-2 * ls[j]);
        for (let r = 0; r < R; r++) {
          // φ = e^E with E = −½·d²/σ²: g = ∂L/∂E; ∂E/∂(log σ) = d²/σ², ∂E/∂x = −(x − c)/σ² = −∂E/∂c
          const g = dy[r * U + j] * out[r * U + j];
          if (!g) continue;
          gl[j] += g * d2[r * U + j] * iv;
          const k = -g * iv;
          for (let i = 0; i < F; i++) { const v = k * (x[r * F + i] - c[j * F + i]); gc[j * F + i] -= v; if (dx) dx[r * F + i] += v; }
        }
      }
      return dx;
    }
  }

  /* Maxout: each unit computes k weighted sums of the input (its pieces) and keeps the largest, so it learns its own
     bend: with two pieces it can be a ReLU, an absolute value or a leaky ramp; with more, any convex shape made of k
     straight pieces (Goodfellow and others, 2013). Weights row j·k + p is piece p of unit j. On a sequence, at every
     position. */
  class Maxout {
    constructor(spec, cfg, inS, outS, ctx) {
      const nIn = inS[inS.length - 1], U = outS[outS.length - 1], k = cfg.pieces;
      Object.assign(this, { type: 'maxout', spec, inShape: inS, outShape: outS, nIn, U, k, rows: inS.length === 2 ? inS[0] : 1, scr: ctx.scr, mem: new Mem(ctx.Arr) });
      this.W = param(ctx, 'weights', [U * k, nIn], true); this.b = param(ctx, 'bias', [U * k], false);
      this.params = [this.W, this.b];
      gauss(ctx, this.W, initSd(ctx.init, nIn, false));
    }
    alloc(B) { const m = this.mem, R = B * this.rows, n = R * this.U; this.z = m.view('z', n * this.k); this.dz = m.view('dz', n * this.k); this.out = m.view('out', n); this.arg = m.view('arg', n, Int32Array); this.dx = m.view('dx', R * this.nIn); }
    forward(x, B) {
      const { nIn, U, k, z, out, arg } = this, R = B * this.rows;
      this.x = x;
      linear(z, x, this.W.w, this.b.w, R, nIn, U * k);
      for (let i = 0, n = R * U; i < n; i++) {
        const o = i * k;
        let best = 0;
        for (let p = 1; p < k; p++) if (z[o + p] > z[o + best]) best = p;
        out[i] = z[o + best]; arg[i] = best;
      }
      return out;
    }
    // the gradient goes only to the piece that won
    backward(dy, B, needDx) {
      const { nIn, U, k, dz, arg } = this, R = B * this.rows;
      dz.fill(0);
      for (let i = 0, n = R * U; i < n; i++) dz[i * k + arg[i]] = dy[i];
      linearBack(this.scr, needDx ? this.dx : null, false, this.W.g, this.b.g, dz, this.x, this.W.w, R, nIn, U * k);
      return needDx ? this.dx : null;
    }
  }

  /* Mixture of experts: E small dense layers (the experts) and a gate, a dense layer with one output per expert. For
     each example (each position of a sequence) the gate's softmax says how much to trust each expert, and the output
     is out = Σ_e g_e · expert_e(x). With top-k routing only the k experts the gate rates highest take part, their
     weights renormalised; the rest get no say and no gradient. (Large models then skip the others entirely, which is
     how they stay fast; here every expert is computed and the unchosen ones are left out.) Expert e's unit j is row
     e·U + j of the expert weights. */
  class MoE {
    constructor(spec, cfg, inS, outS, ctx) {
      const nIn = inS[inS.length - 1], U = outS[outS.length - 1], E = cfg.experts, A = NA.engine.ACTIVATIONS;
      Object.assign(this, { type: 'moe', spec, inShape: inS, outShape: outS, nIn, U, E, top: cfg.top > 0 && cfg.top < E ? cfg.top : E, rows: inS.length === 2 ? inS[0] : 1, scr: ctx.scr, mem: new Mem(ctx.Arr) });
      this.fn = activation(own(A, cfg.act) ? cfg.act : 'relu', ctx.f64);
      this.We = param(ctx, 'expert weights', [E * U, nIn], true); this.be = param(ctx, 'expert biases', [E * U], false);
      this.Wg = param(ctx, 'gate weights', [E, nIn], true); this.bg = param(ctx, 'gate bias', [E], false);
      this.params = [this.We, this.be, this.Wg, this.bg];
      gauss(ctx, this.We, initSd(ctx.init, nIn, this.fn.he)); gauss(ctx, this.Wg, initSd(linearInit(ctx), nIn));
    }
    alloc(B) {
      const m = this.mem, R = B * this.rows, n = R * this.E * this.U;
      for (const key of ['ze', 'ae', 'dze', 'dae']) this[key] = m.view(key, n);
      for (const key of ['logits', 'gate', 'dl']) this[key] = m.view(key, R * this.E);
      this.out = m.view('out', R * this.U); this.dx = m.view('dx', R * this.nIn);
    }
    forward(x, B) {
      const { nIn, U, E, top, ze, ae, logits, gate, out } = this, R = B * this.rows, EU = E * U;
      this.x = x;
      linear(ze, x, this.We.w, this.be.w, R, nIn, EU);
      activate(ae, ze, R * EU, this.fn);
      linear(logits, x, this.Wg.w, this.bg.w, R, nIn, E);
      for (let r = 0; r < R; r++) {
        const o = r * E;
        // an expert is chosen when fewer than top others rate higher (on a tie, the first counts as higher)
        let max = -Infinity, sum = 0;
        for (let e = 0; e < E; e++) {
          const v = logits[o + e];
          let above = 0;
          for (let q = 0; q < E; q++) if (logits[o + q] > v || (q < e && logits[o + q] === v)) above++;
          gate[o + e] = above < top ? 1 : 0;
          if (gate[o + e] && v > max) max = v;
        }
        for (let e = 0; e < E; e++) if (gate[o + e]) { const v = Math.exp(logits[o + e] - max); gate[o + e] = v; sum += v; }
        for (let e = 0; e < E; e++) gate[o + e] /= sum;
        for (let j = 0; j < U; j++) { let s = 0; for (let e = 0; e < E; e++) { const gv = gate[o + e]; if (gv) s += gv * ae[(r * E + e) * U + j]; } out[r * U + j] = s; }
      }
      return out;
    }
    backward(dy, B, needDx) {
      const { nIn, U, E, ae, gate, dae, dze, dl } = this, R = B * this.rows, EU = E * U;
      for (let r = 0; r < R; r++) {
        const o = r * E;
        let dot = 0;
        for (let e = 0; e < E; e++) {
          const gv = gate[o + e], eo = (r * E + e) * U;
          let s = 0;
          for (let j = 0; j < U; j++) { const d = dy[r * U + j]; dae[eo + j] = gv * d; s += d * ae[eo + j]; }
          dl[o + e] = s; dot += gv * s;
        }
        // through the softmax over the chosen experts; an unchosen one has g = 0, so it gets nothing
        for (let e = 0; e < E; e++) dl[o + e] = gate[o + e] * (dl[o + e] - dot);
      }
      activateBack(dze, dae, this.ze, ae, R * EU, this.fn);
      const dx = needDx ? this.dx : null;
      linearBack(this.scr, dx, false, this.We.g, this.be.g, dze, this.x, this.We.w, R, nIn, EU);
      linearBack(this.scr, dx, true, this.Wg.g, this.bg.g, dl, this.x, this.Wg.w, R, nIn, E);
      return dx;
    }
  }

  const LAYERS = { dense: Dense, activation: Act, dropout: Dropout, layernorm: LayerNormBlock, attention: Attention, transformer: Transformer, conv: Conv, pool: Pool, flatten: Flatten, gap: Gap,
    gated: Gated, batchnorm: BatchNorm, noise: Noise, fourier: Fourier, rnn: Recurrent, conv1d: Conv1d, position: Position, seqpool: SeqPool, patches: Patches, avgpool: AvgPool, parallel: Parallel,
    sepconv: SepConv, se: SE, coords: Coords, augment: Augment, gmp: Gmp, attnpool: AttnPool, ssm: SSM, rbf: RBF, maxout: Maxout, moe: MoE };

  // ------------------------------------------------------------------ losses
  function resolveLoss(loss, kind) {
    if (typeof loss === 'string') loss = { id: loss };
    const id = loss && loss.id ? loss.id : kind === 'value' ? 'mse' : 'xent', def = own(LOSSES, id) ? LOSSES[id] : null;
    if (!def) throw new Error(`"${id}" is not a loss this builder knows.`);
    if (!def.outputs.includes(kind)) throw new Error(`The ${def.label.toLowerCase()} loss does not work with ${OUTPUT_NAME[kind]}.`);
    return { id, smooth: id === 'xent' && loss && loss.smooth > 0 ? Math.min(loss.smooth, 0.999) : 0, delta: loss && loss.delta > 0 ? loss.delta : 0.2 };
  }

  // ------------------------------------------------------------------ model
  class Model {
    constructor(blocks, input, output, opts, chk) {
      const f64 = !!opts.f64, Arr = f64 ? Float64Array : Float32Array, seed = opts.seed === undefined ? 1 : opts.seed;
      const init = INITS.includes(opts.init) ? opts.init : 'auto', inp = normInput(input);
      Object.assign(this, { input, output, f64, Arr, seed, init, kind: output.kind, outShape: chk.outShape });
      this.mem = new Mem(Arr);
      this.rng = new NA.RNG((seed * 7919 + 101) >>> 0); // dropout masks, apart from the initialisation stream
      const ctx = { Arr, f64, init, rng: new NA.RNG(seed), scr: this.mem, model: this };
      this.embed = inp.kind === 'tokens' ? new Embed(ctx, inp.vocab, inp.len, inp.embed) : null;
      this.layers = blocks.map((spec, i) => new LAYERS[spec.type](spec, withDefaults(spec), chk.shapes[i].in, chk.shapes[i].out, ctx));
      const last = blocks.length ? chk.shapes[blocks.length - 1].out : inputShape(input).shape, K = chk.outShape[chk.outShape.length - 1];
      const head = { type: 'dense', units: K, act: 'linear' };
      this.layers.push(new Dense(head, head, last, chk.outShape, ctx, true));
      this.params = (this.embed ? this.embed.params : []).concat(...this.layers.map((L) => L.params));
      // the layers inside parallel branches, in order: block by block, branch by branch
      this.nested = [].concat(...this.layers.map((L) => L.inner || []));
      this.paramCount = this.params.reduce((s, p) => s + p.w.length, 0);
      // a layer sends a gradient back to its input only when something before it learns
      let learns = !!this.embed;
      for (const L of this.layers) { L.needDx = learns; if (L.params.length) learns = true; }
      this.inSize = inp.kind === 'tokens' ? inp.len : size(inputShape(input).shape);
      this.outSize = size(chk.outShape);
      this.outK = K;
      this.T = this.kind === 'tokens' ? chk.outShape[0] : 1;
      this.ySize = this.kind === 'classes' ? 1 : this.kind === 'tokens' ? this.T : K; // targets per example
      this._B = 0; this._dB = 0; this._t = 0; this._optId = null; this._state = null;
    }

    _alloc(B) {
      if (B === this._B) return;
      if (this.embed) this.embed.alloc(B); else this.x0 = this.mem.view('x0', B * this.inSize);
      for (const L of this.layers) L.alloc(B);
      this._B = B;
    }

    forward(X, B, train) { return this._forward(X, 0, B, !!train); }
    _forward(X, xo, B, train) {
      if (!isInt(B)) throw new Error('The batch size must be a whole number of at least 1.');
      if (X.length < xo + B * this.inSize) throw new Error(`The input holds fewer than ${B} examples.`);
      this._alloc(B);
      let x;
      if (this.embed) x = this.embed.forward(X, xo, B);
      else { x = this.x0; for (let i = 0, n = B * this.inSize; i < n; i++) x[i] = X[xo + i]; }
      for (const L of this.layers) x = L.forward(x, B, train);
      return x;
    }

    lossAndGrad(out, Y, B, loss) {
      const r = this._loss(out, Y, 0, B, loss, true);
      return { loss: r.loss, correct: r.correct, count: r.count };
    }
    /* Loss of out (B examples) against Y from offset yo: the sum, the mean (over examples, or over the positions
       with a target for tokens), accuracy counts, and with grad the gradient of the mean written into dOut. */
    _loss(out, Y, yo, B, loss, grad) {
      const L = resolveLoss(loss, this.kind), K = this.outK;
      if (!Y || Y.length < yo + B * this.ySize) throw new Error(`The targets hold fewer than ${B} examples.`);
      let d = null;
      if (grad) {
        if (this._dB !== B) { this.dOut = this.mem.view('dOut', B * this.outSize); this._dB = B; }
        d = this.dOut;
      }
      let sum = 0, correct = 0, count = 0;
      if (this.kind === 'value') {
        const delta = L.delta, inv = 1 / B;
        for (let i = 0, n = B * K; i < n; i++) {
          const r = out[i] - Y[yo + i], ar = Math.abs(r);
          let l, g;
          if (L.id === 'mae') { l = ar; g = r > 0 ? 1 : r < 0 ? -1 : 0; }
          else if (L.id === 'huber') { if (ar <= delta) { l = 0.5 * r * r; g = r; } else { l = delta * (ar - 0.5 * delta); g = r > 0 ? delta : -delta; } }
          else { l = 0.5 * r * r; g = r; }
          sum += l;
          if (d) d[i] = g * inv;
        }
        return { loss: sum * inv, sum, correct: 0, count: B };
      }
      // classes and tokens: rows of K logits, each with an integer target (tokens: −1 means no target)
      const tokens = this.kind === 'tokens', rows = tokens ? B * this.T : B;
      if (tokens) { for (let r = 0; r < rows; r++) if (Y[yo + r] >= 0) count++; } else count = B;
      const scale = count ? 1 / count : 0, on = 1 - L.smooth, off = L.smooth / K, p = this.mem.get('probs', K, Float64Array);
      for (let r = 0; r < rows; r++) {
        const o = r * K, y = Y[yo + r];
        if (tokens && y < 0) { if (d) d.fill(0, o, o + K); continue; }
        if (!(y >= 0 && y < K && y === Math.floor(y))) throw new Error(`Target ${y} is not a class between 0 and ${K - 1}.`);
        let best = 0, max = out[o];
        for (let c = 1; c < K; c++) if (out[o + c] > max) { max = out[o + c]; best = c; }
        if (best === y) correct++;
        if (L.id === 'hinge') {
          // Weston–Watkins: every wrong class within a margin of 1 of the right one costs its shortfall
          const zy = out[o + y];
          let l = 0, hits = 0;
          for (let c = 0; c < K; c++) {
            if (c === y) continue;
            const m = 1 + out[o + c] - zy;
            if (m > 0) { l += m; hits++; }
            if (d) d[o + c] = m > 0 ? scale : 0;
          }
          sum += l;
          if (d) d[o + y] = -hits * scale;
          continue;
        }
        let s = 0;
        for (let c = 0; c < K; c++) { const e = Math.exp(out[o + c] - max); p[c] = e; s += e; }
        for (let c = 0; c < K; c++) p[c] /= s;
        if (L.id === 'mse') {
          // ½Σ(p − y)² on the softmax: δⱼ = pⱼ[(pⱼ − yⱼ) − Σₖ pₖ(pₖ − yₖ)]
          let l = 0, dot = 0;
          for (let c = 0; c < K; c++) { const e = p[c] - (c === y ? 1 : 0); l += e * e; dot += p[c] * e; }
          sum += 0.5 * l;
          if (d) for (let c = 0; c < K; c++) d[o + c] = p[c] * (p[c] - (c === y ? 1 : 0) - dot) * scale;
        } else {
          // cross-entropy against q = (1 − ε)·one-hot + ε/K: loss = log Σ eᶻ − Σ q·z, gradient p − q
          let qz = on * out[o + y];
          if (off) { let zs = 0; for (let c = 0; c < K; c++) zs += out[o + c]; qz += off * zs; }
          sum += max + Math.log(s) - qz;
          if (d) { for (let c = 0; c < K; c++) d[o + c] = (p[c] - off) * scale; d[o + y] -= on * scale; }
        }
      }
      return { loss: count ? sum / count : 0, sum, correct, count };
    }

    // Backpropagates the dOut written by lossAndGrad, adding into every parameter's gradient.
    backward() {
      const B = this._B;
      let dy = this.dOut;
      for (let i = this.layers.length - 1; i >= 0 && dy; i--) dy = this.layers[i].backward(dy, B, this.layers[i].needDx);
      if (this.embed) this.embed.backward(dy, B);
    }

    zeroGrad() { for (const p of this.params) p.g.fill(0); }

    /* L2 norm of each layer's gradients, one per entry of model.layers (0 for layers without parameters); the token
       embedding's is the array's .embed (null without one). Each norm is also left on its layer as .gradNorm. */
    gradNorms() {
      const norm = (L) => {
        let s = 0;
        for (const p of L.params) { const g = p.g; for (let i = 0; i < g.length; i++) s += g[i] * g[i]; }
        return (L.gradNorm = Math.sqrt(s));
      };
      const out = this.layers.map(norm);
      out.embed = this.embed ? norm(this.embed) : null;
      return out;
    }

    step(X, Y, B, loss, opt) {
      opt = opt || {};
      this.zeroGrad();
      const r = this._loss(this._forward(X, 0, B, true), Y, 0, B, loss, true);
      this.backward();
      let ss = 0;
      for (const p of this.params) { const g = p.g; for (let i = 0; i < g.length; i++) ss += g[i] * g[i]; }
      const gradNorm = Math.sqrt(ss);
      // a non-finite gradient would turn every weight into NaN; skip the update and let the loss show it
      if (isFinite(gradNorm)) this._update(opt, opt.clip > 0 && gradNorm > opt.clip ? opt.clip / gradNorm : 1);
      return { loss: r.loss, correct: r.correct, count: r.count, gradNorm };
    }

    /* One update from a gradient of your own, for training methods whose loss is not one of the builder's (the
       reinforcement learning methods: a temporal-difference error on one action, a policy gradient). grad(out,
       dOut) reads the B outputs, writes the gradient of the loss for each of them into dOut and returns the loss;
       then the usual backward pass, clipping and optimiser update. */
    stepWith(X, B, grad, opt) {
      opt = opt || {};
      this.zeroGrad();
      const out = this._forward(X, 0, B, true);
      if (this._dB !== B) { this.dOut = this.mem.view('dOut', B * this.outSize); this._dB = B; }
      this.dOut.fill(0);
      const loss = grad(out, this.dOut);
      this.backward();
      let ss = 0;
      for (const p of this.params) { const g = p.g; for (let i = 0; i < g.length; i++) ss += g[i] * g[i]; }
      const gradNorm = Math.sqrt(ss);
      if (isFinite(gradNorm)) this._update(opt, opt.clip > 0 && gradNorm > opt.clip ? opt.clip / gradNorm : 1);
      return { loss, gradNorm };
    }
    /* The gradients of a quantity of your own, with no update: grad(out, dOut) writes its gradient for each of the B
       outputs into dOut; the parameters' gradients are left in .g (getGrads reads them as one array). Evolution's
       safe mutations use it to find how much the outputs depend on each weight. */
    gradWith(X, B, grad) {
      this.zeroGrad();
      const out = this._forward(X, 0, B, true);
      if (this._dB !== B) { this.dOut = this.mem.view('dOut', B * this.outSize); this._dB = B; }
      this.dOut.fill(0);
      grad(out, this.dOut);
      this.backward();
    }
    getGrads(into) {
      const out = into || new Float32Array(this.paramCount);
      let k = 0;
      for (const p of this.params) { out.set(p.g, k); k += p.g.length; }
      return out;
    }
    // every weight and bias as one array, and back (a target network's copy, a genetic algorithm's population)
    getWeights(into) {
      const out = into || new Float32Array(this.paramCount);
      let k = 0;
      for (const p of this.params) { out.set(p.w, k); k += p.w.length; }
      return out;
    }
    setWeights(w) {
      let k = 0;
      for (const p of this.params) { p.w.set(w.subarray(k, k + p.w.length)); k += p.w.length; }
    }

    /* One optimiser update from the gradients in .g, scaled by cs for clipping. Weight decay is L2 added to the
       gradient for sgd, momentum and adam, and a decoupled shrink (w −= lr·wd·w) for adamw; either way only
       on parameters marked decay. Moments restart when the optimiser changes. */
    _update(o, cs) {
      const id = own(OPTIMIZERS, o.id) ? o.id : 'adam';
      if (this._optId !== id) {
        this._optId = id; this._t = 0;
        this._state = this.params.map((p) => ({ m: new this.Arr(p.w.length), v: id === 'adam' || id === 'adamw' ? new this.Arr(p.w.length) : null }));
      }
      const t = ++this._t, lr = o.lr === undefined ? 0.01 : o.lr, wd = o.weightDecay || 0;
      const mu = o.momentum === undefined ? 0.9 : o.momentum, b1 = o.beta1 === undefined ? 0.9 : o.beta1, b2 = o.beta2 === undefined ? 0.999 : o.beta2, eps = o.eps === undefined ? 1e-8 : o.eps;
      const c1 = 1 - Math.pow(b1, t), c2 = 1 - Math.pow(b2, t);
      this.params.forEach((p, k) => {
        const w = p.w, g = p.g, n = w.length, M = this._state[k].m, V = this._state[k].v;
        const l2 = p.decay && id !== 'adamw' ? wd : 0, shrink = p.decay && id === 'adamw' ? lr * wd : 0;
        if (id === 'sgd') for (let i = 0; i < n; i++) w[i] -= lr * (g[i] * cs + l2 * w[i]);
        else if (id === 'momentum') for (let i = 0; i < n; i++) { M[i] = mu * M[i] + g[i] * cs + l2 * w[i]; w[i] -= lr * M[i]; }
        else {
          for (let i = 0; i < n; i++) {
            const gi = g[i] * cs + l2 * w[i];
            M[i] = b1 * M[i] + (1 - b1) * gi;
            V[i] = b2 * V[i] + (1 - b2) * gi * gi;
            w[i] -= lr * (M[i] / c1) / (Math.sqrt(V[i] / c2) + eps) + shrink * w[i];
          }
        }
      });
    }

    // Mean loss and accuracy counts over n examples in evaluation mode, in batches; gradients are not touched.
    evaluate(X, Y, n, loss, batch = 256) {
      const bs = Math.max(1, batch | 0);
      let sum = 0, correct = 0, count = 0;
      for (let s = 0; s < n; s += bs) {
        const b = Math.min(bs, n - s), r = this._loss(this._forward(X, s * this.inSize, b, false), Y, s * this.ySize, b, loss, false);
        sum += r.sum; correct += r.correct; count += r.count;
      }
      return { loss: count ? sum / count : 0, correct, count };
    }

    // The outputs (logits or values) for n examples in evaluation mode, as a new array of n × output size.
    predict(X, n) {
      const out = new this.Arr(n * this.outSize);
      for (let s = 0; s < n; s += 256) {
        const b = Math.min(256, n - s);
        out.set(this._forward(X, s * this.inSize, b, false), s * this.outSize);
      }
      return out;
    }
  }

  // A model for the blocks between input and output; throws with check()'s reason when they do not fit.
  function build(blocks, input, output, opts) {
    const chk = check(blocks, input, output);
    if (!chk.ok) {
      const i = chk.shapes.findIndex((s) => s.problem), problem = i >= 0 ? chk.shapes[i].problem : chk.problem;
      const err = new Error(problem.reason);
      err.problem = problem; err.index = i;
      throw err;
    }
    return new Model(blocks || [], input, output, opts || {}, chk);
  }

  NA.bx = { BLOCKS, LOSSES, OPTIMIZERS, MERGES, check, build, shapeText, stackShape, seHidden };
})(typeof globalThis !== 'undefined' ? (globalThis.NA = globalThis.NA || {}) : (window.NA = window.NA || {}));
