/* radar.js — the radar lab's signals and network. Synthetic I/Q made with the same maths as the training script
   (tools/train-radar.py): 256 complex samples at 20 MS/s holding receiver noise and, usually, one radar pulse;
   the 1-D CNN's forward pass from its exported weights; and the views a signal engineer would look at: the
   spectrogram, the instantaneous frequency and the network's class activation map. */
(function (NA) {
  'use strict';
  const N = 256, CLASSES = ['Noise only', 'Pulse', 'LFM chirp', 'Barker-13', 'Frank code', 'Costas FSK'];
  const BARKER = [1, 1, 1, 1, 1, -1, -1, 1, 1, -1, 1, -1, 1];
  const COSTAS = [];
  for (const g of [3, 5]) for (let s = 0; s < 6; s++) COSTAS.push(Array.from({ length: 6 }, (_, i) => { let v = 1; for (let k = 0; k < i + s; k++) v = (v * g) % 7; return v; }));
  const RD = { N, FS: 20e6, CLASSES, BARKER, COSTAS, SNR: [-12, 20] };

  // ------------------------------------------------------------------ signals
  /* One example. o: { cls, snr } to fix them; anything else is drawn at random like the training data.
     Returns { iq: Float32Array(2N) as [I row, Q row], cls, snr, p: the waveform's parameters, clean: the pulse alone } */
  RD.generate = function (rng, o = {}) {
    const cls = o.cls ?? rng.int(CLASSES.length), snr = o.snr ?? rng.uniform(RD.SNR[0], RD.SNR[1]);
    const L = Math.floor(96 + rng.next() * 129), n0 = Math.floor(rng.next() * (N - L + 1)), f0 = (rng.next() * 2 - 1) * 0.12, ph0 = rng.next() * 2 * Math.PI;
    const bw = 0.05 + rng.next() * 0.2, up = rng.next() < 0.5 ? -1 : 1, M = 4 + Math.floor(rng.next() * 3), costas = COSTAS[rng.int(COSTAS.length)];
    const p = { cls, snr, L, n0, f0, ph0, bw, up, M, costas };
    const amp = cls === 0 ? 0 : Math.pow(10, snr / 20), iq = new Float32Array(2 * N), clean = new Float32Array(2 * N), freq = new Float32Array(N);
    let phase = ph0;
    for (let t = 0; t < N; t++) {
      const m = t - n0, inside = m >= 0 && m < L, u = Math.min(0.999999, Math.max(0, m / L));
      let f = f0;
      if (cls === 2) f = f0 + up * bw * (u - 0.5);
      else if (cls === 5) f = f0 + (costas[Math.floor(u * 6)] - 3.5) * (6 / L);
      if (inside) phase += 2 * Math.PI * f; // the phase accumulates the frequency, sample by sample
      let ph = phase;
      if (cls === 3 && BARKER[Math.floor(u * 13)] < 0) ph += Math.PI;
      if (cls === 4) { const chip = Math.floor(u * M * M); ph += (2 * Math.PI * Math.floor(chip / M) * (chip % M)) / M; }
      freq[t] = inside ? f : NaN;
      if (inside && amp) { clean[t] = amp * Math.cos(ph); clean[N + t] = amp * Math.sin(ph); }
      iq[t] = clean[t] + rng.gauss() * Math.SQRT1_2; iq[N + t] = clean[N + t] + rng.gauss() * Math.SQRT1_2;
    }
    // automatic gain control: the whole window scaled to unit average power
    let pw = 0;
    for (let t = 0; t < N; t++) pw += iq[t] * iq[t] + iq[N + t] * iq[N + t];
    const g = 1 / Math.sqrt(pw / N);
    for (let i = 0; i < 2 * N; i++) { iq[i] *= g; clean[i] *= g; }
    return { iq, clean, cls, snr, p, freq, gain: g };
  };
  // A short description of what was generated, for captions.
  RD.describe = function (s) {
    const p = s.p, us = (n) => `${NA.fmt((n / RD.FS) * 1e6, 1)} µs`, mhz = (f) => `${f >= 0 ? '+' : '−'}${NA.fmt(Math.abs(f * RD.FS) / 1e6, 2)} MHz`;
    if (s.cls === 0) return 'receiver noise only';
    const base = `${us(p.L)} pulse at ${us(p.n0)}, carrier offset ${mhz(p.f0)}`;
    return [null, base, `${base}, ${p.up > 0 ? 'up' : 'down'}-chirp over ${NA.fmt(p.bw * RD.FS / 1e6, 2)} MHz`, `${base}, 13 chips of ${us(p.L / 13)}`,
      `${base}, ${p.M} × ${p.M} = ${p.M * p.M} chips`, `${base}, 6 hops of ${us(p.L / 6)}, ${NA.fmt((6 / p.L) * RD.FS / 1e6, 2)} MHz apart`][s.cls];
  };

  // ------------------------------------------------------------------ the network
  RD.load = function (W) {
    if (RD.net && RD.net.W === W) return RD.net;
    const flat = NA.tf.decodeWeights(W.weights), layers = [];
    let off = 0, cin = 2;
    W.widths.forEach((c, i) => {
      const k = W.ks[i], nw = c * cin * k;
      layers.push({ w: flat.subarray(off, off + nw), b: flat.subarray(off + nw, off + nw + c), co: c, ci: cin, k, pool: W.pools[i] });
      off += nw + c; cin = c;
    });
    const nc = W.classes.length, fc = { w: flat.subarray(off, off + nc * cin), b: flat.subarray(off + nc * cin, off + nc * cin + nc), co: nc, ci: cin };
    RD.net = { W, layers, fc, params: W.params };
    return RD.net;
  };
  /* Forward pass on iq ([I row, Q row]). Keeps every layer: pre (conv + bias, before ReLU), act (after ReLU),
     out (after pooling, the next layer's input); then gap (the global average), logits and probabilities. */
  RD.forward = function (net, iq) {
    let x = { c: 2, n: N, d: iq };
    const rec = [];
    for (const L of net.layers) {
      const n = x.n, p = L.k >> 1, pre = new Float32Array(L.co * n);
      for (let oc = 0; oc < L.co; oc++) {
        const o = pre.subarray(oc * n, (oc + 1) * n);
        o.fill(L.b[oc]);
        for (let ic = 0; ic < L.ci; ic++) for (let k = 0; k < L.k; k++) {
          const w = L.w[(oc * L.ci + ic) * L.k + k], xo = ic * n, sh = k - p, t0 = Math.max(0, -sh), t1 = Math.min(n, n - sh);
          for (let t = t0; t < t1; t++) o[t] += w * x.d[xo + t + sh];
        }
      }
      const act = pre.map((v) => (v > 0 ? v : 0));
      let out = { c: L.co, n, d: act };
      if (L.pool) {
        const h = n >> 1, d = new Float32Array(L.co * h);
        for (let c = 0; c < L.co; c++) for (let t = 0; t < h; t++) d[c * h + t] = Math.max(act[c * n + 2 * t], act[c * n + 2 * t + 1]);
        out = { c: L.co, n: h, d };
      }
      rec.push({ L, inp: x, pre, act, out });
      x = out;
    }
    const gap = new Float32Array(x.c);
    for (let c = 0; c < x.c; c++) { let s = 0; for (let t = 0; t < x.n; t++) s += x.d[c * x.n + t]; gap[c] = s / x.n; }
    const logits = new Float32Array(net.fc.co);
    for (let k = 0; k < net.fc.co; k++) { let s = net.fc.b[k]; for (let c = 0; c < x.c; c++) s += net.fc.w[k * x.c + c] * gap[c]; logits[k] = s; }
    const m = Math.max(...logits), e = logits.map((v) => Math.exp(v - m)), z = e.reduce((a, b) => a + b, 0);
    return { rec, last: x, gap, logits, probs: e.map((v) => v / z) };
  };
  /* Class activation map: the last layer's features weighted by one class's output weights, at each time step.
     Because the network averages over time before its linear layer, the class score is exactly the average of
     this curve plus the bias, so it shows where in the window the evidence for that class came from. */
  RD.cam = function (net, F, k) {
    const x = F.last, out = new Float32Array(x.n);
    for (let t = 0; t < x.n; t++) { let s = 0; for (let c = 0; c < x.c; c++) s += net.fc.w[k * x.c + c] * x.d[c * x.n + t]; out[t] = s; }
    return out;
  };
  // the weighted sum behind one number of layer l's output (before its ReLU): kernel window × every input channel
  RD.convTerms = function (r, oc, t) {
    const L = r.L, p = L.k >> 1, terms = [];
    for (let ic = 0; ic < L.ci; ic++) for (let k = 0; k < L.k; k++) {
      const s = t + k - p, inside = s >= 0 && s < r.inp.n;
      terms.push({ a: inside ? r.inp.d[ic * r.inp.n + s] : 0, w: L.w[(oc * L.ci + ic) * L.k + k], ic, s, inside });
    }
    return { terms, bias: L.b[oc] };
  };

  // ------------------------------------------------------------------ signal views
  // Short-time Fourier transform: Hann windows of `win` samples every `hop`; magnitude in dB, frequency bins
  // from −fs/2 to +fs/2 (row 0 is the most negative frequency)
  RD.spectrogram = function (iq, win = 32, hop = 4) {
    const frames = Math.floor((N - win) / hop) + 1, S = new Float32Array(frames * win), hann = Array.from({ length: win }, (_, i) => 0.5 - 0.5 * Math.cos((2 * Math.PI * i) / (win - 1)));
    let mx = -Infinity;
    for (let f = 0; f < frames; f++) for (let b = 0; b < win; b++) {
      const k = b - win / 2;
      let re = 0, im = 0;
      for (let i = 0; i < win; i++) {
        const t = f * hop + i, a = (-2 * Math.PI * k * i) / win, c = Math.cos(a), s = Math.sin(a), xr = iq[t] * hann[i], xi = iq[N + t] * hann[i];
        re += xr * c - xi * s; im += xr * s + xi * c;
      }
      const db = 10 * Math.log10(re * re + im * im + 1e-9);
      S[f * win + b] = db; mx = Math.max(mx, db);
    }
    return { S, frames, bins: win, hop, win, max: mx };
  };
  // instantaneous frequency (cycles per sample) from the phase step between neighbouring samples
  RD.instFreq = function (iq) {
    const f = new Float32Array(N);
    for (let t = 1; t < N; t++) { const re = iq[t] * iq[t - 1] + iq[N + t] * iq[N + t - 1], im = iq[N + t] * iq[t - 1] - iq[t] * iq[N + t - 1]; f[t] = Math.atan2(im, re) / (2 * Math.PI); }
    f[0] = f[1];
    return f;
  };

  NA.radar = RD;
})(typeof globalThis !== 'undefined' ? (globalThis.NA = globalThis.NA || {}) : (window.NA = window.NA || {}));
