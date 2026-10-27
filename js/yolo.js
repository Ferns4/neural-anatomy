/* yolo.js — the detector of the YOLO26 lab. Scenes of shapes drawn with exactly the maths of the training script
   (tools/train-yolo.py), the network's forward pass from its exported weights (BatchNorm folded into each conv),
   and the two ways of turning its raw output into boxes: the one-to-many head with NMS, the one-to-one head without.
   The drawing and the forward pass run in a Web Worker when the page may start one, so dragging stays smooth. */
(function (NA) {
  'use strict';

  /* The numeric core: no DOM and no other module, so the same source also runs inside the worker. */
  function core(Y) {
    const IMG = 128, NC = 4, STRIDES = [8, 16, 32], NV = 10;
    Object.assign(Y, { IMG, NC, STRIDES, NAMES: ['circle', 'triangle', 'square', 'star'], GRIDS: STRIDES.map((s) => IMG / s) });
    Y.A = Y.GRIDS.reduce((s, g) => s + g * g, 0); // 336 anchors

    // corners of a triangle, square or star: 10 points, short lists repeating their last corner
    Y.vertices = function (o) {
      const R = o.size / 2, out = [];
      for (let k = 0; k < NV; k++) {
        let a, r = R;
        if (o.cls === 1) a = o.ang - Math.PI / 2 + (2 * Math.PI * Math.min(k, 2)) / 3;
        else if (o.cls === 2) a = o.ang + Math.PI / 4 + (Math.PI / 2) * Math.min(k, 3);
        else { a = o.ang - Math.PI / 2 + (Math.PI * k) / 5; if (k % 2) r = 0.5 * R; }
        out.push([o.cx + r * Math.cos(a), o.cy + r * Math.sin(a)]);
      }
      return out;
    };
    // signed distance from (px, py) to a polygon, negative inside (the same formula as the training script)
    function polySdf(px, py, V) {
      const n = V.length;
      let d2 = (px - V[0][0]) ** 2 + (py - V[0][1]) ** 2, s = 1;
      for (let i = 0, j = n - 1; i < n; j = i, i++) {
        const ex = V[j][0] - V[i][0], ey = V[j][1] - V[i][1], wx = px - V[i][0], wy = py - V[i][1];
        const h = Math.min(1, Math.max(0, (wx * ex + wy * ey) / Math.max(ex * ex + ey * ey, 1e-12))), bx = wx - ex * h, by = wy - ey * h;
        d2 = Math.min(d2, bx * bx + by * by);
        const c1 = py >= V[i][1], c2 = py < V[j][1], c3 = ex * wy > ey * wx;
        if ((c1 && c2 && c3) || (!c1 && !c2 && !c3)) s = -s;
      }
      return s * Math.sqrt(d2);
    }
    // how far a point is outside a shape (negative inside), in pixels
    Y.shapeDist = (o, px, py) => (o.cls === 0 ? Math.sqrt((px - o.cx) ** 2 + (py - o.cy) ** 2) - o.size / 2 : polySdf(px, py, Y.vertices(o)));
    /* Draws a scene into a planar RGB image (3 x 128 x 128, values in [0, 1], 8-bit steps like a canvas), adding
       sigma times the given noise field. A line or shape only touches the pixels within a pixel of its outline:
       farther away its coverage is 0. */
    Y.render = function (S, nz) {
      const P = IMG * IMG, img = new Float32Array(3 * P), cs = Math.cos(S.bang), sn = Math.sin(S.bang);
      for (let y = 0; y < IMG; y++) for (let x = 0; x < IMG; x++) {
        const t = Math.min(1, Math.max(0, ((x + 0.5 - IMG / 2) * cs + (y + 0.5 - IMG / 2) * sn) / IMG + 0.5)), p = y * IMG + x;
        for (let c = 0; c < 3; c++) img[c * P + p] = S.c0[c] + (S.c1[c] - S.c0[c]) * t;
      }
      const paint = (bb, dist, col) => {
        const x0 = Math.max(0, Math.floor(bb[0]) - 1), y0 = Math.max(0, Math.floor(bb[1]) - 1), x1 = Math.min(IMG - 1, Math.ceil(bb[2]) + 1), y1 = Math.min(IMG - 1, Math.ceil(bb[3]) + 1);
        for (let y = y0; y <= y1; y++) for (let x = x0; x <= x1; x++) {
          const a = Math.min(1, Math.max(0, 0.5 - dist(x + 0.5, y + 0.5)));
          if (a <= 0) continue;
          const p = y * IMG + x;
          for (let c = 0; c < 3; c++) img[c * P + p] += (col[c] - img[c * P + p]) * a;
        }
      };
      for (const l of S.lines) {
        const bx = l.x1 - l.x0, by = l.y1 - l.y0, bb2 = Math.max(bx * bx + by * by, 1e-9), hw = l.w / 2;
        paint([Math.min(l.x0, l.x1) - hw, Math.min(l.y0, l.y1) - hw, Math.max(l.x0, l.x1) + hw, Math.max(l.y0, l.y1) + hw], (px, py) => {
          const h = Math.min(1, Math.max(0, ((px - l.x0) * bx + (py - l.y0) * by) / bb2)), dx = px - l.x0 - bx * h, dy = py - l.y0 - by * h;
          return Math.sqrt(dx * dx + dy * dy) - hw;
        }, l.col);
      }
      for (const o of S.objs) {
        const R = o.size / 2;
        if (o.cls === 0) paint([o.cx - R, o.cy - R, o.cx + R, o.cy + R], (px, py) => Math.sqrt((px - o.cx) ** 2 + (py - o.cy) ** 2) - R, o.col);
        else { const V = Y.vertices(o); paint([o.cx - R, o.cy - R, o.cx + R, o.cy + R], (px, py) => polySdf(px, py, V), o.col); }
      }
      const sg = nz ? S.sigma : 0;
      for (let i = 0; i < img.length; i++) img[i] = Math.round(Math.min(1, Math.max(0, img[i] + (sg ? sg * nz[i] : 0))) * 255) / 255;
      return img;
    };

    // ---------------------------------------------------------------- tensors and layers
    const T = (c, h, w, d) => ({ c, h, w, d: d || new Float32Array(c * h * w) });
    // SiLU(v) = v·σ(v), with σ read from a table and interpolated (error under 1e-6, several times faster than exp)
    const SIG = new Float32Array(4097);
    for (let i = 0; i <= 4096; i++) SIG[i] = 1 / (1 + Math.exp(-(i / 128 - 16)));
    const silu = (v) => {
      if (v <= -16) return v * Math.exp(v);
      if (v >= 16) return v;
      const f = (v + 16) * 128, i = f | 0;
      return v * (SIG[i] + (SIG[i + 1] - SIG[i]) * (f - i));
    };
    Y.silu = silu;
    /* 2-D convolution with zero padding k/2, stride s and g groups; BatchNorm is already folded into w and b.
       For ordinary convs the input is unrolled into patches (im2col: for each output pixel, the k x k window of
       every input channel in a row), so each output number is a dot product of two contiguous rows; four output
       channels and two pixels are done together, keeping eight running sums in registers. Grouped (depthwise)
       convs are tiny and run directly. */
    let PATCH = new Float32Array(1 << 16);
    function conv(x, L, s, g, act) {
      const k = L.k, kk = k * k, p = k >> 1, ho = Math.floor((x.h + 2 * p - k) / s) + 1, wo = Math.floor((x.w + 2 * p - k) / s) + 1, n = ho * wo;
      const out = T(L.co, ho, wo), od = out.d, w = L.w, b = L.b, xd = x.d, hw = x.h * x.w;
      if (g > 1) {
        const cog = L.co / g;
        for (let oc = 0; oc < L.co; oc++) {
          const gi = Math.floor(oc / cog);
          for (let oy = 0; oy < ho; oy++) for (let ox = 0; ox < wo; ox++) {
            let sum = b[oc];
            for (let icg = 0; icg < L.ci; icg++) for (let ky = 0; ky < k; ky++) {
              const iy = oy * s + ky - p;
              if (iy < 0 || iy >= x.h) continue;
              for (let kx = 0; kx < k; kx++) { const ix = ox * s + kx - p; if (ix >= 0 && ix < x.w) sum += w[((oc * L.ci + icg) * k + ky) * k + kx] * xd[(gi * L.ci + icg) * hw + iy * x.w + ix]; }
            }
            od[oc * n + oy * wo + ox] = sum;
          }
        }
      } else {
        const J = L.ci * kk;
        if (PATCH.length < n * J) PATCH = new Float32Array(n * J);
        const P = PATCH;
        if (p) P.fill(0, 0, n * J); // the padding stays zero
        for (let ic = 0, j = 0; ic < L.ci; ic++) for (let ky = 0; ky < k; ky++) for (let kx = 0; kx < k; kx++, j++) {
          const ox0 = Math.max(0, Math.ceil((p - kx) / s)), ox1 = Math.min(wo, Math.floor((x.w - 1 - kx + p) / s) + 1);
          for (let oy = 0; oy < ho; oy++) {
            const iy = oy * s + ky - p;
            if (iy < 0 || iy >= x.h) continue;
            const row = ic * hw + iy * x.w;
            for (let ox = ox0, ix = ox0 * s + kx - p, q = (oy * wo + ox0) * J + j; ox < ox1; ox++, ix += s, q += J) P[q] = xd[row + ix];
          }
        }
        for (let oc = 0; oc < L.co; oc += 4) {
          const nb = Math.min(4, L.co - oc), w0 = oc * J, w1 = (oc + Math.min(1, nb - 1)) * J, w2 = (oc + Math.min(2, nb - 1)) * J, w3 = (oc + Math.min(3, nb - 1)) * J;
          for (let q = 0; q < n; q += 2) {
            const pa = q * J, two = q + 1 < n, pb = two ? pa + J : pa;
            let a0 = 0, a1 = 0, a2 = 0, a3 = 0, c0 = 0, c1 = 0, c2 = 0, c3 = 0;
            for (let j = 0; j < J; j++) {
              const u = P[pa + j], v = P[pb + j], x0 = w[w0 + j], x1 = w[w1 + j], x2 = w[w2 + j], x3 = w[w3 + j];
              a0 += x0 * u; a1 += x1 * u; a2 += x2 * u; a3 += x3 * u; c0 += x0 * v; c1 += x1 * v; c2 += x2 * v; c3 += x3 * v;
            }
            const o = oc * n + q;
            od[o] = a0 + b[oc]; if (two) od[o + 1] = c0 + b[oc];
            if (nb > 1) { od[o + n] = a1 + b[oc + 1]; if (two) od[o + n + 1] = c1 + b[oc + 1]; }
            if (nb > 2) { od[o + 2 * n] = a2 + b[oc + 2]; if (two) od[o + 2 * n + 1] = c2 + b[oc + 2]; }
            if (nb > 3) { od[o + 3 * n] = a3 + b[oc + 3]; if (two) od[o + 3 * n + 1] = c3 + b[oc + 3]; }
          }
        }
      }
      if (act) for (let i = 0; i < od.length; i++) od[i] = silu(od[i]);
      return out;
    }
    // every derived tensor remembers where its numbers came from (sum, parts, from + c0, up, pool), so the
    // diagram can follow any number back to the convolution that made it
    const add = (a, b) => { const o = T(a.c, a.h, a.w); for (let i = 0; i < o.d.length; i++) o.d[i] = a.d[i] + b.d[i]; o.sum = [a, b]; return o; };
    const cat = (xs) => { const o = T(xs.reduce((s, x) => s + x.c, 0), xs[0].h, xs[0].w); let off = 0; for (const x of xs) { o.d.set(x.d, off); off += x.d.length; } o.parts = xs; return o; };
    const slice = (x, c0, c1) => Object.assign(T(c1 - c0, x.h, x.w, x.d.slice(c0 * x.h * x.w, c1 * x.h * x.w)), { from: x, c0 });
    const up2 = (x) => { const o = T(x.c, x.h * 2, x.w * 2); for (let c = 0; c < x.c; c++) for (let y = 0; y < o.h; y++) for (let xx = 0; xx < o.w; xx++) o.d[(c * o.h + y) * o.w + xx] = x.d[(c * x.h + (y >> 1)) * x.w + (xx >> 1)]; o.up = x; return o; };
    // 5 x 5 max-pool, stride 1, padding 2 (padding never wins a max)
    const pool5 = (x) => {
      const o = T(x.c, x.h, x.w);
      o.pool = x;
      for (let c = 0; c < x.c; c++) for (let y = 0; y < x.h; y++) for (let xx = 0; xx < x.w; xx++) {
        let m = -Infinity;
        for (let dy = -2; dy <= 2; dy++) { const yy = y + dy; if (yy < 0 || yy >= x.h) continue; for (let dx = -2; dx <= 2; dx++) { const xq = xx + dx; if (xq >= 0 && xq < x.w) m = Math.max(m, x.d[(c * x.h + yy) * x.w + xq]); } }
        o.d[(c * x.h + y) * x.w + xx] = m;
      }
      return o;
    };
    Y.tensor = { T, add, cat, slice, up2, pool5, conv };

    /* Runs the network on an image (planar RGB in [0, 1]); P: the layers by PyTorch module name. Returns the head
       outputs, K: each block's output (and its inner parts) and R: every conv's input, output, stride, groups and
       activation, keyed by module name (R[n].name names its weights in P). */
    Y.forward = function (P, img) {
      const R = {}, K = {};
      const C = (n, x, s = 1, g = 1, act = true) => { const y = conv(x, P[n], s, g, act); R[n] = { name: n, inp: x, out: y, s, g, act }; return y; };
      const bott = (n, x) => add(x, C(n + '.cv2', C(n + '.cv1', x)));
      const c3k = (n, x) => C(n + '.cv3', cat([bott(n + '.m.1', bott(n + '.m.0', C(n + '.cv1', x))), C(n + '.cv2', x)]));
      const attn = (n, x, nh) => {
        const Cc = x.c, N = x.h * x.w, hd = Cc / nh, kd = hd >> 1, qkv = C(n + '.qkv', x, 1, 1, false), y = T(Cc, x.h, x.w), v = T(Cc, x.h, x.w), maps = [], sc = 1 / Math.sqrt(kd);
        for (let h = 0; h < nh; h++) {
          const base = h * (2 * kd + hd) * N, att = new Float32Array(N * N);
          for (let i = 0; i < N; i++) {
            let m = -Infinity;
            for (let j = 0; j < N; j++) { let s = 0; for (let d = 0; d < kd; d++) s += qkv.d[base + d * N + i] * qkv.d[base + (kd + d) * N + j]; att[i * N + j] = s * sc; m = Math.max(m, s * sc); }
            let z = 0;
            for (let j = 0; j < N; j++) { att[i * N + j] = Math.exp(att[i * N + j] - m); z += att[i * N + j]; }
            for (let j = 0; j < N; j++) att[i * N + j] /= z;
          }
          for (let d = 0; d < hd; d++) for (let i = 0; i < N; i++) {
            let s = 0;
            for (let j = 0; j < N; j++) s += qkv.d[base + (2 * kd + d) * N + j] * att[i * N + j];
            y.d[(h * hd + d) * N + i] = s;
          }
          v.d.set(qkv.d.subarray(base + 2 * kd * N, base + (2 * kd + hd) * N), h * hd * N);
          maps.push(att);
        }
        K[n + '.att'] = { maps, nh, kd, hd, N, y, v, qkv };
        y.attn = n + '.att';
        v.chmap = { src: qkv, map: Array.from({ length: Cc }, (_, c) => Math.floor(c / hd) * (2 * kd + hd) + 2 * kd + (c % hd)) };
        return C(n + '.proj', add(y, C(n + '.pe', v, 1, Cc, false)), 1, 1, false);
      };
      const psa = (n, x) => { const a = add(x, attn(n + '.attn', x, Math.max(x.c >> 6, 1))); K[n + '.mid'] = a; return add(a, C(n + '.ffn.1', C(n + '.ffn.0', a), 1, 1, false)); };
      const c3k2 = (n, x, kind) => {
        const y1 = C(n + '.cv1', x), h = y1.c >> 1, a = slice(y1, 0, h), b = slice(y1, h, y1.c);
        const m = kind === 'bott' ? bott(n + '.m.0', b) : kind === 'c3k' ? c3k(n + '.m.0', b) : psa(n + '.m.0.1', bott(n + '.m.0.0', b));
        K[n + '.in'] = x; K[n + '.parts'] = [a, b, m];
        return C(n + '.cv2', cat([a, b, m]));
      };
      const sppf = (n, x) => { const y = [C(n + '.cv1', x, 1, 1, false)]; for (let i = 0; i < 3; i++) y.push(pool5(y[i])); K[n + '.parts'] = y; return add(C(n + '.cv2', cat(y)), x); };
      const c2psa = (n, x) => { const y1 = C(n + '.cv1', x), c = y1.c >> 1, a = slice(y1, 0, c), b0 = slice(y1, c, y1.c), b = psa(n + '.m.0', b0); K[n + '.parts'] = [a, b0, b]; return C(n + '.cv2', cat([a, b])); };
      const x0 = T(3, IMG, IMG, img);
      K.input = x0;
      K.b0 = C('b0', x0, 2); K.b1 = C('b1', K.b0, 2); K.b2 = c3k2('b2', K.b1, 'bott');
      K.b3 = C('b3', K.b2, 2); K.b4 = c3k2('b4', K.b3, 'bott');
      K.b5 = C('b5', K.b4, 2); K.b6 = c3k2('b6', K.b5, 'c3k');
      K.b7 = C('b7', K.b6, 2); K.b8 = c3k2('b8', K.b7, 'c3k');
      K.b9 = sppf('b9', K.b8); K.b10 = c2psa('b10', K.b9);
      K.u11 = up2(K.b10); K.h13 = c3k2('h13', cat([K.u11, K.b6]), 'c3k');
      K.u14 = up2(K.h13); K.h16 = c3k2('h16', cat([K.u14, K.b4]), 'c3k');
      K.h17 = C('h17', K.h16, 2); K.h19 = c3k2('h19', cat([K.h17, K.h13]), 'c3k');
      K.h20 = C('h20', K.h19, 2); K.h22 = c3k2('h22', cat([K.h20, K.b10]), 'attn');
      const heads = {};
      for (const [tag, pre] of [['o2m', 'detect.'], ['o2o', 'detect.one2one_']]) {
        heads[tag] = [K.h16, K.h19, K.h22].map((f, i) => {
          const bx = C(`${pre}cv2.${i}.2`, C(`${pre}cv2.${i}.1`, C(`${pre}cv2.${i}.0`, f)), 1, 1, false);
          let c = C(`${pre}cv3.${i}.0.1`, C(`${pre}cv3.${i}.0.0`, f, 1, f.c));
          c = C(`${pre}cv3.${i}.1.1`, C(`${pre}cv3.${i}.1.0`, c, 1, c.c));
          const cls = C(`${pre}cv3.${i}.2`, c, 1, 1, false);
          K[`${tag}.${i}`] = { box: bx, cls };
          return cat([bx, cls]);
        });
      }
      return { R, K, heads };
    };
    return Y;
  }

  const Y = core({});
  const { IMG, NC, STRIDES } = Y;

  // ------------------------------------------------------------------ scenes (page side: they use NA.RNG)
  function hsv(h, s, v) {
    h = (((h % 1) + 1) % 1) * 6;
    const i = Math.floor(h) % 6, f = h - Math.floor(h), p = v * (1 - s), q = v * (1 - s * f), t = v * (1 - s * (1 - f));
    return [[v, t, p], [q, v, p], [p, v, t], [p, q, v], [t, p, v], [v, p, q]][i];
  }
  Y.hsv = hsv;
  const luma = (c) => 0.299 * c[0] + 0.587 * c[1] + 0.114 * c[2];
  Y.boxOf = function (o) {
    let b;
    if (o.cls === 0) { const R = o.size / 2; b = [o.cx - R, o.cy - R, o.cx + R, o.cy + R]; }
    else { b = [Infinity, Infinity, -Infinity, -Infinity]; for (const [x, y] of Y.vertices(o)) { b[0] = Math.min(b[0], x); b[1] = Math.min(b[1], y); b[2] = Math.max(b[2], x); b[3] = Math.max(b[3], y); } }
    return b.map((q) => Math.min(IMG, Math.max(0, q)));
  };
  Y.iou = function (a, b) {
    const iw = Math.max(0, Math.min(a[2], b[2]) - Math.max(a[0], b[0])), ih = Math.max(0, Math.min(a[3], b[3]) - Math.max(a[1], b[1])), inter = iw * ih;
    const u = (a[2] - a[0]) * (a[3] - a[1]) + (b[2] - b[0]) * (b[3] - b[1]) - inter;
    return u > 0 ? inter / u : 0;
  };
  Y.bgAt = (S, x, y) => { const t = Math.min(1, Math.max(0, ((x - IMG / 2) * Math.cos(S.bang) + (y - IMG / 2) * Math.sin(S.bang)) / IMG + 0.5)); return S.c0.map((c, i) => c + (S.c1[i] - c) * t); };
  // A colour that stands out from the background behind (x, y), as in training; null if none of 8 tries does.
  Y.pickColour = function (rng, S, x, y) {
    const bg = Y.bgAt(S, x, y);
    for (let t = 0; t < 8; t++) {
      const c = hsv(rng.next(), rng.uniform(0.45, 1), rng.uniform(0.4, 1));
      if (Math.abs(luma(c) - luma(bg)) >= 0.12 || Math.hypot(c[0] - bg[0], c[1] - bg[1], c[2] - bg[2]) >= 0.4) return c;
    }
    return null;
  };
  // sizes and positions as in training: log-uniform 7 to 44 px across, at least half of the shape inside
  Y.SIZE = [7, 44];
  Y.clampShape = (o) => { const R = o.size / 2; o.cx = Math.min(IMG - R / 2, Math.max(R / 2, o.cx)); o.cy = Math.min(IMG - R / 2, Math.max(R / 2, o.cy)); return o; };
  Y.randomShape = function (rng, S, cls) {
    const size = Math.exp(rng.uniform(Math.log(Y.SIZE[0]), Math.log(Y.SIZE[1]))), R = size / 2;
    const o = { cls: cls ?? rng.int(NC), size, cx: rng.next() * (IMG - R) + R / 2, cy: rng.next() * (IMG - R) + R / 2, ang: rng.uniform(0, 2 * Math.PI) };
    o.col = Y.pickColour(rng, S, o.cx, o.cy);
    return o.col ? o : null;
  };
  // A random scene from the training distribution: a two-colour gradient, up to 4 thin lines, 1-7 shapes.
  Y.randomScene = function (rng, nObj) {
    const h0 = rng.next(), s0 = rng.uniform(0, 0.35), v0 = rng.uniform(0.12, 0.95);
    const S = { c0: hsv(h0, s0, v0), c1: hsv(h0 + rng.uniform(-0.08, 0.08), s0, Math.min(1, Math.max(0.05, v0 + rng.uniform(-0.3, 0.3)))), bang: rng.uniform(0, 2 * Math.PI), sigma: rng.uniform(0, 0.05), lines: [], objs: [], seed: 1 + rng.int(1e9) };
    const nl = rng.int(5);
    for (let i = 0; i < nl; i++) {
      const x0 = rng.uniform(0, IMG), y0 = rng.uniform(0, IMG), a = rng.uniform(0, 2 * Math.PI), l = rng.uniform(20, 90);
      S.lines.push({ x0, y0, x1: x0 + l * Math.cos(a), y1: y0 + l * Math.sin(a), w: rng.uniform(0.8, 2.2), col: hsv(rng.next(), rng.uniform(0, 0.7), rng.uniform(0.15, 1)) });
    }
    const target = nObj || 1 + rng.int(7);
    for (let k = 0; k < 14 && S.objs.length < target; k++) {
      const o = Y.randomShape(rng, S);
      if (o && S.objs.every((p) => Y.iou(Y.boxOf(o), Y.boxOf(p)) < 0.35)) S.objs.push(o);
    }
    return S;
  };
  // The training script's scene format (tests): lists of numbers.
  Y.sceneFromSpec = (sp) => ({ c0: sp.c0, c1: sp.c1, bang: sp.bang, sigma: sp.sigma, seed: 1,
    lines: sp.lines.map((l) => ({ x0: l[0], y0: l[1], x1: l[2], y1: l[3], w: l[4], col: l.slice(5, 8) })),
    objs: sp.objs.map((o) => ({ cls: Math.round(o[0]), cx: o[1], cy: o[2], size: o[3], ang: o[4], col: o.slice(5, 8) })) });
  // a fixed field of Gaussian noise per scene, so moving a shape does not reshuffle the noise
  const noiseCache = new Map();
  Y.noiseField = function (seed) {
    let f = noiseCache.get(seed);
    if (!f) {
      const rng = new NA.RNG(seed);
      f = new Float32Array(3 * IMG * IMG);
      for (let i = 0; i < f.length; i++) f[i] = rng.gauss();
      if (noiseCache.size > 6) noiseCache.delete(noiseCache.keys().next().value);
      noiseCache.set(seed, f);
    }
    return f;
  };
  Y.image = (S, noise = true) => Y.render(S, noise ? Y.noiseField(S.seed) : null);

  // ------------------------------------------------------------------ the network
  Y.load = function (W) {
    if (Y.net && Y.net.W === W) return Y.net;
    const flat = NA.tf.decodeWeights(W.weights), P = {};
    let off = 0;
    for (const [name, shape] of W.layout) {
      const nw = shape.reduce((a, b) => a * b, 1);
      P[name] = { name, w: flat.slice(off, off + nw), b: flat.slice(off + nw, off + nw + shape[0]), co: shape[0], ci: shape[1], k: shape[2] || 1 };
      off += nw + shape[0];
    }
    Y.net = { W, P, params: W.params };
    return Y.net;
  };
  // the record of a forward pass gets its weights back (the worker sends only their names)
  Y.relink = function (net, F) { for (const r of Object.values(F.R)) r.L = net.P[r.name]; return F; };
  Y.run = (net, img) => Y.relink(net, Y.forward(net.P, img));

  /* A detector that works off the page's thread when it can: request(scene) draws and runs the newest scene, one
     at a time; done(result) gets { scene, img, F } with F the full record of the forward pass. If a worker cannot
     be started (or fails), the same code runs on the page instead, a frame later. */
  Y.Runner = class {
    constructor(net, done) {
      this.net = net; this.done = done; this.busy = false; this.next = null; this.id = 0;
      try {
        const src = `const Y = (${core.toString()})({});\nlet P = null;\nonmessage = (e) => {\n  const m = e.data;\n  if (m.P) { P = m.P; return; }\n  const img = Y.render(m.scene, m.noise), t0 = performance.now(), F = Y.forward(P, img);\n  postMessage({ id: m.id, img, F, ms: performance.now() - t0 });\n};`;
        this.url = URL.createObjectURL(new Blob([src], { type: 'text/javascript' }));
        this.worker = new Worker(this.url);
        this.worker.onmessage = (e) => this.receive(e.data);
        this.worker.onerror = (e) => { e.preventDefault && e.preventDefault(); this.fallback(); };
        this.worker.postMessage({ P: net.P });
      } catch (err) { this.fallback(); }
    }
    fallback() {
      if (this.worker) { try { this.worker.terminate(); } catch (e) { /* already gone */ } }
      this.worker = null;
      if (this.busy && this.pending) { const p = this.pending; this.busy = false; this.request(p.scene); }
    }
    request(scene) {
      if (this.busy) { this.next = scene; return; }
      this.busy = true;
      const id = ++this.id, noise = scene.sigma > 0 ? Y.noiseField(scene.seed) : null, snap = JSON.parse(JSON.stringify(scene));
      this.pending = { id, scene: snap };
      if (this.worker) { this.worker.postMessage({ id, scene: snap, noise }); return; }
      setTimeout(() => {
        const img = Y.render(snap, noise), t0 = performance.now(), F = Y.forward(this.net.P, img);
        this.receive({ id, img, F, ms: performance.now() - t0 });
      }, 0);
    }
    receive(m) {
      if (!this.pending || m.id !== this.pending.id) return;
      const scene = this.pending.scene;
      this.busy = false; this.pending = null;
      this.ms = m.ms;
      this.done({ scene, img: m.img, F: Y.relink(this.net, m.F), ms: m.ms });
      if (this.next) { const s = this.next; this.next = null; this.request(s); }
    }
    destroy() { if (this.worker) this.worker.terminate(); if (this.url) URL.revokeObjectURL(this.url); this.worker = null; this.done = () => {}; }
  };

  // The weighted sum behind one number of a conv's output: its inputs (a k x k window of every input channel in
  // its group, zero outside the picture), weights, bias and activation.
  Y.convTerms = function (rec, oc, oy, ox) {
    const L = rec.L, x = rec.inp, k = L.k, p = k >> 1, cog = L.co / rec.g, gi = Math.floor(oc / cog), terms = [];
    for (let icg = 0; icg < L.ci; icg++) for (let ky = 0; ky < k; ky++) for (let kx = 0; kx < k; kx++) {
      const ic = gi * L.ci + icg, iy = oy * rec.s + ky - p, ix = ox * rec.s + kx - p, inside = iy >= 0 && iy < x.h && ix >= 0 && ix < x.w;
      terms.push({ a: inside ? x.d[(ic * x.h + iy) * x.w + ix] : 0, w: L.w[((oc * L.ci + icg) * k + ky) * k + kx], ic, iy, ix, inside });
    }
    return { terms, bias: L.b[oc], act: rec.act };
  };

  // ------------------------------------------------------------------ from raw output to boxes
  /* A head's three maps → every anchor's box (pixels) and class probabilities. No DFL (reg_max = 1): the first
     four channels are the distances from the anchor (the cell's centre) to the box's left, top, right and bottom
     edges, in units of the cell size. */
  Y.decode = function (maps) {
    const A = Y.A, boxes = new Float32Array(A * 4), scores = new Float32Array(A * NC), dist = new Float32Array(A * 4), where = [];
    let a = 0;
    maps.forEach((m, si) => {
      const s = STRIDES[si], n = m.h * m.w;
      for (let gy = 0; gy < m.h; gy++) for (let gx = 0; gx < m.w; gx++, a++) {
        const q = gy * m.w + gx, l = m.d[q], t = m.d[n + q], r = m.d[2 * n + q], b = m.d[3 * n + q];
        dist.set([l, t, r, b], a * 4);
        boxes.set([(gx + 0.5 - l) * s, (gy + 0.5 - t) * s, (gx + 0.5 + r) * s, (gy + 0.5 + b) * s], a * 4);
        for (let c = 0; c < NC; c++) scores[a * NC + c] = 1 / (1 + Math.exp(-m.d[(4 + c) * n + q]));
        where.push({ si, gy, gx, s });
      }
    });
    return { boxes, scores, dist, where };
  };
  const boxAt = (D, a) => Array.from(D.boxes.subarray(a * 4, a * 4 + 4));
  // One-to-one head: the highest anchor-and-class scores, as they are. Duplicates were trained away, so no NMS.
  // (The scores over the threshold, best first, up to maxDet: the same as the top maxDet of all of them, thresholded.)
  Y.oneToOne = function (D, conf, maxDet = 100) {
    const idx = [];
    for (let i = 0; i < Y.A * NC; i++) if (D.scores[i] > conf) idx.push(i);
    idx.sort((p, q) => D.scores[q] - D.scores[p]);
    return idx.slice(0, maxDet).map((i) => ({ a: Math.floor(i / NC), cls: i % NC, score: D.scores[i], box: boxAt(D, Math.floor(i / NC)) }));
  };
  // One-to-many head: every anchor's best class over the threshold, best first; then (unless raw) class-aware
  // greedy NMS, which marks each removed box with the kept box that removed it.
  Y.oneToMany = function (D, conf, iouThr, raw) {
    const cand = [];
    for (let a = 0; a < Y.A; a++) {
      let c = 0;
      for (let k = 1; k < NC; k++) if (D.scores[a * NC + k] > D.scores[a * NC + c]) c = k;
      if (D.scores[a * NC + c] > conf) cand.push({ a, cls: c, score: D.scores[a * NC + c], box: boxAt(D, a) });
    }
    cand.sort((p, q) => q.score - p.score);
    if (raw) return cand;
    const keep = [];
    for (const d of cand) {
      const by = keep.find((k) => k.cls === d.cls && Y.iou(k.box, d.box) > iouThr);
      if (by) d.suppressedBy = by; else keep.push(d);
    }
    keep.suppressed = cand.filter((d) => d.suppressedBy);
    return keep;
  };
  /* Detections against the true shapes: each detection (best first) claims the unclaimed shape of its class it
     overlaps most at IoU >= 0.5; a second box on a claimed shape is a duplicate. */
  Y.match = function (dets, gts, thr = 0.5) {
    const used = new Array(gts.length).fill(-1);
    for (const d of dets.slice().sort((p, q) => q.score - p.score)) {
      let best = -1, bi = thr;
      gts.forEach((g, j) => { if (g.cls !== d.cls || used[j] >= 0) return; const v = Y.iou(d.box, g.box); if (v >= bi) { bi = v; best = j; } });
      if (best >= 0) { used[best] = 1; d.hit = best; d.iou = bi; }
      else { d.hit = -1; d.dup = gts.some((g) => g.cls === d.cls && Y.iou(d.box, g.box) >= thr); }
    }
    return { found: used.filter((u) => u > 0).length, missed: used.map((u, j) => (u < 0 ? j : -1)).filter((j) => j >= 0), dups: dets.filter((d) => d.dup).length, wrong: dets.filter((d) => d.hit < 0 && !d.dup).length };
  };

  NA.yolo = Y;
})(typeof globalThis !== 'undefined' ? (globalThis.NA = globalThis.NA || {}) : (window.NA = window.NA || {}));
