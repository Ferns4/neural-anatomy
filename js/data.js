/* data.js — seeded dataset generators: 2-D classification planes, 1-D regression curves
   and synthetic handwritten digits rendered from stroke templates. */
(function (NA) {
  'use strict';
  const { RNG, clamp } = NA;
  const TAU = Math.PI * 2;

  // ---------------------------------------------------------------- 2-D planes, domain [-1, 1]²
  function planeSet(n, gen, rng, noise) {
    const raw = new Float64Array(n * 2), y = new Float64Array(n);
    for (let i = 0; i < n; i++) {
      const p = gen(i, n, rng);
      raw[2 * i] = clamp(p[0] + rng.gauss(0, noise), -1, 1);
      raw[2 * i + 1] = clamp(p[1] + rng.gauss(0, noise), -1, 1);
      y[i] = p[2];
    }
    return { raw, y };
  }

  const PLANES = {
    blobs: {
      label: 'Two clusters', classes: 2, hint: 'Linearly separable',
      make(n, noise, rng) {
        return planeSet(n, (i, N, r) => {
          const c = i % 2;
          return c ? [r.gauss(0.38, 0.2), r.gauss(0.34, 0.2), 1] : [r.gauss(-0.4, 0.2), r.gauss(-0.32, 0.2), 0];
        }, rng, noise * 0.3);
      },
    },
    xor: {
      label: 'XOR', classes: 2, hint: 'Opposite corners share a class',
      make(n, noise, rng) {
        return planeSet(n, (i, N, r) => {
          const q = i % 4, sx = q & 1 ? 1 : -1, sy = q & 2 ? 1 : -1;
          return [sx * 0.5 + r.gauss(0, 0.13), sy * 0.5 + r.gauss(0, 0.13), sx * sy > 0 ? 1 : 0];
        }, rng, noise * 0.25);
      },
    },
    quadrants: {
      label: 'Quadrants', classes: 2, hint: 'XOR filling the whole plane',
      make(n, noise, rng) {
        return planeSet(n, (i, N, r) => {
          const pad = 0.06, sx = r.next() < 0.5 ? -1 : 1, sy = r.next() < 0.5 ? -1 : 1;
          const x = sx * (pad + r.next() * (0.95 - pad)), y = sy * (pad + r.next() * (0.95 - pad));
          return [x, y, x * y > 0 ? 1 : 0];
        }, rng, noise * 0.2);
      },
    },
    circle: {
      label: 'Ring', classes: 2, hint: 'A disc inside a ring',
      make(n, noise, rng) {
        return planeSet(n, (i, N, r) => {
          const inner = i % 2 === 0, a = r.next() * TAU;
          const rad = inner ? Math.sqrt(r.next()) * 0.42 : 0.62 + r.next() * 0.33;
          return [rad * Math.cos(a), rad * Math.sin(a), inner ? 1 : 0];
        }, rng, noise * 0.2);
      },
    },
    moons: {
      label: 'Two moons', classes: 2, hint: 'Interlocking crescents',
      make(n, noise, rng) {
        return planeSet(n, (i, N, r) => {
          const c = i % 2, t = r.next() * Math.PI;
          const x = c ? 1 - Math.cos(t) : Math.cos(t), y = c ? 0.5 - Math.sin(t) : Math.sin(t);
          return [(x - 0.5) / 1.6, (y - 0.25) / 1.6, c];
        }, rng, noise * 0.22);
      },
    },
    spiral: {
      label: 'Spiral', classes: 2, hint: 'Two interleaved arms',
      make(n, noise, rng) {
        return planeSet(n, (i, N, r) => {
          const c = i % 2, t = r.next();
          const rad = 0.06 + 0.88 * t, a = t * 1.6 * TAU + c * Math.PI;
          return [rad * Math.cos(a), rad * Math.sin(a), c];
        }, rng, noise * 0.1);
      },
    },
    pinwheel: {
      label: 'Pinwheel', classes: 3, hint: 'Three spiral arms, three classes',
      make(n, noise, rng) {
        return planeSet(n, (i, N, r) => {
          const c = i % 3, t = r.next();
          const rad = 0.08 + 0.86 * t, a = t * 0.85 * TAU + (c * TAU) / 3;
          return [rad * Math.cos(a), rad * Math.sin(a), c];
        }, rng, noise * 0.1);
      },
    },
  };

  // ---------------------------------------------------------------- 1-D regression, x ∈ [-1, 1]
  const CURVES = {
    sine: { label: 'Sine wave', f: (x) => 0.8 * Math.sin(1.6 * Math.PI * x) },
    abs: { label: 'Absolute value', f: (x) => 1.4 * Math.abs(x) - 0.6 },
    bumps: { label: 'Two bumps', f: (x) => 0.95 * Math.exp(-((x + 0.45) ** 2) / 0.02) - 0.7 * Math.exp(-((x - 0.4) ** 2) / 0.05) },
    step: { label: 'Step', f: (x) => (x > 0.1 ? 0.65 : -0.45) },
  };
  function makeCurve(id, n, noise, rng) {
    const f = CURVES[id].f, raw = new Float64Array(n), y = new Float64Array(n);
    for (let i = 0; i < n; i++) { const x = -1 + (2 * (i + rng.next())) / n; raw[i] = x; y[i] = f(x) + rng.gauss(0, noise * 0.35); }
    return { raw, y };
  }

  // ---------------------------------------------------------------- input features for planes
  const FEATURES = [
    { id: 'x1', label: 'x₁', f: (a) => a },
    { id: 'x2', label: 'x₂', f: (a, b) => b },
    { id: 'x1sq', label: 'x₁²', f: (a) => a * a },
    { id: 'x2sq', label: 'x₂²', f: (a, b) => b * b },
    { id: 'x1x2', label: 'x₁x₂', f: (a, b) => a * b },
    { id: 'sin1', label: 'sin πx₁', f: (a) => Math.sin(Math.PI * a) },
    { id: 'sin2', label: 'sin πx₂', f: (a, b) => Math.sin(Math.PI * b) },
  ];
  const FEATURE_BY_ID = Object.fromEntries(FEATURES.map((f) => [f.id, f]));

  function featurizePlane(raw, n, ids) {
    const fs = ids.map((id) => FEATURE_BY_ID[id].f), d = fs.length, X = new Float64Array(n * d);
    for (let p = 0; p < n; p++) { const a = raw[2 * p], b = raw[2 * p + 1]; for (let k = 0; k < d; k++) X[p * d + k] = fs[k](a, b); }
    return X;
  }

  // ---------------------------------------------------------------- synthetic handwritten digits
  const IMG = 14;          // digits are IMG × IMG pixels
  const PEN = 0.105;       // mean stroke width as a fraction of the frame
  const FIT = 0.74;        // the longer side of each digit fills this fraction of the frame

  function ellipse(cx, cy, rx, ry, a0, sweep, k = 18) {
    const pts = [];
    for (let i = 0; i <= k; i++) { const a = a0 + (sweep * i) / k; pts.push([cx + rx * Math.cos(a), cy + ry * Math.sin(a)]); }
    return pts;
  }
  const S = (pts) => ({ pts, smooth: true });
  const P = (pts) => ({ pts, smooth: false });
  const U = (r, a, b) => r.uniform(a, b);

  // Each digit maps to template functions; each template returns strokes in a unit box (y down).
  const TEMPLATES = {
    0: [(r) => [S(ellipse(0.5, 0.5, U(r, 0.24, 0.36), U(r, 0.4, 0.47), -Math.PI / 2 + U(r, -0.4, 0.3), TAU + U(r, -0.15, 0.35), 22))]],
    1: [
      () => [P([[0.5, 0.05], [0.5, 0.95]])],
      (r) => [P([[0.3, 0.25 + U(r, -0.05, 0.05)], [0.53, 0.05], [0.53, 0.95]])],
      () => [P([[0.32, 0.24], [0.53, 0.05], [0.53, 0.94]]), P([[0.3, 0.95], [0.76, 0.95]])],
    ],
    2: [
      (r) => [S([[0.2, 0.27], [0.33, 0.09], [0.55, 0.05], [0.75, 0.15], [0.77, 0.35], [0.6, 0.56], [0.38, 0.76], [0.18, 0.95]]), P([[0.18, 0.95], [0.84, 0.93 + U(r, -0.03, 0.03)]])],
      () => [S([[0.22, 0.24], [0.4, 0.06], [0.66, 0.07], [0.76, 0.26], [0.64, 0.5], [0.4, 0.74], [0.2, 0.93], [0.36, 0.88], [0.58, 0.9], [0.84, 0.94]])],
    ],
    3: [
      () => [S([[0.2, 0.16], [0.42, 0.05], [0.68, 0.08], [0.75, 0.25], [0.62, 0.42], [0.42, 0.48]]), S([[0.42, 0.48], [0.68, 0.53], [0.8, 0.72], [0.68, 0.92], [0.42, 0.96], [0.18, 0.86]])],
      () => [P([[0.2, 0.06], [0.78, 0.06], [0.45, 0.42]]), S([[0.45, 0.42], [0.72, 0.52], [0.8, 0.72], [0.66, 0.92], [0.4, 0.96], [0.18, 0.86]])],
    ],
    4: [
      (r) => [P([[0.6, 0.05], [0.13, 0.66], [0.88, 0.66]]), P([[0.63, 0.05 + U(r, 0, 0.25)], [0.63, 0.97]])],
      () => [P([[0.24, 0.05], [0.2, 0.6], [0.86, 0.6]]), P([[0.68, 0.08], [0.68, 0.97]])],
    ],
    5: [
      () => [P([[0.78, 0.07], [0.3, 0.07], [0.27, 0.45]]), S([[0.27, 0.45], [0.5, 0.38], [0.72, 0.48], [0.78, 0.7], [0.65, 0.9], [0.42, 0.96], [0.2, 0.88]])],
      () => [P([[0.3, 0.07], [0.26, 0.44]]), S([[0.26, 0.44], [0.52, 0.38], [0.74, 0.5], [0.76, 0.74], [0.6, 0.93], [0.36, 0.95], [0.2, 0.86]]), P([[0.3, 0.07], [0.8, 0.06]])],
    ],
    6: [
      () => [S([[0.7, 0.05], [0.48, 0.15], [0.3, 0.38], [0.24, 0.62], [0.3, 0.85], [0.5, 0.95], [0.7, 0.86], [0.74, 0.68], [0.6, 0.53], [0.4, 0.53], [0.27, 0.66]])],
    ],
    7: [
      () => [P([[0.15, 0.08], [0.85, 0.08], [0.42, 0.95]])],
      () => [P([[0.15, 0.08], [0.85, 0.08], [0.42, 0.95]]), P([[0.36, 0.52], [0.74, 0.52]])],
      () => [P([[0.15, 0.08], [0.85, 0.08]]), S([[0.85, 0.08], [0.64, 0.38], [0.5, 0.68], [0.45, 0.95]])],
    ],
    8: [
      (r) => [S(ellipse(0.5, 0.27, U(r, 0.17, 0.23), 0.21, -Math.PI / 2, TAU + 0.2)), S(ellipse(0.5, 0.72, U(r, 0.22, 0.28), 0.24, -Math.PI / 2, TAU + 0.2))],
      () => [S([[0.68, 0.12], [0.5, 0.05], [0.32, 0.13], [0.32, 0.33], [0.5, 0.47], [0.7, 0.62], [0.72, 0.84], [0.5, 0.96], [0.28, 0.84], [0.3, 0.62], [0.5, 0.47], [0.68, 0.32], [0.68, 0.12]])],
    ],
    9: [
      (r) => [S(ellipse(0.47, 0.3, 0.24, U(r, 0.18, 0.24), 0, TAU + 0.1)), P([[0.71, 0.3], [0.68 + U(r, -0.06, 0.04), 0.96]])],
      () => [S(ellipse(0.47, 0.3, 0.24, 0.22, 0, TAU + 0.1)), S([[0.71, 0.28], [0.7, 0.6], [0.62, 0.85], [0.45, 0.96]])],
    ],
  };

  // Catmull–Rom subdivision for smooth strokes.
  function smoothPts(pts, k = 5) {
    if (pts.length < 3) return pts.slice();
    const out = [];
    for (let i = 0; i < pts.length - 1; i++) {
      const p0 = pts[Math.max(0, i - 1)], p1 = pts[i], p2 = pts[i + 1], p3 = pts[Math.min(pts.length - 1, i + 2)];
      for (let s = 0; s < k; s++) {
        const t = s / k, t2 = t * t, t3 = t2 * t;
        out.push([0, 1].map((d) => 0.5 * (2 * p1[d] + (-p0[d] + p2[d]) * t + (2 * p0[d] - 5 * p1[d] + 4 * p2[d] - p3[d]) * t2 + (-p0[d] + 3 * p1[d] - 3 * p2[d] + p3[d]) * t3)));
      }
    }
    out.push(pts[pts.length - 1].slice());
    return out;
  }

  // Scale and centre strokes so the longer side of their bounding box fills `fit` of the frame.
  function normalizeStrokes(strokes, fit = FIT, dx = 0, dy = 0) {
    let x0 = Infinity, y0 = Infinity, x1 = -Infinity, y1 = -Infinity;
    for (const s of strokes) for (const p of s) { if (p[0] < x0) x0 = p[0]; if (p[0] > x1) x1 = p[0]; if (p[1] < y0) y0 = p[1]; if (p[1] > y1) y1 = p[1]; }
    if (!isFinite(x0)) return [];
    const w = x1 - x0, h = y1 - y0, s = fit / Math.max(w, h, 1e-6), cx = (x0 + x1) / 2, cy = (y0 + y1) / 2;
    return strokes.map((st) => st.map((p) => [0.5 + dx + (p[0] - cx) * s, 0.5 + dy + (p[1] - cy) * s]));
  }

  function distToSeg(px, py, ax, ay, bx, by) {
    const dx = bx - ax, dy = by - ay, L2 = dx * dx + dy * dy;
    let t = L2 > 0 ? ((px - ax) * dx + (py - ay) * dy) / L2 : 0;
    t = t < 0 ? 0 : t > 1 ? 1 : t;
    const qx = ax + t * dx - px, qy = ay + t * dy - py;
    return Math.sqrt(qx * qx + qy * qy);
  }

  // Anti-aliased thick-line rasteriser into a G×G image with values in [0, 1].
  function rasterize(strokes, G = IMG, width = PEN, out) {
    const img = out || new Float64Array(G * G);
    img.fill(0);
    const r = (width * G) / 2;
    for (const st of strokes) {
      const pts = st.length === 1 ? [st[0], st[0]] : st;
      for (let i = 0; i < pts.length - 1; i++) {
        const ax = pts[i][0] * G, ay = pts[i][1] * G, bx = pts[i + 1][0] * G, by = pts[i + 1][1] * G;
        const xa = Math.max(0, Math.floor(Math.min(ax, bx) - r - 1)), xb = Math.min(G - 1, Math.ceil(Math.max(ax, bx) + r + 1));
        const ya = Math.max(0, Math.floor(Math.min(ay, by) - r - 1)), yb = Math.min(G - 1, Math.ceil(Math.max(ay, by) + r + 1));
        for (let py = ya; py <= yb; py++) {
          for (let px = xa; px <= xb; px++) {
            const d = distToSeg(px + 0.5, py + 0.5, ax, ay, bx, by);
            const cov = r + 0.5 - d;
            if (cov > 0) { const v = cov > 1 ? 1 : cov, k = py * G + px; if (v > img[k]) img[k] = v; }
          }
        }
      }
    }
    return img;
  }

  // One random digit: strokes (normalised) plus pen width.
  function sampleDigit(digit, rng) {
    const tmpl = rng.pick(TEMPLATES[digit]);
    const raw = tmpl(rng);
    const sh = rng.gauss(0, 0.2), th = rng.gauss(0, 0.13), sx = rng.uniform(0.72, 1.18);
    const c = Math.cos(th), s = Math.sin(th);
    const strokes = raw.map((st) => {
      const jit = st.pts.map((p) => [p[0] + rng.gauss(0, 0.036), p[1] + rng.gauss(0, 0.036)]);
      const pts = st.smooth ? smoothPts(jit) : jit;
      return pts.map((p) => {
        let x = (p[0] - 0.5) * sx, y = p[1] - 0.5;
        x += sh * y;
        return [x * c - y * s, x * s + y * c];
      });
    });
    const fit = FIT * rng.uniform(0.88, 1.08);
    return { strokes: normalizeStrokes(strokes, fit, rng.gauss(0, 0.035), rng.gauss(0, 0.035)), width: PEN * rng.uniform(0.7, 1.4) };
  }

  function makeDigits(n, rng, keepStrokes) {
    const raw = new Float64Array(n * IMG * IMG), y = new Float64Array(n), strokes = keepStrokes ? [] : null;
    const tmp = new Float64Array(IMG * IMG);
    for (let i = 0; i < n; i++) {
      const d = i % 10, smp = sampleDigit(d, rng);
      rasterize(smp.strokes, IMG, smp.width, tmp);
      raw.set(tmp, i * IMG * IMG);
      y[i] = d;
      if (strokes) strokes.push(smp);
    }
    return { raw, y, strokes };
  }

  // ---------------------------------------------------------------- dataset assembly
  const DATASETS = {
    blobs: { kind: 'plane', task: 'binary', ...PLANES.blobs },
    xor: { kind: 'plane', task: 'binary', ...PLANES.xor },
    quadrants: { kind: 'plane', task: 'binary', ...PLANES.quadrants },
    circle: { kind: 'plane', task: 'binary', ...PLANES.circle },
    moons: { kind: 'plane', task: 'binary', ...PLANES.moons },
    spiral: { kind: 'plane', task: 'binary', ...PLANES.spiral },
    pinwheel: { kind: 'plane', task: 'multiclass', ...PLANES.pinwheel },
    sine: { kind: 'line', task: 'regression', label: CURVES.sine.label, hint: 'Smooth, periodic' },
    abs: { kind: 'line', task: 'regression', label: CURVES.abs.label, hint: 'Exactly two ReLUs' },
    bumps: { kind: 'line', task: 'regression', label: CURVES.bumps.label, hint: 'Localised features' },
    step: { kind: 'line', task: 'regression', label: CURVES.step.label, hint: 'A discontinuity' },
    digits: { kind: 'image', task: 'multiclass', classes: 10, label: 'Handwritten digits', hint: '14×14 synthetic strokes' },
  };

  // Drawing a few thousand digits takes about 50 ms, and they depend only on their count and seed, so the last
  // set is kept: loading the preset again (or the model builder's digit task) reuses it. The arrays are only
  // ever read. The shuffle that splits them is kept too, since it continues the same random sequence.
  let digitCache = null;

  /* spec: { dataset, n, noise, trainRatio, seed, features } */
  function buildDataset(spec) {
    const def = DATASETS[spec.dataset];
    const rng = new RNG(((spec.seed || 1) * 2654435761) >>> 0);
    const n = spec.n;
    let made, rawDim, classes, idx;
    if (def.kind === 'plane') { made = def.make(n, spec.noise, rng); rawDim = 2; classes = def.classes; }
    else if (def.kind === 'line') { made = makeCurve(spec.dataset, n, spec.noise, rng); rawDim = 1; classes = 0; }
    else {
      const key = n + ':' + (spec.seed || 1);
      if (!digitCache || digitCache.key !== key) { const m = makeDigits(n, rng, true), order = NA.range(n); rng.shuffle(order); digitCache = { key, made: m, order }; }
      made = digitCache.made; idx = digitCache.order.slice(); rawDim = IMG * IMG; classes = 10;
    }
    // Three-way split: training data the model learns from, validation data tracked live while it
    // trains, and a held-out test set that is only touched when you run the final test.
    if (!idx) { idx = NA.range(n); rng.shuffle(idx); }
    const nTrain = Math.min(n - 2, Math.max(2, Math.round(n * spec.trainRatio)));
    const nVal = Math.ceil((n - nTrain) / 2);
    const ds = {
      id: spec.dataset, def, kind: def.kind, task: def.task, classes, n, rawDim,
      raw: made.raw, y: made.y, strokes: made.strokes || null,
      trainIdx: Int32Array.from(idx.slice(0, nTrain)),
      valIdx: Int32Array.from(idx.slice(nTrain, nTrain + nVal)),
      testIdx: Int32Array.from(idx.slice(nTrain + nVal)),
      imgSize: def.kind === 'image' ? IMG : 0,
    };
    ds.split = new Uint8Array(n); // 0 train, 1 validation, 2 test
    for (const k of ds.valIdx) ds.split[k] = 1;
    for (const k of ds.testIdx) ds.split[k] = 2;
    setFeatures(ds, spec.features);
    return ds;
  }

  function setFeatures(ds, ids) {
    if (ds.kind === 'plane') {
      ds.featureIds = ids && ids.length ? ids.slice() : ['x1', 'x2'];
      ds.X = featurizePlane(ds.raw, ds.n, ds.featureIds);
      ds.dim = ds.featureIds.length;
      ds.featureLabels = ds.featureIds.map((id) => FEATURE_BY_ID[id].label);
    } else if (ds.kind === 'line') {
      ds.featureIds = ['x']; ds.X = ds.raw; ds.dim = 1; ds.featureLabels = ['x'];
    } else {
      ds.featureIds = null; ds.X = ds.raw; ds.dim = ds.rawDim;
      ds.featureLabels = NA.range(ds.dim).map((i) => `px(${i % IMG},${Math.floor(i / IMG)})`);
    }
  }

  // Map raw plane points (x1,x2) to the dataset's current feature space.
  function featurizePoints(ds, rawPts, n) {
    if (ds.kind === 'plane') return featurizePlane(rawPts, n, ds.featureIds);
    return rawPts;
  }

  NA.data = {
    DATASETS, PLANES, CURVES, FEATURES, FEATURE_BY_ID, IMG, PEN,
    buildDataset, setFeatures, featurizePlane, featurizePoints,
    rasterize, normalizeStrokes, sampleDigit,
  };
})(typeof globalThis !== 'undefined' ? (globalThis.NA = globalThis.NA || {}) : (window.NA = window.NA || {}));
