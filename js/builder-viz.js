/* builder-viz.js — what the model builder shows of a model's results, for each kind of problem: the regions it
   gives each class in the plane, the curve it fits, the digits it reads (with a confusion matrix), and for
   sequences the answers it writes one token at a time and where its attention looks. Plus a small preview of
   each problem's data for the picker. Everything is drawn on a canvas from the live model. */
(function (NA) {
  'use strict';
  const { clamp, fmt, fmtPct, theme } = NA;
  const BT = NA.builderTasks;
  const V = {};
  const font = (w, px, mono) => `${w} ${px}px ${mono ? getComputedStyle(document.body).getPropertyValue('--font-mono') || 'monospace' : getComputedStyle(document.body).getPropertyValue('--font-ui') || 'sans-serif'}`;
  const text = (ctx, s, x, y, o = {}) => { ctx.font = font(o.w || 500, o.size || 12, o.mono); ctx.fillStyle = o.color || theme.c('ink'); ctx.textAlign = o.align || 'left'; ctx.textBaseline = o.base || 'middle'; ctx.fillText(s, x, y); };
  const rgb = (name) => theme.tokens[name] || [128, 128, 128];
  // a caption that fits its width: shortened with an ellipsis rather than cut mid-word by the edge
  const fit = (ctx, s, w, size = 10.5, weight = 600) => { ctx.font = font(weight, size); if (ctx.measureText(s).width <= w) return s; let t = s; while (t.length > 1 && ctx.measureText(t + '…').width > w) t = t.slice(0, -1); return t.trimEnd() + '…'; };
  V.classColour = (data, k) => BT.CLASS_COLOURS[(data.colourOf ? data.colourOf[k] : k) % 4];
  const softmax = (o, base, K) => { let m = -Infinity; for (let k = 0; k < K; k++) m = Math.max(m, o[base + k]); const e = []; let z = 0; for (let k = 0; k < K; k++) { e.push(Math.exp(o[base + k] - m)); z += e[k]; } return e.map((v) => v / z); };

  // ------------------------------------------------------------------ the plane: regions and points
  // the square [−1, 1]² inside r, as large as fits
  const square = (r) => { const s = Math.min(r.w, r.h); return { x: r.x + (r.w - s) / 2, y: r.y + (r.h - s) / 2, s }; };
  // A G × G image of the model's class probabilities, each pixel the classes' colours mixed by probability,
  // brighter where it is sure. Cached until the model changes.
  function regionImage(L, G) {
    const key = `${L.vizVersion}:${G}:${theme.dark ? 1 : 0}`;
    if (L._region && L._region.key === key) return L._region.c;
    const m = L.model, d = L.data, K = d.classes, X = new Float32Array(G * G * 2);
    for (let i = 0; i < G; i++) for (let j = 0; j < G; j++) { X[(i * G + j) * 2] = -1 + ((j + 0.5) / G) * 2; X[(i * G + j) * 2 + 1] = 1 - ((i + 0.5) / G) * 2; }
    const o = m.predict(X, G * G), bg = rgb('panel'), cols = NA.range(K).map((k) => rgb(V.classColour(d, k)));
    const c = document.createElement('canvas'); c.width = c.height = G;
    const g = c.getContext('2d'), img = g.createImageData(G, G);
    for (let p = 0; p < G * G; p++) {
      const pr = softmax(o, p * K, K), sure = Math.max(...pr), a = 0.22 + 0.5 * clamp((sure - 1 / K) / (1 - 1 / K), 0, 1);
      for (let ch = 0; ch < 3; ch++) { let v = 0; for (let k = 0; k < K; k++) v += pr[k] * cols[k][ch]; img.data[p * 4 + ch] = bg[ch] + (v - bg[ch]) * a; }
      img.data[p * 4 + 3] = 255;
    }
    g.putImageData(img, 0, 0);
    L._region = { key, c };
    return c;
  }
  function drawPlane(ctx, r, L, o = {}) {
    const d = L.data, q = square(r), X = (v) => q.x + ((v + 1) / 2) * q.s, Y = (v) => q.y + ((1 - v) / 2) * q.s;
    ctx.save(); ctx.beginPath(); NA.rr(ctx, q.x, q.y, q.s, q.s, 8); ctx.clip();
    if (L.model && !o.dataOnly) { ctx.imageSmoothingEnabled = true; ctx.drawImage(regionImage(L, q.s > 260 ? 72 : 48), q.x, q.y, q.s, q.s); }
    else { ctx.fillStyle = theme.c('panel-2'); ctx.fillRect(q.x, q.y, q.s, q.s); }
    ctx.strokeStyle = theme.c('line', 0.6); ctx.lineWidth = 1;
    for (let k = -0.5; k <= 0.5; k += 0.5) { ctx.beginPath(); ctx.moveTo(X(k), q.y); ctx.lineTo(X(k), q.y + q.s); ctx.moveTo(q.x, Y(k)); ctx.lineTo(q.x + q.s, Y(k)); ctx.stroke(); }
    const rad = Math.max(2.2, q.s / 120);
    const dots = (Xs, Ys, n, hollow) => {
      for (let i = 0; i < n; i++) {
        const col = theme.c(V.classColour(d, Ys[i])), x = X(Xs[2 * i]), y = Y(Xs[2 * i + 1]);
        ctx.beginPath(); ctx.arc(x, y, rad, 0, Math.PI * 2);
        if (hollow) { ctx.strokeStyle = col; ctx.lineWidth = 1.6; ctx.stroke(); }
        else { ctx.fillStyle = col; ctx.fill(); ctx.strokeStyle = theme.c('panel', 0.9); ctx.lineWidth = 1; ctx.stroke(); }
      }
    };
    if (d && !d.error) { dots(d.Xtr, d.Ytr, d.nTr, false); dots(d.Xva, d.Yva, d.nVa, true); }
    ctx.restore();
    ctx.strokeStyle = theme.c('line-2'); ctx.lineWidth = 1; ctx.beginPath(); NA.rr(ctx, q.x + 0.5, q.y + 0.5, q.s - 1, q.s - 1, 8); ctx.stroke();
    return q;
  }

  // ------------------------------------------------------------------ the curve
  function drawCurve(ctx, r, L) {
    // values are stored scaled to about ±1 for training; the plot shows them in the formula's own units
    const d = L.data, pad = 34, q = { x: r.x + pad, y: r.y + 8, w: r.w - pad - 8, h: r.h - 30 }, u = 1 / (d.scale || 1);
    const ys = [...d.Ytr, ...d.Yva].map((v) => v * u);
    let lo = Math.min(...ys), hi = Math.max(...ys);
    const span = hi - lo || 1; lo -= span * 0.12; hi += span * 0.12;
    const X = (v) => q.x + ((v + 1) / 2) * q.w, Y = (v) => q.y + q.h - ((v - lo) / (hi - lo)) * q.h;
    ctx.fillStyle = theme.c('panel-2'); ctx.beginPath(); NA.rr(ctx, q.x, q.y, q.w, q.h, 8); ctx.fill();
    ctx.strokeStyle = theme.c('line'); ctx.lineWidth = 1;
    const ticks = NA.niceTicks(lo, hi, 4), dec = ticks.length > 1 ? clamp(-Math.floor(Math.log10(Math.abs(ticks[1] - ticks[0]) + 1e-12)), 0, 3) : 1;
    for (const t of ticks) { const y = Math.round(Y(t)) + 0.5; if (y < q.y || y > q.y + q.h) continue; ctx.beginPath(); ctx.moveTo(q.x, y); ctx.lineTo(q.x + q.w, y); ctx.stroke(); text(ctx, t.toFixed(dec).replace('-', NA.MINUS), q.x - 6, y, { size: 10, align: 'right', color: theme.c('muted'), mono: true }); }
    for (const t of [-1, -0.5, 0, 0.5, 1]) text(ctx, String(t), X(t), q.y + q.h + 11, { size: 10, align: 'center', color: theme.c('muted'), mono: true });
    if (d.truth) { ctx.strokeStyle = theme.c('ink-2', 0.55); ctx.setLineDash([4, 4]); ctx.lineWidth = 1.5; ctx.beginPath(); for (let i = 0; i <= 200; i++) { const x = -1 + i / 100, y = d.truth(x) * u; i ? ctx.lineTo(X(x), Y(y)) : ctx.moveTo(X(x), Y(y)); } ctx.stroke(); ctx.setLineDash([]); }
    const dot = (x, y, hollow) => { ctx.beginPath(); ctx.arc(X(x), Y(y), 2.6, 0, Math.PI * 2); if (hollow) { ctx.strokeStyle = theme.c('s2'); ctx.lineWidth = 1.4; ctx.stroke(); } else { ctx.fillStyle = theme.c('s1', 0.8); ctx.fill(); } };
    for (let i = 0; i < d.nTr; i++) dot(d.Xtr[i], d.Ytr[i] * u, false);
    for (let i = 0; i < d.nVa; i++) dot(d.Xva[i], d.Yva[i] * u, true);
    if (L.model) {
      const n = 240, xs = new Float32Array(n);
      for (let i = 0; i < n; i++) xs[i] = -1 + (2 * i) / (n - 1);
      const p = L.model.predict(xs, n);
      ctx.save(); ctx.beginPath(); ctx.rect(q.x, q.y, q.w, q.h); ctx.clip();
      ctx.strokeStyle = theme.c('ink'); ctx.lineWidth = 2.4; ctx.lineJoin = 'round';
      ctx.beginPath(); for (let i = 0; i < n; i++) { const x = X(xs[i]), y = Y(p[i] * u); i ? ctx.lineTo(x, y) : ctx.moveTo(x, y); } ctx.stroke();
      ctx.restore();
    }
    return q;
  }

  // ------------------------------------------------------------------ digits
  function drawDigits(ctx, r, L) {
    const d = L.data, n = Math.min(24, d.nVa), P = 14 * 14, cols = r.w > r.h * 1.3 ? 8 : 6, rows = Math.ceil(n / cols);
    // the confusion matrix beside the digits when there is width to spare, below them when there is height
    const want = L.model && L.confusion, side = want && r.w > 520, cm = side ? { w: Math.min(r.h, r.w * 0.38) } : null, gw = cm ? r.w - cm.w - 18 : r.w;
    const cell = Math.min(gw / cols, (r.h - 4) / rows, want && !side ? 64 : Infinity), img = cell * 0.74;
    const below = want && !side && r.h - rows * cell > 150 ? { x: r.x, y: r.y + rows * cell + 12, w: Math.min(r.w, r.h - rows * cell - 12) } : null;
    const preds = L.model ? L.model.predict(d.Xva.subarray(0, n * P), n) : null;
    const c = document.createElement('canvas'); c.width = c.height = 14;
    const g = c.getContext('2d'), id = g.createImageData(14, 14), ink = rgb('ink'), bg = rgb('panel-2');
    for (let i = 0; i < n; i++) {
      for (let p = 0; p < P; p++) { const v = clamp(d.Xva[i * P + p], 0, 1); for (let ch = 0; ch < 3; ch++) id.data[p * 4 + ch] = bg[ch] + (ink[ch] - bg[ch]) * v; id.data[p * 4 + 3] = 255; }
      g.putImageData(id, 0, 0);
      const x = r.x + (i % cols) * cell, y = r.y + Math.floor(i / cols) * cell;
      ctx.imageSmoothingEnabled = true; ctx.drawImage(c, x + 2, y + 2, img, img);
      if (preds) {
        let b = 0; for (let k = 1; k < 10; k++) if (preds[i * 10 + k] > preds[i * 10 + b]) b = k;
        const ok = b === d.Yva[i];
        ctx.strokeStyle = theme.c(ok ? 'good' : 'critical', 0.9); ctx.lineWidth = 1.6; ctx.strokeRect(x + 1.5, y + 1.5, img + 1, img + 1);
        text(ctx, String(b), x + img + 6, y + img * 0.5, { size: clamp(cell * 0.3, 10, 16), w: 700, color: theme.c(ok ? 'good' : 'critical'), mono: true });
      }
    }
    if (cm || below) {
      const top = cm ? r.y : below.y, bx = cm ? r.x + r.w - cm.w : below.x, bw = cm ? cm.w : below.w;
      const q = { x: bx + 14, y: top + 30, w: bw - 14, h: bw - 30 }, cs = Math.min(q.w, q.h) / 10, M = L.confusion;
      text(ctx, 'true digit (rows) × answer (columns)', bx, top + 4, { size: 10.5, color: theme.c('muted'), w: 600 });
      for (let k = 0; k < 10; k++) {
        text(ctx, String(k), q.x + (k + 0.5) * cs, top + 22, { size: 9.5, align: 'center', color: theme.c('muted'), mono: true });
        text(ctx, String(k), bx + 6, q.y + (k + 0.5) * cs, { size: 9.5, align: 'center', color: theme.c('muted'), mono: true });
      }
      let mx = 1; for (let i = 0; i < 10; i++) for (let j = 0; j < 10; j++) if (i !== j) mx = Math.max(mx, M[i][j]);
      for (let i = 0; i < 10; i++) for (let j = 0; j < 10; j++) {
        const v = M[i][j], tot = M[i].reduce((a, b) => a + b, 0) || 1;
        ctx.fillStyle = i === j ? theme.c('good', 0.12 + 0.75 * (v / tot)) : v ? theme.c('critical', 0.15 + 0.7 * (v / mx)) : theme.c('panel-2');
        ctx.fillRect(q.x + j * cs + 0.5, q.y + i * cs + 0.5, cs - 1, cs - 1);
        if (cs > 16 && v) text(ctx, String(v), q.x + j * cs + cs / 2, q.y + i * cs + cs / 2, { size: 9, align: 'center', color: theme.c('ink'), mono: true });
      }
    }
  }

  // ------------------------------------------------------------------ sequences
  const ATTN_TYPES = new Set(['attention', 'transformer']);
  V.attnLayers = (model) => (model ? model.layers.filter((l) => ATTN_TYPES.has(l.type) && l.attn) : []);
  // one attention map (example 0 of the last forward pass) of layer li, head h
  /* One attention map (example 0 of the last forward pass) of a layer and head: row i is what position i looks at.
     Words get a gutter on the left and run upwards above their columns. */
  function drawAttention(ctx, q, layer, h, T, labels) {
    const A = layer.attn, wordy = labels.some((l) => l.length > 1), gl = wordy ? 40 : 14, gt = wordy ? 38 : 14;
    const cs = Math.max(2, Math.min((q.w - gl) / T, (q.h - gt) / T)), x0 = q.x + gl, y0 = q.y + gt, fs = clamp(cs * 0.62, 7, 12);
    for (let i = 0; i < T; i++) for (let j = 0; j < T; j++) {
      const v = A[(h * T + i) * T + j];
      ctx.fillStyle = theme.seqColor(Math.sqrt(clamp(v, 0, 1)));
      ctx.fillRect(x0 + j * cs, y0 + i * cs, cs - 0.6, cs - 0.6);
    }
    for (let t = 0; t < T; t++) {
      text(ctx, labels[t], x0 - 4, y0 + (t + 0.5) * cs, { size: fs, align: 'right', color: theme.c('muted'), mono: true });
      if (!wordy) { text(ctx, labels[t], x0 + (t + 0.5) * cs, y0 - 7, { size: fs, align: 'center', color: theme.c('muted'), mono: true }); continue; }
      ctx.save(); ctx.translate(x0 + (t + 0.5) * cs, y0 - 4); ctx.rotate(-Math.PI / 2);
      text(ctx, labels[t], 0, 0, { size: fs, align: 'left', color: theme.c('muted'), mono: true });
      ctx.restore();
    }
  }
  function drawAlgo(ctx, r, L) {
    // narrow and tall: fewer examples, so the attention map fits under them
    const d = L.data, Lq = d.len, S = L.seqView || (L.seqView = {}), n = Math.min(r.w > 560 || !L.model ? 6 : clamp(Math.floor((r.h - 230) / 38), 3, 6), d.nVa);
    const prompts = new Int32Array(n * Lq);
    for (let i = 0; i < n; i++) for (let t = 0; t < Lq; t++) prompts[i * Lq + t] = d.Xva[i * 2 * Lq + t];
    const gen = L.model ? BT.generateAlgo(L.model, prompts, n, Lq) : null;
    const attnW = r.w > 560 ? Math.min(r.h, r.w * 0.42) : 0, tw = r.w - attnW - (attnW ? 16 : 0), rowH = Math.min(attnW ? 46 : 38, (r.h - 30) / n);
    const listH = 24 + n * rowH, under = !attnW && r.h - listH > 140 ? Math.min(r.w, r.h - listH - 22) : 0;
    const cw = Math.min(26, tw / (Lq * 3 + 3));
    text(ctx, 'input', r.x, r.y + 8, { size: 10.5, color: theme.c('muted'), w: 600 });
    text(ctx, fit(ctx, 'right answer', cw * (Lq + 1)), r.x + cw * (Lq + 1.2), r.y + 8, { size: 10.5, color: theme.c('muted'), w: 600 });
    text(ctx, fit(ctx, L.model ? 'the model writes' : 'train to see what it writes', Math.max(40, tw - cw * (2 * Lq + 2.4))), r.x + cw * (2 * Lq + 2.4), r.y + 8, { size: 10.5, color: theme.c('muted'), w: 600 });
    for (let i = 0; i < n; i++) {
      const y = r.y + 24 + i * rowH, xs = Array.from(prompts.subarray(i * Lq, (i + 1) * Lq)), ans = BT.algoAnswer(d.task, xs);
      const tile = (x, s, col, bgc) => { ctx.fillStyle = bgc || theme.c('panel-2'); NA.rr(ctx, x, y, cw - 3, rowH - 8, 5); ctx.fill(); text(ctx, s, x + (cw - 3) / 2, y + (rowH - 8) / 2, { size: clamp(cw * 0.55, 10, 15), align: 'center', mono: true, w: 650, color: col || theme.c('ink') }); };
      xs.forEach((v, k) => tile(r.x + k * cw, String(v)));
      ans.forEach((v, k) => tile(r.x + cw * (Lq + 1.2) + k * cw, String(v), theme.c('ink-2')));
      if (gen) ans.forEach((v, k) => { const gv = gen[i * Lq + k], ok = gv === v; tile(r.x + cw * (2 * Lq + 2.4) + k * cw, gv === 10 ? '→' : String(gv), theme.c(ok ? 'good' : 'critical'), theme.c(ok ? 'good' : 'critical', 0.12)); });
    }
    if ((attnW || under) && L.model) {
      const layers = V.attnLayers(L.model), top = attnW ? r.y : r.y + listH + 10, left = attnW ? r.x + r.w - attnW : r.x, side = attnW || under;
      if (layers.length) {
        L.model.predict(d.Xva.subarray(0, 2 * Lq), 1); // fill the attention maps for example 0
        const li = clamp(S.layer || 0, 0, layers.length - 1), lay = layers[li], h = clamp(S.head || 0, 0, lay.heads - 1);
        const q = { x: left, y: top + 18, w: side, h: side - 18 };
        text(ctx, fit(ctx, `attention of the first example: layer ${li + 1} of ${layers.length}, head ${h + 1} (each row looks at the columns)`, side), q.x, top + 8, { size: 10.5, color: theme.c('muted'), w: 600 });
        const labels = Array.from(d.Xva.subarray(0, 2 * Lq)).map((v) => d.vocab[v]);
        drawAttention(ctx, q, lay, h, 2 * Lq, labels);
      } else text(ctx, 'No attention layer: no position can look at another.', left, top + 20, { size: 11.5, color: theme.c('muted') });
    }
    return gen;
  }
  function drawText(ctx, r, L) {
    const d = L.data, S = L.seqView || (L.seqView = {}), T = d.input.len, Vn = d.vocab.length;
    // before the first run there is no model yet: show the prompt alone
    const prompt = S.promptIds || (d.task === 'language' ? NA.lmData.tokenize(S.prompt || 'alice likes tea . bob likes milk . alice drinks').ids : []), cont = (L.model && S.cont) || [];
    const attnW = r.w > 560 ? Math.min(r.h, r.w * 0.42) : 0, tw = r.w - attnW - (attnW ? 16 : 0);
    text(ctx, fit(ctx, L.model ? 'the prompt, then what the model writes (its likeliest next word each time)' : 'the prompt: train the model to see what it writes after it', tw), r.x, r.y + 8, { size: 10.5, color: theme.c('muted'), w: 600 });
    // words as chips that wrap
    let x = r.x, y = r.y + 30;
    const chip = (w, col, bg) => {
      ctx.font = font(600, 13); const cw = ctx.measureText(w).width + 14;
      if (x + cw > r.x + tw) { x = r.x; y += 30; }
      if (y > r.y + r.h - 14) return;
      ctx.fillStyle = bg; NA.rr(ctx, x, y - 11, cw, 22, 6); ctx.fill();
      text(ctx, w, x + 7, y, { size: 13, w: 600, color: col });
      x += cw + 5;
    };
    prompt.forEach((t) => chip(d.vocab[t], theme.c('ink'), theme.c('panel-2')));
    if (L.model) cont.forEach((t) => chip(d.vocab[t], theme.c('ink'), theme.c('pos', 0.16)));
    const under = !attnW && r.y + r.h - (y + 24) > 170 ? Math.min(r.w, r.y + r.h - (y + 24) - 10) : 0, top = attnW ? r.y : y + 24, left = attnW ? r.x + r.w - attnW : r.x, side = attnW || under;
    if (side && L.model && prompt.length) {
      const layers = V.attnLayers(L.model);
      if (layers.length) {
        const ids = prompt.concat(cont).slice(-T), X = new Int32Array(T), pad = T - ids.length;
        for (let t = 0; t < T; t++) X[t] = t < pad ? ids[0] : ids[t - pad];
        L.model.predict(X, 1);
        const li = clamp(S.layer || 0, 0, layers.length - 1), lay = layers[li], h = clamp(S.head || 0, 0, lay.heads - 1);
        const q = { x: left, y: top + 18, w: side, h: side - 18 };
        text(ctx, fit(ctx, `attention over the last ${T} words: layer ${li + 1}, head ${h + 1} (each row looks at the columns)`, side), q.x, top + 8, { size: 10.5, color: theme.c('muted'), w: 600 });
        drawAttention(ctx, q, lay, h, T, Array.from(X).map((v) => d.vocab[v].slice(0, 5)));
      }
    }
    void Vn;
  }

  // ------------------------------------------------------------------ entry points
  V.draw = function (ctx, r, L) {
    const d = L.data;
    if (!d) return;
    if (d.error) return; // the lab shows the problem's error as text above the picture
    if (d.family === 'plane') return drawPlane(ctx, r, L);
    if (d.family === 'curve') return drawCurve(ctx, r, L);
    if (d.family === 'digits') return drawDigits(ctx, r, L);
    return d.task === 'language' || d.task === 'sentences' ? drawText(ctx, r, L) : drawAlgo(ctx, r, L);
  };
  // the plane's coordinates of a point on the results canvas, for drawing your own points
  V.planeAt = (r, px, py) => { const q = square(r); return { x: ((px - q.x) / q.s) * 2 - 1, y: 1 - ((py - q.y) / q.s) * 2, inside: px >= q.x && py >= q.y && px <= q.x + q.s && py <= q.y + q.s }; };

  // a small picture of a problem's data, for the picker
  V.preview = function (ctx, W, Hh, d) {
    ctx.clearRect(0, 0, W, Hh);
    if (!d || d.error) return;
    if (d.family === 'plane') { drawPlane(ctx, { x: 0, y: 0, w: W, h: Hh }, { data: d, model: null }, { dataOnly: true }); return; }
    if (d.family === 'curve') { drawCurve(ctx, { x: 0, y: 0, w: W, h: Hh + 18 }, { data: d, model: null }); return; }
    if (d.family === 'digits') { drawDigits(ctx, { x: 0, y: 0, w: W, h: Hh }, { data: { ...d, nVa: 12 }, model: null }); return; }
    // sequences: a few examples as text
    const T = d.input.len, lines = [];
    for (let i = 0; i < Math.min(4, d.nTr); i++) {
      const xs = Array.from(d.Xtr.subarray(i * T, (i + 1) * T)), ys = Array.from(d.Ytr.subarray(i * T, (i + 1) * T));
      const last = ys.filter((v) => v >= 0).pop();
      lines.push(xs.map((v) => d.vocab[v]).join(' ') + (last !== undefined && d.task !== 'language' ? ' ' + d.vocab[last] : ''));
    }
    ctx.font = font(500, 11.5, true);
    const cut = (l) => { if (ctx.measureText(l).width <= W - 8) return l; const ws = l.split(' '); while (ws.length > 1 && ctx.measureText(ws.join(' ') + ' …').width > W - 8) ws.pop(); return ws.join(' ') + ' …'; };
    lines.forEach((l, i) => text(ctx, cut(l), 4, 12 + i * 18, { size: 11.5, mono: true, color: theme.c('ink-2') }));
  };

  NA.builderViz = V;
})(window.NA);
