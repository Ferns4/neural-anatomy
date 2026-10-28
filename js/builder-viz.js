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
  // a table can have up to 20 classes: eight colours, then they repeat
  const TABLE_COLOURS = ['s1', 's2', 's4', 's7', 's3', 's5', 's6', 's8'];
  V.classColour = (data, k) => (data.table ? TABLE_COLOURS[k % 8] : BT.CLASS_COLOURS[(data.colourOf ? data.colourOf[k] : k) % 4]);
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
        const K = d.classes || 10;
        let b = 0; for (let k = 1; k < K; k++) if (preds[i * K + k] > preds[i * K + b]) b = k;
        const ok = b === d.Yva[i];
        ctx.strokeStyle = theme.c(ok ? 'good' : 'critical', 0.9); ctx.lineWidth = 1.6; ctx.strokeRect(x + 1.5, y + 1.5, img + 1, img + 1);
        text(ctx, d.task === 'shapes' ? d.names[b].slice(0, 3) : String(b), x + img + 4, y + img * 0.5, { size: clamp(cell * (d.task === 'shapes' ? 0.2 : 0.3), 9, 16), w: 700, color: theme.c(ok ? 'good' : 'critical'), mono: true, align: 'left' });
      }
    }
    if ((cm || below) && d.task === 'shapes') { const bx = cm ? r.x + r.w - cm.w : below.x, bw = cm ? cm.w : below.w; drawConfusion(ctx, { x: bx, y: cm ? r.y : below.y, w: bw, h: bw }, L.confusion, d.names, 'true shape (rows) × answer (columns)'); return; }
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

  // ------------------------------------------------------------------ tables (your CSV)
  /* Classes: before training, the rows on the first two input columns, coloured by class; once trained, the
     confusion matrix of the validation rows. A value: before training, the value against the first input column;
     once trained, the model's answer against the true value for the validation rows (right answers on the
     diagonal), in the table's own units. */
  function drawTable(ctx, r, L) {
    const d = L.data, m = L.model, F = d.input.dim;
    if (d.kind === 'classes' && m && L.confusion) return drawConfusion(ctx, r, L.confusion, d.names, `true ${d.targetName} (rows) × the model’s answer (columns)`);
    const pts = [];
    let xl, yl, diag = false;
    if (d.kind === 'value' && m) {
      const n = Math.min(d.nVa, 600), o = m.predict(d.Xva.subarray(0, n * F), n), ys = d.stats.y;
      for (let k = 0; k < n; k++) pts.push({ x: d.Yva[k] * ys.std + ys.mean, y: o[k] * ys.std + ys.mean, c: 's2', ring: true });
      xl = `true ${d.targetName}`; yl = 'the model’s answer'; diag = true;
    } else {
      const n = Math.min(d.nTr, 800);
      for (let k = 0; k < n; k++) {
        if (d.kind === 'value') pts.push({ x: d.Xtr[k * F], y: d.Ytr[k], c: 's1' });
        else pts.push({ x: d.Xtr[k * F], y: F > 1 ? d.Xtr[k * F + 1] : 0, c: V.classColour(d, d.Ytr[k]) });
      }
      xl = d.featNames[0]; yl = d.kind === 'value' ? d.targetName : F > 1 ? d.featNames[1] : '';
    }
    scatter(ctx, r, pts, xl, yl, diag);
  }
  function scatter(ctx, r, pts, xl, yl, diag) {
    if (!pts.length) return;
    let x0 = Infinity, x1 = -Infinity, y0 = Infinity, y1 = -Infinity;
    for (const p of pts) { x0 = Math.min(x0, p.x); x1 = Math.max(x1, p.x); y0 = Math.min(y0, p.y); y1 = Math.max(y1, p.y); }
    if (diag) { x0 = y0 = Math.min(x0, y0); x1 = y1 = Math.max(x1, y1); }
    const px = (x1 - x0) * 0.06 || 1, py = (y1 - y0) * 0.06 || 1;
    x0 -= px; x1 += px; y0 -= py; y1 += py;
    const left = 46, bottom = 34, q = { x: r.x + left, y: r.y + 6, w: Math.max(40, r.w - left - 8), h: Math.max(40, r.h - bottom - 6) };
    if (diag) { const s = Math.min(q.w, q.h); q.x += (q.w - s) / 2; q.w = q.h = s; }
    const X = (v) => q.x + ((v - x0) / (x1 - x0)) * q.w, Y = (v) => q.y + q.h - ((v - y0) / (y1 - y0)) * q.h;
    ctx.fillStyle = theme.c('panel-2'); ctx.fillRect(q.x, q.y, q.w, q.h);
    ctx.strokeStyle = theme.c('line-2'); ctx.lineWidth = 1; ctx.strokeRect(q.x + 0.5, q.y + 0.5, q.w - 1, q.h - 1);
    if (diag) { ctx.strokeStyle = theme.c('ink', 0.35); ctx.setLineDash([4, 4]); ctx.beginPath(); ctx.moveTo(X(x0), Y(y0)); ctx.lineTo(X(x1), Y(y1)); ctx.stroke(); ctx.setLineDash([]); }
    const rad = clamp(Math.min(q.w, q.h) / 120, 1.8, 3.2);
    for (const p of pts) {
      ctx.beginPath(); ctx.arc(X(p.x), Y(p.y), rad, 0, Math.PI * 2);
      if (p.ring) { ctx.strokeStyle = theme.c(p.c, 0.85); ctx.lineWidth = 1.2; ctx.stroke(); } else { ctx.fillStyle = theme.c(p.c, 0.7); ctx.fill(); }
    }
    const lab = (v) => (Math.abs(v) >= 1000 ? NA.fmtInt(v) : fmt(v, Math.abs(v) >= 10 ? 0 : 2));
    text(ctx, lab(x0 + px), X(x0 + px), q.y + q.h + 10, { size: 10, align: 'center', color: theme.c('muted'), mono: true });
    text(ctx, lab(x1 - px), X(x1 - px), q.y + q.h + 10, { size: 10, align: 'center', color: theme.c('muted'), mono: true });
    text(ctx, lab(y0 + py), q.x - 6, Y(y0 + py), { size: 10, align: 'right', color: theme.c('muted'), mono: true });
    text(ctx, lab(y1 - py), q.x - 6, Y(y1 - py), { size: 10, align: 'right', color: theme.c('muted'), mono: true });
    if (xl) text(ctx, fit(ctx, xl, q.w - 80, 10.5), q.x + q.w / 2, q.y + q.h + 24, { size: 10.5, align: 'center', color: theme.c('ink-2'), w: 600 });
    if (yl) { ctx.save(); ctx.translate(r.x + 10, q.y + q.h / 2); ctx.rotate(-Math.PI / 2); text(ctx, fit(ctx, yl, q.h - 20, 10.5), 0, 0, { size: 10.5, align: 'center', color: theme.c('ink-2'), w: 600 }); ctx.restore(); }
  }
  // K × K: row i, column j counts the validation examples of class i the model put in class j
  function drawConfusion(ctx, r, M, names, title) {
    const K = M.length, labW = Math.min(110, r.w * 0.28), top = 44, cs = Math.max(4, Math.min((r.w - labW - 8) / K, (r.h - top - 4) / K));
    const q = { x: r.x + labW, y: r.y + top, s: cs * K };
    text(ctx, fit(ctx, title, r.w, 10.5), r.x, r.y + 8, { size: 10.5, color: theme.c('muted'), w: 600 });
    let mx = 1; for (let i = 0; i < K; i++) for (let j = 0; j < K; j++) if (i !== j) mx = Math.max(mx, M[i][j]);
    for (let i = 0; i < K; i++) {
      const tot = M[i].reduce((a, b) => a + b, 0) || 1;
      text(ctx, fit(ctx, names[i], labW - 8, 10, 500), q.x - 6, q.y + (i + 0.5) * cs, { size: 10, align: 'right', color: theme.c('ink-2') });
      if (cs >= 14) text(ctx, fit(ctx, names[i], Math.max(cs, 30), 9.5, 500).slice(0, Math.max(2, Math.floor(cs / 6))), q.x + (i + 0.5) * cs, q.y - 9, { size: 9.5, align: 'center', color: theme.c('muted') });
      for (let j = 0; j < K; j++) {
        const v = M[i][j];
        ctx.fillStyle = i === j ? theme.c('good', 0.12 + 0.75 * (v / tot)) : v ? theme.c('critical', 0.15 + 0.7 * (v / mx)) : theme.c('panel-2');
        ctx.fillRect(q.x + j * cs + 0.5, q.y + i * cs + 0.5, cs - 1, cs - 1);
        if (cs > 20 && v) text(ctx, String(v), q.x + j * cs + cs / 2, q.y + i * cs + cs / 2, { size: clamp(cs * 0.32, 9, 13), align: 'center', color: theme.c('ink'), mono: true });
      }
    }
  }

  // ------------------------------------------------------------------ series: forecasts
  /* The series (validation part shaded), the model's one-step-ahead predictions on the validation part, and a
     forecast that runs on its own: from where validation starts, each prediction is fed back in as the newest value,
     so small errors compound. */
  function drawSeries(ctx, r, L) {
    const d = L.data, v = d.series, W = d.window, m = L.model, chaos = d.task === 'chaos';
    // about 80 values before validation and 140 after; for chaos, room on the right for its return map
    const from = Math.max(0, d.splitAt - 80), N = Math.min(v.length, d.splitAt + (chaos ? 60 : 140)), mapW = chaos && r.w > 380 ? Math.min(r.h - 34, r.w * 0.36) : 0;
    const q = { x: r.x + 34, y: r.y + 8, w: r.w - 42 - (mapW ? mapW + 16 : 0), h: r.h - 34 };
    let lo = Infinity, hi = -Infinity;
    for (let t = from; t < N; t++) { lo = Math.min(lo, v[t]); hi = Math.max(hi, v[t]); }
    const sp = hi - lo || 1; lo -= sp * 0.15; hi += sp * 0.15;
    const X = (t) => q.x + ((t - from) / Math.max(1, N - 1 - from)) * q.w, Y = (y) => q.y + q.h - ((clamp(y, lo, hi) - lo) / (hi - lo)) * q.h;
    ctx.fillStyle = theme.c('panel-2'); ctx.beginPath(); NA.rr(ctx, q.x, q.y, q.w, q.h, 8); ctx.fill();
    ctx.fillStyle = theme.c('s2', 0.07); ctx.fillRect(X(d.splitAt), q.y, q.x + q.w - X(d.splitAt), q.h);
    text(ctx, 'training', X(from) + 6, q.y + 10, { size: 10, align: 'left', color: theme.c('muted') });
    text(ctx, 'validation: the future', X(d.splitAt) + 6, q.y + 10, { size: 10, align: 'left', color: theme.c('muted') });
    const path = (ts, ys, col, w, dash) => { ctx.strokeStyle = col; ctx.lineWidth = w; ctx.setLineDash(dash || []); ctx.beginPath(); ts.forEach((t, i) => (i ? ctx.lineTo(X(t), Y(ys[i])) : ctx.moveTo(X(t), Y(ys[i])))); ctx.stroke(); ctx.setLineDash([]); };
    const ts = []; for (let t = from; t < N; t++) ts.push(t);
    path(ts, ts.map((t) => v[t]), theme.c('s1', 0.85), 1.6);
    if (m) {
      const key = `${L.vizVersion}`;
      if (!L._series || L._series.key !== key) {
        const n = d.nVa, one = m.predict(d.Xva, n), R = Math.min(n, 140), win = new Float32Array(W), roll = [];
        for (let j = 0; j < W; j++) win[j] = v[d.splitAt - W + j];
        for (let k = 0; k < R; k++) { const y = m.predict(win, 1)[0]; roll.push(y); win.copyWithin(0, 1); win[W - 1] = y; }
        L._series = { key, one: Array.from(one), roll };
      }
      const S = L._series;
      path(S.one.map((_, i) => d.splitAt + i), S.one, theme.c('s2'), 1.8);
      path(S.roll.map((_, i) => d.splitAt + i), S.roll, theme.c('s7'), 1.8, [5, 4]);
    }
    for (const t of [from, d.splitAt, N - 1]) text(ctx, `t ${t}`, X(t), q.y + q.h + 12, { size: 10, align: t === from ? 'left' : t === N - 1 ? 'right' : 'center', color: theme.c('muted'), mono: true });
    if (!mapW) return;
    // the return map: each value against the one before. Chaos hides a simple parabola here, which the model must find.
    const b = { x: q.x + q.w + 16, y: q.y, w: mapW, h: mapW };
    ctx.fillStyle = theme.c('panel-2'); ctx.beginPath(); NA.rr(ctx, b.x, b.y, b.w, b.h, 8); ctx.fill();
    const MX = (a) => b.x + ((a + 1.1) / 2.2) * b.w, MY = (a) => b.y + b.h - ((a + 1.1) / 2.2) * b.h;
    for (let t = 1; t < v.length; t++) { ctx.fillStyle = theme.c('s1', 0.5); ctx.fillRect(MX(v[t - 1]) - 1, MY(v[t]) - 1, 2, 2); }
    if (m && L._series) L._series.one.forEach((y, i) => { const t = d.splitAt + i; ctx.fillStyle = theme.c('s2', 0.9); ctx.fillRect(MX(v[t - 1]) - 1.5, MY(y) - 1.5, 3, 3); });
    text(ctx, 'next value against the last', b.x + b.w / 2, b.y + b.h + 12, { size: 10, align: 'center', color: theme.c('muted') });
  }

  // ------------------------------------------------------------------ surfaces: truth, model, error
  function surfaceGrid(L, G) {
    const key = `${L.vizVersion}:${G}`;
    if (L._surf && L._surf.key === key) return L._surf.z;
    const X = new Float32Array(G * G * 2);
    for (let i = 0; i < G; i++) for (let j = 0; j < G; j++) { X[(i * G + j) * 2] = -1 + ((j + 0.5) / G) * 2; X[(i * G + j) * 2 + 1] = 1 - ((i + 0.5) / G) * 2; }
    const z = L.model.predict(X, G * G);
    L._surf = { key, z };
    return z;
  }
  function drawSurface(ctx, r, L) {
    // in a row when the area is wide, two over one when it is tall
    const d = L.data, G = 48, f = d.truth, panels = L.model ? 3 : 1, gap = 14, th = 18;
    const rowSide = Math.min((r.w - gap * (panels - 1)) / panels, r.h - th - 6), gridSide = panels > 1 ? Math.min((r.w - gap) / 2, (r.h - 2 * th - gap) / 2) : 0;
    const grid = gridSide > rowSide * 1.15, side = grid ? gridSide : rowSide;
    const place = (k) => (grid ? { x: r.x + (r.w - 2 * side - gap) / 2 + (k % 2) * (side + gap), y: r.y + 4 + Math.floor(k / 2) * (side + th + gap) } : { x: r.x + (r.w - (side * panels + gap * (panels - 1))) / 2 + k * (side + gap), y: r.y + 4 });
    const truth = new Float32Array(G * G);
    let top = 1e-9;
    for (let i = 0; i < G; i++) for (let j = 0; j < G; j++) { const v = f(-1 + ((j + 0.5) / G) * 2, 1 - ((i + 0.5) / G) * 2); truth[i * G + j] = v; top = Math.max(top, Math.abs(v)); }
    const model = L.model ? surfaceGrid(L, G) : null;
    const paint = (k, val, title, err) => {
      const c = document.createElement('canvas'); c.width = c.height = G;
      const g = c.getContext('2d'), img = g.createImageData(G, G);
      for (let p = 0; p < G * G; p++) {
        // colours straight from the theme's lookup tables: diverging for heights, sequential for the error
        const lut = err ? theme.gradLUT : theme.div, o = err ? Math.round(clamp(val(p) / top, 0, 1) * 255) * 3 : Math.round((clamp(val(p) / top, -1, 1) + 1) * 128) * 3;
        img.data[p * 4] = lut[o]; img.data[p * 4 + 1] = lut[o + 1]; img.data[p * 4 + 2] = lut[o + 2]; img.data[p * 4 + 3] = 255;
      }
      g.putImageData(img, 0, 0);
      const { x, y } = place(k);
      ctx.imageSmoothingEnabled = true; ctx.drawImage(c, x, y, side, side);
      ctx.strokeStyle = theme.c('line-2'); ctx.lineWidth = 1; ctx.strokeRect(x + 0.5, y + 0.5, side - 1, side - 1);
      text(ctx, fit(ctx, title, side + gap - 2, 10.5), x + side / 2, y + side + 11, { size: 10.5, align: 'center', color: theme.c('ink-2'), w: 600 });
      return { x, y };
    };
    const tp = paint(0, (p) => truth[p], 'the surface and its points');
    for (let i = 0; i < Math.min(d.nTr, 400); i++) { const px = tp.x + ((d.Xtr[2 * i] + 1) / 2) * side, py = tp.y + ((1 - d.Xtr[2 * i + 1]) / 2) * side; ctx.fillStyle = theme.c('ink', 0.5); ctx.fillRect(px - 1, py - 1, 2, 2); }
    if (model) { paint(1, (p) => model[p], 'what the model learned'); paint(2, (p) => Math.abs(model[p] - truth[p]), 'its error', true); }
  }

  // ------------------------------------------------------------------ sequences: examples and answers
  function drawSeqClass(ctx, r, L) {
    const d = L.data, T = d.input.len, K = d.classes, m = L.model, n = Math.min(d.nVa, 12);
    const side = m && L.confusion && r.w > 520, cmW = side ? Math.min(r.h, r.w * 0.36) : 0, lw = r.w - (side ? cmW + 18 : 0);
    const out = m ? m.predict(d.Xva.subarray(0, n * T), n) : null, rowH = clamp((r.h - 10) / n, 16, 26);
    for (let i = 0; i < n; i++) {
      const ids = Array.from(d.Xva.subarray(i * T, (i + 1) * T)), words = ids.map((t) => d.vocab[t]).filter((w, k) => !(d.pad === 0 && ids[k] === 0));
      const textOf = words.join(d.vocab.every((w) => w.length === 1) ? '' : ' ');
      const y = r.y + 8 + i * rowH, truth = d.Yva[i];
      let pred = null;
      if (out) { pred = 0; for (let k = 1; k < K; k++) if (out[i * K + k] > out[i * K + pred]) pred = k; }
      const ok = pred === truth;
      text(ctx, fit(ctx, textOf, lw - 150, 11.5, 500), r.x + 4, y + rowH / 2, { size: 11.5, mono: true, color: theme.c('ink') });
      text(ctx, out ? `${ok ? '✓' : '✗'} ${d.names[pred]}` : d.names[truth], r.x + lw - 4, y + rowH / 2, { size: 11, align: 'right', w: 700, color: out ? theme.c(ok ? 'good' : 'critical') : theme.c('muted') });
    }
    if (side) drawConfusion(ctx, { x: r.x + lw + 18, y: r.y, w: cmW, h: r.h }, L.confusion, d.names, 'true class (rows) × the model’s answer (columns)');
  }

  // ------------------------------------------------------------------ entry points
  V.draw = function (ctx, r, L) {
    const d = L.data;
    if (!d) return;
    if (d.error) return; // the lab shows the problem's error as text above the picture
    if (d.table) return drawTable(ctx, r, L);
    if (d.family === 'series') return drawSeries(ctx, r, L);
    if (d.family === 'surface') return drawSurface(ctx, r, L);
    if (d.family === 'seqclass') return drawSeqClass(ctx, r, L);
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
    if (d.table) { drawTable(ctx, { x: 0, y: 0, w: W, h: Hh }, { data: d, model: null }); return; }
    if (d.family === 'series') { drawSeries(ctx, { x: 0, y: 0, w: W, h: Hh + 14 }, { data: d, model: null }); return; }
    if (d.family === 'surface') { drawSurface(ctx, { x: 0, y: 0, w: W, h: Hh + 10 }, { data: d, model: null }); return; }
    if (d.family === 'seqclass') { drawSeqClass(ctx, { x: 0, y: -4, w: W, h: Hh }, { data: Object.assign({}, d, { nVa: 4 }), model: null }); return; }
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
