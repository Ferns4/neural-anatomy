/* builder-analyse.js — the model builder's Analyse tab: a closer look at a trained model, and at training itself.
   Sweep a setting (train briefly at several values of one setting, from the same start, and compare); luck (the same
   setup from several seeds, and how far their scores spread); how sure it is (calibration: is it right 80% of the time
   when it is 80% sure?); which inputs matter (permutation importance); fooling it (adversarial examples, for pictures);
   and making it smaller (pruning and quantising its weights, and what that costs). Each works on the run on show, a
   copy of its network where it changes weights. NA.banalyse.render(lab, body) draws the tab. */
(function (NA) {
  'use strict';
  const { el, clear, fmt, fmtPct, fmtInt, theme, clamp } = NA;
  const A = {};
  const BT = () => NA.builderTasks;
  const UI = () => NA.labUI;

  // ------------------------------------------------------------------ measuring a network
  const softmaxRow = (o, base, K, T = 1) => { let m = -Infinity; for (let k = 0; k < K; k++) m = Math.max(m, o[base + k] / T); const e = []; let z = 0; for (let k = 0; k < K; k++) { e.push(Math.exp(o[base + k] / T - m)); z += e[k]; } return e.map((v) => v / z); };
  // the lab's measure of a network on n validation examples (its accuracy, R² or own measure)
  function scoreOf(L, m, d, n, X = d.Xva) {
    const own = NA.bgen && NA.bgen.owns(d.family) ? NA.bgen.metric(d, m, X, d.Yva, n, (L.run ? L.run.cfg : L.setup).loss.id) : undefined;
    if (own !== undefined) return own;
    if (BT().isValue(d.family)) return BT().r2(m.predict(X, n), d.Yva, n);
    const r = m.evaluate(X, d.Yva, n, L.run ? L.run.cfg.loss : L.setup.loss);
    return r.count ? r.correct / r.count : 0;
  }
  const scoreText = (L, v) => (L.valued() ? fmt(v, 3) : fmtPct(v, 1));
  // a network like the run's, with weights w (and gradients back to its input when asked)
  function copyOf(L, w, opts = {}) {
    const run = L.run, d = L.data, m = NA.bx.build(run.cfg.blocks, d.input, d.output, Object.assign({ seed: run.cfg.train.seed, init: run.cfg.train.init }, opts));
    m.setWeights(w);
    return m;
  }
  // the run on show, trained, and a network of blocks (not a classic model; not a GAN's or contrastive learning's)
  const liveRun = (L) => (L.run && !L.stale && L.model && !L.model.classic && !L.agent && L.data ? L.run : null);
  const supervised = (L) => !!liveRun(L) && !L.gen;

  // ------------------------------------------------------------------ small pictures
  function canvas(W, H, draw) {
    const c = el('canvas', { class: 'bd-an-canvas' }), ctx = NA.fitCanvas(c, W, H);
    draw(ctx, W, H);
    return c;
  }
  const font = (px, w = 500) => `${w} ${px}px ${getComputedStyle(document.body).getPropertyValue('--font-ui') || 'sans-serif'}`;
  const text = (ctx, s, x, y, o = {}) => { ctx.font = font(o.size || 10.5, o.w || 500); ctx.fillStyle = o.color || theme.c('muted'); ctx.textAlign = o.align || 'left'; ctx.textBaseline = o.base || 'middle'; ctx.fillText(s, x, y); };
  // a line of points (x along, y from 0 up), with its labels: values as they are
  function lineChart(ctx, W, H, pts, o) {
    const l = 34, r = 8, t = 8, b = H - 22, xs = pts.map((p) => p[0]), x0 = Math.min(...xs), x1 = Math.max(...xs), lo = o.lo !== undefined ? o.lo : Math.min(...pts.map((p) => p[1])), hi = o.hi !== undefined ? o.hi : Math.max(...pts.map((p) => p[1]));
    const X = (v) => l + ((v - x0) / Math.max(1e-9, x1 - x0)) * (W - l - r), Y = (v) => b - ((v - lo) / Math.max(1e-9, hi - lo)) * (b - t);
    ctx.strokeStyle = theme.c('line'); ctx.lineWidth = 1;
    for (const v of [lo, (lo + hi) / 2, hi]) { ctx.beginPath(); ctx.moveTo(l, Y(v)); ctx.lineTo(W - r, Y(v)); ctx.stroke(); text(ctx, o.yFmt(v), l - 4, Y(v), { align: 'right', size: 9.5 }); }
    pts.forEach((p, i) => text(ctx, o.xFmt(p[0], i), X(p[0]), H - 9, { align: 'center', size: 9.5 }));
    ctx.strokeStyle = theme.c(o.tone || 's1'); ctx.lineWidth = 2; ctx.beginPath();
    pts.forEach((p, i) => (i ? ctx.lineTo(X(p[0]), Y(p[1])) : ctx.moveTo(X(p[0]), Y(p[1])))); ctx.stroke();
    for (const p of pts) { ctx.beginPath(); ctx.arc(X(p[0]), Y(p[1]), 3.2, 0, Math.PI * 2); ctx.fillStyle = theme.c(o.tone || 's1'); ctx.fill(); }
  }

  // ------------------------------------------------------------------ calibration
  /* For classes: the validation examples sorted into 10 bins by how sure the model is (its largest probability); in
     each, how often it is right. A model that is honest about how sure it is has its bars on the diagonal. ECE: the
     gap between them, weighted by how many fall in each bin. And the temperature that would make it most honest:
     its scores divided by T before the softmax (the best T on the validation examples, by their log-loss). */
  function calibration(L) {
    const d = L.data, m = L.model, K = d.output.k, n = Math.min(d.nVa, 2000), out = m.predict(d.Xva, n);
    const bins = (T) => {
      const B = Array.from({ length: 10 }, () => ({ n: 0, conf: 0, right: 0 }));
      let nll = 0;
      for (let i = 0; i < n; i++) {
        const p = softmaxRow(out, i * K, K, T), y = d.Yva[i];
        let b = 0; for (let k = 1; k < K; k++) if (p[k] > p[b]) b = k;
        const q = B[Math.min(9, Math.floor(p[b] * 10))];
        q.n++; q.conf += p[b]; if (b === y) q.right++;
        nll -= Math.log(Math.max(1e-12, p[y]));
      }
      const ece = B.reduce((s, q) => s + (q.n ? (q.n / n) * Math.abs(q.right / q.n - q.conf / q.n) : 0), 0);
      return { B, ece, nll: nll / n, conf: B.reduce((s, q) => s + q.conf, 0) / n, acc: B.reduce((s, q) => s + q.right, 0) / n };
    };
    const now = bins(1);
    let best = { T: 1, r: now };
    for (let T = 0.25; T <= 4.001; T *= 1.12) { const r = bins(T); if (r.nll < best.r.nll) best = { T, r }; }
    return { now, best, n };
  }
  function calibrationSec(L) {
    const c = calibration(L), r = c.now;
    const pic = canvas(260, 200, (ctx, W, H) => {
      const l = 30, t = 8, s = Math.min(W - l - 8, H - t - 24), X = (v) => l + v * s, Y = (v) => t + s - v * s;
      ctx.strokeStyle = theme.c('line'); ctx.strokeRect(X(0), Y(1), s, s);
      ctx.setLineDash([4, 4]); ctx.beginPath(); ctx.moveTo(X(0), Y(0)); ctx.lineTo(X(1), Y(1)); ctx.stroke(); ctx.setLineDash([]);
      r.B.forEach((q, i) => { if (!q.n) return; const acc = q.right / q.n; ctx.fillStyle = theme.c('s1', 0.75); ctx.fillRect(X(i / 10) + 1, Y(acc), s / 10 - 2, Y(0) - Y(acc)); ctx.fillStyle = theme.c('s2'); ctx.fillRect(X(q.conf / q.n) - 1, Y(q.conf / q.n) - 1, 3, 3); });
      for (const v of [0, 0.5, 1]) { text(ctx, fmtPct(v, 0), l - 4, Y(v), { align: 'right', size: 9.5 }); text(ctx, fmtPct(v, 0), X(v), H - 9, { align: 'center', size: 9.5 }); }
    });
    const over = r.conf > r.acc + 0.01, under = r.conf < r.acc - 0.01;
    return UI().section('How sure is it? (calibration)',
      UI().note(`On ${fmtInt(c.n)} validation examples: bars, how often it is right, for each tenth of how sure it is (across); the dashed line is honest. On average it is ${fmtPct(r.conf, 1)} sure and right ${fmtPct(r.acc, 1)} of the time: ${over ? 'overconfident' : under ? 'underconfident' : 'about as sure as it should be'}.`),
      pic,
      UI().note(`ECE (the gap between how sure and how right, weighted by the examples in each bar): ${fmtPct(r.ece, 1)}. ${c.best.T !== 1 && c.best.r.ece < r.ece ? `Dividing its scores by T = ${fmt(c.best.T, 2)} before the softmax (temperature scaling, fitted on these examples) brings it to ${fmtPct(c.best.r.ece, 1)} without changing a single answer.` : 'Temperature scaling would not make it more honest.'}`));
  }

  // ------------------------------------------------------------------ which inputs matter
  /* Permutation importance (Breiman, 2001): shuffle one input among the validation examples, so it no longer goes with
     the rest of its row, and see how far the score falls. A big fall: the model leans on that input. */
  function importance(L) {
    const d = L.data, m = L.model, F = d.input.dim, n = Math.min(d.nVa, 1000), base = scoreOf(L, m, d, n), rng = new NA.RNG(17), X = new Float32Array(n * F), out = [];
    const names = d.featNames || (F === 1 ? ['x'] : F === 2 ? ['x', 'y'] : NA.range(F).map((j) => `input ${j + 1}`));
    for (let j = 0; j < Math.min(F, 40); j++) {
      X.set(d.Xva.subarray(0, n * F));
      const order = rng.shuffle(NA.range(n));
      for (let i = 0; i < n; i++) X[i * F + j] = d.Xva[order[i] * F + j];
      out.push({ name: names[j], drop: base - scoreOf(L, m, d, n, X) });
    }
    return { base, out: out.sort((a, b) => b.drop - a.drop), n };
  }
  function importanceSec(L) {
    const r = importance(L), mx = Math.max(1e-9, ...r.out.map((q) => Math.abs(q.drop))), rows = r.out.slice(0, 12);
    const pic = canvas(260, 18 * rows.length + 10, (ctx, W) => {
      rows.forEach((q, i) => {
        const y = 6 + i * 18, w = ((W - 100) * Math.abs(q.drop)) / mx;
        text(ctx, q.name.length > 14 ? q.name.slice(0, 13) + '…' : q.name, 92, y + 7, { align: 'right', size: 10, color: theme.c('ink-2') });
        ctx.fillStyle = theme.c(q.drop >= 0 ? 's1' : 's2', 0.8); ctx.fillRect(96, y + 1, Math.max(1, w), 12);
        text(ctx, L.valued() ? fmt(q.drop, 3) : `${fmt(100 * q.drop, 1)} pts`, 100 + w, y + 7, { size: 9.5 });
      });
    });
    return UI().section('Which inputs matter', UI().note(`How far its validation ${L.metricName()} (${scoreText(L, r.base)} as it is) falls when one input is shuffled among ${fmtInt(r.n)} examples, so that it no longer goes with the rest of its row: the more it falls, the more the model leans on that input. An input it ignores costs nothing to shuffle.`), pic);
  }

  // ------------------------------------------------------------------ fooling it: adversarial examples
  /* FGSM (Goodfellow and others, 2014): every pixel nudged by ε in the direction that raises the loss most, the sign of
     the loss's gradient with respect to the picture. The gradient comes from a copy of the network that passes it back
     to its input. */
  function attack(L, eps, n) {
    const d = L.data, P = d.Xva.length / d.nVa, K = d.output.k, adv = copyOf(L, L.model.getWeights(), { inputGrad: true }), B = Math.min(n, 100);
    const X = new Float32Array(n * P), G = new Float32Array(n * P);
    for (let s = 0; s < n; s += B) {
      const b = Math.min(B, n - s), dx = adv.gradWith(d.Xva.subarray(s * P, (s + b) * P), b, (out, dOut) => {
        for (let i = 0; i < b; i++) { const p = softmaxRow(out, i * K, K), y = d.Yva[s + i]; for (let k = 0; k < K; k++) dOut[i * K + k] = (p[k] - (k === y ? 1 : 0)) / b; }
      });
      G.set(dx.subarray(0, b * P), s * P);
    }
    for (let i = 0; i < n * P; i++) X[i] = clamp(d.Xva[i] + eps * Math.sign(G[i]), 0, 1);
    const out = L.model.predict(X, n);
    let right = 0;
    const ans = [];
    for (let i = 0; i < n; i++) { let b = 0; for (let k = 1; k < K; k++) if (out[i * K + k] > out[i * K + b]) b = k; ans.push(b); if (b === d.Yva[i]) right++; }
    return { X, G, ans, acc: right / n };
  }
  function attackSec(L, st) {
    const d = L.data, eps = st.eps === undefined ? 0.1 : st.eps, few = 6, a = attack(L, eps, few), C = d.input.c || 1, S = d.input.h, P = C * S * S;
    const clean = L.model.predict(d.Xva, few), K = d.output.k, cls = (o, i) => { let b = 0; for (let k = 1; k < K; k++) if (o[i * K + k] > o[i * K + b]) b = k; return b; };
    const name = (k) => (d.short ? d.short[k] : d.names ? d.names[k] : String(k));
    const pic = canvas(270, few * 46 + 18, (ctx) => {
      const img = (arr, off, x, y, s, diverging) => {
        const c = document.createElement('canvas'); c.width = c.height = S;
        const g = c.getContext('2d'), id = g.createImageData(S, S), ink = theme.tokens.ink || [20, 20, 20], bg = theme.tokens['panel-2'] || [240, 240, 240], up = theme.tokens.s1 || [40, 90, 200], dn = theme.tokens.s2 || [220, 110, 40];
        for (let p = 0; p < S * S; p++) {
          const v = arr[off + p];
          const col = diverging ? (v > 0 ? up : v < 0 ? dn : bg) : null;
          for (let ch = 0; ch < 3; ch++) id.data[p * 4 + ch] = diverging ? col[ch] : bg[ch] + (ink[ch] - bg[ch]) * clamp(v, 0, 1);
          id.data[p * 4 + 3] = 255;
        }
        g.putImageData(id, 0, 0); ctx.imageSmoothingEnabled = false; ctx.drawImage(c, x, y, s, s); ctx.imageSmoothingEnabled = true;
      };
      text(ctx, 'the picture', 4, 8, { size: 9.5 }); text(ctx, `the nudge (±${eps})`, 96, 8, { size: 9.5 }); text(ctx, 'nudged', 188, 8, { size: 9.5 });
      for (let i = 0; i < few; i++) {
        const y = 16 + i * 46, ok0 = cls(clean, i) === d.Yva[i], ok1 = a.ans[i] === d.Yva[i];
        img(d.Xva, i * P, 4, y, 40); img(a.G, i * P, 96, y, 40, true); img(a.X, i * P, 188, y, 40);
        text(ctx, name(cls(clean, i)), 50, y + 20, { size: 11, w: 700, color: theme.c(ok0 ? 'good' : 'critical') });
        text(ctx, name(a.ans[i]), 234, y + 20, { size: 11, w: 700, color: theme.c(ok1 ? 'good' : 'critical') });
      }
    });
    const slider = UI().slider({ label: 'ε: how far each pixel is nudged', min: 0, max: 0.3, step: 0.02, value: eps, format: (v) => fmt(v, 2), onChange: (v) => { st.eps = v; st.curve = null; A.refresh(L); } });
    // accuracy as ε grows, on 200 pictures (worked out once per run and step)
    const key = `${L.run.n}:${L.run.step}`;
    if (!st.curve || st.curve.key !== key) st.curve = { key, pts: [0, 0.05, 0.1, 0.15, 0.2, 0.3].map((e) => [e, attack(L, e, Math.min(200, d.nVa)).acc]) };
    const chart = canvas(260, 130, (ctx, W, H) => lineChart(ctx, W, H, st.curve.pts, { lo: 0, hi: 1, xFmt: (v) => fmt(v, 2), yFmt: (v) => fmtPct(v, 0), tone: 's2' }));
    return UI().section('Fool it: adversarial examples', UI().note('Each pixel nudged by ε the way that raises its loss most (the sign of the gradient of the loss with respect to the picture: FGSM). The nudges are too small to change what you see, yet they can change what it answers.'),
      slider, pic, UI().note(`Its accuracy on 200 validation pictures as ε grows (across): ${st.curve.pts.map((p) => `${fmt(p[0], 2)} → ${fmtPct(p[1], 0)}`).join(', ')}. Training on nudged pictures (adversarial training) is the usual defence.`), chart);
  }

  // ------------------------------------------------------------------ making it smaller
  /* Pruning (the smallest weights of each layer set to 0) and quantising (each layer's weights rounded to so many bits,
     evenly spaced between −max and max). Biases, norms and the like are left as they are. The size: 4 bytes a number
     as it is; pruned, the weights kept with a 2-byte position each; quantised, b bits a weight. */
  function compress(L) {
    const d = L.data, m = L.model, w0 = m.getWeights(), n = Math.min(d.nVa, 1000), base = scoreOf(L, m, d, n), cp = copyOf(L, w0);
    const spans = []; let k = 0;
    for (const p of m.params) { spans.push({ from: k, to: k + p.w.length, decay: p.decay }); k += p.w.length; }
    const nW = spans.reduce((s, q) => s + (q.decay ? q.to - q.from : 0), 0), nAll = w0.length;
    const prune = (frac) => {
      const w = Float32Array.from(w0);
      for (const q of spans) if (q.decay) { const a = Array.from(w.subarray(q.from, q.to), Math.abs).sort((x, y) => x - y), cut = a[Math.min(a.length - 1, Math.floor(frac * a.length))]; for (let i = q.from; i < q.to; i++) if (Math.abs(w[i]) <= cut) w[i] = 0; }
      cp.setWeights(w);
      return { label: `${Math.round(frac * 100)}% pruned`, score: scoreOf(L, cp, d, n), bytes: 4 * (nAll - nW) + Math.round(nW * (1 - frac)) * 6 };
    };
    const quant = (bits) => {
      const w = Float32Array.from(w0), lv = Math.pow(2, bits - 1) - 1;
      for (const q of spans) if (q.decay) { let mx = 0; for (let i = q.from; i < q.to; i++) mx = Math.max(mx, Math.abs(w[i])); const sc = mx / lv || 1; for (let i = q.from; i < q.to; i++) w[i] = Math.round(w[i] / sc) * sc; }
      cp.setWeights(w);
      return { label: `${bits} bits`, score: scoreOf(L, cp, d, n), bytes: 4 * (nAll - nW) + Math.ceil((nW * bits) / 8) };
    };
    return { base, bytes: 4 * nAll, rows: [...[0.5, 0.8, 0.9, 0.95, 0.99].map(prune), ...[8, 4, 3, 2].map(quant)] };
  }
  const kb = (b) => (b >= 1024 ? `${fmt(b / 1024, 1)} KB` : `${fmtInt(b)} B`);
  function compressSec(L) {
    const r = compress(L);
    const rows = r.rows.map((q) => el('div', { class: 'bd-an-row' }, el('span', { text: q.label }), el('span', { class: 'mono', text: scoreText(L, q.score) }), el('span', { class: `mono ${q.score < r.base - (L.valued() ? 0.02 : 0.02) ? 'lm-warn' : ''}`, text: L.valued() ? `${q.score - r.base >= 0 ? '+' : ''}${fmt(q.score - r.base, 3)}` : `${q.score - r.base >= 0 ? '+' : ''}${fmt(100 * (q.score - r.base), 1)} pts` }), el('span', { class: 'mono', text: kb(q.bytes) })));
    return UI().section('Make it smaller: prune and quantise', UI().note(`As it is: ${scoreText(L, r.base)} with ${kb(r.bytes)} of numbers (4 bytes each). Below, a copy with its smallest weights set to 0 (pruned), or each layer's weights rounded to so few bits (quantised), on the same validation examples, and its size.`),
      el('div', { class: 'bd-an-table' }, el('div', { class: 'bd-an-row bd-an-head' }, el('span', { text: 'copy' }), el('span', { text: L.metricName() }), el('span', { text: 'change' }), el('span', { text: 'size' })), ...rows),
      UI().note('Networks usually keep their score with most of their weights gone, or with 8 bits each instead of 32: why phones can run them. Few weights to spare (a small model) make it cost more.'));
  }

  // ------------------------------------------------------------------ luck: several seeds
  /* The same setup trained from several seeds, one run after another (each the number of steps set): how much of a
     score is the setup, and how much the luck of its starting weights and the order of its examples. */
  A.runSeeds = function (L, count) {
    if (L.training) L.pause();
    const T = L.setup.train, seeds = NA.range(count).map((i) => (T.seed || 1) + i), group = `s${Date.now().toString(36)}`;
    L.seedQueue = { group, left: seeds.slice(1) };
    L.edit((s) => { s.train.seed = seeds[0]; });
    L.stale = true;
    L.guessGo = true; L.play(); L.guessGo = false;
    if (L.run) L.run.seedGroup = group;
  };
  // after a run of a seed group finished: the next seed, or the group is done
  A.afterRun = function (L, status) {
    const q = L.seedQueue;
    if (!q) return;
    if (!L.run || L.run.seedGroup !== q.group) { L.seedQueue = null; return; } // (a run outside the group: it was left)
    if (status !== 'done' || !q.left.length) { L.seedQueue = null; L.inspDirty = true; return; }
    const seed = q.left.shift();
    setTimeout(() => { L.edit((s) => { s.train.seed = seed; }); L.stale = true; L.guessGo = true; L.play(); L.guessGo = false; if (L.run) L.run.seedGroup = q.group; }, 60);
  };
  function seedsSec(L) {
    const groups = {};
    for (const r of L.runs) if (r.seedGroup && r.last && r.last.vm !== null && r.last.vm !== undefined) (groups[r.seedGroup] = groups[r.seedGroup] || []).push(r);
    const last = Object.values(groups).pop(), busy = !!L.seedQueue && L.training;
    const out = [UI().note('The same setup trained from several seeds, one run after another: the seed sets its starting weights and the order of its examples. How far their scores spread is how much of one run’s score is luck.'),
      el('div', { class: 'btn-row' }, ...[3, 5].map((k) => el('button', { class: 'btn', type: 'button', disabled: busy || L.family() === 'control' ? '' : null, onclick: () => A.runSeeds(L, k) }, `Train ${k} seeds`)))];
    if (L.seedQueue) out.push(UI().note(busy ? `Training seed ${L.setup.train.seed}; ${L.seedQueue.left.length} more to come.` : `Stopped at seed ${L.setup.train.seed}, ${L.seedQueue.left.length} more to come: press Train to go on with it, or start again above.`));
    if (last && last.length) {
      const v = last.map((r) => r.last.vm), mean = v.reduce((a, b) => a + b, 0) / v.length, sd = Math.sqrt(v.reduce((a, b) => a + (b - mean) ** 2, 0) / Math.max(1, v.length - 1));
      out.push(UI().note(`${last.length === 1 ? 'One seed so far' : `${last.length} seeds`}: ${last.map((r) => `run ${r.n} (seed ${r.cfg.train.seed}) ${scoreText(L, r.last.vm)}`).join(', ')}. ${last.length > 1 ? `Average ${scoreText(L, mean)}, spread (standard deviation) ${L.valued() ? fmt(sd, 3) : `${fmt(100 * sd, 1)} pts`}, from ${scoreText(L, Math.min(...v))} to ${scoreText(L, Math.max(...v))}. A change to the setup that moves the score by less than this may be luck.` : ''}`));
      if (last.length > 1) out.push(canvas(260, 46, (ctx, W) => {
        const lo = Math.min(...v), hi = Math.max(...v), pad = (hi - lo) * 0.2 || 0.01, X = (x) => 12 + ((x - lo + pad) / (hi - lo + 2 * pad)) * (W - 24);
        ctx.strokeStyle = theme.c('line-2'); ctx.beginPath(); ctx.moveTo(8, 22); ctx.lineTo(W - 8, 22); ctx.stroke();
        ctx.fillStyle = theme.c('s1', 0.25); ctx.fillRect(X(mean - sd), 14, X(mean + sd) - X(mean - sd), 16);
        for (const x of v) { ctx.beginPath(); ctx.arc(X(x), 22, 4, 0, Math.PI * 2); ctx.fillStyle = theme.c('s1'); ctx.fill(); }
        ctx.fillStyle = theme.c('s2'); ctx.fillRect(X(mean) - 1, 10, 2, 24);
        text(ctx, scoreText(L, lo), X(lo), 40, { align: 'center', size: 9.5 }); text(ctx, scoreText(L, hi), X(hi), 40, { align: 'center', size: 9.5 });
      }));
    }
    return UI().section('Luck: several seeds', ...out);
  }

  // ------------------------------------------------------------------ sweep a setting
  /* Train briefly at each value of one setting, from the same starting weights and examples, and compare where each
     got to on the validation examples. The learning rate has its own sweep in the Training tab. */
  const SWEEPS = {
    batch: { label: 'Batch size', values: [4, 8, 16, 32, 64, 128], apply: (s, v) => { s.train.batch = v; }, fmt: String },
    wd: { label: 'Weight decay', values: [0, 0.0001, 0.001, 0.01, 0.1], apply: (s, v) => { s.train.wd = v; }, fmt: (v) => (v ? String(v) : '0') },
    opt: { label: 'Optimizer', values: ['sgd', 'momentum', 'adam', 'adamw'], apply: (s, v) => { s.train.opt = v; }, fmt: (v) => ({ sgd: 'SGD', momentum: 'Momentum', adam: 'Adam', adamw: 'AdamW' })[v] },
    width: { label: 'Width of the first Dense block', values: [2, 4, 8, 16, 32, 64], apply: (s, v) => { const b = s.blocks.find((q) => q.type === 'dense'); if (b) { b.units = v; delete b.acts; } }, fmt: String, needs: (s) => s.blocks.some((q) => q.type === 'dense') },
  };
  A.sweep = function (L, what) {
    const d = L.dataFor(), S0 = JSON.parse(L.snapshot()), def = SWEEPS[what];
    if (d.error || !L.recheck().ok || L.isClassicSetup() || d.family === 'control' || NA.bgen && (d.family === 'gan' || d.family === 'contrast')) { NA.ui.toast.show('A sweep needs a network that fits together, learning from examples with answers.'); return; }
    const steps = Math.min(S0.train.steps || 600, 600);
    L.sweep2 = { what, i: 0, results: [], step: 0, steps, model: null, S0, d, status: 'running', rng: null };
    const tick = () => {
      const sw = L.sweep2;
      if (!sw || sw.status !== 'running') return;
      if (L.training) { sw.status = 'stopped'; A.refresh(L); return; }
      const t0 = performance.now();
      while (performance.now() - t0 < 14 && sw.i < def.values.length) {
        if (!sw.model) {
          const S = JSON.parse(JSON.stringify(sw.S0));
          def.apply(S, def.values[sw.i]);
          const chk = NA.bx.check(S.blocks, d.input, d.output);
          if (!chk.ok) { sw.results.push({ v: def.values[sw.i], score: null }); sw.i++; continue; }
          sw.S = S; sw.model = NA.bx.build(S.blocks, d.input, d.output, { seed: S.train.seed, init: S.train.init }); sw.step = 0; sw.rng = new NA.RNG(S.train.seed * 977 + 1);
          const B = Math.min(S.train.batch, d.nTr), inS = d.Xtr.length / d.nTr, yS = d.Ytr.length / d.nTr;
          sw.bX = new d.Xtr.constructor(B * inS); sw.bY = new d.Ytr.constructor(B * yS); sw.B = B; sw.inS = inS; sw.yS = yS;
        }
        const S = sw.S, T = S.train;
        for (let k = 0; k < 4 && sw.step < sw.steps; k++, sw.step++) {
          for (let b = 0; b < sw.B; b++) { const j = sw.rng.int(d.nTr); sw.bX.set(d.Xtr.subarray(j * sw.inS, (j + 1) * sw.inS), b * sw.inS); sw.bY.set(d.Ytr.subarray(j * sw.yS, (j + 1) * sw.yS), b * sw.yS); }
          const r = sw.model.step(sw.bX, sw.bY, sw.B, S.loss, { id: T.opt, lr: T.lr, weightDecay: T.wd, clip: T.clip });
          if (!Number.isFinite(r.loss)) { sw.step = sw.steps; sw.diverged = true; }
        }
        if (sw.step >= sw.steps) {
          const fake = { run: { cfg: S }, setup: S, valued: () => L.valued() };
          sw.results.push({ v: def.values[sw.i], score: sw.diverged ? null : scoreOf(fake, sw.model, d, Math.min(d.nVa, 1000)) });
          sw.model = null; sw.diverged = false; sw.i++;
        }
      }
      if (sw.i >= def.values.length) sw.status = 'done';
      A.refresh(L);
      if (sw.status === 'running') setTimeout(tick, 0);
    };
    A.refresh(L);
    setTimeout(tick, 0);
  };
  function sweepSec(L, st) {
    const S = L.setup, opts = Object.keys(SWEEPS).filter((k) => !SWEEPS[k].needs || SWEEPS[k].needs(S)), what = opts.includes(st.what) ? st.what : opts[0], sw = L.sweep2 && L.sweep2.what === what ? L.sweep2 : null, def = SWEEPS[what];
    const out = [UI().field('Setting', UI().select(opts.map((k) => ({ value: k, label: SWEEPS[k].label })), what, (v) => { st.what = v; A.refresh(L); }, { 'aria-label': 'Setting to sweep' })),
      el('div', { class: 'btn-row' }, el('button', { class: 'btn', type: 'button', disabled: sw && sw.status === 'running' ? '' : null, onclick: () => A.sweep(L, what) }, sw && sw.status === 'running' ? `Training ${def.fmt(def.values[sw.i] !== undefined ? def.values[sw.i] : def.values[def.values.length - 1])}…` : 'Sweep it'))];
    if (sw && sw.results.length) {
      const pts = sw.results.map((q, i) => [i, q.score]).filter((p) => p[1] !== null);
      if (pts.length > 1) out.push(canvas(260, 140, (ctx, W, H) => lineChart(ctx, W, H, pts, { xFmt: (v) => def.fmt(def.values[v]), yFmt: (v) => scoreText(L, v), tone: 's7' })));
      const best = sw.results.filter((q) => q.score !== null).sort((a, b) => b.score - a.score)[0];
      out.push(UI().note(`${sw.results.map((q) => `${def.fmt(q.v)}: ${q.score === null ? 'blew up or did not fit' : scoreText(L, q.score)}`).join(' · ')}${sw.status === 'done' && best ? `. Best here: ${def.fmt(best.v)}, after ${fmtInt(sw.steps)} steps each from the same start.` : ''}`));
    } else out.push(UI().note(`Trains a fresh copy of your model for ${fmtInt(Math.min(S.train.steps || 600, 600))} steps at each value, from the same starting weights and the same examples, and compares their validation ${L.metricName()}. Everything else stays as it is.`));
    return UI().section('Sweep a setting', ...out);
  }

  // ------------------------------------------------------------------ guess first
  /* Predict before you run: with Guess first on, pressing Train for a new run first asks what score it will reach; the
     run keeps the guess, and when it ends the guess is set beside what it got. The Runs tab says how close the guesses
     are getting. */
  A.guessOn = () => !!NA.store.get('builder.guess', false);
  A.askGuess = function (L, go) {
    const val = L.valued(), ctl = L.family() === 'control', last = L.runs.filter((r) => r.cfg.problem === L.setup.problem && r.last && r.last.vm !== null && r.last.vm !== undefined).pop();
    const f = (x) => (val ? fmt(x, 2) : fmtPct(x, 0)), nNext = (L.runs.length ? L.runs[L.runs.length - 1].n : 0) + 1;
    let v = last ? clamp(last.last.vm, 0, 1) : 0.5;
    const out = el('span', { class: 'mono', text: f(v) }), range = el('input', { type: 'range', min: '0', max: '1', step: '0.01', value: String(v), 'aria-label': 'Your guess' });
    range.addEventListener('input', () => { v = +range.value; out.textContent = f(v); });
    const close = () => dlg.remove();
    const dlg = el('div', { class: 'modal', role: 'dialog', 'aria-modal': 'true', 'aria-labelledby': 'guessTitle' },
      el('div', { class: 'sheet bd-guess-sheet' },
        el('div', { class: 'sheet-head' }, el('div', null, el('h2', { class: 'sheet-title', id: 'guessTitle', text: 'Guess first' }),
          el('p', { class: 'sheet-sub', text: `What ${ctl ? 'score' : `validation ${L.metricName()}`} will run ${nNext} reach${L.setup.train.steps ? ` in ${fmtInt(L.setup.train.steps)} ${ctl && NA.brl.isEvo(L.ps().method) ? 'generations' : 'steps'}` : ''}?${last ? ` Run ${last.n} reached ${f(last.last.vm)}.` : ''}` })),
          el('button', { class: 'icon-btn', type: 'button', 'aria-label': 'Close', onclick: close }, '×')),
        el('div', { class: 'bd-guess-row' }, range, out),
        el('div', { class: 'btn-row' }, el('button', { class: 'btn btn-primary', type: 'button', onclick: () => { close(); go(v); } }, 'Train'), el('button', { class: 'btn btn-quiet', type: 'button', onclick: () => { close(); go(undefined); } }, 'Train without guessing'))));
    dlg.addEventListener('keydown', (e) => { if (e.key === 'Escape') { e.stopPropagation(); close(); } });
    dlg.addEventListener('click', (e) => { if (e.target === dlg) close(); });
    document.body.append(dlg);
    range.focus();
  };
  // a run's guess beside what it got, for the toast at its end
  A.guessText = (L, run) => {
    if (run.guess === undefined || !run.last || run.last.vm === null || run.last.vm === undefined) return '';
    const val = BT().isValue(run.family), dv = run.last.vm - run.guess, f = (x) => (val ? fmt(x, 2) : fmtPct(x, 0));
    return ` You guessed ${f(run.guess)}: ${Math.abs(dv) < 0.03 ? 'spot on' : `${val ? fmt(Math.abs(dv), 2) : `${Math.round(100 * Math.abs(dv))} points`} too ${dv > 0 ? 'low' : 'high'}`}.`;
  };
  A.renderGuess = function (L, box) {
    const on = A.guessOn(), gs = L.runs.filter((r) => r.guess !== undefined && r.done && r.last && r.last.vm !== null && r.last.vm !== undefined), val = L.valued();
    const offs = gs.map((r) => Math.abs(r.last.vm - r.guess)), mean = offs.length ? offs.reduce((a, b) => a + b, 0) / offs.length : 0, f = (x) => (val ? fmt(x, 2) : `${Math.round(100 * x)} points`);
    clear(box).append(UI().field('Guess first', UI().seg([{ value: 'off', label: 'Off' }, { value: 'on', label: 'On' }], on ? 'on' : 'off', (v) => { NA.store.set('builder.guess', v === 'on'); A.renderGuess(L, box); }, 'Guess first')),
      UI().note(on ? 'Each press of Train for a new run first asks what score it will reach. Guessing, then seeing, is how a feel for training grows.' : 'Turn it on, and each new run first asks what score you expect it to reach; at its end it says how close you were.'),
      gs.length ? UI().note(`Your ${gs.length} guess${gs.length === 1 ? '' : 'es'} so far: off by ${f(mean)} on average${gs.length > 2 ? `; the last three by ${gs.slice(-3).map((r) => f(Math.abs(r.last.vm - r.guess))).join(', ')}` : ''}.`) : null);
  };

  // ------------------------------------------------------------------ the tab
  A.refresh = function (L) { L._anKey = null; if (L.tabId === 'analyse') L.updateTab(); };
  A.render = function (L, body) {
    const run = liveRun(L), sup = supervised(L), d = run ? L.data : L.dataFor(), st = L.anState || (L.anState = {});
    const key = JSON.stringify([run && run.n, L.training ? 'training' : run && run.step, L.stale, L.setup.problem, L.setup.train, L.seedQueue && L.seedQueue.left.length, L.sweep2 && [L.sweep2.what, L.sweep2.i, L.sweep2.status], st.eps, st.what, L.runs.length, theme.dark]);
    if (key === L._anKey && body.firstChild) return;
    L._anKey = key;
    clear(body);
    const secs = [];
    if (d.family === 'control') { body.append(UI().section('Analyse', UI().note('A model that learns from rewards has its own closer looks: Test it 100 times in the Training tab, and the views beside the picture.'))); return; }
    secs.push(sweepSec(L, st), seedsSec(L));
    if (sup && L.training) secs.push(UI().section('The trained model', UI().note('How sure it is, which inputs matter, how to fool it and how small it can be: these look at the run once it stops. Pause it, or let it finish.')));
    else if (!sup) secs.push(UI().section('The trained model', UI().note(run && L.gen ? 'A GAN and contrastive learning are measured their own way (beside the picture). The closer looks below are for models that learn from examples with answers.' : 'Train a run to look closer at it: how sure it is, which inputs matter, how to fool it, and how small it can be made.')));
    else {
      try {
        if (d.output.kind === 'classes') secs.push(calibrationSec(L));
        if (d.input.kind === 'vector') secs.push(importanceSec(L));
        if (d.input.kind === 'image' && d.output.kind === 'classes') secs.push(attackSec(L, st));
        secs.push(compressSec(L));
      } catch (e) { secs.push(UI().note(`(Could not look closer at this model: ${e.message})`)); }
    }
    body.append(...secs);
  };
  A.SWEEPS = SWEEPS;
  A.calibration = calibration; A.importance = importance; A.attack = attack; A.compress = compress;
  NA.banalyse = A;
})(window.NA);
