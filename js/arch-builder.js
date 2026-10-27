/* arch-builder.js — the model you built in the model builder, as a zoomable architecture. The top level is the
   whole model, input → your blocks → output, with the shape of the data on every arrow. Each block opens into
   what it really computes on one example (the first validation example): a dense layer's neurons, a
   convolution's channels and windows, attention's heads and query positions, the parts of a transformer block.
   Every number there opens what produced it (a weighted sum, a lookup, an average, a maximum), whose inputs open
   the numbers of the block before, and NA.deep takes each one on down to the arithmetic, the gates and the
   electrons. Before the first run, and once the setup has changed, the model shown has the fresh weights the
   next run would start from. */
(function (NA) {
  'use strict';
  const { fmt, fmtInt, fmtPct, theme, clamp } = NA;

  // ------------------------------------------------------------------ names, colours, shapes
  // the builder's palette: layers, sequences, images and reshapes, then the input and the output
  const TONE = { dense: 's1', activation: 's1', dropout: 's1', layernorm: 's1', attention: 's7', transformer: 's7', conv: 's4', pool: 's4', flatten: 's6', gap: 's6' };
  const NAME = { dense: 'Dense', activation: 'Activation', dropout: 'Dropout', layernorm: 'LayerNorm', attention: 'Self-attention', transformer: 'Transformer', conv: 'Convolution', pool: 'Max pool', flatten: 'Flatten', gap: 'Global average pool' };
  // the attribute each block gets in the Python tab's Model class (js/builder-py.js): self.dense3, self.conv1, …
  const PYMOD = { dense: 'dense', dropout: 'drop', layernorm: 'norm', attention: 'attn', transformer: 'block', conv: 'conv' };
  const PYACT = { relu: (z) => `F.relu(${z})`, leaky: (z) => `F.leaky_relu(${z}, 0.05)`, elu: (z) => `F.elu(${z})`, gelu: (z) => `F.gelu(${z}, approximate="tanh")`, swish: (z) => `F.silu(${z})`,
    softplus: (z) => `F.softplus(${z})`, tanh: (z) => `torch.tanh(${z})`, sigmoid: (z) => `torch.sigmoid(${z})`, sine: (z) => `torch.sin(${z})`, gaussian: (z) => `torch.exp(-${z} ** 2)`, linear: (z) => z };
  const shapeText = (s) => (s && s.length ? s.join(' × ') : '?');
  const pyShape = (s) => `(1, ${s.join(', ')})`;
  const maxAbs = (a) => { let m = 1e-9; for (let i = 0; i < a.length; i++) { const v = Math.abs(a[i]); if (v > m) m = v; } return m; };
  const argmax = (a, o = 0, n = a.length - o) => { let b = 0; for (let k = 1; k < n; k++) if (a[o + k] > a[o + b]) b = k; return b; };
  const softmax = (a, o, n) => { let m = -Infinity; for (let k = 0; k < n; k++) m = Math.max(m, a[o + k]); const e = new Float64Array(n); let z = 0; for (let k = 0; k < n; k++) { e[k] = Math.exp(a[o + k] - m); z += e[k]; } for (let k = 0; k < n; k++) e[k] /= z; return e; };
  const ACT = (id) => NA.engine.ACTIVATIONS[id] || NA.engine.ACTIVATIONS.linear;
  // what NA.deep needs to draw an activation: none for a linear one
  const actOf = (id) => { const a = ACT(id); return a.id === 'linear' ? null : { id: a.id, label: a.label, f: a.f }; };
  // a block's settings with the engine's defaults filled in
  const cfgOf = (spec) => Object.assign({}, (NA.bx.BLOCKS[spec.type] || {}).defaults, ...Object.keys(spec).filter((k) => spec[k] !== undefined && spec[k] !== null).map((k) => ({ [k]: spec[k] })));
  // a dense neuron's own activation: what the engine runs for it (the layer's functions), or what the block asks for
  const unitAct = (lay, spec, j) => (lay ? (lay.fns ? lay.fns[j] : lay.fn).id : Array.isArray(spec.acts) && NA.engine.ACTIVATIONS[spec.acts[j]] ? spec.acts[j] : cfgOf(spec).act || 'relu');

  // ------------------------------------------------------------------ drawing helpers
  /* A matrix drawn as an image, one pixel per number, scaled up without smoothing: crisp cells, and fast even for
     tens of thousands of numbers. Diverging colours around 0, sequential from 0 (seq) or ink on paper (ink, for
     pictures). Cached with the example's pass, so an image is only remade when its numbers change. */
  function heatImg(cache, key, rows, cols, get, o = {}) {
    const k = key + (theme.dark ? '|d' : '|l');
    let cv = cache && cache.get(k);
    if (cv) return cv;
    cv = document.createElement('canvas'); cv.width = Math.max(1, cols); cv.height = Math.max(1, rows);
    let m = o.max;
    if (!m) { m = 1e-9; for (let i = 0; i < rows; i++) for (let j = 0; j < cols; j++) { const v = Math.abs(get(i, j)); if (v > m) m = v; } }
    const g = cv.getContext('2d'), img = g.createImageData(cv.width, cv.height), T = theme.tokens, lut = o.seq ? theme.seq : theme.div;
    const paper = T['panel-2'] || [240, 240, 240], ink = T.ink || [20, 20, 20], grey = T.line || [200, 200, 200];
    for (let i = 0; i < rows; i++) for (let j = 0; j < cols; j++) {
      const p = (i * cols + j) * 4, v = get(i, j);
      if ((o.mask && o.mask(i, j)) || !Number.isFinite(v)) { img.data[p] = grey[0]; img.data[p + 1] = grey[1]; img.data[p + 2] = grey[2]; }
      else if (o.ink) { const t = clamp(v, 0, 1); for (let c = 0; c < 3; c++) img.data[p + c] = paper[c] + (ink[c] - paper[c]) * t; }
      else {
        const q = o.seq ? Math.round(Math.sqrt(clamp(v / m, 0, 1)) * 255) * 3 : Math.round((clamp(v / m, -1, 1) + 1) * 128) * 3;
        img.data[p] = lut[q]; img.data[p + 1] = lut[q + 1]; img.data[p + 2] = lut[q + 2];
      }
      img.data[p + 3] = 255;
    }
    g.putImageData(img, 0, 0);
    if (cache) { if (cache.size > 600) cache.clear(); cache.set(k, cv); }
    return cv;
  }
  // where rows × cols cells land in r: filling it, or square from its top-left corner (as the ArchView lays out a
  // square grid), so a drawing and the grid over it agree
  const fitCells = (r, rows, cols, square) => {
    if (!square) return { x: r.x, y: r.y, w: r.w, h: r.h, cw: r.w / cols, ch: r.h / rows };
    const s = Math.min(r.w / cols, r.h / rows);
    return { x: r.x, y: r.y, w: s * cols, h: s * rows, cw: s, ch: s };
  };
  function drawHeat(ctx, cv, r, rows, cols, square) {
    const q = fitCells(r, rows, cols, square);
    ctx.save(); ctx.imageSmoothingEnabled = false; ctx.drawImage(cv, q.x, q.y, q.w, q.h); ctx.restore();
    ctx.strokeStyle = theme.c('line-2'); ctx.lineWidth = 0.8; ctx.strokeRect(q.x - 0.5, q.y - 0.5, q.w + 1, q.h + 1);
    return q;
  }
  // horizontal bars around a vertical zero line, one row per number: a column that lines up with a matrix's rows
  function hbars(ctx, vals, r, o = {}) {
    const n = vals.length, bh = r.h / Math.max(1, n);
    let m = o.max || 1e-9;
    if (!o.max) for (let i = 0; i < n; i++) if (Math.abs(vals[i]) > m) m = Math.abs(vals[i]);
    const zero = r.x + r.w / 2, span = r.w / 2;
    ctx.strokeStyle = theme.c('line-2'); ctx.lineWidth = 0.8; ctx.beginPath(); ctx.moveTo(zero, r.y); ctx.lineTo(zero, r.y + r.h); ctx.stroke();
    for (let i = 0; i < n; i++) {
      const v = vals[i];
      if (!Number.isFinite(v)) continue;
      const w = (Math.abs(v) / m) * span;
      ctx.fillStyle = o.color ? o.color(i, v) : theme.divColor(clamp(v / m, -1, 1) * 0.85 + Math.sign(v) * 0.15);
      ctx.fillRect(v >= 0 ? zero : zero - w, r.y + i * bh + bh * 0.12, w, Math.max(0.6, bh * 0.76));
    }
  }
  // labels down the left of r's rows (right-aligned against it), or under its columns
  function rowLabels(ctx, S, labels, r, o = {}) {
    const n = labels.length, ch = r.h / Math.max(1, n), size = Math.min(o.size || 11, ch * 0.86);
    if (size < 5.5) return;
    labels.forEach((t, i) => S.text(ctx, t, r.x - 4, r.y + ch * (i + 0.5), { size, mono: true, align: 'right', color: o.hi === i ? theme.c('ink') : theme.c('muted'), w: o.hi === i ? 700 : 500, maxW: o.maxW || 60 }));
  }
  function colLabels(ctx, S, labels, r, o = {}) {
    const n = labels.length, cw = r.w / Math.max(1, n), size = Math.min(o.size || 11, cw * 0.9);
    if (size < 5.5) return;
    labels.forEach((t, j) => S.text(ctx, t, r.x + cw * (j + 0.5), r.y + r.h + size * 0.95, { size, mono: true, align: 'center', color: o.hi === j ? theme.c('ink') : theme.c('muted'), w: o.hi === j ? 700 : 500, maxW: cw + (o.over || 0) }));
  }
  // up to n channels in a grid of square tiles: the column count that makes the tiles largest in a w × h box
  function tileLay(C, n, w, h) {
    const k = Math.min(n, C);
    let best = null;
    for (let cols = 1; cols <= k; cols++) { const rows = Math.ceil(k / cols), s = Math.min(w / cols, h / rows); if (!best || s > best.s) best = { cols, rows, s }; }
    return Object.assign(best, { k });
  }

  // ------------------------------------------------------------------ the model and one example through it
  /* build() runs on every redraw (thirty times a second while a deep level animates), so the example's pass
     through the model is cached and run again only when the model, its weights (a training step) or the setup
     change. Every forward pass reuses the layers' buffers, the results views' whole batches included, so what
     each layer computed for the example is copied out at once. */
  const BUFS = { dense: ['z', 'a', 'out'], activation: ['out'], dropout: ['out'], layernorm: ['out', 'xhat', 'rstd'], attention: ['out'],
    transformer: ['h1', 'xh1', 'r1', 'mid', 'h2', 'xh2', 'r2', 'u', 'g', 'mo', 'out'], conv: ['z', 'out'], pool: ['out'], flatten: ['out'], gap: ['out'] };
  let FRESH = null, PROBE = null, PYCODE = null;
  function runProbe(model, d) {
    const va = d.nVa > 0, n = va ? d.nVa : d.nTr;
    if (!n || !d.Xtr) return null;
    const Xs = va ? d.Xva : d.Xtr, Ys = va ? d.Yva : d.Ytr, x = Xs.subarray(0, model.inSize), ySize = Ys.length / n;
    const out = model.predict(x, 1), cp = (a) => (a ? Float32Array.from(a) : null);
    const P = { va, x: Float32Array.from(x), y: Array.from(Ys.subarray(0, ySize)), logits: Float32Array.from(out), layers: [], img: new Map() };
    if (model.embed) { P.ids = Int32Array.from(x); P.emb = cp(model.embed.out); }
    for (const lay of model.layers) {
      const r = {};
      for (const k of BUFS[lay.type] || ['out']) r[k] = cp(lay[k]);
      if (lay.core) for (const k of ['q', 'k', 'v', 'o', 'y', 'att']) r[k] = cp(lay.core[k]);
      if (lay.type === 'pool') r.arg = Int32Array.from(lay.arg);
      P.layers.push(r);
    }
    return P;
  }
  /* The model to show: the trained one while it matches the setup; otherwise the untrained one the next run
     would start from (same seed, same initialisation), made once per setup. */
  function current(L) {
    const d = L.dataFor(), chk = L.check || L.recheck(), live = L.model && L.run && !L.stale ? L.model : null;
    let model = live, err = null;
    if (!live && chk.ok && !d.error) {
      if (!FRESH || FRESH.chk !== chk || FRESH.d !== d) {
        let m = null, e = null;
        try { m = NA.bx.build(L.setup.blocks, d.input, d.output, { seed: L.setup.train.seed, init: L.setup.train.init }); } catch (ex) { e = ex.message; }
        FRESH = { chk, d, model: m, err: e };
      }
      model = FRESH.model; err = FRESH.err;
    }
    if (!PROBE || PROBE.model !== model || PROBE.step !== L.step || PROBE.viz !== L.vizVersion || PROBE.d !== d) {
      let P = null, e = null;
      if (model && chk.ok && !d.error) { try { P = runProbe(model, d); } catch (ex) { e = ex.message; } }
      PROBE = { model, step: L.step, viz: L.vizVersion, d, P, err: e };
    }
    return { d, chk, model, live: !!live, P: PROBE.P, err: err || PROBE.err };
  }

  // ------------------------------------------------------------------ the tree
  function build(L, size) {
    const wide = size.w / size.h >= 1.05, W2 = (w, t) => (wide ? w : t), X = { wide }, Dp = NA.deep;
    const st = current(L), d = st.d, chk = st.chk, model = st.model, blocks = L.setup.blocks, nb = blocks.length;
    const ok = chk.ok && !d.error, P = ok ? st.P : null, nums = !!(P && model);
    const seq = !d.error && d.input && d.input.kind === 'tokens', out = d.output || {}, K = out.kind === 'tokens' ? out.vocab : out.k;
    const names = d.names || [], vocab = d.vocab || [], word = (id) => (vocab[id] !== undefined ? vocab[id] : String(id));
    const quote = (id) => `"${word(id)}"`, tok = (t) => (P && P.ids ? quote(P.ids[t]) : `position ${t}`);
    const BV = NA.builderViz, cache = P ? P.img : null;
    const shapes = chk.shapes || [];
    const inS = (i) => (i < nb ? (shapes[i] || {}).in : shapes.length ? (shapes[nb - 1] || {}).out : inputShape());
    const outS = (i) => (i < nb ? (shapes[i] || {}).out : chk.outShape);
    function inputShape() { const q = d.input || {}; return q.kind === 'tokens' ? [q.len, q.embed] : q.kind === 'vector' ? [q.dim] : q.kind === 'image' ? [q.c, q.h, q.w] : null; }
    const blockTitle = (i) => { const b = blocks[i]; return `${i + 1} · ${b.type === 'dense' && b.units === 1 ? 'Neuron' : NAME[b.type] || b.type}`; };
    const pyName = (i) => (PYMOD[blocks[i].type] ? `model.${PYMOD[blocks[i].type]}${i + 1}` : null);
    const T0 = seq ? d.input.len : 1, D0 = seq ? d.input.embed : 0;

    // ---- what the example is, and what the model makes of it
    const target = P ? P.y : null;
    let probs = null, pred = null, tokPreds = null, tokRight = 0, tokCount = 0;
    if (nums) {
      if (out.kind === 'classes') { probs = softmax(P.logits, 0, K); pred = argmax(probs); }
      else if (out.kind === 'tokens') {
        tokPreds = [];
        for (let t = 0; t < T0; t++) { const p = softmax(P.logits, t * K, K), b = argmax(p); tokPreds.push({ b, p: p[b], probs: p }); if (target[t] >= 0) { tokCount++; if (b === target[t]) tokRight++; } }
      }
    }
    const inputLabel = (idx) => {
      const q = d.input || {};
      if (q.kind === 'image') { const HW = q.h * q.w, c = Math.floor(idx / HW), y = Math.floor((idx % HW) / q.w), x = idx % q.w; return `${q.c > 1 ? `channel ${c + 1}, ` : ''}pixel (${x}, ${y})`; }
      return q.dim === 1 ? 'x' : `x${NA.sub(idx + 1)}`;
    };
    const exampleText = () => {
      if (!P) return '';
      const fam = d.family;
      if (fam === 'plane') return `the point (${fmt(P.x[0], 2)}, ${fmt(P.x[1], 2)}), ${names[target[0]] || 'class ' + (target[0] + 1)}`;
      if (fam === 'curve') return `x = ${fmt(P.x[0], 3)}, whose target is ${fmt(target[0], 3)}`;
      if (fam === 'digits') return `a handwritten ${names[target[0]]}`;
      const words = Array.from(P.ids, word);
      return `"${words.slice(0, 14).join(' ')}${words.length > 14 ? ' …' : ''}"`;
    };
    const answerText = () => {
      if (!nums) return '';
      if (out.kind === 'classes') return `the model says ${names[pred] || 'class ' + (pred + 1)} (${fmtPct(probs[pred], 0)})${pred === target[0] ? ', right' : `, wrong`}`;
      if (out.kind === 'value') return `the model says ${fmt(P.logits[0], 3)}`;
      return `its next-token guesses are right at ${tokRight} of ${tokCount} position${tokCount === 1 ? '' : 's'}`;
    };
    const lossText = () => {
      if (!nums) return '';
      const id = (L.setup.loss || {}).id;
      if (out.kind === 'value') {
        const r = P.logits[0] - target[0], a = Math.abs(r), dl = (L.setup.loss || {}).delta || 0.2;
        return id === 'mae' ? `|ŷ − y| = ${fmt(a, 4)}` : id === 'huber' ? `Huber(ŷ − y) = ${fmt(a <= dl ? 0.5 * r * r : dl * (a - 0.5 * dl), 4)}` : `½(ŷ − y)² = ${fmt(0.5 * r * r, 4)}`;
      }
      if (out.kind === 'classes') {
        const y = target[0];
        if (id === 'mse') { let s = 0; for (let k = 0; k < K; k++) s += (probs[k] - (k === y ? 1 : 0)) ** 2; return `½ Σ (p − one-hot)² = ${fmt(0.5 * s, 4)}`; }
        if (id === 'hinge') { let s = 0; for (let k = 0; k < K; k++) if (k !== y) s += Math.max(0, 1 + P.logits[k] - P.logits[y]); return `hinge = ${fmt(s, 4)}`; }
        return `−ln p(${names[y] || y}) = ${fmt(-Math.log(Math.max(probs[y], 1e-30)), 4)}`;
      }
      let s = 0;
      for (let t = 0; t < T0; t++) if (target[t] >= 0) s -= Math.log(Math.max(tokPreds[t].probs[target[t]], 1e-30));
      return `mean −ln p(next token) = ${fmt(tokCount ? s / tokCount : 0, 4)}`;
    };
    // the first lines of a snippet that needs what a module received (or returned) for this example
    const pyGrab = (mod, name, expr) => ['grab = {}', `hook = ${mod}.register_forward_hook(lambda m, inp, out: grab.update(${name}=${expr}))`, 'with torch.no_grad(): model(Xva[:1])                # the first validation example, as here', 'hook.remove()'];
    const whose = P ? (P.va && !d.noHoldout ? 'the first validation example' : P.va ? 'the first example (the text is too short to hold any back)' : 'the first training example (there is no validation set)') : '';
    const weightsNote = st.live ? `trained for ${fmtInt(L.step)} step${L.step === 1 ? '' : 's'}` : 'untrained weights: train to see learned ones';

    // ---- every number's story: what block bi hands on at flat index idx (bi = −1: the input or its embedding)
    const inVals = (bi) => (bi === 0 ? (seq ? P.emb : P.x) : P.layers[bi - 1].out);
    const posOf = (s, idx) => (s.length === 3 ? { c: Math.floor(idx / (s[1] * s[2])), y: Math.floor((idx % (s[1] * s[2])) / s[2]), x: idx % s[2] } : s.length === 2 ? { t: Math.floor(idx / s[1]), c: idx % s[1] } : { c: idx });
    const shapeAfter = (bi) => (bi < 0 ? (seq ? [T0, D0] : inputShape()) : outS(bi));
    // what number idx of block bi's output is called
    function nameOf(bi, idx) {
      const s = shapeAfter(bi), p = posOf(s, idx);
      if (bi < 0) return seq ? `x₀[${p.c}] of ${tok(p.t)}` : inputLabel(idx);
      if (s.length === 3) return `channel ${p.c + 1} (${p.x}, ${p.y}) of block ${bi + 1}`;
      if (s.length === 2) return `${tok(p.t)}[${p.c}] after block ${bi + 1}`;
      return blocks[bi].type === 'dense' ? `neuron ${idx + 1} of block ${bi + 1}` : `number ${idx} of block ${bi + 1}`;
    }
    function valueNode(bi, idx) {
      if (bi < 0) {
        if (!seq) return Dp.stored(X, P.x[idx], inputLabel(idx));
        const p = posOf([T0, D0], idx), id = P.ids[p.t], E = model.embed.tok.w, Pp = model.embed.pos.w;
        return Object.assign(Dp.fadd(X, E[id * D0 + p.c], Pp[p.t * D0 + p.c]), { title: `x₀[${p.c}] for ${quote(id)} = E + P`, sub: `its token's row plus row ${p.t} of the position table` });
      }
      const b = blocks[bi], s = outS(bi), p = posOf(s, idx);
      switch (b.type) {
        case 'dense': return denseValue(bi, idx);
        case 'activation': return actValue(bi, idx);
        case 'dropout': return Object.assign(valueNode(bi - 1, idx), { sub: `passed through dropout ${bi + 1} unchanged` });
        case 'layernorm': return lnCell(bi, s.length === 2 ? p.t : 0, s.length === 2 ? p.c : idx);
        case 'attention': return attnValue(bi, p.t, p.c);
        case 'transformer': return tfValue(bi, p.t, p.c);
        case 'conv': return convNeuron(bi, p.c, p.y, p.x);
        case 'pool': return poolValue(bi, idx);
        case 'flatten': return Object.assign(valueNode(bi - 1, idx), { sub: `number ${idx} of the flattened vector` });
        case 'gap': return gapValue(bi, idx);
      }
      return Dp.stored(X, P.layers[bi].out[idx], nameOf(bi, idx));
    }
    // the cell under a value that is the sum of two others: either side, and the float addition itself
    function sumNode(title, formula, a, b, aNode, bNode, about, code) {
      const rs = W2([[0, 0.36, 0.31, 0.58], [0.345, 0.36, 0.31, 0.58], [0.69, 0.36, 0.31, 0.58]], [[0, 0.3, 1, 0.2], [0, 0.54, 1, 0.2], [0, 0.78, 1, 0.2]]);
      return Dp.lazy({
        title, sub: `${fmt(a, 3)} + ${fmt(b, 3)} = ${fmt(a + b, 3)}`, tone: 's2',
        summary: (ctx, r, S) => S.text(ctx, `${fmt(a, 2)} + ${fmt(b, 2)}`, r.x + r.w / 2, r.y + r.h * 0.4, { size: clamp(r.w / 10, 6, 11), mono: true, align: 'center', color: theme.c('ink-2'), maxW: r.w }),
        inner: (ctx, r, S) => S.lines(ctx, [formula, `${fmt(a + b, 4)} = ${fmt(a, 4)} + ${fmt(b, 4)}`, 'Zoom into either side, or into the addition itself.'], S.place(r, [0, 0.02, 1, 0.24]), { mono: true, size: S.u(r, 8, 13), strong: (k) => k === 0 }),
        about, code,
      }, () => [
        Object.assign(aNode(), { id: 'a', rect: rs[0] }),
        Object.assign(bNode(), { id: 'b', rect: rs[1] }),
        Object.assign(Dp.fadd(X, a, b), { id: 'add', rect: rs[2], title: 'The addition' }),
      ]);
    }
    const residualAbout = (what) => [`A residual connection: ${what} is added to the block's own input, number by number, instead of replacing it. The block only has to learn a correction, and during training the gradient flows straight back through the addition, which is what lets deep stacks keep learning.`];

    // ---- dense layers (and the output head, which is one)
    function denseNeuron(bi, t, j) {
      const lay = model.layers[bi], head = bi === nb, F = lay.nIn, U = lay.U, xin = inVals(bi), Wt = lay.W.w, bb = lay.b.w;
      const id = head ? 'linear' : unitAct(lay, blocks[bi], j), act = actOf(id), seqL = lay.rows > 1;
      const mod = head ? 'model.out' : pyName(bi), what = head ? (out.kind === 'classes' ? `the score for ${names[j] || 'class ' + (j + 1)}` : out.kind === 'tokens' ? `the score for ${quote(j)} as the next token` : K === 1 ? 'the output neuron' : `output ${j + 1}`) : `neuron ${j + 1}`;
      const terms = Array.from({ length: F }, (_, p) => ({ a: xin[t * F + p], w: Wt[j * F + p] }));
      const z = terms.reduce((s, q) => s + q.a * q.w, bb[j]);
      return Dp.neuron(X, {
        title: head ? `${what[0].toUpperCase()}${what.slice(1)}${seqL ? ` after ${tok(t)}` : ''}` : `${blockTitle(bi)} · neuron ${j + 1}${seqL ? ` at ${tok(t)}` : ''}`,
        sub: `${F} input${F === 1 ? '' : 's'}${act ? ', ' + act.label : ', no activation'}`, tone: head ? 's2' : 's1',
        terms, bias: bb[j], act, zName: 'z', outName: head ? 'score' : 'a',
        inLabel: (p) => nameOf(bi - 1, t * F + p), wLabel: (p) => `W[${j}, ${p}]`, inWhat: seqL ? `${tok(t)}'s ${F} numbers` : bi ? `what block ${bi} hands on` : 'the input',
        inputNode: (p) => valueNode(bi - 1, t * F + p),
        vars: { w: `${mod}.weight[${j}]`, a: 'x', b: `${mod}.bias[${j}]` },
        codeHead: [...pyGrab(mod, 'x', 'inp[0][0]'), seqL ? `x = grab['x'][${t}]                                 # ${tok(t)}'s ${F} numbers: what the layer reads there` : `x = grab['x']                                      # the ${F} numbers the layer reads`],
        codeWhat: `as ${what} of ${mod}, an nn.Linear(${F}, ${U})`,
        about: [
          head ? `${what[0].toUpperCase()}${what.slice(1)}${seqL ? ` after ${tok(t)}` : ''}: the output head is a dense layer without an activation, so this number is a plain weighted sum of the ${F} numbers that reach it, plus a bias: ${fmt(z, 4)}.${out.kind === 'classes' ? ' The softmax then turns the scores into probabilities.' : out.kind === 'tokens' ? ' The softmax over all the tokens turns these scores into next-token probabilities.' : ' It is the model\'s answer.'}`
            : `Neuron ${j + 1} of block ${bi + 1}${seqL ? `, at ${tok(t)}` : ''}: it multiplies each of its ${F} inputs by its own learned weight, adds them up with its bias (z = ${fmt(z, 4)}) and bends the result with ${act ? act.label : 'no activation'}${act ? `: ${fmt(act.f(z), 4)}` : ''}.${seqL ? ' The same neuron, with the same weights, runs at every position of the sequence on its own.' : ''}`,
          'Zoom into an input bar to open the number of the block below that produced it.',
        ],
      });
    }
    function denseValue(bi, idx) {
      const lay = model.layers[bi], t = Math.floor(idx / lay.U), j = idx % lay.U, r = P.layers[bi];
      if (!lay.residual) return denseNeuron(bi, t, j);
      const xin = inVals(bi);
      return sumNode(`${nameOf(bi, idx)}: input + neuron`, 'out = x + a   (a residual connection)', xin[idx], r.a[idx], () => valueNode(bi - 1, idx), () => denseNeuron(bi, t, j), residualAbout(`neuron ${j + 1}'s output`), `x = torch.randn(1, ${lay.rows > 1 ? lay.rows + ', ' : ''}${lay.nIn})                     # an input of this shape\nout = x + ${(PYACT[lay.fn.id] || PYACT.linear)(`${pyName(bi)}(x)`)}        # the same shape in and out, so they can be added`);
    }

    // ---- element-wise blocks
    function actValue(bi, idx) {
      const id = model.layers[bi].fn.id, z = inVals(bi)[idx], a = ACT(id);
      if (id === 'linear') return Object.assign(valueNode(bi - 1, idx), { sub: 'the linear activation passes it unchanged' });
      return Object.assign(Dp.lib(X, id, { z }), { title: `${blockTitle(bi)}, number ${idx}`, sub: `${a.label}(${fmt(z, 3)}) = ${fmt(a.f(z), 3)}` });
    }
    function lnCell(bi, t, j) {
      const lay = model.layers[bi], D = lay.d, x = inVals(bi), row = Array.from(x.subarray(t * D, t * D + D));
      return Object.assign(Dp.lib(X, 'layernorm', { x: row, j, g: lay.g.w[j], b: lay.b.w[j] }), { title: `LayerNorm ${bi + 1}, number ${j}${lay.rows > 1 ? ` of ${tok(t)}` : ''}` });
    }

    // ---- convolutions, pooling, global average
    function convNeuron(bi, f, oy, ox) {
      const lay = model.layers[bi], { C, H, k, s, pad } = lay, Wd = lay.W, xin = inVals(bi), w = lay.Wt.w, J = C * k * k, id = lay.fn.id, act = actOf(id);
      const terms = [], src = [], tag = [];
      for (let c = 0; c < C; c++) for (let ky = 0; ky < k; ky++) for (let kx = 0; kx < k; kx++) {
        const iy = oy * s + ky - pad, ix = ox * s + kx - pad, inside = iy >= 0 && iy < H && ix >= 0 && ix < Wd, at = (c * H + iy) * Wd + ix;
        terms.push({ a: inside ? xin[at] : 0, w: w[f * J + (c * k + ky) * k + kx] }); src.push(inside ? at : -1); tag.push([c, ky, kx, ix, iy]);
      }
      const z = terms.reduce((q, t) => q + t.a * t.w, lay.b.w[f]), mod = pyName(bi), pads = src.filter((v) => v < 0).length;
      return Dp.neuron(X, {
        title: `${blockTitle(bi)} · channel ${f + 1} at (${ox}, ${oy})`, sub: `${J} inputs${act ? ', ' + act.label : ''}`, tone: 's4',
        terms, bias: lay.b.w[f], act, outName: 'a',
        inLabel: (p) => (src[p] < 0 ? `padding at (${tag[p][3]}, ${tag[p][4]})` : nameOf(bi - 1, src[p])), wLabel: (p) => `kernel[${f}, ${tag[p][0]}, ${tag[p][1]}, ${tag[p][2]}]`,
        inputNode: (p) => (src[p] < 0 ? Object.assign(Dp.stored(X, 0, 'zero padding'), { sub: 'past the edge of the picture' }) : valueNode(bi - 1, src[p])),
        rowNames: [`inputs: a ${k} × ${k} window of all ${C} input channel${C > 1 ? 's' : ''} (${J} numbers)`, 'the kernel’s weights (learned)', 'products wᵢ·aᵢ'],
        vars: { w: `${mod}.weight[${f}].flatten()`, a: 'patch', b: `${mod}.bias[${f}]` },
        codeHead: [...pyGrab(mod, 'x', 'inp[0]'), `patch = F.pad(grab['x'], [${pad}] * 4)[0, :, ${oy * s}:${oy * s + k}, ${ox * s}:${ox * s + k}].flatten()   # ${J} numbers`],
        codeWhat: `as one output of ${mod}, an nn.Conv2d(${C}, ${lay.F}, ${k}${s > 1 ? `, stride=${s}` : ''}, padding=${pad})`,
        about: [
          `One number of block ${bi + 1}'s output: channel ${f + 1} at column ${ox}, row ${oy}. The filter is laid over a ${k} × ${k} window of the input${C > 1 ? `, through all ${C} channels` : ''}${s > 1 ? ` (the window moves ${s} pixels per output, so the map shrinks)` : ''}; each of the ${J} numbers under it is multiplied by its weight and the products are added up with the bias: z = ${fmt(z, 4)}${act ? `, and ${act.label} gives ${fmt(act.f(z), 4)}` : ''}.${pads ? ` ${pads} of the window's positions fall past the edge of the picture, where the input counts as 0 (zero padding).` : ''}`,
          `The same ${J} weights slide over every position of the picture, which is why a convolution has so few parameters for so many outputs. Zoom into an input bar to open the number of the block below that it read.`,
        ],
      });
    }
    function poolValue(bi, idx) {
      const lay = model.layers[bi], { C, H, S: Sz, Ho, Wo } = lay, Wd = lay.W, xin = inVals(bi), r = P.layers[bi];
      const c = Math.floor(idx / (Ho * Wo)), oy = Math.floor((idx % (Ho * Wo)) / Wo), ox = idx % Wo, cells = [];
      for (let dy = 0; dy < Sz; dy++) for (let dx = 0; dx < Sz; dx++) { const iy = oy * Sz + dy, ix = ox * Sz + dx, at = (c * H + iy) * Wd + ix; cells.push({ v: xin[at], at, ix, iy }); }
      // the engine remembers which input won each window (ties go to the first)
      const best = cells.find((q) => q.at === r.arg[idx]) || cells.reduce((p, q) => (q.v > p.v ? q : p)), vals = Float32Array.from(cells.map((q) => q.v)), ties = cells.filter((q) => q.v === best.v).length;
      return Dp.lazy({
        title: `${blockTitle(bi)} · channel ${c + 1} at (${ox}, ${oy})`, sub: `max of ${cells.length} = ${fmt(best.v, 3)}`, tone: 's4',
        summary: (ctx, q, S) => S.bars(ctx, vals, q, { color: (k) => (cells[k] === best ? theme.c('s4') : theme.c('muted', 0.6)) }),
        inner: (ctx, q, S) => {
          const u = S.u(q, 8, 13), br = S.place(q, W2([0, 0.32, 0.6, 0.56], [0, 0.3, 1, 0.3]));
          S.lines(ctx, [`out = the largest of the ${Sz} × ${Sz} window = ${fmt(best.v, 4)}`, `the window: columns ${ox * Sz} to ${ox * Sz + Sz - 1}, rows ${oy * Sz} to ${oy * Sz + Sz - 1} of channel ${c + 1}`, `the winner sits at (${best.ix}, ${best.iy})${ties > 1 ? `: ${ties} numbers tie, the first one counts` : ''}`], S.place(q, [0, 0, 1, 0.2]), { mono: true, size: u * 0.92, strong: (k) => k === 0 });
          S.caption(ctx, 'the window, the largest one marked: zoom into a bar', br, { size: u * 0.8, minW: 200 });
          S.bars(ctx, vals, br, { color: (k) => (cells[k] === best ? theme.c('s4') : theme.c('muted', 0.6)) });
        },
        grids: [{ id: 'win', rect: W2([0, 0.32, 0.6, 0.56], [0, 0.3, 1, 0.3]), rows: 1, cols: cells.length, make: (i, j) => valueNode(bi - 1, cells[j].at) }],
        about: [`Max pooling keeps the largest number in each ${Sz} × ${Sz} window: here ${fmt(best.v, 4)}, from (${best.ix}, ${best.iy}).${ties > 1 ? ` ${ties} of the numbers are equal (typically where the picture is blank, so a filter gives the same answer everywhere); the first of them is kept.` : ''} A feature that fired anywhere in the window still counts, the picture shrinks, and the next layer's filters see a wider area for the same size.`, 'A comparison is a subtraction whose sign is read, so the circuit inside is the same adder as the one under an addition. The winner box opens the number that won.'],
        code: `x = torch.randn(1, ${C}, ${H}, ${Wd})\ny = F.max_pool2d(x, ${Sz})                     # (1, ${C}, ${Ho}, ${Wo})\nprint(y[0, ${c}, ${oy}, ${ox}] == x[0, ${c}, ${oy * Sz}:${oy * Sz + Sz}, ${ox * Sz}:${ox * Sz + Sz}].max())   # True`,
      }, () => [Object.assign(valueNode(bi - 1, best.at), { id: 'max', rect: W2([0.66, 0.32, 0.34, 0.56], [0, 0.68, 1, 0.3]), title: `The winner, at (${best.ix}, ${best.iy})` })]);
    }
    function gapValue(bi, idx) {
      const s = inS(bi), xin = inVals(bi);
      if (s.length === 3) {
        const [C, H, Wd] = s, n = H * Wd, c = idx;
        return Dp.neuron(X, {
          title: `${blockTitle(bi)} · average of channel ${c + 1}`, sub: `${n} numbers × 1/${n}`, tone: 's6',
          terms: Array.from({ length: n }, (_, p) => ({ a: xin[c * n + p], w: 1 / n })), bias: null, act: null, outName: 'g',
          rowNames: [`channel ${c + 1}'s ${n} numbers`, `weights: all 1/${n}`, 'products'], inLabel: (p) => `(${p % Wd}, ${Math.floor(p / Wd)})`, wLabel: () => `1/${n}`,
          inputNode: (p) => valueNode(bi - 1, c * n + p), vars: { w: `torch.full((${n},), 1 / ${n})`, a: `x[0, ${c}].flatten()`, b: '' }, codeHead: [`x = torch.randn(1, ${C}, ${H}, ${Wd})                  # an input of this shape`],
          about: [`An average is a weighted sum whose weights are all 1/${n}: channel ${c + 1} of the ${C} is averaged over its ${H} × ${Wd} positions. It forgets where in the picture the channel fired and keeps how strongly, which is what lets a classifier work wherever the digit sits.`],
        });
      }
      const [T, D] = s, c = idx;
      return Dp.neuron(X, {
        title: `${blockTitle(bi)} · average of feature ${c}`, sub: `${T} positions × 1/${T}`, tone: 's6',
        terms: Array.from({ length: T }, (_, t) => ({ a: xin[t * D + c], w: 1 / T })), bias: null, act: null, outName: 'g',
        rowNames: [`feature ${c} at the ${T} positions`, `weights: all 1/${T}`, 'products'], inLabel: (t) => tok(t), wLabel: () => `1/${T}`,
        inputNode: (t) => valueNode(bi - 1, t * D + c), vars: { w: `torch.full((${T},), 1 / ${T})`, a: `x[0, :, ${c}]`, b: '' }, codeHead: [`x = torch.randn(1, ${T}, ${D})                  # an input of this shape`],
        about: [`Feature ${c} averaged over the ${T} positions of the sequence: a weighted sum whose weights are all 1/${T}. The result no longer says where in the sequence anything was.`],
      });
    }

    // ---- attention: the projections, the heads, a query's weighted average
    function attnData(bi) {
      const lay = model.layers[bi], r = P.layers[bi], core = lay.core, T = lay.T, D = lay.D, H = lay.heads;
      return { bi, lay, r, core, T, D, H, dh: D / H, causal: lay.causal, xin: lay.type === 'transformer' ? r.h1 : inVals(bi), mod: lay.type === 'transformer' ? `${pyName(bi)}.attn` : pyName(bi), inTf: lay.type === 'transformer' };
    }
    const xName = (A) => (A.inTf ? 'x̂ (after LN1)' : 'x');
    // the first line of an attention snippet: an input of the right shape (inside a transformer, after LN1)
    const pyX = (A) => (A.inTf ? `x = ${pyName(A.bi)}.norm1(torch.randn(1, ${A.T}, ${A.D}))   # x̂: the stream after LN1` : `x = torch.randn(1, ${A.T}, ${A.D})                 # any input of this shape`);
    function projNeuron(A, which, t, j) {
      const Wm = A.core['W' + which].w, bm = A.core['b' + which].w, D = A.D, src = which === 'o' ? A.r.o : A.xin;
      const h = Math.floor(j / A.dh), what = { q: 'query', k: 'key', v: 'value', o: 'output' }[which];
      return Dp.neuron(X, {
        title: which === 'o' ? `Attention output ${j} for ${tok(t)}` : `${which}[${j}] for ${tok(t)}`, sub: which === 'o' ? `heads side by side · Wo row ${j}` : `${xName(A)} · W${which} row ${j}`, tone: 's7',
        terms: Array.from({ length: D }, (_, p) => ({ a: src[t * D + p], w: Wm[j * D + p] })), bias: bm[j], act: null, zName: which,
        inLabel: (p) => (which === 'o' ? `head ${Math.floor(p / A.dh) + 1} output ${p % A.dh}` : `${xName(A).split(' ')[0]}[${p}] of ${tok(t)}`), wLabel: (p) => `W${which}[${j}, ${p}]`,
        inputNode: (p) => (which === 'o' ? headOut(A, Math.floor(p / A.dh), t, p % A.dh) : A.inTf ? tfLnCell(A.bi, 1, t, p) : valueNode(A.bi - 1, t * D + p)),
        vars: { w: `${A.mod}.${which}.weight[${j}]`, a: which === 'o' ? 'heads' : 'x', b: `${A.mod}.${which}.bias[${j}]` },
        codeHead: which === 'o' ? [...pyGrab(`${A.mod}.o`, 'h', 'inp[0][0]'), `heads = grab['h'][${t}]                             # ${tok(t)}: the heads' outputs side by side`]
          : [...pyGrab(A.mod, 'x', 'inp[0][0]'), `x = grab['x'][${t}]                                 # ${tok(t)}'s ${D} numbers${A.inTf ? ', after LN1' : ''}`],
        about: [which === 'o'
          ? `What attention hands on for ${tok(t)}, number ${j}: the ${A.H} heads' outputs placed side by side (${D} numbers) times row ${j} of the output matrix Wo, plus a bias. Wo mixes what the heads found back into one vector.`
          : `Number ${j} of the ${what} vector for ${tok(t)}${A.H > 1 ? ` (head ${h + 1} uses numbers ${h * A.dh} to ${(h + 1) * A.dh - 1})` : ''}: a weighted sum of that position's ${D} numbers with row ${j} of W${which}, plus a bias. Every position makes its ${what} with the same weights.`],
      });
    }
    const headW = (A, h, i, j) => A.r.att[(h * A.T + i) * A.T + j];
    const qkv = (A, which, i, h, c) => A.r[which][i * A.D + h * A.dh + c];
    function scoreDot(A, h, i, j) {
      const dh = A.dh;
      return Dp.neuron(X, {
        kind: 'dot', title: `Score: ${tok(i)} looking at ${tok(j)}`, sub: `q · k / √${dh}, head ${h + 1}`, tone: 's7',
        terms: Array.from({ length: dh }, (_, c) => ({ a: qkv(A, 'k', j, h, c), w: qkv(A, 'q', i, h, c) })), bias: null, scale: { v: 1 / Math.sqrt(dh), label: `1/√${dh}` }, zName: 's',
        rowNames: [`key k of ${tok(j)}`, `query q of ${tok(i)}`, 'products qᵢ·kᵢ'], inLabel: (c) => `k[${c}] of ${tok(j)}`, wLabel: (c) => `q[${c}] of ${tok(i)}`,
        inputNode: (c) => projNeuron(A, 'k', j, h * dh + c), vars: { w: `q[${i}]`, a: `k[${j}]` },
        codeHead: [...pyGrab(A.mod, 'x', 'inp[0][0]'), `sl = slice(${h * dh}, ${(h + 1) * dh})                               # head ${h + 1}'s numbers`, `q, k = ${A.mod}.q(grab['x'])[:, sl], ${A.mod}.k(grab['x'])[:, sl]`],
        about: [`How well the query of ${tok(i)} matches the key of ${tok(j)} in head ${h + 1}: the dot product of the two ${dh}-number vectors, scaled by 1/√${dh} so that scores do not grow with the size of the vectors. A higher score means more attention after the softmax.`],
      });
    }
    function headOut(A, h, i, c) {
      const n = A.causal ? i + 1 : A.T;
      return Dp.neuron(X, {
        kind: 'dot', title: `Head ${h + 1} output ${c} for ${tok(i)}`, sub: 'a weighted average of values', tone: 's7',
        terms: Array.from({ length: n }, (_, j) => ({ a: qkv(A, 'v', j, h, c), w: headW(A, h, i, j) })), bias: null, act: null, zName: 'o',
        rowNames: [`value ${c} of each position ${tok(i)} can see`, 'attention weights (a softmax: they add up to 1)', 'weight × value'],
        inLabel: (j) => `v of ${tok(j)}`, wLabel: (j) => `weight on ${tok(j)}`, inputNode: (j) => projNeuron(A, 'v', j, h * A.dh + c), vars: { w: `weights[${i}]`, a: `v[:, ${c}]` },
        codeHead: [...pyGrab(A.mod, 'x', 'inp[0][0]'), `sl = slice(${h * A.dh}, ${(h + 1) * A.dh})                               # head ${h + 1}'s numbers`, `q, k, v = (getattr(${A.mod}, n)(grab['x'])[:, sl] for n in 'qkv')`, `scores = q @ k.T / math.sqrt(${A.dh})`, ...(A.causal ? [`scores = scores.masked_fill(torch.ones(${A.T}, ${A.T}, dtype=torch.bool).triu(1), float('-inf'))`] : []), `weights = scores.softmax(-1)                       # row ${i}: where ${tok(i)} looks`],
        about: [`Head ${h + 1}'s output number ${c} for ${tok(i)}: the values of the ${n} position${n > 1 ? 's' : ''} it can see, each multiplied by its attention weight and added up. The weights come from a softmax, so they are positive and add up to 1: the result is a weighted average, leaning toward the positions whose keys matched the query best. They are not learned parameters: they were computed a moment ago from this example.`],
      });
    }
    function queryLevel(A, h, i, jHi) {
      const n = A.causal ? i + 1 : A.T, dh = A.dh, scores = Array.from({ length: n }, (_, j) => { let s = 0; for (let c = 0; c < dh; c++) s += qkv(A, 'q', i, h, c) * qkv(A, 'k', j, h, c); return s / Math.sqrt(dh); });
      const w = Float32Array.from({ length: n }, (_, j) => headW(A, h, i, j)), o = Float32Array.from({ length: dh }, (_, c) => qkv(A, 'o', i, h, c)), q = Float32Array.from({ length: dh }, (_, c) => qkv(A, 'q', i, h, c));
      const LY = W2({ t: [0, 0, 1, 0.13], q: [0, 0.24, 0.26, 0.14], s: [0.34, 0.24, 0.5, 0.13], w: [0.34, 0.45, 0.5, 0.13], v: [0.34, 0.68, 0.5, 0.3], o: [0.88, 0.68, 0.12, 0.3] },
        { t: [0, 0, 1, 0.14], q: [0, 0.2, 0.82, 0.09], s: [0, 0.35, 0.82, 0.09], w: [0, 0.5, 0.82, 0.09], v: [0, 0.69, 0.82, 0.29], o: [0.87, 0.69, 0.13, 0.29] });
      const top = Array.from({ length: n }, (_, j) => j).sort((a, b) => w[b] - w[a]);
      return {
        title: `Head ${h + 1} · query ${tok(i)}`, sub: `a weighted average of ${n} value${n > 1 ? 's' : ''}`, tone: 's7',
        summary: (ctx, r, S) => S.bars(ctx, w, r, { positive: true, max: 1, color: () => theme.c('s7', 0.8) }),
        inner: (ctx, r, S) => {
          const u = S.u(r, 8, 13), at = (k) => S.place(r, LY[k]);
          S.lines(ctx, [`query ${tok(i)} (position ${i}) looks at ${A.causal ? `positions 0 to ${i}` : `all ${A.T} positions`}`, `w = softmax(q·k / √${dh});  o = Σ w · v`, `most weight: ${top.slice(0, 3).map((j) => `${tok(j)} ${fmtPct(w[j], 0)}`).join(', ')}`], at('t'), { mono: true, size: u * 0.9, strong: (k) => k === 0 });
          const qr = at('q'), sr = at('s'), wr = at('w'), vr = at('v'), or = at('o');
          S.caption(ctx, `query q (${dh} numbers)`, qr, { size: u * 0.8, minW: 110 }); S.bars(ctx, q, qr);
          S.caption(ctx, 'scores q·k / √d, one per position seen', sr, { size: u * 0.8, minW: 200 }); S.bars(ctx, Float32Array.from(scores), sr, { color: (j) => (j === jHi ? theme.c('focus') : theme.divColor(clamp(scores[j] / maxAbs(scores), -1, 1) * 0.85 + Math.sign(scores[j]) * 0.15)) });
          S.caption(ctx, 'softmax → weights (they add up to 1)', wr, { size: u * 0.8, minW: 200 }); S.bars(ctx, w, wr, { positive: true, max: 1, color: (j) => (j === jHi ? theme.c('focus') : theme.c('s7', 0.85)) });
          if (P.ids) colLabels(ctx, S, Array.from({ length: n }, (_, j) => word(P.ids[j])), wr, { size: u * 0.72, hi: jHi });
          S.caption(ctx, `values v, one column per position (${dh} × ${n})`, vr, { size: u * 0.8, minW: 200 });
          drawHeat(ctx, heatImg(cache, `qv${A.bi}:${h}:${i}`, dh, n, (c, j) => qkv(A, 'v', j, h, c)), vr, dh, n);
          S.caption(ctx, 'o = Σ w·v', or, { size: u * 0.8, minW: 50 }); hbars(ctx, o, or);
        },
        grids: [
          { id: 'q', rect: LY.q, rows: 1, cols: dh, make: (a, c) => projNeuron(A, 'q', i, h * dh + c) },
          { id: 's', rect: LY.s, rows: 1, cols: n, make: (a, j) => scoreDot(A, h, i, j) },
          { id: 'w', rect: LY.w, rows: 1, cols: n, make: (a, j) => Object.assign(Dp.lib(X, 'softmax', { scores, j, label: 'w' }), { title: `Weight of ${tok(j)} for ${tok(i)}`, sub: fmtPct(w[j], 1) }) },
          { id: 'v', rect: LY.v, rows: dh, cols: n, make: (c, j) => projNeuron(A, 'v', j, h * dh + c) },
          { id: 'o', rect: LY.o, rows: dh, cols: 1, make: (c) => headOut(A, h, i, c) },
        ],
        about: [
          `One query position, ${tok(i)}, in head ${h + 1}. Its query vector q is compared with the key of every position it can see${A.causal ? ` (positions 0 to ${i}: the causal mask hides the ones after it)` : ''}: each score is the dot product q·k divided by √${dh}. The softmax turns the scores into weights that are positive and add up to 1, and the head's output for this position is the weighted average of the value vectors: o = Σ wⱼ vⱼ.`,
          `Here ${tok(top[0])} gets ${fmtPct(w[top[0]], 0)} of the weight${n > 1 ? `, ${tok(top[1])} ${fmtPct(w[top[1]], 0)}` : ''}. Zoom into a score for its dot product, a weight for its softmax, a value for the projection that made it, or an output bar for the weighted sum.`,
        ],
        code: `${pyX(A)}
B, T, D = x.shape
split = lambda t: t.view(B, T, ${A.H}, ${dh}).transpose(1, 2)
q, k, v = split(${A.mod}.q(x)), split(${A.mod}.k(x)), split(${A.mod}.v(x))
s = q[0, ${h}, ${i}] @ k[0, ${h}, :${n}].T / math.sqrt(${dh})          # ${n} scores for query ${i}
w = s.softmax(-1)                                  # weights: they add up to 1
o = w @ v[0, ${h}, :${n}]                           # (${dh},): the weighted average of the values`,
      };
    }
    function headLevel(A, h) {
      const { T, dh, causal } = A, last = T - 1;
      const score = (i, j) => { let s = 0; for (let c = 0; c < dh; c++) s += qkv(A, 'q', i, h, c) * qkv(A, 'k', j, h, c); return s / Math.sqrt(dh); };
      const LY = W2({ lab: [0, 0.07, 0.07, 0.38], q: [0.08, 0.07, 0.12, 0.38], k: [0.22, 0.07, 0.12, 0.38], v: [0.36, 0.07, 0.12, 0.38], m: [0.53, 0.02, 0.47, 0.44], s: [0.08, 0.56, 0.3, 0.42], w: [0.44, 0.56, 0.3, 0.42], o: [0.8, 0.56, 0.2, 0.42] },
        { lab: [0, 0.05, 0.12, 0.25], q: [0.13, 0.05, 0.27, 0.25], k: [0.43, 0.05, 0.27, 0.25], v: [0.73, 0.05, 0.27, 0.25], m: [0, 0.34, 1, 0.2], s: [0, 0.6, 0.4, 0.38], w: [0.44, 0.6, 0.4, 0.38], o: [0.87, 0.6, 0.13, 0.38] });
      const labels = Array.from({ length: T }, (_, t) => (P.ids ? word(P.ids[t]) : String(t)));
      const keys = Array.from({ length: causal ? last + 1 : T }, (_, j) => j).sort((a, b) => headW(A, h, last, b) - headW(A, h, last, a));
      return {
        title: `Head ${h + 1}`, sub: `${dh} numbers per position`, tone: 's7',
        summary: (ctx, r) => { const s = Math.min(r.w, r.h); drawHeat(ctx, heatImg(cache, `att${A.bi}:${h}`, T, T, (i, j) => headW(A, h, i, j), { seq: true, max: 1, mask: (i, j) => causal && j > i }), { x: r.x + (r.w - s) / 2, y: r.y, w: s, h: s }, T, T, true); },
        inner: (ctx, r, S) => {
          const u = S.u(r, 8, 13), at = (k) => S.place(r, LY[k]);
          const qr = at('q');
          [['q = x·Wq', 'q', qr], ['k = x·Wk', 'k', at('k')], ['v = x·Wv', 'v', at('v')]].forEach(([t, which, br]) => { S.caption(ctx, t, br, { size: u * 0.8, minW: 60 }); drawHeat(ctx, heatImg(cache, `h${which}${A.bi}:${h}`, T, dh, (i, c) => qkv(A, which, i, h, c)), br, T, dh); });
          rowLabels(ctx, S, labels, { x: qr.x, y: qr.y, w: 0, h: qr.h }, { size: u * 0.8, maxW: at('lab').w });
          const sr = at('s'), wr = at('w'), or = at('o');
          S.caption(ctx, `scores q·k / √${dh}`, sr, { size: u * 0.8, minW: 90 });
          const g = drawHeat(ctx, heatImg(cache, `sc${A.bi}:${h}`, T, T, (i, j) => (causal && j > i ? 0 : score(i, j)), { mask: (i, j) => causal && j > i }), sr, T, T, true);
          S.caption(ctx, 'weights: zoom into a row', wr, { size: u * 0.8, minW: 90 });
          drawHeat(ctx, heatImg(cache, `att${A.bi}:${h}`, T, T, (i, j) => headW(A, h, i, j), { seq: true, max: 1, mask: (i, j) => causal && j > i }), wr, T, T, true);
          S.arrow(ctx, sr.x + g.w + 3, sr.y + g.h / 2, wr.x - 4, wr.y + g.h / 2, { lw: 1, head: 5 });
          S.caption(ctx, 'weights · v', or, { size: u * 0.8, minW: 60 });
          drawHeat(ctx, heatImg(cache, `ho${A.bi}:${h}`, T, dh, (i, c) => qkv(A, 'o', i, h, c)), or, T, dh);
          if (causal) S.text(ctx, 'grey: later positions, hidden by the causal mask', sr.x, sr.y + g.h + u * 0.9, { size: u * 0.75, color: theme.c('muted'), maxW: r.w * 0.8 });
          const lines = [`the last query, ${tok(last)}:`, ...keys.slice(0, 5).map((j) => `${word(P.ids ? P.ids[j] : j).padEnd(6).slice(0, 8)} s = ${fmt(score(last, j), 2).padStart(6)}  w = ${fmtPct(headW(A, h, last, j), 0).padStart(4)}`)];
          if (keys.length > 5) lines.push(`… ${keys.length - 5} more share ${fmtPct(keys.slice(5).reduce((s, j) => s + headW(A, h, last, j), 0), 0)}`);
          S.lines(ctx, lines, at('m'), { mono: true, size: u * 0.88, strong: (k) => k === 0 });
        },
        grids: [
          ...['q', 'k', 'v'].map((which) => ({ id: which, rect: LY[which], rows: T, cols: dh, make: (i, c) => projNeuron(A, which, i, h * dh + c) })),
          { id: 's', rect: LY.s, rows: T, cols: T, square: true, skip: (i, j) => causal && j > i, make: (i, j) => (causal && j > i ? null : scoreDot(A, h, i, j)) },
          { id: 'w', rect: LY.w, rows: T, cols: T, square: true, skip: (i, j) => causal && j > i, make: (i, j) => (causal && j > i ? null : queryLevel(A, h, i, j)) },
          { id: 'o', rect: LY.o, rows: T, cols: dh, make: (i, c) => headOut(A, h, i, c) },
        ],
        about: [
          `Head ${h + 1} works in its own ${dh}-number slice of the vectors. Every position makes a query q, a key k and a value v from its ${xName(A)} (the three grids at the top). The score between query i and key j is their dot product divided by √${dh}${causal ? '; scores for later positions are hidden (grey), so a position only sees the ones before it' : ''}. A softmax turns each row of scores into weights that add up to 1, and each position's output is the weighted average of the values.`,
          `For the last position, ${tok(last)}, this head puts ${fmtPct(headW(A, h, last, keys[0]), 0)} of its weight on ${tok(keys[0])}. Zoom into a row of the weights for that query's weighted average, a score for its dot product, or any number of q, k and v for the projection that made it.`,
        ],
        code: `${pyX(A)}
B, T, D = x.shape
split = lambda t: t.view(B, T, ${A.H}, ${dh}).transpose(1, 2)        # (1, heads, ${T}, ${dh})
q, k, v = split(${A.mod}.q(x)), split(${A.mod}.k(x)), split(${A.mod}.v(x))
scores = q[:, ${h}] @ k[:, ${h}].transpose(-2, -1) / math.sqrt(${dh})     # (1, ${T}, ${T})${causal ? `
scores = scores.masked_fill(torch.ones(T, T, dtype=torch.bool).triu(1), float("-inf"))` : ''}
weights = scores.softmax(-1)                       # each row adds up to 1
out = weights @ v[:, ${h}]                          # (1, ${T}, ${dh}): what each position collects`,
      };
    }
    // the attention layer: x → q, k, v → heads → side by side → Wo (→ + x)
    function attnLevel(A, opts = {}) {
      const { T, D, H, dh, causal } = A, res = !opts.noResidual && A.lay.residual;
      const labels = Array.from({ length: T }, (_, t) => (P.ids ? word(P.ids[t]) : String(t)));
      // x on the left, its three projections stacked beside it, the heads beside those; then the heads side by
      // side, Wo and (for a residual) the sum along the bottom
      const LY = W2({ t: [0, 0, 1, 0.07], x: [0.06, 0.14, 0.17, 0.42], q: [0.3, 0.12, 0.17, 0.11], k: [0.3, 0.29, 0.17, 0.11], v: [0.3, 0.46, 0.17, 0.11], heads: [0.54, 0.14, 0.46, 0.42], o: [0.06, 0.7, 0.24, 0.27], y: [0.39, 0.7, 0.24, 0.27], out: [0.72, 0.7, 0.24, 0.27] },
        { t: [0, 0, 1, 0.07], x: [0.12, 0.12, 0.36, 0.25], q: [0.6, 0.11, 0.4, 0.06], k: [0.6, 0.21, 0.4, 0.06], v: [0.6, 0.31, 0.4, 0.06], heads: [0, 0.46, 1, 0.21], o: [0.12, 0.78, 0.26, 0.2], y: [0.42, 0.78, 0.26, 0.2], out: [0.72, 0.78, 0.26, 0.2] });
      const hl = tileLay(H, H, LY.heads[2] * size.w, LY.heads[3] * size.h), hcw = LY.heads[2] / hl.cols, hch = LY.heads[3] / hl.rows;
      const headRect = (h) => [LY.heads[0] + (h % hl.cols) * hcw + hcw * 0.05, LY.heads[1] + Math.floor(h / hl.cols) * hch + hch * 0.05, hcw * 0.9, hch * 0.9];
      const yv = A.r.y, outv = res ? A.r.out : null;
      return Dp.lazy({
        title: opts.title || `Attention`, sub: `${H} head${H > 1 ? 's' : ''} × ${dh} numbers${causal ? ' · causal' : ''}`, tone: 's7',
        summary: (ctx, r) => { const s = Math.max(4, Math.min(r.h, r.w / H - 3)); for (let h = 0; h < H; h++) drawHeat(ctx, heatImg(cache, `att${A.bi}:${h}`, T, T, (i, j) => headW(A, h, i, j), { seq: true, max: 1, mask: (i, j) => causal && j > i }), { x: r.x + (r.w - H * (s + 3)) / 2 + h * (s + 3), y: r.y, w: s, h: s }, T, T, true); },
        inner: (ctx, r, S) => {
          const u = S.u(r, 8, 13), at = (k) => S.place(r, LY[k]);
          S.lines(ctx, [`per head: weights = softmax(q·kᵀ / √${dh}${causal ? ' + causal mask' : ''});  out = weights · v`], at('t'), { size: u * 0.88, mono: true, color: theme.c('ink-2') });
          const xr = at('x'), hr = at('heads'), orr = at('o'), yr = at('y'), kr = at('k'), vr = at('v');
          S.caption(ctx, xName(A), xr, { size: u * 0.8, minW: 80 });
          drawHeat(ctx, heatImg(cache, `ax:${A.bi}`, T, D, (i, j) => A.xin[i * D + j]), xr, T, D);
          rowLabels(ctx, S, labels, xr, { size: u * 0.78, maxW: Math.max(20, xr.x - r.x - 2) });
          [['Q = x·Wqᵀ + bq', 'q'], ['K = x·Wkᵀ + bk', 'k'], ['V = x·Wvᵀ + bv', 'v']].forEach(([t, which]) => {
            const pr = at(which);
            S.caption(ctx, t, pr, { size: u * 0.8, minW: 80 });
            drawHeat(ctx, heatImg(cache, `a${which}:${A.bi}`, T, D, (i, j) => A.r[which][i * D + j]), pr, T, D);
            // the same x makes all three
            S.arrow(ctx, xr.x + xr.w + 3, xr.y + xr.h / 2, pr.x - 4, pr.y + pr.h / 2, { lw: 0.9, head: 4 });
          });
          S.caption(ctx, `the ${H} head${H > 1 ? 's' : ''}: open one`, hr, { size: u * 0.8, minW: 120 });
          ctx.strokeStyle = theme.c('ink-2'); ctx.lineWidth = 1;
          if (wide) {
            S.arrow(ctx, kr.x + kr.w + 3, kr.y + kr.h / 2, hr.x - 4, kr.y + kr.h / 2, { lw: 0.9, head: 4 });
            // down from the heads, back to the left, into the concatenation
            const ym = (hr.y + hr.h + orr.y) / 2, hx = hr.x + hr.w / 2, ox2 = orr.x + orr.w / 2;
            ctx.beginPath(); ctx.moveTo(hx, hr.y + hr.h + 2); ctx.lineTo(hx, ym); ctx.lineTo(ox2, ym); ctx.stroke();
            S.arrow(ctx, ox2, ym, ox2, orr.y - 3, { lw: 0.9, head: 4 });
          } else {
            S.arrow(ctx, vr.x + vr.w / 2, vr.y + vr.h + 2, vr.x + vr.w / 2, hr.y - u * 1.5, { lw: 0.9, head: 4 });
            S.arrow(ctx, orr.x + orr.w / 2, hr.y + hr.h + 2, orr.x + orr.w / 2, orr.y - u * 1.4, { lw: 0.9, head: 4 });
          }
          S.caption(ctx, H > 1 ? 'heads side by side' : 'the head’s output', orr, { size: u * 0.8, minW: 80 });
          drawHeat(ctx, heatImg(cache, `ao:${A.bi}`, T, D, (i, j) => A.r.o[i * D + j]), orr, T, D);
          rowLabels(ctx, S, labels, orr, { size: u * 0.78, maxW: Math.max(20, orr.x - r.x - 2) });
          S.caption(ctx, '× Woᵀ + bo', yr, { size: u * 0.8, minW: 70 });
          drawHeat(ctx, heatImg(cache, `ay:${A.bi}`, T, D, (i, j) => yv[i * D + j]), yr, T, D);
          S.arrow(ctx, orr.x + orr.w + 3, orr.y + orr.h / 2, yr.x - 4, yr.y + yr.h / 2, { lw: 0.9, head: 4 });
          if (res) {
            const qr = at('out');
            S.caption(ctx, '+ x: the residual', qr, { size: u * 0.8, minW: 80 });
            drawHeat(ctx, heatImg(cache, `aout:${A.bi}`, T, D, (i, j) => outv[i * D + j]), qr, T, D);
            S.arrow(ctx, yr.x + yr.w + 3, yr.y + yr.h / 2, qr.x - 4, qr.y + qr.h / 2, { lw: 0.9, head: 4 });
          }
        },
        grids: [
          { id: 'x', rect: LY.x, rows: T, cols: D, make: (i, j) => (A.inTf ? tfLnCell(A.bi, 1, i, j) : valueNode(A.bi - 1, i * D + j)) },
          ...['q', 'k', 'v'].map((which) => ({ id: which, rect: LY[which], rows: T, cols: D, make: (i, j) => projNeuron(A, which, i, j) })),
          { id: 'o', rect: LY.o, rows: T, cols: D, make: (i, j) => headOut(A, Math.floor(j / dh), i, j % dh) },
          { id: 'y', rect: LY.y, rows: T, cols: D, make: (i, j) => projNeuron(A, 'o', i, j) },
          ...(res ? [{ id: 'out', rect: LY.out, rows: T, cols: D, make: (i, j) => attnValue(A.bi, i, j) }] : []),
        ],
        about: [
          `Self-attention is how positions share information. From its ${xName(A)} every position makes three vectors with learned matrices: a query (what it is looking for), a key (what it offers) and a value (what it passes on). Each query is compared with the keys of ${causal ? 'the positions up to its own (the causal mask hides later ones)' : 'every position'}; the better the match, the more of that position's value is copied.`,
          `The ${D} numbers are split into ${H} head${H > 1 ? 's' : ''} of ${dh}, each with its own slice of the matrices, so the layer can look for ${H > 1 ? `${H} things at once` : 'one thing'}. The heads' outputs are placed side by side and mixed by the output matrix Wo${res ? ', and the result is added to the input (a residual connection)' : ''}. Open a head for its scores and weights on this example; any number opens what computed it.`,
        ],
        code: A.inTf ? `${pyX(A)}\ny = ${A.mod}(x)                                 # SelfAttention(${D}, heads=${H}, causal=${causal ? 'True' : 'False'})` : `attn = ${A.mod}                                 # SelfAttention(${D}, heads=${H}, causal=${causal ? 'True' : 'False'})
${pyX(A)}
B, T, D = x.shape
split = lambda t: t.view(B, T, ${H}, ${dh}).transpose(1, 2)
q, k, v = split(attn.q(x)), split(attn.k(x)), split(attn.v(x))      # (1, ${H}, ${T}, ${dh})
scores = q @ k.transpose(-2, -1) / math.sqrt(${dh})${causal ? `
scores = scores.masked_fill(torch.ones(T, T, dtype=torch.bool).triu(1), float("-inf"))` : ''}
w = scores.softmax(-1)                             # (1, ${H}, ${T}, ${T}): each row adds up to 1
y = attn.o((w @ v).transpose(1, 2).reshape(B, T, D))
print(torch.allclose(y, attn(x), atol=1e-6))       # True${res ? '\nout = x + y                                        # the residual' : ''}`,
      }, () => Array.from({ length: H }, (_, h) => Object.assign(headLevel(A, h), { id: 'h' + h, rect: headRect(h) })));
    }
    function attnValue(bi, t, j) {
      const A = attnData(bi), idx = t * A.D + j;
      if (!A.lay.residual) return projNeuron(A, 'o', t, j);
      return sumNode(`${tok(t)}[${j}] after attention`, 'out = x + attention(x)', A.xin[idx], A.r.y[idx], () => valueNode(bi - 1, idx), () => projNeuron(A, 'o', t, j), residualAbout('what attention found'), `x = torch.randn(1, ${A.T}, ${A.D})\nout = x + ${A.mod}(x)                       # the residual: same shape in and out`);
    }

    // ---- transformer blocks: LN1 → attention → + → LN2 → MLP → +
    function tfLnCell(bi, which, t, j) {
      const lay = model.layers[bi], r = P.layers[bi], D = lay.D, src = which === 1 ? inVals(bi) : r.mid, g = which === 1 ? lay.ln1g.w : lay.ln2g.w, b = which === 1 ? lay.ln1b.w : lay.ln2b.w;
      return Object.assign(Dp.lib(X, 'layernorm', { x: Array.from(src.subarray(t * D, t * D + D)), j, g: g[j], b: b[j] }), { title: `LN${which} of block ${bi + 1}, number ${j} of ${tok(t)}` });
    }
    function mlpHidden(bi, t, j) {
      const lay = model.layers[bi], r = P.layers[bi], D = lay.D, Hd = lay.Hd, mod = pyName(bi);
      return Dp.neuron(X, {
        title: `MLP neuron ${j} at ${tok(t)}`, sub: `u = x̂·W1 row + b1, then GELU`, tone: 's7',
        terms: Array.from({ length: D }, (_, p) => ({ a: r.h2[t * D + p], w: lay.W1.w[j * D + p] })), bias: lay.b1.w[j], act: actOf('gelu'), zName: 'u', outName: 'g',
        inLabel: (p) => `x̂[${p}] of ${tok(t)}`, wLabel: (p) => `W1[${j}, ${p}]`, inputNode: (p) => tfLnCell(bi, 2, t, p),
        codeHead: [...pyGrab(`${mod}.mlp`, 'x', 'inp[0][0]'), `x_hat = grab['x'][${t}]                             # ${tok(t)}'s ${D} numbers after LN2`],
        vars: { w: `${mod}.mlp[0].weight[${j}]`, a: 'x_hat', b: `${mod}.mlp[0].bias[${j}]` },
        about: [`Hidden neuron ${j} of the MLP in block ${bi + 1} (one of ${Hd}), at ${tok(t)}: a weighted sum of that position's ${D} normalised numbers, a bias, then GELU, a smooth ReLU.`],
      });
    }
    function mlpOut(bi, t, j) {
      const lay = model.layers[bi], r = P.layers[bi], Hd = lay.Hd, mod = pyName(bi);
      return Dp.neuron(X, {
        title: `MLP output ${j} at ${tok(t)}`, sub: `g · W2 row ${j} + b2`, tone: 's7',
        terms: Array.from({ length: Hd }, (_, p) => ({ a: r.g[t * Hd + p], w: lay.W2.w[j * Hd + p] })), bias: lay.b2.w[j], act: null, zName: 'm',
        inLabel: (p) => `hidden neuron ${p}`, wLabel: (p) => `W2[${j}, ${p}]`, inputNode: (p) => mlpHidden(bi, t, p),
        codeHead: [...pyGrab(`${mod}.mlp[2]`, 'g', 'inp[0][0]'), `g = grab['g'][${t}]                                 # ${tok(t)}'s ${Hd} hidden values, after GELU`],
        vars: { w: `${mod}.mlp[2].weight[${j}]`, a: 'g', b: `${mod}.mlp[2].bias[${j}]` },
        about: [`What the MLP adds to number ${j} of ${tok(t)}: all ${Hd} hidden neurons' outputs, each weighted by row ${j} of W2, plus a bias. Zoom into an input bar to open the hidden neuron behind it.`],
      });
    }
    function tfMid(bi, t, j) {
      const r = P.layers[bi], D = model.layers[bi].D, idx = t * D + j, A = attnData(bi);
      return sumNode(`${tok(t)}[${j}] after attention`, 'mid = x + attention(LN1(x))', inVals(bi)[idx], r.y[idx], () => valueNode(bi - 1, idx), () => projNeuron(A, 'o', t, j), residualAbout('what attention found'), `block, x = ${pyName(bi)}, torch.randn(1, ${A.T}, ${D})\nmid = x + block.attn(block.norm1(x))`);
    }
    function tfValue(bi, t, j) {
      const r = P.layers[bi], D = model.layers[bi].D, idx = t * D + j;
      return sumNode(`${tok(t)}[${j}] after block ${bi + 1}`, 'out = mid + MLP(LN2(mid))', r.mid[idx], r.mo[idx], () => tfMid(bi, t, j), () => mlpOut(bi, t, j), residualAbout('what the MLP computed'), `block, mid = ${pyName(bi)}, torch.randn(1, ${model.layers[bi].T}, ${D})\nout = mid + block.mlp(block.norm2(mid))`);
    }
    // a LayerNorm over the rows of a T × D sequence, or over one vector (rows = 1)
    function lnLevel(o) {
      const { key, rows: T, d: D, x, y, g, b } = o, labels = Array.from({ length: T }, (_, t) => (P.ids && T === T0 ? word(P.ids[t]) : String(t)));
      const stats = Array.from({ length: T }, (_, t) => { let mu = 0; for (let c = 0; c < D; c++) mu += x[t * D + c]; mu /= D; let v = 0; for (let c = 0; c < D; c++) v += (x[t * D + c] - mu) ** 2; v /= D; return { mu, v, sd: Math.sqrt(v + 1e-5) }; });
      const vec = T === 1, xh = Float32Array.from({ length: T * D }, (_, i) => (x[i] - stats[Math.floor(i / D)].mu) / stats[Math.floor(i / D)].sd);
      const LY = vec ? W2({ t: [0, 0, 1, 0.15], x: [0, 0.25, 0.45, 0.22], xh: [0.55, 0.25, 0.45, 0.22], g: [0, 0.62, 0.3, 0.16], b: [0.35, 0.62, 0.3, 0.16], y: [0.7, 0.62, 0.3, 0.16] },
        { t: [0, 0, 1, 0.16], x: [0, 0.22, 1, 0.12], xh: [0, 0.41, 1, 0.12], g: [0, 0.6, 0.48, 0.12], b: [0.52, 0.6, 0.48, 0.12], y: [0, 0.82, 1, 0.14] })
        : W2({ t: [0, 0, 1, 0.12], x: [0.07, 0.21, 0.27, 0.44], xh: [0.385, 0.21, 0.27, 0.44], y: [0.7, 0.21, 0.3, 0.44], s: [0, 0.72, 0.56, 0.28], g: [0.62, 0.74, 0.17, 0.2], b: [0.83, 0.74, 0.17, 0.2] },
          { t: [0, 0, 1, 0.1], x: [0.13, 0.15, 0.87, 0.15], xh: [0.13, 0.35, 0.87, 0.15], y: [0.13, 0.55, 0.87, 0.15], g: [0, 0.76, 0.48, 0.08], b: [0.52, 0.76, 0.48, 0.08], s: [0, 0.88, 1, 0.12] });
      const st0 = stats[0];
      return {
        title: o.title, sub: vec ? 'mean 0, spread 1' : 'per position', tone: o.tone || 's1',
        summary: (ctx, r, S) => { if (vec) S.bars(ctx, y, r); else drawHeat(ctx, heatImg(cache, `${key}y`, T, D, (i, j) => y[i * D + j]), r, T, D); },
        inner: (ctx, r, S) => {
          const u = S.u(r, 8, 13), at = (k) => S.place(r, LY[k]);
          S.lines(ctx, ['y = (x − μ) / √(σ² + ε) · γ + β', vec ? `μ = ${fmt(st0.mu, 4)}   σ² = ${fmt(st0.v, 4)}   √(σ² + ε) = ${fmt(st0.sd, 4)}` : `μ and σ are taken over each position's ${D} numbers`], at('t'), { mono: true, size: u * 0.92, strong: (k) => k === 0 });
          if (vec) {
            [['x: the input', x, 'x'], ['x̂ = (x − μ) / σ: mean 0, spread 1', xh, 'xh'], ['gain γ (learned)', g, 'g'], ['bias β (learned)', b, 'b'], ['y = x̂ · γ + β', y, 'y']].forEach(([t, v, k2]) => { const q = at(k2); S.caption(ctx, t, q, { size: u * 0.8, minW: 120 }); S.bars(ctx, v, q); });
            return;
          }
          const xr = at('x'), yr = at('y'), hr = at('xh');
          S.caption(ctx, 'x: each row one position', xr, { size: u * 0.8, minW: 120 }); drawHeat(ctx, heatImg(cache, `${key}x`, T, D, (i, j) => x[i * D + j]), xr, T, D);
          rowLabels(ctx, S, labels, xr, { size: u * 0.78, maxW: Math.max(20, xr.x - r.x - 2) });
          S.caption(ctx, 'x̂ = (x − μ) / σ, per row', hr, { size: u * 0.8, minW: 100 }); drawHeat(ctx, heatImg(cache, `${key}h`, T, D, (i, j) => xh[i * D + j]), hr, T, D);
          S.caption(ctx, 'y = x̂ · γ + β', yr, { size: u * 0.8, minW: 100 }); drawHeat(ctx, heatImg(cache, `${key}y`, T, D, (i, j) => y[i * D + j]), yr, T, D);
          S.caption(ctx, 'γ', at('g'), { size: u * 0.8 }); S.bars(ctx, g, at('g'));
          S.caption(ctx, 'β', at('b'), { size: u * 0.8 }); S.bars(ctx, b, at('b'));
          const shown = Math.min(T, wide ? 6 : 3);
          S.lines(ctx, [...stats.slice(0, shown).map((q, t) => `${labels[t].padEnd(6).slice(0, 8)} μ = ${fmt(q.mu, 3).padStart(6)}  σ² = ${fmt(q.v, 3)}`), ...(T > shown ? [`… and ${T - shown} more positions`] : [])], at('s'), { mono: true, size: u * 0.82 });
        },
        grids: [
          { id: 'x', rect: LY.x, rows: T, cols: D, make: (i, j) => o.src(i, j) },
          { id: 'y', rect: LY.y, rows: T, cols: D, make: (i, j) => o.cell(i, j) },
          { id: 'xh', rect: LY.xh, rows: T, cols: D, make: (i, j) => o.cell(i, j) },
          { id: 'g', rect: LY.g, rows: 1, cols: D, make: (i, j) => Dp.stored(X, g[j], `γ[${j}], a learned gain`) },
          { id: 'b', rect: LY.b, rows: 1, cols: D, make: (i, j) => Dp.stored(X, b[j], `β[${j}], a learned bias`) },
        ],
        about: [
          `LayerNorm rescales ${vec ? 'the vector' : 'each position\'s vector on its own'}: subtract its mean μ, divide by its standard deviation σ (with a tiny ε = 0.00001 inside the square root, so it never divides by 0), then multiply each number by a learned gain γ and add a learned bias β: y = (x − μ) / √(σ² + ε) · γ + β.${vec ? ` Here μ = ${fmt(st0.mu, 3)} and σ = ${fmt(st0.sd, 3)}.` : ''}`,
          'Whatever the layers before it do, the numbers leave with a steady scale, which keeps training stable in deep stacks; γ and β let the model undo that where it helps. Zoom into a number of y for the arithmetic of one normalisation.',
        ],
        code: o.code,
      };
    }
    function mlpLevel(bi) {
      const lay = model.layers[bi], r = P.layers[bi], T = lay.T, D = lay.D, Hd = lay.Hd, mod = pyName(bi), labels = Array.from({ length: T }, (_, t) => (P.ids ? word(P.ids[t]) : String(t)));
      const LY = W2({ t: [0, 0, 1, 0.08], x: [0.06, 0.16, 0.2, 0.34], u: [0.32, 0.16, 0.42, 0.34], c: [0.8, 0.16, 0.2, 0.34], g: [0.06, 0.62, 0.48, 0.34], o: [0.62, 0.62, 0.3, 0.34] },
        { t: [0, 0, 1, 0.08], x: [0.12, 0.14, 0.36, 0.17], c: [0.56, 0.14, 0.44, 0.17], u: [0.12, 0.39, 0.88, 0.17], g: [0.12, 0.62, 0.88, 0.16], o: [0.12, 0.84, 0.5, 0.15] });
      return {
        title: 'MLP', sub: `${D} → ${Hd} → ${D} at every position`, tone: 's7',
        summary: (ctx, q) => drawHeat(ctx, heatImg(cache, `mg${bi}`, T, Hd, (i, j) => r.g[i * Hd + j]), q, T, Hd),
        inner: (ctx, q, S) => {
          const u = S.u(q, 8, 13), at = (k) => S.place(q, LY[k]);
          S.lines(ctx, ['MLP(x̂) = W2 · GELU(W1 · x̂ + b1) + b2, at each position on its own'], at('t'), { size: u * 0.88, mono: true, color: theme.c('ink-2') });
          const xr = at('x'), ur = at('u'), gr = at('g'), or = at('o'), cr = at('c');
          S.caption(ctx, `x̂ = LN2(mid) (${D})`, xr, { size: u * 0.8, minW: 90 }); drawHeat(ctx, heatImg(cache, `mx${bi}`, T, D, (i, j) => r.h2[i * D + j]), xr, T, D);
          rowLabels(ctx, S, labels, xr, { size: u * 0.78, maxW: Math.max(20, xr.x - q.x - 2) });
          S.caption(ctx, `u = W1 · x̂ + b1   (${Hd} neurons)`, ur, { size: u * 0.8, minW: 150 }); drawHeat(ctx, heatImg(cache, `mu${bi}`, T, Hd, (i, j) => r.u[i * Hd + j]), ur, T, Hd);
          S.caption(ctx, 'GELU(u) = u · Φ(u)', cr, { size: u * 0.8, minW: 90 }); S.frame(ctx, cr); S.curve(ctx, ACT('gelu').f, cr, -3, 3, -0.5, 3, { color: theme.c('s7'), lw: 1.8 });
          S.caption(ctx, 'g = GELU(u)', gr, { size: u * 0.8, minW: 90 }); drawHeat(ctx, heatImg(cache, `mg${bi}`, T, Hd, (i, j) => r.g[i * Hd + j]), gr, T, Hd);
          if (!wide) rowLabels(ctx, S, labels, gr, { size: u * 0.78, maxW: Math.max(20, gr.x - q.x - 2) });
          S.caption(ctx, `out = W2 · g + b2   (${D})`, or, { size: u * 0.8, minW: 130 }); drawHeat(ctx, heatImg(cache, `mo${bi}`, T, D, (i, j) => r.mo[i * D + j]), or, T, D);
        },
        grids: [
          { id: 'x', rect: LY.x, rows: T, cols: D, make: (i, j) => tfLnCell(bi, 2, i, j) },
          { id: 'u', rect: LY.u, rows: T, cols: Hd, make: (i, j) => mlpHidden(bi, i, j) },
          { id: 'g', rect: LY.g, rows: T, cols: Hd, make: (i, j) => mlpHidden(bi, i, j) },
          { id: 'o', rect: LY.o, rows: T, cols: D, make: (i, j) => mlpOut(bi, i, j) },
        ],
        about: [
          `The MLP (feed-forward network) works on each position on its own, with no mixing between them: it widens the ${D} numbers to ${Hd} with W1, bends them with GELU, and projects back to ${D} with W2. Attention gathers information from other positions; the MLP then computes with what each position has.`,
          'GELU(u) = u·Φ(u), where Φ is the normal distribution\'s cumulative curve: a smooth ReLU that passes large positive values, damps negative ones and has a slope everywhere. Zoom into any number for the neuron that computed it.',
        ],
        code: `mlp = ${mod}.mlp                                # Linear(${D}, ${Hd}), GELU, Linear(${Hd}, ${D})
mid = torch.randn(1, ${T}, ${D})                    # the stream after attention
x_hat = ${mod}.norm2(mid)                          # (1, ${T}, ${D})
u = mlp[0](x_hat)                                  # (1, ${T}, ${Hd}): one weighted sum per hidden neuron
g = F.gelu(u, approximate="tanh")
out = mlp[2](g)                                    # (1, ${T}, ${D})
print(torch.allclose(out, mlp(x_hat), atol=1e-6))  # True`,
      };
    }
    function addLevel(bi, k) {
      const lay = model.layers[bi], r = P.layers[bi], T = lay.T, D = lay.D, before = k === 1 ? inVals(bi) : r.mid, delta = k === 1 ? r.y : r.mo, after = k === 1 ? r.mid : r.out, what = k === 1 ? 'attention' : 'MLP';
      const G = [[0.1, 0.06, 0.9, 0.24], [0.1, 0.39, 0.9, 0.24], [0.1, 0.72, 0.9, 0.24]], labels = Array.from({ length: T }, (_, t) => (P.ids ? word(P.ids[t]) : String(t)));
      return {
        title: '+', sub: `residual: x + ${what}`, shape: 'op',
        inner: (ctx, q, S) => {
          const u = S.u(q, 8, 13);
          [['the stream before', before, 'b'], [`+ the ${what} output`, delta, 'd'], ['= the stream after: zoom into a number', after, 'a']].forEach(([t, v, kk], n) => {
            const g = S.place(q, G[n]);
            S.caption(ctx, t, g, { size: u * 0.8, minW: 200 });
            drawHeat(ctx, heatImg(cache, `add${bi}:${k}${kk}`, T, D, (i, j) => v[i * D + j]), g, T, D);
            rowLabels(ctx, S, labels, g, { size: u * 0.75, maxW: g.x - q.x - 2 });
          });
        },
        grids: [
          { id: 'in', rect: G[0], rows: T, cols: D, make: (i, j) => (k === 1 ? valueNode(bi - 1, i * D + j) : tfMid(bi, i, j)) },
          { id: 'f', rect: G[1], rows: T, cols: D, make: (i, j) => (k === 1 ? projNeuron(attnData(bi), 'o', i, j) : mlpOut(bi, i, j)) },
          { id: 'sum', rect: G[2], rows: T, cols: D, make: (i, j) => Object.assign(Dp.fadd(X, before[i * D + j], delta[i * D + j]), { title: `${tok(i)}[${j}] + ${what} output` }) },
        ],
        about: [...residualAbout(`the ${what} output`), `Here ${T} × ${D} additions, one per number of the stream. Zoom into a number of the bottom row to open one.`],
        code: `block = ${pyName(bi)}\n${k === 1 ? `x = torch.randn(1, ${T}, ${D})\nmid = x + block.attn(block.norm1(x))      # same shape in and out` : `mid = torch.randn(1, ${T}, ${D})\nout = mid + block.mlp(block.norm2(mid))`}`,
      };
    }

    // ---- a tensor of one example as a picture: bars ([F]), a heatmap ([T, D]) or channel tiles ([C, H, W])
    // with a grid that opens any number (tiles open a channel first)
    function tensorView(key, s, v, f, o) {
      if (s.length === 3) {
        const C = s[0], lay = tileLay(C, o.tiles || 32, f[2] * size.w, f[3] * size.h), m = o.max || maxAbs(v);
        return {
          draw: (ctx, r, S, u) => { const q = S.place(r, f); if (o.caption) S.caption(ctx, `${o.caption}${C > lay.k ? ` (${lay.k} of ${C} channels)` : ''}`, q, { size: u * 0.8, minW: 160 }); drawTiles(ctx, key, s, v, q, lay, m); },
          grid: { id: o.id, index: true, rect: f, rows: lay.rows, cols: lay.cols, square: true, skip: (i, j) => i * lay.cols + j >= lay.k, make: (i, j) => { const c = i * lay.cols + j; return c < lay.k ? channelNode(key, s, v, c, o) : null; } },
        };
      }
      if (s.length === 2) {
        // a sequence: one row per position, its token written to the left (room for it is kept inside f)
        const [T, D] = s, labels = T === T0 && P.ids ? Array.from(P.ids, word) : null, lw = labels ? Math.min(0.12, 50 / size.w) : 0, g = [f[0] + lw, f[1], f[2] - lw, f[3]];
        return {
          draw: (ctx, r, S, u) => { const q = S.place(r, g); if (o.caption) S.caption(ctx, o.caption, q, { size: u * 0.8, minW: 160 }); drawHeat(ctx, heatImg(cache, key, T, D, (i, j) => v[i * D + j], { max: o.max }), q, T, D); if (labels) rowLabels(ctx, S, labels, q, { size: u * 0.78, maxW: lw * r.w - 6 }); },
          grid: { id: o.id, rect: g, rows: T, cols: D, make: (i, j) => o.make(i * D + j) },
        };
      }
      return {
        draw: (ctx, r, S, u) => { const q = S.place(r, f); if (o.caption) S.caption(ctx, o.caption, q, { size: u * 0.8, minW: 160 }); S.bars(ctx, v, q, { max: o.max }); },
        grid: { id: o.id, rect: f, rows: 1, cols: s[0], make: (i, j) => o.make(j) },
      };
    }
    // channels as square tiles from r's top-left corner, the same cells a square grid of lay.rows × lay.cols has
    function drawTiles(ctx, key, s, v, r, lay, m) {
      const [, H, Wd] = s, n = H * Wd, cs = Math.min(r.w / lay.cols, r.h / lay.rows), gap = Math.max(1, cs * 0.06), px = (cs - gap) / Math.max(H, Wd);
      ctx.save(); ctx.imageSmoothingEnabled = false;
      for (let c = 0; c < lay.k; c++) ctx.drawImage(heatImg(cache, `${key}:c${c}`, H, Wd, (i, j) => v[c * n + i * Wd + j], { max: m }), r.x + (c % lay.cols) * cs + gap / 2, r.y + Math.floor(c / lay.cols) * cs + gap / 2, Wd * px, H * px);
      ctx.restore();
    }
    function miniTensor(ctx, S, key, s, v, r) {
      if (!s || !v) return;
      if (s.length === 3) { drawTiles(ctx, key, s, v, r, tileLay(s[0], 8, r.w, r.h), maxAbs(v)); return; }
      if (s.length === 2) { drawHeat(ctx, heatImg(cache, key, s[0], s[1], (i, j) => v[i * s[1] + j]), r, s[0], s[1]); return; }
      if (s[0] > 600) { drawHeat(ctx, heatImg(cache, key + 'strip', 1, s[0], (i, j) => v[j]), r, 1, s[0]); return; }
      S.bars(ctx, v, r);
    }
    function channelNode(key, s, v, c, o) {
      const [C, H, Wd] = s, n = H * Wd, m = maxAbs(v), rect = W2([0, 0.13, 0.6, 0.85], [0, 0.15, 1, 0.62]), side = W2([0.66, 0.2, 0.34, 0.7], [0, 0.82, 1, 0.17]);
      let lo = Infinity, hi = -Infinity, at = 0;
      for (let p = 0; p < n; p++) { const q = v[c * n + p]; if (q < lo) lo = q; if (q > hi) { hi = q; at = p; } }
      return {
        title: `${o.name} · channel ${c + 1}`, sub: `${H} × ${Wd} numbers`, tone: o.tone,
        summary: (ctx, r) => { const sq = Math.min(r.w, r.h); drawHeat(ctx, heatImg(cache, `${key}:c${c}`, H, Wd, (i, j) => v[c * n + i * Wd + j], { max: m }), { x: r.x + (r.w - sq) / 2, y: r.y, w: sq, h: sq }, H, Wd, true); },
        inner: (ctx, r, S) => {
          const u = S.u(r, 8, 13);
          S.lines(ctx, [`channel ${c + 1} of ${C}: ${H} × ${Wd} numbers`, `from ${fmt(lo, 3)} to ${fmt(hi, 3)} (largest at (${at % Wd}, ${Math.floor(at / Wd)}))`, 'zoom into any cell to open its number'], S.place(r, [0, 0, 1, 0.12]), { mono: true, size: u * 0.88, strong: (k) => k === 0 });
          drawHeat(ctx, heatImg(cache, `${key}:c${c}`, H, Wd, (i, j) => v[c * n + i * Wd + j], { max: m }), S.place(r, rect), H, Wd, true);
          if (o.side) o.side(ctx, S.place(r, side), S, c, u);
        },
        grids: [{ id: 'px', rect, rows: H, cols: Wd, square: true, make: (i, j) => o.make(c * n + i * Wd + j) }],
        about: o.channelAbout ? o.channelAbout(c) : [`Channel ${c + 1} of ${C}: one map of ${H} × ${Wd} numbers. Blue is positive, orange negative. Zoom into any cell to see where its number came from.`],
        code: o.channelCode ? o.channelCode(c) : `x = torch.randn(1, ${C}, ${H}, ${Wd})                  # a tensor of this shape\nchannel = x[0, ${c}]                               # channel ${c + 1}: one map of ${H} × ${Wd} numbers`,
      };
    }

    // ---- the boxes of the top level
    const tone = (i) => ((shapes[i] || {}).problem ? 'critical' : TONE[blocks[i].type] || 's1');
    const problemOf = (i) => (shapes[i] || {}).problem;
    const fixText = (p) => (p && p.fix ? ` The fix: add ${p.fix.map((f) => (L.blockLabel ? L.blockLabel(f) : f.type)).join(' and ')} before it (the red block in the builder has a button for it).` : '');
    // the level shown when there are no numbers: what the part does and its shapes (and what does not fit)
    function plainLevel(lines, problem) {
      return (ctx, r, S) => {
        const u = S.u(r, 8, 13);
        const h = S.lines(ctx, lines, S.place(r, [0, 0, 1, 0.5]), { mono: true, size: u * 0.95, strong: (k) => k === 0 });
        if (problem) S.lines(ctx, [`Does not fit: ${problem.reason}${fixText(problem)}`], { x: r.x, y: r.y + h + u * 1.2, w: r.w, h: r.h * 0.4 }, { size: u * 0.95, w: 600, colorOf: () => theme.c('critical') });
        else S.lines(ctx, [!ok ? 'The model does not fit together yet, so there are no numbers to show.' : st.err ? `No numbers: ${st.err}` : 'No example to run: add some data first.'], { x: r.x, y: r.y + h + u * 1.2, w: r.w, h: r.h * 0.3 }, { size: u * 0.9, colorOf: () => theme.c('muted') });
      };
    }
    const pyHook = (mod, inName, outName) => `grab = {}
hook = ${mod}.register_forward_hook(lambda m, inp, out: grab.update(${inName}=inp[0], ${outName}=out))
with torch.no_grad():
    model(Xva[:1])                                 # the first validation example, as here
hook.remove()`;

    function blockNode(i) {
      const b = blocks[i], c = cfgOf(b), sI = inS(i), sO = outS(i), prob = problemOf(i), base = { id: b.id || 'b' + i, title: blockTitle(i), tone: tone(i) };
      const shp = sI && sO ? `${shapeText(sI)} → ${shapeText(sO)}` : '';
      const have = nums && !prob, lay = have ? model.layers[i] : null, r = have ? P.layers[i] : null, xin = have ? inVals(i) : null;
      const pnum = (shapes[i] || {}).params || 0, mod = pyName(i);
      const plain = (lines, about, code) => Object.assign(base, { inner: plainLevel([...lines, shp ? `${shp}${pnum ? ` · ${fmtInt(pnum)} parameters` : ''}` : ''].filter(Boolean), prob), about: prob ? [...about, `Here it does not fit: ${prob.reason}${fixText(prob)}`] : about, code });
      const sum = (key, s, v) => (ctx, q, S) => miniTensor(ctx, S, key, s, v, q);

      if (b.type === 'dense') {
        // the engine adds the input back only when the shapes match (a layer that changes the width cannot)
        const res = !!c.residual && !!sI && !!sO && sI.join() === sO.join();
        const U = c.units, mixed = Array.isArray(b.acts) && b.acts.some((a) => a !== c.act), actL = mixed ? 'mixed activations' : ACT(c.act).label;
        base.sub = `${U} neuron${U === 1 ? '' : 's'} · ${actL}${res ? ' · residual' : ''}`;
        const about0 = `A dense layer of ${U} neuron${U === 1 ? '' : 's'}. Each neuron multiplies every one of its inputs by its own learned weight, adds them up with a bias and bends the sum with ${mixed ? 'its own activation (you gave some neurons a different one)' : `${ACT(c.act).label}`}: a = f(W·x + b).${sI && sI.length === 2 ? ` On a sequence the same neurons run at each of the ${sI[0]} positions on its own.` : ''}${res ? ' The layer\'s input is added to its output (a residual connection).' : c.residual && sI && sO ? ` Its residual option is on, but the layer changes the width (${sI[sI.length - 1]} → ${U}), so its input cannot be added back and nothing is added.` : ''}`;
        const code = mod ? `layer = ${mod}                                # nn.Linear(${sI ? sI[sI.length - 1] : '?'}, ${U})
${pyHook('layer', 'x_in', 'z')}
x_in, z = grab['x_in'], grab['z']                  # its input ${sI ? pyShape(sI) : ''} and the weighted sums
print(torch.allclose(z, x_in @ layer.weight.T + layer.bias))   # True: one weighted sum per neuron
a = ${mixed ? `model.act${i + 1}(z)` : PYACT[c.act] ? PYACT[c.act]('z') : 'z'}${res ? '\nout = x_in + a                                     # the residual' : ''}` : '';
        if (!have) return plain([`a = ${mixed ? 'f' : ACT(c.act).label}(W·x + b)`], [about0], code);
        const F = lay.nIn, Wt = lay.W.w, bb = lay.b.w, T = lay.rows, seqL = sI.length === 2, acts = Array.from({ length: U }, (_, j) => unitAct(lay, b, j));
        const zs = r.z, as = r.a;
        let alive = 0; for (let j = 0; j < U; j++) { let on = false; for (let t = 0; t < T; t++) if (zs[t * U + j] > 0) on = true; if (on) alive++; }
        if (!seqL) {
          const cols = res ? ['b', 'z', 'a', 'out'] : ['b', 'z', 'a'], cw = 0.36 / cols.length;
          const LY = Object.assign(W2({ t: [0, 0, 1, 0.15], x: [0.1, 0.24, 0.5, 0.1], W: [0.1, 0.39, 0.5, 0.59], lab: [0, 0.39, 0.09, 0.59] }, { t: [0, 0, 1, 0.16], x: [0.13, 0.22, 0.47, 0.08], W: [0.13, 0.35, 0.47, 0.63], lab: [0, 0.35, 0.12, 0.63] }),
            Object.fromEntries(cols.map((k, n) => [k, [0.64 + n * cw, W2(0.39, 0.35), cw - 0.025, W2(0.59, 0.63)]])));
          const capOf = { b: W2('bias b', 'b'), z: W2('z = W·x + b', 'z'), a: W2(mixed ? 'a = f(z)' : `a = ${ACT(c.act).short || ACT(c.act).label}(z)`, 'a'), out: W2('x + a', '+x') };
          const valOf = { b: bb, z: zs, a: as, out: r.out }, zaMax = Math.max(maxAbs(zs), maxAbs(as), res ? maxAbs(r.out) : 0);
          let top = 0; for (let j = 1; j < U; j++) if (as[j] > as[top]) top = j;
          const head = `a = ${mixed ? 'f' : ACT(c.act).label}(W·x + b)`, sizes = `${F} input${F === 1 ? '' : 's'} → ${U} neuron${U === 1 ? '' : 's'} · ${fmtInt(pnum)} parameters`, live = ACT(acts[0]).canDie && !mixed ? `${alive} of ${U} active (z > 0)` : '';
          return Object.assign(base, {
            summary: sum(`dsum${i}`, sO, r.out),
            inner: (ctx, q, S) => {
              const u = S.u(q, 8, 13), at = (k) => S.place(q, LY[k]);
              S.lines(ctx, wide ? [`${head}   ${sizes}`, 'each row of W is one neuron\'s weights; z is its weighted sum, a its output', `here the strongest is neuron ${top + 1}: a = ${fmt(as[top], 3)}${live ? ' · ' + live : ''}`]
                : [head, sizes, `strongest: neuron ${top + 1}, a = ${fmt(as[top], 3)}`, live].filter(Boolean), at('t'), { mono: true, size: u * 0.9, strong: (k) => k === 0 });
              const xr = at('x'), wr = at('W');
              S.caption(ctx, `input x: ${F} number${F === 1 ? '' : 's'}${i ? ` from block ${i}` : ''}`, xr, { size: u * 0.8, minW: 150 }); S.bars(ctx, xin, xr);
              S.caption(ctx, W2(`weights W: ${U} neurons × ${F} inputs`, `W: ${U} × ${F}`), wr, { size: u * 0.8, minW: 90 });
              drawHeat(ctx, heatImg(cache, `W:${i}`, U, F, (a, p) => Wt[a * F + p]), wr, U, F);
              // z and a share a scale, so what the activation does to each sum shows
              for (const k of cols) { const cr = at(k); S.caption(ctx, capOf[k], cr, { size: u * 0.8, minW: 20 }); hbars(ctx, valOf[k], cr, k === 'b' ? {} : { max: zaMax }); }
              // neuron numbers (and their own activations), thinned out when the rows are thin
              const lr = at('lab'), ch = lr.h / U, every = Math.ceil(9 / ch);
              if (ch >= 3) for (let j = 0; j < U; j += every) S.text(ctx, mixed ? `${j + 1} ${ACT(acts[j]).short || acts[j]}` : String(j + 1), lr.x + lr.w, lr.y + ch * (j + 0.5), { size: Math.min(u * 0.78, Math.max(7, ch * 0.8)), mono: true, align: 'right', color: mixed && acts[j] !== c.act ? theme.c('ink') : theme.c('muted'), maxW: lr.w });
            },
            grids: [
              { id: 'x', rect: LY.x, rows: 1, cols: F, make: (a, p) => valueNode(i - 1, p) },
              { id: 'W', rect: LY.W, rows: U, cols: F, make: (j, p) => Dp.stored(X, Wt[j * F + p], `W[${j}, ${p}]: neuron ${j + 1}'s weight on ${nameOf(i - 1, p)}`) },
              { id: 'b', rect: LY.b, rows: U, cols: 1, make: (j) => Dp.stored(X, bb[j], `bias of neuron ${j + 1}`) },
              { id: 'z', rect: LY.z, rows: U, cols: 1, make: (j) => denseNeuron(i, 0, j) },
              { id: 'n', rect: LY.a, rows: U, cols: 1, make: (j) => denseNeuron(i, 0, j) },
              ...(res ? [{ id: 'out', rect: LY.out, rows: U, cols: 1, make: (j) => denseValue(i, j) }] : []),
            ],
            get about() { return [about0, `For this example the ${F} input${F === 1 ? '' : 's'} (top row) meet each row of the weight matrix: neuron ${top + 1} answers most strongly, a = ${fmt(as[top], 3)}.${ACT(acts[0]).canDie && alive < U ? ` ${U - alive} neuron${U - alive === 1 ? ' is' : 's are'} silent here (z ≤ 0, so ReLU gives 0).` : ''}`, 'Zoom into a row of z or a for that neuron\'s weighted sum, into W for a weight\'s bits, or into an input bar for the number of the block below that produced it.']; },
            code,
          });
        }
        const LY = W2({ t: [0, 0, 1, 0.14], x: [0.08, 0.24, 0.26, 0.72], W: [0.39, 0.24, 0.26, 0.72], a: [0.73, 0.24, 0.27, 0.72] }, { t: [0, 0, 1, 0.14], x: [0.14, 0.2, 0.86, 0.22], W: [0.14, 0.48, 0.86, 0.2], a: [0.14, 0.75, 0.86, 0.23] });
        const labels = Array.from({ length: T }, (_, t) => (P.ids ? word(P.ids[t]) : String(t)));
        return Object.assign(base, {
          summary: sum(`dsum${i}`, sO, r.out),
          inner: (ctx, q, S) => {
            const u = S.u(q, 8, 13), at = (k) => S.place(q, LY[k]);
            S.lines(ctx, [`a = ${mixed ? 'f' : ACT(c.act).label}(W·x + b) at each of the ${T} positions${res ? ', + x' : ''}`, `${F} → ${U} numbers per position · ${fmtInt(pnum)} parameters`], at('t'), { mono: true, size: u * 0.9, strong: (k) => k === 0 });
            const xr = at('x'), wr = at('W'), ar = at('a');
            S.caption(ctx, `input: ${T} positions × ${F}`, xr, { size: u * 0.8, minW: 120 }); drawHeat(ctx, heatImg(cache, `dx${i}`, T, F, (t, p) => xin[t * F + p]), xr, T, F);
            rowLabels(ctx, S, labels, xr, { size: u * 0.78, maxW: Math.max(18, xr.x - q.x - 2) });
            S.caption(ctx, `weights W: ${U} × ${F}`, wr, { size: u * 0.8, minW: 100 }); drawHeat(ctx, heatImg(cache, `W:${i}`, U, F, (a, p) => Wt[a * F + p]), wr, U, F);
            const shown = res ? r.out : as;
            S.caption(ctx, `${res ? 'out = x + a' : 'output a'}: ${T} × ${U}, zoom into one`, ar, { size: u * 0.8, minW: 120 }); drawHeat(ctx, heatImg(cache, `da${i}`, T, U, (t, j) => shown[t * U + j]), ar, T, U);
            if (!wide) rowLabels(ctx, S, labels, ar, { size: u * 0.78, maxW: Math.max(18, ar.x - q.x - 2) });
          },
          grids: [
            { id: 'x', rect: LY.x, rows: T, cols: F, make: (t, p) => valueNode(i - 1, t * F + p) },
            { id: 'W', rect: LY.W, rows: U, cols: F, make: (j, p) => Dp.stored(X, Wt[j * F + p], `W[${j}, ${p}], a weight of neuron ${j + 1}`) },
            { id: 'a', rect: LY.a, rows: T, cols: U, make: (t, j) => (res ? denseValue(i, t * U + j) : denseNeuron(i, t, j)) },
          ],
          get about() { return [about0, 'Each row of the input is one position; each row of the output is what the same neurons make of it. Positions do not see each other here: only attention mixes them. Zoom into a number of the output for the neuron behind it.']; },
          code,
        });
      }

      if (b.type === 'activation') {
        const a = ACT(c.act);
        base.sub = a.label;
        const about0 = `${a.label} applied to every number on its own: ${a.formula}. Without bends like this, any stack of dense layers would collapse into a single straight-line (linear) one; the bends are what let a network draw curved boundaries.`;
        const code = `x = torch.randn${sI ? pyShape(sI) : '(1, 8)'}               # any input of this shape\ny = ${PYACT[a.id] ? PYACT[a.id]('x') : 'x'}                      # every number on its own`;
        if (!have) return plain([`y = ${a.label}(x), number by number`], [about0], code);
        const n = xin.length, LY = W2({ t: [0, 0, 0.62, 0.15], c: [0.7, 0.03, 0.3, 0.42], in: [0, 0.3, 0.62, 0.26], out: [0, 0.69, 0.62, 0.28] }, { t: [0, 0, 1, 0.12], c: [0.2, 0.16, 0.6, 0.2], in: [0, 0.45, 1, 0.2], out: [0, 0.76, 1, 0.22] });
        const vin = tensorView(`aci${i}`, sI, xin, LY.in, { id: 'x', caption: 'x: what comes in', make: (k) => valueNode(i - 1, k), name: `Block ${i + 1} input`, tone: base.tone });
        const vout = tensorView(`aco${i}`, sO, r.out, LY.out, { id: 'y', caption: `y = ${a.label}(x): zoom into any number`, make: (k) => actValue(i, k), name: `Block ${i + 1} output`, tone: base.tone });
        let lo = Infinity, hi = -Infinity; for (let k = 0; k < n; k++) { if (xin[k] < lo) lo = xin[k]; if (xin[k] > hi) hi = xin[k]; }
        const x0 = Math.min(-2, lo), x1 = Math.max(2, hi), ys = [a.f(x0), a.f(x1), a.f(0), a.f((x0 + x1) / 2), ...(a.lo !== undefined ? [a.lo, a.hi] : [])], y0 = Math.min(...ys, -0.2), y1 = Math.max(...ys, 1);
        return Object.assign(base, {
          summary: sum(`acs${i}`, sO, r.out),
          inner: (ctx, q, S) => {
            const u = S.u(q, 8, 13), cr = S.place(q, LY.c);
            S.lines(ctx, [`y = f(x) for every number x, with f = ${a.label}`, `f(z) = ${a.formula}`, `${fmtInt(n)} numbers, each on its own`], S.place(q, LY.t), { mono: true, size: u * 0.9, strong: (k) => k === 0 });
            S.caption(ctx, `${a.label}, with this example's numbers`, cr, { size: u * 0.8, minW: 120 }); S.frame(ctx, cr);
            S.curve(ctx, a.f, cr, x0, x1, y0, y1, { color: theme.c(base.tone), lw: 1.8 });
            ctx.fillStyle = theme.c('ink', 0.55);
            for (let k = 0; k < n; k += Math.max(1, Math.floor(n / 300))) { const px = cr.x + ((xin[k] - x0) / (x1 - x0)) * cr.w, py = cr.y + cr.h - ((clamp(r.out[k], y0, y1) - y0) / (y1 - y0)) * cr.h; ctx.beginPath(); ctx.arc(px, py, 2, 0, Math.PI * 2); ctx.fill(); }
            vin.draw(ctx, q, S, u); vout.draw(ctx, q, S, u);
          },
          grids: [vin.grid, vout.grid],
          get about() { return [about0, 'The dots on the curve are this example\'s numbers. Zoom into any number of y to see the function computed on it, step by step, down to the arithmetic.']; },
          code,
        });
      }

      if (b.type === 'dropout') {
        const p = c.p;
        base.sub = `drops ${fmtPct(p, 0)} while training`;
        const about0 = `Dropout fights overfitting. While training, each step switches off a random ${fmtPct(p, 0)} of these numbers (sets them to 0) and scales the rest up by 1/(1 − p) = ${fmt(1 / (1 - p), 2)}, so that the total stays the same on average. The network cannot lean on any single neuron, because it might be gone.`;
        const code = `drop = ${mod}                                 # nn.Dropout(${p})
x = torch.ones(1, 10)
drop.train(); print(drop(x))                       # about ${fmtPct(p, 0)} zeros, the rest ${fmt(1 / (1 - p), 2)}
drop.eval();  print(drop(x))                       # when the model is used: unchanged`;
        if (!have) return plain(['while training: y = x · mask / (1 − p);  when used: y = x'], [about0, 'When the model is used (evaluation), dropout does nothing at all: every number passes through unchanged.'], code);
        const LY = W2({ t: [0, 0, 1, 0.16], in: [0, 0.3, 1, 0.27], tr: [0, 0.7, 1, 0.27] }, { t: [0, 0, 1, 0.16], in: [0, 0.27, 1, 0.28], tr: [0, 0.68, 1, 0.3] });
        const rng = new NA.RNG(977 * (i + 1) + 13), keep = 1 / (1 - p), mask = Float32Array.from({ length: xin.length }, () => (rng.next() < p ? 0 : keep)), tr = Float32Array.from(xin, (v, k) => v * mask[k]), both = Math.max(maxAbs(xin), maxAbs(tr));
        const vin = tensorView(`dri${i}`, sI, xin, LY.in, { id: 'y', max: both, caption: 'when the model is used, as here: y = x, unchanged', make: (k) => Object.assign(valueNode(i - 1, k), { sub: 'unchanged by dropout when the model is used' }), name: `Block ${i + 1}`, tone: base.tone });
        const vtr = tensorView(`drt${i}`, sI, tr, LY.tr, { id: 't', max: both, caption: `one training step: ${fmtPct(p, 0)} set to 0, the rest × ${fmt(keep, 2)} (same scale)`, make: (k) => Dp.stored(X, tr[k], mask[k] ? `kept and scaled: ${fmt(xin[k], 3)} × ${fmt(keep, 2)}` : 'dropped: 0 for this step'), name: `Block ${i + 1} (one training step)`, tone: base.tone });
        return Object.assign(base, {
          summary: sum(`drs${i}`, sO, r.out),
          inner: (ctx, q, S) => {
            const u = S.u(q, 8, 13);
            S.lines(ctx, [`training: y = x · mask / (1 − ${fmt(p, 2)});   used: y = x`, `${fmtInt(xin.length)} numbers; a new random mask every training step`], S.place(q, LY.t), { mono: true, size: u * 0.9, strong: (k) => k === 0 });
            vin.draw(ctx, q, S, u); vtr.draw(ctx, q, S, u);
          },
          grids: [vin.grid, vtr.grid],
          about: [about0, 'When the model is used (evaluation), as on this example, dropout does nothing: every number passes through unchanged. The lower picture is one made-up training step, to show what a mask does.'],
          code,
        });
      }

      if (b.type === 'layernorm') {
        base.sub = sI && sI.length === 2 ? 'each position' : sI ? `${sI[sI.length - 1]} numbers` : '';
        const code = `norm = ${mod}                                 # nn.LayerNorm(${sI ? sI[sI.length - 1] : 'd'})
x = torch.randn${sI ? pyShape(sI) : '(1, 8)'}
mu = x.mean(-1, keepdim=True)
var = x.var(-1, keepdim=True, unbiased=False)
y = (x - mu) / torch.sqrt(var + 1e-5) * norm.weight + norm.bias
print(torch.allclose(y, norm(x), atol=1e-6))       # True`;
        const about0 = 'LayerNorm rescales each example (or each position of a sequence) to mean 0 and spread 1, then applies a learned scale γ and shift β to every number. It keeps the numbers in a steady range however many layers came before, which keeps training stable.';
        if (!have) return plain(['y = (x − μ) / √(σ² + ε) · γ + β'], [about0], code);
        const D = lay.d, T = lay.rows;
        return Object.assign(base, lnLevel({ key: `ln${i}`, title: base.title, tone: base.tone, rows: T, d: D, x: xin, y: r.out, g: lay.g.w, b: lay.b.w, src: (t, j) => valueNode(i - 1, t * D + j), cell: (t, j) => lnCell(i, t, j), code }), { id: base.id, title: base.title, sub: base.sub });
      }

      if (b.type === 'conv') {
        base.sub = `${c.filters} filter${c.filters === 1 ? '' : 's'} · ${c.k}×${c.k}${c.stride > 1 ? ' · stride ' + c.stride : ''} · ${ACT(c.act).label}`;
        const C0 = sI ? sI[0] : '?', pad = c.k >> 1;
        const code = `conv = ${mod}                                 # nn.Conv2d(${C0}, ${c.filters}, ${c.k}${c.stride > 1 ? `, stride=${c.stride}` : ''}, padding=${pad})
${pyHook('conv', 'x_in', 'z')}
x_in, z = grab['x_in'], grab['z']                  # ${sI ? pyShape(sI) : ''} in, ${sO ? pyShape(sO) : ''} out
# one number by hand: channel 0 at row 0, column 0 is a window times the kernel, plus the bias
patch = F.pad(x_in, [${pad}] * 4)[0, :, 0:${c.k}, 0:${c.k}]
print(torch.allclose((patch * conv.weight[0]).sum() + conv.bias[0], z[0, 0, 0, 0]))   # True
a = ${PYACT[c.act] ? PYACT[c.act]('z') : 'z'}`;
        const about0 = `A convolution slides ${c.filters} small learned filter${c.filters === 1 ? '' : 's'} of ${c.k} × ${c.k} over the picture${sI ? `, each looking through all ${sI[0]} input channel${sI[0] === 1 ? '' : 's'}` : ''}${c.stride > 1 ? `, moving ${c.stride} pixels at a time so the picture shrinks` : ''}. At every position a filter multiplies the window under it by its weights and adds them up: one output channel per filter, each a map of where its pattern (an edge, a corner, a stroke) shows up. Then ${ACT(c.act).label}.`;
        if (!have) return plain([`a[f, y, x] = ${ACT(c.act).label}(b[f] + Σ W[f] · window)`], [about0], code);
        const { C, k, s, F } = lay, J = C * k * k, w = lay.Wt.w, [, Ho, Wo] = sO;
        const LY = W2({ t: [0, 0, 0.47, 0.3], kr: [0, 0.42, 0.44, 0.52], out: [0.5, 0.12, 0.5, 0.86] }, { t: [0, 0, 1, 0.18], kr: [0, 0.25, 1, 0.14], out: [0, 0.47, 1, 0.52] });
        let zeros = 0; for (let q = 0; q < r.out.length; q++) if (r.out[q] === 0) zeros++;
        const vout = tensorView(`cv${i}`, sO, r.out, LY.out, {
          id: 'ch', caption: `output: ${F} channel${F === 1 ? '' : 's'} of ${Ho} × ${Wo}, open one`, make: (q) => { const p = posOf(sO, q); return convNeuron(i, p.c, p.y, p.x); }, name: blockTitle(i), tone: base.tone, pyOut: 'a',
          side: (ctx, rr, S, f, u) => {
            // this channel's filter: one k × k kernel per input channel
            const lay2 = tileLay(C, 16, rr.w, rr.h - u * 1.4), cs = Math.min(rr.w / lay2.cols, (rr.h - u * 1.4) / lay2.rows), m = maxAbs(w.subarray(f * J, (f + 1) * J));
            S.text(ctx, `its filter: ${C > lay2.k ? `${lay2.k} of ` : ''}${C} kernel${C === 1 ? '' : 's'} of ${k} × ${k}`, rr.x, rr.y + u * 0.6, { size: u * 0.8, color: theme.c('muted'), maxW: rr.w });
            for (let ch = 0; ch < lay2.k; ch++) drawHeat(ctx, heatImg(cache, `kf${i}:${f}:${ch}`, k, k, (a2, b2) => w[f * J + (ch * k + a2) * k + b2], { max: m }), { x: rr.x + (ch % lay2.cols) * cs + 2, y: rr.y + u * 1.4 + Math.floor(ch / lay2.cols) * cs + 2, w: cs - 4, h: cs - 4 }, k, k, true);
          },
          channelAbout: (f) => [`Channel ${f + 1} of block ${i + 1}: filter ${f + 1} laid over every position of the picture. Each cell is one weighted sum of a ${k} × ${k} window through all ${C} input channel${C === 1 ? '' : 's'}, plus the bias, then ${ACT(c.act).label}: bright where the filter's pattern is found.`, 'The small squares are the filter itself, one kernel per input channel. Zoom into any cell for the weighted sum behind it.'],
          channelCode: (f) => [...pyGrab(mod, 'z', 'out[0]'), `a = ${PYACT[c.act] ? PYACT[c.act](`grab['z'][${f}]`) : `grab['z'][${f}]`}            # channel ${f + 1}: (${Ho}, ${Wo})`, `kernel = ${mod}.weight[${f}]                         # (${C}, ${k}, ${k}): its filter`].join('\n'),
        });
        return Object.assign(base, {
          summary: sum(`cvs${i}`, sO, r.out),
          inner: (ctx, q, S) => {
            const u = S.u(q, 8, 13), kr = S.place(q, LY.kr);
            S.lines(ctx, [`Conv2d(${C}, ${F}, ${k}${s > 1 ? `, stride ${s}` : ''}) + ${ACT(c.act).label}`, `in ${shapeText(sI)}  →  out ${shapeText(sO)}`, `${fmtInt(pnum)} parameters: ${F} filters of ${C} × ${k} × ${k}, and ${F} biases`, `each output = ${ACT(c.act).label}(Σ weights × window + bias)`, ACT(c.act).canDie ? `${fmtPct(zeros / r.out.length, 0)} of the outputs are 0 here` : ''].filter(Boolean), S.place(q, LY.t), { mono: true, size: u * 0.88, strong: (k2) => k2 === 0 });
            S.caption(ctx, `filters: one row per output channel (${F} × ${J})`, kr, { size: u * 0.8, minW: 160 });
            drawHeat(ctx, heatImg(cache, `kr${i}`, F, J, (a2, b2) => w[a2 * J + b2]), kr, F, J);
            vout.draw(ctx, q, S, u);
          },
          grids: [{ id: 'k', rect: LY.kr, rows: F, cols: J, make: (f, j) => Dp.stored(X, w[f * J + j], `kernel[${f}, ${Math.floor(j / (k * k))}, ${Math.floor((j % (k * k)) / k)}, ${j % k}], a weight`) }, vout.grid],
          get about() { return [about0, `Here ${fmtInt(pnum)} parameters make ${fmtInt(r.out.length)} outputs: the same weights are used at every position. ${ACT(c.act).canDie ? `${fmtPct(zeros / r.out.length, 0)} of the outputs are exactly 0 after ${ACT(c.act).label}: each filter answers only where its pattern is. ` : ''}Open a channel, then a cell, for the weighted sum of one window.`]; },
          code,
        });
      }

      if (b.type === 'pool') {
        base.sub = `${c.size}×${c.size} windows`;
        const code = `x = torch.randn${sI ? pyShape(sI) : '(1, 8, 14, 14)'}
y = F.max_pool2d(x, ${c.size})                         # ${sO ? pyShape(sO) : ''}: the largest of each ${c.size} × ${c.size} window
print(y[0, 0, 0, 0] == x[0, 0, :${c.size}, :${c.size}].max())     # True`;
        const about0 = `Max pooling keeps the largest number of each ${c.size} × ${c.size} window, channel by channel${sI && sO && sI[1] % c.size ? ` (the last ${sI[1] % c.size} row${sI[1] % c.size === 1 ? '' : 's'} and column${sI[1] % c.size === 1 ? '' : 's'}, which do not fill a window, are dropped)` : ''}. The picture shrinks ${c.size} times in each direction, the next filters see a wider area, and small shifts of a feature stop mattering. It has no weights to learn.`;
        if (!have) return plain([`y[c, i, j] = max of a ${c.size} × ${c.size} window`], [about0], code);
        const LY = W2({ t: [0, 0, 1, 0.12], in: [0, 0.22, 0.45, 0.76], out: [0.55, 0.22, 0.45, 0.76] }, { t: [0, 0, 1, 0.13], in: [0, 0.2, 1, 0.37], out: [0, 0.62, 1, 0.37] });
        const vin = tensorView(`pi${i}`, sI, xin, LY.in, { id: 'x', caption: `in: ${shapeText(sI)}`, make: (q) => valueNode(i - 1, q), name: `Block ${i + 1} input`, tone: base.tone, tiles: 16 });
        const vout = tensorView(`po${i}`, sO, r.out, LY.out, { id: 'y', caption: `out: ${shapeText(sO)}, open a channel`, make: (q) => poolValue(i, q), name: blockTitle(i), tone: base.tone, tiles: 16 });
        return Object.assign(base, {
          summary: sum(`pos${i}`, sO, r.out),
          inner: (ctx, q, S) => {
            const u = S.u(q, 8, 13);
            S.lines(ctx, [`y = the largest of each ${c.size} × ${c.size} window`, `${shapeText(sI)} → ${shapeText(sO)}, no parameters`], S.place(q, LY.t), { mono: true, size: u * 0.9, strong: (k) => k === 0 });
            vin.draw(ctx, q, S, u); vout.draw(ctx, q, S, u);
          },
          grids: [vin.grid, vout.grid],
          about: [about0, 'Open an output channel, then a cell, to see the window it took the maximum of and the number that won.'],
          code,
        });
      }

      if (b.type === 'flatten') {
        base.sub = sO ? `${fmtInt(sO[0])} numbers in a row` : '';
        const code = `x = torch.randn${sI ? pyShape(sI) : '(1, 8, 7, 7)'}\ny = x.flatten(1)                                   # ${sO ? pyShape(sO) : ''}: the same numbers, one after another\nprint(torch.equal(y[0], x[0].reshape(-1)))         # True`;
        const about0 = 'Flatten lines all the numbers up into one long vector, so that dense layers (and the output) can use them. Nothing is computed: the numbers are already stored in that order, channel by channel and row by row. It keeps every number but forgets which ones were neighbours.';
        if (!have) return plain(['y = x.flatten()'], [about0], code);
        const LY = W2({ t: [0, 0, 1, 0.14], in: [0, 0.22, 1, 0.46], out: [0, 0.79, 1, 0.18] }, { t: [0, 0, 1, 0.14], in: [0, 0.22, 1, 0.48], out: [0, 0.8, 1, 0.18] });
        const vin = tensorView(`fi${i}`, sI, xin, LY.in, { id: 'x', caption: `in: ${shapeText(sI)}`, make: (q) => valueNode(i - 1, q), name: `Block ${i + 1} input`, tone: base.tone, tiles: 16 });
        return Object.assign(base, {
          summary: sum(`fls${i}`, sO, r.out),
          inner: (ctx, q, S) => {
            const u = S.u(q, 8, 13), orr = S.place(q, LY.out);
            S.lines(ctx, [`y = x.flatten():  ${shapeText(sI)} → ${fmtInt(sO[0])}`, sI.length === 3 ? `number k is channel k ÷ ${sI[1] * sI[2]}, then row, then column` : `number k is position k ÷ ${sI[1]}, feature k mod ${sI[1]}`], S.place(q, LY.t), { mono: true, size: u * 0.9, strong: (k) => k === 0 });
            vin.draw(ctx, q, S, u);
            S.caption(ctx, `out: ${fmtInt(sO[0])} numbers in a row, zoom into any`, orr, { size: u * 0.8, minW: 200 });
            S.bars(ctx, r.out, orr);
          },
          grids: [vin.grid, { id: 'y', rect: LY.out, rows: 1, cols: sO[0], make: (a, q) => Object.assign(valueNode(i - 1, q), { sub: `number ${q} of the flattened vector` }) }],
          about: [about0, 'Zoom into a bar of the long vector to see which number it is and where it came from.'],
          code,
        });
      }

      if (b.type === 'gap') {
        base.sub = sI && sI.length === 2 ? 'average over positions' : 'average per channel';
        const code = sI && sI.length === 2 ? `x = torch.randn${pyShape(sI)}\ny = x.mean(1)                                      # ${pyShape(sO)}: each feature averaged over the ${sI[0]} positions` : `x = torch.randn${sI ? pyShape(sI) : '(1, 8, 7, 7)'}\ny = x.mean((2, 3))                                 # ${sO ? pyShape(sO) : ''}: one average per channel`;
        const about0 = sI && sI.length === 2 ? `Global average pooling averages each of the ${sI[1]} features over the ${sI[0]} positions: the whole sequence becomes one vector, whatever its length, with no position information left.` : `Global average pooling averages each channel over all its positions: the whole picture becomes one short vector${sI ? ` of ${sI[0]} numbers` : ''}. It works for any picture size, and it keeps how strongly each filter fired but forgets where.`;
        if (!have) return plain([sI && sI.length === 2 ? 'y[c] = mean over the positions of x[:, c]' : 'y[c] = mean over the pixels of channel c'], [about0], code);
        const LY = W2({ t: [0, 0, 1, 0.14], in: [0, 0.22, 0.6, 0.76], out: [0.66, 0.4, 0.34, 0.4] }, { t: [0, 0, 1, 0.14], in: [0, 0.22, 1, 0.46], out: [0, 0.77, 1, 0.2] });
        const vin = tensorView(`gi${i}`, sI, xin, LY.in, { id: 'x', caption: `in: ${shapeText(sI)}`, make: (q) => valueNode(i - 1, q), name: `Block ${i + 1} input`, tone: base.tone, tiles: 16 });
        return Object.assign(base, {
          summary: sum(`gas${i}`, sO, r.out),
          inner: (ctx, q, S) => {
            const u = S.u(q, 8, 13), orr = S.place(q, LY.out);
            S.lines(ctx, [sI.length === 2 ? `y[c] = (1/${sI[0]}) Σₜ x[t, c]` : `y[c] = (1/${sI[1] * sI[2]}) Σ x[c, i, j]`, `${shapeText(sI)} → ${shapeText(sO)}, no parameters`], S.place(q, LY.t), { mono: true, size: u * 0.9, strong: (k) => k === 0 });
            vin.draw(ctx, q, S, u);
            S.caption(ctx, 'the averages: zoom into any', orr, { size: u * 0.8, minW: 140 });
            S.bars(ctx, r.out, orr);
          },
          grids: [vin.grid, { id: 'y', rect: LY.out, rows: 1, cols: sO[0], make: (a, q) => gapValue(i, q) }],
          about: [about0, 'Zoom into a bar for the average as a weighted sum, all of whose weights are the same.'],
          code,
        });
      }

      if (b.type === 'attention') {
        base.sub = `${c.heads} head${c.heads === 1 ? '' : 's'}${c.causal !== false ? ' · causal' : ''}${c.residual !== false ? ' · residual' : ''}`;
        const D = sI ? sI[1] : '?', T = sI ? sI[0] : '?';
        const code = `attn = ${mod}                                 # SelfAttention(${D}, heads=${c.heads}, causal=${c.causal !== false ? 'True' : 'False'})
x = torch.randn(1, ${T}, ${D})
y = attn(x)                                        # (1, ${T}, ${D})${c.residual !== false ? '\nout = x + y                                        # the residual' : ''}`;
        const about0 = `Self-attention lets every position look at the others${c.causal !== false ? ' before it' : ''}. Each position makes a query, compares it with every position's key, turns the scores into weights with a softmax and takes the weighted average of their values, in ${c.heads} head${c.heads === 1 ? '' : 's'} side by side.`;
        if (!have) return plain(['y = Wo · concat(softmax(q·kᵀ / √d) · v)'], [about0], code);
        return Object.assign(attnLevel(attnData(i)), { id: base.id, title: base.title, sub: base.sub, tone: base.tone });
      }

      if (b.type === 'transformer') {
        base.sub = `${c.heads} heads · MLP ×${c.mlp}${c.causal === false ? ' · sees ahead' : ''}`;
        const D = sI ? sI[1] : '?';
        const code = `block = ${mod}                                # TransformerBlock(${D}, heads=${c.heads}, mlp=${c.mlp}, causal=${c.causal !== false ? 'True' : 'False'})
x = torch.randn${sI ? pyShape(sI) : '(1, 8, 32)'}
mid = x + block.attn(block.norm1(x))               # gather from the other positions, add back
out = mid + block.mlp(block.norm2(mid))            # compute at each position, add back
print(torch.allclose(out, block(x), atol=1e-6))    # True`;
        const about0 = `The block GPT is made of: x ← x + Attention(LayerNorm(x)), then x ← x + MLP(LayerNorm(x)). Attention moves information between positions; the MLP then computes with it at each position. Both add their result to the stream instead of replacing it, so later blocks can still read what earlier ones wrote.`;
        if (!have) return plain(['mid = x + Attn(LN1(x));   out = mid + MLP(LN2(mid))'], [about0], code);
        const T = lay.T, D2 = lay.D, A = attnData(i);
        const R = W2({ ln1: [0, 0.42, 0.11, 0.42], attn: [0.15, 0.3, 0.3, 0.68], add1: [0.475, 0.0, 0.04, 0.2], ln2: [0.53, 0.42, 0.11, 0.42], mlp: [0.68, 0.3, 0.26, 0.68], add2: [0.955, 0.0, 0.04, 0.2] },
          { ln1: [0.14, 0.0, 0.5, 0.08], attn: [0.14, 0.11, 0.86, 0.33], add1: [0.0, 0.46, 0.08, 0.05], ln2: [0.14, 0.53, 0.5, 0.08], mlp: [0.14, 0.64, 0.86, 0.31], add2: [0.0, 0.95, 0.08, 0.05] });
        const lnCode = (k) => `${k === 1 ? 'x' : 'mid'} = torch.randn(1, ${T}, ${D2})\nx_hat = ${mod}.norm${k}(${k === 1 ? 'x' : 'mid'})                 # nn.LayerNorm(${D2}), on each position\nprint(x_hat.shape)`;
        return Dp.lazy(Object.assign(base, {
          summary: (ctx, q) => { const s2 = Math.max(4, Math.min(q.h, q.w / lay.heads - 3)); for (let h = 0; h < lay.heads; h++) drawHeat(ctx, heatImg(cache, `att${i}:${h}`, T, T, (a2, b2) => headW(A, h, a2, b2), { seq: true, max: 1, mask: (a2, b2) => A.causal && b2 > a2 }), { x: q.x + (q.w - lay.heads * (s2 + 3)) / 2 + h * (s2 + 3), y: q.y, w: s2, h: s2 }, T, T, true); },
          inner: (ctx, q, S) => {
            const u = S.u(q, 8, 12), at = (k) => S.place(q, R[k]), ln1 = at('ln1'), att = at('attn'), a1 = at('add1'), ln2 = at('ln2'), mlp = at('mlp'), a2 = at('add2'), cen = (z) => [z.x + z.w / 2, z.y + z.h / 2];
            ctx.strokeStyle = theme.c('ink-2'); ctx.lineWidth = 2.4;
            if (wide) {
              const y = cen(a1)[1];
              ctx.beginPath(); ctx.moveTo(q.x, y); ctx.lineTo(q.x + q.w, y); ctx.stroke();
              S.text(ctx, `the stream: ${T} positions × ${D2} numbers →`, q.x, y - u * 1.1, { size: u * 0.85, color: theme.c('muted'), maxW: a1.x - q.x - 8 });
              S.arrow(ctx, cen(ln1)[0], y, cen(ln1)[0], ln1.y - 3, { lw: 1.1, head: 5 });
              S.arrow(ctx, ln1.x + ln1.w + 2, cen(ln1)[1], att.x - 3, cen(ln1)[1], { lw: 1.1, head: 5 });
              S.arrow(ctx, att.x + att.w + 2, att.y + att.h * 0.25, cen(a1)[0], a1.y + a1.h * 0.5 + Math.min(a1.w, a1.h) / 2 + 2, { lw: 1.1, head: 5, bend: [8, 6] });
              S.arrow(ctx, cen(ln2)[0], y, cen(ln2)[0], ln2.y - 3, { lw: 1.1, head: 5 });
              S.arrow(ctx, ln2.x + ln2.w + 2, cen(ln2)[1], mlp.x - 3, cen(ln2)[1], { lw: 1.1, head: 5 });
              S.arrow(ctx, mlp.x + mlp.w + 2, mlp.y + mlp.h * 0.25, cen(a2)[0], a2.y + a2.h * 0.5 + Math.min(a2.w, a2.h) / 2 + 2, { lw: 1.1, head: 5, bend: [6, 6] });
            } else {
              const x = cen(a1)[0];
              ctx.beginPath(); ctx.moveTo(x, q.y); ctx.lineTo(x, q.y + q.h); ctx.stroke();
              S.arrow(ctx, x, cen(ln1)[1], ln1.x - 3, cen(ln1)[1], { lw: 1.1, head: 5 });
              S.arrow(ctx, cen(ln1)[0], ln1.y + ln1.h + 2, cen(ln1)[0], att.y - 3, { lw: 1.1, head: 5 });
              S.arrow(ctx, att.x + 4, att.y + att.h + 2, x + Math.min(a1.w, a1.h) / 2 + 3, cen(a1)[1], { lw: 1.1, head: 5, bend: [-4, 8] });
              S.arrow(ctx, x, cen(ln2)[1], ln2.x - 3, cen(ln2)[1], { lw: 1.1, head: 5 });
              S.arrow(ctx, cen(ln2)[0], ln2.y + ln2.h + 2, cen(ln2)[0], mlp.y - 3, { lw: 1.1, head: 5 });
              S.arrow(ctx, mlp.x + 4, mlp.y + mlp.h + 2, x + Math.min(a2.w, a2.h) / 2 + 3, cen(a2)[1], { lw: 1.1, head: 5, bend: [-4, 6] });
            }
          },
          about: [about0, `Here ${T} positions of ${D2} numbers, ${lay.heads} heads of ${D2 / lay.heads} and an MLP of ${lay.Hd} hidden neurons. Open a part: each one shows its numbers for this example, and every number opens what computed it.`],
          code,
        }), () => [
          Object.assign(lnLevel({ key: `t1${i}`, title: 'LayerNorm 1', tone: 's7', rows: T, d: D2, x: xin, y: r.h1, g: lay.ln1g.w, b: lay.ln1b.w, src: (t, j) => valueNode(i - 1, t * D2 + j), cell: (t, j) => tfLnCell(i, 1, t, j), code: lnCode(1) }), { id: 'ln1', rect: R.ln1 }),
          Object.assign(attnLevel(A, { noResidual: true }), { id: 'attn', rect: R.attn }),
          Object.assign(addLevel(i, 1), { id: 'add1', rect: R.add1 }),
          Object.assign(lnLevel({ key: `t2${i}`, title: 'LayerNorm 2', tone: 's7', rows: T, d: D2, x: r.mid, y: r.h2, g: lay.ln2g.w, b: lay.ln2b.w, src: (t, j) => tfMid(i, t, j), cell: (t, j) => tfLnCell(i, 2, t, j), code: lnCode(2) }), { id: 'ln2', rect: R.ln2 }),
          Object.assign(mlpLevel(i), { id: 'mlp', rect: R.mlp }),
          Object.assign(addLevel(i, 2), { id: 'add2', rect: R.add2 }),
        ]);
      }
      return Object.assign(base, { inner: plainLevel([NAME[b.type] || b.type], prob), about: ['A block this view does not know how to draw.'], code: '' });
    }

    // ---- input, embedding, output
    function inputNode() {
      const q = d.input || {}, s = inputShape(), base = { id: 'input', title: 'Input', tone: 's3' };
      const prob = d.error ? { reason: d.error } : null;
      base.sub = d.error ? 'no data' : q.kind === 'vector' ? `${q.dim} number${q.dim === 1 ? '' : 's'}` : q.kind === 'image' ? `${q.h} × ${q.w} picture` : `${q.len} tokens`;
      const about0 = d.error ? `The data could not be made: ${d.error}` : q.kind === 'vector' ? `The model reads ${q.dim === 1 ? 'one number, x' : `${q.dim} numbers, a point's coordinates`} per example${d.family === 'curve' ? ', and has to predict the curve\'s value there' : ', and has to say which class the point belongs to'}.`
        : q.kind === 'image' ? `The model reads a picture of ${q.h} × ${q.w} pixels, each a number from 0 (paper) to 1 (ink), ${q.h * q.w} numbers in all.`
          : `The model reads ${q.len} tokens, each a whole number standing for one ${vocab.every((w) => w.length <= 1) ? 'symbol' : 'word'} of a vocabulary of ${q.vocab}. The embedding then turns each into a vector of ${q.embed} numbers.${d.len ? ` The first ${d.len} digits are the question and → marks where the answer starts; after it come the answer's digits, and at every position the model predicts the next token from the ones before it (it only learns from the answer positions: the question is random).` : d.task === 'language' ? ' This is a short story in the tiny language, padded with full stops to a fixed length; at every position the model predicts the next word.' : d.task === 'sentences' ? ' This is a window of your own text; at every position the model predicts the next word.' : ''}`;
      const code = q.kind === 'tokens' ? `ids = Xva[:1]                                      # (1, ${q.len}): the token ids of one example\nprint([VOCAB[i] for i in ids[0]] if 'VOCAB' in dir() else ids)` : q.kind === 'image' ? `import matplotlib.pyplot as plt\nx = Xva[:1]                                        # (1, ${q.c}, ${q.h}, ${q.w}): one picture\nplt.imshow(x[0, 0], cmap="gray_r")\nplt.show()` : `x, y = Xva[:1], Yva[:1]                            # the first validation example\nprint(x, y)\nprint(model(x))                                    # the model's output for it`;
      if (!P) return Object.assign(base, { inner: plainLevel([q.kind ? `${q.kind}: ${shapeText(s)}` : 'no input'], prob), about: [about0], code });
      if (q.kind === 'image') {
        const [C, H, Wd] = s, pic = W2([0.5, 0.02, 0.5, 0.96], [0, 0.22, 1, 0.76]);
        return Object.assign(base, {
          summary: (ctx, r) => { const sq = Math.min(r.w, r.h); drawHeat(ctx, heatImg(cache, 'in', H, Wd, (i, j) => P.x[i * Wd + j], { ink: true }), { x: r.x + (r.w - sq) / 2, y: r.y, w: sq, h: sq }, H, Wd, true); },
          inner: (ctx, r, S) => {
            const u = S.u(r, 8, 13);
            S.lines(ctx, [`${whose}:`, `${exampleText()}`, `${H} × ${Wd} pixels${C > 1 ? ` × ${C} channels` : ''}, 0 = paper, 1 = ink`, nums ? answerText() : ''].filter(Boolean), S.place(r, W2([0, 0, 0.46, 0.4], [0, 0, 1, 0.2])), { size: u * 0.95, strong: (k) => k === 1 });
            drawHeat(ctx, heatImg(cache, 'in', H, Wd, (i, j) => P.x[i * Wd + j], { ink: true }), S.place(r, pic), H, Wd, true);
          },
          grids: [{ id: 'px', rect: pic, rows: H, cols: Wd, square: true, make: (i, j) => Dp.stored(X, P.x[i * Wd + j], inputLabel(i * Wd + j)) }],
          about: [about0, `Shown: ${whose}, ${exampleText()}. Zoom into a pixel to see its number as the 32 bits stored in memory.`], code,
        });
      }
      if (q.kind === 'tokens') {
        const n = q.len, ids = P.ids;
        return Object.assign(base, {
          summary: (ctx, r, S) => S.text(ctx, Array.from(ids, word).join(' '), r.x + r.w / 2, r.y + r.h * 0.4, { size: clamp(r.h * 0.5, 7, 11), mono: true, align: 'center', color: theme.c('ink-2'), maxW: r.w }),
          inner: (ctx, r, S) => {
            const u = S.u(r, 8, 14), per = Math.min(n, wide ? 10 : 5), rowsN = Math.ceil(n / per), w = r.w / per, bh = Math.min((r.h * 0.62) / (rowsN * 2.6), u * 2.6);
            S.lines(ctx, [`${whose}:`, `${n} tokens from a vocabulary of ${q.vocab}`, nums ? answerText() : ''].filter(Boolean), S.place(r, [0, 0, 1, 0.2]), { size: u * 0.95, strong: (k) => k === 0 });
            for (let t = 0; t < n; t++) {
              const rr = Math.floor(t / per), cc = t % per, x = r.x + w * (cc + 0.5), y = r.y + r.h * 0.28 + rr * bh * 2.6;
              S.box(ctx, { x: x - w * 0.45, y, w: w * 0.9, h: bh * 1.2 }, { fill: theme.c('panel-2'), stroke: theme.c('line-2') });
              S.text(ctx, word(ids[t]), x, y + bh * 0.6, { size: Math.min(u, bh * 0.8, w / 4), mono: true, align: 'center', w: 650, maxW: w * 0.86 });
              S.text(ctx, String(ids[t]), x, y + bh * 1.75, { size: Math.min(u * 0.85, bh * 0.7), mono: true, align: 'center', color: theme.c('pos') });
            }
          },
          about: [about0, `Shown: ${whose}, ${exampleText()}. Below each token is its id (blue): from here on the model never sees letters, only these numbers. The next box looks each one up.`], code,
        });
      }
      // a point (or a number) among the data it came from
      const plot = W2([0.52, 0.02, 0.48, 0.96], [0.12, 0.2, 0.76, 0.5]), bars = W2([0, 0.5, 0.42, 0.3], [0, 0.8, 1, 0.16]);
      const drawPlot = (ctx, r, S) => {
        const sq = Math.min(r.w, r.h), b = { x: r.x + (r.w - sq) / 2, y: r.y, w: sq, h: sq }, curve = d.family === 'curve';
        let y0 = -1, y1 = 1;
        if (curve) { y0 = Infinity; y1 = -Infinity; for (const v of d.Ytr) { if (v < y0) y0 = v; if (v > y1) y1 = v; } const sp = (y1 - y0) || 1; y0 -= sp * 0.1; y1 += sp * 0.1; }
        const X0 = (v) => b.x + ((v + 1) / 2) * b.w, Y0 = (v) => b.y + b.h - ((v - y0) / (y1 - y0)) * b.h;
        ctx.fillStyle = theme.c('panel-2'); ctx.fillRect(b.x, b.y, b.w, b.h); S.frame(ctx, b);
        for (let k = 0; k < d.nTr; k++) {
          const px = X0(d.Xtr[curve ? k : 2 * k]), py = curve ? Y0(d.Ytr[k]) : Y0(d.Xtr[2 * k + 1]);
          ctx.fillStyle = curve ? theme.c('s1', 0.5) : theme.c(BV ? BV.classColour(d, d.Ytr[k]) : 's1', 0.45); ctx.beginPath(); ctx.arc(px, py, Math.max(1.4, sq / 160), 0, Math.PI * 2); ctx.fill();
        }
        const px = X0(P.x[0]), py = curve ? Y0(target[0]) : Y0(P.x[1]);
        ctx.strokeStyle = theme.c('focus'); ctx.lineWidth = 2.2; ctx.beginPath(); ctx.arc(px, py, Math.max(5, sq / 40), 0, Math.PI * 2); ctx.stroke();
        if (curve && nums) { ctx.fillStyle = theme.c('ink'); ctx.beginPath(); ctx.arc(px, Y0(clamp(P.logits[0], y0, y1)), 3.2, 0, Math.PI * 2); ctx.fill(); }
      };
      return Object.assign(base, {
        summary: (ctx, r, S) => S.text(ctx, q.dim <= 3 ? `(${Array.from(P.x, (v) => fmt(v, 2)).join(', ')})` : '', r.x + r.w / 2, r.y + r.h * 0.4, { size: clamp(r.h * 0.5, 7, 12), mono: true, align: 'center', color: theme.c('ink-2'), maxW: r.w }),
        inner: (ctx, r, S) => {
          const u = S.u(r, 8, 13), br = S.place(r, bars);
          S.lines(ctx, [`${whose}:`, exampleText(), nums ? answerText() : '', d.family === 'curve' ? 'circle: the example; dot: the model\'s answer' : 'circled: this example, among the training points'].filter(Boolean), S.place(r, W2([0, 0, 0.48, 0.36], [0, 0, 1, 0.18])), { size: u * 0.95, strong: (k) => k === 1 });
          drawPlot(ctx, S.place(r, plot), S);
          S.caption(ctx, 'the input numbers: zoom into one', br, { size: u * 0.8, minW: 150 }); S.bars(ctx, P.x, br, { max: 1 });
          colLabels(ctx, S, Array.from(P.x, (v, k) => `${inputLabel(k)} = ${fmt(v, 3)}`), br, { size: u * 0.8 });
        },
        grids: [{ id: 'x', rect: bars, rows: 1, cols: q.dim, make: (i, j) => Dp.stored(X, P.x[j], inputLabel(j)) }],
        about: [about0, `Shown: ${whose}, ${exampleText()}. Every number inside the model, at every level, is computed from this one example. Zoom into a bar to see the number as the 32 bits stored in memory.`], code,
      });
    }
    function embedNode() {
      const q = d.input, V = q.vocab, T = q.len, D = q.embed, base = { id: 'embed', title: 'Embedding', sub: `token + position → ${D} numbers`, tone: 's3' };
      const about0 = `The embedding is two lookup tables of learned numbers. The token table E has one row of ${D} numbers for each of the ${V} tokens; the position table P has one row for each of the ${T} positions. Each token's starting vector is the sum of its token's row and its position's row: x₀ = E[token] + P[position].`;
      const about1 = 'Attention on its own treats its input as an unordered set, so the position rows are the only thing that tells "a b" from "b a". Both tables are ordinary weights, learned with everything else.';
      const code = `ids = Xva[:1]                                      # (1, ${T}): the example's token ids\nx = model.tok(ids) + model.pos(torch.arange(ids.shape[1]))   # (1, ${T}, ${D})\n# one number by hand: token ${P ? P.ids[0] : 0} at position 0, feature 0\nprint(model.tok.weight[${P ? P.ids[0] : 0}, 0] + model.pos.weight[0, 0], x[0, 0, 0])`;
      if (!nums) return Object.assign(base, { inner: plainLevel([`x₀ = E[token] + P[position]   (${V} × ${D} and ${T} × ${D} tables)`], null), about: [about0, about1], code });
      const E = model.embed.tok.w, Pp = model.embed.pos.w, ids = P.ids, used = new Set(ids), labels = Array.from(ids, word);
      const EMB = W2({ e: [0, 0.07, 0.3, 0.88], p: [0.36, 0.07, 0.26, 0.5], x: [0.74, 0.07, 0.26, 0.5], t: [0.36, 0.68, 0.64, 0.3] }, { e: [0, 0.06, 0.46, 0.38], p: [0.54, 0.06, 0.46, 0.24], x: [0.22, 0.5, 0.78, 0.2], t: [0, 0.76, 1, 0.24] });
      return Object.assign(base, {
        summary: (ctx, r) => drawHeat(ctx, heatImg(cache, 'emb', T, D, (i, j) => P.emb[i * D + j]), r, T, D),
        inner: (ctx, r, S) => {
          const u = S.u(r, 8, 13), er = S.place(r, EMB.e), pr = S.place(r, EMB.p), xr = S.place(r, EMB.x);
          S.caption(ctx, `token table E (${V} × ${D})`, er, { size: u * 0.8, minW: 120 });
          const g = drawHeat(ctx, heatImg(cache, 'E', V, D, (i, j) => E[i * D + j]), er, V, D);
          ctx.strokeStyle = theme.c('focus'); ctx.lineWidth = 1;
          if (g.ch >= 1.5) for (const id of used) ctx.strokeRect(er.x - 0.5, er.y + id * g.ch - 0.5, g.w + 1, g.ch + 1);
          S.caption(ctx, `position table P (${T} × ${D})`, pr, { size: u * 0.8, minW: 120 });
          drawHeat(ctx, heatImg(cache, 'P', T, D, (i, j) => Pp[i * D + j]), pr, T, D);
          S.caption(ctx, `x₀ = E[token] + P[position] (${T} × ${D})`, xr, { size: u * 0.8, minW: 150 });
          drawHeat(ctx, heatImg(cache, 'emb', T, D, (i, j) => P.emb[i * D + j]), xr, T, D);
          rowLabels(ctx, S, labels, xr, { size: u * 0.78, maxW: Math.max(18, xr.x - (wide ? pr.x + pr.w : r.x) - 4) });
          S.lines(ctx, ['outlined in E: the rows this example uses', 'zoom into any number of the three tables'], S.place(r, EMB.t), { size: u * 0.85, colorOf: () => theme.c('muted') });
        },
        grids: [
          { id: 'E', rect: EMB.e, rows: V, cols: D, make: (i, j) => Dp.stored(X, E[i * D + j], `E[${quote(i)}][${j}]`) },
          { id: 'P', rect: EMB.p, rows: T, cols: D, make: (i, j) => Dp.stored(X, Pp[i * D + j], `P[${i}][${j}]`) },
          { id: 'x0', rect: EMB.x, rows: T, cols: D, make: (i, j) => valueNode(-1, i * D + j) },
        ],
        about: [about0, about1, 'Outlined rows of E are the tokens of this example. Zoom into a number of x₀ to see the addition behind it.'],
        code,
      });
    }
    function outputNode() {
      const prob = chk.problem && !d.error ? chk.problem : null, base = { id: 'output', title: 'Output', tone: prob ? 'critical' : 's2' };
      const lossName = (NA.bx.LOSSES[(L.setup.loss || {}).id] || {}).label || 'loss';
      base.sub = out.kind === 'classes' ? `${K} class scores` : out.kind === 'value' ? `${K === 1 ? 'one number' : K + ' numbers'}` : out.kind === 'tokens' ? `next-token scores (${K})` : '';
      const sIn = inS(nb), F = sIn ? sIn[sIn.length - 1] : '?';
      const about0 = out.kind === 'classes' ? `The output head is a dense layer without an activation: ${K} weighted sums of the ${F} numbers that reach it, one score (logit) per class. The softmax turns the scores into probabilities, positive and adding up to 1, and the answer is the class with the largest. Training uses the ${lossName.toLowerCase()} loss.`
        : out.kind === 'value' ? `The output head is a dense layer without an activation: a weighted sum of the ${F} numbers that reach it, plus a bias. That number is the model's answer; training pulls it toward the target with the ${lossName.toLowerCase()} loss.`
          : `At every position the output head scores every token of the vocabulary as the next one: ${K} weighted sums of that position's ${F} numbers. A softmax turns each position's scores into probabilities; training makes the token that really came next more probable (cross-entropy).`;
      const code = `${pyHook('model.out', 'x', 'z')}
z = grab['z']                                      # ${out.kind === 'tokens' ? `(1, ${T0}, ${K}): a score for every token at every position` : `(1, ${K})`}
${out.kind === 'value' ? 'print(z.item(), Yva[0].item())                     # the answer and the target' : `p = z.softmax(-1)                                  # probabilities\nprint(p.argmax(-1), p.max(-1).values)              # the answer${out.kind === 'tokens' ? 's' : ''} and how sure`}`;
      if (!nums) return Object.assign(base, { inner: plainLevel([out.kind === 'value' ? 'ŷ = W·x + b' : 'z = W·x + b;  p = softmax(z)'], prob), about: [about0], code });
      const lay = model.layers[nb], Wt = lay.W.w, bb = lay.b.w, xin = inVals(nb), Fn = lay.nIn;
      const lossLine = `loss for this example: ${lossText()}`;
      if (out.kind === 'tokens') {
        const T = T0, V = K, labels = Array.from(P.ids, word), probsAll = new Float32Array(T * V);
        tokPreds.forEach((q, t) => probsAll.set(q.probs, t * V));
        const LY = W2({ t: [0, 0, 1, 0.12], z: [0.08, 0.2, 0.42, 0.38], p: [0.58, 0.2, 0.42, 0.38], list: [0, 0.65, 1, 0.35] }, { t: [0, 0, 1, 0.14], z: [0.13, 0.2, 0.87, 0.22], p: [0.13, 0.48, 0.87, 0.22], list: [0, 0.75, 1, 0.25] });
        return Object.assign(base, {
          summary: (ctx, r) => drawHeat(ctx, heatImg(cache, 'probs', T, V, (i, j) => probsAll[i * V + j], { seq: true, max: 1 }), r, T, V),
          inner: (ctx, r, S) => {
            const u = S.u(r, 8, 13), at = (k) => S.place(r, LY[k]), zr = at('z'), pr = at('p');
            S.lines(ctx, [`z = W·x + b at every position (${Fn} → ${V} tokens);  p = softmax(z)`, `${answerText()};  ${lossLine}`], at('t'), { mono: true, size: u * 0.88, strong: (k) => k === 0 });
            S.caption(ctx, `scores z: ${T} positions × ${V} tokens`, zr, { size: u * 0.8, minW: 140 }); drawHeat(ctx, heatImg(cache, 'logits', T, V, (i, j) => P.logits[i * V + j]), zr, T, V);
            rowLabels(ctx, S, labels, zr, { size: u * 0.78, maxW: Math.max(18, zr.x - r.x - 2) });
            S.caption(ctx, 'probabilities p: zoom into one', pr, { size: u * 0.8, minW: 140 }); drawHeat(ctx, heatImg(cache, 'probs', T, V, (i, j) => probsAll[i * V + j], { seq: true, max: 1 }), pr, T, V);
            if (!wide) rowLabels(ctx, S, labels, pr, { size: u * 0.78, maxW: Math.max(18, pr.x - r.x - 2) });
            const lines = [];
            for (let t = 0; t < T; t++) {
              const q = tokPreds[t], y = target[t];
              lines.push(`${labels[t].padEnd(6).slice(0, 8)} → ${word(q.b).padEnd(6).slice(0, 8)} ${fmtPct(q.p, 0).padStart(4)}${y >= 0 ? (y === q.b ? '  ✓' : `  ✗ (next: ${word(y)})`) : ''}`);
            }
            S.lines(ctx, [`after each token, its likeliest next one${tokCount < T ? ' (✓ ✗ where there is an answer to learn)' : ''}:`, ...lines], at('list'), { mono: true, size: u * 0.82, strong: (k) => k === 0 });
          },
          grids: [
            { id: 'z', rect: LY.z, rows: T, cols: V, make: (t, v) => denseNeuron(nb, t, v) },
            { id: 'p', rect: LY.p, rows: T, cols: V, make: (t, v) => Object.assign(Dp.lib(X, 'softmax', { scores: Array.from(P.logits.subarray(t * V, (t + 1) * V)), j: v, label: `p(${word(v)})` }), { title: `p(${quote(v)} next) after ${tok(t)}`, sub: fmtPct(probsAll[t * V + v], 1) }) },
          ],
          get about() { return [about0, `On this example ${answerText()}. The list shows, after each token, the token the model finds likeliest to come next. Zoom into a score for its weighted sum, or a probability for its softmax.`]; },
          code,
        });
      }
      if (out.kind === 'value') {
        const LY = W2({ t: [0, 0, 1, 0.2], x: [0, 0.32, 0.55, 0.22], w: [0, 0.66, 0.55, 0.22], n: [0.64, 0.3, 0.36, 0.6] }, { t: [0, 0, 1, 0.2], x: [0, 0.28, 1, 0.12], w: [0, 0.48, 1, 0.12], n: [0, 0.68, 1, 0.3] });
        return Dp.lazy(Object.assign(base, {
          summary: (ctx, r, S) => S.text(ctx, `ŷ = ${fmt(P.logits[0], 3)}`, r.x + r.w / 2, r.y + r.h * 0.4, { size: clamp(r.h * 0.45, 7, 13), mono: true, w: 650, align: 'center', maxW: r.w }),
          inner: (ctx, r, S) => {
            const u = S.u(r, 8, 13), xr = S.place(r, LY.x), wr = S.place(r, LY.w);
            S.lines(ctx, [`ŷ = w·x + b = ${fmt(P.logits[0], 4)}`, `target y = ${fmt(target[0], 4)},  error ŷ − y = ${fmt(P.logits[0] - target[0], 4)}`, lossLine], S.place(r, LY.t), { mono: true, size: u * 0.92, strong: (k) => k === 0 });
            S.caption(ctx, `x: the ${Fn} numbers that reach it`, xr, { size: u * 0.8, minW: 150 }); S.bars(ctx, xin, xr);
            S.caption(ctx, 'w: its weights (learned)', wr, { size: u * 0.8, minW: 150 }); S.bars(ctx, Wt.subarray(0, Fn), wr);
          },
          grids: [{ id: 'x', rect: LY.x, rows: 1, cols: Fn, make: (a, p) => valueNode(nb - 1, p) }, { id: 'w', rect: LY.w, rows: 1, cols: Fn, make: (a, p) => Dp.stored(X, Wt[p], `w[${p}], an output weight`) }],
          about: [about0, `On this example the model says ${fmt(P.logits[0], 3)} where the target is ${fmt(target[0], 3)}. The box opens the output neuron's weighted sum.`],
          code,
        }), () => Array.from({ length: K }, (_, j) => Object.assign(denseNeuron(nb, 0, j), { id: 'n' + j, rect: K === 1 ? LY.n : [LY.n[0], LY.n[1] + (j * LY.n[3]) / K, LY.n[2], LY.n[3] / K - 0.02] })));
      }
      // classes
      const lab = Array.from({ length: K }, (_, k) => (names[k] !== undefined ? String(names[k]) : String(k + 1)));
      const LY = W2({ t: [0, 0, 1, 0.18], x: [0, 0.28, 0.4, 0.13], W: [0, 0.5, 0.4, 0.46], z: [0.48, 0.28, 0.52, 0.24], p: [0.48, 0.66, 0.52, 0.22] }, { t: [0, 0, 1, 0.2], x: [0, 0.27, 1, 0.09], W: [0, 0.41, 1, 0.14], z: [0, 0.62, 1, 0.11], p: [0, 0.82, 1, 0.1] });
      // points in the plane keep the colours of their classes, as in the results; other classes share the output's colour
      const cols = (k) => theme.c(d.family === 'plane' && BV ? BV.classColour(d, k) : 's2', k === pred ? 1 : 0.5);
      return Object.assign(base, {
        summary: (ctx, r, S) => S.bars(ctx, Float32Array.from(probs), r, { positive: true, max: 1, color: (k) => cols(k) }),
        inner: (ctx, r, S) => {
          const u = S.u(r, 8, 13), at = (k) => S.place(r, LY[k]), xr = at('x'), wr = at('W'), zr = at('z'), pr = at('p');
          S.lines(ctx, [`z = W·x + b  (${Fn} → ${K} scores);   p = softmax(z)`, `${answerText()}${pred === target[0] ? '' : `: it is ${names[target[0]] || target[0]}`}`, lossLine], at('t'), { mono: true, size: u * 0.9, strong: (k) => k === 0 });
          S.caption(ctx, `x: ${Fn} numbers`, xr, { size: u * 0.8, minW: 100 }); S.bars(ctx, xin, xr);
          S.caption(ctx, `W: ${K} classes × ${Fn}`, wr, { size: u * 0.8, minW: 100 }); drawHeat(ctx, heatImg(cache, 'Wout', K, Fn, (a, p) => Wt[a * Fn + p]), wr, K, Fn);
          S.caption(ctx, 'scores z (logits): zoom into one', zr, { size: u * 0.8, minW: 150 }); S.bars(ctx, P.logits, zr, d.family === 'plane' ? { color: (k) => cols(k) } : {});
          colLabels(ctx, S, lab, zr, { size: u * 0.78, hi: pred });
          S.caption(ctx, 'probabilities p', pr, { size: u * 0.8, minW: 120 }); S.bars(ctx, Float32Array.from(probs), pr, { positive: true, max: 1, color: (k) => cols(k) });
          colLabels(ctx, S, Array.from(probs, (v) => fmtPct(v, 0)), pr, { size: u * 0.74, hi: pred });
        },
        grids: [
          { id: 'x', rect: LY.x, rows: 1, cols: Fn, make: (a, p) => valueNode(nb - 1, p) },
          { id: 'W', rect: LY.W, rows: K, cols: Fn, make: (k, p) => Dp.stored(X, Wt[k * Fn + p], `W[${k}, ${p}], a weight of the score for ${lab[k]}`) },
          { id: 'z', rect: LY.z, rows: 1, cols: K, make: (a, k) => denseNeuron(nb, 0, k) },
          { id: 'p', rect: LY.p, rows: 1, cols: K, make: (a, k) => Object.assign(Dp.lib(X, 'softmax', { scores: Array.from(P.logits), j: k, label: `p(${lab[k]})` }), { title: `Probability of ${names[k] || lab[k]}`, sub: fmtPct(probs[k], 1) }) },
        ],
        get about() { return [about0, `On this example ${answerText()}${pred === target[0] ? '' : ` (it is ${names[target[0]] || target[0]})`}; ${lossLine}. Zoom into a score for its weighted sum, or a probability for the softmax that made it.`]; },
        code,
      });
    }

    // ---- the top level: the boxes in a row that wraps (wide), or a column that splits (tall)
    const items = [inputNode(), ...(seq && !d.error ? [embedNode()] : []), ...blocks.map((_, i) => blockNode(i)), outputNode()];
    const firstBlock = seq && !d.error ? 2 : 1;
    // the shape on the arrow out of each box
    const arrowLabel = (k) => {
      if (d.error) return '';
      const q = d.input || {};
      if (k === 0) return q.kind === 'tokens' ? `${q.len} ids` : shapeText(inputShape()).replace(/ /g, '');
      if (k === 1 && seq) return shapeText([q.len, q.embed]).replace(/ /g, '');
      const s = outS(k - firstBlock);
      return s ? shapeText(s).replace(/ /g, '') : '?';
    };
    const nItems = items.length, labels = items.slice(0, -1).map((_, k) => arrowLabel(k));
    /* Wide: as many boxes in a row as stay readable, the line wrapping back to the left. Tall: one column while
       its boxes keep room for their pictures, else two or more columns read row by row. The gaps leave room for
       the shape labels; with very many blocks they shrink and the labels go (each level still shows its shapes). */
    const u0 = clamp(Math.min(size.w / 34, size.h / 14), 7.5, 12), lf = u0 * 0.85, top = Math.max(u0 * 3.6, size.h * 0.1);
    const longest = Math.max(1, ...labels.map((s) => s.length)), AW = size.w, AH = Math.max(40, size.h - top), minW = wide ? 112 : 100;
    // boxes no larger than they need to be: a phone column stays narrower than the screen, a row of few boxes square-ish
    const boxOf = (t) => { const single = t.cols === 1 && !wide, w = Math.min(t.w, single ? Math.min(AW * 0.8, 340) : 270); return { w, h: Math.min(t.h, single ? 112 : Math.max(w * 0.95, 60)) }; };
    // the room ArchView.drawClosed needs to draw a box's picture under its title and subtitle (it drops it otherwise)
    const pictureFits = (w, h) => { const ts = clamp(Math.min(h * 0.22, w * 0.085), 8, 16), ss = Math.max(8.5, ts * 0.74); return h - ts * 1.95 - ss * 1.45 - Math.min(8, h * 0.08) >= Math.max(16, ts * 1.6); };
    let lay = null, best = null;
    for (const pass of [0, 1]) {
      const gx = pass ? 18 : Math.max(28, longest * lf * 0.62 + 16), gy = pass ? 12 : lf * 2.8 + 8;
      for (let n = 1; n <= nItems && !lay; n++) {
        const rows = wide ? n : Math.ceil(nItems / n), cols = wide ? Math.ceil(nItems / n) : n, vgap = !wide && cols === 1 ? (pass ? 10 : Math.max(18, lf * 2.2)) : gy;
        const t = { rows, cols, gx, vgap, labels: !pass, w: (AW - (cols - 1) * gx) / cols, h: (AH - (rows - 1) * vgap) / rows }, b = boxOf(t);
        t.score = Math.min(b.w / 2.4, b.h);
        if (b.w >= minW && pictureFits(b.w, b.h)) lay = t;
        else if (!best || t.score > best.score) best = t;
      }
      if (lay) break;
    }
    lay = lay || best;
    const showLabels = lay.labels, { w: bw, h: bh } = boxOf(lay);
    const gapX = lay.cols > 1 ? Math.min((AW - lay.cols * bw) / (lay.cols - 1), Math.max(lay.gx, 60)) : 0, gapY = lay.rows > 1 ? Math.min((AH - lay.rows * bh) / (lay.rows - 1), Math.max(lay.vgap, 48)) : 0;
    const totW = lay.cols * bw + (lay.cols - 1) * gapX, totH = lay.rows * bh + (lay.rows - 1) * gapY, ox = (AW - totW) / 2, oy = top + (AH - totH) / 2;
    const px = items.map((_, k) => { const row = Math.floor(k / lay.cols), col = k % lay.cols; return { x: ox + col * (bw + gapX), y: oy + row * (bh + gapY), w: bw, h: bh }; });
    items.forEach((it, k) => { it.rect = [px[k].x / size.w, px[k].y / size.h, bw / size.w, bh / size.h]; });

    const nParams = ok ? chk.params : null;
    const firstProblem = !ok ? (d.error ? { reason: d.error } : chk.problem || (shapes.find((s) => s && s.problem) || {}).problem) : null;
    const root = {
      id: 'model', title: 'Your model', sub: ok ? `${fmtInt(nParams)} parameters · ${st.live ? `trained, step ${fmtInt(L.step)}` : 'untrained'}` : 'does not fit together yet',
      inner: (ctx, r, S) => {
        const u = S.u(r, 8, 12), k = r.w / size.w, at = (n) => S.place(r, items[n].rect);
        // what is shown: the example and the weights, or why there are no numbers
        const head = !ok ? [`Does not fit together yet: ${firstProblem ? firstProblem.reason : 'see the red block'}`, 'The boxes show the shapes; fix the red part to see numbers.']
          : !P ? [st.err ? `No numbers: ${st.err}` : 'No example to show yet: add some data.', weightsNote]
            : [`${whose[0].toUpperCase()}${whose.slice(1)}: ${exampleText()}. ${nums ? answerText()[0].toUpperCase() + answerText().slice(1) + '.' : ''}`, `${weightsNote[0].toUpperCase()}${weightsNote.slice(1)} · open any box, then any number`];
        S.lines(ctx, head, { x: r.x, y: r.y, w: r.w, h: top * k }, { size: u * 0.92, strong: (n) => n === 0, colorOf: (n) => (n === 0 && !ok ? theme.c('critical') : n === 0 ? theme.c('ink') : theme.c('muted')) });
        const lab = (t, x, y, align = 'center', maxW) => { if (showLabels && t) S.text(ctx, t, x, y, { size: lf * k, mono: true, color: theme.c('muted'), align, maxW }); };
        for (let n = 0; n + 1 < nItems; n++) {
          const a = at(n), b = at(n + 1);
          if (Math.abs(a.y - b.y) < 1) { S.arrow(ctx, a.x + a.w + 2, a.y + a.h / 2, b.x - 3, b.y + b.h / 2, { lw: 1.2, head: 5 }); lab(labels[n], (a.x + a.w + b.x) / 2, a.y + a.h / 2 - lf * k * 1.1, 'center', b.x - a.x - a.w + 8); continue; }
          if (Math.abs(a.x - b.x) < 1) { S.arrow(ctx, a.x + a.w / 2, a.y + a.h + 1, b.x + b.w / 2, b.y - 2, { lw: 1.2, head: 5 }); lab(labels[n], a.x + a.w / 2 + 8, (a.y + a.h + b.y) / 2, 'left', a.w / 2 - 8); continue; }
          // the line wraps: down from the end of a row, back to the left, into the start of the next
          const ym = (a.y + a.h + b.y) / 2;
          ctx.strokeStyle = theme.c('ink-2'); ctx.lineWidth = 1.2; ctx.beginPath(); ctx.moveTo(a.x + a.w / 2, a.y + a.h + 2); ctx.lineTo(a.x + a.w / 2, ym); ctx.lineTo(b.x + b.w / 2, ym); ctx.stroke();
          S.arrow(ctx, b.x + b.w / 2, ym, b.x + b.w / 2, b.y - 3, { lw: 1.2, head: 5 });
          lab(labels[n], (a.x + a.w / 2 + b.x + b.w / 2) / 2, ym - lf * k * 0.95, 'center', Math.abs(a.x - b.x));
        }
      },
      children: items,
      get about() {
        return [
          `The model you built, as boxes: the input, ${nb ? `the ${nb} block${nb === 1 ? '' : 's'} you stacked` : 'no hidden blocks (a linear model)'}${seq ? ', the embedding' : ''} and the output head, with the shape of the data on every arrow.${ok ? ` It has ${fmtInt(nParams)} learned numbers (parameters).` : ` It does not fit together yet: ${firstProblem ? firstProblem.reason : 'see the red block'}`}`,
          P ? `Everything inside is computed on one example, ${whose}: ${exampleText()}. ${answerText()[0] ? answerText()[0].toUpperCase() + answerText().slice(1) : ''}.${st.live ? '' : ' These are untrained weights, the ones the next run starts from: train to see learned ones.'}` : ok ? 'There is no example to run yet.' : 'Once it fits, every box shows what it computes on one example.',
          'Click a box or zoom in to open it, and keep going: a dense layer opens into its neurons, a convolution into its channels and the windows they weigh, attention into its heads and what each position looks at. Every number opens what produced it, down to the arithmetic, the logic gates and the electrons. Zoom out or use the path at the top to come back. The Python tab shows each part in PyTorch, and here the whole script.',
        ];
      },
      get code() {
        if (!NA.builderPy) return '# The PyTorch version appears in the Python tab.';
        if (!PYCODE || PYCODE.chk !== chk || PYCODE.d !== d) PYCODE = { chk, d, code: NA.builderPy.code(L.setup, d) };
        return PYCODE.code;
      },
      codeLabel: 'Python · PyTorch · your model, its data and its training',
    };
    return root;
  }

  NA.archBuilder = { build };
})(window.NA);
