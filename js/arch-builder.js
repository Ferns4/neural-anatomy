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
  const TONE = { dense: 's1', activation: 's1', dropout: 's1', layernorm: 's1', attention: 's7', transformer: 's7', conv: 's4', pool: 's4', flatten: 's6', gap: 's6',
    gated: 's1', batchnorm: 's1', noise: 's1', fourier: 's1', rnn: 's7', conv1d: 's7', position: 's7', seqpool: 's7', patches: 's4', avgpool: 's4', linear: 's5', knn: 's5', tree: 's5', forest: 's5', boost: 's5' };
  const NAME = { dense: 'Dense', activation: 'Activation', dropout: 'Dropout', layernorm: 'LayerNorm', attention: 'Self-attention', transformer: 'Transformer', conv: 'Convolution', pool: 'Max pool', flatten: 'Flatten', gap: 'Global average pool',
    gated: 'Gated (GLU)', batchnorm: 'BatchNorm', noise: 'Gaussian noise', fourier: 'Fourier features', rnn: 'Recurrent', conv1d: 'Conv 1-D', position: 'Positions', seqpool: 'Sequence pool', patches: 'Patches', avgpool: 'Average pool',
    linear: 'Linear model', knn: 'k-nearest neighbours', tree: 'Decision tree', forest: 'Random forest', boost: 'Gradient boosting' };
  const RNN_NAME = { lstm: 'LSTM', gru: 'GRU', rnn: 'RNN' };
  // each recurrent cell's gates in PyTorch's order: [symbol, name, function]
  const RNN_GATES = { lstm: [['i', 'input gate', 'sigmoid'], ['f', 'forget gate', 'sigmoid'], ['g', 'candidate', 'tanh'], ['o', 'output gate', 'sigmoid']], gru: [['r', 'reset gate', 'sigmoid'], ['z', 'update gate', 'sigmoid'], ['n', 'candidate', 'tanh']], rnn: [['h', 'state', 'tanh']] };
  const isClassicB = (t) => !!(NA.cml && NA.cml.isClassic(t));
  // an experiment for each part, added to its explanation: the quickest way to see what it does is to change it
  const TRY = {
    dense: 'halve or double its neurons, train again and compare the two runs in the Runs tab. In the model, click one of its dots to give that neuron its own activation.',
    activation: 'swap the function (ReLU, Tanh, Sine, Gaussian) and watch the boundary or curve change shape.',
    dropout: 'raise p until the validation score drops. With few, noisy examples, compare a run with it and one without.',
    layernorm: 'take it out of a deep stack and see whether training still gets going as quickly.',
    attention: 'open a head, then a query position, to see what it looks at. Turn causal off and see what changes.',
    transformer: 'open its attention, then a head: each head learns its own way of looking. Compare 1 head with 4.',
    conv: 'open a channel to see where its filter fires. Try a stride of 2 instead of a max pool after it.',
    pool: 'swap it for an average pool, or remove it and see how many more parameters the layers after it need.',
    flatten: 'replace it with a Global pool and compare the parameter count and the score.',
    gap: 'replace it with a Flatten and compare the parameter count and the score.',
    gated: 'switch the gate between Sigmoid (GLU), Swish (SwiGLU) and GELU, and compare with a plain Dense layer of the same size.',
    batchnorm: 'put one after each layer of a deep sigmoid network and compare the gradient bars in the Training tab.',
    noise: 'raise σ on a small, noisy problem: the training score falls, and the validation score may rise.',
    fourier: 'raise the scale for wigglier waves; too high, and the fit turns to noise between the examples.',
    rnn: 'switch between LSTM, GRU and plain RNN and compare the runs. Open h at the last step and walk back through time.',
    conv1d: 'make it causal with a dilation of 2 and stack another with 4: each layer sees twice as far back.',
    position: 'take it out in front of attention on a word-order problem and watch the score fall.',
    seqpool: 'switch between last, max and mean: max finds a pattern anywhere, last needs a layer before it that remembers.',
    patches: 'compare patches of 7 with patches of 2: fewer, bigger tokens train much faster, smaller ones see finer detail.',
    avgpool: 'swap it for a max pool on the same model and compare the runs.',
    linear: 'add a ridge penalty λ and watch the weights shrink, and the validation score change.',
    knn: 'set k to 1, then to 25: from following every training point to smoothing over many.',
    tree: 'lower the depth one step at a time and watch the training and validation scores meet.',
    forest: 'compare 1 tree with 50, and inputs per question “all” with “√ of them”: the randomness is what makes the average work.',
    boost: 'set η to 1, then 0.05 with more rounds: big steps fit fast and overfit, small steps generalise.',
  };
  // the attribute each block gets in the Python tab's Model class (js/builder-py.js): self.dense3, self.conv1, …
  const PYMOD = { dense: 'dense', dropout: 'drop', layernorm: 'norm', attention: 'attn', transformer: 'block', conv: 'conv', gated: 'glu', batchnorm: 'bn', noise: 'noise', rnn: 'rnn', conv1d: 'conv', patches: 'patch' };
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
  const cfgOf = (spec) => (isClassicB(spec.type) ? NA.cml.withDefaults(spec) : Object.assign({}, (NA.bx.BLOCKS[spec.type] || {}).defaults, ...Object.keys(spec).filter((k) => spec[k] !== undefined && spec[k] !== null).map((k) => ({ [k]: spec[k] }))));
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
    transformer: ['h1', 'xh1', 'r1', 'mid', 'h2', 'xh2', 'r2', 'u', 'g', 'mo', 'out'], conv: ['z', 'a', 'out'], pool: ['out'], flatten: ['out'], gap: ['out'],
    gated: ['z1', 'z2', 'a2', 'out'], batchnorm: ['out', 'xhat'], noise: ['out'], fourier: ['out'], rnn: ['out', 'gates', 'c'], conv1d: ['z', 'a', 'out'], position: ['out'], seqpool: ['out', 'arg'], patches: ['out'], avgpool: ['out'] };
  let FRESH = null, PROBE = null, PYCODE = null;
  function runProbe(model, d) {
    const va = d.nVa > 0, n = va ? d.nVa : d.nTr;
    if (!n || !d.Xtr) return null;
    const Xs = va ? d.Xva : d.Xtr, Ys = va ? d.Yva : d.Ytr, x = Xs.subarray(0, model.inSize), ySize = Ys.length / n;
    const out = model.predict(x, 1), cp = (a) => (a ? Float32Array.from(a) : null);
    const nb = model.classic && model.lastNeighbours ? Int32Array.from(model.lastNeighbours) : null; // k-NN: the neighbours it just found
    const P = { va, x: Float32Array.from(x), y: Array.from(Ys.subarray(0, ySize)), logits: Float32Array.from(out), layers: [], img: new Map() };
    if (model.embed) { P.ids = Int32Array.from(x); P.emb = cp(model.embed.out); }
    if (nb) P.nb = nb;
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
        try { m = L.isClassicSetup && L.isClassicSetup() ? NA.cml.build(L.setup.blocks, d, { seed: L.setup.train.seed }) : NA.bx.build(L.setup.blocks, d.input, d.output, { seed: L.setup.train.seed, init: L.setup.train.init }); } catch (ex) { e = ex.message; }
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
    function inputShape() { const q = d.input || {}; return q.kind === 'tokens' ? [q.len, q.embed] : q.kind === 'vector' ? [q.dim] : q.kind === 'image' ? [q.c, q.h, q.w] : q.kind === 'series' ? [q.len, q.dim] : null; }
    const blockTitle = (i) => { const b = blocks[i]; return `${i + 1} · ${b.type === 'dense' && b.units === 1 ? 'Neuron' : b.type === 'rnn' ? RNN_NAME[cfgOf(b).cell] || 'LSTM' : NAME[b.type] || b.type}`; };
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
      if (d.table) return d.featNames[idx];
      if (q.kind === 'series') return `x(t−${q.len - 1 - (idx % q.len)})`;
      return q.dim === 1 ? 'x' : `x${NA.sub(idx + 1)}`;
    };
    const exampleText = () => {
      if (!P) return '';
      const fam = d.family;
      if (d.table) {
        const raw = (j) => P.x[j] * d.stats.std[j] + d.stats.mean[j], show = d.featNames.slice(0, 3).map((nm, j) => `${nm} ${fmt(raw(j), 2)}`).join(', ');
        return `a row with ${show}${d.featNames.length > 3 ? ', …' : ''}, whose ${d.targetName} is ${d.kind === 'value' ? fmt(target[0] * d.stats.y.std + d.stats.y.mean, 2) : names[target[0]]}`;
      }
      if (fam === 'plane') return `the point (${fmt(P.x[0], 2)}, ${fmt(P.x[1], 2)}), ${names[target[0]] || 'class ' + (target[0] + 1)}`;
      if (fam === 'curve') return `x = ${fmt(P.x[0], 3)}, whose target is ${fmt(target[0], 3)}`;
      if (fam === 'surface') return `the point (${fmt(P.x[0], 2)}, ${fmt(P.x[1], 2)}), whose height is ${fmt(target[0], 3)}`;
      if (fam === 'series') return `the last ${P.x.length} values of the series, whose next one is ${fmt(target[0], 3)}`;
      if (fam === 'digits') return d.task === 'shapes' ? `a drawn ${names[target[0]]}` : `a handwritten ${names[target[0]]}`;
      const words = Array.from(P.ids, word);
      if (fam === 'seqclass') { const pad = d.pad >= 0 ? word(d.pad) : null; let n = words.length; while (pad !== null && n > 1 && words[n - 1] === pad) n--; return `"${words.slice(0, n).join(' ')}", which is ${names[target[0]]}`; }
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
        case 'conv': return convValue(bi, idx);
        case 'pool': return poolValue(bi, idx);
        case 'flatten': return Object.assign(valueNode(bi - 1, idx), { sub: `number ${idx} of the flattened vector` });
        case 'gap': return gapValue(bi, idx);
        case 'rnn': return rnnValue(bi, p.t, p.c);
        case 'conv1d': return conv1dValue(bi, idx);
        case 'seqpool': return seqPoolValue(bi, idx);
        case 'patches': return patchValue(bi, p.t, p.c);
        case 'batchnorm': return bnValue(bi, idx);
        case 'avgpool': return avgPoolValue(bi, idx);
        case 'noise': return Object.assign(valueNode(bi - 1, idx), { sub: `passed through block ${bi + 1} unchanged (noise is only added while training)` });
        case 'fourier': return fourierValue(bi, idx);
        case 'gated': return gatedValue(bi, idx);
        case 'position': return positionValue(bi, p.t, p.c);
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

    // ---- a product of two numbers (a gate times a value), and a function of a number that keeps the number's story
    function prodNode(title, formula, a, b, aNode, bNode, about, code) {
      const rs = W2([[0, 0.36, 0.31, 0.58], [0.345, 0.36, 0.31, 0.58], [0.69, 0.36, 0.31, 0.58]], [[0, 0.3, 1, 0.2], [0, 0.54, 1, 0.2], [0, 0.78, 1, 0.2]]);
      return Dp.lazy({
        title, sub: `${fmt(a, 3)} × ${fmt(b, 3)} = ${fmt(a * b, 3)}`, tone: 's7',
        summary: (ctx, r, S) => S.text(ctx, `${fmt(a, 2)} × ${fmt(b, 2)}`, r.x + r.w / 2, r.y + r.h * 0.4, { size: clamp(r.w / 10, 6, 11), mono: true, align: 'center', color: theme.c('ink-2'), maxW: r.w }),
        inner: (ctx, r, S) => S.lines(ctx, [formula, `${fmt(a * b, 4)} = ${fmt(a, 4)} × ${fmt(b, 4)}`, 'Zoom into either factor, or into the multiplication itself.'], S.place(r, [0, 0.02, 1, 0.24]), { mono: true, size: S.u(r, 8, 13), strong: (k) => k === 0 }),
        about, code,
      }, () => [
        Object.assign(aNode(), { id: 'a', rect: rs[0] }),
        Object.assign(bNode(), { id: 'b', rect: rs[1] }),
        Object.assign(Dp.fmul(X, a, b), { id: 'mul', rect: rs[2], title: 'The multiplication' }),
      ]);
    }
    function fnNode(title, lib, z, zNode, about) {
      const f = lib === 'tanh' ? Math.tanh(z) : lib === 'sigmoid' ? 1 / (1 + Math.exp(-z)) : Math.sin(z), rs = W2([[0, 0.3, 0.48, 0.66], [0.52, 0.3, 0.48, 0.66]], [[0, 0.28, 1, 0.33], [0, 0.65, 1, 0.33]]);
      return Dp.lazy({
        title, sub: `${lib}(${fmt(z, 3)}) = ${fmt(f, 3)}`, tone: 's7',
        summary: (ctx, r, S) => S.text(ctx, `${lib}(${fmt(z, 2)})`, r.x + r.w / 2, r.y + r.h * 0.4, { size: clamp(r.w / 10, 6, 11), mono: true, align: 'center', color: theme.c('ink-2'), maxW: r.w }),
        inner: (ctx, r, S) => S.lines(ctx, [`${lib}(${fmt(z, 4)}) = ${fmt(f, 4)}`, 'Zoom into the number it is computed on, or into the function.'], S.place(r, [0, 0.02, 1, 0.2]), { mono: true, size: S.u(r, 8, 13), strong: (k) => k === 0 }),
        about,
      }, () => [Object.assign(zNode(), { id: 'z', rect: rs[0] }), Object.assign(Dp.lib(X, lib, { z }), { id: 'f', rect: rs[1] })]);
    }
    // boxes in rows of cols across the level, from y0 to y1
    const boxGrid = (n, cols, y0, y1) => { const rows = Math.ceil(n / cols), gx = 0.03, gy = 0.04, w = (1 - gx * (cols - 1)) / cols, h = (y1 - y0 - gy * (rows - 1)) / rows; return Array.from({ length: n }, (_, k) => [(k % cols) * (w + gx), y0 + Math.floor(k / cols) * (h + gy), w, h]); };

    // ---- recurrent layers: the gates, the memory and the state, one step at a time
    const GATE_DOES = {
      i: 'how much of the candidate to write into the memory (0: nothing, 1: all of it)', f: 'how much of the memory to keep from the step before (0: forget it, 1: keep it all)',
      g: 'the new content on offer for the memory, between −1 and 1', o: 'how much of the memory to show as the output h',
      r: 'how much of the old state the candidate may use', z: 'how much of the old state to keep (1) rather than replace with the candidate (0)', n: 'the new state on offer, made from the input and the reset old state',
    };
    const rnnHook = (mod, t, H) => ['grab = {}', `hook = ${mod}.register_forward_hook(lambda m, inp, out: grab.update(x=inp[0][0], h=out[0][0]))`, 'with torch.no_grad(): model(Xva[:1])                # the first validation example, as here', 'hook.remove()', `x_t, h_prev = grab['x'][${t}], ${t ? `grab['h'][${t - 1}]` : `torch.zeros(${H})`}`];
    // the pre-activation of gate q of unit j at step t: x·W_ih + b_ih + h₋₁·W_hh + b_hh (a GRU's candidate scales the h part by its reset gate)
    function rnnGate(bi, t, q, j) {
      const lay = model.layers[bi], { D, H, cell, GH } = lay, r = P.layers[bi], xin = inVals(bi), row = q * H + j, Wi = lay.Wih.w, Wh = lay.Whh.w, [sym, what, fn] = RNN_GATES[cell][q];
      const reset = cell === 'gru' && q === 2 ? r.gates[t * GH + j] : null, rf = reset === null ? 1 : reset, mod = pyName(bi), rs = reset === null ? '' : 'r * ';
      const terms = [];
      for (let p = 0; p < D; p++) terms.push({ a: xin[t * D + p], w: Wi[row * D + p] });
      for (let p = 0; p < H; p++) terms.push({ a: t ? r.out[(t - 1) * H + p] : 0, w: rf * Wh[row * H + p] });
      const nm = cell === 'rnn' ? `h[${j}]` : `${what} ${sym}[${j}]`, act = actOf(fn), z = terms.reduce((s, u) => s + u.a * u.w, lay.bih.w[row] + rf * lay.bhh.w[row]);
      return Dp.neuron(X, {
        title: `${blockTitle(bi)} · ${nm} at ${tok(t)}`, sub: `${D} + ${H} inputs, ${fn}`, tone: 's7',
        terms, bias: lay.bih.w[row] + rf * lay.bhh.w[row], act, outName: sym,
        inLabel: (p) => (p < D ? nameOf(bi - 1, t * D + p) : `h₋₁[${p - D}]`), wLabel: (p) => (p < D ? `W_ih[${row}, ${p}]` : `${reset === null ? '' : 'r · '}W_hh[${row}, ${p - D}]`),
        inputNode: (p) => (p < D ? valueNode(bi - 1, t * D + p) : t ? rnnValue(bi, t - 1, p - D) : Object.assign(Dp.stored(X, 0, 'the state before the first step'), { sub: 'h starts at 0' })),
        rowNames: [`inputs: x at ${tok(t)} (${D}), then h from the step before (${H})`, `weights: row ${row} of W_ih, then of W_hh${reset === null ? '' : ' times r'} (learned)`, 'products wᵢ·aᵢ'],
        vars: { w: `torch.cat([${mod}.weight_ih_l0[${row}], ${rs}${mod}.weight_hh_l0[${row}]])`, a: 'torch.cat([x_t, h_prev])', b: `(${mod}.bias_ih_l0[${row}] + ${rs}${mod}.bias_hh_l0[${row}])` },
        codeHead: [...rnnHook(mod, t, H), ...(reset === null ? [] : [`r = torch.sigmoid(${mod}.weight_ih_l0[${j}] @ x_t + ${mod}.bias_ih_l0[${j}] + ${mod}.weight_hh_l0[${j}] @ h_prev + ${mod}.bias_hh_l0[${j}])   # the reset gate`])],
        codeWhat: `as row ${row} of ${mod}, an nn.${RNN_NAME[cell]}(${D}, ${H}, batch_first=True)`,
        about: [
          cell === 'rnn' ? `Unit ${j + 1} at ${tok(t)}: a neuron that sees the ${D} numbers of this step and the ${H} outputs the whole layer gave one step earlier, h = tanh(W_ih·x + W_hh·h₋₁ + b) = ${fmt(Math.tanh(z), 4)}. Those last ${H} inputs are the loop that gives the layer its memory.`
            : `The ${what} of unit ${j + 1} at ${tok(t)}, ${sym} = ${fn === 'tanh' ? 'tanh' : 'σ'}(${fmt(z, 3)}) = ${fmt(act.f(z), 4)}: ${GATE_DOES[sym]}. It is a neuron like any other, reading the ${D} numbers of this step and the ${H} numbers of h from the step before.${reset !== null ? ` The reset gate r = ${fmt(reset, 3)} scales the part that comes from h₋₁ before it is added.` : ''}`,
          `Zoom into one of the last ${H} input bars to open h at the step before, and from there the step before that: the same weights at every step, which is how the layer learns from whole sequences.`,
        ],
      });
    }
    // h at step t, unit j: o·tanh(c) for an LSTM, (1 − z)·n + z·h₋₁ for a GRU, the neuron itself for a plain RNN
    function rnnValue(bi, t, j) {
      const lay = model.layers[bi], { H, cell, GH } = lay, r = P.layers[bi], g = (q) => r.gates[t * GH + q * H + j], title = `${blockTitle(bi)} · h[${j}] at ${tok(t)}`, mod = pyName(bi);
      const code = [...rnnHook(mod, t, H), `print(grab['h'][${t}, ${j}])                            # h[${j}] at this step`].join('\n');
      if (cell === 'rnn') return rnnGate(bi, t, 0, j);
      if (cell === 'lstm') {
        const o = g(3), cv = r.c[t * H + j], th = Math.tanh(cv);
        return prodNode(title, 'h = o · tanh(c): the output gate decides how much of the memory to show', o, th, () => rnnGate(bi, t, 3, j),
          () => fnNode(`tanh(c[${j}]) at ${tok(t)}`, 'tanh', cv, () => lstmCell(bi, t, j), [`The memory c squashed into −1…1 by tanh before the output gate scales it. Zoom into c for how the memory got there.`]),
          [`What unit ${j + 1} of the LSTM hands on at ${tok(t)}: its memory c = ${fmt(cv, 3)}, squashed by tanh to ${fmt(th, 3)}, times the output gate o = ${fmt(o, 3)}. With o near 0 the unit keeps its memory to itself; near 1 it shows it.`], code);
      }
      const z = g(1), n = g(2), hp = t ? r.out[(t - 1) * H + j] : 0;
      return sumNode(title, 'h = (1 − z) · n + z · h₋₁: the update gate mixes the new candidate with the old state', (1 - z) * n, z * hp,
        () => prodNode(`(1 − z) · n: the new part`, `1 − z = ${fmt(1 - z, 3)} of the candidate n`, 1 - z, n, () => Object.assign(rnnGate(bi, t, 1, j), { sub: `the factor is 1 − z = ${fmt(1 - z, 3)}` }), () => rnnGate(bi, t, 2, j), [`The share of the new candidate that goes into the state: 1 − z of it. The first factor opens the update gate z (the share is 1 minus it), the second the candidate n.`]),
        () => prodNode(`z · h₋₁: the old part`, `z = ${fmt(z, 3)} of the state before`, z, hp, () => rnnGate(bi, t, 1, j), () => (t ? rnnValue(bi, t - 1, j) : Object.assign(Dp.stored(X, 0, 'the state before the first step'), { sub: 'h starts at 0' })), [`The share of the old state that is kept: z of it. Zoom into h₋₁ to go one step back in time.`]),
        [`What unit ${j + 1} of the GRU hands on at ${tok(t)}: the update gate z = ${fmt(z, 3)} keeps that share of the old state and takes the rest from the candidate n = ${fmt(n, 3)}. A z near 1 carries the old state through unchanged, which is how a GRU remembers across many steps.`], code);
    }
    // an LSTM's memory: c = f·c₋₁ + i·g
    function lstmCell(bi, t, j) {
      const lay = model.layers[bi], { H, GH } = lay, r = P.layers[bi], g = (q) => r.gates[t * GH + q * H + j], cp = t ? r.c[(t - 1) * H + j] : 0;
      return sumNode(`${blockTitle(bi)} · memory c[${j}] at ${tok(t)}`, 'c = f · c₋₁ + i · g: keep some of the old memory, write some new', g(1) * cp, g(0) * g(2),
        () => prodNode('f · c₋₁: what is kept', `the forget gate f = ${fmt(g(1), 3)} times the memory before`, g(1), cp, () => rnnGate(bi, t, 1, j), () => (t ? lstmCell(bi, t - 1, j) : Object.assign(Dp.stored(X, 0, 'the memory before the first step'), { sub: 'c starts at 0' })), [`What survives of the memory: the forget gate f (between 0 and 1) times the memory from the step before. f near 1 keeps it all; near 0 wipes it.`]),
        () => prodNode('i · g: what is written', `the input gate i = ${fmt(g(0), 3)} times the candidate g = ${fmt(g(2), 3)}`, g(0), g(2), () => rnnGate(bi, t, 0, j), () => rnnGate(bi, t, 2, j), [`What is written into the memory: the candidate g (new content between −1 and 1) times the input gate i (how much of it to let in).`]),
        [`The LSTM's memory for unit ${j + 1} at ${tok(t)}: the forget gate keeps a share of the memory from the step before and the input gate adds a share of new content. Because the memory is carried by an addition, not squashed through a function at every step, it (and its gradient) can last for many steps.`, 'Zoom into c₋₁ to go one step back in time.']);
    }

    // ---- 1-D convolutions along a sequence
    function conv1dNeuron(bi, t, f) {
      const lay = model.layers[bi], { T, D, k, dil, left } = lay, xin = inVals(bi), w = lay.Wt.w, J = k * D, act = actOf(lay.fn.id), mod = pyName(bi), reach = (k - 1) * dil;
      const terms = [], src = [], tag = [];
      for (let j = 0; j < k; j++) { const s = t - left + j * dil, inside = s >= 0 && s < T; for (let c = 0; c < D; c++) { terms.push({ a: inside ? xin[s * D + c] : 0, w: w[f * J + j * D + c] }); src.push(inside ? s * D + c : -1); tag.push([j, c, s]); } }
      const z = terms.reduce((q, u) => q + u.a * u.w, lay.b.w[f]), pads = src.filter((v) => v < 0).length / D;
      return Dp.neuron(X, {
        title: `${blockTitle(bi)} · filter ${f + 1} at ${tok(t)}`, sub: `${J} inputs${act ? ', ' + act.label : ''}`, tone: 's7',
        terms, bias: lay.b.w[f], act, outName: 'a',
        inLabel: (p) => (src[p] < 0 ? `padding (position ${tag[p][2]})` : nameOf(bi - 1, src[p])), wLabel: (p) => `kernel[${f}, ${tag[p][1]}, ${tag[p][0]}]`,
        inputNode: (p) => (src[p] < 0 ? Object.assign(Dp.stored(X, 0, 'zero padding'), { sub: tag[p][2] < 0 ? 'before the start of the sequence' : 'past its end' }) : valueNode(bi - 1, src[p])),
        rowNames: [`inputs: ${k} position${k === 1 ? '' : 's'}${dil > 1 ? ` ${dil} apart` : ''} × ${D} feature${D === 1 ? '' : 's'} (${J} numbers)`, 'the filter’s weights (learned)', 'products wᵢ·aᵢ'],
        vars: { w: `${mod}.weight[${f}].T.flatten()`, a: 'window.T.flatten()', b: `${mod}.bias[${f}]` },
        codeHead: [...pyGrab(mod, 'x', 'inp[0]'), `window = grab['x'][0, :, ${t}:${t + reach + 1}${dil > 1 ? `:${dil}` : ''}]          # (${D}, ${k}): the input arrives padded and channels-first`],
        codeWhat: `as one output of ${mod}, an nn.Conv1d(${D}, ${lay.F}, ${k}${dil > 1 ? `, dilation=${dil}` : ''})`,
        about: [
          `Filter ${f + 1} at ${tok(t)}: it reads positions ${tag[0][2]} to ${tag[tag.length - 1][2]}${dil > 1 ? ` (every ${dil}th)` : ''}${lay.left === reach ? ', ending at this one: a causal filter never sees what comes later' : ', around this one'}, all ${D} feature${D === 1 ? '' : 's'} of each, multiplies the ${J} numbers by its weights and adds them up with its bias: z = ${fmt(z, 4)}${act ? `, then ${act.label} gives ${fmt(act.f(z), 4)}` : ''}.${pads ? ` ${pads} of the positions fall outside the sequence and count as 0 (zero padding).` : ''}`,
          'The same filter slides along the whole sequence: a pattern is found wherever it occurs. Zoom into an input bar to open the number of the block below.',
        ],
      });
    }
    function conv1dValue(bi, idx) {
      const lay = model.layers[bi], t = Math.floor(idx / lay.F), f = idx % lay.F;
      if (!lay.residual) return conv1dNeuron(bi, t, f);
      return sumNode(`${nameOf(bi, idx)}: input + filter`, 'out = x + a   (a residual connection)', inVals(bi)[idx], P.layers[bi].a[idx], () => valueNode(bi - 1, idx), () => conv1dNeuron(bi, t, f), residualAbout(`filter ${f + 1}'s output`), `out = x + a                                        # the same shape in and out, so they can be added`);
    }
    // a 2-D convolution's output: the filter's number, plus the input when the block is residual
    function convValue(bi, idx) {
      const lay = model.layers[bi], p = posOf(outS(bi), idx);
      if (!lay.residual) return convNeuron(bi, p.c, p.y, p.x);
      return sumNode(`${nameOf(bi, idx)}: input + filter`, 'out = x + a   (a residual connection, as in a ResNet)', inVals(bi)[idx], P.layers[bi].a[idx], () => valueNode(bi - 1, idx), () => convNeuron(bi, p.c, p.y, p.x), residualAbout(`channel ${p.c + 1}'s filter output`), 'out = x + a                                        # the shape is kept (same channels, stride 1), so they can be added');
    }

    // ---- sequence pooling, patches, average pooling
    function seqPoolValue(bi, c) {
      const lay = model.layers[bi], { T, D, mode } = lay, xin = inVals(bi);
      if (mode === 'mean') return gapValue(bi, c);
      if (mode === 'last') return Object.assign(valueNode(bi - 1, (T - 1) * D + c), { sub: `feature ${c} at the last position, kept as it is` });
      const cells = Array.from({ length: T }, (_, t) => ({ v: xin[t * D + c], at: t * D + c, t })), best = cells.reduce((p, q) => (q.v > p.v ? q : p)), vals = Float32Array.from(cells.map((q) => q.v));
      const br = W2([0, 0.32, 0.6, 0.56], [0, 0.3, 1, 0.3]);
      return Dp.lazy({
        title: `${blockTitle(bi)} · max of feature ${c}`, sub: `the largest of ${T}: ${fmt(best.v, 3)}`, tone: 's7',
        summary: (ctx, q, S) => S.bars(ctx, vals, q, { color: (k) => (k === best.t ? theme.c('s7') : theme.c('muted', 0.6)) }),
        inner: (ctx, q, S) => {
          const u = S.u(q, 8, 13), b2 = S.place(q, br);
          S.lines(ctx, [`y[${c}] = the largest of feature ${c} over the ${T} positions = ${fmt(best.v, 4)}`, `found at ${tok(best.t)}`], S.place(q, [0, 0, 1, 0.2]), { mono: true, size: u * 0.92, strong: (k) => k === 0 });
          S.caption(ctx, `feature ${c} at every position, the largest marked: zoom into a bar`, b2, { size: u * 0.8, minW: 200 });
          S.bars(ctx, vals, b2, { color: (k) => (k === best.t ? theme.c('s7') : theme.c('muted', 0.6)) });
        },
        grids: [{ id: 'seq', rect: br, rows: 1, cols: T, make: (i, j) => valueNode(bi - 1, cells[j].at) }],
        about: [`Max pooling over the sequence keeps each feature's largest value, wherever it was: here ${fmt(best.v, 4)}, at ${tok(best.t)}. It answers "did this pattern appear anywhere?", which is what a sequence classifier often needs, and forgets where.`],
        code: `x = torch.randn(1, ${T}, ${D})\ny = x.max(1).values                                # (1, ${D})\nprint(y[0, ${c}] == x[0, :, ${c}].max())           # True`,
      }, () => [Object.assign(valueNode(bi - 1, best.at), { id: 'max', rect: W2([0.66, 0.32, 0.34, 0.56], [0, 0.68, 1, 0.3]), title: `The winner, at ${tok(best.t)}` })]);
    }
    // patch n's pixels, in the order the projection reads them (channel, row, column)
    function patchSrc(lay, n) {
      const { C, H, p, nw } = lay, Wd = lay.W, py0 = Math.floor(n / nw) * p, px0 = (n % nw) * p, src = [];
      for (let c = 0; c < C; c++) for (let yy = 0; yy < p; yy++) for (let xx = 0; xx < p; xx++) src.push((c * H + py0 + yy) * Wd + px0 + xx);
      return src;
    }
    function patchProj(bi, n, e) {
      const lay = model.layers[bi], { J, p, C } = lay, xin = inVals(bi), src = patchSrc(lay, n), Wp = lay.Wp.w, mod = pyName(bi);
      return Dp.neuron(X, {
        title: `${blockTitle(bi)} · patch ${n + 1}, number ${e}: the projection`, sub: `${J} pixels, no activation`, tone: 's4',
        terms: src.map((at, q) => ({ a: xin[at], w: Wp[e * J + q] })), bias: lay.bp.w[e], act: null, zName: 'e', outName: 'e',
        inLabel: (q) => nameOf(bi - 1, src[q]), wLabel: (q) => `W[${e}, ${q}]`, inputNode: (q) => valueNode(bi - 1, src[q]),
        rowNames: [`the ${p} × ${p} patch${C > 1 ? ` through ${C} channels` : ''}: ${J} numbers`, 'projection weights (learned, the same for every patch)', 'products wᵢ·aᵢ'],
        vars: { w: `${mod}.weight[${e}].flatten()`, a: 'patch', b: `${mod}.bias[${e}]` },
        codeHead: [...pyGrab(mod, 'x', 'inp[0]'), `patch = grab['x'][0, :, ${Math.floor(n / lay.nw) * p}:${Math.floor(n / lay.nw) * p + p}, ${(n % lay.nw) * p}:${(n % lay.nw) * p + p}].flatten()   # ${J} numbers`],
        codeWhat: `as one number of ${mod}, an nn.Conv2d(${C}, ${lay.D}, ${p}, stride=${p}): a patch projection`,
        about: [`Number ${e} of patch ${n + 1}'s vector, before its place is added: a weighted sum of the patch's ${J} pixels with the projection's weights, plus a bias. Every patch goes through the same projection, as every token goes through the same embedding.`],
      });
    }
    function patchValue(bi, n, e) {
      const lay = model.layers[bi], D = lay.D, xin = inVals(bi), src = patchSrc(lay, n), pos = lay.pos.w[n * D + e];
      const proj = src.reduce((s, at, q) => s + xin[at] * lay.Wp.w[e * lay.J + q], lay.bp.w[e]);
      return sumNode(`${blockTitle(bi)} · patch ${n + 1}, number ${e}`, 'out = W · patch + b + pos[n]: what the patch shows, plus where it is', proj, pos, () => patchProj(bi, n, e), () => Dp.stored(X, pos, `pos[${n}, ${e}]: patch ${n + 1}'s place (learned)`),
        [`A patch's vector is its projection plus a learned vector for its place in the picture. Without the place, attention would see the patches as an unordered bag and could not tell top from bottom.`], `out = ${pyName(bi)}(x).flatten(2).transpose(1, 2) + model.pos${bi + 1}   # (1, ${lay.N}, ${D})`);
    }
    function avgPoolValue(bi, idx) {
      const lay = model.layers[bi], { C, H, S: Sz, Ho, Wo } = lay, Wd = lay.W, xin = inVals(bi), c = Math.floor(idx / (Ho * Wo)), oy = Math.floor((idx % (Ho * Wo)) / Wo), ox = idx % Wo, n = Sz * Sz, cells = [];
      for (let dy = 0; dy < Sz; dy++) for (let dx = 0; dx < Sz; dx++) cells.push((c * H + oy * Sz + dy) * Wd + ox * Sz + dx);
      return Dp.neuron(X, {
        title: `${blockTitle(bi)} · channel ${c + 1} at (${ox}, ${oy})`, sub: `${n} numbers × 1/${n}`, tone: 's4',
        terms: cells.map((at) => ({ a: xin[at], w: 1 / n })), bias: null, act: null, outName: 'y',
        rowNames: [`the ${Sz} × ${Sz} window of channel ${c + 1}`, `weights: all 1/${n}`, 'products'], inLabel: (p) => nameOf(bi - 1, cells[p]), wLabel: () => `1/${n}`,
        inputNode: (p) => valueNode(bi - 1, cells[p]), vars: { w: `torch.full((${n},), 1 / ${n})`, a: `x[0, ${c}, ${oy * Sz}:${oy * Sz + Sz}, ${ox * Sz}:${ox * Sz + Sz}].flatten()`, b: '' }, codeHead: [`x = torch.randn(1, ${C}, ${H}, ${Wd})                  # an input of this shape`],
        about: [`Average pooling: the mean of a ${Sz} × ${Sz} window, a weighted sum whose weights are all 1/${n}. Unlike max pooling every number in the window counts, so it keeps a smoother summary of the picture as it shrinks.`],
      });
    }

    // ---- batch norm, Fourier features, gated layers, positions
    function bnValue(bi, idx) {
      const lay = model.layers[bi], c = lay.chan[idx], x = inVals(bi)[idx], mu = lay.rmean[c], v = lay.rvar[c], rs = 1 / Math.sqrt(v + 1e-5), g = lay.g.w[c], bb = lay.b.w[c], d0 = x - mu, xh = d0 * rs, y = xh * g + bb;
      const what = lay.kind === 3 ? `channel ${c + 1}` : `feature ${c}`, R = wide ? [...boxGrid(3, 3, 0.3, 0.6), ...boxGrid(4, 4, 0.66, 0.98)] : boxGrid(7, 2, 0.27, 0.99);
      return Dp.lazy({
        title: `${blockTitle(bi)} · number ${idx} (${what})`, sub: `= ${fmt(y, 4)}`, tone: 's1',
        summary: (ctx, r, S) => S.text(ctx, `(x − μ)/σ · γ + β`, r.x + r.w / 2, r.y + r.h * 0.4, { size: clamp(r.w / 12, 6, 11), mono: true, align: 'center', color: theme.c('ink-2'), maxW: r.w }),
        inner: (ctx, r, S) => S.lines(ctx, ['y = (x − μ) / √(σ² + ε) · γ + β', `  = (${fmt(x, 3)} − ${fmt(mu, 3)}) / √(${fmt(v, 3)} + 0.00001) · ${fmt(g, 3)} + ${fmt(bb, 3)} = ${fmt(y, 4)}`, `μ, σ²: running averages of ${what}'s batch statistics, kept while training`], S.place(r, [0, 0, 1, 0.24]), { mono: true, size: S.u(r, 8, 13), strong: (k) => k === 0 }),
        about: [`Batch norm on one number, the way it runs when the model is used: subtract the running mean μ of ${what}, divide by the running spread √(σ² + ε), then scale by the learned γ and shift by the learned β. While training, the batch's own mean and variance are used instead, and the running averages follow them.`, 'The top row holds the numbers it reads; the bottom row is the four steps of arithmetic, each one float32 operation.'],
        code: `norm = ${pyName(bi)}\nnorm.eval()                                        # in use: the running averages\nmu, var = norm.running_mean[${c}], norm.running_var[${c}]\nprint((${NA.fmtCode(x, 5)} - mu) / torch.sqrt(var + norm.eps) * norm.weight[${c}] + norm.bias[${c}])`,
      }, () => [
        Object.assign(valueNode(bi - 1, idx), { id: 'x', rect: R[0], title: `x = ${fmt(x, 3)}` }),
        Object.assign(Dp.stored(X, mu, `running mean μ of ${what}`), { id: 'mu', rect: R[1] }),
        Object.assign(Dp.stored(X, v, `running variance σ² of ${what}`), { id: 'var', rect: R[2] }),
        Object.assign(Dp.fadd(X, x, -mu), { id: 'sub', rect: R[3], title: 'x − μ' }),
        Object.assign(Dp.fmul(X, d0, rs), { id: 'scale', rect: R[4], title: `÷ √(σ² + ε), as × ${fmt(rs, 3)}` }),
        Object.assign(Dp.fmul(X, xh, g), { id: 'gain', rect: R[5], title: '× γ (learned)' }),
        Object.assign(Dp.fadd(X, xh * g, bb), { id: 'shift', rect: R[6], title: '+ β (learned)' }),
      ]);
    }
    function fourierValue(bi, k) {
      const lay = model.layers[bi], { F, m } = lay, xin = inVals(bi);
      if (k < F) return Object.assign(valueNode(bi - 1, k), { sub: 'passed on unchanged: the Fourier features keep the input as well' });
      const j = (k - F) % m, cosine = k >= F + m, TAU = 2 * Math.PI, terms = Array.from({ length: F }, (_, i) => ({ a: xin[i], w: TAU * lay.freq[j * F + i] }));
      const n = Dp.neuron(X, {
        title: `${blockTitle(bi)} · ${cosine ? 'cos' : 'sin'} wave ${j + 1}`, sub: `${F} input${F === 1 ? '' : 's'}, fixed random frequencies`, tone: 's1',
        terms, bias: cosine ? Math.PI / 2 : null, act: actOf('sine'), zName: 'theta', outName: 'y',
        inLabel: (i) => nameOf(bi - 1, i), wLabel: (i) => `2π·B[${j}, ${i}] (random, fixed)`, inputNode: (i) => valueNode(bi - 1, i),
        rowNames: ['the input numbers', `2π × frequency row ${j + 1} of B (random, never trained)`, 'products'],
        vars: { w: `2 * math.pi * model.B${bi + 1}[${j}]`, a: 'x', b: cosine ? 'math.pi / 2' : '' }, codeHead: ['x = Xva[0]                                          # the first validation example'],
        about: [`Wave ${j + 1}: θ = 2π·(B row ${j + 1} · x)${cosine ? ' + π/2' : ''}, then ${cosine ? 'cos θ, computed as sin(θ + π/2): a cosine is a sine shifted by a quarter turn' : 'sin θ'}. The frequencies were drawn at random once and never change; the layers after learn how much of each wave to use.`],
      });
      return n;
    }
    function gatedLinear(bi, t, j, which) {
      const lay = model.layers[bi], F = lay.nIn, xin = inVals(bi), W = which === 1 ? lay.W1.w : lay.W2.w, bb = which === 1 ? lay.b1.w : lay.b2.w, act = which === 2 ? actOf(lay.fn.id) : null, mod = pyName(bi), seqL = lay.rows > 1;
      return Dp.neuron(X, {
        title: `${blockTitle(bi)} · ${which === 1 ? 'value' : 'gate'} ${j + 1}${seqL ? ` at ${tok(t)}` : ''}`, sub: `${F} inputs${act ? ', ' + act.label : ', no activation'}`, tone: 's1',
        terms: Array.from({ length: F }, (_, p) => ({ a: xin[t * F + p], w: W[j * F + p] })), bias: bb[j], act, zName: which === 1 ? 'v' : 'z2', outName: which === 1 ? 'v' : 'g',
        inputNode: (p) => valueNode(bi - 1, t * F + p), inLabel: (p) => nameOf(bi - 1, t * F + p), wLabel: (p) => `W${which === 1 ? '₁' : '₂'}[${j}, ${p}]`,
        vars: { w: `${mod}.${which === 1 ? 'value' : 'gate'}.weight[${j}]`, a: 'x', b: `${mod}.${which === 1 ? 'value' : 'gate'}.bias[${j}]` },
        codeHead: [...pyGrab(mod, 'x', 'inp[0][0]'), seqL ? `x = grab['x'][${t}]` : `x = grab['x']`], codeWhat: `as the ${which === 1 ? 'value' : 'gate'} half of ${mod}`,
        about: [which === 1 ? `The value half of unit ${j + 1}: a plain weighted sum of the inputs, with no activation.` : `The gate half of unit ${j + 1}: its own weighted sum, bent by ${act ? act.label : 'its activation'}. It decides how much of the value gets through.`],
      });
    }
    function gatedValue(bi, idx) {
      const lay = model.layers[bi], U = lay.U, t = Math.floor(idx / U), j = idx % U, r = P.layers[bi];
      return prodNode(`${nameOf(bi, idx)}: value × gate`, `out = v · ${ACT(lay.fn.id).label}(z₂)`, r.z1[idx], r.a2[idx], () => gatedLinear(bi, t, j, 1), () => gatedLinear(bi, t, j, 2),
        [`A gated unit multiplies two weighted sums of the same input: a value and a gate. The gate can open (let the value through), shut (0) or even flip it, depending on the input, which lets the layer switch features on and off: the idea behind the GLU layers in most large language models today.`], `out = ${pyName(bi)}.value(x) * ${PYACT[lay.fn.id] ? PYACT[lay.fn.id](`${pyName(bi)}.gate(x)`) : `${pyName(bi)}.gate(x)`}`);
    }
    function positionValue(bi, t, c) {
      const lay = model.layers[bi], D = lay.D, x = inVals(bi)[t * D + c], pv = lay.learned ? lay.pos.w[t * D + c] : lay.table[t * D + c];
      const ang = t / Math.pow(10000, (2 * Math.floor(c / 2)) / D);
      const pe = () => Object.assign(Dp.lib(X, 'sine', { z: c % 2 ? ang + Math.PI / 2 : ang }), { title: `PE[${t}, ${c}] = ${c % 2 ? 'cos' : 'sin'}(${fmt(ang, 3)})`, sub: `angle = ${t} / 10000^(${2 * Math.floor(c / 2)}/${D})` });
      return sumNode(`${nameOf(bi, t * D + c)}: x + position`, `out = x + ${lay.learned ? 'P' : 'PE'}[${t}, ${c}]`, x, pv, () => valueNode(bi - 1, t * D + c), lay.learned ? () => Dp.stored(X, pv, `P[${t}, ${c}]: learned for position ${t}`) : pe,
        [lay.learned ? 'The position table is learned: each position has its own row of numbers, added to whatever arrives there.' : 'The original transformer\'s positions: sine and cosine waves of different speeds, a fixed pattern that is different at every position (and lets the model compare positions by their difference).'], lay.learned ? `out = x + model.pos${bi + 1}[${t}, ${c}]` : `angle = ${t} / 10000 ** (${2 * Math.floor(c / 2)} / ${D})\nprint(math.${c % 2 ? 'cos' : 'sin'}(angle))`);
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

    // a block's level, its explanation ending with an experiment to try
    function blockNode(i) {
      const node = blockNode0(i), t = TRY[blocks[i].type];
      if (!node || !t) return node;
      const desc = Object.getOwnPropertyDescriptor(node, 'about'), own = desc && (desc.get ? () => desc.get.call(node) : () => desc.value);
      Object.defineProperty(node, 'about', { configurable: true, enumerable: true, get: () => [...((own && own()) || []), `Try this: ${t}`] });
      return node;
    }
    function blockNode0(i) {
      const b = blocks[i], c = cfgOf(b), sI = inS(i), sO = outS(i), prob = problemOf(i), base = { id: b.id || 'b' + i, title: blockTitle(i), tone: tone(i) };
      const shp = sI && sO ? `${shapeText(sI)} → ${shapeText(sO)}` : '';
      const have = nums && !prob, lay = have ? model.layers[i] : null, r = have ? P.layers[i] : null, xin = have ? inVals(i) : null;
      const pnum = (shapes[i] || {}).params || 0, mod = pyName(i);
      const plain = (lines, about, code) => Object.assign(base, { inner: plainLevel([...lines, shp ? `${shp}${pnum ? ` · ${fmtInt(pnum)} parameters` : ''}` : ''].filter(Boolean), prob), about: prob ? [...about, `Here it does not fit: ${prob.reason}${fixText(prob)}`] : about, code });
      const sum = (key, s, v) => (ctx, q, S) => miniTensor(ctx, S, key, s, v, q);
      if (isClassicB(b.type)) return classicLevel(i, base, prob);

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
      const more = moreBlock(i, { b, c, sI, sO, base, have, lay, r, xin, pnum, mod, plain, sum });
      if (more) return more;
      return Object.assign(base, { inner: plainLevel([NAME[b.type] || b.type], prob), about: ['A block this view does not know how to draw.'], code: '' });
    }
    // the levels of the newer blocks: recurrent, 1-D convolution, sequence pool, patches, batch norm, average pool, noise, Fourier features, gated layers, positions
    function moreBlock(i, k) {
      const { b, c, sI, sO, base, have, lay, r, xin, pnum, mod, plain, sum } = k;
      const seqLabels = (T) => Array.from({ length: T }, (_, t) => (P && P.ids ? word(P.ids[t]) : String(t)));

      if (b.type === 'rnn') {
        const cell = RNN_GATES[c.cell] ? c.cell : 'lstm', NMc = RNN_NAME[cell], H = c.units, T = sI ? sI[0] : '?', Din = sI ? sI[1] : '?', gl = RNN_GATES[cell];
        base.sub = `${H} units · ${T} steps`;
        const about0 = {
          lstm: `An LSTM reads the sequence one position at a time, left to right, and carries two vectors of ${H} numbers from each step to the next: a memory c and an output h. At every step four gates, each a layer of ${H} neurons looking at the new input and the previous h, decide what to forget from the memory (f), what to write into it (i, with the candidate g) and how much of it to show (o): c = f·c₋₁ + i·g, h = o·tanh(c).`,
          gru: `A GRU reads the sequence one position at a time and carries one vector h of ${H} numbers from step to step. A reset gate r decides how much of the old h the candidate n may use, and an update gate z mixes the old h with the candidate: h = (1 − z)·n + z·h₋₁. It is a lighter LSTM: no separate memory, three gates instead of four.`,
          rnn: `A plain recurrent layer reads the sequence one position at a time: its ${H} neurons see the new input and their own outputs from the step before, h = tanh(W·x + U·h₋₁ + b). That loop is its memory. Over long sequences the memory fades (the gradient through many tanh steps vanishes), which is the problem the LSTM and GRU gates solve.`,
        }[cell];
        const about1 = `The same weights are used at every step: ${fmtInt(pnum)} parameters however long the sequence. The layer hands on h at every position (${T} × ${H}); a sequence pool after it (usually the last position, which has read everything) makes one vector of it.`;
        const formula = { lstm: 'c = f·c₋₁ + i·g;  h = o·tanh(c)', gru: 'h = (1 − z)·n + z·h₋₁', rnn: 'h = tanh(W·x + U·h₋₁ + b)' }[cell];
        const step = {
          lstm: `# one step by hand: four gates from x_t and h_prev, in PyTorch's order i, f, g, o
x_t, h_prev, c_prev = x[0, 0], torch.zeros(${H}), torch.zeros(${H})
gates = rnn.weight_ih_l0 @ x_t + rnn.bias_ih_l0 + rnn.weight_hh_l0 @ h_prev + rnn.bias_hh_l0
i, f, g, o = gates.chunk(4)
c = torch.sigmoid(f) * c_prev + torch.sigmoid(i) * torch.tanh(g)
h0 = torch.sigmoid(o) * torch.tanh(c)`,
          gru: `# one step by hand: the gates r, z and the candidate n, in PyTorch's order
x_t, h_prev = x[0, 0], torch.zeros(${H})
gi = rnn.weight_ih_l0 @ x_t + rnn.bias_ih_l0
gh = rnn.weight_hh_l0 @ h_prev + rnn.bias_hh_l0
r, z = torch.sigmoid(gi[:${H}] + gh[:${H}]), torch.sigmoid(gi[${H}:${2 * H}] + gh[${H}:${2 * H}])
n = torch.tanh(gi[${2 * H}:] + r * gh[${2 * H}:])
h0 = (1 - z) * n + z * h_prev`,
          rnn: `# one step by hand (h_prev is 0 at the first step)
h0 = torch.tanh(rnn.weight_ih_l0 @ x[0, 0] + rnn.bias_ih_l0 + rnn.bias_hh_l0)`,
        }[cell];
        const code = `rnn = ${mod}                                   # nn.${NMc}(${Din}, ${H}, batch_first=True)
x = torch.randn${sI ? pyShape(sI) : '(1, 16, 8)'}
h, _ = rnn(x)                                      # ${sO ? pyShape(sO) : ''}: h at every step
${step}
print(torch.allclose(h0, h[0, 0], atol=1e-6))      # True`;
        if (!have) return plain([formula], [about0, about1], code);
        const GH = lay.GH, labels = seqLabels(T);
        const panels = [['x', `x (${Din})`, null], ...gl.map(([s0, w0], q) => ['g' + s0, cell === 'rnn' ? 'h: tanh(…)' : `${s0}: ${w0}`, q]), ...(cell === 'lstm' ? [['c', 'c: memory', null]] : []), ...(cell === 'rnn' ? [] : [['h', 'h: output', null]])];
        const n = panels.length, LY = { t: [0, 0, 1, 0.15] }, colsT = wide ? n : Math.min(3, n);
        if (wide) { const x0 = 0.06, gap = 0.014, w0 = (1 - x0 - gap * (n - 1)) / n; panels.forEach(([key], m) => { LY[key] = [x0 + m * (w0 + gap), 0.25, w0, 0.72]; }); }
        else { const rows = Math.ceil(n / colsT), x0 = 0.11, gap = 0.03, w0 = (1 - x0 - gap * (colsT - 1)) / colsT, h0 = (0.78 - 0.07 * (rows - 1)) / rows; panels.forEach(([key], m) => { LY[key] = [x0 + (m % colsT) * (w0 + gap), 0.21 + Math.floor(m / colsT) * (h0 + 0.07), w0, h0]; }); }
        const val = (key, q) => (key === 'x' ? (t, p) => xin[t * Din + p] : key === 'c' ? (t, j) => r.c[t * H + j] : key === 'h' ? (t, j) => r.out[t * H + j] : (t, j) => r.gates[t * GH + q * H + j]);
        let fSum = 0;
        if (cell === 'lstm') for (let t = 0; t < T; t++) for (let j = 0; j < H; j++) fSum += r.gates[t * GH + H + j];
        return Object.assign(base, {
          summary: sum(`rns${i}`, sO, r.out),
          inner: (ctx, q, S) => {
            const u = S.u(q, 8, 13);
            S.lines(ctx, [`${NMc}(${Din} → ${H}) over ${T} steps · ${fmtInt(pnum)} parameters · ${formula}`, `time runs down every panel, one row per step${cell === 'lstm' ? ` · the forget gates average ${fmt(fSum / (T * H), 2)} here (1 keeps the memory)` : ''}`, 'zoom into any cell: h opens the step that made it, and that the step before'], S.place(q, LY.t), { mono: true, size: u * 0.86, strong: (k2) => k2 === 0 });
            panels.forEach(([key, cap, qi], m) => {
              const pr = S.place(q, LY[key]), cols = key === 'x' ? Din : H, sig = qi !== null && gl[qi][2] === 'sigmoid';
              S.caption(ctx, cap, pr, { size: u * 0.78, minW: 40 });
              drawHeat(ctx, heatImg(cache, `rn${i}:${key}`, T, cols, val(key, qi), sig ? { seq: true, max: 1 } : key === 'x' || key === 'c' ? {} : { max: 1 }), pr, T, cols);
              if (m % colsT === 0) rowLabels(ctx, S, labels, pr, { size: u * 0.74, maxW: Math.max(16, (wide ? LY.x[0] : 0.11) * q.w - 4) });
            });
          },
          grids: panels.map(([key, , qi]) => ({ id: key, rect: LY[key], rows: T, cols: key === 'x' ? Din : H, make: key === 'x' ? (t, p) => valueNode(i - 1, t * Din + p) : key === 'c' ? (t, j) => lstmCell(i, t, j) : key === 'h' ? (t, j) => rnnValue(i, t, j) : (t, j) => rnnGate(i, t, qi, j) })),
          get about() { return [about0, about1, `Each panel is one quantity at every step (rows) and unit (columns)${cell === 'rnn' ? '' : ': the gates run from 0 (shut, dark) to 1 (open, bright), the candidate and h from −1 to 1'}. Zoom into a cell of h for the formula behind it; its parts open the gates' weighted sums, and h${cell === 'lstm' ? ' or c' : ''} from the step before opens that step: you can walk back through time, the way backpropagation through time sends the gradient.`]; },
          code,
        });
      }

      if (b.type === 'conv1d') {
        const F = c.filters, kk = c.k, dil = c.dilation || 1, causal = !!c.causal, D = sI ? sI[1] : '?', T = sI ? sI[0] : '?', reach = (kk - 1) * dil, left = causal ? reach : Math.floor(reach / 2), res = !!c.residual && !!sI && F === sI[1];
        base.sub = `${F} filters · width ${kk}${dil > 1 ? ` · dilation ${dil}` : ''}${causal ? ' · causal' : ''}${res ? ' · residual' : ''}`;
        const about0 = `A 1-D convolution slides ${F} learned filters along the sequence. Each looks at a window of ${kk} position${kk === 1 ? '' : 's'}${dil > 1 ? `, ${dil} apart (dilated, so it reaches ${reach + 1} positions)` : ''}${causal ? ' ending at the current one (causal: it never sees the future, as a forecaster must not)' : ' centred on the current one'}, through all ${D} feature${D === 1 ? '' : 's'}, and adds up the products with its weights: one output channel per filter, high where its pattern is. Then ${ACT(c.act).label}.${res ? ' The input is added back (residual).' : ''} Stacks of dilated causal convolutions (a TCN, or WaveNet) see far back with few layers.`;
        const code = `conv = ${mod}                                  # nn.Conv1d(${D}, ${F}, ${kk}${dil > 1 ? `, dilation=${dil}` : ''})
x = torch.randn${sI ? pyShape(sI) : '(1, 16, 8)'}                    # positions × features
xp = F.pad(x.transpose(1, 2), (${left}, ${reach - left}))             # channels first, padded ${causal ? 'on the left only (causal)' : 'on both sides'}
a = ${PYACT[c.act] ? PYACT[c.act]('conv(xp)') : 'conv(xp)'}.transpose(1, 2)    # ${sO ? pyShape(sO) : ''}: back to positions × filters${res ? '\nout = x + a                                        # the residual' : ''}`;
        if (!have) return plain([`a[t, f] = ${ACT(c.act).label}(b[f] + Σ W[f] · window at t)`], [about0], code);
        const J = kk * D, w = lay.Wt.w, shown = r.out, labels = seqLabels(T);
        const LY = W2({ t: [0, 0, 1, 0.15], x: [0.08, 0.25, 0.26, 0.72], k: [0.4, 0.25, 0.24, 0.72], a: [0.72, 0.25, 0.28, 0.72] }, { t: [0, 0, 1, 0.15], x: [0.14, 0.19, 0.86, 0.24], k: [0.14, 0.48, 0.86, 0.16], a: [0.14, 0.71, 0.86, 0.27] });
        return Object.assign(base, {
          summary: sum(`c1s${i}`, sO, r.out),
          inner: (ctx, q, S) => {
            const u = S.u(q, 8, 13), at = (key) => S.place(q, LY[key]), xr = at('x'), kr = at('k'), ar = at('a');
            S.lines(ctx, [`Conv1d(${D}, ${F}, ${kk}${dil > 1 ? `, dilation ${dil}` : ''})${causal ? ', causal' : ''} + ${ACT(c.act).label}${res ? ' + x' : ''}`, `each output reads ${kk} position${kk === 1 ? '' : 's'}${dil > 1 ? `, ${dil} apart` : ''}${causal ? ' up to its own' : ' around its own'} · ${fmtInt(pnum)} parameters`], at('t'), { mono: true, size: u * 0.9, strong: (k2) => k2 === 0 });
            S.caption(ctx, `input: ${T} positions × ${D}`, xr, { size: u * 0.8, minW: 120 }); drawHeat(ctx, heatImg(cache, `c1x${i}`, T, D, (t, p) => xin[t * D + p]), xr, T, D);
            rowLabels(ctx, S, labels, xr, { size: u * 0.78, maxW: Math.max(18, xr.x - q.x - 2) });
            S.caption(ctx, `filters: ${F} × (${kk} positions × ${D})`, kr, { size: u * 0.8, minW: 120 }); drawHeat(ctx, heatImg(cache, `c1k${i}`, F, J, (f, p) => w[f * J + p]), kr, F, J);
            S.caption(ctx, `${res ? 'out = x + a' : 'output a'}: ${T} × ${F}, zoom into one`, ar, { size: u * 0.8, minW: 120 }); drawHeat(ctx, heatImg(cache, `c1a${i}`, T, F, (t, f) => shown[t * F + f]), ar, T, F);
            if (!wide) rowLabels(ctx, S, labels, ar, { size: u * 0.78, maxW: Math.max(18, ar.x - q.x - 2) });
          },
          grids: [
            { id: 'x', rect: LY.x, rows: T, cols: D, make: (t, p) => valueNode(i - 1, t * D + p) },
            { id: 'k', rect: LY.k, rows: F, cols: J, make: (f, p) => Dp.stored(X, w[f * J + p], `kernel[${f}, ${p % D}, ${Math.floor(p / D)}]: filter ${f + 1}'s weight on feature ${p % D} at window position ${Math.floor(p / D)}`) },
            { id: 'a', rect: LY.a, rows: T, cols: F, make: (t, f) => conv1dValue(i, t * F + f) },
          ],
          get about() { return [about0, `Here ${fmtInt(pnum)} parameters make ${fmtInt(shown.length)} outputs: the same filters at every position. Each row of the filter picture is one filter, its ${kk} position${kk === 1 ? '' : 's'} side by side. Zoom into an output for the window behind it.`]; },
          code,
        });
      }

      if (b.type === 'seqpool') {
        const mode = c.mode, T = sI ? sI[0] : '?', D = sI ? sI[1] : '?';
        base.sub = { last: 'keeps the last position', max: 'max over positions', mean: 'mean over positions' }[mode] || '';
        const about0 = { last: 'Keeps the vector at the last position and drops the rest. After a recurrent layer the last position has read the whole sequence, so it is the usual summary to classify or forecast from.', max: 'Keeps each feature\'s largest value over all the positions: whether a pattern appeared anywhere in the sequence, no matter where.', mean: 'Averages each feature over the positions: how much of each pattern there was in all, no matter where.' }[mode];
        const code = `x = torch.randn${sI ? pyShape(sI) : '(1, 16, 32)'}\ny = ${mode === 'last' ? 'x[:, -1]' : mode === 'max' ? 'x.max(1).values' : 'x.mean(1)'}                                  # ${sO ? pyShape(sO) : ''}: one vector for the whole sequence`;
        if (!have) return plain([{ last: 'y = x[last position]', max: 'y[c] = max over t of x[t, c]', mean: 'y[c] = mean over t of x[t, c]' }[mode]], [about0], code);
        const LY = W2({ t: [0, 0, 1, 0.14], in: [0.08, 0.22, 0.52, 0.76], out: [0.68, 0.4, 0.32, 0.4] }, { t: [0, 0, 1, 0.14], in: [0.14, 0.2, 0.86, 0.5], out: [0, 0.8, 1, 0.17] }), labels = seqLabels(T);
        return Object.assign(base, {
          summary: sum(`sps${i}`, sO, r.out),
          inner: (ctx, q, S) => {
            const u = S.u(q, 8, 13), ir = S.place(q, LY.in), orr = S.place(q, LY.out);
            S.lines(ctx, [{ last: `y = x[${T - 1}]: the last position`, max: 'y[c] = maxₜ x[t, c]', mean: `y[c] = (1/${T}) Σₜ x[t, c]` }[mode], `${shapeText(sI)} → ${shapeText(sO)}, no parameters${mode === 'mean' ? '' : ' · outlined: what is kept'}`], S.place(q, LY.t), { mono: true, size: u * 0.9, strong: (k2) => k2 === 0 });
            S.caption(ctx, `in: ${T} positions × ${D}`, ir, { size: u * 0.8, minW: 120 });
            const g = drawHeat(ctx, heatImg(cache, `spi${i}`, T, D, (t, p) => xin[t * D + p]), ir, T, D);
            rowLabels(ctx, S, labels, ir, { size: u * 0.78, maxW: Math.max(18, ir.x - q.x - 2) });
            ctx.strokeStyle = theme.c('focus'); ctx.lineWidth = 1.4;
            if (mode === 'last') ctx.strokeRect(g.x - 1, g.y + (T - 1) * g.ch - 1, g.w + 2, g.ch + 2);
            if (mode === 'max' && g.cw >= 2 && g.ch >= 2) for (let cc = 0; cc < D; cc++) { const t = Math.floor(Math.round(r.arg[cc]) / D); ctx.strokeRect(g.x + cc * g.cw, g.y + t * g.ch, g.cw, g.ch); }
            S.caption(ctx, 'out: zoom into any number', orr, { size: u * 0.8, minW: 140 }); S.bars(ctx, r.out, orr);
          },
          grids: [{ id: 'x', rect: LY.in, rows: T, cols: D, make: (t, p) => valueNode(i - 1, t * D + p) }, { id: 'y', rect: LY.out, rows: 1, cols: D, make: (a, cc) => seqPoolValue(i, cc) }],
          about: [about0, 'Zoom into a number of the result to see where it came from.'],
          code,
        });
      }

      if (b.type === 'patches') {
        const p = c.size, E = c.dim, C0 = sI ? sI[0] : 1, H0 = sI ? sI[1] : '?', W0 = sI ? sI[2] : '?', N = sO ? sO[0] : '?', J = C0 * p * p;
        base.sub = `${p}×${p} patches → ${E} numbers each`;
        const about0 = `Cuts the picture into ${N} patches of ${p} × ${p} pixels, left to right and top to bottom, and turns each into a vector of ${E} numbers with one learned projection (the same for every patch), plus a learned vector for the patch's place. The picture becomes a sequence of ${N} "tokens" that attention and transformer blocks can work on: this is how a vision transformer (ViT) starts.${sI && (H0 % p || W0 % p) ? ` Rows and columns past the last whole patch are left out.` : ''}`;
        const code = `patch = ${mod}                                  # nn.Conv2d(${C0}, ${E}, ${p}, stride=${p}): one projection per patch
x = torch.randn${sI ? pyShape(sI) : '(1, 1, 14, 14)'}
e = patch(x).flatten(2).transpose(1, 2)            # (1, ${N}, ${E}): one row per patch
out = e + model.pos${i + 1}                                # plus each patch's learned place`;
        if (!have) return plain([`out[n] = W · patch[n] + b + pos[n]   (${N} patches)`], [about0], code);
        const { nw } = lay, w = lay.Wp.w, pos = lay.pos.w, img = C0 === 1, ink = i === 0;
        const LY = W2({ t: [0, 0, 1, 0.14], img: [0, 0.24, 0.27, 0.72], P: [0.31, 0.24, 0.16, 0.72], W: [0.51, 0.24, 0.15, 0.72], pos: [0.7, 0.24, 0.13, 0.72], out: [0.87, 0.24, 0.13, 0.72] },
          { t: [0, 0, 1, 0.13], img: [0, 0.18, 0.48, 0.3], P: [0.54, 0.18, 0.46, 0.3], W: [0, 0.54, 0.46, 0.18], pos: [0.54, 0.54, 0.46, 0.18], out: [0, 0.78, 1, 0.21] });
        const vin = img ? null : tensorView(`pti${i}`, sI, xin, LY.img, { id: 'x', caption: `in: ${shapeText(sI)}`, make: (q2) => valueNode(i - 1, q2), name: `Block ${i + 1} input`, tone: base.tone, tiles: 16 });
        const srcs = Array.from({ length: N }, (_, n) => patchSrc(lay, n));
        return Object.assign(base, {
          summary: sum(`pts${i}`, sO, r.out),
          inner: (ctx, q, S) => {
            const u = S.u(q, 8, 13), at = (key) => S.place(q, LY[key]);
            S.lines(ctx, [`out[n] = W · patch[n] + b + pos[n]:  ${shapeText(sI)} → ${N} patches × ${E}`, `${fmtInt(pnum)} parameters: the projection (${E} × ${J}), its bias and the ${N} × ${E} places`], at('t'), { mono: true, size: u * 0.88, strong: (k2) => k2 === 0 });
            if (img) {
              const ir = at('img');
              S.caption(ctx, `the picture, cut into ${p} × ${p} patches`, ir, { size: u * 0.8, minW: 140 });
              const g = drawHeat(ctx, heatImg(cache, `pti${i}`, H0, W0, (y, x2) => xin[y * W0 + x2], ink ? { ink: true } : {}), ir, H0, W0, true);
              ctx.strokeStyle = theme.c('s4', 0.85); ctx.lineWidth = 1;
              for (let y = p; y < H0; y += p) { ctx.beginPath(); ctx.moveTo(g.x, g.y + y * g.ch); ctx.lineTo(g.x + Math.floor(W0 / p) * p * g.cw, g.y + y * g.ch); ctx.stroke(); }
              for (let x2 = p; x2 < W0; x2 += p) { ctx.beginPath(); ctx.moveTo(g.x + x2 * g.cw, g.y); ctx.lineTo(g.x + x2 * g.cw, g.y + Math.floor(H0 / p) * p * g.ch); ctx.stroke(); }
            } else vin.draw(ctx, q, S, u);
            const pr = at('P'), wr = at('W'), sr = at('pos'), orr = at('out');
            S.caption(ctx, `patches: ${N} × ${J}`, pr, { size: u * 0.8, minW: 80 }); drawHeat(ctx, heatImg(cache, `ptp${i}`, N, J, (n, j) => xin[srcs[n][j]], ink ? { ink: true } : {}), pr, N, J);
            S.caption(ctx, `W: ${E} × ${J}`, wr, { size: u * 0.8, minW: 60 }); drawHeat(ctx, heatImg(cache, `ptw${i}`, E, J, (e, j) => w[e * J + j]), wr, E, J);
            S.caption(ctx, `places: ${N} × ${E}`, sr, { size: u * 0.8, minW: 60 }); drawHeat(ctx, heatImg(cache, `ptq${i}`, N, E, (n, e) => pos[n * E + e]), sr, N, E);
            S.caption(ctx, `out: ${N} × ${E}, open one`, orr, { size: u * 0.8, minW: 80 }); drawHeat(ctx, heatImg(cache, `pto${i}`, N, E, (n, e) => r.out[n * E + e]), orr, N, E);
          },
          grids: [
            img ? { id: 'x', rect: LY.img, rows: H0, cols: W0, square: true, make: (y, x2) => valueNode(i - 1, y * W0 + x2) } : vin.grid,
            { id: 'P', rect: LY.P, rows: N, cols: J, make: (n, j) => Object.assign(valueNode(i - 1, srcs[n][j]), { sub: `pixel ${j} of patch ${n + 1}` }) },
            { id: 'W', rect: LY.W, rows: E, cols: J, make: (e, j) => Dp.stored(X, w[e * J + j], `W[${e}, ${j}], a projection weight`) },
            { id: 'pos', rect: LY.pos, rows: N, cols: E, make: (n, e) => Dp.stored(X, pos[n * E + e], `pos[${n}, ${e}]: patch ${n + 1}'s place (learned)`) },
            { id: 'out', rect: LY.out, rows: N, cols: E, make: (n, e) => patchValue(i, n, e) },
          ],
          get about() { return [about0, `Each row of the patch matrix is one patch's ${J} pixels in a line; the projection turns each row into ${E} numbers, and the place of that patch is added. Zoom into an output number for its sum.`]; },
          code,
        });
      }

      if (b.type === 'batchnorm') {
        const im = !!sI && sI.length === 3, ch = sI ? (im ? sI[0] : sI[sI.length - 1]) : '?', what = im ? 'channel' : 'feature';
        base.sub = im ? `${ch} channels` : sI && sI.length === 2 ? `${ch} features at every position` : `${ch} features`;
        const about0 = `Batch normalisation: while training, each ${what} is shifted and scaled to mean 0 and spread 1 with the statistics of the current batch${im ? ' (over all its pixels too)' : sI && sI.length === 2 ? ' (over all the positions too)' : ''}, then given a learned scale γ and shift β. It also keeps running averages of those means and variances, which replace the batch's when the model is used, as on this example: then it is a fixed y = (x − μ) / √(σ² + ε) · γ + β for each ${what}.`;
        const about1 = 'Unlike LayerNorm, which normalises each example on its own, BatchNorm compares an example with the rest of its batch: it smooths training (deep image networks relied on it for years) but behaves differently in training and in use, and small batches make its statistics noisy.';
        const tr = sI && sI.length === 2;
        const code = `norm = ${mod}                                  # nn.BatchNorm${im ? '2d' : '1d'}(${ch})
norm.eval()                                        # in use: the running averages, not the batch's
x = torch.randn${sI ? pyShape(sI) : '(1, 16)'}
y = ${tr ? 'norm(x.transpose(1, 2)).transpose(1, 2)' : 'norm(x)'}${tr ? '   # features must come second' : ''}
mu, var = norm.running_mean, norm.running_var
print(torch.allclose(y, (x - mu${im ? '[:, None, None]' : ''}) / torch.sqrt(var${im ? '[:, None, None]' : ''} + norm.eps) * norm.weight${im ? '[:, None, None]' : ''} + norm.bias${im ? '[:, None, None]' : ''}, atol=1e-5))`;
        if (!have) return plain(['y = (x − μ) / √(σ² + ε) · γ + β, per ' + what], [about0, about1], code);
        const C = lay.ch, sd = Float32Array.from(lay.rvar, (v) => Math.sqrt(v));
        const LY = W2({ t: [0, 0, 1, 0.14], in: [0, 0.24, 0.44, 0.33], out: [0, 0.66, 0.44, 0.32], mu: [0.52, 0.24, 0.48, 0.13], sd: [0.52, 0.44, 0.48, 0.13], g: [0.52, 0.64, 0.48, 0.13], bb: [0.52, 0.84, 0.48, 0.13] },
          { t: [0, 0, 1, 0.12], in: [0, 0.17, 1, 0.2], out: [0, 0.42, 1, 0.2], mu: [0, 0.68, 0.47, 0.12], sd: [0.53, 0.68, 0.47, 0.12], g: [0, 0.87, 0.47, 0.12], bb: [0.53, 0.87, 0.47, 0.12] });
        const vin = tensorView(`bni${i}`, sI, xin, LY.in, { id: 'x', caption: `in: ${shapeText(sI)}`, make: (q2) => valueNode(i - 1, q2), name: `Block ${i + 1} input`, tone: base.tone, tiles: 16 });
        const vout = tensorView(`bno${i}`, sO, r.out, LY.out, { id: 'y', caption: 'out: zoom into any number', make: (q2) => bnValue(i, q2), name: blockTitle(i), tone: base.tone, tiles: 16 });
        const stats = [['mu', 'running mean μ', lay.rmean], ['sd', 'running spread σ', sd], ['g', 'scale γ (learned)', lay.g.w], ['bb', 'shift β (learned)', lay.b.w]];
        return Object.assign(base, {
          summary: sum(`bns${i}`, sO, r.out),
          inner: (ctx, q, S) => {
            const u = S.u(q, 8, 13);
            S.lines(ctx, [`y = (x − μ) / √(σ² + ε) · γ + β   per ${what}, ${C} of them`, `${fmtInt(pnum)} learned numbers (γ, β); μ and σ² are running averages from training`], S.place(q, LY.t), { mono: true, size: u * 0.88, strong: (k2) => k2 === 0 });
            vin.draw(ctx, q, S, u); vout.draw(ctx, q, S, u);
            for (const [key, cap, vals] of stats) { const sr = S.place(q, LY[key]); S.caption(ctx, `${cap}, per ${what}`, sr, { size: u * 0.78, minW: 120 }); S.bars(ctx, Float32Array.from(vals), sr); }
          },
          grids: [vin.grid, vout.grid, ...stats.map(([key, cap, vals]) => ({ id: key, rect: LY[key], rows: 1, cols: C, make: (a, cc) => Dp.stored(X, vals[cc], `${cap} of ${what} ${im ? cc + 1 : cc}`) }))],
          get about() { return [about0, about1, 'The bars on the right are the numbers it keeps for each ' + what + '. Zoom into an output for the four steps of arithmetic behind it.']; },
          code,
        });
      }

      if (b.type === 'avgpool') {
        base.sub = `${c.size}×${c.size} windows`;
        const code = `x = torch.randn${sI ? pyShape(sI) : '(1, 8, 14, 14)'}
y = F.avg_pool2d(x, ${c.size})                         # ${sO ? pyShape(sO) : ''}: the mean of each ${c.size} × ${c.size} window
print(torch.allclose(y[0, 0, 0, 0], x[0, 0, :${c.size}, :${c.size}].mean()))   # True`;
        const about0 = `Average pooling replaces each ${c.size} × ${c.size} window by its mean, channel by channel. The picture shrinks ${c.size} times in each direction and the next filters see a wider area. Max pooling keeps only the strongest response; the average keeps a smoother summary in which every pixel counts. No weights to learn.`;
        if (!have) return plain([`y[c, i, j] = mean of a ${c.size} × ${c.size} window`], [about0], code);
        const LY = W2({ t: [0, 0, 1, 0.12], in: [0, 0.22, 0.45, 0.76], out: [0.55, 0.22, 0.45, 0.76] }, { t: [0, 0, 1, 0.13], in: [0, 0.2, 1, 0.37], out: [0, 0.62, 1, 0.37] });
        const vin = tensorView(`api${i}`, sI, xin, LY.in, { id: 'x', caption: `in: ${shapeText(sI)}`, make: (q2) => valueNode(i - 1, q2), name: `Block ${i + 1} input`, tone: base.tone, tiles: 16 });
        const vout = tensorView(`apo${i}`, sO, r.out, LY.out, { id: 'y', caption: `out: ${shapeText(sO)}, open a channel`, make: (q2) => avgPoolValue(i, q2), name: blockTitle(i), tone: base.tone, tiles: 16 });
        return Object.assign(base, {
          summary: sum(`aps${i}`, sO, r.out),
          inner: (ctx, q, S) => {
            const u = S.u(q, 8, 13);
            S.lines(ctx, [`y = the mean of each ${c.size} × ${c.size} window`, `${shapeText(sI)} → ${shapeText(sO)}, no parameters`], S.place(q, LY.t), { mono: true, size: u * 0.9, strong: (k2) => k2 === 0 });
            vin.draw(ctx, q, S, u); vout.draw(ctx, q, S, u);
          },
          grids: [vin.grid, vout.grid],
          about: [about0, 'Open an output channel, then a cell, for the average behind it.'],
          code,
        });
      }

      if (b.type === 'noise') {
        const sg = c.sigma;
        base.sub = `σ = ${fmt(sg, 2)} while training`;
        const about0 = `Gaussian noise: while training, every number gets a fresh random nudge, x + σ·ε, with ε drawn from a standard normal distribution and σ = ${fmt(sg, 2)}. The network has to learn answers that survive small changes, which keeps it from memorising the training examples (a regulariser, like dropout). When the model is used it does nothing.`;
        const code = `noise = ${mod}                                 # GaussianNoise(${NA.fmtCode(sg, 3)}), defined in the script
x = torch.ones(1, 6)
noise.train(); print(noise(x))                     # 1 plus a random nudge each
noise.eval();  print(noise(x))                     # when the model is used: unchanged`;
        if (!have) return plain(['training: y = x + σ·ε (ε ~ N(0, 1));  in use: y = x'], [about0], code);
        const LY = W2({ t: [0, 0, 1, 0.16], in: [0, 0.3, 1, 0.27], tr: [0, 0.7, 1, 0.27] }, { t: [0, 0, 1, 0.16], in: [0, 0.27, 1, 0.28], tr: [0, 0.68, 1, 0.3] });
        const rng = new NA.RNG(613 * (i + 1) + 7), tr = Float32Array.from(xin, (v) => v + sg * rng.gauss()), both = Math.max(maxAbs(xin), maxAbs(tr));
        const vin = tensorView(`nzi${i}`, sI, xin, LY.in, { id: 'y', max: both, caption: 'when the model is used, as here: y = x, unchanged', make: (q2) => Object.assign(valueNode(i - 1, q2), { sub: 'unchanged by noise when the model is used' }), name: `Block ${i + 1}`, tone: base.tone, tiles: 16 });
        const vtr = tensorView(`nzt${i}`, sI, tr, LY.tr, { id: 't', max: both, caption: `one training step: every number nudged by σ·ε (σ = ${fmt(sg, 2)})`, make: (q2) => Dp.stored(X, tr[q2], `${fmt(xin[q2], 3)} + noise ${fmt(tr[q2] - xin[q2], 3)}`), name: `Block ${i + 1} (one training step)`, tone: base.tone, tiles: 16 });
        return Object.assign(base, {
          summary: sum(`nzs${i}`, sO, r.out),
          inner: (ctx, q, S) => {
            const u = S.u(q, 8, 13);
            S.lines(ctx, [`training: y = x + ${fmt(sg, 2)}·ε, ε ~ N(0, 1);   used: y = x`, `${fmtInt(xin.length)} numbers; fresh noise every training step`], S.place(q, LY.t), { mono: true, size: u * 0.9, strong: (k2) => k2 === 0 });
            vin.draw(ctx, q, S, u); vtr.draw(ctx, q, S, u);
          },
          grids: [vin.grid, vtr.grid],
          about: [about0, 'The lower picture is one made-up training step, to show what the noise does; on this example, as when the model is used, nothing changes.'],
          code,
        });
      }

      if (b.type === 'fourier') {
        const m = c.features, Fi = sI ? sI[0] : '?';
        base.sub = `${m} waves · scale ${fmt(c.scale, 2)}`;
        const about0 = `Fourier features turn the input x into [x, sin(2π·Bx), cos(2π·Bx)], with ${m} random frequency vectors (the rows of B, drawn once with spread ${fmt(c.scale, 2)} and never trained). Waves of many frequencies give the next layers ready-made wiggles: a plain network learns sharp, high-frequency detail only slowly (its "spectral bias"), and with these it fits it quickly. A larger scale means wigglier waves; too large and the network fits the noise.`;
        const code = `B = model.B${i + 1}                                   # (${m}, ${Fi}): random frequencies, a buffer (not trained)
x = torch.randn(1, ${Fi})
a = 2 * math.pi * x @ B.T
y = torch.cat([x, torch.sin(a), torch.cos(a)], -1)   # (1, ${sO ? sO[0] : '?'})`;
        if (!have) return plain([`y = [x, sin(2π·Bx), cos(2π·Bx)]   (${m} waves)`], [about0], code);
        const freq = lay.freq, LY = W2({ t: [0, 0, 1, 0.14], x: [0, 0.25, 0.24, 0.2], B: [0, 0.55, 0.24, 0.42], waves: [0.3, 0.25, 0.3, 0.72], y: [0.66, 0.25, 0.34, 0.72] }, { t: [0, 0, 1, 0.12], x: [0, 0.17, 0.46, 0.12], B: [0.54, 0.17, 0.46, 0.2], waves: [0, 0.43, 1, 0.26], y: [0, 0.76, 1, 0.22] });
        const nW = Math.min(4, m), TAU = 2 * Math.PI;
        return Object.assign(base, {
          summary: sum(`ffs${i}`, sO, r.out),
          inner: (ctx, q, S) => {
            const u = S.u(q, 8, 13), at = (key) => S.place(q, LY[key]), xr = at('x'), br = at('B'), wr = at('waves'), yr = at('y');
            S.lines(ctx, [`y = [x, sin(2π·Bx), cos(2π·Bx)]:  ${Fi} → ${sO[0]} numbers`, `B: ${m} × ${Fi} random frequencies, fixed · no parameters to learn`], at('t'), { mono: true, size: u * 0.88, strong: (k2) => k2 === 0 });
            S.caption(ctx, `x: ${Fi}`, xr, { size: u * 0.8, minW: 40 }); S.bars(ctx, xin, xr);
            S.caption(ctx, `B: ${m} × ${Fi}`, br, { size: u * 0.8, minW: 60 }); drawHeat(ctx, heatImg(cache, `ffb${i}`, m, Fi, (a, f) => freq[a * Fi + f]), br, m, Fi);
            if (Fi === 1) {
              S.caption(ctx, `the first ${nW} waves over x (dot: this example)`, wr, { size: u * 0.8, minW: 160 }); S.frame(ctx, wr);
              for (let j = 0; j < nW; j++) {
                const hj = wr.h / nW, rr = { x: wr.x + 4, y: wr.y + j * hj + 3, w: wr.w - 8, h: hj - 6 }, f = freq[j];
                S.curve(ctx, (x) => Math.sin(TAU * f * x), rr, -1, 1, -1.1, 1.1, { color: theme.c('s1'), lw: 1.4 });
                const px = rr.x + ((xin[0] + 1) / 2) * rr.w, py = rr.y + rr.h - ((Math.sin(TAU * f * xin[0]) + 1.1) / 2.2) * rr.h;
                ctx.fillStyle = theme.c('ink'); ctx.beginPath(); ctx.arc(px, py, 2.6, 0, Math.PI * 2); ctx.fill();
              }
            } else if (Fi === 2) {
              S.caption(ctx, `the first ${nW} waves (circle: this example)`, wr, { size: u * 0.8, minW: 160 });
              const G = 28, cols2 = Math.min(2, nW), rows2 = Math.ceil(nW / cols2), s2 = Math.min(wr.w / cols2, wr.h / rows2) - 6;
              for (let j = 0; j < nW; j++) {
                const ox = wr.x + (j % cols2) * (s2 + 6), oy = wr.y + Math.floor(j / cols2) * (s2 + 6), fa = freq[j * 2], fb = freq[j * 2 + 1];
                drawHeat(ctx, heatImg(cache, `ffw${i}:${j}`, G, G, (a, b2) => Math.sin(TAU * (fa * (-1 + (2 * b2 + 1) / G) + fb * (1 - (2 * a + 1) / G))), { max: 1 }), { x: ox, y: oy, w: s2, h: s2 }, G, G, true);
                ctx.strokeStyle = theme.c('focus'); ctx.lineWidth = 1.8; ctx.beginPath(); ctx.arc(ox + ((xin[0] + 1) / 2) * s2, oy + ((1 - xin[1]) / 2) * s2, 4, 0, Math.PI * 2); ctx.stroke();
              }
            } else S.lines(ctx, [`${Fi} inputs: each wave runs along a random direction in ${Fi} dimensions`], wr, { size: u * 0.85, colorOf: () => theme.c('muted') });
            S.caption(ctx, 'out: x, then the sines, then the cosines; zoom into one', yr, { size: u * 0.8, minW: 160 }); S.bars(ctx, r.out, yr);
          },
          grids: [
            { id: 'x', rect: LY.x, rows: 1, cols: Fi, make: (a, f) => valueNode(i - 1, f) },
            { id: 'B', rect: LY.B, rows: m, cols: Fi, make: (a, f) => Dp.stored(X, freq[a * Fi + f], `B[${a}, ${f}], a random frequency (fixed)`) },
            { id: 'y', rect: LY.y, rows: 1, cols: sO[0], make: (a, kk) => fourierValue(i, kk) },
          ],
          get about() { return [about0, 'The pictures show what a few of the waves look like across the inputs. Zoom into an output for its weighted sum and the sine of it.']; },
          code,
        });
      }

      if (b.type === 'gated') {
        const U = c.units, g = ACT(c.gate || 'swish'), seqG = !!sI && sI.length === 2, Fi = sI ? sI[sI.length - 1] : '?';
        base.sub = `${U} units · ${g.id === 'sigmoid' ? 'GLU' : g.id === 'swish' ? 'SwiGLU' : g.id === 'gelu' ? 'GEGLU' : g.label + ' gate'}`;
        const about0 = `A gated layer computes two weighted sums of its input for each of its ${U} units, a value v = W₁·x + b₁ and a gate z₂ = W₂·x + b₂, and multiplies them: out = v · ${g.label}(z₂). The gate opens, shuts or flips each value depending on the input, so the layer can switch features on and off. This is the GLU family (with Swish it is SwiGLU, the feed-forward layer of most large language models today).${seqG ? ` On a sequence the same unit runs at each of the ${sI[0]} positions.` : ''}`;
        const code = `glu = ${mod}                                   # Gated(${Fi}, ${U}): two nn.Linear, value and gate
x = torch.randn${sI ? pyShape(sI) : '(1, 8)'}
v, z2 = glu.value(x), glu.gate(x)
out = v * ${PYACT[g.id] ? PYACT[g.id]('z2') : 'z2'}                             # ${sO ? pyShape(sO) : ''}
print(torch.allclose(out, glu(x)))                 # True`;
        if (!have) return plain([`out = (W₁·x + b₁) · ${g.label}(W₂·x + b₂)`], [about0], code);
        const F = lay.nIn, W1 = lay.W1.w, W2w = lay.W2.w, T = lay.rows;
        if (!seqG) {
          const LY = W2({ t: [0, 0, 1, 0.15], x: [0.1, 0.24, 0.42, 0.1], W1: [0.1, 0.39, 0.2, 0.59], W2: [0.32, 0.39, 0.2, 0.59], v: [0.6, 0.39, 0.12, 0.59], a: [0.74, 0.39, 0.12, 0.59], o: [0.88, 0.39, 0.12, 0.59] },
            { t: [0, 0, 1, 0.15], x: [0.12, 0.22, 0.88, 0.08], W1: [0.12, 0.35, 0.42, 0.63], W2: [0.58, 0.35, 0.42, 0.63], v: [0, 0, 0, 0], a: [0, 0, 0, 0], o: [0, 0, 0, 0] });
          const cols = wide ? [['v', 'v = W₁x + b₁', r.z1], ['a', `${g.short || g.label}(z₂)`, r.a2], ['o', 'out = v·gate', r.out]] : [];
          if (!wide) { LY.W1 = [0.12, 0.35, 0.27, 0.6]; LY.W2 = [0.42, 0.35, 0.27, 0.6]; LY.v = [0.72, 0.35, 0.08, 0.6]; LY.a = [0.82, 0.35, 0.08, 0.6]; LY.o = [0.92, 0.35, 0.08, 0.6]; cols.push(['v', 'v', r.z1], ['a', 'g', r.a2], ['o', 'out', r.out]); }
          const m2 = Math.max(maxAbs(r.z1), maxAbs(r.a2), maxAbs(r.out));
          return Object.assign(base, {
            summary: sum(`gls${i}`, sO, r.out),
            inner: (ctx, q, S) => {
              const u = S.u(q, 8, 13), at = (key) => S.place(q, LY[key]);
              S.lines(ctx, [`out = (W₁·x + b₁) · ${g.label}(W₂·x + b₂)`, `${F} inputs → ${U} units · ${fmtInt(pnum)} parameters (two weight matrices)`], at('t'), { mono: true, size: u * 0.9, strong: (k2) => k2 === 0 });
              const xr = at('x'); S.caption(ctx, `input x: ${F}`, xr, { size: u * 0.8, minW: 100 }); S.bars(ctx, xin, xr);
              const w1 = at('W1'), w2 = at('W2');
              S.caption(ctx, `W₁ (value): ${U} × ${F}`, w1, { size: u * 0.78, minW: 60 }); drawHeat(ctx, heatImg(cache, `glw1${i}`, U, F, (a, p) => W1[a * F + p]), w1, U, F);
              S.caption(ctx, `W₂ (gate): ${U} × ${F}`, w2, { size: u * 0.78, minW: 60 }); drawHeat(ctx, heatImg(cache, `glw2${i}`, U, F, (a, p) => W2w[a * F + p]), w2, U, F);
              for (const [key, cap, vals] of cols) { const cr = at(key); S.caption(ctx, cap, cr, { size: u * 0.78, minW: 20 }); hbars(ctx, vals, cr, { max: m2 }); }
            },
            grids: [
              { id: 'x', rect: LY.x, rows: 1, cols: F, make: (a, p) => valueNode(i - 1, p) },
              { id: 'W1', rect: LY.W1, rows: U, cols: F, make: (j, p) => Dp.stored(X, W1[j * F + p], `W₁[${j}, ${p}]: unit ${j + 1}'s value weight`) },
              { id: 'W2', rect: LY.W2, rows: U, cols: F, make: (j, p) => Dp.stored(X, W2w[j * F + p], `W₂[${j}, ${p}]: unit ${j + 1}'s gate weight`) },
              { id: 'v', rect: LY.v, rows: U, cols: 1, make: (j) => gatedLinear(i, 0, j, 1) },
              { id: 'a', rect: LY.a, rows: U, cols: 1, make: (j) => gatedLinear(i, 0, j, 2) },
              { id: 'o', rect: LY.o, rows: U, cols: 1, make: (j) => gatedValue(i, j) },
            ],
            get about() { return [about0, 'The three columns line up with the rows of the weights: the value, the gate and their product for each unit. Zoom into a row of any of them.']; },
            code,
          });
        }
        const LY = W2({ t: [0, 0, 1, 0.14], x: [0.08, 0.24, 0.26, 0.72], W: [0.39, 0.24, 0.26, 0.72], o: [0.73, 0.24, 0.27, 0.72] }, { t: [0, 0, 1, 0.14], x: [0.14, 0.2, 0.86, 0.22], W: [0.14, 0.48, 0.86, 0.2], o: [0.14, 0.75, 0.86, 0.23] }), labels = seqLabels(T);
        return Object.assign(base, {
          summary: sum(`gls${i}`, sO, r.out),
          inner: (ctx, q, S) => {
            const u = S.u(q, 8, 13), at = (key) => S.place(q, LY[key]), xr = at('x'), wr = at('W'), orr = at('o');
            S.lines(ctx, [`out = (W₁·x + b₁) · ${g.label}(W₂·x + b₂) at each of the ${T} positions`, `${F} → ${U} numbers per position · ${fmtInt(pnum)} parameters`], at('t'), { mono: true, size: u * 0.9, strong: (k2) => k2 === 0 });
            S.caption(ctx, `input: ${T} × ${F}`, xr, { size: u * 0.8, minW: 120 }); drawHeat(ctx, heatImg(cache, `glx${i}`, T, F, (t, p) => xin[t * F + p]), xr, T, F);
            rowLabels(ctx, S, labels, xr, { size: u * 0.78, maxW: Math.max(18, xr.x - q.x - 2) });
            S.caption(ctx, `W₁ above W₂: ${U} × ${F} each`, wr, { size: u * 0.8, minW: 120 }); drawHeat(ctx, heatImg(cache, `glw${i}`, 2 * U, F, (a, p) => (a < U ? W1[a * F + p] : W2w[(a - U) * F + p])), wr, 2 * U, F);
            S.caption(ctx, `out: ${T} × ${U}, zoom into one`, orr, { size: u * 0.8, minW: 120 }); drawHeat(ctx, heatImg(cache, `glo${i}`, T, U, (t, j) => r.out[t * U + j]), orr, T, U);
          },
          grids: [
            { id: 'x', rect: LY.x, rows: T, cols: F, make: (t, p) => valueNode(i - 1, t * F + p) },
            { id: 'W', rect: LY.W, rows: 2 * U, cols: F, make: (a, p) => Dp.stored(X, a < U ? W1[a * F + p] : W2w[(a - U) * F + p], a < U ? `W₁[${a}, ${p}] (value)` : `W₂[${a - U}, ${p}] (gate)`) },
            { id: 'o', rect: LY.o, rows: T, cols: U, make: (t, j) => gatedValue(i, t * U + j) },
          ],
          get about() { return [about0, 'Zoom into an output for its value, its gate and their product.']; },
          code,
        });
      }

      if (b.type === 'position') {
        const learned = c.mode !== 'sine', T = sI ? sI[0] : '?', D = sI ? sI[1] : '?';
        base.sub = learned ? 'learned, one row per position' : 'sine waves (fixed)';
        const about0 = learned ? `Adds a learned vector to each of the ${T} positions, so that the layers after can tell positions apart. Attention and pooling treat their input as an unordered set; without positions "not good" and "good not" would look the same.` : `Adds the original transformer's fixed position pattern: PE[t, 2i] = sin(t / 10000^(2i/D)) and PE[t, 2i + 1] = cos(…), waves of many speeds, a different pattern at every position. Nothing is learned, and nearby positions get similar patterns.`;
        const code = learned ? `x = torch.randn${sI ? pyShape(sI) : '(1, 16, 32)'}\nout = x + model.pos${i + 1}                               # (${T}, ${D}) learned, added at every example` : `t = torch.arange(${T})[:, None]\ni = torch.arange(${D})[None, :]\nangle = t / 10000 ** (2 * (i // 2) / ${D})\nPE = torch.where(i % 2 == 0, torch.sin(angle), torch.cos(angle))   # (${T}, ${D})\nout = torch.randn(1, ${T}, ${D}) + PE`;
        if (!have) return plain([`out = x + ${learned ? 'P' : 'PE'}[position]`], [about0], code);
        const Ptab = learned ? lay.pos.w : lay.table, labels = seqLabels(T);
        const LY = W2({ t: [0, 0, 1, 0.14], x: [0.07, 0.24, 0.27, 0.72], P: [0.39, 0.24, 0.27, 0.72], o: [0.71, 0.24, 0.29, 0.72] }, { t: [0, 0, 1, 0.14], x: [0.14, 0.19, 0.86, 0.24], P: [0.14, 0.47, 0.86, 0.24], o: [0.14, 0.75, 0.86, 0.24] });
        return Object.assign(base, {
          summary: sum(`pss${i}`, sO, r.out),
          inner: (ctx, q, S) => {
            const u = S.u(q, 8, 13), at = (key) => S.place(q, LY[key]), xr = at('x'), pr = at('P'), orr = at('o');
            S.lines(ctx, [`out[t] = x[t] + ${learned ? 'P' : 'PE'}[t]   for each of the ${T} positions`, learned ? `${fmtInt(pnum)} learned numbers` : 'a fixed pattern, no parameters'], at('t'), { mono: true, size: u * 0.9, strong: (k2) => k2 === 0 });
            S.caption(ctx, `x: ${T} × ${D}`, xr, { size: u * 0.8, minW: 80 }); drawHeat(ctx, heatImg(cache, `psx${i}`, T, D, (t, p) => xin[t * D + p]), xr, T, D);
            rowLabels(ctx, S, labels, xr, { size: u * 0.78, maxW: Math.max(18, xr.x - q.x - 2) });
            S.caption(ctx, `${learned ? 'P (learned)' : 'PE (sine waves)'}: ${T} × ${D}`, pr, { size: u * 0.8, minW: 80 }); drawHeat(ctx, heatImg(cache, `psp${i}`, T, D, (t, p) => Ptab[t * D + p]), pr, T, D);
            S.caption(ctx, 'out: zoom into one', orr, { size: u * 0.8, minW: 80 }); drawHeat(ctx, heatImg(cache, `pso${i}`, T, D, (t, p) => r.out[t * D + p]), orr, T, D);
          },
          grids: [
            { id: 'x', rect: LY.x, rows: T, cols: D, make: (t, p) => valueNode(i - 1, t * D + p) },
            { id: 'P', rect: LY.P, rows: T, cols: D, make: (t, p) => Dp.stored(X, Ptab[t * D + p], `${learned ? 'P' : 'PE'}[${t}, ${p}]`) },
            { id: 'o', rect: LY.o, rows: T, cols: D, make: (t, p) => positionValue(i, t, p) },
          ],
          get about() { return [about0, 'Each row of the table is added to the same row of the input. Zoom into an output for the addition.']; },
          code,
        });
      }
      return null;
    }

    // ---- classic models: the whole model is one block, so its level shows how it reaches its answer for the example
    const featRaw = (f, v) => (d.table && d.stats ? v * d.stats.std[f] + d.stats.mean[f] : v);
    const treePath = (t, x) => { const path = [0]; let n = 0; while (t.feat[n] >= 0) { n = x[t.feat[n]] <= t.thr[n] ? t.left[n] : t.right[n]; path.push(n); } return path; };
    const className = (k) => (names[k] !== undefined ? String(names[k]) : `class ${k + 1}`);
    const ownColours = d.family === 'plane' || !!d.table || K <= 4;
    const classCol = (k, a = 1) => (ownColours ? theme.c(BV ? BV.classColour(d, k) : 's5', a) : theme.c(k === pred ? 's5' : 'muted', k === pred ? a : a * 0.55));
    function leafSays(t, n, o) {
      const v = t.value[n];
      if (o.boost) return `${v[0] >= 0 ? '+' : ''}${fmt(v[0], 3)}`;
      if (model.cls) { const kk = argmax(v); return `${className(kk)} ${fmtPct(v[kk], 0)}`; }
      return fmt(d.table && d.stats && d.stats.y ? v[0] * d.stats.y.std + d.stats.y.mean : v[0], 3);
    }
    function questionNode(t, n) {
      const f = t.feat[n], v = P.x[f], thr = t.thr[n], le = v <= thr, nm = inputLabel(f);
      const gainWhat = model.type === 'boost' ? 'XGBoost\'s gain ½[G_L²/(H_L + λ) + G_R²/(H_R + λ) − G²/(H + λ)] − γ, from the gradients G and curvatures H of the loss at the training rows' : model.cls ? 'the drop in Gini impurity (how mixed the classes below are)' : 'the drop in squared error';
      return Dp.lazy({
        title: `${nm} ≤ ${fmt(featRaw(f, thr), 3)}?`, sub: `${fmt(featRaw(f, v), 3)}: ${le ? 'yes, left' : 'no, right'}`, tone: 's5',
        inner: (ctx, r, S) => S.lines(ctx, [`is ${nm} ≤ ${fmt(featRaw(f, thr), 4)}?`, `this example: ${nm} = ${fmt(featRaw(f, v), 4)}, so ${le ? 'yes: go left' : 'no: go right'}`, `${t.n[n]} training rows reached this question: ${t.n[t.left[n]]} went left, ${t.n[t.right[n]]} right`, `chosen for the largest gain: ${fmt(t.gain[n], 4)}`], S.place(r, [0, 0, 1, 0.32]), { mono: true, size: S.u(r, 8, 13), strong: (k) => k === 1 }),
        about: [`A question about one input. To pick it, the tree tried every input${model.type === 'forest' ? ' of a random few' : ''} and up to 47 thresholds for each (the quantiles of its training values) and kept the one that best separated the answers below it: ${gainWhat}.`, `${d.table ? 'The table\'s columns are shown in their own units; the model compares them scaled to mean 0 and spread 1, which keeps the order, so the answer is the same. ' : ''}Zoom into the input it reads, or into the comparison: inside the chip a comparison is a subtraction whose sign bit is read.`],
        code: `x = Xva[0].flatten()\nprint(x[${f}] <= ${NA.fmtCode(thr, 5)})   # ${le ? 'True: left' : 'False: right'}`,
      }, () => [
        Object.assign(valueNode(-1, f), { id: 'x', rect: W2([0, 0.42, 0.48, 0.55], [0, 0.38, 1, 0.29]), title: `${nm} = ${fmt(featRaw(f, v), 3)}` }),
        Object.assign(Dp.fadd(X, v, -thr), { id: 'cmp', rect: W2([0.52, 0.42, 0.48, 0.55], [0, 0.7, 1, 0.29]), title: 'The comparison', sub: `${fmt(v, 3)} − ${fmt(thr, 3)}: its sign decides` }),
      ]);
    }
    function leafNode(t, n, o) {
      const v = t.value[n], head = `Leaf: ${leafSays(t, n, o)}`;
      if (o.boost) return Object.assign(Dp.stored(X, v[0], 'the leaf value −G/(H + λ) × η'), { title: head, sub: `${t.n[n]} training rows`, about: [`This leaf adds ${fmt(v[0], 4)} to the score. Its value is −G/(H + λ) × η: G and H are the sums of the loss's gradients and curvatures over the ${t.n[n]} training rows that land here, so the leaf moves their scores in the direction that lowers the loss, by a step shrunk by λ and the learning rate η.`] });
      if (!model.cls) return Object.assign(Dp.stored(X, v[0], 'the leaf value: the mean of its training rows'), { title: head, sub: `${t.n[n]} training rows`, about: [`A leaf answers with the mean target of the ${t.n[n]} training rows that ended up in it.`] });
      const sh = Float32Array.from(v), br = W2([0, 0.3, 1, 0.5], [0, 0.3, 1, 0.45]);
      return {
        title: head, sub: `${t.n[n]} training rows`, tone: 's5',
        summary: (ctx, r, S) => S.bars(ctx, sh, r, { positive: true, max: 1, color: (k) => classCol(k) }),
        inner: (ctx, r, S) => { const u = S.u(r, 8, 13), q = S.place(r, br); S.lines(ctx, [`the shares of the classes among the ${t.n[n]} training rows here`, `it says ${leafSays(t, n, o)}`], S.place(r, [0, 0, 1, 0.2]), { size: u * 0.92, strong: (k) => k === 1 }); S.bars(ctx, sh, q, { positive: true, max: 1, color: (k) => classCol(k) }); colLabels(ctx, S, Array.from(sh, (s2, k) => `${className(k)} ${fmtPct(s2, 0)}`), q, { size: u * 0.78 }); },
        grids: [{ id: 'p', rect: br, rows: 1, cols: sh.length, make: (a, k) => Dp.stored(X, sh[k], `the share of ${className(k)} in this leaf`) }],
        about: [`A leaf answers with the classes' shares among the training rows that ended up in it: the tree's probabilities.`],
      };
    }
    // one tree with this example's path through it; the nodes on the path open the question asked there, and the leaf
    function treeNode(t, o) {
      const x = P.x, path = treePath(t, x), on = new Set(path), leaf = path[path.length - 1], LIM = 3;
      const shown = (n) => t.depth[n] <= LIM || on.has(n), pos = new Map();
      let slots = 0, deep = 0, maxLeaf = 1e-9;
      for (let n = 0; n < t.size; n++) if (t.feat[n] < 0) maxLeaf = Math.max(maxLeaf, Math.abs(t.value[n][0]));
      const walk = (n) => {
        deep = Math.max(deep, t.depth[n]);
        if (t.feat[n] >= 0) { const kids = [t.left[n], t.right[n]].filter(shown); if (kids.length) { const xs = kids.map(walk); pos.set(n, xs.reduce((a, b2) => a + b2, 0) / xs.length); return pos.get(n); } }
        pos.set(n, slots++); return pos.get(n);
      };
      walk(0);
      const TR = W2([0, 0.2, 0.66, 0.78], [0, 0.17, 1, 0.5]), PL = W2([0.69, 0.2, 0.31, 0.78], [0, 0.71, 1, 0.28]);
      const nh = Math.min(0.075, (TR[3] / (deep + 1)) * 0.6), bw = Math.min((TR[2] / Math.max(1, slots)) * 0.94, 0.17);
      const rectOf = (n) => [TR[0] + ((pos.get(n) + 0.5) / Math.max(1, slots)) * TR[2] - bw / 2, TR[1] + (deep ? t.depth[n] / deep : 0) * (TR[3] - nh), bw, nh];
      const colOf = (n, a) => (!o.boost && model.cls ? classCol(argmax(t.value[n]), a) : theme.divColor(clamp(t.value[n][0] / maxLeaf, -1, 1), a));
      const steps = path.slice(0, -1).map((n, k) => { const f = t.feat[n], v = x[f], le = v <= t.thr[n]; return `${k + 1}. ${inputLabel(f)} = ${fmt(featRaw(f, v), 2)} ${le ? '≤' : '>'} ${fmt(featRaw(f, t.thr[n]), 2)} → ${le ? 'left' : 'right'}`; });
      const pathLines = ['this example\'s way down:', ...steps, `leaf: ${leafSays(t, leaf, o)} (${t.n[leaf]} training rows)`];
      return Dp.lazy({
        title: o.title, sub: `${t.size} nodes · depth ${deep} · says ${leafSays(t, leaf, o)}`, tone: 's5',
        summary: (ctx, r) => {
          // the tree's shape in miniature, the path in the focus colour
          for (const [n] of pos) for (const kid of [t.left[n], t.right[n]]) if (kid >= 0 && pos.has(kid)) {
            const a = rectOf(n), b2 = rectOf(kid), hot = on.has(n) && on.has(kid);
            ctx.strokeStyle = hot ? theme.c('focus') : theme.c('line-2'); ctx.lineWidth = hot ? 1.6 : 0.8; ctx.beginPath();
            ctx.moveTo(r.x + (a[0] + a[2] / 2 - TR[0]) / TR[2] * r.w, r.y + (a[1] - TR[1]) / TR[3] * r.h); ctx.lineTo(r.x + (b2[0] + b2[2] / 2 - TR[0]) / TR[2] * r.w, r.y + (b2[1] - TR[1]) / TR[3] * r.h); ctx.stroke();
          }
        },
        inner: (ctx, r, S) => {
          const u = S.u(r, 8, 13), place = (f) => S.place(r, f);
          S.lines(ctx, [o.head, slots < t.leafCount() ? `${t.size} nodes, ${t.leafCount()} leaves: shown are the first ${LIM} levels and this example's path` : `${t.size} nodes, ${t.leafCount()} leaves`, 'outlined: the questions on this example\'s path, zoom into one'], place([0, 0, 1, 0.16]), { size: u * 0.88, strong: (k) => k === 0 });
          for (const [n] of pos) for (const kid of [t.left[n], t.right[n]]) if (kid >= 0 && pos.has(kid)) {
            const a = place(rectOf(n)), b2 = place(rectOf(kid)), hot = on.has(n) && on.has(kid);
            ctx.strokeStyle = hot ? theme.c('focus') : theme.c('line-2'); ctx.lineWidth = hot ? 2.2 : 1;
            ctx.beginPath(); ctx.moveTo(a.x + a.w / 2, a.y + a.h); ctx.lineTo(b2.x + b2.w / 2, b2.y); ctx.stroke();
          }
          for (const [n] of pos) {
            const q = place(rectOf(n)), hot = on.has(n), isLeaf = t.feat[n] < 0, cut = !isLeaf && ![t.left[n], t.right[n]].some((kk) => pos.has(kk));
            S.box(ctx, q, { fill: isLeaf ? colOf(n, hot ? 0.55 : 0.3) : hot ? theme.c('focus', 0.12) : theme.c('panel-2'), stroke: hot ? theme.c('focus') : theme.c('line-2'), lw: hot ? 1.8 : 1, radius: Math.min(5, q.h * 0.3) });
            const label = isLeaf ? leafSays(t, n, o) : cut ? '…' : `${inputLabel(t.feat[n])} ≤ ${fmt(featRaw(t.feat[n], t.thr[n]), 2)}`, ts = Math.min(u * 0.8, q.h * 0.56);
            ctx.font = S.font(hot ? 650 : 500, ts, true);
            const fs = Math.max(Math.min(5.5, ts), Math.min(ts, (ts * (q.w - 5)) / Math.max(1, ctx.measureText(label).width)));
            if (fs >= 5.5) S.text(ctx, label, q.x + q.w / 2, q.y + q.h / 2, { size: fs, mono: true, align: 'center', maxW: q.w - 2, color: hot ? theme.c('ink') : theme.c('muted'), w: hot ? 650 : 500 });
          }
          S.lines(ctx, pathLines, place(PL), { mono: true, size: u * 0.8, strong: (k) => k === pathLines.length - 1, colorOf: (k) => (k === 0 ? theme.c('muted') : theme.c('ink')) });
        },
        about: [o.about || 'A decision tree asks one yes-or-no question about one input at each node and follows the answer down, until a leaf gives the answer. The path this example takes is outlined.', 'Each question was picked while growing the tree, as the split of the training rows that helped most; a tree stops at its maximum depth, or where no question helps. Zoom into a question on the path for the comparison and the input it reads, or into the leaf.'],
        code: o.code || '',
      }, () => path.map((n, k) => Object.assign(k < path.length - 1 ? questionNode(t, n) : leafNode(t, n, o), { id: 'q' + k, shape: 'zone', rect: rectOf(n) })));
    }
    // the example's distance to each of its nearest neighbours: a sum of squared differences
    function knnDist(j, rank, dist) {
      const F = model.F, x = P.x, terms = Array.from({ length: F }, (_, f) => { const v = x[f] - d.Xtr[j * F + f]; return { a: v, w: v }; });
      return Object.assign(Dp.neuron(X, {
        title: `Neighbour ${rank + 1}: training example ${j + 1}`, sub: `distance ${fmt(dist, 3)}`, tone: 's5',
        terms, bias: null, act: null, zName: 'dist2', outName: 'dist2', actLine: `distance = √dist2 = ${fmt(dist, 4)}`,
        rowNames: [`differences: this example minus the neighbour, input by input (${F})`, 'the same differences again', 'their squares, which add up to the squared distance'], inLabel: (f) => `${inputLabel(f)}: difference`, wLabel: (f) => `${inputLabel(f)}: difference`,
        vars: { w: 'diff', a: 'diff', b: '' }, codeHead: [`x, n = Xva[0].flatten(), Xtr[${j}].flatten()   # this example and training example ${j + 1}`, 'diff = x - n'],
      }), { about: [`The distance from this example to training example ${j + 1}: subtract one from the other input by input, square the differences and add them up (then a square root). It is a dot product of the difference with itself, so the same circuits compute it as compute a neuron.`, 'k-NN does this for every training example and keeps the k smallest: the model is the training data, and answering is the slow part.'] });
    }
    function classicLevel(i, base, prob) {
      const b = blocks[i], s = cfgOf(b), kind = b.type, mdl = nums && model && model.classic ? model : null;
      const subs = { linear: out.kind === 'classes' ? `logistic regression${s.lambda ? ` · λ ${fmt(s.lambda, 3)}` : ''}` : s.lambda ? `ridge, λ = ${fmt(s.lambda, 3)}` : 'least squares', knn: `k = ${s.k}${s.weights === 'distance' ? ' · by distance' : ''}`, tree: `depth ≤ ${s.depth}`, forest: `${s.trees} trees · depth ≤ ${s.depth}`, boost: `${s.rounds} rounds · η ${fmt(s.eta, 2)} · depth ${s.depth}` };
      base.sub = subs[kind];
      const cls = out.kind === 'classes';
      const about0 = {
        linear: cls ? `Logistic regression: one weighted sum of the inputs per class, z = W·x + b, turned into probabilities by a softmax. Its boundaries between classes are straight lines (flat planes), however long it trains. It learns by gradient steps on the log loss${s.lambda ? `, with a penalty λ·|W|² that keeps the weights small` : ''}.` : `Linear regression: the answer is a weighted sum of the inputs, ŷ = w·x + b, the straight line (or flat plane) through the data with the least squared error. It has an exact answer, found in one solve${s.lambda ? `, with the ridge penalty λ = ${fmt(s.lambda, 3)} pulling the weights toward 0` : ''}.`,
        knn: `k-nearest neighbours has no weights at all: it keeps every training example and, to answer, finds the ${s.k} closest to the new one (by straight-line distance over all the inputs) and lets them ${cls ? 'vote' : 'average their values'}${s.weights === 'distance' ? ', the closer ones counting more' : ''}. Simple and often strong, but it needs a sensible distance (inputs on the same scale) and every answer searches the whole training set.`,
        tree: `A decision tree asks a yes-or-no question about one input at each step (is x₂ ≤ 0.3?) and follows the answers down to a leaf, which gives the answer. It is grown one level at a time: at each leaf, every input and threshold is tried and the question that best separates the training answers is kept. Deep trees fit anything, including the noise.`,
        forest: `A random forest grows ${s.trees} deep trees, each on a bootstrap sample (the training rows drawn at random with repeats) and choosing among a random few inputs at every question, then averages their answers. Each tree overfits in its own way; the average cancels much of it. Few settings, hard to get badly wrong.`,
        boost: `Gradient boosting (as in XGBoost) adds small trees one at a time, each fitted to what the trees so far still get wrong: the gradient and curvature of the loss at every training row. Each tree's answer is shrunk by the learning rate η = ${fmt(s.eta, 2)} before it is added. For tables of numbers it is usually the model to beat.`,
      }[kind];
      const code = NA.builderPy && NA.builderPy.classicSnippet ? NA.builderPy.classicSnippet(b, d) : '';
      if (!mdl || prob) return Object.assign(base, { inner: plainLevel([{ linear: cls ? 'p = softmax(W·x + b)' : 'ŷ = w·x + b', knn: `the ${s.k} nearest training examples ${cls ? 'vote' : 'average'}`, tree: 'questions down to a leaf', forest: 'the average of the trees', boost: 'score = base + η·(tree₁ + tree₂ + …)' }[kind]], prob), about: [about0], code });
      const m = mdl, x = P.x, F = m.F, K = m.K;
      if (kind === 'linear') {
        const P1 = F + 1, Wt = m.W, img = d.input.kind === 'image', zs = Float32Array.from({ length: K }, (_, k) => { let z = Wt[k * P1 + F]; for (let f = 0; f < F; f++) z += Wt[k * P1 + f] * x[f]; return z; });
        const neuronK = (k) => Dp.neuron(X, {
          title: cls ? `The score for ${className(k)}` : 'The answer ŷ', sub: `${F} inputs, no activation`, tone: 's5',
          terms: Array.from({ length: F }, (_, f) => ({ a: x[f], w: Wt[k * P1 + f] })), bias: Wt[k * P1 + F], act: null, zName: cls ? 'z' : 'y_hat', outName: cls ? 'z' : 'y_hat',
          inLabel: (f) => inputLabel(f), wLabel: (f) => `w[${cls ? k + ', ' : ''}${f}]`, inputNode: (f) => valueNode(-1, f), inWhat: 'the input',
          vars: { w: 'w', a: 'x', b: 'b' }, codeHead: ['x = Xva[0].flatten().numpy()                      # the first validation example', `w, b = ${cls ? `W[${k}], B[${k}]` : 'W[0], B[0]'}                                # its weights and bias, as on the page`],
          about: [cls ? `The score for ${className(k)}: a weighted sum of all ${F} inputs with this class's own weights, plus its bias. The softmax of all ${K} scores gives the probabilities.` : `The model's answer: a weighted sum of all ${F} inputs plus a bias. Each weight says how much the answer goes up when that input goes up by 1.`],
        });
        const LY = img ? W2({ t: [0, 0, 1, 0.16], tiles: [0, 0.24, 0.6, 0.74], z: [0.68, 0.3, 0.32, 0.5] }, { t: [0, 0, 1, 0.14], tiles: [0, 0.2, 1, 0.5], z: [0, 0.78, 1, 0.18] })
          : W2({ t: [0, 0, 1, 0.16], x: [0, 0.27, 0.58, 0.16], W: [0, 0.52, 0.58, 0.44], z: [0.66, 0.27, 0.34, 0.6] }, { t: [0, 0, 1, 0.16], x: [0, 0.24, 1, 0.12], W: [0, 0.44, 1, 0.24], z: [0, 0.76, 1, 0.2] });
        const tl = img ? tileLay(K, K, LY.tiles[2] * size.w, LY.tiles[3] * size.h) : null, H0 = img ? d.input.h : 0, W0 = img ? d.input.w : 0;
        const contrib = !cls ? Float32Array.from({ length: F }, (_, f) => Wt[f] * x[f]) : null;
        return Object.assign(base, {
          summary: (ctx, q, S) => S.bars(ctx, zs, q),
          inner: (ctx, q, S) => {
            const u = S.u(q, 8, 13), at = (key) => S.place(q, LY[key]);
            S.lines(ctx, [cls ? `z = W·x + b for each of the ${K} classes;  p = softmax(z)` : `ŷ = w·x + b = ${fmt(zs[0], 4)}`, `${fmtInt(m.paramCount)} learned numbers · ${m.rounds ? `${cls ? `${fmtInt(m.rounds)} gradient steps` : 'solved exactly'}` : 'not trained yet (all weights 0)'}`, img ? 'each tile: one class\'s weights over the picture (blue: ink here raises its score)' : cls ? 'W: one row of weights per class' : 'the bars: how much each input adds to the answer, w × x'], at('t'), { mono: true, size: u * 0.88, strong: (k) => k === 0 });
            if (img) {
              const tr = at('tiles'), cs = Math.min(tr.w / tl.cols, tr.h / tl.rows), mm = maxAbs(Wt);
              for (let k = 0; k < K; k++) {
                const ox = tr.x + (k % tl.cols) * cs, oy = tr.y + Math.floor(k / tl.cols) * cs;
                drawHeat(ctx, heatImg(cache, `lwt${k}`, H0, W0, (a, b2) => Wt[k * P1 + a * W0 + b2], { max: mm }), { x: ox + 3, y: oy + 3, w: cs - 6, h: cs - 6 }, H0, W0, true);
                S.text(ctx, className(k), ox + cs / 2, oy + cs - 2, { size: Math.min(u * 0.75, cs * 0.14), align: 'center', color: k === pred ? theme.c('ink') : theme.c('muted'), w: k === pred ? 700 : 500, maxW: cs });
              }
            } else {
              const xr = at('x'), wr = at('W');
              S.caption(ctx, `x: the ${F} inputs`, xr, { size: u * 0.8, minW: 100 }); S.bars(ctx, x, xr);
              if (F <= 12) colLabels(ctx, S, Array.from({ length: F }, (_, f) => inputLabel(f)), xr, { size: u * 0.74 });
              if (cls) { S.caption(ctx, `W: ${K} classes × ${F}`, wr, { size: u * 0.8, minW: 100 }); drawHeat(ctx, heatImg(cache, 'lw', K, F, (a, f) => Wt[a * P1 + f]), wr, K, F); }
              else { S.caption(ctx, 'w × x: each input\'s part of the answer', wr, { size: u * 0.8, minW: 140 }); S.bars(ctx, contrib, wr); if (F <= 12) colLabels(ctx, S, Array.from(contrib, (v) => fmt(v, 2)), wr, { size: u * 0.74 }); }
            }
            const zr = at('z');
            S.caption(ctx, cls ? 'scores z: zoom into one' : 'the answer: zoom in for the sum', zr, { size: u * 0.8, minW: 120 }); S.bars(ctx, zs, zr, cls && ownColours ? { color: (k) => classCol(k, k === pred ? 1 : 0.5) } : {});
            if (cls) colLabels(ctx, S, Array.from({ length: K }, (_, k) => className(k)), zr, { size: u * 0.74, hi: pred });
          },
          grids: [
            ...(img ? [{ id: 'w', rect: LY.tiles, rows: tl.rows, cols: tl.cols, square: true, skip: (a, b2) => a * tl.cols + b2 >= K, make: (a, b2) => neuronK(a * tl.cols + b2) }]
              : [{ id: 'x', rect: LY.x, rows: 1, cols: F, make: (a, f) => valueNode(-1, f) }, cls ? { id: 'W', rect: LY.W, rows: K, cols: F, make: (k, f) => Dp.stored(X, Wt[k * P1 + f], `W[${k}, ${f}]: ${className(k)}'s weight on ${inputLabel(f)}`) } : { id: 'c', rect: LY.W, rows: 1, cols: F, make: (a, f) => Object.assign(Dp.fmul(X, Wt[f], x[f]), { title: `${inputLabel(f)}'s part: w × x` }) }]),
            { id: 'z', rect: LY.z, rows: 1, cols: K, make: (a, k) => neuronK(k) },
          ],
          get about() { return [about0, img ? 'Each tile is one class\'s weights laid out as the picture: where it is blue, ink raises that class\'s score; where orange, ink lowers it. A linear model can only learn one such template per class, which is why it reads digits worse than a network.' : cls ? 'Each row of W is one class\'s weights. Zoom into a score for its weighted sum.' : 'The bars show each input\'s share of the answer for this example, weight times value. Zoom into one, or into the answer for the whole sum.']; },
          code,
        });
      }
      if (kind === 'knn') {
        const nb = P.nb ? Array.from(P.nb).filter((j) => j >= 0) : [], kk = nb.length;
        const dist = nb.map((j) => { let q = 0; for (let f = 0; f < F; f++) { const v = x[f] - d.Xtr[j * F + f]; q += v * v; } return Math.sqrt(q); });
        const wts = dist.map((q) => (s.weights === 'distance' ? 1 / (q + 1e-6) : 1)), wsum = wts.reduce((a, b2) => a + b2, 0) || 1;
        const lab = (j) => (cls ? className(d.Ytr[j]) : fmt(d.Ytr[j * K], 3));
        const vote = new Float32Array(cls ? K : kk);
        if (cls) nb.forEach((j, q) => { vote[d.Ytr[j]] += wts[q] / wsum; }); else nb.forEach((j, q) => { vote[q] = d.Ytr[j * K]; });
        const img = d.input.kind === 'image', plane = F === 2 && !img;
        const LY = W2({ t: [0, 0, 1, 0.15], vis: [0, 0.22, 0.48, 0.76], nb: [0.54, 0.25, 0.46, 0.3], vote: [0.54, 0.66, 0.46, 0.28] }, { t: [0, 0, 1, 0.14], vis: [0, 0.18, 1, 0.4], nb: [0, 0.62, 1, 0.15], vote: [0, 0.83, 1, 0.15] });
        return Object.assign(base, {
          summary: (ctx, q, S) => S.bars(ctx, Float32Array.from(dist), q, { positive: true }),
          inner: (ctx, q, S) => {
            const u = S.u(q, 8, 13), at = (key) => S.place(q, LY[key]), vr = at('vis'), nr = at('nb'), wr = at('vote');
            S.lines(ctx, [`the ${kk} training examples nearest to this one ${cls ? 'vote' : 'average their values'}${s.weights === 'distance' ? ' (closer ones count more)' : ''}`, `nearest: example ${nb[0] + 1} at distance ${fmt(dist[0], 3)}, the ${kk}th at ${fmt(dist[kk - 1], 3)}`, `searched: all ${fmtInt(m.n)} training examples`], at('t'), { size: u * 0.88, strong: (k2) => k2 === 0 });
            if (plane) {
              const sq = Math.min(vr.w, vr.h), bx = { x: vr.x + (vr.w - sq) / 2, y: vr.y, w: sq, h: sq };
              let x0 = -1, x1 = 1, y0 = -1, y1 = 1;
              if (d.table) { x0 = Infinity; x1 = -Infinity; y0 = Infinity; y1 = -Infinity; for (let k2 = 0; k2 < m.n; k2++) { x0 = Math.min(x0, d.Xtr[k2 * 2]); x1 = Math.max(x1, d.Xtr[k2 * 2]); y0 = Math.min(y0, d.Xtr[k2 * 2 + 1]); y1 = Math.max(y1, d.Xtr[k2 * 2 + 1]); } const px2 = (x1 - x0) * 0.08, py2 = (y1 - y0) * 0.08; x0 -= px2; x1 += px2; y0 -= py2; y1 += py2; }
              const X0 = (v) => bx.x + ((v - x0) / (x1 - x0)) * bx.w, Y0 = (v) => bx.y + bx.h - ((v - y0) / (y1 - y0)) * bx.h, ym = cls ? 1 : maxAbs(d.Ytr);
              ctx.fillStyle = theme.c('panel-2'); ctx.fillRect(bx.x, bx.y, bx.w, bx.h); S.frame(ctx, bx);
              for (let k2 = 0; k2 < m.n; k2++) { ctx.fillStyle = cls ? classCol(d.Ytr[k2], 0.4) : theme.divColor(clamp(d.Ytr[k2] / ym, -1, 1), 0.55); ctx.beginPath(); ctx.arc(X0(d.Xtr[k2 * 2]), Y0(d.Xtr[k2 * 2 + 1]), Math.max(1.3, sq / 170), 0, Math.PI * 2); ctx.fill(); }
              const cx = X0(x[0]), cy = Y0(x[1]);
              ctx.strokeStyle = theme.c('ink', 0.7); ctx.lineWidth = 1;
              for (const j of nb) { ctx.beginPath(); ctx.moveTo(cx, cy); ctx.lineTo(X0(d.Xtr[j * 2]), Y0(d.Xtr[j * 2 + 1])); ctx.stroke(); }
              ctx.setLineDash([3, 3]); ctx.strokeStyle = theme.c('focus'); ctx.beginPath(); ctx.ellipse(cx, cy, Math.abs(X0(x[0] + dist[kk - 1]) - cx), Math.abs(Y0(x[1] + dist[kk - 1]) - cy), 0, 0, Math.PI * 2); ctx.stroke(); ctx.setLineDash([]);
              ctx.strokeStyle = theme.c('focus'); ctx.lineWidth = 2.2; ctx.beginPath(); ctx.arc(cx, cy, Math.max(5, sq / 40), 0, Math.PI * 2); ctx.stroke();
            } else if (img) {
              const { h: H0, w: W0 } = d.input, sq = Math.min(vr.w, vr.h);
              S.caption(ctx, 'this example', vr, { size: u * 0.8, minW: 100 });
              drawHeat(ctx, heatImg(cache, 'knnx', H0, W0, (a, b2) => x[a * W0 + b2], { ink: true }), { x: vr.x, y: vr.y, w: sq, h: sq }, H0, W0, true);
            } else S.lines(ctx, [`${F} inputs: the distance runs over all of them at once`, 'zoom into a neighbour for its distance, input by input'], vr, { size: u * 0.88, colorOf: () => theme.c('muted') });
            S.caption(ctx, `the ${kk} neighbours, nearest first: zoom into one`, nr, { size: u * 0.8, minW: 160 });
            if (img) {
              const { h: H0, w: W0 } = d.input, cs = Math.min(nr.w / kk, nr.h);
              nb.forEach((j, q2) => { const ox = nr.x + q2 * cs; drawHeat(ctx, heatImg(cache, `knn${j}`, H0, W0, (a, b2) => d.Xtr[j * F + a * W0 + b2], { ink: true }), { x: ox + 2, y: nr.y, w: cs - 4, h: cs - 4 }, H0, W0, true); });
              colLabels(ctx, S, nb.map((j) => lab(j)), { x: nr.x, y: nr.y, w: cs * kk, h: cs - 4 }, { size: u * 0.74 });
            } else {
              S.bars(ctx, Float32Array.from(dist), nr, { positive: true, color: (q2) => (cls ? classCol(d.Ytr[nb[q2]]) : theme.c('s5')) });
              colLabels(ctx, S, nb.map((j) => lab(j)), nr, { size: u * 0.74 });
            }
            S.caption(ctx, cls ? `the vote: ${className(argmax(vote))} wins` : `their values, averaged: ${fmt(P.logits[0], 3)}`, wr, { size: u * 0.8, minW: 140 });
            S.bars(ctx, vote, wr, cls ? { positive: true, max: 1, color: (k2) => classCol(k2) } : {});
            if (cls) colLabels(ctx, S, Array.from(vote, (v, k2) => (K <= 6 ? `${className(k2)} ${fmtPct(v, 0)}` : className(k2))), wr, { size: u * 0.72, hi: argmax(vote) });
          },
          grids: [
            { id: 'nb', rect: LY.nb, rows: 1, cols: Math.max(1, kk), make: (a, q2) => (q2 < kk ? knnDist(nb[q2], q2, dist[q2]) : null) },
            ...(plane || img ? [] : [{ id: 'x', rect: LY.vis, rows: 1, cols: F, make: (a, f) => valueNode(-1, f) }]),
          ],
          get about() { return [about0, `For this example: the ${kk} nearest training examples, their ${cls ? 'classes' : 'values'}, and the ${cls ? 'vote' : 'average'} that makes the answer. Zoom into a neighbour for its distance, computed input by input.`]; },
          code,
        });
      }
      if (kind === 'tree') {
        if (!m.trees.length || !m.trees[0].size) return Object.assign(base, { inner: plainLevel(['No tree yet: press Train to grow it, one level per step.'], null), about: [about0], code });
        return Object.assign(treeNode(m.trees[0], { title: base.title, head: `grown ${fmtInt(m.rounds)} level${m.rounds === 1 ? '' : 's'} deep · ${fmtInt(m.paramCount)} learned numbers`, about: about0, code }), { id: base.id, tone: base.tone });
      }
      if (kind === 'forest') {
        const nt = m.trees.length;
        if (!nt) return Object.assign(base, { inner: plainLevel(['No trees yet: press Train to grow the forest, one tree per step.'], null), about: [about0], code });
        const leaves = m.trees.map((t) => t.value[t.leafOf(x, 0)]), mean = new Float32Array(K);
        leaves.forEach((v) => { for (let k2 = 0; k2 < K; k2++) mean[k2] += v[k2] / nt; });
        const ans = cls ? argmax(mean) : 0, per = Float32Array.from(leaves, (v) => (cls ? v[ans] : v[0])), yv = (v) => (d.table && d.stats && d.stats.y ? v * d.stats.y.std + d.stats.y.mean : v);
        const LY = W2({ t: [0, 0, 1, 0.16], trees: [0, 0.27, 1, 0.34], avg: [0, 0.72, cls ? 0.5 : 1, 0.25] }, { t: [0, 0, 1, 0.16], trees: [0, 0.25, 1, 0.34], avg: [0, 0.72, 1, 0.25] });
        const feats = s.features === 'all' ? F : s.features === 'third' ? Math.max(1, Math.round(F / 3)) : Math.max(1, Math.round(Math.sqrt(F)));
        return Object.assign(base, {
          summary: (ctx, q, S) => S.bars(ctx, per, q, cls ? { positive: true, max: 1, color: (t2) => classCol(argmax(leaves[t2])) } : {}),
          inner: (ctx, q, S) => {
            const u = S.u(q, 8, 13), tr = S.place(q, LY.trees), ar = S.place(q, LY.avg);
            S.lines(ctx, [cls ? `${nt} trees, their class shares averaged: ${className(ans)} with ${fmtPct(mean[ans], 0)}` : `${nt} trees, their answers averaged: ${fmt(yv(mean[0]), 3)}`, `each tree: a bootstrap sample of the rows, ${feats} of ${F} inputs tried at each question, depth ≤ ${s.depth}`, `${fmtInt(m.paramCount)} learned numbers`], S.place(q, LY.t), { size: u * 0.88, strong: (k2) => k2 === 0 });
            S.caption(ctx, cls ? `each tree's share for ${className(ans)} (colour: the class it favours); zoom into one` : 'each tree\'s answer for this example; zoom into one', tr, { size: u * 0.8, minW: 200 });
            S.bars(ctx, per, tr, cls ? { positive: true, max: 1, color: (t2) => classCol(argmax(leaves[t2])) } : {});
            if (cls) { S.caption(ctx, 'the average: the forest\'s answer', ar, { size: u * 0.8, minW: 140 }); S.bars(ctx, mean, ar, { positive: true, max: 1, color: (k2) => classCol(k2) }); colLabels(ctx, S, Array.from(mean, (v, k2) => `${className(k2)} ${fmtPct(v, 0)}`), ar, { size: u * 0.74, hi: ans }); }
            else { S.lines(ctx, [`the mean of the ${nt} answers: ${fmt(yv(mean[0]), 4)}`, 'trees that disagree average out: that is the point of a forest'], ar, { size: u * 0.88, colorOf: () => theme.c('muted') }); }
          },
          grids: [{ id: 'tree', rect: LY.trees, rows: 1, cols: nt, make: (a, t2) => treeNode(m.trees[t2], { title: `Tree ${t2 + 1} of ${nt}`, head: `tree ${t2 + 1}: grown on its own bootstrap sample, trying ${feats} random inputs at each question`, code }) },
            ...(cls ? [{ id: 'avg', rect: LY.avg, rows: 1, cols: K, make: (a, k2) => Dp.stored(X, mean[k2], `the forest's share for ${className(k2)}: the mean of ${nt} trees`) }] : [])],
          get about() { return [about0, 'Each bar is one tree\'s answer for this example. Zoom into one to see that tree and the path this example takes through it.']; },
          code,
        });
      }
      // gradient boosting
      const T = m.T, nr = m.trees.length / T;
      if (!m.trees.length) return Object.assign(base, { inner: plainLevel([`No rounds yet: the score is the starting value ${fmt(m.base[0], 3)}. Press Train to add trees.`], null), about: [about0], code });
      const contrib = m.trees.map((t) => t.value[t.leafOf(x, 0)][0]), sc = Float64Array.from(m.base);
      contrib.forEach((v, r2) => { sc[r2 % T] += v; });
      const follow = T === 1 ? 0 : argmax(sc), mine = [], idx = [];
      contrib.forEach((v, r2) => { if (r2 % T === follow) { mine.push(v); idx.push(r2); } });
      const run = []; let acc = m.base[follow]; for (const v of mine) { acc += v; run.push(acc); }
      const mineA = Float32Array.from(mine), link = !cls ? `ŷ = score = ${fmt(acc, 4)}` : T === 1 ? `p(${className(1)}) = σ(score) = ${fmtPct(1 / (1 + Math.exp(-acc)), 1)}` : `p = softmax of the ${T} class scores: ${className(follow)} ${fmtPct(probs[follow], 1)}`;
      const LY = W2({ t: [0, 0, 1, 0.18], c: [0, 0.3, 1, 0.3], run: [0, 0.7, 1, 0.27] }, { t: [0, 0, 1, 0.18], c: [0, 0.28, 1, 0.3], run: [0, 0.68, 1, 0.3] });
      return Object.assign(base, {
        summary: (ctx, q, S) => S.bars(ctx, mineA, q),
        inner: (ctx, q, S) => {
          const u = S.u(q, 8, 13), cr = S.place(q, LY.c), rr = S.place(q, LY.run);
          S.lines(ctx, [`score${T > 1 ? ` for ${className(follow)}` : ''} = start + Σ trees = ${fmt(m.base[follow], 3)} + ${fmt(acc - m.base[follow], 3)} = ${fmt(acc, 3)}`, link, `${fmtInt(nr)} round${nr === 1 ? '' : 's'}${T > 1 ? `, ${T} trees each (one per class)` : ''} · η = ${fmt(s.eta, 2)} · depth ${s.depth} · λ ${fmt(s.lambda, 2)}${s.gamma ? ` · γ ${fmt(s.gamma, 2)}` : ''}`], S.place(q, LY.t), { mono: true, size: u * 0.86, strong: (k2) => k2 === 0 });
          S.caption(ctx, `what each round's tree adds${T > 1 ? ` to ${className(follow)}'s score` : ''} for this example: zoom into one`, cr, { size: u * 0.8, minW: 220 }); S.bars(ctx, mineA, cr);
          S.caption(ctx, 'the score, round by round', rr, { size: u * 0.8, minW: 140 }); S.frame(ctx, rr);
          let lo = Math.min(m.base[follow], ...run), hi = Math.max(m.base[follow], ...run); const sp = hi - lo || 1; lo -= sp * 0.1; hi += sp * 0.1;
          const Y0 = (v) => rr.y + rr.h - ((v - lo) / (hi - lo)) * rr.h, X0 = (k2) => rr.x + ((k2 + 0.5) / (run.length + 1)) * rr.w;
          if (lo < 0 && hi > 0) { ctx.strokeStyle = theme.c('line-2'); ctx.lineWidth = 0.8; ctx.beginPath(); ctx.moveTo(rr.x, Y0(0)); ctx.lineTo(rr.x + rr.w, Y0(0)); ctx.stroke(); }
          ctx.strokeStyle = theme.c('s5'); ctx.lineWidth = 1.8; ctx.beginPath(); ctx.moveTo(X0(-1), Y0(m.base[follow])); run.forEach((v, k2) => ctx.lineTo(X0(k2), Y0(v))); ctx.stroke();
          ctx.fillStyle = theme.c('ink'); ctx.beginPath(); ctx.arc(X0(run.length - 1), Y0(acc), 3.2, 0, Math.PI * 2); ctx.fill();
        },
        grids: [{ id: 'round', rect: LY.c, rows: 1, cols: mine.length, make: (a, k2) => treeNode(m.trees[idx[k2]], { title: `Round ${Math.floor(idx[k2] / T) + 1}${T > 1 ? `, ${className(follow)}'s tree` : ''}`, head: `round ${Math.floor(idx[k2] / T) + 1}: a small tree fitted to the gradients left by the rounds before; its leaves are already scaled by η`, boost: true, about: 'Each tree in boosting is small and fitted to the gradients of the loss: its leaves say how much to add to the score of the rows that land in them.', code }) }],
        get about() { return [about0, 'The bars are what each round\'s tree adds for this example, the line the score growing round by round. Early trees make big corrections; later ones fine-tune. Zoom into a bar for that tree and the path this example takes through it.']; },
        code,
      });
    }
    // the answer of a classic model: probabilities or a value, with the loss on this example
    function classicOut(base) {
      const kind = (blocks.find((q) => isClassicB(q.type)) || {}).type, name = ((NA.cml && NA.cml.KINDS[kind]) || {}).label || 'model';
      const lossLine = out.kind === 'value' ? `loss for this example: ½(ŷ − y)² = ${fmt(0.5 * (P.logits[0] - target[0]) ** 2, 4)}` : `loss for this example: −ln p(${className(target[0])}) = ${fmt(-Math.log(Math.max(probs[target[0]], 1e-30)), 4)}`;
      const about0 = out.kind === 'value' ? `The ${name.toLowerCase()}'s answer, a number. Training made it as close as it could to the targets, by squared error.` : `The ${name.toLowerCase()}'s answer: a probability for each class, ${{ linear: 'the softmax of the class scores', knn: 'the shares of the neighbours\' votes', tree: 'the class shares in the leaf the example reached', forest: 'the trees\' class shares, averaged', boost: 'the softmax of the boosted scores (a sigmoid for two classes)' }[kind] || ''}. The answer is the most probable class.`;
      const code = out.kind === 'value' ? 'print(model.predict(Xva[:1]), Yva[:1])            # the answer and the target' : 'p = model.predict_proba(Xva[:1])                   # one probability per class\nprint(p, p.argmax(1), Yva[:1])';
      if (out.kind === 'value') return Object.assign(base, {
        summary: (ctx, r, S) => S.text(ctx, `ŷ = ${fmt(P.logits[0], 3)}`, r.x + r.w / 2, r.y + r.h * 0.4, { size: clamp(r.h * 0.45, 7, 13), mono: true, w: 650, align: 'center', maxW: r.w }),
        inner: (ctx, r, S) => S.lines(ctx, [`ŷ = ${fmt(P.logits[0], 4)}`, `target y = ${fmt(target[0], 4)},  error ŷ − y = ${fmt(P.logits[0] - target[0], 4)}`, lossLine], S.place(r, [0, 0, 1, 0.3]), { mono: true, size: S.u(r, 8, 13), strong: (k) => k === 0 }),
        about: [about0, `On this example the model says ${fmt(P.logits[0], 3)} where the target is ${fmt(target[0], 3)}.`], code,
      });
      const pr = W2([0, 0.36, 1, 0.5], [0, 0.34, 1, 0.5]), pv = Float32Array.from(probs);
      return Object.assign(base, {
        summary: (ctx, r, S) => S.bars(ctx, pv, r, { positive: true, max: 1, color: (k) => classCol(k, k === pred ? 1 : 0.5) }),
        inner: (ctx, r, S) => {
          const u = S.u(r, 8, 13), q = S.place(r, pr);
          S.lines(ctx, [answerText(), `${pred === target[0] ? 'right' : `it is ${className(target[0])}`};  ${lossLine}`], S.place(r, [0, 0, 1, 0.22]), { mono: true, size: u * 0.9, strong: (k) => k === 0 });
          S.caption(ctx, 'probabilities: zoom into one', q, { size: u * 0.8, minW: 140 }); S.bars(ctx, pv, q, { positive: true, max: 1, color: (k) => classCol(k, k === pred ? 1 : 0.5) });
          colLabels(ctx, S, Array.from(pv, (v, k) => `${className(k)} ${fmtPct(v, 0)}`), q, { size: u * 0.76, hi: pred });
        },
        grids: [{ id: 'p', rect: pr, rows: 1, cols: K, make: (a, k) => Dp.stored(X, pv[k], `p(${className(k)})`) }],
        about: [about0, `On this example ${answerText()}; ${lossLine}.`], code,
      });
    }


    // ---- input, embedding, output
    function inputNode() {
      const q = d.input || {}, s = inputShape(), base = { id: 'input', title: 'Input', tone: 's3' };
      const prob = d.error ? { reason: d.error } : null;
      base.sub = d.error ? 'no data' : q.kind === 'vector' ? `${q.dim} number${q.dim === 1 ? '' : 's'}` : q.kind === 'image' ? `${q.h} × ${q.w} picture` : q.kind === 'series' ? `${q.len} values in order` : `${q.len} tokens`;
      const about0 = d.error ? `The data could not be made: ${d.error}` : q.kind === 'vector' ? (d.table ? `The model reads one row of your table: ${q.dim === 1 ? 'the number column' : `the ${q.dim} number columns`} ${d.featNames.join(', ')}, each scaled to mean 0 and spread 1, and has to ${d.kind === 'value' ? `predict ${d.targetName}` : `say which ${d.targetName} the row belongs to`}.` : `The model reads ${q.dim === 1 ? 'one number, x' : `${q.dim} numbers, a point's coordinates`} per example${d.family === 'curve' ? ', and has to predict the curve\'s value there' : d.family === 'surface' ? ', and has to predict the height of the surface there' : ', and has to say which class the point belongs to'}.`)
        : q.kind === 'image' ? `The model reads a picture of ${q.h} × ${q.w} pixels, each a number from 0 (paper) to 1 (ink), ${q.h * q.w} numbers in all.`
          : q.kind === 'series' ? `The model reads the last ${q.len} values of the series, in order, and has to predict the next one. The series is cut into such windows; the first 80% of them train and the rest, later in time, are held back to judge the forecast.`
          : `The model reads ${q.len} tokens, each a whole number standing for one ${vocab.every((w) => w.length <= 1) ? 'symbol' : 'word'} of a vocabulary of ${q.vocab}. The embedding then turns each into a vector of ${q.embed} numbers.${d.family === 'seqclass' ? ` The model has to say which class the whole sequence belongs to: ${names.join(' or ')}.` : d.len ? ` The first ${d.len} digits are the question and → marks where the answer starts; after it come the answer's digits, and at every position the model predicts the next token from the ones before it (it only learns from the answer positions: the question is random).` : d.task === 'language' ? ' This is a short story in the tiny language, padded with full stops to a fixed length; at every position the model predicts the next word.' : d.task === 'sentences' ? ' This is a window of your own text; at every position the model predicts the next word.' : ''}`;
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
      if (q.kind === 'series') {
        const n = q.len, vals = P.x, plot = W2([0.46, 0.06, 0.54, 0.88], [0, 0.24, 1, 0.42]), bars = W2([0, 0.56, 0.4, 0.34], [0, 0.78, 1, 0.19]);
        return Object.assign(base, {
          summary: (ctx, r, S) => S.bars(ctx, vals, r),
          inner: (ctx, r, S) => {
            const u = S.u(r, 8, 13), pr = S.place(r, plot), br = S.place(r, bars);
            S.lines(ctx, [`${whose}:`, exampleText(), nums ? answerText() : '', `line: the window the model reads; circle: the value after it${nums ? '; dot: its forecast' : ''}`].filter(Boolean), S.place(r, W2([0, 0, 0.42, 0.5], [0, 0, 1, 0.22])), { size: u * 0.95, strong: (k) => k === 1 });
            ctx.fillStyle = theme.c('panel-2'); ctx.fillRect(pr.x, pr.y, pr.w, pr.h); S.frame(ctx, pr);
            let lo = Math.min(...vals, target[0], nums ? P.logits[0] : 0), hi = Math.max(...vals, target[0], nums ? P.logits[0] : 0);
            const sp = hi - lo || 1; lo -= sp * 0.12; hi += sp * 0.12;
            const X0 = (t) => pr.x + ((t + 0.5) / (n + 1)) * pr.w, Y0 = (v) => pr.y + pr.h - ((v - lo) / (hi - lo)) * pr.h;
            ctx.strokeStyle = theme.c('s3'); ctx.lineWidth = 2; ctx.beginPath();
            for (let t = 0; t < n; t++) { if (t) ctx.lineTo(X0(t), Y0(vals[t])); else ctx.moveTo(X0(t), Y0(vals[t])); }
            ctx.stroke(); ctx.fillStyle = theme.c('s3');
            for (let t = 0; t < n; t++) { ctx.beginPath(); ctx.arc(X0(t), Y0(vals[t]), 2.2, 0, Math.PI * 2); ctx.fill(); }
            ctx.strokeStyle = theme.c('focus'); ctx.lineWidth = 2.2; ctx.beginPath(); ctx.arc(X0(n), Y0(target[0]), 6, 0, Math.PI * 2); ctx.stroke();
            if (nums) { ctx.fillStyle = theme.c('ink'); ctx.beginPath(); ctx.arc(X0(n), Y0(P.logits[0]), 3.4, 0, Math.PI * 2); ctx.fill(); }
            S.caption(ctx, 'the window, value by value: zoom into one', br, { size: u * 0.8, minW: 150 }); S.bars(ctx, vals, br);
          },
          grids: [{ id: 'x', rect: bars, rows: 1, cols: n, make: (i, j) => Dp.stored(X, vals[j], inputLabel(j)) }],
          about: [about0, `Shown: ${whose}, ${exampleText()}. Every number inside the model is computed from these ${n} values. Zoom into a bar to see a value as the 32 bits stored in memory.`], code,
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
        const sq = Math.min(r.w, r.h), b = { x: r.x + (r.w - sq) / 2, y: r.y, w: sq, h: sq }, curve = d.family === 'curve', surf = d.family === 'surface', ymax = surf ? maxAbs(d.Ytr) : 1, F = q.dim;
        // a table's rows sit on its first two input columns (classes) or the first against the value
        const ax = (k) => d.Xtr[k * F], ay = (k) => (curve ? d.Ytr[k] : F > 1 ? d.Xtr[k * F + 1] : 0);
        const span = (f) => { let lo = Infinity, hi = -Infinity; for (let k = 0; k < d.nTr; k++) { const v = f(k); if (v < lo) lo = v; if (v > hi) hi = v; } const sp = (hi - lo) || 1; return [lo - sp * 0.1, hi + sp * 0.1]; };
        let x0 = -1, x1 = 1, y0 = -1, y1 = 1;
        if (d.table) { [x0, x1] = span(ax); [y0, y1] = span(ay); } else if (curve) [y0, y1] = span(ay);
        const X0 = (v) => b.x + ((v - x0) / (x1 - x0)) * b.w, Y0 = (v) => b.y + b.h - ((v - y0) / (y1 - y0)) * b.h;
        ctx.fillStyle = theme.c('panel-2'); ctx.fillRect(b.x, b.y, b.w, b.h); S.frame(ctx, b);
        for (let k = 0; k < d.nTr; k++) {
          const px = X0(ax(k)), py = Y0(ay(k));
          ctx.fillStyle = curve ? theme.c('s1', 0.5) : surf ? theme.divColor(clamp(d.Ytr[k] / ymax, -1, 1), 0.7) : theme.c(BV ? BV.classColour(d, d.Ytr[k]) : 's1', 0.45); ctx.beginPath(); ctx.arc(px, py, Math.max(1.4, sq / 160), 0, Math.PI * 2); ctx.fill();
        }
        const px = X0(P.x[0]), py = curve ? Y0(target[0]) : Y0(F > 1 ? P.x[1] : 0);
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
      if (model && model.classic && nums) return classicOut(base);
      if (!nums) return Object.assign(base, { inner: plainLevel([out.kind === 'value' ? (isClassicB((blocks[0] || {}).type) ? 'ŷ: the model\'s answer' : 'ŷ = W·x + b') : isClassicB((blocks[0] || {}).type) ? 'p: a probability per class' : 'z = W·x + b;  p = softmax(z)'], prob), about: [about0], code });
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
          `${isClassicB((blocks[0] || {}).type) && nb === 1 ? `The model you chose, as boxes: the input, the ${(NAME[blocks[0].type] || '').toLowerCase()} (a whole model on its own, not a network) and its answer.` : `The model you built, as boxes: the input, ${nb ? `the ${nb} block${nb === 1 ? '' : 's'} you stacked` : 'no hidden blocks (a linear model)'}${seq ? ', the embedding' : ''} and the output head, with the shape of the data on every arrow.`}${ok ? ` It has ${fmtInt(nParams)} learned numbers (parameters).` : ` It does not fit together yet: ${firstProblem ? firstProblem.reason : 'see the red block'}`}`,
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
