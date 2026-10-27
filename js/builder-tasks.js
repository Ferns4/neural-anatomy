/* builder-tasks.js — the problems the model builder can learn, and the data they make for the engine (js/builder-
   engine.js): points in a plane to classify (preset shapes, or ones you draw), a curve to fit (preset, or a formula
   you type), handwritten digits, and sequences (reverse, sort or copy digits, the tiny language, or sentences you
   write). Also the starting templates, a small safe formula parser and the helpers that decode a model's output. */
(function (NA) {
  'use strict';
  const BT = {};

  // ------------------------------------------------------------------ the catalogue
  /* family: what kind of data. Each family fixes the input and output shapes, the losses that make sense and the
     measure shown as "how well it does". */
  BT.FAMILIES = {
    plane: { label: 'Classify points', losses: ['xent', 'mse', 'hinge'], metric: 'accuracy', template: 'mlp', icon: '⁘' },
    curve: { label: 'Fit a curve', losses: ['mse', 'mae', 'huber'], metric: 'r2', template: 'mlp', icon: '∿' },
    digits: { label: 'Read digits', losses: ['xent', 'mse', 'hinge'], metric: 'accuracy', template: 'cnn', icon: '▦' },
    sequence: { label: 'Sequences', losses: ['xent'], metric: 'tokens', template: 'transformer', icon: '⋯' },
  };
  BT.PROBLEMS = [
    { id: 'spiral', family: 'plane', dataset: 'spiral', label: 'Two spirals', hint: 'Two arms wound around each other: needs curved, layered boundaries.' },
    { id: 'moons', family: 'plane', dataset: 'moons', label: 'Two moons', hint: 'Two interlocking crescents.' },
    { id: 'circle', family: 'plane', dataset: 'circle', label: 'Circle', hint: 'One class inside a ring of the other: no straight line separates them.' },
    { id: 'xor', family: 'plane', dataset: 'xor', label: 'XOR', hint: 'Opposite corners match: the classic problem a single neuron cannot solve.' },
    { id: 'pinwheel', family: 'plane', dataset: 'pinwheel', label: 'Pinwheel', hint: 'Three classes in curved arms.' },
    { id: 'blobs', family: 'plane', dataset: 'blobs', label: 'Two blobs', hint: 'Two clouds: a straight line is enough.' },
    { id: 'draw', family: 'plane', custom: true, label: 'Draw your own', hint: 'Click to place points of up to four classes, then train a model on them.' },
    { id: 'sine', family: 'curve', dataset: 'sine', label: 'Sine wave', hint: 'Smooth and periodic.' },
    { id: 'abs', family: 'curve', dataset: 'abs', label: 'V shape', hint: '|x|: exactly two ReLUs.' },
    { id: 'bumps', family: 'curve', dataset: 'bumps', label: 'Bumps', hint: 'Narrow, local features.' },
    { id: 'step', family: 'curve', dataset: 'step', label: 'Step', hint: 'A jump: hard for smooth functions.' },
    { id: 'formula', family: 'curve', custom: true, label: 'Your own function', hint: 'Type any formula in x, for example sin(5x) + x^2.' },
    { id: 'digits', family: 'digits', label: 'Handwritten digits', hint: '14 × 14 pictures of the digits 0 to 9, drawn with random strokes.' },
    { id: 'reverse', family: 'sequence', task: 'reverse', label: 'Reverse a sequence', hint: '3 1 4 → 4 1 3. Each answer digit must look back at a different input position.' },
    { id: 'sort', family: 'sequence', task: 'sort', label: 'Sort digits', hint: '3 1 4 → 1 3 4. Needs to compare all the inputs.' },
    { id: 'copy', family: 'sequence', task: 'copy', label: 'Copy a sequence', hint: '3 1 4 → 3 1 4. The easiest look-back task.' },
    { id: 'language', family: 'sequence', task: 'language', label: 'Tiny language', hint: 'Predict the next word of short stories, like the Tiny Transformer model.' },
    { id: 'sentences', family: 'sequence', task: 'sentences', custom: true, label: 'Your own sentences', hint: 'Type some text; the model learns to predict each next word.' },
  ];
  BT.byId = Object.fromEntries(BT.PROBLEMS.map((p) => [p.id, p]));
  BT.CLASS_COLOURS = ['s1', 's2', 's4', 's7'];

  // problem settings with their defaults (the inspector shows the ones that apply)
  BT.defaults = (p) => {
    const f = p.family;
    if (f === 'plane') return { noise: p.id === 'xor' || p.id === 'blobs' ? 0.1 : 0.15, n: 600 };
    if (f === 'curve') return { noise: 0.05, n: 200, formula: 'sin(5x) + x^2' };
    if (f === 'digits') return { n: 2000 };
    if (p.task === 'language') return { n: 1500, embed: 32 };
    if (p.task === 'sentences') return { window: 8, embed: 32, text: SAMPLE_TEXT };
    return { len: 4, n: 3000, embed: 32 };
  };
  const SAMPLE_TEXT = 'the sun is hot . the snow is cold . the fire is hot . the ice is cold .\nthe cat sat on the mat . the dog sat on the log .\nred and blue make purple . blue and yellow make green . red and yellow make orange .';

  // ------------------------------------------------------------------ templates
  const D = (units, act = 'relu') => ({ type: 'dense', units, act });
  BT.TEMPLATES = [
    { id: 'empty', label: 'Empty', hint: 'Just the embedding and the output: build the rest yourself.', families: ['sequence'], blocks: () => [] },
    { id: 'linear', label: 'Linear', hint: 'No hidden layer: one weighted sum per output, so only straight boundaries. Build up from here.', families: ['plane', 'curve', 'digits'], blocks: () => [] },
    { id: 'mlp', label: 'Small MLP', hint: 'Two hidden layers of 16 ReLU neurons.', families: ['plane', 'curve', 'digits'], blocks: () => [D(16), D(16)] },
    { id: 'deep', label: 'Deep MLP', hint: 'Five hidden layers of 24 neurons.', families: ['plane', 'curve', 'digits'], blocks: () => [D(24), D(24), D(24), D(24), D(24)] },
    { id: 'wide', label: 'One wide layer', hint: 'A single hidden layer of 64 neurons.', families: ['plane', 'curve', 'digits'], blocks: () => [D(64)] },
    { id: 'cnn', label: 'Small CNN', hint: 'Two convolution + pooling stages, then a dense layer.', families: ['digits'], blocks: () => [{ type: 'conv', filters: 8, k: 3, stride: 1, act: 'relu' }, { type: 'pool', size: 2 }, { type: 'conv', filters: 16, k: 3, stride: 1, act: 'relu' }, { type: 'pool', size: 2 }, { type: 'flatten' }, D(32)] },
    { id: 'transformer', label: 'Tiny transformer', hint: 'Two transformer blocks with two attention heads each.', families: ['sequence'], blocks: () => [{ type: 'transformer', heads: 2, mlp: 4, causal: true }, { type: 'transformer', heads: 2, mlp: 4, causal: true }, { type: 'layernorm' }] },
    { id: 'attention', label: 'One attention layer', hint: 'A single attention layer and an MLP, the smallest model that can look back.', families: ['sequence'], blocks: () => [{ type: 'attention', heads: 1, causal: true, residual: true }, { type: 'dense', units: 32, act: 'relu', residual: false }] },
    { id: 'noattn', label: 'No attention', hint: 'An MLP applied to each position alone: it cannot look at other positions, so reversing fails.', families: ['sequence'], blocks: () => [D(64), D(32)] },
  ];
  BT.templatesFor = (family) => BT.TEMPLATES.filter((t) => t.families.includes(family));

  // ------------------------------------------------------------------ making data
  /* Returns { family, input, output, Xtr, Ytr, nTr, Xva, Yva, nVa, classes, names, vocab, ... }: inputs as
     Float32Array (vectors, images) or Int32Array (tokens), targets as Int32Array (classes, tokens; −1 = no
     target) or Float32Array (values). Deterministic from the seed. */
  BT.make = function (p, s, seed = 1, custom = {}) {
    const fam = p.family;
    if (fam === 'plane') return p.custom ? drawnPoints(custom.points || [], seed) : planeData(p, s, seed);
    if (fam === 'curve') return curveData(p, s, seed);
    if (fam === 'digits') return digitData(s, seed);
    if (p.task === 'language') return languageData(s, seed);
    if (p.task === 'sentences') return sentenceData(s, seed);
    return algoData(p.task, s, seed);
  };
  function split(n, rng, frac = 0.8) { const idx = NA.range(n); rng.shuffle(idx); const k = Math.max(1, Math.min(n - 1, Math.round(n * frac))); return [idx.slice(0, k), idx.slice(k)]; }
  function gather(X, dim, idx, Type) { const out = new Type(idx.length * dim); idx.forEach((k, i) => { for (let j = 0; j < dim; j++) out[i * dim + j] = X[k * dim + j]; }); return out; }

  function planeData(p, s, seed) {
    const ds = NA.data.buildDataset({ dataset: p.dataset, n: s.n, noise: s.noise, trainRatio: 0.75, seed, features: ['x1', 'x2'] });
    const val = Array.from(ds.valIdx).concat(Array.from(ds.testIdx)), tr = Array.from(ds.trainIdx), K = Math.max(2, ds.classes || 2);
    return { family: 'plane', input: { kind: 'vector', dim: 2 }, output: { kind: 'classes', k: K }, classes: K, names: NA.range(K).map((k) => `class ${k + 1}`),
      Xtr: gather(ds.raw, 2, tr, Float32Array), Ytr: Int32Array.from(tr.map((k) => ds.y[k])), nTr: tr.length,
      Xva: gather(ds.raw, 2, val, Float32Array), Yva: Int32Array.from(val.map((k) => ds.y[k])), nVa: val.length };
  }
  // points you placed: x, y in [−1, 1] and a class 0..3; classes are renumbered so they are 0..K−1
  function drawnPoints(points, seed) {
    const used = [...new Set(points.map((q) => q.c))].sort(), K = Math.max(2, used.length), map = Object.fromEntries(used.map((c, i) => [c, i]));
    const rng = new NA.RNG(seed * 7 + 3), [tr, va] = points.length >= 4 ? split(points.length, rng, 0.8) : [NA.range(points.length), []];
    const X = new Float32Array(points.length * 2), Y = new Int32Array(points.length);
    points.forEach((q, i) => { X[2 * i] = q.x; X[2 * i + 1] = q.y; Y[i] = map[q.c]; });
    return { family: 'plane', input: { kind: 'vector', dim: 2 }, output: { kind: 'classes', k: K }, classes: K, names: used.map((c) => `class ${c + 1}`), colourOf: used,
      Xtr: gather(X, 2, tr, Float32Array), Ytr: Int32Array.from(tr.map((k) => Y[k])), nTr: tr.length,
      Xva: gather(X, 2, va, Float32Array), Yva: Int32Array.from(va.map((k) => Y[k])), nVa: va.length, tooFew: points.length < 8 || used.length < 2 };
  }
  // a formula as the data uses it: it must parse, and have a finite value everywhere on [−1, 1]
  BT.checkFormula = function (src) {
    const c = BT.compile(src);
    if (c.error) return c;
    for (let i = 0; i <= 400; i++) {
      const x = -1 + i / 200, y = c.f(x);
      if (!Number.isFinite(y)) return { error: `The formula has no finite value at x = ${NA.fmt(x, 2)}, for example a division by zero or the log of a negative number.` };
    }
    return c;
  };
  function curveData(p, s, seed) {
    const rng = new NA.RNG(seed * 13 + 5);
    let xs, ys, f = null;
    if (p.custom) {
      const c = BT.checkFormula(s.formula);
      if (c.error) return { family: 'curve', error: c.error, input: { kind: 'vector', dim: 1 }, output: { kind: 'value', k: 1 } };
      f = c.f;
      xs = NA.range(s.n).map(() => rng.uniform(-1, 1));
      ys = xs.map((x) => f(x));
      const bad = ys.findIndex((v) => !Number.isFinite(v));
      if (bad >= 0) return { family: 'curve', error: `The formula has no finite value at x = ${NA.fmt(xs[bad], 2)}, for example a division by zero or the log of a negative number.`, input: { kind: 'vector', dim: 1 }, output: { kind: 'value', k: 1 } };
    } else {
      const ds = NA.data.buildDataset({ dataset: p.dataset, n: s.n, noise: 0, trainRatio: 0.75, seed });
      xs = Array.from(ds.raw); ys = Array.from(ds.y);
    }
    // keep the values on a unit scale (the plot shows the original numbers)
    const scale = Math.max(1e-9, ...ys.map(Math.abs)), k = scale > 1.5 || scale < 0.25 ? 1 / scale : 1;
    const noisy = ys.map((v) => v * k + rng.gauss(0, s.noise));
    const [tr, va] = split(xs.length, rng, 0.75);
    const truth = (x) => (f ? f(x) * k : null);
    return { family: 'curve', input: { kind: 'vector', dim: 1 }, output: { kind: 'value', k: 1 }, scale: k, truth: f ? truth : null,
      Xtr: Float32Array.from(tr.map((i) => xs[i])), Ytr: Float32Array.from(tr.map((i) => noisy[i])), nTr: tr.length,
      Xva: Float32Array.from(va.map((i) => xs[i])), Yva: Float32Array.from(va.map((i) => noisy[i])), nVa: va.length };
  }
  function digitData(s, seed) {
    const ds = NA.data.buildDataset({ dataset: 'digits', n: s.n, noise: 0, trainRatio: 0.8, seed }), P = NA.data.IMG * NA.data.IMG;
    const val = Array.from(ds.valIdx).concat(Array.from(ds.testIdx)), tr = Array.from(ds.trainIdx);
    return { family: 'digits', input: { kind: 'image', c: 1, h: NA.data.IMG, w: NA.data.IMG }, output: { kind: 'classes', k: 10 }, classes: 10, names: NA.range(10).map(String),
      Xtr: gather(ds.raw, P, tr, Float32Array), Ytr: Int32Array.from(tr.map((k) => ds.y[k])), nTr: tr.length,
      Xva: gather(ds.raw, P, val, Float32Array), Yva: Int32Array.from(val.map((k) => ds.y[k])), nVa: val.length };
  }
  /* Digit tasks: the model reads x1 … xL, a → marker, then writes the answer one digit at a time. It is trained
     only on the answer positions (the inputs are random, so predicting them is impossible). */
  const ARROW = 10;
  BT.algoAnswer = (task, xs) => (task === 'reverse' ? xs.slice().reverse() : task === 'sort' ? xs.slice().sort((a, b) => a - b) : xs.slice());
  function algoData(task, s, seed) {
    const L = s.len, T = 2 * L, rng = new NA.RNG(seed * 31 + 7), seen = new Set(), all = [];
    const total = Math.min(s.n, Math.pow(10, L) - 50);
    let guard = 0;
    while (all.length < total && guard++ < total * 20) {
      const xs = NA.range(L).map(() => rng.int(10)), key = xs.join('');
      if (seen.has(key)) continue;
      seen.add(key); all.push(xs);
    }
    const nVa = Math.max(50, Math.round(all.length * 0.12)), nTr = all.length - nVa;
    const X = new Int32Array(all.length * T), Y = new Int32Array(all.length * T).fill(-1);
    all.forEach((xs, i) => {
      const seq = xs.concat([ARROW], BT.algoAnswer(task, xs));
      for (let t = 0; t < T; t++) { X[i * T + t] = seq[t]; if (t >= L) Y[i * T + t] = seq[t + 1]; }
    });
    return { family: 'sequence', task, input: { kind: 'tokens', vocab: 11, len: T, embed: s.embed }, output: { kind: 'tokens', vocab: 11 }, vocab: [...'0123456789', '→'], len: L,
      Xtr: X.slice(0, nTr * T), Ytr: Y.slice(0, nTr * T), nTr, Xva: X.slice(nTr * T), Yva: Y.slice(nTr * T), nVa };
  }
  // the tiny language: stories padded to a fixed length with a mask on the padding
  function languageData(s, seed) {
    const LM = NA.lmData, T = LM.MAX_LEN, stories = LM.makeSet(seed * 101 + 1, s.n), nVa = Math.round(s.n * 0.12), nTr = s.n - nVa;
    const X = new Int32Array(s.n * T), Y = new Int32Array(s.n * T).fill(-1);
    stories.forEach((st, i) => { const ids = st.ids.slice(0, T + 1); for (let t = 0; t < T; t++) { X[i * T + t] = t < ids.length ? ids[t] : LM.W.DOT; if (t + 1 < ids.length) Y[i * T + t] = ids[t + 1]; } });
    return { family: 'sequence', task: 'language', input: { kind: 'tokens', vocab: LM.VOCAB.length, len: T, embed: s.embed }, output: { kind: 'tokens', vocab: LM.VOCAB.length }, vocab: LM.VOCAB.slice(),
      Xtr: X.slice(0, nTr * T), Ytr: Y.slice(0, nTr * T), nTr, Xva: X.slice(nTr * T), Yva: Y.slice(nTr * T), nVa };
  }
  // your own text: words and punctuation become tokens; every window of the text is a training example
  BT.wordsOf = (text) => String(text || '').toLowerCase().replace(/([.,!?;:])/g, ' $1 ').replace(/\s+/g, ' ').trim().split(' ').filter(Boolean);
  function sentenceData(s, seed) {
    const words = BT.wordsOf(s.text), vocab = [...new Set(words)].slice(0, 400), id = Object.fromEntries(vocab.map((w, i) => [w, i]));
    const T = Math.max(3, Math.min(s.window, words.length - 1)), ids = words.filter((w) => w in id).map((w) => id[w]);
    if (ids.length < T + 2 || vocab.length < 3) return { family: 'sequence', error: 'Write a few more words: the model needs at least a couple of sentences to learn from.', input: { kind: 'tokens', vocab: 3, len: 3, embed: s.embed }, output: { kind: 'tokens', vocab: 3 } };
    const n = ids.length - T, rng = new NA.RNG(seed * 17 + 9), order = rng.shuffle(NA.range(n)), nVa = n >= 40 ? Math.round(n * 0.1) : 0, nTr = n - nVa;
    const X = new Int32Array(n * T), Y = new Int32Array(n * T);
    order.forEach((st, i) => { for (let t = 0; t < T; t++) { X[i * T + t] = ids[st + t]; Y[i * T + t] = ids[st + t + 1]; } });
    // too little text to hold some back: the scores are then measured on the training text itself
    const noHoldout = !nVa, va = noHoldout ? 0 : nTr * T;
    return { family: 'sequence', task: 'sentences', input: { kind: 'tokens', vocab: vocab.length, len: T, embed: s.embed }, output: { kind: 'tokens', vocab: vocab.length }, vocab, wordId: id,
      Xtr: X.slice(0, nTr * T), Ytr: Y.slice(0, nTr * T), nTr, Xva: X.slice(va), Yva: Y.slice(va), nVa: noHoldout ? nTr : nVa, noHoldout, textLen: words.length };
  }

  // ------------------------------------------------------------------ measuring and decoding
  // R² of a regression on n examples: 1 − (squared error) / (variance of the targets)
  BT.r2 = (pred, Y, n) => { let m = 0; for (let i = 0; i < n; i++) m += Y[i]; m /= Math.max(1, n); let se = 0, st = 0; for (let i = 0; i < n; i++) { se += (pred[i] - Y[i]) ** 2; st += (Y[i] - m) ** 2; } return st > 1e-12 ? 1 - se / st : 0; };
  // the highest-scoring token at each position: out is n × T × V
  BT.argmaxTokens = (out, n, T, V) => { const r = new Int32Array(n * T); for (let i = 0; i < n * T; i++) { let b = 0; for (let v = 1; v < V; v++) if (out[i * V + v] > out[i * V + b]) b = v; r[i] = b; } return r; };
  /* Writes the answer of a digit task one token at a time, feeding each guess back in (what using the model
     really means: during training it was shown the right answers). Returns n × L digits. */
  BT.generateAlgo = function (model, prompts, n, L) {
    const T = 2 * L, X = new Int32Array(n * T), out = new Int32Array(n * L);
    for (let i = 0; i < n; i++) { for (let t = 0; t < L; t++) X[i * T + t] = prompts[i * L + t]; X[i * T + L] = ARROW; }
    for (let k = 0; k < L; k++) {
      const o = model.predict(X, n), V = 11;
      for (let i = 0; i < n; i++) {
        const base = (i * T + L + k) * V;
        let b = 0; for (let v = 1; v < V; v++) if (o[base + v] > o[base + b]) b = v;
        out[i * L + k] = b;
        if (L + k + 1 < T) X[i * T + L + k + 1] = b;
      }
    }
    return out;
  };
  // Continue a text: the last `len` tokens are the context; greedy, or sampled with a temperature.
  BT.continueText = function (model, ids, len, V, count, temp, rng) {
    const out = ids.slice(), X = new Int32Array(len);
    for (let k = 0; k < count; k++) {
      const ctx = out.slice(-len), pad = len - ctx.length;
      for (let t = 0; t < len; t++) X[t] = t < pad ? ctx[0] ?? 0 : ctx[t - pad];
      const o = model.predict(X, 1), base = (len - 1) * V;
      let pick = 0;
      if (temp > 0 && rng) {
        let m = -Infinity; for (let v = 0; v < V; v++) m = Math.max(m, o[base + v]);
        const p = []; let z = 0; for (let v = 0; v < V; v++) { const e = Math.exp((o[base + v] - m) / temp); p.push(e); z += e; }
        let r = rng.next() * z; for (let v = 0; v < V; v++) { r -= p[v]; if (r <= 0) { pick = v; break; } }
      } else { for (let v = 1; v < V; v++) if (o[base + v] > o[base + pick]) pick = v; }
      out.push(pick);
    }
    return out.slice(ids.length);
  };

  // ------------------------------------------------------------------ a small formula language for "your own function"
  /* Numbers, x, + − × / ^, parentheses, pi and e, and sin cos tan exp log ln sqrt abs tanh sign floor round
     min max. Multiplication can be implied: 2x, 3sin(x), (x+1)(x−1). Returns { f, py } or { error }: f computes
     the formula, py is the same formula as PyTorch code on a tensor x (for the Python tab). No eval. */
  const FUN = { sin: Math.sin, cos: Math.cos, tan: Math.tan, exp: Math.exp, log: Math.log, ln: Math.log, sqrt: Math.sqrt, abs: Math.abs, tanh: Math.tanh, sign: Math.sign, floor: Math.floor, round: Math.round, relu: (v) => Math.max(0, v), sigmoid: (v) => 1 / (1 + Math.exp(-v)) };
  const FUN2 = { min: Math.min, max: Math.max, pow: Math.pow };
  const PYNUM = (v) => (Number.isNaN(v) ? "float('nan')" : !Number.isFinite(v) ? (v > 0 ? 'math.inf' : '-math.inf') : String(+v.toPrecision(12)));
  BT.compile = function (src) {
    const s = String(src || '').replace(/×|·/g, '*').replace(/−/g, '-').replace(/²/g, '^2').replace(/³/g, '^3').replace(/π/g, 'pi').toLowerCase();
    let i = 0;
    const ws = () => { while (s[i] === ' ') i++; };
    const fail = (msg) => { throw new Error(msg); };
    const peek = () => { ws(); return s[i]; };
    /* Each part is { f, py, k, p }: k when it does not depend on x (it is then folded into one number), p the
       precedence of its Python text (1 + −, 2 × /, 3 unary minus, 4 power, 5 a name, number or call). */
    const lit = (v, py) => ({ f: () => v, py: py || PYNUM(v), k: true, p: v < 0 || Object.is(v, -0) ? 3 : 5 });
    const wrap = (a, min) => (a.p < min ? `(${a.py})` : a.py);
    const node = (f, py, k, p) => (k ? lit(f(0)) : { f, py, k: false, p });
    function expr() {
      let a = term();
      for (;;) {
        const c = peek();
        if (c !== '+' && c !== '-') return a;
        i++;
        const b = term(), A = a;
        a = node(c === '+' ? (x) => A.f(x) + b.f(x) : (x) => A.f(x) - b.f(x), `${A.py} ${c} ${wrap(b, 2)}`, A.k && b.k, 1);
      }
    }
    function term() {
      let a = unary();
      for (;;) {
        const c = peek();
        let op, b;
        if (c === '*' || c === '/') { i++; op = c; b = unary(); }
        else if (c && /[a-z0-9.(]/.test(c)) { op = '*'; b = unary(); } // implied multiplication
        else return a;
        const A = a;
        a = node(op === '*' ? (x) => A.f(x) * b.f(x) : (x) => A.f(x) / b.f(x), `${wrap(A, 2)} ${op} ${wrap(b, op === '*' ? 3 : 4)}`, A.k && b.k, 2);
      }
    }
    function unary() {
      const c = peek();
      if (c === '-') { i++; const g = unary(); return node((x) => -g.f(x), `-${wrap(g, 3)}`, g.k, 3); }
      if (c === '+') { i++; return unary(); }
      return power();
    }
    function power() {
      const b = atom();
      if (peek() !== '^') return b;
      i++;
      const e = unary();
      return node((x) => Math.pow(b.f(x), e.f(x)), `${wrap(b, 5)} ** ${wrap(e, 3)}`, b.k && e.k, 4);
    }
    function atom() {
      const c = peek();
      if (c === undefined) fail('The formula ends too early.');
      if (c === '(') { i++; const f = expr(); if (peek() !== ')') fail('A bracket is not closed.'); i++; return f; }
      if (/[0-9.]/.test(c)) { const m = /^\d*\.?\d+(e[+-]?\d+)?/.exec(s.slice(i)); if (!m) fail(`"${c}" is not a number.`); i += m[0].length; return lit(+m[0], m[0]); }
      const m = /^[a-z]+/.exec(s.slice(i));
      if (!m) fail(`I do not understand "${c}".`);
      let w = m[0];
      // a run of letters may hide an implied product, like "xsin": take the longest known name at the start
      const names = ['x', 'pi', 'e', ...Object.keys(FUN), ...Object.keys(FUN2)].sort((a, b) => b.length - a.length);
      const hit = names.find((n) => w.startsWith(n));
      if (!hit) fail(`Unknown name "${w}". Use x, pi, e and sin, cos, tan, exp, log, sqrt, abs, tanh, sign, floor, round, min, max.`);
      w = hit; i += w.length;
      if (w === 'x') return { f: (x) => x, py: 'x', k: false, p: 5 };
      if (w === 'pi') return lit(Math.PI, 'math.pi');
      if (w === 'e' && peek() !== '(') return lit(Math.E, 'math.e');
      if (peek() !== '(') fail(`"${w}" needs brackets, as in ${w}(x).`);
      i++;
      const a = expr();
      if (w in FUN2) {
        if (peek() !== ',') fail(`${w} takes two values, as in ${w}(x, 0).`);
        i++;
        const b = expr();
        if (peek() !== ')') fail('A bracket is not closed.');
        i++;
        const fn = FUN2[w];
        const py = w === 'pow' ? `${wrap(a, 5)} ** ${wrap(b, 3)}` : a.k || b.k ? `torch.clamp(${(a.k ? b : a).py}, ${w === 'min' ? 'max' : 'min'}=${(a.k ? a : b).py})` : `torch.${w === 'min' ? 'minimum' : 'maximum'}(${a.py}, ${b.py})`;
        return node((x) => fn(a.f(x), b.f(x)), py, a.k && b.k, w === 'pow' ? 4 : 5);
      }
      if (peek() !== ')') fail('A bracket is not closed.');
      i++;
      const fn = FUN[w];
      return node((x) => fn(a.f(x)), `torch.${w === 'ln' ? 'log' : w}(${a.py})`, a.k, 5);
    }
    try {
      if (!s.trim()) fail('Type a formula in x.');
      const f = expr();
      if (i < s.length && peek() !== undefined) fail(`Unexpected "${s[i]}".`);
      if (!Number.isFinite(f.f(0.37)) && !Number.isFinite(f.f(-0.61))) fail('The formula has no finite values between −1 and 1.');
      return { f: f.f, py: f.py, constant: f.k };
    } catch (e) { return { error: e.message }; }
  };

  NA.builderTasks = BT;
})(typeof globalThis !== 'undefined' ? (globalThis.NA = globalThis.NA || {}) : (window.NA = window.NA || {}));
