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
    surface: { label: 'Fit a surface', losses: ['mse', 'mae', 'huber'], metric: 'r2', template: 'mlp', icon: '◍' },
    digits: { label: 'Read pictures', losses: ['xent', 'mse', 'hinge'], metric: 'accuracy', template: 'cnn', icon: '▦' },
    series: { label: 'Forecast a series', losses: ['mse', 'mae', 'huber'], metric: 'r2', template: 'lstm', icon: '〰' },
    seqclass: { label: 'Classify sequences', losses: ['xent', 'mse', 'hinge'], metric: 'accuracy', template: 'lstmclass', icon: '⋯' },
    sequence: { label: 'Write sequences', losses: ['xent'], metric: 'tokens', template: 'transformer', icon: '⋯' },
    // a table of your own: classes or a value, depending on the column it predicts (BT.lossesFor gives the losses)
    table: { label: 'Your data', losses: ['xent', 'mse', 'hinge', 'mae', 'huber'], metric: 'auto', template: 'mlp', icon: '▤' },
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
    { id: 'hill', family: 'surface', task: 'hill', label: 'Two hills', hint: 'A hill and a hollow on a plane: a smooth height to learn from (x, y).' },
    { id: 'ripples', family: 'surface', task: 'ripples', label: 'Ripples', hint: 'Rings spreading from the middle, like a pebble dropped in a pond.' },
    { id: 'saddle', family: 'surface', task: 'saddle', label: 'Saddle', hint: 'Up one way, down the other: x² − y².' },
    { id: 'checker', family: 'surface', task: 'checker', label: 'Checkerboard', hint: 'A smooth checkerboard of highs and lows: many bends in both directions.' },
    { id: 'digits', family: 'digits', label: 'Handwritten digits', hint: '14 × 14 pictures of the digits 0 to 9, drawn with random strokes.' },
    { id: 'shapes', family: 'digits', task: 'shapes', label: 'Shapes', hint: 'Circles, squares, triangles, crosses and lines, drawn anywhere at any size and angle: what convolutions are for.' },
    { id: 'waves', family: 'series', task: 'waves', label: 'Two waves', hint: 'Two sine waves added up: predict the next value from the last few.' },
    { id: 'seasonal', family: 'series', task: 'seasonal', label: 'Seasons and trend', hint: 'A slow rise, a repeating season and noise, like a shop’s sales.' },
    { id: 'square', family: 'series', task: 'square', label: 'Square wave', hint: 'Flat stretches and sudden jumps: the model has to know when the next jump comes.' },
    { id: 'chaos', family: 'series', task: 'chaos', label: 'Chaos', hint: 'The logistic map, x → 3.9·x(1 − x): completely determined, yet it never repeats. The next value is a simple curve of the last one.' },
    { id: 'sentiment', family: 'seqclass', task: 'sentiment', label: 'Tiny reviews', hint: 'Short reviews with good and bad words; “not” flips a word and after “but” the second part decides. Word order matters.' },
    { id: 'majority', family: 'seqclass', task: 'majority', label: 'Majority vote', hint: 'More 1s or more 0s? Counting is enough: order does not matter.' },
    { id: 'pattern', family: 'seqclass', task: 'pattern', label: 'Find a pattern', hint: 'Is there a 3 followed straight away by a 7? Only the order of neighbours tells.' },
    { id: 'parity', family: 'seqclass', task: 'parity', label: 'Parity', hint: 'An odd or even number of 1s? Flipping any one bit changes the answer: famously hard to learn.' },
    { id: 'reverse', family: 'sequence', task: 'reverse', label: 'Reverse a sequence', hint: '3 1 4 → 4 1 3. Each answer digit must look back at a different input position.' },
    { id: 'sort', family: 'sequence', task: 'sort', label: 'Sort digits', hint: '3 1 4 → 1 3 4. Needs to compare all the inputs.' },
    { id: 'copy', family: 'sequence', task: 'copy', label: 'Copy a sequence', hint: '3 1 4 → 3 1 4. The easiest look-back task.' },
    { id: 'language', family: 'sequence', task: 'language', label: 'Tiny language', hint: 'Predict the next word of short stories, like the Tiny Transformer model.' },
    { id: 'sentences', family: 'sequence', task: 'sentences', custom: true, label: 'Your own sentences', hint: 'Type some text; the model learns to predict each next word.' },
    { id: 'csv', family: 'table', custom: true, label: 'Your table (CSV)', hint: 'A table of numbers from a CSV file, or a sample: the model learns to predict one column from the others.' },
  ];
  BT.byId = Object.fromEntries(BT.PROBLEMS.map((p) => [p.id, p]));
  // the kinds of data whose answer is a number, scored by R² (a table that predicts a number makes curve-like data)
  BT.isValue = (family) => family === 'curve' || family === 'surface' || family === 'series';
  BT.CLASS_COLOURS = ['s1', 's2', 's4', 's7'];

  // problem settings with their defaults (the inspector shows the ones that apply)
  BT.defaults = (p) => {
    const f = p.family;
    if (f === 'plane') return { noise: p.id === 'xor' || p.id === 'blobs' ? 0.1 : 0.15, n: 600 };
    if (f === 'curve') return { noise: 0.05, n: 200, formula: 'sin(5x) + x^2' };
    if (f === 'digits') return { n: p.task === 'shapes' ? 1500 : 2000 };
    if (f === 'surface') return { noise: 0.02, n: 800 };
    if (f === 'series') return { window: 16, n: 1200, noise: p.task === 'chaos' ? 0 : 0.04 };
    if (f === 'seqclass') return { n: p.task === 'sentiment' ? 2500 : 2000, embed: 16 };
    if (f === 'table') return { target: -1, kind: 'auto' };
    if (p.task === 'language') return { n: 1500, embed: 32 };
    if (p.task === 'sentences') return { window: 8, embed: 32, text: SAMPLE_TEXT };
    return { len: 4, n: 3000, embed: 32 };
  };
  const SAMPLE_TEXT = 'the sun is hot . the snow is cold . the fire is hot . the ice is cold .\nthe cat sat on the mat . the dog sat on the log .\nred and blue make purple . blue and yellow make green . red and yellow make orange .';

  // training settings: the defaults, and what each kind of data starts with when you switch to it
  BT.DEFAULT_TRAIN = { opt: 'adam', lr: 0.01, batch: 32, wd: 0, sched: 'const', clip: 0, init: 'auto', seed: 1, steps: 3000 };
  BT.trainFor = (p) => {
    if (p.family === 'sequence') return { lr: 0.003, batch: p.task === 'language' ? 16 : 32, steps: p.task === 'language' ? 3000 : p.task === 'sentences' ? 1500 : 1000, clip: 1 };
    if (p.family === 'digits') return { lr: 0.003, batch: 32, steps: 1500, clip: 0 };
    if (p.family === 'table') return { lr: 0.01, batch: 32, steps: 2000, clip: 0 };
    if (p.family === 'series') return { lr: 0.003, batch: 32, steps: 2000, clip: 1 };
    if (p.family === 'seqclass') return { lr: 0.003, batch: 32, steps: 1500, clip: 1 };
    return { lr: 0.01, batch: 32, steps: 3000, clip: 0 };
  };

  // ------------------------------------------------------------------ templates
  const D = (units, act = 'relu') => ({ type: 'dense', units, act });
  BT.TEMPLATES = [
    { id: 'empty', label: 'Empty', hint: 'Just the embedding and the output: build the rest yourself.', families: ['sequence'], blocks: () => [] },
    { id: 'linear', label: 'Linear', hint: 'No hidden layer: one weighted sum per output, so only straight boundaries. Build up from here.', families: ['plane', 'curve', 'surface', 'digits', 'table', 'series'], blocks: () => [] },
    { id: 'mlp', label: 'Small MLP', hint: 'Two hidden layers of 16 ReLU neurons.', families: ['plane', 'curve', 'surface', 'digits', 'table', 'series'], blocks: () => [D(16), D(16)] },
    { id: 'deep', label: 'Deep MLP', hint: 'Five hidden layers of 24 neurons.', families: ['plane', 'curve', 'surface', 'digits', 'table'], blocks: () => [D(24), D(24), D(24), D(24), D(24)] },
    { id: 'wide', label: 'One wide layer', hint: 'A single hidden layer of 64 neurons.', families: ['plane', 'curve', 'surface', 'digits', 'table', 'series'], blocks: () => [D(64)] },
    { id: 'cnn', label: 'Small CNN', hint: 'Two convolution + pooling stages, then a dense layer.', families: ['digits'], blocks: () => [{ type: 'conv', filters: 8, k: 3, stride: 1, act: 'relu' }, { type: 'pool', size: 2 }, { type: 'conv', filters: 16, k: 3, stride: 1, act: 'relu' }, { type: 'pool', size: 2 }, { type: 'flatten' }, D(32)] },
    { id: 'transformer', label: 'Tiny transformer', hint: 'Two transformer blocks with two attention heads each.', families: ['sequence'], blocks: () => [{ type: 'transformer', heads: 2, mlp: 4, causal: true }, { type: 'transformer', heads: 2, mlp: 4, causal: true }, { type: 'layernorm' }] },
    { id: 'attention', label: 'One attention layer', hint: 'A single attention layer and an MLP, the smallest model that can look back.', families: ['sequence'], blocks: () => [{ type: 'attention', heads: 1, causal: true, residual: true }, { type: 'dense', units: 32, act: 'relu', residual: false }] },
    { id: 'noattn', label: 'No attention', hint: 'An MLP applied to each position alone: it cannot look at other positions, so reversing fails.', families: ['sequence'], blocks: () => [D(64), D(32)] },
    { id: 'fourier', label: 'Fourier MLP', hint: 'Sine and cosine features of the input, then two layers: sharp, wiggly functions come easily.', families: ['curve', 'surface'], blocks: () => [{ type: 'fourier', features: 32, scale: 2 }, D(32), D(32)] },
    { id: 'lstm', label: 'LSTM', hint: 'A recurrent LSTM reads the window one step at a time, then its last state predicts.', families: ['series'], blocks: () => [{ type: 'rnn', cell: 'lstm', units: 24 }, { type: 'seqpool', mode: 'last' }] },
    { id: 'gru', label: 'GRU', hint: 'A GRU: like an LSTM with two gates instead of three, and no separate memory.', families: ['series'], blocks: () => [{ type: 'rnn', cell: 'gru', units: 24 }, { type: 'seqpool', mode: 'last' }] },
    { id: 'tcn', label: '1-D CNN', hint: 'Causal 1-D convolutions with growing dilation, like WaveNet: each layer sees twice as far back.', families: ['series'], blocks: () => [{ type: 'conv1d', filters: 16, k: 3, dilation: 1, causal: true, act: 'relu' }, { type: 'conv1d', filters: 16, k: 3, dilation: 2, causal: true, act: 'relu' }, { type: 'conv1d', filters: 16, k: 3, dilation: 4, causal: true, act: 'relu' }, { type: 'seqpool', mode: 'last' }] },
    { id: 'seqformer', label: 'Transformer', hint: 'Each value lifted to 16 numbers, positions added, then a transformer block: attention across the whole window.', families: ['series'], blocks: () => [D(16, 'linear'), { type: 'position', mode: 'learned' }, { type: 'transformer', heads: 2, mlp: 2, causal: false }, { type: 'seqpool', mode: 'last' }] },
    { id: 'bag', label: 'Bag of words', hint: 'Averages the word vectors and decides from that: fast, but blind to word order.', families: ['seqclass'], blocks: () => [{ type: 'gap' }, D(16)] },
    { id: 'lstmclass', label: 'LSTM', hint: 'An LSTM reads the sequence word by word; its last state decides.', families: ['seqclass'], blocks: () => [{ type: 'rnn', cell: 'lstm', units: 32 }, { type: 'seqpool', mode: 'last' }] },
    { id: 'convclass', label: '1-D CNN', hint: 'Filters slide along the sequence and look for short patterns; the strongest match anywhere decides.', families: ['seqclass'], blocks: () => [{ type: 'conv1d', filters: 32, k: 3, dilation: 1, causal: false, act: 'relu' }, { type: 'seqpool', mode: 'max' }] },
    { id: 'attnclass', label: 'Transformer', hint: 'A transformer block over the whole sequence, then the average.', families: ['seqclass'], blocks: () => [{ type: 'transformer', heads: 2, mlp: 2, causal: false }, { type: 'gap' }] },
    { id: 'vit', label: 'Vision transformer', hint: 'The picture cut into 2 × 2 patches, each a token, and a transformer over them: the CNN-free way to see.', families: ['digits'], blocks: () => [{ type: 'patches', size: 2, dim: 32 }, { type: 'transformer', heads: 2, mlp: 2, causal: false }, { type: 'gap' }] },
    { id: 'hybrid', label: 'CNN + transformer', hint: 'A convolution finds strokes, then the feature map becomes patches for attention: a hybrid.', families: ['digits'], blocks: () => [{ type: 'conv', filters: 8, k: 3, stride: 1, act: 'relu' }, { type: 'pool', size: 2 }, { type: 'patches', size: 1, dim: 32 }, { type: 'transformer', heads: 2, mlp: 2, causal: false }, { type: 'gap' }] },
    { id: 'c-linear', label: 'Linear / logistic regression', hint: 'The classic: one weight per input. Least squares for numbers (solved in one step), logistic regression for classes.', families: ['plane', 'curve', 'surface', 'digits', 'table', 'series'], blocks: () => [{ type: 'linear', lambda: 0 }] },
    { id: 'c-knn', label: 'k-nearest neighbours', hint: 'No training at all: each answer is a vote of the k most similar training examples.', families: ['plane', 'curve', 'surface', 'digits', 'table', 'series'], blocks: () => [{ type: 'knn', k: 5, weights: 'uniform' }] },
    { id: 'c-tree', label: 'Decision tree', hint: 'A flowchart of yes-or-no questions about the inputs, learned one level at a time.', families: ['plane', 'curve', 'surface', 'digits', 'table', 'series'], blocks: () => [{ type: 'tree', depth: 6, minLeaf: 2 }] },
    { id: 'c-forest', label: 'Random forest', hint: 'Many deep trees, each on a random sample of the data and of the inputs, voting together.', families: ['plane', 'curve', 'surface', 'digits', 'table', 'series'], blocks: () => [{ type: 'forest', trees: 50, depth: 8, features: 'sqrt', minLeaf: 1 }] },
    { id: 'c-boost', label: 'Gradient boosting (XGBoost)', hint: 'Small trees added one after another, each fixing what the others still get wrong: the winner on tables.', families: ['plane', 'curve', 'surface', 'digits', 'table', 'series'], blocks: () => [{ type: 'boost', rounds: 100, eta: 0.3, depth: 4, lambda: 1, gamma: 0, subsample: 1, colsample: 1 }] },
  ];
  BT.templatesFor = (family) => BT.TEMPLATES.filter((t) => t.families.includes(family));
  // the losses that suit a problem; a table's depend on whether its data is classes or a value
  BT.lossesFor = (p, d) => (p.family !== 'table' ? BT.FAMILIES[p.family].losses : d && d.output && d.output.kind === 'value' ? ['mse', 'mae', 'huber'] : ['xent', 'mse', 'hinge']);
  // a template's blocks for a kind of data: dense layers need a flat vector, so on pictures and series the plain
  // layer templates start with a Flatten (classic models read the rows of numbers as they are)
  const FLAT_TEMPLATES = new Set(['linear', 'mlp', 'deep', 'wide']);
  BT.templateBlocks = (t, family) => {
    const blocks = t.blocks();
    // a window of a series read as one row of numbers, like the pixels of a picture (classic models read rows anyway)
    if ((family === 'digits' || family === 'series') && FLAT_TEMPLATES.has(t.id)) blocks.unshift({ type: 'flatten' });
    return blocks;
  };


  // ------------------------------------------------------------------ your own table (CSV)
  /* A table from a CSV file: { name, header: [column names], rows: [[cells as text]], delim, hasHeader }. The model
     learns to predict one column (the target) from the columns that hold numbers. A target with text, or with few
     different whole numbers, is a set of classes; any other number column is a value to predict. Kept small, since
     the setup is saved in the browser: at most 3,000 rows and 40 columns. */
  BT.TABLE_LIMITS = { rows: 3000, cols: 40, classes: 20 };
  const isNumText = (s) => s !== '' && Number.isFinite(Number(s));
  function splitLine(line, delim) {
    const out = [];
    let cur = '', q = false;
    for (let i = 0; i < line.length; i++) {
      const ch = line[i];
      if (q) { if (ch === '"') { if (line[i + 1] === '"') { cur += '"'; i++; } else q = false; } else cur += ch; }
      else if (ch === '"' && cur.trim() === '') { q = true; cur = ''; }
      else if (ch === delim) { out.push(cur.trim()); cur = ''; }
      else cur += ch;
    }
    out.push(cur.trim());
    return out;
  }
  // Reads CSV text: the separator (comma, semicolon or tab) is guessed from the first line, and the first row is
  // taken as column names when its cells are text where the next row's are numbers. Returns a table or { error }.
  BT.parseCSV = function (text, name = 'table.csv') {
    const lines = String(text).replace(/^﻿/, '').split(/\r\n|\n|\r/).filter((l) => l.trim() !== '');
    if (lines.length < 2) return { error: 'The file needs at least two lines: a table of rows.' };
    const count = (c) => lines[0].split(c).length - 1, delim = ['\t', ';', ','].reduce((b, c) => (count(c) > count(b) ? c : b), ',');
    let rows = lines.map((l) => splitLine(l, delim));
    // with semicolons between columns, a comma is usually the decimal mark: 1,5 means 1.5
    if (delim === ';') rows = rows.map((r) => r.map((v) => (/^[-+]?\d+,\d+$/.test(v) ? v.replace(',', '.') : v)));
    const C = Math.max(...rows.slice(0, 50).map((r) => r.length));
    if (C < 2) return { error: 'Only one column found: the table needs at least one column to learn from and one to predict. Columns are separated by commas, semicolons or tabs.' };
    if (C > BT.TABLE_LIMITS.cols) return { error: `${C} columns: the builder takes at most ${BT.TABLE_LIMITS.cols}.` };
    const first = rows[0], second = rows[1];
    const hasHeader = first.some((v, i) => !isNumText(v) && isNumText(second[i] || ''));
    const header = hasHeader ? first.map((v, i) => v || `column ${i + 1}`) : NA.range(C).map((i) => `column ${i + 1}`);
    while (header.length < C) header.push(`column ${header.length + 1}`);
    rows = (hasHeader ? rows.slice(1) : rows).map((r) => { const q = r.slice(0, C); while (q.length < C) q.push(''); return q; });
    const cut = rows.length > BT.TABLE_LIMITS.rows ? rows.length - BT.TABLE_LIMITS.rows : 0;
    if (cut) rows = rows.slice(0, BT.TABLE_LIMITS.rows);
    return { name, header, rows, delim, hasHeader, cut };
  };
  // For each column: does it hold numbers, how many different values, how many cells are empty.
  BT.tableColumns = function (t) {
    return t.header.map((h, i) => {
      let num = 0, empty = 0;
      const seen = new Set();
      for (const r of t.rows) { const v = r[i]; if (v === '') { empty++; continue; } if (isNumText(v)) num++; seen.add(v); }
      return { index: i, name: h, numeric: num > 0 && num === t.rows.length - empty, unique: seen.size, empty };
    });
  };
  // the target column (the last by default) and whether it is classes or a value
  BT.tableTarget = function (t, s) {
    const cols = BT.tableColumns(t), target = Number.isInteger(s.target) && s.target >= 0 && s.target < cols.length ? s.target : cols.length - 1, c = cols[target];
    let kind = s.kind === 'classes' || s.kind === 'value' ? s.kind : c.numeric && c.unique > 12 ? 'value' : 'classes';
    if (!c.numeric) kind = 'classes';
    return { cols, target, kind, col: c };
  };
  function tableData(t, s, seed) {
    const none = { family: 'plane', table: true, input: { kind: 'vector', dim: 1 }, output: { kind: 'classes', k: 2 } };
    if (!t || !Array.isArray(t.rows) || !t.rows.length) return Object.assign(none, { error: 'No table yet: choose a CSV file, or one of the samples, in the Input block’s settings.', empty: true });
    const { cols, target, kind } = BT.tableTarget(t, s);
    const feats = cols.filter((c) => c.index !== target && c.numeric).map((c) => c.index);
    if (!feats.length) return Object.assign(none, { error: 'No column of numbers to learn from: the model reads the columns that hold numbers.' });
    const rows = t.rows.filter((r) => feats.every((i) => isNumText(r[i])) && r[target] !== '' && (kind === 'classes' || isNumText(r[target])));
    if (rows.length < 10) return Object.assign(none, { error: `Only ${rows.length} rows have a number in every input column and something to predict: at least 10 are needed.` });
    const F = feats.length, n = rows.length, X = new Float32Array(n * F), mean = new Array(F).fill(0), std = new Array(F).fill(0);
    rows.forEach((r, k) => feats.forEach((c, j) => { X[k * F + j] = Number(r[c]); }));
    for (let j = 0; j < F; j++) {
      for (let k = 0; k < n; k++) mean[j] += X[k * F + j] / n;
      for (let k = 0; k < n; k++) std[j] += (X[k * F + j] - mean[j]) ** 2 / n;
      std[j] = Math.sqrt(std[j]) || 1;
      for (let k = 0; k < n; k++) X[k * F + j] = (X[k * F + j] - mean[j]) / std[j];
    }
    let Y, labels = null, ystats = null;
    if (kind === 'classes') {
      const numeric = cols[target].numeric, key = (v) => (numeric ? String(Number(v)) : v);
      labels = [...new Set(rows.map((r) => key(r[target])))].sort((a, b) => (numeric ? Number(a) - Number(b) : a.localeCompare(b)));
      if (labels.length < 2) return Object.assign(none, { error: `The column to predict, ${t.header[target]}, has only one value: there is nothing to tell apart.` });
      if (labels.length > BT.TABLE_LIMITS.classes) return Object.assign(none, { error: `${t.header[target]} has ${labels.length} different values: too many classes (at most ${BT.TABLE_LIMITS.classes}).${cols[target].numeric ? ' Predict it as a number instead.' : ''}` });
      const idx = Object.fromEntries(labels.map((l, i) => [l, i]));
      Y = Int32Array.from(rows.map((r) => idx[key(r[target])]));
    } else {
      const y = rows.map((r) => Number(r[target])), m = y.reduce((a, b) => a + b, 0) / n, sd = Math.sqrt(y.reduce((a, b) => a + (b - m) ** 2, 0) / n) || 1;
      Y = Float32Array.from(y.map((v) => (v - m) / sd));
      ystats = { mean: m, std: sd };
    }
    const rng = new NA.RNG(seed * 11 + 7), [tr, va] = split(n, rng, 0.8), K = labels ? labels.length : 1;
    return {
      family: kind === 'value' ? 'curve' : 'plane', table: true, kind, input: { kind: 'vector', dim: F }, output: kind === 'value' ? { kind: 'value', k: 1 } : { kind: 'classes', k: K },
      classes: labels ? K : 0, names: labels || [], featNames: feats.map((i) => t.header[i]), feats, target, targetName: t.header[target], stats: { mean, std, y: ystats },
      rowsUsed: n, rowsDropped: t.rows.length - n, ignored: cols.filter((c) => c.index !== target && !c.numeric).map((c) => c.name),
      Xtr: gather(X, F, tr, Float32Array), Ytr: kind === 'value' ? Float32Array.from(tr.map((k) => Y[k])) : Int32Array.from(tr.map((k) => Y[k])), nTr: tr.length,
      Xva: gather(X, F, va, Float32Array), Yva: kind === 'value' ? Float32Array.from(va.map((k) => Y[k])) : Int32Array.from(va.map((k) => Y[k])), nVa: va.length,
    };
  }
  // Two made-up tables to try it without a file: fruit to tell apart (classes), and house prices (a value).
  BT.SAMPLES = {
    fruit: { label: 'Fruit (classes)', make() {
      const rng = new NA.RNG(41), rows = [], kinds = [['apple', 160, 7.5, 7, 0.75], ['orange', 140, 7.2, 7.3, 0.55], ['lemon', 90, 5.5, 7.5, 0.15], ['pear', 170, 6.5, 9.5, 0.35]];
      for (let i = 0; i < 240; i++) {
        const [nm, w, wd, h, col] = kinds[i % 4], f = rng.gauss(1, 0.12);
        rows.push([(w * f * f + rng.gauss(0, 12)).toFixed(0), (wd * f + rng.gauss(0, 0.45)).toFixed(1), (h * f + rng.gauss(0, 0.5)).toFixed(1), NA.clamp(col + rng.gauss(0, 0.1), 0, 1).toFixed(2), nm]);
      }
      return { name: 'fruit.csv', header: ['weight_g', 'width_cm', 'height_cm', 'redness', 'fruit'], rows, delim: ',', hasHeader: true, cut: 0 };
    } },
    houses: { label: 'House prices (a number)', make() {
      const rng = new NA.RNG(43), rows = [];
      for (let i = 0; i < 300; i++) {
        const area = NA.clamp(rng.gauss(110, 35), 35, 260), rooms = NA.clamp(Math.round(area / 28 + rng.gauss(0, 0.8)), 1, 9), age = Math.floor(rng.uniform(0, 80)), km = rng.uniform(0.5, 25);
        const price = 2.1 * area + 9 * rooms - 0.9 * age - 8 * Math.sqrt(km) + 120 + rng.gauss(0, 18);
        rows.push([area.toFixed(0), String(rooms), String(age), km.toFixed(1), price.toFixed(0)]);
      }
      return { name: 'houses.csv', header: ['area_m2', 'rooms', 'age_years', 'km_to_centre', 'price_k'], rows, delim: ',', hasHeader: true, cut: 0 };
    } },
  };

  // ------------------------------------------------------------------ series to forecast
  /* A long series of numbers; each example is a window of the last W values and the target is the next one. The
     first 80% of the windows train and the rest (later in time) validate, as an honest forecast would be judged. */
  const SERIES = {
    waves: (t) => Math.sin((2 * Math.PI * t) / 25) + 0.5 * Math.sin((2 * Math.PI * t) / 9.3 + 1),
    seasonal: (t) => 0.0015 * t + 0.6 * Math.sin((2 * Math.PI * t) / 24) + 0.3 * Math.sin((2 * Math.PI * t) / 7 + 0.5),
    square: (t) => (Math.sin((2 * Math.PI * t) / 30) >= 0 ? 0.8 : -0.8),
  };
  function seriesData(p, s, seed) {
    const rng = new NA.RNG(seed * 17 + 3), N = s.n, W = s.window, v = new Float64Array(N);
    if (p.task === 'chaos') { let x = 0.31 + 0.02 * rng.next(); for (let t = 0; t < N; t++) { v[t] = 2 * x - 1; x = 3.9 * x * (1 - x); } }
    else for (let t = 0; t < N; t++) v[t] = SERIES[p.task](t);
    for (let t = 0; t < N; t++) v[t] += s.noise * rng.gauss();
    let m = 0; for (let t = 0; t < N; t++) m += v[t] / N;
    let top = 1e-9; for (let t = 0; t < N; t++) top = Math.max(top, Math.abs(v[t] - m));
    for (let t = 0; t < N; t++) v[t] = (v[t] - m) / top;
    const nWin = N - W, nTr = Math.round(nWin * 0.8), nVa = nWin - nTr, X = new Float32Array(nWin * W), Y = new Float32Array(nWin);
    for (let i = 0; i < nWin; i++) { for (let j = 0; j < W; j++) X[i * W + j] = v[i + j]; Y[i] = v[i + W]; }
    return { family: 'series', task: p.task, input: { kind: 'series', len: W, dim: 1 }, output: { kind: 'value', k: 1 }, series: Float32Array.from(v), window: W, splitAt: W + nTr,
      Xtr: X.slice(0, nTr * W), Ytr: Y.slice(0, nTr), nTr, Xva: X.slice(nTr * W), Yva: Y.slice(nTr), nVa, inOrder: true };
  }

  // ------------------------------------------------------------------ sequences to classify
  const REVIEW = {
    nouns: ['food', 'movie', 'service', 'room', 'music', 'staff'], good: ['good', 'great', 'lovely', 'fine', 'fun'], bad: ['bad', 'awful', 'boring', 'terrible', 'rude'], adv: ['really', 'very', 'quite'],
  };
  const REVIEW_VOCAB = ['·', 'the', 'was', 'is', 'not', 'but', 'and', '.', ...REVIEW.nouns, ...REVIEW.good, ...REVIEW.bad, ...REVIEW.adv];
  /* Tiny reviews: "the food was not bad", "the room was great but the staff was rude". A clause's adjective decides it,
     "not" flips it, and after "but" the second clause decides the review: bag-of-words models cannot tell. */
  function review(rng) {
    const clause = () => {
      const pos = rng.next() < 0.5, neg = rng.next() < 0.4, w = ['the', REVIEW.nouns[rng.int(REVIEW.nouns.length)], rng.next() < 0.5 ? 'was' : 'is'];
      if (neg) w.push('not');
      if (rng.next() < 0.3) w.push(REVIEW.adv[rng.int(REVIEW.adv.length)]);
      w.push(pos ? REVIEW.good[rng.int(REVIEW.good.length)] : REVIEW.bad[rng.int(REVIEW.bad.length)]);
      return { w, label: pos !== neg ? 1 : 0 };
    };
    const a = clause();
    if (rng.next() < 0.7) { const b = clause(), but = rng.next() < 0.85; return { words: [...a.w, but ? 'but' : 'and', ...b.w, '.'], label: but ? b.label : a.label === b.label ? a.label : -1 }; }
    return { words: [...a.w, '.'], label: a.label };
  }
  const SEQ_TASKS = {
    sentiment: { len: 15, vocab: REVIEW_VOCAB, classes: ['negative', 'positive'] },
    majority: { len: 11, vocab: ['0', '1'], classes: ['more 0s', 'more 1s'] },
    pattern: { len: 8, vocab: [...'0123456789'], classes: ['no 3 7', 'has 3 7'] },
    parity: { len: 10, vocab: ['0', '1'], classes: ['even', 'odd'] },
  };
  function seqClassData(p, s, seed) {
    const def = SEQ_TASKS[p.task], T = def.len, rng = new NA.RNG(seed * 23 + 11), id = Object.fromEntries(def.vocab.map((w, i) => [w, i])), rows = [], seen = new Set();
    let guard = 0;
    while (rows.length < s.n && guard++ < s.n * 30) {
      let ids, label;
      if (p.task === 'sentiment') {
        const r = review(rng);
        if (r.label < 0 || r.words.length > T) continue;
        ids = r.words.map((w) => id[w]); while (ids.length < T) ids.push(0); label = r.label;
      } else if (p.task === 'majority') { ids = NA.range(T).map(() => rng.int(2)); label = ids.reduce((a, b) => a + b, 0) * 2 > T ? 1 : 0; }
      else if (p.task === 'parity') { ids = NA.range(T).map(() => rng.int(2)); label = ids.reduce((a, b) => a + b, 0) % 2; }
      else {
        // half the sequences get "3 7" planted somewhere; a 3 followed by a 7 anywhere counts
        ids = NA.range(T).map(() => rng.int(10));
        if (rng.next() < 0.5) { const at = rng.int(T - 1); ids[at] = 3; ids[at + 1] = 7; }
        label = ids.some((v, i) => v === 3 && ids[i + 1] === 7) ? 1 : 0;
      }
      const key = ids.join(',');
      if (p.task !== 'sentiment' && seen.has(key)) continue;
      seen.add(key); rows.push({ ids, label });
    }
    const n = rows.length, nVa = Math.max(40, Math.round(n * 0.2)), nTr = n - nVa, X = new Int32Array(n * T), Y = new Int32Array(n);
    rows.forEach((r, i) => { X.set(r.ids, i * T); Y[i] = r.label; });
    return { family: 'seqclass', task: p.task, input: { kind: 'tokens', vocab: def.vocab.length, len: T, embed: s.embed }, output: { kind: 'classes', k: def.classes.length },
      vocab: def.vocab, names: def.classes, classes: def.classes.length, pad: p.task === 'sentiment' ? 0 : -1,
      Xtr: X.slice(0, nTr * T), Ytr: Y.slice(0, nTr), nTr, Xva: X.slice(nTr * T), Yva: Y.slice(nTr), nVa };
  }

  // ------------------------------------------------------------------ surfaces to fit
  const SURFACES = {
    hill: (x, y) => Math.exp(-((x - 0.35) ** 2 + (y - 0.3) ** 2) / 0.08) - 0.8 * Math.exp(-((x + 0.4) ** 2 + (y + 0.35) ** 2) / 0.12),
    ripples: (x, y) => { const r = Math.hypot(x, y); return Math.cos(9 * r) * Math.exp(-1.5 * r); },
    saddle: (x, y) => x * x - y * y,
    checker: (x, y) => Math.sin(4.5 * x) * Math.sin(4.5 * y),
  };
  BT.SURFACES = SURFACES;
  function surfaceData(p, s, seed) {
    const rng = new NA.RNG(seed * 29 + 5), n = s.n, f = SURFACES[p.task], X = new Float32Array(n * 2), Y = new Float32Array(n);
    for (let i = 0; i < n; i++) { const x = rng.uniform(-1, 1), y = rng.uniform(-1, 1); X[2 * i] = x; X[2 * i + 1] = y; Y[i] = f(x, y) + s.noise * rng.gauss(); }
    const [tr, va] = split(n, rng, 0.8);
    return { family: 'surface', task: p.task, input: { kind: 'vector', dim: 2 }, output: { kind: 'value', k: 1 }, truth: f,
      Xtr: gather(X, 2, tr, Float32Array), Ytr: Float32Array.from(tr.map((k) => Y[k])), nTr: tr.length, Xva: gather(X, 2, va, Float32Array), Yva: Float32Array.from(va.map((k) => Y[k])), nVa: va.length };
  }

  // ------------------------------------------------------------------ shapes to recognise
  /* 14 × 14 pictures of five shapes drawn as strokes, anywhere, at any size and angle: circle, square, triangle,
     cross and line. Ink is 1 near the outline and fades over a pixel, like the digits. */
  BT.SHAPES = ['circle', 'square', 'triangle', 'cross', 'line'];
  function segDist(px, py, ax, ay, bx, by) {
    const vx = bx - ax, vy = by - ay, t = Math.max(0, Math.min(1, ((px - ax) * vx + (py - ay) * vy) / (vx * vx + vy * vy || 1)));
    return Math.hypot(px - ax - t * vx, py - ay - t * vy);
  }
  function shapeData(s, seed) {
    const rng = new NA.RNG(seed * 41 + 9), n = s.n, G = 14, P = G * G, X = new Float32Array(n * P), Y = new Int32Array(n);
    for (let e = 0; e < n; e++) {
      const k = e % 5, r = rng.uniform(3, 5.2), cx = rng.uniform(r + 0.5, G - r - 1.5), cy = rng.uniform(r + 0.5, G - r - 1.5), a = rng.uniform(0, Math.PI), w = rng.uniform(0.55, 0.95);
      const pt = (ang, rad) => [cx + rad * Math.cos(ang), cy + rad * Math.sin(ang)];
      let segs = null;
      if (k === 1) segs = [0, 1, 2, 3].map((q) => [...pt(a + (q * Math.PI) / 2, r), ...pt(a + ((q + 1) * Math.PI) / 2, r)]);
      else if (k === 2) segs = [0, 1, 2].map((q) => [...pt(a + (q * 2 * Math.PI) / 3, r), ...pt(a + ((q + 1) * 2 * Math.PI) / 3, r)]);
      else if (k === 3) segs = [[...pt(a, r), ...pt(a + Math.PI, r)], [...pt(a + Math.PI / 2, r), ...pt(a + 1.5 * Math.PI, r)]];
      else if (k === 4) segs = [[...pt(a, r), ...pt(a + Math.PI, r)]];
      for (let p = 0; p < P; p++) {
        const px = (p % G) + 0.5, py = Math.floor(p / G) + 0.5;
        const d = segs ? Math.min(...segs.map((q) => segDist(px, py, q[0], q[1], q[2], q[3]))) : Math.abs(Math.hypot(px - cx, py - cy) - r);
        X[e * P + p] = Math.max(0, Math.min(1, 1 - (d - w / 2))) + 0.04 * rng.gauss();
      }
      Y[e] = k;
    }
    const order = NA.range(n); rng.shuffle(order);
    const nTr = Math.round(n * 0.8), tr = order.slice(0, nTr), va = order.slice(nTr);
    return { family: 'digits', task: 'shapes', input: { kind: 'image', c: 1, h: G, w: G }, output: { kind: 'classes', k: 5 }, classes: 5, names: BT.SHAPES.slice(),
      Xtr: gather(X, P, tr, Float32Array), Ytr: Int32Array.from(tr.map((i) => Y[i])), nTr, Xva: gather(X, P, va, Float32Array), Yva: Int32Array.from(va.map((i) => Y[i])), nVa: va.length };
  }

  // ------------------------------------------------------------------ making data
  /* Returns { family, input, output, Xtr, Ytr, nTr, Xva, Yva, nVa, classes, names, vocab, ... }: inputs as
     Float32Array (vectors, images) or Int32Array (tokens), targets as Int32Array (classes, tokens; −1 = no
     target) or Float32Array (values). Deterministic from the seed. */
  BT.make = function (p, s, seed = 1, custom = {}) {
    const fam = p.family;
    if (fam === 'plane') return p.custom ? drawnPoints(custom.points || [], seed) : planeData(p, s, seed);
    if (fam === 'curve') return curveData(p, s, seed);
    if (fam === 'digits') return p.task === 'shapes' ? shapeData(s, seed) : digitData(s, seed);
    if (fam === 'series') return seriesData(p, s, seed);
    if (fam === 'seqclass') return seqClassData(p, s, seed);
    if (fam === 'surface') return surfaceData(p, s, seed);
    if (fam === 'table') return tableData(custom.table, s, seed);
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
