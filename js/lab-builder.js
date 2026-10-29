/* lab-builder.js — build a model from scratch. Pick a problem (or make your own: draw points, type a function,
   write sentences), drag parts from the palette into a stack between the input and the output (or click them),
   set the loss and the training settings, and train. Each training run is recorded with what changed since the
   one before, so you can see what a change did. The engine is js/builder-engine.js; problems are in
   js/builder-tasks.js, pictures of the results in js/builder-viz.js and the PyTorch version in js/builder-py.js. */
(function (NA) {
  'use strict';
  const { el, clear, clamp, fmt, fmtPct, fmtInt, theme } = NA;
  const H = NA.labHost, UI = NA.labUI, BT = NA.builderTasks, BV = NA.builderViz;
  const ACTS = NA.engine.ACTIVATIONS, ACT_ORDER = NA.engine.ACTIVATION_ORDER;

  // ------------------------------------------------------------------ the parts
  const GROUP_TONE = { Layers: 's1', Sequences: 's7', Images: 's4', Reshape: 's6', Combine: 's3', 'Classic ML': 's5' };
  const PARTS = [
    { type: 'dense', group: 'Layers', label: 'Dense', name: 'Dense layer', make: () => ({ type: 'dense', units: 16, act: 'relu' }),
      hint: 'Neurons that each weigh every input, add a bias and bend the sum with an activation.' },
    { type: 'neuron', group: 'Layers', label: 'Neuron', name: 'Neuron', make: () => ({ type: 'dense', units: 1, act: 'relu' }),
      hint: 'Drop it on a dense layer to add one neuron, or between blocks for a layer of one.' },
    { type: 'gated', group: 'Layers', label: 'Gated', name: 'Gated layer (GLU)', make: () => ({ type: 'gated', units: 16, gate: 'swish' }),
      hint: 'Two projections, one gating the other: the SwiGLU layer inside many large language models.' },
    { type: 'activation', group: 'Layers', label: 'Activation', name: 'Activation', make: () => ({ type: 'activation', act: 'tanh' }),
      hint: 'Bends every number with one function (ReLU, tanh, …).' },
    { type: 'dropout', group: 'Layers', label: 'Dropout', name: 'Dropout', make: () => ({ type: 'dropout', p: 0.2 }),
      hint: 'Randomly switches off some numbers while training, against overfitting.' },
    { type: 'noise', group: 'Layers', label: 'Noise', name: 'Gaussian noise', make: () => ({ type: 'noise', sigma: 0.1 }),
      hint: 'Jitters every number a little while training, so the model cannot rely on exact values.' },
    { type: 'layernorm', group: 'Layers', label: 'LayerNorm', name: 'Layer norm', make: () => ({ type: 'layernorm' }),
      hint: 'Rescales each example to mean 0 and spread 1: keeps deep models stable.' },
    { type: 'batchnorm', group: 'Layers', label: 'BatchNorm', name: 'Batch norm', make: () => ({ type: 'batchnorm' }),
      hint: 'Rescales each feature over the batch to mean 0 and spread 1: the trick that made deep CNNs trainable.' },
    { type: 'fourier', group: 'Layers', label: 'Fourier', name: 'Fourier features', make: () => ({ type: 'fourier', features: 16, scale: 1 }),
      hint: 'Adds sines and cosines of the inputs at random frequencies: sharp, wiggly functions become easy.' },
    { type: 'attention', group: 'Sequences', label: 'Attention', name: 'Self-attention', make: () => ({ type: 'attention', heads: 2, causal: true, residual: true }),
      hint: 'Lets every position look at the others. Needs a sequence.' },
    { type: 'transformer', group: 'Sequences', label: 'Transformer', name: 'Transformer block', make: () => ({ type: 'transformer', heads: 2, mlp: 4, causal: true }),
      hint: 'Attention and an MLP, each with layer norm and a residual: the block GPT is made of.' },
    { type: 'rnn', group: 'Sequences', label: 'LSTM · GRU', name: 'Recurrent layer', make: () => ({ type: 'rnn', cell: 'lstm', units: 32 }),
      hint: 'Reads the sequence one step at a time and carries a memory along: LSTM, GRU or a plain RNN.' },
    { type: 'conv1d', group: 'Sequences', label: 'Conv 1-D', name: '1-D convolution', make: () => ({ type: 'conv1d', filters: 16, k: 3, dilation: 1, causal: false, act: 'relu', residual: false }),
      hint: 'Filters slid along the sequence, each looking for a short pattern of neighbours.' },
    { type: 'position', group: 'Sequences', label: 'Positions', name: 'Positions', make: () => ({ type: 'position', mode: 'learned' }),
      hint: 'Adds a vector for each position, so attention can tell first from last.' },
    { type: 'seqpool', group: 'Sequences', label: 'Seq. pool', name: 'Sequence pool', make: () => ({ type: 'seqpool', mode: 'last' }),
      hint: 'Turns a sequence into one vector: its last step, its largest values or the average.' },
    { type: 'conv', group: 'Images', label: 'Conv', name: 'Convolution', make: () => ({ type: 'conv', filters: 8, k: 3, stride: 1, act: 'relu' }),
      hint: 'Small learned filters slid over an image: each finds one local pattern.' },
    { type: 'pool', group: 'Images', label: 'Max pool', name: 'Max pool', make: () => ({ type: 'pool', size: 2 }),
      hint: 'Keeps the largest value of each 2 × 2 patch: halves the picture.' },
    { type: 'avgpool', group: 'Images', label: 'Avg pool', name: 'Average pool', make: () => ({ type: 'avgpool', size: 2 }),
      hint: 'Averages each 2 × 2 patch: halves the picture, smoothly.' },
    { type: 'patches', group: 'Images', label: 'Patches', name: 'Patches', make: () => ({ type: 'patches', size: 2, dim: 32 }),
      hint: 'Cuts the picture into small squares and turns each into a token: how a vision transformer sees.' },
    { type: 'flatten', group: 'Reshape', label: 'Flatten', name: 'Flatten', make: () => ({ type: 'flatten' }),
      hint: 'Lines all the numbers up into one vector, for dense layers.' },
    { type: 'gap', group: 'Reshape', label: 'Global pool', name: 'Global average pool', make: () => ({ type: 'gap' }),
      hint: 'Averages each channel over every position: one vector, whatever the size.' },
    { type: 'parallel', group: 'Combine', label: 'Parallel', name: 'Parallel branches', make: () => ({ type: 'parallel', merge: 'concat', auto: true, branches: [[], []] }),
      hint: 'Two to four branches side by side on the same input, joined or added at the end: Inception modules, wide-and-deep models, mixtures.' },
    { type: 'linear', group: 'Classic ML', label: 'Linear model', name: 'Linear model', make: () => ({ type: 'linear', lambda: 0 }), classic: true,
      hint: 'Linear regression for numbers, logistic regression for classes: one weight per input, the classic baseline.' },
    { type: 'knn', group: 'Classic ML', label: 'k-NN', name: 'k-nearest neighbours', make: () => ({ type: 'knn', k: 5, weights: 'uniform' }), classic: true,
      hint: 'No training: each answer is a vote of the k most similar training examples.' },
    { type: 'tree', group: 'Classic ML', label: 'Tree', name: 'Decision tree', make: () => ({ type: 'tree', depth: 6, minLeaf: 2 }), classic: true,
      hint: 'A flowchart of yes-or-no questions about the inputs, grown one level at a time.' },
    { type: 'forest', group: 'Classic ML', label: 'Forest', name: 'Random forest', make: () => ({ type: 'forest', trees: 50, depth: 8, features: 'sqrt', minLeaf: 1 }), classic: true,
      hint: 'Many deep trees on random samples of the data and the inputs, voting together.' },
    { type: 'boost', group: 'Classic ML', label: 'Boosting', name: 'Gradient boosting (XGBoost)', make: () => ({ type: 'boost', rounds: 100, eta: 0.3, depth: 4, lambda: 1, gamma: 0, subsample: 1, colsample: 1 }), classic: true,
      hint: 'Small trees added one after another, each fixing what the rest still get wrong: the favourite for tables.' },
  ];
  const isClassic = (type) => !!(PART[type] && PART[type].classic);
  const PART = Object.fromEntries(PARTS.map((p) => [p.type, p]));
  const toneOf = (t) => (t === 'input' ? 's3' : t === 'output' ? 's2' : GROUP_TONE[(PART[t] || PART.dense).group]);
  // how a parallel block's branches meet
  const MERGE_TEXT = { concat: 'joined side by side', add: 'added number by number', mul: 'multiplied number by number', mean: 'averaged number by number' };
  const MERGE_WORD = { concat: 'joined', add: 'added', mul: 'multiplied', mean: 'averaged' }, MERGE_SIGN = { concat: '‖', add: '+', mul: '×', mean: 'x̄' };
  const MERGE_NOTE = {
    concat: 'The outputs are laid side by side: a vector’s numbers one after another, each position’s features of a sequence, or the channels of a picture. Every branch keeps its own numbers, so the next layer can weigh them separately. Branches may give different sizes, but the same kind of data, with the same positions or picture size.',
    add: 'The outputs are added number by number, so every branch must give the same shape. An empty branch adds the input itself: a residual connection around the other branches, as in a ResNet.',
    mul: 'The outputs are multiplied number by number. With a sigmoid branch, one branch becomes a gate that opens or shuts the other’s numbers, example by example: the idea behind the gates of an LSTM and of gated layers.',
    mean: 'The outputs are averaged number by number, like a small ensemble of models that vote. Every branch must give the same shape.',
  };
  // what each block does, for the inspector
  const ABOUT = {
    dense: 'Every neuron computes a weighted sum of all its inputs, adds a bias, and bends the result with its activation function. More neurons let the layer represent more patterns at once; more layers let later patterns be built out of earlier ones.',
    activation: 'Applies one function to every number. Without activations, any stack of dense layers collapses into a single linear one: the bends are what let a network draw curved boundaries.',
    dropout: 'While training, each step switches off a random fraction of these numbers (and scales up the rest), so the network cannot lean on any single neuron. A common defence against overfitting; it does nothing when the model is used.',
    layernorm: 'Rescales each example’s numbers to mean 0 and spread 1, then applies a learned scale and shift. It keeps signals in a healthy range as they pass through many layers, which is why every transformer uses it.',
    attention: 'Self-attention lets every position look at the others. Each position makes a query, compares it with every position’s key (a dot product), turns the scores into weights with a softmax, and takes the weighted average of their values. With several heads, it can look for several things at once. Causal: a position only sees the ones before it, which a model that writes one token at a time needs.',
    transformer: 'The block GPT is made of: layer norm, self-attention, and the input added back (a residual); then layer norm, a two-layer MLP with a GELU, and the input added back again. Stack a few and a model can follow patterns across a whole sequence.',
    conv: 'A convolution slides small learned filters (3 × 3 or 5 × 5) over the image. Each filter responds to one local pattern, such as an edge or a corner, wherever it appears, so it needs far fewer weights than a dense layer on all the pixels.',
    pool: 'Keeps the largest value in each patch. The picture shrinks (and later filters see a wider area), and small shifts of a feature stop mattering.',
    flatten: 'Lines up all the numbers of an image or a sequence into one long vector, so dense layers (and the output) can use them. It keeps every number but forgets which were neighbours.',
    gap: 'Averages each channel over every position, so a whole image or sequence becomes one short vector. Unlike flatten, it works for any size and has no position information left.',
    parallel: 'Runs two to four branches side by side on the same input, then joins what they give (or adds, multiplies or averages it). Each branch can look at the data its own way at once: big and small filters, a recurrent and a convolutional reader, a deep path and a wide one.',
    gated: 'Two projections of the same input: one carries the values, the other (bent by a gate function) decides how much of each value passes. Gating lets the layer switch features on and off per example.',
    noise: 'While training, adds a small random jitter to every number. The model learns to ignore tiny differences, which smooths what it learns. Nothing happens when the model is used.',
    batchnorm: 'Rescales each feature to mean 0 and spread 1 using the statistics of the current batch, then applies a learned scale and shift. Training gets faster and steadier; when the model is used, averages from training replace the batch statistics.',
    fourier: 'Keeps the inputs and adds sines and cosines of them at random fixed frequencies. A network that has these can build sharp, wiggly shapes directly, instead of bending many neurons into them.',
    rnn: 'Reads a sequence one step at a time, updating a hidden state that carries what it has seen so far. An LSTM adds gates and a separate memory cell so information can last many steps; a GRU does the same with fewer gates.',
    conv1d: 'Slides small learned filters along the sequence. Each output position mixes a window of neighbouring positions, so the layer finds short local patterns wherever they occur. Causal windows only look back; dilation spreads the window out.',
    position: 'Attention treats a sequence as an unordered set. Adding a different vector at each position (learned, or fixed sine waves) lets the next layers tell positions apart.',
    seqpool: 'Squeezes a sequence into one vector for a class or value output: the last step (what a recurrent layer has gathered), the largest value of each feature (did a pattern appear anywhere?) or the average.',
    avgpool: 'Averages each small patch of the picture, halving its size. Gentler than max pooling: every pixel counts a little.',
    patches: 'Cuts the picture into small squares, flattens each, and maps it to a vector with a learned position: the picture becomes a sequence of tokens that attention can work on. This is how a vision transformer starts.',
    linear: 'The classic baseline. For numbers it is least-squares linear regression, solved exactly in one step; for classes it is logistic regression, trained by gradient steps. One weight per input: it can only draw straight boundaries and flat surfaces.',
    knn: 'Remembers every training example. To answer, it finds the k most similar examples (smallest distance) and lets them vote, or averages their values. There is no training at all, and it can fit any shape, given enough data.',
    tree: 'Learns a flowchart of yes-or-no questions such as "is x₁ ≤ 0.3?", choosing at each step the question that best separates the answers. Its boundaries are made of straight, axis-aligned steps.',
    forest: 'Grows many deep trees, each on a random resample of the data and choosing among a random few inputs at each question, then averages their votes. One tree overfits; the average of many different ones is steady.',
    boost: 'Adds small trees one at a time, each trained on what the trees so far still get wrong (the gradient of the loss). Shrinkage (η) and penalties (λ, γ) keep each tree modest. This is how XGBoost and LightGBM work, and why they win on tables.',
  };

  // small line icons (24 × 24), drawn with the current text colour
  const ICON = {
    input: ['M4 12h9', 'M10 8l4 4-4 4', 'M17 5v14'],
    output: ['M7 5v14', 'M10 12h9', 'M16 8l4 4-4 4'],
    dense: ['M5 6h0', 'M5 12h0', 'M5 18h0', 'M19 9h0', 'M19 15h0', 'M5 6L19 9', 'M5 6L19 15', 'M5 12L19 9', 'M5 12L19 15', 'M5 18L19 9', 'M5 18L19 15'],
    neuron: ['C12 12 5.5', 'M12 9v6', 'M9 12h6'],
    activation: ['M3 18h6c3 0 3-12 6-12h6'],
    dropout: ['C6 8 1.6', 'C12 8 1.6', 'C18 8 1.6', 'C6 16 1.6', 'C18 16 1.6', 'M10 14l4 4', 'M14 14l-4 4'],
    layernorm: ['M4 19h16', 'M6 19c2-9 4-14 6-14s4 5 6 14'],
    attention: ['M4 18c3-8 13-8 16 0', 'M8 18c2-4 6-4 8 0', 'C12 8 2'],
    transformer: ['M5 4h14v6H5z', 'M5 14h14v6H5z', 'M12 10v4'],
    conv: ['M4 4h16v16H4z', 'M4 9.3h16', 'M4 14.6h16', 'M9.3 4v16', 'M14.6 4v16', 'M4 4h10.6v10.6H4z'],
    pool: ['M4 4h16v16H4z', 'M12 4v16', 'M4 12h16', 'M7 7l2 2', 'M15 7l2 2'],
    flatten: ['M4 6h16', 'M4 10h16', 'M4 14h16', 'M4 18h16'],
    gap: ['M4 12h16', 'M8 6l4 6-4 6', 'M14 8h4', 'M14 16h4'],
    gated: ['M4 8h7', 'M4 16h7', 'M11 8c3 0 3 4 5 4', 'M11 16c3 0 3-4 5-4', 'C18 12 2', 'M20 12h1'],
    noise: ['M3 12l2-3 2 5 2-6 2 7 2-5 2 4 2-3 2 2 2-1'],
    batchnorm: ['M5 19V9', 'M10 19V5', 'M15 19v-7', 'M20 19V8', 'M3 12h18'],
    fourier: ['M3 12c2-6 4-6 6 0s4 6 6 0 4-6 6 0', 'M3 12c1-3 2-3 3 0s2 3 3 0 2-3 3 0 2 3 3 0 2-3 3 0'],
    rnn: ['M4 14h4v4H4z', 'M10 14h4v4h-4z', 'M16 14h4v4h-4z', 'M8 16h2', 'M14 16h2', 'M6 14V8h12v6', 'M16 10l2-2 2 2'],
    conv1d: ['M3 12h18', 'M5 9v6', 'M9 9v6', 'M13 9v6', 'M17 9v6', 'M4 7h10v10H4z'],
    position: ['M4 18V6', 'M4 18h16', 'M7 15c2-6 4-6 6 0', 'M11 12c2-4 4-4 6 0'],
    seqpool: ['M3 6h4', 'M3 12h4', 'M3 18h4', 'M8 6l6 6-6 6', 'M15 12h6'],
    avgpool: ['M4 4h16v16H4z', 'M12 4v16', 'M4 12h16', 'M6 8h4', 'M14 16h4'],
    patches: ['M4 4h7v7H4z', 'M13 4h7v7h-7z', 'M4 13h7v7H4z', 'M13 13h7v7h-7z'],
    linear: ['M4 19L20 5', 'C7 15 1', 'C10 13 1', 'C14 10 1', 'C17 7 1'],
    knn: ['C12 12 6', 'C12 12 1.4', 'C9 9 1', 'C15 10 1', 'C11 16 1', 'C19 5 1', 'C4 19 1'],
    tree: ['M12 4v5', 'M6 15V9h12v6', 'M4 15h4v4H4z', 'M16 15h4v4h-4z', 'C12 4 1.5'],
    forest: ['M6 4v5', 'M3 9h6l-3 6z', 'M18 4v5', 'M15 9h6l-3 6z', 'M12 8v5', 'M9 13h6l-3 6z'],
    boost: ['M4 18h3v-3H4z', 'M10 18h3v-6h-3z', 'M16 18h3V7h-3z', 'M5 12l6-5 6-3'],
    part: ['M4 8l8-4 8 4v8l-8 4-8-4z', 'M4 8l8 4 8-4', 'M12 12v8'],
    parallel: ['M2 12h4', 'M6 12l4-6h5', 'M6 12l4 6h5', 'M15 6l4 6-4 6', 'M19 12h3'],
  };
  function icon(type) {
    const NS = 'http://www.w3.org/2000/svg', s = document.createElementNS(NS, 'svg');
    s.setAttribute('viewBox', '0 0 24 24'); s.setAttribute('aria-hidden', 'true'); s.setAttribute('class', 'ico');
    for (const d of ICON[type] || ICON.dense) {
      let e;
      if (d.startsWith('C')) { const [x, y, r] = d.slice(1).trim().split(/\s+/).map(Number); e = document.createElementNS(NS, 'circle'); e.setAttribute('cx', x); e.setAttribute('cy', y); e.setAttribute('r', r); }
      else { e = document.createElementNS(NS, 'path'); e.setAttribute('d', d); }
      s.append(e);
    }
    return s;
  }

  // ------------------------------------------------------------------ the lab
  // "Tap" on a touch screen, "Click" with a mouse
  const touch = () => (window.matchMedia && matchMedia('(pointer: coarse)').matches) || (NA.mobile && NA.mobile.mobile);
  const Click = () => (touch() ? 'Tap' : 'Click'), click = () => (touch() ? 'tap' : 'click');
  const BL = H.transport({
    id: 'builder', screens: { network: 'Build', data: 'Parts', insp: 'Settings' },
    labels: { play: 'Train', title: 'Train the model you built, or pause (Space)', step: 'One training step (S)', reset: 'Fresh weights: start this setup again (R)' },
    budget: 14, runs: [], undoStack: [], redoStack: [], sel: null, centerView: 'both', tabId: 'block', nextId: 1,
  });
  const DEFAULT_TRAIN = BT.DEFAULT_TRAIN;
  BL.setup = { problem: 'spiral', ps: {}, blocks: [], loss: { id: 'xent', smooth: 0, delta: 0.2 }, train: Object.assign({}, DEFAULT_TRAIN), custom: { points: [] } };

  BL.problem = function () { return BT.byId[this.setup.problem]; };
  BL.ps = function () { const id = this.setup.problem; return this.setup.ps[id] || (this.setup.ps[id] = BT.defaults(BT.byId[id])); };
  BL.family = function () { return this.problem().family; };
  const bid = () => 'b' + BL.nextId++;
  // a parallel block's branches hold blocks with ids of their own: fresh ones whenever a block is made or copied
  const freshIds = (b) => { if (b && Array.isArray(b.branches)) b.branches = b.branches.map((list) => list.map((q) => Object.assign({}, q, { id: bid() }))); return b; };
  const withIds = (blocks) => blocks.map((b) => freshIds(Object.assign({ id: bid() }, b)));
  // a block anywhere in the model: at the top, or in a branch of a parallel block (its owner)
  function findBlock(S, id) {
    for (let i = 0; i < S.blocks.length; i++) {
      const b = S.blocks[i];
      if (b.id === id) return { b, i, owner: null, br: -1, k: -1 };
      if (Array.isArray(b.branches)) for (let br = 0; br < b.branches.length; br++) { const k = b.branches[br].findIndex((q) => q.id === id); if (k >= 0) return { b: b.branches[br][k], i, owner: b, br, k }; }
    }
    return null;
  }
  const LETTERS = 'ABCD';
  /* Parts of your own: blocks boxed together under a name, like a function, and added again as copies. The model stays
     a flat list of blocks (the engine, the checks and every view work on it as before); the blocks of one copy carry
     the same grp (the copy) and part (the definition in setup.parts), and sit next to each other. A copy has its own
     weights; changing a block inside one copy changes the definition and every other copy too. */
  const gid = () => 'g' + BL.nextId++, pid = () => 'p' + BL.nextId++;
  const cleanSpec = (b) => { const c = JSON.parse(JSON.stringify(b)); delete c.id; delete c.grp; delete c.part; if (Array.isArray(c.branches)) c.branches.forEach((list) => list.forEach((q) => { delete q.id; })); return c; };
  const cleanName = (s) => String(s || '').replace(/[^\p{L}\p{N} _\-]/gu, '').replace(/\s+/g, ' ').trim().slice(0, 32);
  const runsOf = (s) => BT.groupsOf(s.blocks);
  const runOf = (s, b) => (b && b.grp ? runsOf(s).find((g) => g.grp === b.grp) : null);
  const defOf = (s, id) => (s.parts || []).find((p) => p.id === id);
  function syncPart(s, q) {
    const run = runOf(s, q), def = defOf(s, q.part);
    if (!run || !def) return;
    const k = s.blocks.indexOf(q) - run.from, spec = cleanSpec(q);
    def.blocks[k] = spec;
    for (const g of runsOf(s)) {
      if (g.part !== q.part || g.grp === q.grp) continue;
      const t = s.blocks[g.from + k];
      if (!t) continue;
      for (const key of Object.keys(t)) if (key !== 'id' && key !== 'grp' && key !== 'part') delete t[key];
      freshIds(Object.assign(t, JSON.parse(JSON.stringify(spec))));
    }
  }
  // the k-th block of a part: removed, moved or inserted in the definition and in every copy at once
  function editPart(s, partId, fn) {
    const def = defOf(s, partId);
    if (!def) return;
    const runs = runsOf(s).filter((g) => g.part === partId).reverse(); // from the last copy, so indices stay valid
    fn(def.blocks, null);
    for (const g of runs) fn(s.blocks, g);
    if (!def.blocks.length) s.parts = s.parts.filter((p) => p !== def);
  }

  // The data (and with it the input and output shapes) for the current problem; rebuilt only when it changes.
  BL.dataFor = function () {
    // a table is known by the stamp it got when it was chosen (comparing all its rows would cost more than the data)
    const S = this.setup, p = this.problem(), key = JSON.stringify([S.problem, this.ps(), S.train.seed, p.id === 'draw' ? S.custom.points : p.family === 'table' && S.custom.table ? S.custom.table.stamp || S.custom.table.name : 0]);
    if (this._data && this._data.key === key) return this._data.d;
    const d = BT.make(this.problem(), this.ps(), S.train.seed, S.custom);
    this._data = { key, d };
    return d;
  };
  BL.recheck = function () {
    const d = this.dataFor();
    this.check = d.error ? { ok: false, shapes: [], problem: { reason: d.error } } : this.isClassicSetup() ? NA.cml.check(this.setup.blocks, d) : NA.bx.check(this.setup.blocks, d.input, d.output);
    return this.check;
  };
  // a classic model (linear, k-NN, tree, forest, boosting) instead of a network
  BL.isClassicSetup = function () { return this.setup.blocks.some((b) => isClassic(b.type)); };

  // every change goes through edit(): undo history, a fresh check, a new run next time Train is pressed
  BL.snapshot = function () { return JSON.stringify(this.setup); };
  BL.edit = function (fn, o = {}) {
    const before = this.snapshot();
    const apply = () => { fn(this.setup); this.afterEdit(before, o); };
    if (o.flip) this.flip(apply); else apply();
  };
  BL.afterEdit = function (before, o = {}) {
    if (this.snapshot() === before) return;
    if (!o.noUndo) { this.undoStack.push(before); if (this.undoStack.length > 80) this.undoStack.shift(); this.redoStack = []; }
    this.changed();
  };
  BL.changed = function () {
    const run = this.run;
    if (run && !this.stale && sameModel(run.cfg, this.setup)) {
      // only training settings changed (learning rate, optimizer, batch, steps, schedule…): the run carries on with
      // them, and the change is written on its timeline
      const now = JSON.parse(this.snapshot()), what = diffSetups(run.cfg, now).filter((w) => w !== 'same setup, fresh weights');
      if (what.length) (run.events || (run.events = [])).push({ step: run.step, what });
      run.cfg = now;
      const T = this.setup.train;
      run.limit = T.steps ? Math.max(run.step + (run.done ? 0 : 1), (run.sched0 || 0) + T.steps) : 0;
      if (run.done && run.status === 'done' && (!run.limit || run.step < run.limit)) { run.done = false; run.status = 'paused'; }
      if (this.training && what.length) NA.ui.toast.show(`From step ${fmtInt(run.step)}: ${what.join(', ')}. Run ${run.n} carries on.`);
    } else {
      this.stale = !!run; // the current run used the old setup
      // a change to the model, problem, loss or start pauses it: the next Train starts a new run, so every run has one model
      if (this.training) { this.pause(); NA.ui.toast.show('Paused for your change: press Train to start a new run with it.'); }
    }
    this.save();
    this.archStale = true;
    this.recheck();
    const T = this.setup.train;
    if (this.hyperRoot && (this.isClassicSetup() ? 'classic' : JSON.stringify([T.opt, T.lr, T.batch, T.steps])) !== this._hyperKey) this.buildHyper();
    this.renderWorkbench(); this.renderLeft(); this.inspDirty = true; this.resultsDirty = true; this.pyDirty = true;
    if (!this.run || this.run.done) this.showDataOnly();
    this.syncUndo();
    this.renderChallenge();
    NA.app.emit('labarch');
    NA.app.emit('training'); // the play button says Continue or Train
  };
  // The setup is kept in this browser, so a model you were building is still there next time.
  BL.save = function () { clearTimeout(this._saveT); this._saveT = setTimeout(() => NA.store.set('builder', this.setup), 300); };
  BL.restore = function () {
    const S = NA.store.get('builder', null);
    if (!usable(S)) return false;
    this.setup = tidy(S);
    return true;
  };
  // A stored or shared setup is usable when its problem, blocks, loss and training settings are all there.
  function usable(S) {
    if (!S || typeof S !== 'object' || !BT.byId[S.problem] || !Array.isArray(S.blocks) || !S.loss || typeof S.loss !== 'object' || !S.train || typeof S.train !== 'object' || !S.ps || typeof S.ps !== 'object') return false;
    // a parallel block: 2 to 4 branches of up to 8 blocks each, none of them another parallel block or a classic model
    const okBlock = (b, nested) => !!b && typeof b === 'object' && (NA.bx.BLOCKS[b.type] ? b.type !== 'parallel' || (!nested && Array.isArray(b.branches) && b.branches.length >= 2 && b.branches.length <= 4 && b.branches.every((l) => Array.isArray(l) && l.length <= 8 && l.every((q) => okBlock(q, true)))) : !nested && isClassic(b.type));
    if (S.parts !== undefined && !(Array.isArray(S.parts) && S.parts.length <= 16 && S.parts.every((p) => p && typeof p.name === 'string' && p.name.length <= 40 && Array.isArray(p.blocks) && p.blocks.length && p.blocks.length <= 12 && p.blocks.every((b) => okBlock(b, false) && !isClassic(b.type))))) return false;
    return S.blocks.length <= 40 && S.blocks.every((b) => okBlock(b, false));
  }
  function tidy(S) {
    S.custom = S.custom && Array.isArray(S.custom.points) ? S.custom : { points: [] };
    S.train = Object.assign({}, DEFAULT_TRAIN, S.train);
    // fresh ids for blocks, copies and parts, so none can clash with ones made later
    const pmap = {}, gmap = {};
    S.parts = (S.parts || []).map((p) => { const id = pid(); pmap[p.id] = id; return { id, name: cleanName(p.name) || 'My part', blocks: p.blocks.map(cleanSpec) }; });
    // a copy is its part's blocks, in order and side by side: anything else becomes ordinary blocks again
    let prev = null, cur = null;
    S.blocks.forEach((b) => {
      b.id = bid(); freshIds(b);
      if (b.grp && pmap[b.part]) {
        if (!(prev && prev.grp === b.grp && prev.part === b.part)) cur = gid();
        prev = { grp: b.grp, part: b.part };
        b.grp = cur; b.part = pmap[b.part];
      } else { prev = null; delete b.grp; delete b.part; }
    });
    for (const r of BT.groupsOf(S.blocks)) {
      const def = S.parts.find((q) => q.id === r.part), inner = S.blocks.slice(r.from, r.to);
      if (!def || def.blocks.length !== inner.length || inner.some((b, k) => b.type !== def.blocks[k].type)) inner.forEach((b) => { delete b.grp; delete b.part; });
    }
    return S;
  }

  /* Share links (share.js) carry the whole setup. One from someone else is held to the ranges the controls offer,
     so a made-up link cannot ask for a network too big to build; the setup it replaces stays one Undo away. */
  BL.shareState = function () { const S = JSON.parse(this.snapshot()); S.blocks.forEach((b) => { delete b.id; }); if (S.custom) delete S.custom.table; return S; };
  BL.shareAbout = function () {
    return 'Opens the model builder with your problem, blocks, loss and training settings; the one who opens it presses Train to train it. It replaces the builder setup in their browser, and Undo brings that back.'
      + (this.setup.problem === 'csv' ? ' Your table itself is not in the link: they choose the same CSV file to train on it.' : '');
  };
  BL.shareNote = 'Your own setup is one Undo away.';
  BL.loadShare = function (S) {
    if (!usable(S)) return false;
    const num = (v, lo, hi) => typeof v === 'number' && isFinite(v) && v >= lo && v <= hi;
    const every = (list) => list.flatMap((b) => [b, ...(Array.isArray(b.branches) ? b.branches.flat() : [])]);
    for (const b of every([...S.blocks, ...(S.parts || []).flatMap((p) => p.blocks)])) for (const k of Object.keys(b)) if (typeof b[k] === 'number' && !num(b[k], -512, 512)) return false; // your parts and branches too
    for (const ps of Object.values(S.ps)) if (ps && typeof ps === 'object') for (const k of Object.keys(ps)) if (typeof ps[k] === 'number' && !num(ps[k], -1e4, 1e4)) return false;
    const T = Object.assign({}, DEFAULT_TRAIN, S.train);
    if (!num(T.lr, 1e-6, 10) || !num(T.batch, 1, 1024) || (T.steps !== undefined && !num(T.steps, 0, 2e5)) || !OPT_LABEL[T.opt] || !SCHED_LABEL[T.sched || 'const']) return false;
    if (S.custom && Array.isArray(S.custom.points) && S.custom.points.length > 3000) return false;
    const next = tidy(S), before = this.snapshot();
    try { this.edit((cur) => { for (const k of Object.keys(cur)) delete cur[k]; Object.assign(cur, next); }); }
    catch (e) { this.setup = JSON.parse(before); this.undoStack.pop(); this.changed(); return false; }
    return true;
  };
  BL.undo = function () { if (!this.undoStack.length) return; this.redoStack.push(this.snapshot()); this.setup = JSON.parse(this.undoStack.pop()); this.changed(); NA.ui.toast.show('Undone'); };
  BL.redo = function () { if (!this.redoStack.length) return; this.undoStack.push(this.snapshot()); this.setup = JSON.parse(this.redoStack.pop()); this.changed(); NA.ui.toast.show('Redone'); };
  BL.syncUndo = function () { if (this.undoBtn) { this.undoBtn.disabled = !this.undoStack.length; this.redoBtn.disabled = !this.redoStack.length; } };

  // ------------------------------------------------------------------ runs and training
  const RUN_COLOURS = ['s1', 's2', 's4', 's7', 's3', 's5', 's6', 's8'];
  // steps per press of Train; 0 trains until you pause
  const STEP_CHOICES = [[300, '300'], [1000, '1,000'], [2000, '2,000'], [3000, '3,000'], [5000, '5,000'], [10000, '10,000'], [0, '∞ until paused']];
  BL.newRun = function () {
    const d = this.dataFor(), chk = this.recheck();
    if (d.error) { NA.ui.toast.show(d.error); return false; }
    if (d.tooFew) { NA.ui.toast.show('Place at least 8 points of two or more classes first (click on the picture).'); return false; }
    if (!chk.ok) { const p = chk.problem || (chk.shapes.find((s) => s.problem) || {}).problem; NA.ui.toast.show(`The model does not fit together yet: ${p ? p.reason : 'see the red block'}`); return false; }
    const T = this.setup.train, cfg = JSON.parse(this.snapshot());
    let model;
    const classic = this.isClassicSetup();
    try { model = classic ? NA.cml.build(this.setup.blocks, d, { seed: T.seed }) : NA.bx.build(this.setup.blocks, d.input, d.output, { seed: T.seed, init: T.init }); }
    catch (e) { NA.ui.toast.show(e.message); return false; }
    const prev = this.runs[this.runs.length - 1], n = (prev ? prev.n : 0) + 1;
    const run = { n, colour: RUN_COLOURS[(n - 1) % RUN_COLOURS.length], cfg, changes: prev ? diffSetups(prev.cfg, cfg) : ['first run'], curve: [], step: 0, ema: null, started: performance.now(), seconds: 0, params: model.paramCount, status: 'running', show: true, family: d.family, limit: classic ? 0 : T.steps || 0, sched0: 0, events: [], classic };
    this.runs.push(run);
    if (this.runs.length > 12) this.runs.splice(0, this.runs.length - 12);
    this.run = run; this.model = model; this.data = d; this.stale = false; this.step = 0;
    this.order = null; this.epochPos = 0; this.rng = new NA.RNG(T.seed * 977 + n);
    if (this.seqView) this.seqView.cont = null; // written by the new model once it has trained a little
    this.vizVersion = (this.vizVersion || 0) + 1; this._region = null; this.confusion = null; this.lastEval = 0; this.evalMs = 0;
    this.evalEvery = classic ? Math.max(1, Math.round(model.target / 60)) : T.steps ? Math.max(1, Math.round(T.steps / 120)) : 25; // about 120 points on the curves
    this.evaluate(true);
    this.inspDirty = true; this.resultsDirty = true; this.renderWorkbench();
    return true;
  };
  // Train: a run that finished (and has not been edited since) carries on from where it stopped; otherwise a new run.
  BL.play = function () {
    if (this.canContinue()) this.extendRun();
    else if (!this.run || this.stale || this.run.done) { if (!this.newRun()) return; }
    this.training = true; this.run.status = 'running'; NA.app.emit('training');
  };
  BL.canContinue = function () { const r = this.run; return !!r && !this.stale && r.done && r.status === 'done' && !!this.model && (!this.model.classic || this.model.canExtend()); };
  BL.extendRun = function () {
    const run = this.run, T = this.setup.train;
    if (this.model.classic) {
      const m = this.model;
      m.extend(); run.done = false;
      (run.events || (run.events = [])).push({ step: run.step, what: ['continued'] });
      NA.ui.toast.show(`Run ${run.n} carries on: up to ${fmtInt(m.target)} ${m.unit()}s.`);
      return;
    }
    run.done = false; run.sched0 = run.step;
    run.limit = T.steps ? run.step + T.steps : 0;
    (run.events || (run.events = [])).push({ step: run.step, what: ['continued'] });
    NA.ui.toast.show(`Run ${run.n} carries on from step ${fmtInt(run.step)}${run.limit ? ` to ${fmtInt(run.limit)}` : ' until you pause'}${T.sched !== 'const' && T.steps ? '; its learning-rate schedule starts again' : ''}. Reset (↺) starts a new run from fresh weights.`);
  };
  // the play button says what it will do
  Object.defineProperty(BL, 'labels', { get() { return this.canContinue() ? { play: 'Continue', title: 'Keep training this run (Space)' } : {}; } });
  BL.onPause = function () { if (this.run && this.run.status === 'running') this.run.status = 'paused'; this.evaluate(true); this.inspDirty = true; };
  BL.stepOnce = function () { if (this.training) this.pause(); if (!this.run || this.stale || this.run.done) { if (!this.newRun()) return; } this.trainChunk(0, 1); this.evaluate(true); };
  BL.reset = function () { const was = this.training; if (was) this.pause(); this.stale = true; if (this.newRun()) { NA.ui.toast.show(`Run ${this.run.n}: the same setup from fresh weights.`); if (was) this.play(); } };
  BL.fastForward = function () {};
  BL.lrAt = function (step) {
    const T = this.setup.train, p = T.steps ? clamp(step / T.steps, 0, 1) : 0;
    if (!T.steps || T.sched === 'const') return T.lr;
    if (T.sched === 'cosine') return T.lr * 0.5 * (1 + Math.cos(Math.PI * p));
    if (T.sched === 'step') return T.lr * (p < 0.5 ? 1 : p < 0.75 ? 0.1 : 0.01);
    return T.lr * (p < 0.05 ? p / 0.05 : 0.5 * (1 + Math.cos((Math.PI * (p - 0.05)) / 0.95))); // warm-up, then cosine
  };
  // one batch of training examples, in a shuffled order that is redrawn every epoch
  BL.batch = function (B) {
    const d = this.data, inSize = d.Xtr.length / d.nTr, ySize = d.Ytr.length / d.nTr;
    if (!this.bX || this.bX.length !== B * inSize || this.bX.constructor !== d.Xtr.constructor) { this.bX = new d.Xtr.constructor(B * inSize); this.bY = new d.Ytr.constructor(B * ySize); }
    for (let b = 0; b < B; b++) {
      if (!this.order || this.epochPos >= this.order.length) { this.order = this.rng.shuffle(NA.range(d.nTr)); this.epochPos = 0; }
      const k = this.order[this.epochPos++];
      this.bX.set(d.Xtr.subarray(k * inSize, (k + 1) * inSize), b * inSize);
      this.bY.set(d.Ytr.subarray(k * ySize, (k + 1) * ySize), b * ySize);
    }
    return [this.bX, this.bY];
  };
  BL.trainChunk = function (budget, maxSteps) {
    const t0 = performance.now(), T = this.setup.train, d = this.data, B = Math.min(T.batch, d.nTr), run = this.run;
    const opt = { id: T.opt, lr: T.lr, weightDecay: T.wd, clip: T.clip }, loss = this.setup.loss;
    // a run lasts at least about three seconds, so you can watch it learn; a slow model is held back by time instead
    const cap = maxSteps === Infinity ? (T.steps ? Math.max(2, Math.ceil(T.steps / 180)) : 25) : maxSteps;
    let k = 0;
    do {
      if (run.limit && run.step >= run.limit) { this.finish('done'); break; }
      let r;
      if (this.model.classic) {
        if (this.model.done()) { this.finish('done'); break; }
        r = this.model.step();
      } else {
        opt.lr = this.lrAt(run.step - (run.sched0 || 0));
        const [X, Y] = this.batch(B);
        r = this.model.step(X, Y, B, loss, opt);
      }
      if (!Number.isFinite(r.loss) || r.loss > 1e6) { this.finish('diverged'); NA.ui.toast.show(`Run ${run.n} blew up: the loss became ${Number.isFinite(r.loss) ? 'enormous' : 'NaN'}. Try a learning rate ten times smaller.`); break; }
      run.ema = run.ema === null ? r.loss : 0.97 * run.ema + 0.03 * r.loss;
      run.step++; this.step = run.step; this.gradNorm = r.gradNorm; k++;
      // a point on the curves every few steps, unless measuring would take more than a quarter of the time
      if (run.step % this.evalEvery === 0 && performance.now() - this.lastEval > 6 * this.evalMs) this.evaluate(false);
    } while (performance.now() - t0 < budget && k < cap);
    if (performance.now() - this.lastEval > (this.training ? Math.max(400, 8 * this.evalMs) : 0)) this.evaluate(false);
    return k;
  };
  BL.finish = function (status) {
    const run = this.run;
    if (!run) return;
    run.status = status; run.done = true;
    if (this.training) { this.training = false; NA.app.emit('training'); }
    this.evaluate(true);
    if (status === 'done') NA.ui.toast.show(this.model && this.model.classic
      ? `Run ${run.n} is complete: ${fmtInt(this.model.rounds)} ${this.model.unit()}${this.model.rounds === 1 ? '' : 's'}, ${this.metricText(run)}.${this.model.canExtend() ? ' Continue adds more.' : ''}`
      : `Run ${run.n} stopped at step ${fmtInt(run.step)}: ${this.metricText(run)}. Press Continue to keep training it, or change the model for a new run to compare.`);
    this.inspDirty = true;
  };
  // validation (and a sample of training) scores, recorded on the run's curve
  BL.evaluate = function (full) {
    const run = this.run, d = this.data, m = this.model, t0 = performance.now();
    if (!run || !m) return;
    // fewer examples for long sequences, which cost more to run
    const per = d.input.kind === 'tokens' ? d.input.len : 1, loss = this.setup.loss;
    const nV = Math.min(d.nVa, Math.max(50, Math.round((full ? 3000 : 600) / per))), nT = Math.min(d.nTr, Math.max(50, Math.round((full ? 1500 : 300) / per)));
    const sub = (X, n) => X.subarray(0, n * (X.length / (X === d.Xtr ? d.nTr : d.nVa)));
    const ev = (X, Y, nX, nY, n) => (n ? m.evaluate(sub(X, n), Y.subarray(0, n * (Y.length / nY)), n, loss) : null);
    const v = ev(d.Xva, d.Yva, d.nVa, d.nVa, nV), t = ev(d.Xtr, d.Ytr, d.nTr, d.nTr, nT);
    const metric = (r, X, Y, n, nAll) => {
      if (!r) return null;
      if (BT.isValue(d.family)) return BT.r2(m.predict(X.subarray(0, n * (d.Xtr.length / d.nTr)), n), Y, n); // numbers per example: a window, a row of a table
      return r.count ? r.correct / r.count : null;
      void nAll;
    };
    const pt = { step: run.step, train: t ? t.loss : null, val: v ? v.loss : null, tm: metric(t, d.Xtr, d.Ytr, nT), vm: metric(v, d.Xva, d.Yva, nV) };
    const last = run.curve[run.curve.length - 1];
    if (!last || last.step !== pt.step) run.curve.push(pt); else run.curve[run.curve.length - 1] = pt;
    if (run.curve.length > 400) run.curve = run.curve.filter((q, i) => i % 2 === 0 || i === run.curve.length - 1);
    run.last = pt; run.seconds = (performance.now() - run.started) / 1000;
    if (full && (d.family === 'digits' || d.family === 'seqclass' || (d.table && d.kind === 'classes'))) this.confusion = this.confusionMatrix();
    this.vizVersion++; this.resultsDirty = true; this.inspDirty = true;
    this.lastEval = performance.now(); if (!full) this.evalMs = this.lastEval - t0;
    this.archStale = true;
    if ((d.task === 'language' || d.task === 'sentences') && (full || this.lastEval - (this.lastWrite || 0) > 1000)) this.writeText();
    this.checkChallenge();
  };
  BL.confusionMatrix = function () {
    const d = this.data, n = Math.min(d.nVa, 600), K = d.classes, P = d.Xva.length / d.nVa, o = this.model.predict(d.Xva.subarray(0, n * P), n), M = NA.range(K).map(() => new Array(K).fill(0));
    for (let i = 0; i < n; i++) { let b = 0; for (let k = 1; k < K; k++) if (o[i * K + k] > o[i * K + b]) b = k; M[d.Yva[i]][b]++; }
    return M;
  };
  BL.metricName = function (fam) { fam = fam || this.dataFor().family || this.family(); return BT.isValue(fam) ? 'R²' : fam === 'sequence' ? 'token accuracy' : 'accuracy'; };
  // whether the current problem predicts a value (scored by R²) rather than classes or tokens
  BL.valued = function () { return BT.isValue(this.dataFor().family); };
  BL.metricText = function (run) { const v = run && run.last && run.last.vm; return v === null || v === undefined ? '—' : `validation ${this.metricName(run.family)} ${BT.isValue(run.family) ? fmt(v, 3) : fmtPct(v, 1)}`; };

  // Whether two setups train the same model on the same data from the same start: they may differ only in how
  // training proceeds (learning rate, optimizer, batch, steps, schedule, weight decay, clipping).
  const TRAINING_ONLY = ['lr', 'opt', 'batch', 'steps', 'sched', 'wd', 'clip'];
  function sameModel(a, b) {
    const strip = (S) => { const c = JSON.parse(JSON.stringify(S)); for (const k of TRAINING_ONLY) delete c.train[k]; delete c.parts; (c.blocks || []).forEach((q) => { delete q.id; delete q.grp; delete q.part; (q.branches || []).forEach((l) => l.forEach((x) => { delete x.id; })); }); return JSON.stringify(c); };
    return strip(a) === strip(b);
  }
  // What changed between two setups, in words: the run list shows it next to each run's result.
  function diffSetups(a, b) {
    const out = [], P = (id) => BT.byId[id].label, val = (k, v) => JSON.stringify(k === 'branches' && Array.isArray(v) ? v.map((l) => l.map(cleanSpec)) : v);
    if (a.problem !== b.problem) out.push(`problem ${P(a.problem)} → ${P(b.problem)}`);
    else {
      const pa = a.ps[a.problem] || {}, pb = b.ps[b.problem] || {};
      for (const k of Object.keys(pb)) if (JSON.stringify(pa[k]) !== JSON.stringify(pb[k])) out.push(k === 'text' ? 'edited the text' : k === 'formula' ? `formula → ${pb.formula}` : `${({ noise: 'noise', n: 'examples', len: 'length', embed: 'embedding size', window: 'window' })[k] || k} ${pa[k]} → ${pb[k]}`);
      if (a.problem === 'draw' && JSON.stringify(a.custom.points) !== JSON.stringify(b.custom.points)) out.push(`points: ${a.custom.points.length} → ${b.custom.points.length}`);
    }
    // blocks: match by type in order (longest common subsequence), then compare settings
    const A = a.blocks, Bb = b.blocks, n = A.length, m = Bb.length, L = NA.range(n + 1).map(() => new Array(m + 1).fill(0));
    for (let i = n - 1; i >= 0; i--) for (let j = m - 1; j >= 0; j--) L[i][j] = A[i].type === Bb[j].type ? L[i + 1][j + 1] + 1 : Math.max(L[i + 1][j], L[i][j + 1]);
    let i = 0, j = 0;
    while (i < n || j < m) {
      if (i < n && j < m && A[i].type === Bb[j].type) {
        const diffs = [];
        for (const k of new Set([...Object.keys(A[i]), ...Object.keys(Bb[j])])) if (k !== 'id' && k !== 'grp' && k !== 'part' && val(k, A[i][k]) !== val(k, Bb[j][k])) diffs.push(k === 'acts' ? 'neuron activations' : k === 'branches' ? 'its branches' : `${k} ${fmtVal(k, A[i][k])} → ${fmtVal(k, Bb[j][k])}`);
        if (diffs.length) out.push(`${blockName(Bb[j])} ${j + 1}: ${diffs.join(', ')}`);
        i++; j++;
      } else if (j < m && (i >= n || L[i][j + 1] >= L[i + 1][j])) { out.push(`added ${blockLabel(Bb[j])}`); j++; }
      else { out.push(`removed ${blockLabel(A[i])}`); i++; }
    }
    if (a.loss.id !== b.loss.id) out.push(`loss ${NA.bx.LOSSES[a.loss.id].label} → ${NA.bx.LOSSES[b.loss.id].label}`);
    if ((a.loss.smooth || 0) !== (b.loss.smooth || 0)) out.push(`label smoothing ${a.loss.smooth || 0} → ${b.loss.smooth || 0}`);
    const TN = { opt: 'optimizer', lr: 'learning rate', batch: 'batch', wd: 'weight decay', sched: 'schedule', clip: 'clipping', init: 'initialisation', seed: 'seed', steps: 'steps' };
    for (const k of Object.keys(TN)) if (a.train[k] !== b.train[k]) out.push(`${TN[k]} ${fmtVal(k, a.train[k])} → ${fmtVal(k, b.train[k])}`);
    const count = new Map();
    for (const c of out) count.set(c, (count.get(c) || 0) + 1);
    return out.length ? [...count].map(([c, k]) => (k > 1 ? `${c} ×${k}` : c)) : ['same setup, fresh weights'];
  }
  const OPT_LABEL = { sgd: 'SGD', momentum: 'Momentum', adam: 'Adam', adamw: 'AdamW' }, SCHED_LABEL = { const: 'constant', cosine: 'cosine', step: 'step', warmup: 'warm-up + cosine' };
  function fmtVal(k, v) {
    if (k === 'act') return (ACTS[v] || {}).label || v;
    if (k === 'opt') return OPT_LABEL[v] || v;
    if (k === 'sched') return SCHED_LABEL[v] || v;
    if (k === 'merge') return MERGE_WORD[v] || v;
    if (k === 'steps') return v ? fmtInt(v) : 'until paused';
    if (typeof v === 'boolean') return v ? 'on' : 'off';
    return String(v);
  }
  const blockName = (b) => (b.type === 'dense' && b.units === 1 ? 'Neuron' : PART[b.type] ? PART[b.type].name : b.type);
  function blockLabel(b) {
    if (b.type === 'dense') return b.units === 1 ? `a neuron (${ACTS[b.act].label})` : `Dense ${b.units} ${ACTS[b.act].label}`;
    if (b.type === 'conv') return `Conv ${b.filters} × ${b.k}×${b.k}`;
    if (b.type === 'attention') return `Attention (${b.heads} head${b.heads > 1 ? 's' : ''})`;
    if (b.type === 'transformer') return `Transformer block (${b.heads} heads)`;
    if (b.type === 'dropout') return `Dropout ${b.p}`;
    if (b.type === 'activation') return `${ACTS[b.act].label} activation`;
    if (b.type === 'rnn') return `${{ rnn: 'RNN', lstm: 'LSTM', gru: 'GRU' }[b.cell] || 'Recurrent'} ${b.units}`;
    if (b.type === 'conv1d') return `Conv 1-D ${b.filters} × ${b.k}`;
    if (b.type === 'seqpool') return `Sequence pool (${b.mode})`;
    if (b.type === 'forest') return `Random forest (${b.trees} trees)`;
    if (b.type === 'boost') return `Gradient boosting (${b.rounds} rounds)`;
    if (b.type === 'parallel') return `Parallel ×${(b.branches || []).length} (${MERGE_WORD[b.merge || 'concat']})`;
    return blockName(b);
  }
  BL.blockLabel = blockLabel;
  BL.PARTS = PARTS;

  // ------------------------------------------------------------------ lifecycle
  BL.activate = function (preset, ui) {
    this.ui = ui; this.preset = preset;
    if (!this.started) {
      this.started = true;
      if (!this.restore()) this.setup.blocks = withIds(BT.TEMPLATES.find((t) => t.id === 'mlp').blocks());
      const ch = NA.store.get('builderChallenge', null);
      this.challenge = BC.byId[ch] ? ch : null;
    }
    document.body.classList.add('builder-on');
    this.ro = UI.readouts(ui.readouts, [{ key: 'run', label: 'Run', keep: true }, { key: 'step', label: 'Step', keep: true }, { key: 'loss', label: 'Train loss' }, { key: 'val', label: 'Val. loss' }, { key: 'metric', label: 'Val. score', keep: true }, { key: 'params', label: 'Parameters' }]);
    this.buildHyper(ui.hyper);
    this.buildLeft(ui.left);
    this.buildCenter(ui.center);
    this.tabs = UI.tabs(ui.right, [{ id: 'block', label: 'Part' }, { id: 'train', label: 'Training' }, { id: 'runs', label: 'Runs' }, { id: 'python', label: 'Python' }], (id, body) => this.buildTab(id, body), this.tabId);
    this.offTheme = theme.onChange(() => { this._region = null; this.vizVersion = (this.vizVersion || 0) + 1; this.resultsDirty = true; this.inspDirty = true; this.archStale = true; this.renderLeft(); });
    this.recheck();
    this.renderWorkbench(); this.renderLeft(); this.syncUndo();
    this._chKey = null; this.renderChallenge();
    if (!this.run) this.showDataOnly();
    this.resultsDirty = true;
  };
  BL.deactivate = function () {
    document.body.classList.remove('builder-on');
    if (this.offTheme) this.offTheme();
    if (this.resizeObs) this.resizeObs.disconnect();
    if (this.arch) this.arch.destroy();
    this.cancelDrag();
  };
  BL.archText = function () { const c = this.check; return c && c.ok ? `your model · ${fmtInt(c.params)} parameters` : 'your model'; };
  BL.stateText = function () { return this.training ? 'Training' : this.run && this.stale ? 'Edited' : this.run ? (this.run.done ? (this.run.status === 'diverged' ? 'Diverged' : 'Finished') : 'Paused') : 'Not trained'; };
  // with no model trained yet, the results area shows the data on its own
  BL.showDataOnly = function () { if (!this.training) { this.data = this.dataFor(); if (!this.run || this.stale) this.modelView = null; this.resultsDirty = true; } };

  BL.frame = function (now) {
    this.runFrame();
    if (now - (this.lastRo || 0) > 150) {
      this.lastRo = now;
      const r = this.run, p = r && !this.stale && r.last;
      this.ro.set('run', r ? `${r.n}` : '—');
      this.ro.set('step', r ? fmtInt(r.step) + (r.limit && !this.stale && !(NA.mobile && NA.mobile.mobile) ? ` / ${fmtInt(r.limit)}` : '') : '0');
      this.ro.set('loss', r && !this.stale && r.ema !== null ? fmt(r.ema, 4) : '—');
      this.ro.set('val', p && p.val !== null ? fmt(p.val, 4) : '—');
      this.ro.set('metric', p && p.vm !== null && p.vm !== undefined ? (BT.isValue(r.family) ? `R² ${fmt(p.vm, 3)}` : fmtPct(p.vm, 1)) : '—');
      this.ro.set('params', this.check && this.check.ok ? fmtInt(this.check.params) : '—');
    }
    if (this.resultsDirty && this.resCanvas.offsetParent && (!this.training || now - (this.lastRes || 0) > (this.vizMs > 25 ? 700 : 280))) { this.lastRes = now; this.resultsDirty = false; this.drawResults(); }
    if (this.inspDirty && this.tabs.body.offsetParent && (!this.training || now - (this.lastInsp || 0) > 500)) { this.lastInsp = now; this.inspDirty = false; this.updateTab(); }
    if (this.training && now - (this.lastBench || 0) > 600) { this.lastBench = now; this.paintNeurons(); }
    if (this.leftDirty && this.preview.parentElement.clientWidth) { this.leftDirty = false; this.renderLeft(); } // drawn while its screen was hidden
    if (this.centerView === 'arch' && this.arch) {
      if (this.archStale && (!this.training || now - (this.lastArch || 0) > 700)) { this.lastArch = now; this.archStale = false; this.arch.dirty = true; this.arch.refreshFocus(); }
      this.arch.frame(now);
    }
  };

  // the quick settings in the control bar (the phone's settings sheet), rebuilt whenever the setup changes them
  BL.buildHyper = function (root) {
    if (root) this.hyperRoot = root;
    const T = this.setup.train, rates = [0.0003, 0.001, 0.003, 0.01, 0.03, 0.1, 0.3, 1];
    if (this.isClassicSetup()) {
      this._hyperKey = 'classic';
      const b = this.setup.blocks.find((q) => isClassic(q.type));
      clear(this.hyperRoot).append(el('p', { class: 'sec-note bd-hyper-note', text: `A ${PART[b.type].name.toLowerCase()} needs no optimizer, learning rate or batches: it trains on all the data, a ${{ linear: 'step', knn: 'look-up', tree: 'level', forest: 'tree', boost: 'round' }[b.type]} at a time. Its settings are on the model block.` }));
      return;
    }
    if (!rates.includes(T.lr)) { rates.push(T.lr); rates.sort((a, b) => a - b); } // a rate set on the slider
    this._hyperKey = JSON.stringify([T.opt, T.lr, T.batch, T.steps]);
    clear(this.hyperRoot).append(
      UI.field('Optimizer', UI.select(Object.entries(OPT_LABEL).map(([value, label]) => ({ value, label })), T.opt, (v) => this.edit((s) => { s.train.opt = v; }))),
      UI.field('Learning rate', UI.select(rates.map((v) => ({ value: v, label: String(v) })), T.lr, (v) => this.edit((s) => { s.train.lr = +v; }))),
      UI.field('Batch', UI.select([8, 16, 32, 64, 128].map((v) => ({ value: v, label: String(v) })), T.batch, (v) => this.edit((s) => { s.train.batch = +v; }))),
      UI.field('Steps', UI.select(STEP_CHOICES.map(([value, label]) => ({ value, label })), T.steps, (v) => this.edit((s) => { s.train.steps = +v; })), 'How many steps each press of Train runs: ∞ trains until you pause'));
  };

  // ------------------------------------------------------------------ left: problem and parts
  BL.buildLeft = function (root) {
    const body = UI.panel(root, 'Problem and parts');
    this.chRoot = el('div', { class: 'bd-ch', 'data-tour': 'challenges' });
    body.append(this.chRoot);
    const groups = Object.keys(BT.FAMILIES).map((f) => el('optgroup', { label: BT.FAMILIES[f].label }, ...BT.PROBLEMS.filter((p) => p.family === f).map((p) => el('option', { value: p.id, text: p.label }))));
    this.problemSel = el('select', { class: 'bd-select', 'aria-label': 'Problem' }, ...groups);
    this.problemSel.addEventListener('change', () => this.setProblem(this.problemSel.value));
    this.problemHint = el('p', { class: 'sec-note' });
    this.preview = el('canvas', { class: 'bd-preview', 'aria-hidden': 'true' });
    this.tplRow = el('div', { class: 'bd-tpls' });
    this.palette = el('div', { class: 'bd-palette' });
    body.append(
      UI.section('1 · Problem', this.problemSel, this.problemHint, el('div', { class: 'bd-preview-box' }, this.preview)),
      UI.section('2 · Start from', this.tplRow),
      UI.section('3 · Parts', UI.note(NA.mobile && NA.mobile.mobile ? 'Tap a part to see how it works and add it to the model.' : 'Drag a part onto the model, or click it to see how it works and add it.'), this.palette));
    for (const g of Object.keys(GROUP_TONE)) {
      const grid = el('div', { class: 'bd-pal-grid' });
      for (const p of PARTS.filter((q) => q.group === g)) {
        const tile = el('button', { class: 'bd-part', type: 'button', title: p.hint, style: { '--bk': `var(--${GROUP_TONE[g]})` }, dataset: { type: p.type } }, el('span', { class: 'bd-part-ico' }, icon(p.type)), el('span', { class: 'bd-part-name', text: p.label }));
        tile.addEventListener('pointerdown', (e) => this.pressStart(e, { kind: 'part', part: p, el: tile }));
        tile.addEventListener('click', (e) => { if (e.detail === 0) this.showGuide(p.type); }); // keyboard
        grid.append(tile);
      }
      this.palette.append(el('div', { class: 'bd-pal-group' }, el('div', { class: 'bd-pal-title', text: g }), grid));
    }
    this.myParts = el('div', { class: 'bd-pal-group bd-myparts', hidden: '' });
    this.palette.append(this.myParts);
  };
  BL.renderLeft = function () {
    if (!this.problemSel) return;
    const p = this.problem(), fam = p.family;
    this.problemSel.value = p.id;
    this.problemHint.textContent = p.hint;
    clear(this.tplRow).append(...BT.templatesFor(fam).map((t) => el('button', { class: 'chip bd-tpl', type: 'button', title: t.hint, onclick: () => this.applyTemplate(t) }, t.label)));
    // which parts fit this problem's kind of data
    const fits = FITS[fam] || FITS.plane;
    const kind = { plane: 'points', curve: 'a curve', surface: 'a surface', digits: 'pictures', sequence: 'sequences', seqclass: 'sequences', series: 'a series', table: 'a table' }[fam];
    this.renderMyParts();
    NA.$$('.bd-part:not(.bd-part-mine)', this.palette).forEach((t) => {
      const off = !fits.includes(t.dataset.type);
      t.classList.toggle('is-unsuited', off);
      t.title = PART[t.dataset.type].hint + (off ? ` Not useful for ${kind}: the model will say why if you add it.` : '');
    });
    // a hidden screen (the phone's Parts screen) has no width yet: draw the preview when it shows
    if (!this.preview.parentElement.clientWidth) { this.leftDirty = true; return; }
    const W = Math.max(160, this.preview.parentElement.clientWidth), Hh = fam === 'sequence' ? 78 : fam === 'digits' ? 64 : 120;
    const ctx = NA.fitCanvas(this.preview, W, Hh);
    BV.preview(ctx, W, Hh, this.dataFor());
  };
  // the parts that suit each kind of data (the others still work where shapes allow, or say why not)
  const VECTOR_PARTS = ['dense', 'neuron', 'gated', 'activation', 'dropout', 'noise', 'layernorm', 'batchnorm', 'fourier', 'parallel'], CLASSIC_PARTS = ['linear', 'knn', 'tree', 'forest', 'boost'];
  const SEQ_PARTS = ['dense', 'neuron', 'gated', 'activation', 'dropout', 'noise', 'layernorm', 'batchnorm', 'attention', 'transformer', 'rnn', 'conv1d', 'position', 'seqpool', 'flatten', 'gap', 'parallel'];
  const FITS = {
    plane: [...VECTOR_PARTS, ...CLASSIC_PARTS], curve: [...VECTOR_PARTS, ...CLASSIC_PARTS], surface: [...VECTOR_PARTS, ...CLASSIC_PARTS], table: [...VECTOR_PARTS, ...CLASSIC_PARTS],
    digits: ['dense', 'neuron', 'gated', 'activation', 'dropout', 'noise', 'batchnorm', 'conv', 'pool', 'avgpool', 'patches', 'flatten', 'gap', 'parallel', 'attention', 'transformer', 'position', 'seqpool', ...CLASSIC_PARTS],
    series: [...SEQ_PARTS, ...CLASSIC_PARTS], seqclass: SEQ_PARTS, sequence: SEQ_PARTS.filter((t) => t !== 'seqpool' && t !== 'gap' && t !== 'flatten'),
  };
  BL.fitsHere = function (type) { return (FITS[this.family()] || FITS.plane).includes(type); };
  BL.setProblem = function (id) {
    if (id === this.setup.problem) return;
    const fam0 = this.family(), fam1 = BT.byId[id].family;
    this.edit((s) => {
      s.problem = id;
      const fl = BT.FAMILIES[fam1];
      if (!fl.losses.includes(s.loss.id) || fam0 !== fam1) s.loss = { id: fl.losses[0], smooth: 0, delta: 0.2 };
      // a different kind of data: start from that kind's usual template, with sensible training settings
      if (fam0 !== fam1) {
        s.blocks = withIds(BT.TEMPLATES.find((t) => t.id === fl.template).blocks());
        Object.assign(s.train, BT.trainFor(BT.byId[id]));
      }
    }, { flip: true });
    this.seqView = {};
    if (this.training) this.pause();
    // a problem you make yourself opens its editor: the Input block's settings, or the drawing surface
    const custom = BT.byId[id].custom, phone = NA.mobile && NA.mobile.mobile;
    this.selectBlock(custom ? { kind: 'input', id: 'input' } : null, true);
    if (custom && phone) {
      if (id === 'draw') { this.centerView = 'results'; this.layoutCenter(); NA.mobile.go('network'); }
      else NA.mobile.go('insp');
    }
    if (id === 'draw' && !this.setup.custom.points.length) NA.ui.toast.show(`${Click()} on the picture to place points; pick the class first.`);
  };
  BL.applyTemplate = function (t) {
    const blocks = BT.templateBlocks(t, this.family());
    this.edit((s) => { s.blocks = withIds(blocks); }, { flip: true });
    this.selectBlock(null);
    NA.ui.toast.show(`${t.label} template. ${t.hint}`);
  };
  // add a part below the selected block (or at the end)
  // the guide to a part, in the Part tab, with a button that adds it (below the block selected before)
  BL.showGuide = function (type) {
    if (!(this.sel && this.sel.kind === 'guide')) this.guideFrom = this.sel;
    this.selectBlock({ kind: 'guide', id: 'guide:' + type, type });
  };
  BL.addPart = function (p) {
    if (this.sel && this.sel.kind === 'guide') this.sel = this.guideFrom || null;
    if (p.classic || this.setup.blocks.some((b) => isClassic(b.type))) {
      const had = this.setup.blocks.length, was = had ? (this.setup.blocks.some((b) => isClassic(b.type)) ? 'classic' : 'network') : null;
      if (p.classic || was === 'classic') {
        const b = this.prepSpec(p.make(), 0);
        this.edit((s) => { s.blocks = [b]; }, { flip: true });
        this.selectBlock({ kind: 'block', id: b.id }, true);
        if (had) NA.ui.toast.show(p.classic ? `A ${p.name.toLowerCase()} is a whole model on its own, so it replaces the ${was === 'classic' ? 'other one' : 'network'}. Undo brings ${was === 'classic' ? 'it' : 'the network'} back.` : `A layer starts a new network in place of the classic model. Undo brings the model back.`);
        return;
      }
    }
    if (p.custom) { this.addCopy(p.custom); return; }
    const i = this.insertAt();
    if (p.type === 'neuron' && this.sel && this.sel.kind === 'block') { const b = this.setup.blocks.find((q) => q.id === this.sel.id); if (b && b.type === 'dense' && !b.grp) { this.addNeuron(b.id); return; } }
    this.insert(p.make(), i);
    if (NA.mobile && NA.mobile.mobile) NA.ui.toast.show(`Added ${p.name.toLowerCase()} to the model.`);
  };
  // where a new part goes: after the selected block (after the whole copy, when it sits in one of your parts), or at the end
  BL.insertAt = function () {
    const S = this.setup, sel = this.sel;
    if (sel && sel.kind === 'input') return 0;
    if (sel && (sel.kind === 'block' || sel.kind === 'group')) {
      const run = runsOf(S).find((g) => g.grp === sel.id || S.blocks.slice(g.from, g.to).some((b) => b.id === sel.id));
      if (run) return run.to;
      const i = S.blocks.findIndex((b) => b.id === sel.id);
      if (i >= 0) return i + 1;
    }
    return S.blocks.length;
  };
  // a new block for position at: its id, fresh ids inside its branches, and for a new parallel block two branches
  // that suit the data arriving there
  BL.prepSpec = function (spec, at) {
    const b = Object.assign({ id: bid() }, spec);
    if (b.type === 'parallel' && b.auto) { delete b.auto; b.branches = defaultLanes(this.shapeAt(at), this.family()); }
    return freshIds(b);
  };
  BL.shapeAt = function (at) {
    const chk = this.check || this.recheck(), q = this.dataFor().input || {};
    if (at > 0 && chk.shapes[at - 1] && chk.shapes[at - 1].out) return chk.shapes[at - 1].out;
    return q.kind === 'tokens' ? [q.len, q.embed] : q.kind === 'series' ? [q.len, q.dim] : q.kind === 'image' ? [q.c, q.h, q.w] : [q.dim || 2];
  };
  const defaultLanes = (s, fam) => (s && s.length === 3 ? [[{ type: 'conv', filters: 8, k: 3, stride: 1, act: 'relu' }], [{ type: 'conv', filters: 8, k: 5, stride: 1, act: 'relu' }]]
    : s && s.length === 2 ? [[{ type: 'conv1d', filters: 16, k: 3, dilation: 1, causal: fam === 'sequence', act: 'relu' }], [{ type: 'dense', units: 16, act: 'gelu' }]]
      : [[{ type: 'dense', units: 16, act: 'relu' }], [{ type: 'dense', units: 16, act: 'tanh' }]]);
  BL.insert = function (spec, at) {
    if (this.setup.blocks.length >= 24) { NA.ui.toast.show('That is plenty of blocks: remove one first.'); return; }
    const inside = runsOf(this.setup).find((g) => at > g.from && at < g.to);
    if (inside) at = inside.to; // never into the middle of a copy of one of your parts
    const b = this.prepSpec(spec, at);
    this.edit((s) => { s.blocks.splice(at, 0, b); }, { flip: true });
    this.selectBlock({ kind: 'block', id: b.id }, true);
  };
  BL.removeBlock = function (id) {
    const fb = findBlock(this.setup, id);
    if (fb && fb.owner) { this.removeFromLane(fb.owner.id, fb.br, fb.k); return fb.i; } // from a branch of a parallel block
    const i = this.setup.blocks.findIndex((b) => b.id === id), b = this.setup.blocks[i], run = runOf(this.setup, b);
    if (run) {
      // a block inside one of your parts goes from the part, and so from every copy
      const k = i - run.from, def = defOf(this.setup, run.part), name = def ? def.name : 'the part';
      this.edit((s) => editPart(s, run.part, (list, g) => list.splice((g ? g.from : 0) + k, 1)), { flip: true });
      if (this.sel && this.sel.id === id) this.selectBlock(null);
      NA.ui.toast.show(`Removed ${blockName(b)} from ${name}, in every copy. Undo brings it back.`);
      return i;
    }
    this.edit((s) => { s.blocks = s.blocks.filter((b) => b.id !== id); }, { flip: true });
    if (this.sel && this.sel.id === id) this.selectBlock(null);
    return i;
  };
  // keyboard: after a block moves or goes, keep the focus on the model (the same block, or its neighbour)
  BL.refocus = function (id, i) {
    requestAnimationFrame(() => {
      const cards = NA.$$('.bd-card', this.stack), c = (id && this.stack.querySelector(`[data-bid="${id}"]`)) || cards[Math.min(cards.length - 1, (i ?? 0) + 1)];
      if (c) c.focus();
    });
  };
  BL.moveBlock = function (id, to) {
    const S = this.setup, i = S.blocks.findIndex((b) => b.id === id), run = i >= 0 ? runOf(S, S.blocks[i]) : null;
    if (run) {
      // inside one of your parts: it moves within the part, in every copy
      const k = i - run.from, k2 = to > i ? k + 1 : k - 1;
      if (k2 < 0 || k2 >= run.to - run.from) return;
      this.edit((s) => editPart(s, run.part, (list, g) => { const o = g ? g.from : 0, t = list[o + k]; list[o + k] = list[o + k2]; list[o + k2] = t; }), { flip: true });
      return;
    }
    const over = runsOf(S).find((g) => to > g.from && to < g.to);
    if (over) to = to > i ? over.to : over.from; // a block steps over a copy of one of your parts, not into it
    if (i < 0 || to === i || to === i + 1) return;
    this.edit((s) => { const [b] = s.blocks.splice(i, 1); s.blocks.splice(to > i ? to - 1 : to, 0, b); }, { flip: true });
  };
  BL.addNeuron = function (id, delta = 1) {
    this.edit((s) => {
      const b = s.blocks.find((q) => q.id === id);
      if (!b || b.type !== 'dense') return;
      const u = clamp(b.units + delta, 1, 256);
      if (b.acts) { b.acts = b.acts.slice(0, u); while (b.acts.length < u) b.acts.push(b.act); }
      b.units = u;
      if (b.grp) syncPart(s, b);
    });
  };
  BL.fixBlock = function (i, fix) {
    const run = runsOf(this.setup).find((g) => i > g.from && i < g.to);
    if (run) {
      // inside one of your parts: the fix goes into the part, in every copy
      const k = i - run.from, def = defOf(this.setup, run.part);
      this.edit((s) => editPart(s, run.part, (list, g) => list.splice((g ? g.from : 0) + k, 0, ...fix.map((f) => (g ? Object.assign({ id: bid() }, cleanSpec(f), { grp: g.grp, part: run.part }) : cleanSpec(f))))), { flip: true });
      NA.ui.toast.show(`Added ${fix.map(blockLabel).join(' and ')} inside ${def ? def.name : 'your part'}, in every copy.`);
      return;
    }
    this.edit((s) => { s.blocks.splice(i, 0, ...withIds(fix)); }, { flip: true });
    NA.ui.toast.show(`Added ${fix.map(blockLabel).join(' and ')}.`);
  };

  // ------------------------------------------------------------------ centre: the workbench and the results
  BL.buildCenter = function (root) {
    this.undoBtn = el('button', { class: 'icon-btn', type: 'button', title: 'Undo (Ctrl+Z)', 'aria-label': 'Undo', onclick: () => this.undo() }, '↶');
    this.redoBtn = el('button', { class: 'icon-btn', type: 'button', title: 'Redo (Ctrl+Y)', 'aria-label': 'Redo', onclick: () => this.redo() }, '↷');
    this.viewSeg = UI.seg([{ value: 'model', label: 'Model' }, { value: 'both', label: 'Both' }, { value: 'results', label: 'Results' }, { value: 'arch', label: 'Architecture', title: 'Zoom into your model: blocks, neurons, their sums, down to the transistors' }], this.centerView, (v) => { this.centerView = v; this.layoutCenter(); }, 'Show');
    this.archStyle = NA.store.get('builder.archStyle', 'network') === 'blocks' ? 'blocks' : 'network';
    this.styleSeg = UI.seg([{ value: 'network', label: 'Network', title: 'The model drawn as a network: neurons, their wires and what each one computes' }, { value: 'blocks', label: 'Boxes', title: 'The model as boxes, with the shape of the data between them' }], this.archStyle, (v) => { this.archStyle = v; NA.store.set('builder.archStyle', v); if (this.arch) { this.arch.dirty = true; this.arch.refreshFocus(); } }, 'Architecture drawn as');
    this.styleSeg.classList.add('bd-style');
    root.append(el('div', { class: 'lab-toolbar bd-toolbar' }, this.viewSeg, el('div', { class: 'bd-tool-right' }, this.styleSeg, this.undoBtn, this.redoBtn,
      el('button', { class: 'btn btn-quiet', type: 'button', title: 'Remove every block', onclick: () => { if (!this.setup.blocks.length) return; this.edit((s) => { s.blocks = []; }, { flip: true }); this.selectBlock(null); NA.ui.toast.show('Removed every block. Undo brings them back.'); } }, 'Clear model'))));
    this.bench = el('div', { class: 'bd-bench', 'aria-label': 'Your model, from input to output' });
    this.stack = el('div', { class: 'bd-stack' });
    this.bench.append(this.stack);
    this.resHead = el('div', { class: 'bd-res-head' });
    this.resCanvas = el('canvas', { class: 'bd-res-canvas', 'aria-label': 'Results of the model on the problem' });
    if (NA.a11y) NA.a11y.describe(this.resCanvas, () => {
      const p = this.problem(), run = this.run && !this.stale ? this.run : null;
      return `Results on ${p.label.toLowerCase()}: ${run ? `run ${run.n}, ${this.metricText(run)} after ${fmtInt(run.step)} steps` : 'the data, before training'}.`;
    });
    this.resBox = el('div', { class: 'bd-res-box' }, this.resCanvas);
    this.lossCanvas = el('canvas');
    this.lossChart = new NA.charts.LineChart(this.lossCanvas, { height: 104, logY: true, format: (v) => fmt(v, 4), xLabel: 'Step' });
    this.resFoot = el('div', { class: 'bd-res-foot' });
    this.results = el('div', { class: 'bd-results' }, this.resHead, this.resBox, this.resFoot);
    this.resFoot.append(el('div', { class: 'chart-box' }, this.lossCanvas), el('div', { class: 'mini-legend' },
      el('span', { class: 'lg-item' }, el('i', { class: 'key-line', style: { background: 'var(--s1)' } }), 'training loss'), el('span', { class: 'lg-item' }, el('i', { class: 'key-line', style: { background: 'var(--s2)' } }), 'validation loss')));
    this.split = el('div', { class: 'bd-split' }, this.bench, this.results);
    root.append(this.split);
    // the architecture: your model as a zoomable diagram, from its blocks down to one multiplication's transistors
    this.archBox = el('div', { class: 'bd-arch' });
    this.archStage = el('div', { class: 'lab-stage arch-stage' });
    this.archBox.append(this.archStage);
    this.arch = new NA.ArchView(this.archStage, { label: `Your model’s architecture: ${NA.openHint('a part', { zoom: true })}`, kicker: 'Architecture',
      build: (size) => NA.archBuilder.build(this, size), onFocus: (node, path) => { if (this.centerView === 'arch') this.info.set(this.arch.infoFor(node, path)); } });
    this.info = UI.infoPanel(this.archBox, { open: !(NA.mobile && NA.mobile.mobile) && root.clientHeight >= 760 });
    root.append(this.archBox);
    this.resCanvas.addEventListener('pointerdown', (e) => this.resultsPointer(e));
    this.resCanvas.addEventListener('contextmenu', (e) => { if (this.problem().id === 'draw') e.preventDefault(); });
    this.resizeObs = new ResizeObserver(() => { this.layoutCenter(); this.resultsDirty = true; });
    this.resizeObs.observe(this.split); this.resizeObs.observe(this.resBox);
    this.layoutCenter();
  };
  // side by side when there is room, otherwise one at a time
  BL.layoutCenter = function () {
    const arch = this.centerView === 'arch';
    this.split.hidden = arch; this.archBox.hidden = !arch;
    if (this.styleSeg) this.styleSeg.hidden = !arch;
    if (arch) { this.viewSeg.set('arch'); NA.$$('button', this.viewSeg).forEach((b) => { if (b.dataset.v === 'both') b.hidden = (this.root || this.archBox.parentElement).clientWidth < 600 || !!(NA.mobile && NA.mobile.mobile); }); this.archStale = true; return; }
    const w = this.split.clientWidth, narrow = w < 600 || (NA.mobile && NA.mobile.mobile);
    let v = this.centerView;
    if (narrow && v === 'both') v = this.run && !this.stale ? 'results' : 'model';
    this.split.dataset.view = v;
    this.viewSeg.hidden = false;
    NA.$$('button', this.viewSeg).forEach((b) => { if (b.dataset.v === 'both') b.hidden = narrow; });
    this.viewSeg.set(narrow && this.centerView === 'both' ? v : this.centerView);
    this.resultsDirty = true;
  };

  BL.renderWorkbench = function () {
    if (!this.stack) return;
    const S = this.setup, chk = this.check || this.recheck(), d = this.dataFor(), scroll = this.bench.scrollTop;
    clear(this.stack);
    const shapeOf = (s) => (s ? NA.bx.shapeText(s) : '?');
    // input
    const p = this.problem();
    this.stack.append(this.card({ kind: 'input', id: 'input', type: 'input', title: 'Input', sum: `${p.label}${d.error ? '' : ' · ' + inputText(d)}`, sel: this.sel && this.sel.kind === 'input' }));
    const runs = runsOf(S);
    for (let i = 0; i < S.blocks.length;) {
      const b = S.blocks[i], sh = chk.shapes[i] || {}, run = runs.find((g) => g.from === i);
      this.stack.append(this.gap(i, sh.in ? shapeOf(sh.in) : ''));
      if (run) { this.stack.append(this.groupCard(run, chk)); i = run.to; continue; }
      const shown = Object.assign({}, b, { residualOk: residualFits(b, sh) });
      this.stack.append(this.card({ kind: 'block', id: b.id, type: b.type === 'dense' && b.units === 1 ? 'neuron' : b.type, b: shown, i, title: blockName(b), sum: blockSummary(shown), params: sh.params, problem: sh.problem, sel: this.sel && this.sel.id === b.id, extra: b.type === 'parallel' ? this.lanesEl(b, sh) : null }));
      i++;
    }
    // what flows into the output: the last block's shape, or the input's when there are no blocks
    const last = chk.shapes.length ? (chk.shapes[chk.shapes.length - 1] || {}).out : null;
    this.stack.append(this.gap(S.blocks.length, d.error ? '' : shapeOf(last || (d.input.kind === 'tokens' ? [d.input.len, d.input.embed] : d.input.kind === 'series' ? [d.input.len, d.input.dim] : d.input.kind === 'vector' ? [d.input.dim] : [d.input.c, d.input.h, d.input.w]))));
    this.stack.append(this.card({ kind: 'output', id: 'output', type: 'output', title: 'Output', sum: d.error ? '' : outputText(d, S.loss), problem: chk.problem && !d.error ? chk.problem : null, sel: this.sel && this.sel.kind === 'output' }));
    if (!S.blocks.length) this.stack.insertBefore(el('div', { class: 'bd-empty' }, touch() ? 'No hidden layers: the input goes straight to the output. Tap parts on the Parts screen to add some.' : 'No hidden layers: the input goes straight to the output. Drag parts here from the palette to add some.'), this.stack.children[2]);
    this.bench.scrollTop = scroll;
    this.paintNeurons();
  };
  // a dense layer's residual (input added back) needs its output to be the size of its input
  function residualFits(b, sh) { return b.type !== 'dense' || !sh.in || !sh.out || sh.in[sh.in.length - 1] === sh.out[sh.out.length - 1]; }
  function inputText(d) {
    const i = d.input;
    if (i.kind === 'vector' && d.table) return `${i.dim} number${i.dim > 1 ? 's' : ''} (${d.featNames.length > 3 ? d.featNames.slice(0, 3).join(', ') + ', …' : d.featNames.join(', ')})`;
    if (i.kind === 'vector') return i.dim === 2 ? '2 numbers (x, y)' : '1 number (x)';
    if (i.kind === 'image') return `${i.h} × ${i.w} picture`;
    if (i.kind === 'series') return `the last ${i.len} values, in order`;
    return `${i.len} tokens → embedding ${i.embed}`;
  }
  function outputText(d, loss) {
    const o = d.output, L = NA.bx.LOSSES[loss.id];
    const what = o.kind === 'classes' ? `${o.k} class scores${loss.id === 'hinge' ? '' : ' (softmax)'}` : o.kind === 'value' ? '1 number' : `next-token scores over ${o.vocab} ${d.task === 'language' || d.task === 'sentences' ? 'words' : 'tokens'}`;
    return `${what} · ${L ? L.label : loss.id} loss`;
  }
  function blockSummary(b) {
    if (b.type === 'dense') return `${b.units} neuron${b.units > 1 ? 's' : ''} · ${b.acts ? 'mixed activations' : ACTS[b.act].label}${b.residual && b.residualOk !== false ? ' · residual' : ''}`;
    if (b.type === 'activation') return ACTS[b.act].label;
    if (b.type === 'dropout') return `drop ${fmtPct(b.p, 0)} while training`;
    if (b.type === 'attention') return `${b.heads} head${b.heads > 1 ? 's' : ''}${b.causal !== false ? ' · causal' : ' · sees the future'}${b.residual !== false ? ' · residual' : ''}`;
    if (b.type === 'transformer') return `${b.heads} heads · MLP ×${b.mlp || 4}${b.causal === false ? ' · sees the future' : ''}`;
    if (b.type === 'conv') return `${b.filters} filters · ${b.k}×${b.k}${b.stride > 1 ? ' · stride ' + b.stride : ''} · ${ACTS[b.act].label}`;
    if (b.type === 'pool' || b.type === 'avgpool') return `${b.size}×${b.size}`;
    if (b.type === 'gated') return `${b.units} units · ${{ swish: 'SwiGLU', sigmoid: 'GLU', gelu: 'GEGLU' }[b.gate] || b.gate}`;
    if (b.type === 'noise') return `σ ${b.sigma} while training`;
    if (b.type === 'batchnorm') return 'per feature, over the batch';
    if (b.type === 'fourier') return `${b.features} frequencies · σ ${b.scale}`;
    if (b.type === 'rnn') return `${{ rnn: 'RNN', lstm: 'LSTM', gru: 'GRU' }[b.cell] || b.cell} · ${b.units} units`;
    if (b.type === 'conv1d') return `${b.filters} filters · window ${b.k}${b.dilation > 1 ? ' · dilation ' + b.dilation : ''}${b.causal ? ' · causal' : ''}${b.residual ? ' · residual' : ''}`;
    if (b.type === 'position') return b.mode === 'sine' ? 'fixed sine waves' : 'learned';
    if (b.type === 'seqpool') return { last: 'the last step', max: 'the largest of each feature', mean: 'the average' }[b.mode] || b.mode;
    if (b.type === 'patches') return `${b.size}×${b.size} patches → ${b.dim} numbers each`;
    if (b.type === 'linear') return b.lambda ? `ridge λ ${b.lambda}` : 'least squares or logistic';
    if (b.type === 'knn') return `k = ${b.k}${b.weights === 'distance' ? ' · weighted by distance' : ''}`;
    if (b.type === 'tree') return `depth ${b.depth}`;
    if (b.type === 'forest') return `${b.trees} trees · depth ${b.depth}`;
    if (b.type === 'boost') return `${b.rounds} rounds · depth ${b.depth} · η ${b.eta}`;
    if (b.type === 'parallel') return `${(b.branches || []).length} branches · ${MERGE_TEXT[b.merge || 'concat']}`;
    return '';
  }
  BL.gap = function (i, label) {
    return el('div', { class: 'bd-gap', dataset: { gap: String(i) } }, el('span', { class: 'bd-gap-line' }), label ? el('span', { class: 'bd-gap-shape', text: label }) : null);
  };
  BL.card = function (o) {
    const tone = toneOf(o.type), io = o.kind !== 'block';
    // hovering a block says in a sentence what it is; clicking it opens the whole guide and its settings
    const guide = NA.partGuide && NA.partGuide.G[o.type], gist = guide ? (guide.what.match(/^.*?\.(\s|$)/) || [guide.what])[0].trim() : null;
    const card = el('div', { class: `bd-card bd-${o.kind}${o.sel ? ' is-selected' : ''}${o.problem ? ' is-bad' : ''}`, style: { '--bk': `var(--${tone})` }, dataset: { bid: o.id }, tabindex: '0', role: 'button', title: gist ? `${gist} Click for more.` : null, 'aria-label': `${o.title}${o.sum ? ': ' + o.sum : ''}` });
    const head = el('div', { class: 'bd-card-head' }, el('span', { class: 'bd-card-ico' }, icon(o.type)), el('span', { class: 'bd-card-text' }, el('span', { class: 'bd-card-title', text: o.title }), o.sum ? el('span', { class: 'bd-card-sum', text: o.sum }) : null));
    if (o.params) head.append(el('span', { class: 'bd-card-params', title: 'learned numbers in this block', text: fmtInt(o.params) }));
    if (!io) head.append(el('button', { class: 'bd-card-x', type: 'button', title: 'Remove (Delete)', 'aria-label': `Remove ${o.title}`, onclick: (e) => { e.stopPropagation(); this.removeBlock(o.id); } }, '×'));
    card.append(head);
    if (o.b && o.b.type === 'dense') card.append(this.neuronRow(o.b));
    if (o.b && o.b.type === 'transformer') card.append(el('div', { class: 'bd-mini' }, ...['norm', 'attention', '+', 'norm', 'MLP', '+'].map((t) => el('span', { class: t === '+' ? 'bd-mini-add' : 'bd-mini-part', text: t }))));
    if (o.b && o.b.residual && o.b.residualOk !== false && (o.b.type === 'dense' || o.b.type === 'attention')) card.append(el('span', { class: 'bd-res-arc', title: 'Residual: the block’s input is added to its output' }));
    if (o.extra) card.append(o.extra);
    if (o.problem) {
      const fix = o.problem.fix;
      card.append(el('div', { class: 'bd-problem' }, el('span', { text: o.problem.reason }),
        fix ? el('button', { class: 'btn btn-quiet bd-fix', type: 'button', onclick: (e) => { e.stopPropagation(); this.fixBlock(o.kind === 'output' ? this.setup.blocks.length : o.i, fix); } }, `Add ${fix.map(blockLabel).join(' + ')}`) : null));
    }
    card.addEventListener('pointerdown', (e) => { if (e.target.closest('button, select, .bd-lane')) return; this.pressStart(e, { kind: io ? o.kind : 'block', id: o.id, el: card }); });
    card.addEventListener('keydown', (e) => {
      if (e.key === 'Enter' || e.key === ' ') { e.preventDefault(); this.selectBlock({ kind: o.kind, id: o.id }); }
      else if (!io && (e.key === 'Delete' || e.key === 'Backspace')) { e.preventDefault(); this.refocus(null, this.removeBlock(o.id)); }
      else if (!io && e.altKey && (e.key === 'ArrowUp' || e.key === 'ArrowDown')) { e.preventDefault(); const i = this.setup.blocks.findIndex((b) => b.id === o.id); this.moveBlock(o.id, e.key === 'ArrowUp' ? i - 1 : i + 2); this.refocus(o.id); }
    });
    return card;
  };
  // a copy of one of your parts: one card, its blocks as chips (click one to change it, in every copy)
  BL.groupCard = function (run, chk) {
    const S = this.setup, def = defOf(S, run.part) || { name: 'Your part' }, inner = S.blocks.slice(run.from, run.to), sh = (chk.shapes || []).slice(run.from, run.to);
    const params = sh.reduce((s, q) => s + ((q && q.params) || 0), 0), bad = sh.find((q) => q && q.problem), copies = runsOf(S).filter((g) => g.part === run.part).length;
    const card = el('div', { class: `bd-card bd-block bd-groupcard${this.sel && this.sel.id === run.grp ? ' is-selected' : ''}${bad ? ' is-bad' : ''}`, style: { '--bk': 'var(--ink-2)' }, dataset: { bid: run.grp }, tabindex: '0', role: 'button', title: `${def.name}, a part of your own. Click it to rename it, add copies or take it apart; click a block inside to change it in every copy.`, 'aria-label': `${def.name}: your part, ${inner.length} blocks` });
    const head = el('div', { class: 'bd-card-head' }, el('span', { class: 'bd-card-ico' }, icon('part')), el('span', { class: 'bd-card-text' }, el('span', { class: 'bd-card-title', text: def.name }), el('span', { class: 'bd-card-sum', text: `your part · ${inner.length} block${inner.length === 1 ? '' : 's'}${copies > 1 ? ` · ${copies} copies` : ''}` })));
    if (params) head.append(el('span', { class: 'bd-card-params', title: 'learned numbers in this copy', text: fmtInt(params) }));
    head.append(el('button', { class: 'bd-card-x', type: 'button', title: 'Remove this copy (Delete)', 'aria-label': `Remove ${def.name}`, onclick: (e) => { e.stopPropagation(); this.removeGroup(run.grp); } }, '×'));
    card.append(head, el('div', { class: 'bd-gchips' }, ...inner.map((b, k) => el('button', { class: 'bd-gchip' + (this.sel && this.sel.id === b.id ? ' is-selected' : '') + (sh[k] && sh[k].problem ? ' is-bad' : ''), type: 'button', style: { '--bk': `var(--${toneOf(b.type)})` }, dataset: { bid: b.id }, title: `${blockLabel(b)}${sh[k] && sh[k].problem ? `: ${sh[k].problem.reason}` : ''}. ${Click()} to change it (in every copy).`,
      onclick: (e) => { e.stopPropagation(); this.selectBlock({ kind: 'block', id: b.id }); } }, el('span', { class: 'bd-card-ico' }, icon(b.type === 'dense' && b.units === 1 ? 'neuron' : b.type)), el('span', { class: 'bd-gchip-name', text: blockSummaryShort(b) })))));
    if (bad) card.append(el('div', { class: 'bd-problem' }, el('span', { text: bad.problem.reason })));
    card.addEventListener('pointerdown', (e) => { if (e.target.closest('button')) return; this.pressStart(e, { kind: 'group', id: run.grp, el: card }); });
    card.addEventListener('keydown', (e) => {
      if (e.target !== card) return;
      if (e.key === 'Enter' || e.key === ' ') { e.preventDefault(); this.selectBlock({ kind: 'group', id: run.grp }); }
      else if (e.key === 'Delete' || e.key === 'Backspace') { e.preventDefault(); this.removeGroup(run.grp); }
      else if (e.altKey && (e.key === 'ArrowUp' || e.key === 'ArrowDown')) { e.preventDefault(); this.moveGroup(run.grp, e.key === 'ArrowUp' ? run.from - 1 : run.to + 1); this.refocus(run.grp); }
    });
    return card;
  };

  // ------------------------------------------------------------------ parallel branches
  // a parallel block's branches on the model: one lane each, its blocks as chips, a list to add one, how they meet
  BL.lanesEl = function (b, sh) {
    const lanes = b.branches || [], brs = sh.branches || [], fits = PARTS.filter((p) => !p.classic && p.type !== 'parallel' && this.fitsHere(p.type));
    const wrap = el('div', { class: 'bd-lanes', style: { '--n': String(lanes.length) } });
    lanes.forEach((list, br) => {
      const shs = brs[br] || [], out = shs.length ? shs[shs.length - 1].out : sh.in, bad = shs.findIndex((q) => q && q.problem);
      const add = UI.select([{ value: '', label: '+ Add a block' }, ...fits.map((p) => ({ value: p.type, label: p.label }))], '', (v) => { if (v) this.addToLane(b.id, br, v); }, { class: 'bd-lane-add', 'aria-label': `Add a block to branch ${LETTERS[br]}` });
      wrap.append(el('div', { class: 'bd-lane' + (bad >= 0 ? ' is-bad' : ''), dataset: { owner: b.id, br: String(br) }, title: `Branch ${LETTERS[br]}: drop a part here to add it to this branch.` },
        el('div', { class: 'bd-lane-head' }, el('span', { class: 'bd-lane-letter', text: LETTERS[br] }), el('span', { class: 'bd-lane-shape', text: out ? NA.bx.shapeText(out) : '' })),
        ...list.map((q, j) => el('button', { class: 'bd-gchip' + (this.sel && this.sel.id === q.id ? ' is-selected' : '') + (shs[j] && shs[j].problem ? ' is-bad' : ''), type: 'button', style: { '--bk': `var(--${toneOf(q.type)})` }, dataset: { bid: q.id }, title: `${blockLabel(q)}${shs[j] && shs[j].problem ? `: ${shs[j].problem.reason}` : ''}. ${Click()} to change it.`,
          onclick: (e) => { e.stopPropagation(); this.selectBlock({ kind: 'block', id: q.id }); } }, el('span', { class: 'bd-card-ico' }, icon(q.type === 'dense' && q.units === 1 ? 'neuron' : q.type)), el('span', { class: 'bd-gchip-name', text: blockSummaryShort(q) }))),
        list.length ? null : el('div', { class: 'bd-lane-empty', text: 'empty: passes the input through' }),
        bad >= 0 && shs[bad].problem.fix ? el('button', { class: 'btn btn-quiet bd-fix', type: 'button', onclick: (e) => { e.stopPropagation(); this.fixInLane(b.id, br, bad, shs[bad].problem.fix); } }, `Add ${shs[bad].problem.fix.map(blockLabel).join(' + ')}`) : null,
        add));
    });
    return el('div', { class: 'bd-lanes-wrap' }, wrap, el('div', { class: 'bd-merge' }, el('span', { class: 'bd-merge-sign', text: MERGE_SIGN[b.merge || 'concat'] }), el('span', { text: `${MERGE_TEXT[b.merge || 'concat']}${sh.out ? ` → ${NA.bx.shapeText(sh.out)}` : ''}` })));
  };
  // a change to a parallel block (its branches, how they meet): in every copy too, when it sits in one of your parts
  BL.editOwner = function (ownerId, fn, o) { this.edit((s) => { const f = findBlock(s, ownerId); if (!f || f.owner) return; fn(f.b, s); if (f.b.grp) syncPart(s, f.b); }, o || { flip: true }); };
  BL.addToLane = function (ownerId, br, type) {
    const p = PART[type], f = findBlock(this.setup, ownerId);
    if (!p || p.classic || p.custom || type === 'parallel' || !f || !f.b.branches || !f.b.branches[br]) return;
    if (f.b.branches[br].length >= 8) { NA.ui.toast.show('A branch holds up to 8 blocks.'); return; }
    const q = Object.assign({ id: bid() }, p.make());
    this.editOwner(ownerId, (o) => { o.branches[br].push(q); });
    this.selectBlock({ kind: 'block', id: q.id }, true);
  };
  // a block dragged from the model onto a branch moves into it
  BL.laneDroppable = function (src) {
    if (src.kind === 'part') return !src.part.custom && !src.part.classic && src.part.type !== 'parallel';
    if (src.kind !== 'block') return false;
    const b = this.setup.blocks.find((q) => q.id === src.id);
    return !!b && b.type !== 'parallel' && !b.grp && !isClassic(b.type);
  };
  BL.moveIntoLane = function (id, ownerId, br) {
    const b = this.setup.blocks.find((q) => q.id === id), f = findBlock(this.setup, ownerId);
    if (!b || !f || f.owner || b.type === 'parallel' || b.grp || isClassic(b.type) || !f.b.branches[br]) return;
    if (f.b.branches[br].length >= 8) { NA.ui.toast.show('A branch holds up to 8 blocks.'); return; }
    this.edit((s) => { const at = s.blocks.findIndex((x) => x.id === id), [q] = s.blocks.splice(at, 1), o = s.blocks.find((x) => x.id === ownerId); o.branches[br].push(q); if (o.grp) syncPart(s, o); }, { flip: true });
    NA.ui.toast.show(`Moved ${blockName(b)} into branch ${LETTERS[br]}.`);
  };
  BL.removeFromLane = function (ownerId, br, k) {
    const f = findBlock(this.setup, ownerId), q = f && f.b.branches && f.b.branches[br] && f.b.branches[br][k];
    if (!q) return;
    this.editOwner(ownerId, (o) => { o.branches[br].splice(k, 1); });
    if (this.sel && this.sel.id === q.id) this.selectBlock({ kind: 'block', id: ownerId }, true);
    NA.ui.toast.show(`Removed ${blockName(q)} from branch ${LETTERS[br]}. Undo brings it back.`);
  };
  BL.moveInLane = function (ownerId, br, k, dir) {
    this.editOwner(ownerId, (o) => { const l = o.branches[br], j = k + dir; if (!l || j < 0 || j >= l.length) return; const t = l[k]; l[k] = l[j]; l[j] = t; });
  };
  BL.dupInLane = function (ownerId, br, k) {
    const f = findBlock(this.setup, ownerId), l = f && f.b.branches && f.b.branches[br];
    if (!l || !l[k]) return;
    if (l.length >= 8) { NA.ui.toast.show('A branch holds up to 8 blocks.'); return; }
    const q = Object.assign(cleanSpec(l[k]), { id: bid() });
    this.editOwner(ownerId, (o) => { o.branches[br].splice(k + 1, 0, q); });
    this.selectBlock({ kind: 'block', id: q.id }, true);
  };
  BL.fixInLane = function (ownerId, br, k, fix) {
    this.editOwner(ownerId, (o) => { o.branches[br].splice(k, 0, ...fix.map((x) => Object.assign({ id: bid() }, x))); });
    NA.ui.toast.show(`Added ${fix.map(blockLabel).join(' and ')} in branch ${LETTERS[br]}.`);
  };
  BL.addLane = function (ownerId) { this.editOwner(ownerId, (o) => { if (o.branches.length < 4) o.branches.push([]); }); };
  BL.removeLane = function (ownerId, br) { this.editOwner(ownerId, (o) => { if (o.branches.length > 2) o.branches.splice(br, 1); }); };
  // the parallel block's own settings: how the branches meet, and what each holds
  BL.parallelSettings = function (sec, b, sh, set) {
    const k = (b.branches || []).length, brs = sh.branches || [], merge = b.merge || 'concat';
    sec.push(UI.section('How the branches meet', UI.seg([{ value: 'concat', label: 'Join side by side' }, { value: 'add', label: 'Add' }, { value: 'mul', label: 'Multiply' }, { value: 'mean', label: 'Average' }], merge, (v) => set((q) => { q.merge = v; }), 'How the branches meet'),
      UI.note(MERGE_NOTE[merge])));
    sec.push(UI.section('Branches', ...(b.branches || []).map((list, br) => {
      const shs = brs[br] || [], out = shs.length ? shs[shs.length - 1].out : sh.in;
      return el('div', { class: 'bd-branch-row' },
        el('div', { class: 'bd-branch-head' }, el('span', { class: 'bd-lane-letter', text: LETTERS[br] }), el('span', { class: 'sec-note', text: `${list.length ? `${list.length} block${list.length === 1 ? '' : 's'}` : 'empty: passes the input through'}${out ? ` → ${NA.bx.shapeText(out)}` : ''}` }),
          k > 2 ? el('button', { class: 'btn btn-quiet', type: 'button', title: `Remove branch ${LETTERS[br]} and its blocks`, onclick: () => this.removeLane(b.id, br) }, 'Remove') : null),
        list.length ? el('div', { class: 'bd-gchips bd-gchips-col' }, ...list.map((q, j) => el('button', { class: 'bd-gchip' + (shs[j] && shs[j].problem ? ' is-bad' : ''), type: 'button', style: { '--bk': `var(--${toneOf(q.type)})` }, onclick: () => this.selectBlock({ kind: 'block', id: q.id }) },
          el('span', { class: 'bd-card-ico' }, icon(q.type === 'dense' && q.units === 1 ? 'neuron' : q.type)), el('span', { class: 'bd-gchip-name', text: `${j + 1}. ${blockLabel(q)}${shs[j] && shs[j].out ? ` → ${NA.bx.shapeText(shs[j].out)}` : ''}` })))) : null);
    }), el('div', { class: 'btn-row' }, el('button', { class: 'btn', type: 'button', disabled: k >= 4, onclick: () => this.addLane(b.id) }, '+ Add a branch')),
      UI.note(`Add blocks with a branch’s “+ Add a block” list on the model, or drag a part from the palette (or a block from the model) onto a branch. ${Click()} a block in a branch to change it.`)));
  };
  const blockSummaryShort = (b) => (b.type === 'dense' ? `${b.units === 1 ? 'Neuron' : 'Dense ' + b.units} ${ACTS[b.act].short || ACTS[b.act].label}` : blockLabel(b));
  // copies of your parts: add one, take one apart, remove one, move one; and the parts themselves: box, rename, delete
  BL.addCopy = function (partId, at) {
    if (this.sel && this.sel.kind === 'guide') this.sel = this.guideFrom || null;
    const S = this.setup, def = defOf(S, partId);
    if (!def) return;
    if (S.blocks.length + def.blocks.length > 24) { NA.ui.toast.show('That is plenty of blocks: remove one first.'); return; }
    if (at === undefined) at = this.insertAt();
    const inside = runsOf(S).find((g) => at > g.from && at < g.to);
    if (inside) at = inside.to;
    const g = gid(), classic = this.isClassicSetup(), copy = def.blocks.map((b) => freshIds(Object.assign({ id: bid() }, JSON.parse(JSON.stringify(b)), { grp: g, part: def.id })));
    this.edit((s) => { if (classic) s.blocks = []; s.blocks.splice(classic ? 0 : at, 0, ...copy); }, { flip: true });
    this.selectBlock({ kind: 'group', id: g }, true);
    NA.ui.toast.show(`Added a copy of ${def.name}${classic ? ' in place of the classic model' : ''}: same blocks, its own weights.`);
  };
  BL.removeGroup = function (grp) {
    const run = runsOf(this.setup).find((g) => g.grp === grp);
    if (!run) return;
    this.edit((s) => { s.blocks = s.blocks.filter((b) => b.grp !== grp); }, { flip: true });
    if (this.sel && (this.sel.id === grp || this.setup.blocks.every((b) => b.id !== this.sel.id))) this.selectBlock(null);
    NA.ui.toast.show('Removed the copy. Undo brings it back; the part stays in your palette.');
  };
  BL.moveGroup = function (grp, to) {
    const run = runsOf(this.setup).find((g) => g.grp === grp);
    if (!run || to < 0 || to > this.setup.blocks.length || (to >= run.from && to <= run.to)) return;
    const over = runsOf(this.setup).find((g) => g.grp !== grp && to > g.from && to < g.to);
    if (over) to = to > run.from ? over.to : over.from;
    this.edit((s) => { const moved = s.blocks.splice(run.from, run.to - run.from); s.blocks.splice(to > run.from ? to - moved.length : to, 0, ...moved); }, { flip: true });
  };
  BL.unboxGroup = function (grp) {
    this.edit((s) => { for (const b of s.blocks) if (b.grp === grp) { delete b.grp; delete b.part; } }, { flip: true });
    this.selectBlock(null);
    NA.ui.toast.show('Taken apart: its blocks are ordinary blocks again. The part stays in your palette.');
  };
  BL.boxBlocks = function (i, count, name) {
    const S = this.setup, range = S.blocks.slice(i, i + count);
    if (!range.length || range.some((b) => b.grp || isClassic(b.type))) return;
    const def = { id: pid(), name: cleanName(name) || `My part ${(S.parts || []).length + 1}`, blocks: range.map(cleanSpec) }, g = gid();
    this.edit((s) => { s.parts = [...(s.parts || []), def]; for (let q = i; q < i + count; q++) Object.assign(s.blocks[q], { grp: g, part: def.id }); }, { flip: true });
    this.selectBlock({ kind: 'group', id: g }, true);
    NA.ui.toast.show(`Boxed ${count} block${count === 1 ? '' : 's'} into ${def.name}. Add copies of it from Your parts in the palette.`);
  };
  BL.renamePart = function (partId, name) {
    const n = cleanName(name);
    if (!n) return;
    this.edit((s) => { const def = defOf(s, partId); if (def) def.name = n; });
  };
  BL.deletePart = function (partId) {
    const def = defOf(this.setup, partId);
    if (!def) return;
    this.edit((s) => { s.parts = s.parts.filter((p) => p.id !== partId); for (const b of s.blocks) if (b.part === partId) { delete b.grp; delete b.part; } });
    this.selectBlock(null);
    NA.ui.toast.show(`Deleted ${def.name} from your parts. Its copies in the model stay, as ordinary blocks. Undo brings it back.`);
  };
  // a dense layer's neurons as dots: click one to give it its own activation; − and + remove and add neurons
  BL.neuronRow = function (b) {
    const row = el('div', { class: 'bd-neurons' }), show = Math.min(b.units, 32);
    for (let j = 0; j < show; j++) {
      const a = b.acts ? b.acts[j] : b.act, sel = this.sel && this.sel.id === b.id && this.sel.neuron === j;
      row.append(el('button', { class: 'bd-dot' + (sel ? ' is-selected' : '') + (b.acts && b.acts[j] !== b.act ? ' is-own' : ''), type: 'button', title: `Neuron ${j + 1}: ${ACTS[a].label}. ${Click()} to change its activation.`, 'aria-label': `Neuron ${j + 1}`, dataset: { j: String(j) },
        onclick: (e) => { e.stopPropagation(); this.selectBlock({ kind: 'block', id: b.id, neuron: j }); } }));
    }
    if (b.units > show) row.append(el('span', { class: 'bd-more', text: `+${b.units - show}` }));
    row.append(el('span', { class: 'bd-nbtns' },
      el('button', { class: 'bd-nbtn', type: 'button', title: 'Remove a neuron', 'aria-label': 'Remove a neuron', disabled: b.units <= 1, onclick: (e) => { e.stopPropagation(); this.addNeuron(b.id, -1); } }, '−'),
      el('button', { class: 'bd-nbtn', type: 'button', title: 'Add a neuron', 'aria-label': 'Add a neuron', onclick: (e) => { e.stopPropagation(); this.addNeuron(b.id, 1); } }, '+')));
    return row;
  };
  // while training, each dot shows how often its neuron is active (dim: rarely, or never: a dead neuron)
  BL.paintNeurons = function () {
    if (!this.model || this.stale || !this.stack || !this.setup.blocks.some((b) => b.type === 'dense')) return;
    const d = this.data, n = Math.min(d.nVa, Math.max(16, Math.round(800 / (d.input.kind === 'tokens' ? d.input.len : 1))));
    if (!n) return;
    const P = d.Xva.length / d.nVa;
    this.model.predict(d.Xva.subarray(0, n * P), n);
    this.setup.blocks.forEach((b, i) => {
      if (b.type !== 'dense') return;
      const lay = this.model.layers[i], card = this.stack.querySelector(`[data-bid="${b.id}"]`);
      const z = lay && (lay.z || lay.a); // each neuron's weighted sum, before its activation
      if (!z || !card) return;
      const U = b.units, rows = z.length / U;
      NA.$$('.bd-dot', card).forEach((dot) => {
        const j = +dot.dataset.j;
        let on = 0;
        for (let r = 0; r < rows; r++) if (z[r * U + j] > 0) on++;
        const f = on / Math.max(1, rows), act = ACTS[b.acts ? b.acts[j] : b.act], dead = !!act.canDie && f === 0;
        dot.style.setProperty('--on', String(f));
        dot.classList.toggle('is-dead', dead);
        dot.title = `Neuron ${j + 1}: ${act.label}. Its weighted sum is positive for ${fmtPct(f, 0)} of validation examples${dead ? ', so it never fires: a dead neuron, which has stopped learning' : ''}. ${Click()} to give it its own activation.`;
      });
    });
  };

  // ------------------------------------------------------------------ dragging (pointer events: mouse, pen and touch)
  BL.pressStart = function (e, src) {
    if (e.pointerType === 'mouse' && e.button !== 0) return;
    const start = { x: e.clientX, y: e.clientY }, pid = e.pointerId;
    let drag = null;
    const move = (ev) => {
      if (ev.pointerId !== pid) return;
      if (!drag) {
        if (Math.hypot(ev.clientX - start.x, ev.clientY - start.y) < 6 || src.kind === 'input' || src.kind === 'output') return;
        drag = this.beginDrag(src, ev);
      }
      ev.preventDefault();
      this.dragMove(drag, ev);
    };
    const up = (ev) => {
      if (ev.pointerId !== pid) return;
      window.removeEventListener('pointermove', move); window.removeEventListener('pointerup', up); window.removeEventListener('pointercancel', up);
      this._unbind = null;
      if (drag) this.dragEnd(drag, ev.type === 'pointercancel');
      else if (src.kind === 'part') this.showGuide(src.part.type);
      else this.selectBlock({ kind: src.kind, id: src.id });
    };
    window.addEventListener('pointermove', move, { passive: false }); window.addEventListener('pointerup', up); window.addEventListener('pointercancel', up);
    this._unbind = () => { window.removeEventListener('pointermove', move); window.removeEventListener('pointerup', up); window.removeEventListener('pointercancel', up); };
  };
  BL.beginDrag = function (src, ev) {
    const isPart = src.kind === 'part', r = src.el.getBoundingClientRect();
    const ghost = isPart
      ? el('div', { class: 'bd-ghost', style: { '--bk': `var(--${GROUP_TONE[src.part.group] || 'ink-2'})` } }, el('span', { class: 'bd-card-ico' }, icon(src.part.custom ? 'part' : src.part.type)), el('span', { text: src.part.name }))
      : Object.assign(src.el.cloneNode(true), { className: src.el.className + ' bd-ghost bd-ghost-card' });
    if (!isPart) { ghost.style.width = r.width + 'px'; src.el.classList.add('is-lifted'); }
    document.body.append(ghost);
    document.body.classList.add('bd-dragging');
    const drag = { src, ghost, dx: isPart ? 18 : ev.clientX - r.left, dy: isPart ? 16 : ev.clientY - r.top, target: null };
    this.drag = drag;
    return drag;
  };
  BL.dragMove = function (drag, ev) {
    drag.ghost.style.transform = `translate(${ev.clientX - drag.dx}px, ${ev.clientY - drag.dy}px) rotate(-1.5deg)`;
    const bench = this.bench.getBoundingClientRect(), x = ev.clientX, y = ev.clientY, overBench = x >= bench.left - 40 && x <= bench.right + 40 && y >= bench.top - 10 && y <= bench.bottom + 10;
    let target = null;
    if (overBench) {
      // a neuron dropped on a dense layer joins it
      if (drag.src.kind === 'part' && drag.src.part.type === 'neuron') {
        const over = NA.$$('.bd-card.bd-block', this.stack).find((c) => { const q = c.getBoundingClientRect(); return y >= q.top && y <= q.bottom && x >= q.left && x <= q.right; });
        const b = over && this.setup.blocks.find((q) => q.id === over.dataset.bid);
        if (b && b.type === 'dense') target = { kind: 'into', id: b.id, el: over };
      }
      // a part, or a block from the model, dropped on a branch of a parallel block joins that branch
      if (!target && this.laneDroppable(drag.src)) {
        const lane = NA.$$('.bd-lane', this.stack).find((l) => { const q = l.getBoundingClientRect(); return y >= q.top && y <= q.bottom && x >= q.left && x <= q.right; });
        if (lane && lane.dataset.owner !== drag.src.id) target = { kind: 'lane', id: lane.dataset.owner, br: +lane.dataset.br, el: lane };
      }
      if (!target) {
        let best = null, bd = Infinity;
        for (const g of NA.$$('.bd-gap', this.stack)) { const q = g.getBoundingClientRect(), dd = Math.abs(y - (q.top + q.height / 2)); if (dd < bd) { bd = dd; best = g; } }
        if (best) target = { kind: 'gap', at: +best.dataset.gap, el: best };
      }
      // scroll the bench when dragging near its edges
      if (y < bench.top + 30) this.bench.scrollTop -= 12; else if (y > bench.bottom - 30) this.bench.scrollTop += 12;
    } else if (drag.src.kind === 'block' || drag.src.kind === 'group') target = { kind: 'remove' };
    if (target && drag.src.kind === 'block' && target.kind === 'gap') {
      const i = this.setup.blocks.findIndex((b) => b.id === drag.src.id);
      if (target.at === i || target.at === i + 1) target = { kind: 'stay', el: target.el };
    }
    if (target && drag.src.kind === 'group' && target.kind === 'gap') {
      const run = runsOf(this.setup).find((g) => g.grp === drag.src.id);
      if (run && target.at >= run.from && target.at <= run.to) target = { kind: 'stay', el: target.el };
    }
    const key = target ? target.kind + (target.at ?? target.id ?? '') + (target.br ?? '') : '';
    if (key !== drag.key) {
      drag.key = key;
      NA.$$('.bd-gap.is-target, .bd-lane.is-target, .bd-card.is-drop-into', this.stack).forEach((n) => n.classList.remove('is-target', 'is-drop-into'));
      if (target && (target.kind === 'gap' || target.kind === 'lane')) target.el.classList.add('is-target');
      if (target && target.kind === 'into') target.el.classList.add('is-drop-into');
      drag.ghost.classList.toggle('is-remove', !!target && target.kind === 'remove');
    }
    drag.target = target;
  };
  BL.dragEnd = function (drag, cancelled) {
    const t = cancelled ? null : drag.target, src = drag.src;
    drag.ghost.remove();
    document.body.classList.remove('bd-dragging');
    NA.$$('.bd-gap.is-target, .bd-lane.is-target, .bd-card.is-drop-into, .bd-card.is-lifted', this.stack).forEach((n) => n.classList.remove('is-target', 'is-drop-into', 'is-lifted'));
    this.drag = null;
    if (!t) return;
    if (t.kind === 'lane') { if (src.kind === 'part') this.addToLane(t.id, t.br, src.part.type); else if (src.kind === 'block') this.moveIntoLane(src.id, t.id, t.br); return; }
    if (src.kind === 'part' && src.part.custom) { if (t.kind === 'gap') this.addCopy(src.part.custom, t.at); return; }
    if (src.kind === 'group') {
      if (t.kind === 'remove') this.removeGroup(src.id);
      else if (t.kind === 'gap') this.moveGroup(src.id, t.at);
      return;
    }
    if (src.kind === 'part') {
      if (t.kind === 'into') this.addNeuron(t.id, 1);
      else if (src.part.classic || this.setup.blocks.some((b) => isClassic(b.type))) this.addPart(src.part);
      else if (t.kind === 'gap') this.insert(src.part.make(), t.at);
    } else if (src.kind === 'block') {
      if (t.kind === 'remove') { this.removeBlock(src.id); NA.ui.toast.show('Removed. Undo with Ctrl+Z.'); }
      else if (t.kind === 'gap') this.moveBlock(src.id, t.at);
    }
  };
  BL.cancelDrag = function () { if (this._unbind) this._unbind(); if (this.drag) this.dragEnd(this.drag, true); };
  // animate blocks from where they were to where they are after a change (FLIP)
  BL.flip = function (mutate) {
    const before = new Map();
    if (this.stack) for (const c of NA.$$('.bd-card', this.stack)) before.set(c.dataset.bid, c.getBoundingClientRect());
    mutate();
    if (!this.stack || NA.prefersReducedMotion()) return;
    for (const c of NA.$$('.bd-card', this.stack)) {
      const b = before.get(c.dataset.bid), a = c.getBoundingClientRect();
      if (!b) { c.animate([{ opacity: 0, transform: 'translateY(-6px) scale(0.96)' }, { opacity: 1, transform: 'none' }], { duration: 240, easing: 'cubic-bezier(0.2, 0.8, 0.2, 1)' }); continue; }
      const dx = b.left - a.left, dy = b.top - a.top;
      if (Math.abs(dx) + Math.abs(dy) > 0.5) c.animate([{ transform: `translate(${dx}px, ${dy}px)` }, { transform: 'none' }], { duration: 260, easing: 'cubic-bezier(0.2, 0.8, 0.2, 1)' });
    }
  };
  // quiet: select without turning to the inspector on a phone (adding parts one after another)
  BL.selectBlock = function (sel, quiet) {
    this.sel = sel;
    if (this.stack) {
      NA.$$('.bd-card', this.stack).forEach((c) => c.classList.toggle('is-selected', !!sel && c.dataset.bid === sel.id));
      NA.$$('.bd-gchip', this.stack).forEach((c) => c.classList.toggle('is-selected', !!sel && c.dataset.bid === sel.id));
      NA.$$('.bd-dot', this.stack).forEach((dEl) => dEl.classList.toggle('is-selected', !!sel && sel.neuron === +dEl.dataset.j && dEl.closest('.bd-card').dataset.bid === sel.id));
    }
    if (sel && this.tabs && this.tabs.current !== 'block') this.tabs.select('block'); else this.inspDirty = true;
    if (sel && !quiet && NA.mobile && NA.mobile.mobile) NA.mobile.go('insp');
  };
  BL.key = function (e) {
    if (e.defaultPrevented) return; // a focused block already handled it
    if ((e.key === 'Delete' || e.key === 'Backspace') && this.sel && this.sel.kind === 'block') { e.preventDefault(); this.removeBlock(this.sel.id); }
    else if ((e.key === 'Delete' || e.key === 'Backspace') && this.sel && this.sel.kind === 'group') { e.preventDefault(); this.removeGroup(this.sel.id); }
  };
  // Ctrl+Z (Cmd+Z) undoes, Ctrl+Shift+Z or Ctrl+Y redoes
  BL.shortcut = function (e) {
    if (!(e.ctrlKey || e.metaKey) || e.altKey) return;
    if (e.key === 'z' || e.key === 'Z') { e.preventDefault(); if (e.shiftKey) this.redo(); else this.undo(); }
    else if (e.key === 'y' || e.key === 'Y') { e.preventDefault(); this.redo(); }
  };

  // ------------------------------------------------------------------ results
  BL.drawResults = function () {
    const live = this.run && !this.stale && this.model;
    this.resFoot.hidden = !live; // the loss curves belong to a run
    this.renderResHead(live); // first: the header's height decides the picture's
    const box = this.resBox, W = Math.max(200, box.clientWidth), Hh = Math.max(200, box.clientHeight), ctx = NA.fitCanvas(this.resCanvas, W, Hh);
    ctx.clearRect(0, 0, W, Hh);
    const L = { data: live ? this.data : this.dataFor(), model: live ? this.model : null, vizVersion: this.vizVersion, _region: this._region, confusion: live ? this.confusion : null, seqView: this.seqView || (this.seqView = {}) };
    const t0 = performance.now();
    this.resRect = { x: 8, y: 8, w: W - 16, h: Hh - 16 };
    BV.draw(ctx, this.resRect, L);
    this.vizMs = performance.now() - t0; this._region = L._region;
    // the run's loss curves (a diverged run's huge values are capped so the scale stays readable)
    const run = live ? this.run : null, cap = (v) => Math.min(v, 1e3);
    this.lossChart.frame();
    this.lossChart.set({ series: run ? [{ name: 'training loss', color: theme.c('s1'), points: run.curve.filter((p) => p.train !== null && Number.isFinite(p.train)).map((p) => [p.step, cap(p.train)]) }, { name: 'validation loss', color: theme.c('s2'), points: run.curve.filter((p) => p.val !== null && Number.isFinite(p.val)).map((p) => [p.step, cap(p.val)]) }] : [] });
  };
  BL.renderResHead = function (live) {
    const d = live ? this.data : this.dataFor(), p = this.problem(), run = live ? this.run : null, S = this.seqView || (this.seqView = {});
    const key = JSON.stringify([p.id, !!live, run && run.n, run && run.last && run.last.vm, run && run.last && run.last.tm, run && run.status, run && Math.floor(run.step / 250), d.family === 'sequence' && S.layer, S.head, this.drawClass, this.drawErase, this.stale, d.names, d.error, this.check && this.check.ok]);
    if (key === this._headKey) return;
    this._headKey = key;
    const parts = [];
    const score = run && run.last && run.last.vm !== null && run.last.vm !== undefined ? (BT.isValue(d.family) ? `R² ${fmt(run.last.vm, 3)}` : fmtPct(run.last.vm, 1)) : null;
    if (d.error) parts.push(el('div', { class: 'bd-res-error', role: 'alert', text: d.error }));
    parts.push(el('div', { class: 'bd-res-title' }, el('span', { class: 'bd-res-k', text: live ? `Run ${run.n} · validation ${this.metricName()}` : this.run && this.stale ? 'Setup edited: press Train for a new run' : 'Press Train to train your model' }), score ? el('span', { class: 'bd-res-v' }, el('i', { class: 'bd-run-sw', style: { background: `var(--${run.colour})` } }), score) : null));
    const lg = legendFor(d, live);
    if (lg) parts.push(lg);
    const ins = NA.builderInsights;
    if (ins) {
      // the most pressing thing about the run, in one line, with the fixes to try; the Training tab has the whole story
      const top = ins.diagnose(this)[0];
      if (top) parts.push(this.insightEl(top, true));
      const howTo = ins.read(d, live);
      if (howTo.length) {
        const det = el('details', { class: 'bd-read' }, el('summary', { text: 'How to read this' }), ...howTo.map((t) => el('p', { text: t })));
        det.open = !!NA.store.get('builder.readOpen', false);
        det.addEventListener('toggle', () => NA.store.set('builder.readOpen', det.open));
        parts.push(det);
      }
    }
    if (p.id === 'draw') {
      this.drawClass = this.drawClass || 0;
      const pick = (c, erase) => { this.drawClass = c; this.drawErase = erase; this._headKey = null; this.resultsDirty = true; };
      parts.push(el('div', { class: 'bd-draw' }, el('span', { class: 'sec-note', text: touch() ? 'Tap to place points of:' : 'Click to place points of (Shift-click removes one):' }),
        ...[0, 1, 2, 3].map((c) => el('button', { class: 'bd-classbtn' + (this.drawClass === c && !this.drawErase ? ' is-on' : ''), type: 'button', style: { '--c': `var(--${BT.CLASS_COLOURS[c]})` }, title: `Place points of class ${c + 1}`, onclick: () => pick(c, false) }, `class ${c + 1}`)),
        el('button', { class: 'bd-classbtn bd-erase' + (this.drawErase ? ' is-on' : ''), type: 'button', title: `${Click()} a point to remove it`, onclick: () => pick(this.drawClass, !this.drawErase) }, 'Erase'),
        el('button', { class: 'btn btn-quiet', type: 'button', onclick: () => { if (this.setup.custom.points.length) this.edit((s) => { s.custom.points = []; }); } }, 'Clear points')));
    }
    if (d.family === 'sequence' && live) {
      const layers = BV.attnLayers(this.model);
      if (layers.length) {
        const li = clamp(S.layer || 0, 0, layers.length - 1);
        parts.push(el('div', { class: 'bd-seq-pick' }, UI.field('Attention layer', UI.select(layers.map((l, i) => ({ value: i, label: String(i + 1) })), li, (v) => { S.layer = +v; S.head = 0; this._headKey = null; this.resultsDirty = true; })),
          UI.field('Head', UI.select(NA.range(layers[li].heads).map((h) => ({ value: h, label: String(h + 1) })), S.head || 0, (v) => { S.head = +v; this.resultsDirty = true; }))));
      }
      if (d.task === 'language' || d.task === 'sentences') {
        if (S.prompt === undefined) S.prompt = defaultPrompt(d, this.ps());
        const inp = el('input', { class: 'lm-input', type: 'text', value: S.prompt, 'aria-label': 'Prompt' });
        const go = () => { S.prompt = inp.value; this.writeText(); };
        inp.addEventListener('keydown', (e) => { if (e.key === 'Enter') go(); });
        parts.push(el('div', { class: 'lm-prompt-row' }, inp, el('button', { class: 'btn btn-primary', type: 'button', onclick: go }, 'Write')));
      }
    }
    clear(this.resHead).append(...parts);
  };
  // one finding about the run: an icon for its level, its title, what it means and buttons that try a fix
  const INS_ICON = { critical: '!', warn: '!', good: '✓', info: 'i' };
  BL.insightEl = function (it, brief) {
    const body = brief ? (it.body.match(/^.*?[.:](\s|$)/) || [it.body])[0].trim() : it.body;
    return el('div', { class: `bd-ins is-${it.level}`, role: it.level === 'critical' ? 'alert' : null },
      el('span', { class: 'bd-ins-ico', 'aria-hidden': 'true', text: INS_ICON[it.level] }),
      el('div', { class: 'bd-ins-text' }, el('strong', { class: 'bd-ins-title', text: it.title }), ' ', el('span', { class: 'bd-ins-body', text: body }),
        brief && body !== it.body ? el('button', { class: 'bd-ins-more', type: 'button', onclick: () => { this.tabs.select('train'); } }, 'Why?') : null),
      it.actions.length ? el('div', { class: 'bd-ins-acts' }, ...it.actions.map((a) => el('button', { class: 'btn btn-quiet', type: 'button', onclick: () => { a.run(); this._headKey = null; this._trainKey = null; } }, a.label))) : null);
  };
  BL.insightsSection = function () {
    const list = NA.builderInsights ? NA.builderInsights.diagnose(this) : [];
    return list.length ? UI.section('What is happening', el('div', { class: 'bd-ins-list' }, ...list.map((it) => this.insightEl(it, false)))) : null;
  };
  // what the marks in the results mean
  function legendFor(d, live) {
    if (!d || d.error) return null;
    const item = (cls, c, label) => el('span', null, el('i', { class: cls, style: c ? { '--c': `var(--${c})` } : null }), label);
    let items = [];
    if (d.family === 'series') items = [item('bd-lg-line', 's1', 'the series'), live ? item('bd-lg-line', 's2', 'one step ahead (true values fed in)') : null, live ? item('bd-lg-dash', 's7', 'on its own (its predictions fed back)') : null];
    else if (d.family === 'surface') items = [el('span', { text: 'blue high, orange low; the dots are the training points' }), live ? el('span', { text: 'error: darker where the model misses more' }) : null];
    else if (d.family === 'seqclass') items = [el('span', { text: live ? 'validation examples, with the model’s answer (✓ right, ✗ wrong)' : 'validation examples with their class' })];
    else if (d.table) {
      if (d.kind === 'classes' && live) items = [el('span', { text: `rows: the true ${d.targetName}; columns: the model’s answer. Green squares are right answers, red ones mistakes.` })];
      else if (d.kind === 'classes') items = [...d.names.slice(0, 8).map((name, k) => item('bd-lg-dot', BV.classColour(d, k), name)), d.names.length > 8 ? el('span', { text: `+${d.names.length - 8} more` }) : null, el('span', { text: d.featNames.length > 1 ? `the rows on ${d.featNames[0]} and ${d.featNames[1]}, scaled` : `the rows on ${d.featNames[0]}, scaled` })];
      else if (live) items = [item('bd-lg-ring', 's2', 'validation rows'), el('span', { text: `across: the true ${d.targetName}; up: the model’s answer. On the diagonal is right.` })];
      else items = [item('bd-lg-dot', 's1', 'training rows'), el('span', { text: `${d.targetName} against ${d.featNames[0]}` })];
    } else if (d.family === 'plane') {
      items = [...d.names.map((name, k) => item('bd-lg-dot', BV.classColour(d, k), name)), item('bd-lg-dot', null, 'training'), item('bd-lg-ring', null, 'validation')];
      if (live) items.push(el('span', { text: 'shading: the model’s answer, stronger where it is sure' }));
    } else if (BT.isValue(d.family)) items = [item('bd-lg-dot', 's1', 'training'), item('bd-lg-ring', 's2', 'validation'), live ? item('bd-lg-line', null, 'the model') : null, d.truth ? item('bd-lg-dash', null, 'your formula') : null];
    else if (d.family === 'digits' && live) items = [item('bd-lg-box', 'good', 'read right'), item('bd-lg-box', 'critical', 'read wrong'), el('span', { text: 'the small number is the model’s answer' })];
    else if (d.noHoldout) items = [el('span', { class: 'lm-warn', text: 'Scored on the training text itself: there is too little text to hold some back.' })];
    items = items.filter(Boolean);
    return items.length ? el('div', { class: 'bd-legend' }, ...items) : null;
  }
  const defaultPrompt = (d, ps) => (d.task === 'language' ? 'alice likes tea . bob likes milk . alice drinks' : BT.wordsOf(ps.text).slice(0, 3).join(' '));
  // continue the prompt with the current model (redone as it trains, so the text keeps up with it)
  BL.writeText = function () {
    const d = this.data, S = this.seqView || (this.seqView = {});
    if (!d || (d.task !== 'language' && d.task !== 'sentences')) return;
    if (S.prompt === undefined) S.prompt = defaultPrompt(d, this.ps());
    const words = d.task === 'language' ? NA.lmData.tokenize(S.prompt || '').ids : BT.wordsOf(S.prompt).filter((w) => w in d.wordId).map((w) => d.wordId[w]);
    S.promptIds = words;
    S.cont = this.model && !this.stale && words.length ? BT.continueText(this.model, words, d.input.len, d.vocab.length, 12, 0, null) : null;
    this.lastWrite = performance.now();
    this.resultsDirty = true;
  };
  BL.resultsPointer = function (e) {
    if (this.problem().id !== 'draw' || !this.resRect) return;
    const r = this.resCanvas.getBoundingClientRect(), q = BV.planeAt(this.resRect, e.clientX - r.left, e.clientY - r.top);
    if (!q.inside) return;
    const pts = this.setup.custom.points, remove = e.shiftKey || e.button === 2 || this.drawErase;
    if (remove) {
      let best = -1, bd = 0.08;
      pts.forEach((p, i) => { const dd = Math.hypot(p.x - q.x, p.y - q.y); if (dd < bd) { bd = dd; best = i; } });
      if (best >= 0) this.edit((s) => { s.custom.points.splice(best, 1); });
    } else if (pts.length < 600) this.edit((s) => { s.custom.points.push({ x: +q.x.toFixed(3), y: +q.y.toFixed(3), c: this.drawClass || 0 }); });
  };

  // ------------------------------------------------------------------ right: the inspector
  BL.buildTab = function (id, body) {
    this.tabId = id; this.parts = { body };
    if (id === 'runs') {
      const c = el('canvas');
      this.parts.chart = new NA.charts.LineChart(c, { height: 160, xLabel: 'Step', format: (v) => (this.valued() ? fmt(v, 3) : fmtPct(v, 1)), yFormat: (v) => (this.valued() ? fmt(v, 1) : fmtPct(v, 0)) });
      this.parts.chartBox = el('div', { class: 'chart-box' }, c);
      this.parts.chartNote = UI.note('Every press of Train after a change starts a new run from fresh weights, so you can compare. Tick runs to show them on the chart.');
      this.parts.list = el('div', { class: 'bd-runs' });
      this.parts.cmp = el('div', { class: 'bd-cmp' });
      body.append(UI.section('Validation score of each run', this.parts.chartBox, this.parts.chartNote), UI.section('Compare two runs', this.parts.cmp), UI.section('Runs', this.parts.list));
    } else if (id === 'python') {
      this.parts.code = el('div');
      body.append(UI.section('Your model in PyTorch', UI.note('The model, loss, optimizer and training loop you set up here, as a script you can run with PyTorch.'), this.parts.code));
    }
    this.updateTab();
  };
  BL.updateTab = function () {
    const id = this.tabId, P = this.parts;
    if (!P) return;
    if (id === 'block') this.renderBlockTab(P.body);
    else if (id === 'train') this.renderTrainTab(P.body);
    else if (id === 'runs') this.renderRunsTab();
    else if (id === 'python' && (this.pyDirty || !P.code.firstChild)) { this.pyDirty = false; clear(P.code).append(NA.builderPy ? UI.codeBox(NA.builderPy.code(this.setup, this.dataFor()), this.isClassicSetup() ? `Python · ${this.setup.blocks.some((b) => b.type === 'boost') ? 'XGBoost' : 'scikit-learn'}` : 'Python · PyTorch') : UI.note('Loading…')); }
  };

  // settings of the selected part, written as controls that edit the setup
  BL.renderBlockTab = function (body) {
    const sel = this.sel, S = this.setup;
    if (this._blockKey === JSON.stringify([sel, this.snapshot()]) && body.firstChild) return;
    this._blockKey = JSON.stringify([sel, this.snapshot()]);
    clear(body);
    if (!sel) { body.append(UI.section('How to build', UI.note(touch() ? 'Tap parts on the Parts screen to add them to the model. Tap a block to see what it does and change it, move it or remove it: the Input block holds the problem and the Output block the loss. Then press Train.' : 'Drag parts from the palette onto the model (or click them), drag blocks to reorder them, and drag one off the model to remove it. Click a block to see what it does and change it: the Input block holds the problem and the Output block the loss. Then press Train.')), UI.modelCard(this.preset, (t) => this.runTry(t))); return; }
    if (sel.kind === 'guide') { this.renderGuide(body, sel.type); return; }
    if (sel.kind === 'input') { this.renderProblemSettings(body); return; }
    if (sel.kind === 'output') { this.renderLossSettings(body); return; }
    if (sel.kind === 'group') { this.renderGroupTab(body, sel.id); return; }
    const fb = findBlock(S, sel.id), b = fb && fb.b;
    if (!b) { body.append(UI.note('That block is gone.')); return; }
    // a block inside one of your parts (or in a branch of a parallel block that is): a change changes every copy
    const set = (fn) => this.edit((s) => { const g = findBlock(s, b.id); if (!g) return; fn(g.b); const top = g.owner || g.b; if (top.grp) syncPart(s, top); });
    const i = fb.i, top = fb.owner || b, chkS = (this.check.shapes || [])[i] || {}, sh = fb.owner ? ((chkS.branches || [])[fb.br] || [])[fb.k] || {} : chkS, run = runOf(S, top), def = run ? defOf(S, run.part) : null;
    const actSel = (v, fn) => UI.select(ACT_ORDER.map((a) => ({ value: a, label: ACTS[a].label })), v, fn);
    body.append(el('div', { class: 'bd-insp-head', style: { '--bk': `var(--${toneOf(b.type)})` } }, el('span', { class: 'bd-card-ico' }, icon(b.type === 'dense' && b.units === 1 ? 'neuron' : b.type)), el('div', null, el('h3', { class: 'bd-insp-title', text: blockName(b) }), el('div', { class: 'sec-note', text: sh.in ? `${NA.bx.shapeText(sh.in)} → ${NA.bx.shapeText(sh.out)} · ${fmtInt(sh.params || 0)} learned numbers` : '' }))));
    if (sh.problem) body.append(el('div', { class: 'diag bd-warn' }, el('p', { class: 'diag-body', text: sh.problem.reason })));
    if (def) {
      const copies = runsOf(S).filter((g) => g.part === run.part).length;
      body.append(el('div', { class: 'bd-inpart' }, el('span', { class: 'bd-card-ico', style: { '--bk': 'var(--ink-2)' } }, icon('part')),
        el('span', null, `Inside your part ${def.name}: a change here changes ${copies > 1 ? `all ${copies} copies` : 'the part, and any copy you add'}.`),
        el('button', { class: 'btn btn-quiet', type: 'button', onclick: () => this.selectBlock({ kind: 'group', id: run.grp }) }, `Select ${def.name}`)));
    }
    if (fb.owner) {
      body.append(el('div', { class: 'bd-inpart' }, el('span', { class: 'bd-card-ico', style: { '--bk': 'var(--s3)' } }, icon('parallel')),
        el('span', null, `In branch ${LETTERS[fb.br]} of a parallel block (block ${fb.k + 1} of ${fb.owner.branches[fb.br].length}). It gets the same input as the other branches, and what it gives is ${MERGE_TEXT[fb.owner.merge || 'concat']} with theirs.`),
        el('button', { class: 'btn btn-quiet', type: 'button', onclick: () => this.selectBlock({ kind: 'block', id: fb.owner.id }) }, 'Select the parallel block')));
    }
    const sec = [];
    if (b.type === 'dense') {
      if (sel.neuron !== undefined && sel.neuron < b.units) {
        const j = sel.neuron, a = b.acts ? b.acts[j] : b.act;
        sec.push(UI.section(`Neuron ${j + 1}`, UI.field('Its activation', actSel(a, (v) => set((q) => { if (!q.acts) q.acts = new Array(q.units).fill(q.act); q.acts[j] = v; if (q.acts.every((x) => x === q.act)) delete q.acts; }))),
          el('div', { class: 'fx' }, el('div', { class: 'fx-line', text: 'z = Σ wᵢ·xᵢ + b   (its weighted sum)' }), el('div', { class: 'fx-line fx-strong', text: `output = ${ACTS[a].label}(z) = ${ACTS[a].formula.replace(/^.*? = /, '')}` })),
          el('div', { class: 'btn-row' }, el('button', { class: 'btn', type: 'button', onclick: () => this.selectBlock({ kind: 'block', id: b.id }) }, 'Back to the layer'), el('button', { class: 'btn btn-quiet', type: 'button', disabled: b.units <= 1, onclick: () => { set((q) => { q.units--; if (q.acts) { q.acts.splice(j, 1); if (q.acts.every((x) => x === q.act)) delete q.acts; } }); this.selectBlock({ kind: 'block', id: b.id }); } }, 'Remove this neuron'))));
      }
      const resOk = residualFits(b, sh);
      sec.push(UI.section('Layer', UI.slider({ label: 'Neurons', min: 1, max: 256, step: 1, value: b.units, format: (v) => String(v), onChange: (v) => set((q) => { q.units = v; if (q.acts) { q.acts = q.acts.slice(0, v); while (q.acts.length < v) q.acts.push(q.act); } }) }),
        UI.field('Activation', actSel(b.act, (v) => set((q) => { q.act = v; delete q.acts; }))),
        el('div', { class: 'chip-row' }, UI.chip('Residual (add the input back)', !!b.residual, (on) => set((q) => { q.residual = on; }), 'Only works when the layer keeps the size of its input')),
        b.residual && !resOk ? UI.note(`The residual is off for now: it needs as many neurons as the layer has inputs (${sh.in[sh.in.length - 1]}).`, 'lm-warn') : null,
        b.acts ? UI.note('Some neurons have their own activation (outlined dots). Choosing one activation for the layer resets them.') : fb.owner ? null : UI.note(`${Click()} a neuron dot on the block to give that neuron its own activation.`)));
      sec.push(UI.section('The activation', el('div', { class: 'fx' }, el('div', { class: 'fx-line fx-strong', text: `${ACTS[b.act].label}: ${ACTS[b.act].formula}` }), el('div', { class: 'fx-line', text: `slope: ${ACTS[b.act].dformula}` }))));
    } else if (b.type === 'parallel') { this.parallelSettings(sec, b, sh, set);
    } else if (b.type === 'activation') sec.push(UI.section('Settings', UI.field('Function', actSel(b.act, (v) => set((q) => { q.act = v; }))), el('div', { class: 'fx' }, el('div', { class: 'fx-line', text: `${ACTS[b.act].label}: ${ACTS[b.act].formula}` }))));
    else if (b.type === 'dropout') sec.push(UI.section('Settings', UI.slider({ label: 'Fraction dropped while training', min: 0, max: 0.7, step: 0.05, value: b.p, format: (v) => fmtPct(v, 0), onChange: (v) => set((q) => { q.p = v; }) })));
    else if (b.type === 'attention' || b.type === 'transformer') {
      const D = sh.in ? sh.in[sh.in.length - 1] : 32, heads = [1, 2, 4, 8].filter((h) => D % h === 0);
      sec.push(UI.section('Settings', UI.field('Heads', UI.seg(heads.map((h) => ({ value: h, label: String(h) })), b.heads, (v) => set((q) => { q.heads = +v; }), 'Heads')),
        b.type === 'transformer' ? UI.field('MLP size', UI.seg([1, 2, 4].map((m) => ({ value: m, label: `×${m}` })), b.mlp || 4, (v) => set((q) => { q.mlp = +v; }), 'MLP size')) : null,
        el('div', { class: 'chip-row' }, UI.chip('Causal (only look back)', b.causal !== false, (on) => set((q) => { q.causal = on; }), 'Each position may only look at earlier positions'),
          b.type === 'attention' ? UI.chip('Residual', b.residual !== false, (on) => set((q) => { q.residual = on; })) : null),
        b.causal === false ? el('div', { class: 'diag bd-warn' }, el('p', { class: 'diag-body', text: 'Without the causal mask every position can see the answer it is asked to predict. Training will look excellent, but writing an answer one token at a time (as the results do) will fail: try it.' })) : null,
        UI.note(`${D} numbers per position, split into ${b.heads} head${b.heads > 1 ? 's' : ''} of ${D / b.heads}.`)));
    } else if (b.type === 'conv') {
      sec.push(UI.section('Settings', UI.slider({ label: 'Filters', min: 1, max: 32, step: 1, value: b.filters, format: String, onChange: (v) => set((q) => { q.filters = v; }) }),
        UI.field('Filter size', UI.seg([{ value: 1, label: '1×1' }, { value: 3, label: '3×3' }, { value: 5, label: '5×5' }], b.k, (v) => set((q) => { q.k = +v; }), 'Filter size')),
        UI.field('Stride', UI.seg([{ value: 1, label: '1' }, { value: 2, label: '2 (halves it)' }], b.stride, (v) => set((q) => { q.stride = +v; }), 'Stride')),
        UI.field('Activation', actSel(b.act, (v) => set((q) => { q.act = v; })))));
    } else if (b.type === 'pool' || b.type === 'avgpool') sec.push(UI.section('Settings', UI.field('Patch', UI.seg([{ value: 2, label: '2×2' }, { value: 3, label: '3×3' }], b.size, (v) => set((q) => { q.size = +v; }), 'Patch'))));
    else if (b.type === 'gated') sec.push(UI.section('Settings', UI.slider({ label: 'Units', min: 1, max: 128, step: 1, value: b.units, format: String, onChange: (v) => set((q) => { q.units = v; }) }),
      UI.field('Gate', UI.seg([{ value: 'sigmoid', label: 'GLU (sigmoid)' }, { value: 'swish', label: 'SwiGLU' }, { value: 'gelu', label: 'GEGLU' }], b.gate, (v) => set((q) => { q.gate = v; }), 'Gate'))));
    else if (b.type === 'noise') sec.push(UI.section('Settings', UI.slider({ label: 'Spread σ while training', min: 0, max: 1, step: 0.05, value: b.sigma, format: (v) => fmt(v, 2), onChange: (v) => set((q) => { q.sigma = v; }) })));
    else if (b.type === 'fourier') sec.push(UI.section('Settings', UI.slider({ label: 'Frequencies', min: 1, max: 64, step: 1, value: b.features, format: String, onChange: (v) => set((q) => { q.features = v; }) }),
      UI.slider({ label: 'Frequency spread σ', min: 0.25, max: 10, step: 0.25, value: b.scale, format: (v) => fmt(v, 2), onChange: (v) => set((q) => { q.scale = v; }) }),
      UI.note('Higher σ: faster wiggles. Too high and the model fits noise; too low and it is no help.')));
    else if (b.type === 'rnn') sec.push(UI.section('Settings', UI.field('Cell', UI.seg([{ value: 'rnn', label: 'RNN' }, { value: 'lstm', label: 'LSTM' }, { value: 'gru', label: 'GRU' }], b.cell, (v) => set((q) => { q.cell = v; }), 'Cell')),
      UI.slider({ label: 'Units (size of the state)', min: 1, max: 128, step: 1, value: b.units, format: String, onChange: (v) => set((q) => { q.units = v; }) }),
      UI.note('The layer outputs its state at every step. Follow it with Sequence pool (last step) for one answer per sequence.')));
    else if (b.type === 'conv1d') sec.push(UI.section('Settings', UI.slider({ label: 'Filters', min: 1, max: 64, step: 1, value: b.filters, format: String, onChange: (v) => set((q) => { q.filters = v; }) }),
      UI.field('Window', UI.seg([2, 3, 5, 7].map((k) => ({ value: k, label: String(k) })), b.k, (v) => set((q) => { q.k = +v; }), 'Window')),
      UI.field('Dilation', UI.seg([1, 2, 4, 8].map((k) => ({ value: k, label: String(k) })), b.dilation || 1, (v) => set((q) => { q.dilation = +v; }), 'Dilation')),
      UI.field('Activation', actSel(b.act, (v) => set((q) => { q.act = v; }))),
      el('div', { class: 'chip-row' }, UI.chip('Causal (only look back)', !!b.causal, (on) => set((q) => { q.causal = on; })), UI.chip('Residual', !!b.residual, (on) => set((q) => { q.residual = on; }), 'Adds the input back when the filters match its features'))));
    else if (b.type === 'position') sec.push(UI.section('Settings', UI.field('Kind', UI.seg([{ value: 'learned', label: 'Learned' }, { value: 'sine', label: 'Sine waves (fixed)' }], b.mode, (v) => set((q) => { q.mode = v; }), 'Kind'))));
    else if (b.type === 'seqpool') sec.push(UI.section('Settings', UI.field('Keep', UI.seg([{ value: 'last', label: 'Last step' }, { value: 'max', label: 'Largest' }, { value: 'mean', label: 'Average' }], b.mode, (v) => set((q) => { q.mode = v; }), 'Keep'))));
    else if (b.type === 'patches') sec.push(UI.section('Settings', UI.field('Patch size', UI.seg([1, 2, 3, 7].map((k) => ({ value: k, label: `${k}×${k}` })), b.size, (v) => set((q) => { q.size = +v; }), 'Patch size')),
      UI.slider({ label: 'Numbers per patch (token width)', min: 4, max: 64, step: 4, value: b.dim, format: String, onChange: (v) => set((q) => { q.dim = v; }) }),
      sh.out ? UI.note(`${sh.out[0]} patches, each a token of ${sh.out[1]} numbers. Attention heads must divide ${sh.out[1]}.`) : null));
    else if (b.type === 'linear') sec.push(UI.section('Settings', UI.field('Ridge penalty λ (shrinks the weights)', UI.select([0, 0.01, 0.1, 1, 10].map((v) => ({ value: v, label: v ? String(v) : 'none' })), b.lambda, (v) => set((q) => { q.lambda = +v; })))));
    else if (b.type === 'knn') sec.push(UI.section('Settings', UI.slider({ label: 'Neighbours k', min: 1, max: 25, step: 1, value: b.k, format: String, onChange: (v) => set((q) => { q.k = v; }) }),
      UI.field('Votes', UI.seg([{ value: 'uniform', label: 'Equal' }, { value: 'distance', label: 'Closer counts more' }], b.weights, (v) => set((q) => { q.weights = v; }), 'Votes'))));
    else if (b.type === 'tree') sec.push(UI.section('Settings', UI.slider({ label: 'Maximum depth', min: 1, max: 16, step: 1, value: b.depth, format: String, onChange: (v) => set((q) => { q.depth = v; }) }),
      UI.slider({ label: 'Fewest examples in a leaf', min: 1, max: 30, step: 1, value: b.minLeaf, format: String, onChange: (v) => set((q) => { q.minLeaf = v; }) })));
    else if (b.type === 'forest') sec.push(UI.section('Settings', UI.slider({ label: 'Trees', min: 1, max: 200, step: 1, value: b.trees, format: String, onChange: (v) => set((q) => { q.trees = v; }) }),
      UI.slider({ label: 'Maximum depth', min: 1, max: 16, step: 1, value: b.depth, format: String, onChange: (v) => set((q) => { q.depth = v; }) }),
      UI.field('Inputs tried at each question', UI.seg([{ value: 'sqrt', label: '√ of all' }, { value: 'third', label: 'a third' }, { value: 'all', label: 'all' }], b.features, (v) => set((q) => { q.features = v; }), 'Inputs tried')),
      UI.slider({ label: 'Fewest examples in a leaf', min: 1, max: 30, step: 1, value: b.minLeaf, format: String, onChange: (v) => set((q) => { q.minLeaf = v; }) })));
    else if (b.type === 'boost') sec.push(UI.section('Settings', UI.slider({ label: 'Rounds (trees)', min: 5, max: 500, step: 5, value: b.rounds, format: String, onChange: (v) => set((q) => { q.rounds = v; }) }),
      UI.field('Learning rate η', UI.select([0.01, 0.03, 0.1, 0.3, 1].map((v) => ({ value: v, label: String(v) })), b.eta, (v) => set((q) => { q.eta = +v; }))),
      UI.slider({ label: 'Tree depth', min: 1, max: 8, step: 1, value: b.depth, format: String, onChange: (v) => set((q) => { q.depth = v; }) }),
      UI.field('Leaf penalty λ', UI.select([0, 0.1, 1, 10].map((v) => ({ value: v, label: String(v) })), b.lambda, (v) => set((q) => { q.lambda = +v; }))),
      UI.field('Split penalty γ', UI.select([0, 0.1, 1, 5].map((v) => ({ value: v, label: String(v) })), b.gamma, (v) => set((q) => { q.gamma = +v; }))),
      UI.field('Rows per tree', UI.select([0.5, 0.8, 1].map((v) => ({ value: v, label: fmtPct(v, 0) })), b.subsample, (v) => set((q) => { q.subsample = +v; }))),
      UI.field('Inputs per tree', UI.select([0.5, 0.8, 1].map((v) => ({ value: v, label: fmtPct(v, 0) })), b.colsample, (v) => set((q) => { q.colsample = +v; })))));
    if (b.type === 'conv') sec.push(el('div', { class: 'chip-row' }, UI.chip('Residual (ResNet-style)', !!b.residual, (on) => set((q) => { q.residual = on; }), 'Adds the input back: needs as many filters as input channels and stride 1')));
    if (this.model && this.model.classic && !this.stale && isClassic(b.type)) sec.push(UI.section('Trained so far', UI.note(`${fmtInt(this.model.rounds)} ${this.model.unit()}${this.model.rounds === 1 ? '' : 's'} · ${fmtInt(this.model.paramCount)} learned numbers. Zoom into it in the Architecture view to see ${b.type === 'knn' ? 'its neighbours' : b.type === 'linear' ? 'its weights' : 'its trees'}.`)));
    const gd = NA.partGuide && NA.partGuide.G[b.type === 'dense' && b.units === 1 ? 'neuron' : b.type], hereText = NA.partGuide && NA.partGuide.inModel ? NA.partGuide.inModel(b, sh) : '';
    sec.push(UI.section('What it does', ...(gd && gd.deep ? gd.deep.map((t) => el('p', { class: 'sec-note bd-deep', text: t })) : [UI.note(ABOUT[b.type] || '')]),
      hereText ? el('div', { class: 'bd-here' }, el('div', { class: 'bd-here-title', text: 'Here, in your model' }), el('p', { text: hereText })) : null));
    if (NA.partGuide) sec.push(NA.partGuide.details(b.type === 'dense' && b.units === 1 ? 'neuron' : b.type, { open: false }));
    if (fb.owner) {
      const o = fb.owner, n = o.branches[fb.br].length;
      sec.push(el('div', { class: 'btn-row bd-block-actions' },
        el('button', { class: 'btn', type: 'button', disabled: fb.k === 0, title: 'Move it up in its branch', onclick: () => this.moveInLane(o.id, fb.br, fb.k, -1) }, '↑ Move up'),
        el('button', { class: 'btn', type: 'button', disabled: fb.k === n - 1, title: 'Move it down in its branch', onclick: () => this.moveInLane(o.id, fb.br, fb.k, 1) }, '↓ Move down'),
        el('button', { class: 'btn', type: 'button', onclick: () => this.dupInLane(o.id, fb.br, fb.k) }, 'Duplicate'),
        el('button', { class: 'btn', type: 'button', onclick: () => this.removeFromLane(o.id, fb.br, fb.k) }, 'Remove from the branch')));
    } else if (run) {
      const k = i - run.from, n = run.to - run.from;
      sec.push(el('div', { class: 'btn-row bd-block-actions' },
        el('button', { class: 'btn', type: 'button', disabled: k === 0, title: 'Move it up inside the part (Alt+↑)', onclick: () => this.moveBlock(b.id, i - 1) }, '↑ Move up'),
        el('button', { class: 'btn', type: 'button', disabled: k === n - 1, title: 'Move it down inside the part (Alt+↓)', onclick: () => this.moveBlock(b.id, i + 2) }, '↓ Move down'),
        el('button', { class: 'btn', type: 'button', title: 'A second one right after it, in every copy', onclick: () => this.edit((s) => editPart(s, run.part, (list, g) => list.splice((g ? g.from : 0) + k + 1, 0, g ? Object.assign({ id: bid() }, cleanSpec(b), { grp: g.grp, part: run.part }) : cleanSpec(b))), { flip: true }) }, 'Duplicate'),
        el('button', { class: 'btn', type: 'button', onclick: () => this.removeBlock(b.id) }, 'Remove from the part')));
    } else {
      sec.push(el('div', { class: 'btn-row bd-block-actions' },
        el('button', { class: 'btn', type: 'button', disabled: i === 0, title: 'Move this block up (Alt+↑)', onclick: () => this.moveBlock(b.id, i - 1) }, '↑ Move up'),
        el('button', { class: 'btn', type: 'button', disabled: i === S.blocks.length - 1, title: 'Move this block down (Alt+↓)', onclick: () => this.moveBlock(b.id, i + 2) }, '↓ Move down'),
        el('button', { class: 'btn', type: 'button', onclick: () => { const c = JSON.parse(JSON.stringify(b)); delete c.id; this.insert(c, i + 1); } }, 'Duplicate'),
        el('button', { class: 'btn', type: 'button', onclick: () => this.removeBlock(b.id) }, 'Remove')));
      // box this block, and the ones after it, into a part of your own
      if (!isClassic(b.type)) {
        let limit = 0;
        for (let q = i; q < S.blocks.length && !S.blocks[q].grp && !isClassic(S.blocks[q].type) && limit < 12; q++) limit++;
        let count = Math.min(2, limit);
        const nameIn = el('input', { class: 'bd-name', type: 'text', value: `My part ${(S.parts || []).length + 1}`, maxlength: '32', 'aria-label': 'Name of the new part' });
        sec.push(UI.section('Make it a part of your own', UI.note('Box this block, and the ones after it if you like, into a part with a name. Like a function, you can then add it again and again from the palette (Your parts): each copy has its own weights, and changing a block inside one changes them all.'),
          UI.field('Blocks to box', UI.select(NA.range(limit).map((q) => ({ value: q + 1, label: q ? `this one and the next ${q}` : 'just this one' })), count, (v) => { count = +v; })),
          UI.field('Name', nameIn),
          el('div', { class: 'btn-row' }, el('button', { class: 'btn btn-primary', type: 'button', onclick: () => this.boxBlocks(i, count, nameIn.value) }, 'Box them'))));
      }
    }
    body.append(...sec);
  };
  // a copy of one of your parts, selected: its name, what is inside, and its copies
  BL.renderGroupTab = function (body, grp) {
    const S = this.setup, run = runsOf(S).find((g) => g.grp === grp), def = run && defOf(S, run.part);
    if (!run || !def) { body.append(UI.note('That part is gone.')); return; }
    const chk = this.check || this.recheck(), sh = (chk.shapes || []).slice(run.from, run.to), params = sh.reduce((s, q) => s + ((q && q.params) || 0), 0), copies = runsOf(S).filter((g) => g.part === def.id).length;
    const nameIn = el('input', { class: 'bd-name', type: 'text', value: def.name, maxlength: '32', 'aria-label': 'Name of the part' });
    nameIn.addEventListener('change', () => this.renamePart(def.id, nameIn.value));
    nameIn.addEventListener('keydown', (e) => { if (e.key === 'Enter') nameIn.blur(); });
    body.append(el('div', { class: 'bd-insp-head', style: { '--bk': 'var(--ink-2)' } }, el('span', { class: 'bd-card-ico' }, icon('part')), el('div', null, el('h3', { class: 'bd-insp-title', text: def.name }), el('div', { class: 'sec-note', text: `${sh[0] && sh[0].in && sh[sh.length - 1].out ? `${NA.bx.shapeText(sh[0].in)} → ${NA.bx.shapeText(sh[sh.length - 1].out)} · ` : ''}${fmtInt(params)} learned numbers in this copy${copies > 1 ? ` · ${copies} copies in the model` : ''}` }))),
      UI.section('Your part', UI.note('Blocks boxed together under one name, like a function you can call again: add more copies from the palette (Your parts) or with the button below. Each copy has its own weights, so two copies can learn different things. Changing a block inside changes every copy, the way editing a function changes every call of it.'), UI.field('Name', nameIn)),
      UI.section('Inside it', el('div', { class: 'bd-gchips bd-gchips-col' }, ...S.blocks.slice(run.from, run.to).map((b, k) => el('button', { class: 'bd-gchip' + (sh[k] && sh[k].problem ? ' is-bad' : ''), type: 'button', style: { '--bk': `var(--${toneOf(b.type)})` }, onclick: () => this.selectBlock({ kind: 'block', id: b.id }) }, el('span', { class: 'bd-card-ico' }, icon(b.type === 'dense' && b.units === 1 ? 'neuron' : b.type)), el('span', { class: 'bd-gchip-name', text: `${k + 1}. ${blockLabel(b)}${sh[k] && sh[k].out ? ` → ${NA.bx.shapeText(sh[k].out)}` : ''}` })))),
        UI.note(`${Click()} a block to change it. The Architecture view shows the part as one box (a dashed box in the network view) that opens into these blocks, and the Python tab writes it as a class of its own.`)),
      UI.section('Copies', el('div', { class: 'btn-row' },
        el('button', { class: 'btn btn-primary', type: 'button', onclick: () => this.addCopy(def.id, run.to) }, 'Add another copy'),
        el('button', { class: 'btn', type: 'button', title: 'Its blocks become ordinary blocks again', onclick: () => this.unboxGroup(grp) }, 'Take this copy apart'),
        el('button', { class: 'btn', type: 'button', onclick: () => this.removeGroup(grp) }, 'Remove this copy'))),
      el('div', { class: 'btn-row bd-block-actions' },
        el('button', { class: 'btn', type: 'button', disabled: run.from === 0, title: 'Move it up (Alt+↑)', onclick: () => this.moveGroup(grp, run.from - 1) }, '↑ Move up'),
        el('button', { class: 'btn', type: 'button', disabled: run.to === S.blocks.length, title: 'Move it down (Alt+↓)', onclick: () => this.moveGroup(grp, run.to + 1) }, '↓ Move down')));
  };
  // the palette's group of your own parts: drag one onto the model, or click it to see it and add it
  const partProto = (def) => ({ type: 'part:' + def.id, group: 'Your parts', name: def.name, label: def.name, custom: def.id, hint: '', make: () => null });
  BL.renderMyParts = function () {
    if (!this.myParts) return;
    const parts = this.setup.parts || [], key = JSON.stringify(parts.map((p) => [p.id, p.name, p.blocks.map((b) => b.type)]));
    if (key === this._myKey) return;
    this._myKey = key;
    clear(this.myParts);
    this.myParts.hidden = !parts.length;
    if (!parts.length) return;
    const grid = el('div', { class: 'bd-pal-grid' });
    for (const def of parts) {
      const proto = partProto(def), tile = el('button', { class: 'bd-part bd-part-mine', type: 'button', title: `${def.name}: ${def.blocks.map(blockLabel).join(' → ')}. Drag it onto the model, or ${click()} it.`, style: { '--bk': 'var(--ink-2)' }, dataset: { type: proto.type } }, el('span', { class: 'bd-part-ico' }, icon('part')), el('span', { class: 'bd-part-name', text: def.name }));
      tile.addEventListener('pointerdown', (e) => this.pressStart(e, { kind: 'part', part: proto, el: tile }));
      tile.addEventListener('click', (e) => { if (e.detail === 0) this.showGuide(proto.type); });
      grid.append(tile);
    }
    this.myParts.append(el('div', { class: 'bd-pal-title', text: 'Your parts' }), grid);
  };
  // the guide to a part, with a button that adds it, and one that switches to a problem it suits when this one does not
  const KIND_TEXT = { plane: 'points', curve: 'a curve', surface: 'a surface', digits: 'pictures', sequence: 'sequences', seqclass: 'sequences', series: 'a series', table: 'a table' };
  BL.renderGuide = function (body, type) {
    if (String(type).startsWith('part:')) {
      const def = defOf(this.setup, type.slice(5));
      if (!def) { body.append(UI.note('That part is gone.')); return; }
      const copies = runsOf(this.setup).filter((g) => g.part === def.id).length;
      const nameIn = el('input', { class: 'bd-name', type: 'text', value: def.name, maxlength: '32', 'aria-label': 'Name of the part' });
      nameIn.addEventListener('change', () => this.renamePart(def.id, nameIn.value));
      body.append(el('div', { class: 'bd-insp-head', style: { '--bk': 'var(--ink-2)' } }, el('span', { class: 'bd-card-ico' }, icon('part')), el('div', null, el('h3', { class: 'bd-insp-title', text: def.name }), el('div', { class: 'sec-note', text: `your part · ${def.blocks.length} block${def.blocks.length === 1 ? '' : 's'} · ${copies ? `${copies} cop${copies === 1 ? 'y' : 'ies'} in the model` : 'not in the model now'}` }))),
        UI.section('Inside it', el('div', { class: 'bd-gchips bd-gchips-col' }, ...def.blocks.map((b, k) => el('span', { class: 'bd-gchip', style: { '--bk': `var(--${toneOf(b.type)})` } }, el('span', { class: 'bd-card-ico' }, icon(b.type === 'dense' && b.units === 1 ? 'neuron' : b.type)), el('span', { class: 'bd-gchip-name', text: `${k + 1}. ${blockLabel(b)}` }))))),
        UI.section('Your part', UI.note('Blocks you boxed together under one name, like a function. Each copy you add has the same blocks and its own weights.'), UI.field('Name', nameIn)),
        el('div', { class: 'btn-row' }, el('button', { class: 'btn btn-primary', type: 'button', onclick: () => this.addCopy(def.id) }, 'Add to the model'), el('button', { class: 'btn', type: 'button', onclick: () => this.deletePart(def.id) }, 'Delete this part')),
        el('div', { class: 'btn-row' }, el('button', { class: 'btn btn-quiet', type: 'button', onclick: () => this.selectBlock(this.guideFrom || null) }, '← Back')));
      return;
    }
    const p = PART[type];
    if (!p || !NA.partGuide) { body.append(UI.note('No guide for this part.')); return; }
    const fits = this.fitsHere(type), g = NA.partGuide.G[type] || {}, tryId = g.try === 'csv' ? 'csv' : g.try, tp = tryId && BT.byId[tryId];
    const add = el('button', { class: 'btn btn-primary', type: 'button', onclick: () => this.addPart(p) }, p.classic ? `Use a ${p.name.toLowerCase()}` : 'Add to the model');
    const switchTo = !fits && tp ? el('button', { class: 'btn', type: 'button', onclick: () => { this.setProblem(tryId); if (tryId === 'csv' && !this.setup.custom.table) this.setTable(BT.SAMPLES.houses.make()); this.showGuide(type); } }, `Switch to ${tp.label.toLowerCase()}`) : null;
    body.append(NA.partGuide.render(type, { part: p, fits, kind: KIND_TEXT[this.family()], add, switchTo }),
      el('div', { class: 'btn-row' }, el('button', { class: 'btn btn-quiet', type: 'button', onclick: () => this.selectBlock(this.guideFrom || null) }, '← Back')));
  };
  BL.renderProblemSettings = function (body) {
    const p = this.problem(), s = this.ps(), fam = p.family, set = (fn) => this.edit((S) => fn(S.ps[S.problem]));
    const groups = Object.keys(BT.FAMILIES).map((f) => el('optgroup', { label: BT.FAMILIES[f].label }, ...BT.PROBLEMS.filter((q) => q.family === f).map((q) => el('option', { value: q.id, text: q.label, selected: q.id === p.id }))));
    const sel = el('select', { class: 'bd-select', 'aria-label': 'Problem' }, ...groups);
    sel.addEventListener('change', () => this.setProblem(sel.value));
    const sec = [UI.section('Problem', sel, UI.note(p.hint))];
    const ctl = [];
    if (fam === 'plane' && !p.custom) ctl.push(UI.slider({ label: 'Noise', min: 0, max: 0.5, step: 0.01, value: s.noise, format: (v) => fmt(v, 2), onChange: (v) => set((q) => { q.noise = v; }) }), UI.slider({ label: 'Examples', min: 100, max: 1500, step: 50, value: s.n, format: String, onChange: (v) => set((q) => { q.n = v; }) }));
    if (p.id === 'draw') ctl.push(UI.note(`${this.setup.custom.points.length} points. Place them on the picture in the results area: pick a class and ${click()} to add, or choose Erase and ${click()} a point to remove it.`), el('div', { class: 'btn-row' }, el('button', { class: 'btn', type: 'button', onclick: () => this.edit((S) => { S.custom.points = seedPoints(); }) }, 'Start from an example'), el('button', { class: 'btn btn-quiet', type: 'button', onclick: () => this.edit((S) => { S.custom.points = []; }) }, 'Clear')));
    if (fam === 'curve') {
      if (p.custom) {
        const inp = el('input', { class: 'lm-input', type: 'text', value: s.formula, spellcheck: 'false', 'aria-label': 'Formula in x' }), msg = el('p', { class: 'sec-note' });
        const check = () => { const c = BT.checkFormula(inp.value); msg.textContent = c.error || (inp.value === s.formula ? 'In use. Edit it and press Enter to try another.' : 'Looks good: press Enter to use it.'); msg.classList.toggle('lm-warn', !!c.error); return c; };
        inp.addEventListener('input', check);
        inp.addEventListener('keydown', (e) => { if (e.key === 'Enter' && !check().error) set((q) => { q.formula = inp.value; }); });
        inp.addEventListener('change', () => { if (!check().error) set((q) => { q.formula = inp.value; }); });
        check();
        ctl.push(el('label', { class: 'field bd-formula' }, el('span', { text: 'y =' }), inp), msg, el('div', { class: 'chip-row' }, ...['sin(5x) + x^2', 'abs(x) - 0.5', 'exp(-10x^2)', 'sign(sin(8x))', 'x^3 - x'].map((f) => el('button', { class: 'chip bd-tpl chip-mono', type: 'button', onclick: () => { inp.value = f; set((q) => { q.formula = f; }); } }, f))));
      }
      ctl.push(UI.slider({ label: 'Noise', min: 0, max: 0.4, step: 0.01, value: s.noise, format: (v) => fmt(v, 2), onChange: (v) => set((q) => { q.noise = v; }) }), UI.slider({ label: 'Examples', min: 20, max: 1000, step: 10, value: s.n, format: String, onChange: (v) => set((q) => { q.n = v; }) }));
    }
    if (fam === 'digits') ctl.push(UI.slider({ label: 'Examples', min: 300, max: 4000, step: 100, value: s.n, format: String, onChange: (v) => set((q) => { q.n = v; }) }));
    if (fam === 'table') ctl.push(...tableControls(this, s));
    if (fam === 'surface') ctl.push(UI.slider({ label: 'Noise', min: 0, max: 0.3, step: 0.01, value: s.noise, format: (v) => fmt(v, 2), onChange: (v) => set((q) => { q.noise = v; }) }), UI.slider({ label: 'Examples', min: 200, max: 3000, step: 100, value: s.n, format: String, onChange: (v) => set((q) => { q.n = v; }) }));
    if (fam === 'series') ctl.push(UI.slider({ label: 'Window (values the model sees)', min: 1, max: 48, step: 1, value: s.window, format: String, onChange: (v) => set((q) => { q.window = v; }) }),
      UI.slider({ label: 'Length of the series', min: 400, max: 4000, step: 100, value: s.n, format: String, onChange: (v) => set((q) => { q.n = v; }) }),
      UI.slider({ label: 'Noise', min: 0, max: 0.3, step: 0.01, value: s.noise, format: (v) => fmt(v, 2), onChange: (v) => set((q) => { q.noise = v; }) }),
      UI.note('The first 80% of the series trains and the last 20% validates: the model is judged on a future it has not seen.'));
    if (fam === 'seqclass') ctl.push(UI.slider({ label: 'Examples', min: 500, max: 5000, step: 250, value: s.n, format: String, onChange: (v) => set((q) => { q.n = v; }) }),
      UI.field('Embedding size', UI.seg([8, 16, 32, 64].map((v) => ({ value: v, label: String(v) })), s.embed, (v) => set((q) => { q.embed = +v; }), 'Embedding size')),
      UI.note('Each token becomes a learned vector of this many numbers, plus one for its position.'));
    if (fam === 'sequence') {
      if (p.task === 'sentences') {
        const ta = el('textarea', { class: 'bd-text', rows: '7', spellcheck: 'false', 'aria-label': 'Your text' });
        ta.value = s.text;
        ta.addEventListener('change', () => set((q) => { q.text = ta.value; }));
        ctl.push(UI.note('Write anything: a few sentences with repeated patterns work best. Each word (and each . , ! ?) is a token.'), ta,
          UI.slider({ label: 'Context window (tokens)', min: 3, max: 16, step: 1, value: s.window, format: String, onChange: (v) => set((q) => { q.window = v; }) }));
      } else if (p.task !== 'language') ctl.push(UI.slider({ label: 'Sequence length', min: 2, max: 6, step: 1, value: s.len, format: String, onChange: (v) => set((q) => { q.len = v; }) }), UI.slider({ label: 'Examples', min: 500, max: 6000, step: 250, value: s.n, format: String, onChange: (v) => set((q) => { q.n = v; }) }));
      else ctl.push(UI.slider({ label: 'Stories', min: 300, max: 4000, step: 100, value: s.n, format: String, onChange: (v) => set((q) => { q.n = v; }) }));
      ctl.push(UI.field('Embedding size', UI.seg([8, 16, 32, 64].map((v) => ({ value: v, label: String(v) })), s.embed, (v) => set((q) => { q.embed = +v; }), 'Embedding size')),
        UI.note('Each token becomes a learned vector of this many numbers, plus a learned vector for its position. Attention heads split it.'));
    }
    if (ctl.length) sec.push(UI.section('Data', ...ctl));
    const d = this.dataFor();
    if (!d.error) sec.push(UI.section('What the model sees', el('div', { class: 'kv-grid' }, ...[['Input', inputText(d)], ['Output', outputText(d, this.setup.loss)], ['Training examples', fmtInt(d.nTr)], ['Validation examples', fmtInt(d.nVa)]].map(([k, v]) => el('div', { class: 'kv' }, el('span', { class: 'kv-k', text: k }), el('span', { class: 'kv-v', text: v }))))));
    body.append(...sec);
  };
  // ------------------------------------------------------------------ your own table (CSV)
  // The file is read here in the browser; it is not sent anywhere.
  BL.loadTableFile = function (f) {
    if (f.size > 3e6) { NA.ui.toast.show('That file is over 3 MB: the builder takes tables of up to 3,000 rows.'); return; }
    const rd = new FileReader();
    rd.onload = () => { const t = BT.parseCSV(rd.result, f.name); if (t.error) NA.ui.toast.show(`${f.name}: ${t.error}`); else this.setTable(t); };
    rd.onerror = () => NA.ui.toast.show(`${f.name} could not be read.`);
    rd.readAsText(f);
  };
  BL.setTable = function (t) {
    if (this.training) this.pause();
    t.stamp = Date.now().toString(36) + Math.random().toString(36).slice(2, 6);
    this.edit((S) => { S.problem = 'csv'; S.custom.table = t; S.ps.csv = { target: -1, kind: 'auto' }; fixTableLoss(S); });
    const d = this.dataFor();
    NA.ui.toast.show(d.error ? `${t.name}: ${d.error}` : `${t.name}: ${fmtInt(t.rows.length)} rows. The model will predict ${d.targetName} (${d.kind === 'value' ? 'a number' : `${d.classes} classes`}) from ${d.featNames.length} column${d.featNames.length > 1 ? 's' : ''}.${t.cut ? ` Only the first ${fmtInt(t.rows.length)} rows were kept.` : ''}`);
  };
  BL.setTableTarget = function (target, kind) { this.edit((S) => { S.ps.csv = { target, kind }; fixTableLoss(S); }); };
  // a table's loss has to suit what it predicts: classes or a number
  function fixTableLoss(S) {
    const p = BT.byId.csv, d = BT.make(p, S.ps.csv, S.train.seed, S.custom), ok = BT.lossesFor(p, d);
    if (!ok.includes(S.loss.id)) S.loss = { id: ok[0], smooth: 0, delta: 0.2 };
  }
  // the first rows of the table, with the column to predict marked
  function tablePreview(t, target) {
    const idx = NA.range(Math.min(t.header.length, 5));
    if (!idx.includes(target)) idx.push(target);
    const more = t.header.length > idx.length;
    return el('div', { class: 'bd-table-wrap', tabindex: '0', role: 'region', 'aria-label': `The first rows of ${t.name}` },
      el('table', { class: 'bd-table' },
        el('thead', null, el('tr', null, ...idx.map((i) => el('th', { class: i === target ? 'is-target' : null, scope: 'col', text: t.header[i] })), more ? el('th', { text: '…' }) : null)),
        el('tbody', null, ...t.rows.slice(0, 5).map((r) => el('tr', null, ...idx.map((i) => el('td', { class: i === target ? 'is-target' : null, text: r[i] })), more ? el('td', { text: '…' }) : null)))));
  }
  function tableControls(L, s) {
    const t = L.setup.custom.table, out = [];
    const file = el('input', { type: 'file', accept: '.csv,.tsv,.txt,text/csv,text/tab-separated-values', class: 'sr-only', tabindex: '-1', 'aria-hidden': 'true' });
    file.addEventListener('change', () => { const f = file.files && file.files[0]; if (f) L.loadTableFile(f); file.value = ''; });
    out.push(el('div', { class: 'btn-row' }, el('button', { class: 'btn' + (t ? '' : ' btn-primary'), type: 'button', onclick: () => file.click() }, t ? 'Choose another file…' : 'Choose a CSV file…'), file),
      el('div', { class: 'chip-row bd-samples' }, el('span', { class: 'sec-note', text: 'Or try a sample:' }), ...Object.values(BT.SAMPLES).map((smp) => el('button', { class: 'chip bd-tpl', type: 'button', onclick: () => L.setTable(smp.make()) }, smp.label))),
      UI.note('A table with a header row and one row per example, columns separated by commas, semicolons or tabs. The file stays on this device.'));
    if (!t) return out;
    const info = BT.tableTarget(t, s), d = L.dataFor();
    out.push(UI.note(`${t.name}: ${fmtInt(t.rows.length)} rows, ${t.header.length} columns${t.cut ? `; ${fmtInt(t.cut)} more rows were left out` : ''}.`),
      UI.field('Predict the column', UI.select(info.cols.map((c) => ({ value: c.index, label: c.name })), info.target, (v) => L.setTableTarget(+v, 'auto'))),
      UI.field('Predict it as', UI.select([{ value: 'auto', label: `automatic: ${info.kind === 'value' ? 'a number' : 'classes'}` }, { value: 'classes', label: 'classes' }, { value: 'value', label: 'a number' }], info.col.numeric ? s.kind || 'auto' : 'classes', (v) => L.setTableTarget(info.target, v), info.col.numeric ? {} : { disabled: true, title: 'A column of text can only be classes.' })));
    if (!d.error) out.push(UI.note(`Learns from ${d.featNames.length} number column${d.featNames.length > 1 ? 's' : ''}: ${d.featNames.join(', ')}.${d.ignored.length ? ` Text columns left out: ${d.ignored.join(', ')}.` : ''}${d.rowsDropped ? ` ${fmtInt(d.rowsDropped)} rows with an empty or non-number cell are skipped.` : ''} Each input column is scaled to mean 0 and spread 1${d.kind === 'value' ? ', and so is the column to predict' : ''}.`));
    out.push(tablePreview(t, info.target));
    return out;
  }
  function seedPoints() {
    const rng = new NA.RNG(5), pts = [];
    for (let i = 0; i < 60; i++) { const a = rng.uniform(0, Math.PI * 2), r = rng.uniform(0, 0.35); pts.push({ x: +(r * Math.cos(a)).toFixed(3), y: +(r * Math.sin(a)).toFixed(3), c: 0 }); }
    for (let i = 0; i < 80; i++) { const a = rng.uniform(0, Math.PI * 2), r = rng.uniform(0.6, 0.9); pts.push({ x: +(r * Math.cos(a)).toFixed(3), y: +(r * Math.sin(a)).toFixed(3), c: 1 }); }
    return pts;
  }
  BL.renderLossSettings = function (body) {
    const fam = this.family(), L = this.setup.loss, d = this.dataFor(), set = (fn) => this.edit((s) => fn(s.loss));
    const ids = BT.lossesFor(this.problem(), d), value = d.output && d.output.kind === 'value';
    const EXPLAIN = {
      xent: 'Cross-entropy: −log of the probability given to the right answer. Confident mistakes cost a lot, and the error signal is simply probability − target. The standard choice for classes and tokens.',
      mse: value ? 'Squared error: ½(prediction − target)². Big misses cost far more than small ones, so outliers pull hard.' : 'Squared error between the softmax probabilities and the one-hot target. It still learns, but slowly when the model is confidently wrong, because the softmax flattens the error signal there.',
      hinge: 'Hinge (multi-class SVM): the right class must beat every other by a margin of 1; once it does, that example stops contributing at all.',
      mae: 'Absolute error |prediction − target|: every miss pulls with the same strength, so outliers matter less and the fit follows the median.',
      huber: 'Huber: squared error for small misses, absolute error beyond δ. Smooth near the target and robust to outliers.',
    };
    body.append(el('div', { class: 'bd-insp-head', style: { '--bk': 'var(--s2)' } }, el('span', { class: 'bd-card-ico' }, icon('output')), el('div', null, el('h3', { class: 'bd-insp-title', text: 'Output and loss' }), el('div', { class: 'sec-note', text: d.error ? '' : outputText(d, L) }))),
      UI.section('Loss function', UI.seg(ids.map((id) => ({ value: id, label: NA.bx.LOSSES[id].label })), L.id, (v) => set((q) => { q.id = v; }), 'Loss'), UI.note(EXPLAIN[L.id] || ''),
        L.id === 'xent' ? UI.slider({ label: 'Label smoothing', min: 0, max: 0.3, step: 0.05, value: L.smooth || 0, format: (v) => fmt(v, 2), onChange: (v) => set((q) => { q.smooth = v; }) }) : null,
        L.id === 'huber' ? UI.slider({ label: 'Huber δ', min: 0.05, max: 1, step: 0.05, value: L.delta || 0.2, format: (v) => fmt(v, 2), onChange: (v) => set((q) => { q.delta = v; }) }) : null),
      this.isClassicSetup() ? UI.section('Classic models', UI.note('A classic model has its own way to learn: squared error for numbers; for classes, log loss (boosting and logistic regression) or the purity of each tree leaf (trees and forests). The loss chosen above is used by networks.')) : null,
      UI.section('The output layer', UI.note(value ? 'One linear neuron: any number can come out.' : fam === 'sequence' ? 'At every position, one score per token in the vocabulary; softmax turns them into the probability of each next token.' : L.id === 'hinge' ? 'One score per class, used as it is: the hinge loss asks the right class to beat the others by a margin. The added dense layer is built for you, sized to the problem.' : 'One score per class; softmax turns the scores into probabilities that add up to 1. The added dense layer is built for you, sized to the problem.')));
  };

  BL.renderTrainTab = function (body) {
    const key = JSON.stringify([this.setup.train, this.run && this.run.n, this.run && this.run.step, this.stale, this.isClassicSetup() && this.snapshot()]);
    if (key === this._trainKey && body.firstChild) { this.updateTrainLive(); return; }
    this._trainKey = key;
    if (this.isClassicSetup()) {
      const b = this.setup.blocks.find((q) => isClassic(q.type)), m = this.model && this.model.classic && !this.stale ? this.model : null, run = this.run;
      clear(body);
      this.parts.live = null;
      const story = this.insightsSection();
      if (story) body.append(story);
      body.append(el('div', { class: 'tiles' }, UI.tile('Run', run ? String(run.n) : '—', run ? (this.stale ? 'setup edited since' : run.status) : 'not trained'),
        UI.tile('Trained', m ? `${fmtInt(m.rounds)} ${m.unit()}${m.rounds === 1 ? '' : 's'}` : '—', m ? `of ${fmtInt(m.target)}` : null), UI.tile('Learned numbers', m ? fmtInt(m.paramCount) : '—'),
        UI.tile(`Val. ${this.metricName()}`, run && run.last && run.last.vm !== null && run.last.vm !== undefined && !this.stale ? scoreText(run.family, run.last.vm) : '—')),
        UI.section('How it trains', UI.note({
          linear: 'For numbers, least squares has an exact answer: one solve, no steps, no learning rate. For classes, logistic regression has none, so it takes gradient steps (300 by default; Continue takes more).',
          knn: 'It does not train: the training examples are the model. Each answer looks up the k nearest ones, which is why answering is slow and training is instant.',
          tree: 'One level per step: at each leaf, every input and threshold is tried and the question that best separates the answers is kept. It stops at the maximum depth, or when no question helps.',
          forest: 'One tree per step, each on a bootstrap sample (the training rows drawn at random with repeats) and choosing among a random few inputs at every question. Continue grows more.',
          boost: 'One round per step: the gradient and curvature of the loss at every training example, then a small tree fitted to them, scaled by η and added on. Continue adds more rounds.',
        }[b.type])),
        UI.section('Settings', UI.note('A classic model’s settings are on its block: select it in the model.')));
      return;
    }
    const T = this.setup.train, set = (fn) => this.edit((s) => fn(s.train));
    clear(body);
    const lr = UI.slider({ label: 'Learning rate', min: -4, max: 0, step: 0.05, value: Math.log10(T.lr), format: (v) => fmtLr(Math.pow(10, v)), onChange: (v) => set((t) => { t.lr = +Math.pow(10, v).toPrecision(2); }) });
    const P = this.parts;
    P.live = el('div', { class: 'tiles' });
    P.grad = el('canvas');
    const story = this.insightsSection();
    if (story) body.append(story);
    body.append(P.live,
      UI.section('Optimizer', UI.seg(Object.entries(OPT_LABEL).map(([value, label]) => ({ value, label })), T.opt, (v) => set((t) => { t.opt = v; }), 'Optimizer'),
        UI.note({ sgd: 'Plain gradient descent: step downhill by learning rate × gradient.', momentum: 'Keeps a running velocity (0.9 of the last one plus the new gradient): faster along consistent directions, damped zig-zags.', adam: 'Adapts the step for every weight from running averages of its gradient and squared gradient. Usually the easiest to tune.', adamw: 'Adam with weight decay applied directly to the weights (decoupled), the usual choice for transformers.' }[T.opt]),
        lr, UI.note('How big each step is. Too small: slow. Too big: the loss jumps around or blows up (the run is marked diverged).')),
      UI.section('Find a learning rate', P.sweep = el('div', { class: 'bd-sweep' })),
      UI.section('Batches and schedule',
        UI.field('Batch size', UI.seg([8, 16, 32, 64, 128].map((v) => ({ value: v, label: String(v) })), T.batch, (v) => set((t) => { t.batch = +v; }), 'Batch size')),
        UI.field('Steps per press of Train', UI.select(STEP_CHOICES.map(([value, label]) => ({ value, label })), T.steps, (v) => set((t) => { t.steps = +v; }))),
        UI.field('Learning-rate schedule', UI.select(Object.entries(SCHED_LABEL).map(([value, label]) => ({ value, label })), T.sched, (v) => set((t) => { t.sched = v; }))),
        UI.note(T.steps ? 'Train runs this many steps; Continue then runs as many again, so a run can go on as long as you like. Cosine and step schedules lower the learning rate over each stretch, for a finer finish.' : 'Training goes on until you pause. Schedules need a fixed number of steps, so the learning rate stays as set.')),
      UI.section('Regularisation and stability',
        UI.field('Weight decay', UI.select([0, 0.0001, 0.001, 0.01, 0.05].map((v) => ({ value: v, label: v ? String(v) : 'none' })), T.wd, (v) => set((t) => { t.wd = +v; }))),
        UI.field('Gradient clipping', UI.select([[0, 'off'], [0.5, '0.5'], [1, '1'], [5, '5']].map(([value, label]) => ({ value, label })), T.clip, (v) => set((t) => { t.clip = +v; }))),
        UI.note('Weight decay pulls weights toward 0 (simpler models, less overfitting). Clipping caps the size of a step, which rescues a learning rate that is a little too high.')),
      UI.section('Starting point',
        UI.field('Initialisation', UI.select([['auto', 'automatic (He or Xavier)'], ['he', 'He'], ['xavier', 'Xavier'], ['small', 'tiny (N(0, 0.02))']].map(([value, label]) => ({ value, label })), T.init, (v) => set((t) => { t.init = v; }))),
        UI.field('Random seed', el('span', { class: 'bd-seed' }, el('span', { class: 'mono', text: String(T.seed) }), el('button', { class: 'btn btn-quiet', type: 'button', title: 'Another random start', onclick: () => set((t) => { t.seed = 1 + Math.floor(Math.random() * 9999); }) }, 'New seed'))),
        UI.note('The same setup with another seed gives different starting weights (and shuffles): train twice to see how much luck matters.')),
      UI.section('Gradient size by layer', el('div', { class: 'chart-box' }, P.grad), UI.note('How strongly the last step changed each layer. Bars that shrink toward the input mean vanishing gradients; huge ones mean the steps are too big.')));
    this.renderSweep();
    this.updateTrainLive();
  };
  const fmtLr = (v) => (v >= 0.01 ? fmt(v, 3) : v.toExponential(1).replace('e-', ' × 10⁻').replace(/(\d)$/, (d) => '⁰¹²³⁴⁵⁶⁷⁸⁹'[d]));
  BL.updateTrainLive = function () {
    const P = this.parts, run = this.run;
    if (!P || !P.live) return;
    const p = run && run.last;
    clear(P.live).append(UI.tile('Run', run ? String(run.n) : '—', run ? (this.stale ? 'setup edited since' : run.status) : 'not trained'), UI.tile('Step', run ? fmtInt(run.step) : '0', run && run.limit && !this.stale ? `of ${fmtInt(run.limit)}` : this.setup.train.steps ? `${fmtInt(this.setup.train.steps)} a press` : 'until you pause'),
      UI.tile('Learning rate now', fmtLr(this.lrAt(run ? run.step : 0))), UI.tile(`Val. ${this.metricName()}`, p && p.vm !== null && p.vm !== undefined ? (BT.isValue(run.family) ? fmt(p.vm, 3) : fmtPct(p.vm, 1)) : '—', p && p.tm !== null && p.tm !== undefined ? `training ${BT.isValue(run.family) ? fmt(p.tm, 3) : fmtPct(p.tm, 1)}` : null));
    // gradient size of each layer, log scale
    const c = P.grad, W = Math.max(200, c.parentElement.clientWidth), Hh = 110, ctx = NA.fitCanvas(c, W, Hh);
    ctx.clearRect(0, 0, W, Hh);
    if (!this.model || this.stale) return;
    const g = this.model.gradNorms(), n = g.length, left = 34, bw = (W - left - 4) / Math.max(1, n), top = 16, base = Hh - 16;
    const lo = -6, hi = 2, Y = (v) => base - ((clamp(Math.log10(Math.max(v, 1e-12)), lo, hi) - lo) / (hi - lo)) * (base - top);
    const mono = '500 10px ' + (getComputedStyle(document.body).getPropertyValue('--font-mono') || 'monospace');
    ctx.font = mono; ctx.textAlign = 'right'; ctx.textBaseline = 'middle';
    for (const e of [-6, -4, -2, 0, 2]) { const y = Y(Math.pow(10, e)); ctx.strokeStyle = theme.c('line'); ctx.lineWidth = 1; ctx.beginPath(); ctx.moveTo(left, Math.round(y) + 0.5); ctx.lineTo(W - 4, Math.round(y) + 0.5); ctx.stroke(); ctx.fillStyle = theme.c('muted'); ctx.fillText(e === 0 ? '1' : e === 2 ? '100' : `10${'⁻'}${'⁰¹²³⁴⁵⁶⁷⁸⁹'[-e]}`, left - 5, y); }
    g.forEach((v, i) => {
      const x = left + i * bw, y = Y(v), h = Math.max(2, base - y);
      ctx.fillStyle = theme.c(i === n - 1 ? 's2' : 's1', 0.85); NA.rr(ctx, x + 3, base - h, Math.max(3, bw - 6), h, 3); ctx.fill();
      ctx.fillStyle = theme.c('muted'); ctx.textAlign = 'center'; ctx.textBaseline = 'top';
      ctx.fillText(i === n - 1 ? 'out' : String(i + 1), x + bw / 2, Hh - 13);
    });
  };

  BL.renderRunsTab = function () {
    // runs on this problem are compared; runs on others are listed, dimmed
    const P = this.parts, fam = this.family(), pid = this.setup.problem, runs = this.runs.filter((r) => r.cfg.problem === pid);
    const series = runs.filter((r) => r.show).map((r) => ({ name: `run ${r.n}`, color: theme.c(r.colour), points: r.curve.filter((p) => p.vm !== null && p.vm !== undefined).map((p) => [p.step, p.vm]) })).filter((s) => s.points.length);
    P.chartBox.hidden = !series.length;
    if (series.length) { P.chart.frame(); P.chart.set({ series, yFixed: this.valued() ? null : [0, 1] }); }
    P.chartNote.textContent = series.length ? 'Every press of Train after a change starts a new run from fresh weights, so you can compare. Tick runs to show them on the chart.' : `No runs on ${this.problem().label.toLowerCase()} yet: press Train, then change something and train again to compare.`;
    const key = JSON.stringify([pid, this.runs.map((r) => [r.n, r.status, r.show, r.last && r.last.vm, r.step, (r.events || []).length]), this.cmp]);
    if (key === this._runsKey) return;
    this._runsKey = key;
    this.renderCompare(P.cmp, runs);
    clear(P.list);
    if (!this.runs.length) { P.list.append(UI.note('No runs yet. Press Train.')); return; }
    const best = runs.reduce((b, r) => (r.last && r.last.vm !== null && r.last.vm !== undefined && (!b || r.last.vm >= b.last.vm) ? r : b), null); // ties: the newer run
    for (const r of this.runs.slice().reverse()) {
      const v = r.last && r.last.vm, vt = v === null || v === undefined ? '—' : BT.isValue(r.family) ? `R² ${fmt(v, 3)}` : fmtPct(v, 1);
      const cb = el('input', { type: 'checkbox', checked: r.show, 'aria-label': `Show run ${r.n} on the chart` });
      cb.addEventListener('change', () => { r.show = cb.checked; this._runsKey = null; this.renderRunsTab(); });
      P.list.append(el('div', { class: 'bd-run' + (r === this.run ? ' is-current' : '') + (r.cfg.problem !== pid ? ' is-other' : '') },
        el('div', { class: 'bd-run-top' }, el('label', { class: 'bd-run-pick' }, cb, el('i', { class: 'bd-run-sw', style: { background: `var(--${r.colour})` } }), el('span', { class: 'bd-run-name', text: `Run ${r.n}` })),
          r.cfg.problem !== pid ? el('span', { class: 'bd-run-prob', text: BT.byId[r.cfg.problem].label }) : null,
          el('span', { class: 'bd-run-v', text: vt }), r === best && runs.length > 1 ? UI.pill('good', 'Best') : null, r.status === 'diverged' ? UI.pill('warn', 'Diverged') : null),
        el('div', { class: 'bd-run-changes' }, ...r.changes.slice(0, 5).map((c) => el('span', { class: 'bd-change', text: c })), r.changes.length > 5 ? el('span', { class: 'bd-change', text: `+${r.changes.length - 5} more` }) : null),
        r.events && r.events.length ? el('div', { class: 'bd-run-events' }, ...r.events.slice(-4).map((e) => el('span', { class: 'bd-event', text: `step ${fmtInt(e.step)}: ${e.what.join(', ')}` })), r.events.length > 4 ? el('span', { class: 'bd-event', text: `${r.events.length - 4} earlier` }) : null) : null,
        el('div', { class: 'bd-run-meta' }, el('span', { text: `${fmtInt(r.step)} steps · ${fmtInt(r.params)} parameters · ${r.seconds < 10 ? r.seconds.toFixed(1) : Math.round(r.seconds)} s${r.last && r.last.val !== null ? ` · val. loss ${fmt(r.last.val, 4)}` : ''}` }),
          el('button', { class: 'btn btn-quiet', type: 'button', title: 'Load this run’s setup (the model, problem and settings) to change it again', onclick: () => this.restoreRun(r) }, 'Use this setup'))));
    }
  };
  BL.restoreRun = function (r) {
    if (this.training) this.pause();
    const before = this.snapshot();
    this.setup = JSON.parse(JSON.stringify(r.cfg));
    this.afterEdit(before);
    NA.ui.toast.show(`Loaded run ${r.n}’s setup. Change something and press Train.`);
  };


  // ------------------------------------------------------------------ challenges
  /* A challenge (js/builder-challenges.js) loads a starting model that falls short of a goal; you change it until a
     run reaches the goal. The active challenge and the ones completed are kept in this browser. */
  const BC = NA.builderChallenges;
  const doneSet = () => new Set(NA.store.get('builderDone', []));
  const scoreText = (fam, v) => (BT.isValue(fam) ? fmt(v, 2) : fmtPct(v, 0));
  BL.startChallenge = function (id) {
    const ch = BC.byId[id];
    if (!ch) return;
    if (this.training) this.pause();
    const S = BC.setup(ch);
    // the drawn points and every other problem's settings stay as they were; Undo brings back the whole setup
    if (S.custom.table) S.custom.table.stamp = Date.now().toString(36) + Math.random().toString(36).slice(2, 6);
    this.edit((s) => { const keep = { ps: Object.assign({}, s.ps, S.ps), custom: S.custom.table ? Object.assign({}, s.custom, { table: S.custom.table }) : s.custom, parts: s.parts || [] }; for (const k of Object.keys(s)) delete s[k]; Object.assign(s, S, keep, { blocks: withIds(S.blocks) }); }, { flip: true });
    this.challenge = id; this._hint = null;
    NA.store.set('builderChallenge', id);
    this.seqView = {};
    this.selectBlock(null, true);
    this.closeChallenges();
    this.renderChallenge();
    NA.ui.toast.show(`Challenge: ${ch.title}. The starting model is in place: press Train to see where it falls short.`);
  };
  BL.leaveChallenge = function () { this.challenge = null; NA.store.set('builderChallenge', null); this.renderChallenge(); };
  // after every measurement: has a run on the challenge's problem met every goal?
  BL.checkChallenge = function () {
    const ch = this.challenge && BC.byId[this.challenge];
    if (!ch || !this.run || !BC.passed(ch, this.run)) { this.renderChallenge(); return; }
    const done = doneSet();
    if (!done.has(ch.id)) {
      done.add(ch.id);
      NA.store.set('builderDone', [...done]);
      const msg = `Challenge complete: ${ch.title}. ${done.size} of ${BC.LIST.length} done.`;
      NA.ui.toast.show(msg);
      if (NA.a11y) NA.a11y.announce(msg);
    }
    this.renderChallenge();
  };
  BL.renderChallenge = function () {
    const root = this.chRoot;
    if (!root) return;
    const ch = this.challenge && BC.byId[this.challenge], done = doneSet();
    const fam = ch ? BC.family(ch) : null, here = ch && this.setup.problem === ch.problem;
    const run = ch && this.run && this.run.cfg.problem === ch.problem ? this.run : null, vm = run && run.last ? run.last.vm : null;
    const params = this.check && this.check.ok ? this.check.params : null, ownData = !ch || BC.sameData(ch, this.setup);
    const key = JSON.stringify([ch && ch.id, [...done], run && run.n, vm, params, this.setup.train.opt, here, this._hint, ownData, ch && ch.goals.some((g) => g.steps !== undefined) && run ? run.step : 0, this.setup.blocks.map((b) => b.type).join()]);
    if (key === this._chKey) return;
    this._chKey = key;
    clear(root);
    if (!ch) {
      root.append(el('div', { class: 'bd-ch-row' },
        el('button', { class: 'btn', type: 'button', onclick: () => this.openChallenges() }, 'Challenges'),
        el('span', { class: 'sec-note', text: done.size ? `${done.size} of ${BC.LIST.length} done` : `${BC.LIST.length} goals to reach, from beginner to expert` })));
      return;
    }
    const won = done.has(ch.id);
    const goals = ch.goals.map((g) => {
      let label, now = '', ok = false;
      if (g.metric !== undefined) {
        label = `Validation ${this.metricName(fam)} of at least ${scoreText(fam, g.metric)}`;
        now = vm === null || vm === undefined ? 'not trained yet' : `last run: ${scoreText(fam, vm)}`;
        ok = vm !== null && vm !== undefined && vm >= g.metric;
      } else if (g.params !== undefined) {
        label = `At most ${fmtInt(g.params)} parameters`;
        now = params === null ? '' : `now ${fmtInt(params)}`;
        ok = params !== null && params <= g.params;
      } else if (g.steps !== undefined) {
        label = `Within the first ${fmtInt(g.steps)} training steps of a run`;
        now = run ? `run ${run.n}: step ${fmtInt(run.step)}${run.step > g.steps ? ' (past it: ↺ starts a new run)' : ''}` : 'not trained yet';
        ok = !!run && run.step <= g.steps;
      } else if (g.classic !== undefined) {
        const classic = this.isClassicSetup();
        label = g.classic ? 'With a classic model, not a network' : 'With a neural network, not a classic model';
        now = `now ${classic ? 'a classic model' : 'a network'}`;
        ok = classic === g.classic;
      } else if (g.dense !== undefined) {
        const n = this.setup.blocks.filter((b) => b.type === 'dense').length;
        label = `Keep at least ${g.dense} dense layers`;
        now = `now ${n}`;
        ok = n >= g.dense;
      } else {
        label = `Keep ${OPT_LABEL[g.opt]} as the optimizer`;
        now = `now ${OPT_LABEL[this.setup.train.opt]}`;
        ok = this.setup.train.opt === g.opt;
      }
      return el('li', { class: 'bd-ch-goal' + (ok ? ' is-met' : '') }, el('span', { class: 'bd-ch-mark', 'aria-hidden': 'true', text: ok ? '✓' : '○' }),
        el('span', { class: 'bd-ch-label' }, label, el('span', { class: 'sr-only', text: ok ? ' (met)' : ' (not yet)' })), el('span', { class: 'bd-ch-now', text: now }));
    });
    const hint = this._hint === ch.id;
    root.append(el('div', { class: 'bd-ch-card' + (won ? ' is-won' : '') },
      el('div', { class: 'eyebrow', text: `Challenge · ${ch.level}${ch.topic ? ' · ' + ch.topic : ''}` }),
      el('h3', { class: 'bd-ch-title' }, won ? el('span', { class: 'bd-ch-tick', 'aria-hidden': 'true', text: '✓' }) : null, ch.title),
      el('p', { class: 'bd-ch-brief', text: ch.brief }),
      el('ul', { class: 'bd-ch-goals', 'aria-label': 'Goals' }, ...goals),
      won ? el('p', { class: 'bd-ch-won', text: 'Completed. Try another challenge, or keep going with this one.' }) : null,
      !here ? el('p', { class: 'bd-ch-away', text: `This challenge is on ${BT.byId[ch.problem].label.toLowerCase()}: switch back to it, or start over.` }) : null,
      here && !ownData ? el('p', { class: 'bd-ch-away', text: `The data has been changed${ch.free ? ` beyond the ${ch.free.join(' and ')}` : ''}: a run only counts on the challenge’s own data. Start over to put it back.` }) : null,
      hint ? el('p', { class: 'bd-ch-hint' }, el('strong', { text: 'Hint. ' }), ch.hint) : null,
      el('div', { class: 'bd-ch-actions' },
        hint || won ? null : el('button', { class: 'btn btn-quiet', type: 'button', onclick: () => { this._hint = ch.id; this.renderChallenge(); } }, 'Show a hint'),
        el('button', { class: 'btn btn-quiet', type: 'button', onclick: () => this.startChallenge(ch.id) }, 'Start over'),
        el('button', { class: 'btn btn-quiet', type: 'button', onclick: () => this.openChallenges() }, 'All challenges'),
        el('button', { class: 'btn btn-quiet', type: 'button', onclick: () => this.leaveChallenge() }, 'Leave'))));
  };
  BL.openChallenges = function () {
    if (!this.chDlg) {
      this.chList = el('div', { class: 'bd-ch-list' });
      this.chDlg = el('div', { class: 'modal', role: 'dialog', 'aria-modal': 'true', 'aria-labelledby': 'chTitle', hidden: '' },
        el('div', { class: 'sheet bd-ch-sheet' },
          el('div', { class: 'sheet-head' },
            el('div', null, el('h2', { class: 'sheet-title', id: 'chTitle', text: 'Challenges' }),
              el('p', { class: 'sheet-sub', text: 'Each one loads a model that falls short of a goal. Change it, train it, and reach the goal: a run that does completes the challenge.' })),
            el('button', { class: 'icon-btn', type: 'button', 'data-close': '', 'aria-label': 'Close' }, '×')),
          this.chList));
      this.chDlg.addEventListener('click', (e) => { if (e.target === this.chDlg || e.target.closest('[data-close]')) this.closeChallenges(); });
      this.chDlg.addEventListener('keydown', (e) => { if (e.key === 'Escape') { e.stopPropagation(); this.closeChallenges(); } });
      document.body.append(this.chDlg);
    }
    if (this.training) this.pause();
    const done = doneSet();
    clear(this.chList);
    let level = null;
    for (const ch of BC.LIST) {
      if (ch.level !== level) {
        level = ch.level;
        const all = BC.LIST.filter((q) => q.level === level), got = all.filter((q) => done.has(q.id)).length;
        this.chList.append(el('h3', { class: 'zoo-group', text: `${level} · ${got} of ${all.length} done` }));
      }
      const fam = BC.family(ch), won = done.has(ch.id);
      const goals = ch.goals.map((g) => (g.metric !== undefined ? `${this.metricName(fam)} ≥ ${scoreText(fam, g.metric)}` : g.params !== undefined ? `≤ ${fmtInt(g.params)} parameters` : g.steps !== undefined ? `within ${fmtInt(g.steps)} steps`
        : g.classic !== undefined ? (g.classic ? 'a classic model' : 'a network') : g.dense !== undefined ? `keep ${g.dense} dense layers` : `keep ${OPT_LABEL[g.opt]}`)).join(' · ');
      this.chList.append(el('div', { class: 'bd-ch-item' + (won ? ' is-won' : '') + (ch.id === this.challenge ? ' is-active' : '') },
        el('div', { class: 'bd-ch-item-text' },
          el('div', { class: 'bd-ch-item-title' }, won ? el('span', { class: 'bd-ch-tick', text: '✓' }) : null, ch.title, el('span', { class: 'bd-ch-item-prob', text: `${ch.topic ? ch.topic + ' · ' : ''}${ch.table ? `${ch.table} table` : BT.byId[ch.problem].label}` })),
          el('p', { class: 'bd-ch-item-brief', text: ch.brief }),
          el('div', { class: 'bd-ch-item-goals', text: 'Goal: ' + goals + (won ? ' · completed' : '') })),
        el('button', { class: 'btn' + (won ? '' : ' btn-primary'), type: 'button', 'aria-label': `${won ? 'Play again' : 'Start'}: ${ch.title}`, onclick: () => this.startChallenge(ch.id) }, won ? 'Play again' : 'Start')));
    }
    this.chDlg.hidden = false;
    const first = this.chDlg.querySelector('.bd-ch-item .btn');
    if (first) first.focus();
  };
  BL.closeChallenges = function () { if (this.chDlg) this.chDlg.hidden = true; };

  // ------------------------------------------------------------------ learning-rate sweep
  /* Trains the current setup briefly at eight learning rates, from fresh weights each time (the same seed and data),
     and shows where each got to: too small barely moves, too big jumps around or blows up. Each rate gets the
     same number of steps, as many as fit in about a second and a half. The sweep runs in slices so the page stays
     responsive, and stops if you press Train or change the setup. */
  const SWEEP_LRS = [0.0003, 0.001, 0.003, 0.01, 0.03, 0.1, 0.3, 1];
  // the setup a sweep belongs to: everything but the learning rate it is choosing
  BL.sweepKey = function () { const S = JSON.parse(this.snapshot()); delete S.train.lr; return JSON.stringify(S); };
  BL.startSweep = function () {
    if (this.isClassicSetup()) return;
    if (this.training) this.pause();
    const d = this.dataFor(), chk = this.recheck();
    if (d.error || d.tooFew || !chk.ok) { NA.ui.toast.show('The model has to fit together, with data to learn, before a sweep.'); return; }
    const T = this.setup.train;
    this.sweep = { key: this.sweepKey(), d, T: Object.assign({}, T), i: 0, results: [], steps: 0, model: null, status: 'running' };
    this.renderSweep();
    const tick = () => {
      const sw = this.sweep;
      if (!sw || sw.status !== 'running') return;
      if (this.training || this.sweepKey() !== sw.key) { sw.status = 'stopped'; this.renderSweep(); return; }
      this.sweepSlice(sw, 14);
      this.renderSweep();
      if (sw.status === 'running') requestAnimationFrame(tick);
    };
    requestAnimationFrame(tick);
  };
  BL.sweepSlice = function (sw, budget) {
    const t0 = performance.now(), d = sw.d, T = sw.T, loss = this.setup.loss;
    const B = Math.min(T.batch, d.nTr), inSize = d.Xtr.length / d.nTr, ySize = d.Ytr.length / d.nTr;
    while (performance.now() - t0 < budget && sw.status === 'running') {
      if (!sw.model) {
        sw.model = NA.bx.build(this.setup.blocks, d.input, d.output, { seed: T.seed, init: T.init });
        sw.rng = new NA.RNG(T.seed * 977 + 7); sw.order = null; sw.pos = 0; sw.done = 0; sw.ema = null; sw.diverged = false; sw.t0 = performance.now();
        if (!sw.bX) { sw.bX = new d.Xtr.constructor(B * inSize); sw.bY = new d.Ytr.constructor(B * ySize); }
      }
      for (let b = 0; b < B; b++) {
        if (!sw.order || sw.pos >= sw.order.length) { sw.order = sw.rng.shuffle(NA.range(d.nTr)); sw.pos = 0; }
        const k = sw.order[sw.pos++];
        sw.bX.set(d.Xtr.subarray(k * inSize, (k + 1) * inSize), b * inSize);
        sw.bY.set(d.Ytr.subarray(k * ySize, (k + 1) * ySize), b * ySize);
      }
      const r = sw.model.step(sw.bX, sw.bY, B, loss, { id: T.opt, lr: SWEEP_LRS[sw.i], weightDecay: T.wd, clip: T.clip });
      sw.done++;
      if (!Number.isFinite(r.loss) || r.loss > 1e6) sw.diverged = true;
      else sw.ema = sw.ema === null ? r.loss : 0.9 * sw.ema + 0.1 * r.loss;
      // the first rate sets the step count: as many as fit in about 1.5 s, between 60 and 400
      if (!sw.steps && sw.done === 12) sw.steps = clamp(Math.round(1500 / ((performance.now() - sw.t0) / 12)), 60, 400);
      if (sw.diverged || (sw.steps && sw.done >= sw.steps)) {
        let val = null, vm = null;
        if (!sw.diverged) {
          const per = d.input.kind === 'tokens' ? d.input.len : 1, n = Math.min(d.nVa, Math.max(50, Math.round(600 / per)));
          const v = sw.model.evaluate(d.Xva.subarray(0, n * inSize), d.Yva.subarray(0, n * ySize), n, loss);
          val = Number.isFinite(v.loss) ? v.loss : null;
          vm = BT.isValue(d.family) ? BT.r2(sw.model.predict(d.Xva.subarray(0, n * inSize), n), d.Yva, n) : v.count ? v.correct / v.count : null;
          if (val === null) sw.diverged = true;
        }
        sw.results.push({ lr: SWEEP_LRS[sw.i], val, vm, diverged: sw.diverged });
        sw.model = null; sw.i++;
        if (sw.i >= SWEEP_LRS.length) {
          sw.status = 'done';
          // the suggestion: the smallest rate that reached the top score. On an easy problem the biggest rates often
          // reach the lowest loss in a short run, but they gain nothing there and are the first to go unstable.
          const ok = sw.results.filter((q) => !q.diverged), top = Math.max(...ok.map((q) => (q.vm === null ? -Infinity : q.vm)));
          sw.best = !ok.length ? null : Number.isFinite(top) ? ok.find((q) => q.vm !== null && q.vm >= top - 0.005) : ok.reduce((b, q) => (q.val < b.val ? q : b));
        }
      }
    }
  };
  BL.renderSweep = function () {
    const box = this.parts && this.parts.sweep;
    if (!box || !box.isConnected) return;
    const sw = this.sweep && (this.sweep.key === this.sweepKey() || this.sweep.status === 'running') ? this.sweep : null;
    const fam = this.family(), cur = this.setup.train.lr;
    clear(box);
    const btn = el('button', { class: 'btn', type: 'button', disabled: sw && sw.status === 'running', onclick: () => this.startSweep() }, sw && sw.status === 'running' ? 'Sweeping…' : sw ? 'Sweep again' : 'Try 8 learning rates');
    box.append(UI.note(`Trains this model briefly at each learning rate from ${SWEEP_LRS[0]} to ${SWEEP_LRS[SWEEP_LRS.length - 1]}, from the same starting weights, and compares where each got to.`), el('div', { class: 'btn-row' }, btn,
      sw && sw.status === 'running' ? el('span', { class: 'sec-note', text: `Trying ${SWEEP_LRS[sw.i]} (${sw.i + 1} of ${SWEEP_LRS.length})${sw.steps ? `, ${sw.steps} steps each` : ''}…` }) : null,
      sw && sw.status === 'stopped' ? el('span', { class: 'sec-note', text: 'Stopped: training started or the setup changed.' }) : null));
    if (!sw || !sw.results.length) return;
    const c = el('canvas', { role: 'img' });
    box.append(el('div', { class: 'chart-box' }, c));
    this.drawSweep(c, sw);
    c.setAttribute('aria-label', 'Validation loss after a short run at each learning rate: ' + sw.results.map((q) => `${q.lr}: ${q.diverged ? 'blew up' : fmt(q.val, 3)}`).join(', '));
    const rows = sw.results.map((q) => el('tr', { class: (sw.best === q ? 'is-best' : '') + (q.lr === cur ? ' is-current' : '') },
      el('td', { class: 'mono', text: String(q.lr) }), el('td', { class: 'mono', text: q.diverged ? 'blew up' : fmt(q.val, 3) }), el('td', { class: 'mono', text: q.diverged || q.vm === null ? '—' : scoreText(fam, q.vm) }),
      el('td', { text: [sw.best === q ? 'suggested' : '', q.lr === cur ? 'current' : ''].filter(Boolean).join(', ') })));
    box.append(el('table', { class: 'bd-sweep-table' }, el('thead', null, el('tr', null, el('th', { text: 'Learning rate' }), el('th', { text: 'Val. loss' }), el('th', { text: fam === 'curve' ? 'R²' : 'Score' }), el('th', { text: '' }))), el('tbody', null, ...rows)));
    if (sw.status === 'done' && sw.best) {
      box.append(el('div', { class: 'btn-row' }, sw.best.lr !== cur ? el('button', { class: 'btn btn-primary', type: 'button', onclick: () => { this.edit((s) => { s.train.lr = sw.best.lr; }); NA.ui.toast.show(`Learning rate set to ${sw.best.lr}. Press Train for a full run.`); } }, `Use ${sw.best.lr}`) : null,
        el('span', { class: 'sec-note', text: `${sw.best.lr === cur ? 'Your learning rate is the suggestion: after' : 'After'} ${sw.steps} steps, ${sw.best.lr} was the smallest learning rate to reach the top score${sw.best.vm !== null ? ` (${scoreText(fam, sw.best.vm)})` : ''}. Smaller rates were still learning; bigger ones gain nothing here and are the first to go unstable on longer runs.` })));
    }
  };
  BL.drawSweep = function (c, sw) {
    const W = Math.max(220, c.parentElement.clientWidth || 300), Hh = 130, ctx = NA.fitCanvas(c, W, Hh);
    const left = 42, right = 10, top = 10, base = Hh - 22, xs = SWEEP_LRS.map((v) => Math.log10(v)), x0 = xs[0], x1 = xs[xs.length - 1];
    const X = (lr) => left + ((Math.log10(lr) - x0) / (x1 - x0)) * (W - left - right);
    const ok = sw.results.filter((q) => !q.diverged), lo = ok.length ? Math.min(...ok.map((q) => q.val)) : 0.01, hi = ok.length ? Math.max(...ok.map((q) => q.val)) : 1;
    const ylo = Math.log10(Math.max(1e-4, lo)) - 0.1, yhi = Math.log10(Math.max(1e-4, hi)) + 0.1;
    const Y = (v) => base - ((Math.log10(Math.max(1e-4, v)) - ylo) / Math.max(1e-6, yhi - ylo)) * (base - top - 8);
    const mono = '500 10px ' + (getComputedStyle(document.body).getPropertyValue('--font-mono') || 'monospace');
    ctx.clearRect(0, 0, W, Hh); ctx.font = mono;
    ctx.strokeStyle = theme.c('line'); ctx.lineWidth = 1;
    ctx.beginPath(); ctx.moveTo(left, base + 0.5); ctx.lineTo(W - right, base + 0.5); ctx.stroke();
    ctx.fillStyle = theme.c('muted'); ctx.textAlign = 'center'; ctx.textBaseline = 'top';
    const roomy = (W - left - right) / SWEEP_LRS.length > 44;
    SWEEP_LRS.forEach((lr, i) => { if (roomy || i % 2 === 1) ctx.fillText(String(lr), X(lr), base + 6); });
    ctx.textAlign = 'right'; ctx.textBaseline = 'middle';
    const tick = (v) => (v >= 0.01 ? fmt(v, 2) : v.toExponential(0).replace(/e-(\d+)$/, (m, e) => '×10⁻' + [...e].map((c) => '⁰¹²³⁴⁵⁶⁷⁸⁹'[c]).join('')));
    for (const v of [lo, hi]) { if (!ok.length) break; ctx.fillText(tick(v), left - 6, Y(v)); }
    ctx.save(); ctx.translate(10, (top + base) / 2); ctx.rotate(-Math.PI / 2); ctx.textAlign = 'center'; ctx.fillText('val. loss', 0, 0); ctx.restore();
    ctx.strokeStyle = theme.c('s1'); ctx.lineWidth = 2; ctx.beginPath();
    let pen = false;
    for (const q of sw.results) { if (q.diverged) { pen = false; continue; } const x = X(q.lr), y = Y(q.val); if (pen) ctx.lineTo(x, y); else ctx.moveTo(x, y); pen = true; }
    ctx.stroke();
    for (const q of sw.results) {
      const x = X(q.lr);
      if (q.diverged) {
        ctx.strokeStyle = theme.c('critical'); ctx.lineWidth = 2; const y = top + 6;
        ctx.beginPath(); ctx.moveTo(x - 4, y - 4); ctx.lineTo(x + 4, y + 4); ctx.moveTo(x + 4, y - 4); ctx.lineTo(x - 4, y + 4); ctx.stroke();
        continue;
      }
      const y = Y(q.val);
      ctx.fillStyle = theme.c('panel'); ctx.beginPath(); ctx.arc(x, y, 5, 0, Math.PI * 2); ctx.fill();
      ctx.fillStyle = theme.c(sw.best === q ? 'good' : 's1'); ctx.beginPath(); ctx.arc(x, y, sw.best === q ? 4.5 : 3.5, 0, Math.PI * 2); ctx.fill();
    }
  };

  // ------------------------------------------------------------------ comparing two runs
  BL.renderCompare = function (box, runs) {
    clear(box);
    if (runs.length < 2) { box.append(UI.note('Train twice on this problem, changing something in between, to compare two runs side by side.')); return; }
    const byN = Object.fromEntries(runs.map((r) => [r.n, r]));
    let a = this.cmp && byN[this.cmp.a], b = this.cmp && byN[this.cmp.b];
    if (!a || !b || a === b) { b = runs[runs.length - 1]; a = runs[runs.length - 2]; }
    this.cmp = { a: a.n, b: b.n };
    const pick = (cur, which) => UI.select(runs.map((r) => ({ value: r.n, label: `Run ${r.n}` })), cur.n, (v) => { this.cmp[which] = +v; this._runsKey = null; this.renderRunsTab(); }, { 'aria-label': which === 'a' ? 'First run to compare' : 'Second run to compare' });
    const fam = this.family(), L = (r) => r.last || {};
    const num = (v, f) => (v === null || v === undefined ? '—' : f(v));
    const delta = (va, vb, f, better, tol) => {
      if (va === null || va === undefined || vb === null || vb === undefined) return '';
      const dv = vb - va;
      if (Math.abs(dv) <= tol) return 'same';
      const good = better === 'up' ? dv > 0 : dv < 0;
      return `${dv > 0 ? '+' : '−'}${f(Math.abs(dv))}${better ? (good ? ' better' : ' worse') : ''}`;
    };
    const vfam = BT.isValue(this.dataFor().family), score = (v) => (vfam ? fmt(v, 3) : fmtPct(v, 1)), scoreD = (v) => (vfam ? fmt(v, 3) : `${fmt(v * 100, 1)} pts`);
    const rows = [
      [`Validation ${this.metricName()}`, L(a).vm, L(b).vm, score, scoreD, 'up', 5e-4],
      ['Validation loss', L(a).val, L(b).val, (v) => fmt(v, 4), (v) => fmt(v, 4), 'down', 5e-5],
      ['Training loss', L(a).train, L(b).train, (v) => fmt(v, 4), (v) => fmt(v, 4), 'down', 5e-5],
      ['Parameters', a.params, b.params, fmtInt, fmtInt, null, 0],
      ['Steps', a.step, b.step, fmtInt, fmtInt, null, 0],
      ['Time', a.seconds, b.seconds, (v) => `${v.toFixed(1)} s`, (v) => `${v.toFixed(1)} s`, null, 0.05],
    ];
    const changes = diffSetups(a.cfg, b.cfg);
    box.append(el('div', { class: 'bd-cmp-pick' }, pick(a, 'a'), el('span', { class: 'sec-note', text: 'against' }), pick(b, 'b')),
      el('table', { class: 'bd-cmp-table' },
        el('thead', null, el('tr', null, el('th', { text: '' }), el('th', null, el('i', { class: 'bd-run-sw', style: { background: `var(--${a.colour})` } }), ` Run ${a.n}`), el('th', null, el('i', { class: 'bd-run-sw', style: { background: `var(--${b.colour})` } }), ` Run ${b.n}`), el('th', { text: 'Change' }))),
        el('tbody', null, ...rows.map(([k, va, vb, f, fd, better, tol]) => el('tr', null, el('th', { scope: 'row', text: k }), el('td', { class: 'mono', text: num(va, f) }), el('td', { class: 'mono', text: num(vb, f) }), el('td', { class: 'bd-cmp-d', text: delta(va, vb, fd, better, tol) }))))),
      el('div', { class: 'bd-cmp-changes' }, el('span', { class: 'sec-note', text: `From run ${a.n} to run ${b.n}: ` }), ...(changes.length ? changes.map((c) => el('span', { class: 'bd-change', text: c })) : [el('span', { class: 'sec-note', text: 'the same setup (another start, or more steps).' })])),
      el('div', { class: 'btn-row' }, el('button', { class: 'btn btn-quiet', type: 'button', onclick: () => { for (const r of this.runs) r.show = r === a || r === b; this._runsKey = null; this.renderRunsTab(); } }, 'Show only these two on the chart')));
  };

  // ------------------------------------------------------------------ the zoo
  BL.runTry = function (t) {
    const go = () => { if (NA.mobile && NA.mobile.mobile) NA.mobile.go('network'); };
    if (this.training) this.pause();
    if (t.action === 'spiralLinear') { this.setProblem('spiral'); this.applyTemplate(BT.TEMPLATES.find((q) => q.id === 'linear')); this.play(); }
    else if (t.action === 'spiralDeep') { this.setProblem('spiral'); this.applyTemplate(BT.TEMPLATES.find((q) => q.id === 'deep')); this.play(); }
    else if (t.action === 'lrHigh') { this.edit((s) => { s.train.lr = 1; s.train.opt = 'sgd'; }); this.play(); }
    else if (t.action === 'reverse') { this.setProblem('reverse'); this.play(); }
    else if (t.action === 'reverseNoAttn') { this.setProblem('reverse'); this.applyTemplate(BT.TEMPLATES.find((q) => q.id === 'noattn')); this.play(); }
    else if (t.action === 'draw') { this.setProblem('draw'); this.edit((s) => { s.custom.points = seedPoints(); }); this.selectBlock({ kind: 'input', id: 'input' }, true); this.centerView = NA.mobile && NA.mobile.mobile ? 'results' : this.centerView; this.layoutCenter(); }
    go();
    // with a prediction attached, the answer waits for the run: the promise gives the measured score
    if (!t.predict || !this.training) return undefined;
    return new Promise((resolve) => {
      const off = NA.app.on('training', () => { if (this.training) return; off(); resolve(this.run && this.run.last ? `Measured: ${this.metricText(this.run)} after ${fmtInt(this.run.step)} steps.` : ''); });
    });
  };
  // Zoo glyph: blocks stacked, one being dropped in.
  BL.glyph = function (ctx, W, Hh) {
    const tones = ['s3', 's1', 's7', 's1', 's2'];
    tones.forEach((t, i) => {
      const y = 6 + i * ((Hh - 12) / 5), off = i === 2 ? 7 : 0;
      ctx.fillStyle = theme.c(t, 0.22); NA.rr(ctx, W * 0.24 + off, y, W * 0.52, (Hh - 12) / 5 - 3, 3); ctx.fill();
      ctx.fillStyle = theme.c(t); ctx.fillRect(W * 0.24 + off, y, 3, (Hh - 12) / 5 - 3);
    });
  };

  H.register(BL);
  NA.builderLab = BL;
})(window.NA);
