/* builder-gen.js — six more kinds of problem for the model builder: squeezing pictures through a narrow middle and
   rebuilding them (autoencoders; the VAE, which draws new ones; cleaning up noisy pictures), a GAN whose generator is
   the model you build, labelling every pixel of a picture (segmentation, with a tiny U-Net), graphs whose nodes learn
   from their neighbours (message passing), learning without labels by matching two views of each picture
   (contrastive learning) and predicting a number with how unsure it is (a middle and a spread). Their data, the
   trainers of the two that are not examples with answers (the GAN's and the contrastive one), their scores, their
   pictures and their settings. They join the catalogue in js/builder-tasks.js; the engine's Upsample, Sample and Graph
   convolution blocks and Gaussian loss are in js/builder-engine.js, their Python in js/builder-py-gen.js. */
(function (NA) {
  'use strict';
  const BT = NA.builderTasks, G = {};
  const clamp = (v, a, b) => (v < a ? a : v > b ? b : v);
  const IMG = 14, PX = IMG * IMG, SEG = 16, NODES = 12, GROUPS = 3, NODE_F = 4;
  const softplus = (x) => (x > 0 ? x + Math.log1p(Math.exp(-x)) : Math.log1p(Math.exp(x)));
  const sigmoid = (x) => 1 / (1 + Math.exp(-x));

  // ------------------------------------------------------------------ the catalogue
  /* The GAN and contrastive learning have losses of their own (the method sets them, as for rewards); the others learn
     from examples with answers: pictures rebuilt (a number per pixel), a class per pixel or per node, and a middle
     with a spread. */
  const FAMILIES = {
    recon: { label: 'Squeeze and rebuild', losses: ['mse', 'mae', 'huber'], metric: 'r2', template: 'ae', icon: '⧖' },
    gan: { label: 'Make new examples (GAN)', losses: ['mse'], metric: 'quality', template: 'gen', icon: '✦' },
    contrast: { label: 'Learn without labels', losses: ['mse'], metric: 'knn', template: 'enc', icon: '⇄' },
    segment: { label: 'Label every pixel', losses: ['xent'], metric: 'iou', template: 'unet', icon: '▩' },
    graph: { label: 'Graphs', losses: ['xent'], metric: 'accuracy', template: 'gnn', icon: '⬡' },
    spread: { label: 'Predict with uncertainty', losses: ['gauss', 'mse'], metric: 'r2', template: 'mlp', icon: '±' },
  };
  G.FAMS = new Set(Object.keys(FAMILIES));
  G.owns = (fam) => G.FAMS.has(fam);
  const PROBLEMS = [
    { id: 'autoenc', family: 'recon', task: 'autoenc', template: 'ae', label: 'Squeeze and rebuild (autoencoder)', hint: 'Each picture goes through a narrow middle and is rebuilt from it, so the model has to find a short code for it. With 2 numbers in the middle, the codes of all the pictures make a map you can look at.' },
    { id: 'vae', family: 'recon', task: 'vae', template: 'vae', label: 'Draw new digits (VAE)', hint: 'A variational autoencoder: its middle is a bell curve for each picture, not a point, and it is pulled toward one bell curve for all of them. Any code drawn from that curve then decodes to something like a digit, so it can draw new ones.' },
    { id: 'denoise', family: 'recon', task: 'denoise', template: 'dae', label: 'Clean up noisy pictures', hint: 'The model sees each digit with speckles and a scratch across it, and has to give back the clean digit. To rebuild what is hidden it has to know what digits look like.' },
    { id: 'ganring', family: 'gan', task: 'ring', label: 'Eight blobs in a ring', hint: 'Your model is the generator: it turns random numbers into points. A second network, the discriminator, learns to tell its points from the real ones, and the generator learns to fool it. Watch for mode collapse: a generator that finds only some of the 8 blobs.' },
    { id: 'ganmoons', family: 'gan', task: 'moons', label: 'Two moons', hint: 'Real points on two crescents: the generator has to bend its random numbers into both curves.' },
    { id: 'gandigits', family: 'gan', task: 'digits', template: 'gendeep', label: 'Tiny digits', hint: 'The generator draws 14 × 14 pictures from 8 random numbers, and the discriminator compares them with handwritten digits: a small version of how GANs drew faces.' },
    { id: 'contrast', family: 'contrast', task: 'contrast', label: 'Two views of each picture', hint: 'No labels at all: every picture is shown twice, shifted, thickened and speckled differently, and the model learns to give the two views the same code and other pictures different ones (SimCLR). A vote of the nearest codes then shows how much it learned about which digit is which, with the labels used only to score.' },
    { id: 'segment', family: 'segment', task: 'segment', label: 'Find the shapes, pixel by pixel', hint: 'Circles, squares and triangles in a picture, and for every pixel which shape it belongs to, if any. A model that sees one pixel at a time sees only ink; a tiny U-Net shrinks the picture to see what each shape is, then grows it back, its skip connection keeping the edges sharp.' },
    { id: 'groups', family: 'graph', task: 'groups', label: 'Find the friend groups', hint: 'Twelve people in three groups of friends: many links within a group, a few across. Only one person in each group says which group they are in; the model has to work out the rest from who is linked to whom. Message passing lets each node hear its neighbours, and with a second round, theirs.' },
    { id: 'noisy', family: 'spread', task: 'noisy', label: 'Noise that grows', hint: 'The points stray further from the curve the further right they are. With two outputs, a middle and a spread, and the Gaussian loss, the model learns both: where the curve is, and how far from it the points fall at each place.' },
    { id: 'branches', family: 'spread', task: 'branches', losses: ['mdn', 'mse'], label: 'More than one right answer', hint: 'For some x there are three right answers, on three branches of a curve. A middle and a spread can only say “somewhere between them”; a mixture of bell curves (a mixture density network) puts a curve on each branch and weighs them.' },
    { id: 'gapdata', family: 'spread', task: 'gap', label: 'A gap in the data', hint: 'No examples between −0.3 and 0.3. The spread it learns is the noise it saw, not what it does not know: across the gap it can be wrong and sure of itself at once. Train it from a few seeds and compare them there.' },
  ];
  // the new families go before your own table and the worlds; the problems likewise
  {
    const F = BT.FAMILIES, tail = ['table', 'control'].map((k) => [k, F[k]]);
    for (const [k] of tail) delete F[k];
    Object.assign(F, FAMILIES);
    for (const [k, v] of tail) F[k] = v;
    const at = BT.PROBLEMS.findIndex((p) => p.family === 'table');
    BT.PROBLEMS.splice(at < 0 ? BT.PROBLEMS.length : at, 0, ...PROBLEMS);
    for (const p of PROBLEMS) BT.byId[p.id] = p;
  }
  // a problem may have losses of its own (More than one right answer: the mixture of bell curves first)
  const lossesFor0 = BT.lossesFor;
  BT.lossesFor = (p, d) => p.losses || lossesFor0(p, d);
  const isValue0 = BT.isValue;
  BT.isValue = (fam) => fam === 'recon' || fam === 'spread' || isValue0(fam);

  const DEFAULTS = {
    recon: (p) => ({ n: 2000, pictures: 'digits', noise: p.task === 'denoise' ? 0.35 : 0 }),
    gan: (p) => (p.task === 'digits' ? { n: 2000, z: 8, disc: 'big' } : { n: 1000, z: 2, disc: 'small' }),
    contrast: () => ({ n: 2000, pictures: 'digits', dim: 16, tau: 0.2, views: 'strong' }),
    segment: () => ({ n: 1200, shapes: 2, noise: 0.1 }),
    graph: () => ({ n: 1000, across: 0.06, hints: 1 }),
    spread: (p) => (p.task === 'branches' ? { n: 600, components: 3 } : { n: 400, noise: 0.3 }),
  };
  const TRAIN = {
    recon: { lr: 0.003, batch: 32, steps: 2000, clip: 0 },
    gan: (p) => ({ lr: 0.001, batch: 64, steps: p.task === 'digits' ? 4000 : 3000, clip: 0 }),
    contrast: { lr: 0.003, batch: 32, steps: 1500, clip: 1 },
    segment: { lr: 0.003, batch: 16, steps: 1500, clip: 1 },
    graph: { lr: 0.01, batch: 32, steps: 1500, clip: 0 },
    spread: { lr: 0.003, batch: 32, steps: 3000, clip: 1 },
  };
  const defaults0 = BT.defaults, trainFor0 = BT.trainFor, make0 = BT.make;
  BT.defaults = (p) => (FAMILIES[p.family] ? DEFAULTS[p.family](p) : defaults0(p));
  BT.trainFor = (p) => (FAMILIES[p.family] ? Object.assign({}, typeof TRAIN[p.family] === 'function' ? TRAIN[p.family](p) : TRAIN[p.family]) : trainFor0(p));
  BT.make = function (p, s, seed = 1, custom = {}) {
    if (!FAMILIES[p.family]) return make0(p, s, seed, custom);
    return MAKE[p.family](p, Object.assign(DEFAULTS[p.family](p), s), seed);
  };

  // ------------------------------------------------------------------ templates
  const D = (units, act = 'relu') => ({ type: 'dense', units, act });
  const conv = (filters, k = 3) => ({ type: 'conv', filters, k, stride: 1, act: 'relu' });
  BT.TEMPLATES.push(
    { id: 'ae', label: 'Autoencoder (2 in the middle)', hint: 'Narrows to 2 numbers and widens again: every picture becomes a point on a map, and is rebuilt from that point alone.', families: ['recon'], blocks: () => [{ type: 'flatten' }, D(64), D(16), D(2, 'linear'), D(16), D(64)] },
    { id: 'ae8', label: 'Autoencoder (8 in the middle)', hint: 'A middle of 8 numbers: sharper pictures than with 2, and a map that needs flattening to be seen.', families: ['recon'], blocks: () => [{ type: 'flatten' }, D(64), D(8, 'linear'), D(64)] },
    { id: 'vae', label: 'VAE', hint: 'The encoder gives a middle and a spread for 2 numbers, Sample draws a code from that bell curve, the decoder rebuilds the picture from it. Its loss adds how far the bell curves are from one standard bell curve.', families: ['recon'], blocks: () => [{ type: 'flatten' }, D(64), D(4, 'linear'), { type: 'sample', beta: 0.3 }, D(64)] },
    { id: 'dae', label: 'Denoising autoencoder', hint: 'An autoencoder with 16 in the middle: from the noisy picture to the clean one.', families: ['recon'], blocks: () => [{ type: 'flatten' }, D(64), D(16, 'linear'), D(64)] },
    { id: 'gen', label: 'Small generator', hint: 'Two layers of 32: random numbers in, a point out.', families: ['gan'], blocks: () => [D(32), D(32)] },
    { id: 'gendeep', label: 'Bigger generator', hint: 'Two layers of 128: enough to draw pictures.', families: ['gan'], blocks: () => [D(128), D(128)] },
    { id: 'genlin', label: 'No hidden layer', hint: 'A straight map from the random numbers: it can stretch and turn a bell curve, never split it into blobs.', families: ['gan'], blocks: () => [] },
    { id: 'enc', label: 'Small CNN encoder', hint: 'Convolutions and pooling, then a dense layer: the usual encoder for pictures.', families: ['contrast'], blocks: () => [conv(8), { type: 'pool', size: 2 }, conv(16), { type: 'pool', size: 2 }, { type: 'flatten' }, D(32)] },
    { id: 'encmlp', label: 'MLP encoder', hint: 'Dense layers on the pixels: it has to learn that a shifted picture is the same picture.', families: ['contrast'], blocks: () => [{ type: 'flatten' }, D(64), D(32)] },
    { id: 'segpix', label: 'Pixel by pixel', hint: 'Each pixel on its own (patches of 1 pixel), then a dense layer: it sees ink, not shapes.', families: ['segment'], blocks: () => [{ type: 'patches', size: 1, dim: 16 }, D(16)] },
    { id: 'segcnn', label: 'Two convolutions', hint: 'Two 3 × 3 convolutions: every pixel sees the 5 × 5 square around it.', families: ['segment'], blocks: () => [conv(8), conv(8), { type: 'patches', size: 1, dim: 16 }] },
    { id: 'unet', label: 'Tiny U-Net', hint: 'A convolution, then two branches joined: the picture as it is (the skip connection), and the picture pooled down to 4 × 4, where one filter sees a whole shape, and upsampled back. A last convolution mixes them for every pixel.', families: ['segment'], blocks: () => [conv(8), { type: 'parallel', merge: 'concat', branches: [[], [{ type: 'pool', size: 2 }, conv(16), { type: 'pool', size: 2 }, conv(16), { type: 'upsample', size: 4 }]] }, conv(8), { type: 'patches', size: 1, dim: 16 }] },
    { id: 'gnn', label: 'Graph network', hint: 'Two rounds of message passing: each node takes in its neighbours, then their neighbours.', families: ['graph'], blocks: () => [{ type: 'graphconv', units: 16, act: 'relu', agg: 'mean' }, { type: 'graphconv', units: 16, act: 'relu', agg: 'mean' }] },
    { id: 'gnn1', label: 'One round', hint: 'One round of message passing: only the direct neighbours are heard.', families: ['graph'], blocks: () => [{ type: 'graphconv', units: 16, act: 'relu', agg: 'mean' }] },
    { id: 'gnnsum', label: 'Graph network (sums)', hint: 'Two rounds that add up the neighbours instead of averaging them: a node with many friends hears them louder.', families: ['graph'], blocks: () => [{ type: 'graphconv', units: 16, act: 'relu', agg: 'sum' }, { type: 'graphconv', units: 16, act: 'relu', agg: 'sum' }] },
    { id: 'nomp', label: 'No message passing', hint: 'Dense layers on each node alone: it reads its own row (its hint, if any, and which numbers it is linked to) but never its neighbours’ features.', families: ['graph'], blocks: () => [D(16), D(16)] },
  );
  for (const t of BT.TEMPLATES) if (['linear', 'mlp', 'deep', 'wide', 'fourier', 'resnet'].includes(t.id) && !t.families.includes('spread')) t.families.push('spread');
  // the parts of version 31's last batch, in templates of their own: Kolmogorov–Arnold layers, skip connections, a
  // Hopfield memory, a U-Net whose skip reaches across
  const skip = (back, merge = 'add') => ({ type: 'skip', back, merge });
  BT.TEMPLATES.push(
    { id: 'kan', label: 'Kolmogorov–Arnold', hint: 'Two Kolmogorov–Arnold layers of 8: every edge a learned curve made of 6 bumps, instead of a number on a fixed bend.', families: ['plane', 'curve', 'surface', 'table', 'spread'], blocks: () => [{ type: 'kan', units: 8, grid: 6 }, { type: 'kan', units: 8, grid: 6 }] },
    { id: 'skips', label: 'Skip connections', hint: 'Three tanh layers, and around each of the last two a skip connection that adds back what went in: a ResNet in miniature.', families: ['plane', 'curve', 'surface', 'table', 'spread'], blocks: () => [D(16, 'tanh'), D(16, 'tanh'), skip(1), D(16, 'tanh'), skip(1)] },
    { id: 'hopdae', label: 'Hopfield memory', hint: 'The picture compared with 32 patterns the model learns to remember, the most alike recalled: a memory that cleans up what it is given.', families: ['recon'], blocks: () => [{ type: 'flatten' }, { type: 'hopfield', memories: 32, beta: 8 }] },
    { id: 'unetskip', label: 'U-Net (skip)', hint: 'Down to 4 × 4 and back up, then a skip connection that joins on the first convolution’s output, at full size: the U-Net’s skip, reaching across.', families: ['segment'], blocks: () => [conv(8), { type: 'pool', size: 2 }, conv(16), { type: 'pool', size: 2 }, conv(16), { type: 'upsample', size: 4 }, skip(5, 'concat'), conv(8), { type: 'patches', size: 1, dim: 16 }] },
  );

  // ------------------------------------------------------------------ data
  function split(n, frac) { const k = Math.max(1, Math.min(n - 1, Math.round(n * frac))); return [k, n - k]; }
  // pictures through a narrow middle: the inputs (noisy ones to clean up), and the clean pictures as the answers
  function reconData(p, s, seed) {
    const pics = s.pictures === 'shapes' ? 'shapes' : 'digits', src = make0(BT.byId[pics], { n: s.n }, seed), rng = new NA.RNG(seed * 59 + 3);
    const dirty = (X, n) => {
      if (p.task !== 'denoise') return X;
      const out = new Float32Array(X.length);
      for (let e = 0; e < n; e++) {
        // speckles everywhere, and a scratch: a thick line from one edge to another
        const a = rng.uniform(0, Math.PI), cx = rng.uniform(4, 10), cy = rng.uniform(4, 10), nx = -Math.sin(a), ny = Math.cos(a);
        for (let q = 0; q < PX; q++) {
          const px = (q % IMG) + 0.5, py = Math.floor(q / IMG) + 0.5, dl = Math.abs((px - cx) * nx + (py - cy) * ny);
          out[e * PX + q] = clamp(Math.max(X[e * PX + q], dl < 0.9 ? 0.85 : 0) + s.noise * rng.gauss(), 0, 1);
        }
      }
      return out;
    };
    return { family: 'recon', task: p.task, pictures: pics, input: src.input, output: { kind: 'value', k: PX }, names: src.names, labels: { tr: src.Ytr, va: src.Yva },
      Xtr: dirty(src.Xtr, src.nTr), Ytr: Float32Array.from(src.Xtr), nTr: src.nTr, Xva: dirty(src.Xva, src.nVa), Yva: Float32Array.from(src.Xva), nVa: src.nVa,
      outText: `${PX} numbers: the picture rebuilt`, inText: p.task === 'denoise' ? 'a noisy 14 × 14 picture' : `a 14 × 14 picture` };
  }
  // the real points (or digits) a GAN imitates; the model's inputs are random numbers, z of them per example
  function ganPoints(task, n, rng) {
    const X = new Float32Array(2 * n);
    for (let i = 0; i < n; i++) {
      let x, y;
      if (task === 'ring') { const k = rng.int(8), a = (k * Math.PI) / 4; x = 0.75 * Math.cos(a) + 0.05 * rng.gauss(); y = 0.75 * Math.sin(a) + 0.05 * rng.gauss(); }
      else { const t = rng.uniform(0, Math.PI), up = rng.next() < 0.5; x = (up ? Math.cos(t) : 1 - Math.cos(t)) * 0.62 - 0.31 + 0.04 * rng.gauss(); y = (up ? Math.sin(t) - 0.25 : 0.25 - Math.sin(t)) * 0.62 + 0.04 * rng.gauss(); }
      X[2 * i] = x; X[2 * i + 1] = y;
    }
    return X;
  }
  function ganData(p, s, seed) {
    const rng = new NA.RNG(seed * 61 + 5), z = s.z, pics = p.task === 'digits';
    let real, n;
    let lab = null;
    if (pics) { const src = make0(BT.byId.digits, { n: s.n }, seed); n = src.nTr + src.nVa; real = new Float32Array(n * PX); real.set(src.Xtr); real.set(src.Xva, src.nTr * PX); lab = new Int32Array(n); lab.set(src.Ytr); lab.set(src.Yva, src.nTr); }
    else { n = s.n; real = ganPoints(p.task, n, rng); }
    const K = pics ? PX : 2, [nTr, nVa] = split(n, 0.8), noise = (m) => { const a = new Float32Array(m * z); for (let i = 0; i < a.length; i++) a[i] = rng.gauss(); return a; };
    return { family: 'gan', task: p.task, pixels: pics, disc: s.disc, input: { kind: 'vector', dim: z }, output: { kind: 'value', k: K }, z, K,
      Xtr: noise(nTr), Ytr: real.slice(0, nTr * K), nTr, Xva: noise(nVa), Yva: real.slice(nTr * K), nVa, labels: lab ? { tr: lab.slice(0, nTr), va: lab.slice(nTr) } : null,
      inText: `${z} random number${z > 1 ? 's' : ''}`, outText: pics ? `${PX} numbers: a picture` : '2 numbers: a point (x, y)' };
  }
  // pictures moved by up to s pixels each way (what moves off the edge is lost)
  function shifted(X, n, s, rng) {
    const out = new Float32Array(X.length);
    for (let e = 0; e < n; e++) {
      const dx = rng.int(2 * s + 1) - s, dy = rng.int(2 * s + 1) - s;
      for (let y = 0; y < IMG; y++) for (let x = 0; x < IMG; x++) { const sx = x - dx, sy = y - dy; out[e * PX + y * IMG + x] = sx >= 0 && sy >= 0 && sx < IMG && sy < IMG ? X[e * PX + sy * IMG + sx] : 0; }
    }
    return out;
  }
  function contrastData(p, s, seed) {
    const pics = s.pictures === 'shapes' ? 'shapes' : 'digits', src = Object.assign({}, make0(BT.byId[pics], { n: s.n }, seed));
    // (digits are drawn in the middle: moved about, two of the same digit no longer share their pixels; shapes are anywhere already)
    if (pics === 'digits') { const rng = new NA.RNG(seed * 79 + 1); src.Xtr = shifted(src.Xtr, src.nTr, 3, rng); src.Xva = shifted(src.Xva, src.nVa, 3, rng); }
    return { family: 'contrast', task: 'contrast', pictures: pics, input: src.input, output: { kind: 'value', k: s.dim }, names: src.names, classes: src.classes, labels: { tr: src.Ytr, va: src.Yva },
      Xtr: src.Xtr, Ytr: new Float32Array(src.nTr * s.dim), nTr: src.nTr, Xva: src.Xva, Yva: new Float32Array(src.nVa * s.dim), nVa: src.nVa, tau: s.tau, views: s.views,
      inText: 'a 14 × 14 picture (two views of it)', outText: `${s.dim} numbers: its code` };
  }
  // pictures of filled shapes, each pixel labelled 0 (nothing), 1 circle, 2 square or 3 triangle
  G.SEG_NAMES = ['nothing', 'circle', 'square', 'triangle'];
  function insideShape(c, px, py, sh) {
    const dx = px - sh.x, dy = py - sh.y;
    if (c === 1) return dx * dx + dy * dy <= sh.r * sh.r;
    const ca = Math.cos(sh.a), sa = Math.sin(sh.a), u = dx * ca + dy * sa, v = -dx * sa + dy * ca;
    if (c === 2) { const h = sh.r * 0.82; return Math.abs(u) <= h && Math.abs(v) <= h; }
    // a triangle: inside all three edges, its corners at radius 1.15 r
    for (let q = 0; q < 3; q++) {
      const a0 = (q * 2 * Math.PI) / 3, a1 = ((q + 1) * 2 * Math.PI) / 3, R = sh.r * 1.15;
      const x0 = R * Math.cos(a0), y0 = R * Math.sin(a0), x1 = R * Math.cos(a1), y1 = R * Math.sin(a1);
      if ((x1 - x0) * (v - y0) - (y1 - y0) * (u - x0) < 0) return false;
    }
    return true;
  }
  function segmentData(p, s, seed) {
    const rng = new NA.RNG(seed * 67 + 11), n = s.n, P = SEG * SEG, X = new Float32Array(n * P), Y = new Int32Array(n * P), k = clamp(s.shapes | 0, 1, 3);
    for (let e = 0; e < n; e++) {
      const kinds = rng.shuffle([1, 2, 3]).slice(0, k), placed = [];
      for (const c of kinds) {
        let sh = null;
        for (let t = 0; t < 40 && !sh; t++) {
          const r = rng.uniform(2.5, k === 3 ? 3.6 : 4.3), q = { x: rng.uniform(r + 0.3, SEG - r - 0.3), y: rng.uniform(r + 0.3, SEG - r - 0.3), r, a: rng.uniform(0, 2 * Math.PI) };
          if (placed.every((o) => Math.hypot(o.x - q.x, o.y - q.y) > (o.r + q.r) * 1.1 + 1) || t === 39) sh = q;
        }
        placed.push(sh);
        for (let py = 0; py < SEG; py++) for (let px = 0; px < SEG; px++) {
          // ink: the share of 3 × 3 points in the pixel inside the shape; the label: its middle
          let ink = 0;
          for (let a = 0; a < 3; a++) for (let b = 0; b < 3; b++) if (insideShape(c, px + (a + 0.5) / 3, py + (b + 0.5) / 3, sh)) ink++;
          const i = e * P + py * SEG + px;
          if (ink) X[i] = Math.max(X[i], ink / 9);
          if (insideShape(c, px + 0.5, py + 0.5, sh)) Y[i] = c;
        }
      }
      for (let q = 0; q < P; q++) X[e * P + q] += s.noise * rng.gauss();
    }
    const [nTr, nVa] = split(n, 0.8);
    return { family: 'segment', task: 'segment', input: { kind: 'image', c: 1, h: SEG, w: SEG }, output: { kind: 'tokens', vocab: 4 }, names: G.SEG_NAMES.slice(), classes: 4,
      Xtr: X.slice(0, nTr * P), Ytr: Y.slice(0, nTr * P), nTr, Xva: X.slice(nTr * P), Yva: Y.slice(nTr * P), nVa,
      inText: `a ${SEG} × ${SEG} picture`, outText: `for each of its ${P} pixels, 4 scores (nothing, circle, square, triangle)` };
  }
  /* Graphs of 12 people in 3 groups of 4, in a random order: linked within a group with odds 0.6 (each at least once),
     across groups with the odds set. A row per node: 4 features (which group it says it is in, one-hot, and whether it
     says so), then its 12 links. The answer: its group, or −1 (no answer) for those that say. */
  function graphData(p, s, seed) {
    const rng = new NA.RNG(seed * 71 + 13), n = s.n, N = NODES, Fi = NODE_F + N, per = N / GROUPS;
    const X = new Float32Array(n * N * Fi), Y = new Int32Array(n * N), truth = new Int8Array(n * N), A = new Uint8Array(N * N);
    for (let e = 0; e < n; e++) {
      const grp = rng.shuffle(NA.range(N)).map((id) => Math.floor(id / per));
      A.fill(0);
      for (let i = 0; i < N; i++) for (let j = i + 1; j < N; j++) if (rng.next() < (grp[i] === grp[j] ? 0.6 : s.across)) A[i * N + j] = A[j * N + i] = 1;
      for (let i = 0; i < N; i++) {
        const mates = NA.range(N).filter((j) => j !== i && grp[j] === grp[i]);
        if (!mates.some((j) => A[i * N + j])) { const j = rng.pick(mates); A[i * N + j] = A[j * N + i] = 1; }
      }
      const said = new Uint8Array(N);
      for (let g = 0; g < GROUPS; g++) for (const i of rng.shuffle(NA.range(N).filter((q) => grp[q] === g)).slice(0, s.hints)) said[i] = 1;
      for (let i = 0; i < N; i++) {
        const row = (e * N + i) * Fi;
        if (said[i]) { X[row + grp[i]] = 1; X[row + 3] = 1; }
        for (let j = 0; j < N; j++) X[row + NODE_F + j] = A[i * N + j];
        Y[e * N + i] = said[i] ? -1 : grp[i];
        truth[e * N + i] = grp[i];
      }
    }
    const [nTr, nVa] = split(n, 0.8), R = N * Fi;
    return { family: 'graph', task: 'groups', input: { kind: 'series', len: N, dim: Fi }, output: { kind: 'tokens', vocab: GROUPS }, names: ['group A', 'group B', 'group C'], classes: GROUPS, nodes: N, feats: NODE_F,
      truth: { tr: truth.slice(0, nTr * N), va: truth.slice(nTr * N) },
      Xtr: X.slice(0, nTr * R), Ytr: Y.slice(0, nTr * N), nTr, Xva: X.slice(nTr * R), Yva: Y.slice(nTr * N), nVa,
      inText: `a graph of ${N} nodes: each one’s ${NODE_F} features and its ${N} links`, outText: `for each node, ${GROUPS} scores (its group)` };
  }
  // a curve whose noise has a size of its own at every x; the targets are [y, 0] (only the middle is read)
  G.spreadTruth = (task, noise) => ({ f: (x) => 0.6 * Math.sin(2.5 * x) + 0.2 * x, sd: task === 'gap' ? (x) => 0.05 + 0.1 * Math.abs(x) : (x) => 0.03 + (noise * (x + 1)) / 2 });
  /* More than one right answer: y from −1 to 1, and x = y's place on a wavy curve (t + 0.3·sin 2πt, with noise),
     scaled to −1…1. Predicting y from x, the middle of the curve has three right answers. The outputs: m mixture
     weights (scores), m middles and m log spreads; the targets [y, 0, …] (only y is read). */
  function branchesData(p, s, seed) {
    const rng = new NA.RNG(seed * 83 + 19), n = s.n, m = clamp(s.components | 0, 1, 5), K = 3 * m, X = new Float32Array(n), Y = new Float32Array(n * K);
    const xOf = (t, e) => (2 * (t + 0.3 * Math.sin(2 * Math.PI * t) + e + 0.1)) / 1.2 - 1;
    for (let i = 0; i < n; i++) { const t = rng.next(); X[i] = xOf(t, rng.uniform(-0.1, 0.1)); Y[i * K] = 2 * t - 1; }
    const [nTr, nVa] = split(n, 0.75);
    return { family: 'spread', task: 'branches', input: { kind: 'vector', dim: 1 }, output: { kind: 'value', k: K }, comps: m, curve: (t) => [xOf(t, 0), 2 * t - 1],
      names: [...NA.range(m).map((k) => `weight ${k + 1}`), ...NA.range(m).map((k) => `middle ${k + 1}`), ...NA.range(m).map((k) => `log spread ${k + 1}`)],
      Xtr: X.slice(0, nTr), Ytr: Y.slice(0, nTr * K), nTr, Xva: X.slice(nTr), Yva: Y.slice(nTr * K), nVa,
      inText: '1 number (x)', outText: `${K} numbers: ${m} bell curve${m > 1 ? 's' : ''}, a weight, a middle and a log spread each` };
  }
  // a mixture's curves at one x: their weights (the softmax of their scores), middles and spreads
  G.mixture = (o, base, m) => {
    let mx = -Infinity; for (let k = 0; k < m; k++) mx = Math.max(mx, o[base + k]);
    const w = []; let z = 0;
    for (let k = 0; k < m; k++) { w.push(Math.exp(o[base + k] - mx)); z += w[k]; }
    return NA.range(m).map((k) => ({ w: w[k] / z, mu: o[base + m + k], sd: Math.exp(clamp(o[base + 2 * m + k], -7, 7)) }));
  };
  function spreadData(p, s, seed) {
    if (p.task === 'branches') return branchesData(p, s, seed);
    const rng = new NA.RNG(seed * 73 + 17), n = s.n, X = new Float32Array(n), Y = new Float32Array(2 * n), T = G.spreadTruth(p.task, s.noise);
    for (let i = 0; i < n; i++) {
      let x;
      do x = rng.uniform(-1, 1); while (p.task === 'gap' && Math.abs(x) < 0.3);
      X[i] = x; Y[2 * i] = T.f(x) + T.sd(x) * rng.gauss();
    }
    const [nTr, nVa] = split(n, 0.75);
    return { family: 'spread', task: p.task, input: { kind: 'vector', dim: 1 }, output: { kind: 'value', k: 2 }, names: ['middle', 'log of the spread²'], truth: T,
      Xtr: X.slice(0, nTr), Ytr: Y.slice(0, 2 * nTr), nTr, Xva: X.slice(nTr), Yva: Y.slice(2 * nTr), nVa,
      inText: '1 number (x)', outText: '2 numbers: the middle, and the log of the spread²' };
  }
  const MAKE = { recon: reconData, gan: ganData, contrast: contrastData, segment: segmentData, graph: graphData, spread: spreadData };

  // ------------------------------------------------------------------ scores
  const r2over = (pred, Y, m) => BT.r2(pred, Y, m);
  // the overlap (intersection over union) of the model's shapes with the true ones, averaged over the 3 shapes
  function iou(out, Y, n, K) {
    const T = SEG * SEG, inter = new Float64Array(K), uni = new Float64Array(K);
    for (let i = 0; i < n * T; i++) {
      let b = 0; for (let c = 1; c < K; c++) if (out[i * K + c] > out[i * K + b]) b = c;
      const y = Y[i];
      for (let c = 1; c < K; c++) { const a = b === c, t = y === c; if (a || t) uni[c]++; if (a && t) inter[c]++; }
    }
    let s = 0, m = 0;
    for (let c = 1; c < K; c++) if (uni[c]) { s += inter[c] / uni[c]; m++; }
    return m ? s / m : 0;
  }
  // the share of validation points inside the model's band of ±2 spreads (about 95% when the spread is right)
  G.inBand = (m, d, n) => {
    const p = m.predict(d.Xva, n);
    let k = 0;
    for (let i = 0; i < n; i++) { const sd = Math.exp(0.5 * clamp(p[2 * i + 1], -7, 7)); if (Math.abs(d.Yva[2 * i] - p[2 * i]) <= 2 * sd) k++; }
    return n ? k / n : 0;
  };
  // the lab's measure for a problem with answers (undefined: the usual one)
  G.metric = function (d, m, X, Y, n, lossId) {
    if (d.family === 'recon') return r2over(m.predict(X, n), Y, n * PX);
    // more than one right answer: for each point, of the curves that carry some weight, the middle nearest its answer
    // (squared error: the one number it gives)
    if (d.family === 'spread' && d.task === 'branches') {
      const K = d.output.k, o = m.predict(X, n), pred = new Float32Array(n), y = new Float32Array(n);
      for (let i = 0; i < n; i++) {
        y[i] = Y[i * K];
        if (lossId !== 'mdn') { pred[i] = o[i * K]; continue; }
        let best = Infinity;
        for (const c of G.mixture(o, i * K, d.comps)) if (c.w >= 0.05 && Math.abs(c.mu - y[i]) < Math.abs(best - y[i])) best = c.mu;
        pred[i] = best;
      }
      return BT.r2(pred, y, n);
    }
    if (d.family === 'spread') {
      const p = m.predict(X, n), mu = new Float32Array(n), y = new Float32Array(n);
      for (let i = 0; i < n; i++) { mu[i] = p[2 * i]; y[i] = Y[2 * i]; }
      return BT.r2(mu, y, n);
    }
    if (d.family === 'segment') return iou(m.predict(X, n), Y, n, 4);
    return undefined;
  };
  G.metricName = (fam, task) => (fam === 'spread' && task === 'branches' ? 'R² of its nearest middle' : null) || ({ recon: 'R²', spread: 'R² of the middle', segment: 'overlap (IoU)', graph: 'accuracy', gan: 'quality (precision and recall)', contrast: 'nearest-code accuracy' }[fam] || null);
  /* Improved precision and recall (Kynkäänniemi and others, 2019): each set's points are balls reaching to their k-th
     nearest neighbour in that set. Precision: the share of generated points inside a ball of the real ones (they look
     real); recall: the share of real points inside a ball of the generated ones (they are all covered). */
  function precisionRecall(R, F, n, K, k = 3) {
    const d2 = (A, i, B, j) => { let s = 0; for (let c = 0; c < K; c++) { const t = A[i * K + c] - B[j * K + c]; s += t * t; } return s; };
    const radii = (A) => { const r = new Float64Array(n), row = new Float64Array(n); for (let i = 0; i < n; i++) { for (let j = 0; j < n; j++) row[j] = i === j ? Infinity : d2(A, i, A, j); r[i] = row.slice().sort()[k - 1]; } return r; };
    const rr = radii(R), rf = radii(F);
    let p = 0, q = 0;
    for (let i = 0; i < n; i++) for (let j = 0; j < n; j++) if (d2(F, i, R, j) <= rr[j]) { p++; break; }
    for (let j = 0; j < n; j++) for (let i = 0; i < n; i++) if (d2(R, j, F, i) <= rf[i]) { q++; break; }
    return { p: p / n, r: q / n };
  }
  G.precisionRecall = precisionRecall;
  /* Pictures are judged by a digit reader instead (in their 196 numbers, nearest neighbours say little about how good
     a drawing is): each drawing's 10 nearest real digits vote, and it is read when 7 of them agree and it is no
     further from the nearest than real digits mostly are from each other. Precision: the share of drawings read;
     recall: the share of the 10 digits among them (each at least 2% of the drawings). */
  function readDigits(F, n, R, yR, nR, far) {
    const K = PX, counts = new Float64Array(10), top = new Float64Array(10), at = new Int32Array(10);
    let read = 0;
    for (let i = 0; i < n; i++) {
      top.fill(Infinity);
      for (let j = 0; j < nR; j++) {
        let s = 0; for (let c = 0; c < K; c++) { const t = F[i * K + c] - R[j * K + c]; s += t * t; }
        if (s >= top[9]) continue;
        let t = 9; while (t > 0 && top[t - 1] > s) { top[t] = top[t - 1]; at[t] = at[t - 1]; t--; }
        top[t] = s; at[t] = j;
      }
      const v = new Float64Array(10);
      for (let t = 0; t < 10; t++) v[yR[at[t]]]++;
      let b = 0; for (let c = 1; c < 10; c++) if (v[c] > v[b]) b = c;
      if (v[b] >= 7 && Math.sqrt(top[0]) <= far) { read++; counts[b]++; }
    }
    return { p: read / n, r: [...counts].filter((c) => c >= 0.02 * n).length / 10 };
  }
  // how far a real digit is from its nearest other (90 in 100 are nearer), and a quarter more: the reader's reach
  function readerReach(R, nR) {
    const near = [];
    for (let i = 0; i < Math.min(300, nR); i++) { let b = Infinity; for (let j = 0; j < nR; j++) if (j !== i) { let s = 0; for (let c = 0; c < PX; c++) { const t = R[i * PX + c] - R[j * PX + c]; s += t * t; } b = Math.min(b, s); } near.push(Math.sqrt(b)); }
    near.sort((a, b) => a - b);
    return 1.25 * near[Math.floor(0.9 * near.length)];
  }
  // each row to length 1 (codes compared by their angle)
  function unitRows(A, n, D) { const out = new Float32Array(n * D); for (let i = 0; i < n; i++) { let s = 0; for (let c = 0; c < D; c++) s += A[i * D + c] ** 2; s = Math.sqrt(s) || 1; for (let c = 0; c < D; c++) out[i * D + c] = A[i * D + c] / s; } return out; }
  // the share of queries whose 5 most similar references mostly have their label
  function knnAccuracy(ref, yRef, nR, q, yQ, nQ, D, K, k = 5) {
    let right = 0;
    const top = new Float64Array(k), at = new Int32Array(k);
    for (let i = 0; i < nQ; i++) {
      top.fill(-Infinity);
      for (let j = 0; j < nR; j++) {
        let s = 0; for (let c = 0; c < D; c++) s += q[i * D + c] * ref[j * D + c];
        if (s <= top[k - 1]) continue;
        let t = k - 1; while (t > 0 && top[t - 1] < s) { top[t] = top[t - 1]; at[t] = at[t - 1]; t--; }
        top[t] = s; at[t] = j;
      }
      const votes = new Float64Array(K);
      for (let t = 0; t < k; t++) votes[yRef[at[t]]] += 1 + 1e-3 * top[t];
      let best = 0; for (let c = 1; c < K; c++) if (votes[c] > votes[best]) best = c;
      if (best === yQ[i]) right++;
    }
    return nQ ? right / nQ : 0;
  }
  G.knnAccuracy = knnAccuracy;

  // ------------------------------------------------------------------ the GAN
  /* Your model is the generator G: random numbers in, an example out. Its rival, the discriminator D (two leaky-ReLU
     layers, 32 or 64 wide), gives a score whose sigmoid is how sure it is that an example is real. Each step: D learns
     from a batch of real examples (answer 1) and of G's (answer 0) by binary cross-entropy; then G learns to make D
     answer 1 for its own, −log D(G(z)) (the non-saturating loss of Goodfellow and others, 2014), its gradient passed
     back through D. Both with the optimiser set, Adam's β₁ 0.5 (as DCGAN). A picture's pixels are the sigmoid of the
     generator's outputs, from 0 to 1 as the real ones are (DCGAN's generator ends in tanh the same way). */
  class Gan {
    constructor(o) {
      const d = o.data, K = d.K, wide = d.disc === 'big' ? 64 : 32;
      Object.assign(this, { kind: 'gan', net: o.net, data: d, K, z: d.z, rng: new NA.RNG(o.seed), B: Math.max(4, Math.min(256, o.batch || 64)), dAcc: null, dLoss: null, gLoss: null, P: null, R: null });
      this.D = NA.bx.build([{ type: 'dense', units: wide, act: 'leaky' }, { type: 'dense', units: wide, act: 'leaky' }], { kind: 'vector', dim: K }, { kind: 'value', k: 1 }, { seed: o.seed * 7 + 3, inputGrad: true });
      this.Z = new Float32Array(this.B * this.z); this.XD = new Float32Array(2 * this.B * K); this.sq = d.pixels ? new Float32Array(this.B * K) : null;
    }
    // the generator's examples from its outputs: pictures squashed to 0…1, points as they are
    look(out, n) { if (!this.data.pixels) return out; const a = new Float32Array(n * this.K); for (let i = 0; i < a.length; i++) a[i] = sigmoid(out[i]); return a; }
    step(opt) {
      const { B, K, z, rng, data: d } = this, o = Object.assign({}, opt, { beta1: 0.5 });
      for (let i = 0; i < B * z; i++) this.Z[i] = rng.gauss();
      for (let b = 0; b < B; b++) { const k = rng.int(d.nTr); this.XD.set(d.Ytr.subarray(k * K, (k + 1) * K), b * K); }
      this.XD.set(this.look(this.net.forward(this.Z, B, true), B).subarray(0, B * K), B * K);
      let right = 0;
      const dr = this.D.stepWith(this.XD, 2 * B, (out, dOut) => {
        let L = 0;
        for (let i = 0; i < 2 * B; i++) { const y = i < B ? 1 : 0, l = out[i]; L += softplus(l) - y * l; dOut[i] = (sigmoid(l) - y) / (2 * B); if (l > 0 === (y === 1)) right++; }
        return L / (2 * B);
      }, o);
      this.dAcc = this.dAcc === null ? right / (2 * B) : 0.95 * this.dAcc + 0.05 * (right / (2 * B));
      const gr = this.net.stepWith(this.Z, B, (out, dOut) => {
        let L = 0;
        const sq = this.sq, x = sq || out;
        if (sq) for (let i = 0; i < B * K; i++) sq[i] = sigmoid(out[i]);
        const dx = this.D.gradWith(x, B, (l, dl) => { for (let b = 0; b < B; b++) { L += softplus(-l[b]); dl[b] = (sigmoid(l[b]) - 1) / B; } });
        for (let i = 0; i < B * K; i++) dOut[i] = sq ? dx[i] * sq[i] * (1 - sq[i]) : dx[i];
        return L / B;
      }, o);
      this.dLoss = dr.loss; this.gLoss = gr.loss;
      return { loss: gr.loss, gradNorm: gr.gradNorm };
    }
    // the F1 of precision and recall: the generator's examples from the validation noise against the real validation
    // ones (points), or what the digit reader makes of its drawings (pictures)
    score() {
      const d = this.data, n = Math.min(d.nVa, d.pixels ? 200 : 300), F = this.look(this.net.predict(d.Xva, n), n);
      if (d.pixels && this.reach === undefined) this.reach = readerReach(d.Ytr, d.nTr);
      const r = d.pixels ? readDigits(F, n, d.Ytr, d.labels.tr, d.nTr, this.reach) : precisionRecall(d.Yva, F, n, this.K);
      this.P = r.p; this.R = r.r;
      return r.p + r.r > 0 ? (2 * r.p * r.r) / (r.p + r.r) : 0;
    }
  }

  // ------------------------------------------------------------------ contrastive learning
  /* One random view of a picture: shifted up to 2 pixels (1 if gentle), its strokes sometimes thickened, its ink made
     fainter or stronger, speckled, and (strong views) a 4 × 4 square sometimes rubbed out. */
  function makeView(x, out, rng, strong, C = 1) {
    const s = strong ? 2 : 1, dx = rng.int(2 * s + 1) - s, dy = rng.int(2 * s + 1) - s, thick = rng.next() < (strong ? 0.4 : 0.2), k = rng.uniform(strong ? 0.6 : 0.8, strong ? 1.3 : 1.15), sd = strong ? 0.1 : 0.04;
    const ex = strong && rng.next() < 0.3 ? [rng.int(IMG - 4), rng.int(IMG - 4)] : null;
    for (let ch = 0; ch < C; ch++) {
      const at = (qx, qy) => (qx >= 0 && qy >= 0 && qx < IMG && qy < IMG ? x[ch * PX + qy * IMG + qx] : 0);
      for (let yy = 0; yy < IMG; yy++) for (let xx = 0; xx < IMG; xx++) {
        const sx = xx - dx, sy = yy - dy;
        let v = at(sx, sy);
        if (thick) v = Math.max(v, 0.7 * Math.max(at(sx - 1, sy), at(sx + 1, sy), at(sx, sy - 1), at(sx, sy + 1)));
        if (ex && xx >= ex[0] && xx < ex[0] + 4 && yy >= ex[1] && yy < ex[1] + 4) v = 0;
        out[ch * PX + yy * IMG + xx] = clamp(v * k, 0, 1) + sd * rng.gauss();
      }
    }
  }
  G.makeView = makeView;
  /* NT-Xent (Chen and others, 2020): 2B codes, each the other view's partner. Each code, made length 1, should be
     closer to its partner than to every other: the loss of each is −sim(partner)/τ + log Σ over the others of
     exp(sim/τ), a cross-entropy whose right answer is the partner. Writes the gradient for the codes as they came out
     (through the making of length 1) into dOut; returns the mean loss and leaves how often the partner was the
     closest in T.pairAcc. */
  function ntxent(out, dOut, N, D, tau, T) {
    const B = N / 2;
    if (!T.zb || T.zb.length !== N * D) { T.zb = new Float64Array(N * D); T.nb = new Float64Array(N); T.pb = new Float64Array(N * N); T.gb = new Float64Array(D); }
    const z = T.zb, nrm = T.nb, P = T.pb, g = T.gb;
    for (let i = 0; i < N; i++) { let s = 0; for (let c = 0; c < D; c++) s += out[i * D + c] ** 2; nrm[i] = Math.sqrt(s) + 1e-8; for (let c = 0; c < D; c++) z[i * D + c] = out[i * D + c] / nrm[i]; }
    let L = 0, right = 0;
    for (let i = 0; i < N; i++) {
      const pi = i < B ? i + B : i - B;
      let mx = -Infinity, best = -1;
      for (let k = 0; k < N; k++) {
        if (k === i) { P[i * N + k] = 0; continue; }
        let s = 0; for (let c = 0; c < D; c++) s += z[i * D + c] * z[k * D + c];
        P[i * N + k] = s / tau;
        if (s / tau > mx) { mx = s / tau; best = k; }
      }
      if (best === pi) right++;
      const sp = P[i * N + pi];
      let sum = 0;
      for (let k = 0; k < N; k++) if (k !== i) { const e = Math.exp(P[i * N + k] - mx); P[i * N + k] = e; sum += e; }
      for (let k = 0; k < N; k++) if (k !== i) P[i * N + k] /= sum;
      L += -sp + mx + Math.log(sum);
    }
    // on the unit codes: (1 / Nτ)[Σ over k ≠ i of (Pᵢₖ + Pₖᵢ)·zₖ − 2·z of its partner]; then through zᵢ = uᵢ / |uᵢ|
    for (let i = 0; i < N; i++) {
      const pi = i < B ? i + B : i - B;
      for (let c = 0; c < D; c++) g[c] = -2 * z[pi * D + c];
      for (let k = 0; k < N; k++) if (k !== i) { const w = P[i * N + k] + P[k * N + i]; for (let c = 0; c < D; c++) g[c] += w * z[k * D + c]; }
      let dot = 0; for (let c = 0; c < D; c++) dot += g[c] * z[i * D + c];
      for (let c = 0; c < D; c++) dOut[i * D + c] = (g[c] - dot * z[i * D + c]) / (nrm[i] * N * tau);
    }
    T.pairAcc = right / N;
    return L / N;
  }
  G.ntxent = ntxent;
  class Contrast {
    constructor(o) {
      const d = o.data;
      Object.assign(this, { kind: 'contrast', net: o.net, data: d, rng: new NA.RNG(o.seed), B: Math.max(4, Math.min(128, o.batch || 32)), tau: d.tau || 0.5, D: d.output.k, C: d.input.c || 1, pairAcc: null, emb: null });
      this.S = this.C * PX; this.V = new Float32Array(2 * this.B * this.S);
      // the same vote on the raw pixels, for comparison
      const nR = Math.min(d.nTr, 1000), nQ = Math.min(d.nVa, 300);
      this.raw = knnAccuracy(unitRows(d.Xtr, nR, this.S), d.labels.tr, nR, unitRows(d.Xva, nQ, this.S), d.labels.va, nQ, this.S, d.classes);
    }
    step(opt) {
      const { B, S, rng, data: d } = this, strong = d.views !== 'gentle';
      for (let b = 0; b < B; b++) {
        const k = rng.int(d.nTr), x = d.Xtr.subarray(k * S, (k + 1) * S);
        makeView(x, this.V.subarray(b * S, (b + 1) * S), rng, strong, this.C);
        makeView(x, this.V.subarray((B + b) * S, (B + b + 1) * S), rng, strong, this.C);
      }
      const r = this.net.stepWith(this.V, 2 * B, (out, dOut) => ntxent(out, dOut, 2 * B, this.D, this.tau, this));
      return { loss: r.loss, gradNorm: r.gradNorm };
    }
    // the vote of the 5 nearest training codes for each validation picture, the labels used only here
    score() {
      const d = this.data, nR = Math.min(d.nTr, 1000), nQ = Math.min(d.nVa, 300), D = this.D;
      const q = unitRows(this.net.predict(d.Xva, nQ), nQ, D);
      this.emb = { codes: q, n: nQ };
      return knnAccuracy(unitRows(this.net.predict(d.Xtr, nR), nR, D), d.labels.tr, nR, q, d.labels.va, nQ, D, d.classes);
    }
  }
  // the trainer of a problem that is not examples with answers (null: the usual training)
  G.trainerFor = (d, net, o) => (d.family === 'gan' ? new Gan(Object.assign({ data: d, net }, o)) : d.family === 'contrast' ? new Contrast(Object.assign({ data: d, net }, o)) : null);

  // ------------------------------------------------------------------ the middle of an autoencoder
  /* The narrowest layer before the output that gives a list of numbers, or a Sample block (a VAE's middle): where a
     picture's code is. { i, k, sample }, or null. */
  G.bottleneck = (m) => {
    if (!m || !m.layers) return null;
    const Ls = m.layers.slice(0, -1);
    const si = Ls.findIndex((L) => L.type === 'sample');
    if (si >= 0) return { i: si, k: Ls[si].outShape[0], sample: true };
    let best = -1;
    Ls.forEach((L, i) => { if (L.outShape && L.outShape.length === 1 && (best < 0 || L.outShape[0] < Ls[best].outShape[0])) best = i; });
    return best >= 0 && best < Ls.length - 1 ? { i: best, k: Ls[best].outShape[0], sample: false } : null;
  };
  // the two directions along which codes spread most (principal components), and every code placed on them
  function pca2(Z, n, Dm) {
    const mean = new Float64Array(Dm), C = new Float64Array(Dm * Dm), vs = [];
    for (let i = 0; i < n; i++) for (let c = 0; c < Dm; c++) mean[c] += Z[i * Dm + c] / n;
    for (let i = 0; i < n; i++) for (let a = 0; a < Dm; a++) { const za = Z[i * Dm + a] - mean[a]; for (let b = 0; b < Dm; b++) C[a * Dm + b] += za * (Z[i * Dm + b] - mean[b]); }
    for (let k = 0; k < Math.min(2, Dm); k++) {
      let v = Float64Array.from({ length: Dm }, (_, i) => Math.sin(1.7 * i + k + 1));
      for (let it = 0; it < 80; it++) {
        const w = new Float64Array(Dm);
        for (let a = 0; a < Dm; a++) for (let b = 0; b < Dm; b++) w[a] += C[a * Dm + b] * v[b];
        for (const u of vs) { let pr = 0; for (let a = 0; a < Dm; a++) pr += w[a] * u[a]; for (let a = 0; a < Dm; a++) w[a] -= pr * u[a]; }
        let s = 0; for (let a = 0; a < Dm; a++) s += w[a] * w[a];
        s = Math.sqrt(s) || 1; v = w.map((q) => q / s);
      }
      vs.push(v);
    }
    const out = new Float32Array(2 * n);
    for (let i = 0; i < n; i++) for (let k = 0; k < vs.length; k++) { let s = 0; for (let c = 0; c < Dm; c++) s += (Z[i * Dm + c] - mean[c]) * vs[k][c]; out[2 * i + k] = s; }
    return out;
  }
  G.pca2 = pca2;
  // a computed picture kept until the model changes (its version)
  const memo = new WeakMap();
  const kept = (obj, key, fn) => { if (!obj) return fn(); const m = memo.get(obj); if (m && m.key === key) return m.v; const v = fn(); memo.set(obj, { key, v }); return v; };

  // ------------------------------------------------------------------ drawing
  const theme = NA.theme;
  const rgb = (name) => (theme.tokens && theme.tokens[name]) || [128, 128, 128];
  const fontOf = (w, px, mono) => { const cs = typeof document !== 'undefined' ? getComputedStyle(document.body) : null; return `${w} ${px}px ${(cs && cs.getPropertyValue(mono ? '--font-mono' : '--font-ui')) || (mono ? 'monospace' : 'sans-serif')}`; };
  const text = (ctx, s, x, y, o = {}) => { ctx.font = fontOf(o.w || 500, o.size || 11, o.mono); ctx.fillStyle = o.color || theme.c('ink'); ctx.textAlign = o.align || 'left'; ctx.textBaseline = o.base || 'middle'; ctx.fillText(s, x, y); };
  const CAT = ['s1', 's2', 's4', 's7', 's3', 's5', 's6', 's8'];
  const SEG_COLOURS = [null, 's1', 's2', 's4'];
  const GRP_COLOURS = ['s1', 's2', 's4'];
  let picC = null;
  // a grey picture (G × G numbers from 0 to 1, from X at off) drawn s wide; or a mask of classes in their colours
  function pic(ctx, X, off, x, y, s, o = {}) {
    const Gs = o.G || IMG, P = Gs * Gs;
    if (!picC || picC.width !== Gs) { picC = document.createElement('canvas'); picC.width = picC.height = Gs; }
    const g = picC.getContext('2d'), id = g.createImageData(Gs, Gs), ink = rgb('ink'), bg = rgb('panel-2'), cols = o.mask ? SEG_COLOURS.map((c) => (c ? rgb(c) : bg)) : null;
    for (let p = 0; p < P; p++) {
      if (cols) { const c = cols[X[off + p]] || bg; for (let ch = 0; ch < 3; ch++) id.data[p * 4 + ch] = c[ch]; }
      else { const v = clamp(X[off + p], 0, 1); for (let ch = 0; ch < 3; ch++) id.data[p * 4 + ch] = bg[ch] + (ink[ch] - bg[ch]) * v; }
      id.data[p * 4 + 3] = 255;
    }
    g.putImageData(id, 0, 0);
    ctx.imageSmoothingEnabled = !o.mask; ctx.drawImage(picC, x, y, s, s); ctx.imageSmoothingEnabled = true;
    if (o.frame) { ctx.strokeStyle = theme.c(o.frame, 0.9); ctx.lineWidth = 1.6; ctx.strokeRect(x - 0.5, y - 0.5, s + 1, s + 1); }
  }
  const label = (ctx, s, x, y, w, o = {}) => { ctx.font = fontOf(o.w || 600, o.size || 10.5); let t = s; while (t.length > 1 && ctx.measureText(t).width > w) t = t.slice(0, -2) + '…'; text(ctx, t, x, y, Object.assign({ size: 10.5, w: 600, color: theme.c('muted') }, o)); };
  const square = (r) => { const s = Math.min(r.w, r.h); return { x: r.x + (r.w - s) / 2, y: r.y + (r.h - s) / 2, s }; };

  // autoencoders: pictures and what it rebuilt; the map of the codes and the pictures drawn from codes, beside them
  function drawRecon(ctx, r, L) {
    const d = L.data, m = L.model, noisy = d.task === 'denoise', bn = m ? G.bottleneck(m) : null, map = bn && (bn.k === 2 || bn.sample || bn.k <= 32);
    const side = map && r.w > r.h * 1.15, gridW = map ? (side ? r.w * 0.5 - 8 : r.w) : r.w, gridH = map && !side ? r.h * 0.48 : r.h;
    const per = noisy ? 3 : 2, n = Math.min(d.nVa, 12), cols = Math.max(1, Math.floor(gridW / (per * 44 + 10))), rows = Math.ceil(n / cols);
    const cell = Math.max(10, Math.min((gridW - cols * 10) / (cols * per), (gridH - 16) / rows - 6, 60));
    const rec = m ? kept(m, `${L.vizVersion}:rec`, () => Float32Array.from(m.predict(d.Xva, n))) : null;
    label(ctx, noisy ? 'given · rebuilt · the clean picture' : m ? 'the picture · rebuilt from its code' : 'the pictures: press Train to squeeze and rebuild them', r.x, r.y + 6, gridW);
    for (let i = 0; i < n; i++) {
      const cx = r.x + (i % cols) * (per * cell + 10), cy = r.y + 16 + Math.floor(i / cols) * (cell + 6);
      pic(ctx, d.Xva, i * PX, cx, cy, cell);
      if (rec) pic(ctx, rec, i * PX, cx + cell + 2, cy, cell);
      else { ctx.strokeStyle = theme.c('line'); ctx.setLineDash([3, 3]); ctx.strokeRect(cx + cell + 2.5, cy + 0.5, cell - 1, cell - 1); ctx.setLineDash([]); }
      if (noisy) pic(ctx, d.Yva, i * PX, cx + 2 * cell + 4, cy, cell);
    }
    if (!map) return;
    const box = side ? { x: r.x + r.w * 0.5 + 8, y: r.y, w: r.w * 0.5 - 8, h: r.h } : { x: r.x, y: r.y + gridH + 8, w: r.w, h: r.h - gridH - 8 };
    const two = bn.k === 2, half = two || bn.sample;
    const mapBox = half ? (box.w > box.h * 1.4 ? { x: box.x, y: box.y, w: box.w / 2 - 6, h: box.h } : { x: box.x, y: box.y, w: box.w, h: box.h / 2 - 6 }) : box;
    const nM = Math.min(d.nVa, 300), codes = kept(m, `${L.vizVersion}:codes`, () => { const c = Float32Array.from(m.encode(d.Xva, nM, bn.i).subarray(0, nM * bn.k)); return { c, xy: bn.k === 2 ? c : pca2(c, nM, bn.k) }; });
    codeMap(ctx, mapBox, codes.xy, nM, d.labels.va, d.names, bn.k === 2 ? `its codes: ${nM} pictures at their 2 numbers` : `its codes, ${bn.k} numbers flattened to 2 (PCA)`);
    if (!half) return;
    const genBox = mapBox.w < box.w ? { x: box.x + box.w / 2 + 6, y: box.y, w: box.w / 2 - 6, h: box.h } : { x: box.x, y: box.y + box.h / 2 + 6, w: box.w, h: box.h / 2 - 6 };
    // codes on a grid (2 numbers: over the bell curve, or over the codes' own range) or drawn from the bell curve
    const g = 6, Z = new Float32Array(g * g * bn.k);
    let lo = [-2, -2], hi = [2, 2];
    if (!bn.sample) { lo = [Infinity, Infinity]; hi = [-Infinity, -Infinity]; for (let i = 0; i < nM; i++) for (let a = 0; a < 2; a++) { lo[a] = Math.min(lo[a], codes.c[2 * i + a]); hi[a] = Math.max(hi[a], codes.c[2 * i + a]); } }
    // (a VAE's codes at even steps of the bell curve's probability: the middle 1/12 to 11/12 of it, where its codes are)
    const rz = new NA.RNG(5), Q = [-1.383, -0.674, -0.21, 0.21, 0.674, 1.383];
    for (let q = 0; q < g * g; q++) {
      if (two && bn.sample) { Z[2 * q] = Q[q % g]; Z[2 * q + 1] = -Q[Math.floor(q / g)]; }
      else if (two) { Z[2 * q] = lo[0] + ((q % g) + 0.5) / g * (hi[0] - lo[0]); Z[2 * q + 1] = hi[1] - (Math.floor(q / g) + 0.5) / g * (hi[1] - lo[1]); }
      else for (let c = 0; c < bn.k; c++) Z[q * bn.k + c] = rz.gauss();
    }
    const outs = kept(m.layers[bn.i], `${L.vizVersion}:dec`, () => Float32Array.from(m.decode(bn.i + 1, Z, g * g).subarray(0, g * g * PX)));
    const s = Math.min((genBox.w - 4) / g, (genBox.h - 18) / g);
    label(ctx, bn.sample ? (two ? 'new pictures from codes across the bell curve' : 'new pictures from codes drawn at random') : 'pictures decoded from codes across the map', genBox.x, genBox.y + 6, genBox.w);
    for (let q = 0; q < g * g; q++) pic(ctx, outs, q * PX, genBox.x + (q % g) * s, genBox.y + 16 + Math.floor(q / g) * s, s - 1);
  }
  // points on a plane, each the numeral (or first letters) of its class in that class's colour
  function codeMap(ctx, b, xy, n, lab, names, title) {
    label(ctx, title, b.x, b.y + 6, b.w);
    const q = { x: b.x, y: b.y + 16, w: b.w, h: b.h - 16 };
    let x0 = Infinity, x1 = -Infinity, y0 = Infinity, y1 = -Infinity;
    for (let i = 0; i < n; i++) { x0 = Math.min(x0, xy[2 * i]); x1 = Math.max(x1, xy[2 * i]); y0 = Math.min(y0, xy[2 * i + 1]); y1 = Math.max(y1, xy[2 * i + 1]); }
    const sx = (x1 - x0) || 1, sy = (y1 - y0) || 1;
    ctx.fillStyle = theme.c('panel-2'); ctx.beginPath(); NA.rr(ctx, q.x, q.y, q.w, q.h, 6); ctx.fill();
    const size = clamp(Math.min(q.w, q.h) / 28, 8, 12);
    for (let i = 0; i < n; i++) {
      const c = lab[i], px = q.x + 8 + ((xy[2 * i] - x0) / sx) * (q.w - 16), py = q.y + q.h - 8 - ((xy[2 * i + 1] - y0) / sy) * (q.h - 16);
      text(ctx, names[c].length > 2 ? names[c].slice(0, 2) : names[c], px, py, { size, w: 700, align: 'center', mono: true, color: theme.c(CAT[c % CAT.length]) });
    }
  }
  // the GAN: the plane with the discriminator's verdict behind, real points and generated ones with where they are pushed
  function drawGan(ctx, r, L) {
    const d = L.data, m = L.model, gen = L.gen && L.gen.net === m ? L.gen : null;
    if (d.pixels) return drawGanPictures(ctx, r, L, gen);
    const q = square(r), S = 1.25, X = (v) => q.x + ((v + S) / (2 * S)) * q.s, Y = (v) => q.y + ((S - v) / (2 * S)) * q.s;
    ctx.save(); ctx.beginPath(); NA.rr(ctx, q.x, q.y, q.s, q.s, 8); ctx.clip();
    ctx.fillStyle = theme.c('panel-2'); ctx.fillRect(q.x, q.y, q.s, q.s);
    if (gen) {
      const g = 40, img = kept(gen.D, `${L.vizVersion}:${theme.dark ? 1 : 0}`, () => {
        const P = new Float32Array(g * g * 2);
        for (let i = 0; i < g; i++) for (let j = 0; j < g; j++) { P[(i * g + j) * 2] = -S + ((j + 0.5) / g) * 2 * S; P[(i * g + j) * 2 + 1] = S - ((i + 0.5) / g) * 2 * S; }
        const o = gen.D.predict(P, g * g), c = document.createElement('canvas'); c.width = c.height = g;
        const cx = c.getContext('2d'), id = cx.createImageData(g, g), bg = rgb('panel-2'), hi = rgb('s1');
        for (let p = 0; p < g * g; p++) { const a = 0.55 * sigmoid(o[p]); for (let ch = 0; ch < 3; ch++) id.data[p * 4 + ch] = bg[ch] + (hi[ch] - bg[ch]) * a; id.data[p * 4 + 3] = 255; }
        cx.putImageData(id, 0, 0);
        return c;
      });
      ctx.imageSmoothingEnabled = true; ctx.drawImage(img, q.x, q.y, q.s, q.s);
    }
    const nR = Math.min(d.nTr, 500);
    ctx.fillStyle = theme.c('ink-2', 0.5);
    for (let i = 0; i < nR; i++) { ctx.beginPath(); ctx.arc(X(d.Ytr[2 * i]), Y(d.Ytr[2 * i + 1]), 1.6, 0, Math.PI * 2); ctx.fill(); }
    if (m) {
      const n = Math.min(d.nVa, 200), fk = kept(m, `${L.vizVersion}:fake`, () => Float32Array.from(m.predict(d.Xva, n).subarray(0, 2 * n)));
      // where the discriminator pushes each point: down the generator's loss, −∂(−log D)/∂x
      if (gen) {
        const push = kept(gen, `${L.vizVersion}:push`, () => Float32Array.from(gen.D.gradWith(fk, n, (l, dl) => { for (let b = 0; b < n; b++) dl[b] = sigmoid(l[b]) - 1; }).subarray(0, 2 * n)));
        let mx = 1e-9; for (let i = 0; i < n; i++) mx = Math.max(mx, Math.hypot(push[2 * i], push[2 * i + 1]));
        ctx.strokeStyle = theme.c('s2', 0.6); ctx.lineWidth = 1.2;
        for (let i = 0; i < n; i++) { const f = (0.07 * q.s) / mx, x = X(fk[2 * i]), y = Y(fk[2 * i + 1]); ctx.beginPath(); ctx.moveTo(x, y); ctx.lineTo(x - push[2 * i] * f, y + push[2 * i + 1] * f); ctx.stroke(); }
      }
      for (let i = 0; i < n; i++) { ctx.beginPath(); ctx.arc(X(fk[2 * i]), Y(fk[2 * i + 1]), 2.8, 0, Math.PI * 2); ctx.fillStyle = theme.c('s2'); ctx.fill(); ctx.strokeStyle = theme.c('panel', 0.9); ctx.lineWidth = 1; ctx.stroke(); }
    }
    ctx.restore();
    ctx.strokeStyle = theme.c('line-2'); ctx.lineWidth = 1; ctx.beginPath(); NA.rr(ctx, q.x + 0.5, q.y + 0.5, q.s - 1, q.s - 1, 8); ctx.stroke();
  }
  function drawGanPictures(ctx, r, L, gen) {
    const d = L.data, m = L.model, cols = r.w > r.h * 1.3 ? 8 : 6, n = Math.min(d.nVa, cols * 4), half = Math.ceil(n / cols), cell = Math.min(r.w / cols, (r.h - 34) / (half + 1));
    label(ctx, m ? 'drawn by the generator from random numbers (green frame: the discriminator thinks it real)' : 'press Train: the generator learns to draw', r.x, r.y + 6, r.w);
    if (m) {
      const fk = kept(m, `${L.vizVersion}:fake`, () => { const o = m.predict(d.Xva, n), a = new Float32Array(n * PX); for (let i = 0; i < a.length; i++) a[i] = sigmoid(o[i]); return a; }), sure = gen ? kept(gen, `${L.vizVersion}:sure`, () => Float32Array.from(gen.D.predict(fk, n).subarray(0, n))) : null;
      for (let i = 0; i < n; i++) pic(ctx, fk, i * PX, r.x + (i % cols) * cell + 2, r.y + 16 + Math.floor(i / cols) * cell + 2, cell - 6, { frame: sure ? (sure[i] > 0 ? 'good' : 'line') : null });
    }
    const y0 = r.y + 16 + half * cell + 6;
    label(ctx, 'real digits, for comparison', r.x, y0 + 4, r.w);
    for (let i = 0; i < cols; i++) pic(ctx, d.Yva, i * PX, r.x + i * cell + 2, y0 + 14, Math.min(cell - 6, r.y + r.h - y0 - 16));
  }
  // contrastive learning: the codes of the validation pictures flattened to 2 (labels for the colours only), and views
  function drawContrast(ctx, r, L) {
    const d = L.data, gen = L.gen && L.gen.net === L.model ? L.gen : null, side = r.w > r.h * 1.2;
    const vb = side ? { x: r.x, y: r.y, w: Math.min(r.w * 0.34, 260), h: r.h } : { x: r.x, y: r.y, w: r.w, h: Math.min(r.h * 0.34, 140) };
    const mb = side ? { x: vb.x + vb.w + 14, y: r.y, w: r.w - vb.w - 14, h: r.h } : { x: r.x, y: vb.y + vb.h + 10, w: r.w, h: r.h - vb.h - 10 };
    // two views of a few pictures, the same ones every time
    const rng = new NA.RNG(31), S = (d.input.c || 1) * PX, v = new Float32Array(S), rowsN = side ? 4 : 2, colsN = side ? 1 : 3, cell = Math.min(vb.w / (colsN * 3 + 0.5), (vb.h - 16) / rowsN - 4);
    label(ctx, 'a picture and two views of it', vb.x, vb.y + 6, vb.w);
    for (let i = 0; i < rowsN * colsN; i++) {
      const x0 = vb.x + (i % colsN) * (3 * cell + 10), y0 = vb.y + 16 + Math.floor(i / colsN) * (cell + 4), x = d.Xva.subarray(i * S, (i + 1) * S);
      pic(ctx, x, 0, x0, y0, cell - 2);
      for (let k = 1; k <= 2; k++) { makeView(x, v, rng, d.views !== 'gentle', d.input.c || 1); pic(ctx, v, 0, x0 + k * cell + 2, y0, cell - 2); }
    }
    if (gen && gen.emb) {
      const xy = kept(gen, `${L.vizVersion}:pca`, () => pca2(gen.emb.codes, gen.emb.n, gen.D));
      codeMap(ctx, mb, xy, gen.emb.n, d.labels.va, d.names, `its codes for ${gen.emb.n} pictures it never saw, flattened to 2 (PCA); the labels only colour them`);
    } else label(ctx, 'press Train: no labels are used to learn, only the two views', mb.x, mb.y + 6, mb.w);
  }
  // segmentation: each picture, the true shapes and the model's
  function drawSegment(ctx, r, L) {
    const d = L.data, m = L.model, P = SEG * SEG, triW = m ? 3 : 2, cols = Math.max(1, Math.floor(r.w / (triW * 50 + 12))), n = Math.min(d.nVa, cols * Math.max(1, Math.floor((r.h - 16) / 56)));
    const cell = Math.min((r.w - cols * 12) / (cols * triW), (r.h - 16) / Math.ceil(n / cols) - 6);
    label(ctx, m ? 'the picture · the true shapes · the model’s' : 'the picture · the true shapes', r.x, r.y + 6, r.w);
    const lab = m ? kept(m, `${L.vizVersion}:seg`, () => { const o = m.predict(d.Xva, n), out = new Int32Array(n * P); for (let i = 0; i < n * P; i++) { let b = 0; for (let c = 1; c < 4; c++) if (o[i * 4 + c] > o[i * 4 + b]) b = c; out[i] = b; } return out; }) : null;
    for (let i = 0; i < n; i++) {
      const x0 = r.x + (i % cols) * (triW * cell + 12), y0 = r.y + 16 + Math.floor(i / cols) * (cell + 6);
      pic(ctx, d.Xva, i * P, x0, y0, cell - 2, { G: SEG });
      pic(ctx, d.Yva, i * P, x0 + cell, y0, cell - 2, { G: SEG, mask: true });
      if (lab) pic(ctx, lab, i * P, x0 + 2 * cell, y0, cell - 2, { G: SEG, mask: true });
    }
  }
  // graphs: each group on its own arc; lines for links; the fill is the model's answer, the ring the true group
  function drawGraph(ctx, r, L) {
    const d = L.data, m = L.model, N = d.nodes, Fi = d.feats + N, cols = r.w > r.h * 1.4 ? 3 : 2, rows = 2, n = Math.min(d.nVa, cols * rows), cw = r.w / cols, ch = (r.h - 16) / rows;
    const out = m ? kept(m, `${L.vizVersion}:graph`, () => Float32Array.from(m.predict(d.Xva, n).subarray(0, n * N * GROUPS))) : null;
    label(ctx, m ? 'fill: the model’s group · ring: the true one · solid: says its group · ✗ wrong' : 'ring: the true group · solid: says its group', r.x, r.y + 6, r.w);
    for (let e = 0; e < n; e++) {
      const bx = r.x + (e % cols) * cw, by = r.y + 16 + Math.floor(e / cols) * ch, R = Math.min(cw, ch) * 0.36, cx = bx + cw / 2, cy = by + ch / 2, pos = [], seen = [0, 0, 0];
      for (let i = 0; i < N; i++) {
        const g = d.truth.va[e * N + i], k = seen[g]++, a = -Math.PI / 2 + (g * 2 * Math.PI) / GROUPS + (k - 1.5) * 0.42;
        pos.push([cx + R * Math.cos(a), cy + R * Math.sin(a)]);
      }
      ctx.strokeStyle = theme.c('line-2'); ctx.lineWidth = 1;
      for (let i = 0; i < N; i++) for (let j = i + 1; j < N; j++) if (d.Xva[(e * N + i) * Fi + d.feats + j] > 0.5) { ctx.beginPath(); ctx.moveTo(pos[i][0], pos[i][1]); ctx.lineTo(pos[j][0], pos[j][1]); ctx.stroke(); }
      const rad = clamp(R / 7, 4, 9);
      for (let i = 0; i < N; i++) {
        const t = d.truth.va[e * N + i], said = d.Xva[(e * N + i) * Fi + 3] > 0.5, [x, y] = pos[i];
        let b = t;
        if (out && !said) { b = 0; for (let c = 1; c < GROUPS; c++) if (out[(e * N + i) * GROUPS + c] > out[(e * N + i) * GROUPS + b]) b = c; }
        ctx.beginPath(); ctx.arc(x, y, rad, 0, Math.PI * 2);
        ctx.fillStyle = said || out ? theme.c(GRP_COLOURS[b], said ? 1 : 0.55) : theme.c('panel-2'); ctx.fill();
        ctx.strokeStyle = theme.c(GRP_COLOURS[t]); ctx.lineWidth = said ? 2.6 : 2; ctx.stroke();
        if (out && !said && b !== t) text(ctx, '✗', x + rad + 2, y - rad, { size: 10, w: 700, color: theme.c('critical') });
      }
    }
  }
  // a middle and a spread: the points, the middle, and the band of ±2 spreads (the true one dashed)
  function drawSpread(ctx, r, L) {
    if (L.data.task === 'branches') return drawBranches(ctx, r, L);
    const d = L.data, m = L.model, lossId = L.run ? L.run.cfg.loss.id : L.lossId || 'gauss', band = !!m && lossId === 'gauss';
    const pad = 34, q = { x: r.x + pad, y: r.y + 8, w: r.w - pad - 8, h: r.h - 30 }, lo = -1.6, hi = 1.6, X = (v) => q.x + ((v + 1) / 2) * q.w, Y = (v) => q.y + q.h - ((v - lo) / (hi - lo)) * q.h;
    ctx.fillStyle = theme.c('panel-2'); ctx.beginPath(); NA.rr(ctx, q.x, q.y, q.w, q.h, 8); ctx.fill();
    for (const t of [-1, 0, 1]) { const y = Math.round(Y(t)) + 0.5; ctx.strokeStyle = theme.c('line'); ctx.lineWidth = 1; ctx.beginPath(); ctx.moveTo(q.x, y); ctx.lineTo(q.x + q.w, y); ctx.stroke(); text(ctx, String(t).replace('-', NA.MINUS || '−'), q.x - 6, y, { size: 10, align: 'right', color: theme.c('muted'), mono: true }); }
    for (const t of [-1, -0.5, 0, 0.5, 1]) text(ctx, String(t).replace('-', NA.MINUS || '−'), X(t), q.y + q.h + 11, { size: 10, align: 'center', color: theme.c('muted'), mono: true });
    ctx.save(); ctx.beginPath(); ctx.rect(q.x, q.y, q.w, q.h); ctx.clip();
    const T = d.truth, line = (fn, col, w, dash) => { ctx.strokeStyle = col; ctx.lineWidth = w; ctx.setLineDash(dash || []); ctx.beginPath(); for (let i = 0; i <= 200; i++) { const x = -1 + i / 100, y = fn(x); i ? ctx.lineTo(X(x), Y(y)) : ctx.moveTo(X(x), Y(y)); } ctx.stroke(); ctx.setLineDash([]); };
    if (m) {
      const n = 200, xs = new Float32Array(n); for (let i = 0; i < n; i++) xs[i] = -1 + (2 * i) / (n - 1);
      const p = kept(m, `${L.vizVersion}:spread`, () => Float32Array.from(m.predict(xs, n).subarray(0, 2 * n)));
      if (band) {
        ctx.fillStyle = theme.c('s1', 0.18); ctx.beginPath();
        for (let i = 0; i < n; i++) { const sd = Math.exp(0.5 * clamp(p[2 * i + 1], -7, 7)); i ? ctx.lineTo(X(xs[i]), Y(p[2 * i] + 2 * sd)) : ctx.moveTo(X(xs[i]), Y(p[2 * i] + 2 * sd)); }
        for (let i = n - 1; i >= 0; i--) { const sd = Math.exp(0.5 * clamp(p[2 * i + 1], -7, 7)); ctx.lineTo(X(xs[i]), Y(p[2 * i] - 2 * sd)); }
        ctx.closePath(); ctx.fill();
      }
      ctx.strokeStyle = theme.c('ink'); ctx.lineWidth = 2.4; ctx.beginPath(); for (let i = 0; i < n; i++) { i ? ctx.lineTo(X(xs[i]), Y(p[2 * i])) : ctx.moveTo(X(xs[i]), Y(p[2 * i])); } ctx.stroke();
    }
    line((x) => T.f(x) + 2 * T.sd(x), theme.c('ink-2', 0.6), 1.2, [4, 4]); line((x) => T.f(x) - 2 * T.sd(x), theme.c('ink-2', 0.6), 1.2, [4, 4]);
    const dot = (x, y, hollow) => { ctx.beginPath(); ctx.arc(X(x), Y(y), 2.4, 0, Math.PI * 2); if (hollow) { ctx.strokeStyle = theme.c('s2'); ctx.lineWidth = 1.3; ctx.stroke(); } else { ctx.fillStyle = theme.c('s1', 0.75); ctx.fill(); } };
    for (let i = 0; i < d.nTr; i++) dot(d.Xtr[i], d.Ytr[2 * i], false);
    for (let i = 0; i < d.nVa; i++) dot(d.Xva[i], d.Yva[2 * i], true);
    if (d.task === 'gap') { ctx.fillStyle = theme.c('warning', 0.08); ctx.fillRect(X(-0.3), q.y, X(0.3) - X(-0.3), q.h); }
    ctx.restore();
    if (band) { const k = G.inBand(m, d, d.nVa); text(ctx, `${Math.round(100 * k)}% of the validation points inside its band (95% if its spread is right)`, q.x + 8, q.y + 12, { size: 10.5, w: 600, color: theme.c('muted') }); }
  }
  // more than one right answer: the mixture's density for y at every x (darker where likelier), each curve's middle
  // where it carries weight, the points and the true curve
  function drawBranches(ctx, r, L) {
    const d = L.data, m = L.model, lossId = L.run ? L.run.cfg.loss.id : L.lossId || 'mdn', mix = !!m && lossId === 'mdn', K = d.output.k;
    const pad = 34, q = { x: r.x + pad, y: r.y + 8, w: r.w - pad - 8, h: r.h - 30 }, lo = -1.4, hi = 1.4, X = (v) => q.x + ((v + 1) / 2) * q.w, Y = (v) => q.y + q.h - ((v - lo) / (hi - lo)) * q.h;
    ctx.fillStyle = theme.c('panel-2'); ctx.beginPath(); NA.rr(ctx, q.x, q.y, q.w, q.h, 8); ctx.fill();
    for (const t of [-1, 0, 1]) { const y = Math.round(Y(t)) + 0.5; ctx.strokeStyle = theme.c('line'); ctx.lineWidth = 1; ctx.beginPath(); ctx.moveTo(q.x, y); ctx.lineTo(q.x + q.w, y); ctx.stroke(); text(ctx, String(t).replace('-', NA.MINUS || '−'), q.x - 6, y, { size: 10, align: 'right', color: theme.c('muted'), mono: true }); }
    for (const t of [-1, -0.5, 0, 0.5, 1]) text(ctx, String(t).replace('-', NA.MINUS || '−'), X(t), q.y + q.h + 11, { size: 10, align: 'center', color: theme.c('muted'), mono: true });
    ctx.save(); ctx.beginPath(); ctx.rect(q.x, q.y, q.w, q.h); ctx.clip();
    const n = 120, xs = new Float32Array(n); for (let i = 0; i < n; i++) xs[i] = -1 + (2 * i) / (n - 1);
    const o = m ? kept(m, `${L.vizVersion}:branches`, () => Float32Array.from(m.predict(xs, n).subarray(0, n * K))) : null;
    if (mix) {
      const gy = 70, img = kept(m, `${L.vizVersion}:density:${theme.dark ? 1 : 0}`, () => {
        const c = document.createElement('canvas'); c.width = n; c.height = gy;
        const g = c.getContext('2d'), id = g.createImageData(n, gy), bg = rgb('panel-2'), hi2 = rgb('s1');
        for (let i = 0; i < n; i++) {
          const cs = G.mixture(o, i * K, d.comps), col = [];
          for (let j = 0; j < gy; j++) { const y = hi - ((j + 0.5) / gy) * (hi - lo); let p = 0; for (const k of cs) p += (k.w / k.sd) * Math.exp(-0.5 * ((y - k.mu) / k.sd) ** 2); col.push(p); }
          const top = Math.max(...col) || 1;
          for (let j = 0; j < gy; j++) { const a = 0.75 * Math.sqrt(col[j] / top); for (let ch = 0; ch < 3; ch++) id.data[(j * n + i) * 4 + ch] = bg[ch] + (hi2[ch] - bg[ch]) * a; id.data[(j * n + i) * 4 + 3] = 255; }
        }
        g.putImageData(id, 0, 0);
        return c;
      });
      ctx.imageSmoothingEnabled = true; ctx.drawImage(img, X(-1), Y(hi), X(1) - X(-1), Y(lo) - Y(hi));
      for (let i = 0; i < n; i++) for (const k of G.mixture(o, i * K, d.comps)) if (k.w >= 0.05) { ctx.fillStyle = theme.c('ink', Math.min(1, 0.25 + k.w)); ctx.beginPath(); ctx.arc(X(xs[i]), Y(k.mu), 1.6, 0, Math.PI * 2); ctx.fill(); }
    } else if (o) { ctx.strokeStyle = theme.c('ink'); ctx.lineWidth = 2.4; ctx.beginPath(); for (let i = 0; i < n; i++) (i ? ctx.lineTo(X(xs[i]), Y(o[i * K])) : ctx.moveTo(X(xs[i]), Y(o[i * K]))); ctx.stroke(); }
    ctx.strokeStyle = theme.c('ink-2', 0.6); ctx.lineWidth = 1.2; ctx.setLineDash([4, 4]); ctx.beginPath();
    for (let i = 0; i <= 200; i++) { const [x, y] = d.curve(i / 200); i ? ctx.lineTo(X(x), Y(y)) : ctx.moveTo(X(x), Y(y)); }
    ctx.stroke(); ctx.setLineDash([]);
    const dot = (x, y, hollow) => { ctx.beginPath(); ctx.arc(X(x), Y(y), 2.3, 0, Math.PI * 2); if (hollow) { ctx.strokeStyle = theme.c('s2'); ctx.lineWidth = 1.3; ctx.stroke(); } else { ctx.fillStyle = theme.c('s1', 0.6); ctx.fill(); } };
    for (let i = 0; i < d.nTr; i++) dot(d.Xtr[i], d.Ytr[i * K], false);
    for (let i = 0; i < d.nVa; i++) dot(d.Xva[i], d.Yva[i * K], true);
    ctx.restore();
  }
  const DRAW = { recon: drawRecon, gan: drawGan, contrast: drawContrast, segment: drawSegment, graph: drawGraph, spread: drawSpread };
  G.draw = (ctx, r, L) => DRAW[L.data.family](ctx, r, L);
  // the problem picker's small picture of the data
  G.preview = function (ctx, W, H, d) {
    const r = { x: 2, y: 2, w: W - 4, h: H - 4 }, f = d.family;
    if (f === 'recon' || f === 'contrast') { const n = Math.min(d.nVa, 10), s = Math.min(r.h - 4, r.w / n - 4); for (let i = 0; i < n; i++) pic(ctx, d.Xva, i * PX, r.x + i * (s + 4), r.y + (r.h - s) / 2, s); return; }
    if (f === 'segment') { const P = SEG * SEG, n = Math.min(d.nVa, 4), s = Math.min(r.h - 4, r.w / (2 * n) - 6); for (let i = 0; i < n; i++) { pic(ctx, d.Xva, i * P, r.x + i * (2 * s + 12), r.y + (r.h - s) / 2, s, { G: SEG }); pic(ctx, d.Yva, i * P, r.x + i * (2 * s + 12) + s + 3, r.y + (r.h - s) / 2, s, { G: SEG, mask: true }); } return; }
    if (f === 'gan' && d.pixels) { const n = Math.min(d.nVa, 10), s = Math.min(r.h - 4, r.w / n - 4); for (let i = 0; i < n; i++) pic(ctx, d.Yva, i * PX, r.x + i * (s + 4), r.y + (r.h - s) / 2, s); return; }
    if (f === 'gan') { const q = square(r), S = 1.25; ctx.fillStyle = theme.c('ink-2', 0.6); for (let i = 0; i < Math.min(d.nTr, 400); i++) { ctx.beginPath(); ctx.arc(q.x + ((d.Ytr[2 * i] + S) / (2 * S)) * q.s, q.y + ((S - d.Ytr[2 * i + 1]) / (2 * S)) * q.s, 1.4, 0, Math.PI * 2); ctx.fill(); } return; }
    if (f === 'graph') { drawGraph(ctx, { x: r.x, y: r.y - 16, w: r.w, h: r.h + 16 }, { data: Object.assign({}, d, { nVa: 2 }), model: null }); return; }
    drawSpread(ctx, { x: r.x, y: r.y, w: r.w, h: r.h + 18 }, { data: d, model: null });
  };
  // the key under the picture: el(...) items
  G.legend = function (d, live, item, el, L) {
    const f = d.family, gen = L && L.gen;
    if (f === 'recon') return [el('span', { text: live ? 'R² over every pixel of the pictures it rebuilt' : 'pictures to squeeze through the middle and rebuild' })];
    if (f === 'gan') return d.pixels ? [el('span', { text: gen && gen.P !== null ? `a digit reader reads ${Math.round(100 * gen.P)}% of its drawings (precision), ${Math.round(100 * gen.R)}% of the 10 digits among them (recall)` : 'real digits; press Train and the generator draws its own' })]
      : [item('bd-lg-dot', 'ink-2', 'real points'), live ? item('bd-lg-dot', 's2', 'the generator’s, with where the discriminator pushes them') : null, live ? item('bd-lg-box', 's1', 'where the discriminator thinks points real') : null,
        gen && gen.P !== null ? el('span', { text: `precision ${Math.round(100 * gen.P)}% · recall ${Math.round(100 * gen.R)}%` }) : null];
    if (f === 'contrast') return [el('span', { text: gen && gen.raw !== undefined ? `the same vote on the raw pixels: ${Math.round(100 * gen.raw)}%${gen.pairAcc !== null ? ` · its two views closest to each other: ${Math.round(100 * gen.pairAcc)}% of the time` : ''}` : 'pictures, and the two random views each is shown as' })];
    if (f === 'segment') return [item('bd-lg-box', 's1', 'circle'), item('bd-lg-box', 's2', 'square'), item('bd-lg-box', 's4', 'triangle')];
    if (f === 'graph') return [item('bd-lg-dot', 's1', 'group A'), item('bd-lg-dot', 's2', 'group B'), item('bd-lg-dot', 's4', 'group C'), el('span', { text: 'scored on the nodes that do not say' })];
    if (d.task === 'branches') return [item('bd-lg-dot', 's1', 'training'), item('bd-lg-ring', 's2', 'validation'), live && L && L.lossId === 'mdn' ? item('bd-lg-box', 's1', 'how likely each y is, at each x (its mixture)') : live ? item('bd-lg-line', null, 'its one answer') : null, live && L && L.lossId === 'mdn' ? item('bd-lg-dot', 'ink', 'the middles of its curves, darker the more weight') : null, item('bd-lg-dash', null, 'the true curve')];
    return [item('bd-lg-dot', 's1', 'training'), item('bd-lg-ring', 's2', 'validation'), live ? item('bd-lg-line', null, 'its middle') : null, live && L && L.lossId === 'gauss' ? item('bd-lg-box', 's1', '± 2 of its spreads') : null, item('bd-lg-dash', null, 'the true ± 2 spreads')];
  };
  // what the Output tab says instead of a choice of loss (the GAN's and contrastive learning's are their own)
  G.lossNote = (d) => (d.family === 'gan' ? 'No loss to choose: the generator learns from its rival. The discriminator learns to answer 1 for real examples and 0 for the generator’s (binary cross-entropy); the generator learns to make it answer 1, −log D(G(z)), its gradient coming back through the discriminator.'
    : d.family === 'contrast' ? `No loss to choose: NT-Xent (SimCLR’s). Each code should be nearer its partner, the other view of the same picture, than any other code in the batch: a cross-entropy over the batch whose right answer is the partner, with similarities divided by a temperature τ of ${d.tau}.` : null);
  // the problem's own settings, in the Data section
  G.controls = function (p, s, set, UI, el, fmt) {
    const f = p.family, out = [], n = (min, max, step) => UI.slider({ label: 'Examples', min, max, step, value: s.n, format: String, onChange: (v) => set((q) => { q.n = v; }) });
    const pics = () => UI.field('Pictures', UI.seg([{ value: 'digits', label: 'Digits' }, { value: 'shapes', label: 'Shapes' }], s.pictures, (v) => set((q) => { q.pictures = v; }), 'Pictures'));
    if (f === 'recon') {
      out.push(pics(), n(500, 4000, 100));
      if (p.task === 'denoise') out.push(UI.slider({ label: 'Speckles', min: 0, max: 0.6, step: 0.05, value: s.noise, format: (v) => fmt(v, 2), onChange: (v) => set((q) => { q.noise = v; }) }));
      out.push(UI.note(p.task === 'vae' ? 'Sample, in the middle of the VAE template, sets how hard the codes are pulled toward one bell curve (β, in its settings): harder gives blurrier pictures but a map with no holes, so any code drawn makes a picture.' : 'The narrower the middle, the more the model has to keep only what matters. Watch the map of the codes: similar pictures end up near each other, though no label was ever given.'));
    } else if (f === 'gan') {
      out.push(UI.field('Random numbers in', UI.seg([1, 2, 4, 8].map((v) => ({ value: v, label: String(v) })), s.z, (v) => set((q) => { q.z = +v; }), 'How many random numbers the generator is given')),
        UI.field('The discriminator', UI.seg([{ value: 'small', label: '2 × 32' }, { value: 'big', label: '2 × 64' }], s.disc, (v) => set((q) => { q.disc = v; }), 'The size of its rival')),
        n(p.task === 'digits' ? 500 : 200, 4000, 100),
        UI.note('The generator never sees a real example: all it knows comes back through the discriminator. Too strong a discriminator and its verdicts are flat everywhere; too weak and it cannot tell the generator what is wrong.'));
    } else if (f === 'contrast') {
      out.push(pics(), n(500, 4000, 100),
        UI.field('Code size', UI.seg([8, 16, 32].map((v) => ({ value: v, label: String(v) })), s.dim, (v) => set((q) => { q.dim = +v; }), 'Numbers in each code')),
        UI.field('Temperature τ', UI.seg([0.1, 0.2, 0.5, 1].map((v) => ({ value: v, label: String(v) })), s.tau, (v) => set((q) => { q.tau = +v; }), 'Temperature')),
        UI.field('Views', UI.seg([{ value: 'gentle', label: 'Gentle' }, { value: 'strong', label: 'Strong' }], s.views, (v) => set((q) => { q.views = v; }), 'How different the two views are')),
        UI.note('What it learns to ignore is what the views change: shifts, thickness, faintness, speckles. Stronger views teach it more about what stays the same.'));
    } else if (f === 'segment') {
      out.push(UI.field('Shapes in each', UI.seg([1, 2, 3].map((v) => ({ value: v, label: String(v) })), s.shapes, (v) => set((q) => { q.shapes = +v; }), 'Shapes in each picture')), n(300, 3000, 100),
        UI.slider({ label: 'Noise', min: 0, max: 0.3, step: 0.02, value: s.noise, format: (v) => fmt(v, 2), onChange: (v) => set((q) => { q.noise = v; }) }),
        UI.note('Scored by the overlap of its shapes with the true ones (intersection over union), so the background, most of every picture, does not count for free.'));
    } else if (f === 'graph') {
      out.push(n(200, 3000, 100),
        UI.slider({ label: 'Links across groups', min: 0, max: 0.3, step: 0.02, value: s.across, format: (v) => `${Math.round(100 * v)}%`, onChange: (v) => set((q) => { q.across = v; }) }),
        UI.field('Who says their group', UI.seg([{ value: 1, label: '1 in each' }, { value: 2, label: '2 in each' }], s.hints, (v) => set((q) => { q.hints = +v; }), 'How many in each group say which it is')),
        UI.note('Within a group, two people are linked 6 times in 10. The more links across groups, the more a node hears from the wrong group.'));
    } else if (p.task === 'branches') {
      out.push(UI.field('Bell curves', UI.seg([1, 2, 3, 5].map((v) => ({ value: v, label: String(v) })), s.components, (v) => set((q) => { q.components = +v; }), 'How many bell curves in its mixture')), n(100, 2000, 50),
        UI.note(`Its outputs: ${3 * s.components} numbers, a weight, a middle and a log spread for each of ${s.components} bell curve${s.components > 1 ? 's' : ''}, scored by the likelihood of the answer under their mixture. With 1 it can only spread one curve across the branches; with 3, one for each. Squared error gives the average of the branches, where there are no points at all.`));
    } else {
      out.push(n(50, 1000, 10));
      if (p.task === 'noisy') out.push(UI.slider({ label: 'Noise at the right', min: 0, max: 0.5, step: 0.05, value: s.noise, format: (v) => fmt(v, 2), onChange: (v) => set((q) => { q.noise = v; }) }));
      out.push(UI.note('Two outputs: the middle, and the log of the spread squared (a log keeps the spread above 0 whatever the network gives). With the squared error loss it learns only the middle.'));
    }
    return out;
  };

  NA.bgen = G;
})(typeof globalThis !== 'undefined' ? (globalThis.NA = globalThis.NA || {}) : (window.NA = window.NA || {}));
