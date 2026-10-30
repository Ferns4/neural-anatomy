/* builder-tasks.js — the problems the model builder can learn, and the data they make for the engine (js/builder-
   engine.js): points in a plane to classify (preset shapes, or ones you draw), a curve to fit (preset, or a formula
   you type), handwritten digits, sequences (reverse, sort or copy digits, the tiny language, or sentences you
   write), and worlds to act in, learning from rewards (js/rl-envs.js, trained by js/builder-rl.js). Also the
   starting templates, a small safe formula parser and the helpers that decode a model's output. */
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
    // no examples with answers: the model acts in a world and learns from the rewards (the method sets the loss)
    control: { label: 'Learn from rewards', losses: ['huber'], metric: 'score', template: 'mlp', icon: '⚑' },
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
    { id: 'wiggles', family: 'curve', task: 'wiggles', label: 'Fast wiggles', hint: 'A wave that goes up and down 8 times between −1 and 1. A plain MLP draws slow, smooth curves first and takes a long time over fast ones; Fourier features hand it fast waves to build from.' },
    { id: 'formula', family: 'curve', custom: true, label: 'Your own function', hint: 'Type any formula in x, for example sin(5x) + x^2.' },
    { id: 'hill', family: 'surface', task: 'hill', label: 'Two hills', hint: 'A hill and a hollow on a plane: a smooth height to learn from (x, y).' },
    { id: 'ripples', family: 'surface', task: 'ripples', label: 'Ripples', hint: 'Rings spreading from the middle, like a pebble dropped in a pond.' },
    { id: 'saddle', family: 'surface', task: 'saddle', label: 'Saddle', hint: 'Up one way, down the other: x² − y².' },
    { id: 'checker', family: 'surface', task: 'checker', label: 'Checkerboard', hint: 'A smooth checkerboard of highs and lows: many bends in both directions.' },
    { id: 'picture', family: 'surface', task: 'picture', label: 'A picture', hint: 'A smiling face as a height to learn from (x, y): the face bright, its eyes and mouth dark. Sharp edges are fine detail: a plain MLP blurs them, Fourier features draw them.' },
    { id: 'digits', family: 'digits', label: 'Handwritten digits', hint: '14 × 14 pictures of the digits 0 to 9, drawn with random strokes.' },
    { id: 'shapes', family: 'digits', task: 'shapes', label: 'Shapes', hint: 'Circles, squares, triangles, crosses and lines, drawn anywhere at any size and angle: what convolutions are for.' },
    { id: 'dots', family: 'digits', task: 'dots', label: 'Count the dots', hint: 'One to five dots of different sizes, anywhere: how many are there? One detector used everywhere and its finds added up counts them: convolutions, then a global average pool. Where the dots are does not matter.' },
    { id: 'stripes', family: 'digits', task: 'stripes', label: 'Which way the stripes run', hint: 'Faint stripes in noise, at any spacing: across, up and down, rising or falling. A texture, what the picture is made of rather than where anything is: small convolution filters see it anywhere, a dense layer has to learn it at every place.' },
    { id: 'colours', family: 'digits', task: 'colours', label: 'Colour and shape', hint: 'A red, green or blue circle or cross, as three channels of colour. The three colours are equally bright, so in grey, one channel, red, green and blue look alike: why a picture has channels.' },
    { id: 'inside', family: 'digits', task: 'inside', label: 'Inside or outside', hint: 'A ring and a dot, anywhere and at any size: is the dot inside the ring? The answer is not in either one but in how they lie, so the model needs a view of the whole picture.' },
    { id: 'waves', family: 'series', task: 'waves', label: 'Two waves', hint: 'Two sine waves added up: predict the next value from the last few.' },
    { id: 'seasonal', family: 'series', task: 'seasonal', label: 'Seasons and trend', hint: 'A slow rise, a repeating season and noise, like a shop’s sales.' },
    { id: 'square', family: 'series', task: 'square', label: 'Square wave', hint: 'Flat stretches and sudden jumps: the model has to know when the next jump comes.' },
    { id: 'chaos', family: 'series', task: 'chaos', label: 'Chaos', hint: 'The logistic map, x → 3.9·x(1 − x): completely determined, yet it never repeats. The next value is a simple curve of the last one.' },
    { id: 'walk', family: 'series', task: 'walk', label: 'Random walk', hint: 'Each step a random nudge up or down. Nothing in the past says where it goes next, so the best forecast is where it is now: no model beats repeating the last value. A lesson in what cannot be learned.' },
    { id: 'heartbeat', family: 'series', task: 'heartbeat', label: 'Heartbeat', hint: 'Beats of a sharp spike and a broad bump, with a flat line between them. Only a window that reaches back to the last beat can tell when the next one comes.' },
    { id: 'sentiment', family: 'seqclass', task: 'sentiment', label: 'Tiny reviews', hint: 'Short reviews with good and bad words; “not” flips a word and after “but” the second part decides. Word order matters.' },
    { id: 'majority', family: 'seqclass', task: 'majority', label: 'Majority vote', hint: 'More 1s or more 0s? Counting is enough: order does not matter.' },
    { id: 'pattern', family: 'seqclass', task: 'pattern', label: 'Find a pattern', hint: 'Is there a 3 followed straight away by a 7? Only the order of neighbours tells.' },
    { id: 'parity', family: 'seqclass', task: 'parity', label: 'Parity', hint: 'An odd or even number of 1s? Flipping any one bit changes the answer: famously hard to learn.' },
    { id: 'brackets', family: 'seqclass', task: 'brackets', label: 'Balanced brackets', hint: 'Do the brackets close properly, as in (()())? Counting them is not enough: the count of open ones must never drop below zero on the way. A recurrent layer can keep a running count; a bag of words cannot.' },
    { id: 'palindrome', family: 'seqclass', task: 'palindrome', label: 'Palindromes', hint: 'Does the sequence read the same backwards, as 4 7 2 7 4 does? The near misses differ in one digit, so every position must be compared with its mirror image: attention can pair them up.' },
    { id: 'reverse', family: 'sequence', task: 'reverse', label: 'Reverse a sequence', hint: '3 1 4 → 4 1 3. Each answer digit must look back at a different input position.' },
    { id: 'sort', family: 'sequence', task: 'sort', label: 'Sort digits', hint: '3 1 4 → 1 3 4. Needs to compare all the inputs.' },
    { id: 'copy', family: 'sequence', task: 'copy', label: 'Copy a sequence', hint: '3 1 4 → 3 1 4. The easiest look-back task.' },
    { id: 'add', family: 'sequence', task: 'add', label: 'Add two numbers', hint: '123 + 456 → 579, written digit by digit, the largest first. Each digit of the answer needs the two digits above it and the carry from the digits to their right.' },
    { id: 'language', family: 'sequence', task: 'language', label: 'Tiny language', hint: 'Predict the next word of short stories, like the Tiny Transformer model.' },
    { id: 'sentences', family: 'sequence', task: 'sentences', custom: true, label: 'Your own sentences', hint: 'Type some text; the model learns to predict each next word.' },
    { id: 'csv', family: 'table', custom: true, label: 'Your table (CSV)', hint: 'A table of numbers from a CSV file, or a sample: the model learns to predict one column from the others.' },
    // the worlds to learn from rewards, by level: 1 beginner, 2 intermediate, 3 advanced, 4 expert
    { id: 'levers', family: 'control', task: 'levers', level: 1, label: 'Levers (a bandit)', hint: 'A light shows a colour, and the lever of that colour usually pays: one pull per episode. No examples with answers, only rewards: the simplest problem where the best action depends on what you see.' },
    { id: 'catch', family: 'control', task: 'catch', level: 1, label: 'Catch', hint: 'Move a paddle to catch a falling, drifting ball. It sees 4 numbers, or pixels: two pictures of the board, now and a step ago.' },
    { id: 'rlmaze', family: 'control', task: 'maze', level: 1, label: 'Maze', hint: 'Find the goal and keep out of the pits. It can see the square it is on, the squares around it, or a picture of the whole maze, and train on one maze or a new random one every episode.' },
    { id: 'thermo', family: 'control', task: 'thermo', level: 1, label: 'Keep a room warm', hint: 'Switch the heating on and off to keep a room within 1 °C of the temperature wanted, all day. The radiator is slow to warm up and slow to cool down, so it has to switch before the room gets there. It sees the room, outside and radiator temperatures, the target and the time.' },
    { id: 'hurdles', family: 'control', task: 'hurdles', level: 1, label: 'Hurdles', hint: 'Run a 150 m course and jump the hurdles; on the mixed course, duck under the bars too. It sees how far the next two obstacles are and what they are, its height, and whether it is ducking or on the ground.' },
    { id: 'pole', family: 'control', task: 'pole', level: 2, label: 'Balance a pole', hint: 'Push a cart left or right to keep a pole upright, the classic control problem. It sees 4 numbers, or only the positions and angles of the last 4 steps.' },
    { id: 'memory', family: 'control', task: 'memory', level: 2, label: 'Remember the lamp', hint: 'A lamp at the start of a corridor says which way to turn at its end, then goes dark. Seeing only the present, it can do no better than guess at the junction, half of them right: give it the last 16 readings and it can look back.' },
    { id: 'mountain', family: 'control', task: 'mountain', level: 2, label: 'Mountain cart', hint: 'A cart in a valley, its engine too weak to drive up the slope: it has to rock back and forth to build up speed. Every step costs 1 until it reaches the flag, so random driving teaches almost nothing. Evolution finds the way.' },
    { id: 'swingup', family: 'control', task: 'swingup', level: 2, label: 'Swing up a pendulum', hint: 'A motor too weak to lift a pendulum straight up: it has to swing it higher with every swing, then catch it upside down and balance it there. It sees the angle and how fast it turns.' },
    { id: 'arm', family: 'control', task: 'arm', level: 2, label: 'Robot arm', hint: 'Turn the shoulder and the elbow to bring the hand onto a target and stop there; a new target every episode. It sees its joint angles and speeds, and where the target is.' },
    { id: 'traffic', family: 'control', task: 'traffic', level: 2, label: 'Traffic lights', hint: 'Keep the lights or switch them, so the cars at a crossroads get through quickly. Every switch costs 4 steps of amber, and the rush hour moves from one road to the other.' },
    { id: 'track', family: 'control', task: 'track', level: 3, label: 'Race track', hint: 'Drive a car 4 laps without touching the walls: steer, and brake for the tight bends. It sees 7 distances to the walls and its speed, or a camera picture from above. Evolution learns it best: watch a whole generation drive at once.' },
    { id: 'arena', family: 'control', task: 'arena', level: 3, label: 'Drone arena (first person)', hint: 'A first-person arena: turn, walk and fire at the glowing drones before they reach you. It sees what each column of its view holds, a radar and its lives; or the picture itself, as agents that learned first-person games did.' },
    { id: 'walker', family: 'control', task: 'walker', level: 2, label: 'Walker', hint: 'A creature of bones, springs and two muscles: find the rhythm of muscle moves that walks it 10 m without toppling over. It has a clock to keep time by.' },
    { id: 'trap', family: 'control', task: 'trap', level: 3, label: 'Deceptive maze', hint: 'The goal is behind a wall: heading straight for it ends in a dead end. Paying for getting closer is a trap here; novelty search, which pays for ending up somewhere new, gets out.' },
    { id: 'lander', family: 'control', task: 'lander', level: 4, label: 'Land a rocket', hint: 'Fire the main engine and the side thrusters to touch down softly on the pad before the fuel runs out. It sees where it is, how fast it moves, its tilt, spin and fuel.' },
    { id: 'chase', family: 'control', task: 'chase', level: 4, label: 'Escape the chaser', hint: 'Run from a chaser for 400 steps in a walled arena. It is a little slower and heads straight for you, but running straight away still ends in a corner: it has to learn to circle round. A chaser that aims where you are going is much harder to shake off.' },
    { id: 'herd', family: 'control', task: 'herd', level: 4, label: 'Herd the sheep', hint: 'A sheepdog: the sheep flock together and flee from it. Get them through the gate into the pen, the stragglers too.' },
  ];
  BT.byId = Object.fromEntries(BT.PROBLEMS.map((p) => [p.id, p]));
  BT.LEVELS = ['', 'beginner', 'intermediate', 'advanced', 'expert'];
  // the problems as the pickers list them: by kind, and the worlds to learn from rewards by level too
  BT.problemGroups = () => Object.keys(BT.FAMILIES).flatMap((f) => {
    const ps = BT.PROBLEMS.filter((p) => p.family === f);
    if (f !== 'control') return [{ label: BT.FAMILIES[f].label, problems: ps }];
    return [1, 2, 3, 4].map((l) => ({ label: `${BT.FAMILIES[f].label}: ${BT.LEVELS[l]}`, problems: ps.filter((p) => p.level === l) }));
  });
  // the kinds of data whose answer is a number, scored by R² (a table that predicts a number makes curve-like data)
  BT.isValue = (family) => family === 'curve' || family === 'surface' || family === 'series';
  BT.CLASS_COLOURS = ['s1', 's2', 's4', 's7'];

  // problem settings with their defaults (the inspector shows the ones that apply)
  BT.defaults = (p) => {
    const f = p.family;
    if (f === 'plane') return { noise: p.id === 'xor' || p.id === 'blobs' ? 0.1 : 0.15, n: 600 };
    if (f === 'curve') return p.task === 'wiggles' ? { noise: 0.05, n: 400, waves: 8 } : { noise: 0.05, n: 200, formula: 'sin(5x) + x^2' };
    if (f === 'digits') return Object.assign({ n: p.task === 'shapes' || p.task === 'stripes' || p.task === 'colours' ? 1500 : 2000 }, { dots: { most: 5, sizes: 'mixed' }, stripes: { dirs: 4, noise: 0.25 }, colours: { colour: 'colour' } }[p.task]);
    if (f === 'surface') return p.task === 'picture' ? { noise: 0, n: 2000, edges: 'sharp' } : { noise: 0.02, n: 800 };
    if (f === 'series') return p.task === 'heartbeat' ? { window: 32, n: 1200, noise: 0.03, rhythm: 'steady' } : { window: 16, n: 1200, noise: p.task === 'chaos' || p.task === 'walk' ? 0 : 0.04 };
    if (f === 'seqclass') return Object.assign({ n: p.task === 'sentiment' ? 2500 : 2000, embed: 16 }, { brackets: { len: 16 }, palindrome: { len: 10 } }[p.task]);
    if (f === 'table') return { target: -1, kind: 'auto' };
    if (f === 'control') return Object.assign(NA.brl.defaults(p.task), WORLD_DEFAULTS[p.task]);
    if (p.task === 'language') return { n: 1500, embed: 32 };
    if (p.task === 'sentences') return { window: 8, embed: 32, text: SAMPLE_TEXT };
    if (p.task === 'add') return { digits: 3, n: 3000, embed: 32 };
    return { len: 4, n: 3000, embed: 32 };
  };
  const SAMPLE_TEXT = 'the sun is hot . the snow is cold . the fire is hot . the ice is cold .\nthe cat sat on the mat . the dog sat on the log .\nred and blue make purple . blue and yellow make green . red and yellow make orange .';

  // training settings: the defaults, and what each kind of data starts with when you switch to it
  BT.DEFAULT_TRAIN = { opt: 'adam', lr: 0.01, batch: 32, wd: 0, sched: 'const', clip: 0, init: 'auto', seed: 1, steps: 3000 };
  BT.trainFor = (p) => {
    if (p.family === 'control') return { lr: 0.003, batch: 32, steps: NA.brl.stepsFor(p.task, (WORLD_DEFAULTS[p.task] || {}).method || 'dqn'), clip: 1 };
    if (p.family === 'sequence') return { lr: 0.003, batch: p.task === 'language' ? 16 : 32, steps: p.task === 'language' ? 3000 : p.task === 'add' ? 2000 : p.task === 'sentences' ? 1500 : 1000, clip: 1 };
    if (p.family === 'digits') return { lr: 0.003, batch: 32, steps: 1500, clip: 0 };
    if (p.family === 'table') return { lr: 0.01, batch: 32, steps: 2000, clip: 0 };
    if (p.family === 'series') return { lr: 0.003, batch: 32, steps: 2000, clip: 1 };
    if (p.family === 'seqclass') return { lr: 0.003, batch: 32, steps: 1500, clip: 1 };
    return { lr: 0.01, batch: 32, steps: 3000, clip: 0 };
  };

  // ------------------------------------------------------------------ templates
  const D = (units, act = 'relu') => ({ type: 'dense', units, act });
  const PB = (merge, ...branches) => ({ type: 'parallel', merge, branches }); // parallel branches
  BT.TEMPLATES = [
    { id: 'empty', label: 'Empty', hint: 'Just the embedding and the output: build the rest yourself.', families: ['sequence'], blocks: () => [] },
    { id: 'linear', label: 'Linear', hint: 'No hidden layer: one weighted sum per output, so only straight boundaries. Build up from here.', families: ['plane', 'curve', 'surface', 'digits', 'table', 'series', 'control'], blocks: () => [] },
    { id: 'mlp', label: 'Small MLP', hint: 'Two hidden layers of 16 ReLU neurons.', families: ['plane', 'curve', 'surface', 'digits', 'table', 'series', 'control'], blocks: () => [D(16), D(16)] },
    { id: 'deep', label: 'Deep MLP', hint: 'Five hidden layers of 24 neurons.', families: ['plane', 'curve', 'surface', 'digits', 'table', 'control'], blocks: () => [D(24), D(24), D(24), D(24), D(24)] },
    { id: 'wide', label: 'One wide layer', hint: 'A single hidden layer of 64 neurons.', families: ['plane', 'curve', 'surface', 'digits', 'table', 'series', 'control'], blocks: () => [D(64)] },
    { id: 'rlcnn', label: 'Small CNN', hint: 'A convolution finds the ball, the walls and you wherever they are, a pooling halves the picture, then a dense layer decides.', families: ['control'], kinds: ['image'], blocks: () => [{ type: 'conv', filters: 8, k: 3, stride: 1, act: 'relu' }, { type: 'pool', size: 2 }, { type: 'flatten' }, D(32)] },
    { id: 'rlgru', label: 'GRU', hint: 'A GRU reads the readings one step at a time, oldest first: its memory can work out how fast things move.', families: ['control'], kinds: ['series'], blocks: () => [{ type: 'rnn', cell: 'gru', units: 16 }, { type: 'seqpool', mode: 'last' }, D(16)] },
    { id: 'cnn', label: 'Small CNN', hint: 'Two convolution + pooling stages, then a dense layer.', families: ['digits'], blocks: () => [{ type: 'conv', filters: 8, k: 3, stride: 1, act: 'relu' }, { type: 'pool', size: 2 }, { type: 'conv', filters: 16, k: 3, stride: 1, act: 'relu' }, { type: 'pool', size: 2 }, { type: 'flatten' }, D(32)] },
    { id: 'cnngap', label: 'CNN, averaged', hint: 'Convolutions, then a global average pool: each filter’s matches averaged over the whole picture. It finds things wherever they are and adds them up, so it counts well; where they were, it forgets.', families: ['digits'], blocks: () => [{ type: 'conv', filters: 8, k: 3, stride: 1, act: 'relu' }, { type: 'pool', size: 2 }, { type: 'conv', filters: 16, k: 3, stride: 1, act: 'relu' }, { type: 'gap' }, D(16)] },
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
    // parallel branches: blocks side by side on the same input, joined or added
    { id: 'branches', label: 'Two branches', hint: 'Two dense branches side by side, ReLU and tanh, joined, then one more layer: two views of the same input at once.', families: ['plane', 'curve', 'surface', 'table'], blocks: () => [PB('concat', [D(16)], [D(16, 'tanh')]), D(16)] },
    { id: 'resnet', label: 'Residual branches', hint: 'Each stage adds its own input back (a branch with nothing in it): the trick that lets very deep networks train.', families: ['plane', 'curve', 'surface'], blocks: () => [D(16, 'tanh'), PB('add', [D(16, 'tanh'), D(16, 'tanh')], []), PB('add', [D(16, 'tanh'), D(16, 'tanh')], [])] },
    { id: 'widedeep', label: 'Wide & deep', hint: 'The raw inputs (wide) next to two learned layers (deep), joined for the output: a classic for tables and recommendations.', families: ['table'], blocks: () => [PB('concat', [], [D(32), D(16)])] },
    { id: 'inception', label: 'Inception-style', hint: '1×1, 3×3 and 5×5 filters side by side on the same feature maps, joined: small and large patterns at once.', families: ['digits'], blocks: () => [{ type: 'conv', filters: 8, k: 3, stride: 1, act: 'relu' }, { type: 'pool', size: 2 }, PB('concat', [{ type: 'conv', filters: 8, k: 1, stride: 1, act: 'relu' }], [{ type: 'conv', filters: 8, k: 3, stride: 1, act: 'relu' }], [{ type: 'conv', filters: 4, k: 5, stride: 1, act: 'relu' }]), { type: 'pool', size: 2 }, { type: 'flatten' }, D(32)] },
    { id: 'convgru', label: 'Conv + GRU', hint: 'A causal 1-D convolution and a GRU read the window side by side, joined; then the last step answers.', families: ['series'], blocks: () => [PB('concat', [{ type: 'conv1d', filters: 16, k: 3, dilation: 1, causal: true, act: 'relu' }], [{ type: 'rnn', cell: 'gru', units: 16 }]), { type: 'seqpool', mode: 'last' }] },
    { id: 'convgruclass', label: 'Conv + GRU', hint: 'A 1-D convolution and a GRU read the sentence side by side, joined; the strongest match anywhere decides.', families: ['seqclass'], blocks: () => [PB('concat', [{ type: 'conv1d', filters: 16, k: 3, dilation: 1, causal: false, act: 'relu' }], [{ type: 'rnn', cell: 'gru', units: 16 }]), { type: 'seqpool', mode: 'max' }] },
    { id: 'c-linear', label: 'Linear / logistic regression', hint: 'The classic: one weight per input. Least squares for numbers (solved in one step), logistic regression for classes.', families: ['plane', 'curve', 'surface', 'digits', 'table', 'series'], blocks: () => [{ type: 'linear', lambda: 0 }] },
    { id: 'c-knn', label: 'k-nearest neighbours', hint: 'No training at all: each answer is a vote of the k most similar training examples.', families: ['plane', 'curve', 'surface', 'digits', 'table', 'series'], blocks: () => [{ type: 'knn', k: 5, weights: 'uniform' }] },
    { id: 'c-tree', label: 'Decision tree', hint: 'A flowchart of yes-or-no questions about the inputs, learned one level at a time.', families: ['plane', 'curve', 'surface', 'digits', 'table', 'series'], blocks: () => [{ type: 'tree', depth: 6, minLeaf: 2 }] },
    { id: 'c-forest', label: 'Random forest', hint: 'Many deep trees, each on a random sample of the data and of the inputs, voting together.', families: ['plane', 'curve', 'surface', 'digits', 'table', 'series'], blocks: () => [{ type: 'forest', trees: 50, depth: 8, features: 'sqrt', minLeaf: 1 }] },
    { id: 'c-boost', label: 'Gradient boosting (XGBoost)', hint: 'Small trees added one after another, each fixing what the others still get wrong: the winner on tables.', families: ['plane', 'curve', 'surface', 'digits', 'table', 'series'], blocks: () => [{ type: 'boost', rounds: 100, eta: 0.3, depth: 4, lambda: 1, gamma: 0, subsample: 1, colsample: 1 }] },
  ];
  // the templates for a kind of data (with the data, only those that suit its input: a CNN for a picture of a world)
  BT.templatesFor = (family, d) => BT.TEMPLATES.filter((t) => t.families.includes(family) && (!t.kinds || (d && d.input && t.kinds.includes(d.input.kind))));
  // the losses that suit a problem; a table's depend on whether its data is classes or a value
  BT.lossesFor = (p, d) => (p.family !== 'table' ? BT.FAMILIES[p.family].losses : d && d.output && d.output.kind === 'value' ? ['mse', 'mae', 'huber'] : ['xent', 'mse', 'hinge']);
  // a template's blocks for a kind of data: dense layers need a flat vector, so on pictures and series the plain
  // layer templates start with a Flatten (classic models read the rows of numbers as they are)
  const FLAT_TEMPLATES = new Set(['linear', 'mlp', 'deep', 'wide']);
  BT.templateBlocks = (t, family, kind) => {
    const blocks = t.blocks();
    // a window of a series read as one row of numbers, like the pixels of a picture (classic models read rows anyway);
    // a world seen as a picture or as readings likewise
    if ((family === 'digits' || family === 'series' || kind === 'image' || kind === 'series') && FLAT_TEMPLATES.has(t.id)) blocks.unshift({ type: 'flatten' });
    return blocks;
  };
  // runs of consecutive blocks that make one copy of a part of your own (they carry the same grp): [{ grp, part, from, to }]
  BT.groupsOf = (blocks) => {
    const out = [];
    (blocks || []).forEach((b, i) => {
      if (!b || !b.grp) return;
      const last = out[out.length - 1];
      if (last && last.grp === b.grp && last.to === i) last.to++;
      else out.push({ grp: b.grp, part: b.part, from: i, to: i + 1 });
    });
    return out;
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
  // Made-up tables to try it without a file: fruit, flowers and penguins to tell apart (classes), house prices and
  // bike hires (a value). Each small enough to go into the Python script whole (at most 2,400 cells).
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
    // three kinds of flower from four measurements: one kind stands apart, the other two overlap a little
    flowers: { label: 'Flowers (3 kinds)', make() {
      const rng = new NA.RNG(47), rows = [], kinds = [['bluebell', 4.8, 3.3, 1.6, 0.3], ['iris', 6.0, 2.7, 4.4, 1.4], ['lily', 6.7, 3.0, 5.7, 2.1]];
      for (let i = 0; i < 180; i++) {
        const [nm, sl, sw, pl, pw] = kinds[i % 3], f = rng.gauss(1, 0.07);
        rows.push([(sl * f + rng.gauss(0, 0.3)).toFixed(1), (sw * f + rng.gauss(0, 0.28)).toFixed(1), Math.max(0.8, pl * f + rng.gauss(0, 0.12 * pl)).toFixed(1), Math.max(0.1, pw * f + rng.gauss(0, 0.12 * pw + 0.03)).toFixed(1), nm]);
      }
      return { name: 'flowers.csv', header: ['sepal_length_cm', 'sepal_width_cm', 'petal_length_cm', 'petal_width_cm', 'kind'], rows, delim: ',', hasHeader: true, cut: 0 };
    } },
    // penguins: two columns of text, which the model leaves out, and a few empty cells, whose rows it skips
    penguins: { label: 'Penguins (gaps and text)', make() {
      const rng = new NA.RNG(53), rows = [], gap = (v) => (rng.next() < 0.025 ? '' : v);
      const kinds = [['rockhopper', 40.5, 18.6, 191, 3700, ['north', 'south', 'east']], ['macaroni', 48.5, 18.3, 197, 3750, ['north', 'north', 'east']], ['king', 47.2, 15.0, 218, 5100, ['south', 'south', 'east']]];
      for (let i = 0; i < 300; i++) {
        const [nm, bl, bd, fl, m, isl] = kinds[i % 3], male = rng.next() < 0.5, s = male ? 1.04 : 0.96;
        rows.push([rng.pick(isl), gap((bl * s + rng.gauss(0, 2.4)).toFixed(1)), gap((bd * s + rng.gauss(0, 0.9)).toFixed(1)), gap((fl * (male ? 1.02 : 0.98) + rng.gauss(0, 6)).toFixed(0)), gap((m * (male ? 1.08 : 0.92) + rng.gauss(0, 300)).toFixed(0)), male ? 'male' : 'female', nm]);
      }
      return { name: 'penguins.csv', header: ['island', 'bill_length_mm', 'bill_depth_mm', 'flipper_mm', 'mass_g', 'sex', 'species'], rows, delim: ',', hasHeader: true, cut: 0 };
    } },
    // bikes hired in an hour: rush hours on working days, a midday hump at weekends, fewer in the cold and the wet. A
    // straight line cannot follow the hours, trees can; and the day of the month does not matter at all.
    bikes: { label: 'Bike hires (a number)', make() {
      const rng = new NA.RNG(59), rows = [], bump = (h, at, w) => Math.exp(-((h - at) ** 2) / (2 * w * w));
      for (let i = 0; i < 400; i++) {
        const hour = rng.int(24), work = rng.next() < 5 / 7 ? 1 : 0, temp = NA.clamp(rng.gauss(14, 8), -6, 34), hum = rng.uniform(30, 98), day = 1 + rng.int(31);
        const base = work ? 20 + 330 * bump(hour, 8, 1.1) + 400 * bump(hour, 17.5, 1.4) + 110 * bump(hour, 13, 3) : 15 + 260 * bump(hour, 14, 3.2);
        const hires = base * NA.clamp(0.25 + (temp + 4) / 30, 0.15, 1.15) * (hum > 85 ? 0.55 : 1) + rng.gauss(0, 15);
        rows.push([String(hour), String(work), temp.toFixed(1), hum.toFixed(0), String(day), String(Math.max(0, Math.round(hires)))]);
      }
      return { name: 'bikes.csv', header: ['hour', 'working_day', 'temp_c', 'humidity_pct', 'day_of_month', 'hires'], rows, delim: ',', hasHeader: true, cut: 0 };
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
  /* A heartbeat: each beat a sharp spike (a small dip, the peak, a deeper dip) and a broad bump after it, as bumps
     [when, how wide, how high] around the beat; a beat every 30 steps, or every 22 to 38 when the rhythm is irregular.
     Between beats the line is flat, so only a window that reaches back to the last beat can tell when the next comes. */
  BT.BEAT = [[-1, 0.5, -0.12], [0, 0.55, 1], [1, 0.6, -0.28], [6, 1.6, 0.3]];
  function heartbeat(v, rng, steady) {
    const N = v.length, beats = [];
    for (let t = rng.int(20); t < N + 20; t += steady ? 30 : 22 + rng.int(17)) beats.push(t);
    for (let t = 0; t < N; t++) for (const b of beats) if (t - b > -6 && t - b < 14) for (const [at, w, h] of BT.BEAT) v[t] += h * Math.exp(-((t - b - at) ** 2) / (2 * w * w));
  }
  function seriesData(p, s, seed) {
    const rng = new NA.RNG(seed * 17 + 3), N = s.n, W = s.window, v = new Float64Array(N);
    if (p.task === 'chaos') { let x = 0.31 + 0.02 * rng.next(); for (let t = 0; t < N; t++) { v[t] = 2 * x - 1; x = 3.9 * x * (1 - x); } }
    // a random walk: each step a coin toss of random size, so the best forecast is where it is now
    else if (p.task === 'walk') { let x = 0; for (let t = 0; t < N; t++) { x += 0.05 * rng.gauss(); v[t] = x; } }
    else if (p.task === 'heartbeat') heartbeat(v, rng, s.rhythm !== 'irregular');
    else for (let t = 0; t < N; t++) v[t] = SERIES[p.task](t);
    for (let t = 0; t < N; t++) v[t] += s.noise * rng.gauss();
    let m = 0; for (let t = 0; t < N; t++) m += v[t] / N;
    let top = 1e-9; for (let t = 0; t < N; t++) top = Math.max(top, Math.abs(v[t] - m));
    for (let t = 0; t < N; t++) v[t] = (v[t] - m) / top;
    const nWin = N - W, nTr = Math.round(nWin * 0.8), nVa = nWin - nTr, X = new Float32Array(nWin * W), Y = new Float32Array(nWin);
    for (let i = 0; i < nWin; i++) { for (let j = 0; j < W; j++) X[i * W + j] = v[i + j]; Y[i] = v[i + W]; }
    // the simplest forecast to beat: the next value is the last one again (its R² on the validation part)
    const Yva = Y.slice(nTr), naive = BT.r2(Float32Array.from({ length: nVa }, (_, i) => v[nTr + i + W - 1]), Yva, nVa);
    return { family: 'series', task: p.task, input: { kind: 'series', len: W, dim: 1 }, output: { kind: 'value', k: 1 }, series: Float32Array.from(v), window: W, splitAt: W + nTr,
      Xtr: X.slice(0, nTr * W), Ytr: Y.slice(0, nTr), nTr, Xva: X.slice(nTr * W), Yva, nVa, inOrder: true, naive };
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
    // their length is a setting
    brackets: { vocab: ['(', ')'], classes: ['unbalanced', 'balanced'] },
    palindrome: { vocab: [...'0123456789'], classes: ['not a palindrome', 'palindrome'] },
  };
  // brackets as 0 = ( and 1 = ): balanced when the count of open ones never drops below zero and ends at zero
  const balanced = (ids) => { let h = 0; for (const v of ids) { h += v ? -1 : 1; if (h < 0) return false; } return h === 0; };
  function seqClassData(p, s, seed) {
    const def = SEQ_TASKS[p.task], T = def.len || s.len, rng = new NA.RNG(seed * 23 + 11), id = Object.fromEntries(def.vocab.map((w, i) => [w, i])), rows = [], seen = new Set();
    let guard = 0;
    while (rows.length < s.n && guard++ < s.n * 30) {
      let ids, label;
      if (p.task === 'sentiment') {
        const r = review(rng);
        if (r.label < 0 || r.words.length > T) continue;
        ids = r.words.map((w) => id[w]); while (ids.length < T) ids.push(0); label = r.label;
      } else if (p.task === 'majority') { ids = NA.range(T).map(() => rng.int(2)); label = ids.reduce((a, b) => a + b, 0) * 2 > T ? 1 : 0; }
      else if (p.task === 'parity') { ids = NA.range(T).map(() => rng.int(2)); label = ids.reduce((a, b) => a + b, 0) % 2; }
      else if (p.task === 'brackets') {
        // as many ( as ) in a random order, kept when it is the class due next (they take turns); a third of the
        // unbalanced ones have two more of one kind. All open first and close last, as balanced ones do, so the ends
        // give nothing away: only a running count tells.
        const want = rows.length % 2, opens = want || rng.next() < 0.67 ? T / 2 : T / 2 + (rng.next() < 0.5 ? 1 : -1);
        ids = NA.range(T).map((i) => (i < opens ? 0 : 1)); rng.shuffle(ids);
        label = balanced(ids) ? 1 : 0;
        if (label !== want || ids[0] !== 0 || ids[T - 1] !== 1) continue;
      } else if (p.task === 'palindrome') {
        // a palindrome, or a near miss: one digit changed, so exactly one pair no longer matches
        ids = NA.range(T).map(() => rng.int(10));
        for (let i = 0; i < T >> 1; i++) ids[T - 1 - i] = ids[i];
        label = rows.length % 2;
        if (!label) { const at = rng.int(T); ids[at] = (ids[at] + 1 + rng.int(9)) % 10; if (ids[at] === ids[T - 1 - at]) continue; }
      } else {
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
  /* A picture as a height to learn from (x, y): a smiling face, the face high, its eyes and mouth low. Each edge is a
     sigmoid of the distance across it, `soft` wide: sharp edges are fine detail that a plain MLP blurs. */
  const sig = (d, soft) => 1 / (1 + Math.exp(-d / soft));
  BT.face = (soft) => (x, y) => {
    const face = sig(0.78 - Math.hypot(x, y), soft);
    const eyes = Math.max(sig(0.12 - Math.hypot(x + 0.28, y - 0.26), soft), sig(0.12 - Math.hypot(x - 0.28, y - 0.26), soft));
    const mouth = sig(0.075 - Math.abs(Math.hypot(x, y - 0.12) - 0.45), soft) * sig(-0.12 - y, soft);
    return -0.6 + 1.4 * face - 1.5 * Math.max(eyes, mouth) * face;
  };
  BT.FACE_SOFT = { sharp: 0.008, soft: 0.06 };
  function surfaceData(p, s, seed) {
    const f = p.task === 'picture' ? BT.face(BT.FACE_SOFT[s.edges] || BT.FACE_SOFT.sharp) : SURFACES[p.task];
    const rng = new NA.RNG(seed * 29 + 5), n = s.n, X = new Float32Array(n * 2), Y = new Float32Array(n);
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

  // ------------------------------------------------------------------ more pictures
  /* 14 × 14 pictures where particular parts shine: dots to count (one detector used everywhere, then added up),
     stripes whose direction is the class (a texture: what, not where), coloured shapes (three channels: red, green,
     blue) and a dot inside or outside a ring (how the parts of a picture relate). Each also gives short labels for
     the results (short), the word for its classes (what) and a few words about one example (about). */
  const IMG = 14, PX = IMG * IMG;
  function pictureSet(X, Y, n, c, rng, task, names, more) {
    const order = NA.range(n); rng.shuffle(order);
    const nTr = Math.round(n * 0.8), tr = order.slice(0, nTr), va = order.slice(nTr), S = c * PX;
    return Object.assign({ family: 'digits', task, input: { kind: 'image', c, h: IMG, w: IMG }, output: { kind: 'classes', k: names.length }, classes: names.length, names,
      Xtr: gather(X, S, tr, Float32Array), Ytr: Int32Array.from(tr.map((i) => Y[i])), nTr, Xva: gather(X, S, va, Float32Array), Yva: Int32Array.from(va.map((i) => Y[i])), nVa: va.length }, more);
  }
  // a round dot of ink: 1 inside, fading to 0 over a pixel at its edge, like the strokes of the shapes
  const dotInk = (px, py, x, y, r) => Math.max(0, Math.min(1, r + 0.5 - Math.hypot(px - x, py - y)));
  // 1 to `most` dots, apart from each other. Of one size, the ink in all says how many; of mixed sizes it does not,
  // and the model has to find each dot.
  function dotsData(s, seed) {
    const rng = new NA.RNG(seed * 43 + 13), n = s.n, K = s.most, X = new Float32Array(n * PX), Y = new Int32Array(n), mixed = s.sizes !== 'same';
    for (let e = 0; e < n; e++) {
      const k = 1 + (e % K);
      let pts = [];
      for (let guard = 0; pts.length < k && guard < 4000; guard++) {
        if (guard % 200 === 199) pts = []; // too crowded for one more: start the picture again
        const r = mixed ? rng.uniform(0.55, 1.6) : 1, x = rng.uniform(r + 0.3, IMG - r - 0.3), y = rng.uniform(r + 0.3, IMG - r - 0.3);
        if (pts.every((q) => Math.hypot(q.x - x, q.y - y) > q.r + r + 1.2)) pts.push({ x, y, r });
      }
      for (let p = 0; p < PX; p++) {
        const px = (p % IMG) + 0.5, py = Math.floor(p / IMG) + 0.5;
        let v = 0;
        for (const q of pts) v = Math.max(v, dotInk(px, py, q.x, q.y, q.r));
        X[e * PX + p] = v + 0.04 * rng.gauss();
      }
      Y[e] = k - 1;
    }
    const names = NA.range(K).map((k) => `${k + 1} dot${k ? 's' : ''}`);
    return pictureSet(X, Y, n, 1, rng, 'dots', names, { short: NA.range(K).map((k) => String(k + 1)), what: 'count', about: (k) => `a picture of ${names[k]}` });
  }
  // stripes at any spacing and shift, faint or strong, in noise: the class is the way they run
  BT.STRIPES = [['—', 0, 'across'], ['|', Math.PI / 2, 'up and down'], ['/', Math.PI / 4, 'rising'], ['\\', -Math.PI / 4, 'falling']];
  function stripesData(s, seed) {
    const rng = new NA.RNG(seed * 47 + 17), n = s.n, K = s.dirs, X = new Float32Array(n * PX), Y = new Int32Array(n);
    for (let e = 0; e < n; e++) {
      const k = e % K, a = BT.STRIPES[k][1], per = rng.uniform(2.6, 5), ph = rng.uniform(0, 2 * Math.PI), c = rng.uniform(0.2, 0.5), nx = -Math.sin(a), ny = Math.cos(a);
      for (let p = 0; p < PX; p++) {
        const px = (p % IMG) + 0.5, py = Math.floor(p / IMG) + 0.5;
        X[e * PX + p] = 0.5 + c * Math.sin((2 * Math.PI * (px * nx + py * ny)) / per + ph) + s.noise * rng.gauss();
      }
      Y[e] = k;
    }
    return pictureSet(X, Y, n, 1, rng, 'stripes', BT.STRIPES.slice(0, K).map((q) => q[0]), { what: 'direction', about: (k) => `stripes running ${BT.STRIPES[k][2]} (${BT.STRIPES[k][0]})` });
  }
  // a red, green or blue circle or cross. The colours are equally bright (the same mean of red, green and blue), so
  // in grey, one channel, they all look alike.
  BT.COLOURS = [['red', [0.95, 0.15, 0.1]], ['green', [0.15, 0.85, 0.2]], ['blue', [0.2, 0.25, 0.75]]];
  function colourData(s, seed) {
    const rng = new NA.RNG(seed * 53 + 19), n = s.n, grey = s.colour === 'grey', C = grey ? 1 : 3, X = new Float32Array(n * C * PX), Y = new Int32Array(n), names = [];
    for (const [nm] of BT.COLOURS) names.push(`${nm} circle`, `${nm} cross`);
    for (let e = 0; e < n; e++) {
      const k = e % 6, col = BT.COLOURS[k >> 1][1], cross = k & 1, r = rng.uniform(2.6, 4.6), x = rng.uniform(r + 0.5, IMG - r - 0.5), y = rng.uniform(r + 0.5, IMG - r - 0.5), a = rng.uniform(0, Math.PI / 2), b = rng.uniform(0.75, 1);
      for (let p = 0; p < PX; p++) {
        const px = (p % IMG) + 0.5 - x, py = Math.floor(p / IMG) + 0.5 - y;
        // a disc, or two bars about two pixels thick, turned by a
        const u = Math.abs(px * Math.cos(a) + py * Math.sin(a)), v = Math.abs(-px * Math.sin(a) + py * Math.cos(a));
        const ink = cross ? Math.max(0, Math.min(1, 1.3 - Math.min(u, v), r + 0.5 - Math.max(u, v))) : Math.max(0, Math.min(1, r + 0.5 - Math.hypot(px, py)));
        if (grey) X[e * PX + p] = 0.1 + (ink * b * (col[0] + col[1] + col[2])) / 3 + 0.04 * rng.gauss();
        else for (let ch = 0; ch < 3; ch++) X[e * 3 * PX + ch * PX + p] = 0.1 + ink * b * col[ch] + 0.04 * rng.gauss();
      }
      Y[e] = k;
    }
    return pictureSet(X, Y, n, C, rng, 'colours', names, { short: ['r●', 'r+', 'g●', 'g+', 'b●', 'b+'], what: 'class', rgb: !grey, about: (k) => `a ${names[k]}${grey ? ', in grey' : ''}` });
  }
  // a ring anywhere, at any size, and a dot inside it or outside it
  function insideData(s, seed) {
    const rng = new NA.RNG(seed * 73 + 41), n = s.n, X = new Float32Array(n * PX), Y = new Int32Array(n);
    for (let e = 0; e < n; e++) {
      const k = e % 2, r = rng.uniform(2.8, 5), cx = rng.uniform(r + 0.6, IMG - r - 0.6), cy = rng.uniform(r + 0.6, IMG - r - 0.6);
      let x = cx, y = cy;
      // clear of the ring by more than a pixel, so the answer is never in doubt
      for (let g = 0; g < 500; g++) { x = rng.uniform(1, IMG - 1); y = rng.uniform(1, IMG - 1); const d = Math.hypot(x - cx, y - cy); if (k ? d < r - 1.6 : d > r + 1.6) break; }
      for (let p = 0; p < PX; p++) {
        const px = (p % IMG) + 0.5, py = Math.floor(p / IMG) + 0.5, ring = Math.max(0, Math.min(1, 1.4 - Math.abs(Math.hypot(px - cx, py - cy) - r)));
        X[e * PX + p] = Math.max(ring, dotInk(px, py, x, y, 0.7)) + 0.04 * rng.gauss();
      }
      Y[e] = k;
    }
    return pictureSet(X, Y, n, 1, rng, 'inside', ['outside', 'inside'], { short: ['out', 'in'], what: 'answer', about: (k) => `a dot ${k ? 'inside' : 'outside'} a ring` });
  }
  const PICTURES = { shapes: shapeData, dots: dotsData, stripes: stripesData, colours: colourData, inside: insideData };

  // ------------------------------------------------------------------ worlds to act in
  /* The settings of each world beside the learning ones (BRL.defaults): how the model sees it (view), the inputs
     it is not given (hide), the extra ones it is (extra) and sensor noise; for the maze which one, a bonus for
     getting closer, a new random maze every episode, a slippery floor and the longest episode; for the pole its
     push, length, gravity, track, the lean it falls at, how far from still it starts and the longest episode; the
     catch's drift, board and paddle; for the levers how many and how often they pay; and each bigger world's own
     (rl-worlds*.js). Every default is the world as it always was. The rewards (rGoal…, or a formula rFormula) are
     left out until you change them: the world then pays as the RL lab's does. */
  const WORLD_DEFAULTS = {
    levers: { levers: 3, pRight: 0.8, pOther: 0.2, hide: [], extra: [], noise: 0 },
    maze: { view: 'around', maze: 'pits', closer: true, random: false, slip: 0, maxSteps: 80, hide: [], extra: [], noise: 0 },
    pole: { view: 'state', force: 10, poleLen: 1, gravity: 9.8, maxSteps: 200, track: 4.8, fall: 12, start: 0.05, hide: [], extra: [], noise: 0 },
    catch: { view: 'numbers', drift: true, width: 5, height: 8, paddle: 1, hide: [], extra: [], noise: 0 },
    // the bigger worlds start with the method that learns them best: evolution for the track and the rocket, PPO for the arena
    track: { view: 'rays', track: 'esses', maxSteps: 0, laps: 4, vmax: 24, grip: 14, range: 40, hide: [], extra: [], noise: 0, method: 'ga' },
    arena: { view: 'rays', map: 'room', drones: 2, speed: 'slow', maxSteps: 500, lives: 3, goal: 10, recharge: 3, hide: [], extra: [], noise: 0, method: 'ppo' },
    lander: { view: 'state', gravity: 1.62, wind: false, pad: 'middle', fuel: 150, maxSteps: 300, padWidth: 12, engine: 'normal', hide: [], extra: [], noise: 0, method: 'ga' },
    thermo: { view: 'state', climate: 'mild', insulation: 'good', schedule: 'day', days: 1, warm: 21, band: 1, radiator: 60, hide: [], extra: [], noise: 0, method: 'ppo' },
    hurdles: { view: 'state', obstacles: 'hurdles', course: 150, run: 8, jump: 5.5, hide: [], extra: [], noise: 0, method: 'ga' },
    memory: { view: 'now', length: 6, lit: 1, wait: 5, hide: [], extra: [], noise: 0, method: 'ppo' },
    mountain: { view: 'state', engine: 'normal', maxSteps: 200, start: 'valley', hide: [], extra: [], noise: 0, method: 'ga' },
    swingup: { view: 'state', motor: 'strong', gravity: 9.8, maxSteps: 300, start: 'down', friction: 0.1, hide: [], extra: [], noise: 0, method: 'ga' },
    arm: { view: 'state', maxSteps: 120, reach: 0.12, motors: 6, hide: [], extra: [], noise: 0, method: 'a2c' },
    // (traffic: over 10 runs from the page's own start, the genetic algorithm got all 10 to 0.95–0.99, DQN 9, PPO 3, A2C 4)
    traffic: { view: 'state', pattern: 'shifting', maxSteps: 400, amber: 4, busy: 1, hide: [], extra: [], noise: 0, method: 'ga' },
    walker: { view: 'state', ground: 'grippy', maxSteps: 300, gravity: 9.8, muscles: 'normal', hide: [], extra: [], noise: 0, method: 'ga' },
    trap: { view: 'state', maze: 'trap', maxSteps: 0, speed: 3, hide: [], extra: [], noise: 0, method: 'novelty' },
    chase: { view: 'state', map: 'open', chasers: 1, speed: 'slow', aim: 'straight', maxSteps: 400, hide: [], extra: [], noise: 0, method: 'ga' },
    herd: { view: 'state', sheep: 3, maxSteps: 600, flee: 5, dog: 0.45, hide: [], extra: [], noise: 0, method: 'ga' },
  };
  BT.VIEWS = {
    maze: [{ value: 'around', label: 'Around it', title: 'The 24 squares around it (walls, pits, the goal when it is that close, where it has been), and which way the goal is, near or far: 104 inputs. What it learns can carry over to other mazes' }, { value: 'square', label: 'Its square', title: 'Which square it is on, one input per square (48). Nothing tells it where the goal is: like a table, it finds it from the rewards, and learns this maze and no other' }, { value: 'picture', label: 'Picture', title: 'The whole maze as a picture of 5 channels, 6 × 8: walls, pits, the goal, where it is, where it has been. For convolutions' }],
    pole: [{ value: 'state', label: '4 numbers', title: 'The cart’s position and speed, the pole’s angle and how fast it turns' }, { value: 'readings', label: 'Last 4 readings', title: 'Only the cart’s position and the pole’s angle, for the last 4 steps (a sequence of 4 × 2): the speeds have to be worked out from how they change' }],
    catch: [{ value: 'numbers', label: '4 numbers', title: 'Where the ball is, which way it drifts, where the paddle is' }, { value: 'pixels', label: 'Pixels', title: 'Two pictures of the 8 × 5 board, now and a step ago (2 channels): which way the ball drifts is in the difference' }],
    track: [{ value: 'rays', label: '7 sensors', title: 'How far the wall is at 90°, 50° and 20° to each side and straight ahead (out to 40 m), and its speed: 8 numbers' }, { value: 'camera', label: 'Camera', title: 'A picture from above, 16 × 16, of the track 32 m ahead and 12 m to each side, turned with the car, and its speed as a second channel. For convolutions' }],
    arena: [{ value: 'rays', label: 'Columns', title: 'For each of 11 columns across its view: how near the wall is and how near a drone is; a radar (which way the nearest drone is) and its lives and gun: 26 numbers' }, { value: 'pixels', label: 'Pixels', title: 'What it sees, 10 × 20 pixels in two channels (walls, brighter when nearer, and drones): learning to aim from the picture. For convolutions' }],
    memory: [{ value: 'now', label: 'Only now', title: 'What it sees this step: the lamp (lit at the first step only), whether it is at the junction and how far along it is: 4 numbers. By the junction the lamp is long dark' }, { value: 'history', label: 'Last 16 readings', title: 'The same 4 readings for each of the last 16 steps (a sequence of 16 × 4): the lamp is still in there when it reaches the junction. Dense layers take it through a Flatten; a GRU reads it in order, oldest first' }],
  };
  // the settings the world itself takes (rl-envs.js names): yours where you set them, its own where you did not
  const WORLD_SETTINGS = {
    levers: (s) => ({ levers: s.levers, p: s.pRight, q: s.pOther }),
    maze: (s) => ({ view: s.view, maze: s.maze, closer: !!s.closer, random: !!s.random, slip: s.slip || 0, maxSteps: s.maxSteps }),
    pole: (s) => ({ view: s.view, force: s.force, length: s.poleLen, gravity: s.gravity, steps: s.maxSteps, track: s.track, fall: s.fall, start: s.start }),
    catch: (s) => ({ view: s.view, drift: s.drift !== false, width: s.width, height: s.height, paddle: s.paddle }),
    track: (s) => ({ view: s.view, track: s.track, steps: s.maxSteps, laps: s.laps, vmax: s.vmax, grip: s.grip, range: s.range }),
    arena: (s) => ({ view: s.view, map: s.map, drones: s.drones, speed: s.speed, steps: s.maxSteps, lives: s.lives, goal: s.goal, recharge: s.recharge }),
    lander: (s) => ({ gravity: s.gravity, wind: !!s.wind, pad: s.pad, fuel: s.fuel, steps: s.maxSteps, padWidth: s.padWidth, engine: s.engine }),
    thermo: (s) => ({ climate: s.climate, insulation: s.insulation, schedule: s.schedule, days: s.days, warm: s.warm, band: s.band, radiator: s.radiator }),
    hurdles: (s) => ({ obstacles: s.obstacles, course: s.course, run: s.run, jump: s.jump }),
    memory: (s) => ({ view: s.view, length: s.length, lit: s.lit, wait: s.wait }),
    mountain: (s) => ({ engine: s.engine, steps: s.maxSteps, start: s.start }),
    swingup: (s) => ({ motor: s.motor, gravity: s.gravity, steps: s.maxSteps, start: s.start, friction: s.friction }),
    arm: (s) => ({ steps: s.maxSteps, reach: s.reach, motors: s.motors }),
    traffic: (s) => ({ pattern: s.pattern, steps: s.maxSteps, amber: s.amber, busy: s.busy }),
    walker: (s) => ({ ground: s.ground, steps: s.maxSteps, gravity: s.gravity, muscles: s.muscles }),
    trap: (s) => ({ maze: s.maze, steps: s.maxSteps, speed: s.speed }),
    chase: (s) => ({ map: s.map, chasers: s.chasers, speed: s.speed, aim: s.aim, steps: s.maxSteps }),
    herd: (s) => ({ sheep: s.sheep, steps: s.maxSteps, flee: s.flee, dog: s.dog }),
  };
  BT.worldSettings = (p, s) => {
    const t = p.task, o = (WORLD_SETTINGS[t] || WORLD_SETTINGS.catch)(s);
    for (const r of NA.rlEnvs.REWARDS[t]) if (typeof s[r.key] === 'number') o[r.key] = s[r.key];
    if (s.rFormula) o.rFormula = s.rFormula;
    if (Array.isArray(s.hide) && s.hide.length) o.hide = s.hide.slice();
    if (Array.isArray(s.extra) && s.extra.length) o.extra = s.extra.slice();
    if (s.noise) o.noise = s.noise;
    return o;
  };
  function controlData(p, s, seed) {
    const E = NA.rlEnvs, BRL = NA.brl, ws = BT.worldSettings(p, s), world = E.make(p.task, ws), method = BRL.METHODS[s.method] ? s.method : 'dqn', K = BRL.outputs(method, world.nA);
    // no examples with answers: observations from random play, so every view that shows an input has one to show
    const n = 200, nTr = 160, X = E.samples(world, n, new NA.RNG(seed * 31 + 7)), F = world.obsSize;
    const names = method === 'dqn' || method === 'ga' ? world.actions.map((a) => (method === 'dqn' ? `Q(${a})` : a)) : [...world.actions, ...(K > world.nA ? ['value'] : [])];
    return { family: 'control', task: p.task, world: p.task, ws, method, s, input: world.input, output: { kind: 'value', k: K }, nA: world.nA, actions: world.actions.slice(), obsNames: world.obsNames, names,
      keep: world.keep ? Array.from(world.keep) : null, hidden: world.hidden || [], extra: world.shown || [], ownReward: !!world.rf,
      Xtr: X.slice(0, nTr * F), Ytr: new Float32Array(nTr * K), nTr, Xva: X.slice(nTr * F), Yva: new Float32Array((n - nTr) * K), nVa: n - nTr };
  }

  // ------------------------------------------------------------------ making data
  /* Returns { family, input, output, Xtr, Ytr, nTr, Xva, Yva, nVa, classes, names, vocab, ... }: inputs as
     Float32Array (vectors, images) or Int32Array (tokens), targets as Int32Array (classes, tokens; −1 = no
     target) or Float32Array (values). Deterministic from the seed. */
  BT.make = function (p, s, seed = 1, custom = {}) {
    const fam = p.family;
    if (fam === 'plane') return p.custom ? drawnPoints(custom.points || [], seed) : planeData(p, s, seed);
    if (fam === 'curve') return curveData(p, s, seed);
    if (fam === 'digits') return PICTURES[p.task] ? PICTURES[p.task](s, seed) : digitData(s, seed);
    if (fam === 'series') return seriesData(p, s, seed);
    if (fam === 'seqclass') return seqClassData(p, s, seed);
    if (fam === 'surface') return surfaceData(p, s, seed);
    if (fam === 'table') return tableData(custom.table, s, seed);
    if (fam === 'control') return controlData(p, s, seed);
    if (p.task === 'language') return languageData(s, seed);
    if (p.task === 'sentences') return sentenceData(s, seed);
    return p.task === 'add' ? addData(s, seed) : algoData(p.task, s, seed);
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
  // fast wiggles: a sine wave that goes up and down `waves` times between −1 and 1
  BT.wiggles = (waves) => (x) => 0.8 * Math.sin(Math.PI * waves * x + 0.4);
  function curveData(p, s, seed) {
    const rng = new NA.RNG(seed * 13 + 5);
    let xs, ys, f = null;
    if (p.custom || p.task === 'wiggles') {
      const c = p.custom ? BT.checkFormula(s.formula) : { f: BT.wiggles(s.waves) };
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
  const ARROW = 10, PLUS = 11;
  // adding: the question is a + b, D digits each, and the answer their sum in D + 1 digits, the largest first
  BT.addAnswer = (xs) => { const D = (xs.length - 1) / 2, out = []; let c = 0; for (let i = D - 1; i >= 0; i--) { const v = xs[i] + xs[D + 1 + i] + c; out.push(v % 10); c = v >= 10 ? 1 : 0; } out.push(c); return out.reverse(); };
  BT.algoAnswer = (task, xs) => (task === 'add' ? BT.addAnswer(xs) : task === 'reverse' ? xs.slice().reverse() : task === 'sort' ? xs.slice().sort((a, b) => a - b) : xs.slice());
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
  /* Adding two numbers: "3 4 7 + 2 5 8 = 0 6 0 5". Two numbers of D digits (leading zeros allowed), then = (in the
     arrow's place), then the sum in D + 1 digits, written as usual, the largest first. Each answer digit needs the two
     digits above it and a carry from the digits to their right. Trained only on the answer, like the tasks above:
     the question is D + 1 + D tokens (len) and the answer D + 1 (ans). */
  function addData(s, seed) {
    const D = s.digits, Q = 2 * D + 1, A = D + 1, T = Q + A, rng = new NA.RNG(seed * 71 + 37), seen = new Set(), all = [];
    const total = Math.min(s.n, Math.pow(10, 2 * D) - 50);
    let guard = 0;
    while (all.length < total && guard++ < total * 20) {
      const xs = [...NA.range(D).map(() => rng.int(10)), PLUS, ...NA.range(D).map(() => rng.int(10))], key = xs.join(',');
      if (seen.has(key)) continue;
      seen.add(key); all.push(xs);
    }
    const nVa = Math.max(50, Math.round(all.length * 0.12)), nTr = all.length - nVa;
    const X = new Int32Array(all.length * T), Y = new Int32Array(all.length * T).fill(-1);
    all.forEach((xs, i) => {
      const seq = xs.concat([ARROW], BT.addAnswer(xs));
      for (let t = 0; t < T; t++) { X[i * T + t] = seq[t]; if (t >= Q) Y[i * T + t] = seq[t + 1]; }
    });
    return { family: 'sequence', task: 'add', input: { kind: 'tokens', vocab: 12, len: T, embed: s.embed }, output: { kind: 'tokens', vocab: 12 }, vocab: [...'0123456789', '=', '+'], len: Q, ans: A, digits: D,
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
     really means: during training it was shown the right answers). Returns n × A tokens. */
  BT.generateAlgo = function (model, prompts, n, L, A = L, V = 11) {
    // a question of L tokens, an answer of A (adding writes one more digit than each number has); V tokens in all
    const T = L + A, X = new Int32Array(n * T), out = new Int32Array(n * A);
    for (let i = 0; i < n; i++) { for (let t = 0; t < L; t++) X[i * T + t] = prompts[i * L + t]; X[i * T + L] = ARROW; }
    for (let k = 0; k < A; k++) {
      const o = model.predict(X, n);
      for (let i = 0; i < n; i++) {
        const base = (i * T + L + k) * V;
        let b = 0; for (let v = 1; v < V; v++) if (o[base + v] > o[base + b]) b = v;
        out[i * A + k] = b;
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
     min max. Multiplication can be implied: 2x, 3sin(x), (x+1)(x−1). Comparisons (< <= > >= == !=, = for ==)
     are 1 when true and 0 when not; and, or, not (or && || !) combine them; a choice is c ? a : b, or if(c, a,
     b): a where c is not 0, b where it is. Returns { f, py } or { error }: f computes the formula, py is the same
     formula as PyTorch code on a tensor x (for the Python tab). No eval.
     vars: other names in place of x (a reward in the names a world offers); f then takes an object of their
     values. opts.scalar: py on plain Python numbers (math.sin, abs, min…) rather than tensors. */
  const FUN = { sin: Math.sin, cos: Math.cos, tan: Math.tan, exp: Math.exp, log: Math.log, ln: Math.log, sqrt: Math.sqrt, abs: Math.abs, tanh: Math.tanh, sign: Math.sign, floor: Math.floor, round: Math.round, relu: (v) => Math.max(0, v), sigmoid: (v) => 1 / (1 + Math.exp(-v)) };
  const FUN2 = { min: Math.min, max: Math.max, pow: Math.pow };
  const PYNUM = (v) => (Number.isNaN(v) ? "float('nan')" : !Number.isFinite(v) ? (v > 0 ? 'math.inf' : '-math.inf') : String(+v.toPrecision(12)));
  // the same functions on one Python number
  const PYS = { sin: 'math.sin', cos: 'math.cos', tan: 'math.tan', exp: 'math.exp', log: 'math.log', ln: 'math.log', sqrt: 'math.sqrt', abs: 'abs', tanh: 'math.tanh', floor: 'math.floor', round: 'round' };
  const pyScalar = (w, a) => (w === 'sign' ? `(1.0 if (${a}) > 0 else -1.0 if (${a}) < 0 else 0.0)` : w === 'relu' ? `max(0.0, ${a})` : w === 'sigmoid' ? `1 / (1 + math.exp(-(${a})))` : `${PYS[w]}(${a})`);
  BT.compile = function (src, vars, opts = {}) {
    const V = vars || ['x'], named = !!vars, scalar = !!opts.scalar;
    const s = String(src || '').replace(/×|·/g, '*').replace(/−/g, '-').replace(/²/g, '^2').replace(/³/g, '^3').replace(/π/g, 'pi').replace(/≤/g, '<=').replace(/≥/g, '>=').replace(/≠/g, '!=').toLowerCase();
    let i = 0;
    const ws = () => { while (s[i] === ' ') i++; };
    const fail = (msg) => { throw new Error(msg); };
    const peek = () => { ws(); return s[i]; };
    const at = (t) => { ws(); return s.startsWith(t, i); };
    // and, or, not are words, not names (nor an implied product: "goal and pit")
    const word = () => { ws(); const m = /^(and|or|not)(?![a-z])/.exec(s.slice(i)); return m ? m[1] : null; };
    const takeWord = (w) => { if (word() !== w) return false; i += w.length; return true; };
    /* Each part is { f, py, k, p }: k when it does not depend on x (it is then folded into one number), p the
       precedence of its Python text (1 + −, 2 × /, 3 unary minus, 4 power, 5 a name, number or call). A test (a
       comparison, and, or, not) is 1 or 0 and also carries its Python as a condition, pc (of precedence pp: 1 or,
       2 and, 3 not, 4 a comparison), so that a choice reads "-1.0 if wall != 0 else 0.0", not as floats. */
    const lit = (v, py) => ({ f: () => v, py: py || PYNUM(v), k: true, p: v < 0 || Object.is(v, -0) ? 3 : 5 });
    const wrap = (a, min) => (a.p < min ? `(${a.py})` : a.py);
    const node = (f, py, k, p) => (k ? lit(f(0)) : { f, py, k: false, p });
    const cond = (a) => a.pc || (scalar ? `${a.py} != 0` : `(${a.py} != 0)`), condP = (a) => (a.pc ? a.pp : 4);
    const par = (a, min) => (condP(a) < min ? `(${cond(a)})` : cond(a));
    const test = (f, pc, pp, parts) => (parts.every((q) => q.k) ? lit(f(0)) : { f, py: scalar ? `float(${pc})` : `${pc}.float()`, pc, pp, k: false, p: 5 });
    // a whole number as Python's float, so a choice between two of them is a float too
    const fl = (a) => (a.k && /^-?\d+$/.test(a.py) ? a.py + '.0' : a.py);
    const choose = (c, a, b) => (c.k ? (c.f(0) !== 0 ? a : b)
      : { f: (x) => (c.f(x) !== 0 ? a.f(x) : b.f(x)), py: scalar ? `(${fl(a)} if ${cond(c)} else ${fl(b)})` : `torch.where(${cond(c)}, ${fl(a)}, ${fl(b)})`, k: false, p: 5 });
    // c ? a : b, the loosest of all (a ? b : c ? d : e reads a ? b : (c ? d : e))
    function full() {
      const c = either();
      if (peek() !== '?') return c;
      i++;
      const a = full();
      if (peek() !== ':') fail('A choice has two answers: c ? a : b, as in wall ? -1 : 0.');
      i++;
      return choose(c, a, full());
    }
    function either() {
      let a = both();
      while (takeWord('or') || (at('||') && (i += 2))) a = joint('or', a, both());
      return a;
    }
    function both() {
      let a = negation();
      while (takeWord('and') || (at('&&') && (i += 2))) a = joint('and', a, negation());
      return a;
    }
    // and, or: with a known value on one side, and with 0 is 0, or with anything but 0 is 1, else the other side decides
    function joint(w, A, b) {
      const known = A.k !== b.k ? (A.k ? A : b) : null, other = known === A ? b : A, and = w === 'and';
      if (known) { const v = known.f(0) !== 0; if (and !== v) return lit(and ? 0 : 1); return other.pc ? other : test((x) => (other.f(x) !== 0 ? 1 : 0), cond(other), condP(other), [other]); }
      return test(and ? (x) => (A.f(x) !== 0 && b.f(x) !== 0 ? 1 : 0) : (x) => (A.f(x) !== 0 || b.f(x) !== 0 ? 1 : 0),
        scalar ? `${par(A, and ? 2 : 1)} ${w} ${par(b, and ? 2 : 1)}` : `(${cond(A)} ${and ? '&' : '|'} ${cond(b)})`, and ? 2 : 1, [A, b]);
    }
    function negation() {
      let not = takeWord('not');
      if (!not && at('!') && !at('!=')) { i++; not = true; }
      if (!not) return compare();
      const a = negation();
      return test((x) => (a.f(x) === 0 ? 1 : 0), scalar ? `not ${par(a, 3)}` : `(~${cond(a)})`, 3, [a]);
    }
    const CMP = { '<': (a, b) => a < b, '<=': (a, b) => a <= b, '>': (a, b) => a > b, '>=': (a, b) => a >= b, '==': (a, b) => a === b, '!=': (a, b) => a !== b };
    function compare() {
      let a = expr();
      for (;;) {
        ws();
        const op = ['<=', '>=', '==', '!=', '<>', '<', '>', '='].find((o) => s.startsWith(o, i));
        if (!op) return a;
        i += op.length;
        const o = op === '=' ? '==' : op === '<>' ? '!=' : op, fn = CMP[o], A = a, b = expr();
        a = test((x) => (fn(A.f(x), b.f(x)) ? 1 : 0), scalar ? `${A.py} ${o} ${b.py}` : `(${A.py} ${o} ${b.py})`, 4, [A, b]);
      }
    }
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
        else if (c && /[a-z0-9.(]/.test(c) && !word()) { op = '*'; b = unary(); } // implied multiplication
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
      if (c === '(') { i++; const f = full(); if (peek() !== ')') fail('A bracket is not closed.'); i++; return f; }
      if (/[0-9.]/.test(c)) { const m = /^\d*\.?\d+(e[+-]?\d+)?/.exec(s.slice(i)); if (!m) fail(`"${c}" is not a number.`); i += m[0].length; return lit(+m[0], m[0]); }
      const m = /^[a-z]+/.exec(s.slice(i));
      if (!m) fail(`I do not understand "${c}".`);
      let w = m[0];
      // a run of letters may hide an implied product, like "xsin": take the longest known name at the start
      const names = [...V, 'pi', 'e', 'if', ...Object.keys(FUN), ...Object.keys(FUN2)].sort((a, b) => b.length - a.length);
      const hit = names.find((n) => w.startsWith(n));
      if (!hit) fail(`Unknown name "${w}". Use ${V.join(', ')}, pi, e and sin, cos, tan, exp, log, sqrt, abs, tanh, sign, floor, round, min, max, if.`);
      w = hit; i += w.length;
      if (V.includes(w)) { const name = w; return { f: named ? (x) => x[name] : (x) => x, py: name, k: false, p: 5 }; }
      if (w === 'pi') return lit(Math.PI, 'math.pi');
      if (w === 'e' && peek() !== '(') return lit(Math.E, 'math.e');
      if (peek() !== '(') fail(w === 'if' ? 'if needs brackets and three values, as in if(wall, -1, 0).' : `"${w}" needs brackets, as in ${w}(x).`);
      i++;
      const a = full();
      if (w === 'if') {
        const next = () => { if (peek() !== ',') fail('if takes three values: if(c, a, b) is a where c is not 0, and b where it is.'); i++; return full(); };
        const yes = next(), no = next();
        if (peek() !== ')') fail('A bracket is not closed.');
        i++;
        return choose(a, yes, no);
      }
      if (w in FUN2) {
        if (peek() !== ',') fail(`${w} takes two values, as in ${w}(x, 0).`);
        i++;
        const b = full();
        if (peek() !== ')') fail('A bracket is not closed.');
        i++;
        const fn = FUN2[w];
        const py = w === 'pow' ? `${wrap(a, 5)} ** ${wrap(b, 3)}` : scalar ? `${w}(${a.py}, ${b.py})` : a.k || b.k ? `torch.clamp(${(a.k ? b : a).py}, ${w === 'min' ? 'max' : 'min'}=${(a.k ? a : b).py})` : `torch.${w === 'min' ? 'minimum' : 'maximum'}(${a.py}, ${b.py})`;
        return node((x) => fn(a.f(x), b.f(x)), py, a.k && b.k, w === 'pow' ? 4 : 5);
      }
      if (peek() !== ')') fail('A bracket is not closed.');
      i++;
      const fn = FUN[w];
      return node((x) => fn(a.f(x)), scalar ? pyScalar(w, a.py) : `torch.${w === 'ln' ? 'log' : w}(${a.py})`, a.k, scalar && w === 'sigmoid' ? 2 : 5);
    }
    try {
      if (!s.trim()) fail(named ? 'Type a formula.' : 'Type a formula in x.');
      const f = full();
      if (i < s.length && peek() !== undefined) fail(`Unexpected "${s[i]}".`);
      const at = (v) => (named ? Object.fromEntries(V.map((n) => [n, v])) : v);
      if (!Number.isFinite(f.f(at(0.37))) && !Number.isFinite(f.f(at(-0.61)))) fail(named ? 'The formula gives no finite values.' : 'The formula has no finite values between −1 and 1.');
      return { f: f.f, py: f.py, constant: f.k };
    } catch (e) { return { error: e.message }; }
  };

  NA.builderTasks = BT;
})(typeof globalThis !== 'undefined' ? (globalThis.NA = globalThis.NA || {}) : (window.NA = window.NA || {}));
