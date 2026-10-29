/* builder-challenges.js — guided challenges for the model builder: a problem, a starting model that falls short,
   and a goal to reach by changing it, from Beginner to Expert. Goals: a validation score (accuracy, R² or token
   accuracy; for learning from rewards, the best score of a run, since a policy can get worse again as it
   learns), at most so many parameters, an optimizer or a way of learning from rewards to keep, a number of
   training steps to do it in, a classic model or a network, a number of dense layers to keep. A run counts only
   on the challenge's own data: its problem settings may not change, except the ones a challenge leaves free.
   Every challenge has a known solution, trained by tools/builder-challenges-check.js, which also checks that the
   starting model does not already pass. */
(function (NA) {
  'use strict';
  const D = (units, act = 'relu') => ({ type: 'dense', units, act });
  const LEVELS = ['Beginner', 'Intermediate', 'Advanced', 'Expert'];
  const LIST = [
    // ---------------------------------------------------------------- Beginner: one change that makes the difference
    {
      id: 'bend', level: 'Beginner', topic: 'Layers', title: 'Bend the line', problem: 'moons', template: 'linear',
      brief: 'Two interlocking moons. The model starts with no hidden layer, so all it can draw is a straight line.',
      goals: [{ metric: 0.95 }],
      hint: 'Add a Dense layer between the input and the output: its neurons give the boundary somewhere to bend.',
      solution: { blocks: [D(16)] },
    },
    {
      id: 'ring', level: 'Beginner', topic: 'Layers', title: 'Close the ring', problem: 'circle', blocks: [D(2)],
      brief: 'A disc inside a ring. The hidden layer has two neurons, and each ReLU neuron draws one straight edge: two edges make a band or a wedge, never a closed shape.',
      goals: [{ metric: 0.95 }],
      hint: 'A shape that closes around the disc needs at least three edges, like a triangle: give the layer a third neuron (select it and raise its size), or more.',
      solution: { blocks: [D(3)] },
    },
    {
      id: 'timid', level: 'Beginner', topic: 'Training', title: 'Too timid to learn', problem: 'moons', template: 'mlp', train: { opt: 'sgd', lr: 0.001 },
      brief: 'The network is big enough, but plain SGD with a learning rate of 0.001 takes steps so small that 3,000 of them barely get anywhere. Keep SGD and make it learn.',
      goals: [{ metric: 0.95 }, { opt: 'sgd' }],
      hint: 'Each step moves every weight by the learning rate times its gradient. Try a learning rate a hundred times bigger: 0.1.',
      solution: { train: { lr: 0.1 } },
    },
    {
      id: 'questions', level: 'Beginner', topic: 'Classic ML', title: 'Ask more questions', problem: 'xor', blocks: [{ type: 'tree', depth: 1, minLeaf: 2 }],
      brief: 'A decision tree of depth 1 asks a single yes-or-no question, so it can only cut the square in two. The four corners of XOR need more.',
      goals: [{ metric: 0.95 }],
      hint: 'Select the tree and raise its maximum depth. A tree picks its questions greedily, one at a time, and on XOR the first question helps little by itself, so give it room: a depth of 4.',
      solution: { blocks: [{ type: 'tree', depth: 4, minLeaf: 2 }] },
    },
    {
      id: 'straight', level: 'Beginner', topic: 'Classic ML', title: 'A straight answer', problem: 'csv', table: 'houses', template: 'mlp',
      brief: 'Predict house prices from area, rooms, age and distance with an R² of at least 0.95, using at most 10 parameters. The network uses hundreds and still falls short.',
      goals: [{ metric: 0.95 }, { params: 10 }],
      hint: 'These prices are close to a weighted sum of the columns. The Linear model (in Classic ML) learns exactly that, one weight per column and a bias, solved exactly in one step.',
      solution: { template: 'c-linear' },
    },
    {
      id: 'hill', level: 'Beginner', topic: 'Layers', title: 'Bend a surface', problem: 'hill', template: 'linear',
      brief: 'A hill and a valley over the square. With no hidden layer the model is a tilted flat plane: R² 0.23 at best.',
      goals: [{ metric: 0.95 }],
      hint: 'A hidden layer lets the surface curve: add a Dense layer or two, or pick the MLP template.',
      solution: { template: 'mlp' },
    },
    {
      id: 'neighbours', level: 'Beginner', topic: 'Classic ML', title: 'Ask the neighbours', problem: 'spiral', template: 'c-linear',
      brief: 'Two spirals, and a classic linear model that can only draw one straight line. Reach 95% with a classic model, no network.',
      goals: [{ metric: 0.95 }, { classic: true }],
      hint: 'k-nearest neighbours answers with the classes of the training points closest to each new point: it needs no training at all. Drop a k-NN part onto the model.',
      solution: { template: 'c-knn' },
    },
    // ---------------------------------------------------------------- Intermediate: know why it works
    {
      id: 'v', level: 'Intermediate', topic: 'Activations', title: 'Two neurons for a V', problem: 'abs', template: 'mlp',
      brief: 'Fit the V shape |x| with a tiny model: an R² of at least 0.97 with no more than 10 parameters. The starting model fits it well, with 321.',
      goals: [{ metric: 0.97 }, { params: 10 }],
      hint: '|x| = ReLU(x) + ReLU(−x), so two neurons are enough in principle. But a ReLU that is off on one side gets no gradient from it, so it may never learn that side: a Leaky ReLU keeps a small slope there.',
      solution: { blocks: [D(2, 'leaky')] },
    },
    {
      id: 'calm', level: 'Intermediate', topic: 'Training', title: 'Calm the steps', problem: 'circle', template: 'mlp', train: { opt: 'momentum', lr: 3 },
      brief: 'Momentum at a learning rate of 3 overshoots: the velocity it builds up carries every step too far, and the network ends up stuck at a coin toss. Keep momentum, and get it to learn.',
      goals: [{ metric: 0.97 }, { opt: 'momentum' }],
      hint: 'Momentum adds up the last steps, so it moves about ten times as far as the learning rate alone. Try a learning rate of 0.1, then press ↺ for a fresh run: the overshooting one has wrecked its weights, and continuing it will not bring them back.',
      solution: { train: { lr: 0.1 }, fresh: true },
    },
    {
      id: 'vanish', level: 'Intermediate', topic: 'Activations', title: 'Wake up a deep network', problem: 'moons', blocks: [D(16, 'sigmoid'), D(16, 'sigmoid'), D(16, 'sigmoid'), D(16, 'sigmoid'), D(16, 'sigmoid'), D(16, 'sigmoid')], train: { opt: 'sgd', lr: 0.1 },
      brief: 'Six layers of sigmoid neurons trained with SGD stay stuck at a coin toss. Keep all six layers and SGD, and get past 95%.',
      goals: [{ metric: 0.95 }, { opt: 'sgd' }, { dense: 6 }],
      hint: 'The sigmoid’s slope is at most 0.25, so the gradient shrinks at least four times at every layer it passes back through, and the first layers barely learn (vanishing gradients). Switch the activations to ReLU, or put a BatchNorm after each layer.',
      solution: { blocks: [D(16), D(16), D(16), D(16), D(16), D(16)] },
    },
    {
      id: 'speed', level: 'Intermediate', topic: 'Training', title: 'Speed run', problem: 'moons', template: 'mlp', train: { opt: 'sgd', lr: 0.01 },
      brief: 'Plain SGD gets there in the end, but slowly. Reach 95% within the first 300 training steps of a run.',
      goals: [{ metric: 0.95 }, { steps: 300 }],
      hint: 'Adam adapts every weight’s step to the size of its recent gradients and gets moving much faster: switch the optimizer to Adam (or give SGD a learning rate ten times larger), then start a new run.',
      solution: { train: { opt: 'adam' } },
    },
    {
      id: 'spiral', level: 'Intermediate', topic: 'Layers', title: 'Small but mighty', problem: 'spiral', template: 'deep',
      brief: 'Five layers of 24 neurons wind around the two spirals easily, with almost 2,500 parameters. Do it with at most 600.',
      goals: [{ metric: 0.95 }, { params: 600 }],
      hint: 'Depth matters more than width here: two or three narrow layers can still wind around twice.',
      solution: { blocks: [D(16), D(16)] },
    },
    {
      id: 'wiggle', level: 'Intermediate', topic: 'Activations', title: 'A wiggle on a budget', problem: 'formula', template: 'mlp', ps: { formula: 'sin(5x) + x^2' },
      brief: 'Fit sin(5x) + x² with an R² of at least 0.96, using at most 40 parameters. The starting model gets there with 321.',
      goals: [{ metric: 0.96 }, { params: 40 }],
      hint: 'The curve is mostly a sine wave, and a neuron’s activation can be one: try a single layer of about a dozen Sine neurons.',
      solution: { blocks: [D(12, 'sine')] },
    },
    {
      id: 'order', level: 'Intermediate', topic: 'Sequences', title: 'Word order matters', problem: 'sentiment', template: 'bag',
      brief: 'Tiny reviews like “the food was good but the staff was rude”. The bag of words averages the word vectors, so it cannot tell which clause comes after “but”. Reach 97%.',
      goals: [{ metric: 0.97 }],
      hint: 'Read the words in order: an LSTM goes through them one by one, and a 1-D convolution sees each word with its neighbours. Try the 1-D CNN or LSTM template.',
      solution: { template: 'convclass' },
    },
    {
      id: 'pattern', level: 'Intermediate', topic: 'Sequences', title: 'Spot the pattern', problem: 'pattern', template: 'bag',
      brief: 'Does a 3 come right before a 7 somewhere in the digits? Averaging the digits’ vectors loses which digit sits next to which. Reach 98%.',
      goals: [{ metric: 0.98 }],
      hint: 'A 1-D convolution with a window of 2 or 3 looks at neighbouring digits together, and a max pool over the positions finds the match wherever it is: the 1-D CNN template does both.',
      solution: { template: 'convclass' },
    },
    {
      id: 'chaos', level: 'Intermediate', topic: 'Forecasting', title: 'Predict chaos', problem: 'chaos', template: 'linear',
      brief: 'The logistic map: a chaotic series that never repeats, yet each value follows from the one before. A linear forecaster gets an R² of 0.31. Reach 0.95 with a network.',
      goals: [{ metric: 0.95 }, { classic: false }],
      hint: 'The rule is a curve (a parabola of the last value), which no weighted sum can follow. Add a hidden layer after the Flatten, or use a recurrent layer: the MLP or LSTM template.',
      solution: { template: 'mlp' },
    },
    {
      id: 'digits', level: 'Intermediate', topic: 'Vision', title: 'Read the digits', problem: 'digits', template: 'linear',
      brief: 'A single weighted sum per digit can only match one template per class. Reach 97% validation accuracy.',
      goals: [{ metric: 0.97 }],
      hint: 'Convolutions look for the same small stroke everywhere in the picture: try the Small CNN template.',
      solution: { template: 'cnn' },
    },
    {
      id: 'shapes', level: 'Intermediate', topic: 'Vision', title: 'Find the shapes', problem: 'shapes', template: 'mlp',
      brief: 'Circles, squares, triangles, crosses and lines, anywhere in the picture, at any size and angle. A plain network has to learn each shape at every position separately. Reach 95%.',
      goals: [{ metric: 0.95 }],
      hint: 'Convolutions use the same filters at every position, so a stroke learned in one place is found everywhere: try the Small CNN template.',
      solution: { template: 'cnn' },
    },
    // ---------------------------------------------------------------- Advanced: pick the right tool
    {
      id: 'lookback', level: 'Advanced', topic: 'Sequences', title: 'Look back', problem: 'reverse', template: 'noattn',
      brief: 'Reverse a sequence of digits. Without attention, each position only sees itself, so the answers are guesses. Reach 90% token accuracy.',
      goals: [{ metric: 0.9 }],
      hint: 'Add a Self-attention block before the dense layers, so every answer position can look back at the input it needs.',
      solution: { template: 'attention' },
    },
    {
      id: 'ripples', level: 'Advanced', topic: 'Activations', title: 'Ripples', problem: 'ripples', template: 'mlp',
      brief: 'Rings of ripples fading out from the centre. The network fits the broad shape and smooths the ripples away: an R² of 0.8 to 0.9. Reach 0.93 within the first 3,000 steps of a run.',
      goals: [{ metric: 0.93 }, { steps: 3000 }],
      hint: 'Networks learn smooth shapes first and fine wiggles slowly (their “spectral bias”). Fourier features hand them waves of many frequencies to build with: put a Fourier part first (or use the Fourier MLP template) and tune its scale, the spread of the frequencies. Too high gives noise, too low no help: try 1.5.',
      solution: { blocks: [{ type: 'fourier', features: 32, scale: 1.5 }, D(32), D(32)] },
    },
    {
      id: 'checker', level: 'Advanced', topic: 'Classic ML', title: 'Checkerboard', problem: 'checker', template: 'mlp',
      brief: 'A surface of bumps and dips like a checkerboard. In its first 1,500 steps the network gets an R² of about 0.85. Reach 0.92 within 1,500 steps, with any model.',
      goals: [{ metric: 0.92 }, { steps: 1500 }],
      hint: 'Every square of the board is its own little bump, so what matters is what the nearby points say: k-nearest neighbours answers from exactly those.',
      solution: { template: 'c-knn' },
    },
    {
      id: 'forest', level: 'Advanced', topic: 'Classic ML', title: 'Wisdom of the trees', problem: 'moons', ps: { noise: 0.5, n: 400 }, blocks: [{ type: 'tree', depth: 8, minLeaf: 1 }],
      brief: 'Very noisy moons. A deep decision tree carves out a box for every stray point and gets 88% on new ones. Reach 93% with a classic model.',
      goals: [{ metric: 0.93 }, { classic: true }],
      hint: 'A random forest grows many trees, each on its own random sample of the rows, and averages them: the noise each tree follows averages out. Try a forest with a depth of about 5.',
      solution: { blocks: [{ type: 'forest', trees: 50, depth: 5, features: 'sqrt', minLeaf: 1 }] },
    },
    {
      id: 'saddle', level: 'Advanced', topic: 'Activations', title: 'A smooth saddle', problem: 'saddle', blocks: [D(4)],
      brief: 'The saddle x² − y² with four neurons. ReLU builds it from flat pieces and gets an R² of 0.95. Reach 0.99 with at most 20 parameters.',
      goals: [{ metric: 0.99 }, { params: 20 }],
      hint: 'Four flat pieces cannot make a smooth curve. Keep four neurons and give them a smooth activation that can bend, such as Tanh or Gaussian.',
      solution: { blocks: [D(4, 'tanh')] },
    },
    {
      id: 'stumps', level: 'Advanced', topic: 'Classic ML', title: 'More than stumps', problem: 'spiral', blocks: [{ type: 'boost', rounds: 100, eta: 0.3, depth: 1, lambda: 1, gamma: 0, subsample: 1, colsample: 1 }],
      brief: 'Gradient boosting with trees of depth 1 (“stumps”): each asks about one input only, and 100 of them reach 91% on the spirals. Reach 98%.',
      goals: [{ metric: 0.98 }],
      hint: 'A sum of stumps is a function of x₁ plus a function of x₂: it cannot capture the two together, and a spiral needs exactly that. Give the trees a depth of 3 or 4.',
      solution: { blocks: [{ type: 'boost', rounds: 100, eta: 0.3, depth: 4, lambda: 1, gamma: 0, subsample: 1, colsample: 1 }] },
    },
    {
      id: 'chaostrees', level: 'Advanced', topic: 'Forecasting', title: 'Chaos with trees', problem: 'chaos', template: 'c-linear',
      brief: 'The chaotic series again, now with classic models only. Linear regression gets an R² of 0.31. Reach 0.97 without a network.',
      goals: [{ metric: 0.97 }, { classic: true }],
      hint: 'Trees cut the inputs into boxes and give each box its own answer, so they can follow a curve that a straight line cannot. Try gradient boosting or a random forest.',
      solution: { template: 'c-boost' },
    },
    // ---------------------------------------------------------------- Expert: insight, and more than one change
    {
      id: 'parity', level: 'Expert', topic: 'Sequences', title: 'Odd one out', problem: 'parity', template: 'lstmclass',
      brief: 'Is the number of 1s among ten bits odd? Every bit flips the answer, so nothing short of all ten helps, and the LSTM sits at a coin toss for thousands of steps. Reach 90% within the first 2,000 steps of a run.',
      goals: [{ metric: 0.9 }, { steps: 2000 }],
      hint: 'An LSTM gets there in the end, but slowly: it has to carry one bit of memory, flipped by every 1, through all ten steps. Let dense layers see all ten bits at once instead: a Flatten, then two Dense layers of 64, with a learning rate of 0.003.',
      solution: { blocks: [{ type: 'flatten' }, D(64), D(64)], train: { lr: 0.003, steps: 2000 } },
    },
    {
      id: 'tinychaos', level: 'Expert', topic: 'Forecasting', title: 'Chaos in a nutshell', problem: 'chaos', template: 'mlp', free: ['window'],
      brief: 'The network forecasts the chaotic series well, from a window of 16 values with 561 parameters. Do it with at most 10: R² of 0.95 or more. You may change the window.',
      goals: [{ metric: 0.95 }, { params: 10 }],
      hint: 'Each value follows from the previous one alone, by a parabola. Set the window to 1 in the problem’s settings, keep the Flatten, and use a single Dense neuron with a Gaussian activation: a bump, much like a parabola.',
      solution: { ps: { window: 1 }, blocks: [{ type: 'flatten' }, D(1, 'gaussian')] },
    },
    {
      id: 'budget', level: 'Expert', topic: 'Sequences', title: 'Reviews on a budget', problem: 'sentiment', template: 'lstmclass', free: ['embed'],
      brief: 'The LSTM reads the reviews perfectly, with over 7,000 parameters. Reach 97% with at most 600. You may change the embedding size.',
      goals: [{ metric: 0.97 }, { params: 600 }],
      hint: 'Most parameters sit in the embedding: 16 numbers for every word and position. Set the embedding to 8 in the problem’s settings, then use a small 1-D convolution (8 filters, width 3) and a Sequence pool set to max.',
      solution: { ps: { embed: 8 }, blocks: [{ type: 'conv1d', filters: 8, k: 3, dilation: 1, causal: false, act: 'relu', residual: false }, { type: 'seqpool', mode: 'max' }] },
    },
  ];
  // ---------------------------------------------------------------- learning from rewards
  LIST.push(
    {
      id: 'explore', level: 'Beginner', topic: 'Rewards', title: 'Try the other levers', problem: 'levers', template: 'mlp', ps: { levers: 5, explore: 0 }, free: ['explore'],
      brief: 'Five lights and five levers. DQN starts without exploring: it always pulls the lever that looks best so far, so a lever that paid once gets pulled for good, and the others are never tried.',
      goals: [{ metric: 0.95 }],
      hint: 'Select the Output block and let it explore: random moves for the first 300 moves or more, fewer and fewer as it learns.',
      solution: { ps: { explore: 300 } },
    },
    {
      id: 'warmer', level: 'Beginner', topic: 'Rewards', title: 'Warmer, colder', problem: 'rlmaze', template: 'mlp', ps: { closer: false, method: 'a2c' }, free: ['closer'],
      brief: 'The Pits maze with rewards only at the end: +1 at the goal, −1 in a pit. Random moves almost never reach the goal, so the actor-critic never finds out there is one.',
      goals: [{ metric: 0.95 }],
      hint: 'Turn on the closer bonus in the Input block’s settings: a little reward for every step toward the goal, and as much off for every step away (reward shaping).',
      solution: { ps: { closer: true } },
    },
    {
      id: 'motion', level: 'Intermediate', topic: 'Rewards', title: 'Feel the motion', problem: 'pole', ps: { view: 'readings', method: 'pg' }, blocks: [{ type: 'seqpool', mode: 'last' }, D(16), D(16)],
      brief: 'The pole seen as its last 4 readings of position and angle. The model starts with a Sequence pool that keeps only the latest reading, so it cannot tell a pole that is falling from one that is rising.',
      goals: [{ metric: 0.95 }],
      hint: 'Let the model see all 4 readings: replace the Sequence pool with a Flatten (or read them with a recurrent layer). How the readings change is the speed.',
      solution: { blocks: [{ type: 'flatten' }, D(16), D(16)] },
    },
    {
      id: 'farsight', level: 'Intermediate', topic: 'Rewards', title: 'Look further ahead', problem: 'catch', template: 'mlp', ps: { gamma: 0, method: 'a2c' }, free: ['gamma'], train: { steps: 5000 },
      brief: 'Catch pays only when the ball lands. With a discount of 0 nothing that comes later counts, so the moves that bring the paddle under the ball never learn that they helped.',
      goals: [{ metric: 0.9 }],
      hint: 'Raise the discount γ on the Output block to 0.95: a reward a few steps away still counts now, and flows back to the moves that earned it.',
      solution: { ps: { gamma: 0.95 } },
    },
    {
      id: 'steady', level: 'Intermediate', topic: 'Rewards', title: 'Steady hands', problem: 'pole', template: 'mlp', ps: { method: 'a2c' }, train: { lr: 0.3 },
      brief: 'Actor-critic on the pole with a learning rate of 0.3: every update throws the policy somewhere new, and it never settles. Learning from rewards is touchier than learning from answers.',
      goals: [{ metric: 0.95 }],
      hint: 'Take smaller steps: a learning rate of 0.003, then press ↺ for a fresh run. The jumpy run has already settled on moves that drop the pole, and carrying on from it will not bring it back.',
      solution: { train: { lr: 0.003 }, fresh: true },
    },
    {
      id: 'evolve', level: 'Advanced', topic: 'Rewards', title: 'Evolve a balancer', problem: 'pole', template: 'mlp', ps: { method: 'ga' },
      brief: 'Balance the pole by evolution alone, with at most 12 weights. The genetic algorithm starts with the Small MLP: 386 of them.',
      goals: [{ metric: 0.95 }, { params: 12 }, { method: 'ga' }],
      hint: 'A weighted sum of the 4 numbers can keep the pole up. Pick the Linear template (no hidden layer: 10 weights): with so few, every mutation counts.',
      solution: { template: 'linear' },
    },
    {
      id: 'pixels', level: 'Advanced', topic: 'Rewards', title: 'Catch from pixels', problem: 'catch', template: 'mlp', ps: { view: 'pixels', method: 'a2c' }, train: { steps: 5000 },
      brief: 'Catch from two pictures of the board, now and a step ago, with at most 300 weights. The Small MLP needs 1,636 for the 80 pixels.',
      goals: [{ metric: 0.95 }, { params: 300 }],
      hint: 'A convolution shares a few weights across the whole picture: a Convolution of 4 filters with a stride of 2 (it moves 2 pixels at a time, so the picture it gives is half the size), then a Flatten.',
      solution: { blocks: [{ type: 'conv', filters: 4, k: 3, stride: 2, act: 'relu' }, { type: 'flatten' }] },
    },
    {
      id: 'anymaze', level: 'Expert', topic: 'Rewards', title: 'Any maze', problem: 'rlmaze', template: 'mlp', ps: { random: true, view: 'square', method: 'a2c' }, free: ['view'], train: { steps: 10000 },
      brief: 'A new random maze every episode, judged on 20 mazes it has never seen. Seeing only which square it is on, it cannot know where the walls, the pits or the goal of a new maze are.',
      goals: [{ metric: 0.85 }],
      hint: 'Let it see what is around it: set It sees to Around it, in the Input block’s settings. Walls, pits and the way to the goal look the same in every maze, so what it learns carries over.',
      solution: { ps: { view: 'around' } },
    },
  );
  const byId = Object.fromEntries(LIST.map((c) => [c.id, c]));
  const isClassic = (type) => !!(NA.cml && NA.cml.isClassic(type));
  const TABLE_NAMES = {};
  const tableName = (id) => TABLE_NAMES[id] || (TABLE_NAMES[id] = NA.builderTasks.SAMPLES[id].make().name);

  // the problem settings a challenge is played with
  const dataSettings = (ch) => Object.assign(NA.builderTasks.defaults(NA.builderTasks.byId[ch.problem]), ch.ps);
  // whether a setup (or a run's) uses the challenge's data: the same settings, except the ones it leaves free
  function sameData(ch, cfg) {
    const want = dataSettings(ch), got = ((cfg && cfg.ps) || {})[ch.problem] || {}, free = ch.free || [], def = NA.builderTasks.defaults(NA.builderTasks.byId[ch.problem]);
    if (ch.table && !(cfg.custom && cfg.custom.table && cfg.custom.table.name === tableName(ch.table))) return false;
    // a setting saved before it existed is its default
    return Object.keys(want).every((k) => free.includes(k) || JSON.stringify(got[k] === undefined ? def[k] : got[k]) === JSON.stringify(want[k]));
  }

  /* How a run measures up: one entry per goal { goal, ok, now } (now: the run's value, for the progress line), and
     last whether it trained on the challenge's data. run: { cfg (the setup it trained), params, step, last: { vm } }. */
  function progress(ch, run) {
    const cfg = run ? run.cfg : null, blocks = cfg ? cfg.blocks : [];
    const out = ch.goals.map((g) => {
      // learning from rewards: the best score of the run (a policy can get worse again as it learns)
      if (g.metric !== undefined) { const v = run && run.bestVm !== undefined && run.bestVm !== null ? run.bestVm : run && run.last ? run.last.vm : null; return { goal: g, ok: v !== null && v !== undefined && v >= g.metric, now: v }; }
      if (g.method !== undefined) { const v = cfg ? ((cfg.ps || {})[ch.problem] || {}).method : null; return { goal: g, ok: v === g.method, now: v }; }
      if (g.params !== undefined) { const v = run ? run.params : null; return { goal: g, ok: v !== null && v <= g.params, now: v }; }
      if (g.opt !== undefined) { const v = cfg ? cfg.train.opt : null; return { goal: g, ok: v === g.opt, now: v }; }
      if (g.steps !== undefined) { const v = run ? run.step : null; return { goal: g, ok: v !== null && v !== undefined && v <= g.steps, now: v }; }
      if (g.classic !== undefined) { const v = cfg ? blocks.some((b) => isClassic(b.type)) : null; return { goal: g, ok: v === g.classic, now: v }; }
      if (g.dense !== undefined) { const v = cfg ? blocks.filter((b) => b.type === 'dense').length : null; return { goal: g, ok: v !== null && v >= g.dense, now: v }; }
      return { goal: g, ok: false, now: null };
    });
    out.push({ goal: { data: true }, ok: !!cfg && sameData(ch, cfg), now: null });
    return out;
  }
  const passed = (ch, run) => !!run && run.cfg.problem === ch.problem && progress(ch, run).every((p) => p.ok);
  // a goal in a few words, for tools and short lists
  const describe = (g) => (g.metric !== undefined ? `score ≥ ${g.metric}` : g.params !== undefined ? `≤ ${g.params} params` : g.opt !== undefined ? `optimizer ${g.opt}` : g.method !== undefined ? `learns by ${g.method}` : g.steps !== undefined ? `within ${g.steps} steps`
    : g.classic !== undefined ? (g.classic ? 'a classic model' : 'a network') : g.dense !== undefined ? `≥ ${g.dense} dense layers` : g.data ? 'its own data' : '?');

  // The setup a challenge starts from (blocks without ids), or with solved, its known solution.
  function setup(ch, solved) {
    const BT = NA.builderTasks, p = BT.byId[ch.problem], sol = solved ? ch.solution : {};
    const custom = { points: [] };
    if (ch.table) custom.table = BT.SAMPLES[ch.table].make();
    const ps = Object.assign(BT.defaults(p), ch.ps, sol.ps), d = ch.table ? BT.make(p, ps, 1, custom) : null;
    // a world seen as a picture or as readings: the plain templates start with a Flatten
    const kind = p.family === 'control' ? BT.make(p, ps, 1, custom).input.kind : null, tpl = (id) => BT.templateBlocks(BT.TEMPLATES.find((t) => t.id === id), p.family, kind);
    return {
      problem: ch.problem, ps: { [ch.problem]: ps },
      blocks: JSON.parse(JSON.stringify(sol.blocks || (sol.template ? tpl(sol.template) : ch.blocks || tpl(ch.template)))),
      loss: { id: BT.lossesFor(p, d)[0], smooth: 0, delta: 0.2 },
      train: Object.assign({}, BT.DEFAULT_TRAIN, BT.trainFor(p), p.family === 'control' ? { steps: NA.brl.stepsFor(p.task, ps.method) } : null, ch.train, sol.train),
      custom,
    };
  }

  // the kind of data a challenge is scored on: a table's is a value (R²) or classes (accuracy), whichever its target is
  const FAM = {};
  function family(ch) {
    if (FAM[ch.id]) return FAM[ch.id];
    const BT = NA.builderTasks, p = BT.byId[ch.problem];
    if (!ch.table) return (FAM[ch.id] = p.family);
    const S = setup(ch);
    return (FAM[ch.id] = BT.make(p, S.ps[ch.problem], 1, S.custom).family);
  }

  NA.builderChallenges = { LIST, LEVELS, byId, progress, passed, setup, sameData, describe, family };
})(window.NA);
