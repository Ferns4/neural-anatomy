/* lab-builder.js — build a model from scratch. Pick a problem (or make your own: draw points, type a function,
   write sentences), drag parts from the palette into a stack between the input and the output (or click them),
   set the loss and the training settings, and train. Each training run is recorded with what changed since the
   one before, so you can see what a change did. The engine is js/builder-engine.js; problems are in
   js/builder-tasks.js, pictures of the results in js/builder-viz.js and the PyTorch version in js/builder-py.js.
   Learning from rewards has no examples with answers: the network acts in a world (js/rl-envs.js) and a method
   from js/builder-rl.js (DQN, REINFORCE, actor-critic, PPO, or one of evolution's methods) turns what happens into steps. */
(function (NA) {
  'use strict';
  const { el, clear, clamp, fmt, fmtPct, fmtInt, theme } = NA;
  const H = NA.labHost, UI = NA.labUI, BT = NA.builderTasks, BV = NA.builderViz;
  const ACTS = NA.engine.ACTIVATIONS, ACT_ORDER = NA.engine.ACTIVATION_ORDER;

  // ------------------------------------------------------------------ the parts
  const GROUP_TONE = { Layers: 's1', Sequences: 's7', Images: 's4', Graphs: 's5', Reshape: 's6', Combine: 's3', 'Classic ML': 's5', Rewards: 's8' };
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
    { type: 'rbf', group: 'Layers', label: 'RBF', name: 'Radial basis units', make: () => ({ type: 'rbf', units: 16, width: 0.5 }),
      hint: 'Units that answer how close the input is to their own learned centre: bumps that cover the input space.' },
    { type: 'maxout', group: 'Layers', label: 'Maxout', name: 'Maxout', make: () => ({ type: 'maxout', units: 16, pieces: 2 }),
      hint: 'Each unit keeps the largest of a few weighted sums, so it learns its own bend instead of using a fixed activation.' },
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
    { type: 'attnpool', group: 'Sequences', label: 'Attn. pool', name: 'Attention pool', make: () => ({ type: 'attnpool' }),
      hint: 'Turns a sequence (or a picture) into one vector by a weighted average: it learns where to look for each example.' },
    { type: 'ssm', group: 'Sequences', label: 'State space', name: 'State-space layer', make: () => ({ type: 'ssm', state: 16, units: 16, act: 'gelu' }),
      hint: 'A memory of moving averages with learned time scales, read one step at a time: the idea behind S4 and Mamba.' },
    { type: 'conv', group: 'Images', label: 'Conv', name: 'Convolution', make: () => ({ type: 'conv', filters: 8, k: 3, stride: 1, act: 'relu' }),
      hint: 'Small learned filters slid over an image: each finds one local pattern.' },
    { type: 'pool', group: 'Images', label: 'Max pool', name: 'Max pool', make: () => ({ type: 'pool', size: 2 }),
      hint: 'Keeps the largest value of each 2 × 2 patch: halves the picture.' },
    { type: 'avgpool', group: 'Images', label: 'Avg pool', name: 'Average pool', make: () => ({ type: 'avgpool', size: 2 }),
      hint: 'Averages each 2 × 2 patch: halves the picture, smoothly.' },
    { type: 'patches', group: 'Images', label: 'Patches', name: 'Patches', make: () => ({ type: 'patches', size: 2, dim: 32 }),
      hint: 'Cuts the picture into small squares and turns each into a token: how a vision transformer sees.' },
    { type: 'sepconv', group: 'Images', label: 'Separable', name: 'Separable convolution', make: () => ({ type: 'sepconv', filters: 16, k: 3, dilation: 1, stride: 1, act: 'relu' }),
      hint: 'A filter for each channel on its own, then a 1 × 1 mix: most of a convolution for far fewer weights (MobileNet). Dilation widens its view.' },
    { type: 'se', group: 'Images', label: 'Squeeze-excite', name: 'Squeeze-and-excitation', make: () => ({ type: 'se', ratio: 4 }),
      hint: 'Rates each channel from its average and turns it up or down: the network learns which features matter for this picture.' },
    { type: 'coords', group: 'Images', label: 'Coordinates', name: 'Coordinate channels', make: () => ({ type: 'coords', radius: false }),
      hint: 'Adds two channels holding each pixel’s column and row, so the filters after it know where they are.' },
    { type: 'augment', group: 'Images', label: 'Shifts', name: 'Random shifts', make: () => ({ type: 'augment', shift: 2, flip: false }),
      hint: 'Moves each picture a pixel or two at random while training (and can mirror it): the model learns that small moves do not matter.' },
    { type: 'kan', group: 'Layers', label: 'KAN', name: 'Kolmogorov–Arnold layer', make: () => ({ type: 'kan', units: 8, grid: 6 }),
      hint: 'A layer whose every edge is a learned curve, not a number: it learns the bends themselves.' },
    { type: 'hopfield', group: 'Layers', label: 'Hopfield', name: 'Hopfield memory', make: () => ({ type: 'hopfield', memories: 16, beta: 4 }),
      hint: 'Remembers patterns it learns, and gives back the one most like what comes in: attention over a memory.' },
    { type: 'skip', group: 'Combine', label: 'Skip', name: 'Skip connection', make: () => ({ type: 'skip', back: 1, merge: 'add' }),
      hint: 'Adds (or joins) the output of an earlier block: a shortcut for the signal and its gradient, as in ResNets and U-Nets.' },
    { type: 'upsample', group: 'Images', label: 'Upsample', name: 'Upsample', make: () => ({ type: 'upsample', size: 2 }),
      hint: 'Makes the picture bigger again, each pixel repeated: the way back up in a U-Net or a decoder.' },
    { type: 'sample', group: 'Layers', label: 'Sample', name: 'Sample (VAE)', make: () => ({ type: 'sample', beta: 0.3 }),
      hint: 'The middle of a variational autoencoder: it reads a middle and a spread for each number and, while it learns, draws a code from that bell curve.' },
    { type: 'graphconv', group: 'Graphs', label: 'Graph conv', name: 'Graph convolution', make: () => ({ type: 'graphconv', units: 16, act: 'relu', agg: 'mean' }),
      hint: 'Message passing: each node’s new features come from its own and its neighbours’. Needs a graph.' },
    { type: 'flatten', group: 'Reshape', label: 'Flatten', name: 'Flatten', make: () => ({ type: 'flatten' }),
      hint: 'Lines all the numbers up into one vector, for dense layers.' },
    { type: 'gap', group: 'Reshape', label: 'Global pool', name: 'Global average pool', make: () => ({ type: 'gap' }),
      hint: 'Averages each channel over every position: one vector, whatever the size.' },
    { type: 'gmp', group: 'Reshape', label: 'Global max', name: 'Global max pool', make: () => ({ type: 'gmp' }),
      hint: 'Keeps each channel’s largest value anywhere: did this pattern appear at all?' },
    { type: 'parallel', group: 'Combine', label: 'Parallel', name: 'Parallel branches', make: () => ({ type: 'parallel', merge: 'concat', auto: true, branches: [[], []] }),
      hint: 'Two to four branches side by side on the same input, joined or added at the end: Inception modules, wide-and-deep models, mixtures.' },
    { type: 'moe', group: 'Combine', label: 'Experts', name: 'Mixture of experts', make: () => ({ type: 'moe', experts: 4, units: 16, top: 2, act: 'relu' }),
      hint: 'Several small expert layers and a gate that chooses, for each example, which of them answer.' },
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
    // learning from rewards: not layers but ways to train the network you built (they set the Output block's Learns by)
    { type: 'rl-dqn', group: 'Rewards', label: 'DQN', name: 'DQN (Q-learning)', method: 'dqn', world: 'rlmaze',
      hint: 'Learns a Q-value for every action, the reward it expects for it, from a replay memory of past moves.' },
    { type: 'rl-pg', group: 'Rewards', label: 'REINFORCE', name: 'REINFORCE (policy gradient)', method: 'pg', world: 'pole',
      hint: 'Learns the probability of each action: the moves before a good return become more likely.' },
    { type: 'rl-a2c', group: 'Rewards', label: 'Actor-critic', name: 'Actor-critic (A2C)', method: 'a2c', world: 'catch',
      hint: 'A policy (the actor) and a judge of each state (the critic), learning from the surprise of every few moves.' },
    { type: 'rl-ppo', group: 'Rewards', label: 'PPO', name: 'PPO (proximal policy optimisation)', method: 'ppo', world: 'catch',
      hint: 'Actor-critic with a safety catch: each step clipped so no action’s probability moves more than 20%.' },
    { type: 'rl-ga', group: 'Rewards', label: 'Evolution', name: 'Genetic algorithm', method: 'ga', world: 'pole',
      hint: 'No gradients: copies of your network play, the fittest have children with mixed, mutated weights.' },
    { type: 'rl-grow', group: 'Rewards', label: 'Grow & prune', name: 'Grow and prune (evolving the network itself)', method: 'ga', grow: true, world: 'arena',
      hint: 'Evolution that reshapes the network: children can gain or lose neurons, and gain new layers of one neuron that widen later; a size penalty keeps them small.' },
    { type: 'rl-es', group: 'Rewards', label: 'ES', name: 'Evolution strategies', method: 'es', world: 'arena',
      hint: 'One network nudged in many random directions: the nudges that did better show which way to step. A gradient made from scores alone.' },
    { type: 'rl-me', group: 'Rewards', label: 'MAP-Elites', name: 'MAP-Elites (many styles)', method: 'me', world: 'track',
      hint: 'The best network for every style of play, in a grid of 12 × 12 styles, instead of a single best.' },
    { type: 'rl-novelty', group: 'Rewards', label: 'Novelty', name: 'Novelty search', method: 'novelty', world: 'trap',
      hint: 'Evolution that ignores the reward: the networks that behave most unlike anything seen before have the children. It walks past dead ends that look like progress.' },
    { type: 'rl-cem', group: 'Rewards', label: 'Cross-entropy', name: 'Cross-entropy method', method: 'cem', world: 'lander',
      hint: 'A bell curve over the weights: draw networks from it, keep the best fifth, refit the curve to them, again and again.' },
    { type: 'rl-erl', group: 'Rewards', label: 'ERL', name: 'ERL (evolution and DQN)', method: 'erl', world: 'lander',
      hint: 'A genetic algorithm whose every move teaches a DQN; every few generations the DQN’s network joins the population.' },
    { type: 'rl-bc', group: 'Rewards', label: 'Copy you', name: 'Copy you (behaviour cloning)', method: 'bc', world: 'pole',
      hint: 'Jump in and play: it learns to make your moves, as a classifier learns its classes. Then rewards can take it further.' },
    { type: 'rl-pref', group: 'Rewards', label: 'Which did better?', name: 'Which did better? (learning from preferences)', method: 'pref', world: 'pole',
      hint: 'You say which of two tries did better; a reward model learns from your answers and PPO from the reward model: the idea behind RLHF.' },
    { type: 'rl-neat', group: 'Rewards', label: 'NEAT', name: 'NEAT (it grows its own network)', method: 'neat', world: 'pole',
      hint: 'Networks that grow their own structure from no hidden neuron: new connections and new neurons by mutation, species that protect new ideas.' },
    { type: 'rl-gauss', group: 'Rewards', label: 'PPO (numbers)', name: 'PPO with a Gaussian policy', method: 'gauss', world: 'arm',
      hint: 'For controls that are numbers (a motor’s push, the steering): a bell curve for each, its middle from the network and its spread learned; PPO’s small clipped steps.' },
    { type: 'rl-sac', group: 'Rewards', label: 'SAC', name: 'SAC (soft actor-critic)', method: 'sac', world: 'arm',
      hint: 'For numbers as controls: two critics learn what a try is worth from a replay memory, and the network makes the tries they value while keeping its curves as wide as it can.' },
    { type: 'rl-td3', group: 'Rewards', label: 'TD3', name: 'TD3 (twin delayed DDPG)', method: 'td3', world: 'arm',
      hint: 'For numbers as controls: one try per state, with noise to explore; two critics, the smaller trusted; the network learns every second step.' },
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
    sepconv: 'A convolution in two steps: a small filter slides over each channel on its own, then a 1 × 1 convolution mixes the channels. It sees as much as a full convolution with far fewer weights; dilation spreads its filter out so it sees further.',
    se: 'Averages each channel, passes the averages through a small network, and multiplies each channel by the gate it gets (0 to 1): attention over channels, decided afresh for every picture.',
    coords: 'Adds channels that hold each pixel’s column and row. A convolution looks the same everywhere, so on its own it cannot tell where it is; with these channels it can.',
    augment: 'While training, moves each picture by a pixel or two (and can mirror it), so the model never sees exactly the same picture twice. Nothing changes when the model is used.',
    gmp: 'Keeps the largest value of each channel over all positions: whether a pattern appeared anywhere, and how strongly.',
    attnpool: 'Scores every position with one learned query, turns the scores into weights with a softmax and takes the weighted average: one vector, from the positions that matter for this example.',
    ssm: 'Keeps a set of memories that are moving averages of the input with learned time scales, step by step, and reads its output from them. Linear inside the loop, so it remembers long and trains steadily.',
    rbf: 'Each unit has a centre and a width and answers how close the input is to its centre: 1 at the centre, near 0 far away. The layer covers the inputs with bumps for the next layer to combine.',
    maxout: 'Each unit computes a few weighted sums and keeps the largest. It needs no activation function: the largest of straight pieces is already a bend, and it learns which bend.',
    moe: 'Several small dense layers (experts) and a gate. For each example the gate rates the experts, and the output is their answers weighted by those ratings, often only the top one or two.',
    upsample: 'Makes a picture bigger by repeating each pixel as a small square. A decoder or a U-Net shrinks the picture to see more of it at once, then grows it back with this, so it can give an answer for every pixel.',
    sample: 'The middle of a variational autoencoder. It reads a middle and a spread for each number of the code and, while learning, draws the code from that bell curve (the middle plus the spread times a random number, so the gradient still flows). Its pull toward one standard bell curve, weighted by β, is added to the loss, so any code drawn from that curve decodes to something sensible.',
    skip: 'Adds (or joins side by side) what an earlier block gave to what reaches it: the signal can skip the blocks between, and on the way back the gradient can too. Residual connections (ResNets) add around one or two blocks; a U-Net joins its early, sharp layers to its late, coarse ones.',
    kan: 'Kolmogorov–Arnold layer: each edge from an input to a unit carries a function of its own, a weighted sum of bumps plus a weighted SiLU, all learned. A dense layer learns numbers between fixed bends; this learns the bends, one per edge.',
    hopfield: 'A memory of patterns it learns. What comes in is compared with each pattern, a softmax turns the similarities into weights, and out comes their mix: with a large β, the pattern most like the input. A modern Hopfield network, and attention whose keys and values are the memories.',
    graphconv: 'Message passing on a graph: each node averages (or adds up) its own features and its neighbours’, then one dense layer, the same for every node, makes its new features. One round hears the neighbours; two rounds, their neighbours too.',
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
    'rl-dqn': ['M3 20h18', 'M6 20v-7', 'M10 20V6', 'M14 20v-9', 'M18 20v-5', 'M8 4l2 2 2-2'],
    'rl-pg': ['M3 20h18', 'M4 20c3-1 4-9 7-9s4 8 7 9', 'M15 7l3-3', 'M15 4h3v3'],
    'rl-a2c': ['C12 5 2.2', 'M12 7.2V12', 'M12 12l-5 5', 'M12 12l5 5', 'C6 19 1.6', 'C18 19 1.6'],
    'rl-ppo': ['M3 8h18', 'M3 16h18', 'M4 18l4-10h8l4 10'],
    'rl-ga': ['C6 6 2', 'C18 6 2', 'C12 18 2.4', 'M7.2 7.8L10.8 15.8', 'M16.8 7.8L13.2 15.8', 'M3 12h2', 'M19 12h2'],
    'rl-grow': ['C5 19 1.7', 'C12 19 1.7', 'C19 19 1.7', 'C8.5 11.5 1.7', 'C15.5 11.5 1.7', 'C12 4 1.7', 'M6 17.4l1.6-4.3', 'M12 17.3v-4', 'M18 17.4l-1.6-4.3', 'M9.6 10l1.6-4.3', 'M14.4 10l-1.6-4.3'],
    'rl-es': ['C8 12 1.8', 'M8 12l-4-5', 'M8 12l-4 5', 'M8 12l3-7', 'M8 12l3 7', 'M8 12h12', 'M17 9l3 3-3 3'],
    'rl-me': ['M4 4h16v16H4z', 'M9.33 4v16', 'M14.67 4v16', 'M4 9.33h16', 'M4 14.67h16', 'C6.67 6.67 1.3', 'C12 12 1.3', 'C17.33 17.33 1.3', 'C17.33 6.67 1.3', 'C6.67 17.33 1.3'],
    sepconv: ['M3 3h7v7H3z', 'M3 14h7v7H3z', 'M10 6.5h3', 'M10 17.5h3', 'M13 6.5l4 5.5', 'M13 17.5l4-5.5', 'C19 12 2'],
    se: ['M6 3v18', 'M12 3v18', 'M18 3v18', 'M4 15h4', 'M10 8h4', 'M16 17h4'],
    coords: ['M4 3v17h17', 'M2 6l2-3 2 3', 'M18 18l3 2-3 2', 'C13 9 1.8', 'M13 11v9', 'M4 9h7'],
    augment: ['M8 8h8v8H8z', 'M12 1v4', 'M12 19v4', 'M1 12h4', 'M19 12h4', 'M10 3l2-2 2 2', 'M10 21l2 2 2-2', 'M3 10l-2 2 2 2', 'M21 10l2 2-2 2'],
    gmp: ['M3 19l4-6 3 3 4-9 3 6 4-3', 'C14 7 1.8', 'M14 3v2'],
    attnpool: ['M3 5h4', 'M3 12h4', 'M3 19h4', 'M7 5l10 7', 'M7 12h10', 'M7 19l10-7', 'C19 12 2'],
    ssm: ['M3 20h18', 'M3 4c4 0 6 14 18 15', 'M3 9c5 0 8 9 18 10', 'M3 4v16'],
    rbf: ['M2 20h20', 'M3 19c4 0 5-12 9-12s5 12 9 12', 'C12 7 1.5', 'M12 9v10'],
    maxout: ['M3 19L21 9', 'M3 9l18 10', 'M3 5l9 5 9-5'],
    'rl-neat': ['C4 6 2', 'C4 18 2', 'C12 12 2', 'C20 9 2', 'M6 6l4 5', 'M6 18l4-5', 'M14 12l4-2', 'M6 7l12 2'],
    'rl-gauss': ['M2 19c4 0 5-13 10-13s6 13 10 13', 'M2 21h20', 'M12 6v15'], 'rl-sac': ['M2 20c3 0 4-10 7-10s4 10 7 10', 'M17 4h5v5h-5z', 'M17 12h5v5h-5z'], 'rl-td3': ['M3 21h18', 'C12 8 2.5', 'M7 8h2', 'M15 8h2', 'M6 16h4v5H6z', 'M14 16h4v5h-4z'],
    'rl-bc': ['C6 6 2.6', 'M2 18c0-4 1.8-6.5 4-6.5s4 2.5 4 6.5', 'M13 9h6', 'M13 13h6', 'M13 17h6', 'M19 7l2 2-2 2'], 'rl-pref': ['M2 5h8v8H2z', 'M14 5h8v8h-8z', 'M5 18l2 2 4-4', 'M17 17l3 3', 'M20 17l-3 3'],
    skip: ['M4 20V8', 'M4 8c0-5 14-5 14 0v8', 'M15 13l3 3 3-3', 'M2 20h4'],
    kan: ['M3 19c3-9 5 1 8-6s6 3 10-8', 'C3 19 1.6', 'C21 5 1.6', 'M3 12h18'],
    hopfield: ['C6 6 2', 'C18 6 2', 'C6 18 2', 'C18 18 2', 'M8 6h8', 'M8 18h8', 'M6 8v8', 'M18 8v8', 'M8 8l8 8', 'M16 8l-8 8'],
    upsample: ['M3 15h6v6H3z', 'M11 3h10v10H11z', 'M7 13l5-5', 'M9 8h3v3'],
    sample: ['M2 19c4 0 5-12 10-12s6 12 10 12', 'M2 21h20', 'C15 11 1.6'],
    graphconv: ['C5 6 2', 'C19 6 2', 'C12 19 2', 'C12 10 1.6', 'M7 6.8l3.5 2.3', 'M17 6.8l-3.5 2.3', 'M12 11.6v5.4', 'M6 8l5 9'],
    moe: ['C5 12 2', 'M7 12l4-7', 'M7 12h4', 'M7 12l4 7', 'M12 3h8v4h-8z', 'M12 10h8v4h-8z', 'M12 17h8v4h-8z'],
    full: ['M4 9V4h5', 'M15 4h5v5', 'M20 15v5h-5', 'M9 20H4v-5'], restart: ['M6 5v14', 'M19 5l-10 7 10 7z'], back: ['M12 6l-7 6 7 6z', 'M20 6l-7 6 7 6z'], fwd: ['M12 6l7 6-7 6z', 'M4 6l7 6-7 6z'],
    play: ['M8 5l11 7-11 7z'], pause: ['M8 5h2.5v14H8z', 'M13.5 5H16v14h-2.5z'], rec: ['C12 12 5'], unfull: ['M9 4v5H4', 'M15 4v5h5', 'M20 15h-5v5', 'M4 15h5v5'],
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
  // a part of a panel you were taken to: scrolled into view and marked for a moment
  function flash(target) {
    if (!target) return;
    target.scrollIntoView({ block: 'center' });
    target.classList.remove('is-flash'); void target.offsetWidth; target.classList.add('is-flash');
    clearTimeout(target._flashT); target._flashT = setTimeout(() => target.classList.remove('is-flash'), 1900);
  }
  const BL = H.transport({
    id: 'builder', screens: { network: 'Build', data: 'Parts', insp: 'Settings' },
    labels: { play: 'Train', title: 'Train the model you built, or pause (Space)', step: 'One training step (S)', reset: 'Fresh weights: start this setup again (R)' },
    budget: 14, runs: [], undoStack: [], redoStack: [], sel: null, centerView: 'both', tabId: 'block', nextId: 1,
  });
  BL.flash = (t) => flash(t); // (Find a setting marks what it takes you to)
  const DEFAULT_TRAIN = BT.DEFAULT_TRAIN;
  BL.setup = { problem: 'spiral', ps: {}, blocks: [], loss: { id: 'xent', smooth: 0, delta: 0.2 }, train: Object.assign({}, DEFAULT_TRAIN), custom: { points: [] } };

  BL.problem = function () { return BT.byId[this.setup.problem]; };
  // what a run counts in: evolution in generations, everything else in steps
  const keepsBest = (L) => { const s = L.run && L.run.cfg ? (L.run.cfg.ps || {})[L.run.cfg.problem] || {} : {}; return s.keepBest !== false; };
  BL.unitOf = function (run) { return run && run.family === 'control' && NA.brl.isEvo(run.method) ? 'generation' : 'step'; };
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
    if (this.check.ok && this.setup.blocks.some((b) => b.type === 'skip' && b.grp)) this.check = Object.assign({}, this.check, { ok: false, problem: { reason: 'A skip connection cannot go inside one of your parts yet: it would reach outside the part. Take it out of the part.', fix: null } });
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
    this.syncTabs(); // (a problem that learns from rewards has a Reward tab; on a phone the panel may be out of sight)
    const T = this.setup.train;
    if (this.hyperRoot && this.hyperKey() !== this._hyperKey) this.buildHyper();
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
    // the maze's inputs by their names before version 29 (the exit: the goal)
    const mz = S.ps && S.ps.rlmaze, OLD = { 'the exit': 'the goal', 'which way the exit is': 'which way the goal is' };
    if (mz && Array.isArray(mz.hide)) mz.hide = mz.hide.map((h) => OLD[h] || h);
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
    for (const ps of Object.values(S.ps)) if (ps && ((ps.rFormula !== undefined && (typeof ps.rFormula !== 'string' || ps.rFormula.length > 200)) || (ps.hide !== undefined && !(Array.isArray(ps.hide) && ps.hide.length <= 8 && ps.hide.every((h) => typeof h === 'string' && h.length <= 40))))) return false;
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
  // what a press of Train runs: steps, or generations for evolution (and the number set now, if it is not a choice)
  const stepChoices = (evo, cur) => {
    const c = evo ? [[50, '50'], [100, '100'], [150, '150'], [200, '200'], [300, '300'], [500, '500'], [1000, '1,000'], [0, '∞ until paused']] : STEP_CHOICES.slice();
    if (!c.some(([v]) => v === cur)) { c.push([cur, fmtInt(cur)]); c.sort((a, b) => (a[0] || Infinity) - (b[0] || Infinity)); }
    return c.map(([value, label]) => ({ value, label }));
  };
  BL.newRun = function () {
    const d = this.dataFor(), chk = this.recheck();
    if (d.error) { NA.ui.toast.show(d.error); return false; }
    if (d.family === 'control' && d.method === 'bc') { const B = this.demosFor(d); if (!B || B.acts.length < 20) { NA.ui.toast.show('Copy you learns from your moves: press Jump in, above the picture, and play an episode or two first.'); return false; } }
    if (NA.bgen && NA.bgen.owns(d.family) && this.isClassicSetup()) { NA.ui.toast.show('A classic model cannot learn this kind of problem: use a network (a template, or parts).'); return false; }
    if (d.family === 'control' && this.isClassicSetup()) { NA.ui.toast.show('A classic model learns from examples with answers, and here there are only rewards: use a network (a template, or Dense parts).'); return false; }
    if (d.tooFew) { NA.ui.toast.show('Place at least 8 points of two or more classes first (click on the picture).'); return false; }
    if (!chk.ok) { const p = chk.problem || (chk.shapes.find((s) => s.problem) || {}).problem; NA.ui.toast.show(`The model does not fit together yet: ${p ? p.reason : 'see the red block'}`); return false; }
    const T = this.setup.train, cfg = JSON.parse(this.snapshot());
    let model;
    const classic = this.isClassicSetup();
    try { model = classic ? NA.cml.build(this.setup.blocks, d, { seed: T.seed }) : NA.bx.build(this.setup.blocks, d.input, d.output, { seed: T.seed, init: T.init }); }
    catch (e) { NA.ui.toast.show(e.message); return false; }
    const prev = this.runs[this.runs.length - 1], n = (prev ? prev.n : 0) + 1;
    if (prev && prev === this.run && !prev.net) this.keepNet(prev); // (an earlier run keeps its network, to race it)
    if (this.race) this.stopRace();
    const run = { n, colour: RUN_COLOURS[(n - 1) % RUN_COLOURS.length], cfg, changes: prev ? diffSetups(prev.cfg, cfg) : ['first run'], curve: [], step: 0, ema: null, started: performance.now(), seconds: 0, params: model.paramCount, status: 'running', show: true, family: d.family, limit: classic ? 0 : T.steps || 0, sched0: 0, events: [], classic };
    this.runs.push(run);
    if (this.pendingGuess !== undefined) { run.guess = this.pendingGuess; this.pendingGuess = undefined; }
    if (this.runs.length > 12) this.runs.splice(0, this.runs.length - 12);
    this.run = run; this.model = model; this.data = d; this.stale = false; this.step = 0;
    // (My models: a network you kept, its weights copied in; builder-shelf.js)
    if (this.loadedW && !classic && this.loadedW.w.length === model.paramCount) { model.setWeights(this.loadedW.w); run.from = 'kept'; run.changes = [`opened “${this.loadedW.name}”`]; }
    // learning from rewards: the network plays in its world, and the method turns what happens into steps
    // (Start from: another run's network, copied in before the method reads it)
    if (d.family === 'control' && this.startFrom && this.startChoices(d).some((r) => r.n === this.startFrom) && this.warmStart(model, this.startFrom)) { run.from = this.startFrom; run.changes = [...run.changes.filter((c) => c !== 'first run'), `started from run ${this.startFrom}`]; }
    this.prefShow = null; this.wantTrain = false;
    this.agent = d.family === 'control' ? NA.brl.make({ world: d.world, ws: d.ws, net: model, makeNet: () => NA.bx.build(cfg.blocks, d.input, d.output, { seed: T.seed, init: T.init }), blocks: cfg.blocks, method: d.method, seed: T.seed * 977 + n, s: Object.assign({}, d.s, { batch: T.batch }), demos: d.method === 'bc' ? this.demosFor(d) : null }) : null;
    if (this.agent) { run.method = d.method; run.bestVm = null; run.grow = !!(this.agent.evo && this.agent.evo.grow); this.demo = null; this.watchCell = null; this.genReplay = null; }
    // a GAN, or contrastive learning: a trainer of their own (builder-gen.js)
    this.gen = !this.agent && !classic && NA.bgen ? NA.bgen.trainerFor(d, model, { seed: T.seed * 977 + n, batch: T.batch }) : null;
    // grow and prune while it learns (problems with answers): a few networks side by side, reshaped every generation
    this.grower = !this.agent && !classic && T.grow && !(NA.bgen && NA.bgen.owns(d.family)) && NA.bevo.evolvable(cfg.blocks, d.input) && d.output.kind !== 'tokens'
      ? new NA.bevo.Grower({ blocks: cfg.blocks, model, input: d.input, output: d.output, seed: T.seed, init: T.init, opts: NA.bevo.growOptions(T, true), judge: (m) => this.judgeModel(m) }) : null;
    if (this.grower) run.grow = true;
    this.order = null; this.epochPos = 0; this.rng = new NA.RNG(T.seed * 977 + n);
    if (this.seqView) this.seqView.cont = null; // written by the new model once it has trained a little
    this.vizVersion = (this.vizVersion || 0) + 1; this._region = null; this.confusion = null; this.lastEval = 0; this.evalMs = 0;
    this.evalEvery = classic ? Math.max(1, Math.round(model.target / 60)) : T.steps ? Math.max(1, Math.round(T.steps / 120)) : 25; // about 120 points on the curves
    this.evaluate(true);
    run.film = this.filmFor(run); run.film.tick(true);
    this.inspDirty = true; this.resultsDirty = true; this.renderWorkbench();
    return true;
  };
  // Train: a run that finished (and has not been edited since) carries on from where it stopped; otherwise a new run.
  BL.play = function () {
    // (Guess first: a new run first asks what it will reach; builder-analyse.js)
    if (NA.banalyse && NA.banalyse.guessOn() && !this.guessGo && !this.youPlay && !this.canContinue() && (!this.run || this.stale || this.run.done)) {
      NA.banalyse.askGuess(this, (g) => { this.pendingGuess = g; this.guessGo = true; try { this.play(); } finally { this.guessGo = false; } });
      return;
    }
    if (this.youPlay) this.playYourself(false); // training takes the controls back
    if (this.race) this.stopRace();
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
  BL.onPause = function () { if (this.run && this.run.status === 'running') this.run.status = 'paused'; if (this.run) this.keepNet(this.run); if (this.run && NA.shelf) NA.shelf.keepLatest(this); this.evaluate(true); if (this.run && this.run.film && !this.stale) this.run.film.tick(true); this.inspDirty = true; };
  // ------------------------------------------------------------------ a run's film (growth.js)
  // the layers with weights, named by their blocks: their arrays as they are now
  BL.layersOf = function (model, cfg) {
    if (!model || model.classic) return null;
    const blocks = (cfg || this.run.cfg).blocks, out = [];
    if (model.embed && model.embed.params.length) out.push({ name: 'Embedding', arrays: model.embed.params.map((p) => p.w) });
    model.layers.forEach((L, i) => {
      if (!L.params || !L.params.length) return;
      const b = blocks[i], t = b ? b.type : 'output';
      out.push({ name: b ? `${i + 1}. ${t === 'rnn' ? (b.cell || 'rnn').toUpperCase() : t[0].toUpperCase() + t.slice(1)}` : 'Output', arrays: L.params.map((p) => p.w) });
    });
    return out;
  };
  /* A run's film: its results picture at moments spread on a log scale of steps, the run's score then, how big each
     layer's weights were and how far they had moved from the run's start, and the numbers of a step then */
  BL.filmFor = function (run) {
    // (a network that grows and prunes changes shape, so there is no fixed set of layers to follow)
    const model = this.model, d = this.data, agent = this.agent, gen = this.gen, layers = run.grow ? null : this.layersOf(model, run.cfg), start = layers ? NA.growth.copyLayers(layers) : null;
    return new NA.growth.Recorder({
      step: () => run.step,
      draw: (ctx, w, h) => BV.draw(ctx, { x: 4, y: 4, w: w - 8, h: h - 8 }, { data: d, model, vizVersion: -1 - run.step, _region: null, confusion: this.confusion, seqView: {}, demo: this.demo, run, gen, lossId: run.cfg.loss.id }),
      metrics: () => (run.last && run.last.vm !== null && run.last.vm !== undefined ? { [d.family === 'control' ? 'Score of its best moves' : `Validation ${this.metricName(d.family)}`]: BT.isValue(d.family) ? fmt(run.last.vm, 3) : fmtPct(run.last.vm, 1) } : {}),
      inside: () => (layers ? NA.growth.sizes(this.layersOf(model, run.cfg), start) : null),
      numbers: () => ({ loss: run.ema, val: run.last ? run.last.val : null, ret: run.last ? run.last.ret : null, grad: this.gradNorm, lr: this.lrAt(run.step - (run.sched0 || 0)), eps: agent && agent.eps ? agent.eps() : null,
        layerGrads: layers && model.gradNorms ? (() => { const g = model.gradNorms(); return model.layers.map((L, i) => (L.params && L.params.length ? g[i] : null)).filter((v) => v !== null); })() : null,
        best: agent ? agent.genBest : undefined, mean: agent ? agent.genMean : undefined }),
    });
  };
  // the curve the film is set against: the score of its best moves (rewards), or the losses
  BL.filmCurve = function () {
    const run = this.run;
    if (!run) return null;
    if (run.family === 'control' || this.data.family === 'control') return { label: 'Score of its best moves', higher: true, fmt: (v) => fmtPct(v, 0), points: run.curve.map((p) => [p.step, p.vm]) };
    if (this.model && this.model.classic) return { label: `Validation ${this.metricName(this.data.family)}`, higher: true, fmt: (v) => (BT.isValue(this.data.family) ? fmt(v, 3) : fmtPct(v, 1)), points: run.curve.map((p) => [p.step, p.vm]) };
    return { label: 'Training loss', higher: false, fmt: (v) => fmt(v, 4), points: run.curve.map((p) => [p.step, p.train]), second: { label: 'validation loss', points: run.curve.map((p) => [p.step, p.val]) } };
  };
  // the math of a moment: what it minimises (or is paid), the rule of a step, with that moment's numbers
  BL.filmMath = function (f, before) {
    const run = this.run, d = this.data, x = f.x || {}, T = run.cfg.train, G = NA.growth.math, names = (f.inside || []).map((q) => q.name);
    const moved = (q) => (q && q.inside && q.inside.length ? q.inside.reduce((s, L) => s + L.moved, 0) / q.inside.length : undefined);
    const step = G.descent({ opt: T.opt, lr: x.lr, grad: x.grad, gradWhat: 'length of the whole gradient', norm: true, decay: T.wd, layers: (x.layerGrads || []).map((g, i) => ({ name: names[i] || `Layer ${i + 1}`, grad: g })) });
    if (this.model && this.model.classic) return [{ title: 'How it learns', lines: [{ t: this.model.unit ? `It is built ${this.model.unit()} by ${this.model.unit()}, with no gradient steps` : 'It is built step by step, with no gradient steps', strong: true }], note: 'Classic models are fitted by their own recipes (splitting the data, keeping neighbours, adding trees), so there is no learning rate and no gradient here.' }];
    if (d.family === 'control') {
      const ps = run.cfg.ps[run.cfg.problem] || {}, m = d.method, E = NA.rlEnvs, vars = E.REWARD_VARS[d.world].map((v) => v[0]), own = ps.rFormula && !BT.compile(ps.rFormula, vars).error;
      const rw = E.REWARDS[d.world].filter((r) => r.key !== 'rCloser' || ps.closer).map((r) => { const v = typeof ps[r.key] === 'number' ? ps[r.key] : r.def; return `${v > 0 ? '+' : ''}${fmt(v, Math.abs(v) < 0.1 && v ? 2 : v % 1 ? 1 : 0)}  ${r.label.toLowerCase().replace(/ \(.*\)$/, '')}`; });
      const pays = { title: 'What it is paid: its reward', lines: own ? [{ t: `r = ${ps.rFormula}`, strong: true }] : rw.map((l, i) => (i ? l : { t: l, strong: true })), note: `It learns to collect as much reward as it can: ${own ? 'your formula' : 'this table'} is the goal it is given. The score measures the real goal: ${E.WORLDS[d.world].goal}.` };
      const g = ps.gamma, ev = this.agent && this.agent.evo, h = ev && ev.history.length ? ev.history[ev.history.length - 1] : null;
      const rule = m === 'dqn' ? G.dqn({ gamma: g, eps: x.eps, loss: x.loss }) : m === 'pg' ? G.pg({ gamma: g, ret: x.ret }) : m === 'a2c' ? G.a2c({ gamma: g, ret: x.ret }) : m === 'ppo' ? G.ppo({ gamma: g, ret: x.ret })
        : m === 'es' ? G.es({ pop: ps.pop, sigma: ps.esSigma, lr: ps.esLr, best: x.best, mean: x.mean }) : m === 'me' ? G.me({ batch: ps.pop, sigma: ps.sigma, filled: h && h.filled, axes: NA.rlEnvs.WORLDS[d.world].axes, best: x.best })
          : m === 'cem' && ev ? G.cem({ pop: ev.size, elite: Math.max(2, Math.round(ev.size * ev.cemElite)), spread: h && h.spread, floor: ev.cemFloor, best: x.best })
            : m === 'novelty' && ev && ev.novel ? G.novelty({ pop: ev.size, k: ev.novK, mix: ev.novMix, archive: ev.novArchive.length, best: x.best })
              : m === 'erl' && ev ? G.erl({ pop: ev.size, sigma: ev.sigma, steps: ev.erlSteps, every: ev.erlEvery, loss: ev.rlLoss, best: x.best })
                : G.ga({ pop: ps.pop, sigma: ps.sigma, best: x.best, mean: x.mean, select: ps.select, cross: ps.cross, mutate: ps.mutate, grow: run.grow, penalty: ev ? ev.penalty * ev.scale : 0, units: h && h.units, weights: h && h.weights, layerSize: ps.layerSize, changes: run.grow ? NA.bevo.describe(NA.bevo.growOptions(ps)) : null });
      return [pays, rule, NA.brl.isEvo(m) ? null : step];
    }
    if (run.grow && this.grower) {
      const gw = this.grower, h = gw.history.length ? gw.history[gw.history.length - 1] : null;
      return [G.loss(run.cfg.loss.id, x.loss, { of: 'on the training examples, smoothed' }), step, G.grow({ size: gw.members.length, every: gw.every, penalty: gw.penalty, units: h && h.units, weights: h && h.weights, score: h && h.score, value: BT.isValue(d.family), layerSize: gw.layerSize, changes: NA.bevo.describe(gw.gopt, true) })];
    }
    return [G.loss(run.cfg.loss.id, x.loss, { of: 'on the training examples, smoothed' }), step,
      before ? G.change('step', { step: before.step, v: before.x && before.x.loss, moved: moved(before) }, { step: f.step, v: x.loss, moved: moved(f) }, 'L') : null];
  };
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
      } else if (this.gen) {
        opt.lr = this.lrAt(run.step - (run.sched0 || 0));
        r = this.gen.step(opt);
      } else if (this.agent) {
        opt.lr = this.lrAt(run.step - (run.sched0 || 0));
        r = this.agent.step(opt, budget ? t0 + budget : undefined);
        // an evolutionary method judges its generation a few members at a time: no step until all have played
        if (r.partial && r.ask) { this.askPref(r.ask); break; }
        if (r.partial) { this.genProgress = r; this.resultsDirty = true; break; }
        this.genProgress = null;
        if (this.agent.net !== this.model) { this.model = this.agent.net; run.params = this.model.paramCount; this.archStale = true; this.vizVersion++; }
      } else {
        opt.lr = this.lrAt(run.step - (run.sched0 || 0));
        const [X, Y] = this.batch(B);
        r = this.grower ? this.grower.step(X, Y, B, loss, opt) : this.model.step(X, Y, B, loss, opt);
        if (this.grower && this.grower.best.model !== this.model) { this.model = this.grower.best.model; run.params = this.model.paramCount; this.archStale = true; this.vizVersion++; this._region = null; }
      }
      // evolution has no learning rate and its "loss" is a fitness, as big as the rewards make it: only no number at all stops it
      const evo = !!(this.agent && this.agent.evo);
      if (!Number.isFinite(r.loss) || (r.loss > 1e6 && !evo)) {
        this.finish('diverged');
        NA.ui.toast.show(evo ? `Run ${run.n} stopped: a fitness came out as no number at all. Some reward gave no number (a division by zero in a reward formula?).`
          : this.agent ? `Run ${run.n} blew up: the loss became ${Number.isFinite(r.loss) ? 'enormous' : 'NaN'}. Rewards that add up to very large returns are hard for the network to reach (try smaller amounts, or normalising the rewards); or try a learning rate ten times smaller.`
            : `Run ${run.n} blew up: the loss became ${Number.isFinite(r.loss) ? 'enormous' : 'NaN'}. Try a learning rate ten times smaller.`);
        break;
      }
      run.ema = run.ema === null ? r.loss : 0.97 * run.ema + 0.03 * r.loss;
      run.step++; this.step = run.step; this.gradNorm = r.gradNorm; k++;
      // a point on the curves every few steps, unless measuring would take more than a quarter of the time
      if (run.step % this.evalEvery === 0 && performance.now() - this.lastEval > 6 * this.evalMs) this.evaluate(false);
    } while (performance.now() - t0 < budget && k < cap);
    if (performance.now() - this.lastEval > (this.training ? Math.max(400, 8 * this.evalMs) : 0)) this.evaluate(false);
    if (run.film && run.step >= run.film.next) { if (performance.now() - this.lastEval > 50) this.evaluate(false); run.film.tick(); }
    return k;
  };
  BL.finish = function (status) {
    const run = this.run;
    if (!run) return;
    run.status = status; run.done = true;
    if (this.training) { this.training = false; NA.app.emit('training'); }
    this.evaluate(true);
    // Keep the best network: a run that ends below its best score ends with the weights it had then
    const A = this.agent, was = run.last ? run.last.vm : null;
    let kept = false;
    if (status === 'done' && A && A.net && run.bestW && run.bestW.length === A.net.paramCount && keepsBest(this) && was !== null && run.bestVm > was) {
      A.net.setWeights(run.bestW);
      if (A.target) A.target.setWeights(run.bestW);
      this.evaluate(true);
      run.kept = { at: run.bestAt, was }; kept = true;
      if (run.events) run.events.push({ step: run.step, what: [`kept its best network, from ${this.unitOf(run)} ${fmtInt(run.bestAt)}`] });
    }
    if (run.film) run.film.tick(true);
    this.keepNet(run);
    if (NA.banalyse) NA.banalyse.afterRun(this, status); // (several seeds: the next one)
    const gx = NA.banalyse && status === 'done' ? NA.banalyse.guessText(this, run) : '';
    if (NA.shelf && status === 'done') NA.shelf.keepLatest(this); // (the latest run, kept across reloads)
    if (kept) {
      NA.ui.toast.show(`Run ${run.n} is complete. It scored ${fmtPct(run.kept.was, 0)} at the end but ${fmtPct(run.bestVm, 0)} at ${this.unitOf(run)} ${fmtInt(run.bestAt)}, so it keeps the network it had then (Keep the best network, in the Training tab).${gx}`);
      this.inspDirty = true;
      return;
    }
    if (status === 'done') NA.ui.toast.show(this.model && this.model.classic
      ? `Run ${run.n} is complete: ${fmtInt(this.model.rounds)} ${this.model.unit()}${this.model.rounds === 1 ? '' : 's'}, ${this.metricText(run)}.${this.model.canExtend() ? ' Continue adds more.' : ''}${gx}`
      : `Run ${run.n} stopped at ${this.unitOf(run)} ${fmtInt(run.step)}: ${this.metricText(run)}.${gx || ' Press Continue to keep training it, or change the model for a new run to compare.'}`);
    this.inspDirty = true;
  };
  // validation (and a sample of training) scores, recorded on the run's curve
  BL.evaluate = function (full) {
    const run = this.run, d = this.data, m = this.model, t0 = performance.now();
    if (!run || !m) return;
    if (this.agent && d.family === 'control') { this.evaluateControl(full, t0); return; }
    if (this.gen) { this.evaluateGen(full, t0); return; }
    // fewer examples for long sequences, which cost more to run
    const per = d.input.kind === 'tokens' ? d.input.len : 1, loss = this.setup.loss;
    const nV = Math.min(d.nVa, Math.max(50, Math.round((full ? 3000 : 600) / per))), nT = Math.min(d.nTr, Math.max(50, Math.round((full ? 1500 : 300) / per)));
    const sub = (X, n) => X.subarray(0, n * (X.length / (X === d.Xtr ? d.nTr : d.nVa)));
    const ev = (X, Y, nX, nY, n) => (n ? m.evaluate(sub(X, n), Y.subarray(0, n * (Y.length / nY)), n, loss) : null);
    const v = ev(d.Xva, d.Yva, d.nVa, d.nVa, nV), t = ev(d.Xtr, d.Ytr, d.nTr, d.nTr, nT);
    const metric = (r, X, Y, n, nAll) => {
      if (!r) return null;
      const own = NA.bgen && NA.bgen.owns(d.family) ? NA.bgen.metric(d, m, X, Y, n, (run.cfg.loss || {}).id) : undefined;
      if (own !== undefined) return own;
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
  // a GAN or contrastive learning: its trainer's measure (and its own loss, the training curve)
  BL.evaluateGen = function (full, t0) {
    const run = this.run, sc = this.gen.score(full), pt = { step: run.step, train: run.ema, val: null, tm: null, vm: sc };
    const last = run.curve[run.curve.length - 1];
    if (!last || last.step !== pt.step) run.curve.push(pt); else run.curve[run.curve.length - 1] = pt;
    if (run.curve.length > 400) run.curve = run.curve.filter((q, i) => i % 2 === 0 || i === run.curve.length - 1);
    run.last = pt; run.seconds = (performance.now() - run.started) / 1000;
    this.vizVersion++; this.resultsDirty = true; this.inspDirty = true;
    this.lastEval = performance.now(); if (!full) this.evalMs = this.lastEval - t0;
    this.archStale = true;
    this.checkChallenge();
  };
  // learning from rewards: the score of its best moves (0 to 1), the average return of its episodes, the best score so far
  BL.evaluateControl = function (full, t0) {
    // a genetic algorithm: its generation's best fitness in place of the return while exploring
    const run = this.run, A = this.agent, sc = A.score(), pt = { step: run.step, train: run.ema, val: null, tm: null, vm: sc, ret: NA.brl.isEvo(A.method) ? (A.genBest === undefined ? null : A.genBest) : A.retEma };
    const last = run.curve[run.curve.length - 1];
    if (!last || last.step !== pt.step) run.curve.push(pt); else run.curve[run.curve.length - 1] = pt;
    if (run.curve.length > 400) run.curve = run.curve.filter((q, i) => i % 2 === 0 || i === run.curve.length - 1);
    if (run.bestVm === null || run.bestVm === undefined || sc > run.bestVm) {
      run.bestVm = sc; run.bestAt = run.step;
      // (the gradient methods: its weights then, to end with; evolution plays its best member already)
      if (A.net && !NA.brl.isEvo(A.method) && keepsBest(this)) run.bestW = A.net.getWeights(run.bestW && run.bestW.length === A.net.paramCount ? run.bestW : undefined);
    }
    run.last = pt; run.seconds = (performance.now() - run.started) / 1000;
    if (A.evo && A.evo.archive) run.styles = A.evo.archive.size; // MAP-Elites: the styles of play found so far
    this.vizVersion++; this.resultsDirty = true; this.inspDirty = true;
    this.lastEval = performance.now(); if (!full) this.evalMs = this.lastEval - t0;
    this.archStale = true;
    this.checkChallenge();
  };
  /* The results picture plays episodes with the network's best moves, a move every 150 ms (random moves before it
     has trained): what it sees, its outputs for it, the move they pick. A new episode starts when one ends. */
  /* The results' playback (the bar under the picture): paused or playing, the speed (the network's moves and yours
     come faster or slower), and the move looked at when you go back through the episode (null: the latest). */
  const vidOf = (L) => L.vid || (L.vid = { paused: false, speed: 1, at: null, step: 0, atT: 0 });
  BL.HIST = 1500; // moves of an episode kept to go back through, at most (the oldest go first)
  // one move of the episode, as the picture draws it
  const frameOf = (D) => ({ snap: D.snap, out: D.out, a: D.a, t: D.t, ret: D.ret, end: D.end, last: D.last, took: D.took, gave: D.gave, parts: D.parts ? JSON.parse(JSON.stringify(D.parts)) : null });
  BL.demoTick = function (now) {
    // the run's own world while it is current; once the setup has changed (another world), the setup's, as the picture draws it
    const d = this.run && !this.stale && this.data && this.data.family === 'control' ? this.data : this.dataFor();
    if (d.family !== 'control' || d.error) { this.demo = null; return; }
    // you play (Jump in): your moves, the network's outputs beside them
    const P = this.youPlay ? PLAY[d.world] : null;
    if (this.youPlay && !P) this.playYourself(false);
    // (who plays is not part of the key: you jump into the episode where it is, and hand it back the same way)
    const live = this.run && !this.stale && this.agent ? this.agent : null, key = JSON.stringify([d.world, d.ws, d.method, !!live, this.watchCell === undefined ? null : this.watchCell]);
    let D = this.demo;
    if (!D || D.key !== key) { D = this.demo = { key, world: NA.rlEnvs.make(d.world, d.ws), rng: new NA.RNG(99), ep: 0, at: 0, hist: [] }; vidOf(this).at = null; }
    // (the chaser that learns plays in the picture too, its best moves)
    D.world.brain = this.agent && this.agent.pp && !this.stale ? this.agent.pp : null;
    // a whole generation replaying: the picture moves every frame
    if (this.evoView === 'generation' && live && live.evo && live.evo.last && !P) this.resultsDirty = true;
    const W = NA.rlEnvs.WORLDS[d.world] || {}, V = vidOf(this);
    // a recording ends a moment after its episode does (or after a minute)
    if (this.rec && ((D.ep === this.rec.ep && D.end && now - D.at > 900) || D.ep > this.rec.ep || now - this.rec.t0 > 60000)) this.recordVideo(false);
    // going back through the episode: playing on shows the moves kept, at the speed set, then carries on from the latest
    if (V.at !== null) {
      if (V.paused || now - V.atT < (W.demoMs || 150) / V.speed) return;
      V.atT = now;
      if (V.at < D.hist.length - 1) { V.at++; this.resultsDirty = true; return; }
      V.at = null;
    }
    if (V.paused && !V.step) return;
    // your moves come at a person's pace and the network's at its own, both scaled by the speed; a world that waits for
    // each move waits for your key
    if (!V.step && now - D.at < (D.end ? 1000 : (P ? P.ms || 0 : W.demoMs || 150) / V.speed)) return;
    if (P && P.turn && D.obs && !D.end && !this.playQueue.length) { V.step = 0; return; }
    V.step = 0;
    D.at = now;
    if (!D.obs || D.end) { D.obs = D.world.reset(D.rng); D.ep++; D.t = 0; D.ret = 0; D.end = null; D.last = null; D.parts = null; D.took = null; D.gave = null; D.hist = []; }
    else {
      const a = P ? (P.turn ? this.playQueue.shift() : heldMove(P, this.playHeld)) : D.a;
      // (your moves are kept, with what the network saw, for Copy you)
      if (P && a !== null && a !== undefined) this.keepMove(d, D.obs, a);
      const t = D.world.step(a, D.rng);
      D.t++; D.ret += t.r; D.obs = t.obs; D.last = { a, r: t.r, parts: t.parts || null };
      // where the return came from: each term of the reward table summed over the episode, and how often it paid (builder-depth.js draws it)
      if (t.parts) for (const k in t.parts) { const q = (D.parts || (D.parts = {}))[k] || (D.parts[k] = { sum: 0, n: 0 }); q.sum += t.parts[k]; q.n++; }
      if (t.done) { D.end = d.world === 'catch' ? (D.world.caught ? 'caught' : 'missed') : d.world === 'pole' ? (t.fell ? 'fell' : 'up') : d.world === 'levers' ? 'pulled' : D.world.end || 'over'; if (P) this.keepMove(d, null, null, true); }
    }
    D.snap = D.world.snap();
    // (an episode of one move, the levers: its outputs and its pull stay up while the result shows)
    const keep = D.end && D.world.maxSteps === 1 && !P;
    D.out = live && !D.end ? live.outputs(D.obs, this.watchNet(live)) : keep ? D.out : null;
    D.a = D.end ? (keep ? D.a : null) : P ? (P.turn ? null : heldMove(P, this.playHeld)) : D.out ? NA.brl.moveOf(D.out, d) : D.rng.int(d.nA);
    // (the pole's arrow: the push it is about to make, the move its bars pick, not the one it made a step ago)
    if (d.world === 'pole' && D.snap && D.a !== null && D.a !== undefined) D.snap.a = D.a;
    D.hist.push(frameOf(D));
    if (D.hist.length > BL.HIST) D.hist.splice(0, D.hist.length - BL.HIST);
    this.resultsDirty = true;
  };
  /* Your moves while you play (Jump in, or racing a network), each with what the network would have seen before it,
     kept by world and inputs for Copy you (behaviour cloning); an episode of yours that ended is counted. */
  const demoKey = (d) => JSON.stringify([d.world, d.input, d.obsNames || null, d.nA]);
  BL.demosFor = function (d) { return (this.demos || {})[demoKey(d)] || null; };
  BL.keepMove = function (d, obs, a, ended) {
    const all = this.demos || (this.demos = {}), k = demoKey(d), B = all[k] || (all[k] = { world: d.world, nA: d.nA, obs: [], acts: [], eps: 0 });
    if (ended) { B.eps++; this._headKey = null; this.inspDirty = true; return; }
    if (B.acts.length >= 20000 || !obs) return;
    B.obs.push(Float32Array.from(obs)); B.acts.push(a);
    if (B.acts.length % 25 === 1) { this._headKey = null; this.inspDirty = true; }
  };
  BL.clearDemos = function (d) { if (this.demos) delete this.demos[demoKey(d)]; this._headKey = null; this.inspDirty = true; };
  // the playback bar's buttons: pause or play, a move back or on, a new episode, the speed, a place in the episode
  BL.vidAct = function (cmd, v) {
    const V = vidOf(this), D = this.demo, last = D && D.hist ? D.hist.length - 1 : -1;
    if (cmd === 'toggle') { V.paused = !V.paused; if (!V.paused) V.atT = 0; }
    else if (cmd === 'back' && last > 0) { V.paused = true; V.at = Math.max(0, (V.at === null ? last : V.at) - 1); }
    else if (cmd === 'fwd') { V.paused = true; if (V.at !== null) { V.at++; if (V.at >= last) V.at = null; } else V.step = 1; }
    else if (cmd === 'restart' && D) { V.at = null; if (D.obs) D.end = D.end || 'over'; D.obs = null; D.at = 0; V.step = 1; if (this.race) this.raceEpisode(); }
    else if (cmd === 'speed') V.speed = +v || 1;
    else if (cmd === 'seek' && last >= 0) { V.paused = true; V.at = +v >= last ? null : Math.max(0, +v); }
    this.resultsDirty = true; this.syncVid();
  };
  const VID_SPEEDS = [0.25, 0.5, 1, 2, 4];
  const REC_TYPES = ['video/webm;codecs=vp9', 'video/webm;codecs=vp8', 'video/webm', 'video/mp4'];
  // the bar under the picture: playback, the place in the episode, the speed, recording, racing
  BL.vidBar = function (d) {
    const V = vidOf(this), btn = (ico, label, cmd) => el('button', { class: 'icon-btn bd-vid-btn', type: 'button', title: label, 'aria-label': label, dataset: { cmd }, onclick: () => this.vidAct(cmd) }, icon(ico));
    const seek = el('input', { type: 'range', class: 'bd-vid-seek', min: 0, max: 0, value: 0, step: 1, 'aria-label': 'The move of the episode shown' });
    seek.addEventListener('input', () => this.vidAct('seek', seek.value));
    const where = el('span', { class: 'bd-vid-at mono' });
    const speed = UI.select(VID_SPEEDS.map((v) => ({ value: v, label: `${v}×` })), V.speed, (v) => this.vidAct('speed', v), { class: 'bd-vid-speed', 'aria-label': 'Speed', title: 'Speed: the moves of the network (and yours) come this much faster or slower' });
    const canRec = !!(window.MediaRecorder && this.resCanvas.captureStream && REC_TYPES.some((t) => MediaRecorder.isTypeSupported(t)));
    const rec = el('button', { class: 'btn bd-vid-rec' + (this.rec ? ' is-on' : ''), type: 'button', disabled: !canRec, title: canRec ? (this.rec ? 'Stop and save the video now' : 'Record a whole episode as a video, from its start to its end, and save it') : 'This browser cannot record the picture', onclick: () => this.recordVideo(!this.rec) }, icon('rec'), this.rec ? 'Stop and save' : 'Record a video');
    const opts = this.raceChoices(d);
    const race = opts.length || this.race ? UI.select([{ value: '', label: this.race ? 'Stop the race' : 'Race…' }, ...opts], '', (v) => { if (v === '') this.stopRace(); else this.startRace(v); }, { class: 'bd-vid-race', 'aria-label': 'Race two side by side', title: 'Race: two play the same world side by side, from the same start every episode' }) : null;
    const bar = el('div', { class: 'bd-vid', role: 'group', 'aria-label': 'Playback' },
      el('span', { class: 'bd-vid-btns' }, btn('restart', 'New episode', 'restart'), btn('back', 'One move back', 'back'), el('button', { class: 'icon-btn bd-vid-btn bd-vid-play', type: 'button', title: 'Pause or play (K)', 'aria-label': 'Pause', dataset: { cmd: 'toggle' }, onclick: () => this.vidAct('toggle') }, icon('pause')), btn('fwd', 'One move on', 'fwd')),
      seek, where, el('span', { class: 'bd-vid-more' }, speed, rec, race));
    this.vidUI = { bar, seek, where, play: bar.querySelector('.bd-vid-play') };
    this.syncVid();
    return bar;
  };
  // the bar kept up with the picture: the place in the episode, pause or play
  BL.syncVid = function () {
    const U = this.vidUI, D = this.demo, V = vidOf(this);
    if (!U || !U.bar.isConnected) return;
    const last = D && D.hist ? Math.max(0, D.hist.length - 1) : 0, at = V.at === null ? last : V.at, R = this.race;
    U.seek.max = String(last); U.seek.value = String(at); U.seek.disabled = !!R;
    U.where.textContent = R ? `episode ${R.ep}` : D && D.hist && D.hist.length ? `move ${fmtInt((D.hist[at] || {}).t || 0)}${V.at !== null ? ` of ${fmtInt((D.hist[last] || {}).t || 0)}` : ''}` : '';
    const paused = !!V.paused;
    if (U.play.dataset.p !== String(paused)) { U.play.dataset.p = String(paused); U.play.replaceChildren(icon(paused ? 'play' : 'pause')); U.play.setAttribute('aria-label', paused ? 'Play' : 'Pause'); }
  };
  /* Record a video: a new episode starts and the picture is recorded (the browser's MediaRecorder, as WebM or MP4)
     until a moment after it ends, or Stop and save; then it is saved as a file. */
  BL.recordVideo = function (on) {
    if (!on) { if (this.rec && this.rec.r.state !== 'inactive') this.rec.r.stop(); return; }
    if (this.rec) return;
    const type = window.MediaRecorder && REC_TYPES.find((t) => MediaRecorder.isTypeSupported(t));
    if (!type || !this.resCanvas.captureStream) { NA.ui.toast.show('This browser cannot record the picture.'); return; }
    const stream = this.resCanvas.captureStream(30), r = new MediaRecorder(stream, { mimeType: type, videoBitsPerSecond: 5e6 }), chunks = [], d = this.dataFor(), n = this.run ? this.run.n : 0;
    r.ondataavailable = (e) => { if (e.data && e.data.size) chunks.push(e.data); };
    r.onstop = () => {
      stream.getTracks().forEach((t) => t.stop());
      this.rec = null; this._headKey = null; this.resultsDirty = true;
      const blob = new Blob(chunks, { type: type.split(';')[0] });
      if (!blob.size) { NA.ui.toast.show('Nothing was recorded.'); return; }
      const name = `neural-anatomy-${d.world}${n ? `-run${n}` : ''}.${/mp4/.test(type) ? 'mp4' : 'webm'}`;
      this.saveBlob(blob, name);
      NA.ui.toast.show(`Saved ${name} (${(blob.size / 1e6).toFixed(1)} MB).`);
    };
    const V = vidOf(this);
    V.paused = false; V.at = null;
    this.vidAct('restart');
    this.rec = { r, ep: (this.demo ? this.demo.ep : 0) + 1, t0: performance.now() };
    r.start(250);
    this._headKey = null; this.resultsDirty = true;
    NA.ui.toast.show('Recording a whole episode: it saves itself when the episode ends (or press Stop and save).');
  };
  // a file made in the page, saved (tests take it here instead)
  BL.saveBlob = function (blob, name) {
    const url = URL.createObjectURL(blob), a = el('a', { href: url, download: name });
    document.body.append(a); a.click(); a.remove();
    setTimeout(() => URL.revokeObjectURL(url), 4000);
  };

  /* Race: two play the same world side by side, from the same start every episode (each world with its own random
     numbers from the same seed, so they stay alike while their moves do), each by its best moves; or you against a
     network. Who can race: this run and earlier runs on this world whose networks see the same inputs and make the same
     moves (each run keeps its network: keepNet), and you. */
  BL.keepNet = function (run) {
    if (!run || run.family !== 'control' || !this.agent || this.run !== run || !this.agent.net || this.agent.neat) return; // (NEAT's network is a genome of its own)
    const net = this.agent.net;
    run.net = { blocks: JSON.parse(JSON.stringify(net.evoBlocks || run.cfg.blocks)), w: net.getWeights(), input: this.data.input, output: this.data.output, nA: this.data.nA, world: this.data.world, method: run.method };
  };
  const sameIO = (a, d) => !!a && a.world === d.world && a.nA === d.nA && JSON.stringify(a.input) === JSON.stringify(d.input);
  BL.raceChoices = function (d) {
    if (!d || d.family !== 'control' || d.error) return [];
    const runs = this.runs.filter((r) => r.family === 'control' && (r === this.run ? !this.stale && this.agent && this.data.world === d.world : sameIO(r.net, d))).slice().reverse();
    const name = (r) => `run ${r.n} (${NA.brl.METHODS[r.method] ? NA.brl.METHODS[r.method].short : r.method}${r.bestVm !== null && r.bestVm !== undefined ? `, best ${fmtPct(r.bestVm, 0)}` : ''})`, out = [];
    if (PLAY[d.world]) for (const r of runs.slice(0, 3)) out.push({ value: 'you:' + r.n, label: `You against ${name(r)}` });
    for (let i = 0; i < runs.length; i++) for (let j = i + 1; j < runs.length && out.length < 12; j++) out.push({ value: `run:${runs[i].n}:${runs[j].n}`, label: `${name(runs[i]).replace(/^r/, 'R')} against ${name(runs[j])}` });
    return out;
  };
  // a run's network, ready to play (this run's own, or one kept from an earlier run)
  BL.netOfRun = function (n) {
    const run = this.runs.find((r) => r.n === n);
    if (!run) return null;
    if (run === this.run && this.agent && this.agent.net && !this.stale) return { net: this.agent.net, run };
    if (!run.net) return null;
    const net = NA.bx.build(run.net.blocks, run.net.input, run.net.output, { seed: 1 });
    net.setWeights(run.net.w);
    return { net, run };
  };
  BL.startRace = function (choice) {
    const d = this.dataFor(), parts = String(choice).split(':');
    if (d.family !== 'control' || d.error) return;
    if (this.training) this.pause();
    if (this.youPlay) this.playYourself(false);
    const side = (who) => {
      if (who === 'you') return { you: true, label: 'You' };
      const g = this.netOfRun(+who);
      return g ? { net: g.net, label: `Run ${g.run.n}`, method: g.run.method } : null;
    };
    const A = side(parts[0] === 'you' ? 'you' : parts[1]), B = side(parts[0] === 'you' ? parts[1] : parts[2]);
    if (!A || !B) { NA.ui.toast.show('That run’s network is no longer kept.'); return; }
    this.race = { d, sides: [A, B], ep: 0, wins: [0, 0], at: 0, done: 0 };
    for (const S of this.race.sides) S.world = NA.rlEnvs.make(d.world, d.ws);
    this.raceEpisode();
    if (A.you) { this.playHeld = []; this.playQueue = []; this.listenPlay(true); this.raceYou = true; }
    this._headKey = null; this.resultsDirty = true;
    NA.ui.toast.show(A.you ? `Your race against ${B.label}: ${PLAY[d.world].help}. The same start for both, every episode.` : `${A.label} against ${B.label}: the same world and the same start every episode.`);
  };
  BL.stopRace = function () {
    if (!this.race) return;
    if (this.raceYou && !this.youPlay) this.listenPlay(false);
    this.race = null; this.raceYou = false; this._headKey = null; this.resultsDirty = true;
  };
  // a new episode of the race: both worlds from the same seed
  BL.raceEpisode = function () {
    const R = this.race;
    R.ep++; R.done = 0; R.at = 0; R.winner = null;
    for (const S of R.sides) { S.rng = new NA.RNG(7000 + R.ep); S.obs = S.world.reset(S.rng); S.ret = 0; S.t = 0; S.end = null; S.snap = S.world.snap(); S.out = S.net ? Float64Array.from(S.net.forward(Float32Array.from(S.obs), 1, false).subarray(0, R.d.nA)) : null; S.a = null; }
  };
  BL.raceTick = function (now) {
    const R = this.race, d = R.d, V = vidOf(this), W = NA.rlEnvs.WORLDS[d.world] || {}, P = PLAY[d.world], you = R.sides.find((S) => S.you);
    // (the world or what it sees changed since the race began: the race is over)
    const cur = this.dataFor();
    if (cur.family !== 'control' || cur.world !== d.world || JSON.stringify(cur.ws) !== JSON.stringify(d.ws) || JSON.stringify(cur.input) !== JSON.stringify(d.input)) { this.stopRace(); return; }
    if (V.paused && !V.step) return;
    const gap = R.done ? 1600 : ((you ? (P && P.ms) || 90 : W.demoMs || 150) / V.speed);
    if (!V.step && now - R.at < gap) return;
    // (a world of single moves waits for your key, and the network moves when you do)
    if (you && P && P.turn && !R.done && !this.playQueue.length && !V.step) return;
    V.step = 0; R.at = now;
    if (R.done) { this.raceEpisode(); this.resultsDirty = true; return; }
    for (const S of R.sides) {
      if (S.end) continue;
      const a = S.you ? (P.turn ? this.playQueue.shift() : heldMove(P, this.playHeld)) : NA.brl.moveOf(S.out, d);
      if (S.you && a !== null && a !== undefined) this.keepMove(d, S.obs, a);
      const t = S.world.step(a, S.rng);
      S.t++; S.ret += t.r; S.obs = t.obs; S.a = a;
      if (t.done) { S.end = d.world === 'catch' ? (S.world.caught ? 'caught' : 'missed') : d.world === 'pole' ? (t.fell ? 'fell' : 'up') : d.world === 'levers' ? 'pulled' : S.world.end || 'over'; if (S.you) this.keepMove(d, null, null, true); }
      S.snap = S.world.snap();
      S.out = S.net && !S.end ? Float64Array.from(S.net.forward(Float32Array.from(S.obs), 1, false).subarray(0, d.nA)) : null;
      if (d.world === 'pole' && S.snap && !S.end && S.out) S.snap.a = NA.brl.moveOf(S.out, d);
    }
    if (R.sides.every((S) => S.end)) {
      R.done = 1;
      const [a, b] = R.sides;
      R.winner = a.ret > b.ret + 1e-9 ? 0 : b.ret > a.ret + 1e-9 ? 1 : -1;
      if (R.winner >= 0) R.wins[R.winner]++;
    }
    this.resultsDirty = true;
  };
  /* Which did better?: a pair of tries waiting for your answer (the trainer stops until it has one). The results show
     both tries side by side, replayed from their pictures, and the line under the title asks; ← → answer, ↓ says
     about the same. Training goes on after the answer if it was going. */
  BL.askPref = function (pair) {
    this.prefShow = { pair, at: 0, t: 0 };
    this.wantTrain = this.wantTrain || this.training;
    if (this.training) { this.training = false; if (this.run) this.run.status = 'paused'; NA.app.emit('training'); }
    if (NA.mobile && NA.mobile.mobile) NA.mobile.go('network');
    if (this.centerView === 'model' || this.centerView === 'arch' || this.centerView === 'growth') { this.centerView = 'results'; this.layoutCenter(); }
    this._headKey = null; this.resultsDirty = true;
  };
  BL.answerPref = function (y) {
    const A = this.agent;
    if (!A || !A.pending || !this.prefShow) return;
    A.answer(y);
    this.prefShow = null; this._headKey = null; this.resultsDirty = true; this.inspDirty = true;
    const go = this.wantTrain;
    this.wantTrain = false;
    if (go && this.run && !this.run.done) { this.training = true; this.run.status = 'running'; NA.app.emit('training'); }
  };
  // the pair replayed side by side: a move every so often (at the playback's speed), again from the start a moment after both end
  BL.prefTick = function (now) {
    const S = this.prefShow, V = vidOf(this), W = NA.rlEnvs.WORLDS[this.dataFor().world] || {};
    if (!S || V.paused || now - S.t < (W.demoMs || 150) / V.speed) return;
    S.t = now;
    const n = Math.max(S.pair.A.snaps.length, S.pair.B.snaps.length);
    S.at = S.at + 1 >= n + 8 ? 0 : S.at + 1;
    this.resultsDirty = true;
  };
  // the question, under the results' title
  BL.prefRow = function () {
    const A = this.agent, st = A && A.prefStats ? A.prefStats() : null;
    const b = (label, y, title) => el('button', { class: y === 1 || y === 0 ? 'btn btn-primary' : 'btn', type: 'button', title, onclick: () => this.answerPref(y) }, label);
    return el('div', { class: 'bd-pref' },
      el('p', { class: 'bd-pref-q' }, el('strong', { text: 'Which did better? ' }), 'Two tries from the same start, side by side; the second makes a quarter of its moves at random. Watch them, then answer.'),
      el('div', { class: 'btn-row' }, b('◀ The first', 1, 'The first try did better (←)'), b('About the same', 0.5, 'Neither did better (↓)'), b('The second ▶', 0, 'The second try did better (→)'), b('Skip', null, 'No answer: show another pair')),
      st && st.answered ? UI.note(`Answered ${st.answered}.${st.rm !== null ? ` The reward model agrees with ${fmtPct(st.rm, 0)} of your answers` : ''}${st.world !== null ? `; your answers agree with the world’s own reward on ${fmtPct(st.world, 0)}` : ''}.`) : UI.note('Your first few answers come one after another; then it learns, and asks again every so often.'));
  };
  /* Start from (the Training tab): a run begins from another run's network instead of fresh weights, when they have the
     same layers (the outputs may differ: Copy you's move outputs go to PPO's, whose value output starts fresh). Which
     runs can: those that kept a network (keepNet) for this world with the same inputs and blocks. */
  const sameBlocks = (a, b) => JSON.stringify(a.map(cleanSpec)) === JSON.stringify(b.map(cleanSpec));
  BL.startChoices = function (d) {
    return this.runs.filter((r) => r.family === 'control' && r.net && r.net.world === d.world && r.net.nA === d.nA && JSON.stringify(r.net.input) === JSON.stringify(d.input) && sameBlocks(r.net.blocks, this.setup.blocks)).slice().reverse();
  };
  BL.warmStart = function (model, n) {
    const g = this.netOfRun(n);
    if (!g) return false;
    const A = g.net.params, B = model.params;
    if (A.length !== B.length) return false;
    for (let i = 0; i < A.length - 2; i++) if (A[i].w.length !== B[i].w.length) return false;
    const [Wa, ba] = A.slice(-2), [Wb, bb] = B.slice(-2), Ka = ba.w.length, Kb = bb.w.length, I = Wa.w.length / Ka;
    if (Wb.w.length / Kb !== I) return false;
    for (let i = 0; i < A.length - 2; i++) B[i].w.set(A[i].w);
    // the output layer, row by row for the outputs both have
    const K = Math.min(Ka, Kb);
    Wb.w.set(Wa.w.subarray(0, K * I)); bb.w.set(ba.w.subarray(0, K));
    return true;
  };
  // the Training tab's sections for Copy you, Which did better? and Start from
  BL.teachSecs = function (d) {
    const s = this.ps(), m = s.method, set = (fn) => this.edit((S) => fn(S.ps[S.problem])), out = [];
    if (m === 'bc') {
      const B = this.demosFor(d), n = B ? B.acts.length : 0, eps = B ? B.eps : 0;
      out.push(UI.section('Your moves', UI.note(n ? `You have shown it ${fmtInt(n)} move${n === 1 ? '' : 's'}${eps ? ` in ${fmtInt(eps)} finished episode${eps === 1 ? '' : 's'}` : ''} in this world, as it sees it now. Each step of a run teaches it to pick your move in one of those moments.` : 'None yet. Press Jump in, above the picture, and play: every move you make is kept, with what the network would have seen. A few episodes are enough to start; the better and the more varied your play, the better the copy.'),
        el('div', { class: 'btn-row' }, el('button', { class: 'btn btn-primary', type: 'button', onclick: () => { if (NA.mobile && NA.mobile.mobile) NA.mobile.go('network'); if (this.centerView === 'model' || this.centerView === 'arch' || this.centerView === 'growth') { this.centerView = 'results'; this.layoutCenter(); } this.playYourself(true); } }, n ? 'Jump in and show it more' : 'Jump in and play'),
          n ? el('button', { class: 'btn btn-quiet', type: 'button', onclick: () => this.clearDemos(d) }, 'Forget your moves') : null),
        UI.note('Your moves stay in this tab of the browser and go nowhere else; they are forgotten when the page is closed. The Python tab can download them, for its script.')));
    }
    if (m === 'pref') {
      out.push(UI.section('Which did better?', UI.field('Who answers', UI.seg([{ value: 'you', label: 'You' }, { value: 'world', label: 'The world’s reward' }], s.prefAsk || 'you', (v) => set((q) => { q.prefAsk = v; }), 'Who answers')),
        UI.field('Ask every', UI.seg([100, 300, 600].map((v) => ({ value: v, label: `${v} steps` })), s.prefEvery || 300, (v) => set((q) => { q.prefEvery = +v; }), 'Ask every')),
        UI.note(`${(s.prefAsk || 'you') === 'you' ? 'You are shown two tries and asked which did better; training waits for your answer. The pole learns from about a dozen answers; most worlds need many more, the way chatbots need many people answering.' : 'The world’s own reward answers at once (the try with the larger return did better), five times as often, as in the first experiments with this method (Christiano and others, 2017): it shows how much a reward model can learn from comparisons alone.'} The reward model never sees the world’s reward, only which try won; the score still measures the world’s own goal.`)));
    }
    // Start from: fresh weights, or an earlier run's network
    const runs = this.startChoices(d);
    if (this.startFrom && !runs.some((r) => r.n === this.startFrom)) this.startFrom = null;
    if (runs.length) {
      const name = (r) => `Run ${r.n} (${NA.brl.METHODS[r.method] ? NA.brl.METHODS[r.method].short : r.method}${r.bestVm !== null && r.bestVm !== undefined ? `, best ${fmtPct(r.bestVm, 0)}` : ''})`;
      out.push(UI.section('Start from', UI.field('Weights', UI.select([{ value: '', label: 'Fresh weights' }, ...runs.map((r) => ({ value: r.n, label: name(r) }))], this.startFrom || '', (v) => { this.startFrom = v ? +v : null; this._trainKey = null; this.inspDirty = true; }), 'Start from'),
        UI.note(this.startFrom ? `The next run starts from run ${this.startFrom}’s network, its weights and all, and learns on from there (fine-tuning). A run that copied you, then PPO: rewards take it past what you showed it. Outputs it did not have (PPO’s value) start fresh.` : 'A run can start from an earlier run’s network with the same layers, instead of fresh weights: Copy you first, then PPO from it, and rewards take it past what you showed it.')));
    }
    return out;
  };
  /* The network the results play: the one on show, or with MAP-Elites the elite of the cell you picked (a copy of
     the network with that elite's weights). */
  BL.watchNet = function (live) {
    const ev = live && live.evo;
    if (!ev || !ev.archive || this.watchCell === null || this.watchCell === undefined) return live.net;
    const e = ev.elite(this.watchCell);
    if (!e) return live.net;
    if (!this._eliteNet || this._eliteNet.paramCount !== live.net.paramCount) this._eliteNet = live.makeNet();
    if (this._eliteFor !== e) { this._eliteNet.setWeights(e.w); this._eliteFor = e; }
    return this._eliteNet;
  };
  /* Try it yourself: each world's keys (and buttons, for a touch screen or a mouse), the move when nothing is held,
     and how often a move comes (slower than the network's, for a person); a world of single moves (the maze, the
     levers) waits for each press. The arena has no move for standing still: with nothing held it fires whenever the
     gun is ready. */
  const PLAY = {
    track: { idle: 1, ms: 70, keys: { ArrowLeft: 0, ArrowRight: 2, ArrowDown: 3 }, help: '← → steer, ↓ brakes', pad: [[0, '◀ Steer'], [3, 'Brake'], [2, 'Steer ▶']] },
    arena: { idle: 3, ms: 90, keys: { ArrowLeft: 0, ArrowRight: 1, ArrowUp: 2, ' ': 3 }, help: '← → turn, ↑ walks; it fires whenever no key is held and the gun is ready', pad: [[0, '◀ Turn'], [2, 'Walk'], [1, 'Turn ▶']] },
    lander: { idle: 0, ms: 70, keys: { ArrowUp: 1, ArrowLeft: 2, ArrowRight: 3 }, help: '↑ fires the main engine, ← → the side thrusters', pad: [[2, '◀ Turn'], [1, 'Engine'], [3, 'Turn ▶']] },
    pole: { idle: 2, ms: 45, keys: { ArrowLeft: 0, ArrowRight: 1 }, help: '← → push the cart', pad: [[0, '◀ Push'], [1, 'Push ▶']] },
    catch: { idle: 1, ms: 380, keys: { ArrowLeft: 0, ArrowRight: 2 }, help: '← → move the paddle', pad: [[0, '◀'], [2, '▶']] },
    maze: { turn: true, keys: { ArrowUp: 0, ArrowRight: 1, ArrowDown: 2, ArrowLeft: 3 }, help: 'the arrow keys move a square a press', pad: [[3, '◀'], [0, '▲'], [2, '▼'], [1, '▶']] },
    levers: { turn: true, keys: { 1: 0, 2: 1, 3: 2, 4: 3, 5: 4 }, help: 'the keys 1 to 5 pull a lever' },
    thermo: { idle: 0, ms: 120, keys: { ArrowUp: 1, ' ': 1 }, help: 'hold ↑ (or the space bar) to keep the heating on', pad: [[1, 'Heating on']] },
    hurdles: { idle: 0, ms: 70, keys: { ArrowUp: 1, ' ': 1, ArrowDown: 2 }, help: '↑ (or the space bar) jumps, ↓ ducks', pad: [[1, 'Jump'], [2, 'Duck']] },
    memory: { turn: true, keys: { ArrowUp: 0, ArrowLeft: 1, ArrowRight: 2 }, help: '↑ walks on, ← → turn at the junction', pad: [[1, '◀ Left'], [0, '▲ On'], [2, 'Right ▶']] },
    mountain: { idle: 1, ms: 40, keys: { ArrowLeft: 0, ArrowRight: 2 }, help: '← → push the cart', pad: [[0, '◀ Push'], [2, 'Push ▶']] },
    swingup: { idle: 1, ms: 60, keys: { ArrowLeft: 0, ArrowRight: 2 }, help: '← → turn the motor', pad: [[0, '◀ Turn'], [2, 'Turn ▶']] },
    arm: { idle: 4, ms: 60, keys: { ArrowLeft: 0, ArrowRight: 1, ArrowUp: 2, ArrowDown: 3 }, help: '← → turn the shoulder, ↑ ↓ the elbow; with nothing held it brakes', pad: [[0, '↺ Shoulder'], [1, 'Shoulder ↻'], [2, '↺ Elbow'], [3, 'Elbow ↻']] },
    traffic: { idle: 0, ms: 150, keys: { ' ': 1, ArrowUp: 1 }, help: 'the space bar (or ↑) switches the lights', pad: [[1, 'Switch']] },
    walker: { idle: 3, ms: 70, keys: { ArrowLeft: 0, ArrowRight: 1, ArrowDown: 2 }, help: '← → swing the legs, ↓ crouches; with nothing held both muscles go long', pad: [[0, '◀ Swing'], [2, 'Crouch'], [1, 'Swing ▶']] },
    trap: { idle: 0, ms: 50, keys: { ArrowLeft: 1, ArrowRight: 2 }, help: '← → turn; with nothing held it drives forward', pad: [[1, '◀ Turn'], [2, 'Turn ▶']] },
    chase: { idle: 4, ms: 60, keys: { ArrowUp: 0, ArrowDown: 1, ArrowLeft: 2, ArrowRight: 3 }, help: 'the arrow keys run', pad: [[2, '◀'], [0, '▲'], [1, '▼'], [3, '▶']] },
    herd: { idle: 4, ms: 50, keys: { ArrowUp: 0, ArrowDown: 1, ArrowLeft: 2, ArrowRight: 3 }, help: 'the arrow keys move the dog', pad: [[2, '◀'], [0, '▲'], [1, '▼'], [3, '▶']] },
  };
  const heldMove = (P, held) => (held && held.length ? held[held.length - 1] : P.idle); // the key pressed last wins
  /* Jump in: you take the controls of the episode on show where it is (the move it has reached, its return so far);
     Let it play hands them back the same way, and the network carries on from where you left it. Between episodes, a
     new one starts. */
  BL.playYourself = function (on) {
    const d = this.dataFor(), P = PLAY[d.world];
    if (on && (d.family !== 'control' || !P)) return;
    if (!!on === !!this.youPlay) return;
    if (on && this.training) this.pause();
    this.youPlay = !!on; this.playHeld = []; this.playQueue = [];
    const D = this.demo;
    if (D && !D.end && D.t) { if (on) D.took = D.t; else D.gave = D.t; } // (an episode that has ended: the next one starts as usual)
    if (on) this.evoView = 'best';
    this.listenPlay(this.youPlay);
    this._headKey = null; this.resultsDirty = true;
    const touchy = (NA.mobile && NA.mobile.mobile) || (window.matchMedia && matchMedia('(pointer: coarse)').matches);
    // (in full screen the line above the picture says it already)
    if (on && !this.full) NA.ui.toast.show(`Your turn${D && D.took ? `, from move ${fmtInt(D.took)}` : ''}: ${touchy ? `press and hold the buttons ${this.full ? 'under' : 'above'} the picture` : P.help}.${this.run && !this.stale ? ' The bars show what the network would do.' : ''}`);
  };
  // the keys of a world, for the hint (the arrows it uses, the number keys)
  const keysOf = (P) => { const A = { ArrowLeft: '←', ArrowRight: '→', ArrowUp: '↑', ArrowDown: '↓' }, k = Object.keys(P.keys); return [...k.filter((x) => A[x]).map((x) => A[x]), ...(k.some((x) => /^\d$/.test(x)) ? ['1–' + k.filter((x) => /^\d$/.test(x)).length] : [])].join(' '); };
  /* While the network plays, a key of the world's jumps in at once, the key itself the first move: in full screen, or
     with the picture (or nothing) focused. Not the space bar: it trains. */
  BL.listenJump = function (on) {
    if (this._jumpKey) { document.removeEventListener('keydown', this._jumpKey, true); this._jumpKey = null; }
    if (!on) return;
    this._jumpKey = (e) => {
      if (this.youPlay || this.prefShow || e.ctrlKey || e.metaKey || e.altKey || e.key === ' ' || NA.labHost.lab !== this || this.family() !== 'control') return;
      const a0 = document.activeElement, mine = !a0 || a0 === document.body || a0 === document.documentElement || this.results.contains(a0);
      if (!mine || /^(INPUT|SELECT|TEXTAREA)$/.test(a0 && a0.tagName) || (a0 && a0.isContentEditable) || (!this.full && !this.resCanvas.offsetParent)) return;
      const d = this.dataFor(), P = PLAY[d.world], a = P && P.keys[e.key];
      if (d.error || a === undefined || a >= d.nA) return;
      e.preventDefault(); e.stopPropagation();
      this.playYourself(true);
      if (P.turn) this.playQueue.push(a); else if (!this.playHeld.includes(a)) this.playHeld.push(a);
    };
    document.addEventListener('keydown', this._jumpKey, true);
  };
  /* Full screen: the results over the whole page (and the browser's own full screen where it allows it, not inside a
     frame that forbids it), the picture as big as the screen with the run's line, Train, Jump in and the buttons to
     hold. Esc, or the button, puts them back where they were. */
  BL.setFull = function (on) {
    on = !!on;
    if (on === !!this.full) return;
    this.full = on;
    const R = this.results, doc = document, fsEl = () => doc.fullscreenElement || doc.webkitFullscreenElement;
    if (on) {
      this._fullHome = { parent: R.parentElement, next: R.nextSibling };
      doc.body.append(R);
      R.classList.add('is-full'); doc.body.classList.add('bd-full-on');
      // (the page under it out of reach: Tab, clicks and screen readers stay in full screen; dialogs and toasts sit outside it)
      const app = doc.getElementById('app');
      if (app) app.inert = true;
      this._fullWent = false;
      const de = doc.documentElement;
      try { const q = de.requestFullscreen ? de.requestFullscreen({ navigationUI: 'hide' }) : de.webkitRequestFullscreen ? de.webkitRequestFullscreen() : null; if (q && q.catch) q.catch(() => {}); } catch (e) { /* the page's own full screen is enough */ }
      // the browser's full screen ending (its Esc, its own button) ends this one too
      this._fullChange = () => { if (fsEl()) this._fullWent = true; else if (this._fullWent && this.full) this.setFull(false); };
      doc.addEventListener('fullscreenchange', this._fullChange); doc.addEventListener('webkitfullscreenchange', this._fullChange);
      this._fullEsc = (e) => { if (e.key === 'Escape' && this.full && !e.defaultPrevented && !doc.querySelector('.modal:not([hidden])')) { e.preventDefault(); this.setFull(false); } };
      doc.addEventListener('keydown', this._fullEsc);
      setTimeout(() => { if (this.full && !R.contains(doc.activeElement)) this.resCanvas.focus({ preventScroll: true }); }, 0);
    } else {
      R.classList.remove('is-full'); doc.body.classList.remove('bd-full-on');
      const app = doc.getElementById('app');
      if (app) app.inert = false;
      const h = this._fullHome;
      if (h && h.parent && h.parent.isConnected) h.parent.insertBefore(R, h.next && h.next.parentElement === h.parent ? h.next : null); else R.remove();
      doc.removeEventListener('fullscreenchange', this._fullChange); doc.removeEventListener('webkitfullscreenchange', this._fullChange); doc.removeEventListener('keydown', this._fullEsc);
      if (fsEl()) { try { const q = (doc.exitFullscreen || doc.webkitExitFullscreen).call(doc); if (q && q.catch) q.catch(() => {}); } catch (e) { /* already out */ } }
    }
    clear(this.resPad); this.resPad.hidden = true;
    this._headKey = null; this.resultsDirty = true; this.layoutCenter();
  };
  // the keys while you play (taken before the page's own shortcuts; never while typing in a field)
  BL.listenPlay = function (on) {
    const K = this._playKeys;
    if (K) { document.removeEventListener('keydown', K.down, true); document.removeEventListener('keyup', K.up, true); window.removeEventListener('blur', K.blur); this._playKeys = null; }
    if (!on) return;
    const P = () => (this.youPlay && NA.labHost && NA.labHost.lab === this ? PLAY[this.dataFor().world] : null);
    const typing = (e) => !!e.target && (/^(INPUT|SELECT|TEXTAREA)$/.test(e.target.tagName) || e.target.isContentEditable);
    const down = (e) => {
      const p = P();
      if (!p || typing(e) || e.ctrlKey || e.metaKey || e.altKey) return;
      const a = p.keys[e.key];
      if (a === undefined || a >= this.dataFor().nA) return;
      e.preventDefault(); e.stopPropagation();
      if (p.turn) { if (!e.repeat) this.playQueue.push(a); } else if (!this.playHeld.includes(a)) this.playHeld.push(a);
    };
    const up = (e) => { const p = P(); if (!p || p.keys[e.key] === undefined) return; e.preventDefault(); this.playHeld = this.playHeld.filter((q) => q !== p.keys[e.key]); };
    const blur = () => { this.playHeld = []; };
    this._playKeys = { down, up, blur };
    document.addEventListener('keydown', down, true); document.addEventListener('keyup', up, true); window.addEventListener('blur', blur);
  };
  // the row under the results' title: Jump in, and while you play the controls and the buttons to hold ({ row, pad })
  BL.playRow = function (d) {
    const P = PLAY[d.world], on = !!this.youPlay, touch = (NA.mobile && NA.mobile.mobile) || (window.matchMedia && matchMedia('(pointer: coarse)').matches), keys = keysOf(P);
    const row = el('div', { class: 'bd-play' },
      el('button', { class: on ? 'btn' : 'btn btn-primary', type: 'button', title: on ? 'Hand the controls back: the network carries on from here' : 'Take the controls of this episode where it is', onclick: () => this.playYourself(!on) }, on ? 'Let it play' : 'Jump in'),
      el('span', { class: 'sec-note', text: on ? `${touch ? 'Press and hold the buttons' : P.help[0].toUpperCase() + P.help.slice(1)}.${this.run && !this.stale ? ' The bars: what the network would do.' : ''} Let it play hands back the controls where you are.`
        : `Take the controls where it is now${touch ? '' : ` (or press ${keys}${this.full ? '' : ' with the picture clicked'})`}, then hand back and watch how the network carries on from there.` }));
    if (!on) return { row, pad: null };
    const pad = el('div', { class: 'bd-pad', role: 'group', 'aria-label': 'Your controls' }, ...(P.pad || d.actions.map((a, i) => [i, a])).filter(([a]) => a < d.nA).map(([a, label]) => {
      const b = el('button', { class: 'btn bd-pad-btn', type: 'button', text: label, 'aria-label': d.actions[a] || label });
      const release = () => { this.playHeld = this.playHeld.filter((q) => q !== a); };
      b.addEventListener('pointerdown', (e) => { e.preventDefault(); if (P.turn) this.playQueue.push(a); else if (!this.playHeld.includes(a)) this.playHeld.push(a); try { b.setPointerCapture(e.pointerId); } catch (_) { /* a synthetic event */ } });
      for (const ev of ['pointerup', 'pointercancel', 'lostpointercapture']) b.addEventListener(ev, release);
      b.addEventListener('keydown', (e) => { if (P.turn && (e.key === 'Enter' || e.key === ' ')) { e.preventDefault(); this.playQueue.push(a); } });
      return b;
    }));
    return { row, pad };
  };
  // grow and prune: how a network does on the validation examples (its loss, and its accuracy or R²)
  BL.judgeModel = function (m) {
    const d = this.data, loss = this.setup.loss, n = Math.min(d.nVa, 600), X = d.Xva.subarray(0, n * (d.Xva.length / d.nVa)), Y = d.Yva.subarray(0, n * (d.Yva.length / d.nVa)), r = m.evaluate(X, Y, n, loss);
    return { loss: r.loss, score: BT.isValue(d.family) ? BT.r2(m.predict(X, n), Y, n) : r.count ? r.correct / r.count : 0 };
  };
  // whether the model can grow and prune: Dense blocks only, on a list of numbers
  BL.evolvable = function () { const d = this.dataFor(); return !d.error && NA.bevo.evolvable(this.setup.blocks, d.input); };
  BL.confusionMatrix = function () {
    const d = this.data, n = Math.min(d.nVa, 600), K = d.classes, P = d.Xva.length / d.nVa, o = this.model.predict(d.Xva.subarray(0, n * P), n), M = NA.range(K).map(() => new Array(K).fill(0));
    for (let i = 0; i < n; i++) { let b = 0; for (let k = 1; k < K; k++) if (o[i * K + k] > o[i * K + b]) b = k; M[d.Yva[i]][b]++; }
    return M;
  };
  BL.metricName = function (fam) { fam = fam || this.dataFor().family || this.family(); const own = NA.bgen && NA.bgen.metricName(fam, (this.dataFor() || {}).task); if (own) return own; return BT.isValue(fam) ? 'R²' : fam === 'sequence' ? 'token accuracy' : fam === 'control' ? 'score' : 'accuracy'; };
  // whether the current problem predicts a value (scored by R²) rather than classes or tokens
  BL.valued = function () { return BT.isValue(this.dataFor().family); };
  BL.metricText = function (run) {
    const v = run && run.last && run.last.vm;
    if (v === null || v === undefined) return '—';
    if (run.family === 'control') return `score ${fmtPct(v, 0)}${run.bestVm > v ? ` (best ${fmtPct(run.bestVm, 0)} at ${this.unitOf(run)} ${fmtInt(run.bestAt)})` : ''}`;
    return `validation ${this.metricName(run.family)} ${BT.isValue(run.family) ? fmt(v, 3) : fmtPct(v, 1)}`;
  };

  // Whether two setups train the same model on the same data from the same start: they may differ only in how
  // training proceeds (learning rate, optimizer, batch, steps, schedule, weight decay, clipping).
  const TRAINING_ONLY = ['lr', 'opt', 'batch', 'steps', 'sched', 'wd', 'clip'];
  function sameModel(a, b) {
    const strip = (S) => { const c = JSON.parse(JSON.stringify(S)); for (const k of TRAINING_ONLY) delete c.train[k]; delete c.parts; (c.blocks || []).forEach((q) => { delete q.id; delete q.grp; delete q.part; (q.branches || []).forEach((l) => l.forEach((x) => { delete x.id; })); }); return JSON.stringify(c); };
    return strip(a) === strip(b);
  }
  // What changed between two setups, in words: the run list shows it next to each run's result.
  /* A problem setting's value in the run list's words: on or off, a method's name, an amount with its sign, the table
     (no formula of your own), a list, or the value it starts with when the older setup did not have it yet. */
  const OWN_EDITS = { mazeRows: 'your maze', trackPts: 'your track', obst: 'your course', flock: 'where the sheep start', dogAt: 'where the dog starts' };
  function psVal(k, v, p, inner) {
    if (v === 'own') return 'your own';
    if (Array.isArray(v)) return v.length ? v.join(', ') : 'none';
    if (k === 'rFormula') return v || 'the table';
    if (k === 'keepBest') return v === false ? 'off' : 'on'; // (on unless set off: not among the defaults, so a challenge does not mind it)
    const r = p.family === 'control' ? (NA.rlEnvs.REWARDS[p.task] || []).find((q) => q.key === k) : null;
    if (r) return amountText(typeof v === 'number' ? v : r.def);
    if (v === undefined) { const d0 = inner ? undefined : BT.defaults(p)[k]; return d0 === undefined ? 'none' : psVal(k, d0, p, true); }
    if (typeof v === 'boolean') return v ? 'on' : 'off';
    if (k === 'method' && NA.brl.METHODS[v]) return NA.brl.METHODS[v].short;
    return String(v);
  }
  function diffSetups(a, b) {
    const out = [], P = (id) => BT.byId[id].label, val = (k, v) => JSON.stringify(k === 'branches' && Array.isArray(v) ? v.map((l) => l.map(cleanSpec)) : v);
    if (a.problem !== b.problem) out.push(`problem ${P(a.problem)} → ${P(b.problem)}`);
    else {
      const pa = a.ps[a.problem] || {}, pb = b.ps[b.problem] || {}, ga = NA.bevo.growOptions(pa), gb = NA.bevo.growOptions(pb), va = growVals(ga), vb = growVals(gb);
      // (grow and prune's settings as it uses them: one missing from an older setup is its default, and the derived "−1 layer" counts)
      for (const k of Object.keys(GROW_WORDS)) if (JSON.stringify(va[k]) !== JSON.stringify(vb[k])) out.push(`${GROW_WORDS[k]} ${fmtVal(k, va[k])} → ${fmtVal(k, vb[k])}`);
      for (const k of Object.keys(pb)) if (!GROW_WORDS[k] && JSON.stringify(pa[k]) !== JSON.stringify(pb[k])) out.push(k === 'text' ? 'edited the text' : OWN_EDITS[k] ? `${pa[k] === undefined ? 'made' : 'edited'} ${OWN_EDITS[k]}` : k === 'formula' ? `formula → ${pb.formula}` : `${({ noise: 'noise', n: 'examples', len: 'length', embed: 'embedding size', window: 'window', method: 'learns by', view: 'sees', closer: 'closer bonus', random: 'random mazes', gamma: 'γ', explore: 'exploration', pop: 'population', sigma: 'mutation', slip: 'slippery floor', maxSteps: 'longest episode', force: 'push', poleLen: 'pole length', gravity: 'gravity', drift: 'ball drift', pRight: 'matching lever pays', pOther: 'other levers pay', hide: 'hidden inputs', rFormula: 'reward', select: 'choosing parents', cross: 'mixing parents', mutate: 'mutation kind', grow: 'grow and prune', penalty: 'size penalty', newLayer: 'new layers’ weights', layerSize: 'new layers start with', esSigma: 'nudge size', esLr: 'step size', novK: 'nearest behaviours', novMix: 'chosen for', novAdd: 'behaviours kept a generation', cemElite: 'refit to', cemSigma: 'starting spread', cemFloor: 'spread floor', erlSteps: 'DQN steps a generation', erlEvery: 'DQN joins every', erlLr: 'DQN learning rate', prefAsk: 'who answers', prefEvery: 'asks every', curriculum: 'curriculum', worldModel: 'model of the world', chaserLearns: 'the chaser learns', pbt: 'learners', ppoClip: 'clip', ppoEpochs: 'passes', ppoLam: 'GAE λ', ppoEnt: 'entropy bonus', dqnMem: 'replay memory', dqnTarget: 'target refresh', dqnDouble: 'double DQN', dqnDueling: 'dueling', dqnN: 'looks ahead', warm: 'random tries first', sheepAt: 'sheep start', track: 'track', map: 'map', drones: 'drones', speed: 'speed', wind: 'wind', pad: 'pad', fuel: 'fuel', climate: 'weather', insulation: 'insulation', schedule: 'warm', obstacles: 'course', length: 'corridor', engine: 'engine', motor: 'motor', pattern: 'traffic', ground: 'ground', maze: 'maze', chasers: 'chasers', sheep: 'sheep', normReward: 'normalised rewards', keepBest: 'keep the best network', waves: 'waves', most: 'up to', sizes: 'dot sizes', dirs: 'directions', colour: 'colour', edges: 'edges', rhythm: 'rhythm', digits: 'digits', pictures: 'pictures', z: 'random numbers in', disc: 'discriminator', dim: 'code size', tau: 'temperature', views: 'views', shapes: 'shapes in each', across: 'links across groups', hints: 'who says their group', components: 'bell curves', ...REWARD_WORDS })[k] || k} ${psVal(k, pa[k], BT.byId[a.problem])} → ${psVal(k, pb[k], BT.byId[b.problem])}`);
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
    // (grow and prune's settings, missing from older setups, count as their defaults)
    const TN = Object.assign({ opt: 'optimizer', lr: 'learning rate', batch: 'batch', wd: 'weight decay', sched: 'schedule', clip: 'clipping', init: 'initialisation', seed: 'seed', steps: 'steps', grow: 'grow and prune', growPenalty: 'size penalty', growWidth: 'new layers start with', growNew: 'new layers’ weights', growSize: 'networks side by side', growEvery: 'a generation every' }, GROW_WORDS);
    // (each as grow and prune uses it: a setting missing from an older setup is its default)
    const eff = (T, k) => { const g = NA.bevo.growOptions(T, true); return { grow: !!T.grow, growPenalty: g.penalty, growWidth: g.size, growNew: g.mode, growSize: g.members, growEvery: g.every, ...growVals(g) }[k] ?? T[k]; };
    for (const k of Object.keys(TN)) { const va = eff(a.train, k), vb = eff(b.train, k); if (JSON.stringify(va) !== JSON.stringify(vb)) out.push(`${TN[k]} ${fmtVal(k, va)} → ${fmtVal(k, vb)}`); }
    const count = new Map();
    for (const c of out) count.set(c, (count.get(c) || 0) + 1);
    return out.length ? [...count].map(([c, k]) => (k > 1 ? `${c} ×${k}` : c)) : ['same setup, fresh weights'];
  }
  const OPT_LABEL = { sgd: 'SGD', momentum: 'Momentum', adam: 'Adam', adamw: 'AdamW' }, SCHED_LABEL = { const: 'constant', cosine: 'cosine', step: 'step', warmup: 'warm-up + cosine' };
  // grow and prune's settings in the run list's words, and their values as grow and prune uses them (EV.growOptions)
  const GROW_WORDS = { growStart: 'starts from', growAddNeuron: '+1 neuron', growCutNeuron: '−1 neuron', growAddLayer: '+1 layer', growCutLayer: '−1 layer', growAct: 'new activations', growLean: 'leaning', growRate: 'children that change shape',
    growWhere: 'new layers go', growNeuron: 'new neurons start', growPrune: 'pruning takes', growMaxLayers: 'most hidden layers', growMaxUnits: 'most neurons a layer', growProtect: 'protect new shapes' };
  const growVals = (g) => ({ growStart: g.start, growAddNeuron: g.ops['neuron+'], growCutNeuron: g.ops['neuron-'], growAddLayer: g.ops['layer+'], growCutLayer: g.ops['layer-'], growAct: g.ops.act, growLean: g.lean, growRate: g.rate,
    growWhere: g.where, growNeuron: g.neuron, growPrune: g.prune, growMaxLayers: g.maxLayers, growMaxUnits: g.maxUnits, growProtect: g.protect });
  const GROW_SAY = { growStart: { built: 'your network', small: 'no hidden layer' }, growLean: { grow: 'to grow', even: 'even', prune: 'to prune' }, growWhere: { any: 'anywhere', first: 'after the input', last: 'before the output' },
    growNeuron: { silent: 'silent', random: 'at random' }, growPrune: { weak: 'the weakest', random: 'one at random' } };
  function fmtVal(k, v) {
    if (GROW_SAY[k] && GROW_SAY[k][v]) return GROW_SAY[k][v];
    if (k === 'growRate') return GROW_PCT(v);
    if (k === 'act') return (ACTS[v] || {}).label || v;
    if (k === 'opt') return OPT_LABEL[v] || v;
    if (k === 'sched') return SCHED_LABEL[v] || v;
    if (k === 'merge') return MERGE_WORD[v] || v;
    if (k === 'steps') return v ? fmtInt(v) : 'until paused';
    if (typeof v === 'boolean') return v ? 'on' : 'off';
    return String(v);
  }
  BL.blockName = (b) => blockName(b); // (Find a setting names the blocks so)
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
    const c = withDef(b);
    if (b.type === 'sepconv') return `Separable conv ${c.filters} × ${c.k}×${c.k}`;
    if (b.type === 'se') return `Squeeze-excite ÷${c.ratio}`;
    if (b.type === 'augment') return `Random shifts ±${c.shift}${c.flip ? ' + mirror' : ''}`;
    if (b.type === 'ssm') return `State space ${c.state} → ${c.units}`;
    if (b.type === 'rbf') return `RBF ${c.units}`;
    if (b.type === 'maxout') return `Maxout ${c.units} × ${c.pieces}`;
    if (b.type === 'moe') return `Experts ${c.experts} (${c.top ? 'top ' + c.top : 'all'})`;
    return blockName(b);
  }
  // a block with the engine's defaults filled in, for the newest parts' labels
  const withDef = (b) => Object.assign({}, (NA.bx.BLOCKS[b.type] || {}).defaults, b);
  BL.blockLabel = blockLabel;
  BL.PARTS = PARTS;

  // ------------------------------------------------------------------ lifecycle
  BL.activate = function (preset, ui) {
    this.ui = ui; this.preset = preset;
    if (NA.shelf) NA.shelf.background(this); // (training goes on while the page is hidden)
    if (!this.started) {
      this.started = true;
      if (!this.restore()) this.setup.blocks = withIds(BT.TEMPLATES.find((t) => t.id === 'mlp').blocks());
      const ch = NA.store.get('builderChallenge', null);
      this.challenge = BC.byId[ch] ? ch : null;
    }
    document.body.classList.add('builder-on');
    this.ro = UI.readouts(ui.readouts, [{ key: 'run', label: 'Run' }, { key: 'step', label: 'Step', keep: true }, { key: 'loss', label: 'Train loss' }, { key: 'val', label: 'Val. loss' }, { key: 'metric', label: 'Val. score', short: 'Score', keep: true }, { key: 'params', label: 'Parameters' }]);
    this.buildHyper(ui.hyper);
    this.buildLeft(ui.left);
    this.buildCenter(ui.center);
    this.listenJump(true); // (a world's key jumps in while the network plays)
    // (the Reward tab only for problems that learn from rewards: syncTabs)
    this.tabs = UI.tabs(ui.right, [{ id: 'block', label: 'Part' }, { id: 'reward', label: 'Reward' }, { id: 'train', label: 'Training' }, { id: 'runs', label: 'Runs' }, ...(NA.banalyse ? [{ id: 'analyse', label: 'Analyse' }] : []), { id: 'python', label: 'Python' }], (id, body) => this.buildTab(id, body), this.tabId);
    this.syncTabs();
    // Find a setting, above the tabs (builder-find.js; / puts the cursor in it)
    if (NA.builderFind) { this.finder = NA.builderFind.build(this); ui.right.prepend(this.finder); }
    this.offTheme = theme.onChange(() => { this._region = null; this.vizVersion = (this.vizVersion || 0) + 1; this.resultsDirty = true; this.inspDirty = true; this.archStale = true; this.renderLeft(); });
    this.recheck();
    this.renderWorkbench(); this.renderLeft(); this.syncUndo();
    this._chKey = null; this.renderChallenge();
    if (!this.run) this.showDataOnly();
    this.resultsDirty = true;
  };
  BL.deactivate = function () {
    if (this.full) this.setFull(false);
    this.listenJump(false);
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
      // (the limit beside the step where the bar has room for it, below 1600 px only in its tooltip: a run must not wrap the bar)
      const lim = r && r.limit && !this.stale ? fmtInt(r.limit) : null, wide = !(NA.mobile && NA.mobile.mobile) && window.innerWidth >= 1600;
      this.ro.set('step', r ? fmtInt(r.step) + (lim && wide ? ` / ${lim}` : '') : '0');
      this.ro.title && this.ro.title('step', lim ? `${fmtInt(r.step)} of ${lim} ${this.unitOf(r)}s` : '');
      const ctl = this.family() === 'control';
      // an evolutionary method has no loss: its step reports the population's average fitness (as a loss, negated)
      // (the run's method while it is current; once the setup has changed, the setup's)
      const ga = ctl && NA.brl.isEvo(r && !this.stale ? r.method : this.ps().method);
      this.ro.label('step', ga ? 'Generation' : 'Step', ga ? 'Gen.' : '');
      this.ro.label('loss', ga ? 'Avg. fitness' : ctl ? 'Loss' : 'Train loss'); this.ro.label('val', ga ? 'Best fitness' : ctl ? 'Return' : 'Val. loss'); this.ro.label('metric', ctl ? 'Score' : 'Val. score');
      this.ro.set('loss', r && !this.stale && r.ema !== null ? (ga ? fmt(-r.ema, 2) : fmt(r.ema, 4)) : '—');
      this.ro.set('val', ctl ? (p && p.ret !== null && p.ret !== undefined ? fmt(p.ret, 2) : '—') : p && p.val !== null ? fmt(p.val, 4) : '—');
      this.ro.set('metric', p && p.vm !== null && p.vm !== undefined ? (BT.isValue(r.family) ? `R² ${fmt(p.vm, 3)}` : fmtPct(p.vm, 1)) : '—');
      // (a network evolution has reshaped: its own count)
      this.ro.set('params', r && !this.stale && this.model && this.model.evoBlocks ? fmtInt(this.model.paramCount) : this.check && this.check.ok ? fmtInt(this.check.params) : '—');
    }
    if (this.family() === 'control') { if (this.prefShow) this.prefTick(now); else if (this.race) this.raceTick(now); else this.demoTick(now); }
    // (a world moving in small steps keeps moving while it trains)
    if (this.resultsDirty && this.resCanvas.offsetParent && (!this.training || now - (this.lastRes || 0) > (this.vizMs > 25 ? 700 : this.family() === 'control' ? 110 : 280))) { this.lastRes = now; this.resultsDirty = false; this.drawResults(); }
    if (this.inspDirty && this.tabs.body.offsetParent && (!this.training || now - (this.lastInsp || 0) > 500)) { this.lastInsp = now; this.inspDirty = false; this.updateTab(); }
    if (this.training && now - (this.lastBench || 0) > 600) { this.lastBench = now; this.paintNeurons(); }
    if (this.leftDirty && this.preview.parentElement.clientWidth) { this.leftDirty = false; this.renderLeft(); } // drawn while its screen was hidden
    if (this.centerView === 'arch' && this.arch) {
      if (this.archStale && (!this.training || now - (this.lastArch || 0) > 700)) { this.lastArch = now; this.archStale = false; this.arch.dirty = true; this.arch.refreshFocus(); }
      this.arch.frame(now);
    }
    if (this.centerView === 'growth' && this.growth && (!this.training || now - (this.lastGrow || 0) > 500)) { this.lastGrow = now; this.growth.frame(now); }
  };

  // the quick settings in the control bar (the phone's settings sheet), rebuilt whenever the setup changes them
  BL.hyperKey = function () { const T = this.setup.train; return this.isClassicSetup() ? 'classic' : JSON.stringify([T.opt, T.lr, T.batch, T.steps, this.family() === 'control' ? [this.problem().task, this.ps().method] : null]); };
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
    this._hyperKey = this.hyperKey();
    // learning from rewards: the method first; evolution takes no optimizer, learning rate or batch
    const ctl = this.family() === 'control', method = ctl ? this.ps().method : null, ga = !!method && NA.brl.isEvo(method);
    clear(this.hyperRoot).append(...[
      ctl ? UI.field('Learns by', UI.select(methodChoices(this.problem().task, method).map((m) => ({ value: m, label: NA.brl.METHODS[m].short })), method, (v) => this.setMethod(v)), 'How the network learns from the rewards') : null,
      ga ? null : UI.field('Optimizer', UI.select(Object.entries(OPT_LABEL).map(([value, label]) => ({ value, label })), T.opt, (v) => this.edit((s) => { s.train.opt = v; }))),
      ga ? null : UI.field('Learning rate', UI.select(rates.map((v) => ({ value: v, label: String(v) })), T.lr, (v) => this.edit((s) => { s.train.lr = +v; }))),
      ga ? null : UI.field('Batch', UI.select([8, 16, 32, 64, 128].map((v) => ({ value: v, label: String(v) })), T.batch, (v) => this.edit((s) => { s.train.batch = +v; }))),
      UI.field(ga ? 'Generations' : 'Steps', UI.select(stepChoices(ga, T.steps), T.steps, (v) => this.edit((s) => { s.train.steps = +v; })), ga ? 'How many generations each press of Train runs: ∞ evolves until you pause' : 'How many steps each press of Train runs: ∞ trains until you pause')].filter(Boolean));
  };

  // ------------------------------------------------------------------ left: problem and parts
  BL.buildLeft = function (root) {
    const body = UI.panel(root, 'Problem and parts');
    this.chRoot = el('div', { class: 'bd-ch', 'data-tour': 'challenges' });
    body.append(this.chRoot);
    const groups = BT.problemGroups().map((g) => el('optgroup', { label: g.label }, ...g.problems.map((p) => el('option', { value: p.id, text: p.label }))));
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
    // each group folds away (its header is a button); which are open is remembered once you change one
    this.palGroups = {};
    for (const g of Object.keys(GROUP_TONE)) {
      const grid = el('div', { class: 'bd-pal-grid', id: `bd-pal-${g.replace(/\W+/g, '')}` });
      const list = PARTS.filter((q) => q.group === g);
      for (const p of list) {
        const tile = el('button', { class: 'bd-part', type: 'button', title: p.hint, style: { '--bk': `var(--${GROUP_TONE[g]})` }, dataset: { type: p.type } }, el('span', { class: 'bd-part-ico' }, icon(p.type)), el('span', { class: 'bd-part-name', text: p.label }));
        tile.addEventListener('pointerdown', (e) => this.pressStart(e, { kind: 'part', part: p, el: tile }));
        tile.addEventListener('click', (e) => { if (e.detail === 0) this.showGuide(p.type); }); // keyboard
        grid.append(tile);
      }
      this.palette.append(this.palGroup(g, list.length, grid, GROUP_TONE[g]));
    }
    this.myParts = el('div', { class: 'bd-pal-group bd-myparts', hidden: '' });
    this.palette.append(this.myParts);
  };
  // a group of parts with a header that folds it away: the group's colour, its name, how many parts, a chevron
  BL.palGroup = function (name, count, grid, tone) {
    const head = el('button', { class: 'bd-pal-title bd-pal-toggle', type: 'button', 'aria-expanded': 'true', 'aria-controls': grid.id || null, title: `Show or hide the ${name} parts` },
      el('i', { class: 'bd-pal-dot', style: { background: `var(--${tone || 'ink-2'})` } }), el('span', { class: 'bd-pal-name', text: name }), el('span', { class: 'bd-pal-n', text: String(count) }), el('span', { class: 'bd-pal-chev', 'aria-hidden': 'true', text: '▾' }));
    const box = el('div', { class: 'bd-pal-group', dataset: { group: name } }, head, grid);
    head.addEventListener('click', () => { const open = head.getAttribute('aria-expanded') !== 'true'; this.palOpen = Object.assign({}, this.palOpenNow(), { [name]: open }); NA.store.set('builder.palOpen', this.palOpen); this.syncPalette(); });
    this.palGroups[name] = { box, head, grid };
    return box;
  };
  // the groups open by default: the layers, and the group that suits the problem (pictures, sequences, rewards, tables)
  BL.palOpenNow = function () {
    if (this.palOpen) return this.palOpen;
    const fam = this.family(), k = (this.dataFor().input || {}).kind;
    return { Layers: true, Sequences: fam === 'sequence' || fam === 'seqclass' || fam === 'series' || k === 'series', Images: fam === 'digits' || k === 'image', Reshape: false, Combine: false, 'Classic ML': fam === 'table', Rewards: fam === 'control', 'Your parts': true };
  };
  BL.syncPalette = function () {
    const open = this.palOpenNow();
    for (const [name, g] of Object.entries(this.palGroups || {})) {
      const on = open[name] !== false && (open[name] || false);
      g.head.setAttribute('aria-expanded', String(!!on)); g.grid.hidden = !on; g.box.classList.toggle('is-folded', !on);
    }
  };
  BL.renderLeft = function () {
    if (!this.problemSel) return;
    const p = this.problem(), fam = p.family;
    this.problemSel.value = p.id;
    this.problemHint.textContent = p.hint;
    clear(this.tplRow).append(...BT.templatesFor(fam, this.dataFor()).map((t) => el('button', { class: 'chip bd-tpl', type: 'button', title: t.hint, onclick: () => this.applyTemplate(t) }, t.label)));
    // which parts fit this problem's kind of data
    const fits = this.fits();
    const kind = { plane: 'points', curve: 'a curve', surface: 'a surface', digits: 'pictures', sequence: 'sequences', seqclass: 'sequences', series: 'a series', table: 'a table', control: 'this world as it is seen' }[fam];
    if (this.palOpen === undefined) this.palOpen = NA.store.get('builder.palOpen', null);
    this.renderMyParts();
    this.syncPalette();
    NA.$$('.bd-part:not(.bd-part-mine)', this.palette).forEach((t) => {
      const P = PART[t.dataset.type], off = !P.method && !fits.includes(t.dataset.type);
      t.classList.toggle('is-unsuited', off);
      t.title = P.hint + (off ? ` Not useful for ${kind}: the model will say why if you add it.` : P.method && fam !== 'control' ? ` It learns from rewards: using it switches the problem to ${BT.byId[P.world].label.toLowerCase()}.` : '');
    });
    // a hidden screen (the phone's Parts screen) has no width yet: draw the preview when it shows
    if (!this.preview.parentElement.clientWidth) { this.leftDirty = true; return; }
    const W = Math.max(160, this.preview.parentElement.clientWidth), Hh = fam === 'sequence' ? 78 : fam === 'digits' || fam === 'recon' || fam === 'contrast' || fam === 'segment' ? 64 : 120;
    const ctx = NA.fitCanvas(this.preview, W, Hh);
    BV.preview(ctx, W, Hh, this.dataFor());
  };
  // the parts that suit each kind of data (the others still work where shapes allow, or say why not)
  const VECTOR_PARTS = ['dense', 'neuron', 'gated', 'activation', 'dropout', 'noise', 'layernorm', 'batchnorm', 'fourier', 'rbf', 'maxout', 'kan', 'hopfield', 'parallel', 'skip', 'moe'], CLASSIC_PARTS = ['linear', 'knn', 'tree', 'forest', 'boost'];
  const SEQ_PARTS = ['dense', 'neuron', 'gated', 'activation', 'dropout', 'noise', 'layernorm', 'batchnorm', 'maxout', 'kan', 'hopfield', 'skip', 'attention', 'transformer', 'rnn', 'conv1d', 'position', 'seqpool', 'attnpool', 'ssm', 'se', 'flatten', 'gap', 'gmp', 'parallel', 'moe'];
  const FITS = {
    plane: [...VECTOR_PARTS, ...CLASSIC_PARTS], curve: [...VECTOR_PARTS, ...CLASSIC_PARTS], surface: [...VECTOR_PARTS, ...CLASSIC_PARTS], table: [...VECTOR_PARTS, ...CLASSIC_PARTS],
    digits: ['dense', 'neuron', 'gated', 'activation', 'dropout', 'noise', 'batchnorm', 'maxout', 'skip', 'conv', 'pool', 'avgpool', 'patches', 'sepconv', 'se', 'coords', 'augment', 'flatten', 'gap', 'gmp', 'parallel', 'moe', 'attention', 'transformer', 'position', 'seqpool', 'attnpool', ...CLASSIC_PARTS],
    // a token at every position: nothing that squeezes the sequence, or (squeeze-and-excitation) lets a position see the ones after it
    series: [...SEQ_PARTS, ...CLASSIC_PARTS], seqclass: SEQ_PARTS, sequence: SEQ_PARTS.filter((t) => !['seqpool', 'gap', 'flatten', 'attnpool', 'gmp', 'se'].includes(t)),
  };
  // the kinds of problem of builder-gen.js: networks only
  Object.assign(FITS, {
    recon: ['dense', 'neuron', 'activation', 'dropout', 'noise', 'layernorm', 'batchnorm', 'sample', 'kan', 'hopfield', 'skip', 'conv', 'pool', 'avgpool', 'upsample', 'sepconv', 'augment', 'flatten', 'gap', 'parallel'],
    gan: VECTOR_PARTS, contrast: FITS.digits.filter((t) => !CLASSIC_PARTS.includes(t)), spread: VECTOR_PARTS,
    segment: ['conv', 'pool', 'avgpool', 'upsample', 'skip', 'sepconv', 'se', 'coords', 'batchnorm', 'activation', 'dropout', 'noise', 'patches', 'dense', 'neuron', 'parallel', 'attention', 'transformer', 'position'],
    graph: ['graphconv', 'skip', 'dense', 'neuron', 'activation', 'dropout', 'noise', 'layernorm', 'batchnorm', 'parallel', 'attention', 'transformer'],
  });
  // a world: the parts for the way it is seen (numbers, a picture or readings); classic models need answers to learn from
  BL.fits = function () {
    const fam = this.family();
    if (fam !== 'control') return FITS[fam] || FITS.plane;
    const k = (this.dataFor().input || {}).kind;
    return (k === 'image' ? FITS.digits : k === 'series' ? SEQ_PARTS : VECTOR_PARTS).filter((t) => !CLASSIC_PARTS.includes(t));
  };
  BL.fitsHere = function (type) { return PART[type] && PART[type].method ? this.family() === 'control' : this.fits().includes(type); };
  // learning from rewards: another method, with the number of steps that suits it here
  /* The methods it can learn by in this world: not MAP-Elites or novelty search on the levers, where an episode is one
     pull and there is no style of play to tell apart (unless it learns by one already, so the choice shows it). */
  const methodChoices = (world, cur) => NA.brl.ORDER.filter((m) => m === cur || (!(world === 'levers' && (m === 'me' || m === 'novelty')) && (!NA.brl.isCont(m) || NA.brl.contWorld(world))));
  BL.setMethod = function (m) {
    const p = this.problem(), s = this.ps();
    if (p.family !== 'control' || s.method === m || !NA.brl.METHODS[m]) return;
    const steps = NA.brl.stepsFor(p.task, m), C = NA.brl.isCont, lr = C(m) && !C(s.method) ? 0.001 : !C(m) && C(s.method) ? 0.003 : null;
    this.edit((S) => { S.ps[S.problem].method = m; S.train.steps = steps; if (lr) S.train.lr = lr; });
    this.evoView = 'best'; this.genReplay = null; this.watchCell = null; this._headKey = null; this.resultsDirty = true;
    NA.ui.toast.show(`Learns by ${NA.brl.METHODS[m].label} now: ${fmtInt(steps)} ${NA.brl.isEvo(m) ? 'generations' : 'steps'} a press of Train${lr ? `, at a learning rate of ${lr}${C(m) ? ', which suits a method that sets numbers' : ''}` : ''}.`);
  };
  // another way to see the world: when the blocks no longer fit (dense layers on a picture), a Flatten goes in front
  BL.setWorldView = function (v) {
    let reset = false;
    this.edit((S) => {
      S.ps[S.problem].view = v;
      const d = BT.make(BT.byId[S.problem], S.ps[S.problem], S.train.seed, S.custom);
      if (d.error || NA.bx.check(S.blocks, d.input, d.output).ok) return;
      const flat = [...withIds([{ type: 'flatten' }]), ...S.blocks];
      if ((d.input.kind === 'image' || d.input.kind === 'series') && NA.bx.check(flat, d.input, d.output).ok) { S.blocks = flat; return; }
      // blocks made for another kind of input (a convolution, with 4 numbers to see): the Small MLP again
      S.blocks = withIds(BT.templateBlocks(BT.TEMPLATES.find((t) => t.id === 'mlp'), 'control', d.input.kind)); reset = true;
    });
    if (reset) NA.ui.toast.show('Your blocks need another kind of input, so the model is the Small MLP again for this view. Undo brings yours back.');
  };
  BL.setProblem = function (id) {
    if (id === this.setup.problem) return;
    if (this.race) this.stopRace();
    const fam0 = this.family(), fam1 = BT.byId[id].family, tpl0 = BT.byId[this.setup.problem].template || BT.FAMILIES[fam0].template;
    this.edit((s) => {
      s.problem = id;
      const fl = BT.FAMILIES[fam1], tpl1 = BT.byId[id].template || fl.template, ls = BT.byId[id].losses || fl.losses;
      if (!ls.includes(s.loss.id) || fam0 !== fam1) s.loss = { id: ls[0], smooth: 0, delta: 0.2 };
      // a different kind of data, or a problem with a template of its own (a VAE after an autoencoder): start from
      // that template, with sensible training settings
      if (fam0 !== fam1 || tpl1 !== tpl0) {
        s.blocks = withIds(BT.TEMPLATES.find((t) => t.id === tpl1).blocks());
        Object.assign(s.train, BT.trainFor(BT.byId[id]));
      }
      // another world: the steps that suit its method, and the Small MLP again if the model cannot take what it sees
      if (fam1 === 'control') {
        const p = BT.byId[id], ps = s.ps[id] || (s.ps[id] = BT.defaults(p)), d = BT.make(p, ps, s.train.seed, s.custom);
        s.train.steps = NA.brl.stepsFor(p.task, ps.method);
        if (!d.error && !NA.bx.check(s.blocks, d.input, d.output).ok) s.blocks = withIds(BT.templateBlocks(BT.TEMPLATES.find((t) => t.id === 'mlp'), fam1, d.input.kind));
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
    const blocks = BT.templateBlocks(t, this.family(), (this.dataFor().input || {}).kind);
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
  // learning from rewards: a method part (DQN, PPO, …) sets how the network learns; on a problem without rewards it
  // first switches to one that has them (Undo brings the old setup back)
  BL.useMethod = function (p) {
    const fam = this.family(), numbers = NA.brl.isCont(p.method) && fam === 'control' && !NA.brl.contWorld(this.problem().task), switched = fam !== 'control' || numbers;
    if (switched) this.setProblem(p.world);
    const already = this.ps().method === p.method && (!p.grow || this.ps().grow);
    if (already) NA.ui.toast.show(`${switched ? `${BT.byId[p.world].label}, learning from rewards: ` : ''}it learns by ${NA.brl.METHODS[p.method].label}${p.grow ? ', growing and pruning,' : ''} already.`);
    else {
      if (this.ps().method !== p.method) this.setMethod(p.method);
      if (p.grow) { this.edit((S) => { S.ps[S.problem].grow = true; }); NA.ui.toast.show(`Grow and prune: children of the genetic algorithm can now gain or lose neurons, and gain new layers of one neuron${this.evolvable() ? '' : '. Your model has blocks other than Dense, or sees a picture, so only its weights evolve: try the Small MLP'}.`); }
    }
    if (switched) NA.ui.toast.show(`${p.label} ${numbers ? 'sets numbers rather than picking moves' : 'learns from rewards'}, so the problem switches to “${BT.byId[p.world].label}”, learning by ${NA.brl.METHODS[p.method].label}. Undo brings your setup back.`);
    // (its settings: the Training tab, at grow and prune for that part)
    this.openTrain(p.grow ? 'Grow and prune' : 'Learns by');
  };
  BL.addPart = function (p) {
    if (this.sel && this.sel.kind === 'guide') this.sel = this.guideFrom || null;
    if (p.method) { this.useMethod(p); return; }
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
    this.viewSeg = UI.seg([{ value: 'model', label: 'Model' }, { value: 'both', label: 'Both' }, { value: 'results', label: 'Results' }, { value: 'arch', label: 'Architecture', title: 'Zoom into your model: blocks, neurons, their sums, down to the transistors' }, { value: 'growth', label: 'Growth', title: 'How it learned: a film of the run, its learning curve and the math of each step' }], this.centerView, (v) => { this.centerView = v; this.layoutCenter(); }, 'Show');
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
      return `Results on ${p.label.toLowerCase()}: ${run ? `run ${run.n}, ${this.metricText(run)} after ${fmtInt(run.step)} ${this.unitOf(run)}s` : 'the data, before training'}.`;
    });
    this.resBox = el('div', { class: 'bd-res-box' }, this.resCanvas);
    // (full screen: the buttons to hold go under the picture, where thumbs are)
    this.resPad = el('div', { class: 'bd-res-pad', hidden: '' });
    // (a reward world: the playback bar, under the picture)
    this.resVid = el('div', { class: 'bd-res-vid', hidden: '' });
    this.lossCanvas = el('canvas');
    this.lossChart = new NA.charts.LineChart(this.lossCanvas, { height: 104, logY: true, format: (v) => fmt(v, 4), xLabel: 'Step' });
    this.retCanvas = el('canvas');
    this.retChart = new NA.charts.LineChart(this.retCanvas, { height: 104, format: (v) => fmt(v, 2), xLabel: 'Step' });
    this.resFoot = el('div', { class: 'bd-res-foot' });
    this.results = el('div', { class: 'bd-results', role: 'region', 'aria-label': 'Results' }, this.resHead, this.resBox, this.resVid, this.resPad, this.resFoot);
    this.lossBox = el('div', { class: 'chart-box' }, this.lossCanvas); this.retBox = el('div', { class: 'chart-box', hidden: '' }, this.retCanvas);
    this.lossKey = el('div', { class: 'mini-legend' },
      el('span', { class: 'lg-item' }, el('i', { class: 'key-line', style: { background: 'var(--s1)' } }), 'training loss'), el('span', { class: 'lg-item' }, el('i', { class: 'key-line', style: { background: 'var(--s2)' } }), 'validation loss'));
    this.retKeyText = el('span', { text: 'return per episode while it learns (average of the latest)' });
    this.retKey = el('div', { class: 'mini-legend', hidden: '' }, el('span', { class: 'lg-item' }, el('i', { class: 'key-line', style: { background: 'var(--s1)' } }), this.retKeyText));
    this.resFoot.append(this.lossBox, this.retBox, this.lossKey, this.retKey);
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
    // how it learned: the film of the run (growth.js)
    this.growStage = el('div', { class: 'lab-stage gw-pane' });
    this.growBox = el('div', { class: 'bd-arch bd-grow' }, this.growStage);
    root.append(this.growBox);
    const L = this;
    this.growth = new NA.growth.View(this.growStage, {
      get unit() { return L.unitOf(L.run); },
      version: () => `${this.run ? this.run.n : 0}:${this.run && this.run.film ? this.run.film.version() : 0}:${this.stale ? 1 : 0}`,
      frames: () => (this.run && this.run.film && !this.stale ? this.run.film.frames() : []),
      curve: () => this.filmCurve(),
      math: (f, before) => this.filmMath(f, before),
      fromScratch: () => { this.reset(); if (!this.training) this.play(); },
      scratchLabel: 'New run from scratch',
    }, { title: 'How it learned', what: 'Its results', emptyText: 'Press Train. The moments of the run collect here: what it does after 1, 2, 5, 10, 20… steps, the learning curve with its stages, how its weights grow, and the math of each step.' });
    this.resCanvas.addEventListener('contextmenu', (e) => { if (this.problem().id === 'draw') e.preventDefault(); });
    this.resCanvas.tabIndex = 0; // (a click on the picture, then a key of the world's: you jump in)
    this.resCanvas.addEventListener('pointerdown', (e) => this.resultsPointer(e));
    this.resCanvas.addEventListener('pointermove', (e) => { if (e.pointerType === 'mouse') this.resultsHover(e); });
    this.resCanvas.addEventListener('pointerleave', (e) => { if (e.pointerType === 'mouse') this.resultsHover(null); });
    this.resizeObs = new ResizeObserver(() => { this.layoutCenter(); this.resultsDirty = true; });
    this.resizeObs.observe(this.split); this.resizeObs.observe(this.resBox);
    this.layoutCenter();
  };
  // side by side when there is room, otherwise one at a time
  BL.layoutCenter = function () {
    const arch = this.centerView === 'arch', grow = this.centerView === 'growth';
    this.split.hidden = arch || grow; this.archBox.hidden = !arch;
    if (this.growBox) this.growBox.hidden = !grow;
    if (this.styleSeg) this.styleSeg.hidden = !arch;
    if (grow) { this.viewSeg.set('growth'); NA.$$('button', this.viewSeg).forEach((b) => { if (b.dataset.v === 'both') b.hidden = (this.root || this.growBox.parentElement).clientWidth < 600 || !!(NA.mobile && NA.mobile.mobile); }); this.growth.dirty = true; return; }
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
    if (d.family === 'control') return i.kind === 'image' ? `a picture, ${i.c} channel${i.c > 1 ? 's' : ''} of ${i.h} × ${i.w}` : i.kind === 'series' ? `the last ${i.len} readings of ${i.dim} numbers` : `${i.dim} number${i.dim > 1 ? 's' : ''}${i.dim <= 6 ? ` (${d.obsNames.join(', ')})` : ''}`;
    if (d.inText) return d.inText;
    if (i.kind === 'vector' && d.table) return `${i.dim} number${i.dim > 1 ? 's' : ''} (${d.featNames.length > 3 ? d.featNames.slice(0, 3).join(', ') + ', …' : d.featNames.join(', ')})`;
    if (i.kind === 'vector') return i.dim === 2 ? '2 numbers (x, y)' : '1 number (x)';
    if (i.kind === 'image') return `${i.h} × ${i.w} picture`;
    if (i.kind === 'series') return `the last ${i.len} values, in order`;
    return `${i.len} tokens → embedding ${i.embed}`;
  }
  function outputText(d, loss) {
    const o = d.output, L = NA.bx.LOSSES[loss.id];
    if (d.family === 'control') return `${o.k} outputs (${d.names.join(', ')}) · ${NA.brl.METHODS[d.method].short}`;
    if (d.outText) return `${d.outText} · ${NA.bgen && NA.bgen.lossNote(d) ? 'a loss of its own' : `${L ? L.label : loss.id} loss`}`;
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
    const c = withDef(b);
    if (b.type === 'sepconv') return `${c.filters} filters · ${c.k}×${c.k}${c.dilation > 1 ? ' · dilation ' + c.dilation : ''}${c.stride > 1 ? ' · stride ' + c.stride : ''} · ${ACTS[c.act].label}`;
    if (b.type === 'se') return `a gate for each channel · squeeze ÷${c.ratio}`;
    if (b.type === 'coords') return c.radius ? 'column, row and distance' : 'column and row';
    if (b.type === 'augment') return `up to ${c.shift} px while training${c.flip ? ' · mirrored half the time' : ''}`;
    if (b.type === 'gmp') return 'the largest of each channel';
    if (b.type === 'attnpool') return 'a learned weighted average';
    if (b.type === 'ssm') return `${c.state} memories → ${c.units} · ${ACTS[c.act].label}`;
    if (b.type === 'rbf') return `${c.units} bumps · width ${c.width}`;
    if (b.type === 'maxout') return `${c.units} units · ${c.pieces} pieces each`;
    if (b.type === 'upsample') return `each pixel ${c.size} × ${c.size}`;
    if (b.type === 'skip') return `${c.merge === 'concat' ? 'joins on' : 'adds'} what went into the block${c.back > 1 ? ` ${c.back} before it` : ' before it'}`;
    if (b.type === 'kan') return `${c.units} units · a curve of ${c.grid} bumps on every edge`;
    if (b.type === 'hopfield') return `${c.memories} patterns · β ${c.beta}`;
    if (b.type === 'sample') return `draws a code · β ${c.beta}`;
    if (b.type === 'graphconv') return `${c.units} units · ${c.agg === 'sum' ? 'sum' : 'mean'} of the neighbours · ${ACTS[c.act].label}`;
    if (b.type === 'moe') return `${c.experts} experts of ${c.units} · ${c.top ? 'top ' + c.top : 'all, weighted'} · ${ACTS[c.act].label}`;
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
    if (src.kind === 'part') return !src.part.custom && !src.part.classic && !src.part.method && src.part.type !== 'parallel';
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
    if (src.kind === 'part' && src.part.method) { this.useMethod(src.part); return; } // dropped anywhere on the model
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
    if (this.prefShow && (e.key === 'ArrowLeft' || e.key === 'ArrowRight' || e.key === 'ArrowDown')) { e.preventDefault(); this.answerPref(e.key === 'ArrowLeft' ? 1 : e.key === 'ArrowRight' ? 0 : 0.5); return; }
    if ((e.key === 'k' || e.key === 'K') && this.family() === 'control') { e.preventDefault(); this.vidAct('toggle'); return; } // (the results' pause and play)
    if (e.key === '/' && this.finder) { e.preventDefault(); if (NA.mobile && NA.mobile.mobile) NA.mobile.go('insp'); this.finder.focusInput(); return; }
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
    const ev = live && this.agent ? this.agent.evo : null, gr = live ? this.grower : null;
    const L = { data: live ? this.data : this.dataFor(), model: live ? this.model : null, vizVersion: this.vizVersion, _region: this._region, confusion: live ? this.confusion : null, seqView: this.seqView || (this.seqView = {}), demo: this.demo, run: live ? this.run : null,
      evo: ev, grower: gr, gen: live ? this.gen : null, lossId: (live && this.run ? this.run.cfg : this.setup).loss.id, evoView: this.evoView || 'best', watchCell: this.watchCell, you: !!this.youPlay, genProgress: this.genProgress, replay: this.genReplay || (this.genReplay = { at: performance.now(), gen: -1 }), now: performance.now() };
    // (going back through the episode: the move looked at, drawn as it was; a race: both sides)
    const V = vidOf(this);
    if (this.demo && V.at !== null && this.demo.hist && this.demo.hist[V.at]) L.demo = Object.assign({}, this.demo, this.demo.hist[V.at], { lookBack: { at: this.demo.hist[V.at].t, of: this.demo.hist[this.demo.hist.length - 1].t } });
    if (this.race) L.race = this.race;
    if (this.prefShow) L.prefShow = this.prefShow;
    // learning from rewards: the network on show (for what it looked at) and what the views keep between pictures, for this run
    if (L.data.family === 'control') Object.assign(L, { mapAxes: this.mapAxes, agent: live ? this.agent : null, net: live && this.agent ? (this.agent.neat ? this.agent.neat.adapter() : this.watchNet(this.agent)) : null, training: !!this.training,
      depth: this.depth && this.depth.run === (live ? this.run : null) ? this.depth : (this.depth = { run: live ? this.run : null, recent: [], divHist: [] }) });
    const t0 = performance.now();
    this.resRect = { x: 8, y: 8, w: W - 16, h: Hh - 16 };
    BV.draw(ctx, this.resRect, L);
    this.vizMs = performance.now() - t0; this._region = L._region; this._hits = L.hits || null;
    this.syncVid();
    // the run's loss curves (a diverged run's huge values are capped so the scale stays readable)
    const run = live ? this.run : null, cap = (v) => Math.min(v, 1e3), ctl = L.data.family === 'control';
    this.lossBox.hidden = ctl; this.lossKey.hidden = ctl; this.retBox.hidden = !ctl; this.retKey.hidden = !ctl;
    if (ctl) this.retKeyText.textContent = run && run.method === 'bc' ? 'how often its best move is yours, on the moves it learned from' : run && run.method === 'pref' ? 'return per episode while it learns (the world’s reward; it learns from the reward model’s)' : run && NA.brl.isEvo(run.method) ? `${run.method === 'me' ? 'the best of all the styles' : run.method === 'es' ? 'the best nudged copy of each generation' : run.method === 'cem' ? 'the best network drawn in each generation' : run.method === 'novelty' ? 'the best of each generation by return (it is chosen for novelty, not for this)' : 'the fittest of each generation'}: the return of its best moves` : 'return per episode while it learns (average of the latest)';
    if (ctl) {
      // learning from rewards: the average return of its episodes as it learns (the loss of a policy says little)
      this.retChart.frame();
      this.retChart.opts.xLabel = this.unitOf(run) === 'generation' ? 'Generation' : 'Step';
      this.retChart.set({ series: run ? [{ name: NA.brl.isEvo(run.method) ? 'best fitness' : 'return per episode', color: theme.c('s1'), points: run.curve.filter((p) => p.ret !== null && p.ret !== undefined && Number.isFinite(p.ret)).map((p) => [p.step, p.ret]) }, { name: 'score', color: theme.c('s2'), points: [] }].filter((s) => s.points.length) : [] });
      return;
    }
    this.lossChart.frame();
    const lossSeries = run ? [{ name: 'training loss', color: theme.c('s1'), points: run.curve.filter((p) => p.train !== null && Number.isFinite(p.train)).map((p) => [p.step, cap(p.train)]) }, { name: 'validation loss', color: theme.c('s2'), points: run.curve.filter((p) => p.val !== null && Number.isFinite(p.val)).map((p) => [p.step, cap(p.val)]) }] : [];
    // (a loss that goes below 0, as the Gaussian one can, on a straight scale)
    this.lossChart.opts.logY = !lossSeries.some((s) => s.points.some((p) => p[1] <= 0));
    this.lossChart.set({ series: lossSeries });
  };
  BL.renderResHead = function (live) {
    const d = live ? this.data : this.dataFor(), p = this.problem(), run = live ? this.run : null, S = this.seqView || (this.seqView = {});
    const narrow = !!this.resBox && this.resBox.clientWidth - 16 <= 520;
    const key = JSON.stringify([!!this.full, this.full && [this.training, this.canContinue()], !!this.rec, !!this.race, this.runs.length, this.runs.map((r) => !!r.net).join(), !!this.prefShow, this.agent && this.agent.prefs ? this.agent.prefs.length : 0, this.mapAxes || null, !!(this.agent && this.agent.visits && this.agent.visits.total), this.evoView, narrow, !!this.youPlay, this.watchCell, run && run.method, live && this.model && this.model.evoBlocks ? this.model.paramCount + ':' + this.model.evoBlocks.map((b) => b.units).join('-') : '', p.id, !!live, run && run.n, run && run.last && run.last.vm, run && run.last && run.last.tm, run && run.status, run && Math.floor(run.step / 250), d.family === 'sequence' && S.layer, S.head, this.drawClass, this.drawErase, this.stale, d.names, d.error, this.check && this.check.ok, d.family === 'control' && this.rewardSummary()]);
    if (key === this._headKey) return;
    this._headKey = key;
    if (this._resTip) { NA.tooltip.hide(); this._resTip = false; } // (a tooltip of the picture before: the view, problem or run has changed)
    const parts = [];
    const score = run && run.last && run.last.vm !== null && run.last.vm !== undefined ? (BT.isValue(d.family) ? `R² ${fmt(run.last.vm, 3)}` : fmtPct(run.last.vm, 1)) : null;
    if (d.error) parts.push(el('div', { class: 'bd-res-error', role: 'alert', text: d.error }));
    // full screen: the picture over the whole page (and back); in full screen, Train beside it
    const full = !!this.full, fullBtn = el('button', { class: 'icon-btn bd-full-btn', type: 'button', title: full ? 'Leave full screen (Esc)' : 'Full screen: the picture as big as the screen', 'aria-label': full ? 'Leave full screen' : 'Full screen', 'aria-pressed': String(full), onclick: () => this.setFull(!full) }, icon(full ? 'unfull' : 'full'));
    const trainBtn = full && !d.error ? el('button', { class: 'btn bd-full-train', type: 'button', onclick: () => (this.training ? this.pause() : this.play()) }, this.training ? 'Pause' : this.canContinue() ? 'Continue' : 'Train') : null;
    parts.push(el('div', { class: 'bd-res-title' }, el('span', { class: 'bd-res-k', text: live ? `Run ${run.n} · ${d.family === 'control' ? `score of its best moves${run.bestVm > (run.last ? run.last.vm : 0) ? ` (best ${fmtPct(run.bestVm, 0)})` : ''}` : `validation ${this.metricName()}`}` : this.run && this.stale ? 'Setup edited: press Train for a new run' : 'Press Train to train your model' }),
      el('span', { class: 'bd-res-right' }, score ? el('span', { class: 'bd-res-v' }, el('i', { class: 'bd-run-sw', style: { background: `var(--${run.colour})` } }), score) : null, trainBtn, fullBtn)));
    // what it is paid, in a line, which opens the Reward tab (not while you play, nor in full screen: the picture gets the room)
    const paid = d.family === 'control' && !d.error && !this.youPlay && !full ? this.rewardSummary() : '';
    // (Copy you is paid nothing: it copies your moves; Which did better? learns its reward from the answers; the world's own is for the score)
    const mm = d.family === 'control' ? this.ps().method : null, paidSay = mm === 'bc' ? `none: it copies your moves (the world’s: ${paid})` : mm === 'pref' ? `learned from ${(this.ps().prefAsk || 'you') === 'you' ? 'your answers' : 'the world’s answers'} (the world’s: ${paid})` : paid;
    if (paid) parts.push(el('button', { class: 'bd-rew-chip', type: 'button', title: `What it is paid: ${paidSay}. ${Click()} to change it.`, onclick: () => this.openReward() }, el('span', { class: 'bd-rew-k', text: 'Reward' }), el('span', { class: 'bd-rew-v', text: paidSay })));
    // learning from rewards: the views of a run (builder-depth.js): it plays, what it looked at, the whole generation, MAP-Elites'
    // grid of styles on a small screen, the family line of the best, the map of styles; a list when a small screen has many
    const views = live && d.family === 'control' && !this.youPlay && NA.builderDepth ? NA.builderDepth.views({ method: run.method, world: d.world, narrow, evo: this.agent && this.agent.evo, data: d, visits: !!(this.agent && this.agent.visits && this.agent.visits.total) }) : [];
    if (this.evoView && this.evoView !== 'best' && !views.some((v) => v.value === this.evoView)) { this.evoView = 'best'; this.genReplay = null; this.resultsDirty = true; }
    if (views.length > 1) {
      const cur = views.some((v) => v.value === this.evoView) ? this.evoView : 'best', pick = (v) => { this.evoView = v; this.genReplay = null; this._headKey = null; this.resultsDirty = true; NA.tooltip.hide(); };
      parts.push(el('div', { class: 'bd-evoview' }, narrow && views.length > 3 ? UI.field('Show', UI.select(views, cur, pick)) : UI.seg(views, cur, pick, 'Show')));
    }
    // (the value map: which inputs across and up, and what colours it)
    if (live && this.evoView === 'map' && NA.rlx && NA.rlx.mapOk(d) && views.some((v) => v.value === 'map')) parts.push(NA.rlx.mapControls(this, d));
    if (live && run.method === 'me' && this.watchCell !== null && this.watchCell !== undefined) parts.push(el('p', { class: 'sec-note', text: `Watching the elite of one style. ${Click()} its cell again to watch the best of all.` }));
    // grow and prune: the shape it has evolved so far, next to the one built
    // (with activations that can change, each layer's, where they are not all ReLU)
    const evo = live && run.grow && this.model && this.model.evoBlocks, actsSay = (bs) => (bs.some((b) => (b.act || 'relu') !== 'relu') ? ` (${bs.map((b) => (ACTS[b.act || 'relu'] || {}).label || b.act).join(', ')})` : '');
    const hidden = (bs) => (bs.length ? `${bs.map((b) => b.units).join(' → ')} neuron${bs.length === 1 && bs[0].units === 1 ? '' : 's'}${actsSay(bs)}` : 'no hidden layer');
    // (a run that started small says so: none of the network you built was in it)
    const small = evo && NA.bevo.growOptions(d.family === 'control' ? run.cfg.ps[run.cfg.problem] || {} : run.cfg.train, d.family !== 'control').start === 'small';
    if (evo) parts.push(el('p', { class: 'sec-note bd-res-shape', text: `Its shape now: ${hidden(evo)}, ${fmtInt(this.model.paramCount)} weights (${small ? 'it started with no hidden layer; ' : ''}you built ${hidden(this.setup.blocks.filter((b) => b.type === 'dense'))}).` }));
    // learning from rewards: Jump in; the buttons to hold under the row, or in full screen under the picture
    let pad = null;
    if (this.prefShow && d.family === 'control' && !d.error) parts.push(this.prefRow());
    else if (d.family === 'control' && !d.error && PLAY[d.world]) { const pr = this.playRow(d); parts.push(pr.row); pad = pr.pad; if (pad && !full) parts.push(pad); }
    clear(this.resPad).append(...(pad && full ? [pad] : []));
    this.resPad.hidden = !(pad && full);
    // the playback bar under the picture (a reward world)
    const vid = d.family === 'control' && !d.error;
    clear(this.resVid).append(...(vid ? [this.vidBar(d)] : []));
    this.resVid.hidden = !vid;
    // (while you play, and in full screen, the picture gets the room: no diagnosis, and no legend where it is a sentence)
    const lg = this.youPlay || (full && d.family === 'control') ? null : legendFor(d, live, this);
    if (lg) parts.push(lg);
    const ins = this.youPlay || full ? null : NA.builderInsights;
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
    // (brief: its first sentence, whole: a colon inside it is not its end)
    const body = brief ? (it.body.match(/^.*?[.!?](\s|$)/) || [it.body])[0].trim() : it.body;
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
  function legendFor(d, live, lab) {
    if (!d || d.error) return null;
    const item = (cls, c, label) => el('span', null, el('i', { class: cls, style: c ? { '--c': `var(--${c})` } : null }), label);
    let items = [];
    if (NA.bgen && NA.bgen.owns(d.family)) items = NA.bgen.legend(d, live, item, el, { gen: lab && lab.gen, lossId: lab ? (lab.run && !lab.stale ? lab.run.cfg.loss.id : lab.setup.loss.id) : null });
    else if (d.family === 'control') items = [el('span', { text: live ? (d.cont ? 'the network playing with its best moves, live; bars: the numbers it sets for what it sees now, from −1 to 1, and their spread' : 'the network playing with its best moves, live; bars: its outputs for what it sees now, the move they pick in colour') : 'random moves for now: press Train and it learns from the rewards' })];
    else if (d.family === 'series') items = [item('bd-lg-line', 's1', 'the series'), live ? item('bd-lg-line', 's2', 'one step ahead (true values fed in)') : null, live ? item('bd-lg-dash', 's7', 'on its own (its predictions fed back)') : null];
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
    if (this._hits && this.family() === 'control') {
      const r = this.resCanvas.getBoundingClientRect(), x = e.clientX - r.left, y = e.clientY - r.top, h = this._hits.find((q) => x >= q.x && x <= q.x + q.w && y >= q.y && y <= q.y + q.h);
      // a cell of MAP-Elites' grid: watch that style's elite (again: back to the best of all)
      if (h && h.cell !== undefined) { this.watchCell = this.watchCell === h.cell ? null : h.cell; if (this.evoView === 'grid') this.evoView = 'best'; this.demo = null; this.resultsDirty = true; this._headKey = null; return; }
      // a touch on a part that has more to say (a bar of what it looked at, an ancestor, a style): its tooltip, as hover gives a mouse
      if (e.pointerType !== 'mouse') { this.resultsHover(e); if (this.resultsHit(x, y)) return; }
    }
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
  // the part of the results under the pointer that has a tooltip: of the small marks the nearest, otherwise the last drawn
  BL.resultsHit = function (x, y) {
    let best = null, bd = Infinity;
    for (const q of this._hits || []) {
      if (!q.tip || x < q.x || x > q.x + q.w || y < q.y || y > q.y + q.h) continue;
      const dd = q.cx !== undefined ? Math.hypot(q.cx - x, q.cy - y) : 1e6;
      if (dd <= bd) { bd = dd; best = q; }
    }
    return best;
  };
  // hover over the results of learning from rewards: the part's tooltip, and its mark picked out in the picture
  BL.resultsHover = function (e) {
    const S = this.depth, was = S ? JSON.stringify([S.hoverId, S.hoverPt, S.hoverGen]) : '';
    if (S) { S.hoverId = null; S.hoverPt = null; S.hoverGen = null; }
    const r = e && this._hits && this.family() === 'control' ? this.resCanvas.getBoundingClientRect() : null, x = r ? e.clientX - r.left : 0, y = r ? e.clientY - r.top : 0, h = r ? this.resultsHit(x, y) : null;
    if (S && h && h.id !== undefined) S.hoverId = h.id;
    const rows = h ? h.tip(x, y) : null;
    this._resTip = !!(rows && rows.length);
    if (this._resTip) NA.tooltip.show(rows, e.clientX, e.clientY); else NA.tooltip.hide();
    // a tap's tooltip goes after a few seconds (there is no pointer to leave)
    clearTimeout(this._resTipTimer);
    if (this._resTip && e && e.pointerType !== 'mouse') this._resTipTimer = setTimeout(() => { if (this._resTip) { NA.tooltip.hide(); this._resTip = false; } }, 4000);
    if (S && JSON.stringify([S.hoverId, S.hoverPt, S.hoverGen]) !== was) this.resultsDirty = true;
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
      const chartSec = UI.section('Validation score of each run', this.parts.chartBox, this.parts.chartNote);
      this.parts.chartTitle = chartSec.querySelector('.sec-title');
      this.parts.shelf = el('div', { class: 'bd-shelf' });
      this.parts.guess = el('div', { class: 'bd-guess' });
      body.append(chartSec, UI.section('Compare two runs', this.parts.cmp), UI.section('Runs', this.parts.list), NA.banalyse ? UI.section('Guess first', this.parts.guess) : null, NA.shelf ? UI.section('My models', this.parts.shelf) : null);
    } else if (id === 'python') {
      this.parts.code = el('div');
      body.append(UI.section('Your model in PyTorch', UI.note('The model, loss, optimizer and training loop you set up here, as a script you can run with PyTorch.'), this.parts.code));
    }
    this.updateTab();
  };
  // the Reward tab is there while the problem learns from rewards; away from one, the Part tab takes its place
  BL.syncTabs = function () {
    const b = this.tabs && this.tabs.bar.querySelector('[data-tab="reward"]');
    if (!b) return false;
    const on = this.family() === 'control', an = this.tabs.bar.querySelector('[data-tab="analyse"]');
    b.hidden = !on;
    if (an) an.hidden = on; // (the Analyse tab: for problems with answers)
    if (!on && this.tabId === 'reward') { this.tabs.select('block'); return true; }
    if (on && this.tabId === 'analyse') { this.tabs.select('train'); return true; }
    return false;
  };
  // the reward tab, opened from the reward's summary (on a phone, the panel's own screen)
  BL.openReward = function () {
    if (!this.tabs || this.family() !== 'control') return;
    if (NA.mobile && NA.mobile.mobile) NA.mobile.go('insp');
    this.tabs.select('reward');
  };
  BL.updateTab = function () {
    const id = this.tabId, P = this.parts;
    if (!P || this.syncTabs()) return;
    if (id === 'block') this.renderBlockTab(P.body);
    else if (id === 'reward') this.renderRewardTab(P.body);
    else if (id === 'train') this.renderTrainTab(P.body);
    else if (id === 'runs') this.renderRunsTab();
    else if (id === 'analyse' && NA.banalyse) NA.banalyse.render(this, P.body);
    // the script, editable where builder-py-edit.js is loaded (it keeps an editor that is open as it is)
    else if (id === 'python' && NA.builderPy && NA.builderPyEdit && (this.pyDirty || !P.code.firstChild)) { this.pyDirty = false; NA.builderPyEdit.render(this, P.code); }
    else if (id === 'python' && (this.pyDirty || !P.code.firstChild)) { this.pyDirty = false; clear(P.code).append(NA.builderPy ? UI.codeBox(NA.builderPy.code(this.setup, this.dataFor()), this.isClassicSetup() ? `Python · ${this.setup.blocks.some((b) => b.type === 'boost') ? 'XGBoost' : 'scikit-learn'}` : 'Python · PyTorch') : UI.note('Loading…')); }
  };

  /* The settings of the newest parts (b with its defaults filled in; set edits the block, in every copy of a part of
     your own), each with a line on what the setting changes, in numbers where it helps. */
  const NEW_SETTINGS = new Set(['sepconv', 'se', 'coords', 'augment', 'ssm', 'rbf', 'maxout', 'moe']);
  BL.newPartSettings = function (sec, b, sh, set, actSel) {
    const nums = (list, fmtV = String) => list.map((v) => ({ value: v, label: fmtV(v) }));
    if (b.type === 'sepconv') {
      const C = sh.in && sh.in.length === 3 ? sh.in[0] : null, F = b.filters, k2 = b.k * b.k, reach = b.dilation * (b.k - 1) + 1;
      sec.push(UI.section('Settings', UI.slider({ label: 'Filters (output channels)', min: 1, max: 64, step: 1, value: F, format: String, onChange: (v) => set((q) => { q.filters = v; }) }),
        UI.field('Filter size', UI.seg(nums([3, 5, 7], (k) => `${k}×${k}`), b.k, (v) => set((q) => { q.k = +v; }), 'Filter size')),
        UI.field('Dilation', UI.seg(nums([1, 2, 3]), b.dilation, (v) => set((q) => { q.dilation = +v; }), 'Dilation')),
        UI.field('Stride', UI.seg([{ value: 1, label: '1' }, { value: 2, label: '2 (halves it)' }], b.stride, (v) => set((q) => { q.stride = +v; }), 'Stride')),
        UI.field('Activation', actSel(b.act, (v) => set((q) => { q.act = v; }))),
        C ? UI.note(`${fmtInt(C * k2 + C * F)} weights (${C} × ${k2} for the filters, ${C} × ${F} for the mix) where a full convolution of the same size needs ${fmtInt(C * k2 * F)}.${b.dilation > 1 ? ` With dilation ${b.dilation} each ${b.k}×${b.k} filter spreads over ${reach} × ${reach} pixels, with gaps.` : ''}`) : null));
    } else if (b.type === 'se') {
      const C = sh.in ? (sh.in.length === 3 ? sh.in[0] : sh.in[sh.in.length - 1]) : null;
      sec.push(UI.section('Settings', UI.field('Squeeze ratio', UI.seg(nums([1, 2, 4, 8, 16], (r) => `÷${r}`), b.ratio, (v) => set((q) => { q.ratio = +v; }), 'Squeeze ratio')),
        C ? UI.note(`${C} channel averages → ${NA.bx.seHidden(C, b.ratio)} ReLU unit${NA.bx.seHidden(C, b.ratio) === 1 ? '' : 's'} → ${C} gates between 0 and 1. A narrower middle learns fewer weights and has to summarise more.`) : null));
    } else if (b.type === 'coords') {
      sec.push(UI.section('Settings', el('div', { class: 'chip-row' }, UI.chip('Distance from the centre too', !!b.radius, (on) => set((q) => { q.radius = on; }), 'A third channel: how far each pixel is from the middle of the picture')),
        UI.note('They help when where something is matters: finding a position, or telling the top of a picture from the bottom. A plain convolution answers the same wherever a pattern is.')));
    } else if (b.type === 'augment') {
      sec.push(UI.section('Settings', UI.field('Move by up to', UI.seg(nums([0, 1, 2, 3], (k) => `${k} px`), b.shift, (v) => set((q) => { q.shift = +v; }), 'Move by up to')),
        el('div', { class: 'chip-row' }, UI.chip('Mirror left to right', !!b.flip, (on) => set((q) => { q.flip = on; }), 'Half of the pictures are flipped while training')),
        b.flip && this.problem().id === 'digits' ? UI.note('A mirrored 2 or 7 is not a digit any more: mirroring suits pictures whose class stays the same when flipped, such as shapes.', 'lm-warn') : null,
        UI.note('Only while training: a new move for every picture at every step. When the model is used, pictures pass through unchanged. Compare the training and validation scores of a run with it and one without.')));
    } else if (b.type === 'ssm') {
      sec.push(UI.section('Settings', UI.slider({ label: 'Memories (state size)', min: 1, max: 64, step: 1, value: b.state, format: String, onChange: (v) => set((q) => { q.state = v; }) }),
        UI.slider({ label: 'Output units', min: 1, max: 128, step: 1, value: b.units, format: String, onChange: (v) => set((q) => { q.units = v; }) }),
        UI.field('Activation', actSel(b.act, (v) => set((q) => { q.act = v; }))),
        UI.note('The memories start with time scales from about 2 to 32 steps and learn their own. It gives a vector at every step: follow it with Sequence pool (last step) for one answer per sequence.')));
    } else if (b.type === 'rbf') {
      sec.push(UI.section('Settings', UI.slider({ label: 'Units (bumps)', min: 1, max: 128, step: 1, value: b.units, format: String, onChange: (v) => set((q) => { q.units = v; }) }),
        UI.field('Starting width σ', UI.seg(nums([0.1, 0.25, 0.5, 1, 2]), b.width, (v) => set((q) => { q.width = +v; }), 'Starting width')),
        UI.note('Narrow bumps fit sharp detail but leave gaps between them; wide ones overlap and blur. Centres and widths both move while training.')));
    } else if (b.type === 'maxout') {
      sec.push(UI.section('Settings', UI.slider({ label: 'Units', min: 1, max: 128, step: 1, value: b.units, format: String, onChange: (v) => set((q) => { q.units = v; }) }),
        UI.field('Pieces per unit', UI.seg(nums([2, 3, 4, 5]), b.pieces, (v) => set((q) => { q.pieces = +v; }), 'Pieces per unit')),
        UI.note(`Each piece is a weighted sum of its own: ${b.pieces} pieces need ${b.pieces} times the weights of a dense layer as wide, and give a bend made of up to ${b.pieces} straight parts.`)));
    } else if (b.type === 'moe') {
      sec.push(UI.section('Settings', UI.field('Experts', UI.seg(nums([2, 3, 4, 8]), b.experts, (v) => set((q) => { q.experts = +v; if (q.top > q.experts) q.top = q.experts; }), 'Experts')),
        UI.slider({ label: 'Units in each expert', min: 1, max: 64, step: 1, value: b.units, format: String, onChange: (v) => set((q) => { q.units = v; }) }),
        UI.field('Each example goes to', UI.seg([{ value: 0, label: 'All, weighted' }, { value: 2, label: 'Top 2' }, { value: 1, label: 'Top 1' }].filter((o) => o.value <= b.experts), b.top, (v) => set((q) => { q.top = +v; }), 'Each example goes to')),
        UI.field('Expert activation', actSel(b.act, (v) => set((q) => { q.act = v; }))),
        UI.note(b.top === 1 ? 'Top 1: a single expert answers for each example. Watch for one expert taking every example (collapse); large models add a loss that spreads the work.' : b.top ? `Top ${b.top}: the ${b.top} experts the gate rates highest answer, mixed by the gate's weights; the others get no say and no gradient.` : 'All: every expert answers and the gate mixes them, a soft mixture.')));
    }
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
    else if (b.type === 'skip') {
      const opts = NA.range(i).map((k0) => { const k = k0 + 1, src = i - 1 - k; return { value: k, label: src < 0 ? 'the input' : `block ${src + 1}'s output: ${blockLabel(S.blocks[src])}` }; });
      sec.push(UI.section('Settings', opts.length ? UI.field('Reach back to', UI.select(opts, b.back || 1, (v) => set((q) => { q.back = +v; }), { 'aria-label': 'Reach back to' })) : UI.note('Put a block before it first: it reaches around blocks.'),
        UI.field('Then', UI.seg([{ value: 'add', label: 'Add' }, { value: 'concat', label: 'Join side by side' }], b.merge || 'add', (v) => set((q) => { q.merge = v; }), 'Add or join')),
        UI.note((b.merge || 'add') === 'add' ? 'Added number by number: the earlier output must have the same shape as this one. The blocks between then learn only what to change, and the gradient has a short way back.' : 'Joined side by side: a vector’s numbers, each position’s features, or a picture’s channels, one after the other. Pictures must be the same size.')));
    }
    else if (b.type === 'kan') sec.push(UI.section('Settings', UI.slider({ label: 'Units', min: 1, max: 64, step: 1, value: b.units, format: String, onChange: (v) => set((q) => { q.units = v; }) }),
      UI.field('Bumps on each edge', UI.seg([3, 4, 6, 8, 12].map((k) => ({ value: k, label: String(k) })), b.grid || 6, (v) => set((q) => { q.grid = +v; }), 'Bumps on each edge')),
      UI.note('Each edge’s curve is a weighted sum of this many bumps spread over −2 to 2, plus a weighted SiLU: more bumps, finer curves and more numbers to learn.')));
    else if (b.type === 'hopfield') sec.push(UI.section('Settings', UI.slider({ label: 'Patterns it remembers', min: 1, max: 64, step: 1, value: b.memories, format: String, onChange: (v) => set((q) => { q.memories = v; }) }),
      UI.field('Sharpness β', UI.seg([1, 2, 4, 8, 16].map((k) => ({ value: k, label: String(k) })), b.beta || 4, (v) => set((q) => { q.beta = +v; }), 'Sharpness')),
      UI.note('A large β recalls the one pattern most like the input; a small one gives a blend of many.')));
    else if (b.type === 'upsample') sec.push(UI.section('Settings', UI.field('Bigger by', UI.seg([2, 3, 4].map((k) => ({ value: k, label: `${k}×` })), b.size || 2, (v) => set((q) => { q.size = +v; }), 'Bigger by')),
      UI.note('Each pixel becomes a square of pixels with its value (nearest neighbour). A convolution after it smooths the blocks.')));
    else if (b.type === 'sample') sec.push(UI.section('Settings', UI.slider({ label: 'β: the pull toward one bell curve', min: 0, max: 2, step: 0.05, value: b.beta === undefined ? 0.3 : b.beta, format: (v) => fmt(v, 2), onChange: (v) => set((q) => { q.beta = v; }) }),
      UI.note(`It reads ${sh.in ? sh.in[0] : 'an even number of'} numbers: the first half are middles, the second half the logs of their spreads squared. While it learns it draws each code from its bell curve; when it is used it gives the middles. β × the KL divergence from one standard bell curve is added to the loss: 0 makes it a plain autoencoder; more makes codes easier to draw from, and pictures blurrier.`)));
    else if (b.type === 'graphconv') sec.push(UI.section('Settings', UI.slider({ label: 'Units', min: 1, max: 64, step: 1, value: b.units, format: String, onChange: (v) => set((q) => { q.units = v; }) }),
      UI.field('From the neighbours', UI.seg([{ value: 'mean', label: 'Average' }, { value: 'sum', label: 'Sum' }], b.agg || 'mean', (v) => set((q) => { q.agg = v; }), 'How the neighbours are combined')),
      UI.field('Activation', actSel(b.act, (v) => set((q) => { q.act = v; }))),
      UI.note('Each node averages (or adds up) its own features and its neighbours’, then one dense layer, the same for every node, makes its new features. The links are passed on, so another round can follow.')));
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
    else if (NEW_SETTINGS.has(b.type)) this.newPartSettings(sec, withDef(b), sh, set, actSel);
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
    const grid = el('div', { class: 'bd-pal-grid', id: 'bd-pal-mine' });
    for (const def of parts) {
      const proto = partProto(def), tile = el('button', { class: 'bd-part bd-part-mine', type: 'button', title: `${def.name}: ${def.blocks.map(blockLabel).join(' → ')}. Drag it onto the model, or ${click()} it.`, style: { '--bk': 'var(--ink-2)' }, dataset: { type: proto.type } }, el('span', { class: 'bd-part-ico' }, icon('part')), el('span', { class: 'bd-part-name', text: def.name }));
      tile.addEventListener('pointerdown', (e) => this.pressStart(e, { kind: 'part', part: proto, el: tile }));
      tile.addEventListener('click', (e) => { if (e.detail === 0) this.showGuide(proto.type); });
      grid.append(tile);
    }
    const g = this.palGroup('Your parts', parts.length, grid, 'ink-2');
    this.myParts.append(...g.childNodes);
    this.palGroups['Your parts'].box = this.myParts;
    this.syncPalette();
  };
  // the guide to a part, with a button that adds it, and one that switches to a problem it suits when this one does not
  const KIND_TEXT = { plane: 'points', curve: 'a curve', surface: 'a surface', digits: 'pictures', sequence: 'sequences', seqclass: 'sequences', series: 'a series', table: 'a table', control: 'this world as it is seen' };
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
    const add = el('button', { class: 'btn btn-primary', type: 'button', onclick: () => this.addPart(p) }, p.method ? (this.family() === 'control' ? `Learn by ${p.label}` : `Learn by ${p.label}: ${BT.byId[p.world].label.toLowerCase()}`) : p.classic ? `Use a ${p.name.toLowerCase()}` : 'Add to the model');
    const switchTo = !fits && tp && !p.method ? el('button', { class: 'btn', type: 'button', onclick: () => { this.setProblem(tryId); if (tryId === 'csv' && !this.setup.custom.table) this.setTable(BT.SAMPLES.houses.make()); this.showGuide(type); } }, `Switch to ${tp.label.toLowerCase()}`) : null;
    // a method part fits any problem: its button switches to one with rewards when needed
    body.append(NA.partGuide.render(type, { part: p, fits: fits || !!p.method, kind: KIND_TEXT[this.family()], add, switchTo }),
      el('div', { class: 'btn-row' }, el('button', { class: 'btn btn-quiet', type: 'button', onclick: () => this.selectBlock(this.guideFrom || null) }, '← Back')));
  };
  BL.renderProblemSettings = function (body) {
    const p = this.problem(), s = this.ps(), fam = p.family, set = (fn) => this.edit((S) => fn(S.ps[S.problem]));
    const groups = BT.problemGroups().map((g) => el('optgroup', { label: g.label }, ...g.problems.map((q) => el('option', { value: q.id, text: q.label, selected: q.id === p.id }))));
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
      if (p.task === 'wiggles') ctl.push(UI.slider({ label: 'Waves', min: 1, max: 16, step: 1, value: s.waves, format: String, onChange: (v) => set((q) => { q.waves = v; }) }),
        UI.note(`It goes up and down ${s.waves} time${s.waves === 1 ? '' : 's'} between −1 and 1. The more waves, the longer a plain MLP takes: it learns the slow shapes of a curve first and the fast ones last, if at all. Fourier features (a part, or the template) start it from waves of many speeds.`));
      ctl.push(UI.slider({ label: 'Noise', min: 0, max: 0.4, step: 0.01, value: s.noise, format: (v) => fmt(v, 2), onChange: (v) => set((q) => { q.noise = v; }) }), UI.slider({ label: 'Examples', min: 20, max: 1000, step: 10, value: s.n, format: String, onChange: (v) => set((q) => { q.n = v; }) }));
    }
    if (fam === 'digits') ctl.push(UI.slider({ label: 'Examples', min: 300, max: 4000, step: 100, value: s.n, format: String, onChange: (v) => set((q) => { q.n = v; }) }));
    // the pictures' own settings
    if (p.task === 'dots') ctl.push(UI.field('Up to', UI.seg([3, 5, 7].map((v) => ({ value: v, label: `${v} dots` })), s.most, (v) => set((q) => { q.most = +v; }), 'Up to how many dots')),
      UI.field('Their sizes', UI.seg([{ value: 'same', label: 'All the same' }, { value: 'mixed', label: 'Mixed' }], s.sizes, (v) => set((q) => { q.sizes = v; }), 'Dot sizes')),
      UI.note(s.sizes === 'same' ? 'Every dot the same size: the ink in the whole picture already says how many there are, so even a dense layer can count.' : 'Dots of every size: the amount of ink no longer says how many, so the model has to find each dot. A filter used everywhere, then added up, does that.'));
    if (p.task === 'stripes') ctl.push(UI.field('Directions', UI.seg([{ value: 2, label: '— and |' }, { value: 4, label: 'All four' }], s.dirs, (v) => set((q) => { q.dirs = +v; }), 'Directions')),
      UI.slider({ label: 'Noise', min: 0, max: 0.6, step: 0.05, value: s.noise, format: (v) => fmt(v, 2), onChange: (v) => set((q) => { q.noise = v; }) }),
      UI.note('Stripes of any spacing, shifted anywhere, faint or strong, in noise. Which way they run is the same in every corner of the picture: a small filter slid over it sees that, wherever it looks.'));
    if (p.task === 'colours') ctl.push(UI.field('The pictures', UI.seg([{ value: 'colour', label: 'In colour' }, { value: 'grey', label: 'In grey' }], s.colour, (v) => set((q) => { q.colour = v; }), 'Colour or grey')),
      UI.note(s.colour === 'grey' ? 'In grey there is one channel, each pixel its brightness. The three colours are equally bright, so red, green and blue look the same: only the shape can be told.' : 'Three channels, red, green and blue, as a screen has. A convolution reads all three at once at every place: its filters can be red-sensitive, green-sensitive or anything between.'));
    if (NA.bgen && NA.bgen.owns(fam)) ctl.push(...NA.bgen.controls(p, s, set, UI, el, fmt));
    if (fam === 'table') ctl.push(...tableControls(this, s));
    if (fam === 'control') ctl.push(...worldControls(this, p, s));
    if (fam === 'surface') ctl.push(UI.slider({ label: 'Noise', min: 0, max: 0.3, step: 0.01, value: s.noise, format: (v) => fmt(v, 2), onChange: (v) => set((q) => { q.noise = v; }) }), UI.slider({ label: 'Examples', min: 200, max: 3000, step: 100, value: s.n, format: String, onChange: (v) => set((q) => { q.n = v; }) }));
    if (p.task === 'picture') ctl.push(UI.field('Edges', UI.seg([{ value: 'sharp', label: 'Sharp' }, { value: 'soft', label: 'Soft' }], s.edges, (v) => set((q) => { q.edges = v; }), 'Edges')),
      UI.note(s.edges === 'soft' ? 'Soft edges: the face fades into its eyes and mouth. Slow, smooth changes are what a plain MLP learns first, so it does well.' : 'Sharp edges: the height jumps at the rim of the face, the eyes and the mouth. A plain MLP blurs fine detail like that; Fourier features draw it.'));
    if (fam === 'series') ctl.push(UI.slider({ label: 'Window (values the model sees)', min: 1, max: 48, step: 1, value: s.window, format: String, onChange: (v) => set((q) => { q.window = v; }) }),
      UI.slider({ label: 'Length of the series', min: 400, max: 4000, step: 100, value: s.n, format: String, onChange: (v) => set((q) => { q.n = v; }) }),
      UI.slider({ label: 'Noise', min: 0, max: 0.3, step: 0.01, value: s.noise, format: (v) => fmt(v, 2), onChange: (v) => set((q) => { q.noise = v; }) }),
      UI.note('The first 80% of the series trains and the last 20% validates: the model is judged on a future it has not seen.'));
    if (p.task === 'heartbeat') ctl.push(UI.field('Rhythm', UI.seg([{ value: 'steady', label: 'Steady' }, { value: 'irregular', label: 'Irregular' }], s.rhythm, (v) => set((q) => { q.rhythm = v; }), 'Rhythm')),
      UI.note(s.rhythm === 'irregular' ? 'A beat every 22 to 38 steps: when the next comes cannot be known for sure, only guessed from how long ago the last one was.' : `A beat every 30 steps. A window of ${s.window} ${s.window >= 30 ? 'reaches back to the last beat, so the next can be timed' : 'can miss the last beat altogether: between beats the line is flat, and flat says nothing about when the next comes'}.`));
    if (p.task === 'walk') ctl.push(UI.note('However it is built, a model cannot beat repeating the last value here: each step is a fresh coin toss. The R² looks high only because the walk moves little in one step.'));
    if (p.task === 'brackets') ctl.push(UI.slider({ label: 'Length', min: 6, max: 24, step: 2, value: s.len, format: String, onChange: (v) => set((q) => { q.len = v; }) }),
      UI.note('Most sequences have as many open brackets as closing ones, so counting them cannot tell: the count of open ones must never drop below zero on the way.'));
    if (p.task === 'palindrome') ctl.push(UI.slider({ label: 'Length', min: 4, max: 16, step: 1, value: s.len, format: String, onChange: (v) => set((q) => { q.len = v; }) }),
      UI.note('Half are palindromes, half differ from one in a single digit: each digit must be checked against its mirror image across the middle.'));
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
      } else if (p.task === 'add') ctl.push(UI.field('Digits', UI.seg([2, 3, 4].map((v) => ({ value: v, label: String(v) })), s.digits, (v) => set((q) => { q.digits = +v; }), 'Digits in each number')),
        UI.slider({ label: 'Examples', min: 500, max: 6000, step: 250, value: s.n, format: String, onChange: (v) => set((q) => { q.n = v; }) }),
        UI.note(`Two numbers of ${s.digits} digits and their sum in ${s.digits + 1}, written the largest digit first. Each digit of the answer depends on the carry from the right, which the model works out from digits it has to look back at.`));
      else if (p.task !== 'language') ctl.push(UI.slider({ label: 'Sequence length', min: 2, max: 6, step: 1, value: s.len, format: String, onChange: (v) => set((q) => { q.len = v; }) }), UI.slider({ label: 'Examples', min: 500, max: 6000, step: 250, value: s.n, format: String, onChange: (v) => set((q) => { q.n = v; }) }));
      else ctl.push(UI.slider({ label: 'Stories', min: 300, max: 4000, step: 100, value: s.n, format: String, onChange: (v) => set((q) => { q.n = v; }) }));
      ctl.push(UI.field('Embedding size', UI.seg([8, 16, 32, 64].map((v) => ({ value: v, label: String(v) })), s.embed, (v) => set((q) => { q.embed = +v; }), 'Embedding size')),
        UI.note('Each token becomes a learned vector of this many numbers, plus a learned vector for its position. Attention heads split it.'));
    }
    if (ctl.length) sec.push(UI.section(fam === 'control' ? 'The world' : 'Data', ...ctl));
    if (fam === 'control') sec.push(...worldSenses(this, p, s));
    const d = this.dataFor();
    if (!d.error) sec.push(UI.section('What the model sees', el('div', { class: 'kv-grid' }, ...(fam === 'control' ? [['Input', inputText(d)], ['Output', outputText(d, this.setup.loss)], ['Actions', d.actions.join(', ')], ['Learns by', NA.brl.METHODS[d.method].label]] : [['Input', inputText(d)], ['Output', outputText(d, this.setup.loss)], ['Training examples', fmtInt(d.nTr)], ['Validation examples', fmtInt(d.nVa)]]).map(([k, v]) => el('div', { class: 'kv' }, el('span', { class: 'kv-k', text: k }), el('span', { class: 'kv-v', text: v }))))));
    body.append(...sec);
  };
  // the run list's words for a reward's amount, and for the worlds' other settings
  const REWARD_WORDS = Object.assign(Object.fromEntries(Object.values(NA.rlEnvs.REWARDS).flat().map((r) => [r.key, r.label.toLowerCase().replace(/ \(.*\)$/, '')])),
    { extra: 'extra inputs', laps: 'laps', vmax: 'top speed', grip: 'grip', range: 'sensor range', lives: 'lives', goal: 'drones to destroy', recharge: 'gun recharge', padWidth: 'pad width', days: 'days', warm: 'warm to',
      band: 'comfortable within', radiator: 'radiator', course: 'course length', run: 'running speed', jump: 'jump', lit: 'lamp lit for', wait: 'time to turn', start: 'start', friction: 'friction', reach: 'reach within',
      motors: 'motors', amber: 'amber', busy: 'how busy', muscles: 'muscles', aim: 'chasers aim', flee: 'sheep run from', dog: 'dog speed', width: 'board width', height: 'board height', paddle: 'paddle', fall: 'falls past' });
  // A choice among values for a world setting, with the one in use added when a link set another
  function worldPick(L, s, label, key, values, def, show, title) {
    const cur = typeof s[key] === 'number' ? s[key] : def, vals = values.includes(cur) ? values : [...values, cur].sort((a, b) => a - b);
    return UI.field(label, UI.select(vals.map((v) => ({ value: v, label: show ? show(v) : String(v) })), cur, (v) => L.edit((S) => { S.ps[S.problem][key] = +v; })), title);
  }
  const signed = (v) => (v > 0 ? '+' : '') + fmt(v, Math.abs(v) < 0.1 && v !== 0 ? 2 : Math.abs(v) % 1 ? 1 : 0);
  const pctOf = (v) => fmtPct(v, 0);
  const GRAVITY = { 1.6: 'Moon', 3.7: 'Mars', 9.8: 'Earth', 24.8: 'Jupiter' };
  // A world's settings: for the maze which one, random mazes, a bonus for getting closer, a slippery floor and the
  // longest episode; the pole's push, length, gravity, track, the lean it falls at, its start and longest episode;
  // the catch's drift, board and paddle; for the levers how many and how often they pay; and each bigger world's own,
  // each with a note on what it means as it is set now.
  function worldControls(L, p, s) {
    const set = (fn) => L.edit((S) => fn(S.ps[S.problem])), out = [];
    const seg = (label, key, pairs, def, title) => UI.field(label, UI.seg(pairs.map(([value, text, tip]) => ({ value, label: text, title: tip })), s[key] ?? def, (v) => set((q) => { q[key] = v; }), label), title);
    const longest = (values, def, show) => worldPick(L, s, 'Longest episode', 'maxSteps', values, def, show || ((v) => `${fmtInt(v)} steps`), 'The most steps an episode can last');
    const plural = (n, one, many) => `${fmt(n, n % 1 ? 1 : 0)} ${n === 1 ? one : many}`;
    /* a layout of your own (the maze, the track, the course, where the sheep start; builder-worldedit.js): choosing
       Your own uses the one you made, or opens the editor to make one; its button opens the editor at any time */
    const WE = NA.worldEdit, OWN_KEY = { maze: 'maze', track: 'track', hurdles: 'obstacles', herd: 'sheepAt' };
    const own = (kind, v, fn) => { if (v === 'own' && WE && !WE.inUse(kind, Object.assign({}, s, { [OWN_KEY[kind]]: 'own' }))) { WE.open(L, kind); return; } set(fn); };
    const editBtn = (kind, label) => el('div', { class: 'btn-row bd-own-row' }, el('button', { class: 'btn', type: 'button', disabled: !WE, onclick: (e) => WE.open(L, kind, e.currentTarget) }, label));
    if (p.task === 'levers') {
      out.push(UI.field('Levers', UI.seg([3, 5].map((v) => ({ value: v, label: String(v) })), s.levers, (v) => set((q) => { q.levers = +v; }), 'Levers')),
        worldPick(L, s, 'The matching lever pays', 'pRight', [0.5, 0.6, 0.8, 0.9, 1], 0.8, pctOf), worldPick(L, s, 'Any other lever pays', 'pOther', [0, 0.1, 0.2, 0.4], 0.2, pctOf),
        UI.note(`The lever that matches the light pays ${pctOf(s.pRight ?? 0.8)} of the time, any other ${pctOf(s.pOther ?? 0.2)}. One pull per episode.`));
    }
    if (p.task === 'pole') {
      const fall = s.fall ?? 12, ends = (s.track ?? 4.8) / 2;
      out.push(worldPick(L, s, 'Push', 'force', [5, 10, 20], 10, (v) => `${v} N`, 'How hard each push is'), worldPick(L, s, 'Pole length', 'poleLen', [0.5, 1, 2], 1, (v) => `${v} m`),
        worldPick(L, s, 'Gravity', 'gravity', [1.6, 3.7, 9.8, 24.8], 9.8, (v) => `${GRAVITY[v] || 'Gravity'}: ${v} m/s²`), longest([100, 200, 500, 1000], 200),
        worldPick(L, s, 'Track', 'track', [2.4, 4.8, 9.6], 4.8, (v) => `${v} m long`, 'How far the cart can roll before it runs off the end'), worldPick(L, s, 'Falls past', 'fall', [6, 12, 24], 12, (v) => `${v}°`, 'How far the pole may lean before it counts as fallen'),
        worldPick(L, s, 'Starts', 'start', [0.05, 0.1, 0.2], 0.05, (v) => ({ 0.05: 'Nearly still', 0.1: 'Wobbling', 0.2: 'Swaying' })[v] || String(v), 'How far from upright and still each episode may begin'),
        UI.note(`A longer pole tips over more slowly, and so does any pole where gravity is weaker: both are easier to balance. A stronger push corrects faster but overshoots more. It falls when it leans past ${fall}° or the cart rolls ${fmt(ends, ends % 1 ? 1 : 0)} m from the middle; what it sees is scaled to those limits. The score is the share of the longest episode it keeps the pole up.`));
    }
    if (p.task === 'catch') {
      const H = s.height || 8;
      out.push(UI.field('Ball drift', UI.seg([{ value: true, label: 'On' }, { value: false, label: 'Off' }], s.drift !== false, (v) => set((q) => { q.drift = v; }), 'Ball drift')),
        UI.note(s.drift !== false ? 'Each ball drifts left, right or straight down, bouncing off the walls: where it lands depends on which way it drifts.' : 'Every ball falls straight down: where it starts is where it lands.'),
        worldPick(L, s, 'Board width', 'width', [5, 7, 9], 5, (v) => `${v} squares`), worldPick(L, s, 'Board height', 'height', [6, 8, 12], 8, (v) => `${v} rows`), worldPick(L, s, 'Paddle', 'paddle', [1, 2, 3], 1, (v) => `${plural(v, 'square', 'squares')} wide`),
        UI.note(`The ball takes ${H - 1} steps to fall, so an episode is ${H - 1} steps long. A wider board leaves the paddle further to go; a wider paddle makes catching easier. The pixels it sees grow with the board.`));
    }
    if (p.task === 'track') {
      const laps = s.laps || 4;
      out.push(UI.field('Track', UI.seg([['oval', 'Oval'], ['esses', 'Esses'], ['hairpins', 'Hairpins'], ['random', 'Random', 'A new random track every episode, and scored on 4 it never saw'], ['own', 'Your own', 'A track you shape yourself']].map(([value, label, title]) => ({ value, label, title })), s.track || 'esses', (v) => own('track', v, (q) => { q.track = v; }), 'Track')),
        UI.note({ oval: 'Two long straights and two wide bends: it can keep its foot down almost all the way.', esses: 'A long sweeping bend, then four tight S-bends one after another: it has to brake for them, and steer left and right in turn.', hairpins: 'Four hairpins, each turning it right round: twice as long, and it has to slow right down for every one.', random: 'A new track every episode: what it learns has to work on any track. It is scored on 4 tracks it never trained on.', own: 'The track you shaped, every episode.' }[s.track || 'esses']),
        editBtn('track', s.track === 'own' ? 'Edit your track' : 'Shape your own track'),
        worldPick(L, s, 'Laps', 'laps', [1, 2, 4, 8], 4, (v) => plural(v, 'lap', 'laps')), longest([0, 1500, 3000], 0, (v) => (v ? `${fmtInt(v)} steps` : `${plural(laps, 'lap', 'laps')} at 10 m/s`)),
        worldPick(L, s, 'Top speed', 'vmax', [16, 24, 32], 24, (v) => `${v} m/s`), worldPick(L, s, 'Grip', 'grip', [8, 14, 20], 14, (v) => `${({ 8: 'Wet', 14: 'Dry', 20: 'Racing tyres' })[v] || 'Grip'}: ${v} m/s²`, 'How hard it can turn before it slides'),
        worldPick(L, s, 'Sensors see', 'range', [20, 40, 60], 40, (v) => `${v} m`, 'How far its 7 distance sensors reach'),
        UI.note(`It sees 7 distances to the walls (out to ${s.range || 40} m) and its speed, or a camera picture from above. Throttle builds its speed toward ${s.vmax || 24} m/s; grip (${s.grip || 14} m/s² sideways) limits how sharply it can turn, so it has to brake for tight bends, and the less grip, the more. The score is the share of ${plural(laps, 'lap', 'laps')} it drives.`));
    }
    if (p.task === 'arena') {
      const lives = s.lives || 3, rc = s.recharge ?? 3;
      out.push(UI.field('Map', UI.seg([['room', 'Room'], ['pillars', 'Pillars'], ['halls', 'Halls']].map(([value, label]) => ({ value, label })), s.map || 'room', (v) => set((q) => { q.map = v; }), 'Map')),
        UI.field('Drones at once', UI.seg([1, 2, 3].map((v) => ({ value: v, label: String(v) })), s.drones || 2, (v) => set((q) => { q.drones = v; }), 'Drones at once')),
        UI.field('Drones fly', UI.seg([['slow', 'Slowly'], ['fast', 'Fast']].map(([value, label]) => ({ value, label })), s.speed || 'slow', (v) => set((q) => { q.speed = v; }), 'Drone speed')),
        worldPick(L, s, 'Lives', 'lives', [1, 3, 5], 3, String), worldPick(L, s, 'Drones to destroy', 'goal', [5, 10, 20], 10, String, 'How many drones end the game'),
        worldPick(L, s, 'Gun recharges in', 'recharge', [0, 3, 8], 3, (v) => (v ? plural(v, 'step', 'steps') : 'No time'), 'How long after a shot it can fire again'), longest([300, 500, 1000], 500),
        UI.note(`${{ room: 'An open room with four pillars.', pillars: 'More pillars to hide behind: drones vanish from view and come back.', halls: 'Walls split it into halls and doorways: a drone can be close but out of sight.' }[s.map || 'room']} Drones drift toward it and cost a life on reaching it (${plural(lives, 'life', 'lives')}); a shot that crosses one destroys it. ${rc ? `After a shot the gun takes ${plural(rc, 'step', 'steps')} to recharge.` : 'The gun can fire every step.'} The score is the share of ${s.goal || 10} drones it destroys.`));
    }
    if (p.task === 'lander') {
      out.push(worldPick(L, s, 'Gravity', 'gravity', [1.62, 3.71, 9.81], 1.62, (v) => `${{ 1.62: 'Moon', 3.71: 'Mars', 9.81: 'Earth' }[v] || 'Gravity'}: ${v} m/s²`),
        UI.field('Wind', UI.seg([{ value: false, label: 'Calm' }, { value: true, label: 'Gusts' }], !!s.wind, (v) => set((q) => { q.wind = v; }), 'Wind')),
        UI.field('The pad', UI.seg([['middle', 'In the middle'], ['random', 'Anywhere']].map(([value, label]) => ({ value, label })), s.pad || 'middle', (v) => set((q) => { q.pad = v; }), 'Pad')),
        worldPick(L, s, 'Fuel', 'fuel', [100, 150, 300], 150, (v) => `${v} steps of engine`),
        seg('Engine', 'engine', [['weak', 'Weak', 'Lifts 1.8 times gravity, plus 1 m/s²'], ['normal', 'Normal', 'Lifts 2.4 times gravity, plus 1 m/s²'], ['strong', 'Strong', 'Lifts 3.5 times gravity, plus 1 m/s²']], 'normal'),
        worldPick(L, s, 'Pad width', 'padWidth', [8, 12, 20], 12, (v) => `${v} m`), longest([300, 600, 1000], 300),
        UI.note(`It sees where it is from the pad, how fast it moves, its tilt and spin, and its fuel. The main engine pushes along its axis (more strongly where gravity is stronger, and ${s.engine === 'weak' ? 'only just enough to stop a fall in time' : s.engine === 'strong' ? 'strongly enough to stop a fall quickly' : 'enough to slow a fall'}); the side thrusters turn it. Landing is gentle on the pad (${s.padWidth || 12} m wide), slower than 3 m/s down and 2 m/s across and within 25° of upright; anything else is a crash, and so is running out of time. The score is the share of 10 starts it lands.`));
    }
    if (p.task === 'maze') {
      out.push(UI.field('Maze', UI.select([...NA.rl.MAZE_SET.map((m) => ({ value: m.id, label: m.name })), { value: 'own', label: 'Your own' }], s.maze, (v) => own('maze', v, (q) => { q.maze = v; }))),
        editBtn('maze', s.maze === 'own' ? 'Edit your maze' : 'Draw your own maze'),
        UI.field('Train on', UI.seg([{ value: false, label: 'This maze' }, { value: true, label: 'Random mazes', title: 'A new random maze every episode' }], !!s.random, (v) => set((q) => { q.random = v; }), 'Train on')),
        UI.note(s.random ? 'A new random maze every episode. The score is the share of 20 new mazes it finds the goal of, none of them seen in training: does what it learned carry over?' : 'Every episode is the maze chosen; the score is 1 when its best moves reach the goal.'),
        UI.field('Closer bonus', UI.seg([{ value: true, label: 'On' }, { value: false, label: 'Off' }], !!s.closer, (v) => set((q) => { q.closer = v; }), 'Bonus for getting closer')),
        // (a formula of your own pays instead of the whole table, this bonus included)
        UI.note(s.rFormula && !BT.compile(s.rFormula, NA.rlEnvs.REWARD_VARS.maze.map((v) => v[0])).error ? `Your formula, r = ${s.rFormula}, pays it instead of the reward table, this bonus included: put closer into the formula (as in goal - pit + 0.1closer) to pay for getting closer.` : s.closer ? `${amountText(typeof s.rCloser === 'number' ? s.rCloser : 0.1).replace('+', '')} for every step that brings it closer to the goal by the shortest safe way, and as much off for every step away. Reward shaping: the bonuses along any route to the goal add up to the same, so they hardly change which route is best (exactly so only when nothing is discounted, with γ = 1), but they say which way is warmer long before the goal is ever found.` : 'Rewards only at the end: +1 at the goal, −1 in a pit, and −0.02 a step. Random moves almost never find the goal of most mazes, so there is next to nothing to learn from.'),
        UI.field('Floor', UI.seg([{ value: 0, label: 'Dry' }, { value: 0.1, label: 'Slippery' }, { value: 0.25, label: 'Icy' }], s.slip || 0, (v) => set((q) => { q.slip = v; }), 'Floor')),
        UI.note(`${s.slip ? `On this floor ${s.slip >= 0.25 ? 'one move in four' : 'one move in ten'} goes sideways` : 'On a slippery floor one move in ten goes sideways (on ice, one in four)'}: a safe route keeps away from the pits.`),
        worldPick(L, s, 'Longest episode', 'maxSteps', [40, 80, 160], 80, (v) => `${v} steps`));
    }
    // the worlds of rl-worlds2.js and rl-worlds3.js: their choices, and what they mean as they are set now
    if (p.task === 'thermo') {
      const days = s.days || 1, warm = s.warm || 21, band = s.band || 1;
      out.push(seg('Weather', 'climate', [['mild', 'Mild'], ['cold', 'Cold']], 'mild'), seg('Insulation', 'insulation', [['good', 'Good'], ['poor', 'Poor']], 'good'),
        seg('Warm', 'schedule', [['day', 'All day'], ['home', 'When people are in', `${warm} °C from 6 to 8 in the morning and from 5 to 11 in the evening, ${warm - 5} °C the rest of the time`]], 'day'),
        worldPick(L, s, 'Warm to', 'warm', [19, 21, 23], 21, (v) => `${v} °C`, 'The temperature wanted while people are in'), worldPick(L, s, 'Comfortable within', 'band', [0.5, 1, 2], 1, (v) => `${fmt(v, v % 1 ? 1 : 0)} °C`, 'How near the target counts as comfortable'),
        worldPick(L, s, 'Radiator', 'radiator', [45, 60, 80], 60, (v) => `${({ 45: 'Small', 60: 'Medium', 80: 'Large' })[v] || 'Radiator'}: up to ${v} °C`), worldPick(L, s, 'Days', 'days', [1, 2, 3], 1, (v) => plural(v, 'day', 'days'), 'How long an episode lasts'),
        UI.note(`${days === 1 ? 'A day' : `${days} days`} of ${fmtInt(144 * days)} steps, 10 minutes each, from midnight. ${s.climate === 'cold' ? 'Cold: −2 °C outside on average' : 'Mild: 8 °C outside on average'}, coldest before dawn and warmest at three in the afternoon; ${s.insulation === 'poor' ? 'poor insulation lets the heat out two and a half times as fast' : 'good insulation keeps the heat in'}. ${s.schedule === 'home' ? `People are in from 6 to 8 in the morning and from 5 to 11 in the evening: ${warm} °C then and ${warm - 5} °C the rest of the time, so it has to start warming the room before they come in.` : `${warm} °C all day.`} The radiator is slow to warm up and slow to cool down: switched off, it goes on warming the room for a while. The score is the share of the time the room is within ${fmt(band, band % 1 ? 1 : 0)} °C of the target, over 3 runs.`));
    }
    if (p.task === 'hurdles') {
      const mine = s.obstacles === 'own', fits = !mine || (WE && WE.inUse('hurdles', s)), myBars = mine && Array.isArray(s.obst) && s.obst.some((q) => q[1]);
      out.push(UI.field('Course', UI.seg([['hurdles', 'Hurdles'], ['mixed', 'Hurdles and bars'], ['own', 'Your own', 'A course you lay out yourself']].map(([value, label, title]) => ({ value, label, title })), s.obstacles || 'hurdles', (v) => own('hurdles', v, (q) => { q.obstacles = v; }), 'Course')),
        editBtn('hurdles', mine ? 'Edit your course' : 'Lay out your own course'),
        ...(fits ? [] : [UI.note('Your course has obstacles past the end of this length, so it runs a random course instead: edit your course, or make the course longer.')]),
        worldPick(L, s, 'Length', 'course', [100, 150, 300], 150, (v) => `${v} m`),
        worldPick(L, s, 'Running speed', 'run', [6, 8, 10], 8, (v) => `${v} m/s`), worldPick(L, s, 'Jump', 'jump', [4.5, 5.5, 6.5], 5.5, (v) => `${({ 4.5: 'Low', 5.5: 'Normal', 6.5: 'High' })[v] || 'Jump'}: ${v} m/s up`, 'How fast it leaves the ground'),
        UI.note(`${s.course || 150} m at ${s.run || 8} m/s, ${mine && fits ? `your own ${s.obst.length} obstacle${s.obst.length === 1 ? '' : 's'}, the same every episode` : 'an obstacle every 7 to 14 m'}. A jump keeps its feet above a hurdle (0.7 m) only for a few metres, so it has to take off at the right moment: running faster or jumping lower makes that moment shorter${s.obstacles === 'mixed' || (myBars && fits) ? '; under a bar (its underside 1.2 m up) it has to duck, and a jump there hits it' : ''}. Touching anything ends the run. The score is the share of 5 courses it runs.`));
    }
    if (p.task === 'memory') {
      const lit = s.lit || 1, wait = s.wait || 5;
      out.push(worldPick(L, s, 'Corridor', 'length', [2, 4, 6, 10, 14], 6, (v) => `${v} squares`), worldPick(L, s, 'Lamp lit for', 'lit', [1, 2, 4], 1, (v) => plural(v, 'step', 'steps')), worldPick(L, s, 'Time to turn', 'wait', [2, 5, 10], 5, (v) => plural(v, 'step', 'steps'), 'How long it may stand at the junction'),
        UI.note(`The lamp is lit for the first ${lit === 1 ? 'step only' : plural(lit, 'step', 'steps')}. The robot walks on by itself, a square a step, and at the junction has ${plural(wait, 'step', 'steps')} to turn: +1 the way the lamp said, −1 the other way or for waiting too long. ${s.view === 'history' ? 'It sees the last 16 readings, so at the junction the lamp is still among them.' : `It sees only the present: ${lit >= (s.length || 6) ? 'here the lamp is still lit when it reaches the junction, so it needs no memory.' : 'by the junction the lamp is long dark, so whatever it learns, it is right half the time. The Last 16 readings view (below) gives it something to remember.'}`} The score is the share of 20 corridors it turns the right way in.`));
    }
    if (p.task === 'mountain') out.push(seg('Engine', 'engine', [['normal', 'Weak'], ['strong', 'Stronger']], 'normal'), longest([200, 500, 1000], 200),
      seg('Start', 'start', [['valley', 'In the valley', 'At rest near the bottom, every episode'], ['anywhere', 'Anywhere', 'A random place and speed every episode']], 'valley'),
      UI.note(`Too weak to drive up either slope: it has to rock, up the slope behind it and then forward, a little higher every time. Every step costs 1 until it reaches the flag, so a cart that never gets there learns nothing from its rewards: every try is as bad as the next. Evolution needs no reward on the way: the carts that happen to rock furthest are kept and bred. For the other methods, the bonus for gaining energy (in Its reward) gives learning something to go on (DQN learns it then), and so ${s.start === 'anywhere' ? 'does starting anywhere (as now: some starts are already near the flag, and what it learns there spreads back)' : 'can starting anywhere'}; a longer episode or a stronger engine alone does not. The score is the share of 10 starts it reaches the flag from.`));
    if (p.task === 'swingup') {
      const steps = s.maxSteps || 300;
      out.push(seg('Motor', 'motor', [['strong', 'Strong'], ['weak', 'Weak']], 'strong'), worldPick(L, s, 'Gravity', 'gravity', [3.7, 9.8, 24.8], 9.8, (v) => `${GRAVITY[v] || 'Gravity'}: ${v} m/s²`),
        seg('Start', 'start', [['down', 'Hanging'], ['anywhere', 'Anywhere'], ['up', 'Upright', 'Only the balancing is left to learn']], 'down'), worldPick(L, s, 'Friction', 'friction', [0, 0.1, 0.5], 0.1, (v) => (v ? String(v) : 'None'), 'How much the hinge slows it'), longest([200, 300, 600], 300),
        UI.note(`The motor turns it with ${s.motor === 'weak' ? '2.5' : '4'} against the ${fmt(s.gravity || 9.8, 1)} the weight pulls with when the pole is level${(s.gravity || 9.8) <= (s.motor === 'weak' ? 2.5 : 4) ? ': here it can lift it straight up' : ': it cannot lift it straight up, so it swings it higher and higher, pushing with each swing, then has to catch it upside down and keep it there'}. Within 15° of upright counts as balanced. The score is the share of the last ${fmtInt(steps / 2)} of ${fmtInt(steps)} steps it keeps it upright, from 3 starts.`));
    }
    if (p.task === 'arm') out.push(worldPick(L, s, 'Reach within', 'reach', [0.06, 0.12, 0.25], 0.12, (v) => `${Math.round(v * 100)} cm`, 'How near the target the hand has to stop'), worldPick(L, s, 'Motors', 'motors', [3, 6, 12], 6, (v) => ({ 3: 'Weak', 6: 'Normal', 12: 'Strong' })[v] || String(v), 'How hard each push turns a joint'), longest([60, 120, 240], 120),
      UI.note(`A 1 m upper arm and a 0.8 m forearm on a bench. Each step pushes the shoulder or the elbow one way or the other, or brakes both; the joints keep turning, slowed by friction, so it has to brake before it gets there${s.motors > 6 ? ', and the stronger the motors, the sooner' : ''}. Reaching is the hand within ${Math.round((s.reach || 0.12) * 100)} cm of the target and moving slowly. A new target every episode: the score is the share of 10 targets it reaches within ${fmtInt(s.maxSteps || 120)} steps.`));
    if (p.task === 'traffic') out.push(seg('Traffic', 'pattern', [['shifting', 'Rush hours'], ['steady', 'Steady']], 'shifting'), worldPick(L, s, 'How busy', 'busy', [0.7, 1, 1.3], 1, (v) => ({ 0.7: 'Quiet', 1: 'Normal', 1.3: 'Busy' })[v] || `× ${v}`, 'How often cars arrive'),
      worldPick(L, s, 'Amber', 'amber', [2, 4, 8], 4, (v) => plural(v, 'step', 'steps'), 'How long a switch shows amber, when nobody moves'), longest([200, 400, 800], 400),
      UI.note(`${s.pattern === 'steady' ? 'Cars come from every side at a steady rate, a little more often from the north and south.' : 'North–south is busy for the first half and east–west for the second, as at rush hours: the lights have to notice the change.'} On green a car crosses from each side every step; a switch shows amber for ${plural(s.amber || 4, 'step', 'steps')}, when nobody moves, so every switch costs time. The score is the share of cars through within 20 steps of arriving.`));
    if (p.task === 'walker') out.push(seg('Ground', 'ground', [['grippy', 'Grippy'], ['ice', 'Icy']], 'grippy'), worldPick(L, s, 'Gravity', 'gravity', [3.7, 9.8, 24.8], 9.8, (v) => `${GRAVITY[v] || 'Gravity'}: ${v} m/s²`),
      seg('Muscles', 'muscles', [['weak', 'Weak', 'They pull in and let out 12%'], ['normal', 'Normal', '18%'], ['strong', 'Strong', '25%']], 'normal'), longest([150, 300, 600], 300),
      UI.note(`Two stiff legs, a stiff body and two muscles, each crossing from a hip to the other foot. Each step it sets both muscles short or long, and they pull towards that length, swinging the legs. ${s.ground === 'ice' ? 'On ice a foot holds only while it presses down hard: push sideways and it slips.' : 'Its feet grip the ground.'} A clock that goes round every 20 steps is among its inputs, so a gait can be a rhythm. The score is the share of 10 m it walks, from 2 starts.`));
    if (p.task === 'trap') out.push(seg('Maze', 'maze', [['trap', 'Trap'], ['zigzag', 'Zigzag']], 'trap'), worldPick(L, s, 'Speed', 'speed', [2, 3, 5], 3, (v) => `${v} units a step`, 'How far a step forward takes it'),
      longest([0, 200, 400, 700, 1400], 0, (v) => (v ? `${fmtInt(v)} steps` : `The maze’s own: ${s.maze === 'zigzag' ? 700 : 400}`)),
      UI.note(`${s.maze === 'zigzag' ? 'Two walls: the way runs left, up the far side and left again, and the second wall is the trap.' : 'One wall, just under the goal: the way round is at the far end, which first leads away from the goal.'} By default it is paid for getting closer in a straight line, so it learns to head straight for the goal, and ends against the wall: the reward deceives it. Novelty search pays for ending up somewhere new instead (so can its reward: a part of the maze not yet visited); MAP-Elites keeps the best of every place it ends. The score is the share of 3 starts it reaches the goal from.`));
    if (p.task === 'chase') out.push(seg('Arena', 'map', [['open', 'Open'], ['pillars', 'Four pillars']], 'open'), seg('Chasers', 'chasers', [[1, '1'], [2, '2']], 1), seg('Chaser speed', 'speed', [['slow', 'Slower'], ['fast', 'Nearly as fast']], 'slow'),
      seg('Chasers aim', 'aim', [['straight', 'At you', 'Each chaser heads straight for where you are'], ['ahead', 'Where you are going', 'Each chaser heads for where you will be a moment later']], 'straight'), longest([200, 400, 800], 400),
      UI.note(`A walled arena 20 m across. It is faster than a chaser, but ${s.aim === 'ahead' ? 'a chaser that aims where it is going cuts the corners off every turn and is much harder to shake off' : 'running straight away from one that heads straight for it still ends in a corner'}: it has to circle round. Coins turn up one at a time. Caught, the game ends. The score is the share of ${fmtInt(s.maxSteps || 400)} steps it stays free, over 3 games.`));
    if (p.task === 'herd') out.push(UI.field('Sheep start', UI.seg([['meadow', 'Grazing', 'In the bottom left, a little apart: new places every game'], ['own', 'Where you put them', 'The same places every game']].map(([value, label, title]) => ({ value, label, title })), s.sheepAt === 'own' ? 'own' : 'meadow', (v) => own('herd', v, (q) => { q.sheepAt = v; }), 'Sheep start')),
      ...(s.sheepAt === 'own' ? [] : [seg('Sheep', 'sheep', [[3, '3'], [5, '5'], [7, '7']], 3)]), editBtn('herd', s.sheepAt === 'own' ? 'Move your sheep and dog' : 'Place the sheep yourself'),
      worldPick(L, s, 'Sheep run from', 'flee', [3, 5, 8], 5, (v) => `${v} m`, 'How near the dog comes before the sheep run'),
      worldPick(L, s, 'Dog speed', 'dog', [0.35, 0.45, 0.6], 0.45, (v) => ({ 0.35: 'Slow', 0.45: 'Normal', 0.6: 'Fast' })[v] || String(v)), longest([300, 600, 1200], 600),
      UI.note(`A field 24 m across, the pen in the top right corner, open at the bottom. Sheep graze, keep together and run from the dog when it comes within ${s.flee || 5} m; one that walks into the pen stays there. So the dog has to get behind them, on the side away from the gate, and push. The score is the share of the sheep penned in ${fmtInt(s.maxSteps || 600)} steps, over 3 games.`));
    return out;
  }
  /* A reward's amount: any number you type (Enter, or leaving the box, uses it), or a quick pick from the list
     beside it (the amount it starts with marked). */
  const amountText = (v) => (v > 0 ? '+' : v < 0 ? '−' : '') + String(+Math.abs(v).toPrecision(8));
  function rewardAmount(L, s, r, picks) {
    const cur = typeof s[r.key] === 'number' ? s[r.key] : r.def, put = (v) => L.edit((S) => { S.ps[S.problem][r.key] = v; });
    const inp = el('input', { type: 'text', inputmode: 'decimal', class: 'bd-amt-in', value: amountText(cur), spellcheck: 'false', autocomplete: 'off', 'aria-label': `${r.label}: the amount` });
    const use = () => {
      const t = inp.value.trim().replace(/[−–]/g, '-').replace(/,/g, '.').replace(/^\+/, ''), v = Number(t);
      if (!t || !Number.isFinite(v) || Math.abs(v) > 1000) { NA.ui.toast.show(`${r.label}: type a number from −1000 to 1000, such as −1 or 0.05.`); inp.value = amountText(cur); return; }
      if (v !== cur) put(v); else inp.value = amountText(cur);
    };
    inp.addEventListener('keydown', (e) => { if (e.key === 'Enter') { e.preventDefault(); use(); } });
    inp.addEventListener('change', use);
    const list = [...new Set([...(picks || []), r.def, cur])].sort((a, b) => b - a);
    const pick = el('select', { class: 'bd-amt-pick', 'aria-label': `Quick picks: ${r.label}` }, el('option', { value: '', disabled: true, selected: true }, '▾'),
      ...list.map((v) => el('option', { value: String(v) }, `${amountText(v)}${v === r.def ? ' (to start with)' : ''}`)));
    pick.addEventListener('change', () => { const v = Number(pick.value); pick.value = ''; if (Number.isFinite(v) && v !== cur) put(v); });
    const f = UI.field(r.label, el('span', { class: 'bd-amt-row' }, inp, pick), `To start with: ${amountText(r.def)}`);
    f.classList.add('bd-amt');
    if (cur !== r.def) f.classList.add('is-set');
    return f;
  }
  /* What the model is given: how it sees the world, which inputs (and extra ones) and with how much noise; then what
     it is paid, in a line, with the way to the Reward tab (worldReward) where that is set. */
  function worldSenses(L, p, s) {
    const E = NA.rlEnvs, set = (fn) => L.edit((S) => fn(S.ps[S.problem])), views = BT.VIEWS[p.task], ins = [];
    if (views) ins.push(UI.field('It sees', UI.seg(views, s.view, (v) => L.setWorldView(v), 'What it sees')), UI.note(`${(views.find((q) => q.value === s.view) || views[0]).title}.`));
    const groups = (E.INPUT_GROUPS[p.task] || {})[s.view], hide = Array.isArray(s.hide) ? s.hide : [];
    if (groups) {
      ins.push(el('div', { class: 'chip-row', role: 'group', 'aria-label': 'Inputs it is given' }, ...groups.map(([name]) => UI.chip(name, !hide.includes(name), (on) => {
        if (!on && groups.every(([g]) => g === name || hide.includes(g))) { NA.ui.toast.show('It needs at least one input.'); L.inspDirty = true; return; }
        set((q) => { const h = new Set(q.hide || []); if (on) h.delete(name); else h.add(name); q.hide = groups.map(([g]) => g).filter((g) => h.has(g)); });
      }, `${hide.includes(name) ? 'Not given' : 'Given'}: ${name}`))),
      UI.note(hide.length ? `It is not given ${hide.join(', ')}: the model has fewer inputs and has to do without them${p.task === 'pole' ? '. The Last 4 readings view gives it a history instead, from which a memory (a GRU or LSTM block) can work the speeds out' : ''}.` : 'Each chip is an input it is given. Switch one off and the model learns without it: can it still do the job?'));
    }
    // extra inputs, on top of what it sees (when that is a list of numbers)
    const extras = E.EXTRAS[p.task] || [], extra = Array.isArray(s.extra) ? s.extra : [], d = L.dataFor(), on = extras.filter((e) => extra.includes(e.name));
    if (extras.length && !d.error && d.input.kind === 'vector') {
      ins.push(el('div', { class: 'chip-row', role: 'group', 'aria-label': 'Extra inputs it can be given' }, ...extras.map((e) => UI.chip(`+ ${e.name}`, extra.includes(e.name), (yes) => set((q) => { const x = new Set(q.extra || []); if (yes) x.add(e.name); else x.delete(e.name); q.extra = extras.map((f) => f.name).filter((n) => x.has(n)); }), `${extra.includes(e.name) ? 'Given' : 'Not given'}: ${e.name}, ${e.what}`))));
      if (on.length) ins.push(el('ul', { class: 'bd-extras' }, ...on.map((e) => el('li', null, el('strong', { text: e.name }), `: ${e.what}`))));
      ins.push(UI.note(on.length ? `${on.length === 1 ? 'This extra input comes' : 'These extra inputs come'} after the others, ${fmtInt(d.input.dim)} in all.` : 'The chips with a + are extra inputs it can be given on top of what it sees: a hint, a clock, its last move, sometimes the answer itself, and one that is only noise. Switch one on and watch how much easier, or harder, learning gets.'));
    } else if (extras.length && !d.error) ins.push(UI.note('Extra inputs (its last move, the time left and more) can be added when it sees a list of numbers, not a picture or a history.'));
    ins.push(worldPick(L, s, 'Sensor noise', 'noise', [0, 0.05, 0.1, 0.3], 0, (v) => (v ? `± ${v}` : 'None'), 'Random noise on every input'),
      UI.note(s.noise ? `Every input, every step, is off by a random amount (a spread of ${s.noise}): real sensors are never exact.` : 'No noise: every input is exact.'));
    return [UI.section('Its inputs', ...ins), UI.section('Its reward', el('p', { class: 'bd-rew-line', text: L.rewardSummary() }),
      UI.note('What it is paid for, step by step: the goal it is given. Change the amounts, or write a formula of your own, in the Reward tab.'),
      el('div', { class: 'btn-row' }, el('button', { class: 'btn', type: 'button', onclick: () => L.openReward() }, 'Change the reward')))];
  }
  /* What it is paid, in the Reward tab: a table of amounts you type or pick, or a formula of your own (the table can
     be written as one, to start from), with the names it can use and what a random player earns with it. */
  function worldReward(L, p, s) {
    const E = NA.rlEnvs, set = (fn) => L.edit((S) => fn(S.ps[S.problem])), d = L.dataFor();
    // the rewards: quick picks for each amount (by world where one key means different things in different worlds)
    const vars = E.REWARD_VARS[p.task], names = vars.map((v) => v[0]), table = E.REWARDS[p.task].filter((r) => r.key !== 'rCloser' || s.closer);
    const CHOICES = { rPay: [0.5, 1, 2, 5], rGoal: [0.5, 1, 2, 5, 10], rPit: [0, -0.5, -1, -2, -5, -10], rStep: [0, -0.01, -0.02, -0.05, -0.1], rWall: [0, -0.05, -0.1, -0.5, -1], rNew: [0, 0.01, 0.05, 0.1, 0.5],
      rCloser: [0.05, 0.1, 0.2, 0.5], rUp: [0, 0.1, 1], rFell: [0, -1, -10, -100], rCatch: [0.5, 1, 2, 5], rMiss: [0, -0.5, -1, -2, -5], rMove: [0, -0.01, -0.05, -0.1, -0.5],
      rProgress: [0, 1, 10, 50], rCrash: [0, -0.5, -1, -5, -10], rLap: [0, 1, 5, 20], rHit: [0.5, 1, 2, 5], rHurt: [0, -0.5, -1, -2, -5], rShot: [0, -0.01, -0.05, -0.2], rLand: [0, 1, 2, 5, 10], rShape: [0, 0.5, 1, 2], rFuel: [0, -0.003, -0.01, -0.03],
      rOff: [0, -0.05, -0.1, -0.2, -0.5], rHeat: [0, -0.01, -0.02, -0.05], rBand: [0, 0.05, 0.1, 0.5], rRun: [0, 0.01, 0.05], rBump: [0, -0.5, -1, -2, -5], rFinish: [0, 1, 5, 10],
      rRight: [0.5, 1, 2], rWrong: [0, -0.5, -1, -2], rTick: [0, -0.1, -1], rFlag: [0, 1, 10, 100], rEnergy: [0, 10, 100, 1000], rHeight: [0, 0.01, 0.05, 0.1, 1], rUpright: [0, 0.01, 0.05, 0.1], rPush: [0, -0.001, -0.005, -0.01, -0.1],
      rReach: [0, 1, 5, 10], rApproach: [0, 0.5, 1, 5], rStraight: [0, 0.5, 1, 2], rFree: [0, 0.01, 0.02, 0.05], rCoin: [0, 0.5, 1, 2], rCaught: [0, -1, -5, -10], rWait: [0, -0.005, -0.01, -0.05], rSwitch: [0, -0.01, -0.05, -0.1],
      rForward: [0.5, 1, 5], rFall: [0, -1, -5], rPenned: [0.5, 1, 5], rNearer: [0, 0.05, 0.1, 0.5],
      rNoPay: [0, -0.5, -1], rMatch: [0, 0.5, 1], rRevisit: [0, -0.01, -0.05, -0.1], rNearPit: [0, -0.01, -0.05, -0.1], rTurn: [0, -0.001, -0.01, -0.05], rLost: [0, -0.5, -1],
      rLean: [0, -0.01, -0.05, -0.1], rLasted: [0, 1, 10, 100], rNear: [0, 0.25, 0.5], rUnder: [0, 0.01, 0.05, 0.1],
      rSpeed: [0, 0.001, 0.01, 0.05], rCenter: [0, -0.001, -0.01, -0.05], rNearWall: [0, -0.01, -0.05, -0.1], rSteer: [0, -0.001, -0.01], rBrake: [0, -0.001, -0.01],
      rClear: [0, 1, 5], rAim: [0, 0.001, 0.01, 0.05], rSee: [0, 0.001, 0.01], rWalk: [0, -0.001, -0.01], rOffPad: [0, 0.5, -0.5], rSpare: [0, 0.5, 1], rTilt: [0, -0.001, -0.01],
      rCold: [0, -0.05, -0.1, -0.2], rHot: [0, -0.05, -0.1, -0.2], rJump: [0, -0.01, -0.05], rDuck: [0, -0.001, -0.01], rAir: [0, -0.001, 0.001], rEarly: [0, -0.05, -0.1], rSpin: [0, -0.0001, -0.001],
      rPass: [0, 0.01, 0.05, 0.1], rLate: [0, -0.05, -0.1], rLong: [0, -0.001, -0.01], rRecord: [0, 0.5, 1, 2], rDist: [0, 0.001, 0.01], rAll: [0, 1, 5], rFlock: [0, -0.001, -0.01],
      'pole.rOff': [0, -0.1, -0.5, -1], 'arena.rMiss': [0, -0.01, -0.05, -0.2], 'hurdles.rClear': [0, 0.1, 0.5, 1], 'memory.rWait': [0, -0.05, -0.1], 'chase.rNear': [0, -0.01, -0.05, -0.1],
      'herd.rRun': [0, -0.01, -0.05], 'arm.rBump': [0, -0.01, -0.1, -0.5], 'trap.rBump': [0, -0.01, -0.05, -0.1, -1], 'chase.rWall': [0, -0.01, -0.05, -0.1] };
    const own = !!s.rFormula && !BT.compile(s.rFormula, names).error, rw = [], moved = table.filter((r) => typeof s[r.key] === 'number' && s[r.key] !== r.def);
    rw.push(UI.note(`What it is paid for, step by step. It learns to collect as much reward as it can, so the reward is the goal it is given. The score still measures the real goal (${E.WORLDS[p.task].goal}): a reward that pays for something else shows there.`));
    const grid = el('div', { class: 'bd-rewards' + (own ? ' is-off' : '') }, ...table.map((r) => rewardAmount(L, s, r, CHOICES[`${p.task}.${r.key}`] || CHOICES[r.key])));
    rw.push(grid);
    rw.push(UI.note(own ? 'Your formula below is in use instead of this table.' : 'Type any amount and press Enter, or pick one from the list beside it: more than 0 pays it for that, less than 0 costs it. They add up each step.'));
    // the table as a formula, to start one from: it pays exactly what the table does
    const asFormula = own ? null : tableAsFormula(p.task, s);
    if (!own) rw.push(el('div', { class: 'btn-row' }, moved.length ? el('button', { class: 'btn btn-quiet', type: 'button', onclick: () => set((q) => { for (const r of table) delete q[r.key]; }) }, `Back to the amounts it started with (${moved.length} changed)`) : null,
      asFormula ? el('button', { class: 'btn btn-quiet', type: 'button', title: `r = ${asFormula}`, onclick: () => {
        if (asFormula.length > 200) { NA.ui.toast.show('Written as a formula, this table is longer than 200 characters: set fewer amounts, or write a shorter formula of your own.'); return; }
        inp.value = asFormula; set((q) => { q.rFormula = asFormula; });
        NA.ui.toast.show('The table, written as a formula: it pays exactly what the table did. Now change it as you like.');
      } }, 'Write the table as a formula') : null));
    const inp = el('input', { class: 'lm-input', type: 'text', value: s.rFormula || '', spellcheck: 'false', maxlength: '200', placeholder: 'empty: the table above', 'aria-label': 'Your own reward, a formula' }), msg = el('p', { class: 'sec-note' });
    // a formula that does not work is never put to use: the last one that did (or the table) goes on paying
    const stored = s.rFormula ? BT.compile(s.rFormula, names) : null;
    const check = () => {
      const v = inp.value.trim(), c = v ? BT.compile(v, names) : null, fallback = stored && !stored.error && v !== s.rFormula ? ` Until it does, it goes on using r = ${s.rFormula}.` : ' Until it does, the table above pays it.';
      msg.textContent = !v ? (s.rFormula ? 'Press Enter to go back to the table.' : 'Or write the reward as a formula of the names below, and press Enter.') : c.error ? `${c.error}${fallback}` : v === s.rFormula ? 'In use: this is its reward now.' : 'Looks good: press Enter to use it.';
      msg.classList.toggle('lm-warn', !!(c && c.error));
      return c;
    };
    const use = () => { const v = inp.value.trim(), c = check(); if (!v) set((q) => { q.rFormula = ''; }); else if (!c.error && v !== s.rFormula) set((q) => { q.rFormula = v; }); };
    inp.addEventListener('input', check);
    inp.addEventListener('keydown', (e) => { if (e.key === 'Enter') use(); });
    inp.addEventListener('change', use);
    check();
    const EXAMPLES = { levers: ['paid', 'right', '2paid - 1', 'right ? 1 : -1'], maze: ['goal - pit', 'goal - pit + 0.05new', 'goal - pit - (wall ? 1 : 0.02)', 'goal - pit - 0.1nearpit', '-dist/10'],
      pole: ['1 - abs(angle)/12', '1 - abs(x)/2.4', '-10fell', 'abs(angle) < 3 ? 1 : 0', 'fell ? -100 : 1'], catch: ['caught - missed - 0.1moved', '1 - abs(dx)/4', 'missed - caught', 'near ? -0.5 : caught - missed', 'under ? 0.1 : 0'],
      track: ['10progress - crashed', '10progress - crashed - 0.02center', 'speed/24 - crashed', 'wall < 2 ? -0.1 : 10progress', '20lap - crashed'], arena: ['hit - hurt', 'hit - hurt - 0.05shot', 'hit - hurt + 0.02aimed', 'missed ? -0.1 : hit - hurt', 'hit + 0.01seen - hurt'],
      lander: ['landed - crashed + closer', '3landed - crashed + closer - 0.005engine', 'landed - crashed', 'h < 5 and speed > 3 ? -0.1 : closer', '-dist/30 - speed/10'],
      thermo: ['-off/10 - 0.01heating', 'inside', 'inside - 0.05heating', 'cold > 2 ? -1 : -off/10', '-abs(temp - target)/10'], hurdles: ['0.01 - hit + finished', 'finished - hit', 'cleared - 5hit', 'gap < 2 and air ? 0.05 : 0.01 - hit', 'dist/150 - hit'],
      memory: ['right - wrong - lost', 'right', 'right - wrong - lost - 0.1junction', 'early ? -0.1 : right - wrong', '2right - 1'], mountain: ['goal - 1', '100gained - 1', 'abs(speed)', 'speed > 0 ? height/10 : -0.01', 'goal'],
      swingup: ['height', 'upright', 'height + upright', 'upright ? 1 : height/10', '-abs(angle)/180'], arm: ['reached + closer', 'reached', 'reached - dist/100', 'dist < 0.3 ? closer - 0.01speed : closer', '-dist/10'],
      traffic: ['-0.01waiting', 'passed/10', '-0.01waiting - 0.05switched', 'longest > 20 ? -1 : -0.01waiting', '-longest/100'], walker: ['gained - fell', 'gained', 'gained - 5fell', 'abs(tilt) > 30 ? -0.1 : gained', 'gained - fell - abs(tilt)/3000'],
      trap: ['goal + closer', 'goal', 'goal + 0.1new', 'goal + record', 'bump ? -0.1 : goal + closer'], chase: ['0.01 + coin - 5caught', '0.01 - 5caught', 'dist/100 - 5caught', 'near ? -0.05 : 0.01 + coin', '0.01 + 2coin - 5caught'],
      herd: ['penned + 0.1nearer', 'penned', 'penned + 0.1nearer - dist/1000', 'dist > 8 ? -0.01 : penned + 0.1nearer', '-left/100'] };
    rw.push(el('label', { class: 'field bd-formula' }, el('span', { text: 'r =' }), inp), msg);
    if (stored && stored.error) rw.push(UI.note(`The formula it was given, r = ${s.rFormula}, does not work: ${stored.error} Until it does, the table above pays it.`, 'lm-warn'));
    // steps where the formula gave no number (∞, or 0 / 0): they paid 0
    const trouble = own ? E.formulaTrouble(p.task, s.rFormula) : null;
    if (trouble && trouble.bad) rw.push(UI.note(`Your formula gave no number ${trouble.bad === 1 ? 'once' : `${fmtInt(trouble.bad)} times`} so far, first when ${Object.entries(trouble.first).filter(([, v]) => v).map(([k, v]) => `${k} = ${fmt(v, Math.abs(v) < 10 && v % 1 ? 2 : 0)}`).slice(0, 4).join(', ') || 'every name was 0'} (for example a division by 0): those steps paid 0.`, 'lm-warn'));
    rw.push(UI.note('Write it with the names below, + − × / ^, brackets and functions such as abs(), sqrt(), min(a, b) and max(a, b). A comparison such as dist < 3 or wall == 1 is 1 when it is true and 0 when not; and, or and not combine them; c ? a : b is a when c is true and b when not, so wall ? -1 : 0 costs 1 for every bump. Each step pays what the formula comes to then.'),
      el('div', { class: 'chip-row' }, ...EXAMPLES[p.task].map((f) => el('button', { class: 'chip bd-tpl chip-mono', type: 'button', onclick: () => { inp.value = f; set((q) => { q.rFormula = f; }); } }, f))),
      el('dl', { class: 'bd-vars' }, ...vars.flatMap(([n, what]) => [el('dt', { class: 'mono', text: n }), el('dd', { text: what })])));
    // the size of an episode's return, against what it was with the reward it started with
    if (!d.error) {
      const s0 = Object.assign({}, s, { rFormula: '' });
      for (const r of E.REWARDS[p.task]) delete s0[r.key];
      const now = randomReturn(p.task, d.ws), was = randomReturn(p.task, BT.worldSettings(p, s0)), a = Math.abs(now), b = Math.abs(was);
      const say = (v) => fmt(v, Math.abs(v) >= 100 ? 0 : Math.abs(v) >= 1 ? 1 : 2), why = 'the gradient methods (DQN, REINFORCE, actor-critic, PPO) learn best with returns of about 1 to 100';
      const grown = a > 10 * Math.max(b, 0.01), shrunk = b > 10 * Math.max(a, 0.01);
      rw.push(UI.note(`A player moving at random earns about ${say(now)} an episode with this reward${now !== was ? ` (${say(was)} with the one it started with)` : ''}.${grown ? ` That is over 10 times as much: ${why}, which the network’s outputs have to reach.` : shrunk ? ` That is under a tenth as much: ${why}, and a reward this small hardly moves their outputs.` : ''}${grown || shrunk ? ' Evolution only compares scores, so it does not mind.' : ''}`, grown || shrunk ? 'lm-warn' : ''));
    }
    return [UI.section('Its reward', ...rw)];
  }
  /* What each amount of a reward table pays for, in the names a formula can use: the table written as a formula pays
     exactly what the table does (checked by playing every world both ways, step by step). */
  const REWARD_AS = {
    levers: { rPay: 'paid', rNoPay: 'unpaid', rMatch: 'right' },
    maze: { rGoal: 'goal', rPit: 'pit', rStep: '(1 - goal - pit)', rWall: 'wall', rNew: 'new', rCloser: 'closer', rRevisit: 'revisit', rNearPit: 'nearpit', rTurn: 'turned', rLost: 'lost' },
    pole: { rUp: '1', rFell: 'fell', rLean: 'abs(angle)', rOff: 'abs(x)', rSwitch: 'switched', rLasted: 'lasted' },
    catch: { rCatch: 'caught', rMiss: 'missed', rMove: 'moved', rNear: 'near', rUnder: 'under', rStep: '1' },
    track: { rProgress: 'progress', rCrash: 'crashed', rLap: 'lap', rStep: '1', rFinish: 'finished', rSpeed: 'speed', rCenter: 'center', rNearWall: '(wall < 2)', rSteer: 'abs(turn)', rBrake: 'brake' },
    arena: { rHit: 'hit', rHurt: 'hurt', rShot: 'shot', rStep: '1', rMiss: 'missed', rClear: 'cleared', rAim: 'aimed', rSee: 'seen', rWalk: 'moved', rTurn: 'turned', rBump: 'bump' },
    lander: { rLand: 'landed', rCrash: 'crashed', rShape: 'closer', rFuel: 'engine', rOffPad: 'offpad', rSpare: 'landed*fuel', rTurn: 'turn', rTilt: 'abs(tilt)', rStep: '1' },
    thermo: { rOff: 'off', rHeat: 'heating', rBand: 'inside', rCold: 'cold', rHot: 'hot', rSwitch: 'switched' },
    hurdles: { rRun: '1', rBump: 'hit', rFinish: 'finished', rClear: 'cleared', rJump: 'jumped', rDuck: 'ducking', rAir: 'air' },
    memory: { rRight: 'right', rWrong: '(wrong + lost)', rStep: '1', rWait: 'stood', rEarly: 'early' },
    mountain: { rTick: '1', rFlag: 'goal', rEnergy: 'gained', rHeight: 'height', rPush: 'pushed' },
    swingup: { rHeight: 'height', rUpright: 'upright', rPush: 'abs(push)', rSpin: 'abs(spin)', rSwitch: 'switched' },
    arm: { rReach: 'reached', rApproach: 'closer', rStep: '1', rPush: '(1 - braked)', rBrake: 'braked', rBump: 'bump' },
    traffic: { rWait: 'waiting', rSwitch: 'switched', rPass: 'passed', rLate: 'late', rLong: 'longest' },
    walker: { rForward: 'gained', rFall: 'fell', rStep: '1', rTilt: 'abs(tilt)', rHeight: 'height', rSwitch: 'switched' },
    trap: { rGoal: 'goal', rStraight: 'closer', rStep: '1', rBump: 'bump', rNew: 'new', rRecord: 'record', rTurn: 'turned' },
    chase: { rFree: '1', rCoin: 'coin', rCaught: 'caught', rNear: 'near', rDist: 'dist', rWall: 'wall', rPush: 'pushed' },
    herd: { rPenned: 'penned', rNearer: 'nearer', rStep: '1', rAll: 'last', rFlock: 'dist', rRun: 'ran' },
  };
  // the table of task with settings s as a formula: each amount that is not 0 times what it pays for (null: a term
  // with no name to write it in)
  function tableAsFormula(task, s) {
    const as = REWARD_AS[task] || {}, out = [];
    for (const r of NA.rlEnvs.REWARDS[task]) {
      if (r.key === 'rCloser' && !s.closer) continue;
      const a = typeof s[r.key] === 'number' ? s[r.key] : r.def, e = as[r.key];
      if (!a) continue;
      if (!e) return null;
      const n = String(+Math.abs(a).toPrecision(8));
      out.push(`${a < 0 ? '-' : '+'} ${e === '1' ? n : Math.abs(a) === 1 ? e : `${n}*${e}`}`);
    }
    return out.length ? out.join(' ').replace(/^\+ /, '').replace(/^- /, '-') : '0';
  }
  BL.tableAsFormula = tableAsFormula; // (for tests)
  /* What it is paid, in a line: your formula, or the table's amounts that are not 0 (the chip under the results'
     title, and the Input block's panel). */
  BL.rewardSummary = function () {
    const p = this.problem();
    if (!p || p.family !== 'control') return '';
    const E = NA.rlEnvs, s = this.ps(), names = E.REWARD_VARS[p.task].map((v) => v[0]);
    if (s.rFormula && !BT.compile(s.rFormula, names).error) return `r = ${s.rFormula}`;
    const paid = E.REWARDS[p.task].filter((r) => r.key !== 'rCloser' || s.closer).map((r) => [r, typeof s[r.key] === 'number' ? s[r.key] : r.def]).filter(([, v]) => v);
    return paid.length ? paid.map(([r, v]) => `${amountText(v)} ${r.label.charAt(0).toLowerCase()}${r.label.slice(1)}`).join(' · ') : 'nothing: every amount is 0';
  };
  BL.renderRewardTab = function (body) {
    const key = JSON.stringify([this.snapshot(), this.family()]);
    if (this._rewardKey === key && body.firstChild) return;
    this._rewardKey = key;
    clear(body);
    const p = this.problem();
    if (p.family !== 'control') { body.append(UI.note('A reward is what a model that learns from rewards is paid: pick a problem from the Learn from rewards group.')); return; }
    body.append(...worldReward(this, p, this.ps()));
  };
  // what a player moving at random earns in an episode with these settings: up to 4 episodes and 1,500 steps (kept)
  const RETURNS = new Map();
  function randomReturn(task, ws) {
    const key = task + JSON.stringify(ws);
    if (RETURNS.has(key)) return RETURNS.get(key);
    const W = NA.rlEnvs.make(task, ws), rng = new NA.RNG(99);
    let total = 0, eps = 0, ret = 0;
    W.reset(rng);
    for (let i = 0; i < 1500 && eps < 4; i++) { const t = W.step(rng.int(W.nA), rng); ret += t.r; if (t.done) { total += ret; eps++; ret = 0; W.reset(rng); } }
    const v = eps ? total / eps : ret;
    if (RETURNS.size > 80) RETURNS.clear();
    RETURNS.set(key, v);
    return v;
  }
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
  // grow and prune: what a new layer is, by how wide it starts (builder-evo.js EV.LAYER_SIZE)
  const GROW_LAYER_SAY = {
    one: 'A new layer starts with one neuron, passing on the signal the next layer leans on most; the rest is cut off. That is a bottleneck: unlike a new neuron it changes what the network does, and selection keeps it only if the child still does well. Later “+1 neuron” changes widen it without changing anything, and a layer goes when its last neuron is pruned.',
    some: 'A new layer starts with a few neurons, from 1 to one less than the numbers it reads, each passing on one of the signals the next layer leans on most: still a bottleneck, a looser one, kept only if the child still does well. A layer goes when its last neuron is pruned.',
    whole: 'A new layer starts as wide as what it reads, passing values straight through, so a child that grows a layer does exactly what its parent did (after a ReLU layer). This is the older way, here to compare: turn on “−1 layer” too and whole layers can also go at once.',
  };
  const GROW_WHERE_SAY = { any: 'A new layer goes between any two layers, at random.', first: 'A new layer goes straight after the input: a new first stage, the one that sees the raw numbers.', last: 'A new layer goes just before the output: a last stage that reads what all the others made.' };
  const GROW_PCT = (v) => `${v >= 0.095 ? Math.round(100 * v) : (100 * v).toFixed(1)}%`;
  /* Grow and prune's settings, the same for evolution (the Output block; o.sup false: the problem's settings) and for
     learning from examples (the Training tab; o.sup true: the training settings, whose older keys have other names):
     what it starts from; the changes it can make, their odds and how many children change; how new neurons and
     layers start and where layers go; what pruning takes; the largest network, the pressure to stay small, and
     protection for new shapes (evolution) or the networks side by side and how often they are reshaped (examples).
     o: { sup, s (the settings), set(key, value), can (the model can change shape) }. Each section says what its
     choices do, with the odds worked out. */
  BL.growSecs = function (o) {
    const EV = NA.bevo, s = o.s, sup = o.sup, g = EV.growOptions(s, sup), on = !!s.grow, K = sup ? { penalty: 'growPenalty', size: 'growWidth', mode: 'growNew' } : { penalty: 'penalty', size: 'layerSize', mode: 'newLayer' };
    const seg = (label, key, opts, cur, title) => UI.field(label, UI.seg(opts.map(([value, l, t]) => ({ value, label: l, title: t })), cur, (v) => o.set(key, v), title || label));
    const N = g.members, ga = !sup;
    const head = UI.section('Grow and prune', seg('Architecture', 'grow', [[false, 'Fixed'], [true, 'Grows and prunes']], on),
      on && o.can ? seg('Start from', 'growStart', [['built', 'Your network', 'The first networks are the one you built and copies of it'], ['small', 'No hidden layer', 'The first networks have no hidden layer: every neuron and layer they end with, evolution added']], g.start) : null,
      UI.note(!o.can ? (sup ? 'Only a model of Dense blocks on a list of numbers can change shape. Start from the Linear template, or one Dense block of 1 neuron, and let it grow.' : 'Only a model of Dense blocks on a list of numbers can change shape; yours keeps its shape and only its weights evolve. The Small MLP can grow, and so can the Linear template, from no hidden layer at all.')
        : !on ? `${sup ? `Turn it on and ${N} networks learn side by side from your model; every ${g.every} steps the worse half are replaced by copies of the better half, each with one change to its shape: a neuron more or less, or a new layer of one neuron. Start from one neuron on Two spirals and watch it grow.` : 'Turn it on, and a child may also gain or lose a neuron, or gain a new layer, as it is born. A new neuron starts with its outgoing weights at 0, so it does no harm until later mutations put it to work. A new layer starts with one neuron: a bottleneck that changes what the network does, so it is kept only if the child still plays well.'} Every part of it can be set: the changes it can make and how often, how new neurons and layers start, what pruning takes, the largest network, and more.`
          : g.start === 'small' ? `${sup ? `All ${N} networks start` : 'Every network of the first generation starts'} with no hidden layer: the inputs go straight to the outputs, as NEAT starts (Stanley and Miikkulainen, 2002). Whatever hidden neurons and layers ${sup ? 'the best' : 'the winner'} ends with, evolution added one change at a time and kept only because ${sup ? 'it learned better' : 'it played better'} with them.${sup ? ' The network you built is not among them: only its kind of blocks counts.' : ''}`
            : `The ${sup ? `${N} networks start as the one you built (and ${N - 1} fresh copies of its shape)` : 'first generation is the network you built and copies of it, moved at random'}: every change starts from its shape. The Architecture view shows the best network as it is now.`));
    if (!on || !o.can) return [head];
    // the changes it can make, their odds, and how many children change
    const OPS = [['neuron+', 'growAddNeuron', 'A hidden layer gains a neuron'], ['neuron-', 'growCutNeuron', 'A hidden layer loses a neuron (its last takes the layer with it)'], ['layer+', 'growAddLayer', 'A new hidden layer'],
      ['layer-', 'growCutLayer', 'A whole hidden layer goes at once, folded into the next'], ['act', 'growAct', `A hidden layer switches to another activation: ${EV.ACTS.map((a) => (ACTS[a] || {}).label || a).join(', ')}`]];
    const odds = EV.changeOdds(g), allowed = OPS.filter(([k]) => g.ops[k]), none = !allowed.length;
    const chips = el('div', { class: 'chip-row bd-grow-ops', role: 'group', 'aria-label': 'Changes it can make' }, ...OPS.map(([k, key, title]) => UI.chip(EV.CHANGES[k], g.ops[k], (v) => o.set(key, v), title)));
    const rateSay = { 0.1: 'One child in ten is', 0.25: 'A quarter of the children are', 0.5: 'Half the children are', 1: 'Every child is' }[g.rate] || `${GROW_PCT(g.rate)} of the children are`;
    const changes = UI.section('Changes to its shape', UI.field('It can make', chips),
      none ? null : seg('Leaning', 'growLean', [['grow', 'To grow', 'Gaining neurons and layers is likelier'], ['even', 'Even', 'The usual odds'], ['prune', 'To prune', 'Losing neurons and layers is likelier']], g.lean),
      none || sup ? null : seg('Children that change', 'growRate', [[0.1, '1 in 10'], [0.25, '1 in 4'], [0.5, 'Half'], [1, 'All']], g.rate, 'How many children are born with a change to their shape'),
      UI.note(none ? 'Every change is off, so its shape stays as it is: turn one on.'
        : `Of the changes it makes: ${allowed.map(([k]) => `${EV.CHANGES[k]} ${GROW_PCT(odds[k])}`).join(', ')}${g.ops['layer+'] ? ' (with no hidden layer, a new layer is the only change it can make)' : ''}. ${{ grow: 'Leaning to grow: networks get bigger sooner, and the size penalty has more to hold back.', even: 'Even: a neuron more is likeliest, then a neuron less, then a new layer, so it grows slowly and prunes what does not pay.', prune: 'Leaning to prune: it keeps networks small and tries out removing parts, so it suits a network you built too big.' }[g.lean]} ${sup ? `Each copy that replaces one of the worse half is born with one change; then its gradient steps teach it to use what it has.` : `${rateSay} born with one change; the rest only have their weights mutated. More changes explore more shapes, but fewer children get to tune the shape they have.`}${g.ops.act ? ' A new activation keeps the layer’s weights but changes what it computes: ReLU cuts negatives to 0, tanh and sigmoid squash into a range, leaky ReLU and ELU let a little of the negatives through.' : ''}`));
    // how new neurons and layers start, and where layers go
    const grows = g.ops['layer+'] || g.ops['neuron+'];
    const parts = !grows ? null : UI.section('New neurons and layers',
      g.ops['layer+'] ? seg('New layers start with', K.size, [['one', 'One neuron'], ['some', 'A few, at random'], ['whole', 'A whole layer']], g.size) : null,
      g.ops['layer+'] ? seg('Their weights', K.mode, [['identity', 'Passing values on'], ['random', 'At random']], g.mode, 'A new layer’s weights') : null,
      g.ops['layer+'] ? seg('New layers go', 'growWhere', [['any', 'Anywhere'], ['first', 'After the input'], ['last', 'Before the output']], g.where) : null,
      g.ops['neuron+'] ? seg('New neurons start', 'growNeuron', [['silent', 'Silent', 'Outgoing weights 0: it changes nothing at first'], ['random', 'At random', 'Outgoing weights at random: it joins in at once']], g.neuron) : null,
      UI.note([g.ops['layer+'] ? `${GROW_LAYER_SAY[g.size] || GROW_LAYER_SAY.one}${g.mode === 'random' ? ` Its weights start at random, so right after it is added the network forgets much of what it did: ${sup ? 'it has to learn again' : 'children that grow a layer rarely win'}.` : ''} ${GROW_WHERE_SAY[g.where]}` : '',
        g.ops['neuron+'] ? (g.neuron === 'random' ? 'A new neuron’s outgoing weights start at random: it joins in at once, so a child that grows does differently from its parent, often worse at first.' : `A new neuron’s outgoing weights start at 0: it changes nothing until ${sup ? 'gradient steps' : 'mutations'} put it to work, so growing never costs a ${sup ? 'network' : 'child'} anything at first.`) : ''].filter(Boolean).join(' ')));
    // what pruning takes
    const prunes = g.ops['neuron-'] || g.ops['layer-'];
    const pruning = !prunes ? null : UI.section('Pruning',
      g.ops['neuron-'] ? seg('Pruning takes', 'growPrune', [['weak', 'The weakest', 'The neuron the next layer listens to least'], ['random', 'One at random']], g.prune) : null,
      UI.note([g.ops['neuron-'] ? (g.prune === 'random' ? 'Any neuron, at random: sometimes one the network relies on, so pruning is more often a step back, and selection has to catch it.' : 'The weakest neuron: the one the next layer listens to least (the smallest outgoing weights, in absolute size), as magnitude pruning does. Taking it out changes the least.') : '',
        g.ops['neuron-'] && g.size !== 'whole' ? 'When a layer’s last neuron goes, the layer goes too, folded into the next.' : '',
        g.ops['layer-'] ? '“−1 layer” takes a whole layer out at once: the next layer’s weights are multiplied through (W₂·W₁), as if the activation let every value pass, so it changes what the network does unless that activation is linear.' : ''].filter(Boolean).join(' ')));
    // its limits and the pressure to stay small; evolution: protecting new shapes; examples: how many networks, how often
    const pen = g.penalty, penSay = pen === 'off' ? 'With no size penalty nothing holds them back: they tend to swell with weights that do nothing.'
      : sup ? `The penalty adds ${EV.PENALTY_EX[pen]} to a network’s validation loss for every 100 weights, so the smaller of two equal networks wins.` : `The penalty takes ${GROW_PCT(EV.PENALTY[pen])} of a good episode’s return off the fitness for every 100 weights, so the smaller of two equal networks wins.`;
    const limits = UI.section('Limits and pressure',
      seg('Hidden layers, at most', 'growMaxLayers', [[2, '2'], [4, '4'], [6, '6'], [8, '8']], g.maxLayers),
      seg('Neurons in a layer, at most', 'growMaxUnits', [[16, '16'], [32, '32'], [48, '48'], [64, '64']], g.maxUnits),
      seg('Size penalty', K.penalty, [['off', 'None'], ['light', 'Light'], ['strong', 'Strong']], pen),
      ga ? seg('Protect new shapes', 'growProtect', [[false, 'Off'], [true, `For ${EV.PROTECT.gens} generations`]], g.protect) : null,
      sup ? seg('Networks side by side', 'growSize', [[4, '4'], [6, '6'], [10, '10']], N) : null,
      sup ? seg('A generation every', 'growEvery', [[50, '50 steps'], [150, '150'], [500, '500']], g.every, 'Steps between one reshaping and the next') : null,
      UI.note([`No network grows past ${g.maxLayers} hidden layer${g.maxLayers === 1 ? '' : 's'} of ${g.maxUnits} neurons: a change that would is not made.`, penSay,
        ga ? (g.protect ? `A new shape usually starts a little worse (a new neuron does nothing yet, a new layer is a bottleneck). For ${EV.PROTECT.gens} generations after a shape appears, its best member carries on unchanged (up to ${EV.PROTECT.most} shapes at a time), so mutations get time to tune it before it has to beat the rest: a simple form of NEAT’s speciation, which protects new structure the same way.`
          : 'Off: a new shape has to beat the rest at once, and most die in their first generation, before mutations can tune them. Turn it on to give them a few generations.')
          : `More networks try more shapes at once, and each step takes longer. A generation every 50 steps reshapes often, but a copy has had little time to learn its new shape; every 500, slowly but fairly.`].join(' ')));
    return [head, changes, parts, pruning, limits].filter(Boolean);
  };
  // learning from rewards: the output block says what the outputs mean and how they learn
  /* Learning from rewards: how it learns, in the Training tab (where problems with answers keep theirs too): the method
     and its settings, evolution's (grow and prune among them), the discount, exploration, normalised rewards and
     keeping the best network. The Output block says what the outputs mean and points here. */
  BL.controlLearnSecs = function (d) {
    const s = this.ps(), M = NA.brl.METHODS, set = (fn) => this.edit((S) => fn(S.ps[S.problem])), m = s.method, evo = NA.brl.isEvo(m), def = NA.brl.defaults(d.world), EV = NA.bevo;
    // a choice among a few values, the one in use from your settings or the method's default
    const pick = (label, key, opts, title) => UI.field(label, UI.seg(opts.map(([value, l, t]) => ({ value, label: l, title: t })), key === 'grow' ? !!s.grow : s[key] ?? def[key], (v) => set((q) => { q[key] = v; }), title || label));
    const evoSecs = [];
    if (evo) {
      const axes = (NA.rlEnvs.WORLDS[d.world] || {}).axes;
      evoSecs.push(UI.section(m === 'me' ? 'MAP-Elites' : m === 'es' ? 'Evolution strategies' : m === 'cem' ? 'Cross-entropy method' : m === 'novelty' ? 'Novelty search' : m === 'erl' ? 'ERL: the population' : 'Evolution',
        pick(m === 'me' ? 'New networks a step' : m === 'cem' ? 'Networks a generation' : 'Population', 'pop', [[12, '12'], [24, '24'], [48, '48'], [96, '96']]),
        m === 'es' ? pick('Nudge size σ', 'esSigma', [[0.02, '0.02'], [0.05, '0.05'], [0.1, '0.1']]) : m === 'cem' ? pick('Starting spread σ', 'cemSigma', [[0.1, '0.1'], [0.3, '0.3'], [0.5, '0.5']]) : pick('Mutation', 'sigma', [[0.05, '0.05'], [0.2, '0.2'], [0.5, '0.5']], 'Mutation size'),
        m === 'es' ? pick('Step size', 'esLr', [[0.01, '0.01'], [0.03, '0.03'], [0.1, '0.1']]) : null,
        m === 'cem' ? pick('Refit to', 'cemElite', [[0.1, 'Best 10%'], [0.2, 'Best 20%'], [0.3, 'Best 30%']], 'The elite: the share of each generation the bell curve is refitted to') : null,
        m === 'cem' ? pick('Spread floor', 'cemFloor', [[0.01, '0.01'], [0.05, '0.05'], [0.1, '0.1']], 'The least spread any weight keeps') : null,
        m === 'novelty' ? pick('Chosen for', 'novMix', [['novelty', 'Novelty only'], ['half', 'Half novelty, half reward']], 'What a member is chosen by') : null,
        m === 'novelty' ? pick('Nearest behaviours', 'novK', [[5, '5'], [10, '10'], [20, '20']], 'Novelty: the average distance to this many of the nearest behaviours') : null,
        UI.note(m === 'es' ? 'Each generation tries the network nudged in half as many random directions as the population, each both ways (plus and minus). Ranks, not raw scores, weigh each direction, and Adam steps the network toward the better ones. It needs no backpropagation, only scores; but it moves one network, so on a hard world it can stall at the first obstacle where a population would have spread out to look for a way past.'
          : m === 'me' ? `A grid of 12 × 12 styles: ${axes ? axes.join(' across, ') + ' down' : 'two numbers the world measures'}. Each step plays this many new networks (random at first, then mutated copies of elites picked at random); each goes to the cell of its style and stays if it beats the one there. ${Click()} a cell in the results to watch that style’s best.`
            : m === 'cem' ? `Each generation draws this many networks from the bell curve: every weight its middle plus its spread times a random number. The best ${Math.round(100 * (s.cemElite ?? def.cemElite))}% set the new middle (their average) and the new spread (how much they disagree), never below the floor. A small floor settles quickly; a bigger one keeps looking. The middle is the network on show. It suits small networks: the more weights, the more networks it needs to draw.`
              : m === 'novelty' ? `Its behaviour: ${axes ? axes.join(' and ') : 'two numbers the world measures'}. A member’s novelty is the average distance from its behaviour to the ${s.novK ?? def.novK} nearest, among its generation and an archive that keeps the 2 most novel of every generation, so a behaviour once seen stops being new. ${(s.novMix ?? def.novMix) === 'half' ? 'Half novelty, half reward: each member is ranked both ways and the two ranks averaged, so it explores but still leans toward the goal.' : 'Novelty only: the reward plays no part in choosing parents, which is what lets it walk past dead ends that look like progress.'} The network on show is the best by score found so far. Mutation: as in the genetic algorithm, one weight in ten of each child moves by about this much.`
                : m === 'erl' ? 'Each generation plays every member, and every move goes into the DQN’s replay memory. One weight in ten of each child moves by a random amount of about this size, as in the genetic algorithm.'
                  : 'Each generation plays every member; one weight in ten of each child moves by a random amount of about this size. Small mutations refine, big ones explore. A bigger population searches more widely, and each generation takes longer.')));
      if (m === 'erl') evoSecs.push(UI.section('ERL: the DQN', pick('Steps a generation', 'erlSteps', [[50, '50'], [200, '200'], [500, '500']], 'The DQN’s gradient steps after each generation'),
        pick('Joins every', 'erlEvery', [[1, 'generation'], [5, '5 generations'], [10, '10']], 'How often the DQN’s network takes the worst member’s place'),
        pick('Learning rate', 'erlLr', [[0.0003, '0.0003'], [0.001, '0.001'], [0.003, '0.003']], 'The DQN’s learning rate (Adam)'),
        UI.note(`Between generations the DQN takes ${s.erlSteps ?? def.erlSteps} gradient steps on batches of 32 moves drawn from the memory, toward reward + γ × the next state’s value (a target network, refreshed every 500 steps, says what that is worth). Every ${(s.erlEvery ?? def.erlEvery) === 1 ? 'generation' : `${s.erlEvery ?? def.erlEvery} generations`} its network takes the place of the population’s worst member: if it plays well, it breeds.`)));
      if (m === 'ga' || m === 'novelty' || m === 'erl') {
        const sel = s.select || def.select, cr = s.cross || def.cross, mu = s.mutate || def.mutate, can = this.evolvable();
        evoSecs.push(UI.section('Choosing parents', pick('Parents', 'select', [['tournament', 'Tournament'], ['truncation', 'Top fifth'], ['rank', 'By rank']]),
          UI.note(`${EV.SELECT[sel]}. ${{ tournament: 'A little luck lets weaker members breed now and then, which keeps the population varied.', truncation: 'Only the best breed: quick at first, but the population soon looks all alike and can get stuck.', rank: 'Every member can breed, the better the likelier: gentle pressure, and more variety.' }[sel]} The best two always carry on unchanged.`)),
          UI.section('Mixing parents', pick('Mix', 'cross', [['uniform', 'Weight by weight'], ['neuron', 'Neuron by neuron'], ['none', 'Not at all']]),
            UI.note(`${EV.CROSS[cr]}. ${{ uniform: 'Fast mixing, but a neuron can end up with half its weights from one parent and half from the other, which may make sense to neither.', neuron: 'Each neuron is a working unit, so this mixes whole features: a child gets some of each parent’s neurons as they were.', none: 'Mutation alone: every child is a copy of one parent, nudged. Two networks can do the same job with their neurons in a different order, so mixing them can break both: on the race track and the rocket, copies alone found their way sooner.' }[cr]} Only parents of the same shape can be mixed.`)),
          UI.section('Mutation', pick('Kind', 'mutate', [['plain', 'Plain'], ['safe', 'Safe']]),
            UI.note(mu === 'safe' ? 'Safe mutations: before nudging, it measures how much the moves depend on each weight (the gradient of every output, over moments it lived through) and divides each nudge by that. Weights the moves hang on barely move, the others move more, with the same total nudge as plain: children stay close to what worked. Lehman and others, 2018.' : 'Plain: one weight in ten moves by a random amount, whatever it does. A nudge to a weight the moves depend on can wreck a good network.')),
          ...(m === 'ga' ? this.growSecs({ sup: false, s, can, set: (k, v) => set((q) => { q[k] = v; }) }) : []));
      }
    }
    return [UI.section('Learns by', UI.select(methodChoices(d.world, m).filter((k) => k !== 'neat' || k === m || NA.brl.neatOk(d.input)).map((k) => ({ value: k, label: M[k].label })), m, (v) => this.setMethod(v)), UI.note(M[m].hint), (m === 'me' || m === 'novelty') && d.world === 'levers' ? UI.note(m === 'me' ? 'The levers have no styles of play to tell apart: MAP-Elites needs a world with episodes of many moves.' : 'The levers have no behaviour to tell apart: novelty search needs a world with episodes of many moves, so here it chooses by reward, as the genetic algorithm does.') : null),
      ...this.teachSecs(d),
      ...evoSecs,
      (!evo || m === 'erl') && d.world !== 'levers' && m !== 'bc' ? UI.section('Discount', UI.field('Each step later counts ×', UI.seg([0, 0.9, 0.95, 0.97, 0.99].map((v) => ({ value: v, label: String(v) })), s.gamma, (v) => set((q) => { q.gamma = v; }), 'Discount')),
        UI.note('How much a reward one step later is worth: with 0.9, a reward ten steps away counts about a third. At 0 only the very next reward counts, so a move that pays off later is never learned.')) : null,
      !evo ? UI.section('At the end', UI.field('Keep', UI.seg([{ value: true, label: 'Its best network' }, { value: false, label: 'Its last' }], s.keepBest !== false, (v) => set((q) => { q.keepBest = v; }), 'Keep the best network')),
        UI.note(s.keepBest !== false ? 'Learning from rewards is unstable: a network that played well can play worse a little later. When a run ends below its best score, it ends with the weights it had at its best (the score of its best moves is measured every so often). The Python does the same.'
          : 'It ends with the weights it has at the last step, however it played before.')) : null,
      m === 'a2c' || m === 'ppo' || m === 'pref' || m === 'gauss' ? UI.section('Rewards as it learns', UI.field('Normalise rewards', UI.seg([{ value: true, label: 'On' }, { value: false, label: 'Off' }], s.normReward !== false, (v) => set((q) => { q.normReward = v; }), 'Normalise rewards')),
        UI.note(s.normReward !== false ? 'Each reward it learns from is divided by how much its returns have varied so far (the running spread of the discounted return), and kept within ±10: a world that pays in hundreds learns like one that pays in ones. The return and the score shown stay the world’s own.'
          : 'Off: it learns from the rewards as the world pays them. Returns in the hundreds are hard to learn: the critic’s output has to reach them, and its big errors crowd out the policy’s share of every step.')) : null,
      m === 'ppo' || m === 'pref' || m === 'gauss' ? UI.section('PPO’s steps', pick('Clip', 'ppoClip', [[0.1, '0.1'], [0.2, '0.2'], [0.3, '0.3']], 'How far a move’s probability may move from the policy that played it before the clip stops the push'),
        pick('Passes over each batch', 'ppoEpochs', [[1, '1'], [4, '4'], [10, '10']]), pick('GAE λ', 'ppoLam', [[0.9, '0.9'], [0.95, '0.95'], [0.99, '0.99'], [1, '1']], 'How far ahead its advantages look: 1 the whole return, lower trusts the critic sooner'),
        pick('Entropy bonus', 'ppoEnt', [[0, 'none'], [0.01, '0.01'], [0.05, '0.05']], 'A little reward for keeping its choices open'),
        UI.note(`Each batch of 128 ${m === 'gauss' ? 'tries' : 'moves'} is gone over ${s.ppoEpochs ?? def.ppoEpochs} time${(s.ppoEpochs ?? def.ppoEpochs) === 1 ? '' : 's'} in minibatches; once a ${m === 'gauss' ? 'try’s density' : 'move’s probability'} has moved ${Math.round(100 * (s.ppoClip ?? def.ppoClip))}% from the policy that played it, the clip stops pushing it. More passes squeeze more from each batch and risk going too far; a wider clip lets each step change more. GAE λ trades the critic’s guesses (low) against the returns themselves (1): less noise, or less bias. The entropy bonus keeps it trying${m === 'gauss' ? ' (a tenth of it, for bell curves)' : ''}.`)) : null,
      m === 'dqn' ? UI.section('Memory and targets', pick('Replay memory', 'dqnMem', [[1000, '1,000'], [10000, '10,000'], [50000, '50,000']], 'How many past moves it keeps to learn from'),
        pick('Target refreshed every', 'dqnTarget', [[100, '100'], [500, '500'], [2000, '2,000']], 'How many updates between copies of the network into the target network'),
        pick('Double DQN', 'dqnDouble', [[true, 'On'], [false, 'Off']], 'The network picks the next move and the target network says what it is worth'),
        pick('Dueling', 'dqnDueling', [[false, 'Off'], [true, 'On']], 'One more output, the value of the state: each move then needs only its advantage over the others'),
        pick('Looks ahead', 'dqnN', [[1, '1 step'], [3, '3 steps'], [5, '5 steps']], 'n-step returns: the rewards of its next n moves, then the value of where it got to'),
        UI.note(`It learns from ${fmtInt(s.dqnMem ?? def.dqnMem)} past moves at most, a batch at a time. Its targets come from a copy of the network refreshed every ${fmtInt(s.dqnTarget ?? def.dqnTarget)} updates: a target that moved with every step would chase itself. ${(s.dqnDouble ?? def.dqnDouble) ? 'Double DQN: the network picks the next move and the copy values it, so a move both happen to overrate is not taken at its word.' : 'Without double DQN the copy both picks and values the next move, and overrates whatever it happens to overrate.'} ${(s.dqnDueling ?? def.dqnDueling) ? 'Dueling: one output is the value of the state and the others each move’s advantage, so what a state is worth is learned once for all its moves. ' : ''}${(s.dqnN ?? def.dqnN) > 1 ? `Each move is valued from its next ${s.dqnN ?? def.dqnN} rewards and where they led: rewards travel back faster.` : ''}`)) : null,
      m === 'sac' || m === 'td3' ? UI.section('Before it learns', pick('Random tries first', 'warm', [[0, 'none'], [1000, '1,000'], [5000, '5,000']], 'Moves made at random before it learns, to fill its memory with something to learn from'),
        UI.note('Its replay memory starts empty: random tries first give its critics a spread of moves to learn from, and the score only starts to move after them.')) : null,
      NA.rlx && NA.brl.canModel(d.input, m) ? NA.rlx.worldModelSection(this, d, pick) : null,
      NA.rlx && NA.brl.pbtOk(m) ? NA.rlx.pbtSection(this, d, pick) : null,
      NA.brl.canChase(d.world, m) ? UI.section('The chaser', pick('The chaser', 'chaserLearns', [[false, 'Follows its rule'], [true, 'Learns too']], 'Whether the first chaser follows its rule or learns as the runner does'),
        UI.note(s.chaserLearns ? `Predator and prey: the first chaser learns too, a network of its own (DQN), paid for catching the runner and for closing in, while the runner learns to get away. Each gets better against the other as it is now: an arms race, as in self-play. Its score is still measured against the chaser that follows its rule.${this.agent && this.agent.pp && !this.stale && this.agent.pp.caught() !== null ? ` Lately the learning chaser has caught it in ${fmtPct(this.agent.pp.caught(), 0)} of its games.` : ''}` : 'The chasers head for the runner by a rule. With Learns too, the first one learns as the runner does: predator and prey, each learning against the other.')) : null,
      NA.brl.hasCurriculum(d.world, m) ? UI.section('Curriculum', pick('Start', 'curriculum', [['off', 'At full difficulty'], ['on', 'Easy first']], 'Whether the world starts easier and gets harder as it succeeds'),
        UI.note(s.curriculum === 'on' ? `Easy first: ${NA.brl.CURRICULUM[d.world].say}. Each time 3 in 4 of its last 10 episodes or more succeed, the world gets a quarter harder, up to full difficulty; its score is always measured at full difficulty.${this.agent && this.agent.curr && !this.stale ? ` Now: level ${this.agent.curr.level * 4} of 4${this.agent.curr.level >= 1 ? ', full difficulty' : ''}.` : ''}` : 'It learns in the world as it is from the first step. Easy first would start it in an easier world and make it harder as it succeeds: the way people learn, and often quicker for networks too.')) : null,
      m === 'dqn' ? UI.section('Exploration', UI.field('Random moves for', UI.select([[0, 'none'], [300, '300 moves'], [1500, '1,500 moves'], [3000, '3,000 moves'], [4000, '4,000 moves'], [10000, '10,000 moves']].map(([value, label]) => ({ value, label })), s.explore, (v) => set((q) => { q.explore = +v; }))),
        UI.note('Its chance of a random move falls from 1 to 0.05 over this many moves. Without exploring it only ever tries what already looks best, and may never find out that something else pays more.')) : null].filter(Boolean);
  };
  // learning from rewards, the Output block: what its outputs mean, and where its learning is set
  BL.renderControlOutput = function (body, d) {
    const m = this.ps().method, M = NA.brl.METHODS, evo = NA.brl.isEvo(m);
    const what = m === 'dqn' ? `${d.nA} Q-values, one per action: what it expects to collect from here on if it takes that action now. It plays the largest.`
      : m === 'erl' ? `${d.nA} scores, one per action, and it plays the largest. In the population nothing says what they should be: the weights that play best are kept. The DQN learns its outputs as Q-values, what each action is expected to collect from here on.`
      : evo ? `${d.nA} scores, one per action, and it plays the largest. Nothing says what they should be: the weights that play best are kept.`
        : `${d.nA} scores that softmax turns into the probability of each action${d.output.k > d.nA ? ', and one more output: the critic’s value of the state, what it expects to collect from here on' : ''}. While it learns it samples its moves from the probabilities; its best moves take the likeliest.`;
    body.append(el('div', { class: 'bd-insp-head', style: { '--bk': 'var(--s2)' } }, el('span', { class: 'bd-card-ico' }, icon('output')), el('div', null, el('h3', { class: 'bd-insp-title', text: 'Output: what comes out' }), el('div', { class: 'sec-note', text: d.error ? '' : outputText(d, this.setup.loss) }))),
      UI.section('What its outputs mean', UI.note(`${what} The added dense layer is built for you, sized to the world.`)),
      UI.section('How it learns', UI.note(`It learns by ${M[m].label}. How it learns, and every setting of it${m === 'ga' ? ', grow and prune among them' : ''}, is in the Training tab, as for problems with answers.`),
        el('div', { class: 'btn-row' }, el('button', { class: 'btn', type: 'button', onclick: () => this.openTrain('Learns by') }, 'Open the Training tab'))));
  };
  /* The Training tab, opened (on a phone, the settings screen), at a section when given its title: scrolled to it and
     marked for a moment, so you see where you have landed. */
  BL.openTrain = function (title) {
    if (!this.tabs) return;
    if (NA.mobile && NA.mobile.mobile) NA.mobile.go('insp');
    this.tabs.select('train');
    if (title) setTimeout(() => this.flashSection(title), 30);
  };
  BL.flashSection = function (title) {
    const sec = [...this.tabs.body.querySelectorAll('.insp-sec')].find((x) => { const t = x.querySelector('.sec-title'); return t && t.textContent === title; });
    if (sec) flash(sec);
  };
  BL.renderLossSettings = function (body) {
    const fam = this.family(), L = this.setup.loss, d = this.dataFor(), set = (fn) => this.edit((s) => fn(s.loss));
    if (fam === 'control' && !d.error) { this.renderControlOutput(body, d); return; }
    const own = !d.error && NA.bgen && NA.bgen.lossNote(d);
    if (own) { body.append(el('div', { class: 'bd-insp-head', style: { '--bk': 'var(--s2)' } }, el('span', { class: 'bd-card-ico' }, icon('output')), el('div', null, el('h3', { class: 'bd-insp-title', text: 'Output and loss' }), el('div', { class: 'sec-note', text: outputText(d, L) }))), UI.section('How it learns', UI.note(own))); return; }
    const ids = BT.lossesFor(this.problem(), d), value = d.output && d.output.kind === 'value';
    const EXPLAIN = {
      xent: 'Cross-entropy: −log of the probability given to the right answer. Confident mistakes cost a lot, and the error signal is simply probability − target. The standard choice for classes and tokens.',
      mse: value ? 'Squared error: ½(prediction − target)². Big misses cost far more than small ones, so outliers pull hard.' : 'Squared error between the softmax probabilities and the one-hot target. It still learns, but slowly when the model is confidently wrong, because the softmax flattens the error signal there.',
      hinge: 'Hinge (multi-class SVM): the right class must beat every other by a margin of 1; once it does, that example stops contributing at all.',
      mae: 'Absolute error |prediction − target|: every miss pulls with the same strength, so outliers matter less and the fit follows the median.',
      huber: 'Huber: squared error for small misses, absolute error beyond δ. Smooth near the target and robust to outliers.',
      mdn: 'A mixture of bell curves (a mixture density network, Bishop 1994): the model gives m weights, m middles and m spreads, and pays −log of how likely the target is under their mixture. Where several answers are right it can put a curve on each, where squared error would give their average.',
      gauss: 'Gaussian: for each number the model gives a middle μ and s, the log of a spread squared, and pays ½(s + (y − μ)²·e^(−s)), the negative log-likelihood of the target under that bell curve. A wide spread costs s; a miss costs less the wider the spread, so it learns to say how unsure it is.',
    };
    body.append(el('div', { class: 'bd-insp-head', style: { '--bk': 'var(--s2)' } }, el('span', { class: 'bd-card-ico' }, icon('output')), el('div', null, el('h3', { class: 'bd-insp-title', text: 'Output and loss' }), el('div', { class: 'sec-note', text: d.error ? '' : outputText(d, L) }))),
      UI.section('Loss function', UI.seg(ids.map((id) => ({ value: id, label: NA.bx.LOSSES[id].label })), L.id, (v) => set((q) => { q.id = v; }), 'Loss'), UI.note(EXPLAIN[L.id] || ''),
        L.id === 'xent' ? UI.slider({ label: 'Label smoothing', min: 0, max: 0.3, step: 0.05, value: L.smooth || 0, format: (v) => fmt(v, 2), onChange: (v) => set((q) => { q.smooth = v; }) }) : null,
        L.id === 'huber' ? UI.slider({ label: 'Huber δ', min: 0.05, max: 1, step: 0.05, value: L.delta || 0.2, format: (v) => fmt(v, 2), onChange: (v) => set((q) => { q.delta = v; }) }) : null),
      this.isClassicSetup() ? UI.section('Classic models', UI.note('A classic model has its own way to learn: squared error for numbers; for classes, log loss (boosting and logistic regression) or the purity of each tree leaf (trees and forests). The loss chosen above is used by networks.')) : null,
      UI.section('The output layer', UI.note(value ? (d.output.k > 1 ? `${d.output.k} linear neurons: any numbers can come out.` : 'One linear neuron: any number can come out.') : fam === 'sequence' ? 'At every position, one score per token in the vocabulary; softmax turns them into the probability of each next token.' : L.id === 'hinge' ? 'One score per class, used as it is: the hinge loss asks the right class to beat the others by a margin. The added dense layer is built for you, sized to the problem.' : 'One score per class; softmax turns the scores into probabilities that add up to 1. The added dense layer is built for you, sized to the problem.')));
  };

  BL.renderTrainTab = function (body) {
    // (learning from rewards: the method's settings are here too, so they redraw it; grow and prune asks whether the model can change shape)
    const ctl0 = this.family() === 'control';
    const key = JSON.stringify([this.setup.train, this.run && this.run.n, this.run && this.run.step, this.stale, this.isClassicSetup() && this.snapshot(), ctl0 && this.ps(), ctl0 && this.setup.problem, ctl0 && this.evolvable(), ctl0 && this.endTest && [this.endTest.run, this.endTest.done], ctl0 && [this.startFrom, this.runs.map((r) => (r.net ? r.n : 0)).join(), this.agent && this.agent.prefs ? this.agent.prefs.length : 0, (this.demosFor(this.dataFor()) || { acts: [] }).acts.length > 0]]);
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
    const T = this.setup.train, set = (fn) => this.edit((s) => fn(s.train)), ctl = this.family() === 'control', ga = ctl && NA.brl.isEvo(this.ps().method), dd = ctl ? this.dataFor() : null;
    clear(body);
    const lr = UI.slider({ label: 'Learning rate', min: -4, max: 0, step: 0.05, value: Math.log10(T.lr), format: (v) => fmtLr(Math.pow(10, v)), onChange: (v) => set((t) => { t.lr = +Math.pow(10, v).toPrecision(2); }) });
    const P = this.parts;
    P.live = el('div', { class: 'tiles' });
    P.grad = el('canvas');
    const story = this.insightsSection();
    if (story) body.append(story);
    P.sweep = null;
    body.append(...[P.live,
      // learning from rewards: the method and its settings first (the Output block only says what the outputs mean)
      ...(ctl && !dd.error && NA.rlx ? [NA.rlx.endingsSection(this, dd)] : []),
      ...(ctl && !dd.error ? this.controlLearnSecs(dd) : []),
      ga ? UI.section(this.ps().method === 'erl' ? 'Evolution and a DQN' : 'No gradients', UI.note(this.ps().method === 'erl' ? 'The population needs no optimizer, learning rate or batches: each generation plays every member and the next is made from the better ones. Its DQN does take gradient steps, with Adam at its own learning rate, set above with the rest of its settings.' : 'Evolution needs no optimizer, learning rate or batches: each generation plays every member and the next is made from the better ones (evolution strategies step by their own size, set above).'),
        UI.field('Generations per press of Train', UI.select(stepChoices(true, T.steps), T.steps, (v) => set((t) => { t.steps = +v; })))) : null,
      ...(!ctl && !this.isClassicSetup() ? this.growSection() : []),
      ga ? null : UI.section('Optimizer', UI.seg(Object.entries(OPT_LABEL).map(([value, label]) => ({ value, label })), T.opt, (v) => set((t) => { t.opt = v; }), 'Optimizer'),
        UI.note({ sgd: 'Plain gradient descent: step downhill by learning rate × gradient.', momentum: 'Keeps a running velocity (0.9 of the last one plus the new gradient): faster along consistent directions, damped zig-zags.', adam: 'Adapts the step for every weight from running averages of its gradient and squared gradient. Usually the easiest to tune.', adamw: 'Adam with weight decay applied directly to the weights (decoupled), the usual choice for transformers.' }[T.opt]),
        lr, UI.note('How big each step is. Too small: slow. Too big: the loss jumps around or blows up (the run is marked diverged).')),
      ctl ? null : UI.section('Find a learning rate', P.sweep = el('div', { class: 'bd-sweep' })),
      ga ? null : UI.section('Batches and schedule',
        UI.field('Batch size', UI.seg([8, 16, 32, 64, 128].map((v) => ({ value: v, label: String(v) })), T.batch, (v) => set((t) => { t.batch = +v; }), 'Batch size')),
        UI.field('Steps per press of Train', UI.select(stepChoices(false, T.steps), T.steps, (v) => set((t) => { t.steps = +v; }))),
        UI.field('Learning-rate schedule', UI.select(Object.entries(SCHED_LABEL).map(([value, label]) => ({ value, label })), T.sched, (v) => set((t) => { t.sched = v; }))),
        UI.note(T.steps ? 'Train runs this many steps; Continue then runs as many again, so a run can go on as long as you like. Cosine and step schedules lower the learning rate over each stretch, for a finer finish.' : 'Training goes on until you pause. Schedules need a fixed number of steps, so the learning rate stays as set.')),
      ga ? null : UI.section('Regularisation and stability',
        UI.field('Weight decay', UI.select([0, 0.0001, 0.001, 0.01, 0.05].map((v) => ({ value: v, label: v ? String(v) : 'none' })), T.wd, (v) => set((t) => { t.wd = +v; }))),
        UI.field('Gradient clipping', UI.select([[0, 'off'], [0.5, '0.5'], [1, '1'], [5, '5']].map(([value, label]) => ({ value, label })), T.clip, (v) => set((t) => { t.clip = +v; }))),
        UI.note('Weight decay pulls weights toward 0 (simpler models, less overfitting). Clipping caps the size of a step, which rescues a learning rate that is a little too high.')),
      UI.section('Starting point',
        UI.field('Initialisation', UI.select([['auto', 'automatic (He or Xavier)'], ['he', 'He'], ['xavier', 'Xavier'], ['small', 'tiny (N(0, 0.02))']].map(([value, label]) => ({ value, label })), T.init, (v) => set((t) => { t.init = v; }))),
        UI.field('Random seed', el('span', { class: 'bd-seed' }, el('span', { class: 'mono', text: String(T.seed) }), el('button', { class: 'btn btn-quiet', type: 'button', title: 'Another random start', onclick: () => set((t) => { t.seed = 1 + Math.floor(Math.random() * 9999); }) }, 'New seed'))),
        UI.note('The same setup with another seed gives different starting weights (and shuffles): train twice to see how much luck matters.')),
      NA.shelf ? NA.shelf.backgroundSection() : null,
      UI.section('Gradient size by layer', el('div', { class: 'chart-box' }, P.grad), UI.note(ga ? 'A genetic algorithm takes no gradient steps, so these stay empty.' : 'How strongly the last step changed each layer. Bars that shrink toward the input mean vanishing gradients; huge ones mean the steps are too big.'))].filter(Boolean));
    this.renderSweep();
    this.updateTrainLive();
  };
  /* Grow and prune while it learns: 6 networks (unless set) side by side from your model, the worse half replaced
     every 150 steps (unless set) by copies of the better half, each with one change to its shape (problems with
     answers; Dense blocks on numbers). Its sections (BL.growSecs), or none. */
  BL.growSection = function () {
    const d = this.dataFor();
    if (d.error || (d.output && d.output.kind === 'tokens') || (NA.bgen && NA.bgen.owns(d.family))) return [];
    const secs = this.growSecs({ sup: true, s: this.setup.train, can: this.evolvable(), set: (k, v) => this.edit((s) => { s.train[k] = v; }) });
    // (while it learns: how a generation is judged, after the settings' own notes)
    if (this.setup.train.grow && this.evolvable()) secs[0].append(UI.note(`Every ${NA.bevo.growOptions(this.setup.train, true).every} steps each network is judged by its validation loss (finer than accuracy, so a network that has just grown is not tied with its parent), plus the size penalty, and the worse half become copies of the better half, weights and all. The best so far is the one on show, in the results and the Architecture view.`));
    return secs;
  };
  const fmtLr = (v) => (v >= 0.01 ? fmt(v, 3) : v.toExponential(1).replace('e-', ' × 10⁻').replace(/(\d)$/, (d) => '⁰¹²³⁴⁵⁶⁷⁸⁹'[d]));
  BL.updateTrainLive = function () {
    const P = this.parts, run = this.run;
    if (!P || !P.live) return;
    const p = run && run.last;
    clear(P.live).append(UI.tile('Run', run ? String(run.n) : '—', run ? (this.stale ? 'setup edited since' : run.status) : 'not trained'), UI.tile(this.unitOf(run) === 'generation' ? 'Generation' : 'Step', run ? fmtInt(run.step) : '0', run && run.limit && !this.stale ? `of ${fmtInt(run.limit)}` : this.setup.train.steps ? `${fmtInt(this.setup.train.steps)} a press` : 'until you pause'),
      UI.tile('Learning rate now', fmtLr(this.lrAt(run ? run.step : 0))),
      run && run.family === 'control' ? UI.tile('Score', p && p.vm !== null && p.vm !== undefined ? fmtPct(p.vm, 0) : '—', run.bestVm !== null && run.bestVm !== undefined ? `best ${fmtPct(run.bestVm, 0)} at step ${fmtInt(run.bestAt)}` : null)
        : UI.tile(`Val. ${this.metricName()}`, p && p.vm !== null && p.vm !== undefined ? (BT.isValue(run.family) ? fmt(p.vm, 3) : fmtPct(p.vm, 1)) : '—', p && p.tm !== null && p.tm !== undefined ? `training ${BT.isValue(run.family) ? fmt(p.tm, 3) : fmtPct(p.tm, 1)}` : null));
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
    // (a run that learns from rewards is scored on its best moves, not on validation examples)
    if (P.chartTitle) P.chartTitle.textContent = fam === 'control' ? 'Score of each run (its best moves)' : 'Validation score of each run';
    P.chartBox.hidden = !series.length;
    if (series.length) { P.chart.frame(); P.chart.set({ series, yFixed: this.valued() ? null : [0, 1] }); }
    P.chartNote.textContent = series.length ? 'Every press of Train after a change starts a new run from fresh weights, so you can compare. Tick runs to show them on the chart.' : `No runs on ${this.problem().label.toLowerCase()} yet: press Train, then change something and train again to compare.`;
    if (P.shelf && NA.shelf) NA.shelf.render(this, P.shelf);
    const key = JSON.stringify([pid, this.runs.map((r) => [r.n, r.status, r.show, r.last && r.last.vm, r.step, (r.events || []).length]), this.cmp]);
    if (key === this._runsKey) return;
    this._runsKey = key;
    if (P.guess && NA.banalyse) NA.banalyse.renderGuess(this, P.guess);
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
        r.events && r.events.length ? el('div', { class: 'bd-run-events' }, ...r.events.slice(-4).map((e) => el('span', { class: 'bd-event', text: `${this.unitOf(r)} ${fmtInt(e.step)}: ${e.what.join(', ')}` })), r.events.length > 4 ? el('span', { class: 'bd-event', text: `${r.events.length - 4} earlier` }) : null) : null,
        el('div', { class: 'bd-run-meta' }, el('span', { text: `${fmtInt(r.step)} ${this.unitOf(r)}${r.step === 1 ? '' : 's'} · ${fmtInt(r.params)} parameters · ${r.seconds < 10 ? r.seconds.toFixed(1) : Math.round(r.seconds)} s${r.last && r.last.val !== null ? ` · val. loss ${fmt(r.last.val, 4)}` : ''}` }),
          r.guess !== undefined ? el('span', { class: 'bd-run-guess', text: `you guessed ${BT.isValue(r.family) ? fmt(r.guess, 2) : fmtPct(r.guess, 0)}` }) : null,
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
      const P = BC.pathOf ? BC.pathOf(ch.id) : null, nx = P && BC.nextIn(P, done), msg = `Challenge complete: ${ch.title}. ${done.size} of ${BC.LIST.length} done.${nx ? ` Next in the path ${P.title}: ${BC.byId[nx].title}.` : P ? ` That completes the path ${P.title}.` : ''}`;
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
    const run = ch && this.run && this.run.cfg.problem === ch.problem ? this.run : null, vm = run ? (run.family === 'control' ? run.bestVm : run.last ? run.last.vm : null) : null;
    const params = this.check && this.check.ok ? this.check.params : null, ownData = !ch || BC.sameData(ch, this.setup);
    const key = JSON.stringify([ch && ch.id, [...done], run && run.n, vm, run && run.styles, params, this.setup.train.opt, !!this.setup.train.grow, here, this._hint, ownData, ch && ch.goals.some((g) => g.steps !== undefined) && run ? run.step : 0, this.setup.blocks.map((b) => b.type).join()]);
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
        label = fam === 'control' ? `A score of at least ${scoreText(fam, g.metric)} at some point in a run` : `Validation ${this.metricName(fam)} of at least ${scoreText(fam, g.metric)}`;
        now = vm === null || vm === undefined ? 'not trained yet' : `last run: ${fam === 'control' ? 'best ' : ''}${scoreText(fam, vm)}`;
        ok = vm !== null && vm !== undefined && vm >= g.metric;
      } else if (g.method !== undefined) {
        const m = (this.setup.ps[ch.problem] || {}).method;
        label = `Learn by ${NA.brl.METHODS[g.method].label}`;
        now = m ? `now ${NA.brl.METHODS[m].short}` : '';
        ok = m === g.method;
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
      } else if (g.styles !== undefined) {
        label = `Find at least ${g.styles} of the 144 styles of play`;
        now = run && run.styles !== undefined ? `last run: ${run.styles}` : run ? 'last run: none (only MAP-Elites keeps styles)' : 'not trained yet';
        ok = !!run && run.styles >= g.styles;
      } else if (g.grow !== undefined) {
        const on = fam === 'control' ? !!(this.setup.ps[ch.problem] || {}).grow : !!this.setup.train.grow;
        label = fam === 'control' ? 'Let evolution grow and prune the network' : 'Let it grow and prune its own layers (the Training tab)';
        now = on ? 'on' : 'off';
        ok = on === g.grow;
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
    // paths first: the challenges in an order that builds one idea on another
    if (BC.PATHS) {
      this.chList.append(el('h3', { class: 'zoo-group', text: 'Paths' }));
      for (const P of BC.PATHS) {
        const got = P.ids.filter((id) => done.has(id)).length, nx = BC.nextIn(P, done), next = nx ? BC.byId[nx] : null;
        this.chList.append(el('div', { class: 'bd-ch-item bd-ch-path' + (got === P.ids.length ? ' is-won' : '') },
          el('div', { class: 'bd-ch-item-text' },
            el('div', { class: 'bd-ch-item-title' }, got === P.ids.length ? el('span', { class: 'bd-ch-tick', text: '✓' }) : null, P.title, el('span', { class: 'bd-ch-item-prob', text: `${got} of ${P.ids.length} done` })),
            el('p', { class: 'bd-ch-item-brief', text: P.blurb }),
            el('div', { class: 'bd-ch-steps', 'aria-hidden': 'true' }, ...P.ids.map((id) => el('span', { class: 'bd-ch-step' + (done.has(id) ? ' is-done' : '') + (id === nx ? ' is-next' : ''), title: BC.byId[id].title, text: done.has(id) ? '✓' : '' })))),
          el('button', { class: 'btn' + (next ? ' btn-primary' : ''), type: 'button', 'aria-label': next ? `Next in ${P.title}: ${next.title}` : `Start ${P.title} again`, onclick: () => this.startChallenge(next ? next.id : P.ids[0]) }, next ? `Next: ${next.title}` : 'Start again')));
      }
    }
    // by level, each level once (the list adds challenges level by level, topic by topic)
    const levels = [...new Set(BC.LIST.map((q) => q.level))], ordered = levels.flatMap((lv) => BC.LIST.filter((q) => q.level === lv));
    let level = null;
    for (const ch of ordered) {
      if (ch.level !== level) {
        level = ch.level;
        const all = BC.LIST.filter((q) => q.level === level), got = all.filter((q) => done.has(q.id)).length;
        this.chList.append(el('h3', { class: 'zoo-group', text: `${level} · ${got} of ${all.length} done` }));
      }
      const fam = BC.family(ch), won = done.has(ch.id);
      const goals = ch.goals.map((g) => (g.metric !== undefined ? `${this.metricName(fam)} ≥ ${scoreText(fam, g.metric)}` : g.params !== undefined ? `≤ ${fmtInt(g.params)} parameters` : g.steps !== undefined ? `within ${fmtInt(g.steps)} steps`
        : g.classic !== undefined ? (g.classic ? 'a classic model' : 'a network') : g.dense !== undefined ? `keep ${g.dense} dense layers` : g.method !== undefined ? `by ${NA.brl.METHODS[g.method].short}` : g.styles !== undefined ? `at least ${fmtInt(g.styles)} of the 144 styles of play` : g.grow !== undefined ? 'grow and prune on'
        : g.opt !== undefined ? `keep ${OPT_LABEL[g.opt]}` : '')).filter(Boolean).join(' · ');
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
    // learning from rewards: the score now and at its best, and the return of its episodes, in place of losses
    const rows = fam === 'control' ? [
      ['Score', L(a).vm, L(b).vm, score, scoreD, 'up', 5e-4],
      ['Best score', a.bestVm, b.bestVm, score, scoreD, 'up', 5e-4],
      ['Return per episode', L(a).ret, L(b).ret, (v) => fmt(v, 2), (v) => fmt(v, 2), 'up', 5e-3],
    ] : [
      [`Validation ${this.metricName()}`, L(a).vm, L(b).vm, score, scoreD, 'up', 5e-4],
      ['Validation loss', L(a).val, L(b).val, (v) => fmt(v, 4), (v) => fmt(v, 4), 'down', 5e-5],
      ['Training loss', L(a).train, L(b).train, (v) => fmt(v, 4), (v) => fmt(v, 4), 'down', 5e-5],
    ];
    rows.push(
      ['Parameters', a.params, b.params, fmtInt, fmtInt, null, 0],
      ['Steps', a.step, b.step, fmtInt, fmtInt, null, 0],
      ['Time', a.seconds, b.seconds, (v) => `${v.toFixed(1)} s`, (v) => `${v.toFixed(1)} s`, null, 0.05]);
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
    else if (t.action === 'poleGA') { this.setProblem('pole'); this.edit((s) => { s.ps.pole.view = 'state'; }); this.applyTemplate(BT.TEMPLATES.find((q) => q.id === 'mlp')); this.setMethod('ga'); this.play(); }
    else if (t.action === 'mazeShaping') {
      this.setProblem('rlmaze'); this.applyTemplate(BT.TEMPLATES.find((q) => q.id === 'mlp'));
      this.edit((s) => { Object.assign(s.ps.rlmaze, { view: 'around', maze: 'pits', random: false, closer: false, method: 'a2c' }); s.train.steps = 2000; });
      this.play();
      // when it stops: the bonus on, and a new run to compare
      const off = NA.app.on('training', () => { if (this.training) return; off(); if (!this.run || this.run.status !== 'done' || this.setup.problem !== 'rlmaze') return; this.edit((s) => { s.ps.rlmaze.closer = true; }); NA.ui.toast.show('Now with the bonus for every step closer: run 2 starts. Compare the two in Runs.'); this.play(); });
    }
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
