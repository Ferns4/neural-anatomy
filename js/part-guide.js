/* part-guide.js — the guide to every part of the model builder: a small animated picture of what the part does,
   then what it is for, how it computes step by step, its maths, how many numbers it learns, when to use it, what to
   watch out for, and the same thing in PyTorch or scikit-learn. Opened by clicking a part in the palette (with a
   button that adds it) and folded under the settings of a selected block. */
(function (NA) {
  'use strict';
  const { el, theme, clamp } = NA;

  const G = {
    dense: {
      what: 'The basic layer of a neural network. Every neuron looks at every input, weighs each one, adds a bias and bends the total with an activation function. A layer of many neurons computes many such features at once.',
      how: ['Multiply each input by its weight and add them up, plus a bias: one number z per neuron.', 'Bend z with the activation function (ReLU keeps positives and zeroes negatives).', 'Pass the results on: they are the next layer’s inputs.'],
      math: ['z = W·x + b      (W has one row of weights per neuron)', 'a = f(z)'],
      params: 'inputs × neurons weights + one bias per neuron',
      use: ['The default building block for points, tables and curves.', 'Two or three layers of 16–64 neurons solve most small problems.', 'On a sequence it is applied to each position separately.'],
      watch: ['Without an activation, stacking layers adds nothing: it is all one straight line.', 'Too many neurons on little data memorises the noise (overfitting).'],
      code: 'nn.Linear(in_features, units)   # then an activation, e.g. nn.ReLU()', try: 'spiral',
    },
    neuron: {
      what: 'A single neuron: the smallest part of a network. On its own it draws one straight boundary; dropped onto a dense layer it adds one more neuron to it.',
      how: ['Weigh each input and add them up with a bias.', 'Bend the sum with its activation function.'],
      math: ['z = w₁x₁ + w₂x₂ + … + b', 'a = f(z)'],
      params: 'one weight per input, plus a bias', use: ['To see exactly what one neuron can and cannot do.', 'To grow a layer one neuron at a time.'], watch: ['One neuron alone cannot solve XOR: no straight line separates it.'],
      code: 'nn.Linear(in_features, 1)', try: 'xor',
    },
    gated: {
      what: 'A layer with two paths from the same input: one carries values, the other becomes a gate between 0 and 1 (or a smooth Swish) that scales each value. The network learns to switch features on and off depending on the input. SwiGLU is the feed-forward layer of LLaMA and many other language models.',
      how: ['Project the input twice: values v = W₁x + b₁ and gate inputs g = W₂x + b₂.', 'Bend the gate: σ(g) for GLU, Swish(g) for SwiGLU, GELU(g) for GEGLU.', 'Multiply them, number by number.'],
      math: ['out = (W₁x + b₁) ⊙ gate(W₂x + b₂)'], params: '2 × (inputs × units + units)',
      use: ['In place of a dense layer when features should depend on each other multiplicatively.', 'Inside transformer-like models.'], watch: ['Twice the weights of a dense layer of the same width.'],
      code: 'class GLU(nn.Module):\n    def __init__(self, d_in, units):\n        super().__init__()\n        self.v, self.g = nn.Linear(d_in, units), nn.Linear(d_in, units)\n    def forward(self, x):\n        return self.v(x) * F.silu(self.g(x))   # SwiGLU', try: 'moons',
    },
    activation: {
      what: 'Bends every number with one function. Activations are what make a network more than a straight line: ReLU, tanh, sigmoid, GELU and others each bend differently.',
      how: ['Apply the function to every number, one at a time.'], math: ['ReLU(z) = max(0, z)', 'tanh(z) = (eᶻ − e⁻ᶻ) / (eᶻ + e⁻ᶻ)', 'σ(z) = 1 / (1 + e⁻ᶻ)'], params: 'none',
      use: ['After a layer created with a linear activation.', 'To try different shapes of bend on the same layer.'], watch: ['Sigmoid and tanh flatten out for big inputs, so their gradients vanish in deep stacks.', 'ReLU neurons can die: stuck at 0 for every input.'],
      code: 'nn.ReLU()   # or nn.Tanh(), nn.Sigmoid(), nn.GELU()', try: 'circle',
    },
    dropout: {
      what: 'While training, switches off a random share of the numbers passing through (sets them to 0) and scales the rest up to keep the total the same. The network cannot lean on any one neuron, so it learns spread-out, sturdier features. When the model is used, dropout does nothing.',
      how: ['For every number, draw a coin: with probability p set it to 0.', 'Scale the kept numbers by 1/(1 − p).', 'When the model is used: pass everything through unchanged.'],
      math: ['training: y = x · m / (1 − p),  m ~ Bernoulli(1 − p)', 'when used: y = x'], params: 'none',
      use: ['Against overfitting, after dense layers or in a transformer.', 'Rates of 0.1–0.5.'], watch: ['Slows learning: the training loss stays higher.', 'Pointless on a model that is already too small.'],
      code: 'nn.Dropout(p=0.2)', try: 'moons',
    },
    noise: {
      what: 'While training, adds a little random noise to every number. Like dropout, it stops the model relying on exact values and smooths what it learns; on inputs it is a simple form of data augmentation.',
      how: ['Draw a standard normal number for every value.', 'Add σ times it while training; do nothing when the model is used.'], math: ['training: y = x + σ·ε,  ε ~ N(0, 1)', 'when used: y = x'], params: 'none',
      use: ['Right after the input on small, noisy datasets.', 'σ around 0.05–0.3 of the inputs’ spread.'], watch: ['Too much noise drowns the signal.'],
      code: 'class GaussianNoise(nn.Module):\n    def forward(self, x):\n        return x + 0.1 * torch.randn_like(x) if self.training else x', try: 'moons',
    },
    layernorm: {
      what: 'Rescales the numbers of each example (each position of a sequence) to mean 0 and spread 1, then applies a learned scale and shift. It keeps signals in a healthy range through many layers; every transformer uses it.',
      how: ['Take the mean and standard deviation of the example’s numbers.', 'Subtract the mean and divide by the standard deviation.', 'Multiply by a learned gain γ and add a learned bias β, per feature.'],
      math: ['x̂ = (x − mean(x)) / √(var(x) + ε)', 'y = γ ⊙ x̂ + β'], params: '2 × features (γ and β)',
      use: ['Between layers of deep networks and transformers.', 'Works the same in training and use, at any batch size.'], watch: ['It removes the overall size of an example: if that matters, keep a path around it (a residual).'],
      code: 'nn.LayerNorm(features)', try: 'reverse',
    },
    batchnorm: {
      what: 'Rescales each feature to mean 0 and spread 1 using the numbers of the whole batch (for pictures: each channel over all pixels), then applies a learned scale and shift. It made very deep convolutional networks trainable. When the model is used, running averages from training replace the batch statistics.',
      how: ['For each feature, take its mean and variance over the batch.', 'Normalise every value of that feature with them.', 'Scale by γ and shift by β (learned).', 'Keep running averages of the means and variances for when the model is used.'],
      math: ['x̂ = (x − μ_batch) / √(σ²_batch + ε)', 'y = γ·x̂ + β', 'μ_run ← 0.9·μ_run + 0.1·μ_batch'], params: '2 × features (γ, β), plus running statistics that are not learned',
      use: ['After convolutions or dense layers, before the activation.', 'With larger learning rates.'], watch: ['Needs batches of more than a few examples.', 'Training and use behave differently: validation scores lag early on.'],
      code: 'nn.BatchNorm1d(features)   # nn.BatchNorm2d(channels) for pictures', try: 'shapes',
    },
    fourier: {
      what: 'Keeps the inputs and adds sines and cosines of them at random, fixed frequencies. Networks find slow, smooth functions easy and sharp wiggles hard ("spectral bias"); with these features, a wiggle is one weight away. A trick from neural rendering (NeRF) and physics-informed networks.',
      how: ['Draw m random frequency vectors bᵢ once (spread σ).', 'For each input x compute 2π·bᵢ·x.', 'Output x together with the sine and cosine of each.'],
      math: ['γ(x) = [x, sin(2π·Bx), cos(2π·Bx)]', 'B ~ N(0, σ²), fixed'], params: 'none (B is random and fixed)',
      use: ['Fitting sharp curves and surfaces from coordinates.', 'σ sets the wiggliness: start near 1–3.'], watch: ['A large σ fits the noise between data points.'],
      code: 'B = torch.randn(m, d_in) * sigma   # fixed, not learned\ngamma = torch.cat([x, torch.sin(2 * math.pi * x @ B.T), torch.cos(2 * math.pi * x @ B.T)], -1)', try: 'ripples',
    },
    attention: {
      what: 'Lets every position of a sequence look at the others and take what it needs. Each position makes a query; every position offers a key and a value; the query’s match with each key (a dot product) becomes a weight, and the output is the weighted average of the values. With several heads it can look for several things at once.',
      how: ['Project each position to a query q, a key k and a value v.', 'Score every pair: q·k / √d.', 'Turn each row of scores into weights with a softmax (causal: only earlier positions).', 'Average the values with those weights, then project back.'],
      math: ['A = softmax(Q·Kᵀ / √d)', 'out = A·V·W_o'], params: '4 × (d × d + d)  for Q, K, V and the output projection',
      use: ['Whenever an answer depends on far-away positions (reversing, looking up a fact).', 'After patches, to let parts of a picture talk to each other.'], watch: ['It does not know positions: add a position block (token inputs have one built in).', 'The cost grows with the square of the length.'],
      code: 'nn.MultiheadAttention(d, heads, batch_first=True)', try: 'reverse',
    },
    transformer: {
      what: 'The block GPT and BERT are made of: layer norm and self-attention with the input added back, then layer norm and a two-layer MLP with the input added back again. Stacked, these blocks can follow patterns across a whole sequence.',
      how: ['h = x + Attention(LayerNorm(x))', 'out = h + MLP(LayerNorm(h)), the MLP being Linear → GELU → Linear.'],
      math: ['h = x + Attn(LN₁(x))', 'y = h + W₂·GELU(W₁·LN₂(h) + b₁) + b₂'], params: 'attention 4(d² + d) + MLP 2·d·(mlp·d) + biases + 4d for the norms',
      use: ['Sequences of tokens: language, sorting, reversing.', 'Patches of pictures (a vision transformer).'], watch: ['Data-hungry: on small problems a CNN or LSTM can win.'],
      code: 'nn.TransformerEncoderLayer(d, heads, mlp * d, norm_first=True, batch_first=True)', try: 'sort',
    },
    rnn: {
      what: 'Reads a sequence one step at a time, keeping a hidden state that summarises what it has seen. A plain RNN overwrites its state every step and forgets quickly. An LSTM adds a memory cell and three gates (input, forget, output) that decide what to write, keep and show, so information can last many steps; a GRU does the same with two gates and no separate memory.',
      how: ['Start with a state of zeros.', 'At each position, combine the input and the previous state into new gates and a new state.', 'Output the state at every position (follow with Sequence pool → last step for one answer).'],
      math: ['RNN:  hₜ = tanh(W·xₜ + U·hₜ₋₁ + b)', 'LSTM: cₜ = f ⊙ cₜ₋₁ + i ⊙ g,  hₜ = o ⊙ tanh(cₜ)', 'GRU:  hₜ = (1 − z) ⊙ n + z ⊙ hₜ₋₁'], params: 'gates × units × (inputs + units + 2): 1 gate for an RNN, 4 for an LSTM, 3 for a GRU',
      use: ['Series and sentences where order matters.', 'Small data: they need fewer examples than transformers.'], watch: ['Slow: the steps run one after another.', 'Plain RNNs forget after a few steps (vanishing gradients through time).'],
      code: 'nn.LSTM(input_size, units, batch_first=True)   # or nn.GRU, nn.RNN', try: 'chaos',
    },
    conv1d: {
      what: 'Slides small learned filters along a sequence. Each output position mixes a short window of neighbours, so a filter finds a local pattern ("3 then 7", a rising edge) wherever it occurs. Causal windows only look back, as forecasting needs; dilation spaces the window out so stacked layers see far back cheaply (WaveNet).',
      how: ['For each position, gather the window of k neighbours (spaced by the dilation).', 'Each filter takes a weighted sum of the window, plus a bias.', 'Bend with the activation.'],
      math: ['yₜ,f = act(b_f + Σⱼ Σ_c W[f, j, c] · x[t − left + j·d, c])'], params: 'filters × window × input features + filters',
      use: ['Local patterns in series or sentences.', 'A fast alternative to recurrent layers.'], watch: ['Each layer only sees its window: stack layers or dilate to see further.'],
      code: 'nn.Conv1d(in_channels, filters, k, padding=k // 2)   # [batch, channels, length]', try: 'pattern',
    },
    position: {
      what: 'Adds a vector to each position of a sequence, different for every position. Attention by itself treats a sequence as an unordered bag; with positions added, "dog bites man" and "man bites dog" look different.',
      how: ['Look up the vector for each position: learned, or fixed sine and cosine waves.', 'Add it to that position’s numbers.'],
      math: ['learned: y_t = x_t + P_t', 'sine: PE[t, 2i] = sin(t / 10000^(2i/d)),  PE[t, 2i+1] = cos(…)'], params: 'learned: positions × features; sine: none',
      use: ['Before attention on series, patches or anything without built-in positions.'], watch: ['Learned positions do not extend past the trained length.'],
      code: 'self.pos = nn.Parameter(torch.zeros(1, T, d))   # x = x + self.pos', try: 'waves',
    },
    seqpool: {
      what: 'Turns a sequence into one vector, so a class or a value can be predicted for the whole sequence. "Last step" keeps what a recurrent layer has gathered by the end; "largest" keeps, per feature, its strongest response anywhere (did a pattern appear?); "average" pools evenly.',
      how: ['Last: take the final position.', 'Largest: for each feature, the maximum over positions.', 'Average: the mean over positions.'],
      math: ['last: y = x_T', 'max: y_c = maxₜ x_t,c', 'mean: y_c = (1/T) Σₜ x_t,c'], params: 'none',
      use: ['After an LSTM or GRU: the last step.', 'After 1-D convolutions: the largest.'], watch: ['Averaging forgets order: it is a bag of features.'],
      code: 'x[:, -1]   # last;  x.amax(1)  # largest;  x.mean(1)  # average', try: 'sentiment',
    },
    conv: {
      what: 'Slides small learned filters (3 × 3 or 5 × 5) over a picture. Each filter responds to one local pattern, an edge or a corner, wherever it appears, so it needs far fewer weights than a dense layer on every pixel, and a shifted picture gives a shifted answer.',
      how: ['Place the filter over every pixel’s neighbourhood.', 'Take the weighted sum of the pixels under it (across all channels), plus a bias.', 'Bend with the activation: one output picture (feature map) per filter.'],
      math: ['y[f, i, j] = act(b_f + Σ_c Σ_u,v W[f, c, u, v] · x[c, i+u, j+v])'], params: 'filters × channels × k × k + filters',
      use: ['Pictures, always; usually conv → pool, repeated, then flatten or global pool.', 'Residual convolutions (ResNet) let very deep stacks train.'], watch: ['Each layer sees only its window: stack and pool to see the whole picture.'],
      code: 'nn.Conv2d(in_channels, filters, k, padding=k // 2)', try: 'shapes',
    },
    pool: {
      what: 'Keeps the largest value in each small patch, halving the picture. Later filters see a wider area for the same cost, and small shifts of a feature stop mattering.',
      how: ['Cut each feature map into s × s patches.', 'Keep each patch’s maximum.'], math: ['y[c, i, j] = max over the patch of x'], params: 'none',
      use: ['After a convolution or two.'], watch: ['Throws away exact positions: not for tasks that need them.'], code: 'nn.MaxPool2d(2)', try: 'digits',
    },
    avgpool: {
      what: 'Averages each small patch, halving the picture. Smoother than max pooling: every pixel counts a little.',
      how: ['Cut each feature map into s × s patches.', 'Keep each patch’s mean.'], math: ['y[c, i, j] = mean of the patch'], params: 'none',
      use: ['Where gentle downsampling suits, e.g. before a classifier.'], watch: ['Blurs sharp, rare features that max pooling would keep.'], code: 'nn.AvgPool2d(2)', try: 'shapes',
    },
    patches: {
      what: 'Cuts the picture into small squares, flattens each one and maps it to a vector, plus a learned vector for its place: the picture becomes a sequence of tokens. A transformer can then let every patch look at every other. This is a vision transformer’s first step; after a convolution it makes a CNN-transformer hybrid.',
      how: ['Cut the picture into p × p squares, left to right, top to bottom.', 'Flatten each square’s pixels and multiply by a learned projection.', 'Add the learned position vector of each square.'],
      math: ['token_n = W·vec(patch_n) + b + P_n'], params: 'dim × channels × p² + dim + patches × dim',
      use: ['Before attention or transformer blocks on pictures.', 'Size 1 after a convolution: each pixel of the feature map is a token.'], watch: ['Small pictures and little data favour plain CNNs.', 'Many patches make attention slow.'],
      code: 'nn.Conv2d(channels, dim, p, stride=p)   # then .flatten(2).transpose(1, 2) + pos', try: 'shapes',
    },
    flatten: {
      what: 'Lines up all the numbers of a picture or a sequence into one long vector, so dense layers and the output can use them. It keeps every number but forgets which were neighbours.',
      how: ['Read the numbers row by row, channel by channel.'], math: ['[C, H, W] → [C·H·W]'], params: 'none',
      use: ['Between convolutions and dense layers.', 'To read a window of a series as one row of numbers.'], watch: ['Makes the next dense layer large.'], code: 'nn.Flatten()', try: 'digits',
    },
    gap: {
      what: 'Averages each channel over all positions: a whole picture or sequence becomes one short vector, whatever its size.',
      how: ['For each channel or feature, take the mean over every pixel or position.'], math: ['y_c = mean over positions of x_c'], params: 'none',
      use: ['At the end of a CNN, in place of flatten.', 'After a transformer, for one answer per sequence.'], watch: ['Position information is gone.'], code: 'x.mean(dim=(2, 3))   # pictures;  x.mean(1)  # sequences', try: 'sentiment',
    },
    linear: {
      what: 'The classic statistical model. For numbers it is linear regression: the prediction is a weighted sum of the inputs, and least squares finds the best weights exactly, in one step. For classes it is logistic regression: a weighted sum per class, turned into probabilities by a softmax and fitted by gradient steps. The baseline every other model should beat.',
      how: ['Numbers: solve (XᵀX + λI)·w = Xᵀy exactly.', 'Classes: start at zero weights and take gradient steps on the log loss.'],
      math: ['ŷ = w·x + b', 'w = (XᵀX + λI)⁻¹ Xᵀy      (ridge when λ > 0)', 'p(class k) = softmax(W·x + b)_k'], params: 'classes (or 1) × (inputs + 1)',
      use: ['As a baseline.', 'When the relationship really is linear (it then wins, quickly).'], watch: ['Only straight boundaries and flat surfaces: no spirals.'],
      code: 'from sklearn.linear_model import LinearRegression, Ridge, LogisticRegression', try: 'csv', classic: true,
    },
    knn: {
      what: 'Remembers every training example. To answer, it measures the distance to all of them, takes the k nearest and lets them vote (or averages their values). There is no training at all, and it can follow any shape, given enough examples.',
      how: ['Store the training set.', 'For a new example, compute its distance to every stored one.', 'Take the k nearest; the most common class (or the mean value) is the answer.'],
      math: ['d(x, xᵢ) = ‖x − xᵢ‖', 'ŷ = majority (or mean) of the k nearest yᵢ'], params: 'stores every training example',
      use: ['Small data with few inputs.', 'A quick baseline with no training.'], watch: ['Slow to answer on big data.', 'Distances lose meaning with many inputs, or inputs on different scales.'],
      code: 'from sklearn.neighbors import KNeighborsClassifier   # KNeighborsRegressor', try: 'moons', classic: true,
    },
    tree: {
      what: 'A flowchart of yes-or-no questions: "is x₁ ≤ 0.3?", then another question on each branch, down to leaves that give the answer. Grown greedily: at every node, the question that best separates the answers is chosen. Easy to read, but one deep tree memorises its training data.',
      how: ['At the root, try every input and threshold; keep the one that makes the two halves purest.', 'Repeat in each half, level by level.', 'Stop at the maximum depth, or when no question helps; each leaf answers with its examples’ average.'],
      math: ['impurity (classes): Gini = 1 − Σₖ pₖ²', 'gain = impurity(parent) − weighted impurity(children)'], params: '2 per question (input, threshold) + an answer per leaf',
      use: ['When you want to read the model.', 'Tables with mixed inputs.'], watch: ['Deep trees overfit; shallow ones underfit.', 'Boundaries are axis-aligned staircases.'],
      code: 'from sklearn.tree import DecisionTreeClassifier   # DecisionTreeRegressor', try: 'circle', classic: true,
    },
    forest: {
      what: 'Many deep trees, each grown on a bootstrap sample (the training rows drawn at random with repeats) and choosing among a random few inputs at every question, so they all differ. Their average is far steadier than any one tree: a strong model with almost no tuning.',
      how: ['For each tree: draw a bootstrap sample of the rows.', 'Grow a deep tree, considering only a random subset of inputs at each question.', 'Answer with the average of all the trees’ answers.'],
      math: ['ŷ = (1/T) Σₜ treeₜ(x)', 'inputs tried per question: √(inputs) for classes'], params: 'sum over trees of 2 per question + an answer per leaf',
      use: ['Tables: a strong first try.', 'When you need little tuning.'], watch: ['Big and slowish to answer.', 'Cannot predict beyond the range of the training targets.'],
      code: 'from sklearn.ensemble import RandomForestClassifier   # RandomForestRegressor', try: 'csv', classic: true,
    },
    boost: {
      what: 'Builds trees one after another, each fitted to what the trees so far still get wrong. XGBoost uses the gradient and curvature of the loss at every example, keeps each tree small, scales its answer by a learning rate η, and penalises big leaves (λ) and extra questions (γ). Gradient boosting is the method that wins most competitions on tables.',
      how: ['Start from a constant guess.', 'Compute each example’s gradient g and curvature h of the loss.', 'Grow a small tree that groups examples with similar g; each leaf answers −G/(H + λ).', 'Add the tree, scaled by η, and repeat.'],
      math: ['leaf value w = −G / (H + λ)', 'split gain = ½[G_L²/(H_L+λ) + G_R²/(H_R+λ) − G²/(H+λ)] − γ', 'F_m(x) = F_(m−1)(x) + η·tree_m(x)'], params: 'rounds × trees per round × (2 per question + 1 per leaf)',
      use: ['Tables of numbers: usually the best single model.', 'Learning rate 0.05–0.3 with 100–1000 rounds.'], watch: ['Many rounds at a high η overfit: watch validation loss.', 'Less natural for pictures and sequences.'],
      code: 'from xgboost import XGBClassifier   # XGBRegressor', try: 'csv', classic: true,
    },
  };

  /* What each part does, in depth: the idea, what happens to the numbers, what it learns and how training shapes it,
     what its settings change and where it goes wrong, for any level. Shown under "What it does" wherever a part is
     explained: the palette's guide, a selected block, and the block's level in the Architecture view. */
  const DEEP = {
    dense: [
      `Think of each neuron as a small judge with its own opinion of every input. A weight says how much one input matters and in which direction: a positive weight pushes the neuron's score up when that input is large, a negative one pulls it down. The neuron adds up its weighted inputs and a bias (its starting score) into one number z, and the activation function decides what to pass on: ReLU passes positive scores and silences negative ones, tanh squashes everything into −1…1. Because every neuron sees every input, the layer is called fully connected.`,
      `Seen as geometry, a neuron's weighted sum is a straight line through the plane of its inputs (a flat plane, or hyperplane, with more inputs): z is positive on one side and negative on the other, and the weights set the line's angle while the bias slides it. The activation bends the output along that line, so a ReLU neuron is a ramp that starts at its line. A layer of 16 neurons lays 16 ramps over the input space, and the next layer adds them up with its own weights: that is how a network builds curved boundaries out of straight pieces. More neurons give more pieces; more layers let pieces be combined into pieces of pieces.`,
      `In training, each weight changes in proportion to two things: how large its input was, and how much the loss wanted the neuron's output to change (the gradient arriving from the layer above). A neuron whose weighted sum is negative for every example has slope 0 under ReLU, so its weights stop changing for good: a dead neuron. Leaky ReLU, GELU or a smaller learning rate keep neurons alive; the dots on the block show which ones fire.`,
      `Its settings: the number of neurons is the layer's width (16 is plenty for points in a plane; pictures need more). The activation is the bend. Residual adds the layer's input back to its output, which only works when the width is unchanged; it lets a deep stack learn small corrections and keeps the gradient flowing to the early layers. On a sequence, the same layer, with the same weights, runs on every position separately.`,
    ],
    neuron: [
      `A single neuron is the smallest thing that learns: z = w₁x₁ + w₂x₂ + … + b, then an activation. With two inputs, z = 0 is a straight line through the plane: the weights set its angle and the bias its position, and the neuron's output rises on one side of it.`,
      `Alone it can only separate classes with one straight line. It can learn AND or OR, but never XOR, which needs two lines: the limit made famous by Minsky and Papert's 1969 book on perceptrons, and the reason networks use layers of many neurons.`,
      `Drop it on a dense layer to add one neuron (one more line, one more ramp) to that layer, or between blocks for a layer of one neuron: a bottleneck that squeezes everything through a single number.`,
    ],
    gated: [
      `A gated layer computes two weighted sums of the same input for each of its units: a value (what to say) and a gate (how loudly to say it). The gate goes through a function, a sigmoid in the original GLU, Swish in SwiGLU or GELU in GEGLU, and multiplies the value number by number.`,
      `Multiplying makes the layer's response depend on the input in a richer way than a single bend: the same value can be switched on for some examples and off for others. That is why gated layers learn more per weight, and why most large language models (PaLM, LLaMA and their successors) use SwiGLU in every transformer block instead of a plain MLP.`,
      `It has twice the weights of a dense layer of the same width: two full matrices and two biases. A sigmoid gate stays between 0 and 1, a soft switch; Swish and GELU gates can dip slightly below 0 and grow without limit, which keeps gradients from vanishing.`,
    ],
    activation: [
      `An activation block applies one fixed function to every number separately; it has nothing to learn. Its job is the bend: a stack of weighted sums without bends is still a single weighted sum, a straight line or a flat plane however deep it is, because a linear function of a linear function is linear.`,
      `Each function bends differently. ReLU, max(0, z), is cheap and keeps a slope of 1 for positive numbers, so gradients travel well through deep stacks, but negative numbers get slope 0. Tanh and sigmoid are smooth S-curves that flatten at the ends (they saturate), where the slope is almost 0 and learning slows. GELU and Swish are smooth versions of ReLU used in transformers. Sine makes waves, good for periodic patterns and coordinates; Gaussian makes bumps.`,
      `When the model learns, the gradient passing back through this block is multiplied by the function's slope at each number, so the shape of the slope matters as much as the shape of the function. Try one model with each function and watch the boundary change: straight pieces with ReLU, smooth curves with tanh, waves with sine.`,
    ],
    dropout: [
      `While training, dropout sets a random fraction p of the numbers passing through to 0 at every step, and multiplies the others by 1/(1 − p) so that their average stays the same. A new random set is dropped at every step. When the model is evaluated or used, nothing is dropped: the block does nothing.`,
      `Why it helps: a neuron can never count on any one of its inputs being there, so the network cannot build fragile chains in which one neuron only works because of another's exact quirks. It behaves like training a huge ensemble of thinned networks that share their weights, and using their average at the end. Hinton's group introduced it in 2012, and it made large networks practical on small data.`,
      `It costs training speed (the loss is noisier and falls more slowly), and it only helps when the model overfits: if the training and validation scores are both poor, dropout makes things worse. Typical rates are 0.1 to 0.5; transformers usually use 0.1.`,
    ],
    noise: [
      `While training, every number passing through gets a small random amount added, drawn from a normal distribution of spread σ; when the model is used, nothing is added. It is like showing the model a slightly different version of every example each time it sees it.`,
      `The model learns that tiny differences in what it receives should not change its answer, so the function it learns becomes smoother between the training examples. Adding noise of variance σ² to the inputs acts much like penalising the model's steepness, a classic result (Bishop, 1995).`,
      `Too much σ drowns the signal: if the noise is as large as the differences between the classes, there is nothing left to learn. Right after the input it works as data augmentation; between layers it makes hidden features robust.`,
    ],
    layernorm: [
      `Layer norm takes the numbers of one example (or of one position of a sequence), computes their mean μ and spread σ, and rescales them to mean 0 and spread 1: x̂ = (x − μ) / σ. Two learned numbers per feature, a gain γ and a bias β, then set whatever scale and shift the next layer prefers.`,
      `Without it, the size of the signal drifts as it passes through many layers: it grows until activations saturate, or shrinks until it is lost in rounding. Normalising every example keeps each layer's input in a steady range, so one learning rate suits every layer and training is far steadier. Every transformer block uses it twice.`,
      `Unlike batch norm it does not depend on the other examples in the batch, so it behaves the same in training and in use, for any batch size, one example at a time included. That is why sequence models prefer it.`,
    ],
    batchnorm: [
      `Batch norm normalises each feature (each neuron, or each channel of a picture) across the examples of the current batch: it subtracts the batch's mean of that feature and divides by its spread, then applies a learned gain γ and bias β.`,
      `It made very deep convolutional networks trainable when it appeared in 2015: each layer sees inputs with a steady distribution even while the layers before it change, so larger learning rates work. Batch statistics also add a little noise, because they depend on which examples share a batch, and that acts as a mild regulariser.`,
      `When the model is used there may be no batch, so batch norm keeps running averages of the mean and variance from training and uses those instead. That difference is its main pitfall: tiny batches give noisy statistics, and a model can behave differently in use than it did in training.`,
    ],
    fourier: [
      `Fourier features keep the inputs and add sin(2π·b·x) and cos(2π·b·x) for a set of random frequency vectors b, drawn once from a normal distribution of spread σ and never trained. A point in the plane becomes a long vector of waves running in many directions at many frequencies.`,
      `Plain networks of ReLU or tanh neurons learn smooth, slowly varying shapes first and need many neurons and steps to fit sharp detail (their spectral bias). Handing them ready-made waves lets the next dense layer combine them into fine detail directly. NeRF, which rebuilds 3-D scenes from photographs, depends on this trick (Tancik and others, 2020).`,
      `σ sets the frequencies. Too low, and nothing changes; too high, and the model wiggles between the training points and fits noise. For inputs between −1 and 1 start between 1 and 3 and compare runs.`,
    ],
    attention: [
      `Self-attention lets every position of a sequence gather information from the others. Each position turns its vector into three: a query (what am I looking for?), a key (what do I contain?) and a value (what do I pass on?). How well a query matches each key (a dot product, divided by √d) becomes a score, a softmax turns the scores into weights that add up to 1, and the position's output is the weighted average of the values.`,
      `Nothing about order is built in: attention treats the sequence as a set, which is why models add positions first. Causal attention masks the future, so position t sees only positions up to t, as a model that writes one token at a time must. With several heads the vector is split into slices and each head attends on its own, so one head can track grammar while another follows a name.`,
      `Its cost grows with the square of the length, every position against every other, which is why long contexts are expensive. The residual adds the input back, so the layer only has to learn what to add to each position.`,
    ],
    transformer: [
      `A transformer block, the building block of GPT and most modern language models, runs two steps, each wrapped in a residual connection. First self-attention, where the positions exchange information; then an MLP, where each position thinks on its own with a hidden layer usually four times wider and a GELU. A layer norm before each step keeps the signal steady (the pre-norm arrangement used since GPT-2).`,
      `It helps to picture a shared notepad, the residual stream, flowing through the stack: every attention layer and every MLP reads it and writes a small update into it. Attention moves information between positions; the MLP transforms it at each position, and is where much of a model's stored knowledge seems to live.`,
      `Its settings: heads split the attention; the MLP size multiplies the hidden layer (×4 is standard); causal makes it a writer that only looks back. A block of width d has about 12·d² weights, so wider models grow fast.`,
    ],
    rnn: [
      `A recurrent layer reads a sequence one step at a time and carries a state vector h, a running summary of what it has seen. At every step the new state is computed from the current input and the previous state, always with the same weights: h_t = f(W·x_t + U·h_(t−1) + b). Its output at each position is the state.`,
      `A plain RNN struggles with long gaps: the gradient flowing back through many steps is multiplied by the same matrix again and again, so it fades away or blows up. The LSTM (Hochreiter and Schmidhuber, 1997) adds a separate memory cell c and gates that decide what to write, what to keep and what to show, so information can survive hundreds of steps. The GRU (2014) does much the same with two gates and no separate cell.`,
      `The units are the size of the state, the memory's capacity. Follow it with a sequence pool (last step) for one answer per sequence. Because it reads the steps in order it cannot be spread over time the way attention can, one reason transformers replaced it for long texts.`,
    ],
    conv1d: [
      `A 1-D convolution slides small filters along the sequence. Each filter looks at a window of k neighbouring positions (all of their features), multiplies them by its weights and adds them up, and the same filter is used at every position, so a pattern is found wherever it occurs.`,
      `Dilation spreads the window out (dilation 2 looks at every second position) and a causal window only looks back, so the model never sees the future. Stacking causal convolutions with dilations 1, 2, 4, 8 makes the span each output sees double with every layer: the WaveNet design for audio, and the basis of temporal convolutional networks.`,
      `Compared with a recurrent layer it reads all positions at once (fast to train) and has a fixed reach; compared with attention, its cost grows only linearly with the length.`,
    ],
    position: [
      `Attention and dense layers treat a sequence like a bag: shuffle its positions and they give the same outputs, shuffled. A positions block adds a different vector at each position, so the layers after it can tell first from last and near from far.`,
      `Learned positions are one trainable vector per position, as in GPT-2 and the Vision Transformer. Sine positions are fixed waves of many frequencies, as in the original transformer (2017): position t gets sin(t / 10000^(2i/d)) and cos(…) in alternate features, so an offset between two positions becomes a rotation a layer can detect.`,
      `Recurrent and convolutional layers already know the order, because they step or slide, so positions matter most in front of attention. Many recent models use rotary embeddings (RoPE) instead, which rotate the queries and keys rather than adding vectors.`,
    ],
    seqpool: [
      `A class or a single value needs one vector, but a sequence layer gives one per position. Sequence pooling squeezes the sequence: last keeps the final position (what a recurrent layer has gathered after reading everything), max keeps the largest value of each feature over all positions (did this pattern appear anywhere?) and mean averages them (how much of it was there overall?).`,
      `Last suits recurrent layers and causal attention, whose final position has seen everything. Max suits convolutions that detect local patterns, such as a phrase that decides the sentiment. Mean is a steady summary that uses every position.`,
      `It has nothing to learn. The gradient goes only to the last position, to the position that held each maximum, or equally to every position.`,
    ],
    conv: [
      `A convolution slides small learned filters (1×1, 3×3 or 5×5) over the picture. At each place a filter multiplies the pixels under it, in every input channel, by its weights and adds them up with a bias, giving one number per place: a feature map that shows where the filter's pattern occurs. Eight filters give eight feature maps, the output's channels.`,
      `Two ideas make it efficient: locality (each output looks at a small neighbourhood only) and weight sharing (the same filter everywhere), so a pattern is found wherever it appears with far fewer weights than a dense layer would need. Stacked layers see wider areas: early filters find edges and spots, later ones combine them into strokes, corners and shapes, the design that runs from LeNet (1989) to ResNet (2015).`,
      `Stride 2 moves the filter two pixels at a time and halves the map; padding keeps the borders so the size stays the same. A 1×1 filter looks at no neighbours at all: it mixes the channels at each pixel, a cheap way to change their number, used in Inception modules and ResNet bottlenecks.`,
    ],
    pool: [
      `Max pooling cuts each feature map into small patches (2×2) and keeps only the largest value of each, so the map shrinks by half in each direction and holds a quarter of the numbers.`,
      `It keeps the strongest response of each detector (was the pattern anywhere in this patch?) and forgets exactly where, which makes the model less sensitive to small shifts and lets later filters of the same size see a wider area. It has nothing to learn, and the gradient flows only to the value that was the maximum.`,
      `Many modern networks replace it with stride-2 convolutions, which learn how to shrink.`,
    ],
    avgpool: [
      `Average pooling replaces each small patch (2×2) of a feature map with its average, halving the size in each direction. Unlike max pooling every value counts a little, so it smooths rather than picking the strongest response.`,
      `Use it when the overall amount of a feature matters more than its strongest spot, or to shrink maps gently. It has nothing to learn; the gradient is shared equally by the values of each patch.`,
    ],
    patches: [
      `Patches turn a picture into a sequence of tokens for attention. The picture is cut into non-overlapping squares, each square's pixels are flattened and multiplied by one shared learned matrix into a vector, and a learned position vector is added to each so the model knows where it came from.`,
      `This is how a Vision Transformer (2020) starts: once patched, a picture is just a sentence of patch-words, and a standard transformer can relate any two patches, near or far, from its first layer. Convolutions build a wide view slowly; attention has it at once, but needs more data to learn what convolutions take for granted, that neighbours matter.`,
      `Bigger patches give fewer tokens, faster and coarser; smaller patches give more tokens and finer detail, more slowly, because attention's cost grows with the square of the number of tokens.`,
    ],
    flatten: [
      `Flatten lines up all the numbers of a picture (channels × height × width) or of a sequence (positions × features) into one long vector, so that dense layers and a class or value output can use them. Nothing is computed and nothing is learned; the numbers are only relabelled.`,
      `It keeps every number, but the dense layer after it treats each as a separate input with its own weight, so it no longer knows which numbers were neighbours and needs many weights: one per pixel per neuron. Global pooling is the cheaper choice when only the presence of features matters.`,
    ],
    gap: [
      `Global average pooling averages each channel over every position: a picture of C channels becomes C numbers, and a sequence of D features becomes D numbers, whatever its size.`,
      `Compared with flatten it needs no weights and no fixed size, and it makes the model ignore where in the picture a feature appeared; most image classifiers since ResNet end with it. The price is that position is gone.`,
    ],
    parallel: [
      `Parallel branches run two to four small stacks of blocks side by side on the same input, then combine what they give. Each branch is a little model of its own with its own weights, so the block can look at the data in several ways at once: big and small filters, a recurrent and a convolutional reader, a deep path and a shallow one.`,
      `How they meet matters. Joining lays the outputs side by side: every branch's numbers reach the next layer, which learns how much to trust each one; this is the Inception module (2014) and the wide-and-deep model (2016). Adding needs equal shapes and mixes them number by number; with one branch left empty it adds the input back, a residual connection around the other branches, as in ResNet. Multiplying lets one branch gate the other, the idea inside LSTMs and gated layers. Averaging makes the branches vote, like a small ensemble.`,
      `When the model learns, the gradient reaching the merge is shared out among the branches: each gets the slice for its own numbers (joined), all of it (added), a k-th of it (averaged), or all of it times the other branches' outputs (multiplied). Every branch then runs it back through its own blocks, and what reaches the input is the sum over the branches, because the input fed all of them.`,
      `On the model, each branch is a lane: add blocks with its list or by dragging parts onto it, and an empty lane passes the input straight through. Branches can give different sizes when joined, but the same kind of data, with the same positions or picture size.`,
    ],
    linear: [
      `A linear model computes one weighted sum of the inputs plus a bias for each output: ŷ = w·x + b. For numbers (regression) the weights are found exactly, in one step, by least squares: the formula for the weights that minimise the squared error. For classes it is logistic regression: one weighted sum per class, turned into probabilities by a softmax and fitted by gradient steps on the cross-entropy.`,
      `Its boundaries are straight lines (flat planes) and its fitted surfaces are flat, so it fails on curved problems; but it is fast, needs little data, and every weight can be read: how much the answer changes per unit of each input. The ridge penalty λ shrinks the weights toward 0, which steadies them when inputs are correlated or examples are few.`,
      `It is the right first baseline for any problem: a network that cannot beat it has not learned anything useful.`,
    ],
    knn: [
      `k-nearest neighbours keeps every training example. To answer for a new input it measures the distance to all of them, takes the k closest, and lets them vote (classes) or averages their values (numbers); weighting by distance gives the closest ones more say.`,
      `There is no training and no formula: the answer follows the data itself, so it can fit any shape. But it must store and search all the data for every answer, it is misled by inputs on very different scales, and with many inputs all points end up far apart (the curse of dimensionality).`,
      `k sets the smoothness: k = 1 follows every point, noisy ones included, while a large k averages over a wide neighbourhood and blurs the boundary. Choose it by the validation score.`,
    ],
    tree: [
      `A decision tree is a flowchart of yes-or-no questions about the inputs, such as “is x₁ ≤ 0.3?”. It is grown from the top: at each node it tries every input and every threshold, keeps the question that best separates the training answers (the one that most reduces their impurity: Gini for classes, squared error for numbers), splits the examples, and asks a new question on each side, until a depth limit or too few examples.`,
      `A leaf answers with the most common class, or the average value, of the training examples that reached it, so the boundaries are rectangles: axis-aligned steps. Trees need no scaling of the inputs, mix kinds of data easily and are easy to read.`,
      `A deep tree memorises the training data, a leaf for every one or two examples, and overfits; the depth and the smallest leaf are its brakes. On its own it is unstable, since a small change in the data changes its questions, which is what forests and boosting fix.`,
    ],
    forest: [
      `A random forest grows many deep trees, each on a bootstrap sample of the training examples (drawn with replacement) and choosing each question among a random few of the inputs. To answer, every tree votes, or their answers are averaged.`,
      `Each tree overfits in its own way, but their mistakes differ, so averaging cancels much of the noise: a steady model built from unsteady parts. The randomness is essential, since trees grown on the same data from all the inputs would be nearly identical and gain nothing from averaging. Breiman introduced it in 2001, and it is still one of the most reliable models for tables.`,
      `More trees never overfit; they only cost time. The depth and the inputs per question set how different, and how strong, each tree is.`,
    ],
    boost: [
      `Gradient boosting builds its answer one small tree at a time. It starts from a constant guess; each new tree is trained on the gradient of the loss for every example (for squared error, simply what the model so far still gets wrong) and is added with a small weight η, the learning rate.`,
      `XGBoost (2016) grows each tree with a second-order rule: a leaf's value is −G/(H + λ), from the sums of the gradients G and curvatures H of its examples, and a split is kept only if it gains more than γ. Shrinkage, the leaf penalty and sampling rows and inputs per tree keep each step modest, so many weak trees add up to a strong model.`,
      `It usually wins on tables of numbers. Its main risk is overfitting with too many rounds at a high η: watch the validation score and stop where it is best.`,
    ],
  };
  for (const k of Object.keys(DEEP)) if (G[k]) G[k].deep = DEEP[k];
  G.parallel = {
    what: 'Two to four branches of blocks side by side on the same input, their outputs joined, added, multiplied or averaged.',
    how: ['Copy the input to every branch.', 'Run each branch’s blocks in order, each with its own weights.', 'Join the outputs side by side, or add, multiply or average them number by number.'],
    math: ['y_b = branch_b(x)      for b = A, B, …', 'joined: y = [y_A, y_B, …]      added: y = Σ y_b', 'multiplied: y = Π y_b      averaged: y = (1/k) Σ y_b'],
    params: 'the learned numbers of the blocks in its branches; the merge itself learns nothing',
    use: ['Several views of the data at once: small and large filters (Inception), a recurrent and a convolutional reader.', 'A wide (raw inputs) and a deep (learned features) path for tables.', 'An empty branch and Add: a residual connection.', 'A sigmoid branch and Multiply: a gate.'],
    watch: ['Add, multiply and average need every branch to give the same shape.', 'Joined images need the same height and width; joined sequences the same positions.'],
    code: 'outs = [branch(x) for branch in self.branches]\ny = torch.cat(outs, -1)        # or sum(outs), torch.stack(outs).prod(0), .mean(0)',
    deep: DEEP.parallel,
  };

  // What the block does here, in numbers: its shapes, its learned numbers and how they add up, what its settings mean.
  function inModel(b, sh) {
    if (!b || !sh || !sh.in || !sh.out) return '';
    const c = Object.assign({}, (NA.bx && NA.bx.BLOCKS[b.type] || {}).defaults, b), si = sh.in, so = sh.out, P = sh.params || 0;
    const T = (s) => s.join(' × '), last = si[si.length - 1], seq = si.length === 2, n = (v) => Number(v).toLocaleString('en');
    const pos = seq ? `, the same weights at each of the ${si[0]} positions` : '';
    const A = (id) => ((NA.engine && NA.engine.ACTIVATIONS[id]) || {}).label || id;
    switch (b.type) {
      case 'dense': return `Here: ${T(si)} in, ${T(so)} out. Each of its ${c.units} neuron${c.units === 1 ? '' : 's'} has ${last} weight${last === 1 ? '' : 's'} (one per input) and a bias: ${last + 1} learned numbers each, ${c.units} × ${last + 1} = ${n(P)} in all${pos}. Its activation is ${Array.isArray(b.acts) ? 'mixed: some neurons have their own' : A(c.act)}${c.residual ? (last === c.units ? ', and its input is added back (residual)' : `; the residual is off, because it would need ${last} neurons`) : ''}.`;
      case 'gated': return `Here: ${T(si)} in, ${T(so)} out. Each of its ${c.units} units has a value and a gate, each with ${last} weights and a bias: 2 × ${c.units} × ${last + 1} = ${n(P)} learned numbers${pos}. The gate's function is ${A(c.gate)}.`;
      case 'activation': return `Here it bends each of the ${n(si.reduce((a, q) => a * q, 1))} numbers it receives (${T(si)}) with ${A(c.act)}, and learns nothing.`;
      case 'dropout': return `Here, while training, it drops ${Math.round(c.p * 100)}% of the ${n(si.reduce((a, q) => a * q, 1))} numbers it receives at every step and scales the rest by ${(1 / (1 - c.p)).toFixed(2)}; in use it passes all of them unchanged.`;
      case 'noise': return `Here, while training, it adds noise of spread σ = ${c.sigma} to each of the ${n(si.reduce((a, q) => a * q, 1))} numbers it receives; in use it passes them unchanged.`;
      case 'layernorm': return `Here it normalises ${seq ? `each of the ${si[0]} positions` : 'each example'}: ${last} numbers to mean 0 and spread 1, then a gain and a bias per feature, 2 × ${last} = ${n(P)} learned numbers.`;
      case 'batchnorm': { const C = si.length === 3 ? si[0] : last; return `Here it normalises each of ${C} ${si.length === 3 ? 'channels' : 'features'} over the batch, with a learned gain and bias each: 2 × ${C} = ${n(P)} learned numbers, plus a running mean and variance per ${si.length === 3 ? 'channel' : 'feature'} for use.`; }
      case 'fourier': return `Here it keeps the ${si[0]} input${si[0] === 1 ? '' : 's'} and adds the sine and cosine of ${c.features} random frequencies (spread σ = ${c.scale}): ${si[0]} + 2 × ${c.features} = ${so[0]} numbers out, none of them learned.`;
      case 'attention': { const D = last, h = c.heads; return `Here every one of the ${si[0]} positions makes a query, a key and a value of ${D} numbers, split into ${h} head${h === 1 ? '' : 's'} of ${D / h}; ${c.causal !== false ? 'each position sees only itself and the ones before it' : 'every position sees every other'}. The four projections (query, key, value and output) are ${D} × ${D} matrices with biases: 4 × (${D} × ${D} + ${D}) = ${n(P)} learned numbers.`; }
      case 'transformer': { const D = last, Hd = Math.max(1, Math.round((c.mlp || 4) * D)); return `Here each of the ${si[0]} positions carries ${D} numbers. Attention has ${c.heads} head${c.heads === 1 ? '' : 's'}${c.causal !== false ? ' and only looks back' : ''}; the MLP widens to ${Hd} hidden neurons and back. Learned numbers: attention 4 × (${D}² + ${D}), the two layer norms 4 × ${D}, the MLP ${D} × ${Hd} + ${Hd} + ${Hd} × ${D} + ${D}: ${n(P)} in all.`; }
      case 'conv': { const C = si[0], k = c.k; return `Here it slides ${c.filters} filters of ${k} × ${k} over a ${T(si)} input${c.stride > 1 ? `, ${c.stride} pixels at a time` : ''}. Each filter reaches into all ${C} input channel${C === 1 ? '' : 's'}: ${C} × ${k} × ${k} = ${C * k * k} weights and a bias, ${C * k * k + 1} learned numbers; ${c.filters} filters make ${n(P)}. Out come ${so[0]} feature maps of ${so[1]} × ${so[2]}.`; }
      case 'pool': case 'avgpool': return `Here it turns ${T(si)} into ${T(so)}, keeping the ${b.type === 'pool' ? 'largest value' : 'average'} of each ${c.size} × ${c.size} patch of each channel, and learns nothing.`;
      case 'flatten': return `Here it lays the ${T(si)} numbers out as one vector of ${so[0]}, and learns nothing.`;
      case 'gap': return `Here it averages each of the ${so[0]} ${si.length === 3 ? 'channels over its ' + si[1] + ' × ' + si[2] + ' pixels' : 'features over the ' + si[0] + ' positions'}, and learns nothing.`;
      case 'seqpool': return `Here it squeezes ${si[0]} positions of ${si[1]} numbers into one vector of ${so[0]}: ${({ last: 'the last position', max: 'the largest value of each feature', mean: 'the average of each feature' })[c.mode] || c.mode}, and learns nothing.`;
      case 'rnn': { const G = { rnn: 1, lstm: 4, gru: 3 }[c.cell] || 4, H = c.units, D = last; return `Here a ${({ rnn: 'plain RNN', lstm: 'LSTM', gru: 'GRU' })[c.cell] || c.cell} reads ${si[0]} steps of ${D} numbers and carries a state of ${H}. It has ${G} gate${G === 1 ? '' : 's'} (or candidates), each with ${D} input weights, ${H} state weights and two biases per unit: ${G} × ${H} × (${D} + ${H} + 2) = ${n(P)} learned numbers, the same at every step.`; }
      case 'conv1d': { const k = c.k, D = last, reach = (k - 1) * (c.dilation || 1) + 1; return `Here ${c.filters} filters each look at ${k} positions${c.dilation > 1 ? `, ${c.dilation} apart (a reach of ${reach})` : ''}${c.causal ? ', only backwards' : ''}, with all ${D} numbers at each: ${k} × ${D} weights and a bias, ${k * D + 1} learned numbers per filter, ${n(P)} in all.`; }
      case 'position': return c.mode === 'sine' ? `Here it adds fixed sine and cosine waves to the ${si[0]} positions of ${last} numbers; nothing is learned.` : `Here it adds a learned vector of ${last} numbers to each of the ${si[0]} positions: ${si[0]} × ${last} = ${n(P)} learned numbers.`;
      case 'patches': { const p = c.size, C = si[0], N = so[0]; return `Here it cuts the ${T(si)} picture into ${N} patches of ${p} × ${p}, and projects each patch's ${C * p * p} pixel values into ${c.dim} numbers with one shared matrix (${C * p * p} × ${c.dim} weights and ${c.dim} biases), then adds a learned place for each patch (${N} × ${c.dim}): ${n(P)} learned numbers.`; }
      case 'parallel': { const k = (b.branches || []).length, merge = c.merge || 'concat'; return `Here ${k} branches each receive the ${T(si)} input; what they give is ${({ concat: 'joined side by side', add: 'added', mul: 'multiplied', mean: 'averaged' })[merge]} into ${T(so)}. Its ${n(P)} learned numbers are those of the blocks in its branches.`; }
      default: return `Here: ${T(si)} in, ${T(so)} out${P ? `, ${n(P)} learned numbers` : ''}.`;
    }
  }

  // ------------------------------------------------------------------ the pictures
  /* A small picture of each part, drawn into ctx (W × H) at time t seconds. Colours come from the theme: s1 blue
     for data, s2 orange for what the part makes, ink for structure. */
  function draw(type, ctx, W, H, t) {
    const c = (n, a) => theme.c(n, a), ink = c('ink'), muted = c('muted');
    const line = (x1, y1, x2, y2, col = muted, w = 1.2) => { ctx.strokeStyle = col; ctx.lineWidth = w; ctx.beginPath(); ctx.moveTo(x1, y1); ctx.lineTo(x2, y2); ctx.stroke(); };
    const dot = (x, y, r, col) => { ctx.fillStyle = col; ctx.beginPath(); ctx.arc(x, y, r, 0, Math.PI * 2); ctx.fill(); };
    const box = (x, y, w, h, fill, stroke) => { if (fill) { ctx.fillStyle = fill; ctx.fillRect(x, y, w, h); } if (stroke) { ctx.strokeStyle = stroke; ctx.lineWidth = 1; ctx.strokeRect(x + 0.5, y + 0.5, w - 1, h - 1); } };
    // a caption that fits: a smaller size first, then an ellipsis
    const label = (s, x, y, o = {}) => {
      const fam = getComputedStyle(document.body).getPropertyValue(o.mono ? '--font-mono' : '--font-ui') || 'sans-serif', room = o.maxW || W - 12;
      let size = o.size || 10.5;
      const setFont = () => { ctx.font = `${o.w || 500} ${size}px ${fam}`; };
      setFont();
      while (size > 8.5 && ctx.measureText(s).width > room) { size -= 0.5; setFont(); }
      while (s.length > 4 && ctx.measureText(s).width > room) s = s.slice(0, -2).trimEnd() + '…';
      ctx.fillStyle = o.color || muted; ctx.textAlign = o.align || 'center'; ctx.textBaseline = 'middle';
      const half = ctx.measureText(s).width / 2, cx = (o.align || 'center') === 'center' ? clamp(x, 6 + half, W - 6 - half) : x;
      ctx.fillText(s, cx, y);
    };
    const rnd = (i) => { const v = Math.sin(i * 12.9898 + 78.233) * 43758.5453; return v - Math.floor(v); };
    const mid = H / 2, pulse = (t % 2) / 2;
    ctx.clearRect(0, 0, W, H);
    if (String(type).startsWith('rl-')) { drawRL(type, ctx, W, H, t, { c, line, dot, box, label, rnd, mid }); return; }
    // parts whose guides live in another file (NA.partGuide.drawMore[type], with the same helpers)
    const more = NA.partGuide.drawMore && NA.partGuide.drawMore[type];
    if (more) { more(ctx, W, H, t, { c, ink, muted, line, dot, box, label, rnd, mid, pulse, theme, clamp }); return; }
    if (type === 'dense' || type === 'neuron' || type === 'gated') {
      const ins = type === 'neuron' ? 3 : 4, outs = type === 'neuron' ? 1 : 3, x0 = W * 0.22, x1 = W * 0.7;
      const yi = (i) => mid + (i - (ins - 1) / 2) * (H / (ins + 1)), yo = (j) => mid + (j - (outs - 1) / 2) * (H / (outs + 1.5));
      for (let i = 0; i < ins; i++) for (let j = 0; j < outs; j++) { const w = rnd(i * 7 + j) * 2 - 1; ctx.strokeStyle = theme.divColor(w, 0.75); ctx.lineWidth = 0.6 + 2.4 * Math.abs(w); ctx.beginPath(); ctx.moveTo(x0, yi(i)); ctx.lineTo(x1, yo(j)); ctx.stroke(); }
      for (let i = 0; i < ins; i++) { dot(x0, yi(i), 6, c('s1')); const p = clamp(pulse * 1.3 - i * 0.05, 0, 1); dot(x0 + (x1 - x0) * p, yi(i) + (yo(i % outs) - yi(i)) * p, 2.5, ink); }
      for (let j = 0; j < outs; j++) { box(x1 - 7, yo(j) - 7, 14, 14, c('s2', 0.85)); if (type === 'gated') { dot(x1 + 22, yo(j), 5, c('s7')); label('×', x1 + 22, yo(j), { color: c('panel'), size: 9, w: 700 }); } }
      label('inputs', x0, H - 8); label(type === 'gated' ? 'value × gate' : 'neurons: Σ w·x + b, then f', x1, H - 8);
    } else if (type === 'activation') {
      const fns = [['ReLU', (z) => Math.max(0, z), 's1'], ['tanh', Math.tanh, 's2'], ['sigmoid', (z) => 1 / (1 + Math.exp(-z)), 's4'], ['GELU', (z) => 0.5 * z * (1 + Math.tanh(0.7978845608 * (z + 0.044715 * z * z * z))), 's7']];
      const X = (z) => W * 0.1 + ((z + 3) / 6) * W * 0.8, Y = (v) => mid - v * (H * 0.28);
      line(W * 0.1, mid, W * 0.9, mid); line(X(0), 8, X(0), H - 8);
      fns.forEach(([n, f, col], k) => { ctx.strokeStyle = c(col); ctx.lineWidth = 2; ctx.beginPath(); for (let i = 0; i <= 80; i++) { const z = -3 + (6 * i) / 80; i ? ctx.lineTo(X(z), Y(f(z))) : ctx.moveTo(X(z), Y(f(z))); } ctx.stroke(); label(n, W * 0.12 + k * 46, 10, { color: c(col), align: 'left', w: 600 }); });
    } else if (type === 'dropout' || type === 'noise') {
      const n = 12, step = Math.floor(t * 1.5);
      for (let i = 0; i < n; i++) {
        const x = W * 0.08 + (i + 0.5) * (W * 0.84 / n), off = type === 'dropout' && rnd(i + step * 31) < 0.3, jit = type === 'noise' ? (rnd(i + step * 17) - 0.5) * 18 : 0;
        dot(x, mid - 10 + jit, 8, off ? c('panel-2') : c('s1'));
        if (off) { line(x - 6, mid - 16, x + 6, mid - 4, c('critical'), 1.6); line(x + 6, mid - 16, x - 6, mid - 4, c('critical'), 1.6); }
      }
      label(type === 'dropout' ? 'a new random 30% switched off at every step (training only)' : 'every number jittered a little at every step (training only)', W / 2, H - 10);
    } else if (type === 'layernorm' || type === 'batchnorm') {
      const rows = 4, cols = 6, cw = W * 0.34 / cols, ch = (H - 30) / rows;
      for (let half = 0; half < 2; half++) {
        const x0 = half ? W * 0.58 : W * 0.06;
        for (let r = 0; r < rows; r++) for (let k = 0; k < cols; k++) {
          const raw = rnd(r * 9 + k) * (k + 1) * 0.4 + r * 0.3, v = half ? clamp(0.5 + (rnd(r * 9 + k) - 0.5) * 0.9, 0, 1) : clamp(raw / 3, 0, 1);
          box(x0 + k * cw, 8 + r * ch, cw - 2, ch - 2, theme.seqColor(v));
        }
        const hl = type === 'layernorm' ? { x: x0 - 2, y: 8 + ((Math.floor(t) % rows) * ch) - 2, w: cols * cw + 2, h: ch + 2 } : { x: x0 + ((Math.floor(t) % cols) * cw) - 2, y: 6, w: cw + 2, h: rows * ch + 2 };
        ctx.strokeStyle = c('s2'); ctx.lineWidth = 2; ctx.strokeRect(hl.x, hl.y, hl.w, hl.h);
      }
      label('→', W / 2, mid, { size: 18, color: ink });
      label(type === 'layernorm' ? 'each example (row) to mean 0, spread 1' : 'each feature (column) over the batch', W / 2, H - 9);
    } else if (type === 'fourier') {
      const X = (u) => W * 0.06 + u * W * 0.88;
      [[1, 's1'], [2.3, 's2'], [4.1, 's4'], [7, 's7']].forEach(([f, col], k) => { const y0 = 14 + k * ((H - 24) / 4) + 6; ctx.strokeStyle = c(col); ctx.lineWidth = 1.6; ctx.beginPath(); for (let i = 0; i <= 120; i++) { const u = i / 120, y = y0 + Math.sin(2 * Math.PI * f * u + t) * 5; i ? ctx.lineTo(X(u), y) : ctx.moveTo(X(u), y); } ctx.stroke(); });
      label('sin and cos of the input at fixed random frequencies', W / 2, H - 6);
    } else if (type === 'attention' || type === 'transformer') {
      if (type === 'transformer') {
        const parts = ['LayerNorm', 'Attention', '+', 'LayerNorm', 'MLP', '+'], bw = (W - 20) / parts.length;
        parts.forEach((p, k) => { const x = 10 + k * bw; if (p === '+') { dot(x + bw / 2, mid, 9, c('s2')); label('+', x + bw / 2, mid, { color: c('panel'), w: 700, size: 13 }); } else { box(x + 3, mid - 14, bw - 6, 28, c(p === 'MLP' ? 's1' : p === 'Attention' ? 's7' : 'panel-2', p === 'LayerNorm' ? 1 : 0.85), c('line-2')); label(p, x + bw / 2, mid, { color: p === 'LayerNorm' ? ink : c('panel'), size: 9.5, w: 600 }); } });
        ctx.strokeStyle = muted; ctx.setLineDash([3, 3]); ctx.beginPath(); ctx.moveTo(10, mid - 22); ctx.lineTo(10 + bw * 2.5, mid - 22); ctx.lineTo(10 + bw * 2.5, mid - 9); ctx.moveTo(10 + bw * 3, mid + 22); ctx.lineTo(10 + bw * 5.5, mid + 22); ctx.lineTo(10 + bw * 5.5, mid + 9); ctx.stroke(); ctx.setLineDash([]);
        label('residual paths: the input is added back after each half', W / 2, H - 8);
      } else {
        const T = 6, x0 = W * 0.1, gap = (W * 0.8) / (T - 1), q = Math.floor(t * 0.8) % T;
        for (let j = 0; j <= q; j++) { const w = rnd(q * 7 + j) + (j === q ? 0.3 : 0); ctx.strokeStyle = c('s7', 0.25 + 0.6 * Math.min(1, w)); ctx.lineWidth = 1 + 4 * w; ctx.beginPath(); ctx.moveTo(x0 + q * gap, mid + 12); ctx.quadraticCurveTo((x0 + q * gap + x0 + j * gap) / 2, mid - 40, x0 + j * gap, mid + 12); ctx.stroke(); }
        for (let j = 0; j < T; j++) { box(x0 + j * gap - 12, mid + 12, 24, 18, j === q ? c('s2') : c('s1', 0.8)); }
        label(`position ${q + 1} looks back at the earlier ones (line width = attention weight)`, W / 2, H - 6);
      }
    } else if (type === 'rnn') {
      const T = 5, gap = (W - 40) / T, k = Math.floor(t * 1.2) % (T + 1);
      for (let j = 0; j < T; j++) { const x = 20 + j * gap + gap / 2; box(x - 16, mid - 14, 32, 28, j < k ? c('s7', 0.85) : c('panel-2'), c('line-2')); label('h' + (j + 1), x, mid, { color: j < k ? c('panel') : muted, w: 600 }); dot(x, mid + 34, 5, c('s1')); line(x, mid + 29, x, mid + 14); if (j) line(x - gap + 16, mid, x - 16, mid, ink, 1.6); }
      label('one step at a time: each state reads its input and the state before it', W / 2, 10);
    } else if (type === 'conv1d' || type === 'seqpool' || type === 'position') {
      const T = 10, cw = (W - 30) / T, y0 = mid - 18;
      for (let j = 0; j < T; j++) box(15 + j * cw + 1, y0, cw - 2, 20, type === 'position' ? theme.divColor(Math.sin(j * 0.9 + t), 0.8) : c('s1', 0.3 + 0.6 * rnd(j)));
      if (type === 'conv1d') { const p = Math.floor(t * 2) % (T - 2); ctx.strokeStyle = c('s2'); ctx.lineWidth = 2.2; ctx.strokeRect(15 + p * cw, y0 - 3, cw * 3, 26); box(15 + (p + 1) * cw + 1, y0 + 30, cw - 2, 14, c('s2')); label('a window of 3 neighbours → one output', W / 2, H - 6); }
      else if (type === 'seqpool') { box(W / 2 - 12, y0 + 34, 24, 16, c('s2')); for (let j = 0; j < T; j++) line(15 + (j + 0.5) * cw, y0 + 20, W / 2, y0 + 34, c('muted', 0.4)); label('the whole sequence → one vector', W / 2, H - 6); }
      else label('a different vector added at each position', W / 2, H - 6);
    } else if (type === 'conv' || type === 'pool' || type === 'avgpool' || type === 'patches' || type === 'flatten' || type === 'gap') {
      const n = type === 'pool' || type === 'avgpool' || type === 'patches' ? 4 : 5, s = Math.min((H - 24) / n, W * 0.36 / n), x0 = W * 0.08, y0 = 8;
      for (let i = 0; i < n; i++) for (let j = 0; j < n; j++) box(x0 + j * s, y0 + i * s, s - 1, s - 1, theme.seqColor(rnd(i * n + j)));
      if (type === 'conv') { const p = Math.floor(t * 2) % ((n - 2) * (n - 2)), pi = Math.floor(p / (n - 2)), pj = p % (n - 2); ctx.strokeStyle = c('s2'); ctx.lineWidth = 2.2; ctx.strokeRect(x0 + pj * s, y0 + pi * s, 3 * s, 3 * s); const ox = W * 0.62; for (let i = 0; i < n - 2; i++) for (let j = 0; j < n - 2; j++) box(ox + j * s, y0 + i * s + s, s - 1, s - 1, i * (n - 2) + j <= p ? c('s2', 0.3 + 0.6 * rnd(i + j * 3)) : c('panel-2')); label('a 3 × 3 filter slides over the picture', W / 2, H - 6); }
      else if (type === 'pool' || type === 'avgpool') { for (let i = 0; i < 2; i++) for (let j = 0; j < 2; j++) { ctx.strokeStyle = c('s2'); ctx.lineWidth = 2; ctx.strokeRect(x0 + j * 2 * s, y0 + i * 2 * s, 2 * s - 1, 2 * s - 1); box(W * 0.64 + j * s * 1.2, y0 + s + i * s * 1.2, s, s, c('s2', 0.75)); } label(type === 'pool' ? 'each 2 × 2 patch → its largest value' : 'each 2 × 2 patch → its average', W / 2, H - 6); }
      else if (type === 'patches') { for (let i = 0; i < 2; i++) for (let j = 0; j < 2; j++) { ctx.strokeStyle = c('s2'); ctx.lineWidth = 2; ctx.strokeRect(x0 + j * 2 * s, y0 + i * 2 * s, 2 * s - 1, 2 * s - 1); box(W * 0.58 + (i * 2 + j) * (s + 6), mid - s / 2, s, s, c('s7', 0.8)); } label('2 × 2 patches → 4 tokens for attention', W / 2, H - 6); }
      else if (type === 'flatten') { for (let k = 0; k < n * n; k++) box(W * 0.5 + (k % 13) * 8, mid - 10 + Math.floor(k / 13) * 10, 7, 8, theme.seqColor(rnd(k))); label('rows laid end to end: one long vector', W / 2, H - 6); }
      else { box(W * 0.62, mid - 9, 18, 18, theme.seqColor(0.5)); label('every position averaged → one number per channel', W / 2, H - 6); }
    } else if (type === 'parallel') {
      // the input splits into branches of different blocks, then they meet again
      const k = 3, x0 = W * 0.1, x1 = W * 0.78, lanes = [['s1', 2], ['s7', 1], ['s4', 3]], ph = Math.floor(t * 1.2) % 4;
      dot(x0, mid, 7, c('s1'));
      lanes.forEach(([col, n2], b) => {
        const y = 16 + b * ((H - 34) / (k - 1));
        ctx.strokeStyle = muted; ctx.lineWidth = 1.2; ctx.beginPath(); ctx.moveTo(x0 + 7, mid); ctx.bezierCurveTo(x0 + 30, mid, x0 + 30, y, x0 + 46, y); ctx.lineTo(x1 - 40, y); ctx.bezierCurveTo(x1 - 20, y, x1 - 20, mid, x1 - 9, mid); ctx.stroke();
        for (let j = 0; j < n2; j++) box(x0 + 54 + j * 34, y - 8, 26, 16, c(col, ph > j ? 0.95 : 0.45), c('line-2'));
        label('ABC'[b], x0 + 40, y - 9, { size: 9.5, w: 700, color: c('s3') });
      });
      dot(x1, mid, 9, c('s3')); label('‖', x1, mid, { color: c('panel'), w: 700, size: 12 });
      label('one input, three branches, joined at the end', W / 2, H - 6);
    } else if (type === 'linear' || type === 'knn' || type === 'tree' || type === 'forest' || type === 'boost') {
      const X = (u) => W * 0.08 + u * W * 0.5, Y = (v) => 10 + (1 - v) * (H - 24);
      const pts = Array.from({ length: 26 }, (_, i) => { const u = rnd(i * 3 + 1), v = rnd(i * 5 + 2); return { u, v, k: v > 0.25 + 0.5 * u + 0.12 * Math.sin(u * 9) ? 1 : 0 }; });
      if (type === 'linear') { line(X(0), Y(0.2), X(1), Y(0.8), c('s2'), 2.2); }
      if (type === 'tree' || type === 'forest' || type === 'boost') { ctx.strokeStyle = c('s2', 0.8); ctx.lineWidth = 1.6; ctx.beginPath(); ctx.moveTo(X(0), Y(0.32)); ctx.lineTo(X(0.35), Y(0.32)); ctx.lineTo(X(0.35), Y(0.48)); ctx.lineTo(X(0.7), Y(0.48)); ctx.lineTo(X(0.7), Y(0.68)); ctx.lineTo(X(1), Y(0.68)); ctx.stroke(); }
      for (const p of pts) dot(X(p.u), Y(p.v), 3.4, c(p.k ? 's1' : 's2'));
      if (type === 'knn') { const q = { u: 0.5 + 0.3 * Math.sin(t * 0.7), v: 0.55 + 0.25 * Math.cos(t * 0.5) }; const near = pts.map((p) => [Math.hypot(p.u - q.u, p.v - q.v), p]).sort((a, b) => a[0] - b[0]).slice(0, 5); ctx.strokeStyle = ink; ctx.lineWidth = 1.2; ctx.beginPath(); ctx.arc(X(q.u), Y(q.v), near[4][0] * W * 0.5 + 4, 0, Math.PI * 2); ctx.stroke(); dot(X(q.u), Y(q.v), 4, ink); for (const [, p] of near) line(X(q.u), Y(q.v), X(p.u), Y(p.v), ink, 0.8); }
      const rx = W * 0.66;
      if (type === 'tree' || type === 'forest') {
        const trees = type === 'forest' ? 3 : 1;
        for (let k = 0; k < trees; k++) { const x = rx + 10 + k * 42, y = 16; dot(x + 12, y, 5, ink); line(x + 12, y, x, y + 22); line(x + 12, y, x + 24, y + 22); dot(x, y + 24, 4.5, c(k % 2 ? 's1' : 's2')); dot(x + 24, y + 24, 4.5, c(k % 2 ? 's2' : 's1')); line(x + 24, y + 24, x + 18, y + 44); line(x + 24, y + 24, x + 30, y + 44); dot(x + 18, y + 46, 4, c('s1')); dot(x + 30, y + 46, 4, c('s2')); }
        label(type === 'forest' ? 'many trees vote' : 'yes / no questions', rx + 60, H - 8);
      } else if (type === 'boost') {
        const n = 6, k = Math.floor(t * 1.5) % (n + 1);
        for (let i = 0; i < n; i++) { const h = (H - 34) * Math.exp(-i * 0.45); box(rx + 6 + i * 20, H - 20 - h, 14, h, i < k ? c('s2', 0.85) : c('panel-2')); }
        label('each tree fixes what is left', rx + 66, H - 8);
      } else if (type === 'linear') label('one weight per input: a straight line', rx + 60, mid);
      else label(`a vote of the 5 nearest`, rx + 60, mid);
    }
  }

  // ------------------------------------------------------------------ the guide
  /* render(type, o): the full guide. o: { part (palette entry: name, group, hint), fits (bool), kind ('points'…),
     add (button element), switchTo (button element offering a problem where it fits) }. */
  // learning from rewards: the methods (not layers; they set how the network you built learns)
  const RL_CODE = {
    dqn: 'q = model(x).gather(1, a[:, None])[:, 0]                 # Q of the moves made\nwith torch.no_grad():\n    best = model(x2).argmax(1)                              # double DQN: the network picks,\n    y = r + gamma * alive * target(x2).gather(1, best[:, None])[:, 0]   # the target network values\nloss = F.smooth_l1_loss(q, y)',
    pg: 'logp = model(x).log_softmax(1)[range(len(a)), a]       # log π(move) for every move played\nadv = (G - G.mean()) / G.std()                         # better or worse than usual\nloss = -(adv * logp).mean()',
    a2c: 'out = model(x)\nlogp = out[:, :n].log_softmax(1)[range(len(a)), a]\nadv = R - out[:, n].detach()                           # rewards + γ·V(where it ended) − V(here)\nloss = -(adv * logp).mean() + 0.25 * ((out[:, n] - R) ** 2).mean()',
    ppo: 'ratio = (logp_new - logp_old).exp()                    # how far the policy has moved\nloss = -torch.min(ratio * adv, ratio.clamp(0.8, 1.2) * adv).mean()\nloss = loss - 0.01 * entropy.mean()                    # a little exploring',
    ga: 'fit = [play(w) for w in population]                    # each network plays\nparents = [tournament(population, fit) for _ in range(2)]\nchild = torch.where(torch.rand(n) < 0.5, *parents)      # each weight from one parent\nchild += (torch.rand(n) < 0.1) * sigma * torch.randn(n) # a few mutate',
    grow: 'child = mutate(copy(parent))\nif random.random() < 0.25:                             # a quarter of the children change shape\n    child = random.choice([add_neuron, remove_neuron, add_layer])(child)\n# add_layer: one neuron, passing on the signal the next layer leans on most;\n# remove_neuron on a layer\'s last neuron removes the layer (folded into the next)\nfitness = total_reward - penalty * n_weights / 100     # of two that play as well, the smaller wins',
    es: 'eps = torch.randn(pairs, n)                            # random directions\nf = [play(theta + sigma * e) - play(theta - sigma * e) for e in eps]   # (ranked, on the page)\ngrad = sum(fk * e for fk, e in zip(f, eps)) / (2 * pairs * sigma)\ntheta += lr * grad                                     # (with Adam, on the page)',
    me: 'cell = style_cell(play(w))                             # e.g. (average speed, side of the track): one of 12 x 12\nif cell not in archive or fit > archive[cell][0]:\n    archive[cell] = (fit, w)                           # the best of each style is kept\nw = mutate(random.choice(list(archive.values()))[1])   # new ones come from any elite',
    novelty: 'styles = [play(w)[1] for w in population]             # what each did: e.g. where it ended up\ndef novelty(s):                                        # mean distance to its k nearest, now and before\n    return mean(sorted(dist(s, t) for t in styles + archive)[1:k + 1])\nfit = [novelty(s) for s in styles]                     # chosen for what is new, not for reward\narchive += most_novel(styles, 2)                       # a place once reached stops being new',
    cem: 'samples = [mu + sd * torch.randn_like(mu) for _ in range(n)]   # a generation from the bell curve\nfit = [play(w) for w in samples]\nelite = torch.stack(best_of(samples, fit, n // 5))     # the best fifth\nmu = elite.mean(0)                                     # the middle moves to them\nsd = (elite.var(0) + floor ** 2).sqrt()                # the spread: how much they disagree',
    erl: 'for w in population:\n    play(w, memory)                                    # every move goes into the replay memory\nfor _ in range(200):\n    dqn_step(q_net, sample(memory, 32))                # the DQN learns from all of them\npopulation = breed(population)                         # the genetic algorithm, as usual\nif gen % 5 == 4:\n    population[-1] = copy(q_net)                       # the DQN joins in place of the worst',
  };
  Object.assign(G, {
    'rl-dqn': {
      what: 'Not a layer: a way to learn from rewards. Your network gives one output per action, its Q-value: the reward it expects in the end if it takes that action now. It plays the largest, and while it learns it sometimes plays at random to find out what else pays.',
      deep: ['Each move goes into a replay memory. Every step, 32 moves drawn from it are learned from at once: each Q-value is pulled toward the reward it got plus γ times the best Q-value of where it landed. A copy of the network (the target network), refreshed every 500 steps, gives those targets so they hold still while it learns.'],
      how: ['Play the move with the largest Q-value, or a random one while exploring (ε-greedy).', 'Remember (state, move, reward, next state).', 'Draw 32 remembered moves; for each, target = reward + γ × the best Q-value of the next state.', 'One gradient step pulls each Q(state, move) toward its target (Huber loss).', 'Every 500 steps, copy the network into the target network.'],
      math: ['target = r + γ · Q_target(s′, argmax Q(s′, ·))', 'loss = Huber(Q(s, a) − target)'],
      params: 'none of its own: it trains the network you built, with a copy of it as the target network',
      use: ['A few actions, and rewards that come from the world: mazes, catching, games. DQN learned Atari games from the screen this way (2013–15).', 'When experience is precious: the replay memory learns from each move many times.'],
      watch: ['Without exploring it may never find what pays.', 'Its play can get worse again for a while: a smaller learning rate steadies it.'],
      code: RL_CODE.dqn, try: 'rlmaze',
    },
    'rl-pg': {
      what: 'A way to learn from rewards that trains the policy itself. Your network gives one score per action; softmax turns them into the probability of playing each. Once it has played whole episodes adding up to at least 64 moves (one long episode, or many short ones), every move it made becomes more likely in proportion to how much better than usual the return after it was, and less likely if worse.',
      how: ['Play whole episodes, drawing each move from the probabilities.', 'For each move, add up the rewards that came after it (discounted by γ): its return.', 'Compare with the batch’s average return: the advantage.', 'One gradient step raises log π(move) by its advantage, plus a little bonus for staying uncertain (entropy).'],
      math: ['∇J = E[ A · ∇ log π(a | s) ]', 'A = (G − mean G) / spread of G'],
      params: 'none of its own: it trains the network you built',
      use: ['Short episodes and actions that should stay a little random.', 'The idea under every policy method since (Williams, 1992).'],
      watch: ['Noisy: it needs many episodes, since a whole episode judges each move.', 'Long episodes blur which moves earned the reward.'],
      code: RL_CODE.pg, try: 'pole',
    },
    'rl-a2c': {
      what: 'Actor-critic: your network gets one more output than there are actions. The actions’ outputs are the policy (the actor); the extra one, the critic, estimates how good the current state is. It plays 8 moves in each of 8 copies of the world, then learns from the advantage: what it got plus the critic’s value of where it ended, minus what the critic expected.',
      how: ['Play 8 moves in each of 8 copies of the world, drawing moves from the policy.', 'At the end, ask the critic how good each copy’s state is.', 'Work back: return R = reward + γ × the next return; advantage A = R − the critic’s value.', 'One gradient step: the actor raises moves with a positive advantage, the critic moves toward R.'],
      math: ['A = r + γ·V(s′) − V(s)', 'loss = −A · log π(a | s) + ¼ (V(s) − R)²'],
      params: 'none of its own; its one extra output (the critic) is part of the output layer',
      use: ['When single moves should be judged without waiting for the end of the episode.', 'Steadier than REINFORCE: the critic is the baseline.'],
      watch: ['A large learning rate throws the policy about: it may never settle.', 'The critic has to learn too: early on its values are guesses.', 'Rewards in the hundreds would give the critic huge errors that crowd out the policy’s share of a clipped step: Normalise rewards (on by default) divides them by their running spread.'],
      code: RL_CODE.a2c, try: 'catch',
    },
    'rl-ppo': {
      what: 'Proximal policy optimisation: actor-critic with a safety catch. It plays 128 moves, then takes 16 small steps on them. Once the probability of a move has changed by 20% from the policy that played it, the clip stops pushing it further, so the policy changes in small, steady steps (the steps for other moves can still carry it a little past). It is the usual choice today, from robots to tuning chatbots with human feedback.',
      how: ['Play 16 moves in each of 8 copies of the world; keep each move’s probability.', 'Work out advantages (GAE, λ = 0.95) and scale them to spread 1.', 'Take 16 steps over the batch in minibatches; each compares the new probability with the old one.', 'Where the ratio is already beyond 0.8–1.2 in the helpful direction, that move adds nothing more.', 'A small bonus for staying uncertain (entropy) keeps it exploring.'],
      math: ['ratio = π_new(a | s) / π_old(a | s)', 'loss = −min(ratio·A, clip(ratio, 0.8, 1.2)·A) + ¼ (V − R)² − 0.01·H(π)'],
      params: 'none of its own; like actor-critic, the critic is one extra output',
      use: ['The default for most problems: steady, and forgiving of its settings.'],
      watch: ['Each batch is reused 4 times: the steps are many but the play is the same 128 moves.', 'Rewards in the hundreds would give the critic huge errors that crowd out the policy’s share of a clipped step: Normalise rewards (on by default) divides them by their running spread.'],
      code: RL_CODE.ppo, try: 'catch',
    },
    'rl-ga': {
      what: 'Evolution instead of gradients: a population of copies of your network (24, or 48 in the bigger worlds), each a little different, play. The best two pass on unchanged; every other place goes to a child of fitter ones, mixed weight by weight, neuron by neuron or not at all, then mutated: a few weights moved at random, or by safe mutations, which move most the weights the moves depend on least. One step is one generation. No loss, no learning rate, no gradient.',
      how: ['Every network in the population plays the same episodes; its fitness is the reward it collects.', 'Keep the best two as they are (elitism).', 'For each other place, pick parents: the fitter of three drawn at random (a tournament), one of the top fifth, or by rank.', 'Mix them (or copy one), then move one weight in ten by a random amount (mutation).'],
      math: ['child = mix(parent a, parent b) + N(0, σ²) on 10% of its weights', 'safe: each nudge ÷ how much the moves depend on that weight'],
      params: 'none of its own; the population is copies of your network’s weights',
      use: ['Small networks, and goals with no gradient to follow: only a score at the end.', 'Neuroevolution: see Evolution in the RL lab.'],
      watch: ['Thousands of weights make most mutations noise: it is slow on big networks.', 'Each generation plays every member, so a step takes longer.'],
      code: RL_CODE.ga, try: 'pole',
    },
    'rl-grow': {
      what: 'A genetic algorithm that evolves the network itself, not only its weights: a child may gain or lose a neuron, or gain a new hidden layer. A new neuron starts with outgoing weights of 0, so it changes nothing until mutation puts it to work. A new layer starts with a single neuron that passes on the signal the next layer leans on most: a bottleneck, so unlike a new neuron it changes what the network does, and selection keeps it only if the child still plays well. Later “+1 neuron” changes widen it without changing anything, and a layer goes when its last neuron is pruned. A small size penalty decides between two that play equally well, so networks grow only when growing pays.',
      deep: ['Starting small and adding one neuron at a time is the idea behind NEAT (Stanley and Miikkulainen, 2002): the smallest change first, kept only if it pays. You can set new layers to start with a few neurons, or as a whole layer as wide as what it reads, passing its input straight through (the older way here), to compare.',
        'Every part of it can be set on the Output block: start from no hidden layer at all (as NEAT does) or from your network; which changes it can make (a neuron more or less, a new layer, a whole layer out at once, a new activation) and whether it leans to grow or to prune; how many children change; where new layers go and how new neurons start; whether pruning takes the weakest neuron or one at random; the largest network; and protecting new shapes, NEAT’s answer to a new part that needs time: for 5 generations the best of a new shape carries on unchanged.'],
      how: ['Every network plays; its fitness is its return, less a little for every 100 weights.', 'The best two pass on unchanged; parents are picked, mixed and mutated as in the genetic algorithm.', 'A quarter of the children also change shape: a neuron more (most often) or less, or a new layer of one neuron.', 'Removing a neuron takes the one the next layer listens to least; removing the last neuron of a layer folds the layer into the next.'],
      math: ['fitness = return − penalty × weights / 100', 'new neuron: incoming weights random, outgoing 0', 'new layer of one neuron: weight 1 on the input j the next layer leans on most (the largest Σ|W₂[:, j]|), read by the next layer with W₂[:, j]', 'last neuron removed: W = W₂W₁, b = W₂b₁ + b₂   (the layer folded into the next)'],
      params: 'none of its own: each member is a network of its own shape (Dense layers, at most 6 of up to 48 neurons unless you set other limits)',
      use: ['To find how big a network a problem needs: start with no hidden layer and watch it grow.', 'Small policies: on the rocket and the track the winners often shrink to a few dozen weights.'],
      watch: ['A new one-neuron layer squeezes everything through one number: right after it appears the child usually plays worse, and most such children are not chosen. Those that are kept widen neuron by neuron.', 'Only a model of Dense blocks on a list of numbers can change shape; any other evolves its weights only.', 'With no size penalty, networks drift bigger: growing costs nothing.'],
      code: RL_CODE.grow, try: 'arena',
    },
    'rl-es': {
      what: 'Evolution strategies: one network, nudged in many random directions at once. Each direction is tried both ways (plus and minus); the directions whose nudge scored better pull the network their way. That is an estimate of the gradient made from scores alone, with no backpropagation, and Adam takes a step along it. In 2017 OpenAI spread it over 1,440 processor cores and taught a simulated humanoid to walk in 10 minutes.',
      how: ['Draw random directions ε, as many as half the population.', 'Play θ + σε and θ − σε for each, from the same starts.', 'Rank all the scores from −½ to ½, so one lucky score cannot run away with the step.', 'Add up the directions weighted by their score differences: the gradient estimate. Adam steps uphill.'],
      math: ['g = (1 / 2nσ) Σₖ (uₖ⁺ − uₖ⁻) εₖ', 'θ ← θ + Adam(g)'],
      params: 'none of its own: it moves the one network you built',
      use: ['A score only at the end of long episodes, where backpropagating rewards is hard.', 'Many processors: every nudged copy plays on its own.'],
      watch: ['It follows the slope where it stands: at a bend where nearly every nudge crashes the same way it stalls (try the race track).', 'σ too small: the scores barely differ; too large: the estimate is for some other network.'],
      code: RL_CODE.es, try: 'track',
    },
    'rl-me': {
      what: 'MAP-Elites: instead of one best network, the best network of every style of play. Two numbers the world measures (on the race track, average speed and which side of the track it keeps) place each network in a grid of 12 × 12 cells, and a new one takes its cell if the cell is empty or it beats the network there. New networks are mutated copies of elites picked at random, so a style that does badly today can lead to one that does well tomorrow.',
      how: ['At first, random networks fill some cells.', 'Then each step: pick elites at random from the grid and mutate copies of them.', 'Play each; measure its style (two numbers from 0 to 1) and its fitness.', 'It goes into its cell if the cell is empty or it beats the elite there.'],
      math: ['cell = (⌊12 · style₁⌋, ⌊12 · style₂⌋)', 'archive[cell] ← child   if fitness(child) > fitness(archive[cell])'],
      params: 'none of its own: the grid keeps up to 144 copies of your network’s weights',
      use: ['To see the many ways a world can be played, not just one.', 'Deceptive problems, where the way to the best goes through styles that score worse at first.'],
      watch: ['The best score climbs more slowly than a genetic algorithm’s: the effort is spread over every style.', 'The styles are the world’s own two measures; the levers have none to tell apart.'],
      code: RL_CODE.me, try: 'track',
    },
    'rl-novelty': {
      what: 'Novelty search (Lehman and Stanley, 2011): a genetic algorithm that ignores the reward. Each network is judged by how different its behaviour is, two numbers the world measures (in the deceptive maze, where it ended up), from the behaviours of its generation and of an archive of earlier ones. Doing something new is what pays, so the population spreads out over every way of behaving, and a goal that every step toward the reward leads away from can still be found. The network on show is the best by score found along the way.',
      deep: ['In the deceptive maze, the reward for getting closer leads straight into a wall below the goal, and a genetic algorithm usually ends up pressed against it (it reached the goal in 1 of 5 runs here; novelty search in 5 of 5). Novelty search fills the maze with endings instead: once the space below the wall is covered, only going round it is new, then the corridor beyond, then the goal.'],
      how: ['Every network plays; the world measures its behaviour: two numbers from 0 to 1 (in the maze, where it ended up).', 'Its novelty: the average distance from its behaviour to the 10 nearest, among its generation and the archive.', 'The two most novel behaviours of each generation join the archive, so a place once reached stops being new.', 'Parents are picked by novelty (or half novelty, half reward), then mixed and mutated as in the genetic algorithm.', 'The generation’s best by reward is scored; if it beats the best so far, it becomes the network on show.'],
      math: ['novelty(x) = (1/k) Σᵢ ‖b(x) − b(μᵢ)‖   (μᵢ: its k nearest behaviours)', 'archive ← archive + the 2 most novel of each generation', 'half and half: fitness = ½·rank(novelty) + ½·rank(reward)'],
      params: 'none of its own: the population is copies of your network’s weights; the archive keeps behaviours, not networks',
      use: ['Deceptive problems, where the reward points the wrong way: the deceptive maze is the classic.', 'To see what a world allows: the archive maps the behaviours it can reach.'],
      watch: ['In a world with many ways to be different that do not matter, it wanders: half novelty, half reward keeps it nearer the goal.', 'The two numbers are everything: a behaviour that misses what matters leads nowhere.', 'The levers have no behaviour to tell apart: there it is the genetic algorithm.'],
      code: RL_CODE.novelty, try: 'trap',
    },
    'rl-cem': {
      what: 'The cross-entropy method: a bell curve (a Gaussian) over the network’s weights, a middle and a spread for each one. Every generation draws networks from it, plays them, and refits the bell curve to the best fifth: the middle moves to their average, and each weight’s spread becomes how much they disagree about it. A floor under the spread keeps it looking. The middle is the network on show.',
      deep: ['It is a cousin of evolution strategies: both move a cloud of networks uphill, but where evolution strategies estimates a gradient from every score, the cross-entropy method simply moves to where the winners were. With few weights it is quick and steady: on the rocket, the Linear template (32 weights) landed in 5 of 5 runs. With many, most random directions do nothing and the winners’ average is mostly noise: the Small MLP (468 weights) landed in 1 of 5.'],
      how: ['Draw a generation of networks: each weight its middle plus its spread times a random number.', 'Play them all from the same starts.', 'Keep the best fifth (the elite).', 'New middle: the elite’s average. New spread: how far the elite’s weights lie from it, never below the floor.'],
      math: ['wₖ = μ + σ ⊙ εₖ,   εₖ ~ N(0, I)', 'μ ← mean of the elite,   σ² ← variance of the elite + floor²'],
      params: 'none of its own: a middle and a spread for each of your network’s weights',
      use: ['Small networks, a few dozen to a few hundred weights.', 'Problems where only a score at the end counts: Tetris players were famously trained this way (Szita and Lőrincz, 2006).'],
      watch: ['Too small a floor and the spread collapses: it stops looking and settles wherever it is.', 'Many weights need many samples: on the Small MLP it is slower than the genetic algorithm.'],
      code: RL_CODE.cem, try: 'lander',
    },
    'rl-erl': {
      what: 'Evolutionary reinforcement learning (ERL, Khadka and Tumer, 2018): a genetic algorithm and a DQN side by side. Every move the population makes, good or bad, goes into a replay memory, and between generations the DQN, a network like yours, learns Q-values from it by gradient steps. Every few generations the DQN’s network is copied into the population in place of the worst member: if it plays well it breeds, so what the gradient learned spreads through evolution, and evolution’s exploring feeds the gradient.',
      how: ['Every network of the population plays; each of its moves goes into the replay memory.', 'Breed the next generation as the genetic algorithm does.', 'The DQN takes 200 gradient steps on batches of 32 moves drawn from the memory (double DQN, with a target network).', 'Every 5 generations the DQN’s network takes the place of the worst member.'],
      math: ['target = r + γ · Q_target(s′, argmax Q(s′, ·))', 'loss = Huber(Q(s, a) − target)', 'every 5 generations: population[worst] ← Q'],
      params: 'none of its own: the population is copies of your network; the DQN is one more copy, with its target network',
      use: ['Worlds where many moves teach something: the DQN learns from all of them, not only from the best.', 'When evolution finds rewards that a gradient method alone would miss, and a gradient method could polish what evolution finds.'],
      watch: ['The DQN learns from moves it did not choose: its Q-values can be off for a while, and its copy simply loses to the population until they are right.', 'Each generation also takes gradient steps: a little slower than the genetic algorithm.'],
      code: RL_CODE.erl, try: 'lander',
    },
  });

  function render(type, o = {}) {
    const g = G[type] || {}, p = o.part || {};
    const canvas = el('canvas', { class: 'pg-canvas', role: 'img', 'aria-label': `How ${(p.name || type).toLowerCase()} works: ${g.what || ''}` });
    const root = el('div', { class: 'pg' },
      el('div', { class: 'pg-head' }, el('div', { class: 'eyebrow', text: `${p.group || ''} · part` }), el('h3', { class: 'pg-title', text: p.name || type }), el('p', { class: 'pg-hint', text: p.hint || '' })),
      el('div', { class: 'pg-visual' }, canvas),
      o.add || o.switchTo ? el('div', { class: 'pg-actions' }, o.add || null, o.switchTo || null, !o.fits && o.kind ? el('p', { class: 'sec-note pg-misfit', text: `It does not suit ${o.kind}: if you add it, the model says why and what would fit.` }) : null) : null,
      ...sections(g));
    animate(canvas, type);
    return root;
  }
  function sections(g, o = {}) {
    const S = (title, ...kids) => el('section', { class: 'insp-sec pg-sec' }, el('h3', { class: 'sec-title', text: title }), ...kids);
    const list = (items, ordered) => el(ordered ? 'ol' : 'ul', { class: 'pg-list' }, ...items.map((t) => el('li', { text: t })));
    const out = [];
    if (g.what && !o.noWhat) out.push(S('What it does', el('p', { class: 'pg-p', text: g.what }), ...(g.deep || []).map((t) => el('p', { class: 'pg-p pg-deep', text: t }))));
    if (g.how) out.push(S('Step by step', list(g.how, true)));
    if (g.math) out.push(S('The maths', el('div', { class: 'fx' }, ...g.math.map((m) => el('div', { class: 'fx-line', text: m })))));
    if (g.params) out.push(S(g.classic ? 'What it learns' : 'Learned numbers', el('p', { class: 'pg-p', text: g.params })));
    if (g.use) out.push(S('When to use it', list(g.use)));
    if (g.watch) out.push(S('Watch out for', list(g.watch)));
    if (g.code && NA.labUI && NA.labUI.codeBox) out.push(S(g.classic ? 'In scikit-learn' : 'In PyTorch', NA.labUI.codeBox(g.code, g.classic ? 'Python · scikit-learn' : 'Python · PyTorch')));
    return out;
  }
  // folded under a block's settings: the same guide, opened on demand
  function details(type, o = {}) {
    const g = G[type];
    if (!g) return el('span');
    const canvas = el('canvas', { class: 'pg-canvas', role: 'img', 'aria-label': `How this part works: ${g.what}` });
    const d = el('details', { class: 'pg-details' }, el('summary', { text: 'Step by step, the maths and the code' }), el('div', { class: 'pg-visual' }, canvas), ...sections(g, { noWhat: true }));
    if (o.open) d.open = true;
    d.addEventListener('toggle', () => { if (d.open) animate(canvas, type); });
    if (d.open) animate(canvas, type);
    return d;
  }
  // the methods of learning from rewards, as moving pictures
  function drawRL(type, ctx, W, H, t, k) {
    const { c, line, dot, box, label, rnd, mid } = k;
    const bars = (x0, w, vals, best, top, bottom, toks) => {
      const n = vals.length, bw = w / n, hi = Math.max(...vals, 1e-9);
      vals.forEach((v, i) => { const h = (bottom - top) * (v / hi); ctx.fillStyle = c(i === best ? 's2' : (toks && toks[i]) || 's1', i === best ? 0.95 : 0.55); ctx.fillRect(x0 + i * bw + 3, bottom - h, bw - 6, h); });
    };
    const phase = (t % 4) / 4;
    if (type === 'rl-dqn') {
      // Q-values for 4 moves; the largest is played; a remembered move travels from the memory to the network
      const q = [0.35 + 0.1 * Math.sin(t), 0.8 + 0.08 * Math.sin(t * 1.3), 0.5 + 0.1 * Math.cos(t), 0.3], best = 1;
      bars(W * 0.08, W * 0.36, q, best, 14, H - 22);
      label('Q-values: it plays the largest', W * 0.26, H - 9, { maxW: W * 0.46 });
      for (let i = 0; i < 6; i++) { ctx.fillStyle = c('ink-2', 0.25 + 0.1 * (i % 2)); ctx.fillRect(W * 0.62, 14 + i * 12, W * 0.28, 9); }
      const p = phase, y = 14 + Math.floor(rnd(Math.floor(t / 4)) * 6) * 12 + 4;
      dot(W * 0.62 + (W * 0.28) * 0.5 - (W * 0.18) * p, y + (H * 0.5 - y) * p, 4, c('s7'));
      label('replay memory', W * 0.76, H - 9, { maxW: W * 0.44 });
    } else if (type === 'rl-pg' || type === 'rl-a2c' || type === 'rl-ppo') {
      // probabilities of 3 moves; the one that paid grows; actor-critic adds the critic's value, PPO a clip band
      const grow = 0.25 + 0.5 * phase, p = [0.33 * (1 - grow) + 0.05, 0.34 + grow * 0.5, 0.33 * (1 - grow) + 0.05];
      bars(W * 0.08, W * 0.36, p, 1, 14, H - 22, ['s1', 's1', 's1']);
      label(type === 'rl-pg' ? 'a move that paid: more likely' : 'actor: move probabilities', W * 0.26, H - 9, { maxW: W * 0.46 });
      if (type === 'rl-pg') {
        const xs = W * 0.58, xe = W * 0.94;
        for (let i = 0; i < 8; i++) { const x = xs + (i / 7) * (xe - xs), r = i === 7 ? 1 : 0; dot(x, mid, 4, c(r ? 'good' : 'muted', 0.9)); if (i) line(x - (xe - xs) / 7 + 5, mid, x - 5, mid, c('muted'), 1); }
        label('an episode, then its return', (xs + xe) / 2, H - 9, { maxW: W * 0.42 });
      } else if (type === 'rl-a2c') {
        const v = 0.3 + 0.4 * (0.5 + 0.5 * Math.sin(t * 1.4)), cx = W * 0.76, r = Math.min(W * 0.12, H * 0.3);
        ctx.strokeStyle = c('line-2'); ctx.lineWidth = 6; ctx.beginPath(); ctx.arc(cx, mid + 6, r, Math.PI, 2 * Math.PI); ctx.stroke();
        ctx.strokeStyle = c('s7'); ctx.beginPath(); ctx.arc(cx, mid + 6, r, Math.PI, Math.PI + Math.PI * v); ctx.stroke();
        label('critic: the state’s value', cx, H - 9, { maxW: W * 0.44 });
      } else {
        const x0 = W * 0.58, x1 = W * 0.94, y = (v) => mid + 4 - (v - 1) * (H * 0.9), lo = y(1.2), hi = y(0.8);
        ctx.fillStyle = c('s7', 0.15); ctx.fillRect(x0, lo, x1 - x0, hi - lo);
        line(x0, lo, x1, lo, c('s7'), 1); line(x0, hi, x1, hi, c('s7'), 1);
        const r = 1 + 0.35 * Math.sin(t * 1.6), rc = Math.min(1.2, Math.max(0.8, r));
        dot(x0 + (x1 - x0) * phase, y(r), 3, c('muted'));
        dot(x0 + (x1 - x0) * phase, y(rc), 4.5, c('s2'));
        label('pushed no further than ±20%', (x0 + x1) / 2, H - 9, { maxW: W * 0.42 });
      }
    } else if (type === 'rl-grow') {
      // a network whose shape changes: a new neuron (its outgoing weights 0: dashed), a new layer of one neuron (it
      // passes on one signal: the line it reads bright, the ones it cuts off faint), widening, pruned to nothing
      const STEPS = [{ s: [2], op: 'grow and prune: the shape evolves with the weights' }, { s: [3], op: '+1 neuron: its outgoing weights start at 0', at: [1, 2] },
        { s: [3, 1], op: '+1 layer of one neuron: it passes on one signal', at: [2, 0], layer: true }, { s: [3, 2], op: '+1 neuron: the new layer widens', at: [2, 1] },
        { s: [4, 2], op: '+1 neuron', at: [1, 3] }, { s: [4, 1], op: '−1 neuron: the one listened to least' }, { s: [4], op: '−1 neuron, its last: the layer is folded into the next' }, { s: [3], op: '−1 neuron' }];
      const st = STEPS[Math.floor(t / 2.2) % STEPS.length], sh = [3, ...st.s, 2], n = sh.length, at = st.at || [-1, -1], fresh = Math.min(1, (t % 2.2) / 1.2);
      const x = (i) => W * 0.12 + (i / (n - 1)) * W * 0.76, y = (i, j) => mid - 8 + (j - (sh[i] - 1) / 2) * Math.min(15, (H - 44) / 3);
      const isNew = (i, j) => i === at[0] && (at[1] < 0 || j === at[1]);
      for (let i = 0; i + 1 < n; i++) for (let a = 0; a < sh[i]; a++) for (let b = 0; b < sh[i + 1]; b++) {
        const zero = isNew(i, a) && at[1] >= 0 && !st.layer, into = st.layer && i + 1 === at[0];
        ctx.setLineDash(zero ? [3, 3] : []); line(x(i), y(i, a), x(i + 1), y(i + 1, b), zero ? c('s2', 0.8) : into ? (a === 0 ? c('s2', 0.9) : c('line-2', 0.25)) : c('line-2'), into && a === 0 ? 1.6 : 1); ctx.setLineDash([]);
      }
      sh.forEach((m, i) => { for (let j = 0; j < m; j++) { dot(x(i), y(i, j), 4.5, c(i === 0 ? 's3' : i === n - 1 ? 's2' : 's1', 0.9)); if (isNew(i, j)) { ctx.strokeStyle = c('s2', 1 - 0.6 * fresh); ctx.lineWidth = 2; ctx.beginPath(); ctx.arc(x(i), y(i, j), 6 + 5 * fresh, 0, Math.PI * 2); ctx.stroke(); } } });
      label(st.op, W / 2, H - 9, { maxW: W - 12 });
    } else if (type === 'rl-es') {
      // a hill of fitness (its top at the right); nudged copies both ways around θ, better ones green; θ steps uphill
      const cx = W * 0.8, cy = mid - 4, R = Math.min(W * 0.3, H * 0.5);
      for (let r = 1; r <= 4; r++) { ctx.strokeStyle = c('line-2'); ctx.lineWidth = 1; ctx.beginPath(); ctx.ellipse(cx, cy, (R * r) / 4 * 1.6, (R * r) / 4 * 0.75, 0, 0, Math.PI * 2); ctx.stroke(); }
      dot(cx, cy, 3, c('muted'));
      const k = Math.floor(t / 1.8) % 6, f = (t % 1.8) / 1.8, P = (j) => { const a = j / 6; return [W * 0.12 + (cx - W * 0.12) * (1 - (1 - a) ** 2), cy + H * 0.24 * (1 - a) * (j % 2 ? 1 : -0.7)]; };
      const [ax, ay] = P(k), [bx, by] = P(k + 1), mv = Math.max(0, (f - 0.6) / 0.4), px = ax + (bx - ax) * mv, py = ay + (by - ay) * mv;
      if (f < 0.75) for (let i = 0; i < 4; i++) {
        const a = rnd(k * 7 + i) * Math.PI * 2, L = 16, ex = Math.cos(a) * L, ey = Math.sin(a) * L * 0.7, d1 = Math.hypot(ax + ex - cx, (ay + ey - cy) * 2), d2 = Math.hypot(ax - ex - cx, (ay - ey - cy) * 2);
        for (const [sx, sy, good] of [[ex, ey, d1 < d2], [-ex, -ey, d2 <= d1]]) { line(ax, ay, ax + sx, ay + sy, c(good ? 'good' : 'critical', 0.7), 1); dot(ax + sx, ay + sy, 3, c(good ? 'good' : 'critical')); }
      }
      dot(px, py, 5, c('s2'));
      label('nudged both ways; it steps toward the better ones', W / 2, H - 9, { maxW: W - 12 });
    } else if (type === 'rl-me') {
      // the grid of styles filling in, each cell the best network of its style (darker: fitter); new elites ringed
      const G = 12, side = Math.min(H - 30, W * 0.42), cs = side / G, gx = W * 0.3 - side / 2, gy = 6, cyc = (t % 9) / 9, fill = 0.06 + 0.84 * cyc;
      for (let cell = 0; cell < G * G; cell++) {
        const i = cell % G, j = Math.floor(cell / G), u = rnd(cell * 3 + 1) * (0.55 + 0.45 * Math.exp(-((i - 7) ** 2 + (j - 5) ** 2) / 30)), on = u < fill * 0.75;
        box(gx + i * cs + 0.5, gy + j * cs + 0.5, cs - 1, cs - 1, on ? theme.seqColor(0.2 + 0.8 * rnd(cell + 77) * Math.min(1, fill * 1.4)) : c('line', 0.3));
        if (on && u > (fill - 0.04) * 0.75) { ctx.strokeStyle = c('s2'); ctx.lineWidth = 1.5; ctx.strokeRect(gx + i * cs + 0.5, gy + j * cs + 0.5, cs - 1, cs - 1); }
      }
      label('average speed →', gx + side / 2, gy + side + 9, { maxW: side + 20 });
      label('144 styles, the best of each kept', W * 0.74, mid, { maxW: W * 0.48 });
    } else if (type === 'rl-novelty') {
      // the deceptive maze from above: endings fill it, the archive faint and this generation bright; the space below
      // the wall fills first, then up past its open end, then along the top to the goal
      const bh = H - 16, bw = Math.min(W * 0.52, bh * 1.25), bx = W * 0.05, by = 6, wy = by + bh * 0.42, gap = bx + bw * 0.8, n = Math.floor(8 + 72 * ((t % 9) / 9));
      box(bx, by, bw, bh, c('panel-2'), c('line-2'));
      line(bx, wy, gap, wy, c('ink'), 2.5);
      for (let i = 0; i < n; i++) {
        const f = i / 80, u = rnd(i * 3 + 1), v = rnd(i * 5 + 2), fresh = i >= n - 8;
        const px = f < 0.5 ? bx + bw * (0.05 + 0.9 * u) : f < 0.7 ? gap + (bx + bw - gap) * (0.15 + 0.7 * u) : bx + bw * (0.1 + 0.85 * (1 - ((f - 0.7) / 0.3) * u));
        const py = f < 0.5 ? wy + 6 + (by + bh - wy - 12) * v : by + 6 + (wy - by - 12) * v;
        dot(px, py, fresh ? 2.8 : 2, fresh ? c('s2') : c('s1', 0.4));
      }
      dot(bx + bw * 0.14, by + bh * 0.16, 5, c('good'));
      dot(bx + bw * 0.14, by + bh * 0.82, 4, c('s3'));
      const rx = bx + bw + (W - bx - bw) / 2, room = W - bx - bw - 10;
      label('the endings spread out:', rx, mid - 9, { maxW: room });
      label('only what is new is chosen', rx, mid + 9, { maxW: room });
    } else if (type === 'rl-cem') {
      // fitness contours (the best at the right); the bell curve (an ellipse) and the networks drawn from it; the best
      // fifth ringed; the next bell curve round them, smaller
      const cx = W * 0.8, cy = mid - 4, R = Math.min(W * 0.3, H * 0.5);
      for (let r = 1; r <= 4; r++) { ctx.strokeStyle = c('line-2'); ctx.lineWidth = 1; ctx.beginPath(); ctx.ellipse(cx, cy, (R * r) / 4 * 1.6, (R * r) / 4 * 0.75, 0, 0, Math.PI * 2); ctx.stroke(); }
      dot(cx, cy, 3, c('muted'));
      const k = Math.floor(t / 1.8) % 6, f = (t % 1.8) / 1.8, P = (j) => [W * 0.14 + (cx - W * 0.14) * (j / 6), cy + H * 0.16 * Math.sin(j * 1.3)], [mx, my] = P(k), [nx, ny] = P(k + 1), s0 = 22 * Math.pow(0.8, k) + 5;
      ctx.strokeStyle = c('s7', 0.8); ctx.lineWidth = 1.5; ctx.beginPath(); ctx.ellipse(mx, my, s0 * 1.4, s0 * 0.8, 0, 0, Math.PI * 2); ctx.stroke();
      const pts = Array.from({ length: 10 }, (_, i) => { const a = rnd(k * 31 + i) * Math.PI * 2, r = Math.sqrt(rnd(k * 17 + i + 5)) * s0; return [mx + Math.cos(a) * r * 1.4, my + Math.sin(a) * r * 0.8]; });
      const elite = pts.map((p, i) => [Math.hypot(p[0] - cx, (p[1] - cy) * 2), i]).sort((a, b) => a[0] - b[0]).slice(0, 2).map((e) => e[1]);
      pts.forEach((p, i) => { const e = elite.includes(i); dot(p[0], p[1], 3, c(e ? 's2' : 'muted')); if (e && f > 0.35) { ctx.strokeStyle = c('s2'); ctx.lineWidth = 1.5; ctx.beginPath(); ctx.arc(p[0], p[1], 6, 0, Math.PI * 2); ctx.stroke(); } });
      if (f > 0.7) { ctx.setLineDash([3, 3]); ctx.strokeStyle = c('s7'); ctx.lineWidth = 1.5; ctx.beginPath(); ctx.ellipse(nx, ny, s0 * 0.8 * 1.4, s0 * 0.8 * 0.8, 0, 0, Math.PI * 2); ctx.stroke(); ctx.setLineDash([]); }
      label('draw from the bell curve; refit it to the best fifth', W / 2, H - 9, { maxW: W - 12 });
    } else if (type === 'rl-erl') {
      // a population (bars) whose moves stream into a replay memory that a DQN learns from; every other generation in
      // the picture, the DQN's copy (its colour) takes the worst place
      const n = 8, x0 = W * 0.04, pw = W * 0.34, bw = pw / n, gen = Math.floor(t / 2.4), ph = (t % 2.4) / 2.4, inj = gen % 2 === 1;
      const fit = Array.from({ length: n }, (_, i) => 0.3 + 0.6 * rnd(i + gen * 7)).sort((a, b) => b - a);
      fit.forEach((v, i) => { const h = (H - 40) * v, dq = inj && i === n - 1; ctx.fillStyle = c(dq ? 's7' : i < 2 ? 's2' : 's1', dq || i < 2 ? 0.95 : 0.55); ctx.fillRect(x0 + i * bw + 2, H - 24 - h, bw - 4, h); });
      const mx = W * 0.46, mw = W * 0.17, qx = W * 0.76, qw = W * 0.2;
      for (let i = 0; i < 6; i++) { ctx.fillStyle = c('ink-2', 0.22 + 0.1 * (i % 2)); ctx.fillRect(mx, 12 + i * 11, mw, 8); }
      box(qx, mid - 15, qw, 30, c('s7', 0.85), c('line-2'));
      label('DQN', qx + qw / 2, mid, { color: c('panel'), w: 700, maxW: qw - 4 });
      dot(x0 + pw + (mx - x0 - pw) * ph, 16 + 50 * rnd(gen * 3 + Math.floor(t * 2)), 3, c('s1'));
      dot(mx + mw + (qx - mx - mw) * ph, 16 + 50 * rnd(gen * 5 + Math.floor(t * 2)) * (1 - ph) + (mid - 16) * ph, 3, c('s3'));
      label('population', x0 + pw / 2, H - 9, { maxW: pw + 8 });
      label('memory', mx + mw / 2, H - 9, { maxW: mw + 16 });
      label(inj ? 'its copy joins' : 'learns', qx + qw / 2, H - 9, { maxW: qw + 16, color: inj ? c('s7') : undefined });
    } else {
      // a population of networks by fitness; the best two kept; the next generation bred from the fitter ones
      const n = 10, x0 = W * 0.06, w = W * 0.88, bw = w / n, gen = Math.floor(t / 2);
      const fit = Array.from({ length: n }, (_, i) => 0.25 + 0.7 * rnd(i + gen * 13) * (0.6 + 0.4 * Math.min(1, gen / 6))).sort((a, b) => b - a);
      fit.forEach((f, i) => { const h = (H - 40) * f; ctx.fillStyle = c(i < 2 ? 's2' : 's1', i < 2 ? 0.95 : 0.55); ctx.fillRect(x0 + i * bw + 3, H - 24 - h, bw - 6, h); });
      label(`generation ${gen + 1}, fittest first · orange: kept as they are`, W / 2, H - 9, { maxW: W - 12 });
    }
  }
  // draws the picture every frame while it is on screen (once, with reduced motion)
  function animate(canvas, type) {
    const t0 = performance.now(), still = NA.prefersReducedMotion && NA.prefersReducedMotion();
    let started = false;
    const frame = (now) => {
      if (started && !canvas.isConnected) return;
      const W = Math.max(200, (canvas.parentElement && canvas.parentElement.clientWidth) || 300), H = 112;
      if (canvas.offsetParent || !started) {
        const ctx = NA.fitCanvas(canvas, W, H);
        try { draw(type, ctx, W, H, still ? 1.5 : (now - t0) / 1000); } catch (e) { /* a picture mid-change */ }
      }
      started = true;
      if (!still) requestAnimationFrame(frame);
    };
    requestAnimationFrame(frame);
  }

  // drawMore: pictures for the guides another file adds to G (js/part-guide2.js)
  NA.partGuide = { G, render, details, draw, inModel, drawMore: {} };
})(window.NA);
