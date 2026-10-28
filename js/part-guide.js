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
  function sections(g) {
    const S = (title, ...kids) => el('section', { class: 'insp-sec pg-sec' }, el('h3', { class: 'sec-title', text: title }), ...kids);
    const list = (items, ordered) => el(ordered ? 'ol' : 'ul', { class: 'pg-list' }, ...items.map((t) => el('li', { text: t })));
    const out = [];
    if (g.what) out.push(S('What it does', el('p', { class: 'pg-p', text: g.what })));
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
    const d = el('details', { class: 'pg-details' }, el('summary', { text: 'How it works, in depth' }), el('div', { class: 'pg-visual' }, canvas), ...sections(g));
    if (o.open) d.open = true;
    d.addEventListener('toggle', () => { if (d.open) animate(canvas, type); });
    if (d.open) animate(canvas, type);
    return d;
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

  NA.partGuide = { G, render, details, draw };
})(window.NA);
