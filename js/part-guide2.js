/* part-guide2.js — the guides to the model builder's newest parts (separable convolution, squeeze-and-excitation,
   coordinate channels, random shifts, global max pool, attention pool, the state-space layer, radial basis units,
   maxout and a mixture of experts), in the same form as js/part-guide.js: what each does, in depth, step by step,
   its maths, its learned numbers, when to use it, what to watch out for and PyTorch code; a sentence on what it does
   in your model; and a small moving picture of each. Needs part-guide.js first. */
(function (NA) {
  'use strict';
  const PG = NA.partGuide;
  if (!PG) return;

  Object.assign(PG.G, {
    sepconv: {
      what: 'A convolution in two cheap steps. First a small filter slides over each channel on its own (depthwise); then a 1 × 1 convolution mixes the channels at every pixel (pointwise). It finds much the same patterns as a full convolution with a fraction of the weights, which is why phone-sized networks such as MobileNet are built from it. Dilation spaces the filter’s taps apart, so it sees a wider area for the same cost.',
      deep: [
        'A full convolution with C input channels and F filters of k × k has C·k²·F weights: every filter looks at every channel through its own k × k window. A separable convolution splits that job in two. The depthwise step gives each channel a single k × k filter that looks at that channel only (C·k² weights). The pointwise step is a 1 × 1 convolution, a dense layer run at every pixel, that mixes the C channels into F outputs (C·F weights). With 32 channels in and out and 3 × 3 filters, that is 1,312 weights instead of 9,248.',
        'The bet is that "where are the edges in each channel?" and "how do the channels combine?" can be learned separately. It nearly always pays: Xception (2016) and MobileNet (2017) matched full convolutions on ImageNet with far fewer weights and sums, and MobileNet-style blocks still run in phone cameras.',
        'Dilation spreads the filter’s taps apart: a 3 × 3 filter with dilation 2 covers 5 × 5 pixels with the same 9 weights, skipping every other pixel. Layers with growing dilation see a wide area quickly without pooling, as in DeepLab’s image segmentation. Too much dilation on a small picture leaves holes: thin strokes can fall between the taps.',
      ],
      how: ['Depthwise: slide each channel’s own k × k filter over that channel alone (its taps d pixels apart when dilated), and add a bias.', 'Pointwise: at every pixel, mix the channels with a 1 × 1 convolution into the outputs, plus a bias.', 'Bend the result with the activation.'],
      math: ['m[c, i, j] = b_c + Σ_u,v K[c, u, v] · x[c, i + d·u, j + d·v]', 'y[f, i, j] = act(b_f + Σ_c W[f, c] · m[c, i, j])'],
      params: 'channels × k² + channels (depthwise) + filters × channels + filters (pointwise)',
      use: ['Pictures, in place of a convolution, when weights or speed matter.', 'Dilation 2 or 3 to see further without pooling.', 'With batch norm and squeeze-and-excitation after it: a MobileNet block.'],
      watch: ['On a single channel (a grey picture as input) the depthwise step has little to work with: put a plain convolution first.', 'Large dilations on small pictures skip over thin strokes.'],
      code: 'nn.Conv2d(c_in, c_in, k, padding=d * (k - 1) // 2, dilation=d, groups=c_in)   # depthwise\nnn.Conv2d(c_in, filters, 1)                                                # pointwise', try: 'digits',
    },
    se: {
      what: 'Attention over channels. Each channel is squeezed to one number, its average; a small network turns those numbers into one gate per channel, between 0 and 1; and every channel is multiplied by its gate. For each picture, the network can turn up the features that matter and turn down the rest.',
      deep: [
        'A convolution’s channels are features: one fires on horizontal strokes, another on curves. Which of them matter depends on the picture. Squeeze-and-excitation lets the network decide that example by example. The squeeze averages each channel over all its pixels, C numbers that sum up the picture. The excitation is a small network, C → C/r → C, with a ReLU in the middle and a sigmoid at the end, giving C gates between 0 and 1. Each channel is then multiplied by its gate.',
        'The narrow middle (the ratio r) keeps it cheap and makes it summarise: with 64 channels and r = 4 the gates come from 16 units. The block adds only about 2C²/r weights, yet SENet won the ImageNet challenge in 2017, and the block is part of EfficientNet and MobileNetV3.',
        'It works on sequences too (each feature averaged over the positions) and on plain vectors (each number gated by a small network of the whole vector). One caution on sequences: the average includes the later positions, so a model that predicts the next token would be peeking at the answer.',
      ],
      how: ['Squeeze: average each channel (or feature) over its positions, one number each.', 'Excite: a dense layer of C/r ReLU units, then a dense layer of C sigmoids: one gate per channel.', 'Scale: multiply every number of each channel by its gate.'],
      math: ['s_c = mean over positions of x_c', 'g = σ(W₂ · ReLU(W₁ · s + b₁) + b₂)', 'y_c = g_c · x_c'],
      params: '2 × channels × (channels / r) + channels / r + channels',
      use: ['After a convolution, before pooling: a cheap boost for picture classifiers.', 'Inside MobileNet-style blocks, after a separable convolution.'],
      watch: ['A large ratio on few channels leaves a middle of one unit: little to work with.', 'On models that predict the next token it lets positions see the future.'],
      code: 's = x.mean((2, 3))                                  # squeeze: (batch, channels)\ng = torch.sigmoid(fc2(F.relu(fc1(s))))              # excite: one gate per channel\ny = x * g[:, :, None, None]                         # scale', try: 'shapes',
    },
    coords: {
      what: 'Adds channels that hold each pixel’s column and row, from −1 to 1, and if you like its distance from the centre. A convolution uses the same filter everywhere, so it cannot tell where it is; with these channels the filters after it can.',
      deep: [
        'Weight sharing is a convolution’s strength: one filter finds an edge wherever it appears. It is also a blind spot. Asked "where is the dot?", a stack of convolutions has nothing that tells the left of the picture from the right, apart from the zero padding at the edges. CoordConv (Uber, 2018) showed that convolutions fail even at turning a coordinate into a picture of a dot, and that two extra channels fix it.',
        'The channels are a ramp from −1 at the left to 1 at the right, and one from −1 at the top to 1 at the bottom (and, with radius, the distance from the centre). A filter that reads them can learn to fire only in part of the picture. Where position does not matter, the network can learn to give them weights near 0.',
      ],
      how: ['Make a channel whose every pixel holds its column, scaled to −1…1, and one with its row.', 'With radius, a third channel with each pixel’s distance from the centre, 0…1.', 'Put them after the picture’s own channels.'],
      math: ['out = [x, X, Y] (channels)', 'X[i, j] = −1 + 2j / (W − 1),   Y[i, j] = −1 + 2i / (H − 1)', 'R[i, j] = √(X² + Y²) / √2'],
      params: 'none',
      use: ['Before the first convolution, when the answer depends on where something is: a position, a side, the top or the bottom.', 'Learning from rewards from pixels, where the agent has to know where things are.'],
      watch: ['They make it easier to memorise where things were in the training pictures: pair them with random shifts if position should not matter.'],
      code: 'y, x = torch.meshgrid(torch.linspace(-1, 1, H), torch.linspace(-1, 1, W), indexing="ij")\nout = torch.cat([img, torch.stack([x, y]).expand(len(img), -1, -1, -1)], 1)', try: 'shapes',
    },
    augment: {
      what: 'Data augmentation inside the model. While training, each picture is moved by up to a few pixels in any direction, and can be mirrored left to right; when the model is used, nothing happens. The model never sees the same picture twice, so it learns what stays the same when a picture moves.',
      deep: [
        'With few training pictures a network can memorise them pixel by pixel: the training score climbs while the validation score stalls. Moving each picture by a pixel or two at random gives it endless slightly different copies, all with the same answer. To fit them all it has to learn that a 3 moved one pixel left is still a 3, which is exactly the generalisation that was missing.',
        'This is data augmentation, as old as the first digit readers and used by every image model since AlexNet (2012), which also flipped and cropped its pictures. Here it is a block in the model, like Keras’s RandomTranslation, so it runs on every batch and switches itself off when the model is used.',
        'Choose moves that do not change the answer. Small shifts suit almost any picture. Mirroring suits shapes, animals and faces, but not digits or letters: a mirrored 2 is not a 2. Too large a shift pushes the subject out of the picture.',
      ],
      how: ['For each picture in the batch, draw a move of −s…s pixels across and down (and a coin for the mirror).', 'Move the picture; the pixels that come in from outside are 0.', 'When the model is used, pass the pictures through unchanged.'],
      math: ['training: y[c, i, j] = x[c, i + Δy, j′ + Δx],   Δx, Δy ∈ {−s, …, s}', 'j′ = W − 1 − j when mirrored, otherwise j', 'when used: y = x'],
      params: 'none',
      use: ['First in a picture model, especially with few examples.', 'Together with dropout or noise, for a model that overfits.'],
      watch: ['The training score drops a little and learning is slower: judge it by the validation score.', 'Do not mirror pictures whose meaning depends on the side.'],
      code: 'class RandomShift(nn.Module):           # one move for the whole batch here; the Python tab moves each picture\n    def forward(self, x):\n        if not self.training:\n            return x\n        s = self.shift\n        oy, ox = torch.randint(0, 2 * s + 1, (2,)).tolist()\n        return F.pad(x, (s, s, s, s))[:, :, oy:oy + x.shape[2], ox:ox + x.shape[3]]', try: 'digits',
    },
    gmp: {
      what: 'Keeps the largest value of each channel over the whole picture (or of each feature over a sequence): one number per channel, the strength of its best match anywhere.',
      deep: [
        'After a few convolutions each channel is a map of where one pattern was found. Global max pooling keeps only each map’s highest point: it answers "did this pattern appear anywhere, and how strongly?" and forgets where. Global average pooling instead asks how much of the pattern there is overall.',
        'Max suits detectors: one clear stroke or corner is enough evidence, and a large empty background does not dilute it as it dilutes an average. The average suits textures and amounts. Both make the model indifferent to where things are and to the size of the picture.',
        'Learning goes only through the winners: each channel’s gradient reaches one pixel per picture, so a filter learns from its best match alone.',
      ],
      how: ['For each channel, find the largest of its numbers over all its pixels (or positions).', 'Pass those on: one number per channel.'],
      math: ['y_c = max over i, j of x[c, i, j]'], params: 'none',
      use: ['At the end of a picture model, in place of flatten or global average pooling.', 'After 1-D convolutions on sentences: did a key phrase appear?'],
      watch: ['One stray bright pixel decides the whole channel: noisy inputs suit the average better.'],
      code: 'x.amax(dim=(2, 3))   # pictures;  x.amax(1)  # sequences', try: 'shapes',
    },
    attnpool: {
      what: 'Turns a sequence (or a picture) into one vector by a weighted average of its positions, with weights the model works out for each example. A learned query scores every position, and a softmax turns the scores into weights that add up to 1.',
      deep: [
        'To classify a sequence a model needs one vector for the whole of it. The last step keeps only one position; the average gives every position the same say, filler and all. Attention pooling learns where to look: a query vector q, learned like any weight, is compared with every position x_t (a dot product), a softmax turns the scores into weights a_t that are positive and add up to 1, and the output is Σ a_t x_t.',
        'On a review it can learn to put its weight on the words that carry the opinion, and the weights show you which ones it chose: open them in the Architecture view. It is attention with a single learned query instead of one query per position, much like the [CLS] token of BERT or the attention pooling at the end of CLIP’s image encoder.',
        'It knows no order: shuffle the positions and the output is the same. Put a recurrent layer, a convolution or positions before it when order matters.',
      ],
      how: ['Score every position: s_t = q · x_t.', 'Softmax over the positions: a_t = e^(s_t) / Σ e^(s).', 'Average the positions with those weights: y = Σ a_t x_t.'],
      math: ['s_t = q · x_t', 'a = softmax(s)', 'y = Σ_t a_t x_t'], params: 'features (one query vector)',
      use: ['In place of Sequence pool for sentences and series: it learns which positions matter.', 'On a picture, as a learned alternative to global pooling.'],
      watch: ['It ignores order: put positions, or a layer that reads in order, before it.', 'Early in training its weights are nearly even, so it starts as an average.'],
      code: 'a = self.query(x).softmax(1)     # (batch, positions, 1);  query = nn.Linear(d, 1, bias=False)\ny = (a * x).sum(1)               # (batch, d)', try: 'sentiment',
    },
    ssm: {
      what: 'Reads a sequence one step at a time, like a recurrent layer, but its memory is linear: a set of moving averages of the input, each with its own learned time scale. Quick memories follow the latest steps; slow ones remember far back. The idea behind S4 and Mamba.',
      deep: [
        'Each of the N memories is updated as h = a·h₋₁ + (1 − a)·u: it keeps a share a of what it held and takes in the rest of the new input u, a learned mix of this step’s numbers. A memory with a = 0.5 forgets within a couple of steps; one with a = 0.97 averages over about 30. The values of a are learned (as θ, with a = σ(θ), so they stay between 0 and 1) and start spread from about 2 to 32 steps. The output at each step is a learned weighted sum of the memories, then an activation.',
        'An LSTM squashes its state through gates and tanh at every step; this layer does not, and that is the point. A linear update neither fades nor explodes the gradient over long stretches, and it can be computed for every step at once, as one long convolution or a parallel scan, instead of one step after another. Structured state-space models (S4, 2021) built on this to read sequences tens of thousands of steps long; Mamba (2023) made a depend on the input, so the model can choose what to remember, and rivals transformers on language.',
        'Its weakness is the same linearity: on its own a layer can only average. The activation after it, and stacking layers, add the bends.',
      ],
      how: ['Mix each step’s numbers into N inputs: u_t = B·x_t.', 'Update each memory: h_t = a ⊙ h_(t−1) + (1 − a) ⊙ u_t, starting from 0.', 'Read the output from the memories: y_t = act(C·h_t + c).'],
      math: ['a = σ(θ)   (one per memory, learned)', 'h_t = a ⊙ h_(t−1) + (1 − a) ⊙ B·x_t', 'y_t = act(C·h_t + c)'],
      params: 'state + state × inputs (B) + units × state (C) + units',
      use: ['Series with patterns at several time scales, fast and slow.', 'Long sequences, where an RNN forgets and attention gets slow.'],
      watch: ['One layer only averages: keep the activation, or stack two.', 'It gives a vector at every step: pool it (last step) for one answer.'],
      code: 'a = torch.sigmoid(self.theta)               # (state,), learned\nh, u = torch.zeros(len(x), state), self.B(x)   # u: (batch, T, state)\nfor t in range(T):\n    h = a * h + (1 - a) * u[:, t]\n    ys.append(self.C(h))', try: 'seasonal',
    },
    rbf: {
      what: 'Each unit has a centre and a width, both learned, and answers how close the input is to its centre: 1 at the centre, fading to 0 further away. A layer of them covers the input space with bumps, which the next layer adds up.',
      deep: [
        'A ReLU neuron divides the input space with a straight line and is large on one side of it. A radial basis unit marks out a round region instead: φ = exp(−‖x − c‖² / 2σ²) is 1 at its centre c, about 0.6 at a distance σ and nearly 0 beyond 3σ. With enough of them, well placed, the next layer can build any smooth shape as a sum of bumps: the radial basis function network (Broomhead and Lowe, 1988).',
        'Both the centres and the widths are learned: training moves each centre towards the examples it helps with and widens or narrows its bump. Where they start matters more than for ordinary layers. The centres start spread at random over −1…1, where the builder’s inputs lie, and the widths start at the width you choose.',
        'Because each unit only answers near its centre, the model is local: far from every training example all the units are near 0, and the model falls back on its biases instead of carrying a straight line on. That is calmer, but it means it cannot extend a trend.',
      ],
      how: ['For each unit, measure the squared distance from the input to its centre: d² = Σ (x_i − c_i)².', 'Turn it into a bump: φ = exp(−d² / 2σ²).', 'Pass the bumps on: the next layer weighs and adds them.'],
      math: ['d_j² = ‖x − c_j‖²', 'φ_j = exp(−d_j² / 2σ_j²)', 'σ_j = e^(s_j)   (s_j learned, so σ stays positive)'],
      params: 'units × inputs (the centres) + units (the widths)',
      use: ['Points in a plane, and smooth surfaces with few inputs.', 'Followed by a dense layer or the output, which adds the bumps up.'],
      watch: ['With many inputs the distances all look alike and the bumps blur (the curse of dimensionality).', 'Too narrow a start leaves gaps where no unit answers; too wide blurs everything.'],
      code: 'd2 = ((x[..., None, :] - self.centres) ** 2).sum(-1)   # (batch, units)\nphi = torch.exp(-d2 / (2 * torch.exp(2 * self.log_width)))', try: 'spiral',
    },
    maxout: {
      what: 'Each unit computes a few weighted sums of its inputs, its pieces, and keeps the largest. It needs no activation function: the largest of straight pieces is already a bend, and the unit learns which bend.',
      deep: [
        'A ReLU is max(0, z): the larger of a learned line and a fixed 0. Maxout (Goodfellow and others, 2013) replaces the fixed 0 with more learned lines: a = max(w₁·x + b₁, …, w_k·x + b_k). With two pieces it can become a ReLU (one piece learns to stay at 0), a leaky ReLU or an absolute value |z|; with more pieces, any convex bend made of k straight parts.',
        'Its gradient does not vanish: whichever piece wins passes the gradient straight to its own weights, so maxout units do not die the way ReLU units can. It was designed to go with dropout, and set records on the image benchmarks of its day.',
        'The cost is weights: every unit has k full sets, so a maxout layer of 16 units with 3 pieces is as big as a dense layer of 48. And only the winning piece learns from each example.',
      ],
      how: ['For each unit, compute k weighted sums of the inputs, each with its own weights and bias.', 'Keep the largest: that is the unit’s output.', 'Only the winning piece’s weights learn from this example.'],
      math: ['z_jp = w_jp · x + b_jp   for p = 1 … k', 'a_j = max_p z_jp'], params: 'pieces × units × (inputs + 1)',
      use: ['In place of a dense layer and its activation, when each unit should learn its own shape of bend.', 'With dropout, its original partner.'],
      watch: ['k times the weights of a dense layer as wide.', 'Every piece is straight: it bends only where pieces meet.'],
      code: 'z = self.lin(x).view(*x.shape[:-1], units, pieces)   # lin = nn.Linear(d_in, units * pieces)\na = z.max(-1).values', try: 'moons',
    },
    moe: {
      what: 'Several small expert layers and a gate. For each example the gate scores the experts, and the output is their answers weighted by those scores, often only the top one or two. Different experts can specialise in different kinds of example.',
      deep: [
        'Each expert is an ordinary dense layer with its own weights. The gate is a dense layer with one output per expert; a softmax over its outputs gives weights that add up to 1, and the layer’s output is Σ g_e · expert_e(x). Training shapes both: the gate learns which expert does best on which kind of example, and each expert learns mostly from the examples sent its way. The idea goes back to Jacobs, Jordan, Nowlan and Hinton (1991).',
        'Top-k routing keeps only the k experts with the highest scores and renormalises their weights. Large language models such as the Switch Transformer and Mixtral use it at enormous scale: each token runs through only 1 or 2 of 8 or more expert layers, so the model holds far more weights than it uses for any one token. Here every expert is computed and the unchosen ones are left out, which shows the same thing at a small scale.',
        'The classic failure is collapse: the gate prefers one expert early on, that expert gets all the training and gets better, and the others never learn. Large models add a load-balancing loss that rewards spreading the examples evenly. You can watch for it in the Architecture view, where the gate’s weights show which experts each example used.',
      ],
      how: ['Run every expert on the input: E answers of the same size.', 'Score the experts with the gate; with top-k, keep only the k highest scores.', 'Softmax the kept scores into weights g.', 'Add up the experts’ answers, weighted by g.'],
      math: ['g = softmax(top_k(W_g·x + b_g))', 'y = Σ_e g_e · act(W_e·x + b_e)'],
      params: 'experts × (inputs × units + units) + inputs × experts + experts (the gate)',
      use: ['Problems made of a few different regimes, such as a pinwheel’s arms or a world with distinct situations.', 'To see routing, the idea behind today’s largest language models, at work.'],
      watch: ['Collapse: one expert takes every example. Try all-weighted routing, or train for longer.', 'With top 1 the output jumps where the gate switches experts.'],
      code: 'g = self.gate(x).softmax(-1)                               # (batch, experts)\nouts = torch.stack([e(x) for e in self.experts], -1)       # (batch, units, experts)\ny = (outs * g[:, None, :]).sum(-1)', try: 'pinwheel',
    },
  });

  // ------------------------------------------------------------------ here, in your model
  const n = (v) => Number(v).toLocaleString('en'), T = (s) => s.join(' × ');
  const HERE = {
    sepconv: (c, si, so, P) => { const C = si[0], k2 = c.k * c.k, F = c.filters; return `Here each of the ${C} input channel${C === 1 ? '' : 's'} has its own ${c.k} × ${c.k} filter (${C} × ${k2} weights and ${C} biases), then a 1 × 1 mix makes ${F} channels (${C} × ${F} weights and ${F} biases): ${n(P)} learned numbers, where a full convolution of the same size would need ${n(C * k2 * F + F)}. Out come ${F} maps of ${so[1]} × ${so[2]}.`; },
    se: (c, si, so, P) => { const im = si.length === 3, C = im ? si[0] : si[si.length - 1], h = NA.bx.seHidden(C, c.ratio); return `Here the ${C} ${im ? 'channel' : 'feature'} averages go through ${h} ReLU unit${h === 1 ? '' : 's'} to ${C} gates: ${C} × ${h} + ${h} + ${h} × ${C} + ${C} = ${n(P)} learned numbers. The shape stays ${T(si)}.`; },
    coords: (c, si, so) => `Here it adds ${so[0] - si[0]} channels to the ${si[0]} it receives, ${T(si)} → ${T(so)}, and learns nothing.`,
    augment: (c, si) => `Here, while training, it moves each ${si[1]} × ${si[2]} picture by up to ${c.shift} pixel${c.shift === 1 ? '' : 's'} each way${c.flip ? ' and mirrors half of them' : ''}; when the model is used it passes them unchanged. It learns nothing.`,
    gmp: (c, si, so) => `Here it keeps the largest of each of the ${so[0]} ${si.length === 3 ? `channels' ${si[1]} × ${si[2]} pixels` : `features over the ${si[0]} positions`}, and learns nothing.`,
    attnpool: (c, si, so, P) => { const im = si.length === 3; return `Here a query of ${n(P)} learned numbers scores each of the ${im ? si[1] * si[2] : si[0]} ${im ? 'pixels' : 'positions'}; their weighted average is one vector of ${so[0]}.`; },
    ssm: (c, si, so, P) => `Here ${c.state} memories read ${si[0]} steps of ${si[1]} number${si[1] === 1 ? '' : 's'}: ${c.state} decays, B of ${c.state} × ${si[1]}, C of ${c.units} × ${c.state} and ${c.units} biases, ${n(P)} learned numbers, the same at every step.`,
    rbf: (c, si, so, P) => `Here ${c.units} units each have a centre of ${si[si.length - 1]} numbers and a width: ${c.units} × ${si[si.length - 1]} + ${c.units} = ${n(P)} learned numbers${si.length === 2 ? `, the same at each of the ${si[0]} positions` : ''}.`,
    maxout: (c, si, so, P) => { const F = si[si.length - 1]; return `Here ${c.units} units have ${c.pieces} pieces each, every piece ${F} weights and a bias: ${c.pieces} × ${c.units} × ${F + 1} = ${n(P)} learned numbers${si.length === 2 ? `, the same at each of the ${si[0]} positions` : ''}.`; },
    moe: (c, si, so, P) => { const F = si[si.length - 1], E = c.experts, U = c.units; return `Here ${E} experts of ${U} units (${E} × ${n(F * U + U)} learned numbers) and a gate with ${E} outputs (${n(F * E + E)}): ${n(P)} in all. ${c.top ? `Each ${si.length === 2 ? 'position' : 'example'} uses the top ${c.top}.` : `Every ${si.length === 2 ? 'position' : 'example'} uses every expert.`}`; },
  };
  const baseInModel = PG.inModel;
  PG.inModel = function (b, sh) {
    if (!b || !HERE[b.type] || !sh || !sh.in || !sh.out) return baseInModel(b, sh);
    return HERE[b.type](Object.assign({}, ((NA.bx && NA.bx.BLOCKS[b.type]) || {}).defaults, b), sh.in, sh.out, sh.params || 0);
  };

  // ------------------------------------------------------------------ the pictures
  /* Each one is drawn into ctx (W × H) at time t seconds with part-guide.js's helpers (k): s1 blue for the data, s2
     orange for what the part makes, ink for structure. */
  const D = PG.drawMore;
  const grid = (k, x0, y0, n, s, val) => { for (let i = 0; i < n; i++) for (let j = 0; j < n; j++) k.box(x0 + j * s, y0 + i * s, s - 1, s - 1, k.theme.seqColor(val(i, j))); };

  D.sepconv = (ctx, W, H, t, k) => {
    // three channels, each with its own small filter at the same place; their three results mixed into one output
    const n = 5, s = Math.min((H - 30) / (3 * n) * 1.6, 9), tones = ['s1', 's4', 's7'], p = Math.floor(t * 1.6) % 9, pi = Math.floor(p / 3), pj = p % 3;
    tones.forEach((tn, ch) => {
      const x0 = W * 0.06 + ch * 7, y0 = 6 + ch * (n * s * 0.62);
      grid(k, x0, y0, n, s, (i, j) => k.rnd(ch * 31 + i * n + j) * 0.8);
      ctx.strokeStyle = k.c(tn); ctx.lineWidth = 2; ctx.strokeRect(x0 + pj * s, y0 + pi * s, 3 * s, 3 * s);
      const mx = W * 0.5, my = 14 + ch * 22;
      k.line(x0 + (pj + 3) * s, y0 + (pi + 1.5) * s, mx - 8, my, k.c(tn, 0.6), 1.2);
      k.box(mx - 7, my - 7, 14, 14, k.c(tn, 0.85));
      k.line(mx + 8, my, W * 0.78 - 9, k.mid - 6, k.c('muted'), 1.2);
    });
    k.box(W * 0.78 - 9, k.mid - 15, 18, 18, k.c('s2', 0.9));
    k.label('own filter per channel', W * 0.5, H - 20, { maxW: W * 0.5, size: 9.5 });
    k.label('then a 1 × 1 mix into each output', W / 2, H - 7, { maxW: W - 12 });
  };
  D.se = (ctx, W, H, t, k) => {
    // channel strengths → a small network → gates that turn the channels up or down; two examples take turns
    const C = 8, ex = Math.floor(t / 2.5) % 2, bw = (W * 0.28) / C, top = 12, bot = H - 26;
    const strength = (c) => 0.25 + 0.7 * k.rnd(c * 7 + 3), gate = (c) => k.clamp(0.5 + 0.48 * Math.sin(c * 1.7 + ex * 2.4), 0.04, 1);
    for (let c = 0; c < C; c++) {
      const h = (bot - top) * strength(c);
      ctx.fillStyle = k.c('s1', 0.75); ctx.fillRect(W * 0.04 + c * bw + 1, bot - h, bw - 2, h);
      const g = gate(c), gh = (bot - top) * g;
      ctx.fillStyle = k.c('s7', 0.85); ctx.fillRect(W * 0.43 + c * bw * 0.7 + 1, bot - gh, bw * 0.7 - 2, gh);
      const oh = h * g;
      ctx.fillStyle = k.c('s2', 0.85); ctx.fillRect(W * 0.7 + c * bw + 1, bot - oh, bw - 2, oh);
    }
    k.dot(W * 0.37, k.mid - 6, 4, k.c('ink-2')); k.dot(W * 0.37, k.mid + 6, 4, k.c('ink-2'));
    k.label('channels', W * 0.18, H - 12, { size: 9.5, maxW: W * 0.3 }); k.label('gates', W * 0.52, H - 12, { size: 9.5, maxW: W * 0.2 }); k.label('× gate', W * 0.84, H - 12, { size: 9.5, maxW: W * 0.28 });
  };
  D.coords = (ctx, W, H, t, k) => {
    // the column and row channels as ramps, and a filter that can now read where it is
    const n = 7, s = Math.min((H - 28) / n, W * 0.2 / n), y0 = 6, xs = [W * 0.06, W * 0.37, W * 0.68];
    grid(k, xs[0], y0, n, s, (i, j) => 0.2 + 0.6 * k.rnd(i * n + j) * (Math.abs(i - 3) + Math.abs(j - 3) < 3 ? 1 : 0.2));
    grid(k, xs[1], y0, n, s, (i, j) => j / (n - 1));
    grid(k, xs[2], y0, n, s, (i) => i / (n - 1));
    const p = Math.floor(t * 1.5) % 25, pi = Math.floor(p / 5), pj = p % 5;
    for (const x0 of xs) { ctx.strokeStyle = k.c('s2'); ctx.lineWidth = 2; ctx.strokeRect(x0 + pj * s, y0 + pi * s, 3 * s, 3 * s); }
    k.label('picture', xs[0] + n * s / 2, H - 18, { size: 9.5, maxW: n * s + 10 }); k.label('column', xs[1] + n * s / 2, H - 18, { size: 9.5, maxW: n * s + 10 }); k.label('row', xs[2] + n * s / 2, H - 18, { size: 9.5, maxW: n * s + 10 });
    k.label(`the filter reads where it is: column ${(-1 + (2 * (pj + 1)) / (n - 1)).toFixed(1)}, row ${(-1 + (2 * (pi + 1)) / (n - 1)).toFixed(1)}`, W / 2, H - 5, { maxW: W - 12 });
  };
  D.augment = (ctx, W, H, t, k) => {
    // the same picture, moved a little differently at every training step
    const n = 9, s = Math.min((H - 22) / n, 10), step = Math.floor(t * 1.5), shape = (i, j) => (i === 1 && j >= 2 && j <= 6) || (j === 6 - Math.floor((i - 1) / 2) && i >= 1 && i <= 7);
    for (let f = 0; f < 3; f++) {
      const oy = Math.floor(k.rnd(step * 3 + f) * 5) - 2, ox = Math.floor(k.rnd(step * 7 + f * 5) * 5) - 2, x0 = W * 0.06 + f * (n * s + W * 0.07);
      for (let i = 0; i < n; i++) for (let j = 0; j < n; j++) { const si = i + oy, sj = j + ox; k.box(x0 + j * s, 4 + i * s, s - 1, s - 1, si >= 0 && si < n && sj >= 0 && sj < n && shape(si, sj) ? k.c('s1', 0.9) : k.c('panel-2')); }
    }
    k.label('a new small move for every picture, every training step', W / 2, H - 6, { maxW: W - 12 });
  };
  D.gmp = (ctx, W, H, t, k) => {
    // a channel's map with one bright spot (which wanders); only the largest value goes on
    const n = 6, s = Math.min((H - 24) / n, W * 0.3 / n), x0 = W * 0.08, y0 = 6, pk = { i: Math.floor((Math.sin(t * 0.7) + 1) * 2.4), j: Math.floor((Math.cos(t * 0.5) + 1) * 2.4) };
    grid(k, x0, y0, n, s, (i, j) => (i === pk.i && j === pk.j ? 1 : 0.15 + 0.35 * k.rnd(i * n + j)));
    ctx.strokeStyle = k.c('s2'); ctx.lineWidth = 2.2; ctx.strokeRect(x0 + pk.j * s - 1, y0 + pk.i * s - 1, s + 1, s + 1);
    k.line(x0 + (pk.j + 1) * s + 2, y0 + (pk.i + 0.5) * s, W * 0.68, k.mid - 6, k.c('s2', 0.7), 1.4);
    k.box(W * 0.68, k.mid - 15, 18, 18, k.theme.seqColor(1));
    k.label('each channel → its largest value', W / 2, H - 6, { maxW: W - 12 });
  };
  D.attnpool = (ctx, W, H, t, k) => {
    // positions weighed by a softmax that peaks where the query matches; their weighted average is the output
    const T = 7, x0 = W * 0.08, gap = (W * 0.84) / (T - 1), peak = (Math.sin(t * 0.6) + 1) * 3, w = Array.from({ length: T }, (_, j) => Math.exp(-((j - peak) ** 2) / 1.6)), z = w.reduce((p, v) => p + v, 0), ox = W / 2, oy = 16;
    for (let j = 0; j < T; j++) { const a = w[j] / z; ctx.strokeStyle = k.c('s7', 0.2 + 0.8 * a * 2); ctx.lineWidth = 0.6 + 7 * a; ctx.beginPath(); ctx.moveTo(x0 + j * gap, H - 34); ctx.quadraticCurveTo(x0 + j * gap, oy + 20, ox, oy + 8); ctx.stroke(); }
    for (let j = 0; j < T; j++) k.box(x0 + j * gap - 10, H - 34, 20, 16, k.c('s1', 0.35 + 0.6 * k.rnd(j)));
    k.box(ox - 10, oy - 6, 20, 14, k.c('s2', 0.9));
    k.label('weights from a learned query (line width) → one weighted average', W / 2, H - 7, { maxW: W - 12 });
  };
  D.ssm = (ctx, W, H, t, k) => {
    // an input with jumps; a quick memory follows it, a slow one averages it; the trace is drawn up to now
    const N = 120, X = (i) => W * 0.04 + (i / (N - 1)) * W * 0.92, Y = (v) => k.mid - 8 - v * (H * 0.26), now = Math.floor(((t * 30) % (N + 30)));
    const u = Array.from({ length: N }, (_, i) => (Math.floor(i / 20) % 2 ? 1 : -0.6) + 0.25 * Math.sin(i * 0.9));
    const run = (a) => { let h = 0; return u.map((v) => (h = a * h + (1 - a) * v)); }, fast = run(0.6), slow = run(0.95);
    const path = (vals, col, lw) => { ctx.strokeStyle = col; ctx.lineWidth = lw; ctx.beginPath(); for (let i = 0; i < Math.min(N, now); i++) (i ? ctx.lineTo(X(i), Y(vals[i])) : ctx.moveTo(X(i), Y(vals[i]))); ctx.stroke(); };
    path(u, k.c('muted', 0.6), 1); path(fast, k.c('s1'), 2); path(slow, k.c('s2'), 2.4);
    k.label('input', W * 0.12, 10, { color: k.c('muted'), size: 9.5, align: 'left' }); k.label('a quick memory', W * 0.32, 10, { color: k.c('s1'), size: 9.5, align: 'left' }); k.label('a slow one', W * 0.62, 10, { color: k.c('s2'), size: 9.5, align: 'left' });
    k.label('h = a·h₋₁ + (1 − a)·u: each memory has its own a', W / 2, H - 6, { maxW: W - 12 });
  };
  D.rbf = (ctx, W, H, t, k) => {
    // bumps in a plane; the moving point lights up the ones it is near
    const px = W * 0.5 + Math.cos(t * 0.7) * W * 0.3, py = k.mid - 4 + Math.sin(t * 1.1) * H * 0.22;
    for (let j = 0; j < 7; j++) {
      const cx = W * (0.12 + 0.76 * k.rnd(j * 3 + 1)), cy = 12 + (H - 34) * k.rnd(j * 5 + 2), r = 14 + 14 * k.rnd(j + 9), phi = Math.exp(-((px - cx) ** 2 + (py - cy) ** 2) / (2 * r * r));
      ctx.fillStyle = k.c('s1', 0.08 + 0.6 * phi); ctx.beginPath(); ctx.arc(cx, cy, r, 0, Math.PI * 2); ctx.fill();
      ctx.strokeStyle = k.c('s1', 0.5); ctx.lineWidth = 1; ctx.stroke(); k.dot(cx, cy, 1.8, k.c('ink-2'));
    }
    k.dot(px, py, 4.5, k.c('s2'));
    k.label('each unit answers how close the input is to its centre', W / 2, H - 6, { maxW: W - 12 });
  };
  D.maxout = (ctx, W, H, t, k) => {
    // three learned lines; the unit is their upper envelope, a bend it learned
    const X = (z) => W * 0.08 + ((z + 2) / 4) * W * 0.84, Y = (v) => k.mid - 4 - v * (H * 0.18);
    const lines = [[-0.9, -0.3], [0.15 + 0.15 * Math.sin(t), 0.1], [1.1, -0.6]];
    lines.forEach(([a, b], p) => { ctx.strokeStyle = k.c(['s1', 's4', 's7'][p], 0.55); ctx.lineWidth = 1.2; ctx.beginPath(); ctx.moveTo(X(-2), Y(a * -2 + b)); ctx.lineTo(X(2), Y(a * 2 + b)); ctx.stroke(); });
    ctx.strokeStyle = k.c('s2'); ctx.lineWidth = 3; ctx.beginPath();
    for (let i = 0; i <= 80; i++) { const z = -2 + (4 * i) / 80, v = Math.max(...lines.map(([a, b]) => a * z + b)); i ? ctx.lineTo(X(z), Y(v)) : ctx.moveTo(X(z), Y(v)); }
    ctx.stroke();
    k.label('the largest of a few lines: a bend the unit learns', W / 2, H - 6, { maxW: W - 12 });
  };
  D.moe = (ctx, W, H, t, k) => {
    // a gate sends each example to its top two of four experts; the chosen ones light up and feed the output
    const step = Math.floor(t / 1.4), chosen = new Set([step % 4, (step * 3 + 1) % 4 === step % 4 ? (step + 2) % 4 : (step * 3 + 1) % 4]);
    const gx = W * 0.2, ex = W * 0.5, ox = W * 0.82, ys = [0, 1, 2, 3].map((e) => 12 + e * ((H - 34) / 3));
    k.dot(W * 0.06, k.mid - 4, 5, k.c('s1')); k.line(W * 0.06 + 5, k.mid - 4, gx - 8, k.mid - 4, k.c('muted'), 1.2);
    k.box(gx - 8, k.mid - 14, 16, 20, k.c('s3', 0.85)); k.label('gate', gx, k.mid + 15, { size: 9, maxW: 50 });
    ys.forEach((y, e) => {
      const on = chosen.has(e);
      k.line(gx + 8, k.mid - 4, ex - 14, y, on ? k.c('s3') : k.c('line-2'), on ? 2 : 1);
      k.box(ex - 14, y - 7, 28, 14, on ? k.c('s1', 0.9) : k.c('panel-2'), k.c('line-2'));
      if (on) k.line(ex + 14, y, ox - 9, k.mid - 4, k.c('s1', 0.8), 1.6);
    });
    k.box(ox - 9, k.mid - 13, 18, 18, k.c('s2', 0.9));
    k.label('each example goes to its top 2 experts', W / 2, H - 6, { maxW: W - 12 });
  };
})(window.NA);
