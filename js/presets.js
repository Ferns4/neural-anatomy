/* presets.js — the model zoo: ready-to-inspect networks, each with notes on what to look for. */
(function (NA) {
  'use strict';

  // `try` entries may carry an action: a patch applied through App.applyPatch().
  const PRESETS = [
    {
      id: 'neuron', group: 'Foundations', name: 'Single Neuron', tagline: 'One neuron draws one straight line.',
      data: { dataset: 'blobs', n: 400, noise: 0.15, trainRatio: 0.7, features: ['x1', 'x2'], seed: 1 },
      arch: { hidden: [], act: 'tanh' },
      hyper: { lr: 0.3, optimizer: 'sgd', batch: 16, reg: 'none', regRate: 0, init: 'auto', seed: 1 },
      pretrain: 300,
      lenses: { edges: 'weight', nodes: 'activation', maps: true, values: true, flow: false, bias: true, health: false },
      about: 'This is logistic regression: two inputs, two weights and a bias feeding a sigmoid. The neuron computes z = w₁x₁ + w₂x₂ + b and squashes it into a probability. Every point where z = 0 lies on one straight line, which is the decision boundary.',
      look: [
        'The boundary runs perpendicular to the weight vector (w₁, w₂). Larger weights make the change from orange to blue sharper.',
        'Hover over the Output map. The probe moves with the pointer, and the network shows z and the probability for that point.',
        'In the Neuron tab, drag a weight slider and the line turns; drag the bias and it slides.',
      ],
      try: [
        { text: 'Train on the XOR dataset. No single line separates it, so accuracy stalls near 50%.', label: 'Train on XOR', patch: { dataset: 'xor', train: 600 } },
        { text: 'Keep XOR and add the feature x₁x₂. The product is positive in two quadrants and negative in the other two, so one neuron solves it.', label: 'Add x₁x₂ and train', patch: { dataset: 'xor', features: ['x1', 'x2', 'x1x2'], train: 600 } },
        { text: 'Or keep the raw inputs and add a hidden layer of four neurons, which lets the boundary bend.', label: 'Add hidden layer', patch: { dataset: 'xor', features: ['x1', 'x2'], hidden: [4], train: 1000 } },
      ],
    },
    {
      id: 'xor', group: 'Foundations', name: 'XOR', tagline: 'Why hidden layers exist.',
      data: { dataset: 'xor', n: 400, noise: 0.1, trainRatio: 0.7, features: ['x1', 'x2'], seed: 1 },
      arch: { hidden: [2], act: 'tanh' },
      hyper: { lr: 0.03, optimizer: 'adam', batch: 16, reg: 'none', regRate: 0, init: 'auto', seed: 1 },
      pretrain: 1000,
      lenses: { edges: 'weight', nodes: 'activation', maps: true, values: true, flow: false, bias: false, health: false },
      view: 'output', overlays: { neuronLines: true },
      about: 'XOR puts the same class in opposite corners, which no single line can separate. Two hidden tanh neurons each draw one line, and the output neuron combines them. Seen from the hidden layer, the four clusters are folded so that one straight line separates the classes.',
      look: [
        'Open Hidden space and step from Input to Hidden 1. Both blue clusters land on one side of a line and both orange clusters on the other.',
        'The two neuron lines in the Output view run roughly parallel and carve out a diagonal band.',
        'Each hidden neuron’s map is a soft half-plane. The output is a weighted vote of the two.',
      ],
      try: [
        { text: 'Re-initialize with seed 2. This network gets stuck in a local minimum near 79% training accuracy, a real failure mode of tiny networks.', label: 'Seed 2, retrain', patch: { seed: 2, train: 1000 } },
        { text: 'Drop to one hidden neuron and retrain. One fold is not enough for XOR.', label: 'One neuron', patch: { hidden: [1], train: 1000 } },
        { text: 'Switch to Quadrants, where the classes fill whole quadrants. Two monotonic neurons cannot carve that X-shaped boundary; four ReLUs can.', label: 'Quadrants + 4 ReLU', patch: { dataset: 'quadrants', hidden: [4], act: 'relu', train: 1500 } },
      ],
    },
    {
      id: 'ring', group: 'Foundations', name: 'ReLU Polygon', tagline: 'Each ReLU is a fold along one line.',
      data: { dataset: 'circle', n: 400, noise: 0.1, trainRatio: 0.7, features: ['x1', 'x2'], seed: 1 },
      arch: { hidden: [5], act: 'relu' },
      hyper: { lr: 0.02, optimizer: 'adam', batch: 16, reg: 'none', regRate: 0, init: 'auto', seed: 1 },
      pretrain: 1500,
      lenses: { edges: 'weight', nodes: 'activation', maps: true, values: false, flow: false, bias: false, health: true },
      view: 'output', overlays: { neuronLines: true },
      about: 'Five ReLU neurons each switch on along one side of a straight line, which is where their pre-activation crosses zero. The output neuron adds up these ramps, so the 0.5 contour becomes a polygon wrapped around the inner disc.',
      look: [
        'The neuron lines in the Output view are the five folds. Together they frame the disc.',
        'Each neuron map is flat on one side of its line and ramps up on the other.',
        'Select a hidden neuron and ablate it in the Neuron tab: one side of the polygon opens up.',
      ],
      try: [
        { text: 'Use three hidden neurons and the boundary becomes a triangle. (With seed 1 one of the three dies early and a side stays open; this uses seed 2.)', label: 'Three neurons', patch: { hidden: [3], seed: 2, train: 1500 } },
        { text: 'Remove the hidden layer and add x₁² and x₂². In that feature space the circle is a straight line, so one neuron solves it.', label: 'Squared features', patch: { hidden: [], features: ['x1', 'x2', 'x1sq', 'x2sq'], train: 800 } },
        { text: 'Switch to Tanh and the corners round off.', label: 'Use Tanh', patch: { act: 'tanh', train: 1500 } },
      ],
    },
    {
      id: 'spiral', group: 'Representation', name: 'Deep Spiral', tagline: 'Depth builds features out of features.',
      data: { dataset: 'spiral', n: 500, noise: 0.1, trainRatio: 0.7, features: ['x1', 'x2'], seed: 1 },
      arch: { hidden: [12, 12, 6], act: 'tanh' },
      hyper: { lr: 0.015, optimizer: 'adam', batch: 16, reg: 'none', regRate: 0, init: 'auto', seed: 1 },
      pretrain: 2500,
      lenses: { edges: 'weight', nodes: 'activation', maps: true, values: false, flow: false, bias: false, health: false },
      about: 'Two interleaved spirals need a boundary that winds around more than once. The first layer can only make soft half-planes. Each deeper layer combines the previous layer’s maps into curvier shapes, until the last hidden layer has features that a single output neuron can separate.',
      look: [
        'Read the neuron maps left to right: straight edges in Hidden 1, bands and wedges in Hidden 2, spiral fragments in Hidden 3.',
        'In Hidden space, press Play to watch the two arms untangle layer by layer.',
        'Drag the history timeline back to step 0 to watch the boundary form.',
      ],
      try: [
        { text: 'Keep only one hidden layer of 12 neurons and retrain. A single layer struggles to wind around twice.', label: 'One layer', patch: { hidden: [12], train: 2500 } },
        { text: 'Switch to ReLU. The maps become polygonal.', label: 'Use ReLU', patch: { act: 'relu', train: 2500 } },
        { text: 'Restart with a learning rate of 0.3. The first updates are so large that the tanh neurons saturate, the gradients collapse to almost nothing and the network never recovers.', label: 'Restart at LR 0.3', patch: { lr: 0.3, reset: true, train: 1500 } },
      ],
    },
    {
      id: 'pinwheel', group: 'Representation', name: 'Three Classes', tagline: 'Softmax splits the plane three ways.',
      data: { dataset: 'pinwheel', n: 600, noise: 0.1, trainRatio: 0.7, features: ['x1', 'x2'], seed: 1 },
      arch: { hidden: [12, 12], act: 'relu' },
      hyper: { lr: 0.01, optimizer: 'adam', batch: 16, reg: 'none', regRate: 0, init: 'auto', seed: 1 },
      pretrain: 2500,
      lenses: { edges: 'weight', nodes: 'activation', maps: true, values: false, flow: false, bias: false, health: true },
      about: 'With three classes the output layer has three neurons. Their raw scores (logits) go through softmax, which turns them into probabilities that sum to 1. Each output neuron votes for its own class, and the boundaries fall where two votes tie.',
      look: [
        'Each output neuron’s map shows where its class wins.',
        'The confusion matrix in the Training tab shows which arms get mixed up.',
        'In the Backprop tab, the output error for softmax with cross-entropy is just ŷ − y, one entry per class.',
      ],
      try: [
        { text: 'Use one hidden layer of 4 neurons. Watch the confusion matrix fill in.', label: 'Shrink to 4', patch: { hidden: [4], train: 2500 } },
        { text: 'Raise the noise to 80% and retrain. The arms blur into each other, and the confusion matrix shows which ones.', label: 'Noisier data', patch: { noise: 0.8, train: 2500 } },
      ],
    },
    {
      id: 'sine', group: 'Function fitting', name: 'Curve Fitting', tagline: 'Sums of ReLUs approximate any curve.',
      data: { dataset: 'sine', n: 200, noise: 0.1, trainRatio: 0.7, seed: 1 },
      arch: { hidden: [12], act: 'relu' },
      hyper: { lr: 0.02, optimizer: 'adam', batch: 16, reg: 'none', regRate: 0, init: 'auto', seed: 1 },
      pretrain: 3000,
      lenses: { edges: 'weight', nodes: 'activation', maps: true, values: false, flow: false, bias: false, health: true },
      overlays: { contributions: true },
      about: 'With one input and one output, a ReLU network is a piecewise-linear function. Each hidden neuron contributes a ramp that begins at its kink, the x where its pre-activation crosses zero. The output neuron adds the ramps, weighted, plus a bias.',
      look: [
        'Contributions in the Output view: each thin curve is one neuron’s w·a, and their sum plus the bias is the fit.',
        'The ticks under the x-axis mark the kinks. The fit can only bend there.',
        'A dead neuron contributes a flat line at zero everywhere.',
      ],
      try: [
        { text: 'Load Absolute value with 2 hidden neurons. |x| = ReLU(x) + ReLU(−x), so two neurons can fit it exactly.', label: '|x| with 2 neurons', patch: { dataset: 'abs', hidden: [2], train: 2000 } },
        { text: 'Switch to Tanh for a smooth fit built from S-curves.', label: 'Use Tanh', patch: { act: 'tanh', train: 3000 } },
        { text: 'Load Step. A continuous network can only approximate the jump with a steep ramp.', label: 'Load Step', patch: { dataset: 'step', train: 3000 } },
      ],
    },
    {
      id: 'digits', group: 'Perception', name: 'Digit Recognizer', tagline: 'Hidden neurons learn stroke templates.',
      data: { dataset: 'digits', n: 2500, noise: 0, trainRatio: 0.8, seed: 1 },
      arch: { hidden: [16], act: 'relu' },
      hyper: { lr: 0.005, optimizer: 'adam', batch: 32, reg: 'none', regRate: 0, init: 'auto', seed: 1 },
      pretrain: 900,
      lenses: { edges: 'signal', nodes: 'activation', maps: true, values: false, flow: false, bias: false, health: true },
      about: 'A 14×14 image is 196 input pixels. Each of the 16 hidden neurons has 196 weights, which can be shown as a picture: blue pixels push the neuron up and orange pixels push it down. The 10 output neurons combine these stroke detectors into a score for each digit. The digits are synthetic, drawn from stroke templates with random slant, jitter and pen width.',
      look: [
        'Hidden thumbnails are weight templates. Some look like strokes or loops; others mostly mark where ink should not be.',
        'Draw a digit. Only pixels with ink send a signal, so with Connections set to Signal most input edges vanish.',
        'Turn on Saliency under the drawing pad to see which pixels push the predicted digit up or down.',
      ],
      try: [
        { text: 'Load a 4 from the samples, then draw a line closing its top. Watch the scores shift toward 9.', label: 'Load a 4', patch: { sampleDigit: 4 } },
        { text: 'Find the most important hidden neuron in the Layer tab, then ablate it and check accuracy.', label: 'Open Layer tab', patch: { tab: 'layer' } },
        { text: 'Retrain with 4 hidden neurons. Templates get blurrier and the confusion matrix fills in.', label: '4 hidden neurons', patch: { hidden: [4], train: 900 } },
      ],
    },
    {
      id: 'vanishing', group: 'Failure modes', name: 'Vanishing Gradients', tagline: 'Deep sigmoid stacks starve early layers.',
      data: { dataset: 'circle', n: 400, noise: 0.1, trainRatio: 0.7, features: ['x1', 'x2'], seed: 1 },
      arch: { hidden: [8, 8, 8, 8, 8], act: 'sigmoid' },
      hyper: { lr: 1, optimizer: 'sgd', batch: 16, reg: 'none', regRate: 0, init: 'auto', seed: 1 },
      pretrain: 1500,
      lenses: { edges: 'gradient', nodes: 'gradient', maps: false, values: false, flow: true, bias: false, health: false },
      tab: 'training',
      about: 'Five sigmoid layers deep, trained with plain SGD. A sigmoid’s slope is at most 0.25, so the error signal shrinks every time it passes back through a layer. The first layer receives gradients tens of thousands of times smaller than the last one, so it barely changes and the network never gets off the ground.',
      look: [
        'Connections and Neurons are set to Gradient. The edges fade from right to left.',
        'Training tab → Gradient flow: the per-layer lines sit decades apart on the log scale.',
        'Training accuracy stays at 50%: the network outputs the same guess everywhere, so each split scores whatever share of it belongs to that class.',
      ],
      try: [
        { text: 'Switch the activation to ReLU, whose slope is exactly 1 wherever it is on, and retrain (at a learning rate of 0.1, since the gradients no longer shrink).', label: 'Use ReLU', patch: { act: 'relu', lr: 0.1, train: 1500 } },
        { text: 'Keep sigmoid but switch to Adam, which rescales each weight’s step by its own gradient history and partly compensates.', label: 'Use Adam', patch: { optimizer: 'adam', lr: 0.01, train: 1500 } },
        { text: 'Remove three of the five layers and retrain. The same sigmoid units learn fine when the error only has to pass back through two of them.', label: 'Two layers', patch: { hidden: [8, 8], train: 3000 } },
      ],
    },
    {
      id: 'overfit', group: 'Failure modes', name: 'Overfitting', tagline: 'Memorizing noise instead of the shape.',
      data: { dataset: 'moons', n: 100, noise: 0.9, trainRatio: 0.5, features: ['x1', 'x2'], seed: 1 },
      arch: { hidden: [32, 32], act: 'relu' },
      hyper: { lr: 0.01, optimizer: 'adam', batch: 10, reg: 'none', regRate: 0, init: 'auto', seed: 1 },
      pretrain: 3000,
      lenses: { edges: 'weight', nodes: 'activation', maps: false, values: false, flow: false, bias: false, health: false },
      tab: 'training', overlays: { errors: true },
      about: 'More than a thousand parameters trained on just 50 noisy points. The network can memorize every training point, carving small islands around individual examples. Training loss falls toward zero while validation loss climbs.',
      look: [
        'Training tab: the train and validation loss curves split apart. The step where validation loss bottoms out is when the model was best.',
        'Output view: small pockets around single training points. Hide the validation points to see them clearly.',
        'Drag the history timeline back to where validation loss was lowest. Stopping there is called early stopping. Then run the held-out test to see how the stopped model really does.',
      ],
      try: [
        { text: 'Add L2 regularization at 0.01 and retrain from scratch. The boundary smooths out and validation loss drops from about 3 to under 0.5.', label: 'Add L2 0.01', patch: { reg: 'l2', regRate: 0.01, reset: true, train: 3000 } },
        { text: 'Shrink to one hidden layer of 4 neurons.', label: 'Smaller network', patch: { hidden: [4], train: 3000 } },
        { text: 'Give it more data: 400 points instead of 100.', label: 'More data', patch: { n: 400, train: 3000 } },
      ],
    },
  ];

  // Models that are not plain MLPs open in their own lab (lab.js); `try` entries name lab actions.
  PRESETS.push(
    {
      id: 'transformer', group: 'Language models', lab: 'transformer', name: 'Tiny Transformer', tagline: 'A small language model, from embeddings to attention.',
      meta: 'GPT-style · 2 layers × 4 heads · 28k parameters · 36 words',
      about: 'A GPT-style transformer trained to predict the next word of a tiny made-up language. The embedding layer turns each word into a vector of 32 numbers and adds a vector for its position; two blocks of self-attention and MLPs then let every word gather information from the words before it. The stories are built so that some words can only be predicted by looking back: in "alice likes tea . bob likes milk . alice drinks", the last word depends on what Alice likes, which changes from story to story.',
      look: [
        'Prompt: the model predicts "tea". Change "alice likes tea" to "alice likes juice" and the prediction follows, so it is reading the context, not remembering pairs.',
        'Embeddings: names, drinks, singular and plural animals form their own clusters, although nothing told the model which words belong together.',
        'Training: lookups stay at chance for about 2,000 steps, then jump to 100% within a few hundred steps, when the circuit that combines "who" with "what" forms.',
        'Attention: knock out one of the heads marked Lookup and lookups fall to guessing, while agreement is untouched.',
        'Logit lens: the early stages already know a drink comes next, but often name the wrong drink from the story; only block 2’s MLP settles on the right one.',
        'Architecture: click or zoom into the model, then a block, then its attention and a single head, to see the maths with live numbers; the Python tab shows each part, and the whole model, in PyTorch.',
      ],
      try: [
        { text: 'Switch off the head that matters most for lookups and watch the drink prediction fall apart.', label: 'Knock out the lookup head', action: 'ablateLookup' },
        { text: 'Load a one-layer model trained the same way. With one layer, "drinks" cannot combine who is asking with what they like, so lookups stay near chance.', label: 'One layer', action: 'oneLayer' },
        { text: 'Agreement across a distractor: "the cat near the dogs" should continue with sleeps, runs or sings.', label: 'Try agreement', action: 'prompt', prompt: 'the cat near the dogs' },
        { text: 'Start from random weights and train live. Grammar appears within seconds; lookups take about 2,000 steps (a minute or two).', label: 'Train from scratch', action: 'scratch' },
      ],
    },
    {
      id: 'pinn', group: 'Physics-informed', lab: 'pinn', name: 'Physics-Informed Network', tagline: 'A network that obeys a differential equation.',
      meta: 'PINN · 1–32–32–32–1 tanh · damped oscillator',
      about: 'Ten noisy measurements of a damped spring, all from its first third of a second. A plain network fits them and then has no idea what happens next. A physics-informed neural network (PINN) is trained on the same measurements plus one more loss: how far its curve is from obeying the oscillator equation u″ + 2ζω·u′ + ω²·u = 0 at 40 collocation points. Its derivatives u′ and u″ come from differentiating the network itself, exactly, so the equation can be checked anywhere, even where there is no data.',
      look: [
        'Main plot: both networks pass through the measurements. After the shaded region only the PINN keeps following the true motion (about 5% error against more than 50% for the plain network).',
        'Residual plot: the plain network breaks the equation by orders of magnitude wherever it has no data; the PINN keeps it near zero everywhere.',
        'Phase portrait: plotting u against u′ turns the damped motion into an inward spiral. The PINN traces it; the plain network wanders off.',
        'Architecture: zoom into a hidden layer to see the value and its first and second derivatives flowing through it, and into a neuron for the chain rule with numbers. Every part has its PyTorch code.',
        'Losses tab: data and physics terms pull against each other early on; the curve that satisfies both is the real one.',
      ],
      try: [
        { text: 'Remove the measurements entirely: physics plus the starting conditions u(0) = 1, u′(0) = 0 are enough to solve the equation (about 2% error).', label: 'Physics only', action: 'physicsOnly' },
        { text: 'Less damping (ζ = 0.03): the spring keeps swinging, so guessing beyond the data is even harder for the plain network.', label: 'Less damping', action: 'lowDamping' },
        { text: 'Measure across the whole window: with data everywhere the plain network catches up, and physics matters much less.', label: 'More measurements', action: 'moreData' },
        { text: 'Back to the original problem and the pretrained networks.', label: 'Reset the example', action: 'default' },
      ],
    },
    {
      id: 'diffusion', group: 'Generative models', lab: 'diffusion', name: 'Diffusion Model', tagline: 'Making new data by learning to remove noise.',
      meta: 'DDPM · 2-D points · 100 noise steps · 9k parameters',
      about: 'A denoising diffusion model, the idea behind modern image generators, shrunk to points in a plane. Training takes points of a pinwheel, adds a random amount of Gaussian noise and teaches a small network to predict exactly which noise was added. To generate, start from pure noise and repeatedly remove a little of the predicted noise: over 100 steps the points flow back onto the pinwheel, although no point is copied from the data.',
      look: [
        'Generate: 600 points of pure noise organise into the three arms. Trails follow 40 of them: with DDPM, a point’s position says nothing about which arm it will join until about t = 60, and most have chosen by t = 20. The cloud keeps the same overall spread the whole way; it is rearranged, not shrunk.',
        'Add noise: drag the slider to watch the training data dissolve into a round Gaussian blob. This is the forward process the network learns to undo.',
        'Denoiser: arrows point to where the network thinks each noisy point came from. At high noise they all point toward the middle of the data; at low noise they snap to the nearest arm.',
        'Training tab: the loss is near zero at high noise, where a point is almost all noise, and highest at low-to-middle noise.',
        'Architecture: the training and generating loops as one diagram. Open any part (the forward process, the time features, the network and its layers, the loss, the sampler) for its maths with one training point’s live numbers and its PyTorch code.',
      ],
      try: [
        { text: 'Deterministic sampling (DDIM): no fresh noise on the way back, so trails turn into smooth curves.', label: 'Use DDIM', action: 'ddim' },
        { text: 'Sample in 10 big steps instead of 100: much faster, visibly blurrier.', label: 'Only 10 steps', action: 'fewSteps' },
        { text: 'Drop points into the denoising field at a middle noise level and follow them to the data.', label: 'Explore the field', action: 'field' },
        { text: 'Train a new denoiser on two moons from random weights; samples refresh as it learns.', label: 'Train on two moons', action: 'moons' },
      ],
    },
    {
      id: 'yolo26', group: 'Perception', lab: 'yolo26', name: 'YOLO26 Object Detector', tagline: 'Every object in one pass, with no NMS.',
      meta: 'YOLO26 layout · 385k parameters · 4 classes · 128 px',
      about: 'YOLO26 (Ultralytics, January 2026) is the newest of the YOLO real-time detectors. This is its layer list from yolo26.yaml (Conv, C3k2, SPPF and C2PSA blocks, an FPN-PAN neck and a detection head at three scales) at half the width of YOLO26n, trained here on 614,400 synthetic scenes of circles, triangles, squares and stars. Like the real model it has two heads: a one-to-many head that proposes several overlapping boxes per object and needs non-maximum suppression (NMS) to clean them up, and a one-to-one head trained to put exactly one box on each object, so its output is final as it comes.',
      look: [
        'Detect: drag a shape, resize it with the wheel, double-click to remove it; the boxes follow, computed by the network in your browser.',
        'Compare heads: the one-to-many head proposes several boxes per object, and NMS has to delete the extras. The one-to-one head gives one box per object with no NMS at all; that is what makes YOLO26 end-to-end.',
        'Grids: on pictures this small the fine 16 × 16 grid finds almost every shape; only shapes past about 30 pixels move to the 8 × 8 grid, and the 4 × 4 grid never wins.',
        'Architecture: open the backbone, a block, a convolution, one number of its feature map, and keep going down to the transistors.',
      ],
      try: [
        { text: 'Switch to the one-to-many head and show its raw boxes: every object is covered several times before NMS.', label: 'See the duplicates', action: 'nms' },
        { text: 'Two squares overlapping (IoU 0.35, as much as any pair in training) with a strict NMS threshold of 0.2: NMS deletes one of the real squares. The one-to-one head has no threshold to get wrong and keeps both.', label: 'Overlapping shapes', action: 'overlap' },
        { text: 'Tiny shapes, 7 to 10 pixels across: all found on the finest grid, which YOLO26’s small-target-aware assignment makes sure gets trained on them.', label: 'Small objects', action: 'small' },
        { text: 'Sensor noise at σ = 0.15, three times the most seen in training: the detector finds only 3 of the 6 shapes and adds 3 false boxes.', label: 'Add noise', action: 'noise' },
      ],
    },
    {
      id: 'radar', group: 'Signal processing', lab: 'radar', name: 'Radar Pulse Classifier', tagline: 'Naming a radar waveform from raw I/Q samples.',
      meta: '1-D CNN · 27k parameters · 6 classes · 256 I/Q samples',
      about: 'A receiver samples the radio spectrum as I/Q: 256 complex samples, 12.8 µs at 20 million samples a second. Somewhere in that window there may be a radar pulse, buried in noise. A small 1-D convolutional network reads the raw samples and names the waveform: an unmodulated pulse, a linear-FM chirp, a Barker-13 phase code, a Frank polyphase code, a Costas frequency-hopping code, or nothing but noise. It was trained on about 3 million synthetic signals with random timing, carrier offset, phase, code parameters and signal-to-noise ratio (−12 to +20 dB per sample).',
      look: [
        'Signal: I and Q are two views of one rotating arrow. The Spectrogram shows each waveform’s fingerprint: a flat line for a plain pulse, a slope for a chirp, six steps for a Costas code.',
        'Accuracy: 99% at +2 dB and above, 97% at 0 dB, 77% at −4 dB, and close to guessing below −10 dB. The chirp fades first; a plain pulse holds out longest, because spotting it only takes energy.',
        'What the CNN sees: the class activation map lights up over the pulse and stays dark over the noise around it, although the network was never told where the pulse is.',
        'Architecture: open a layer for its feature maps, then one neuron’s weighted sum over its window of samples, and keep going down to the transistors.',
      ],
      try: [
        { text: 'Lower the SNR to −6 dB: the network still names the waveform about half the time.', label: 'Weak signal', action: 'lowSnr' },
        { text: 'A chirp in the spectrogram: its frequency climbs or falls in a straight line across the pulse.', label: 'Show a chirp', action: 'chirp' },
        { text: 'The two phase codes, Barker-13 and Frank, have almost the same spectrogram; the network tells them apart by their phase jumps.', label: 'Phase codes', action: 'codes' },
        { text: 'Stream signals live: a new intercept every second at a random SNR, with a running score.', label: 'Start the stream', action: 'stream' },
      ],
    });

  const GROUPS = ['Foundations', 'Representation', 'Function fitting', 'Perception', 'Language models', 'Physics-informed', 'Generative models', 'Signal processing', 'Failure modes'];

  NA.presets = { PRESETS, GROUPS, byId: Object.fromEntries(PRESETS.map((p) => [p.id, p])) };
})(typeof globalThis !== 'undefined' ? (globalThis.NA = globalThis.NA || {}) : (window.NA = window.NA || {}));
