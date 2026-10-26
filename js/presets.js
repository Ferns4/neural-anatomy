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
        { text: 'Re-initialize with seed 2. This network gets stuck in a local minimum near 79% accuracy, a real failure mode of tiny networks.', label: 'Seed 2, retrain', patch: { seed: 2, train: 1000 } },
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
        'Accuracy stays near 50%: the network outputs the same guess everywhere.',
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
      about: 'More than a thousand parameters trained on just 50 noisy points. The network can memorize every training point, carving small islands around individual examples. Training loss falls toward zero while test loss climbs.',
      look: [
        'Training tab: the train and test loss curves split apart. The step where test loss bottoms out is when the model was best.',
        'Output view: small pockets around single training points. Hide the test points to see them clearly.',
        'Drag the history timeline back to where test loss was lowest. Stopping there is called early stopping.',
      ],
      try: [
        { text: 'Add L2 regularization at 0.01 and retrain from scratch. The boundary smooths out and test loss drops from above 4 to under 0.5.', label: 'Add L2 0.01', patch: { reg: 'l2', regRate: 0.01, reset: true, train: 3000 } },
        { text: 'Shrink to one hidden layer of 4 neurons.', label: 'Smaller network', patch: { hidden: [4], train: 3000 } },
        { text: 'Give it more data: 400 points instead of 100.', label: 'More data', patch: { n: 400, train: 3000 } },
      ],
    },
  ];

  const GROUPS = ['Foundations', 'Representation', 'Function fitting', 'Perception', 'Failure modes'];

  NA.presets = { PRESETS, GROUPS, byId: Object.fromEntries(PRESETS.map((p) => [p.id, p])) };
})(typeof globalThis !== 'undefined' ? (globalThis.NA = globalThis.NA || {}) : (window.NA = window.NA || {}));
