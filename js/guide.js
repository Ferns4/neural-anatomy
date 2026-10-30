/* guide.js — the Guide: everything this site shows and explains, in one place, on a computer or a phone. Its
   chapters go from getting around to the electrons in a transistor; the models are listed from the gallery's own
   data and the terms from the glossary's, so it names every one of them. One search runs over all of it, and a
   section can open what it describes. Text about clicking says the right thing for a mouse or a touch screen. */
(function (NA) {
  'use strict';
  const { el, clear } = NA;
  const App = NA.app;
  const G = { open: false };
  const touch = () => !!(NA.touchFirst && NA.touchFirst());
  // a paragraph is a string, or { mouse, touch }: said the way that suits this device
  const say = (p) => (typeof p === 'string' ? p : touch() ? p.touch : p.mouse);

  // ------------------------------------------------------------------ opening what a section describes
  const presets = () => NA.presets.byId;
  // wait for a condition (a model to be on screen), then act; gives up quietly after a few seconds
  const when = (ok, then, tries = 75) => { if (ok()) then(); else if (tries > 0) setTimeout(() => when(ok, then, tries - 1), 80); };
  const isOn = (id) => () => App.preset && App.preset.id === id && (!presets()[id].lab || (NA.labHost.lab && NA.labHost.lab.ui));
  const load = (id, then) => {
    G.close();
    if (!(App.preset && App.preset.id === id)) App.loadPreset(id);
    if (then) when(isOn(id), () => setTimeout(then, 60));
  };
  // a lab's view (its Architecture, its Growth), whichever way that lab switches its views; in a network model
  // the Growth view is a tab of the explanations
  const show = (id, v) => load(id, () => {
    const L = presets()[id].lab ? NA.labHost.lab : null;
    if (!L) { if (NA.mobile && NA.mobile.mobile) NA.mobile.go('insp'); App.emit('tab', v); return; }
    if (NA.mobile && NA.mobile.mobile) NA.mobile.go('network');
    if (L.setCenterView) L.setCenterView(v);
    else if (L.setView) L.setView(v);
    else if (L.layoutCenter) { L.centerView = v; L.layoutCenter(); }
  });
  const arch = (id) => show(id, 'arch');
  const growth = (id) => show(id, 'growth');
  // a neuron of a network model, opened: its levels down to the transistors
  const neuron = (id = 'spiral', l = 1, j = 0) => load(id, () => when(() => NA.netView && NA.netView.layout, () => NA.netDeep && NA.netDeep.openNeuron(l, j)));
  const scale = () => load('spiral', () => when(() => NA.netView && NA.netView.layout, () => { if (NA.mobile && NA.mobile.mobile) NA.mobile.go('network'); setTimeout(() => NA.scaleView && NA.scaleView.show(), 120); }));
  const lab = (id, fn) => load(id, () => fn(NA.labHost.lab));
  const open = (id) => ['Open ' + presets()[id].name, () => load(id)];

  // ------------------------------------------------------------------ the chapters
  const tap = (lower) => (touch() ? (lower ? 'tap' : 'Tap') : (lower ? 'click' : 'Click'));
  const CHAPTERS = [
    { id: 'start', title: 'Getting around', sections: () => [
      { title: 'What this site is', paras: ['An interactive lab for neural networks. Pick a model, train it right here, and watch what every neuron computes and how it learns. Then zoom into any part of it: a neuron opens into its weighted sum and its code, and the numbers open into their bits, the adders and logic gates that compute with them, the transistors that make the gates, and the electrons in a transistor’s channel.',
        'Every model trains and runs in your browser, on your own device. Most open already trained, with notes on what to look for and experiments to try; nothing needs installing.'],
      acts: [['Open the gallery', () => { G.close(); NA.chrome.openZoo(); }]] },
      { title: 'Picking a model', paras: ['The model name at the top opens the gallery: every model of the site, in groups from Foundations to Reinforcement learning, each with a small picture of what it does. The Models chapter of this Guide lists them all with what each one shows.',
        'The network models (the first groups) share one page: the network in the middle, the data and its maps beside it, the explanations on the right. The labs (a transformer, a diffusion model, an object detector and more) each have a page of their own with an Architecture view.'] },
      { title: 'Training', paras: ['The play button trains and pauses (Space), the step button applies one training step (S), the fast-forward button trains 500 more steps quickly, and the reset button starts again from new random weights (R). The settings button (the sliders) holds the learning rate, the optimizer, the batch size and the rest. The labs have the same controls for their own training.',
        'Once a model is trained, Run held-out test (T) scores it on examples it never saw while training: the honest measure of how well it learned.'] },
      { title: 'Seeing what a part is, and opening it', paras: [
        { mouse: 'Point at a neuron or a connection to see what it computes. Click it to select it: the panel on the right explains it. In a zoom view, click a part to see what it is, with its code; double-click it, zoom in on it, or press Enter to open it. Esc or Backspace goes back one level, and the path at the top goes back further. The mouse wheel zooms and dragging pans.',
          touch: 'Tap a neuron, or a part of any zoom view, to open it. To see what it is instead, hold a finger on it for half a second: a ring fills under your finger, then the explanation opens. Pinch to zoom, drag to pan, and use ← or the path at the top to go back.' },
        { mouse: 'On a touch screen it is the other way round: a tap opens, holding a finger on a part shows it.', touch: 'With a mouse it is the other way round: a click shows a part, a double-click opens it.' }] },
      { title: 'On a phone', paras: ['The page shows one screen at a time: the bar at the bottom switches between the diagram, the data and the explanations (Network, Output and Inspect for a network model; each lab has its own three, such as Board, Agent and Inspect in the reinforcement learning lab). The settings open as a sheet from the sliders button.',
        'To keep the site like an app: in Safari tap Share, then Add to Home Screen; on Android open the browser menu and tap Install app.'] },
      { title: 'Sharing a model', paras: ['The link button copies a link to the model on screen with its settings, so whoever opens it sees the same model the same way.'] },
      { title: 'Help, the tour and the glossary', paras: ['? opens how to read the network and the keyboard shortcuts. From there, Take the tour points at each part of the page in turn, and Glossary lists every term the site uses. In the notes, a quietly underlined word is a glossary term: ' + tap(true) + ' it for its definition.'],
        acts: [['Take the tour', () => { G.close(); NA.tour && NA.tour.start(); }], ['Open the glossary', () => { G.close(); NA.glossary && NA.glossary.open(); }]] },
    ] },
    { id: 'learn', title: 'How a network learns', sections: () => [
      { title: 'A neuron', paras: ['A neuron multiplies each input by a weight, adds them up with a bias, and passes the sum through an activation function. One neuron can only draw one straight line across its inputs: Single Neuron shows that line turning and sliding as the weights learn.'], acts: [open('neuron')] },
      { title: 'Activation functions', paras: ['The activation bends the sum. ReLU keeps a positive sum and turns a negative one into 0; tanh and sigmoid squash it into a range; others sit in between. Without a bend, any stack of layers would still draw only straight lines. The chip under each layer picks its function, and a selected neuron can have its own.'] },
      { title: 'Hidden layers and depth', paras: ['Hidden layers combine simple pieces into complicated ones. XOR cannot be split by one line, so it needs a hidden layer; with ReLU each hidden neuron folds the plane along one line, and the folds add up to a polygon (ReLU Polygon). Deeper layers build features out of features, enough to untangle a spiral (Deep Spiral).'], acts: [open('xor'), open('ring'), open('spiral')] },
      { title: 'Outputs and the loss', paras: ['The output layer turns the last sums into a prediction: softmax gives a probability for each class (Three Classes), a single sigmoid a yes or no, and a plain sum a number. The loss scores how wrong the prediction is: cross-entropy for classes, squared error for numbers. Training is nothing but making the loss smaller.'], acts: [open('pinwheel')] },
      { title: 'Gradients and backpropagation', paras: ['For every weight, the gradient says how much the loss would change if that weight moved a little. Backpropagation works out all of them in one backward pass, applying the chain rule from the output back toward the input. The Backprop tab walks through one training step phase by phase, with the real numbers; the gradient lenses show the same on the diagram, in violet.'] },
      { title: 'Gradient descent and optimizers', paras: ['Each step moves every weight a little against its gradient; the learning rate sets how far. SGD follows the gradient of a small batch of examples, momentum keeps a running velocity so steady directions speed up, and Adam scales each weight’s step by the size of its own recent gradients. The loss landscape shows the path the weights took.'] },
      { title: 'Batches, epochs, validation and test', paras: ['A batch is the handful of examples one step learns from; an epoch is one pass through all of them. Part of the data is held back. The validation set is scored while training, to show how well the model does on examples it does not learn from; the held-out test set is scored only when you run the test, once you have finished changing things.'] },
      { title: 'Fitting a curve', paras: ['A network can approximate any curve. With ReLU, each hidden neuron adds one kink, and the output adds the pieces up: Curve Fitting builds a wave that way.'], acts: [open('sine')] },
      { title: 'Learning to see', paras: ['Digit Recognizer reads handwritten digits from their pixels. Its hidden neurons learn stroke templates, a loop here and a bar there, which the Maps lens draws inside each neuron.'], acts: [open('digits')] },
      { title: 'When training goes wrong', paras: ['Overfitting: the model learns the training examples, noise and all, and does worse on new ones; validation loss rises while training loss falls. L2 regularization, dropout and early stopping fight it (Overfitting).',
        'Vanishing gradients: in a deep stack of sigmoid layers the gradient shrinks at every layer on its way back, and the early layers barely learn (Vanishing Gradients). Dead ReLUs (always 0, so they never learn), saturated neurons (stuck on a flat tail) and steps that blow up show in the Health lens and in Insights, with what to do about them.'], acts: [open('overfit'), open('vanishing')] },
      { title: 'Watching it learn: Growth', paras: [
        'Every model can show how it learned: the Growth tab of the network models, and the Growth view of each lab, the model builder and the reinforcement learning examples. It keeps a film of the training, a moment each time the count of steps passes 1, 2, 5, 10, 20… (a log scale: a model changes fastest at the start). Play runs the moments as a timelapse; the strip, the slider and the learning curve pick one. A model trained in advance learns again from scratch for its film.',
        'The curve names the stages it went through: warming up, learning fast and polishing, and where they show, memorising (validation getting worse while training improves) or slipping back (a policy forgetting what worked). Inside it draws how big each layer’s weights are and how far they have moved from their start. The math at each moment writes out the loss, the rule of one training step (gradient descent, momentum, Adam, or the update of Q-learning, a policy gradient or evolution) and the numbers it used then: the loss, the learning rate, the size of the gradient, the change since the moment before. YOLO26 and the radar classifier trained offline in PyTorch: their films are their training logs, a moment per epoch.'],
        acts: [['Growth in Overfitting', () => growth('overfit')], ['Growth in the transformer', () => growth('transformer')], ['Growth in Maze Runner', () => growth('rl-maze')]] },
    ] },
    { id: 'read', title: 'Reading the network', sections: () => [
      { title: 'Colours', paras: ['Blue is positive and orange is negative everywhere: weights, activations, signals, and the two classes of a yes-or-no problem. Violet measures gradients: the pressure on a number to change.'] },
      { title: 'Lenses', paras: ['The connections can show their weights, the signal flowing through them for the current input, or their gradients (keys 1, 2, 3); the neurons their output, their sum before the activation, or their gradient (4, 5, 6).',
        'Maps (M) draws inside each neuron what it outputs for every possible input, so you can watch simple shapes in the first layer combine into complicated ones deeper in. Values (V), Bias (B), Flow (F), Functions (A) and Health (H) add the numbers, the biases, the signal flowing through, each neuron’s function, and warnings about neurons in trouble.'] },
      { title: 'The probe and the output map', paras: [{ mouse: 'One input is always being traced through the network: the probe. Hover over the output map to move it, click to pin it, or click a data point to use that example. The Neuron tab shows its arithmetic.', touch: 'One input is always being traced through the network: the probe. Drag on the output map to move it, or tap a data point to use that example. The Neuron tab shows its arithmetic.' }] },
      { title: 'Hidden space and the loss landscape', paras: ['Hidden space shows how each layer rearranges the data, layer after layer, until the classes come apart. The loss landscape maps the loss around the current weights along the two directions training moved in most, with the path it took across that map.'] },
      { title: 'The explanations', paras: ['Insights says what is going on and suggests experiments, some of which ask you to predict the result first. Neuron and Layer give the numbers of what is selected, Training the loss curves, and Backprop one training step worked through. On a phone they are the Inspect screen.'] },
      { title: 'History', paras: ['The timeline under the diagram keeps the training history: step back through it (← and → with the timeline focused), and Train from here goes on from an earlier step.'] },
      { title: 'Changing the network', paras: ['− and + under a layer change its size; the + between layers, or the Hidden layers control, add depth; the chip under each layer picks its activation function, and a selected neuron can have its own. The maps and the numbers follow at once.'] },
    ] },
    { id: 'models', title: 'The models', generated: 'models' },
    { id: 'labs', title: 'The labs', sections: () => [
      { title: 'Model Builder', paras: ['Build a model from parts and train it: drag parts between the Input and the Output (or ' + tap(true) + ' them), drag blocks to reorder them, and drag one off to remove it. There are dense layers and neurons, convolutions and pooling, attention, recurrent cells (RNN, LSTM, GRU), normalization and dropout, gated layers (GLU, SwiGLU, GEGLU), and parallel branches joined side by side or by adding, multiplying or averaging. A few blocks can be boxed into a part of your own and reused, like a function.',
        'Each part has its own guide, with a small moving picture of what it does. What is happening explains a run in plain words, challenges set a goal to reach, classic models (trees, boosting and others) train on the same data for comparison, runs can be compared side by side, and the Python tab writes the whole script in PyTorch. Its Architecture view draws your model as a network or as boxes, and Growth films each run with the math of its steps.',
        'Learn from rewards has problems with no examples at all: levers that pay when they match a light, a maze, a pole to balance, a ball to catch, a race track, a drone arena seen in the first person, and a rocket to land on a pad. Each can be seen more than one way (a few numbers, a picture for convolutions, or the last few readings for a recurrent layer), and the network you build learns by DQN, REINFORCE, actor-critic, PPO, a genetic algorithm, evolution strategies or MAP-Elites, chosen on the Output block. With grow and prune on, evolution reshapes the network itself, a neuron or a layer at a time; and on problems with answers, the Training tab can grow and prune a network while it learns. Its inputs and Its reward change the world itself: leave inputs out or add sensor noise, set what each event pays or write the reward as a formula of your own, and change the pole’s length, push and gravity, the levers’ odds, the longest episode or the ball’s drift. The results play it live with its outputs beside the world, the score is how its best moves do at the real goal, and the Python tab writes the world and the method’s whole loop.'],
      acts: [open('builder'), ['Its Architecture view', () => arch('builder')], ['Its Growth view', () => growth('builder')], ['Balance a pole', () => lab('builder', (L) => L.setProblem('pole'))], ['Race a car', () => lab('builder', (L) => L.setProblem('track'))]] },
      { title: 'Tiny Transformer', paras: ['A small language model that runs in your browser. Type a prompt and follow it through the token embeddings, the attention layers and the MLP blocks, to the probability of each next token. The Tokens view shows every token through every stage; the embedding views map tokens, output vectors and positions; the Architecture view opens every part, down to one attention head and what each position looks at. Growth films it learning again from scratch: grammar within seconds, lookups after about 2,000 steps.'],
        acts: [open('transformer'), ['Its Architecture view', () => arch('transformer')], ['Its Growth view', () => growth('transformer')]] },
      { title: 'Physics-Informed Network', paras: ['Ten noisy measurements of a damped spring, all from its first third of a second: a plain network fits them and has no idea what happens next. The physics-informed network is also trained on how far its curve is from obeying the spring’s equation at 40 points in between, using the network’s own exact derivatives, so it carries on correctly where there is no data. The Plots view shows the fits, the residual and the phase portrait; the Architecture view opens the network and its loss; Growth films both networks learning, with the PINN’s loss written out term by term.'],
        acts: [open('pinn'), ['Its Architecture view', () => arch('pinn')], ['Its Growth view', () => growth('pinn')]] },
      { title: 'Diffusion Model', paras: ['It makes new data by learning to remove noise. Add noise shows the forward process that buries the data in noise; Denoiser shows where the network thinks each noisy point came from; Generate turns pure noise into new points step by step, with DDPM (random) or DDIM (deterministic) sampling. Growth films noise learning to become the data.'],
        acts: [open('diffusion'), ['Its Architecture view', () => arch('diffusion')], ['Its Growth view', () => growth('diffusion')]] },
      { title: 'YOLO26 Object Detector', paras: [{ mouse: 'It finds every object in a scene in one pass. In Detect, drag a shape to move it, scroll on it to resize it and double-click to remove or add one: the boxes follow, computed in your browser. Compare heads puts the one-to-one head (no NMS needed) beside the one-to-many head with NMS; Grids shows the three scales it predicts at; Architecture opens the whole detector; Growth plays its training log, epoch by epoch, with its loss and the MuSGD optimiser written out.', touch: 'It finds every object in a scene in one pass. In Detect, drag a shape to move it, pinch to resize it and twist to turn it, and double-tap to remove or add one: the boxes follow, computed on your phone. Compare heads puts the one-to-one head (no NMS needed) beside the one-to-many head with NMS; Grids shows the three scales it predicts at; Architecture opens the whole detector; Growth plays its training log, epoch by epoch, with its loss and the MuSGD optimiser written out.' }],
        acts: [open('yolo26'), ['Its Architecture view', () => arch('yolo26')], ['Its Growth view', () => growth('yolo26')]] },
      { title: 'Radar Pulse Classifier', paras: ['A convolutional network that names a radar waveform from its raw I/Q samples. Signal shows the samples, Spectrogram how the frequency changes over time, What the CNN sees the network’s feature maps, and Accuracy how well it does at each signal-to-noise ratio, from −12 to +20 dB. Growth plays its training log: clean signals are learned first, noisy ones last.'],
        acts: [open('radar'), ['Its Architecture view', () => arch('radar')], ['Its Growth view', () => growth('radar')]] },
      { title: 'Reinforcement learning', paras: ['Five examples where an agent learns from rewards alone: a maze with a table, mazes with a network, two board games learned by playing itself, and a network bred by evolution to balance a pole. The next chapter but one has them in full.'], acts: [open('rl-maze'), open('rl-mazes'), open('rl-evolve')] },
    ] },
    { id: 'down', title: 'All the way down', sections: () => [
      { title: 'Zoom views', paras: [`Every model has a zoom view: the labs’ Architecture view, and in the network models any neuron (zoom far into it, or ${touch() ? 'tap' : 'double-click'} it). Each level fills the screen and is computed from the live model, so the numbers inside are the real ones for the current input. ${NA.openHint ? NA.openHint('a part', { cap: true }) : ''}.`],
        acts: [['Open a neuron', () => neuron('spiral', 1, 0)]] },
      { title: 'From a model to the electrons', paras: ['The levels go: the model and its blocks; a layer; one neuron; its weighted sum and the PyTorch line that computes it; what that library call does inside; one floating-point multiplication or addition, bit by bit (sign, exponent and mantissa); the partial products and the adders that sum them; full adders; logic gates; the transistors of a gate (CMOS); and the electrons moving through a transistor’s channel. In the Architecture views every heatmap cell and bar opens the same way.'] },
      { title: 'Where numbers live', paras: ['Any stored number opens into its bits, and a bit into the memory cell that holds it: in DRAM, one transistor and one capacitor, read through a word line, a bit line and a sense amplifier. Each number says whether it is an input, a learned weight or an activation, and so how long it has to stay in memory.'] },
      { title: 'Merge', paras: ['Merge keeps the level of detail you were at when you zoom back out. The parts around it are drawn the same way, side by side, each framed and named: stored numbers join into one memory array with its word lines and bit lines, gates and transistors appear inside their circuit’s own picture, and the full adders of an adder line up as a carry chain.'] },
      { title: 'Linked words', paras: [{ mouse: 'In the explanations, words that name a part of the picture are linked to it: point at the words and the part lights up; point at the part and its words light up.', touch: 'In the explanations, words that name a part of the picture are linked to it: tap the words and the part lights up.' }] },
      { title: 'Follow a weight', paras: ['At a neuron’s weighted sum, Follow a weight ▶ takes one learned weight on a guided walk: down through the multiplication to the transistors that compute it, and back up to the loss, one level at a time.'], acts: [['Open a neuron', () => neuron('spiral', 1, 0)]] },
      { title: 'Flip bits and number formats', paras: [`At a stored weight, turn on Flip bits and ${tap(true)} any bit to flip it: the network changes at once, and a flipped exponent bit can wreck it (↶ Undo puts it back). The format menu stores every weight as bfloat16, FP16 or FP8 (E4M3) instead of float32, to see how much precision a network really needs; float32 brings the originals back exactly.`] },
      { title: 'The adder’s clock', paras: ['At an adder, ⏱ Step the clock runs an addition one gate delay at a time: the carry ripples along, and the counts show how many gates and transistors it takes.'] },
      { title: 'Build your own full adder', paras: ['At a full adder, Build your own opens a board where you wire logic gates from A, B and carry-in to the sum and carry-out, with a truth table that checks all 8 cases as you go. Put it to work and every adder in the zoom views is made of your circuit, right or wrong: a mistake shows up in the multiplications and in the neurons above them.'] },
      { title: 'Zooming out past the network', paras: ['Zoom out below the network’s smallest size and the view keeps going: your network shrinks to a dot beside the models of this site’s labs, all of them fit in a corner of the smallest GPT-2, and GPT-2 in a corner of one GPU’s memory. Each square has the area of its number of weights.'], acts: [['Zoom out past the network', () => scale()]] },
      { title: 'Code at every level', paras: ['The Python tab beside each explanation shows that level as code: the PyTorch line of a neuron, a library routine written out, a float multiplication in plain Python, a full adder as logic. In a lab it starts with the whole model.'] },
    ] },
    { id: 'rl', title: 'Reinforcement learning', sections: () => [
      { title: 'Learning from rewards', paras: ['No examples of good play: the agent only gets rewards (reaching a goal, winning a game) and has to find out what earns them. Its Architecture view draws it as a loop: what it sees, its model, its choice, the answer of the maze or the game, and how it learns from that. Each box opens; the model opens into its layers and neurons, down to the transistors.'],
        acts: [['The agent’s loop', () => arch('rl-mazes')]] },
      { title: 'Maze Runner: Q-learning', paras: ['A Q-table holds one number for each square and move: the reward it expects if it makes that move there and then plays its best. After every move, that entry moves toward the reward it got plus γ times the best value of where it landed, so values spread back from the goal until the arrows show a route. ε sets how often it explores at random, α how far each update moves, γ how much later rewards count. On a slippery floor it learns to keep away from the pits. It learns this maze, not mazes: Train on can give it a new random maze every episode, and a table learns nothing from that, since its rows are squares, not situations.'],
        acts: [open('rl-maze'), ['Watch it learn', () => lab('rl-maze', (L) => L.runTry({ action: 'watch' }))]] },
      { title: 'Many Mazes: a Q-network', paras: ['A table fits one maze: its rows are that maze’s squares. The Q-network is not told its square at all. It sees the 24 squares around it (walls, pits, the goal, where it has been) and which way the goal lies, so the same weights work in any maze. Every training episode is a new random maze, easy ones first (while it trains, the board shows them), with a replay memory and a target network as DQN did on Atari games. It never trained on the five named mazes and gets through all of them, and through most random mazes nobody has seen. Switch Train on to the maze shown only and it overfits: it keeps that maze and loses the rest.'],
        acts: [open('rl-mazes'), ['A maze nobody has seen', () => lab('rl-mazes', (L) => L.newMaze())]] },
      { title: 'Three-line and Four-line: self-play', paras: ['Board games made for this site: three in a row on a 4 × 4 board, and four in a row on 5 × 5. The agent plays both sides and learns by temporal differences: after each move, the value of its previous position moves toward the value of the next one, and at the end toward the result. Three-line can keep a table of the positions it has met or use a small network, which learns faster since what it learns about one position carries over to similar ones; Four-line has far too many positions for a table.'],
        acts: [open('rl-three'), open('rl-four')] },
      { title: 'Testing it slowly', paras: [`Test run (in a maze) and Watch it play (in a game, against a rule-follower that wins when it can and blocks when it must) play its best moves back one step at a time: pause, step back and forth (or use the arrow keys), and change the speed. Beside the board, Inside its model shows what the model computes at that step: what it sees, which hidden neurons fire, and what comes out. In a game, ${touch() ? 'touching' : 'pointing at'} an empty cell shows the network on that move.`],
        acts: [['Test run on a maze', () => lab('rl-mazes', (L) => L.startTest())], ['Watch Four-line play', () => lab('rl-four', (L) => L.startTest())]] },
      { title: 'Playing against it', paras: [`Play blue (first) or Play orange (second) above the board, or ${tap(true)} an empty cell to start as blue. It always plays its best move by what it has learned so far: train it more, and it gets harder to beat.`],
        acts: [['Play Four-line', () => lab('rl-four', (L) => L.startPlay(1))]] },
      { title: 'Evolution: a genetic algorithm', paras: ['No gradients and no reward for any single move: a population of 30 small networks (4 inputs, 8 tanh neurons, 2 outputs: push the cart left or right) takes turns balancing a pole, and each one’s fitness is how long the pole stayed up. The best two carry on unchanged; every other place goes to a child of two fit parents, with each weight from one or the other and a few mutated. The bars under the cart are the last generation, fittest first; the Training tab follows the best, the average and the worst.',
        `Try it yourself balances the pole with your own hands: hold ← or → (or press either half of the board), then see how long the network lasts from the same start. The experiments switch off mutation or the elite, or shrink the population, to see what each part of the algorithm does.`],
        acts: [open('rl-evolve'), ['Watch it evolve', () => lab('rl-evolve', (L) => L.runTry({ action: 'watch' }))], ['Try it yourself', () => lab('rl-evolve', (L) => L.startPlay())]] },
      { title: 'In the model builder', paras: ['Learn from rewards puts the same ideas in your hands: build any network, pick a world (levers, a maze, a pole, catching a ball, a race track, a drone arena seen in the first person, landing a rocket), how it sees it, and how it learns: DQN, REINFORCE, actor-critic, PPO, a genetic algorithm, evolution strategies or MAP-Elites. Its challenges are about rewards themselves: exploring, a bonus for getting closer, looking further ahead, memory, a learning rate that is too large, finding the way through mazes it has never seen, getting round a track, a slope for evolution to climb, and a network that grows its own layers.',
        'Evolution in the builder shows a whole generation at once: every car of a generation driving the track from the same start, coloured as their bars, or every rocket coming down together. Its settings are on the Output block: how parents are chosen (a tournament, the top fifth, by rank), how they are mixed (weight by weight, neuron by neuron, or not at all), and plain or safe mutations. Grow and prune lets a child gain or lose a neuron or a whole layer, each change harmless at first (a new neuron’s outgoing weights are 0, a new layer passes values straight through), with a size penalty so that it grows only when growing pays. MAP-Elites keeps the best network of each of 144 styles of play; a click on a cell of its grid watches that style play. Try it yourself, above the picture, hands you the controls of any world (the arrow keys, or buttons to hold on a touch screen), with the trained network’s outputs beside each of your moves.',
        'You set the world too. Its inputs: leave some out (the pole’s speeds, the ball’s drift) or add sensor noise, and see whether the network can do without. Its reward: what each event pays (the goal, a pit, every step, a fall), or a formula of your own over the world’s numbers, such as 1 − abs(angle)/12 for the pole. The agent only ever collects reward, so the reward is the goal it is given; the score keeps measuring the real goal, and a reward that pays for something else shows there. The world itself: the pole’s length, push and gravity (Moon to Jupiter), the levers’ odds, the longest episode, the ball’s drift.'],
        acts: [['Learn from rewards', () => lab('builder', (L) => L.setProblem('rlmaze'))]] },
    ] },
    { id: 'keys', title: 'Keys and gestures', sections: () => [
      { title: 'Keyboard', keys: [['Space', 'Train or pause (in a test of the RL lab: play or pause it)'], ['S', 'One training step (in a test: one step on)'], ['R', 'Start again from new random weights'], ['T', 'Run the held-out test (in the RL lab: a test run)'], ['1  2  3', 'Connections: weight, signal, gradient'], ['4  5  6', 'Neurons: activation, pre-activation, gradient'], ['M  V  B  F  A  H', 'Maps, values, bias, flow, functions, health'], ['Enter', 'Open the selected part of a zoom view'], ['Esc  ·  Backspace', 'Back out one level'], ['←  →', 'Step through the history (with the timeline focused), or through a test'], ['Ctrl Z  ·  Ctrl Y', 'Undo and redo in the model builder (Delete removes the selected block)'], ['?', 'How to read the network, and these keys']] },
      { title: 'Mouse', keys: [['Point', 'What a neuron or a part is, in a tooltip'], ['Click', 'Select it: the explanation shows it'], ['Double-click', 'Open it'], ['Wheel', 'Zoom; far enough in, the part under the pointer opens'], ['Drag', 'Pan']] },
      { title: 'Touch', keys: [['Tap', 'Open a neuron or a part of a zoom view (a part that does not open is shown instead)'], ['Hold', 'See what it is: the explanation opens'], ['Pinch', 'Zoom; far enough in, the part under your fingers opens'], ['Drag', 'Pan'], ['Double-tap', 'Zoom the network diagram in or out']] },
    ] },
    { id: 'terms', title: 'Glossary', generated: 'terms' },
  ];

  // the gallery and the glossary, from their own data
  function generated(kind) {
    if (kind === 'models') {
      const { PRESETS, GROUPS } = NA.presets;
      return GROUPS.flatMap((g) => PRESETS.filter((p) => p.group === g).map((p) => ({ title: p.name, kicker: g, paras: [p.tagline, p.about].filter(Boolean), meta: p.meta, acts: [['Open', () => load(p.id)]] })));
    }
    const T = (NA.glossary && NA.glossary.terms) || [];
    return T.slice().sort((a, b) => a.term.localeCompare(b.term)).map((t) => ({ title: t.term, paras: [t.def], term: true }));
  }
  const sectionsOf = (c) => (c.generated ? generated(c.generated) : c.sections());
  const textOf = (s) => [s.title, s.kicker, s.meta, ...(s.paras || []).map(say), ...(s.keys || []).flat()].filter(Boolean).join(' ').toLowerCase();

  // ------------------------------------------------------------------ the dialog
  function build() {
    G.search = el('input', { class: 'gd-search', type: 'search', placeholder: 'Search the guide', 'aria-label': 'Search the guide', spellcheck: 'false', autocomplete: 'off' });
    G.toc = el('nav', { class: 'gd-toc', 'aria-label': 'Chapters' });
    G.main = el('div', { class: 'gd-main', tabindex: '-1' });
    G.count = el('p', { class: 'gd-count', 'aria-live': 'polite' });
    G.closeBtn = el('button', { class: 'icon-btn', type: 'button', 'data-close': '', 'aria-label': 'Close the guide' }, '×');
    G.dlg = el('div', { class: 'modal gd-modal', id: 'guide', role: 'dialog', 'aria-modal': 'true', 'aria-labelledby': 'gdTitle', hidden: '' },
      el('div', { class: 'sheet gd-sheet' },
        el('div', { class: 'sheet-head gd-head' },
          el('div', null, el('h2', { class: 'sheet-title', id: 'gdTitle', text: 'Guide' }), el('p', { class: 'sheet-sub', text: 'Everything this site shows and explains, and how to use it.' })),
          G.closeBtn),
        el('div', { class: 'gd-find' }, G.search, G.count),
        el('div', { class: 'gd-body' }, G.toc, G.main)));
    G.dlg.addEventListener('click', (e) => { if (e.target === G.dlg || e.target.closest('[data-close]')) G.close(); });
    G.dlg.addEventListener('keydown', (e) => { if (e.key === 'Escape') { e.stopPropagation(); G.close(); } });
    G.search.addEventListener('input', () => render());
    // the chapter being read is lit in the list of chapters
    G.main.addEventListener('scroll', () => { if (!G.ticking) { G.ticking = true; requestAnimationFrame(() => { G.ticking = false; spy(); }); } }, { passive: true });
    document.body.append(G.dlg);
  }
  function render() {
    const q = G.search.value.trim().toLowerCase(), words = q.split(/\s+/).filter(Boolean);
    clear(G.toc); clear(G.main);
    let shown = 0;
    for (const c of CHAPTERS) {
      const secs = sectionsOf(c).filter((s) => !words.length || words.every((w) => textOf(s).includes(w)));
      if (!secs.length) continue;
      shown += secs.length;
      G.toc.append(el('button', { class: 'gd-tab', type: 'button', dataset: { ch: c.id }, onclick: () => jump(c.id) }, c.title, words.length ? el('span', { class: 'gd-n', text: String(secs.length) }) : null));
      const body = c.id === 'terms' ? el('dl', { class: 'gd-terms' }, ...secs.flatMap((s) => [el('dt', { text: s.title }), el('dd', { text: s.paras[0] })])) : el('div', { class: 'gd-secs' }, ...secs.map(section));
      G.main.append(el('section', { class: 'gd-ch', id: 'gd-' + c.id, dataset: { ch: c.id } }, el('h2', { class: 'gd-ch-title', text: c.title }), body));
    }
    if (!shown) G.main.append(el('p', { class: 'gd-none', text: 'Nothing in the guide matches. Try another word, or fewer.' }));
    G.count.textContent = words.length ? `${shown} ${shown === 1 ? 'section matches' : 'sections match'}` : '';
    spy();
  }
  function section(s) {
    return el('article', { class: 'gd-sec' },
      s.kicker ? el('p', { class: 'gd-kicker', text: s.kicker }) : null,
      el('h3', { class: 'gd-sec-title', text: s.title }),
      ...(s.paras || []).map((p) => el('p', { text: say(p) })),
      s.meta ? el('p', { class: 'gd-meta', text: s.meta }) : null,
      s.keys ? el('dl', { class: 'gd-keys' }, ...s.keys.flatMap(([k, v]) => [el('dt', null, ...k.split('  ').map((x) => el('kbd', { text: x.trim() }))), el('dd', { text: v })])) : null,
      s.acts && s.acts.length ? el('div', { class: 'gd-acts' }, ...s.acts.map(([label, fn]) => el('button', { class: 'btn', type: 'button', onclick: fn }, label))) : null);
  }
  function jump(id) {
    const t = G.main.querySelector(`[data-ch="${id}"]`);
    if (t) { G.main.scrollTop = t.offsetTop - G.main.offsetTop - 4; spy(id); }
  }
  function spy(force) {
    const chs = [...G.main.querySelectorAll('.gd-ch')], top = G.main.scrollTop + G.main.offsetTop + 40;
    let cur = force || (chs[0] && chs[0].dataset.ch);
    if (!force) for (const c of chs) if (c.offsetTop <= top) cur = c.dataset.ch;
    G.toc.querySelectorAll('.gd-tab').forEach((b) => { const on = b.dataset.ch === cur; b.setAttribute('aria-current', on ? 'true' : 'false'); if (on && NA.mobile && NA.mobile.mobile) b.scrollIntoView({ block: 'nearest', inline: 'nearest' }); });
  }
  G.show = function (chapter, query) {
    if (!G.dlg) build();
    if (NA.chrome && NA.chrome.toggleHelp) NA.chrome.toggleHelp(false);
    G.opener = document.activeElement;
    G.search.value = query || '';
    render();
    G.dlg.hidden = false; G.open = true;
    if (chapter) jump(chapter); else G.main.scrollTop = 0;
    // the search box on a computer; on a phone the keyboard would cover the guide, so the close button
    (touch() ? G.closeBtn : G.search).focus();
  };
  G.close = function () {
    if (!G.dlg || G.dlg.hidden) return;
    G.dlg.hidden = true; G.open = false;
    if (G.opener && G.opener.focus && document.contains(G.opener)) G.opener.focus();
  };
  G.toggle = function () { if (G.open) G.close(); else G.show(); };
  G.chapters = CHAPTERS;
  NA.guide = G;
})(window.NA);
