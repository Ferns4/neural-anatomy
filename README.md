# Neural Anatomy

An interactive lab for looking inside small neural networks. Pick a pre-trained model, then switch on lenses to see its weights, activations, gradients and hidden representations, and step through backpropagation one phase at a time.

**Live site:** https://ferns4.github.io/neural-anatomy/

## New in version 31

- **One home for how it learns:** every setting of the method (grow and prune too) is in the model builder's **Training tab**; the Output block says what comes out and what its outputs mean, and opens the Training tab at the right place.
- **Find a setting:** a search box at the top of the right panel (press **/**). Type a few letters of a setting, a choice or a feature ("learning rate", "explore", "bigger network", "record") and pick one: its tab opens, and the setting is scrolled to and lit up.
- **Watching it play:** a bar under the picture pauses and plays (**K**), steps a move back or on, slides through the episode, and plays from a quarter of the speed to four times; New episode starts another. **Record a video** saves an episode as a video file. **Race…** plays two runs side by side from the same start, or you against a run, and counts the wins.
- **Teach it yourself:**
  - **Copy you** (behaviour cloning): Jump in and play, then train; it learns your move in every moment you played, as a classifier learns its classes. What is happening says when it copies your moves but not your results.
  - **Which did better?** (learning from preferences, the idea behind RLHF): two tries from the same start side by side; you say which did better (← ↓ →), a reward model learns from your answers and PPO from the reward model. Or let the world's reward answer, to compare.
  - **Start from:** a run can begin from another run's network, such as a copy of your play handed to PPO.

  Both are parts of the palette too (Rewards), with their guides, and both have their Python: Copy you reads your moves from the file the Python tab saves (Download your moves).
- **Controls that are numbers:** the robot arm, the rocket, the walker and the race track can be set by numbers from −1 to 1 (each motor's push, the throttle, each muscle's length, the steering) by three new methods: **PPO with a Gaussian policy**, **SAC** (soft actor-critic) and **TD3**. SAC and TD3 bring two critics of their own and a replay memory. The bars beside the picture show each number and its spread; the Python writes all three in PyTorch.
- **My models:** keep a run's network under a name, in this browser, and open it again later (its setup back in the builder, its weights in a new run that starts where they were); download it as a file and load it in any browser. The latest run is kept by itself, so a reload loses nothing.
- **Training in the background:** in a tab you are not looking at, training goes on (a worker keeps the time), and the tab's title says how far it has got.
- **Worlds of your own:** an editor for the maze (walls, pits, the start and the goal), the race track (drag its points; it checks the bends and that it never runs into itself), the hurdles course (each hurdle and bar) and where the sheep and the dog start. It says what is wrong as you go and what the world pays; Use this puts it in the settings, so Undo, share links, the run list and the Python all carry it.
- **Looking closer at a learner:** Test it 100 times (how its episodes end, its returns and lengths), a **value map** over any two of its inputs (the value it expects, or the move it would make), and **where it went** while it learned (the maze, the race track, the herding field).
- **The knobs of PPO and DQN:** PPO's clip, passes over each batch, GAE λ and entropy bonus; DQN's replay memory, target refresh, double DQN, **dueling** heads and **n-step** returns; SAC's and TD3's random tries first.
- **Easy first, dreams and rivals:** a **curriculum** for the pole, the arm, the rocket and the hurdles; a learned **model of the world** whose dreams are drawn beside the world; and **predator and prey**, a chaser that learns too.
- **NEAT:** networks that grow their own structure (connections and neurons added by mutation, genes lined up by innovation number, species with shared fitness), with views of **the network it grew** and **its species**; and its Python.
- **Family tree of shapes** (grow and prune) and **population-based training** (4 or 8 learners, the worst copying the best with a nudged learning rate).
- **Six more kinds of problem**, each with its own pictures, settings, templates and Python:
  - **Squeeze and rebuild:** an autoencoder (with a map of its codes), a **VAE** that draws new digits, and cleaning up noisy pictures.
  - **Make new examples:** a **GAN** whose generator is your model, on eight blobs in a ring (watch for mode collapse), two moons or tiny digits; the discriminator's verdict behind the points and where it pushes each one; scored by precision and recall.
  - **Learn without labels:** contrastive learning on two views of each picture (SimCLR's NT-Xent), scored by a vote of the nearest codes against the raw pixels.
  - **Label every pixel:** segmentation of circles, squares and triangles, with a **tiny U-Net**, scored by the overlap of its shapes (IoU).
  - **Graphs:** friend groups found by **message passing**, against a model that only sees each node alone.
  - **Predict with uncertainty:** a middle and a spread with the **Gaussian loss**, its band of ±2 spreads and the share of points inside it.

  New parts for them: **Upsample**, **Sample** (a VAE's middle, its KL divergence added to the loss) and **Graph convolution**, with their guides.
- **More parts:** a **skip connection** from any earlier block (added or joined), a **Kolmogorov–Arnold layer**, a **Hopfield memory**; and **More than one right answer**, where a **mixture of bell curves** (a mixture density network) finds every branch.
- **The Analyse tab:** sweep a setting (batch size, weight decay, the optimizer, a layer's width), **luck** across 3 or 5 seeds, **calibration** (with temperature scaling), **which inputs matter** (permutation importance), **adversarial examples** (FGSM, with accuracy as ε grows) and **pruning and quantising** a copy of the network.
- **Guess first** (in the Runs tab): each new run asks what score you expect, and says at its end how close you were; and **paths** that lead through the challenges in order.
- Smaller fixes: the run list counts generations for evolution, the toast no longer covers the buttons in full screen on a phone, and the smallest labels are easier to read; What is happening no longer warns that the first layers hardly learn when the first block is a Flatten (it has no weights to learn); the loss chart turns to a straight scale when a loss goes below 0; and Find a setting finds the palette's parts too.

## New in version 30

- **Grow and prune, fully set:** on the Output block (and in the Training tab for problems with answers), every part of it can be chosen, with what each choice does written beside it:
  - start from the network you built or from **no hidden layer at all**, as NEAT does;
  - the changes it can make, each on or off: a neuron more or less, a new layer, a whole layer out at once, and a **new activation** (ReLU, tanh, sigmoid, leaky ReLU, ELU); whether it leans to grow or to prune, with the odds of each change worked out; and how many children change shape;
  - how a new layer starts (one neuron, a few, or a whole layer; passing values on or at random) and **where it goes** (anywhere, after the input, before the output); whether a new neuron starts silent or joins in at random; whether pruning takes the weakest neuron or one at random;
  - the largest network, the size penalty, and **protecting new shapes**: for 5 generations the best member of a new shape carries on unchanged, so mutations get time to tune it (a simple form of NEAT's speciation). Learning from examples, how many networks learn side by side and how often they are reshaped.

  The whole generation's view counts its shapes, the line under the results gives each layer's activation, the run list names every change, and the Python writes all of it (and reads edits of it back).
- **Full screen results:** a button at the top right of the results puts the picture over the whole screen, with Train beside it; Esc (or the button) puts it back.
- **Jump in:** takes the controls of the episode on show where it is, the network's move scores still beside your moves; **Let it play** hands them back and the network carries on from where you left it. In full screen a key of the world's jumps in at once, and on a touch screen the buttons to hold sit under the picture.
- **Editing the Python is easier to find:** a clear Edit button, a line above the script that says how it works, and a double-click (or double-tap) on any line opens the editor there with the number under the pointer selected.

## New in version 29

- **Eleven new worlds to learn from rewards**, and the model builder's problem list groups all 18 by level:
  - *Beginner:* levers, catch and the maze, and now **Keep a room warm** (a thermostat whose radiator is slow to warm up and slow to cool down, so it has to switch off early) and **Hurdles** (take off at the right moment; on the mixed course, duck under the bars).
  - *Intermediate:* balancing a pole, and now **Remember the lamp** (the lamp says which way to turn, then goes dark: a test of memory), **Mountain cart** (rock out of a valley; every step costs 1 until the flag), **Swing up a pendulum**, **Robot arm** (joints that keep turning: brake before the target), **Traffic lights** (each switch costs 4 steps of amber, and the rush hour moves) and **Walker** (bones, springs and two muscles: find a rhythm).
  - *Advanced:* the race track and the drone arena, and now the **Deceptive maze** (the reward for getting closer leads into a wall; novelty search, its own method, gets round).
  - *Expert:* landing a rocket, and now **Escape the chaser** (running straight away ends in a corner) and **Herd the sheep**.

  Each starts with the method that learns it best and has its own settings, inputs and reward table, and Jump in puts you at its controls. Its **What is happening** notes play the network's best moves again and say what they get stuck on, with the numbers (a room that overshoots, a lamp it cannot remember, a cart that never reaches the flag, a robot pressed against the wall under the goal, a runner caught in a corner) and the setting that helps.
- **Three more ways to learn:** novelty search (Lehman and Stanley), which breeds the networks that behave least like anything seen before and gets out of the deceptive maze; the cross-entropy method; and evolutionary RL (Khadka and Tumer), a genetic algorithm and a DQN sharing one replay memory.
- **Grow and prune, a neuron at a time:** a new hidden layer starts with a single neuron instead of arriving whole, later neurons widen it, and a layer goes only when its last neuron is pruned.
- **Inside a run:** the results switch between views. **It plays** shows its best moves live, with its outputs and, under the picture, the episode's return term by term (which part of the reward paid what). **What it looked at** nudges each input a little and shows how much that changes the move it picks. For evolution there are also **The whole generation** (every member playing at once, and how varied the population is from generation to generation), the **Family line** of the best (each ancestor with what changed and its score) and the **Styles of play** it has found. MAP-Elites' grid of styles has a key to its colours, and PPO shows how often its updates were clipped.
- **Ten more parts and eleven more problems** in the model builder. Parts: radial basis units and maxout among the layers; an attention pool and a state-space layer for sequences; separable convolutions, squeeze-and-excitation, coordinate channels and random shifts for pictures; global max pooling; and a mixture of experts. Problems: fast wiggles and a picture to fit, a random walk and a heartbeat to forecast, counting dots, which way the stripes run, colour and shape, inside or outside, balanced brackets, palindromes, and adding two numbers digit by digit.
- **More control over rewards and inputs, in all 18 worlds:**
  - A **Reward** tab beside the model, opened from the line under the results' title. Its table lists everything the world can pay for or charge, with any amount typed or picked: in the maze, going back to a square it has already been on, standing next to a pit, changing direction or running out of steps; for the pole, each degree it leans and each metre from the middle; and so on in every world.
  - A formula of your own over the world's numbers, now with comparisons, if(…), and, or and not. **Write the table as a formula** starts one from the amounts, and a step where a formula gives no number pays 0 and says so.
  - Extra inputs to switch on, on top of leaving some out and sensor noise: its last move, the time left, a random number that tells it nothing, and some of each world's own (the weather in an hour for the thermostat, where the ball will land in catch).
  - Each world's own settings: the track's laps, top speed, grip and sensor range; the pole's track and the lean it falls at; the catch's board and paddle; the arena's lives, goal and gun; and more.
  - Rewards normalised for actor-critic and PPO, and **Keep the best network**: a run that ends below its best score ends with the weights it had at its best.
  - The Python of every world follows the page step for step, its settings, rewards and extra inputs included.
- **Editable Python:** change the PyTorch script in the Python tab and press Apply, and the builder finds the settings that would write your lines (sizes and activations, training and evolution settings, the world and its rewards) and changes them, keeping the change only if the script it then writes has every line you changed. If it cannot follow a change, it says which lines and why and puts the script back as it was; Undo takes back one it applied, and Wide gives the script the whole window. The Python itself never runs in the page.
- **The Guide** has new sections on the reward worlds level by level, reward shaping, deceptive rewards and novelty search, memory, the new methods and growing a network a neuron at a time.

## What you can do

- **Build a model from scratch.** The **Build your own** button opens a sandbox: pick a problem or make your own (place points of up to four classes, type any formula in x, or write sentences), then drag parts between the input and the output: dense layers, single neurons, activations, dropout, layer norm, convolutions and max pooling for pictures, self-attention and whole transformer blocks for sequences. Every block shows the shape of the data passing through it and its parameter count; a block that does not fit says why and offers the fix. Choose the loss, optimizer (SGD, momentum, Adam, AdamW), learning rate and schedule, batch size, weight decay, gradient clipping, initialisation and seed, and train it in the browser. Each run is listed with what changed since the one before and its validation score, so you can compare them; the Architecture view opens the model you built like the others (a block, a neuron's weighted sum, down to the transistors), and the whole setup can be copied as a PyTorch script. A model can also learn from rewards instead of answers, in 18 worlds from beginner to expert (a row of levers, a maze, a room to keep warm, hurdles, a pole on a cart, a lamp to remember, a cart in a valley, a pendulum to swing up, a robot arm, traffic lights, a race track, a drone arena seen in the first person, a walker, a deceptive maze, a rocket to land, a chaser to escape and sheep to herd), by DQN, REINFORCE, actor-critic, PPO, a genetic algorithm, novelty search, evolution strategies, the cross-entropy method, MAP-Elites or evolutionary RL; with grow and prune, evolution changes the network itself a neuron at a time, and a whole generation can be watched playing at once; Try it yourself puts you at the controls of any world, with the network's choice beside each of your moves, and What is happening says what its best moves get stuck on. Challenges set goals to reach, from beginner to expert.
- **Model zoo.** Nineteen models, most of them already trained when they open:
  - Foundations: Single Neuron, XOR, ReLU Polygon
  - Representation: Deep Spiral, Three Classes
  - Function fitting: Curve Fitting
  - Perception: Digit Recognizer (draw your own digits), YOLO26 Object Detector
  - Language models: Tiny Transformer
  - Physics-informed: Physics-Informed Network
  - Generative models: Diffusion Model
  - Signal processing: Radar Pulse Classifier
  - Failure modes: Vanishing Gradients, Overfitting
  - Reinforcement learning: Maze Runner, Many Mazes, Three-line, Four-line, Evolution
- **Build your own network.** Add and remove hidden layers, resize each one, and choose the activation function for a whole layer or for any single neuron: ReLU, Leaky ReLU, ELU, GELU, Swish, Softplus, Tanh, Sigmoid, Sine, Gaussian or Linear. Each one is shown with its formula, its slope and the value it gives for the current input. Pick the loss function too: cross-entropy, squared error or hinge for classifiers; squared, absolute or Huber error for curve fitting.
- **Validation and a held-out test.** Every dataset is split into training, validation and test sets. Validation is scored live as the network trains; the test set stays untouched until you press **Run test**, which reports accuracy, loss, a confusion matrix and the worst mistakes, and says how far to trust the validation numbers.
- **Lenses on the network diagram.** Colour connections by weight, signal or gradient, and neurons by activation, pre-activation or error signal. Overlays add neuron maps, values, biases, animated flow, activation-function labels and neuron health. Pinch, double-tap or Ctrl + scroll to zoom into any part of the diagram, the output map or the hidden space.
- **Inspector.** Notes and one-click experiments for each model, automatic diagnostics, per-neuron arithmetic with editable weights and ablation, layer statistics, training curves, gradient flow by layer and a confusion matrix.
- **Backprop walkthrough.** One training example through a full step of gradient descent, with the real numbers.
- **Hidden space.** PCA projections of each layer, with an animated, warped input grid.
- **History.** Scrub back through training to see the network at any earlier step.
- **All the way down.** Zoom into any model's architecture: a block, a layer, a single neuron's weighted sum, its PyTorch code, the float32 multiply bit by bit, the adders and logic gates that do it, their CMOS transistors, and the electrons drifting through a transistor's channel. Every level uses the real numbers of the model on screen, with an explanation and runnable Python.

You can link straight to a model with its id, for example `#digits`, `#xor`, `#transformer`, `#pinn`, `#diffusion`, `#yolo26`, `#radar`, `#rl-maze`, `#rl-evolve` or `#builder`.

## Beyond the MLP

These open in labs of their own:

- **Tiny Transformer** (`#transformer`). A GPT-style language model (token and position embeddings, two blocks of 4-head causal self-attention and MLPs, 28k parameters) trained on a tiny made-up language whose stories can only be predicted by looking back: "alice likes tea . bob likes milk . alice drinks" → "tea". Edit the prompt or let the model write; see every token's residual stream with attention arcs and the logit lens; knock out heads to find the lookup circuit; explore the embedding space; watch lookups jump from chance to 100% when the circuit forms during training.
- **Physics-Informed Network** (`#pinn`). A plain network and a PINN learn a damped oscillator from ten early measurements. The PINN also minimises the residual of u″ + 2ζω·u′ + ω²·u = 0, using exact derivatives of the network itself, and keeps following the true motion where the plain network has no data. Change the physics, the data or the loss terms and both retrain live.
- **Diffusion Model** (`#diffusion`). A denoising diffusion model on 2-D points: watch noise turn into a pinwheel step by step, dissolve the data with the forward process, and drop points into the learned denoising field. DDPM and DDIM sampling, 10 to 100 steps, three pretrained datasets.
- **YOLO26 Object Detector** (`#yolo26`). The layer list of Ultralytics' YOLO26 (Conv, C3k2, SPPF and C2PSA blocks, an FPN-PAN neck, detection at three scales, no DFL) at half the width of YOLO26n, trained on synthetic scenes of four kinds of shape with this repository's versions of YOLO26's ProgLoss, STAL and MuSGD. Drag, resize, add and remove shapes and the boxes follow, computed in your browser. Compare the one-to-many head, which needs non-maximum suppression, with the one-to-one head, which gives one box per object with no NMS (mAP@0.5 96.0% against 93.4% on 500 validation scenes).
- **Radar Pulse Classifier** (`#radar`). A 1-D CNN that names a radar waveform (plain pulse, LFM chirp, Barker-13, Frank code, Costas FSK, or noise alone) from 256 raw I/Q samples, trained on about 3 million synthetic intercepts at −12 to +20 dB SNR. See each signal as I/Q traces, an I/Q plane, a spectrogram and the network's class activation map; measure accuracy by SNR live (99% at +2 dB and above, 77% at −4 dB).
- **Reinforcement learning** (`#rl-maze`, `#rl-mazes`, `#rl-three`, `#rl-four`, `#rl-evolve`). Agents that learn from rewards alone: a Q-table finds its way through a maze, one Q-network gets through mazes it never saw, a value table or network learns two board games of its own (Three-line and Four-line) by playing itself, and a genetic algorithm breeds a network that balances a pole without any gradients. Watch it, test it, play against it, and zoom into its architecture.

## On your phone

On a phone the site switches to its own layout: one screen at a time (Network, Output, Inspect) behind a tab bar (at the bottom, or down the side with the phone turned), training settings in a bottom sheet, and a card over the diagram when you tap a neuron (with its activation function and formula, one tap to change it) or step through backpropagation. Menus and dialogs open as bottom sheets, and each picture zooms on its own with a pinch.

It also installs as an app and then works offline:

- **iPhone or iPad:** open the site in Safari, tap Share, then **Add to Home Screen**.
- **Android:** open the site in Chrome and tap **Install app** (or the install button in the site's top bar).
- **Computer:** use the install button in the browser's address bar.

## How it works

Plain HTML, CSS and JavaScript: no build step and no dependencies (fonts come from Google Fonts). Every model runs in your browser, and all but two also train there (so does anything you build); the YOLO26 detector and the radar classifier were trained in PyTorch on a GPU, and their full training code is in each lab's Architecture view (top level, Python tab).

| File | Role |
|---|---|
| `js/engine.js` | Multilayer perceptron with a per-neuron activation function and a choice of loss: forward and backward passes, SGD, momentum and Adam, PCA, contours |
| `js/data.js` | Seeded dataset generators, including synthetic handwritten digits drawn from stroke templates |
| `js/presets.js` | The model zoo and its notes |
| `js/app.js` | State, training loop, history snapshots and analysis caches |
| `js/network-view.js` | The network diagram and its lenses |
| `js/data-view.js`, `js/hidden-space.js` | Output map, curve fit, drawing pad and hidden-space projections |
| `js/inspector.js`, `js/charts.js`, `js/insights.js`, `js/walkthrough.js` | Inspector tabs, charts, diagnostics and the backprop walkthrough |
| `js/ui.js` | Shared pieces: activation glyphs, option menus and sheets, zoom, toasts |
| `js/lab.js` | Hosts the labs: panels, settings, readouts and training loop, plus shared interface pieces |
| `js/transformer.js`, `js/lm-data.js`, `js/lab-transformer.js` | The tiny GPT (hand-written backpropagation), its language and tokenizer, and its lab |
| `js/pinn.js`, `js/lab-pinn.js` | The physics-informed network (value, u′ and u″ streams through every layer) and its lab |
| `js/diffusion.js`, `js/lab-diffusion.js` | The diffusion model (cosine schedule, DDPM and DDIM sampling) and its lab |
| `js/yolo.js`, `js/lab-yolo.js` | The YOLO26-style detector: scene drawing, a forward pass that runs in a Web Worker, both heads' decoding and NMS, and its lab |
| `js/radar.js`, `js/lab-radar.js` | Synthetic radar I/Q signals, the 1-D CNN, spectrogram and class activation map, and its lab |
| `js/builder-engine.js` | The model builder's engine: dense, attention, transformer, convolution, pooling, LayerNorm and dropout blocks with hand-written backpropagation, five losses and four optimizers, checked against finite differences by `tools/builder-gradcheck.js` |
| `js/builder-tasks.js`, `js/builder-viz.js`, `js/builder-depth.js`, `js/builder-insights.js`, `js/builder-py.js`, `js/builder-py-worlds.js`, `js/builder-py-worlds2.js`, `js/builder-py-edit.js`, `js/arch-builder.js`, `js/lab-builder.js` | The builder's problems and formula parser, its results pictures and what goes on inside a reward run, "What is happening" (a run diagnosed in plain words), the PyTorch script it writes (every world included) and the editing of it, its zoomable architecture view, and the drag-and-drop lab |
| `js/rl.js`, `js/rl-envs.js`, `js/rl-worlds.js`, `js/rl-worlds2.js`, `js/rl-worlds3.js`, `js/rl-evolve.js`, `js/builder-rl.js`, `js/builder-evo.js`, `js/lab-rl.js` | Reinforcement learning: the Q-learning, value-network and genetic-algorithm agents, the worlds (levers, maze, pole, catch; the race track, drone arena and rocket; the room, hurdles, lamp, mountain cart, pendulum and robot arm; the deceptive maze, chaser, traffic lights, walker and sheep), the builder's ways to learn from rewards, its evolution (a genetic algorithm with grow and prune, novelty search, evolution strategies, the cross-entropy method, MAP-Elites, evolutionary RL), and the RL lab |
| `js/guide.js`, `js/tour.js`, `js/glossary.js` | The Guide to everything on the site, the first-visit tour and the glossary |
| `js/archview.js`, `js/arch-*.js`, `js/net-deep.js` | The zoomable architecture views; `arch-deep.js` holds the shared levels from a neuron down to the electrons |
| `js/*-weights.js` | Pretrained weights (float16): trained offline with exactly the engine code above, or for YOLO26 and the radar in PyTorch with BatchNorm folded in |
| `js/mobile.js` | The phone layout: screens, tab bar, settings sheet and peek card |
| `js/chrome.js` | Header, controls, model zoo, timeline, keyboard shortcuts and start-up |
| `pwa.js`, `sw.js`, `manifest.webmanifest` | Install as an app and offline support (the service worker is generated by the build) |

To run it locally, serve this folder with any static file server (for example `python -m http.server`) and open http://localhost:8000.

---

Built with [Claude Code](https://claude.com/claude-code).
