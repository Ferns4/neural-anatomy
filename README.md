# Neural Anatomy

An interactive lab for looking inside small neural networks. Pick a pre-trained model, then switch on lenses to see its weights, activations, gradients and hidden representations, and step through backpropagation one phase at a time.

**Live site:** https://ferns4.github.io/neural-anatomy/

## What you can do

- **Build a model from scratch.** The **Build your own** button opens a sandbox: pick a problem or make your own (place points of up to four classes, type any formula in x, or write sentences), then drag parts between the input and the output: dense layers, single neurons, activations, dropout, layer norm, convolutions and max pooling for pictures, self-attention and whole transformer blocks for sequences. Every block shows the shape of the data passing through it and its parameter count; a block that does not fit says why and offers the fix. Choose the loss, optimizer (SGD, momentum, Adam, AdamW), learning rate and schedule, batch size, weight decay, gradient clipping, initialisation and seed, and train it in the browser. Each run is listed with what changed since the one before and its validation score, so you can compare them; the Architecture view opens the model you built like the others (a block, a neuron's weighted sum, down to the transistors), and the whole setup can be copied as a PyTorch script.
- **Model zoo.** Fourteen models that load already trained:
  - Foundations: Single Neuron, XOR, ReLU Polygon
  - Representation: Deep Spiral, Three Classes
  - Function fitting: Curve Fitting
  - Perception: Digit Recognizer (draw your own digits), YOLO26 Object Detector
  - Language models: Tiny Transformer
  - Physics-informed: Physics-Informed Network
  - Generative models: Diffusion Model
  - Signal processing: Radar Pulse Classifier
  - Failure modes: Vanishing Gradients, Overfitting
- **Build your own network.** Add and remove hidden layers, resize each one, and choose the activation function for a whole layer or for any single neuron: ReLU, Leaky ReLU, ELU, GELU, Swish, Softplus, Tanh, Sigmoid, Sine, Gaussian or Linear. Each one is shown with its formula, its slope and the value it gives for the current input. Pick the loss function too: cross-entropy, squared error or hinge for classifiers; squared, absolute or Huber error for curve fitting.
- **Validation and a held-out test.** Every dataset is split into training, validation and test sets. Validation is scored live as the network trains; the test set stays untouched until you press **Run test**, which reports accuracy, loss, a confusion matrix and the worst mistakes, and says how far to trust the validation numbers.
- **Lenses on the network diagram.** Colour connections by weight, signal or gradient, and neurons by activation, pre-activation or error signal. Overlays add neuron maps, values, biases, animated flow, activation-function labels and neuron health. Pinch, double-tap or Ctrl + scroll to zoom into any part of the diagram, the output map or the hidden space.
- **Inspector.** Notes and one-click experiments for each model, automatic diagnostics, per-neuron arithmetic with editable weights and ablation, layer statistics, training curves, gradient flow by layer and a confusion matrix.
- **Backprop walkthrough.** One training example through a full step of gradient descent, with the real numbers.
- **Hidden space.** PCA projections of each layer, with an animated, warped input grid.
- **History.** Scrub back through training to see the network at any earlier step.
- **All the way down.** Zoom into any model's architecture: a block, a layer, a single neuron's weighted sum, its PyTorch code, the float32 multiply bit by bit, the adders and logic gates that do it, their CMOS transistors, and the electrons drifting through a transistor's channel. Every level uses the real numbers of the model on screen, with an explanation and runnable Python.

You can link straight to a model with its id, for example `#digits`, `#xor`, `#transformer`, `#pinn`, `#diffusion`, `#yolo26` or `#radar`.

## Beyond the MLP

Five models open in labs of their own:

- **Tiny Transformer** (`#transformer`). A GPT-style language model (token and position embeddings, two blocks of 4-head causal self-attention and MLPs, 28k parameters) trained on a tiny made-up language whose stories can only be predicted by looking back: "alice likes tea . bob likes milk . alice drinks" → "tea". Edit the prompt or let the model write; see every token's residual stream with attention arcs and the logit lens; knock out heads to find the lookup circuit; explore the embedding space; watch lookups jump from chance to 100% when the circuit forms during training.
- **Physics-Informed Network** (`#pinn`). A plain network and a PINN learn a damped oscillator from ten early measurements. The PINN also minimises the residual of u″ + 2ζω·u′ + ω²·u = 0, using exact derivatives of the network itself, and keeps following the true motion where the plain network has no data. Change the physics, the data or the loss terms and both retrain live.
- **Diffusion Model** (`#diffusion`). A denoising diffusion model on 2-D points: watch noise turn into a pinwheel step by step, dissolve the data with the forward process, and drop points into the learned denoising field. DDPM and DDIM sampling, 10 to 100 steps, three pretrained datasets.
- **YOLO26 Object Detector** (`#yolo26`). The layer list of Ultralytics' YOLO26 (Conv, C3k2, SPPF and C2PSA blocks, an FPN-PAN neck, detection at three scales, no DFL) at half the width of YOLO26n, trained on synthetic scenes of four kinds of shape with this repository's versions of YOLO26's ProgLoss, STAL and MuSGD. Drag, resize, add and remove shapes and the boxes follow, computed in your browser. Compare the one-to-many head, which needs non-maximum suppression, with the one-to-one head, which gives one box per object with no NMS (mAP@0.5 96.0% against 93.4% on 500 validation scenes).
- **Radar Pulse Classifier** (`#radar`). A 1-D CNN that names a radar waveform (plain pulse, LFM chirp, Barker-13, Frank code, Costas FSK, or noise alone) from 256 raw I/Q samples, trained on about 3 million synthetic intercepts at −12 to +20 dB SNR. See each signal as I/Q traces, an I/Q plane, a spectrogram and the network's class activation map; measure accuracy by SNR live (99% at +2 dB and above, 77% at −4 dB).

## On your phone

On a phone the site switches to its own layout: one screen at a time (Network, Output, Inspect) behind a bottom tab bar, training settings in a bottom sheet, and a card over the diagram when you tap a neuron (with its activation function and formula, one tap to change it) or step through backpropagation. Menus and dialogs open as bottom sheets, and each picture zooms on its own with a pinch.

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
| `js/builder-tasks.js`, `js/builder-viz.js`, `js/builder-py.js`, `js/arch-builder.js`, `js/lab-builder.js` | The builder's problems and formula parser, its results pictures, the PyTorch script it writes, its zoomable architecture view, and the drag-and-drop lab |
| `js/archview.js`, `js/arch-*.js`, `js/net-deep.js` | The zoomable architecture views; `arch-deep.js` holds the shared levels from a neuron down to the electrons |
| `js/*-weights.js` | Pretrained weights (float16): trained offline with exactly the engine code above, or for YOLO26 and the radar in PyTorch with BatchNorm folded in |
| `js/mobile.js` | The phone layout: screens, tab bar, settings sheet and peek card |
| `js/chrome.js` | Header, controls, model zoo, timeline, keyboard shortcuts and start-up |
| `pwa.js`, `sw.js`, `manifest.webmanifest` | Install as an app and offline support (the service worker is generated by the build) |

To run it locally, serve this folder with any static file server (for example `python -m http.server`) and open http://localhost:8000.

---

Built with [Claude Code](https://claude.com/claude-code).
