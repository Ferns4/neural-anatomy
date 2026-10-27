# Neural Anatomy

An interactive lab for looking inside small neural networks. Pick a pre-trained model, then switch on lenses to see its weights, activations, gradients and hidden representations, and step through backpropagation one phase at a time.

**Live site:** https://ferns4.github.io/neural-anatomy/

## What you can do

- **Model zoo.** Twelve models that load already trained:
  - Foundations: Single Neuron, XOR, ReLU Polygon
  - Representation: Deep Spiral, Three Classes
  - Function fitting: Curve Fitting
  - Perception: Digit Recognizer (draw your own digits)
  - Language models: Tiny Transformer
  - Physics-informed: Physics-Informed Network
  - Generative models: Diffusion Model
  - Failure modes: Vanishing Gradients, Overfitting
- **Build your own network.** Add and remove hidden layers, resize each one, and choose the activation function for a whole layer or for any single neuron: ReLU, Leaky ReLU, ELU, GELU, Swish, Softplus, Tanh, Sigmoid, Sine, Gaussian or Linear. Each one is shown with its formula, its slope and the value it gives for the current input. Pick the loss function too: cross-entropy, squared error or hinge for classifiers; squared, absolute or Huber error for curve fitting.
- **Validation and a held-out test.** Every dataset is split into training, validation and test sets. Validation is scored live as the network trains; the test set stays untouched until you press **Run test**, which reports accuracy, loss, a confusion matrix and the worst mistakes, and says how far to trust the validation numbers.
- **Lenses on the network diagram.** Colour connections by weight, signal or gradient, and neurons by activation, pre-activation or error signal. Overlays add neuron maps, values, biases, animated flow, activation-function labels and neuron health. Pinch, double-tap or Ctrl + scroll to zoom into any part of the diagram, the output map or the hidden space.
- **Inspector.** Notes and one-click experiments for each model, automatic diagnostics, per-neuron arithmetic with editable weights and ablation, layer statistics, training curves, gradient flow by layer and a confusion matrix.
- **Backprop walkthrough.** One training example through a full step of gradient descent, with the real numbers.
- **Hidden space.** PCA projections of each layer, with an animated, warped input grid.
- **History.** Scrub back through training to see the network at any earlier step.

You can link straight to a model with its id, for example `#digits`, `#xor`, `#transformer`, `#pinn` or `#diffusion`.

## Beyond the MLP

Three models open in labs of their own:

- **Tiny Transformer** (`#transformer`). A GPT-style language model (token and position embeddings, two blocks of 4-head causal self-attention and MLPs, 28k parameters) trained on a tiny made-up language whose stories can only be predicted by looking back: "alice likes tea . bob likes milk . alice drinks" → "tea". Edit the prompt or let the model write; see every token's residual stream with attention arcs and the logit lens; knock out heads to find the lookup circuit; explore the embedding space; watch lookups jump from chance to 100% when the circuit forms during training.
- **Physics-Informed Network** (`#pinn`). A plain network and a PINN learn a damped oscillator from ten early measurements. The PINN also minimises the residual of u″ + 2ζω·u′ + ω²·u = 0, using exact derivatives of the network itself, and keeps following the true motion where the plain network has no data. Change the physics, the data or the loss terms and both retrain live.
- **Diffusion Model** (`#diffusion`). A denoising diffusion model on 2-D points: watch noise turn into a pinwheel step by step, dissolve the data with the forward process, and drop points into the learned denoising field. DDPM and DDIM sampling, 10 to 100 steps, three pretrained datasets.

## On your phone

On a phone the site switches to its own layout: one screen at a time (Network, Output, Inspect) behind a bottom tab bar, training settings in a bottom sheet, and a card over the diagram when you tap a neuron (with its activation function and formula, one tap to change it) or step through backpropagation. Menus and dialogs open as bottom sheets, and each picture zooms on its own with a pinch.

It also installs as an app and then works offline:

- **iPhone or iPad:** open the site in Safari, tap Share, then **Add to Home Screen**.
- **Android:** open the site in Chrome and tap **Install app** (or the install button in the site's top bar).
- **Computer:** use the install button in the browser's address bar.

## How it works

Plain HTML, CSS and JavaScript: no build step and no dependencies (fonts come from Google Fonts). Every model trains in your browser.

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
| `js/*-weights.js` | Pretrained weights (float16), trained offline with exactly the engine code above |
| `js/mobile.js` | The phone layout: screens, tab bar, settings sheet and peek card |
| `js/chrome.js` | Header, controls, model zoo, timeline, keyboard shortcuts and start-up |
| `pwa.js`, `sw.js`, `manifest.webmanifest` | Install as an app and offline support (the service worker is generated by the build) |

To run it locally, serve this folder with any static file server (for example `python -m http.server`) and open http://localhost:8000.

---

Built with [Claude Code](https://claude.com/claude-code).
