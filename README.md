# Neural Anatomy

An interactive lab for looking inside small neural networks. Pick a pre-trained model, then switch on lenses to see its weights, activations, gradients and hidden representations, and step through backpropagation one phase at a time.

**Live site:** https://ferns4.github.io/neural-anatomy/

## What you can do

- **Model zoo.** Nine networks that load already trained:
  - Foundations: Single Neuron, XOR, ReLU Polygon
  - Representation: Deep Spiral, Three Classes
  - Function fitting: Curve Fitting
  - Perception: Digit Recognizer (draw your own digits)
  - Failure modes: Vanishing Gradients, Overfitting
- **Lenses on the network diagram.** Colour connections by weight, signal or gradient, and neurons by activation, pre-activation or error signal. Overlays add neuron maps, values, biases, animated flow and neuron health.
- **Inspector.** Notes and one-click experiments for each model, automatic diagnostics, per-neuron arithmetic with editable weights and ablation, layer statistics, training curves, gradient flow by layer and a confusion matrix.
- **Backprop walkthrough.** One training example through a full step of gradient descent, with the real numbers.
- **Hidden space.** PCA projections of each layer, with an animated, warped input grid.
- **History.** Scrub back through training to see the network at any earlier step.

You can link straight to a model with its id, for example `#digits`, `#xor` or `#vanishing`.

## How it works

Plain HTML, CSS and JavaScript: no build step and no dependencies (fonts come from Google Fonts). Every model trains in your browser.

| File | Role |
|---|---|
| `js/engine.js` | Multilayer perceptron: forward and backward passes, SGD, momentum and Adam, PCA, contours |
| `js/data.js` | Seeded dataset generators, including synthetic handwritten digits drawn from stroke templates |
| `js/presets.js` | The model zoo and its notes |
| `js/app.js` | State, training loop, history snapshots and analysis caches |
| `js/network-view.js` | The network diagram and its lenses |
| `js/data-view.js`, `js/hidden-space.js` | Output map, curve fit, drawing pad and hidden-space projections |
| `js/inspector.js`, `js/charts.js`, `js/insights.js`, `js/walkthrough.js` | Inspector tabs, charts, diagnostics and the backprop walkthrough |
| `js/chrome.js` | Header, controls, model zoo, timeline, keyboard shortcuts and start-up |

To run it locally, serve this folder with any static file server (for example `python -m http.server`) and open http://localhost:8000.

---

Built with [Claude Code](https://claude.com/claude-code).
