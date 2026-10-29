/* glossary.js — short definitions of the words the site uses. In the notes, experiments, insights, challenges and
   the backprop walkthrough, the first mention of a term in a panel becomes a quiet link (at most three in one
   passage); clicking it shows the definition beside it. The help dialog opens the whole list, with a search. */
(function (NA) {
  'use strict';
  const { el, $ } = NA;

  // [term, words that refer to it, definition]. The more specialised terms come first: in a passage that mentions
  // many, those are the ones that get linked.
  const TERMS = [
    // learning from rewards
    ['Reward shaping', ['reward shaping', 'closer bonus', 'bonus for getting closer'], 'Extra rewards that point the way, such as a little for every step closer to a goal. Done as a difference of “how close”, it cannot change which route is best, but it gives a learner something to go on long before it first reaches the goal.'],
    ['PPO', ['ppo', 'proximal policy optimisation', 'proximal policy optimization'], 'Proximal policy optimisation (2017): an actor-critic that takes several small steps on each batch of experience, each clipped so the probability of an action moves at most about 20% from the policy that played. Steady and simple, it is the usual choice for robots and for tuning chatbots.'],
    ['Actor-critic', ['actor-critic', 'a2c', 'critic'], 'A policy (the actor) and an estimate of how good each state is (the critic), learned together. Each action is judged by its advantage: what it got, plus the value of where it ended up, minus what the critic expected.'],
    ['REINFORCE', ['reinforce', 'policy gradient', 'policy gradients'], 'The simplest policy-gradient method (Williams, 1992): after some episodes, raise the probability of each action taken in proportion to how much better than usual the return after it was.'],
    ['DQN', ['dqn', 'deep q-network', 'q-network'], 'A neural network that outputs a Q-value for each action, trained toward reward + γ × the best Q-value of the next state, with a replay memory of past moves and a target network. It learned Atari games from the screen (2013–15).'],
    ['Q-value', ['q-value', 'q-values', 'q-learning', 'q-table'], 'How much reward an agent expects in the end if it takes a given action in a given state and plays its best after that. Q-learning (Watkins, 1989) learns these from rewards alone.'],
    ['Discount', ['discount', 'discount factor', 'γ'], 'How much a reward one step later is worth now (γ, between 0 and 1). With γ = 0.9 a reward ten steps away counts about a third; with 0 only the next reward counts.'],
    ['Exploration', ['exploration', 'exploring', 'ε-greedy'], 'Trying actions that do not look best yet, to find out whether they pay more. ε-greedy explores by playing a random move a share ε of the time.'],
    ['Genetic algorithm', ['genetic algorithm', 'neuroevolution', 'evolution strategies', 'elitism', 'crossover', 'mutation', 'mutations'], 'Search by breeding: a population of candidates (here, networks’ weights) is scored, the fittest are kept or become parents, and children mix their parents and mutate a little. It needs only a score, no gradient; evolving neural networks this way is neuroevolution.'],
    ['Self-play', ['self-play'], 'Learning a game by playing both sides against itself, so the opponent always matches its own level.'],
    ['Return', ['return', 'returns'], 'In reinforcement learning, all the rewards of an episode from a moment on, later ones discounted: what an agent tries to make as large as it can.'],
    ['Vanishing gradients', ['vanishing gradients', 'vanishing gradient'], 'Gradients that shrink at every layer on the way back, so the early layers barely learn. Sigmoid layers are prone to it: their slope is at most 0.25.'],
    ['Dead ReLU', ['dead relu', 'dead relus', 'dead neuron', 'dead neurons'], 'A ReLU neuron whose sum is negative for every input: it always outputs 0, gets no gradient, and so stops learning.'],
    ['Saturation', ['saturated', 'saturate', 'saturation', 'flat tail'], 'A squashing neuron (sigmoid or tanh) pushed far out onto its flat tail, where its slope, and with it the gradient, is nearly 0.'],
    ['Overfitting', ['overfitting', 'overfit', 'memorizing', 'memorising'], 'Learning the training examples, noise and all, instead of the pattern behind them: training loss keeps falling while validation loss rises.'],
    ['Early stopping', ['early stopping'], 'Keeping the weights from the step where validation loss was lowest, before the model began to overfit.'],
    ['Regularization', ['regularization', 'regularisation'], 'Anything that discourages overfitting, such as L2 (keeping weights small) or dropout.'],
    ['L2 regularization', ['l2', 'weight decay'], 'Adds the squared size of the weights to the loss, so training keeps every weight small unless it earns its keep. Also called weight decay.'],
    ['Dropout', ['dropout'], 'While training, switching off a random share of the neurons for each example and scaling up the rest, so no neuron can rely on any other. It is off when the model is used.'],
    ['Backpropagation', ['backpropagation', 'backpropagate', 'backprop'], 'The chain rule applied from the output back towards the input: one backward pass works out the gradient of the loss for every weight.'],
    ['Gradient descent', ['gradient descent'], 'Training by repeatedly nudging every weight a little against its gradient, downhill on the loss.'],
    ['Learning rate', ['learning rate', 'learning rates'], 'How big each gradient-descent step is. Too small and training crawls; too big and the loss jumps around or blows up.'],
    ['Cross-entropy', ['cross-entropy', 'cross entropy'], 'A loss for classifying: minus the log of the probability the model gave the right answer. Confident mistakes cost the most.'],
    ['Squared error', ['squared error', 'mse'], 'A loss: the square of the difference between the prediction and the target, often halved. Big misses cost far more than small ones.'],
    ['Softmax', ['softmax'], 'Turns a list of scores into probabilities that add up to 1, giving the largest scores the largest shares.'],
    ['Logit', ['logit', 'logits'], 'A raw score, before softmax or a sigmoid turns it into a probability.'],
    ['Optimizer', ['optimizer', 'optimizers', 'optimiser'], 'The rule that turns gradients into weight changes: plain SGD, momentum or Adam.'],
    ['Adam', ['adam', 'adamw'], 'An optimizer that scales each weight’s step by the running size of that weight’s own gradients, so every weight moves at a sensible pace. AdamW also shrinks the weights a little each step.'],
    ['Momentum', ['momentum'], 'An optimizer that keeps a running velocity of past steps: steady directions speed up and zig-zags cancel out.'],
    ['SGD', ['sgd', 'stochastic gradient descent'], 'Stochastic gradient descent: each step follows the gradient of a small random batch of examples rather than all of them.'],
    ['ReLU', ['relu', 'relus', 'leaky relu'], 'Rectified linear unit, max(0, z): passes positive sums unchanged and turns negative ones into 0. Leaky ReLU keeps a small slope for negative sums.'],
    ['Sigmoid', ['sigmoid', 'sigmoids'], 'σ(z) = 1 / (1 + e⁻ᶻ): squashes any number into the range 0 to 1, so a yes-or-no output reads as a probability.'],
    ['Tanh', ['tanh'], 'The hyperbolic tangent: an S-shaped curve that squashes any number into −1 to 1, centred on 0.'],
    ['Activation function', ['activation function', 'activation functions', 'activation'], 'The function a neuron applies to its weighted sum, such as ReLU or tanh. Without one, any stack of layers could only compute a straight line.'],
    ['Decision boundary', ['decision boundary', 'boundary'], 'Where the model’s answer switches from one class to another.'],
    ['Validation set', ['validation', 'validation set', 'validation points'], 'Examples held back from training and scored as it goes, to see whether the model works on data it has not learned from.'],
    ['Test set', ['held-out test', 'test set', 'held out'], 'Examples kept aside until the very end for one honest final score, after every choice has been made.'],
    ['Epoch', ['epoch', 'epochs'], 'One pass through all the training examples.'],
    ['Batch', ['batch', 'batches', 'mini-batch'], 'The handful of training examples one step learns from; their gradients are averaged.'],
    ['Initialization', ['initialization', 'initialisation', 're-initialize'], 'The random starting weights. Their scale matters: too small and signals fade away, too large and neurons saturate.'],
    ['R²', ['r²', 'r2'], 'How much of the variation in the targets the model explains: 1 is perfect, 0 is no better than always guessing the average.'],
    ['Accuracy', ['accuracy'], 'The share of examples the model gets right.'],
    ['Gradient', ['gradient', 'gradients'], 'For each weight, how fast the loss changes when that weight changes. It points uphill, so training steps the opposite way.'],
    ['Loss', ['loss', 'losses'], 'One number for how wrong the model is on the training examples. Training changes the weights to make it smaller.'],
    ['Attention', ['attention', 'self-attention'], 'A layer in which each position looks at the others and takes a weighted mix of what it finds; the weights come from how well its query matches their keys.'],
    ['Transformer', ['transformer', 'transformers'], 'A stack of blocks of attention and small dense layers with residual connections: the design behind modern language models.'],
    ['Embedding', ['embedding', 'embeddings'], 'A learned vector of numbers that stands for a token.'],
    ['Token', ['token', 'tokens'], 'One unit of text, such as a word or a punctuation mark, as the model sees it: a whole number in a vocabulary.'],
    ['Convolution', ['convolution', 'convolutions'], 'A small filter slid across a picture, computing the same weighted sum at every position, so a pattern is found wherever it appears.'],
    ['PCA', ['pca', 'principal component', 'principal components'], 'Principal component analysis: the directions in which data varies most, used to show many dimensions in two.'],
    ['Loss landscape', ['loss landscape'], 'The loss as a surface over the weights. Training walks downhill on it.'],
    ['Hidden layer', ['hidden layer', 'hidden layers'], 'A layer between the input and the output. Its neurons build intermediate features that the next layer combines.'],
    ['Bias', ['bias', 'biases'], 'A number a neuron adds to its weighted sum before its activation function; it shifts where the neuron switches on.'],
    ['Weight', ['weight', 'weights'], 'A number on a connection: whatever comes along it is multiplied by the weight. Training adjusts the weights.'],
    ['Parameter', ['parameter', 'parameters'], 'A number the model learns: its weights and biases.'],
    ['Neuron', ['neuron', 'neurons'], 'A unit that multiplies each input by a weight, adds them up with a bias, and passes the sum through its activation function: one number out.'],
    ['Feature', ['feature', 'features'], 'A number describing the input that the model reads, such as x₁ or x₁²; inside a network, what a neuron has learned to detect.'],
  ].map(([term, words, def], rank) => ({ term, words, def, rank }));
  const esc = (s) => s.replace(/[.*+?^${}()|[\]\\]/g, '\\$&');
  // one pattern per term: any of its words, whole, in any case
  for (const t of TERMS) t.re = new RegExp(`(^|[^\\p{L}\\p{N}])(${t.words.slice().sort((a, b) => b.length - a.length).map(esc).join('|')})(?![\\p{L}\\p{N}])`, 'iu');

  // passages whose words get links: written explanations, not controls, labels or numbers
  const PASSAGES = '.mc-about, .look-list > li, .try-list > li > p, .diag-body, .wk-body > p, .help-cols p, .bd-ch-brief, .bd-ch-item-brief, .predict-q';
  const PER_PASSAGE = 3;
  function linkIn(node, used) {
    node.dataset.gl = '1';
    const texts = [];
    const walk = document.createTreeWalker(node, NodeFilter.SHOW_TEXT, { acceptNode: (n) => (n.parentElement.closest('button, a, input, textarea, code, kbd, .gl-term, .predict-tag') ? NodeFilter.FILTER_REJECT : NodeFilter.FILTER_ACCEPT) });
    for (let n = walk.nextNode(); n; n = walk.nextNode()) texts.push(n);
    let linked = 0;
    for (const t of TERMS) {
      if (linked >= PER_PASSAGE) break;
      for (let k = 0; k < texts.length; k++) {
        const n = texts[k], m = t.re.exec(n.data);
        if (!m || used.has(t.term)) continue;
        const start = m.index + m[1].length, word = m[2];
        const after = n.splitText(start), rest = after.splitText(word.length);
        const b = el('button', { class: 'gl-term', type: 'button', 'aria-haspopup': 'dialog', title: `What is “${t.term.toLowerCase()}”?` }, word);
        b.addEventListener('click', (e) => { e.stopPropagation(); G.define(t, b); });
        after.replaceWith(b);
        texts.splice(k, 1, n, rest); // the text before and after the link can still hold other terms
        used.add(t.term); linked++;
        break;
      }
    }
  }
  let pending = null;
  // a term is linked once in each batch of new passages (a panel drawn at once), so common words are not linked over and over
  function scan() { pending = null; const used = new Set(); for (const n of document.querySelectorAll(PASSAGES)) if (!n.dataset.gl) linkIn(n, used); }

  const G = { TERMS };
  // the definition beside a term
  G.define = function (t, anchor) {
    G.close();
    const pop = el('div', { class: 'gl-pop', role: 'dialog', 'aria-label': t.term },
      el('div', { class: 'gl-pop-term', text: t.term }), el('p', { class: 'gl-pop-def', text: t.def }),
      el('button', { class: 'link-btn', type: 'button', onclick: () => { G.close(); G.open(t.term); } }, 'All terms'));
    document.body.append(pop);
    const r = anchor.getBoundingClientRect(), w = pop.offsetWidth, h = pop.offsetHeight, vw = window.innerWidth, vh = window.innerHeight;
    pop.style.left = Math.round(Math.min(Math.max(8, r.left), vw - w - 8)) + 'px';
    pop.style.top = Math.round(r.bottom + 6 + h > vh ? Math.max(8, r.top - h - 6) : r.bottom + 6) + 'px';
    G.pop = pop; G.anchor = anchor;
    pop.querySelector('.link-btn').focus();
  };
  G.close = function () { if (G.pop) { G.pop.remove(); G.pop = null; } };
  // the whole list, searchable
  G.open = function (focusTerm) {
    if (NA.chrome && NA.chrome.toggleHelp) NA.chrome.toggleHelp(false);
    if (!G.dlg) {
      G.search = el('input', { class: 'gl-search', type: 'search', placeholder: 'Search the terms', 'aria-label': 'Search the terms', spellcheck: 'false' });
      G.list = el('dl', { class: 'gl-list' });
      G.dlg = el('div', { class: 'modal', role: 'dialog', 'aria-modal': 'true', 'aria-labelledby': 'glTitle', hidden: '' },
        el('div', { class: 'sheet gl-sheet' },
          el('div', { class: 'sheet-head' }, el('div', null, el('h2', { class: 'sheet-title', id: 'glTitle', text: 'Glossary' }), el('p', { class: 'sheet-sub', text: 'The words this site uses, in a sentence or two each.' })),
            el('button', { class: 'icon-btn', type: 'button', 'data-close': '', 'aria-label': 'Close' }, '×')),
          G.search, G.list));
      G.dlg.addEventListener('click', (e) => { if (e.target === G.dlg || e.target.closest('[data-close]')) G.dlg.hidden = true; });
      G.dlg.addEventListener('keydown', (e) => { if (e.key === 'Escape') { e.stopPropagation(); G.dlg.hidden = true; } });
      G.search.addEventListener('input', render);
      document.body.append(G.dlg);
    }
    G.search.value = '';
    render();
    G.dlg.hidden = false;
    const f = focusTerm && G.list.querySelector(`[data-term="${CSS.escape(focusTerm)}"]`);
    if (f) { f.scrollIntoView({ block: 'center' }); f.classList.add('is-focus'); }
    G.search.focus();
  };
  function render() {
    const q = G.search.value.trim().toLowerCase();
    const items = TERMS.slice().sort((a, b) => a.term.localeCompare(b.term)).filter((t) => !q || t.term.toLowerCase().includes(q) || t.def.toLowerCase().includes(q) || t.words.some((w) => w.includes(q)));
    NA.clear(G.list).append(...items.flatMap((t) => [el('dt', { text: t.term, dataset: { term: t.term } }), el('dd', { text: t.def })]));
    if (!items.length) G.list.append(el('dd', { class: 'sec-note', text: 'No term matches.' }));
  }

  if (typeof document !== 'undefined') {
    const go = () => {
      new MutationObserver(() => { if (!pending) pending = setTimeout(scan, 160); }).observe(document.body, { childList: true, subtree: true });
      scan();
      document.addEventListener('click', (e) => { if (G.pop && !G.pop.contains(e.target)) G.close(); });
      document.addEventListener('keydown', (e) => { if (e.key === 'Escape' && G.pop) { const a = G.anchor; G.close(); if (a) a.focus(); } });
      window.addEventListener('scroll', () => G.close(), true);
    };
    if (document.readyState === 'loading') document.addEventListener('DOMContentLoaded', go); else go();
  }
  G.terms = TERMS; // (the Guide lists them all)
  NA.glossary = G;
})(window.NA);
