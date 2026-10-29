/* tour.js — a short first-visit tour: a card beside each part of the page in turn, with that part outlined and the
   rest dimmed. One tour for the network models, one for the model builder and one for the other labs. Each runs
   by itself the first time its kind of page opens (remembered in this browser), and again from the help dialog.
   Steps whose part is not on screen (a phone shows one screen at a time) switch to its screen or are skipped. */
(function (NA) {
  'use strict';
  const { el, $ } = NA;
  const App = NA.app;

  const TOURS = {
    mlp: [
      { sel: '#modelPicker', title: 'Pick a model', text: 'Every model in the zoo opens already trained, with notes on what to look for and experiments to try. This one is a small network learning to tell two classes of points apart.' },
      { sel: '#netStage', screen: 'network', title: 'The network', text: `Columns are layers, squares are neurons and the curves are weighted connections; blue is positive, orange negative. ${NA.touchFirst && NA.touchFirst() ? 'Hold a finger on a neuron to see what it computes, or tap it to go inside, all the way down to the arithmetic.' : 'Point at a neuron to see what it computes, or zoom in to go all the way down to the arithmetic.'}` },
      { sel: '.lensbar', screen: 'network', title: 'Lenses', text: 'Lenses change what the colours mean: the weights, the signal for one input, or the gradient training follows. Overlays add maps, values and health checks.' },
      { sel: '#btnPlay', title: 'Train it', text: 'Train runs gradient descent, Step makes a single update so you can see what changes, and the settings button holds the learning rate, optimizer, regularization and dropout.' },
      { sel: '#dataPanel', screen: 'data', title: 'What it learned', text: 'The output over the whole input space, with the data on top. Hidden space shows how each layer rearranges the data; Loss landscape, the terrain training walked down.' },
      { sel: '#inspTabs', screen: 'insp', title: 'Explanations', text: 'Insights explains what is going on and suggests experiments, some asking you to predict the result first. Neuron and Layer show the numbers; Backprop works through one training step by hand.' },
      { sel: '.timeline', title: 'Rewind', text: 'Training is recorded as it goes: drag the history to see the network as it was at any step.' },
      { sel: '#buildBtn', title: 'Build your own', text: 'Put a model together from layers, pick a problem (or upload a table), train it and compare runs. Challenges set goals to reach.' },
    ],
    builder: [
      { sel: '#labLeft', screen: 'data', title: 'Problem and parts', text: 'Pick what the model learns, start from a template, then drag parts into the model. Challenges load a model that falls short of a goal for you to fix.' },
      { sel: '.bd-bench', screen: 'network', title: 'Your model', text: 'Blocks stacked from the input to the output, with the shape of the numbers between them. Click a block to change it, drag it to reorder; one outlined in red does not fit, and its button adds what is missing.' },
      { sel: '.bd-results', screen: 'network', title: 'Results', text: 'What the model does with the data, updated live while it trains, with the loss below.' },
      { sel: '.bd-toolbar .seg', screen: 'network', title: 'Views', text: 'Show the model, the results or both, or the Architecture: your model as a diagram you can zoom into, down to a single multiplication.' },
      { sel: '#labRight', screen: 'insp', title: 'Settings and runs', text: 'The selected part’s settings, training (with a learning-rate sweep), every run with what changed and a side-by-side comparison, and your model as a PyTorch script.' },
      { sel: '#btnPlay', title: 'Train', text: 'Each press of Train after a change starts a new run from fresh weights, so runs can be compared.' },
    ],
    lab: [
      { sel: '#labCenter', screen: 'network', title: 'The model', text: 'A trained model you can take apart: zoom into any part, down to the arithmetic. The buttons above switch between views of what it does.' },
      { sel: '#labLeft', screen: 'data', title: 'Inputs', text: 'What goes into the model, and its settings.' },
      { sel: '#labRight', screen: 'insp', title: 'Notes and experiments', text: 'What to look for, experiments to try, and the maths and code of the part you select.' },
      { sel: '#btnPlay', title: 'Train', text: 'Continue training, pause or step, where the lab trains live.' },
    ],
  };
  const kindOf = () => (App.mode !== 'lab' ? 'mlp' : App.preset && App.preset.id === 'builder' ? 'builder' : 'lab');
  const phone = () => !!(NA.mobile && NA.mobile.mobile);
  const visible = (e) => !!e && !!e.offsetParent && e.getBoundingClientRect().width > 0;

  const T = { steps: null, i: 0, root: null };
  T.start = function (kind) {
    kind = kind || kindOf();
    T.kind = kind; T.steps = TOURS[kind]; T.i = 0;
    if (App.training) App.pause();
    if (NA.chrome && NA.chrome.toggleHelp) NA.chrome.toggleHelp(false);
    if (!T.root) build();
    T.root.hidden = false;
    document.body.classList.add('touring');
    T.show(0, 1);
  };
  T.end = function () {
    if (!T.root || T.root.hidden) return;
    T.root.hidden = true;
    document.body.classList.remove('touring');
    NA.store.set('tour.' + T.kind, true);
    if (phone() && NA.mobile.go) NA.mobile.go('network');
    const b = $('#helpBtn'); if (b && visible(b)) b.focus();
  };
  T.isOpen = () => !!T.root && !T.root.hidden;
  // the first time each kind of page opens
  T.maybeStart = function () {
    const kind = kindOf();
    if (NA.store.get('tour.' + kind, false) || T.isOpen()) return;
    if (document.querySelector('.modal:not([hidden])')) return;
    setTimeout(() => { if (!T.isOpen() && kindOf() === kind && !NA.store.get('tour.' + kind, false)) T.start(kind); }, 700);
  };

  function build() {
    T.ring = el('div', { class: 'tour-ring', 'aria-hidden': 'true' });
    T.count = el('span', { class: 'tour-count' });
    T.title = el('h2', { class: 'tour-title', id: 'tourTitle' });
    T.text = el('p', { class: 'tour-text', id: 'tourText' });
    T.back = el('button', { class: 'btn btn-quiet', type: 'button', onclick: () => T.show(T.i - 1, -1) }, 'Back');
    T.next = el('button', { class: 'btn btn-primary', type: 'button', onclick: () => (T.i >= T.steps.length - 1 ? T.end() : T.show(T.i + 1, 1)) }, 'Next');
    T.card = el('div', { class: 'tour-card', role: 'dialog', 'aria-modal': 'true', 'aria-labelledby': 'tourTitle', 'aria-describedby': 'tourText' },
      el('div', { class: 'tour-top' }, T.count, el('button', { class: 'link-btn tour-skip', type: 'button', onclick: () => T.end() }, 'Skip the tour')),
      T.title, T.text, el('div', { class: 'tour-nav' }, T.back, T.next));
    T.root = el('div', { class: 'tour', hidden: '' }, el('div', { class: 'tour-block' }), T.ring, T.card);
    T.root.addEventListener('keydown', (e) => {
      if (e.key === 'Escape') { e.stopPropagation(); T.end(); }
      else if (e.key === 'ArrowRight') { e.preventDefault(); T.next.click(); }
      else if (e.key === 'ArrowLeft' && T.i > 0) { e.preventDefault(); T.show(T.i - 1, -1); }
      else if (e.key === 'Tab') { // keep the keyboard inside the card
        const f = [...T.card.querySelectorAll('button:not([disabled])')], k = f.indexOf(document.activeElement);
        if (e.shiftKey && k <= 0) { e.preventDefault(); f[f.length - 1].focus(); } else if (!e.shiftKey && k === f.length - 1) { e.preventDefault(); f[0].focus(); }
      }
    });
    window.addEventListener('resize', () => { if (T.isOpen()) place(); });
    document.body.append(T.root);
  }

  // shows step i, or the nearest one in direction dir whose part is on screen
  T.show = function (i, dir) {
    for (let k = i; k >= 0 && k < T.steps.length; k += dir || 1) {
      const s = T.steps[k];
      if (phone() && s.screen && NA.mobile.go) NA.mobile.go(s.screen);
      const target = document.querySelector(s.sel);
      if (!visible(target)) continue;
      T.i = k; T.target = target;
      target.scrollIntoView({ block: 'nearest', inline: 'nearest' });
      T.count.textContent = `${k + 1} of ${T.steps.length}`;
      T.title.textContent = s.title;
      T.text.textContent = s.text;
      T.back.disabled = k === 0;
      T.next.textContent = k >= T.steps.length - 1 ? 'Done' : 'Next';
      place();
      T.next.focus();
      return;
    }
    if (dir > 0) T.end();
  };
  // the ring around the part, and the card beside it where it fits (docked at the bottom on a phone)
  function place() {
    const r = T.target.getBoundingClientRect(), vw = window.innerWidth, vh = window.innerHeight, pad = 6;
    Object.assign(T.ring.style, { left: r.left - pad + 'px', top: r.top - pad + 'px', width: r.width + pad * 2 + 'px', height: r.height + pad * 2 + 'px' });
    const c = T.card;
    if (phone()) { c.classList.add('is-docked'); Object.assign(c.style, { left: '', top: '' }); return; }
    c.classList.remove('is-docked');
    const cw = c.offsetWidth, ch = c.offsetHeight, gap = 14;
    const options = [
      { x: r.left, y: r.bottom + gap, ok: r.bottom + gap + ch < vh },
      { x: r.left, y: r.top - gap - ch, ok: r.top - gap - ch > 0 },
      { x: r.right + gap, y: r.top, ok: r.right + gap + cw < vw },
      { x: r.left - gap - cw, y: r.top, ok: r.left - gap - cw > 0 },
    ];
    const o = options.find((q) => q.ok) || { x: (vw - cw) / 2, y: vh - ch - 16 };
    c.style.left = Math.round(Math.min(Math.max(12, o.x), vw - cw - 12)) + 'px';
    c.style.top = Math.round(Math.min(Math.max(12, o.y), vh - ch - 12)) + 'px';
  }

  NA.tour = T;
})(window.NA);
