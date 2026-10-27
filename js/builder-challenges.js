/* builder-challenges.js — guided challenges for the model builder: a problem, a starting model that falls short,
   and a goal to reach by changing it. Goals: a validation score (accuracy, R² or token accuracy), at most so many
   parameters, or an optimizer to keep. Every challenge has a known solution, trained by
   tools/builder-challenges-check.js, which also checks that the starting model does not already pass. */
(function (NA) {
  'use strict';
  const D = (units, act = 'relu') => ({ type: 'dense', units, act });
  const LIST = [
    {
      id: 'bend', level: 'Warm-up', title: 'Bend the line', problem: 'moons', template: 'linear',
      brief: 'Two interlocking moons. The model starts with no hidden layer, so all it can draw is a straight line.',
      goals: [{ metric: 0.95 }],
      hint: 'Add a Dense layer between the input and the output: its neurons give the boundary somewhere to bend.',
      solution: { blocks: [D(16)] },
    },
    {
      id: 'v', level: 'Warm-up', title: 'Two neurons for a V', problem: 'abs', template: 'mlp',
      brief: 'Fit the V shape |x| with a tiny model: an R² of at least 0.97 with no more than 10 parameters. The starting model fits it well, with 321.',
      goals: [{ metric: 0.97 }, { params: 10 }],
      hint: '|x| = ReLU(x) + ReLU(−x), so two neurons are enough in principle. But a ReLU that is off on one side gets no gradient from it, so it may never learn that side: a Leaky ReLU keeps a small slope there.',
      solution: { blocks: [D(2, 'leaky')] },
    },
    {
      id: 'calm', level: 'Training', title: 'Calm the steps', problem: 'circle', template: 'mlp', train: { opt: 'momentum', lr: 1 },
      brief: 'Momentum at a learning rate of 1 overshoots: the velocity it builds up carries every step too far, and the loss jumps around instead of settling. Keep momentum, and get it to learn.',
      goals: [{ metric: 0.97 }, { opt: 'momentum' }],
      hint: 'Momentum adds up the last steps, so it moves about ten times as far as the learning rate alone. Try a learning rate ten times smaller.',
      solution: { train: { lr: 0.1 } },
    },
    {
      id: 'spiral', level: 'Design', title: 'Small but mighty', problem: 'spiral', template: 'deep',
      brief: 'Five layers of 24 neurons wind around the two spirals easily, with almost 2,500 parameters. Do it with at most 600.',
      goals: [{ metric: 0.95 }, { params: 600 }],
      hint: 'Depth matters more than width here: two or three narrow layers can still wind around twice.',
      solution: { blocks: [D(16), D(16)] },
    },
    {
      id: 'wiggle', level: 'Design', title: 'A wiggle on a budget', problem: 'formula', template: 'mlp', ps: { formula: 'sin(5x) + x^2' },
      brief: 'Fit sin(5x) + x² with an R² of at least 0.96, using at most 40 parameters. The starting model gets there with 321.',
      goals: [{ metric: 0.96 }, { params: 40 }],
      hint: 'The curve is mostly a sine wave, and a neuron’s activation can be one: try a single layer of about a dozen Sine neurons.',
      solution: { blocks: [D(12, 'sine')] },
    },
    {
      id: 'digits', level: 'Design', title: 'Read the digits', problem: 'digits', template: 'linear',
      brief: 'A single weighted sum per digit can only match one template per class. Reach 97% validation accuracy.',
      goals: [{ metric: 0.97 }],
      hint: 'Convolutions look for the same small stroke everywhere in the picture: try the Small CNN template.',
      solution: { template: 'cnn' },
    },
    {
      id: 'lookback', level: 'Sequences', title: 'Look back', problem: 'reverse', template: 'noattn',
      brief: 'Reverse a sequence of digits. Without attention, each position only sees itself, so the answers are guesses. Reach 90% token accuracy.',
      goals: [{ metric: 0.9 }],
      hint: 'Add a Self-attention block before the dense layers, so every answer position can look back at the input it needs.',
      solution: { template: 'attention' },
    },
  ];
  const byId = Object.fromEntries(LIST.map((c) => [c.id, c]));

  /* How a run measures up: one entry per goal { goal, ok, now } (now: the run's value, for the progress line).
     run: { cfg (the setup it trained), params, last: { vm } } from the model builder. */
  function progress(ch, run) {
    return ch.goals.map((g) => {
      if (g.metric !== undefined) { const v = run && run.last ? run.last.vm : null; return { goal: g, ok: v !== null && v !== undefined && v >= g.metric, now: v }; }
      if (g.params !== undefined) { const v = run ? run.params : null; return { goal: g, ok: v !== null && v <= g.params, now: v }; }
      if (g.opt !== undefined) { const v = run ? run.cfg.train.opt : null; return { goal: g, ok: v === g.opt, now: v }; }
      return { goal: g, ok: false, now: null };
    });
  }
  const passed = (ch, run) => !!run && run.cfg.problem === ch.problem && progress(ch, run).every((p) => p.ok);

  // The setup a challenge starts from (blocks without ids), or with solved, its known solution.
  function setup(ch, solved) {
    const BT = NA.builderTasks, p = BT.byId[ch.problem], fam = BT.FAMILIES[p.family], sol = solved ? ch.solution : {};
    const tpl = (id) => BT.templateBlocks(BT.TEMPLATES.find((t) => t.id === id), p.family);
    return {
      problem: ch.problem, ps: { [ch.problem]: Object.assign(BT.defaults(p), ch.ps) },
      blocks: sol.blocks ? JSON.parse(JSON.stringify(sol.blocks)) : tpl(sol.template || ch.template),
      loss: { id: fam.losses[0], smooth: 0, delta: 0.2 },
      train: Object.assign({}, BT.DEFAULT_TRAIN, BT.trainFor(p), ch.train, sol.train),
      custom: { points: [] },
    };
  }

  NA.builderChallenges = { LIST, byId, progress, passed, setup };
})(window.NA);
