/* builder-insights.js — what is going on in the model builder, in plain words. diagnose() reads the current run (its
   training and validation curves, the gradient sizes, the model's settings) and says whether the model is learning,
   has learned, is stuck, has blown up, memorises its examples (overfitting) or is too simple (underfitting), with the
   numbers behind it and one-click changes to try; for learning from rewards, whether it ever finds the reward,
   explores, looks ahead, or has got worse again, and in each world what its best moves get stuck on (a room that
   overshoots, a lamp it cannot remember, a cart that never reaches the flag, a robot pressed against the wall of a
   deceptive maze, a runner caught in a corner…), read from its episodes played again. read() explains how to read
   the results picture for each kind of problem. Both are shown in the builder's results and its Training tab. */
(function (NA) {
  'use strict';
  const { fmt, fmtPct, fmtInt } = NA;
  const ORDER = { critical: 0, warn: 1, good: 2, info: 3 };
  const lr10 = (lr, k) => +(lr * k).toPrecision(2);
  // after a blow-up, a learning rate that usually works for the optimizer (or a tenth of the old one, if smaller)
  const saneLr = (T) => Math.min(lr10(T.lr, 0.1), T.opt === 'adam' || T.opt === 'adamw' ? 0.01 : T.opt === 'momentum' ? 0.05 : 0.1);

  function diagnose(L) {
    const out = [], add = (level, title, body, actions = []) => out.push({ level, title, body, actions: actions.filter(Boolean) });
    const BT = NA.builderTasks, d = L.run && !L.stale && L.model ? L.data : L.dataFor();
    if (!d || d.error) return out;
    const chk = L.check || L.recheck();
    if (!chk.ok) { add('critical', 'The model does not fit together', 'A block marked in red cannot take what reaches it. Select it to see why: its card offers the part that fixes it.'); return out; }
    if (d.family === 'control') { diagnoseControl(L, d, add, out); return out.sort((a, b) => ORDER[a.level] - ORDER[b.level]); }
    const run = L.run, classic = L.isClassicSetup(), T = L.setup.train, value = BT.isValue(d.family), metric = L.metricName();
    const sc = (v) => (value ? fmt(v, 3) : fmtPct(v, 1)), K = d.output && d.output.kind === 'classes' ? d.output.k : 0, chance = K ? 1 / K : 0;
    const newRun = { label: 'New run', run: () => { L.reset(); if (!L.training) L.play(); } };
    if (!run) {
      add('info', 'Not trained yet', classic ? 'Press Train: a classic model learns from all the training examples at once, a step at a time (one level of a tree, one tree of a forest, one round of boosting).'
        : `The weights are random, so the answers are noise for now. Press Train: every step nudges all ${fmtInt(chk.params)} weights a little downhill on the loss, using a batch of ${T.batch} examples.`);
      return out;
    }
    if (L.stale) { add('info', 'The setup has changed since the last run', 'The pictures show the last run. Train starts a new run with the setup as it is now; the Runs tab keeps the earlier runs to compare.'); return out; }
    const c = run.curve.filter((p) => p.vm !== null && p.vm !== undefined && Number.isFinite(p.val)), n = c.length, last = run.last || {};
    const vm = last.vm, tm = last.tm, gap = tm !== null && tm !== undefined && vm !== null && vm !== undefined ? tm - vm : null;
    if (run.status === 'diverged') {
      add('critical', 'The loss blew up', `At step ${fmtInt(run.step)} the loss grew too large to compute. Each step overshot the valley it was heading for, the next gradient was bigger still, and the weights ran off to infinity. A smaller learning rate (or gradient clipping) keeps the steps in check.`,
        [classic ? null : { label: `Learning rate ${saneLr(T)}, new run`, run: () => { const lr = saneLr(L.setup.train); L.edit((s) => { s.train.lr = lr; }); L.reset(); if (!L.training) L.play(); } }]);
      return out;
    }
    if (n < 2) { add('info', 'Learning', `Step ${fmtInt(run.step)}. The first measurements are on their way.`); return out; }
    const first = c[0], vals = c.map((p) => p.val), minVal = Math.min(...vals), minAt = c[vals.indexOf(minVal)].step, lastVal = vals[n - 1];
    const recent = c.slice(Math.floor(n * 0.6)), gain = recent.length > 1 ? recent[recent.length - 1].vm - recent[0].vm : 0;
    const planned = run.limit || 0, progress = planned ? run.step / planned : 1, settled = run.done || progress >= 0.5 || run.step >= 1500;
    const blocks = L.setup.blocks, dense = blocks.filter((b) => b.type === 'dense'), cb = classic ? blocks.find((b) => NA.cml.isClassic(b.type)) : null;
    const guess = (v) => v !== null && v !== undefined && (value ? v < 0.05 : v <= chance + Math.max(0.03, 0.1 * chance));
    const stuck = !classic && run.step >= 300 && n >= 4 && guess(vm) && (tm === null || tm === undefined || guess(tm)) && (first.val - lastVal) / Math.max(first.val, 1e-9) < 0.05;
    if (stuck) {
      add('critical', 'Stuck at guessing', `${value ? `An R² of ${fmt(vm, 3)} is no better than always answering the average` : `${fmtPct(vm, 1)} is what guessing ${K === 2 ? 'one class' : 'at random'} gets`}, and the loss has stopped falling. Common causes: a learning rate so large that the first steps wrecked the weights (ReLU neurons pushed to output 0 for every input stay “dead”: no gradient reaches them), one so small that nothing moves, or gradients that vanish on the way through many layers. Changing a training setting continues these same weights; a new run starts from fresh ones.`,
        [T.lr >= 0.1 ? { label: `Learning rate ${saneLr(T)}, new run`, run: () => { const lr = saneLr(L.setup.train); L.edit((s) => { s.train.lr = lr; }); L.reset(); if (!L.training) L.play(); } } : null, newRun]);
    }
    // the gradient through the layers: does it reach the first ones?
    if (!classic && !stuck && L.model && L.model.gradNorms && run.step >= 50) {
      const g = L.model.gradNorms(), top = Math.max(...g, 1e-30), ratio = top / Math.max(g[0], 1e-30);
      if (g.length >= 4 && g[0] < 1e-3 * top && top > 1e-8) {
        const squash = dense.some((b) => b.act === 'sigmoid' || b.act === 'tanh');
        add('warn', 'The first layers hardly learn (vanishing gradients)', `${g[0] === 0 ? 'The last step did not change the first layer at all' : `The last step changed the first layer ${ratio > 1e5 ? 'over 100,000' : fmtInt(Math.round(ratio))} times less than the most-changed one`}. The gradient is multiplied by each layer’s slope on its way back, and ${squash ? 'sigmoid and tanh slopes are small (a sigmoid’s is at most 0.25), so it fades layer by layer' : 'small slopes add up over many layers'}. ReLU keeps a slope of 1 where it is active; BatchNorm, LayerNorm and residual connections also keep the gradient alive. The gradient chart in the Training tab shows every layer.`,
          [squash ? { label: 'Switch to ReLU', run: () => L.edit((s) => { s.blocks.forEach((b) => { if (b.type === 'dense' && (b.act === 'sigmoid' || b.act === 'tanh')) b.act = 'relu'; }); }) } : null]);
      }
    }
    // memorising the training examples
    const overfit = gap !== null && n >= (classic ? 3 : 5) && gap >= (value ? 0.08 : 0.06) && (lastVal > minVal * 1.1 || gap >= 0.15);
    if (overfit) {
      const firstDense = blocks.findIndex((b) => b.type === 'dense');
      add('warn', 'Memorising the training examples (overfitting)', `${sc(tm)} on the examples it trains on, but ${sc(vm)} on new ones${lastVal > minVal * 1.1 ? `, and the validation loss has risen ${fmtPct(lastVal / minVal - 1, 0)} since its lowest point at step ${fmtInt(minAt)}` : ''}. The model is fitting the quirks and noise of its own examples, which do not carry over. ${classic ? { tree: 'A shallower tree asks fewer, broader questions; a forest averages many trees.', knn: 'A larger k averages more neighbours, so single noisy points matter less.', forest: 'Shallower trees, or more of them, smooth the answer.', boost: 'A smaller learning rate η, shallower trees or row sampling (subsample) make each round count less.', linear: 'A ridge penalty λ keeps the weights small.' }[cb.type] : 'Less capacity (fewer neurons), dropout, Gaussian noise, weight decay or more examples all push it toward the general pattern.'}`,
        classic ? [cb.type === 'tree' && cb.depth > 2 ? { label: `Depth ${Math.max(2, cb.depth - 3)}`, run: () => L.edit((s) => { const b = s.blocks.find((q) => q.type === 'tree'); b.depth = Math.max(2, b.depth - 3); }) } : null,
          cb.type === 'knn' && cb.k < 15 ? { label: `k = ${Math.min(25, cb.k * 3 + 2)}`, run: () => L.edit((s) => { const b = s.blocks.find((q) => q.type === 'knn'); b.k = Math.min(25, b.k * 3 + 2); }) } : null,
          cb.type === 'boost' && cb.eta > 0.1 ? { label: 'η 0.1', run: () => L.edit((s) => { const b = s.blocks.find((q) => q.type === 'boost'); b.eta = 0.1; }) } : null]
          : [!T.wd ? { label: 'Weight decay 0.01', run: () => L.edit((s) => { s.train.wd = 0.01; if (s.train.opt === 'adam') s.train.opt = 'adamw'; }) } : null,
            dense.some((b) => b.units > 8) ? { label: 'Halve the neurons', run: () => L.edit((s) => { s.blocks.forEach((b) => { if (b.type === 'dense' && b.units > 8) b.units = Math.round(b.units / 2); }); }) } : null,
            firstDense >= 0 && !value && !blocks.some((b) => b.type === 'dropout') ? { label: 'Add dropout', run: () => L.fixBlock(firstDense + 1, [{ type: 'dropout', p: 0.2 }]) } : null]);
    }
    // too simple: it cannot even fit its own training examples
    const underfit = !stuck && !overfit && settled && tm !== null && tm !== undefined && tm < 0.85 && (gap === null || gap < 0.15);
    if (underfit) {
      add('warn', 'Too simple for this problem (underfitting)', `Even on its own training examples it scores ${sc(tm)}: the model cannot represent the pattern${classic ? '' : ', or has not trained long enough'}. ${classic ? { linear: 'A linear model can only draw straight lines (flat planes): trees, k-NN or a network can bend.', tree: 'A deeper tree can ask more questions.', knn: 'A smaller k follows the data more closely.', forest: 'Deeper trees can carve finer regions.', boost: 'Deeper trees, more rounds or a larger η fit more.' }[cb.type] : 'More neurons or layers, an activation that bends (ReLU, tanh), or more steps.'}`,
        classic ? [cb.type === 'tree' || cb.type === 'forest' || cb.type === 'boost' ? { label: 'Deeper trees', run: () => L.edit((s) => { const b = s.blocks.find((q) => q.type === cb.type); b.depth = Math.min(16, (b.depth || 4) + 2); }) } : null]
          : [{ label: 'Add a Dense layer', run: () => L.addPart(L.PARTS.find((p) => p.type === 'dense')) }, L.canContinue() ? { label: 'Continue training', run: () => L.play() } : null]);
    }
    // a run that has stopped improving
    if (!stuck && !overfit && !underfit && !classic && settled && n >= 8 && Math.abs(gain) < 0.005 && vm < (value ? 0.97 : 0.97)) {
      add('info', 'On a plateau', `The validation ${metric} has stayed near ${sc(vm)} over the last part of the run. More of the same training will not change much. ${T.opt === 'sgd' ? 'Adam or momentum often get past flat stretches. ' : ''}A learning-rate schedule (cosine) settles into a finer minimum; a model with more capacity may reach higher.`,
        [T.opt === 'sgd' ? { label: 'Switch to Adam', run: () => L.edit((s) => { s.train.opt = 'adam'; }) } : null, T.sched === 'const' && T.steps ? { label: 'Cosine schedule', run: () => L.edit((s) => { s.train.sched = 'cosine'; }) } : null]);
    }
    // the validation loss jumping about: steps too large
    if (!classic && n >= 10 && !stuck) {
      const tail = vals.slice(-8), jumps = tail.slice(1).map((v, i) => Math.abs(v - tail[i]) / Math.max(tail[i], 1e-9)), mean = jumps.reduce((a, b) => a + b, 0) / Math.max(1, jumps.length);
      if (mean > 0.25) add('warn', 'The loss jumps around', `The validation loss changes by ${fmtPct(mean, 0)} from one measurement to the next on average. The steps are too large to settle into the bottom of the valley: they bounce from wall to wall. A smaller learning rate or gradient clipping calms them.`,
        [{ label: `Learning rate ${lr10(T.lr, 1 / 3)}`, run: () => L.edit((s) => { s.train.lr = lr10(s.train.lr, 1 / 3); }) }, !T.clip ? { label: 'Clip gradients at 1', run: () => L.edit((s) => { s.train.clip = 1; }) } : null]);
    }
    // boosting past its best
    if (classic && cb.type === 'boost' && n >= 6 && minAt < run.step * 0.6 && lastVal > minVal * 1.05) add('info', 'Boosting is past its best', `The validation loss was lowest after round ${fmtInt(minAt)} of ${fmtInt(run.step)}: the later rounds mostly fit the training rows' noise. Real boosting libraries stop early when the validation loss stops falling.`);
    if (classic && cb.type === 'knn' && cb.k === 1) add('info', 'Every answer copies one example', 'With k = 1 the model answers with the class (or value) of the single nearest training example, noise included. Training accuracy is then always perfect, which says nothing about new examples.');
    // all good
    if (!out.some((q) => q.level === 'critical' || q.level === 'warn') && vm !== null && vm !== undefined && vm >= (value ? 0.97 : 0.97) && (gap === null || gap < 0.04)) {
      add('good', 'Learned it', `${sc(vm)} on examples it never trained on${tm !== null && tm !== undefined ? `, and ${sc(tm)} on the ones it did` : ''}: it has found a pattern that carries over to new data. ${classic ? 'Try another kind of model on the same problem and compare them in the Runs tab.' : `Next: how small can the model be and still do this? It has ${fmtInt(chk.params)} parameters now. Or try a challenge.`}`);
    }
    if (!out.length) {
      const up = n >= 2 ? vm - c[Math.max(0, n - 6)].vm : 0;
      add('info', run.done ? 'Finished' : 'Learning', `${run.done ? `After ${fmtInt(run.step)} ${classic ? `${L.model.unit()}s` : 'steps'}` : `At step ${fmtInt(run.step)}`}: ${sc(vm)} on the validation examples${tm !== null && tm !== undefined ? `, ${sc(tm)} on training ones` : ''}.${!run.done && up > 0.002 ? ` Still improving: up ${value ? fmt(up, 3) : fmtPct(up, 1)} over the last few measurements.` : ''} The training and validation losses below should both fall; when only the training one keeps falling, the model has started to memorise.`);
    }
    return out.sort((a, b) => ORDER[a.level] - ORDER[b.level]);
  }

  // learning from rewards: what the run shows, with changes to try
  function diagnoseControl(L, d, add, out) {
    const run = L.run, s = L.ps(), M = NA.brl.METHODS[s.method], ga = NA.brl.isEvo(s.method), unit = ga ? 'generation' : 'step';
    const again = (label, fn) => ({ label, run: () => { L.edit(fn); L.reset(); if (!L.training) L.play(); } }), setPs = (fn) => (S) => fn(S.ps[S.problem]);
    const how = { ga: 'each generation the best players are bred and the rest replaced', es: 'each generation nudged copies of it play, and it moves toward the ones that did better', me: 'it keeps the best player of every style it finds and breeds new ones from them' }[s.method];
    if (!run) { add('info', 'Not trained yet', `Press Train: the network plays in the world, and ${how || `${M.short} turns what happens into steps on its weights`}. The picture plays it live, with its best moves.`); return; }
    if (L.stale) { add('info', 'The setup has changed since the last run', 'The pictures show the last run. Train starts a new run with the setup as it is now; the Runs tab keeps the earlier runs to compare.'); return; }
    if (run.status === 'diverged') {
      // evolution stops only when a fitness is no number at all; gradient methods also when the returns grow huge
      const own = !!d.ownReward && !!s.rFormula, table = again('Use the reward table, new run', setPs((q) => { q.rFormula = ''; }));
      const big = (NA.rlEnvs.REWARDS[d.world] || []).filter((r) => typeof d.ws[r.key] === 'number' && Math.abs(d.ws[r.key]) >= 10 && Math.abs(d.ws[r.key]) >= 10 * Math.abs(r.def || 0.1));
      if (ga) add('critical', 'A fitness came out as no number', `At generation ${fmtInt(run.step)} a member’s episodes added up to no number at all, so it could not be ranked. ${own ? `Your reward formula, “${s.rFormula}”, gave no number at some step: a division by zero, or the square root or logarithm of a number below 0. Mend the formula, or go back to the reward table.` : 'Some reward gave no number.'} Evolution has no learning rate: its rewards can be as large as you like, but each has to be a number.`, [own ? table : null]);
      else add('critical', 'The loss blew up', `At step ${fmtInt(run.step)} the loss grew too large to compute. ${own ? `Your reward formula, “${s.rFormula}”, may give very large amounts, or at some step no number at all (a division by zero?). ` : big.length ? `Some of its rewards are large (${big.map((r) => `${r.label.replace(/ \(.*\)$/, '').toLowerCase()}: ${fmt(d.ws[r.key], 0)}`).join('; ')}), and over an episode they add up to returns far bigger than the network’s outputs can follow. ` : 'Learning from rewards is touchy. '}Smaller reward amounts${s.method === 'a2c' || s.method === 'ppo' ? ', normalising the rewards' : ''} or a learning rate ten times smaller usually cure it.`,
        [again(`Learning rate ${+(L.setup.train.lr / 10).toPrecision(2)}, new run`, (S) => { S.train.lr = +(S.train.lr / 10).toPrecision(2); }), own ? table : big.length ? again('Rewards as at first, new run', setPs((q) => { for (const r of big) q[r.key] = r.def; })) : null]);
      return;
    }
    const c = run.curve.filter((p) => p.vm !== null && p.vm !== undefined), n = c.length, vm = run.last ? run.last.vm : null, best = run.bestVm, ret = run.last ? run.last.ret : null;
    if (n < 2) { add('info', 'Learning', `${ga ? 'Generation' : 'Step'} ${fmtInt(run.step)}. The first scores are on their way.`); return; }
    const low = vm !== null && vm < 0.9, w = d.world;
    // (shaped: the closer bonus in the table, or a formula that pays for getting closer)
    const shaped = d.ownReward ? /(closer|dist)/.test(s.rFormula || '') : s.closer;
    if (w === 'maze' && !shaped && !best && run.step >= (ga ? 50 : 1000)) {
      const cost = d.ownReward ? 0 : amount(d, 'rStep');
      add('warn', 'It never finds the goal', `Its best moves have not reached the goal once. ${d.ownReward ? 'Your formula pays' : 'Its reward pays'} nothing that says where the goal is until it gets there${cost ? `: the cost of a step (${String(cost).replace('-', '−')}) is the same whichever way it goes` : ''}, so random moves almost never find the way and every route it tries looks as bad as the next. A bonus for each step closer to the goal (reward shaping) says which way is warmer from the first move.`,
        [d.ownReward ? ((s.rFormula || '').length < 186 ? again('Add + 0.1closer to your formula, new run', setPs((q) => { q.rFormula = `${q.rFormula} + 0.1closer`; })) : null) : again('Closer bonus on, new run', setPs((q) => { q.closer = true; }))].filter(Boolean));
    }
    if (s.method === 'dqn' && !s.explore && low && run.step >= 300) {
      add('warn', 'It never tries anything new', `With no exploration it always plays the move that looks best so far. A move that paid once looks best from then on, and the others are never tried, so it never finds out they pay more. Random moves at first, fewer and fewer (ε-greedy), are how DQN explores.`,
        [again('Explore for 1,500 moves, new run', setPs((q) => { q.explore = 1500; }))]);
    }
    if (!ga && w !== 'levers' && s.gamma === 0 && low && run.step >= 500) {
      add('warn', 'It only cares about the next reward', `With a discount of 0, a reward that comes later is worth nothing now, so the moves that lead up to it learn nothing. Here the reward comes at the end of an episode: a discount near 1 carries it back to the moves that earned it.`,
        [again('γ = 0.9, new run', setPs((q) => { q.gamma = 0.9; }))]);
    }
    if (d.input.kind === 'series' && L.setup.blocks[0] && L.setup.blocks[0].type === 'seqpool' && L.setup.blocks[0].mode === 'last' && low) {
      add('warn', 'It sees only the latest reading', w === 'memory' ? 'The Sequence pool keeps the last reading and drops the other 15, the lamp among them, so at the junction it sees only the present again. Something that reads all 16 can look back: a Flatten, or a recurrent layer first.'
        : `The Sequence pool keeps the last reading and drops the rest, so the model gets a position and an angle but cannot tell which way they are moving, and a pole at the same angle can be falling or rising. Something that reads all 4 readings can work the speeds out: a Flatten, or a recurrent layer.`,
        [again('Flatten instead, new run', (S) => { S.blocks[0] = { id: S.blocks[0].id, type: 'flatten' }; })]);
    }
    // what its best moves get stuck on in this world, from its episodes played again
    const notes = WORLD_NOTES[w], got = notes && L.agent ? replay(L, d) : null;
    if (got) {
      const planned = run.limit || NA.brl.stepsFor(w, s.method);
      notes({ L, d, s, run, eps: got.eps, W: got.W, add, again, setPs, ga, method: s.method, own: !!d.ownReward, settled: run.done || run.step >= 0.25 * planned, switchTo: (m) => switchTo(L, m) });
    }
    if (ga && L.check && L.check.params > 1500 && low && run.step >= 100) {
      // (in the words of the method it learns by)
      const est = s.method === 'es' || s.method === 'cem', how = { es: 'Evolution strategies work out which way to move every weight from a few nudged copies of the network', cem: 'The cross-entropy method draws whole networks from a bell curve over the weights and refits the curve to the best',
        me: 'MAP-Elites mutates weights and keeps the best network of each style of play', novelty: 'Novelty search mutates weights and keeps what behaves in new ways', erl: 'Evolution mutates weights and keeps what plays better, with a DQN joining it now and then' }[s.method] || 'A genetic algorithm searches blindly, by mutating weights and keeping what plays better';
      add('warn', 'A lot of weights to evolve', `${how}. With ${fmtInt(L.check.params)} weights, ${est ? 'what a few dozen tries can tell it about each one is mostly noise' : 'most mutations are noise'}, and progress is slow. A smaller network gives each ${est ? 'try' : 'mutation'} more say.`);
    }
    /* grow and prune: new shapes that keep losing. The best has kept one shape for 40 generations or more while other
       shapes played beside it (the generations' shape counts), and it has not learned the world yet. */
    const ev = L.agent && L.agent.evo;
    if (ga && run.grow && ev && ev.grow && ev.history.length >= 40 && low) {
      const H = ev.history, now = H[H.length - 1].units, same = (u) => u.length === now.length && u.every((x, i) => x === now[i]);
      let k = H.length - 1;
      while (k > 0 && same(H[k - 1].units)) k--;
      const stuck = H.length - k, tried = Math.max(1, ...H.slice(k).map((h) => h.shapes || 1)), g = ev.gopt;
      if (stuck >= 40 && tried > 1) {
        add('info', 'New shapes keep losing', `For ${fmtInt(stuck)} generations the best network has kept the shape ${now.length ? now.join(' → ') : 'with no hidden layer'}, though up to ${tried} shapes played in a generation. A new shape usually starts a little worse (${g.neuron === 'random' ? 'a new neuron joins in at random' : 'a new neuron does nothing yet'}, a new layer is a bottleneck), so it is thrown out before mutations can tune it${g.penalty === 'strong' ? ', and a strong size penalty makes every bigger shape start further behind' : ''}. ${g.protect ? 'New shapes are protected already: this shape may simply be the right size for the world.' : 'Protecting new shapes keeps the best of each new one for 5 generations, so it gets time to catch up.'}`,
          [g.protect ? null : again('Protect new shapes, new run', setPs((q) => { q.growProtect = true; })), g.penalty === 'strong' ? again('Light size penalty, new run', setPs((q) => { q.penalty = 'light'; })) : null,
            g.neuron === 'random' ? again('Silent new neurons, new run', setPs((q) => { q.growNeuron = 'silent'; })) : null].filter(Boolean));
      }
    }
    if (best !== null && best !== undefined && vm !== null && best - vm >= 0.3 && n >= 6) {
      add('warn', 'It got worse again', ga ? `Its best moves scored ${fmtPct(best, 0)} at generation ${fmtInt(run.bestAt)} and ${fmtPct(vm, 0)} now. Each generation is judged on its own episodes, and the fittest of one can do worse from the starts the score uses. Smaller mutations change the best less from one generation to the next.` : `Its best moves scored ${fmtPct(best, 0)} at ${unit} ${fmtInt(run.bestAt)} and ${fmtPct(vm, 0)} now. Learning from rewards is unstable: the network makes its own data by playing, so a change for the worse also changes what it learns from next. A smaller learning rate steadies it${s.method === 'pg' || s.method === 'a2c' ? '; PPO is built to keep each change small' : ''}.`,
        [ga ? null : again(`Learning rate ${+(L.setup.train.lr / 3).toPrecision(2)}, new run`, (S) => { S.train.lr = +(S.train.lr / 3).toPrecision(2); }), s.method === 'pg' || s.method === 'a2c' ? { label: 'Switch to PPO', run: () => L.setMethod('ppo') } : null]);
    }
    // reward hacking: a reward of your own that it collects more and more of, while the real goal (the score) stays out
    // of reach (the return: its episodes', or evolution's best fitness; early and late, a third of the run each)
    const rets = c.filter((p) => Number.isFinite(p.ret)), changed = !!d.ownReward || (NA.rlEnvs.REWARDS[w] || []).some((r) => typeof s[r.key] === 'number' && s[r.key] !== r.def);
    const far = run.done || run.step >= 0.4 * (run.limit || NA.brl.stepsFor(w, s.method)); // (not while an ordinary run is still getting there)
    if (changed && far && rets.length >= 6 && vm !== null && vm < 0.1 && (best === null || best === undefined || best < 0.15)) {
      const k = Math.max(2, Math.floor(rets.length / 3)), avg = (q) => q.reduce((a, p) => a + p.ret, 0) / q.length, early = avg(rets.slice(0, k)), late = avg(rets.slice(-k));
      if (late - early > Math.max(0.5 * Math.abs(early), 0.5)) {
        add('warn', 'It collects the reward, not the goal', `Its ${ga ? 'best fitness' : 'episodes'} went from about ${fmt(early, 2)} to ${fmt(late, 2)}, but its best moves still score ${fmtPct(vm, 0)}: it has found a way to collect ${d.ownReward ? 'what your formula pays' : 'what your reward table pays'} without doing what the score measures (${NA.rlEnvs.WORLDS[w].goal}). An agent makes the most of the reward it is given, not of the goal meant by it: reward hacking. What does it pay for that can be had without the goal?`,
          [again('Rewards as at first, new run', setPs((q) => { q.rFormula = ''; for (const r of NA.rlEnvs.REWARDS[w] || []) delete q[r.key]; }))]);
      }
    }
    if (!out.some((q) => q.level === 'warn' || q.level === 'critical') && vm !== null && vm >= 0.95) {
      add('good', 'Learned it', `Its best moves score ${fmtPct(vm, 0)}${d.ws.random ? ' on mazes it has never seen' : ''}. ${ga ? 'Evolution found weights that play well without a single gradient.' : 'From nothing but rewards, it has found what to do.'} Try another method on the same world, or another way of seeing it, and compare the runs.`);
    }
    if (!out.length) add('info', run.done ? 'Finished' : 'Learning', `${run.done ? `After ${fmtInt(run.step)} ${unit}s` : `At ${unit} ${fmtInt(run.step)}`}: its best moves score ${fmtPct(vm, 0)}${best > vm ? ` (best so far ${fmtPct(best, 0)})` : ''}${ret !== null && ret !== undefined ? `, and its episodes return ${fmt(ret, 2)} on average` : ''}.${run.done && low ? ' Press Continue to train it longer.' : ''}`);
  }

  // ------------------------------------------------------------------ what it gets stuck on, world by world
  /* Its best moves, played again from the starts its score uses (the score's own episodes, in a world of their own
     made with the run's settings): how each episode ended, and what that world's notes need, gathered as it plays
     (WATCH: before and after each step, and at the end). Once per measurement, since the notes are asked for many
     times between two. { eps, W }: the episodes, and the world, whose settings the notes quote. */
  const replays = new WeakMap();
  function replay(L, d) {
    const A = L.agent, run = L.run, w = WATCH[d.world];
    if (!A || !w || A.worldId !== d.world || typeof A.policy !== 'function') return null;
    const key = `${run.n}:${run.last ? run.last.step : -1}`, got = replays.get(A);
    if (got && got.key === key) return got.r;
    const W = NA.rlEnvs.make(A.worldId, A.ws), base = W.step, eps = [];
    let acc = {}, r = null;
    W.step = function (a, rng) {
      if (w.before) w.before(W, acc);
      const t = base.call(W, a, rng);
      if (w.after) w.after(W, acc);
      if (t.done) { eps.push(Object.assign({ end: W.end, t: W.t }, w.end(W, acc))); acc = {}; }
      return t;
    };
    try { W.score((o) => A.policy(o)); r = eps.length ? { eps, W } : null; } catch (e) { r = null; }
    replays.set(A, { key, r });
    return r;
  }
  // a setting as a plain number: 4, 2.5, 0.12
  const num = (v) => String(+(+v).toFixed(2));
  const max = (a, k) => a.reduce((m, e) => Math.max(m, e[k]), -Infinity), min = (a, k) => a.reduce((m, e) => Math.min(m, e[k]), Infinity);
  const mean = (a, k) => a.reduce((s, e) => s + e[k], 0) / Math.max(1, a.length);
  // "in all 3 games", "in both tries", "in 2 of 10 starts"
  const of = (k, n, things) => (k === n ? (n === 2 ? `both ${things}` : `all ${n} ${things}`) : `${k} of ${n} ${things}`);
  const minutes = (steps) => (steps < 9 ? `${Math.max(1, Math.round(steps)) * 10} minutes` : `${fmt(steps / 6, 1).replace(/\.0$/, '')} hours`);
  // a reward table's amount as the world uses it (yours where set, else its default)
  const amount = (d, key) => { const r = (NA.rlEnvs.REWARDS[d.world] || []).find((q) => q.key === key); return typeof d.ws[key] === 'number' ? d.ws[key] : r ? r.def : 0; };
  const isHidden = (s, name) => Array.isArray(s.hide) && s.hide.includes(name);
  const show = (name) => (q) => { q.hide = (q.hide || []).filter((h) => h !== name); };
  const METHOD_NAME = { ga: 'a genetic algorithm', a2c: 'actor-critic', me: 'MAP-Elites', novelty: 'novelty search', cem: 'the cross-entropy method', erl: 'evolutionary RL' };
  function switchTo(L, m) { const M = NA.brl.METHODS[m]; return M ? { label: `Switch to ${METHOD_NAME[m] || M.short}`, run: () => L.setMethod(m) } : null; }
  const keepGoing = (L) => (L.canContinue && L.canContinue() ? { label: 'Continue training', run: () => L.play() } : null);
  const newRun = (L) => ({ label: 'New run', run: () => { L.reset(); if (!L.training) L.play(); } });
  // does the segment p–q cross the segment a–b?
  const side = (ax, ay, bx, by, px, py) => Math.sign((bx - ax) * (py - ay) - (by - ay) * (px - ax));
  const crosses = (px, py, qx, qy, [ax, ay, bx, by]) => side(ax, ay, bx, by, px, py) * side(ax, ay, bx, by, qx, qy) < 0 && side(px, py, qx, qy, ax, ay) * side(px, py, qx, qy, bx, by) < 0;
  const segDist = (px, py, [ax, ay, bx, by]) => { const vx = bx - ax, vy = by - ay, t = Math.max(0, Math.min(1, ((px - ax) * vx + (py - ay) * vy) / (vx * vx + vy * vy || 1))); return Math.hypot(px - ax - t * vx, py - ay - t * vy); };
  /* The room as it is set up, by its own physics (steps of 10 minutes; the target T, the band b, 5 °C less when
     nobody is in): how long it takes to warm from T − 5 to T − b with the heating on and to cool from T to T − 5 + b
     with it off, and how warm it still gets when the heating goes off half a degree short of T (the radiator's heat
     still to come). */
  const rooms = new Map();
  function roomTimes(ws) {
    const key = JSON.stringify([ws.climate, ws.insulation, ws.warm, ws.band, ws.radiator]);
    if (!rooms.has(key)) {
      const W = NA.rlEnvs.make('thermo', Object.assign({}, ws, { schedule: 'home', noise: 0 })), T = W.warm || 21, b = W.band || 1;
      let warm = 0, cool = 0;
      W.reset(new NA.RNG(2));
      W.t = 30; W.T = T - 5; W.R = W.T; while (W.T < T - b && warm < 300) { W.step(1); warm++; }
      W.t = 48; W.T = T; W.R = T; while (W.T > T - 5 + b && cool < 300) { W.step(0); cool++; }
      W.t = 60; W.T = T - 3; W.R = W.T; for (let k = 0; W.T < T - 0.5 && k < 300; k++) W.step(1);
      let top = W.T; for (let k = 0; k < 30; k++) { W.step(0); top = Math.max(top, W.T); }
      rooms.set(key, { warm, cool, top });
    }
    return rooms.get(key);
  }
  /* Hurdles at its running and jumping speeds: the distances before a hurdle from which a jump clears it, [from, to]
     in metres (from its own physics, a take-off every 10 cm tried; the longest unbroken stretch, since a step can
     happen to carry it past a hurdle it would have touched). */
  const leaps = new Map();
  function takeOff(ws) {
    const key = JSON.stringify([ws.run, ws.jump]);
    if (!leaps.has(key)) {
      let best = null, from = null;
      for (let k = 1; k <= 151; k++) {
        const D = k / 10, W = NA.rlEnvs.make('hurdles', { run: ws.run, jump: ws.jump });
        W.reset(new NA.RNG(1)); W.obst = [{ x: 30, bar: false }]; W.x = 30 - D;
        for (let s = 0; s < 400 && !W.hit && W.x < 31; s++) W.step(s ? 0 : 1);
        const clear = !W.hit && k <= 150;
        if (clear && from === null) from = D;
        if (!clear && from !== null) { const span = [from, D - 0.1]; if (!best || span[1] - span[0] > best[1] - best[0]) best = span; from = null; }
      }
      leaps.set(key, best && best[1] - best[0] >= 0.3 ? best : null);
    }
    return leaps.get(key);
  }

  /* What each world's notes read from an episode. The room, a day at a time: steps too warm while the radiator still
     heats it (overshoot), too warm with the heating long off or just after the target drops (a room cools slowly),
     too cold in the two hours after the target rises (people come in) and at other times, and whether the heating
     was on for half of the hour before each rise. */
  const WATCH = {
    thermo: {
      after(W, a) {
        const at = (t) => W.target((Math.max(0, t) / 6) % 24), tg = at(W.t), off = W.T - tg, b = W.band || 1;
        a.on = (a.on || []).concat(W.heat).slice(-7);
        if (tg > at(W.t - 1)) { a.rises = (a.rises || 0) + 1; if (a.on.slice(0, -1).filter(Boolean).length >= 3) a.ready = (a.ready || 0) + 1; }
        if (off > b) { if (W.R > W.T + 2 && at(W.t - 6) <= tg) { a.over = (a.over || 0) + 1; a.hot = Math.max(a.hot || 0, off); } else a.drop = (a.drop || 0) + 1; }
        else if (off < -b) { if (at(W.t - 12) < tg) a.late = (a.late || 0) + 1; else a.cold = (a.cold || 0) + 1; }
      },
      end: (W, a) => { const days = Math.max(1, W.t / 144); return { over: (a.over || 0) / days, hot: a.hot || 0, drop: (a.drop || 0) / days, late: (a.late || 0) / days, cold: (a.cold || 0) / days, rises: a.rises || 0, ready: a.ready || 0, on: W.used / Math.max(1, W.t), comfy: W.inBand / Math.max(1, W.t) }; },
    },
    // where it was when it touched something: in the air (rising or falling) or on the ground, ducking or not
    hurdles: { end: (W) => ({ run: Math.min(1, W.x / (W.course || 150)), bar: !!(W.hit && W.hit.bar), air: W.y > 0.02, rising: W.vy > 0, ducking: W.duck > 0 }) },
    memory: { end: (W) => ({ choice: W.choice }) },
    mountain: { end: (W) => ({ hi: W.hi }) },
    // the pendulum: how high its weight got (1 is straight up)
    swingup: { after(W, a) { a.top = Math.max(a.top ?? -1, Math.cos(W.th)); }, end: (W, a) => ({ first: W.first, late: W.upLate / (W.maxSteps / 2), top: a.top }) },
    // the arm: steps its hand was near enough the target (12 cm unless set) but too fast to count (reaching needs < 0.6 m/s)
    arm: {
      before(W, a) { a.x = W.hx; a.y = W.hy; },
      after(W, a) { if (W.d < (W.reach || 0.12) && Math.hypot(W.hx - a.x, W.hy - a.y) / 0.05 >= 0.6) a.fast = (a.fast || 0) + 1; a.near = Math.min(a.near ?? Infinity, W.d); },
      end: (W, a) => ({ fast: a.fast || 0, near: a.near ?? W.d, d: W.d }),
    },
    traffic: { after(W, a) { for (const q of W.q) a.q = Math.max(a.q || 0, q.length); }, end: (W, a) => ({ switches: W.switches, ns: W.nsGreen / Math.max(1, W.t), queue: a.q || 0 }) },
    walker: { end: (W) => ({ dist: W.cx() - W.x0 }) },
    // the deceptive maze: is the goal behind a wall from where it ended, and is it up against that wall?
    trap: {
      end(W) {
        const [gx, gy] = W.maze.goal, between = W.walls.filter((s) => crosses(W.x, W.y, gx, gy, s));
        return { d: W.d, bumps: W.bumps, behind: between.length > 0, pressed: between.some((s) => segDist(W.x, W.y, s) < 12) };
      },
    },
    // the chase: how near the edge of the arena (20 m across) it was caught, and whether in a corner
    chase: { end: (W) => { const ex = Math.min(W.me.x, 20 - W.me.x), ey = Math.min(W.me.y, 20 - W.me.y); return { edge: Math.min(ex, ey), corner: Math.max(ex, ey) < 2.5 }; } },
    herd: { end: (W) => ({ penned: W.sheep.filter((q) => q.penned).length, n: W.n }) },
  };

  /* The notes, from the episodes: what went wrong in them, in numbers, why, and the change that helps here. Every
     change offered was measured in node from the page's own start (the network built with seed 1, runs 1 to 5):
     on the mountain cart a genetic algorithm reached the flag from all 10 starts in 5 of 5 runs, and with the
     energy bonus at 100 DQN did too (5 of 5) and REINFORCE (2 of 2), but not actor-critic or PPO (0 of 2 each),
     nor PPO or DQN given a 1,000-step episode or the stronger engine (0 of 2 each); the robot arm by actor-critic
     reached 9 targets of 10 with the approach reward and none without it (0 of 2); the sheep were never penned
     without the bonus for bringing them nearer the gate (0 of 2), and evolution penned all 3 at its best with it;
     the room's heating stayed on all day when PPO was not shown the radiator (2 of 2) and was right 99% of the
     day when it was; the lamp was remembered every time with the last 16 readings (PPO, DQN and evolution with a
     Flatten, PPO with a GRU) and half the time with only the present, by every method. Two changes that sound
     right did not help, so they are not offered: a fall costing 5 taught the walker to stand still (PPO, 2 of 2),
     and a cost for each switch left the traffic lights flickering. */
  const WORLD_NOTES = {
    thermo(c) {
      const { eps, add, s, d } = c, n = eps.length, on = mean(eps, 'on'), over = mean(eps, 'over'), drop = mean(eps, 'drop'), late = mean(eps, 'late'), cold = mean(eps, 'cold');
      if (!c.settled || mean(eps, 'comfy') >= 0.95) return;
      const radiator = isHidden(s, 'radiator temperature') ? c.again('Show it the radiator, new run', c.setPs(show('radiator temperature'))) : null;
      if (on >= 0.9) { add('warn', 'The heating never goes off', `The heating is on ${fmtPct(on, 0)} of the day, and the room gets up to ${fmt(max(eps, 'hot'), 0)} °C too warm. With the heating on the room only gets warmer: it has to switch off once the room is warm enough, and a little before, since the radiator stays hot.${radiator ? ' It is not shown the radiator’s temperature, which says how much warmth is still to come.' : ''}`, [radiator]); return; }
      if (on <= 0.02) { add('warn', 'The heating never comes on', `The heating stays off all day, and the room is too cold for ${minutes(cold + late)} a day. ${!c.own && amount(d, 'rHeat') < -0.01 ? `Each step of heating costs ${fmt(-amount(d, 'rHeat'))} in its reward: heating looks dearer to it than the cold.` : 'It has yet to find out that heating pays.'}`); return; }
      const room = roomTimes(d.ws), T = c.W.warm || 21, b = c.W.band || 1;
      if (over >= 3) add('warn', 'It overshoots', `The room gets too warm for ${minutes(over)} a day while the radiator is still hot, up to ${fmt(max(eps, 'hot'), 1)} °C over the target. The radiator goes on warming the room after the heating is switched off, so the time to switch off is before the room is warm enough${room.top > T + b ? `: switched off at ${num(T - 0.5)} °C, this room still climbs to ${fmt(room.top, 1)} °C` : ''}. The radiator’s temperature, among its inputs, says how much warmth is still to come.`, [radiator]);
      const rises = eps.reduce((k, e) => k + e.rises, 0), ready = eps.reduce((k, e) => k + e.ready, 0);
      if (rises && late >= 9 && ready <= rises / 2) add('warn', 'It waits for people to come in', `The target jumps from ${num(T - 5)} to ${num(T)} °C at 6 in the morning and 5 in the evening, and ${ready ? `only ${ready} of the ${rises} times` : `not once in ${rises}`} was the heating on in the hour before, so the room is too cold for ${minutes(late)} a day just after. This room takes ${minutes(room.warm)} to warm from ${num(T - 5)} to ${num(T - b)} °C: the heating has to go on before people come in, which the time of day among its inputs can tell it.`,
        [isHidden(s, 'time of day') ? c.again('Show it the time, new run', c.setPs(show('time of day'))) : null]);
      if (cold >= 6) add('warn', 'The room gets too cold', `The room is more than ${num(b)} °C below the target for ${minutes(cold)} a day, with the heating on ${fmtPct(on, 0)} of the time.${!c.own && amount(d, 'rHeat') < -0.01 ? ` Each step of heating costs ${fmt(-amount(d, 'rHeat'))} in its reward, which can make the cold look cheaper.` : ''}`);
      if (drop >= 12) add('info', 'Too warm after the target drops', `For ${minutes(drop)} a day the room is too warm with the heating off: when the target drops from ${num(T)} to ${num(T - 5)} °C, at 8 in the morning and 11 at night, the room can only cool as fast as the walls let the heat out, and from ${num(T)} to ${num(T - 5 + b)} °C that takes ${minutes(room.cool)} here. No thermostat can help that, so this setup cannot score 100%.`);
    },
    hurdles(c) {
      const { eps, add } = c, n = eps.length, hits = eps.filter((e) => e.end === 'hurdlehit' || e.end === 'barhit');
      if (!c.settled || !hits.length || mean(eps, 'run') >= 0.95) return;
      const span = takeOff(c.d.ws), when = span ? `between about ${span[0].toFixed(1)} and ${span[1].toFixed(1)} m before it` : 'at just the right distance before it';
      const KIND = (e) => (e.bar ? (e.air ? 'barjump' : 'tall') : !e.air ? (e.ducking ? 'duck' : 'none') : e.rising ? 'late' : 'early');
      const k = {}; for (const e of hits) k[KIND(e)] = (k[KIND(e)] || 0) + 1;
      const top = Object.keys(k).sort((a, b) => k[b] - k[a])[0], times = (q) => (k[q] === 1 ? 'once' : k[q] === 2 ? 'twice' : `${k[q]} times`);
      const WHAT = { early: `came down on a hurdle ${times('early')} (it took off too early)`, late: `hit a hurdle on the way up ${times('late')} (it took off too late)`, duck: `ducked at a hurdle ${times('duck')}`, none: `ran into a hurdle without jumping ${times('none')}`, barjump: `jumped into a bar ${times('barjump')}`, tall: `ran into a bar without ducking ${times('tall')}` };
      const TITLE = { early: 'It takes off too early', late: 'It takes off too late', duck: 'It ducks at the hurdles', none: 'It runs into the hurdles', barjump: 'It jumps into the bars', tall: 'It runs into the bars' };
      const WHY = { early: `A jump keeps its feet above a hurdle’s 0.7 m for only a short stretch of the run: it clears one only when it takes off ${when}, and the input “next: how far” says when.`, duck: 'Ducking keeps its head under a bar, but its feet stay on the ground: a hurdle has to be jumped. The inputs say whether the next obstacle is a hurdle or a bar.', none: `A jump clears a hurdle only when it takes off ${when}; the input “next: how far” says when.`, barjump: 'A jump takes its head higher still, into the bar, whose underside is 1.2 m up: ducking keeps its head at 1 m, under it.', tall: 'Standing, its head is at 1.8 m, above a bar’s underside at 1.2 m: it has to duck as it reaches one, and stay down until it is past.' };
      WHY.late = WHY.early;
      add('warn', TITLE[top], `${of(hits.length, n, 'courses').replace(/^./, (x) => x.toUpperCase())} ended at an obstacle: it ${Object.keys(k).sort((a, b) => k[b] - k[a]).map((q) => WHAT[q]).join(', ')}. ${WHY[top]}`,
        [c.method !== 'ga' ? c.switchTo('ga') : null]);
    },
    memory(c) {
      const { eps, add, s, L, W } = c, n = eps.length, right = eps.filter((e) => e.end === 'right').length, lost = eps.filter((e) => e.end === 'lost').length, turns = eps.filter((e) => e.choice !== null && e.choice !== undefined), left = turns.filter((e) => e.choice === 0).length;
      const wait = `${W.wait || 5} step${(W.wait || 5) === 1 ? '' : 's'}`, dark = (W.lit || 1) <= (W.len || 6);
      if (s.view !== 'history' && dark && right <= 0.6 * n) {
        const same = turns.length >= n / 2 && (left === turns.length || left === 0) ? `, turning ${left ? 'left' : 'right'} every time` : '';
        add('warn', 'It cannot remember the lamp', `It turned the way the lamp said in ${right} of ${n} corridors${same}${lost ? `, and in ${lost} it stood at the junction until its ${wait} ran out` : ''}. Seeing only the present, it reaches the junction with nothing to tell it which way the lamp pointed, steps after it went dark: whatever it learns, half its turns are wrong. The Last 16 readings view gives it its recent past, the lamp among it.`,
          [{ label: 'Last 16 readings, new run', run: () => { L.setWorldView('history'); L.reset(); if (!L.training) L.play(); } }]);
      } else if (c.settled && lost >= n / 2) add('warn', 'It waits at the junction', `In ${lost} of ${n} corridors it stood at the junction until its ${wait} ran out, without turning. Waiting costs as much as a wrong turn: it has not yet found that turning the way the lamp said pays.`);
    },
    mountain(c) {
      const { eps, add, d, ga } = c, n = eps.length;
      if (!c.settled || eps.some((e) => e.end === 'flag')) return;
      const tick = amount(d, 'rTick'), energy = amount(d, 'rEnergy'), steps = c.W.maxSteps, flat = !c.own && tick < 0 && !energy && !amount(d, 'rFlag');
      const bonus = (c.method === 'dqn' || c.method === 'pg') && !c.own && !energy, critic = c.method === 'ppo' || c.method === 'a2c';
      const why = energy > 0 && !c.own ? (critic ? ' Even paid for gaining energy, PPO and actor-critic never got there in our runs, where DQN and REINFORCE did.' : '') : ' Its random moves have never rocked it as far as the flag, so it has never seen anything pay better.';
      add('warn', 'It never reaches the flag', `From ${of(n, n, 'starts')} it ran out of time, never further right than x = ${fmt(max(eps, 'hi'), 2)} (the flag is at 0.5). ${flat ? `Every step costs ${fmt(-tick, 0)} until the flag, so every episode it plays returns −${fmtInt(-tick * steps)} whatever it does, and nothing tells a better try from a worse one. ` : ''}The engine is too weak to drive straight up: it has to rock, first up the slope behind it.${ga ? '' : `${why} A genetic algorithm tries whole networks instead, and one that pushes the way the cart is already rolling gets out within a few generations.${bonus ? ' A bonus for gaining energy (height plus speed) pays for rocking from the first swing.' : ''}`}`,
        [ga ? null : c.switchTo('ga'), bonus ? c.again('Energy bonus 100, new run', c.setPs((q) => { q.rEnergy = 100; })) : null]);
    },
    swingup(c) {
      const { eps, add, ga, L, W, d } = c, n = eps.length, never = eps.filter((e) => e.first === null).length, fallen = eps.filter((e) => e.first !== null && e.end !== 'balanced');
      if (!c.settled) return;
      /* by method, as measured: evolution swings it up and balances it; DQN does too when the height of the weight
         pays 1 a step (5 runs of 5), but hardly at 0.01; PPO and actor-critic never balanced it (0 of 7 runs) */
      const weak = W.torque < W.g, height = c.own ? null : amount(d, 'rHeight'), dqn = c.method === 'dqn', faint = dqn && height !== null && height > 0 && height < 0.5, pg = !ga && !dqn;
      const louder = faint ? c.again('Height pays 1, new run', c.setPs((q) => { q.rHeight = 1; })) : null;
      const why = faint ? ` Paid only ${num(height)} a step for the weight’s height, DQN can hardly tell a good swing from a bad one; paid 1, it learned to swing it up and balance it in 5 runs of 5.`
        : pg ? ' Pushes at random this way and that cancel out, and PPO and actor-critic never balanced it in our runs; DQN, paid well for height, and evolution did.' : '';
      if (never === n) {
        const swing = Math.round(180 - (Math.acos(Math.max(-1, Math.min(1, max(eps, 'top')))) * 180) / Math.PI);
        add('warn', 'It never gets it up', `In ${of(n, n, 'tries')} the pendulum never came within 15° of upright: its biggest swing went ${swing}° up from hanging. ${weak ? `The motor is too weak to lift it straight up (${num(W.torque)} against the ${num(W.g)} the weight pulls with), so it has to swing it, pushing with each swing, higher and higher.` : ''}${why}`,
          [louder, pg ? c.switchTo('ga') : null, pg && height !== null && height >= 0.5 ? c.switchTo('dqn') : null]);
      } else if (fallen.length && mean(eps, 'late') < 0.8) {
        const kept = eps.filter((e) => e.end === 'balanced').length;
        add('warn', 'It swings it up but cannot hold it', `In ${of(fallen.length, n, 'tries')} it gets it upright (first after ${fmtInt(min(fallen, 'first'))} steps) but lets it fall again: upright for only ${fmtPct(mean(fallen, 'late'), 0)} of the last ${fmtInt(W.maxSteps / 2)} steps${kept ? `, where in the other ${kept === 1 ? 'one' : kept} it keeps it up` : ''}. Catching it is a skill of its own: braking as it comes over the top, then small pushes either way to keep it balanced. ${ga ? 'Evolution usually finds it, given more generations.' : faint ? why.trim() : dqn ? 'DQN usually finds it, given more steps.' : 'Evolution and DQN find it more often than PPO and actor-critic do.'}`,
          [ga || (dqn && !faint) ? keepGoing(L) : null, louder, pg ? c.switchTo('ga') : null]);
      }
    },
    arm(c) {
      const { eps, add, d } = c, n = eps.length, lost = eps.filter((e) => e.end !== 'reached'), over = lost.filter((e) => e.fast > 0).length;
      if (!c.settled || !lost.length) return;
      const a2c = c.method === 'a2c', near = `${fmtInt((c.W.reach || 0.12) * 100)} cm`;
      if (over >= 2) add('warn', 'It overshoots', `In ${over} of ${n} tries its hand came within ${near} of the target too fast to count, and went on past. Reaching means stopping there, slower than 0.6 m/s. The joints keep turning once pushed, so it has to brake before it gets there, not when it arrives.${a2c ? '' : ' Actor-critic learns to brake in time here.'}`,
        [a2c ? null : c.switchTo('a2c')]);
      else if (lost.length >= 3 && !c.own && amount(d, 'rApproach') <= 0) add('warn', 'It never finds the targets', `In ${of(lost.length, n, 'tries')} its hand never came within ${near} of the target, and ended ${fmtInt(mean(lost, 'd') * 100)} cm away on average. With no reward for getting closer, it is paid only on arrival, and random moves almost never arrive: there is nothing to learn from.`,
        [c.again('Approach reward on, new run', c.setPs((q) => { q.rApproach = 1; }))]);
      else if (lost.length >= 3) add(a2c || lost.length < 5 ? 'info' : 'warn', 'It misses targets', `In ${of(lost.length, n, 'tries')} its hand never came within ${near} of the target: it ended ${fmtInt(mean(lost, 'd') * 100)} cm away on average, the nearest it got ${fmtInt(mean(lost, 'near') * 100)} cm.${a2c ? '' : ' Actor-critic reached 9 targets of 10 in our runs.'}`,
        [a2c ? null : c.switchTo('a2c')]);
    },
    traffic(c) {
      const { eps, add, d, ga, W } = c, n = eps.length, sw = mean(eps, 'switches'), most = max(eps, 'switches');
      if (!c.settled) return;
      const shifting = d.ws.pattern !== 'steady', steps = W.maxSteps, amber = W.amberFor || 4, usual = amber === 4 && (W.busy || 1) === 1;
      if (most === 0) {
        const road = eps[0].ns > 0.5 ? 'North–south' : 'East–west';
        add('warn', 'It never switches', `In ${of(n, n, 'tests')} of ${fmtInt(steps)} steps the lights never changed: ${road.toLowerCase()} kept green while up to ${fmtInt(max(eps, 'queue'))} cars queued on the other road. Every switch costs ${amber} step${amber === 1 ? '' : 's'} of amber, when no car moves, so at first switching only looks bad, and a learner that has never seen what it buys never tries it twice.${ga ? '' : ' A genetic algorithm, trying whole networks, finds the ones that switch.'}`,
          [ga ? null : c.switchTo('ga')]);
      } else if (sw < steps / 80 && shifting && eps.some((e) => e.end === 'jammed')) add('warn', 'It hardly ever switches', `It switched ${num(sw)} times in ${fmtInt(steps)} steps on average. Halfway through, the rush hour moves from the north–south road to the east–west one, and the lights have to follow it.${ga ? '' : ' A genetic algorithm finds the switching networks more reliably.'}`,
        [ga ? null : c.switchTo('ga')]);
      else if (most >= steps / 10) {
        const busy = eps.filter((e) => e.switches >= steps / 10), k = mean(busy, 'switches');
        add('warn', 'It switches too often', `In ${of(busy.length, n, 'tests')} it switched ${fmtInt(k)} times in ${fmtInt(steps)} steps, every ${fmtInt(steps / k)} steps: each switch is ${amber} step${amber === 1 ? '' : 's'} of amber when no car crosses, so the lights stood at amber ${fmtPct(Math.min(1, (amber * k) / steps), 0)} of the time and the queues grew.${usual ? ' Lights that work well here switch about every 15 to 20 steps.' : ''}`,
          [ga ? keepGoing(c.L) : null]);
      }
    },
    walker(c) {
      const { eps, add, d, ga, W } = c, n = eps.length, fell = eps.filter((e) => e.end === 'toppled'), dist = mean(eps, 'dist'), fall = amount(d, 'rFall'), metre = amount(d, 'rForward');
      if (!c.settled) return;
      const secs = W.maxSteps / 15, span = `${fmtInt(W.maxSteps)} steps (${num(secs)} s)`, pace = 10 / secs;
      if (fell.length) add('warn', 'It topples over', `In ${of(fell.length, n, 'tries')} it fell after ${fmtInt(mean(fell, 't'))} steps (${fmt(mean(fell, 't') / 15, 1)} s), ${fmt(mean(fell, 'dist'), 1)} m along. ${!c.own && 2 * metre + fall > 0 ? `A fall costs ${fmt(-fall, 0)} and each metre pays ${fmt(metre, 0)}, so a lunge that gets a couple of metres before it falls still beats standing still. A dearer fall mostly teaches it to stand still instead. ` : ''}A gait has to keep its body up step after step${ga ? '.' : ': evolution finds one more surely.'}`,
        [ga ? null : c.switchTo('ga')]);
      else if (dist < 1) add('warn', 'It stands still', `${dist < 0.05 ? 'It hardly moved' : `It moved ${fmt(dist, 1)} m`} in ${span}.${!c.own && fall < -1 ? ` A fall costs ${fmt(-fall, 0)} here, and each metre pays ${fmt(metre, 0)}: standing still is the safest thing it has found.` : ''}`,
        [!c.own && fall < -1 ? c.again('A fall costs 1 again, new run', c.setPs((q) => { q.rFall = -1; })) : null, ga ? null : c.switchTo('ga')]);
      else {
        const slow = eps.filter((e) => e.end === 'slow');
        if (slow.length && eps.reduce((sc, e) => sc + Math.max(0, Math.min(1, e.dist / 10)), 0) / n < 0.95) add('warn', 'It walks too slowly', `In ${of(slow.length, n, 'tries')} it walked only ${fmt(mean(slow, 'dist'), 1)} m of the 10 in ${span}: it needs ${pace === 0.5 ? 'half a metre' : `${num(pace)} m`} a second.`, [ga ? keepGoing(c.L) : c.switchTo('ga')]);
      }
    },
    trap(c) {
      const { eps, add, d, s, L } = c, n = eps.length, stuck = eps.filter((e) => e.end !== 'goal' && e.behind), pressed = stuck.filter((e) => e.pressed);
      if (!c.settled || !stuck.length) return;
      const M = NA.brl.METHODS, deceived = !c.own && amount(d, 'rStraight') > 0, explorer = s.method === 'me' || s.method === 'novelty', against = pressed.length >= stuck.length / 2;
      add('warn', against ? 'Stuck against the wall' : 'It never gets round the wall', `From ${of(stuck.length, n, 'starts')} it ended ${against ? `pressed against a wall, ${fmtInt(mean(pressed, 'd'))} units from the goal in a straight line, with the goal on the other side (it walked into the wall ${fmtInt(mean(pressed, 'bumps'))} times)` : `behind a wall from the goal, ${fmtInt(mean(stuck, 'd'))} units from it in a straight line`}. ${deceived ? `Its reward pays for getting closer in a straight line, and the way round leads away from the goal first: every step of it looks bad, so the reward itself ${against ? 'holds it at the wall' : 'keeps it on the wrong side of the wall'}. ` : ''}${explorer ? `${M[s.method].short} looks past the reward, for places no network has ended yet; one way round is enough, and more training finds more of the maze.` : 'Methods that climb the reward almost always stop there. Novelty search, which pays for ending up somewhere new, and MAP-Elites, which keeps the best network for every place a run ends, look past it.'}`,
        explorer ? [keepGoing(L), newRun(L)] : [M.novelty ? c.switchTo('novelty') : null, c.switchTo('me')]);
    },
    chase(c) {
      const { eps, add, d, ga, L } = c, n = eps.length, caught = eps.filter((e) => e.end === 'tagged'), edge = caught.filter((e) => e.edge < 1.5), corner = edge.filter((e) => e.corner);
      if (!c.settled || !caught.length || edge.length < caught.length / 2) return;
      const ahead = d.ws.aim === 'ahead';
      add('warn', corner.length >= edge.length / 2 ? 'Caught in a corner' : 'Caught against the wall', `In ${of(caught.length, n, 'games')} the chaser caught it after about ${fmtInt(mean(caught, 't'))} steps of the ${fmtInt(c.W.maxSteps)}, ${corner.length >= edge.length / 2 ? 'in a corner' : 'against the wall'}. It runs straight away from the chaser, which works until a wall stops it. ${ahead ? `This chaser heads for where it is going, so it cuts off a straight run, and a wide circle too: only some runs of evolution find a way to keep clear of it.${ga ? ' Train it on, or start a new run.' : ' Gradient methods learn to run away, and stop there.'}` : `It is faster than the chaser, so running round it, through the open middle of the arena, keeps it free for good. ${ga ? 'Most runs of evolution find that; if this one has not, train it on or start a new run.' : 'Evolution finds that; gradient methods learn to run away, and stop there.'}`}`,
        ga ? [keepGoing(L), newRun(L)] : [c.switchTo('ga')]);
    },
    herd(c) {
      const { eps, add, d, ga, L } = c, n = eps.length, N = eps[0].n, loose = eps.filter((e) => e.end === 'loose'), avg = mean(eps, 'penned');
      if (!c.settled || !loose.length) return;
      const nearer = amount(d, 'rNearer'), unpaid = !c.own && nearer <= 0;
      add('warn', avg === 0 ? 'No sheep penned' : 'Sheep left loose', `${avg === 0 ? `It penned none of the ${N} sheep in its ${n} games of ${fmtInt(c.W.maxSteps)} steps` : `It penned ${String(+avg.toFixed(1))} of the ${N} sheep on average, and ${of(loose.length, n, 'games')} ended with sheep still out after ${fmtInt(c.W.maxSteps)} steps`}. ${unpaid ? 'With no bonus for bringing sheep nearer the gate, nothing pays until a sheep is in the pen, and a dog that has not learned yet never gets one in: there is nothing to learn from. ' : ''}Sheep run from the dog, so it has to come round behind them, on the side away from the gate, and push; chased from the wrong side, they run away from the pen, and a straggler has to be fetched back to the flock.`,
        unpaid ? [c.again('Pay for sheep nearer the gate, new run', c.setPs((q) => { q.rNearer = 0.1; })), ga ? null : c.switchTo('ga')] : [ga ? keepGoing(L) : c.switchTo('ga')]);
    },
  };

  // how to read the results picture, for each kind of problem
  function read(d, live) {
    if (!d || d.error) return [];
    const fam = d.family, loss = live ? 'Below it: the loss over the run (log scale). The training loss is measured on examples the model learns from, the validation loss on examples held back. Both falling means it is learning; validation rising while training keeps falling means it is memorising.' : null;
    const P = {
      plane: ['Every dot is one example at its two input numbers, coloured by its class: filled dots train the model, rings are held back to test it.', live ? 'The background is the model’s answer at every point of the square: the colour of the class it picks, stronger where it is surer. Where two colours meet is the decision boundary.' : 'Once trained, the background shows the model’s answer at every point, and where the colours meet is its decision boundary.', 'Look for a boundary that winds around single dots (memorising noise) or one too straight to follow the classes (too simple).'],
      curve: ['Each dot is an example: the input x across, the target y up. Filled dots train the model, rings test it.', live ? 'The line is the function the model has learned, drawn at every x, including between and beyond the examples.' : 'Once trained, a line shows the function the model has learned.', 'Look for wiggles between the dots (memorising the noise) or a line too stiff to follow them, and see what it does past the last example: models rarely extrapolate well.'],
      surface: ['The surface seen from above: blue is high, orange low. Left: the true surface, with the training points.', live ? 'Middle: what the model has learned. Right: its error, darker where it misses more.' : 'Once trained, the model’s own surface and its error appear beside it.', 'Look at where the error concentrates: sharp peaks and fine ripples are the hardest parts for a network, which learns smooth shapes first.'],
      series: ['The series runs from left to right. The model sees a window of the last values and predicts the next one; the line marks where the training part ends and the held-back future begins.', live ? 'One step ahead, the true values are fed in each time; on its own, the model feeds its own predictions back, so its errors pile up. That second test is the honest one for a forecaster.' : 'Once trained, its forecasts are drawn over the future part.', 'Look at how long the free-running forecast stays on track before it drifts.'],
      digits: ['Pictures the model never trained on, each with its answer: green read right, red read wrong.', 'With a confusion matrix: rows are the true classes, columns the model’s answers. Everything on the diagonal is right; a bright square off it is a pair of classes the model mixes up.', 'Look at the mistakes: are they ones a person could make too?'],
      seqclass: ['Validation sequences with their true class, and once trained the model’s answer: ✓ right, ✗ wrong.', 'Look at which ones it gets wrong: is it the word order, a long distance between the words that matter, or one token that flips the answer?'],
      sequence: ['The inputs and, once trained, the model’s output at every position. With attention, the map shows which positions each one looks at: a bright square means a position takes most of its information from that one.', 'Look for the pattern the task needs: reversing looks at the mirror position, copying at the same one.'],
      table: d.kind === 'classes' ? ['The rows of your table on their first two columns, scaled to mean 0 and spread 1, coloured by class.', 'Once trained, a confusion matrix: rows are the true classes, columns the model’s answers; the diagonal is right.'] : ['The rows of your table: the true value across and, once trained, the model’s answer up. On the diagonal is right; the further a point sits from it, the bigger the miss.'],
    };
    if (fam === 'control') return ['The world, played live: with its best moves once it has trained, at random before. A new episode starts when one ends; the line above says how the last one went.', 'Beside it, its outputs for what it sees right now, and in colour the move they pick: Q-values for DQN (what it expects to collect), probabilities for policy methods, plain scores for a genetic algorithm.', live ? 'Below: the return of its episodes while it learns, averaged over the latest. It counts its exploring moves too, so it can trail the score, which judges its best moves alone.' : null].filter(Boolean);
    return [...(P[d.table ? 'table' : fam] || []), loss].filter(Boolean);
  }

  NA.builderInsights = { diagnose, read };
})(window.NA);
