/* builder-insights.js — what is going on in the model builder, in plain words. diagnose() reads the current run (its
   training and validation curves, the gradient sizes, the model's settings) and says whether the model is learning,
   has learned, is stuck, has blown up, memorises its examples (overfitting) or is too simple (underfitting), with the
   numbers behind it and one-click changes to try; for learning from rewards, whether it ever finds the reward,
   explores, looks ahead, or has got worse again. read() explains how to read the results picture for each kind of
   problem. Both are shown in the builder's results and its Training tab. */
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
    const run = L.run, s = L.ps(), M = NA.brl.METHODS[s.method], ga = s.method === 'ga', unit = ga ? 'generation' : 'step';
    const again = (label, fn) => ({ label, run: () => { L.edit(fn); L.reset(); if (!L.training) L.play(); } }), setPs = (fn) => (S) => fn(S.ps[S.problem]);
    if (!run) { add('info', 'Not trained yet', `Press Train: the network plays in the world, and ${ga ? 'each generation the best players are bred and the rest replaced' : `${M.short} turns what happens into steps on its weights`}. The picture plays it live, with its best moves.`); return; }
    if (L.stale) { add('info', 'The setup has changed since the last run', 'The pictures show the last run. Train starts a new run with the setup as it is now; the Runs tab keeps the earlier runs to compare.'); return; }
    if (run.status === 'diverged') { add('critical', 'The loss blew up', `At step ${fmtInt(run.step)} the weights ran off to infinity. Learning from rewards is touchy: a smaller learning rate, or clipping, usually cures it.`, [again(`Learning rate ${+(L.setup.train.lr / 10).toPrecision(2)}, new run`, (S) => { S.train.lr = +(S.train.lr / 10).toPrecision(2); })]); return; }
    const c = run.curve.filter((p) => p.vm !== null && p.vm !== undefined), n = c.length, vm = run.last ? run.last.vm : null, best = run.bestVm, ret = run.last ? run.last.ret : null;
    if (n < 2) { add('info', 'Learning', `${ga ? 'Generation' : 'Step'} ${fmtInt(run.step)}. The first scores are on their way.`); return; }
    const low = vm !== null && vm < 0.9, w = d.world;
    if (w === 'maze' && !s.closer && !best && run.step >= (ga ? 50 : 1000)) {
      add('warn', 'It never finds the goal', `Its best moves have not reached the goal once. The only rewards come at the very end, and random moves almost never get there, so there is next to nothing to learn from: every episode looks equally bad. A bonus for each step closer to the goal (reward shaping) says which way is warmer from the first move.`,
        [again('Closer bonus on, new run', setPs((q) => { q.closer = true; }))]);
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
      add('warn', 'It sees only the latest reading', `The Sequence pool keeps the last reading and drops the rest, so the model gets a position and an angle but cannot tell which way they are moving, and a pole at the same angle can be falling or rising. Something that reads all 4 readings can work the speeds out: a Flatten, or a recurrent layer.`,
        [again('Flatten instead, new run', (S) => { S.blocks[0] = { id: S.blocks[0].id, type: 'flatten' }; })]);
    }
    if (ga && L.check && L.check.params > 1500 && low && run.step >= 100) {
      add('warn', 'A lot of weights to evolve', `A genetic algorithm searches blindly, by mutating weights and keeping what plays better. With ${fmtInt(L.check.params)} weights most mutations are noise, and progress is slow. A smaller network gives each mutation more say.`);
    }
    if (best !== null && best !== undefined && vm !== null && best - vm >= 0.3 && n >= 6) {
      add('warn', 'It got worse again', `Its best moves scored ${fmtPct(best, 0)} at ${unit} ${fmtInt(run.bestAt)} and ${fmtPct(vm, 0)} now. Learning from rewards is unstable: the network makes its own data by playing, so a change for the worse also changes what it learns from next. A smaller learning rate steadies it${s.method === 'pg' || s.method === 'a2c' ? '; PPO is built to keep each change small' : ''}.`,
        [ga ? null : again(`Learning rate ${+(L.setup.train.lr / 3).toPrecision(2)}, new run`, (S) => { S.train.lr = +(S.train.lr / 3).toPrecision(2); }), s.method === 'pg' || s.method === 'a2c' ? { label: 'Switch to PPO', run: () => L.setMethod('ppo') } : null]);
    }
    if (!out.some((q) => q.level === 'warn' || q.level === 'critical') && vm !== null && vm >= 0.95) {
      add('good', 'Learned it', `Its best moves score ${fmtPct(vm, 0)}${d.ws.random ? ' on mazes it has never seen' : ''}. ${ga ? 'Evolution found weights that play well without a single gradient.' : 'From nothing but rewards, it has found what to do.'} Try another method on the same world, or another way of seeing it, and compare the runs.`);
    }
    if (!out.length) add('info', run.done ? 'Finished' : 'Learning', `${run.done ? `After ${fmtInt(run.step)} ${unit}s` : `At ${unit} ${fmtInt(run.step)}`}: its best moves score ${fmtPct(vm, 0)}${best > vm ? ` (best so far ${fmtPct(best, 0)})` : ''}${ret !== null && ret !== undefined ? `, and its episodes return ${fmt(ret, 2)} on average` : ''}.${run.done && low ? ' Press Continue to train it longer.' : ''}`);
  }

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
