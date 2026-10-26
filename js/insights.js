/* insights.js — automatic diagnostics: reads the network's statistics and training history
   and explains what is going on, with one-click experiments. */
(function (NA) {
  'use strict';
  const { fmt, fmtPct } = NA;
  const App = NA.app;
  const ORDER = { critical: 0, warn: 1, info: 2, good: 3 };

  function names(list) {
    const s = list.slice(0, 4).map((n) => App.neuronName(n.l, n.j, true)).join(', ');
    return list.length > 4 ? `${s} and ${list.length - 4} more` : s;
  }
  const plural = (n, w) => `${n} ${w}${n === 1 ? '' : 's'}`;

  function diagnose() {
    const out = [];
    const add = (level, title, body, actions = []) => out.push({ level, title, body, actions });
    const net = App.net, ds = App.ds, L = net.L, act = net.act, h = App.hyper;
    const pts = App.history.points, last = App.metrics(), step = App.displayStep();
    const cls = ds.task !== 'regression';
    const accName = cls ? 'accuracy' : 'R²';

    if (App.diverged) {
      add('critical', 'Training diverged', `The loss became infinite at step ${NA.fmtInt(App.step)}. With a learning rate of ${h.lr}, each update overshoots, the weights grow without bound and the arithmetic overflows.`,
        [{ label: `Learning rate ${+(h.lr / 10).toPrecision(2)} and reset`, patch: { lr: +(h.lr / 10).toPrecision(2), reset: true } }]);
    }
    if (step === 0) add('info', 'Untrained network', 'The weights are still random, so the output is noise. Press Train to run gradient descent, or Step to apply a single update and watch what changes.');
    if (App.viewing !== null) add('info', `Viewing step ${NA.fmtInt(step)} from history`, 'Everything on screen shows the network as it was at this step. Training again continues from here and discards the later history.', [{ label: 'Back to latest', run: () => App.backToLive() }]);
    const nAbl = App.ablatedCount();
    if (nAbl) add('info', `${plural(nAbl, 'neuron')} ablated`, 'Ablated neurons are forced to output 0. Compare accuracy with and without them to see what they contributed.', [{ label: 'Restore all', run: () => App.restoreAll() }]);
    if (act.id === 'linear' && L > 1) add('warn', 'Linear layers collapse', 'Stacking linear layers still produces a single linear function, so the decision boundary stays straight however many layers you add.', [{ label: 'Use ReLU', patch: { act: 'relu', train: 1500 } }]);

    if (step > 0 && last) {
      // quality of fit
      if (cls && last.testAcc >= 0.97 && last.trainAcc - last.testAcc < 0.03) add('good', 'Generalizes well', `Test accuracy is ${fmtPct(last.testAcc, 1)} and close to training accuracy (${fmtPct(last.trainAcc, 1)}), so the network learned the underlying shape rather than individual points.`);
      if (!cls && last.testAcc >= 0.95) add('good', 'Close fit', `R² on held-out points is ${fmt(last.testAcc, 3)}: the network explains ${fmtPct(last.testAcc, 1)} of the variance it has not seen.`);
      // overfitting
      if (pts.length > 12) {
        let best = pts[0];
        for (const p of pts) if (p.step <= step && p.testLoss < best.testLoss) best = p;
        if (last.testLoss > best.testLoss * 1.25 && last.testLoss - last.trainLoss > 0.08 && last.trainLoss < best.testLoss) {
          const snapIdx = nearestSnap(best.step);
          add('warn', 'Overfitting', `Test loss bottomed out at ${fmt(best.testLoss, 3)} around step ${NA.fmtInt(best.step)} and has since risen to ${fmt(last.testLoss, 3)}, while training loss kept falling to ${fmt(last.trainLoss, 3)}. The network is memorising the training points.`,
            [{ label: 'Add L2 0.01 and retrain', patch: { reg: 'l2', regRate: 0.01, reset: true, train: Math.max(500, step) } }, { label: `View step ${NA.fmtInt(App.history.snaps[snapIdx].step)}`, run: () => App.viewSnapshot(snapIdx) }]);
        }
      }
      // stalled training
      if (pts.length > 20 && step > 300) {
        const old = pts[Math.floor(pts.length * 0.6)];
        const rel = (old.trainLoss - last.trainLoss) / Math.max(1e-9, old.trainLoss);
        const poor = cls ? last.trainAcc < 0.85 : last.trainAcc < 0.8;
        if (poor && rel < 0.03 && !App.training) {
          let why = 'The network may lack the capacity for this shape, or be stuck in a poor region.';
          if (L === 1) why = 'With no hidden layer the network can only draw a straight line (or a flat plane), and this data needs a curved boundary.';
          else if (L >= 3) { const g = App.gradReport().layerRms; if (g[0] > 0 && g[L - 1] / g[0] > 50) why = 'The vanishing gradients reported here are the likely cause: the early layers are not receiving enough signal to learn.'; }
          add('warn', 'Learning has stalled', `Training loss changed by less than 3% over the last ${NA.fmtInt(step - old.step)} steps, with ${accName} at ${cls ? fmtPct(last.trainAcc, 1) : fmt(last.trainAcc, 2)}. ${why}`,
            L === 1 ? [{ label: 'Add a hidden layer', patch: { hidden: [6], act: 'relu', train: 1500 } }] : []);
        }
      }
      // oscillation
      if (pts.length > 16 && App.history.points.length) {
        const tail = pts.slice(-14);
        let ups = 0;
        for (let i = 1; i < tail.length; i++) if (tail[i].trainLoss > tail[i - 1].trainLoss * 1.05) ups++;
        if (ups >= 5 && h.lr >= 0.03) add('warn', 'Loss is bouncing', `In the last ${tail.length} checkpoints the training loss rose ${ups} times. Updates of size ${h.lr} overshoot the minimum and jump between its walls.`, [{ label: `Learning rate ${+(h.lr / 3).toPrecision(2)}`, patch: { lr: +(h.lr / 3).toPrecision(2) } }]);
      }
    }

    // dead and saturated neurons
    if (L > 1) {
      const stats = App.stats();
      for (let l = 1; l < L; l++) {
        const dead = [], sat = [];
        for (let j = 0; j < net.sizes[l]; j++) {
          if (!net.mask[l][j]) continue;
          const s = stats[l][j];
          if (act.family === 'rectifier' ? s.activeFrac === 0 : s.std < 1e-4) dead.push({ l, j });
          else if (s.satFrac > 0.9) sat.push({ l, j });
        }
        if (dead.length && step > 0 && act.family === 'rectifier') add('warn', `${plural(dead.length, 'dead neuron')} in Hidden ${l}`, `${names(dead)} never switch on for any training input. Their output is always 0, so their gradient is 0 too and they cannot recover. ${act.id === 'relu' ? 'Leaky ReLU keeps a small slope below zero, so neurons can come back.' : ''}`,
          [{ label: 'Show them', highlight: dead }].concat(act.id === 'relu' ? [{ label: 'Use Leaky ReLU', patch: { act: 'leaky', train: Math.max(500, step) } }] : []));
        else if (dead.length && step > 0) add('warn', `${plural(dead.length, 'flat neuron')} in Hidden ${l}`, `${names(dead)} give almost the same output for every training input, so they carry no information about the input and pass almost no gradient back.`, [{ label: 'Show them', highlight: dead }]);
        if (sat.length) add('warn', `${plural(sat.length, 'saturated neuron')} in Hidden ${l}`, `${names(sat)} sit on the flat tails of ${act.label} for more than 90% of inputs. The slope there is close to 0, so very little gradient passes through.`,
          [{ label: 'Show them', highlight: sat }, { label: 'Use ReLU', patch: { act: 'relu', train: Math.max(500, step) } }]);
      }
    }

    // gradient flow
    if (L >= 3 && step > 0) {
      const g = App.gradReport().layerRms, first = g[0], lastG = g[L - 1];
      if (first > 0 && lastG / first > 50) {
        add(lastG / first > 1000 ? 'critical' : 'warn', 'Vanishing gradients', `The gradient reaching the first layer is ${fmtRatio(lastG / first)} smaller than at the last layer, so the early layers barely change. ${act.family === 'squash' ? `Each ${act.label} layer multiplies the error signal by its slope, at most ${act.id === 'sigmoid' ? '0.25' : '1'}.` : ''}`,
          [{ label: 'Show gradients', run: () => { App.setLens('edges', 'gradient'); App.setLens('nodes', 'gradient'); } }].concat(act.family === 'squash' ? [{ label: 'Use ReLU', patch: { act: 'relu', train: Math.max(800, step) } }] : []).concat(h.optimizer !== 'adam' ? [{ label: 'Use Adam', patch: { optimizer: 'adam', lr: 0.01 } }] : []));
      } else if (lastG > 0 && first / lastG > 50) {
        add('warn', 'Exploding gradients', `The first layer receives gradients ${fmtRatio(first / lastG)} larger than the last. Updates to early weights can become erratic.`, [{ label: 'Lower learning rate', patch: { lr: +(h.lr / 3).toPrecision(2) } }]);
      }
    }
    if (last && step > 0 && last.ratio && last.ratio.length) {
      const big = Math.max(...last.ratio);
      if (big > 0.05) add('warn', 'Large weight updates', `At the last checkpoint, one layer’s weights moved by ${fmtPct(big)} of their size in a single step. A healthy ratio is roughly 0.1%–1%.`, [{ label: 'Lower learning rate', patch: { lr: +(h.lr / 3).toPrecision(2) } }]);
    }

    // redundancy and importance (only when idle: these need extra passes over the data)
    if (!App.training && step > 0 && L > 1) {
      const F = App.dataPass(), stats = App.stats();
      for (let l = 1; l < L; l++) {
        const m = net.sizes[l];
        if (m > 24) continue;
        const pairs = [];
        for (let a = 0; a < m; a++) for (let b = a + 1; b < m; b++) {
          if (!net.mask[l][a] || !net.mask[l][b] || stats[l][a].std < 1e-3 || stats[l][b].std < 1e-3) continue;
          const c = NA.engine.correlation(F.a[l], F.n, a, b);
          if (Math.abs(c) > 0.985) pairs.push([a, b, c]);
        }
        if (pairs.length) {
          const [a, b, c] = pairs[0];
          add('info', `Redundant neurons in Hidden ${l}`, `${App.neuronName(l, a, true)} and ${App.neuronName(l, b, true)} are ${c > 0 ? '' : 'negatively '}correlated at ${fmt(Math.abs(c), 3)}: they compute nearly the same feature${pairs.length > 1 ? `, and ${plural(pairs.length - 1, 'other pair')} do too` : ''}. The layer could probably be smaller.`, [{ label: 'Show them', highlight: [{ l, j: a }, { l, j: b }] }]);
        }
      }
      const imp = App.importance();
      let best = null, idle = [];
      for (let l = 1; l < L; l++) for (let j = 0; j < net.sizes[l]; j++) {
        const r = imp.layers[l][j];
        if (r.off) continue;
        if (!best || r.dLoss > best.r.dLoss) best = { l, j, r };
        if (Math.abs(r.dLoss) < 0.002 && Math.abs(r.dAcc) < 0.005) idle.push({ l, j });
      }
      if (best && best.r.dLoss > 0.01) add('info', `Most important neuron: ${App.neuronName(best.l, best.j, true)}`, `Knocking it out raises test loss by ${fmt(best.r.dLoss, 3)}${cls ? ` and changes accuracy by ${fmtPct(best.r.dAcc, 1)}` : ''}. Select it to see what it computes.`, [{ label: 'Select it', run: () => App.select({ l: best.l, j: best.j }, { fromGraph: true }) }]);
      if (idle.length >= 2) add('info', `${plural(idle.length, 'neuron')} barely matter`, `Removing any one of ${names(idle)} changes test loss by less than 0.002. Not every neuron ends up with a distinct job.`, [{ label: 'Show them', highlight: idle }]);
    }
    out.sort((a, b) => ORDER[a.level] - ORDER[b.level]);
    return out;
  }

  function fmtRatio(r) { return r >= 1000 ? `${NA.fmtCompact(r)}×` : `${Math.round(r)}×`; }
  function nearestSnap(step) {
    const s = App.history.snaps;
    let bi = 0;
    for (let i = 0; i < s.length; i++) if (Math.abs(s[i].step - step) < Math.abs(s[bi].step - step)) bi = i;
    return bi;
  }

  function runAction(a) {
    if (a.patch) App.applyPatch(a.patch);
    if (a.run) a.run();
    if (a.highlight) { App.highlight = { list: a.highlight, until: performance.now() + 3600 }; App.emit('highlight'); }
  }

  NA.insights = { diagnose, runAction };
})(window.NA);
