/* insights.js — automatic diagnostics: reads the network's statistics and training history
   and explains what is going on, with one-click experiments. */
(function (NA) {
  'use strict';
  const { fmtSci: fmt, fmtPct } = NA; // fmtSci: tiny values as 4.1×10⁻⁴, not 4.1e−4
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
    const net = App.net, ds = App.ds, L = net.L, h = App.hyper, E = NA.engine;
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
    // activation choices across the hidden neurons
    const hiddenLayers = NA.range(L - 1).map((i) => i + 1);
    const linearLayers = hiddenLayers.filter((l) => net.layerActId(l) === 'linear');
    if (L > 1 && linearLayers.length === hiddenLayers.length) add('warn', 'Linear layers collapse', 'Every hidden neuron is linear, and stacking linear layers still produces a single linear function, so the decision boundary stays straight however many layers you add.', [{ label: 'Use ReLU', patch: { act: 'relu', train: 1500 } }]);
    else if (linearLayers.length) add('info', `${linearLayers.map((l) => App.layerName(l)).join(' and ')} ${linearLayers.length > 1 ? 'are' : 'is'} linear`, 'A linear layer adds no bends of its own; it only re-mixes the features of the layer before it.');
    const mixed = hiddenLayers.filter((l) => net.layerActId(l) === 'mixed');
    if (mixed.length) {
      const l = mixed[0], count = {};
      for (const id of net.actIds[l]) count[id] = (count[id] || 0) + 1;
      const parts = Object.keys(count).map((id) => `${E.ACTIVATIONS[id].label} × ${count[id]}`).join(', ');
      add('info', `Mixed activations in ${App.layerName(l)}${mixed.length > 1 ? ` and ${plural(mixed.length - 1, 'other layer')}` : ''}`, `${parts}. Each neuron bends its input with its own function; the small curve on each node shows which. Different shapes in one layer let it build features no single function would.`);
    }

    if (step > 0 && last) {
      // quality of fit
      if (cls && last.valAcc >= 0.97 && last.trainAcc - last.valAcc < 0.03) add('good', 'Generalizes well', `Validation accuracy is ${fmtPct(last.valAcc, 1)} and close to training accuracy (${fmtPct(last.trainAcc, 1)}), so the network learned the underlying shape rather than individual points.`);
      if (!cls && last.valAcc >= 0.95) add('good', 'Close fit', `R² on validation points is ${fmt(last.valAcc, 3)}: the network explains ${fmtPct(last.valAcc, 1)} of the variance in points it never trained on.`);
      // overfitting
      if (pts.length > 12) {
        let best = pts[0];
        for (const p of pts) if (p.step <= step && p.valLoss < best.valLoss) best = p;
        if (last.valLoss > best.valLoss * 1.25 && last.valLoss - last.trainLoss > 0.08 && last.trainLoss < best.valLoss) {
          const snapIdx = nearestSnap(best.step);
          add('warn', 'Overfitting', `Validation loss bottomed out at ${fmt(best.valLoss, 3)} around step ${NA.fmtInt(best.step)} and has since risen to ${fmt(last.valLoss, 3)}, while training loss kept falling to ${fmt(last.trainLoss, 3)}. The network is memorising the training points.`,
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
      // squared error on a probability output: slow when confidently wrong
      if (cls && net.lossId === 'mse' && last.trainAcc < 0.97 && step > 200) add('info', 'Squared error on a classifier', `With squared error the error signal is (p − y)·p(1 − p), which shrinks toward 0 when the output is confidently wrong (p near 0 or 1). Cross-entropy drops the p(1 − p) factor, so the worst mistakes get the strongest push.`,
        [{ label: 'Use cross-entropy', run: () => App.setLoss('xent') }]);
      // the held-out test, once run
      const tr = App.testResult;
      if (tr) add('info', tr.version === App.version ? 'Held-out test result' : 'Held-out test is out of date',
        `${cls ? `Test accuracy ${fmtPct(tr.acc, 1)} against ${fmtPct(tr.valAcc, 1)} on validation` : `Test R² ${fmt(tr.acc, 3)} against ${fmt(tr.valAcc, 3)} on validation`}, measured at step ${NA.fmtInt(tr.step)}.${tr.version === App.version ? '' : ' The model has changed since; run the test again for a current number.'}`,
        [{ label: 'Open the report', run: () => App.emit('tab', 'training') }]);
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
        // each neuron is judged by its own activation function
        const dead = [], flat = [], sat = [];
        for (let j = 0; j < net.sizes[l]; j++) {
          if (!net.mask[l][j]) continue;
          const kind = E.deadKind(net, stats, l, j), a = net.actFns[l][j];
          if (kind === 'dead') dead.push({ l, j });
          else if (kind === 'flat') flat.push({ l, j });
          else if (a.sat && stats[l][j].satFrac > 0.9) sat.push({ l, j });
        }
        if (dead.length && step > 0) add('warn', `${plural(dead.length, 'dead neuron')} in Hidden ${l}`, `${names(dead)} never switch on for any training input. ReLU outputs 0 there and its slope is 0 too, so no gradient reaches them and they cannot recover. Leaky ReLU keeps a small slope below zero, so a neuron can come back.`,
          [{ label: 'Show them', highlight: dead }, { label: 'Make them Leaky ReLU', run: () => App.setActs(dead, 'leaky') }]);
        if (flat.length && step > 0) add('warn', `${plural(flat.length, 'flat neuron')} in Hidden ${l}`, `${names(flat)} give almost the same output for every training input, so they carry no information about the input and pass almost no gradient back.`, [{ label: 'Show them', highlight: flat }]);
        if (sat.length) {
          const fns = [...new Set(sat.map((n) => net.actFns[n.l][n.j].label))].join(' or ');
          add('warn', `${plural(sat.length, 'saturated neuron')} in Hidden ${l}`, `${names(sat)} sit on the flat tails of ${fns} for more than 90% of inputs. The slope there is close to 0, so very little gradient passes through.`,
            [{ label: 'Show them', highlight: sat }, { label: 'Make them ReLU', run: () => App.setActs(sat, 'relu') }]);
        }
      }
    }

    // gradient flow
    if (L >= 3 && step > 0) {
      const g = App.gradReport().layerRms, first = g[0], lastG = g[L - 1];
      if (first > 0 && lastG / first > 50) {
        // which squashing functions are in the way, if any
        const squash = new Set();
        for (let l = 1; l < L; l++) for (const a of net.actFns[l]) if (a.family === 'squash') squash.add(a.id);
        const why = squash.size ? `Every ${[...squash].map((id) => E.ACTIVATIONS[id].label).join(' or ')} neuron multiplies the error signal by its slope, at most ${squash.has('sigmoid') ? '0.25 for sigmoid' : '1'}${squash.has('sigmoid') && squash.has('tanh') ? ' and 1 for tanh' : ''}, and those factors compound layer after layer.` : '';
        add(lastG / first > 1000 ? 'critical' : 'warn', 'Vanishing gradients', `The gradient reaching the first layer is ${fmtRatio(lastG / first)} smaller than at the last layer, so the early layers barely change. ${why}`,
          [{ label: 'Show gradients', run: () => { App.setLens('edges', 'gradient'); App.setLens('nodes', 'gradient'); } }].concat(squash.size ? [{ label: 'Use ReLU', patch: { act: 'relu', train: Math.max(800, step) } }] : []).concat(h.optimizer !== 'adam' ? [{ label: 'Use Adam', patch: { optimizer: 'adam', lr: 0.01 } }] : []));
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
      if (best && best.r.dLoss > 0.01) add('info', `Most important neuron: ${App.neuronName(best.l, best.j, true)}`, `Knocking it out raises validation loss by ${fmt(best.r.dLoss, 3)}${cls ? ` and changes accuracy by ${fmtPct(best.r.dAcc, 1)}` : ''}. Select it to see what it computes.`, [{ label: 'Select it', run: () => App.select({ l: best.l, j: best.j }, { fromGraph: true }) }]);
      if (idle.length >= 2) add('info', `${plural(idle.length, 'neuron')} barely matter`, `Removing any one of ${names(idle)} changes validation loss by less than 0.002. Not every neuron ends up with a distinct job.`, [{ label: 'Show them', highlight: idle }]);
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
    if (a.highlight) {
      // a phone shows one screen at a time: the pulse plays on the diagram, so go there first
      if (NA.mobile && NA.mobile.mobile && NA.mobile.screen !== 'network') NA.mobile.go('network');
      App.highlight = { list: a.highlight, until: performance.now() + 3600 }; App.emit('highlight');
    }
  }

  NA.insights = { diagnose, runAction };
})(window.NA);
