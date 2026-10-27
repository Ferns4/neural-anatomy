/* walkthrough.js — one training step, phase by phase: forward pass, loss, backpropagation, update.
   Drives the network view through App.walk while the Backprop tab is open. */
(function (NA) {
  'use strict';
  const { fmtSci: fmt, fmtPct, el, clamp } = NA; // fmtSci: tiny values as 4.1×10⁻⁴, not 4.1e−4
  const App = NA.app;
  const E = NA.engine;

  const WK = { active: false, phase: 0, sample: null, eta: 0.1, auto: null, history: [] };

  // ------------------------------------------------------------------ lifecycle
  WK.enter = function () {
    if (App.training) App.pause();
    WK.active = true;
    const p = App.pinned;
    if (WK.sample === null || WK.sample >= App.ds.n) WK.sample = p && p.sample !== null ? p.sample : App.ds.trainIdx[0];
    WK.recompute();
  };
  WK.exit = function () {
    WK.active = false; stopAuto();
    if (App.walk) { App.walk = null; App.emit('walk'); }
  };
  App.on('model', () => { WK.sample = null; WK.phase = 0; WK.history = []; if (WK.active) WK.enter(); });
  App.on('weights', () => { if (WK.active && !WK._applying) { WK.recompute(true); } });
  App.on('training', () => { if (WK.active && App.training) stopAuto(); });

  WK.recompute = function (keepPhase) {
    const ds = App.ds, net = App.net, k = WK.sample, d = ds.dim;
    const tr = net.trace(ds.X, k * d, ds.y[k]);
    WK.trace = tr;
    WK.nPhases = 2 * net.L + 3;
    if (!keepPhase) WK.phase = clamp(WK.phase, 0, WK.nPhases - 1);
    WK.phase = clamp(WK.phase, 0, WK.nPhases - 1);
    publish();
  };

  // Phase layout: 0 input · 1..L forward · L+1 loss · L+2 output δ · L+3..2L+1 hidden δ (L-1 → 1) · 2L+2 update
  function phaseInfo(ph) {
    const L = App.net.L;
    if (ph === 0) return { type: 'input', layer: 0 };
    if (ph <= L) return { type: 'forward', layer: ph };
    if (ph === L + 1) return { type: 'loss', layer: L };
    if (ph === L + 2) return { type: 'outdelta', layer: L };
    if (ph <= 2 * L + 1) return { type: 'backprop', layer: L - (ph - (L + 2)) };
    return { type: 'update', layer: null };
  }

  function phaseTitle(ph) {
    const info = phaseInfo(ph);
    if (info.type === 'input') return 'The input';
    if (info.type === 'forward') return `Forward through ${App.layerName(info.layer)}`;
    if (info.type === 'loss') return 'Prediction and loss';
    if (info.type === 'outdelta') return 'Error signal at the output';
    if (info.type === 'backprop') return `Backpropagate into ${App.layerName(info.layer)}`;
    return 'Update the weights';
  }
  // used by the phone layout's walkthrough bar
  WK.summary = () => ({ phase: WK.phase, total: WK.nPhases, title: phaseTitle(WK.phase) });
  WK.step = (d) => go(WK.phase + d);

  function publish() {
    const L = App.net.L, info = phaseInfo(WK.phase), tr = WK.trace;
    const fwdUpTo = info.type === 'input' ? 0 : info.type === 'forward' ? info.layer : L;
    const bwdFrom = info.type === 'outdelta' ? L : info.type === 'backprop' ? info.layer : info.type === 'update' ? 1 : null;
    const deltaMax = tr.delta.map((d) => NA.maxAbs(d));
    App.walk = {
      active: true, trace: tr, focusLayer: info.layer, deltaMax,
      edgeMode(l) {
        if (bwdFrom !== null) return l + 1 >= bwdFrom ? 'walkgrad' : 'off';
        return l + 1 <= fwdUpTo ? 'signal' : 'off';
      },
      nodeMode(l) {
        if (bwdFrom !== null && l >= bwdFrom && l >= 1 && info.type !== 'update') return 'walkdelta';
        return l <= fwdUpTo ? 'activation' : 'pending';
      },
      valueMode(l) { return this.nodeMode(l); },
      showDelta(l) { return bwdFrom !== null && l >= bwdFrom && l >= 1; },
    };
    App.emit('walk');
    if (WK.root) WK.render();
  }

  // ------------------------------------------------------------------ UI
  WK.build = function (root) {
    WK.root = root;
    const ds = App.ds;
    WK.sampleEl = el('div', { class: 'wk-sample' });
    WK.progress = el('div', { class: 'wk-progress', role: 'tablist', 'aria-label': 'Phases of one training step' });
    WK.titleEl = el('h2', { class: 'wk-title' });
    WK.bodyEl = el('div', { class: 'wk-body' });
    WK.prevBtn = el('button', { class: 'btn', type: 'button', onclick: () => go(WK.phase - 1) }, '← Back');
    WK.nextBtn = el('button', { class: 'btn btn-primary', type: 'button', onclick: () => go(WK.phase + 1) }, 'Next →');
    WK.autoBtn = el('button', { class: 'btn btn-quiet', type: 'button', onclick: () => (WK.auto ? stopAuto() : startAuto()) }, 'Auto-play');
    root.append(
      el('p', { class: 'sec-note', text: 'Follow one training example through a full step of gradient descent: forward through every layer, the loss, the error signal flowing backward, and the weight update. The network diagram shows each phase.' }),
      WK.sampleEl,
      WK.progress,
      el('article', { class: 'wk-card' }, WK.titleEl, WK.bodyEl),
      el('div', { class: 'btn-row wk-nav' }, WK.prevBtn, WK.nextBtn, WK.autoBtn));
    WK.render();
  };
  WK.update = function () { /* rendering is driven by publish() */ };

  function go(ph) {
    WK.phase = clamp(ph, 0, WK.nPhases - 1);
    publish();
  }
  function startAuto() {
    WK.autoBtn.textContent = 'Stop';
    const tick = () => { if (WK.phase >= WK.nPhases - 1) { stopAuto(); return; } go(WK.phase + 1); WK.auto = setTimeout(tick, 2600); };
    WK.auto = setTimeout(tick, 600);
  }
  function stopAuto() { if (WK.auto) clearTimeout(WK.auto); WK.auto = null; if (WK.autoBtn) WK.autoBtn.textContent = 'Auto-play'; }

  WK.render = function () {
    if (!WK.root || !WK.trace) return;
    const ds = App.ds, net = App.net, L = net.L, tr = WK.trace, k = WK.sample, info = phaseInfo(WK.phase);
    // sample line
    const where = ds.kind === 'plane' ? `at (${fmt(ds.raw[2 * k], 2)}, ${fmt(ds.raw[2 * k + 1], 2)})` : ds.kind === 'line' ? `at x = ${fmt(ds.raw[k], 2)}` : '';
    const label = ds.task === 'regression' ? `target y = ${fmt(ds.y[k], 3)}` : `label: ${App.className(ds.y[k])}`;
    NA.clear(WK.sampleEl).append(
      el('div', { class: 'wk-sample-txt' }, el('span', { class: 'eyebrow', text: ['Training example', 'Validation example', 'Test example'][ds.split[k]] }), el('span', { text: ` #${k} ${where} · ${label}` })),
      el('div', { class: 'btn-row' },
        el('button', { class: 'btn btn-quiet', type: 'button', onclick: () => { WK.sample = ds.trainIdx[Math.floor(Math.random() * ds.trainIdx.length)]; WK.phase = 0; WK.history = []; WK.recompute(); App.setProbeSample(WK.sample); } }, 'Random example'),
        App.pinned && App.pinned.sample !== null && App.pinned.sample !== k ? el('button', { class: 'btn btn-quiet', type: 'button', onclick: () => { WK.sample = App.pinned.sample; WK.phase = 0; WK.history = []; WK.recompute(); } }, 'Use the probe') : null));
    // progress
    NA.clear(WK.progress);
    for (let p = 0; p < WK.nPhases; p++) {
      const pi = phaseInfo(p);
      const lab = pi.type === 'input' ? 'Input' : pi.type === 'forward' ? 'Fwd ' + App.layerName(pi.layer, true) : pi.type === 'loss' ? 'Loss' : pi.type === 'outdelta' ? 'δ out' : pi.type === 'backprop' ? 'δ ' + App.layerName(pi.layer, true) : 'Update';
      WK.progress.appendChild(el('button', { type: 'button', role: 'tab', class: 'wk-step wk-' + (pi.type === 'input' || pi.type === 'forward' ? 'fwd' : pi.type === 'update' ? 'upd' : 'bwd'), 'aria-selected': String(p === WK.phase), title: lab, onclick: () => go(p) }, el('span', { class: 'wk-step-lab', text: lab })));
    }
    WK.prevBtn.disabled = WK.phase === 0;
    WK.nextBtn.disabled = WK.phase === WK.nPhases - 1;
    NA.clear(WK.bodyEl);
    const P = (t) => WK.bodyEl.appendChild(el('p', { text: t }));
    const M = (...lines) => WK.bodyEl.appendChild(el('div', { class: 'wk-math' }, ...lines.filter(Boolean).map((t) => el('div', { text: t }))));
    const focusJ = (l) => {
      if (App.selected && App.selected.l === l) return App.selected.j;
      const v = info.type === 'backprop' || info.type === 'outdelta' ? tr.delta[l] : tr.a[l];
      let best = 0; for (let j = 1; j < v.length; j++) if (Math.abs(v[j]) > Math.abs(v[best])) best = j;
      return best;
    };
    const vec = (arr, n = 6) => '[' + Array.from(arr).slice(0, n).map((v) => fmt(v, 2)).join(', ') + (arr.length > n ? `, … ${arr.length - n} more` : '') + ']';
    const sym = (l) => (l === 0 ? 'x' : l === L ? 'out' : 'h' + l);

    if (info.type === 'input') {
      WK.titleEl.textContent = 'The input';
      P(ds.kind === 'image' ? `A 14×14 image of the digit ${ds.y[k]}, flattened into ${ds.dim} numbers between 0 (blank) and 1 (ink). These are the activations of the input layer.` : `The example’s ${ds.dim === 1 ? 'value is' : ds.dim + ' input features are'} the activations of the input layer. Nothing has been computed yet; every later layer starts out unknown.`);
      M(`a⁽⁰⁾ = ${ds.kind === 'image' ? `${ds.dim} pixels, ${Array.from(tr.a[0]).filter((v) => v > 0.05).length} with ink` : vec(tr.a[0])}`);
      if (ds.kind === 'plane') M(...ds.featureLabels.map((f, i) => `${f} = ${fmt(tr.a[0][i], 3)}`));
    } else if (info.type === 'forward') {
      const l = info.layer, j = focusJ(l), nIn = net.sizes[l - 1], Wl = net.W[l - 1];
      const lid = l < L ? net.layerActId(l) : null, act = net.actOf(l, j);
      const fname = l === L ? { sigmoid: 'σ', softmax: 'softmax', identity: 'identity' }[net.out] : lid === 'mixed' ? 'its own activation function (this layer mixes several)' : `${E.ACTIVATIONS[lid].label}, f(z) = ${E.ACTIVATIONS[lid].formula}`;
      WK.titleEl.textContent = `Forward through ${App.layerName(l)}`;
      P(`Every neuron in ${App.layerName(l)} computes a weighted sum of the ${nIn} activations below it plus its bias, z = W·a + b, then applies ${fname}. The connections now show each signal w·a for this example.`);
      const terms = [];
      const order = NA.range(nIn).sort((p, q) => Math.abs(Wl[j * nIn + q] * tr.a[l - 1][q]) - Math.abs(Wl[j * nIn + p] * tr.a[l - 1][p]));
      for (const i of order.slice(0, 4)) terms.push(`${fmt(Wl[j * nIn + i], 2)}×${fmt(tr.a[l - 1][i], 2)}`);
      const rest = nIn > 4 ? ` + … (${nIn - 4} more)` : '';
      M(`For ${App.neuronName(l, j)}:`,
        `z = ${terms.join(' + ')}${rest} + ${fmt(net.b[l - 1][j], 2)}`,
        `  = ${fmt(tr.z[l][j], 4)}`,
        l < L ? `f(z) = ${act.formula}   (${act.label})` : null,
        l < L ? `a = ${act.apply(fmt(tr.z[l][j], 3))} = ${fmt(tr.a[l][j], 4)}` : net.out === 'sigmoid' ? `p = σ(z) = ${fmt(tr.a[l][j], 4)}` : net.out === 'softmax' ? `p = softmax(z) = ${fmt(tr.a[l][j], 4)}` : `ŷ = z = ${fmt(tr.a[l][j], 4)}`);
      M(`a⁽${l}⁾ = ${vec(tr.a[l])}`);
      if (l < L) {
        const n = net.sizes[l], fns = net.actFns[l];
        const off = NA.range(n).filter((q) => fns[q].canDie && tr.z[l][q] <= 0).length;
        if (off) P(off === 1 ? `1 of ${n} neurons is a ReLU with z ≤ 0 for this example, so it outputs exactly 0 and will pass no gradient back either.` : `${off} of ${n} neurons are ReLUs with z ≤ 0 for this example, so they output exactly 0 and will pass no gradient back either.`);
        const sat = NA.range(n).filter((q) => fns[q].sat && fns[q].sat(tr.a[l][q])).length;
        if (sat) P(sat === 1 ? `1 of ${n} neurons sits on a flat tail of its activation here: its slope is nearly 0, so little error will flow back through it.` : `${sat} of ${n} neurons sit on a flat tail of their activation here: their slope is nearly 0, so little error will flow back through them.`);
        const p = App.hyper.dropout;
        if (p > 0) P(`Dropout is on (${fmtPct(p, 0)}). In a real training step each of these neurons would be switched off for this example with probability ${p}, and the ones kept scaled up by 1/(1 − p) = ${fmt(1 / (1 - p), 2)}. This walkthrough keeps every neuron on, as when the model is used, so each number can be checked by hand.`);
      }
    } else if (info.type === 'loss') {
      WK.titleEl.textContent = 'Prediction and loss';
      const a = tr.a[L], y = ds.y[k], id = net.lossId, Ls = E.LOSSES[id];
      if (ds.task === 'binary') {
        const t = y ? 1 : -1, z = tr.z[L][0], right = (a[0] >= 0.5 ? 1 : 0) === y, conf = Math.max(a[0], 1 - a[0]);
        P(`The output neuron gives p = ${fmt(a[0], 4)}, the network’s probability that this point is blue. The true label is y = ${y}. ${id === 'hinge' ? 'Hinge loss looks at the raw score z instead, with the label written as t = ±1; only scores inside the margin (t·z < 1) cost anything:' : id === 'mse' ? 'Squared error measures the squared distance between the probability and the label:' : 'Binary cross-entropy measures how surprised the network is by the truth:'}`);
        if (id === 'hinge') M(Ls.formula.sigmoid, `  t = ${t > 0 ? '+1' : '−1'}, z = ${fmt(z, 3)}, t·z = ${fmt(t * z, 3)}`, `  = ${fmt(tr.loss, 4)}`);
        else if (id === 'mse') M(Ls.formula.sigmoid, `  = ½(${fmt(a[0], 4)} − ${y})²`, `  = ${fmt(tr.loss, 4)}`);
        else M(Ls.formula.sigmoid, `  = −ln ${y ? fmt(a[0], 4) : fmt(1 - a[0], 4)}`, `  = ${fmt(tr.loss, 4)}`);
        P(id === 'hinge' ? (tr.loss === 0 ? 'Zero loss: the score is beyond the margin, so this example produces no gradient at all.' : t * z > 0 ? 'The score is on the right side but inside the margin, so it still pushes the boundary further away.' : 'The score is on the wrong side of the boundary, so it costs more than 1 and pulls hard.')
          : !right && id === 'mse' && conf > 0.9 ? 'A large miss, yet the push back will be weak: far out on the sigmoid its slope p(1 − p) is nearly 0. This is why squared error learns slowly from confident mistakes.'
          : !right ? 'A large loss: the network leans the wrong way on this example, so the gradients will be large.' : conf > 0.9 ? 'A small loss: the network is already confident and right about this example.' : 'A moderate loss: the network leans the right way but is not yet confident.');
      } else if (ds.task === 'multiclass') {
        P(`Softmax turns the ${net.sizes[L]} output scores into probabilities that sum to 1. The true class is ${App.className(y)}. ${id === 'mse' ? 'Squared error compares every probability with the one-hot target (1 for the true class, 0 elsewhere):' : 'Cross-entropy is the negative log of the probability given to the true class:'}`);
        if (id === 'mse') M(`p = ${vec(a, 10)}`, `y = onehot(${y})`, `${Ls.formula.softmax} = ${fmt(tr.loss, 4)}`);
        else M(`p = ${vec(a, 10)}`, `L = −ln p[${y}] = −ln ${fmt(a[y], 4)} = ${fmt(tr.loss, 4)}`);
      } else {
        const r = a[0] - y;
        P(`The network predicts ŷ = ${fmt(a[0], 4)}; the target is y = ${fmt(y, 4)}. ${id === 'mae' ? 'The loss is the absolute size of the miss:' : id === 'huber' ? `Huber loss is squared for misses up to ${E.HUBER} and grows linearly beyond:` : 'The loss is half the squared error:'}`);
        if (id === 'mae') M(Ls.formula.identity, `  = |${fmt(a[0], 3)} − ${fmt(y, 3)}|`, `  = ${fmt(tr.loss, 5)}`);
        else if (id === 'huber') M(Ls.formula.identity, `  r = ŷ − y = ${fmt(r, 4)}, |r| ${Math.abs(r) <= E.HUBER ? '≤' : '>'} ${E.HUBER}`, `  = ${fmt(tr.loss, 5)}`);
        else M(Ls.formula.identity, `  = ½(${fmt(a[0], 3)} − ${fmt(y, 3)})²`, `  = ${fmt(tr.loss, 5)}`);
      }
    } else if (info.type === 'outdelta') {
      WK.titleEl.textContent = 'Error signal at the output';
      const d = tr.delta[L], a = tr.a[L], y = ds.y[k], id = net.lossId, Ls = E.LOSSES[id];
      const how = {
        'binary:xent': 'For sigmoid with cross-entropy, the calculus simplifies to prediction minus target:',
        'binary:mse': 'For sigmoid with squared error, the chain rule multiplies the miss by the sigmoid’s slope p(1 − p):',
        'binary:hinge': 'For hinge loss the error signal is −t while the score is inside the margin, and 0 once it is beyond:',
        'multiclass:xent': 'For softmax with cross-entropy, the calculus simplifies to prediction minus the one-hot target:',
        'multiclass:mse': 'With squared error through softmax, each output’s δ depends on all the others, because softmax couples them:',
        'regression:mse': 'For a linear output with squared error, δ is simply the miss:',
        'regression:mae': 'For absolute error, δ is only the sign of the miss, so every example pulls equally hard:',
        'regression:huber': `For Huber loss, δ is the miss while it is small, capped at ±${E.HUBER} beyond:`,
      }[ds.task + ':' + id];
      P(`Backpropagation starts at the output. δ = ∂L/∂z measures how much the loss would change if the output neuron’s z moved up a little. ${how}`);
      if (ds.task === 'multiclass') M(id === 'mse' ? Ls.delta.softmax : 'δ = p − onehot(y)', `  = ${vec(d, 10)}`);
      else if (ds.task === 'binary' && id === 'mse') M(Ls.delta.sigmoid, `  = (${fmt(a[0], 4)} − ${y}) × ${fmt(a[0] * (1 - a[0]), 4)}`, `  = ${fmt(d[0], 5)}`);
      else if (ds.task === 'binary' && id === 'hinge') { const t = y ? 1 : -1, tz = t * tr.z[L][0]; M(Ls.delta.sigmoid, `  t·z = ${fmt(tz, 3)} ${tz < 1 ? '< 1' : '≥ 1'}, so δ = ${fmt(d[0], 0)}`); }
      else if (ds.task === 'binary') M('δ = p − y', `  = ${fmt(a[0], 4)} − ${y} = ${fmt(d[0], 4)}`);
      else if (id === 'mae') M(Ls.delta.identity, `  = sign(${fmt(a[0] - y, 4)}) = ${fmt(d[0], 0)}`);
      else if (id === 'huber') M(Ls.delta.identity, `  r = ${fmt(a[0] - y, 4)}, so δ = ${fmt(d[0], 4)}`);
      else M('δ = ŷ − y', `  = ${fmt(a[0], 4)} − ${fmt(y, 4)} = ${fmt(d[0], 4)}`);
      P('Each weight into the output then gets the gradient ∂L/∂w = δ × (the activation it multiplies). The violet connections show these gradients.');
      const g = bestGrad(L);
      M(`Largest: ∂L/∂w for ${App.neuronName(L - 1, g.i, true)} → ${App.neuronName(L, g.j, true)} = ${fmt(d[g.j], 3)} × ${fmt(tr.a[L - 1][g.i], 3)} = ${fmt(g.g, 4)}`);
    } else if (info.type === 'backprop') {
      const l = info.layer, j = focusJ(l), nOut = net.sizes[l + 1], Wn = net.W[l], dn = tr.delta[l + 1];
      let s = 0; for (let q = 0; q < nOut; q++) s += Wn[q * net.sizes[l] + j] * dn[q];
      const act = net.actOf(l, j), slope = act.d(tr.z[l][j], tr.a[l][j]);
      WK.titleEl.textContent = `Backpropagate into ${App.layerName(l)}`;
      P(`Each neuron in ${App.layerName(l)} collects the error signals of the layer above, weighted by the connections it feeds, then multiplies by its own slope f′(z). This is the chain rule, applied one layer at a time:`);
      M('δᵢ = f′(zᵢ) · Σⱼ wⱼᵢ·δⱼ', `For ${App.neuronName(l, j)}:`, `  Σⱼ wⱼᵢ·δⱼ = ${fmt(s, 4)}`, `  f′(z) = ${act.dformula}   (${act.label})`, `  f′(${fmt(tr.z[l][j], 3)}) = ${fmt(slope, 4)}`, `  δ = ${fmt(slope * s, 5)}`);
      const mUp = rms(tr.delta[l + 1]), mHere = rms(tr.delta[l]);
      P(`Average |δ| here is ${fmt(mHere, 5)}, versus ${fmt(mUp, 5)} one layer up${mUp > 0 ? ` (×${fmt(mHere / mUp, 2)})` : ''}. ${mHere < mUp * 0.3 ? 'The signal shrinks as it travels back; across many layers this is how gradients vanish.' : mHere > mUp * 3 ? 'The signal grows as it travels back; across many layers it can explode.' : 'The signal keeps roughly the same size, which lets every layer learn.'}`);
      const g = bestGrad(l);
      M(`Largest weight gradient into ${App.layerName(l)}: ${fmt(g.g, 5)} (${App.neuronName(l - 1, g.i, true)} → ${App.neuronName(l, g.j, true)})`);
    } else {
      WK.titleEl.textContent = 'Update the weights';
      P(`Gradient descent moves every weight a small step against its gradient, w ← w − η·∂L/∂w. Here we take one plain SGD step on this single example. (During training the gradients are averaged over a batch of ${App.hyper.batch} and the ${App.hyper.optimizer === 'adam' ? 'Adam' : App.hyper.optimizer === 'momentum' ? 'momentum' : 'SGD'} optimizer decides the step size.)`);
      const all = [];
      for (let l = 0; l < L; l++) for (let j = 0; j < net.sizes[l + 1]; j++) for (let i = 0; i < net.sizes[l]; i++) all.push({ l, i, j, g: tr.delta[l + 1][j] * tr.a[l][i] });
      all.sort((a, b) => Math.abs(b.g) - Math.abs(a.g));
      const rows = all.slice(0, 5).map((r) => { const w = net.W[r.l][r.j * net.sizes[r.l] + r.i]; return `${App.neuronName(r.l, r.i, true)} → ${App.neuronName(r.l + 1, r.j, true)}: ${fmt(w, 3)} − ${WK.eta}×${fmt(r.g, 4)} = ${fmt(w - WK.eta * r.g, 3)}`; });
      M('The five weights with the largest gradients:', ...rows);
      const etaSel = el('select', { id: 'wkEta', 'aria-label': 'Step size η' }, ...[0.01, 0.03, 0.1, 0.3, 1].map((v) => el('option', { value: v, selected: v === WK.eta }, 'η = ' + v)));
      etaSel.addEventListener('change', () => { WK.eta = +etaSel.value; WK.render(); });
      const apply = el('button', { class: 'btn btn-primary', type: 'button', onclick: applyUpdate }, 'Apply this update');
      WK.bodyEl.appendChild(el('div', { class: 'btn-row' }, etaSel, apply));
      if (WK.history.length) {
        const h = WK.history[WK.history.length - 1];
        WK.bodyEl.appendChild(el('p', { class: 'wk-result' }, `Applied ${WK.history.length} update${WK.history.length > 1 ? 's' : ''}. Loss on this example: ${fmt(h.before, 4)} → ${fmt(h.after, 4)}`, h.after < h.before ? ' (lower: the step helped)' : ' (higher: the step overshot; try a smaller η)'));
      }
      P('Repeat this for every batch, thousands of times, and the network learns. Use the Train button to watch it happen at full speed.');
    }
  };

  function rms(v) { let s = 0; for (const x of v) s += x * x; return Math.sqrt(s / Math.max(1, v.length)); }
  function bestGrad(l) {
    const net = App.net, tr = WK.trace, nIn = net.sizes[l - 1];
    let best = { i: 0, j: 0, g: 0 };
    for (let j = 0; j < net.sizes[l]; j++) for (let i = 0; i < nIn; i++) { const g = tr.delta[l][j] * tr.a[l - 1][i]; if (Math.abs(g) > Math.abs(best.g)) best = { i, j, g }; }
    return best;
  }

  function applyUpdate() {
    const net = App.net, tr = WK.trace, L = net.L, before = tr.loss;
    for (let l = 0; l < L; l++) {
      const nIn = net.sizes[l];
      for (let j = 0; j < net.sizes[l + 1]; j++) {
        const d = tr.delta[l + 1][j];
        for (let i = 0; i < nIn; i++) net.W[l][j * nIn + i] -= WK.eta * d * tr.a[l][i];
        net.b[l][j] -= WK.eta * d;
      }
    }
    WK._applying = true;
    App.weightsEdited();
    WK._applying = false;
    WK.recompute(true);
    WK.history.push({ before, after: WK.trace.loss });
    WK.render();
  }

  NA.walk = WK;
})(window.NA);
