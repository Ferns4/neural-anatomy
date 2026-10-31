/* builder-evo.js — evolution for the networks of the model builder's "Learn from rewards" (builder-rl.js hands
   it the world and the network you built). Six methods:
   · a genetic algorithm: a population of networks plays, each judged by the return of its best moves; the best
     few pass on unchanged and the rest are children of parents picked by a tournament (the best of 3), from the
     top fifth, or by rank; mixed weight by weight, neuron by neuron, or not at all; mutated by plain random
     nudges to one weight in ten, or by "safe" ones, each nudge divided by how much the moves depend on that
     weight. With grow and prune on, a child may also gain or lose a neuron, or gain a hidden layer. A new neuron
     starts with outgoing weights of 0, so it changes nothing until mutations put it to work. A new layer starts
     with one neuron (or a few at random, or as wide as what it reads, for comparison): a narrow layer is a
     bottleneck that passes on only the signals the next layer leans on most, so unlike a new neuron it changes
     what the network does, and selection keeps it only if the child still does well; later "+1 neuron" changes
     widen it without changing anything. A layer goes when its last neuron is pruned, folded into the next. A size
     penalty (the fitness less a little for every 100 weights) keeps them small. Each of these can be set
     (EV.growOptions): which changes are allowed (a whole layer out at once and a new activation too), their odds,
     how many children change, where new layers go, how new neurons start, which neuron pruning takes, the largest
     network, starting from no hidden layer, and protecting new shapes for a few generations.
   · novelty search (Lehman and Stanley, 2011): the same breeding, but a member is judged by how different its
     behaviour is (where it ended up, how it played: two numbers each world measures) from its nearest neighbours
     in its generation and in an archive of behaviours met before, optionally mixed with the reward. It never aims
     at the goal, so a goal that a straight climb leads away from (the deceptive maze) can still be found on the
     way. The network on show is the best by score found so far.
   · evolution strategies: one network, nudged in many random directions at once; how each nudged copy does says
     which way is uphill (an estimate of the gradient, with no backpropagation), and Adam takes a step that way.
   · the cross-entropy method: a bell curve (a Gaussian) over the weights, its middle and its spread for every
     weight; each generation samples networks from it and refits it to the best fifth, with a small floor under
     the spread so it never stops looking.
   · MAP-Elites: a grid of 12 × 12 styles of play (two numbers each world measures, such as average speed and
     which side of the track it keeps); each new network goes to the cell of its style and stays if it beats the
     one there, so it finds the best of every style rather than one best.
   · evolutionary reinforcement learning (ERL, Khadka and Tumer, 2018): a genetic algorithm whose every move goes
     into a replay memory that a DQN learns from by gradient steps between generations; every few generations the
     DQN's network joins the population in place of its worst member, and breeds if it does well.
   Every member is judged on the same starts, a few members per call (step(deadline)), so a big world never
   stalls the page; and every member keeps the path it took, so a whole generation can be shown at once. Runs in
   the page and in node.

   What the depth views draw (builder-viz.js), kept cheap:
   · evo.history[k], one a generation: { gen, best, mean, weights, units, adj, div, divW } and by method
       div: behaviour diversity, the mean distance of the members' behaviours (world.behavior(): 2 numbers in
            [0, 1]) from their centroid; null when the world measures none
       divW: weight diversity, the root mean square over members and weights of each weight's distance from the
            members' mean, among the members of the best's shape; null when fewer than 2 share it
       spread (the cross-entropy method): the mean per-weight spread σ its generation was drawn with
       filled (MAP-Elites): how many of the 144 cells hold an elite
       rlLoss (ERL): the DQN's average loss over its gradient steps after this generation (null before it learns)
   · evo.last = { gen, members, best, mean }: members best first, each { id, parents, fit, adj, units, weights, beh,
       trace, end, op, born } and by method
       id: a number, unique in the run; parents: [] (the first generation, random or sampled ones, the DQN's
            network), [id] (a mutated copy) or [id, id] (two parents mixed); an elite passed on unchanged keeps its id
       op: what made it ('mutated', 'mixed and mutated', '+1 neuron in layer 2', …; null: unchanged)
       nov (novelty search): its novelty; fromRL (ERL): true for the DQN's network copied into the population
       grow and prune: prot (carried on unchanged as a new shape, protected), acts (its hidden layers' activations,
       when they can change); evo.last.shapes and history[k].shapes: how many shapes the generation has
   · evo.lineage(id = the best): the line of first parents from the oldest ancestor still known to that member,
       oldest first: [{ id, gen, fit, op, units }] (gen: the generation it was first judged in, fit: its fitness
       then). A map of records is kept, pruned to the ancestors of the current population (and the network on
       show, MAP-Elites' elites), at most EV.KEEP of them; evo.bestId() is the best's id.
   · evo.novArchive (novelty search): [{ beh: [x, y], gen }], the behaviours kept as stepping stones. */
(function (NA) {
  'use strict';
  const EV = {};
  const now = () => (typeof performance !== 'undefined' ? performance.now() : Date.now());

  // ------------------------------------------------------------------ architectures that grow and shrink
  EV.LIMITS = { layers: 6, units: 48 }; // the defaults: a run can set its own (growMaxLayers, growMaxUnits)
  // how wide a new hidden layer starts: one neuron (the default), a few at random, or as wide as what it reads
  EV.LAYER_SIZE = { one: 'One neuron', some: 'A few, at random', whole: 'A whole layer' };
  EV.layerSize = (v) => (EV.LAYER_SIZE[v] ? v : 'one');
  /* The changes to a network's shape, each of which can be turned off, and their odds by leaning: toward growing,
     even (the default, as before these could be set), or toward pruning. '−1 layer' takes a whole layer out at once,
     folded into the next; 'act' switches a hidden layer to another of EV.ACTS (its weights stay). */
  EV.CHANGES = { 'neuron+': '+1 neuron', 'neuron-': '−1 neuron', 'layer+': '+1 layer', 'layer-': '−1 layer', act: 'New activation' };
  EV.LEAN = { grow: { 'neuron+': 4, 'neuron-': 1, 'layer+': 2, 'layer-': 0.5, act: 1 }, even: { 'neuron+': 3, 'neuron-': 2, 'layer+': 1.4, 'layer-': 1, act: 1 }, prune: { 'neuron+': 1.5, 'neuron-': 4, 'layer+': 0.7, 'layer-': 2, act: 1 } };
  EV.ACTS = ['relu', 'tanh', 'sigmoid', 'leaky', 'elu'];
  EV.PROTECT = { gens: 5, most: 3 }; // a new shape's best carries on unchanged for 5 generations; at most 3 such a generation
  const clampInt = (v, lo, hi, def) => (typeof v === 'number' && Number.isFinite(v) ? Math.max(lo, Math.min(hi, Math.round(v))) : def);
  /* Grow and prune's settings as one object, from a reward problem's settings (s: penalty, layerSize, newLayer and the
     grow… keys) or, with sup, from the training settings of a problem with answers (growPenalty, growWidth, growNew
     and the same grow… keys): how wide new layers start and their weights (size, mode), where they go, how a new
     neuron starts, which neuron pruning takes, the changes allowed and their odds (weights), the limits, how many
     children change shape (rate: the genetic algorithm), whether a new shape is protected for a few generations,
     whether it starts from no hidden layer (start: 'small'), and while it learns from examples the networks side
     by side (members) and the steps between generations (every). Missing settings are the defaults; "−1 layer"
     is on by default only with whole layers, as it was before it could be set. */
  EV.growOptions = function (s, sup) {
    s = s || {};
    const size = EV.layerSize(sup ? s.growWidth : s.layerSize), on = (k, def) => (typeof s[k] === 'boolean' ? s[k] : def), lean = EV.LEAN[s.growLean] ? s.growLean : 'even';
    const ops = { 'neuron+': on('growAddNeuron', true), 'neuron-': on('growCutNeuron', true), 'layer+': on('growAddLayer', true), 'layer-': on('growCutLayer', size === 'whole'), act: on('growAct', false) };
    const weights = {};
    for (const k of Object.keys(ops)) weights[k] = ops[k] ? EV.LEAN[lean][k] : 0;
    const pen = sup ? s.growPenalty : s.penalty;
    return { size, mode: (sup ? s.growNew : s.newLayer) === 'random' ? 'random' : 'identity', where: s.growWhere === 'first' || s.growWhere === 'last' ? s.growWhere : 'any',
      neuron: s.growNeuron === 'random' ? 'random' : 'silent', prune: s.growPrune === 'random' ? 'random' : 'weak', lean, ops, weights,
      maxLayers: clampInt(s.growMaxLayers, 1, 12, EV.LIMITS.layers), maxUnits: clampInt(s.growMaxUnits, 1, 128, EV.LIMITS.units),
      rate: typeof s.growRate === 'number' && s.growRate > 0 ? Math.min(1, s.growRate) : 0.25, protect: !!s.growProtect, start: s.growStart === 'small' ? 'small' : 'built',
      penalty: EV.PENALTY[pen] !== undefined ? pen : 'light', members: clampInt(s.growSize, 2, 12, 6), every: clampInt(s.growEvery, 10, 5000, 150) };
  };
  /* The changes of shape in a line, for the Growth view's math: who changes (the genetic algorithm: a share of the
     children; while it learns from examples: every copy), and the changes allowed. */
  EV.describe = function (o, sup) {
    const W = o.weights, rate = { 0.1: 'one child in ten', 0.25: 'a quarter of the children', 0.5: 'half the children', 1: 'every child' }[o.rate] || `${Math.round(100 * o.rate)}% of the children`;
    const n = W['neuron+'] && W['neuron-'] ? 'a neuron more or less' : W['neuron+'] ? 'a neuron more' : W['neuron-'] ? 'a neuron less' : null;
    const l = W['layer+'] && W['layer-'] ? 'a layer more or less' : W['layer+'] ? `a new layer of ${o.size === 'one' ? 'one neuron' : o.size === 'some' ? 'a few neurons' : 'the width it reads'}` : W['layer-'] ? 'a layer less' : null;
    const list = [n, l, W.act ? 'a new activation' : null].filter(Boolean);
    return `${sup ? 'each copy' : rate}: ${list.length > 1 ? `${list.slice(0, -1).join(', ')} or ${list[list.length - 1]}` : list[0] || 'no change (every change is off)'}`;
  };
  // how likely each change is, as a share of all of them, for a network with hidden layers (none: a new layer is all it can do)
  EV.changeOdds = function (o, n = 1) {
    const w = Object.assign({}, o.weights, n ? {} : { 'neuron+': 0, 'neuron-': 0, 'layer-': 0, act: 0 }, n >= o.maxLayers ? { 'layer+': 0 } : {}), total = Object.values(w).reduce((a, b) => a + b, 0);
    return Object.fromEntries(Object.keys(w).map((k) => [k, total ? w[k] / total : 0]));
  };
  // a model evolution can reshape: Dense blocks only (one activation each, no shortcut), on a list of numbers
  EV.evolvable = (blocks, input) => !!input && input.kind === 'vector' && blocks.every((b) => b.type === 'dense' && !b.residual && !Array.isArray(b.acts) && !b.grp);
  const unitsOf = (blocks) => blocks.map((b) => Math.max(1, Math.round(b.units) || 1));
  EV.unitsOf = unitsOf;
  EV.sizes = (blocks, nIn, K) => [nIn, ...unitsOf(blocks), K];
  EV.weights = (sz) => { let n = 0; for (let l = 1; l < sz.length; l++) n += sz[l] * (sz[l - 1] + 1); return n; };
  // the engine's order: each layer's weights row by row (a row per neuron), then its biases
  function split(sz, w) {
    const out = [];
    let k = 0;
    for (let l = 1; l < sz.length; l++) { const U = sz[l], I = sz[l - 1]; out.push({ W: w.slice(k, k + U * I), b: w.slice(k + U * I, k + U * I + U), U, I }); k += U * I + U; }
    return out;
  }
  function join(L) {
    const w = new Float32Array(L.reduce((s, q) => s + q.W.length + q.b.length, 0));
    let k = 0;
    for (const q of L) { w.set(q.W, k); k += q.W.length; w.set(q.b, k); k += q.b.length; }
    return w;
  }
  const withUnits = (b, units) => Object.assign({}, b, { units });
  /* A neuron more in hidden layer h: its incoming weights at random, its outgoing ones 0 (o.neuron 'silent', the
     default), so the network computes exactly what it did and later mutations can put the new neuron to work; or
     its outgoing weights at random too ('random'), so it joins in at once, for better or worse. */
  function addNeuron(g, h, rng, io, o) {
    const L = split(EV.sizes(g.blocks, io.nIn, io.K), g.w), q = L[h], nx = L[h + 1], I = q.I, U = q.U;
    if (U >= (o ? o.maxUnits : EV.LIMITS.units)) return null;
    const W = new Float32Array((U + 1) * I), b = new Float32Array(U + 1), W2 = new Float32Array(nx.U * (U + 1)), sd = Math.sqrt(2 / I);
    W.set(q.W); b.set(q.b);
    for (let i = 0; i < I; i++) W[U * I + i] = rng.gauss(0, sd);
    for (let r = 0; r < nx.U; r++) W2.set(nx.W.subarray(r * U, r * U + U), r * (U + 1));
    if (o && o.neuron === 'random') { const so = Math.sqrt(1 / (U + 1)); for (let r = 0; r < nx.U; r++) W2[r * (U + 1) + U] = rng.gauss(0, so); }
    L[h] = { W, b, U: U + 1, I }; L[h + 1] = { W: W2, b: nx.b, U: nx.U, I: U + 1 };
    return { blocks: g.blocks.map((x, i) => (i === h ? withUnits(x, U + 1) : x)), w: join(L), op: `+1 neuron in layer ${h + 1}` };
  }
  // hidden layer h switched to another activation (its weights stay, what it computes changes)
  const actLabel = (a) => ((NA.engine && NA.engine.ACTIVATIONS && NA.engine.ACTIVATIONS[a]) || {}).label || a;
  function changeAct(g, h, rng) {
    const cur = g.blocks[h].act || 'relu', other = EV.ACTS.filter((a) => a !== cur), to = other[rng.int(other.length)];
    return { blocks: g.blocks.map((x, i) => (i === h ? Object.assign({}, x, { act: to }) : x)), w: Float32Array.from(g.w), op: `layer ${h + 1}: ${actLabel(cur)} → ${actLabel(to)}` };
  }
  // the neuron of layer h the next layer listens to least (the smallest outgoing weights)
  function weakest(g, h, io) {
    const L = split(EV.sizes(g.blocks, io.nIn, io.K), g.w), nx = L[h + 1];
    let best = 0, bs = Infinity;
    for (let j = 0; j < nx.I; j++) { let s = 0; for (let r = 0; r < nx.U; r++) s += Math.abs(nx.W[r * nx.I + j]); if (s < bs) { bs = s; best = j; } }
    return best;
  }
  function removeNeuron(g, h, j, io) {
    const L = split(EV.sizes(g.blocks, io.nIn, io.K), g.w), q = L[h], nx = L[h + 1], I = q.I, U = q.U;
    if (U <= 1) return null;
    const W = new Float32Array((U - 1) * I), b = new Float32Array(U - 1), W2 = new Float32Array(nx.U * (U - 1));
    for (let r = 0, k = 0; r < U; r++) if (r !== j) { W.set(q.W.subarray(r * I, r * I + I), k * I); b[k] = q.b[r]; k++; }
    for (let r = 0; r < nx.U; r++) for (let c = 0, k = 0; c < U; c++) if (c !== j) W2[r * (U - 1) + k++] = nx.W[r * U + c];
    L[h] = { W, b, U: U - 1, I }; L[h + 1] = { W: W2, b: nx.b, U: nx.U, I: U - 1 };
    return { blocks: g.blocks.map((x, i) => (i === h ? withUnits(x, U - 1) : x)), w: join(L), op: `−1 neuron in layer ${h + 1}` };
  }
  /* A new hidden layer at position p (0: straight after the input), reading the I numbers there. size 'whole': I
     neurons, its weights the identity, so it passes its input straight through (exactly, after a ReLU layer), or
     at random. 'one' (1 neuron) or 'some' (1 to I − 1, at random): a narrow layer. It passes on the signals the
     next layer leans on most (the inputs whose columns of the next layer's weights are largest, in absolute size):
     each of its neurons picks one of them (a weight of 1; or its weights at random), and the next layer reads them
     with those same columns. What it leaves out is lost, so it changes what the network does: a bottleneck. */
  function addLayer(g, p, mode, rng, io, size = 'whole', o = null) {
    const most = o ? o.maxUnits : EV.LIMITS.units;
    if (g.blocks.length >= (o ? o.maxLayers : EV.LIMITS.layers)) return null;
    const sz = EV.sizes(g.blocks, io.nIn, io.K), I = sz[p], L = split(sz, g.w), nx = L[p];
    const U = size === 'whole' ? I : size === 'some' ? 1 + rng.int(Math.max(1, Math.min(I - 1, most))) : 1;
    if (U > most) return null;
    const W = new Float32Array(U * I), b = new Float32Array(U);
    if (size === 'whole') {
      if (mode === 'random') { const sd = Math.sqrt(2 / U); for (let i = 0; i < W.length; i++) W[i] = rng.gauss(0, sd); }
      else for (let i = 0; i < U; i++) W[i * U + i] = 1;
    } else {
      // the inputs the next layer leans on most (the first of equals first), and its columns for them
      const col = new Float64Array(I);
      for (let r = 0; r < nx.U; r++) for (let j = 0; j < I; j++) col[j] += Math.abs(nx.W[r * I + j]);
      const keep = NA.range(I).sort((a, c) => col[c] - col[a] || a - c).slice(0, U), W2 = new Float32Array(nx.U * U);
      if (mode === 'random') { const sd = Math.sqrt(2 / I); for (let i = 0; i < W.length; i++) W[i] = rng.gauss(0, sd); }
      else keep.forEach((j, k) => { W[k * I + j] = 1; });
      for (let r = 0; r < nx.U; r++) keep.forEach((j, k) => { W2[r * U + k] = nx.W[r * I + j]; });
      L[p] = { W: W2, b: nx.b, U: nx.U, I: U };
    }
    L.splice(p, 0, { W, b, U, I });
    const like = g.blocks[p - 1] || g.blocks[p] || { act: 'relu' }, blocks = g.blocks.slice();
    blocks.splice(p, 0, { type: 'dense', units: U, act: like.act || 'relu' });
    return { blocks, w: join(L), op: `+1 layer (${U}) at ${p + 1}${mode === 'random' ? ', random' : ''}` };
  }
  // a layer taken out, folded into the next: W = W₂·W₁, b = W₂·b₁ + b₂ (as if its activation let values through)
  function removeLayer(g, h, io) {
    if (!g.blocks.length) return null;
    const L = split(EV.sizes(g.blocks, io.nIn, io.K), g.w), a = L[h], c = L[h + 1], W = new Float32Array(c.U * a.I), b = new Float32Array(c.U);
    for (let r = 0; r < c.U; r++) {
      let s = c.b[r];
      for (let m = 0; m < a.U; m++) { const wc = c.W[r * a.U + m]; if (!wc) continue; s += wc * a.b[m]; for (let i = 0; i < a.I; i++) W[r * a.I + i] += wc * a.W[m * a.I + i]; }
      b[r] = s;
    }
    L.splice(h, 2, { W, b, U: c.U, I: a.I });
    const blocks = g.blocks.slice();
    blocks.splice(h, 1);
    return { blocks, w: join(L), op: `−1 layer (${a.U}) at ${h + 1}` };
  }
  // the last neuron of hidden layer h pruned: the layer goes, folded into the next
  function pruneLast(g, h, io) { const r = removeLayer(g, h, io); if (r) r.op = `−1 neuron: layer ${h + 1} gone`; return r; }
  /* One change to the architecture, picked by the odds of o.weights (EV.growOptions): by default a neuron more (most
     often) or less, or a layer more. A new layer goes anywhere, straight after the input or just before the output
     (o.where), and starts as wide as o.size says. Pruning takes the weakest neuron, or one at random (o.prune); a
     layer goes when its last neuron is pruned, except with whole layers, which go at once ("−1 layer"). The older
     call (g, rng, mode, io, size) still works. */
  EV.mutateArch = function (g, rng, o, io, size) {
    if (!o || typeof o !== 'object') o = EV.growOptions({ newLayer: o, layerSize: size || 'one' });
    const n = g.blocks.length, whole = o.size === 'whole', units = unitsOf(g.blocks), W = o.weights;
    // (in this order, so the default odds draw exactly as before the changes could be set)
    const ops = [['neuron+', n ? W['neuron+'] : 0], ['neuron-', n ? W['neuron-'] : 0], ['layer+', n < o.maxLayers ? W['layer+'] : 0], ['layer-', n ? W['layer-'] : 0], ['act', n ? W.act : 0]];
    const total = ops.reduce((s, q) => s + q[1], 0), last = ops.filter((q) => q[1] > 0).pop();
    if (!(total > 0)) return null;
    for (let tries = 0; tries < 8; tries++) {
      let u = rng.next() * total, op = last[0];
      for (const [name, wgt] of ops) { u -= wgt; if (u <= 0 && wgt > 0) { op = name; break; } }
      let r;
      if (op === 'neuron+') r = addNeuron(g, rng.int(n), rng, io, o);
      else if (op === 'neuron-') { const h = rng.int(n); r = units[h] <= 1 && !whole ? pruneLast(g, h, io) : removeNeuron(g, h, o.prune === 'random' ? rng.int(units[h]) : weakest(g, h, io), io); }
      else if (op === 'layer+') r = addLayer(g, o.where === 'first' ? 0 : o.where === 'last' ? n : rng.int(n + 1), o.mode, rng, io, o.size, o);
      else if (op === 'layer-') r = removeLayer(g, rng.int(n), io);
      else r = changeAct(g, rng.int(n), rng);
      if (r) return r;
    }
    return null;
  };
  Object.assign(EV, { split, join, addNeuron, removeNeuron, addLayer, removeLayer, pruneLast, weakest, changeAct });

  // ------------------------------------------------------------------ networks for every architecture met
  class Nets {
    constructor(input, output) { this.input = input; this.output = output; this.map = new Map(); }
    static key(blocks) { return JSON.stringify(blocks); }
    get(blocks) {
      const k = Nets.key(blocks);
      let m = this.map.get(k);
      if (!m) { m = this.make(blocks); this.map.set(k, m); if (this.map.size > 48) this.map.delete(this.map.keys().next().value); }
      return m;
    }
    make(blocks) { const m = NA.bx.build(blocks, this.input, this.output, { seed: 1 }); m.evoBlocks = blocks; m.evoChk = NA.bx.check(blocks, this.input, this.output); return m; }
  }

  // ------------------------------------------------------------------ measures of a population
  // how spread out weights are: the root mean square, over members and weights, of each weight's distance from the mean
  function spreadOf(list) {
    if (list.length < 2) return null;
    const n = list[0].length, mean = new Float64Array(n);
    for (const w of list) for (let i = 0; i < n; i++) mean[i] += w[i] / list.length;
    let s = 0;
    for (const w of list) for (let i = 0; i < n; i++) { const d = w[i] - mean[i]; s += d * d; }
    return Math.sqrt(s / (list.length * n || 1));
  }
  // how spread out behaviours are: their mean distance from their centroid
  function behSpread(behs) {
    const B = behs.filter(Boolean);
    if (!B.length) return null;
    let cx = 0, cy = 0;
    for (const b of B) { cx += b[0] / B.length; cy += b[1] / B.length; }
    let s = 0;
    for (const b of B) s += Math.hypot(b[0] - cx, b[1] - cy);
    return s / B.length;
  }
  // each value's rank from 0 (the lowest) to 1 (the highest)
  function ranks(v) { const n = v.length, r = new Float64Array(n); NA.range(n).sort((a, b) => v[a] - v[b]).forEach((k, i) => { r[k] = n > 1 ? i / (n - 1) : 0; }); return r; }
  const argmax = (o, base, n) => { let a = 0; for (let k = 1; k < n; k++) if (o[base + k] > o[base + a]) a = k; return a; };
  EV.spreadOf = spreadOf; EV.behSpread = behSpread;

  // ------------------------------------------------------------------ the population
  EV.SELECT = { tournament: 'A tournament: the best of 3 at random', truncation: 'The top fifth, at random', rank: 'By rank: the better, the likelier' };
  EV.CROSS = { uniform: 'Weight by weight: each from one parent or the other', neuron: 'Neuron by neuron: each neuron whole, with all its weights, from one parent', none: 'No mixing: a copy of one parent' };
  EV.MUTATE = { plain: 'Plain: one weight in ten nudged at random', safe: 'Safe: each nudge divided by how much the moves depend on that weight' };
  EV.PENALTY = { off: 0, light: 0.0005, strong: 0.003 };
  EV.NOVELTY = { novelty: 'Novelty only', half: 'Half novelty, half reward' };
  EV.GRID = 12;
  EV.KEEP = 400; // records kept for lineages, at most
  EV.ARCHIVE = 1000; // behaviours novelty search keeps, at most (the oldest go first)
  /* T: the trainer (world, the network on show, rng, policy); s: the settings (pop, sigma, select, cross, mutate,
     grow, penalty, newLayer, layerSize; esSigma, esLr for evolution strategies; novK, novMix, novAdd for novelty
     search; cemElite, cemSigma, cemFloor for the cross-entropy method; erlSteps, erlEvery, erlLr for ERL). */
  class Evolution {
    constructor(T, s, blocks) {
      Object.assign(this, { T, kind: T.method, rng: T.rng, world: T.world, gen: 0, history: [], last: null, job: null, nextId: 1, records: new Map() });
      const net = T.net;
      this.io = { nIn: T.inSize, K: T.K };
      this.nets = new Nets(net.input, net.output);
      this.blocks = blocks || net.evoBlocks || [];
      this.size = Math.max(4, Math.round(s.pop) || 24); this.sigma = s.sigma > 0 ? s.sigma : 0.2;
      this.select = EV.SELECT[s.select] ? s.select : 'tournament'; this.cross = EV.CROSS[s.cross] ? s.cross : 'uniform'; this.mutation = EV.MUTATE[s.mutate] ? s.mutate : 'plain';
      // grow and prune: the genetic algorithm's alone (its settings in this.gopt: EV.growOptions)
      this.grow = this.kind === 'ga' && !!s.grow && EV.evolvable(this.blocks, net.input); this.gopt = EV.growOptions(s);
      this.penalty = this.grow ? EV.PENALTY[this.gopt.penalty] : 0; this.newLayer = this.gopt.mode; this.layerSize = this.gopt.size;
      const W = (NA.brl.WORLD[T.worldId] || {});
      this.episodes = (W.gaEpisodes || 2) + (T.ws.random || T.ws.slip || (T.worldId === 'track' && T.ws.track === 'random') ? 2 : 0); this.scale = W.scale || 1;
      const w0 = net.getWeights();
      if (this.kind === 'es') {
        this.theta = Float32Array.from(w0); this.m = new Float32Array(w0.length); this.v = new Float32Array(w0.length); this.t = 0;
        this.esSigma = s.esSigma > 0 ? s.esSigma : 0.05; this.esLr = s.esLr > 0 ? s.esLr : 0.03; this.pairs = Math.max(2, Math.round(this.size / 2));
      } else if (this.kind === 'me') {
        this.archive = new Map(); this.w0 = w0; this.batch = this.size; this.axes = (NA.rlEnvs.WORLDS[T.worldId] || {}).axes || ['style 1', 'style 2'];
      } else if (this.kind === 'cem') {
        // the bell curve over the weights: its middle the network you built, the same spread for every weight
        this.cemSigma = s.cemSigma > 0 ? s.cemSigma : 0.3; this.cemElite = s.cemElite > 0 && s.cemElite < 1 ? s.cemElite : 0.2; this.cemFloor = s.cemFloor >= 0 ? s.cemFloor : 0.05;
        this.mu = Float32Array.from(w0); this.sd = new Float32Array(w0.length).fill(this.cemSigma);
      } else {
        /* the first generation: the network you built and copies of it, each moved at random (the same architecture);
           with grow and prune starting small, a network with no hidden layer (the inputs straight to the outputs)
           and copies of it, so every neuron and layer it ends with is one evolution added */
        const small = this.grow && this.gopt.start === 'small', b0 = small ? [] : this.blocks, s0 = small ? this.nets.get([]).getWeights() : w0;
        // (grow and prune: the family tree of shapes, each shape that appears with the one it came from)
        if (this.grow) this.shapeTree = new Map([[Nets.key(b0), { key: Nets.key(b0), parent: null, gen: 0, units: unitsOf(b0), most: 0, best: -Infinity, last: 0 }]]);
        this.pop = NA.range(this.size).map((k) => { const w = Float32Array.from(s0); if (k) for (let i = 0; i < w.length; i++) w[i] += this.rng.gauss(0, 0.5); return { blocks: b0, w, op: k ? 'a copy moved at random' : small ? 'no hidden layer: the smallest network' : 'the network you built', born: 0, id: this.newId(), parents: [] }; });
        // novelty search: judged by how new its behaviour is (a world that measures none: by its return, as the genetic algorithm)
        this.novel = this.kind === 'novelty' && typeof this.world.behavior === 'function';
        if (this.novel) { this.novArchive = []; this.novK = Math.max(1, Math.round(s.novK) || 10); this.novMix = EV.NOVELTY[s.novMix] ? s.novMix : 'novelty'; this.novAdd = Math.max(1, Math.round(s.novAdd) || 2); this.champ = null; }
        if (this.kind === 'erl') {
          // the DQN: a network like yours (and its target network), learning from every move the population makes
          this.q = T.makeNet(); this.q.setWeights(w0); this.qTarget = T.makeNet(); this.qTarget.setWeights(w0);
          this.buf = []; this.bufNext = 0; this.bufCap = this.io.nIn > 64 ? 5000 : 20000; this.rlUpdates = 0; this.rlLoss = null;
          this.erlSteps = Math.max(1, Math.round(s.erlSteps) || 200); this.erlEvery = Math.max(1, Math.round(s.erlEvery) || 5); this.erlLr = s.erlLr > 0 ? s.erlLr : 0.001; this.gamma = T.gamma;
        }
      }
    }
    newId() { return this.nextId++; }
    // how many weights an architecture has
    weightsOf(blocks) { return EV.weights(EV.sizes(blocks, this.io.nIn, this.io.K)); }
    /* One member's episodes with its best moves, all from the same starts this generation: its fitness (the average
       return), its style (behavior), the path it took and its ending (the first episode), and some of what it saw
       (for safe mutations). ERL keeps every move in its replay memory. */
    play(net, seed, keep) {
      const W = this.world, rng = new NA.RNG(seed), T = this.T, seen = [], mem = this.buf;
      let total = 0, beh = null, trace = null, end = null;
      for (let k = 0; k < this.episodes; k++) {
        let o = W.reset(rng), done = false, steps = 0;
        while (!done) {
          if (keep && k === 0 && seen.length < 32 && steps % 5 === 0) seen.push(Float32Array.from(o));
          const a = T.policy(o, net), t = W.step(a, rng);
          if (mem) this.remember(o, a, t.r, t.done ? null : t.obs);
          total += t.r; o = t.obs; done = t.done; steps++;
        }
        if (k === 0) { beh = W.behavior ? W.behavior() : null; trace = keep && W.trace ? Float32Array.from(W.trace) : null; end = W.end || null; }
        T.envSteps += steps; T.episodes++;
      }
      return { fit: total / this.episodes, beh, trace, end, seen };
    }
    step(deadline) { const k = this.kind; return k === 'es' ? this.stepES(deadline) : k === 'me' ? this.stepME(deadline) : k === 'cem' ? this.stepCEM(deadline) : this.stepGA(deadline); }

    // ---------------------------------------------------------------- a genetic algorithm (and novelty search, ERL)
    stepGA(deadline) {
      const P = this.pop;
      if (!this.job) this.job = { i: 0, seed: 1000 + this.gen, res: [] };
      const J = this.job;
      while (J.i < P.length) {
        const g = P[J.i], net = this.nets.get(g.blocks);
        net.setWeights(g.w);
        J.res[J.i] = this.play(net, J.seed, true);
        J.i++;
        if (deadline && now() > deadline && J.i < P.length) return { partial: true, done: J.i, of: P.length };
      }
      this.job = null;
      // the fitness each is selected by: its return, less the size penalty; novelty search: how new its behaviour is
      const n = P.length, fit = J.res.map((r) => r.fit);
      let adj, nov = null;
      if (this.novel) {
        nov = this.novelty(J.res.map((r) => r.beh));
        if (this.novMix === 'half') { const a = ranks(nov), b = ranks(fit); adj = NA.range(n).map((k) => 0.5 * a[k] + 0.5 * b[k]); } else adj = Array.from(nov);
      } else adj = P.map((g, k) => fit[k] - this.penalty * this.scale * (this.weightsOf(g.blocks) / 100));
      const order = NA.range(n).sort((a, b) => adj[b] - adj[a]), best = P[order[0]];
      let mean = 0; for (const f of fit) mean += f / n;
      // what is shown: the members best first (novelty search: by their return, the novelty each had beside it)
      const shown = this.novel ? NA.range(n).sort((a, b) => fit[b] - fit[a]) : order, same = P.filter((g) => Nets.key(g.blocks) === Nets.key(best.blocks)).map((g) => g.w);
      if (this.novel) this.keepNovel(J.res, nov);
      if (this.kind === 'erl') this.learnRL();
      // (grow and prune: how many shapes the generation has, and each member's activations when they can change)
      const extra = this.kind === 'erl' ? { divW: spreadOf(same), rlLoss: this.rlLoss } : { divW: spreadOf(same) };
      if (this.grow) extra.shapes = new Set(P.map((g) => Nets.key(g.blocks))).size;
      this.record(shown.map((k) => ({ id: P[k].id, parents: P[k].parents, fit: fit[k], adj: adj[k], units: unitsOf(P[k].blocks), weights: this.weightsOf(P[k].blocks), beh: J.res[k].beh, trace: J.res[k].trace, end: J.res[k].end, op: P[k].op, born: P[k].born, nov: nov ? nov[k] : undefined, fromRL: P[k].fromRL || undefined,
        prot: P[k].prot || undefined, acts: this.grow && this.gopt.ops.act ? P[k].blocks.map((b) => b.act || 'relu') : undefined })),
        fit[shown[0]], mean, this.novel ? null : best, extra);
      if (this.grow) this.last.shapes = extra.shapes;
      // (the family tree: how many members each shape has had at most, its best fitness, when it was last seen)
      if (this.shapeTree) { const count = new Map(); P.forEach((g, k) => { const key = Nets.key(g.blocks), q = this.shapeTree.get(key); if (!q) return; count.set(key, (count.get(key) || 0) + 1); q.best = Math.max(q.best, fit[k]); q.last = this.gen; }); for (const [key, c] of count) { const q = this.shapeTree.get(key); q.most = Math.max(q.most, c); } this.bestShape = Nets.key(best.blocks); }
      if (this.novel) this.champion(P[shown[0]], fit[shown[0]]);
      // the next generation: the best two unchanged (op null: passed on as they were), then children
      const next = [best, P[order[1]]].map((g) => Object.assign(this.copy(g), { op: null })), sens = new Map();
      /* Protect new shapes: a new shape usually starts a little worse (its new neuron does nothing yet, its new
         layer is a bottleneck), so the best member of each shape that appeared in the last few generations carries
         on unchanged too, and mutations get time to tune it before it has to beat the rest (a simple form of NEAT's
         speciation). A shape's age is its member's: the generation its shape last changed. */
      if (this.grow && this.gopt.protect) {
        const have = new Set(next.map((g) => Nets.key(g.blocks))), most = Math.max(0, Math.min(EV.PROTECT.most, n - 3));
        for (const k of order) {
          if (next.length >= 2 + most) break;
          const g = P[k], key = Nets.key(g.blocks);
          if (have.has(key) || g.shapeGen === undefined || this.gen - g.shapeGen >= EV.PROTECT.gens) continue;
          have.add(key); next.push(Object.assign(this.copy(g), { op: null, prot: true }));
        }
      }
      while (next.length < n) {
        const ia = this.pick(order, adj), a = P[ia];
        let child = this.copy(a), mixed = false;
        Object.assign(child, { born: this.gen + 1, id: this.newId(), parents: [a.id], fromRL: undefined });
        if (this.cross !== 'none') { const b = P[this.pick(order, adj)]; if (b !== a && Nets.key(a.blocks) === Nets.key(b.blocks)) { child.w = this.mix(a, b); child.parents = [a.id, b.id]; mixed = true; } }
        if (this.mutation === 'safe' && !sens.has(ia)) sens.set(ia, this.sensitivity(a, J.res[ia].seen));
        this.mutate(child.w, this.mutation === 'safe' ? sens.get(ia) : null);
        child.op = mixed ? 'mixed and mutated' : 'mutated';
        // grow and prune: some of the children (a quarter unless set) are born with one change to their shape
        if (this.grow && this.rng.next() < this.gopt.rate) {
          const r = EV.mutateArch(child, this.rng, this.gopt, this.io);
          if (r) {
            const from = Nets.key(child.blocks);
            Object.assign(child, r, { shapeGen: this.gen + 1 });
            const k = Nets.key(child.blocks);
            if (this.shapeTree && !this.shapeTree.has(k) && this.shapeTree.size < 600) this.shapeTree.set(k, { key: k, parent: from, gen: this.gen + 1, units: unitsOf(child.blocks), most: 0, best: -Infinity, last: this.gen + 1, how: r.op || child.op });
          }
        }
        next.push(child);
      }
      // ERL: every few generations the DQN's network takes the place of the worst
      if (this.kind === 'erl' && (this.gen + 1) % this.erlEvery === 0) next[n - 1] = { blocks: this.blocks, w: this.q.getWeights(), op: 'the DQN’s network', born: this.gen + 1, id: this.newId(), parents: [], fromRL: true };
      this.pop = next; this.gen++;
      return { loss: -mean, gradNorm: 0 };
    }
    copy(g) { return { blocks: g.blocks, w: Float32Array.from(g.w), op: g.op, born: g.born, id: g.id, parents: g.parents, fromRL: g.fromRL, shapeGen: g.shapeGen }; }
    // a parent, by the selection chosen (order: the population best first; adj: their fitness)
    pick(order, adj) {
      const n = order.length, rng = this.rng;
      if (this.select === 'truncation') return order[rng.int(Math.max(2, Math.round(n / 5)))];
      if (this.select === 'rank') { let u = rng.next() * ((n * (n + 1)) / 2); for (let r = 0; r < n; r++) { u -= n - r; if (u <= 0) return order[r]; } return order[n - 1]; }
      let b = rng.int(n);
      for (let k = 0; k < 2; k++) { const c = rng.int(n); if (adj[c] > adj[b]) b = c; }
      return b;
    }
    // two parents of one architecture mixed: weight by weight, or each neuron (its row of weights and its bias) whole
    mix(a, b) {
      const w = new Float32Array(a.w.length), rng = this.rng;
      if (this.cross === 'uniform') { for (let i = 0; i < w.length; i++) w[i] = rng.next() < 0.5 ? a.w[i] : b.w[i]; return w; }
      const sz = EV.sizes(a.blocks, this.io.nIn, this.io.K);
      let k = 0;
      if (a.blocks.length && !EV.evolvable(a.blocks, this.nets.input)) { for (let i = 0; i < w.length; i++) w[i] = rng.next() < 0.5 ? a.w[i] : b.w[i]; return w; }
      for (let l = 1; l < sz.length; l++) {
        const U = sz[l], I = sz[l - 1], pick = NA.range(U).map(() => (rng.next() < 0.5 ? a.w : b.w));
        for (let r = 0; r < U; r++) for (let i = 0; i < I; i++) w[k + r * I + i] = pick[r][k + r * I + i];
        for (let r = 0; r < U; r++) w[k + U * I + r] = pick[r][k + U * I + r];
        k += U * I + U;
      }
      return w;
    }
    /* How much the moves depend on each weight (the gradient of each output, summed over observations it met; the
       root of the sum of their squares over the outputs: SM-G-SUM, Lehman and others, 2018). */
    sensitivity(g, seen) {
      if (!seen || !seen.length) return null;
      const net = this.nets.get(g.blocks), B = seen.length, nIn = this.io.nIn, K = this.io.K, nA = this.T.nA, X = new Float32Array(B * nIn), P = net.paramCount, s = new Float32Array(P), gr = new Float32Array(P);
      net.setWeights(g.w);
      seen.forEach((o, b) => X.set(o, b * nIn));
      for (let a = 0; a < nA; a++) {
        net.gradWith(X, B, (out, d) => { for (let b = 0; b < B; b++) d[b * K + a] = 1 / B; });
        net.getGrads(gr);
        for (let i = 0; i < P; i++) s[i] += gr[i] * gr[i];
      }
      for (let i = 0; i < P; i++) s[i] = Math.sqrt(s[i]);
      return s;
    }
    /* One weight in ten nudged: plain, by σ at random; safe, divided by its sensitivity and scaled so the nudges are
       as big on average as plain ones (the weights the moves depend on least move most). */
    mutate(w, sens) {
      const rng = this.rng, sg = this.sigma;
      if (!sens || sens.length !== w.length) { for (let i = 0; i < w.length; i++) if (rng.next() < 0.1) w[i] += rng.gauss(0, sg); return; }
      let mean = 0;
      for (let i = 0; i < sens.length; i++) mean += sens[i];
      const floor = 0.05 * (mean / sens.length) + 1e-6, idx = [], d = [];
      let sum = 0;
      for (let i = 0; i < w.length; i++) if (rng.next() < 0.1) { const v = rng.gauss(0, 1) / (sens[i] + floor); idx.push(i); d.push(v); sum += Math.abs(v); }
      const scale = idx.length ? (sg * 0.7979 * idx.length) / (sum || 1) : 0;
      for (let k = 0; k < idx.length; k++) w[idx[k]] += d[k] * scale;
    }
    /* What a generation leaves behind: its members (best first), the history (with how spread out the members'
       behaviours and weights are: extra adds the method's own), its members' records for lineages, and the network
       on show (the best, g; null: the method shows its own). */
    record(members, best, mean, g, extra) {
      const top = members[0];
      this.last = { gen: this.gen, members, best, mean };
      this.history.push(Object.assign({ gen: this.gen, best, mean, weights: top.weights, units: top.units, adj: top.adj, div: behSpread(members.map((m) => m.beh)), divW: null }, extra));
      if (this.history.length > 2000) this.history = this.history.filter((h, i) => i % 2 === 0 || i === this.history.length - 1);
      for (const m of members) if (m.id && !this.records.has(m.id)) this.records.set(m.id, { id: m.id, gen: this.gen, fit: m.fit, op: m.op || null, units: m.units, parents: m.parents || [] });
      const T = this.T;
      T.lastEp = best; T.retEma = mean; T.genBest = best; T.genMean = mean;
      if (g) this.show(g.blocks, g.w);
      this.keepLineage();
    }
    // the records worth keeping: the ancestors (first parents) of the members just judged, of the network on show and of MAP-Elites' elites
    keepLineage() {
      const R = this.records, keep = new Set(), roots = (this.last ? this.last.members.map((m) => m.id) : []).concat(this.champ ? [this.champ.id] : [], this.archive ? [...this.archive.values()].map((e) => e.id) : []);
      for (const id of roots) { let q = R.get(id); while (q && !keep.has(q.id)) { keep.add(q.id); q = q.parents.length ? R.get(q.parents[0]) : null; } }
      for (const id of [...R.keys()]) if (!keep.has(id)) R.delete(id);
      if (R.size > EV.KEEP) { const root = new Set(roots), old = [...R.values()].filter((q) => !root.has(q.id)).sort((a, b) => a.gen - b.gen); for (const q of old.slice(0, R.size - EV.KEEP)) R.delete(q.id); }
    }
    // the best's id: the network on show (novelty search), MAP-Elites' best elite, or the best of the last generation
    bestId() {
      if (this.champ) return this.champ.id;
      if (this.archive && this.archive.size) { let b = null; for (const e of this.archive.values()) if (!b || e.fit > b.fit) b = e; return b.id; }
      return this.last && this.last.members.length ? this.last.members[0].id : null;
    }
    // the line of first parents that led to a member, the oldest known first
    lineage(id) {
      if (id === undefined || id === null) id = this.bestId();
      const out = [], seen = new Set();
      let q = this.records.get(id);
      while (q && !seen.has(q.id)) { seen.add(q.id); out.push({ id: q.id, gen: q.gen, fit: q.fit, op: q.op, units: q.units }); q = q.parents.length ? this.records.get(q.parents[0]) : null; }
      return out.reverse();
    }
    // the network on show: the same model with new weights, or a model of a new architecture
    show(blocks, w) {
      const T = this.T;
      if (Nets.key(blocks) !== Nets.key(T.net.evoBlocks || this.blocks)) { T.net = this.nets.make(blocks); T.archChanged = (T.archChanged || 0) + 1; }
      T.net.setWeights(w);
    }

    // ---------------------------------------------------------------- novelty search
    // how new each behaviour is: its mean distance to its k nearest among the generation's others and the archive
    novelty(behs) {
      const k = this.novK, A = this.novArchive, n = behs.length, out = new Float64Array(n), d = [];
      for (let i = 0; i < n; i++) {
        const b = behs[i];
        if (!b) continue;
        d.length = 0;
        for (let j = 0; j < n; j++) if (j !== i && behs[j]) d.push(Math.hypot(b[0] - behs[j][0], b[1] - behs[j][1]));
        for (const e of A) d.push(Math.hypot(b[0] - e.beh[0], b[1] - e.beh[1]));
        d.sort((x, y) => x - y);
        const m = Math.min(k, d.length);
        let s = 0;
        for (let q = 0; q < m; q++) s += d[q];
        out[i] = m ? s / m : 0;
      }
      return out;
    }
    // the archive: the most novel behaviours of every generation, so a place once visited stops being new
    keepNovel(res, nov) {
      const top = NA.range(res.length).filter((k) => res[k].beh).sort((a, b) => nov[b] - nov[a]).slice(0, this.novAdd);
      for (const k of top) this.novArchive.push({ beh: [res[k].beh[0], res[k].beh[1]], gen: this.gen });
      if (this.novArchive.length > EV.ARCHIVE) this.novArchive.splice(0, this.novArchive.length - EV.ARCHIVE);
    }
    /* The network on show: novelty search never aims at the reward, so the best by score found so far is kept aside.
       The generation's best by return is scored (its best moves, as the page scores the one on show) and replaces it
       if it scores higher, or as high with a higher return. */
    champion(g, fit) {
      const net = this.nets.get(g.blocks), T = this.T;
      net.setWeights(g.w);
      this.judge = this.judge || NA.rlEnvs.make(T.worldId, T.ws);
      const sc = this.judge.score((o) => T.policy(o, net), new NA.RNG(777)), c = this.champ;
      if (!c || sc > c.score || (sc === c.score && fit > c.fit)) { this.champ = { id: g.id, blocks: g.blocks, w: Float32Array.from(g.w), fit, score: sc, gen: this.gen }; this.show(g.blocks, g.w); }
    }

    // ---------------------------------------------------------------- ERL: a DQN learning from the population's moves
    remember(o, a, r, o2) {
      const e = { o: Float32Array.from(o), a, r, o2: o2 ? Float32Array.from(o2) : null }, B = this.buf;
      if (B.length < this.bufCap) B.push(e); else { B[this.bufNext] = e; this.bufNext = (this.bufNext + 1) % this.bufCap; }
    }
    /* Gradient steps of a DQN on moves drawn from the memory, as builder-rl.js's DQN learns: toward reward + γ × the
       next state's value (double DQN: the network picks the next move, its target network says what it is worth),
       a Huber loss, Adam; the target network refreshed every 500 steps. Returns the average loss. */
    learnRL() {
      const buf = this.buf, B = 32, nIn = this.io.nIn, K = this.io.K, nA = this.T.nA, g = this.gamma, opt = { id: 'adam', lr: this.erlLr, clip: 1 };
      if (buf.length < 64) return null;
      const X = new Float32Array(B * nIn), X2 = new Float32Array(B * nIn), y = new Float64Array(B), batch = new Array(B);
      let total = 0;
      for (let s = 0; s < this.erlSteps; s++) {
        for (let b = 0; b < B; b++) { const q = buf[this.rng.int(buf.length)]; batch[b] = q; X.set(q.o, b * nIn); if (q.o2) X2.set(q.o2, b * nIn); }
        if (g) {
          const q2 = Float64Array.from(this.qTarget.forward(X2, B, false)), pickNext = this.q.forward(X2, B, false);
          for (let b = 0; b < B; b++) y[b] = batch[b].r + (batch[b].o2 ? g * q2[b * K + argmax(pickNext, b * K, nA)] : 0);
        } else for (let b = 0; b < B; b++) y[b] = batch[b].r;
        const r = this.q.stepWith(X, B, (out, d) => {
          let loss = 0;
          for (let b = 0; b < B; b++) { const i = b * K + batch[b].a, err = out[i] - y[b], ae = Math.abs(err); loss += ae <= 1 ? 0.5 * err * err : ae - 0.5; d[i] = Math.max(-1, Math.min(1, err)) / B; }
          return loss / B;
        }, opt);
        total += r.loss;
        if (++this.rlUpdates % 500 === 0) this.qTarget.setWeights(this.q.getWeights());
      }
      this.rlLoss = total / this.erlSteps;
      return this.rlLoss;
    }

    // ---------------------------------------------------------------- evolution strategies
    noise(seed, n) { const r = new NA.RNG(seed), e = new Float32Array(n); for (let i = 0; i < n; i++) e[i] = r.gauss(); return e; }
    stepES(deadline) {
      const n = this.theta.length, N = 2 * this.pairs;
      if (!this.job) this.job = { i: 0, seed: 1000 + this.gen, seeds: NA.range(this.pairs).map(() => 1 + this.rng.int(2e9)), res: [], w: new Float32Array(n), ids: NA.range(N).map(() => this.newId()) };
      const J = this.job, net = this.nets.get(this.blocks);
      while (J.i < N) {
        const e = this.noise(J.seeds[J.i >> 1], n), sg = (J.i & 1 ? -1 : 1) * this.esSigma;
        for (let i = 0; i < n; i++) J.w[i] = this.theta[i] + sg * e[i];
        net.setWeights(J.w);
        J.res[J.i] = this.play(net, J.seed, true);
        J.i++;
        if (deadline && now() > deadline && J.i < N) return { partial: true, done: J.i, of: N };
      }
      this.job = null;
      // fitness shaping: ranks from −½ to ½, so one lucky score cannot run away with the step
      const fit = J.res.map((r) => r.fit), order = NA.range(N).sort((a, b) => fit[a] - fit[b]), u = new Float32Array(N);
      order.forEach((k, r) => { u[k] = r / (N - 1) - 0.5; });
      const g = new Float32Array(n);
      let ee = 0;
      for (let k = 0; k < this.pairs; k++) { const e = this.noise(J.seeds[k], n), c = (u[2 * k] - u[2 * k + 1]) / (N * this.esSigma); for (let i = 0; i < n; i++) { g[i] += c * e[i]; ee += e[i] * e[i]; } }
      // Adam, going uphill
      const b1 = 0.9, b2 = 0.999, t = ++this.t, c1 = 1 - b1 ** t, c2 = 1 - b2 ** t;
      let gn = 0;
      for (let i = 0; i < n; i++) { const gi = g[i]; gn += gi * gi; this.m[i] = b1 * this.m[i] + (1 - b1) * gi; this.v[i] = b2 * this.v[i] + (1 - b2) * gi * gi; this.theta[i] += (this.esLr * (this.m[i] / c1)) / (Math.sqrt(this.v[i] / c2) + 1e-8); }
      let mean = 0; for (const f of fit) mean += f / N;
      const sorted = NA.range(N).sort((a, b) => fit[b] - fit[a]), wn = this.weightsOf(this.blocks), units = unitsOf(this.blocks);
      // (the nudged copies sit around θ, each weight σ·ε from it: their spread is σ times the root mean square of ε)
      this.record(sorted.map((k) => ({ id: J.ids[k], parents: [], fit: fit[k], adj: fit[k], units, weights: wn, beh: J.res[k].beh, trace: J.res[k].trace, end: J.res[k].end, op: k & 1 ? 'nudged one way' : 'nudged the other way', born: this.gen })), fit[sorted[0]], mean, null,
        { divW: this.esSigma * Math.sqrt(ee / (this.pairs * n || 1)) });
      this.T.net.setWeights(this.theta); // on show: the mean of the population, the network that learns
      this.gradNorm = Math.sqrt(gn); this.gen++;
      return { loss: -mean, gradNorm: this.gradNorm };
    }

    // ---------------------------------------------------------------- the cross-entropy method
    stepCEM(deadline) {
      const n = this.mu.length, N = this.size;
      if (!this.job) {
        // a generation drawn from the bell curve: each weight its middle plus its spread times a random number
        const kids = NA.range(N).map(() => { const w = new Float32Array(n); for (let i = 0; i < n; i++) w[i] = this.mu[i] + this.sd[i] * this.rng.gauss(); return w; });
        let spread = 0; for (let i = 0; i < n; i++) spread += this.sd[i] / n;
        this.job = { i: 0, seed: 1000 + this.gen, kids, res: [], ids: kids.map(() => this.newId()), spread };
      }
      const J = this.job, net = this.nets.get(this.blocks);
      while (J.i < N) {
        net.setWeights(J.kids[J.i]);
        J.res[J.i] = this.play(net, J.seed, true);
        J.i++;
        if (deadline && now() > deadline && J.i < N) return { partial: true, done: J.i, of: N };
      }
      this.job = null;
      // the best fifth (the elite) refits the bell curve: its mean, and each weight's spread among them, never below the floor
      const fit = J.res.map((r) => r.fit), order = NA.range(N).sort((a, b) => fit[b] - fit[a]), E = Math.max(2, Math.round(N * this.cemElite));
      const mu = new Float64Array(n), v = new Float64Array(n), f2 = this.cemFloor * this.cemFloor;
      for (let k = 0; k < E; k++) { const w = J.kids[order[k]]; for (let i = 0; i < n; i++) mu[i] += w[i] / E; }
      for (let k = 0; k < E; k++) { const w = J.kids[order[k]]; for (let i = 0; i < n; i++) { const d = w[i] - mu[i]; v[i] += (d * d) / E; } }
      for (let i = 0; i < n; i++) { this.mu[i] = mu[i]; this.sd[i] = Math.sqrt(v[i] + f2); }
      let mean = 0; for (const f of fit) mean += f / N;
      const wn = this.weightsOf(this.blocks), units = unitsOf(this.blocks);
      this.record(order.map((k, r) => ({ id: J.ids[k], parents: [], fit: fit[k], adj: fit[k], units, weights: wn, beh: J.res[k].beh, trace: J.res[k].trace, end: J.res[k].end, op: r < E ? 'drawn; in the elite' : 'drawn', born: this.gen })), fit[order[0]], mean, null,
        { divW: spreadOf(J.kids), spread: J.spread });
      this.T.net.setWeights(this.mu); // on show: the middle of the bell curve
      this.gen++;
      return { loss: -mean, gradNorm: 0 };
    }

    // ---------------------------------------------------------------- MAP-Elites
    cellOf(beh) { const G = EV.GRID; return beh ? Math.min(G - 1, Math.floor(beh[0] * G)) + G * Math.min(G - 1, Math.floor(beh[1] * G)) : -1; }
    stepME(deadline) {
      if (!this.job) {
        // a batch: random networks while the grid is nearly empty, then mutated copies of elites picked at random
        const elites = [...this.archive.values()], kids = NA.range(this.batch).map(() => {
          if (elites.length < this.batch) return { w: Float32Array.from(this.w0, (v) => v + this.rng.gauss(0, 0.5)), parents: [], op: 'random' };
          const e = elites[this.rng.int(elites.length)], w = Float32Array.from(e.w);
          this.mutate(w, null);
          return { w, parents: [e.id], op: 'mutated' };
        });
        for (const k of kids) k.id = this.newId();
        this.job = { i: 0, seed: 1000 + this.gen, kids, res: [] };
      }
      const J = this.job, net = this.nets.get(this.blocks);
      while (J.i < J.kids.length) {
        net.setWeights(J.kids[J.i].w);
        J.res[J.i] = this.play(net, J.seed, true);
        J.i++;
        if (deadline && now() > deadline && J.i < J.kids.length) return { partial: true, done: J.i, of: J.kids.length };
      }
      this.job = null;
      let added = 0, mean = 0;
      J.res.forEach((r, k) => {
        mean += r.fit / J.res.length;
        const c = this.cellOf(r.beh), cur = this.archive.get(c);
        if (c >= 0 && (!cur || r.fit > cur.fit)) { this.archive.set(c, { w: J.kids[k].w, fit: r.fit, beh: r.beh, trace: r.trace, end: r.end, gen: this.gen, id: J.kids[k].id }); added++; }
      });
      const all = [...this.archive.values()].sort((a, b) => b.fit - a.fit), top = all[0], wn = this.weightsOf(this.blocks), units = unitsOf(this.blocks);
      const order = NA.range(J.res.length).sort((a, b) => J.res[b].fit - J.res[a].fit);
      this.added = added;
      this.record(order.map((k) => ({ id: J.kids[k].id, parents: J.kids[k].parents, fit: J.res[k].fit, adj: J.res[k].fit, units, weights: wn, beh: J.res[k].beh, trace: J.res[k].trace, end: J.res[k].end, op: J.kids[k].op, born: this.gen })), top ? top.fit : 0, mean, null,
        { divW: spreadOf(J.kids.map((q) => q.w)), filled: this.archive.size });
      this.last.filled = this.archive.size; this.last.added = added;
      if (top) this.T.net.setWeights(top.w); // on show: the best of all the styles
      this.gen++;
      return { loss: -mean, gradNorm: 0 };
    }
    // the elite of a cell, for watching it
    elite(c) { return this.archive ? this.archive.get(c) || null : null; }
  }
  EV.Evolution = Evolution;
  EV.Nets = Nets;

  // ------------------------------------------------------------------ grow and prune while it learns
  /* For problems with examples and answers: a few networks (6) learn side by side, each taking a step of your
     optimiser on the same batch. Every generation (a number of steps) each is judged on the validation examples
     by its loss there (finer than accuracy, so a network that has just grown is not tied with its parent), plus
     the size penalty, and the worse half are replaced by copies of the better half, weights and all, each changed
     by one mutation of its architecture: a neuron more or less, or a new layer (of one neuron by default: a
     bottleneck the copy then learns to use, and widens neuron by neuron). Population-based training, with the
     architecture evolving instead of the learning rate. The best network so far is the one on show. */
  EV.PENALTY_EX = { off: 0, light: 0.004, strong: 0.015 }; // added to the validation loss for every 100 weights
  class Grower {
    /* o: blocks, input, output, seed, init, judge(model) → { score, loss }, and the settings: opts (EV.growOptions(train
       settings, true): changes, odds, limits, start, members, every) or, as before, size, every, penalty, newLayer,
       layerSize. Starting small, every member starts with no hidden layer (the network you built is not one of them). */
    constructor(o) {
      const opt = o.opts || EV.growOptions({ growPenalty: o.penalty, growNew: o.newLayer, growWidth: o.layerSize, growSize: o.size, growEvery: o.every }, true);
      Object.assign(this, { input: o.input, output: o.output, init: o.init, every: Math.max(10, opt.every), judge: o.judge, rng: new NA.RNG((o.seed || 1) * 7907 + 3), t: 0, gen: 0, history: [], gopt: opt });
      this.penalty = EV.PENALTY_EX[opt.penalty] ?? EV.PENALTY_EX.light; this.newLayer = opt.mode; this.layerSize = opt.size;
      this.io = { nIn: o.input.dim, K: o.output.kind === 'classes' ? o.output.k : o.output.k || 1 };
      const n = Math.max(2, opt.members), small = opt.start === 'small', b0 = small ? [] : o.blocks;
      this.members = NA.range(n).map((k) => ({ blocks: b0, model: k === 0 && o.model && !small ? o.model : this.make(b0, (o.seed || 1) * 31 + k), born: 0, op: small ? 'no hidden layer' : null, score: null, fit: null }));
      this.best = this.members[0];
    }
    make(blocks, seed) { const m = NA.bx.build(blocks, this.input, this.output, { seed, init: this.init }); m.evoBlocks = blocks; m.evoChk = NA.bx.check(blocks, this.input, this.output); return m; }
    weightsOf(blocks) { return EV.weights(EV.sizes(blocks, this.io.nIn, this.io.K)); }
    // every member learns from the same batch; the answer is the one on show's
    step(X, Y, B, loss, opt) {
      let out = null;
      for (const m of this.members) { const r = m.model.step(X, Y, B, loss, opt); if (m === this.best) out = r; }
      this.t++;
      if (this.t % this.every === 0) this.generation();
      return out;
    }
    generation() {
      const M = this.members;
      for (const m of M) { const j = this.judge(m.model); m.score = j.score; m.loss = j.loss; m.weights = this.weightsOf(m.blocks); m.fit = -j.loss - this.penalty * (m.weights / 100); }
      const order = M.slice().sort((a, b) => b.fit - a.fit), half = Math.ceil(M.length / 2);
      this.best = order[0];
      this.history.push({ gen: this.gen, step: this.t, score: this.best.score, loss: this.best.loss, weights: this.best.weights, units: unitsOf(this.best.blocks), members: order.map((m) => ({ score: m.score, loss: m.loss, weights: m.weights, units: unitsOf(m.blocks), op: m.op })) });
      // the worse half: copies of the better half, each with one change to its architecture
      for (let k = half; k < order.length; k++) {
        const parent = order[k - half], g = { blocks: parent.blocks, w: parent.model.getWeights() }, r = EV.mutateArch(g, this.rng, this.gopt, this.io) || g;
        const child = order[k];
        child.blocks = r.blocks; child.model = this.make(r.blocks, 1); child.model.setWeights(r.w); child.op = r.op || 'a copy'; child.born = this.gen + 1; child.score = null;
      }
      this.gen++;
    }
  }
  EV.Grower = Grower;
  NA.bevo = EV;
})(typeof globalThis !== 'undefined' && globalThis.NA ? globalThis.NA : window.NA);
