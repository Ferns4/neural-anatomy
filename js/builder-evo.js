/* builder-evo.js — evolution for the networks of the model builder's "Learn from rewards" (builder-rl.js hands
   it the world and the network you built). Three methods:
   · a genetic algorithm: a population of networks plays, each judged by the return of its best moves; the best
     few pass on unchanged and the rest are children of parents picked by a tournament (the best of 3), from the
     top fifth, or by rank; mixed weight by weight, neuron by neuron, or not at all; mutated by plain random
     nudges to one weight in ten, or by "safe" ones, each nudge divided by how much the moves depend on that
     weight. With grow and prune on, a child may also gain or lose a neuron or a whole layer: a new neuron starts
     with outgoing weights of 0 and a new layer passing its input straight through (or at random), so neither
     changes what the network does at first, and a removed layer is folded into the next. A size penalty (the
     fitness less a little for every 100 weights) keeps them small.
   · evolution strategies: one network, nudged in many random directions at once; how each nudged copy does says
     which way is uphill (an estimate of the gradient, with no backpropagation), and Adam takes a step that way.
   · MAP-Elites: a grid of 12 × 12 styles of play (two numbers each world measures, such as average speed and
     which side of the track it keeps); each new network goes to the cell of its style and stays if it beats the
     one there, so it finds the best of every style rather than one best.
   Every member is judged on the same starts, a few members per call (step(deadline)), so a big world never
   stalls the page; and every member keeps the path it took, so a whole generation can be shown at once. Runs in
   the page and in node. */
(function (NA) {
  'use strict';
  const EV = {};
  const now = () => (typeof performance !== 'undefined' ? performance.now() : Date.now());

  // ------------------------------------------------------------------ architectures that grow and shrink
  EV.LIMITS = { layers: 6, units: 48 };
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
  /* A neuron more in hidden layer h: its incoming weights at random, its outgoing ones 0, so the network computes
     exactly what it did, and later mutations can put the new neuron to work. */
  function addNeuron(g, h, rng, io) {
    const L = split(EV.sizes(g.blocks, io.nIn, io.K), g.w), q = L[h], nx = L[h + 1], I = q.I, U = q.U;
    if (U >= EV.LIMITS.units) return null;
    const W = new Float32Array((U + 1) * I), b = new Float32Array(U + 1), W2 = new Float32Array(nx.U * (U + 1)), sd = Math.sqrt(2 / I);
    W.set(q.W); b.set(q.b);
    for (let i = 0; i < I; i++) W[U * I + i] = rng.gauss(0, sd);
    for (let r = 0; r < nx.U; r++) W2.set(nx.W.subarray(r * U, r * U + U), r * (U + 1));
    L[h] = { W, b, U: U + 1, I }; L[h + 1] = { W: W2, b: nx.b, U: nx.U, I: U + 1 };
    return { blocks: g.blocks.map((x, i) => (i === h ? withUnits(x, U + 1) : x)), w: join(L), op: `+1 neuron in layer ${h + 1}` };
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
  /* A new layer at position p (0: straight after the input), as wide as what it reads: its weights the identity, so
     it passes its input straight through (exactly, after a ReLU layer), or at random. */
  function addLayer(g, p, mode, rng, io) {
    if (g.blocks.length >= EV.LIMITS.layers) return null;
    const sz = EV.sizes(g.blocks, io.nIn, io.K), U = sz[p];
    if (U > EV.LIMITS.units) return null;
    const L = split(sz, g.w), W = new Float32Array(U * U), b = new Float32Array(U);
    if (mode === 'random') { const sd = Math.sqrt(2 / U); for (let i = 0; i < W.length; i++) W[i] = rng.gauss(0, sd); }
    else for (let i = 0; i < U; i++) W[i * U + i] = 1;
    L.splice(p, 0, { W, b, U, I: U });
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
  // one change to the architecture: a neuron more (most often) or less, a layer more or less
  EV.mutateArch = function (g, rng, mode, io) {
    const n = g.blocks.length, ops = [['neuron+', n ? 3 : 0], ['neuron-', n ? 2 : 0], ['layer+', n < EV.LIMITS.layers ? 1.4 : 0], ['layer-', n ? 1 : 0]];
    const total = ops.reduce((s, o) => s + o[1], 0);
    for (let tries = 0; tries < 8; tries++) {
      let u = rng.next() * total, op = ops[0][0];
      for (const [name, wgt] of ops) { u -= wgt; if (u <= 0) { op = name; break; } }
      const r = op === 'neuron+' ? addNeuron(g, rng.int(n), rng, io) : op === 'neuron-' ? (() => { const h = rng.int(n); return removeNeuron(g, h, weakest(g, h, io), io); })()
        : op === 'layer+' ? addLayer(g, rng.int(n + 1), mode, rng, io) : removeLayer(g, rng.int(n), io);
      if (r) return r;
    }
    return null;
  };
  Object.assign(EV, { split, join, addNeuron, removeNeuron, addLayer, removeLayer, weakest });

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

  // ------------------------------------------------------------------ the population
  EV.SELECT = { tournament: 'A tournament: the best of 3 at random', truncation: 'The top fifth, at random', rank: 'By rank: the better, the likelier' };
  EV.CROSS = { uniform: 'Weight by weight: each from one parent or the other', neuron: 'Neuron by neuron: each neuron whole, with all its weights, from one parent', none: 'No mixing: a copy of one parent' };
  EV.MUTATE = { plain: 'Plain: one weight in ten nudged at random', safe: 'Safe: each nudge divided by how much the moves depend on that weight' };
  EV.PENALTY = { off: 0, light: 0.0005, strong: 0.003 };
  EV.GRID = 12;
  /* T: the trainer (world, the network on show, rng, policy); s: the settings (pop, sigma, select, cross, mutate,
     grow, penalty, newLayer; esSigma, esLr for evolution strategies). */
  class Evolution {
    constructor(T, s, blocks) {
      Object.assign(this, { T, kind: T.method, rng: T.rng, world: T.world, gen: 0, history: [], last: null, job: null });
      const net = T.net;
      this.io = { nIn: T.inSize, K: T.K };
      this.nets = new Nets(net.input, net.output);
      this.blocks = blocks || net.evoBlocks || [];
      this.size = Math.max(4, Math.round(s.pop) || 24); this.sigma = s.sigma > 0 ? s.sigma : 0.2;
      this.select = EV.SELECT[s.select] ? s.select : 'tournament'; this.cross = EV.CROSS[s.cross] ? s.cross : 'uniform'; this.mutation = EV.MUTATE[s.mutate] ? s.mutate : 'plain';
      this.grow = !!s.grow && EV.evolvable(this.blocks, net.input); this.penalty = this.grow ? EV.PENALTY[s.penalty] ?? EV.PENALTY.light : 0; this.newLayer = s.newLayer === 'random' ? 'random' : 'identity';
      const W = (NA.brl.WORLD[T.worldId] || {});
      this.episodes = (W.gaEpisodes || 2) + (T.ws.random || T.ws.slip || (T.worldId === 'track' && T.ws.track === 'random') ? 2 : 0); this.scale = W.scale || 1;
      const w0 = net.getWeights();
      if (this.kind === 'es') {
        this.theta = Float32Array.from(w0); this.m = new Float32Array(w0.length); this.v = new Float32Array(w0.length); this.t = 0;
        this.esSigma = s.esSigma > 0 ? s.esSigma : 0.05; this.esLr = s.esLr > 0 ? s.esLr : 0.03; this.pairs = Math.max(2, Math.round(this.size / 2));
      } else if (this.kind === 'me') {
        this.archive = new Map(); this.w0 = w0; this.batch = this.size; this.axes = (NA.rlEnvs.WORLDS[T.worldId] || {}).axes || ['style 1', 'style 2'];
      } else {
        // the first generation: the network you built and copies of it, each moved at random (the same architecture)
        this.pop = NA.range(this.size).map((k) => { const w = Float32Array.from(w0); if (k) for (let i = 0; i < w.length; i++) w[i] += this.rng.gauss(0, 0.5); return { blocks: this.blocks, w, op: null, born: 0 }; });
      }
    }
    // how many weights an architecture has
    weightsOf(blocks) { return EV.weights(EV.sizes(blocks, this.io.nIn, this.io.K)); }
    /* One member's episodes with its best moves, all from the same starts this generation: its fitness (the average
       return), its style (behavior), the path it took and its ending (the first episode), and some of what it saw
       (for safe mutations). */
    play(net, seed, keep) {
      const W = this.world, rng = new NA.RNG(seed), T = this.T, seen = [];
      let total = 0, beh = null, trace = null, end = null;
      for (let k = 0; k < this.episodes; k++) {
        let o = W.reset(rng), done = false, steps = 0;
        while (!done) {
          if (keep && k === 0 && seen.length < 32 && steps % 5 === 0) seen.push(Float32Array.from(o));
          const t = W.step(T.policy(o, net), rng);
          total += t.r; o = t.obs; done = t.done; steps++;
        }
        if (k === 0) { beh = W.behavior ? W.behavior() : null; trace = keep && W.trace ? Float32Array.from(W.trace) : null; end = W.end || null; }
        T.envSteps += steps; T.episodes++;
      }
      return { fit: total / this.episodes, beh, trace, end, seen };
    }
    step(deadline) { return this.kind === 'es' ? this.stepES(deadline) : this.kind === 'me' ? this.stepME(deadline) : this.stepGA(deadline); }

    // ---------------------------------------------------------------- a genetic algorithm
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
      // the fitness each is selected by: its return, less the size penalty
      const n = P.length, fit = J.res.map((r) => r.fit), adj = P.map((g, k) => fit[k] - this.penalty * this.scale * (this.weightsOf(g.blocks) / 100));
      const order = NA.range(n).sort((a, b) => adj[b] - adj[a]), best = P[order[0]];
      let mean = 0; for (const f of fit) mean += f / n;
      this.record(order.map((k) => ({ fit: fit[k], adj: adj[k], units: unitsOf(P[k].blocks), weights: this.weightsOf(P[k].blocks), beh: J.res[k].beh, trace: J.res[k].trace, end: J.res[k].end, op: P[k].op, born: P[k].born })), fit[order[0]], mean, best);
      // the next generation: the best two unchanged, then children
      const next = [this.copy(best), this.copy(P[order[1]])], sens = new Map();
      while (next.length < n) {
        const ia = this.pick(order, adj), a = P[ia];
        let child = this.copy(a);
        child.born = this.gen + 1; child.op = null;
        if (this.cross !== 'none') { const b = P[this.pick(order, adj)]; if (b !== a && Nets.key(a.blocks) === Nets.key(b.blocks)) child.w = this.mix(a, b); }
        if (this.mutation === 'safe' && !sens.has(ia)) sens.set(ia, this.sensitivity(a, J.res[ia].seen));
        this.mutate(child.w, this.mutation === 'safe' ? sens.get(ia) : null);
        if (this.grow && this.rng.next() < 0.25) { const r = EV.mutateArch(child, this.rng, this.newLayer, this.io); if (r) Object.assign(child, r); }
        next.push(child);
      }
      this.pop = next; this.gen++;
      return { loss: -mean, gradNorm: 0 };
    }
    copy(g) { return { blocks: g.blocks, w: Float32Array.from(g.w), op: g.op, born: g.born }; }
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
    // what a generation leaves behind: its members (best first), the history, and the network on show (the best)
    record(members, best, mean, g) {
      const top = members[0];
      this.last = { gen: this.gen, members, best, mean };
      this.history.push({ gen: this.gen, best, mean, weights: top.weights, units: top.units, adj: top.adj });
      if (this.history.length > 2000) this.history = this.history.filter((h, i) => i % 2 === 0 || i === this.history.length - 1);
      const T = this.T;
      T.lastEp = best; T.retEma = mean; T.genBest = best; T.genMean = mean;
      if (g) this.show(g.blocks, g.w);
    }
    // the network on show: the same model with new weights, or a model of a new architecture
    show(blocks, w) {
      const T = this.T;
      if (Nets.key(blocks) !== Nets.key(T.net.evoBlocks || this.blocks)) { T.net = this.nets.make(blocks); T.archChanged = (T.archChanged || 0) + 1; }
      T.net.setWeights(w);
    }

    // ---------------------------------------------------------------- evolution strategies
    noise(seed, n) { const r = new NA.RNG(seed), e = new Float32Array(n); for (let i = 0; i < n; i++) e[i] = r.gauss(); return e; }
    stepES(deadline) {
      const n = this.theta.length, N = 2 * this.pairs;
      if (!this.job) this.job = { i: 0, seed: 1000 + this.gen, seeds: NA.range(this.pairs).map(() => 1 + this.rng.int(2e9)), res: [], w: new Float32Array(n) };
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
      for (let k = 0; k < this.pairs; k++) { const e = this.noise(J.seeds[k], n), c = (u[2 * k] - u[2 * k + 1]) / (N * this.esSigma); for (let i = 0; i < n; i++) g[i] += c * e[i]; }
      // Adam, going uphill
      const b1 = 0.9, b2 = 0.999, t = ++this.t, c1 = 1 - b1 ** t, c2 = 1 - b2 ** t;
      let gn = 0;
      for (let i = 0; i < n; i++) { const gi = g[i]; gn += gi * gi; this.m[i] = b1 * this.m[i] + (1 - b1) * gi; this.v[i] = b2 * this.v[i] + (1 - b2) * gi * gi; this.theta[i] += (this.esLr * (this.m[i] / c1)) / (Math.sqrt(this.v[i] / c2) + 1e-8); }
      let mean = 0; for (const f of fit) mean += f / N;
      const sorted = NA.range(N).sort((a, b) => fit[b] - fit[a]), wn = this.weightsOf(this.blocks), units = unitsOf(this.blocks);
      this.record(sorted.map((k) => ({ fit: fit[k], adj: fit[k], units, weights: wn, beh: J.res[k].beh, trace: J.res[k].trace, end: J.res[k].end })), fit[sorted[0]], mean, null);
      this.T.net.setWeights(this.theta); // on show: the mean of the population, the network that learns
      this.gradNorm = Math.sqrt(gn); this.gen++;
      return { loss: -mean, gradNorm: this.gradNorm };
    }

    // ---------------------------------------------------------------- MAP-Elites
    cellOf(beh) { const G = EV.GRID; return beh ? Math.min(G - 1, Math.floor(beh[0] * G)) + G * Math.min(G - 1, Math.floor(beh[1] * G)) : -1; }
    stepME(deadline) {
      if (!this.job) {
        // a batch: random networks while the grid is nearly empty, then mutated copies of elites picked at random
        const elites = [...this.archive.values()], kids = NA.range(this.batch).map(() => {
          const w = elites.length < this.batch ? Float32Array.from(this.w0, (v) => v + this.rng.gauss(0, 0.5)) : Float32Array.from(elites[this.rng.int(elites.length)].w);
          if (elites.length >= this.batch) this.mutate(w, null);
          return w;
        });
        this.job = { i: 0, seed: 1000 + this.gen, kids, res: [] };
      }
      const J = this.job, net = this.nets.get(this.blocks);
      while (J.i < J.kids.length) {
        net.setWeights(J.kids[J.i]);
        J.res[J.i] = this.play(net, J.seed, true);
        J.i++;
        if (deadline && now() > deadline && J.i < J.kids.length) return { partial: true, done: J.i, of: J.kids.length };
      }
      this.job = null;
      let added = 0, mean = 0;
      J.res.forEach((r, k) => {
        mean += r.fit / J.res.length;
        const c = this.cellOf(r.beh), cur = this.archive.get(c);
        if (c >= 0 && (!cur || r.fit > cur.fit)) { this.archive.set(c, { w: J.kids[k], fit: r.fit, beh: r.beh, trace: r.trace, end: r.end, gen: this.gen }); added++; }
      });
      const all = [...this.archive.values()].sort((a, b) => b.fit - a.fit), top = all[0], wn = this.weightsOf(this.blocks), units = unitsOf(this.blocks);
      const order = NA.range(J.res.length).sort((a, b) => J.res[b].fit - J.res[a].fit);
      this.added = added;
      this.record(order.map((k) => ({ fit: J.res[k].fit, adj: J.res[k].fit, units, weights: wn, beh: J.res[k].beh, trace: J.res[k].trace, end: J.res[k].end })), top ? top.fit : 0, mean, null);
      this.last.filled = this.archive.size; this.last.added = added;
      this.history[this.history.length - 1].filled = this.archive.size;
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
     the size penalty, and the worse half are replaced by copies of the better half,
     weights and all, each changed by one mutation of its architecture: a neuron or a layer more or less, the new
     ones not changing what it computes. Population-based training, with the architecture evolving instead of the
     learning rate. The best network so far is the one on show. */
  EV.PENALTY_EX = { off: 0, light: 0.004, strong: 0.015 }; // added to the validation loss for every 100 weights
  class Grower {
    // o: blocks, input, output, seed, init, size, every (steps a generation), penalty, newLayer, judge(model) → { score, loss }
    constructor(o) {
      Object.assign(this, { input: o.input, output: o.output, init: o.init, every: Math.max(10, o.every || 150), judge: o.judge, rng: new NA.RNG((o.seed || 1) * 7907 + 3), t: 0, gen: 0, history: [] });
      this.penalty = EV.PENALTY_EX[o.penalty] ?? EV.PENALTY_EX.light; this.newLayer = o.newLayer === 'random' ? 'random' : 'identity';
      this.io = { nIn: o.input.dim, K: o.output.kind === 'classes' ? o.output.k : o.output.k || 1 };
      const n = Math.max(2, o.size || 6);
      this.members = NA.range(n).map((k) => ({ blocks: o.blocks, model: k === 0 && o.model ? o.model : this.make(o.blocks, (o.seed || 1) * 31 + k), born: 0, op: null, score: null, fit: null }));
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
        const parent = order[k - half], g = { blocks: parent.blocks, w: parent.model.getWeights() }, r = EV.mutateArch(g, this.rng, this.newLayer, this.io) || g;
        const child = order[k];
        child.blocks = r.blocks; child.model = this.make(r.blocks, 1); child.model.setWeights(r.w); child.op = r.op || 'a copy'; child.born = this.gen + 1; child.score = null;
      }
      this.gen++;
    }
  }
  EV.Grower = Grower;
  NA.bevo = EV;
})(typeof globalThis !== 'undefined' && globalThis.NA ? globalThis.NA : window.NA);
