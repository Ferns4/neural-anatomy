/* builder-neat.js — NEAT (NeuroEvolution of Augmenting Topologies, Stanley and Miikkulainen, 2002) for the model
   builder's worlds that learn from rewards. Each member is a genome: its neurons (the inputs, a bias, the outputs and
   any hidden ones it has grown) and its connections, each with a weight, on or off, and an innovation number that
   says when that connection first appeared anywhere. Every network starts with no hidden neuron, every input wired to
   every output; mutations nudge weights, add a connection between two neurons (never making a loop), or add a neuron
   by splitting a connection in two. Children of two parents line up their genes by innovation number. Members are
   grouped into species by how much their genes differ, and share their fitness within their species (each member's
   fitness divided by its species' size), so a new structure that starts out worse has room to improve before it has
   to beat the rest; a species that stops improving for 15 generations dies out. The networks are not layers, so the
   page draws them itself: The network it grew, and Species over time. NE.Neat (the method's engine, used by
   builder-rl.js), NE.draw (the two views). */
(function (NA) {
  'use strict';
  const NE = {};
  const { clamp, fmt, fmtInt, theme } = NA;
  const now = () => (typeof performance !== 'undefined' ? performance.now() : Date.now());
  const C1 = 1, C2 = 1, C3 = 0.4, STALE = 15, TARGET = 10;

  // ------------------------------------------------------------------ a genome's network
  /* What a genome computes: its neurons in an order where every connection runs forward (inputs and the bias first,
     then the hidden ones, then the outputs), tanh on the hidden ones, the outputs as they are (the move picked is the
     largest). net(g) is cached on the genome. */
  function compile(g) {
    if (g.net) return g.net;
    const on = g.conns.filter((c) => c.on), into = new Map();
    for (const c of on) { if (!into.has(c.b)) into.set(c.b, []); into.get(c.b).push(c); }
    const order = [], seen = new Set(g.nodes.filter((n) => n.kind === 'in' || n.kind === 'bias').map((n) => n.id));
    // (Kahn's order over the hidden and output neurons: a neuron once everything feeding it is known)
    const rest = g.nodes.filter((n) => n.kind === 'hid' || n.kind === 'out').map((n) => n.id);
    let left = rest.slice();
    for (let guard = 0; left.length && guard < 1000; guard++) {
      const next = [];
      for (const id of left) { const ins = into.get(id) || []; if (ins.every((c) => seen.has(c.a))) { order.push(id); seen.add(id); } else next.push(id); }
      if (next.length === left.length) { order.push(...next); break; } // (a loop cannot happen; if it did, the rest in any order)
      left = next;
    }
    const kind = new Map(g.nodes.map((n) => [n.id, n.kind]));
    g.net = { order, into, kind };
    return g.net;
  }
  function forward(g, x, out, nIn, nOut) {
    const N = compile(g), v = g.vals || (g.vals = new Map());
    v.clear();
    for (let i = 0; i < nIn; i++) v.set(i, x[i]);
    v.set(nIn, 1);
    for (const id of N.order) {
      let s = 0;
      for (const c of N.into.get(id) || []) s += c.w * (v.get(c.a) || 0);
      v.set(id, N.kind.get(id) === 'hid' ? Math.tanh(s) : s);
    }
    for (let k = 0; k < nOut; k++) out[k] = v.get(nIn + 1 + k) || 0;
    return out;
  }
  // a genome as a network the page can play and look into: forward(X, B) gives B rows of outputs
  NE.adapter = (g, nIn, nOut) => ({ neat: true, paramCount: g.conns.filter((c) => c.on).length, forward(X, B = 1) { const out = new Float64Array(B * nOut), row = new Float64Array(nOut); for (let b = 0; b < B; b++) { forward(g, X.subarray(b * nIn, b * nIn + nIn), row, nIn, nOut); out.set(row, b * nOut); } return out; } });

  // ------------------------------------------------------------------ the engine
  class Neat {
    constructor(T, s) {
      Object.assign(this, { T, rng: T.rng, world: T.world, gen: 0, history: [], last: null, job: null, grow: false, species: [], speciesHist: [], nextSp: 1 });
      this.nIn = T.inSize; this.nOut = T.nA; this.size = Math.max(12, Math.round(s.pop) || 96); this.sigma = s.sigma > 0 ? s.sigma : 0.5; this.ct = 3;
      const W = NA.brl.WORLD[T.worldId] || {};
      this.episodes = (W.gaEpisodes || 2) + (T.ws.random || T.ws.slip ? 2 : 0);
      // the innovation numbers: one for each connection (from, to) ever made; one hidden neuron id for each connection ever split
      this.inno = new Map(); this.split = new Map(); this.nextInno = 0; this.nextNode = this.nIn + 1 + this.nOut;
      const nodes = [...NA.range(this.nIn).map((i) => ({ id: i, kind: 'in' })), { id: this.nIn, kind: 'bias' }, ...NA.range(this.nOut).map((k) => ({ id: this.nIn + 1 + k, kind: 'out' }))];
      this.pop = NA.range(this.size).map(() => {
        const conns = [];
        for (let a = 0; a <= this.nIn; a++) for (let k = 0; k < this.nOut; k++) conns.push({ a, b: this.nIn + 1 + k, w: this.rng.uniform(-1, 1), on: true, inn: this.innov(a, this.nIn + 1 + k) });
        return { nodes: nodes.map((n) => ({ ...n })), conns };
      });
      this.champ = this.pop[0];
      T.neat = this;
    }
    innov(a, b) { const k = `${a}>${b}`; if (!this.inno.has(k)) this.inno.set(k, this.nextInno++); return this.inno.get(k); }
    // the network on show: the best of the last generation
    outputs(o) { return forward(this.champ, o, new Float64Array(this.nOut), this.nIn, this.nOut); }
    act(o) { const out = this.outputs(o); let b = 0; for (let k = 1; k < this.nOut; k++) if (out[k] > out[b]) b = k; return b; }
    adapter() { return this._ad && this._ad.g === this.champ ? this._ad.a : (this._ad = { g: this.champ, a: NE.adapter(this.champ, this.nIn, this.nOut) }).a; }
    play(g, seed) {
      const W = this.world, rng = new NA.RNG(seed), T = this.T, out = new Float64Array(this.nOut);
      let total = 0, beh = null, trace = null, end = null;
      for (let k = 0; k < this.episodes; k++) {
        let o = W.reset(rng), done = false, steps = 0;
        while (!done) {
          forward(g, o, out, this.nIn, this.nOut);
          let a = 0; for (let j = 1; j < this.nOut; j++) if (out[j] > out[a]) a = j;
          const t = W.step(a, rng);
          total += t.r; o = t.obs; done = t.done; steps++;
        }
        if (k === 0) { beh = W.behavior ? W.behavior() : null; trace = W.trace ? Float32Array.from(W.trace) : null; end = W.end || null; }
        T.envSteps += steps; T.episodes++;
      }
      return { fit: total / this.episodes, beh, trace, end };
    }
    // how different two genomes are: excess and disjoint genes, and the weights of the matching ones
    distance(x, y) {
      const A = new Map(x.conns.map((c) => [c.inn, c])), B = new Map(y.conns.map((c) => [c.inn, c])), maxA = Math.max(-1, ...A.keys()), maxB = Math.max(-1, ...B.keys());
      let E = 0, D = 0, Wd = 0, M = 0;
      for (const [i, c] of A) { const d = B.get(i); if (d) { Wd += Math.abs(c.w - d.w); M++; } else if (i > maxB) E++; else D++; }
      for (const i of B.keys()) if (!A.has(i)) { if (i > maxA) E++; else D++; }
      const N = Math.max(A.size, B.size) < 20 ? 1 : Math.max(A.size, B.size);
      return (C1 * E + C2 * D) / N + C3 * (M ? Wd / M : 0);
    }
    step(deadline) {
      const P = this.pop;
      if (!this.job) this.job = { i: 0, seed: 1000 + this.gen, res: [] };
      const J = this.job;
      while (J.i < P.length) {
        J.res[J.i] = this.play(P[J.i], J.seed); J.i++;
        if (deadline && now() > deadline && J.i < P.length) return { partial: true, done: J.i, of: P.length };
      }
      this.job = null;
      const n = P.length, fit = J.res.map((r) => r.fit);
      P.forEach((g, k) => { g.fit = fit[k]; });
      this.speciate();
      // shared fitness: each member's (shifted to be above 0) divided by the size of its species
      const lo = Math.min(...fit), order = NA.range(n).sort((a, b) => fit[b] - fit[a]);
      let mean = 0; for (const f of fit) mean += f / n;
      for (const S of this.species) {
        S.members.sort((a, b) => b.fit - a.fit);
        const top = S.members[0].fit;
        if (top > S.best + 1e-9) { S.best = top; S.since = 0; } else S.since++;
        S.adj = S.members.reduce((a, g) => a + (g.fit - lo + 1e-3) / S.members.length, 0);
      }
      const best = P[order[0]];
      this.champ = best;
      // the views: the members best first (as the genetic algorithm's), the species' sizes this generation
      this.last = { gen: this.gen, members: order.map((k) => ({ fit: fit[k], adj: fit[k], beh: J.res[k].beh, trace: J.res[k].trace, end: J.res[k].end, units: [P[k].nodes.filter((q) => q.kind === 'hid').length], weights: P[k].conns.filter((c) => c.on).length, sp: P[k].sp })), best: fit[order[0]], mean };
      this.history.push({ gen: this.gen, best: fit[order[0]], mean, weights: this.last.members[0].weights, units: this.last.members[0].units, div: null, divW: null, species: this.species.length });
      this.speciesHist.push({ gen: this.gen, sizes: this.species.map((S) => ({ id: S.id, n: S.members.length, best: S.members[0].fit, born: S.born, from: S.from })) });
      if (this.history.length > 2000) this.history = this.history.filter((h, i) => i % 2 === 0 || i === this.history.length - 1);
      if (this.speciesHist.length > 400) this.speciesHist.shift();
      const T = this.T;
      T.lastEp = this.last.best; T.retEma = mean; T.genBest = this.last.best; T.genMean = mean;
      this.reproduce(best);
      this.gen++;
      return { loss: -mean, gradNorm: 0 };
    }
    // each member into the first species whose representative (one of its members last generation) it is near enough to; the threshold (3, as in the paper) moves 5% a generation when there are more than 10 species, or only one
    speciate() {
      const old = this.species;
      for (const S of old) S.members = [];
      for (const g of this.pop) {
        let home = old.find((S) => this.distance(g, S.rep) < this.ct);
        if (!home) { home = { id: this.nextSp++, rep: g, members: [], best: -Infinity, since: 0, born: this.gen, from: g.sp || null }; old.push(home); }
        home.members.push(g); g.sp = home.id;
      }
      this.species = old.filter((S) => S.members.length);
      for (const S of this.species) S.rep = S.members[this.rng.int(S.members.length)];
      this.ct = Math.max(0.5, this.ct * (this.species.length < 2 ? 0.95 : this.species.length > TARGET ? 1.05 : 1));
    }
    // the next generation: each species gets children in proportion to its shared fitness; its best carries on when it has 5 or more
    reproduce(best) {
      const n = this.pop.length, alive = this.species.filter((S) => S.since < STALE || S.members.includes(best));
      const total = alive.reduce((a, S) => a + S.adj, 0) || 1, next = [this.clone(best)];
      const share = alive.map((S) => Math.floor((S.adj / total) * (n - 1)));
      let left = n - 1 - share.reduce((a, b) => a + b, 0);
      for (let i = 0; left > 0; i = (i + 1) % alive.length, left--) share[i]++;
      alive.forEach((S, i) => {
        let k = share[i];
        if (k > 0 && S.members.length >= 5 && S.members[0] !== best) { next.push(this.clone(S.members[0])); k--; }
        const parents = S.members.slice(0, Math.max(1, Math.ceil(S.members.length / 2)));
        while (k-- > 0) {
          const a = parents[this.rng.int(parents.length)], b = parents[this.rng.int(parents.length)];
          const child = this.rng.next() < 0.75 && a !== b ? this.cross(a.fit >= b.fit ? a : b, a.fit >= b.fit ? b : a) : this.clone(a);
          this.mutate(child);
          next.push(child);
        }
      });
      this.pop = next.slice(0, n);
      while (this.pop.length < n) { const c = this.clone(best); this.mutate(c); this.pop.push(c); }
    }
    clone(g) { return { nodes: g.nodes.map((q) => ({ ...q })), conns: g.conns.map((c) => ({ ...c })), sp: g.sp }; }
    // genes lined up by innovation number: the matching ones from either parent, the rest from the fitter
    cross(fit, other) {
      const B = new Map(other.conns.map((c) => [c.inn, c])), conns = fit.conns.map((c) => { const d = B.get(c.inn), src = d && this.rng.next() < 0.5 ? d : c; return { ...src, on: (c.on && (!d || d.on)) || this.rng.next() > 0.75 }; });
      return { nodes: fit.nodes.map((q) => ({ ...q })), conns, sp: fit.sp };
    }
    mutate(g) {
      g.net = null;
      if (this.rng.next() < 0.8) for (const c of g.conns) c.w = this.rng.next() < 0.9 ? clamp(c.w + this.rng.gauss(0, this.sigma), -8, 8) : this.rng.uniform(-2, 2);
      if (this.rng.next() < 0.1) this.addConn(g);
      if (this.rng.next() < 0.04) this.addNode(g);
      if (this.rng.next() < 0.01 && g.conns.length) { const c = g.conns[this.rng.int(g.conns.length)]; c.on = !c.on; }
    }
    // a connection between two neurons not yet joined, never into an input, never making a loop
    addConn(g) {
      const from = g.nodes.filter((q) => q.kind !== 'out'), to = g.nodes.filter((q) => q.kind === 'hid' || q.kind === 'out');
      for (let tries = 0; tries < 20; tries++) {
        const a = from[this.rng.int(from.length)].id, b = to[this.rng.int(to.length)].id;
        if (a === b || g.conns.some((c) => c.a === a && c.b === b) || this.reaches(g, b, a)) continue;
        g.conns.push({ a, b, w: this.rng.uniform(-1, 1), on: true, inn: this.innov(a, b) });
        return;
      }
    }
    reaches(g, from, to) { const seen = new Set([from]), stack = [from]; while (stack.length) { const x = stack.pop(); if (x === to) return true; for (const c of g.conns) if (c.a === x && !seen.has(c.b)) { seen.add(c.b); stack.push(c.b); } } return false; }
    // a neuron in the middle of a connection: the old one off, in with weight 1, out with the old weight
    addNode(g) {
      const on = g.conns.filter((c) => c.on);
      if (!on.length) return;
      const c = on[this.rng.int(on.length)];
      if (!this.split.has(c.inn)) this.split.set(c.inn, this.nextNode++);
      const id = this.split.get(c.inn);
      if (g.nodes.some((q) => q.id === id)) return;
      c.on = false;
      g.nodes.push({ id, kind: 'hid' });
      g.conns.push({ a: c.a, b: id, w: 1, on: true, inn: this.innov(c.a, id) }, { a: id, b: c.b, w: c.w, on: true, inn: this.innov(id, c.b) });
    }
  }
  NE.Neat = Neat;

  // ------------------------------------------------------------------ the views
  const font = (w, px) => `${w} ${px}px ${(typeof getComputedStyle === 'function' && getComputedStyle(document.body).getPropertyValue('--font-ui').trim()) || 'sans-serif'}`;
  const text = (ctx, s, x, y, o = {}) => { ctx.font = font(o.w || 500, o.size || 11); ctx.fillStyle = o.color || theme.c('ink'); ctx.textAlign = o.align || 'left'; ctx.textBaseline = 'middle'; ctx.fillText(s, x, y); };
  const SP = ['s1', 's2', 's4', 's7', 's3', 's5', 's6'];
  // the network it grew: the inputs on the left, the outputs on the right, the hidden ones between by depth; blue and red by sign, thicker for bigger weights
  function drawNet(ctx, r, L) {
    const E = L.agent && L.agent.neat, d = L.data;
    if (!E) return;
    const g = E.champ, N = compile(g), names = d.obsNames || [], depth = new Map();
    for (const q of g.nodes) if (q.kind === 'in' || q.kind === 'bias') depth.set(q.id, 0);
    for (const id of N.order) depth.set(id, 1 + Math.max(0, ...(N.into.get(id) || []).map((c) => depth.get(c.a) || 0)));
    const maxD = Math.max(1, ...g.nodes.filter((q) => q.kind === 'hid').map((q) => depth.get(q.id) || 1)) + 1;
    for (const q of g.nodes) if (q.kind === 'out') depth.set(q.id, maxD);
    const cols = new Map();
    for (const q of g.nodes) { const k = depth.get(q.id) || 0; if (!cols.has(k)) cols.set(k, []); cols.get(k).push(q); }
    const top = r.y + 26, h = r.h - 36, left = r.x + Math.min(120, r.w * 0.28), right = r.x + r.w - Math.min(110, r.w * 0.24), pos = new Map();
    for (const [k, list] of cols) list.forEach((q, i) => pos.set(q.id, [left + ((right - left) * k) / maxD, top + ((i + 0.5) * h) / list.length]));
    const hid = g.nodes.filter((q) => q.kind === 'hid').length, on = g.conns.filter((c) => c.on);
    text(ctx, `The network it grew: ${hid} hidden neuron${hid === 1 ? '' : 's'}, ${on.length} connections (generation ${fmtInt(E.gen)}, species ${E.champ.sp})`, r.x + 2, r.y + 9, { w: 650, size: 12, color: theme.c('ink-2') });
    for (const c of on) {
      const p = pos.get(c.a), q = pos.get(c.b);
      if (!p || !q) continue;
      ctx.strokeStyle = theme.divColor(clamp(c.w / 2, -1, 1), 0.85); ctx.lineWidth = Math.min(4, 0.6 + Math.abs(c.w));
      ctx.beginPath(); ctx.moveTo(p[0], p[1]); ctx.bezierCurveTo((p[0] + q[0]) / 2, p[1], (p[0] + q[0]) / 2, q[1], q[0], q[1]); ctx.stroke();
    }
    for (const q of g.nodes) {
      const p = pos.get(q.id);
      if (!p) continue;
      ctx.fillStyle = theme.c(q.kind === 'hid' ? 's7' : q.kind === 'out' ? 's2' : q.kind === 'bias' ? 'muted' : 's1'); ctx.beginPath(); ctx.arc(p[0], p[1], q.kind === 'hid' ? 6 : 5, 0, Math.PI * 2); ctx.fill();
      if (q.kind === 'in') text(ctx, names[q.id] || `input ${q.id + 1}`, p[0] - 9, p[1], { align: 'right', size: 10.5, color: theme.c('ink-2') });
      if (q.kind === 'bias') text(ctx, 'bias (1)', p[0] - 9, p[1], { align: 'right', size: 10.5, color: theme.c('muted') });
      if (q.kind === 'out') text(ctx, d.actions[q.id - E.nIn - 1] || 'out', p[0] + 9, p[1], { size: 10.5, color: theme.c('ink-2') });
    }
  }
  // species over time: each species a band, as wide as its members, from the generation it appeared
  function drawSpecies(ctx, r, L) {
    const E = L.agent && L.agent.neat;
    if (!E || !E.speciesHist.length) { text(ctx, 'Species appear once a generation has been judged.', r.x + 4, r.y + 14, { color: theme.c('muted') }); return; }
    const H = E.speciesHist, n = E.size, x0 = r.x + 34, y0 = r.y + 24, w = r.w - 44, h = r.h - 46, X = (i) => x0 + (H.length > 1 ? (i / (H.length - 1)) * w : w / 2);
    text(ctx, `Species over time: ${E.species.length} now, ${fmtInt(E.nextSp - 1)} so far; each band as wide as its members`, r.x + 2, r.y + 9, { w: 650, size: 12, color: theme.c('ink-2') });
    const ids = [...new Set(H.flatMap((q) => q.sizes.map((s) => s.id)))].sort((a, b) => a - b);
    const base = H.map(() => 0);
    for (const id of ids) {
      ctx.fillStyle = theme.c(SP[id % SP.length], 0.75);
      ctx.beginPath();
      H.forEach((q, i) => { const y = y0 + (base[i] / n) * h; if (i) ctx.lineTo(X(i), y); else ctx.moveTo(X(i), y); });
      const tops = H.map((q, i) => { const s = q.sizes.find((z) => z.id === id); return base[i] + (s ? s.n : 0); });
      for (let i = H.length - 1; i >= 0; i--) ctx.lineTo(X(i), y0 + (tops[i] / n) * h);
      ctx.closePath(); ctx.fill();
      tops.forEach((t, i) => { base[i] = t; });
    }
    ctx.strokeStyle = theme.c('line-2'); ctx.strokeRect(x0 + 0.5, y0 + 0.5, w - 1, h - 1);
    text(ctx, `generation ${fmtInt(H[0].gen + 1)}`, x0, y0 + h + 10, { size: 10, color: theme.c('muted') });
    text(ctx, fmtInt(H[H.length - 1].gen + 1), x0 + w, y0 + h + 10, { size: 10, color: theme.c('muted'), align: 'right' });
    text(ctx, 'members', r.x + 2, y0 + 8, { size: 10, color: theme.c('muted') });
  }
  // ------------------------------------------------------------------ the family tree of shapes (grow and prune)
  /* Every shape the genetic algorithm's networks have had: placed at the generation it first appeared, linked to the
     shape it came from, as big as the most members that ever had it, darker for a higher best score; the ring is the
     shape of the best now. Only shapes that lasted, or lead to one alive now, so the tree stays readable. */
  function drawShapes(ctx, r, L) {
    const ev = L.evo, T = ev && ev.shapeTree;
    if (!T || !T.size) { text(ctx, 'The family tree grows as children change shape.', r.x + 4, r.y + 14, { color: theme.c('muted') }); return; }
    const all = [...T.values()], gen = ev.gen, keep = new Set();
    for (const q of all) if (q.most >= 2 || q.last >= gen - 1 || q.key === ev.bestShape) { let p = q; while (p && !keep.has(p.key)) { keep.add(p.key); p = p.parent ? T.get(p.parent) : null; } }
    let nodes = all.filter((q) => keep.has(q.key));
    if (nodes.length > 60) { const top = new Set(nodes.sort((a, b) => b.most - a.most).slice(0, 40).map((q) => q.key)); nodes = nodes.filter((q) => top.has(q.key) || q.parent === null); }
    // rows: a parent's children below it in the order they appeared (depth first)
    const kids = new Map(), row = new Map();
    for (const q of nodes) { const p = q.parent && nodes.some((z) => z.key === q.parent) ? q.parent : '_'; if (!kids.has(p)) kids.set(p, []); kids.get(p).push(q); }
    let next = 0;
    const place = (p) => { for (const q of (kids.get(p) || []).sort((a, b) => a.gen - b.gen)) { row.set(q.key, next++); place(q.key); } };
    place('_');
    const rows = Math.max(1, next), x0 = r.x + 14, y0 = r.y + 26, w = r.w - 28 - 90, h = r.h - 44, maxG = Math.max(1, gen), X = (g) => x0 + (g / maxG) * w, Y = (k) => y0 + ((k + 0.5) / rows) * h;
    const fits = nodes.map((q) => q.best).filter(Number.isFinite), lo = Math.min(...fits), hi = Math.max(...fits);
    text(ctx, `The family tree of shapes: ${nodes.length} shapes shown of ${T.size}, from generation 1 to ${fmtInt(gen)}`, r.x + 2, r.y + 9, { w: 650, size: 12, color: theme.c('ink-2') });
    ctx.lineWidth = 1.2;
    for (const q of nodes) {
      const p = q.parent && row.has(q.parent) ? q.parent : null;
      if (!p) continue;
      const x = X(q.gen), y = Y(row.get(q.key)), py = Y(row.get(p));
      ctx.strokeStyle = theme.c('line-2'); ctx.beginPath(); ctx.moveTo(x, py); ctx.lineTo(x, y); ctx.lineTo(Math.max(x + 2, X(q.last)), y); ctx.stroke();
    }
    for (const q of nodes) {
      const x = X(q.gen), y = Y(row.get(q.key)), rad = 3 + Math.min(7, Math.sqrt(q.most)), v = Number.isFinite(q.best) && hi > lo ? (q.best - lo) / (hi - lo) : 0.5;
      ctx.fillStyle = theme.seqColor(0.25 + 0.75 * v); ctx.beginPath(); ctx.arc(x, y, rad, 0, Math.PI * 2); ctx.fill();
      if (q.key === ev.bestShape) { ctx.strokeStyle = theme.c('s7'); ctx.lineWidth = 2.5; ctx.beginPath(); ctx.arc(x, y, rad + 3, 0, Math.PI * 2); ctx.stroke(); ctx.lineWidth = 1.2; }
      if (rows <= 24 || q.key === ev.bestShape) text(ctx, q.units.length ? q.units.join('·') : 'none', Math.min(r.x + r.w - 30, Math.max(X(q.last), x) + rad + 8), y, { size: 10, color: theme.c(q.key === ev.bestShape ? 'ink' : 'muted'), w: q.key === ev.bestShape ? 700 : 500 });
    }
  }
  NE.views = function (o) {
    const out = o.method === 'neat' ? [{ value: 'neatnet', label: 'The network it grew', title: 'The best network of the last generation: its neurons and connections, blue and red by the sign of each weight' }, { value: 'species', label: 'Species', title: 'Its species over the generations, each as wide as its members' }] : [];
    if (o.evo && o.evo.shapeTree) out.push({ value: 'shapes', label: 'Family tree of shapes', title: 'Every shape its networks have had, linked to the shape it came from' });
    return out;
  };
  NE.draw = function (view, ctx, r, L) { if (view === 'neatnet') drawNet(ctx, r, L); else if (view === 'shapes') drawShapes(ctx, r, L); else drawSpecies(ctx, r, L); };
  NA.neat = NE;
})(typeof globalThis !== 'undefined' && globalThis.NA ? globalThis.NA : window.NA);
