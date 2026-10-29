/* rl-evolve.js — Evolution, the RL lab's fourth example: a genetic algorithm evolves the weights of a small neural
   network that balances a pole on a cart (4 inputs, 8 tanh neurons, 2 outputs: push left or push right). No
   gradients and no lesson from any single move: a population of networks each play, the ones that keep the pole
   up longest pass on their weights, and children mix two parents and mutate a little. The lab (lab-rl.js) hands
   this example's board, panels, test, play and settings to the functions here, and its architecture view
   (arch-rl.js) the tree built here. */
(function (NA) {
  'use strict';
  const { el, clear, clamp, fmt, fmtPct, fmtInt, theme } = NA;
  const UI = NA.labUI;
  const EV = {};
  const NIN = 4, NH = 8, NOUT = 2, MAXT = 1000;
  const OFF = { W1: 0, b1: NIN * NH, W2: NIN * NH + NH, b2: NIN * NH + NH + NH * NOUT }, SIZE = OFF.b2 + NOUT;
  Object.assign(EV, { NIN, NH, NOUT, MAXT, SIZE, OFF });
  EV.INPUTS = ['cart position', 'cart speed', 'pole angle', 'pole turning speed'];
  EV.ACTIONS = ['push left', 'push right'];
  EV.DEFAULTS = { pop: 30, sigma: 0.3, rate: 0.1, elite: 2 };
  const TANH = { id: 'tanh', label: 'tanh', f: Math.tanh }, LINEAR = { id: 'linear', label: 'none' };
  const K = () => NA.archKit;
  const deg = (th) => (th * 180) / Math.PI;

  // ------------------------------------------------------------------ the network and the pole
  // one network's computation for one observation: the hidden sums z, their tanh h, and the 2 outputs o
  EV.forward = function (w, x) {
    const z = new Float64Array(NH), h = new Float64Array(NH), o = new Float64Array(NOUT);
    for (let j = 0; j < NH; j++) { let s = w[OFF.b1 + j]; for (let i = 0; i < NIN; i++) s += x[i] * w[OFF.W1 + i * NH + j]; z[j] = s; h[j] = Math.tanh(s); }
    for (let k = 0; k < NOUT; k++) { let s = w[OFF.b2 + k]; for (let j = 0; j < NH; j++) s += h[j] * w[OFF.W2 + j * NOUT + k]; o[k] = s; }
    return { z, h, o };
  };
  // it pushes the way of its larger output
  EV.act = (w, x) => { const f = EV.forward(w, x); return f.o[1] > f.o[0] ? 1 : 0; };
  // 1,000 steps (20 seconds), each starting up to 8.6° from upright and half a unit from the middle
  EV.pole = () => new NA.rlEnvs.Pole({ steps: MAXT, start: 0.15, xStart: 0.5 });
  // one episode: how many steps the pole stays up (and, with keep, every step's state and computation)
  EV.episode = function (w, rng, keep, P = EV.pole()) {
    let o = P.reset(rng), done = false, fell = false;
    const frames = keep ? [] : null;
    while (!done) {
      const f = EV.forward(w, o), a = f.o[1] > f.o[0] ? 1 : 0;
      if (keep) frames.push({ x: P.x, th: P.th, v: P.v, om: P.w, t: P.t, obs: Array.from(o), z: f.z, h: f.h, o: f.o, a });
      const t = P.step(a);
      o = t.obs; done = t.done; fell = t.fell;
    }
    if (keep) frames.push({ x: P.x, th: P.th, v: P.v, om: P.w, t: P.t, obs: Array.from(o), end: fell ? 'fell' : 'up' });
    return { steps: P.t, fell, frames };
  };
  // the score: how long its moves keep the pole up from 5 fixed starts, as a share of the episode
  EV.score = (w) => EV.pole().score((obs) => EV.act(w, obs));

  /* A population of networks, and the genetic algorithm that breeds it. Each generation every member plays two
     episodes from the same two starts (so the comparison is fair) and its fitness is the average number of steps
     the pole stayed up. The best `elite` pass on unchanged; every other place goes to a child of two parents, each
     the fitter of three members drawn at random (a tournament), with each weight copied from one parent or the
     other, then a share `rate` of its weights moved by a random amount of size `sigma` (mutation). */
  class Evolution {
    constructor(o, seed = 7) {
      this.o = o; this.rng = new NA.RNG(seed); this.gen = 0; this.history = []; this.world = EV.pole();
      this.pop = NA.range(o.pop).map(() => ({ w: this.random(), kind: 'random' }));
      this.last = null; this.best = this.pop[0].w; this.bestFit = null;
    }
    random() { const w = new Float64Array(SIZE); for (let i = 0; i < SIZE; i++) w[i] = this.rng.gauss(0, 0.5); return w; }
    fitness(w, seed) { const r = new NA.RNG(seed); let s = 0; for (let e = 0; e < 2; e++) s += EV.episode(w, r, false, this.world).steps; return s / 2; }
    generation() {
      const o = this.o, n = this.pop.length, seed = 1000 + this.gen;
      for (const m of this.pop) m.fit = this.fitness(m.w, seed);
      const ranked = this.pop.slice().sort((a, b) => b.fit - a.fit), fits = ranked.map((m) => m.fit);
      const mean = fits.reduce((a, b) => a + b, 0) / n;
      this.last = { gen: this.gen + 1, ranked, best: fits[0], mean, worst: fits[n - 1] };
      this.best = ranked[0].w; this.bestFit = fits[0];
      const tour = () => { let b = this.rng.int(n); for (let k = 0; k < 2; k++) { const c = this.rng.int(n); if (ranked[c].fit > ranked[b].fit) b = c; } return b; };
      const next = ranked.slice(0, Math.min(o.elite, n, o.pop)).map((m, i) => ({ w: Float64Array.from(m.w), kind: 'elite', parents: [i] }));
      while (next.length < o.pop) {
        const ia = tour(), ib = tour(), a = ranked[ia].w, b = ranked[ib].w, w = new Float64Array(SIZE);
        let mut = 0;
        for (let i = 0; i < SIZE; i++) { w[i] = this.rng.next() < 0.5 ? a[i] : b[i]; if (this.rng.next() < o.rate) { w[i] += this.rng.gauss(0, o.sigma); mut++; } }
        next.push({ w, kind: 'child', parents: [ia, ib], mutated: mut });
      }
      this.pop = next; this.gen++;
      const q = { n: this.gen, best: fits[0], mean, worst: fits[n - 1] };
      if (this.gen <= 10 || this.gen % 5 === 0) q.score = EV.score(this.best);
      this.history.push(q);
      if (this.history.length > 400) { const keep = this.history.filter((p, i) => i % 2 === 0 || i === this.history.length - 1); this.history.length = 0; this.history.push(...keep); }
      return this.last;
    }
  }
  EV.Evolution = Evolution;
  const lastScore = (h) => { for (let i = h.length - 1; i >= 0; i--) if (h[i].score != null) return h[i]; return null; };
  EV.lastScore = lastScore;

  // ------------------------------------------------------------------ the lab's state
  EV.state = function (key) {
    const o = Object.assign({}, EV.DEFAULTS), evo = new Evolution(o, 7);
    return { key, id: 'evolve', model: 'ga', n: 0, rng: NA.rl.rand(44), ev: NA.rl.rand(99), o, evo, history: evo.history };
  };
  /* Train: about 8 generations a second, slow enough to watch the bars and the average climb. An experiment that
     asks for a number of generations runs them as fast as it can. */
  EV.trainChunk = function (L, budget, maxSteps) {
    const S = L.S, t0 = performance.now();
    if (maxSteps === Infinity) { if (t0 - (L.genAt || 0) < 120) return 0; L.genAt = t0; maxSteps = 1; }
    let k = 0;
    if (L.filmTick) L.filmTick(); // its film starts before the first generation
    do { S.evo.generation(); S.n = S.evo.gen; k++; if (S.film && S.evo.gen >= S.film.next) S.film.tick(); } while (performance.now() - t0 < budget && k < maxSteps);
    L.step = S.n; L.bump();
    return k;
  };
  // watching: a generation a second, and the board plays each new best network
  EV.watchTick = function (L) { const S = L.S; S.evo.generation(); S.n = S.evo.gen; S.show = null; L.step = S.n; L.bump(); };

  // the board plays the best network so far, in real time (an episode from a new start each time)
  EV.frame = function (L, now) {
    const S = L.S, P = S.play;
    if (P && !P.over) {
      // you balance it: one 0.02 s step per 20 ms, pushed the way you hold
      const due = Math.min(5, Math.floor((now - P.at) / 20));
      for (let k = 0; k < due && !P.over; k++) {
        const t = P.pole.step(P.push === 1 ? 1 : P.push === -1 ? 0 : -1);
        P.at += 20;
        if (t.done) { P.over = true; P.fell = t.fell; L.renderBar(); NA.ui.toast.show(t.fell ? `The pole fell after ${P.pole.t} steps. Its best network: ${EV.netSteps(L)} steps from the same start.` : `You kept it up for all ${MAXT} steps.`); }
      }
      if (due) { L.dirty = true; L.archStale = true; }
      return;
    }
    if (L.test || P) return;
    const sh = S.show;
    if (!sh || (sh.k >= sh.frames.length - 1 && now >= sh.endAt)) {
      const best = S.evo.bestFit === null ? S.evo.pop[0].w : S.evo.best;
      S.show = { frames: EV.episode(best, S.showRng || (S.showRng = new NA.RNG(321)), true).frames, k: 0, at: now, gen: S.evo.gen };
      L.dirty = true; L.archStale = true;
      return;
    }
    const k = Math.min(sh.frames.length - 1, Math.floor((now - sh.at) / 20));
    if (k !== sh.k) { sh.k = k; if (k === sh.frames.length - 1) sh.endAt = now + 900; L.dirty = true; if (L.centerView !== 'arch' || now - (sh.archAt || 0) > 250) { sh.archAt = now; L.archStale = true; } }
  };
  // how long its best network keeps the pole up from the start you had
  EV.netSteps = (L) => { const P = L.S.play; return EV.episode(L.S.evo.best, new NA.RNG(P.seed), false).steps; };

  // ------------------------------------------------------------------ test: the best network, slowly
  EV.makeFrames = function (L) {
    const n = (L.tests = (L.tests || 0) + 1), S = L.S, w = S.evo.bestFit === null ? S.evo.pop[0].w : S.evo.best;
    return EV.episode(w, new NA.RNG(500 + n), true).frames;
  };
  EV.testResult = (L) => { const f = L.test.frames[L.test.frames.length - 1]; return f.end === 'up' ? `kept the pole up for all ${MAXT} steps` : `the pole fell after ${f.t} steps`; };
  EV.testCount = (L) => { const T = L.test, n = T.frames.length, last = T.k >= n - 1; return last ? `Step ${n - 1}: ${EV.testResult(L)}` : `Step ${T.k + 1} of ${n - 1}`; };

  // ------------------------------------------------------------------ you try: hold ← or → (or press either half of the board)
  EV.startPlay = function (L) {
    const S = L.S;
    if (L.training) L.pause();
    L.watch = false; if (L.watchChip) L.watchChip.setAttribute('aria-pressed', 'false');
    L.test = null;
    const seed = 900 + ((L.plays = (L.plays || 0) + 1) % 50), pole = EV.pole();
    pole.reset(new NA.RNG(seed));
    S.play = { pole, push: 0, over: false, at: performance.now() + 900, seed };
    EV.listen(L, true);
    L.bump(); L.renderBar();
    NA.ui.toast.show(NA.mobile && NA.mobile.mobile ? 'Press and hold the left or right half of the board to push the cart.' : `Hold ← or → to push the cart (or press either half of the board). Keep the pole up for ${fmtInt(MAXT)} steps.`);
  };
  EV.listen = function (L, on) {
    if (L._evoKeys) { document.removeEventListener('keydown', L._evoKeys.down, true); document.removeEventListener('keyup', L._evoKeys.up, true); L.canvas.removeEventListener('pointerdown', L._evoKeys.pdown); window.removeEventListener('pointerup', L._evoKeys.pup); L._evoKeys = null; }
    if (!on) return;
    const P = () => L.S && L.S.play;
    const down = (e) => { const p = P(); if (!p || p.over || !(e.key === 'ArrowLeft' || e.key === 'ArrowRight')) return; e.preventDefault(); e.stopPropagation(); p.push = e.key === 'ArrowLeft' ? -1 : 1; };
    const up = (e) => { const p = P(); if (!p || !(e.key === 'ArrowLeft' || e.key === 'ArrowRight')) return; e.preventDefault(); if ((e.key === 'ArrowLeft' && p.push === -1) || (e.key === 'ArrowRight' && p.push === 1)) p.push = 0; };
    const pdown = (e) => { const p = P(); if (!p || p.over) return; const b = L.canvas.getBoundingClientRect(), B = L.layout().board; p.push = e.clientX - b.left < B.x + B.w / 2 ? -1 : 1; };
    const pup = () => { const p = P(); if (p) p.push = 0; };
    L._evoKeys = { down, up, pdown, pup };
    document.addEventListener('keydown', down, true); document.addEventListener('keyup', up, true); L.canvas.addEventListener('pointerdown', pdown); window.addEventListener('pointerup', pup);
  };
  EV.playBar = function (L, bar, btn) {
    const P = L.S.play;
    bar.append(el('span', { class: 'rl-status', text: P.over ? (P.fell ? `It fell after ${P.pole.t} steps` : `All ${MAXT} steps: well done`) : `Balancing: hold ← or →` }),
      btn('Try again', () => EV.startPlay(L), P.over ? 'btn btn-primary' : 'btn'), btn('Stop', () => { EV.listen(L, false); L.stopPlay(); }, 'btn btn-quiet'));
  };

  // ------------------------------------------------------------------ what is on the board
  // the state looked at: your run, the test's step, or the best network's show
  EV.probe = function (L) {
    const S = L.S, T = L.test, P = S.play, best = S.evo.bestFit === null ? S.evo.pop[0].w : S.evo.best;
    let st, why, w = best, f = null;
    if (T) { f = T.frames[T.k]; st = f; why = 'test'; }
    else if (P) { st = { x: P.pole.x, th: P.pole.th, v: P.pole.v, om: P.pole.w, t: P.pole.t, obs: Array.from(P.pole.obs()) }; why = 'you'; }
    else if (S.show) { st = S.show.frames[S.show.k]; why = 'show'; }
    else { const p = EV.pole(); p.reset(new NA.RNG(321)); st = { x: p.x, th: p.th, v: p.v, om: p.w, t: 0, obs: Array.from(p.obs()) }; why = 'start'; }
    const c = EV.forward(w, st.obs), a = c.o[1] > c.o[0] ? 1 : 0;
    return { st, why, w, x: st.obs, z: c.z, h: c.h, o: c.o, a, f, T, end: st.end };
  };
  EV.draw = function (L) {
    const Lo = L.layout(), ctx = NA.fitCanvas(L.canvas, Lo.W, Lo.H), S = L.S, Pr = EV.probe(L), B = Lo.board;
    ctx.clearRect(0, 0, Lo.W, Lo.H);
    const showPop = L.show.values && !S.play && !L.test, popH = showPop ? Math.min(150, B.h * 0.36) : 0;
    const top = { x: B.x + 10, y: B.y + 26, w: B.w - 20, h: B.h - 34 - popH };
    const st = Pr.st, last = S.evo.last;
    const line = Pr.why === 'test' ? `Test: step ${st.t} of ${MAXT}${Pr.end ? (Pr.end === 'up' ? ', still up' : ', the pole fell') : ''}`
      : Pr.why === 'you' ? `You: ${st.t} steps${S.play.over ? (S.play.fell ? ', it fell' : '') : ''}`
        : Pr.why === 'show' ? (last ? `The best of generation ${fmtInt(S.show.gen)}: step ${st.t}` : `A random network, before any evolution: step ${st.t}`) : 'The pole at the start';
    K().text(ctx, line, B.x + 12, B.y + 14, { size: 12.5, w: 650, color: theme.c('ink-2'), maxW: B.w - 24 });
    NA.rlEnvs.draw(ctx, top, 'pole', { x: st.x, th: st.th, a: Pr.why === 'you' ? (S.play.push === 1 ? 1 : S.play.push === -1 ? 0 : null) : Pr.end ? null : Pr.a, t: st.t }, theme);
    if (showPop) EV.drawPopulation(ctx, { x: B.x + 12, y: B.y + B.h - popH, w: B.w - 24, h: popH - 6 }, S);
    if (Lo.inside) EV.drawInside(ctx, Lo.inside, L, Pr);
  };
  // the last generation judged: one bar per member, fittest first; the best few carried on (elite)
  EV.drawPopulation = function (ctx, r, S) {
    const last = S.evo.last, u = 11;
    if (!last) { K().text(ctx, `A population of ${S.o.pop} random networks: press Train to judge them and breed the next generation.`, r.x, r.y + r.h / 2, { size: u, color: theme.c('muted'), maxW: r.w }); return; }
    const n = last.ranked.length, gap = n > 60 ? 1 : 2, bw = Math.max(1.5, (r.w - gap * (n - 1)) / n), base = r.y + r.h - 14, hh = r.h - 34;
    K().text(ctx, `Generation ${fmtInt(last.gen)} · best ${fmtInt(Math.round(last.best))} · average ${fmtInt(Math.round(last.mean))} steps`, r.x, r.y + 7, { size: u, w: 600, color: theme.c('ink-2'), maxW: r.w });
    ctx.strokeStyle = theme.c('line'); ctx.lineWidth = 1; ctx.beginPath(); ctx.moveTo(r.x, base + 0.5); ctx.lineTo(r.x + r.w, base + 0.5); ctx.stroke();
    last.ranked.forEach((m, i) => {
      const h = Math.max(1.5, (m.fit / MAXT) * hh), x = r.x + i * (bw + gap), elite = i < S.o.elite;
      ctx.fillStyle = theme.c(elite ? 's2' : m.kind === 'random' ? 'muted' : 's1', elite ? 0.95 : 0.7);
      NA.rr(ctx, x, base - h, bw, h, Math.min(3, bw / 2)); ctx.fill();
    });
    K().text(ctx, S.o.elite ? `${n} networks, fittest first · orange: the best ${S.o.elite}, kept as they are` : `${n} networks, fittest first · no elite: even the best is replaced`, r.x, base + 8, { size: u * 0.9, color: theme.c('muted'), maxW: r.w });
  };
  // beside the board: the network's computation for the state looked at
  EV.drawInside = function (ctx, r, L, Pr = EV.probe(L)) {
    const pad = 10, u = clamp(Math.min(r.w / 30, r.h / 17), 10.5, 13);
    K().box(ctx, r, { fill: theme.c('panel-2'), stroke: theme.c('line') });
    K().text(ctx, Pr.why === 'you' ? 'What its best network would do here' : 'Inside its best network', r.x + pad, r.y + pad + u * 0.6, { size: u * 1.05, w: 700, maxW: r.w - 2 * pad });
    const st = Pr.st, lean = deg(st.th);
    K().text(ctx, `Step ${st.t}: the pole leans ${fmt(Math.abs(lean), 1)}° ${lean >= 0 ? 'right' : 'left'}${Pr.end ? '' : `; it would ${EV.ACTIONS[Pr.a]}`}`, r.x + pad, r.y + pad + u * 1.95, { size: u * 0.92, color: theme.c('ink-2'), maxW: r.w - 2 * pad });
    const Bx = { x: r.x + pad, y: r.y + pad + u * 3.4, w: r.w - 2 * pad, h: r.h - 2 * pad - u * 3.6 };
    if (Bx.w < 150 || Bx.h < 90) return;
    EV.drawNet(ctx, Bx, Pr, u);
  };
  // inputs (signed bars), 8 tanh neurons, 2 outputs; the wires to the chosen output coloured by what each neuron adds
  EV.drawNet = function (ctx, B, Pr, u) {
    const ci = { x: B.x, y: B.y, w: B.w * 0.36, h: B.h }, chx = B.x + B.w * 0.55, co = { x: B.x + B.w * 0.68, y: B.y, w: B.w * 0.32, h: B.h };
    const rowI = Math.min(ci.h / 4, u * 3.4), rowH = Math.min((B.h - u * 1.4) / NH, u * 2.2), y0h = B.y + u * 1.4 + (B.h - u * 1.4 - rowH * NH) / 2, rad = Math.min(rowH * 0.36, u * 0.75);
    const iy = (i) => B.y + (B.h - rowI * 4) / 2 + (i + 0.5) * rowI, hy = (j) => y0h + (j + 0.5) * rowH, oy = (k) => B.y + B.h / 2 + (k - 0.5) * u * 3.2;
    // wires: inputs to hidden (faint), hidden to the chosen output (by contribution)
    for (let i = 0; i < NIN; i++) for (let j = 0; j < NH; j++) { const v = Pr.x[i] * Pr.w[OFF.W1 + i * NH + j]; ctx.strokeStyle = theme.divColor(clamp(v, -1, 1), 0.18 + 0.4 * Math.min(1, Math.abs(v))); ctx.lineWidth = 0.6 + 1.2 * Math.min(1, Math.abs(v)); ctx.beginPath(); ctx.moveTo(ci.x + ci.w + 2, iy(i)); ctx.lineTo(chx - rad - 2, hy(j)); ctx.stroke(); }
    if (!Pr.end) for (let j = 0; j < NH; j++) { const v = Pr.h[j] * Pr.w[OFF.W2 + j * NOUT + Pr.a]; ctx.strokeStyle = theme.divColor(clamp(v, -1, 1), 0.25 + 0.6 * Math.min(1, Math.abs(v))); ctx.lineWidth = 0.8 + 2 * Math.min(1, Math.abs(v)); ctx.beginPath(); ctx.moveTo(chx + rad + 2, hy(j)); ctx.lineTo(co.x - 4, oy(Pr.a)); ctx.stroke(); }
    // the 4 inputs
    for (let i = 0; i < NIN; i++) {
      const y = iy(i), v = Pr.x[i], bw = ci.w * 0.42, bx = ci.x + ci.w - bw - 2, mid = bx + bw / 2;
      K().text(ctx, EV.INPUTS[i], ci.x, y - u * 0.55, { size: u * 0.82, color: theme.c('muted'), maxW: ci.w });
      ctx.fillStyle = theme.c('line', 0.6); ctx.fillRect(bx, y + u * 0.15, bw, u * 0.7);
      const len = clamp(v, -1, 1) * bw / 2; ctx.fillStyle = theme.divColor(clamp(v, -1, 1), 0.9); ctx.fillRect(Math.min(mid, mid + len), y + u * 0.15, Math.abs(len), u * 0.7);
      K().text(ctx, fmt(v, 2), bx - 4, y + u * 0.5, { size: u * 0.85, mono: true, align: 'right', color: theme.c('ink-2') });
    }
    // the 8 hidden neurons
    for (let j = 0; j < NH; j++) { ctx.fillStyle = theme.divColor(Pr.h[j] * 0.95, 0.95); ctx.beginPath(); ctx.arc(chx, hy(j), rad, 0, Math.PI * 2); ctx.fill(); ctx.strokeStyle = theme.c('line-2'); ctx.lineWidth = 1; ctx.stroke(); }
    K().text(ctx, '8 tanh', chx, y0h - u * 0.7, { size: u * 0.82, color: theme.c('muted'), align: 'center' });
    // the 2 outputs
    for (let k = 0; k < NOUT; k++) {
      const y = oy(k), pick = !Pr.end && k === Pr.a;
      NA.rr(ctx, co.x, y - u * 1.1, co.w, u * 2.2, 5); ctx.fillStyle = pick ? theme.c('s2', 0.18) : theme.c('panel'); ctx.fill(); ctx.strokeStyle = theme.c(pick ? 's2' : 'line-2'); ctx.lineWidth = pick ? 2 : 1; ctx.stroke();
      K().text(ctx, `${EV.ACTIONS[k]} ${fmt(Pr.o[k], 2)}`, co.x + 6, y, { size: u * 0.88, w: pick ? 700 : 500, color: theme.c(pick ? 'ink' : 'ink-2'), maxW: co.w - 10 });
    }
  };

  // ------------------------------------------------------------------ the panels
  EV.readoutDefs = () => [{ key: 'n', label: 'Generation', short: 'Gen.', keep: true }, { key: 'a', label: 'Best', keep: true, title: `Steps the fittest network kept the pole up this generation (of ${fmtInt(MAXT)}), averaged over its 2 episodes` }, { key: 'b', label: 'Average', title: 'The population’s average' }, { key: 'c', label: 'Score', title: 'The best network from 5 fixed starts, as a share of 500 steps' }];
  EV.readouts = function (L) {
    const S = L.S, last = S.evo.last, sc = lastScore(S.history);
    L.ro.set('n', fmtInt(S.n)); L.ro.set('a', last ? `${fmt(last.best, 0)} steps` : '—'); L.ro.set('b', last ? fmt(last.mean, 0) : '—'); L.ro.set('c', sc ? fmtPct(sc.score, 0) : '—');
  };
  const setO = (L, key, v) => { L.S.o[key] = v; L.leftDirty = true; };
  EV.buildHyper = function (L, root) {
    const S = L.S;
    root.append(
      UI.field('Population', UI.select([10, 30, 60, 100].map((v) => ({ value: v, label: String(v) })), S.o.pop, (v) => setO(L, 'pop', +v)), 'Networks in each generation'),
      UI.field('Mutation size', UI.select([0, 0.05, 0.1, 0.3, 0.6, 1.5].map((v) => ({ value: v, label: String(v) })), S.o.sigma, (v) => setO(L, 'sigma', +v)), 'How far a mutation moves a weight'),
      UI.field('Mutation rate', UI.select([0.02, 0.05, 0.1, 0.25, 0.5].map((v) => ({ value: v, label: fmtPct(v, 0) })), S.o.rate, (v) => setO(L, 'rate', +v)), 'The share of a child’s weights that mutate'),
      el('div', { class: 'sheet-actions' },
        el('button', { class: 'btn', type: 'button', onclick: () => { NA.mobile && NA.mobile.sheet(false); L.watch = false; L.test = null; L.fastForward(100); } }, 'Evolve 100 generations'),
        el('button', { class: 'btn', type: 'button', onclick: () => { NA.mobile && NA.mobile.sheet(false); L.reset(); } }, 'Start from nothing')));
  };
  EV.buildLeft = function (L, root) {
    const S = L.S, body = UI.panel(root, 'Evolution'), set = (key) => (v) => setO(L, key, v);
    L.watchChip = UI.chip('Watch it evolve, a generation a second', L.watch, (on) => { L.watch = on; L.test = null; L.renderBar(); if (on && !L.training) L.play(); L.bump(); }, 'Slows evolution down to one generation a second, and the board plays each new best network');
    body.append(
      UI.section('The task', UI.note(`Keep a pole upright on a cart by pushing the cart left or right, for up to ${fmtInt(MAXT)} steps (20 seconds), from a slightly different start each time. The pole falls if it leans more than 12° or the cart runs off its track. Each network sees 4 numbers (the cart’s position and speed, the pole’s angle and how fast it turns) and pushes the way of its larger output.`)),
      UI.section('How it learns: a genetic algorithm', UI.note(`No gradients, and no lesson from any single move. A population of ${S.o.pop} networks, each with its own ${SIZE} weights (its genome), starts at random. Each generation every network plays two episodes from the same two starts; its fitness is how long the pole stayed up. The best ${S.o.elite} pass on unchanged. Every other place goes to a child: two parents, each the fitter of three drawn at random, give it each weight from one or the other, and a few of its weights mutate by a random amount.`),
        UI.slider({ label: 'Population (networks per generation)', min: 4, max: 100, step: 2, value: S.o.pop, format: String, onInput: (v) => setO(L, 'pop', Math.round(v)) }),
        UI.slider({ label: 'Mutation size σ', min: 0, max: 1.5, step: 0.05, value: S.o.sigma, format: (v) => v.toFixed(2), onInput: set('sigma') }),
        UI.slider({ label: 'Mutation rate (share of weights that mutate)', min: 0, max: 0.5, step: 0.01, value: S.o.rate, format: (v) => fmtPct(v, 0), onInput: set('rate') }),
        UI.slider({ label: 'Elite (the best, carried on unchanged)', min: 0, max: 6, step: 1, value: S.o.elite, format: String, onInput: (v) => setO(L, 'elite', Math.round(v)) })),
      UI.section('Watch, test and try', UI.note('Watch slows evolution to a generation a second. Test run plays the best network slowly, with its computation beside the board at every step; ◀ ▶ (or the arrow keys) step through it. Try it yourself: balance the pole with ← and →, and compare.'), L.watchChip));
  };
  EV.updateTrain = function (L, P) {
    const S = L.S, h = S.history, last = S.evo.last, sc = lastScore(h);
    clear(P.tiles).append(UI.tile('Generations', fmtInt(S.n), `${fmtInt(S.n * S.o.pop)} networks judged`), UI.tile('Best this generation', last ? `${fmt(last.best, 0)} steps` : '—', last ? `average ${fmt(last.mean, 0)}` : null), UI.tile('Score', sc ? fmtPct(sc.score, 0) : '—', 'the best, from 5 fixed starts'));
    P.a.frame(); P.a.set({ series: [{ name: 'best', color: theme.c('s2'), points: h.map((q) => [q.n, q.best]) }, { name: 'average', color: theme.c('s1'), points: h.map((q) => [q.n, q.mean]) }, { name: 'worst', color: theme.c('ink-2'), points: h.map((q) => [q.n, q.worst]) }], yFixed: [0, MAXT] });
    P.b.frame(); P.b.set({ series: [{ name: 'score', color: theme.c('s7'), points: h.filter((q) => q.score != null).map((q) => [q.n, q.score]) }], yFixed: [0, 1] });
  };
  EV.buildTrain = function (L, body, mk, lg) {
    const P = L.parts;
    P.tiles = el('div', { class: 'tiles' }); body.append(P.tiles);
    P.a = mk('Fitness: steps the pole stays up', { height: 150, format: (v) => fmt(v, 0) }, lg([['the best', 's2'], ['the average', 's1'], ['the worst', 'ink-2']]), 'Each generation, over the two episodes every network plays. The best rises in jumps, when a lucky child appears; the average follows as its genes spread.');
    P.b = mk('Score of the best network', { height: 120, format: (v) => fmtPct(v, 0), yFormat: (v) => fmtPct(v, 0) }, null, `From 5 fixed starts it never trained on (none of the generations’ starts), as a share of ${fmtInt(MAXT)} steps.`);
  };
  // the population: its best members, and where each came from
  EV.inspect = function (L) {
    const S = L.S, last = S.evo.last;
    if (!last) return [UI.section('The population', UI.note(`${S.o.pop} networks with random weights, none of them judged yet. Press Train.`))];
    const kv = (k, v, note) => el('div', { class: 'kv' }, el('span', { class: 'kv-k', text: k }), el('span', { class: 'kv-v', text: v }), note ? el('span', { class: 'kv-note', text: note }) : null);
    const born = (m) => (m.kind === 'elite' ? 'carried on unchanged from the generation before' : m.kind === 'random' ? 'random weights' : `child of #${m.parents[0] + 1} and #${m.parents[1] + 1} of the generation before${m.mutated ? `, ${m.mutated} weight${m.mutated > 1 ? 's' : ''} mutated` : ', no mutation'}`);
    const w = S.evo.best, norm = Math.sqrt(w.reduce((s, v) => s + v * v, 0));
    return [UI.section(`Generation ${fmtInt(last.gen)}: the fittest`, UI.note(`Fitness is the average number of steps the pole stayed up in its two episodes (${fmtInt(MAXT)} at most).`),
      el('div', { class: 'kv-grid' }, ...last.ranked.slice(0, 10).map((m, i) => kv(`#${i + 1}`, `${fmt(m.fit, 0)} steps`, born(m))))),
    UI.section('The best network’s genome', UI.note(`${SIZE} numbers: ${NIN} × ${NH} input weights and ${NH} biases for the hidden neurons, ${NH} × ${NOUT} output weights and ${NOUT} biases. Their overall size (the length of the genome as a vector) is ${fmt(norm, 2)}; evolution changes them only by mixing parents and by mutation.`))];
  };
  EV.findings = function (L) {
    const S = L.S, last = S.evo.last, h = S.history, sc = lastScore(h), out = [];
    if (!last) return [['info', 'Not evolved yet', `Press Train. ${S.o.pop} networks with random weights: most push one way all the time, and the pole falls within a few dozen steps.`]];
    if (sc) out.push([sc.score >= 0.98 ? 'good' : 'info', `Its best network keeps the pole up ${fmtPct(sc.score, 0)} of the time`, sc.score >= 0.98 ? `From 5 starts it never met, for all ${MAXT} steps: evolved, never trained by a gradient.` : 'From 5 fixed starts, as a share of 500 steps. Keep evolving: the best rises in jumps.']);
    if (!S.o.sigma || !S.o.rate) out.push(['warn', 'No mutation', 'Children only mix their parents’ weights, so nothing new ever appears: once the population is all copies of a few networks, evolution stops.']);
    if (!S.o.elite) out.push(['warn', 'No elite', 'Even the best network is replaced by children every generation, and a child can be worse than its parents: watch the best line drop now and then.']);
    if (S.o.sigma >= 1) out.push(['warn', 'Mutations are large', 'A mutation this size can wreck a good network: children of the best are often worse than their parents.']);
    const n = h.length, gain = n > 10 ? h[n - 1].best - h[Math.max(0, n - 11)].best : null;
    if (gain !== null && Math.abs(gain) < 1 && last.best < MAXT * 0.9) out.push(['info', 'On a plateau', 'The best has not improved in the last generations. A bigger population or larger mutations search more widely.']);
    out.push(['info', `${SIZE} weights, ${S.o.pop} networks a generation`, `Gradient methods need a loss they can differentiate; evolution needs only a score. It shines for small networks and odd objectives, and was used to evolve networks for robots and games long before deep RL (NEAT, 2002); OpenAI showed in 2017 that evolution strategies scale to large ones.`]);
    return out;
  };
  EV.runTry = function (L, t) {
    const S = L.S, retrain = (msg, n) => { L.test = null; L.fresh(); L.watch = false; L.buildAgain(); L.renderBar(); L.fastForward(n || t.n || 60); if (msg) NA.ui.toast.show(msg); };
    if (t.action === 'watch') { if (S.play) { EV.listen(L, false); S.play = null; } L.test = null; L.watch = true; L.buildAgain(); L.renderBar(); L.play(); NA.ui.toast.show('Watching it evolve: a generation a second, and the board plays each new best network.'); }
    else if (t.action === 'train') { L.watch = false; L.test = null; L.fastForward(t.n || 60); }
    else if (t.action === 'scratch') retrain('A new population of random networks.');
    else if (t.action === 'test') L.startTest();
    else if (t.action === 'play') EV.startPlay(L);
    else if (t.action === 'noMutation') { S.o.sigma = 0; retrain('No mutation (σ = 0): children only mix their parents. Evolving from scratch.'); }
    else if (t.action === 'bigMutation') { S.o.sigma = 1.5; retrain('Large mutations (σ = 1.5). Evolving from scratch.'); }
    else if (t.action === 'tinyPop') { S.o.pop = 6; retrain('A population of 6. Evolving from scratch.'); }
    else if (t.action === 'noElite') { S.o.elite = 0; retrain('No elite: even the best is replaced each generation. Evolving from scratch.'); }
    else if (t.action === 'default') { Object.assign(S.o, EV.DEFAULTS); L.test = null; L.fresh(); L.buildAgain(); L.renderBar(); }
  };
  EV.measured = function (L) {
    const S = L.S, h = S.history, sc = lastScore(h), first = h.find((q) => q.best >= MAXT);
    if (!h.length) return '';
    return `Measured: after ${fmtInt(S.n)} generations the best keeps the pole up ${sc ? fmtPct(sc.score, 0) : '—'} of the time from new starts${first ? `; a network first lasted all ${MAXT} steps in generation ${fmtInt(first.n)}` : ''}.`;
  };
  EV.boardInfo = (L) => ({ kicker: 'Evolution', title: 'A network evolved, not trained', about: [
    `No gradients here, and no reward for any single move: a population of ${L.S.o.pop} small networks, each with its own ${SIZE} weights, takes turns on the cart. The ones that keep the pole up longest pass their weights on; their children mix two parents and mutate a little. The bars under the cart are the last generation, fittest first.`,
    `At first most networks push one way and the pole falls in a few dozen steps. A lucky few lean the right way for a moment longer, and within a few generations one keeps it up for all ${fmtInt(MAXT)} steps. Then the rest catch up: their children inherit what works, and the population's average climbs over dozens of generations. The board plays the best network so far.`,
    'Beside the cart: the best network’s computation for the step shown, its 4 inputs, 8 tanh neurons and 2 outputs; it pushes the way of the larger output. Test run plays it slowly, one step at a time. Try it yourself: balance it with ← and →, then see how long the network lasts from your start.',
    'This is neuroevolution: evolving neural networks instead of training them, as NEAT (2002) did and as evolution strategies did at scale (OpenAI, 2017). It needs only a score, so it works where there is no gradient to follow; in the model builder, Genetic algorithm trains any network you build this way.'], code: EV.py(L) });
  EV.archText = (L) => `${L.S.o.pop} networks of ${NIN} → ${NH} tanh → ${NOUT}, ${SIZE} weights each, bred by a genetic algorithm`;
  EV.stateText = (L) => { const S = L.S; if (L.test) return 'Testing'; if (S.play) return S.play.over ? 'Fallen' : 'Your turn'; return L.training ? (L.watch ? 'Watching it evolve' : 'Evolving') : S.n === 0 ? 'Not evolved' : 'Paused'; };
  // the gallery's picture: a cart with its pole, and a row of bars
  EV.glyph = function (ctx, W, Hh) {
    NA.rlEnvs.draw(ctx, { x: 4, y: 0, w: W - 8, h: Hh * 0.72 }, 'pole', { x: -0.2, th: 0.12, a: 1, t: 0 }, theme);
    const n = 12, bw = (W - 16) / n;
    for (let i = 0; i < n; i++) { const h = (Hh * 0.22) * (1 - i / (n + 2)); ctx.fillStyle = theme.c(i < 2 ? 's2' : 's1', 0.8); ctx.fillRect(8 + i * bw + 1, Hh - 2 - h, bw - 2, h); }
  };

  // ------------------------------------------------------------------ the architecture: the evolution loop
  EV.archTree = function (L, size) {
    const S = L.S, Pr = EV.probe(L), wide = size.w / size.h >= 1.05, D = NA.deep, w = Pr.w, X = { wide };
    const R = wide ? { pop: [0, 0.04, 0.22, 0.44], net: [0.28, 0, 0.44, 0.6], world: [0.78, 0.04, 0.22, 0.44], select: [0.28, 0.7, 0.44, 0.28] }
      : { pop: [0, 0, 0.47, 0.18], world: [0.53, 0, 0.47, 0.18], net: [0, 0.24, 1, 0.42], select: [0, 0.72, 1, 0.26] };
    const hidden = (j) => D.neuron(X, {
      title: `Hidden neuron ${j + 1}`, sub: `tanh: ${fmt(Pr.h[j], 3)}`, tone: 's1',
      terms: NA.range(NIN).map((i) => ({ a: Pr.x[i], w: w[OFF.W1 + i * NH + j] })), bias: w[OFF.b1 + j], act: TANH,
      inLabel: (i) => EV.INPUTS[i], wLabel: (i) => `weight from the ${EV.INPUTS[i]}`, inWhat: 'the 4 numbers it sees',
      vars: { w: `W1[:, ${j}]`, a: 'x', b: `b1[${j}]` }, codeWhat: `as one column of the ${NIN} × ${NH} matrix W1`, outName: 'h',
      about: [`One of the network’s ${NH} hidden neurons, at step ${Pr.st.t}: a weighted sum of the 4 numbers it sees, plus a bias, bent by tanh into −1 … 1. Evolution set its 5 numbers: no gradient ever touched them, only mutations that happened to keep the pole up longer.`],
    });
    const output = (k) => Object.assign(D.neuron(X, {
      title: `Output: ${EV.ACTIONS[k]}`, sub: `${fmt(Pr.o[k], 3)}${k === Pr.a ? ': the larger, it does this' : ''}`, tone: k === Pr.a ? 's2' : 's4',
      terms: NA.range(NH).map((j) => ({ a: Pr.h[j], w: w[OFF.W2 + j * NOUT + k] })), bias: w[OFF.b2 + k], act: LINEAR,
      inLabel: (j) => `hidden neuron ${j + 1}`, wLabel: (j) => `weight from hidden neuron ${j + 1}`, inWhat: `the ${NH} hidden neurons`,
      vars: { w: `W2[:, ${k}]`, a: 'h', b: `b2[${k}]` }, codeWhat: `as one column of the ${NH} × ${NOUT} matrix W2`, outName: 'o',
      about: [`The score for “${EV.ACTIONS[k]}”: a weighted sum of the 8 hidden neurons. It pushes the way of the larger of the two outputs; nothing makes them probabilities or values, they only have to come out in the right order.`],
    }), { id: `out${k}` });
    const HID = [0.36, 0.08, 0.16, 0.84], OUT = [[0.72, 0.18, 0.28, 0.26], [0.72, 0.56, 0.28, 0.26]];
    const net = {
      id: 'net', title: 'The best network', sub: `${NIN} → ${NH} tanh → ${NOUT}, ${SIZE} weights`, tone: 's1',
      summary: (ctx, r) => { EV.drawNet(ctx, r, Pr, clamp(r.h / 9, 7, 11)); },
      inner: (ctx, c, Sk) => {
        const u = Sk.u(c, 8, 12), ri = Sk.place(c, [0, 0.08, 0.28, 0.84]), rh = Sk.place(c, HID);
        for (let i = 0; i < NIN; i++) Sk.text(ctx, `${EV.INPUTS[i]}: ${fmt(Pr.x[i], 2)}`, ri.x, ri.y + (i + 0.5) * ri.h / NIN, { size: u * 0.9, color: theme.c('ink-2'), maxW: ri.w });
        Sk.text(ctx, '8 hidden neurons (tanh)', rh.x + rh.w / 2, rh.y - 8, { size: u * 0.8, color: theme.c('muted'), align: 'center' });
        const hs = rh.h / NH;
        for (let j = 0; j < NH; j++) { ctx.fillStyle = theme.divColor(Pr.h[j] * 0.95, 0.95); ctx.beginPath(); ctx.arc(rh.x + rh.w / 2, rh.y + (j + 0.5) * hs, Math.min(hs, rh.w) * 0.36, 0, Math.PI * 2); ctx.fill(); }
        Sk.arrow(ctx, ri.x + ri.w + 4, c.y + c.h * 0.5, rh.x - 4, c.y + c.h * 0.5, { lw: 1.3 });
      },
      grids: [{ id: 'hid', rect: HID, rows: NH, cols: 1, square: true, make: (i) => hidden(i) }],
      children: [Object.assign(output(0), { rect: OUT[0] }), Object.assign(output(1), { rect: OUT[1] })],
      about: [`The fittest network of the last generation, on the state being looked at (step ${Pr.st.t}${Pr.why === 'test' ? ' of the test' : Pr.why === 'you' ? ' of your run' : ''}): the 4 numbers it sees, 8 tanh neurons, and 2 outputs. It pushes the way of the larger output: here, ${EV.ACTIONS[Pr.a]}.`,
        `Its ${SIZE} weights are its genome. Nothing in it was learned by gradient descent: every number is a parent’s, sometimes moved by a mutation, kept because the networks that had it balanced longer. ${NA.openHint('a neuron', { cap: true })}: it opens down to the transistors.`],
      code: EV.netPy(),
    };
    const popNode = {
      id: 'pop', title: 'The population', sub: S.evo.last ? `generation ${fmtInt(S.evo.last.gen)}: ${S.evo.last.ranked.length} networks` : `${S.o.pop} random networks`, tone: 's7',
      summary: (ctx, r) => { EV.drawPopulation(ctx, r, S); },
      inner: (ctx, c, Sk) => { EV.drawPopulation(ctx, Sk.place(c, [0, 0.05, 1, 0.9]), S); },
      about: [`${S.o.pop} networks of the same shape, each with its own ${SIZE} weights. The first generation is random. Each generation every network plays two episodes from the same two starts, and its fitness is the average number of steps the pole stayed up. The bars are the last generation judged, fittest first.`],
      code: 'population = [rng.normal(0, 0.5, SIZE) for _ in range(POP)]   # each network: one vector of weights\nfitness = [play(w) for w in population]                        # steps the pole stayed up',
    };
    const world = {
      id: 'world', title: 'The pole: fitness', sub: `steps it stays up, up to ${MAXT}`, tone: 's2',
      summary: (ctx, r) => { NA.rlEnvs.draw(ctx, r, 'pole', { x: Pr.st.x, th: Pr.st.th, a: Pr.end ? null : Pr.a, t: Pr.st.t }, theme); },
      inner: (ctx, c, Sk) => { NA.rlEnvs.draw(ctx, Sk.place(c, [0, 0.04, 0.55, 0.8]), 'pole', { x: Pr.st.x, th: Pr.st.th, a: Pr.end ? null : Pr.a, t: Pr.st.t }, theme); Sk.lines(ctx, ['a cart on a 4.8-unit track, a 1 m pole', 'push left or right: 10 N', 'falls past 12°, or off the track', `fitness: steps up (${fmtInt(MAXT)} at most)`, 'averaged over 2 episodes', 'no reward for any single move'], Sk.place(c, [0.6, 0.06, 0.4, 0.7]), { size: Sk.u(c, 9, 13) }); },
      about: ['The classic balancing problem (Barto, Sutton and Anderson, 1983): every 0.02 s the network pushes the cart left or right, and the pole swings by the laws of motion. Evolution never sees a single move’s effect: only the total, how long the pole stayed up, decides who has children.'],
      code: EV.polePy(),
    };
    const select = {
      id: 'select', title: 'Selection and breeding', sub: `the best ${S.o.elite} carry on; the rest are children`, tone: 's3',
      summary: (ctx, r) => { K().lines(ctx, ['parents: best of 3', 'child: each weight from one parent', `mutate ${fmtPct(S.o.rate, 0)} by N(0, ${fmt(S.o.sigma, 2)})`], r, { mono: true, size: clamp(r.h / 4.2, 7, 11) }); },
      inner: (ctx, c, Sk) => { Sk.lines(ctx, [`elitism: the best ${S.o.elite} pass on unchanged`, 'tournament: each parent is the fitter of 3 drawn at random', 'crossover: each weight from one parent or the other', `mutation: ${fmtPct(S.o.rate, 0)} of a child’s weights move by N(0, ${fmt(S.o.sigma, 2)})`, `then the next generation of ${S.o.pop} plays`], Sk.place(c, [0, 0.06, 1, 0.9]), { size: Sk.u(c, 9, 13), mono: true }); },
      about: ['Selection keeps what works: the fitter a network, the more likely it is a parent, and the very best are copied unchanged (elitism), so the best can never get worse. Crossover mixes two good networks; mutation brings in something new. Too little mutation and the population ends up as copies of each other; too much and children lose what their parents knew.'],
      code: EV.breedPy(),
    };
    return {
      id: 'evolution', title: 'Evolution: a genetic algorithm', sub: 'a population → each plays → fitness → the best breed → the next generation', tone: 's1',
      inner: (ctx, c, Sk) => {
        const at = (k) => Sk.place(c, R[k]), A = (p, q, o) => Sk.arrow(ctx, p[0], p[1], q[0], q[1], Object.assign({ lw: 1.4, head: 7 }, o));
        const p = at('pop'), n = at('net'), wd = at('world'), s = at('select'), u = Sk.u(c, 9, 12);
        const tag = (t, x, y, al = 'center') => Sk.text(ctx, t, x, y, { size: u * 0.9, color: theme.c('ink-2'), align: al });
        if (wide) {
          A([p.x + p.w + 4, p.y + p.h / 2], [n.x - 4, p.y + p.h / 2]); tag('each one', (p.x + p.w + n.x) / 2, p.y + p.h / 2 - 9);
          A([n.x + n.w + 4, wd.y + wd.h / 2], [wd.x - 4, wd.y + wd.h / 2]); tag('plays', (n.x + n.w + wd.x) / 2, wd.y + wd.h / 2 - 9);
          A([wd.x + wd.w / 2, wd.y + wd.h + 4], [s.x + s.w + 4, s.y + s.h / 2], { bend: [0, c.h * 0.2] }); tag('fitness', wd.x + wd.w / 2 + 6, (wd.y + wd.h + s.y) / 2 + 10, 'left');
          A([s.x - 4, s.y + s.h / 2], [p.x + p.w / 2, p.y + p.h + 4], { bend: [0, c.h * 0.2], dash: [5, 4] }); tag('the next generation', p.x + 4, p.y + p.h + u * 2.2, 'left');
        } else {
          A([p.x + p.w / 2, p.y + p.h + 3], [n.x + n.w * 0.25, n.y - 3]); tag('each network', p.x + p.w / 2 + 6, (p.y + p.h + n.y) / 2, 'left');
          A([n.x + n.w * 0.75, n.y - 3], [wd.x + wd.w / 2, wd.y + wd.h + 3]); tag('plays', wd.x + wd.w / 2 + 6, (wd.y + wd.h + n.y) / 2, 'left');
          A([n.x + n.w / 2, n.y + n.h + 3], [s.x + s.w / 2, s.y - 3]); tag('fitness, the best breed', s.x + s.w / 2 + 6, (n.y + n.h + s.y) / 2, 'left');
        }
      },
      children: [Object.assign(net, { rect: R.net }), Object.assign(popNode, { rect: R.pop }), Object.assign(world, { rect: R.world }), Object.assign(select, { rect: R.select })],
      about: [`Neuroevolution: instead of following a gradient, a population of ${S.o.pop} networks is bred. Each plays; the fittest are copied or become parents; children mix two parents’ weights and mutate a little; and the next generation plays. ${NA.openHint('a box', { cap: true })}.`,
        'It needs only a score at the end, not a loss to differentiate or a reward for each move, which is why it works for odd objectives and non-differentiable parts; the price is many more plays than gradient methods need on large networks.'],
      code: EV.py(L),
    };
  };

  // ------------------------------------------------------------------ in Python
  EV.netPy = () => `import numpy as np

NIN, NH, NOUT = ${NIN}, ${NH}, ${NOUT}
SIZE = NIN * NH + NH + NH * NOUT + NOUT          # ${SIZE} weights: the genome

def unpack(w):
    """The genome as the network's matrices: W1 (${NIN} x ${NH}), b1, W2 (${NH} x ${NOUT}), b2."""
    i = NIN * NH
    return w[:i].reshape(NIN, NH), w[i:i + NH], w[i + NH:i + NH + NH * NOUT].reshape(NH, NOUT), w[-NOUT:]

def act(w, x):
    W1, b1, W2, b2 = unpack(w)
    h = np.tanh(x @ W1 + b1)                         # 8 hidden neurons
    o = h @ W2 + b2                                  # push left, push right
    return int(o[1] > o[0])                          # the way of the larger output`;
  EV.polePy = () => `import math

def play(w, rng, max_steps=${MAXT}):
    """One episode: how many steps the pole stays up (Barto, Sutton and Anderson, 1983)."""
    x = rng.uniform(-0.5, 0.5)                       # a start a little off: the cart up to half a unit from the middle,
    v, th, om = rng.uniform(-0.15, 0.15, 3)          # the pole up to 8.6° from upright, moving a little
    for t in range(max_steps):
        obs = np.array([x / 2.4, v / 3, th / 0.2095, om / 3.5])
        force = 10.0 if act(w, obs) else -10.0
        ct, st = math.cos(th), math.sin(th)
        tmp = (force + 0.05 * om * om * st) / 1.1
        acc = (9.8 * st - ct * tmp) / (0.5 * (4 / 3 - 0.1 * ct * ct / 1.1))
        xacc = tmp - 0.05 * acc * ct / 1.1
        x, v, th, om = x + 0.02 * v, v + 0.02 * xacc, th + 0.02 * om, om + 0.02 * acc
        if abs(x) > 2.4 or abs(th) > 0.2095:
            return t + 1                             # it fell
    return max_steps`;
  EV.breedPy = () => `def next_generation(population, fitness, rng, elite=2, rate=0.1, sigma=0.3):
    order = np.argsort(fitness)[::-1]                # fittest first
    nxt = [population[i].copy() for i in order[:elite]]   # elitism: the best carry on unchanged
    def parent():                                    # tournament: the fitter of 3 drawn at random
        pick = rng.integers(len(population), size=3)
        return population[pick[np.argmax(np.asarray(fitness)[pick])]]
    while len(nxt) < len(population):
        a, b = parent(), parent()
        child = np.where(rng.random(SIZE) < 0.5, a, b)   # crossover: each weight from one parent
        mutate = rng.random(SIZE) < rate
        nxt.append(child + mutate * rng.normal(0, sigma, SIZE))   # mutation
    return nxt`;
  EV.py = (L) => {
    const o = L.S.o;
    return `${EV.netPy()}


${EV.polePy()}


${EV.breedPy()}


rng = np.random.default_rng(7)
POP = ${o.pop}
population = [rng.normal(0, 0.5, SIZE) for _ in range(POP)]   # random networks
for gen in range(100):
    seed = 1000 + gen                                # the same two starts for everyone this generation
    fitness = [np.mean([play(w, np.random.default_rng(seed + e)) for e in range(2)]) for w in population]
    print(f"generation {gen + 1:3d}  best {max(fitness):5.1f}  average {np.mean(fitness):5.1f}")
    population = next_generation(population, fitness, rng, elite=${o.elite}, rate=${NA.fmtCode(o.rate, 2)}, sigma=${NA.fmtCode(o.sigma, 2)})`;
  };

  NA.rlEvo = EV;
})(window.NA);
