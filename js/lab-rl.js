/* lab-rl.js — the reinforcement learning lab: an agent learns from rewards alone, with no examples of good play.
   Three examples share it. A maze, learned with a table of Q-values (one maze at a time) or with a Q-network that
   sees what is around it and so finds its way through mazes it never met (DQN). Three-line, a 4 × 4 board game it
   learns by playing against itself, with a table of positions or a small network. Four-line, a 5 × 5 board game
   with far too many positions for a table, learned the same way with a network. Train it, watch it learn one move
   at a time, test it slowly with what its model computes drawn beside the board at every step, and play against it. */
(function (NA) {
  'use strict';
  const { el, clear, clamp, fmt, fmtPct, fmtInt, theme } = NA;
  const App = NA.app, H = NA.labHost, UI = NA.labUI, RL = NA.rl;
  const EVERY = { three: 250, four: 250 }; // self-play games between two checks of how well it plays
  const WATCH_MS = { maze: 110, three: 420, four: 420 };
  const TEST_MS = { maze: 750, three: 1300, four: 1300 }, SPEED = { slow: 1.8, normal: 1, fast: 0.35 }; // a test's pace
  const NAMES = { maze: 'Maze', three: 'Three-line', four: 'Four-line' };
  const STONE = ['', 'blue', 'orange'];

  const LAB = H.transport({ id: 'rl', screens: { network: 'Board', data: 'Agent', insp: 'Inspect' }, budget: 12, version: 0, st: {}, centerView: 'board',
    show: { values: true, arrows: true, inside: true }, models: { maze: 'table', three: 'table', four: 'net' }, mazeId: 'pits', extra: [], speed: 'normal', test: null });

  // ------------------------------------------------------------------ the mazes
  LAB.mazeList = function () { return RL.MAZE_SET.concat(this.extra); };
  LAB.mazeDef = function (id = this.mazeId) { return this.mazeList().find((m) => m.id === id) || RL.MAZE_SET[0]; };
  LAB.mazeName = function () { return this.mazeDef().name; };
  let NAMED = null;
  const namedMazes = () => NAMED || (NAMED = RL.MAZE_SET.map((m) => new RL.Maze(m.rows)));
  // a new random maze, added to the list (the last six are kept)
  LAB.newMaze = function () {
    const n = (this.made = (this.made || 0) + 1), rows = RL.randomMaze(RL.rand((Date.now() ^ Math.imul(n, 2654435761)) >>> 0));
    this.extra.push({ id: `r${n}`, name: `Random maze ${n}`, rows, random: true });
    if (this.extra.length > 6) this.extra.shift();
    this.setMaze(`r${n}`);
  };
  LAB.setMaze = function (id) { if (this.training) this.pause(); this.mazeId = id; this.watch = false; this.switchAgent(); };
  LAB.setModel = function (m) { if (this.training) this.pause(); this.models[this.ex] = m; this.watch = false; if (this.S.play) this.S.play = null; this.switchAgent(); };
  // after the maze or the model changes: its agent, and every panel that shows its settings
  LAB.switchAgent = function () { this.use(this.ex); this.buildReadouts(); this.buildAgain(); this.renderBar(); this.tabs.select(this.tabId); this.arch.dirty = true; if (this.centerView === 'arch') this.arch.refreshFocus(); else this.info.set(BOARD_INFO(this)); };

  // ------------------------------------------------------------------ the agents
  /* Each example and model keeps its own agent, so switching keeps what each has learned: a Q-table per maze, one
     Q-network for all mazes, a table and a network for Three-line, a network for Four-line. */
  LAB.agentKey = function (ex = this.ex) {
    const m = this.models[ex];
    return ex === 'maze' ? (m === 'net' ? 'maze/net' : `maze/table/${this.mazeId}`) : ex === 'three' ? `three/${m}` : 'four';
  };
  LAB.state = function (key) {
    if (this.st[key]) return this.st[key];
    const ex = key.split('/')[0], model = ex === 'four' ? 'net' : key.split('/')[1];
    const S = { key, id: ex, model, n: 0, rng: RL.rand({ maze: 11, three: 22, four: 33 }[ex]), ev: RL.rand(99), history: [] };
    if (ex === 'maze') {
      S.mazeId = this.mazeId; S.maze = new RL.Maze(this.mazeDef().rows); S.sel = S.maze.start;
      if (model === 'table') { S.Q = new RL.QTable(S.maze.n, 4); S.o = { eps: 0.1, alpha: 0.5, gamma: 0.95 }; }
      else { S.net = new RL.MazeNet(64, 7); S.agent = new RL.DQN(S.net); S.gen = RL.rand(5); S.o = { eps: 0.05, lr: 0.01, gamma: 0.97 }; }
    } else {
      S.game = RL.games[ex];
      S.o = model === 'table' ? { eps: 0.1, alpha: 0.2 } : { eps: 0.1, lr: 0.05 };
      S.brain = model === 'table' ? new RL.ValueTable(S.game) : new RL.ValueNet(S.game.n, ex === 'three' ? 32 : 64, 7);
      S.window = [0, 0, 0]; // self-play results since the last check: draws, first player won, second player won
      S.shown = null; // the game on the board: the last self-play game, one being watched, or yours
    }
    return (this.st[key] = S);
  };
  // a new agent that knows nothing, keeping the settings
  LAB.fresh = function (key = this.agentKey()) {
    const old = this.st[key];
    delete this.st[key];
    const S = this.state(key);
    if (old) { Object.assign(S.o, old.o); if (old.maze) S.maze.slip = old.maze.slip; }
    if (key === this.agentKey()) this.use(this.ex);
    return S;
  };
  // the networks trained offline (js/rl-weights.js, with this same code)
  LAB.loadTrained = function (ex = this.ex) {
    const W = NA.rlWeights;
    if (!W) return false;
    if (ex === 'four') {
      const S = this.state('four');
      S.brain = RL.ValueNet.from(W.four); S.n = W.games; S.pretrained = true; S.window = [0, 0, 0];
      S.history = W.curve.map(([n, rw, rl, tw, tl]) => ({ n, rnd: { win: rw, loss: rl, draw: Math.max(0, 1 - rw - rl) }, rules: { win: tw, loss: tl, draw: Math.max(0, 1 - tw - tl) } }));
    } else if (ex === 'maze' && W.maze) {
      const S = this.state('maze/net');
      S.net = RL.MazeNet.from(W.maze); S.agent = new RL.DQN(S.net); S.n = W.mazeEpisodes; S.pretrained = true; S.acc = null;
      S.history = W.mazeCurve.map(([n, named, fresh, route]) => ({ n, named, fresh, freshRoute: route }));
    } else return false;
    if (this.ex === ex) this.use(ex);
    return true;
  };
  // the trained networks load the first time only: starting from nothing must stay at nothing
  LAB.use = function (ex) {
    const key = this.agentKey(ex), first = !this.st[key];
    this.ex = ex; this.S = this.state(key);
    if (first && (key === 'four' || key === 'maze/net')) this.loadTrained(ex);
    const S = this.S;
    // one network for every maze: it is shown on the chosen one
    if (S.model === 'net' && S.id === 'maze' && S.mazeId !== this.mazeId) { const slip = S.maze.slip; S.maze = new RL.Maze(this.mazeDef().rows); S.maze.slip = slip; S.mazeId = this.mazeId; S.sel = S.maze.start; S.walk = S.walkDone = S.update = null; }
    this.step = S.n; this.test = null;
    this.bump();
  };

  // ------------------------------------------------------------------ training
  LAB.trainChunk = function (budget, maxSteps) {
    if (this.watch) return 0; // watching: one move at a time, in frame()
    const S = this.S, t0 = performance.now();
    let k = 0;
    do {
      if (S.id === 'maze') this.episodeDone(S.model === 'net' ? this.netEpisode() : RL.mazeEpisode(S.maze, S.Q, S.o, S.rng));
      else this.gameDone(RL.selfPlay(S.game, S.brain, S.o, S.rng));
      k++;
    } while (performance.now() - t0 < budget && k < maxSteps);
    this.step = S.n; this.bump();
    return k;
  };
  // the network learns on random mazes, easy ones first (none of the named mazes)
  LAB.netEpisode = function () {
    const S = this.S, m = RL.curriculumMaze(S.n, S.gen);
    m.slip = S.maze.slip;
    return RL.netEpisode(m, S.agent, { eps: RL.netSchedule(S.n, S.o.eps).eps, lr: S.o.lr, gamma: S.o.gamma }, S.rng);
  };
  // the maze: now and then, the average length and return of the episodes, and how its best moves do (no exploring):
  // on this maze (a table), or on the named mazes and 100 new ones (the network)
  LAB.episodeDone = function (e) {
    const S = this.S, net = S.model === 'net', a = S.acc || (S.acc = { n: 0, steps: 0, ret: 0, goal: 0 });
    S.n++; S.lastEp = e;
    a.n++; a.steps += e.steps; a.ret += e.total; a.goal += e.end === 'goal' ? 1 : 0;
    if (a.n >= (net ? 500 : 20)) {
      const q = { n: S.n, steps: a.steps / a.n, ret: a.ret / a.n, goal: a.goal / a.n };
      if (net) { const nm = RL.netEval(namedMazes(), S.net, S.ev), fr = RL.netEval(RL.testMazes(), S.net, S.ev); Object.assign(q, { named: nm.success, fresh: fr.success, freshRoute: fr.steps }); }
      else { const g = RL.mazeEval(S.maze, S.Q, S.ev, 20); Object.assign(q, { greedy: g.success, route: g.steps }); }
      S.history.push(q); S.acc = null; thin(S.history);
    }
  };
  // a board game: every few hundred games, how it does against a random player and against the rule-follower
  LAB.gameDone = function (M) {
    const S = this.S;
    S.n++; S.window[M.winner]++;
    if (!this.watch) S.shown = { b: M.b, line: M.line, order: M.order, winner: M.winner, kind: 'self' };
    if (S.n % EVERY[S.id] === 0) this.check();
  };
  LAB.check = function () {
    const S = this.S, games = S.id === 'four' ? 100 : 160, w = S.window, tot = w[0] + w[1] + w[2] || 1;
    S.history.push({ n: S.n, rnd: RL.evaluate(S.game, S.brain, RL.randomMove, games, S.ev), rules: RL.evaluate(S.game, S.brain, RL.tacticalMove, games, S.ev), first: w[1] / tot, draws: w[0] / tot });
    S.window = [0, 0, 0]; thin(S.history);
  };
  const thin = (h) => { if (h.length > 400) { const keep = h.filter((q, i) => i % 2 === 0 || i === h.length - 1); h.length = 0; h.push(...keep); } };
  // the network's best moves on the maze shown (worked out again only when something changed)
  LAB.here = function () {
    const S = this.S;
    if (!this.hereC || this.hereC.v !== this.version || this.hereC.S !== S) this.hereC = { v: this.version, S, r: RL.netRun(S.maze, S.net, S.ev) };
    return this.hereC.r;
  };
  // watching: it learns one move at a time, slowly enough to follow (the network on the maze shown)
  LAB.watchTick = function () {
    const S = this.S;
    if (S.id === 'maze') {
      const M = S.maze, net = S.model === 'net';
      const w = S.walk || (S.walk = { s: M.start, path: [M.start], steps: 0, total: 0, seen: (() => { const u = new Uint8Array(M.n); u[M.start] = 1; return u; })() });
      if (net) {
        const x = RL.MazeNet.inputs(M, w.s, w.seen), q = Float64Array.from(S.net.forward(x)), explore = S.rng() < RL.netSchedule(S.n, S.o.eps).eps;
        const intended = explore ? Math.floor(S.rng() * 4) : RL.argmax4(q), t = M.step(w.s, intended, S.rng);
        w.seen[t.s] = 1;
        const x2 = t.done ? null : RL.MazeNet.inputs(M, t.s, w.seen);
        let target = t.r, next = null;
        if (x2) { const q2 = S.agent.target.forward(x2); next = Math.max(q2[0], q2[1], q2[2], q2[3]); target += S.o.gamma * next; }
        S.agent.remember({ x, a: intended, r: t.r, x2 });
        S.agent.replay(4, S.o.lr, S.o.gamma, S.rng);
        S.update = { s: w.s, a: t.a, intended, slipped: t.slipped, r: t.r, s2: t.s, done: t.done, before: q[intended], after: S.net.forward(x)[intended], target, next, explore, net: true };
        w.s = t.s; w.path.push(t.s); w.steps++; w.total += t.r;
        if (t.done || w.steps >= 80) { this.episodeDone({ total: w.total, steps: w.steps, end: t.done ? (t.r > 0 ? 'goal' : 'pit') : 'timeout', path: w.path }); S.walkDone = w; S.walk = null; this.pauseUntil = performance.now() + 500; }
      } else {
        // the entry of the move it chose learns, wherever a slip took it
        const intended = S.Q.act(w.s, S.o.eps, S.rng), q0 = S.Q.Q[w.s * 4 + intended], t = M.step(w.s, intended, S.rng);
        const err = S.Q.update(w.s, intended, t.r, t.s, t.done, S.o.alpha, S.o.gamma);
        S.update = { s: w.s, a: t.a, intended, slipped: t.slipped, r: t.r, s2: t.s, done: t.done, err, before: q0, after: S.Q.Q[w.s * 4 + intended] };
        w.s = t.s; w.path.push(t.s); w.seen[t.s] = 1; w.steps++; w.total += t.r;
        if (t.done || w.steps >= M.maxSteps) { this.episodeDone({ total: w.total, steps: w.steps, end: t.done ? (t.r > 0 ? 'goal' : 'pit') : 'timeout', path: w.path }); S.walkDone = w; S.walk = null; this.pauseUntil = performance.now() + 500; }
      }
    } else {
      const M = S.live && !S.live.done ? S.live : (S.live = new RL.Match(S.game, S.brain, S.o, S.rng));
      S.lastMove = M.step();
      S.shown = { b: M.b, line: M.line, order: M.order, winner: M.winner, kind: 'watch', done: M.done };
      if (M.done) { this.gameDone(M); this.pauseUntil = performance.now() + 1100; }
    }
    this.step = S.n; this.bump();
  };
  LAB.onPause = function () { this.bump(); };
  LAB.stepOnce = function () {
    if (this.test) { this.testAuto(false); this.testGo(this.test.k + 1); return; }
    if (this.training) this.pause(); if (this.S.play) this.S.play = null; this.watch ? this.watchTick() : this.trainChunk(0, 1);
  };
  LAB.reset = function () { if (this.training) this.pause(); this.test = null; this.fresh(); this.buildAgain(); this.renderBar(); NA.ui.toast.show(`A new ${NAMES[this.ex]} agent that knows nothing yet. Press Train.`); };
  LAB.bump = function () { this.version++; this.dirty = true; this.inspDirty = true; this.leftDirty = true; this.archStale = true; this.infoStale = true; };
  // Space plays and pauses a test while one is on; the play button starts training (and ends the test)
  const transportToggle = LAB.toggle, transportPlay = LAB.play;
  LAB.toggle = function () { if (this.test) this.testAuto(!this.test.auto); else transportToggle.call(this); };
  LAB.play = function () { if (this.test) { this.test = null; this.renderBar(); } transportPlay.call(this); };

  // ------------------------------------------------------------------ testing: its best moves, slowly
  /* A test is a run made in advance, step by step: the maze from S with its best move on every square, or a game
     against the rule-follower. Each step keeps what its model saw and computed, and plays back at a walking pace
     (or one step at a time) while the board and the picture beside it show that step. */
  LAB.startTest = function () {
    const S = this.S;
    if (this.training) this.pause();
    this.watch = false; if (this.watchChip) this.watchChip.setAttribute('aria-pressed', 'false');
    if (S.play) S.play = null;
    const frames = this.makeFrames();
    this.test = { frames, k: 0, auto: true, at: performance.now() + this.testMs() * 0.6, me: frames.me };
    this.bump(); this.renderBar();
    if (this.centerView === 'arch') this.arch.refreshFocus();
  };
  LAB.runTest = function () { this.startTest(); };
  LAB.testMs = function () { return TEST_MS[this.ex] * SPEED[this.speed]; };
  LAB.makeFrames = function () {
    const S = this.S;
    if (S.id === 'maze') {
      if (S.model === 'net') return RL.netRun(S.maze, S.net, S.rng, true, 60).frames;
      const frames = [];
      let s = S.maze.start;
      for (let k = 0; k < 60; k++) {
        const a = S.Q.best(s, S.rng), t = S.maze.step(s, a, S.rng);
        frames.push({ s, a, went: t.a, slipped: t.slipped, r: t.r, s2: t.s });
        s = t.s;
        if (t.done) break;
      }
      return frames;
    }
    // a game against the rule-follower: it moves first in every other test
    const g = S.game, b = new Int8Array(g.n), me = (this.tests = (this.tests || 0) + 1) % 2 ? 1 : 2, frames = [], order = [];
    let p = 1;
    for (;;) {
      const agent = p === me, best = agent ? RL.bestMove(g, S.brain, b, p, S.rng) : null, mv = best || RL.moveValues(g, S.brain, b, p);
      const m = agent ? best.move : RL.tacticalMove(g, b, p, S.rng);
      frames.push({ b: Int8Array.from(b), p, who: agent ? 'agent' : 'rules', moves: mv.moves, vals: mv.vals, m, order: order.slice() });
      b[m] = p; order.push(m);
      const line = g.wins(b, m);
      if (line || order.length === g.n) { frames.push({ b: Int8Array.from(b), p: 0, over: true, winner: line ? p : 0, line, order: order.slice() }); break; }
      p = 3 - p;
    }
    frames.me = me;
    return frames;
  };
  LAB.testGo = function (k) {
    const T = this.test;
    if (!T) return;
    T.k = clamp(k, 0, T.frames.length - 1); T.at = performance.now() + this.testMs();
    this.bump(); this.updateBar();
  };
  LAB.testAuto = function (on) {
    const T = this.test;
    if (!T) return;
    if (on && T.k >= T.frames.length - 1) T.k = 0;
    T.auto = on; T.at = performance.now() + this.testMs() * 0.5;
    this.bump(); this.updateBar();
  };
  LAB.endTest = function () { this.test = null; this.bump(); this.renderBar(); };
  // how the test went, and where it is
  LAB.testResult = function () {
    const T = this.test, f = T.frames[T.frames.length - 1];
    if (this.S.id === 'maze') return f.r > 0 ? `reached the goal in ${T.frames.length} steps` : f.r === -1 ? `fell into a pit after ${T.frames.length} steps` : `did not get through in ${T.frames.length} steps`;
    return f.winner === T.me ? `it won as ${STONE[T.me]}` : f.winner ? 'the rule-follower won' : 'a draw';
  };
  LAB.testCount = function () {
    const T = this.test, n = T.frames.length, last = T.k >= n - 1;
    if (this.S.id === 'maze') return `Step ${T.k + 1} of ${n}${last ? `: ${this.testResult()}` : ''}`;
    return last ? `Game over: ${this.testResult()}` : `Move ${T.k + 1} of ${n - 1}`;
  };
  LAB.key = function (e) {
    const T = this.test;
    if (!T) return;
    const go = { ArrowRight: T.k + 1, ArrowLeft: T.k - 1, Home: 0, End: T.frames.length - 1 }[e.key];
    if (go !== undefined) { e.preventDefault(); this.testAuto(false); this.testGo(go); }
    else if (e.key === 'Escape') this.endTest();
  };

  // ------------------------------------------------------------------ playing against it
  LAB.startPlay = function (human, quiet) {
    const S = this.S;
    if (this.training) this.pause();
    this.watch = false; if (this.watchChip) this.watchChip.setAttribute('aria-pressed', 'false');
    this.test = null;
    S.play = { b: new Int8Array(S.game.n), human, turn: 1, over: false, winner: 0, line: null, order: [] };
    this.agentAt = human === 1 ? Infinity : performance.now() + 700;
    this.bump(); this.renderBar();
    if (!quiet) NA.ui.toast.show(human === 1 ? `You are ${STONE[1]} and move first: ${NA.tapWord(true)} a cell.` : `You are ${STONE[2]}: it moves first.`);
  };
  LAB.stopPlay = function () { this.S.play = null; this.bump(); this.renderBar(); };
  LAB.place = function (i) {
    const S = this.S, P = S.play;
    P.b[i] = P.turn; P.order.push(i);
    const w = S.game.wins(P.b, i);
    if (w) { P.over = true; P.winner = P.turn; P.line = w; }
    else if (P.order.length === S.game.n) P.over = true;
    else P.turn = 3 - P.turn;
    if (P.over) NA.ui.toast.show(P.winner === P.human ? 'You win.' : P.winner ? 'It wins.' : 'A draw.');
    this.bump(); this.renderBar();
  };
  LAB.humanMove = function (i) {
    const P = this.S.play;
    if (!P || P.over || P.turn !== P.human || P.b[i]) return;
    this.place(i);
    if (!P.over) this.agentAt = performance.now() + 550;
  };

  // ------------------------------------------------------------------ lifecycle
  LAB.activate = function (preset, ui) {
    this.ui = ui; this.preset = preset;
    const ex = preset.example || 'maze';
    this.models[ex] = preset.model || (ex === 'four' ? 'net' : 'table');
    if (ex === 'maze') this.mazeId = preset.maze || 'pits';
    this.watch = false; this.test = null;
    this.use(ex);
    this.buildReadouts();
    this.buildHyper(ui.hyper);
    this.buildLeft(ui.left);
    this.buildCenter(ui.center);
    this.tabs = UI.tabs(ui.right, [{ id: 'overview', label: 'Overview' }, { id: 'train', label: 'Training' }, { id: 'inspect', label: ex === 'maze' ? 'Q-values' : 'Its view' }], (id, body) => this.buildTab(id, body));
    this.offTheme = theme.onChange(() => this.bump());
    this.bump();
  };
  LAB.deactivate = function () { if (this.offTheme) this.offTheme(); if (this.resizeObs) this.resizeObs.disconnect(); if (this.arch) this.arch.destroy(); this.watch = false; this.test = null; };
  LAB.archText = function () {
    const S = this.S;
    if (S.id === 'maze') return S.model === 'net' ? `Q-network ${S.net.nIn} → ${S.net.nH} ReLU → 4, learned on random mazes (DQN)` : `Q-table: ${S.maze.cell.filter((c) => c !== '#').length} squares × 4 moves`;
    return S.model === 'table' ? `value table: ${fmtInt(S.brain.size)} positions, learned by self-play` : `value network ${S.brain.nIn} → ${S.brain.nH} tanh → 1 sigmoid, learned by self-play`;
  };
  LAB.stateText = function () {
    const S = this.S;
    if (this.test) return 'Testing';
    if (S.play) return S.play.over ? 'Game over' : S.play.turn === S.play.human ? 'Your move' : 'Its move';
    return this.training ? (this.watch ? 'Watching it learn' : 'Training') : S.n === 0 ? 'Untrained' : 'Paused';
  };
  LAB.frame = function (now) {
    const S = this.S, T = this.test;
    if (this.training && this.watch) { if (now >= (this.pauseUntil || 0) && now - (this.lastTick || 0) >= WATCH_MS[S.id]) { this.lastTick = now; this.watchTick(); } }
    else this.runFrame();
    // a test plays back at its pace
    if (T && T.auto && now >= T.at) { if (T.k < T.frames.length - 1) this.testGo(T.k + 1); else { T.auto = false; this.updateBar(); } }
    // its move, a moment after yours
    const P = S.play;
    if (P && !P.over && P.turn !== P.human && now >= this.agentAt) { this.agentAt = Infinity; this.place(RL.bestMove(S.game, S.brain, P.b, P.turn, S.rng).move); }
    if (now - (this.lastRo || 0) > 150) { this.lastRo = now; this.readouts(); }
    if (this.dirty && this.canvas.offsetParent && (!this.training || this.watch || now - (this.lastDraw || 0) > 90)) { this.lastDraw = now; this.dirty = false; this.draw(); }
    if (this.centerView === 'arch') {
      if (this.archStale && (!this.training || now - (this.lastArch || 0) > 500)) { this.lastArch = now; this.archStale = false; this.arch.dirty = true; }
      this.arch.frame(now);
    }
    if (this.infoStale && (!this.training || now - (this.lastInfo || 0) > 1500) && !(this.arch && this.arch.anim)) { this.lastInfo = now; this.infoStale = false; if (this.centerView === 'arch') this.arch.refreshFocus(); else this.info.set(BOARD_INFO(this)); }
    if (this.leftDirty && this.ui.left.offsetParent && (!this.training || now - (this.lastLeft || 0) > 400)) { this.lastLeft = now; this.leftDirty = false; }
    if (this.inspDirty && this.tabs.body.offsetParent && (!this.training || now - (this.lastInsp || 0) > 450)) { this.lastInsp = now; this.inspDirty = false; this.updateTab(); }
  };
  LAB.buildReadouts = function () {
    const S = this.S, root = this.ui.readouts;
    clear(root);
    this.ro = UI.readouts(root, S.id === 'maze'
      ? (S.model === 'net'
        ? [{ key: 'n', label: 'Episodes', keep: true }, { key: 'a', label: 'This maze', keep: true, title: 'Its best move on every square, from S, on the maze shown' }, { key: 'b', label: 'New mazes', title: 'The share of 100 random mazes it never trained on that its best moves get through' }]
        : [{ key: 'n', label: 'Episodes', keep: true }, { key: 'a', label: 'Reaches the goal', keep: true, title: 'Greedy route (no exploring) from the start, over 20 tries' }, { key: 'b', label: 'Route', title: 'Steps on the greedy route' }])
      : [{ key: 'n', label: 'Games', keep: true }, { key: 'a', label: 'Beats random', keep: true, title: 'Share of games it wins against a player who moves at random' }, { key: 'b', label: 'Beats rules', title: 'Share it wins against a rule-follower: win if it can, block if it must' }, { key: 'c', label: S.model === 'table' ? 'Positions' : 'Parameters' }]);
  };
  LAB.readouts = function () {
    const S = this.S, h = S.history[S.history.length - 1];
    this.ro.set('n', fmtInt(S.n));
    if (S.id === 'maze' && S.model === 'net') { const r = this.here(); this.ro.set('a', r.end === 'goal' ? `${r.steps} steps` : r.end === 'pit' ? 'falls in a pit' : 'gets lost'); this.ro.set('b', h && h.fresh != null ? fmtPct(h.fresh, 0) : '—'); }
    else if (S.id === 'maze') { this.ro.set('a', h ? fmtPct(h.greedy, 0) : '—'); this.ro.set('b', h && h.greedy ? `${fmt(h.route, 1)} steps` : '—'); }
    else { this.ro.set('a', h ? fmtPct(h.rnd.win, 0) : '—'); this.ro.set('b', h ? fmtPct(h.rules.win, 0) : '—'); this.ro.set('c', S.model === 'table' ? fmtInt(S.brain.size) : fmtInt(S.brain.paramCount)); }
  };

  LAB.buildHyper = function (root) {
    const S = this.S, maze = S.id === 'maze', net = S.model === 'net';
    root.append(
      UI.field('Exploration ε', UI.select([0, 0.05, 0.1, 0.2, 0.3].map((v) => ({ value: v, label: String(v) })), S.o.eps, (v) => { S.o.eps = +v; this.leftDirty = true; }), 'The share of moves it makes at random, to try things out'),
      !net ? UI.field('Learning rate α', UI.select([0.05, 0.1, 0.2, 0.5, 1].map((v) => ({ value: v, label: String(v) })), S.o.alpha, (v) => { S.o.alpha = +v; this.leftDirty = true; }))
        : UI.field('Learning rate', UI.select((maze ? [0.003, 0.01, 0.03] : [0.01, 0.02, 0.05, 0.1]).map((v) => ({ value: v, label: String(v) })), S.o.lr, (v) => { S.o.lr = +v; this.leftDirty = true; })),
      el('div', { class: 'sheet-actions' },
        el('button', { class: 'btn', type: 'button', onclick: () => { NA.mobile && NA.mobile.sheet(false); this.watch = false; this.test = null; this.fastForward(maze ? (net ? 5000 : 500) : 2000); } }, maze ? (net ? 'Train 5,000 mazes' : 'Train 500 episodes') : 'Train 2,000 games'),
        el('button', { class: 'btn', type: 'button', onclick: () => { NA.mobile && NA.mobile.sheet(false); this.reset(); } }, 'Start from nothing')));
  };

  // ------------------------------------------------------------------ left: the task, how it learns
  LAB.buildLeft = function (root) {
    const S = this.S, maze = S.id === 'maze', net = S.model === 'net', body = UI.panel(root, maze ? 'The maze' : NAMES[S.id]);
    this.leftParts = {};
    const P = this.leftParts, set = (key) => (v) => { S.o[key] = v; this.leftDirty = true; };
    P.eps = UI.slider({ label: maze && net ? 'Exploration ε (after its first 24,000 mazes)' : 'Exploration ε (moves made at random)', min: 0, max: 0.5, step: 0.01, value: S.o.eps, format: (v) => v.toFixed(2), onInput: set('eps') });
    P.rate = !net ? UI.slider({ label: 'Learning rate α', min: 0.01, max: 1, step: 0.01, value: S.o.alpha, format: (v) => v.toFixed(2), onInput: set('alpha') })
      : maze ? UI.slider({ label: 'Learning rate', min: 0.001, max: 0.05, step: 0.001, value: S.o.lr, format: (v) => v.toFixed(3), onInput: set('lr') })
        : UI.slider({ label: 'Learning rate', min: 0.005, max: 0.2, step: 0.005, value: S.o.lr, format: (v) => v.toFixed(3), onInput: set('lr') });
    this.watchChip = UI.chip('Watch it learn, move by move', this.watch, (on) => { this.watch = on; this.test = null; this.renderBar(); if (on && !this.training) this.play(); this.bump(); }, 'Slows training down to one move at a time');
    const learn = [P.eps, P.rate];
    const testNote = UI.note(`To see what it does now, without learning, use ${maze ? 'Test run' : 'Watch it play'} above the board: its best moves, slowly, with what its model computes at every step beside the board. ◀ ▶ (or the arrow keys) step through it.`);
    if (maze) {
      P.gamma = UI.slider({ label: 'Discount γ (how much later rewards count)', min: 0.5, max: 0.99, step: 0.01, value: S.o.gamma, format: (v) => v.toFixed(2), onInput: set('gamma') });
      P.slip = UI.slider({ label: 'Slippery floor (moves that go sideways)', min: 0, max: 0.3, step: 0.01, value: S.maze.slip, format: (v) => fmtPct(v, 0), onInput: (v) => { S.maze.slip = v; this.bump(); } });
      learn.push(P.gamma);
      body.append(
        UI.section('The task', UI.note(`Get from S to the green goal (+1). Falling into a red pit ends the run at −1, and every step costs 0.02, so shorter is better. The agent is told nothing else: it learns only from these rewards. Choose the maze above the board${net ? '' : '; each maze gets a table of its own'}.`), P.slip),
        net ? UI.section('How it learns: a Q-network (DQN)', UI.note('A table has a row per square of one maze. This network has none: it sees the 24 squares around it (walls, pits, the goal, where it has been) and which way the goal lies, and outputs a Q-value for each move. It learned on random mazes, easy ones first, as DQN did on Atari games: a replay memory of past moves and a target network. The same weights then work in mazes it never met.'), ...learn)
          : UI.section('How it learns: Q-learning', UI.note('It keeps a table Q(square, move): how much reward it expects from that square if it makes that move and then plays its best. After every move it nudges that entry toward the reward it got plus γ times the best value of where it landed.'), ...learn),
        UI.section('Watch and test', UI.note(net ? 'Watch slows learning down to one move at a time, on the maze shown.' : 'Train runs thousands of episodes a second. Watch slows it down to one move at a time, so you can see it wander, find the goal, and the values spread back from it.'), this.watchChip, testNote));
      if (net) body.append(UI.section('The trained network', UI.note(`It opened with a network trained offline with this same code on ${fmtInt((NA.rlWeights && NA.rlWeights.mazeEpisodes) || 0)} random mazes, none of the named ones. Start from nothing to watch one learn: it takes a minute or two to get through most mazes.`),
        el('div', { class: 'sheet-actions' }, el('button', { class: 'btn', type: 'button', onclick: () => this.runTry({ action: 'trained' }) }, 'Load the trained network'), el('button', { class: 'btn', type: 'button', onclick: () => this.reset() }, 'Start from nothing'))));
    } else {
      const g = S.game;
      body.append(
        UI.section('The game', UI.note(`A game made for this site: two players take turns placing a stone on a ${g.size} × ${g.size} board, ${STONE[1]} first. ${g.k} in a row, across, down or diagonally, wins; a full board is a draw. Play against it with the buttons above the board, or ${NA.tapWord(true)} a cell to start as ${STONE[1]}.`)),
        UI.section(`How it learns: self-play${net ? ' with a neural network' : ''}`, UI.note(!net
          ? 'It plays both sides against itself. It keeps a value for every position it meets: its estimate of the chance that the player who just moved will win. To move, it looks at the position each empty cell would lead to and picks the best. After each move, the value of its previous position moves a little toward the value of the next one, and at the end toward the result.'
          : S.id === 'three' ? 'The same self-play learning, with a small network in place of the table: 32 inputs (where each player’s stones are), 32 hidden neurons, one output: the chance to win. Every lesson changes weights shared by all positions, so it judges positions it has never met, and it learns this game faster than the table does.'
            : 'The same learning as Three-line, but a 5 × 5 board has billions of positions, so instead of a table a small neural network learns to judge them: 50 inputs (where each player’s stones are), 64 hidden neurons, one output: the chance to win. It generalises, so a position it never saw still gets a sensible value.'), ...learn),
        UI.section('Watch and test', UI.note('Watch slows self-play down to one move at a time: you see each game it learns from.'), this.watchChip, testNote));
    }
    if (S.id === 'four') body.append(UI.section('The trained network', UI.note('It opened with a network trained offline with this same code for 100,000 games of self-play. Start from nothing to watch one learn from scratch.'),
      el('div', { class: 'sheet-actions' }, el('button', { class: 'btn', type: 'button', onclick: () => this.runTry({ action: 'trained' }) }, 'Load the trained network'), el('button', { class: 'btn', type: 'button', onclick: () => this.reset() }, 'Start from nothing'))));
  };

  // ------------------------------------------------------------------ centre: the board, its bar, the picture beside it
  LAB.buildCenter = function (root) {
    const maze = this.ex === 'maze';
    this.exSeg = UI.seg([{ value: 'maze', label: 'Maze' }, { value: 'three', label: 'Three-line' }, { value: 'four', label: 'Four-line' }], this.ex, (v) => { if (v !== this.ex) App.loadPreset(`rl-${v}`); }, 'Example');
    this.viewSeg = UI.seg([{ value: 'board', label: maze ? 'Maze' : 'Board' }, { value: 'arch', label: 'Architecture', title: 'The agent and its model, part by part: click a part for details, double-click it or zoom in to open it' }], this.centerView, (v) => this.setCenterView(v), 'View');
    this.boardTools = el('span', { class: 'lab-tools' }, el('span', { class: 'lens-label', text: 'Show' }),
      UI.chip(maze ? 'Values' : 'Its view of each move', this.show.values, (on) => { this.show.values = on; this.dirty = true; }, maze ? 'Colour each square by the value of its best move' : 'In each empty cell, the chance to win it gives the player to move'),
      maze ? UI.chip('Best moves', this.show.arrows, (on) => { this.show.arrows = on; this.dirty = true; }, 'An arrow for the move it rates best on each square') : null,
      UI.chip('Inside its model', this.show.inside, (on) => { this.show.inside = on; this.stage.classList.toggle('rl-tall', on); this.dirty = true; }, 'Beside the board: what its model computes for the state being looked at'));
    root.append(el('div', { class: 'lab-toolbar' }, this.exSeg, this.viewSeg, this.boardTools));
    this.bar = el('div', { class: 'lab-toolbar rl-bar' });
    root.append(this.bar);
    this.stage = el('div', { class: 'lab-stage rl-stage' + (this.show.inside ? ' rl-tall' : '') });
    this.canvas = el('canvas', { 'aria-label': maze ? 'The maze, with what the agent has learned about each square, and beside it what its model computes' : 'The board, and beside it what its model computes' });
    this.stage.append(this.canvas);
    this.archStage = el('div', { class: 'lab-stage arch-stage' });
    this.arch = new NA.ArchView(this.archStage, { label: 'The agent and its model: click a part for details; double-click it or zoom in to open it', kicker: 'Architecture', build: (size) => NA.archRL.build(this, size), onFocus: (node, path) => { if (this.centerView === 'arch') this.info.set(this.arch.infoFor(node, path)); } });
    root.append(this.stage, this.archStage);
    this.info = UI.infoPanel(root, { open: !(NA.mobile && NA.mobile.mobile) && root.clientHeight >= 680 });
    root.append(this.foot = el('div', { class: 'lab-foot' }, ...(maze
      ? [key('pos', 'goal +1'), key('neg', 'pit −1'), el('span', { class: 'lg-item', text: 'each step −0.02 · arrow: its best move · colour: the value of that move' })]
      : [el('span', { class: 'lg-item' }, el('i', { class: 'dot dot-filled', style: { background: 'var(--s1)', borderColor: 'var(--s1)' } }), `${STONE[1]} moves first`), el('span', { class: 'lg-item' }, el('i', { class: 'dot', style: { border: '2px solid var(--s2)', boxSizing: 'border-box' } }), STONE[2]), el('span', { class: 'lg-item', text: 'numbers: its chance to win after each move, for the player to move' })])));
    this.canvas.addEventListener('click', (e) => this.clickBoard(e));
    this.canvas.addEventListener('pointermove', (e) => { const i = this.cellAt(e); if (i !== this.hoverCell) { this.hoverCell = i; this.dirty = true; if (!this.S.maze) this.archStale = true; } });
    this.canvas.addEventListener('pointerleave', () => { this.hoverCell = -1; this.dirty = true; });
    this.resizeObs = new ResizeObserver(() => { this.dirty = true; });
    this.resizeObs.observe(this.stage);
    this.renderBar();
    this.setCenterView(this.centerView);
  };
  const key = (tok, text) => el('span', { class: 'lg-item' }, el('i', { class: 'key-sq', style: { background: `var(--${tok})` } }), text);
  // the bar above the board: which maze and model; play, test, and the test's controls
  LAB.renderBar = function () {
    const bar = this.bar, S = this.S;
    if (!bar) return;
    clear(bar);
    const T = this.test, maze = S.id === 'maze', btn = (label, on, cls = 'btn', title) => el('button', { class: cls, type: 'button', title: title || null, onclick: on }, label);
    if (maze && !T) bar.append(UI.field('Maze', UI.select(this.mazeList().map((m) => ({ value: m.id, label: m.name })).concat([{ value: '__new', label: 'New random maze…' }]), this.mazeId, (v) => (v === '__new' ? this.newMaze() : this.setMaze(v)), { 'aria-label': 'Maze' })));
    if ((maze || S.id === 'three') && !T) bar.append(UI.seg(maze ? [{ value: 'table', label: 'Q-table', title: 'One maze: a row of 4 values for each square' }, { value: 'net', label: 'Q-network', title: 'Any maze: a network that sees what is around it' }] : [{ value: 'table', label: 'Value table', title: 'A value for every position it has met' }, { value: 'net', label: 'Value network', title: 'A small network that judges any position' }], S.model, (v) => this.setModel(v), 'Model'));
    this.cnt = null; this.autoBtn = null;
    if (T) {
      const ic = (glyph, label, on) => el('button', { class: 'btn btn-quiet rl-ic', type: 'button', title: label, 'aria-label': label, onclick: on }, glyph);
      this.autoBtn = ic(T.auto ? '❚❚' : '▶', T.auto ? 'Pause' : 'Play', () => this.testAuto(!this.test.auto));
      bar.append(el('span', { class: 'rl-test' }, ic('⏮', 'Back to the start', () => { this.testAuto(false); this.testGo(0); }), ic('◀', 'One step back', () => { this.testAuto(false); this.testGo(this.test.k - 1); }), this.autoBtn, ic('▶▏', 'One step on', () => { this.testAuto(false); this.testGo(this.test.k + 1); })),
        this.cnt = el('span', { class: 'rl-count', text: this.testCount() }),
        UI.field('Speed', UI.select([{ value: 'slow', label: 'Slow' }, { value: 'normal', label: 'Normal' }, { value: 'fast', label: 'Fast' }], this.speed, (v) => { this.speed = v; }, { 'aria-label': 'Speed' })),
        ic('↻', maze ? 'A new run' : 'A new game', () => this.startTest()), btn('End test', () => this.endTest(), 'btn btn-quiet'));
    } else if (S.play) {
      const pl = S.play;
      bar.append(el('span', { class: 'rl-status', text: pl.over ? (pl.winner === pl.human ? 'You won.' : pl.winner ? 'It won.' : 'A draw.') : pl.turn === pl.human ? `You are ${STONE[pl.human]}: your move` : 'It is thinking…' }),
        btn('New game', () => this.startPlay(pl.human), pl.over ? 'btn btn-primary' : 'btn'), btn(`Play ${STONE[3 - pl.human]} instead`, () => this.startPlay(3 - pl.human), 'btn btn-quiet'), btn('Stop', () => this.stopPlay(), 'btn btn-quiet'));
    } else {
      bar.append(btn(maze ? '▶ Test run' : '▶ Watch it play', () => this.startTest(), 'btn btn-primary', maze ? 'Its best move on every square, slowly, with what its model computes at each step' : 'It plays the rule-follower, one move at a time, with what its model computes at each move'));
      if (!maze) bar.append(btn(`Play ${STONE[1]} (first)`, () => this.startPlay(1)), btn(`Play ${STONE[2]} (second)`, () => this.startPlay(2)));
    }
  };
  LAB.updateBar = function () {
    if (!this.test) return;
    if (!this.cnt || !this.cnt.isConnected) { this.renderBar(); return; }
    this.cnt.textContent = this.testCount();
    const a = this.test.auto ? '❚❚' : '▶', l = this.test.auto ? 'Pause' : 'Play';
    if (this.autoBtn.textContent !== a) { this.autoBtn.textContent = a; this.autoBtn.title = l; this.autoBtn.setAttribute('aria-label', l); }
  };
  LAB.setCenterView = function (v) {
    this.centerView = v;
    const arch = v === 'arch';
    this.stage.hidden = arch; this.foot.hidden = arch; this.boardTools.hidden = arch; this.archStage.hidden = !arch;
    this.viewSeg.set(v);
    this.dirty = true; this.arch.dirty = true;
    if (arch) this.arch.refreshFocus(); else this.info.set(BOARD_INFO(this));
  };
  // the board on the left (or on top) of the canvas, the picture of its model beside it
  LAB.layout = function () {
    const W = this.stage.clientWidth, Hh = this.stage.clientHeight;
    if (!this.show.inside || W < 260) return { board: { x: 0, y: 0, w: W, h: Hh }, inside: null, W, H: Hh };
    if (W >= 640 && W / Hh >= 1.2) { const bw = Math.round(W * 0.5); return { board: { x: 0, y: 0, w: bw, h: Hh }, inside: { x: bw + 4, y: 12, w: W - bw - 16, h: Hh - 24 }, W, H: Hh }; }
    const bh = Math.round(Hh * 0.55);
    return { board: { x: 0, y: 0, w: W, h: bh }, inside: { x: 10, y: bh + 4, w: W - 20, h: Hh - bh - 12 }, W, H: Hh };
  };
  // the board's squares on the canvas
  LAB.grid = function () {
    const S = this.S, cols = S.maze ? S.maze.W : S.game.size, rows = S.maze ? S.maze.H : S.game.size, Lo = this.layout(), B = Lo.board;
    const cs = Math.max(8, Math.floor(Math.min((B.w - 28) / cols, (B.h - 40) / rows)));
    return { x: Math.round(B.x + (B.w - cs * cols) / 2), y: Math.round(B.y + (B.h - cs * rows) / 2 + 8), cs, cols, rows, W: Lo.W, H: Lo.H, inside: Lo.inside };
  };
  LAB.cellAt = function (e) {
    const b = this.canvas.getBoundingClientRect(), G = this.grid(), c = Math.floor((e.clientX - b.left - G.x) / G.cs), r = Math.floor((e.clientY - b.top - G.y) / G.cs);
    return c < 0 || r < 0 || c >= G.cols || r >= G.rows ? -1 : r * G.cols + c;
  };
  LAB.clickBoard = function (e) {
    const i = this.cellAt(e), S = this.S;
    if (i < 0) return;
    if (S.maze) {
      if (this.test) { NA.ui.toast.show('A test is on: End test to look at other squares.'); return; }
      if (S.maze.open(i)) { S.sel = i; this.bump(); if (this.tabs.current !== 'inspect' && !(NA.mobile && NA.mobile.mobile)) this.tabs.select('inspect'); }
      return;
    }
    if (this.test) return;
    if (S.play) { this.humanMove(i); return; }
    // a click on the board starts a game, with you as blue and that cell your first move
    this.startPlay(1, true); this.humanMove(i);
    NA.ui.toast.show(`You play ${STONE[1]}: your first stone is down, and it answers.`);
  };
  // the position on the board, and whose move it is (0: nobody's, a finished game)
  LAB.position = function () {
    const S = this.S, T = this.test;
    if (T) { const f = T.frames[T.k]; return f.over ? { b: f.b, line: f.line, order: f.order, toMove: 0, test: null } : { b: f.b, line: null, order: f.order, toMove: f.p, test: f }; }
    if (S.play) return { b: S.play.b, line: S.play.line, order: S.play.order, toMove: S.play.over ? 0 : S.play.turn, play: S.play };
    const sh = S.shown;
    if (!sh) return { b: new Int8Array(S.game.n), line: null, order: [], toMove: 1 };
    return { b: sh.b, line: sh.line, order: sh.order, toMove: sh.winner || sh.order.length === S.game.n || sh.kind === 'self' || sh.done ? 0 : (sh.order.length % 2) + 1 };
  };
  // its view of each move is shown, except to you while it is your move
  LAB.valuesShown = function (pos) { return this.show.values && !!pos.toMove && !(pos.play && pos.play.turn === pos.play.human); };

  LAB.draw = function () {
    const G = this.grid(), ctx = NA.fitCanvas(this.canvas, G.W, G.H);
    ctx.clearRect(0, 0, G.W, G.H);
    if (this.S.maze) this.drawMaze(ctx, G); else this.drawBoard(ctx, G);
    if (G.inside) NA.archRL.drawInside(ctx, G.inside, this);
  };
  // the value of its best move, and that move, on every square: the table's entries, or the network asked as if it
  // stood there (having been nowhere else)
  LAB.field = function () {
    const S = this.S, M = S.maze;
    if (S.model === 'table') return { v: (i) => S.Q.max(i), best: (i) => S.Q.best(i), known: (i) => S.Q.seen[i] > 0, q: (i, a) => S.Q.Q[i * 4 + a] };
    if (!this.fieldC || this.fieldC.v !== this.version || this.fieldC.S !== S) {
      const Q = new Float64Array(M.n * 4), seen = new Uint8Array(M.n);
      for (let i = 0; i < M.n; i++) { if (M.cell[i] === '#' || M.cell[i] === 'G' || M.cell[i] === 'X') continue; seen.fill(0); seen[i] = 1; Q.set(S.net.forward(RL.MazeNet.inputs(M, i, seen)), i * 4); }
      this.fieldC = { v: this.version, S, Q };
    }
    const Q = this.fieldC.Q, q = (i, a) => Q[i * 4 + a];
    return { v: (i) => Math.max(q(i, 0), q(i, 1), q(i, 2), q(i, 3)), best: (i) => { let a = 0; for (let b = 1; b < 4; b++) if (q(i, b) > q(i, a)) a = b; return a; }, known: () => true, q };
  };
  LAB.drawMaze = function (ctx, G) {
    const S = this.S, M = S.maze, F = this.field(), T = this.test, cs = G.cs, at = (i) => ({ x: G.x + (i % M.W) * cs, y: G.y + Math.floor(i / M.W) * cs });
    const plain = (i) => M.cell[i] === '.' || M.cell[i] === 'S';
    let vmax = 0.05;
    for (let i = 0; i < M.n; i++) if (plain(i)) vmax = Math.max(vmax, Math.abs(F.v(i)));
    for (let i = 0; i < M.n; i++) {
      const { x, y } = at(i), c = M.cell[i];
      ctx.fillStyle = c === '#' ? theme.c('line-2') : theme.c('panel'); ctx.fillRect(x, y, cs, cs);
      if (c === 'G' || c === 'X') { ctx.fillStyle = theme.c(c === 'G' ? 'pos' : 'neg', 0.32); ctx.fillRect(x, y, cs, cs); }
      else if (c !== '#' && this.show.values && F.known(i)) { ctx.fillStyle = theme.divColor(clamp(F.v(i) / vmax, -1, 1) * 0.9, 0.5); ctx.fillRect(x, y, cs, cs); }
      ctx.strokeStyle = theme.c('line'); ctx.lineWidth = 1; ctx.strokeRect(x + 0.5, y + 0.5, cs - 1, cs - 1);
      const u = cs * 0.3;
      if (c === 'G') S_text(ctx, '+1', x + cs / 2, y + cs / 2, u, 'pos');
      else if (c === 'X') S_text(ctx, '−1', x + cs / 2, y + cs / 2, u, 'neg');
      else if (c === 'S') S_text(ctx, 'S', x + cs * 0.16, y + cs * 0.2, u * 0.6, 'ink-2');
      // its best move, more opaque the clearer the choice
      if (plain(i) && this.show.arrows && F.known(i) && !T) {
        const a = F.best(i), qs = [0, 1, 2, 3].map((k) => F.q(i, k)).sort((p, q) => q - p), gap = clamp((qs[0] - qs[1]) / (vmax * 0.3 + 1e-9), 0.25, 1);
        const [dx, dy] = RL.DIRS[a], cx = x + cs / 2, cy = y + cs / 2, L = cs * 0.3;
        NA.archKit.arrow(ctx, cx - dx * L * 0.6, cy - dy * L * 0.6, cx + dx * L, cy + dy * L, { color: theme.c('ink', gap * (T ? 0.35 : 1)), lw: Math.max(1.2, cs * 0.045), head: Math.max(4, cs * 0.12) });
      }
      if (!T && i === S.sel && this.show.inside) { ctx.strokeStyle = theme.c('focus'); ctx.lineWidth = 2.5; ctx.strokeRect(x + 1.5, y + 1.5, cs - 3, cs - 3); }
      if (i === this.hoverCell && c !== '#' && !T) { ctx.strokeStyle = theme.c('focus', 0.6); ctx.lineWidth = 1.5; ctx.strokeRect(x + 1, y + 1, cs - 2, cs - 2); }
    }
    // the path: the test so far, the episode being watched, or the route it would take now (greedy, from the start)
    const walk = S.walk || (this.watch ? S.walkDone : null);
    let path, here = null, move = null;
    if (T) { const f = T.frames[T.k]; path = T.frames.slice(0, T.k + 1).map((q) => q.s); here = f.s; move = f.a; if (T.k === T.frames.length - 1 && !T.auto) { path.push(f.s2); } }
    else if (walk) { path = walk.path; here = walk.path[walk.path.length - 1]; }
    else path = this.greedyRoute();
    if (path && path.length > 1) {
      ctx.strokeStyle = T || walk ? theme.c('s7', 0.85) : theme.c('ink', 0.55); ctx.lineWidth = Math.max(2, cs * 0.06); ctx.setLineDash(T || walk ? [] : [cs * 0.12, cs * 0.1]); ctx.lineJoin = 'round';
      ctx.beginPath(); path.forEach((s, k) => { const p = at(s); k ? ctx.lineTo(p.x + cs / 2, p.y + cs / 2) : ctx.moveTo(p.x + cs / 2, p.y + cs / 2); }); ctx.stroke(); ctx.setLineDash([]);
    }
    if (here !== null) {
      const p = at(here), cx = p.x + cs / 2, cy = p.y + cs / 2;
      // what the network sees: the 5 × 5 squares around it
      if (T && S.model === 'net') { ctx.strokeStyle = theme.c('s7', 0.9); ctx.lineWidth = 2; ctx.setLineDash([5, 4]); ctx.strokeRect(cx - 2.5 * cs, cy - 2.5 * cs, 5 * cs, 5 * cs); ctx.setLineDash([]); }
      if (move !== null) { const [dx, dy] = RL.DIRS[move]; NA.archKit.arrow(ctx, cx + dx * cs * 0.22, cy + dy * cs * 0.22, cx + dx * cs * 0.85, cy + dy * cs * 0.85, { color: theme.c('ink'), lw: Math.max(2, cs * 0.06), head: Math.max(6, cs * 0.18) }); }
      ctx.fillStyle = theme.c('s7'); ctx.beginPath(); ctx.arc(cx, cy, cs * 0.2, 0, Math.PI * 2); ctx.fill(); ctx.strokeStyle = theme.c('panel'); ctx.lineWidth = 2; ctx.stroke();
    }
    const title = `${this.mazeName()} · ${S.model === 'net' ? 'Q-network' : 'Q-table'}${T ? ` · test: ${this.testCount().toLowerCase()}` : walk ? ' · learning' : ''}`;
    S_text(ctx, title, G.x + (cs * M.W) / 2, Math.max(10, G.y - 14), 12, 'ink-2', 600);
  };
  // where its greedy policy goes from the start (the table: stops at a loop)
  LAB.greedyRoute = function () {
    const S = this.S, M = S.maze;
    if (S.model === 'net') return this.here().path;
    if (!S.n) return null;
    let s = M.start;
    const path = [s], seen = new Set([s]);
    for (let k = 0; k < 60; k++) { s = M.move(s, S.Q.best(s)); path.push(s); if (M.cell[s] === 'G' || M.cell[s] === 'X' || seen.has(s)) break; seen.add(s); }
    return path;
  };
  LAB.drawBoard = function (ctx, G) {
    const S = this.S, g = S.game, cs = G.cs, pos = this.position(), b = pos.b, f = pos.test, at = (i) => ({ x: G.x + (i % g.size) * cs, y: G.y + Math.floor(i / g.size) * cs });
    ctx.fillStyle = theme.c('panel-2'); ctx.fillRect(G.x, G.y, cs * g.size, cs * g.size);
    // its view: for the player to move, the chance to win that each empty cell leads to
    const vals = this.valuesShown(pos) ? (f ? { moves: f.moves, vals: f.vals } : RL.moveValues(g, S.brain, b, pos.toMove)) : null;
    const vOf = new Map(vals ? vals.moves.map((m, k) => [m, vals.vals[k]]) : []);
    let best = -1, bestV = -1;
    for (const [m, v] of vOf) if (v > bestV) { bestV = v; best = m; }
    for (let i = 0; i < g.n; i++) {
      const { x, y } = at(i);
      ctx.strokeStyle = theme.c('line-2'); ctx.lineWidth = 1; ctx.strokeRect(x + 0.5, y + 0.5, cs - 1, cs - 1);
      if (b[i]) {
        const r = cs * 0.36, cx = x + cs / 2, cy = y + cs / 2;
        ctx.beginPath(); ctx.arc(cx, cy, r, 0, Math.PI * 2);
        if (b[i] === 1) { ctx.fillStyle = theme.c('s1'); ctx.fill(); } else { ctx.fillStyle = theme.c('s2', 0.18); ctx.fill(); ctx.strokeStyle = theme.c('s2'); ctx.lineWidth = Math.max(2.5, cs * 0.08); ctx.stroke(); }
        const k = pos.order.indexOf(i);
        if (k >= 0 && cs >= 34) S_text(ctx, String(k + 1), cx, cy, cs * 0.2, b[i] === 1 ? 'panel' : 's2');
        if (k === pos.order.length - 1) { ctx.strokeStyle = theme.c('focus'); ctx.lineWidth = 2; ctx.strokeRect(x + 3, y + 3, cs - 6, cs - 6); }
      } else if (vOf.has(i)) {
        const v = vOf.get(i);
        ctx.fillStyle = theme.seqColor(clamp(v, 0, 1), 0.55); ctx.beginPath(); ctx.arc(x + cs / 2, y + cs / 2, cs * 0.3, 0, Math.PI * 2); ctx.fill();
        if (cs >= 30) S_text(ctx, Math.round(v * 100) + '%', x + cs / 2, y + cs / 2, cs * 0.19, i === best ? 'ink' : 'ink-2', i === best ? 700 : 500);
        if (i === best && !f) { ctx.strokeStyle = theme.c('ink', 0.7); ctx.lineWidth = 1.5; ctx.beginPath(); ctx.arc(x + cs / 2, y + cs / 2, cs * 0.33, 0, Math.PI * 2); ctx.stroke(); }
      }
      // a test: the move about to be played, in its player's colour
      if (f && i === f.m) { ctx.strokeStyle = theme.c(f.p === 1 ? 's1' : 's2'); ctx.lineWidth = Math.max(2.5, cs * 0.07); ctx.setLineDash([5, 4]); ctx.beginPath(); ctx.arc(x + cs / 2, y + cs / 2, cs * 0.4, 0, Math.PI * 2); ctx.stroke(); ctx.setLineDash([]); }
      if (i === this.hoverCell && !b[i] && (pos.play ? !pos.play.over && pos.play.turn === pos.play.human : !this.test)) { ctx.strokeStyle = theme.c(pos.play ? (pos.play.human === 1 ? 's1' : 's2') : 'focus', 0.8); ctx.lineWidth = 2; ctx.beginPath(); ctx.arc(x + cs / 2, y + cs / 2, cs * 0.36, 0, Math.PI * 2); ctx.stroke(); }
    }
    if (pos.line) {
      const a = at(pos.line[0]), z = at(pos.line[pos.line.length - 1]);
      ctx.strokeStyle = theme.c('ink', 0.85); ctx.lineWidth = Math.max(3, cs * 0.09); ctx.lineCap = 'round';
      ctx.beginPath(); ctx.moveTo(a.x + cs / 2, a.y + cs / 2); ctx.lineTo(z.x + cs / 2, z.y + cs / 2); ctx.stroke(); ctx.lineCap = 'butt';
    }
    // what is on the board
    const sh = S.shown, T = this.test;
    const label = T ? (f ? `Test, move ${f.order.length + 1}: ${f.who === 'agent' ? `it plays ${STONE[f.p]}` : `the rule-follower plays ${STONE[f.p]}`}` : `Test over: ${this.testResult()}`)
      : S.play ? (S.play.over ? (S.play.winner === S.play.human ? 'You won' : S.play.winner ? 'It won' : 'A draw') : S.play.turn === S.play.human ? 'Your move' : 'Its move')
        : !sh ? `An empty board: ${NA.tapWord(true)} a cell to play, or press Train` : sh.kind === 'watch' && !sh.done ? `Self-play, move ${sh.order.length}: ${STONE[(sh.order.length % 2) + 1]} to move` : `Its last self-play game (game ${fmtInt(S.n)}): ${sh.winner ? `${STONE[sh.winner]} won` : 'a draw'} in ${sh.order.length} moves`;
    S_text(ctx, label, G.x + (cs * g.size) / 2, Math.max(10, G.y - 14), 12, 'ink-2', 600);
  };
  const S_text = (ctx, s, x, y, size, tok, w = 650) => NA.archKit.text(ctx, s, x, y, { size, w, align: 'center', color: theme.c(tok) });

  // ------------------------------------------------------------------ right: overview, training, its view
  LAB.buildTab = function (id, body) {
    this.tabId = id; const P = (this.parts = {}), S = this.S, maze = !!S.maze, net = S.model === 'net';
    if (id === 'overview') {
      body.append(UI.modelCard(this.preset, (t) => this.runTry(t)));
      P.findings = el('div', { class: 'diag-list' });
      body.append(UI.section('How it is doing', P.findings));
    } else if (id === 'train') {
      const mk = (title, opts, legend, note) => { const c = el('canvas'), chart = new NA.charts.LineChart(c, opts); body.append(UI.section(title, el('div', { class: 'chart-box' }, c), legend, note ? UI.note(note) : null)); return chart; };
      const lg = (items) => el('div', { class: 'mini-legend' }, ...items.map(([n, tok]) => el('span', { class: 'lg-item' }, el('i', { class: 'key-line', style: { background: `var(--${tok})` } }), n)));
      P.tiles = el('div', { class: 'tiles' }); body.append(P.tiles);
      if (maze && net) {
        P.a = mk('Mazes it gets through', { height: 150, format: (v) => fmtPct(v, 0), yFormat: (v) => fmtPct(v, 0) }, lg([['100 new mazes', 's1'], ['the 5 named mazes', 's7']]), 'Its best moves from S, no exploring. It never trains on any of these mazes: the named ones and the 100 new ones only test it. Networks trained this way wobble: a maze it got through can be lost again for a while.');
        P.b = mk('Steps per episode while learning', { height: 120, format: (v) => fmt(v, 1) }, null, 'On the random mazes it learns from, exploring. Its first mazes are easy, so this rises before it falls.');
      } else if (maze) {
        P.a = mk('Steps per episode', { height: 140, format: (v) => fmt(v, 1) }, lg([['while training (exploring)', 'ink-2'], ['its best route', 'pos']]), 'Average over every 20 episodes. Exploring adds detours; the best route is the greedy one, with no random moves.');
        P.b = mk('Reward per episode', { height: 130, format: (v) => fmt(v, 2) }, null, '+1 at the goal minus 0.02 a step, or −1 in a pit.');
      } else {
        P.a = mk('Against a random player', { height: 140, format: (v) => fmtPct(v, 0), yFormat: (v) => fmtPct(v, 0) }, lg([['wins', 'pos'], ['losses', 'neg']]), 'Half the games moving first. A random player is easy: this shows it learned the basics.');
        P.b = mk('Against the rule-follower', { height: 140, format: (v) => fmtPct(v, 0), yFormat: (v) => fmtPct(v, 0) }, lg([['wins', 'pos'], ['losses', 'neg']]), 'The rule-follower wins when it can and blocks your winning cell when it must, otherwise plays at random. Beating it takes some planning.');
      }
    } else if (id === 'inspect') {
      P.box = el('div');
      body.append(P.box);
    }
    this.updateTab();
  };
  LAB.updateTab = function () {
    const id = this.tabId, P = this.parts, S = this.S, h = S.history, last = h[h.length - 1];
    if (!P) return;
    if (id === 'overview') clear(P.findings).append(...FINDINGS(this).map(([lvl, t, x]) => el('div', { class: 'diag diag-' + lvl }, el('div', { class: 'diag-head' }, UI.pill(lvl, { good: 'Good', info: 'Note', warn: 'Watch', critical: 'Problem' }[lvl]), el('h4', { class: 'diag-title', text: t })), el('p', { class: 'diag-body', text: x }))));
    else if (id === 'train') {
      if (S.maze && S.model === 'net') {
        clear(P.tiles).append(UI.tile('Episodes', fmtInt(S.n), S.pretrained ? 'mostly trained offline' : null), UI.tile('Named mazes', last && last.named != null ? `${Math.round(last.named * 5)} of 5` : '—'), UI.tile('New mazes', last && last.fresh != null ? fmtPct(last.fresh, 0) : '—', 'of 100 it never trained on'));
        P.a.frame(); P.a.set({ series: [{ name: 'new mazes', color: theme.c('s1'), points: h.filter((q) => q.fresh != null).map((q) => [q.n, q.fresh]) }, { name: 'named mazes', color: theme.c('s7'), points: h.filter((q) => q.named != null).map((q) => [q.n, q.named]) }] });
        P.b.frame(); P.b.set({ series: [{ name: 'steps', color: theme.c('ink-2'), points: h.filter((q) => q.steps != null).map((q) => [q.n, q.steps]) }] });
      } else if (S.maze) {
        clear(P.tiles).append(UI.tile('Episodes', fmtInt(S.n)), UI.tile('Best route', last && last.greedy ? `${fmt(last.route, 1)} steps` : '—', last ? `reaches the goal ${fmtPct(last.greedy, 0)}` : null), UI.tile('Exploration', fmt(S.o.eps, 2)));
        P.a.frame(); P.a.set({ series: [{ name: 'training', color: theme.c('ink-2'), points: h.map((q) => [q.n, q.steps]) }, { name: 'best route', color: theme.c('pos'), points: h.filter((q) => q.greedy > 0).map((q) => [q.n, q.route]) }] });
        P.b.frame(); P.b.set({ series: [{ name: 'reward', color: theme.c('s1'), points: h.map((q) => [q.n, q.ret]) }] });
      } else {
        clear(P.tiles).append(UI.tile('Games', fmtInt(S.n), S.pretrained ? 'includes 100,000 trained offline' : null), UI.tile(S.model === 'table' ? 'Positions in its table' : 'Parameters', S.model === 'table' ? fmtInt(S.brain.size) : fmtInt(S.brain.paramCount)), UI.tile('Beats random', last ? fmtPct(last.rnd.win, 0) : '—'));
        const ser = (key) => [{ name: 'wins', color: theme.c('pos'), points: h.map((q) => [q.n, q[key].win]) }, { name: 'losses', color: theme.c('neg'), points: h.map((q) => [q.n, q[key].loss]) }];
        P.a.frame(); P.a.set({ series: ser('rnd') });
        P.b.frame(); P.b.set({ series: ser('rules') });
      }
    } else if (id === 'inspect') clear(P.box).append(...(S.maze ? this.inspectMaze() : this.inspectGame()));
  };
  const kv = (k, v, note) => el('div', { class: 'kv' }, el('span', { class: 'kv-k', text: k }), el('span', { class: 'kv-v', text: v }), note ? el('span', { class: 'kv-note', text: note }) : null);
  LAB.inspectMaze = function () {
    const S = this.S, M = S.maze, P = NA.archRL.probe(this), s = P.s, x = s % M.W, y = Math.floor(s / M.W), u = S.update, net = S.model === 'net';
    const rows = RL.DIR_NAMES.map((d, a) => kv(`Q(${x}, ${y}, ${d})`, fmt(P.q[a], 4), a === P.a ? 'its best move here' : M.move(s, a) === s ? 'into a wall: it stays put' : null));
    const out = [UI.section(`Square (${x}, ${y})${P.why === 'test' ? ', in the test' : ''}`, UI.note(net ? `What its network outputs on this square${P.why === 'test' ? ', having been where the test went' : ', as if it had been nowhere else'}: the reward it expects for each move. ${NA.tapWord()} any square of the maze to see it there.` : `${NA.tapWord()} any square of the maze to see its four values. Each is how much reward it expects if it makes that move here and then plays its best.`), el('div', { class: 'kv-grid' }, ...rows))];
    if (u) {
      const ux = u.s % M.W, uy = Math.floor(u.s / M.W), how = `at (${ux}, ${uy}) it ${u.slipped ? `meant to go ${RL.DIR_NAMES[u.intended]} but slipped ${RL.DIR_NAMES[u.a]}` : `went ${RL.DIR_NAMES[u.a]}${u.explore ? ' (exploring)' : ''}`}: reward ${fmt(u.r, 2)}`;
      if (u.net) out.push(UI.section('The last update', el('div', { class: 'fx' },
        el('div', { class: 'fx-line', text: how }),
        el('div', { class: 'fx-line', text: `target = r + γ·max Q_target(next) = ${fmt(u.r, 2)}${u.done ? ' (the run ended)' : ` + ${fmt(S.o.gamma, 2)} × ${fmt(u.next, 3)}`} = ${fmt(u.target, 3)}` }),
        el('div', { class: 'fx-line fx-strong', text: `Q(${RL.DIR_NAMES[u.intended]}) here: ${fmt(u.before, 3)} → ${fmt(u.after, 3)} after 4 gradient steps on moves drawn from its memory` })), UI.note('Shown while you watch it learn. The move goes into its replay memory; the 4 moves it learns from are drawn at random from there, so this one may not be among them yet.')));
      else {
        const mx = S.Q.max(u.s2);
        out.push(UI.section('The last update', el('div', { class: 'fx' },
          el('div', { class: 'fx-line', text: how }),
          el('div', { class: 'fx-line', text: `target = r + γ·max Q(next) = ${fmt(u.r, 2)}${u.done ? ' (the episode ended)' : ` + ${fmt(S.o.gamma, 2)} × ${fmt(mx, 3)}`} = ${fmt(u.before + u.err, 3)}` }),
          el('div', { class: 'fx-line fx-strong', text: `Q ← Q + α·(target − Q) = ${fmt(u.before, 3)} + ${fmt(S.o.alpha, 2)} × ${fmt(u.err, 3)} = ${fmt(u.after, 3)}` })), UI.note('Shown while you watch it learn: the update after each move. A slip does not change which entry learns: the one for the move it chose.')));
      }
    }
    return out;
  };
  LAB.inspectGame = function () {
    const S = this.S, pos = this.position(), g = S.game;
    if (!pos.toMove) return [UI.section('Its view', UI.note('The game on the board is over. Train, watch it play, or play against it to see how it rates each move.'))];
    const { moves, vals } = pos.test ? pos.test : RL.moveValues(g, S.brain, pos.b, pos.toMove), order = moves.map((m, k) => [m, vals[k]]).sort((a, b) => b[1] - a[1]);
    const name = (m) => `row ${Math.floor(m / g.size) + 1}, column ${(m % g.size) + 1}`;
    return [UI.section(`${STONE[pos.toMove]} to move: how it rates each move`, UI.note(`For each empty cell, the chance that ${STONE[pos.toMove]} goes on to win after playing there, by its ${S.model === 'table' ? 'table (0.5 for a position it has never met)' : 'network'}. It plays the top one.`),
      el('div', { class: 'kv-grid' }, ...order.slice(0, 10).map(([m, v], k) => kv(name(m), fmtPct(v, 0), k === 0 ? 'its choice' : v === 1 ? 'wins at once' : null))))];
  };

  // ------------------------------------------------------------------ explanations
  const FINDINGS = (L) => {
    const S = L.S, h = S.history, last = h[h.length - 1], out = [];
    if (!S.n) return [['info', 'Not trained yet', 'Press Train. It starts knowing nothing: every value is 0 (the maze) or a coin flip (the games).']];
    if (S.maze && S.model === 'net') {
      const r = L.here(), d = RL.shortest(S.maze);
      out.push(r.end === 'goal' ? ['good', `Gets through ${L.mazeName()} in ${r.steps} steps`, r.steps === d ? 'The shortest way through, in a maze it never trained on.' : `The shortest way is ${d} steps. It sees only 2 squares around it, so it does not plan the whole route: ${S.maze.cell.includes('X') ? 'it keeps its distance from pits and takes a safer way' : 'where it cannot see the way, it tries a side and remembers where it has been'}.`]
        : ['warn', `Does not get through ${L.mazeName()}`, `Its route ${r.end === 'pit' ? 'ends in a pit' : 'wanders until the 60 steps run out'}. It sees only 2 squares around it and which way the goal lies: a way that first leads away from the goal, or a dead end it cannot see the end of, can fool it. More training helps a little; seeing further, or a network with memory, helps more.`]);
      if (last && last.fresh != null) out.push([last.fresh >= 0.8 ? 'good' : 'info', `Gets through ${fmtPct(last.fresh, 0)} of new mazes`, 'Of 100 random mazes it never trained on, from S with its best moves. A table could not do this at all: it has no row for a square it never stood on.']);
      out.push(['info', `${fmtInt(S.net.paramCount)} parameters`, 'One Q-table for one maze holds about 120 numbers. The network holds more, but they are shared by every maze: the price of not knowing in advance which maze it will meet.']);
    } else if (S.maze) {
      if (!last) out.push(['info', 'Exploring', 'It has not found the goal often enough to have a route yet.']);
      else if (last.greedy === 1) out.push(['good', `Found a route: ${fmt(last.route, 0)} steps`, S.maze.slip ? 'On a slippery floor every step near a pit is a risk, so the best route may be longer than the shortest one.' : `The shortest way through is ${RL.shortest(S.maze)} steps: it learned its route from rewards alone.`]);
      else out.push(['warn', `Its route reaches the goal ${fmtPct(last.greedy, 0)} of the time`, S.maze.slip ? 'The floor is slippery: some slips end in a pit whatever it does. A smaller learning rate averages the luck out.' : 'Still learning: keep training.']);
      if (S.o.eps === 0) out.push(['warn', 'No exploring', 'With ε = 0 it only ever tries the move it already rates best, so it can settle on the first route it found, or none.']);
      out.push(['info', 'This table fits one maze', 'Its rows are this maze’s squares. Pick another maze and a new table starts from nothing; switch the model to Q-network for one that works in any maze.']);
    } else {
      out.push([last && last.rnd.win > 0.9 ? 'good' : 'info', `Beats a random player ${last ? fmtPct(last.rnd.win, 0) : '—'} of the time`, 'A random player is easy; this shows it learned to make lines and to finish them.']);
      if (last) out.push([last.rules.win > 0.7 ? 'good' : 'warn', `Against the rule-follower: wins ${fmtPct(last.rules.win, 0)}, loses ${fmtPct(last.rules.loss, 0)}`, last.rules.loss > 0.2 ? 'It still walks into some threats the rule-follower blocks or completes. More games, or less exploring, help.' : 'It sets up positions with two ways to win, so blocking one is not enough.']);
      out.push(['info', S.model === 'table' ? `${fmtInt(S.brain.size)} positions in its table` : `${fmtInt(S.brain.paramCount)} parameters`, S.model === 'table' ? 'Each one met in self-play, with rotations and mirror images counted once. A table only knows positions it has seen.' : S.id === 'three' ? 'The network judges any position from the pattern of stones, so each lesson also teaches it about similar positions: it overtakes the table within a few hundred games.' : 'A table for this board would need billions of entries; the network judges any position from the pattern of stones.']);
    }
    return out;
  };
  const BOARD_INFO = (L) => {
    const S = L.S;
    if (S.maze && S.model === 'net') return { kicker: 'Maze', title: 'One network for every maze', about: [
      `The colours and arrows show what its network says on every square of ${L.mazeName()}, as if it stood there: blue where it expects reward, orange where it expects to lose. It has never trained on this maze, nor on any of the named ones: it learned on ${fmtInt(S.n)} random mazes, easy ones first.`,
      'A table cannot do that: it has a row for each square of one maze. The network has none. It sees the 24 squares around it (walls, pits, the goal, where it has been) and which way the goal lies, so the same surroundings get the same answer in any maze. Pick New random maze to give it one nobody has seen.',
      'Test run walks its route slowly, with the 5 × 5 squares it sees outlined, and draws its network beside the maze at every step: the inputs, the 64 hidden neurons that fire, and the four Q-values. This is DQN (2013–15), the method that learned Atari games from the screen, here on mazes.'], code: NA.archRL.py.mazeNet(L) };
    if (S.maze) return { kicker: 'Maze', title: 'Q-learning: values from rewards', about: [
      'Each square is coloured by the value of its best move (blue: it expects reward from here, orange: it expects to lose), and the arrow is that move. At first every value is 0. The first time it stumbles into the goal, the square next to it learns that one move pays; on later visits that value is copied, a little discounted, to the squares before it. The values spread back from the goal like a scent, and the arrows line up into a route.',
      `It explores: ${fmtPct(S.o.eps, 0)} of its moves are random (ε), so it keeps finding new ways. The dashed line is the route it would take now, choosing its best move on every square. Tick Watch to see one move at a time; Test run walks the route slowly with its table entries beside the maze.`,
      'This is Q-learning (Watkins, 1989), the start of deep reinforcement learning: DQN (2013) replaced the table with a neural network. Switch the model to Q-network to see one that works in any maze.'], code: NA.archRL.py.maze(L) };
    const three = S.id === 'three', net = S.model === 'net';
    return { kicker: NAMES[S.id], title: three ? (net ? 'A network learns a game by playing itself' : 'Learning a game by playing itself') : 'A network that judges positions', about: [
      three && !net ? 'It learns Three-line by playing both sides against itself, starting from nothing: every position it has not met is worth 0.5, a coin flip. It keeps a value for each position it meets, the chance that the player who just moved will win, and moves to the position it values most. When a game ends, the last positions move toward the result (1 for the winner, 0 for the loser), and on later games those values spread back to the moves that led there. This is temporal-difference learning, as in the classic self-play program for backgammon, TD-Gammon (1992).'
        : three ? 'The same self-play learning as with the table, but a small network judges the positions: the board goes in (32 inputs: where each player’s stones are), 32 tanh neurons look for patterns, and one sigmoid output gives the chance to win. Each lesson changes weights that every position shares, so it learns about positions it has never met and needs far fewer games than the table.'
          : 'Four-line has billions of positions, too many to list, so a neural network judges them: the board goes in (50 inputs: where each player’s stones are), 64 tanh neurons look for patterns, and one sigmoid output gives the chance to win. It learns exactly as Three-line does, by self-play and temporal differences, but each lesson changes weights shared by every position, so it can judge positions it has never seen. AlphaZero (2017) added search on top of this idea to master chess and Go.',
      'The numbers on the empty cells are its view of each move: the chance the player to move would go on to win after playing there. A cell that wins at once shows 100%. When it plays, it picks the highest. Beside the board, what its model computes for the move under the pointer (or its last move).',
      `Play against it with the buttons above the board, or ${NA.tapWord(true)} a cell to start. Watch it play runs a game against the rule-follower slowly, with its model drawn at every move.`], code: three ? (net ? NA.archRL.py.threeNet(L) : NA.archRL.py.three(L)) : NA.archRL.py.four(L) };
  };

  // ------------------------------------------------------------------ experiments, and the gallery's picture
  LAB.runTry = function (t) {
    const S = this.S, retrain = (msg) => { this.test = null; this.fresh(); this.watch = false; this.buildAgain(); this.renderBar(); this.fastForward(t.n || (S.maze ? 1000 : 3000)); if (msg) NA.ui.toast.show(msg); };
    if (t.action === 'watch') {
      // the maze table starts over, so you see it from nothing; the rest keep what they know
      if (S.maze && S.model === 'table') this.fresh();
      if (this.S.play) this.S.play = null;
      this.test = null; this.watch = true; this.buildAgain(); this.renderBar(); this.play();
      NA.ui.toast.show(S.maze ? (S.model === 'table' ? 'Watching it learn from nothing, one move at a time.' : 'Watching the network learn on this maze, one move at a time.') : 'Watching it play itself, one move at a time.');
    }
    else if (t.action === 'slippery') { S.maze.slip = 0.15; S.o.alpha = 0.1; retrain('A slippery floor (15% of moves go sideways) and a smaller learning rate: retraining.'); }
    else if (t.action === 'noExplore') { S.o.eps = 0; retrain('No exploring (ε = 0): retraining.'); }
    else if (t.action === 'shortSighted') { S.o.gamma = 0.5; retrain('Discount γ = 0.5: rewards a few steps away count for little. Retraining.'); }
    else if (t.action === 'train') { this.watch = false; this.test = null; this.fastForward(t.n || 3000); }
    else if (t.action === 'scratch') { retrain(S.maze ? 'A new network that knows nothing: watch the charts as it learns on random mazes.' : 'A new network that knows nothing: watch the charts as it learns.'); }
    else if (t.action === 'trained') { this.pause(); this.test = null; this.loadTrained(); this.buildAgain(); this.renderBar(); NA.ui.toast.show(S.maze ? 'The network trained on random mazes is back.' : 'The network trained for 100,000 games is back.'); }
    else if (t.action === 'playFirst') this.startPlay(1);
    else if (t.action === 'playSecond') this.startPlay(2);
    else if (t.action === 'test') this.startTest();
    else if (t.action === 'table' || t.action === 'net') this.setModel(t.action);
    else if (t.action === 'newMaze') this.newMaze();
    else if (t.action === 'maze') this.setMaze(t.maze);
    else if (t.action === 'default') {
      this.test = null;
      Object.assign(S.o, S.maze ? (S.model === 'net' ? { eps: 0.05, lr: 0.01, gamma: 0.97 } : { eps: 0.1, alpha: 0.5, gamma: 0.95 }) : S.model === 'table' ? { eps: 0.1, alpha: 0.2 } : { eps: 0.1, lr: 0.05 });
      if (S.maze) S.maze.slip = 0;
      this.fresh();
      if (this.S.id === 'four' || (this.S.maze && this.S.model === 'net')) this.loadTrained();
      this.buildAgain(); this.renderBar();
    }
    if (this.tabs.current !== this.tabId) this.tabs.select(this.tabId); else this.updateTab();
    // an experiment that trains reports its result when it is done
    if (t.predict) return new Promise((res) => { const wait = () => (this.training ? setTimeout(wait, 200) : res(this.measured())); wait(); });
    return null;
  };
  LAB.measured = function () {
    const S = this.S, h = S.history, last = h[h.length - 1];
    if (!last) return '';
    if (S.maze && S.model === 'net') return `Measured: it gets through ${last.named != null ? `${Math.round(last.named * 5)} of the 5 named mazes and ` : ''}${fmtPct(last.fresh, 0)} of 100 new ones.`;
    if (S.maze) return `Measured: its route reaches the goal ${fmtPct(last.greedy, 0)} of the time${last.greedy ? `, in ${fmt(last.route, 1)} steps` : ''}.`;
    const first = h.filter((q) => q.first !== undefined), ninety = h.find((q) => q.rnd.win >= 0.9);
    return `Measured: ${first.length ? `blue won ${fmtPct(first.reduce((s, q) => s + q.first, 0) / first.length, 0)} of its self-play games; ` : ''}it beats a random player ${fmtPct(last.rnd.win, 0)} of the time${ninety ? ` (9 in 10 by game ${fmtInt(ninety.n)})` : ''}.`;
  };
  // the left panel and the hyper sheet show the settings: rebuilt after an experiment changes them
  LAB.buildAgain = function () { clear(this.ui.left); this.buildLeft(this.ui.left); clear(this.ui.hyper); this.buildHyper(this.ui.hyper); this.bump(); };
  LAB.glyph = function (ctx, W, Hh, presetId) {
    const ex = (presetId || 'rl-maze').slice(3);
    const maze = (rows, x0, y0, cs) => new RL.Maze(rows).cell.forEach((c, i) => { ctx.fillStyle = c === '#' ? theme.c('line-2') : c === 'G' ? theme.c('pos', 0.7) : c === 'X' ? theme.c('neg', 0.6) : theme.c('line', 0.5); ctx.fillRect(x0 + (i % 8) * cs + 0.5, y0 + Math.floor(i / 8) * cs + 0.5, cs - 1, cs - 1); });
    if (ex === 'maze') { const cs = Math.min((W - 8) / 8, (Hh - 8) / 6); maze(RL.MAZE, (W - cs * 8) / 2, (Hh - cs * 6) / 2, cs); return; }
    if (ex === 'mazes') {
      // two of the mazes, one in front of the other like cards
      const cs = Math.min((W - 10) / 8, (Hh - 6) / 6) * 0.72;
      [[RL.MAZE_SET[3], 3, 2], [RL.MAZE_SET[4], W - cs * 8 - 3, Hh - cs * 6 - 2]].forEach(([m, x0, y0]) => { ctx.fillStyle = theme.c('panel'); ctx.fillRect(x0 - 2, y0 - 2, cs * 8 + 4, cs * 6 + 4); maze(m.rows, x0, y0, cs); });
      return;
    }
    const g = RL.games[ex], cs = Math.min((W - 8) / g.size, (Hh - 8) / g.size), x0 = (W - cs * g.size) / 2, y0 = (Hh - cs * g.size) / 2;
    const stones = ex === 'three' ? { 5: 1, 6: 2, 10: 1, 9: 2, 15: 1 } : { 6: 1, 7: 2, 12: 1, 13: 2, 18: 1, 11: 2, 24: 1 };
    for (let i = 0; i < g.n; i++) {
      const x = x0 + (i % g.size) * cs, y = y0 + Math.floor(i / g.size) * cs;
      ctx.strokeStyle = theme.c('line-2'); ctx.lineWidth = 1; ctx.strokeRect(x + 0.5, y + 0.5, cs - 1, cs - 1);
      if (stones[i]) { ctx.beginPath(); ctx.arc(x + cs / 2, y + cs / 2, cs * 0.32, 0, Math.PI * 2); if (stones[i] === 1) { ctx.fillStyle = theme.c('s1'); ctx.fill(); } else { ctx.strokeStyle = theme.c('s2'); ctx.lineWidth = 1.6; ctx.stroke(); } }
    }
  };

  H.register(LAB);
})(window.NA);
