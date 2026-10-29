/* lab-rl.js — the reinforcement learning lab: an agent learns from rewards alone, with no examples of good play.
   Three examples share it: a maze, learned with a table of Q-values; Three-line, a 4 × 4 board game it learns by
   playing against itself, keeping a value for every position it meets; and Four-line, a 5 × 5 board game with far
   too many positions for a table, learned the same way with a small neural network. Train it, watch it learn one
   move at a time, then play against it. */
(function (NA) {
  'use strict';
  const { el, clear, clamp, fmt, fmtPct, fmtInt, theme } = NA;
  const App = NA.app, H = NA.labHost, UI = NA.labUI, RL = NA.rl;
  const EVERY = { three: 250, four: 250 }; // self-play games between two checks of how well it plays
  const WATCH_MS = { maze: 110, three: 420, four: 420 };
  const NAMES = { maze: 'Maze', three: 'Three-line', four: 'Four-line' };
  const STONE = ['', 'blue', 'orange'];

  const LAB = H.transport({ id: 'rl', screens: { network: 'Board', data: 'Agent', insp: 'Inspect' }, budget: 12, version: 0, st: {}, centerView: 'board', show: { values: true, arrows: true } });

  // ------------------------------------------------------------------ the agents, one per example
  // Each example keeps its own agent, so switching between them keeps what each has learned.
  LAB.state = function (id) {
    if (this.st[id]) return this.st[id];
    const S = { id, n: 0, rng: RL.rand({ maze: 11, three: 22, four: 33 }[id]), ev: RL.rand(99), history: [] };
    if (id === 'maze') {
      S.maze = new RL.Maze(); S.Q = new RL.QTable(S.maze.n, 4); S.o = { eps: 0.1, alpha: 0.5, gamma: 0.95 }; S.sel = S.maze.start;
    } else {
      S.game = RL.games[id];
      S.o = id === 'three' ? { eps: 0.1, alpha: 0.2 } : { eps: 0.1, lr: 0.05 };
      S.brain = id === 'three' ? new RL.ValueTable(S.game) : new RL.ValueNet(S.game.n, 64, 7);
      S.window = [0, 0, 0]; // self-play results since the last check: draws, first player won, second player won
      S.shown = null; // the game on the board: the last self-play game, one being watched, or yours
    }
    return (this.st[id] = S);
  };
  // a new agent that knows nothing, keeping the settings
  LAB.fresh = function (id = this.ex) {
    const old = this.st[id];
    delete this.st[id];
    const S = this.state(id);
    if (old) { Object.assign(S.o, old.o); if (old.maze) S.maze.slip = old.maze.slip; }
    if (id === this.ex) this.use(id);
    return S;
  };
  // the network trained offline (js/rl-weights.js, the same code for 100,000 games of self-play)
  LAB.loadTrained = function () {
    const W = NA.rlWeights;
    if (!W) return false;
    const S = this.state('four');
    S.brain = RL.ValueNet.from(W.four); S.n = W.games; S.pretrained = true; S.window = [0, 0, 0];
    S.history = W.curve.map(([n, rw, rl, tw, tl]) => ({ n, rnd: { win: rw, loss: rl, draw: Math.max(0, 1 - rw - rl) }, rules: { win: tw, loss: tl, draw: Math.max(0, 1 - tw - tl) } }));
    if (this.ex === 'four') this.use('four');
    return true;
  };
  // Four-line opens on the trained network the first time only: starting from nothing must stay at nothing
  LAB.use = function (id) {
    const first = !this.st[id];
    this.ex = id; this.S = this.state(id); this.step = this.S.n;
    if (id === 'four' && first) this.loadTrained();
    this.bump();
  };

  // ------------------------------------------------------------------ training
  LAB.trainChunk = function (budget, maxSteps) {
    if (this.watch) return 0; // watching: one move at a time, in frame()
    const S = this.S, t0 = performance.now();
    let k = 0;
    do {
      if (S.id === 'maze') this.episodeDone(RL.mazeEpisode(S.maze, S.Q, S.o, S.rng));
      else this.gameDone(RL.selfPlay(S.game, S.brain, S.o, S.rng));
      k++;
    } while (performance.now() - t0 < budget && k < maxSteps);
    this.step = S.n; this.bump();
    return k;
  };
  // the maze: every 20 episodes, their average length and return, and how the greedy route does (no exploring)
  LAB.episodeDone = function (e) {
    const S = this.S, a = S.acc || (S.acc = { n: 0, steps: 0, ret: 0, goal: 0 });
    S.n++; S.lastEp = e;
    a.n++; a.steps += e.steps; a.ret += e.total; a.goal += e.end === 'goal' ? 1 : 0;
    if (a.n >= 20) {
      const g = RL.mazeEval(S.maze, S.Q, S.ev, 20);
      S.history.push({ n: S.n, steps: a.steps / a.n, ret: a.ret / a.n, goal: a.goal / a.n, greedy: g.success, route: g.steps });
      S.acc = null; thin(S.history);
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
  // watching: it learns one move at a time, slowly enough to follow
  LAB.watchTick = function () {
    const S = this.S;
    if (S.id === 'maze') {
      const w = S.walk || (S.walk = { s: S.maze.start, path: [S.maze.start], steps: 0, total: 0 });
      const intended = S.Q.act(w.s, S.o.eps, S.rng), q0 = S.Q.Q[w.s * 4 + intended], t = S.maze.step(w.s, intended, S.rng);
      const err = S.Q.update(w.s, t.a, t.r, t.s, t.done, S.o.alpha, S.o.gamma);
      S.update = { s: w.s, a: t.a, intended, slipped: t.slipped, r: t.r, s2: t.s, done: t.done, err, before: t.a === intended ? q0 : S.Q.Q[w.s * 4 + t.a] - S.o.alpha * err, after: S.Q.Q[w.s * 4 + t.a] };
      w.s = t.s; w.path.push(t.s); w.steps++; w.total += t.r;
      if (t.done || w.steps >= S.maze.maxSteps) { this.episodeDone({ total: w.total, steps: w.steps, end: t.done ? (t.r > 0 ? 'goal' : 'pit') : 'timeout', path: w.path }); S.walkDone = w; S.walk = null; this.pauseUntil = performance.now() + 500; }
    } else {
      const M = S.live && !S.live.done ? S.live : (S.live = new RL.Match(S.game, S.brain, S.o, S.rng));
      S.lastMove = M.step();
      S.shown = { b: M.b, line: M.line, order: M.order, winner: M.winner, kind: 'watch', done: M.done };
      if (M.done) { this.gameDone(M); this.pauseUntil = performance.now() + 1100; }
    }
    this.step = S.n; this.bump();
  };
  LAB.onPause = function () { this.bump(); };
  LAB.stepOnce = function () { if (this.training) this.pause(); if (this.S.play) this.S.play = null; this.watch ? this.watchTick() : this.trainChunk(0, 1); };
  LAB.reset = function () { if (this.training) this.pause(); this.fresh(); NA.ui.toast.show(`A new ${NAMES[this.ex]} agent that knows nothing yet. Press Train.`); };
  LAB.bump = function () { this.version++; this.dirty = true; this.inspDirty = true; this.leftDirty = true; this.archStale = true; this.infoStale = true; };

  // ------------------------------------------------------------------ playing against it
  LAB.startPlay = function (human) {
    const S = this.S;
    if (this.training) this.pause();
    this.watch = false; if (this.watchChip) this.watchChip.setAttribute('aria-pressed', 'false');
    S.play = { b: new Int8Array(S.game.n), human, turn: 1, over: false, winner: 0, line: null, order: [] };
    this.agentAt = human === 1 ? Infinity : performance.now() + 700;
    this.bump(); this.renderPlay();
    NA.ui.toast.show(human === 1 ? `You are ${STONE[1]} and move first: ${NA.tapWord(true)} a cell.` : `You are ${STONE[2]}: it moves first.`);
  };
  LAB.stopPlay = function () { this.S.play = null; this.bump(); this.renderPlay(); };
  LAB.place = function (i) {
    const S = this.S, P = S.play;
    P.b[i] = P.turn; P.order.push(i);
    const w = S.game.wins(P.b, i);
    if (w) { P.over = true; P.winner = P.turn; P.line = w; }
    else if (P.order.length === S.game.n) P.over = true;
    else P.turn = 3 - P.turn;
    if (P.over) NA.ui.toast.show(P.winner === P.human ? 'You win.' : P.winner ? 'It wins.' : 'A draw.');
    this.bump(); this.renderPlay();
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
    this.use(preset.example || 'maze');
    this.watch = false;
    const maze = this.ex === 'maze';
    this.ro = UI.readouts(ui.readouts, maze
      ? [{ key: 'n', label: 'Episodes', keep: true }, { key: 'a', label: 'Reaches the goal', keep: true, title: 'Greedy route (no exploring) from the start, over 20 tries' }, { key: 'b', label: 'Route', title: 'Steps on the greedy route' }]
      : [{ key: 'n', label: 'Games', keep: true }, { key: 'a', label: 'Beats random', keep: true, title: 'Share of games it wins against a player who moves at random' }, { key: 'b', label: 'Beats rules', title: 'Share it wins against a rule-follower: win if it can, block if it must' }, { key: 'c', label: this.ex === 'three' ? 'Positions' : 'Parameters' }]);
    this.buildHyper(ui.hyper);
    this.buildLeft(ui.left);
    this.buildCenter(ui.center);
    this.tabs = UI.tabs(ui.right, [{ id: 'overview', label: 'Overview' }, { id: 'train', label: 'Training' }, { id: 'inspect', label: maze ? 'Q-values' : 'Its view' }], (id, body) => this.buildTab(id, body));
    this.offTheme = theme.onChange(() => this.bump());
    this.bump();
  };
  LAB.deactivate = function () { if (this.offTheme) this.offTheme(); if (this.resizeObs) this.resizeObs.disconnect(); if (this.arch) this.arch.destroy(); this.watch = false; };
  LAB.archText = function () {
    const S = this.S;
    return S.id === 'maze' ? `Q-table: ${S.maze.cell.filter((c) => c !== '#').length} squares × 4 moves` : S.id === 'three' ? `value table: ${fmtInt(S.brain.size)} positions, learned by self-play` : `value network ${S.brain.nIn} → ${S.brain.nH} tanh → 1 sigmoid, learned by self-play`;
  };
  LAB.stateText = function () {
    const S = this.S;
    if (S.play) return S.play.over ? 'Game over' : S.play.turn === S.play.human ? 'Your move' : 'Its move';
    return this.training ? (this.watch ? 'Watching it learn' : 'Training') : S.n === 0 ? 'Untrained' : 'Paused';
  };
  LAB.frame = function (now) {
    const S = this.S;
    if (this.training && this.watch) { if (now >= (this.pauseUntil || 0) && now - (this.lastTick || 0) >= WATCH_MS[S.id]) { this.lastTick = now; this.watchTick(); } }
    else this.runFrame();
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
    if (this.leftDirty && this.ui.left.offsetParent && (!this.training || now - (this.lastLeft || 0) > 400)) { this.lastLeft = now; this.leftDirty = false; this.renderLeft(); }
    if (this.inspDirty && this.tabs.body.offsetParent && (!this.training || now - (this.lastInsp || 0) > 450)) { this.lastInsp = now; this.inspDirty = false; this.updateTab(); }
  };
  LAB.readouts = function () {
    const S = this.S, h = S.history[S.history.length - 1];
    this.ro.set('n', fmtInt(S.n));
    if (S.id === 'maze') { this.ro.set('a', h ? fmtPct(h.greedy, 0) : '—'); this.ro.set('b', h && h.greedy ? `${fmt(h.route, 1)} steps` : '—'); }
    else { this.ro.set('a', h ? fmtPct(h.rnd.win, 0) : '—'); this.ro.set('b', h ? fmtPct(h.rules.win, 0) : '—'); this.ro.set('c', S.id === 'three' ? fmtInt(S.brain.size) : fmtInt(S.brain.paramCount)); }
  };

  LAB.buildHyper = function (root) {
    const S = this.S, maze = S.id === 'maze';
    root.append(
      UI.field('Exploration ε', UI.select([0, 0.05, 0.1, 0.2, 0.3].map((v) => ({ value: v, label: String(v) })), S.o.eps, (v) => { S.o.eps = +v; this.leftDirty = true; }), 'The share of moves it makes at random, to try things out'),
      maze || S.id === 'three' ? UI.field('Learning rate α', UI.select([0.05, 0.1, 0.2, 0.5, 1].map((v) => ({ value: v, label: String(v) })), S.o.alpha, (v) => { S.o.alpha = +v; this.leftDirty = true; })) : UI.field('Learning rate', UI.select([0.01, 0.02, 0.05, 0.1].map((v) => ({ value: v, label: String(v) })), S.o.lr, (v) => { S.o.lr = +v; this.leftDirty = true; })),
      el('div', { class: 'sheet-actions' },
        el('button', { class: 'btn', type: 'button', onclick: () => { NA.mobile && NA.mobile.sheet(false); this.watch = false; this.fastForward(maze ? 500 : 2000); } }, maze ? 'Train 500 episodes' : 'Train 2,000 games'),
        el('button', { class: 'btn', type: 'button', onclick: () => { NA.mobile && NA.mobile.sheet(false); this.reset(); } }, 'Start from nothing')));
  };

  // ------------------------------------------------------------------ left: the task, how it learns, playing it
  LAB.buildLeft = function (root) {
    const S = this.S, maze = S.id === 'maze', body = UI.panel(root, maze ? 'The maze' : NAMES[S.id]);
    this.leftParts = {};
    const P = this.leftParts, set = (key) => (v) => { S.o[key] = v; this.leftDirty = true; };
    P.eps = UI.slider({ label: 'Exploration ε (moves made at random)', min: 0, max: 0.5, step: 0.01, value: S.o.eps, format: (v) => v.toFixed(2), onInput: set('eps') });
    P.rate = maze || S.id === 'three'
      ? UI.slider({ label: 'Learning rate α', min: 0.01, max: 1, step: 0.01, value: S.o.alpha, format: (v) => v.toFixed(2), onInput: set('alpha') })
      : UI.slider({ label: 'Learning rate', min: 0.005, max: 0.2, step: 0.005, value: S.o.lr, format: (v) => v.toFixed(3), onInput: set('lr') });
    this.watchChip = UI.chip('Watch it learn, move by move', this.watch, (on) => { this.watch = on; if (on && !this.training) this.play(); this.bump(); }, 'Slows training down to one move at a time');
    const learn = [P.eps, P.rate];
    if (maze) {
      P.gamma = UI.slider({ label: 'Discount γ (how much later rewards count)', min: 0.5, max: 0.99, step: 0.01, value: S.o.gamma, format: (v) => v.toFixed(2), onInput: set('gamma') });
      P.slip = UI.slider({ label: 'Slippery floor (moves that go sideways)', min: 0, max: 0.3, step: 0.01, value: S.maze.slip, format: (v) => fmtPct(v, 0), onInput: (v) => { S.maze.slip = v; this.bump(); } });
      learn.push(P.gamma);
      body.append(
        UI.section('The task', UI.note('Get from S to the green goal (+1). Falling into a red pit ends the episode at −1, and every step costs 0.02, so shorter is better. The agent is told nothing else: it learns only from these rewards.'), P.slip),
        UI.section('How it learns: Q-learning', UI.note('It keeps a table Q(square, move): how much reward it expects from that square if it makes that move and then plays its best. After every move it nudges that entry toward the reward it got plus γ times the best value of where it landed.'), ...learn),
        UI.section('Watch', UI.note('Train runs thousands of episodes a second. Watch slows it down to one move at a time, so you can see it wander, find the goal, and the values spread back from it.'), this.watchChip));
    } else {
      const g = S.game;
      P.play = el('div', { class: 'rl-play' });
      body.append(
        UI.section('The game', UI.note(`A game made for this site: two players take turns placing a stone on a ${g.size} × ${g.size} board, ${STONE[1]} first. ${g.k} in a row, across, down or diagonally, wins; a full board is a draw.`)),
        UI.section(`How it learns: self-play${S.id === 'four' ? ' with a neural network' : ''}`, UI.note(S.id === 'three'
          ? 'It plays both sides against itself. It keeps a value for every position it meets: its estimate of the chance that the player who just moved will win. To move, it looks at the position each empty cell would lead to and picks the best. After each game step, the value of its previous position moves a little toward the value of the next one, and at the end toward the result.'
          : 'The same learning as Three-line, but a 5 × 5 board has billions of positions, so instead of a table a small neural network learns to judge them: 50 inputs (where each player’s stones are), 64 hidden neurons, one output: the chance to win. It generalises, so a position it never saw still gets a sensible value.'), ...learn),
        UI.section('Watch', UI.note('Watch slows self-play down to one move at a time: you see each game it learns from.'), this.watchChip),
        UI.section('Play against it', P.play));
      this.renderPlay();
    }
    if (S.id === 'four') body.append(UI.section('The trained network', UI.note('It opened with a network trained offline with this same code for 100,000 games of self-play. Start from nothing to watch one learn from scratch.'),
      el('div', { class: 'sheet-actions' }, el('button', { class: 'btn', type: 'button', onclick: () => { if (this.training) this.pause(); this.loadTrained(); NA.ui.toast.show('The network trained for 100,000 games is back.'); } }, 'Load the trained network'), el('button', { class: 'btn', type: 'button', onclick: () => this.reset() }, 'Start from nothing'))));
  };
  LAB.renderPlay = function () {
    const P = this.leftParts && this.leftParts.play, S = this.S;
    if (!P) return;
    const pl = S.play;
    clear(P).append(
      UI.note(pl ? (pl.over ? (pl.winner === pl.human ? 'You won. Play again?' : pl.winner ? 'It won. Play again?' : 'A draw. Play again?') : pl.turn === pl.human ? `Your move: ${NA.tapWord(true)} an empty cell.` : 'It is thinking…') : 'It plays its best move every time (no exploring), using what it has learned so far.'),
      el('div', { class: 'sheet-actions' },
        el('button', { class: 'btn' + (pl ? '' : ' btn-primary'), type: 'button', onclick: () => this.startPlay(1) }, `Play ${STONE[1]} (first)`),
        el('button', { class: 'btn', type: 'button', onclick: () => this.startPlay(2) }, `Play ${STONE[2]} (second)`),
        pl ? el('button', { class: 'btn btn-quiet', type: 'button', onclick: () => this.stopPlay() }, 'Stop') : null));
  };
  LAB.renderLeft = function () { /* the sliders show their own values; the play box follows the game */ this.renderPlay(); };

  // ------------------------------------------------------------------ centre: the board
  LAB.buildCenter = function (root) {
    const maze = this.ex === 'maze';
    this.exSeg = UI.seg([{ value: 'maze', label: 'Maze' }, { value: 'three', label: 'Three-line' }, { value: 'four', label: 'Four-line' }], this.ex, (v) => { if (v !== this.ex) App.loadPreset(`rl-${v}`); }, 'Example');
    this.viewSeg = UI.seg([{ value: 'board', label: maze ? 'Maze' : 'Board' }, { value: 'arch', label: maze || this.ex === 'three' ? 'Table' : 'Network', title: 'What it has learned, as numbers: click a part for details, double-click or zoom in to open it' }], this.centerView, (v) => this.setCenterView(v), 'View');
    this.boardTools = el('span', { class: 'lab-tools' }, el('span', { class: 'lens-label', text: 'Show' }),
      UI.chip(maze ? 'Values' : 'Its view of each move', this.show.values, (on) => { this.show.values = on; this.dirty = true; }, maze ? 'Colour each square by the value of its best move' : 'In each empty cell, the chance to win it gives the player to move'),
      maze ? UI.chip('Best moves', this.show.arrows, (on) => { this.show.arrows = on; this.dirty = true; }, 'An arrow for the move it rates best on each square') : null);
    root.append(el('div', { class: 'lab-toolbar' }, this.exSeg, this.viewSeg, this.boardTools));
    this.stage = el('div', { class: 'lab-stage' });
    this.canvas = el('canvas', { 'aria-label': maze ? 'The maze, with what the agent has learned about each square' : 'The board' });
    this.stage.append(this.canvas);
    this.archStage = el('div', { class: 'lab-stage arch-stage' });
    this.arch = new NA.ArchView(this.archStage, { label: 'What it has learned: click a part for details; double-click it or zoom in to open it', kicker: maze ? 'Q-table' : this.ex === 'three' ? 'Value table' : 'Value network', build: (size) => NA.archRL.build(this, size), onFocus: (node, path) => { if (this.centerView === 'arch') this.info.set(this.arch.infoFor(node, path)); } });
    root.append(this.stage, this.archStage);
    this.info = UI.infoPanel(root, { open: !(NA.mobile && NA.mobile.mobile) && root.clientHeight >= 680 });
    root.append(this.foot = el('div', { class: 'lab-foot' }, ...(maze
      ? [key('pos', 'goal +1'), key('neg', 'pit −1'), el('span', { class: 'lg-item', text: 'each step −0.02 · arrow: its best move · colour: the value of that move' })]
      : [el('span', { class: 'lg-item' }, el('i', { class: 'dot dot-filled', style: { background: 'var(--s1)', borderColor: 'var(--s1)' } }), `${STONE[1]} moves first`), el('span', { class: 'lg-item' }, el('i', { class: 'dot', style: { border: '2px solid var(--s2)', boxSizing: 'border-box' } }), STONE[2]), el('span', { class: 'lg-item', text: 'numbers: its chance to win after each move, for the player to move' })])));
    this.canvas.addEventListener('click', (e) => this.clickBoard(e));
    this.canvas.addEventListener('pointermove', (e) => { const i = this.cellAt(e); if (i !== this.hoverCell) { this.hoverCell = i; this.dirty = true; } });
    this.canvas.addEventListener('pointerleave', () => { this.hoverCell = -1; this.dirty = true; });
    this.resizeObs = new ResizeObserver(() => { this.dirty = true; });
    this.resizeObs.observe(this.stage);
    this.setCenterView(this.centerView);
  };
  const key = (tok, text) => el('span', { class: 'lg-item' }, el('i', { class: 'key-sq', style: { background: `var(--${tok})` } }), text);
  LAB.setCenterView = function (v) {
    this.centerView = v;
    const arch = v === 'arch';
    this.stage.hidden = arch; this.foot.hidden = arch; this.boardTools.hidden = arch; this.archStage.hidden = !arch;
    this.viewSeg.set(v);
    this.dirty = true; this.arch.dirty = true;
    if (arch) this.arch.refreshFocus(); else this.info.set(BOARD_INFO(this));
  };
  // the board's squares on the canvas
  LAB.grid = function () {
    const S = this.S, cols = S.maze ? S.maze.W : S.game.size, rows = S.maze ? S.maze.H : S.game.size, W = this.stage.clientWidth, Hh = this.stage.clientHeight;
    const cs = Math.max(8, Math.floor(Math.min((W - 28) / cols, (Hh - 28) / rows)));
    return { x: Math.round((W - cs * cols) / 2), y: Math.round((Hh - cs * rows) / 2), cs, cols, rows, W, H: Hh };
  };
  LAB.cellAt = function (e) {
    const b = this.canvas.getBoundingClientRect(), G = this.grid(), c = Math.floor((e.clientX - b.left - G.x) / G.cs), r = Math.floor((e.clientY - b.top - G.y) / G.cs);
    return c < 0 || r < 0 || c >= G.cols || r >= G.rows ? -1 : r * G.cols + c;
  };
  LAB.clickBoard = function (e) {
    const i = this.cellAt(e), S = this.S;
    if (i < 0) return;
    if (S.maze) { if (S.maze.open(i)) { S.sel = i; this.bump(); if (this.tabs.current !== 'inspect') this.tabs.select('inspect'); } return; }
    if (S.play) this.humanMove(i);
    else NA.ui.toast.show('To play against it, choose a side under Play against it.');
  };
  // the position on the board, and whose move it is (or null: a finished game)
  LAB.position = function () {
    const S = this.S;
    if (S.play) return { b: S.play.b, line: S.play.line, order: S.play.order, toMove: S.play.over ? 0 : S.play.turn, play: S.play };
    const sh = S.shown;
    if (!sh) return { b: new Int8Array(S.game.n), line: null, order: [], toMove: 1 };
    return { b: sh.b, line: sh.line, order: sh.order, toMove: sh.winner || sh.order.length === S.game.n || sh.kind === 'self' || sh.done ? 0 : (sh.order.length % 2) + 1 };
  };

  LAB.draw = function () {
    const G = this.grid(), ctx = NA.fitCanvas(this.canvas, G.W, G.H);
    ctx.clearRect(0, 0, G.W, G.H);
    if (this.S.maze) this.drawMaze(ctx, G); else this.drawBoard(ctx, G);
  };
  LAB.drawMaze = function (ctx, G) {
    const S = this.S, M = S.maze, Q = S.Q, cs = G.cs, at = (i) => ({ x: G.x + (i % M.W) * cs, y: G.y + Math.floor(i / M.W) * cs });
    let vmax = 0.05;
    for (let i = 0; i < M.n; i++) if (M.cell[i] === '.' || M.cell[i] === 'S') vmax = Math.max(vmax, Math.abs(Q.max(i)));
    for (let i = 0; i < M.n; i++) {
      const { x, y } = at(i), c = M.cell[i];
      ctx.fillStyle = c === '#' ? theme.c('line-2') : theme.c('panel'); ctx.fillRect(x, y, cs, cs);
      if (c === 'G' || c === 'X') { ctx.fillStyle = theme.c(c === 'G' ? 'pos' : 'neg', 0.32); ctx.fillRect(x, y, cs, cs); }
      else if (c !== '#' && this.show.values && Q.seen[i]) { ctx.fillStyle = theme.divColor(clamp(Q.max(i) / vmax, -1, 1) * 0.9, 0.5); ctx.fillRect(x, y, cs, cs); }
      ctx.strokeStyle = theme.c('line'); ctx.lineWidth = 1; ctx.strokeRect(x + 0.5, y + 0.5, cs - 1, cs - 1);
      const u = cs * 0.3;
      if (c === 'G') S_text(ctx, '+1', x + cs / 2, y + cs / 2, u, 'pos');
      else if (c === 'X') S_text(ctx, '−1', x + cs / 2, y + cs / 2, u, 'neg');
      else if (c === 'S') S_text(ctx, 'S', x + cs * 0.16, y + cs * 0.2, u * 0.6, 'ink-2');
      // its best move, more opaque the clearer the choice
      if (c !== '#' && c !== 'G' && c !== 'X' && this.show.arrows && Q.seen[i]) {
        const a = Q.best(i), qs = Array.from({ length: 4 }, (_, k) => Q.Q[i * 4 + k]).sort((p, q) => q - p), gap = clamp((qs[0] - qs[1]) / (vmax * 0.3 + 1e-9), 0.25, 1);
        const [dx, dy] = RL.DIRS[a], cx = x + cs / 2, cy = y + cs / 2, L = cs * 0.3;
        NA.archKit.arrow(ctx, cx - dx * L * 0.6, cy - dy * L * 0.6, cx + dx * L, cy + dy * L, { color: theme.c('ink', gap), lw: Math.max(1.2, cs * 0.045), head: Math.max(4, cs * 0.12) });
      }
      if (i === S.sel && this.tabs && this.tabs.current === 'inspect') { ctx.strokeStyle = theme.c('focus'); ctx.lineWidth = 2.5; ctx.strokeRect(x + 1.5, y + 1.5, cs - 3, cs - 3); }
      if (i === this.hoverCell && c !== '#') { ctx.strokeStyle = theme.c('focus', 0.6); ctx.lineWidth = 1.5; ctx.strokeRect(x + 1, y + 1, cs - 2, cs - 2); }
    }
    // the path: the episode being watched, or the route it would take now (greedy, from the start)
    const walk = S.walk || (this.watch ? S.walkDone : null), path = walk ? walk.path : this.greedyRoute();
    if (path && path.length > 1) {
      ctx.strokeStyle = walk ? theme.c('s7', 0.85) : theme.c('ink', 0.55); ctx.lineWidth = Math.max(2, cs * 0.06); ctx.setLineDash(walk ? [] : [cs * 0.12, cs * 0.1]); ctx.lineJoin = 'round';
      ctx.beginPath(); path.forEach((s, k) => { const p = at(s); k ? ctx.lineTo(p.x + cs / 2, p.y + cs / 2) : ctx.moveTo(p.x + cs / 2, p.y + cs / 2); }); ctx.stroke(); ctx.setLineDash([]);
    }
    const here = walk ? walk.path[walk.path.length - 1] : null;
    if (here !== null) { const p = at(here); ctx.fillStyle = theme.c('s7'); ctx.beginPath(); ctx.arc(p.x + cs / 2, p.y + cs / 2, cs * 0.2, 0, Math.PI * 2); ctx.fill(); ctx.strokeStyle = theme.c('panel'); ctx.lineWidth = 2; ctx.stroke(); }
  };
  // where its greedy policy goes from the start (stops at a loop)
  LAB.greedyRoute = function () {
    const S = this.S, M = S.maze;
    if (!S.n) return null;
    let s = M.start;
    const path = [s], seen = new Set([s]);
    for (let k = 0; k < 60; k++) { s = M.move(s, S.Q.best(s)); path.push(s); if (M.cell[s] === 'G' || M.cell[s] === 'X' || seen.has(s)) break; seen.add(s); }
    return path;
  };
  LAB.drawBoard = function (ctx, G) {
    const S = this.S, g = S.game, cs = G.cs, pos = this.position(), b = pos.b, at = (i) => ({ x: G.x + (i % g.size) * cs, y: G.y + Math.floor(i / g.size) * cs });
    ctx.fillStyle = theme.c('panel-2'); ctx.fillRect(G.x, G.y, cs * g.size, cs * g.size);
    // its view: for the player to move, the chance to win that each empty cell leads to
    const vals = this.show.values && pos.toMove && !(pos.play && pos.play.turn === pos.play.human && !this.hints) ? RL.moveValues(g, S.brain, b, pos.toMove) : null;
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
        if (i === best) { ctx.strokeStyle = theme.c('ink', 0.7); ctx.lineWidth = 1.5; ctx.beginPath(); ctx.arc(x + cs / 2, y + cs / 2, cs * 0.33, 0, Math.PI * 2); ctx.stroke(); }
      }
      if (i === this.hoverCell && pos.play && !pos.play.over && pos.play.turn === pos.play.human && !b[i]) { ctx.strokeStyle = theme.c(pos.play.human === 1 ? 's1' : 's2'); ctx.lineWidth = 2; ctx.beginPath(); ctx.arc(x + cs / 2, y + cs / 2, cs * 0.36, 0, Math.PI * 2); ctx.stroke(); }
    }
    if (pos.line) {
      const a = at(pos.line[0]), z = at(pos.line[pos.line.length - 1]);
      ctx.strokeStyle = theme.c('ink', 0.85); ctx.lineWidth = Math.max(3, cs * 0.09); ctx.lineCap = 'round';
      ctx.beginPath(); ctx.moveTo(a.x + cs / 2, a.y + cs / 2); ctx.lineTo(z.x + cs / 2, z.y + cs / 2); ctx.stroke(); ctx.lineCap = 'butt';
    }
    // what is on the board
    const sh = S.shown, label = S.play ? (S.play.over ? (S.play.winner === S.play.human ? 'You won' : S.play.winner ? 'It won' : 'A draw') : S.play.turn === S.play.human ? 'Your move' : 'Its move')
      : !sh ? 'An empty board: press Train, or play against it' : sh.kind === 'watch' && !sh.done ? `Self-play, move ${sh.order.length}: ${STONE[(sh.order.length % 2) + 1]} to move` : `Its last self-play game (game ${fmtInt(S.n)}): ${sh.winner ? `${STONE[sh.winner]} won` : 'a draw'} in ${sh.order.length} moves`;
    S_text(ctx, label, G.W / 2, Math.max(10, G.y - 12), 12, 'ink-2', 600);
  };
  const S_text = (ctx, s, x, y, size, tok, w = 650) => NA.archKit.text(ctx, s, x, y, { size, w, align: 'center', color: theme.c(tok) });

  // ------------------------------------------------------------------ right: overview, training, its view
  LAB.buildTab = function (id, body) {
    this.tabId = id; const P = (this.parts = {}), S = this.S, maze = !!S.maze;
    if (id === 'overview') {
      body.append(UI.modelCard(this.preset, (t) => this.runTry(t)));
      P.findings = el('div', { class: 'diag-list' });
      body.append(UI.section('How it is doing', P.findings));
    } else if (id === 'train') {
      const mk = (title, opts, legend, note) => { const c = el('canvas'), chart = new NA.charts.LineChart(c, opts); body.append(UI.section(title, el('div', { class: 'chart-box' }, c), legend, note ? UI.note(note) : null)); return chart; };
      const lg = (items) => el('div', { class: 'mini-legend' }, ...items.map(([n, tok]) => el('span', { class: 'lg-item' }, el('i', { class: 'key-line', style: { background: `var(--${tok})` } }), n)));
      P.tiles = el('div', { class: 'tiles' }); body.append(P.tiles);
      if (maze) {
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
      if (S.maze) {
        clear(P.tiles).append(UI.tile('Episodes', fmtInt(S.n)), UI.tile('Best route', last && last.greedy ? `${fmt(last.route, 1)} steps` : '—', last ? `reaches the goal ${fmtPct(last.greedy, 0)}` : null), UI.tile('Exploration', fmt(S.o.eps, 2)));
        P.a.frame(); P.a.set({ series: [{ name: 'training', color: theme.c('ink-2'), points: h.map((q) => [q.n, q.steps]) }, { name: 'best route', color: theme.c('pos'), points: h.filter((q) => q.greedy > 0).map((q) => [q.n, q.route]) }] });
        P.b.frame(); P.b.set({ series: [{ name: 'reward', color: theme.c('s1'), points: h.map((q) => [q.n, q.ret]) }] });
      } else {
        clear(P.tiles).append(UI.tile('Games', fmtInt(S.n), S.pretrained ? 'includes 100,000 trained offline' : null), UI.tile(S.id === 'three' ? 'Positions in its table' : 'Parameters', S.id === 'three' ? fmtInt(S.brain.size) : fmtInt(S.brain.paramCount)), UI.tile('Beats random', last ? fmtPct(last.rnd.win, 0) : '—'));
        const ser = (key) => [{ name: 'wins', color: theme.c('pos'), points: h.map((q) => [q.n, q[key].win]) }, { name: 'losses', color: theme.c('neg'), points: h.map((q) => [q.n, q[key].loss]) }];
        P.a.frame(); P.a.set({ series: ser('rnd') });
        P.b.frame(); P.b.set({ series: ser('rules') });
      }
    } else if (id === 'inspect') clear(P.box).append(...(S.maze ? this.inspectMaze() : this.inspectGame()));
  };
  const kv = (k, v, note) => el('div', { class: 'kv' }, el('span', { class: 'kv-k', text: k }), el('span', { class: 'kv-v', text: v }), note ? el('span', { class: 'kv-note', text: note }) : null);
  LAB.inspectMaze = function () {
    const S = this.S, M = S.maze, s = S.sel, x = s % M.W, y = Math.floor(s / M.W), best = S.Q.best(s), u = S.update;
    const rows = RL.DIR_NAMES.map((d, a) => kv(`Q(${x}, ${y}, ${d})`, fmt(S.Q.Q[s * 4 + a], 4), a === best ? 'its best move here' : M.move(s, a) === s ? 'into a wall: it stays put' : null));
    const out = [UI.section(`Square (${x}, ${y})`, UI.note(`${NA.tapWord()} any square of the maze to see its four values. Each is how much reward it expects if it makes that move here and then plays its best.`), el('div', { class: 'kv-grid' }, ...rows))];
    if (u) {
      const ux = u.s % M.W, uy = Math.floor(u.s / M.W), mx = S.Q.max(u.s2);
      out.push(UI.section('The last update', el('div', { class: 'fx' },
        el('div', { class: 'fx-line', text: `at (${ux}, ${uy}) it ${u.slipped ? `meant to go ${RL.DIR_NAMES[u.intended]} but slipped ${RL.DIR_NAMES[u.a]}` : `went ${RL.DIR_NAMES[u.a]}`}: reward ${fmt(u.r, 2)}` }),
        el('div', { class: 'fx-line', text: `target = r + γ·max Q(next) = ${fmt(u.r, 2)}${u.done ? ' (the episode ended)' : ` + ${fmt(S.o.gamma, 2)} × ${fmt(mx, 3)}`} = ${fmt(u.done ? u.r : u.r + S.o.gamma * mx, 3)}` }),
        el('div', { class: 'fx-line fx-strong', text: `Q ← Q + α·(target − Q) = ${fmt(u.before, 3)} + ${fmt(S.o.alpha, 2)} × ${fmt(u.err, 3)} = ${fmt(u.after, 3)}` })), UI.note('Shown while you watch it learn: the update after each move.')));
    }
    return out;
  };
  LAB.inspectGame = function () {
    const S = this.S, pos = this.position(), g = S.game;
    if (!pos.toMove) return [UI.section('Its view', UI.note('The game on the board is over. Train, watch it learn, or play against it to see how it rates each move.'))];
    const { moves, vals } = RL.moveValues(g, S.brain, pos.b, pos.toMove), order = moves.map((m, k) => [m, vals[k]]).sort((a, b) => b[1] - a[1]);
    const name = (m) => `row ${Math.floor(m / g.size) + 1}, column ${(m % g.size) + 1}`;
    return [UI.section(`${STONE[pos.toMove]} to move: how it rates each move`, UI.note(`For each empty cell, the chance that ${STONE[pos.toMove]} goes on to win after playing there, by its ${S.id === 'three' ? 'table (0.5 for a position it has never met)' : 'network'}. It plays the top one.`),
      el('div', { class: 'kv-grid' }, ...order.slice(0, 10).map(([m, v], k) => kv(name(m), fmtPct(v, 0), k === 0 ? 'its choice' : v === 1 ? 'wins at once' : null))))];
  };

  // ------------------------------------------------------------------ explanations
  const FINDINGS = (L) => {
    const S = L.S, h = S.history, last = h[h.length - 1], out = [];
    if (!S.n) return [['info', 'Not trained yet', 'Press Train. It starts knowing nothing: every value is 0 (the maze) or a coin flip (the games).']];
    if (S.maze) {
      if (!last) out.push(['info', 'Exploring', 'It has not found the goal often enough to have a route yet.']);
      else if (last.greedy === 1) out.push(['good', `Found a route: ${fmt(last.route, 0)} steps`, S.maze.slip ? 'On a slippery floor every step near a pit is a risk, so the best route may be longer than the shortest one.' : 'The shortest way through is 12 steps: it learned it from rewards alone.']);
      else out.push(['warn', `Its route reaches the goal ${fmtPct(last.greedy, 0)} of the time`, S.maze.slip ? 'The floor is slippery: some slips end in a pit whatever it does. A smaller learning rate averages the luck out.' : 'Still learning: keep training.']);
      if (S.o.eps === 0) out.push(['warn', 'No exploring', 'With ε = 0 it only ever tries the move it already rates best, so it can settle on the first route it found, or none.']);
    } else {
      out.push([last && last.rnd.win > 0.9 ? 'good' : 'info', `Beats a random player ${last ? fmtPct(last.rnd.win, 0) : '—'} of the time`, 'A random player is easy; this shows it learned to make lines and to finish them.']);
      if (last) out.push([last.rules.win > 0.7 ? 'good' : 'warn', `Against the rule-follower: wins ${fmtPct(last.rules.win, 0)}, loses ${fmtPct(last.rules.loss, 0)}`, last.rules.loss > 0.2 ? 'It still walks into some threats the rule-follower blocks or completes. More games, or less exploring, help.' : 'It sets up positions with two ways to win, so blocking one is not enough.']);
      out.push(['info', S.id === 'three' ? `${fmtInt(S.brain.size)} positions in its table` : `${fmtInt(S.brain.paramCount)} parameters`, S.id === 'three' ? 'Each one met in self-play, with rotations and mirror images counted once. A table only knows positions it has seen.' : 'A table for this board would need billions of entries; the network judges any position from the pattern of stones.']);
    }
    return out;
  };
  const BOARD_INFO = (L) => {
    const S = L.S;
    if (S.maze) return { kicker: 'Maze', title: 'Q-learning: values from rewards', about: [
      'Each square is coloured by the value of its best move (blue: it expects reward from here, orange: it expects to lose), and the arrow is that move. At first every value is 0. The first time it stumbles into the goal, the square next to it learns that one move pays; on later visits that value is copied, a little discounted, to the squares before it. The values spread back from the goal like a scent, and the arrows line up into a route.',
      `It explores: ${fmtPct(S.o.eps, 0)} of its moves are random (ε), so it keeps finding new ways. The dashed line is the route it would take now, choosing its best move on every square. Tick Watch to see one move at a time.`,
      'This is Q-learning (Watkins, 1989), the start of deep reinforcement learning: DQN (2013) replaced the table with a neural network to play Atari games from the screen.'], code: NA.archRL.py.maze(L) };
    const three = S.id === 'three';
    return { kicker: NAMES[S.id], title: three ? 'Learning a game by playing itself' : 'A network that judges positions', about: [
      three ? 'It learns Three-line by playing both sides against itself, starting from nothing: every position it has not met is worth 0.5, a coin flip. It keeps a value for each position it meets, the chance that the player who just moved will win, and moves to the position it values most. When a game ends, the last positions move toward the result (1 for the winner, 0 for the loser), and on later games those values spread back to the moves that led there. This is temporal-difference learning, as in the classic self-play program for backgammon, TD-Gammon (1992).'
        : 'Four-line has billions of positions, too many to list, so a neural network judges them: the board goes in (50 inputs: where each player’s stones are), 64 tanh neurons look for patterns, and one sigmoid output gives the chance to win. It learns exactly as Three-line does, by self-play and temporal differences, but each lesson changes weights shared by every position, so it can judge positions it has never seen. AlphaZero (2017) added search on top of this idea to master chess and Go.',
      'The numbers on the empty cells are its view of each move: the chance the player to move would go on to win after playing there. A cell that wins at once shows 100%. When it plays, it picks the highest.',
      'Play against it under Play against it. It always plays its best move by what it has learned; train more and it gets harder to beat.'], code: three ? NA.archRL.py.three(L) : NA.archRL.py.four(L) };
  };

  // ------------------------------------------------------------------ experiments, and the gallery's picture
  LAB.runTry = function (t) {
    const S = this.S, retrain = (msg) => { this.fresh(); this.watch = false; this.buildAgain(); this.fastForward(t.n || (S.maze ? 1000 : 3000)); if (msg) NA.ui.toast.show(msg); };
    if (t.action === 'watch') {
      // the maze starts over, so you see it from nothing; a game keeps what it knows and shows its next games
      if (S.maze) this.fresh();
      if (S.play) S.play = null;
      this.watch = true; this.buildAgain(); this.play();
      NA.ui.toast.show(S.maze ? 'Watching it learn from nothing, one move at a time.' : 'Watching it play itself, one move at a time.');
    }
    else if (t.action === 'slippery') { S.maze.slip = 0.15; S.o.alpha = 0.1; retrain('A slippery floor (15% of moves go sideways) and a smaller learning rate: retraining.'); }
    else if (t.action === 'noExplore') { S.o.eps = 0; retrain('No exploring (ε = 0): retraining.'); }
    else if (t.action === 'shortSighted') { S.o.gamma = 0.5; retrain('Discount γ = 0.5: rewards a few steps away count for little. Retraining.'); }
    else if (t.action === 'train') { this.watch = false; this.fastForward(t.n || 3000); }
    else if (t.action === 'scratch') { retrain('A new network that knows nothing: watch the charts as it learns.'); }
    else if (t.action === 'trained') { this.pause(); this.loadTrained(); NA.ui.toast.show('The network trained for 100,000 games is back.'); }
    else if (t.action === 'playFirst') this.startPlay(1);
    else if (t.action === 'playSecond') this.startPlay(2);
    else if (t.action === 'default') { Object.assign(S.o, S.maze ? { eps: 0.1, alpha: 0.5, gamma: 0.95 } : S.id === 'three' ? { eps: 0.1, alpha: 0.2 } : { eps: 0.1, lr: 0.05 }); if (S.maze) S.maze.slip = 0; this.fresh(); if (S.id === 'four') this.loadTrained(); this.buildAgain(); }
    this.tabs.select(this.tabId);
    // an experiment that trains reports its result when it is done
    if (t.predict) return new Promise((res) => { const wait = () => (this.training ? setTimeout(wait, 200) : res(this.measured())); wait(); });
    return null;
  };
  LAB.measured = function () {
    const S = this.S, h = S.history, last = h[h.length - 1];
    if (!last) return '';
    if (S.maze) return `Measured: its route reaches the goal ${fmtPct(last.greedy, 0)} of the time${last.greedy ? `, in ${fmt(last.route, 1)} steps` : ''}.`;
    const first = h.filter((q) => q.first !== undefined), ninety = h.find((q) => q.rnd.win >= 0.9);
    return `Measured: ${first.length ? `blue won ${fmtPct(first.reduce((s, q) => s + q.first, 0) / first.length, 0)} of its self-play games; ` : ''}it beats a random player ${fmtPct(last.rnd.win, 0)} of the time${ninety ? ` (9 in 10 by game ${fmtInt(ninety.n)})` : ''}.`;
  };
  // the left panel and the hyper sheet show the settings: rebuilt after an experiment changes them
  LAB.buildAgain = function () { clear(this.ui.left); this.buildLeft(this.ui.left); clear(this.ui.hyper); this.buildHyper(this.ui.hyper); this.bump(); };
  LAB.glyph = function (ctx, W, Hh, presetId) {
    const ex = (presetId || 'rl-maze').slice(3);
    if (ex === 'maze') {
      const M = new RL.Maze(), cs = Math.min((W - 8) / M.W, (Hh - 8) / M.H), x0 = (W - cs * M.W) / 2, y0 = (Hh - cs * M.H) / 2;
      M.cell.forEach((c, i) => { ctx.fillStyle = c === '#' ? theme.c('line-2') : c === 'G' ? theme.c('pos', 0.7) : c === 'X' ? theme.c('neg', 0.6) : theme.c('line', 0.5); ctx.fillRect(x0 + (i % M.W) * cs + 0.5, y0 + Math.floor(i / M.W) * cs + 0.5, cs - 1, cs - 1); });
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
