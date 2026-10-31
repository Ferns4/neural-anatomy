/* rl-envs.js — small worlds to learn in from rewards, shared by the RL lab and the model builder: levers that pay
   when they match a light (a contextual bandit), a maze (seen as the square it is on, or as what is around it), a
   pole to keep upright on a moving cart, and a falling ball to catch with a paddle. Each gives an observation,
   takes one of a few actions, and answers with a reward and the next observation until the episode ends. Most can
   be seen more than one way (view): a few numbers, a picture, or the last few readings; input says which, in the
   model builder's terms (a vector, an image of channels × height × width, or a sequence of positions × numbers).
   The model builder can change how each one pays (a reward table of everything it could be paid for, any
   amounts, or a formula of its own), leave inputs out or give it extra ones (its last move, the time left, a
   random number, and some of each world's own), add sensor noise, and change the worlds themselves: the pole's
   push, length, gravity, track and the lean it falls at, the catch's board and paddle. Everything is seeded, so
   a run can be repeated exactly. Needs rl.js (the mazes); a reward formula needs builder-tasks.js (the formula
   language). */
(function (NA) {
  'use strict';
  const E = {};
  const RL = () => NA.rl;
  const clamp = (v, a, b) => (v < a ? a : v > b ? b : v);

  // ------------------------------------------------------------------ rewards, inputs and noise
  /* What each world pays for, and how much: its reward table, with the RL lab's amounts as the defaults. A key
     given in the options replaces its default; rFormula replaces the whole table with a formula in the names of
     REWARD_VARS (the builder's formula language: no eval). */
  E.REWARDS = {
    levers: [{ key: 'rPay', label: 'When a lever pays', def: 1 }, { key: 'rNoPay', label: 'When it does not pay', def: 0 }, { key: 'rMatch', label: 'Pulling the lever of the light’s colour, paid or not', def: 0 }],
    maze: [{ key: 'rGoal', label: 'Reaching the goal', def: 1 }, { key: 'rPit', label: 'Falling in a pit', def: -1 }, { key: 'rStep', label: 'Each ordinary step', def: -0.02 },
      { key: 'rWall', label: 'Walking into a wall', def: 0 }, { key: 'rNew', label: 'A square not yet visited', def: 0 }, { key: 'rCloser', label: 'The closer bonus: a step closer to the goal', def: 0.1 },
      { key: 'rRevisit', label: 'Going back to a square it has been on', def: 0 }, { key: 'rNearPit', label: 'Each step next to a pit', def: 0 }, { key: 'rTurn', label: 'Changing direction', def: 0 }, { key: 'rLost', label: 'Running out of steps', def: 0 }],
    pole: [{ key: 'rUp', label: 'Every step it stays up', def: 1 }, { key: 'rFell', label: 'When it falls', def: 0 }, { key: 'rLean', label: 'Each step, per degree the pole leans', def: 0 },
      { key: 'rOff', label: 'Each step, per metre the cart is from the middle', def: 0 }, { key: 'rSwitch', label: 'Each change in the push’s direction', def: 0 }, { key: 'rLasted', label: 'Lasting the whole episode', def: 0 }],
    catch: [{ key: 'rCatch', label: 'Catching the ball', def: 1 }, { key: 'rMiss', label: 'Missing it', def: -1 }, { key: 'rMove', label: 'Moving the paddle', def: 0 },
      { key: 'rNear', label: 'A near miss: the ball lands next to the paddle', def: 0 }, { key: 'rUnder', label: 'Each step the paddle is under the ball', def: 0 }, { key: 'rStep', label: 'Every step', def: 0 }],
  };
  E.REWARD_VARS = {
    levers: [['paid', '1 when the lever pays out'], ['right', '1 when it was the lever of the light’s colour'], ['unpaid', '1 when it does not pay'], ['lever', 'the lever it pulled: 0 for A, 1 for B…'], ['light', 'the light that is on: 0 for A, 1 for B…']],
    maze: [['goal', '1 on reaching the goal'], ['pit', '1 on falling in a pit'], ['wall', '1 on walking into a wall (it stays put)'], ['closer', '1 for a step closer to the goal by the shortest safe way, −1 for a step away'],
      ['new', '1 on a square it has not been on this episode'], ['dist', 'steps from here to the goal, the safe way (from a pit: through pits)'], ['t', 'steps so far'], ['revisit', '1 on going back to a square it has been on'],
      ['nearpit', '1 when a pit is right next to it'], ['turned', '1 when it moved in another direction than last time'], ['lost', '1 when it runs out of steps'], ['slipped', '1 when the floor sent it sideways'],
      ['x', 'its square across, 0 at the left'], ['y', 'its square down, 0 at the top'], ['visited', 'how many squares it has been on']],
    pole: [['x', 'the cart’s position: −2.4 and 2.4 are the ends of the track'], ['v', 'the cart’s speed'], ['angle', 'the pole’s lean in degrees: it falls past ±12'],
      ['spin', 'how fast the pole turns, degrees a second'], ['fell', '1 on the step it falls'], ['push', '1 for a push right, −1 left'], ['t', 'steps so far'],
      ['switched', '1 when the push changed direction'], ['lasted', '1 on the last step, when it never fell'], ['left', 'steps left until the end']],
    catch: [['caught', '1 when the ball lands on the paddle'], ['missed', '1 when it lands beside it'], ['moved', '1 when the paddle moved'],
      ['dx', 'the ball across minus the paddle across, in squares'], ['down', 'the ball’s row, 0 at the top to 7'], ['t', 'steps so far'], ['near', '1 when it lands right next to the paddle'],
      ['under', '1 while the paddle is under the ball'], ['bx', 'the ball across, 0 at the left'], ['px', 'the paddle across (its middle)'], ['land', 'where the ball will land, across'], ['drift', 'which way the ball drifts: −1 left, 0, 1 right']],
  };
  const rewardTable = (id, o) => Object.fromEntries(E.REWARDS[id].map((r) => [r.key, typeof o[r.key] === 'number' && Number.isFinite(o[r.key]) ? o[r.key] : r.def]));
  /* The table's reward in parts: each step adds its amounts in turn with pay(parts, key, amount, r) → the new r, and
     keeps the ones that are not 0 in parts (what the reward came from, for the pictures of an episode). Added up in
     the order the world always added them, r is the same number to the last bit. */
  const pay = (parts, key, v, r) => { if (!v) return r; parts[key] = (parts[key] || 0) + v; return r + v; };
  /* Your own reward, compiled once; null (the table) when there is none or it does not compile. A step where it is
     no number (∞, or 0 / 0) pays 0 instead, and is noted (how often, and its names' values the first time) for the
     reward panel to say so: E.formulaTrouble(id, src). */
  const TROUBLE = new Map();
  E.formulaTrouble = (id, src) => TROUBLE.get(`${id}|${String(src || '').slice(0, 200)}`) || null;
  E.rewardFormula = (id, src) => {
    const C = NA.builderTasks && NA.builderTasks.compile;
    if (!src || !C) return null;
    const text = String(src).slice(0, 200), c = C(text, E.REWARD_VARS[id].map((v) => v[0]));
    if (c.error) return null;
    const f = c.f, key = `${id}|${text}`;
    return (v) => {
      const r = f(v);
      if (Number.isFinite(r)) return r;
      let t = TROUBLE.get(key);
      if (!t) { if (TROUBLE.size > 50) TROUBLE.clear(); TROUBLE.set(key, (t = { bad: 0, first: Object.assign({}, v) })); }
      t.bad++;
      return 0;
    };
  };
  /* Inputs that can be left out, by name, in the views that are a list of numbers. A model that is not told the
     speeds, say, has to do without them (or work them out, given a memory). */
  const span = (a, b) => NA.range(b - a).map((i) => a + i);
  E.INPUT_GROUPS = {
    pole: { state: [['cart position', [0]], ['cart speed', [1]], ['pole angle', [2]], ['pole turning speed', [3]]] },
    catch: { numbers: [['ball across', [0]], ['ball down', [1]], ['ball drift', [2]], ['paddle across', [3]]] },
    maze: { around: [['walls', span(0, 24)], ['pits', span(24, 48)], ['the goal', span(48, 72)], ['where it has been', span(72, 96)], ['which way the goal is', span(96, 104)]] },
  };
  /* Extra inputs it can be given on top of what it sees, by name, in the views that are a list of numbers: in every
     world its last move (one input per move), the time left and a random number, and some of each world's own
     (E.EXTRAS, added by the worlds). Each is { name, what, n: how many inputs (or a function of the world), names:
     theirs, fill(world, out, at) }; they come after the inputs it keeps, in the order listed. */
  const rngOf = (w) => w.rng || (w.rng = new NA.RNG(5));
  E.COMMON_EXTRAS = [
    { name: 'its last move', what: 'the move it made a step ago, one input per move (all 0 at the start): it can see what it has just done', n: (w) => w.nA, names: (w) => w.actions.map((a) => `last move: ${a}`),
      fill: (w, o, k) => { const a = w.lastMove; if (Number.isInteger(a) && a >= 0 && a < w.nA) o[k + a] = 1; } },
    { name: 'time left', what: 'how much of the longest episode is still to come: 1 at the start, 0 at the end', n: 1, names: ['time left'], fill: (w, o, k) => { o[k] = 1 - w.t / w.maxSteps; } },
    { name: 'a random number', what: 'a new random number from −1 to 1 every step, which tells it nothing: can it learn to ignore it?', n: 1, names: ['a random number'], fill: (w, o, k) => { o[k] = rngOf(w).uniform(-1, 1); } },
  ];
  E.EXTRAS = {};
  // the inputs a world keeps (hide: names of groups to leave out), the extra ones it is given (extra: their names) and
  // the noise on each (noise: its spread)
  const OLD_NAMES = { 'the exit': 'the goal', 'which way the exit is': 'which way the goal is' }; // (the maze's inputs before version 29)
  function sensors(w, id, o) {
    const groups = (E.INPUT_GROUPS[id] || {})[w.view], hide = Array.isArray(o.hide) ? o.hide.map((h) => OLD_NAMES[h] || h) : [];
    w.noise = typeof o.noise === 'number' && o.noise > 0 ? Math.min(o.noise, 2) : 0;
    w.hidden = []; w.extras = null; w.shown = [];
    const drop = new Set(groups ? groups.filter(([name]) => hide.includes(name)).flatMap(([, idx]) => idx) : []);
    if (drop.size && drop.size < w.obsSize) { // at least one input stays
      w.hidden = groups.filter(([name]) => hide.includes(name)).map(([name]) => name);
      w.keep = NA.range(w.obsSize).filter((i) => !drop.has(i));
      w.obsNames = w.keep.map((i) => w.obsNames[i]);
      w.obsSize = w.keep.length;
      // a sequence that loses the same numbers at every position stays a sequence, of fewer numbers each
      const I = w.input, cols = I.kind === 'series' ? NA.range(I.dim).filter((j) => !drop.has(j)) : [];
      const even = cols.length > 0 && NA.range(I.len).every((t) => NA.range(I.dim).every((j) => drop.has(t * I.dim + j) === drop.has(j)));
      w.input = even ? { kind: 'series', len: I.len, dim: cols.length } : { kind: 'vector', dim: w.obsSize };
    }
    const want = Array.isArray(o.extra) ? o.extra : [], list = w.input.kind === 'vector' ? (E.EXTRAS[id] || []).filter((e) => want.includes(e.name)) : [];
    if (!list.length) return;
    w.extras = list.map((e) => ({ fill: e.fill, n: typeof e.n === 'function' ? e.n(w) : e.n }));
    w.shown = list.map((e) => e.name);
    w.obsNames = [...w.obsNames, ...list.flatMap((e) => (typeof e.names === 'function' ? e.names(w) : e.names))];
    w.obsSize += w.extras.reduce((sum, e) => sum + e.n, 0);
    w.input = { kind: 'vector', dim: w.obsSize };
    // its last move: noted before each step (none at the start of an episode)
    if (w.shown.includes('its last move')) {
      const step = w.step, reset = w.reset;
      w.step = function (a, rng) { this.lastMove = a; return step.call(this, a, rng); };
      w.reset = function (...args) { this.lastMove = -1; return reset.apply(this, args); };
    }
  }
  // what the model is given: the world's numbers, those it keeps and the extra ones, plus noise
  function sensed() {
    const x = this.rawObs();
    if (!this.keep && !this.noise && !this.extras) return x;
    let y = this.keep ? Float32Array.from(this.keep, (i) => x[i]) : x;
    if (this.extras) { const z = new Float32Array(this.obsSize); z.set(y); let k = y.length; for (const e of this.extras) { e.fill(this, z, k); k += e.n; } y = z; }
    if (this.noise) { const r = this.rng || (this.rng = new NA.RNG(5)); for (let i = 0; i < y.length; i++) y[i] += this.noise * r.gauss(); }
    return y;
  }

  // ------------------------------------------------------------------ levers: a contextual bandit
  /* A light shows one of k colours; pulling the lever of that colour pays 1 most of the time (p), any other lever
     only now and then (q). One pull per episode: the simplest problem where the right action depends on what you
     see, and the only one here with no delayed reward. */
  class Levers {
    constructor(o = {}) {
      this.k = o.levers || 3; this.p = typeof o.p === 'number' ? o.p : 0.8; this.q = typeof o.q === 'number' ? o.q : 0.2;
      this.obsSize = this.k; this.input = { kind: 'vector', dim: this.k }; this.nA = this.k; this.maxSteps = 1;
      this.actions = ['lever A', 'lever B', 'lever C', 'lever D', 'lever E'].slice(0, this.k);
      this.obsNames = this.actions.map((_, i) => `light ${'ABCDE'[i]}`);
      this.rw = rewardTable('levers', o); this.rf = E.rewardFormula('levers', o.rFormula); sensors(this, 'levers', o);
    }
    reset(rng) { this.rng = rng; this.light = rng.int(this.k); this.t = 0; this.last = null; return this.obs(); }
    rawObs() { const o = new Float32Array(this.k); o[this.light] = 1; return o; }
    step(a, rng) {
      const right = a === this.light, paid = rng.next() < (right ? this.p : this.q), rw = this.rw, parts = this.rf ? null : {};
      let r = 0;
      if (this.rf) r = this.rf({ paid: +paid, right: +right, unpaid: +!paid, lever: a, light: this.light });
      else { r = pay(parts, 'rPay', paid ? rw.rPay : 0, r); r = pay(parts, 'rNoPay', paid ? 0 : rw.rNoPay, r); r = pay(parts, 'rMatch', right ? rw.rMatch : 0, r); }
      this.t++; this.last = { a, r, paid };
      return { obs: this.obs(), r, done: true, parts };
    }
    snap() { return { light: this.light, last: this.last }; }
    // the share of lights for which its best lever is the right one (every light once)
    score(policy) { let ok = 0; for (let l = 0; l < this.k; l++) { this.light = l; if (policy(this.obs()) === l) ok++; } return ok / this.k; }
  }

  // ------------------------------------------------------------------ a maze
  /* One of the RL lab's mazes. Rewards as there: +1 at the goal, −1 in a pit, −0.02 a step. view 'square': the
     observation is which square it is on (one input per square, a one-hot of 48): like a table, it can learn this
     maze and no other. view 'around': the 24 squares around it (walls, pits, the goal, where it has been) and which
     way the goal is (104 inputs), as the lab's Q-network sees it. view 'picture': the whole maze as an image of 5
     channels, 6 × 8 (walls, pits, the goal, where it is, where it has been), for convolutions. random: a new random
     maze every episode. closer: a bonus of 0.1 for every step that brings it closer to the goal by the shortest
     safe way, and as much off for every step away (reward shaping: it cannot change which route is best, since
     any route's bonuses add up to the same, but it says which way is warmer long before the goal is found). */
  /* A maze of your own (the model builder's editor draws it): 6 rows of 8 squares, each floor (.), a wall (#), a pit
     (X), the start (S) or the goal (G); one start, one goal, and a safe way from one to the other. { ok, why, steps }. */
  E.mazeCheck = function (rows) {
    if (!Array.isArray(rows) || rows.length !== 6 || !rows.every((r) => typeof r === 'string' && /^[.#XSG]{8}$/.test(r))) return { ok: false, why: 'A maze is 6 rows of 8 squares.' };
    const all = rows.join(''), nS = all.split('S').length - 1, nG = all.split('G').length - 1;
    if (nS !== 1) return { ok: false, why: nS ? 'It has more than one start: keep one.' : 'It needs a start: put one with the Start tool.' };
    if (nG !== 1) return { ok: false, why: nG ? 'It has more than one goal: keep one.' : 'It needs a goal: put one with the Goal tool.' };
    const steps = RL().shortest(new (RL().Maze)(rows));
    return steps ? { ok: true, steps } : { ok: false, why: 'There is no safe way from the start to the goal: walls or pits block every route.' };
  };
  class MazeWorld {
    constructor(o = {}) {
      this.view = ['square', 'picture'].includes(o.view) ? o.view : 'around'; this.random = !!o.random; this.slip = o.slip || 0; this.closer = !!o.closer;
      const def = o.maze === 'own' && E.mazeCheck(o.rows).ok ? { rows: o.rows } : RL().MAZE_SET.find((m) => m.id === o.maze) || RL().MAZE_SET[0];
      this.fixed = new (RL().Maze)(def.rows); this.fixed.slip = this.slip; this.maze = this.fixed; this.dist = MazeWorld.distances(this.fixed);
      const n = this.fixed.n, W = this.fixed.W, H = this.fixed.H;
      this.input = this.view === 'square' ? { kind: 'vector', dim: n } : this.view === 'picture' ? { kind: 'image', c: 5, h: H, w: W } : { kind: 'vector', dim: 104 };
      this.obsSize = this.view === 'square' ? n : this.view === 'picture' ? 5 * n : 104; this.nA = 4; this.maxSteps = o.maxSteps > 0 ? Math.min(o.maxSteps, 1000) : 80;
      this.actions = ['up', 'right', 'down', 'left'];
      const CH = ['a wall', 'a pit', 'the goal', 'you', 'been'];
      this.obsNames = this.view === 'square' ? NA.range(n).map((i) => `on square (${i % W}, ${Math.floor(i / W)})`)
        : this.view === 'picture' ? NA.range(5 * n).map((i) => `${CH[Math.floor(i / n)]} at (${(i % n) % W}, ${Math.floor((i % n) / W)})`)
          : NA.range(104).map((i) => RL().MazeNet.inputName(i));
      this.rw = rewardTable('maze', o); this.rf = E.rewardFormula('maze', o.rFormula); sensors(this, 'maze', o);
      // the safe distances to the goal, for each random maze too when something needs them (and, for a formula, the
      // distances through pits as well: how far the goal is from a pit it fell in)
      this.needDist = this.closer || !!this.rf || this.shown.includes('safe steps to the goal');
      this.anyDist = this.rf ? MazeWorld.distances(this.fixed, true) : null;
    }
    // where it is on the maze, 0 to 1 each way (one maze only: on random mazes, none)
    where() { if (this.random) return null; const M = this.maze; return [((this.s % M.W) + 0.5) / M.W, (Math.floor(this.s / M.W) + 0.5) / M.H]; }
    // the fewest safe steps from every square to the goal (-1: none); pits: through pits as well
    static distances(M, pits) {
      const d = new Int16Array(M.n).fill(-1), q = [M.goal];
      d[M.goal] = 0;
      while (q.length) { const s = q.shift(); for (let a = 0; a < 4; a++) { const t = M.move(s, a); if (d[t] >= 0 || (M.cell[t] === 'X' && !pits) || M.cell[t] === '#') continue; d[t] = d[s] + 1; q.push(t); } }
      return d;
    }
    reset(rng) {
      this.rng = rng;
      if (this.random) { this.maze = new (RL().Maze)(RL().randomMaze(() => rng.next())); this.maze.slip = this.slip; if (this.needDist) this.dist = MazeWorld.distances(this.maze); if (this.rf) this.anyDist = MazeWorld.distances(this.maze, true); }
      else if (this.maze !== this.fixed) { this.maze = this.fixed; this.dist = MazeWorld.distances(this.fixed); if (this.rf) this.anyDist = MazeWorld.distances(this.fixed, true); }
      this.begin(this.maze);
      return this.obs();
    }
    // at the start of maze M: no steps yet, only S visited, no direction yet
    begin(M) { this.s = M.start; this.seen = new Uint8Array(M.n); this.seen[this.s] = 1; this.t = 0; this.path = [this.s]; this.end = null; this.dir = -1; this.visited = 1; }
    // a pit on one of the four squares next to s
    pitNear(s) { const M = this.maze; for (let a = 0; a < 4; a++) { const q = M.move(s, a); if (q !== s && M.cell[q] === 'X') return true; } return false; }
    rawObs() {
      const x = new Float32Array(this.view === 'square' ? this.maze.n : this.view === 'picture' ? 5 * this.maze.n : 104), M = this.maze, n = M.n;
      if (this.view === 'square') x[this.s] = 1;
      else if (this.view === 'picture') {
        for (let i = 0; i < n; i++) { const c = M.cell[i]; if (c === '#') x[i] = 1; else if (c === 'X') x[n + i] = 1; else if (c === 'G') x[2 * n + i] = 1; if (this.seen[i]) x[4 * n + i] = 1; }
        x[3 * n + this.s] = 1;
      } else for (const i of RL().MazeNet.inputs(M, this.s, this.seen)) x[i] = 1;
      return x;
    }
    step(a, rng) {
      const s0 = this.s, t = this.maze.step(this.s, a, rng.fn || (rng.fn = () => rng.next())), fresh = !this.seen[t.s];
      this.s = t.s; this.seen[t.s] = 1; this.t++; this.path.push(t.s);
      const done = t.done || this.t >= this.maxSteps;
      if (done) this.end = t.done ? (t.r > 0 ? 'goal' : 'pit') : 'lost';
      const M = this.maze, c = M.cell[t.s], goal = c === 'G', pit = c === 'X', wall = t.s === s0, d0 = this.dist[s0], d1 = this.dist[t.s];
      const closer = !pit && d0 >= 0 && d1 >= 0 ? d0 - d1 : 0, rw = this.rw;
      // going back to a square (not staying put at a wall), a turn (the way it really went, slipping or not), a pit next to it
      const revisit = !fresh && !wall, turned = this.dir >= 0 && t.a !== this.dir, lost = this.end === 'lost', nearPit = (this.rf || rw.rNearPit) && !pit && !goal ? this.pitNear(t.s) : false;
      this.dir = t.a; if (fresh) this.visited++;
      const parts = this.rf ? null : {};
      let r = 0;
      // (dist: the safe way; from a pit, or a square no safe way leaves, the way through pits)
      if (this.rf) r = this.rf({ goal: +goal, pit: +pit, wall: +wall, closer, new: +(fresh && !goal && !pit), dist: d1 >= 0 ? d1 : Math.max(0, this.anyDist[t.s]), t: this.t, revisit: +revisit, nearpit: +nearPit, turned: +turned, lost: +lost, slipped: +!!t.slipped, x: t.s % M.W, y: Math.floor(t.s / M.W), visited: this.visited });
      else {
        r = pay(parts, goal ? 'rGoal' : pit ? 'rPit' : 'rStep', goal ? rw.rGoal : pit ? rw.rPit : rw.rStep, r);
        if (wall) r = pay(parts, 'rWall', rw.rWall, r);
        if (fresh && !goal && !pit) r = pay(parts, 'rNew', rw.rNew, r);
        if (this.closer) r = pay(parts, 'rCloser', rw.rCloser * closer, r);
        if (revisit) r = pay(parts, 'rRevisit', rw.rRevisit, r);
        if (nearPit) r = pay(parts, 'rNearPit', rw.rNearPit, r);
        if (turned) r = pay(parts, 'rTurn', rw.rTurn, r);
        if (lost) r = pay(parts, 'rLost', rw.rLost, r);
      }
      return { obs: this.obs(), r, done, parts };
    }
    // (pay: the amounts at the goal and in a pit, which the picture shows; none with a formula of your own)
    snap() { return { rows: this.maze.rows, s: this.s, path: this.path.slice(), end: this.end, pay: this.rf ? null : { goal: this.rw.rGoal, pit: this.rw.rPit } }; }
    // where its moves stop (across, and up from the bottom row): its style of play, for MAP-Elites
    behavior() { const W = this.maze.W, H = this.maze.H; return [((this.s % W) + 0.5) / W, 1 - (Math.floor(this.s / W) + 0.5) / H]; }
    /* Its best moves from S: 1 if they reach the goal (a slippery floor: the share of 5 tries). Trained on random
       mazes, the share of 20 new mazes it reaches the goal of (the same 20 every time, none of them trained on). */
    score(policy, rng) {
      const random = this.random, keep = this.maze, list = random ? RL().testMazes().slice(0, 20) : NA.range(this.slip ? 5 : 1).map(() => this.fixed);
      let ok = 0;
      this.random = false;
      for (const m of list) {
        this.maze = m; this.dist = MazeWorld.distances(m); if (this.rf) this.anyDist = MazeWorld.distances(m, true);
        this.begin(m);
        let o = this.obs(), done = false;
        while (!done) ({ obs: o, done } = this.step(policy(o), rng));
        if (this.end === 'goal') ok++;
      }
      this.random = random; this.maze = keep; this.dist = MazeWorld.distances(keep); if (this.rf) this.anyDist = MazeWorld.distances(keep, true);
      return ok / list.length;
    }
  }

  // ------------------------------------------------------------------ a pole on a cart
  /* The classic balancing problem (Barto, Sutton and Anderson, 1983): a pole hinged on a cart that can only be
     pushed left or right. It falls if it leans more than 12°, or the cart runs off its 4.8-unit track. +1 for
     every step it stays up, up to steps. view 'state': the cart's position and speed, the pole's angle and how
     fast it turns, scaled to about ±1. view 'readings': only the position and the angle, but the last 4 of them
     (a sequence of 4 × 2): the speeds are not given, so the model has to work them out from how the readings
     change. track (m) and fall (degrees) change the track's length and the lean it falls at; the position and
     the angle it sees are scaled by them. */
  class Pole {
    constructor(o = {}) {
      // start: how far from upright and still each episode may begin (xStart: the cart, from the middle)
      this.view = o.view === 'readings' ? 'readings' : 'state'; this.start = o.start || 0.05; this.xStart = o.xStart || this.start;
      this.maxSteps = o.steps > 0 ? Math.min(o.steps, 5000) : 200; this.nA = 2; this.actions = ['push left', 'push right'];
      // the physics: a push of force N, a pole of length m (0.1 kg) on a 1 kg cart, gravity in m/s²
      this.force = o.force > 0 ? o.force : 10; this.len = o.length > 0 ? o.length : 1; this.g = o.gravity > 0 ? o.gravity : 9.8;
      this.half = this.len / 2; this.pml = 0.1 * this.half;
      // the ends of the track (±2.4 m) and the lean it falls past (12°, as 0.2095 rad)
      this.edge = o.track > 0 && o.track !== 4.8 ? Math.min(o.track, 20) / 2 : 2.4; this.limit = o.fall > 0 && o.fall !== 12 ? (Math.min(o.fall, 60) * Math.PI) / 180 : 0.2095;
      if (this.view === 'readings') { this.input = { kind: 'series', len: 4, dim: 2 }; this.obsSize = 8; this.obsNames = [3, 2, 1, 0].flatMap((k) => [`cart position ${k ? k + ' steps ago' : 'now'}`, `pole angle ${k ? k + ' steps ago' : 'now'}`]); }
      else { this.input = { kind: 'vector', dim: 4 }; this.obsSize = 4; this.obsNames = ['cart position', 'cart speed', 'pole angle', 'pole turning speed']; }
      this.rw = rewardTable('pole', o); this.rf = E.rewardFormula('pole', o.rFormula); sensors(this, 'pole', o);
    }
    reset(rng) {
      const s = this.start;
      this.rng = rng;
      this.x = rng.uniform(-this.xStart, this.xStart); this.v = rng.uniform(-s, s); this.th = rng.uniform(-s, s); this.w = rng.uniform(-s, s); this.t = 0; this.a = null; this.sx = 0; this.sth = 0; this.pushed = 0;
      this.gNow = this.g * (0.35 + 0.65 * (this.level ?? 1)); // (a curriculum's level: weaker gravity at first)
      this.hist = [0, 0, 0].map(() => [this.x / this.edge, this.th / this.limit]);
      return this.obs();
    }
    rawObs() {
      if (this.view === 'state') return Float32Array.of(this.x / this.edge, this.v / 3, this.th / this.limit, this.w / 3.5);
      const o = new Float32Array(8);
      [...this.hist, [this.x / this.edge, this.th / this.limit]].forEach((r, k) => o.set(r, 2 * k));
      return o;
    }
    step(a) {
      // the equations of motion, one 0.02 s Euler step (by default gravity 9.8, cart 1 kg, pole 0.1 kg and 1 m, push 10 N)
      // a = 1 pushes right, 0 left; anything else (you, trying it yourself, not pushing) not at all
      const F = a === 1 ? this.force : a === 0 ? -this.force : 0, ct = Math.cos(this.th), st = Math.sin(this.th);
      const tmp = (F + this.pml * this.w * this.w * st) / 1.1;
      const acc = ((this.gNow ?? this.g) * st - ct * tmp) / (this.half * (4 / 3 - (0.1 * ct * ct) / 1.1)), xacc = tmp - (this.pml * acc * ct) / 1.1;
      this.hist.push([this.x / this.edge, this.th / this.limit]); this.hist.shift();
      this.x += 0.02 * this.v; this.v += 0.02 * xacc; this.th += 0.02 * this.w; this.w += 0.02 * acc;
      this.t++; this.a = a; this.sx += this.x; this.sth += Math.abs(this.th);
      const fell = Math.abs(this.x) > this.edge || Math.abs(this.th) > this.limit, deg = 180 / Math.PI, rw = this.rw;
      // a push the other way from the last one, and lasting to the end without falling
      const push = a === 1 ? 1 : a === 0 ? -1 : 0, switched = push !== 0 && this.pushed !== 0 && push !== this.pushed, lasted = !fell && this.t >= this.maxSteps;
      if (push) this.pushed = push;
      const parts = this.rf ? null : {};
      let r = 0;
      if (this.rf) r = this.rf({ x: this.x, v: this.v, angle: this.th * deg, spin: this.w * deg, fell: +fell, push, t: this.t, switched: +switched, lasted: +lasted, left: this.maxSteps - this.t });
      else {
        r = pay(parts, 'rUp', rw.rUp, r); if (fell) r = pay(parts, 'rFell', rw.rFell, r);
        if (rw.rLean) r = pay(parts, 'rLean', rw.rLean * Math.abs(this.th * deg), r);
        if (rw.rOff) r = pay(parts, 'rOff', rw.rOff * Math.abs(this.x), r);
        if (switched) r = pay(parts, 'rSwitch', rw.rSwitch, r);
        if (lasted) r = pay(parts, 'rLasted', rw.rLasted, r);
      }
      return { obs: this.obs(), r, done: fell || this.t >= this.maxSteps, fell, parts };
    }
    snap() { return { x: this.x, th: this.th, a: this.a, t: this.t, len: this.len, edge: this.edge, limit: this.limit }; }
    // where the cart keeps (left to right) and how much the pole sways: its style, for MAP-Elites
    behavior() { const n = Math.max(1, this.t); return [clamp(0.5 + this.sx / n / (2 * this.edge), 0, 1), clamp(Math.sqrt(this.sth / n / this.limit), 0, 1)]; }
    // how long its best moves keep the pole up, as a share of the episode, over 5 starts
    score(policy) { const r = new NA.RNG(4242); let s = 0; for (let k = 0; k < 5; k++) { let o = this.reset(r), done = false; while (!done) ({ obs: o, done } = this.step(policy(o))); s += this.t; } return s / (5 * this.maxSteps); }
  }

  // ------------------------------------------------------------------ catch
  /* A ball falls down a 5 × 8 board, one row a step, drifting sideways and bouncing off the walls; a paddle on
     the bottom row moves left, stays or moves right. +1 if the ball lands on the paddle, −1 if it misses. view
     'numbers': where the ball is, which way it drifts, and where the paddle is. view 'pixels': two pictures of the
     board, 8 × 5, now and one step ago (an image of 2 channels), as DQN saw Atari games: which way the ball drifts
     is in the difference. view 'frame': the picture of now alone, from which nobody can tell the drift. width,
     height and paddle change the board (an episode is as many steps as the board is high, less one) and the
     paddle's width in squares; what it sees is scaled to the board. */
  class Catch {
    constructor(o = {}) {
      this.W = clamp(Math.round(o.width) || 5, 3, 15); this.H = clamp(Math.round(o.height) || 8, 4, 20); this.pw = clamp(Math.round(o.paddle) || 1, 1, this.W - 1);
      this.drift = o.drift !== false; this.nA = 3; this.maxSteps = this.H - 1;
      this.view = ['pixels', 'frame'].includes(o.view) ? o.view : 'numbers';
      const ch = this.view === 'pixels' ? 2 : 1, n = this.W * this.H, at = (i) => `(${i % this.W}, ${Math.floor(i / this.W)})`;
      if (this.view === 'numbers') { this.input = { kind: 'vector', dim: 4 }; this.obsSize = 4; this.obsNames = ['ball across', 'ball down', 'ball drift', 'paddle across']; }
      else { this.input = { kind: 'image', c: ch, h: this.H, w: this.W }; this.obsSize = ch * n; this.obsNames = NA.range(ch * n).map((i) => `${i < n ? 'now' : 'a step ago'}: pixel ${at(i % n)}`); }
      this.actions = ['left', 'stay', 'right'];
      this.rw = rewardTable('catch', o); this.rf = E.rewardFormula('catch', o.rFormula); sensors(this, 'catch', o);
    }
    reset(rng) { this.rng = rng; this.bx = rng.int(this.W); this.by = 0; this.vx = this.drift ? rng.int(3) - 1 : 0; this.px = Math.floor((this.W - this.pw) / 2); this.t = 0; this.caught = null; this.prev = null; this.moves = 0; return this.obs(); }
    picture(o, at, bx, by, px) { const W = this.W; o[at + by * W + bx] = 1; for (let k = 0; k < this.pw; k++) o[at + (this.H - 1) * W + px + k] = 1; }
    // the paddle's middle, across; whether a ball across at x is over the paddle
    mid() { return this.px + (this.pw - 1) / 2; }
    over(x) { return x >= this.px && x < this.px + this.pw; }
    // where the ball will land, across: its drift and its bounces played forward
    landing() { let x = this.bx, v = this.vx; for (let y = this.by; y < this.H - 1; y++) { x += v; if (x < 0) { x = 1; v = 1; } else if (x > this.W - 1) { x = this.W - 2; v = -1; } } return x; }
    rawObs() {
      if (this.view === 'numbers') return Float32Array.of(this.bx / (this.W - 1), this.by / (this.H - 1), this.vx, this.mid() / (this.W - 1));
      const n = this.W * this.H, o = new Float32Array((this.view === 'pixels' ? 2 : 1) * n);
      this.picture(o, 0, this.bx, this.by, this.px);
      if (this.view === 'pixels') { const p = this.prev || [this.bx, this.by, this.px]; this.picture(o, n, p[0], p[1], p[2]); }
      return o;
    }
    step(a) {
      this.prev = [this.bx, this.by, this.px];
      this.px = clamp(this.px + a - 1, 0, this.W - this.pw);
      this.by++; this.bx += this.vx;
      // a wall turns it back: it bounces off, one square in from the edge
      if (this.bx < 0) { this.bx = 1; this.vx = 1; } else if (this.bx > this.W - 1) { this.bx = this.W - 2; this.vx = -1; }
      this.t++;
      const moved = this.px !== this.prev[2], end = this.by >= this.H - 1;
      if (moved) this.moves++;
      if (end) this.caught = this.over(this.bx);
      // a near miss: landing on a square right beside the paddle
      const under = this.over(this.bx), near = end && !this.caught && (this.bx === this.px - 1 || this.bx === this.px + this.pw), rw = this.rw, parts = this.rf ? null : {};
      let r = 0;
      if (this.rf) r = this.rf({ caught: +(end && this.caught), missed: +(end && !this.caught), moved: +moved, dx: this.bx - this.mid(), down: this.by, t: this.t, near: +near, under: +under, bx: this.bx, px: this.mid(), land: this.landing(), drift: this.vx });
      else {
        if (moved) r = pay(parts, 'rMove', rw.rMove, r);
        if (end) r = this.caught ? pay(parts, 'rCatch', rw.rCatch, r) : pay(parts, 'rMiss', rw.rMiss, r);
        if (near) r = pay(parts, 'rNear', rw.rNear, r);
        if (under) r = pay(parts, 'rUnder', rw.rUnder, r);
        r = pay(parts, 'rStep', rw.rStep, r);
      }
      return { obs: this.obs(), r, done: end, parts };
    }
    snap() { return { bx: this.bx, by: this.by, px: this.px, caught: this.caught, W: this.W, H: this.H, pw: this.pw }; }
    // how much the paddle moves and where it ends up: its style, for MAP-Elites
    behavior() { return [clamp(this.moves / Math.max(1, this.t), 0, 1), this.px / (this.W - 1)]; }
    // the share of 40 balls its best moves catch (the same 40 every time)
    score(policy) { const r = new NA.RNG(4242); let ok = 0; for (let k = 0; k < 40; k++) { let o = this.reset(r), done = false; while (!done) ({ obs: o, done } = this.step(policy(o))); if (this.caught) ok++; } return ok / 40; }
  }

  for (const W of [Levers, MazeWorld, Pole, Catch]) W.prototype.obs = sensed;

  // the extra inputs each of these can be given (a pull of a lever is its whole episode: no last move, no time left)
  const [, , RANDOM] = E.COMMON_EXTRAS;
  Object.assign(E.EXTRAS, {
    levers: [RANDOM],
    maze: [...E.COMMON_EXTRAS,
      { name: 'where it is', what: 'its square across and down, 0 to 1', n: 2, names: ['its square across', 'its square down'], fill: (w, o, k) => { const M = w.maze; o[k] = ((w.s % M.W) + 0.5) / M.W; o[k + 1] = (Math.floor(w.s / M.W) + 0.5) / M.H; } },
      { name: 'the goal from here', what: 'how far across and down the goal is from it, as a share of the maze', n: 2, names: ['the goal across from here', 'the goal down from here'], fill: (w, o, k) => { const M = w.maze; o[k] = ((M.goal % M.W) - (w.s % M.W)) / M.W; o[k + 1] = (Math.floor(M.goal / M.W) - Math.floor(w.s / M.W)) / M.H; } },
      { name: 'safe steps to the goal', what: 'the fewest steps from here to the goal that pass no pit, a twentieth each (1 when there is no safe way)', n: 1, names: ['safe steps to the goal'], fill: (w, o, k) => { const d = w.dist[w.s]; o[k] = d >= 0 ? d / 20 : 1; } }],
    pole: E.COMMON_EXTRAS.slice(),
    catch: [...E.COMMON_EXTRAS,
      { name: 'the ball from the paddle', what: 'the ball across minus the paddle across, as a share of the board: 0 when it is right above', n: 1, names: ['the ball from the paddle'], fill: (w, o, k) => { o[k] = (w.bx - w.mid()) / (w.W - 1); } },
      { name: 'where it will land', what: 'the square the ball will land on, across, worked out from its drift and the bounces: the answer itself', n: 1, names: ['where it will land'], fill: (w, o, k) => { o[k] = w.landing() / (w.W - 1); } }],
  });

  // ------------------------------------------------------------------ the catalogue
  E.WORLDS = {
    levers: { name: 'Levers', make: (o) => new Levers(o), what: 'a light, one input per colour', goal: 'pull the lever that matches the light', demoMs: 600 },
    maze: { name: 'Maze', make: (o) => new MazeWorld(o), what: 'its square, what is around it, or a picture of the maze', goal: 'reach the goal, keep out of the pits', axes: ['where it stops, across', 'where it stops, up'] },
    pole: { name: 'Balance a pole', make: (o) => new Pole(o), what: 'the cart and the pole: 4 numbers, or the last 4 readings', goal: 'keep the pole upright', axes: ['where the cart keeps', 'how much the pole sways'] },
    catch: { name: 'Catch', make: (o) => new Catch(o), what: 'the ball and the paddle: 4 numbers, or pixels', goal: 'catch the falling ball', axes: ['how much it moves', 'where it ends up'] },
  };
  E.make = (id, o) => E.WORLDS[id].make(o || {});
  // observations from random play: example inputs for a model (its zoom view opens on one)
  E.samples = function (world, n, rng) {
    const X = new Float32Array(n * world.obsSize);
    let o = world.reset(rng), k = 0;
    while (k < n) { X.set(o, k * world.obsSize); k++; const t = world.step(rng.int(world.nA), rng); o = t.done ? world.reset(rng) : t.obs; }
    return X;
  };

  // ------------------------------------------------------------------ pictures
  /* draw(ctx, r, id, snap, T): a world's state from snap() inside r. T is the theme (NA.theme). Worlds defined
     elsewhere (rl-worlds.js) add their own pictures to E.DRAW. */
  E.draw = function (ctx, r, id, s, T, o = {}) { return (E.DRAW[id] || drawCatch)(ctx, r, s, T, o); };
  function drawLevers(ctx, r, s, T, o) {
    const k = o.k || 3, cols = ['s1', 's2', 's4', 's7', 's3'], cw = Math.min(r.w / k, 70), x0 = r.x + (r.w - cw * k) / 2, cy = r.y + r.h * 0.3, rad = Math.min(cw * 0.3, r.h * 0.18);
    ctx.beginPath(); ctx.arc(r.x + r.w / 2, r.y + r.h * 0.12 + rad * 0.5, rad * 0.8, 0, Math.PI * 2); ctx.fillStyle = T.c(cols[s.light]); ctx.fill();
    for (let i = 0; i < k; i++) {
      const x = x0 + (i + 0.5) * cw, on = s.last && s.last.a === i;
      ctx.strokeStyle = T.c(cols[i]); ctx.lineWidth = 3; ctx.beginPath(); ctx.moveTo(x, cy + rad * 2.8); ctx.lineTo(x + (on ? rad : -rad * 0.4), cy + rad * 0.6); ctx.stroke();
      ctx.beginPath(); ctx.arc(x + (on ? rad : -rad * 0.4), cy + rad * 0.6, rad * 0.35, 0, Math.PI * 2); ctx.fillStyle = T.c(cols[i]); ctx.fill();
      ctx.fillStyle = T.c('line-2'); ctx.fillRect(x - rad * 0.8, cy + rad * 2.8, rad * 1.6, rad * 0.5);
      if (on) { const r = s.last.r; ctx.fillStyle = T.c(r > 0 ? 'pos' : r < 0 ? 'neg' : 'muted'); ctx.font = `700 ${Math.max(11, rad * 0.6)}px sans-serif`; ctx.textAlign = 'center'; ctx.fillText(r > 0 ? '+' + NA.fmt(r, r % 1 ? 2 : 0) : NA.fmt(r, r % 1 ? 2 : 0), x, cy + rad * 4.4); }
    }
  }
  // the maze as the RL lab draws it: the goal (the exit) +1, pits −1 (or the amounts its reward table pays there; none
  // with a formula of your own), S where every episode starts
  const amount = (v) => (v > 0 ? '+' : v < 0 ? '−' : '') + String(+Math.abs(v).toPrecision(3));
  function drawMaze(ctx, r, s, T) {
    const M = new (RL().Maze)(s.rows), cs = Math.min(r.w / M.W, r.h / M.H), x0 = r.x + (r.w - cs * M.W) / 2, y0 = r.y + (r.h - cs * M.H) / 2;
    const fam = (typeof getComputedStyle === 'function' && getComputedStyle(document.body).getPropertyValue('--font-ui')) || 'sans-serif';
    const say = (t, x, y, size, tok, w = 700) => { ctx.font = `${w} ${size}px ${fam}`; ctx.fillStyle = T.c(tok); ctx.textAlign = 'center'; ctx.textBaseline = 'middle'; ctx.fillText(t, x, y); };
    const pay = s.pay === undefined ? { goal: 1, pit: -1 } : s.pay, atGoal = pay ? amount(pay.goal) : '', inPit = pay ? amount(pay.pit) : '';
    for (let i = 0; i < M.n; i++) {
      const c = M.cell[i], x = x0 + (i % M.W) * cs, y = y0 + Math.floor(i / M.W) * cs;
      ctx.fillStyle = c === '#' ? T.c('line-2') : c === 'G' ? T.c('pos', 0.45) : c === 'X' ? T.c('neg', 0.42) : T.c('panel'); ctx.fillRect(x, y, cs, cs);
      ctx.strokeStyle = T.c('line'); ctx.lineWidth = 1; ctx.strokeRect(x + 0.5, y + 0.5, cs - 1, cs - 1);
      if (c === 'G') {
        ctx.strokeStyle = T.c('pos'); ctx.lineWidth = Math.max(2, cs * 0.05); ctx.strokeRect(x + 2, y + 2, cs - 4, cs - 4);
        if (cs >= 34) { if (atGoal) say(atGoal, x + cs / 2, y + cs * 0.4, Math.max(10, cs * 0.26), 'pos'); say('goal', x + cs / 2, y + cs * (atGoal ? 0.72 : 0.5), Math.max(8.5, cs * 0.17), 'ink-2', 600); }
        else if (atGoal) say(atGoal, x + cs / 2, y + cs / 2, Math.max(8, cs * 0.34), 'pos');
      } else if (c === 'X' && inPit) say(inPit, x + cs / 2, y + cs / 2, Math.max(8, cs * (cs >= 34 ? 0.26 : 0.34)), 'neg');
      else if (i === M.start && cs >= 20) say('S', x + cs * 0.2, y + cs * 0.22, Math.max(8, cs * 0.2), 'ink-2', 600);
    }
    if (s.path && s.path.length > 1) { ctx.strokeStyle = T.c('s7', 0.85); ctx.lineWidth = Math.max(2, cs * 0.07); ctx.lineJoin = 'round'; ctx.beginPath(); s.path.forEach((q, k) => { const x = x0 + (q % M.W + 0.5) * cs, y = y0 + (Math.floor(q / M.W) + 0.5) * cs; k ? ctx.lineTo(x, y) : ctx.moveTo(x, y); }); ctx.stroke(); }
    ctx.fillStyle = T.c('s7'); ctx.beginPath(); ctx.arc(x0 + (s.s % M.W + 0.5) * cs, y0 + (Math.floor(s.s / M.W) + 0.5) * cs, cs * 0.22, 0, Math.PI * 2); ctx.fill();
  }
  function drawPole(ctx, r, s, T) {
    const E2 = s.edge || 2.4, gy = r.y + r.h * 0.78, sc = r.w / (2 * E2 + 0.8), cx = r.x + r.w / 2 + clamp(s.x, -E2 - 0.2, E2 + 0.2) * sc, cw = Math.max(26, sc * 0.5), ch = cw * 0.5, L = Math.min(r.h * 0.6, sc * 1.6) * Math.sqrt(clamp(s.len || 1, 0.25, 4));
    ctx.strokeStyle = T.c('line-2'); ctx.lineWidth = 2; ctx.beginPath(); ctx.moveTo(r.x + r.w / 2 - E2 * sc, gy); ctx.lineTo(r.x + r.w / 2 + E2 * sc, gy); ctx.stroke();
    for (const e of [-E2, E2]) { ctx.beginPath(); ctx.moveTo(r.x + r.w / 2 + e * sc, gy - 8); ctx.lineTo(r.x + r.w / 2 + e * sc, gy + 8); ctx.stroke(); }
    ctx.fillStyle = T.c('ink-2'); ctx.fillRect(cx - cw / 2, gy - ch, cw, ch);
    const tx = cx + Math.sin(s.th) * L, ty = gy - ch - Math.cos(s.th) * L, fallen = Math.abs(s.th) > (s.limit || 0.2095);
    ctx.strokeStyle = T.c(fallen ? 'neg' : 's1'); ctx.lineWidth = Math.max(4, cw * 0.18); ctx.lineCap = 'round'; ctx.beginPath(); ctx.moveTo(cx, gy - ch); ctx.lineTo(tx, ty); ctx.stroke(); ctx.lineCap = 'butt';
    if (s.a !== null && s.a !== undefined) { const d = s.a === 1 ? 1 : -1; ctx.fillStyle = T.c('s7'); ctx.beginPath(); ctx.moveTo(cx - d * (cw / 2 + 4), gy - ch / 2); ctx.lineTo(cx - d * (cw / 2 + 18), gy - ch / 2 - 7); ctx.lineTo(cx - d * (cw / 2 + 18), gy - ch / 2 + 7); ctx.closePath(); ctx.fill(); }
  }
  function drawCatch(ctx, r, s, T) {
    const W = s.W || 5, H = s.H || 8, pw = s.pw || 1, cs = Math.min(r.w / W, r.h / H), x0 = r.x + (r.w - cs * W) / 2, y0 = r.y + (r.h - cs * H) / 2;
    ctx.strokeStyle = T.c('line'); ctx.lineWidth = 1;
    for (let i = 0; i < W; i++) for (let j = 0; j < H; j++) ctx.strokeRect(x0 + i * cs + 0.5, y0 + j * cs + 0.5, cs - 1, cs - 1);
    ctx.fillStyle = T.c(s.caught === false ? 'neg' : 's2'); ctx.beginPath(); ctx.arc(x0 + (s.bx + 0.5) * cs, y0 + (s.by + 0.5) * cs, cs * 0.3, 0, Math.PI * 2); ctx.fill();
    ctx.fillStyle = T.c(s.caught ? 'pos' : 's1'); ctx.fillRect(x0 + s.px * cs + cs * 0.08, y0 + (H - 1) * cs + cs * 0.62, cs * (pw - 0.16), cs * 0.26);
  }

  E.DRAW = { levers: drawLevers, maze: drawMaze, pole: drawPole, catch: drawCatch };
  // for the worlds of rl-worlds.js: reward tables (and their parts), inputs left out or added and noise, the same way as here
  Object.assign(E, { Levers, MazeWorld, Pole, Catch, rewardTable, pay, sensors, sensed });
  NA.rlEnvs = E;
})(typeof globalThis !== 'undefined' && globalThis.NA ? globalThis.NA : window.NA);
