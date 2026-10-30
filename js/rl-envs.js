/* rl-envs.js — small worlds to learn in from rewards, shared by the RL lab and the model builder: levers that pay
   when they match a light (a contextual bandit), a maze (seen as the square it is on, or as what is around it), a
   pole to keep upright on a moving cart, and a falling ball to catch with a paddle. Each gives an observation,
   takes one of a few actions, and answers with a reward and the next observation until the episode ends. Most can
   be seen more than one way (view): a few numbers, a picture, or the last few readings; input says which, in the
   model builder's terms (a vector, an image of channels × height × width, or a sequence of positions × numbers).
   The model builder can change how each one pays (a reward table, or a formula of its own), leave inputs out,
   add sensor noise, and for the pole its push, length and gravity. Everything is seeded, so a run can be repeated
   exactly. Needs rl.js (the mazes); a reward formula needs builder-tasks.js (the formula language). */
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
    levers: [{ key: 'rPay', label: 'When a lever pays', def: 1 }],
    maze: [{ key: 'rGoal', label: 'Reaching the exit', def: 1 }, { key: 'rPit', label: 'Falling in a pit', def: -1 }, { key: 'rStep', label: 'Every other step', def: -0.02 },
      { key: 'rWall', label: 'Walking into a wall', def: 0 }, { key: 'rNew', label: 'A square not yet visited', def: 0 }, { key: 'rCloser', label: 'A step closer (with the closer bonus on)', def: 0.1 }],
    pole: [{ key: 'rUp', label: 'Every step it stays up', def: 1 }, { key: 'rFell', label: 'When it falls', def: 0 }],
    catch: [{ key: 'rCatch', label: 'Catching the ball', def: 1 }, { key: 'rMiss', label: 'Missing it', def: -1 }, { key: 'rMove', label: 'Moving the paddle', def: 0 }],
  };
  E.REWARD_VARS = {
    levers: [['paid', '1 when the lever pays out'], ['right', '1 when it was the lever of the light’s colour']],
    maze: [['goal', '1 on reaching the exit'], ['pit', '1 on falling in a pit'], ['wall', '1 on walking into a wall (it stays put)'], ['closer', '1 for a step closer to the exit by the shortest safe way, −1 for a step away'],
      ['new', '1 on a square it has not been on this episode'], ['dist', 'safe steps from here to the exit'], ['t', 'steps so far']],
    pole: [['x', 'the cart’s position: −2.4 and 2.4 are the ends of the track'], ['v', 'the cart’s speed'], ['angle', 'the pole’s lean in degrees: it falls past ±12'],
      ['spin', 'how fast the pole turns, degrees a second'], ['fell', '1 on the step it falls'], ['push', '1 for a push right, −1 left'], ['t', 'steps so far']],
    catch: [['caught', '1 when the ball lands on the paddle'], ['missed', '1 when it lands beside it'], ['moved', '1 when the paddle moved'],
      ['dx', 'the ball across minus the paddle across, in squares'], ['down', 'the ball’s row, 0 at the top to 7'], ['t', 'steps so far']],
  };
  const rewardTable = (id, o) => Object.fromEntries(E.REWARDS[id].map((r) => [r.key, typeof o[r.key] === 'number' && Number.isFinite(o[r.key]) ? o[r.key] : r.def]));
  // your own reward, compiled once; null (the table) when there is none or it does not compile
  E.rewardFormula = (id, src) => {
    const C = NA.builderTasks && NA.builderTasks.compile;
    if (!src || !C) return null;
    const c = C(String(src).slice(0, 200), E.REWARD_VARS[id].map((v) => v[0]));
    return c.error ? null : c.f;
  };
  /* Inputs that can be left out, by name, in the views that are a list of numbers. A model that is not told the
     speeds, say, has to do without them (or work them out, given a memory). */
  const span = (a, b) => NA.range(b - a).map((i) => a + i);
  E.INPUT_GROUPS = {
    pole: { state: [['cart position', [0]], ['cart speed', [1]], ['pole angle', [2]], ['pole turning speed', [3]]] },
    catch: { numbers: [['ball across', [0]], ['ball down', [1]], ['ball drift', [2]], ['paddle across', [3]]] },
    maze: { around: [['walls', span(0, 24)], ['pits', span(24, 48)], ['the exit', span(48, 72)], ['where it has been', span(72, 96)], ['which way the exit is', span(96, 104)]] },
  };
  // the inputs a world keeps (hide: names of groups to leave out) and the noise on each (noise: its spread)
  function sensors(w, id, o) {
    const groups = (E.INPUT_GROUPS[id] || {})[w.view], hide = Array.isArray(o.hide) ? o.hide : [];
    w.noise = typeof o.noise === 'number' && o.noise > 0 ? Math.min(o.noise, 2) : 0;
    w.hidden = [];
    if (!groups || !hide.length) return;
    const drop = new Set(groups.filter(([name]) => hide.includes(name)).flatMap(([, idx]) => idx));
    if (!drop.size || drop.size >= w.obsSize) return; // at least one input stays
    w.hidden = groups.filter(([name]) => hide.includes(name)).map(([name]) => name);
    w.keep = NA.range(w.obsSize).filter((i) => !drop.has(i));
    w.obsNames = w.keep.map((i) => w.obsNames[i]);
    w.obsSize = w.keep.length;
    w.input = { kind: 'vector', dim: w.obsSize };
  }
  // what the model is given: the world's numbers, those it keeps, plus noise
  function sensed() {
    const x = this.rawObs();
    if (!this.keep && !this.noise) return x;
    const y = this.keep ? Float32Array.from(this.keep, (i) => x[i]) : x;
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
      const right = a === this.light, paid = rng.next() < (right ? this.p : this.q);
      const r = this.rf ? this.rf({ paid: +paid, right: +right }) : paid ? this.rw.rPay : 0;
      this.t++; this.last = { a, r, paid };
      return { obs: this.obs(), r, done: true };
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
  class MazeWorld {
    constructor(o = {}) {
      this.view = ['square', 'picture'].includes(o.view) ? o.view : 'around'; this.random = !!o.random; this.slip = o.slip || 0; this.closer = !!o.closer;
      const def = RL().MAZE_SET.find((m) => m.id === o.maze) || RL().MAZE_SET[0];
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
    }
    // the fewest safe steps from every square to the goal (-1: none)
    static distances(M) {
      const d = new Int16Array(M.n).fill(-1), q = [M.goal];
      d[M.goal] = 0;
      while (q.length) { const s = q.shift(); for (let a = 0; a < 4; a++) { const t = M.move(s, a); if (d[t] >= 0 || M.cell[t] === 'X' || M.cell[t] === '#') continue; d[t] = d[s] + 1; q.push(t); } }
      return d;
    }
    reset(rng) {
      this.rng = rng;
      if (this.random) { this.maze = new (RL().Maze)(RL().randomMaze(() => rng.next())); this.maze.slip = this.slip; if (this.closer || this.rf) this.dist = MazeWorld.distances(this.maze); }
      else if (this.maze !== this.fixed) { this.maze = this.fixed; this.dist = MazeWorld.distances(this.fixed); }
      const M = this.maze;
      this.s = M.start; this.seen = new Uint8Array(M.n); this.seen[this.s] = 1; this.t = 0; this.path = [this.s]; this.end = null;
      return this.obs();
    }
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
      const c = this.maze.cell[t.s], goal = c === 'G', pit = c === 'X', wall = t.s === s0, d0 = this.dist[s0], d1 = this.dist[t.s];
      const closer = !pit && d0 >= 0 && d1 >= 0 ? d0 - d1 : 0, rw = this.rw;
      let r;
      if (this.rf) r = this.rf({ goal: +goal, pit: +pit, wall: +wall, closer, new: +(fresh && !goal && !pit), dist: Math.max(0, d1), t: this.t });
      else {
        r = goal ? rw.rGoal : pit ? rw.rPit : rw.rStep;
        if (wall) r += rw.rWall;
        if (fresh && !goal && !pit) r += rw.rNew;
        if (this.closer) r += rw.rCloser * closer;
      }
      return { obs: this.obs(), r, done };
    }
    snap() { return { rows: this.maze.rows, s: this.s, path: this.path.slice(), end: this.end }; }
    // where its moves stop (across, and up from the bottom row): its style of play, for MAP-Elites
    behavior() { const W = this.maze.W, H = this.maze.H; return [((this.s % W) + 0.5) / W, 1 - (Math.floor(this.s / W) + 0.5) / H]; }
    /* Its best moves from S: 1 if they reach the goal (a slippery floor: the share of 5 tries). Trained on random
       mazes, the share of 20 new mazes it reaches the goal of (the same 20 every time, none of them trained on). */
    score(policy, rng) {
      const random = this.random, keep = this.maze, list = random ? RL().testMazes().slice(0, 20) : NA.range(this.slip ? 5 : 1).map(() => this.fixed);
      let ok = 0;
      this.random = false;
      for (const m of list) {
        this.maze = m; this.dist = MazeWorld.distances(m);
        this.s = m.start; this.seen = new Uint8Array(m.n); this.seen[this.s] = 1; this.t = 0; this.path = [this.s]; this.end = null;
        let o = this.obs(), done = false;
        while (!done) ({ obs: o, done } = this.step(policy(o), rng));
        if (this.end === 'goal') ok++;
      }
      this.random = random; this.maze = keep; this.dist = MazeWorld.distances(keep);
      return ok / list.length;
    }
  }

  // ------------------------------------------------------------------ a pole on a cart
  /* The classic balancing problem (Barto, Sutton and Anderson, 1983): a pole hinged on a cart that can only be
     pushed left or right. It falls if it leans more than 12°, or the cart runs off its 4.8-unit track. +1 for
     every step it stays up, up to steps. view 'state': the cart's position and speed, the pole's angle and how
     fast it turns, scaled to about ±1. view 'readings': only the position and the angle, but the last 4 of them
     (a sequence of 4 × 2): the speeds are not given, so the model has to work them out from how the readings
     change. */
  class Pole {
    constructor(o = {}) {
      // start: how far from upright and still each episode may begin (xStart: the cart, from the middle)
      this.view = o.view === 'readings' ? 'readings' : 'state'; this.start = o.start || 0.05; this.xStart = o.xStart || this.start;
      this.maxSteps = o.steps > 0 ? Math.min(o.steps, 5000) : 200; this.nA = 2; this.actions = ['push left', 'push right'];
      // the physics: a push of force N, a pole of length m (0.1 kg) on a 1 kg cart, gravity in m/s²
      this.force = o.force > 0 ? o.force : 10; this.len = o.length > 0 ? o.length : 1; this.g = o.gravity > 0 ? o.gravity : 9.8;
      this.half = this.len / 2; this.pml = 0.1 * this.half;
      if (this.view === 'readings') { this.input = { kind: 'series', len: 4, dim: 2 }; this.obsSize = 8; this.obsNames = [3, 2, 1, 0].flatMap((k) => [`cart position ${k ? k + ' steps ago' : 'now'}`, `pole angle ${k ? k + ' steps ago' : 'now'}`]); }
      else { this.input = { kind: 'vector', dim: 4 }; this.obsSize = 4; this.obsNames = ['cart position', 'cart speed', 'pole angle', 'pole turning speed']; }
      this.rw = rewardTable('pole', o); this.rf = E.rewardFormula('pole', o.rFormula); sensors(this, 'pole', o);
    }
    reset(rng) {
      const s = this.start;
      this.rng = rng;
      this.x = rng.uniform(-this.xStart, this.xStart); this.v = rng.uniform(-s, s); this.th = rng.uniform(-s, s); this.w = rng.uniform(-s, s); this.t = 0; this.a = null; this.sx = 0; this.sth = 0;
      this.hist = [0, 0, 0].map(() => [this.x / 2.4, this.th / 0.2095]);
      return this.obs();
    }
    rawObs() {
      if (this.view === 'state') return Float32Array.of(this.x / 2.4, this.v / 3, this.th / 0.2095, this.w / 3.5);
      const o = new Float32Array(8);
      [...this.hist, [this.x / 2.4, this.th / 0.2095]].forEach((r, k) => o.set(r, 2 * k));
      return o;
    }
    step(a) {
      // the equations of motion, one 0.02 s Euler step (by default gravity 9.8, cart 1 kg, pole 0.1 kg and 1 m, push 10 N)
      // a = 1 pushes right, 0 left; anything else (you, trying it yourself, not pushing) not at all
      const F = a === 1 ? this.force : a === 0 ? -this.force : 0, ct = Math.cos(this.th), st = Math.sin(this.th);
      const tmp = (F + this.pml * this.w * this.w * st) / 1.1;
      const acc = (this.g * st - ct * tmp) / (this.half * (4 / 3 - (0.1 * ct * ct) / 1.1)), xacc = tmp - (this.pml * acc * ct) / 1.1;
      this.hist.push([this.x / 2.4, this.th / 0.2095]); this.hist.shift();
      this.x += 0.02 * this.v; this.v += 0.02 * xacc; this.th += 0.02 * this.w; this.w += 0.02 * acc;
      this.t++; this.a = a; this.sx += this.x; this.sth += Math.abs(this.th);
      const fell = Math.abs(this.x) > 2.4 || Math.abs(this.th) > 0.2095, deg = 180 / Math.PI;
      const r = this.rf ? this.rf({ x: this.x, v: this.v, angle: this.th * deg, spin: this.w * deg, fell: +fell, push: a === 1 ? 1 : a === 0 ? -1 : 0, t: this.t }) : this.rw.rUp + (fell ? this.rw.rFell : 0);
      return { obs: this.obs(), r, done: fell || this.t >= this.maxSteps, fell };
    }
    snap() { return { x: this.x, th: this.th, a: this.a, t: this.t, len: this.len }; }
    // where the cart keeps (left to right) and how much the pole sways: its style, for MAP-Elites
    behavior() { const n = Math.max(1, this.t); return [clamp(0.5 + this.sx / n / 4.8, 0, 1), clamp(Math.sqrt(this.sth / n / 0.2095), 0, 1)]; }
    // how long its best moves keep the pole up, as a share of the episode, over 5 starts
    score(policy) { const r = new NA.RNG(4242); let s = 0; for (let k = 0; k < 5; k++) { let o = this.reset(r), done = false; while (!done) ({ obs: o, done } = this.step(policy(o))); s += this.t; } return s / (5 * this.maxSteps); }
  }

  // ------------------------------------------------------------------ catch
  /* A ball falls down a 5 × 8 board, one row a step, drifting sideways and bouncing off the walls; a paddle on
     the bottom row moves left, stays or moves right. +1 if the ball lands on the paddle, −1 if it misses. view
     'numbers': where the ball is, which way it drifts, and where the paddle is. view 'pixels': two pictures of the
     board, 8 × 5, now and one step ago (an image of 2 channels), as DQN saw Atari games: which way the ball drifts
     is in the difference. view 'frame': the picture of now alone, from which nobody can tell the drift. */
  class Catch {
    constructor(o = {}) {
      this.W = 5; this.H = 8; this.drift = o.drift !== false; this.nA = 3; this.maxSteps = this.H - 1;
      this.view = ['pixels', 'frame'].includes(o.view) ? o.view : 'numbers';
      const ch = this.view === 'pixels' ? 2 : 1, n = this.W * this.H, at = (i) => `(${i % this.W}, ${Math.floor(i / this.W)})`;
      if (this.view === 'numbers') { this.input = { kind: 'vector', dim: 4 }; this.obsSize = 4; this.obsNames = ['ball across', 'ball down', 'ball drift', 'paddle across']; }
      else { this.input = { kind: 'image', c: ch, h: this.H, w: this.W }; this.obsSize = ch * n; this.obsNames = NA.range(ch * n).map((i) => `${i < n ? 'now' : 'a step ago'}: pixel ${at(i % n)}`); }
      this.actions = ['left', 'stay', 'right'];
      this.rw = rewardTable('catch', o); this.rf = E.rewardFormula('catch', o.rFormula); sensors(this, 'catch', o);
    }
    reset(rng) { this.rng = rng; this.bx = rng.int(this.W); this.by = 0; this.vx = this.drift ? rng.int(3) - 1 : 0; this.px = 2; this.t = 0; this.caught = null; this.prev = null; this.moves = 0; return this.obs(); }
    picture(o, at, bx, by, px) { const W = this.W; o[at + by * W + bx] = 1; o[at + (this.H - 1) * W + px] = 1; }
    rawObs() {
      if (this.view === 'numbers') return Float32Array.of(this.bx / (this.W - 1), this.by / (this.H - 1), this.vx, this.px / (this.W - 1));
      const n = this.W * this.H, o = new Float32Array((this.view === 'pixels' ? 2 : 1) * n);
      this.picture(o, 0, this.bx, this.by, this.px);
      if (this.view === 'pixels') { const p = this.prev || [this.bx, this.by, this.px]; this.picture(o, n, p[0], p[1], p[2]); }
      return o;
    }
    step(a) {
      this.prev = [this.bx, this.by, this.px];
      this.px = clamp(this.px + a - 1, 0, this.W - 1);
      this.by++; this.bx += this.vx;
      // a wall turns it back: it bounces off, one square in from the edge
      if (this.bx < 0) { this.bx = 1; this.vx = 1; } else if (this.bx > this.W - 1) { this.bx = this.W - 2; this.vx = -1; }
      this.t++;
      const moved = this.px !== this.prev[2], end = this.by >= this.H - 1;
      if (moved) this.moves++;
      if (end) this.caught = this.bx === this.px;
      const rw = this.rw, r = this.rf ? this.rf({ caught: +(end && this.caught), missed: +(end && !this.caught), moved: +moved, dx: this.bx - this.px, down: this.by, t: this.t })
        : (moved ? rw.rMove : 0) + (end ? (this.caught ? rw.rCatch : rw.rMiss) : 0);
      return { obs: this.obs(), r, done: end };
    }
    snap() { return { bx: this.bx, by: this.by, px: this.px, caught: this.caught }; }
    // how much the paddle moves and where it ends up: its style, for MAP-Elites
    behavior() { return [clamp(this.moves / Math.max(1, this.t), 0, 1), this.px / (this.W - 1)]; }
    // the share of 40 balls its best moves catch (the same 40 every time)
    score(policy) { const r = new NA.RNG(4242); let ok = 0; for (let k = 0; k < 40; k++) { let o = this.reset(r), done = false; while (!done) ({ obs: o, done } = this.step(policy(o))); if (this.caught) ok++; } return ok / 40; }
  }

  for (const W of [Levers, MazeWorld, Pole, Catch]) W.prototype.obs = sensed;

  // ------------------------------------------------------------------ the catalogue
  E.WORLDS = {
    levers: { name: 'Levers', make: (o) => new Levers(o), what: 'a light, one input per colour', goal: 'pull the lever that matches the light' },
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
  // the maze as the RL lab draws it: the goal (the exit) +1, pits −1, S where every episode starts
  function drawMaze(ctx, r, s, T) {
    const M = new (RL().Maze)(s.rows), cs = Math.min(r.w / M.W, r.h / M.H), x0 = r.x + (r.w - cs * M.W) / 2, y0 = r.y + (r.h - cs * M.H) / 2;
    const fam = (typeof getComputedStyle === 'function' && getComputedStyle(document.body).getPropertyValue('--font-ui')) || 'sans-serif';
    const say = (t, x, y, size, tok, w = 700) => { ctx.font = `${w} ${size}px ${fam}`; ctx.fillStyle = T.c(tok); ctx.textAlign = 'center'; ctx.textBaseline = 'middle'; ctx.fillText(t, x, y); };
    for (let i = 0; i < M.n; i++) {
      const c = M.cell[i], x = x0 + (i % M.W) * cs, y = y0 + Math.floor(i / M.W) * cs;
      ctx.fillStyle = c === '#' ? T.c('line-2') : c === 'G' ? T.c('pos', 0.45) : c === 'X' ? T.c('neg', 0.42) : T.c('panel'); ctx.fillRect(x, y, cs, cs);
      ctx.strokeStyle = T.c('line'); ctx.lineWidth = 1; ctx.strokeRect(x + 0.5, y + 0.5, cs - 1, cs - 1);
      if (c === 'G') {
        ctx.strokeStyle = T.c('pos'); ctx.lineWidth = Math.max(2, cs * 0.05); ctx.strokeRect(x + 2, y + 2, cs - 4, cs - 4);
        if (cs >= 34) { say('+1', x + cs / 2, y + cs * 0.4, Math.max(10, cs * 0.26), 'pos'); say('exit', x + cs / 2, y + cs * 0.72, Math.max(8.5, cs * 0.17), 'ink-2', 600); }
        else say('+1', x + cs / 2, y + cs / 2, Math.max(8, cs * 0.34), 'pos');
      } else if (c === 'X') say('−1', x + cs / 2, y + cs / 2, Math.max(8, cs * (cs >= 34 ? 0.26 : 0.34)), 'neg');
      else if (i === M.start && cs >= 20) say('S', x + cs * 0.2, y + cs * 0.22, Math.max(8, cs * 0.2), 'ink-2', 600);
    }
    if (s.path && s.path.length > 1) { ctx.strokeStyle = T.c('s7', 0.85); ctx.lineWidth = Math.max(2, cs * 0.07); ctx.lineJoin = 'round'; ctx.beginPath(); s.path.forEach((q, k) => { const x = x0 + (q % M.W + 0.5) * cs, y = y0 + (Math.floor(q / M.W) + 0.5) * cs; k ? ctx.lineTo(x, y) : ctx.moveTo(x, y); }); ctx.stroke(); }
    ctx.fillStyle = T.c('s7'); ctx.beginPath(); ctx.arc(x0 + (s.s % M.W + 0.5) * cs, y0 + (Math.floor(s.s / M.W) + 0.5) * cs, cs * 0.22, 0, Math.PI * 2); ctx.fill();
  }
  function drawPole(ctx, r, s, T) {
    const gy = r.y + r.h * 0.78, sc = r.w / 5.6, cx = r.x + r.w / 2 + clamp(s.x, -2.6, 2.6) * sc, cw = Math.max(26, sc * 0.5), ch = cw * 0.5, L = Math.min(r.h * 0.6, sc * 1.6) * Math.sqrt(clamp(s.len || 1, 0.25, 4));
    ctx.strokeStyle = T.c('line-2'); ctx.lineWidth = 2; ctx.beginPath(); ctx.moveTo(r.x + r.w / 2 - 2.4 * sc, gy); ctx.lineTo(r.x + r.w / 2 + 2.4 * sc, gy); ctx.stroke();
    for (const e of [-2.4, 2.4]) { ctx.beginPath(); ctx.moveTo(r.x + r.w / 2 + e * sc, gy - 8); ctx.lineTo(r.x + r.w / 2 + e * sc, gy + 8); ctx.stroke(); }
    ctx.fillStyle = T.c('ink-2'); ctx.fillRect(cx - cw / 2, gy - ch, cw, ch);
    const tx = cx + Math.sin(s.th) * L, ty = gy - ch - Math.cos(s.th) * L, fallen = Math.abs(s.th) > 0.2095;
    ctx.strokeStyle = T.c(fallen ? 'neg' : 's1'); ctx.lineWidth = Math.max(4, cw * 0.18); ctx.lineCap = 'round'; ctx.beginPath(); ctx.moveTo(cx, gy - ch); ctx.lineTo(tx, ty); ctx.stroke(); ctx.lineCap = 'butt';
    if (s.a !== null && s.a !== undefined) { const d = s.a === 1 ? 1 : -1; ctx.fillStyle = T.c('s7'); ctx.beginPath(); ctx.moveTo(cx - d * (cw / 2 + 4), gy - ch / 2); ctx.lineTo(cx - d * (cw / 2 + 18), gy - ch / 2 - 7); ctx.lineTo(cx - d * (cw / 2 + 18), gy - ch / 2 + 7); ctx.closePath(); ctx.fill(); }
  }
  function drawCatch(ctx, r, s, T) {
    const W = 5, H = 8, cs = Math.min(r.w / W, r.h / H), x0 = r.x + (r.w - cs * W) / 2, y0 = r.y + (r.h - cs * H) / 2;
    ctx.strokeStyle = T.c('line'); ctx.lineWidth = 1;
    for (let i = 0; i < W; i++) for (let j = 0; j < H; j++) ctx.strokeRect(x0 + i * cs + 0.5, y0 + j * cs + 0.5, cs - 1, cs - 1);
    ctx.fillStyle = T.c(s.caught === false ? 'neg' : 's2'); ctx.beginPath(); ctx.arc(x0 + (s.bx + 0.5) * cs, y0 + (s.by + 0.5) * cs, cs * 0.3, 0, Math.PI * 2); ctx.fill();
    ctx.fillStyle = T.c(s.caught ? 'pos' : 's1'); ctx.fillRect(x0 + s.px * cs + cs * 0.08, y0 + (H - 1) * cs + cs * 0.62, cs * 0.84, cs * 0.26);
  }

  E.DRAW = { levers: drawLevers, maze: drawMaze, pole: drawPole, catch: drawCatch };
  // for the worlds of rl-worlds.js: reward tables, inputs left out and noise, the same way as here
  Object.assign(E, { Levers, MazeWorld, Pole, Catch, rewardTable, sensors, sensed });
  NA.rlEnvs = E;
})(typeof globalThis !== 'undefined' && globalThis.NA ? globalThis.NA : window.NA);
