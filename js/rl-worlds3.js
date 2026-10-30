/* rl-worlds3.js — the hardest worlds of the model builder's "Learn from rewards": a deceptive maze (the way that
   looks closer leads into a dead end), escaping a chaser, a crossroads' traffic lights, a walking creature of
   springs and muscles, and a dog herding sheep into a pen. The same interface as rl-envs.js (reset, step, snap,
   behavior, score; reward tables and formulas, inputs left out or added, sensor noise; a trace for the whole
   generation at once), each with its picture. Needs rl-worlds.js (its drawing kit). */
(function (NA) {
  'use strict';
  const E = NA.rlEnvs, pay = E.pay, { clamp, TAU, DEG, wrapAngle, seeded, rgbOf, mix, css, clock, rounded, GHOSTS } = E.kit;
  // the small preview beside the problem list leaves the words out, as the other worlds' previews do
  let quiet = false;
  const label = (...a) => { if (!quiet) E.kit.label(...a); }, banner = (...a) => { if (!quiet) E.kit.banner(...a); };
  const wordless = (draw) => (ctx, r, s, T, o = {}) => { quiet = !!o.small; try { return draw(ctx, r, s, T, o); } finally { quiet = false; } };
  const play = (W, policy, rng) => { let o = W.reset(rng), done = false; while (!done) ({ obs: o, done } = W.step(policy(o))); };
  // the distance from (px, py) to the segment (ax, ay)–(bx, by)
  const segDist = (px, py, ax, ay, bx, by) => { const vx = bx - ax, vy = by - ay, t = clamp(((px - ax) * vx + (py - ay) * vy) / (vx * vx + vy * vy || 1), 0, 1); return Math.hypot(px - ax - t * vx, py - ay - t * vy); };
  // how far a ray from (x, y) at angle a runs before it meets one of the segments (at most max)
  function rayCast(segs, x, y, a, max) {
    const dx = Math.cos(a), dy = Math.sin(a);
    let best = max;
    for (const [ax, ay, bx, by] of segs) {
      const ex = bx - ax, ey = by - ay, den = dx * ey - dy * ex;
      if (Math.abs(den) < 1e-9) continue;
      const t = ((ax - x) * ey - (ay - y) * ex) / den, u = ((ax - x) * dy - (ay - y) * dx) / den;
      if (t > 0 && u >= 0 && u <= 1 && t < best) best = t;
    }
    return best;
  }

  // =====================================================================================================
  // a deceptive maze
  // =====================================================================================================
  /* A round robot in a maze seen from above, 200 units square, with 5 rangefinders (90° and 45° to each side and
     straight ahead, out to 100) and a radar of 4 quarters that says in which quarter, around it, the goal is (but
     not how far). Rewards: by default, every step, how much closer it got to the goal in a straight line, and +1
     on reaching it. Closer is a trap here: the straight way runs into a wall just under the goal, and the way
     round first leads away from it, so an agent that only ever gets closer ends up pressed against that wall
     (the deceptive maze of Lehman and Stanley's novelty search). Its score: the share of 3 starts it reaches the
     goal from. maze 'trap': one wall, the way round on the far side; 'zigzag': two walls, the second the trap. */
  const MAZES = {
    trap: { name: 'Trap', walls: [[0, 70, 165, 70]], start: [30, 135, -Math.PI / 2], goal: [30, 28], steps: 400 },
    zigzag: { name: 'Zigzag', walls: [[0, 60, 158, 60], [42, 125, 200, 125]], start: [172, 175, Math.PI], goal: [26, 26], steps: 700 },
  };
  const BOX = [[0, 0, 200, 0], [200, 0, 200, 200], [200, 200, 0, 200], [0, 200, 0, 0]], FINDERS = [-90, -45, 0, 45, 90].map((d) => d * DEG), BOT = 5;
  // (novelty: the maze in 10 × 10 parts of 20 units, each new one it reaches can be paid for)
  const partOf = (x, y) => clamp(Math.floor(x / 20), 0, 9) + 10 * clamp(Math.floor(y / 20), 0, 9);
  class Trap {
    constructor(o = {}) {
      this.mazeId = MAZES[o.maze] ? o.maze : 'trap'; this.maze = MAZES[this.mazeId]; this.walls = [...BOX, ...this.maze.walls];
      this.maxSteps = o.steps > 0 ? clamp(Math.round(o.steps), 50, 5000) : this.maze.steps; this.view = 'state'; this.nA = 3; this.actions = ['forward', 'turn left', 'turn right'];
      this.speed = o.speed > 0 ? clamp(o.speed, 1, 8) : 3; // units a step forward (a turn moves it 1)
      this.input = { kind: 'vector', dim: 9 }; this.obsSize = 9;
      this.obsNames = ['wall 90° left', 'wall 45° left', 'wall ahead', 'wall 45° right', 'wall 90° right', 'goal ahead', 'goal to the left', 'goal behind', 'goal to the right'];
      this.rw = E.rewardTable('trap', o); this.rf = E.rewardFormula('trap', o.rFormula); E.sensors(this, 'trap', o);
    }
    dist() { return Math.hypot(this.maze.goal[0] - this.x, this.maze.goal[1] - this.y); }
    reset(rng) {
      const r = rng || seeded(29), [sx, sy, sa] = this.maze.start;
      this.rng = rng; this.x = sx + r.uniform(-4, 4); this.y = sy + r.uniform(-4, 4); this.th = sa + r.uniform(-0.3, 0.3); this.t = 0; this.end = null; this.bumps = 0;
      this.d = this.dist(); this.best = this.d; this.see(); this.trace = [this.x, this.y, this.th];
      this.parts = new Uint8Array(100); this.parts[partOf(this.x, this.y)] = 1; this.visited = 1;
      return this.obs();
    }
    see() { this.rays = FINDERS.map((a) => rayCast(this.walls, this.x, this.y, this.th + a, 100)); }
    rawObs() {
      const o = new Float32Array(9), rel = wrapAngle(Math.atan2(this.maze.goal[1] - this.y, this.maze.goal[0] - this.x) - this.th);
      for (let i = 0; i < 5; i++) o[i] = this.rays[i] / 100;
      // the radar's quarters: ahead, left, behind, right (angles grow clockwise, y pointing down)
      o[5 + (Math.abs(rel) < Math.PI / 4 ? 0 : rel < -Math.PI / 4 && rel > (-3 * Math.PI) / 4 ? 1 : Math.abs(rel) > (3 * Math.PI) / 4 ? 2 : 3)] = 1;
      return o;
    }
    step(a) {
      const turn = a === 1 ? -0.3 : a === 2 ? 0.3 : 0, speed = a === 0 ? this.speed : 1;
      this.th = wrapAngle(this.th + turn);
      const nx = this.x + Math.cos(this.th) * speed, ny = this.y + Math.sin(this.th) * speed;
      let bump = 0;
      if (this.walls.every(([ax, ay, bx, by]) => segDist(nx, ny, ax, ay, bx, by) >= BOT)) { this.x = nx; this.y = ny; } else { this.bumps++; bump = 1; }
      this.t++;
      this.see();
      // a part of the maze it had not reached before; nearer the goal than ever before (by how much, in hundreds of units)
      const d = this.dist(), closer = (this.d - d) / 100, goal = d < 8, part = partOf(this.x, this.y), fresh = !this.parts[part], record = Math.max(0, this.best - d) / 100;
      this.d = d; this.best = Math.min(this.best, d);
      if (fresh) { this.parts[part] = 1; this.visited++; }
      if (this.t % 2 === 0) this.trace.push(this.x, this.y, this.th);
      const done = goal || this.t >= this.maxSteps;
      if (done) this.end = goal ? 'goal' : 'lost';
      const rw = this.rw;
      let r = this.rf ? this.rf({ goal: +goal, closer, dist: d, x: this.x, y: this.y, bump, t: this.t, new: +fresh, turned: turn ? 1 : 0, visited: this.visited, best: this.best, record })
        : 0;
      // the table's reward, term by term (parts: what each paid this step, for the breakdown under the picture)
      const parts = this.rf ? null : {};
      if (!this.rf) { if (goal) r = pay(parts, 'rGoal', rw.rGoal, r); r = pay(parts, 'rStraight', rw.rStraight * closer, r); r = pay(parts, 'rStep', rw.rStep, r); if (bump) r = pay(parts, 'rBump', rw.rBump, r); if (fresh) r = pay(parts, 'rNew', rw.rNew, r); if (turn) r = pay(parts, 'rTurn', rw.rTurn, r); if (record) r = pay(parts, 'rRecord', rw.rRecord * record, r); }
      return { obs: this.obs(), r, done, parts };
    }
    snap() { return { x: this.x, y: this.y, th: this.th, rays: this.rays.slice(), maze: this.maze, walls: this.walls, t: this.t, d: this.d, end: this.end, trail: this.trace.slice(-240) }; }
    // where it ended up: across, and up the map
    behavior() { return [clamp(this.x / 200, 0, 1), clamp(1 - this.y / 200, 0, 1)]; }
    score(policy) { const r = seeded(4242); let ok = 0; for (let k = 0; k < 3; k++) { play(this, policy, seeded(r.int(1e9))); if (this.end === 'goal') ok++; } return ok / 3; }
  }

  // =====================================================================================================
  // escaping a chaser
  // =====================================================================================================
  /* An arena 20 m square, from above: it runs from a chaser that heads straight for it (aim 'ahead': for where it
     is going, a moment ahead, much harder to shake off), a little slower than it (fast: nearly as fast), and picks
     up the coins that turn up one at a time. Caught, the game ends. It moves with momentum: a push up, down, left
     or right each step, or none. map 'pillars': four pillars to dodge round.
     Rewards: a little every step it is still free, +1 a coin, −5 when caught. Its score: the share of 400 steps it
     stays free, over 3 games. */
  const PILLARS = [[6, 6], [14, 6], [6, 14], [14, 14]], ARENA_W = 20, PR = 1.4;
  class Chase {
    constructor(o = {}) {
      this.pillars = o.map === 'pillars' ? PILLARS : []; this.k = o.chasers === 2 ? 2 : 1; this.cs = o.speed === 'fast' ? 0.33 : 0.27; this.lead = o.aim === 'ahead' ? 0.5 : 0;
      this.maxSteps = o.steps > 0 ? clamp(Math.round(o.steps), 50, 5000) : 400; this.view = 'state'; this.nA = 5; this.actions = ['up', 'down', 'left', 'right', 'stay'];
      this.input = { kind: 'vector', dim: 18 }; this.obsSize = 18;
      this.obsNames = ['chaser across', 'chaser up', 'chaser speed across', 'chaser speed up', 'second chaser across', 'second chaser up', 'its speed across', 'its speed up', 'coin across', 'coin up',
        ...['up', 'up-right', 'right', 'down-right', 'down', 'down-left', 'left', 'up-left'].map((d) => `space ${d}`)];
      this.rw = E.rewardTable('chase', o); this.rf = E.rewardFormula('chase', o.rFormula); E.sensors(this, 'chase', o);
    }
    free(x, y, m) { return x > m && y > m && x < ARENA_W - m && y < ARENA_W - m && this.pillars.every(([px, py]) => Math.hypot(x - px, y - py) > PR + m); }
    // pushed back out of the walls and the pillars, losing the speed it had into them (against a wall it stops, rather
    // than pressing on with a speed it does not have)
    confine(o, m) {
      if (o.x < m) o.vx = Math.max(0, o.vx); if (o.x > ARENA_W - m) o.vx = Math.min(0, o.vx); if (o.y < m) o.vy = Math.max(0, o.vy); if (o.y > ARENA_W - m) o.vy = Math.min(0, o.vy);
      o.x = clamp(o.x, m, ARENA_W - m); o.y = clamp(o.y, m, ARENA_W - m);
      for (const [px, py] of this.pillars) {
        const dx = o.x - px, dy = o.y - py, d = Math.hypot(dx, dy) || 1;
        if (d < PR + m) { o.x = px + (dx / d) * (PR + m); o.y = py + (dy / d) * (PR + m); const vr = (o.vx * dx + o.vy * dy) / d; if (vr < 0) { o.vx -= (vr * dx) / d; o.vy -= (vr * dy) / d; } }
      }
    }
    spot(r, near) { for (let k = 0; k < 200; k++) { const x = r.uniform(1.5, ARENA_W - 1.5), y = r.uniform(1.5, ARENA_W - 1.5); if (this.free(x, y, 0.8) && (!near || Math.hypot(x - near.x, y - near.y) > 6)) return { x, y }; } return { x: ARENA_W / 2, y: 2 }; }
    reset(rng) {
      const r = rng || seeded(31);
      this.rng = r; this.me = { x: 10, y: 17, vx: 0, vy: 0 }; this.t = 0; this.end = null; this.coins = 0; this.sumD = 0;
      this.chasers = NA.range(this.k).map((i) => ({ x: 4 + i * 12, y: 3, vx: 0, vy: 0 }));
      this.coin = this.spot(r, this.me); this.trace = [this.me.x, this.me.y, 0];
      return this.obs();
    }
    near() { let b = this.chasers[0], bd = Infinity; for (const c of this.chasers) { const d = Math.hypot(c.x - this.me.x, c.y - this.me.y); if (d < bd) { bd = d; b = c; } } return [b, bd]; }
    rawObs() {
      const m = this.me, [c] = this.near(), c2 = this.chasers.find((q) => q !== c), o = new Float32Array(18);
      o[0] = (c.x - m.x) / 10; o[1] = (m.y - c.y) / 10; o[2] = c.vx / 0.35; o[3] = -c.vy / 0.35;
      if (c2) { o[4] = (c2.x - m.x) / 10; o[5] = (m.y - c2.y) / 10; }
      o[6] = m.vx / 0.4; o[7] = -m.vy / 0.4; o[8] = (this.coin.x - m.x) / 10; o[9] = (m.y - this.coin.y) / 10;
      for (let k = 0; k < 8; k++) { const a = -Math.PI / 2 + (k * Math.PI) / 4; let d = 0; while (d < 8 && this.free(m.x + Math.cos(a) * d, m.y + Math.sin(a) * d, 0.3)) d += 0.5; o[10 + k] = d / 8; }
      return o;
    }
    step(a) {
      const m = this.me, ax = a === 2 ? -1 : a === 3 ? 1 : 0, ay = a === 0 ? -1 : a === 1 ? 1 : 0;
      m.vx = (m.vx + 0.08 * ax) * 0.85; m.vy = (m.vy + 0.08 * ay) * 0.85;
      const sp = Math.hypot(m.vx, m.vy); if (sp > 0.4) { m.vx *= 0.4 / sp; m.vy *= 0.4 / sp; }
      m.x += m.vx; m.y += m.vy;
      // touching a wall or a pillar: pushed back out of it
      const wx = m.x, wy = m.y;
      this.confine(m, 0.4);
      const wall = m.x !== wx || m.y !== wy ? 1 : 0;
      // each chaser heads for where it is (aim 'ahead': where it will be, a moment ahead)
      for (const c of this.chasers) {
        const d = Math.hypot(m.x - c.x, m.y - c.y), lead = Math.min(8, d / this.cs) * this.lead, gx = m.x + m.vx * lead - c.x, gy = m.y + m.vy * lead - c.y, gl = Math.hypot(gx, gy) || 1;
        c.vx = c.vx * 0.6 + (gx / gl) * this.cs * 0.4; c.vy = c.vy * 0.6 + (gy / gl) * this.cs * 0.4;
        c.x += c.vx; c.y += c.vy; this.confine(c, 0.45);
      }
      this.t++;
      let coin = 0;
      if (Math.hypot(this.coin.x - m.x, this.coin.y - m.y) < 0.8) { coin = 1; this.coins++; this.coin = this.spot(this.rng, m); }
      const [, d] = this.near(), caught = d < 0.8;
      this.sumD += d;
      if (this.t % 2 === 0) this.trace.push(m.x, m.y, 0);
      const done = caught || this.t >= this.maxSteps;
      if (done) this.end = caught ? 'tagged' : 'escaped';
      // a chaser within 3 m; a push (any move but stay)
      const rw = this.rw, near = d < 3 ? 1 : 0, pushed = ax || ay ? 1 : 0;
      let r = this.rf ? this.rf({ caught: +caught, coin, dist: d, speed: Math.hypot(m.vx, m.vy) * 10, t: this.t, near, wall, pushed, coins: this.coins, cdist: Math.hypot(this.coin.x - m.x, this.coin.y - m.y) })
        : 0;
      const parts = this.rf ? null : {};
      if (!this.rf) { r = pay(parts, 'rFree', rw.rFree, r); if (coin) r = pay(parts, 'rCoin', coin * rw.rCoin, r); if (caught) r = pay(parts, 'rCaught', rw.rCaught, r); if (near) r = pay(parts, 'rNear', rw.rNear, r); if (wall) r = pay(parts, 'rWall', rw.rWall, r); if (pushed) r = pay(parts, 'rPush', rw.rPush, r); r = pay(parts, 'rDist', rw.rDist * d, r); }
      return { obs: this.obs(), r, done, parts };
    }
    snap() { return { me: { ...this.me }, chasers: this.chasers.map((c) => ({ ...c })), coin: { ...this.coin }, pillars: this.pillars, t: this.t, maxSteps: this.maxSteps, coins: this.coins, end: this.end, trail: this.trace.slice(-120) }; }
    // how far it keeps from the chaser, and how many coins it picks up
    behavior() { return [clamp(this.sumD / Math.max(1, this.t) / 14, 0, 1), clamp(this.coins / 10, 0, 1)]; }
    score(policy) { const r = seeded(4242); let s = 0; for (let k = 0; k < 3; k++) { play(this, policy, seeded(r.int(1e9))); s += this.t / this.maxSteps; } return s / 3; }
  }

  // =====================================================================================================
  // traffic lights
  // =====================================================================================================
  /* A crossroads: cars arrive from the north, south, east and west at random, and wait in a queue for green. One
     of the two roads has green at a time; switching first shows amber for 4 steps, when nobody moves. On green,
     each road lets a car through from each side every step. pattern 'shifting': the north–south road is busy for
     the first half and the east–west road for the second, as at rush hours. Rewards: a little off every step for
     every car waiting. Its score: the share of cars through within 20 steps of arriving. Settings: steps (400),
     amber (4 steps) and busy (how often cars come, × 1). */
  const LANES = ['north', 'south', 'east', 'west'];
  class Traffic {
    constructor(o = {}) {
      this.shifting = o.pattern !== 'steady'; this.maxSteps = o.steps > 0 ? clamp(Math.round(o.steps), 50, 5000) : 400; this.view = 'state'; this.nA = 2; this.actions = ['keep', 'switch'];
      this.amberFor = o.amber > 0 ? clamp(Math.round(o.amber), 1, 20) : 4; this.busy = o.busy > 0 ? clamp(o.busy, 0.2, 2.5) : 1;
      this.input = { kind: 'vector', dim: 8 }; this.obsSize = 8;
      this.obsNames = ['cars waiting from the north', 'from the south', 'from the east', 'from the west', 'which road has green', 'amber', 'how long since the last switch', 'longest wait at a red'];
      this.rw = E.rewardTable('traffic', o); this.rf = E.rewardFormula('traffic', o.rFormula); E.sensors(this, 'traffic', o);
    }
    rates(t) {
      const base = !this.shifting ? [0.3, 0.3, 0.2, 0.2] : t < this.maxSteps / 2 ? [0.38, 0.38, 0.1, 0.1] : [0.1, 0.1, 0.38, 0.38];
      return this.busy === 1 ? base : base.map((p) => Math.min(1, p * this.busy));
    }
    reset(rng) {
      const r = rng || seeded(37);
      this.rng = r; this.q = [[], [], [], []]; this.phase = 0; this.amber = 0; this.since = 0; this.t = 0; this.end = null; this.arrived = 0; this.served = 0; this.prompt = 0; this.switches = 0; this.nsGreen = 0; this.passing = [];
      return this.obs();
    }
    waitMax() { let w = 0; for (let l = 0; l < 4; l++) if (Math.floor(l / 2) !== this.phase || this.amber) if (this.q[l].length) w = Math.max(w, this.t - this.q[l][0]); return w; }
    rawObs() { return Float32Array.of(...this.q.map((q) => Math.min(q.length, 30) / 15), this.phase, this.amber ? 1 : 0, Math.min(this.since, 60) / 30, Math.min(this.waitMax(), 60) / 30); }
    step(a) {
      let switched = 0;
      if (a === 1 && !this.amber && this.since >= 2) { this.amber = this.amberFor; this.switches++; switched = 1; }
      // arrivals
      const rt = this.rates(this.t);
      for (let l = 0; l < 4; l++) if (this.rng.next() < rt[l]) { this.q[l].push(this.t); this.arrived++; }
      // departures on green (late: after waiting more than 20 steps)
      this.passing = [];
      let late = 0;
      if (this.amber) { this.amber--; if (!this.amber) { this.phase = 1 - this.phase; this.since = 0; } }
      else for (const l of this.phase ? [2, 3] : [0, 1]) if (this.q[l].length) { const t0 = this.q[l].shift(); this.served++; if (this.t - t0 <= 20) this.prompt++; else late++; this.passing.push(l); }
      this.since++; this.t++; if (!this.phase && !this.amber) this.nsGreen++;
      const waiting = this.q.reduce((s, q) => s + q.length, 0), done = this.t >= this.maxSteps;
      if (done) this.end = this.promptShare() >= 0.8 ? 'flowing' : 'jammed';
      const rw = this.rw, passed = this.passing.length;
      let r = this.rf ? this.rf({ waiting, switched, passed, longest: this.waitMax(), green: this.phase, t: this.t, late, ns: this.q[0].length + this.q[1].length, ew: this.q[2].length + this.q[3].length, amber: this.amber ? 1 : 0, since: this.since })
        : 0;
      const parts = this.rf ? null : {};
      if (!this.rf) { r = pay(parts, 'rWait', rw.rWait * waiting, r); if (switched) r = pay(parts, 'rSwitch', rw.rSwitch, r); if (passed) r = pay(parts, 'rPass', rw.rPass * passed, r); if (late) r = pay(parts, 'rLate', rw.rLate * late, r); r = pay(parts, 'rLong', rw.rLong * this.waitMax(), r); }
      return { obs: this.obs(), r, done, parts };
    }
    // cars through within 20 steps, of all that arrived (those still waiting longer than that count as late)
    promptShare() { let late = 0; for (const q of this.q) for (const t0 of q) if (this.t - t0 > 20) late++; return this.arrived ? this.prompt / Math.max(1, this.served + late) : 1; }
    snap() { return { q: this.q.map((q) => q.map((t0) => this.t - t0)), phase: this.phase, amber: this.amber, since: this.since, t: this.t, maxSteps: this.maxSteps, passing: this.passing.slice(), served: this.served, share: this.promptShare(), end: this.end }; }
    // how often it switches, and how much of the time the north–south road has green
    behavior() { const n = Math.max(1, this.t); return [clamp((this.switches / n) * 8, 0, 1), clamp(this.nsGreen / n, 0, 1)]; }
    score(policy) { const r = seeded(4242); let s = 0; for (let k = 0; k < 3; k++) { play(this, policy, seeded(r.int(1e9))); s += this.promptShare(); } return s / 3; }
  }

  // =====================================================================================================
  // a walking creature
  // =====================================================================================================
  /* A creature of springs, seen from the side: a stiff body (two hips 0.8 m apart), two stiff legs, and two muscles
     crossing from each hip to the other foot. Each step it sets both muscles short or long (four moves); the
     muscles pull towards their new length, which swings the legs. Its feet grip the ground; its body must not touch
     it. A clock (the time, going round every 20 steps) is among its inputs, so a gait can be a rhythm. One step is
     1/15 s (4 steps of the physics), 300 steps. Rewards: per metre walked right, and −1 for falling (a hip on the
     ground ends it). Its score: the share of 10 m it walks. Settings: steps (300), gravity (9.8 m/s²) and muscles
     (how far they pull in and let out: 'weak' 12%, 'normal' 18%, 'strong' 25%). */
  const W_REST = [[0, 1, 0.8], [0, 2, 1.04], [1, 3, 1.04]], W_MUSCLES = [[0, 3], [1, 2]], W_START = [[-0.4, 1], [0.4, 1], [-0.6, 0], [0.6, 0]];
  const W_PULL = { weak: [0.88, 1.12], normal: [0.82, 1.18], strong: [0.75, 1.25] };
  class Walker {
    constructor(o = {}) {
      this.grip = o.ground === 'ice' ? 2.5 : 6; this.maxSteps = o.steps > 0 ? clamp(Math.round(o.steps), 50, 5000) : 300; this.view = 'state'; this.nA = 4;
      this.g = o.gravity > 0 ? Math.min(o.gravity, 40) : 9.8;
      const [lo, hi] = W_PULL[o.muscles] || W_PULL.normal;
      this.moves = [[lo, hi], [hi, lo], [lo, lo], [hi, hi]]; // each move's muscle lengths, as a share of their rest lengths
      this.actions = ['back short, front long', 'back long, front short', 'both short', 'both long'];
      this.input = { kind: 'vector', dim: 11 }; this.obsSize = 11;
      this.obsNames = ['body tilt (sine)', 'body tilt (cosine)', 'body height', 'speed across', 'speed up', 'back foot down', 'front foot down', 'back muscle', 'front muscle', 'clock (sine)', 'clock (cosine)'];
      this.rw = E.rewardTable('walker', o); this.rf = E.rewardFormula('walker', o.rFormula); E.sensors(this, 'walker', o);
      this.rest = W_MUSCLES.map(([i, j]) => Math.hypot(W_START[i][0] - W_START[j][0], W_START[i][1] - W_START[j][1]));
    }
    reset(rng) {
      const r = rng || seeded(41);
      this.rng = rng; this.p = W_START.map(([x, y]) => [x + r.uniform(-0.02, 0.02), y + r.uniform(0, 0.02)]); this.q = this.p.map((v) => v.slice());
      this.len = this.rest.slice(); this.goal = this.rest.slice(); this.t = 0; this.end = null; this.a = 3; this.sumH = 0;
      this.x0 = this.cx(); this.lastX = this.x0; this.trace = [this.cx(), this.cy(), this.tilt()];
      return this.obs();
    }
    cx() { return (this.p[0][0] + this.p[1][0]) / 2; }
    cy() { return (this.p[0][1] + this.p[1][1]) / 2; }
    tilt() { return Math.atan2(this.p[1][1] - this.p[0][1], this.p[1][0] - this.p[0][0]); }
    rawObs() {
      const a = this.tilt(), vx = (this.cx() - (this.q[0][0] + this.q[1][0]) / 2) * 60, vy = (this.cy() - (this.q[0][1] + this.q[1][1]) / 2) * 60, ph = (TAU * this.t) / 20;
      return Float32Array.of(Math.sin(a), Math.cos(a), this.cy() - 0.8, vx / 2, vy / 2, this.p[2][1] < 0.02 ? 1 : 0, this.p[3][1] < 0.02 ? 1 : 0, (this.len[0] / this.rest[0] - 1) * 4, (this.len[1] / this.rest[1] - 1) * 4, Math.sin(ph), Math.cos(ph));
    }
    // one 1/60 s of physics: Verlet with gravity, then the bones and muscles held at their lengths, then the ground
    physics() {
      const g = this.g / 3600;
      for (let i = 0; i < 4; i++) { const p = this.p[i], q = this.q[i], vx = (p[0] - q[0]) * 0.995, vy = (p[1] - q[1]) * 0.995; q[0] = p[0]; q[1] = p[1]; p[0] += vx; p[1] += vy - g; }
      for (let m = 0; m < 2; m++) this.len[m] += clamp(this.goal[m] - this.len[m], -0.012, 0.012);
      for (let it = 0; it < 8; it++) {
        for (const [i, j, L] of W_REST) this.hold(i, j, L, 1);
        W_MUSCLES.forEach(([i, j], m) => this.hold(i, j, this.len[m], 0.5));
      }
      for (let i = 0; i < 4; i++) {
        const p = this.p[i], q = this.q[i];
        // the ground: friction as strong as the foot presses (how far it sank this moment), so the foot that carries
        // the weight grips and the other slides
        if (p[1] < 0) { const press = -p[1], vt = p[0] - q[0], hold = (i >= 2 ? this.grip : 1) * press; p[1] = 0; q[1] = p[1] + (p[1] - q[1]) * 0.2; q[0] = Math.abs(vt) <= hold ? p[0] : p[0] - (vt - Math.sign(vt) * hold); }
      }
    }
    hold(i, j, L, k) { const a = this.p[i], b = this.p[j], dx = b[0] - a[0], dy = b[1] - a[1], d = Math.hypot(dx, dy) || 1e-9, c = ((d - L) / d) * 0.5 * k; a[0] += dx * c; a[1] += dy * c; b[0] -= dx * c; b[1] -= dy * c; }
    step(a) {
      const switched = a !== this.a ? 1 : 0; // another move than the last
      this.a = a;
      const [s0, s1] = this.moves[a] || [1, 1];
      this.goal = [this.rest[0] * s0, this.rest[1] * s1];
      for (let k = 0; k < 4; k++) this.physics();
      this.t++;
      const x = this.cx(), gained = x - this.lastX, fell = this.p[0][1] < 0.1 || this.p[1][1] < 0.1;
      this.lastX = x; this.sumH += this.cy();
      if (this.t % 2 === 0) this.trace.push(x, this.cy(), this.tilt());
      const done = fell || this.t >= this.maxSteps;
      if (done) this.end = fell ? 'toppled' : x - this.x0 >= 10 ? 'walked' : 'slow';
      const rw = this.rw, tilt = this.tilt() / DEG;
      let r = this.rf ? this.rf({ gained, dist: x - this.x0, fell: +fell, height: this.cy(), tilt, t: this.t, speed: gained * 15, feet: (this.p[2][1] < 0.02 ? 1 : 0) + (this.p[3][1] < 0.02 ? 1 : 0), switched })
        : 0;
      const parts = this.rf ? null : {};
      if (!this.rf) { r = pay(parts, 'rForward', rw.rForward * gained, r); if (fell) r = pay(parts, 'rFall', rw.rFall, r); r = pay(parts, 'rStep', rw.rStep, r); r = pay(parts, 'rTilt', rw.rTilt * Math.abs(tilt), r); r = pay(parts, 'rHeight', rw.rHeight * this.cy(), r); if (switched) r = pay(parts, 'rSwitch', rw.rSwitch, r); }
      return { obs: this.obs(), r, done, parts };
    }
    snap() { return { p: this.p.map((v) => v.slice()), len: this.len.slice(), rest: this.rest, a: this.a, t: this.t, x0: this.x0, end: this.end, trail: this.trace.slice(-150) }; }
    // how fast it walks, and how high it carries its body
    behavior() { const n = Math.max(1, this.t); return [clamp((this.cx() - this.x0) / (n / 15) / 1.5 + 0.2, 0, 1), clamp((this.sumH / n - 0.4) / 0.8, 0, 1)]; }
    score(policy) { const r = seeded(4242); let s = 0; for (let k = 0; k < 2; k++) { play(this, policy, seeded(r.int(1e9))); s += clamp((this.cx() - this.x0) / 10, 0, 1); } return s / 2; }
  }

  // =====================================================================================================
  // herding sheep
  // =====================================================================================================
  /* A field 24 m square from above, a pen in its top right corner (open at the bottom), sheep, and the dog. Sheep
     graze and keep together, and run from the dog when it comes within 5 m; one that walks into the pen stays. The
     dog moves with momentum, a push up, down, left or right each step, or none. Rewards: +1 for each sheep penned
     and, by default, a little for every metre the sheep come nearer the pen's gate (shaping). Its score: the share
     of the sheep penned by the end, over 3 games. Settings: steps (600), flee (how near the dog comes before they
     run, 5 m) and dog (its top speed, 0.45 m a step). */
  const FIELD = 24, PEN = { x0: 16, x1: 22, y1: 6 }, GATE = [19, 6.5];
  class Herd {
    constructor(o = {}) {
      this.n = [3, 5, 7].includes(o.sheep) ? o.sheep : 3; this.maxSteps = o.steps > 0 ? clamp(Math.round(o.steps), 50, 5000) : 600; this.view = 'state'; this.nA = 5; this.actions = ['up', 'down', 'left', 'right', 'stay'];
      this.flee = o.flee > 0 ? clamp(o.flee, 1, 12) : 5; this.dogV = o.dog > 0 ? clamp(o.dog, 0.1, 1.5) : 0.45;
      this.input = { kind: 'vector', dim: 12 }; this.obsSize = 12;
      this.obsNames = ['dog across', 'dog up', 'flock across', 'flock up', 'straggler across', 'straggler up', 'gate from the flock, across', 'gate from the flock, up', 'its speed across', 'its speed up', 'share penned', 'how spread the flock is'];
      this.rw = E.rewardTable('herd', o); this.rf = E.rewardFormula('herd', o.rFormula); E.sensors(this, 'herd', o);
    }
    inPen(s) { return s.x > PEN.x0 && s.x < PEN.x1 && s.y < PEN.y1; }
    reset(rng) {
      const r = rng || seeded(43);
      this.rng = r; this.dog = { x: 2, y: 22, vx: 0, vy: 0 }; this.t = 0; this.end = null; this.path = 0; this.sumD = 0;
      this.sheep = NA.range(this.n).map(() => ({ x: r.uniform(4, 12), y: r.uniform(11, 20), vx: 0, vy: 0, penned: false, wx: r.uniform(-1, 1), wy: r.uniform(-1, 1) }));
      this.far = this.spread(); this.trace = [this.dog.x, this.dog.y, 0];
      return this.obs();
    }
    flock() { const f = this.sheep.filter((s) => !s.penned); if (!f.length) return { f, cx: GATE[0], cy: GATE[1] }; return { f, cx: f.reduce((a, s) => a + s.x, 0) / f.length, cy: f.reduce((a, s) => a + s.y, 0) / f.length }; }
    // the free sheep's distances to the gate, added up
    spread() { return this.sheep.reduce((a, s) => a + (s.penned ? 0 : Math.hypot(s.x - GATE[0], s.y - GATE[1])), 0); }
    rawObs() {
      const d = this.dog, { f, cx, cy } = this.flock();
      let str = f[0] || { x: cx, y: cy }, sd = -1;
      for (const s of f) { const dd = Math.hypot(s.x - GATE[0], s.y - GATE[1]); if (dd > sd) { sd = dd; str = s; } }
      const spr = f.length ? f.reduce((a, s) => a + Math.hypot(s.x - cx, s.y - cy), 0) / f.length : 0;
      return Float32Array.of(d.x / 12 - 1, 1 - d.y / 12, (cx - d.x) / 12, (d.y - cy) / 12, (str.x - d.x) / 12, (d.y - str.y) / 12, (GATE[0] - cx) / 12, (cy - GATE[1]) / 12, d.vx / this.dogV, -d.vy / this.dogV, 1 - f.length / this.n, spr / 6);
    }
    step(a) {
      const d = this.dog, ax = a === 2 ? -1 : a === 3 ? 1 : 0, ay = a === 0 ? -1 : a === 1 ? 1 : 0, ox = d.x, oy = d.y, R = this.flee;
      d.vx = (d.vx + 0.09 * ax) * 0.85; d.vy = (d.vy + 0.09 * ay) * 0.85;
      const sp = Math.hypot(d.vx, d.vy); if (sp > this.dogV) { d.vx *= this.dogV / sp; d.vy *= this.dogV / sp; }
      d.x = clamp(d.x + d.vx, 0.4, FIELD - 0.4); d.y = clamp(d.y + d.vy, 0.4, FIELD - 0.4);
      // at the fence it stops, rather than pressing on with a speed it does not have
      if (d.x <= 0.4) d.vx = Math.max(0, d.vx); if (d.x >= FIELD - 0.4) d.vx = Math.min(0, d.vx); if (d.y <= 0.4) d.vy = Math.max(0, d.vy); if (d.y >= FIELD - 0.4) d.vy = Math.min(0, d.vy);
      const ran = Math.hypot(d.x - ox, d.y - oy);
      this.path += ran;
      const { f, cx, cy } = this.flock();
      let penned = 0;
      for (const s of f) {
        let fx = 0, fy = 0;
        const dd = Math.hypot(s.x - d.x, s.y - d.y) || 1e-6;
        if (dd < R) { fx += ((s.x - d.x) / dd) * 0.13 * (1 - dd / R); fy += ((s.y - d.y) / dd) * 0.13 * (1 - dd / R); }
        const dc = Math.hypot(cx - s.x, cy - s.y); if (dc > 1.5) { fx += ((cx - s.x) / dc) * 0.012; fy += ((cy - s.y) / dc) * 0.012; }
        for (const o of f) if (o !== s) { const dx = s.x - o.x, dy = s.y - o.y, dn = Math.hypot(dx, dy); if (dn < 1 && dn > 0) { fx += (dx / dn) * 0.04; fy += (dy / dn) * 0.04; } }
        if (this.rng.next() < 0.05) { s.wx = this.rng.uniform(-1, 1); s.wy = this.rng.uniform(-1, 1); }
        fx += s.wx * 0.006; fy += s.wy * 0.006;
        s.vx = (s.vx + fx) * 0.8; s.vy = (s.vy + fy) * 0.8;
        const v = Math.hypot(s.vx, s.vy); if (v > 0.32) { s.vx *= 0.32 / v; s.vy *= 0.32 / v; }
        let nx = clamp(s.x + s.vx, 0.4, FIELD - 0.4), ny = clamp(s.y + s.vy, 0.4, FIELD - 0.4);
        // the pen's sides are fences: only its gate (the bottom) lets a sheep in
        if (ny < PEN.y1 && ((s.x <= PEN.x0) !== (nx <= PEN.x0) || (s.x >= PEN.x1) !== (nx >= PEN.x1))) nx = s.x;
        s.x = nx; s.y = ny;
        if (this.inPen(s)) { s.penned = true; penned++; s.vx = 0; s.vy = 0; }
      }
      this.t++;
      const now = this.spread(), nearer = this.far - now, left = this.sheep.filter((s) => !s.penned).length;
      this.far = now; this.sumD += Math.hypot(cx - d.x, cy - d.y);
      if (this.t % 2 === 0) this.trace.push(d.x, d.y, 0);
      const done = !left || this.t >= this.maxSteps;
      if (done) this.end = !left ? 'penned' : 'loose';
      const rw = this.rw, dist = Math.hypot(cx - d.x, cy - d.y), last = !left && penned > 0 ? 1 : 0;
      // (for a formula, the free sheep now: their middle's distance to the gate, and how spread they are about it)
      const G = this.rf ? this.flock() : null, n = G && G.f.length;
      let r = this.rf ? this.rf({ penned, nearer, left, dist, t: this.t, ran, last, gate: n ? Math.hypot(G.cx - GATE[0], G.cy - GATE[1]) : 0, spread: n ? G.f.reduce((q, s) => q + Math.hypot(s.x - G.cx, s.y - G.cy), 0) / n : 0 })
        : 0;
      const parts = this.rf ? null : {};
      if (!this.rf) { if (penned) r = pay(parts, 'rPenned', penned * rw.rPenned, r); r = pay(parts, 'rNearer', rw.rNearer * nearer, r); r = pay(parts, 'rStep', rw.rStep, r); r = pay(parts, 'rFlock', rw.rFlock * dist, r); r = pay(parts, 'rRun', rw.rRun * ran, r); if (last) r = pay(parts, 'rAll', rw.rAll, r); }
      return { obs: this.obs(), r, done, parts };
    }
    snap() { return { dog: { ...this.dog }, sheep: this.sheep.map((s) => ({ x: s.x, y: s.y, vx: s.vx, vy: s.vy, penned: s.penned })), t: this.t, maxSteps: this.maxSteps, n: this.n, end: this.end, trail: this.trace.slice(-150) }; }
    // how close it keeps to the flock, and how far it runs
    behavior() { const n = Math.max(1, this.t); return [clamp(this.sumD / n / 12, 0, 1), clamp(this.path / (n * 0.45), 0, 1)]; }
    score(policy) { const r = seeded(4242); let s = 0; for (let k = 0; k < 3; k++) { play(this, policy, seeded(r.int(1e9))); s += this.sheep.filter((q) => q.penned).length / this.n; } return s / 3; }
  }

  for (const W of [Trap, Chase, Traffic, Walker, Herd]) W.prototype.obs = E.sensed;

  // ------------------------------------------------------------------ rewards, inputs, the catalogue
  Object.assign(E.REWARDS, {
    trap: [{ key: 'rGoal', label: 'Reaching the goal', def: 1 }, { key: 'rStraight', label: 'Getting closer to it in a straight line, per 100 units', def: 1 }, { key: 'rStep', label: 'Every step', def: 0 },
      { key: 'rBump', label: 'Walking into a wall', def: 0 }, { key: 'rNew', label: 'Reaching a part of the maze it has not been to (novelty)', def: 0 },
      { key: 'rRecord', label: 'Getting nearer the goal than ever before, per 100 units', def: 0 }, { key: 'rTurn', label: 'Each turn', def: 0 }],
    chase: [{ key: 'rFree', label: 'Every step it is still free', def: 0.01 }, { key: 'rCoin', label: 'Picking up a coin', def: 1 }, { key: 'rCaught', label: 'Being caught', def: -5 },
      { key: 'rNear', label: 'Each step a chaser is within 3 m', def: 0 }, { key: 'rDist', label: 'Each step, per metre from the nearest chaser', def: 0 }, { key: 'rWall', label: 'Each step it touches a wall or a pillar', def: 0 },
      { key: 'rPush', label: 'Each push (any move but stay)', def: 0 }],
    traffic: [{ key: 'rWait', label: 'Every car waiting, every step', def: -0.01 }, { key: 'rSwitch', label: 'Each switch of the lights', def: 0 }, { key: 'rPass', label: 'Each car through', def: 0 },
      { key: 'rLate', label: 'Each car through late (after more than 20 steps)', def: 0 }, { key: 'rLong', label: 'Every step, per step of the longest wait at a red', def: 0 }],
    walker: [{ key: 'rForward', label: 'Every metre walked right', def: 1 }, { key: 'rFall', label: 'Falling', def: -1 }, { key: 'rStep', label: 'Every step', def: 0 },
      { key: 'rTilt', label: 'Each step, per degree its body tilts', def: 0 }, { key: 'rHeight', label: 'Each step, per metre its body is off the ground', def: 0 }, { key: 'rSwitch', label: 'Each change of muscle move', def: 0 }],
    herd: [{ key: 'rPenned', label: 'Each sheep penned', def: 1 }, { key: 'rNearer', label: 'Sheep nearer the gate, per metre (shaping)', def: 0.1 }, { key: 'rStep', label: 'Every step', def: 0 },
      { key: 'rAll', label: 'Penning the last sheep', def: 0 }, { key: 'rFlock', label: 'Each step, per metre the dog is from the flock', def: 0 }, { key: 'rRun', label: 'Each metre the dog runs', def: 0 }],
  });
  Object.assign(E.REWARD_VARS, {
    trap: [['goal', '1 on reaching the goal'], ['closer', 'how much closer it got this step, in hundreds of units'], ['dist', 'its distance to the goal in a straight line (the maze is 200 across)'], ['x', 'where it is across, 0 to 200'], ['y', 'where it is down, 0 at the top to 200'], ['bump', '1 when it walks into a wall'], ['t', 'steps so far'],
      ['new', '1 on reaching a part of the maze (a square of 20 × 20) it has not been to'], ['visited', 'how many of those parts it has been to'], ['best', 'its nearest yet to the goal'], ['record', 'how much nearer than ever before it got this step, in hundreds of units'], ['turned', '1 when it turned']],
    chase: [['caught', '1 when the chaser catches it'], ['coin', '1 on picking up a coin'], ['dist', 'its distance to the nearest chaser, m'], ['speed', 'its speed'], ['t', 'steps so far'],
      ['near', '1 while a chaser is within 3 m'], ['wall', '1 while it touches a wall or a pillar'], ['pushed', '1 when it pushed (any move but stay)'], ['coins', 'coins picked up so far'], ['cdist', 'how far the coin is, m']],
    traffic: [['waiting', 'how many cars are waiting'], ['switched', '1 on switching the lights'], ['passed', 'how many cars crossed this step'], ['longest', 'the longest wait at a red light, steps'], ['green', '0 while north–south has green, 1 east–west'], ['t', 'steps so far'],
      ['late', 'cars that crossed late this step (after more than 20 steps)'], ['ns', 'cars waiting north and south'], ['ew', 'cars waiting east and west'], ['amber', '1 while the lights show amber'], ['since', 'steps since the lights last switched']],
    walker: [['gained', 'how far right it got this step, m'], ['dist', 'how far it has walked, m'], ['fell', '1 when its body touches the ground'], ['height', 'its body’s height, m'], ['tilt', 'its body’s tilt, degrees'], ['t', 'steps so far'],
      ['speed', 'how fast it walks right, m/s'], ['feet', 'feet on the ground: 0, 1 or 2'], ['switched', '1 when it changed its muscle move']],
    herd: [['penned', 'sheep penned this step'], ['nearer', 'how much nearer the gate the sheep got this step, m'], ['left', 'sheep still out'], ['dist', 'the dog’s distance to the flock, m'], ['t', 'steps so far'],
      ['last', '1 on penning the last sheep'], ['ran', 'how far the dog ran this step, m'], ['gate', 'the flock’s distance to the gate, m'], ['spread', 'how spread out the flock is, m']],
  });
  // the extra inputs these can be given
  Object.assign(E.EXTRAS, {
    trap: [...E.COMMON_EXTRAS,
      { name: 'where it is', what: 'its place in the maze, across and down, 0 to 1', n: 2, names: ['its place across', 'its place down'], fill: (w, o, k) => { o[k] = w.x / 200; o[k + 1] = w.y / 200; } },
      { name: 'its heading', what: 'which way it faces (sine and cosine)', n: 2, names: ['facing (sine)', 'facing (cosine)'], fill: (w, o, k) => { o[k] = Math.sin(w.th); o[k + 1] = Math.cos(w.th); } },
      { name: 'distance to the goal', what: 'how far the goal is in a straight line (the radar gives only the quarter it is in): 1 is the maze’s width', n: 1, names: ['distance to the goal'], fill: (w, o, k) => { o[k] = w.d / 200; } }],
    chase: [...E.COMMON_EXTRAS,
      { name: 'the nearest chaser’s distance', what: 'how far the nearest chaser is, 1 at 14 m', n: 1, names: ['nearest chaser’s distance'], fill: (w, o, k) => { o[k] = w.near()[1] / 14; } },
      { name: 'where it is', what: 'its place in the arena, across and up, 0 to 1', n: 2, names: ['its place across', 'its place up'], fill: (w, o, k) => { o[k] = w.me.x / 20; o[k + 1] = 1 - w.me.y / 20; } }],
    traffic: [...E.COMMON_EXTRAS,
      { name: 'cars by road', what: 'the cars waiting on each road, north–south and east–west: 1 for 30 cars (2 at most)', n: 2, names: ['cars waiting north–south', 'cars waiting east–west'], fill: (w, o, k) => { o[k] = Math.min(w.q[0].length + w.q[1].length, 60) / 30; o[k + 1] = Math.min(w.q[2].length + w.q[3].length, 60) / 30; } }],
    walker: [...E.COMMON_EXTRAS,
      { name: 'its feet', what: 'how far each foot is in front of its body (behind: less than 0), m', n: 2, names: ['back foot from the body', 'front foot from the body'], fill: (w, o, k) => { const x = w.cx(); o[k] = w.p[2][0] - x; o[k + 1] = w.p[3][0] - x; } },
      { name: 'how far it has walked', what: 'how far it has walked, a tenth for each metre', n: 1, names: ['how far it has walked'], fill: (w, o, k) => { o[k] = (w.cx() - w.x0) / 10; } }],
    herd: [...E.COMMON_EXTRAS,
      { name: 'the nearest sheep', what: 'where the nearest sheep still out is, from the dog', n: 2, names: ['nearest sheep across', 'nearest sheep up'], fill: (w, o, k) => { let b = null, bd = Infinity; for (const s of w.sheep) { if (s.penned) continue; const dd = Math.hypot(s.x - w.dog.x, s.y - w.dog.y); if (dd < bd) { bd = dd; b = s; } } if (b) { o[k] = (b.x - w.dog.x) / 12; o[k + 1] = (w.dog.y - b.y) / 12; } } },
      { name: 'the gate from the dog', what: 'where the pen’s gate is, from the dog', n: 2, names: ['gate across from the dog', 'gate up from the dog'], fill: (w, o, k) => { o[k] = (GATE[0] - w.dog.x) / 12; o[k + 1] = (w.dog.y - GATE[1]) / 12; } }],
  });
  Object.assign(E.INPUT_GROUPS, {
    trap: { state: [['rangefinders', [0, 1, 2, 3, 4]], ['the goal’s radar', [5, 6, 7, 8]]] },
    chase: { state: [['the chaser', [0, 1, 2, 3]], ['the second chaser', [4, 5]], ['its own speed', [6, 7]], ['the coin', [8, 9]], ['space around it', [10, 11, 12, 13, 14, 15, 16, 17]]] },
    traffic: { state: [['queues', [0, 1, 2, 3]], ['the lights', [4, 5, 6]], ['longest wait', [7]]] },
    walker: { state: [['tilt and height', [0, 1, 2]], ['speed', [3, 4]], ['feet down', [5, 6]], ['muscles', [7, 8]], ['the clock', [9, 10]]] },
    herd: { state: [['the dog', [0, 1]], ['the flock', [2, 3]], ['the straggler', [4, 5]], ['the gate', [6, 7]], ['its speed', [8, 9]], ['penned and spread', [10, 11]]] },
  });
  Object.assign(E.WORLDS, {
    trap: { name: 'Deceptive maze', make: (o) => new Trap(o), what: '5 rangefinders and which quarter around it the goal is in', goal: 'reach the goal', axes: ['where it ends, across', 'where it ends, up'], demoMs: 30, trace: true },
    chase: { name: 'Escape the chaser', make: (o) => new Chase(o), what: 'where the chasers are and how they move, its own speed, the coin, and the space around it', goal: 'stay free for 400 steps', axes: ['how far it keeps from the chaser', 'how many coins it picks up'], demoMs: 45, trace: true },
    traffic: { name: 'Traffic lights', make: (o) => new Traffic(o), what: 'the four queues, the lights, how long since they switched and the longest wait at a red', goal: 'get every car through within 20 steps', axes: ['how often it switches', 'how much green north–south gets'], demoMs: 110, trace: false },
    walker: { name: 'Walker', make: (o) => new Walker(o), what: 'its tilt, height and speed, which feet are down, its muscles and a clock', goal: 'walk 10 m without falling', axes: ['how fast it walks', 'how high it carries its body'], demoMs: 40, trace: true },
    herd: { name: 'Herd the sheep', make: (o) => new Herd(o), what: 'where it is, the flock, the straggler and the gate, its speed, and how many are penned', goal: 'pen every sheep', axes: ['how close it keeps to the flock', 'how far it runs'], demoMs: 40, trace: true },
  });

  // =====================================================================================================
  // pictures
  // =====================================================================================================
  const fitBox = (r, aspect) => { let w = r.w, h = r.h; if (w / h > aspect) w = h * aspect; else h = w / aspect; return { x: r.x + (r.w - w) / 2, y: r.y + (r.h - h) / 2, w, h }; };
  const clip = (ctx, r, rad = 10) => { rounded(ctx, r.x, r.y, r.w, r.h, rad); ctx.clip(); };

  // ------------------------------------------------------------------ the maze
  /* A blueprint of the maze: its walls, the start, the goal glowing, the robot with its rangefinders, its trail.
     o.ghosts: a generation's robots, and where each one is. */
  function drawTrap(ctx, r, s, T, o = {}) {
    const V = fitBox(r, 1), sc = V.w / 200, X = (x) => V.x + x * sc, Y = (y) => V.y + y * sc, t = clock() / 1000;
    ctx.save(); clip(ctx, V);
    ctx.fillStyle = '#0d1b33'; ctx.fillRect(V.x, V.y, V.w, V.h);
    ctx.strokeStyle = 'rgba(120,170,255,0.09)'; ctx.lineWidth = 1;
    for (let k = 10; k < 200; k += 10) { ctx.beginPath(); ctx.moveTo(X(k), V.y); ctx.lineTo(X(k), V.y + V.h); ctx.moveTo(V.x, Y(k)); ctx.lineTo(V.x + V.w, Y(k)); ctx.stroke(); }
    ctx.strokeStyle = 'rgba(200,225,255,0.9)'; ctx.lineWidth = Math.max(3, sc * 3); ctx.lineCap = 'round';
    for (const [ax, ay, bx, by] of s.walls) { ctx.beginPath(); ctx.moveTo(X(ax), Y(ay)); ctx.lineTo(X(bx), Y(by)); ctx.stroke(); }
    ctx.lineCap = 'butt';
    // the goal and the start
    const [gx, gy] = s.maze.goal;
    for (let k = 3; k >= 1; k--) { ctx.strokeStyle = `rgba(255,214,102,${0.12 * k + (s.end === 'goal' ? 0.3 : 0.1 * Math.sin(t * 3 + k))})`; ctx.lineWidth = 2; ctx.beginPath(); ctx.arc(X(gx), Y(gy), (4 + k * 4) * sc, 0, TAU); ctx.stroke(); }
    ctx.fillStyle = '#ffd666'; ctx.beginPath(); for (let k = 0; k < 10; k++) { const a = -Math.PI / 2 + (k * Math.PI) / 5, rr = (k % 2 ? 2.4 : 5.5) * sc; ctx.lineTo(X(gx) + Math.cos(a) * rr, Y(gy) + Math.sin(a) * rr); } ctx.closePath(); ctx.fill();
    label(ctx, 'goal', X(gx), Y(gy) + 12 * sc, 10.5, 'rgba(255,214,102,0.9)', 'center', 700);
    const [sx, sy] = s.maze.start;
    ctx.strokeStyle = 'rgba(150,230,180,0.6)'; ctx.setLineDash([3, 3]); ctx.beginPath(); ctx.arc(X(sx), Y(sy), 8 * sc, 0, TAU); ctx.stroke(); ctx.setLineDash([]);
    label(ctx, 'start', X(sx), Y(sy) + 14 * sc, 10.5, 'rgba(150,230,180,0.85)', 'center', 650);
    const bot = (x, y, th, tone, big) => { ctx.fillStyle = tone; ctx.beginPath(); ctx.arc(X(x), Y(y), BOT * sc * (big ? 1 : 0.8), 0, TAU); ctx.fill(); if (big) { ctx.strokeStyle = '#0d1b33'; ctx.lineWidth = 2; ctx.beginPath(); ctx.moveTo(X(x), Y(y)); ctx.lineTo(X(x + Math.cos(th) * BOT), Y(y + Math.sin(th) * BOT)); ctx.stroke(); } };
    (o.ghosts || []).forEach((q, k) => bot(q.x, q.y, q.th, T.c(GHOSTS[(q.k ?? k) % 8], q.done ? 0.5 : 0.8), false));
    if (!o.ghostsOnly) {
      if (s.trail && s.trail.length > 6) { ctx.strokeStyle = 'rgba(120,200,255,0.45)'; ctx.lineWidth = 1.5; ctx.beginPath(); for (let i = 0; i < s.trail.length; i += 3) { if (i) ctx.lineTo(X(s.trail[i]), Y(s.trail[i + 1])); else ctx.moveTo(X(s.trail[i]), Y(s.trail[i + 1])); } ctx.stroke(); }
      if (!s.end) s.rays.forEach((d, i) => { const a = s.th + FINDERS[i], c = mix([255, 90, 90], [120, 230, 160], clamp(d / 60, 0, 1)); ctx.strokeStyle = css(c, 0.7); ctx.lineWidth = 1.2; ctx.beginPath(); ctx.moveTo(X(s.x), Y(s.y)); ctx.lineTo(X(s.x + Math.cos(a) * d), Y(s.y + Math.sin(a) * d)); ctx.stroke(); });
      bot(s.x, s.y, s.th, s.end === 'goal' ? T.c('good') : '#7fd1ff', true);
      label(ctx, `${Math.round(s.d)} from the goal · step ${s.t}`, V.x + 8, V.y + 12, 11, 'rgba(220,235,255,0.9)', 'left', 650);
      if (s.end) banner(ctx, V, s.end === 'goal' ? 'Found the goal' : 'Out of time', s.end === 'goal' ? T.c('good') : T.c('critical'));
    }
    ctx.restore();
  }

  // ------------------------------------------------------------------ the chase
  /* A dark arena from above: the pillars, the coin turning, the chasers (orange, their eyes on it) and it (blue,
     with its trail), coins taken and time left. o.ghosts: a generation's runners. */
  function drawChase(ctx, r, s, T, o = {}) {
    const V = fitBox(r, 1), sc = V.w / ARENA_W, X = (x) => V.x + x * sc, Y = (y) => V.y + y * sc, t = clock() / 1000;
    ctx.save(); clip(ctx, V);
    ctx.fillStyle = '#0b1020'; ctx.fillRect(V.x, V.y, V.w, V.h);
    ctx.strokeStyle = 'rgba(110,140,255,0.08)'; ctx.lineWidth = 1; for (let k = 1; k < ARENA_W; k++) { ctx.beginPath(); ctx.moveTo(X(k), V.y); ctx.lineTo(X(k), V.y + V.h); ctx.moveTo(V.x, Y(k)); ctx.lineTo(V.x + V.w, Y(k)); ctx.stroke(); }
    ctx.strokeStyle = 'rgba(140,170,255,0.5)'; ctx.lineWidth = 3; ctx.strokeRect(V.x + 1.5, V.y + 1.5, V.w - 3, V.h - 3);
    for (const [px, py] of s.pillars) { const g = ctx.createRadialGradient(X(px), Y(py), 1, X(px), Y(py), PR * sc * 1.5); g.addColorStop(0, 'rgba(150,120,255,0.9)'); g.addColorStop(0.7, 'rgba(80,60,180,0.85)'); g.addColorStop(1, 'rgba(80,60,180,0)'); ctx.fillStyle = g; ctx.beginPath(); ctx.arc(X(px), Y(py), PR * sc * 1.5, 0, TAU); ctx.fill(); }
    // the coin, turning
    const cw = Math.abs(Math.cos(t * 3)) * 0.35 * sc + 1.5; ctx.fillStyle = '#ffd24d'; ctx.beginPath(); ctx.ellipse(X(s.coin.x), Y(s.coin.y), cw, 0.35 * sc, 0, 0, TAU); ctx.fill();
    (o.ghosts || []).forEach((q, k) => { ctx.fillStyle = T.c(GHOSTS[(q.k ?? k) % 8], q.done ? 0.3 : 0.8); ctx.beginPath(); ctx.arc(X(q.x), Y(q.y), 0.3 * sc, 0, TAU); ctx.fill(); });
    if (!o.ghostsOnly) {
      if (s.trail && s.trail.length > 6) { ctx.strokeStyle = 'rgba(110,200,255,0.35)'; ctx.lineWidth = 2; ctx.beginPath(); for (let i = 0; i < s.trail.length; i += 3) { if (i) ctx.lineTo(X(s.trail[i]), Y(s.trail[i + 1])); else ctx.moveTo(X(s.trail[i]), Y(s.trail[i + 1])); } ctx.stroke(); }
      const m = s.me, g = ctx.createRadialGradient(X(m.x), Y(m.y), 1, X(m.x), Y(m.y), 0.9 * sc); g.addColorStop(0, '#e8f7ff'); g.addColorStop(0.5, '#4cc3ff'); g.addColorStop(1, 'rgba(76,195,255,0)');
      ctx.fillStyle = g; ctx.beginPath(); ctx.arc(X(m.x), Y(m.y), 0.9 * sc, 0, TAU); ctx.fill();
      for (const c of s.chasers) {
        const cg = ctx.createRadialGradient(X(c.x), Y(c.y), 1, X(c.x), Y(c.y), 0.95 * sc); cg.addColorStop(0, '#fff1e0'); cg.addColorStop(0.45, '#ff7a3d'); cg.addColorStop(1, 'rgba(255,90,40,0)');
        ctx.fillStyle = cg; ctx.beginPath(); ctx.arc(X(c.x), Y(c.y), 0.95 * sc, 0, TAU); ctx.fill();
        const a = Math.atan2(m.y - c.y, m.x - c.x);
        for (const e of [-0.45, 0.45]) { const ex = X(c.x + Math.cos(a + e) * 0.32), ey = Y(c.y + Math.sin(a + e) * 0.32); ctx.fillStyle = '#fff'; ctx.beginPath(); ctx.arc(ex, ey, 0.12 * sc, 0, TAU); ctx.fill(); ctx.fillStyle = '#111'; ctx.beginPath(); ctx.arc(ex + Math.cos(a) * 0.05 * sc, ey + Math.sin(a) * 0.05 * sc, 0.06 * sc, 0, TAU); ctx.fill(); }
      }
      ctx.fillStyle = 'rgba(255,255,255,0.12)'; ctx.fillRect(V.x, V.y, V.w, 3); ctx.fillStyle = '#4cc3ff'; ctx.fillRect(V.x, V.y, (V.w * s.t) / s.maxSteps, 3);
      label(ctx, `${s.coins} coins · ${s.maxSteps - s.t} steps to go`, V.x + 8, V.y + 14, 11, 'rgba(220,235,255,0.9)', 'left', 650);
      if (s.end) banner(ctx, V, s.end === 'tagged' ? 'Caught' : 'Got away', s.end === 'tagged' ? T.c('critical') : T.c('good'));
    }
    ctx.restore();
  }

  // ------------------------------------------------------------------ the crossroads
  /* The crossroads from above: the two roads, a stop line and a light at each approach, the queues of cars (the
     longer a car has waited, the redder), cars crossing, and the share of cars through in time. */
  const CAR_TONES = [[86, 140, 230], [235, 90, 80], [240, 196, 80], [120, 200, 140], [190, 120, 220], [230, 230, 235]];
  function drawTraffic(ctx, r, s, T) {
    const V = fitBox(r, 1), c = V.w / 2, R = V.w * 0.11, X = (u) => V.x + u, Y = (u) => V.y + u, car = V.w * 0.034;
    ctx.save(); clip(ctx, V);
    ctx.fillStyle = '#3d6b3a'; ctx.fillRect(V.x, V.y, V.w, V.h);
    ctx.fillStyle = '#38393e'; ctx.fillRect(X(c - R), V.y, 2 * R, V.h); ctx.fillRect(V.x, Y(c - R), V.w, 2 * R);
    ctx.strokeStyle = 'rgba(255,255,255,0.75)'; ctx.lineWidth = 2; ctx.setLineDash([8, 8]);
    ctx.beginPath(); ctx.moveTo(X(c), V.y); ctx.lineTo(X(c), Y(c - R)); ctx.moveTo(X(c), Y(c + R)); ctx.lineTo(X(c), V.y + V.h); ctx.moveTo(V.x, Y(c)); ctx.lineTo(X(c - R), Y(c)); ctx.moveTo(X(c + R), Y(c)); ctx.lineTo(V.x + V.w, Y(c)); ctx.stroke(); ctx.setLineDash([]);
    // zebra crossings
    ctx.fillStyle = 'rgba(255,255,255,0.6)';
    for (let k = 0; k < 6; k++) { const u = c - R + (k + 0.25) * (2 * R / 6); ctx.fillRect(X(u), Y(c - R - car * 1.6), (2 * R) / 12, car * 1.2); ctx.fillRect(X(u), Y(c + R + car * 0.4), (2 * R) / 12, car * 1.2); ctx.fillRect(X(c - R - car * 1.6), Y(u), car * 1.2, (2 * R) / 12); ctx.fillRect(X(c + R + car * 0.4), Y(u), car * 1.2, (2 * R) / 12); }
    // each approach: where its queue stands and which way it heads (north: from the top, in the left lane going down)
    const ap = [{ x: c - R / 2, y: c - R, dx: 0, dy: -1 }, { x: c + R / 2, y: c + R, dx: 0, dy: 1 }, { x: c + R, y: c - R / 2, dx: 1, dy: 0 }, { x: c - R, y: c + R / 2, dx: -1, dy: 0 }];
    s.q.forEach((q, l) => {
      const a = ap[l], green = !s.amber && Math.floor(l / 2) === s.phase;
      q.slice(0, 22).forEach((w, k) => {
        const d = car * 2 + k * car * 1.55, x = X(a.x + a.dx * d), y = Y(a.y + a.dy * d), tone = mix(CAR_TONES[(k + l * 3) % CAR_TONES.length], [255, 60, 40], clamp(w / 40, 0, 0.7));
        ctx.fillStyle = css(tone); rounded(ctx, x - (a.dx ? car * 0.65 : car * 0.42), y - (a.dy ? car * 0.65 : car * 0.42), a.dx ? car * 1.3 : car * 0.84, a.dy ? car * 1.3 : car * 0.84, 3); ctx.fill();
      });
      if (q.length > 22) label(ctx, `+${q.length - 22}`, X(a.x + a.dx * car * 37), Y(a.y + a.dy * car * 37), 10.5, '#fff', 'center', 700);
      // its light
      const lx = X(a.x + a.dx * car * 0.9 + (a.dy ? -a.dy * R * 0.75 : 0)), ly = Y(a.y + a.dy * car * 0.9 + (a.dx ? a.dx * R * 0.75 : 0));
      ctx.fillStyle = '#111'; rounded(ctx, lx - car * 0.45, ly - car * 0.45, car * 0.9, car * 0.9, 3); ctx.fill();
      ctx.fillStyle = s.amber ? '#ffb020' : green ? '#3ddc6e' : '#ff4040'; ctx.beginPath(); ctx.arc(lx, ly, car * 0.3, 0, TAU); ctx.fill();
      label(ctx, String(q.length), X(a.x + a.dx * car * 0.9 + (a.dy ? a.dy * R * 0.9 : 0)), Y(a.y + a.dy * car * 0.9 + (a.dx ? -a.dx * R * 0.9 : 0)), 11, '#fff', 'center', 750);
    });
    // cars crossing this step
    for (const l of s.passing) { const a = ap[l], x = X(c - a.dx * R * 0.2 + (a.dy ? (l === 0 ? -R / 2 : R / 2) : 0)), y = Y(c - a.dy * R * 0.2 + (a.dx ? (l === 2 ? -R / 2 : R / 2) : 0)); ctx.fillStyle = 'rgba(240,240,250,0.9)'; ctx.beginPath(); ctx.arc(x, y, car * 0.45, 0, TAU); ctx.fill(); }
    ctx.fillStyle = 'rgba(10,15,25,0.65)'; rounded(ctx, V.x + 6, V.y + 6, Math.min(V.w - 12, 250), 22, 6); ctx.fill();
    label(ctx, `${Math.round(s.share * 100)}% through in time · step ${s.t} of ${s.maxSteps}`, V.x + 14, V.y + 17, 11, '#fff', 'left', 650);
    if (s.end) banner(ctx, V, s.end === 'flowing' ? 'Traffic kept flowing' : 'Jammed', s.end === 'flowing' ? T.c('good') : T.c('critical'));
    ctx.restore();
  }

  // ------------------------------------------------------------------ the walker
  /* From the side, the camera following it: the sky, the ground with a mark every metre, and the creature: its
     body and legs as bones, its muscles as red bands (thicker when short), its feet. o.ghosts: a generation's
     bodies (the camera on the leader). */
  function drawWalker(ctx, r, s, T, o = {}) {
    const V = fitBox(r, 2), sc = V.h / 2.6, gy = V.y + V.h * 0.82, x0 = s.p ? (s.p[0][0] + s.p[1][0]) / 2 : 0, lead = o.ghosts && o.ghosts.length && o.ghostsOnly ? Math.max(...o.ghosts.map((q) => q.x)) : x0;
    const X = (x) => V.x + V.w * 0.35 + (x - lead) * sc, Y = (y) => gy - y * sc;
    ctx.save(); clip(ctx, V);
    let g = ctx.createLinearGradient(0, V.y, 0, gy); g.addColorStop(0, '#9ccbe8'); g.addColorStop(1, '#e7f2f7'); ctx.fillStyle = g; ctx.fillRect(V.x, V.y, V.w, V.h);
    ctx.fillStyle = '#6d8f4e'; ctx.fillRect(V.x, gy, V.w, V.y + V.h - gy); ctx.fillStyle = '#87ad61'; ctx.fillRect(V.x, gy, V.w, 4);
    const from = Math.floor(lead - (V.w * 0.35) / sc), to = Math.ceil(lead + (V.w * 0.65) / sc), start = s.x0 || 0;
    for (let m = from; m <= to; m++) { ctx.fillStyle = 'rgba(255,255,255,0.7)'; ctx.fillRect(X(m) - 1, gy, 2, 8); if ((m - Math.round(start)) % 2 === 0) label(ctx, `${m - Math.round(start)} m`, X(m), gy + 16, 10.5, 'rgba(255,255,255,0.9)', 'center', 650); }
    (o.ghosts || []).forEach((q, k) => { const c = T.c(GHOSTS[(q.k ?? k) % 8], q.done ? 0.3 : 0.6), dx = Math.cos(q.th) * 0.4, dy = Math.sin(q.th) * 0.4; ctx.strokeStyle = c; ctx.lineWidth = 3; ctx.beginPath(); ctx.moveTo(X(q.x - dx), Y(q.y - dy)); ctx.lineTo(X(q.x + dx), Y(q.y + dy)); ctx.stroke(); ctx.beginPath(); ctx.moveTo(X(q.x - dx), Y(q.y - dy)); ctx.lineTo(X(q.x - dx - 0.15), Y(0)); ctx.moveTo(X(q.x + dx), Y(q.y + dy)); ctx.lineTo(X(q.x + dx + 0.15), Y(0)); ctx.stroke(); });
    if (!o.ghostsOnly && s.p) {
      const P = s.p.map(([x, y]) => [X(x), Y(y)]);
      if (s.trail && s.trail.length > 6) { ctx.strokeStyle = 'rgba(80,110,160,0.35)'; ctx.lineWidth = 2; ctx.beginPath(); for (let i = 0; i < s.trail.length; i += 3) { if (i) ctx.lineTo(X(s.trail[i]), Y(s.trail[i + 1])); else ctx.moveTo(X(s.trail[i]), Y(s.trail[i + 1])); } ctx.stroke(); }
      // muscles: thicker and redder when short
      W_MUSCLES.forEach(([i, j], m) => { const k = clamp((s.rest[m] * 1.18 - s.len[m]) / (s.rest[m] * 0.36), 0, 1); ctx.strokeStyle = css(mix([230, 150, 140], [205, 40, 50], k), 0.85); ctx.lineWidth = 3 + 6 * k; ctx.lineCap = 'round'; ctx.beginPath(); ctx.moveTo(P[i][0], P[i][1]); ctx.lineTo(P[j][0], P[j][1]); ctx.stroke(); });
      ctx.strokeStyle = '#2d3a52'; ctx.lineWidth = 7;
      for (const [i, j] of W_REST) { ctx.beginPath(); ctx.moveTo(P[i][0], P[i][1]); ctx.lineTo(P[j][0], P[j][1]); ctx.stroke(); }
      ctx.lineCap = 'butt';
      P.forEach(([x, y], i) => { ctx.fillStyle = i >= 2 ? '#1b2333' : '#4b6189'; ctx.beginPath(); ctx.arc(x, y, i >= 2 ? 6 : 7, 0, TAU); ctx.fill(); });
      label(ctx, `${((x0 - start) || 0).toFixed(1)} m · step ${s.t}`, V.x + 10, V.y + 14, 11.5, 'rgba(20,35,60,0.85)', 'left', 650);
      if (s.end) banner(ctx, V, s.end === 'toppled' ? 'Fell over' : s.end === 'walked' ? 'Walked 10 m' : 'Too slow', s.end === 'walked' ? T.c('good') : T.c('critical'));
    }
    ctx.restore();
  }

  // ------------------------------------------------------------------ the field
  /* The field from above: grass, the pen and its gate, the sheep (woolly, heads the way they go; penned ones
     inside), the dog with its trail, and how many are penned. o.ghosts: a generation's dogs. */
  function drawHerd(ctx, r, s, T, o = {}) {
    const V = fitBox(r, 1), sc = V.w / FIELD, X = (x) => V.x + x * sc, Y = (y) => V.y + y * sc;
    ctx.save(); clip(ctx, V);
    ctx.fillStyle = '#5f8f45'; ctx.fillRect(V.x, V.y, V.w, V.h);
    ctx.strokeStyle = 'rgba(40,80,30,0.25)'; ctx.lineWidth = 1;
    for (let k = 0; k < 120; k++) { const u = (k * 37.7) % FIELD, v = (k * 53.3) % FIELD; ctx.beginPath(); ctx.moveTo(X(u), Y(v)); ctx.lineTo(X(u + 0.15), Y(v - 0.4)); ctx.stroke(); }
    ctx.fillStyle = 'rgba(150,120,70,0.35)'; ctx.fillRect(X(PEN.x0), Y(0), (PEN.x1 - PEN.x0) * sc, PEN.y1 * sc);
    ctx.strokeStyle = '#7b5a36'; ctx.lineWidth = Math.max(3, sc * 0.25);
    ctx.beginPath(); ctx.moveTo(X(PEN.x0), Y(0)); ctx.lineTo(X(PEN.x0), Y(PEN.y1)); ctx.moveTo(X(PEN.x1), Y(0)); ctx.lineTo(X(PEN.x1), Y(PEN.y1)); ctx.stroke();
    ctx.setLineDash([4, 5]); ctx.strokeStyle = 'rgba(255,255,255,0.5)'; ctx.lineWidth = 1.5; ctx.beginPath(); ctx.moveTo(X(PEN.x0), Y(PEN.y1)); ctx.lineTo(X(PEN.x1), Y(PEN.y1)); ctx.stroke(); ctx.setLineDash([]);
    label(ctx, 'pen', X((PEN.x0 + PEN.x1) / 2), Y(1.2), 11, 'rgba(255,245,225,0.9)', 'center', 700);
    for (const sh of s.sheep) {
      const a = Math.hypot(sh.vx, sh.vy) > 0.02 ? Math.atan2(sh.vy, sh.vx) : -Math.PI / 2;
      ctx.fillStyle = 'rgba(0,0,0,0.18)'; ctx.beginPath(); ctx.ellipse(X(sh.x) + 2, Y(sh.y) + 2, 0.55 * sc, 0.4 * sc, a, 0, TAU); ctx.fill();
      ctx.fillStyle = sh.penned ? '#fff9ee' : '#f4f1e8'; ctx.beginPath(); ctx.ellipse(X(sh.x), Y(sh.y), 0.55 * sc, 0.4 * sc, a, 0, TAU); ctx.fill();
      ctx.fillStyle = '#2b2b2b'; ctx.beginPath(); ctx.arc(X(sh.x + Math.cos(a) * 0.55), Y(sh.y + Math.sin(a) * 0.55), 0.2 * sc, 0, TAU); ctx.fill();
    }
    (o.ghosts || []).forEach((q, k) => { ctx.fillStyle = T.c(GHOSTS[(q.k ?? k) % 8], q.done ? 0.35 : 0.85); ctx.beginPath(); ctx.arc(X(q.x), Y(q.y), 0.35 * sc, 0, TAU); ctx.fill(); });
    if (!o.ghostsOnly) {
      if (s.trail && s.trail.length > 6) { ctx.strokeStyle = 'rgba(255,240,200,0.35)'; ctx.lineWidth = 2; ctx.beginPath(); for (let i = 0; i < s.trail.length; i += 3) { if (i) ctx.lineTo(X(s.trail[i]), Y(s.trail[i + 1])); else ctx.moveTo(X(s.trail[i]), Y(s.trail[i + 1])); } ctx.stroke(); }
      const d = s.dog, a = Math.hypot(d.vx, d.vy) > 0.02 ? Math.atan2(d.vy, d.vx) : 0;
      ctx.save(); ctx.translate(X(d.x), Y(d.y)); ctx.rotate(a);
      ctx.fillStyle = '#5a3a1e'; ctx.beginPath(); ctx.ellipse(0, 0, 0.55 * sc, 0.28 * sc, 0, 0, TAU); ctx.fill();
      ctx.beginPath(); ctx.arc(0.55 * sc, 0, 0.22 * sc, 0, TAU); ctx.fill();
      ctx.strokeStyle = '#5a3a1e'; ctx.lineWidth = 3; ctx.beginPath(); ctx.moveTo(-0.5 * sc, 0); ctx.lineTo(-0.85 * sc, Math.sin(clock() / 90) * 0.2 * sc); ctx.stroke();
      ctx.fillStyle = '#f2e6d0'; ctx.beginPath(); ctx.arc(0.7 * sc, 0, 0.08 * sc, 0, TAU); ctx.fill();
      ctx.restore();
      const n = s.sheep.filter((q) => q.penned).length;
      ctx.fillStyle = 'rgba(20,35,15,0.6)'; rounded(ctx, V.x + 6, V.y + 6, 190, 22, 6); ctx.fill();
      label(ctx, `${n} of ${s.n} penned · step ${s.t} of ${s.maxSteps}`, V.x + 14, V.y + 17, 11, '#fff', 'left', 650);
      if (s.end) banner(ctx, V, s.end === 'penned' ? 'Every sheep penned' : `${n} of ${s.n} penned`, s.end === 'penned' ? T.c('good') : T.c('critical'));
    }
    ctx.restore();
  }

  Object.assign(E.DRAW, { trap: wordless(drawTrap), chase: wordless(drawChase), traffic: wordless(drawTraffic), walker: wordless(drawWalker), herd: wordless(drawHerd) });
  Object.assign(E, { Trap, Chase, Traffic, Walker, Herd, MAZES_TRAP: MAZES });
})(typeof globalThis !== 'undefined' && globalThis.NA ? globalThis.NA : window.NA);
