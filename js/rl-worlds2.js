/* rl-worlds2.js — more worlds for the model builder's "Learn from rewards", beginner to intermediate: a room to keep
   warm (a heater whose radiator is slow to warm and slow to cool), hurdles to jump and bars to duck, a lamp to
   remember down a corridor (memory), a cart in a valley that has to rock to climb out, a pendulum to swing up and
   balance, and a two-joint robot arm that reaches for targets. The same interface as rl-envs.js and rl-worlds.js
   (reset, step, snap, behavior, score; reward tables and formulas, inputs left out or added, sensor noise; a trace
   for the whole generation at once), each with its picture. Needs rl-worlds.js (its drawing kit). */
(function (NA) {
  'use strict';
  const E = NA.rlEnvs, pay = E.pay, { clamp, TAU, DEG, wrapAngle, seeded, rgbOf, mix, css, clock, rounded, GHOSTS } = E.kit;
  // the small preview beside the problem list leaves the words out, as the other worlds' previews do
  let quiet = false;
  const label = (...a) => { if (!quiet) E.kit.label(...a); }, banner = (...a) => { if (!quiet) E.kit.banner(...a); };
  const wordless = (draw) => (ctx, r, s, T, o = {}) => { quiet = !!o.small; try { return draw(ctx, r, s, T, o); } finally { quiet = false; } };
  const play = (W, policy, rng) => { let o = W.reset(rng), done = false; while (!done) ({ obs: o, done } = W.step(policy(o))); };

  // =====================================================================================================
  // a room to keep warm
  // =====================================================================================================
  /* A day in 144 steps of 10 minutes, from midnight. The heating warms a radiator and the radiator warms the room:
     switched on, the room warms only as fast as the radiator does, and switched off it goes on warming until the
     radiator has cooled, so a good thermostat switches off before the room is warm enough. The room loses heat to
     the outside, coldest before dawn and warmest at three in the afternoon (poor insulation: faster). schedule
     'home': 21 °C in the morning (6 to 8) and the evening (17 to 23), 16 °C otherwise; 'day': 21 °C all day.
     Rewards: a little off for every degree from the target, every step, and for every step the heating is on.
     Its score: the share of the day within 1 °C of the target. Settings: days (1), warm (the target, 21 °C; 5 less
     when nobody is in), band (how near the target counts as comfortable, 1 °C) and radiator (its hottest, 60 °C). */
  const CLIMATES = { mild: { mean: 8, amp: 5 }, cold: { mean: -2, amp: 6 } };
  class Thermo {
    constructor(o = {}) {
      this.climate = CLIMATES[o.climate] ? o.climate : 'mild'; this.leak = o.insulation === 'poor' ? 0.025 : 0.01; this.home = o.schedule === 'home';
      this.view = 'state'; this.maxSteps = 144 * (o.days > 0 ? clamp(Math.round(o.days), 1, 7) : 1); this.nA = 2; this.actions = ['heating off', 'heating on'];
      this.warm = o.warm > 0 ? clamp(o.warm, 10, 30) : 21; this.band = o.band > 0 ? Math.min(o.band, 5) : 1; this.hot = o.radiator > 0 ? clamp(o.radiator, 30, 95) : 60;
      this.input = { kind: 'vector', dim: 7 }; this.obsSize = 7;
      this.obsNames = ['room, from the target', 'outside', 'radiator', 'the target', 'time of day (sine)', 'time of day (cosine)', 'heating on'];
      this.rw = E.rewardTable('thermo', o); this.rf = E.rewardFormula('thermo', o.rFormula); E.sensors(this, 'thermo', o);
    }
    target(h) { return !this.home || (h >= 6 && h < 8) || (h >= 17 && h < 23) ? this.warm : this.warm - 5; }
    outside(t) { const c = CLIMATES[this.climate], h = (t / 6) % 24; return c.mean + this.shift + c.amp * Math.cos((TAU * (h - 15)) / 24); }
    reset(rng) {
      const r = rng || seeded(3);
      this.rng = rng; this.shift = r.uniform(-2, 2); this.t = 0; this.heat = 0; this.used = 0; this.inBand = 0; this.sumOff = 0; this.end = null;
      this.T = this.target(0) + r.uniform(-3, 1); this.R = this.T; this.To = this.outside(0);
      this.hist = [this.T, this.To, 0]; this.trace = [0, this.T, 0];
      return this.obs();
    }
    rawObs() { const h = (this.t / 6) % 24, tg = this.target(h); return Float32Array.of((this.T - tg) / 5, (this.To - 10) / 10, (this.R - 20) / 40, (tg - 18) / 5, Math.sin((TAU * h) / 24), Math.cos((TAU * h) / 24), this.heat); }
    step(a) {
      const was = this.heat;
      this.heat = a === 1 ? 1 : 0;
      this.R += (this.heat ? this.hot - this.R : this.T - this.R) * 0.25;
      this.To = this.outside(this.t);
      this.T += 0.025 * (this.R - this.T) + this.leak * (this.To - this.T);
      this.t++;
      const h = (this.t / 6) % 24, tg = this.target(h), off = Math.abs(this.T - tg), ok = off <= this.band;
      this.used += this.heat; this.inBand += ok ? 1 : 0; this.sumOff += this.T - tg;
      this.hist.push(this.T, this.To, this.heat);
      if (this.t % 2 === 0) this.trace.push(this.t, this.T, this.heat);
      const done = this.t >= this.maxSteps;
      if (done) this.end = this.inBand / this.t >= 0.8 ? 'comfy' : 'uncomfy';
      // too cold or too warm, by how much; the heating switched on or off
      const cold = Math.max(0, tg - this.T), hot = Math.max(0, this.T - tg), switched = this.heat !== was, rw = this.rw;
      const parts = this.rf ? null : {};
      let r = 0;
      if (this.rf) r = this.rf({ temp: this.T, target: tg, off, inside: +ok, outside: this.To, heating: this.heat, radiator: this.R, hour: h, t: this.t, cold, hot, switched: +switched });
      else {
        r = pay(parts, 'rOff', rw.rOff * off, r); r = pay(parts, 'rHeat', rw.rHeat * this.heat, r); if (ok) r = pay(parts, 'rBand', rw.rBand, r);
        if (rw.rCold) r = pay(parts, 'rCold', rw.rCold * cold, r); if (rw.rHot) r = pay(parts, 'rHot', rw.rHot * hot, r); if (switched) r = pay(parts, 'rSwitch', rw.rSwitch, r);
      }
      return { obs: this.obs(), r, done, parts };
    }
    snap() { return { t: this.t, T: this.T, To: this.To, R: this.R, heat: this.heat, target: this.target((this.t / 6) % 24), home: this.home, warm: this.warm, band: this.band, maxSteps: this.maxSteps, hist: this.hist.slice(), end: this.end, comfy: this.inBand / Math.max(1, this.t) }; }
    // how much it heats, and how warm it keeps the room (from 3 °C below the target to 3 °C above)
    behavior() { const n = Math.max(1, this.t); return [clamp(this.used / n, 0, 1), clamp(0.5 + this.sumOff / n / 6, 0, 1)]; }
    // the share of the day within 1 °C of the target, over 3 days (the same 3 every time)
    score(policy) { const r = seeded(4242); let s = 0; for (let k = 0; k < 3; k++) { play(this, policy, seeded(r.int(1e9))); s += this.inBand / this.t; } return s / 3; }
  }

  // =====================================================================================================
  // hurdles
  // =====================================================================================================
  /* A runner at 8 m/s down a 150 m course of hurdles to jump (0.7 m high) and, with bars on, bars to duck under
     (their underside 1.2 m up). A jump leaves the ground at 5.5 m/s against 15 m/s² of gravity: 0.73 s in the
     air, its feet above a hurdle's height for about 3 m, so it has to take off at the right moment. Ducking keeps
     its head down at 1 m while it holds the move, and a moment longer. One step is 0.05 s. Rewards: a little for
     every step, −1 on hitting anything (the run ends), +1 at the finish. Its score: the share of the course run.
     Settings: course (its length, 150 m), run (its speed, 8 m/s) and jump (the speed it takes off at, 5.5 m/s). */
  const RUN = 8, DTR = 0.05, JUMP = 5.5, GRAV = 15, HURDLE = 0.7, BAR = 1.2, TALL = 1.8, LOW = 1.0, COURSE = 150;
  /* A course of your own (the model builder's editor): 1 to 60 obstacles, each [metres from the start, 1 for a bar or
     0 for a hurdle], from 6 m in to 3 m before the finish, none within a metre of another; the same course every
     episode. { ok, why, close (the nearest two, m apart) }. */
  E.courseCheck = function (list, course) {
    const C = course > 0 ? clamp(course, 40, 1000) : COURSE;
    if (!Array.isArray(list) || !list.length || list.length > 60) return { ok: false, why: Array.isArray(list) && list.length > 60 ? 'A course has 60 obstacles at most.' : 'Put at least one obstacle on the course.' };
    if (!list.every((q) => Array.isArray(q) && Number.isFinite(q[0]) && q[0] >= 6 && q[0] <= C - 3)) return { ok: false, why: `Every obstacle has to stand between 6 m and ${C - 3} m, on this ${C} m course.` };
    const xs = list.map((q) => +q[0]).sort((a, b) => a - b);
    let close = Infinity;
    for (let i = 1; i < xs.length; i++) {
      if (xs[i] - xs[i - 1] < 1) return { ok: false, why: `Two obstacles stand at ${xs[i].toFixed(1)} m, less than a metre apart: move one.` };
      close = Math.min(close, xs[i] - xs[i - 1]);
    }
    return { ok: true, close };
  };
  class Hurdles {
    constructor(o = {}) {
      // (a course of your own: its obstacles, the same every episode)
      this.own = o.obstacles === 'own' && E.courseCheck(o.obst, o.course).ok ? o.obst.map((q) => ({ x: +q[0], bar: !!q[1] })).sort((a, b) => a.x - b.x) : null;
      this.bars = o.obstacles === 'mixed' || !!(this.own && this.own.some((q) => q.bar)); this.view = 'state'; this.nA = 3; this.actions = ['run', 'jump', 'duck'];
      this.course = o.course > 0 ? clamp(o.course, 40, 1000) : COURSE; this.run = o.run > 0 ? clamp(o.run, 2, 20) : RUN; this.jumpV = o.jump > 0 ? clamp(o.jump, 2, 12) : JUMP;
      this.maxSteps = Math.ceil(this.course / (this.run * DTR)) + 10;
      this.input = { kind: 'vector', dim: 9 }; this.obsSize = 9;
      this.obsNames = ['next: how far', 'next: a hurdle', 'next: a bar', 'the one after: how far', 'the one after: a hurdle', 'its height off the ground', 'its speed up', 'ducking', 'on the ground'];
      this.rw = E.rewardTable('hurdles', o); this.rf = E.rewardFormula('hurdles', o.rFormula); E.sensors(this, 'hurdles', o);
    }
    reset(rng) {
      const r = rng || seeded(11);
      this.rng = rng; this.obst = [];
      if (this.own) this.obst = this.own.map((q) => ({ x: q.x, bar: q.bar }));
      else for (let x = 14 + r.uniform(0, 4); x < this.course - 4; x += r.uniform(7, 14)) this.obst.push({ x, bar: this.bars && r.next() < 0.4 });
      if ((this.level ?? 1) < 1) { const lim = (this.course - 4) * (0.25 + 0.75 * this.level); this.obst = this.obst.filter((q) => q.x < lim); } // (a curriculum's level: obstacles early on only)
      this.x = 0; this.y = 0; this.vy = 0; this.duck = 0; this.t = 0; this.end = null; this.jumps = 0; this.ducking = 0; this.hit = null;
      this.trace = [this.x, this.y, 0];
      return this.obs();
    }
    // the first obstacle not yet passed from x
    nextOf(x) { let i = 0; while (i < this.obst.length && this.obst[i].x < x - 0.3) i++; return i; }
    rawObs() {
      const i = this.nextOf(this.x), a = this.obst[i], b = this.obst[i + 1], near = (q) => (q ? clamp((q.x - this.x) / 15, 0, 1) : 1);
      return Float32Array.of(near(a), a && !a.bar ? 1 : 0, a && a.bar ? 1 : 0, near(b), b && !b.bar ? 1 : 0, this.y / 1.5, this.vy / this.jumpV, this.duck > 0 ? 1 : 0, this.y <= 0 ? 1 : 0);
    }
    step(a) {
      const ground = this.y <= 0 && this.vy <= 0, ahead = this.nextOf(this.x);
      let jumped = 0;
      if (a === 1 && ground) { this.vy = this.jumpV; this.jumps++; jumped = 1; }
      if (a === 2 && ground) this.duck = 4; else if (this.duck > 0) this.duck--;
      const x0 = this.x, y0 = this.y;
      if (this.y > 0 || this.vy > 0) { this.y += this.vy * DTR; this.vy -= GRAV * DTR; if (this.y <= 0) { this.y = 0; this.vy = 0; } }
      this.x += this.run * DTR; this.t++;
      if (this.duck > 0) this.ducking++;
      /* touching an obstacle: one it passed this step (between where it was and where it is now, so none is skipped
         however fast it runs), judged at its height as it passed: a hurdle with the feet below its top, a bar with
         the head above its underside */
      for (const q of this.obst) {
        if (q.x <= x0 || q.x > this.x) continue;
        const y = y0 + ((q.x - x0) / (this.x - x0)) * (this.y - y0), head = y + (this.duck > 0 ? LOW : TALL);
        if (q.bar ? head > BAR : y < HURDLE) { this.hit = q; break; }
      }
      if (this.t % 2 === 0) this.trace.push(this.x, this.y, this.duck > 0 ? 1 : 0);
      const finished = !this.hit && this.x >= this.course, done = !!this.hit || finished || this.t >= this.maxSteps;
      if (done) this.end = this.hit ? (this.hit.bar ? 'barhit' : 'hurdlehit') : finished ? 'course' : 'time';
      // an obstacle passed this step, untouched; the next one; its feet off the ground
      const rw = this.rw, hit = this.hit ? 1 : 0, next = this.nextOf(this.x), cleared = !hit && next > ahead ? 1 : 0, air = this.y > 0 ? 1 : 0, q = this.obst[next];
      const parts = this.rf ? null : {};
      let r = 0;
      if (this.rf) r = this.rf({ hit, finished: +finished, jumped, ducking: this.duck > 0 ? 1 : 0, height: this.y, dist: this.x, t: this.t, cleared, gap: q ? q.x - this.x : 99, air, bar: q && q.bar ? 1 : 0 });
      else {
        r = pay(parts, 'rRun', rw.rRun, r); if (hit) r = pay(parts, 'rBump', rw.rBump, r); if (finished) r = pay(parts, 'rFinish', rw.rFinish, r);
        if (jumped) r = pay(parts, 'rJump', rw.rJump, r); if (this.duck > 0) r = pay(parts, 'rDuck', rw.rDuck, r); if (cleared) r = pay(parts, 'rClear', rw.rClear, r); if (air) r = pay(parts, 'rAir', rw.rAir, r);
      }
      return { obs: this.obs(), r, done, parts };
    }
    snap() { return { x: this.x, y: this.y, vy: this.vy, duck: this.duck > 0, obst: this.obst, t: this.t, end: this.end, jumps: this.jumps, course: this.course }; }
    // how often it jumps (per obstacle passed) and how much of the time it ducks
    behavior() { const passed = Math.max(1, this.nextOf(this.x)); return [clamp(this.jumps / passed / 2, 0, 1), clamp(this.ducking / Math.max(1, this.t), 0, 1)]; }
    // the share of the course it runs, over 5 courses (the same 5 every time)
    score(policy) { const r = seeded(4242); let s = 0; for (let k = 0; k < 5; k++) { play(this, policy, seeded(r.int(1e9))); s += clamp(this.x / this.course, 0, 1); } return s / 5; }
  }

  // =====================================================================================================
  // a lamp to remember
  // =====================================================================================================
  /* A lamp at the start of a corridor shows which way to go at its end, left or right, at the first step only; then
     it goes dark. The robot walks down the corridor a square a step by itself (what it does there makes no
     difference); at the end, a junction, where it has to turn: the way the lamp said pays +1, the other way −1 (and
     so does standing there 5 steps). Seeing only what is in front of it now (view 'now') it cannot know by then which way the
     lamp said: whatever it learns, half its turns are wrong. View 'history': the last 16 readings, so a memory (a
     recurrent block) can carry the lamp down the corridor. Its score: the share of 20 corridors it turns right in
     (10 lamps each way). Settings: lit (how many steps the lamp shows, 1) and wait (the steps it has at the
     junction to turn, 5). */
  const HIST = 16, READ = ['lamp left', 'lamp right', 'at the junction', 'how far along'];
  class Memory {
    constructor(o = {}) {
      this.len = clamp(Math.round(o.length) || 6, 2, 14); this.view = o.view === 'history' ? 'history' : 'now';
      this.lit = o.lit > 0 ? clamp(Math.round(o.lit), 1, 14) : 1; this.wait = o.wait > 0 ? clamp(Math.round(o.wait), 1, 20) : 5;
      this.nA = 3; this.actions = ['walk on', 'turn left', 'turn right']; this.maxSteps = this.len + this.wait;
      if (this.view === 'history') { this.input = { kind: 'series', len: HIST, dim: 4 }; this.obsSize = HIST * 4; this.obsNames = NA.range(HIST * 4).map((i) => { const ago = HIST - 1 - Math.floor(i / 4); return `${READ[i % 4]}, ${ago ? ago + ' steps ago' : 'now'}`; }); }
      else { this.input = { kind: 'vector', dim: 4 }; this.obsSize = 4; this.obsNames = READ.slice(); }
      this.rw = E.rewardTable('memory', o); this.rf = E.rewardFormula('memory', o.rFormula); E.sensors(this, 'memory', o);
    }
    reset(rng, side) {
      const r = rng || seeded(13);
      this.rng = rng; this.side = side === 0 || side === 1 ? side : r.int(2); this.pos = 0; this.t = 0; this.end = null; this.choice = null; this.waits = 0;
      this.past = []; this.see();
      return this.obs();
    }
    reading() { return [this.t < this.lit && this.side === 0 ? 1 : 0, this.t < this.lit && this.side === 1 ? 1 : 0, this.pos >= this.len ? 1 : 0, this.pos / this.len]; }
    see() { this.past.push(this.reading()); if (this.past.length > HIST) this.past.shift(); }
    rawObs() {
      if (this.view === 'now') return Float32Array.from(this.past[this.past.length - 1]);
      const o = new Float32Array(HIST * 4), off = HIST - this.past.length;
      this.past.forEach((q, k) => o.set(q, (off + k) * 4));
      return o;
    }
    step(a) {
      // (early: a turn tried in the corridor, which does nothing; waited: a step at the junction without turning)
      let right = 0, wrong = 0, early = 0, waited = 0;
      if (this.pos < this.len) { this.pos++; if (a === 1 || a === 2) early = 1; }
      else if (a === 1 || a === 2) { this.choice = a === 1 ? 0 : 1; if (this.choice === this.side) right = 1; else wrong = 1; }
      else { waited = 1; this.waits++; }
      this.t++;
      const lost = !right && !wrong && this.t >= this.maxSteps;
      if (right || wrong || lost) this.end = right ? 'right' : wrong ? 'wrong' : 'lost';
      this.see();
      const rw = this.rw;
      const parts = this.rf ? null : {};
      let r = 0;
      if (this.rf) r = this.rf({ right, wrong, lost: +lost, junction: this.pos >= this.len ? 1 : 0, pos: this.pos, t: this.t, early, waited: this.waits, lamp: this.side ? 1 : -1, stood: waited });
      else {
        if (right) r = pay(parts, 'rRight', rw.rRight, r); if (wrong || lost) r = pay(parts, 'rWrong', rw.rWrong, r); r = pay(parts, 'rStep', rw.rStep, r);
        if (waited) r = pay(parts, 'rWait', rw.rWait, r); if (early) r = pay(parts, 'rEarly', rw.rEarly, r);
      }
      return { obs: this.obs(), r, done: !!this.end, parts };
    }
    snap() { return { len: this.len, pos: this.pos, side: this.side, t: this.t, lit: this.lit, choice: this.choice, end: this.end, past: this.past.map((q) => q.slice()), view: this.view }; }
    // which way it turned, and how long it took
    behavior() { return [this.choice === null ? 0.5 : this.choice, clamp(this.t / this.maxSteps, 0, 1)]; }
    // the share of 20 corridors it turns the right way in, the lamp left and right in turn
    score(policy) {
      let ok = 0;
      for (let k = 0; k < 20; k++) { let o = this.reset(null, k % 2), done = false; while (!done) ({ obs: o, done } = this.step(policy(o))); if (this.end === 'right') ok++; }
      return ok / 20;
    }
  }

  // =====================================================================================================
  // a cart in a valley
  // =====================================================================================================
  /* A cart in a valley (the "mountain car" of Moore, 1990): its engine is too weak to drive straight up the slope
     on the right to the flag, so it has to rock, first up the slope behind it to gain speed. A step costs −1 until
     it reaches the flag (at most 200 steps): an agent that never gets there sees −1 whatever it does, and has
     nothing to learn from; one that happens to rock knows at once. An energy bonus (shaping, off by default) pays
     for every gain in height plus speed. Its score: the share of 10 starts it reaches the flag from. start
     'anywhere': each episode starts at a random place and speed instead of at rest in the bottom (exploring starts). */
  const flagX = 0.5;
  class Mountain {
    constructor(o = {}) {
      this.maxSteps = o.steps > 0 ? Math.min(o.steps, 2000) : 200; this.power = o.engine === 'strong' ? 0.0013 : 0.001; this.anywhere = o.start === 'anywhere';
      this.view = 'state'; this.nA = 3; this.actions = ['push left', 'coast', 'push right'];
      this.input = { kind: 'vector', dim: 2 }; this.obsSize = 2; this.obsNames = ['position', 'speed'];
      this.rw = E.rewardTable('mountain', o); this.rf = E.rewardFormula('mountain', o.rFormula); E.sensors(this, 'mountain', o);
    }
    // height plus speed, as energy: 0 at rest in the bottom, about 1.7 at rest at the flag
    energy() { return 1000 * (0.5 * this.v * this.v + (0.0025 / 3) * (Math.sin(3 * this.x) + 1)); }
    reset(rng) {
      const r = rng || seeded(17);
      this.rng = rng;
      if (this.anywhere) { this.x = r.uniform(-1.2, 0.45); this.v = r.uniform(-0.07, 0.07); } else { this.x = r.uniform(-0.6, -0.4); this.v = 0; }
      this.t = 0; this.end = null; this.lo = this.x; this.hi = this.x; this.push = 0;
      this.phi = this.energy(); this.trace = [this.x, this.v, 0];
      return this.obs();
    }
    rawObs() { return Float32Array.of((this.x + 0.3) / 0.9, this.v / 0.07); }
    step(a) {
      this.push = a === 0 ? -1 : a === 2 ? 1 : 0;
      this.v = clamp(this.v + this.push * this.power - 0.0025 * Math.cos(3 * this.x), -0.07, 0.07);
      this.x += this.v;
      let wall = 0;
      if (this.x < -1.2) { this.x = -1.2; this.v = 0; wall = 1; }
      this.t++; this.lo = Math.min(this.lo, this.x); this.hi = Math.max(this.hi, this.x);
      if (this.t % 2 === 0) this.trace.push(this.x, this.v, this.push);
      const goal = this.x >= flagX, done = goal || this.t >= this.maxSteps;
      if (done) this.end = goal ? 'flag' : 'lost';
      const phi = this.energy(), gained = phi - this.phi, height = Math.sin(3 * this.x);
      this.phi = phi;
      const rw = this.rw;
      const parts = this.rf ? null : {};
      let r = 0;
      if (this.rf) r = this.rf({ goal: +goal, x: this.x, speed: this.v * 100, height, energy: phi, gained, push: this.push, t: this.t, pushed: this.push ? 1 : 0, farthest: this.hi, wall });
      else {
        r = pay(parts, 'rTick', rw.rTick, r); if (goal) r = pay(parts, 'rFlag', rw.rFlag, r); r = pay(parts, 'rEnergy', rw.rEnergy * gained, r);
        if (this.push) r = pay(parts, 'rPush', rw.rPush, r); if (rw.rHeight) r = pay(parts, 'rHeight', rw.rHeight * height, r);
      }
      return { obs: this.obs(), r, done, parts };
    }
    snap() { return { x: this.x, v: this.v, push: this.push, t: this.t, maxSteps: this.maxSteps, end: this.end, energy: this.phi, trail: this.trace.slice(-90) }; }
    // how far left and how far right it went
    behavior() { return [clamp((this.lo + 1.2) / 0.7, 0, 1), clamp((this.hi + 1.2) / 1.8, 0, 1)]; }
    // the share of 10 starts it reaches the flag from (the same 10 every time)
    score(policy) { const r = seeded(4242); let ok = 0; for (let k = 0; k < 10; k++) { play(this, policy, seeded(r.int(1e9))); if (this.end === 'flag') ok++; } return ok / 10; }
  }

  // =====================================================================================================
  // a pendulum to swing up
  // =====================================================================================================
  /* A pendulum on a motor: a 1 m pole with its weight at the end, hanging down at the start. The motor is too weak
     to lift it straight up (4 against the 9.8 the weight pulls with; a weak one 2.5), so it has to swing it higher and higher,
     pushing with each swing, then catch it upside down and balance it there. One step is 0.05 s, 300 of them.
     Rewards: the weight's height every step (+1 straight up, −1 hanging), times a small amount. Its score: the
     share of the last 150 steps it stays within 15° of upright, over 3 starts. Settings: steps (300), friction
     (0.1) and start ('down' hanging, 'anywhere', or 'up': only the balancing is left to learn). */
  class SwingUp {
    constructor(o = {}) {
      this.torque = o.motor === 'weak' ? 2.5 : 4; this.g = o.gravity > 0 ? o.gravity : 9.8; this.maxSteps = o.steps > 0 ? clamp(Math.round(o.steps), 50, 3000) : 300;
      this.fric = typeof o.friction === 'number' && o.friction >= 0 ? Math.min(o.friction, 3) : 0.1; this.start = o.start === 'up' || o.start === 'anywhere' ? o.start : 'down';
      this.view = 'state'; this.nA = 3; this.actions = ['turn left', 'rest', 'turn right'];
      this.input = { kind: 'vector', dim: 3 }; this.obsSize = 3; this.obsNames = ['angle (sine)', 'angle (cosine)', 'turning speed'];
      this.rw = E.rewardTable('swingup', o); this.rf = E.rewardFormula('swingup', o.rFormula); E.sensors(this, 'swingup', o);
    }
    reset(rng) {
      const r = rng || seeded(19);
      this.rng = rng; this.th = this.start === 'up' ? r.uniform(-0.2, 0.2) : this.start === 'anywhere' ? r.uniform(-Math.PI, Math.PI) : Math.PI + r.uniform(-0.2, 0.2); this.w = r.uniform(-0.3, 0.3); this.t = 0; this.tq = 0; this.end = null;
      this.up = 0; this.upLate = 0; this.first = null; this.spin = 0; this.pushed = 0; this.trace = [this.th, this.w, 0];
      return this.obs();
    }
    rawObs() { return Float32Array.of(Math.sin(this.th), Math.cos(this.th), this.w / 8); }
    // its energy: 1 when it has just enough swing to reach the top, −1 hanging still
    energy() { return (0.5 * this.w * this.w + this.g * Math.cos(this.th)) / this.g; }
    step(a) {
      const dt = 0.05;
      this.tq = (a === 0 ? -1 : a === 2 ? 1 : 0) * this.torque;
      this.w = clamp(this.w + (this.g * Math.sin(this.th) + this.tq - this.fric * this.w) * dt, -12, 12);
      this.th = wrapAngle(this.th + this.w * dt);
      this.t++;
      const upright = Math.abs(this.th) < 15 * DEG;
      if (upright) { this.up++; if (this.first === null) this.first = this.t; if (this.t > this.maxSteps / 2) this.upLate++; }
      this.spin += Math.abs(this.w);
      if (this.t % 2 === 0) this.trace.push(this.th, this.w, this.tq);
      const done = this.t >= this.maxSteps;
      if (done) this.end = this.upLate >= 0.8 * (this.maxSteps / 2) ? 'balanced' : this.first !== null ? 'swung' : 'hanging';
      // the motor pushing the other way from its last push
      const h = Math.cos(this.th), rw = this.rw, switched = this.tq !== 0 && this.pushed !== 0 && Math.sign(this.tq) !== this.pushed;
      if (this.tq) this.pushed = Math.sign(this.tq);
      const parts = this.rf ? null : {};
      let r = 0;
      if (this.rf) r = this.rf({ height: h, angle: this.th / DEG, spin: this.w / DEG, upright: +upright, push: Math.sign(this.tq), t: this.t, energy: this.energy(), switched: +switched });
      else {
        r = pay(parts, 'rHeight', rw.rHeight * h, r); if (upright) r = pay(parts, 'rUpright', rw.rUpright, r); if (this.tq) r = pay(parts, 'rPush', rw.rPush, r);
        if (rw.rSpin) r = pay(parts, 'rSpin', rw.rSpin * Math.abs(this.w / DEG), r); if (switched) r = pay(parts, 'rSwitch', rw.rSwitch, r);
      }
      return { obs: this.obs(), r, done, parts };
    }
    snap() { return { th: this.th, w: this.w, tq: this.tq, torque: this.torque, t: this.t, up: this.up, end: this.end, trail: this.trace.slice(-60) }; }
    // how soon it first gets it upright, and how fast it swings
    behavior() { return [this.first === null ? 1 : clamp(this.first / this.maxSteps, 0, 1), clamp(this.spin / Math.max(1, this.t) / 8, 0, 1)]; }
    score(policy) { const r = seeded(4242); let s = 0; for (let k = 0; k < 3; k++) { play(this, policy, seeded(r.int(1e9))); s += this.upLate / (this.maxSteps / 2); } return s / 3; }
  }

  // =====================================================================================================
  // a robot arm
  // =====================================================================================================
  /* A two-joint arm on a bench (a 1 m upper arm, a 0.8 m forearm) and a target somewhere it can reach: it has to
     bring its hand onto the target and stop there. Each step pushes one joint one way or the other, or brakes
     both; the joints carry on turning, slowed by friction. The shoulder sweeps above the bench, the elbow bends
     150° either way, and the hand never goes below the bench. Rewards: +1 on reaching the target (the hand within
     12 cm and moving slowly) and, by default, every step the change in its distance to the target (shaping). Its
     score: the share of 10 targets it reaches within 120 steps. */
  const L1 = 1, L2 = 0.8;
  class Arm {
    constructor(o = {}) {
      this.maxSteps = o.steps > 0 ? clamp(Math.round(o.steps), 20, 2000) : 120; this.view = 'state'; this.nA = 5; this.actions = ['shoulder ↺', 'shoulder ↻', 'elbow ↺', 'elbow ↻', 'brake'];
      this.dA = 3; this.cActions = ['shoulder', 'elbow', 'brake']; // (continuous: each motor from −1, full ↻, to 1, full ↺, both at once; the brake from 0, off, to 1, full)
      // how near counts as reaching it (12 cm), and how hard its motors push (6 rad/s²)
      this.reach = o.reach > 0 ? clamp(o.reach, 0.02, 0.5) : 0.12; this.acc = o.motors > 0 ? clamp(o.motors, 1, 30) : 6;
      this.input = { kind: 'vector', dim: 10 }; this.obsSize = 10;
      this.obsNames = ['shoulder angle (sine)', 'shoulder angle (cosine)', 'elbow angle (sine)', 'elbow angle (cosine)', 'shoulder speed', 'elbow speed', 'target from the hand, across', 'target from the hand, up', 'target across', 'target up'];
      this.rw = E.rewardTable('arm', o); this.rf = E.rewardFormula('arm', o.rFormula); E.sensors(this, 'arm', o);
    }
    hand(q1 = this.q1, q2 = this.q2) { return [L1 * Math.cos(q1) + L2 * Math.cos(q1 + q2), L1 * Math.sin(q1) + L2 * Math.sin(q1 + q2)]; }
    reset(rng) {
      const r = rng || seeded(23);
      this.rng = rng; this.q1 = r.uniform(0.4, 2.7); this.q2 = r.uniform(-1.2, 1.2); this.w1 = 0; this.w2 = 0; this.t = 0; this.end = null; this.path = 0; this.bend = 0; this.a = null;
      const R = r.uniform(0.6, 1.65), A = r.uniform(0.3, Math.PI - 0.3);
      this.tx = R * Math.cos(A); this.ty = R * Math.sin(A);
      if (this.hand()[1] < 0.05) this.q2 = 0;
      // (a curriculum's level: the target nearer the hand at first)
      if ((this.level ?? 1) < 1) { const [hx, hy] = this.hand(), k = 0.3 + 0.7 * this.level; this.tx = hx + (this.tx - hx) * k; this.ty = Math.max(0.15, hy + (this.ty - hy) * k); }
      [this.hx, this.hy] = this.hand(); this.d = Math.hypot(this.tx - this.hx, this.ty - this.hy);
      this.trace = [this.q1, this.q2, 0];
      return this.obs();
    }
    rawObs() { return Float32Array.of(Math.sin(this.q1), Math.cos(this.q1), Math.sin(this.q2), Math.cos(this.q2), this.w1 / 3, this.w2 / 3, (this.tx - this.hx) / 2, (this.ty - this.hy) / 2, this.tx / 2, this.ty / 2); }
    step(a) {
      const dt = 0.05, acc = this.acc, ca = typeof a === 'number' ? null : a;
      if (ca) a = ca[2] > 0.5 ? 4 : Math.abs(ca[0]) >= Math.abs(ca[1]) ? (ca[0] >= 0 ? 0 : 1) : ca[1] >= 0 ? 2 : 3; // (the nearest move)
      this.a = a;
      if (ca) { const k = 1 - 0.5 * clamp(ca[2], 0, 1); this.w1 = (this.w1 + clamp(ca[0], -1, 1) * acc * dt) * k; this.w2 = (this.w2 + clamp(ca[1], -1, 1) * acc * dt) * k; }
      else if (a === 0) this.w1 += acc * dt; else if (a === 1) this.w1 -= acc * dt; else if (a === 2) this.w2 += acc * dt; else if (a === 3) this.w2 -= acc * dt; else { this.w1 *= 0.5; this.w2 *= 0.5; }
      this.w1 *= 0.94; this.w2 *= 0.94;
      // (bump: a joint at the end of its swing, or the hand stopped by the bench)
      let q1 = this.q1 + this.w1 * dt, q2 = this.q2 + this.w2 * dt, bump = 0;
      if (q1 < 0 || q1 > Math.PI) { q1 = clamp(q1, 0, Math.PI); this.w1 = 0; bump = 1; }
      if (Math.abs(q2) > 2.6) { q2 = Math.sign(q2) * 2.6; this.w2 = 0; bump = 1; }
      if (this.hand(q1, q2)[1] < 0.02) { q1 = this.q1; q2 = this.q2; this.w1 = 0; this.w2 = 0; bump = 1; } // the bench stops it
      this.q1 = q1; this.q2 = q2;
      const [hx, hy] = this.hand(), moved = Math.hypot(hx - this.hx, hy - this.hy), d = Math.hypot(this.tx - hx, this.ty - hy), closer = this.d - d;
      this.hx = hx; this.hy = hy; this.d = d; this.path += moved; this.t++; this.bend += Math.abs(this.q2);
      const reached = d < this.reach && moved / dt < 0.6, done = reached || this.t >= this.maxSteps;
      if (this.t % 2 === 0) this.trace.push(this.q1, this.q2, d);
      if (done) this.end = reached ? 'reached' : 'lost';
      const rw = this.rw, pushed = ca ? Math.max(Math.abs(clamp(ca[0], -1, 1)), Math.abs(clamp(ca[1], -1, 1))) : a >= 0 && a <= 3 ? 1 : 0, braked = ca ? clamp(ca[2], 0, 1) : a === 4 ? 1 : 0, parts = this.rf ? null : {};
      let r = 0;
      if (this.rf) r = this.rf({ reached: +reached, dist: d, closer, speed: moved / dt, elbow: this.q2 / DEG, t: this.t, shoulder: this.q1 / DEG, bump, braked, hx, hy });
      else {
        if (reached) r = pay(parts, 'rReach', rw.rReach, r); r = pay(parts, 'rApproach', rw.rApproach * closer, r); r = pay(parts, 'rStep', rw.rStep, r);
        if (pushed) r = pay(parts, 'rPush', rw.rPush * pushed, r); if (braked) r = pay(parts, 'rBrake', rw.rBrake * braked, r); if (bump) r = pay(parts, 'rBump', rw.rBump, r);
      }
      return { obs: this.obs(), r, done, parts };
    }
    snap() { return { q1: this.q1, q2: this.q2, tx: this.tx, ty: this.ty, hx: this.hx, hy: this.hy, d: this.d, a: this.a, t: this.t, end: this.end, reach: this.reach, trail: this.trace.slice(-90) }; }
    // how bent it keeps its elbow, and how far its hand travels
    behavior() { const n = Math.max(1, this.t); return [clamp(this.bend / n / 2.6, 0, 1), clamp(this.path / 4, 0, 1)]; }
    score(policy) { const r = seeded(4242); let ok = 0; for (let k = 0; k < 10; k++) { play(this, policy, seeded(r.int(1e9))); if (this.end === 'reached') ok++; } return ok / 10; }
  }

  for (const W of [Thermo, Hurdles, Memory, Mountain, SwingUp, Arm]) W.prototype.obs = E.sensed;

  // ------------------------------------------------------------------ rewards, inputs, the catalogue
  Object.assign(E.REWARDS, {
    thermo: [{ key: 'rOff', label: 'Each degree from the target, every step', def: -0.1 }, { key: 'rHeat', label: 'Each step the heating is on', def: -0.01 }, { key: 'rBand', label: 'Each step it is comfortable (within the band)', def: 0 },
      { key: 'rCold', label: 'Each degree too cold, every step', def: 0 }, { key: 'rHot', label: 'Each degree too warm, every step', def: 0 }, { key: 'rSwitch', label: 'Each time it switches the heating on or off', def: 0 }],
    hurdles: [{ key: 'rRun', label: 'Every step it keeps running', def: 0.01 }, { key: 'rBump', label: 'Hitting a hurdle or a bar', def: -1 }, { key: 'rFinish', label: 'Reaching the finish', def: 1 },
      { key: 'rClear', label: 'Each obstacle passed', def: 0 }, { key: 'rJump', label: 'Each jump', def: 0 }, { key: 'rDuck', label: 'Each step it ducks', def: 0 }, { key: 'rAir', label: 'Each step in the air', def: 0 }],
    memory: [{ key: 'rRight', label: 'Turning the way the lamp said', def: 1 }, { key: 'rWrong', label: 'Turning the other way, or running out of time', def: -1 }, { key: 'rStep', label: 'Every step', def: 0 },
      { key: 'rWait', label: 'Each step it waits at the junction', def: 0 }, { key: 'rEarly', label: 'Trying to turn before the junction', def: 0 }],
    mountain: [{ key: 'rTick', label: 'Every step until the flag', def: -1 }, { key: 'rFlag', label: 'Reaching the flag', def: 0 }, { key: 'rEnergy', label: 'Gaining height or speed (shaping)', def: 0 },
      { key: 'rHeight', label: 'Each step, by its height (−1 to 1)', def: 0 }, { key: 'rPush', label: 'Each push of the engine', def: 0 }],
    swingup: [{ key: 'rHeight', label: 'Height of the weight, every step (+1 up, −1 down)', def: 1 }, { key: 'rUpright', label: 'Each step within 15° of upright', def: 0 }, { key: 'rPush', label: 'Each step the motor pushes', def: 0 },
      { key: 'rSpin', label: 'Each step, per degree a second it turns', def: 0 }, { key: 'rSwitch', label: 'Each change in the motor’s direction', def: 0 }],
    arm: [{ key: 'rReach', label: 'Reaching the target', def: 1 }, { key: 'rApproach', label: 'Getting closer to it, per metre (shaping)', def: 1 }, { key: 'rStep', label: 'Every step', def: 0 },
      { key: 'rPush', label: 'Each push of a joint', def: 0 }, { key: 'rBrake', label: 'Each brake', def: 0 }, { key: 'rBump', label: 'Hitting the bench or a joint’s limit', def: 0 }],
  });
  Object.assign(E.REWARD_VARS, {
    thermo: [['temp', 'the room’s temperature, °C'], ['target', 'the temperature wanted now, °C'], ['off', 'how far the room is from the target, °C'], ['inside', '1 while the room is comfortable: within the band (1 °C) of the target'],
      ['outside', 'the temperature outside, °C'], ['heating', '1 while the heating is on'], ['radiator', 'the radiator’s temperature, °C'], ['hour', 'the time of day, 0 to 24'], ['t', 'steps so far (10 minutes each)'],
      ['cold', 'how many degrees too cold (0 when warm enough)'], ['hot', 'how many degrees too warm (0 when not)'], ['switched', '1 when the heating was switched on or off']],
    hurdles: [['hit', '1 on hitting a hurdle or a bar'], ['finished', '1 on reaching the finish'], ['jumped', '1 on taking off'], ['ducking', '1 while it ducks'], ['height', 'its feet off the ground, m'], ['dist', 'metres run'], ['t', 'steps so far'],
      ['cleared', '1 on passing an obstacle'], ['gap', 'how far the next obstacle is, m (99 when none is left)'], ['air', '1 while its feet are off the ground'], ['bar', '1 when the next obstacle is a bar']],
    memory: [['right', '1 on turning the way the lamp said'], ['wrong', '1 on turning the other way'], ['lost', '1 on running out of time'], ['junction', '1 while it stands at the junction'], ['pos', 'squares along the corridor'], ['t', 'steps so far'],
      ['early', '1 on trying to turn before the junction'], ['waited', 'steps it has stood at the junction so far'], ['stood', '1 on a step it stands at the junction without turning'], ['lamp', 'the way the lamp said: −1 left, 1 right']],
    mountain: [['goal', '1 on reaching the flag'], ['x', 'its position: −1.2 the far left, 0.5 the flag'], ['speed', 'its speed (to the right is more than 0)'], ['height', 'how high up it is, −1 to 1'],
      ['energy', 'its height plus speed, as energy'], ['gained', 'the energy it gained this step'], ['push', '1 pushing right, −1 left'], ['t', 'steps so far'],
      ['pushed', '1 while the engine pushes'], ['farthest', 'the furthest right it has been'], ['wall', '1 on hitting the far left end']],
    swingup: [['height', 'the weight’s height: 1 straight up, −1 hanging'], ['angle', 'its angle from upright, degrees'], ['spin', 'how fast it turns, degrees a second'], ['upright', '1 while within 15° of upright'], ['push', '1 or −1 while the motor pushes'], ['t', 'steps so far'],
      ['energy', 'its swing: 1 with just enough to reach the top, −1 hanging still'], ['switched', '1 when the motor pushed the other way from last time']],
    arm: [['reached', '1 on reaching the target'], ['dist', 'the hand’s distance to the target, m'], ['closer', 'how much closer it got this step, m'], ['speed', 'the hand’s speed, m/s'], ['elbow', 'the elbow’s bend, degrees'], ['t', 'steps so far'],
      ['shoulder', 'the shoulder’s angle, degrees (0 pointing right, 180 left)'], ['bump', '1 on hitting the bench or a joint’s limit'], ['braked', '1 on braking'], ['hx', 'the hand across, m'], ['hy', 'the hand’s height above the bench, m']],
  });
  // the extra inputs these can be given (each is seen as a list of numbers; the lamp's history is not)
  Object.assign(E.EXTRAS, {
    thermo: [...E.COMMON_EXTRAS,
      { name: 'the target in an hour', what: 'the temperature that will be wanted an hour from now (it changes when people come in)', n: 1, names: ['the target in an hour'], fill: (w, o, k) => { o[k] = (w.target(((w.t + 6) / 6) % 24) - 18) / 5; } },
      { name: 'the weather in an hour', what: 'the temperature outside an hour from now: a forecast', n: 1, names: ['outside in an hour'], fill: (w, o, k) => { o[k] = (w.outside(w.t + 6) - 10) / 10; } }],
    hurdles: [...E.COMMON_EXTRAS,
      { name: 'how far it has run', what: 'how much of the course it has run, 0 to 1', n: 1, names: ['how far it has run'], fill: (w, o, k) => { o[k] = clamp(w.x / w.course, 0, 1); } },
      { name: 'the one after: a bar', what: 'whether the obstacle after the next is a bar (it is told only whether that one is a hurdle)', n: 1, names: ['the one after: a bar'], fill: (w, o, k) => { const q = w.obst[w.nextOf(w.x) + 1]; o[k] = q && q.bar ? 1 : 0; } }],
    memory: [...E.COMMON_EXTRAS,
      { name: 'what the lamp said', what: 'the way the lamp said, at every step: the answer it would otherwise have to remember', n: 2, names: ['the lamp said left', 'the lamp said right'], fill: (w, o, k) => { o[k + w.side] = 1; } },
      { name: 'steps waited', what: 'how long it has stood at the junction, as a share of the time it has there', n: 1, names: ['steps waited'], fill: (w, o, k) => { o[k] = w.waits / w.wait; } }],
    mountain: [...E.COMMON_EXTRAS,
      { name: 'its height', what: 'how high up the valley it is, −1 to 1', n: 1, names: ['its height'], fill: (w, o, k) => { o[k] = Math.sin(3 * w.x); } },
      { name: 'its energy', what: 'its height plus speed as energy: 0 at rest in the bottom, about 1 at rest by the flag', n: 1, names: ['its energy'], fill: (w, o, k) => { o[k] = w.energy() / 1.7; } }],
    swingup: [...E.COMMON_EXTRAS,
      { name: 'its energy', what: 'its swing: 1 with just enough to reach the top, −1 hanging still. Swinging it up is raising this to 1', n: 1, names: ['its energy'], fill: (w, o, k) => { o[k] = w.energy(); } },
      { name: 'how far from upright', what: 'its angle from upright, 0 up to 1 hanging, whichever way', n: 1, names: ['how far from upright'], fill: (w, o, k) => { o[k] = Math.abs(w.th) / Math.PI; } }],
    arm: [...E.COMMON_EXTRAS,
      { name: 'the hand', what: 'where the hand is, across and up', n: 2, names: ['the hand across', 'the hand up'], fill: (w, o, k) => { o[k] = w.hx / 2; o[k + 1] = w.hy / 2; } },
      { name: 'distance to the target', what: 'how far the hand is from the target', n: 1, names: ['distance to the target'], fill: (w, o, k) => { o[k] = w.d / 2; } }],
  });
  Object.assign(E.INPUT_GROUPS, {
    thermo: { state: [['room temperature', [0]], ['outside temperature', [1]], ['radiator temperature', [2]], ['the target', [3]], ['time of day', [4, 5]], ['heating on', [6]]] },
    hurdles: { state: [['the next obstacle', [0, 1, 2]], ['the one after', [3, 4]], ['its height and speed', [5, 6]], ['ducking, on the ground', [7, 8]]] },
    memory: { now: [['the lamp', [0, 1]], ['the junction', [2]], ['how far along', [3]]], history: [['the lamp', NA.range(64).filter((i) => i % 4 < 2)], ['the junction', NA.range(64).filter((i) => i % 4 === 2)], ['how far along', NA.range(64).filter((i) => i % 4 === 3)]] },
    mountain: { state: [['position', [0]], ['speed', [1]]] },
    swingup: { state: [['angle', [0, 1]], ['turning speed', [2]]] },
    arm: { state: [['joint angles', [0, 1, 2, 3]], ['joint speeds', [4, 5]], ['target from the hand', [6, 7]], ['target', [8, 9]]] },
  });
  Object.assign(E.WORLDS, {
    thermo: { name: 'Keep a room warm', make: (o) => new Thermo(o), what: 'the room, outside and radiator temperatures, the target, the time and the heating', goal: 'keep the room within 1 °C of the target all day', axes: ['how much it heats', 'how warm it keeps the room'], demoMs: 60, trace: true },
    hurdles: { name: 'Hurdles', make: (o) => new Hurdles(o), what: 'how far the next two obstacles are and what they are, its height, speed and stance', goal: 'run the whole course without touching anything', axes: ['how often it jumps', 'how much it ducks'], demoMs: 45, trace: true },
    memory: { name: 'Remember the lamp', make: (o) => new Memory(o), what: 'the lamp (at the first step only), the junction and how far along it is, or the last 16 of these', goal: 'turn at the junction the way the lamp said', axes: ['which way it turns', 'how long it takes'], demoMs: 260, trace: false },
    mountain: { name: 'Mountain cart', make: (o) => new Mountain(o), what: 'its position and speed', goal: 'reach the flag on the right-hand hill', axes: ['how far left it goes', 'how far right it goes'], demoMs: 30, trace: true },
    swingup: { name: 'Swing up a pendulum', make: (o) => new SwingUp(o), what: 'the pendulum’s angle and how fast it turns', goal: 'swing it up and keep it upright', axes: ['how soon it gets it up', 'how fast it swings'], demoMs: 35, trace: true },
    arm: { name: 'Robot arm', cont: true, make: (o) => new Arm(o), what: 'its joint angles and speeds, and where the target is', goal: 'bring the hand onto the target and stop there', axes: ['how bent it keeps its elbow', 'how far its hand travels'], demoMs: 45, trace: true },
  });

  // =====================================================================================================
  // pictures
  // =====================================================================================================
  const ink = (T, a) => T.c('ink', a), clip = (ctx, r, rad = 10) => { rounded(ctx, r.x, r.y, r.w, r.h, rad); ctx.clip(); };
  // a picture's shape: the largest box of the given width to height inside r, centred
  const fitBox = (r, aspect) => { let w = r.w, h = r.h; if (w / h > aspect) w = h * aspect; else h = w / aspect; return { x: r.x + (r.w - w) / 2, y: r.y + (r.h - h) / 2, w, h }; };

  // ------------------------------------------------------------------ the room
  /* The room in a cut-away house (the sky outside by the time of day, the radiator glowing as hot as it is, a
     thermometer), and the day so far as a chart: the room's temperature, the outside, the target band, and when
     the heating was on. o.ghosts: a generation, each a dot at its room temperature now. */
  function drawThermo(ctx, r, s, T, o = {}) {
    const wide = r.w > r.h * 1.25, hr = wide ? { x: r.x, y: r.y, w: r.w * 0.44, h: r.h } : { x: r.x, y: r.y, w: r.w, h: r.h * 0.5 }, cr = wide ? { x: r.x + r.w * 0.47, y: r.y + 6, w: r.w * 0.53, h: r.h - 12 } : { x: r.x, y: r.y + r.h * 0.53, w: r.w, h: r.h * 0.47 };
    const H = fitBox(hr, 1.05), h = ((s.t / 6) % 24), day = clamp(Math.cos((TAU * (h - 13)) / 24) * 1.6 + 0.4, 0, 1);
    ctx.save(); clip(ctx, H);
    // the sky by the time of day, the sun or the moon on its arc
    const night = [10, 18, 42], noon = [120, 172, 225], sky = mix(night, noon, day), g0 = ctx.createLinearGradient(0, H.y, 0, H.y + H.h);
    g0.addColorStop(0, css(mix(sky, [0, 0, 0], 0.15))); g0.addColorStop(1, css(mix(sky, [255, 220, 190], 0.25 * day)));
    ctx.fillStyle = g0; ctx.fillRect(H.x, H.y, H.w, H.h);
    const sa = Math.PI * (1 - (((h - 6 + 24) % 24) / 12)), isSun = h >= 6 && h < 18, ax = H.x + H.w / 2 + Math.cos(isSun ? sa : sa + Math.PI) * H.w * 0.42, ay = H.y + H.h * 0.42 - Math.sin(isSun ? sa : sa + Math.PI) * H.h * 0.32;
    ctx.fillStyle = isSun ? 'rgba(255,214,120,0.95)' : 'rgba(230,236,255,0.85)'; ctx.beginPath(); ctx.arc(ax, ay, H.w * 0.035, 0, TAU); ctx.fill();
    // the house: walls, roof, a window onto the sky, the floor
    const hx = H.x + H.w * 0.12, hw = H.w * 0.76, hy = H.y + H.h * 0.36, hh = H.h * 0.56, wall = rgbOf(T, 'panel'), line = rgbOf(T, 'line-2');
    ctx.fillStyle = css(mix(wall, [0, 0, 0], 0.08)); ctx.beginPath(); ctx.moveTo(hx - H.w * 0.05, hy); ctx.lineTo(hx + hw / 2, H.y + H.h * 0.14); ctx.lineTo(hx + hw + H.w * 0.05, hy); ctx.closePath(); ctx.fill();
    ctx.fillStyle = css(mix(wall, [255, 255, 255], 0.04)); ctx.fillRect(hx, hy, hw, hh);
    // the room's air, warmer is redder
    const warm = clamp((s.T - 14) / 12, 0, 1); ctx.fillStyle = css(mix([90, 140, 230], [235, 120, 70], warm), 0.12); ctx.fillRect(hx + 6, hy + 6, hw - 12, hh - 12);
    const wx = hx + hw * 0.55, wy = hy + hh * 0.16, ww = hw * 0.32, wh = hh * 0.34;
    ctx.fillStyle = css(sky); ctx.fillRect(wx, wy, ww, wh); ctx.strokeStyle = css(line); ctx.lineWidth = 3; ctx.strokeRect(wx, wy, ww, wh); ctx.beginPath(); ctx.moveTo(wx + ww / 2, wy); ctx.lineTo(wx + ww / 2, wy + wh); ctx.moveTo(wx, wy + wh / 2); ctx.lineTo(wx + ww, wy + wh / 2); ctx.stroke();
    ctx.fillStyle = css(mix(wall, [0, 0, 0], 0.18)); ctx.fillRect(hx, hy + hh - 6, hw, 6);
    // the radiator, glowing as hot as it is
    const heat = clamp((s.R - 20) / 40, 0, 1), rx = wx - ww * 0.05, ry = hy + hh * 0.62, rw2 = ww * 1.1, rh = hh * 0.26, col = mix([150, 158, 172], [240, 96, 54], heat);
    if (heat > 0.05) { const gl = ctx.createRadialGradient(rx + rw2 / 2, ry + rh / 2, 2, rx + rw2 / 2, ry + rh / 2, rw2); gl.addColorStop(0, css([255, 140, 80], 0.35 * heat)); gl.addColorStop(1, css([255, 140, 80], 0)); ctx.fillStyle = gl; ctx.fillRect(rx - rw2 / 2, ry - rh, rw2 * 2, rh * 3); }
    for (let k = 0; k < 7; k++) { ctx.fillStyle = css(mix(col, [255, 255, 255], k % 2 ? 0.08 : 0)); rounded(ctx, rx + (k * rw2) / 7 + 1, ry, rw2 / 7 - 2, rh, 3); ctx.fill(); }
    // the thermometer and what it reads
    const tx = hx + hw * 0.16, ty = hy + hh * 0.18, th = hh * 0.6, fill = clamp((s.T - 10) / 20, 0, 1);
    ctx.fillStyle = 'rgba(255,255,255,0.75)'; rounded(ctx, tx - 5, ty, 10, th, 5); ctx.fill();
    ctx.fillStyle = css(mix([90, 150, 255], [235, 90, 60], warm)); rounded(ctx, tx - 3, ty + th * (1 - fill), 6, th * fill, 3); ctx.fill();
    ctx.beginPath(); ctx.arc(tx, ty + th + 4, 8, 0, TAU); ctx.fill();
    const band = (v) => ty + th * (1 - clamp((v - 10) / 20, 0, 1));
    ctx.strokeStyle = T.c('good'); ctx.lineWidth = 2; for (const v of [s.target - (s.band || 1), s.target + (s.band || 1)]) { ctx.beginPath(); ctx.moveTo(tx + 7, band(v)); ctx.lineTo(tx + 14, band(v)); ctx.stroke(); }
    label(ctx, `${s.T.toFixed(1)} °C`, tx + 18, ty + th * 0.15, Math.max(11, H.w * 0.05), T.c('ink'), 'left', 700);
    label(ctx, `want ${s.target} °C`, tx + 18, ty + th * 0.15 + Math.max(13, H.w * 0.05), 10.5, T.c('ink-2'), 'left', 600);
    label(ctx, `${String(Math.floor(h)).padStart(2, '0')}:${String(Math.floor((h % 1) * 60)).padStart(2, '0')} · outside ${s.To.toFixed(1)} °C`, H.x + 10, H.y + 14, 11, day > 0.5 ? 'rgba(10,20,40,0.85)' : 'rgba(255,255,255,0.85)', 'left', 650);
    label(ctx, s.heat ? 'heating on' : 'heating off', H.x + H.w - 10, H.y + 14, 11, s.heat ? 'rgba(255,170,110,0.95)' : (day > 0.5 ? 'rgba(10,20,40,0.6)' : 'rgba(255,255,255,0.6)'), 'right', 700);
    ctx.restore();
    // the day so far
    drawDayChart(ctx, cr, s, T, o);
    if (s.end && !o.ghostsOnly) banner(ctx, r, s.end === 'comfy' ? `Comfortable ${Math.round(s.comfy * 100)}% of the day` : `Comfortable only ${Math.round(s.comfy * 100)}% of the day`, s.end === 'comfy' ? T.c('good') : T.c('critical'));
  }
  function drawDayChart(ctx, r, s, T, o) {
    // (a run of several days: the whole run, a label every 12 or 24 hours)
    const span = s.maxSteps || 144, days = span / 144, warm = s.warm || 21, band = s.band || 1;
    const lo = -12, hi = 30, X = (t) => r.x + 30 + ((r.w - 36) * t) / span, Y = (v) => r.y + 8 + (r.h - 30) * (1 - (v - lo) / (hi - lo)), n = s.hist.length / 3;
    ctx.save();
    ctx.strokeStyle = T.c('line'); ctx.lineWidth = 1;
    for (const v of [0, 10, 20, 30]) { ctx.beginPath(); ctx.moveTo(X(0), Y(v)); ctx.lineTo(X(span), Y(v)); ctx.stroke(); label(ctx, `${v}°`, r.x + 26, Y(v), 10.5, T.c('muted'), 'right', 500); }
    for (let hh = 0; hh <= 24 * days; hh += days <= 1 ? 6 : days <= 2 ? 12 : 24) label(ctx, `${hh}h`, X(hh * 6), r.y + r.h - 8, 10.5, T.c('muted'), 'center', 500);
    // the target band, step by step through the day
    ctx.fillStyle = T.c('good', 0.16);
    for (let t = 0; t < span; t++) { const h = (t / 6) % 24, tg = !s.home || (h >= 6 && h < 8) || (h >= 17 && h < 23) ? warm : warm - 5; ctx.fillRect(X(t), Y(tg + band), X(t + 1) - X(t) + 0.5, Y(tg - band) - Y(tg + band)); }
    // when the heating was on
    ctx.fillStyle = T.c('s2', 0.45);
    for (let k = 1; k < n; k++) if (s.hist[3 * k + 2]) ctx.fillRect(X(k - 1), Y(lo) - 6, X(k) - X(k - 1) + 0.5, 6);
    const curve = (off, tok, w) => { ctx.strokeStyle = tok; ctx.lineWidth = w; ctx.beginPath(); for (let k = 0; k < n; k++) { const x = X(k), y = Y(s.hist[3 * k + off]); if (k) ctx.lineTo(x, y); else ctx.moveTo(x, y); } ctx.stroke(); };
    if (!o.ghostsOnly) { curve(1, T.c('muted'), 1.5); curve(0, T.c('s1'), 2.2); }
    (o.ghosts || []).forEach((q, k) => { ctx.fillStyle = T.c(GHOSTS[(q.k ?? k) % 8], q.done ? 0.35 : 0.85); ctx.beginPath(); ctx.arc(X(q.x), Y(q.y), (q.k ?? k) ? 2.6 : 4, 0, TAU); ctx.fill(); });
    ctx.strokeStyle = T.c('ink', 0.35); ctx.setLineDash([3, 3]); ctx.beginPath(); ctx.moveTo(X(s.t), Y(hi)); ctx.lineTo(X(s.t), Y(lo)); ctx.stroke(); ctx.setLineDash([]);
    label(ctx, 'room', X(span) - 2, r.y + 10, 10.5, T.c('s1'), 'right', 700); label(ctx, 'outside', X(span) - 42, r.y + 10, 10.5, T.c('muted'), 'right', 600);
    ctx.restore();
  }

  // ------------------------------------------------------------------ hurdles
  // a stick figure running, in the air or ducking (phase: where its legs are in their stride)
  function runner(ctx, x, g, sc, y, duck, phase, color, w = 3) {
    const hip = g - (y + (duck ? 0.55 : 0.95)) * sc, top = g - (y + (duck ? LOW : TALL)) * sc, air = y > 0.02, lean = duck ? 0.5 : 0.15;
    ctx.strokeStyle = color; ctx.fillStyle = color; ctx.lineWidth = w; ctx.lineCap = 'round';
    const neck = [x + lean * 0.4 * sc, top + 0.28 * sc];
    ctx.beginPath(); ctx.moveTo(x, hip); ctx.lineTo(neck[0], neck[1]); ctx.stroke();
    ctx.beginPath(); ctx.arc(neck[0] + 0.05 * sc, top + 0.13 * sc, 0.13 * sc, 0, TAU); ctx.fill();
    const legs = air ? [[-0.25, 0.2], [0.3, 0.25]] : [[Math.sin(phase) * 0.35, 0], [-Math.sin(phase) * 0.35, 0]];
    for (const [dx, lift] of legs) { const kx = x + dx * 0.6 * sc + 0.08 * sc, ky = hip + (air ? 0.25 : 0.42) * sc, fx = x + dx * sc, fy = g - (y + lift) * sc; ctx.beginPath(); ctx.moveTo(x, hip); ctx.lineTo(kx, ky); ctx.lineTo(fx, fy); ctx.stroke(); }
    const sh = [x + lean * 0.3 * sc, top + 0.4 * sc];
    for (const s2 of [1, -1]) { ctx.beginPath(); ctx.moveTo(sh[0], sh[1]); ctx.lineTo(sh[0] + s2 * Math.sin(phase) * 0.3 * sc, sh[1] + 0.32 * sc); ctx.stroke(); }
    ctx.lineCap = 'butt';
  }
  /* From the side, the camera running with it: the sky and hills, the track with its distance marks, hurdles
     (white with red stripes) and bars (yellow and black) ahead, the finish, and the runner. o.ghosts: a
     generation's runners, the camera on the leader. */
  function drawHurdles(ctx, r, s, T, o = {}) {
    const V = fitBox(r, 2.4), g = V.y + V.h * 0.8, sc = V.w / 24, lead = o.ghosts && o.ghosts.length && o.ghostsOnly ? Math.max(...o.ghosts.map((q) => q.x)) : s.x, cam = lead - 5, X = (x) => V.x + (x - cam) * sc;
    ctx.save(); clip(ctx, V);
    let gr = ctx.createLinearGradient(0, V.y, 0, g); gr.addColorStop(0, '#9cc8ec'); gr.addColorStop(1, '#e2f0f9'); ctx.fillStyle = gr; ctx.fillRect(V.x, V.y, V.w, V.h);
    // hills far away move slowly
    ctx.fillStyle = 'rgba(110,160,120,0.55)'; ctx.beginPath(); ctx.moveTo(V.x, g);
    for (let px = 0; px <= V.w; px += 8) { const wx = (px / sc + cam * 0.3) * 0.35; ctx.lineTo(V.x + px, g - (0.9 + 0.6 * Math.sin(wx) + 0.35 * Math.sin(wx * 2.3 + 1)) * sc); }
    ctx.lineTo(V.x + V.w, g); ctx.closePath(); ctx.fill();
    ctx.fillStyle = '#b5543c'; ctx.fillRect(V.x, g, V.w, V.h * 0.2); ctx.fillStyle = 'rgba(255,255,255,0.55)'; ctx.fillRect(V.x, g + V.h * 0.08, V.w, 2);
    for (let m = Math.ceil(cam / 10) * 10; m < cam + 26; m += 10) { ctx.fillStyle = 'rgba(255,255,255,0.8)'; ctx.fillRect(X(m) - 1, g, 2, V.h * 0.05); label(ctx, `${m} m`, X(m), g + V.h * 0.13, 10.5, 'rgba(255,255,255,0.9)', 'center', 650); }
    // the finish
    if (X(s.course) < V.x + V.w + 20) { const fx = X(s.course); for (let k = 0; k < 10; k++) { ctx.fillStyle = k % 2 ? '#111' : '#fff'; ctx.fillRect(fx - 3, g - (k + 1) * 0.22 * sc, 6, 0.22 * sc); } label(ctx, 'finish', fx, g - 2.4 * sc, 11, '#1a1a1a', 'center', 750); }
    for (const q of s.obst) {
      const x = X(q.x);
      if (x < V.x - 20 || x > V.x + V.w + 20) continue;
      if (!q.bar) {
        ctx.fillStyle = '#e8e8e8'; ctx.fillRect(x - 0.32 * sc, g - HURDLE * sc, 3, HURDLE * sc); ctx.fillRect(x + 0.32 * sc - 3, g - HURDLE * sc, 3, HURDLE * sc);
        for (let k = 0; k < 5; k++) { ctx.fillStyle = k % 2 ? '#d33c3c' : '#ffffff'; ctx.fillRect(x - 0.32 * sc + (k * 0.64 * sc) / 5, g - HURDLE * sc - 0.12 * sc, (0.64 * sc) / 5 + 0.5, 0.12 * sc); }
      } else {
        ctx.fillStyle = '#5a5a5a'; ctx.fillRect(x - 0.4 * sc, g - 2.2 * sc, 3, 2.2 * sc); ctx.fillRect(x + 0.4 * sc - 3, g - 2.2 * sc, 3, 2.2 * sc);
        for (let k = 0; k < 6; k++) { ctx.fillStyle = k % 2 ? '#1c1c1c' : '#f2c230'; ctx.fillRect(x - 0.4 * sc + (k * 0.8 * sc) / 6, g - (BAR + 0.18) * sc, (0.8 * sc) / 6 + 0.5, 0.18 * sc); }
      }
    }
    (o.ghosts || []).forEach((q, k) => { const c = q.k ?? k; runner(ctx, X(q.x), g, sc, q.y, q.th > 0.5, q.x * 1.6, T.c(GHOSTS[c % 8], q.done ? 0.25 : c ? 0.55 : 0.95), c ? 2 : 3); });
    if (!o.ghostsOnly) {
      runner(ctx, X(s.x), g, sc, s.y, s.duck, s.x * 1.6, s.end && s.end !== 'course' ? T.c('critical') : '#1d3a6e', 3.2);
      label(ctx, `${Math.round(s.x)} of ${s.course} m · ${s.jumps} jumps`, V.x + 10, V.y + 14, 11.5, 'rgba(15,30,60,0.85)', 'left', 650);
      if (s.end) banner(ctx, V, s.end === 'course' ? 'Finished!' : s.end === 'barhit' ? 'Hit a bar' : s.end === 'hurdlehit' ? 'Hit a hurdle' : 'Out of time', s.end === 'course' ? T.c('good') : T.c('critical'));
    }
    ctx.restore();
  }

  // ------------------------------------------------------------------ the lamp and the corridor
  /* The corridor from above, the lamp at its start (lit at the first step, dark after: the faint ring shows where
     it was lit), the robot walking, the two doors at the junction (green for the right one once it has turned);
     under it, with the history view, what the model is given: its last 16 readings, newest on the right. */
  function drawMemory(ctx, r, s, T) {
    const hist = s.view === 'history', area = hist ? { x: r.x, y: r.y, w: r.w, h: r.h * 0.7 } : r, n = s.len + 1;
    const V = fitBox(area, (n + 2.2) / 3.2), cs = V.w / (n + 2.2), cy = V.y + V.h / 2, x0 = V.x + cs * 0.6, floor = rgbOf(T, 'panel'), wall = rgbOf(T, 'line-2');
    ctx.save();
    // the floor, a tile a step
    for (let i = 0; i < n; i++) { ctx.fillStyle = css(mix(floor, [0, 0, 0], i % 2 ? 0.05 : 0.1)); ctx.fillRect(x0 + i * cs, cy - cs / 2, cs, cs); }
    // the junction's two arms: up is left (it walks to the right), down is right
    const jx = x0 + (n - 1) * cs;
    for (const d of [-1, 1]) { ctx.fillStyle = css(mix(floor, [0, 0, 0], 0.07)); ctx.fillRect(jx, cy + (d < 0 ? -1.5 : 0.5) * cs, cs, cs); }
    ctx.strokeStyle = css(wall); ctx.lineWidth = 3;
    ctx.beginPath(); ctx.moveTo(x0, cy - cs / 2); ctx.lineTo(jx, cy - cs / 2); ctx.lineTo(jx, cy - 1.5 * cs); ctx.moveTo(x0, cy + cs / 2); ctx.lineTo(jx, cy + cs / 2); ctx.lineTo(jx, cy + 1.5 * cs);
    ctx.moveTo(jx + cs, cy - 1.5 * cs); ctx.lineTo(jx + cs, cy + 1.5 * cs); ctx.moveTo(x0, cy - cs / 2); ctx.lineTo(x0, cy + cs / 2); ctx.stroke();
    // the doors: green the right one once it has turned
    for (const [d, side] of [[-1, 0], [1, 1]]) {
      const yy = cy + (d < 0 ? -1.5 : 1.5) * cs, chosen = s.choice === side, good = side === s.side && s.end;
      ctx.fillStyle = s.end ? (good ? T.c('good', 0.85) : chosen ? T.c('critical', 0.85) : T.c('line', 0.8)) : T.c('line', 0.8);
      ctx.fillRect(jx + cs * 0.15, yy - 4, cs * 0.7, 8);
      label(ctx, side ? 'right' : 'left', jx + cs / 2, yy + d * 14, 10.5, T.c('muted'), 'center', 600);
    }
    // the lamp at the start: lit at the first step only (or the first few)
    const shows = s.t < (s.lit || 1);
    for (const side of [0, 1]) {
      const ly = cy + (side ? 1 : -1) * cs * 0.5, lit = shows && s.side === side, was = s.side === side;
      if (lit) { const gl = ctx.createRadialGradient(x0 + cs / 2, ly, 1, x0 + cs / 2, ly, cs * 1.2); gl.addColorStop(0, 'rgba(255,222,120,0.9)'); gl.addColorStop(1, 'rgba(255,222,120,0)'); ctx.fillStyle = gl; ctx.fillRect(x0 - cs, ly - cs * 1.2, cs * 3, cs * 2.4); }
      ctx.fillStyle = lit ? '#ffd76a' : T.c('line-2'); ctx.beginPath(); ctx.arc(x0 + cs / 2, ly, cs * 0.12, 0, TAU); ctx.fill();
      if (was && !lit) { ctx.strokeStyle = 'rgba(255,215,106,0.45)'; ctx.lineWidth = 1.5; ctx.setLineDash([3, 3]); ctx.beginPath(); ctx.arc(x0 + cs / 2, ly, cs * 0.2, 0, TAU); ctx.stroke(); ctx.setLineDash([]); }
    }
    // the robot
    const px = x0 + (Math.min(s.pos, n - 1) + 0.5) * cs, py = cy + (s.choice === null ? 0 : (s.choice ? 1 : -1) * cs);
    ctx.fillStyle = T.c('s1'); ctx.beginPath(); ctx.arc(px, py, cs * 0.28, 0, TAU); ctx.fill();
    ctx.fillStyle = '#fff'; ctx.beginPath(); ctx.arc(px + cs * 0.12, py, cs * 0.08, 0, TAU); ctx.fill();
    label(ctx, shows && !s.end ? `The lamp says ${s.side ? 'right' : 'left'}: remember it` : s.end ? (s.end === 'right' ? 'Turned the right way' : s.end === 'wrong' ? 'Turned the wrong way' : 'Out of time') : `Step ${s.t}: the lamp is dark now`, V.x + 4, V.y + 10, 11.5, T.c(s.end ? (s.end === 'right' ? 'good' : 'critical') : 'ink-2'), 'left', 650);
    ctx.restore();
    // what the model is given with the history view: its last 16 readings
    if (hist && s.past) {
      const hr = { x: r.x, y: r.y + r.h * 0.74, w: r.w, h: r.h * 0.26 }, cw = Math.min(28, (hr.w - 90) / HIST), ch = Math.min(12, (hr.h - 16) / 4), gx = hr.x + 90, off = HIST - s.past.length;
      label(ctx, 'what it is given:', hr.x, hr.y + 6, 10.5, T.c('muted'), 'left', 600);
      READ.forEach((nm, j) => label(ctx, nm, gx - 6, hr.y + 16 + j * ch + ch / 2, 10.5, T.c('muted'), 'right', 500));
      for (let k = 0; k < HIST; k++) for (let j = 0; j < 4; j++) { const v = k >= off ? s.past[k - off][j] : 0; ctx.fillStyle = v ? T.c('s2', 0.25 + 0.75 * v) : T.c('line', 0.35); ctx.fillRect(gx + k * cw + 1, hr.y + 16 + j * ch + 1, cw - 2, ch - 2); }
      label(ctx, 'now →', gx + HIST * cw, hr.y + 6, 10.5, T.c('muted'), 'right', 600);
    }
  }

  // ------------------------------------------------------------------ the valley
  /* The valley and its hills, the flag on the right, the cart on the slope (an arrow while it pushes), where it has
     been, and its energy (height and speed) as a bar. o.ghosts: a generation's carts. */
  function drawMountain(ctx, r, s, T, o = {}) {
    const V = fitBox(r, 1.7), x0 = -1.2, x1 = 0.6, X = (x) => V.x + ((x - x0) / (x1 - x0)) * V.w, Y = (y) => V.y + V.h * 0.86 - ((y + 1.1) / 2.3) * V.h * 0.72;
    ctx.save(); clip(ctx, V);
    let g = ctx.createLinearGradient(0, V.y, 0, V.y + V.h); g.addColorStop(0, '#1c2a52'); g.addColorStop(0.6, '#c97b63'); g.addColorStop(1, '#f2c48a'); ctx.fillStyle = g; ctx.fillRect(V.x, V.y, V.w, V.h);
    ctx.beginPath(); ctx.moveTo(V.x, V.y + V.h);
    for (let px = 0; px <= V.w; px += 3) { const x = x0 + (px / V.w) * (x1 - x0); ctx.lineTo(V.x + px, Y(Math.sin(3 * x))); }
    ctx.lineTo(V.x + V.w, V.y + V.h); ctx.closePath();
    g = ctx.createLinearGradient(0, Y(1), 0, V.y + V.h); g.addColorStop(0, '#4d6b47'); g.addColorStop(1, '#22331f'); ctx.fillStyle = g; ctx.fill();
    ctx.strokeStyle = 'rgba(220,240,200,0.5)'; ctx.lineWidth = 2; ctx.stroke();
    // the flag
    const fx = X(flagX), fy = Y(Math.sin(3 * flagX));
    ctx.strokeStyle = '#eee'; ctx.lineWidth = 2; ctx.beginPath(); ctx.moveTo(fx, fy); ctx.lineTo(fx, fy - V.h * 0.16); ctx.stroke();
    ctx.fillStyle = s.end === 'flag' ? T.c('good') : '#f2c230'; ctx.beginPath(); ctx.moveTo(fx, fy - V.h * 0.16); ctx.lineTo(fx + V.w * 0.05, fy - V.h * 0.13); ctx.lineTo(fx, fy - V.h * 0.1); ctx.closePath(); ctx.fill();
    const cart = (x, push, alpha, big, tone) => {
      const y = Math.sin(3 * x), ang = Math.atan2(-3 * Math.cos(3 * x) * (V.h * 0.72 / 2.3), V.w / (x1 - x0)), cx = X(x), cy = Y(y), w = V.w * (big ? 0.06 : 0.045), h = w * 0.45;
      ctx.save(); ctx.translate(cx, cy); ctx.rotate(ang);
      ctx.fillStyle = tone; ctx.globalAlpha = alpha; rounded(ctx, -w / 2, -h - w * 0.12, w, h, h * 0.3); ctx.fill();
      ctx.fillStyle = '#1b1b1b'; for (const e of [-0.3, 0.3]) { ctx.beginPath(); ctx.arc(e * w, -w * 0.1, w * 0.12, 0, TAU); ctx.fill(); }
      if (push && big) { ctx.fillStyle = 'rgba(255,240,200,0.95)'; ctx.beginPath(); const d = push; ctx.moveTo(-d * w * 0.55, -h * 0.6); ctx.lineTo(-d * w * 0.95, -h * 0.95); ctx.lineTo(-d * w * 0.95, -h * 0.25); ctx.closePath(); ctx.fill(); }
      ctx.restore();
    };
    (o.ghosts || []).forEach((q, k) => cart(q.x, 0, q.done ? 0.3 : 0.75, false, T.c(GHOSTS[(q.k ?? k) % 8])));
    if (!o.ghostsOnly) {
      if (s.trail) { for (let i = 0; i < s.trail.length; i += 3) { const a = 0.15 + (0.5 * i) / s.trail.length; ctx.fillStyle = `rgba(255,255,255,${a})`; ctx.beginPath(); ctx.arc(X(s.trail[i]), Y(Math.sin(3 * s.trail[i])) - 3, 2, 0, TAU); ctx.fill(); } }
      cart(s.x, s.push, 1, true, '#e8eef8');
      // energy: how high it is plus how fast it goes, against what the flag needs
      const bx = V.x + 12, by = V.y + 14, bw = Math.min(140, V.w * 0.3), e = clamp(s.energy / 2.5, 0, 1);
      label(ctx, 'energy', bx, by, 10.5, 'rgba(255,255,255,0.85)', 'left', 650);
      ctx.fillStyle = 'rgba(255,255,255,0.18)'; ctx.fillRect(bx + 46, by - 3, bw, 6); ctx.fillStyle = '#f2c230'; ctx.fillRect(bx + 46, by - 3, bw * e, 6);
      ctx.fillStyle = 'rgba(255,255,255,0.8)'; ctx.fillRect(bx + 46 + bw * (1.66 / 2.5), by - 6, 2, 12);
      label(ctx, `step ${s.t} of ${s.maxSteps}`, V.x + V.w - 10, by, 10.5, 'rgba(255,255,255,0.85)', 'right', 600);
      if (s.end) banner(ctx, V, s.end === 'flag' ? 'Reached the flag' : 'Out of time', s.end === 'flag' ? T.c('good') : T.c('critical'));
    }
    ctx.restore();
  }

  // ------------------------------------------------------------------ the pendulum
  /* The motor's hub (a gear that turns with the pole), the pole and its weight, the upright zone (±15°), an arrow
     around the hub while the motor pushes, and where the weight has been. o.ghosts: a generation's pendulums. */
  function drawSwing(ctx, r, s, T, o = {}) {
    const V = fitBox(r, 1.25), cx = V.x + V.w / 2, cy = V.y + V.h * 0.5, L = Math.min(V.w, V.h) * 0.38, tip = (th) => [cx + Math.sin(th) * L, cy - Math.cos(th) * L];
    ctx.save(); clip(ctx, V);
    const g = ctx.createRadialGradient(cx, cy, 10, cx, cy, V.w * 0.7); g.addColorStop(0, css(mix(rgbOf(T, 'panel'), [255, 255, 255], 0.04))); g.addColorStop(1, css(mix(rgbOf(T, 'panel'), [0, 0, 0], 0.12))); ctx.fillStyle = g; ctx.fillRect(V.x, V.y, V.w, V.h);
    // the upright zone
    ctx.fillStyle = T.c('good', 0.14); ctx.beginPath(); ctx.moveTo(cx, cy); ctx.arc(cx, cy, L * 1.15, -Math.PI / 2 - 15 * DEG, -Math.PI / 2 + 15 * DEG); ctx.closePath(); ctx.fill();
    ctx.strokeStyle = T.c('line', 0.8); ctx.lineWidth = 1; ctx.setLineDash([4, 5]); ctx.beginPath(); ctx.arc(cx, cy, L, 0, TAU); ctx.stroke(); ctx.setLineDash([]);
    (o.ghosts || []).forEach((q, k) => { const [x, y] = tip(q.x), c = T.c(GHOSTS[(q.k ?? k) % 8], q.done ? 0.25 : 0.55); ctx.strokeStyle = c; ctx.lineWidth = 2; ctx.beginPath(); ctx.moveTo(cx, cy); ctx.lineTo(x, y); ctx.stroke(); ctx.fillStyle = c; ctx.beginPath(); ctx.arc(x, y, L * 0.06, 0, TAU); ctx.fill(); });
    if (!o.ghostsOnly) {
      if (s.trail) for (let i = 0; i < s.trail.length; i += 3) { const [x, y] = tip(s.trail[i]); ctx.fillStyle = T.c('s2', 0.1 + (0.5 * i) / s.trail.length); ctx.beginPath(); ctx.arc(x, y, 3, 0, TAU); ctx.fill(); }
      const [x, y] = tip(s.th), up = Math.abs(s.th) < 15 * DEG;
      ctx.strokeStyle = T.c(up ? 'good' : 'ink-2'); ctx.lineWidth = Math.max(4, L * 0.05); ctx.lineCap = 'round'; ctx.beginPath(); ctx.moveTo(cx, cy); ctx.lineTo(x, y); ctx.stroke(); ctx.lineCap = 'butt';
      const bg = ctx.createRadialGradient(x - L * 0.03, y - L * 0.03, 1, x, y, L * 0.1); bg.addColorStop(0, '#ffffff'); bg.addColorStop(1, T.c(up ? 'good' : 's1')); ctx.fillStyle = bg; ctx.beginPath(); ctx.arc(x, y, L * 0.1, 0, TAU); ctx.fill();
      // the hub: a gear turning with it
      ctx.save(); ctx.translate(cx, cy); ctx.rotate(s.th); ctx.fillStyle = T.c('line-2');
      for (let k = 0; k < 10; k++) { ctx.rotate(TAU / 10); ctx.fillRect(-L * 0.02, -L * 0.13, L * 0.04, L * 0.04); }
      ctx.beginPath(); ctx.arc(0, 0, L * 0.1, 0, TAU); ctx.fill(); ctx.restore();
      if (s.tq) { const d = Math.sign(s.tq); ctx.strokeStyle = T.c('s2'); ctx.lineWidth = 3; ctx.beginPath(); ctx.arc(cx, cy, L * 0.2, -Math.PI / 2 - d * 0.9, -Math.PI / 2 + d * 0.9, d < 0); ctx.stroke(); const ea = -Math.PI / 2 + d * 0.9, ex = cx + Math.cos(ea) * L * 0.2, ey = cy + Math.sin(ea) * L * 0.2; ctx.fillStyle = T.c('s2'); ctx.beginPath(); ctx.arc(ex, ey, 4, 0, TAU); ctx.fill(); }
      label(ctx, `${Math.round(Math.abs(s.th) / DEG)}° from upright · up ${s.up} steps`, V.x + 10, V.y + 14, 11, T.c('ink-2'), 'left', 650);
      if (s.end) banner(ctx, V, s.end === 'balanced' ? 'Balanced upside down' : s.end === 'swung' ? 'Got it up, not for long' : 'Never got it up', s.end === 'balanced' ? T.c('good') : T.c('critical'));
    }
    ctx.restore();
  }

  // ------------------------------------------------------------------ the arm
  /* The bench and the arm's base, its reach, the target (a ring that turns green when reached), the hand's path,
     and the arm: upper arm, forearm, a gripper. o.ghosts: a generation's arms, thin. */
  function drawArm(ctx, r, s, T, o = {}) {
    const V = fitBox(r, 1.7), sc = Math.min(V.w / 4.2, V.h / 2.3), bx = V.x + V.w / 2, by = V.y + V.h * 0.86, X = (x) => bx + x * sc, Y = (y) => by - y * sc;
    ctx.save(); clip(ctx, V);
    ctx.fillStyle = css(mix(rgbOf(T, 'panel'), [0, 0, 0], 0.06)); ctx.fillRect(V.x, V.y, V.w, V.h);
    ctx.strokeStyle = T.c('line', 0.6); ctx.lineWidth = 1; for (let k = -4; k <= 4; k++) { ctx.beginPath(); ctx.moveTo(X(k * 0.5), V.y); ctx.lineTo(X(k * 0.5), by); ctx.stroke(); }
    ctx.fillStyle = T.c('line', 0.25); ctx.beginPath(); ctx.arc(bx, by, (L1 + L2) * sc, Math.PI, 0); ctx.arc(bx, by, Math.abs(L1 - L2) * sc, 0, Math.PI, true); ctx.closePath(); ctx.fill();
    ctx.fillStyle = '#8a6a4a'; ctx.fillRect(V.x, by, V.w, V.h * 0.14); ctx.fillStyle = 'rgba(255,255,255,0.15)'; ctx.fillRect(V.x, by, V.w, 3);
    // the target
    const tx = X(s.tx), ty = Y(s.ty), hit = s.end === 'reached', pulse = 1 + 0.12 * Math.sin(clock() / 180);
    ctx.strokeStyle = hit ? T.c('good') : T.c('s2'); ctx.lineWidth = 2.5; ctx.beginPath(); ctx.arc(tx, ty, (s.reach || 0.12) * sc * pulse, 0, TAU); ctx.stroke();
    ctx.beginPath(); ctx.moveTo(tx - 0.18 * sc, ty); ctx.lineTo(tx + 0.18 * sc, ty); ctx.moveTo(tx, ty - 0.18 * sc); ctx.lineTo(tx, ty + 0.18 * sc); ctx.stroke();
    const arm = (q1, q2, w, tone, grip) => {
      const ex = X(L1 * Math.cos(q1)), ey = Y(L1 * Math.sin(q1)), hx = X(L1 * Math.cos(q1) + L2 * Math.cos(q1 + q2)), hy = Y(L1 * Math.sin(q1) + L2 * Math.sin(q1 + q2));
      ctx.strokeStyle = tone; ctx.lineCap = 'round'; ctx.lineWidth = w; ctx.beginPath(); ctx.moveTo(bx, by); ctx.lineTo(ex, ey); ctx.stroke(); ctx.lineWidth = w * 0.8; ctx.beginPath(); ctx.moveTo(ex, ey); ctx.lineTo(hx, hy); ctx.stroke(); ctx.lineCap = 'butt';
      if (grip) {
        ctx.fillStyle = T.c('ink-2'); for (const [x, y, rr] of [[bx, by, w * 0.75], [ex, ey, w * 0.6]]) { ctx.beginPath(); ctx.arc(x, y, rr, 0, TAU); ctx.fill(); }
        const a = q1 + q2, c = Math.cos(a), si = Math.sin(a);
        ctx.strokeStyle = T.c('ink-2'); ctx.lineWidth = 3;
        for (const d of [-1, 1]) { ctx.beginPath(); ctx.moveTo(hx, hy); ctx.lineTo(hx + (c * 0.12 - d * si * 0.07) * sc, hy - (si * 0.12 + d * c * 0.07) * sc); ctx.stroke(); }
      }
    };
    (o.ghosts || []).forEach((q, k) => arm(q.x, q.y, 2, T.c(GHOSTS[(q.k ?? k) % 8], q.done ? 0.25 : 0.5), false));
    if (!o.ghostsOnly) {
      if (s.trail) { ctx.strokeStyle = T.c('s7', 0.6); ctx.lineWidth = 1.5; ctx.beginPath(); for (let i = 0; i < s.trail.length; i += 3) { const q1 = s.trail[i], q2 = s.trail[i + 1], x = X(L1 * Math.cos(q1) + L2 * Math.cos(q1 + q2)), y = Y(L1 * Math.sin(q1) + L2 * Math.sin(q1 + q2)); if (i) ctx.lineTo(x, y); else ctx.moveTo(x, y); } ctx.stroke(); }
      arm(s.q1, s.q2, Math.max(7, sc * 0.09), T.c('s1'), true);
      ctx.fillStyle = T.c('ink-2'); ctx.fillRect(bx - sc * 0.18, by - sc * 0.06, sc * 0.36, sc * 0.08);
      label(ctx, `${(s.d * 100).toFixed(0)} cm from the target · step ${s.t}`, V.x + 10, V.y + 14, 11, T.c('ink-2'), 'left', 650);
      if (s.end) banner(ctx, V, hit ? 'Reached it' : 'Out of time', hit ? T.c('good') : T.c('critical'));
    }
    ctx.restore();
  }

  Object.assign(E.DRAW, { thermo: wordless(drawThermo), hurdles: wordless(drawHurdles), memory: wordless(drawMemory), mountain: wordless(drawMountain), swingup: wordless(drawSwing), arm: wordless(drawArm) });
  Object.assign(E, { Thermo, Hurdles, Memory, Mountain, SwingUp, Arm });
})(typeof globalThis !== 'undefined' && globalThis.NA ? globalThis.NA : window.NA);
