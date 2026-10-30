/* rl-worlds.js — three bigger worlds for the model builder's "Learn from rewards", beside the small ones of
   rl-envs.js: a race track (a car that sees seven distances to the walls and its speed, or a camera picture from
   above), a drone arena seen in the first person (what each column of its view holds, a radar and its health, or
   the picture itself) and a rocket to land on a pad. The same interface as rl-envs.js (reset, step, snap, score),
   the same reward tables and formulas, inputs that can be left out and sensor noise. Each also says how it
   behaved (behavior(): two numbers from 0 to 1, the axes of MAP-Elites' grid of styles) and keeps a trace of
   where it went (trace: x, y, heading every other step, so a whole generation can be shown driving at once).
   Pictures: drawTrack, drawArena (a raycast first-person view with a map in the corner) and drawLander, each also
   drawing other members of a population (o.ghosts). Everything is seeded. Needs rl-envs.js. */
(function (NA) {
  'use strict';
  const E = NA.rlEnvs;
  const clamp = (v, a, b) => (v < a ? a : v > b ? b : v);
  const TAU = Math.PI * 2, DEG = Math.PI / 180;
  const wrapAngle = (a) => { a = (a + Math.PI) % TAU; if (a < 0) a += TAU; return a - Math.PI; };
  // a few random numbers from a seed, for the starts of a score, without disturbing training's stream
  const seeded = (seed) => new NA.RNG(seed);

  // =====================================================================================================
  // the race track
  // =====================================================================================================
  /* A closed loop through control points (a centripetal Catmull-Rom spline), 9 m wide, resampled to points 1 m
     apart along its middle line. Its walls are found from a grid of distances to that middle line (each cell 1 m),
     so a sensor's ray steps along by the distance to the nearest wall each time (sphere tracing). */
  const HW = 4.5; // half the track's width, m
  const TRACKS = {
    oval: { name: 'Oval', pts: [[80, 18], [122, 21], [147, 42], [148, 72], [124, 93], [80, 96], [36, 93], [12, 72], [12, 42], [38, 21]] },
    esses: { name: 'Esses', pts: [[30, 96], [80, 101], [130, 96], [151, 71], [143, 43], [119, 27], [97, 43], [75, 27], [53, 43], [33, 27], [14, 47], [14, 77]] },
    hairpins: { name: 'Hairpins', pts: [[26, 14], [134, 14], [152, 31], [133, 48], [50, 48], [31, 63], [50, 78], [133, 78], [152, 95], [131, 111], [24, 111], [8, 93], [8, 34]] },
  };
  function buildTrack(pts, name, check) {
    const n = pts.length, raw = [], d = (a, b) => Math.hypot(b[0] - a[0], b[1] - a[1]);
    for (let i = 0; i < n; i++) {
      const p0 = pts[(i - 1 + n) % n], p1 = pts[i], p2 = pts[(i + 1) % n], p3 = pts[(i + 2) % n];
      const t0 = 0, t1 = t0 + Math.sqrt(d(p0, p1)), t2 = t1 + Math.sqrt(d(p1, p2)), t3 = t2 + Math.sqrt(d(p2, p3));
      for (let k = 0; k < 48; k++) {
        const t = t1 + ((t2 - t1) * k) / 48, L = (a, b, ta, tb) => [((tb - t) * a[0] + (t - ta) * b[0]) / (tb - ta), ((tb - t) * a[1] + (t - ta) * b[1]) / (tb - ta)];
        const A1 = L(p0, p1, t0, t1), A2 = L(p1, p2, t1, t2), A3 = L(p2, p3, t2, t3), B1 = L(A1, A2, t0, t2), B2 = L(A2, A3, t1, t3);
        raw.push(L(B1, B2, t1, t2));
      }
    }
    // resampled 1 m apart (as near as a whole number of points allows)
    const cum = [0];
    for (let i = 1; i <= raw.length; i++) cum.push(cum[i - 1] + d(raw[i - 1], raw[i % raw.length]));
    const total = cum[raw.length], N = Math.round(total), ds = total / N, cx = new Float32Array(N), cy = new Float32Array(N);
    for (let k = 0, j = 0; k < N; k++) {
      const s = k * ds;
      while (cum[j + 1] < s) j++;
      const f = (s - cum[j]) / (cum[j + 1] - cum[j] || 1), a = raw[j], b = raw[(j + 1) % raw.length];
      cx[k] = a[0] + (b[0] - a[0]) * f; cy[k] = a[1] + (b[1] - a[1]) * f;
    }
    const tx = new Float32Array(N), ty = new Float32Array(N), curv = new Float32Array(N);
    for (let k = 0; k < N; k++) { const a = (k - 1 + N) % N, b = (k + 1) % N, dx = cx[b] - cx[a], dy = cy[b] - cy[a], l = Math.hypot(dx, dy) || 1; tx[k] = dx / l; ty[k] = dy / l; }
    let minR = Infinity;
    for (let k = 0; k < N; k++) {
      const a = (k - 2 + N) % N, b = (k + 2) % N, turn = wrapAngle(Math.atan2(ty[b], tx[b]) - Math.atan2(ty[a], tx[a]));
      curv[k] = turn / (4 * ds); minR = Math.min(minR, Math.abs(1 / (curv[k] || 1e-9)));
    }
    if (check && !check(minR, total)) return null;
    // the grid of distances to the middle line, filled only near the track (elsewhere: far outside)
    let x0 = Infinity, y0 = Infinity, x1 = -Infinity, y1 = -Infinity;
    for (let k = 0; k < N; k++) { x0 = Math.min(x0, cx[k]); y0 = Math.min(y0, cy[k]); x1 = Math.max(x1, cx[k]); y1 = Math.max(y1, cy[k]); }
    const m = HW + 4, ox = Math.floor(x0 - m), oy = Math.floor(y0 - m), gw = Math.ceil(x1 + m) - ox + 1, gh = Math.ceil(y1 + m) - oy + 1, grid = new Float32Array(gw * gh).fill(1e9), R = HW + 3;
    for (let k = 0; k < N; k++) {
      const ax = cx[k], ay = cy[k], bx = cx[(k + 1) % N], by = cy[(k + 1) % N], vx = bx - ax, vy = by - ay, vv = vx * vx + vy * vy || 1;
      const i0 = Math.max(0, Math.floor(Math.min(ax, bx) - R - ox)), i1 = Math.min(gw - 1, Math.ceil(Math.max(ax, bx) + R - ox)), j0 = Math.max(0, Math.floor(Math.min(ay, by) - R - oy)), j1 = Math.min(gh - 1, Math.ceil(Math.max(ay, by) + R - oy));
      for (let j = j0; j <= j1; j++) for (let i = i0; i <= i1; i++) {
        const px = ox + i + 0.5, py = oy + j + 0.5, t = clamp(((px - ax) * vx + (py - ay) * vy) / vv, 0, 1), dd = Math.hypot(px - ax - t * vx, py - ay - t * vy), at = j * gw + i;
        if (dd < grid[at]) grid[at] = dd;
      }
    }
    return { name, pts, N, ds, L: total, cx, cy, tx, ty, curv, minR, ox, oy, gw, gh, grid, box: [x0 - HW, y0 - HW, x1 + HW, y1 + HW] };
  }
  // distance from (x, y) to the track's middle line (bilinear in the grid); 1e9 well outside it
  function centerDist(T, x, y) {
    const fx = x - T.ox - 0.5, fy = y - T.oy - 0.5, i = Math.floor(fx), j = Math.floor(fy);
    if (i < 0 || j < 0 || i >= T.gw - 1 || j >= T.gh - 1) return 1e9;
    const g = T.grid, a = j * T.gw + i, v00 = g[a], v10 = g[a + 1], v01 = g[a + T.gw], v11 = g[a + T.gw + 1];
    if (v00 > 1e8 || v10 > 1e8 || v01 > 1e8 || v11 > 1e8) return 1e9;
    const u = fx - i, w = fy - j;
    return (v00 * (1 - u) + v10 * u) * (1 - w) + (v01 * (1 - u) + v11 * u) * w;
  }
  // how far a ray from (x, y) at angle a runs before it meets a wall (at most maxD)
  function trackRay(T, x, y, a, maxD) {
    const ux = Math.cos(a), uy = Math.sin(a);
    let t = 0;
    for (let k = 0; k < 64; k++) {
      const d = HW - centerDist(T, x + ux * t, y + uy * t);
      if (d < 0.06) return Math.max(0, t);
      t += Math.max(d * 0.92, 0.18);
      if (t >= maxD) return maxD;
    }
    return Math.min(t, maxD);
  }
  // the middle-line point nearest (x, y), searching around point k
  function locate(T, x, y, k) {
    let best = k, bd = Infinity;
    for (let j = -14; j <= 14; j++) { const q = (k + j + T.N) % T.N, dx = x - T.cx[q], dy = y - T.cy[q], dd = dx * dx + dy * dy; if (dd < bd) { bd = dd; best = q; } }
    return best;
  }
  const arcAt = (T, x, y, k) => k * T.ds + (x - T.cx[k]) * T.tx[k] + (y - T.cy[k]) * T.ty[k];
  const sideAt = (T, x, y, k) => (x - T.cx[k]) * -T.ty[k] + (y - T.cy[k]) * T.tx[k]; // + to the right of the way round
  const TRACK_CACHE = {};
  const trackOf = (id) => TRACK_CACHE[id] || (TRACK_CACHE[id] = buildTrack(TRACKS[id].pts, TRACKS[id].name));
  // a new track: 9 to 12 points around an oval, each at its own distance from the middle; kept when its bends are not
  // too tight for its width (the same seed always gives the same track)
  function randomTrack(seed) {
    const key = 'r' + seed;
    if (TRACK_CACHE[key]) return TRACK_CACHE[key];
    const rng = seeded(seed);
    let T = null;
    for (let tries = 0; tries < 60 && !T; tries++) {
      const n = 9 + rng.int(4), pts = [];
      for (let k = 0; k < n; k++) { const a = ((k + rng.uniform(-0.3, 0.3)) / n) * TAU, rr = rng.uniform(0.5, 1); pts.push([80 + Math.cos(a) * 68 * rr, 62 + Math.sin(a) * 46 * rr]); }
      T = buildTrack(pts, 'A random track', (minR, L) => minR > HW + 2.5 && L > 230);
    }
    T = T || trackOf('oval');
    const keys = Object.keys(TRACK_CACHE).filter((k) => k[0] === 'r');
    if (keys.length > 80) delete TRACK_CACHE[keys[0]];
    return (TRACK_CACHE[key] = T);
  }
  // the 4 tracks a model trained on random tracks is scored on: never trained on (seeds far from training's)
  const TEST_SEEDS = [7001, 7002, 7003, 7004];
  E.TRACKS = TRACKS;

  /* The car: speed builds toward 24 m/s with the throttle and drops with the brake (never below 3); steering turns
     it at up to 1.6 rad/s, but never sharper than its grip allows (14 m/s² sideways: at speed, a bend has to be
     taken slower). One step is 0.1 s. It crashes when it touches a wall. Rewards: progress along the track (by
     default 10 a lap), −1 for crashing; 4 laps end the episode, and so does time (by default 4 laps at an
     average of 10 m/s). view 'rays': 7 distances to the walls (90°, 50°,
     20° to each side and straight ahead, out to 40 m) and its speed. view 'camera': a picture from above, 16 × 16,
     of the track 32 m ahead and 12 m to each side, turned with the car, and its speed as a second channel. */
  const VMAX = 24, VMIN = 3, ACC = 9, BRAKE = 16, GRIP = 14, STEER = 1.6, CAR_R = 1, DT = 0.1, RANGE = 40;
  const RAYS = [-90, -50, -20, 0, 20, 50, 90];
  const RAY_NAMES = ['wall 90° left', 'wall 50° left', 'wall 20° left', 'wall ahead', 'wall 20° right', 'wall 50° right', 'wall 90° right'];
  class Track {
    constructor(o = {}) {
      this.view = o.view === 'camera' ? 'camera' : 'rays';
      this.trackId = TRACKS[o.track] || o.track === 'random' ? o.track : 'esses'; this.random = this.trackId === 'random';
      this.track = this.random ? randomTrack(TEST_SEEDS[0]) : trackOf(this.trackId);
      this.laps = 4; this.steps = o.steps > 0 ? Math.min(o.steps, 6000) : 0; this.nA = 4;
      this.actions = ['steer left', 'straight on', 'steer right', 'brake'];
      if (this.view === 'camera') { this.input = { kind: 'image', c: 2, h: 16, w: 16 }; this.obsSize = 512; this.obsNames = NA.range(512).map((i) => (i < 256 ? `track at (${i % 16}, ${Math.floor(i / 16)})` : 'speed')); }
      else { this.input = { kind: 'vector', dim: 8 }; this.obsSize = 8; this.obsNames = [...RAY_NAMES, 'speed']; }
      this.rays = new Float32Array(7);
      this.rw = E.rewardTable('track', o); this.rf = E.rewardFormula('track', o.rFormula); E.sensors(this, 'track', o);
    }
    reset(rng) {
      this.rng = rng;
      if (this.random && rng) this.track = randomTrack(1 + rng.int(64));
      const T = this.track;
      this.maxSteps = this.steps || Math.round(this.laps * T.L); // by default 4 laps at an average of 10 m/s
      // always the same start, so the score (from it) is what evolution judged: with a start turned a little at random,
      // a driver that got round one start could crash from the next, and the score jumped about
      this.k = 0; this.x = T.cx[0]; this.y = T.cy[0]; this.th = Math.atan2(T.ty[0], T.tx[0]); this.v = 8;
      this.s = arcAt(T, this.x, this.y, 0); this.prog = 0; this.lap = 0; this.t = 0; this.end = null; this.a = null;
      this.sumV = 0; this.sumSide = 0; this.trace = [this.x, this.y, this.th];
      this.see();
      return this.obs();
    }
    see() { for (let i = 0; i < 7; i++) this.rays[i] = trackRay(this.track, this.x, this.y, this.th + RAYS[i] * DEG, RANGE); }
    rawObs() {
      if (this.view === 'camera') return this.camera();
      const o = new Float32Array(8);
      for (let i = 0; i < 7; i++) o[i] = this.rays[i] / RANGE;
      o[7] = this.v / VMAX;
      return o;
    }
    // the picture from above: forward up, 32 m ahead (from 4 m behind), 12 m to each side
    camera() {
      const o = new Float32Array(512), c = Math.cos(this.th), s = Math.sin(this.th), sp = this.v / VMAX;
      for (let i = 0; i < 16; i++) {
        const fwd = ((15.5 - i) / 16) * 32 - 4;
        for (let j = 0; j < 16; j++) {
          const right = ((j + 0.5) / 16) * 24 - 12, x = this.x + fwd * c - right * s, y = this.y + fwd * s + right * c;
          o[i * 16 + j] = centerDist(this.track, x, y) < HW ? 1 : 0; o[256 + i * 16 + j] = sp;
        }
      }
      return o;
    }
    step(a) {
      const T = this.track, gas = a !== 3, steer = a === 0 ? -1 : a === 2 ? 1 : 0;
      if (gas) this.v += ACC * DT * (1 - this.v / VMAX); else this.v = Math.max(VMIN, this.v - BRAKE * DT);
      this.th += steer * Math.min(STEER, GRIP / Math.max(this.v, 0.5)) * DT;
      this.x += this.v * Math.cos(this.th) * DT; this.y += this.v * Math.sin(this.th) * DT;
      this.t++; this.a = a;
      // progress: how far along the middle line it got (backwards counts against it)
      this.k = locate(T, this.x, this.y, this.k);
      const s = arcAt(T, this.x, this.y, this.k);
      let ds = s - this.s;
      if (ds < -T.L / 2) ds += T.L; else if (ds > T.L / 2) ds -= T.L;
      this.s = s; this.prog += ds;
      const lapsNow = Math.max(0, Math.floor(this.prog / T.L)), lap = lapsNow > this.lap;
      if (lap) this.lap = lapsNow;
      const side = sideAt(T, this.x, this.y, this.k), crashed = centerDist(T, this.x, this.y) > HW - CAR_R;
      this.sumV += this.v; this.sumSide += clamp(side / HW, -1, 1);
      if (this.t % 2 === 0) this.trace.push(this.x, this.y, this.th);
      if (!crashed) this.see();
      const done = crashed || this.prog >= this.laps * T.L || this.t >= this.maxSteps;
      if (done) this.end = crashed ? 'crashed' : this.prog >= this.laps * T.L ? 'finished' : 'time';
      const rw = this.rw, progress = ds / T.L;
      const r = this.rf ? this.rf({ progress, speed: this.v, crashed: +crashed, lap: +lap, center: Math.min(1, Math.abs(side) / HW), t: this.t, turn: steer, brake: +!gas })
        : rw.rProgress * progress + (lap ? rw.rLap : 0) + (crashed ? rw.rCrash : 0) + rw.rStep;
      return { obs: this.obs(), r, done };
    }
    snap() { return { track: this.track, x: this.x, y: this.y, th: this.th, v: this.v, rays: Array.from(this.rays), laps: this.laps, lap: this.lap, prog: this.prog / this.track.L, t: this.t, a: this.a, end: this.end, trail: this.trace.slice(-90) }; }
    // its average speed (slow to fast) and which side of the middle it keeps (left to right): its style of driving
    behavior() { const n = Math.max(1, this.t); return [clamp(this.sumV / n / VMAX, 0, 1), clamp(0.5 + (1.2 * this.sumSide) / n, 0, 1)]; }
    // the share of 4 laps its best moves drive (on random tracks: on 4 it never trained on)
    score(policy) {
      const list = this.random ? TEST_SEEDS.map(randomTrack) : [this.track], keep = this.track, random = this.random;
      let total = 0;
      this.random = false;
      for (const T of list) { this.track = T; let o = this.reset(null), done = false; while (!done) ({ obs: o, done } = this.step(policy(o))); total += clamp(this.prog / (this.laps * T.L), 0, 1); }
      this.random = random; this.track = keep;
      return total / list.length;
    }
  }

  // =====================================================================================================
  // the drone arena
  // =====================================================================================================
  /* A first-person arena: a 12 × 12 grid of walls and floor. It turns 12° a step, walks 0.22 m, or fires (once
     every 4 steps: the gun has to recharge). Glowing drones drift toward it; a shot that crosses one destroys it
     (+1), one that reaches it costs a life (−1, 3 lives). The episode ends when 10 drones are destroyed, the lives
     run out, or after 500 steps. view 'rays': what each of 11 columns across its 66° view holds (how near the wall
     is, how near a drone is, 0 when none), a radar (which way the nearest drone is, behind walls too), its lives and
     whether the gun is ready: 26 numbers. view 'pixels': the view itself, 10 × 20 pixels, walls (brighter nearer)
     and drones as two channels: learning to aim from pixels, as agents learned to play first-person games. */
  const MAPS = {
    room: { name: 'Room', rows: ['############', '#..........#', '#..........#', '#..#....#..#', '#..........#', '#..........#', '#....S.....#', '#..........#', '#..#....#..#', '#..........#', '#..........#', '############'] },
    pillars: { name: 'Pillars', rows: ['############', '#..........#', '#.#..#..#..#', '#..........#', '#..#....#..#', '#.....#....#', '#..#.S.....#', '#.......#..#', '#.#..#.....#', '#..........#', '#....#...#.#', '############'] },
    halls: { name: 'Halls', rows: ['############', '#....#.....#', '#....#.....#', '#..........#', '#....#..####', '###.##.....#', '#....S.....#', '#.....#....#', '####..#....#', '#.....#....#', '#..........#', '############'] },
  };
  E.ARENAS = MAPS;
  const FOV = 66 * DEG, COLS = 11, PCOLS = 20, PROWS = 10, VIEW = 8, DRONE_R = 0.3, ME_R = 0.25, TURN = 12 * DEG, WALK = 0.22;
  const mapOf = (id) => { const m = MAPS[id] || MAPS.room, rows = m.rows, H = rows.length, W = rows[0].length, s = rows.join('').indexOf('S'); return { id, name: m.name, rows, W, H, wall: (x, y) => x < 0 || y < 0 || x >= W || y >= H || rows[y][x] === '#', start: [(s % W) + 0.5, Math.floor(s / W) + 0.5] }; };
  /* A ray through the grid (DDA): its length to the first wall, which face it hit (0: one running north–south,
     1: east–west) and where along that face (0 to 1, for the walls' texture). */
  function castRay(M, x, y, a, maxD = 24) {
    const dx = Math.cos(a), dy = Math.sin(a);
    let mx = Math.floor(x), my = Math.floor(y);
    const ddx = Math.abs(1 / (dx || 1e-12)), ddy = Math.abs(1 / (dy || 1e-12)), sx = dx < 0 ? -1 : 1, sy = dy < 0 ? -1 : 1;
    let tx = (dx < 0 ? x - mx : mx + 1 - x) * ddx, ty = (dy < 0 ? y - my : my + 1 - y) * ddy, side = 0;
    for (let k = 0; k < 64; k++) {
      if (tx < ty) { mx += sx; side = 0; if (M.wall(mx, my)) { const d = tx, u = y + d * dy; return { d: Math.min(d, maxD), side, u: u - Math.floor(u) }; } tx += ddx; }
      else { my += sy; side = 1; if (M.wall(mx, my)) { const d = ty, u = x + d * dx; return { d: Math.min(d, maxD), side, u: u - Math.floor(u) }; } ty += ddy; }
      if (Math.min(tx, ty) > maxD) break;
    }
    return { d: maxD, side, u: 0 };
  }
  // a circle of radius r at (x, y) clear of every wall cell around it
  const clearAt = (M, x, y, r) => { for (let j = Math.floor(y - r); j <= Math.floor(y + r); j++) for (let i = Math.floor(x - r); i <= Math.floor(x + r); i++) if (M.wall(i, j)) { const cx = clamp(x, i, i + 1), cy = clamp(y, j, j + 1); if ((x - cx) ** 2 + (y - cy) ** 2 < r * r) return false; } return true; };
  // a move that slides along walls: across, then along
  function slide(M, o, dx, dy, r) { if (clearAt(M, o.x + dx, o.y, r)) o.x += dx; if (clearAt(M, o.x, o.y + dy, r)) o.y += dy; }
  class Arena {
    constructor(o = {}) {
      this.view = o.view === 'pixels' ? 'pixels' : 'rays';
      this.map = mapOf(MAPS[o.map] ? o.map : 'room'); this.mapId = this.map.id;
      this.k = clamp(Math.round(o.drones) || 1, 1, 3); this.fast = o.speed === 'fast'; this.maxSteps = o.steps > 0 ? Math.min(o.steps, 3000) : 500; this.goal = 10; this.lives = 3;
      this.nA = 4; this.actions = ['turn left', 'turn right', 'walk forward', 'fire'];
      if (this.view === 'pixels') { this.input = { kind: 'image', c: 2, h: PROWS, w: PCOLS }; this.obsSize = 2 * PROWS * PCOLS; this.obsNames = NA.range(this.obsSize).map((i) => `${i < PROWS * PCOLS ? 'wall' : 'drone'} at (${i % PCOLS}, ${Math.floor((i % (PROWS * PCOLS)) / PCOLS)})`); }
      else {
        const at = (c) => (c === 5 ? 'ahead' : `${Math.abs(c - 5) * 6.6}° ${c < 5 ? 'left' : 'right'}`.replace('.0°', '°'));
        this.input = { kind: 'vector', dim: 26 }; this.obsSize = 26;
        this.obsNames = [...NA.range(COLS).map((c) => `wall ${at(c)}`), ...NA.range(COLS).map((c) => `drone ${at(c)}`), 'radar: across', 'radar: ahead', 'lives', 'gun ready'];
      }
      this.rw = E.rewardTable('arena', o); this.rf = E.rewardFormula('arena', o.rFormula); E.sensors(this, 'arena', o);
    }
    reset(rng) {
      this.rng = rng || seeded(91);
      const M = this.map;
      [this.x, this.y] = M.start; this.th = 0; this.t = 0; this.health = this.lives; this.kills = 0; this.cool = 0; this.end = null; this.a = null;
      this.moved = 0; this.shots = 0; this.fx = null; this.trace = [this.x, this.y, this.th];
      this.drones = NA.range(this.k).map((i) => ({ x: 0, y: 0, alive: false, wait: i * 6, phase: this.rng.uniform(0, TAU) }));
      for (const d of this.drones) if (!d.wait) this.spawn(d);
      return this.obs();
    }
    // a drone on a free square at least 4 m away
    spawn(d) {
      const M = this.map;
      for (let tries = 0; tries < 200; tries++) {
        const x = this.rng.int(M.W) + 0.5, y = this.rng.int(M.H) + 0.5;
        if (M.wall(Math.floor(x), Math.floor(y)) || Math.hypot(x - this.x, y - this.y) < 4) continue;
        Object.assign(d, { x, y, alive: true, wait: 0 });
        return;
      }
    }
    // where a drone is in its view: angle from straight ahead, distance, angular radius, and whether a wall hides it
    sight(d) {
      const dx = d.x - this.x, dy = d.y - this.y, dist = Math.hypot(dx, dy), rel = wrapAngle(Math.atan2(dy, dx) - this.th), rad = Math.atan(DRONE_R / Math.max(dist, 0.05));
      const inView = Math.abs(rel) - rad < FOV / 2, hidden = castRay(this.map, this.x, this.y, this.th + rel, dist + 1).d < dist - DRONE_R;
      return { rel, dist, rad, visible: inView && !hidden };
    }
    rawObs() {
      if (this.view === 'pixels') return this.picture();
      const o = new Float32Array(26), half = FOV / (COLS - 1) / 2;
      for (let c = 0; c < COLS; c++) o[c] = 1 - Math.min(castRay(this.map, this.x, this.y, this.th + (c / (COLS - 1) - 0.5) * FOV).d, VIEW) / VIEW;
      let near = null;
      for (const d of this.drones) {
        if (!d.alive) continue;
        const s = this.sight(d);
        if (!near || s.dist < near.dist) near = s;
        if (!s.visible || s.dist > VIEW) continue;
        for (let c = 0; c < COLS; c++) { const ac = (c / (COLS - 1) - 0.5) * FOV; if (Math.abs(ac - s.rel) < half + s.rad) o[COLS + c] = Math.max(o[COLS + c], 1 - s.dist / VIEW); }
      }
      if (near) { o[22] = Math.sin(near.rel); o[23] = Math.cos(near.rel); }
      o[24] = this.health / this.lives; o[25] = this.cool ? 0 : 1;
      return o;
    }
    // what it sees, 10 × 20: walls (brighter when nearer) and drones, each column with the nearest thing in it
    picture() {
      const o = new Float32Array(2 * PROWS * PCOLS), z = new Float32Array(PCOLS), n = PROWS * PCOLS;
      for (let j = 0; j < PCOLS; j++) {
        const a = ((j + 0.5) / PCOLS - 0.5) * FOV, h = castRay(this.map, this.x, this.y, this.th + a), dp = Math.max(0.1, h.d * Math.cos(a)), half = Math.min(PROWS / 2, (PROWS * 0.55) / dp), v = 1 - 0.8 * Math.min(h.d, VIEW) / VIEW;
        z[j] = h.d;
        for (let i = 0; i < PROWS; i++) if (Math.abs(i + 0.5 - PROWS / 2) < half) o[i * PCOLS + j] = v;
      }
      const colA = FOV / PCOLS;
      for (const d of this.drones) {
        if (!d.alive) continue;
        const s = this.sight(d);
        if (!s.visible) continue;
        const xc = (s.rel / FOV + 0.5) * PCOLS, rr = Math.max(0.6, s.rad / colA), yc = PROWS * 0.45;
        for (let j = Math.max(0, Math.floor(xc - rr)); j <= Math.min(PCOLS - 1, Math.floor(xc + rr)); j++) {
          if (z[j] < s.dist - DRONE_R) continue;
          for (let i = Math.max(0, Math.floor(yc - rr)); i <= Math.min(PROWS - 1, Math.floor(yc + rr)); i++) if ((j + 0.5 - xc) ** 2 + (i + 0.5 - yc) ** 2 <= rr * rr) o[n + i * PCOLS + j] = 1;
        }
      }
      return o;
    }
    step(a) {
      const M = this.map, turn = a === 0 ? -1 : a === 1 ? 1 : 0, walk = a === 2;
      this.th = wrapAngle(this.th + turn * TURN);
      if (walk) { const x0 = this.x, y0 = this.y; slide(M, this, Math.cos(this.th) * WALK, Math.sin(this.th) * WALK, ME_R); if (Math.hypot(this.x - x0, this.y - y0) > 0.01) this.moved++; }
      // the gun: a shot crosses the nearest drone in line before the wall
      let shot = false, hit = null;
      if (this.cool) this.cool--;
      else if (a === 3) {
        shot = true; this.shots++; this.cool = 3;
        const wall = castRay(M, this.x, this.y, this.th).d, ux = Math.cos(this.th), uy = Math.sin(this.th);
        let best = null, bd = Infinity;
        for (const d of this.drones) {
          if (!d.alive) continue;
          const along = (d.x - this.x) * ux + (d.y - this.y) * uy, across = Math.abs(-(d.x - this.x) * uy + (d.y - this.y) * ux);
          if (along > 0 && along < wall && across < DRONE_R + 0.06 && along < bd) { bd = along; best = d; }
        }
        if (best) { hit = [best.x, best.y]; best.alive = false; best.wait = 8; this.kills++; }
      }
      // the drones drift toward it, weaving a little; one that reaches it costs a life
      let hurt = 0;
      const sp = this.fast ? 0.085 : 0.045;
      for (const d of this.drones) {
        if (!d.alive) { if (d.wait > 0 && --d.wait === 0) this.spawn(d); continue; }
        const dx = this.x - d.x, dy = this.y - d.y, l = Math.hypot(dx, dy) || 1, w = 0.55 * Math.sin(d.phase + this.t * 0.12), c = Math.cos(w), s = Math.sin(w);
        slide(M, d, ((dx * c - dy * s) / l) * sp, ((dx * s + dy * c) / l) * sp, DRONE_R);
        if (Math.hypot(this.x - d.x, this.y - d.y) < DRONE_R + ME_R + 0.05) { hurt++; d.alive = false; d.wait = 14; }
      }
      this.health -= hurt; this.t++; this.a = a;
      if (this.t % 2 === 0) this.trace.push(this.x, this.y, this.th);
      this.fx = shot || hurt ? { shot, hit, hurt } : null;
      const done = this.health <= 0 || this.kills >= this.goal || this.t >= this.maxSteps;
      if (done) this.end = this.kills >= this.goal ? 'cleared' : this.health <= 0 ? 'dead' : 'time';
      let aimed = 0, seen = 0;
      if (this.rf) for (const d of this.drones) { if (!d.alive) continue; const s = this.sight(d); if (s.visible) { seen++; if (Math.abs(s.rel) < s.rad + 0.02) aimed = 1; } }
      const rw = this.rw;
      const r = this.rf ? this.rf({ hit: hit ? 1 : 0, hurt, shot: +shot, aimed, seen, lives: this.health, t: this.t, moved: +walk, turned: Math.abs(turn) })
        : (hit ? rw.rHit : 0) + hurt * rw.rHurt + (shot ? rw.rShot : 0) + rw.rStep;
      return { obs: this.obs(), r, done };
    }
    snap() { return { map: this.map, x: this.x, y: this.y, th: this.th, drones: this.drones.map((d) => ({ x: d.x, y: d.y, alive: d.alive })), health: this.health, lives: this.lives, kills: this.kills, goal: this.goal, t: this.t, maxSteps: this.maxSteps, cool: this.cool, a: this.a, end: this.end, fx: this.fx }; }
    // how much it walks (stands still to always on the move) and how often it fires: its style of play
    behavior() { const n = Math.max(1, this.t); return [clamp(this.moved / n, 0, 1), clamp((4 * this.shots) / n, 0, 1)]; }
    // the share of 10 drones its best moves destroy, over 3 games (the same 3 every time)
    score(policy) { const r = seeded(4242); let s = 0; for (let k = 0; k < 3; k++) { let o = this.reset(seeded(r.int(1e9))), done = false; while (!done) ({ obs: o, done } = this.step(policy(o))); s += this.kills / this.goal; } return s / 3; }
  }

  // =====================================================================================================
  // the rocket
  // =====================================================================================================
  /* A rocket comes down 30 m above the ground, somewhere up to 12 m to either side of a 12 m pad, drifting and
     tilted a little. Its main engine pushes along its axis (2.4 × gravity, plus 1 m/s²); its side thrusters turn
     it, while its fuel lasts (150 steps of the main engine): hovering until it runs dry ends in a fall. It lands
     when a leg touches the ground: on the pad, slower than 3 m/s down and 2 m/s across, within 25° of upright,
     it has landed (+1); anything else is a crash (−1), and so is still being up there after 300 steps. By default
     it is also paid for getting closer and slower (shaping: the change in −(distance/10 + speed/3 + tilt), so the
     total only depends on where it starts and ends). Wind, if on, pushes it sideways in gusts. One step is 0.08 s,
     at most 300 of them. */
  const LEG = 1.8, FOOT = 1.4, PAD = 6;
  class Lander {
    constructor(o = {}) {
      this.g = o.gravity > 0 ? o.gravity : 1.62; this.wind = !!o.wind; this.randomPad = o.pad === 'random';
      this.maxSteps = o.steps > 0 ? Math.min(o.steps, 3000) : 300; this.thrust = 2.4 * this.g + 1; this.fuelMax = o.fuel > 0 ? Math.min(o.fuel, 3000) : 150;
      this.nA = 4; this.actions = ['nothing', 'main engine', 'turn left', 'turn right'];
      this.input = { kind: 'vector', dim: 7 }; this.obsSize = 7;
      this.obsNames = ['across from the pad', 'height', 'speed across', 'speed up', 'tilt', 'spin', 'fuel left'];
      this.rw = E.rewardTable('lander', o); this.rf = E.rewardFormula('lander', o.rFormula); E.sensors(this, 'lander', o);
    }
    // the ground: gentle hills, flat for 3 m beyond each end of the pad (heights above the pad's level)
    ground(x) { const d = Math.abs(x - this.pad); if (d < PAD + 3) return 0; const h = 3 + 2.2 * Math.sin(0.17 * x + 1.3) + 1.6 * Math.sin(0.41 * x + 0.4) + 0.9 * Math.sin(0.83 * x + 2.1); return Math.max(0, h) * Math.min(1, (d - PAD - 3) / 4); }
    reset(rng) {
      const r = rng || seeded(5);
      this.rng = rng;
      this.pad = this.randomPad ? r.uniform(-20, 20) : 0;
      this.x = this.pad + r.uniform(-12, 12); this.y = 30; this.vx = r.uniform(-2, 2); this.vy = r.uniform(-2, 0); this.th = r.uniform(-0.3, 0.3); this.w = r.uniform(-0.2, 0.2);
      this.windPhase = r.uniform(0, TAU); this.t = 0; this.end = null; this.a = null; this.engine = 0; this.burn = 0; this.landX = null; this.fuel = this.fuelMax;
      this.phi = this.potential(); this.trace = [this.x, this.y, this.th];
      return this.obs();
    }
    potential() { return -(Math.hypot(this.x - this.pad, this.y) / 10 + Math.hypot(this.vx, this.vy) / 3 + Math.abs(this.th)); }
    windNow() { return this.wind ? 0.5 * Math.sqrt(this.g) * (Math.sin(0.35 * this.t * 0.08 * 3 + this.windPhase) + 0.6 * Math.sin(1.3 * this.t * 0.08 + 2 * this.windPhase)) : 0; }
    rawObs() { return Float32Array.of((this.x - this.pad) / 20, this.y / 30, this.vx / 5, this.vy / 5, this.th / 0.6, this.w / 1.5, this.fuel / this.fuelMax); }
    feet() { const c = Math.cos(this.th), s = Math.sin(this.th); return [-FOOT, FOOT].map((lx) => [this.x + lx * c - LEG * s, this.y - lx * s - LEG * c]); }
    step(a) {
      const dt = 0.08, main = a === 1 && this.fuel > 0, turn = a === 2 ? -1 : a === 3 ? 1 : 0;
      let ax = -0.04 * this.vx + this.windNow(), ay = -this.g - 0.04 * this.vy;
      if (main) { ax += this.thrust * Math.sin(this.th); ay += this.thrust * Math.cos(this.th); this.burn++; this.fuel--; }
      this.w = (this.w + turn * 2.6 * dt) * 0.985;
      this.vx += ax * dt; this.vy += ay * dt; this.x += this.vx * dt; this.y += this.vy * dt; this.th += this.w * dt;
      this.t++; this.a = a; this.engine = main ? 1 : 0;
      if (this.t % 2 === 0) this.trace.push(this.x, this.y, this.th);
      // touching down: a foot at or below the ground, or the body itself
      const F = this.feet(), touch = F.some(([fx, fy]) => fy <= this.ground(fx)) || this.y - 0.9 <= this.ground(this.x);
      let landed = false, crashed = false;
      if (touch) {
        const onPad = F.every(([fx]) => Math.abs(fx - this.pad) <= PAD), soft = this.vy > -3 && Math.abs(this.vx) < 2 && Math.abs(this.th) < 25 * DEG;
        landed = onPad && soft; crashed = !landed; this.landX = this.x;
        this.end = landed ? 'landed' : soft ? 'offpad' : 'crashed';
      } else if (Math.abs(this.x - this.pad) > 50 || this.y > 60) { crashed = true; this.end = 'away'; }
      else if (this.t >= this.maxSteps) { crashed = true; this.end = 'time'; } // out of time counts as a failure: no hovering for ever
      const phi = this.potential(), closer = phi - this.phi;
      this.phi = phi;
      const rw = this.rw, dist = Math.hypot(this.x - this.pad, this.y), speed = Math.hypot(this.vx, this.vy);
      const r = this.rf ? this.rf({ landed: +landed, crashed: +crashed, dist, speed, tilt: this.th / DEG, engine: +main, turn: Math.abs(turn), h: this.y, vx: this.vx, vy: this.vy, fuel: this.fuel / this.fuelMax, t: this.t, closer })
        : (landed ? rw.rLand : 0) + (crashed ? rw.rCrash : 0) + (main ? rw.rFuel : 0) + rw.rShape * closer;
      return { obs: this.obs(), r, done: !!this.end };
    }
    snap() { return { x: this.x, y: this.y, th: this.th, vx: this.vx, vy: this.vy, fuel: this.fuel, fuelMax: this.fuelMax, engine: this.engine, turn: this.a === 2 ? -1 : this.a === 3 ? 1 : 0, t: this.t, end: this.end, pad: this.pad, wind: this.windNow(), ground: (x) => this.ground(x), trail: this.trace.slice(-120) }; }
    // where it came down (left of the pad to right) and how much it burned: its style of landing
    behavior() { const n = Math.max(1, this.t), x = this.landX === null ? this.x : this.landX; return [clamp(0.5 + (x - this.pad) / 40, 0, 1), clamp(this.burn / n, 0, 1)]; }
    // the share of 10 starts its best moves land on the pad (the same 10 every time)
    score(policy) { const r = seeded(4242); let ok = 0; for (let k = 0; k < 10; k++) { let o = this.reset(seeded(r.int(1e9))), done = false; while (!done) ({ obs: o, done } = this.step(policy(o))); if (this.end === 'landed') ok++; } return ok / 10; }
  }

  for (const W of [Track, Arena, Lander]) W.prototype.obs = E.sensed;

  // ------------------------------------------------------------------ rewards, inputs, the catalogue
  Object.assign(E.REWARDS, {
    track: [{ key: 'rProgress', label: 'Progress, per lap of the track', def: 10 }, { key: 'rCrash', label: 'Crashing into a wall', def: -1 }, { key: 'rLap', label: 'Finishing a lap', def: 0 }, { key: 'rStep', label: 'Every step', def: 0 }],
    arena: [{ key: 'rHit', label: 'Destroying a drone', def: 1 }, { key: 'rHurt', label: 'A drone reaching you', def: -1 }, { key: 'rShot', label: 'Firing a shot', def: 0 }, { key: 'rStep', label: 'Every step', def: 0 }],
    lander: [{ key: 'rLand', label: 'Landing softly on the pad', def: 1 }, { key: 'rCrash', label: 'Crashing, flying off or running out of time', def: -1 }, { key: 'rShape', label: 'Getting closer and slower (shaping)', def: 1 }, { key: 'rFuel', label: 'Each step the main engine fires', def: 0 }],
  });
  Object.assign(E.REWARD_VARS, {
    track: [['progress', 'how far along the track it got this step, in laps (backwards: less than 0)'], ['speed', 'its speed, m/s (3 to 24)'], ['crashed', '1 on touching a wall'], ['lap', '1 on finishing a lap'],
      ['center', 'how far from the middle of the track: 0 in the middle, 1 at a wall'], ['turn', '−1 steering left, 1 right, 0 straight'], ['brake', '1 when braking'], ['t', 'steps so far']],
    arena: [['hit', '1 on destroying a drone'], ['hurt', '1 when a drone reaches it'], ['shot', '1 on firing'], ['aimed', '1 while a drone is in its sights'], ['seen', 'how many drones it can see'],
      ['lives', 'lives left (3 at the start)'], ['moved', '1 when it walked'], ['turned', '1 when it turned'], ['t', 'steps so far']],
    lander: [['landed', '1 on landing softly on the pad'], ['crashed', '1 on crashing, flying off or running out of time'], ['closer', 'the shaping: how much closer, slower and straighter it got this step'], ['dist', 'distance to the pad, m'],
      ['speed', 'its speed, m/s'], ['tilt', 'its tilt, degrees'], ['engine', '1 while the main engine fires'], ['turn', '1 while a side thruster fires'], ['h', 'height above the pad, m'], ['vx', 'speed across, m/s'], ['vy', 'speed up, m/s'], ['fuel', 'fuel left: 1 full, 0 empty'], ['t', 'steps so far']],
  });
  Object.assign(E.INPUT_GROUPS, {
    track: { rays: [['sensors to the left', [0, 1, 2]], ['sensor ahead', [3]], ['sensors to the right', [4, 5, 6]], ['speed', [7]]] },
    arena: { rays: [['walls in view', NA.range(11)], ['drones in view', NA.range(11).map((i) => 11 + i)], ['radar', [22, 23]], ['lives and gun', [24, 25]]] },
    lander: { state: [['position', [0, 1]], ['speed', [2, 3]], ['tilt', [4]], ['spin', [5]], ['fuel', [6]]] },
  });
  /* axes: what the two numbers of behavior() measure (MAP-Elites' grid); demoMs: a move every this many ms when the
     results play it (these move in small steps); trace: a whole generation can be drawn at once */
  Object.assign(E.WORLDS, {
    track: { name: 'Race track', make: (o) => new Track(o), what: '7 distances to the walls and its speed, or a camera above it', goal: 'drive 4 laps without touching the walls', axes: ['average speed', 'side of the track it keeps'], demoMs: 40, trace: true },
    arena: { name: 'Drone arena', make: (o) => new Arena(o), what: 'what each column of its view holds, a radar and its lives, or the picture itself', goal: 'destroy 10 drones before they reach it', axes: ['how much it walks', 'how often it fires'], demoMs: 70, trace: true },
    lander: { name: 'Land a rocket', make: (o) => new Lander(o), what: 'where it is, how fast it moves, its tilt, spin and fuel', goal: 'touch down softly on the pad', axes: ['where it comes down', 'how much it burns'], demoMs: 40, trace: true },
  });
  // =====================================================================================================
  // pictures
  // =====================================================================================================
  const rgbOf = (T, name) => (T.tokens && T.tokens[name]) || [128, 128, 128];
  const mix = (a, b, t) => [a[0] + (b[0] - a[0]) * t, a[1] + (b[1] - a[1]) * t, a[2] + (b[2] - a[2]) * t];
  const css = (c, a = 1) => `rgba(${c[0] | 0},${c[1] | 0},${c[2] | 0},${a})`;
  const clock = () => (typeof performance !== 'undefined' ? performance.now() : 0);
  const fontOf = () => (typeof getComputedStyle === 'function' && typeof document !== 'undefined' && getComputedStyle(document.body).getPropertyValue('--font-ui')) || 'sans-serif';
  const label = (ctx, t, x, y, size, color, align = 'left', w = 650) => { ctx.font = `${w} ${size}px ${fontOf()}`; ctx.fillStyle = color; ctx.textAlign = align; ctx.textBaseline = 'middle'; ctx.fillText(t, x, y); };
  const rounded = (ctx, x, y, w, h, r) => { if (NA.rr) NA.rr(ctx, x, y, w, h, r); else { ctx.beginPath(); ctx.rect(x, y, w, h); } };
  const GHOSTS = ['s1', 's2', 's3', 's4', 's5', 's6', 's7', 's8'];
  // a banner across the middle of a picture: how an episode ended
  function banner(ctx, r, text, color) {
    ctx.font = `700 ${Math.max(13, Math.min(22, r.w / 22))}px ${fontOf()}`;
    const w = ctx.measureText(text).width + 28, h = Math.max(28, Math.min(40, r.w / 14));
    ctx.fillStyle = 'rgba(8,10,18,0.72)'; rounded(ctx, r.x + (r.w - w) / 2, r.y + r.h * 0.42 - h / 2, w, h, 8); ctx.fill();
    ctx.fillStyle = color; ctx.textAlign = 'center'; ctx.textBaseline = 'middle'; ctx.fillText(text, r.x + r.w / 2, r.y + r.h * 0.42);
  }

  // ------------------------------------------------------------------ the race track, from above
  /* The track to scale in r: kerbs on the tight bends, the start line, the car's trail and its seven sensors
     (green far, red near), and the car. o.ghosts: other members of a generation, [{ x, y, th, done }]. */
  function drawTrack(ctx, r, s, T, o = {}) {
    const tr = s.track, [bx0, by0, bx1, by1] = tr.box, m = 3, sc = Math.min(r.w / (bx1 - bx0 + 2 * m), r.h / (by1 - by0 + 2 * m));
    const ox = r.x + (r.w - (bx1 - bx0 + 2 * m) * sc) / 2 - (bx0 - m) * sc, oy = r.y + (r.h - (by1 - by0 + 2 * m) * sc) / 2 - (by0 - m) * sc;
    const X = (x) => ox + x * sc, Y = (y) => oy + y * sc, N = tr.N, ink = rgbOf(T, 'ink'), panel = rgbOf(T, 'panel');
    ctx.save();
    ctx.beginPath(); ctx.rect(r.x, r.y, r.w, r.h); ctx.clip();
    ctx.fillStyle = T.c('good', T.dark ? 0.09 : 0.11); ctx.fillRect(r.x, r.y, r.w, r.h);
    const path = new Path2D();
    for (let k = 0; k <= N; k++) { const q = k % N; if (k) path.lineTo(X(tr.cx[q]), Y(tr.cy[q])); else path.moveTo(X(tr.cx[q]), Y(tr.cy[q])); }
    ctx.lineJoin = 'round'; ctx.lineCap = 'round';
    ctx.strokeStyle = css(mix(panel, ink, 0.62)); ctx.lineWidth = 2 * HW * sc + Math.max(2, sc * 0.9); ctx.stroke(path);
    ctx.strokeStyle = css(mix(panel, ink, T.dark ? 0.16 : 0.2)); ctx.lineWidth = 2 * HW * sc; ctx.stroke(path);
    // kerbs where the track bends tightly: red and white along both edges
    ctx.lineCap = 'butt'; ctx.lineWidth = Math.max(2, sc * 0.75);
    for (let k = 0; k < N; k++) {
      if (Math.abs(tr.curv[k]) < 1 / 16) continue;
      const q = (k + 1) % N, nx = -tr.ty[k], ny = tr.tx[k], nx2 = -tr.ty[q], ny2 = tr.tx[q], e = HW - 0.35;
      ctx.strokeStyle = Math.floor(k / 2) % 2 ? T.c('critical', 0.9) : css(panel, 0.95);
      for (const sd of [-1, 1]) { ctx.beginPath(); ctx.moveTo(X(tr.cx[k] + sd * e * nx), Y(tr.cy[k] + sd * e * ny)); ctx.lineTo(X(tr.cx[q] + sd * e * nx2), Y(tr.cy[q] + sd * e * ny2)); ctx.stroke(); }
    }
    ctx.setLineDash([Math.max(3, 2 * sc), Math.max(4, 3 * sc)]); ctx.strokeStyle = T.c('ink', 0.16); ctx.lineWidth = 1; ctx.stroke(path); ctx.setLineDash([]);
    // the start line, chequered across the track
    {
      const nx = -tr.ty[0], ny = tr.tx[0], cells = 6, cs = (2 * HW) / cells;
      for (let i = 0; i < cells; i++) for (let j = 0; j < 2; j++) {
        const a = -HW + i * cs, b = (j - 1) * cs * 0.9, x0 = tr.cx[0] + a * nx + b * tr.tx[0], y0 = tr.cy[0] + a * ny + b * tr.ty[0];
        ctx.fillStyle = (i + j) % 2 ? css(ink, 0.85) : css(panel, 0.95);
        ctx.beginPath(); ctx.moveTo(X(x0), Y(y0)); ctx.lineTo(X(x0 + cs * nx), Y(y0 + cs * ny)); ctx.lineTo(X(x0 + cs * nx + cs * 0.9 * tr.tx[0]), Y(y0 + cs * ny + cs * 0.9 * tr.ty[0])); ctx.lineTo(X(x0 + cs * 0.9 * tr.tx[0]), Y(y0 + cs * 0.9 * tr.ty[0])); ctx.closePath(); ctx.fill();
      }
    }
    const L = Math.max(9, 4.4 * sc), Wd = Math.max(5, 2 * sc);
    // a generation, all driving at once
    (o.ghosts || []).forEach((g, k) => { const q = g.k ?? k; drawCar(ctx, X(g.x), Y(g.y), g.th, L * (q ? 0.92 : 1.1), Wd * (q ? 0.92 : 1.1), T.c(GHOSTS[q % 8], g.done ? 0.25 : q ? 0.62 : 1), g.done === 'crashed', q ? null : T); });
    if (!o.ghostsOnly) {
      if (s.trail && s.trail.length > 6) {
        ctx.strokeStyle = T.c('s2', 0.55); ctx.lineWidth = Math.max(1.5, sc * 0.35); ctx.lineJoin = 'round'; ctx.beginPath();
        for (let i = 0; i < s.trail.length; i += 3) { if (i) ctx.lineTo(X(s.trail[i]), Y(s.trail[i + 1])); else ctx.moveTo(X(s.trail[i]), Y(s.trail[i + 1])); }
        ctx.stroke();
      }
      // its sensors: a line to each wall it sees, green when far and red when near
      if (s.rays && !s.end) {
        const good = rgbOf(T, 'good'), bad = rgbOf(T, 'critical');
        s.rays.forEach((d, i) => {
          const a = s.th + RAYS[i] * DEG, ex = s.x + Math.cos(a) * d, ey = s.y + Math.sin(a) * d, c = mix(bad, good, clamp(d / 20, 0, 1));
          ctx.strokeStyle = css(c, 0.75); ctx.lineWidth = 1.2; ctx.beginPath(); ctx.moveTo(X(s.x), Y(s.y)); ctx.lineTo(X(ex), Y(ey)); ctx.stroke();
          ctx.fillStyle = css(c, 0.95); ctx.beginPath(); ctx.arc(X(ex), Y(ey), 2.4, 0, TAU); ctx.fill();
        });
      }
      drawCar(ctx, X(s.x), Y(s.y), s.th, L, Wd, s.end === 'crashed' ? T.c('critical') : T.c('s1'), s.end === 'crashed', T);
      // laps done (filled) and the speedometer (not in a small picture)
      if (!o.small) {
        const lr = 5, lx = r.x + r.w - 10 - (s.laps - 1) * 14, ly = r.y + 12;
        for (let i = 0; i < s.laps; i++) { ctx.beginPath(); ctx.arc(lx + i * 14, ly, lr, 0, TAU); ctx.fillStyle = i < s.lap ? T.c('good') : T.c('ink', 0.14); ctx.fill(); }
        label(ctx, `${Math.round(s.v)} m/s`, r.x + r.w - 10, ly + 18, 11.5, T.c('ink-2'), 'right', 600);
        const sw = 52; ctx.fillStyle = T.c('ink', 0.12); ctx.fillRect(r.x + r.w - 10 - sw, ly + 29, sw, 4); ctx.fillStyle = T.c('s1'); ctx.fillRect(r.x + r.w - 10 - sw, ly + 29, sw * clamp(s.v / VMAX, 0, 1), 4);
      }
      if (s.end) banner(ctx, r, s.end === 'crashed' ? 'Crashed' : s.end === 'finished' ? `${s.laps} laps!` : 'Out of time', s.end === 'finished' ? T.c('good') : s.end === 'crashed' ? T.c('critical') : '#ffffff');
    }
    ctx.restore();
  }
  // a car seen from above, pointing along th: body, windscreen, and a puff of smoke when it has crashed
  function drawCar(ctx, x, y, th, L, W, fill, crashed, T) {
    ctx.save(); ctx.translate(x, y); ctx.rotate(th);
    ctx.fillStyle = 'rgba(0,0,0,0.22)'; rounded(ctx, -L / 2 + 1, -W / 2 + 1.5, L, W, W * 0.32); ctx.fill();
    ctx.fillStyle = fill; rounded(ctx, -L / 2, -W / 2, L, W, W * 0.32); ctx.fill();
    ctx.fillStyle = 'rgba(10,14,24,0.55)'; rounded(ctx, L * 0.05, -W * 0.34, L * 0.22, W * 0.68, W * 0.15); ctx.fill();
    if (T) { ctx.fillStyle = 'rgba(255,255,255,0.8)'; ctx.fillRect(L / 2 - 1.5, -W * 0.38, 1.5, W * 0.22); ctx.fillRect(L / 2 - 1.5, W * 0.16, 1.5, W * 0.22); }
    if (crashed) { ctx.strokeStyle = 'rgba(255,255,255,0.85)'; ctx.lineWidth = 1.6; ctx.beginPath(); ctx.moveTo(-W * 0.4, -W * 0.4); ctx.lineTo(W * 0.4, W * 0.4); ctx.moveTo(W * 0.4, -W * 0.4); ctx.lineTo(-W * 0.4, W * 0.4); ctx.stroke(); }
    ctx.restore();
  }

  // ------------------------------------------------------------------ the drone arena, in the first person
  /* What it sees, raycast column by column (walls lit by distance and side, with panel seams and a glowing
     trim), the drones as glowing orbs hidden by any wall in front of them, its gun (a flash and a tracer when it
     fires, a burst where a shot lands, red at the edges when a drone reaches it), its lives, drones destroyed
     and time; and in the corner a map: walls, it and its field of view, the drones (o.ghosts: a generation). */
  function drawArena(ctx, r, s, T, o = {}) {
    const M = s.map, t = (o.now === undefined ? clock() : o.now) / 1000;
    let w = r.w, h = r.h;
    if (w / h > 2) w = h * 2; else if (w / h < 1.25) h = w / 1.25;
    const V = { x: r.x + (r.w - w) / 2, y: r.y + (r.h - h) / 2, w, h }, hz = V.y + V.h * 0.5, base = rgbOf(T, 's1'), glow = rgbOf(T, 's2');
    ctx.save();
    rounded(ctx, V.x, V.y, V.w, V.h, 10); ctx.clip();
    let g = ctx.createLinearGradient(0, V.y, 0, hz);
    g.addColorStop(0, '#05070f'); g.addColorStop(1, '#151c33'); ctx.fillStyle = g; ctx.fillRect(V.x, V.y, V.w, V.h / 2 + 1);
    g = ctx.createLinearGradient(0, hz, 0, V.y + V.h);
    g.addColorStop(0, '#0d1224'); g.addColorStop(1, '#2a3354'); ctx.fillStyle = g; ctx.fillRect(V.x, hz, V.w, V.h / 2);
    // the floor's grid, to show depth: a line every metre
    ctx.strokeStyle = css(base, 0.1); ctx.lineWidth = 1;
    for (let d = 1; d < 12; d++) { const y = hz + V.w / (1.3 * d) / 2; if (y > V.y + V.h) continue; ctx.beginPath(); ctx.moveTo(V.x, y); ctx.lineTo(V.x + V.w, y); ctx.stroke(); }
    // the walls
    const n = Math.max(60, Math.min(340, Math.floor(V.w / 2))), cw = V.w / n, dx = Math.cos(s.th), dy = Math.sin(s.th), pl = Math.tan(FOV / 2), z = new Float32Array(n), dark = [6, 9, 18];
    // steel panels tinted with the first series colour, two to a block, lit by distance and side
    const steel = mix(base, [92, 104, 132], 0.55), trim = mix(base, [255, 255, 255], 0.45);
    for (let c = 0; c < n; c++) {
      const cam = (2 * (c + 0.5)) / n - 1, rx = dx - dy * pl * cam, ry = dy + dx * pl * cam, a = Math.atan2(ry, rx), hit = castRay(M, s.x, s.y, a, 30);
      const perp = Math.max(0.05, hit.d * Math.cos(a - s.th)), lh = V.w / (1.3 * perp), top = hz - lh / 2, fog = 1 / (1 + 0.045 * perp * perp), u2 = (hit.u * 2) % 1, seam = u2 < 0.03 || u2 > 0.97;
      z[c] = perp;
      const lit = (hit.side ? 0.58 : 0.86) * fog * (0.9 + 0.1 * Math.cos(hit.u * TAU)), col = mix(dark, steel, lit * (seam ? 0.5 : 1)), x = V.x + c * cw, cw1 = cw + 0.7, line = Math.max(1, lh * 0.012);
      ctx.fillStyle = css(col); ctx.fillRect(x, top, cw1, lh);
      ctx.fillStyle = css(mix(col, dark, 0.5)); ctx.fillRect(x, top + lh * 0.22, cw1, line); ctx.fillRect(x, top + lh * 0.62, cw1, line);
      ctx.fillStyle = css(mix(col, [255, 255, 255], 0.07)); ctx.fillRect(x, top + lh * 0.22 + line, cw1, Math.max(1, lh * 0.006));
      // a light strip along the foot of the wall
      ctx.fillStyle = css(trim, 0.85 * fog); ctx.fillRect(x, top + lh * 0.9, cw1, Math.max(1, lh * 0.018));
      ctx.fillStyle = css(trim, 0.18 * fog); ctx.fillRect(x, top + lh * 0.86, cw1, Math.max(1, lh * 0.04));
    }
    // the drones, far ones first, each hidden by any wall nearer than it
    const sprites = [];
    for (const d of s.drones) {
      if (!d.alive) continue;
      const ex = d.x - s.x, ey = d.y - s.y, depth = dx * ex + dy * ey, side = (dx * ey - dy * ex) / pl;
      if (depth < 0.15) continue;
      sprites.push({ depth, sx: V.x + (V.w / 2) * (1 + side / depth), size: (V.w / (1.3 * depth)) * 2 * DRONE_R, d });
    }
    sprites.sort((a, b) => b.depth - a.depth);
    let aimed = false;
    for (const sp of sprites) {
      const R = sp.size / 2, cy = hz - sp.size * 0.15 + Math.sin(t * 3 + sp.d.x * 2) * sp.size * 0.06, c0 = Math.max(0, Math.floor((sp.sx - R * 1.6 - V.x) / cw)), c1 = Math.min(n - 1, Math.ceil((sp.sx + R * 1.6 - V.x) / cw));
      const clip = new Path2D();
      let any = false;
      for (let c = c0; c <= c1; c++) if (z[c] > sp.depth - DRONE_R) { clip.rect(V.x + c * cw, V.y, cw + 0.7, V.h); any = true; }
      if (!any) continue;
      if (Math.abs(sp.sx - (V.x + V.w / 2)) < R && z[Math.floor(n / 2)] > sp.depth - DRONE_R) aimed = true;
      ctx.save(); ctx.clip(clip);
      // a hovering orb: its glow, a dark shell, a ring that turns, and a camera eye glowing in the middle
      const fade = 1 / (1 + 0.02 * sp.depth * sp.depth), halo = ctx.createRadialGradient(sp.sx, cy, R * 0.3, sp.sx, cy, R * 2.4);
      halo.addColorStop(0, css(glow, 0.5 * fade)); halo.addColorStop(1, css(glow, 0)); ctx.fillStyle = halo; ctx.beginPath(); ctx.arc(sp.sx, cy, R * 2.4, 0, TAU); ctx.fill();
      const shell = ctx.createRadialGradient(sp.sx - R * 0.35, cy - R * 0.4, R * 0.1, sp.sx, cy, R);
      shell.addColorStop(0, '#5d6680'); shell.addColorStop(0.6, '#262c40'); shell.addColorStop(1, '#10131f'); ctx.fillStyle = shell; ctx.beginPath(); ctx.arc(sp.sx, cy, R, 0, TAU); ctx.fill();
      const spin = t * 2.2 + sp.d.y;
      ctx.strokeStyle = css(mix(glow, [255, 255, 255], 0.35), 0.95); ctx.lineWidth = Math.max(1.2, R * 0.1); ctx.beginPath(); ctx.ellipse(sp.sx, cy, R * 1.45, R * (0.25 + 0.12 * Math.sin(spin)), 0.12 * Math.sin(spin * 0.7), 0, TAU); ctx.stroke();
      const eye = ctx.createRadialGradient(sp.sx, cy, 0, sp.sx, cy, R * 0.48);
      eye.addColorStop(0, '#fffaf0'); eye.addColorStop(0.35, css(mix(glow, [255, 255, 255], 0.25))); eye.addColorStop(1, css(glow, 0)); ctx.fillStyle = eye; ctx.beginPath(); ctx.arc(sp.sx, cy, R * 0.48, 0, TAU); ctx.fill();
      ctx.fillStyle = 'rgba(255,255,255,0.35)'; ctx.beginPath(); ctx.arc(sp.sx - R * 0.42, cy - R * 0.45, R * 0.14, 0, TAU); ctx.fill();
      ctx.restore();
    }
    const fx = s.fx || {}, cx = V.x + V.w / 2;
    // where a shot landed: a burst
    if (fx.hit) {
      const ex = fx.hit[0] - s.x, ey = fx.hit[1] - s.y, depth = Math.max(0.3, dx * ex + dy * ey), sx = V.x + (V.w / 2) * (1 + (dx * ey - dy * ex) / pl / depth), R = (V.w / (1.3 * depth)) * DRONE_R;
      for (let k = 0; k < 10; k++) { const a = (k / 10) * TAU + t; ctx.strokeStyle = css(mix(glow, [255, 255, 255], 0.4), 0.9); ctx.lineWidth = 2; ctx.beginPath(); ctx.moveTo(sx + Math.cos(a) * R * 0.8, hz + Math.sin(a) * R * 0.8); ctx.lineTo(sx + Math.cos(a) * R * 2.2, hz + Math.sin(a) * R * 2.2); ctx.stroke(); }
      ctx.strokeStyle = css(glow, 0.7); ctx.lineWidth = 3; ctx.beginPath(); ctx.arc(sx, hz, R * 1.6, 0, TAU); ctx.stroke();
    }
    // the gun, a flash and a tracer when it fires
    const gw = V.w * 0.13, gh = V.h * 0.26, kick = fx.shot ? gh * 0.08 : 0, gx = cx + V.w * 0.07, gy = V.y + V.h - gh + kick;
    if (fx.shot) {
      ctx.strokeStyle = css(mix(glow, [255, 255, 255], 0.6), 0.85); ctx.lineWidth = 2.5; ctx.beginPath(); ctx.moveTo(gx - gw * 0.12, gy + gh * 0.05); ctx.lineTo(cx, hz); ctx.stroke();
      const fl = ctx.createRadialGradient(gx - gw * 0.12, gy, 0, gx - gw * 0.12, gy, gw * 0.6);
      fl.addColorStop(0, 'rgba(255,250,220,0.95)'); fl.addColorStop(0.4, css(glow, 0.7)); fl.addColorStop(1, css(glow, 0)); ctx.fillStyle = fl; ctx.beginPath(); ctx.arc(gx - gw * 0.12, gy, gw * 0.6, 0, TAU); ctx.fill();
    }
    ctx.fillStyle = '#1a2036'; ctx.beginPath(); ctx.moveTo(gx - gw * 0.22, gy); ctx.lineTo(gx + gw * 0.05, gy); ctx.lineTo(gx + gw * 0.55, V.y + V.h); ctx.lineTo(gx - gw * 0.45, V.y + V.h); ctx.closePath(); ctx.fill();
    ctx.fillStyle = css(base, 0.85); ctx.beginPath(); ctx.moveTo(gx - gw * 0.16, gy + gh * 0.12); ctx.lineTo(gx + gw * 0.02, gy + gh * 0.12); ctx.lineTo(gx + gw * 0.2, gy + gh * 0.5); ctx.lineTo(gx - gw * 0.12, gy + gh * 0.5); ctx.closePath(); ctx.fill();
    ctx.fillStyle = s.cool ? 'rgba(255,255,255,0.18)' : css(glow, 0.95); ctx.fillRect(gx - gw * 0.1, gy + gh * 0.6, gw * 0.22, gh * 0.06);
    // the sights: green with a drone in them
    ctx.strokeStyle = aimed ? T.c('good') : 'rgba(255,255,255,0.85)'; ctx.lineWidth = 2;
    for (const [ax, ay] of [[1, 0], [-1, 0], [0, 1], [0, -1]]) { ctx.beginPath(); ctx.moveTo(cx + ax * 5, hz + ay * 5); ctx.lineTo(cx + ax * 12, hz + ay * 12); ctx.stroke(); }
    // a drone reached it: red at the edges
    if (fx.hurt) { const v = ctx.createRadialGradient(cx, hz, V.h * 0.25, cx, hz, V.w * 0.7); v.addColorStop(0, 'rgba(208,59,59,0)'); v.addColorStop(1, 'rgba(208,59,59,0.6)'); ctx.fillStyle = v; ctx.fillRect(V.x, V.y, V.w, V.h); }
    // lives, drones destroyed, time, and the map in the corner (only the time in a small picture)
    ctx.fillStyle = 'rgba(255,255,255,0.12)'; ctx.fillRect(V.x, V.y, V.w, 3); ctx.fillStyle = css(glow, 0.9); ctx.fillRect(V.x, V.y, V.w * clamp(s.t / s.maxSteps, 0, 1), 3);
    if (o.small) { ctx.restore(); return; }
    const pip = Math.max(7, V.h * 0.03);
    for (let i = 0; i < s.lives; i++) { const x = V.x + 14 + i * pip * 2.4, y = V.y + V.h - 16; ctx.fillStyle = i < s.health ? T.c('critical') : 'rgba(255,255,255,0.18)'; ctx.beginPath(); ctx.moveTo(x, y + pip * 0.7); ctx.lineTo(x - pip, y - pip * 0.2); ctx.arc(x - pip * 0.5, y - pip * 0.35, pip * 0.52, Math.PI, 0); ctx.arc(x + pip * 0.5, y - pip * 0.35, pip * 0.52, Math.PI, 0); ctx.closePath(); ctx.fill(); }
    label(ctx, `${s.kills} / ${s.goal} drones`, V.x + V.w - 14, V.y + V.h - 16, Math.max(11.5, V.h * 0.045), '#ffffff', 'right', 700);
    const ms = Math.min(V.h * 0.32, V.w * 0.22), cs = ms / Math.max(M.W, M.H), mx = V.x + V.w - ms - 10, my = V.y + 12;
    ctx.fillStyle = 'rgba(4,6,14,0.72)'; rounded(ctx, mx - 5, my - 5, ms + 10, ms + 10, 6); ctx.fill();
    ctx.fillStyle = css(steel, 0.85);
    for (let j = 0; j < M.H; j++) for (let i = 0; i < M.W; i++) if (M.rows[j][i] === '#') ctx.fillRect(mx + i * cs, my + j * cs, cs + 0.3, cs + 0.3);
    ctx.fillStyle = 'rgba(255,255,255,0.13)'; ctx.beginPath(); ctx.moveTo(mx + s.x * cs, my + s.y * cs); ctx.arc(mx + s.x * cs, my + s.y * cs, cs * 4, s.th - FOV / 2, s.th + FOV / 2); ctx.closePath(); ctx.fill();
    (o.ghosts || []).forEach((q, k) => { ctx.fillStyle = T.c(GHOSTS[(q.k ?? k) % 8], q.done ? 0.25 : 0.75); ctx.beginPath(); ctx.arc(mx + q.x * cs, my + q.y * cs, Math.max(1.5, cs * 0.18), 0, TAU); ctx.fill(); });
    for (const d of s.drones) if (d.alive) { ctx.fillStyle = css(glow, 0.95); ctx.beginPath(); ctx.arc(mx + d.x * cs, my + d.y * cs, Math.max(2, cs * 0.28), 0, TAU); ctx.fill(); }
    ctx.fillStyle = '#ffffff'; ctx.beginPath(); ctx.arc(mx + s.x * cs, my + s.y * cs, Math.max(2.2, cs * 0.3), 0, TAU); ctx.fill();
    if (s.end) banner(ctx, V, s.end === 'cleared' ? 'Arena cleared' : s.end === 'dead' ? 'Out of lives' : 'Out of time', s.end === 'cleared' ? T.c('good') : s.end === 'dead' ? T.c('critical') : '#ffffff');
    ctx.restore();
  }

  // ------------------------------------------------------------------ the rocket
  // stars at the same places every time: [across, down, size]
  const STARS = (() => { const r = seeded(77); return NA.range(70).map(() => [r.next(), r.next() * 0.75, r.next()]); })();
  /* The rocket over the ground, 72 m across and 40 m up around the pad: the night sky, the hills and the pad (lit
     green once it has landed), the wind, its trail, and the rocket with its flame and side puffs; its height,
     speed and fuel. o.ghosts: a generation, [{ x, y, th, done }]. */
  function drawLander(ctx, r, s, T, o = {}) {
    const t = (o.now === undefined ? clock() : o.now) / 1000, spanX = 72, spanY = 40, sc = Math.min(r.w / spanX, r.h / spanY), w = spanX * sc, h = spanY * sc, pad = s.pad;
    const V = { x: r.x + (r.w - w) / 2, y: r.y + (r.h - h) / 2, w, h }, X = (x) => V.x + (x - pad + spanX / 2) * sc, Y = (y) => V.y + h - (y + 4) * sc, accent = rgbOf(T, 's1'), fire = rgbOf(T, 's2');
    ctx.save();
    rounded(ctx, V.x, V.y, V.w, V.h, 10); ctx.clip();
    let g = ctx.createLinearGradient(0, V.y, 0, V.y + V.h);
    g.addColorStop(0, '#04060f'); g.addColorStop(1, '#1b2442'); ctx.fillStyle = g; ctx.fillRect(V.x, V.y, V.w, V.h);
    for (const [u, v, z] of STARS) { ctx.fillStyle = `rgba(255,255,255,${(0.25 + 0.5 * z) * (0.75 + 0.25 * Math.sin(t * 2 + u * 40))})`; ctx.fillRect(V.x + u * V.w, V.y + v * V.h, z > 0.8 ? 2 : 1.2, z > 0.8 ? 2 : 1.2); }
    // the ground
    ctx.beginPath(); ctx.moveTo(V.x, V.y + V.h);
    for (let x = pad - spanX / 2; x <= pad + spanX / 2 + 0.5; x += 0.75) ctx.lineTo(X(x), Y(s.ground(x)));
    ctx.lineTo(V.x + V.w, V.y + V.h); ctx.closePath();
    g = ctx.createLinearGradient(0, Y(8), 0, V.y + V.h); g.addColorStop(0, '#3a4262'); g.addColorStop(1, '#151a2c'); ctx.fillStyle = g; ctx.fill();
    ctx.strokeStyle = 'rgba(200,210,240,0.35)'; ctx.lineWidth = 1.2; ctx.stroke();
    // the pad: lit along its edge, its lights blinking (green once it has landed)
    const landed = s.end === 'landed', pc = landed ? rgbOf(T, 'good') : accent;
    ctx.fillStyle = css(mix(pc, [255, 255, 255], 0.15)); ctx.fillRect(X(pad - PAD), Y(0), 2 * PAD * sc, Math.max(3, 0.45 * sc));
    if (landed) { const gl = ctx.createRadialGradient(X(pad), Y(0), 0, X(pad), Y(0), PAD * sc * 1.4); gl.addColorStop(0, css(pc, 0.45)); gl.addColorStop(1, css(pc, 0)); ctx.fillStyle = gl; ctx.beginPath(); ctx.arc(X(pad), Y(0), PAD * sc * 1.4, 0, TAU); ctx.fill(); }
    for (const e of [-PAD, PAD]) { ctx.fillStyle = Math.sin(t * 5) > 0 || landed ? css(mix(pc, [255, 255, 255], 0.4)) : 'rgba(255,255,255,0.15)'; ctx.beginPath(); ctx.arc(X(pad + e), Y(0) - 3, 2.6, 0, TAU); ctx.fill(); }
    // the wind: an arrow along the top, as long as it is strong
    if (s.wind) { const ax = V.x + V.w / 2, ay = V.y + 14, L = clamp(s.wind * 22, -V.w * 0.3, V.w * 0.3); ctx.strokeStyle = 'rgba(255,255,255,0.6)'; ctx.lineWidth = 2; ctx.beginPath(); ctx.moveTo(ax, ay); ctx.lineTo(ax + L, ay); ctx.lineTo(ax + L - Math.sign(L) * 6, ay - 4); ctx.moveTo(ax + L, ay); ctx.lineTo(ax + L - Math.sign(L) * 6, ay + 4); ctx.stroke(); label(ctx, 'wind', ax, ay + 12, 10.5, 'rgba(255,255,255,0.6)', 'center', 600); }
    (o.ghosts || []).forEach((q, k) => { const g = q.k ?? k; drawRocket(ctx, X(q.x), Y(q.y), q.th, sc * (g ? 0.9 : 1), css(mix(rgbOf(T, GHOSTS[g % 8]), [255, 255, 255], 0.2), q.done ? 0.25 : g ? 0.55 : 1), null); });
    if (!o.ghostsOnly) {
      if (s.trail && s.trail.length > 6) {
        ctx.fillStyle = 'rgba(255,255,255,0.35)';
        for (let i = 0; i < s.trail.length; i += 6) ctx.fillRect(X(s.trail[i]) - 1, Y(s.trail[i + 1]) - 1, 2, 2);
      }
      const crashed = s.end === 'crashed' || s.end === 'away' || s.end === 'time' || s.end === 'offpad';
      if (s.end === 'crashed') { for (let k = 0; k < 3; k++) { const b = ctx.createRadialGradient(X(s.x), Y(s.y), 0, X(s.x), Y(s.y), sc * (2 + k * 1.6)); b.addColorStop(0, 'rgba(255,240,200,0.8)'); b.addColorStop(0.5, css(fire, 0.45)); b.addColorStop(1, css(fire, 0)); ctx.fillStyle = b; ctx.beginPath(); ctx.arc(X(s.x), Y(s.y), sc * (2 + k * 1.6), 0, TAU); ctx.fill(); } }
      drawRocket(ctx, X(s.x), Y(s.y), s.th, sc, crashed && s.end === 'crashed' ? 'rgba(220,225,235,0.5)' : '#dfe4ee', s.end ? null : { engine: s.engine, turn: s.turn, fire, t });
      // height, speed and fuel (not in a small picture)
      if (!o.small) {
        const hx = V.x + 12, hy = V.y + 14, f = clamp(s.fuel / s.fuelMax, 0, 1);
        label(ctx, `height ${s.y.toFixed(1)} m`, hx, hy, 11.5, 'rgba(255,255,255,0.85)', 'left', 600);
        label(ctx, `falling ${Math.max(0, -s.vy).toFixed(1)} m/s · across ${Math.abs(s.vx).toFixed(1)}`, hx, hy + 16, 11.5, 'rgba(255,255,255,0.7)', 'left', 500);
        label(ctx, 'fuel', hx, hy + 32, 10.5, 'rgba(255,255,255,0.7)', 'left', 600);
        ctx.fillStyle = 'rgba(255,255,255,0.15)'; ctx.fillRect(hx + 28, hy + 29, 70, 6); ctx.fillStyle = f > 0.2 ? css(accent) : T.c('critical'); ctx.fillRect(hx + 28, hy + 29, 70 * f, 6);
      }
      if (s.end) banner(ctx, V, landed ? 'Landed' : s.end === 'offpad' ? 'Down, but off the pad' : s.end === 'away' ? 'Flew away' : s.end === 'time' ? 'Out of time' : 'Crashed', landed ? T.c('good') : T.c('critical'));
    }
    ctx.restore();
  }
  // the rocket at (x, y) on screen, tilted th, sc pixels a metre; f: its flame and puffs ({ engine, turn, fire, t })
  function drawRocket(ctx, x, y, th, sc, body, f) {
    ctx.save(); ctx.translate(x, y); ctx.rotate(th); ctx.scale(sc, -sc);
    if (f && f.engine) {
      const L = 2.2 + 0.6 * Math.sin(f.t * 40) + 0.4 * Math.sin(f.t * 23), fl = ctx.createLinearGradient(0, -1.2, 0, -1.2 - L);
      fl.addColorStop(0, 'rgba(255,250,220,0.95)'); fl.addColorStop(0.35, css(f.fire, 0.85)); fl.addColorStop(1, css(f.fire, 0));
      ctx.fillStyle = fl; ctx.beginPath(); ctx.moveTo(-0.5, -1.2); ctx.lineTo(0.5, -1.2); ctx.lineTo(0, -1.2 - L); ctx.closePath(); ctx.fill();
    }
    if (f && f.turn) { const sx = f.turn < 0 ? 1.2 : -1.2; ctx.fillStyle = 'rgba(235,240,255,0.75)'; ctx.beginPath(); ctx.arc(sx + Math.sign(sx) * 0.45, 0.9, 0.42, 0, TAU); ctx.arc(sx + Math.sign(sx) * 0.95, 0.95, 0.28, 0, TAU); ctx.fill(); }
    ctx.lineWidth = 1.6 / sc; ctx.strokeStyle = body; ctx.lineCap = 'round';
    ctx.beginPath(); ctx.moveTo(-0.8, -0.4); ctx.lineTo(-FOOT, -LEG); ctx.moveTo(0.8, -0.4); ctx.lineTo(FOOT, -LEG); ctx.moveTo(-FOOT - 0.35, -LEG); ctx.lineTo(-FOOT + 0.35, -LEG); ctx.moveTo(FOOT - 0.35, -LEG); ctx.lineTo(FOOT + 0.35, -LEG); ctx.stroke();
    ctx.fillStyle = '#7d869c'; ctx.beginPath(); ctx.moveTo(-0.4, -0.55); ctx.lineTo(0.4, -0.55); ctx.lineTo(0.62, -1.2); ctx.lineTo(-0.62, -1.2); ctx.closePath(); ctx.fill();
    ctx.fillStyle = body; rounded(ctx, -1.1, -0.6, 2.2, 2.0, 0.55); ctx.fill();
    ctx.fillStyle = 'rgba(30,40,70,0.9)'; ctx.beginPath(); ctx.arc(0, 0.55, 0.42, 0, TAU); ctx.fill();
    ctx.fillStyle = 'rgba(160,200,255,0.55)'; ctx.beginPath(); ctx.arc(-0.12, 0.66, 0.15, 0, TAU); ctx.fill();
    ctx.restore();
  }
  Object.assign(E.DRAW, { track: drawTrack, arena: drawArena, lander: drawLander });

  Object.assign(E, { Track, Arena, Lander, buildTrack, trackOf, randomTrack, centerDist, castRay, mapOf, HW });
})(typeof globalThis !== 'undefined' && globalThis.NA ? globalThis.NA : window.NA);
