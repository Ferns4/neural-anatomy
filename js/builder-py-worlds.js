/* builder-py-worlds.js — for builder-py.js (the model builder's PyTorch scripts): the three bigger worlds of
   rl-worlds.js in Python (the race track, the drone arena seen in the first person, the rocket), their physics
   step for step as on the page (their random numbers come from Python's random, so the starts, where the drones
   appear and the random tracks differ), each measuring its style of play for MAP-Elites; and the evolutionary
   methods of builder-evo.js: a genetic algorithm with its ways of choosing, mixing and mutating parents, grow and
   prune (networks whose shape evolves: lists of weight matrices that gain or lose neurons, and gain layers that
   start with one neuron), novelty search, evolution strategies, the cross-entropy method, MAP-Elites and ERL (a
   genetic algorithm teaching a DQN); and grow and prune while a network learns from examples. */
(function (NA) {
  'use strict';
  const PW = { worlds: {} };
  const rows = (s) => s.split('\n');
  const terms = (list) => list.filter(Boolean).join(' + ') || '0.0';

  // ------------------------------------------------------------------ the race track
  PW.worlds.track = function (d, h) {
    const ws = d.ws, rw = h.rw, f = h.f, num = h.num, say = h.say, E = NA.rlEnvs;
    const id = ws.track === 'random' ? 'random' : E.TRACKS[ws.track] ? ws.track : 'esses', random = id === 'random', cam = ws.view === 'camera';
    const steps = ws.steps > 0 ? Math.min(ws.steps, 6000) : 0, pts = (k) => `[${E.TRACKS[k].pts.map(([x, y]) => `(${x}, ${y})`).join(', ')}]`;
    const W = E.make('track', ws), laps = W.laps, extra = d.extra || [], heading = !!f || extra.includes('its heading'), bends = !!f || extra.includes('the bends ahead');
    const reward = f ? ['        progress, speed, crashed, lap, t = ds / T.L, self.v, float(crashed), float(lap), self.t', '        center, turn, brake = min(1.0, abs(side) / HW), float(steer), float(not gas)',
      '        wall, ahead, finished, laps = max(0.0, HW - abs(side)), self.rays[3], float(self.end == "finished"), self.lap',
      '        side, bend, heading = min(max(side / HW, -1.0), 1.0), self.bend(0) / DEG, self.heading() / DEG', ...h.ownLine]
      : ['        progress = ds / T.L                        # laps of progress this step', `        r = ${terms([rw.rProgress ? `${num(rw.rProgress)} * progress` : null, rw.rLap ? `(${num(rw.rLap)} if lap else 0.0)` : null, rw.rCrash ? `(${num(rw.rCrash)} if crashed else 0.0)` : null, rw.rStep ? num(rw.rStep) : null,
        rw.rSpeed ? `${num(rw.rSpeed)} * self.v` : null, rw.rCenter ? `${num(rw.rCenter)} * min(1.0, abs(side) / HW)` : null, rw.rSteer ? `(${num(rw.rSteer)} if steer else 0.0)` : null,
        rw.rBrake ? `(${num(rw.rBrake)} if not gas else 0.0)` : null, rw.rNearWall ? `(${num(rw.rNearWall)} if HW - abs(side) < 2 else 0.0)` : null, rw.rFinish ? `(${num(rw.rFinish)} if self.end == "finished" else 0.0)` : null])}`];
    const pays = h.pays(`${say(rw.rProgress)} for a lap of progress (backwards counts against it)${rw.rCrash ? `, ${say(rw.rCrash)} for crashing` : ''}${rw.rLap ? `, ${say(rw.rLap)} for each lap done` : ''}${rw.rStep ? `, ${say(rw.rStep)} a step` : ''}${rw.rFinish ? `, ${say(rw.rFinish)} for finishing the last lap` : ''}${rw.rSpeed ? `, ${say(rw.rSpeed)} × its speed (m/s) every step` : ''}${rw.rCenter ? `, ${say(rw.rCenter)} × how far it is from the middle (1 at a wall) every step` : ''}${rw.rNearWall ? `, ${say(rw.rNearWall)} every step within 2 m of a wall` : ''}${rw.rSteer ? `, ${say(rw.rSteer)} every step it steers` : ''}${rw.rBrake ? `, ${say(rw.rBrake)} every step it brakes` : ''}.`);
    const obs = cam ? rows(String.raw`    def obs(self):                                 # (2, 16, 16): the track from above, ahead up, 32 m from 4 m behind and 12 m to each side; its speed
        c, s, T = math.cos(self.th), math.sin(self.th), self.track
        near = [[0.0] * 16 for _ in range(16)]
        for i in range(16):
            ahead = (15.5 - i) / 16 * 32 - 4
            for j in range(16):
                right = (j + 0.5) / 16 * 24 - 12
                near[i][j] = float(T.center_dist(self.x + ahead * c - right * s, self.y + ahead * s + right * c) < HW)
        return torch.tensor([near, [[self.v / VMAX] * 16] * 16])`)
      : rows(String.raw`    def obs(self):                                 # 7 distances to the walls (of ${String(+W.range)} m) and its speed (of ${String(+W.vmax)} m/s)
        return torch.tensor([r / RANGE for r in self.rays] + [self.v / VMAX])`);
    return [
      ...rows(String.raw`TRACK = ${pts(random ? 'oval' : id)}   # ${random ? 'the Oval, if no random track works out' : E.TRACKS[id].name}: points its middle line passes through, m
HW = 4.5                                           # half the track's width, m
VMAX, VMIN, ACC, BRAKE, GRIP, STEER = ${num(W.vmax)}, 3.0, 9.0, 16.0, ${num(W.grip)}, 1.6   # speeds (m/s), throttle and brake (m/s²), grip (m/s² sideways), steering (rad/s)
CAR_R, DT, RANGE = 1.0, 0.1, ${num(W.range)}${' '.repeat(Math.max(1, 21 - num(W.range).length))}# the car's radius (m), a step (s), how far its sensors see (m)
LAPS, DEG = ${laps}, math.pi / 180${' '.repeat(Math.max(1, 30 - String(laps).length))}# the laps to drive; a degree in radians
RAYS = [math.radians(a) for a in (-90, -50, -20, 0, 20, 50, 90)]   # its 7 sensors, from straight ahead


def wrap(a):                                       # an angle, from -pi to pi
    return (a + math.pi) % (2 * math.pi) - math.pi


class Track:
    """A closed loop through the points (a centripetal Catmull-Rom spline), resampled to points 1 m apart along its
    middle line. Its walls are found from a grid of distances to that line (1 m cells), so a sensor's ray steps
    along by the distance to the nearest wall each time (sphere tracing)."""

    def __init__(self, pts, check=None):
        n, line = len(pts), []
        for i in range(n):
            p0, p1, p2, p3 = pts[i - 1], pts[i], pts[(i + 1) % n], pts[(i + 2) % n]
            t1 = math.dist(p0, p1) ** 0.5
            t2 = t1 + math.dist(p1, p2) ** 0.5
            t3 = t2 + math.dist(p2, p3) ** 0.5
            for k in range(48):
                t = t1 + (t2 - t1) * k / 48
                at = lambda a, b, ta, tb: (((tb - t) * a[0] + (t - ta) * b[0]) / (tb - ta), ((tb - t) * a[1] + (t - ta) * b[1]) / (tb - ta))
                a1, a2, a3 = at(p0, p1, 0, t1), at(p1, p2, t1, t2), at(p2, p3, t2, t3)
                line.append(at(at(a1, a2, 0, t2), at(a2, a3, t1, t3), t1, t2))
        cum = [0.0]                                # resampled 1 m apart (as near as a whole number of points allows)
        for i in range(len(line)):
            cum.append(cum[-1] + math.dist(line[i], line[(i + 1) % len(line)]))
        self.L = cum[-1]                           # its length along the middle, m
        self.N = N = round(self.L)
        self.ds, self.cx, self.cy, j = self.L / N, [], [], 0
        for k in range(N):
            s = k * self.ds
            while cum[j + 1] < s:
                j += 1
            f = (s - cum[j]) / ((cum[j + 1] - cum[j]) or 1)
            a, b = line[j], line[(j + 1) % len(line)]
            self.cx.append(a[0] + (b[0] - a[0]) * f)
            self.cy.append(a[1] + (b[1] - a[1]) * f)
        self.tx, self.ty = [], []                  # the way round, at each point
        for k in range(N):
            dx, dy = self.cx[(k + 1) % N] - self.cx[k - 1], self.cy[(k + 1) % N] - self.cy[k - 1]
            l = math.hypot(dx, dy) or 1
            self.tx.append(dx / l)
            self.ty.append(dy / l)
        heading = [math.atan2(y, x) for x, y in zip(self.tx, self.ty)]
        self.curv = [wrap(heading[(k + 2) % N] - heading[k - 2]) / (4 * self.ds) for k in range(N)]   # how sharply it bends, rad a metre (> 0: right)
        self.min_radius = min(abs(1 / (c or 1e-9)) for c in self.curv)
        self.ok = check is None or check(self)
        if self.ok:
            self.distances()

    def distances(self):
        """The grid of distances to the middle line, filled only near the track (elsewhere: far outside)."""
        m, R, N = HW + 4, HW + 3, self.N
        self.ox, self.oy = math.floor(min(self.cx) - m), math.floor(min(self.cy) - m)
        self.gw, self.gh = math.ceil(max(self.cx) + m) - self.ox + 1, math.ceil(max(self.cy) + m) - self.oy + 1
        self.grid = [[1e9] * self.gw for _ in range(self.gh)]
        for k in range(N):
            ax, ay, bx, by = self.cx[k], self.cy[k], self.cx[(k + 1) % N], self.cy[(k + 1) % N]
            vx, vy = bx - ax, by - ay
            vv = vx * vx + vy * vy or 1
            for j in range(max(0, math.floor(min(ay, by) - R - self.oy)), min(self.gh - 1, math.ceil(max(ay, by) + R - self.oy)) + 1):
                row = self.grid[j]
                for i in range(max(0, math.floor(min(ax, bx) - R - self.ox)), min(self.gw - 1, math.ceil(max(ax, bx) + R - self.ox)) + 1):
                    px, py = self.ox + i + 0.5, self.oy + j + 0.5
                    t = min(max(((px - ax) * vx + (py - ay) * vy) / vv, 0.0), 1.0)
                    row[i] = min(row[i], math.hypot(px - ax - t * vx, py - ay - t * vy))

    def center_dist(self, x, y):                   # distance from (x, y) to the middle line (bilinear in the grid)
        fx, fy = x - self.ox - 0.5, y - self.oy - 0.5
        i, j = math.floor(fx), math.floor(fy)
        if i < 0 or j < 0 or i >= self.gw - 1 or j >= self.gh - 1:
            return 1e9
        g0, g1 = self.grid[j], self.grid[j + 1]
        v00, v10, v01, v11 = g0[i], g0[i + 1], g1[i], g1[i + 1]
        if max(v00, v10, v01, v11) > 1e8:
            return 1e9
        u, w = fx - i, fy - j
        return (v00 * (1 - u) + v10 * u) * (1 - w) + (v01 * (1 - u) + v11 * u) * w

    def ray(self, x, y, a):                        # how far a sensor at angle a sees before a wall
        ux, uy, t = math.cos(a), math.sin(a), 0.0
        for _ in range(64):
            d = HW - self.center_dist(x + ux * t, y + uy * t)
            if d < 0.06:
                return max(0.0, t)
            t += max(d * 0.92, 0.18)
            if t >= RANGE:
                return RANGE
        return min(t, RANGE)

    def locate(self, x, y, k):                     # the middle-line point nearest (x, y), searching around point k
        return min(((k + j) % self.N for j in range(-14, 15)), key=lambda q: (x - self.cx[q]) ** 2 + (y - self.cy[q]) ** 2)

    def arc(self, x, y, k):                        # how far along the middle line, m
        return k * self.ds + (x - self.cx[k]) * self.tx[k] + (y - self.cy[k]) * self.ty[k]

    def side(self, x, y, k):                       # how far to the right of the middle line, m
        return -(x - self.cx[k]) * self.ty[k] + (y - self.cy[k]) * self.tx[k]`),
      ...(random ? ['', '', ...rows(String.raw`SEEN = {}                                          # the random tracks made so far, by seed


def random_track(seed):
    """A new track: 9 to 12 points around an oval, each at its own distance from the middle, kept when its bends
    are not too tight for its width (the same seed, the same track)."""
    if seed not in SEEN:
        rng, T = random.Random(seed), None
        for _ in range(60):
            n = 9 + rng.randrange(4)
            pts = []
            for k in range(n):
                a, r = (k + rng.uniform(-0.3, 0.3)) / n * 2 * math.pi, rng.uniform(0.5, 1)
                pts.append((80 + math.cos(a) * 68 * r, 62 + math.sin(a) * 46 * r))
            T = Track(pts, check=lambda T: T.min_radius > HW + 2.5 and T.L > 230)
            if T.ok:
                break
        SEEN[seed] = T if T.ok else Track(TRACK)
    return SEEN[seed]`)] : []),
      '', '',
      ...h.head(rows(String.raw`class World:
    """A car drives ${laps} lap${laps === 1 ? '' : 's'} of ${random ? 'a random track (a new one every episode)' : `the ${E.TRACKS[id].name.toLowerCase()} track`}: it steers, and brakes for the tight bends. ${pays}
    It crashes when it touches a wall; ${laps} lap${laps === 1 ? '' : 's'}, or ${steps || `as many steps as ${laps === 1 ? 'a lap' : laps + ' laps'} at 10 m/s take`}${steps ? ' steps' : ''}, also end an episode."""
    actions = ["steer left", "straight on", "steer right", "brake"]`)), '',
      ...rows(String.raw`    def __init__(self):
        ${random ? 'self.tests = [random_track(s) for s in (7001, 7002, 7003, 7004)]   # 4 tracks it never trains on' : 'self.fixed = Track(TRACK)'}

    def reset(self, track=None):                   # at the start line, pointing along the track
        self.track = T = track or ${random ? 'random_track(1 + random.randrange(64))' : 'self.fixed'}
        self.max_steps = ${steps || 'round(LAPS * T.L)'}${steps ? '' : '              # its laps at an average of 10 m/s'}
        self.k, self.x, self.y, self.v = 0, T.cx[0], T.cy[0], 8.0
        self.th = math.atan2(T.ty[0], T.tx[0])
        self.s, self.prog, self.lap, self.t, self.end = T.arc(self.x, self.y, 0), 0.0, 0, 0, None
        self.sum_v = self.sum_side = 0.0
        self.see()
        return self.obs()

    def see(self):                                 # what its 7 sensors read
        self.rays = [self.track.ray(self.x, self.y, self.th + a) for a in RAYS]
${heading ? `
    def heading(self):                             # its heading to the way the track runs, rad (more than 0: to the right)
        T = self.track
        return wrap(self.th - math.atan2(T.ty[self.k], T.tx[self.k]))
` : ''}${bends ? `
    def bend(self, d):                             # how sharply the track bends d m ahead, rad a metre
        T = self.track
        return T.curv[(self.k + math.floor(d / T.ds + 0.5)) % T.N]
` : ''}`),
      ...h.sense(obs), '',
      ...rows(String.raw`    def step(self, a):
        T, gas, steer = self.track, a != 3, (-1 if a == 0 else 1 if a == 2 else 0)
        if gas:                                    # the throttle: toward top speed
            self.v += ACC * DT * (1 - self.v / VMAX)
        else:
            self.v = max(VMIN, self.v - BRAKE * DT)
        self.th += steer * min(STEER, GRIP / max(self.v, 0.5)) * DT   # never sharper than its grip allows
        self.x += self.v * math.cos(self.th) * DT
        self.y += self.v * math.sin(self.th) * DT
        self.t += 1
        self.k = T.locate(self.x, self.y, self.k)  # progress: how far along the middle line it got
        s = T.arc(self.x, self.y, self.k)
        ds = s - self.s
        if ds < -T.L / 2:                          # across the start line
            ds += T.L
        elif ds > T.L / 2:
            ds -= T.L
        self.s, self.prog = s, self.prog + ds
        laps = max(0, math.floor(self.prog / T.L))
        lap = laps > self.lap
        self.lap = max(self.lap, laps)
        side = T.side(self.x, self.y, self.k)
        crashed = T.center_dist(self.x, self.y) > HW - CAR_R
        self.sum_v += self.v
        self.sum_side += min(max(side / HW, -1.0), 1.0)
        if not crashed:
            self.see()
        done = crashed or self.prog >= LAPS * T.L or self.t >= self.max_steps
        if done:
            self.end = "crashed" if crashed else "finished" if self.prog >= LAPS * T.L else "time"`),
      ...reward, '        return self.obs(), r, done', '',
      ...rows(String.raw`    def behavior(self):                            # its style: average speed (slow to fast), the side it keeps (left to right)
        n = max(1, self.t)
        return min(max(self.sum_v / n / VMAX, 0.0), 1.0), min(max(0.5 + 1.2 * self.sum_side / n, 0.0), 1.0)

    def score(self, policy):
        """The share of ${laps === 1 ? 'a lap' : laps + ' laps'} its best moves drive${random ? ', on 4 tracks it never trained on' : ''}."""
        total = 0.0
        for T in ${random ? 'self.tests' : '[self.fixed]'}:
            o, done = self.reset(T), False
            while not done:
                o, _, done = self.step(policy(o))
            total += min(max(self.prog / (LAPS * T.L), 0.0), 1.0)
        return total / ${random ? 4 : 1}`)];
  };

  // ------------------------------------------------------------------ the drone arena, in the first person
  PW.worlds.arena = function (d, h) {
    const ws = d.ws, rw = h.rw, f = h.f, num = h.num, say = h.say, E = NA.rlEnvs, px = ws.view === 'pixels';
    const map = E.ARENAS[ws.map] ? ws.map : 'room', k = Math.min(3, Math.max(1, Math.round(ws.drones) || 1)), fast = ws.speed === 'fast', most = ws.steps > 0 ? Math.min(ws.steps, 3000) : 500;
    const W = E.make('arena', ws), lives = W.lives, goal = W.goal, recharge = W.recharge;
    // the drones in view and in its sights, worked out only when they are paid for
    const sights = f || rw.rAim || rw.rSee ? ['        aimed = seen = 0                           # a drone in its sights; how many it can see', '        for d in self.drones:', '            if d.alive:', '                rel, dist, rad, visible = self.sight(d)', '                if visible:', '                    seen += 1', '                    aimed = max(aimed, int(abs(rel) < rad + 0.02))'] : [];
    const reward = f ? [...sights, '        near = min([math.hypot(d.x - self.x, d.y - self.y) for d in self.drones if d.alive] + [12.0])   # the nearest drone, m',
      '        hit, hurt, shot, lives, t, moved, turned = float(hit), float(hurt), float(shot), float(self.health), self.t, float(walk), float(abs(turn))',
      '        missed, kills, bump, cleared, ready = float(shot and not hit), self.kills, float(bump), float(self.end == "cleared"), float(self.cool == 0)', ...h.ownLine]
      : [...sights, `        r = ${terms([rw.rHit ? `(${num(rw.rHit)} if hit else 0.0)` : null, rw.rHurt ? `${num(rw.rHurt)} * hurt` : null, rw.rShot ? `(${num(rw.rShot)} if shot else 0.0)` : null, rw.rStep ? num(rw.rStep) : null,
        rw.rMiss ? `(${num(rw.rMiss)} if shot and not hit else 0.0)` : null, rw.rAim ? `(${num(rw.rAim)} if aimed else 0.0)` : null, rw.rSee ? `${num(rw.rSee)} * seen` : null,
        rw.rWalk ? `(${num(rw.rWalk)} if walk else 0.0)` : null, rw.rTurn ? `(${num(rw.rTurn)} if turn else 0.0)` : null, rw.rBump ? `(${num(rw.rBump)} if bump else 0.0)` : null,
        rw.rClear ? `(${num(rw.rClear)} if self.end == "cleared" else 0.0)` : null])}`];
    const pays = h.pays(`${say(rw.rHit)} for each drone destroyed, ${say(rw.rHurt)} for each that reaches it${rw.rShot ? `, ${say(rw.rShot)} a shot` : ''}${rw.rStep ? `, ${say(rw.rStep)} a step` : ''}${rw.rMiss ? `, ${say(rw.rMiss)} for a shot that hits nothing` : ''}${rw.rClear ? `, ${say(rw.rClear)} for destroying the last drone` : ''}${rw.rAim ? `, ${say(rw.rAim)} every step a drone is in its sights` : ''}${rw.rSee ? `, ${say(rw.rSee)} every step for each drone in view` : ''}${rw.rWalk ? `, ${say(rw.rWalk)} every step it walks` : ''}${rw.rTurn ? `, ${say(rw.rTurn)} every step it turns` : ''}${rw.rBump ? `, ${say(rw.rBump)} for walking into a wall` : ''}.`);
    const obs = px ? rows(String.raw`    def obs(self):
        """What it sees, (2, 10, 20): walls (brighter nearer) and drones, each column showing the nearest thing in it."""
        p, z = torch.zeros(2, PROWS, PCOLS), [0.0] * PCOLS
        for j in range(PCOLS):
            a = ((j + 0.5) / PCOLS - 0.5) * FOV
            z[j] = d = cast(self.x, self.y, self.th + a)
            half = min(PROWS / 2, PROWS * 0.55 / max(0.1, d * math.cos(a)))
            for i in range(PROWS):
                if abs(i + 0.5 - PROWS / 2) < half:
                    p[0, i, j] = 1 - 0.8 * min(d, VIEW) / VIEW
        for dr in self.drones:
            if not dr.alive:
                continue
            rel, dist, rad, visible = self.sight(dr)
            if not visible:
                continue
            xc, rr, yc = (rel / FOV + 0.5) * PCOLS, max(0.6, rad / (FOV / PCOLS)), PROWS * 0.45
            for j in range(max(0, math.floor(xc - rr)), min(PCOLS - 1, math.floor(xc + rr)) + 1):
                if z[j] < dist - DRONE_R:              # behind a wall in this column
                    continue
                for i in range(max(0, math.floor(yc - rr)), min(PROWS - 1, math.floor(yc + rr)) + 1):
                    if (j + 0.5 - xc) ** 2 + (i + 0.5 - yc) ** 2 <= rr * rr:
                        p[1, i, j] = 1.0
        return p`)
      : rows(String.raw`    def obs(self):
        """What each of 11 columns across its view holds (how near the wall is, how near a drone is: 0 when none), a
        radar (which way the nearest drone is, behind walls too), its lives and whether the gun is ready: 26 numbers."""
        v, half, near = [0.0] * 26, FOV / (COLS - 1) / 2, None
        for c in range(COLS):
            v[c] = 1 - min(cast(self.x, self.y, self.th + (c / (COLS - 1) - 0.5) * FOV), VIEW) / VIEW
        for d in self.drones:
            if not d.alive:
                continue
            rel, dist, rad, visible = self.sight(d)
            if near is None or dist < near[1]:
                near = (rel, dist)
            if visible and dist <= VIEW:
                for c in range(COLS):
                    if abs((c / (COLS - 1) - 0.5) * FOV - rel) < half + rad:
                        v[COLS + c] = max(v[COLS + c], 1 - dist / VIEW)
        if near:
            v[22], v[23] = math.sin(near[0]), math.cos(near[0])
        v[24], v[25] = self.health / LIVES, float(self.cool == 0)
        return torch.tensor(v)`);
    return [
      `ARENA = [${E.ARENAS[map].rows.map((r) => JSON.stringify(r)).join(', ')}]   # ${E.ARENAS[map].name}: # a wall, S where it starts`,
      ...rows(String.raw`MH, MW = len(ARENA), len(ARENA[0])
FOV, COLS, VIEW = math.radians(66), 11, 8.0         # its field of view, the columns it sees in, how far it sees (m)${px ? '\nPROWS, PCOLS = 10, 20                              # the picture it sees' : ''}
DRONE_R, ME_R, TURN, WALK = 0.3, 0.25, math.radians(12), 0.22   # sizes (m), a turn, a step's walk (m)
DRONES, SPEED, MAX_STEPS = ${k}, ${fast ? 0.085 : 0.045}, ${most}              # drones at once, how far they fly a step (m), the longest episode
LIVES, GOAL, RECHARGE = ${lives}, ${goal}, ${recharge}${' '.repeat(Math.max(1, 25 - String(lives).length - String(goal).length - String(recharge).length))}# its lives, the drones to destroy, the steps the gun needs after a shot


def wrap(a):                                       # an angle, from -pi to pi
    return (a + math.pi) % (2 * math.pi) - math.pi


def is_wall(i, j):                                 # a wall, or outside the map
    return not (0 <= i < MW and 0 <= j < MH) or ARENA[j][i] == "#"


def cast(x, y, a, most=24.0):
    """How far a ray from (x, y) at angle a runs before a wall, stepping from one grid line to the next (DDA)."""
    dx, dy = math.cos(a), math.sin(a)
    mx, my = math.floor(x), math.floor(y)
    ddx, ddy = abs(1 / (dx or 1e-12)), abs(1 / (dy or 1e-12))
    sx, sy = (-1 if dx < 0 else 1), (-1 if dy < 0 else 1)
    tx, ty = (x - mx if dx < 0 else mx + 1 - x) * ddx, (y - my if dy < 0 else my + 1 - y) * ddy
    for _ in range(64):
        if tx < ty:
            mx += sx
            if is_wall(mx, my):
                return min(tx, most)
            tx += ddx
        else:
            my += sy
            if is_wall(mx, my):
                return min(ty, most)
            ty += ddy
        if min(tx, ty) > most:
            break
    return most


def clear_at(x, y, r):                             # a circle of radius r at (x, y) touches no wall
    for j in range(math.floor(y - r), math.floor(y + r) + 1):
        for i in range(math.floor(x - r), math.floor(x + r) + 1):
            if is_wall(i, j):
                cx, cy = min(max(x, i), i + 1), min(max(y, j), j + 1)
                if (x - cx) ** 2 + (y - cy) ** 2 < r * r:
                    return False
    return True


def slide(o, dx, dy, r):                           # a move that slides along walls: across, then along
    if clear_at(o.x + dx, o.y, r):
        o.x += dx
    if clear_at(o.x, o.y + dy, r):
        o.y += dy


class Drone:
    def __init__(self, wait):
        self.x, self.y, self.alive, self.wait, self.phase = 0.0, 0.0, False, wait, random.uniform(0, 2 * math.pi)`), '', '',
      ...h.head(rows(String.raw`class World:
    """A first-person arena: it turns 12 degrees, walks 0.22 m or fires (${recharge ? `the gun then recharges for ${recharge} step${recharge === 1 ? '' : 's'}` : 'the gun is ready again at once'}). Glowing
    drones drift toward it, weaving; a shot that crosses one destroys it, one that reaches it costs a life (${lives} ${lives === 1 ? 'life' : 'lives'}).
    ${pays} ${goal} drone${goal === 1 ? '' : 's'} destroyed, no lives left or ${most} steps end an episode."""
    actions = ["turn left", "turn right", "walk forward", "fire"]`)), '',
      ...rows(String.raw`    def reset(self):
        s = "".join(ARENA).index("S")
        self.x, self.y, self.th = s % MW + 0.5, s // MW + 0.5, 0.0
        self.t, self.health, self.kills, self.cool, self.end, self.moved, self.shots = 0, LIVES, 0, 0, None, 0, 0
        self.drones = [Drone(6 * i) for i in range(DRONES)]   # the first now, the next ones a little later
        for d in self.drones:
            if not d.wait:
                self.spawn(d)
        return self.obs()

    def spawn(self, d):                            # a drone on a free square at least 4 m away
        for _ in range(200):
            x, y = random.randrange(MW) + 0.5, random.randrange(MH) + 0.5
            if not is_wall(math.floor(x), math.floor(y)) and math.hypot(x - self.x, y - self.y) >= 4:
                d.x, d.y, d.alive, d.wait = x, y, True, 0
                return

    def sight(self, d):
        """Where a drone is in its view: its angle from straight ahead, distance, angular radius, and whether it can be
        seen (in its field of view, and no wall in front of it)."""
        dx, dy = d.x - self.x, d.y - self.y
        dist, rel = math.hypot(dx, dy), wrap(math.atan2(dy, dx) - self.th)
        rad = math.atan(DRONE_R / max(dist, 0.05))
        hidden = cast(self.x, self.y, self.th + rel, dist + 1) < dist - DRONE_R
        return rel, dist, rad, abs(rel) - rad < FOV / 2 and not hidden
`),
      ...h.sense(obs), '',
      ...rows(String.raw`    def step(self, a):
        turn, walk = (-1 if a == 0 else 1 if a == 1 else 0), a == 2
        self.th = wrap(self.th + turn * TURN)
        bump = False                               # walking into a wall: a step that went less than half as far as it should
        if walk:
            x0, y0 = self.x, self.y
            slide(self, math.cos(self.th) * WALK, math.sin(self.th) * WALK, ME_R)
            went = math.hypot(self.x - x0, self.y - y0)
            self.moved += went > 0.01
            bump = went < WALK / 2
        shot = hit = False
        if self.cool:                              # the gun recharging
            self.cool -= 1
        elif a == 3:                               # a shot: the nearest drone in line before the wall
            shot, self.cool = True, RECHARGE
            self.shots += 1
            wall, ux, uy, best = cast(self.x, self.y, self.th), math.cos(self.th), math.sin(self.th), None
            for d in self.drones:
                along = (d.x - self.x) * ux + (d.y - self.y) * uy
                across = abs(-(d.x - self.x) * uy + (d.y - self.y) * ux)
                if d.alive and 0 < along < wall and across < DRONE_R + 0.06 and (best is None or along < best[0]):
                    best = (along, d)
            if best:
                best[1].alive, best[1].wait, hit = False, 8, True
                self.kills += 1
        hurt = 0
        for d in self.drones:                      # the drones drift toward it, weaving; one that reaches it costs a life
            if not d.alive:
                if d.wait > 0:
                    d.wait -= 1
                    if d.wait == 0:
                        self.spawn(d)
                continue
            dx, dy = self.x - d.x, self.y - d.y
            l, w = math.hypot(dx, dy) or 1, 0.55 * math.sin(d.phase + self.t * 0.12)
            c, s = math.cos(w), math.sin(w)
            slide(d, (dx * c - dy * s) / l * SPEED, (dx * s + dy * c) / l * SPEED, DRONE_R)
            if math.hypot(self.x - d.x, self.y - d.y) < DRONE_R + ME_R + 0.05:
                hurt += 1
                d.alive, d.wait = False, 14
        self.health -= hurt
        self.t += 1
        done = self.health <= 0 or self.kills >= GOAL or self.t >= MAX_STEPS
        if done:
            self.end = "cleared" if self.kills >= GOAL else "dead" if self.health <= 0 else "time"`),
      ...reward, '        return self.obs(), r, done', '',
      ...rows(String.raw`    def behavior(self):                            # its style: how much it walks, how often it fires
        n = max(1, self.t)
        return min(self.moved / n, 1.0), min(4 * self.shots / n, 1.0)

    def score(self, policy):
        """The share of ${goal} drone${goal === 1 ? '' : 's'} its best moves destroy, over 3 games (the same 3 every time)."""
        total = 0.0`),
      ...h.SEEDED(4242, ['for _ in range(3):', ...h.PLAY('    '), '    total += self.kills / GOAL']).map((l) => '        ' + l), '        return total / 3'];
  };

  // ------------------------------------------------------------------ the rocket
  PW.worlds.lander = function (d, h) {
    const ws = d.ws, rw = h.rw, f = h.f, num = h.num, say = h.say;
    const W = NA.rlEnvs.make('lander', ws), g = W.g, fuel = W.fuelMax, wind = W.wind, randomPad = W.randomPad, most = W.maxSteps;
    const engine = ws.engine === 'strong' ? 3.5 : ws.engine === 'weak' ? 1.8 : 2.4, wide = String(+(2 * W.padHalf).toFixed(2));
    const reward = f ? ['        dist, speed, tilt = math.hypot(self.x - self.pad, self.y), math.hypot(self.vx, self.vy), self.th / DEG',
      '        landed, crashed, engine, turn, h, vx, vy, fuel, t = float(landed), float(crashed), float(main), float(abs(turn)), self.y, self.vx, self.vy, self.fuel / FUEL, self.t',
      '        x, spin, offpad, away, timeup = self.x - self.pad, self.w / DEG, float(self.end == "offpad"), float(self.end == "away"), float(self.end == "time")',
      '        ground, wind = self.y - self.ground(self.x), self.wind()', ...h.ownLine]
      : [`        r = ${terms([rw.rLand ? `(${num(rw.rLand)} if landed else 0.0)` : null, rw.rCrash ? `(${num(rw.rCrash)} if crashed else 0.0)` : null, rw.rFuel ? `(${num(rw.rFuel)} if main else 0.0)` : null, rw.rShape ? `${num(rw.rShape)} * closer` : null,
        rw.rTurn ? `(${num(rw.rTurn)} if turn else 0.0)` : null, rw.rTilt ? `${num(rw.rTilt)} * abs(self.th / DEG)` : null, rw.rStep ? num(rw.rStep) : null,
        rw.rOffPad ? `(${num(rw.rOffPad)} if self.end == "offpad" else 0.0)` : null, rw.rSpare ? `(${num(rw.rSpare)} * self.fuel / FUEL if landed else 0.0)` : null])}`];
    const pays = h.pays(`${say(rw.rLand)} for landing softly on the pad, ${say(rw.rCrash)} for anything else${rw.rShape ? `; and ${rw.rShape === 1 ? '' : `${say(rw.rShape)} × `}how much closer, slower and straighter it got each step (shaping)` : ''}${rw.rFuel ? `; ${say(rw.rFuel)} each step the main engine fires` : ''}${rw.rTurn ? `; ${say(rw.rTurn)} each step a side thruster fires` : ''}${rw.rTilt ? `; ${say(rw.rTilt)} × its tilt (degrees) every step` : ''}${rw.rStep ? `; ${say(rw.rStep)} every step` : ''}${rw.rOffPad ? `; ${say(rw.rOffPad)} for touching down softly off the pad` : ''}${rw.rSpare ? `; ${say(rw.rSpare)} × the share of its fuel left on landing` : ''}.`);
    return [
      ...rows(String.raw`G = ${num(g)}                                     # gravity, m/s²
THRUST = ${engine} * G + 1${' '.repeat(Math.max(1, 32 - String(engine).length))}# the main engine's push along the rocket (${ws.engine === 'strong' || ws.engine === 'weak' ? ws.engine : 'normal'}), m/s²
FUEL, MAX_STEPS = ${fuel}, ${most}${' '.repeat(Math.max(1, 30 - String(fuel).length - String(most).length))}# steps of burning in its tank; the longest episode
LEG, FOOT, PAD = 1.8, 1.4, ${num(W.padHalf)}${' '.repeat(Math.max(1, 23 - num(W.padHalf).length))}# its legs reach 1.8 m down and 1.4 m out; the pad is ${wide} m wide
DEG = math.pi / 180`), '', '',
      ...h.head(rows(String.raw`class World:
    """A rocket comes down from 30 m, up to 12 m to either side of the pad, drifting and tilted a little. Its main
    engine pushes along its axis, its side thrusters turn it, while the fuel lasts. When a leg touches the ground on
    the pad, slower than 3 m/s down and 2 m/s across and within 25 degrees of upright, it has landed. ${pays}
    Still up there after ${most} steps of 0.08 s counts as a crash.${wind ? ' Wind pushes it sideways in gusts.' : ''}"""
    actions = ["nothing", "main engine", "turn left", "turn right"]`)), '',
      ...rows(String.raw`    def ground(self, x):                           # gentle hills, flat for 3 m beyond the pad (heights above it)
        d = abs(x - self.pad)
        if d < PAD + 3:
            return 0.0
        h = 3 + 2.2 * math.sin(0.17 * x + 1.3) + 1.6 * math.sin(0.41 * x + 0.4) + 0.9 * math.sin(0.83 * x + 2.1)
        return max(0.0, h) * min(1.0, (d - PAD - 3) / 4)

    def reset(self):
        self.pad = ${randomPad ? 'random.uniform(-20, 20)            # the pad somewhere new every time' : '0.0'}
        self.x, self.y = self.pad + random.uniform(-12, 12), 30.0
        self.vx, self.vy = random.uniform(-2, 2), random.uniform(-2, 0)
        self.th, self.w = random.uniform(-0.3, 0.3), random.uniform(-0.2, 0.2)
        self.gusts = random.uniform(0, 2 * math.pi)
        self.t, self.fuel, self.burn, self.land_x, self.end = 0, FUEL, 0, None, None
        self.phi = self.potential()
        return self.obs()

    def potential(self):                           # higher the nearer, slower and straighter: the shaping is its change
        return -(math.hypot(self.x - self.pad, self.y) / 10 + math.hypot(self.vx, self.vy) / 3 + abs(self.th))

    def wind(self):                                # sideways gusts, m/s²
        ${wind ? 'return 0.5 * math.sqrt(G) * (math.sin(0.35 * self.t * 0.08 * 3 + self.gusts) + 0.6 * math.sin(1.3 * self.t * 0.08 + 2 * self.gusts))' : 'return 0.0'}
`),
      ...h.sense(rows(String.raw`    def obs(self):                                 # where it is from the pad, its speed, tilt, spin and fuel, scaled
        return torch.tensor([(self.x - self.pad) / 20, self.y / 30, self.vx / 5, self.vy / 5, self.th / 0.6, self.w / 1.5, self.fuel / FUEL])`)), '',
      ...rows(String.raw`    def feet(self):
        c, s = math.cos(self.th), math.sin(self.th)
        return [(self.x + lx * c - LEG * s, self.y - lx * s - LEG * c) for lx in (-FOOT, FOOT)]

    def step(self, a):
        dt, main, turn = 0.08, a == 1 and self.fuel > 0, (-1 if a == 2 else 1 if a == 3 else 0)
        ax, ay = -0.04 * self.vx + self.wind(), -G - 0.04 * self.vy
        if main:
            ax, ay = ax + THRUST * math.sin(self.th), ay + THRUST * math.cos(self.th)
            self.fuel, self.burn = self.fuel - 1, self.burn + 1
        self.w = (self.w + turn * 2.6 * dt) * 0.985
        self.vx, self.vy = self.vx + ax * dt, self.vy + ay * dt
        self.x, self.y, self.th = self.x + self.vx * dt, self.y + self.vy * dt, self.th + self.w * dt
        self.t += 1
        feet, landed, crashed = self.feet(), False, False
        if any(fy <= self.ground(fx) for fx, fy in feet) or self.y - 0.9 <= self.ground(self.x):   # touching down
            soft = self.vy > -3 and abs(self.vx) < 2 and abs(self.th) < math.radians(25)
            landed = soft and all(abs(fx - self.pad) <= PAD for fx, _ in feet)
            crashed, self.land_x = not landed, self.x
            self.end = "landed" if landed else "offpad" if soft else "crashed"   # (offpad: soft, but off the pad)
        elif abs(self.x - self.pad) > 50 or self.y > 60:
            crashed, self.end = True, "away"       # flown off
        elif self.t >= MAX_STEPS:                  # out of time counts as a failure: no hovering for ever
            crashed, self.end = True, "time"
        phi = self.potential()
        closer, self.phi = phi - self.phi, phi`),
      ...reward, '        return self.obs(), r, self.end is not None', '',
      ...rows(String.raw`    def behavior(self):                            # its style: where it came down (left of the pad to right), how much it burned
        x = self.x if self.land_x is None else self.land_x
        return min(max(0.5 + (x - self.pad) / 40, 0.0), 1.0), min(self.burn / max(1, self.t), 1.0)

    def score(self, policy):
        """The share of 10 starts its best moves land on the pad (the same 10 every time)."""
        landed = 0`),
      ...h.SEEDED(4242, ['for _ in range(10):', ...h.PLAY('    '), '    landed += self.end == "landed"']).map((l) => '        ' + l), '        return landed / 10'];
  };

  // ------------------------------------------------------------------ styles of play for the small worlds (MAP-Elites)
  PW.style = function (w) {
    if (w === 'maze') return rows(String.raw`class World(World):                               # the same world, also saying how it played (for MAP-Elites)
    def behavior(self):                            # its style: where its moves stopped, across and up from the bottom row
        m = self.maze
        return (self.s % m.W + 0.5) / m.W, 1 - (self.s // m.W + 0.5) / m.H`);
    if (w === 'pole') return rows(String.raw`class World(World):                               # the same world, also saying how it played (for MAP-Elites)
    def reset(self):
        self.sum_x = self.sum_angle = 0.0
        return super().reset()

    def step(self, a):
        out = super().step(a)
        self.sum_x, self.sum_angle = self.sum_x + self.x, self.sum_angle + abs(self.th)
        return out

    def behavior(self):                            # its style: where the cart keeps (left to right), how much the pole sways
        n = max(1, self.t)
        return min(max(0.5 + self.sum_x / n / 4.8, 0.0), 1.0), min((self.sum_angle / n / 0.2095) ** 0.5, 1.0)`);
    if (w === 'catch') return rows(String.raw`class World(World):                               # the same world, also saying how it played (for MAP-Elites)
    def reset(self):
        self.moves = 0
        return super().reset()

    def step(self, a):
        out = super().step(a)
        self.moves += self.px != self.prev[2]
        return out

    def behavior(self):                            # its style: how much the paddle moves, where it ends up
        return min(self.moves / max(1, self.t), 1.0), self.px / 4`);
    return [];
  };

  // ------------------------------------------------------------------ evolution
  /* x: { flt, num, big, ACT_FN, setup, evolvable (the model's blocks can change shape), every }. A member is its
     weights as one vector (the model's own parameters), except with grow and prune on: then each member is a network
     of its own, a list of weight matrices and biases and the activations of its hidden layers. */
  // style: what its style is for (true: MAP-Elites); mem: every move also goes into the DQN's memory (ERL)
  const PLAY = (args, load, move, style, keep, mem) => rows(String.raw`def play(${args}, seed${keep ? ', keep=False' : ''}):
    """Its best moves over a few episodes, from the same starts as everyone else in its generation: the average return
    (its fitness)${style ? `, its style in the first (for ${style === true ? 'MAP-Elites' : style})` : ''}${keep ? ' and some of what it saw there (for safe mutations)' : ''}.${mem ? ' Every\n    move also goes into the memory the DQN learns from.' : ''}"""${load ? `\n    ${load}` : ''}
    total, style, seen = 0.0, None, []
    state = random.getstate()
    random.seed(seed)
    for k in range(episodes):
        o, done, t = env.reset(), False, 0
        while not done:${keep ? '\n            if keep and k == 0 and len(seen) < 32 and t % 5 == 0:\n                seen.append(o)' : ''}${mem ? `
            a = ${move}(o)
            o2, r, done = env.step(a)
            memory.append((o, a, r, None if done else o2))
            o, total, t = o2, total + r, t + 1` : `
            o, r, done = env.step(${move}(o))
            total, t = total + r, t + 1`}${style ? '\n        if k == 0:\n            style = env.behavior()' : ''}
    random.setstate(state)
    return total / episodes, style, seen`);
  const PICK = {
    tournament: ['def pick(order, fit):                              # a parent: the best of 3 picked at random (a tournament)', '    best = random.randrange(len(fit))', '    for _ in range(2):', '        c = random.randrange(len(fit))', '        if fit[c] > fit[best]:', '            best = c', '    return best'],
    truncation: ['def pick(order, fit):                              # a parent: one of the top fifth, at random', '    return order[random.randrange(max(2, round(len(fit) / 5)))]'],
    rank: ['def pick(order, fit):                              # a parent by rank: the best n times as likely as the worst', '    return random.choices(order, weights=range(len(fit), 0, -1))[0]'],
  };
  const episodesOf = (d) => (NA.brl.WORLD[d.world] || NA.brl.WORLD.pole).gaEpisodes + (d.ws.random || d.ws.slip || (d.world === 'track' && d.ws.track === 'random') ? 2 : 0);

  // a population's members mixed: each weight, or each neuron whole, from one parent or the other (or not at all)
  function mixPy(s, x) {
    const cross = ['uniform', 'neuron', 'none'].includes(s.cross) ? s.cross : 'uniform', neuron = cross === 'neuron' && x.evolvable;
    return cross === 'none' ? [] : neuron ? [...rows(String.raw`# which neuron each weight belongs to: a row of a layer's weights and its bias go together
unit_of, n_units = [], 0
for layer in (m for m in model.modules() if isinstance(m, nn.Linear)):
    U, I = layer.weight.shape
    unit_of += [n_units + torch.arange(U).repeat_interleave(I), n_units + torch.arange(U)]
    n_units += U
unit_of = torch.cat(unit_of)


def mix(a, b):                                     # each neuron whole, with all its weights, from one parent or the other
    return torch.where((torch.rand(n_units) < 0.5)[unit_of], a, b)`), '', '']
      : [...(cross === 'neuron' ? ['# (neuron by neuron needs a model of Dense blocks only: here, weight by weight)'] : []), 'def mix(a, b):                                     # each weight from one parent or the other', '    return torch.where(torch.rand_like(a) < 0.5, a, b)', '', ''];
  }
  // one weight in ten nudged: plain, or safe (each nudge divided by how much the moves depend on that weight)
  const mutatePy = (safe) => (safe ? rows(String.raw`def sensitivity(w, seen):
    """How much the moves depend on each weight: each output's gradient, averaged over what the parent saw; the root
    of the sum of their squares over the outputs (SM-G-SUM, Lehman and others, 2018)."""
    vector_to_parameters(w, model.parameters())
    X, s = torch.stack(seen), torch.zeros_like(w)
    for a in range(n_actions):
        model.zero_grad()
        model(X)[:, a].mean().backward()
        s += parameters_to_vector([p.grad if p.grad is not None else torch.zeros_like(p) for p in model.parameters()]) ** 2
    return s.sqrt()


def mutate(w, sens):
    """One weight in ten nudged, each nudge divided by how much the moves depend on that weight (safe), then scaled
    so the nudges are as big on average as plain ones (sigma at random)."""
    chosen = (torch.rand_like(w) < 0.1).float()
    d = chosen * torch.randn_like(w) / (sens + 0.05 * sens.mean() + 1e-6)
    return w + d * sigma * 0.7979 * chosen.sum() / d.abs().sum().clamp_min(1e-12)`) : ['def mutate(w):                                     # one weight in ten nudged by sigma at random', '    return w + (torch.rand_like(w) < 0.1) * sigma * torch.randn_like(w)']);
  // the next generation (inside the loop): the best two unchanged, then children of parents picked by fit
  function breedPy(s) {
    const cross = ['uniform', 'neuron', 'none'].includes(s.cross) ? s.cross : 'uniform', safe = s.mutate === 'safe';
    return ['    children = [population[order[0]].clone(), population[order[1]].clone()]   # the best two carry on unchanged', safe ? '    sens = {}' : null,
      '    while len(children) < pop_size:', '        i = pick(order, fit)', '        child = population[i]',
      ...(cross === 'none' ? [] : ['        j = pick(order, fit)', '        if j != i:', '            child = mix(population[i], population[j])']),
      ...(safe ? ['        if i not in sens:', '            sens[i] = sensitivity(population[i], results[i][2])', '        children.append(mutate(child, sens[i]))'] : ['        children.append(mutate(child))'])];
  }
  // the first generation of a population of weight vectors
  const FIRST = ['w0 = parameters_to_vector(model.parameters()).detach()', '# the first generation: the network you built and copies of it, each moved at random',
    'population = [w0.clone()] + [w0 + 0.5 * torch.randn_like(w0) for _ in range(pop_size - 1)]'];

  // what a method does, for the script's first lines (builder-py.js): the methods of builder-evo.js that it does not name itself
  PW.how = (d) => { const s = d.s; return { novelty: `a population of ${s.pop} chosen for new behaviour, not reward`, cem: `${Math.max(4, Math.round(s.pop) || 24)} networks a generation drawn from a bell curve over the weights`, erl: `a population of ${s.pop} whose moves teach a DQN` }[d.method] || ''; };
  // the methods whose worlds must say how they played (the small worlds' Python adds it: PW.style)
  PW.needsStyle = (method) => method === 'me' || method === 'novelty';
  PW.method = function (d, T, steps, x) {
    const s = d.s, flat = d.world === 'levers' && PW.needsStyle(d.method), m = flat ? 'ga' : d.method, every = x.every;
    if (m === 'es') return esPy(d, steps, x);
    if (m === 'me') return mePy(d, steps, x);
    if (m === 'cem') return cemPy(d, steps, x);
    if (m === 'novelty') return novPy(d, steps, x);
    if (m === 'erl') return erlPy(d, steps, x);
    const note = flat ? [d.method === 'me' ? '# MAP-Elites keeps the best network of every style of play, and a pull of a lever has no style to tell apart:' : '# Novelty search looks for behaviour unlike any before, and a pull of a lever has no style to tell apart:', '# this runs the genetic algorithm instead.'] : [];
    if (s.grow && x.evolvable) return [...note, ...growPy(d, steps, x)];
    return [...note, s.grow ? '# Grow and prune is on, but only a model of Dense blocks on a list of numbers can change shape: here only the weights evolve.' : null, ...gaPy(d, steps, x)].filter((l) => l !== null);
    function gaPy() {
      const sel = PICK[s.select] ? s.select : 'tournament', safe = s.mutate === 'safe';
      return [
        'from torch.nn.utils import parameters_to_vector, vector_to_parameters', `pop_size, sigma, episodes = ${s.pop}, ${x.flt(s.sigma)}, ${episodesOf(d)}`, ...FIRST, '', '',
        ...PLAY('w', 'vector_to_parameters(w, model.parameters())', 'best_move', false, safe), '', '', ...PICK[sel], '', '', ...mixPy(s, x), ...mutatePy(safe), '', '',
        `for gen in range(${x.big(steps)}):`, `    results = [play(w, 1000 + gen${safe ? ', keep=True' : ''}) for w in population]`, '    fit = [r[0] for r in results]', '    order = sorted(range(pop_size), key=lambda k: -fit[k])',
        ...breedPy(s),
        '    population = children', '    vector_to_parameters(population[0], model.parameters())             # on show: the best of the generation',
        `    if gen % ${every} == 0:`, '        print(f"generation {gen:4d}  best {max(fit):.2f}  average {sum(fit) / pop_size:.2f}  score {judge.score(best_move):.3f}")'].filter((l) => l !== null);
    }
  };

  /* The operations on a network whose shape can change: its outputs, its size, a neuron more or less, a layer more
     (size: how wide a new layer starts, 'one', 'some' or 'whole', as builder-evo.js), a layer less ('whole': at
     once; otherwise when its last neuron goes), a new activation, and the one change of shape a child is born with,
     by the odds and limits of o (EV.growOptions). The older call (acts, newLayer, ACT_FN, size) still works. */
  // a docstring's text in lines of at most 116 characters, the first after its opening quotes, the rest indented 4
  function wrapDoc(text) {
    const out = [];
    let line = '';
    for (const w of text.split(' ')) { const room = out.length ? 112 : 109; if (line && line.length + 1 + w.length > room) { out.push(line); line = w; } else line = line ? `${line} ${w}` : w; }
    out.push(line);
    return out.join('\n    ');
  }
  function netOps(acts, o, ACT_FN, size) {
    if (!o || typeof o !== 'object') o = NA.bevo.growOptions({ newLayer: o, layerSize: size || 'one' });
    const whole = o.size === 'whole', rnd = o.mode === 'random', W = o.weights, act = W.act > 0, used = [...new Set([...acts, 'relu', ...(act ? NA.bevo.ACTS : [])])];
    const where = o.where === 'first' ? '0' : o.where === 'last' ? 'n' : 'random.randrange(n + 1)', w = (v) => String(+v.toFixed(3));
    // the changes allowed, with their odds (a change it cannot make here is left out)
    const odds = [W['neuron+'] ? `"neuron+": ${w(W['neuron+'])} if n else 0` : null, W['neuron-'] ? `"neuron-": ${w(W['neuron-'])} if n else 0` : null, W['layer+'] ? `"layer+": ${w(W['layer+'])} if n < MOST_LAYERS else 0` : null,
      W['layer-'] ? `"layer-": ${w(W['layer-'])} if n else 0` : null, act ? `"act": ${w(W.act)} if n else 0` : null].filter(Boolean);
    const say = [W['neuron+'] ? 'a neuron more' : null, W['neuron-'] ? 'a neuron less' : null, W['layer+'] ? 'a layer more' : null, W['layer-'] ? 'a layer less' : null, act ? 'a new activation' : null].filter(Boolean);
    // change_shape's docstring, wrapped as the page's other docstrings are
    const shapeDoc = wrapDoc(`One change to its shape: ${say.length > 1 ? `${say.slice(0, -1).join(', ')} or ${say[say.length - 1]}` : say[0] || 'none'}, by these odds${W['layer+'] ? ` (a new layer of ${o.size === 'one' ? 'one neuron' : o.size === 'some' ? 'a few' : 'the width it reads'}, ${o.where === 'first' ? 'straight after the input' : o.where === 'last' ? 'just before the output' : 'anywhere'})` : ''}${whole || !W['neuron-'] ? '' : '; a layer goes when its last neuron does'}.`);
    return rows(String.raw`ACT = {${used.map((a) => `"${a}": ${ACT_FN[a] || 'F.relu'}`).join(', ')}}
MOST_LAYERS, MOST_UNITS = ${o.maxLayers}, ${o.maxUnits}                   # the largest it may grow: hidden layers, neurons in a layer


def forward(net, x):                               # its outputs: its layers in turn, the hidden ones with their activation
    w, acts = net["w"], net["acts"]
    for k in range(0, len(w), 2):
        x = x @ w[k].T + w[k + 1]
        if k // 2 < len(acts):
            x = ACT[acts[k // 2]](x)
    return x


def size(net):                                     # how many weights it has
    return sum(t.numel() for t in net["w"])


def shape(net):                                    # its hidden layers' widths
    return [W.shape[0] for W in net["w"][:-2:2]]


def add_neuron(net, h):
    """A neuron more in hidden layer h: its incoming weights at random, ${o.neuron === 'random' ? 'and its outgoing ones too, so it joins in at once,\n    for better or worse.' : 'its outgoing ones 0, so it computes exactly\n    what it did, and later mutations can put the new neuron to work.'}"""
    w = list(net["w"])
    W, b, W2 = w[2 * h], w[2 * h + 1], w[2 * h + 2]
    if W.shape[0] >= MOST_UNITS:
        return None
    w[2 * h] = torch.cat([W, torch.randn(1, W.shape[1]) * math.sqrt(2 / W.shape[1])])
    w[2 * h + 1] = torch.cat([b, torch.zeros(1)])
    w[2 * h + 2] = torch.cat([W2, ${o.neuron === 'random' ? 'torch.randn(W2.shape[0], 1) * math.sqrt(1 / (W.shape[0] + 1))' : 'torch.zeros(W2.shape[0], 1)'}], 1)
    return {"w": w, "acts": net["acts"]}


def remove_neuron(net, h):
    """${o.prune === 'random' ? 'A neuron of hidden layer h, picked at random, taken out.' : 'The neuron of hidden layer h that the next layer listens to least (the smallest outgoing weights), taken out.'}${whole ? '' : '\n    The last neuron of a layer takes the layer with it, folded into the next.'}"""
    w = list(net["w"])
    W, b, W2 = w[2 * h], w[2 * h + 1], w[2 * h + 2]
    if W.shape[0] <= 1:
        return ${whole ? 'None' : 'remove_layer(net, h)'}
    j = ${o.prune === 'random' ? 'random.randrange(W.shape[0])' : 'int(W2.abs().sum(0).argmin())'}
    keep = [i for i in range(W.shape[0]) if i != j]
    w[2 * h], w[2 * h + 1], w[2 * h + 2] = W[keep], b[keep], W2[:, keep]
    return {"w": w, "acts": net["acts"]}


${whole ? `def add_layer(net, p):
    """A new hidden layer at position p (0: straight after the input), as wide as what it reads: ${rnd ? 'its weights at random' : 'its weights the\n    identity, so it passes its input straight through (exactly, after a ReLU layer)'}."""
    acts = list(net["acts"])
    U = net["w"][2 * p].shape[1]                   # the width of what it reads
    if len(acts) >= MOST_LAYERS or U > MOST_UNITS:
        return None
    w = list(net["w"])
    w[2 * p:2 * p] = [${rnd ? 'torch.randn(U, U) * math.sqrt(2 / U)' : 'torch.eye(U)'}, torch.zeros(U)]
    acts.insert(p, acts[p - 1] if p > 0 else acts[0] if acts else "relu")
    return {"w": w, "acts": acts}` : `def add_layer(net, p):
    """A new hidden layer at position p (0: straight after the input) of ${o.size === 'one' ? 'one neuron' : 'a few neurons (1 to one less than the width it reads,\n    at random)'}: it passes on ${o.size === 'one' ? 'the signal' : 'the signals'} the next layer leans on most (the largest column${o.size === 'one' ? '' : 's'} of its weights),
    ${rnd ? 'its own weights at random' : `each neuron picking one with a weight of 1`}, and the next layer reads ${o.size === 'one' ? 'it' : 'them'} with ${o.size === 'one' ? 'that same column' : 'those same columns'}. The rest
    is cut off: a bottleneck, so unlike a new neuron it changes what the network does, until "+1 neuron" widens it."""
    acts = list(net["acts"])
    if len(acts) >= MOST_LAYERS:
        return None
    w = list(net["w"])
    W2 = w[2 * p]                                  # the next layer's weights: it will read the new layer instead
    I = W2.shape[1]                                # the width of what the new layer reads
    U = ${o.size === 'one' ? '1' : 'random.randint(1, max(1, min(I - 1, MOST_UNITS)))'}
    lean = W2.abs().sum(0).tolist()
    keep = sorted(range(I), key=lambda j: -lean[j])[:U]   # the inputs the next layer leans on most
${rnd ? '    new = torch.randn(U, I) * math.sqrt(2 / I)' : '    new = torch.zeros(U, I)\n    for k, j in enumerate(keep):                   # each neuron passes one of them on\n        new[k, j] = 1.0'}
    w[2 * p:2 * p + 1] = [new, torch.zeros(U), W2[:, keep]]
    acts.insert(p, acts[p - 1] if p > 0 else acts[0] if acts else "relu")
    return {"w": w, "acts": acts}`}


def remove_layer(net, h):
    """Hidden layer h taken out and folded into the next: W = W2 W1, b = W2 b1 + b2 (as if its activation let
    values through)."""
    if not net["acts"]:
        return None
    w = list(net["w"])
    W1, b1, W2, b2 = w[2 * h:2 * h + 4]
    w[2 * h:2 * h + 4] = [W2 @ W1, W2 @ b1 + b2]
    return {"w": w, "acts": net["acts"][:h] + net["acts"][h + 1:]}
${act ? String.raw`

NEW_ACTS = ${JSON.stringify(NA.bevo.ACTS)}


def change_activation(net, h):
    """Hidden layer h switched to another activation: its weights stay, what it computes changes."""
    acts = list(net["acts"])
    acts[h] = random.choice([a for a in NEW_ACTS if a != acts[h]])
    return {"w": net["w"], "acts": acts}` : ''}


def change_shape(net):
    """${shapeDoc}"""
    n = len(net["acts"])
    ops = {${odds.join(', ')}}
    if not any(ops.values()):
        return net
    for _ in range(8):
        op = random.choices(list(ops), weights=list(ops.values()))[0]
        if op == "neuron+":
            r = add_neuron(net, random.randrange(n))
        elif op == "neuron-":
            r = remove_neuron(net, random.randrange(n))
        elif op == "layer+":
            r = add_layer(net, ${where})
        elif op == "layer-":
            r = remove_layer(net, random.randrange(n))
        else:
            r = change_activation(net, random.randrange(n))
        if r:
            return r
    return net`);
  }
  const actsOf = (blocks) => blocks.filter((b) => b.type === 'dense').map((b) => b.act || 'relu');
  // the changes of shape a child can be born with, in words (o: EV.growOptions)
  function shapeSay(o) {
    const W = o.weights, n = W['neuron+'] && W['neuron-'] ? 'gain or lose a neuron' : W['neuron+'] ? 'gain a neuron' : W['neuron-'] ? 'lose a neuron' : null;
    const l = W['layer+'] ? `gain a new layer of ${o.size === 'one' ? 'one neuron' : o.size === 'some' ? 'a few neurons' : 'the width it reads'}` : null;
    const list = [n, l, W['layer-'] ? 'lose a whole layer' : null, W.act ? 'switch a layer to another activation' : null].filter(Boolean);
    return (list.length > 1 ? `${list.slice(0, -1).join(', ')} or ${list[list.length - 1]}` : list[0] || 'not change at all') + (W['neuron-'] && o.size !== 'whole' ? '; a layer goes when its last neuron does' : '');
  }
  const START = 'start = {"w": [t.detach().clone() for m in model.modules() if isinstance(m, nn.Linear) for t in (m.weight, m.bias)], "acts": ACTS}';
  const START_SMALL = ['first = nn.Linear(Xva.shape[1], n_actions)          # the smallest network: no hidden layer, the inputs straight to the outputs', 'start = {"w": [first.weight.detach().clone(), first.bias.detach().clone()], "acts": []}'];

  // a genetic algorithm whose networks grow and shrink
  function growPy(d, steps, x) {
    const s = d.s, sel = PICK[s.select] ? s.select : 'tournament', cross = ['uniform', 'neuron', 'none'].includes(s.cross) ? s.cross : 'uniform', safe = s.mutate === 'safe';
    const o = NA.bevo.growOptions(s), W = NA.brl.WORLD[d.world] || NA.brl.WORLD.pole, pen = NA.bevo.PENALTY[o.penalty] * (W.scale || 1), small = o.start === 'small', prot = o.protect, P = NA.bevo.PROTECT;
    const rate = o.rate >= 1 ? null : o.rate === 0.25 ? 'a quarter of the children' : o.rate === 0.5 ? 'half the children' : o.rate === 0.1 ? 'one child in ten' : `${Math.round(o.rate * 100)}% of the children`;
    // a child born with a new shape (protecting new shapes: its shape's age starts again)
    const reshape = (ind) => (prot ? [`${ind}new = change_shape(child)`, `${ind}if new is not child:`, `${ind}    child = dict(new, since=gen + 1)   # a new shape: protected for ${P.gens} generations`] : [`${ind}child = change_shape(child)`]);
    return [
      '# Each member is a network of its own: its Dense layers as a list [W1, b1, W2, b2, ..., W_out, b_out] and an', ...`# ${wrapDoc(`activation for each hidden layer. Its shape can change: a child may ${shapeSay(o)}.`)}`.split('\n').map((l, i) => (i ? `# ${l.trim()}` : l)),
      `ACTS = ${JSON.stringify(actsOf(x.setup.blocks))}`, ...netOps(actsOf(x.setup.blocks), o, x.ACT_FN), '', '',
      `pop_size, sigma, episodes = ${s.pop}, ${x.flt(s.sigma)}, ${episodesOf(d)}`,
      `penalty = ${x.num(pen)}                                  # taken off the fitness for every 100 weights: of two that play as well, the smaller wins`,
      rate ? `grow_rate = ${x.flt(o.rate)}                                # the share of children born with a change of shape` : null,
      ...(small ? START_SMALL : [START]), small ? '# the first generation: the smallest network and copies of it, each moved at random: every neuron and layer it ends with is one evolution added' : '# the first generation: the network you built and copies of it, each moved at random',
      'population = [start] + [{"w": [t + 0.5 * torch.randn_like(t) for t in start["w"]], "acts": start["acts"]} for _ in range(pop_size - 1)]', '', '',
      ...(prot ? ['def kind(net):                                     # a shape: its hidden layers\' widths and activations', '    return tuple(shape(net)), tuple(net["acts"])', '', ''] : []),
      'def moves_of(net):                                 # a member\'s best move: its largest output', '    return lambda o: int(forward(net, o[None])[0, :n_actions].argmax())', '', '',
      ...PLAY('net', 'move = moves_of(net)', 'move', false, safe), '', '', ...PICK[sel], '', '',
      ...(cross === 'none' ? [] : cross === 'neuron' ? ['def mix(a, b):                                     # each neuron whole (its weights and its bias) from one parent or the other', '    w = []', '    for W1, b1, W2, b2 in zip(a["w"][::2], a["w"][1::2], b["w"][::2], b["w"][1::2]):', '        take = torch.rand(W1.shape[0]) < 0.5', '        w += [torch.where(take[:, None], W1, W2), torch.where(take, b1, b2)]', '    return {"w": w, "acts": a["acts"]}', '', '']
        : ['def mix(a, b):                                     # each weight from one parent or the other', '    return {"w": [torch.where(torch.rand_like(p) < 0.5, p, q) for p, q in zip(a["w"], b["w"])], "acts": a["acts"]}', '', '']),
      ...(safe ? rows(String.raw`def sensitivity(net, seen):
    """How much the moves depend on each weight: each output's gradient, averaged over what the parent saw; the root
    of the sum of their squares over the outputs (SM-G-SUM, Lehman and others, 2018)."""
    w = [t.clone().requires_grad_() for t in net["w"]]
    X, s = torch.stack(seen), [torch.zeros_like(t) for t in net["w"]]
    for a in range(n_actions):
        grads = torch.autograd.grad(forward({"w": w, "acts": net["acts"]}, X)[:, a].mean(), w)
        s = [si + g ** 2 for si, g in zip(s, grads)]
    return [si.sqrt() for si in s]


def mutate(net, sens):
    """One weight in ten nudged, each nudge divided by how much the moves depend on that weight (safe), then scaled
    so the nudges are as big on average as plain ones (sigma at random)."""
    chosen = [(torch.rand_like(t) < 0.1).float() for t in net["w"]]
    floor = 0.05 * sum(si.sum() for si in sens) / size(net) + 1e-6
    d = [c * torch.randn_like(t) / (si + floor) for c, t, si in zip(chosen, net["w"], sens)]
    scale = sigma * 0.7979 * sum(c.sum() for c in chosen) / sum(q.abs().sum() for q in d).clamp_min(1e-12)
    return {"w": [t + q * scale for t, q in zip(net["w"], d)], "acts": net["acts"]}`) : ['def mutate(net):                                   # one weight in ten nudged by sigma at random', '    return {"w": [t + (torch.rand_like(t) < 0.1) * sigma * torch.randn_like(t) for t in net["w"]], "acts": net["acts"]}']), '', '',
      `for gen in range(${x.big(steps)}):`, `    results = [play(net, 1000 + gen${safe ? ', keep=True' : ''}) for net in population]`, '    fit = [r[0] for r in results]',
      '    adj = [f - penalty * size(net) / 100 for f, net in zip(fit, population)]   # the fitness it is chosen by', '    order = sorted(range(pop_size), key=lambda k: -adj[k])',
      '    children = [population[order[0]], population[order[1]]]   # the best two carry on unchanged',
      ...(prot ? [`    # protect new shapes: the best of each shape that appeared in the last ${P.gens} generations carries on unchanged too`, '    kept = {kind(c) for c in children}', '    for k in order:', `        if len(children) >= 2 + min(${P.most}, pop_size - 3):`, '            break',
        '        net = population[k]', `        if gen - net.get("since", -99) < ${P.gens} and kind(net) not in kept:`, '            kept.add(kind(net))', '            children.append(net)'] : []),
      safe ? '    sens = {}' : null,
      '    while len(children) < pop_size:', '        i = pick(order, adj)', '        child = population[i]',
      ...(cross === 'none' ? [] : ['        j = pick(order, adj)', '        if j != i and shape(population[j]) == shape(child) and population[j]["acts"] == child["acts"]:   # only parents of one shape mix', '            child = mix(child, population[j])']),
      ...(safe ? ['        if i not in sens:', '            sens[i] = sensitivity(population[i], results[i][2])', '        child = mutate(child, sens[i]) if shape(child) == shape(population[i]) else child'] : ['        child = mutate(child)']),
      prot ? '        child = dict(child, since=population[i].get("since", -99))   # its shape is as old as its parent\'s' : null,
      ...(rate ? [`        if random.random() < grow_rate:            # ${rate}: one change to their shape`, ...reshape('            ')] : ['        # every child: one change to its shape', ...reshape('        ')]), '        children.append(child)',
      '    population = children', '    best_move = moves_of(population[0])           # on show: the best of the generation',
      `    if gen % ${x.every} == 0:`, '        print(f"generation {gen:4d}  best {max(fit):.2f}  shape {shape(population[0])}, {size(population[0])} weights  score {judge.score(best_move):.3f}")'].filter((l) => l !== null);
  }

  // evolution strategies: one network, nudged both ways along random directions; Adam steps along their estimate
  function esPy(d, steps, x) {
    const s = d.s, pairs = Math.max(2, Math.round((s.pop || 24) / 2));
    return [
      'from torch.nn.utils import parameters_to_vector, vector_to_parameters', `pairs, sigma, lr, episodes = ${pairs}, ${x.flt(s.esSigma > 0 ? s.esSigma : 0.05)}, ${x.flt(s.esLr > 0 ? s.esLr : 0.03)}, ${episodesOf(d)}`,
      'theta = parameters_to_vector(model.parameters()).detach().clone()   # the one network that learns', 'm, v = torch.zeros_like(theta), torch.zeros_like(theta)              # Adam\'s running averages', '', '',
      ...PLAY('w', 'vector_to_parameters(w, model.parameters())', 'best_move', false, false), '', '',
      `for gen in range(${x.big(steps)}):`, '    noise = [torch.randn_like(theta) for _ in range(pairs)]', '    fit = []',
      '    for e in noise:                                # each direction tried both ways, from the same starts', '        fit += [play(theta + sigma * e, 1000 + gen)[0], play(theta - sigma * e, 1000 + gen)[0]]', '    n = len(fit)',
      '    u = torch.empty(n)', '    u[torch.tensor(fit).argsort(stable=True)] = torch.arange(n) / (n - 1) - 0.5   # fitness shaping: ranks from -1/2 to 1/2',
      '    g = sum((u[2 * k] - u[2 * k + 1]) * e for k, e in enumerate(noise)) / (n * sigma)   # the gradient, estimated from scores alone',
      '    m, v = 0.9 * m + 0.1 * g, 0.999 * v + 0.001 * g * g', '    theta = theta + lr * (m / (1 - 0.9 ** (gen + 1))) / ((v / (1 - 0.999 ** (gen + 1))).sqrt() + 1e-8)   # Adam, going uphill',
      '    vector_to_parameters(theta, model.parameters())                    # on show: the network that learns',
      `    if gen % ${x.every} == 0:`, '        print(f"generation {gen:4d}  best {max(fit):.2f}  average {sum(fit) / n:.2f}  score {judge.score(best_move):.3f}")'];
  }

  // MAP-Elites: the best network of every style of play, in a grid of 12 × 12 styles
  function mePy(d, steps, x) {
    const s = d.s, axes = (NA.rlEnvs.WORLDS[d.world] || {}).axes || ['style 1', 'style 2'];
    return [
      'from torch.nn.utils import parameters_to_vector, vector_to_parameters', `batch, sigma, episodes = ${s.pop}, ${x.flt(s.sigma)}, ${episodesOf(d)}`,
      'w0 = parameters_to_vector(model.parameters()).detach()', `archive = {}                                       # a cell of the grid of styles (${axes.join(', ')}) -> (fitness, weights) of its best network`, '', '',
      ...PLAY('w', 'vector_to_parameters(w, model.parameters())', 'best_move', true, false), '', '',
      'def cell(style):                                   # which of the 12 x 12 cells a style falls in', '    return min(11, int(style[0] * 12)) + 12 * min(11, int(style[1] * 12))', '', '',
      'def mutate(w):                                     # one weight in ten nudged by sigma at random', '    return w + (torch.rand_like(w) < 0.1) * sigma * torch.randn_like(w)', '', '',
      `for gen in range(${x.big(steps)}):`, '    elites = list(archive.values())', '    for _ in range(batch):',
      '        # random networks while the grid is nearly empty, then mutated copies of elites picked at random', '        w = w0 + 0.5 * torch.randn_like(w0) if len(elites) < batch else mutate(random.choice(elites)[1])',
      '        f, style, _ = play(w, 1000 + gen)', '        c = cell(style)', '        if c not in archive or f > archive[c][0]:  # it stays if its cell is empty, or it beats the elite there', '            archive[c] = (f, w)',
      '    best = max(archive.values(), key=lambda e: e[0])', '    vector_to_parameters(best[1], model.parameters())                  # on show: the best of all the styles',
      `    if gen % ${x.every} == 0:`, '        print(f"generation {gen:4d}  {len(archive)} of 144 styles found  best {best[0]:.2f}  score {judge.score(best_move):.3f}")'];
  }

  // novelty search: the genetic algorithm's breeding, each member chosen for how new its behaviour is
  function novPy(d, steps, x) {
    const s = d.s, sel = PICK[s.select] ? s.select : 'tournament', safe = s.mutate === 'safe', half = s.novMix === 'half', axes = (NA.rlEnvs.WORLDS[d.world] || {}).axes || ['style 1', 'style 2'];
    const k = Math.max(1, Math.round(s.novK) || 10), add = Math.max(1, Math.round(s.novAdd) || 2), most = NA.bevo ? NA.bevo.ARCHIVE : 1000;
    return [
      'from torch.nn.utils import parameters_to_vector, vector_to_parameters', `pop_size, sigma, episodes = ${s.pop}, ${x.flt(s.sigma)}, ${episodesOf(d)}`,
      `k_nearest, keep_new = ${k}, ${add}                         # novelty: the mean distance to the k nearest; the most novel few kept a generation`, ...FIRST,
      `archive = []                                       # behaviours (${axes.join(', ')}) kept as stepping stones: a place once reached stops being new`,
      'champion = (-1.0, -math.inf, w0.clone())           # the best by score found so far (its score, its return, its weights): the one on show', '', '',
      ...PLAY('w', 'vector_to_parameters(w, model.parameters())', 'best_move', 'its novelty', safe), '', '',
      ...rows(String.raw`def novelty(styles):
    """How new each behaviour is: its mean distance to its k nearest among the generation's others and the archive."""
    out = []
    for i, a in enumerate(styles):
        near = sorted([math.dist(a, b) for j, b in enumerate(styles) if j != i] + [math.dist(a, b) for b in archive])
        out.append(sum(near[:k_nearest]) / max(1, min(k_nearest, len(near))))
    return out`), '', '',
      ...(half ? [...rows(String.raw`def ranks(v):                                      # each value's rank, from 0 (the lowest) to 1 (the highest)
    r = [0.0] * len(v)
    for i, k in enumerate(sorted(range(len(v)), key=lambda k: v[k])):
        r[k] = i / max(1, len(v) - 1)
    return r`), '', ''] : []),
      ...PICK[sel], '', '', ...mixPy(s, x), ...mutatePy(safe), '', '',
      `for gen in range(${x.big(steps)}):`, `    results = [play(w, 1000 + gen${safe ? ', keep=True' : ''}) for w in population]`, '    ret, nov = [r[0] for r in results], novelty([r[1] for r in results])',
      half ? '    fit = [(a + b) / 2 for a, b in zip(ranks(nov), ranks(ret))]   # half novelty, half reward (as ranks)' : '    fit = nov                                      # chosen for novelty alone: the reward plays no part',
      '    archive += [results[k][1] for k in sorted(range(pop_size), key=lambda k: -nov[k])[:keep_new]]', `    del archive[:-${most}]                           # (the oldest go first)`,
      '    top = max(range(pop_size), key=lambda k: ret[k])   # the best by return, scored: it is on show if it beats the champion', '    vector_to_parameters(population[top], model.parameters())',
      '    champion = max(champion, (judge.score(best_move), ret[top], population[top].clone()), key=lambda c: c[:2])', '    order = sorted(range(pop_size), key=lambda k: -fit[k])',
      ...breedPy(s),
      '    population = children', '    vector_to_parameters(champion[2], model.parameters())                # on show: the best by score found so far',
      `    if gen % ${x.every} == 0:`, '        print(f"generation {gen:4d}  most novel {max(nov):.3f}  best return {max(ret):.2f}  archive {len(archive)}  score {judge.score(best_move):.3f}")'].filter((l) => l !== null);
  }

  // the cross-entropy method: a bell curve over the weights, refitted every generation to the best it drew
  function cemPy(d, steps, x) {
    const s = d.s, N = Math.max(4, Math.round(s.pop) || 24), E = Math.max(2, Math.round(N * (s.cemElite > 0 && s.cemElite < 1 ? s.cemElite : 0.2)));
    return [
      'from torch.nn.utils import parameters_to_vector, vector_to_parameters', `pop_size, elite, episodes = ${N}, ${E}, ${episodesOf(d)}                  # the best ${E} of each generation refit the bell curve`,
      'mu = parameters_to_vector(model.parameters()).detach().clone()   # the middle of the bell curve: the network you built',
      `sd = torch.full_like(mu, ${x.flt(s.cemSigma > 0 ? s.cemSigma : 0.3)})                         # its spread, the same for every weight at first`,
      `sd_floor = ${x.flt(s.cemFloor >= 0 ? s.cemFloor : 0.05)}                                    # the spread never falls below this, so it keeps looking`, '', '',
      ...PLAY('w', 'vector_to_parameters(w, model.parameters())', 'best_move', false, false), '', '',
      `for gen in range(${x.big(steps)}):`, '    samples = [mu + sd * torch.randn_like(mu) for _ in range(pop_size)]   # a generation drawn from the bell curve',
      '    fit = [play(w, 1000 + gen)[0] for w in samples]', '    best = torch.stack([samples[k] for k in sorted(range(pop_size), key=lambda k: -fit[k])[:elite]])',
      '    mu, sd = best.mean(0), (best.var(0, unbiased=False) + sd_floor ** 2).sqrt()   # refitted to the elite',
      '    vector_to_parameters(mu, model.parameters())                       # on show: the middle of the bell curve',
      `    if gen % ${x.every} == 0:`, '        print(f"generation {gen:4d}  best {max(fit):.2f}  average {sum(fit) / pop_size:.2f}  spread {sd.mean():.3f}  score {judge.score(best_move):.3f}")'];
  }

  // ERL: a genetic algorithm whose every move teaches a DQN; the DQN's network joins the population every few generations
  function erlPy(d, steps, x) {
    const s = d.s, sel = PICK[s.select] ? s.select : 'tournament', safe = s.mutate === 'safe';
    // (as on the page: a smaller memory when each observation is big, a picture or a long list)
    const join = Math.max(1, Math.round(s.erlEvery) || 5), learn = Math.max(1, Math.round(s.erlSteps) || 200), lr = s.erlLr > 0 ? s.erlLr : 0.001, cap = d.Xtr.length / d.nTr > 64 ? 5000 : 20000;
    return [
      'import collections', 'import copy', 'from torch.nn.utils import parameters_to_vector, vector_to_parameters',
      `pop_size, sigma, episodes = ${s.pop}, ${x.flt(s.sigma)}, ${episodesOf(d)}`,
      `gamma, rl_steps, rl_every, batch = ${x.flt(s.gamma)}, ${learn}, ${join}, 32   # the DQN: its discount, its steps a generation, how often it joins`, ...FIRST,
      `memory = collections.deque(maxlen=${cap})           # every move the population makes: (observation, move, reward, next observation or None)`,
      'q_net, q_target = copy.deepcopy(model), copy.deepcopy(model).eval()   # the DQN: a network like yours, and its target network',
      `q_opt = torch.optim.Adam(q_net.parameters(), lr=${x.flt(lr)})`, 'updates = 0', '', '',
      ...PLAY('w', 'vector_to_parameters(w, model.parameters())', 'best_move', false, safe, true), '', '',
      ...rows(String.raw`def learn():
    """Gradient steps of the DQN on moves drawn from the memory, as the page's DQN learns: toward the reward plus gamma
    times the next state's value (double DQN: the DQN picks the next move, its target network says what it is worth),
    a Huber loss, Adam; the target network refreshed every 500 steps. Returns the average loss."""
    global updates
    total = 0.0
    for _ in range(rl_steps):
        sample = [memory[random.randrange(len(memory))] for _ in range(batch)]
        X, A = torch.stack([q[0] for q in sample]), torch.tensor([q[1] for q in sample])
        R, alive = torch.tensor([float(q[2]) for q in sample]), torch.tensor([float(q[3] is not None) for q in sample])
        X2 = torch.stack([q[0] if q[3] is None else q[3] for q in sample])
        with torch.no_grad():
            q_net.eval()
            nxt = q_net(X2)[:, :n_actions].argmax(1)
            q_net.train()
            y = R + gamma * alive * q_target(X2).gather(1, nxt[:, None])[:, 0]
        loss = F.smooth_l1_loss(q_net(X).gather(1, A[:, None])[:, 0], y)
        q_opt.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(q_net.parameters(), 1.0)
        q_opt.step()
        total, updates = total + loss.item(), updates + 1
        if updates % 500 == 0:
            q_target.load_state_dict(q_net.state_dict())
    return total / rl_steps`), '', '',
      ...PICK[sel], '', '', ...mixPy(s, x), ...mutatePy(safe), '', '',
      `for gen in range(${x.big(steps)}):`, `    results = [play(w, 1000 + gen${safe ? ', keep=True' : ''}) for w in population]`, '    fit = [r[0] for r in results]', '    order = sorted(range(pop_size), key=lambda k: -fit[k])',
      '    rl_loss = learn() if len(memory) >= 64 else float("nan")   # the DQN learns from the moves, between generations',
      ...breedPy(s),
      '    if (gen + 1) % rl_every == 0:                  # every few generations the DQN\'s network takes the place of the worst',
      '        children[-1] = parameters_to_vector(q_net.parameters()).detach().clone()',
      '    population = children', '    vector_to_parameters(population[0], model.parameters())             # on show: the best of the generation',
      `    if gen % ${x.every} == 0:`, '        print(f"generation {gen:4d}  best {max(fit):.2f}  average {sum(fit) / pop_size:.2f}  DQN loss {rl_loss:.4f}  score {judge.score(best_move):.3f}")'].filter((l) => l !== null);
  }

  /* Grow and prune while it learns from examples: 6 networks (unless set) learn side by side from the same batches;
     every 150 steps (unless set) each is judged on the validation examples and the worse half replaced by copies of
     the better half, each with one change to its shape, by the odds and limits set (EV.growOptions). x: { flt, num,
     big, ACT_FN, setup, d, evalEvery, gains (each layer's starting gain, or null for "tiny"), clip, lrAt (the
     schedule's lines, or null) } */
  PW.growSupervised = function (x) {
    const T = x.setup.train, o = NA.bevo.growOptions(T, true), steps = T.steps || 3000, every = o.every, N = o.members, pen = NA.bevo.PENALTY_EX[o.penalty], acts = actsOf(x.setup.blocks), small = o.start === 'small';
    const opt = { sgd: 'torch.optim.SGD(groups, lr=LR)', momentum: 'torch.optim.SGD(groups, lr=LR, momentum=0.9)', adam: 'torch.optim.Adam(groups, lr=LR)', adamw: 'torch.optim.AdamW(groups, lr=LR)' }[T.opt] || 'torch.optim.Adam(groups, lr=LR)';
    return [
      '# ---------------------------------------------------------------- grow and prune while it learns',
      `# ${N} networks learn side by side from the same batches, each a list of Dense layers [W1, b1, ..., W_out, b_out]`, `# whose shape can change. Every ${every} steps each is judged on the validation examples (its loss there, plus`,
      `# ${x.num(pen)} for every 100 weights) and the worse half replaced by copies of the better half, each born with one`, ...`# ${wrapDoc(`change to its shape (it may ${shapeSay(o)}):`)}`.split('\n').map((l, i) => (i ? `# ${l.trim()}` : l)), '# population-based training, with the architecture evolving instead of the learning rate.',
      `ACTS = ${JSON.stringify(acts)}`, ...netOps(acts, o, x.ACT_FN), '', '',
      `LR, WD, PENALTY = ${x.flt(T.lr)}, ${x.flt(T.wd || 0)}, ${x.num(pen)}`, '', '',
      'def member(w, acts):                               # a network that learns: its weights and its own optimizer', '    w = [t.detach().clone().requires_grad_() for t in w]',
      '    groups = [{"params": w[::2], "weight_decay": WD}, {"params": w[1::2], "weight_decay": 0.0}]   # decay on the weights, not the biases', `    return {"w": w, "acts": acts, "opt": ${opt}}`, '', '',
      `GAINS = ${x.gains ? JSON.stringify(x.gains) : 'None'}${x.gains ? '                              # each layer\'s starting-weight gain, as on the page' : '                                   # "tiny" starting weights'}`,
      'def fresh():                                       # a new network of the shape you built, started as on the page', '    m = Model()', '    layers = [l for l in m.modules() if isinstance(l, nn.Linear)]',
      '    for k, layer in enumerate(layers):', `        ${x.gains ? 'init_(layer, GAINS[k])' : 'init_(layer)'}`, '    return [t for l in layers for t in (l.weight, l.bias)]', '', '',
      ...(small ? ['def smallest():                                    # the smallest network: no hidden layer, the inputs straight to the outputs', '    out = [l for l in Model().modules() if isinstance(l, nn.Linear)][-1].out_features',
        '    layer = nn.Linear(Xtr.shape[1], out)', `    ${x.gains ? 'init_(layer, GAINS[-1])' : 'init_(layer)'}`, '    return [layer.weight, layer.bias]', '', '',
        `members = [member(smallest(), []) for k in range(${N})]   # every one starts small: each neuron and layer is one evolution added`]
        : ['mine = [t for l in model.modules() if isinstance(l, nn.Linear) for t in (l.weight, l.bias)]', `members = [member(mine if k == 0 else fresh(), ACTS) for k in range(${N})]`]), 'best = members[0]', '', '',
      ...(x.lrAt ? [...x.lrAt, '', ''] : []),
      'def generation():', '    """Each judged on the validation examples; the worse half replaced by copies of the better half, weights and all,', '    each with one change to its shape."""', '    global best',
      '    with torch.no_grad():', '        for m in members:', '            m["fit"] = -loss_fn(forward(m, Xva), Yva).item() - PENALTY * size(m) / 100', '    members.sort(key=lambda m: -m["fit"])', '    best, half = members[0], (len(members) + 1) // 2',
      '    for k in range(half, len(members)):', '        child = change_shape({"w": [t.detach() for t in members[k - half]["w"]], "acts": members[k - half]["acts"]})', '        members[k] = member(child["w"], child["acts"])', '', '',
      'def evaluate(X, Y):', '    with torch.no_grad():', '        out = forward(best, X)', '        return loss_fn(out, Y).item(), score(out, Y).item()', '', '',
      '# ---------------------------------------------------------------- training', `batch = ${T.batch}`, 'order, pos = torch.randperm(len(Xtr)), 0',
      `for step in range(${x.big(steps)}):`, '    if pos + batch > len(order):                  # a new shuffle every pass over the data', '        order, pos = torch.randperm(len(Xtr)), 0',
      '    i = order[pos:pos + batch]', '    pos += batch', '    for m in members:                            # every network takes a step on the same batch',
      x.lrAt ? '        for g in m["opt"].param_groups:' : null, x.lrAt ? '            g["lr"] = lr_at(step)' : null,
      '        loss = loss_fn(forward(m, Xtr[i]), Ytr[i])', '        m["opt"].zero_grad()', '        loss.backward()', x.clip ? `        torch.nn.utils.clip_grad_norm_(m["w"], ${x.num(x.clip)})` : null, '        m["opt"].step()',
      `    if (step + 1) % ${every} == 0:`, '        generation()',
      `    if step % ${x.evalEvery} == 0:`, '        val_loss, val_score = evaluate(Xva, Yva)', '        print(f"step {step:5d}  validation loss {val_loss:.4f}  score {val_score:.3f}  the best: {shape(best)}, {size(best)} weights")', '',
      'val_loss, val_score = evaluate(Xva, Yva)', 'print(f"done: validation loss {val_loss:.4f}, score {val_score:.3f}; the best network: hidden layers {shape(best)}, {size(best)} weights")'].filter((l) => l !== null);
  };

  PW.netOps = netOps; // (for tests)
  NA.builderPyWorlds = PW;
})(typeof globalThis !== 'undefined' && globalThis.NA ? globalThis.NA : window.NA);
