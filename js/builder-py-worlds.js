/* builder-py-worlds.js — for builder-py.js (the model builder's PyTorch scripts): the three bigger worlds of
   rl-worlds.js in Python (the race track, the drone arena seen in the first person, the rocket), their physics
   step for step as on the page (their random numbers come from Python's random, so the starts, where the drones
   appear and the random tracks differ), each measuring its style of play for MAP-Elites; and the evolutionary
   methods of builder-evo.js: a genetic algorithm with its ways of choosing, mixing and mutating parents, grow and
   prune (networks whose shape evolves: lists of weight matrices that gain or lose neurons and layers), evolution
   strategies and MAP-Elites; and grow and prune while a network learns from examples. */
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
    const reward = f ? ['        progress, speed, crashed, lap, t = ds / T.L, self.v, float(crashed), float(lap), self.t', '        center, turn, brake = min(1.0, abs(side) / HW), float(steer), float(not gas)', ...h.ownLine]
      : ['        progress = ds / T.L                        # laps of progress this step', `        r = ${terms([rw.rProgress ? `${num(rw.rProgress)} * progress` : null, rw.rLap ? `(${num(rw.rLap)} if lap else 0.0)` : null, rw.rCrash ? `(${num(rw.rCrash)} if crashed else 0.0)` : null, rw.rStep ? num(rw.rStep) : null])}`];
    const pays = h.pays(`${say(rw.rProgress)} for a lap of progress (backwards counts against it)${rw.rCrash ? `, ${say(rw.rCrash)} for crashing` : ''}${rw.rLap ? `, ${say(rw.rLap)} for each lap done` : ''}${rw.rStep ? `, ${say(rw.rStep)} a step` : ''}.`);
    const obs = cam ? rows(String.raw`    def obs(self):                                 # (2, 16, 16): the track from above, ahead up, 32 m from 4 m behind and 12 m to each side; its speed
        c, s, T = math.cos(self.th), math.sin(self.th), self.track
        near = [[0.0] * 16 for _ in range(16)]
        for i in range(16):
            ahead = (15.5 - i) / 16 * 32 - 4
            for j in range(16):
                right = (j + 0.5) / 16 * 24 - 12
                near[i][j] = float(T.center_dist(self.x + ahead * c - right * s, self.y + ahead * s + right * c) < HW)
        return torch.tensor([near, [[self.v / VMAX] * 16] * 16])`)
      : rows(String.raw`    def obs(self):                                 # 7 distances to the walls (of 40 m) and its speed (of 24 m/s)
        return torch.tensor([r / RANGE for r in self.rays] + [self.v / VMAX])`);
    return [
      ...rows(String.raw`TRACK = ${pts(random ? 'oval' : id)}   # ${random ? 'the Oval, if no random track works out' : E.TRACKS[id].name}: points its middle line passes through, m
HW = 4.5                                           # half the track's width, m
VMAX, VMIN, ACC, BRAKE, GRIP, STEER = 24.0, 3.0, 9.0, 16.0, 14.0, 1.6   # speeds (m/s), throttle and brake (m/s²), grip (m/s² sideways), steering (rad/s)
CAR_R, DT, RANGE = 1.0, 0.1, 40.0                  # the car's radius (m), a step (s), how far its sensors see (m)
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
        self.min_radius = min(4 * self.ds / (abs(wrap(heading[(k + 2) % N] - heading[k - 2])) or 1e-9) for k in range(N))
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
    """A car drives 4 laps of ${random ? 'a random track (a new one every episode)' : `the ${E.TRACKS[id].name.toLowerCase()} track`}: it steers, and brakes for the tight bends. ${pays}
    It crashes when it touches a wall; 4 laps, or ${steps || 'as many steps as 4 laps at 10 m/s take'}${steps ? ' steps' : ''}, also end an episode."""
    actions = ["steer left", "straight on", "steer right", "brake"]`)), '',
      ...rows(String.raw`    def __init__(self):
        ${random ? 'self.tests = [random_track(s) for s in (7001, 7002, 7003, 7004)]   # 4 tracks it never trains on' : 'self.fixed = Track(TRACK)'}

    def reset(self, track=None):                   # at the start line, pointing along the track
        self.track = T = track or ${random ? 'random_track(1 + random.randrange(64))' : 'self.fixed'}
        self.max_steps = ${steps || 'round(4 * T.L)'}${steps ? '' : '                 # 4 laps at an average of 10 m/s'}
        self.k, self.x, self.y, self.v = 0, T.cx[0], T.cy[0], 8.0
        self.th = math.atan2(T.ty[0], T.tx[0])
        self.s, self.prog, self.lap, self.t, self.end = T.arc(self.x, self.y, 0), 0.0, 0, 0, None
        self.sum_v = self.sum_side = 0.0
        self.see()
        return self.obs()

    def see(self):                                 # what its 7 sensors read
        self.rays = [self.track.ray(self.x, self.y, self.th + a) for a in RAYS]
`),
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
        done = crashed or self.prog >= 4 * T.L or self.t >= self.max_steps
        if done:
            self.end = "crashed" if crashed else "finished" if self.prog >= 4 * T.L else "time"`),
      ...reward, '        return self.obs(), r, done', '',
      ...rows(String.raw`    def behavior(self):                            # its style: average speed (slow to fast), the side it keeps (left to right)
        n = max(1, self.t)
        return min(max(self.sum_v / n / VMAX, 0.0), 1.0), min(max(0.5 + 1.2 * self.sum_side / n, 0.0), 1.0)

    def score(self, policy):
        """The share of 4 laps its best moves drive${random ? ', on 4 tracks it never trained on' : ''}."""
        total = 0.0
        for T in ${random ? 'self.tests' : '[self.fixed]'}:
            o, done = self.reset(T), False
            while not done:
                o, _, done = self.step(policy(o))
            total += min(max(self.prog / (4 * T.L), 0.0), 1.0)
        return total / ${random ? 4 : 1}`)];
  };

  // ------------------------------------------------------------------ the drone arena, in the first person
  PW.worlds.arena = function (d, h) {
    const ws = d.ws, rw = h.rw, f = h.f, num = h.num, say = h.say, E = NA.rlEnvs, px = ws.view === 'pixels';
    const map = E.ARENAS[ws.map] ? ws.map : 'room', k = Math.min(3, Math.max(1, Math.round(ws.drones) || 1)), fast = ws.speed === 'fast', most = ws.steps > 0 ? Math.min(ws.steps, 3000) : 500;
    const reward = f ? ['        aimed = seen = 0.0                         # a drone in its sights; how many it can see', '        for d in self.drones:', '            if d.alive:', '                rel, dist, rad, visible = self.sight(d)', '                if visible:', '                    seen += 1', '                    aimed = max(aimed, float(abs(rel) < rad + 0.02))',
      '        hit, hurt, shot, lives, t, moved, turned = float(hit), float(hurt), float(shot), float(self.health), self.t, float(walk), float(abs(turn))', ...h.ownLine]
      : [`        r = ${terms([rw.rHit ? `(${num(rw.rHit)} if hit else 0.0)` : null, rw.rHurt ? `${num(rw.rHurt)} * hurt` : null, rw.rShot ? `(${num(rw.rShot)} if shot else 0.0)` : null, rw.rStep ? num(rw.rStep) : null])}`];
    const pays = h.pays(`${say(rw.rHit)} for each drone destroyed, ${say(rw.rHurt)} for each that reaches it${rw.rShot ? `, ${say(rw.rShot)} a shot` : ''}${rw.rStep ? `, ${say(rw.rStep)} a step` : ''}.`);
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
        v[24], v[25] = self.health / 3, float(self.cool == 0)
        return torch.tensor(v)`);
    return [
      `ARENA = [${E.ARENAS[map].rows.map((r) => JSON.stringify(r)).join(', ')}]   # ${E.ARENAS[map].name}: # a wall, S where it starts`,
      ...rows(String.raw`MH, MW = len(ARENA), len(ARENA[0])
FOV, COLS, VIEW = math.radians(66), 11, 8.0         # its field of view, the columns it sees in, how far it sees (m)${px ? '\nPROWS, PCOLS = 10, 20                              # the picture it sees' : ''}
DRONE_R, ME_R, TURN, WALK = 0.3, 0.25, math.radians(12), 0.22   # sizes (m), a turn, a step's walk (m)
DRONES, SPEED, MAX_STEPS = ${k}, ${fast ? 0.085 : 0.045}, ${most}              # drones at once, how far they fly a step (m), the longest episode


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
    """A first-person arena: it turns 12 degrees, walks 0.22 m or fires (the gun then recharges for 3 steps). Glowing
    drones drift toward it, weaving; a shot that crosses one destroys it, one that reaches it costs a life (3 lives).
    ${pays} 10 drones destroyed, no lives left or ${most} steps end an episode."""
    actions = ["turn left", "turn right", "walk forward", "fire"]`)), '',
      ...rows(String.raw`    def reset(self):
        s = "".join(ARENA).index("S")
        self.x, self.y, self.th = s % MW + 0.5, s // MW + 0.5, 0.0
        self.t, self.health, self.kills, self.cool, self.end, self.moved, self.shots = 0, 3, 0, 0, None, 0, 0
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
        if walk:
            x0, y0 = self.x, self.y
            slide(self, math.cos(self.th) * WALK, math.sin(self.th) * WALK, ME_R)
            self.moved += math.hypot(self.x - x0, self.y - y0) > 0.01
        shot = hit = False
        if self.cool:                              # the gun recharging
            self.cool -= 1
        elif a == 3:                               # a shot: the nearest drone in line before the wall
            shot, self.cool = True, 3
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
        done = self.health <= 0 or self.kills >= 10 or self.t >= MAX_STEPS
        if done:
            self.end = "cleared" if self.kills >= 10 else "dead" if self.health <= 0 else "time"`),
      ...reward, '        return self.obs(), r, done', '',
      ...rows(String.raw`    def behavior(self):                            # its style: how much it walks, how often it fires
        n = max(1, self.t)
        return min(self.moved / n, 1.0), min(4 * self.shots / n, 1.0)

    def score(self, policy):
        """The share of 10 drones its best moves destroy, over 3 games (the same 3 every time)."""
        total = 0.0`),
      ...h.SEEDED(4242, ['for _ in range(3):', ...h.PLAY('    '), '    total += self.kills / 10']).map((l) => '        ' + l), '        return total / 3'];
  };

  // ------------------------------------------------------------------ the rocket
  PW.worlds.lander = function (d, h) {
    const ws = d.ws, rw = h.rw, f = h.f, num = h.num, say = h.say;
    const g = ws.gravity > 0 ? ws.gravity : 1.62, fuel = ws.fuel > 0 ? Math.min(ws.fuel, 3000) : 150, wind = !!ws.wind, randomPad = ws.pad === 'random';
    const reward = f ? ['        dist, speed, tilt = math.hypot(self.x - self.pad, self.y), math.hypot(self.vx, self.vy), math.degrees(self.th)',
      '        landed, crashed, engine, turn, h, vx, vy, fuel, t = float(landed), float(crashed), float(main), float(abs(turn)), self.y, self.vx, self.vy, self.fuel / FUEL, self.t', ...h.ownLine]
      : [`        r = ${terms([rw.rLand ? `(${num(rw.rLand)} if landed else 0.0)` : null, rw.rCrash ? `(${num(rw.rCrash)} if crashed else 0.0)` : null, rw.rFuel ? `(${num(rw.rFuel)} if main else 0.0)` : null, rw.rShape ? `${num(rw.rShape)} * closer` : null])}`];
    const pays = h.pays(`${say(rw.rLand)} for landing softly on the pad, ${say(rw.rCrash)} for anything else${rw.rShape ? `; and ${rw.rShape === 1 ? '' : `${say(rw.rShape)} × `}how much closer, slower and straighter it got each step (shaping)` : ''}${rw.rFuel ? `; ${say(rw.rFuel)} each step the main engine fires` : ''}.`);
    return [
      ...rows(String.raw`G = ${num(g)}                                     # gravity, m/s²
THRUST = 2.4 * G + 1                               # the main engine's push along the rocket, m/s²
FUEL, MAX_STEPS = ${fuel}, 300                          # steps of burning in its tank; the longest episode
LEG, FOOT, PAD = 1.8, 1.4, 6.0                     # its legs reach 1.8 m down and 1.4 m out; the pad is 12 m wide`), '', '',
      ...h.head(rows(String.raw`class World:
    """A rocket comes down from 30 m, up to 12 m to either side of the pad, drifting and tilted a little. Its main
    engine pushes along its axis, its side thrusters turn it, while the fuel lasts. When a leg touches the ground on
    the pad, slower than 3 m/s down and 2 m/s across and within 25 degrees of upright, it has landed. ${pays}
    Still up there after 300 steps of 0.08 s counts as a crash.${wind ? ' Wind pushes it sideways in gusts.' : ''}"""
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
            self.end = "landed" if landed else "off the pad" if soft else "crashed"
        elif abs(self.x - self.pad) > 50 or self.y > 60:
            crashed, self.end = True, "flew away"
        elif self.t >= MAX_STEPS:                  # out of time counts as a failure: no hovering for ever
            crashed, self.end = True, "out of time"
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
  const PLAY = (args, load, move, style, keep) => rows(String.raw`def play(${args}, seed${keep ? ', keep=False' : ''}):
    """Its best moves over a few episodes, from the same starts as everyone else in its generation: the average return
    (its fitness)${style ? ', its style in the first (for MAP-Elites)' : ''}${keep ? ' and some of what it saw there (for safe mutations)' : ''}."""${load ? `\n    ${load}` : ''}
    total, style, seen = 0.0, None, []
    state = random.getstate()
    random.seed(seed)
    for k in range(episodes):
        o, done, t = env.reset(), False, 0
        while not done:${keep ? '\n            if keep and k == 0 and len(seen) < 32 and t % 5 == 0:\n                seen.append(o)' : ''}
            o, r, done = env.step(${move}(o))
            total, t = total + r, t + 1${style ? '\n        if k == 0:\n            style = env.behavior()' : ''}
    random.setstate(state)
    return total / episodes, style, seen`);
  const PICK = {
    tournament: ['def pick(order, fit):                              # a parent: the best of 3 picked at random (a tournament)', '    best = random.randrange(len(fit))', '    for _ in range(2):', '        c = random.randrange(len(fit))', '        if fit[c] > fit[best]:', '            best = c', '    return best'],
    truncation: ['def pick(order, fit):                              # a parent: one of the top fifth, at random', '    return order[random.randrange(max(2, round(len(fit) / 5)))]'],
    rank: ['def pick(order, fit):                              # a parent by rank: the best n times as likely as the worst', '    return random.choices(order, weights=range(len(fit), 0, -1))[0]'],
  };
  const episodesOf = (d) => (NA.brl.WORLD[d.world] || NA.brl.WORLD.pole).gaEpisodes + (d.ws.random || d.ws.slip || (d.world === 'track' && d.ws.track === 'random') ? 2 : 0);

  PW.method = function (d, T, steps, x) {
    const s = d.s, m = d.method === 'me' && d.world === 'levers' ? 'ga' : d.method, every = x.every;
    if (m === 'es') return esPy(d, steps, x);
    if (m === 'me') return mePy(d, steps, x);
    const note = d.method === 'me' ? ['# MAP-Elites keeps the best network of every style of play, and a pull of a lever has no style to tell apart:', '# this runs the genetic algorithm instead.'] : [];
    if (s.grow && x.evolvable) return [...note, ...growPy(d, steps, x)];
    return [...note, s.grow ? '# Grow and prune is on, but only a model of Dense blocks on a list of numbers can change shape: here only the weights evolve.' : null, ...gaPy(d, steps, x)].filter((l) => l !== null);
    function gaPy() {
      const sel = PICK[s.select] ? s.select : 'tournament', cross = ['uniform', 'neuron', 'none'].includes(s.cross) ? s.cross : 'uniform', neuron = cross === 'neuron' && x.evolvable, safe = s.mutate === 'safe';
      return [
        'from torch.nn.utils import parameters_to_vector, vector_to_parameters', `pop_size, sigma, episodes = ${s.pop}, ${x.flt(s.sigma)}, ${episodesOf(d)}`,
        'w0 = parameters_to_vector(model.parameters()).detach()', '# the first generation: the network you built and copies of it, each moved at random',
        'population = [w0.clone()] + [w0 + 0.5 * torch.randn_like(w0) for _ in range(pop_size - 1)]', '', '',
        ...PLAY('w', 'vector_to_parameters(w, model.parameters())', 'best_move', false, safe), '', '', ...PICK[sel], '', '',
        ...(cross === 'none' ? [] : neuron ? [...rows(String.raw`# which neuron each weight belongs to: a row of a layer's weights and its bias go together
unit_of, n_units = [], 0
for layer in (m for m in model.modules() if isinstance(m, nn.Linear)):
    U, I = layer.weight.shape
    unit_of += [n_units + torch.arange(U).repeat_interleave(I), n_units + torch.arange(U)]
    n_units += U
unit_of = torch.cat(unit_of)


def mix(a, b):                                     # each neuron whole, with all its weights, from one parent or the other
    return torch.where((torch.rand(n_units) < 0.5)[unit_of], a, b)`), '', '']
          : [...(cross === 'neuron' ? ['# (neuron by neuron needs a model of Dense blocks only: here, weight by weight)'] : []), 'def mix(a, b):                                     # each weight from one parent or the other', '    return torch.where(torch.rand_like(a) < 0.5, a, b)', '', '']),
        ...(safe ? rows(String.raw`def sensitivity(w, seen):
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
    return w + d * sigma * 0.7979 * chosen.sum() / d.abs().sum().clamp_min(1e-12)`) : ['def mutate(w):                                     # one weight in ten nudged by sigma at random', '    return w + (torch.rand_like(w) < 0.1) * sigma * torch.randn_like(w)']), '', '',
        `for gen in range(${x.big(steps)}):`, `    results = [play(w, 1000 + gen${safe ? ', keep=True' : ''}) for w in population]`, '    fit = [r[0] for r in results]', '    order = sorted(range(pop_size), key=lambda k: -fit[k])',
        '    children = [population[order[0]].clone(), population[order[1]].clone()]   # the best two carry on unchanged', safe ? '    sens = {}' : null,
        '    while len(children) < pop_size:', '        i = pick(order, fit)', '        child = population[i]',
        ...(cross === 'none' ? [] : ['        j = pick(order, fit)', '        if j != i:', '            child = mix(population[i], population[j])']),
        ...(safe ? ['        if i not in sens:', '            sens[i] = sensitivity(population[i], results[i][2])', '        children.append(mutate(child, sens[i]))'] : ['        children.append(mutate(child))']),
        '    population = children', '    vector_to_parameters(population[0], model.parameters())             # on show: the best of the generation',
        `    if gen % ${every} == 0:`, '        print(f"generation {gen:4d}  best {max(fit):.2f}  average {sum(fit) / pop_size:.2f}  score {judge.score(best_move):.3f}")'].filter((l) => l !== null);
    }
  };

  // the operations on a network whose shape can change: its outputs, its size, a neuron or a layer more or less
  function netOps(acts, newLayer, ACT_FN) {
    const used = [...new Set([...acts, 'relu'])];
    return rows(String.raw`ACT = {${used.map((a) => `"${a}": ${ACT_FN[a] || 'F.relu'}`).join(', ')}}
MOST_LAYERS, MOST_UNITS = 6, 48


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
    """A neuron more in hidden layer h: its incoming weights at random, its outgoing ones 0, so it computes exactly
    what it did, and later mutations can put the new neuron to work."""
    w = list(net["w"])
    W, b, W2 = w[2 * h], w[2 * h + 1], w[2 * h + 2]
    if W.shape[0] >= MOST_UNITS:
        return None
    w[2 * h] = torch.cat([W, torch.randn(1, W.shape[1]) * math.sqrt(2 / W.shape[1])])
    w[2 * h + 1] = torch.cat([b, torch.zeros(1)])
    w[2 * h + 2] = torch.cat([W2, torch.zeros(W2.shape[0], 1)], 1)
    return {"w": w, "acts": net["acts"]}


def remove_neuron(net, h):
    """The neuron of hidden layer h that the next layer listens to least (the smallest outgoing weights), taken out."""
    w = list(net["w"])
    W, b, W2 = w[2 * h], w[2 * h + 1], w[2 * h + 2]
    if W.shape[0] <= 1:
        return None
    j = int(W2.abs().sum(0).argmin())
    keep = [i for i in range(W.shape[0]) if i != j]
    w[2 * h], w[2 * h + 1], w[2 * h + 2] = W[keep], b[keep], W2[:, keep]
    return {"w": w, "acts": net["acts"]}


def add_layer(net, p):
    """A new hidden layer at position p (0: straight after the input), as wide as what it reads: ${newLayer === 'random' ? 'its weights at random' : 'its weights the\n    identity, so it passes its input straight through (exactly, after a ReLU layer)'}."""
    acts = list(net["acts"])
    U = net["w"][2 * p].shape[1]                   # the width of what it reads
    if len(acts) >= MOST_LAYERS or U > MOST_UNITS:
        return None
    w = list(net["w"])
    w[2 * p:2 * p] = [${newLayer === 'random' ? 'torch.randn(U, U) * math.sqrt(2 / U)' : 'torch.eye(U)'}, torch.zeros(U)]
    acts.insert(p, acts[p - 1] if p > 0 else acts[0] if acts else "relu")
    return {"w": w, "acts": acts}


def remove_layer(net, h):
    """Hidden layer h taken out and folded into the next: W = W2 W1, b = W2 b1 + b2 (as if its activation let
    values through)."""
    if not net["acts"]:
        return None
    w = list(net["w"])
    W1, b1, W2, b2 = w[2 * h:2 * h + 4]
    w[2 * h:2 * h + 4] = [W2 @ W1, W2 @ b1 + b2]
    return {"w": w, "acts": net["acts"][:h] + net["acts"][h + 1:]}


def change_shape(net):
    """One change to its shape: a neuron more (most often) or less, or a layer more or less."""
    n = len(net["acts"])
    ops = {"neuron+": 3 if n else 0, "neuron-": 2 if n else 0, "layer+": 1.4 if n < MOST_LAYERS else 0, "layer-": 1 if n else 0}
    for _ in range(8):
        op = random.choices(list(ops), weights=list(ops.values()))[0]
        if op == "neuron+":
            r = add_neuron(net, random.randrange(n))
        elif op == "neuron-":
            r = remove_neuron(net, random.randrange(n))
        elif op == "layer+":
            r = add_layer(net, random.randrange(n + 1))
        else:
            r = remove_layer(net, random.randrange(n))
        if r:
            return r
    return net`);
  }
  const actsOf = (blocks) => blocks.filter((b) => b.type === 'dense').map((b) => b.act || 'relu');
  const START = 'start = {"w": [t.detach().clone() for m in model.modules() if isinstance(m, nn.Linear) for t in (m.weight, m.bias)], "acts": ACTS}';

  // a genetic algorithm whose networks grow and shrink
  function growPy(d, steps, x) {
    const s = d.s, sel = PICK[s.select] ? s.select : 'tournament', cross = ['uniform', 'neuron', 'none'].includes(s.cross) ? s.cross : 'uniform', safe = s.mutate === 'safe';
    const W = NA.brl.WORLD[d.world] || NA.brl.WORLD.pole, pen = (NA.bevo ? NA.bevo.PENALTY[s.penalty] ?? NA.bevo.PENALTY.light : 0.0005) * (W.scale || 1);
    return [
      '# Each member is a network of its own: its Dense layers as a list [W1, b1, W2, b2, ..., W_out, b_out] and an', '# activation for each hidden layer. Its shape can change: a child may gain or lose a neuron or a whole layer.',
      `ACTS = ${JSON.stringify(actsOf(x.setup.blocks))}`, ...netOps(actsOf(x.setup.blocks), s.newLayer, x.ACT_FN), '', '',
      `pop_size, sigma, episodes = ${s.pop}, ${x.flt(s.sigma)}, ${episodesOf(d)}`,
      `penalty = ${x.num(pen)}                                  # taken off the fitness for every 100 weights: of two that play as well, the smaller wins`,
      START, '# the first generation: the network you built and copies of it, each moved at random',
      'population = [start] + [{"w": [t + 0.5 * torch.randn_like(t) for t in start["w"]], "acts": ACTS} for _ in range(pop_size - 1)]', '', '',
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
      '    children = [population[order[0]], population[order[1]]]   # the best two carry on unchanged', safe ? '    sens = {}' : null,
      '    while len(children) < pop_size:', '        i = pick(order, adj)', '        child = population[i]',
      ...(cross === 'none' ? [] : ['        j = pick(order, adj)', '        if j != i and shape(population[j]) == shape(child) and population[j]["acts"] == child["acts"]:   # only parents of one shape mix', '            child = mix(child, population[j])']),
      ...(safe ? ['        if i not in sens:', '            sens[i] = sensitivity(population[i], results[i][2])', '        child = mutate(child, sens[i]) if shape(child) == shape(population[i]) else child'] : ['        child = mutate(child)']),
      '        if random.random() < 0.25:                 # a quarter of the children: a neuron or a layer more or less', '            child = change_shape(child)', '        children.append(child)',
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

  /* Grow and prune while it learns from examples: 6 networks learn side by side from the same batches; every 150
     steps each is judged on the validation examples and the worse half replaced by copies of the better half, each
     with one change to its shape. x: { flt, num, big, ACT_FN, setup, d, evalEvery, gains (each layer's starting
     gain, or null for "tiny"), clip, lrAt (the schedule's lines, or null) } */
  PW.growSupervised = function (x) {
    const T = x.setup.train, steps = T.steps || 3000, every = T.growEvery || 150, pen = NA.bevo ? NA.bevo.PENALTY_EX[T.growPenalty] ?? NA.bevo.PENALTY_EX.light : 0.004, acts = actsOf(x.setup.blocks);
    const opt = { sgd: 'torch.optim.SGD(groups, lr=LR)', momentum: 'torch.optim.SGD(groups, lr=LR, momentum=0.9)', adam: 'torch.optim.Adam(groups, lr=LR)', adamw: 'torch.optim.AdamW(groups, lr=LR)' }[T.opt] || 'torch.optim.Adam(groups, lr=LR)';
    return [
      '# ---------------------------------------------------------------- grow and prune while it learns',
      `# ${x.size || 6} networks learn side by side from the same batches, each a list of Dense layers [W1, b1, ..., W_out, b_out]`, `# whose shape can change. Every ${every} steps each is judged on the validation examples (its loss there, plus`,
      `# ${x.num(pen)} for every 100 weights) and the worse half replaced by copies of the better half, each with a neuron or a`, '# layer more or less: population-based training, with the architecture evolving instead of the learning rate.',
      `ACTS = ${JSON.stringify(acts)}`, ...netOps(acts, T.growNew, x.ACT_FN), '', '',
      `LR, WD, PENALTY = ${x.flt(T.lr)}, ${x.flt(T.wd || 0)}, ${x.num(pen)}`, '', '',
      'def member(w, acts):                               # a network that learns: its weights and its own optimizer', '    w = [t.detach().clone().requires_grad_() for t in w]',
      '    groups = [{"params": w[::2], "weight_decay": WD}, {"params": w[1::2], "weight_decay": 0.0}]   # decay on the weights, not the biases', `    return {"w": w, "acts": acts, "opt": ${opt}}`, '', '',
      `GAINS = ${x.gains ? JSON.stringify(x.gains) : 'None'}${x.gains ? '                              # each layer\'s starting-weight gain, as on the page' : '                                   # "tiny" starting weights'}`,
      'def fresh():                                       # a new network of the shape you built, started as on the page', '    m = Model()', '    layers = [l for l in m.modules() if isinstance(l, nn.Linear)]',
      '    for k, layer in enumerate(layers):', `        ${x.gains ? 'init_(layer, GAINS[k])' : 'init_(layer)'}`, '    return [t for l in layers for t in (l.weight, l.bias)]', '', '',
      'mine = [t for l in model.modules() if isinstance(l, nn.Linear) for t in (l.weight, l.bias)]', `members = [member(mine if k == 0 else fresh(), ACTS) for k in range(${x.size || 6})]`, 'best = members[0]', '', '',
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
