/* builder-py-worlds2.js — for builder-py.js (the model builder's PyTorch scripts): the eleven worlds of
   rl-worlds2.js and rl-worlds3.js in Python, step for step as on the page (their random numbers come from Python's
   random, so the starts differ): a room to keep warm, hurdles, a lamp to remember, the mountain cart, a pendulum to
   swing up, a robot arm, the deceptive maze, a chaser to escape, traffic lights, a walker and a sheepdog. Each pays
   from its reward table or your own formula, leaves out the inputs you hid, adds sensor noise, scores its best
   moves as the page does and says how it played (its style, for MAP-Elites). Its settings are read off the page's
   own world, made with the same options, so the two always agree. Needs builder-py-worlds.js. */
(function (NA) {
  'use strict';
  const PW = NA.builderPyWorlds;
  if (!PW) return;
  const rows = (s) => s.split('\n');
  const terms = (list) => list.filter(Boolean).join(' + ') || '0.0';
  // the page's world with these settings: what its options come to (lengths, strengths, maps)
  const make = (d) => NA.rlEnvs.make(d.world, d.ws);
  // a line of code with its comment, the comments lined up
  const note = (code, text) => (code.length < 50 ? code.padEnd(51) : code + '   ') + '# ' + text;
  // an angle as Python, computed as the page computes it: a fraction of pi written as one
  const angle = (v, num) => {
    for (const m of [1, 2, 3, 4, 6]) for (let k = -12; k <= 12; k++) if (k && v === (k * Math.PI) / m) return `${k === -1 ? '-' : k === 1 ? '' : k + ' * '}math.pi${m > 1 ? ' / ' + m : ''}`;
    return num(v);
  };
  const WRAP = rows(String.raw`def wrap(a):                                       # an angle, from -pi to pi
    return (a + math.pi) % (2 * math.pi) - math.pi`);

  // ------------------------------------------------------------------ a room to keep warm
  PW.worlds.thermo = function (d, h) {
    const W = make(d), rw = h.rw, num = h.num, say = h.say, home = W.home;
    // the climate: the day's average outside, and how far 15:00 (the warmest) is above it and 03:00 (the coldest) below
    W.shift = 0;
    const warm = W.outside(90), cold = W.outside(18), mean = (warm + cold) / 2, swing = (warm - cold) / 2;
    const reward = h.f ? ['        temp, target, inside, outside, heating, radiator, hour, t = self.T, tg, float(ok), self.To, self.heat, self.R, h, self.t', ...h.ownLine]
      : [`        r = ${terms([rw.rOff ? `${num(rw.rOff)} * off` : null, rw.rHeat ? `${num(rw.rHeat)} * self.heat` : null, rw.rBand ? `(${num(rw.rBand)} if ok else 0.0)` : null,
        rw.rCold ? `${num(rw.rCold)} * cold` : null, rw.rHot ? `${num(rw.rHot)} * hot` : null, rw.rSwitch ? `(${num(rw.rSwitch)} if switched else 0.0)` : null])}`];
    const pays = h.pays(`${say(rw.rOff)} for every degree from the target, every step${rw.rHeat ? `, ${say(rw.rHeat)} for every step the heating is on` : ''}${rw.rBand ? `, ${say(rw.rBand)} for every step within 1 °C of it` : ''}.`);
    return [
      note(`MEAN, SWING = ${num(mean)}, ${num(swing)}`, `outside (a ${W.climate} climate): the day's average, and how far 15:00 is above it and 03:00 below, °C`),
      note(`LEAK = ${num(W.leak)}`, `the share of the gap to the outside the room loses every step (${d.ws.insulation === 'poor' ? 'poor' : 'good'} insulation)`),
      note(`WARM, BAND, HOT = ${num(W.warm)}, ${num(W.band)}, ${num(W.hot)}`, `the temperature wanted (${home ? '5 less when nobody is in' : 'all day'}), how near counts as comfortable, the radiator's hottest, °C`), '', '',
      ...h.head(rows(String.raw`class World:
    """A day in ${W.maxSteps} steps of 10 minutes, from midnight. The heating warms a radiator and the radiator warms the room:
    switched on, the room warms only as fast as the radiator does, and switched off it goes on warming until the
    radiator has cooled, so a good thermostat switches off before the room is warm enough. The room loses heat to the
    outside, coldest before dawn and warmest at three in the afternoon. It should be ${home ? `${W.warm} °C in the morning (6 to 8) and\n    the evening (17 to 23), ${W.warm - 5} °C otherwise` : `${W.warm} °C all day`}.
    ${pays}"""
    actions = ["heating off", "heating on"]`)), '',
      ...rows(String.raw`    def target(self, h):                           # the temperature wanted at hour h, °C
        return ${home ? 'WARM if 6 <= h < 8 or 17 <= h < 23 else WARM - 5' : 'WARM'}

    def outside(self, t):                          # the temperature outside at step t, °C
        h = (t / 6) % 24
        return MEAN + self.shift + SWING * math.cos(2 * math.pi * (h - 15) / 24)

    def reset(self):
        self.shift = random.uniform(-2, 2)         # a day a little warmer or colder than most
        self.t, self.heat, self.used, self.in_band, self.sum_off, self.end = 0, 0, 0, 0, 0.0, None
        self.T = self.target(0) + random.uniform(-3, 1)   # the room, °C
        self.R, self.To = self.T, self.outside(0)  # the radiator and the outside, °C
        return self.obs()
`),
      ...h.sense(rows(String.raw`    def obs(self):                                 # the room from the target, outside, radiator, the target, the time of day, the heating
        h = (self.t / 6) % 24
        tg = self.target(h)
        return torch.tensor([(self.T - tg) / 5, (self.To - 10) / 10, (self.R - 20) / 40, (tg - 18) / 5,
                             math.sin(2 * math.pi * h / 24), math.cos(2 * math.pi * h / 24), float(self.heat)])`)), '',
      ...rows(String.raw`    def step(self, a):                             # 10 minutes
        was = self.heat
        self.heat = 1 if a == 1 else 0
        self.R += ((HOT - self.R) if self.heat else (self.T - self.R)) * 0.25  # on: the radiator heats toward HOT; off: it cools toward the room
        self.To = self.outside(self.t)
        self.T += 0.025 * (self.R - self.T) + LEAK * (self.To - self.T)        # warmed by the radiator, cooled through the walls
        self.t += 1
        h = (self.t / 6) % 24
        tg = self.target(h)
        off = abs(self.T - tg)                     # degrees from the target
        ok = off <= BAND
        cold, hot, switched = max(0.0, tg - self.T), max(0.0, self.T - tg), float(self.heat != was)   # under, over the target; switched
        self.used += self.heat
        self.in_band += ok
        self.sum_off += self.T - tg
        done = self.t >= ${W.maxSteps}
        if done:
            self.end = "comfy" if self.in_band / self.t >= 0.8 else "uncomfy"`),
      ...reward, '        return self.obs(), r, done', '',
      ...rows(String.raw`    def behavior(self):                            # its style: how much it heats, how warm it keeps the room (3 °C under the target to 3 over)
        n = max(1, self.t)
        return min(max(self.used / n, 0.0), 1.0), min(max(0.5 + self.sum_off / n / 6, 0.0), 1.0)

    def score(self, policy):
        """The share of the day its best moves keep within 1 °C of the target, over 3 days (the same 3 every time)."""
        total = 0.0`),
      ...h.SEEDED(4242, ['for _ in range(3):', ...h.PLAY('    '), '    total += self.in_band / self.t']).map((l) => '        ' + l), '        return total / 3'];
  };

  // ------------------------------------------------------------------ hurdles
  PW.worlds.hurdles = function (d, h) {
    const W = make(d), rw = h.rw, num = h.num, say = h.say, bars = W.bars, J = W.jumpV, n1 = (v) => String(+v.toFixed(1));
    const reward = h.f ? ['        q = self.obst[nxt] if nxt < len(self.obst) else None',
      '        hit, finished, ducking, height, dist, t = float(hit), float(finished), float(self.duck > 0), self.y, self.x, self.t',
      '        cleared, air, gap, bar = float(cleared), float(air), q[0] - self.x if q else 99.0, float(q is not None and q[1])', ...h.ownLine]
      : [`        r = ${terms([rw.rRun ? num(rw.rRun) : null, rw.rBump ? `(${num(rw.rBump)} if hit else 0.0)` : null, rw.rFinish ? `(${num(rw.rFinish)} if finished else 0.0)` : null,
        rw.rJump ? `(${num(rw.rJump)} if jumped else 0.0)` : null, rw.rDuck ? `(${num(rw.rDuck)} if self.duck > 0 else 0.0)` : null, rw.rClear ? `(${num(rw.rClear)} if cleared else 0.0)` : null,
        rw.rAir ? `(${num(rw.rAir)} if air else 0.0)` : null])}`];
    const pays = h.pays(`${say(rw.rRun)} for every step it keeps running, ${say(rw.rBump)} for touching a hurdle${bars ? ' or a bar' : ''}, ${say(rw.rFinish)} at the finish${rw.rClear ? `, ${say(rw.rClear)} for each obstacle it passes` : ''}${rw.rJump ? `, ${say(rw.rJump)} for each jump` : ''}${rw.rDuck ? `, ${say(rw.rDuck)} for each step it ducks` : ''}${rw.rAir ? `, ${say(rw.rAir)} for each step in the air` : ''}.`);
    // a jump: its time in the air, and how far it runs with its feet above a hurdle's top (none when it cannot get that high)
    const above = J * J > 2 * 15 * 0.7 ? (2 * Math.sqrt(J * J - 2 * 15 * 0.7) / 15) * W.run : 0;
    return [
      ...rows(String.raw`RUN, DT, JUMP, GRAV = ${num(W.run)}, 0.05, ${num(J)}, 15.0          # its running speed (m/s), a step (s), its take-off speed (m/s), gravity (m/s²)
HURDLE, BAR, TALL, LOW = 0.7, 1.2, 1.8, 1.0         # a hurdle's top, a bar's underside, its height standing and ducking, m
COURSE = ${num(W.course)}                                     # m`), '', '',
      ...h.head(rows(String.raw`class World:
    """A runner at ${n1(W.run)} m/s down a ${n1(W.course)} m course of hurdles to jump (0.7 m high)${bars ? ' and bars to duck under (their underside\n    1.2 m up)' : ''}. A jump leaves the ground at ${n1(J)} m/s against 15 m/s² of gravity: ${(2 * J / 15).toFixed(2)} s in the air, ${above ? `its feet above
    a hurdle's height for about ${n1(above)} m, so it has to take off at the right moment` : `never as high
    as a hurdle's top (that needs a take-off of 4.6 m/s), so it cannot clear one`}. Ducking keeps its head down at 1 m
    while it holds the move, and a moment longer. Touching anything ends the run.
    ${pays}"""
    actions = ["run", "jump", "duck"]
    max_steps = ${W.maxSteps}`)), '',
      ...rows(String.raw`    def reset(self):
        self.obst, x = [], 14 + random.uniform(0, 4)   # the obstacles, (where, a bar?), 7 to 14 m apart
        while x < COURSE - 4:
            self.obst.append((x, ${bars ? 'random.random() < 0.4))   # two in five a bar' : 'False))'}
            x += random.uniform(7, 14)
        self.x, self.y, self.vy = 0.0, 0.0, 0.0    # how far it has run, its feet off the ground (m), its speed up (m/s)
        self.duck, self.t, self.end, self.jumps, self.ducking, self.hit = 0, 0, None, 0, 0, None
        return self.obs()

    def next_of(self, x):                          # the first obstacle not yet passed from x
        i = 0
        while i < len(self.obst) and self.obst[i][0] < x - 0.3:
            i += 1
        return i
`),
      ...h.sense(rows(String.raw`    def obs(self):                                 # the next two obstacles (how far, of 15 m; which kind), its height, speed up, ducking, on the ground
        i = self.next_of(self.x)
        a, b = (self.obst[i:i + 2] + [None, None])[:2]
        near = lambda q: min(max((q[0] - self.x) / 15, 0.0), 1.0) if q else 1.0
        kind = lambda q, bar: float(q is not None and q[1] == bar)
        return torch.tensor([near(a), kind(a, False), kind(a, True), near(b), kind(b, False), self.y / 1.5, self.vy / JUMP, float(self.duck > 0), float(self.y <= 0)])`)), '',
      ...rows(String.raw`    def step(self, a):                             # 0.05 s
        ground, jumped, ahead = self.y <= 0 and self.vy <= 0, 0, self.next_of(self.x)
        if a == 1 and ground:                      # a jump
            self.vy, jumped = JUMP, 1
            self.jumps += 1
        if a == 2 and ground:                      # ducking: its head down while it holds the move, and 3 steps longer
            self.duck = 4
        elif self.duck > 0:
            self.duck -= 1
        x0, y0 = self.x, self.y
        if self.y > 0 or self.vy > 0:              # in the air
            self.y += self.vy * DT
            self.vy -= GRAV * DT
            if self.y <= 0:
                self.y, self.vy = 0.0, 0.0
        self.x += RUN * DT
        self.t += 1
        if self.duck > 0:
            self.ducking += 1
        # touching an obstacle it passed this step (none is skipped however fast it runs), at its height as it passed:
        # a hurdle with its feet below the top, a bar with its head above the underside
        for q in self.obst:
            if x0 < q[0] <= self.x:
                y = y0 + (q[0] - x0) / (self.x - x0) * (self.y - y0)
                head = y + (LOW if self.duck > 0 else TALL)
                if (head > BAR) if q[1] else (y < HURDLE):
                    self.hit = q
                    break
        hit = self.hit is not None
        finished = not hit and self.x >= COURSE
        done = hit or finished or self.t >= self.max_steps
        if done:
            self.end = ("barhit" if self.hit[1] else "hurdlehit") if hit else "course" if finished else "time"
        nxt = self.next_of(self.x)                 # the next obstacle
        cleared, air = not hit and nxt > ahead, self.y > 0   # an obstacle passed untouched this step; its feet off the ground`),
      ...reward, '        return self.obs(), r, done', '',
      ...rows(String.raw`    def behavior(self):                            # its style: how often it jumps (per obstacle passed), how much of the time it ducks
        passed = max(1, self.next_of(self.x))
        return min(max(self.jumps / passed / 2, 0.0), 1.0), min(max(self.ducking / max(1, self.t), 0.0), 1.0)

    def score(self, policy):
        """The share of the course its best moves run, over 5 courses (the same 5 every time)."""
        total = 0.0`),
      ...h.SEEDED(4242, ['for _ in range(5):', ...h.PLAY('    '), '    total += min(max(self.x / COURSE, 0.0), 1.0)']).map((l) => '        ' + l), '        return total / 5'];
  };

  // ------------------------------------------------------------------ a lamp to remember
  PW.worlds.memory = function (d, h) {
    const W = make(d), rw = h.rw, num = h.num, say = h.say, hist = W.view === 'history', H = NA.rlEnvs.make('memory', { view: 'history' }).input.len;
    const reward = h.f ? ['        lost, junction, pos, t = float(lost), float(self.pos >= LENGTH), self.pos, self.t',
      '        early, waited, lamp = float(early), float(self.waits), 1.0 if self.side else -1.0', ...h.ownLine]
      : [`        r = ${terms([rw.rRight ? `(${num(rw.rRight)} if right else 0.0)` : null, rw.rWrong ? `(${num(rw.rWrong)} if wrong or lost else 0.0)` : null, rw.rStep ? num(rw.rStep) : null,
        rw.rWait ? `(${num(rw.rWait)} if stood else 0.0)` : null, rw.rEarly ? `(${num(rw.rEarly)} if early else 0.0)` : null])}`];
    const pays = h.pays(`${say(rw.rRight)} for turning the way the lamp said, ${say(rw.rWrong)} for the other way or for standing at the junction ${W.wait} steps${rw.rStep ? `, ${say(rw.rStep)} every step` : ''}${rw.rWait ? `, ${say(rw.rWait)} for each step it waits at the junction` : ''}${rw.rEarly ? `, ${say(rw.rEarly)} for each turn tried before the junction` : ''}.`);
    // the last readings, with inputs left out: the readings as one list of numbers, then those it keeps
    const sense = (ls) => h.sense(ls).map((l) => l.replace('self.raw_obs()[self.KEEP]', 'self.raw_obs().flatten()[self.KEEP]'));
    return [
      note(`LENGTH, LIT, WAIT = ${W.len}, ${W.lit}, ${W.wait}`, 'squares down the corridor; the steps the lamp shows; the steps it has at the junction to turn'),
      note('MAX_STEPS = LENGTH + WAIT', 'the longest episode'), ...(hist ? [note(`HIST = ${H}`, `the readings it is given (the last ${H})`)] : []), '', '',
      ...h.head(rows(String.raw`class World:
    """A lamp at the start of a corridor shows which way to go at its end, left or right, ${W.lit === 1 ? 'at the first step only' : `for the first ${W.lit} steps`}; then
    it goes dark. The robot walks down the corridor a square a step by itself (what it does there makes no
    difference); at the end, a junction, it has to turn. ${hist ? `It is given the last ${H} readings, so a memory (a recurrent
    layer) can carry the lamp down the corridor.` : `It sees only the present: by the junction the lamp is
    long dark, so whatever it learns, half its turns are wrong.`}
    ${pays}"""
    actions = ["walk on", "turn left", "turn right"]`)), '',
      ...rows(String.raw`    def reset(self, side=None):                    # side: which way the lamp says, 0 left, 1 right (at random when not given)
        self.side = random.randrange(2) if side is None else side
        self.pos, self.t, self.end, self.choice, self.waits = 0, 0, None, None, 0${hist ? '\n        self.past = []\n        self.see()' : ''}
        return self.obs()

    def reading(self):                             # the lamp (lit for the first LIT steps), at the junction, how far along
        lit = self.t < LIT
        return [float(lit and self.side == 0), float(lit and self.side == 1), float(self.pos >= LENGTH), self.pos / LENGTH]
`),
      ...(hist ? [...rows(String.raw`    def see(self):                                 # the readings so far, the last ${H} of them
        self.past.append(self.reading())
        del self.past[:-HIST]
`), ...sense(rows(String.raw`    def obs(self):                                 # (${H}, 4): the last ${H} readings, oldest first (rows of 0 before the start)
        return torch.tensor([[0.0] * 4] * (HIST - len(self.past)) + self.past)`))]
        : h.sense(rows(String.raw`    def obs(self):                                 # what it sees now: 4 numbers
        return torch.tensor(self.reading())`))), '',
      ...rows(String.raw`    def step(self, a):
        right = wrong = early = stood = 0          # early: a turn tried in the corridor; stood: a step at the junction without turning
        if self.pos < LENGTH:                      # walking down the corridor, whatever it does (a turn does nothing)
            self.pos += 1
            early = int(a in (1, 2))
        elif a in (1, 2):                          # a turn at the junction
            self.choice = a - 1                    # 0: left, 1: right
            right, wrong = int(self.choice == self.side), int(self.choice != self.side)
        else:                                      # standing at the junction
            stood = 1
            self.waits += 1
        self.t += 1
        lost = not right and not wrong and self.t >= MAX_STEPS   # it stood at the junction too long
        if right or wrong or lost:
            self.end = "right" if right else "wrong" if wrong else "lost"${hist ? '\n        self.see()' : ''}`),
      ...reward, '        return self.obs(), r, self.end is not None', '',
      ...rows(String.raw`    def behavior(self):                            # its style: which way it turned (0.5: it did not), how long it took
        return (0.5 if self.choice is None else float(self.choice)), min(max(self.t / MAX_STEPS, 0.0), 1.0)

    def score(self, policy):
        """The share of 20 corridors its best moves turn the right way in, the lamp left and right in turn."""
        right = 0
        for k in range(20):
            o, done = self.reset(side=k % 2), False
            while not done:
                o, _, done = self.step(policy(o))
            right += self.end == "right"
        return right / 20`)];
  };

  // ------------------------------------------------------------------ the mountain cart
  PW.worlds.mountain = function (d, h) {
    const W = make(d), rw = h.rw, num = h.num, say = h.say;
    const reward = h.f ? ['        goal, x, speed, height, energy, push, t = float(goal), self.x, self.v * 100, math.sin(3 * self.x), phi, self.push, self.t',
      '        pushed, farthest, wall = float(self.push != 0), self.hi, float(wall)', ...h.ownLine]
      : [`        r = ${terms([rw.rTick ? num(rw.rTick) : null, rw.rFlag ? `(${num(rw.rFlag)} if goal else 0.0)` : null, rw.rEnergy ? `${num(rw.rEnergy)} * gained` : null,
        rw.rPush ? `(${num(rw.rPush)} if self.push else 0.0)` : null, rw.rHeight ? `${num(rw.rHeight)} * math.sin(3 * self.x)` : null])}`];
    const pays = h.pays(`${say(rw.rTick)} every step until it reaches the flag${rw.rFlag ? `, ${say(rw.rFlag)} on reaching it` : ''}${rw.rEnergy ? `, ${say(rw.rEnergy)} × the energy it gains every step (shaping)` : ''}${rw.rPush ? `, ${say(rw.rPush)} for each push of the engine` : ''}${rw.rHeight ? `, ${say(rw.rHeight)} × its height (−1 to 1) every step` : ''}.`);
    return [
      note(`FLAG, POWER, MAX_STEPS = 0.5, ${num(W.power)}, ${W.maxSteps}`, `where the flag is; the engine's push (${d.ws.engine === 'strong' ? 'strong' : 'normal'}); the longest episode`), '', '',
      ...h.head(rows(String.raw`class World:
    """A cart in a valley (the "mountain car" of Moore, 1990): its engine is too weak to drive straight up the slope on
    the right to the flag, so it has to rock, first up the slope behind it to gain speed.${W.anywhere ? ` Each episode starts at a
    random place and speed instead of at rest in the bottom (exploring starts).` : ''}
    ${pays}"""
    actions = ["push left", "coast", "push right"]`)), '',
      ...rows(String.raw`    def energy(self):                              # height plus speed, as energy: 0 at rest in the bottom, about 1.7 at rest at the flag
        return 1000 * (0.5 * self.v * self.v + 0.0025 / 3 * (math.sin(3 * self.x) + 1))

    def reset(self):
        ${W.anywhere ? 'self.x, self.v = random.uniform(-1.2, 0.45), random.uniform(-0.07, 0.07)   # anywhere: its position (the bottom is at -0.52) and speed' : 'self.x, self.v = random.uniform(-0.6, -0.4), 0.0   # its position (the valley\'s bottom is at -0.52) and speed'}
        self.t, self.end, self.push = 0, None, 0
        self.lo = self.hi = self.x                 # how far left and right it has been
        self.phi = self.energy()
        return self.obs()
`),
      ...h.sense(rows(String.raw`    def obs(self):                                 # its position and speed, scaled to about ±1
        return torch.tensor([(self.x + 0.3) / 0.9, self.v / 0.07])`)), '',
      ...rows(String.raw`    def step(self, a):
        self.push = -1 if a == 0 else 1 if a == 2 else 0
        self.v = min(max(self.v + self.push * POWER - 0.0025 * math.cos(3 * self.x), -0.07), 0.07)   # the engine against the slope
        self.x += self.v
        wall = self.x < -1.2
        if wall:                                   # the wall on the far left stops it
            self.x, self.v = -1.2, 0.0
        self.t += 1
        self.lo, self.hi = min(self.lo, self.x), max(self.hi, self.x)
        goal = self.x >= FLAG
        done = goal or self.t >= MAX_STEPS
        if done:
            self.end = "flag" if goal else "lost"
        phi = self.energy()
        gained, self.phi = phi - self.phi, phi     # the energy it gained this step`),
      ...reward, '        return self.obs(), r, done', '',
      ...rows(String.raw`    def behavior(self):                            # its style: how far left it went, how far right
        return min(max((self.lo + 1.2) / 0.7, 0.0), 1.0), min(max((self.hi + 1.2) / 1.8, 0.0), 1.0)

    def score(self, policy):
        """The share of 10 starts its best moves reach the flag from (the same 10 every time)."""
        reached = 0`),
      ...h.SEEDED(4242, ['for _ in range(10):', ...h.PLAY('    '), '    reached += self.end == "flag"']).map((l) => '        ' + l), '        return reached / 10'];
  };

  // ------------------------------------------------------------------ a pendulum to swing up
  PW.worlds.swingup = function (d, h) {
    const W = make(d), rw = h.rw, num = h.num, say = h.say;
    const reward = h.f ? ['        angle, spin, upright, t = self.th / DEG, self.w / DEG, float(upright), self.t', '        energy, switched = self.energy(), float(switched)', ...h.ownLine]
      : [`        r = ${terms([rw.rHeight ? `${num(rw.rHeight)} * height` : null, rw.rUpright ? `(${num(rw.rUpright)} if upright else 0.0)` : null, rw.rPush ? `(${num(rw.rPush)} if self.tq else 0.0)` : null,
        rw.rSpin ? `${num(rw.rSpin)} * abs(self.w / DEG)` : null, rw.rSwitch ? `(${num(rw.rSwitch)} if switched else 0.0)` : null])}`];
    const pays = h.pays(`${say(rw.rHeight)} × the weight's height every step (1 straight up, -1 hanging)${rw.rUpright ? `, ${say(rw.rUpright)} every step within 15° of upright` : ''}${rw.rPush ? `, ${say(rw.rPush)} every step the motor pushes` : ''}${rw.rSpin ? `, ${say(rw.rSpin)} × how fast it turns (degrees a second) every step` : ''}${rw.rSwitch ? `, ${say(rw.rSwitch)} each time the motor pushes the other way` : ''}.`);
    // where it starts, and whether the motor can lift it straight up
    const start = { up: ['upright (give or take 11°) at the start: only the balancing is left to learn', 'random.uniform(-0.2, 0.2)', 'upright'],
      anywhere: ['at any angle at the start', 'random.uniform(-math.pi, math.pi)', 'anywhere'], down: ['hanging down at the start', 'math.pi + random.uniform(-0.2, 0.2)', 'hanging down'] }[W.start];
    const strong = W.torque >= W.g;
    return [
      note(`G, TORQUE, FRICTION = ${num(W.g)}, ${num(W.torque)}, ${num(W.fric)}`, `gravity; the motor's push (${d.ws.motor === 'weak' ? 'weak' : 'strong'}); friction`),
      note(`DT, MAX_STEPS = 0.05, ${W.maxSteps}`, 'a step (s); the longest episode'),
      'DEG = math.pi / 180', '', '', ...WRAP, '', '',
      ...h.head(rows(String.raw`class World:
    """A pendulum on a motor: a 1 m pole with its weight at the end, ${start[0]}. The motor is ${strong ? `strong enough
    to lift it straight up (${num(W.torque)} against the ${num(W.g)} the weight pulls with), so it can drive it up and balance it there` : `too weak
    to lift it straight up (${num(W.torque)} against the ${num(W.g)} the weight pulls with), so it has to swing it higher and higher,
    pushing with each swing, then catch it upside down and balance it there`}. ${W.maxSteps} steps of 0.05 s.
    ${pays}"""
    actions = ["turn left", "rest", "turn right"]`)), '',
      '    def reset(self):', note(`        self.th = ${start[1]}`, `its angle from upright (rad): ${start[2]}`),
      ...rows(String.raw`        self.w = random.uniform(-0.3, 0.3)         # how fast it turns, rad/s
        self.t, self.tq, self.end, self.up_late, self.first, self.spin, self.pushed = 0, 0.0, None, 0, None, 0.0, 0
        return self.obs()

    def energy(self):                              # its swing: 1 with just enough to reach the top, -1 hanging still
        return (0.5 * self.w * self.w + G * math.cos(self.th)) / G
`),
      ...h.sense(rows(String.raw`    def obs(self):                                 # its angle (sine, cosine) and how fast it turns
        return torch.tensor([math.sin(self.th), math.cos(self.th), self.w / 8])`)), '',
      ...rows(String.raw`    def step(self, a):
        self.tq = (-1 if a == 0 else 1 if a == 2 else 0) * TORQUE
        self.w = min(max(self.w + (G * math.sin(self.th) + self.tq - FRICTION * self.w) * DT, -12.0), 12.0)   # gravity, the motor and friction
        self.th = wrap(self.th + self.w * DT)
        self.t += 1
        upright = abs(self.th) < 15 * DEG
        if upright:
            if self.first is None:                 # the first time it is up
                self.first = self.t
            if self.t > MAX_STEPS / 2:
                self.up_late += 1                  # up in the second half: what the score counts
        self.spin += abs(self.w)
        done = self.t >= MAX_STEPS
        if done:
            self.end = "balanced" if self.up_late >= 0.8 * (MAX_STEPS / 2) else "swung" if self.first is not None else "hanging"
        height = math.cos(self.th)                 # the weight's height: 1 straight up, -1 hanging
        push = (self.tq > 0) - (self.tq < 0)       # which way the motor pushes: 1, -1, or 0 at rest
        switched = push != 0 and self.pushed != 0 and push != self.pushed   # the other way from its last push
        if push:
            self.pushed = push`),
      ...reward, '        return self.obs(), r, done', '',
      ...rows(String.raw`    def behavior(self):                            # its style: how soon it first gets it upright, how fast it swings
        return (1.0 if self.first is None else min(max(self.first / MAX_STEPS, 0.0), 1.0)), min(max(self.spin / max(1, self.t) / 8, 0.0), 1.0)

    def score(self, policy):
        """The share of the second half its best moves keep it within 15° of upright, over 3 starts (the same 3 every
        time)."""
        total = 0.0`),
      ...h.SEEDED(4242, ['for _ in range(3):', ...h.PLAY('    '), '    total += self.up_late / (MAX_STEPS / 2)']).map((l) => '        ' + l), '        return total / 3'];
  };

  // ------------------------------------------------------------------ a robot arm
  PW.worlds.arm = function (d, h) {
    const W = make(d), rw = h.rw, num = h.num, say = h.say;
    const reward = h.f ? ['        reached, dist, speed, elbow, t = float(reached), d, moved / DT, self.q2 / DEG, self.t',
      '        shoulder, bump, braked = self.q1 / DEG, float(bump), float(a == 4)', ...h.ownLine]
      : [`        r = ${terms([rw.rReach ? `(${num(rw.rReach)} if reached else 0.0)` : null, rw.rApproach ? `${num(rw.rApproach)} * closer` : null, rw.rStep ? num(rw.rStep) : null,
        rw.rPush ? `(${num(rw.rPush)} if a <= 3 else 0.0)` : null, rw.rBrake ? `(${num(rw.rBrake)} if a == 4 else 0.0)` : null, rw.rBump ? `(${num(rw.rBump)} if bump else 0.0)` : null])}`];
    const pays = h.pays(`${say(rw.rReach)} on reaching the target${rw.rApproach ? `, ${say(rw.rApproach)} for every metre the hand gets closer to it (shaping)` : ''}${rw.rStep ? `, ${say(rw.rStep)} every step` : ''}${rw.rPush ? `, ${say(rw.rPush)} for each push of a joint` : ''}${rw.rBrake ? `, ${say(rw.rBrake)} for each brake` : ''}${rw.rBump ? `, ${say(rw.rBump)} for hitting the bench or the end of a joint's swing` : ''}.`);
    const cm = String(+(W.reach * 100).toFixed(1));
    return [
      note(`L1, L2, DT, MAX_STEPS = 1.0, 0.8, 0.05, ${W.maxSteps}`, 'the upper arm and the forearm (m), a step (s), the longest episode'),
      note(`ACC, REACH = ${num(W.acc)}, ${num(W.reach)}`, 'how hard a push turns a joint (rad/s²); how near counts as reaching the target (m)'),
      'DEG = math.pi / 180', '', '',
      ...h.head(rows(String.raw`class World:
    """A two-joint arm on a bench (a 1 m upper arm, a 0.8 m forearm) and a target somewhere it can reach: it has to
    bring its hand onto the target and stop there (within ${cm} cm, slower than 0.6 m/s). Each step pushes one joint one
    way or the other, or brakes both; the joints carry on turning, slowed by friction. The shoulder sweeps above the
    bench, the elbow bends 150° either way, and the hand never goes below the bench.
    ${pays}"""
    actions = ["shoulder anticlockwise", "shoulder clockwise", "elbow anticlockwise", "elbow clockwise", "brake"]`)), '',
      ...rows(String.raw`    def hand(self, q1, q2):                        # where the hand is (m): across from the shoulder, up from the bench
        return L1 * math.cos(q1) + L2 * math.cos(q1 + q2), L1 * math.sin(q1) + L2 * math.sin(q1 + q2)

    def reset(self):
        self.q1, self.q2 = random.uniform(0.4, 2.7), random.uniform(-1.2, 1.2)   # the shoulder's and the elbow's angles (rad)
        self.w1, self.w2 = 0.0, 0.0                # and how fast they turn (rad/s)
        self.t, self.end, self.path, self.bend = 0, None, 0.0, 0.0
        R, A = random.uniform(0.6, 1.65), random.uniform(0.3, math.pi - 0.3)    # a new target, somewhere it can reach
        self.tx, self.ty = R * math.cos(A), R * math.sin(A)
        if self.hand(self.q1, self.q2)[1] < 0.05:  # no start with the hand in the bench
            self.q2 = 0.0
        self.hx, self.hy = self.hand(self.q1, self.q2)
        self.d = math.hypot(self.tx - self.hx, self.ty - self.hy)   # the hand's distance to the target
        return self.obs()
`),
      ...h.sense(rows(String.raw`    def obs(self):                                 # its angles (sine, cosine) and speeds, the target from the hand, the target
        return torch.tensor([math.sin(self.q1), math.cos(self.q1), math.sin(self.q2), math.cos(self.q2), self.w1 / 3, self.w2 / 3,
                             (self.tx - self.hx) / 2, (self.ty - self.hy) / 2, self.tx / 2, self.ty / 2])`)), '',
      ...rows(String.raw`    def step(self, a):
        if a == 0:                                 # a push: one joint, one way
            self.w1 += ACC * DT
        elif a == 1:
            self.w1 -= ACC * DT
        elif a == 2:
            self.w2 += ACC * DT
        elif a == 3:
            self.w2 -= ACC * DT
        else:                                      # the brake: both slow to half
            self.w1 *= 0.5
            self.w2 *= 0.5
        self.w1 *= 0.94                            # friction
        self.w2 *= 0.94
        q1, q2 = self.q1 + self.w1 * DT, self.q2 + self.w2 * DT
        bump = False                               # a joint at the end of its swing, or the hand stopped by the bench
        if q1 < 0 or q1 > math.pi:                 # the shoulder sweeps only above the bench
            q1, self.w1, bump = min(max(q1, 0.0), math.pi), 0.0, True
        if abs(q2) > 2.6:                          # the elbow bends 150° at most
            q2, self.w2, bump = math.copysign(2.6, q2), 0.0, True
        if self.hand(q1, q2)[1] < 0.02:            # the bench stops it
            q1, q2, self.w1, self.w2, bump = self.q1, self.q2, 0.0, 0.0, True
        self.q1, self.q2 = q1, q2
        hx, hy = self.hand(q1, q2)
        moved = math.hypot(hx - self.hx, hy - self.hy)
        d = math.hypot(self.tx - hx, self.ty - hy)
        closer = self.d - d                        # how much closer the hand got, m
        self.hx, self.hy, self.d = hx, hy, d
        self.path += moved
        self.t += 1
        self.bend += abs(self.q2)
        reached = d < REACH and moved / DT < 0.6
        done = reached or self.t >= MAX_STEPS
        if done:
            self.end = "reached" if reached else "lost"`),
      ...reward, '        return self.obs(), r, done', '',
      ...rows(String.raw`    def behavior(self):                            # its style: how bent it keeps its elbow, how far its hand travels
        n = max(1, self.t)
        return min(max(self.bend / n / 2.6, 0.0), 1.0), min(max(self.path / 4, 0.0), 1.0)

    def score(self, policy):
        """The share of 10 targets its best moves reach (the same 10 every time)."""
        reached = 0`),
      ...h.SEEDED(4242, ['for _ in range(10):', ...h.PLAY('    '), '    reached += self.end == "reached"']).map((l) => '        ' + l), '        return reached / 10'];
  };

  // ------------------------------------------------------------------ the deceptive maze
  PW.worlds.trap = function (d, h) {
    const W = make(d), rw = h.rw, num = h.num, say = h.say, M = W.maze, [sx, sy, sa] = M.start;
    const reward = h.f ? ['        goal, dist, x, y, t = float(goal), d, self.x, self.y, self.t',
      '        new, turned, visited, best = float(fresh), float(turn != 0), len(self.parts), self.best', ...h.ownLine]
      : [`        r = ${terms([rw.rGoal ? `(${num(rw.rGoal)} if goal else 0.0)` : null, rw.rStraight ? `${num(rw.rStraight)} * closer` : null, rw.rStep ? num(rw.rStep) : null,
        rw.rBump ? `(${num(rw.rBump)} if bump else 0.0)` : null, rw.rNew ? `(${num(rw.rNew)} if fresh else 0.0)` : null, rw.rTurn ? `(${num(rw.rTurn)} if turn else 0.0)` : null,
        rw.rRecord ? `${num(rw.rRecord)} * record` : null])}`];
    const pays = h.pays(`${say(rw.rGoal)} on reaching the goal, ${say(rw.rStraight)} for every 100 units it gets closer to it in a straight line${rw.rStep ? `, ${say(rw.rStep)} every step` : ''}${rw.rBump ? `, ${say(rw.rBump)} for walking into a wall` : ''}${rw.rNew ? `, ${say(rw.rNew)} for each part of the maze (a square of 20 × 20) it reaches for the first time (novelty)` : ''}${rw.rRecord ? `, ${say(rw.rRecord)} for every 100 units it gets nearer the goal than ever before` : ''}${rw.rTurn ? `, ${say(rw.rTurn)} for each turn` : ''}.`);
    const why = W.mazeId === 'zigzag' ? 'two walls zigzag across it, and the way round each first leads away from the goal'
      : 'the straight way runs into a wall just under the goal, and the way round first leads away from it';
    return [
      note(`WALLS = [${W.walls.map((w) => `(${w.join(', ')})`).join(', ')}]`, `(x1, y1, x2, y2): the box and the ${M.name} maze's walls`),
      note(`START, GOAL, MAX_STEPS = (${sx}, ${sy}, ${angle(sa, num)}), (${M.goal.join(', ')}), ${W.maxSteps}`, 'where it starts (x, y, heading; y points down), the goal, the longest episode'),
      note(`DEG, BOT, SPEED = math.pi / 180, 5, ${num(W.speed)}`, "a degree in radians; the robot's radius; how far a step forward takes it"),
      ...rows(String.raw`FINDERS = [a * DEG for a in (-90, -45, 0, 45, 90)]   # its 5 rangefinders' angles from straight ahead`), '', '', ...WRAP, '', '',
      ...rows(String.raw`def seg_dist(px, py, ax, ay, bx, by):              # the distance from (px, py) to the wall from (ax, ay) to (bx, by)
    vx, vy = bx - ax, by - ay
    t = min(max(((px - ax) * vx + (py - ay) * vy) / ((vx * vx + vy * vy) or 1), 0.0), 1.0)
    return math.hypot(px - ax - t * vx, py - ay - t * vy)


def ray(x, y, a, most=100.0):                      # how far a rangefinder at (x, y) pointing at angle a sees
    dx, dy, best = math.cos(a), math.sin(a), most
    for ax, ay, bx, by in WALLS:
        ex, ey = bx - ax, by - ay
        den = dx * ey - dy * ex
        if abs(den) < 1e-9:                        # parallel to this wall
            continue
        t = ((ax - x) * ey - (ay - y) * ex) / den  # how far along the ray it meets the wall's line
        u = ((ax - x) * dy - (ay - y) * dx) / den  # and where on the wall (0 to 1: on it)
        if t > 0 and 0 <= u <= 1 and t < best:
            best = t
    return best


def part_of(x, y):                                 # which of the maze's 10 × 10 parts, 20 units square, (x, y) is in
    return min(max(math.floor(x / 20), 0), 9) + 10 * min(max(math.floor(y / 20), 0), 9)`), '', '',
      ...h.head(rows(String.raw`class World:
    """A round robot in a maze seen from above, 200 units square, with 5 rangefinders (90° and 45° to each side and
    straight ahead, out to 100) and a radar of 4 quarters that says in which quarter around it the goal is, but not
    how far. Closer is a trap here: ${why}, so an agent that only ever gets closer ends up pressed against a wall (the
    deceptive maze of Lehman and Stanley's novelty search).
    ${pays}"""
    actions = ["forward", "turn left", "turn right"]`)), '',
      ...rows(String.raw`    def dist(self):                                # to the goal, in a straight line
        return math.hypot(GOAL[0] - self.x, GOAL[1] - self.y)

    def reset(self):
        self.x, self.y = START[0] + random.uniform(-4, 4), START[1] + random.uniform(-4, 4)
        self.th = START[2] + random.uniform(-0.3, 0.3)   # its heading
        self.t, self.end = 0, None
        self.d = self.best = self.dist()           # its distance to the goal, and its nearest yet
        self.parts = {part_of(self.x, self.y)}     # the parts of the maze it has been to
        self.see()
        return self.obs()

    def see(self):                                 # what its rangefinders read
        self.rays = [ray(self.x, self.y, self.th + a) for a in FINDERS]
`),
      ...h.sense(rows(String.raw`    def obs(self):                                 # its 5 rangefinders (of 100), then the goal's quarter: ahead, left, behind, right
        v = [r / 100 for r in self.rays] + [0.0] * 4
        rel = wrap(math.atan2(GOAL[1] - self.y, GOAL[0] - self.x) - self.th)   # the goal's angle from straight ahead (clockwise: y points down)
        v[5 + (0 if abs(rel) < math.pi / 4 else 1 if -3 * math.pi / 4 < rel < -math.pi / 4 else 2 if abs(rel) > 3 * math.pi / 4 else 3)] = 1.0
        return torch.tensor(v)`)), '',
      ...rows(String.raw`    def step(self, a):
        turn, speed = (-0.3 if a == 1 else 0.3 if a == 2 else 0.0), (SPEED if a == 0 else 1)   # forward: SPEED units; while turning, 1
        self.th = wrap(self.th + turn)
        nx, ny = self.x + math.cos(self.th) * speed, self.y + math.sin(self.th) * speed
        bump = 0
        if all(seg_dist(nx, ny, *w) >= BOT for w in WALLS):
            self.x, self.y = nx, ny
        else:                                      # a wall in the way: it stays put
            bump = 1
        self.t += 1
        self.see()
        d = self.dist()
        closer, goal = (self.d - d) / 100, d < 8   # how much closer it got (in hundreds); there
        record = max(0.0, self.best - d) / 100     # how much nearer than ever before (in hundreds)
        part = part_of(self.x, self.y)
        fresh = part not in self.parts             # a part of the maze it had not been to
        self.d, self.best = d, min(self.best, d)
        self.parts.add(part)
        done = goal or self.t >= MAX_STEPS
        if done:
            self.end = "goal" if goal else "lost"`),
      ...reward, '        return self.obs(), r, done', '',
      ...rows(String.raw`    def behavior(self):                            # its style: where it ended up, across and up the map
        return min(max(self.x / 200, 0.0), 1.0), min(max(1 - self.y / 200, 0.0), 1.0)

    def score(self, policy):
        """The share of 3 starts its best moves reach the goal from (the same 3 every time)."""
        reached = 0`),
      ...h.SEEDED(4242, ['for _ in range(3):', ...h.PLAY('    '), '    reached += self.end == "goal"']).map((l) => '        ' + l), '        return reached / 3'];
  };

  // ------------------------------------------------------------------ escaping a chaser
  PW.worlds.chase = function (d, h) {
    const W = make(d), rw = h.rw, num = h.num, say = h.say, ahead = W.lead > 0, fast = d.ws.speed === 'fast';
    const reward = h.f ? ['        caught, speed, t = float(caught), math.hypot(m.vx, m.vy) * 10, self.t',
      '        near, wall, pushed, coins, cdist = float(dist < 3), float(wall), float(ax != 0 or ay != 0), self.coins, math.hypot(self.coin[0] - m.x, self.coin[1] - m.y)', ...h.ownLine]
      : [`        r = ${terms([rw.rFree ? num(rw.rFree) : null, rw.rCoin ? `(${num(rw.rCoin)} if coin else 0.0)` : null, rw.rCaught ? `(${num(rw.rCaught)} if caught else 0.0)` : null,
        rw.rNear ? `(${num(rw.rNear)} if dist < 3 else 0.0)` : null, rw.rWall ? `(${num(rw.rWall)} if wall else 0.0)` : null, rw.rPush ? `(${num(rw.rPush)} if ax or ay else 0.0)` : null,
        rw.rDist ? `${num(rw.rDist)} * dist` : null])}`];
    const pays = h.pays(`${say(rw.rFree)} every step it is still free${rw.rCoin ? `, ${say(rw.rCoin)} a coin` : ''}, ${say(rw.rCaught)} when it is caught${rw.rNear ? `, ${say(rw.rNear)} every step a chaser is within 3 m` : ''}${rw.rDist ? `, ${say(rw.rDist)} × its distance from the nearest chaser (m) every step` : ''}${rw.rWall ? `, ${say(rw.rWall)} every step it touches a wall or a pillar` : ''}${rw.rPush ? `, ${say(rw.rPush)} for each push` : ''}.`);
    // each chaser heads for the runner, or (aim ahead) for where it will be a moment later
    const aim = ahead ? [`            lead = min(8, math.hypot(m.x - c.x, m.y - c.y) / CHASE_SPEED) * ${num(W.lead)}   # how far ahead it aims, steps`, '            gx, gy = m.x + m.vx * lead - c.x, m.y + m.vy * lead - c.y']
      : ['            gx, gy = m.x - c.x, m.y - c.y'];
    return [
      note('SIZE, PILLAR_R = 20.0, 1.4', "the arena, 20 m square (y points down), and the pillars' radius, m"),
      note(`PILLARS = [${W.pillars.map(([x, y]) => `(${x}, ${y})`).join(', ')}]`, `where the pillars are${W.pillars.length ? '' : ' (none in the open arena)'}`),
      note(`CHASERS, CHASE_SPEED, MAX_STEPS = ${W.k}, ${num(W.cs)}, ${W.maxSteps}`, `chasers; how far one goes a step (${fast ? 'fast' : 'slow'}; it runs 0.4 at most); the longest game`), '', '',
      ...rows(String.raw`class Body:                                        # where it is and how it moves (m, and m a step)
    def __init__(self, x, y):
        self.x, self.y, self.vx, self.vy = x, y, 0.0, 0.0`), '', '',
      ...h.head(rows(String.raw`class World:
    """An arena 20 m square, from above: it runs from ${W.k === 2 ? 'two chasers that head' : 'a chaser that heads'} ${ahead ? 'for where it is going, a moment ahead' : 'straight for it'}, ${fast ? 'nearly as fast as it' : 'a little slower than it'},
    and picks up the coins that turn up one at a time. Caught, the game ends. It moves with momentum: a push up, down,
    left or right each step, or none.${W.pillars.length ? ' Four pillars stand in the arena to dodge round.' : ''}
    ${pays}"""
    actions = ["up", "down", "left", "right", "stay"]`)), '',
      ...rows(String.raw`    def free(self, x, y, m):                       # a circle of radius m at (x, y) clear of the walls and the pillars
        return m < x < SIZE - m and m < y < SIZE - m and all(math.hypot(x - px, y - py) > PILLAR_R + m for px, py in PILLARS)

    def confine(self, o, m):                       # pushed back out of the walls and the pillars, losing the speed it had into them
        o.vx = max(0.0, o.vx) if o.x < m else min(0.0, o.vx) if o.x > SIZE - m else o.vx   # against a wall it stops,
        o.vy = max(0.0, o.vy) if o.y < m else min(0.0, o.vy) if o.y > SIZE - m else o.vy   # rather than pressing on
        o.x, o.y = min(max(o.x, m), SIZE - m), min(max(o.y, m), SIZE - m)
        for px, py in PILLARS:
            dx, dy = o.x - px, o.y - py
            d = math.hypot(dx, dy) or 1
            if d < PILLAR_R + m:
                o.x, o.y = px + dx / d * (PILLAR_R + m), py + dy / d * (PILLAR_R + m)
                vr = (o.vx * dx + o.vy * dy) / d   # its speed into the pillar: lost
                if vr < 0:
                    o.vx, o.vy = o.vx - vr * dx / d, o.vy - vr * dy / d

    def spot(self, near):                          # somewhere free for a coin, more than 6 m from near
        for _ in range(200):
            x, y = random.uniform(1.5, SIZE - 1.5), random.uniform(1.5, SIZE - 1.5)
            if self.free(x, y, 0.8) and math.hypot(x - near.x, y - near.y) > 6:
                return x, y
        return SIZE / 2, 2.0

    def reset(self):
        self.me = Body(10.0, 17.0)
        self.chasers = [Body(4.0 + 12 * i, 3.0) for i in range(CHASERS)]
        self.t, self.end, self.coins, self.sum_d = 0, None, 0, 0.0
        self.coin = self.spot(self.me)
        return self.obs()

    def nearest(self):                             # the nearest chaser, and how far it is
        best, bd = self.chasers[0], math.inf
        for c in self.chasers:
            d = math.hypot(c.x - self.me.x, c.y - self.me.y)
            if d < bd:
                best, bd = c, d
        return best, bd
`),
      ...h.sense(rows(String.raw`    def obs(self):
        """The nearest chaser from it (up is positive) and how it moves, the second chaser, its own speed, the coin,
        and how far it can go in 8 directions before a wall or a pillar (of 8 m): 18 numbers."""
        m, (c, _) = self.me, self.nearest()
        c2 = next((q for q in self.chasers if q is not c), None)
        v = [(c.x - m.x) / 10, (m.y - c.y) / 10, c.vx / 0.35, -c.vy / 0.35, 0.0, 0.0, m.vx / 0.4, -m.vy / 0.4, (self.coin[0] - m.x) / 10, (m.y - self.coin[1]) / 10]
        if c2 is not None:
            v[4], v[5] = (c2.x - m.x) / 10, (m.y - c2.y) / 10
        for k in range(8):                         # up, up-right, right, ...: free space, half a metre at a time
            a, d = -math.pi / 2 + k * math.pi / 4, 0.0
            while d < 8 and self.free(m.x + math.cos(a) * d, m.y + math.sin(a) * d, 0.3):
                d += 0.5
            v.append(d / 8)
        return torch.tensor(v)`)), '',
      ...rows(String.raw`    def step(self, a):
        m = self.me
        ax, ay = (-1 if a == 2 else 1 if a == 3 else 0), (-1 if a == 0 else 1 if a == 1 else 0)
        m.vx, m.vy = (m.vx + 0.08 * ax) * 0.85, (m.vy + 0.08 * ay) * 0.85   # a push, with momentum
        sp = math.hypot(m.vx, m.vy)
        if sp > 0.4:                               # its top speed: 0.4 m a step
            m.vx, m.vy = m.vx * (0.4 / sp), m.vy * (0.4 / sp)
        m.x, m.y = m.x + m.vx, m.y + m.vy
        wx, wy = m.x, m.y
        self.confine(m, 0.4)
        wall = m.x != wx or m.y != wy              # touching a wall or a pillar
        for c in self.chasers:                     # each heads ${ahead ? 'for where it will be, a moment ahead' : 'straight for it'}`), ...aim,
      ...rows(String.raw`            gl = math.hypot(gx, gy) or 1
            c.vx, c.vy = c.vx * 0.6 + gx / gl * CHASE_SPEED * 0.4, c.vy * 0.6 + gy / gl * CHASE_SPEED * 0.4
            c.x, c.y = c.x + c.vx, c.y + c.vy
            self.confine(c, 0.45)
        self.t += 1
        coin = 0
        if math.hypot(self.coin[0] - m.x, self.coin[1] - m.y) < 0.8:   # a coin: the next one turns up somewhere else
            coin = 1
            self.coins += 1
            self.coin = self.spot(m)
        _, dist = self.nearest()
        caught = dist < 0.8
        self.sum_d += dist
        done = caught or self.t >= MAX_STEPS
        if done:
            self.end = "tagged" if caught else "escaped"`),
      ...reward, '        return self.obs(), r, done', '',
      ...rows(String.raw`    def behavior(self):                            # its style: how far it keeps from the chaser, how many coins it picks up
        return min(max(self.sum_d / max(1, self.t) / 14, 0.0), 1.0), min(max(self.coins / 10, 0.0), 1.0)

    def score(self, policy):
        """The share of ${W.maxSteps} steps its best moves stay free, over 3 games (the same 3 every time)."""
        total = 0.0`),
      ...h.SEEDED(4242, ['for _ in range(3):', ...h.PLAY('    '), '    total += self.t / MAX_STEPS']).map((l) => '        ' + l), '        return total / 3'];
  };

  // ------------------------------------------------------------------ traffic lights
  PW.worlds.traffic = function (d, h) {
    const W = make(d), rw = h.rw, num = h.num, say = h.say;
    const early = W.rates(0), late = W.rates(W.maxSteps / 2), list = (r) => `[${r.map(num).join(', ')}]`, shifts = list(early) !== list(late);
    const reward = h.f ? ['        longest, green, t = self.wait_max(), self.phase, self.t',
      '        ns, ew, amber, since = len(self.q[0]) + len(self.q[1]), len(self.q[2]) + len(self.q[3]), float(self.amber > 0), self.since', ...h.ownLine]
      : [`        r = ${terms([rw.rWait ? `${num(rw.rWait)} * waiting` : null, rw.rSwitch ? `(${num(rw.rSwitch)} if switched else 0.0)` : null, rw.rPass ? `${num(rw.rPass)} * passed` : null,
        rw.rLate ? `${num(rw.rLate)} * late` : null, rw.rLong ? `${num(rw.rLong)} * self.wait_max()` : null])}`];
    const pays = h.pays(`${say(rw.rWait)} for every car waiting, every step${rw.rSwitch ? `, ${say(rw.rSwitch)} for each switch of the lights` : ''}${rw.rPass ? `, ${say(rw.rPass)} for each car through` : ''}${rw.rLate ? `, ${say(rw.rLate)} for each car through late (after more than 20 steps)` : ''}${rw.rLong ? `, ${say(rw.rLong)} × the longest wait at a red (steps) every step` : ''}.`);
    return [
      note(`MAX_STEPS, AMBER = ${W.maxSteps}, ${W.amberFor}`, 'the length of a day; how long the lights show amber; steps'), '', '',
      'def rates(t):                                      # the chance of a car arriving each step from the north, south, east and west',
      shifts ? `    return ${list(early)} if t < MAX_STEPS / 2 else ${list(late)}   # the north-south road busy first, then the east-west road` : `    return ${list(early)}`, '', '',
      ...h.head(rows(String.raw`class World:
    """A crossroads: cars arrive from the north, south, east and west at random, and wait in a queue for green. One of
    the two roads has green at a time; switching first shows amber for ${W.amberFor} step${W.amberFor === 1 ? '' : 's'}, when nobody moves. On green, each road
    lets a car through from each side every step.${shifts ? ' The north-south road is busy for the first half of the day and the\n    east-west road for the second, as at rush hours.' : ''}
    ${pays}"""
    actions = ["keep", "switch"]`)), '',
      ...rows(String.raw`    def reset(self):
        self.q = [[], [], [], []]                  # each queue: the steps its cars arrived at
        self.phase, self.amber, self.since, self.t, self.end = 0, 0, 0, 0, None   # phase 0: the north-south road has green
        self.arrived = self.served = self.prompt = self.switches = self.ns_green = 0
        return self.obs()

    def wait_max(self):                            # the longest wait at a red light, steps
        w = 0
        for l in range(4):
            if (l // 2 != self.phase or self.amber) and self.q[l]:
                w = max(w, self.t - self.q[l][0])
        return w
`),
      ...h.sense(rows(String.raw`    def obs(self):                                 # the 4 queues (of 15 cars), which road has green, amber, steps since a switch, the longest wait
        return torch.tensor([min(len(q), 30) / 15 for q in self.q] + [float(self.phase), float(self.amber > 0), min(self.since, 60) / 30, min(self.wait_max(), 60) / 30])`)), '',
      ...rows(String.raw`    def step(self, a):
        switched = 0
        if a == 1 and not self.amber and self.since >= 2:   # a switch: amber first
            self.amber, switched = AMBER, 1
            self.switches += 1
        rt = rates(self.t)
        for l in range(4):                         # arrivals
            if random.random() < rt[l]:
                self.q[l].append(self.t)
                self.arrived += 1
        passed = late = 0                          # cars through this step, and those of them late (after more than 20 steps)
        if self.amber:
            self.amber -= 1
            if not self.amber:                     # the other road's turn
                self.phase, self.since = 1 - self.phase, 0
        else:
            for l in ((2, 3) if self.phase else (0, 1)):   # a car from each side of the road with green
                if self.q[l]:
                    t0 = self.q[l].pop(0)
                    self.served += 1
                    if self.t - t0 <= 20:
                        self.prompt += 1
                    else:
                        late += 1
                    passed += 1
        self.since += 1
        self.t += 1
        if not self.phase and not self.amber:
            self.ns_green += 1
        waiting, done = sum(len(q) for q in self.q), self.t >= MAX_STEPS
        if done:
            self.end = "flowing" if self.prompt_share() >= 0.8 else "jammed"`),
      ...reward, '        return self.obs(), r, done', '',
      ...rows(String.raw`    def prompt_share(self):                        # cars through within 20 steps, of all that arrived (those still waiting longer count as late)
        late = sum(self.t - t0 > 20 for q in self.q for t0 in q)
        return self.prompt / max(1, self.served + late) if self.arrived else 1.0

    def behavior(self):                            # its style: how often it switches, how much of the time north-south has green
        n = max(1, self.t)
        return min(max(self.switches / n * 8, 0.0), 1.0), min(max(self.ns_green / n, 0.0), 1.0)

    def score(self, policy):
        """The share of cars its best moves get through within 20 steps of arriving, over 3 days (the same 3 every time)."""
        total = 0.0`),
      ...h.SEEDED(4242, ['for _ in range(3):', ...h.PLAY('    '), '    total += self.prompt_share()']).map((l) => '        ' + l), '        return total / 3'];
  };

  // ------------------------------------------------------------------ a walking creature
  PW.worlds.walker = function (d, h) {
    const W = make(d), rw = h.rw, num = h.num, say = h.say;
    const reward = h.f ? ['        dist, fell, height, tilt, t = x - self.x0, float(fell), self.cy(), self.tilt() / DEG, self.t',
      '        speed, feet, switched = gained * 15, float(self.p[2][1] < 0.02) + float(self.p[3][1] < 0.02), float(switched)', ...h.ownLine]
      : [`        r = ${terms([rw.rForward ? `${num(rw.rForward)} * gained` : null, rw.rFall ? `(${num(rw.rFall)} if fell else 0.0)` : null, rw.rStep ? num(rw.rStep) : null,
        rw.rTilt ? `${num(rw.rTilt)} * abs(self.tilt() / DEG)` : null, rw.rHeight ? `${num(rw.rHeight)} * self.cy()` : null, rw.rSwitch ? `(${num(rw.rSwitch)} if switched else 0.0)` : null])}`];
    const pays = h.pays(`${say(rw.rForward)} for every metre it walks right, ${say(rw.rFall)} for falling (a hip on the ground ends it)${rw.rStep ? `, ${say(rw.rStep)} every step` : ''}${rw.rTilt ? `, ${say(rw.rTilt)} × its body's tilt (degrees) every step` : ''}${rw.rHeight ? `, ${say(rw.rHeight)} × its body's height (m) every step` : ''}${rw.rSwitch ? `, ${say(rw.rSwitch)} each time it changes its muscle move` : ''}.`);
    const muscles = ['weak', 'strong'].includes(d.ws.muscles) ? d.ws.muscles : 'normal', ice = d.ws.ground === 'ice';
    return [
      ...rows(String.raw`BONES = [(0, 1, 0.8), (0, 2, 1.04), (1, 3, 1.04)]  # stiff, (point, point, length in m): the body between the hips, and the legs
MUSCLES = [(0, 3), (1, 2)]                         # each from a hip to the other foot
START = [(-0.4, 1.0), (0.4, 1.0), (-0.6, 0.0), (0.6, 0.0)]   # the points: back hip, front hip, back foot, front foot (x, y up), m
REST = [math.hypot(START[i][0] - START[j][0], START[i][1] - START[j][1]) for i, j in MUSCLES]   # the muscles' lengths at rest`),
      note(`G, GRIP = ${num(W.g)}, ${num(W.grip)}`, `gravity (m/s²); how hard its feet grip the ground (${ice ? 'ice' : 'grippy'})`),
      note(`MOVES = [${W.moves.map(([a, b]) => `(${num(a)}, ${num(b)})`).join(', ')}]`, `each move's muscle lengths, back and front, of their rest lengths (${muscles} muscles)`),
      note(`MAX_STEPS, DEG = ${W.maxSteps}, math.pi / 180`, 'the longest episode; a degree in radians'), '', '',
      ...h.head(rows(String.raw`class World:
    """A creature of springs, seen from the side: a stiff body (two hips 0.8 m apart), two stiff legs, and two muscles
    crossing from each hip to the other foot. Each step it sets both muscles short or long (four moves); the muscles
    pull toward their new length, which swings the legs. Its feet grip the ground${ice ? ' (ice: they slip easily)' : ''}; its body must not touch it.${W.g !== 9.8 ? ` Gravity
    here is ${num(W.g)} m/s² (the Earth's is 9.8).` : ''} A clock
    among its inputs (going round every 20 steps) lets a gait be a rhythm. One step is 1/15 s (4 steps of the
    physics), ${W.maxSteps} steps.
    ${pays}"""
    actions = ["back short, front long", "back long, front short", "both short", "both long"]`)), '',
      ...rows(String.raw`    def reset(self):
        self.p = [[x + random.uniform(-0.02, 0.02), y + random.uniform(0, 0.02)] for x, y in START]   # where each point is
        self.q = [p[:] for p in self.p]            # where each was a moment ago (Verlet: the difference is its speed)
        self.len, self.goal = REST[:], REST[:]     # the muscles' lengths now, and the lengths they pull toward
        self.t, self.end, self.sum_h, self.a = 0, None, 0.0, 3   # (a: its last move)
        self.x0 = self.last_x = self.cx()
        return self.obs()

    def cx(self):                                  # the body's middle, across
        return (self.p[0][0] + self.p[1][0]) / 2

    def cy(self):                                  # and its height
        return (self.p[0][1] + self.p[1][1]) / 2

    def tilt(self):                                # the body's tilt, rad
        return math.atan2(self.p[1][1] - self.p[0][1], self.p[1][0] - self.p[0][0])
`),
      ...h.sense(rows(String.raw`    def obs(self):
        """Its body's tilt (sine, cosine), height and speed, which feet are down, how long its muscles are, and a clock
        that goes round every 20 steps (sine, cosine): 11 numbers."""
        p, q, a, ph = self.p, self.q, self.tilt(), 2 * math.pi * self.t / 20
        vx, vy = (self.cx() - (q[0][0] + q[1][0]) / 2) * 60, (self.cy() - (q[0][1] + q[1][1]) / 2) * 60   # m/s
        return torch.tensor([math.sin(a), math.cos(a), self.cy() - 0.8, vx / 2, vy / 2, float(p[2][1] < 0.02), float(p[3][1] < 0.02),
                             (self.len[0] / REST[0] - 1) * 4, (self.len[1] / REST[1] - 1) * 4, math.sin(ph), math.cos(ph)])`)), '',
      ...rows(String.raw`    def hold(self, i, j, L, k):                    # points i and j pulled toward L apart (k: how firmly; 1 for a bone)
        a, b = self.p[i], self.p[j]
        dx, dy = b[0] - a[0], b[1] - a[1]
        d = math.hypot(dx, dy) or 1e-9
        c = (d - L) / d * 0.5 * k
        a[0] += dx * c
        a[1] += dy * c
        b[0] -= dx * c
        b[1] -= dy * c

    def physics(self):
        """1/60 s: Verlet with gravity, then the bones and muscles held at their lengths, then the ground."""
        g = G / 3600
        for p, q in zip(self.p, self.q):
            vx, vy = (p[0] - q[0]) * 0.995, (p[1] - q[1]) * 0.995   # its speed, a little damped
            q[0], q[1] = p[0], p[1]
            p[0] += vx
            p[1] += vy - g
        for m in range(2):                         # the muscles shorten or lengthen, 1.2 cm at most a moment
            self.len[m] += min(max(self.goal[m] - self.len[m], -0.012), 0.012)
        for _ in range(8):
            for i, j, L in BONES:
                self.hold(i, j, L, 1)
            for m, (i, j) in enumerate(MUSCLES):
                self.hold(i, j, self.len[m], 0.5)
        for i, (p, q) in enumerate(zip(self.p, self.q)):
            if p[1] < 0:                           # the ground: friction as strong as the foot presses, so the foot carrying the weight grips
                press, vt = -p[1], p[0] - q[0]
                friction = (GRIP if i >= 2 else 1) * press
                p[1] = 0.0
                q[1] = p[1] + (p[1] - q[1]) * 0.2
                q[0] = p[0] if abs(vt) <= friction else p[0] - (vt - (1 if vt > 0 else -1) * friction)

    def step(self, a):
        switched, self.a = a != self.a, a          # another move than the last
        s0, s1 = MOVES[a]                          # each muscle short or long
        self.goal = [REST[0] * s0, REST[1] * s1]
        for _ in range(4):
            self.physics()
        self.t += 1
        x = self.cx()
        gained, fell = x - self.last_x, self.p[0][1] < 0.1 or self.p[1][1] < 0.1   # how far right it got; a hip on the ground
        self.last_x = x
        self.sum_h += self.cy()
        done = fell or self.t >= MAX_STEPS
        if done:
            self.end = "toppled" if fell else "walked" if x - self.x0 >= 10 else "slow"`),
      ...reward, '        return self.obs(), r, done', '',
      ...rows(String.raw`    def behavior(self):                            # its style: how fast it walks, how high it carries its body
        n = max(1, self.t)
        return min(max((self.cx() - self.x0) / (n / 15) / 1.5 + 0.2, 0.0), 1.0), min(max((self.sum_h / n - 0.4) / 0.8, 0.0), 1.0)

    def score(self, policy):
        """The share of 10 m its best moves walk, over 2 starts (the same 2 every time)."""
        total = 0.0`),
      ...h.SEEDED(4242, ['for _ in range(2):', ...h.PLAY('    '), '    total += min(max((self.cx() - self.x0) / 10, 0.0), 1.0)']).map((l) => '        ' + l), '        return total / 2'];
  };

  // ------------------------------------------------------------------ herding sheep
  PW.worlds.herd = function (d, h) {
    const W = make(d), rw = h.rw, num = h.num, say = h.say;
    const reward = h.f ? ['        f, cx, cy = self.flock()                   # the sheep still out now',
      '        gate = math.hypot(cx - GATE[0], cy - GATE[1]) if f else 0.0',
      '        spread = sum(math.hypot(s.x - cx, s.y - cy) for s in f) / len(f) if f else 0.0',
      '        t, last = self.t, float(last)', ...h.ownLine]
      : [`        r = ${terms([rw.rPenned ? `${num(rw.rPenned)} * penned` : null, rw.rNearer ? `${num(rw.rNearer)} * nearer` : null, rw.rStep ? num(rw.rStep) : null,
        rw.rFlock ? `${num(rw.rFlock)} * dist` : null, rw.rRun ? `${num(rw.rRun)} * ran` : null, rw.rAll ? `(${num(rw.rAll)} if last else 0.0)` : null])}`];
    const pays = h.pays(`${say(rw.rPenned)} for each sheep penned${rw.rNearer ? `, ${say(rw.rNearer)} for every metre the sheep come nearer the gate (shaping)` : ''}${rw.rStep ? `, ${say(rw.rStep)} every step` : ''}${rw.rFlock ? `, ${say(rw.rFlock)} × the dog's distance from the flock (m) every step` : ''}${rw.rRun ? `, ${say(rw.rRun)} for every metre the dog runs` : ''}${rw.rAll ? `, ${say(rw.rAll)} on penning the last sheep` : ''}.`);
    const n2 = (v) => String(+(+v).toFixed(2)), nearest = (d.extra || []).includes('the nearest sheep');
    return [
      ...rows(String.raw`FIELD, GATE = 24.0, (19.0, 6.5)                    # the field, 24 m square (y points down); the pen's gate, the middle of its open side
PEN_X0, PEN_X1, PEN_Y = 16.0, 22.0, 6.0            # the pen: from 16 to 22 m across, the top 6 m`),
      note(`SHEEP, MAX_STEPS = ${W.n}, ${W.maxSteps}`, 'the sheep; the longest game'),
      note(`FLEE, DOG_SPEED = ${num(W.flee)}, ${num(W.dogV)}`, "how near the dog comes before the sheep run (m); the dog's top speed (m a step)"), '', '',
      ...rows(String.raw`class Animal:                                      # where it is and how it moves (m, and m a step)
    def __init__(self, x, y):
        self.x, self.y, self.vx, self.vy = x, y, 0.0, 0.0`), '', '',
      ...h.head(rows(String.raw`class World:
    """A field 24 m square from above, a pen in its top right corner (open at the bottom), ${W.n} sheep and the dog. Sheep
    graze and keep together, and run from the dog when it comes within ${n2(W.flee)} m; one that walks into the pen stays. The
    dog moves with momentum: a push up, down, left or right each step, or none.
    ${pays}"""
    actions = ["up", "down", "left", "right", "stay"]`)), '',
      ...rows(String.raw`    def reset(self):
        self.dog, self.sheep = Animal(2.0, 22.0), []
        for _ in range(SHEEP):                     # grazing in the bottom left, each wandering its own way
            s = Animal(random.uniform(4, 12), random.uniform(11, 20))
            s.penned, s.wx, s.wy = False, random.uniform(-1, 1), random.uniform(-1, 1)
            self.sheep.append(s)
        self.t, self.end, self.path, self.sum_d = 0, None, 0.0, 0.0
        self.far = self.spread()
        return self.obs()

    def flock(self):                               # the sheep still out, and their middle (the gate when there are none)
        f = [s for s in self.sheep if not s.penned]
        if not f:
            return f, GATE[0], GATE[1]
        cx = cy = 0.0
        for s in f:
            cx, cy = cx + s.x, cy + s.y
        return f, cx / len(f), cy / len(f)
${nearest ? `
    def nearest_sheep(self):                       # the nearest sheep still out, from the dog, up positive (0, 0 when none is)
        best, bd = None, math.inf
        for s in self.sheep:
            dd = math.hypot(s.x - self.dog.x, s.y - self.dog.y)
            if not s.penned and dd < bd:
                best, bd = s, dd
        return ((best.x - self.dog.x) / 12, (self.dog.y - best.y) / 12) if best else (0.0, 0.0)
` : ''}
    def spread(self):                              # the sheep still out: their distances to the gate, added up
        total = 0.0
        for s in self.sheep:
            if not s.penned:
                total += math.hypot(s.x - GATE[0], s.y - GATE[1])
        return total
`),
      ...h.sense(rows(String.raw`    def obs(self):
        """The dog, the flock's middle and the straggler (the sheep farthest from the gate) from the dog, the gate from
        the flock, its speed, the share penned and how spread the flock is: 12 numbers, up positive, about ±1."""
        d, (f, cx, cy) = self.dog, self.flock()
        sx, sy, far = cx, cy, -1.0
        for s in f:
            dd = math.hypot(s.x - GATE[0], s.y - GATE[1])
            if dd > far:
                sx, sy, far = s.x, s.y, dd
        spr = 0.0
        for s in f:
            spr += math.hypot(s.x - cx, s.y - cy)
        spr = spr / len(f) if f else 0.0
        return torch.tensor([d.x / 12 - 1, 1 - d.y / 12, (cx - d.x) / 12, (d.y - cy) / 12, (sx - d.x) / 12, (d.y - sy) / 12,
                             (GATE[0] - cx) / 12, (cy - GATE[1]) / 12, d.vx / DOG_SPEED, -d.vy / DOG_SPEED, 1 - len(f) / SHEEP, spr / 6])`)), '',
      ...rows(String.raw`    def step(self, a):
        d, ox, oy = self.dog, self.dog.x, self.dog.y
        ax, ay = (-1 if a == 2 else 1 if a == 3 else 0), (-1 if a == 0 else 1 if a == 1 else 0)
        d.vx, d.vy = (d.vx + 0.09 * ax) * 0.85, (d.vy + 0.09 * ay) * 0.85   # a push, with momentum
        sp = math.hypot(d.vx, d.vy)
        if sp > DOG_SPEED:                         # its top speed
            d.vx, d.vy = d.vx * (DOG_SPEED / sp), d.vy * (DOG_SPEED / sp)
        d.x, d.y = min(max(d.x + d.vx, 0.4), FIELD - 0.4), min(max(d.y + d.vy, 0.4), FIELD - 0.4)
        d.vx = max(0.0, d.vx) if d.x <= 0.4 else min(0.0, d.vx) if d.x >= FIELD - 0.4 else d.vx   # at the fence it stops,
        d.vy = max(0.0, d.vy) if d.y <= 0.4 else min(0.0, d.vy) if d.y >= FIELD - 0.4 else d.vy   # rather than pressing on
        ran = math.hypot(d.x - ox, d.y - oy)       # how far it ran, m
        self.path += ran
        f, cx, cy = self.flock()
        penned = 0
        for s in f:                                # each sheep in turn
            fx = fy = 0.0
            dd = math.hypot(s.x - d.x, s.y - d.y) or 1e-6
            if dd < FLEE:                          # away from the dog, the harder the nearer it is
                fx += (s.x - d.x) / dd * 0.13 * (1 - dd / FLEE)
                fy += (s.y - d.y) / dd * 0.13 * (1 - dd / FLEE)
            dc = math.hypot(cx - s.x, cy - s.y)
            if dc > 1.5:                           # toward the flock's middle
                fx += (cx - s.x) / dc * 0.012
                fy += (cy - s.y) / dc * 0.012
            for o in f:                            # not too close to the others
                if o is not s:
                    dx, dy = s.x - o.x, s.y - o.y
                    dn = math.hypot(dx, dy)
                    if 0 < dn < 1:
                        fx += dx / dn * 0.04
                        fy += dy / dn * 0.04
            if random.random() < 0.05:             # now and then a new way to wander
                s.wx, s.wy = random.uniform(-1, 1), random.uniform(-1, 1)
            fx += s.wx * 0.006
            fy += s.wy * 0.006
            s.vx, s.vy = (s.vx + fx) * 0.8, (s.vy + fy) * 0.8
            v = math.hypot(s.vx, s.vy)
            if v > 0.32:                           # a sheep's top speed
                s.vx, s.vy = s.vx * (0.32 / v), s.vy * (0.32 / v)
            nx, ny = min(max(s.x + s.vx, 0.4), FIELD - 0.4), min(max(s.y + s.vy, 0.4), FIELD - 0.4)
            if ny < PEN_Y and ((s.x <= PEN_X0) != (nx <= PEN_X0) or (s.x >= PEN_X1) != (nx >= PEN_X1)):
                nx = s.x                           # the pen's sides are fences: only its gate lets a sheep in
            s.x, s.y = nx, ny
            if PEN_X0 < s.x < PEN_X1 and s.y < PEN_Y:   # in the pen: it stays
                s.penned, s.vx, s.vy = True, 0.0, 0.0
                penned += 1
        self.t += 1
        now = self.spread()
        nearer, self.far = self.far - now, now     # how much nearer the gate the sheep got, m
        left = sum(not s.penned for s in self.sheep)
        dist = math.hypot(cx - d.x, cy - d.y)      # the dog's distance to the flock
        self.sum_d += dist
        done = not left or self.t >= MAX_STEPS
        if done:
            self.end = "loose" if left else "penned"
        last = not left and penned > 0             # the last sheep penned`),
      ...reward, '        return self.obs(), r, done', '',
      ...rows(String.raw`    def behavior(self):                            # its style: how close it keeps to the flock, how far it runs
        n = max(1, self.t)
        return min(max(self.sum_d / n / 12, 0.0), 1.0), min(max(self.path / (n * 0.45), 0.0), 1.0)

    def score(self, policy):
        """The share of the sheep its best moves pen, over 3 games (the same 3 every time)."""
        total = 0.0`),
      ...h.SEEDED(4242, ['for _ in range(3):', ...h.PLAY('    '), '    total += sum(s.penned for s in self.sheep) / SHEEP']).map((l) => '        ' + l), '        return total / 3'];
  };
})(typeof globalThis !== 'undefined' && globalThis.NA ? globalThis.NA : window.NA);
