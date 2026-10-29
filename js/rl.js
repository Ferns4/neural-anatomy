/* rl.js — reinforcement learning, small enough to watch learn: the environments and agents of the RL lab.
   A maze, learned with a table of Q-values (Q-learning). And two board games made for this site, learned by
   playing against themselves: Three-line (a 4 × 4 board, three in a row wins) with a table holding a value for
   every position it has met, and Four-line (a 5 × 5 board, four in a row wins) with a small neural network,
   since there are far too many positions for a table. No rewards are given for anything but the result. */
(function (NA) {
  'use strict';
  const RL = {};
  // a small seeded random generator (xorshift32), so a run can be repeated exactly
  RL.rand = function (seed) {
    let s = (seed | 0) || 0x2545f491;
    return () => { s ^= s << 13; s ^= s >>> 17; s ^= s << 5; return (s >>> 0) / 4294967296; };
  };

  // ------------------------------------------------------------------ the maze
  /* # wall, X pit (−1, and the episode ends), G goal (+1, ends), S start; every other step costs 0.02, so shorter
     is better. The short way runs between pits; a longer way, down the right, passes none. A slippery floor sends
     a move sideways now and then, and then the short way is a gamble. */
  const MAZE = [
    'S...#...',
    '.##.#.#.',
    '..#...#.',
    '#.#X.X#.',
    '...X.X#.',
    '.##....G',
  ];
  const DIRS = [[0, -1], [1, 0], [0, 1], [-1, 0]], DIR_NAMES = ['up', 'right', 'down', 'left'];
  class Maze {
    constructor(rows = MAZE) {
      this.rows = rows; this.H = rows.length; this.W = rows[0].length; this.n = this.W * this.H;
      this.cell = rows.join('').split('');
      this.start = this.cell.indexOf('S'); this.goal = this.cell.indexOf('G');
      this.stepCost = -0.02; this.slip = 0; this.maxSteps = 100;
    }
    // where action a leads from s (into a wall or off the board: nowhere)
    move(s, a) {
      const x = (s % this.W) + DIRS[a][0], y = Math.floor(s / this.W) + DIRS[a][1];
      if (x < 0 || y < 0 || x >= this.W || y >= this.H || this.cell[y * this.W + x] === '#') return s;
      return y * this.W + x;
    }
    step(s, a, rng) {
      const slipped = this.slip > 0 && rng() < this.slip;
      if (slipped) a = (a + (rng() < 0.5 ? 1 : 3)) % 4;
      const s2 = this.move(s, a), c = this.cell[s2];
      return { s: s2, a, slipped, r: c === 'G' ? 1 : c === 'X' ? -1 : this.stepCost, done: c === 'G' || c === 'X' };
    }
    open(s) { return this.cell[s] !== '#'; }
  }

  // Q(s, a) for every square and move, learned by Q-learning
  class QTable {
    constructor(nS, nA) { this.nS = nS; this.nA = nA; this.Q = new Float64Array(nS * nA); this.seen = new Uint32Array(nS); }
    max(s) { let v = -Infinity; for (let a = 0; a < this.nA; a++) v = Math.max(v, this.Q[s * this.nA + a]); return v; }
    // the best move (ties broken at random when rng is given, so equal values do not always pick "up")
    best(s, rng) {
      let best = [], v = -Infinity;
      for (let a = 0; a < this.nA; a++) { const q = this.Q[s * this.nA + a]; if (q > v + 1e-12) { v = q; best = [a]; } else if (Math.abs(q - v) <= 1e-12) best.push(a); }
      return rng ? best[Math.floor(rng() * best.length)] : best[0];
    }
    act(s, eps, rng) { return rng() < eps ? Math.floor(rng() * this.nA) : this.best(s, rng); }
    // Q(s, a) ← Q(s, a) + α·(r + γ·max Q(s′, ·) − Q(s, a)); returns the TD error
    update(s, a, r, s2, done, alpha, gamma) {
      const i = s * this.nA + a, target = r + (done ? 0 : gamma * this.max(s2)), err = target - this.Q[i];
      this.Q[i] += alpha * err; this.seen[s]++;
      return err;
    }
  }
  // one episode of Q-learning: its return, length, how it ended, and (if asked) the squares it visited
  RL.mazeEpisode = function (maze, Q, o, rng, keepPath) {
    let s = maze.start, total = 0, steps = 0, end = 'timeout';
    const path = keepPath ? [s] : null;
    while (steps < maze.maxSteps) {
      const a = Q.act(s, o.eps, rng), t = maze.step(s, a, rng);
      if (o.learn !== false) Q.update(s, a, t.r, t.s, t.done, o.alpha, o.gamma);
      total += t.r; steps++; s = t.s;
      if (path) path.push(s);
      if (t.done) { end = t.r > 0 ? 'goal' : 'pit'; break; }
    }
    return { total, steps, end, path };
  };
  // how the greedy policy does (no exploring, no learning): the share of episodes that reach the goal, and their length
  RL.mazeEval = function (maze, Q, rng, n = 40) {
    let goal = 0, steps = 0;
    for (let k = 0; k < n; k++) { const e = RL.mazeEpisode(maze, Q, { eps: 0, learn: false }, rng); if (e.end === 'goal') { goal++; steps += e.steps; } }
    return { success: goal / n, steps: goal ? steps / goal : NaN };
  };

  // ------------------------------------------------------------------ board games: k in a row
  /* Players take turns placing a stone on an empty cell; k in a row (across, down or diagonally) wins, and a full
     board is a draw. Cells are numbered row by row; a board is an Int8Array: 0 empty, 1 and 2 the players, 1 first. */
  class LineGame {
    constructor(id, name, size, k) {
      this.id = id; this.name = name; this.size = size; this.k = k; this.n = size * size;
      const lines = [];
      for (let r = 0; r < size; r++) for (let c = 0; c < size; c++) for (const [dr, dc] of [[0, 1], [1, 0], [1, 1], [1, -1]]) {
        const er = r + dr * (k - 1), ec = c + dc * (k - 1);
        if (er < 0 || er >= size || ec < 0 || ec >= size) continue;
        lines.push(Array.from({ length: k }, (_, t) => (r + dr * t) * size + c + dc * t));
      }
      this.lines = lines;
      this.through = Array.from({ length: this.n }, () => []);
      lines.forEach((L, li) => L.forEach((c) => this.through[c].push(li)));
      // the board's 8 symmetries (4 rotations, each mirrored): perm[t][i] is where cell i goes
      const S = size, at = (r, c) => r * S + c;
      const maps = [(r, c) => at(r, c), (r, c) => at(c, S - 1 - r), (r, c) => at(S - 1 - r, S - 1 - c), (r, c) => at(S - 1 - c, r),
        (r, c) => at(r, S - 1 - c), (r, c) => at(c, r), (r, c) => at(S - 1 - r, c), (r, c) => at(S - 1 - c, S - 1 - r)];
      this.perms = maps.map((f) => { const p = new Int16Array(this.n); for (let r = 0; r < S; r++) for (let c = 0; c < S; c++) p[at(r, c)] = f(r, c); return p; });
    }
    // the winning line through cell i (just played), or null
    wins(b, i) {
      const p = b[i];
      for (const li of this.through[i]) { const L = this.lines[li]; let all = true; for (const c of L) if (b[c] !== p) { all = false; break; } if (all) return L; }
      return null;
    }
    empties(b) { const out = []; for (let i = 0; i < this.n; i++) if (!b[i]) out.push(i); return out; }
    transform(b, t) { const out = new Int8Array(this.n), p = this.perms[t]; for (let i = 0; i < this.n; i++) out[p[i]] = b[i]; return out; }
  }
  RL.games = { three: new LineGame('three', 'Three-line', 4, 3), four: new LineGame('four', 'Four-line', 5, 4) };

  /* Values of positions, always from the side of the player who has just moved: the chance that player goes on to
     win (a draw counts half). An agent picks its move by looking at the position each empty cell would lead to
     (its "afterstate") and taking the one it values most. */

  // a table with one value per position met, a position and its 7 rotations and mirror images sharing one entry
  class ValueTable {
    constructor(game) { this.game = game; this.V = new Map(); this.kind = 'table'; }
    key(b, mover) {
      const g = this.game, n = g.n;
      let best = null;
      const ch = new Array(n);
      for (const p of g.perms) {
        for (let i = 0; i < n; i++) ch[p[i]] = b[i] === 0 ? '.' : b[i] === mover ? 'x' : 'o';
        const s = ch.join('');
        if (best === null || s < best) best = s;
      }
      return best;
    }
    value(b, mover) { const v = this.V.get(this.key(b, mover)); return v === undefined ? 0.5 : v; }
    remember(b, mover) { return this.key(b, mover); }
    learn(key, target, alpha) { const v = this.V.has(key) ? this.V.get(key) : 0.5; this.V.set(key, v + alpha * (target - v)); }
    get size() { return this.V.size; }
  }

  /* A small network for the same job: the board seen from the mover's side (2 inputs per cell: a stone of theirs,
     a stone of the other player's) → a hidden layer of tanh neurons → one sigmoid output, the chance to win.
     W1 is stored input by input (W1[i·H + j]), so adding a stone adds one column of weights: the network can score
     every possible move from one sum. */
  class ValueNet {
    constructor(n, hidden = 64, seed = 7) {
      this.kind = 'net'; this.n = n; this.nIn = 2 * n; this.nH = hidden;
      const rng = RL.rand(seed), g = () => { let u = 0; for (let k = 0; k < 6; k++) u += rng(); return (u - 3) / Math.sqrt(0.5); };
      this.W1 = Float64Array.from({ length: this.nIn * hidden }, () => g() * Math.sqrt(1 / this.nIn));
      this.b1 = new Float64Array(hidden);
      this.W2 = Float64Array.from({ length: hidden }, () => g() * Math.sqrt(1 / hidden));
      this.b2 = 0;
      this.h = new Float64Array(hidden); this.z = new Float64Array(hidden);
    }
    get paramCount() { return this.nIn * this.nH + 2 * this.nH + 1; }
    // the hidden layer's sums for board b seen by mover
    pre(b, mover, out = new Float64Array(this.nH)) {
      const H = this.nH;
      out.set(this.b1);
      for (let i = 0; i < this.n; i++) {
        if (!b[i]) continue;
        const o = (b[i] === mover ? i : this.n + i) * H;
        for (let j = 0; j < H; j++) out[j] += this.W1[o + j];
      }
      return out;
    }
    fromPre(z) { let s = this.b2; for (let j = 0; j < this.nH; j++) s += this.W2[j] * Math.tanh(z[j]); return 1 / (1 + Math.exp(-s)); }
    value(b, mover) { return this.fromPre(this.pre(b, mover, this.z)); }
    remember(b, mover) { return { b: Int8Array.from(b), mover }; }
    /* One step of gradient descent on the cross-entropy between its output and the target (a later position's
       value, or the result): with a sigmoid output the error signal is simply output − target. Each step uses
       one of the board's 8 symmetries at random, so what it learns about a corner holds for every corner. */
    learn(tok, target, lr, game, rng) {
      const b = game && rng ? game.transform(tok.b, Math.floor(rng() * 8)) : tok.b, H = this.nH, z = this.pre(b, tok.mover, this.z), h = this.h;
      let s = this.b2;
      for (let j = 0; j < H; j++) { h[j] = Math.tanh(z[j]); s += this.W2[j] * h[j]; }
      const v = 1 / (1 + Math.exp(-s)), d = v - target;
      this.b2 -= lr * d;
      for (let j = 0; j < H; j++) {
        const gz = d * this.W2[j] * (1 - h[j] * h[j]);
        this.W2[j] -= lr * d * h[j];
        h[j] = gz; // reused: the hidden layer's error
        this.b1[j] -= lr * gz;
      }
      for (let i = 0; i < this.n; i++) {
        if (!b[i]) continue;
        const o = (b[i] === tok.mover ? i : this.n + i) * H;
        for (let j = 0; j < H; j++) this.W1[o + j] -= lr * h[j];
      }
      return v;
    }
    toJSON() { return { n: this.n, hidden: this.nH, W1: Array.from(this.W1, (v) => +v.toFixed(5)), b1: Array.from(this.b1, (v) => +v.toFixed(5)), W2: Array.from(this.W2, (v) => +v.toFixed(5)), b2: +this.b2.toFixed(5) }; }
    static from(o) { const net = new ValueNet(o.n, o.hidden); net.W1.set(o.W1); net.b1.set(o.b1); net.W2.set(o.W2); net.b2 = o.b2; return net; }
  }

  /* The value of every possible move for the player to move: a win is 1, filling the board 0.5, anything else the
     brain's value of the position it leads to. → { moves, vals } */
  RL.moveValues = function (game, brain, b, mover) {
    const moves = game.empties(b), vals = new Float64Array(moves.length), last = moves.length === 1;
    const net = brain.kind === 'net', z0 = net ? brain.pre(b, mover) : null, z = net ? new Float64Array(brain.nH) : null;
    moves.forEach((m, k) => {
      b[m] = mover;
      if (game.wins(b, m)) vals[k] = 1;
      else if (last) vals[k] = 0.5;
      else if (net) { const o = m * brain.nH; for (let j = 0; j < brain.nH; j++) z[j] = z0[j] + brain.W1[o + j]; vals[k] = brain.fromPre(z); }
      else vals[k] = brain.value(b, mover);
      b[m] = 0;
    });
    return { moves, vals };
  };
  // the move it values most (ties at random)
  RL.bestMove = function (game, brain, b, mover, rng) {
    const { moves, vals } = RL.moveValues(game, brain, b, mover);
    let best = [], v = -Infinity;
    vals.forEach((x, k) => { if (x > v + 1e-9) { v = x; best = [moves[k]]; } else if (Math.abs(x - v) <= 1e-9) best.push(moves[k]); });
    return { move: best[Math.floor(rng() * best.length)], value: v, moves, vals };
  };

  /* One game against itself, learning as it goes (temporal-difference learning, TD(0)): after each of a player's
     moves, the value of that player's previous position moves a step toward the value of the new one; at the end,
     each player's last position moves toward the result (1 win, 0.5 draw, 0 loss). Exploring moves (taken at
     random with probability ε) are not learned from, except at the end, since the result is real. */
  class Match {
    constructor(game, brain, o, rng) { Object.assign(this, { game, brain, o, rng, b: new Int8Array(game.n), last: [null, null, null], p: 1, moves: 0, done: false, winner: 0, line: null, order: [] }); }
    // one move by the player to move, and what it learns from it
    step() {
      const { game, brain, o, rng, b, last } = this, p = this.p, lr = brain.kind === 'net' ? o.lr : o.alpha;
      const explore = rng() < o.eps;
      const m = explore ? RL.randomMove(game, b, p, rng) : RL.bestMove(game, brain, b, p, rng).move;
      b[m] = p; this.moves++; this.order.push(m);
      const won = game.wins(b, m), full = this.moves === game.n;
      if (won || full) {
        if (last[p]) brain.learn(last[p], won ? 1 : 0.5, lr, game, rng);
        if (last[3 - p]) brain.learn(last[3 - p], won ? 0 : 0.5, lr, game, rng);
        Object.assign(this, { done: true, winner: won ? p : 0, line: won });
        return { m, explore, p };
      }
      const v = brain.value(b, p);
      if (last[p] && !explore) brain.learn(last[p], v, lr, game, rng);
      last[p] = brain.remember(b, p);
      this.p = 3 - p;
      return { m, explore, p, v };
    }
  }
  RL.Match = Match;
  RL.selfPlay = function (game, brain, o, rng) {
    const M = new Match(game, brain, o, rng);
    while (!M.done) M.step();
    return M;
  };

  // ------------------------------------------------------------------ opponents, and how well it plays
  RL.randomMove = (game, b, p, rng) => { const e = game.empties(b); return e[Math.floor(rng() * e.length)]; };
  // a simple rule-follower: win if it can, block the other player's winning cell if there is one, else play at random
  RL.tacticalMove = function (game, b, p, rng) {
    const e = game.empties(b);
    for (const who of [p, 3 - p]) for (const m of e) { b[m] = who; const w = game.wins(b, m); b[m] = 0; if (w) return m; }
    return e[Math.floor(rng() * e.length)];
  };
  // n games against an opponent, half moving first: the shares it wins, draws and loses
  RL.evaluate = function (game, brain, opponent, n, rng) {
    let win = 0, draw = 0, loss = 0;
    for (let k = 0; k < n; k++) {
      const me = k % 2 ? 2 : 1, b = new Int8Array(game.n);
      let p = 1, moves = 0, winner = 0;
      for (;;) {
        const m = p === me ? RL.bestMove(game, brain, b, p, rng).move : opponent(game, b, p, rng);
        b[m] = p; moves++;
        if (game.wins(b, m)) { winner = p; break; }
        if (moves === game.n) break;
        p = 3 - p;
      }
      if (winner === me) win++; else if (winner) loss++; else draw++;
    }
    return { win: win / n, draw: draw / n, loss: loss / n };
  };

  Object.assign(RL, { MAZE, DIRS, DIR_NAMES, Maze, QTable, LineGame, ValueTable, ValueNet });
  NA.rl = RL;
})(window.NA);
