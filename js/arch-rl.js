/* arch-rl.js — what the RL lab's agents have learned, as numbers to zoom into, and each example in Python. The
   maze's Q-table: a square opens into its four values, each down to its bits. Three-line's value table: the
   entries for the positions each move leads to. Four-line's value network: the board as 50 inputs, 64 tanh
   neurons and one sigmoid output, every neuron opening into its weighted sum and on down to the transistors. */
(function (NA) {
  'use strict';
  const { fmt, fmtInt, fmtPct, theme, clamp } = NA;
  const D = NA.deep, RL = NA.rl;
  const TANH = { id: 'tanh', label: 'tanh', f: Math.tanh }, SIGMOID = { id: 'sigmoid', label: 'sigmoid', f: (z) => 1 / (1 + Math.exp(-z)) };
  const STONE = ['', 'blue', 'orange'];
  const cellName = (g, i) => `row ${Math.floor(i / g.size) + 1}, column ${(i % g.size) + 1}`;

  // a small board picture: stones, and an optional cell to outline
  function miniBoard(ctx, g, b, r, mark) {
    const cs = Math.min(r.w, r.h) / g.size, x0 = r.x + (r.w - cs * g.size) / 2, y0 = r.y + (r.h - cs * g.size) / 2;
    for (let i = 0; i < g.n; i++) {
      const x = x0 + (i % g.size) * cs, y = y0 + Math.floor(i / g.size) * cs;
      ctx.strokeStyle = theme.c('line-2'); ctx.lineWidth = 1; ctx.strokeRect(x + 0.5, y + 0.5, cs - 1, cs - 1);
      if (b[i]) { ctx.beginPath(); ctx.arc(x + cs / 2, y + cs / 2, cs * 0.34, 0, Math.PI * 2); if (b[i] === 1) { ctx.fillStyle = theme.c('s1'); ctx.fill(); } else { ctx.strokeStyle = theme.c('s2'); ctx.lineWidth = Math.max(1.2, cs * 0.08); ctx.stroke(); } }
      if (i === mark) { ctx.strokeStyle = theme.c('focus'); ctx.lineWidth = 2; ctx.strokeRect(x + 2, y + 2, cs - 4, cs - 4); }
    }
  }

  // ------------------------------------------------------------------ the maze's Q-table
  const QRECT = [0, 0.02, 1, 0.96];
  function mazeTree(L, size) {
    const S = L.S, M = S.maze, Q = S.Q, X = { wide: size.w / size.h >= 1.05 }, open = M.cell.filter((c) => c === '.' || c === 'S').length;
    const square = (s) => {
      const x = s % M.W, y = Math.floor(s / M.W), vals = [0, 1, 2, 3].map((a) => Q.Q[s * 4 + a]), best = Q.best(s);
      return D.lazy({
        title: `Square (${x}, ${y})`, sub: `best: ${RL.DIR_NAMES[best]} (${fmt(vals[best], 3)})`, tone: 's1',
        summary: (ctx, r, Sk) => Sk.bars(ctx, Float32Array.from(vals), r),
        inner: (ctx, r, Sk) => {
          const u = Sk.u(r, 8, 13);
          Sk.lines(ctx, [...vals.map((v, a) => `Q(${x}, ${y}, ${RL.DIR_NAMES[a].padEnd(5)}) = ${fmt(v, 4)}${a === best ? '   ← its best move' : M.move(s, a) === s ? '   (a wall: it stays)' : ''}`), '',
            'after a move:  Q ← Q + α·(r + γ·max Q(next square) − Q)', `α = ${fmt(S.o.alpha, 2)}, γ = ${fmt(S.o.gamma, 2)}, visited ${fmtInt(Q.seen[s])} times`], Sk.place(r, [0, 0, 1, 0.48]), { mono: true, size: u, strong: (i) => i === best });
          Sk.caption(ctx, 'its four values, one per move: zoom into one to see the number as it is stored', Sk.place(r, [0, 0.62, 1, 0.2]), { size: u * 0.8, minW: 220 });
          Sk.bars(ctx, Float32Array.from(vals), Sk.place(r, [0, 0.62, 1, 0.2]));
        },
        grids: [{ id: 'q', rect: [0, 0.62, 1, 0.2], rows: 1, cols: 4, make: (i, a) => D.stored(X, vals[a], `Q(${x}, ${y}, ${RL.DIR_NAMES[a]})`, 'weight') }],
        about: [`The four entries of the Q-table for square (${x}, ${y}): the reward it expects if it moves ${RL.DIR_NAMES.join(', ')} from here and then plays its best. It moves ${RL.DIR_NAMES[best]}, the largest. Moving into a wall leaves it where it is, which only costs a step.`,
          'Each entry changes only when it makes that move from this square: the entry moves a fraction α of the way toward the reward it got plus γ times the best entry of the square it landed on. So values flow backward, one square per visit, from the goal and the pits.'],
        code: `# Q[(x, y)] holds the values of the moves up, right, down, left from that square\nQ = {(${x}, ${y}): [${vals.map((v) => NA.fmtCode(v, 4)).join(', ')}]}\nmoves = ['up', 'right', 'down', 'left']\nq = Q[(${x}, ${y})]\nprint('best move:', moves[q.index(max(q))])`,
      });
    };
    return {
      id: 'qtable', title: 'Q-table', sub: `${open} squares × 4 moves = ${open * 4} numbers`, tone: 's1',
      inner: (ctx, c) => {
        const gr = NA.archKit.place(c, QRECT), cs = Math.min(gr.w / M.W, gr.h / M.H);
        let vmax = 0.05; for (let i = 0; i < M.n; i++) if (M.open(i)) vmax = Math.max(vmax, Math.abs(Q.max(i)));
        for (let i = 0; i < M.n; i++) {
          const x = gr.x + (i % M.W) * cs, y = gr.y + Math.floor(i / M.W) * cs, ch = M.cell[i];
          ctx.fillStyle = ch === '#' ? theme.c('line-2') : ch === 'G' ? theme.c('pos', 0.3) : ch === 'X' ? theme.c('neg', 0.3) : theme.divColor(clamp(Q.max(i) / vmax, -1, 1) * 0.9, 0.35);
          ctx.fillRect(x, y, cs, cs); ctx.strokeStyle = theme.c('line'); ctx.strokeRect(x + 0.5, y + 0.5, cs - 1, cs - 1);
          if (ch === 'G' || ch === 'X') { NA.archKit.text(ctx, ch === 'G' ? '+1' : '−1', x + cs / 2, y + cs / 2, { size: cs * 0.28, w: 700, align: 'center', color: theme.c(ch === 'G' ? 'pos' : 'neg') }); continue; }
          if (ch === '#' || cs < 40) continue;
          // the four values at the four sides of the square
          const f = clamp(cs * 0.15, 7, 11);
          [[0.5, 0.2], [0.8, 0.5], [0.5, 0.8], [0.2, 0.5]].forEach(([fx, fy], a) => NA.archKit.text(ctx, fmt(Q.Q[i * 4 + a], 2), x + cs * fx, y + cs * fy, { size: f, mono: true, align: 'center', w: a === Q.best(i) ? 700 : 500, color: theme.c(a === Q.best(i) ? 'ink' : 'ink-2') }));
        }
      },
      grids: [{ id: 'sq', rect: QRECT, rows: M.H, cols: M.W, square: true, skip: (i, j) => !(M.cell[i * M.W + j] === '.' || M.cell[i * M.W + j] === 'S'), make: (i, j) => square(i * M.W + j) }],
      about: [`Everything the agent knows: ${open * 4} numbers, four for each square it can stand on (up, right, down, left), drawn at the square's four sides. The largest of the four is the move it takes there. ${NA.tapWord()} a square for details; double-${NA.tapWord(true)} it to open it.`,
        'There is no neural network here: a table is enough when there are few situations. When there are too many to list (a game board, a screen of pixels), a network takes the table’s place, as in Four-line.'],
      code: NA.archRL.py.maze(L),
    };
  }

  // ------------------------------------------------------------------ Three-line's value table
  function threeTree(L, size) {
    const S = L.S, g = S.game, T = S.brain, X = { wide: size.w / size.h >= 1.05 }, pos = L.position();
    const mover = pos.toMove || 1, b = pos.toMove ? Int8Array.from(pos.b) : new Int8Array(g.n);
    const { moves, vals } = RL.moveValues(g, T, b, mover), shown = moves.slice(0, 16);
    const entry = (k) => {
      const m = shown[k], after = Int8Array.from(b); after[m] = mover;
      const win = !!g.wins(after, m), key = T.key(after, mover), known = T.V.has(key), v = vals[k];
      return D.lazy({
        title: `Move at ${cellName(g, m)}`, sub: win ? 'wins at once' : `${fmtPct(v, 0)}${known ? '' : ' (never met: 0.5)'}`, tone: 's4',
        summary: (ctx, r) => miniBoard(ctx, g, after, r, m),
        inner: (ctx, r, Sk) => {
          const u = Sk.u(r, 8, 13);
          miniBoard(ctx, g, after, Sk.place(r, [0, 0, 0.42, 0.62]), m);
          Sk.lines(ctx, [`the position after ${STONE[mover]} plays here`, `seen by ${STONE[mover]}: x = theirs, o = the other's`, `key: ${key}`, win ? 'a win: worth 1, no need to look it up' : `value: ${fmt(v, 4)}${known ? '' : ' (not in the table: 0.5)'}`],
            Sk.place(r, [0.46, 0.02, 0.54, 0.6]), { mono: true, size: u * 0.92, strong: (i) => i === 3 });
        },
        grids: win ? [] : [{ id: 'v', rect: [0, 0.74, 1, 0.14], rows: 1, cols: 1, make: () => D.stored(X, v, `V(${key})`, 'weight') }],
        about: [`If ${STONE[mover]} plays at ${cellName(g, m)}, this is the position. The table is keyed by the position as the player who just moved sees it (x their stones, o the other's), turned and mirrored to the first of its 8 symmetric versions, so all 8 share one entry.`,
          win ? 'This move makes three in a row: it wins outright, worth 1.' : known ? `Its entry says ${fmtPct(v, 0)}: the chance it estimates that ${STONE[mover]} goes on to win from here.` : 'It has never met this position, so it counts as 0.5, a coin flip.'],
        code: `# the value table maps a position (as the player who just moved sees it) to their chance of winning\nV = {'${key}': ${NA.fmtCode(known ? v : 0.5, 4)}}\nprint(V.get('${key}', 0.5))`,
      });
    };
    return {
      id: 'table', title: 'Value table', sub: `${fmtInt(T.size)} positions`, tone: 's4',
      inner: (ctx, c, Sk) => {
        const u = Sk.u(c, 8, 13);
        miniBoard(ctx, g, b, Sk.place(c, [0, 0.04, 0.3, 0.46]), -1);
        Sk.lines(ctx, [pos.toMove ? `${STONE[mover]} to move` : 'an empty board: blue to move', 'right: the table entry for each move'], Sk.place(c, [0, 0.54, 0.3, 0.12]), { size: u * 0.9 });
      },
      // one box per move, four to a row on the right
      children: shown.map((m, k) => { const rows = Math.ceil(shown.length / 4), cw = 0.67 / 4, rh = 0.96 / rows; return Object.assign(entry(k), { id: `m${m}`, rect: [0.33 + (k % 4) * cw + 0.006, 0.02 + Math.floor(k / 4) * rh + 0.01, cw - 0.012, rh - 0.02] }); }),
      about: [`One entry per position it has met in self-play: ${fmtInt(T.size)} so far. To move, it looks up the position each empty cell leads to and takes the best. Each box on the right is one move from the position on the left, with its entry.`,
        'A table learns fast and exactly, but only about positions it has met; it knows nothing about a position one stone away from one it has seen. On a 5 × 5 board there are far too many positions to meet them all, which is why Four-line uses a network.'],
      code: NA.archRL.py.three(L),
    };
  }

  // ------------------------------------------------------------------ Four-line's value network
  function fourTree(L, size) {
    const S = L.S, g = S.game, net = S.brain, X = { wide: size.w / size.h >= 1.05 }, pos = L.position(), b = pos.b, H = net.nH;
    // the position it judges: the board seen by the player who placed the last stone
    const mover = pos.order.length ? ((pos.order.length - 1) % 2) + 1 : 2;
    const x = new Float64Array(net.nIn);
    for (let i = 0; i < g.n; i++) if (b[i]) x[b[i] === mover ? i : g.n + i] = 1;
    const z = net.pre(b, mover), h = Array.from(z, Math.tanh);
    let s = net.b2; for (let j = 0; j < H; j++) s += net.W2[j] * h[j];
    const v = 1 / (1 + Math.exp(-s));
    const inName = (i) => `${i < g.n ? 'their stone' : 'other stone'} at ${cellName(g, i % g.n)}`;
    const hidden = (j) => Object.assign(D.neuron(X, {
      title: `Hidden neuron ${j + 1}`, sub: `tanh: ${fmt(h[j], 3)}`, tone: 's1',
      terms: Array.from({ length: net.nIn }, (_, i) => ({ a: x[i], w: net.W1[i * H + j] })), bias: net.b1[j], act: TANH,
      inLabel: (i) => inName(i), wLabel: (i) => `weight from ${inName(i)}`, inWhat: 'the board, 2 inputs per cell',
      vars: { w: `net[0].weight[${j}]`, a: 'x', b: `net[0].bias[${j}]` }, codeWhat: 'as one row of nn.Linear(50, 64)', outName: 'h',
      about: [`One of the network’s ${H} hidden neurons, on the position on the board (seen by ${STONE[mover]}, who moved last). Its inputs are the 50 board inputs, 1 where a stone is and 0 elsewhere, so only the weights of occupied cells count: it adds up the weights of the stones on the board.`],
    }), {});
    const out = Object.assign(D.neuron(X, {
      title: 'Output: the chance to win', sub: `sigmoid: ${fmtPct(v, 1)}`, tone: 's4',
      terms: h.map((hj, j) => ({ a: hj, w: net.W2[j] })), bias: net.b2, act: SIGMOID,
      inLabel: (j) => `hidden neuron ${j + 1}`, wLabel: (j) => `weight from hidden neuron ${j + 1}`, inWhat: 'the 64 hidden neurons',
      vars: { w: 'net[2].weight[0]', a: 'h', b: 'net[2].bias[0]' }, codeWhat: 'as nn.Linear(64, 1) and a sigmoid', outName: 'p',
      about: [`The output neuron weighs the ${H} hidden neurons and squashes the sum with a sigmoid into a chance between 0 and 1: ${fmtPct(v, 1)} that ${STONE[mover]}, who moved last, goes on to win this position.`],
    }), { id: 'out' });
    const IN = [0, 0.04, 0.24, 0.92], HID = [0.32, 0.12, 0.4, 0.76], OUT = [0.8, 0.36, 0.2, 0.28], HC = Math.ceil(Math.sqrt(H));
    return {
      id: 'net', title: 'Value network', sub: `${net.nIn} → ${H} tanh → 1 sigmoid, ${fmtInt(net.paramCount)} parameters`, tone: 's1',
      inner: (ctx, c, Sk) => {
        const u = Sk.u(c, 8, 12), ri = Sk.place(c, IN), rh = Sk.place(c, HID), ro = Sk.place(c, OUT);
        // the two input planes: the mover's stones and the other player's
        const half = { x: ri.x, y: ri.y, w: ri.w, h: ri.h / 2 - 8 }, cs = Math.min(half.w / g.size, half.h / g.size);
        [0, 1].forEach((p) => {
          const y0 = ri.y + p * (ri.h / 2) + 14;
          Sk.text(ctx, p ? 'other player’s stones' : `${STONE[mover]}’s stones (moved last)`, ri.x, y0 - 8, { size: u * 0.8, color: theme.c('muted'), maxW: ri.w });
          for (let i = 0; i < g.n; i++) { const on = x[p * g.n + i]; ctx.fillStyle = on ? theme.c(p ? 's2' : 's1', 0.85) : theme.c('line', 0.5); ctx.fillRect(ri.x + (i % g.size) * cs + 1, y0 + Math.floor(i / g.size) * cs + 1, cs - 2, cs - 2); }
        });
        Sk.text(ctx, `${H} hidden neurons (tanh): blue +, orange −`, rh.x, rh.y - 8, { size: u * 0.8, color: theme.c('muted'), maxW: rh.w });
        // the same square cells the grid of neurons uses (so a dot sits where a click opens it)
        const hs = Math.min(rh.w / HC, rh.h / Math.ceil(H / HC));
        for (let j = 0; j < H; j++) { ctx.fillStyle = theme.divColor(h[j] * 0.9, 0.9); ctx.beginPath(); ctx.arc(rh.x + (j % HC + 0.5) * hs, rh.y + (Math.floor(j / HC) + 0.5) * hs, hs * 0.38, 0, Math.PI * 2); ctx.fill(); }
        Sk.arrow(ctx, ri.x + ri.w + 6, c.y + c.h * 0.5, rh.x - 6, c.y + c.h * 0.5, { lw: 1.3 });
        Sk.arrow(ctx, rh.x + HC * hs + 6, rh.y + hs * HC * 0.5, ro.x - 6, ro.y + ro.h / 2, { lw: 1.3 });
      },
      grids: [{ id: 'hid', rect: HID, rows: Math.ceil(H / HC), cols: HC, square: true, skip: (i, j) => i * HC + j >= H, make: (i, j) => (i * HC + j < H ? hidden(i * HC + j) : null) }],
      children: [Object.assign(out, { rect: OUT })],
      about: [`The network that judges Four-line positions, on the position on the board as ${STONE[mover]} (who moved last) sees it: the left squares are its 50 inputs, the dots its ${H} hidden neurons (blue positive, orange negative), and the box its output, the chance to win: ${fmtPct(v, 1)}.`,
        'To choose a move it runs this for every empty cell, on the position that move would make, and plays the highest. Self-play nudges every weight after each move, so patterns that led to wins (lines of three with room to grow, blocks) raise the output. Click a neuron for details; double-click to open it, down to the transistors.'],
      code: NA.archRL.py.four(L),
    };
  }

  // ------------------------------------------------------------------ the examples in Python
  const LINES_PY = (N, K) => `N, K = ${N}, ${K}                                 # a ${N} x ${N} board: ${K} in a row wins
LINES = []                                       # every run of K cells: across, down and both diagonals
for r in range(N):
    for c in range(N):
        for dr, dc in ((0, 1), (1, 0), (1, 1), (1, -1)):
            cells = [(r + dr * t, c + dc * t) for t in range(K)]
            if all(0 <= rr < N and 0 <= cc < N for rr, cc in cells):
                LINES.append([rr * N + cc for rr, cc in cells])
THROUGH = [[line for line in LINES if i in line] for i in range(N * N)]

# the 8 rotations and mirror images of the board, as where each cell goes
MAPS = [lambda r, c: (r, c), lambda r, c: (c, N - 1 - r), lambda r, c: (N - 1 - r, N - 1 - c), lambda r, c: (N - 1 - c, r),
        lambda r, c: (r, N - 1 - c), lambda r, c: (c, r), lambda r, c: (N - 1 - r, c), lambda r, c: (N - 1 - c, N - 1 - r)]
SYMS = [[f(i // N, i % N)[0] * N + f(i // N, i % N)[1] for i in range(N * N)] for f in MAPS]

def wins(board, i):
    """Did the stone just placed at cell i make K in a row?"""
    return any(all(board[c] == board[i] for c in line) for line in THROUGH[i])

def empties(board):
    return [i for i, v in enumerate(board) if v == 0]`;
  const VS_RANDOM_PY = `def against_random(games):
    """Its share of wins against a random player, moving first in half the games."""
    won = 0
    for g in range(games):
        me, board, p, moves = 1 + g % 2, [0] * (N * N), 1, 0
        while True:
            m = best_move(board, p) if p == me else random.choice(empties(board))
            board[m] = p
            moves += 1
            if wins(board, m):
                won += p == me
                break
            if moves == N * N:
                break
            p = 3 - p
    return won / games`;
  const py = {
    maze: (L) => {
      const S = L.S, o = S.o;
      return `import random

MAZE = [
${S.maze.rows.map((r) => `    '${r}',`).join('\n')}
]                                                # S start, G goal (+1), X pit (-1), # wall
H, W = len(MAZE), len(MAZE[0])
MOVES = [(0, -1), (1, 0), (0, 1), (-1, 0)]       # up, right, down, left
START = next((x, y) for y in range(H) for x in range(W) if MAZE[y][x] == 'S')

def step(x, y, a, slip=${NA.fmtCode(S.maze.slip, 2)}):
    """One move: the next square, the reward, and whether the episode is over."""
    if random.random() < slip:                   # a slippery floor: now and then the move goes sideways
        a = (a + random.choice([1, 3])) % 4
    nx, ny = x + MOVES[a][0], y + MOVES[a][1]
    if not (0 <= nx < W and 0 <= ny < H) or MAZE[ny][nx] == '#':
        nx, ny = x, y                            # a wall: stay put
    cell = MAZE[ny][nx]
    if cell == 'G':
        return (nx, ny), 1.0, True
    if cell == 'X':
        return (nx, ny), -1.0, True
    return (nx, ny), -0.02, False                # every step costs a little

Q = {}                                           # Q[(x, y)] = expected reward of each move from that square
def q(s):
    return Q.setdefault(s, [0.0] * 4)

alpha, gamma, eps = ${NA.fmtCode(o.alpha, 2)}, ${NA.fmtCode(o.gamma, 2)}, ${NA.fmtCode(o.eps, 2)}
for episode in range(3000):
    s = START
    for t in range(100):
        a = random.randrange(4) if random.random() < eps else max(range(4), key=lambda k: q(s)[k])
        s2, r, done = step(*s, a)
        target = r if done else r + gamma * max(q(s2))
        q(s)[a] += alpha * (target - q(s)[a])    # Q-learning: a step toward reward + discounted best next value
        s = s2
        if done:
            break

# the route it learned: its best move on every square, from the start
s, route = START, [START]
for t in range(60):
    s, r, done = step(*s, max(range(4), key=lambda k: q(s)[k]), slip=0)
    route.append(s)
    if done:
        break
print(len(route) - 1, 'steps:', route)`;
    },
    three: (L) => {
      const o = L.S.o;
      return `import random

${LINES_PY(4, 3)}

V = {}                                           # position -> chance that the player who just moved wins

def key(board, me):
    """The position as player \`me\` sees it (x mine, o theirs), the same for all 8 symmetric versions."""
    best = None
    for perm in SYMS:
        cells = ['.'] * (N * N)
        for i, v in enumerate(board):
            cells[perm[i]] = '.' if v == 0 else ('x' if v == me else 'o')
        s = ''.join(cells)
        if best is None or s < best:
            best = s
    return best

def move_value(board, m, me):
    """What \`me\` gets by playing at m: 1 for a win, 0.5 for filling the board, else the table's value."""
    board[m] = me
    v = 1.0 if wins(board, m) else 0.5 if not empties(board) else V.get(key(board, me), 0.5)
    board[m] = 0
    return v

def best_move(board, me):
    vals = {m: move_value(board, m, me) for m in empties(board)}
    top = max(vals.values())
    return random.choice([m for m, v in vals.items() if v == top])

def learn(k, target, alpha):
    v = V.get(k, 0.5)
    V[k] = v + alpha * (target - v)

def self_play(eps=${NA.fmtCode(o.eps, 2)}, alpha=${NA.fmtCode(o.alpha, 2)}):
    """One game against itself, learning as it goes: TD(0) on the positions after each move."""
    board, last, p, moves = [0] * (N * N), {1: None, 2: None}, 1, 0
    while True:
        explore = random.random() < eps
        m = random.choice(empties(board)) if explore else best_move(board, p)
        board[m] = p
        moves += 1
        won, full = wins(board, m), moves == N * N
        if won or full:                          # the result: 1 for the winner, 0 for the loser, 0.5 each for a draw
            if last[p]:
                learn(last[p], 1.0 if won else 0.5, alpha)
            if last[3 - p]:
                learn(last[3 - p], 0.0 if won else 0.5, alpha)
            return
        k = key(board, p)
        if last[p] and not explore:              # its previous position moves toward this one's value
            learn(last[p], V.get(k, 0.5), alpha)
        last[p] = k
        p = 3 - p

${VS_RANDOM_PY}

for game in range(10000):
    self_play()
print(len(V), 'positions in the table; beats a random player', against_random(400))`;
    },
    four: (L) => {
      const o = L.S.o, H = L.S.brain.nH;
      return `import random
import torch
import torch.nn as nn

${LINES_PY(5, 4)}

net = nn.Sequential(nn.Linear(2 * N * N, ${H}), nn.Tanh(), nn.Linear(${H}, 1))   # outputs the logit of the chance to win
opt = torch.optim.SGD(net.parameters(), lr=${NA.fmtCode(o.lr, 3)})
loss_fn = nn.BCEWithLogitsLoss()                  # with a sigmoid output, its gradient is (output - target)

def features(board, me):
    """The board as player \`me\` sees it: one plane for its stones, one for the other player's."""
    x = torch.zeros(2 * N * N)
    for i, v in enumerate(board):
        if v:
            x[i if v == me else N * N + i] = 1.0
    return x

def value(board, me):
    with torch.no_grad():
        return torch.sigmoid(net(features(board, me))).item()

def best_move(board, me):
    """Score the position every empty cell leads to, all in one batch, and take the best."""
    moves, xs, fixed = empties(board), [], {}
    for m in moves:
        board[m] = me
        if wins(board, m):
            fixed[m] = 1.0
        elif len(moves) == 1:
            fixed[m] = 0.5
        xs.append(features(board, me))
        board[m] = 0
    with torch.no_grad():
        v = torch.sigmoid(net(torch.stack(xs))).squeeze(1).tolist()
    vals = [fixed.get(m, v[k]) for k, m in enumerate(moves)]
    top = max(vals)
    return random.choice([m for m, x in zip(moves, vals) if x == top])

def learn(board, me, target):
    """One gradient step toward a target, on one of the 8 symmetric views of the position."""
    perm, b = random.choice(SYMS), [0] * (N * N)
    for i, v in enumerate(board):
        b[perm[i]] = v
    loss = loss_fn(net(features(b, me)), torch.tensor([float(target)]))
    opt.zero_grad()
    loss.backward()
    opt.step()

def self_play(eps=${NA.fmtCode(o.eps, 2)}):
    """One game against itself, learning as it goes: TD(0), with the network in place of a table."""
    board, last, p, moves = [0] * (N * N), {1: None, 2: None}, 1, 0
    while True:
        explore = random.random() < eps
        m = random.choice(empties(board)) if explore else best_move(board, p)
        board[m] = p
        moves += 1
        won, full = wins(board, m), moves == N * N
        if won or full:
            if last[p]:
                learn(*last[p], 1.0 if won else 0.5)
            if last[3 - p]:
                learn(*last[3 - p], 0.0 if won else 0.5)
            return
        if last[p] and not explore:
            learn(*last[p], value(board, p))
        last[p] = (board[:], p)
        p = 3 - p

${VS_RANDOM_PY}

for game in range(1500):                         # a minute or so; it beats a random player within a few hundred
    self_play()
print('beats a random player', against_random(200))`;
    },
  };

  NA.archRL = { build: (L, size) => (L.S.maze ? mazeTree(L, size) : L.S.id === 'three' ? threeTree(L, size) : fourTree(L, size)), py };
})(window.NA);
