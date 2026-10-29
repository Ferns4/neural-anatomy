/* arch-rl.js — the RL lab's agents as architectures to zoom into, a picture of what the model computes beside the
   board, and each example in Python. The top level is the agent's loop: what it sees, its model, its choice, the
   maze or game, and how it learns. The models: the maze's Q-table (a square opens into its four values, each down
   to its bits), the maze Q-network (104 inputs, 64 ReLU neurons, 4 Q-values), Three-line's value table, and the
   value networks of Three-line and Four-line; every neuron opens into its weighted sum and on down to the
   transistors. Everything is computed for one state, the one being looked at: the test's current step, the game on
   the board (or the move under the pointer), or the chosen square of the maze. */
(function (NA) {
  'use strict';
  const { fmt, fmtInt, fmtPct, theme, clamp } = NA;
  const D = NA.deep, RL = NA.rl;
  const TANH = { id: 'tanh', label: 'tanh', f: Math.tanh }, SIGMOID = { id: 'sigmoid', label: 'sigmoid', f: (z) => 1 / (1 + Math.exp(-z)) };
  const RELU = { id: 'relu', label: 'ReLU', f: (z) => Math.max(0, z) }, LINEAR = { id: 'linear', label: 'none' };
  const STONE = ['', 'blue', 'orange'], ARROW = ['↑', '→', '↓', '←'], DIR = RL.DIR_NAMES, NV = 24;
  const K = () => NA.archKit;
  const cellName = (g, i) => `row ${Math.floor(i / g.size) + 1}, column ${(i % g.size) + 1}`;
  const sqName = (M, s) => `(${s % M.W}, ${Math.floor(s / M.W)})`;
  const sigmoid = (z) => 1 / (1 + Math.exp(-z));

  // ------------------------------------------------------------------ small pictures
  // a board: stones, and an optional cell to outline
  function miniBoard(ctx, g, b, r, mark) {
    const cs = Math.min(r.w, r.h) / g.size, x0 = r.x + (r.w - cs * g.size) / 2, y0 = r.y + (r.h - cs * g.size) / 2;
    for (let i = 0; i < g.n; i++) {
      const x = x0 + (i % g.size) * cs, y = y0 + Math.floor(i / g.size) * cs;
      ctx.strokeStyle = theme.c('line-2'); ctx.lineWidth = 1; ctx.strokeRect(x + 0.5, y + 0.5, cs - 1, cs - 1);
      if (b[i]) { ctx.beginPath(); ctx.arc(x + cs / 2, y + cs / 2, cs * 0.34, 0, Math.PI * 2); if (b[i] === 1) { ctx.fillStyle = theme.c('s1'); ctx.fill(); } else { ctx.strokeStyle = theme.c('s2'); ctx.lineWidth = Math.max(1.2, cs * 0.08); ctx.stroke(); } }
      if (i === mark) { ctx.strokeStyle = theme.c('focus'); ctx.lineWidth = 2; ctx.strokeRect(x + 2, y + 2, cs - 4, cs - 4); }
    }
    return { x0, y0, cs };
  }
  // a maze: walls, pits, the goal, squares visited (dots), the agent, its 5 × 5 view, and an arrow for a move
  function miniMaze(ctx, M, r, o = {}) {
    const cs = Math.min(r.w / M.W, r.h / M.H), x0 = r.x + (r.w - cs * M.W) / 2, y0 = r.y + (r.h - cs * M.H) / 2;
    for (let i = 0; i < M.n; i++) {
      const x = x0 + (i % M.W) * cs, y = y0 + Math.floor(i / M.W) * cs, c = M.cell[i];
      ctx.fillStyle = c === '#' ? theme.c('line-2') : c === 'G' ? theme.c('pos', 0.45) : c === 'X' ? theme.c('neg', 0.45) : theme.c('panel');
      ctx.fillRect(x, y, cs, cs); ctx.strokeStyle = theme.c('line'); ctx.lineWidth = 0.6; ctx.strokeRect(x + 0.5, y + 0.5, cs - 1, cs - 1);
      if (o.seen && o.seen[i] && c !== 'G' && i !== o.agent) { ctx.fillStyle = theme.c('muted', 0.75); ctx.beginPath(); ctx.arc(x + cs / 2, y + cs / 2, Math.max(1.2, cs * 0.1), 0, Math.PI * 2); ctx.fill(); }
      if (o.mark === i) { ctx.strokeStyle = theme.c('focus'); ctx.lineWidth = 2; ctx.strokeRect(x + 1.5, y + 1.5, cs - 3, cs - 3); }
    }
    if (o.agent != null) {
      const ax = x0 + (o.agent % M.W + 0.5) * cs, ay = y0 + (Math.floor(o.agent / M.W) + 0.5) * cs;
      if (o.view) { ctx.strokeStyle = theme.c('s7', 0.9); ctx.lineWidth = 1.5; ctx.setLineDash([3, 2]); ctx.strokeRect(ax - 2.5 * cs, ay - 2.5 * cs, 5 * cs, 5 * cs); ctx.setLineDash([]); }
      if (o.move != null) K().arrow(ctx, ax, ay, ax + RL.DIRS[o.move][0] * cs * 0.95, ay + RL.DIRS[o.move][1] * cs * 0.95, { color: theme.c('ink'), lw: 1.6, head: Math.max(4, cs * 0.28) });
      ctx.fillStyle = theme.c('s7'); ctx.beginPath(); ctx.arc(ax, ay, cs * 0.26, 0, Math.PI * 2); ctx.fill();
    }
    return { x0, y0, cs };
  }
  // what the maze network sees, from its inputs: the 5 × 5 squares around it (it is the middle one)
  function drawView(ctx, x, r) {
    const on = new Set(x), cs = Math.min(r.w, r.h) / 5, x0 = r.x + (r.w - cs * 5) / 2, y0 = r.y;
    let k = 0;
    for (let kk = 0; kk < 25; kk++) {
      const px = x0 + (kk % 5) * cs, py = y0 + Math.floor(kk / 5) * cs;
      if (kk === 12) {
        ctx.fillStyle = theme.c('panel'); ctx.fillRect(px, py, cs, cs);
        ctx.fillStyle = theme.c('s7'); ctx.beginPath(); ctx.arc(px + cs / 2, py + cs / 2, cs * 0.28, 0, Math.PI * 2); ctx.fill();
      } else {
        const wall = on.has(k), pit = on.has(NV + k), goal = on.has(2 * NV + k), seen = on.has(3 * NV + k);
        ctx.fillStyle = wall ? theme.c('line-2') : pit ? theme.c('neg', 0.5) : goal ? theme.c('pos', 0.5) : theme.c('panel');
        ctx.fillRect(px, py, cs, cs);
        if (seen) { ctx.fillStyle = theme.c('muted', 0.8); ctx.beginPath(); ctx.arc(px + cs / 2, py + cs / 2, Math.max(1.5, cs * 0.12), 0, Math.PI * 2); ctx.fill(); }
        k++;
      }
      ctx.strokeStyle = theme.c('line'); ctx.lineWidth = 0.8; ctx.strokeRect(px + 0.5, py + 0.5, cs - 1, cs - 1);
    }
    return { x0, y0, cs, h: cs * 5 };
  }
  // which way the goal is, from the last 8 inputs
  const GOAL_WORDS = ['1–2 right', '3+ right', '1–2 left', '3+ left', '1–2 below', '3+ below', '1–2 above', '3+ above'];
  const goalText = (x) => { const w = x.filter((i) => i >= 4 * NV).map((i) => GOAL_WORDS[i - 4 * NV]); return w.length ? `goal: ${w.join(', ')}` : 'goal: here'; };
  // the 8 goal inputs as two rows of lit squares (near, far) under the four directions
  function drawGoalIns(ctx, x, r, u) {
    const on = new Set(x), cw = r.w / 4, s = Math.min(cw * 0.42, r.h / 3.2);
    ['→', '←', '↓', '↑'].forEach((t, d) => {
      const cx = r.x + (d + 0.5) * cw;
      K().text(ctx, t, cx, r.y + s * 0.6, { size: u, w: 700, align: 'center', color: theme.c('ink-2') });
      for (let f = 0; f < 2; f++) {
        const lit = on.has(4 * NV + d * 2 + f), y = r.y + s * 1.25 + f * (s + 2);
        ctx.fillStyle = lit ? theme.c('pos', 0.75) : theme.c('panel'); ctx.fillRect(cx - s / 2, y, s, s);
        ctx.strokeStyle = theme.c('line-2'); ctx.lineWidth = 0.8; ctx.strokeRect(cx - s / 2 + 0.5, y + 0.5, s - 1, s - 1);
      }
    });
  }

  // ------------------------------------------------------------------ the state being looked at
  /* What the agent is looking at now, with every number its model computes there: the test's current step, the
     square of a walk being watched, or the chosen square of the maze; in a game, the move under the pointer, the
     test's move, the last move on the board, or (an empty board) the first move it would make. */
  function probe(L) {
    const S = L.S;
    if (S.id === 'maze') {
      const T = L.test, f = T && T.frames[T.k], w = !f && L.watch && S.walk ? S.walk : null, M = (w && w.maze) || S.maze;
      const s = f ? f.s : w ? w.s : S.sel, why = f ? 'test' : w ? 'watch' : 'square';
      if (S.model === 'table') { const q = [0, 1, 2, 3].map((a) => S.Q.Q[s * 4 + a]); return { kind: 'mazeTable', M, s, q, a: f ? f.a : S.Q.best(s), why, f, T }; }
      let seen = f ? f.seen : w && w.seen ? w.seen : null;
      if (!seen) { seen = new Uint8Array(M.n); seen[s] = 1; }
      const x = RL.MazeNet.inputs(M, s, seen), q = Float64Array.from(S.net.forward(x));
      return { kind: 'mazeNet', M, s, seen, x, q, z: Float64Array.from(S.net.z), h: Float64Array.from(S.net.h), a: RL.argmax4(q), why, f, T, net: S.net };
    }
    const g = S.game, pos = L.position(), hov = L.hoverCell, toMove = pos.toMove;
    let b = pos.b, mover = 0, m = -1, why;
    if (toMove && L.valuesShown(pos) && hov >= 0 && hov < g.n && !pos.b[hov]) { b = Int8Array.from(pos.b); b[hov] = toMove; mover = toMove; m = hov; why = 'hover'; }
    else if (pos.test && pos.test.m != null) { b = Int8Array.from(pos.b); b[pos.test.m] = pos.test.p; mover = pos.test.p; m = pos.test.m; why = pos.test.who === 'agent' ? 'choice' : 'opponent'; }
    else if (pos.order.length) { m = pos.order[pos.order.length - 1]; mover = b[m]; why = 'last'; }
    else {
      const { moves, vals } = RL.moveValues(g, S.brain, b, 1);
      let k = 0; vals.forEach((v, i) => { if (v > vals[k]) k = i; });
      b = Int8Array.from(b); m = moves[k]; b[m] = 1; mover = 1; why = 'first';
    }
    const win = m >= 0 && !!g.wins(b, m), base = { g, b, mover, m, why, win, pos, test: pos.test || null };
    if (S.model === 'table') {
      const key = S.brain.key(b, mover), known = S.brain.V.has(key);
      return Object.assign(base, { kind: 'gameTable', key, known, v: win ? 1 : known ? S.brain.V.get(key) : 0.5, table: S.brain });
    }
    const net = S.brain, x = new Float64Array(net.nIn);
    for (let i = 0; i < g.n; i++) if (b[i]) x[b[i] === mover ? i : g.n + i] = 1;
    const z = Float64Array.from(net.pre(b, mover)), h = Array.from(z, Math.tanh);
    let s = net.b2; for (let j = 0; j < net.nH; j++) s += net.W2[j] * h[j];
    return Object.assign(base, { kind: 'gameNet', x, z, h, s, v: sigmoid(s), net });
  }

  // ------------------------------------------------------------------ the picture beside the board
  function contextLine(L, P) {
    if (P.kind === 'mazeNet' || P.kind === 'mazeTable') {
      const at = `square ${sqName(P.M, P.s)}`;
      if (P.why === 'test') return `Step ${P.T.k + 1} of ${P.T.frames.length}: ${at}, it moves ${DIR[P.a]}`;
      if (P.why === 'watch') return `Learning, move by move: ${at}`;
      return `${at[0].toUpperCase() + at.slice(1)}: its best move is ${DIR[P.a]} (${NA.tapWord()} a square)`;
    }
    const where = cellName(P.g, P.m), who = STONE[P.mover];
    if (P.why === 'hover') return `If ${who} played at ${where}`;
    if (P.why === 'choice') return `Move ${P.test.order.length + 1}: it plays ${who} at ${where}`;
    if (P.why === 'opponent') return `Move ${P.test.order.length + 1}: the rule-follower (${who}) plays ${where}`;
    if (P.why === 'first') return `Its first move as blue: ${where}`;
    return `After ${who}’s move at ${where}`;
  }
  function label(ctx, s, x, y, u, w) { K().text(ctx, s, x, y, { size: u * 0.9, w: 650, color: theme.c('muted'), maxW: w }); }
  // lines from the hidden neurons to an output, each as strong as what that neuron adds to it (blue raises it, orange lowers it)
  function contributions(ctx, pts, vals, to) {
    let m = 1e-9; for (const v of vals) m = Math.max(m, Math.abs(v));
    vals.forEach((v, j) => {
      if (!v) return;
      ctx.strokeStyle = theme.c(v > 0 ? 'pos' : 'neg', 0.08 + 0.55 * Math.abs(v) / m); ctx.lineWidth = 0.6 + 1.6 * Math.abs(v) / m;
      ctx.beginPath(); ctx.moveTo(pts[j][0], pts[j][1]); ctx.lineTo(to[0], to[1]); ctx.stroke();
    });
  }
  // a grid of neurons as dots: ReLU outputs (0 and up) in one colour, tanh outputs (−1 to 1) in two
  function dots(ctx, h, r, cols, signed) {
    const rows = Math.ceil(h.length / cols), s = Math.min(r.w / cols, r.h / rows), x0 = r.x + (r.w - s * cols) / 2, pts = [];
    let m = 1e-9; for (const v of h) m = Math.max(m, Math.abs(v));
    h.forEach((v, j) => {
      const x = x0 + (j % cols + 0.5) * s, y = r.y + (Math.floor(j / cols) + 0.5) * s;
      pts.push([x, y]);
      ctx.beginPath(); ctx.arc(x, y, s * 0.36, 0, Math.PI * 2);
      if (signed) { ctx.fillStyle = theme.divColor(clamp(v, -1, 1) * 0.95, 0.95); ctx.fill(); }
      else if (v > 0) { ctx.fillStyle = theme.c('s1', 0.18 + 0.82 * v / m); ctx.fill(); }
      ctx.strokeStyle = theme.c('line-2'); ctx.lineWidth = 0.8; ctx.stroke();
    });
    return { pts, s, h: rows * s };
  }
  // one row per move: its name, its value, a bar around zero; the chosen one bold and outlined
  function qRows(ctx, q, a, r, u) {
    let m = 1e-9; for (const v of q) m = Math.max(m, Math.abs(v));
    const rh = r.h / 4, ys = [];
    for (let k = 0; k < 4; k++) {
      const y = r.y + k * rh, mid = r.x + r.w / 2, bw = (r.w / 2 - 4) * Math.abs(q[k]) / m;
      ys.push(y + rh * 0.5);
      if (k === a) { ctx.fillStyle = theme.c('focus', 0.1); ctx.fillRect(r.x - 3, y + 1, r.w + 6, rh - 2); ctx.strokeStyle = theme.c('focus'); ctx.lineWidth = 1.5; ctx.strokeRect(r.x - 3 + 0.5, y + 1.5, r.w + 5, rh - 3); }
      K().text(ctx, `${ARROW[k]} ${DIR[k]}`, r.x + 2, y + rh * 0.32, { size: u * 0.92, w: k === a ? 700 : 500, maxW: r.w * 0.6 });
      K().text(ctx, fmt(q[k], 2), r.x + r.w - 2, y + rh * 0.32, { size: u * 0.92, w: k === a ? 700 : 500, align: 'right', mono: true });
      ctx.fillStyle = theme.c(q[k] >= 0 ? 'pos' : 'neg', 0.7); ctx.fillRect(q[k] >= 0 ? mid : mid - bw, y + rh * 0.6, bw, Math.max(3, rh * 0.18));
      ctx.strokeStyle = theme.c('line-2'); ctx.lineWidth = 1; ctx.beginPath(); ctx.moveTo(mid, y + rh * 0.55); ctx.lineTo(mid, y + rh * 0.85); ctx.stroke();
    }
    return ys;
  }
  function insideMazeNet(ctx, B, P, u) {
    const ci = { x: B.x, y: B.y, w: B.w * 0.28, h: B.h }, ch = { x: B.x + B.w * 0.33, y: B.y, w: B.w * 0.32, h: B.h }, co = { x: B.x + B.w * 0.71, y: B.y, w: B.w * 0.29, h: B.h };
    label(ctx, 'It sees', ci.x, ci.y + u * 0.5, u, ci.w);
    const v = drawView(ctx, P.x, { x: ci.x, y: ci.y + u * 1.3, w: ci.w, h: Math.min(ci.w, ci.h * 0.56) });
    drawGoalIns(ctx, P.x, { x: ci.x, y: ci.y + u * 1.6 + v.h, w: ci.w, h: Math.min(ci.h - v.h - u * 1.8, u * 3.6) }, u);
    const active = P.h.filter((t) => t > 0).length;
    label(ctx, '64 neurons (ReLU)', ch.x, ch.y + u * 0.5, u, ch.w);
    const qr = { x: co.x, y: co.y + u * 1.3, w: co.w, h: Math.min(co.h - u * 1.4, u * 13) };
    // the hidden neurons' pull on the chosen move: drawn first, under the dots
    const hr = { x: ch.x, y: ch.y + u * 1.3, w: ch.w, h: ch.h - u * 1.4 };
    const ctr = Array.from(P.h, (t, j) => t * P.net.W2[j * 4 + P.a]), probeDots = { pts: [] };
    { const rows = 8, s = Math.min(hr.w / 8, hr.h / rows), x0 = hr.x + (hr.w - s * 8) / 2; for (let j = 0; j < 64; j++) probeDots.pts.push([x0 + (j % 8 + 0.5) * s, hr.y + (Math.floor(j / 8) + 0.5) * s]); }
    contributions(ctx, probeDots.pts, ctr, [qr.x - 4, qr.y + (P.a + 0.5) * qr.h / 4]);
    const dd = dots(ctx, P.h, hr, 8, false);
    K().text(ctx, `${active} of 64 on`, hr.x + hr.w / 2, hr.y + dd.h + u * 0.9, { size: u * 0.85, color: theme.c('ink-2'), align: 'center', maxW: hr.w });
    label(ctx, 'Q-values', co.x, co.y + u * 0.5, u, co.w);
    qRows(ctx, P.q, P.a, qr, u);
    K().arrow(ctx, ci.x + ci.w + 3, ci.y + u * 1.3 + v.h / 2, hr.x - 2, ci.y + u * 1.3 + v.h / 2, { lw: 1.1, head: 5 });
    K().text(ctx, goalText(P.x), B.x, B.y + B.h - u * 0.4, { size: u * 0.85, color: theme.c('ink-2'), maxW: B.w * 0.62 });
  }
  function insideMazeTable(ctx, B, P, u, S) {
    const open = []; for (let i = 0; i < P.M.n; i++) if (P.M.cell[i] === '.' || P.M.cell[i] === 'S') open.push(i);
    const cm = { x: B.x, y: B.y, w: B.w * 0.34, h: B.h }, ct = { x: B.x + B.w * 0.42, y: B.y, w: B.w * 0.16, h: B.h }, cq = { x: B.x + B.w * 0.66, y: B.y, w: B.w * 0.34, h: B.h };
    label(ctx, 'Its square', cm.x, cm.y + u * 0.5, u, cm.w);
    miniMaze(ctx, P.M, { x: cm.x, y: cm.y + u * 1.3, w: cm.w, h: Math.min(cm.h * 0.6, cm.w * 0.8) }, { agent: P.s });
    label(ctx, `${open.length} rows × 4`, ct.x, ct.y + u * 0.5, u, ct.w + 20);
    // the table: one thin row per square, this square's row lit
    const tr = { x: ct.x, y: ct.y + u * 1.3, w: ct.w, h: ct.h - u * 1.6 }, rh = tr.h / open.length;
    let vmax = 0.05; for (const i of open) vmax = Math.max(vmax, Math.abs(S.Q.max(i)));
    open.forEach((i, k) => { for (let a = 0; a < 4; a++) { ctx.fillStyle = theme.divColor(clamp(S.Q.Q[i * 4 + a] / vmax, -1, 1) * 0.9, 0.85); ctx.fillRect(tr.x + a * tr.w / 4, tr.y + k * rh, tr.w / 4 - 0.5, Math.max(0.8, rh - 0.5)); } });
    const k = open.indexOf(P.s);
    if (k >= 0) { ctx.strokeStyle = theme.c('focus'); ctx.lineWidth = 2; ctx.strokeRect(tr.x - 2, tr.y + k * rh - 1, tr.w + 4, rh + 2); K().arrow(ctx, tr.x + tr.w + 4, tr.y + (k + 0.5) * rh, cq.x - 6, cq.y + u * 1.3 + Math.min(cq.h - u * 1.4, u * 13) * (P.a + 0.5) / 4, { lw: 1.1, head: 5 }); }
    label(ctx, 'Its row: Q-values', cq.x, cq.y + u * 0.5, u, cq.w);
    qRows(ctx, P.q, P.a, { x: cq.x, y: cq.y + u * 1.3, w: cq.w, h: Math.min(cq.h - u * 1.4, u * 13) }, u);
    K().text(ctx, 'No layers: the square picks a row, the largest number its move.', B.x, B.y + B.h - u * 0.4, { size: u * 0.85, color: theme.c('ink-2'), maxW: B.w });
  }
  function insideGameNet(ctx, B, P, u) {
    const g = P.g, ci = { x: B.x, y: B.y, w: B.w * 0.26, h: B.h }, ch = { x: B.x + B.w * 0.34, y: B.y, w: B.w * 0.3, h: B.h }, co = { x: B.x + B.w * 0.72, y: B.y, w: B.w * 0.28, h: B.h };
    label(ctx, `Its inputs (${g.n * 2})`, ci.x, ci.y + u * 0.5, u, ci.w + 12);
    const bh = Math.min(ci.w, (ci.h - u * 4) / 2);
    [0, 1].forEach((p) => {
      const y = ci.y + u * 1.4 + p * (bh + u * 1.4);
      K().text(ctx, p ? 'other stones' : `${STONE[P.mover]}’s stones`, ci.x, y + u * 0.3, { size: u * 0.8, color: theme.c('ink-2'), maxW: ci.w });
      const cs = bh / g.size, x0 = ci.x + (ci.w - cs * g.size) / 2;
      for (let i = 0; i < g.n; i++) { const on = P.x[p * g.n + i]; ctx.fillStyle = on ? theme.c(p ? 's2' : 's1', 0.85) : theme.c('line', 0.45); ctx.fillRect(x0 + (i % g.size) * cs + 1, y + u * 0.9 + Math.floor(i / g.size) * cs + 1, cs - 2, cs - 2); if (i === P.m) { ctx.strokeStyle = theme.c('focus'); ctx.lineWidth = 1.5; ctx.strokeRect(x0 + (i % g.size) * cs + 0.5, y + u * 0.9 + Math.floor(i / g.size) * cs + 0.5, cs - 1, cs - 1); } }
    });
    const H = P.h.length, HC = Math.ceil(Math.sqrt(H)), hr = { x: ch.x, y: ch.y + u * 1.4, w: ch.w, h: ch.h - u * 4 };
    label(ctx, `${H} neurons (tanh)`, ch.x, ch.y + u * 0.5, u, ch.w);
    const rows = Math.ceil(H / HC), s = Math.min(hr.w / HC, hr.h / rows), x0 = hr.x + (hr.w - s * HC) / 2, pts = [];
    for (let j = 0; j < H; j++) pts.push([x0 + (j % HC + 0.5) * s, hr.y + (Math.floor(j / HC) + 0.5) * s]);
    const out = [co.x + co.w * 0.1, co.y + u * 1.4 + Math.min(co.h * 0.3, u * 4)];
    contributions(ctx, pts, P.h.map((t, j) => t * P.net.W2[j]), out);
    dots(ctx, P.h, hr, HC, true);
    label(ctx, 'Chance to win', co.x, co.y + u * 0.5, u, co.w + 8);
    const big = clamp(co.w * 0.28, 14, 30);
    K().text(ctx, P.win ? '100%' : fmtPct(P.v, 0), co.x + co.w * 0.1, out[1], { size: big, w: 750, color: theme.c('ink') });
    K().text(ctx, `for ${STONE[P.mover]}`, co.x + co.w * 0.1, out[1] + big * 0.95, { size: u * 0.85, color: theme.c('ink-2'), maxW: co.w * 0.9 });
    const by = out[1] + big * 0.95 + u * 1.2, bw = co.w * 0.8;
    ctx.fillStyle = theme.c('line', 0.6); ctx.fillRect(co.x + co.w * 0.1, by, bw, 6); ctx.fillStyle = theme.c('s1'); ctx.fillRect(co.x + co.w * 0.1, by, bw * (P.win ? 1 : P.v), 6);
    if (P.win) K().text(ctx, `a win: it counts 1 without asking the network (which says ${fmtPct(P.v, 0)})`, co.x + co.w * 0.1, by + u * 1.6, { size: u * 0.8, color: theme.c('ink-2'), maxW: co.w * 0.95 });
    K().arrow(ctx, ci.x + ci.w + 3, B.y + B.h * 0.42, hr.x - 2, B.y + B.h * 0.42, { lw: 1.1, head: 5 });
    K().text(ctx, 'line thickness: how much each neuron adds (blue) or takes away (orange)', B.x, B.y + B.h - u * 0.4, { size: u * 0.8, color: theme.c('ink-2'), maxW: B.w });
  }
  function insideGameTable(ctx, B, P, u) {
    const cb = { x: B.x, y: B.y, w: B.w * 0.36, h: B.h }, ct = { x: B.x + B.w * 0.42, y: B.y, w: B.w * 0.58, h: B.h };
    label(ctx, 'The position it looks up', cb.x, cb.y + u * 0.5, u, cb.w + 30);
    miniBoard(ctx, P.g, P.b, { x: cb.x, y: cb.y + u * 1.4, w: cb.w, h: Math.min(cb.w, cb.h - u * 2) }, P.m);
    const lines = [`seen by ${STONE[P.mover]} (x theirs, o the other’s),`, 'turned to the first of its 8 versions:', P.key, '',
      P.win ? 'a win: worth 1, no need to look it up' : P.known ? `its entry: ${fmt(P.v, 3)}` : 'not in the table yet: 0.5, a coin flip', `${fmtInt(P.table.size)} entries in the table`];
    K().lines(ctx, lines, { x: ct.x, y: ct.y + u * 1.4, w: ct.w, h: Math.min(ct.h - u * 1.6, u * 10) }, { size: u * 0.9, mono: true, strong: (i) => i === 4 });
  }
  /* Draws the picture into r; returns what it showed. */
  function drawInside(ctx, r, L) {
    const P = probe(L), pad = 10, u = clamp(Math.min(r.w / 30, r.h / 17), 10.5, 13);
    K().box(ctx, r, { fill: theme.c('panel-2'), stroke: theme.c('line') });
    const head = { mazeNet: 'Inside its Q-network', mazeTable: 'Inside its Q-table', gameNet: 'Inside its value network', gameTable: 'Inside its value table' }[P.kind];
    K().text(ctx, head, r.x + pad, r.y + pad + u * 0.6, { size: u * 1.05, w: 700, maxW: r.w - 2 * pad });
    K().text(ctx, contextLine(L, P), r.x + pad, r.y + pad + u * 1.95, { size: u * 0.92, color: theme.c('ink-2'), maxW: r.w - 2 * pad });
    const B = { x: r.x + pad, y: r.y + pad + u * 3.2, w: r.w - 2 * pad, h: r.h - 2 * pad - u * 3.2 };
    if (B.w < 120 || B.h < 80) return P;
    if (P.kind === 'mazeNet') insideMazeNet(ctx, B, P, u);
    else if (P.kind === 'mazeTable') insideMazeTable(ctx, B, P, u, L.S);
    else if (P.kind === 'gameNet') insideGameNet(ctx, B, P, u);
    else insideGameTable(ctx, B, P, u);
    return P;
  }

  // ------------------------------------------------------------------ the models
  const QRECT = [0, 0.02, 1, 0.96];
  function mazeTableTree(L, size, P) {
    const S = L.S, M = S.maze, Q = S.Q, X = { wide: size.w / size.h >= 1.05 }, open = M.cell.filter((c) => c === '.' || c === 'S').length;
    const square = (s) => {
      const x = s % M.W, y = Math.floor(s / M.W), vals = [0, 1, 2, 3].map((a) => Q.Q[s * 4 + a]), best = Q.best(s);
      return D.lazy({
        title: `Square (${x}, ${y})`, sub: `best: ${DIR[best]} (${fmt(vals[best], 3)})`, tone: 's1',
        summary: (ctx, r, Sk) => Sk.bars(ctx, Float32Array.from(vals), r),
        inner: (ctx, r, Sk) => {
          const u = Sk.u(r, 8, 13);
          Sk.lines(ctx, [...vals.map((v, a) => `Q(${x}, ${y}, ${DIR[a].padEnd(5)}) = ${fmt(v, 4)}${a === best ? '   ← its best move' : M.move(s, a) === s ? '   (a wall: it stays)' : ''}`), '',
            'after a move:  Q ← Q + α·(r + γ·max Q(next square) − Q)', `α = ${fmt(S.o.alpha, 2)}, γ = ${fmt(S.o.gamma, 2)}, visited ${fmtInt(Q.seen[s])} times`], Sk.place(r, [0, 0, 1, 0.48]), { mono: true, size: u, strong: (i) => i === best });
          Sk.caption(ctx, 'its four values, one per move: zoom into one to see the number as it is stored', Sk.place(r, [0, 0.62, 1, 0.2]), { size: u * 0.8, minW: 220 });
          Sk.bars(ctx, Float32Array.from(vals), Sk.place(r, [0, 0.62, 1, 0.2]));
        },
        grids: [{ id: 'q', rect: [0, 0.62, 1, 0.2], rows: 1, cols: 4, make: (i, a) => D.stored(X, vals[a], `Q(${x}, ${y}, ${DIR[a]})`, 'weight') }],
        about: [`The four entries of the Q-table for square (${x}, ${y}): the reward it expects if it moves ${DIR.join(', ')} from here and then plays its best. It moves ${DIR[best]}, the largest. Moving into a wall leaves it where it is, which only costs a step.`,
          'Each entry changes only when it makes that move from this square: the entry moves a fraction α of the way toward the reward it got plus γ times the best entry of the square it landed on. So values flow backward, one square per visit, from the goal and the pits.'],
        code: `# Q[(x, y)] holds the values of the moves up, right, down, left from that square\nQ = {(${x}, ${y}): [${vals.map((v) => NA.fmtCode(v, 4)).join(', ')}]}\nmoves = ['up', 'right', 'down', 'left']\nq = Q[(${x}, ${y})]\nprint('best move:', moves[q.index(max(q))])`,
      });
    };
    return {
      id: 'model', title: 'Q-table', sub: `${open} squares × 4 moves = ${open * 4} numbers`, tone: 's1',
      summary: (ctx, r) => { const cs = Math.min(r.w / M.W, r.h / M.H), x0 = r.x + (r.w - cs * M.W) / 2; let vmax = 0.05; for (let i = 0; i < M.n; i++) if (M.cell[i] !== '#') vmax = Math.max(vmax, Math.abs(Q.max(i))); for (let i = 0; i < M.n; i++) { ctx.fillStyle = M.cell[i] === '#' ? theme.c('line-2') : theme.divColor(clamp(Q.max(i) / vmax, -1, 1) * 0.9, 0.6); ctx.fillRect(x0 + (i % M.W) * cs, r.y + Math.floor(i / M.W) * cs, cs - 0.5, cs - 0.5); } },
      inner: (ctx, c) => {
        const gr = NA.archKit.place(c, QRECT), cs = Math.min(gr.w / M.W, gr.h / M.H);
        let vmax = 0.05; for (let i = 0; i < M.n; i++) if (M.open(i)) vmax = Math.max(vmax, Math.abs(Q.max(i)));
        for (let i = 0; i < M.n; i++) {
          const x = gr.x + (i % M.W) * cs, y = gr.y + Math.floor(i / M.W) * cs, ch = M.cell[i];
          ctx.fillStyle = ch === '#' ? theme.c('line-2') : ch === 'G' ? theme.c('pos', 0.3) : ch === 'X' ? theme.c('neg', 0.3) : theme.divColor(clamp(Q.max(i) / vmax, -1, 1) * 0.9, 0.35);
          ctx.fillRect(x, y, cs, cs); ctx.strokeStyle = theme.c('line'); ctx.strokeRect(x + 0.5, y + 0.5, cs - 1, cs - 1);
          if (i === P.s) { ctx.strokeStyle = theme.c('s7'); ctx.lineWidth = 2.5; ctx.strokeRect(x + 2, y + 2, cs - 4, cs - 4); ctx.lineWidth = 1; }
          if (ch === 'G' || ch === 'X') { NA.archKit.text(ctx, ch === 'G' ? '+1' : '−1', x + cs / 2, y + cs / 2, { size: cs * 0.28, w: 700, align: 'center', color: theme.c(ch === 'G' ? 'pos' : 'neg') }); continue; }
          if (ch === '#' || cs < 40) continue;
          // the four values at the four sides of the square
          const f = clamp(cs * 0.15, 7, 11);
          [[0.5, 0.2], [0.8, 0.5], [0.5, 0.8], [0.2, 0.5]].forEach(([fx, fy], a) => NA.archKit.text(ctx, fmt(Q.Q[i * 4 + a], 2), x + cs * fx, y + cs * fy, { size: f, mono: true, align: 'center', w: a === Q.best(i) ? 700 : 500, color: theme.c(a === Q.best(i) ? 'ink' : 'ink-2') }));
        }
      },
      grids: [{ id: 'sq', rect: QRECT, rows: M.H, cols: M.W, square: true, skip: (i, j) => !(M.cell[i * M.W + j] === '.' || M.cell[i * M.W + j] === 'S'), make: (i, j) => square(i * M.W + j) }],
      about: [`Everything the agent knows: ${open * 4} numbers, four for each square it can stand on (up, right, down, left), drawn at the square's four sides. The largest of the four is the move it takes there. The square outlined is the one being looked at. ${NA.openHint('a square', { cap: true })}.`,
        'There is no neural network here: a table is enough when there are few situations, but it fits one maze only, since its rows are this maze’s squares. Switch the model to Network for one that sees what is around it and so finds its way through mazes it never met.'],
      code: NA.archRL.py.maze(L),
    };
  }

  function mazeNetTree(L, size, P) {
    const net = P.net, H = net.nH, X = { wide: size.w / size.h >= 1.05 }, x01 = new Float64Array(net.nIn);
    for (const i of P.x) x01[i] = 1;
    const name = (i) => RL.MazeNet.inputName(i);
    const hidden = (j) => D.neuron(X, {
      title: `Hidden neuron ${j + 1}`, sub: `ReLU: ${fmt(P.h[j], 3)}${P.h[j] > 0 ? '' : ' (off)'}`, tone: 's1',
      terms: Array.from({ length: net.nIn }, (_, i) => ({ a: x01[i], w: net.W1[i * H + j] })), bias: net.b1[j], act: RELU,
      inLabel: (i) => `input ${i + 1}: ${name(i)}`, wLabel: (i) => `weight from input ${i + 1} (${name(i)})`, inWhat: `the ${net.nIn} inputs`,
      vars: { w: `net[0].weight[${j}]`, a: 'x', b: `net[0].bias[${j}]` }, codeWhat: `as one row of nn.Linear(${net.nIn}, ${H})`, outName: 'h',
      about: [`One of the ${H} hidden neurons, on square ${sqName(P.M, P.s)}. Its inputs are the ${net.nIn} inputs, 1 for what is there and 0 for what is not, so it adds up the weights of the things it sees (${P.x.length} of them here) plus its bias. ReLU keeps a positive sum and turns a negative one into 0: this neuron is ${P.h[j] > 0 ? `on, at ${fmt(P.h[j], 3)}` : 'off here'}.`,
        'Nobody told it what to look for. Training shaped each neuron into a detector of some pattern (a wall ahead, a pit beside it, the goal to the right and an open square that way) that helps tell the moves apart.'],
    });
    const out = (a) => Object.assign(D.neuron(X, {
      title: `Q(${DIR[a]})`, sub: `${fmt(P.q[a], 3)}${a === P.a ? ': its move' : ''}`, tone: 's4',
      terms: Array.from(P.h, (t, j) => ({ a: t, w: net.W2[j * 4 + a] })), bias: net.b2[a], act: LINEAR,
      inLabel: (j) => `hidden neuron ${j + 1}`, wLabel: (j) => `weight from hidden neuron ${j + 1}`, inWhat: `the ${H} hidden neurons`,
      vars: { w: `net[2].weight[${a}]`, a: 'h', b: `net[2].bias[${a}]` }, codeWhat: `as one row of nn.Linear(${H}, 4)`, outName: 'q',
      about: [`The output for moving ${DIR[a]}: its weighted sum of the ${H} hidden neurons, with no squashing, so it can be any number. It is the reward the network expects if it moves ${DIR[a]} now and plays its best afterwards: +1 for the goal, less 0.02 a step, −1 for a pit, discounted by γ each step.`, a === P.a ? 'It is the largest of the four, so this is the move it makes.' : `It is smaller than Q(${DIR[P.a]}), so it does not move ${DIR[a]}.`],
    }), { id: `q${a}` });
    const IN = [0, 0.05, 0.27, 0.93], HID = [0.35, 0.1, 0.3, 0.8], OUT = [0.75, 0.04, 0.25, 0.92];
    const plane = (p) => [IN[0] + (p % 2) * IN[2] * 0.52, IN[1] + 0.04 + Math.floor(p / 2) * IN[3] * 0.36, IN[2] * 0.46, IN[3] * 0.3];
    const PLANE = ['walls and edge', 'pits', 'goal', 'visited'], GOALR = [IN[0], IN[1] + IN[3] * 0.8, IN[2], IN[3] * 0.18];
    const planeGrid = (p) => ({ id: ['wall', 'pit', 'goal', 'seen'][p], rect: plane(p), rows: 5, cols: 5, square: true, skip: (i, j) => i === 2 && j === 2,
      make: (i, j) => { const kk = i * 5 + j, k = kk < 12 ? kk : kk - 1, idx = p * NV + k; return D.stored(X, x01[idx], `input ${idx + 1}: ${name(idx)}`, 'input'); } });
    return {
      id: 'model', title: 'Q-network', sub: `${net.nIn} inputs → ${H} ReLU → 4 Q-values, ${fmtInt(net.paramCount)} parameters`, tone: 's1',
      summary: (ctx, r) => { dots(ctx, P.h, { x: r.x + r.w * 0.3, y: r.y, w: r.w * 0.4, h: r.h }, 8, false); },
      inner: (ctx, c, Sk) => {
        const u = Sk.u(c, 8, 12);
        for (let p = 0; p < 4; p++) {
          const r = Sk.place(c, plane(p)), cs = Math.min(r.w, r.h) / 5;
          Sk.text(ctx, PLANE[p], r.x, r.y - 6, { size: u * 0.8, color: theme.c('muted'), maxW: r.w + 8 });
          for (let kk = 0; kk < 25; kk++) {
            const x = r.x + (kk % 5) * cs, y = r.y + Math.floor(kk / 5) * cs;
            if (kk === 12) { ctx.fillStyle = theme.c('s7'); ctx.beginPath(); ctx.arc(x + cs / 2, y + cs / 2, cs * 0.25, 0, Math.PI * 2); ctx.fill(); continue; }
            const on = x01[p * NV + (kk < 12 ? kk : kk - 1)];
            ctx.fillStyle = on ? theme.c(['ink-2', 'neg', 'pos', 's7'][p], 0.75) : theme.c('line', 0.45); ctx.fillRect(x + 1, y + 1, cs - 2, cs - 2);
          }
        }
        const gr = Sk.place(c, GOALR), cw = gr.w / 4, ch = gr.h / 2;
        Sk.text(ctx, 'goal direction (near, far)', gr.x, gr.y - 6, { size: u * 0.8, color: theme.c('muted'), maxW: gr.w + 8 });
        for (let i = 0; i < 2; i++) for (let j = 0; j < 4; j++) {
          const on = x01[4 * NV + j * 2 + i], x = gr.x + j * cw, y = gr.y + i * ch;
          ctx.fillStyle = on ? theme.c('pos', 0.75) : theme.c('line', 0.45); ctx.fillRect(x + 1, y + 1, cw - 2, ch - 2);
          if (ch > 12) Sk.text(ctx, ['→', '←', '↓', '↑'][j], x + cw / 2, y + ch / 2, { size: Math.min(ch * 0.6, 12), align: 'center', color: theme.c(on ? 'panel' : 'muted') });
        }
        const hr = Sk.place(c, HID), hs = Math.min(hr.w / 8, hr.h / 8);
        Sk.text(ctx, `${H} hidden neurons (ReLU): darker = larger`, hr.x, hr.y - 8, { size: u * 0.8, color: theme.c('muted'), maxW: hr.w + 30 });
        let m = 1e-9; for (const v of P.h) m = Math.max(m, v);
        for (let j = 0; j < H; j++) { const x = hr.x + (j % 8 + 0.5) * hs, y = hr.y + (Math.floor(j / 8) + 0.5) * hs; ctx.beginPath(); ctx.arc(x, y, hs * 0.36, 0, Math.PI * 2); if (P.h[j] > 0) { ctx.fillStyle = theme.c('s1', 0.18 + 0.82 * P.h[j] / m); ctx.fill(); } ctx.strokeStyle = theme.c('line-2'); ctx.lineWidth = 0.8; ctx.stroke(); }
        Sk.arrow(ctx, c.x + c.w * (IN[0] + IN[2]) + 4, c.y + c.h * 0.5, hr.x - 6, c.y + c.h * 0.5, { lw: 1.3 });
        Sk.arrow(ctx, hr.x + hs * 8 + 6, c.y + c.h * 0.5, c.x + c.w * OUT[0] - 6, c.y + c.h * 0.5, { lw: 1.3 });
      },
      grids: [0, 1, 2, 3].map(planeGrid).concat([
        { id: 'dir', rect: GOALR, rows: 2, cols: 4, make: (i, j) => { const idx = 4 * NV + j * 2 + i; return D.stored(X, x01[idx], `input ${idx + 1}: ${name(idx)}`, 'input'); } },
        { id: 'hid', rect: HID, rows: 8, cols: 8, square: true, make: (i, j) => hidden(i * 8 + j) },
      ]),
      children: [0, 1, 2, 3].map((a) => Object.assign(out(a), { rect: [OUT[0], OUT[1] + a * OUT[3] / 4 + 0.01, OUT[2], OUT[3] / 4 - 0.02] })),
      about: [`The network that chooses its moves, on square ${sqName(P.M, P.s)} of ${P.M === L.S.maze ? L.mazeName() : 'a random training maze'}: left, its ${net.nIn} inputs (the 24 squares around it, each as a wall or the edge, a pit, the goal, or visited; and 8 for the goal’s direction); the dots, its ${H} hidden neurons; right, its 4 outputs, one Q-value per move. It moves ${DIR[P.a]}, the largest.`,
        `It has no row for any square and has never seen this maze: the same weights decide in every maze, from what is around it. That is why it can carry over to new mazes, where a table cannot. ${NA.openHint('a part', { cap: true })}: every level opens, down to the transistors.`],
      code: NA.archRL.py.mazeNet(L),
    };
  }

  function threeTableTree(L, size) {
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
      id: 'model', title: 'Value table', sub: `${fmtInt(T.size)} positions`, tone: 's4',
      summary: (ctx, r) => miniBoard(ctx, g, b, r, -1),
      inner: (ctx, c, Sk) => {
        const u = Sk.u(c, 8, 13);
        miniBoard(ctx, g, b, Sk.place(c, [0, 0.04, 0.3, 0.46]), -1);
        Sk.lines(ctx, [pos.toMove ? `${STONE[mover]} to move` : 'an empty board: blue to move', 'right: the table entry for each move'], Sk.place(c, [0, 0.54, 0.3, 0.12]), { size: u * 0.9 });
      },
      // one box per move, four to a row on the right
      children: shown.map((m, k) => { const rows = Math.ceil(shown.length / 4), cw = 0.67 / 4, rh = 0.96 / rows; return Object.assign(entry(k), { id: `m${m}`, rect: [0.33 + (k % 4) * cw + 0.006, 0.02 + Math.floor(k / 4) * rh + 0.01, cw - 0.012, rh - 0.02] }); }),
      about: [`One entry per position it has met in self-play: ${fmtInt(T.size)} so far. To move, it looks up the position each empty cell leads to and takes the best. Each box on the right is one move from the position on the left, with its entry.`,
        'A table learns exactly, but only about positions it has met; it knows nothing about a position one stone away from one it has seen. Switch the model to Network: it judges any position from the pattern of stones, and on this board it learns faster than the table.'],
      code: NA.archRL.py.three(L),
    };
  }

  // Three-line's or Four-line's value network, on the position being looked at
  function valueNetTree(L, size, P) {
    const g = P.g, net = P.net, X = { wide: size.w / size.h >= 1.05 }, H = net.nH, nIn = net.nIn, HC = Math.ceil(Math.sqrt(H)), b = P.b, mover = P.mover, x = P.x, h = P.h, v = P.v;
    const inName = (i) => `${i < g.n ? 'their stone' : 'other stone'} at ${cellName(g, i % g.n)}`;
    const hidden = (j) => D.neuron(X, {
      title: `Hidden neuron ${j + 1}`, sub: `tanh: ${fmt(h[j], 3)}`, tone: 's1',
      terms: Array.from({ length: nIn }, (_, i) => ({ a: x[i], w: net.W1[i * H + j] })), bias: net.b1[j], act: TANH,
      inLabel: (i) => inName(i), wLabel: (i) => `weight from ${inName(i)}`, inWhat: `the board, 2 inputs per cell`,
      vars: { w: `net[0].weight[${j}]`, a: 'x', b: `net[0].bias[${j}]` }, codeWhat: `as one row of nn.Linear(${nIn}, ${H})`, outName: 'h',
      about: [`One of the network’s ${H} hidden neurons, on the position after ${STONE[mover]}’s move at ${cellName(g, P.m)} (seen by ${STONE[mover]}). Its inputs are the ${nIn} board inputs, 1 where a stone is and 0 elsewhere, so only the weights of occupied cells count: it adds up the weights of the stones on the board.`],
    });
    const out = Object.assign(D.neuron(X, {
      title: 'Output: the chance to win', sub: `sigmoid: ${fmtPct(v, 1)}`, tone: 's4',
      terms: h.map((hj, j) => ({ a: hj, w: net.W2[j] })), bias: net.b2, act: SIGMOID,
      inLabel: (j) => `hidden neuron ${j + 1}`, wLabel: (j) => `weight from hidden neuron ${j + 1}`, inWhat: `the ${H} hidden neurons`,
      vars: { w: 'net[2].weight[0]', a: 'h', b: 'net[2].bias[0]' }, codeWhat: `as nn.Linear(${H}, 1) and a sigmoid`, outName: 'p',
      about: [`The output neuron weighs the ${H} hidden neurons and squashes the sum with a sigmoid into a chance between 0 and 1: ${fmtPct(v, 1)} that ${STONE[mover]}, who moved last, goes on to win this position.`],
    }), { id: 'out' });
    const IN = [0, 0.04, 0.24, 0.92], HID = [0.32, 0.12, 0.4, 0.76], OUT = [0.8, 0.36, 0.2, 0.28];
    return {
      id: 'model', title: 'Value network', sub: `${nIn} → ${H} tanh → 1 sigmoid, ${fmtInt(net.paramCount)} parameters`, tone: 's1',
      summary: (ctx, r) => { dots(ctx, h, { x: r.x + r.w * 0.3, y: r.y, w: r.w * 0.4, h: r.h }, HC, true); },
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
      about: [`The network that judges ${g.name} positions, on the position after ${STONE[mover]}’s move at ${cellName(g, P.m)}, as ${STONE[mover]} sees it: the left squares are its ${nIn} inputs, the dots its ${H} hidden neurons (blue positive, orange negative), and the box its output, the chance to win: ${fmtPct(v, 1)}.`,
        `To choose a move it runs this for every empty cell, on the position that move would make, and plays the highest. Self-play nudges every weight after each move, so patterns that led to wins (lines with room to grow, blocks) raise the output. ${NA.openHint('a neuron', { cap: true })}: it opens down to the transistors.`],
      code: g.id === 'three' ? NA.archRL.py.threeNet(L) : NA.archRL.py.four(L),
    };
  }

  // ------------------------------------------------------------------ the agent's loop: the top level
  function seeNode(L, P) {
    const S = L.S;
    if (P.kind === 'mazeNet') return {
      title: 'What it sees', sub: `${P.x.length} of 104 inputs on`, tone: 's7',
      summary: (ctx, r) => { drawView(ctx, P.x, { x: r.x, y: r.y, w: r.w, h: r.h }); },
      inner: (ctx, c, Sk) => {
        const u = Sk.u(c, 9, 13);
        miniMaze(ctx, P.M, Sk.place(c, [0, 0.06, 0.44, 0.7]), { agent: P.s, seen: P.seen, view: true });
        Sk.arrow(ctx, c.x + c.w * 0.47, c.y + c.h * 0.4, c.x + c.w * 0.55, c.y + c.h * 0.4, { lw: 1.3 });
        const v = drawView(ctx, P.x, Sk.place(c, [0.58, 0.06, 0.26, 0.5]));
        drawGoalIns(ctx, P.x, { x: c.x + c.w * 0.58, y: v.y0 + v.h + u, w: c.w * 0.26, h: u * 3.6 }, u);
        Sk.lines(ctx, [`square ${sqName(P.M, P.s)}; ${goalText(P.x)}`, `${P.x.length} inputs are 1, the other ${104 - P.x.length} are 0`], Sk.place(c, [0, 0.82, 1, 0.16]), { size: u });
      },
      about: ['It does not know which maze it is in or which square it is on. It sees the 24 squares around it, 2 in every direction (the dashed square on the maze), each as 4 inputs: a wall or the edge, a pit, the goal, or a square it has already been on this run. 8 more inputs say which way the goal lies, 1–2 squares away or further. 104 inputs in all, each 0 or 1.',
        'The same surroundings give the same inputs in any maze: that is what lets what it learned carry over to mazes it has never seen. Remembering where it has been keeps it from walking in circles.'],
      code: `W, H, VIEW = 8, 6, 2\n\ndef inputs(maze, x, y, seen, goal):\n    """The 104 inputs: 4 for each of the 24 squares around (x, y), then 8 for the goal's direction."""\n    planes = [[], [], [], []]                            # wall or edge, pit, goal, visited\n    for dy in range(-VIEW, VIEW + 1):\n        for dx in range(-VIEW, VIEW + 1):\n            if dx == 0 and dy == 0:\n                continue                                 # its own square is not an input\n            X, Y = x + dx, y + dy\n            c = maze[Y][X] if 0 <= X < W and 0 <= Y < H else '#'\n            planes[0].append(c == '#'); planes[1].append(c == 'X'); planes[2].append(c == 'G')\n            planes[3].append(c not in '#XG' and (X, Y) in seen)\n    gx, gy = goal[0] - x, goal[1] - y\n    goal_dir = [0 < gx <= 2, gx > 2, -2 <= gx < 0, gx < -2, 0 < gy <= 2, gy > 2, -2 <= gy < 0, gy < -2]\n    return [float(v) for p in planes for v in p] + [float(v) for v in goal_dir]`,
    };
    if (P.kind === 'mazeTable') return {
      title: 'What it sees', sub: `its square, ${sqName(P.M, P.s)}`, tone: 's7',
      summary: (ctx, r) => { miniMaze(ctx, P.M, r, { agent: P.s }); },
      inner: (ctx, c, Sk) => { miniMaze(ctx, P.M, Sk.place(c, [0.1, 0.05, 0.8, 0.72]), { agent: P.s, mark: P.s }); Sk.lines(ctx, [`square ${sqName(P.M, P.s)}: that is all it knows`, 'it picks one row of the table'], Sk.place(c, [0, 0.82, 1, 0.16]), { size: Sk.u(c, 9, 13) }); },
      about: ['The table’s input is the square the agent is on, nothing else: each square has its own row of values. So a table only fits the maze it learned: in another maze, the same square has other walls around it and another way to the goal.'],
      code: 'state = (x, y)          # the square: the row of the table to look up',
    };
    const g = P.g;
    return {
      title: 'What it sees', sub: P.kind === 'gameNet' ? `the board: ${g.n * 2} inputs` : 'the position, as a key', tone: 's7',
      summary: (ctx, r) => { miniBoard(ctx, g, P.b, r, P.m); },
      inner: (ctx, c, Sk) => {
        const u = Sk.u(c, 9, 13);
        miniBoard(ctx, g, P.b, Sk.place(c, [0, 0.05, 0.42, 0.7]), P.m);
        Sk.lines(ctx, P.kind === 'gameNet' ? [`the position after ${STONE[P.mover]}’s move,`, `as ${STONE[P.mover]} sees it:`, `${g.n} inputs: 1 where ${STONE[P.mover]} has a stone`, `${g.n} inputs: 1 where the other player has one`] : [`the position after ${STONE[P.mover]}’s move,`, 'as a string: x theirs, o the other’s,', 'turned and mirrored to the first', 'of its 8 versions:', P.key],
          Sk.place(c, [0.46, 0.06, 0.54, 0.7]), { size: u, mono: P.kind !== 'gameNet' });
      },
      about: [P.kind === 'gameNet' ? `It judges positions, not moves: for each empty cell it looks at the position that move would make (its afterstate), as the player who would make it sees it: ${g.n} inputs for that player’s stones, ${g.n} for the other’s.` : 'It judges positions, not moves: for each empty cell it looks up the position that move would make, as the player who would make it sees it. Rotations and mirror images count as one position.'],
      code: P.kind === 'gameNet' ? `def features(board, me):\n    x = [0.0] * (2 * N * N)\n    for i, v in enumerate(board):\n        if v:\n            x[i if v == me else N * N + i] = 1.0\n    return x` : `key = position_key(board, me)    # the same for all 8 rotations and mirror images`,
    };
  }
  function pickNode(L, P) {
    const S = L.S, eps = S.o.eps;
    if (P.kind === 'mazeNet' || P.kind === 'mazeTable') return {
      title: 'Its choice', sub: `${DIR[P.a]}: the largest Q-value`, tone: 's4',
      summary: (ctx, r) => { qRows(ctx, P.q, P.a, r, clamp(r.h / 7, 7, 11)); },
      inner: (ctx, c, Sk) => { const u = Sk.u(c, 9, 13); qRows(ctx, P.q, P.a, Sk.place(c, [0.05, 0.04, 0.5, 0.7]), u); Sk.lines(ctx, ['move = argmax Q', `while training: a random move`, `${fmtPct(eps, 0)} of the time (ε)`], Sk.place(c, [0.6, 0.1, 0.4, 0.4]), { size: u, mono: true }); },
      about: [`It takes the move with the largest Q-value: ${DIR[P.a]} (${fmt(P.q[P.a], 3)}). While it trains, a share ε of its moves are random instead (exploring), so it keeps finding out about moves it rates low.${S.model === 'net' ? ' A network that has just started explores more: ε starts at 1 and falls over its first 24,000 mazes.' : ''} Testing and playing use no random moves.`],
      code: `move = max(range(4), key=lambda a: q[a])          # the largest Q-value\nif training and random.random() < eps:\n    move = random.randrange(4)                     # sometimes explore`,
    };
    const g = P.g, pos = P.pos, toMove = pos.toMove || P.mover, mv = RL.moveValues(g, S.brain, P.why === 'last' || P.why === 'first' ? P.pos.b : P.pos.b, toMove);
    const order = mv.moves.map((m, k) => [m, mv.vals[k]]).sort((a, b) => b[1] - a[1]).slice(0, 6);
    return {
      title: 'Its choice', sub: order.length ? `${STONE[toMove]}: ${cellName(g, order[0][0])}` : 'the game is over', tone: 's4',
      summary: (ctx, r) => { miniBoard(ctx, g, pos.b, r, order.length ? order[0][0] : -1); },
      inner: (ctx, c, Sk) => { Sk.lines(ctx, order.length ? [`${STONE[toMove]} to move: its best moves`, ...order.map(([m, v], k) => `${cellName(g, m).padEnd(18)} ${fmtPct(v, 0)}${k === 0 ? '  ← it plays this' : ''}`)] : ['no moves left'], Sk.place(c, [0, 0.04, 1, 0.9]), { size: Sk.u(c, 9, 13), mono: true, strong: (i) => i === 1 }); },
      about: [`For every empty cell it asks its model how good the position after that move is for ${STONE[toMove]}, and plays the best. A move that wins at once counts 1 without asking. While it trains by self-play, ${fmtPct(eps, 0)} of its moves are random instead (ε), to try new lines; against you it never explores.`],
      code: `def best_move(board, me):\n    vals = {m: move_value(board, m, me) for m in empties(board)}\n    return max(vals, key=vals.get)`,
    };
  }
  function worldNode(L, P) {
    const S = L.S;
    if (P.kind === 'mazeNet' || P.kind === 'mazeTable') {
      const M = P.M;
      return {
        title: P.M === S.maze ? L.mazeName() : 'A random training maze', sub: '+1 goal · −1 pit · −0.02 a step', tone: 's2',
        summary: (ctx, r) => { miniMaze(ctx, M, r, { agent: P.s, move: P.a }); },
        inner: (ctx, c, Sk) => { miniMaze(ctx, M, Sk.place(c, [0, 0.04, 0.58, 0.8]), { agent: P.s, move: P.a, seen: P.seen }); Sk.lines(ctx, ['the goal: +1, the run ends', 'a pit: −1, the run ends', 'any other step: −0.02', 'a wall: it stays put', M.slip ? `slippery: ${fmtPct(M.slip, 0)} of moves go sideways` : 'the floor is not slippery', `shortest way through: ${RL.shortest(M)} steps`], Sk.place(c, [0.62, 0.06, 0.38, 0.7]), { size: Sk.u(c, 9, 13) }); },
        about: ['The world answers each move with a reward and the next square. The goal gives +1 and a pit −1, both ending the run; every other step costs 0.02, so a shorter way is worth more. Nothing tells the agent where the goal is or how to get there: it learns only from these numbers.', M.slip ? `The floor is slippery: ${fmtPct(M.slip, 0)} of moves go sideways instead, so routes near pits are risky.` : 'The floor is not slippery: every move goes where it is meant to.'],
        code: `def step(x, y, move):\n    nx, ny = x + MOVES[move][0], y + MOVES[move][1]\n    if not (0 <= nx < W and 0 <= ny < H) or MAZE[ny][nx] == '#':\n        nx, ny = x, y                     # a wall: stay put\n    cell = MAZE[ny][nx]\n    reward = 1.0 if cell == 'G' else -1.0 if cell == 'X' else -0.02\n    return (nx, ny), reward, cell in 'GX'`,
      };
    }
    const g = P.g, who = L.S.play ? 'you' : L.test ? 'the rule-follower' : 'itself';
    return {
      title: 'The game', sub: `${g.name} against ${who} · win 1, draw 0.5`, tone: 's2',
      summary: (ctx, r) => { miniBoard(ctx, g, P.pos.b, r, -1); },
      inner: (ctx, c, Sk) => { miniBoard(ctx, g, P.pos.b, Sk.place(c, [0, 0.04, 0.46, 0.8]), -1); Sk.lines(ctx, [`a ${g.size} × ${g.size} board, ${g.k} in a row wins`, 'a full board is a draw', `it plays against ${who}`, 'its reward: only the result,', 'win 1, draw 0.5, loss 0'], Sk.place(c, [0.5, 0.06, 0.5, 0.7]), { size: Sk.u(c, 9, 13) }); },
      about: [`The game is the world: after its move, the other player moves, and the next position is what it sees. It is never told what a good move is; the only reward is the result at the end: 1 for a win, 0.5 for a draw, 0 for a loss. ${L.S.play ? 'Now the other player is you.' : L.test ? 'In this test the other player is the rule-follower: it wins when it can, blocks when it must, and otherwise plays at random.' : 'In training the other player is itself (self-play), so it always has an opponent at its own level.'}`],
      code: `board[move] = me                      # the move\nif wins(board, move): reward = 1.0      # the only reward is the result\nelif not empties(board): reward = 0.5`,
    };
  }
  function learnNode(L, P) {
    const S = L.S, o = S.o;
    if (P.kind === 'mazeTable') {
      const u = S.update;
      return {
        title: 'Learning', sub: 'Q-learning: one entry per move', tone: 's3',
        summary: (ctx, r) => { K().lines(ctx, ['Q ← Q + α·(r + γ·max Q′ − Q)'], r, { mono: true, size: clamp(r.h / 3, 7, 11) }); },
        inner: (ctx, c, Sk) => { Sk.lines(ctx, ['after each move from s to s′ with reward r:', '  Q(s, a) ← Q(s, a) + α·(r + γ·max Q(s′, ·) − Q(s, a))', `α = ${fmt(o.alpha, 2)} (how far it moves), γ = ${fmt(o.gamma, 2)} (how much later counts)`, '', ...(u ? [`last update: at ${sqName(S.maze, u.s)}, ${DIR[u.intended]}, reward ${fmt(u.r, 2)}`, `  Q ${fmt(u.before, 3)} → ${fmt(u.after, 3)}`] : ['(tick Watch to see each update)'])], Sk.place(c, [0, 0.04, 1, 0.9]), { size: Sk.u(c, 9, 13), mono: true }); },
        about: ['After every move it changes one number: the entry for the square it left and the move it made moves a fraction α of the way toward the reward it got plus γ times the best entry of the square it reached. Nothing else changes, which is why a table learns one maze quickly and exactly, and knows nothing about the next.'],
        code: `target = r if done else r + gamma * max(Q[s2])\nQ[s][a] += alpha * (target - Q[s][a])`,
      };
    }
    if (P.kind === 'mazeNet') {
      const ag = S.agent;
      return {
        title: 'Learning', sub: 'DQN: replay memory, target network', tone: 's3',
        summary: (ctx, r) => { K().lines(ctx, ['loss = (Q − (r + γ·max Q′))²'], r, { mono: true, size: clamp(r.h / 3, 7, 11) }); },
        inner: (ctx, c, Sk) => { Sk.lines(ctx, ['after each move it stores (inputs, move, r, next inputs)', `in a memory of the last 50,000 moves (now ${fmtInt(ag.mem.length)}),`, 'then learns from 4 drawn at random:', '  target = r + γ·max Q_target(next)', '  loss = (Q(inputs, move) − target)², error clipped to ±1', `  one step of gradient descent, learning rate ${fmt(o.lr, 3)}`, `the target network: a copy refreshed every 1,000 moves (${fmtInt(ag.moves)} so far)`, `γ = ${fmt(o.gamma, 2)}`], Sk.place(c, [0, 0.04, 1, 0.92]), { size: Sk.u(c, 9, 13), mono: true }); },
        about: ['A network cannot set one entry the way a table does: every weight takes part in every answer. So it learns by gradient descent on the gap between its Q-value and the target r + γ·max Q(next), like any network fitting numbers. Two tricks from DQN (the network that learned Atari games from the screen, 2013–15) keep that stable: a replay memory, so it learns from a mix of old and new moves in many mazes rather than only the last few steps of one, and a target network, a copy of itself refreshed every 1,000 moves, so the target does not shift each time it learns.',
          `It also had a curriculum: its first mazes had the goal a few steps away, and harder ones came in as it improved. ${S.pretrained ? `This network learned offline on ${fmtInt(S.n)} random mazes (none of the named ones).` : ''}`],
        code: `q = net(inputs)[move]\nwith torch.no_grad():\n    target = r if done else r + gamma * target_net(next_inputs).max()\nloss = F.smooth_l1_loss(q, torch.tensor(target))   # the squared error, clipped to ±1\nopt.zero_grad(); loss.backward(); opt.step()`,
      };
    }
    const net = P.kind === 'gameNet';
    return {
      title: 'Learning', sub: net ? 'TD(0) by gradient descent' : 'TD(0): one entry at a time', tone: 's3',
      summary: (ctx, r) => { K().lines(ctx, [net ? 'w ← w − lr·(V − V′)·∂V/∂w' : 'V ← V + α·(V′ − V)'], r, { mono: true, size: clamp(r.h / 3, 7, 11) }); },
      inner: (ctx, c, Sk) => { Sk.lines(ctx, ['after each of a player’s moves:', net ? '  the value of its previous position moves toward' : '  V(previous) ← V(previous) + α·(V(new) − V(previous))', net ? '  the value of its new one: one gradient step' : '', 'at the end, toward the result: 1, 0.5 or 0', net ? `learning rate ${fmt(o.lr, 3)}; each step on one of the 8 symmetric views` : `α = ${fmt(o.alpha, 2)}`, `exploring moves (${fmtPct(o.eps, 0)}) are not learned from`].filter(Boolean), Sk.place(c, [0, 0.04, 1, 0.9]), { size: Sk.u(c, 9, 13), mono: true }); },
      about: [net ? 'Temporal-difference learning with a network: after each move, its previous position’s value takes one gradient step toward the value of the position it reached, and at the end toward the result. Every weight takes part, so a lesson about one position changes its view of similar ones: that is why the network learns faster here than the table.' : 'Temporal-difference learning: after each move, the entry for its previous position moves a fraction α toward the entry of the position it reached, and at the end toward the result. Values spread back, game after game, from the end to the opening.'],
      code: net ? `loss = F.binary_cross_entropy_with_logits(net(features(prev, me)), torch.tensor([target]))\nopt.zero_grad(); loss.backward(); opt.step()` : `V[prev] = V.get(prev, 0.5) + alpha * (target - V.get(prev, 0.5))`,
    };
  }
  function agentTree(L, size) {
    const S = L.S, P = probe(L), wide = size.w / size.h >= 1.05;
    const model = P.kind === 'mazeNet' ? mazeNetTree(L, size, P) : P.kind === 'mazeTable' ? mazeTableTree(L, size, P) : P.kind === 'gameTable' ? threeTableTree(L, size) : valueNetTree(L, size, P);
    const R = wide ? { see: [0, 0.06, 0.19, 0.4], model: [0.26, 0, 0.46, 0.6], pick: [0.79, 0.06, 0.21, 0.4], world: [0.79, 0.64, 0.21, 0.34], learn: [0.26, 0.7, 0.46, 0.28] }
      : { see: [0, 0, 0.47, 0.17], pick: [0.53, 0, 0.47, 0.17], model: [0, 0.23, 1, 0.4], world: [0.53, 0.69, 0.47, 0.17], learn: [0, 0.69, 0.47, 0.17] };
    const parts = { see: seeNode(L, P), pick: pickNode(L, P), world: worldNode(L, P), learn: learnNode(L, P) };
    const kids = [Object.assign(model, { rect: R.model }), ...['see', 'pick', 'world', 'learn'].map((k) => Object.assign(parts[k], { id: k, rect: R[k] }))];
    const what = { mazeNet: 'Q-network', mazeTable: 'Q-table', gameNet: 'value network', gameTable: 'value table' }[P.kind];
    return {
      id: 'agent', title: S.id === 'maze' ? (S.model === 'net' ? 'The agent: DQN' : 'The agent: Q-learning') : `The agent: self-play with a ${what}`,
      sub: `sees → ${what} → chooses → ${S.id === 'maze' ? 'the maze' : 'the game'} → learns`, tone: 's1',
      inner: (ctx, c, Sk) => {
        const at = (k) => Sk.place(c, R[k]), A = (p, q, o) => Sk.arrow(ctx, p[0], p[1], q[0], q[1], Object.assign({ lw: 1.4, head: 7 }, o));
        const s = at('see'), m = at('model'), p = at('pick'), w = at('world'), l = at('learn'), u = Sk.u(c, 9, 12);
        const tag = (t, x, y, al = 'center') => Sk.text(ctx, t, x, y, { size: u * 0.9, color: theme.c('ink-2'), align: al });
        if (wide) {
          A([s.x + s.w + 4, s.y + s.h / 2], [m.x - 4, s.y + s.h / 2]); tag('inputs', (s.x + s.w + m.x) / 2, s.y + s.h / 2 - 9);
          A([m.x + m.w + 4, p.y + p.h / 2], [p.x - 4, p.y + p.h / 2]); tag(S.id === 'maze' ? 'Q-values' : 'values', (m.x + m.w + p.x) / 2, p.y + p.h / 2 - 9);
          A([p.x + p.w / 2, p.y + p.h + 4], [w.x + w.w / 2, w.y - 4]); tag('its move', p.x + p.w / 2 + 6, (p.y + p.h + w.y) / 2, 'left');
          A([w.x - 4, w.y + w.h / 2], [l.x + l.w + 4, w.y + w.h / 2]); tag('reward', (w.x + l.x + l.w) / 2, w.y + w.h / 2 - 9);
          A([l.x + l.w / 2, l.y - 4], [m.x + m.w / 2, m.y + m.h + 4], { dash: [5, 4] }); tag('changes its numbers', l.x + l.w / 2 + 8, (l.y + m.y + m.h) / 2, 'left');
          A([w.x + w.w * 0.15, w.y + w.h + 2], [s.x + s.w / 2, s.y + s.h + 4], { bend: [-(w.x - s.x) * 0.2, c.h * 0.42], dash: [2, 3] }); tag('next state', s.x + 4, s.y + s.h + u * 1.4, 'left');
        } else {
          A([s.x + s.w / 2, s.y + s.h + 3], [s.x + s.w / 2, m.y - 3]); tag('inputs', s.x + s.w / 2 + 6, (s.y + s.h + m.y) / 2, 'left');
          A([p.x + p.w / 2, m.y - 3], [p.x + p.w / 2, p.y + p.h + 3]); tag(S.id === 'maze' ? 'Q-values' : 'values', p.x + p.w / 2 + 6, (p.y + p.h + m.y) / 2, 'left');
          A([p.x + p.w / 2, m.y + m.h + 3], [w.x + w.w / 2, w.y - 3]); tag('its move', w.x + w.w / 2 + 6, (m.y + m.h + w.y) / 2, 'left');
          A([w.x - 3, w.y + w.h / 2], [l.x + l.w + 3, l.y + l.h / 2]); tag('reward', (w.x + l.x + l.w) / 2, w.y + w.h / 2 - 9);
          A([l.x + l.w / 2, l.y - 3], [l.x + l.w / 2, m.y + m.h + 3], { dash: [5, 4] }); tag('learns', l.x + l.w / 2 + 6, (l.y + m.y + m.h) / 2, 'left');
        }
      },
      children: kids,
      about: [`A reinforcement learning agent is a loop. It sees the state, its ${what} scores the possible moves, it picks one, the ${S.id === 'maze' ? 'maze' : 'game'} answers with a reward and the next state, and it learns from that. Nothing else tells it what to do: no examples of good play, only rewards.`,
        `Each box is computed for the state being looked at now (${S.id === 'maze' ? 'the square outlined on the maze, or the test’s current step' : 'the move outlined on the board, or the test’s current move'}). ${NA.openHint('a box', { cap: true })}. The ${what} in the middle holds everything it has learned.`],
      code: S.id === 'maze' ? `state = start\nwhile True:\n    q = model(state)                         # its ${what} scores the 4 moves\n    move = argmax(q) if random.random() > eps else random.randrange(4)\n    state2, reward, done = step(state, move) # the maze answers\n    learn(state, move, reward, state2, done) # and it learns from that\n    state = state2\n    if done:\n        break` : `board, me = [0] * (N * N), 1\nwhile True:\n    move = best_move(board, me)              # its ${what} judges each move\n    board[move] = me\n    learn(previous[me], board)               # the previous position moves toward this one\n    if wins(board, move) or not empties(board):\n        break                                # the result: the only reward\n    me = 3 - me`,
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
  // a value network learned by self-play (Three-line's and Four-line's), in PyTorch
  const VALUE_NET_PY = (N, K, H, o, games, note) => `import random
import torch
import torch.nn as nn

${LINES_PY(N, K)}

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

for game in range(${games}):${note}
    self_play()
print('beats a random player', against_random(200))`;
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
    mazeNet: (L) => {
      const S = L.S, o = S.o;
      return `import random
import torch
import torch.nn as nn
import torch.nn.functional as F

W, H, VIEW = 8, 6, 2
MOVES = [(0, -1), (1, 0), (0, 1), (-1, 0)]       # up, right, down, left

def shortest(maze):
    """The fewest steps from S to G without a pit (0: no way through)."""
    start = next((x, y) for y in range(H) for x in range(W) if maze[y][x] == 'S')
    dist, todo = {start: 0}, [start]
    while todo:
        x, y = todo.pop(0)
        for dx, dy in MOVES:
            nx, ny = x + dx, y + dy
            if 0 <= nx < W and 0 <= ny < H and maze[ny][nx] not in '#X' and (nx, ny) not in dist:
                dist[(nx, ny)] = dist[(x, y)] + 1
                if maze[ny][nx] == 'G':
                    return dist[(nx, ny)]
                todo.append((nx, ny))
    return 0

def random_maze(min_d=8, max_d=99):
    """About one square in five a wall, 1 to 3 pits, S and G anywhere, with a way through of min_d to max_d steps."""
    while True:
        g = [['#' if random.random() < 0.22 else '.' for x in range(W)] for y in range(H)]
        for k in range(random.randint(1, 3)):
            g[random.randrange(H)][random.randrange(W)] = 'X'
        free = [(x, y) for y in range(H) for x in range(W) if g[y][x] == '.']
        (sx, sy), (gx, gy) = random.sample(free, 2)
        g[sy][sx], g[gy][gx] = 'S', 'G'
        maze = [''.join(r) for r in g]
        if min_d <= shortest(maze) <= max_d:
            return maze

def inputs(maze, x, y, seen):
    """104 inputs: wall or edge, pit, goal, visited, for each of the 24 squares around (x, y); then the goal's direction."""
    planes = [[], [], [], []]
    for dy in range(-VIEW, VIEW + 1):
        for dx in range(-VIEW, VIEW + 1):
            if dx == 0 and dy == 0:
                continue
            X, Y = x + dx, y + dy
            c = maze[Y][X] if 0 <= X < W and 0 <= Y < H else '#'
            planes[0].append(c == '#'); planes[1].append(c == 'X'); planes[2].append(c == 'G')
            planes[3].append(c not in '#XG' and (X, Y) in seen)
    gx, gy = next((X - x, Y - y) for Y in range(H) for X in range(W) if maze[Y][X] == 'G')
    goal_dir = [0 < gx <= 2, gx > 2, -2 <= gx < 0, gx < -2, 0 < gy <= 2, gy > 2, -2 <= gy < 0, gy < -2]
    return torch.tensor([float(v) for p in planes for v in p] + [float(v) for v in goal_dir])

def step(maze, x, y, a):
    nx, ny = x + MOVES[a][0], y + MOVES[a][1]
    if not (0 <= nx < W and 0 <= ny < H) or maze[ny][nx] == '#':
        nx, ny = x, y
    c = maze[ny][nx]
    return (nx, ny), (1.0 if c == 'G' else -1.0 if c == 'X' else -0.02), c in 'GX'

net = nn.Sequential(nn.Linear(104, 64), nn.ReLU(), nn.Linear(64, 4))   # 4 Q-values: up, right, down, left
target_net = nn.Sequential(nn.Linear(104, 64), nn.ReLU(), nn.Linear(64, 4))
target_net.load_state_dict(net.state_dict())
opt = torch.optim.SGD(net.parameters(), lr=${NA.fmtCode(o.lr, 3)})
gamma, memory, moves = ${NA.fmtCode(o.gamma, 2)}, [], 0

def remember(move):
    """The replay memory: the last 50,000 moves, the oldest overwritten first."""
    global moves
    if len(memory) < 50000:
        memory.append(move)
    else:
        memory[moves % 50000] = move
    moves += 1
    if moves % 1000 == 0:
        target_net.load_state_dict(net.state_dict())   # refresh the target network every 1,000 moves

def learn(batch=16):
    """DQN: moves drawn from the replay memory, each Q-value a step toward r + gamma * max Q_target(next)."""
    xs, acts, rs, nexts = zip(*random.sample(memory, min(batch, len(memory))))
    done = torch.tensor([x2 is None for x2 in nexts])
    with torch.no_grad():
        future = target_net(torch.stack([x if x2 is None else x2 for x, x2 in zip(xs, nexts)])).max(1).values.masked_fill(done, 0.0)
    q = net(torch.stack(xs)).gather(1, torch.tensor(acts)[:, None]).squeeze(1)
    loss = F.smooth_l1_loss(q, torch.tensor(rs) + gamma * future, reduction='sum')   # each squared error clipped to +-1
    opt.zero_grad()
    loss.backward()
    opt.step()

def run(maze, eps=0.0, train=False):
    """One run from S: True if it reaches the goal within 80 moves."""
    x, y = next((X, Y) for Y in range(H) for X in range(W) if maze[Y][X] == 'S')
    seen = {(x, y)}
    s = inputs(maze, x, y, seen)
    for t in range(80):
        if random.random() < eps:
            a = random.randrange(4)
        else:
            with torch.no_grad():
                a = int(net(s).argmax())
        (x, y), r, done = step(maze, x, y, a)
        seen.add((x, y))
        s2 = None if done else inputs(maze, x, y, seen)
        if train:
            remember((s, a, r, s2))
            if moves % 4 == 0:
                learn()                          # every 4 moves, from 16 drawn at random: 4 a move, as on the page
        s = s2
        if done:
            return r > 0
    return False

# a curriculum: the goal close at first, further away later; exploring less as it goes
EPISODES, WARM, GROW = 20000, 8000, 15000        # a few minutes; the page's network learned on far more mazes (exploring less over 24,000, the curriculum over 60,000)
for episode in range(EPISODES):
    max_d = round(3 + 30 * min(1, episode / GROW))
    eps = max(${NA.fmtCode(o.eps, 2)}, 1 - episode / WARM)
    run(random_maze(min(8, max(1, max_d - 6)), max_d), eps, train=True)
    if episode % 2000 == 1999:
        print(episode + 1, 'mazes; new mazes solved:', sum(run(random_maze()) for k in range(100)), '/ 100')

MAZE = [
${S.maze.rows.map((r) => `    '${r}',`).join('\n')}
]
print('${L.mazeName().replace(/'/g, '')}:', 'solved' if run(MAZE) else 'not solved')`;
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
    threeNet: (L) => VALUE_NET_PY(4, 3, L.S.brain.nH, L.S.o, 1000, '                          # a few hundred games are enough on this board'),
    four: (L) => VALUE_NET_PY(5, 4, L.S.brain.nH, L.S.o, 1500, '                         # a minute or so; it beats a random player within a few hundred'),
  };

  NA.archRL = { build: (L, size) => agentTree(L, size), probe, drawInside, py };
})(window.NA);
