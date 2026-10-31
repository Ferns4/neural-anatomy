/* builder-rlx.js — more ways to look at a network that learns from rewards, in the model builder. Test it 100 times
   (the Training tab): 100 episodes of its best moves from 100 starts it has not been scored on, and how each ended,
   with the returns and lengths. The value map (a view of the results): over two of its inputs, the others as they
   are now, what the network makes of each place: the value it expects there (DQN's best Q-value, the critic's
   value) or the move it would make, with where it is now marked. Where it went (a view of the results): every place
   its moves took it while it learned, over the world's picture (the maze, the race track, the herding field). */
(function (NA) {
  'use strict';
  const RX = {};
  const { el, clear, fmt, fmtPct, fmtInt, clamp, theme } = NA;
  const UI = () => NA.labUI;
  const font = (w, px) => `${w} ${px}px ${(typeof getComputedStyle === 'function' && getComputedStyle(document.body).getPropertyValue('--font-ui').trim()) || 'sans-serif'}`;
  const text = (ctx, s, x, y, o = {}) => { ctx.font = font(o.w || 500, o.size || 11); ctx.fillStyle = o.color || theme.c('ink'); ctx.textAlign = o.align || 'left'; ctx.textBaseline = 'middle'; ctx.fillText(s, x, y); };
  const COLS = ['s1', 's2', 's4', 's7', 's3', 's5', 's6'];

  // ------------------------------------------------------------------ how an episode ended, in words
  const ENDS = {
    maze: { goal: 'reached the goal', pit: 'fell in a pit', lost: 'ran out of steps' }, trap: { goal: 'reached the goal', lost: 'ran out of steps' },
    track: { finished: 'drove every lap', crashed: 'crashed', time: 'ran out of time' }, lander: { landed: 'landed on the pad', offpad: 'landed off the pad', crashed: 'crashed', away: 'flew away', time: 'ran out of time' },
    arm: { reached: 'reached the target', lost: 'never got there' }, walker: { walked: 'walked 10 m', slow: 'too slow', toppled: 'fell over' }, hurdles: { course: 'ran the whole course', hurdlehit: 'hit a hurdle', barhit: 'hit a bar', time: 'ran out of time' },
    herd: { penned: 'penned every sheep', loose: 'some still loose' }, chase: { escaped: 'got away', tagged: 'was caught' }, mountain: { flag: 'reached the flag', lost: 'ran out of steps' },
    memory: { right: 'turned the right way', wrong: 'turned the wrong way', lost: 'waited too long' }, thermo: { comfy: 'kept it comfortable', uncomfy: 'too cold or too hot' },
    swingup: { balanced: 'balanced it', swung: 'swung up, then lost it', hanging: 'never swung up' }, arena: { cleared: 'cleared the drones', dead: 'was shot down', time: 'ran out of time' },
    traffic: { flowing: 'kept the traffic moving', jammed: 'jams built up' }, catch: { caught: 'caught the ball', missed: 'missed it' }, pole: { up: 'kept it up', fell: 'let it fall' }, levers: { paid: 'was paid', unpaid: 'was not paid' },
  };
  // (the worlds that set no end of their own: as the page's results say)
  const endOf = (world, w, t) => (world === 'catch' ? (w.caught ? 'caught' : 'missed') : world === 'pole' ? (t.fell ? 'fell' : 'up') : world === 'levers' ? (t.r > 0 ? 'paid' : 'unpaid') : w.end || 'over');
  const GOOD = new Set(['goal', 'finished', 'landed', 'reached', 'walked', 'course', 'penned', 'escaped', 'flag', 'right', 'comfy', 'balanced', 'cleared', 'flowing', 'caught', 'up', 'paid']);
  RX.endWords = (world, end) => (ENDS[world] && ENDS[world][end]) || end;

  // ------------------------------------------------------------------ Test it 100 times
  /* 100 episodes of its best moves, each from a start of its own (seeds it is never scored on), a few each frame. */
  RX.test = function (lab, n = 100) {
    const d = lab.data, A = lab.agent;
    if (!A || !d || d.family !== 'control' || lab.stale) return;
    const T = lab.endTest = { run: lab.run.n, step: lab.run.step, n, done: 0, ends: {}, rets: [], lens: [] };
    const W = NA.rlEnvs.make(d.world, d.ws), net = lab.watchNet ? lab.watchNet(A) : A.net;
    const one = (k) => {
      const rng = new NA.RNG(910000 + k);
      let o = W.reset(rng), t = { done: false, r: 0 }, ret = 0, len = 0;
      while (!t.done && len < 5000) { t = W.step(A.policy(o, net), rng); ret += t.r; len++; o = t.obs; }
      const e = endOf(d.world, W, t);
      T.ends[e] = (T.ends[e] || 0) + 1; T.rets.push(ret); T.lens.push(len);
    };
    const tick = () => {
      if (lab.endTest !== T) return;
      const t0 = performance.now();
      while (T.done < n && performance.now() - t0 < 25) one(T.done++);
      lab.inspDirty = true;
      if (T.done < n) requestAnimationFrame(tick);
    };
    tick();
  };
  RX.endingsSection = function (lab, d) {
    const live = lab.agent && lab.run && !lab.stale && lab.data && lab.data.world === d.world, T = lab.endTest, ok = T && live && T.run === lab.run.n;
    const btn = el('button', { class: 'btn', type: 'button', disabled: !live || (ok && T.done < T.n), onclick: () => RX.test(lab) }, ok && T.done < T.n ? `Testing… ${T.done} of ${T.n}` : ok ? 'Test it again' : 'Test it 100 times');
    const out = [btn];
    if (ok && T.done) {
      const list = Object.entries(T.ends).sort((a, b) => (GOOD.has(b[0]) ? 1 : 0) - (GOOD.has(a[0]) ? 1 : 0) || b[1] - a[1]), max = Math.max(...list.map((e) => e[1]));
      out.push(el('div', { class: 'bd-ends', role: 'list' }, ...list.map(([e, c]) => el('div', { class: 'bd-end' + (GOOD.has(e) ? ' is-good' : ''), role: 'listitem' },
        el('span', { class: 'bd-end-k', text: RX.endWords(d.world, e) }), el('span', { class: 'bd-end-bar' }, el('span', { style: `width:${(100 * c) / max}%` })), el('span', { class: 'bd-end-v mono', text: `${c}` })))));
      const mean = (a) => a.reduce((s, v) => s + v, 0) / a.length, sd = (a) => { const m = mean(a); return Math.sqrt(a.reduce((s, v) => s + (v - m) ** 2, 0) / a.length); };
      out.push(UI().note(`${T.done < T.n ? `${T.done} of ${T.n} so far. ` : ''}Its returns: ${fmt(mean(T.rets), 2)} on average (spread ${fmt(sd(T.rets), 2)}); ${fmtInt(mean(T.lens))} steps an episode. Run ${T.run} at ${lab.unitOf(lab.run)} ${fmtInt(T.step)}${T.step !== lab.run.step ? `: it has trained on since, so test it again to see how it does now` : ''}.`));
    } else out.push(UI().note('Its score comes from a few episodes; this plays 100, each from a start it has not been scored on, and counts how they ended: how sure the score is, and what goes wrong when it fails.'));
    return UI().section('Test it 100 times', ...out);
  };

  // ------------------------------------------------------------------ the views
  RX.mapOk = (d) => d && d.family === 'control' && d.input && d.input.kind === 'vector' && d.input.dim >= 2;
  RX.visitsOk = (L) => !!(L.agent && L.agent.visits && L.agent.visits.total > 0);
  RX.views = function (o) {
    const out = [];
    if (o.data && RX.mapOk(o.data)) out.push({ value: 'map', label: 'Value map', title: 'Over two of its inputs, the others as they are now: the value it expects there, or the move it would make' });
    if (o.visits) out.push({ value: 'visits', label: 'Where it went', title: 'Every place its moves took it while it learned, brighter where it went more often' });
    return out;
  };
  // what the map shows: the value (Q or the critic's), or the move
  const valueOf = (d) => (d.method === 'dqn' ? 'q' : d.output.k > (d.cont ? d.dA : d.nA) && !NA.brl.isEvo(d.method) && d.method !== 'sac' ? 'v' : null);
  RX.mapControls = function (lab, d) {
    const names = d.obsNames || [], n = d.input.dim, M = lab.mapAxes || (lab.mapAxes = { x: 0, y: Math.min(1, n - 1), show: valueOf(d) ? 'value' : 'move' });
    if (M.x >= n) M.x = 0;
    if (M.y >= n) M.y = Math.min(1, n - 1);
    if (!valueOf(d)) M.show = 'move';
    const opts = names.slice(0, n).map((nm, i) => ({ value: i, label: nm || `input ${i + 1}` }));
    const redo = () => { lab._headKey = null; lab.resultsDirty = true; };
    return el('div', { class: 'bd-mapctl' },
      UI().field('Across', UI().select(opts, M.x, (v) => { M.x = +v; redo(); }, { 'aria-label': 'The input across the map' })),
      UI().field('Up', UI().select(opts, M.y, (v) => { M.y = +v; redo(); }, { 'aria-label': 'The input up the map' })),
      valueOf(d) ? UI().field('Colour', UI().seg([{ value: 'value', label: 'Its value' }, { value: 'move', label: d.cont ? 'Its numbers' : 'Its move' }], M.show, (v) => { M.show = v; redo(); }, 'Colour')) : null);
  };
  // the range of each input: from random play, a little wider
  function rangeOf(d, i) {
    const F = d.input.dim, X = d.Xtr, n = d.nTr;
    let lo = Infinity, hi = -Infinity;
    for (let k = 0; k < n; k++) { const v = X[k * F + i]; if (v < lo) lo = v; if (v > hi) hi = v; }
    if (!(hi > lo)) { lo -= 1; hi += 1; }
    const pad = 0.1 * (hi - lo);
    return [lo - pad, hi + pad];
  }
  const G = 40;
  function computeMap(L, M) {
    const d = L.data, D = L.demo, net = L.net, F = d.input.dim, K = d.output.k, nA = d.cont ? d.dA : d.nA;
    const rx = rangeOf(d, M.x), ry = rangeOf(d, M.y), X = new Float32Array(G * G * F), base = D.obs || d.Xtr.subarray(0, F);
    for (let j = 0; j < G; j++) for (let i = 0; i < G; i++) {
      const k = (j * G + i) * F;
      X.set(base, k);
      X[k + M.x] = rx[0] + ((i + 0.5) / G) * (rx[1] - rx[0]); X[k + M.y] = ry[1] - ((j + 0.5) / G) * (ry[1] - ry[0]);
    }
    const out = net.forward(X, G * G, false), val = new Float64Array(G * G), mv = new Int32Array(G * G), kind = valueOf(d);
    let lo = Infinity, hi = -Infinity;
    for (let c = 0; c < G * G; c++) {
      const o = out.subarray(c * K, c * K + K);
      if (d.cont) val[c] = NA.brl.contMove(d.method, o, d.dA)[M.num || 0];
      else { const q = d.duel ? NA.brl.qValues(o, nA, true) : o; mv[c] = NA.brl.argmax(q, nA); val[c] = kind === 'q' ? q[mv[c]] : kind === 'v' ? o[nA] : 0; }
      if (val[c] < lo) lo = val[c]; if (val[c] > hi) hi = val[c];
    }
    return { rx, ry, val, mv, lo, hi };
  }
  function drawMap(ctx, r, L) {
    const d = L.data, D = L.demo, M = L.mapAxes || { x: 0, y: 1, show: 'move' }, names = d.obsNames || [];
    if (!L.net) { text(ctx, 'Train it first: the map shows what the network makes of each place.', r.x + 4, r.y + 14, { color: theme.c('muted') }); return; }
    const key = [L.run ? `${L.run.n}:${L.run.step}` : '', M.x, M.y, M.show, D.obs ? Array.from(D.obs).map((v) => v.toFixed(2)).join() : ''].join('|');
    const C = RX._map && RX._map.key === key && performance.now() - RX._map.at < 400 ? RX._map.m : computeMap(L, M);
    if (!RX._map || RX._map.key !== key) RX._map = { key, m: C, at: performance.now() };
    const side = Math.max(60, Math.min(r.w - 70, r.h - 64)), x0 = r.x + 44, y0 = r.y + 22, cs = side / G, moves = M.show === 'move';
    text(ctx, moves ? (d.cont ? `What it sets ${d.cActions[M.num || 0]} to, from −1 (blue) to 1 (red)` : 'The move it would make at each place') : d.method === 'dqn' ? 'The value it expects: its best Q-value' : 'The value it expects: the critic’s', r.x + 2, r.y + 9, { w: 650, size: 12, color: theme.c('ink-2') });
    for (let j = 0; j < G; j++) for (let i = 0; i < G; i++) {
      const c = j * G + i;
      ctx.fillStyle = moves ? (d.cont ? theme.divColor(C.val[c], 0.95) : theme.c(COLS[C.mv[c] % COLS.length], 0.85)) : theme.seqColor((C.val[c] - C.lo) / (C.hi - C.lo || 1));
      ctx.fillRect(x0 + i * cs, y0 + j * cs, Math.ceil(cs), Math.ceil(cs));
    }
    ctx.strokeStyle = theme.c('line-2'); ctx.lineWidth = 1; ctx.strokeRect(x0 + 0.5, y0 + 0.5, side - 1, side - 1);
    // where it is now
    if (D.obs) {
      const px = x0 + ((D.obs[M.x] - C.rx[0]) / (C.rx[1] - C.rx[0])) * side, py = y0 + ((C.ry[1] - D.obs[M.y]) / (C.ry[1] - C.ry[0])) * side;
      if (px >= x0 && px <= x0 + side && py >= y0 && py <= y0 + side) { ctx.strokeStyle = '#fff'; ctx.lineWidth = 3; ctx.beginPath(); ctx.arc(px, py, 6, 0, Math.PI * 2); ctx.stroke(); ctx.strokeStyle = theme.c('ink'); ctx.lineWidth = 1.5; ctx.stroke(); }
    }
    text(ctx, `${names[M.x] || 'input'} →`, x0 + side / 2, y0 + side + 12, { align: 'center', size: 11, color: theme.c('ink-2') });
    text(ctx, fmt(C.rx[0], 1), x0, y0 + side + 12, { size: 10, color: theme.c('muted') }); text(ctx, fmt(C.rx[1], 1), x0 + side, y0 + side + 12, { size: 10, color: theme.c('muted'), align: 'right' });
    ctx.save(); ctx.translate(r.x + 12, y0 + side / 2); ctx.rotate(-Math.PI / 2); text(ctx, `${names[M.y] || 'input'} →`, 0, 0, { align: 'center', size: 11, color: theme.c('ink-2') }); ctx.restore();
    text(ctx, fmt(C.ry[1], 1), x0 - 4, y0 + 6, { size: 10, color: theme.c('muted'), align: 'right' }); text(ctx, fmt(C.ry[0], 1), x0 - 4, y0 + side - 6, { size: 10, color: theme.c('muted'), align: 'right' });
    // the key: the moves' colours, or the value's range
    const kx = x0 + side + 14, kw = r.x + r.w - kx;
    if (kw > 60) {
      let y = y0 + 4;
      if (moves && !d.cont) d.actions.forEach((a, k) => { ctx.fillStyle = theme.c(COLS[k % COLS.length], 0.85); ctx.fillRect(kx, y, 12, 12); text(ctx, a, kx + 18, y + 6, { size: 11 }); y += 18; });
      else if (!moves) { text(ctx, `highest ${fmt(C.hi, 2)}`, kx, y + 6, { size: 11 }); text(ctx, `lowest ${fmt(C.lo, 2)}`, kx, y + 24, { size: 11 }); }
      text(ctx, 'The other inputs as they are now: the map changes as it plays.', kx, y + 44, { size: 10.5, color: theme.c('muted') });
    }
  }
  // where on the picture a world's frame is (the frame its where() measures in, 0 to 1 each way)
  function frameOf(world, r, snap) {
    if (world === 'maze' && snap.rows) { const W = snap.rows[0].length, H = snap.rows.length, cs = Math.min(r.w / W, r.h / H); return { x: r.x + (r.w - cs * W) / 2, y: r.y + (r.h - cs * H) / 2, w: cs * W, h: cs * H }; }
    if (world === 'track' && snap.track) { const [bx0, by0, bx1, by1] = snap.track.box, m = 3, sc = Math.min(r.w / (bx1 - bx0 + 2 * m), r.h / (by1 - by0 + 2 * m)); const ox = r.x + (r.w - (bx1 - bx0 + 2 * m) * sc) / 2 - (bx0 - m) * sc, oy = r.y + (r.h - (by1 - by0 + 2 * m) * sc) / 2 - (by0 - m) * sc; return { x: ox + bx0 * sc, y: oy + by0 * sc, w: (bx1 - bx0) * sc, h: (by1 - by0) * sc }; }
    if (world === 'herd') { const s = Math.min(r.w, r.h); return { x: r.x + (r.w - s) / 2, y: r.y + (r.h - s) / 2, w: s, h: s }; }
    return null;
  }
  function drawVisits(ctx, r, L) {
    const d = L.data, D = L.demo, V = L.agent && L.agent.visits;
    text(ctx, V ? `Where it went while it learned: ${fmtInt(V.total)} moves, brighter where it went more often` : 'Where it went', r.x + 2, r.y + 9, { w: 650, size: 12, color: theme.c('ink-2') });
    if (!V || !D || !D.snap) return;
    const wr = { x: r.x, y: r.y + 22, w: r.w, h: r.h - 22 };
    NA.rlEnvs.draw(ctx, wr, d.world, D.snap, theme, { k: d.nA, now: L.now, small: wr.w < 330 });
    const F = frameOf(d.world, wr, D.snap);
    if (!F) return;
    let max = 0; for (let i = 0; i < V.c.length; i++) if (V.c[i] > max) max = V.c[i];
    const cw = F.w / V.n, ch = F.h / V.m, lm = Math.log(1 + max);
    for (let j = 0; j < V.m; j++) for (let i = 0; i < V.n; i++) {
      const c = V.c[j * V.n + i];
      if (!c) continue;
      const v = Math.log(1 + c) / lm;
      ctx.fillStyle = `rgba(255, ${Math.round(200 - 150 * v)}, 40, ${0.15 + 0.55 * v})`;
      ctx.fillRect(F.x + i * cw, F.y + j * ch, Math.ceil(cw), Math.ceil(ch));
    }
  }
  // ------------------------------------------------------------------ a model of the world
  /* Its section in the Training tab: whether it learns one, and once it has learned from a few hundred moves, a dream
     beside the world: one input over 60 moves of its best moves, as the world went and as the model imagined it. */
  RX.worldModelSection = function (lab, d, pick) {
    const A = lab.agent, live = A && A.wm && lab.run && !lab.stale, s = lab.ps(), out = [pick('Learn one beside it', 'worldModel', [['off', 'No'], ['on', 'Yes']], 'A small network of its own that learns what comes next from every move it makes')];
    if (!live || A.wm.buf.length < 256) { out.push(UI().note(s.worldModel === 'on' ? 'Every move it makes teaches a small network of its own (64 and 64 neurons) what comes next: the change in what it sees, and the reward. Once it has a few hundred moves, a dream is drawn here beside the world.' : 'A model of the world predicts what comes next from what it sees and the move it makes. With one, a method can practise in its imagination (Dyna, World Models, MuZero); here you can watch how far its imagination can be trusted.')); return UI().section('A model of the world', ...out); }
    const names = d.obsNames || [], M = lab.dreamAt || (lab.dreamAt = { input: 0, seed: 1 }), key = [lab.run.n, lab.run.step, M.input, M.seed].join('|');
    if (!lab.dreamRes || lab.dreamRes.key !== key) { const net = lab.watchNet ? lab.watchNet(A) : A.net; lab.dreamRes = { key, r: A.wm.dream(d.world, d.ws, (o) => A.policy(o, net), 777000 + M.seed) }; }
    const R = lab.dreamRes.r, c = el('canvas'), chart = new NA.charts.LineChart(c, { height: 150, xLabel: 'Move', format: (v) => fmt(v, 3), yFormat: (v) => fmt(v, 2) }), sum = (a) => a.reduce((x, y) => x + y, 0);
    const pick2 = UI().select(names.slice(0, d.input.dim).map((n, i) => ({ value: i, label: n || `input ${i + 1}` })), M.input, (v) => { M.input = +v; lab._trainKey = null; lab.inspDirty = true; }, { 'aria-label': 'The input to draw' });
    out.push(UI().note(`One move ahead, its predictions are off by ${fmt(Math.sqrt(A.wm.err || 0), 3)} on average (the root of the mean squared error, in the units of its inputs), from ${fmtInt(A.wm.buf.length)} moves kept.`),
      el('div', { class: 'bd-dream-ctl' }, UI().field('Draw', pick2), el('button', { class: 'btn btn-quiet', type: 'button', onclick: () => { M.seed++; lab._trainKey = null; lab.inspDirty = true; } }, 'Dream again')),
      el('div', { class: 'chart-box' }, c),
      UI().note(`${R.rr.length} moves of its best moves from one start: the world (solid) and its dream (each move fed back what the model predicted). The world paid ${fmt(sum(R.rr), 2)}; it imagined ${fmt(sum(R.ri), 2)}. Small errors compound: the dream drifts from the world, the further the sooner.`));
    requestAnimationFrame(() => { if (!c.isConnected) return; chart.frame(); chart.set({ series: [{ name: 'the world', color: theme.c('s1'), points: R.real.map((o, t) => [t, o[M.input]]) }, { name: 'its dream', color: theme.c('s2'), points: R.imag.map((o, t) => [t, o[M.input]]) }] }); });
    return UI().section('A model of the world', ...out);
  };
  // ------------------------------------------------------------------ population-based training
  RX.pbtSection = function (lab, d, pick) {
    const A = lab.agent, P = A && A.pbt && lab.run && !lab.stale ? A : null, s = lab.ps(), out = [pick('Learners', 'pbt', [[1, 'One'], [4, '4 side by side'], [8, '8']], 'Population-based training: several learners, the worst copying the best now and then')];
    out.push(UI().note(s.pbt > 1 ? `${s.pbt} learners train side by side, each from its own seed and with its own learning rate (the one set, times 0.3 to 3), so a step takes ${s.pbt} times as long. Every 500 steps each is scored, and the worst quarter take the weights and learning rate of the best quarter, nudged by ×0.8 or ×1.25: a search for a good learning rate while it learns. The best is the one on show.${P && P.hist.length ? ` Now: learner ${P.best.pbtId} is best, its learning rate × ${fmt(P.best.lrMul, 2)}; ${P.hist.reduce((a, h) => a + h.moved.length, 0)} copies so far.` : ''}` : 'One learner. With several side by side, the worst copy the best every so often and nudge their learning rates (population-based training): a search for good settings that costs no extra runs.'));
    return UI().section('Population-based training', ...out);
  };
  RX.draw = function (view, ctx, r, L) { if (view === 'map') drawMap(ctx, r, L); else drawVisits(ctx, r, L); };
  NA.rlx = RX;
})(window.NA);
