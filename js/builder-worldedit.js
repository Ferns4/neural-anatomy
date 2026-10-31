/* builder-worldedit.js — Build your own world, in the model builder: an editor over the page for the maze (walls,
   pits, the start and the goal on its 8 × 6 squares), the race track (the points its middle line runs through), the
   hurdles course (where each hurdle and bar stands) and the sheep (where the flock and the dog start). The world's
   settings in the Input block open it (WE.open(lab, kind)). Use this puts the layout into the problem's settings, so
   Undo takes it back and a share link carries it; the world reads it from there, and checks it again (rl-envs.js and
   rl-worlds*.js), so a layout that cannot work is never used. Beside the picture: what the world pays, from the
   Reward tab. Pointer: tap or drag to paint, to add or to move. Keys on the picture: arrows move the cursor or what is
   picked, [ and ] pick another, Space paints, Delete removes; Ctrl+Z undoes. WE.KINDS: each kind's model, check and
   picture, for tests. */
(function (NA) {
  'use strict';
  const WE = {};
  const { el, clamp, fmt, fmtInt } = NA;
  const RE = () => NA.rlEnvs, RL = () => NA.rl, TH = () => NA.theme;
  const half = (v) => Math.round(v * 2) / 2; // to the half metre
  const num = (v) => String(+Math.abs(v).toPrecision(3));
  const signed = (v) => (v > 0 ? '+' : '−') + num(v);
  const font = (w, px) => `${w} ${px}px ${(typeof getComputedStyle === 'function' && getComputedStyle(document.body).getPropertyValue('--font-ui').trim()) || 'sans-serif'}`;
  const say = (ctx, t, x, y, px, col, align = 'center', w = 650) => { ctx.font = font(w, px); ctx.fillStyle = col; ctx.textAlign = align; ctx.textBaseline = 'middle'; ctx.fillText(t, x, y); };

  // ------------------------------------------------------------------ the maze
  // the shortest safe way through (the squares, start to goal), for the picture; [] when there is none
  function mazePath(rows) {
    const M = new (RL().Maze)(rows), from = new Int16Array(M.n).fill(-1), q = [M.start];
    from[M.start] = M.start;
    while (q.length) {
      const s = q.shift();
      if (s === M.goal) { const p = [s]; let k = s; while (k !== M.start) { k = from[k]; p.unshift(k); } return p; }
      for (let a = 0; a < 4; a++) { const t = M.move(s, a); if (from[t] >= 0 || M.cell[t] === 'X') continue; from[t] = s; q.push(t); }
    }
    return [];
  }
  const mazeGeo = (r) => { const cs = Math.min(r.w / 8, r.h / 6); return { cs, x0: r.x + (r.w - cs * 8) / 2, y0: r.y + (r.h - cs * 6) / 2 }; };
  const MAZE = {
    world: 'maze', title: 'Your own maze', use: 'Use this maze', aspect: 6 / 8,
    sub: 'Paint walls, pits and floor on its 8 × 6 squares, and put the start and the goal where you like. Every episode is this maze.',
    tools: [['#', 'Wall'], ['X', 'Pit'], ['.', 'Floor'], ['S', 'Start'], ['G', 'Goal']],
    help: 'Tap or drag to paint. Keys: arrows move the cursor, Space paints.',
    from(s) { const own = s.maze === 'own' && RE().mazeCheck(s.mazeRows).ok, def = RL().MAZE_SET.find((m) => m.id === s.maze) || RL().MAZE_SET[0]; return { rows: (own ? s.mazeRows : def.rows).slice() }; },
    starts: () => [...RL().MAZE_SET.map((m) => ({ label: m.name, make: () => ({ rows: m.rows.slice() }) })),
      { label: 'A random maze', make: () => ({ rows: RL().randomMaze(Math.random) }) }, { label: 'An empty one', make: () => ({ rows: ['S.......', '........', '........', '........', '........', '.......G'] }) }],
    check(ed) { const c = RE().mazeCheck(ed.m.rows); return c.ok ? { ok: true, say: `The shortest safe way from the start to the goal takes ${c.steps} steps.` } : { ok: false, say: c.why }; },
    put(q, m) { q.maze = 'own'; q.mazeRows = m.rows.slice(); q.random = false; },
    draw(ed, ctx, r) {
      const rows = ed.m.rows, M = new (RL().Maze)(rows), s = ed.s, path = mazePath(rows);
      RE().draw(ctx, r, 'maze', { rows, s: M.start, path: path.length > 1 ? path : null, pay: s.rFormula ? null : { goal: s.rGoal ?? 1, pit: s.rPit ?? -1 } }, TH());
      const { cs, x0, y0 } = mazeGeo(r);
      if (ed.keys) { ctx.setLineDash([5, 4]); ctx.strokeStyle = TH().c('ink'); ctx.lineWidth = 2; ctx.strokeRect(x0 + ed.cur.x * cs + 3, y0 + ed.cur.y * cs + 3, cs - 6, cs - 6); ctx.setLineDash([]); }
    },
    cell(ed, p) { const { cs, x0, y0 } = mazeGeo(ed.r), x = Math.floor((p.x - x0) / cs), y = Math.floor((p.y - y0) / cs); return x >= 0 && x < 8 && y >= 0 && y < 6 ? { x, y } : null; },
    paint(ed, c) {
      const t = ed.tool, g = ed.m.rows.map((row) => row.split('')), was = g[c.y][c.x];
      if (was === t) return;
      if (t === 'S' || t === 'G') {
        if (was === 'S' || was === 'G') { ed.note = `The ${was === 'S' ? 'start' : 'goal'} is there: put the ${t === 'S' ? 'start' : 'goal'} on another square.`; return; }
        for (const row of g) for (let k = 0; k < 8; k++) if (row[k] === t) row[k] = '.';
      } else if (was === 'S' || was === 'G') { ed.note = `That is the ${was === 'S' ? 'start' : 'goal'}: move it with the ${was === 'S' ? 'Start' : 'Goal'} tool.`; return; }
      g[c.y][c.x] = t; ed.m = { rows: g.map((row) => row.join('')) };
    },
    down(ed, p) { const c = this.cell(ed, p); if (!c) return; ed.cur = c; this.paint(ed, c); },
    move(ed, p) { if (ed.tool === 'S' || ed.tool === 'G') return; const c = this.cell(ed, p); if (c) { ed.cur = c; this.paint(ed, c); } },
    key(ed, e) {
      const dx = { ArrowLeft: -1, ArrowRight: 1 }[e.key] || 0, dy = { ArrowUp: -1, ArrowDown: 1 }[e.key] || 0;
      if (dx || dy) { ed.cur = { x: clamp(ed.cur.x + dx, 0, 7), y: clamp(ed.cur.y + dy, 0, 5) }; return 'moved'; }
      if (e.key === ' ' || e.key === 'Enter') { this.paint(ed, ed.cur); return 'changed'; }
      return null;
    },
    tell(ed) { const c = ed.cur, ch = ed.m.rows[c.y][c.x]; return `square ${c.x + 1} across, ${c.y + 1} down: ${({ '#': 'a wall', X: 'a pit', S: 'the start', G: 'the goal', '.': 'floor' })[ch]}`; },
  };

  // ------------------------------------------------------------------ the race track
  const FW = 160, FH = 125; // the field the points sit in, m (the built-in tracks fit in it)
  const trackGeo = (r) => { const sc = Math.min(r.w / FW, r.h / FH); return { sc, ox: r.x + (r.w - FW * sc) / 2, oy: r.y + (r.h - FH * sc) / 2 }; };
  const nearSeg = (p, a, b) => { const dx = b[0] - a[0], dy = b[1] - a[1], l = dx * dx + dy * dy || 1, t = clamp(((p[0] - a[0]) * dx + (p[1] - a[1]) * dy) / l, 0, 1); return Math.hypot(a[0] + t * dx - p[0], a[1] + t * dy - p[1]); };
  const TRACK = {
    world: 'track', title: 'Your own track', use: 'Use this track', aspect: FH / FW,
    sub: 'The track runs through its points in order, a smooth loop 9 m wide. Drag a point to move it; tap the grass to add one between the two nearest. The car starts at point 1, heading to point 2.',
    tools: null, help: 'Drag points, tap to add. Keys: [ and ] pick a point, arrows move it, Delete removes it.',
    from(s) { const own = s.track === 'own' && RE().trackCheck(s.trackPts).ok, T = RE().TRACKS; return { pts: (own ? s.trackPts : T[T[s.track] ? s.track : 'esses'].pts).map((p) => p.slice()) }; },
    starts: () => [...Object.values(RE().TRACKS).map((t) => ({ label: t.name, make: () => ({ pts: t.pts.map((p) => p.slice()) }) })),
      { label: 'A random track', make: () => ({ pts: RE().randomTrack(1 + Math.floor(Math.random() * 64)).pts.map((p) => [half(p[0]), half(p[1])]) }) }],
    check(ed) {
      const c = RE().trackCheck(ed.m.pts);
      ed.built = c.T || null; ed.at = c.at || null;
      return c.ok ? { ok: true, say: `${fmtInt(c.T.L)} m round, ${ed.m.pts.length} points; its tightest bend is ${fmt(c.T.minR, 1)} m from its middle.` } : { ok: false, say: c.why };
    },
    put(q, m) { q.track = 'own'; q.trackPts = m.pts.map((p) => [half(p[0]), half(p[1])]); },
    extra: (ed) => [{ label: 'Remove point', on: ed.sel >= 0 && ed.m.pts.length > 5, run: () => TRACK.remove(ed) }],
    remove(ed) { if (ed.sel < 0 || ed.m.pts.length <= 5) return; ed.m.pts.splice(ed.sel, 1); ed.sel = Math.min(ed.sel, ed.m.pts.length - 1); },
    draw(ed, ctx, r) {
      const { sc, ox, oy } = trackGeo(r), X = (x) => ox + x * sc, Y = (y) => oy + y * sc, th = TH(), T = ed.built, P = ed.m.pts, HW = RE().HW;
      ctx.fillStyle = th.c('good', th.dark ? 0.09 : 0.12); ctx.fillRect(ox, oy, FW * sc, FH * sc);
      ctx.strokeStyle = th.c('line'); ctx.lineWidth = 1; ctx.strokeRect(ox + 0.5, oy + 0.5, FW * sc - 1, FH * sc - 1);
      if (T) {
        const loop = () => { ctx.beginPath(); for (let k = 0; k <= T.N; k++) { const j = k % T.N; if (k) ctx.lineTo(X(T.cx[j]), Y(T.cy[j])); else ctx.moveTo(X(T.cx[j]), Y(T.cy[j])); } };
        ctx.lineJoin = 'round';
        loop(); ctx.strokeStyle = th.c('ink-2', 0.55); ctx.lineWidth = 2 * HW * sc + 2; ctx.stroke();
        loop(); ctx.strokeStyle = th.dark ? '#4b4f55' : '#9aa0a6'; ctx.lineWidth = 2 * HW * sc - 1; ctx.stroke();
        loop(); ctx.setLineDash([6, 6]); ctx.strokeStyle = 'rgba(255,255,255,0.7)'; ctx.lineWidth = 1.2; ctx.stroke(); ctx.setLineDash([]);
        // the start line, across the track at point 1, and an arrow the way round
        const nx = -T.ty[0], ny = T.tx[0];
        ctx.strokeStyle = '#fff'; ctx.lineWidth = 3; ctx.beginPath(); ctx.moveTo(X(T.cx[0] + nx * HW), Y(T.cy[0] + ny * HW)); ctx.lineTo(X(T.cx[0] - nx * HW), Y(T.cy[0] - ny * HW)); ctx.stroke();
        const k = Math.min(T.N - 1, Math.round(12 / T.ds)), ax = X(T.cx[k]), ay = Y(T.cy[k]), ang = Math.atan2(T.ty[k], T.tx[k]);
        ctx.save(); ctx.translate(ax, ay); ctx.rotate(ang); ctx.fillStyle = th.c('s7'); ctx.beginPath(); ctx.moveTo(9, 0); ctx.lineTo(-5, -6); ctx.lineTo(-5, 6); ctx.closePath(); ctx.fill(); ctx.restore();
      }
      if (ed.at) { ctx.strokeStyle = th.c('critical'); ctx.lineWidth = 3; ctx.beginPath(); ctx.arc(X(ed.at[0]), Y(ed.at[1]), Math.max(12, 7 * sc), 0, Math.PI * 2); ctx.stroke(); }
      // the points, numbered in order
      ctx.strokeStyle = th.c('ink-2', 0.5); ctx.lineWidth = 1; ctx.setLineDash([2, 4]); ctx.beginPath(); P.forEach((p, i) => (i ? ctx.lineTo(X(p[0]), Y(p[1])) : ctx.moveTo(X(p[0]), Y(p[1])))); ctx.closePath(); ctx.stroke(); ctx.setLineDash([]);
      P.forEach((p, i) => {
        const on = i === ed.sel, rad = on ? 11 : 9;
        ctx.fillStyle = on ? th.c('s7') : th.c('panel'); ctx.strokeStyle = th.c(on ? 's7' : 'ink'); ctx.lineWidth = 2;
        ctx.beginPath(); ctx.arc(X(p[0]), Y(p[1]), rad, 0, Math.PI * 2); ctx.fill(); ctx.stroke();
        say(ctx, String(i + 1), X(p[0]), Y(p[1]) + 0.5, 10.5, on ? th.c('panel') : th.c('ink'), 'center', 700);
      });
    },
    world2(ed, p) { const { sc, ox, oy } = trackGeo(ed.r); return [(p.x - ox) / sc, (p.y - oy) / sc]; },
    hit(ed, p) { const { sc, ox, oy } = trackGeo(ed.r); let best = -1, bd = ed.touch ? 22 : 14; ed.m.pts.forEach((q, i) => { const d = Math.hypot(ox + q[0] * sc - p.x, oy + q[1] * sc - p.y); if (d < bd) { bd = d; best = i; } }); return best; },
    down(ed, p) {
      const i = this.hit(ed, p), w = this.world2(ed, p);
      if (i >= 0) { ed.sel = i; ed.grab = true; return; }
      if (w[0] < 0 || w[0] > FW || w[1] < 0 || w[1] > FH) return;
      if (ed.m.pts.length >= 16) { ed.note = 'A track has at most 16 points: remove one first.'; return; }
      const P = ed.m.pts; let at = 0, bd = Infinity;
      for (let k = 0; k < P.length; k++) { const d = nearSeg(w, P[k], P[(k + 1) % P.length]); if (d < bd) { bd = d; at = k + 1; } }
      P.splice(at, 0, [clamp(half(w[0]), 5, FW - 5), clamp(half(w[1]), 5, FH - 5)]); ed.sel = at; ed.grab = true;
    },
    move(ed, p) { if (!ed.grab || ed.sel < 0) return; const w = this.world2(ed, p); ed.m.pts[ed.sel] = [clamp(half(w[0]), 5, FW - 5), clamp(half(w[1]), 5, FH - 5)]; },
    key(ed, e) {
      const P = ed.m.pts, step = e.shiftKey ? 5 : 1, dx = { ArrowLeft: -step, ArrowRight: step }[e.key] || 0, dy = { ArrowUp: -step, ArrowDown: step }[e.key] || 0;
      if (e.key === '[' || e.key === ']') { ed.sel = ed.sel < 0 ? 0 : (ed.sel + (e.key === ']' ? 1 : -1) + P.length) % P.length; return 'moved'; }
      if ((dx || dy) && ed.sel < 0) ed.sel = 0;
      if (dx || dy) { const q = P[ed.sel]; P[ed.sel] = [clamp(q[0] + dx, 5, FW - 5), clamp(q[1] + dy, 5, FH - 5)]; return 'changed'; }
      if ((e.key === 'Delete' || e.key === 'Backspace') && ed.sel >= 0) { if (P.length <= 5) { ed.note = 'A track needs at least 5 points.'; return 'moved'; } this.remove(ed); return 'changed'; }
      return null;
    },
    tell(ed) { const q = ed.m.pts[ed.sel]; return q ? `point ${ed.sel + 1} of ${ed.m.pts.length}, at ${fmt(q[0], 1)} m across and ${fmt(q[1], 1)} m down` : ''; },
  };

  // ------------------------------------------------------------------ the hurdles course
  const ROW = 50; // metres of course a row of the picture shows
  const courseOf = (s) => (s.course > 0 ? clamp(s.course, 40, 1000) : 150);
  const randomCourse = (C, bars) => { const out = []; for (let x = 14 + Math.random() * 4; x < C - 4; x += 7 + Math.random() * 7) out.push([half(x), bars && Math.random() < 0.4 ? 1 : 0]); return out; };
  const courseGeo = (ed, r) => { const C = courseOf(ed.s), rows = Math.ceil(C / ROW), rh = r.h / rows, pm = (r.w - 28) / ROW; return { C, rows, rh, pm, x0: r.x + 14, y0: r.y }; };
  const HURDLES = {
    world: 'hurdles', title: 'Your own course', use: 'Use this course',
    height: (ed, w) => Math.ceil(courseOf(ed.s) / ROW) * Math.max(62, Math.min(90, w * 0.13)) + 6,
    sub: 'Where each obstacle stands, metres from the start: a hurdle to jump, or a bar to duck under. Tap the track to add one, drag one to move it, tap one with the other tool to change it. Every episode is this course.',
    tools: [['hurdle', 'Hurdle'], ['bar', 'Bar'], ['remove', 'Remove']],
    help: 'Keys: [ and ] pick an obstacle, ← → move it (Shift: 5 m), B turns it into the other kind, Delete removes it.',
    from(s) {
      if (s.obstacles === 'own' && RE().courseCheck(s.obst, s.course).ok) return { obst: s.obst.map((q) => [q[0], q[1] ? 1 : 0]) };
      const w = RE().make('hurdles', { obstacles: s.obstacles, course: s.course, run: s.run, jump: s.jump }); w.reset(RE().kit.seeded(11));
      return { obst: w.obst.map((q) => [half(q.x), q.bar ? 1 : 0]) };
    },
    starts: (ed) => [{ label: 'A random course of hurdles', make: () => ({ obst: randomCourse(courseOf(ed.s), false) }) }, { label: 'A random course with bars', make: () => ({ obst: randomCourse(courseOf(ed.s), true) }) }, { label: 'An empty course', make: () => ({ obst: [] }) }],
    check(ed) {
      const c = RE().courseCheck(ed.m.obst, ed.s.course), s = ed.s, J = s.jump > 0 ? s.jump : 5.5, run = s.run > 0 ? s.run : 8, land = (2 * J / 15) * run;
      if (!c.ok) return { ok: false, say: c.why };
      const nb = ed.m.obst.filter((q) => q[1]).length, n = ed.m.obst.length;
      return { ok: true, say: `${n} obstacle${n === 1 ? '' : 's'}${nb ? `, ${nb} of them bar${nb === 1 ? '' : 's'}` : ''}${n > 1 ? `; the closest two are ${fmt(c.close, 1)} m apart` : ''}.${n > 1 && c.close < land ? ` A jump lands about ${fmt(land, 1)} m on at ${fmt(run, 0)} m/s, so two that close are hard to clear one after the other.` : ''}` };
    },
    put(q, m) { q.obstacles = 'own'; q.obst = m.obst.slice().sort((a, b) => a[0] - b[0]).map((x) => [x[0], x[1] ? 1 : 0]); },
    draw(ed, ctx, r) {
      const th = TH(), G = courseGeo(ed, r), mpx = Math.min(30, (G.rh - 24) / 2.1); // px a metre of height (a bar's top, 1.9 m up, inside its row)
      for (let k = 0; k < G.rows; k++) {
        const gy = G.y0 + (k + 1) * G.rh - 16, from = k * ROW, to = Math.min(G.C, from + ROW), X = (x) => G.x0 + (x - from) * G.pm;
        ctx.fillStyle = th.c('good', th.dark ? 0.1 : 0.14); ctx.fillRect(X(from), gy, (to - from) * G.pm, 5);
        ctx.strokeStyle = th.c('ink-2', 0.6); ctx.lineWidth = 1.5; ctx.beginPath(); ctx.moveTo(X(from), gy); ctx.lineTo(X(to), gy); ctx.stroke();
        for (let m = from; m <= to; m += 10) { ctx.beginPath(); ctx.moveTo(X(m), gy); ctx.lineTo(X(m), gy + 4); ctx.stroke(); say(ctx, `${m}${m === from ? ' m' : ''}`, X(m), gy + 10, 9.5, th.c('muted'), m === from ? 'left' : 'center', 600); }
        if (from === 0) say(ctx, 'start ▸', X(0), gy - mpx * 1.9, 10, th.c('ink-2'), 'left', 700);
        if (to === G.C) { ctx.strokeStyle = th.c('good'); ctx.lineWidth = 3; ctx.beginPath(); ctx.moveTo(X(to), gy); ctx.lineTo(X(to), gy - mpx * 1.6); ctx.stroke(); say(ctx, 'finish', X(to) - 4, gy - mpx * 1.75, 10, th.c('good'), 'right', 700); }
        ed.m.obst.forEach((q, i) => {
          if (q[0] < from || q[0] >= to + (to === G.C ? 1 : 0)) return;
          const x = X(q[0]), on = i === ed.sel, col = th.c(on ? 's7' : 'ink'), lw = on ? 3 : 2;
          ctx.strokeStyle = col; ctx.lineWidth = lw;
          if (q[1]) { const yb = gy - 1.2 * mpx, yt = gy - 1.9 * mpx; ctx.lineWidth = 1; ctx.beginPath(); ctx.moveTo(x - 7, gy); ctx.lineTo(x - 7, yt); ctx.moveTo(x + 7, gy); ctx.lineTo(x + 7, yt); ctx.stroke(); ctx.fillStyle = col; ctx.fillRect(x - 9, yb - 4, 18, 5); }
          else { const yt = gy - 0.7 * mpx; ctx.beginPath(); ctx.moveTo(x - 5, gy); ctx.lineTo(x - 5, yt); ctx.moveTo(x + 5, gy); ctx.lineTo(x + 5, yt); ctx.stroke(); ctx.fillStyle = col; ctx.fillRect(x - 8, yt - 2, 16, 4); }
          if (on) say(ctx, `${fmt(q[0], 1)} m`, x + 11, gy - 7, 10, th.c('s7'), 'left', 700);
        });
      }
    },
    at(ed, p) { const G = courseGeo(ed, ed.r), k = clamp(Math.floor((p.y - G.y0) / G.rh), 0, G.rows - 1); return { k, x: k * ROW + (p.x - G.x0) / G.pm, G }; },
    hit(ed, p) { const a = this.at(ed, p); let best = -1, bd = (ed.touch ? 16 : 11) / a.G.pm; ed.m.obst.forEach((q, i) => { const d = Math.abs(q[0] - a.x); if (Math.floor(q[0] / ROW) === a.k || Math.abs(q[0] - a.x) < 1) { if (d < bd) { bd = d; best = i; } } }); return best; },
    down(ed, p) {
      const i = this.hit(ed, p), a = this.at(ed, p), C = a.G.C;
      ed.tapped = i; ed.moved = false;
      if (i >= 0) { if (ed.tool === 'remove') { ed.m.obst.splice(i, 1); ed.sel = -1; return; } ed.sel = i; ed.grab = true; return; }
      if (ed.tool === 'remove' || a.x < 0 || a.x > C) return;
      if (ed.m.obst.length >= 60) { ed.note = 'A course has at most 60 obstacles.'; return; }
      ed.m.obst.push([clamp(half(a.x), 6, C - 3), ed.tool === 'bar' ? 1 : 0]); ed.sel = ed.m.obst.length - 1; ed.grab = true; ed.tapped = -1;
    },
    move(ed, p) { if (!ed.grab || ed.sel < 0) return; const a = this.at(ed, p), q = ed.m.obst[ed.sel], x = clamp(half(a.x), 6, a.G.C - 3); if (x !== q[0]) { q[0] = x; ed.moved = true; } },
    up(ed) { if (ed.tapped >= 0 && !ed.moved && ed.tapped === ed.sel) { const q = ed.m.obst[ed.sel], want = ed.tool === 'bar' ? 1 : 0; if (q && q[1] !== want) q[1] = want; } },
    key(ed, e) {
      const O = ed.m.obst, order = O.map((q, i) => i).sort((a, b) => O[a][0] - O[b][0]), C = courseOf(ed.s), step = e.shiftKey ? 5 : 0.5;
      if (e.key === '[' || e.key === ']') { if (!O.length) return null; const k = order.indexOf(ed.sel); ed.sel = order[k < 0 ? 0 : (k + (e.key === ']' ? 1 : -1) + O.length) % O.length]; return 'moved'; }
      if (ed.sel < 0 || !O[ed.sel]) return null;
      const dx = { ArrowLeft: -step, ArrowRight: step }[e.key] || 0;
      if (dx) { O[ed.sel][0] = clamp(O[ed.sel][0] + dx, 6, C - 3); return 'changed'; }
      if (e.key === 'b' || e.key === 'B') { O[ed.sel][1] = O[ed.sel][1] ? 0 : 1; return 'changed'; }
      if (e.key === 'Delete' || e.key === 'Backspace') { O.splice(ed.sel, 1); ed.sel = -1; return 'changed'; }
      return null;
    },
    tell(ed) { const q = ed.m.obst[ed.sel]; return q ? `a ${q[1] ? 'bar' : 'hurdle'} at ${fmt(q[0], 1)} m` : ''; },
  };

  // ------------------------------------------------------------------ the sheep
  const FIELD = 24;
  const herdGeo = (r) => { const s = Math.min(r.w, r.h), x = r.x + (r.w - s) / 2, y = r.y + (r.h - s) / 2; return { x, y, sc: s / FIELD }; };
  const grazing = (n) => ({ sheep: Array.from({ length: n }, () => [half(4 + Math.random() * 8), half(11 + Math.random() * 9)]), dog: [2, 22] });
  const HERD = {
    world: 'herd', title: 'Where the sheep start', use: 'Use these places', aspect: 1,
    sub: 'Drag the sheep and the dog to where each game starts; add up to 7 sheep, or take some away. The pen and its gate stay in the top right corner.',
    tools: [['move', 'Move'], ['add', 'Add a sheep'], ['remove', 'Remove a sheep']],
    help: 'Keys: [ and ] pick the dog or a sheep, arrows move it (Shift: 2 m), Delete removes a sheep.',
    from(s) {
      if (s.sheepAt === 'own' && RE().flockCheck(s.flock, s.dogAt).ok) return { sheep: s.flock.map((p) => p.slice()), dog: s.dogAt.slice() };
      const w = RE().make('herd', { sheep: s.sheep }); w.reset(RE().kit.seeded(43));
      return { sheep: w.sheep.map((q) => [half(q.x), half(q.y)]), dog: [w.dog.x, w.dog.y] };
    },
    starts: () => [3, 5, 7].map((n) => ({ label: `${n} sheep grazing in the bottom left`, make: () => grazing(n) })).concat([{ label: '5 sheep all over the field', make: () => ({ sheep: Array.from({ length: 5 }, () => { for (;;) { const p = [half(2 + Math.random() * 20), half(2 + Math.random() * 20)]; if (!(p[0] > 15 && p[1] < 7.5)) return p; } }), dog: [12, 23] }) }]),
    check(ed) {
      const c = RE().flockCheck(ed.m.sheep, ed.m.dog), S = ed.m.sheep, n = S.length;
      if (!c.ok) return { ok: false, say: c.why };
      const cx = S.reduce((a, p) => a + p[0], 0) / n, cy = S.reduce((a, p) => a + p[1], 0) / n;
      return { ok: true, say: `${n} sheep, their middle ${fmt(Math.hypot(cx - 19, cy - 6.5), 0)} m from the gate; the dog ${fmt(Math.hypot(cx - ed.m.dog[0], cy - ed.m.dog[1]), 0)} m from them.` };
    },
    put(q, m) { q.sheepAt = 'own'; q.flock = m.sheep.map((p) => [half(p[0]), half(p[1])]); q.dogAt = [half(m.dog[0]), half(m.dog[1])]; },
    draw(ed, ctx, r) {
      const th = TH(), { x, y, sc } = herdGeo(r), m = ed.m;
      RE().draw(ctx, { x, y, w: sc * FIELD, h: sc * FIELD }, 'herd', { dog: { x: m.dog[0], y: m.dog[1], vx: 0, vy: 0 }, sheep: m.sheep.map((p) => ({ x: p[0], y: p[1], vx: 0, vy: 0, penned: false })), t: 0, maxSteps: ed.s.maxSteps || 600, n: m.sheep.length, end: null, trail: [] }, th, { small: true });
      const ring = (p, on) => { ctx.strokeStyle = on ? th.c('s7') : 'rgba(255,255,255,0.75)'; ctx.lineWidth = on ? 3 : 1.5; ctx.beginPath(); ctx.arc(x + p[0] * sc, y + p[1] * sc, Math.max(11, 0.85 * sc), 0, Math.PI * 2); ctx.stroke(); };
      m.sheep.forEach((p, i) => ring(p, ed.sel === i));
      ring(m.dog, ed.sel === -2);
      say(ctx, 'dog', x + m.dog[0] * sc, y + m.dog[1] * sc - Math.max(16, 1.2 * sc), 10.5, '#fff', 'center', 700);
    },
    world2(ed, p) { const { x, y, sc } = herdGeo(ed.r); return [(p.x - x) / sc, (p.y - y) / sc]; },
    hit(ed, p) { const w = this.world2(ed, p), { sc } = herdGeo(ed.r), lim = Math.max(1.1, (ed.touch ? 20 : 13) / sc); let best = null, bd = lim; ed.m.sheep.forEach((q, i) => { const d = Math.hypot(q[0] - w[0], q[1] - w[1]); if (d < bd) { bd = d; best = i; } }); const dd = Math.hypot(ed.m.dog[0] - w[0], ed.m.dog[1] - w[1]); if (dd < bd) best = -2; return best; },
    place: (w) => [clamp(half(w[0]), 1, FIELD - 1), clamp(half(w[1]), 1, FIELD - 1)],
    down(ed, p) {
      const i = this.hit(ed, p), w = this.world2(ed, p);
      if (ed.tool === 'remove') { if (i === null || i === -2) return; if (ed.m.sheep.length <= 1) { ed.note = 'Keep one sheep at least.'; return; } ed.m.sheep.splice(i, 1); ed.sel = -1; return; }
      if (i !== null) { ed.sel = i; ed.grab = true; return; }
      if (ed.tool === 'add' && w[0] >= 0 && w[0] <= FIELD && w[1] >= 0 && w[1] <= FIELD) { if (ed.m.sheep.length >= 7) { ed.note = 'Seven sheep at most.'; return; } ed.m.sheep.push(this.place(w)); ed.sel = ed.m.sheep.length - 1; ed.grab = true; }
    },
    move(ed, p) { if (!ed.grab || ed.sel === null || ed.sel === -1) return; const q = this.place(this.world2(ed, p)); if (ed.sel === -2) ed.m.dog = q; else ed.m.sheep[ed.sel] = q; },
    key(ed, e) {
      const S = ed.m.sheep, ids = [-2, ...S.map((q, i) => i)], step = e.shiftKey ? 2 : 0.5;
      if (e.key === '[' || e.key === ']') { const k = ids.indexOf(ed.sel); ed.sel = ids[k < 0 ? 0 : (k + (e.key === ']' ? 1 : -1) + ids.length) % ids.length]; return 'moved'; }
      if (ed.sel === null || ed.sel === -1) return null;
      const dx = { ArrowLeft: -step, ArrowRight: step }[e.key] || 0, dy = { ArrowUp: -step, ArrowDown: step }[e.key] || 0;
      if (dx || dy) { const q = ed.sel === -2 ? ed.m.dog : S[ed.sel], n = this.place([q[0] + dx, q[1] + dy]); if (ed.sel === -2) ed.m.dog = n; else S[ed.sel] = n; return 'changed'; }
      if ((e.key === 'Delete' || e.key === 'Backspace') && ed.sel >= 0) { if (S.length <= 1) { ed.note = 'Keep one sheep at least.'; return 'moved'; } S.splice(ed.sel, 1); ed.sel = -1; return 'changed'; }
      return null;
    },
    tell(ed) { const q = ed.sel === -2 ? ed.m.dog : ed.m.sheep[ed.sel]; return q ? `${ed.sel === -2 ? 'the dog' : `sheep ${ed.sel + 1} of ${ed.m.sheep.length}`} at ${fmt(q[0], 1)} m across, ${fmt(q[1], 1)} m down` : ''; },
  };
  const KINDS = { maze: MAZE, track: TRACK, hurdles: HURDLES, herd: HERD };
  WE.KINDS = KINDS;
  // which kind of layout a world can have (the problem's task → the editor's kind)
  WE.kindOf = (task) => (KINDS[task] ? task : null);
  // a layout of your own is in use (its own choice made and the layout fine)
  WE.inUse = function (task, s) {
    if (task === 'maze') return s.maze === 'own' && RE().mazeCheck(s.mazeRows).ok;
    if (task === 'track') return s.track === 'own' && RE().trackCheck(s.trackPts).ok;
    if (task === 'hurdles') return s.obstacles === 'own' && RE().courseCheck(s.obst, s.course).ok;
    if (task === 'herd') return s.sheepAt === 'own' && RE().flockCheck(s.flock, s.dogAt).ok;
    return false;
  };

  // ------------------------------------------------------------------ what the world pays
  function paysText(world, s) {
    if (s.rFormula) return `It is paid by your formula, r = ${s.rFormula}.`;
    const on = (RE().REWARDS[world] || []).map((r) => [r, typeof s[r.key] === 'number' ? s[r.key] : r.def]).filter(([r, v]) => v && !(r.key === 'rCloser' && world === 'maze' && !s.closer));
    return on.length ? `What it is paid: ${on.map(([r, v]) => `${r.label.charAt(0).toLowerCase()}${r.label.slice(1)} ${signed(v)}`).join('; ')}.` : 'It is paid nothing at all yet.';
  }

  // ------------------------------------------------------------------ the editor
  let open = null;
  WE.isOpen = () => !!open;
  WE.open = function (lab, kind, from) {
    const K = KINDS[kind];
    if (!K || open) return null;
    if (lab.training) lab.pause();
    if (lab.youPlay) lab.playYourself(false);
    const s = JSON.parse(JSON.stringify(lab.ps()));
    const ed = { lab, kind, K, s, m: K.from(s), sel: -1, tool: K.tools ? K.tools[0][0] : null, hist: [], cur: { x: 0, y: 0 }, keys: false, note: '', touch: false, opener: from || document.activeElement };
    if (kind === 'maze') { const M = new (RL().Maze)(ed.m.rows); ed.cur = { x: M.start % 8, y: Math.floor(M.start / 8) }; }
    const canvas = el('canvas', { class: 'we-canvas', tabindex: '0', role: 'img', 'aria-label': `${K.title}. ${K.help}` });
    const status = el('p', { class: 'we-status', 'aria-live': 'polite' }), tellLine = el('p', { class: 'we-tell sr-only', 'aria-live': 'polite' });
    const toolRow = el('div', { class: 'we-tools', role: 'toolbar', 'aria-label': 'Tools' });
    const useBtn = el('button', { class: 'btn btn-primary', type: 'button', onclick: () => use() }, K.use);
    const undoBtn = el('button', { class: 'btn btn-quiet', type: 'button', title: 'Undo (Ctrl+Z)', onclick: () => undo() }, '↶ Undo');
    const extraRow = el('span', { class: 'we-extra' });
    const startSel = NA.labUI.select([{ value: '', label: 'Start from…' }, ...K.starts(ed).map((o, i) => ({ value: String(i), label: o.label }))], '', (v) => { if (v === '') return; push(); ed.m = K.starts(ed)[+v].make(); ed.sel = -1; ed.note = ''; startSel.value = ''; draw(); }, { class: 'we-start', 'aria-label': 'Start from another layout' });
    if (K.tools) K.tools.forEach(([value, label]) => toolRow.append(el('button', { class: 'btn we-tool', type: 'button', 'aria-pressed': String(value === ed.tool), dataset: { tool: value }, onclick: () => { ed.tool = value; syncTools(); canvas.focus(); } }, label)));
    toolRow.append(startSel, undoBtn, extraRow);
    const stage = el('div', { class: 'we-stage' }, canvas);
    const payBtn = el('button', { class: 'btn btn-quiet', type: 'button', onclick: () => { close(); if (NA.mobile && NA.mobile.mobile) NA.mobile.go('insp'); lab.tabs.select('reward'); } }, 'Change what it pays');
    const dlg = el('div', { class: 'modal we-modal', role: 'dialog', 'aria-modal': 'true', 'aria-labelledby': 'weTitle' },
      el('div', { class: 'sheet we-sheet' },
        el('div', { class: 'sheet-head' },
          el('div', null, el('h2', { class: 'sheet-title', id: 'weTitle', text: K.title }), el('p', { class: 'sheet-sub', text: K.sub })),
          el('button', { class: 'icon-btn', type: 'button', 'data-close': '', 'aria-label': 'Close without using it' }, '×')),
        toolRow, stage, status, tellLine,
        el('p', { class: 'we-pay' }, el('span', { text: paysText(K.world, s) + ' ' }), payBtn),
        el('p', { class: 'we-help', text: K.help }),
        el('div', { class: 'btn-row we-actions' }, useBtn, el('button', { class: 'btn', type: 'button', 'data-close': '' }, 'Cancel'))));
    function syncTools() { toolRow.querySelectorAll('.we-tool').forEach((b) => b.setAttribute('aria-pressed', String(b.dataset.tool === ed.tool))); }
    function push() { ed.hist.push(JSON.stringify(ed.m)); if (ed.hist.length > 80) ed.hist.shift(); }
    function undo() { if (!ed.hist.length) return; ed.m = JSON.parse(ed.hist.pop()); ed.sel = -1; ed.note = ''; draw(); }
    function draw() {
      const w = Math.max(240, stage.clientWidth || 600), h = Math.round(K.height ? K.height(ed, w) : Math.min(w * K.aspect, Math.max(260, window.innerHeight * 0.55)));
      const dpr = Math.min(2, window.devicePixelRatio || 1);
      if (canvas.width !== Math.round(w * dpr) || canvas.height !== Math.round(h * dpr)) { canvas.width = Math.round(w * dpr); canvas.height = Math.round(h * dpr); canvas.style.height = h + 'px'; }
      const ctx = canvas.getContext('2d');
      ctx.setTransform(dpr, 0, 0, dpr, 0, 0); ctx.clearRect(0, 0, w, h);
      ed.r = { x: 0, y: 0, w, h };
      const c = K.check(ed);
      K.draw(ed, ctx, ed.r);
      ed.ok = c.ok;
      status.textContent = (ed.note ? ed.note + ' ' : '') + c.say;
      status.classList.toggle('is-bad', !c.ok);
      useBtn.disabled = !c.ok; undoBtn.disabled = !ed.hist.length;
      // (its own buttons are made once and kept: one drawn again under the pointer would never get its click)
      (K.extra ? K.extra(ed) : []).forEach((x, i) => {
        let b = extraRow.children[i];
        if (!b) { b = el('button', { class: 'btn btn-quiet', type: 'button', onclick: () => { const now = K.extra(ed)[i]; if (!now.on) return; push(); now.run(); draw(); } }, x.label); extraRow.append(b); }
        b.disabled = !x.on;
      });
      tellLine.textContent = ed.keys && K.tell ? K.tell(ed) : '';
    }
    ed.draw = draw;
    const pos = (e) => { const q = canvas.getBoundingClientRect(); return { x: e.clientX - q.left, y: e.clientY - q.top }; };
    let before = null;
    canvas.addEventListener('pointerdown', (e) => {
      if (e.button) return;
      e.preventDefault(); canvas.focus({ preventScroll: true });
      ed.touch = e.pointerType === 'touch'; ed.keys = false; ed.note = ''; ed.grab = false;
      before = JSON.stringify(ed.m);
      try { canvas.setPointerCapture(e.pointerId); } catch (err) { /* (a pointer the browser has already let go of) */ }
      K.down(ed, pos(e)); draw();
    });
    canvas.addEventListener('pointermove', (e) => { if (before === null) return; K.move(ed, pos(e)); draw(); });
    const up = () => { if (before === null) return; if (K.up) K.up(ed); if (JSON.stringify(ed.m) !== before) { ed.hist.push(before); if (ed.hist.length > 80) ed.hist.shift(); } before = null; ed.grab = false; draw(); };
    canvas.addEventListener('pointerup', up); canvas.addEventListener('pointercancel', up);
    canvas.addEventListener('keydown', (e) => {
      if (e.ctrlKey || e.metaKey || e.altKey) return;
      const was = JSON.stringify(ed.m); ed.keys = true; ed.note = '';
      const r = K.key(ed, e);
      if (!r) return;
      e.preventDefault(); e.stopPropagation();
      if (r === 'changed' && JSON.stringify(ed.m) !== was) { ed.hist.push(was); if (ed.hist.length > 80) ed.hist.shift(); }
      draw();
    });
    canvas.addEventListener('focus', () => { ed.keys = true; draw(); });
    canvas.addEventListener('blur', () => { ed.keys = false; draw(); });
    dlg.addEventListener('click', (e) => { if (e.target === dlg || e.target.closest('[data-close]')) close(); });
    dlg.addEventListener('keydown', (e) => {
      if (e.key === 'Escape') { e.preventDefault(); e.stopPropagation(); close(); return; }
      if ((e.ctrlKey || e.metaKey) && (e.key === 'z' || e.key === 'Z') && !e.shiftKey) { e.preventDefault(); e.stopPropagation(); undo(); }
    });
    const onResize = () => draw();
    function use() {
      if (!K.check(ed).ok) return;
      const m = ed.m;
      close(true);
      lab.edit((S) => K.put(S.ps[S.problem], m));
      NA.ui.toast.show(`${K.title} is in use: the next run trains on it, and a share link carries it. Undo takes it back.`);
      refocus();
    }
    // back to the button that opened it (drawn again when the panel was: its successor, the editor's button there)
    function refocus() {
      const b = ed.opener && ed.opener.isConnected ? ed.opener : document.querySelector('#labRight .bd-own-row button');
      if (b) b.focus({ preventScroll: true });
    }
    function close(used) {
      if (!open) return;
      window.removeEventListener('resize', onResize);
      dlg.remove(); open = null;
      // (Your own chosen in the panel but nothing used: the panel, drawn again, shows what is in use)
      if (!used) { lab._blockKey = null; lab.updateTab(); refocus(); }
    }
    open = { ed, dlg, close, use, draw };
    document.body.append(dlg);
    window.addEventListener('resize', onResize);
    draw();
    requestAnimationFrame(() => { draw(); const first = toolRow.querySelector('button, select'); if (first) first.focus(); });
    return open;
  };
  WE.current = () => open;
  WE.close = () => { if (open) open.close(); };
  NA.worldEdit = WE;
})(window.NA);
