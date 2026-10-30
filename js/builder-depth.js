/* builder-depth.js — what goes on inside learning from rewards, drawn in the model builder's results beside the world
   being played: what the network looked at for the move it picks (each input nudged a little, and how much the
   move's lead over the other moves changes: signed bars named by the world's inputs, and for a picture a heat map
   over the world or beside it), and for the evolutionary methods how varied the population is generation by
   generation (in how it plays and in its weights), the family line of the best (its ancestors, each with what
   changed and its score) and a map of the styles of play the members of recent generations ended up with (novelty
   search's archive too). Which views a run has (views) is shared with the switcher above the results. Drawn on the
   results canvas; every part leaves hits whose tip() gives its tooltip. */
(function (NA) {
  'use strict';
  const { clamp, fmt, fmtInt, theme } = NA;
  const BD = {};
  const font = (w, px, mono) => `${w} ${px}px ${NA.getFont ? NA.getFont(mono ? 'mono' : 'ui') : mono ? 'monospace' : 'sans-serif'}`;
  const text = (ctx, s, x, y, o = {}) => { ctx.font = font(o.w || 500, o.size || 11, o.mono); ctx.fillStyle = o.color || theme.c('ink'); ctx.textAlign = o.align || 'left'; ctx.textBaseline = o.base || 'middle'; ctx.fillText(s, x, y); };
  // a caption that fits its width: shortened with an ellipsis rather than cut by the edge
  const fit = (ctx, s, w, size = 10.5, weight = 600, mono) => { ctx.font = font(weight, size, mono); if (ctx.measureText(s).width <= w) return s; let t = s; while (t.length > 1 && ctx.measureText(t + '…').width > w) t = t.slice(0, -1); return t.trimEnd() + '…'; };
  // a note in up to max lines of width w (the last one shortened if it still does not fit); returns the lines
  const lines = (ctx, s, w, size = 10.5, max = 2) => {
    ctx.font = font(500, size);
    const out = [];
    let cur = '';
    for (const word of s.split(' ')) {
      const t = cur ? cur + ' ' + word : word;
      if (ctx.measureText(t).width <= w || !cur) cur = t;
      else { out.push(cur); cur = word; }
    }
    if (cur) out.push(cur);
    if (out.length > max) { out.length = max; out[max - 1] = fit(ctx, out[max - 1] + ' …', w, size, 500); }
    return out;
  };
  const note = (ctx, ls, x, y, size = 10.5) => ls.forEach((l, i) => text(ctx, l, x, y + i * (size + 3), { size, color: theme.c('muted') }));
  // the longest of several wordings that fits w (the shortest, shortened, when none does)
  const fitting = (ctx, list, w, size, weight) => { ctx.font = font(weight, size); return list.find((s) => ctx.measureText(s).width <= w) || fit(ctx, list[list.length - 1], w, size, weight); };
  const signed = (v, d = 2) => (v > 0 && fmt(v, d) !== fmt(0, d) ? '+' : '') + fmt(v, d);
  const el = (...a) => NA.el(...a);
  const tipTitle = (s) => el('div', { class: 'tt-title', text: s });
  const tipRow = (k, v) => el('div', { class: 'tt-row' }, el('span', { class: 'tt-k', text: k }), el('span', { class: 'tt-v', text: v }));
  const tipHint = (s) => el('div', { class: 'tt-hint', text: s });
  const now = () => (typeof performance !== 'undefined' ? performance.now() : Date.now());
  const isEvo = (m) => !!(NA.brl && NA.brl.isEvo(m));

  // ------------------------------------------------------------------ the views of a run
  /* What the switcher above the results offers a run that learns from rewards: o { method, world, narrow, evo }.
     On a wide screen the best's moves come with what it looked at, so that has a view of its own only on a small
     one (and with MAP-Elites, whose grid sits under the moves). The family line needs a method that breeds (and a
     run that keeps it), the map of styles the world's two measures. */
  BD.views = function (o) {
    const m = o.method, evo = isEvo(m), Wd = (NA.rlEnvs.WORLDS || {})[o.world] || {}, ev = o.evo, out = [];
    out.push({ value: 'best', label: evo && m !== 'me' ? 'The best plays' : 'It plays', title: 'The network playing with its best moves, live, and its outputs for what it sees now' });
    if (o.narrow || m === 'me') out.push({ value: 'looked', label: 'What it looked at', title: 'Which inputs decide the move it picks now: each one nudged a little, and how much that changes its choice' });
    if (evo && m !== 'me') out.push({ value: 'generation', label: 'The whole generation', title: Wd.trace ? 'Every member of the last generation at once, from the same start: the colours match their bars. And how varied they are, generation by generation' : 'Every member of the last generation, best first, and how varied they are, generation by generation' });
    if (m === 'me' && o.narrow) out.push({ value: 'grid', label: 'The grid of styles', title: 'The best network of each style of play: tap a cell to watch it' });
    if (evo && m !== 'es' && m !== 'cem' && ev && typeof ev.lineage === 'function') out.push({ value: 'family', label: 'Family line', title: 'The best network’s ancestors, generation by generation: what changed in each, and its score' });
    if (evo && m !== 'me' && Wd.axes) out.push({ value: 'styles', label: 'Styles of play', title: `Where each member’s way of playing put it: ${Wd.axes[0]} across, ${Wd.axes[1]} up` });
    return out;
  };
  // the view to draw: the one picked if this run has it, otherwise the best plays
  BD.view = (L, wide) => { const v = L.evoView || 'best', d = L.data; return v !== 'best' && BD.views({ method: d.method, world: d.world, narrow: !wide, evo: L.evo }).some((x) => x.value === v) ? v : 'best'; };

  // ------------------------------------------------------------------ what it looked at
  /* Inputs with the same name move together (the track camera's speed fills a whole channel of the picture): the
     inputs in groups, one a name, in the order they first come. */
  const GROUPS = new WeakMap();
  function groupsOf(names, n) {
    if (names && GROUPS.has(names) && GROUPS.get(names).n === n) return GROUPS.get(names).G;
    const at = new Map(), G = [];
    for (let i = 0; i < n; i++) {
      const name = names && names[i] !== undefined ? names[i] : `input ${i + 1}`;
      let g = at.get(name);
      if (!g) { g = { name, idx: [] }; at.set(name, g); G.push(g); }
      g.idx.push(i);
    }
    if (names) GROUPS.set(names, { n, G });
    return G;
  }
  // its lead for move a over the average of the others: what decides the move (outputs that all rise together leave it as it was)
  const lead = (o, at, a, nA) => { let s = 0; for (let k = 0; k < nA; k++) if (k !== a) s += o[at + k]; return o[at + a] - s / Math.max(1, nA - 1); };
  /* What it looked at: each input (each group of one name) nudged up and down by 0.05, and how much the lead of move
     a changes over that 0.1. A picture's many inputs are nudged up only, by 0.1, for half the passes. All in batches
     of forward passes, 128 at a time (so a picture does not grow the network's buffers a hundredfold). Returns
     { groups, eff }: eff[j] is the change for group j. */
  BD.saliency = function (net, obs, a, nA, names) {
    const n = obs.length, G = groupsOf(names, n), m = G.length, one = m > 64, rows = one ? m + 1 : 2 * m, h = one ? 0.1 : 0.05, CH = 128;
    const K = net.outSize || net.outK, X = new Float32Array(Math.min(CH, rows) * n), lv = new Float64Array(rows);
    for (let r0 = 0; r0 < rows; r0 += CH) {
      const B = Math.min(CH, rows - r0);
      for (let b = 0; b < B; b++) {
        const row = r0 + b, x0 = b * n, g = one ? (row ? G[row - 1] : null) : G[row >> 1], s = one || !(row & 1) ? h : -h;
        X.set(obs, x0);
        if (g) for (const i of g.idx) X[x0 + i] += s;
      }
      const o = net.forward(X, B, false);
      for (let b = 0; b < B; b++) lv[r0 + b] = lead(o, b * K, a, nA);
    }
    const eff = new Float64Array(m);
    for (let j = 0; j < m; j++) eff[j] = one ? lv[j + 1] - lv[0] : lv[2 * j] - lv[2 * j + 1];
    return { groups: G, eff };
  };
  /* The saliency of the move on show (L.demo), worked out again when the move changes, but not more often than its
     cost allows: about 30 times its cost apart while training (a picture through convolutions takes ~10 ms), 5 when
     not, so it never takes more than a few per cent of the time. Kept in L.depth. */
  BD.looked = function (L) {
    const D = L.demo, S = L.depth, net = L.net, d = L.data;
    if (!S) return null;
    if (!net || !D || !D.out || D.a === null || D.a === undefined || L.you || !D.obs || D.obs.length !== net.inSize || !Number.isFinite(D.out[D.a])) return S.sal || null;
    const key = `${D.ep}:${D.t}:${L.vizVersion}:${L.watchCell}`;
    if (S.sal && S.sal.key === key) return S.sal;
    const t = now();
    if (S.sal && S.sal.ep === D.ep && S.sal.cell === L.watchCell && t - S.sal.at < (L.training ? 30 : 5) * (S.cost || 0)) return S.sal;
    let r;
    try { r = BD.saliency(net, D.obs, D.a, d.nA, d.obsNames); } catch (e) { return S.sal || null; }
    const c = now() - t;
    S.cost = S.cost ? 0.7 * S.cost + 0.3 * c : c;
    S.sal = Object.assign(r, { key, at: t, ep: D.ep, t: D.t, a: D.a, cell: L.watchCell, obs: Float32Array.from(D.obs) });
    return S.sal;
  };
  // the groups in order of how much they count (the largest change first)
  const ranked = (S) => { if (!S.order) S.order = Array.from(S.eff.keys()).sort((i, j) => Math.abs(S.eff[j]) - Math.abs(S.eff[i])); return S.order; };
  /* How a picture's inputs lie on it: for each group, its square (−1 for a group spread over many, like the camera's
     speed), from the input's place in the picture's c × h × w. */
  function pixelsOf(S, input) {
    if (S.pix) return S.pix;
    const n = input.h * input.w;
    return (S.pix = S.groups.map((g) => (g.idx.length === 1 ? g.idx[0] % n : -1)));
  }
  /* Where the inputs lie on the world's own picture, for the worlds drawn as a grid of squares: the maze (its picture,
     the square it is on, or the 24 squares around it) and catch's board. Returns { W, H, sq: square of each group
     (−1: none) } or null. */
  function squaresOf(S, L) {
    const d = L.data, D = L.demo, snap = D && D.snap;
    if (d.world === 'catch' && d.input.kind === 'image') return { W: 5, H: 8, sq: pixelsOf(S, d.input) };
    if (d.world !== 'maze' || !snap || !snap.rows) return null;
    const W = snap.rows[0].length, H = snap.rows.length, n = W * H, view = d.input.kind === 'image' ? 'picture' : d.obsNames.length && /^on square/.test(d.obsNames[0]) ? 'square' : 'around';
    if (view === 'picture') return { W, H, sq: pixelsOf(S, d.input) };
    if (view === 'square') return { W, H, sq: S.groups.map((g) => g.idx[0] % n) };
    const MN = NA.rl && NA.rl.MazeNet, s = snap.s, x0 = s % W, y0 = Math.floor(s / W);
    if (!MN || !MN.inputAt) return null;
    // around it: each input's offset from the square it is on (inputs left out are not in the list, so through keep)
    return { W, H, sq: S.groups.map((g) => { const at = MN.inputAt(d.keep ? d.keep[g.idx[0]] : g.idx[0]); if (!at) return -1; const x = x0 + at.dx, y = y0 + at.dy; return x < 0 || y < 0 || x >= W || y >= H ? -1 : y * W + x; }) };
  }
  // whether what it looked at is drawn over the world itself (a grid of squares it sees): the maze, catch's pixels
  BD.overlays = (d) => d.world === 'maze' || (d.world === 'catch' && d.input.kind === 'image');
  // the saliency of each square (several inputs on one square add up), and the largest
  function heat(S, sq, n) {
    const v = new Float64Array(n);
    sq.forEach((p, j) => { if (p >= 0 && p < n) v[p] += S.eff[j]; });
    let mx = 0;
    for (let i = 0; i < n; i++) mx = Math.max(mx, Math.abs(v[i]));
    return { v, mx };
  }
  // the k squares that count most
  const topOf = (v, k) => Array.from(v.keys()).filter((i) => v[i]).sort((i, j) => Math.abs(v[j]) - Math.abs(v[i])).slice(0, k);
  const moveName = (d, a) => (d.actions && d.actions[a]) || `move ${a + 1}`;
  // the inputs of square p, for a tooltip
  const namesOn = (S, sq, p) => S.groups.filter((g, j) => sq[j] === p).map((g, j) => g.name);
  const effTip = (S, d, title, v, names) => [tipTitle(title), tipRow(`its lead for ‘${moveName(d, S.a)}’`, `${signed(v, 3)}`), names && names.length ? tipHint(names.slice(0, 4).join(' · ') + (names.length > 4 ? ` · ${names.length - 4} more` : '')) : null, tipHint('The change when the input is nudged up by 0.1: + makes this move more likely, − less.')];

  /* Over the world's picture, for a grid world (maze, catch): each square tinted by how much it counts (blue: toward
     the move it picks, orange: away), the three that count most outlined. Its hit gives each square's tooltip. */
  BD.overlay = function (ctx, wr, L) {
    const S = L.depth && L.depth.sal, d = L.data;
    if (!S) return;
    const Q = squaresOf(S, L);
    if (!Q) return;
    const n = Q.W * Q.H, { v, mx } = heat(S, Q.sq, n);
    if (!mx) return;
    const cs = Math.min(wr.w / Q.W, wr.h / Q.H), x0 = wr.x + (wr.w - cs * Q.W) / 2, y0 = wr.y + (wr.h - cs * Q.H) / 2, stale = S.ep !== (L.demo && L.demo.ep);
    for (let p = 0; p < n; p++) {
      const u = v[p] / mx;
      if (Math.abs(u) < 0.04) continue;
      ctx.fillStyle = theme.divColor(u, (stale ? 0.5 : 1) * (0.14 + 0.5 * Math.abs(u)));
      ctx.fillRect(x0 + (p % Q.W) * cs + 1, y0 + Math.floor(p / Q.W) * cs + 1, cs - 2, cs - 2);
    }
    ctx.strokeStyle = theme.c('ink', stale ? 0.4 : 0.85); ctx.lineWidth = 2;
    for (const p of topOf(v, 3)) { NA.rr(ctx, x0 + (p % Q.W) * cs + 2, y0 + Math.floor(p / Q.W) * cs + 2, cs - 4, cs - 4, 4); ctx.stroke(); }
    L.hits.push({ x: x0, y: y0, w: cs * Q.W, h: cs * Q.H, tip: (px, py) => {
      const p = clamp(Math.floor((py - y0) / cs), 0, Q.H - 1) * Q.W + clamp(Math.floor((px - x0) / cs), 0, Q.W - 1);
      return v[p] ? effTip(S, d, `Square (${p % Q.W}, ${Math.floor(p / Q.W)})`, v[p], namesOn(S, Q.sq, p)) : [tipTitle(`Square (${p % Q.W}, ${Math.floor(p / Q.W)})`), tipHint('Nudging what it sees here changes nothing about this move.')];
    } });
  };

  /* A picture input drawn as the network gets it (the brightest of its channels, in grey), tinted by how much each
     square counts, the three that count most outlined, what the colours say under it (r.capW wide): for the
     pictures that are not the world's own (the track's camera, the arena's view). Returns the size used. */
  function drawPicture(ctx, r, L, S) {
    const d = L.data, I = d.input, n = I.h * I.w, pix = pixelsOf(S, I), { v, mx } = heat(S, pix, n), cs = Math.max(2, Math.min(r.w / I.w, r.h / I.h)), pw = cs * I.w, ph = cs * I.h, x0 = r.x, y0 = r.y;
    const c = document.createElement('canvas'); c.width = I.w; c.height = I.h;
    const g = c.getContext('2d'), img = g.createImageData(I.w, I.h), bg = theme.tokens['panel-2'] || [240, 240, 240], ink = theme.tokens['ink-2'] || [80, 80, 80];
    for (let p = 0; p < n; p++) {
      let b = 0;
      for (let ch = 0; ch < I.c; ch++) b = Math.max(b, S.obs[ch * n + p] || 0);
      b = clamp(b, 0, 1) * 0.6;
      const u = mx ? v[p] / mx : 0, hc = theme.divRGB(u), al = Math.abs(u) < 0.04 ? 0 : 0.2 + 0.7 * Math.abs(u);
      for (let k = 0; k < 3; k++) { const base = bg[k] + (ink[k] - bg[k]) * b; img.data[p * 4 + k] = base + (hc[k] - base) * al; }
      img.data[p * 4 + 3] = 255;
    }
    g.putImageData(img, 0, 0);
    ctx.imageSmoothingEnabled = false; ctx.drawImage(c, x0, y0, pw, ph); ctx.imageSmoothingEnabled = true;
    ctx.strokeStyle = theme.c('line-2'); ctx.lineWidth = 1; ctx.strokeRect(x0 + 0.5, y0 + 0.5, pw - 1, ph - 1);
    ctx.strokeStyle = theme.c('ink', 0.9); ctx.lineWidth = 1.5;
    for (const p of topOf(v, 3)) ctx.strokeRect(x0 + (p % I.w) * cs + 0.75, y0 + Math.floor(p / I.w) * cs + 0.75, cs - 1.5, cs - 1.5);
    // what the colours say, under it (r.capW wide; none when it is 0)
    const cap = r.capW ? lines(ctx, `What it sees, ${I.h} × ${I.w}. Blue makes ‘${moveName(d, S.a)}’ likelier, orange less; the 3 that count most are outlined.`, r.capW, 10.5, 3) : [];
    note(ctx, cap, x0, y0 + ph + 9);
    L.hits.push({ x: x0, y: y0, w: pw, h: ph, tip: (px, py) => {
      const p = clamp(Math.floor((py - y0) / cs), 0, I.h - 1) * I.w + clamp(Math.floor((px - x0) / cs), 0, I.w - 1), names = S.groups.filter((gr, j) => pix[j] === p).map((gr) => gr.name);
      return v[p] ? effTip(S, d, `Pixel (${p % I.w}, ${Math.floor(p / I.w)})`, v[p], names) : [tipTitle(`Pixel (${p % I.w}, ${Math.floor(p / I.w)})`), tipHint('Nudging this pixel changes nothing about this move.')];
    } });
    return { w: pw, h: ph + 4 + cap.length * 13.5 };
  }

  /* What it looked at: the inputs that decide the move it picks now, as signed bars about a zero line (blue toward the
     move, orange away; the top 8 at most, as many as fit), named by the world's inputs, with a picture's heat map
     beside or above them when the picture is not the world's own. o.note: room for the line on how to read it.
     Returns the height used. */
  BD.drawLooked = function (ctx, r, L, o = {}) {
    const d = L.data, D = L.demo, S = BD.looked(L);
    text(ctx, fit(ctx, S ? `What it looked at to pick ‘${moveName(d, S.a)}’` : 'What it looked at', r.w, 11, 650), r.x, r.y + 4, { size: 11, w: 650, color: theme.c('muted') });
    if (!S) { text(ctx, fit(ctx, L.net ? 'Between episodes: it looks again at the next move.' : 'Random moves for now: train it, then see what it looks at.', r.w, 11.5, 500), r.x, r.y + 24, { size: 11.5, color: theme.c('ink-2') }); return 34; }
    const stale = !D || S.ep !== D.ep || !D.out;
    let y = r.y + 16, x = r.x, w = r.w, pic = false;
    // a picture that is not the world's own (its colours explained under it): beside the bars when the panel is wide
    // and short, above them when not, as large as leaves room for 3 bars
    if (d.input.kind === 'image' && !squaresOf(S, L)) {
      // (short of room, the picture goes without its caption: the bars' signs say what the colours mean)
      const I = d.input, asp = I.w / I.h, avail = r.y + r.h - y, side = w >= 330 && avail < 260, cap = avail >= (side ? 150 : 190);
      const ph = side ? Math.min(avail - (cap ? 46 : 4), (w * 0.42) / asp) : clamp(Math.min(avail - (cap ? 46 : 8) - 3 * 16, w / asp), 40, 170), pw = ph * asp, colW = side ? Math.max(pw, cap ? 130 : 0) : w;
      const used = drawPicture(ctx, { x, y: y + 2, w: pw, h: ph, capW: cap ? colW : 0 }, L, S);
      if (side) { x += colW + 14; w -= colW + 14; } else y += used.h + 8;
      pic = true;
    }
    // the lines on how to read it (a picture has them under it), unless they would leave room for fewer than 4 bars
    const more = S.groups.length > 8 ? ` The ${Math.min(8, S.groups.length)} of its ${fmtInt(S.groups.length)} inputs that count most.` : '';
    let say = pic || o.note === false ? [] : lines(ctx, stale ? `From its last move.${more}` : `Each input nudged up a little: blue makes this move likelier, orange less.${more}`, r.w, 10.5, 2);
    if (say.length && r.y + r.h - y - say.length * 13.5 - 6 < 4 * 15) say = [];
    const room = r.y + r.h - y - (say.length ? say.length * 13.5 + 6 : 2), order = ranked(S), rowH = clamp(Math.floor(room / 8), 15, 18), rows = Math.min(8, order.length, Math.floor(room / rowH));
    if (rows < 1) return y - r.y;
    const lab = Math.min(140, w * 0.46), bx = x + lab + 6, bw = Math.max(30, w - lab - 6 - 50), z = bx + bw / 2;
    let mx = 0;
    for (let k = 0; k < rows; k++) mx = Math.max(mx, Math.abs(S.eff[order[k]]));
    for (let k = 0; k < rows; k++) {
      const j = order[k], v = S.eff[j], g = S.groups[j], yy = y + k * rowH, len = mx ? (Math.abs(v) / mx) * (bw / 2) : 0;
      text(ctx, fit(ctx, g.name, lab, 11, 500), x, yy + rowH / 2, { size: 11, color: theme.c('ink-2') });
      ctx.fillStyle = theme.c('line', 0.5); NA.rr(ctx, bx, yy + 3, bw, rowH - 6, 3); ctx.fill();
      if (len >= 0.5) { ctx.fillStyle = theme.c(v >= 0 ? 'pos' : 'neg', stale ? 0.45 : 0.9); NA.rr(ctx, v >= 0 ? z : z - len, yy + 3, Math.max(2, len), rowH - 6, 3); ctx.fill(); }
      text(ctx, Math.abs(v) < 0.0005 ? '0.000' : signed(v, Math.abs(v) < 0.1 ? 3 : 2), bx + bw + 6, yy + rowH / 2, { size: 10.5, mono: true, color: theme.c(stale ? 'muted' : 'ink-2') });
      const val = S.obs[g.idx[0]];
      L.hits.push({ x, y: yy, w, h: rowH, tip: () => [tipTitle(g.name), tipRow('what it sees now', fmt(val, 2)), tipRow(`its lead for ‘${moveName(d, S.a)}’`, signed(v, 3)), tipHint(`Nudged up by 0.1${g.idx.length > 1 ? ` (all ${g.idx.length} inputs of this name)` : ''}, this input ${v > 0 ? 'makes the move more likely' : v < 0 ? 'makes the move less likely' : 'changes nothing'}. ${k ? `Number ${k + 1} of its ${fmtInt(S.groups.length)} inputs` : `The one of its ${fmtInt(S.groups.length)} inputs that counts most`}.`)] });
    }
    // the zero line
    ctx.strokeStyle = theme.c('ink-2', 0.55); ctx.lineWidth = 1; ctx.beginPath(); ctx.moveTo(Math.round(z) + 0.5, y + 1); ctx.lineTo(Math.round(z) + 0.5, y + rows * rowH - 1); ctx.stroke();
    y += rows * rowH;
    if (say.length) { note(ctx, say, r.x, y + 9); y += say.length * 13.5 + 6; }
    return y - r.y;
  };

  // ------------------------------------------------------------------ where the return came from
  /* The episode on show, term by term of the world's reward table (L.demo.parts: { key: { sum, n } }, added up from
     the parts each step reports; none when your own reward formula pays): a bar of the return, the terms that paid
     to the right of zero and those that cost to the left, each named below with its total and how often it came
     (−1.00 walking into a wall, 1×). Only for worlds whose steps report their parts. */
  const PART_COLOURS = ['s1', 's3', 's4', 's7', 's5', 's2', 's8', 's6'];
  BD.hasRewards = (L) => { const D = L.demo; if (D && D.parts) D.partsSeen = true; return !!(D && D.partsSeen); };
  BD.drawRewards = function (ctx, r, L) {
    const d = L.data, D = L.demo, P = (D && D.parts) || {}, table = (NA.rlEnvs.REWARDS || {})[d.world] || [], keys = Object.keys(P).filter((k) => P[k].n);
    const label = (k) => { const e = table.find((t) => t.key === k); return e ? e.label.replace(/ \(.*\)$/, '').replace(/^./, (c) => c.toLowerCase()) : k; };
    const colour = (k) => { const i = table.findIndex((t) => t.key === k); return PART_COLOURS[(i < 0 ? table.length + keys.indexOf(k) : i) % PART_COLOURS.length]; };
    keys.sort((a, b) => P[b].sum - P[a].sum);
    const total = keys.reduce((s, k) => s + P[k].sum, 0), last = D && D.last && D.last.parts ? Object.keys(D.last.parts).filter((k) => D.last.parts[k]) : [];
    const lastText = last.length ? `last move: ${last.map((k) => `${signed(D.last.parts[k], 2)} ${label(k)}`).join(', ')}` : '';
    text(ctx, fitting(ctx, [`Where its return of ${fmt(total, 2)} came from, this episode`, `Its return of ${fmt(total, 2)}, term by term`, 'Its return, term by term'], r.w * (lastText ? 0.5 : 1), 11, 650), r.x, r.y + 5, { size: 11, w: 650, color: theme.c('muted') });
    if (lastText) text(ctx, fit(ctx, lastText, r.w * 0.48, 10.5, 500), r.x + r.w, r.y + 5, { size: 10.5, align: 'right', color: theme.c('ink-2') });
    const by = r.y + 14, bh = 12;
    ctx.fillStyle = theme.c('line', 0.45); NA.rr(ctx, r.x, by, r.w, bh, 3); ctx.fill();
    if (!keys.length) { text(ctx, 'Nothing paid or cost yet this episode.', r.x + 6, by + bh / 2, { size: 10.5, color: theme.c('muted') }); return; }
    // the bar: what cost to the left of zero, what paid to the right, in proportion
    const pos = keys.filter((k) => P[k].sum > 0), neg = keys.filter((k) => P[k].sum < 0), sp = pos.reduce((s, k) => s + P[k].sum, 0), sn = -neg.reduce((s, k) => s + P[k].sum, 0), span = sp + sn || 1, z = r.x + (sn / span) * r.w;
    const tip = (k) => () => [tipTitle(label(k).replace(/^./, (c) => c.toUpperCase())), tipRow('this episode', signed(P[k].sum, 2)), tipRow('times', fmtInt(P[k].n)), tipRow('each time, on average', signed(P[k].sum / P[k].n, 3)), tipHint(`Of its return of ${fmt(total, 2)} so far. The reward table says what each thing pays; the network learns to collect as much as it can.`)];
    const seg = (list, from, dir) => { let x = from; for (const k of list) { const w = (Math.abs(P[k].sum) / span) * r.w, x0 = dir > 0 ? x : x - w; ctx.fillStyle = theme.c(colour(k), 0.5); ctx.fillRect(x0, by, Math.max(1, w), bh); ctx.fillStyle = theme.c(colour(k)); ctx.fillRect(x0, by + bh - 3, Math.max(1, w), 3); if (w > 38) text(ctx, signed(P[k].sum, Math.abs(P[k].sum) < 10 ? 2 : 1), x0 + w / 2, by + bh / 2, { size: 10, align: 'center', mono: true, w: 600, color: theme.c('ink') }); L.hits.push({ x: x0, y: by - 2, w: Math.max(4, w), h: bh + 4, tip: tip(k) }); x += dir * w; } };
    seg(pos, z, 1); seg(neg.slice().reverse(), z, -1);
    if (sn && sp) { ctx.strokeStyle = theme.c('ink', 0.8); ctx.lineWidth = 1.5; ctx.beginPath(); ctx.moveTo(Math.round(z) + 0.5, by - 3); ctx.lineTo(Math.round(z) + 0.5, by + bh + 3); ctx.stroke(); }
    // the terms by name, in the order of the bar: what paid, then what cost
    let x = r.x, y = by + bh + 10;
    ctx.font = font(500, 10.5);
    for (const k of [...pos, ...neg]) {
      const s = `${signed(P[k].sum, 2)} ${label(k)}${P[k].n > 1 ? ` (${fmtInt(P[k].n)}×)` : ''}`, w = ctx.measureText(s).width + 16;
      if (x > r.x && x + w > r.x + r.w) { x = r.x; y += 14; }
      if (y > r.y + r.h - 4) break;
      ctx.fillStyle = theme.c(colour(k), 0.9); ctx.fillRect(x, y - 4, 8, 8);
      text(ctx, fit(ctx, s, r.x + r.w - x - 12, 10.5, 500), x + 11, y, { size: 10.5, color: theme.c('ink-2') });
      L.hits.push({ x, y: y - 6, w: Math.min(w, r.x + r.w - x), h: 12, tip: tip(k) });
      x += w + 6;
    }
  };

  // ------------------------------------------------------------------ PPO's clip at work
  /* The share of the moves in each batch whose probability the steps moved past the clip (more than 20% from the
     policy that played): the trainer's record of it, read from what it keeps (agent.clipHist: numbers or { update,
     share }; or agent.history entries with clip, clipShare or clipFrac). Null until it keeps one. */
  BD.clipSeries = function (agent) {
    if (!agent) return null;
    const src = Array.isArray(agent.clipHist) ? agent.clipHist : Array.isArray(agent.history) ? agent.history : null;
    if (!src || src.length < 2) return null;
    const val = (h) => (typeof h === 'number' ? h : h && [h.share, h.clip, h.clipShare, h.clipFrac, h.clipped].find((v) => Number.isFinite(v)));
    const pts = src.map((h, i) => [h && Number.isFinite(h.update) ? h.update : h && Number.isFinite(h.step) ? h.step : i, val(h)]).filter((p) => Number.isFinite(p[1]));
    return pts.length >= 2 ? pts : null;
  };
  BD.drawClip = function (ctx, r, L) {
    const P = BD.clipSeries(L.agent);
    if (!P) return 0;
    text(ctx, fitting(ctx, ['The clip at work: moves whose probability moved more than 20%', 'The clip at work'], r.w, 11, 650), r.x, r.y + 4, { size: 11, w: 650, color: theme.c('muted') });
    const q = { x: r.x, y: r.y + 14, w: r.w, h: r.h - 14 }, last = P[P.length - 1], x0 = P[0][0], x1 = Math.max(x0 + 1, last[0]);
    let hi = 0;
    for (const p of P) hi = Math.max(hi, p[1]);
    hi = Math.max(0.05, hi * 1.15);
    const top = q.y + 14, h = q.h - 16, X = (v) => q.x + ((v - x0) / (x1 - x0)) * q.w, Y = (v) => top + h - (v / hi) * h;
    text(ctx, 'share of each batch', q.x, q.y + 4, { size: 10.5, color: theme.c('ink-2') });
    text(ctx, NA.fmtPct(last[1], 0), q.x + q.w, q.y + 4, { size: 10.5, align: 'right', mono: true, color: theme.c('ink') });
    ctx.fillStyle = theme.c('panel-2'); NA.rr(ctx, q.x, top, q.w, h, 4); ctx.fill();
    ctx.strokeStyle = theme.c('s2'); ctx.lineWidth = 1.6; ctx.lineJoin = 'round'; ctx.beginPath();
    P.forEach((p, i) => (i ? ctx.lineTo(X(p[0]), Y(p[1])) : ctx.moveTo(X(p[0]), Y(p[1])))); ctx.stroke();
    L.hits.push({ x: q.x, y: top, w: q.w, h, tip: (px) => {
      const xv = x0 + ((px - q.x) / q.w) * (x1 - x0), p = P.reduce((a, b) => (Math.abs(b[0] - xv) < Math.abs(a[0] - xv) ? b : a), P[0]);
      return [tipTitle(`Update ${fmtInt(p[0])}`), tipRow('moves past the clip', NA.fmtPct(p[1], 1)), tipHint('PPO takes several small steps on each batch of moves; a move whose probability has moved more than 20% from the policy that played it stops pulling. A high share: the steps are large, and the clip is holding them back.')];
    } });
    return r.h;
  };

  // ------------------------------------------------------------------ the population over the generations
  /* Each generation seen while the run is drawn: its members' styles (for the map) and, when the run does not keep
     its own, how varied it was. A generation that went by between two pictures is missed (only the diversity the
     run keeps itself has every one). */
  BD.observe = function (L) {
    const ev = L.evo, S = L.depth;
    if (!ev || !S || !ev.last || S.seenGen === ev.last.gen) return;
    S.seenGen = ev.last.gen;
    const M = ev.last.members || [];
    (S.recent || (S.recent = [])).push({ gen: ev.last.gen, pts: M.map((m, k) => ({ m, k })).filter((p) => p.m.beh && Number.isFinite(p.m.beh[0]) && Number.isFinite(p.m.beh[1])) });
    if (S.recent.length > 8) S.recent.shift();
    if (keepsDiversity(ev)) return;
    (S.divHist || (S.divHist = [])).push({ gen: ev.last.gen, div: behSpread(M), divW: weightSpread(ev) });
    if (S.divHist.length > 600) S.divHist = S.divHist.filter((h, i) => i % 2 === 0 || i === S.divHist.length - 1);
  };
  const keepsDiversity = (ev) => ev.history.length > 0 && ev.history.some((h) => Number.isFinite(h.div) || Number.isFinite(h.divW) || Number.isFinite(h.spread));
  // how far apart the members' styles are: their average distance from the average style, in the world's two measures (as the engine measures it)
  function behSpread(M) {
    const B = M.map((m) => m.beh).filter((b) => b && Number.isFinite(b[0]) && Number.isFinite(b[1])), n = B.length;
    if (n < 2) return null;
    let cx = 0, cy = 0, s = 0;
    for (const b of B) { cx += b[0] / n; cy += b[1] / n; }
    for (const b of B) s += Math.hypot(b[0] - cx, b[1] - cy);
    return s / n;
  }
  // how far apart their weights are: the typical distance of a weight from the population's average (of the members of the commonest shape)
  function weightSpread(ev) {
    const P = (ev.pop || []).filter((g) => g && g.w);
    if (P.length < 2) return null;
    const count = new Map();
    for (const g of P) count.set(g.w.length, (count.get(g.w.length) || 0) + 1);
    const len = [...count.entries()].sort((a, b) => b[1] - a[1])[0][0], Q = P.filter((g) => g.w.length === len), mean = new Float64Array(len);
    if (Q.length < 2) return null;
    for (const g of Q) for (let i = 0; i < len; i++) mean[i] += g.w[i] / Q.length;
    let s = 0;
    for (const g of Q) for (let i = 0; i < len; i++) s += (g.w[i] - mean[i]) ** 2;
    return Math.sqrt(s / (Q.length * len));
  }
  // the diversity of each generation: the run's own (div, divW; CEM's spread), or what was seen of it
  BD.diversity = function (ev, S) {
    if (keepsDiversity(ev)) return ev.history.map((h) => ({ gen: h.gen, div: Number.isFinite(h.div) ? h.div : null, divW: Number.isFinite(h.divW) ? h.divW : Number.isFinite(h.spread) ? h.spread : null, spread: Number.isFinite(h.spread) && !Number.isFinite(h.divW) }));
    return (S && S.divHist) || [];
  };

  /* How varied the population is, generation by generation: two small charts, how far apart the members' styles of
     play are and how far apart their weights are, side by side when there is room. Falling lines: the population
     is settling on one way of playing, and only mutation still explores. Returns the height used. */
  BD.drawDiversity = function (ctx, r, L, ev, o = {}) {
    const H = BD.diversity(ev, L.depth), cur = L.data.method;
    const series = [{ key: 'div', name: 'how far apart their styles are', short: 'styles apart' }, { key: 'divW', name: H.some((h) => h.spread) ? 'how far apart their weights are (its spread)' : 'how far apart their weights are', short: 'weights apart' }]
      .map((s) => Object.assign(s, { pts: H.filter((h) => h[s.key] !== null && h[s.key] !== undefined).map((h) => [h.gen, h[s.key]]) })).filter((s) => s.pts.length >= 2);
    text(ctx, fitting(ctx, ['How varied the population is, generation by generation', 'How varied it is, generation by generation', 'How varied it is'], r.w, 11, 650), r.x, r.y + 4, { size: 11, w: 650, color: theme.c('muted') });
    if (!series.length) { text(ctx, fit(ctx, cur === 'es' ? 'It shows from the second generation on.' : 'Two generations, and the lines begin.', r.w, 11, 500), r.x, r.y + 22, { size: 11, color: theme.c('ink-2') }); return 30; }
    let say = o.note !== false && r.h >= 96 ? lines(ctx, 'When these fall, the population is settling on one way of playing, and only mutation still explores.', r.w, 10.5, 3) : [];
    if (say.length > 2) say = lines(ctx, 'Falling: it is settling on one way of playing.', r.w, 10.5, 2);
    const noteH = say.length ? say.length * 13.5 + 4 : 0, side = r.w >= 280 && series.length > 1, ch = clamp((side ? r.h - 18 - noteH : (r.h - 18 - noteH) / series.length - 6), 34, 70), cw = side ? (r.w - 14) / 2 : r.w;
    series.forEach((s, i) => {
      const q = { x: r.x + (side ? i * (cw + 14) : 0), y: r.y + 16 + (side ? 0 : i * (ch + 6)), w: cw, h: ch };
      if (q.y + q.h > r.y + r.h + 1) return;
      miniChart(ctx, q, s, L, H);
    });
    const used = 16 + (side ? ch : series.length * (ch + 6));
    if (say.length && r.y + used + noteH <= r.y + r.h + 2) { note(ctx, say, r.x, r.y + used + 9); return used + noteH; }
    return used;
  };
  // one line over the generations: its name and latest value above, a faint zero line, recessive axes
  function miniChart(ctx, q, s, L, H) {
    const P = s.pts, last = P[P.length - 1], g0 = P[0][0], g1 = Math.max(g0 + 1, last[0]);
    let hi = 0;
    for (const p of P) hi = Math.max(hi, p[1]);
    hi = hi * 1.1 || 1;
    const top = q.y + 14, h = q.h - 16, X = (g) => q.x + ((g - g0) / (g1 - g0)) * q.w, Y = (v) => top + h - (v / hi) * h;
    text(ctx, fit(ctx, s.short, q.w - 46, 10.5, 500), q.x, q.y + 4, { size: 10.5, color: theme.c('ink-2') });
    text(ctx, fmt(last[1], last[1] < 0.1 ? 3 : 2), q.x + q.w, q.y + 4, { size: 10.5, align: 'right', mono: true, color: theme.c('ink') });
    ctx.fillStyle = theme.c('panel-2'); NA.rr(ctx, q.x, top, q.w, h, 4); ctx.fill();
    ctx.strokeStyle = theme.c('line-2'); ctx.lineWidth = 1; ctx.beginPath(); ctx.moveTo(q.x, Math.round(top + h) - 0.5); ctx.lineTo(q.x + q.w, Math.round(top + h) - 0.5); ctx.stroke();
    ctx.strokeStyle = theme.c('s1'); ctx.lineWidth = 1.6; ctx.lineJoin = 'round'; ctx.beginPath();
    P.forEach((p, i) => (i ? ctx.lineTo(X(p[0]), Y(p[1])) : ctx.moveTo(X(p[0]), Y(p[1])))); ctx.stroke();
    ctx.fillStyle = theme.c('s1'); ctx.beginPath(); ctx.arc(X(last[0]), Y(last[1]), 2.6, 0, Math.PI * 2); ctx.fill();
    const hv = L.depth && L.depth.hoverGen;
    if (hv !== undefined && hv !== null && hv >= g0 && hv <= g1) { ctx.strokeStyle = theme.c('ink', 0.4); ctx.beginPath(); ctx.moveTo(Math.round(X(hv)) + 0.5, top); ctx.lineTo(Math.round(X(hv)) + 0.5, top + h); ctx.stroke(); }
    L.hits.push({ x: q.x, y: top, w: q.w, h, key: 'div', tip: (px) => {
      const gv = g0 + ((px - q.x) / q.w) * (g1 - g0), hh = H.reduce((a, b) => (Math.abs(b.gen - gv) < Math.abs(a.gen - gv) ? b : a), H[0]);
      if (L.depth) L.depth.hoverGen = hh.gen;
      return [tipTitle(`Generation ${fmtInt(hh.gen + 1)}`), hh.div !== null && hh.div !== undefined ? tipRow('styles apart', fmt(hh.div, 3)) : null, hh.divW !== null && hh.divW !== undefined ? tipRow(hh.spread ? 'its spread' : 'weights apart', fmt(hh.divW, 3)) : null,
        tipHint('Styles apart: how far a member’s style of play is, on average, from the population’s average style (the world’s two measures, 0 to 1 each). Weights apart: how far a weight typically is from the population’s average.')];
    } });
  }

  // ------------------------------------------------------------------ the family line of the best
  // what changed from parent to child, in a few words, and a mark that says it without colour
  const OPS = {
    start: { mark: 'star', say: 'the first generation' }, kept: { mark: 'square', say: 'passed on unchanged' }, mixed: { mark: 'diamond', say: 'mixed from two parents' },
    grew: { mark: 'up', say: 'grew' }, shrank: { mark: 'down', say: 'shrank' }, rl: { mark: 'ring', say: 'copied in from the learner' }, nudged: { mark: 'dot', say: 'weights mutated' },
  };
  /* The kind of change, from the engine's words for it (builder-evo.js: 'mutated', 'mixed and mutated', '+1 neuron in
     layer 2', '−1 neuron: layer 2 gone', 'the network you built', 'a copy moved at random', 'random', 'the DQN’s
     network'; null: passed on unchanged) */
  function opKind(a, prev) {
    const op = String(a.op || '').toLowerCase();
    if (a.fromRL || /learner|\brl\b|dqn|gradient/.test(op)) return 'rl';
    if (/^\+|grew|added|new layer/.test(op)) return 'grew';
    if (/^[−-]|pruned|removed|shrank|folded|gone/.test(op)) return 'shrank';
    if (/mix|cross|parents/.test(op) || (a.parents && a.parents.length > 1)) return 'mixed';
    if (/you built|at random|^random|first generation|drawn/.test(op) || (!prev && !op)) return 'start';
    if (/unchanged|kept|carried|passed on/.test(op)) return 'kept';
    return 'nudged';
  }
  const shapeText = (u) => (u && u.length ? u.join(' → ') : 'no hidden layer');
  const opText = (a, kind) => (a.op && kind !== 'kept' && kind !== 'start' && kind !== 'nudged' ? String(a.op) : OPS[kind].say);
  function mark(ctx, kind, x, y, s, fill, stroke) {
    ctx.beginPath();
    const m = OPS[kind].mark;
    if (m === 'square') ctx.rect(x - s * 0.8, y - s * 0.8, s * 1.6, s * 1.6);
    else if (m === 'diamond') { ctx.moveTo(x, y - s * 1.15); ctx.lineTo(x + s * 1.15, y); ctx.lineTo(x, y + s * 1.15); ctx.lineTo(x - s * 1.15, y); ctx.closePath(); }
    else if (m === 'up') { ctx.moveTo(x, y - s * 1.2); ctx.lineTo(x + s * 1.1, y + s * 0.8); ctx.lineTo(x - s * 1.1, y + s * 0.8); ctx.closePath(); }
    else if (m === 'down') { ctx.moveTo(x, y + s * 1.2); ctx.lineTo(x + s * 1.1, y - s * 0.8); ctx.lineTo(x - s * 1.1, y - s * 0.8); ctx.closePath(); }
    else if (m === 'star') { for (let k = 0; k < 10; k++) { const a = -Math.PI / 2 + (k * Math.PI) / 5, rr = k % 2 ? s * 0.55 : s * 1.3; k ? ctx.lineTo(x + rr * Math.cos(a), y + rr * Math.sin(a)) : ctx.moveTo(x + rr * Math.cos(a), y + rr * Math.sin(a)); } ctx.closePath(); }
    else ctx.arc(x, y, s * (m === 'ring' ? 1.05 : 0.9), 0, Math.PI * 2);
    if (m === 'ring') { ctx.lineWidth = 2; ctx.strokeStyle = stroke; ctx.stroke(); return; }
    ctx.fillStyle = fill; ctx.fill();
    if (stroke) { ctx.lineWidth = 1.2; ctx.strokeStyle = stroke; ctx.stroke(); }
  }
  // the best's ancestors, oldest first, each with what changed (cached until a new generation or a new best)
  BD.family = function (ev, S) {
    const top = ev.last && ev.last.members && ev.last.members[0], id = typeof ev.bestId === 'function' ? ev.bestId() : top && top.id;
    if (id === undefined || id === null || typeof ev.lineage !== 'function') return null;
    const key = `${ev.gen}:${id}`;
    if (S && S.line && S.line.key === key) return S.line.list;
    let list = [];
    try { list = (ev.lineage(id) || []).filter((a) => a && Number.isFinite(a.gen)).slice().sort((a, b) => a.gen - b.gen); } catch (e) { list = []; }
    list.forEach((a, i) => { a.kind = opKind(a, i ? list[i - 1] : null); });
    if (S) S.line = { key, list };
    return list;
  };
  /* The family line: each ancestor's score against its generation (a line through them, a mark for what changed),
     then the most recent changes, newest first. Hover a mark or a row for its details. */
  BD.drawFamily = function (ctx, r, L, ev) {
    const list = BD.family(ev, L.depth) || [];
    if (!list.length) { text(ctx, fit(ctx, 'The family line starts after the first generation.', r.w, 11.5, 500), r.x, r.y + 8, { size: 11.5, color: theme.c('ink-2') }); return; }
    const first = list[0], lastA = list[list.length - 1], hov = L.depth ? L.depth.hoverId : null;
    const n = list.length - 1;
    text(ctx, fitting(ctx, [`The best’s family line: ${n ? `${fmtInt(n)} ancestor${n === 1 ? '' : 's'}, back to generation ${fmtInt(first.gen + 1)}` : 'no ancestors yet'}`, `The best’s ${fmtInt(n)} ancestors`], r.w, 11, 650), r.x, r.y + 4, { size: 11, w: 650, color: theme.c('muted') });
    // the chart (its generation labels inside the picture); rows below it when there is room
    let listRows = Math.max(0, Math.min(8, list.length, Math.floor((r.h - 120 - 38) / 17)));
    if (listRows < 2) listRows = 0;
    const chartH = clamp(r.h - (listRows ? listRows * 17 + 38 : 14), 70, 220);
    const q = { x: r.x + 34, y: r.y + 18, w: r.w - 40, h: chartH - 18 };
    let lo = Infinity, hi = -Infinity;
    for (const a of list) if (Number.isFinite(a.fit)) { lo = Math.min(lo, a.fit); hi = Math.max(hi, a.fit); }
    if (!Number.isFinite(lo)) { lo = 0; hi = 1; }
    const sp = hi - lo || Math.abs(hi) || 1; lo -= sp * 0.1; hi += sp * 0.12;
    const g0 = first.gen, g1 = Math.max(g0 + 1, lastA.gen), X = (g) => q.x + ((g - g0) / (g1 - g0)) * q.w, Y = (v) => q.y + q.h - ((v - lo) / (hi - lo)) * q.h;
    ctx.fillStyle = theme.c('panel-2'); NA.rr(ctx, q.x, q.y, q.w, q.h, 6); ctx.fill();
    ctx.strokeStyle = theme.c('line'); ctx.lineWidth = 1;
    for (const t of NA.niceTicks ? NA.niceTicks(lo, hi, 3) : [lo, hi]) { const y = Math.round(Y(t)) + 0.5; if (y < q.y + 2 || y > q.y + q.h - 2) continue; ctx.beginPath(); ctx.moveTo(q.x, y); ctx.lineTo(q.x + q.w, y); ctx.stroke(); text(ctx, fmt(t, Math.abs(t) < 10 ? 1 : 0), q.x - 5, y, { size: 10, align: 'right', mono: true, color: theme.c('muted') }); }
    text(ctx, `gen ${fmtInt(g0 + 1)}`, q.x, q.y + q.h + 9, { size: 10, mono: true, color: theme.c('muted') });
    text(ctx, `gen ${fmtInt(g1 + 1)}`, q.x + q.w, q.y + q.h + 9, { size: 10, mono: true, align: 'right', color: theme.c('muted') });
    // a step line: each ancestor's score holds until its child
    ctx.strokeStyle = theme.c('s1', 0.8); ctx.lineWidth = 1.6; ctx.beginPath();
    list.forEach((a, i) => { if (!Number.isFinite(a.fit)) return; const x = X(a.gen), y = Y(a.fit); if (!i) ctx.moveTo(x, y); else { ctx.lineTo(x, Y(list[i - 1].fit)); ctx.lineTo(x, y); } });
    ctx.stroke();
    // marks: the changes to the shape and the mixes always; nudges too when they are few enough to see
    const every = list.length <= Math.max(12, q.w / 7), sz = list.length > 60 ? 3 : 3.8;
    list.forEach((a, i) => {
      if (!Number.isFinite(a.fit) || (!every && (a.kind === 'nudged' || a.kind === 'kept') && i !== list.length - 1)) return;
      const x = X(a.gen), y = Y(a.fit), on = hov !== null && hov === a.id;
      mark(ctx, a.kind, x, y, on ? sz + 1.5 : sz, theme.c(a.kind === 'grew' ? 's3' : a.kind === 'shrank' ? 's2' : a.kind === 'mixed' ? 's7' : a.kind === 'rl' ? 's4' : 's1'), theme.c('panel'));
      L.hits.push({ x: x - 6, y: y - 6, w: 12, h: 12, id: a.id, cx: x, cy: y, tip: () => famTip(a, list[i - 1]) });
    });
    // the most recent changes, newest first
    if (!listRows) return;
    let y = r.y + chartH + 27;
    text(ctx, 'The latest first', r.x, y, { size: 10.5, w: 600, color: theme.c('muted') });
    y += 9;
    const recent = list.slice(-listRows).reverse();
    recent.forEach((a, k) => {
      const yy = y + k * 17, on = hov !== null && hov === a.id, prev = list[list.length - 2 - k];
      if (on) { ctx.fillStyle = theme.c('line', 0.6); ctx.fillRect(r.x, yy, r.w, 17); }
      mark(ctx, a.kind, r.x + 7, yy + 8.5, 3.6, theme.c(a.kind === 'grew' ? 's3' : a.kind === 'shrank' ? 's2' : a.kind === 'mixed' ? 's7' : a.kind === 'rl' ? 's4' : 's1'), null);
      text(ctx, `gen ${fmtInt(a.gen + 1)}`, r.x + 18, yy + 8.5, { size: 10.5, mono: true, color: theme.c('ink-2') });
      text(ctx, Number.isFinite(a.fit) ? fmt(a.fit, Math.abs(a.fit) < 10 ? 2 : 1) : '—', r.x + 92, yy + 8.5, { size: 10.5, mono: true, align: 'right', color: theme.c('ink') });
      text(ctx, fit(ctx, opText(a, a.kind) + (a.kind === 'grew' || a.kind === 'shrank' ? ` (${shapeText(a.units)})` : ''), r.w - 104, 11, 500), r.x + 102, yy + 8.5, { size: 11, color: theme.c(a.kind === 'nudged' || a.kind === 'kept' ? 'ink-2' : 'ink') });
      L.hits.push({ x: r.x, y: yy, w: r.w, h: 17, id: a.id, tip: () => famTip(a, prev) });
    });
  };
  function famTip(a, prev) {
    const d = prev && Number.isFinite(prev.fit) && Number.isFinite(a.fit) ? a.fit - prev.fit : null;
    return [tipTitle(`Generation ${fmtInt(a.gen + 1)}`), tipRow('fitness', Number.isFinite(a.fit) ? fmt(a.fit, 2) : '—'), d !== null ? tipRow('than its parent', signed(d, 2)) : null, tipRow('what changed', opText(a, a.kind)), tipRow('its shape', shapeText(a.units)),
      tipHint(a.kind === 'kept' ? 'One of the best two of its generation: it passed on as it was.' : a.kind === 'start' ? 'The first generation: the network you built and copies of it moved at random.' : a.kind === 'rl' ? 'A copy of the network that learns by gradient steps, put into the population.' : 'A child of the generation before, changed by evolution.')];
  }

  // ------------------------------------------------------------------ the map of styles
  /* Where each member's style of play put it, in the world's two measures (0 to 1 each, the axes MAP-Elites' grid
     uses): the members of the last few generations, older ones fainter, the best of the latest ringed; novelty
     search's archive of novel styles as hollow diamonds. With o.diversity (a small screen, where the whole
     generation's view has no room for it) how varied the population is, beside it or under it. Hover a point for
     its details. */
  BD.drawStyles = function (ctx, r, L, ev, Wd, o = {}) {
    const axes = (ev && ev.axes) || Wd.axes || ['style 1', 'style 2'], R = (L.depth && L.depth.recent) || [], arch = ev && Array.isArray(ev.novArchive) ? ev.novArchive : null;
    const nowGen = R.length ? R[R.length - 1].gen : null;
    text(ctx, fit(ctx, `Styles of play: the last ${R.length > 1 ? `${R.length} generations seen` : 'generation'}${arch ? ', and the archive of novel ones' : ''}`, r.w, 11, 650), r.x, r.y + 4, { size: 11, w: 650, color: theme.c('muted') });
    // the square, and the diversity beside it or under it, whichever leaves the square larger
    // (alone, a line under it says how to read it)
    const pad = 30, besideS = o.diversity ? Math.min(r.h - 46, r.w - pad - 10 - 176) : 0, underS = o.diversity ? Math.min(r.w - pad - 10, r.h - 46 - 104) : 0, alone = Math.min(r.w - pad - 10, r.h - 46 - (o.diversity ? 0 : 16));
    const lay = besideS >= 110 && besideS >= underS ? 'beside' : underS >= 110 ? 'under' : 'alone', side = lay === 'beside' ? besideS : lay === 'under' ? underS : alone;
    // (points kept 5 px inside its edges)
    const q = { x: r.x + pad, y: r.y + 18, s: Math.max(40, side) }, X = (v) => q.x + 5 + clamp(v, 0, 1) * (q.s - 10), Y = (v) => q.y + q.s - 5 - clamp(v, 0, 1) * (q.s - 10);
    ctx.fillStyle = theme.c('panel-2'); ctx.fillRect(q.x, q.y, q.s, q.s);
    ctx.strokeStyle = theme.c('line'); ctx.lineWidth = 1;
    for (const t of [0.25, 0.5, 0.75]) { ctx.beginPath(); ctx.moveTo(Math.round(X(t)) + 0.5, q.y); ctx.lineTo(Math.round(X(t)) + 0.5, q.y + q.s); ctx.moveTo(q.x, Math.round(Y(t)) + 0.5); ctx.lineTo(q.x + q.s, Math.round(Y(t)) + 0.5); ctx.stroke(); }
    ctx.strokeStyle = theme.c('line-2'); ctx.strokeRect(q.x + 0.5, q.y + 0.5, q.s - 1, q.s - 1);
    text(ctx, fit(ctx, `${axes[0]} →`, q.s, 10.5, 500), q.x + q.s / 2, q.y + q.s + 10, { size: 10.5, align: 'center', color: theme.c('muted') });
    ctx.save(); ctx.translate(r.x + 9, q.y + q.s / 2); ctx.rotate(-Math.PI / 2); text(ctx, fit(ctx, `${axes[1]} →`, q.s, 10.5, 500), 0, 0, { size: 10.5, align: 'center', color: theme.c('muted') }); ctx.restore();
    const pts = [], hov = L.depth ? L.depth.hoverPt : null;
    // the archive: hollow diamonds, the latest few hundred
    if (arch) {
      ctx.strokeStyle = theme.c('ink-2', 0.55); ctx.lineWidth = 1;
      for (const a of arch.slice(-400)) {
        if (!a || !a.beh) continue;
        const x = X(a.beh[0]), y = Y(a.beh[1]);
        ctx.beginPath(); ctx.moveTo(x, y - 3.2); ctx.lineTo(x + 3.2, y); ctx.lineTo(x, y + 3.2); ctx.lineTo(x - 3.2, y); ctx.closePath(); ctx.stroke();
        pts.push({ x, y, arch: a });
      }
    }
    // the members: older generations fainter; a member's fitness in the depth of its colour, the best of the latest ringed
    let lo = Infinity, hi = -Infinity;
    for (const g of R) for (const p of g.pts) if (Number.isFinite(p.m.fit)) { lo = Math.min(lo, p.m.fit); hi = Math.max(hi, p.m.fit); }
    R.forEach((g, gi) => {
      const age = R.length - 1 - gi, al = age ? Math.max(0.12, 0.5 - age * 0.06) : 0.95;
      for (const p of g.pts.slice().reverse()) {
        const x = X(p.m.beh[0]), y = Y(p.m.beh[1]), f = Number.isFinite(p.m.fit) && hi > lo ? (p.m.fit - lo) / (hi - lo) : 0.6;
        ctx.fillStyle = theme.seqColor(0.3 + 0.7 * f, al); ctx.beginPath(); ctx.arc(x, y, age ? 2.4 : 3.2, 0, Math.PI * 2); ctx.fill();
        if (!age) { ctx.strokeStyle = theme.c('panel', 0.9); ctx.lineWidth = 0.8; ctx.stroke(); }
        pts.push({ x, y, m: p.m, gen: g.gen, rank: p.k, latest: !age });
      }
    });
    const lastG = R.length ? R[R.length - 1] : null, top = lastG && lastG.pts.find((p) => p.k === 0);
    if (top) {
      // the best of the latest generation ringed and named (the name kept inside the square)
      const x = X(top.m.beh[0]), y = Y(top.m.beh[1]), right = x > q.x + q.s - 64;
      ctx.strokeStyle = theme.c('ink'); ctx.lineWidth = 2; ctx.beginPath(); ctx.arc(x, y, 6.5, 0, Math.PI * 2); ctx.stroke();
      text(ctx, 'the best', right ? x - 10 : x + 10, clamp(y < q.y + 18 ? y + 12 : y - 10, q.y + 8, q.y + q.s - 8), { size: 10.5, w: 600, color: theme.c('ink'), align: right ? 'right' : 'left' });
    }
    if (hov) { ctx.strokeStyle = theme.c('s2'); ctx.lineWidth = 2; ctx.beginPath(); ctx.arc(hov.x, hov.y, 6, 0, Math.PI * 2); ctx.stroke(); }
    if (!R.length) text(ctx, 'The first generation is playing…', q.x + q.s / 2, q.y + q.s / 2, { size: 11.5, align: 'center', color: theme.c('muted') });
    L.hits.push({ x: q.x - 4, y: q.y - 4, w: q.s + 8, h: q.s + 8, key: 'styles', tip: (px, py) => {
      let b = null, bd = 10;
      for (const p of pts) { const dd = Math.hypot(p.x - px, p.y - py); if (dd < bd) { bd = dd; b = p; } }
      if (L.depth) L.depth.hoverPt = b ? { x: b.x, y: b.y } : null;
      if (!b) return null;
      if (b.arch) return [tipTitle('In the archive of novel styles'), tipRow(axes[0], fmt(b.arch.beh[0], 2)), tipRow(axes[1], fmt(b.arch.beh[1], 2)), Number.isFinite(b.arch.gen) ? tipRow('kept in generation', fmtInt(b.arch.gen + 1)) : null, tipHint('Novelty search scores a new member by how far its style is from these and from the population’s.')];
      return [tipTitle(`Generation ${fmtInt(b.gen + 1)}, ${b.rank ? `member ${b.rank + 1} by fitness` : 'its best'}`), tipRow('fitness', fmt(b.m.fit, 2)), Number.isFinite(b.m.nov) ? tipRow('novelty', fmt(b.m.nov, 3)) : null, tipRow(axes[0], fmt(b.m.beh[0], 2)), tipRow(axes[1], fmt(b.m.beh[1], 2)), b.m.fromRL ? tipHint('Copied in from the network that learns by gradient steps.') : null];
    } });
    if (nowGen !== null && lay !== 'alone') {
      const dr = lay === 'beside' ? { x: q.x + q.s + 14, y: r.y + 18, w: r.x + r.w - (q.x + q.s + 14), h: q.s + 10 } : { x: r.x, y: q.y + q.s + 24, w: r.w, h: r.y + r.h - (q.y + q.s + 24) };
      BD.drawDiversity(ctx, dr, L, ev, { note: dr.h >= 110 });
    } else if (r.y + r.h - (q.y + q.s + 20) >= 12) note(ctx, [fitting(ctx, [`Fainter: older generations. Deeper colour: fitter. Ringed: the best now.${arch ? ' ◇: the archive.' : ''}`, `Fainter: older. Deeper: fitter. Ringed: the best now.${arch ? ' ◇: archive.' : ''}`], r.w, 10.5, 500)], r.x, q.y + q.s + 25);
  };

  NA.builderDepth = BD;
})(typeof globalThis !== 'undefined' && globalThis.NA ? globalThis.NA : window.NA);
