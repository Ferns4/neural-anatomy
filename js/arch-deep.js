/* arch-deep.js — the levels below any number in a model, shared by every architecture view. A neuron opens into
   its weighted sum; that into its PyTorch code; a call in the code into what the library does inside; each step
   of that into one float32 multiply or add, bit by bit; those into the integer adders and partial products that
   do them; an adder into full adders; a full adder into logic gates; a gate into its CMOS transistors; a
   transistor into its cross-section; and the channel into the electrons moving through the silicon. Stored
   numbers take the other road down: float32 bits → a DRAM cell → its transistor and capacitor → electrons.
   Every level is built from the real numbers it was opened on, with an explanation and code for the panel. */
(function (NA) {
  'use strict';
  const { fmt, theme, clamp } = NA;
  const D = {};
  const num = (v, d = 4) => NA.fmtCode(v, d); // numbers inside code
  // a Python list of numbers, eight to a line
  const pyList = (vals) => {
    const v = Array.from(vals, (x) => num(x, 4)), rows = [];
    for (let i = 0; i < v.length; i += 8) rows.push(v.slice(i, i + 8).join(', '));
    return rows.length > 1 ? `[\n    ${rows.join(',\n    ')},\n]` : `[${rows.join('')}]`;
  };
  const W2 = (X, w, t) => (X.wide ? w : t);
  const PERIOD = 3600; // one replay of a switching event, in milliseconds
  const phase = () => ((NA.archKit.now || performance.now()) % PERIOD) / PERIOD;
  const f32 = Math.fround;

  // A node whose children are only built when a level needs them: the deep tree is far too big to build whole.
  function node(props, kids) {
    if (kids) {
      let memo = null;
      Object.defineProperty(props, 'children', { get: () => memo || (memo = kids().filter(Boolean)), enumerable: true, configurable: true });
    }
    return props;
  }

  // ------------------------------------------------------------------ float32 arithmetic, the hardware way
  const F = new Float32Array(1), U = new Uint32Array(F.buffer);
  function bitsOf(v) { F[0] = v; const u = U[0]; return { u, s: u >>> 31, e: (u >>> 23) & 255, m: u & 0x7fffff, v: F[0] }; }
  function fromBits(s, e, m) { U[0] = ((s << 31) | (e << 23) | m) >>> 0; return F[0]; }
  const bin = (x, n) => { let s = ''; for (let i = n - 1; i >= 0; i--) s += Math.floor(x / 2 ** i) % 2; return s; };
  const normal = (B) => B.e > 0 && B.e < 255;

  // a × b: XOR the signs, add the exponents, multiply the 24-bit significands, round to nearest even
  function mulSteps(a, b) {
    a = f32(a); b = f32(b);
    const A = bitsOf(a), B = bitsOf(b), c = f32(a * b), o = { a, b, c, A, B, C: bitsOf(c), s: A.s ^ B.s };
    if (!normal(A) || !normal(B)) { o.special = true; return o; }
    const ma = A.m + 0x800000, mb = B.m + 0x800000, prod = ma * mb; // exact: below 2^48
    const top = prod >= 2 ** 47 ? 1 : 0, sh = 23 + top, q = Math.floor(prod / 2 ** sh), rem = prod - q * 2 ** sh, half = 2 ** (sh - 1);
    let m24 = q + (rem > half || (rem === half && q % 2 === 1) ? 1 : 0), e = A.e + B.e - 127 + top;
    if (m24 >= 2 ** 24) { m24 /= 2; e += 1; }
    Object.assign(o, { ma, mb, prod, top, e, m24, rounded: m24 !== q });
    o.ok = e > 0 && e < 255 && fromBits(o.s, e, m24 - 0x800000) === c;
    return o;
  }
  // a + b: line the significands up on the larger exponent (with guard, round and sticky bits), add or
  // subtract, normalise, round to nearest even
  function addSteps(a, b) {
    a = f32(a); b = f32(b);
    const c = f32(a + b), o = { a, b, c, A: bitsOf(a), B: bitsOf(b), C: bitsOf(c) };
    let x = o.A, y = o.B;
    if (!normal(x) || !normal(y)) { o.special = true; return o; }
    if (y.e > x.e || (y.e === x.e && y.m > x.m)) { [x, y] = [y, x]; o.swapped = true; }
    const d = x.e - y.e, pa = (x.m + 0x800000) * 8, pb0 = (y.m + 0x800000) * 8, sh = Math.min(d, 27);
    const sticky = sh && pb0 % 2 ** sh ? 1 : 0, pb = Math.floor(pb0 / 2 ** sh) + (sticky && Math.floor(pb0 / 2 ** sh) % 2 === 0 ? 1 : 0);
    const sub = x.s !== y.s, sum = sub ? pa - pb : pa + pb;
    Object.assign(o, { big: x, small: y, d, pa, pb, sticky, sub, sum });
    if (sum === 0) { o.zero = true; o.ok = c === 0; return o; }
    let s = sum, e = x.e, shifts = 0;
    while (s >= 2 ** 27) { s = Math.floor(s / 2) + (s % 2 && Math.floor(s / 2) % 2 === 0 ? 1 : 0); e++; shifts++; }
    while (s < 2 ** 26) { s *= 2; e--; shifts--; }
    let keep = Math.floor(s / 8);
    const low = s % 8;
    if (low > 4 || (low === 4 && keep % 2)) { keep++; if (keep >= 2 ** 24) { keep /= 2; e++; } }
    Object.assign(o, { e, keep, shifts, low });
    o.ok = e > 0 && e < 255 && fromBits(x.s, e, keep - 0x800000) === c;
    return o;
  }
  // ripple-carry addition of two n-bit numbers, keeping every bit and carry
  function ripple(x, y, n, cin = 0) {
    const r = { a: [], b: [], c: [cin], s: [], n };
    for (let i = 0; i < n; i++) {
      const ai = Math.floor(x / 2 ** i) % 2, bi = Math.floor(y / 2 ** i) % 2, ci = r.c[i];
      r.a.push(ai); r.b.push(bi); r.s.push(ai ^ bi ^ ci); r.c.push((ai & bi) | ((ai ^ bi) & ci));
    }
    r.cout = r.c[n];
    return r;
  }
  const GATE = { AND: (a, b) => a & b, OR: (a, b) => a | b, XOR: (a, b) => a ^ b, NAND: (a, b) => 1 - (a & b), NOR: (a, b) => 1 - (a | b), NOT: (a) => 1 - a };
  // inputs a gate had just before, chosen so that its output flips: the replay shows that switch
  function previous(type, ins) {
    const out = GATE[type](...ins);
    if (ins.length === 1) return [1 - ins[0]];
    const cands = [[1 - ins[0], ins[1]], [ins[0], 1 - ins[1]], [1 - ins[0], 1 - ins[1]]];
    return cands.find((p) => GATE[type](...p) !== out) || cands[0];
  }
  D.math = { bitsOf, fromBits, mulSteps, addSteps, ripple, GATE };

  // ------------------------------------------------------------------ drawing helpers
  let MONO = 0;
  const monoRatio = () => {
    if (!MONO) { const c = document.createElement('canvas').getContext('2d'); c.font = NA.archKit.font(500, 100, true); MONO = c.measureText('MMMMMMMMMM').width / 1000 || 0.6; }
    return MONO;
  };
  // A block of monospaced lines laid out in fractions of the content box, so parts of it can be zoom zones.
  function block(lines, area, maxRow = 0.1) {
    const maxLen = Math.max(10, ...lines.map((l) => l.length)), rowH = Math.min(area[3] / Math.max(1, lines.length), maxRow), colW = (area[2] * 0.985) / maxLen;
    return { lines, area, rowH, colW, maxLen,
      span: (k, col, len) => [area[0] + col * colW - colW * 0.2, area[1] + k * rowH + rowH * 0.1, (len + 0.4) * colW, rowH * 0.8],
      row: (k, len) => [area[0], area[1] + k * rowH + rowH * 0.06, Math.min(area[2], (len + 1) * colW), rowH * 0.88] };
  }
  const KW = new Set(['def', 'return', 'for', 'in', 'if', 'else', 'elif', 'import', 'from', 'as', 'lambda', 'class', 'with', 'not', 'and', 'or', 'True', 'False', 'None', 'while', 'range']);
  function pyTokens(line) {
    const out = [], re = /(#.*$)|('[^']*'|"[^"]*")|(\b\d+\.?\d*(?:e-?\d+)?\b)|([A-Za-z_][\w.]*)|(\s+)|(.)/g;
    let m;
    while ((m = re.exec(line))) {
      const t = m[0];
      const c = m[1] ? theme.c('muted') : m[2] ? theme.c('s3') : m[3] ? theme.c('s4') : m[4] ? (KW.has(t) ? theme.c('s7') : line[re.lastIndex] === '(' ? theme.c('s1') : theme.c('ink')) : theme.c('ink-2');
      out.push({ t, c });
      if (!t) break;
    }
    return out;
  }
  // code: one character per column, so a zone's dotted outline sits exactly on its call or operator
  function drawCode(ctx, S, r, B) {
    const cw = B.colW * r.w, rh = B.rowH * r.h, size = Math.min(rh * 0.62, cw / monoRatio(), 24);
    ctx.font = S.font(500, size, true); ctx.textBaseline = 'middle'; ctx.textAlign = 'left';
    B.lines.forEach((line, k) => {
      const y = r.y + (B.area[1] + (k + 0.5) * B.rowH) * r.h;
      let col = 0;
      for (const tk of pyTokens(line)) { ctx.fillStyle = tk.c; for (const ch of tk.t) { ctx.fillText(ch, r.x + (B.area[0] + col * B.colW) * r.w, y); col++; } }
    });
  }
  // maths rows: each row may be a zone that opens the operation it shows
  function drawRows(ctx, S, r, B, strong) {
    const rh = B.rowH * r.h, size = Math.min(rh * 0.6, (B.colW * r.w) / monoRatio(), 20);
    B.lines.forEach((line, k) => S.text(ctx, line, r.x + B.area[0] * r.w, r.y + (B.area[1] + (k + 0.5) * B.rowH) * r.h, { size, mono: true, w: strong && strong(k) ? 700 : 500, color: line.startsWith('…') ? theme.c('muted') : theme.c('ink'), maxW: B.area[2] * r.w }));
  }
  const FIELD = (i) => (i === 0 ? 's7' : i <= 8 ? 's4' : 's1');
  // 32 bits of a float32: sign (violet), exponent (amber), mantissa (blue); a filled cell is a 1
  function drawBits(ctx, S, r, u, n = 32, field = FIELD) {
    const cw = r.w / n, h = Math.min(r.h, cw * 1.8), y = r.y + (r.h - h) / 2;
    for (let i = 0; i < n; i++) {
      const bit = Math.floor(u / 2 ** (n - 1 - i)) % 2, x = r.x + i * cw, col = field(i);
      ctx.fillStyle = bit ? theme.c(col, 0.85) : theme.c(col, 0.1);
      if (cw > 7) { ctx.beginPath(); NA.rr(ctx, x + 0.75, y, cw - 1.5, h, Math.min(3, cw * 0.18)); ctx.fill(); } else ctx.fillRect(x + 0.5, y, Math.max(1, cw - 1), h);
      if (cw > 9) S.text(ctx, String(bit), x + cw / 2, y + h / 2, { size: Math.min(cw * 0.7, h * 0.7), mono: true, align: 'center', color: bit ? theme.c('panel') : theme.c('muted') });
    }
    return { cw, y, h };
  }
  // A wire: a 1 glows and carries pulses travelling along it; a 0 is a plain dim line.
  function wire(ctx, pts, on, lw = 1.6) {
    const path = () => { ctx.beginPath(); pts.forEach(([x, y], i) => (i ? ctx.lineTo(x, y) : ctx.moveTo(x, y))); };
    ctx.lineJoin = 'round'; ctx.lineCap = 'round';
    if (on) {
      ctx.strokeStyle = theme.c('pos', 0.2); ctx.lineWidth = lw + 5; path(); ctx.stroke();
      ctx.strokeStyle = theme.c('pos'); ctx.lineWidth = lw + 0.8; path(); ctx.stroke();
      if (!NA.prefersReducedMotion()) {
        ctx.strokeStyle = 'rgba(255,255,255,0.8)'; ctx.lineWidth = Math.max(1, lw * 0.8); ctx.setLineDash([2, 16]); ctx.lineDashOffset = -(((NA.archKit.now || performance.now()) / 22) % 18);
        path(); ctx.stroke(); ctx.setLineDash([]); ctx.lineDashOffset = 0;
      }
    } else { ctx.strokeStyle = theme.c('line-2'); ctx.lineWidth = lw; path(); ctx.stroke(); }
    ctx.lineCap = 'butt';
  }
  const ON = () => theme.c('pos'), OFF = () => theme.c('line-2');
  // the classic gate symbols, drawn into r
  function gateGlyph(ctx, type, r, out) {
    const s = Math.min(r.w / 1.5, r.h), w = s * 1.25, h = s * 0.9, x = r.x + (r.w - w) / 2, y = r.y + (r.h - h) / 2, inv = type === 'NAND' || type === 'NOR' || type === 'NOT';
    const bw = inv ? w - s * 0.16 : w;
    ctx.beginPath();
    if (type === 'AND' || type === 'NAND') { ctx.moveTo(x, y); ctx.lineTo(x + bw * 0.5, y); ctx.arc(x + bw * 0.5, y + h / 2, h / 2, -Math.PI / 2, Math.PI / 2); ctx.lineTo(x, y + h); ctx.closePath(); }
    else if (type === 'NOT') { ctx.moveTo(x, y); ctx.lineTo(x + bw, y + h / 2); ctx.lineTo(x, y + h); ctx.closePath(); }
    else { ctx.moveTo(x, y); ctx.quadraticCurveTo(x + bw * 0.6, y, x + bw, y + h / 2); ctx.quadraticCurveTo(x + bw * 0.6, y + h, x, y + h); ctx.quadraticCurveTo(x + bw * 0.25, y + h / 2, x, y); }
    ctx.fillStyle = out ? theme.c('pos', 0.22) : theme.c('panel-2'); ctx.fill();
    ctx.strokeStyle = theme.c('ink-2'); ctx.lineWidth = 1.3; ctx.stroke();
    if (type === 'XOR') { ctx.beginPath(); ctx.moveTo(x - s * 0.1, y); ctx.quadraticCurveTo(x + bw * 0.25 - s * 0.1, y + h / 2, x - s * 0.1, y + h); ctx.stroke(); }
    if (inv) { ctx.beginPath(); ctx.arc(x + bw + s * 0.08, y + h / 2, s * 0.07, 0, Math.PI * 2); ctx.stroke(); }
  }
  // a MOSFET symbol: channel (vertical), gate plate, a bubble on the gate for PMOS
  function fetGlyph(ctx, kind, r, on) {
    const cx = r.x + r.w * 0.58, top = r.y + r.h * 0.08, bot = r.y + r.h * 0.92, gx = r.x + r.w * 0.44;
    ctx.strokeStyle = theme.c('ink-2'); ctx.lineWidth = 1.3;
    ctx.beginPath(); ctx.moveTo(cx, top); ctx.lineTo(cx, r.y + r.h * 0.28); ctx.lineTo(cx + r.w * 0.0, r.y + r.h * 0.28); ctx.moveTo(cx, bot); ctx.lineTo(cx, r.y + r.h * 0.72); ctx.stroke();
    ctx.strokeStyle = on ? theme.c('pos') : theme.c('muted', 0.6); ctx.lineWidth = on ? 3 : 1.5; ctx.setLineDash(on ? [] : [3, 3]);
    ctx.beginPath(); ctx.moveTo(cx, r.y + r.h * 0.26); ctx.lineTo(cx, r.y + r.h * 0.74); ctx.stroke(); ctx.setLineDash([]);
    ctx.strokeStyle = theme.c('ink-2'); ctx.lineWidth = 1.3;
    ctx.beginPath(); ctx.moveTo(gx, r.y + r.h * 0.3); ctx.lineTo(gx, r.y + r.h * 0.7); ctx.stroke();
    const bub = kind === 'p' ? r.w * 0.06 : 0;
    ctx.beginPath(); ctx.moveTo(r.x + r.w * 0.06, r.y + r.h / 2); ctx.lineTo(gx - bub * 2, r.y + r.h / 2); ctx.stroke();
    if (bub) { ctx.beginPath(); ctx.arc(gx - bub, r.y + r.h / 2, bub, 0, Math.PI * 2); ctx.stroke(); }
  }
  // dots moving along a polyline: f in [0, 1) is how far along; used for electrons in wires
  function along(pts, f) {
    let L = 0; const seg = [];
    for (let i = 1; i < pts.length; i++) { const l = Math.hypot(pts[i][0] - pts[i - 1][0], pts[i][1] - pts[i - 1][1]); seg.push(l); L += l; }
    let d = (((f % 1) + 1) % 1) * L;
    for (let i = 0; i < seg.length; i++) { if (d <= seg[i]) { const t = seg[i] ? d / seg[i] : 0; return [pts[i][0] + (pts[i + 1][0] - pts[i][0]) * t, pts[i][1] + (pts[i + 1][1] - pts[i][1]) * t]; } d -= seg[i]; }
    return pts[pts.length - 1];
  }
  // Small pictures drawn once and reused every frame: a carrier's glow, an atom's shading (per theme and size).
  const SPRITES = new Map();
  function sprite(key, rad, draw) {
    const dpr = Math.min(3, window.devicePixelRatio || 1), k = `${key}:${rad.toFixed(1)}:${dpr}:${theme.dark ? 1 : 0}`;
    let c = SPRITES.get(k);
    if (!c) {
      if (SPRITES.size > 160) SPRITES.clear();
      c = document.createElement('canvas'); c.width = c.height = Math.ceil(rad * 2 * dpr) + 2;
      const g = c.getContext('2d'); g.scale(dpr, dpr); draw(g, rad + 1 / dpr, rad);
      SPRITES.set(k, c);
    }
    return c;
  }
  const rgba = (name, a) => { const t = theme.tokens[name] || [128, 128, 128]; return `rgba(${t[0]},${t[1]},${t[2]},${a})`; };
  // an electron: a bright core in a soft blue glow; a hole: a ring with an orange glow (a missing electron)
  function electron(ctx, x, y, rad, hole, alpha = 1) {
    const R = rad * 2.6, s = sprite(hole ? 'hole' : 'electron', R, (g, c, R0) => {
      const gr = g.createRadialGradient(c, c, 0, c, c, R0), col = hole ? 's2' : 's1';
      gr.addColorStop(0, rgba(col, hole ? 0.35 : 0.95)); gr.addColorStop(0.32, rgba(col, hole ? 0.28 : 0.55)); gr.addColorStop(1, rgba(col, 0));
      g.fillStyle = gr; g.beginPath(); g.arc(c, c, R0, 0, Math.PI * 2); g.fill();
      g.beginPath(); g.arc(c, c, R0 / 2.6, 0, Math.PI * 2);
      if (hole) { g.strokeStyle = rgba('s2', 1); g.lineWidth = Math.max(1, R0 / 7); g.stroke(); }
      else { const core = g.createRadialGradient(c - R0 * 0.1, c - R0 * 0.1, 0, c, c, R0 / 2.6); core.addColorStop(0, 'rgba(255,255,255,0.95)'); core.addColorStop(1, rgba('s1', 1)); g.fillStyle = core; g.fill(); }
    });
    if (alpha < 1) ctx.globalAlpha *= alpha;
    ctx.drawImage(s, x - R - 0.5, y - R - 0.5, s.width / Math.min(3, window.devicePixelRatio || 1), s.height / Math.min(3, window.devicePixelRatio || 1));
    if (alpha < 1) ctx.globalAlpha /= alpha;
  }
  // a silicon atom: a shaded sphere lit from the upper left
  function atom(ctx, x, y, rad) {
    const s = sprite('atom', rad, (g, c, R0) => {
      const gr = g.createRadialGradient(c - R0 * 0.35, c - R0 * 0.38, R0 * 0.08, c, c, R0);
      const hi = theme.dark ? [150, 160, 176] : [205, 210, 218], mid = theme.dark ? [92, 101, 116] : [150, 158, 170], lo = theme.dark ? [48, 54, 64] : [104, 111, 124];
      gr.addColorStop(0, `rgb(${hi})`); gr.addColorStop(0.55, `rgb(${mid})`); gr.addColorStop(1, `rgb(${lo})`);
      g.fillStyle = gr; g.beginPath(); g.arc(c, c, R0, 0, Math.PI * 2); g.fill();
    });
    const d = Math.min(3, window.devicePixelRatio || 1);
    ctx.drawImage(s, x - rad - 0.5, y - rad - 0.5, s.width / d, s.height / d);
  }
  /* Carriers that keep moving between frames (levels are rebuilt on every draw, so their state lives here):
     straight flights at the thermal speed, a random new direction at each collision (a brief flash), and, when a
     field is on, a steady push toward the drain. Positions are fractions of the box; x wraps around. */
  const SIMS = new Map();
  function carriers(key, n) {
    let s = SIMS.get(key);
    if (!s) { if (SIMS.size > 24) SIMS.clear(); s = { ps: [], t: null }; SIMS.set(key, s); }
    while (s.ps.length < n) { const a = Math.random() * Math.PI * 2, i = s.ps.length; s.ps.push({ x: ((i * 0.618034) % 1), y: 0.08 + 0.84 * ((i * 0.41421) % 1), vx: Math.cos(a) * 0.45, vy: Math.sin(a) * 0.45, trail: [], flash: 0 }); }
    return s;
  }
  function stepCarriers(s, now, drift) {
    const dt = s.t === null ? 0 : Math.min(0.05, Math.max(0, (now - s.t) / 1000)), VT = 0.45, TAU = 0.14;
    s.t = now;
    if (NA.prefersReducedMotion()) return;
    for (const p of s.ps) {
      p.vx += drift * dt;
      if (Math.random() < dt / TAU) { const a = Math.random() * Math.PI * 2, sp = VT * (0.55 + Math.random() * 0.9); p.vx = Math.cos(a) * sp; p.vy = Math.sin(a) * sp * 1.4; p.flash = 1; p.fx = p.x; p.fy = p.y; }
      p.x += p.vx * dt; p.y += p.vy * dt;
      if (p.y < 0.04 || p.y > 0.96) { p.vy = -p.vy; p.y = clamp(p.y, 0.04, 0.96); }
      if (p.x > 1 || p.x < 0) { p.x = ((p.x % 1) + 1) % 1; p.trail.length = 0; }
      p.trail.push(p.x, p.y);
      if (p.trail.length > 16) p.trail.splice(0, 2);
      p.flash = Math.max(0, p.flash - dt * 5);
    }
  }
  function drawCarriers(ctx, s, box, rad, hole, count, alpha = 1) {
    const col = hole ? 's2' : 's1';
    s.ps.forEach((p, i) => {
      const a = alpha * clamp(count - i, 0, 1);
      if (a <= 0.01) return;
      // the path since the last few frames, fading out behind
      const tr = p.trail;
      for (let k = 2; k < tr.length; k += 2) {
        ctx.strokeStyle = rgba(col, (0.45 * a * k) / tr.length); ctx.lineWidth = Math.max(1, rad * 0.55);
        ctx.beginPath(); ctx.moveTo(box.x + tr[k - 2] * box.w, box.y + tr[k - 1] * box.h); ctx.lineTo(box.x + tr[k] * box.w, box.y + tr[k + 1] * box.h); ctx.stroke();
      }
      if (p.flash > 0.05) { ctx.strokeStyle = rgba(col, 0.5 * p.flash * a); ctx.lineWidth = 1; ctx.beginPath(); ctx.arc(box.x + p.fx * box.w, box.y + p.fy * box.h, rad * (1.5 + (1 - p.flash) * 3), 0, Math.PI * 2); ctx.stroke(); }
      electron(ctx, box.x + p.x * box.w, box.y + p.y * box.h, rad, hole, a);
    });
  }
  // A crystal of silicon, drawn once per size: atoms in a square lattice, each bond two shared electrons.
  const LATTICE = new Map();
  function lattice(w, h, g) {
    const dpr = Math.min(3, window.devicePixelRatio || 1), key = `${Math.round(w)}x${Math.round(h)}:${g.toFixed(1)}:${dpr}:${theme.dark ? 1 : 0}`;
    let c = LATTICE.get(key);
    if (!c) {
      if (LATTICE.size > 8) LATTICE.clear();
      c = document.createElement('canvas'); c.width = Math.ceil(w * dpr); c.height = Math.ceil(h * dpr);
      const ctx = c.getContext('2d'); ctx.scale(dpr, dpr);
      const bg = ctx.createLinearGradient(0, 0, 0, h);
      bg.addColorStop(0, rgba('panel-2', 1)); bg.addColorStop(1, rgba('panel', 1));
      ctx.fillStyle = bg; ctx.fillRect(0, 0, w, h);
      const off = g / 2, bond = (x1, y1, x2, y2) => {
        const nx = (y2 - y1) / g, ny = (x1 - x2) / g, d = g * 0.07;
        ctx.strokeStyle = rgba('muted', 0.3); ctx.lineWidth = 1;
        ctx.beginPath(); ctx.moveTo(x1 + nx * d, y1 + ny * d); ctx.lineTo(x2 + nx * d, y2 + ny * d); ctx.moveTo(x1 - nx * d, y1 - ny * d); ctx.lineTo(x2 - nx * d, y2 - ny * d); ctx.stroke();
        // the two electrons of the bond, sitting between the atoms
        ctx.fillStyle = rgba('s1', 0.35);
        for (const sgn of [-1, 1]) { ctx.beginPath(); ctx.arc((x1 + x2) / 2 + nx * d * sgn, (y1 + y2) / 2 + ny * d * sgn, Math.max(0.8, g * 0.035), 0, Math.PI * 2); ctx.fill(); }
      };
      for (let y = off; y < h; y += g) for (let x = off; x < w; x += g) { if (x + g < w) bond(x, y, x + g, y); if (y + g < h) bond(x, y, x, y + g); }
      for (let y = off; y < h; y += g) for (let x = off; x < w; x += g) atom(ctx, x, y, g * 0.2);
      LATTICE.set(key, c);
    }
    return c;
  }
  // the replay's gate voltage: the old value, then a quick ramp to the new one as the switch happens
  const ramp = (before, after) => { const p = phase(); return before + (after - before) * clamp((p - 0.25) / 0.07, 0, 1); };
  // the replay's time line: before → inputs switch → output moves → settled
  function timeline(ctx, S, r, labels) {
    const p = phase(), u = S.u(r, 7, 12);
    ctx.fillStyle = theme.c('line'); ctx.fillRect(r.x, r.y + r.h * 0.45, r.w, 2);
    const marks = [0, 0.25, 0.3, 0.7, 1];
    labels.forEach((t, i) => S.text(ctx, t, r.x + ((marks[i] + marks[i + 1]) / 2) * r.w, r.y + r.h * 0.05 + u * 0.4, { size: u * 0.8, align: 'center', color: p >= marks[i] && p < marks[i + 1] ? theme.c('ink') : theme.c('muted'), w: p >= marks[i] && p < marks[i + 1] ? 650 : 500, maxW: (marks[i + 1] - marks[i]) * r.w + 30 }));
    ctx.fillStyle = theme.c('focus'); ctx.beginPath(); ctx.arc(r.x + p * r.w, r.y + r.h * 0.45 + 1, 4, 0, Math.PI * 2); ctx.fill();
  }
  const stage = () => { const p = phase(); return p < 0.25 ? 0 : p < 0.3 ? 1 : p < 0.7 ? 2 : 3; };

  // ------------------------------------------------------------------ a neuron: the weighted sum
  const TORCH = { tanh: 'torch.tanh', sigmoid: 'torch.sigmoid', relu: 'torch.relu', leaky: 'F.leaky_relu', elu: 'F.elu', gelu: 'F.gelu', swish: 'F.silu', silu: 'F.silu', softplus: 'F.softplus', sine: 'torch.sin', gaussian: 'torch.exp' };
  const torchCall = (id, z) => (id === 'leaky' ? `F.leaky_relu(${z}, 0.05)` : id === 'gelu' ? `F.gelu(${z}, approximate='tanh')` : id === 'gaussian' ? `torch.exp(-${z}**2)` : `${TORCH[id]}(${z})`);
  /* s: { title, sub, terms: [{ a, w }], bias, scale: { v, label }, act: { id, label, f }, zName, outName,
          inLabel(i), wLabel(i), inputNode(i) (optional: what an input bar opens), vars: { w, a, b }, codeHead,
          about: [model-specific paragraphs] } */
  D.neuron = function (X, s) {
    const n = s.terms.length, prods = s.terms.map((t) => t.a * t.w), dot = prods.reduce((p, v) => p + v, 0);
    const scaled = s.scale ? dot * s.scale.v : dot, b = s.bias, z = scaled + (b == null ? 0 : b);
    const act = s.act && s.act.id !== 'linear' ? s.act : null, out = act ? act.f(z) : z;
    let kMax = 0; for (let i = 1; i < n; i++) if (Math.abs(prods[i]) > Math.abs(prods[kMax])) kMax = i;
    let before = b == null || s.scale ? 0 : 0; for (let i = 0; i < kMax; i++) before += prods[i];
    const zn = s.zName || 'z', on = s.outName || 'y', v = Object.assign({ w: 'w', a: 'a', b: 'b' }, s.vars);
    const inL = s.inLabel || ((i) => `input ${i}`), wL = s.wLabel || ((i) => `weight ${i}`);
    const head = s.codeHead || [], L0 = head.length;
    const lines = [...head, `${zn} = ${v.w} @ ${v.a}${s.scale ? ` * ${num(s.scale.v, 4)}` : ''}${b != null ? ` + ${v.b}` : ''}`];
    if (act) lines.push(`${on} = ${torchCall(act.id, zn)}`);
    const zones = [{ line: L0, find: '@', id: 'dot', make: () => D.lib(X, 'dot', { terms: s.terms }) }];
    if (s.scale) zones.push({ line: L0, find: '*', id: 'scale', make: () => D.fmul(X, dot, s.scale.v) });
    if (b != null) zones.push({ line: L0, find: '+', id: 'bias', make: () => D.fadd(X, scaled, b) });
    if (act) zones.push({ line: L0 + 1, find: act.id === 'gaussian' ? 'torch.exp' : TORCH[act.id], id: 'act', make: () => D.lib(X, act.id, { z }) });
    const code = lines.join('\n');
    const rows = W2(X, { t: [0, 0, 1, 0.15], a: [0, 0.22, 1, 0.1], w: [0, 0.38, 1, 0.1], p: [0, 0.54, 1, 0.15] }, { t: [0, 0, 1, 0.14], a: [0, 0.2, 1, 0.08], w: [0, 0.34, 1, 0.08], p: [0, 0.48, 1, 0.12] });
    const box = W2(X, { code: [0, 0.77, 0.235, 0.22], mul: [0.255, 0.77, 0.235, 0.22], acc: [0.51, 0.77, 0.235, 0.22], act: [0.765, 0.77, 0.235, 0.22] },
      { code: [0, 0.66, 0.48, 0.15], mul: [0.52, 0.66, 0.48, 0.15], acc: [0, 0.84, 0.48, 0.15], act: [0.52, 0.84, 0.48, 0.15] });
    const bars = (vals) => Float32Array.from(vals);
    return node({
      title: s.title, sub: s.sub || `${n} inputs`, tone: s.tone || 's1',
      summary: (ctx, r, S) => S.bars(ctx, bars(prods), r),
      inner: (ctx, r, S) => {
        const u = S.u(r, 8, 14);
        S.lines(ctx, [
          `${zn} = Σᵢ wᵢ·aᵢ${s.scale ? ' × ' + s.scale.label : ''}${b != null ? ' + b' : ''}      (${n} products, ${n} additions)`,
          `  = ${fmt(dot, 4)}${s.scale ? ` × ${fmt(s.scale.v, 4)}` : ''}${b != null ? ` + ${fmt(b, 4)}` : ''} = ${fmt(z, 4)}`,
          s.actLine || (act ? `${on} = ${act.label}(${zn}) = ${fmt(out, 4)}` : `${on} = ${zn} (no activation)`),
        ], S.place(r, rows.t), { mono: true, size: u, strong: (i) => i === 2 });
        const ra = S.place(r, rows.a), rw = S.place(r, rows.w), rp = S.place(r, rows.p);
        const names = s.rowNames || [`inputs aᵢ${s.inWhat ? ': ' + s.inWhat : ''}`, 'weights wᵢ (learned)', 'products wᵢ·aᵢ'];
        S.caption(ctx, names[0], ra, { size: u * 0.8, minW: 200 }); S.bars(ctx, bars(s.terms.map((t) => t.a)), ra);
        S.caption(ctx, names[1], rw, { size: u * 0.8, minW: 200 }); S.bars(ctx, bars(s.terms.map((t) => t.w)), rw);
        S.caption(ctx, `${names[2]}: they add up to ${fmt(dot, 3)}. Zoom into any bar`, rp, { size: u * 0.8, minW: 240 }); S.bars(ctx, bars(prods), rp);
      },
      grids: [
        { id: 'in', rect: rows.a, rows: 1, cols: n, make: (i, j) => (s.inputNode ? s.inputNode(j) : D.stored(X, s.terms[j].a, inL(j))) },
        { id: 'w', rect: rows.w, rows: 1, cols: n, make: (i, j) => D.stored(X, s.terms[j].w, wL(j)) },
        { id: 'p', rect: rows.p, rows: 1, cols: n, make: (i, j) => Object.assign(D.fmul(X, s.terms[j].w, s.terms[j].a), { title: `w·a for ${inL(j)}` }) },
      ],
      about: [
        ...(s.about || []),
        s.kind === 'dot'
          ? `A dot product: ${n} pairs of numbers multiplied together and the products added up${s.scale ? `, then scaled by ${s.scale.label}` : ''}: ${zn} = ${fmt(z, 4)}. It has the same shape as a neuron's weighted sum, but neither row is a learned weight: both were computed from this prompt a moment ago.`
          : `A neuron does two things. First a weighted sum: each of its ${n} inputs aᵢ is multiplied by its own learned weight wᵢ, the products are added up${b != null ? ', and the bias b is added' : ''}: ${zn} = ${fmt(z, 4)}. ${act ? `Then a function bends the result: ${on} = ${act.label}(${zn}) = ${fmt(out, 4)}.` : 'Here the sum is the output; there is no activation.'}`,
        `So this is ${n} multiplications and ${n} additions. The rows show the two sets of numbers and their products; zoom into any bar to open that number. The boxes below lead to the PyTorch code, one multiplication, one step of the sum${act ? ' and the activation' : ''}, and from each of them down to the circuits and electrons that compute it.`,
        s.kind === 'dot'
          ? 'Why the scale: a dot product of n random numbers grows like √n, so large vectors would give huge scores, and a softmax over huge scores puts almost all the weight on one position, where its slope is nearly 0 and learning stalls. Dividing by √n keeps the scores in a range where the softmax stays soft.'
          : `How it learns: when the model trains, each weight moves by the learning rate times its input times δ, how much the loss changes when this sum changes (δ arrives from the layers above). So the weights on inputs that were large for this example move the most, and a weight on an input of 0 does not move at all. ${act && act.id === 'relu' ? 'With ReLU, a negative sum has slope 0, so δ is 0 and nothing in this neuron learns from this example. ' : ''}A model does this for every weight and every example of a batch, then averages.`,
      ],
      code: s.code || code,
    }, () => [
      Object.assign(D.code(X, { title: 'PyTorch code', lines, zones, about: [`This neuron in PyTorch${s.codeWhat ? ', ' + s.codeWhat : ''}. Zoom into a dotted outline: the @ (a dot product), the + or the function call opens what the library does inside.`] }), { id: 'code', rect: box.code }),
      Object.assign(D.fmul(X, s.terms[kMax].w, s.terms[kMax].a), { id: 'mul', rect: box.mul, title: 'One multiplication', sub: `the largest product, ${inL(kMax)}` }),
      Object.assign(D.fadd(X, before, prods[kMax]), { id: 'acc', rect: box.acc, title: 'One addition', sub: 'adding it to the running sum' }),
      s.actNode ? Object.assign(s.actNode(), { id: 'act', rect: box.act }) : act ? Object.assign(D.lib(X, act.id, { z }), { id: 'act', rect: box.act }) : null,
    ]);
  };

  // ------------------------------------------------------------------ code, with zoomable calls and operators
  D.code = function (X, s) {
    const B = block(s.lines, [0.01, 0.06, 0.98, 0.88], 0.12);
    return node({
      title: s.title || 'Code', sub: s.sub || 'PyTorch', tone: 's3',
      summary: (ctx, r, S) => S.text(ctx, s.lines[s.lines.length - 1].trim(), r.x, r.y + r.h * 0.35, { size: clamp(r.w / 22, 6, 11), mono: true, color: theme.c('ink-2'), maxW: r.w }),
      inner: (ctx, r, S) => { ctx.fillStyle = theme.c('panel-2'); ctx.beginPath(); NA.rr(ctx, r.x, r.y, r.w, r.h * 0.96, 8); ctx.fill(); drawCode(ctx, S, r, B); },
      about: s.about || ['The code for this part. Zoom into a dotted outline to see what the library does inside.'],
      code: s.lines.join('\n'),
    }, () => (s.zones || []).map((zn) => {
      const line = s.lines[zn.line] || '';
      let col = -1, from = 0;
      for (let k = 0; k <= (zn.nth || 0); k++) { col = line.indexOf(zn.find, from); if (col < 0) break; from = col + 1; }
      if (col < 0) return null;
      return Object.assign(zn.make(), { id: zn.id, shape: 'zone', rect: B.span(zn.line, col, zn.find.length) });
    }));
  };

  // ------------------------------------------------------------------ inside a library call
  const LN2 = Math.LN2, PI = Math.PI;
  const tiers = (py, aten, hw) => [['Python', py], ['C++ core (ATen)', aten], ['hardware', hw]];
  const HW = 'SIMD: 8 floats per AVX2 instruction';
  // e^x the way math libraries do it: x = k·ln2 + r, e^x = 2^k · e^r, e^r from a short polynomial
  function expParts(x) {
    const k = Math.round(x / LN2), r = x - k * LN2, c = [1, 1, 1 / 2, 1 / 6, 1 / 24, 1 / 120, 1 / 720];
    let h = c[6]; const hs = [];
    for (let i = 5; i >= 0; i--) { const p = h * r; hs.push({ h, p, c: c[i], next: p + c[i] }); h = p + c[i]; }
    return { k, r, p: h, hs, y: Math.exp(x) };
  }
  const LIBS = {
    dot: (a) => {
      const t = a.terms, n = t.length, steps = [];
      let acc = 0;
      for (let i = 0; i < Math.min(n, 3); i++) {
        const p = t[i].w * t[i].a;
        steps.push({ t: `p = w[${i}] × a[${i}] = ${fmt(t[i].w, 3)} × ${fmt(t[i].a, 3)} = ${fmt(p, 4)}`, op: { k: 'mul', a: t[i].w, b: t[i].a } });
        steps.push({ t: `acc = acc + p = ${fmt(acc, 4)} + ${fmt(p, 4)} = ${fmt(acc + p, 4)}`, op: { k: 'add', a: acc, b: p } });
        acc += p;
      }
      for (let i = 3; i < n; i++) acc += t[i].w * t[i].a;
      if (n > 3) steps.push({ t: `… ${n - 3} more multiply-and-add steps …`, op: null });
      steps.push({ t: `w @ a = ${fmt(acc, 4)}`, op: null });
      return { title: 'Inside @', sub: `a dot product of ${n}`, tiers: tiers('w @ a', 'aten::matmul → dot / BLAS', 'fused multiply-add, 8 lanes'), steps,
        about: [`The @ operator multiplies matching numbers and adds the products up. PyTorch's C++ core (ATen) hands the work to an optimised linear-algebra library (BLAS: MKL or OpenBLAS on a CPU, cuBLAS on a GPU) whose inner loop is exactly this: multiply, add to a running total, repeat ${n} times. Each row opens one multiplication or one addition.`, 'Real CPUs do each step as a fused multiply-add (one instruction computes a × b + c with a single rounding), on 8 or 16 numbers at once; whole layers are matrix products, so that loop runs millions of times per forward pass.'],
        code: `import torch\n\ndef dot(w, a):\n    acc = 0.0\n    for wi, ai in zip(w, a):      # ${n} steps\n        acc = acc + wi * ai       # one multiply, one add (a CPU fuses them: fma)\n    return acc\n\nw = ${pyList(t.map((q) => q.w))}\na = ${pyList(t.map((q) => q.a))}\nprint(dot(w, a), torch.dot(torch.tensor(w), torch.tensor(a)).item())   # the same, up to rounding` };
    },
    tanh: ({ z }) => {
      const z2 = 2 * z, e = Math.exp(z2), q = 2 / (e + 1);
      return { title: 'Inside torch.tanh', sub: `tanh(${fmt(z, 3)})`, tiers: tiers('torch.tanh(z)', 'aten::tanh → vectorized kernel', HW),
        steps: [
          { t: `2z = 2 × ${fmt(z, 4)} = ${fmt(z2, 4)}`, op: { k: 'mul', a: 2, b: z } },
          { t: `e = exp(2z) = ${fmt(e, 4)}`, op: { k: 'exp', x: z2 } },
          { t: `e + 1 = ${fmt(e + 1, 4)}`, op: { k: 'add', a: e, b: 1 } },
          { t: `q = 2 / (e + 1) = ${fmt(q, 4)}`, op: { k: 'div', a: 2, b: e + 1 } },
          { t: `tanh(z) = 1 − q = ${fmt(1 - q, 4)}`, op: { k: 'add', a: 1, b: -q } },
        ],
        about: ['torch.tanh dispatches to a vectorized CPU kernel (on x86 builds of PyTorch it comes from the SLEEF math library), which evaluates a carefully fitted approximation on 8 or 16 numbers at a time. The rows show the same function written with the exponential, tanh(z) = 1 − 2/(e^{2z} + 1): an exponential, an addition, a division and a subtraction, each of which opens further.', 'There is no "tanh circuit": the chip only adds, multiplies, compares and moves bits. Everything else is built from those.'],
        code: `import math, torch\n\ndef tanh(z):\n    e = math.exp(2 * z)\n    return 1 - 2 / (e + 1)\n\nz = ${num(z)}\nprint(tanh(z), torch.tanh(torch.tensor(z)).item())   # the same, up to float32 rounding` };
    },
    sigmoid: ({ z }) => {
      const e = Math.exp(-z);
      return { title: 'Inside torch.sigmoid', sub: `σ(${fmt(z, 3)})`, tiers: tiers('torch.sigmoid(z)', 'aten::sigmoid → vectorized kernel', HW),
        steps: [{ t: `e = exp(−z) = exp(${fmt(-z, 4)}) = ${fmt(e, 4)}`, op: { k: 'exp', x: -z } }, { t: `1 + e = ${fmt(1 + e, 4)}`, op: { k: 'add', a: 1, b: e } }, { t: `σ(z) = 1 / (1 + e) = ${fmt(1 / (1 + e), 4)}`, op: { k: 'div', a: 1, b: 1 + e } }],
        about: ['The sigmoid squashes any number into (0, 1): σ(z) = 1 / (1 + e^{−z}). The kernel computes an exponential, adds 1 and takes the reciprocal, on many numbers at once.'],
        code: `import math\nz = ${num(z)}\nprint(1 / (1 + math.exp(-z)))` };
    },
    relu: ({ z }) => {
      const B = bitsOf(z);
      return { title: 'Inside torch.relu', sub: 'max(0, z): just the sign bit', tiers: tiers('torch.relu(z)', 'aten::relu → clamp_min(0)', 'one compare/select per number'),
        steps: [{ t: `z = ${fmt(z, 4)}: sign bit = ${B.s}`, op: { k: 'relu', z } }, { t: `relu(z) = ${B.s ? '0 (the sign bit is 1)' : 'z unchanged (the sign bit is 0)'} = ${fmt(Math.max(0, z), 4)}`, op: null }],
        about: ['ReLU needs no arithmetic at all. max(0, z) only asks whether z is negative, and the answer is the float\'s first bit, the sign bit. If it is 0 the number passes through unchanged; if it is 1 the output is 0. That is one reason ReLU is so cheap.'],
        code: `import torch\nz = torch.tensor(${num(z)})\nprint(torch.relu(z), torch.clamp_min(z, 0))   # the same thing` };
    },
    leaky: ({ z }) => ({ title: 'Inside F.leaky_relu', sub: 'max(0.05z, z)', tiers: tiers('F.leaky_relu(z, 0.05)', 'aten::leaky_relu', 'compare, then one multiply'),
      steps: z > 0 ? [{ t: `z = ${fmt(z, 4)} > 0: passes through`, op: { k: 'relu', z } }] : [{ t: `z = ${fmt(z, 4)} < 0: multiply by 0.05`, op: { k: 'relu', z } }, { t: `0.05 × z = ${fmt(0.05 * z, 4)}`, op: { k: 'mul', a: 0.05, b: z } }],
      about: ['Leaky ReLU lets a small slope through for negative inputs, so neurons never go completely silent: z if z > 0, otherwise 0.05·z. The sign bit decides which.'], code: `import torch.nn.functional as F, torch\nprint(F.leaky_relu(torch.tensor(${num(z)}), 0.05))` }),
    elu: ({ z }) => ({ title: 'Inside F.elu', sub: 'z, or eᶻ − 1 below 0', tiers: tiers('F.elu(z)', 'aten::elu', HW),
      steps: z > 0 ? [{ t: `z = ${fmt(z, 4)} > 0: passes through`, op: { k: 'relu', z } }] : [{ t: `e = exp(z) = ${fmt(Math.exp(z), 4)}`, op: { k: 'exp', x: z } }, { t: `e − 1 = ${fmt(Math.expm1(z), 4)}`, op: { k: 'add', a: Math.exp(z), b: -1 } }],
      about: ['ELU is z for positive inputs and e^z − 1 for negative ones: smooth at 0 and never below −1.'], code: `import torch.nn.functional as F, torch\nprint(F.elu(torch.tensor(${num(z)})))` }),
    gelu: ({ z }) => {
      const z3 = z * z * z, inner = 0.7978845608 * (z + 0.044715 * z3), t = Math.tanh(inner);
      return { title: 'Inside F.gelu', sub: "approximate='tanh'", tiers: tiers("F.gelu(z, approximate='tanh')", 'aten::gelu → vectorized kernel', HW),
        steps: [
          { t: `z³ = ${fmt(z, 3)} × ${fmt(z, 3)} × ${fmt(z, 3)} = ${fmt(z3, 4)}`, op: { k: 'mul', a: z * z, b: z } },
          { t: `u = 0.7979 × (z + 0.044715·z³) = ${fmt(inner, 4)}`, op: { k: 'mul', a: 0.7978845608, b: z + 0.044715 * z3 } },
          { t: `t = tanh(u) = ${fmt(t, 4)}`, op: { k: 'tanh', z: inner } },
          { t: `1 + t = ${fmt(1 + t, 4)}`, op: { k: 'add', a: 1, b: t } },
          { t: `gelu(z) = 0.5 × z × (1 + t) = ${fmt(0.5 * z * (1 + t), 4)}`, op: { k: 'mul', a: 0.5 * z, b: 1 + t } },
        ],
        about: ['GELU(z) = z·Φ(z), where Φ is the normal cumulative distribution: a smooth ReLU that gently lets small negative values through. With approximate=\'tanh\' (as GPT-2 uses) it is computed as 0.5·z·(1 + tanh(√(2/π)·(z + 0.044715·z³))): a few multiplications, a tanh and an addition.'],
        code: `import math, torch, torch.nn.functional as F\nz = ${num(z)}\nu = math.sqrt(2 / math.pi) * (z + 0.044715 * z ** 3)\nprint(0.5 * z * (1 + math.tanh(u)), F.gelu(torch.tensor(z), approximate='tanh').item())` };
    },
    swish: ({ z }) => {
      const sg = 1 / (1 + Math.exp(-z));
      return { title: 'Inside F.silu', sub: 'z · σ(z)', tiers: tiers('F.silu(z)', 'aten::silu → vectorized kernel', HW),
        steps: [{ t: `s = σ(z) = ${fmt(sg, 4)}`, op: { k: 'sigmoid', z } }, { t: `silu(z) = z × s = ${fmt(z, 4)} × ${fmt(sg, 4)} = ${fmt(z * sg, 4)}`, op: { k: 'mul', a: z, b: sg } }],
        about: ['SiLU (also called swish) is z·σ(z): the input times its own sigmoid. Like GELU it is a smooth ReLU; it is the activation in the diffusion model\'s denoiser.'],
        code: `import torch, torch.nn.functional as F\nz = torch.tensor(${num(z)})\nprint(F.silu(z), z * torch.sigmoid(z))` };
    },
    softplus: ({ z }) => ({ title: 'Inside F.softplus', sub: 'ln(1 + eᶻ)', tiers: tiers('F.softplus(z)', 'aten::softplus', HW),
      steps: [{ t: `e = exp(z) = ${fmt(Math.exp(z), 4)}`, op: { k: 'exp', x: z } }, { t: `1 + e = ${fmt(1 + Math.exp(z), 4)}`, op: { k: 'add', a: 1, b: Math.exp(z) } }, { t: `ln(1 + e) = ${fmt(Math.log1p(Math.exp(z)), 4)}`, op: { k: 'log', x: 1 + Math.exp(z) } }],
      about: ['Softplus, ln(1 + e^z), is a smooth ReLU whose slope is the sigmoid.'], code: `import torch.nn.functional as F, torch\nprint(F.softplus(torch.tensor(${num(z)})))` }),
    sine: ({ z }) => {
      const k = Math.round(z / PI), r = z - k * PI, r2 = r * r;
      return { title: 'Inside torch.sin', sub: `sin(${fmt(z, 3)})`, tiers: tiers('torch.sin(z)', 'aten::sin → vectorized kernel', HW),
        steps: [
          { t: `k = round(z / π) = round(${fmt(z / PI, 4)}) = ${k}`, op: { k: 'mul', a: z, b: 1 / PI } },
          { t: `r = z − k·π = ${fmt(r, 4)}      (|r| ≤ π/2)`, op: { k: 'add', a: z, b: -k * PI } },
          { t: `r² = ${fmt(r2, 4)}`, op: { k: 'mul', a: r, b: r } },
          { t: `sin r ≈ r(1 − r²/6(1 − r²/20(1 − r²/42(1 − r²/72)))) = ${fmt(Math.sin(r), 4)}`, op: { k: 'mul', a: r2, b: 1 / 72 } },
          { t: `sin z = ${k % 2 ? '−' : ''}sin r = ${fmt(Math.sin(z), 4)}`, op: null },
        ],
        about: ['Sines are computed by range reduction: subtract the nearest multiple of π, so the remainder r is small, then evaluate a short polynomial in r. Libraries use tuned coefficients and extra precision for the reduction, but the steps are multiplications and additions.'],
        code: `import math\nz = ${num(z)}\nk = round(z / math.pi); r = z - k * math.pi\ns = r * (1 - r*r/6 * (1 - r*r/20 * (1 - r*r/42 * (1 - r*r/72))))\nprint((-1) ** k * s, math.sin(z))` };
    },
    gaussian: ({ z }) => ({ title: 'Inside exp(−z²)', sub: 'a Gaussian bump', tiers: tiers('torch.exp(-z**2)', 'aten::pow, aten::neg, aten::exp', HW),
      steps: [{ t: `z² = ${fmt(z * z, 4)}`, op: { k: 'mul', a: z, b: z } }, { t: `exp(−z²) = ${fmt(Math.exp(-z * z), 4)}`, op: { k: 'exp', x: -z * z } }],
      about: ['A Gaussian activation: a square and an exponential.'], code: `import math\nprint(math.exp(-(${num(z)}) ** 2))` }),
    exp: ({ x }) => {
      const E = expParts(x);
      return { title: 'Inside exp', sub: `e^${fmt(x, 3)}`, tiers: tiers('torch.exp(x)', 'aten::exp → vectorized kernel', 'multiplies, adds and an integer add on the exponent'),
        steps: [
          { t: `k = round(x / ln 2) = round(${fmt(x / LN2, 4)}) = ${E.k}`, op: { k: 'mul', a: x, b: 1 / LN2 } },
          { t: `k·ln 2 = ${E.k} × 0.6931 = ${fmt(E.k * LN2, 4)}`, op: { k: 'mul', a: E.k, b: LN2 } },
          { t: `r = x − k·ln 2 = ${fmt(E.r, 4)}      (|r| ≤ 0.347)`, op: { k: 'add', a: x, b: -E.k * LN2 } },
          { t: `h = 1/720 × r + 1/120 = ${fmt(E.hs[0].next, 5)}`, op: { k: 'mul', a: E.hs[0].h, b: E.r } },
          { t: `… four more steps h = h × r + c …`, op: null },
          { t: `e^r ≈ h × r + 1 = ${fmt(E.p, 5)}`, op: { k: 'add', a: E.hs[5].p, b: 1 } },
          { t: `e^x = 2^${E.k} × e^r: add ${E.k} to the exponent bits = ${fmt(E.y, 4)}`, op: { k: 'scale2', v: E.p, by: E.k } },
        ],
        about: ['There is no exponential circuit either. Libraries split x = k·ln 2 + r with |r| ≤ ½·ln 2, because e^x = 2^k · e^r. The small part e^r is a short polynomial in r, evaluated by Horner\'s rule (multiply, add, multiply, add…); libraries fit its coefficients carefully, and the Taylor version is shown here.', 'Multiplying by 2^k needs no multiplication at all: in a float it is an integer addition of k to the 8 exponent bits. Open the last row to see that adder.'],
        code: `import math\n\ndef exp(x):\n    k = round(x / math.log(2))\n    r = x - k * math.log(2)              # small: |r| <= 0.347\n    h = 1 / 720\n    for c in [1/120, 1/24, 1/6, 1/2, 1, 1]:\n        h = h * r + c                     # Horner's rule: multiply, add\n    return math.ldexp(h, k)               # times 2**k: adds k to the exponent bits\n\nprint(exp(${num(x)}), math.exp(${num(x)}))` };
    },
    log: ({ x }) => {
      const B = bitsOf(x), e = B.e - 127, m = x / 2 ** e, t = m - 1;
      return { title: 'Inside log', sub: `ln ${fmt(x, 3)}`, tiers: tiers('torch.log(x)', 'aten::log → vectorized kernel', HW),
        steps: [
          { t: `x = 2^${e} × ${fmt(m, 4)}  (read straight from the bits)`, op: { k: 'read', v: x, label: 'x' } },
          { t: `t = m − 1 = ${fmt(t, 4)}`, op: { k: 'add', a: m, b: -1 } },
          { t: `ln m ≈ t − t²/2 + t³/3 − … = ${fmt(Math.log(m), 4)}`, op: { k: 'mul', a: t, b: t } },
          { t: `ln x = ${e} × ln 2 + ln m = ${fmt(Math.log(x), 4)}`, op: { k: 'mul', a: e, b: LN2 } },
        ],
        about: ['A logarithm splits the float into its exponent and mantissa, x = 2^e · m: the exponent part is just e·ln 2, and ln m is a polynomial (real libraries reduce m further and use tuned coefficients).'],
        code: `import math\nx = ${num(x)}\nm, e = math.frexp(x)            # x = m * 2**e\nprint(e * math.log(2) + math.log(m), math.log(x))` };
    },
    div: ({ a, b }) => {
      let r = 1 / f32(b); const B = bitsOf(b), r0 = 2 ** (127 - B.e) * 0.75, e1 = 2 - b * r0, r1 = r0 * e1, r2 = r1 * (2 - b * r1);
      r = r2;
      return { title: 'Inside a division', sub: `${fmt(a, 3)} / ${fmt(b, 3)}`, tiers: tiers('a / b', 'aten::div', 'a divider unit, or a reciprocal and Newton steps'),
        steps: [
          { t: `r₀ ≈ 1/b from a small table: ${fmt(r0, 4)}`, op: null },
          { t: `r₁ = r₀ × (2 − b × r₀) = ${fmt(r1, 4)}`, op: { k: 'mul', a: b, b: r0 } },
          { t: `r₂ = r₁ × (2 − b × r₁) = ${fmt(r2, 5)}`, op: { k: 'mul', a: r1, b: 2 - b * r1 } },
          { t: `… each step doubles the correct bits …`, op: null },
          { t: `a / b = a × (1/b) = ${fmt(a, 4)} × ${fmt(r, 5)} = ${fmt(a / b, 4)}`, op: { k: 'mul', a, b: 1 / b } },
        ],
        about: ['Division is the slowest of the basic operations. One common method computes the reciprocal 1/b first with Newton\'s method, which needs only multiplications and additions: from a rough guess r (read from a small table), repeat r ← r·(2 − b·r); each step doubles the number of correct bits. Then a / b = a × r. Hardware dividers also use digit-by-digit (SRT) methods.'],
        code: `a, b = ${num(a)}, ${num(b)}\nr = 0.75 * 2.0 ** -(int.bit_length(int(b)) or 1)   # a rough first guess\nfor _ in range(5):\n    r = r * (2 - b * r)                  # Newton's method for 1/b: multiplies and adds\nprint(a * r, a / b)` };
    },
    sqrt: ({ x }) => {
      const y0 = 2 ** Math.round(Math.log2(x) / 2), y1 = 0.5 * (y0 + x / y0), y2 = 0.5 * (y1 + x / y1);
      return { title: 'Inside a square root', sub: `√${fmt(x, 3)}`, tiers: tiers('torch.sqrt(x)', 'aten::sqrt', 'a square-root unit, or Newton steps'),
        steps: [
          { t: `y₀ = 2^(exponent / 2) = ${fmt(y0, 4)}      (halve the exponent bits)`, op: null },
          { t: `y₁ = ½(y₀ + x / y₀) = ${fmt(y1, 5)}`, op: { k: 'div', a: x, b: y0 } },
          { t: `y₂ = ½(y₁ + x / y₁) = ${fmt(y2, 5)}`, op: { k: 'add', a: y1, b: x / y1 } },
          { t: `√x = ${fmt(Math.sqrt(x), 5)}`, op: null },
        ],
        about: ['A square root can start from halving the exponent (√(2^e) = 2^(e/2)) and refine with Newton\'s method, y ← ½(y + x/y), which doubles the correct digits each step. CPUs also have dedicated square-root instructions.'],
        code: `x = ${num(x)}\ny = 1.0\nfor _ in range(6):\n    y = 0.5 * (y + x / y)\nprint(y, x ** 0.5)` };
    },
    softmax: ({ scores, j, label }) => {
      const mx = Math.max(...scores), es = scores.map((v) => Math.exp(v - mx)), sum = es.reduce((p, v) => p + v, 0);
      return { title: 'Inside softmax', sub: `weight ${j} of ${scores.length}`, tiers: tiers('scores.softmax(-1)', 'aten::_softmax', 'max, exp, sum, divide per row'),
        steps: [
          { t: `m = max of the row = ${fmt(mx, 4)}`, op: null },
          { t: `e = exp(s − m) = exp(${fmt(scores[j] - mx, 4)}) = ${fmt(es[j], 4)}`, op: { k: 'exp', x: scores[j] - mx } },
          { t: `Σ e over the row = ${es.map((v) => fmt(v, 2)).slice(0, 4).join(' + ')}${es.length > 4 ? ' + …' : ''} = ${fmt(sum, 4)}`, op: { k: 'add', a: es[0], b: es[1] || 0 } },
          { t: `${label || 'w'} = e / Σ = ${fmt(es[j] / sum, 4)}`, op: { k: 'div', a: es[j], b: sum } },
        ],
        about: ['Softmax turns a row of scores into weights that are positive and add up to 1: exponentiate each, divide by the total. The kernel first subtracts the row\'s largest score, which changes nothing mathematically (it cancels in the division) but keeps the exponentials from overflowing.'],
        code: `import math\ns = [${scores.map((v) => num(v)).join(', ')}]\nm = max(s)\ne = [math.exp(v - m) for v in s]\nprint(e[${j}] / sum(e))` };
    },
    layernorm: ({ x, j, g, b }) => {
      const n = x.length, mu = x.reduce((p, v) => p + v, 0) / n, va = x.reduce((p, v) => p + (v - mu) ** 2, 0) / n, sd = Math.sqrt(va + 1e-5), xh = (x[j] - mu) / sd;
      return { title: 'Inside LayerNorm', sub: `number ${j} of ${n}`, tiers: tiers('nn.LayerNorm(d)(x)', 'aten::native_layer_norm', 'sums, one square root, a divide'),
        steps: [
          { t: `μ = Σx / ${n} = ${fmt(mu, 4)}`, op: { k: 'add', a: x[0], b: x[1] } },
          { t: `σ² = Σ(x − μ)² / ${n} = ${fmt(va, 4)}`, op: { k: 'mul', a: x[j] - mu, b: x[j] - mu } },
          { t: `√(σ² + ε) = ${fmt(sd, 4)}`, op: { k: 'sqrt', x: va + 1e-5 } },
          { t: `x̂ = (x[${j}] − μ) / σ = (${fmt(x[j], 3)} − ${fmt(mu, 3)}) / ${fmt(sd, 3)} = ${fmt(xh, 4)}`, op: { k: 'div', a: x[j] - mu, b: sd } },
          { t: `y = x̂ × γ + β = ${fmt(xh, 3)} × ${fmt(g, 3)} + ${fmt(b, 3)} = ${fmt(xh * g + b, 4)}`, op: { k: 'mul', a: xh, b: g } },
        ],
        about: ['LayerNorm rescales one token\'s vector: subtract the mean, divide by the standard deviation, then apply a learned gain γ and bias β, one number at a time.'],
        code: `import torch\nx = torch.tensor(${pyList(x)})\ngamma, beta = ${num(g)}, ${num(b)}\nx_hat = (x - x.mean()) / torch.sqrt(x.var(unbiased=False) + 1e-5)\nprint(x_hat[${j}] * gamma + beta)   # ${num(xh * g + b)}` };
    },
  };
  LIBS.silu = LIBS.swish;

  // Every row of a library's breakdown opens one operation.
  D.op = function (X, op) {
    if (op.k === 'mul') return D.fmul(X, op.a, op.b);
    if (op.k === 'add') return D.fadd(X, op.a, op.b);
    if (op.k === 'read') return D.stored(X, op.v, op.label);
    if (op.k === 'relu') return D.signBit(X, op.z);
    if (op.k === 'scale2') { const B = bitsOf(op.v); return D.adder(X, B.e, (op.by + 256) % 256, 8, { title: 'Exponent adder', what: `exponent field ${B.e} + ${op.by} = ${B.e + op.by}`, about: [`Multiplying a float by 2^${op.by} only changes its exponent: this 8-bit adder adds ${op.by} to the exponent bits (${op.by < 0 ? 'a negative k is added as its two\'s complement' : 'the mantissa is untouched'}).`] }); }
    return D.lib(X, op.k, op);
  };
  D.lib = function (X, kind, a) {
    const L = (LIBS[kind] || LIBS.dot)(a), tierR = W2(X, [0, 0.02, 1, 0.2], [0, 0.02, 1, 0.27]), area = W2(X, [0, 0.3, 1, 0.68], [0, 0.34, 1, 0.64]);
    const B = block(L.steps.map((s) => s.t), area, 0.12);
    return node({
      title: L.title, sub: L.sub, tone: 's3',
      summary: (ctx, r, S) => S.text(ctx, L.steps[L.steps.length - 1].t, r.x, r.y + r.h * 0.35, { size: clamp(r.w / 24, 6, 11), mono: true, color: theme.c('ink-2'), maxW: r.w }),
      inner: (ctx, r, S) => {
        const t = S.place(r, tierR), u = S.u(r, 8, 13), n = L.tiers.length, gap = 10;
        L.tiers.forEach(([h, body], i) => {
          const q = X.wide ? { x: t.x + i * (t.w + gap) / n, y: t.y, w: (t.w + gap) / n - gap, h: t.h } : { x: t.x, y: t.y + i * (t.h / n), w: t.w, h: t.h / n - 4 };
          S.box(ctx, q, { fill: theme.c('panel-2'), stroke: theme.c('line') });
          S.text(ctx, h, q.x + 8, q.y + Math.min(q.h * 0.3, u * 1.1), { size: u * 0.8, w: 650, color: theme.c('muted'), maxW: q.w - 12 });
          S.text(ctx, body, q.x + 8, q.y + q.h * (X.wide ? 0.66 : 0.68), { size: u * (X.wide ? 0.88 : 0.82), mono: true, maxW: q.w - 12 });
          if (X.wide && i < n - 1) S.arrow(ctx, q.x + q.w + 1, q.y + q.h / 2, q.x + q.w + gap - 1, q.y + q.h / 2, { lw: 1, head: 4 });
        });
        S.text(ctx, 'with this number, step by step (zoom into a row):', r.x, r.y + (area[1] - 0.03) * r.h, { size: u * 0.82, color: theme.c('muted') });
        drawRows(ctx, S, r, B, (k) => k === L.steps.length - 1);
      },
      about: L.about, code: L.code,
    }, () => L.steps.map((s, k) => (s.op ? Object.assign(D.op(X, s.op), { id: 's' + k, shape: 'zone', rect: B.row(k, s.t.length) }) : null)));
  };

  // ------------------------------------------------------------------ one float32 operation, bit by bit
  function bitRows(ctx, S, r, rows, u) {
    const lh = r.h / rows.length;
    rows.forEach(([lab, B], i) => {
      const y = r.y + i * lh;
      S.text(ctx, lab, r.x, y + lh * 0.2, { size: u * 0.85, mono: true, w: i === rows.length - 1 ? 700 : 500, maxW: r.w });
      drawBits(ctx, S, { x: r.x, y: y + lh * 0.36, w: r.w, h: lh * 0.52 }, B.u);
    });
  }
  const fieldNote = (B) => `sign ${B.s} · exponent ${B.e} (2^${B.e - 127}) · mantissa 1.${bin(B.m, 23).replace(/0+$/, '') || '0'}`;
  D.fmul = function (X, a, b) {
    const M = mulSteps(a, b);
    const area = W2(X, { bits: [0, 0.02, 0.62, 0.5], steps: [0, 0.56, 0.62, 0.42], sign: [0.68, 0.02, 0.32, 0.27], exp: [0.68, 0.36, 0.32, 0.27], mant: [0.68, 0.7, 0.32, 0.28] },
      { bits: [0, 0.02, 1, 0.36], steps: [0, 0.41, 1, 0.3], sign: [0, 0.74, 0.31, 0.25], exp: [0.345, 0.74, 0.31, 0.25], mant: [0.69, 0.74, 0.31, 0.25] });
    const steps = M.special
      ? [`${fmt(M.a, 4)} × ${fmt(M.b, 4)} = ${fmt(M.c, 4)}`, 'a zero (or denormal) operand is a special bit pattern:', 'the multiplier just outputs zero, no arithmetic']
      : [`sign:      ${M.A.s} XOR ${M.B.s} = ${M.s}`, `exponent:  ${M.A.e} + ${M.B.e} − 127${M.top ? ' + 1' : ''} = ${M.e}`, `significand: 1.${bin(M.A.m, 23).slice(0, 6)}… × 1.${bin(M.B.m, 23).slice(0, 6)}… = ${fmt(M.prod / 2 ** 46, 6)}`, `round the 48-bit product to 24 bits${M.rounded ? ' (rounded up)' : ''}`, `= ${fmt(M.c, 6)}${M.ok ? '   ✓ exactly a × b in float32' : ''}`];
    return node({
      title: 'float32 multiply', sub: `${fmt(M.a, 4)} × ${fmt(M.b, 4)}`, tone: 's1',
      summary: (ctx, r, S) => { const h = r.h / 3; [M.A, M.B, M.C].forEach((B, i) => drawBits(ctx, S, { x: r.x, y: r.y + i * h, w: r.w, h: h * 0.8 }, B.u)); },
      inner: (ctx, r, S) => {
        const u = S.u(r, 8, 13);
        bitRows(ctx, S, S.place(r, area.bits), [[`a = ${fmt(M.a, 6)}`, M.A], [`b = ${fmt(M.b, 6)}`, M.B], [`a × b = ${fmt(M.c, 6)}`, M.C]], u);
        S.lines(ctx, steps, S.place(r, area.steps), { mono: true, size: u * 0.92, strong: (i) => i === steps.length - 1 });
      },
      about: [
        'Inside the chip a float32 is 32 bits: 1 sign bit (violet), 8 exponent bits (amber) and 23 mantissa bits (blue), standing for ±1.mantissa × 2^(exponent − 127). The leading 1 is not stored ("the hidden bit").',
        M.special ? 'One of these numbers is zero (or too small to be normal), a special bit pattern that the multiplier handles without arithmetic.' : `Multiplying two floats is three separate jobs, each opened by a box: XOR the sign bits (${M.A.s} XOR ${M.B.s} = ${M.s}), add the exponents and subtract the bias 127 (${M.A.e} + ${M.B.e} − 127${M.top ? ' + 1' : ''} = ${M.e}), and multiply the two 24-bit significands, 1.mantissa with the hidden bit put back. The 48-bit product is rounded to the nearest 24-bit value (ties go to the even one), which gives exactly what a × b is in float32.`,
        'A hardware float multiplier does all of this in a few clock cycles with tens of thousands of transistors; a modern GPU has tens of thousands of these units.',
        'To go faster and fit more, models often use smaller formats: bfloat16 keeps float32\'s 8 exponent bits but only 7 mantissa bits, FP16 has 5 and 10, and models in use may run on 8-bit or even 4-bit numbers. Fewer mantissa bits mean coarser rounding, so the products are usually added up in float32 (mixed precision), the job of the tensor cores in modern GPUs.',
      ],
      code: `import struct\n\ndef f32_bits(x):\n    u = struct.unpack('>I', struct.pack('>f', x))[0]\n    return u >> 31, (u >> 23) & 0xFF, u & 0x7FFFFF       # sign, exponent, mantissa\n\ndef f32(s, e, m):\n    return struct.unpack('>f', struct.pack('>I', (s << 31) | (e << 23) | m))[0]\n\ndef fmul32(a, b):\n    """float32 multiply the hardware way (normal numbers)."""\n    (sa, ea, ma), (sb, eb, mb) = f32_bits(a), f32_bits(b)\n    sign = sa ^ sb                              # an XOR gate\n    if ea == 0 or eb == 0:                      # a zero (or a tiny denormal): the product is zero\n        return f32(sign, 0, 0)\n    prod = (ma | 1 << 23) * (mb | 1 << 23)      # 24 x 24 -> 48-bit integer product\n    top = prod >> 47                            # 1 if the product is 2 or more\n    exp = ea + eb - 127 + top                   # an 8-bit adder, minus the bias\n    shift = 23 + top\n    keep, rest, half = prod >> shift, prod & ((1 << shift) - 1), 1 << (shift - 1)\n    if rest > half or (rest == half and keep & 1):   # round to nearest, ties to even\n        keep += 1\n    if keep >> 24:\n        keep >>= 1\n        exp += 1\n    if exp <= 0:                                # too small for a float32: flush to zero\n        return f32(sign, 0, 0)\n    if exp >= 255:                              # too large: infinity\n        return f32(sign, 255, 0)\n    return f32(sign, exp, keep & 0x7FFFFF)\n\na, b = f32(*f32_bits(${num(M.a, 7)})), f32(*f32_bits(${num(M.b, 7)}))\nprint(fmul32(a, b), f32(*f32_bits(a * b)))    # the same float32`,
    }, () => (M.special ? [] : [
      Object.assign(D.gate(X, 'XOR', [M.A.s, M.B.s]), { id: 'sign', rect: area.sign, title: 'Sign: XOR', sub: `${M.A.s} XOR ${M.B.s} = ${M.s}` }),
      Object.assign(D.adder(X, M.A.e, M.B.e, 8, { title: 'Exponent adder', what: `${M.A.e} + ${M.B.e} = ${M.A.e + M.B.e}`, about: [`The exponents are added by an 8-bit adder: ${M.A.e} + ${M.B.e} = ${M.A.e + M.B.e} (a 9th bit, the carry, catches overflow). A second adder then subtracts the bias 127, since each stored exponent already includes it once.`] }), { id: 'exp', rect: area.exp }),
      Object.assign(D.products(X, M.ma, M.mb), { id: 'mant', rect: area.mant }),
    ]));
  };
  D.fadd = function (X, a, b) {
    const A = addSteps(a, b), sub = !!A.sub;
    const area = W2(X, { bits: [0, 0.02, 0.62, 0.5], steps: [0, 0.56, 0.62, 0.42], exp: [0.68, 0.06, 0.32, 0.38], mant: [0.68, 0.54, 0.32, 0.42] },
      { bits: [0, 0.02, 1, 0.36], steps: [0, 0.41, 1, 0.32], exp: [0, 0.76, 0.48, 0.23], mant: [0.52, 0.76, 0.48, 0.23] });
    const steps = A.special
      ? [`${fmt(A.a, 4)} + ${fmt(A.b, 4)} = ${fmt(A.c, 4)}`, 'one operand is zero (a special bit pattern):', 'the adder passes the other one through']
      : A.zero ? [`${fmt(A.a, 4)} + ${fmt(A.b, 4)}: the significands cancel exactly`, '= 0']
        : [`exponents ${A.big.e} and ${A.small.e}: shift the smaller by d = ${A.d}`, `align:  1.${bin(A.small.m, 23).slice(0, 5)}… × 2^${A.small.e - 127} → 0.${'0'.repeat(Math.max(0, Math.min(A.d - 1, 6)))}1… × 2^${A.big.e - 127}`, `${sub ? 'subtract' : 'add'} the 27-bit significands (guard bits kept)`, `normalise: exponent ${A.big.e} → ${A.e}, round to 24 bits`, `= ${fmt(A.c, 6)}${A.ok ? '   ✓ exactly a + b in float32' : ''}`];
    return node({
      title: 'float32 add', sub: `${fmt(A.a, 4)} + ${fmt(A.b, 4)}`, tone: 's1',
      summary: (ctx, r, S) => { const h = r.h / 3; [A.A, A.B, A.C].forEach((B, i) => drawBits(ctx, S, { x: r.x, y: r.y + i * h, w: r.w, h: h * 0.8 }, B.u)); },
      inner: (ctx, r, S) => {
        const u = S.u(r, 8, 13);
        bitRows(ctx, S, S.place(r, area.bits), [[`a = ${fmt(A.a, 6)}`, A.A], [`b = ${fmt(A.b, 6)}`, A.B], [`a + b = ${fmt(A.c, 6)}`, A.C]], u);
        S.lines(ctx, steps, S.place(r, area.steps), { mono: true, size: u * 0.92, strong: (i) => i === steps.length - 1 });
      },
      about: [
        'Adding floats is harder than multiplying them, because the two numbers first have to be lined up. The one with the smaller exponent has its significand shifted right by the difference of the exponents' + (A.d !== undefined ? ` (here d = ${A.d})` : '') + ', so both are counted in the same power of two. Then the significands are added, or subtracted if the signs differ, and the sum is shifted back so it starts with a 1 again, adjusting the exponent.',
        'Three extra bits (guard, round and sticky) remember what was shifted out, so the final rounding to 24 bits is still correctly rounded. The result is exactly what a + b is in float32. The boxes open the two integer adders that do the work.',
        'Because every addition rounds, float addition is not associative: (a + b) + c can differ from a + (b + c) in the last bits. A GPU adds the terms of a long sum in parallel, in an order that can change from run to run, which is why two runs of the same model can disagree in the seventh digit. Long sums can use compensated (Kahan) summation, which carries the lost bits along.',
      ],
      code: `import struct\n\ndef f32_bits(x):\n    u = struct.unpack('>I', struct.pack('>f', x))[0]\n    return u >> 31, (u >> 23) & 0xFF, u & 0x7FFFFF\n\ndef f32(s, e, m):\n    return struct.unpack('>f', struct.pack('>I', (s << 31) | (e << 23) | m))[0]\n\ndef fadd32(a, b):\n    """float32 add the way an FPU does it (normal numbers), with guard, round and sticky bits."""\n    if a == 0 or b == 0:                          # adding zero changes nothing\n        return f32(*f32_bits(a + b))\n    (sa, ea, ma), (sb, eb, mb) = f32_bits(a), f32_bits(b)\n    pa, pb = (ma | 1 << 23) << 3, (mb | 1 << 23) << 3\n    if (eb, pb) > (ea, pa):                       # a gets the larger magnitude\n        sa, ea, pa, sb, eb, pb = sb, eb, pb, sa, ea, pa\n    d = min(ea - eb, 27)\n    sticky = 1 if pb & ((1 << d) - 1) else 0\n    pb = (pb >> d) | sticky                       # align the smaller significand\n    s = pa + pb if sa == sb else pa - pb          # a 28-bit adder (subtract: add the complement + 1)\n    if s == 0:\n        return 0.0\n    e = ea\n    while s >> 27:                                # carried out: shift right\n        s = (s >> 1) | (s & 1); e += 1\n    while not s >> 26:                            # cancelled: shift left\n        s <<= 1; e -= 1\n    keep, low = s >> 3, s & 7                     # round to nearest, ties to even\n    if low > 4 or (low == 4 and keep & 1):\n        keep += 1\n        if keep >> 24:\n            keep >>= 1; e += 1\n    return f32(sa, e, keep & 0x7FFFFF)\n\na, b = f32(*f32_bits(${num(A.a, 7)})), f32(*f32_bits(${num(A.b, 7)}))\nprint(fadd32(a, b), f32(*f32_bits(a + b)))     # the same float32`,
    }, () => (A.special || A.zero ? [] : [
      Object.assign(D.adder(X, A.big.e, 255 - A.small.e, 8, { cin: 1, title: 'Exponent subtractor', what: `${A.big.e} − ${A.small.e} = ${A.d}`, about: [`To line the numbers up the chip needs d = ${A.big.e} − ${A.small.e} = ${A.d}. An adder subtracts by adding the complement: invert every bit of ${A.small.e} and add 1 through the carry-in. The carry out of the top bit is dropped.`] }), { id: 'exp', rect: area.exp }),
      Object.assign(D.adder(X, A.pa, sub ? 2 ** 28 - 1 - A.pb : A.pb, 28, { cin: sub ? 1 : 0, title: sub ? 'Significand subtractor' : 'Significand adder', what: `28 bits${sub ? ', subtracting' : ''}`, about: [`The aligned significands, each 24 bits plus 3 guard bits, go through a 28-bit adder${sub ? '; to subtract, the smaller one is inverted and 1 is added through the carry-in' : ''}. The top bits of the sum show whether the result has to be shifted to start with a 1 again.`] }), { id: 'mant', rect: area.mant }),
    ]));
  };

  // the 24 × 24 grid of partial-product bits: each cell is one AND gate
  D.products = function (X, ma, mb) {
    const n = 24, bit = (x, i) => Math.floor(x / 2 ** i) % 2, grid = W2(X, [0, 0.12, 0.62, 0.86], [0.1, 0.1, 0.8, 0.56]), rowsBox = W2(X, [0.68, 0.3, 0.32, 0.4], [0, 0.74, 1, 0.24]);
    return node({
      title: 'Significand multiplier', sub: '24 × 24 bits', tone: 's1',
      summary: (ctx, r) => { const s = Math.min(r.w, r.h), c = s / n; for (let i = 0; i < n; i += 2) for (let j = 0; j < n; j += 2) { ctx.fillStyle = bit(mb, i) & bit(ma, n - 1 - j) ? theme.c('pos', 0.8) : theme.c('line', 0.6); ctx.fillRect(r.x + (r.w - s) / 2 + j * c, r.y + i * c, c * 2 - 0.5, c * 2 - 0.5); } },
      inner: (ctx, r, S) => {
        const u = S.u(r, 8, 13), g = S.place(r, grid), c = Math.min(g.w, g.h) / n, ox = g.x;
        S.text(ctx, 'row i: bit i of b selects a copy of a (or zeros); each cell = (bit of a) AND (bit of b)', r.x, r.y + u * 0.6, { size: u * 0.82, color: theme.c('muted'), maxW: r.w });
        for (let i = 0; i < n; i++) for (let j = 0; j < n; j++) { const on = bit(mb, i) & bit(ma, n - 1 - j); ctx.fillStyle = on ? theme.c('pos', 0.85) : theme.c('line', 0.55); ctx.fillRect(ox + j * c + 0.5, g.y + i * c + 0.5, c - 1, c - 1); }
      },
      grids: [{ id: 'and', rect: grid, rows: n, cols: n, square: true, make: (i, j) => Object.assign(D.gate(X, 'AND', [bit(ma, n - 1 - j), bit(mb, i)]), { title: `AND: bit ${n - 1 - j} of a, bit ${i} of b` }) }],
      about: ['Multiplying two binary numbers is long multiplication with only 0s and 1s. Each bit of b either selects a copy of a (if it is 1) or zeros, shifted into place: 24 rows. Each cell here is one bit of one row, the AND of one bit of a and one bit of b, so this step alone is 576 AND gates. Zoom into a cell to open its gate.', 'The 24 rows are then added up. A simple "array multiplier" does that with rows of adders like the one in the box; fast chips use Booth encoding and Wallace or Dadda trees of adders to cut the delay, built from the same kind of full adders.'],
      code: `a, b = ${ma}, ${mb}            # the two 24-bit significands, hidden 1 included\nrows = [(a << i) if (b >> i) & 1 else 0 for i in range(24)]   # each bit of b selects a shifted copy of a\nprint(sum(rows) == a * b)                                     # adding the rows gives the product`,
    }, () => {
      const r0 = bit(mb, 0) ? ma : 0, r1 = bit(mb, 1) ? ma * 2 : 0;
      return [Object.assign(D.adder(X, r0, r1, 26, { title: 'Adding two rows', what: 'rows 0 and 1, 26 bits', about: ['The first two partial-product rows, added by a 26-bit ripple-carry adder. The array multiplier repeats this for every row, accumulating the 48-bit product.'] }), { id: 'rows', rect: rowsBox })];
    });
  };

  // ------------------------------------------------------------------ adders and gates
  D.adder = function (X, x, y, n, o = {}) {
    const R = ripple(x, y, n, o.cin || 0), val = (arr) => arr.reduce((p, v, i) => p + v * 2 ** i, 0), cells = W2(X, [0, 0.34, 1, 0.3], [0, 0.36, 1, 0.26]);
    return node({
      title: o.title || `${n}-bit adder`, sub: o.what || `${x} + ${y}`, tone: 's4', animate: true,
      summary: (ctx, r, S) => drawBits(ctx, S, r, val(R.s) + (R.cout ? 0 : 0), n, () => 's4'),
      inner: (ctx, r, S) => {
        const u = S.u(r, 8, 13), g = S.place(r, cells), cw = g.w / n;
        S.lines(ctx, [`  ${bin(x, n)}   (${o.xLabel || x})`, `+ ${bin(y, n)}   (${o.yLabel || y})${o.cin ? ' + carry-in 1' : ''}`, `= ${R.cout ? '1' : ' '}${bin(val(R.s), n)}   carry out ${R.cout}`], S.place(r, [0, 0.0, 1, 0.26]), { mono: true, size: u, strong: (i) => i === 2 });
        // the carry ripples from bit 0 (right) to the top bit (left): each sum is final only once it has arrived
        const reach = NA.prefersReducedMotion() ? n : ((S.now / 2800) % 1) * (n + 3) - 1;
        for (let j = 0; j < n; j++) {
          const i = n - 1 - j, x0 = g.x + j * cw, done = i < reach, here = i === Math.floor(reach);
          ctx.fillStyle = theme.c('s4', here ? 0.32 : done ? 0.14 : 0.06); ctx.beginPath(); NA.rr(ctx, x0 + 1, g.y, cw - 2, g.h, Math.min(5, cw * 0.2)); ctx.fill();
          ctx.strokeStyle = theme.c(here ? 'focus' : 's4', here ? 1 : 0.6); ctx.lineWidth = here ? 2 : 1; ctx.beginPath(); NA.rr(ctx, x0 + 1.5, g.y + 0.5, cw - 3, g.h - 1, Math.min(5, cw * 0.2)); ctx.stroke();
          const t = Math.min(cw * 0.55, u);
          S.text(ctx, `${R.a[i]}${R.b[i]}`, x0 + cw / 2, g.y + g.h * 0.22, { size: t, mono: true, align: 'center', color: theme.c('ink-2') });
          S.text(ctx, done || here ? String(R.s[i]) : '·', x0 + cw / 2, g.y + g.h * 0.68, { size: t * 1.2, mono: true, w: 700, align: 'center', color: done && R.s[i] ? theme.c('pos') : theme.c('muted') });
          if (R.c[i + 1] && i + 1 <= reach) { ctx.fillStyle = theme.c('s2'); ctx.beginPath(); ctx.arc(x0 + 1, g.y + g.h * 0.45, Math.max(2, Math.min(4, cw * 0.15)), 0, Math.PI * 2); ctx.fill(); }
        }
        S.text(ctx, 'each box is one full adder (bit 0 on the right); top: the two input bits, bottom: the sum bit, final once the carry has rippled in; an orange dot is a carry passed to the left', r.x, g.y + g.h + u * 1.2, { size: u * 0.8, color: theme.c('muted'), maxW: r.w });
      },
      grids: [{ id: 'fa', rect: cells, rows: 1, cols: n, make: (i, j) => { const b = n - 1 - j; return Object.assign(D.fullAdder(X, R.a[b], R.b[b], R.c[b]), { title: `Full adder, bit ${b}` }); } }],
      about: [...(o.about || []), `A ripple-carry adder adds binary numbers the way you add on paper: column by column from the right, each column's carry feeding the next. Each box is a full adder that adds two bits and the incoming carry. Here ${n} of them chained give ${bin(val(R.s), n).replace(/^0+(?=.)/, '')} in binary. Zoom into a box to open its gates.`, 'The carry has to ripple through every box, so wide adders in fast chips use carry-lookahead or parallel-prefix designs (such as Kogge–Stone) that work out all the carries in a few gate delays, at the cost of more gates.'],
      code: `def ripple_add(x, y, n=${n}, carry=${o.cin || 0}):\n    s = 0\n    for i in range(n):                     # one full adder per bit\n        a, b = (x >> i) & 1, (y >> i) & 1\n        s |= (a ^ b ^ carry) << i           # sum bit\n        carry = (a & b) | ((a ^ b) & carry)  # carry to the next column\n    return s, carry\n\nprint(ripple_add(${x}, ${y}))`,
    });
  };
  D.fullAdder = function (X, a, b, cin) {
    const p = a ^ b, s = p ^ cin, g1 = a & b, g2 = p & cin, cout = g1 | g2;
    const R = { x1: [0.2, 0.1, 0.15, 0.22], x2: [0.52, 0.18, 0.15, 0.22], a2: [0.52, 0.52, 0.15, 0.2], a1: [0.2, 0.7, 0.15, 0.2], o1: [0.78, 0.6, 0.15, 0.22] };
    return node({
      title: 'Full adder', sub: `${a} + ${b} + ${cin} = ${cout}${s} (carry, sum)`, tone: 's4', animate: true,
      summary: (ctx, r, S) => S.text(ctx, `${a}+${b}+${cin} → ${s}, c${cout}`, r.x + r.w / 2, r.y + r.h * 0.4, { size: clamp(r.w / 8, 6, 12), mono: true, align: 'center', maxW: r.w }),
      inner: (ctx, r, S) => {
        const P = (k) => S.place(r, R[k]), pin = (q, side, f) => [side ? q.x + q.w : q.x, q.y + q.h * f];
        const x1 = P('x1'), x2 = P('x2'), a1 = P('a1'), a2 = P('a2'), o1 = P('o1'), u = S.u(r, 8, 14);
        const yA = r.y + r.h * 0.15, yB = r.y + r.h * 0.28, yC = r.y + r.h * 0.44, xIn = r.x + r.w * 0.12;
        [['A', a, yA], ['B', b, yB], ['Cin', cin, yC]].forEach(([t, v, y]) => S.text(ctx, `${t} = ${v}`, r.x, y, { size: u, mono: true, w: 650, color: v ? theme.c('pos') : theme.c('ink-2') }));
        const busA = r.x + r.w * 0.14, busB = r.x + r.w * 0.17;
        wire(ctx, [[xIn, yA], [busA, yA], [busA, pin(x1, 0, 0.3)[1]], pin(x1, 0, 0.3)], a); wire(ctx, [[busA, pin(x1, 0, 0.3)[1]], [busA, pin(a1, 0, 0.3)[1]], pin(a1, 0, 0.3)], a);
        wire(ctx, [[xIn, yB], [busB, yB], [busB, pin(x1, 0, 0.7)[1]], pin(x1, 0, 0.7)], b); wire(ctx, [[busB, pin(x1, 0, 0.7)[1]], [busB, pin(a1, 0, 0.7)[1]], pin(a1, 0, 0.7)], b);
        const busC = r.x + r.w * 0.44, busP = r.x + r.w * 0.4, pOut = pin(x1, 1, 0.5);
        wire(ctx, [[xIn, yC], [busC, yC], [busC, pin(x2, 0, 0.7)[1]], pin(x2, 0, 0.7)], cin); wire(ctx, [[busC, pin(x2, 0, 0.7)[1]], [busC, pin(a2, 0, 0.7)[1]], pin(a2, 0, 0.7)], cin);
        wire(ctx, [pOut, [busP, pOut[1]], [busP, pin(x2, 0, 0.3)[1]], pin(x2, 0, 0.3)], p); wire(ctx, [[busP, pin(x2, 0, 0.3)[1]], [busP, pin(a2, 0, 0.3)[1]], pin(a2, 0, 0.3)], p);
        const g1o = pin(a1, 1, 0.5), g2o = pin(a2, 1, 0.5), busO = r.x + r.w * 0.73;
        wire(ctx, [g2o, [busO, g2o[1]], [busO, pin(o1, 0, 0.3)[1]], pin(o1, 0, 0.3)], g2); wire(ctx, [g1o, [busO - 6, g1o[1]], [busO - 6, pin(o1, 0, 0.7)[1]], pin(o1, 0, 0.7)], g1);
        const sOut = pin(x2, 1, 0.5), cOut = pin(o1, 1, 0.5);
        wire(ctx, [sOut, [r.x + r.w * 0.97, sOut[1]]], s); wire(ctx, [cOut, [r.x + r.w * 0.97, cOut[1]]], cout);
        S.text(ctx, `S = ${s}`, r.x + r.w, sOut[1] - u, { size: u, mono: true, w: 700, align: 'right', color: s ? theme.c('pos') : theme.c('ink-2') });
        S.text(ctx, `Cout = ${cout}`, r.x + r.w, o1.y - u * 0.9, { size: u, mono: true, w: 700, align: 'right', color: cout ? theme.c('pos') : theme.c('ink-2') });
        S.text(ctx, `P=${p}`, pOut[0] + 4, pOut[1] - u * 0.8, { size: u * 0.8, mono: true, color: theme.c('muted') });
        S.text(ctx, 'lit wires carry a 1', r.x, r.y + r.h * 0.97, { size: u * 0.8, color: theme.c('muted') });
      },
      about: [`A full adder adds three bits, A + B + carry-in, and gives a sum bit and a carry-out: here ${a} + ${b} + ${cin} = ${cout * 2 + s} = binary ${cout}${s}. The sum is S = A XOR B XOR Cin (1 when an odd number of inputs are 1); the carry is Cout = A·B + (A XOR B)·Cin (1 when at least two are 1). Five gates do it; zoom into one.`, 'Lit wires carry a 1, dim ones a 0. Every adder in the chip, and so every addition in the model, is a chain of these.'],
      code: `def full_adder(a, b, cin):\n    p = a ^ b                       # XOR\n    s = p ^ cin                     # XOR\n    cout = (a & b) | (p & cin)      # two ANDs and an OR\n    return s, cout\n\nprint(full_adder(${a}, ${b}, ${cin}))   # (${s}, ${cout})`,
    }, () => [
      Object.assign(D.gate(X, 'XOR', [a, b]), { id: 'x1', rect: R.x1, title: 'XOR', sub: `A, B → ${p}` }),
      Object.assign(D.gate(X, 'XOR', [p, cin]), { id: 'x2', rect: R.x2, title: 'XOR', sub: `P, Cin → ${s}` }),
      Object.assign(D.gate(X, 'AND', [p, cin]), { id: 'a2', rect: R.a2, title: 'AND', sub: `P, Cin → ${g2}` }),
      Object.assign(D.gate(X, 'AND', [a, b]), { id: 'a1', rect: R.a1, title: 'AND', sub: `A, B → ${g1}` }),
      Object.assign(D.gate(X, 'OR', [g1, g2]), { id: 'o1', rect: R.o1, title: 'OR', sub: `→ ${cout}` }),
    ]);
  };
  const GATE_DEPTH = 'Gates are the vocabulary of every digital chip: NAND (or NOR) alone can build any logic function, and adders, multipliers, memories and control are all made of them. A modern GPU has tens of billions of transistors, billions of gates, each switching in a few picoseconds and using a tiny amount of energy every time it does, which is where a chip\'s power goes.';
  const GATE_ABOUT = {
    XOR: 'XOR built from four NAND gates: N1 = NAND(A, B), then NAND(A, N1) and NAND(B, N1), then the NAND of those two. NAND alone can build any logic function. Chips often make XOR from fewer transistors with other arrangements, but the logic is the same.',
    AND: 'In CMOS the natural gates invert their result: NAND and NOR. An AND gate is a NAND followed by an inverter (NOT), 6 transistors in all.',
    OR: 'An OR gate is a NOR followed by an inverter (NOT): 6 transistors.',
  };
  D.gate = function (X, type, ins) {
    const out = GATE[type](...ins);
    if (type === 'NAND' || type === 'NOR' || type === 'NOT') return D.cmos(X, type, ins);
    const [a, b] = ins, n1 = GATE.NAND(a, b);
    const parts = type === 'XOR'
      ? [['n1', 'NAND', [a, b], [0.22, 0.34, 0.16, 0.32]], ['n2', 'NAND', [a, n1], [0.5, 0.06, 0.16, 0.32]], ['n3', 'NAND', [b, n1], [0.5, 0.62, 0.16, 0.32]], ['n4', 'NAND', [GATE.NAND(a, n1), GATE.NAND(b, n1)], [0.78, 0.34, 0.16, 0.32]]]
      : type === 'AND' ? [['n1', 'NAND', [a, b], [0.25, 0.3, 0.2, 0.4]], ['n2', 'NOT', [n1], [0.62, 0.33, 0.16, 0.34]]]
        : [['n1', 'NOR', [a, b], [0.25, 0.3, 0.2, 0.4]], ['n2', 'NOT', [GATE.NOR(a, b)], [0.62, 0.33, 0.16, 0.34]]];
    return node({
      title: `${type} gate`, sub: `${ins.join(', ')} → ${out}`, tone: 's7', animate: true,
      summary: (ctx, r) => gateGlyph(ctx, type, r, out),
      inner: (ctx, r, S) => {
        const u = S.u(r, 8, 14), P = (f) => S.place(r, f), pin = (q, side, f) => [side ? q.x + q.w : q.x, q.y + q.h * f];
        const yA = r.y + r.h * 0.25, yB = r.y + r.h * 0.75, xIn = r.x + r.w * 0.06;
        S.text(ctx, `A = ${a}`, r.x, yA - u, { size: u, mono: true, w: 650, color: a ? theme.c('pos') : theme.c('ink-2') });
        S.text(ctx, `B = ${b}`, r.x, yB - u, { size: u, mono: true, w: 650, color: b ? theme.c('pos') : theme.c('ink-2') });
        const q = Object.fromEntries(parts.map(([id, , , f]) => [id, P(f)]));
        if (type === 'XOR') {
          const x1 = r.x + r.w * 0.12, x2 = r.x + r.w * 0.16, n1o = pin(q.n1, 1, 0.5), xm = r.x + r.w * 0.44;
          wire(ctx, [[xIn, yA], [x1, yA], [x1, pin(q.n1, 0, 0.3)[1]], pin(q.n1, 0, 0.3)], a); wire(ctx, [[x1, yA], [x1, pin(q.n2, 0, 0.3)[1]], pin(q.n2, 0, 0.3)], a);
          wire(ctx, [[xIn, yB], [x2, yB], [x2, pin(q.n1, 0, 0.7)[1]], pin(q.n1, 0, 0.7)], b); wire(ctx, [[x2, yB], [x2, pin(q.n3, 0, 0.7)[1]], pin(q.n3, 0, 0.7)], b);
          wire(ctx, [n1o, [xm, n1o[1]], [xm, pin(q.n2, 0, 0.7)[1]], pin(q.n2, 0, 0.7)], n1); wire(ctx, [[xm, n1o[1]], [xm, pin(q.n3, 0, 0.3)[1]], pin(q.n3, 0, 0.3)], n1);
          const o2 = pin(q.n2, 1, 0.5), o3 = pin(q.n3, 1, 0.5), xr = r.x + r.w * 0.72;
          wire(ctx, [o2, [xr, o2[1]], [xr, pin(q.n4, 0, 0.3)[1]], pin(q.n4, 0, 0.3)], GATE.NAND(a, n1)); wire(ctx, [o3, [xr + 6, o3[1]], [xr + 6, pin(q.n4, 0, 0.7)[1]], pin(q.n4, 0, 0.7)], GATE.NAND(b, n1));
          const o4 = pin(q.n4, 1, 0.5); wire(ctx, [o4, [r.x + r.w, o4[1]]], out);
          S.text(ctx, `Y = ${out}`, r.x + r.w, o4[1] - u, { size: u, mono: true, w: 700, align: 'right', color: out ? theme.c('pos') : theme.c('ink-2') });
        } else {
          const x1 = r.x + r.w * 0.15, mid = pin(q.n1, 1, 0.5), o = pin(q.n2, 1, 0.5);
          wire(ctx, [[xIn, yA], [x1, yA], [x1, pin(q.n1, 0, 0.3)[1]], pin(q.n1, 0, 0.3)], a);
          wire(ctx, [[xIn, yB], [x1 + 8, yB], [x1 + 8, pin(q.n1, 0, 0.7)[1]], pin(q.n1, 0, 0.7)], b);
          wire(ctx, [mid, pin(q.n2, 0, 0.5)], 1 - out); wire(ctx, [o, [r.x + r.w, o[1]]], out);
          S.text(ctx, `Y = ${out}`, r.x + r.w, o[1] - u, { size: u, mono: true, w: 700, align: 'right', color: out ? theme.c('pos') : theme.c('ink-2') });
        }
      },
      about: [GATE_ABOUT[type], `With inputs ${ins.join(' and ')} the output is ${out}. Zoom into a gate to see its transistors.`, GATE_DEPTH],
      code: type === 'XOR' ? `nand = lambda a, b: 1 - (a & b)\n\ndef xor(a, b):\n    n1 = nand(a, b)\n    return nand(nand(a, n1), nand(b, n1))\n\nprint(xor(${a}, ${b}))   # ${out}` : `nand = lambda a, b: 1 - (a & b)\nnor = lambda a, b: 1 - (a | b)\nnot_ = lambda a: 1 - a\nprint(${type === 'AND' ? `not_(nand(${a}, ${b}))` : `not_(nor(${a}, ${b}))`})   # ${out}`,
    }, () => parts.map(([id, t, pins, f]) => Object.assign(D.cmos(X, t, pins), { id, rect: f })));
  };

  // ------------------------------------------------------------------ transistors
  /* CMOS: PMOS transistors (top) connect the output to the supply when their gate is 0; NMOS (bottom) connect it
     to ground when their gate is 1. Each layout lists the transistors with their input, the series/parallel
     wiring is drawn between them, and the path that carries the last switch is animated. */
  D.cmos = function (X, type, ins) {
    const prev = previous(type, ins), out = GATE[type](...ins), outBefore = GATE[type](...prev), charge = out > outBefore;
    const T = type === 'NOT'
      ? [['p1', 'p', 0, [0.38, 0.12, 0.2, 0.3]], ['n1', 'n', 0, [0.38, 0.58, 0.2, 0.3]]]
      : type === 'NAND' ? [['p1', 'p', 0, [0.2, 0.1, 0.2, 0.28]], ['p2', 'p', 1, [0.5, 0.1, 0.2, 0.28]], ['n1', 'n', 0, [0.35, 0.48, 0.2, 0.2]], ['n2', 'n', 1, [0.35, 0.74, 0.2, 0.2]]]
        : [['p1', 'p', 0, [0.35, 0.06, 0.2, 0.2]], ['p2', 'p', 1, [0.35, 0.32, 0.2, 0.2]], ['n1', 'n', 0, [0.2, 0.62, 0.2, 0.28]], ['n2', 'n', 1, [0.5, 0.62, 0.2, 0.28]]];
    const onNow = (k, i) => (k === 'p' ? !ins[i] : !!ins[i]), onBefore = (k, i) => (k === 'p' ? !prev[i] : !!prev[i]);
    // the transistors that carry the switching current: the network that turns on, along its conducting path
    const inPath = (id, k, i) => (charge ? k === 'p' : k === 'n') && onNow(k, i);
    return node({
      title: `${type} in CMOS`, sub: `${ins.join(', ')} → ${out}`, tone: 's7', animate: true,
      summary: (ctx, r) => gateGlyph(ctx, type, r, out),
      inner: (ctx, r, S) => {
        const u = S.u(r, 8, 13), st = stage(), p = phase(), cur = st === 0 ? prev : ins, P = (f) => S.place(r, f);
        const vdd = r.y + r.h * 0.02, gnd = r.y + r.h * 0.97, xo = r.x + r.w * 0.86;
        // the supply rails: VDD glows (it is the 1), ground does not
        ctx.lineCap = 'round';
        ctx.strokeStyle = rgba('s2', 0.22); ctx.lineWidth = 9; ctx.beginPath(); ctx.moveTo(r.x + r.w * 0.12, vdd); ctx.lineTo(r.x + r.w * 0.8, vdd); ctx.stroke();
        ctx.strokeStyle = theme.c('s2'); ctx.lineWidth = 3; ctx.beginPath(); ctx.moveTo(r.x + r.w * 0.12, vdd); ctx.lineTo(r.x + r.w * 0.8, vdd); ctx.stroke();
        ctx.strokeStyle = theme.c('ink-2'); ctx.beginPath(); ctx.moveTo(r.x + r.w * 0.12, gnd); ctx.lineTo(r.x + r.w * 0.8, gnd); ctx.stroke();
        ctx.lineCap = 'butt';
        S.text(ctx, 'VDD (1)', r.x + r.w * 0.82, vdd + 2, { size: u * 0.85, w: 650, color: theme.c('s2') });
        S.text(ctx, 'ground (0)', r.x + r.w * 0.82, gnd - 2, { size: u * 0.85, w: 650, color: theme.c('ink-2') });
        const q = Object.fromEntries(T.map(([id, , , f]) => [id, P(f)])), top = (b) => [b.x + b.w * 0.58, b.y], bot = (b) => [b.x + b.w * 0.58, b.y + b.h];
        // the output node: where the pull-up and pull-down networks meet
        let yOut, paths = [];
        const line = (pts) => { ctx.strokeStyle = theme.c('ink-2'); ctx.lineWidth = 1.4; ctx.beginPath(); pts.forEach(([x, y], i) => (i ? ctx.lineTo(x, y) : ctx.moveTo(x, y))); ctx.stroke(); };
        if (type === 'NOT') {
          yOut = (bot(q.p1)[1] + top(q.n1)[1]) / 2;
          line([[top(q.p1)[0], vdd], top(q.p1)]); line([bot(q.p1), top(q.n1)]); line([bot(q.n1), [bot(q.n1)[0], gnd]]); line([[bot(q.p1)[0], yOut], [xo, yOut]]);
          paths = { p: [[xo, yOut], [bot(q.p1)[0], yOut], bot(q.p1), top(q.p1), [top(q.p1)[0], vdd]], n: [[bot(q.n1)[0], gnd], bot(q.n1), top(q.n1), [top(q.n1)[0], yOut], [xo, yOut]] };
        } else if (type === 'NAND') {
          yOut = (bot(q.p1)[1] + top(q.n1)[1]) / 2;
          line([[top(q.p1)[0], vdd], top(q.p1)]); line([[top(q.p2)[0], vdd], top(q.p2)]);
          line([bot(q.p1), [bot(q.p1)[0], yOut], [bot(q.p2)[0], yOut], bot(q.p2)]); line([[top(q.n1)[0], yOut], top(q.n1)]); line([bot(q.n1), top(q.n2)]); line([bot(q.n2), [bot(q.n2)[0], gnd]]); line([[bot(q.p2)[0], yOut], [xo, yOut]]);
          const pp = ins[0] ? 'p2' : 'p1';
          paths = { p: [[xo, yOut], [bot(q[pp])[0], yOut], bot(q[pp]), top(q[pp]), [top(q[pp])[0], vdd]], n: [[bot(q.n2)[0], gnd], bot(q.n2), top(q.n2), bot(q.n1), top(q.n1), [top(q.n1)[0], yOut], [xo, yOut]] };
        } else {
          yOut = (bot(q.p2)[1] + top(q.n1)[1]) / 2;
          line([[top(q.p1)[0], vdd], top(q.p1)]); line([bot(q.p1), top(q.p2)]); line([bot(q.p2), [bot(q.p2)[0], yOut]]);
          line([top(q.n1), [top(q.n1)[0], yOut], [top(q.n2)[0], yOut], top(q.n2)]); line([bot(q.n1), [bot(q.n1)[0], gnd]]); line([bot(q.n2), [bot(q.n2)[0], gnd]]); line([[top(q.n2)[0], yOut], [xo, yOut]]);
          const nn = ins[0] ? 'n1' : 'n2';
          paths = { p: [[xo, yOut], [bot(q.p2)[0], yOut], bot(q.p2), top(q.p2), bot(q.p1), top(q.p1), [top(q.p1)[0], vdd]], n: [[bot(q[nn])[0], gnd], bot(q[nn]), top(q[nn]), [top(q[nn])[0], yOut], [xo, yOut]] };
        }
        // inputs to the gates: a short lead into each gate, so no wire crosses a neighbouring transistor
        T.forEach(([id, , i]) => {
          const b = q[id], y = b.y + b.h / 2, x0 = b.x - r.w * 0.085;
          ctx.strokeStyle = cur[i] ? ON() : OFF(); ctx.lineWidth = cur[i] ? 2.2 : 1.4; ctx.beginPath(); ctx.moveTo(x0, y); ctx.lineTo(b.x + b.w * 0.06, y); ctx.stroke();
          S.text(ctx, `${'AB'[i]}=${cur[i]}`, x0, y - u * 0.75, { size: u * 0.8, mono: true, w: 650, color: cur[i] ? theme.c('pos') : theme.c('muted') });
        });
        // the output voltage: an RC curve from its old value to its new one, the wiring's capacitance charging
        // (or discharging) through the transistors that just turned on
        const vAt = (q) => (q < 0.3 ? outBefore : out + (outBefore - out) * Math.exp(-(q - 0.3) / 0.085)), level = vAt(p);
        const glow = ctx.createRadialGradient(xo, yOut, 0, xo, yOut, 16);
        glow.addColorStop(0, rgba('pos', 0.55 * level)); glow.addColorStop(1, rgba('pos', 0));
        ctx.fillStyle = glow; ctx.beginPath(); ctx.arc(xo, yOut, 16, 0, Math.PI * 2); ctx.fill();
        ctx.fillStyle = theme.c('pos', 0.25 + 0.75 * level); ctx.beginPath(); ctx.arc(xo, yOut, 6, 0, Math.PI * 2); ctx.fill();
        S.text(ctx, `Y = ${level > 0.995 ? '1' : level < 0.005 ? '0' : fmt(level, 2)}`, xo + 10, yOut, { size: u, mono: true, w: 700 });
        // the same voltage over the whole replay, with a marker at the present moment
        const vr = P(type === 'NOR' ? [0.62, 0.3, 0.36, 0.15] : [0.62, 0.55, 0.36, 0.15]);
        ctx.strokeStyle = theme.c('line'); ctx.lineWidth = 1; ctx.strokeRect(vr.x + 0.5, vr.y + 0.5, vr.w - 1, vr.h - 1);
        S.text(ctx, 'V(Y) over the replay', vr.x, vr.y - u * 0.7, { size: u * 0.75, color: theme.c('muted'), w: 600 });
        const VX = (q) => vr.x + q * vr.w, VY = (v) => vr.y + vr.h - 3 - v * (vr.h - 6);
        ctx.strokeStyle = rgba('pos', 0.35); ctx.lineWidth = 1.5; ctx.beginPath();
        for (let i = 0; i <= 60; i++) { const q = i / 60; i ? ctx.lineTo(VX(q), VY(vAt(q))) : ctx.moveTo(VX(q), VY(vAt(q))); }
        ctx.stroke();
        ctx.strokeStyle = theme.c('pos'); ctx.lineWidth = 2; ctx.beginPath();
        for (let i = 0; i <= 60 && i / 60 <= p; i++) { const q = i / 60; i ? ctx.lineTo(VX(q), VY(vAt(q))) : ctx.moveTo(VX(q), VY(vAt(q))); }
        ctx.stroke();
        ctx.fillStyle = theme.c('focus'); ctx.beginPath(); ctx.arc(VX(p), VY(level), 3.5, 0, Math.PI * 2); ctx.fill();
        // electrons along the conducting path while the output charges or discharges (they move opposite to the
        // current), each with a short fading tail; they thin out as the voltage settles
        if (st === 2) {
          const path = charge ? paths.p : paths.n, k = 10, t = (p - 0.3) / 0.4, strength = Math.abs(out - level) / Math.max(1e-6, Math.abs(out - outBefore));
          for (let i = 0; i < k; i++) for (let j = 4; j >= 0; j--) { const [x, y] = along(path, i / k + t * 0.9 - j * 0.012); electron(ctx, x, y, j ? 2.2 : 3, false, (j ? 0.18 * (5 - j) / 5 : 1) * clamp(0.25 + strength, 0, 1)); }
        }
        timeline(ctx, S, { x: r.x + r.w * 0.6, y: r.y + r.h * (type === 'NOR' ? 0.12 : 0.76), w: r.w * 0.38, h: u * 2.2 }, ['before', 'switch', charge ? 'charging' : 'discharging', 'settled']);
      },
      about: [
        `CMOS logic pairs two kinds of transistors. PMOS transistors (top) conduct when their gate is 0 and connect the output to the supply, VDD, which is logic 1; NMOS transistors (bottom) conduct when their gate is 1 and connect the output to ground, logic 0. ${type === 'NAND' ? 'In a NAND gate the two PMOS are in parallel and the two NMOS in series, so the output is 0 only when both inputs are 1.' : type === 'NOR' ? 'In a NOR gate the PMOS are in series and the NMOS in parallel, so the output is 1 only when both inputs are 0.' : 'An inverter is one of each: whichever is on decides the output.'}`,
        `For any input one network is off, so once the output has settled no current flows from VDD to ground. Current flows only while the output switches, charging or discharging the wiring and the gates it drives. The animation replays the switch that produced this output (${prev.join(', ')} → ${ins.join(', ')}): ${charge ? 'the output charges through the PMOS' : 'the output discharges through the NMOS'}. The blue dots are electrons, which move the opposite way to the conventional current. Zoom into a transistor.`,
      ],
      code: `def ${type.toLowerCase()}_cmos(${ins.map((_, i) => 'AB'[i].toLowerCase()).join(', ')}):\n    """Switch-level model: which network connects the output, and to what."""\n${type === 'NOT' ? '    pull_up = not a                 # PMOS on when its gate is 0\n    pull_down = bool(a)             # NMOS on when its gate is 1' : type === 'NAND' ? '    pull_up = (not a) or (not b)    # two PMOS in parallel\n    pull_down = a and b             # two NMOS in series' : '    pull_up = (not a) and (not b)   # two PMOS in series\n    pull_down = a or b              # two NMOS in parallel'}\n    assert pull_up != pull_down       # exactly one network conducts: no path from VDD to ground\n    return 1 if pull_up else 0\n\nprint(${type.toLowerCase()}_cmos(${ins.join(', ')}))   # ${out}`,
    }, () => T.map(([id, k, i, f]) => Object.assign(D.mos(X, { kind: k, on: onNow(k, i), before: onBefore(k, i), path: inPath(id, k, i), input: 'AB'[i], v: ins[i], vBefore: prev[i], charge }), { id, rect: f, title: `${k === 'p' ? 'PMOS' : 'NMOS'} ${id.toUpperCase()}`, sub: `${'AB'[i]} = ${ins[i]}: ${onNow(k, i) ? 'on' : 'off'}`, summary: (ctx, r) => fetGlyph(ctx, k, r, onNow(k, i)) })));
  };

  // One transistor in cross-section, replaying the switch: the channel forms or vanishes, carriers flow or stop.
  D.mos = function (X, t) {
    const n = t.kind === 'n', hole = !n, chan = W2(X, [0.31, 0.47, 0.38, 0.07], [0.31, 0.46, 0.38, 0.07]);
    return node({
      title: n ? 'NMOS transistor' : 'PMOS transistor', sub: t.on ? 'on' : 'off', tone: n ? 's1' : 's2', animate: true,
      summary: (ctx, r) => fetGlyph(ctx, t.kind, r, t.on),
      inner: (ctx, r, S) => {
        const u = S.u(r, 8, 13), st = stage(), lvl = st === 0 ? (t.before ? 1 : 0) : ramp(t.before ? 1 : 0, t.on ? 1 : 0), on = lvl > 0.5, flow = st === 2 && t.path;
        const P = (f) => S.place(r, f), sub = P([0.04, 0.44, 0.92, 0.42]), src = P([0.08, 0.44, 0.24, 0.18]), drn = P([0.68, 0.44, 0.24, 0.18]), ox = P([0.3, 0.385, 0.4, 0.055]), gate = P([0.3, 0.24, 0.4, 0.145]), ch = P(chan);
        const body = n ? 's2' : 's1', dop = n ? 's1' : 's2';
        // the body: doped silicon, fading with depth
        let gr = ctx.createLinearGradient(0, sub.y, 0, sub.y + sub.h);
        gr.addColorStop(0, rgba(body, 0.18)); gr.addColorStop(1, rgba(body, 0.05));
        ctx.fillStyle = gr; ctx.beginPath(); NA.rr(ctx, sub.x, sub.y, sub.w, sub.h, 8); ctx.fill();
        // source and drain: implanted wells, rounded at the bottom
        const well = (b) => {
          ctx.beginPath(); ctx.moveTo(b.x, b.y); ctx.lineTo(b.x + b.w, b.y); ctx.lineTo(b.x + b.w, b.y + b.h * 0.5);
          ctx.quadraticCurveTo(b.x + b.w, b.y + b.h, b.x + b.w / 2, b.y + b.h); ctx.quadraticCurveTo(b.x, b.y + b.h, b.x, b.y + b.h * 0.5); ctx.closePath();
          const g2 = ctx.createLinearGradient(0, b.y, 0, b.y + b.h); g2.addColorStop(0, rgba(dop, 0.45)); g2.addColorStop(1, rgba(dop, 0.16));
          ctx.fillStyle = g2; ctx.fill(); ctx.strokeStyle = rgba(dop, 0.55); ctx.lineWidth = 1; ctx.stroke();
        };
        well(src); well(drn);
        // the inversion layer: a thin conducting sheet that appears as the gate voltage rises
        if (lvl > 0.01) {
          const g3 = ctx.createLinearGradient(0, ch.y, 0, ch.y + ch.h * 1.6);
          g3.addColorStop(0, rgba(dop, 0.6 * lvl)); g3.addColorStop(1, rgba(dop, 0));
          ctx.fillStyle = g3; ctx.fillRect(ch.x, ch.y, ch.w, ch.h * 1.6);
        }
        // the oxide, glassy, and the metal gate; when the gate is on its field reaches through the oxide
        ctx.fillStyle = rgba('line-2', 0.9); ctx.fillRect(ox.x, ox.y, ox.w, ox.h);
        ctx.fillStyle = theme.dark ? 'rgba(255,255,255,0.14)' : 'rgba(255,255,255,0.6)'; ctx.fillRect(ox.x, ox.y, ox.w, 1.5);
        gr = ctx.createLinearGradient(0, gate.y, 0, gate.y + gate.h);
        gr.addColorStop(0, rgba('ink-2', 0.55 + 0.4 * lvl)); gr.addColorStop(1, rgba('ink-2', 0.38 + 0.37 * lvl));
        ctx.fillStyle = gr; ctx.beginPath(); NA.rr(ctx, gate.x, gate.y, gate.w, gate.h, 5); ctx.fill();
        if (lvl > 0.01) {
          ctx.strokeStyle = rgba(dop, 0.55 * lvl); ctx.lineWidth = 1.2; ctx.setLineDash([3, 4]); ctx.lineDashOffset = -(S.now / 60) % 7;
          for (let k = 0; k < 7; k++) { const x = gate.x + gate.w * (0.1 + (0.8 * k) / 6); ctx.beginPath(); ctx.moveTo(x, gate.y + gate.h + 1); ctx.lineTo(x, ch.y + ch.h * 0.3); ctx.stroke(); }
          ctx.setLineDash([]); ctx.lineDashOffset = 0;
        }
        const lab = (s, x, y, o = {}) => S.text(ctx, s, x, y, Object.assign({ size: u * 0.82, align: 'center', color: theme.c('ink-2') }, o));
        lab(n ? 'n+ source' : 'p+ source', src.x + src.w / 2, src.y + src.h * 0.5); lab(n ? 'n+ drain' : 'p+ drain', drn.x + drn.w / 2, drn.y + drn.h * 0.5);
        lab(n ? 'p-type silicon (body)' : 'n-type silicon (body)', sub.x + sub.w / 2, sub.y + sub.h * 0.82, { color: theme.c('muted') });
        lab('gate', gate.x + gate.w / 2, gate.y + gate.h / 2, { color: theme.c('panel'), w: 650 });
        lab('oxide, about 1 nm', ox.x + ox.w + 4, ox.y + ox.h / 2, { align: 'left', size: u * 0.72, color: theme.c('muted') });
        // terminals
        const term = (x, y0, s, v) => { ctx.strokeStyle = theme.c('ink-2'); ctx.lineWidth = 1.4; ctx.beginPath(); ctx.moveTo(x, y0); ctx.lineTo(x, r.y + r.h * 0.1); ctx.stroke(); lab(s, x, r.y + r.h * 0.06, { w: 650, color: theme.c('ink') }); if (v) lab(v, x, r.y + r.h * 0.155, { size: u * 0.72, color: theme.c('muted') }); };
        term(src.x + src.w / 2, src.y, 'S', n ? 'ground' : 'VDD'); term(drn.x + drn.w / 2, drn.y, 'D', 'to the output'); term(gate.x + gate.w / 2, gate.y, 'G', `${t.input} = ${st === 0 ? t.vBefore : t.v} (${on ? 'on' : 'off'})`);
        // carriers: plenty in the source and drain, jiggling with heat; in the channel a sheet of them that
        // appears with the gate voltage and streams toward the drain while current flows
        const T = S.now / 1000, dots = 24;
        for (let i = 0; i < dots; i++) {
          const jx = Math.sin(T * 9 + i * 2.1) * 1.8, jy = Math.cos(T * 7 + i * 1.7) * 1.8;
          const xs = src.x + 4 + ((i * 37) % 100) / 100 * (src.w - 8), ys = src.y + 4 + ((i * 53) % 100) / 100 * src.h * 0.62;
          electron(ctx, xs + jx, ys + jy, 2.1, hole);
          const xd = drn.x + 4 + ((i * 41) % 100) / 100 * (drn.w - 8), yd = drn.y + 4 + ((i * 29) % 100) / 100 * drn.h * 0.62;
          electron(ctx, xd - jx, yd + jy, 2.1, hole);
        }
        const sim = carriers(`mos:${t.kind}:${t.path ? 1 : 0}`, 18);
        stepCarriers(sim, S.now, flow ? 3.2 : 0);
        ctx.save(); ctx.beginPath(); ctx.rect(ch.x, ch.y - 2, ch.w, ch.h + 4); ctx.clip();
        drawCarriers(ctx, sim, ch, 2.1, hole, 18 * lvl);
        ctx.restore();
        if (on) lab(flow ? `${hole ? 'holes' : 'electrons'} flowing through the channel →` : 'channel formed: no voltage across it now, no net flow', ch.x + ch.w / 2, ch.y + ch.h + u * 1.4, { size: u * 0.8, color: flow ? theme.c(dop) : theme.c('muted') });
        else lab(lvl > 0.01 ? 'the channel is forming…' : 'no channel: the transistor is off', ch.x + ch.w / 2, ch.y + ch.h + u * 1.4, { size: u * 0.8, color: theme.c('muted') });
        timeline(ctx, S, { x: r.x + r.w * 0.1, y: r.y + r.h * 0.89, w: r.w * 0.8, h: u * 2.2 }, ['before', 'gate switches', t.path ? (t.charge ? 'output charges' : 'output discharges') : 'its neighbour switches', 'settled']);
      },
      about: n ? [
        'An n-channel MOSFET in cross-section. The source and drain are silicon doped to have spare electrons (n+), set in silicon doped the other way (p-type). Between them, under a gate separated by an insulating oxide only about a nanometre thick, there is normally no conducting path: the transistor is off.',
        `A positive voltage on the gate (logic 1) pulls electrons to the surface just under the oxide and forms a thin conducting channel. With a voltage between source and drain, electrons then drift from the source through the channel to the drain. ${t.path ? 'In the replay this transistor carries the switching current, so electrons stream through it until the output has discharged; then the voltage across it is gone and the flow stops, although the channel stays.' : 'In the replay it is not on the path that switches, so no current flows through it.'}`,
        'In current chips the channel is a few tens of nanometres long, and transistors are 3-D fins or stacked nanosheets rather than this flat textbook shape, but they switch the same way. Zoom into the channel to see the electrons themselves.',
      ] : [
        'A p-channel MOSFET is the mirror image: p+ source and drain in n-type silicon, and it turns on when its gate is at 0. Its channel conducts with holes, places where an electron is missing (the hollow circles); as electrons hop into those gaps, the holes move the other way, from source to drain.',
        `${t.path ? 'In the replay this transistor carries the switching current: holes flow through it while the output charges up to VDD, then the flow stops.' : 'In the replay it is not on the path that switches.'} Zoom into the channel to see the carriers.`,
      ],
      code: `# Square-law model of a MOSFET in saturation: drain current from the gate voltage\nmu_Cox_WL = 2e-4     # mobility x oxide capacitance x width / length (A/V^2), a typical order\nVt = 0.3             # threshold voltage (V)\n\ndef drain_current(Vgs):\n    return 0.5 * mu_Cox_WL * max(0.0, Vgs - Vt) ** 2   # no channel below the threshold\n\nfor Vgs in [0.0, 0.3, 0.5, 0.8]:\n    print(Vgs, 'V on the gate ->', drain_current(Vgs) * 1e6, 'microamps')`,
    }, () => [Object.assign(D.electrons(X, { kind: t.kind, on: t.on, before: t.before, path: t.path }), { id: 'chan', shape: 'zone', rect: chan, title: 'Into the channel' })]);
  };

  // The bottom: silicon atoms in their lattice, and the carriers moving through it.
  D.electrons = function (X, t) {
    const n = t.kind !== 'p', cap = t.mode === 'cap';
    return node({
      title: cap ? 'Electrons on the capacitor' : 'Electrons in the channel', sub: cap ? (t.charged ? 'charged: a stored 1' : 'empty: a stored 0') : n ? 'free electrons in silicon' : 'holes in silicon', tone: 's1', animate: true,
      summary: (ctx, r) => { for (let i = 0; i < 12; i++) electron(ctx, r.x + ((i * 37) % 100) / 100 * r.w, r.y + ((i * 61) % 100) / 100 * r.h, 1.8, !n); },
      inner: (ctx, r, S) => {
        const u = S.u(r, 8, 13), st = stage(), T = S.now / 1000;
        const box = S.place(r, X.wide ? [0, 0.04, 0.64, 0.8] : [0, 0.03, 1, 0.56]), g = Math.max(18, Math.min(box.w, box.h) / 7);
        // the crystal (drawn once per size), framed like a view through a microscope
        ctx.save(); ctx.beginPath(); NA.rr(ctx, box.x, box.y, box.w, box.h, 10); ctx.clip();
        ctx.drawImage(lattice(box.w, box.h, g), box.x, box.y, box.w, box.h);
        if (cap) {
          // two plates: the charged state has electrons pushed onto one plate and pulled off the other
          const k = t.charged ? 30 : 4, plate = (y) => { const gr = ctx.createLinearGradient(0, y - 4, 0, y + 4); gr.addColorStop(0, rgba('ink-2', 0.2)); gr.addColorStop(0.5, rgba('ink-2', 0.65)); gr.addColorStop(1, rgba('ink-2', 0.2)); ctx.fillStyle = gr; ctx.fillRect(box.x, y - 4, box.w, 8); };
          plate(box.y + box.h * 0.3); plate(box.y + box.h * 0.7);
          for (let i = 0; i < k; i++) { const x = box.x + ((i * 37) % 100) / 100 * box.w, y = box.y + box.h * (0.06 + 0.2 * (((i * 53) % 100) / 100)) + Math.sin(T * 6 + i) * 1.5; electron(ctx, x, y, 2.8); }
          if (t.charged) for (let i = 0; i < 30; i++) S.text(ctx, '+', box.x + ((i * 41) % 100) / 100 * box.w, box.y + box.h * (0.76 + 0.18 * (((i * 29) % 100) / 100)), { size: u, w: 700, color: theme.c('s2'), align: 'center' });
          ctx.restore();
          S.text(ctx, t.charged ? 'extra electrons on one plate, missing electrons (+) on the other' : 'no charge separated: a stored 0', box.x, box.y + box.h + u * 1.3, { size: u * 0.85, color: theme.c('ink-2'), maxW: box.w });
        } else {
          // free carriers: fast random thermal motion, collisions, and a steady drift when there is a field
          const before = t.before ? 1 : 0, after = t.on ? 1 : 0, level = st === 0 ? before : ramp(before, after), field = t.on && t.path && st === 2;
          const sim = carriers(`chan:${t.kind}`, 30);
          stepCarriers(sim, S.now, field ? 2.6 : 0);
          // the push of the field: a faint wash from source to drain while current flows
          if (field) { const gr = ctx.createLinearGradient(box.x, 0, box.x + box.w, 0); gr.addColorStop(0, rgba(n ? 's1' : 's2', 0.0)); gr.addColorStop(1, rgba(n ? 's1' : 's2', 0.12)); ctx.fillStyle = gr; ctx.fillRect(box.x, box.y, box.w, box.h); }
          drawCarriers(ctx, sim, box, Math.max(2.4, g * 0.11), !n, 3 + 27 * level);
          ctx.restore();
          S.text(ctx, 'source', box.x + 6, box.y + 10, { size: u * 0.75, w: 600, color: theme.c('muted') });
          S.text(ctx, 'drain', box.x + box.w - 6, box.y + 10, { size: u * 0.75, w: 600, color: theme.c('muted'), align: 'right' });
          if (field) S.arrow(ctx, box.x + box.w * 0.2, box.y + box.h + u * 1.2, box.x + box.w * 0.8, box.y + box.h + u * 1.2, { lw: 1.5, head: 6, color: theme.c(n ? 's1' : 's2') });
          S.text(ctx, field ? 'drift toward the drain: this is the current' : level > 0.5 ? 'channel on, but no voltage across it: only random thermal motion' : 'transistor off: almost no free carriers here', box.x, box.y + box.h + u * (field ? 2.6 : 1.3), { size: u * 0.85, color: field ? theme.c(n ? 's1' : 's2') : theme.c('muted'), maxW: box.w });
        }
        ctx.strokeStyle = theme.c('line-2'); ctx.lineWidth = 1; ctx.beginPath(); NA.rr(ctx, box.x + 0.5, box.y + 0.5, box.w - 1, box.h - 1, 10); ctx.stroke();
        const side = S.place(r, X.wide ? [0.68, 0.04, 0.32, 0.9] : [0, 0.68, 1, 0.3]);
        S.lines(ctx, cap ? ['A DRAM capacitor holds', 'a few femtofarads; a stored 1', 'is tens of thousands of', 'electrons\' worth of charge.', 'It leaks away, so every', 'cell is refreshed every', '64 ms or so.'] : ['grey: silicon atoms', n ? 'blue: free electrons' : 'hollow: holes', '', 'thermal speed ≈ 10⁵ m/s', 'a collision every ≈ 0.1 ps', 'drift: a small push', 'along the field on top', 'channel: tens of nm long', 'crossing it: < 1 ps'], side, { size: u * 0.92, colorOf: (i) => (i < 2 ? theme.c('ink') : theme.c('ink-2')) });
      },
      about: cap ? [
        `A DRAM cell stores its bit as charge on a tiny capacitor, a few femtofarads. ${t.charged ? 'This one holds a 1: electrons have been pushed onto one side and pulled off the other, tens of thousands of electrons\' worth of charge in today\'s chips.' : 'This one holds a 0: no charge is separated.'}`,
        'The charge slowly leaks through the transistor and the silicon, so the memory reads and rewrites every cell about every 64 milliseconds (refresh). Reading also destroys the charge, so every read is followed by a write-back.',
      ] : [
        n ? 'Inside the channel: silicon atoms (grey) bonded in their crystal lattice, and the free electrons (blue) that carry the current. This is as deep as the zoom goes; below it the story is quantum mechanics, electrons as waves spread through the crystal\'s energy bands.' : 'Inside a PMOS channel: silicon atoms (grey) and holes (hollow), the gaps left where electrons are missing. When an electron jumps into a neighbouring gap the hole moves the other way, so holes act like positive charges flowing from source to drain.',
        'Even with no voltage applied, every carrier races around at about 100 km/s (10⁵ m/s) from heat alone, bouncing off lattice vibrations and impurities roughly every tenth of a picosecond. That random motion goes nowhere on average. A voltage between source and drain adds a small push in one direction on top of it, a drift, and the drift of a vast number of carriers is the current.',
        'In a channel a few tens of nanometres long a carrier crosses in well under a picosecond, which is why transistors can switch billions of times a second. Every multiply, add and comparison in the model above comes down to this, repeated across billions of transistors.',
      ],
      code: cap ? `# A DRAM cell: charge on a capacitor, leaking away until it is refreshed\nC = 10e-15            # 10 femtofarads\nV = 1.0               # volts for a stored 1\nq = 1.602e-19\nprint(C * V / q, 'electrons of charge')    # about 62,000\n\nimport math\ntau = 1.0             # leakage time constant in seconds (varies widely)\nfor t in [0.0, 0.032, 0.064]:\n    print(t, 's:', round(math.exp(-t / tau), 3), 'of the charge left')   # refreshed every ~64 ms` : `import random\n\n# Drude model: random thermal motion, a push from the field, and collisions\ntau = 1e-13            # mean time between collisions (s)\nm = 0.26 * 9.109e-31   # effective mass of an electron in silicon (kg)\nq = 1.602e-19\nE = 1e6                # field in a short channel (V/m)\n\nv_drift = q * E * tau / m\nprint(f'drift speed ~ {v_drift:.0f} m/s')    # comparable to the thermal speed: a strong field\n\nx, v, dt = 0.0, 0.0, 1e-15\nfor step in range(1000):                    # one picosecond\n    v += q * E / m * dt                      # the field accelerates it\n    if random.random() < dt / tau:           # a collision randomises the velocity\n        v = random.gauss(0, 1.0e5)\n    x += v * dt\nprint(f'moved {x * 1e9:.1f} nm in 1 ps')`,
    });
  };

  // ------------------------------------------------------------------ stored numbers: bits in memory
  /* What kind of number a stored value is: the caller says (kind), or its label does. An input is data, a weight or
     bias was learned, an activation was computed for this example, and a few are constants, fixed at random,
     running averages or random noise; each is stored the same way, but lives in memory for different reasons. */
  const KIND_TESTS = [
    ['constant', /zero padding|padding outside|before the first step|^1\/\d+$/i],
    ['stat', /running (mean|variance|spread)/i],
    ['fixed', /\(fixed\)|random frequency|^PE\[/],
    ['random', /noise number|^ε/],
    ['activation', /^weight on\b|attention/i],
    ['bias', /\bbias\b|β|\bshift\b/],
    ['weight', /^W[\d₀-₉]*\[|^w[\d₀-₉]|^U\[|^E\[|^P\[|^pos\[|kernel|projection|learned|γ|leaf value|in this leaf|\bweight\b/],
    ['input', /\(an input\)|\binput\b|pixel|raw sample|^x₀, |x coordinate|noisy point|^x = 2t/i],
  ];
  const kindOf = (label) => { const s = String(label || ''); for (const [k, re] of KIND_TESTS) if (re.test(s)) return k; return 'activation'; };
  const CELL = ' Zoom into a bit to see the memory cell that holds it.';
  const KIND_TEXT = {
    input: 'This is an input: a number of the example itself, data rather than something the model learned. Inputs, weights and activations all sit in memory the same way: in the computer’s main memory (DRAM, or a GPU’s high-bandwidth memory), copied into caches and registers when a calculation needs them.' + CELL,
    weight: 'This is a learned weight, one of the model’s parameters: training set it, and it is the same for every example. The weights stay in the computer’s main memory (DRAM, or a GPU’s high-bandwidth memory) for as long as the model is used, and are copied into caches and registers for every calculation that needs them.' + CELL,
    bias: 'This is a learned bias (or shift), one of the model’s parameters: training set it, and it is the same for every example. Like the weights, it stays in main memory (DRAM, or a GPU’s high-bandwidth memory) and is copied into caches and registers when it is used.' + CELL,
    activation: 'This is an activation: a number the model computed for this one example from what came before, so another example gives another value. It lives in memory only as long as it is needed: while training it is kept until the backward pass has used it, and when the model is only used it can go as soon as the next layer has read it.' + CELL,
    constant: 'This is a constant: a fixed number the computation starts from or pads with, the same for every example and never learned. It is stored like any other number.' + CELL,
    fixed: 'This number was set when the model was made (at random, or by a formula) and is never trained: it is the same for every example and every step. It is stored like any other number.' + CELL,
    stat: 'This is a running statistic: an average gathered over the training batches, not learned by the gradient, and used in place of a batch’s own statistics when the model is used. It is stored like any other number.' + CELL,
    random: 'This number was drawn at random for this example (noise), so it is different every time. It is stored like any other number.' + CELL,
  };
  D.storedKind = kindOf;
  D.stored = function (X, v, label, kind) {
    const B = bitsOf(v), g = W2(X, [0, 0.3, 1, 0.16], [0, 0.32, 1, 0.12]);
    return node({
      title: label || 'A stored number', sub: `${fmt(B.v, 5)} as float32`, tone: 's3',
      summary: (ctx, r, S) => drawBits(ctx, S, r, B.u),
      inner: (ctx, r, S) => {
        const u = S.u(r, 8, 14), gr = S.place(r, g);
        S.lines(ctx, [`${label || 'value'} = ${fmt(v, 6)}`, `stored as float32: ${fmt(B.v, 8)}`], S.place(r, [0, 0, 1, 0.2]), { mono: true, size: u, strong: (i) => i === 0 });
        S.text(ctx, 'sign · exponent (8 bits) · mantissa (23 bits): zoom into a bit', gr.x, gr.y - u * 0.9, { size: u * 0.8, color: theme.c('muted') });
        drawBits(ctx, S, gr, B.u);
        S.lines(ctx, [`(−1)^${B.s} × 1.${bin(B.m, 23).replace(/0+$/, '') || '0'} (binary) × 2^(${B.e} − 127)`, `= ${B.s ? '−' : ''}${fmt(1 + B.m / 2 ** 23, 7)} × 2^${B.e - 127}${B.e === 0 ? '   (zero / denormal)' : ''}`], S.place(r, [0, 0.56, 1, 0.2]), { mono: true, size: u * 0.95 });
      },
      grids: [{ id: 'bit', rect: g, rows: 1, cols: 32, make: (i, j) => Object.assign(D.memCell(X, Math.floor(B.u / 2 ** (31 - j)) % 2, j), { title: `Bit ${31 - j} (${j === 0 ? 'sign' : j <= 8 ? 'exponent' : 'mantissa'})` }) }],
      about: [`In memory, ${label ? label + ' is' : 'this number is'} just these 32 bits: 1 sign bit, 8 exponent bits and 23 mantissa bits, meaning (−1)^sign × 1.mantissa × 2^(exponent − 127). Float32 keeps about 7 significant decimal digits, which is why the stored value can differ slightly from the exact one.`, KIND_TEXT[kind || kindOf(label)] || KIND_TEXT.activation],
      code: `import struct\nv = ${num(v, 7)}\nu = struct.unpack('>I', struct.pack('>f', v))[0]\nbits = format(u, '032b')\nprint(bits[0], bits[1:9], bits[9:])        # sign, exponent, mantissa\nprint(struct.unpack('>f', struct.pack('>I', u))[0])   # the float32 value`,
    });
  };
  D.signBit = function (X, z) {
    const B = bitsOf(z);
    return Object.assign(D.gate(X, 'AND', [Math.floor(B.u / 2 ** 22) % 2, 1 - B.s]), {
      title: 'Sign test', sub: `sign bit ${B.s}: ${B.s ? 'blocked' : 'passed'}`,
      about: [`ReLU keeps z only if its sign bit is 0. In hardware each of z's bits goes through an AND gate whose other input is NOT(sign bit): with sign ${B.s} the gates ${B.s ? 'block every bit, giving 0' : 'pass every bit unchanged'}. This is one of those gates, for one mantissa bit.`, 'That is why ReLU is so cheap: no multiplication and no exponential, only the sign bit choosing between the number and 0. A sigmoid, tanh or GELU needs an exponential, computed by a library routine, a lookup table or a polynomial, which costs many operations: one reason ReLU took over deep learning around 2011.'],
    });
  };
  D.memCell = function (X, bit, idx) {
    const fet = [0.18, 0.3, 0.22, 0.3], cap = [0.5, 0.36, 0.2, 0.3];
    return node({
      title: 'DRAM cell', sub: `holding a ${bit}`, tone: 's3', animate: true,
      summary: (ctx, r, S) => S.text(ctx, String(bit), r.x + r.w / 2, r.y + r.h * 0.4, { size: clamp(r.h * 0.6, 8, 20), mono: true, w: 700, align: 'center', color: bit ? theme.c('pos') : theme.c('muted') }),
      inner: (ctx, r, S) => {
        const u = S.u(r, 8, 13), st = stage(), P = (f) => S.place(r, f), f = P(fet), c = P(cap), wl = r.y + r.h * 0.16, bl = r.x + r.w * 0.08, read = st === 2;
        ctx.strokeStyle = read ? theme.c('pos') : theme.c('ink-2'); ctx.lineWidth = read ? 3 : 2; ctx.beginPath(); ctx.moveTo(r.x, wl); ctx.lineTo(r.x + r.w * 0.9, wl); ctx.stroke();
        S.text(ctx, `word line ${read ? '(reading: on)' : '(off)'}`, r.x + r.w * 0.9, wl - u * 0.8, { size: u * 0.85, align: 'right', color: read ? theme.c('pos') : theme.c('muted') });
        ctx.strokeStyle = theme.c('ink-2'); ctx.lineWidth = 2; ctx.beginPath(); ctx.moveTo(bl, wl + 10); ctx.lineTo(bl, r.y + r.h * 0.95); ctx.stroke();
        S.text(ctx, 'bit line', bl + 6, r.y + r.h * 0.92, { size: u * 0.85, color: theme.c('muted') });
        ctx.beginPath(); ctx.moveTo(f.x + f.w * 0.44, wl); ctx.lineTo(f.x + f.w * 0.44, f.y); ctx.stroke();
        ctx.beginPath(); ctx.moveTo(bl, f.y + f.h / 2); ctx.lineTo(f.x, f.y + f.h / 2); ctx.moveTo(f.x + f.w, f.y + f.h / 2); ctx.lineTo(c.x, f.y + f.h / 2); ctx.stroke();
        ctx.beginPath(); ctx.moveTo(c.x + c.w / 2, c.y + c.h); ctx.lineTo(c.x + c.w / 2, r.y + r.h * 0.9); ctx.stroke();
        S.text(ctx, 'plate', c.x + c.w / 2 + 6, r.y + r.h * 0.88, { size: u * 0.8, color: theme.c('muted') });
        S.text(ctx, `${read ? 'charge shared with the bit line, sensed, written back' : bit ? 'charged: a 1' : 'empty: a 0'}`, r.x + r.w * 0.74, c.y + c.h / 2, { size: u * 0.85, color: theme.c('ink-2'), maxW: r.w * 0.26 });
        timeline(ctx, S, { x: r.x + r.w * 0.1, y: r.y + r.h * 0.74, w: r.w * 0.6, h: u * 2.2 }, ['idle', 'word line on', 'read', 'rewritten']);
      },
      about: [`A DRAM memory cell stores one bit as charge on a capacitor, reached through one transistor: ${bit ? 'this one is charged, a 1' : 'this one is empty, a 0'}. To read it, the word line turns the transistor on and the charge spills onto the bit line, where a sense amplifier detects the tiny change. Reading drains the cell, so the bit is written straight back. Billions of these cells make up a memory chip.`, 'Fetching a number from DRAM costs far more time and energy than doing arithmetic with it: hundreds of picojoules against about one for a multiply-add. That is why chips keep numbers in caches and registers as long as they can, why GPUs have very wide memory buses, and why large models are often limited by how fast memory can feed them rather than by arithmetic.', 'Zoom into the transistor or the capacitor.'],
      code: `class DRAMCell:\n    def __init__(self, bit):\n        self.charge = 1.0 if bit else 0.0   # fraction of a full capacitor\n\n    def read(self, bitline=0.5):\n        bit = 1 if self.charge > bitline else 0   # the sense amplifier compares\n        self.charge = float(bit)                    # and writes the bit back\n        return bit\n\nprint(DRAMCell(${bit}).read())`,
    }, () => [
      Object.assign(D.mos(X, { kind: 'n', on: true, before: false, path: !!bit, input: 'word line', v: 1, vBefore: 0, charge: false }), { id: 'fet', rect: fet, title: 'Access transistor', sub: 'opened by the word line', summary: (ctx, r) => fetGlyph(ctx, 'n', r, stage() === 2) }),
      Object.assign(D.electrons(X, { mode: 'cap', charged: !!bit, kind: 'n' }), { id: 'cap', rect: cap, title: 'Capacitor', sub: bit ? 'charged' : 'empty', summary: (ctx, r) => { ctx.strokeStyle = theme.c('ink-2'); ctx.lineWidth = 2.5; ctx.beginPath(); ctx.moveTo(r.x + r.w * 0.2, r.y + r.h * 0.4); ctx.lineTo(r.x + r.w * 0.8, r.y + r.h * 0.4); ctx.moveTo(r.x + r.w * 0.2, r.y + r.h * 0.6); ctx.lineTo(r.x + r.w * 0.8, r.y + r.h * 0.6); ctx.stroke(); if (bit) for (let i = 0; i < 6; i++) electron(ctx, r.x + r.w * (0.25 + i * 0.1), r.y + r.h * 0.3, 2); } }),
    ]);
  };

  D.lazy = node;
  NA.deep = D;
})(window.NA);
