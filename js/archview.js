/* archview.js — zoomable architecture diagrams with semantic zoom. The view shows one level at a time,
   filling the screen: the whole model, then a block, then its attention layer, then one head. Zoom in on a
   part (mouse wheel, pinch or the + button) or click it and the view flies into it, the part opening into its
   own screen with its inner workings and maths; zoom out (or the breadcrumbs) to come back. Each level is
   rebuilt from live data on every draw, so the numbers inside always belong to the current model and input.

   Node: { id, title, sub, tone, rect: [fx, fy, fw, fh] in the parent's content box, shape: 'box' | 'op' | 'zone',
           summary(ctx, r, S) when drawn small inside its parent, inner(ctx, c, S) when it is the open level,
           children: [nodes], grids: [{ id, rect, rows, cols, square, make(i, j) → node }], animate,
           about: [paragraphs], code: 'python', codeLabel }
   A 'zone' child is part of its parent's own drawing (a call in a line of code, a row of maths): it is only
   outlined. A grid turns a heatmap or a row of bars into cells that open on demand, so any number under the
   cursor can be zoomed into. Levels marked animate are redrawn every frame (S.now is the time). */
(function (NA) {
  'use strict';
  const { el, clear, clamp, theme } = NA;
  const smooth = (t) => { t = clamp(t, 0, 1); return t * t * (3 - 2 * t); };
  const easeInOut = (t) => (t < 0.5 ? 4 * t * t * t : 1 - Math.pow(-2 * t + 2, 3) / 2);
  let FONTS = null;
  const fonts = () => {
    if (!FONTS) { const cs = getComputedStyle(document.body); FONTS = { ui: cs.getPropertyValue('--font-ui') || 'sans-serif', mono: cs.getPropertyValue('--font-mono') || 'monospace' }; }
    return FONTS;
  };
  const font = (w, px, mono) => `${w} ${px.toFixed(1)}px ${mono ? fonts().mono : fonts().ui}`;

  // ------------------------------------------------------------------ drawing kit shared by every diagram
  const S = {
    font,
    place: (c, f) => ({ x: c.x + f[0] * c.w, y: c.y + f[1] * c.h, w: f[2] * c.w, h: f[3] * c.h }),
    // a font unit that grows with the box but stays readable
    u: (c, lo = 7.5, hi = 14) => clamp(Math.min(c.w / 34, c.h / 14), lo, hi),
    fit(ctx, s, maxW) {
      if (!(maxW > 0) || ctx.measureText(s).width <= maxW) return s;
      let lo = 0, hi = s.length;
      while (lo < hi) { const m = (lo + hi + 1) >> 1; if (ctx.measureText(s.slice(0, m) + '…').width <= maxW) lo = m; else hi = m - 1; }
      return lo > 0 ? s.slice(0, lo) + '…' : '';
    },
    text(ctx, s, x, y, o = {}) {
      ctx.font = font(o.w || 500, o.size || 12, o.mono);
      ctx.fillStyle = o.color || theme.c('ink');
      ctx.textAlign = o.align || 'left'; ctx.textBaseline = o.base || 'middle';
      ctx.fillText(o.maxW ? S.fit(ctx, String(s), o.maxW) : String(s), x, y);
    },
    // lines of text from the top-left of r: the font shrinks to fit, down to a readable size; past that,
    // long lines wrap at a space and carry on indented
    lines(ctx, lines, r, o = {}) {
      let size = o.size || S.u(r);
      ctx.font = font(o.w || 500, size, o.mono);
      const widest = Math.max(...lines.map((l) => ctx.measureText(l).width), 1);
      if (widest > r.w) size = Math.max(Math.min(size, o.min || 9), size * (r.w / widest));
      ctx.font = font(o.w || 500, size, o.mono);
      const rows = [];
      lines.forEach((l, i) => {
        let rest = l, ind = '';
        while (ctx.measureText(ind + rest).width > r.w) {
          let cut = -1;
          for (let k = rest.length - 1; k > 0; k--) if (rest[k] === ' ' && rest[k - 1] !== ' ' && ctx.measureText(ind + rest.slice(0, k)).width <= r.w) { cut = k; break; }
          if (cut <= 0) break;
          rows.push([ind + rest.slice(0, cut), i]);
          rest = rest.slice(cut).trimStart(); ind = '    ';
        }
        rows.push([ind + rest, i]);
      });
      const lh = size * (o.lh || 1.45), n = Math.max(1, Math.floor(r.h / lh));
      rows.slice(0, n).forEach(([l, i], k) => S.text(ctx, l, r.x, r.y + lh * (k + 0.5), { size, mono: o.mono, w: (o.strong && o.strong(i)) ? 700 : o.w || 500, color: o.colorOf ? o.colorOf(i) : o.color }));
      return lh * Math.min(n, rows.length);
    },
    box(ctx, r, o = {}) {
      const rad = o.radius ?? Math.min(10, r.w * 0.05, r.h * 0.2);
      ctx.beginPath(); NA.rr(ctx, r.x, r.y, r.w, r.h, rad);
      if (o.fill !== false) { ctx.fillStyle = o.fill || theme.c('panel'); ctx.fill(); }
      if (o.stroke !== false) { ctx.strokeStyle = o.stroke || theme.c('line-2'); ctx.lineWidth = o.lw || 1; ctx.stroke(); }
    },
    /* matrix heatmap. get(i, j) → value; o.seq for 0..1 data, else diverging around 0; o.square keeps cells square */
    heat(ctx, get, rows, cols, r, o = {}) {
      let cw = r.w / cols, ch = r.h / rows;
      if (o.square) cw = ch = Math.min(cw, ch);
      let m = o.max;
      if (!m) { m = 1e-9; for (let i = 0; i < rows; i++) for (let j = 0; j < cols; j++) { const v = get(i, j); if (isFinite(v)) m = Math.max(m, Math.abs(v)); } }
      for (let i = 0; i < rows; i++) for (let j = 0; j < cols; j++) {
        const v = get(i, j);
        ctx.fillStyle = o.mask && o.mask(i, j) ? theme.c('line', 0.6) : o.seq ? theme.seqColor(Math.sqrt(clamp(v / m, 0, 1))) : theme.divColor(v / m);
        ctx.fillRect(r.x + j * cw, r.y + i * ch, cw + 0.35, ch + 0.35);
      }
      if (o.hiRow !== undefined && o.hiRow !== null) { ctx.strokeStyle = theme.c('focus'); ctx.lineWidth = 1.2; ctx.strokeRect(r.x - 0.5, r.y + o.hiRow * ch - 0.5, cols * cw + 1, ch + 1); }
      if (o.frame !== false) { ctx.strokeStyle = theme.c('line-2'); ctx.lineWidth = 0.8; ctx.strokeRect(r.x - 0.5, r.y - 0.5, cols * cw + 1, rows * ch + 1); }
      return { cw, ch, w: cols * cw, h: rows * ch };
    },
    // bars around a zero line
    bars(ctx, vals, r, o = {}) {
      const n = vals.length, bw = r.w / n;
      let m = o.max || 1e-9;
      if (!o.max) for (let i = 0; i < n; i++) m = Math.max(m, Math.abs(vals[i]));
      const zero = o.positive ? r.y + r.h : r.y + r.h / 2, span = o.positive ? r.h : r.h / 2;
      ctx.strokeStyle = theme.c('line-2'); ctx.lineWidth = 0.8; ctx.beginPath(); ctx.moveTo(r.x, zero); ctx.lineTo(r.x + r.w, zero); ctx.stroke();
      for (let i = 0; i < n; i++) {
        const v = vals[i], h = (Math.abs(v) / m) * span;
        ctx.fillStyle = o.color ? o.color(i, v) : theme.divColor(clamp(v / m, -1, 1) * 0.85 + Math.sign(v) * 0.15);
        ctx.fillRect(r.x + i * bw + bw * 0.12, v >= 0 ? zero - h : zero, Math.max(0.6, bw * 0.76), h);
      }
    },
    arrow(ctx, x1, y1, x2, y2, o = {}) {
      ctx.strokeStyle = o.color || theme.c('ink-2'); ctx.fillStyle = o.color || theme.c('ink-2'); ctx.lineWidth = o.lw || 1.3;
      if (o.dash) ctx.setLineDash(o.dash);
      ctx.beginPath(); ctx.moveTo(x1, y1);
      let a;
      if (o.bend) { const cx = (x1 + x2) / 2 + o.bend[0], cy = (y1 + y2) / 2 + o.bend[1]; ctx.quadraticCurveTo(cx, cy, x2, y2); a = Math.atan2(y2 - cy, x2 - cx); }
      else { ctx.lineTo(x2, y2); a = Math.atan2(y2 - y1, x2 - x1); }
      ctx.stroke(); ctx.setLineDash([]);
      const h = o.head || 6;
      ctx.beginPath(); ctx.moveTo(x2, y2); ctx.lineTo(x2 - h * Math.cos(a - 0.42), y2 - h * Math.sin(a - 0.42)); ctx.lineTo(x2 - h * Math.cos(a + 0.42), y2 - h * Math.sin(a + 0.42)); ctx.closePath(); ctx.fill();
    },
    curve(ctx, fn, r, x0, x1, y0, y1, o = {}) {
      ctx.strokeStyle = o.color || theme.c('ink'); ctx.lineWidth = o.lw || 1.6; if (o.dash) ctx.setLineDash(o.dash);
      ctx.beginPath();
      for (let i = 0; i <= 80; i++) { const x = x0 + ((x1 - x0) * i) / 80, y = clamp(fn(x), y0, y1), px = r.x + (i / 80) * r.w, py = r.y + r.h - ((y - y0) / (y1 - y0)) * r.h; i ? ctx.lineTo(px, py) : ctx.moveTo(px, py); }
      ctx.stroke(); ctx.setLineDash([]);
    },
    caption(ctx, s, r, o = {}) { S.text(ctx, s, r.x, r.y - (o.size || 9) * 0.9, { size: o.size || 9, w: 600, color: o.color || theme.c('muted'), maxW: Math.max(r.w, o.minW || 0) }); },
    frame(ctx, r) { ctx.strokeStyle = theme.c('line'); ctx.lineWidth = 1; ctx.strokeRect(r.x + 0.5, r.y + 0.5, r.w - 1, r.h - 1); },
    // choose fractions by shape: wide (landscape) or tall (portrait) content
    pick: (c, wide, tall) => (c.w / c.h >= 1.05 ? wide : tall),
  };

  // ------------------------------------------------------------------ the view
  class ArchView {
    /* stage: positioned container; opts: { build(size) → root node, onFocus(node, path, isLevel), label,
       outer: { title, exit() } when the root itself was opened from another view (backing out of it returns) } */
    constructor(stage, opts) {
      this.stage = stage; this.opts = opts; this.dirty = true;
      this.path = []; this.anim = null; this.selected = null;
      this.canvas = el('canvas', { class: 'arch-canvas', tabindex: '0', 'aria-label': opts.label || 'Architecture diagram: click a part to open it' });
      this.back = el('button', { class: 'icon-btn arch-back', type: 'button', 'aria-label': 'Back out one level', title: 'Back out one level (Esc)', onclick: () => this.exit() }, '←');
      this.crumbs = el('nav', { class: 'arch-crumbs', 'aria-label': 'Where you are in the model' });
      this.bar = el('div', { class: 'arch-bar' }, this.back, this.crumbs);
      stage.append(this.canvas, this.bar);
      this.zoom = new NA.ui.Zoom(this.canvas, {
        host: stage, panOne: true, max: 3, noDouble: true, wheel: 'always', wheelGate: () => this.wheelOpen(),
        onChange: () => { this.dirty = true; NA.tooltip.hide(); if (!this.locked() && this.zoom.k > 2.3) this.zoomedIn(); },
        onUnderflow: () => { if (!this.locked()) this.exit(); },
      });
      this.canvas.addEventListener('click', (e) => this.click(e));
      this.canvas.addEventListener('pointermove', (e) => this.hover(e));
      const release = () => { if (this.zoom.ptrs.size === 0) this.gestureLock = false; };
      this.canvas.addEventListener('pointerup', release); this.canvas.addEventListener('pointercancel', release);
      this.canvas.addEventListener('pointerleave', () => { this.hoverId = null; this.dirty = true; NA.tooltip.hide(); });
      this.canvas.addEventListener('keydown', (e) => { if (e.key === 'Escape' || e.key === 'Backspace') { e.preventDefault(); this.exit(); } });
      this.ro = new ResizeObserver(() => { this.dirty = true; });
      this.ro.observe(stage);
    }
    destroy() { this.ro.disconnect(); this.anim = null; }
    size() { return [Math.max(160, this.stage.clientWidth), Math.max(160, this.stage.clientHeight)]; }
    levelRect() { const [W, H] = this.size(); return { x: 10, y: 46, w: W - 20, h: H - 56 }; }
    content(node, r) { const hy = node.header ?? (this.hy || 0.07); return { x: r.x + r.w * 0.02, y: r.y + r.h * hy, w: r.w * 0.96, h: r.h * (1 - hy - 0.02) }; }
    node() { let n = this.root; for (const id of this.path.slice(1)) n = this.childOf(n, id) || n; return n; }
    // a part of a level by id: a listed child, or a cell of one of its grids ("grid:i:j"), made on demand
    childOf(parent, id) {
      if (!parent) return null;
      const ch = (parent.children || []).find((c) => c.id === id);
      if (ch || !parent.grids) return ch || null;
      const m = /^(.+):(\d+):(\d+)$/.exec(id), g = m && parent.grids.find((q) => q.id === m[1]);
      return g && +m[2] < g.rows && +m[3] < g.cols ? this.cellNode(parent, g, +m[2], +m[3]) : null;
    }
    cellNode(parent, g, i, j) {
      const node = g.make(i, j);
      if (!node) return null;
      const c = this.content(parent, this.levelRect()), gr = S.place(c, g.rect), { cw, ch } = this.cellSize(g, gr);
      return Object.assign(node, { id: `${g.id}:${i}:${j}`, rect: [(gr.x + j * cw - c.x) / c.w, (gr.y + i * ch - c.y) / c.h, cw / c.w, ch / c.h] });
    }
    cellSize(g, gr) { let cw = gr.w / g.cols, ch = gr.h / g.rows; if (g.square) cw = ch = Math.min(cw, ch); return { cw, ch }; }
    // the open level (or the part it shows small) moves: keep redrawing
    levelAnimates() { const n = this.root && this.node(); return !!(n && (n.animate || (n.children || []).some((c) => c.animate))); }
    camRect(r, cam) { return { x: r.x * cam.k + cam.tx, y: r.y * cam.k + cam.ty, w: r.w * cam.k, h: r.h * cam.k }; }
    camNow() { return { k: this.zoom.k, tx: this.zoom.tx, ty: this.zoom.ty }; }
    // one long pinch or trackpad gesture should open one level, not tunnel through several
    locked() { return !!this.anim || performance.now() < (this.lockUntil || 0) || (this.zoom.ptrs.size > 0 && this.gestureLock); }
    canEnter(node) { return !!node && (node.shape !== 'op' || !!node.inner) && !!(node.inner || node.children); }

    frame(now) {
      if (this.anim) { this.dirty = true; this.stepAnim(now); }
      else if (now - (this.lastMotion || 0) > 32 && this.levelAnimates()) { this.lastMotion = now; this.dirty = true; }
      if (!this.dirty || !this.canvas.offsetParent) return;
      this.dirty = false;
      this.draw();
    }
    draw() {
      const [W, H] = this.size(), ctx = NA.fitCanvas(this.canvas, W, H), R = this.levelRect();
      ctx.clearRect(0, 0, W, H);
      S.now = performance.now();
      this.hy = Math.max(0.07, 30 / R.h); // room for the level title, whatever the stage height
      const chip = this.zoom.chip;
      this.chipW = chip && chip.offsetParent ? chip.offsetWidth + 14 : 0;
      this.root = this.opts.build({ w: R.w * 0.96, h: R.h * 0.91 });
      if (!this.path.length) this.path = [this.root.id];
      // parts can disappear when the model changes shape: stay at the deepest level that still exists
      let n = this.root, keep = 1;
      for (const id of this.path.slice(1)) { const c = this.childOf(n, id); if (!c) break; n = c; keep++; }
      if (keep < this.path.length) { this.path = this.path.slice(0, keep); this.selected = null; this.focusKey = null; }
      this.W = W; this.H = H;
      const a = this.anim;
      if (!a) { const node = this.node(); this.drawLevel(ctx, node, R, this.camNow(), 1); this.drawHoverCell(ctx, node, R); }
      else {
        /* The parent zooms toward the part, and the part's own level is drawn inside the part's box, growing with
           it until it fills the view: you travel into the box rather than cutting to a new screen. Backing out
           plays the same thing in reverse. */
        const s = a.s, parent = a.parent, child = a.child;
        const cam = { k: 1, tx: 0, ty: 0 };
        const lk = Math.exp(Math.log(a.c0.k) + (Math.log(a.c1.k) - Math.log(a.c0.k)) * s);
        const cx = a.w0[0] + (a.w1[0] - a.w0[0]) * s, cy = a.w0[1] + (a.w1[1] - a.w0[1]) * s;
        cam.k = lk; cam.tx = R.x + R.w / 2 - cx * lk; cam.ty = R.y + R.h / 2 - cy * lk;
        const childAlpha = smooth((s - 0.18) / 0.5), parentAlpha = 1 - smooth((s - 0.5) / 0.45) * 0.92;
        this.drawLevel(ctx, parent, R, cam, parentAlpha, false, child.id, childAlpha);
        if (childAlpha > 0) {
          // where the part's box is on screen right now, and the child level scaled to fit inside it
          const box = this.camRect({ x: a.w1[0] - a.bw / 2, y: a.w1[1] - a.bh / 2, w: a.bw, h: a.bh }, cam);
          const sc = Math.min(1, Math.min(box.w / R.w, box.h / R.h)), ox = box.x + box.w / 2 - (R.x + R.w / 2) * sc, oy = box.y + box.h / 2 - (R.y + R.h / 2) * sc;
          ctx.save();
          ctx.translate(ox, oy); ctx.scale(sc, sc);
          this.drawLevel(ctx, child, R, { k: 1, tx: 0, ty: 0 }, childAlpha, true);
          ctx.restore();
        }
      }
      ctx.globalAlpha = 1;
      this.syncBar();
    }
    // fadeId: the part being entered, whose small preview fades as its own level grows inside it
    drawLevel(ctx, node, R, cam, alpha, cover, fadeId, fade = 0) {
      const r = this.camRect(R, cam);
      ctx.globalAlpha = alpha;
      // a level drawn over another covers it, as a panel with soft corners growing out of its box
      if (cover) { ctx.fillStyle = theme.c('panel'); ctx.beginPath(); NA.rr(ctx, r.x - 6, r.y - 6, r.w + 12, r.h + 12, 12); ctx.fill(); }
      // the zoom buttons sit at the top right, so the open level's title stops short of them
      S.text(ctx, node.title + (node.sub ? '  ·  ' + node.sub : ''), r.x + 2, r.y + 14, { size: clamp(14 * cam.k, 9, 40), w: 650, maxW: cam.k > 1.001 ? r.w : Math.max(60, r.w - (this.chipW || 0)) });
      const c = this.content(node, r);
      if (node.inner) { ctx.save(); ctx.globalAlpha = alpha; node.inner(ctx, c, S); ctx.restore(); }
      for (const ch of node.children || []) this.drawClosed(ctx, ch, S.place(c, ch.rect), ch.id === fadeId ? alpha * (1 - fade) : alpha);
    }
    // the grid cell under the cursor gets an outline, so it is clear which number zooming in will open
    drawHoverCell(ctx, node, R) {
      const m = this.hoverId && node.grids && /^(.+):(\d+):(\d+)$/.exec(this.hoverId), g = m && node.grids.find((q) => q.id === m[1]);
      if (!g) return;
      const c = this.content(node, this.camRect(R, this.camNow())), gr = S.place(c, g.rect), { cw, ch } = this.cellSize(g, gr);
      ctx.strokeStyle = theme.c('focus'); ctx.lineWidth = 2;
      ctx.strokeRect(gr.x + +m[3] * cw - 1, gr.y + +m[2] * ch - 1, cw + 2, ch + 2);
    }
    drawClosed(ctx, node, r, alpha) {
      if (r.x > this.W || r.y > this.H || r.x + r.w < 0 || r.y + r.h < 0 || r.w < 2) return;
      const hov = this.hoverId === node.id || this.selected === node.id;
      ctx.globalAlpha = alpha;
      if (node.shape === 'zone') {
        // part of the parent's own drawing: a dotted outline says it opens, a solid one that it is under the cursor
        ctx.setLineDash(hov ? [] : [2, 3]); ctx.strokeStyle = hov ? theme.c('focus') : theme.c('muted', 0.55); ctx.lineWidth = hov ? 1.6 : 1;
        ctx.beginPath(); NA.rr(ctx, r.x - 2, r.y - 1, r.w + 4, r.h + 2, 4);
        if (hov) { ctx.fillStyle = theme.c('focus', 0.1); ctx.fill(); }
        ctx.stroke(); ctx.setLineDash([]);
        return;
      }
      if (node.shape === 'op') {
        const s = Math.min(r.w, r.h), cx = r.x + r.w / 2, cy = r.y + r.h / 2;
        ctx.beginPath(); ctx.arc(cx, cy, s / 2, 0, Math.PI * 2); ctx.fillStyle = theme.c('panel'); ctx.fill();
        ctx.strokeStyle = hov ? theme.c('focus') : theme.c('ink-2'); ctx.lineWidth = hov ? 2 : 1.3; ctx.stroke();
        S.text(ctx, node.title, cx, cy + 0.5, { size: s * 0.62, w: 600, align: 'center' });
        return;
      }
      S.box(ctx, r, { fill: theme.c('panel'), stroke: hov ? theme.c('focus') : node.tone ? theme.c(node.tone, 0.7) : theme.c('line-2'), lw: hov ? 2 : 1.1 });
      if (node.tone) { ctx.fillStyle = theme.c(node.tone, theme.dark ? 0.1 : 0.07); ctx.beginPath(); NA.rr(ctx, r.x, r.y, r.w, r.h, Math.min(10, r.w * 0.05, r.h * 0.2)); ctx.fill(); }
      const ts = clamp(Math.min(r.h * 0.2, r.w * 0.085), 7, 16), hasSum = !!node.summary && r.h > ts * 3.4;
      const ty = hasSum ? r.y + ts * 1.1 : r.y + r.h / 2 - (node.sub && r.h > ts * 2.6 ? ts * 0.45 : 0);
      S.text(ctx, node.title, r.x + r.w / 2, ty, { size: ts, w: 650, align: 'center', maxW: r.w - 8 });
      let y = ty + ts * 0.85;
      if (node.sub && r.h > ts * 2.6) { S.text(ctx, node.sub, r.x + r.w / 2, ty + ts * 1.1, { size: ts * 0.72, color: theme.c('muted'), align: 'center', maxW: r.w - 8 }); y = ty + ts * 1.75; }
      if (hasSum) { ctx.save(); ctx.globalAlpha = alpha; node.summary(ctx, { x: r.x + r.w * 0.06, y, w: r.w * 0.88, h: r.y + r.h - y - Math.min(8, r.h * 0.08) }, S); ctx.restore(); }
      if (this.canEnter(node) && r.w > 46 && r.h > 30) S.text(ctx, '⤢', r.x + r.w - 6, r.y + 9, { size: 10, align: 'right', color: theme.c('muted') });
    }

    // children of the current level under a screen point
    pick(px, py) {
      if (!this.root) return null;
      const node = this.node(), c = this.content(node, this.camRect(this.levelRect(), this.camNow()));
      for (const ch of node.children || []) {
        const r = S.place(c, ch.rect);
        const pad = ch.shape === 'op' ? 6 : 0;
        if (px >= r.x - pad && py >= r.y - pad && px <= r.x + r.w + pad && py <= r.y + r.h + pad) return { node: ch, r };
      }
      for (const g of node.grids || []) {
        const gr = S.place(c, g.rect), { cw, ch } = this.cellSize(g, gr), i = Math.floor((py - gr.y) / ch), j = Math.floor((px - gr.x) / cw);
        if (py < gr.y || px < gr.x || i >= g.rows || j >= g.cols || (g.skip && g.skip(i, j))) continue;
        const cell = this.cellNode(node, g, i, j);
        if (cell) return { node: cell, r: { x: gr.x + j * cw, y: gr.y + i * ch, w: cw, h: ch } };
      }
      return null;
    }
    click(e) {
      if (this.zoom.suppressClick || this.anim) return;
      const b = this.canvas.getBoundingClientRect(), f = this.pick(e.clientX - b.left, e.clientY - b.top);
      if (!f) { if (this.selected) { this.selected = null; this.dirty = true; this.notify(); } return; }
      if (this.canEnter(f.node)) this.enter(f.node);
      else { this.selected = f.node.id; this.dirty = true; this.notify(); }
    }
    hover(e) {
      if (this.zoom.busy || this.anim) return;
      const b = this.canvas.getBoundingClientRect(), f = this.pick(e.clientX - b.left, e.clientY - b.top), id = f ? f.node.id : null;
      if (id !== this.hoverId) { this.hoverId = id; this.dirty = true; }
      this.canvas.style.cursor = f ? (this.canEnter(f.node) ? 'zoom-in' : 'pointer') : 'default';
      if (f) NA.tooltip.show([el('div', { class: 'tt-title', text: f.node.title }), f.node.sub ? el('div', { class: 'tt-hint', text: f.node.sub }) : null, el('div', { class: 'tt-hint', text: this.canEnter(f.node) ? 'Click or zoom in to open it' : 'Click for the explanation and code' })], e.clientX, e.clientY);
      else NA.tooltip.hide();
    }
    // zoomed far enough into the level: open the part nearest the middle of the view
    zoomedIn() {
      // the part under the cursor or the fingers, or under the middle for the + button
      const [W, H] = this.size(), a = this.zoom.anchor, cx = a ? a[0] : W / 2, cy = a ? a[1] : (H + 46) / 2;
      let f = this.pick(cx, cy);
      if (!f || !this.canEnter(f.node)) {
        // nearest enterable part to that point
        const node = this.node(), c = this.content(node, this.camRect(this.levelRect(), this.camNow()));
        let best = null, bd = Infinity;
        for (const ch of node.children || []) {
          if (!this.canEnter(ch)) continue;
          const r = S.place(c, ch.rect), dx = clamp(cx, r.x, r.x + r.w) - cx, dy = clamp(cy, r.y, r.y + r.h) - cy, dd = dx * dx + dy * dy;
          if (dd < bd) { bd = dd; best = { node: ch, r }; }
        }
        if (!best || bd > (W * 0.25) ** 2) return;
        f = best;
      }
      this.enter(f.node);
    }
    // animate into a part of the current level
    enter(child) {
      const R = this.levelRect(), parent = this.node(), c = this.content(parent, R), cr = S.place(c, child.rect);
      const k1 = Math.min(R.w / cr.w, R.h / cr.h), c0 = this.camNow();
      this.anim = { parent, child, s: 0, t0: performance.now(), dir: 1, c0, c1: { k: k1 }, bw: cr.w, bh: cr.h,
        w0: [(R.x + R.w / 2 - c0.tx) / c0.k, (R.y + R.h / 2 - c0.ty) / c0.k], w1: [cr.x + cr.w / 2, cr.y + cr.h / 2], done: () => { this.path = this.path.concat(child.id); } };
      this.selected = null; this.wheelHold = true;
      this.dirty = true;
      NA.tooltip.hide();
    }
    // back out to the parent (or to an ancestor at depth i)
    exit(depth) {
      if (this.anim) return;
      if (this.path.length < 2) { if (this.opts.outer && depth === undefined) this.opts.outer.exit(); return; }
      const target = depth === undefined ? this.path.length - 2 : depth;
      if (target === this.path.length - 1) return;
      const R = this.levelRect();
      const parentPath = this.path.slice(0, target + 1), childId = this.path[target + 1];
      let parent = this.root; for (const id of parentPath.slice(1)) parent = this.childOf(parent, id) || parent;
      const child = this.childOf(parent, childId), cur = this.node();
      if (!child) { this.path = parentPath; this.dirty = true; return; }
      const cr = S.place(this.content(parent, R), child.rect), k1 = Math.min(R.w / cr.w, R.h / cr.h);
      this.path = parentPath;
      this.zoomSet(1, 0, 0);
      this.anim = { parent, child: cur, s: 1, t0: performance.now(), dir: -1, c0: { k: 1 }, c1: { k: k1 }, bw: cr.w, bh: cr.h, w0: [R.x + R.w / 2, R.y + R.h / 2], w1: [cr.x + cr.w / 2, cr.y + cr.h / 2], done: null };
      this.selected = null; this.wheelHold = true;
      this.dirty = true;
    }
    stepAnim(now) {
      const a = this.anim, dur = NA.prefersReducedMotion() ? 1 : 560, p = clamp((now - a.t0) / dur, 0, 1), e = easeInOut(p);
      a.s = a.dir > 0 ? e : 1 - e;
      if (p >= 1) {
        this.anim = null;
        // fingers still down must lift before a pinch can open another level; the wheel waits for a pause (wheelOpen)
        this.lockUntil = now + 120; this.gestureLock = this.zoom.ptrs.size > 0;
        if (a.done) a.done();
        this.zoomSet(1, 0, 0);
        this.focusKey = null;
        this.notify();
      }
    }
    zoomSet(k, tx, ty) { const keep = this.zoom.onChange; this.zoom.onChange = () => {}; this.zoom.set(k, tx, ty); this.zoom.onChange = keep; this.zoom.rebase(); }
    // one long scroll or trackpad pinch should open one level, not several: once a level changes, the wheel
    // has to pause briefly before it zooms again
    wheelOpen() {
      const now = performance.now(), quiet = now - (this.lastWheel || 0) > 220;
      this.lastWheel = now;
      if (this.anim || (this.wheelHold && !quiet)) return false;
      this.wheelHold = false;
      return true;
    }
    // tell the lab what is being looked at: the level, or a part clicked inside it
    notify() {
      if (!this.root) return;
      const level = this.node(), sel = this.selected ? (level.children || []).find((c) => c.id === this.selected) : null;
      const key = this.path.join('/') + '|' + (sel ? sel.id : '');
      if (key === this.focusKey) return;
      this.focusKey = key;
      if (this.opts.onFocus) this.opts.onFocus(sel || level, this.path, !sel);
    }
    refreshFocus() { this.focusKey = null; this.notify(); }
    // what the info panel shows for a part: where it sits, its explanation and its code
    infoFor(node, path) {
      const names = [];
      let n = this.root;
      for (const id of path.slice(1)) { n = this.childOf(n, id) || n; names.push(n.title); }
      const trail = node.id === path[path.length - 1] ? names.slice(0, -1) : names;
      return { kicker: [this.opts.kicker || 'Architecture', ...trail].join(' › '), title: node.title + (node.sub ? ' · ' + node.sub : ''), about: node.about, code: node.code, codeLabel: node.codeLabel };
    }
    syncBar() {
      const key = this.path.join('/');
      const outer = this.opts.outer;
      this.back.hidden = this.path.length < 2 && !outer;
      if (key === this.crumbKey) return;
      this.crumbKey = key;
      clear(this.crumbs);
      // opened from another view: its name comes first and leads back to it
      if (outer) this.crumbs.append(el('button', { type: 'button', class: 'arch-crumb', onclick: () => outer.exit() }, outer.title), el('span', { class: 'arch-sep', 'aria-hidden': 'true', text: '›' }));
      let n = this.root;
      this.path.forEach((id, i) => {
        if (i) { n = this.childOf(n, id) || n; this.crumbs.append(el('span', { class: 'arch-sep', 'aria-hidden': 'true', text: '›' })); }
        const here = i === this.path.length - 1;
        this.crumbs.append(el('button', { type: 'button', class: 'arch-crumb' + (here ? ' is-here' : ''), 'aria-current': here ? 'location' : null, onclick: () => { if (!here) this.exit(i); } }, n.title));
      });
      this.crumbs.scrollLeft = this.crumbs.scrollWidth; // keep the current level in view
      if (!this.anim) this.notify();
    }
    goRoot() { if (this.path.length > 1) this.exit(0); }
  }

  NA.ArchView = ArchView;
  NA.archKit = S;
})(window.NA);
