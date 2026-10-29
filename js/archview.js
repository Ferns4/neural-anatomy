/* archview.js — zoomable architecture diagrams with semantic zoom. The view shows one level at a time,
   filling the screen: the whole model, then a block, then its attention layer, then one head. Zoom in on a
   part (mouse wheel, pinch or the + button) or click it and the view flies into it, the part opening into its
   own screen with its inner workings and maths; zoom out (or the breadcrumbs) to come back. Each level is
   rebuilt from live data on every draw, so the numbers inside always belong to the current model and input.

   Node: { id, title, sub, tone, rect: [fx, fy, fw, fh] in the parent's content box, shape: 'box' | 'op' | 'zone',
           summary(ctx, r, S) when drawn small inside its parent, inner(ctx, c, S) when it is the open level,
           children: [nodes], grids: [{ id, rect, rows, cols, square, make(i, j) → node }], animate,
           about: [paragraphs], code: 'python', codeLabel,
           terms: [{ key, words, ids, cells(id) }] (phrases of the explanation that point at parts of the picture),
           count: 'what it is made of', tools(view) → [buttons], toolsKey, onPick(part, view), pickHint(part, view),
           portal(view) (entering the part leaves this view instead) }
   A 'zone' child is part of its parent's own drawing (a call in a line of code, a row of maths): it is only
   outlined. A grid turns a heatmap or a row of bars into cells that open on demand, so any number under the
   cursor can be zoomed into. Levels marked animate are redrawn every frame (S.now is the time). Words of the
   explanation that name a part are linked to it: pointing at the words lights the part up, and pointing at a
   part lights its words (S.hl is the lit key while a level draws). */
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
      const wrap = () => {
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
        return rows;
      };
      let rows = wrap();
      // more lines than the box is tall: a smaller font (down to a readable 8 px) rather than lines left out
      const fit = r.h / ((o.lh || 1.45) * rows.length);
      if (fit < size && size > 8) { size = Math.max(8, fit); rows = wrap(); }
      const lh = size * (o.lh || 1.45), n = Math.max(1, Math.floor(r.h / lh + 0.01));
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
    // a part the explanation is pointing at: a glowing outline
    lit(ctx, r, rad = 6) {
      ctx.save(); ctx.globalAlpha = 1;
      ctx.strokeStyle = theme.c('focus', 0.28); ctx.lineWidth = 6; ctx.beginPath(); NA.rr(ctx, r.x - 3, r.y - 3, r.w + 6, r.h + 6, rad + 3); ctx.stroke();
      ctx.strokeStyle = theme.c('focus'); ctx.lineWidth = 2; ctx.beginPath(); NA.rr(ctx, r.x - 1.5, r.y - 1.5, r.w + 3, r.h + 3, rad + 1.5); ctx.stroke();
      ctx.restore();
    },
    // choose fractions by shape: wide (landscape) or tall (portrait) content
    pick: (c, wide, tall) => (c.w / c.h >= 1.05 ? wide : tall),
  };

  // ------------------------------------------------------------------ the view
  // the kind of view a level is (a memory cell, a logic gate, a block…), for Merge; null when it does not say
  const formOf = (n) => (n && (n.form || null)) || null;
  const formsOf = (f, n) => (n === 1 || f === 'code' ? f : f.startsWith('piece of') ? 'pieces' + f.slice(5) : f + 's');

  class ArchView {
    /* stage: positioned container; opts: { build(size) → root node, onFocus(node, path, isLevel), label,
       outer: { title, exit() } when the root itself was opened from another view (backing out of it returns) } */
    constructor(stage, opts) {
      this.stage = stage; this.opts = opts; this.dirty = true;
      this.path = []; this.anim = null; this.selected = null;
      this.canvas = el('canvas', { class: 'arch-canvas', tabindex: '0', 'aria-label': opts.label || 'Architecture diagram: click a part for details; double-click it to open it' });
      this.back = el('button', { class: 'icon-btn arch-back', type: 'button', 'aria-label': 'Back out one level', title: 'Back out one level (Esc)', onclick: () => this.exit() }, '←');
      this.crumbs = el('nav', { class: 'arch-crumbs', 'aria-label': 'Where you are in the model' });
      this.merge = false; this.mergeDepth = 0; this.mergeForm = null; this.mergeHits = null;
      this.mergeBtn = el('button', { class: 'arch-merge', type: 'button', 'aria-pressed': 'false', title: 'Merge: when you zoom out, keep this level of detail. The parts around this one are drawn the same way, side by side, each with a thin line around it and its name.', onclick: () => this.setMerge(!this.merge) }, el('span', { class: 'arch-merge-dot', 'aria-hidden': 'true' }), 'Merge');
      this.bar = el('div', { class: 'arch-bar' }, this.back, this.crumbs, this.mergeBtn);
      // the open level's own controls (flip a bit, step the clock…) at the right of its title, and the card of a
      // guided walk through the levels
      this.toolsEl = el('div', { class: 'arch-tools', role: 'toolbar', 'aria-label': 'Tools for this level', hidden: true });
      this.followEl = el('section', { class: 'arch-follow', 'aria-label': 'Following one number', 'aria-live': 'polite', hidden: true });
      this.hl = null; this.links = []; this.clock = null; this.queue = null; this.tour = null; this.flipMode = false;
      stage.append(this.canvas, this.bar, this.toolsEl, this.followEl);
      this.zoom = new NA.ui.Zoom(this.canvas, {
        host: stage, panOne: true, max: 3, noDouble: true, wheel: 'always', wheelGate: () => this.wheelOpen(),
        onChange: () => { this.dirty = true; NA.tooltip.hide(); if (!this.locked() && this.zoom.k > 2.3) this.zoomedIn(); },
        onUnderflow: () => { if (!this.locked()) this.exit(); },
      });
      this.canvas.addEventListener('click', (e) => this.click(e));
      this.canvas.addEventListener('pointermove', (e) => this.hover(e));
      const release = () => { if (this.zoom.ptrs.size === 0) this.gestureLock = false; };
      this.canvas.addEventListener('pointerup', release); this.canvas.addEventListener('pointercancel', release);
      this.canvas.addEventListener('pointerleave', () => { this.hoverId = null; this.dirty = true; NA.tooltip.hide(); this.syncLinks(); });
      this.canvas.addEventListener('keydown', (e) => {
        if (e.key === 'Escape' || e.key === 'Backspace') { e.preventDefault(); this.exit(); }
        else if (e.key === 'Enter' && this.selected) { e.preventDefault(); this.openSelected(); }
      });
      this.ro = new ResizeObserver(() => { this.dirty = true; });
      this.ro.observe(stage);
    }
    destroy() { this.ro.disconnect(); this.anim = null; }
    size() { return [Math.max(160, this.stage.clientWidth), Math.max(160, this.stage.clientHeight)]; }
    // the open level's box: under the bar, and above a walk's card while one is going on
    levelRect() { const [W, H] = this.size(), f = this.tour ? this.followH || 0 : 0; return { x: 10, y: 46, w: W - 20, h: H - 56 - f >= 220 ? H - 56 - f : H - 56 }; } // (a short stage: the card goes over it)
    content(node, r) { const hy = node.header ?? (this.hy || 0.07); return { x: r.x + r.w * 0.02, y: r.y + r.h * hy, w: r.w * 0.96, h: r.h * (1 - hy - 0.02) }; }
    node() { let n = this.root; for (const id of this.path.slice(1)) n = this.childOf(n, id) || n; return n; }
    /* A part of a level by id: a listed child, or a cell of one of its grids, made on demand: "grid:i:j", or
       "grid:k" for a grid with index set, which numbers its cells row by row so that a cell keeps its id when
       the grid's layout follows the stage size (tiles of channels). */
    childOf(parent, id) {
      if (!parent) return null;
      const ch = (parent.children || []).find((c) => c.id === id);
      if (ch || !parent.grids) return ch || null;
      let m = /^(.+):(\d+):(\d+)$/.exec(id), g = m && parent.grids.find((q) => q.id === m[1] && !q.index), i, j;
      if (g) { i = +m[2]; j = +m[3]; }
      else if ((m = /^(.+):(\d+)$/.exec(id)) && (g = parent.grids.find((q) => q.id === m[1] && q.index))) { i = Math.floor(+m[2] / g.cols); j = +m[2] % g.cols; }
      else return null;
      return i < g.rows && j < g.cols ? this.cellNode(parent, g, i, j) : null;
    }
    cellNode(parent, g, i, j) {
      const node = g.make(i, j);
      if (!node) return null;
      const c = this.content(parent, this.levelRect()), gr = S.place(c, g.rect), { cw, ch } = this.cellSize(g, gr);
      return Object.assign(node, { id: g.index ? `${g.id}:${i * g.cols + j}` : `${g.id}:${i}:${j}`, rect: [(gr.x + j * cw - c.x) / c.w, (gr.y + i * ch - c.y) / c.h, cw / c.w, ch / c.h] });
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
      else if (now - (this.lastMotion || 0) > 32 && (this.levelAnimates() || this.mergeAnim)) { this.lastMotion = now; this.dirty = true; }
      if (!this.dirty || !this.canvas.offsetParent) return;
      this.dirty = false;
      this.draw();
    }
    draw() {
      this.followH = this.tour ? this.followEl.offsetHeight + 8 : 0;
      const [W, H] = this.size(), ctx = NA.fitCanvas(this.canvas, W, H), R = this.levelRect();
      ctx.clearRect(0, 0, W, H);
      S.now = performance.now(); S.hl = this.hl; S.flipMode = this.flipMode;
      this.mergeAnim = false;
      this.hy = Math.max(0.07, (30 + (this.anim ? 0 : this.toolsH || 0)) / R.h); // room for the level title (and tools stacked under it)
      const chip = this.zoom.chip;
      this.chipW = chip && chip.offsetParent ? chip.offsetWidth + 14 : 0;
      this.root = this.opts.build({ w: R.w * 0.96, h: R.h * 0.91 });
      if (!this.path.length) this.path = [this.root.id];
      // parts can disappear when the model changes shape: stay at the deepest level that still exists
      let n = this.root, keep = 1;
      const titles = [n.title]; // (the breadcrumbs follow a title that changes under the same id: another model)
      for (const id of this.path.slice(1)) { const c = this.childOf(n, id); if (!c) break; n = c; keep++; titles.push(c.title); }
      if (keep < this.path.length) { this.path = this.path.slice(0, keep); this.selected = null; this.focusKey = null; }
      this.pathTitles = titles.join('/');
      if (this.focusLater) { this.focusLater = false; this.notify(); }
      this.W = W; this.H = H;
      const a = this.anim;
      if (!a) { const node = this.node(); this.drawLevel(ctx, node, R, this.camNow(), 1, false, null, 0, this.mergeK(this.path.length)); this.drawHoverCell(ctx, node, R); }
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
        this.drawLevel(ctx, parent, R, cam, parentAlpha, false, child.id, childAlpha, this.mergeK(this.path.length));
        if (childAlpha > 0) {
          // where the part's box is on screen right now, and the child level scaled to fit inside it
          const box = this.camRect({ x: a.w1[0] - a.bw / 2, y: a.w1[1] - a.bh / 2, w: a.bw, h: a.bh }, cam);
          const sc = Math.min(1, Math.min(box.w / R.w, box.h / R.h)), ox = box.x + box.w / 2 - (R.x + R.w / 2) * sc, oy = box.y + box.h / 2 - (R.y + R.h / 2) * sc;
          ctx.save();
          ctx.translate(ox, oy); ctx.scale(sc, sc);
          this.drawLevel(ctx, child, R, { k: 1, tx: 0, ty: 0 }, childAlpha, true, null, 0, this.mergeK(this.path.length + 1));
          ctx.restore();
        }
      }
      ctx.globalAlpha = 1;
      this.syncBar();
      this.syncTools();
    }
    // fadeId: the part being entered, whose small preview fades as its own level grows inside it
    drawLevel(ctx, node, R, cam, alpha, cover, fadeId, fade = 0, k = 0) {
      const r = this.camRect(R, cam);
      ctx.globalAlpha = alpha;
      // a level drawn over another covers it, as a panel with soft corners growing out of its box
      if (cover) { ctx.fillStyle = theme.c('panel'); ctx.beginPath(); NA.rr(ctx, r.x - 6, r.y - 6, r.w + 12, r.h + 12, 12); ctx.fill(); }
      const c = this.content(node, r), hits = k > 0 ? [] : null, merged = k > 0 ? this.drawMerged(ctx, node, c, k, alpha, hits) : null;
      if (!cover) { this.mergeHits = merged ? hits : null; this.mergeCut = merged && merged.cut ? merged.count : 0; }
      // the zoom buttons sit at the top right, and the open level's tools left of them, so its title stops short
      const open = !cover && !this.anim;
      S.text(ctx, node.title + (node.sub ? '  ·  ' + node.sub : '') + (merged ? `  ·  merged: ${merged.count}${merged.cut ? '+' : ''} ${this.mergeForm ? formsOf(this.mergeForm, merged.count) : merged.count === 1 ? 'part' : 'parts'}` : ''), r.x + 2, r.y + 14, { size: clamp(14 * cam.k, 9, 40), w: 650, maxW: cam.k > 1.001 ? r.w : Math.max(60, r.w - (this.chipW || 0) - (open ? this.toolsW || 0 : 0)) });
      if (merged) return;
      // a level with a clock (an adder you step through) draws the moment it is set to
      S.clock = open && this.clock && this.clock.key === this.path.join('/') ? this.clock : null;
      if (node.inner) { ctx.save(); ctx.globalAlpha = alpha; node.inner(ctx, c, S); ctx.restore(); }
      S.clock = null;
      for (const ch of node.children || []) {
        const r1 = S.place(c, ch.rect), lit = !!this.hl && this.isLit(ch.id, node);
        this.drawClosed(ctx, ch, r1, ch.id === fadeId ? alpha * (1 - fade) : alpha, lit);
        if (lit) S.lit(ctx, r1, ch.shape === 'op' ? Math.min(r1.w, r1.h) / 2 : 8);
      }
      if (this.hl) this.drawLitCells(ctx, node, c);
    }
    // is a part of this level the one the explanation points at (by its id, or one of a phrase's parts)?
    isLit(id, node) {
      const h = this.hl;
      if (!h) return false;
      if (h === id) return true;
      const t = (node.terms || []).find((q) => q.key === h);
      return !!t && !t.own && this.termHas(t, id);
    }
    termHas(t, id) { return !!t && ((!!t.ids && t.ids.includes(id)) || (!!t.cells && t.cells(id))); }
    // grid cells lit by the explanation: one outline for each run of them along a row
    drawLitCells(ctx, node, c) {
      for (const g of node.grids || []) {
        if (g.rows * g.cols > 4000) continue;
        const gr = S.place(c, g.rect), { cw, ch } = this.cellSize(g, gr);
        for (let i = 0; i < g.rows; i++) {
          let run = -1;
          for (let j = 0; j <= g.cols; j++) {
            const on = j < g.cols && this.isLit(g.index ? `${g.id}:${i * g.cols + j}` : `${g.id}:${i}:${j}`, node);
            if (on && run < 0) run = j;
            else if (!on && run >= 0) { S.lit(ctx, { x: gr.x + run * cw, y: gr.y + i * ch, w: (j - run) * cw, h: ch }, 3); run = -1; }
          }
        }
      }
    }

    /* Merge: zoom out without leaving the kind of view you were in. Turned on at a level, the view remembers that
       level's form (a memory cell, a logic gate, a weighted sum, a block…; levels without one count by depth) and
       its depth. Every level above it is then drawn as all the parts around of that same form, found down to that
       depth: grouped by the part of this level they belong to, each group framed and named, each part drawn the
       way its own level looks, with a thin line around it and its name. A click opens that part directly. */
    setMerge(on) {
      this.merge = !!on;
      this.mergeDepth = this.path.length;
      this.mergeForm = this.root ? formOf(this.node()) : null;
      this.mergeBtn.setAttribute('aria-pressed', String(this.merge));
      this.dirty = true; this.focusKey = null; this.notify();
      if (NA.ui && NA.ui.toast) NA.ui.toast.show(this.merge ? `Merge on: zoom out, and everything around this ${this.mergeForm || 'part'} is shown the same way.` : 'Merge off: zooming out shows each level as it usually is.');
    }
    mergeK(depth) { return this.merge ? Math.max(0, (this.mergeDepth || 0) - depth) : 0; }
    // a level's parts as nodes with their ids: its children, then its grids' cells (made on demand)
    partNodes(node, which, limit = 400) {
      const out = [];
      if (which !== 'cells') for (const ch of node.children || []) { if (ch.inner || ch.children || ch.grids || ch.shape !== 'op') out.push(ch); if (out.length >= limit) return out; }
      if (which !== 'kids') for (const g of node.grids || []) for (const n of this.eachCell(g)) { out.push(n); if (out.length >= limit) return out; }
      return out;
    }
    *eachCell(g) {
      for (let i = 0; i < g.rows; i++) for (let j = 0; j < g.cols; j++) {
        if (g.skip && g.skip(i, j)) continue;
        let n = null;
        try { n = g.make(i, j); } catch (e) { n = null; }
        if (n) { n.id = g.index ? `${g.id}:${i * g.cols + j}` : `${g.id}:${i}:${j}`; n._grid = g; yield n; }
      }
    }
    // the parts of the remembered form within k levels below node, grouped by the part of node they sit in; each
    // keeps the part it sits directly in (up). The parts of node with none of them come last, as Other parts
    mergedGroups(node, k) {
      const F = this.mergeForm, budget = { visit: 8000, found: 0, part: 0, partVisit: 0, cut: false };
      const want = (n, d) => (F ? formOf(n) === F : d === k);
      const collect = (n, d, rel, out, up, trail) => {
        if (budget.visit-- <= 0 || budget.partVisit-- <= 0 || budget.found <= 0 || budget.part <= 0) return;
        if (want(n, d)) { out.push({ node: n, rel, up, trail }); budget.found--; budget.part--; return; }
        if (d >= k) return;
        for (const q of this.partNodes(n, 'kids', 300)) collect(q, d + 1, rel.concat(q.id), out, n, trail.concat(q.title));
        // a grid's cells are made alike: when the first few hold none of the form, the rest are not searched (a
        // 24 × 24 grid of adders would otherwise use up the search before the parts after it are looked at)
        for (const g of n.grids || []) {
          let dry = 0, seen = 0;
          for (const q of this.eachCell(g)) {
            if (++seen > 300 || budget.visit <= 0 || budget.partVisit <= 0 || budget.found <= 0 || budget.part <= 0) break;
            const before = out.length;
            collect(q, d + 1, rel.concat(q.id), out, n, trail.concat(q.title));
            if (out.length > before) dry = -Infinity; else if (++dry >= 4) break;
          }
        }
      };
      // the column (child) a grid sits in, for a picture of columns: its title names the group
      const colOf = (g) => { const x = g.rect[0] + g.rect[2] / 2, y = g.rect[1] + g.rect[3] / 2; return (node.children || []).find((ch) => ch.rect && x >= ch.rect[0] && x <= ch.rect[0] + ch.rect[2] && y >= ch.rect[1] && y <= ch.rect[1] + ch.rect[3]); };
      const gather = (which) => {
        const flat = { title: null, items: [] }, byGrid = new Map(), groups = [], rest = [], noRoom = [], parts = this.partNodes(node, which);
        const visits = Math.max(300, Math.floor(4000 / Math.max(1, parts.length)));
        const stopped = () => budget.part <= 0 || budget.partVisit <= 0 || budget.visit <= 0 || budget.found <= 0;
        // first which parts hold any of the form at all (true, false, or null: not found before the search ran out),
        // then those parts share the room evenly, so one big part cannot crowd out the others
        budget.found = Infinity;
        const holds = parts.map((p) => { const t = []; budget.part = 1; budget.partVisit = visits; collect(p, 1, [p.id], t, node, [p.title]); return t.length ? true : stopped() ? null : false; });
        const n = holds.filter((h) => h).length, share = Math.max(32, Math.floor(520 / Math.max(1, n)));
        budget.found = n <= 24 ? share * n : 520;
        parts.forEach((p, pi) => {
          const items = [];
          if (holds[pi]) { budget.part = share; budget.partVisit = visits; collect(p, 1, [p.id], items, node, [p.title]); }
          // a search that stopped short: there may be more than were found. The parts of the last number (gate…) it
          // reached are dropped, so each one shown is whole
          const short = holds[pi] === null || (holds[pi] && stopped());
          if (short) budget.cut = true;
          if (short && items.length > 1) {
            const up = (it) => it.rel.slice(0, -1).join('/'), last = up(items[items.length - 1]);
            let m = items.length;
            while (m > 0 && up(items[m - 1]) === last) m--;
            if (m > 0) items.length = m;
          }
          // a part with none of that form close enough keeps its usual box; so do a few there was no room for
          if (!items.length) { if (p.shape !== 'ghost') (holds[pi] ? noRoom : rest).push({ node: p, rel: [p.id], up: node, trail: [p.title], closed: true }); return; }
          if (items.length === 1 && items[0].node === p) {
            if (node.gridsFirst && p._grid) { let gg = byGrid.get(p._grid); if (!gg) { const col = colOf(p._grid); gg = { title: col ? col.title : null, items: [] }; byGrid.set(p._grid, gg); groups.push(gg); } gg.items.push(items[0]); }
            else flat.items.push(items[0]);
          } else groups.push({ title: p.title, items });
        });
        const out = flat.items.length ? [flat, ...groups] : groups;
        // the parts with none of that form close enough keep their usual boxes, so the level is all there
        if (noRoom.length <= 12) rest.push(...noRoom);
        if (out.length && rest.length) out.push({ title: 'Other parts', items: rest.slice(0, 60), rest: true });
        return out;
      };
      // a level drawn as one picture has its nodes in grids inside its columns: those first, so nothing comes twice
      let out = gather(node.gridsFirst ? 'cells' : 'all');
      if (!out.length && node.gridsFirst) out = gather('kids');
      out.cut = budget.cut;
      out.count = out.reduce((sum, g) => sum + (g.rest ? 0 : g.items.length), 0);
      return out;
    }
    drawMerged(ctx, node, c, k, alpha, hits) {
      const groups = this.mergedGroups(node, k);
      if (!groups.length) return null;
      const rest = groups[groups.length - 1].rest ? groups[groups.length - 1] : null;
      if (rest && groups.length > 1) {
        const sh = clamp(c.h * 0.2, 46, 96), main = { x: c.x, y: c.y, w: c.w, h: c.h - sh - 8 };
        if (!this.drawMemory(ctx, groups.slice(0, -1), main, alpha, hits)) this.layGroups(ctx, groups.slice(0, -1), main, alpha, hits, 0, null);
        this.layGroups(ctx, [rest], { x: c.x, y: c.y + c.h - sh, w: c.w, h: sh }, alpha, hits, 0, null);
      } else if (!this.drawMemory(ctx, groups, c, alpha, hits)) this.layGroups(ctx, groups, c, alpha, hits, 0, null);
      return groups;
    }
    /* Merged memory, wired up the way a memory array is: each number on its own word line (a row, opened all at
       once), each bit position on its own bit line (a column) running down to a sense amplifier, and every cell
       showing the bit it holds. Pointing at a row (or a cell) opens that row: its word line lights, and so do the
       bit lines its 1s pull up. False: not memory. */
    drawMemory(ctx, groups, c, alpha, hits) {
      const F = this.mergeForm;
      if (F !== 'memory cell' && F !== 'stored number') return false;
      const rows = [];
      for (const g of groups) {
        if (g.rest) continue;
        if (F === 'stored number') { for (const it of g.items) rows.push({ title: it.node.title, group: g.title, rel: it.rel, node: it.node, cells: null }); continue; }
        const by = new Map();
        for (const it of g.items) {
          const key = it.rel.slice(0, -1).join('/');
          let row = by.get(key);
          if (!row) by.set(key, (row = { title: (it.up && it.up.title) || g.title || 'a number', group: g.title, rel: it.rel.slice(0, -1), node: it.up, cells: [] }));
          row.cells.push(it);
        }
        rows.push(...by.values());
      }
      if (!rows.length) return false;
      const bitsOf = (row) => (row.node && row.node.grids && row.node.grids[0] ? row.node.grids[0].cols : row.cells.length);
      const ncol = Math.max(...rows.map(bitsOf)), u = clamp(Math.min(c.w / 62, c.h / 20), 8, 12);
      const labelW = clamp(c.w * 0.17, 60, 160), decW = 10, top = u * 2.4, senseH = clamp(c.h * 0.09, 18, 30), foot = u * 1.5;
      const ax = c.x + labelW + decW + 10, aw = c.x + c.w - ax - 2, cw = aw / ncol;
      const room = c.h - top - senseH - foot, shown = rows.slice(0, Math.max(1, Math.floor(room / 12)));
      const rh = Math.min(room / shown.length, Math.max(cw * 2.4, 24), 46), ay = c.y + top, ah = rh * shown.length, sy = ay + ah + 3;
      const colX = (p) => ax + (ncol - 1 - p + 0.5) * cw, wireX = (p) => colX(p) - cw * 0.3;
      const jOf = (id) => +String(id).split(':')[2];
      // each row's cells, in bit order: the merged ones, or (for stored numbers) the cells of the number's bits
      for (const row of shown) {
        if (!row.cells) row.cells = row.node && row.node.grids && row.node.grids[0] ? [...this.eachCell(row.node.grids[0])].map((cell) => ({ node: cell, rel: row.rel.concat(cell.id) })) : [];
        row.n = bitsOf(row);
      }
      // the row being read: the one under the pointer, or the selected one
      const rowOf = (key) => (key && key[0] === '~' ? shown.findIndex((row) => key === '~' + row.rel.join('/') || key.startsWith('~' + row.rel.join('/') + '/')) : -1);
      const lit = rowOf(this.hoverId) >= 0 ? rowOf(this.hoverId) : rowOf(this.selected);
      const ones = new Set(lit >= 0 ? shown[lit].cells.filter((it) => it.node.bit).map((it) => shown[lit].n - 1 - jOf(it.node.id)) : []);
      ctx.save(); ctx.globalAlpha = alpha;
      // bit lines, one per bit position, down every row to its sense amplifier
      for (let p = 0; p < ncol; p++) {
        const on = ones.has(p);
        ctx.strokeStyle = on ? theme.c('pos') : theme.c('line-2'); ctx.lineWidth = on ? 1.8 : 1;
        ctx.beginPath(); ctx.moveTo(wireX(p), ay - 2); ctx.lineTo(wireX(p), sy + 2); ctx.stroke();
      }
      // bit numbers over the columns, and what the rows and columns are
      for (let p = ncol - 1; p >= 0; p--) if (p === ncol - 1 || p % 8 === 0 || (cw > 16 && p % 4 === 0)) S.text(ctx, String(p), colX(p), ay - u * 0.75, { size: u * 0.85, mono: true, align: 'center', color: theme.c('muted') });
      S.text(ctx, 'bit lines ↓', ax + aw, ay - u * 1.85, { size: u * 0.82, align: 'right', color: theme.c('muted') });
      S.text(ctx, 'word lines →', c.x, ay - u * 0.75, { size: u * 0.82, color: theme.c('muted'), maxW: labelW });
      // the row decoder: it picks the row to open
      const dx = c.x + labelW + 4;
      ctx.fillStyle = theme.c('panel-2'); ctx.fillRect(dx, ay, decW, ah); ctx.strokeStyle = theme.c('line-2'); ctx.lineWidth = 1; ctx.strokeRect(dx + 0.5, ay + 0.5, decW - 1, ah - 1);
      shown.forEach((row, ri) => {
        const y = ay + ri * rh, wy = y + rh * 0.26, open = ri === lit, rkey = '~' + row.rel.join('/');
        const ls = clamp(Math.min(u, rh * 0.55), 7.5, 12);
        S.text(ctx, row.title, c.x + labelW - 2, wy + 1, { size: ls, w: open ? 700 : 600, align: 'right', maxW: labelW - 4, color: theme.c(open ? 'ink' : 'ink-2') });
        if (row.group && row.group !== row.title && rh > ls * 2.3) S.text(ctx, `in ${row.group}`, c.x + labelW - 2, wy + ls * 1.15, { size: ls * 0.82, align: 'right', maxW: labelW - 4, color: theme.c('muted') });
        // the word line: all of this row's cells open at once
        ctx.strokeStyle = open ? theme.c('focus') : theme.c('ink-2', 0.55); ctx.lineWidth = open ? 2.2 : 1.1;
        ctx.beginPath(); ctx.moveTo(dx + decW, wy); ctx.lineTo(ax + aw, wy); ctx.stroke();
        for (const it of row.cells) {
          const j = jOf(it.node.id), p = row.n - 1 - j, x = colX(p), bit = it.node.bit ? 1 : 0;
          const field = /\((sign|exponent|mantissa)\)/.exec(it.node.title || ''), tone = !field ? 's3' : field[1] === 'sign' ? 's7' : field[1] === 'exponent' ? 's4' : 's1';
          const s = Math.max(3, Math.min(cw * 0.56, rh * 0.56)), bx = x - s * 0.15, by = y + rh * 0.42;
          // the access transistor: from the bit line to the capacitor, its gate on the word line
          if (cw >= 7) {
            ctx.strokeStyle = theme.c('ink-2', 0.7); ctx.lineWidth = 1;
            ctx.beginPath(); ctx.moveTo(wireX(p), by + s / 2); ctx.lineTo(bx, by + s / 2); ctx.moveTo((wireX(p) + bx) / 2, by + s / 2 - 2); ctx.lineTo((wireX(p) + bx) / 2, wy); ctx.stroke();
          }
          // the capacitor: charged for a 1
          ctx.fillStyle = bit ? theme.c(tone, 0.85) : theme.c(tone, 0.1);
          ctx.beginPath(); NA.rr(ctx, bx, by, s, s, Math.min(3, s * 0.2)); ctx.fill();
          ctx.strokeStyle = theme.c(tone, 0.7); ctx.lineWidth = 1; ctx.stroke();
          if (s >= 9) S.text(ctx, String(bit), bx + s / 2, by + s / 2 + 0.5, { size: s * 0.66, mono: true, w: 700, align: 'center', color: bit ? theme.c('panel') : theme.c('muted') });
          const r = { x: x - cw / 2, y, w: cw, h: rh }, key = '~' + it.rel.join('/');
          if (this.hoverId === key || this.selected === key) { ctx.strokeStyle = theme.c('focus'); ctx.lineWidth = this.selected === key ? 2.5 : 1.8; ctx.strokeRect(r.x + 0.5, r.y + 0.5, r.w - 1, r.h - 1); }
          if (hits) hits.push({ r, rel: it.rel, node: it.node, group: row.title });
        }
        const lr = { x: c.x, y, w: labelW + decW + 4, h: rh };
        if (this.selected === rkey) { ctx.strokeStyle = theme.c('focus'); ctx.lineWidth = 2; ctx.strokeRect(lr.x + 0.5, lr.y + 0.5, lr.w - 1, lr.h - 1); }
        if (hits && row.node) hits.push({ r: lr, rel: row.rel, node: row.node, group: row.group });
      });
      // a sense amplifier at the foot of every bit line
      const tw = Math.min(cw * 0.6, 10), th = senseH * 0.42;
      for (let p = 0; p < ncol; p++) {
        const x = wireX(p), on = ones.has(p);
        ctx.fillStyle = on ? theme.c('pos', 0.85) : theme.c('panel'); ctx.strokeStyle = on ? theme.c('pos') : theme.c('ink-2', 0.6); ctx.lineWidth = 1;
        ctx.beginPath(); ctx.moveTo(x - tw / 2, sy); ctx.lineTo(x + tw / 2, sy); ctx.lineTo(x, sy + th); ctx.closePath(); ctx.fill(); ctx.stroke();
      }
      S.text(ctx, lit >= 0 ? `reading ${shown[lit].title}: its row is open, and its 1s show on the sense amplifiers` : 'sense amplifiers, one per bit line: point at a row to read it', ax, sy + th + u * 0.9, { size: u * 0.85, color: theme.c(lit >= 0 ? 'ink-2' : 'muted'), maxW: aw });
      if (rows.length > shown.length) S.text(ctx, `… and ${rows.length - shown.length} more`, c.x, sy + th + u * 0.9, { size: u * 0.85, color: theme.c('muted'), maxW: labelW });
      ctx.restore();
      return true;
    }
    // a group's parts split by the part they sit in, at the first level down where they differ (one multiplication
    // of several, one gate of four…): framed again inside the group, numbered where several have the same name
    // (the number first, so a name cut short still says which one)
    splitGroup(g) {
      if (g.rest || g.title === null) return null;
      const deepest = Math.max(...g.items.map((it) => it.rel.length));
      for (let L = 2; L < deepest; L++) {
        const by = new Map();
        for (const it of g.items) {
          const at = Math.min(L, it.rel.length - 1), key = it.rel.slice(0, at).join('/');
          let sub = by.get(key);
          if (!sub) by.set(key, (sub = { title: (it.trail && it.trail[at - 1]) || 'part', items: [] }));
          sub.items.push(it);
        }
        if (by.size < 2) continue;
        const subs = [...by.values()], same = new Map(), seen = new Map();
        for (const sub of subs) same.set(sub.title, (same.get(sub.title) || 0) + 1);
        for (const sub of subs) if (same.get(sub.title) > 1) { const n = (seen.get(sub.title) || 0) + 1; seen.set(sub.title, n); sub.title = n + ' · ' + sub.title; }
        return subs;
      }
      return null;
    }
    // groups in a grid filling c, each framed and named; a group's parts in a grid of tiles about 1.4 times wider
    // than tall. A group whose parts sit in different parts of it is framed once more inside. False: no room for that
    layGroups(ctx, groups, c, alpha, hits, depth, outer) {
      const G = groups.length, gap = depth ? 5 : 8, framed = depth > 0 || G > 1 || groups.some((g) => g.title !== null);
      const fw = framed ? (depth ? 8 : 10) : 0, fh = framed ? (depth ? 20 : 26) : 0;
      // the number of group columns that gives the parts the largest tiles (judged by their smaller side)
      const tilesFor = (w, h, m) => { const ic = Math.max(1, Math.min(m, Math.round(Math.sqrt((m * w) / (h * 1.4))))), ir = Math.ceil(m / ic); return Math.min(w / ic / 1.4, h / ir); };
      let gc = 0, best = -1;
      for (let q = 1; q <= G; q++) {
        const rows = Math.ceil(G / q), w = (c.w - gap * (q - 1)) / q, h = (c.h - gap * (rows - 1)) / rows;
        if (depth && (w < 54 || h < 40)) continue;
        const score = Math.min(...groups.map((g) => tilesFor(w - fw, h - fh, g.items.length)));
        if (score > best) { best = score; gc = q; }
      }
      if (!gc) { if (depth) return false; gc = 1; }
      const gr = Math.ceil(G / gc), gw = (c.w - gap * (gc - 1)) / gc, gh = (c.h - gap * (gr - 1)) / gr;
      groups.forEach((g, gi) => {
        const x = c.x + (gi % gc) * (gw + gap), y = c.y + Math.floor(gi / gc) * (gh + gap), name = outer ? `${outer} › ${g.title}` : g.title;
        let body = { x, y, w: gw, h: gh };
        if (framed) {
          const band = depth ? clamp(gh * 0.08, 10, 14) : clamp(gh * 0.09, 12, 18), pad = depth ? 4 : 5;
          ctx.globalAlpha = alpha;
          ctx.strokeStyle = theme.c('ink-2', depth ? 0.3 : 0.45); ctx.lineWidth = depth ? 1 : 1.2;
          ctx.beginPath(); NA.rr(ctx, x + 0.5, y + 0.5, gw - 1, gh - 1, depth ? 6 : 8); ctx.stroke();
          S.text(ctx, g.title || 'these parts', x + pad + 3, y + band / 2 + 3, { size: Math.min(depth ? 10.5 : 12, band * 0.72), w: depth ? 600 : 650, color: theme.c(depth ? 'muted' : 'ink-2'), maxW: gw - 2 * pad - 6 });
          body = { x: x + pad, y: y + band + pad - 1, w: gw - 2 * pad, h: gh - band - 2 * pad + 1 };
        }
        const subs = depth ? null : this.splitGroup(g);
        if (subs && this.layGroups(ctx, subs, body, alpha, hits, 1, g.title)) return;
        // parts wired together in one circuit: drawn in that circuit's own picture, with its wires
        if (this.drawAttached(ctx, g, body, alpha, hits, name)) return;
        // the full adders of one adder: in bit order, each passing its carry to the next (a ripple)
        const chain = this.mergeForm === 'full adder' && g.items.length > 1 && g.items.every((it) => it.up === g.items[0].up && it.node.io && /^fa:/.test(it.node.id));
        const m = g.items.length, ic = Math.max(1, Math.min(m, Math.round(Math.sqrt((m * body.w) / (body.h * 1.4))))), ir = Math.ceil(m / ic), tg = chain ? clamp(body.w * 0.03, 10, 18) : Math.min(4, body.w * 0.01);
        const tw = (body.w - tg * (ic - 1)) / ic, th = (body.h - tg * (ir - 1)) / ir, placed = [];
        g.items.forEach((it, q) => {
          const r = { x: body.x + (q % ic) * (tw + tg), y: body.y + Math.floor(q / ic) * (th + tg), w: tw, h: th };
          this.drawItem(ctx, it.node, r, alpha, it.closed, '~' + it.rel.join('/'));
          placed.push(r);
          if (hits) hits.push({ r, rel: it.rel, node: it.node, group: name });
        });
        if (chain) this.drawCarries(ctx, g.items, placed, tg, alpha);
      });
      return true;
    }
    // the carry of each full adder in a chain, into the next one up (left, or at the end of the row above): lit for a 1
    drawCarries(ctx, items, rs, tg, alpha) {
      ctx.save(); ctx.globalAlpha = alpha;
      for (let q = items.length - 1; q > 0; q--) {
        const from = rs[q], to = rs[q - 1], on = !!items[q].node.io.cout, y0 = from.y + from.h * 0.5, y1 = to.y + to.h * 0.5;
        ctx.strokeStyle = on ? theme.c('pos') : theme.c('ink-2', 0.5); ctx.fillStyle = ctx.strokeStyle; ctx.lineWidth = on ? 2 : 1.2;
        const pts = Math.abs(from.y - to.y) < 1 ? [[from.x, y0], [to.x + to.w, y1]]
          : [[from.x, y0], [from.x - tg / 2, y0], [from.x - tg / 2, from.y - tg / 2], [to.x + to.w + tg / 2, from.y - tg / 2], [to.x + to.w + tg / 2, y1], [to.x + to.w, y1]];
        ctx.beginPath(); pts.forEach(([x, y], i) => (i ? ctx.lineTo(x, y) : ctx.moveTo(x, y))); ctx.stroke();
        const [ex, ey] = pts[pts.length - 1], h = Math.min(6, tg * 0.45);
        ctx.beginPath(); ctx.moveTo(ex, ey); ctx.lineTo(ex + h, ey - h * 0.6); ctx.lineTo(ex + h, ey + h * 0.6); ctx.closePath(); ctx.fill();
      }
      ctx.restore();
    }
    /* Parts of a merged level that sit in one circuit (the gates of a full adder or of a gate, the transistors of a
       gate) are drawn in that circuit's own picture, its wires joining them, each opened in place. False: they do
       not (or there is no room), so they are drawn as tiles. */
    drawAttached(ctx, g, body, alpha, hits, name) {
      const up = g.items[0] && g.items[0].up, F = this.mergeForm;
      if (!up || !up.inner || (F !== 'logic gate' && F !== 'transistor') || g.items.length < 2 || g.items.some((it) => it.up !== up || it.closed)) return false;
      const kids = up.children || [], ids = new Map(g.items.map((it) => [it.node.id, it]));
      if (!g.items.every((it) => kids.some((ch) => ch.id === it.node.id))) return false;
      const L = this.levelRect(), vw = L.w * 0.96, vh = L.h * 0.86, sc = Math.min(body.w / vw, body.h / vh);
      if (sc < 0.22) return false;
      const ox = body.x + (body.w - vw * sc) / 2, oy = body.y + (body.h - vh * sc) / 2, cv = { x: 0, y: 0, w: vw, h: vh }, placed = [];
      if (up.animate) this.mergeAnim = true;
      ctx.save(); ctx.globalAlpha = alpha; ctx.translate(ox, oy); ctx.scale(sc, sc);
      const W0 = this.W, H0 = this.H; this.W = Infinity; this.H = Infinity; // drawn in its own space: nothing is off screen there
      ctx.save(); up.inner(ctx, cv, S); ctx.restore();
      for (const ch of kids) {
        const r = S.place(cv, ch.rect), it = ids.get(ch.id);
        if (it) { this.drawItem(ctx, it.node, r, alpha, false, '~' + it.rel.join('/'), sc); placed.push([it, r]); }
        else this.drawClosed(ctx, ch, r, alpha);
      }
      this.W = W0; this.H = H0;
      ctx.restore();
      if (hits) for (const [it, r] of placed) hits.push({ r: { x: ox + r.x * sc, y: oy + r.y * sc, w: r.w * sc, h: r.h * sc }, rel: it.rel, node: it.node, group: name });
      return true;
    }
    // one part in a merged level: drawn the way its own level looks, framed and named; too small, or one of the
    // other parts, as its usual box. key: its path ('~' + rel), for the pointer and the selection; scale: how much
    // the space it is drawn in is shrunk on screen (a part drawn inside its circuit's picture)
    drawItem(ctx, n, r, alpha, closed, key, scale = 1) {
      if (r.x > this.W || r.y > this.H || r.x + r.w < 0 || r.y + r.h < 0 || r.w < 2 || r.h < 2) return;
      const hov = !!key && (this.hoverId === key || this.selected === key), sw = r.w * scale, sh = r.h * scale;
      if (closed || sw < 62 || sh < 44 || !(n.inner || n.children)) {
        if (n.summary && sw >= 10 && sh >= 10 && (sw < 62 || sh < 34)) {
          ctx.globalAlpha = alpha;
          ctx.fillStyle = theme.c('panel'); ctx.fillRect(r.x, r.y, r.w, r.h);
          if (n.tone) { ctx.fillStyle = theme.c(n.tone, theme.dark ? 0.12 : 0.08); ctx.fillRect(r.x, r.y, r.w, r.h); }
          ctx.save(); ctx.beginPath(); ctx.rect(r.x, r.y, r.w, r.h); ctx.clip(); n.summary(ctx, { x: r.x + 2, y: r.y + r.h * 0.12, w: r.w - 4, h: r.h * 0.8 }, S); ctx.restore();
          ctx.strokeStyle = hov ? theme.c('focus') : theme.c('line-2'); ctx.lineWidth = hov ? 2 : 1; ctx.strokeRect(r.x + 0.5, r.y + 0.5, r.w - 1, r.h - 1);
          return;
        }
        if (sw < 16 || sh < 12) {
          ctx.globalAlpha = alpha; ctx.fillStyle = n.tone ? theme.c(n.tone, 0.55) : theme.c('line-2'); ctx.fillRect(r.x, r.y, r.w, r.h);
          if (hov) { ctx.strokeStyle = theme.c('focus'); ctx.lineWidth = 2; ctx.strokeRect(r.x - 0.5, r.y - 0.5, r.w + 1, r.h + 1); }
          return;
        }
        if (n.shape === 'zone' || n.shape === 'ghost') {
          ctx.globalAlpha = alpha;
          S.box(ctx, r, { fill: theme.c('panel'), stroke: hov ? theme.c('focus') : theme.c('line-2'), lw: hov ? 2 : 1.1 });
          S.text(ctx, n.title, r.x + r.w / 2, r.y + r.h / 2, { size: clamp(Math.min(r.h * 0.3, r.w * 0.09), 8, 14), w: 650, align: 'center', maxW: r.w - 8 });
          return;
        }
        this.drawClosed(ctx, n, r, alpha, hov); return;
      }
      if (n.animate) this.mergeAnim = true;
      const lab = clamp(sh * 0.11, 9, 13) / scale, box = { x: r.x + 3, y: r.y + lab + 4, w: r.w - 6, h: r.h - lab - 7 }; // (its label stays readable when shrunk)
      const L = this.levelRect(), vw = Math.max(box.w, L.w * 0.5), vh = Math.max(box.h, L.h * 0.5), sc = Math.min(box.w / vw, box.h / vh);
      const c = { x: 0, y: 0, w: vw, h: vh };
      ctx.save();
      ctx.globalAlpha = alpha;
      ctx.beginPath(); ctx.rect(r.x, r.y, r.w, r.h); ctx.clip();
      ctx.fillStyle = theme.c('panel'); ctx.fillRect(r.x, r.y, r.w, r.h);
      if (n.tone) { ctx.fillStyle = theme.c(n.tone, theme.dark ? 0.07 : 0.05); ctx.fillRect(r.x, r.y, r.w, r.h); }
      ctx.translate(box.x + (box.w - vw * sc) / 2, box.y + (box.h - vh * sc) / 2); ctx.scale(sc, sc);
      const W0 = this.W, H0 = this.H; this.W = Infinity; this.H = Infinity; // drawn in its own space: nothing is off screen there
      if (n.inner) { ctx.save(); n.inner(ctx, c, S); ctx.restore(); }
      for (const ch of n.children || []) this.drawClosed(ctx, ch, S.place(c, ch.rect), alpha);
      this.W = W0; this.H = H0;
      ctx.restore();
      ctx.globalAlpha = alpha;
      ctx.strokeStyle = hov ? theme.c('focus') : theme.c('line-2'); ctx.lineWidth = hov ? 2 : 1; ctx.strokeRect(r.x + 0.5, r.y + 0.5, r.w - 1, r.h - 1);
      S.text(ctx, n.title, r.x + 5, r.y + lab / 2 + 3, { size: Math.max(7.5, lab * 0.78), w: 600, color: theme.c('muted'), maxW: r.w - 10 });
    }
    // the grid cell under the cursor gets an outline, so it is clear which number zooming in will open
    // (and the selected one a thicker one)
    drawHoverCell(ctx, node, R) {
      for (const [id, lw] of [[this.hoverId, 2], [this.selected, 3]]) {
        const m = id && node.grids && /^(.+):(\d+):(\d+)$/.exec(id), g = m && node.grids.find((q) => q.id === m[1]);
        if (!g) continue;
        const c = this.content(node, this.camRect(R, this.camNow())), gr = S.place(c, g.rect), { cw, ch } = this.cellSize(g, gr);
        ctx.strokeStyle = theme.c('focus'); ctx.lineWidth = lw;
        ctx.strokeRect(gr.x + +m[3] * cw - 1, gr.y + +m[2] * ch - 1, cw + 2, ch + 2);
      }
    }
    drawClosed(ctx, node, r, alpha, lit) {
      if (r.x > this.W || r.y > this.H || r.x + r.w < 0 || r.y + r.h < 0 || r.w < 2) return;
      const hov = lit || this.hoverId === node.id || this.selected === node.id;
      ctx.globalAlpha = alpha;
      if (node.shape === 'ghost') {
        // part of the parent's drawing, outlined only under the pointer
        if (hov) { ctx.strokeStyle = theme.c('focus', 0.85); ctx.lineWidth = 1.6; ctx.beginPath(); NA.rr(ctx, r.x - 2, r.y - 2, r.w + 4, r.h + 4, 8); ctx.stroke(); ctx.fillStyle = theme.c('focus', 0.05); ctx.fill(); }
        return;
      }
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
      /* Title, subtitle and a small picture, each only where it fits at a readable size: a subtitle under 8.5 px
         or a picture squeezed into a few pixels is dropped rather than drawn over the rest. The title keeps
         clear of the ⤢ that marks a part you can open. */
      const icon = this.canEnter(node) && r.w > 46 && r.h > 30, maxW = r.w - (icon ? 34 : 8);
      const ts = clamp(Math.min(r.h * 0.17, r.w * 0.085), 8, 16), ss = Math.max(8.5, ts * 0.74);
      const subFits = !!node.sub && r.h >= ts * 1.4 + ss * 1.5 + 6;
      const top = r.y + ts * 1.1, below = top + ts * 0.85 + (subFits ? ss * 1.45 : 0), room = r.y + r.h - below - Math.min(8, r.h * 0.08);
      const hasSum = !!node.summary && room >= Math.max(14, ts * 1.3);
      const ty = hasSum ? top : r.y + r.h / 2 - (subFits ? ss * 0.62 : 0);
      // only text in the top band shares a line with the ⤢; text lower down may use the full width
      const wAt = (yy, size) => (icon && yy - size * 0.6 < r.y + 17 ? maxW : r.w - 8);
      // a line slightly too long first shrinks (to 85%), and only then is shortened with an ellipsis
      const shrink = (s, size, w, weight) => { ctx.font = `${weight} ${size}px ${NA.getFont('ui')}`; const tw = ctx.measureText(s).width; return tw > w ? Math.max(size * 0.85, (size * w) / tw) : size; };
      S.text(ctx, node.title, r.x + r.w / 2, ty, { size: shrink(node.title, ts, wAt(ty, ts), 650), w: 650, align: 'center', maxW: wAt(ty, ts) });
      let y = ty + ts * 0.85;
      if (subFits) { const sy = ty + ts * 0.55 + ss * 0.95; S.text(ctx, node.sub, r.x + r.w / 2, sy, { size: shrink(node.sub, ss, wAt(sy, ss), 500), color: theme.c('muted'), align: 'center', maxW: wAt(sy, ss) }); y = ty + ts * 0.85 + ss * 1.45; }
      if (hasSum) { ctx.save(); ctx.globalAlpha = alpha; node.summary(ctx, { x: r.x + r.w * 0.06, y, w: r.w * 0.88, h: r.y + r.h - y - Math.min(8, r.h * 0.08) }, S); ctx.restore(); }
      if (icon) S.text(ctx, '⤢', r.x + r.w - 6, r.y + 9, { size: 10, align: 'right', color: theme.c('muted') });
    }

    // children of the current level under a screen point
    pick(px, py) {
      if (!this.root) return null;
      if (this.mergeHits) { for (const h of this.mergeHits) if (px >= h.r.x && py >= h.r.y && px <= h.r.x + h.r.w && py <= h.r.y + h.r.h) return { node: h.node, r: h.r, rel: h.rel, group: h.group }; return null; }
      const node = this.node(), c = this.content(node, this.camRect(this.levelRect(), this.camNow()));
      const kids = () => {
        for (const ch of node.children || []) {
          const r = S.place(c, ch.rect);
          const pad = ch.shape === 'op' ? 6 : 0;
          if (px >= r.x - pad && py >= r.y - pad && px <= r.x + r.w + pad && py <= r.y + r.h + pad) return { node: ch, r };
        }
        return null;
      };
      const cells = () => {
        for (const g of node.grids || []) {
          const gr = S.place(c, g.rect), { cw, ch } = this.cellSize(g, gr), i = Math.floor((py - gr.y) / ch), j = Math.floor((px - gr.x) / cw);
          if (py < gr.y || px < gr.x || i >= g.rows || j >= g.cols || (g.skip && g.skip(i, j))) continue;
          const cell = this.cellNode(node, g, i, j);
          if (cell) return { node: cell, r: { x: gr.x + j * cw, y: gr.y + i * ch, w: cw, h: ch } };
        }
        return null;
      };
      // a level drawn as one picture (the network view) has its nodes inside its parts: they come first there
      return node.gridsFirst ? cells() || kids() : kids() || cells();
    }
    /* A click selects a part: it is outlined and the panel says what it is (with an Open tool); a second click on
       it straight after (a double-click or double-tap) opens it, as zooming in on it does. A part of a merged
       level is keyed by its path from here ('~' + rel), since many of them share an id. */
    click(e) {
      if (this.zoom.suppressClick || this.anim) return;
      const b = this.canvas.getBoundingClientRect(), x = e.clientX - b.left, y = e.clientY - b.top, f = this.pick(x, y);
      if (!f) { this.lastTap = null; this.select(null); return; }
      // the level can take the click itself (flipping a bit, say)
      const lvl = this.node();
      if (!f.rel && lvl.onPick && lvl.onPick(f.node, this)) { this.dirty = true; NA.tooltip.hide(); this.lastTap = null; return; }
      const key = f.rel ? '~' + f.rel.join('/') : f.node.id, now = performance.now(), t = this.lastTap;
      if (t && t.key === key && now - t.t < 450 && Math.hypot(x - t.x, y - t.y) < 26) {
        this.lastTap = null;
        if (f.rel) this.jump(f.rel); else if (this.canEnter(f.node)) this.enter(f.node);
        return;
      }
      this.lastTap = { key, t: now, x, y };
      this.select(key);
    }
    select(key) { if (this.selected === key) return; this.selected = key; this.dirty = true; this.notify(); this.refreshTools(); }
    // the selected part's node: a part of this level (a child or a grid cell), or a merged part by its path
    selectedNode() {
      const key = this.selected;
      if (!key || !this.root) return null;
      let n = this.node();
      for (const id of key[0] === '~' ? key.slice(1).split('/') : [key]) { n = this.childOf(n, id); if (!n) return null; }
      return n;
    }
    openSelected() {
      const key = this.selected;
      if (!key) return;
      if (key[0] === '~') { this.jump(key.slice(1).split('/')); return; }
      const n = this.childOf(this.node(), key);
      if (n && this.canEnter(n)) this.enter(n);
    }
    hover(e) {
      if (this.zoom.busy || this.anim) return;
      const b = this.canvas.getBoundingClientRect(), f = this.pick(e.clientX - b.left, e.clientY - b.top), id = f ? f.node.id : null;
      const hid = f && f.rel ? '~' + f.rel.join('/') : id;
      if (hid !== this.hoverId) { this.hoverId = hid; this.dirty = true; this.syncLinks(); }
      const lvl = this.node(), hint = f && !f.rel && lvl.pickHint ? lvl.pickHint(f.node, this) : null, opens = !!f && (!!f.rel || this.canEnter(f.node));
      this.canvas.style.cursor = f ? 'pointer' : 'default';
      if (f) NA.tooltip.show([el('div', { class: 'tt-title', text: f.node.title }), f.group ? el('div', { class: 'tt-hint', text: `in ${f.group}` }) : null, f.node.sub ? el('div', { class: 'tt-hint', text: f.node.sub }) : null, el('div', { class: 'tt-hint', text: hint || (opens ? `${NA.tapWord()} for details, double-${NA.tapWord(true)} to open it` : `${NA.tapWord()} for the explanation and code`) })], e.clientX, e.clientY);
      else NA.tooltip.hide();
    }
    // zoomed far enough into the level: open the part nearest the middle of the view
    zoomedIn() {
      // the part under the cursor or the fingers, or under the middle for the + button
      const [W, H] = this.size(), a = this.zoom.anchor, cx = a ? a[0] : W / 2, cy = a ? a[1] : (H + 46) / 2;
      let f = this.pick(cx, cy);
      if (f && f.rel) { this.jump(f.rel); return; }
      if (this.mergeHits) return;
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
    // open a part of a merged level, however far down it is: the path goes straight there
    jump(rel) {
      this.path = this.path.concat(rel);
      if (this.merge && this.path.length >= this.mergeDepth) { this.mergeDepth = this.path.length; this.mergeForm = formOf(this.node()) || this.mergeForm; }
      this.mergeHits = null; this.selected = null; this.hoverId = null;
      this.zoomSet(1, 0, 0); this.lockUntil = performance.now() + 200; this.wheelHold = true;
      this.dirty = true; this.focusKey = null; NA.tooltip.hide();
    }
    // animate into a part of the current level (dur: a quicker flight, for a walk through several levels)
    enter(child, dur) {
      if (child.portal) { child.portal(this); return; }
      const R = this.levelRect(), parent = this.node(), c = this.content(parent, R), cr = S.place(c, child.rect);
      const k1 = Math.min(R.w / cr.w, R.h / cr.h), c0 = this.camNow();
      this.anim = { parent, child, s: 0, t0: performance.now(), dir: 1, dur, c0, c1: { k: k1 }, bw: cr.w, bh: cr.h,
        w0: [(R.x + R.w / 2 - c0.tx) / c0.k, (R.y + R.h / 2 - c0.ty) / c0.k], w1: [cr.x + cr.w / 2, cr.y + cr.h / 2], done: () => { this.path = this.path.concat(child.id); if (this.merge && this.path.length >= this.mergeDepth) { this.mergeDepth = this.path.length; this.mergeForm = formOf(this.node()) || this.mergeForm; } } };
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
      const a = this.anim, dur = NA.prefersReducedMotion() ? 1 : a.dur || 560, p = clamp((now - a.t0) / dur, 0, 1), e = easeInOut(p);
      a.s = a.dir > 0 ? e : 1 - e;
      if (p >= 1) {
        this.anim = null;
        // fingers still down must lift before a pinch can open another level; the wheel waits for a pause (wheelOpen)
        this.lockUntil = now + 120; this.gestureLock = this.zoom.ptrs.size > 0;
        if (a.done) a.done();
        this.zoomSet(1, 0, 0);
        this.focusKey = null;
        this.notify();
        if (this.queue) this.nextQueued();
      }
    }
    // end a flight at once, where it was going
    finishAnim() {
      const a = this.anim;
      if (!a) return;
      this.anim = null;
      if (a.done) a.done();
      this.zoomSet(1, 0, 0);
      this.focusKey = null;
    }

    /* A guided walk through the levels: each step flies to a level (through the ones between), lights the part it
       is about and says what happens there, with this model's own numbers. steps: [{ path, hl, text }] */
    follow(steps, title) {
      if (!steps || !steps.length) return;
      this.tour = { steps, k: 0, title: title || 'Follow one number' };
      this.goStep(0);
    }
    goStep(k) {
      const T = this.tour;
      if (!T) return;
      if (this.anim) this.finishAnim();
      T.k = clamp(k, 0, T.steps.length - 1);
      const st = T.steps[T.k];
      this.goPath(st.path);
      this.hl = st.hl || null; this.dirty = true;
      this.renderFollow();
    }
    endFollow() { this.tour = null; this.queue = null; this.hl = null; this.followEl.hidden = true; this.dirty = true; }
    renderFollow() {
      const T = this.tour, st = T.steps[T.k], last = T.k === T.steps.length - 1;
      clear(this.followEl);
      this.followEl.append(
        el('div', { class: 'af-head' }, el('span', { class: 'af-kicker', text: T.title }), el('span', { class: 'af-count', text: `${T.k + 1} of ${T.steps.length}` }),
          el('button', { type: 'button', class: 'icon-btn af-close', 'aria-label': 'Stop following', title: 'Stop following', onclick: () => this.endFollow() }, '✕')),
        el('p', { class: 'af-text', text: st.text }),
        el('div', { class: 'af-nav' },
          el('button', { type: 'button', class: 'btn btn-quiet', disabled: T.k === 0, onclick: () => this.goStep(T.k - 1) }, '◀ Back'),
          el('button', { type: 'button', class: 'btn btn-primary', onclick: () => (last ? this.endFollow() : this.goStep(T.k + 1)) }, last ? 'Done' : 'Next ▶')));
      this.followEl.hidden = false;
    }
    // fly to a path: back out to where it branches off, then into each level in turn
    goPath(target) {
      let p = 0;
      while (p < this.path.length && p < target.length && this.path[p] === target[p]) p++;
      if (!p) return;
      this.queue = target.slice(p);
      this.selected = null; this.mergeHits = null;
      if (this.path.length > p) this.exit(p - 1);
      else this.nextQueued();
    }
    nextQueued() {
      const q = this.queue;
      if (!q || !q.length) { this.queue = null; this.dirty = true; this.notify(); return; }
      const ch = this.childOf(this.node(), q.shift());
      if (!ch || !this.canEnter(ch)) { this.queue = null; this.dirty = true; return; }
      this.enter(ch, 420);
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
      const level = this.node(), sel = this.selectedNode();
      const key = this.path.join('/') + '|' + (sel ? this.selected : '');
      if (key === this.focusKey) return;
      this.focusKey = key;
      // a part selected deep inside a merged level: the path to the part it sits in, for the panel's trail
      const path = sel && this.selected[0] === '~' ? this.path.concat(this.selected.slice(1).split('/').slice(0, -1)) : this.path;
      this.focusSel = sel;
      if (this.opts.onFocus) this.opts.onFocus(sel || level, path, !sel);
    }
    refreshFocus() { this.focusKey = null; if (this.dirty) this.focusLater = true; else this.notify(); }
    // what the info panel shows for a part: where it sits, its explanation and its code
    infoFor(node, path) {
      const names = [];
      let n = this.root;
      for (const id of path.slice(1)) { n = this.childOf(n, id) || n; names.push(n.title); }
      // the open level itself, or a part selected in it (which may share the level's id)
      const level = node !== this.focusSel && node.id === path[path.length - 1] && path.join('/') === this.path.join('/');
      const trail = level ? names.slice(0, -1) : names;
      const k = level ? this.mergeK(path.length) : 0;
      const F = this.mergeForm, merged = k > 0 && !!this.mergeHits, mem = F === 'memory cell' || F === 'stored number';
      const about = merged ? [mem
        ? `Merged and wired up as a memory array: each ${F === 'memory cell' ? 'number' : 'stored number'} within ${k === 1 ? 'one level' : `${k} levels`} of this one sits on its own word line (a row), and each bit position on its own bit line (a column) running down to a sense amplifier. Opening a row connects every cell on it to its bit line at once (point at a row to see it), which is why reading neighbouring numbers is fast. A real chip has thousands of cells on each line, and on a PC memory stick the bytes of one number are usually spread over several chips, one byte in each. ${NA.tapWord()} a cell or a row for details; double-${NA.tapWord(true)} to open it.${this.mergeCut ? ` There are more than fit here, so only the first ${this.mergeCut} are drawn.` : ''}`
        : `Merged: every ${F || 'part'} within ${k === 1 ? 'one level' : `${k} levels`} of this one, side by side and drawn the way its own level looks, framed by the part of this level it belongs to (and inside that, by the part it sits in). Parts that are wired together are drawn wired: the gates of one circuit inside its own diagram, the full adders of one adder passing their carries along. ${F ? `Parts with no ${F} that close` : 'Parts with nothing that far down'} keep their usual boxes, under Other parts${this.mergeCut ? ' (so do any there was no room for)' : ''}. ${NA.tapWord()} a part to see what it is; double-${NA.tapWord(true)} it to open it. Turn Merge off to see this level as it usually is.${this.mergeCut ? ` There are more than fit here, so only the first ${this.mergeCut} are drawn.` : ''}`, ...(node.about || [])] : node.about;
      // the open level's words that name its parts are linked to them; what it is made of comes first
      this.links = [];
      let out = level && !merged && about ? this.linkify(node, about) : about;
      if (node.count) out = [el('p', { class: 'ip-count', text: node.count }), ...(out || [])];
      return { kicker: [this.opts.kicker || 'Architecture', ...trail].join(' › '), title: node.title + (node.sub ? ' · ' + node.sub : ''), about: out, code: node.code, codeLabel: node.codeLabel };
    }
    /* Words that name a part of the level (its listed phrases, then its parts' own names) become links: pointing
       at them lights that part of the picture. The first mention of each in a paragraph, longest names first. */
    linkify(node, about) {
      const targets = [];
      for (const t of node.terms || []) for (const w of t.words || []) targets.push({ key: t.key, w, cs: !!t.cs });
      for (const ch of node.children || []) {
        const w = ch.title;
        if (!w || w.length < 3 || ch.shape === 'ghost' || targets.some((q) => q.key === ch.id)) continue;
        targets.push({ key: ch.id, w, cs: w === w.toUpperCase() }); // AND, OR, XOR: only as capitals
      }
      if (!targets.length) return about;
      targets.sort((a, b) => b.w.length - a.w.length);
      const wordChar = /[\p{L}\p{N}_]/u;
      return about.map((para) => {
        if (typeof para !== 'string') return para;
        const low = para.toLowerCase(), found = [], used = new Set();
        for (const t of targets) {
          if (used.has(t.key)) continue;
          const hay = t.cs ? para : low, needle = t.cs ? t.w : t.w.toLowerCase();
          for (let at = hay.indexOf(needle); at >= 0; at = hay.indexOf(needle, at + 1)) {
            const end = at + needle.length;
            if ((at > 0 && wordChar.test(para[at - 1])) || (end < para.length && wordChar.test(para[end])) || found.some((f) => at < f.end && end > f.at)) continue;
            found.push({ at, end, key: t.key }); used.add(t.key);
            break;
          }
        }
        if (!found.length) return para;
        found.sort((a, b) => a.at - b.at);
        const p = el('p');
        let i = 0;
        for (const f of found) { if (f.at > i) p.append(para.slice(i, f.at)); p.append(this.linkSpan(para.slice(f.at, f.end), f.key)); i = f.end; }
        if (i < para.length) p.append(para.slice(i));
        return p;
      });
    }
    linkSpan(text, key) {
      const s = el('span', { class: 'arch-link', tabindex: '0', text });
      const on = () => this.setHl(key), off = () => { if (this.hl === key && !this.tour) this.setHl(null); };
      s.addEventListener('pointerenter', on);
      s.addEventListener('pointerleave', (e) => { if (e.pointerType !== 'touch') off(); });
      s.addEventListener('focus', on); s.addEventListener('blur', off);
      // a tap (a touch screen has no hover) lights the part for a moment
      s.addEventListener('click', () => { on(); clearTimeout(this.hlTimer); this.hlTimer = setTimeout(off, 2400); });
      this.links.push({ el: s, key });
      return s;
    }
    setHl(key) { if (this.hl !== key) { this.hl = key; this.dirty = true; } }
    // the words for the part under the pointer light up in the explanation
    syncLinks() {
      const node = this.root && this.node(), id = this.hoverId;
      for (const L of this.links) {
        const t = node && (node.terms || []).find((q) => q.key === L.key);
        L.el.classList.toggle('is-lit', !!id && (L.key === id || this.termHas(t, id)));
      }
    }

    // ------------------------------------------------------------------ the open level's tools
    syncTools() {
      const node = this.root && !this.anim ? this.node() : null;
      const key = node ? `${this.path.join('/')}|${node.toolsKey || ''}|${this.toolsRev || 0}|${this.tour ? 1 : 0}|${this.selected || ''}` : '';
      if (key === this.toolsKey) return;
      this.toolsKey = key;
      clear(this.toolsEl);
      const items = node && node.tools && !this.tour ? node.tools(this).filter(Boolean) : []; // a walk drives the view itself
      // a selected part that opens: Open, for a touch screen, or anyone who would rather not double-click
      const sel = node && !this.tour ? this.selectedNode() : null;
      if (sel && (this.selected[0] === '~' || this.canEnter(sel))) items.push(this.tool('Open ▸', () => this.openSelected(), { primary: true, title: `Open ${sel.title} (or double-click it)` }));
      this.toolsEl.append(...items);
      this.toolsEl.hidden = !items.length;
      // on a narrow stage the tools get their own row(s) under the title, and the level starts below them
      const stacked = this.narrow();
      this.toolsEl.classList.toggle('is-stacked', stacked);
      const w = items.length && !stacked ? this.toolsEl.offsetWidth + 12 : 0, h = items.length && stacked ? this.toolsEl.offsetHeight + 6 : 0;
      if (w !== this.toolsW || h !== this.toolsH) { this.toolsW = w; this.toolsH = h; this.dirty = true; }
    }
    narrow() { return this.stage.clientWidth < 560; }
    refreshTools() { this.toolsRev = (this.toolsRev || 0) + 1; this.dirty = true; }
    // a level's tool: a small button; o.pressed makes it a toggle
    tool(label, onClick, o = {}) {
      return el('button', { type: 'button', class: 'arch-tool' + (o.primary ? ' is-primary' : ''), title: o.title, 'aria-pressed': o.pressed === undefined ? null : String(!!o.pressed), disabled: !!o.disabled, onclick: onClick }, label);
    }
    toolSelect(options, value, onChange, label) {
      const s = el('select', { class: 'arch-tool arch-tool-select', 'aria-label': label, title: label }, ...options.map(([v, t]) => el('option', { value: v, selected: v === value }, t)));
      s.addEventListener('change', () => onChange(s.value));
      return s;
    }
    syncBar() {
      const key = this.path.join('/') + '|' + (this.pathTitles || '');
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
      this.crumbs.classList.toggle('is-cut', this.crumbs.scrollWidth > this.crumbs.clientWidth + 1); // and fade what scrolled out
      if (!this.anim) this.notify();
    }
    goRoot() { if (this.path.length > 1) this.exit(0); }
  }

  NA.ArchView = ArchView;
  NA.archKit = S;
})(window.NA);
