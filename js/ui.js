/* ui.js — shared interface pieces: activation-curve glyphs, the option menu (a popover on desktop,
   a bottom sheet on phones) and toast messages. */
(function (NA) {
  'use strict';
  const { el, clamp } = NA;
  const SVGNS = 'http://www.w3.org/2000/svg';

  // ------------------------------------------------------------------ activation glyphs
  // Samples f(z) for z in [-3, 3] and fits it into a w×h box, keeping zero visible.
  function actPoints(act, w, h, pad = 2) {
    const N = 40, zs = [], fs = [];
    for (let i = 0; i <= N; i++) { const z = -3 + (6 * i) / N; zs.push(z); fs.push(act.f(z)); }
    let lo = Math.min(0, ...fs), hi = Math.max(0, ...fs);
    if (hi - lo < 1e-6) { hi += 1; lo -= 1; }
    return { pts: zs.map((z, i) => [pad + ((z + 3) / 6) * (w - 2 * pad), h - pad - ((fs[i] - lo) / (hi - lo)) * (h - 2 * pad)]), zeroY: h - pad - ((0 - lo) / (hi - lo)) * (h - 2 * pad) };
  }
  function actGlyph(id, w = 30, h = 20) {
    const act = NA.engine.ACTIVATIONS[id];
    const svg = document.createElementNS(SVGNS, 'svg');
    svg.setAttribute('width', w); svg.setAttribute('height', h); svg.setAttribute('viewBox', `0 0 ${w} ${h}`); svg.setAttribute('aria-hidden', 'true');
    svg.setAttribute('class', 'act-glyph');
    if (!act) return svg;
    const { pts, zeroY } = actPoints(act, w, h);
    const axis = document.createElementNS(SVGNS, 'path');
    axis.setAttribute('d', `M1 ${zeroY.toFixed(1)}H${w - 1}M${w / 2} 1V${h - 1}`);
    axis.setAttribute('class', 'ag-axis');
    const curve = document.createElementNS(SVGNS, 'path');
    curve.setAttribute('d', pts.map((p, i) => (i ? 'L' : 'M') + p[0].toFixed(1) + ' ' + p[1].toFixed(1)).join(''));
    curve.setAttribute('class', 'ag-curve');
    svg.append(axis, curve);
    return svg;
  }
  // Canvas version for badges on the network diagram.
  function drawActGlyph(ctx, act, x, y, w, h, color, axisColor, scale = 1) {
    const { pts, zeroY } = actPoints(act, w, h, 1.5 * scale);
    ctx.save();
    ctx.translate(x, y);
    ctx.strokeStyle = axisColor; ctx.lineWidth = 0.8 * scale;
    ctx.beginPath(); ctx.moveTo(0.5 * scale, zeroY); ctx.lineTo(w - 0.5 * scale, zeroY); ctx.stroke();
    ctx.strokeStyle = color; ctx.lineWidth = 1.4 * scale; ctx.lineJoin = 'round'; ctx.lineCap = 'round';
    ctx.beginPath(); pts.forEach((p, i) => (i ? ctx.lineTo(p[0], p[1]) : ctx.moveTo(p[0], p[1]))); ctx.stroke();
    ctx.restore();
  }

  // ------------------------------------------------------------------ menu
  const menu = { root: null, onSelect: null, anchor: null };
  const isMobile = () => document.body.dataset.layout === 'mobile';

  /* opts: { anchor, title, note, items:[{id, label, sub, glyph, selected, tone, divider}], onSelect } */
  menu.open = function (opts) {
    // a second click on the same chip closes the menu instead of reopening it
    if (menu.lastClose && opts.anchor && menu.lastClose.anchor === opts.anchor && performance.now() - menu.lastClose.t < 400) return;
    menu.close(false);
    const sheet = isMobile();
    const list = el('div', { class: 'menu-list', role: 'listbox', 'aria-label': opts.title || 'Options' });
    for (const it of opts.items) {
      if (it.divider) { list.appendChild(el('div', { class: 'menu-div', role: 'separator' })); continue; }
      const b = el('button', { class: 'menu-item' + (it.tone ? ' tone-' + it.tone : ''), type: 'button', role: 'option', 'aria-selected': String(!!it.selected), title: it.sub || null, dataset: { id: it.id } },
        it.glyph ? actGlyph(it.glyph, 30, 20) : el('span', { class: 'menu-ico', 'aria-hidden': 'true', text: it.icon || '' }),
        el('span', { class: 'menu-text' }, el('span', { class: 'menu-label', text: it.label }), it.sub ? el('span', { class: 'menu-sub', text: it.sub }) : null),
        el('span', { class: 'menu-check', 'aria-hidden': 'true', text: it.selected ? '✓' : '' }));
      b.addEventListener('click', () => { const fn = opts.onSelect; menu.close(); fn(it.id); });
      list.appendChild(b);
    }
    const head = (opts.title || sheet) ? el('div', { class: 'menu-head' },
      el('div', null, el('div', { class: 'menu-title', text: opts.title || '' }), opts.note ? el('div', { class: 'menu-note', text: opts.note }) : null),
      sheet ? el('button', { class: 'icon-btn', type: 'button', 'aria-label': 'Close', onclick: () => menu.close() }, '×') : null) : null;
    const panel = el('div', { class: 'menu-panel' + (sheet ? ' is-sheet' : ''), role: 'dialog', 'aria-modal': sheet ? 'true' : 'false', 'aria-label': opts.title || 'Options' }, head, list);
    const root = el('div', { class: 'menu-root' + (sheet ? ' is-sheet' : '') }, sheet ? el('div', { class: 'menu-scrim', onclick: () => menu.close() }) : null, panel);
    document.body.appendChild(root);
    menu.root = root; menu.anchor = opts.anchor || null;
    if (!sheet && opts.anchor) {
      const r = opts.anchor.getBoundingClientRect(), pw = Math.min(320, window.innerWidth - 16);
      panel.style.width = pw + 'px';
      // open below the anchor, or above it when there is more room there, never covering it
      const ph = panel.getBoundingClientRect().height, below = window.innerHeight - r.bottom - 14, above = r.top - 14;
      const down = ph <= below || below >= above, h = Math.min(ph, Math.max(140, down ? below : above));
      panel.style.maxHeight = h + 'px';
      panel.style.left = clamp(r.left + r.width / 2 - pw / 2, 8, window.innerWidth - pw - 8) + 'px';
      panel.style.top = (down ? r.bottom + 6 : Math.max(8, r.top - 6 - h)) + 'px';
    }
    setTimeout(() => {
      document.addEventListener('pointerdown', outside, true);
      document.addEventListener('keydown', keys, true);
    }, 0);
    const first = list.querySelector('[aria-selected="true"]') || list.querySelector('.menu-item');
    if (first) first.focus();
  };
  menu.close = function (refocus = true) {
    if (!menu.root) return;
    menu.lastClose = { anchor: menu.anchor, t: performance.now() };
    menu.root.remove(); menu.root = null;
    document.removeEventListener('pointerdown', outside, true);
    document.removeEventListener('keydown', keys, true);
    if (refocus && menu.anchor && document.contains(menu.anchor)) menu.anchor.focus();
  };
  function outside(e) { if (menu.root && !e.target.closest('.menu-panel')) { e.stopPropagation(); menu.close(false); } }
  function keys(e) {
    if (!menu.root) return;
    if (e.key === 'Escape') { e.preventDefault(); e.stopPropagation(); menu.close(); return; }
    if (e.key !== 'ArrowDown' && e.key !== 'ArrowUp') return;
    e.preventDefault();
    const items = NA.$$('.menu-item', menu.root), i = items.indexOf(document.activeElement);
    const next = items[(i + (e.key === 'ArrowDown' ? 1 : items.length - 1)) % items.length];
    if (next) next.focus();
  }

  // Menu items for every activation function, with the formula under the name.
  function activationItems(selectedId) {
    return NA.engine.ACTIVATION_ORDER.map((id) => {
      const a = NA.engine.ACTIVATIONS[id];
      return { id, label: a.label, sub: 'f(z) = ' + a.formula, glyph: id, selected: id === selectedId };
    });
  }

  // ------------------------------------------------------------------ zoom
  /* Pinch (touch, or a trackpad: ctrl+wheel, or Safari's gesture events), wheel, double-tap and button zoom
     for one canvas, with panning. Drawing code calls apply(ctx, dpr) before drawing in unzoomed ("world")
     coordinates, and maps pointer positions back with toWorld(). panOne: one finger or the mouse pans once
     zoomed in. wheel: 'ctrl' (the plain wheel scrolls the page; for canvases inside scrolling panels),
     'fixed' (the plain wheel zooms while the page itself does not scroll) or 'always'. wheelGate(e) may
     veto a wheel zoom, e.g. to stop one long scroll from running through several levels. */
  const pageFixed = () => document.documentElement.scrollHeight <= window.innerHeight + 2;
  class Zoom {
    constructor(canvas, opts = {}) {
      this.c = canvas; this.k = 1; this.tx = 0; this.ty = 0; this.max = opts.max || 4;
      this.onChange = opts.onChange || (() => {});
      this.onUnderflow = opts.onUnderflow || null;
      this.panOne = !!opts.panOne;
      this.noDouble = !!opts.noDouble; // the architecture view uses clicks to fly instead
      this.wheelMode = opts.wheel || 'ctrl';
      this.wheelGate = opts.wheelGate || null;
      this.onDouble = opts.onDouble || null; // (px, py) → true when the view handled the double-click itself
      this.ptrs = new Map(); this.g = null; this.start = null; this.lastTap = null; this.suppressClick = false;
      canvas.addEventListener('pointerdown', (e) => this.down(e));
      canvas.addEventListener('pointermove', (e) => this.move(e));
      canvas.addEventListener('pointerup', (e) => this.up(e, false));
      canvas.addEventListener('pointercancel', (e) => this.up(e, true));
      canvas.addEventListener('wheel', (e) => this.wheel(e), { passive: false });
      // Safari reports a trackpad pinch as gesture events rather than ctrl+wheel; on a touch screen the
      // pointer events above already handle the pinch, so the gesture events are ignored there
      canvas.addEventListener('gesturestart', (e) => { if (this.ptrs.size) return; e.preventDefault(); this.gs = 1; });
      canvas.addEventListener('gesturechange', (e) => {
        if (this.ptrs.size || !this.gs) return;
        e.preventDefault();
        const p = this.local(e), f = e.scale / this.gs;
        this.gs = e.scale;
        if (!this.wheelGate || this.wheelGate(e)) this.zoomAt(p[0], p[1], f);
      });
      canvas.addEventListener('gestureend', () => { this.gs = 0; });
      if (!this.noDouble) canvas.addEventListener('dblclick', (e) => { const p = this.local(e); if (!(this.onDouble && this.onDouble(p[0], p[1]))) this.toggleAt(p[0], p[1]); });
      if (opts.host) this.buildChip(opts.host);
    }
    get wheelZooms() { return this.wheelMode === 'always' || (this.wheelMode === 'fixed' && pageFixed()); }
    get pinching() { return !!(this.g && this.g.type === 'pinch') || this.ptrs.size > 1; }
    get busy() { return !!this.g || this.ptrs.size > 1; }
    local(e) { const r = this.c.getBoundingClientRect(); return [e.clientX - r.left, e.clientY - r.top]; }
    size() { const r = this.c.getBoundingClientRect(); return [r.width, r.height]; }
    toWorld(px, py) { return [(px - this.tx) / this.k, (py - this.ty) / this.k]; }
    apply(ctx, dpr) { ctx.setTransform(dpr * this.k, 0, 0, dpr * this.k, dpr * this.tx, dpr * this.ty); }
    set(k, tx, ty) {
      this.k = clamp(k, 1, this.max);
      const [w, h] = this.size();
      this.tx = this.k === 1 ? 0 : clamp(tx, w - w * this.k, 0);
      this.ty = this.k === 1 ? 0 : clamp(ty, h - h * this.k, 0);
      this.sync(); this.onChange(this);
    }
    zoomAt(px, py, factor) {
      this.anchor = [px, py]; // where the zoom is centred: the cursor, the fingers or the middle
      // zooming out while already at 1× is reported, so a view can step back out a level
      if (factor < 1 && this.k <= 1.0001 && this.onUnderflow) { this.onUnderflow(px, py); return; }
      const k = clamp(this.k * factor, 1, this.max), f = k / this.k;
      this.set(k, px - (px - this.tx) * f, py - (py - this.ty) * f);
    }
    toggleAt(px, py) { if (this.k > 1.2) this.set(1, 0, 0); else this.zoomAt(px, py, 2.5); }
    reset() { if (this.k !== 1) this.set(1, 0, 0); }
    // after the view jumps (a new level opens mid-pinch), measure the rest of the pinch from here
    rebase() {
      if (!this.g || this.g.type !== 'pinch' || this.ptrs.size < 2) return;
      const [a, b] = [...this.ptrs.values()];
      Object.assign(this.g, { d0: Math.hypot(a[0] - b[0], a[1] - b[1]) || 1, m0: [(a[0] + b[0]) / 2, (a[1] + b[1]) / 2], k0: this.k, tx0: this.tx, ty0: this.ty, under: false });
    }
    down(e) {
      this.ptrs.set(e.pointerId, this.local(e));
      if (this.ptrs.size === 1) { this.suppressClick = false; this.start = { p: this.local(e), tx: this.tx, ty: this.ty }; }
      if (this.ptrs.size === 2) {
        const [a, b] = [...this.ptrs.values()];
        this.g = { type: 'pinch', d0: Math.hypot(a[0] - b[0], a[1] - b[1]) || 1, m0: [(a[0] + b[0]) / 2, (a[1] + b[1]) / 2], k0: this.k, tx0: this.tx, ty0: this.ty };
        this.suppressClick = true;
      }
    }
    move(e) {
      if (!this.ptrs.has(e.pointerId)) return;
      const p = this.local(e);
      this.ptrs.set(e.pointerId, p);
      if (this.g && this.g.type === 'pinch' && this.ptrs.size >= 2) {
        const [a, b] = [...this.ptrs.values()], g = this.g;
        const d = Math.hypot(a[0] - b[0], a[1] - b[1]) || 1, m = [(a[0] + b[0]) / 2, (a[1] + b[1]) / 2];
        this.anchor = m;
        if (this.onUnderflow && g.k0 <= 1.0001 && (g.k0 * d) / g.d0 < 0.72 && !g.under) { g.under = true; this.onUnderflow(m[0], m[1]); return; }
        const k = clamp((g.k0 * d) / g.d0, 1, this.max), wx = (g.m0[0] - g.tx0) / g.k0, wy = (g.m0[1] - g.ty0) / g.k0;
        this.set(k, m[0] - wx * k, m[1] - wy * k);
      } else if (this.ptrs.size === 1 && this.start && this.k > 1 && (this.panOne || (this.g && this.g.type === 'pan'))) {
        const dx = p[0] - this.start.p[0], dy = p[1] - this.start.p[1];
        if (this.g || Math.hypot(dx, dy) > 6) { this.g = { type: 'pan' }; this.suppressClick = true; this.set(this.k, this.start.tx + dx, this.start.ty + dy); }
      }
    }
    up(e, cancel) {
      const had = this.ptrs.delete(e.pointerId);
      if (this.ptrs.size === 0) {
        if (!cancel && had && !this.g && e.pointerType === 'touch' && !this.noDouble) {
          const p = this.local(e), now = performance.now(), lt = this.lastTap;
          if (lt && now - lt.t < 320 && Math.hypot(p[0] - lt.p[0], p[1] - lt.p[1]) < 28) { if (!(this.onDouble && this.onDouble(p[0], p[1]))) this.toggleAt(p[0], p[1]); this.lastTap = null; this.suppressClick = true; }
          else this.lastTap = { p, t: now };
        }
        this.g = null; this.start = null;
      } else if (this.ptrs.size === 1 && this.g && this.g.type === 'pinch') {
        const p = [...this.ptrs.values()][0];
        this.start = { p, tx: this.tx, ty: this.ty };
        this.g = { type: 'pan' }; // keep panning with the remaining finger
      }
    }
    wheel(e) {
      const unit = e.deltaMode === 1 ? 33 : e.deltaMode === 2 ? this.size()[1] : 1, dx = e.deltaX * unit, dy = e.deltaY * unit;
      if (e.ctrlKey || (this.wheelZooms && Math.abs(dy) >= Math.abs(dx))) {
        // a trackpad pinch arrives as ctrl + small steps; a mouse-wheel notch (about 100, with or without ctrl)
        // zooms about 1.6×
        e.preventDefault();
        if (this.wheelGate && !this.wheelGate(e)) return;
        const p = this.local(e), rate = e.ctrlKey && Math.abs(dy) < 50 ? 0.01 : 0.0045;
        this.zoomAt(p[0], p[1], Math.exp(-clamp(dy, -240, 240) * rate));
      } else if (this.k > 1) { e.preventDefault(); this.set(this.k, this.tx - dx, this.ty - dy); }
    }
    buildChip(host) {
      const center = () => { const [w, h] = this.size(); return [w / 2, h / 2]; };
      const minus = el('button', { type: 'button', class: 'zc-btn', 'aria-label': 'Zoom out', title: 'Zoom out', onclick: () => { const c = center(); this.zoomAt(c[0], c[1], 1 / 1.6); } }, '−');
      this.levelEl = el('button', { type: 'button', class: 'zc-level', 'aria-label': 'Reset zoom', title: this.noDouble ? 'Reset zoom' : 'Reset zoom (or double-tap)', onclick: () => this.reset() }, '1×');
      const plus = el('button', { type: 'button', class: 'zc-btn', 'aria-label': 'Zoom in', onclick: () => { const c = center(); this.zoomAt(c[0], c[1], 1.6); } }, '+');
      // the hint names the gestures that work right now (the plain wheel zooms only where it is not needed to scroll)
      plus.addEventListener('pointerenter', () => {
        const how = [this.wheelZooms ? 'scroll' : 'Ctrl + scroll', 'pinch', this.noDouble ? null : 'double-tap'].filter(Boolean);
        plus.title = `Zoom in (${how.slice(0, -1).join(', ')} or ${how[how.length - 1]})`;
      });
      this.chip = el('div', { class: 'zoom-chip', role: 'group', 'aria-label': 'Zoom' }, minus, this.levelEl, plus);
      host.appendChild(this.chip);
    }
    sync() {
      this.c.classList.toggle('is-zoomed', this.k > 1.01);
      if (!this.levelEl) return;
      this.levelEl.textContent = (Math.round(this.k * 10) / 10).toString().replace(/\.0$/, '') + '×';
      this.chip.classList.toggle('is-zoomed', this.k > 1.01);
    }
  }

  // ------------------------------------------------------------------ toast
  const toast = {
    show(msg) {
      if (!toast.el) { toast.el = el('div', { class: 'toast', role: 'status', 'aria-live': 'polite' }); document.body.appendChild(toast.el); }
      toast.el.textContent = msg;
      toast.el.classList.add('is-on');
      clearTimeout(toast.timer);
      toast.timer = setTimeout(() => toast.el.classList.remove('is-on'), 3200);
    },
  };

  // ------------------------------------------------------------------ digit thumbnails
  // Sample k of the current image dataset as a G×G canvas: ink on the panel colour.
  function digitCanvas(k) {
    const ds = NA.app.ds, G = ds.imgSize, c = document.createElement('canvas');
    c.width = G; c.height = G;
    const cx = c.getContext('2d'), im = cx.createImageData(G, G), ink = NA.theme.tokens.ink, base = NA.theme.tokens['panel-2'];
    for (let i = 0; i < G * G; i++) {
      const t = clamp(ds.raw[k * G * G + i], 0, 1);
      for (let ch = 0; ch < 3; ch++) im.data[4 * i + ch] = base[ch] + (ink[ch] - base[ch]) * t;
      im.data[4 * i + 3] = 255;
    }
    cx.putImageData(im, 0, 0);
    return c;
  }

  /* Screen readers. A canvas has no text, so a picture registers a function that describes what it shows now
     (a11y.describe(canvas, () => 'text')); descriptions of the pictures on screen are refreshed every 2 s.
     a11y.announce(message) speaks a short message, such as a training run finishing. */
  const a11y = { items: new Map(), live: null };
  a11y.describe = (canvas, fn) => { a11y.items.set(canvas, fn); canvas.setAttribute('role', 'img'); };
  a11y.refresh = () => {
    for (const [c, fn] of a11y.items) {
      if (!c.isConnected) { a11y.items.delete(c); continue; }
      if (!c.offsetParent) continue;
      try { const t = fn(); if (t && c.getAttribute('aria-label') !== t) c.setAttribute('aria-label', t); } catch (e) { /* a picture mid-update */ }
    }
  };
  a11y.announce = (msg) => {
    if (!a11y.live) { a11y.live = el('div', { class: 'sr-only', 'aria-live': 'polite', role: 'status' }); document.body.append(a11y.live); }
    a11y.live.textContent = '';
    setTimeout(() => { a11y.live.textContent = msg; }, 60);
  };
  if (typeof setInterval !== 'undefined') setInterval(a11y.refresh, 2000);
  NA.a11y = a11y;

  NA.ui = { actGlyph, drawActGlyph, menu, activationItems, toast, Zoom, digitCanvas };
})(window.NA);
