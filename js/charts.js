/* charts.js — small canvas charts for the inspector: line charts with a crosshair,
   histograms, the activation-function plot, per-neuron ranges and Hinton diagrams. */
(function (NA) {
  'use strict';
  const { theme, clamp, fmt, el, rr } = NA;

  const mono = () => NA.getFont('mono');
  const ui = () => NA.getFont('ui');

  function niceLogTicks(lo, hi) {
    const out = [];
    for (let e = Math.floor(Math.log10(lo)); e <= Math.ceil(Math.log10(hi)); e++) { const v = Math.pow(10, e); if (v >= lo * 0.999 && v <= hi * 1.001) out.push(v); }
    return out;
  }
  // powers of ten as people write them: 0.01 to 100 plainly, beyond that 10⁻⁴ or 10⁵
  const SUP = '⁰¹²³⁴⁵⁶⁷⁸⁹';
  const expLabel = (v) => { const e = Math.round(Math.log10(v)); return e >= -2 && e <= 2 ? String(+Math.pow(10, e).toFixed(Math.max(0, -e))) : '10' + (e < 0 ? '⁻' : '') + String(Math.abs(e)).split('').map((c) => SUP[c]).join(''); };

  /* Line chart with hover crosshair. series: [{name, color, points:[[x,y]], width}] */
  class LineChart {
    constructor(canvas, opts = {}) {
      this.canvas = canvas; this.opts = opts; this.data = null; this.hoverX = null;
      canvas.addEventListener('pointermove', (e) => this.onMove(e));
      canvas.addEventListener('pointerleave', () => { this.hoverX = null; NA.tooltip.hide(); this.draw(); });
    }
    set(data) { this.data = data; this.draw(); }
    frame() {
      const c = this.canvas, w = c.parentElement ? c.parentElement.clientWidth : 300;
      this.W = Math.max(120, w); this.H = this.opts.height || 150;
      this.ctx = NA.fitCanvas(c, this.W, this.H);
    }
    scales() {
      const d = this.data, o = this.opts;
      const padL = 40, padR = 12, padT = 10, padB = 24;
      let xmin = Infinity, xmax = -Infinity, ymin = Infinity, ymax = -Infinity;
      for (const s of d.series) for (const p of s.points) {
        if (!isFinite(p[1]) || (o.logY && p[1] <= 0)) continue;
        xmin = Math.min(xmin, p[0]); xmax = Math.max(xmax, p[0]);
        ymin = Math.min(ymin, p[1]); ymax = Math.max(ymax, p[1]);
      }
      if (!isFinite(xmin)) { xmin = 0; xmax = 1; ymin = 0; ymax = 1; }
      if (xmax === xmin) xmax = xmin + 1;
      if (d.yFixed) { ymin = d.yFixed[0]; ymax = d.yFixed[1]; }
      else if (o.logY) { ymin = Math.pow(10, Math.floor(Math.log10(Math.max(ymin, 1e-12)))); ymax = Math.pow(10, Math.ceil(Math.log10(Math.max(ymax, ymin * 10)))); }
      else { if (o.zeroBased) ymin = Math.min(0, ymin); const sp = ymax - ymin || 1; ymax += sp * 0.08; if (!o.zeroBased) ymin -= sp * 0.05; }
      const w = this.W - padL - padR, h = this.H - padT - padB;
      const X = (x) => padL + ((x - xmin) / (xmax - xmin)) * w;
      const Y = o.logY ? (y) => padT + h - ((Math.log10(Math.max(y, 1e-30)) - Math.log10(ymin)) / (Math.log10(ymax) - Math.log10(ymin))) * h : (y) => padT + h - ((y - ymin) / (ymax - ymin)) * h;
      return { padL, padR, padT, padB, w, h, X, Y, xmin, xmax, ymin, ymax };
    }
    draw() {
      if (!this.data) return;
      if (!this.ctx) this.frame();
      const ctx = this.ctx, d = this.data, o = this.opts, sc = this.scales();
      this.sc = sc;
      ctx.clearRect(0, 0, this.W, this.H);
      ctx.font = `500 10px ${mono()}`;
      // gridlines + y ticks
      let ticks = o.logY ? niceLogTicks(sc.ymin, sc.ymax) : NA.niceTicks(sc.ymin, sc.ymax, 4);
      // no more labels than fit (a log axis over many decades): keep every k-th
      const room = Math.max(2, Math.floor(sc.h / 15));
      if (ticks.length > room) { const k = Math.ceil(ticks.length / room); ticks = ticks.filter((t, i) => i % k === 0); }
      // the same number of decimals on every label of an axis (0.0, 0.2 … 1.0, not 1.0 and 0.80)
      const step = ticks.length > 1 ? Math.abs(ticks[1] - ticks[0]) : 1, dec = clamp(-Math.floor(Math.log10(step + 1e-12)), 0, 4);
      ctx.strokeStyle = theme.c('line'); ctx.lineWidth = 1; ctx.fillStyle = theme.c('muted');
      ctx.textAlign = 'right'; ctx.textBaseline = 'middle';
      for (const t of ticks) {
        const y = Math.round(sc.Y(t)) + 0.5;
        if (y < sc.padT - 1 || y > sc.padT + sc.h + 1) continue;
        ctx.beginPath(); ctx.moveTo(sc.padL, y); ctx.lineTo(sc.padL + sc.w, y); ctx.stroke();
        ctx.fillText(o.yFormat ? o.yFormat(t) : o.logY ? expLabel(t) : t.toFixed(dec).replace('-', NA.MINUS), sc.padL - 6, y);
      }
      // x ticks
      ctx.textAlign = 'center'; ctx.textBaseline = 'top';
      const xt = NA.niceTicks(sc.xmin, sc.xmax, 4);
      for (const t of xt) { const x = sc.X(t); if (x < sc.padL - 2 || x > sc.padL + sc.w + 2) continue; ctx.fillText(NA.fmtCompact(t), x, sc.padT + sc.h + 7); }
      ctx.strokeStyle = theme.c('line-2'); ctx.beginPath(); ctx.moveTo(sc.padL, sc.padT + sc.h + 0.5); ctx.lineTo(sc.padL + sc.w, sc.padT + sc.h + 0.5); ctx.stroke();
      // band (e.g. healthy range)
      if (d.band) {
        const y1 = sc.Y(d.band[1]), y2 = sc.Y(d.band[0]);
        ctx.fillStyle = theme.c('good', 0.08); ctx.fillRect(sc.padL, Math.max(sc.padT, y1), sc.w, Math.min(sc.padT + sc.h, y2) - Math.max(sc.padT, y1));
      }
      ctx.save(); ctx.beginPath(); ctx.rect(sc.padL, sc.padT - 2, sc.w, sc.h + 4); ctx.clip();
      for (const s of d.series) {
        const pts = s.points.filter((p) => isFinite(p[1]) && (!o.logY || p[1] > 0));
        if (!pts.length) continue;
        if (s.area) {
          ctx.beginPath(); ctx.moveTo(sc.X(pts[0][0]), sc.padT + sc.h);
          for (const p of pts) ctx.lineTo(sc.X(p[0]), sc.Y(p[1]));
          ctx.lineTo(sc.X(pts[pts.length - 1][0]), sc.padT + sc.h); ctx.closePath();
          ctx.fillStyle = s.color.replace('rgb(', 'rgba(').replace(')', ',0.1)'); ctx.fill();
        }
        ctx.strokeStyle = s.color; ctx.lineWidth = s.width || 2; ctx.lineJoin = 'round'; ctx.lineCap = 'round';
        ctx.beginPath(); pts.forEach((p, i) => (i ? ctx.lineTo(sc.X(p[0]), sc.Y(p[1])) : ctx.moveTo(sc.X(p[0]), sc.Y(p[1])))); ctx.stroke();
      }
      ctx.restore();
      // each series' latest value as a dot, drawn outside the clip so the edge does not cut it
      for (const s of d.series) {
        const pts = s.points.filter((p) => isFinite(p[1]) && (!o.logY || p[1] > 0)), last = pts[pts.length - 1];
        if (!last) continue;
        const y = sc.Y(last[1]);
        if (y < sc.padT - 4 || y > sc.padT + sc.h + 4) continue;
        ctx.beginPath(); ctx.arc(sc.X(last[0]), y, 3.5, 0, Math.PI * 2); ctx.fillStyle = s.color; ctx.fill();
        ctx.lineWidth = 2; ctx.strokeStyle = theme.c('panel'); ctx.stroke();
      }
      if (d.marker !== undefined && d.marker !== null) {
        const x = Math.round(sc.X(d.marker)) + 0.5;
        ctx.strokeStyle = theme.c('ink', 0.5); ctx.lineWidth = 1;
        ctx.beginPath(); ctx.moveTo(x, sc.padT); ctx.lineTo(x, sc.padT + sc.h); ctx.stroke();
      }
      if (this.hoverX !== null) this.drawHover();
    }
    onMove(e) {
      if (!this.data || !this.sc) return;
      const r = this.canvas.getBoundingClientRect(), mx = e.clientX - r.left, sc = this.sc;
      if (mx < sc.padL || mx > sc.padL + sc.w) { this.hoverX = null; NA.tooltip.hide(); this.draw(); return; }
      const xv = sc.xmin + ((mx - sc.padL) / sc.w) * (sc.xmax - sc.xmin);
      const s0 = this.data.series.find((s) => s.points.length);
      if (!s0) return;
      let best = s0.points[0][0];
      for (const p of s0.points) if (Math.abs(p[0] - xv) < Math.abs(best - xv)) best = p[0];
      this.hoverX = best;
      this.draw();
      const rows = [el('div', { class: 'tt-title', text: `${this.opts.xLabel || 'Step'} ${NA.fmtInt(best)}` })];
      for (const s of this.data.series) {
        const p = s.points.find((q) => q[0] === best) || s.points.reduce((a, q) => (Math.abs(q[0] - best) < Math.abs(a[0] - best) ? q : a), s.points[0]);
        if (!p) continue;
        rows.push(el('div', { class: 'tt-row' }, el('span', { class: 'tt-v', text: this.opts.format ? this.opts.format(p[1]) : fmt(p[1], 3) }), el('span', { class: 'tt-k' }, el('i', { class: 'key-line', style: { background: s.color } }), s.name)));
      }
      NA.tooltip.show(rows, e.clientX, e.clientY);
    }
    drawHover() {
      const ctx = this.ctx, sc = this.sc, x = Math.round(sc.X(this.hoverX)) + 0.5;
      ctx.strokeStyle = theme.c('ink', 0.4); ctx.lineWidth = 1;
      ctx.beginPath(); ctx.moveTo(x, sc.padT); ctx.lineTo(x, sc.padT + sc.h); ctx.stroke();
      for (const s of this.data.series) {
        const p = s.points.find((q) => q[0] === this.hoverX);
        if (!p || !isFinite(p[1]) || (this.opts.logY && p[1] <= 0)) continue;
        ctx.beginPath(); ctx.arc(x, sc.Y(p[1]), 4, 0, Math.PI * 2); ctx.fillStyle = s.color; ctx.fill();
        ctx.lineWidth = 2; ctx.strokeStyle = theme.c('panel'); ctx.stroke();
      }
    }
  }

  /* Histogram of values. opts: {range:[lo,hi], bins, color, markers:[{x,label,color}], height, xLabel} */
  function histogram(canvas, values, opts = {}) {
    const W = Math.max(120, canvas.parentElement.clientWidth), H = opts.height || 110;
    const ctx = NA.fitCanvas(canvas, W, H);
    const padL = 8, padR = 8, padT = 8, padB = 22, w = W - padL - padR, h = H - padT - padB;
    let lo = opts.range ? opts.range[0] : Infinity, hi = opts.range ? opts.range[1] : -Infinity;
    if (!opts.range) for (const v of values) { if (v < lo) lo = v; if (v > hi) hi = v; }
    if (!isFinite(lo)) { lo = -1; hi = 1; }
    if (hi - lo < 1e-9) { lo -= 0.5; hi += 0.5; }
    const B = opts.bins || 30, counts = new Float64Array(B);
    for (const v of values) { const b = Math.floor(((v - lo) / (hi - lo)) * B); counts[clamp(b, 0, B - 1)]++; }
    const cmax = Math.max(1, ...counts);
    ctx.clearRect(0, 0, W, H);
    const bw = w / B;
    for (let b = 0; b < B; b++) {
      if (!counts[b]) continue;
      const bh = Math.max(1.5, (counts[b] / cmax) * h), x = padL + b * bw;
      const mid = lo + ((b + 0.5) / B) * (hi - lo);
      ctx.fillStyle = opts.colorFn ? opts.colorFn(mid) : opts.color || theme.c('pos');
      rr(ctx, x + 1, padT + h - bh, Math.max(1, bw - 2), bh, Math.min(2, bw / 3)); ctx.fill();
    }
    ctx.strokeStyle = theme.c('line-2'); ctx.lineWidth = 1;
    ctx.beginPath(); ctx.moveTo(padL, padT + h + 0.5); ctx.lineTo(padL + w, padT + h + 0.5); ctx.stroke();
    ctx.font = `500 10px ${mono()}`; ctx.fillStyle = theme.c('muted'); ctx.textBaseline = 'top';
    ctx.textAlign = 'left'; ctx.fillText(fmt(lo, 2), padL, padT + h + 6);
    ctx.textAlign = 'right'; ctx.fillText(fmt(hi, 2), padL + w, padT + h + 6);
    if (lo < 0 && hi > 0) { const x = padL + ((0 - lo) / (hi - lo)) * w; ctx.textAlign = 'center'; ctx.fillText('0', x, padT + h + 6); ctx.strokeStyle = theme.c('ink', 0.25); ctx.beginPath(); ctx.moveTo(Math.round(x) + 0.5, padT); ctx.lineTo(Math.round(x) + 0.5, padT + h); ctx.stroke(); }
    for (const m of opts.markers || []) {
      const x = Math.round(padL + ((m.x - lo) / (hi - lo)) * w) + 0.5;
      ctx.strokeStyle = m.color || theme.c('ink'); ctx.lineWidth = 1.5;
      ctx.beginPath(); ctx.moveTo(x, padT); ctx.lineTo(x, padT + h); ctx.stroke();
    }
  }

  /* The activation function f(z), its slope f'(z), the distribution of z over the data and the probe's z. */
  function activationPlot(canvas, act, zValues, zProbe, opts = {}) {
    const W = Math.max(160, canvas.parentElement.clientWidth), H = opts.height || 150;
    const ctx = NA.fitCanvas(canvas, W, H);
    const padL = 34, padR = 10, padT = 10, padB = 24, w = W - padL - padR, h = H - padT - padB;
    let zmax = 3;
    for (const v of zValues) zmax = Math.max(zmax, Math.abs(v));
    zmax = Math.min(zmax * 1.08, 12);
    if (zProbe !== null && isFinite(zProbe)) zmax = Math.max(zmax, Math.min(12, Math.abs(zProbe) * 1.1));
    const lo = -zmax, hi = zmax;
    // value range of f over the window (always showing 0), and the slope f′ scaled to 45% of it,
    // sharing f's zero line, so curves with negative slopes (sine, Gaussian) fit as well
    let fmin = 0, fmax = 0, dmin = 0, dmax = 0;
    for (let i = 0; i <= 120; i++) {
      const z = lo + ((hi - lo) * i) / 120, a = act.f(z), d = act.d(z, a);
      fmin = Math.min(fmin, a); fmax = Math.max(fmax, a); dmin = Math.min(dmin, d); dmax = Math.max(dmax, d);
    }
    if (act.lo !== undefined) { fmin = Math.min(fmin, act.lo); fmax = Math.max(fmax, act.hi); }
    if (fmax - fmin < 1e-6) fmax = fmin + 1;
    const dk = (0.45 * (fmax - fmin)) / Math.max(1e-9, dmax, -dmin);
    const pad = Math.max(0.1, 0.06 * (fmax - fmin));
    const flo = Math.min(fmin, dmin * dk) - pad, fhi = Math.max(fmax, dmax * dk) + pad;
    const X = (z) => padL + ((z - lo) / (hi - lo)) * w, Y = (a) => padT + h - ((a - flo) / (fhi - flo)) * h;
    ctx.clearRect(0, 0, W, H);
    // distribution of z as soft columns behind the curve
    const B = 40, counts = new Float64Array(B);
    for (const v of zValues) counts[clamp(Math.floor(((v - lo) / (hi - lo)) * B), 0, B - 1)]++;
    const cmax = Math.max(1, ...counts);
    ctx.fillStyle = theme.c('pos', 0.16);
    for (let b = 0; b < B; b++) { if (!counts[b]) continue; const bh = (counts[b] / cmax) * h * 0.92; ctx.fillRect(padL + (b * w) / B + 0.5, padT + h - bh, w / B - 1, bh); }
    // axes
    ctx.strokeStyle = theme.c('line-2'); ctx.lineWidth = 1;
    ctx.beginPath(); ctx.moveTo(padL, Math.round(Y(0)) + 0.5); ctx.lineTo(padL + w, Math.round(Y(0)) + 0.5); ctx.moveTo(Math.round(X(0)) + 0.5, padT); ctx.lineTo(Math.round(X(0)) + 0.5, padT + h); ctx.stroke();
    ctx.font = `500 10px ${mono()}`; ctx.fillStyle = theme.c('muted');
    ctx.textAlign = 'center'; ctx.textBaseline = 'top';
    for (const t of NA.niceTicks(lo, hi, 4)) ctx.fillText(fmt(t, Math.abs(t) < 2 && t !== 0 ? 1 : 0), X(t), padT + h + 6);
    ctx.textAlign = 'right'; ctx.textBaseline = 'middle';
    for (const t of NA.niceTicks(flo, fhi, 3)) ctx.fillText(fmt(t, 1), padL - 5, Y(t));
    // slope f'(z), scaled into the same box
    ctx.strokeStyle = theme.c('muted', 0.8); ctx.lineWidth = 1.2;
    ctx.beginPath();
    for (let i = 0; i <= 120; i++) { const z = lo + ((hi - lo) * i) / 120, a = act.f(z), y = Y(act.d(z, a) * dk); i ? ctx.lineTo(X(z), y) : ctx.moveTo(X(z), y); }
    ctx.stroke();
    // f(z)
    ctx.strokeStyle = theme.c('ink'); ctx.lineWidth = 2; ctx.lineJoin = 'round';
    ctx.beginPath();
    for (let i = 0; i <= 160; i++) { const z = lo + ((hi - lo) * i) / 160; i ? ctx.lineTo(X(z), Y(act.f(z))) : ctx.moveTo(X(z), Y(act.f(z))); }
    ctx.stroke();
    if (zProbe !== null && isFinite(zProbe)) {
      const zc = clamp(zProbe, lo, hi), x = X(zc), y = Y(act.f(zc));
      ctx.strokeStyle = theme.c('ink', 0.35); ctx.lineWidth = 1;
      ctx.beginPath(); ctx.moveTo(x, padT); ctx.lineTo(x, padT + h); ctx.stroke();
      ctx.beginPath(); ctx.arc(x, y, 5, 0, Math.PI * 2); ctx.fillStyle = theme.c('pos'); ctx.fill(); ctx.lineWidth = 2; ctx.strokeStyle = theme.c('panel'); ctx.stroke();
    }
  }

  /* One row per neuron: whisker p5–p95, box p25–p75, median tick. */
  function rangeStrips(canvas, stats, opts = {}) {
    const n = stats.length, rowH = clamp(150 / n, 9, 18), W = Math.max(160, canvas.parentElement.clientWidth), H = n * rowH + 26;
    const ctx = NA.fitCanvas(canvas, W, H);
    const padL = 46, padR = 10, w = W - padL - padR;
    let lo = 0, hi = 0;
    for (const s of stats) { lo = Math.min(lo, s.q[0], s.min); hi = Math.max(hi, s.q[4], s.max); }
    if (opts.range) { lo = opts.range[0]; hi = opts.range[1]; }
    if (hi - lo < 1e-6) hi = lo + 1;
    const X = (v) => padL + ((v - lo) / (hi - lo)) * w;
    ctx.clearRect(0, 0, W, H);
    ctx.strokeStyle = theme.c('line'); ctx.lineWidth = 1;
    if (lo < 0 && hi > 0) { const x = Math.round(X(0)) + 0.5; ctx.beginPath(); ctx.moveTo(x, 0); ctx.lineTo(x, n * rowH); ctx.stroke(); }
    ctx.font = `500 10px ${mono()}`;
    stats.forEach((s, j) => {
      const yc = j * rowH + rowH / 2;
      const sel = opts.selected === j;
      ctx.fillStyle = sel ? theme.c('ink') : theme.c('ink-2'); ctx.textAlign = 'right'; ctx.textBaseline = 'middle';
      ctx.fillText(opts.label ? opts.label(j) : String(j + 1), padL - 8, yc);
      const dead = opts.isDead && opts.isDead(j);
      const col = dead ? theme.c('muted') : theme.c('pos');
      ctx.strokeStyle = col; ctx.lineWidth = 1.2;
      ctx.beginPath(); ctx.moveTo(X(s.q[0]), yc); ctx.lineTo(X(s.q[4]), yc); ctx.stroke();
      ctx.fillStyle = dead ? theme.c('muted', 0.5) : theme.c('pos', 0.55);
      const bx = X(s.q[1]), bw = Math.max(2, X(s.q[3]) - bx), bh = Math.min(rowH - 4, 9);
      rr(ctx, bx, yc - bh / 2, bw, bh, 2); ctx.fill();
      ctx.fillStyle = theme.c('ink'); ctx.fillRect(X(s.q[2]) - 1, yc - bh / 2 - 1, 2, bh + 2);
      if (sel) { ctx.strokeStyle = theme.c('focus'); ctx.lineWidth = 1; ctx.strokeRect(padL - 40, j * rowH + 1, W - padL + 38, rowH - 2); }
    });
    ctx.fillStyle = theme.c('muted'); ctx.textBaseline = 'top';
    ctx.textAlign = 'left'; ctx.fillText(fmt(lo, 2), padL, n * rowH + 8);
    ctx.textAlign = 'right'; ctx.fillText(fmt(hi, 2), padL + w, n * rowH + 8);
    return { rowH };
  }

  /* Hinton diagram: square area ∝ |w|, colour = sign. rows = neurons, cols = inputs. */
  function hinton(canvas, W, rows, cols, opts = {}) {
    const avail = Math.max(160, canvas.parentElement.clientWidth);
    const cell = clamp(Math.floor((avail - 40) / cols), 6, 26), w = cols * cell + 40, h = rows * cell + 20;
    const ctx = NA.fitCanvas(canvas, w, h);
    ctx.clearRect(0, 0, w, h);
    const m = NA.maxAbs(W) || 1;
    ctx.font = `500 9px ${mono()}`; ctx.fillStyle = theme.c('muted');
    for (let j = 0; j < rows; j++) {
      ctx.textAlign = 'right'; ctx.textBaseline = 'middle';
      ctx.fillStyle = opts.selected === j ? theme.c('ink') : theme.c('muted');
      ctx.fillText(String(j + 1), 34, 20 + j * cell + cell / 2);
      for (let i = 0; i < cols; i++) {
        const v = W[j * cols + i], s = Math.sqrt(Math.abs(v) / m) * (cell - 2);
        ctx.fillStyle = theme.divColor(Math.sign(v) * (0.35 + 0.65 * Math.abs(v) / m));
        ctx.fillRect(40 + i * cell + (cell - s) / 2, 20 + j * cell + (cell - s) / 2, s, s);
      }
    }
    ctx.fillStyle = theme.c('muted'); ctx.textAlign = 'center'; ctx.textBaseline = 'alphabetic';
    if (opts.colLabel) for (let i = 0; i < cols; i++) if (cols <= 16 || i % 4 === 0) ctx.fillText(opts.colLabel(i), 40 + i * cell + cell / 2, 13);
    if (opts.selected !== undefined && opts.selected !== null) { ctx.strokeStyle = theme.c('focus'); ctx.lineWidth = 1.5; ctx.strokeRect(38.5, 20 + opts.selected * cell - 1.5, cols * cell + 3, cell + 3); }
    return { cell, top: 20, left: 40 };
  }

  NA.charts = { LineChart, histogram, activationPlot, rangeStrips, hinton };
})(window.NA);
