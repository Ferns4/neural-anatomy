/* growth.js — How it learned: the film of a model learning, shared by every model on the site. A recorder keeps
   moments of training spread evenly on a log scale of steps (1, 1.4, 2, 3, 5, 7, 10, 14…), so the quick early
   changes get as many frames as the slow later ones: at each, a picture of what the model does, its numbers, and
   how big its weights are and how far they have moved, layer by layer. The view plays them as a timelapse, lays
   them out as a strip, draws the learning curve with its stages named (warming up, learning fast, polishing,
   memorising, slipping back) and explained where they show, and draws how the insides grew.

   A source tells the view what to show: { unit, frames(), curve(), version(), math?, fromScratch?, scratchLabel? }.
   math(frame, before, first): the math at a moment, as blocks { title, lines: [text | { t, strong, muted }], note }
   (G.math writes the common ones: the loss, the optimiser's rule with that moment's numbers, the change since the
   moment before, and the update rules of the reinforcement learning methods).
   frames(): [{ step, draw(ctx, w, h), m: { name: text }, inside: [{ name, size, moved }] }]; curve():
   { label, higher, fmt, tok, points: [[step, value]], second: { label, tok, points, judge } } (second:
   validation, for memorising, or with judge: false another curve to compare with); sub(frames): the line under
   the title when the moments are not kept on a log scale. version() changes whenever any of them does.
   Recorder makes frames() for models whose pictures are only drawn live (the labs and the builder); the
   network models redraw theirs from the history they keep. */
(function (NA) {
  'use strict';
  const { el, clear, clamp, fmt, fmtPct, fmtInt, theme } = NA;
  const G = {};
  const font = (w, s, mono) => `${w} ${s}px ${NA.getFont(mono ? 'mono' : 'ui')}`;

  // ------------------------------------------------------------------ when to record
  const STEPS = [1, 1.4, 2, 3, 5, 7];
  G.nextMark = function (step) {
    if (step < 1) return 1;
    for (let p = 1; ; p *= 10) for (const m of STEPS) { const v = Math.round(m * p); if (v > step) return v; }
  };
  // a few frames, as evenly spread as possible on a log scale (always the first and the last)
  G.pick = function (frames, n) {
    if (frames.length <= n) return frames.map((f, i) => i);
    const lg = (s) => Math.log10(s + 1), a = lg(frames[0].step), b = lg(frames[frames.length - 1].step), out = [];
    for (let k = 0; k < n; k++) {
      const t = a + ((b - a) * k) / (n - 1);
      let best = 0;
      frames.forEach((f, i) => { if (Math.abs(lg(f.step) - t) < Math.abs(lg(frames[best].step) - t)) best = i; });
      if (!out.includes(best)) out.push(best);
    }
    return out.sort((x, y) => x - y);
  };

  /* Recorder: frames of a model whose picture is drawn live. o: { step(), draw(ctx, w, h), metrics(), inside(),
     thumbs }. tick() after training keeps a frame when the step count has passed the next mark; tick(true) keeps
     the latest whatever it is (training paused or finished). Pictures are kept at 320 × 240: at most 32 of them.
     thumbs: a picture with words in it is drawn a second time, small (draw sees a width under 180 and leaves the
     words out), for the strip, where shrunken words would only be smudges. */
  class Recorder {
    constructor(o) { this.o = o; this.list = []; this.next = 0; this.v = 0; }
    reset() { this.list = []; this.next = 0; this.v++; }
    tick(force) {
      const step = this.o.step();
      if (!force && step < this.next && this.list.length) return false;
      const last = this.list[this.list.length - 1];
      if (last && last.step === step) { if (!force) return false; this.list.pop(); }
      if (last && step < last.step) this.reset(); // a new run started over
      this.list.push(this.capture(step));
      this.next = G.nextMark(step);
      if (this.list.length > 32) this.list = this.list.filter((f, i) => i === 0 || i === this.list.length - 1 || i % 2 === 1);
      this.v++;
      return true;
    }
    capture(step) {
      const shot = (W, H) => {
        const c = document.createElement('canvas');
        c.width = W; c.height = H;
        try { this.o.draw(c.getContext('2d'), W, H); } catch (e) { console.error(e); }
        return c;
      };
      return { step, img: shot(320, 240), thumb: this.o.thumbs ? shot(144, 108) : null, m: this.o.metrics ? this.o.metrics() : {}, inside: this.o.inside ? this.o.inside() : null, x: this.o.numbers ? this.o.numbers() : null };
    }
    frames() { return this.list; }
    version() { return this.v; }
  }
  G.Recorder = Recorder;

  // ------------------------------------------------------------------ the insides
  // the size of a layer's weights (their root mean square) and how far they have moved from a copy taken at the start
  G.sizes = function (layers, start) {
    return layers.map((L, i) => {
      let ss = 0, dd = 0, n = 0;
      L.arrays.forEach((w, k) => { const w0 = start && start[i] && start[i][k]; for (let j = 0; j < w.length; j++) { ss += w[j] * w[j]; if (w0) { const d = w[j] - w0[j]; dd += d * d; } } n += w.length; });
      return { name: L.name, size: n ? Math.sqrt(ss / n) : 0, moved: n && start ? Math.sqrt(dd / n) : 0 };
    });
  };
  G.copyLayers = (layers) => layers.map((L) => L.arrays.map((w) => Float32Array.from(w)));

  // ------------------------------------------------------------------ stages of the learning curve
  /* From the best value so far: warming up until a tenth of the way to the best, learning fast to 80%, then
     polishing. Memorising: the validation curve rises well above its lowest while training goes on (only with a
     loss and its validation). Slipping back: a score falls well below its best and stays there (learning from
     rewards can forget). */
  G.stages = function (c) {
    const P = (c && c.points || []).filter((p) => Number.isFinite(p[1]));
    if (P.length < 4) return [];
    const hi = !!c.higher, v0 = P[0][1], vals = P.map((p) => p[1]), vb = hi ? Math.max(...vals) : Math.min(...vals);
    if (Math.abs(vb - v0) < 1e-9 * (1 + Math.abs(v0))) return [{ kind: 'flat', from: P[0][0], to: P[P.length - 1][0] }];
    let run = v0;
    const prog = P.map((p) => { run = hi ? Math.max(run, p[1]) : Math.min(run, p[1]); return Math.abs(run - v0) / Math.abs(vb - v0); });
    const at = (x) => { const i = prog.findIndex((q) => q >= x); return i < 0 ? P.length - 1 : i; };
    const a = P[at(0.1)][0], b = P[at(0.8)][0], end = P[P.length - 1][0], out = [];
    if (a > P[0][0]) out.push({ kind: 'warm', from: P[0][0], to: a });
    if (b > a) out.push({ kind: 'fast', from: a, to: b });
    if (end > b) out.push({ kind: 'polish', from: b, to: end });
    // memorising: validation's lowest point, then a clear rise
    const S = c.second && c.second.judge !== false && (c.second.points || []).filter((p) => Number.isFinite(p[1]));
    if (!hi && S && S.length > 5) {
      let k = 0;
      S.forEach((p, i) => { if (p[1] < S[k][1]) k = i; });
      const top = Math.max(...S.slice(k).map((p) => p[1])), span = Math.max(1e-9, S[0][1] - S[k][1]);
      if (k < S.length - 2 && top - S[k][1] > Math.max(0.15 * span, 0.08 * S[k][1])) cut(out, { kind: 'memorise', from: S[k][0], to: S[S.length - 1][0] });
    }
    // slipping back: the last fifth of the run well below its best
    if (hi) {
      const tail = P.slice(Math.floor(P.length * 0.8)), mean = tail.reduce((s, p) => s + p[1], 0) / tail.length;
      const iBest = vals.indexOf(vb);
      if (iBest < P.length * 0.8 && vb - mean > 0.3 * Math.abs(vb - v0)) cut(out, { kind: 'slip', from: P[iBest][0], to: end });
    }
    return out;
  };
  // a late stage takes over the ones it overlaps
  function cut(list, s) {
    for (let i = list.length - 1; i >= 0; i--) {
      const q = list[i];
      if (q.from >= s.from) list.splice(i, 1);
      else if (q.to > s.from) q.to = s.from;
    }
    list.push(s);
  }
  G.STAGES = {
    warm: { name: 'Warming up', tok: 's8', say: 'The weights are still close to their random start: the gradient points the way, but the steps have not added up to anything you can see yet.', rl: 'It has not found much reward yet, so there is little to learn from: its moves are still mostly chance.' },
    fast: { name: 'Learning fast', tok: 's1', say: 'Each step now helps a lot: the model finds the main shape of the answer. Most of what it ever learns happens here.', rl: 'It has found the reward and learns where it comes from: the score climbs fast.' },
    polish: { name: 'Polishing', tok: 's2', say: 'The big picture is in place. It refines the details, and each step helps less than the one before.', rl: 'The policy works; what is left is sharpening it, a little at a time.' },
    memorise: { name: 'Memorising', tok: 'neg', say: 'The training loss keeps falling while the validation loss rises: it is learning its own training examples, noise and all, instead of the rule.' },
    slip: { name: 'Slipping back', tok: 'neg', say: 'It was better before: learning from rewards can forget, when new experience pushes its weights away from what worked.' },
    flat: { name: 'No change', tok: 'muted', say: 'The curve has not moved: the learning rate may be too small, or the model cannot learn this as it is.' },
  };

  // ------------------------------------------------------------------ the view
  class View {
    /* root: an element to fill (its height is the view's); src: as at the top; o: { title, what (a sentence on
       what each frame shows), key (to its colours, as G.key takes it), rl (rewards: words for its stages),
       emptyText } */
    constructor(root, src, o = {}) {
      this.root = root; this.src = src; this.o = o; this.sel = -1; this.play = null; this.seen = -1; this.dirty = true;
      root.classList.add('gw');
      this.title = el('h3', { class: 'gw-title', text: o.title || 'How it learned' });
      this.sub = el('p', { class: 'gw-sub' });
      this.playBtn = el('button', { class: 'btn btn-primary', type: 'button', onclick: () => this.toggle() }, '▶ Play');
      this.scratchBtn = src.fromScratch ? el('button', { class: 'btn', type: 'button', title: 'Forget what it learned and train it again from new random weights, recording as it goes', onclick: () => { this.stop(); src.fromScratch(); this.sel = -1; this.dirty = true; } }, src.scratchLabel || 'Learn again from scratch') : null;
      this.big = el('canvas', { class: 'gw-big', role: 'img' });
      this.cap = el('div', { class: 'gw-cap', 'aria-live': 'polite' });
      this.scrub = el('input', { class: 'gw-scrub', type: 'range', min: '0', max: '0', step: '1', value: '0', 'aria-label': 'Moment of training' });
      this.scrub.addEventListener('input', () => { this.stop(); this.select(+this.scrub.value); });
      this.strip = el('div', { class: 'gw-strip', role: 'list', 'aria-label': 'Moments of training' });
      this.curve = el('canvas', { class: 'gw-curve', role: 'img' });
      this.curve.addEventListener('click', (e) => this.clickCurve(e));
      this.stageKey = el('div', { class: 'gw-stages' });
      this.inside = el('canvas', { class: 'gw-inside', role: 'img' });
      this.insideNote = el('p', { class: 'gw-note' });
      this.empty = el('p', { class: 'gw-empty', hidden: '' });
      this.mathBody = el('div', { class: 'gw-mathbody' });
      this.mathSec = el('section', { class: 'gw-sec gw-math' }, el('h4', { class: 'gw-k', text: 'The math at this moment' }), this.mathBody);
      this.body = el('div', { class: 'gw-body' },
        el('div', { class: 'gw-main' }, this.big, el('div', { class: 'gw-side' }, o.key ? G.key(o.key) : null, this.cap, this.scrub)),
        this.strip,
        this.mathSec,
        el('section', { class: 'gw-sec' }, el('h4', { class: 'gw-k', text: 'The learning curve' }), this.curve, this.curveKey = el('p', { class: 'gw-note gw-ckey' }), this.stageKey),
        this.insideSec = el('section', { class: 'gw-sec' }, el('h4', { class: 'gw-k', text: 'Inside it: how its weights grew' }), this.inside, this.insideNote));
      root.append(el('div', { class: 'gw-head' }, el('div', { class: 'gw-titles' }, this.title, this.sub), el('div', { class: 'gw-acts' }, this.playBtn, this.scratchBtn)), this.empty, this.body);
      this.ro = new ResizeObserver(() => { this.dirty = true; });
      this.ro.observe(root);
      theme.onChange(() => { this.dirty = true; });
    }
    get frames() { return this.src.frames() || []; }
    select(i) {
      const F = this.frames;
      if (!F.length) return;
      this.sel = clamp(i, 0, F.length - 1);
      this.scrub.value = String(this.sel);
      this.drawBig(); this.drawCurve(); this.drawInside(); this.markStrip();
    }
    toggle() { if (this.play) this.stop(); else this.start(); }
    start() {
      const F = this.frames;
      if (F.length < 2) return;
      if (this.sel < 0 || this.sel >= F.length - 1) this.select(0);
      this.play = { at: performance.now() };
      this.playBtn.textContent = '❚❚ Pause';
      // it plays by itself, wherever it is shown, and stops when it is hidden
      const loop = (t) => { if (!this.play) return; if (!this.root.offsetParent) { this.stop(); return; } this.frame(t); requestAnimationFrame(loop); };
      requestAnimationFrame(loop);
    }
    stop() { if (!this.play) return; this.play = null; this.playBtn.textContent = '▶ Play'; }
    frame(now) {
      if (!this.root.offsetParent) return;
      const v = this.src.version();
      if (v !== this.seen) { this.seen = v; this.dirty = true; }
      if (this.play && now - this.play.at > (this.sel === 0 ? 900 : 420)) {
        this.play.at = now;
        if (this.sel >= this.frames.length - 1) this.stop(); else this.select(this.sel + 1);
      }
      if (this.dirty) { this.dirty = false; this.render(); }
    }
    render() {
      const F = this.frames, c = this.src.curve ? this.src.curve() : null, unit = this.src.unit || 'step';
      this.root.classList.toggle('is-narrow', this.root.clientWidth < 600);
      this.empty.hidden = F.length > 1;
      this.body.hidden = F.length < 2;
      this.playBtn.disabled = F.length < 2;
      if (F.length < 2) {
        this.empty.textContent = this.o.emptyText || 'Train it, and the moments of its learning collect here: what it does after 1, 2, 5, 10, 20… steps.';
        this.sub.textContent = '';
        return;
      }
      if (this.sel < 0 || this.sel >= F.length || (!this.play && this.wasLast)) this.sel = F.length - 1;
      this.wasLast = this.sel === F.length - 1;
      this.scrub.max = String(F.length - 1); this.scrub.value = String(this.sel);
      this.sub.textContent = this.src.sub ? this.src.sub(F) : `${F.length} moments, from ${unit} ${fmtInt(F[0].step)} to ${unit} ${fmtInt(F[F.length - 1].step)}, spaced evenly on a log scale: the early ones are close together because that is when it changes fastest.`;
      this.stages = G.stages(c);
      // which line is which: the curve has no other label
      if (c) clear(this.curveKey).append(...G.keyItems([{ tok: c.tok || 's1', label: c.label }, c.second ? { tok: c.second.tok || 's2', label: c.second.label, dash: true } : null].filter(Boolean)));
      this.renderStrip(F);
      this.drawBig(); this.drawCurve(); this.drawInside(); this.markStrip();
      const rl = !!this.o.rl;
      clear(this.stageKey).append(...(this.stages.length ? this.stages : []).map((s) => {
        const S = G.STAGES[s.kind];
        return el('p', { class: 'gw-stage' }, el('i', { class: 'gw-swatch', style: { background: `var(--${S.tok})` } }), el('b', { text: `${S.name}` }), ` (${unit}s ${fmtInt(s.from)} to ${fmtInt(s.to)}): ${(rl && S.rl) || S.say}`);
      }));
    }
    renderStrip(F) {
      const idx = G.pick(F, 8), key = idx.map((i) => F[i].step).join(',') + ':' + this.seen + ':' + (theme.dark ? 1 : 0);
      if (this.stripKey === key) return;
      this.stripKey = key;
      clear(this.strip);
      this.thumbs = idx.map((i) => {
        const f = F[i], c = el('canvas', { 'aria-hidden': 'true' }), b = el('button', { class: 'gw-thumb', type: 'button', role: 'listitem', 'aria-label': `${this.src.unit || 'step'} ${fmtInt(f.step)}`, onclick: () => { this.stop(); this.select(i); } }, c, el('span', { class: 'gw-tstep', text: fmtInt(f.step) }));
        this.strip.append(b);
        requestAnimationFrame(() => { const w = c.clientWidth || 96, h = Math.round(w * 0.75); const ctx = NA.fitCanvas(c, w, h); paint(ctx, f, w, h); });
        return { i, b };
      });
    }
    markStrip() { for (const t of this.thumbs || []) t.b.classList.toggle('is-on', t.i === this.sel); }
    drawBig() {
      const F = this.frames, f = F[this.sel];
      if (!f) return;
      const w = Math.max(160, this.big.clientWidth || 320), h = Math.round(w * 0.75), ctx = NA.fitCanvas(this.big, w, h);
      paint(ctx, f, w, h);
      const c = this.src.curve ? this.src.curve() : null, unit = this.src.unit || 'step', v = c ? valueAt(c, f.step) : null, first = c ? valueAt(c, F[0].step) : null;
      const st = (this.stages || []).find((s) => f.step >= s.from && f.step <= s.to);
      const lines = [el('p', { class: 'gw-when' }, el('b', { text: `${unit[0].toUpperCase()}${unit.slice(1)} ${fmtInt(f.step)}` }), ` of ${fmtInt(F[F.length - 1].step)}`)];
      if (c && v !== null) lines.push(el('p', {}, `${c.label}: `, el('b', { class: 'mono', text: c.fmt(v) }), first !== null && this.sel > 0 ? ` (${c.fmt(first)} at the start)` : ''));
      for (const [k, val] of Object.entries(f.m || {})) if (!c || k !== c.label) lines.push(el('p', {}, `${k}: `, el('b', { class: 'mono', text: val })));
      if (st) lines.push(el('p', { class: 'gw-stage-now' }, el('i', { class: 'gw-swatch', style: { background: `var(--${G.STAGES[st.kind].tok})` } }), G.STAGES[st.kind].name));
      if (f.inside && f.inside.length && F[0].inside) {
        const moved = f.inside.reduce((s, q) => s + q.moved, 0) / f.inside.length, most = f.inside.reduce((a, q) => (q.moved > a.moved ? q : a), f.inside[0]);
        if (moved > 0) lines.push(el('p', { class: 'gw-small', text: `Its weights have moved ${fmt(moved, moved < 0.01 ? 4 : 3)} on average from where they started; ${most.name} the most.` }));
      }
      clear(this.cap).append(...lines);
      // the math: the loss, the rule, the numbers of this moment
      const M = this.src.math ? this.src.math(f, this.sel > 0 ? F[this.sel - 1] : null, F[0]) : null;
      this.mathSec.hidden = !M || !M.length;
      if (M && M.length) clear(this.mathBody).append(...M.filter(Boolean).map((b) => el('div', { class: 'gw-mblock' }, b.title ? el('p', { class: 'gw-mtitle', text: b.title }) : null,
        el('div', { class: 'fx' }, ...b.lines.filter((l) => l !== null && l !== undefined && l !== '').map((l) => el('div', { class: 'fx-line' + (l.strong ? ' fx-strong' : '') + (l.muted ? ' fx-muted' : ''), text: typeof l === 'string' ? l : l.t }))),
        b.note ? el('p', { class: 'fx-note', text: b.note }) : null)));
      this.big.setAttribute('aria-label', `${this.o.what || 'What it does'} at ${unit} ${fmtInt(f.step)}`);
    }
    // the curve: log steps across; its stages as bands; a tick for every frame, a line for the one shown
    drawCurve() {
      const c = this.src.curve ? this.src.curve() : null, F = this.frames;
      const w = Math.max(200, this.curve.clientWidth || 300), h = 150, ctx = NA.fitCanvas(this.curve, w, h);
      ctx.clearRect(0, 0, w, h);
      if (!c || !c.points.length) { this.curve.parentElement.hidden = true; return; }
      this.curve.parentElement.hidden = false;
      const P = c.points.filter((p) => Number.isFinite(p[1])), S = c.second ? c.second.points.filter((p) => Number.isFinite(p[1])) : [];
      const all = [...P, ...S].map((p) => p[1]), lo0 = Math.min(...all), hi0 = Math.max(...all);
      const logY = !c.higher && lo0 > 0 && hi0 / lo0 > 30;
      const ty = (v) => (logY ? Math.log10(v) : v), lo = ty(lo0), hi = ty(hi0) + (ty(hi0) === lo ? 1 : 0);
      const pad = { l: 44, r: 10, t: 10, b: 22 }, maxS = Math.max(1, F.length ? F[F.length - 1].step : P[P.length - 1][0], P[P.length - 1][0]);
      const X = (s) => pad.l + (Math.log10(s + 1) / Math.log10(maxS + 1)) * (w - pad.l - pad.r), Y = (v) => pad.t + (1 - (ty(v) - lo) / (hi - lo)) * (h - pad.t - pad.b);
      this.cg = { X, pad, w };
      for (const s of this.stages || []) { ctx.fillStyle = theme.c(G.STAGES[s.kind].tok, 0.1); ctx.fillRect(X(s.from), pad.t, Math.max(1, X(s.to) - X(s.from)), h - pad.t - pad.b); }
      ctx.strokeStyle = theme.c('line'); ctx.lineWidth = 1; ctx.font = font(500, 10, true); ctx.fillStyle = theme.c('muted');
      ctx.textAlign = 'right'; ctx.textBaseline = 'middle';
      for (const v of [lo0, hi0]) { const y = Math.round(Y(v)) + 0.5; ctx.beginPath(); ctx.moveTo(pad.l, y); ctx.lineTo(w - pad.r, y); ctx.stroke(); ctx.fillText(c.fmt(v), pad.l - 5, y); }
      ctx.textAlign = 'center'; ctx.textBaseline = 'top';
      for (let p = 1; p <= maxS; p *= 10) ctx.fillText(NA.fmtCompact(p), X(p), h - pad.b + 6);
      const line = (pts, tok, wd, dash) => { ctx.strokeStyle = theme.c(tok); ctx.lineWidth = wd; ctx.setLineDash(dash || []); ctx.beginPath(); pts.forEach((p, i) => (i ? ctx.lineTo(X(p[0]), Y(p[1])) : ctx.moveTo(X(p[0]), Y(p[1])))); ctx.stroke(); ctx.setLineDash([]); };
      if (S.length) line(S, c.second.tok || 's2', 1.5, [4, 3]);
      line(P, c.tok || 's1', 2);
      ctx.fillStyle = theme.c('ink-2');
      for (const f of F) ctx.fillRect(Math.round(X(f.step)) - 0.5, h - pad.b - 4, 1, 4);
      const f = F[this.sel];
      if (f) { const x = Math.round(X(f.step)) + 0.5; ctx.strokeStyle = theme.c('ink'); ctx.lineWidth = 1.5; ctx.beginPath(); ctx.moveTo(x, pad.t); ctx.lineTo(x, h - pad.b); ctx.stroke(); }
      this.curve.setAttribute('aria-label', `${c.label} over ${this.src.unit || 'step'}s, on a log scale${c.second ? `, with ${c.second.label} dashed` : ''}: ${c.fmt(P[0][1])} at the start, ${c.fmt(P[P.length - 1][1])} now`);
    }
    clickCurve(e) {
      if (!this.cg) return;
      const r = this.curve.getBoundingClientRect(), x = e.clientX - r.left, F = this.frames;
      let best = 0;
      F.forEach((f, i) => { if (Math.abs(this.cg.X(f.step) - x) < Math.abs(this.cg.X(F[best].step) - x)) best = i; });
      this.stop(); this.select(best);
    }
    // how big each layer's weights are, and how far they have moved from the start, over the same log steps
    drawInside() {
      const F = this.frames.filter((f) => f.inside && f.inside.length);
      this.insideSec.hidden = F.length < 2;
      if (F.length < 2) return;
      const w = Math.max(200, this.inside.clientWidth || 300), half = w < 460 ? w : (w - 16) / 2, rows = w < 460 ? 2 : 1, ph = 112, h = rows * ph + (rows - 1) * 12, ctx = NA.fitCanvas(this.inside, w, h);
      ctx.clearRect(0, 0, w, h);
      const names = F[F.length - 1].inside.map((q) => q.name), maxS = F[F.length - 1].step, cols = ['s1', 's2', 's4', 's7', 's3', 's5', 's6', 's8'];
      const panel = (x0, y0, key, label) => {
        const pad = { l: 40, r: 8, t: 16, b: 18 }, vals = F.flatMap((f) => f.inside.map((q) => q[key])), top = Math.max(1e-9, ...vals);
        const X = (s) => x0 + pad.l + (Math.log10(s + 1) / Math.log10(maxS + 1)) * (half - pad.l - pad.r), Y = (v) => y0 + pad.t + (1 - v / top) * (ph - pad.t - pad.b);
        ctx.font = font(600, 10.5); ctx.fillStyle = theme.c('muted'); ctx.textAlign = 'left'; ctx.textBaseline = 'top'; ctx.fillText(label, x0 + pad.l, y0);
        ctx.strokeStyle = theme.c('line'); ctx.lineWidth = 1; ctx.beginPath(); ctx.moveTo(x0 + pad.l, Y(0) + 0.5); ctx.lineTo(x0 + half - pad.r, Y(0) + 0.5); ctx.stroke();
        ctx.font = font(500, 10, true); ctx.textAlign = 'right'; ctx.textBaseline = 'middle'; ctx.fillText(fmt(top, top < 0.1 ? 3 : 2), x0 + pad.l - 4, Y(top)); ctx.fillText('0', x0 + pad.l - 4, Y(0));
        names.forEach((n, j) => {
          ctx.strokeStyle = theme.c(cols[j % cols.length]); ctx.lineWidth = 1.8; ctx.beginPath();
          F.forEach((f, i) => { const q = f.inside[j]; if (!q) return; const x = X(f.step), y = Y(q[key]); i ? ctx.lineTo(x, y) : ctx.moveTo(x, y); });
          ctx.stroke();
        });
        const f = this.frames[this.sel];
        if (f) { const x = Math.round(X(f.step)) + 0.5; ctx.strokeStyle = theme.c('ink', 0.6); ctx.lineWidth = 1; ctx.beginPath(); ctx.moveTo(x, y0 + pad.t); ctx.lineTo(x, y0 + ph - pad.b); ctx.stroke(); }
      };
      panel(0, 0, 'size', 'Size of its weights (root mean square)');
      if (rows === 1) panel(half + 16, 0, 'moved', 'How far they have moved from the start'); else panel(0, ph + 12, 'moved', 'How far they have moved from the start');
      clear(this.insideNote).append(...names.map((n, j) => el('span', { class: 'gw-lg' }, el('i', { class: 'gw-swatch', style: { background: `var(--${cols[j % cols.length]})` } }), n)),
        el('span', { text: ' Each line is a layer. Training changes the weights most early on, and not every layer moves as much: the one that moves most is where the learning happens.' }));
      this.inside.setAttribute('aria-label', `For each layer (${names.join(', ')}): the size of its weights and how far they have moved from the start, over training`);
    }
  }
  View.prototype.show = function (on) { this.root.hidden = !on; if (on) this.dirty = true; else this.stop(); };
  G.View = View;
  // a lab's Growth view: a stage of its own, after the one given (the architecture's), hidden until shown
  G.mount = function (after, src, o) {
    const stage = el('div', { class: 'lab-stage gw-pane', hidden: '' });
    after.after(stage);
    return new View(stage, src, o);
  };
  // what the info panel under a lab's picture says beside the Growth view
  /* kind: 'rl' (or true: learning from rewards, which can slip back), 'val' (a validation curve beside the
     training one, which shows memorising), 'logged' (trained offline: its film is its training log, a moment
     for every epoch), or none */
  G.info = (unit = 'step', kind = null) => {
    const rl = kind === 'rl' || kind === true, logged = kind === 'logged';
    return {
      kicker: 'Growth view', title: 'How it learned',
      about: [
        logged ? `A film of its training, from the log kept while it trained offline in PyTorch: how it scored at the end of every ${unit}. Press Play to watch them in turn, or pick one from the strip, the slider or the curve.`
          : `A film of its training: what it did after 1, 2, 5, 10, 20… ${unit}s, a moment kept every time the count passes the next mark. On this log scale the early moments are close together, because that is when it changes fastest; press Play to watch them in turn, or pick one from the strip, the slider or the curve.`,
        `The learning curve below names its stages from how far it has come: warming up, learning fast${rl ? ', polishing, and slipping back when a policy forgets what worked' : kind === 'val' ? ', polishing, and memorising when the validation loss turns up while the training loss keeps falling' : ' and polishing'}. Each says what is going on inside.`,
        logged ? 'The math shows the loss it was trained to make smaller and the optimiser’s rule for every step, with the numbers logged at that moment.'
          : 'The math shows the rule every step follows, with the numbers of the moment shown: the loss (or the reward), the learning rate and, where the page has it, the size of the gradient. Inside it: how big each layer’s weights are, and how far they have moved from where they started.',
      ],
    };
  };
  // ------------------------------------------------------------------ keys and small pictures for frames
  // a key to colours: [{ tok, label, dash, dot, box, faint }]; an item without tok is a note in words
  G.keyItems = (items) => items.map((k) => el('span', { class: 'gw-lg' + (k.tok ? '' : ' is-text') },
    k.tok ? el('i', { class: `gw-swatch${k.dot ? ' gw-dot' : k.box ? '' : ' gw-line'}${k.dash ? ' is-dashed' : ''}`, style: k.dot || k.box ? { background: `var(--${k.tok})`, opacity: k.faint ? '0.5' : '' } : { color: `var(--${k.tok})` } }) : null, k.label));
  G.key = (items) => el('p', { class: 'gw-note gw-ckey' }, ...G.keyItems(items));
  /* lines over x on a frame, with a shaded band and points: o { xs, x0, x1, y0, y1, band: [a, b],
     lines: [{ ys, tok, width, dash }], points: [[x, y]], pointTok } */
  G.plot = function (ctx, w, h, o) {
    const pad = 10, X = (x) => pad + ((x - o.x0) / (o.x1 - o.x0)) * (w - 2 * pad), Y = (y) => h - pad - ((clamp(y, o.y0, o.y1) - o.y0) / (o.y1 - o.y0)) * (h - 2 * pad);
    ctx.fillStyle = theme.c('panel-2'); ctx.fillRect(0, 0, w, h);
    if (o.band) { ctx.fillStyle = theme.c('ink', theme.dark ? 0.07 : 0.05); ctx.fillRect(X(o.band[0]), pad, X(o.band[1]) - X(o.band[0]), h - 2 * pad); }
    ctx.strokeStyle = theme.c('line'); ctx.lineWidth = 1; ctx.beginPath(); ctx.moveTo(pad, Math.round(Y(0)) + 0.5); ctx.lineTo(w - pad, Math.round(Y(0)) + 0.5); ctx.stroke();
    for (const L of o.lines) {
      ctx.strokeStyle = theme.c(L.tok); ctx.lineWidth = L.width || 2; ctx.setLineDash(L.dash || []); ctx.beginPath();
      L.ys.forEach((y, i) => (i ? ctx.lineTo(X(o.xs[i]), Y(y)) : ctx.moveTo(X(o.xs[i]), Y(y))));
      ctx.stroke(); ctx.setLineDash([]);
    }
    ctx.fillStyle = theme.c(o.pointTok || 'ink');
    for (const p of o.points || []) { ctx.beginPath(); ctx.arc(X(p[0]), Y(p[1]), 2.6, 0, Math.PI * 2); ctx.fill(); }
  };
  // bars under a title, each a share from 0 to 1: [{ label, value, tok }] (the films made from a training log).
  // In a thumbnail (narrower than 180 px) the bars alone: text that small cannot be read.
  G.bars = function (ctx, w, h, title, bars) {
    const small = w < 180;
    ctx.fillStyle = theme.c('panel-2'); ctx.fillRect(0, 0, w, h);
    if (!small) { ctx.textAlign = 'left'; ctx.textBaseline = 'top'; ctx.font = font(600, 13); ctx.fillStyle = theme.c('ink'); ctx.fillText(title, 12, 12, w - 24); }
    const top = small ? 6 : 44, row = Math.min(40, (h - top - (small ? 6 : 8)) / bars.length), lw = small ? 6 : Math.round(w * 0.46), bw = w - lw - (small ? 6 : 58), bh = row * (small ? 0.6 : 0.4);
    bars.forEach((b, i) => {
      const y = top + i * row, v = clamp(b.value || 0, 0, 1);
      if (!small) { ctx.textBaseline = 'middle'; ctx.font = font(500, 12); ctx.fillStyle = theme.c('ink-2'); ctx.fillText(b.label, 12, y + row / 2, lw - 18); }
      ctx.fillStyle = theme.c('line'); ctx.fillRect(lw, y + (row - bh) / 2, bw, bh);
      ctx.fillStyle = theme.c(b.tok || 's1'); ctx.fillRect(lw, y + (row - bh) / 2, Math.max(1, bw * v), bh);
      if (!small) { ctx.font = font(600, 12, true); ctx.fillStyle = theme.c('ink'); ctx.fillText(fmtPct(v, 0), lw + bw + 8, y + row / 2); }
    });
  };
  // a frame: its own drawing, or its picture fitted in
  function paint(ctx, f, w, h) {
    ctx.clearRect(0, 0, w, h);
    if (f.draw) { f.draw(ctx, w, h); return; }
    const img = f.thumb && w < 180 ? f.thumb : f.img;
    if (img) { const s = Math.min(w / img.width, h / img.height), dw = img.width * s, dh = img.height * s; ctx.imageSmoothingQuality = 'high'; ctx.drawImage(img, (w - dw) / 2, (h - dh) / 2, dw, dh); }
  }
  // the curve's value at a step: the last point at or before it
  function valueAt(c, step) {
    let v = null;
    for (const p of c.points) { if (p[0] > step) break; if (Number.isFinite(p[1])) v = p[1]; }
    return v;
  }

  // ------------------------------------------------------------------ the math
  const N = (v, d = 4) => (v === null || v === undefined || !Number.isFinite(v) ? '—' : fmt(v, d));
  const LOSS = {
    xent: ['L = −(1/N) Σᵢ log pᵢ[yᵢ]', 'Cross-entropy: the average of −log of the probability it gives the right answer. 0 when it is sure and right; it grows without limit when it is sure and wrong.'],
    bce: ['L = −(1/N) Σᵢ [ yᵢ·log pᵢ + (1 − yᵢ)·log(1 − pᵢ) ]', 'Binary cross-entropy: −log of the probability it gives the right one of two classes, averaged.'],
    mse: ['L = (1/N) Σᵢ (ŷᵢ − yᵢ)²', 'Squared error: how far each answer is from the right one, squared and averaged. A miss twice as big counts four times as much.'],
    huber: ['L = (1/N) Σᵢ h(ŷᵢ − yᵢ),  h(e) = ½e² for |e| ≤ δ, else δ·(|e| − ½δ)', 'Huber: squared for small misses, straight for big ones, so one wild example cannot dominate.'],
    mae: ['L = (1/N) Σᵢ |ŷᵢ − yᵢ|', 'Absolute error: how far each answer is from the right one, averaged.'],
    hinge: ['L = (1/N) Σᵢ max(0, 1 − yᵢ·ŷᵢ)', 'Hinge: zero for answers on the right side with a margin of 1; it only pushes on the others.'],
  };
  G.math = {
    // the loss it minimises, and its value now
    loss(id, value, o = {}) {
      const L = LOSS[id] || [o.formula || 'L = the loss', o.note || ''];
      return { title: 'What training makes smaller: the loss', lines: [{ t: o.formula || L[0], strong: true }, `L = ${N(value)}${o.of ? ` ${o.of}` : ''}`], note: o.note || L[1] };
    },
    /* One step of gradient descent with this moment's numbers. o: { opt (sgd, momentum, adam, adamw), lr, mom,
       b2 (Adam's β₂, if not 0.999), grad: the gradient's size (root mean square of ∂L/∂w over the weights, or
       say gradWhat; norm: its length, an L2 norm over all of them), decay, layers: [{ name, grad, ratio }]
       (ratio: one step's change of a layer's weights over their size), extra: one more line } */
    descent(o) {
      const opt = o.opt || 'sgd', eta = o.lr, g = o.grad, lines = [], bars = o.norm ? '‖g‖' : '|g|';
      if (opt === 'momentum') lines.push({ t: 'v ← μ·v + g,   w ← w − η·v      (g = ∂L/∂w)', strong: true });
      else if (opt === 'adam' || opt === 'adamw') lines.push({ t: 'm ← β₁·m + (1 − β₁)·g,   s ← β₂·s + (1 − β₂)·g²', strong: true }, { t: `w ← w − η·m̂ / (√ŝ + ε)${opt === 'adamw' ? ' − η·λ·w' : ''}      (g = ∂L/∂w)`, strong: true });
      else lines.push({ t: 'w ← w − η·g      (g = ∂L/∂w, the gradient: which way each weight raises the loss)', strong: true });
      const nums = [`η = ${N(eta)}`];
      if (opt === 'momentum') nums.push(`μ = ${N(o.mom ?? 0.9, 2)}`);
      if (opt === 'adam' || opt === 'adamw') nums.push('β₁ = 0.9', `β₂ = ${o.b2 ?? 0.999}`);
      if (opt === 'adamw' && o.decay) nums.push(`λ = ${N(o.decay)}`);
      if (g !== null && g !== undefined) nums.push(`${bars} = ${N(g)}${o.gradWhat ? ` (${o.gradWhat})` : ''}`);
      lines.push(nums.join('   '));
      if (o.extra) lines.push(o.extra);
      let note;
      if (g !== null && g !== undefined && Number.isFinite(g) && eta) {
        if (opt === 'sgd') { lines.push(o.norm ? `the whole step, all weights together, has length η·‖g‖ = ${N(eta * g)}` : `one step moves a weight by about η·|g| = ${N(eta * g)}`); note = 'The gradient shrinks as the loss flattens out, so the steps get smaller by themselves near the bottom.'; }
        else if (opt === 'momentum') { lines.push(`in a steady direction a step adds up to about η·${bars}/(1 − μ) = ${N((eta * g) / (1 - (o.mom ?? 0.9)))}`); note = 'Momentum keeps a running sum of past gradients: steps that agree add up, steps that disagree cancel.'; }
        else { lines.push(`each weight moves up to about η = ${N(eta)} a step, whatever the size of its gradient`); note = 'Adam divides each weight’s average gradient by its typical size, so every weight takes steps of about η: big gradients do not take big steps, and small ones are not ignored.'; }
      }
      for (const q of o.layers || []) lines.push({ t: `${q.name.padEnd(12)} ${bars} = ${N(q.grad)}${q.ratio !== undefined && q.ratio !== null ? `   |Δw| / |w| = ${N(q.ratio)}` : ''}`, muted: true });
      return { title: 'One step of training', lines, note };
    },
    // what changed since the moment before
    change(unit, a, b, label = 'L') {
      if (!a || !b || a.v === null || b.v === null || a.v === undefined || b.v === undefined) return null;
      const d = b.v - a.v, n = Math.max(1, b.step - a.step), lines = [`${label}: ${N(a.v)} → ${N(b.v)}   Δ = ${d >= 0 ? '+' : ''}${N(d)}, ${d >= 0 ? '+' : ''}${N(d / n)} a ${unit}`];
      if (a.moved !== undefined && b.moved !== undefined && b.moved !== null) lines.push(`‖w − w₀‖ (how far its weights have moved, root mean square): ${N(a.moved)} → ${N(b.moved)}`);
      return { title: `Since ${unit} ${fmtInt(a.step)} (the moment before)`, lines, note: 'Many small steps add up: each one is tiny, but together they carry the weights from a random start to a solution.' };
    },
    // the rules of learning from rewards, with their numbers
    qlearn(o) { return { title: 'The rule it learns by: Q-learning', lines: [{ t: 'Q(s, a) ← Q(s, a) + α·[ r + γ·maxₐ′ Q(s′, a′) − Q(s, a) ]', strong: true }, `α = ${N(o.alpha, 2)}   γ = ${N(o.gamma, 2)}   ε = ${N(o.eps, 2)} (moves made at random)`, o.extra || null],
      note: 'After every move it nudges its value of that move toward the reward plus the best value it sees from where it lands: values spread back from the goal, a step at a time.' }; },
    dqn(o) { return { title: 'The rule it learns by: DQN', lines: [{ t: 'y = r + γ·Q̄(s′, argmaxₐ′ Q(s′, a′))      (Q̄: a copy, refreshed every 500 updates)', strong: true }, { t: 'L = huber(Q(s, a) − y),   then a gradient step on L', strong: true }, `γ = ${N(o.gamma, 2)}   ε = ${N(o.eps, 2)} (moves made at random now)${o.loss !== undefined ? `   L = ${N(o.loss)}` : ''}`],
      note: 'Its network’s value of a move is pulled toward what the move earned plus the value of where it led. Remembered moves are replayed in random order, so it does not only learn from the last few.' }; },
    pg(o) { return { title: 'The rule it learns by: REINFORCE', lines: [{ t: 'Gₜ = rₜ + γ·rₜ₊₁ + γ²·rₜ₊₂ + …', strong: true }, { t: '∇J ≈ (1/T) Σₜ ∇ log π(aₜ | sₜ) · Ĝₜ      (Ĝ: the returns, normalised)', strong: true }, `γ = ${N(o.gamma, 2)}${o.ret !== undefined ? `   average return now: ${N(o.ret, 3)}` : ''}`],
      note: 'Moves followed by more reward than usual become more likely, the others less: the gradient of the log-probability, weighted by how well things went.' }; },
    a2c(o) { return { title: 'The rule it learns by: actor-critic', lines: [{ t: 'A = r + γ·V(s′) − V(s)      (the advantage: better or worse than expected)', strong: true }, { t: 'L = −log π(a | s)·A + ¼·A² − 0.01·H(π)', strong: true }, `γ = ${N(o.gamma, 2)}${o.ret !== undefined ? `   average return now: ${N(o.ret, 3)}` : ''}`],
      note: 'The critic learns how good each situation is; the actor makes moves that turned out better than the critic expected more likely. The entropy term H keeps it from becoming sure too soon.' }; },
    ppo(o) { return { title: 'The rule it learns by: PPO', lines: [{ t: 'ρ = π(a | s) / π_old(a | s)', strong: true }, { t: 'L = −min( ρ·A, clip(ρ, 0.8, 1.2)·A ) + ¼·(V − R)²', strong: true }, `γ = ${N(o.gamma, 2)}   λ = 0.95 (advantages, GAE)${o.ret !== undefined ? `   average return now: ${N(o.ret, 3)}` : ''}`],
      note: 'Like actor-critic, but each batch of experience is used for 4 passes, and the clip stops any one update from moving the policy more than 20% away from the one that played.' }; },
    ga(o) { return { title: 'The rule it evolves by: a genetic algorithm', lines: [{ t: 'fitness(w) = total reward of its episodes', strong: true }, { t: 'child = mix(parent₁, parent₂);  10% of its weights += σ·N(0, 1)', strong: true }, `population ${o.pop}   σ = ${N(o.sigma, 2)}${o.best !== undefined ? `   best ${N(o.best, 2)}, average ${N(o.mean, 2)}` : ''}`],
      note: 'No gradients at all: the fittest networks are kept and bred, and random mutations try out changes. The good ones survive into the next generation.' }; },
    td(o) { return { title: 'The rule it learns by: temporal difference', lines: [{ t: o.formula || 'V(s) ← V(s) + α·[ V(s′) − V(s) ]      (at the end: the result, 1, 0 or ½)', strong: true }, o.nums || null],
      note: o.note || 'Each position’s value is pulled toward the value of the position after it: the final result flows back, game after game, to the moves that led there.' }; },
  };

  NA.growth = G;
})(window.NA);
