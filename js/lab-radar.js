/* lab-radar.js — the radar lab: synthetic I/Q intercepts of six kinds (receiver noise, a plain pulse, an LFM chirp,
   a Barker-13 code, a Frank code, a Costas code) at any signal-to-noise ratio, named by a small 1-D CNN that runs
   in the browser. Views: the signal itself, its spectrogram, what the network's layers see, its accuracy by SNR
   and the zoomable architecture. Stream plays a live feed of random intercepts with a running score. */
(function (NA) {
  'use strict';
  const { el, clear, clamp, fmt, fmtPct, fmtInt, theme } = NA;
  const H = NA.labHost, UI = NA.labUI, RD = NA.radar;
  const CL = RD.CLASSES, NC = CL.length, N = RD.N;
  const SHORT = ['Noise', 'Pulse', 'LFM', 'Barker', 'Frank', 'Costas'];
  const CCOL = ['muted', 's3', 's7', 's4', 's5', 's8'];
  const ccol = (k, a = 1) => theme.c(CCOL[k], a);
  const dB = (v) => `${v > 0.05 ? '+' : v < -0.05 ? '−' : ''}${fmt(Math.abs(v), Math.abs(v - Math.round(v)) < 0.05 ? 0 : 1)} dB`;
  const us = (n) => fmt(n / 20, 2); // samples → microseconds (20 samples per µs)
  const mhz = (f) => `${f >= 0 ? '+' : '−'}${fmt(Math.abs(f) * 20, 2)} MHz`; // cycles per sample → MHz
  const uf = (w, px) => `${w} ${px}px ${NA.getFont('ui')}`, mf = (w, px) => `${w} ${px}px ${NA.getFont('mono')}`;
  const argmax = (a) => { let k = 0; for (let i = 1; i < a.length; i++) if (a[i] > a[k]) k = i; return k; };
  const maxAbs = (a) => { let m = 0; for (let i = 0; i < a.length; i++) if (Math.abs(a[i]) > m) m = Math.abs(a[i]); return m; };

  const RL = H.transport({
    id: 'radar', screens: { network: 'Classify', data: 'Signal', insp: 'Inspect' },
    labels: { play: 'Stream', title: 'Stream live intercepts, or pause (Space)', step: 'Next signal (S)', reset: 'New random signal (R)' },
    view: 'signal', pick: 2, snr: 8, every: 1000, streamSnr: 'random', version: 0, count: 0,
    show: { clean: false, env: true, law: true, derot: false, specLaw: true },
    live: { right: 0, total: 0, marks: [] }, accSrc: 'eval', test: null,
  });
  RL.fastForward = function () {};
  RL.stateText = function () { return this.training ? 'Live' : 'Paused'; };

  // ------------------------------------------------------------------ signals
  RL.init = function () {
    if (this.net) return;
    this.net = RD.load(NA.radarWeights);
    this.rng = new NA.RNG(20261026);
    this.make(2, 8, 41); // open on a clear chirp: its spectrogram is the most telling
  };
  // A signal of a given kind, SNR and random draw. The same draw at another SNR keeps the waveform and the noise:
  // the SNR slider only changes how strong the pulse is.
  RL.make = function (cls, snr, seed, same) {
    const s = RD.generate(new NA.RNG(seed), { cls, snr }), F = RD.forward(this.net, s.iq), pred = argmax(F.probs);
    this.sig = { seed, cls, snr, s, F, pred, cache: {} };
    if (!same) this.count++;
    this.version++;
    this.dirty = true; this.leftDirty = true; this.inspDirty = true; this.archStale = true; this.infoStale = true;
  };
  RL.next = function () { this.make(this.pick >= 0 ? this.pick : this.rng.int(NC), this.snr, 1 + this.rng.int(2 ** 30)); };
  RL.randomSignal = function () {
    this.pick = -1; this.snr = Math.round(this.rng.uniform(RD.SNR[0], RD.SNR[1]));
    this.syncControls();
    this.make(this.rng.int(NC), this.snr, 1 + this.rng.int(2 ** 30));
  };
  // the slider: the same waveform and the same noise, only the pulse's strength changes
  RL.setSnr = function (v) { this.snr = v; this.make(this.sig.cls, v, this.sig.seed, true); };
  RL.stepOnce = function () { if (this.training) this.pause(); this.next(); };
  RL.reset = function () { if (this.training) this.pause(); this.live = { right: 0, total: 0, marks: [] }; this.randomSignal(); };
  // one intercept of the live stream: a random kind at a random SNR (or the slider's)
  RL.stream = function () {
    const cls = this.rng.int(NC), snr = this.streamSnr === 'slider' ? this.snr : Math.round(this.rng.uniform(RD.SNR[0], RD.SNR[1]) * 2) / 2;
    this.make(cls, snr, 1 + this.rng.int(2 ** 30));
    const ok = this.sig.pred === cls;
    this.live.total++; if (ok) this.live.right++;
    this.live.marks.push({ ok, cls, pred: this.sig.pred, snr });
    if (this.live.marks.length > 24) this.live.marks.shift();
  };
  // derived views of the current signal, computed once each
  RL.memo = function (key, fn) { const c = this.sig.cache; return c[key] || (c[key] = fn()); };
  RL.spec = function () { return this.memo('spec', () => RD.spectrogram(this.sig.s.iq, 32, 4)); };
  // instantaneous frequency: the phase step between neighbours, raw and smoothed over 9 samples (the lag
  // products are summed before taking the angle, which is far steadier than averaging angles)
  RL.ifreq = function () {
    return this.memo('if', () => {
      const iq = this.sig.s.iq, raw = RD.instFreq(iq), sm = new Float32Array(N), re = new Float32Array(N), im = new Float32Array(N);
      for (let t = 1; t < N; t++) { re[t] = iq[t] * iq[t - 1] + iq[N + t] * iq[N + t - 1]; im[t] = iq[N + t] * iq[t - 1] - iq[t] * iq[N + t - 1]; }
      for (let t = 0; t < N; t++) {
        let a = 0, b = 0;
        for (let k = Math.max(1, t - 4); k <= Math.min(N - 1, t + 4); k++) { a += re[k]; b += im[k]; }
        sm[t] = Math.atan2(b, a) / (2 * Math.PI);
      }
      return { raw, sm };
    });
  };
  RL.cam = function () { return this.memo('cam', () => RD.cam(this.net, this.sig.F, this.sig.pred)); };
  // share of the class activation map's positive evidence that lies inside the pulse (steps are 8 samples)
  RL.camShare = function () {
    const c = this.cam(), p = this.sig.s.p;
    let ins = 0, all = 0;
    c.forEach((v, t) => { if (v <= 0) return; all += v; const mid = t * 8 + 4; if (mid >= p.n0 && mid < p.n0 + p.L) ins += v; });
    return all > 0 ? ins / all : 0;
  };

  // ------------------------------------------------------------------ lifecycle
  RL.activate = function (preset, ui) {
    this.ui = ui; this.preset = preset;
    this.init();
    this.ro = UI.readouts(ui.readouts, [{ key: 'n', label: 'Signal', keep: true }, { key: 'sent', label: 'Sent' }, { key: 'said', label: 'Network says' }, { key: 'score', label: 'Live score', keep: true }]);
    this.buildHyper(ui.hyper);
    this.buildLeft(ui.left);
    this.buildCenter(ui.center);
    this.tabs = UI.tabs(ui.right, [{ id: 'overview', label: 'Overview' }, { id: 'train', label: 'Training' }, { id: 'waves', label: 'Waveforms' }, { id: 'net', label: 'Network' }], (id, body) => this.buildTab(id, body));
    this.offTheme = theme.onChange(() => { this.dirty = true; this.leftDirty = true; this.inspDirty = true; this.archStale = true; });
    this.dirty = true; this.leftDirty = true;
  };
  RL.deactivate = function () {
    if (this.offTheme) this.offTheme();
    if (this.resizeObs) this.resizeObs.disconnect();
    if (this.arch) this.arch.destroy();
    this.test = null;
  };
  RL.archText = function () { return '1-D CNN · 2 × 256 I/Q samples → 6 waveforms'; };

  RL.frame = function (now) {
    if (this.training && now - (this.lastStream || 0) >= this.every) { this.lastStream = now; this.stream(); }
    if (this.test && !this.test.done) this.runTestChunk(now);
    if (now - (this.lastRo || 0) > 120) {
      this.lastRo = now;
      const g = this.sig;
      this.ro.set('n', '#' + fmtInt(this.count));
      this.ro.set('sent', g.cls ? `${SHORT[g.cls]} ${dB(g.snr)}` : 'Noise');
      this.ro.set('said', `${SHORT[g.pred]} ${fmtPct(g.F.probs[g.pred], 0)}${g.pred === g.cls ? ' ✓' : ' ✗'}`);
      this.ro.set('score', this.live.total ? `${this.live.right}/${this.live.total}` : '—');
    }
    if (this.dirty && this.canvas.offsetParent) { this.dirty = false; this.draw(); }
    if (this.view === 'arch') {
      if (this.archStale && (!this.training || now - (this.lastArch || 0) > 900)) { this.lastArch = now; this.archStale = false; this.arch.dirty = true; }
      this.arch.frame(now);
    }
    if (this.infoStale && !this.arch.anim && (!this.training || now - (this.lastInfo || 0) > 900)) { this.lastInfo = now; this.infoStale = false; this.showInfo(); }
    if (this.leftDirty && this.ui.left.offsetParent) { this.leftDirty = false; this.renderLeft(); }
    if (this.inspDirty && this.tabs.body.offsetParent && (!this.training || now - (this.lastInsp || 0) > 900)) { this.lastInsp = now; this.inspDirty = false; this.updateTab(); }
  };

  RL.buildHyper = function (root) {
    root.append(
      UI.field('Stream every', UI.select([500, 1000, 2000].map((v) => ({ value: v, label: v < 1000 ? '0.5 s' : `${v / 1000} s` })), this.every, (v) => { this.every = +v; })),
      UI.field('Stream SNR', UI.select([{ value: 'random', label: 'Random, −12 to +20 dB' }, { value: 'slider', label: 'The slider’s value' }], this.streamSnr, (v) => { this.streamSnr = v; })),
      el('div', { class: 'sheet-actions' },
        el('button', { class: 'btn', type: 'button', onclick: () => { NA.mobile && NA.mobile.sheet(false); this.stepOnce(); } }, 'Next signal'),
        el('button', { class: 'btn', type: 'button', onclick: () => { NA.mobile && NA.mobile.sheet(false); this.reset(); } }, 'Random signal')));
  };

  // ------------------------------------------------------------------ left: the signal
  RL.buildLeft = function (root) {
    const body = UI.panel(root, 'Signal');
    this.tiles = el('div', { class: 'ds-tiles', role: 'radiogroup', 'aria-label': 'Waveform to send' });
    for (const k of [-1, 0, 1, 2, 3, 4, 5]) {
      const c = el('canvas', { 'aria-hidden': 'true' });
      const b = el('button', { class: 'ds-tile', type: 'button', role: 'radio', 'aria-checked': String(k === this.pick), title: k < 0 ? 'A random waveform each time' : CL[k], onclick: () => this.choose(k) }, c, k < 0 ? 'Random' : CL[k]);
      b._c = c; b._k = k;
      this.tiles.append(b);
    }
    this.snrSlider = UI.slider({ label: 'Signal-to-noise ratio', min: -15, max: 25, step: 1, value: this.snr, format: dB,
      onInput: (v) => this.setSnr(v) });
    this.sentBox = el('div', { class: 'fx' });
    body.append(
      UI.section('Waveform', UI.note('What the emitter sends. The receiver gets it with noise, at an unknown time and carrier offset.'), this.tiles),
      UI.section('Strength', this.snrSlider, UI.note('Power of the pulse against the noise in each sample. At 0 dB they are equal; at −10 dB the noise is 10 times stronger.')),
      UI.section(null, el('div', { class: 'btn-row' },
        el('button', { class: 'btn btn-primary', type: 'button', title: 'Same waveform and SNR, a new random draw (S)', onclick: () => this.stepOnce() }, 'New signal'),
        el('button', { class: 'btn', type: 'button', title: 'A random waveform at a random SNR (R)', onclick: () => this.reset() }, 'Random'))),
      UI.section('What was sent', this.sentBox),
      UI.section('How it works',
        el('div', { class: 'fx' },
          el('div', { class: 'fx-line fx-strong', text: 'x[n] = A·e^{jφ[n]} + w[n],  n = 0 … 255' }),
          el('div', { class: 'fx-line', text: 'I = Re x,  Q = Im x,  A² = SNR,  w: noise of power 1' }),
          el('div', { class: 'fx-line fx-muted', text: 'pulse   φ = 2π·f₀·n + φ₀' }),
          el('div', { class: 'fx-line fx-muted', text: 'chirp   f = f₀ ± B·(n/L − ½)' }),
          el('div', { class: 'fx-line fx-muted', text: 'Barker  + π where the code is −1' }),
          el('div', { class: 'fx-line fx-muted', text: 'Frank   + 2π·i·j/M on chip (i, j)' }),
          el('div', { class: 'fx-line fx-muted', text: 'Costas  f hops over 6 tones' })),
        UI.note('The window is scaled to an average power of 1 before the network sees it, so loudness alone gives nothing away: it has to recognise the pattern.')));
  };
  RL.choose = function (k) {
    this.pick = k;
    NA.$$('.ds-tile', this.tiles).forEach((x) => x.setAttribute('aria-checked', String(x._k === k)));
    this.next();
  };
  RL.syncControls = function () {
    if (this.snrSlider) this.snrSlider.set(this.snr);
    if (this.tiles) NA.$$('.ds-tile', this.tiles).forEach((x) => x.setAttribute('aria-checked', String(x._k === this.pick)));
  };
  RL.renderLeft = function () {
    for (const b of NA.$$('.ds-tile', this.tiles)) { const ctx = NA.fitCanvas(b._c, 46, 46); tileGlyph(ctx, b._k, 46, 46); }
    const g = this.sig, p = g.s.p;
    const lines = g.cls ? [
      [`${CL[g.cls]} at ${dB(g.snr)}`, true],
      [`pulse: ${us(p.L)} µs (${p.L} samples), from ${us(p.n0)} µs`],
      [`carrier offset ${mhz(p.f0)}`],
      ...(g.cls === 2 ? [[`${p.up > 0 ? 'up' : 'down'}-chirp over ${fmt(p.bw * 20, 2)} MHz`]] : []),
      ...(g.cls === 3 ? [[`13 chips of ${us(p.L / 13)} µs: + + + + + − − + + − + − +`]] : []),
      ...(g.cls === 4 ? [[`${p.M} × ${p.M} = ${p.M * p.M} chips of ${us(p.L / (p.M * p.M))} µs`]] : []),
      ...(g.cls === 5 ? [[`6 hops of ${us(p.L / 6)} µs, ${fmt((6 / p.L) * 20, 2)} MHz apart`], [`tone order ${p.costas.join(' ')}`]] : []),
    ] : [['Noise only', true], ['no pulse: just the receiver’s own noise']];
    clear(this.sentBox).append(...lines.map(([t, strong]) => el('div', { class: 'fx-line' + (strong ? ' fx-strong' : ''), text: t })));
  };
  // Tile icons: little time–frequency sketches of each waveform.
  function tileGlyph(ctx, k, W, Hh) {
    ctx.clearRect(0, 0, W, Hh);
    ctx.fillStyle = theme.c('panel-2'); NA.rr(ctx, 1, 1, W - 2, Hh - 2, 6); ctx.fill();
    const c = (a) => theme.c(k < 0 ? 'ink-2' : CCOL[k], a), x0 = W * 0.16, x1 = W * 0.84, w = x1 - x0;
    ctx.lineCap = 'round';
    if (k < 0) { // a die: random
      ctx.strokeStyle = c(0.9); ctx.lineWidth = 1.6; NA.rr(ctx, W * 0.24, Hh * 0.24, W * 0.52, Hh * 0.52, 5); ctx.stroke();
      ctx.fillStyle = c(0.9);
      for (const [u, v] of [[0.37, 0.37], [0.63, 0.63], [0.5, 0.5], [0.63, 0.37], [0.37, 0.63]]) { ctx.beginPath(); ctx.arc(W * u, Hh * v, 2.1, 0, Math.PI * 2); ctx.fill(); }
    } else if (k === 0) {
      const rng = new NA.RNG(7);
      for (let i = 0; i < 70; i++) { ctx.fillStyle = theme.c('ink-2', 0.18 + 0.45 * rng.next()); ctx.fillRect(5 + rng.next() * (W - 10), 5 + rng.next() * (Hh - 10), 1.7, 1.7); }
    } else if (k === 1) { ctx.strokeStyle = c(0.95); ctx.lineWidth = 3.5; ctx.beginPath(); ctx.moveTo(x0, Hh * 0.5); ctx.lineTo(x1, Hh * 0.5); ctx.stroke(); }
    else if (k === 2) { ctx.strokeStyle = c(0.95); ctx.lineWidth = 3.5; ctx.beginPath(); ctx.moveTo(x0, Hh * 0.76); ctx.lineTo(x1, Hh * 0.24); ctx.stroke(); }
    else if (k === 3) { const cw = w / 13; RD.BARKER.forEach((b, i) => { ctx.fillStyle = c(b > 0 ? 0.95 : 0.3); ctx.fillRect(x0 + i * cw + 0.4, Hh * 0.38, cw - 0.8, Hh * 0.24); }); }
    else if (k === 4) { const M = 4, cw = w / (M * M); for (let i = 0; i < M; i++) for (let j = 0; j < M; j++) { ctx.fillStyle = c(0.25 + (0.7 * ((i * j) % M)) / (M - 1)); ctx.fillRect(x0 + (i * M + j) * cw + 0.3, Hh * 0.38, cw - 0.6, Hh * 0.24); } }
    else if (k === 5) { const a = [3, 2, 6, 4, 5, 1], cw = w / 6; ctx.strokeStyle = c(0.95); ctx.lineWidth = 3.2; a.forEach((v, i) => { const y = Hh * (0.82 - (v - 1) * 0.12); ctx.beginPath(); ctx.moveTo(x0 + i * cw + 1.5, y); ctx.lineTo(x0 + (i + 1) * cw - 1.5, y); ctx.stroke(); }); }
  }
  RL.tileGlyph = tileGlyph;

  // ------------------------------------------------------------------ centre
  const VIEWS = [
    { value: 'signal', label: 'Signal', title: 'The I/Q samples, the instantaneous frequency and the I/Q plane' },
    { value: 'spec', label: 'Spectrogram', title: 'Power over time and frequency' },
    { value: 'cnn', label: 'What the CNN sees', title: 'The first and last layers’ feature maps and where the evidence came from' },
    { value: 'acc', label: 'Accuracy', title: 'How often the network is right at each SNR, and where its mistakes go' },
    { value: 'arch', label: 'Architecture', title: 'The model’s parts: click or zoom in to open them' },
  ];
  RL.buildCenter = function (root) {
    this.viewSeg = UI.seg(VIEWS, this.view, (v) => this.setView(v), 'View');
    this.chipSets = {
      signal: el('div', { class: 'chip-row' },
        UI.chip('Clean pulse', this.show.clean, (on) => { this.show.clean = on; this.dirty = true; }, 'Overlay the pulse as sent, before the noise'),
        UI.chip('True frequency', this.show.law, (on) => { this.show.law = on; this.dirty = true; }, 'The waveform’s frequency law, dashed'),
        UI.chip('Remove carrier', this.show.derot, (on) => { this.show.derot = on; this.dirty = true; this.infoStale = true; }, 'Undo the carrier offset in the I/Q plane (using the true offset, which the network never gets)')),
      spec: el('div', { class: 'chip-row' }, UI.chip('True frequency', this.show.specLaw, (on) => { this.show.specLaw = on; this.dirty = true; }, 'The waveform’s frequency law, dashed')),
      cnn: el('div', { class: 'chip-row' }),
      acc: el('div', { class: 'chip-row' },
        this.testBtn = el('button', { class: 'btn btn-quiet', type: 'button', title: 'Run 100 fresh signals of each kind at the slider’s SNR through the network (T)', onclick: () => this.runTest() }, 'Test now'),
        this.srcSeg = UI.seg([{ value: 'eval', label: 'After training', title: '400 signals of each kind at each SNR, measured after training' }, { value: 'live', label: 'Your test', title: 'Your test in this browser' }], this.accSrc, (v) => { if (v === 'live' && !this.test) { this.srcSeg.set('eval'); return; } this.accSrc = v; this.dirty = true; this.infoStale = true; }, 'Which confusion matrix')),
      arch: el('div', { class: 'chip-row' }),
    };
    root.append(el('div', { class: 'lab-toolbar' }, this.viewSeg, ...Object.values(this.chipSets)));
    this.stage = el('div', { class: 'lab-stage' });
    this.canvas = el('canvas', { 'aria-label': 'The radar signal and the network’s verdict' });
    this.stage.append(this.canvas);
    this.archStage = el('div', { class: 'lab-stage arch-stage' });
    this.arch = new NA.ArchView(this.archStage, { label: 'Radar classifier architecture: click a part or zoom in to open it', kicker: 'Architecture',
      build: (size) => NA.archRadar.build(this, size), onFocus: (node, path) => { if (this.view === 'arch') this.info.set(this.arch.infoFor(node, path)); } });
    root.append(this.stage, this.archStage);
    // the explanation starts open only when the panel is tall enough to leave the pictures room
    this.info = UI.infoPanel(root, { open: !(NA.mobile && NA.mobile.mobile) && root.clientHeight >= 680 });
    this.foot = el('div', { class: 'lab-foot' });
    root.append(this.foot);
    this.canvas.addEventListener('pointermove', (e) => this.hover(e));
    this.canvas.addEventListener('pointerleave', () => { this.hoverT = null; NA.tooltip.hide(); this.dirty = true; });
    this.resizeObs = new ResizeObserver(() => { this.dirty = true; });
    this.resizeObs.observe(this.stage);
    this.setView(this.view);
  };
  RL.setView = function (v) {
    this.view = v;
    this.viewSeg.set(v);
    const arch = v === 'arch';
    this.stage.hidden = arch; this.archStage.hidden = !arch;
    for (const [k, node] of Object.entries(this.chipSets)) node.hidden = k !== v || !node.childElementCount;
    this.renderFoot();
    this.dirty = true; this.archStale = true; this.hoverT = null;
    this.showInfo();
  };
  RL.renderFoot = function () {
    const item = (col, text, line) => el('span', { class: 'lg-item' }, el('i', { class: line ? 'key-line' : 'dot', style: { background: col } }), text);
    const v = this.view, f = clear(this.foot);
    this.foot.hidden = v === 'arch';
    if (v === 'signal') f.append(item('var(--s1)', 'I', true), item('var(--s2)', 'Q', true), item('var(--ink)', 'envelope |x|', true), item('var(--s7)', 'frequency', true), el('span', { class: 'lg-item', text: 'shaded: where the pulse is' }));
    else if (v === 'spec') f.append(el('span', { class: 'lg-item', text: 'brighter: more power, over the top 30 dB' }), el('span', { class: 'lg-item', text: 'dashed: the frequency law as sent' }), el('span', { class: 'lg-item', text: 'bar under the time axis: the pulse' }));
    else if (v === 'cnn') f.append(el('span', { class: 'lg-item', text: 'darker: zero after ReLU' }), item(ccol(this.sig.pred), 'evidence for the answer', false), item('var(--muted)', 'against it', false));
    else if (v === 'acc') f.append(item('var(--ink)', 'average', true), ...CL.map((c, k) => item(ccol(k), c, true)));
  };
  RL.hover = function (e) {
    const L = this.lay;
    if (!L) return;
    const r = this.canvas.getBoundingClientRect(), x = e.clientX - r.left, y = e.clientY - r.top;
    if (L.time && (this.view === 'signal' || this.view === 'cnn' || this.view === 'spec')) {
      const tr = L.time, t = Math.round(((x - tr.x) / tr.w) * tr.span + tr.t0);
      if (x < tr.x || x > tr.x + tr.w || y < L.top || y > L.bottom || t < 0 || t >= N) { if (this.hoverT !== null) { this.hoverT = null; NA.tooltip.hide(); this.dirty = true; } return; }
      this.hoverT = t; this.dirty = true;
      const s = this.sig.s, p = s.p, inside = this.sig.cls && t >= p.n0 && t < p.n0 + p.L, f = this.ifreq();
      NA.tooltip.show([el('div', { class: 'tt-title', text: `t = ${us(t)} µs (sample ${t})` }),
        el('div', { class: 'tt-row' }, el('span', { class: 'tt-v', text: `${fmt(s.iq[t], 3)}, ${fmt(s.iq[N + t], 3)}` }), el('span', { class: 'tt-k', text: 'I, Q' })),
        el('div', { class: 'tt-row' }, el('span', { class: 'tt-v', text: `${fmt(f.sm[t] * 20, 2)} MHz` }), el('span', { class: 'tt-k', text: 'frequency, smoothed' })),
        el('div', { class: 'tt-hint', text: inside ? 'inside the pulse' : this.sig.cls ? 'outside the pulse: noise only' : 'noise only' })], e.clientX, e.clientY);
    } else if (L.acc && this.view === 'acc') {
      const a = L.acc, s = Math.round((a.s0 + ((x - a.x) / a.w) * (a.s1 - a.s0)) / 2) * 2;
      if (x < a.x || x > a.x + a.w || y < a.y || y > a.y + a.h) { NA.tooltip.hide(); return; }
      const W = NA.radarWeights, i = W.snrs.indexOf(clamp(s, W.snrs[0], W.snrs[W.snrs.length - 1]));
      if (i < 0) return;
      NA.tooltip.show([el('div', { class: 'tt-title', text: `${dB(W.snrs[i])}: ${fmtPct(W.acc[i].reduce((p, v) => p + v, 0) / NC, 1)} on average` }),
        ...CL.map((c, k) => el('div', { class: 'tt-row' }, el('span', { class: 'tt-v', text: fmtPct(W.acc[i][k], 0) }), el('span', { class: 'tt-k' }, el('i', { class: 'key-line', style: { background: ccol(k) } }), c)))], e.clientX, e.clientY);
    }
  };

  // ------------------------------------------------------------------ drawing
  const title = (ctx, r, text) => { ctx.font = uf(600, 10.5); ctx.fillStyle = theme.c('muted'); ctx.textAlign = 'left'; ctx.textBaseline = 'alphabetic'; ctx.fillText(text.toUpperCase(), r.x, r.y - 7); };
  const frame = (ctx, r) => { ctx.strokeStyle = theme.c('line'); ctx.lineWidth = 1; ctx.strokeRect(r.x + 0.5, r.y + 0.5, r.w - 1, r.h - 1); };
  const shadePulse = (ctx, r, X, p, label) => {
    ctx.fillStyle = theme.c('ink', theme.dark ? 0.07 : 0.045);
    const a = X(p.n0), b = X(p.n0 + p.L - 1);
    ctx.fillRect(Math.max(r.x, a), r.y, Math.min(r.x + r.w, b) - Math.max(r.x, a), r.h);
    if (label) { ctx.font = uf(500, 10); ctx.fillStyle = theme.c('muted'); ctx.textAlign = 'center'; ctx.textBaseline = 'top'; ctx.fillText(label, (a + b) / 2, r.y + 4); }
  };
  const timeTicks = (ctx, r, X) => {
    ctx.font = mf(500, 10); ctx.fillStyle = theme.c('muted'); ctx.textAlign = 'center'; ctx.textBaseline = 'top';
    for (let u = 0; u <= 12; u += 2) { const x = X(u * 20); if (x >= r.x - 1 && x <= r.x + r.w + 1) ctx.fillText(u === 12 ? '12 µs' : String(u), x, r.y + r.h + 4); }
  };
  const yTicks = (ctx, r, Y, vals, fmtv) => {
    ctx.font = mf(500, 10); ctx.textAlign = 'right'; ctx.textBaseline = 'middle';
    for (const v of vals) { const y = Y(v); ctx.strokeStyle = theme.c('line', v === 0 ? 1 : 0.6); ctx.beginPath(); ctx.moveTo(r.x, Math.round(y) + 0.5); ctx.lineTo(r.x + r.w, Math.round(y) + 0.5); ctx.stroke(); ctx.fillStyle = theme.c('muted'); ctx.fillText(fmtv(v), r.x - 5, y); }
  };
  const cursor = (ctx, r, x) => { ctx.strokeStyle = theme.c('ink', 0.45); ctx.lineWidth = 1; ctx.setLineDash([2, 3]); ctx.beginPath(); ctx.moveTo(Math.round(x) + 0.5, r.y); ctx.lineTo(Math.round(x) + 0.5, r.y + r.h); ctx.stroke(); ctx.setLineDash([]); };
  function wrap(ctx, text, maxW) {
    const words = text.split(' '), out = [];
    let line = '';
    for (const w of words) { const t = line ? line + ' ' + w : w; if (ctx.measureText(t).width > maxW && line) { out.push(line); line = w; } else line = t; }
    if (line) out.push(line);
    return out;
  }

  RL.draw = function () {
    const W = Math.max(240, this.stage.clientWidth), Hh = Math.max(220, this.stage.clientHeight), ctx = NA.fitCanvas(this.canvas, W, Hh);
    ctx.clearRect(0, 0, W, Hh);
    this.lay = {};
    if (this.view === 'acc') { this.drawAcc(ctx, { x: 12, y: 12, w: W - 24, h: Hh - 20 }); return; }
    // the verdict: a column on the right when the stage is roomy, a compact strip across the top when not
    const side = W >= 700 && Hh >= 320, two = W < 470, sh = two ? 62 : 34;
    let M;
    if (side) {
      const vw = clamp(W * 0.27, 190, 260);
      this.drawVerdict(ctx, { x: W - vw - 12, y: 28, w: vw, h: Hh - 40 });
      M = { x: 12, y: 12, w: W - vw - 40, h: Hh - 20 };
    } else {
      this.drawStrip(ctx, { x: 12, y: 8, w: W - 24, h: sh }, two);
      M = { x: 12, y: sh + 26, w: W - 24, h: Hh - sh - 32 };
    }
    if (this.view === 'signal') this.drawSignal(ctx, M);
    else if (this.view === 'spec') this.drawSpec(ctx, M);
    else if (this.view === 'cnn') this.drawCNN(ctx, M);
  };
  // The six probabilities in a row (or two rows of three): ▸ marks what was sent; the answer is outlined in
  // green when it is right and red when it is wrong.
  RL.drawStrip = function (ctx, R, two) {
    const g = this.sig, F = g.F, cls = g.cls, pred = g.pred, per = two ? 3 : 6, gap = 8, sw = (R.w - gap * (per - 1)) / per, rh = two ? 28 : R.h;
    for (let k = 0; k < NC; k++) {
      const x = R.x + (k % per) * (sw + gap), y = R.y + Math.floor(k / per) * (rh + 6), isP = k === pred;
      if (isP) { ctx.strokeStyle = theme.c(pred === cls ? 'good' : 'critical'); ctx.lineWidth = 1.6; NA.rr(ctx, x - 4, y - 2, sw + 8, rh + 2, 6); ctx.stroke(); }
      ctx.font = uf(isP ? 700 : 500, 11.5); ctx.fillStyle = isP ? theme.c('ink') : theme.c('ink-2'); ctx.textAlign = 'left'; ctx.textBaseline = 'middle';
      ctx.fillText((k === cls ? '▸ ' : '') + SHORT[k], x, y + 8);
      ctx.font = mf(500, 10.5); ctx.textAlign = 'right'; ctx.fillStyle = isP ? theme.c('ink') : theme.c('muted');
      ctx.fillText(fmtPct(F.probs[k], 0), x + sw, y + 8);
      ctx.fillStyle = theme.c('panel-2'); NA.rr(ctx, x, y + rh - 9, sw, 6, 3); ctx.fill();
      ctx.fillStyle = ccol(k, isP ? 1 : 0.75); NA.rr(ctx, x, y + rh - 9, Math.max(2, sw * F.probs[k]), 6, 3); ctx.fill();
    }
  };

  RL.drawSignal = function (ctx, M) {
    const g = this.sig, s = g.s, p = s.p, iq = s.iq, cls = g.cls, f = this.ifreq();
    // the two time plots share one column (and one time scale); the I/Q plane sits beside them when there is room
    const padL = 40, side = M.w >= 500, cs = side ? Math.min(M.w * 0.3, M.h * 0.6) : 0, colW = M.w - padL - 4 - (side ? cs + 30 : 0);
    const avail = M.h - 18 - 20 - 26 - 20; // two titles and two rows of time ticks
    const th = Math.max(40, avail * 0.53), fh = Math.max(34, avail - th);
    const top = { x: M.x + padL, y: M.y + 18, w: colW, h: th };
    const by = top.y + th + 46, fr = { x: top.x, y: by, w: colW, h: fh };
    const cr = side ? { x: top.x + colW + 30, y: top.y + (fr.y + fr.h - top.y - cs) / 2, w: cs, h: cs } : null;
    const X = (t) => top.x + (t / (N - 1)) * top.w, lim = Math.max(1.5, maxAbs(iq) * 1.06), Y = (v) => top.y + top.h / 2 - (v / lim) * (top.h / 2);
    this.lay = { time: { x: top.x, w: top.w, t0: 0, span: N - 1 }, top: top.y, bottom: fr.y + fr.h };
    // I and Q
    title(ctx, top, cls ? `I/Q samples · ${CL[cls]} at ${dB(g.snr)}` : 'I/Q samples · receiver noise only');
    if (cls) shadePulse(ctx, top, X, p, 'pulse');
    const tick = lim > 3 ? 2 : 1;
    yTicks(ctx, top, Y, [-tick, 0, tick], (v) => fmt(v, 0));
    frame(ctx, top);
    ctx.save(); ctx.beginPath(); ctx.rect(top.x, top.y, top.w, top.h); ctx.clip();
    const line = (arr, off, col, lw, dash) => { ctx.strokeStyle = col; ctx.lineWidth = lw; ctx.setLineDash(dash || []); ctx.lineJoin = 'round'; ctx.beginPath(); for (let t = 0; t < N; t++) { const x = X(t), y = Y(arr[off + t]); t ? ctx.lineTo(x, y) : ctx.moveTo(x, y); } ctx.stroke(); ctx.setLineDash([]); };
    const fade = this.show.clean && cls ? 0.32 : 0.85;
    line(iq, 0, theme.c('s1', fade), 1.1); line(iq, N, theme.c('s2', fade), 1.1);
    if (this.show.env) {
      ctx.strokeStyle = theme.c('ink', 0.6); ctx.lineWidth = 1.2; ctx.beginPath();
      for (let t = 0; t < N; t++) { const x = X(t), y = Y(Math.hypot(iq[t], iq[N + t])); t ? ctx.lineTo(x, y) : ctx.moveTo(x, y); }
      ctx.stroke();
    }
    if (this.show.clean && cls) { line(s.clean, 0, theme.c('s1'), 1.8); line(s.clean, N, theme.c('s2'), 1.8); }
    ctx.restore();
    timeTicks(ctx, top, X);
    // instantaneous frequency
    const FY = (v) => fr.y + fr.h / 2 - (clamp(v, -0.5, 0.5) / 0.5) * (fr.h / 2);
    title(ctx, fr, 'Instantaneous frequency (MHz)');
    if (cls) shadePulse(ctx, fr, X, p);
    yTicks(ctx, fr, FY, [-0.25, 0, 0.25], (v) => (v ? `${v > 0 ? '+' : '−'}${Math.abs(v * 20)}` : '0'));
    frame(ctx, fr);
    ctx.save(); ctx.beginPath(); ctx.rect(fr.x, fr.y, fr.w, fr.h); ctx.clip();
    ctx.fillStyle = theme.c('muted', 0.4);
    for (let t = 1; t < N; t++) ctx.fillRect(X(t) - 0.8, FY(f.raw[t]) - 0.8, 1.6, 1.6);
    ctx.strokeStyle = theme.c('s7'); ctx.lineWidth = 1.6; ctx.beginPath();
    for (let t = 1; t < N; t++) { const x = X(t), y = FY(f.sm[t]); t > 1 ? ctx.lineTo(x, y) : ctx.moveTo(x, y); }
    ctx.stroke();
    if (this.show.law && cls) {
      ctx.strokeStyle = theme.c('ink', 0.9); ctx.lineWidth = 1.6; ctx.setLineDash([6, 4]); ctx.beginPath();
      let on = false;
      for (let t = 0; t < N; t++) { if (!isFinite(s.freq[t])) { on = false; continue; } const x = X(t), y = FY(s.freq[t]); on ? ctx.lineTo(x, y) : ctx.moveTo(x, y); on = true; }
      ctx.stroke(); ctx.setLineDash([]);
    }
    ctx.restore();
    timeTicks(ctx, fr, X);
    if (this.hoverT !== null && this.hoverT !== undefined) { cursor(ctx, top, X(this.hoverT)); cursor(ctx, fr, X(this.hoverT)); }
    // the I/Q plane
    if (cr) this.drawConstellation(ctx, cr, lim);
  };
  RL.drawConstellation = function (ctx, r, lim) {
    const g = this.sig, s = g.s, p = s.p, iq = s.iq, cls = g.cls, der = this.show.derot && cls;
    const P = (v) => (v / lim) * (r.w / 2), cx = r.x + r.w / 2, cy = r.y + r.h / 2;
    title(ctx, r, der ? 'I/Q plane · carrier removed' : 'I/Q plane');
    frame(ctx, r);
    ctx.strokeStyle = theme.c('line'); ctx.beginPath(); ctx.moveTo(r.x, cy + 0.5); ctx.lineTo(r.x + r.w, cy + 0.5); ctx.moveTo(cx + 0.5, r.y); ctx.lineTo(cx + 0.5, r.y + r.h); ctx.stroke();
    ctx.save(); ctx.beginPath(); ctx.rect(r.x, r.y, r.w, r.h); ctx.clip();
    if (cls) { // the pulse's amplitude, as scaled by the gain control
      const a = Math.pow(10, g.snr / 20) * s.gain;
      ctx.strokeStyle = ccol(cls, 0.6); ctx.setLineDash([3, 3]); ctx.beginPath(); ctx.arc(cx, cy, P(a), 0, Math.PI * 2); ctx.stroke(); ctx.setLineDash([]);
    }
    for (let pass = 0; pass < 2; pass++) for (let t = 0; t < N; t++) {
      const inside = cls && t >= p.n0 && t < p.n0 + p.L;
      if ((pass === 0) === !!inside) continue; // noise first, the pulse on top
      let I = iq[t], Q = iq[N + t];
      if (der && inside) { const ph = -(p.ph0 + 2 * Math.PI * p.f0 * (t - p.n0 + 1)), c = Math.cos(ph), sn = Math.sin(ph); [I, Q] = [I * c - Q * sn, I * sn + Q * c]; }
      ctx.fillStyle = inside ? ccol(cls, 0.8) : theme.c('muted', 0.45);
      ctx.beginPath(); ctx.arc(cx + P(I), cy - P(Q), inside ? 1.9 : 1.5, 0, Math.PI * 2); ctx.fill();
    }
    ctx.restore();
    ctx.font = uf(600, 10); ctx.fillStyle = theme.c('muted'); ctx.textAlign = 'right'; ctx.textBaseline = 'bottom'; ctx.fillText('I', r.x + r.w - 5, cy - 3);
    ctx.textAlign = 'left'; ctx.textBaseline = 'top'; ctx.fillText('Q', cx + 5, r.y + 4);
  };

  RL.drawSpec = function (ctx, M) {
    const g = this.sig, s = g.s, p = s.p, sp = this.spec(), cls = g.cls;
    const r = { x: M.x + 46, y: M.y + 18, w: M.w - 50, h: M.h - 18 - 26 };
    // frames are 32 samples long and 4 apart: the picture spans the frame centres, sample 16 to 240, ± half a hop
    const t0 = sp.win / 2 - sp.hop / 2, span = (sp.frames - 1) * sp.hop + sp.hop, X = (t) => r.x + ((t - t0) / span) * r.w;
    const FY = (f) => r.y + ((sp.bins / 2 - 0.5) / sp.bins - f) * r.h; // row i holds bin (bins − 1 − i), centred on (bins/2 − 1 − i)/bins
    this.lay = { time: { x: r.x, w: r.w, t0, span }, top: r.y, bottom: r.y + r.h };
    title(ctx, r, `Spectrogram · ${cls ? `${CL[cls]} at ${dB(g.snr)}` : 'receiver noise only'}`);
    NA.archRadar.heat(ctx, sp.bins, sp.frames, (i, j) => sp.S[j * sp.bins + (sp.bins - 1 - i)], r, sp.max - 30, sp.max, { lin: true });
    ctx.save(); ctx.beginPath(); ctx.rect(r.x, r.y, r.w, r.h); ctx.clip();
    if (this.show.specLaw && cls) {
      ctx.strokeStyle = theme.dark ? 'rgba(255,255,255,0.9)' : 'rgba(10,12,16,0.85)'; ctx.lineWidth = 1.5; ctx.setLineDash([5, 4]); ctx.beginPath();
      let on = false;
      for (let t = 0; t < N; t++) { if (!isFinite(s.freq[t])) { on = false; continue; } const x = X(t), y = FY(s.freq[t]); on ? ctx.lineTo(x, y) : ctx.moveTo(x, y); on = true; }
      ctx.stroke(); ctx.setLineDash([]);
    }
    ctx.restore();
    ctx.font = mf(500, 10); ctx.fillStyle = theme.c('muted'); ctx.textAlign = 'right'; ctx.textBaseline = 'middle';
    for (const m of [-10, -5, 0, 5, 10]) { const y = FY(m / 20); if (y >= r.y - 1 && y <= r.y + r.h + 1) ctx.fillText(m > 0 ? `+${m}` : m < 0 ? `−${-m}` : '0', r.x - 5, y); }
    ctx.save(); ctx.translate(M.x + 8, r.y + r.h / 2); ctx.rotate(-Math.PI / 2); ctx.textAlign = 'center'; ctx.font = uf(600, 10); ctx.fillText('MHz', 0, 0); ctx.restore();
    timeTicks(ctx, r, X);
    if (cls) { ctx.fillStyle = ccol(cls); ctx.fillRect(Math.max(r.x, X(p.n0)), r.y + r.h + 1, Math.min(r.x + r.w, X(p.n0 + p.L)) - Math.max(r.x, X(p.n0)), 2.5); }
    if (this.hoverT !== null && this.hoverT !== undefined) cursor(ctx, r, X(this.hoverT));
  };

  RL.drawCNN = function (ctx, M) {
    const g = this.sig, s = g.s, p = s.p, F = g.F, cls = g.cls, pred = g.pred, padL = 34;
    const x = M.x + padL, w = M.w - padL - 4, X = (t) => x + (t / N) * w;
    const gap = 27, hs = [0.17, 0.25, 0.3, 0.28], tot = Math.max(100, M.h - 18 - gap * 3 - 18), rows = [];
    let y = M.y + 18;
    for (const f of hs) { rows.push({ x, y, w, h: tot * f }); y += tot * f + gap; }
    const [ri, r1, r5, rc] = rows;
    this.lay = { time: { x, w, t0: 0, span: N }, top: ri.y, bottom: rc.y + rc.h };
    title(ctx, ri, 'Input: I and Q');
    NA.archRadar.traces(ctx, s.iq, ri, p.n0, cls ? p.L : 0, 0.9);
    frame(ctx, ri);
    const a1 = F.rec[0].act, n1 = F.rec[0].inp.n, c1 = F.rec[0].L.co, last = F.rec[F.rec.length - 1], a5 = last.act, n5 = last.inp.n, c5 = last.L.co;
    title(ctx, r1, `Layer 1: ${c1} filters × ${n1} steps, after ReLU`);
    NA.archRadar.heat(ctx, c1, n1, (i, j) => a1[i * n1 + j], r1, 0, maxAbs(a1));
    title(ctx, r5, `Layer 5: ${c5} channels × ${n5} steps${r5.w >= 520 ? ' (each sees 50 samples)' : ''}`);
    NA.archRadar.heat(ctx, c5, n5, (i, j) => a5[i * n5 + j], r5, 0, maxAbs(a5));
    // class activation map of the answer, one bar per step of layer 5 (8 samples each)
    const cam = this.cam(), m = Math.max(1e-6, maxAbs(cam)), zero = rc.y + rc.h * 0.62, up = rc.h * 0.58, down = rc.h * 0.36;
    title(ctx, rc, rc.w < 520 ? `Evidence for ${SHORT[pred]}` : `Evidence for ${SHORT[pred]} (class activation map)`);
    frame(ctx, rc);
    if (cls) shadePulse(ctx, rc, (t) => X(t + 0.5), p);
    ctx.strokeStyle = theme.c('line-2'); ctx.beginPath(); ctx.moveTo(rc.x, zero + 0.5); ctx.lineTo(rc.x + rc.w, zero + 0.5); ctx.stroke();
    cam.forEach((v, t) => { const hh = (Math.abs(v) / m) * (v > 0 ? up : down); ctx.fillStyle = v > 0 ? ccol(pred, 0.85) : theme.c('muted', 0.55); ctx.fillRect(X(t * 8) + 1, v > 0 ? zero - hh : zero, (w / n5) - 2, hh); });
    const avg = cam.reduce((a, b) => a + b, 0) / cam.length, ya = zero - (Math.max(0, avg) / m) * up + (Math.max(0, -avg) / m) * down;
    ctx.strokeStyle = theme.c('ink', 0.7); ctx.setLineDash([4, 3]); ctx.beginPath(); ctx.moveTo(rc.x, ya); ctx.lineTo(rc.x + rc.w, ya); ctx.stroke(); ctx.setLineDash([]);
    // the sum that makes the score (dashed: the average): on the title line, or inside the plot when narrow
    const narrow = rc.w < 520;
    ctx.font = mf(500, 10); ctx.fillStyle = theme.c('ink-2'); ctx.textAlign = 'right'; ctx.textBaseline = narrow ? 'top' : 'alphabetic';
    ctx.fillText(`${narrow ? '' : 'average '}${fmt(avg, 2)} + bias ${fmt(this.net.fc.b[pred], 2)} = score ${fmt(F.logits[pred], 2)}`, rc.x + rc.w - (narrow ? 4 : 0), narrow ? rc.y + 4 : rc.y - 7);
    for (const r of [ri, r1, r5]) { if (cls) { ctx.fillStyle = ccol(cls); ctx.fillRect(Math.max(r.x, X(p.n0)), r.y + r.h + 1, Math.min(r.x + r.w, X(p.n0 + p.L)) - Math.max(r.x, X(p.n0)), 2.5); } }
    timeTicks(ctx, rc, (t) => X(t));
    if (this.hoverT !== null && this.hoverT !== undefined) for (const r of rows) cursor(ctx, r, X(this.hoverT + 0.5));
  };

  // the six probabilities, the truth and whether the network got it, as a column beside the plots
  RL.drawVerdict = function (ctx, V) {
    const g = this.sig, F = g.F, cls = g.cls, pred = g.pred, ok = pred === cls, rowH = 24, nameW = 84, pctW = 42;
    title(ctx, V, 'The network says');
    for (let k = 0; k < NC; k++) {
      const y = V.y + k * rowH, bx = V.x + nameW, bw = V.w - nameW - pctW;
      ctx.font = uf(k === pred ? 700 : 500, 12); ctx.fillStyle = k === pred ? theme.c('ink') : theme.c('ink-2'); ctx.textAlign = 'left'; ctx.textBaseline = 'middle';
      ctx.fillText(SHORT[k], V.x + 12, y + rowH / 2);
      if (k === cls) { ctx.fillStyle = theme.c('ink'); ctx.beginPath(); ctx.moveTo(V.x + 1, y + rowH / 2 - 4); ctx.lineTo(V.x + 7, y + rowH / 2); ctx.lineTo(V.x + 1, y + rowH / 2 + 4); ctx.fill(); }
      ctx.fillStyle = theme.c('panel-2'); NA.rr(ctx, bx, y + rowH / 2 - 4, bw, 8, 4); ctx.fill();
      ctx.fillStyle = ccol(k, k === pred ? 1 : 0.7); NA.rr(ctx, bx, y + rowH / 2 - 4, Math.max(2, bw * F.probs[k]), 8, 4); ctx.fill();
      ctx.font = mf(500, 11); ctx.fillStyle = theme.c('ink-2'); ctx.textAlign = 'right';
      ctx.fillText(fmtPct(F.probs[k], 0), bx + bw + pctW, y + rowH / 2);
    }
    let y = V.y + NC * rowH + 16;
    ctx.font = uf(650, 13); ctx.textAlign = 'left'; ctx.textBaseline = 'middle';
    ctx.fillStyle = ok ? theme.c('good') : theme.c('critical');
    ctx.fillText(ok ? `✓ Right: ${CL[pred]}` : `✗ Wrong: it was ${cls ? 'a ' + CL[cls] : 'noise only'}`, V.x, y);
    y += 20;
    ctx.font = uf(500, 11.5); ctx.fillStyle = theme.c('ink-2');
    const sent = cls ? `▸ marks what was sent: a ${CL[cls]} at ${dB(g.snr)}.` : '▸ marks what was sent: receiver noise, no pulse.';
    for (const l of wrap(ctx, sent, V.w)) { ctx.fillText(l, V.x, y); y += 15; }
    if (this.live.total) {
      y += 10;
      ctx.font = uf(600, 10.5); ctx.fillStyle = theme.c('muted'); ctx.fillText('LIVE STREAM', V.x, y); y += 16;
      ctx.font = uf(650, 13); ctx.fillStyle = theme.c('ink'); ctx.fillText(`${this.live.right} of ${this.live.total} right (${fmtPct(this.live.right / this.live.total, 0)})`, V.x, y); y += 16;
      const marks = this.live.marks, mw = Math.min(10, (V.w - 2) / 24);
      marks.forEach((mk, i) => { ctx.fillStyle = mk.ok ? theme.c('good', 0.85) : theme.c('critical', 0.85); NA.rr(ctx, V.x + i * mw, y, mw - 2, 10, 2); ctx.fill(); });
      y += 18;
      const wrong = marks.filter((mk) => !mk.ok);
      if (wrong.length) { ctx.font = uf(500, 11); ctx.fillStyle = theme.c('muted'); for (const l of wrap(ctx, `misses in view: ${wrong.slice(-3).map((mk) => `${SHORT[mk.cls]} at ${dB(mk.snr)}`).join(', ')}`, V.w)) { ctx.fillText(l, V.x, y); y += 14; } }
    }
  };

  RL.drawAcc = function (ctx, M) {
    // the chart and the matrix side by side, or (on a phone) one above the other
    const W = NA.radarWeights, snrs = W.snrs, A = W.acc, wide = M.w >= 480;
    const pr = wide ? { x: M.x + 40, y: M.y + 18, w: M.w * 0.58 - 40, h: M.h - 18 - 40 } : { x: M.x + 40, y: M.y + 18, w: M.w - 44, h: Math.max(110, M.h * 0.42) };
    const s0 = snrs[0], s1 = snrs[snrs.length - 1], X = (s) => pr.x + ((s - s0) / (s1 - s0)) * pr.w, Y = (v) => pr.y + (1 - v) * pr.h;
    this.lay = { acc: { x: pr.x, y: pr.y, w: pr.w, h: pr.h, s0, s1 } };
    title(ctx, pr, 'Accuracy by SNR');
    // the SNRs seen in training, shaded and named on the title line
    ctx.fillStyle = theme.c('ink', theme.dark ? 0.05 : 0.035); ctx.fillRect(X(RD.SNR[0]), pr.y, X(RD.SNR[1]) - X(RD.SNR[0]), pr.h);
    ctx.font = uf(500, 10); ctx.fillStyle = theme.c('muted'); ctx.textAlign = 'right'; ctx.textBaseline = 'alphabetic';
    ctx.fillText('shaded: trained on −12 to +20 dB', pr.x + pr.w, pr.y - 7);
    yTicks(ctx, pr, Y, [0, 0.25, 0.5, 0.75, 1], (v) => fmtPct(v, 0));
    frame(ctx, pr);
    ctx.font = mf(500, 10); ctx.fillStyle = theme.c('muted'); ctx.textAlign = 'center'; ctx.textBaseline = 'top';
    for (let s = -16; s <= 20; s += 4) ctx.fillText(s > 0 ? `+${s}` : s < 0 ? `−${-s}` : '0', X(s), pr.y + pr.h + 4);
    ctx.fillText('SNR (dB)', pr.x + pr.w / 2, pr.y + pr.h + 18);
    const chance = Y(1 / NC);
    ctx.strokeStyle = theme.c('muted', 0.7); ctx.setLineDash([2, 3]); ctx.beginPath(); ctx.moveTo(pr.x, chance); ctx.lineTo(pr.x + pr.w, chance); ctx.stroke(); ctx.setLineDash([]);
    ctx.textBaseline = 'bottom'; ctx.textAlign = 'right'; ctx.fillText('chance, 1 in 6', pr.x + pr.w - 4, chance - 2);
    const curve = (vals, col, lw) => { ctx.strokeStyle = col; ctx.lineWidth = lw; ctx.lineJoin = 'round'; ctx.beginPath(); vals.forEach((v, i) => (i ? ctx.lineTo(X(snrs[i]), Y(v)) : ctx.moveTo(X(snrs[i]), Y(v)))); ctx.stroke(); };
    for (let k = 0; k < NC; k++) curve(A.map((row) => row[k]), ccol(k, 0.9), 1.5);
    const mean = A.map((row) => row.reduce((a, b) => a + b, 0) / NC);
    curve(mean, theme.c('ink'), 2.6);
    // the slider's SNR
    const sv = clamp(this.snr, s0, s1), xs = X(sv), i0 = Math.max(0, Math.min(snrs.length - 2, Math.floor((sv - s0) / 2))), f = (sv - snrs[i0]) / 2, mv = mean[i0] + (mean[i0 + 1] - mean[i0]) * f;
    cursor(ctx, pr, xs);
    ctx.fillStyle = theme.c('ink'); ctx.beginPath(); ctx.arc(xs, Y(mv), 4.5, 0, Math.PI * 2); ctx.fill(); ctx.strokeStyle = theme.c('panel'); ctx.lineWidth = 2; ctx.stroke();
    // the label goes above the point, or below it when the point is near the top
    const below = Y(mv) - 22 < pr.y;
    ctx.font = uf(650, 11.5); ctx.textAlign = xs > pr.x + pr.w * 0.7 ? 'right' : 'left'; ctx.textBaseline = below ? 'top' : 'bottom'; ctx.fillStyle = theme.c('ink');
    ctx.fillText(`${fmtPct(mv, 0)} at ${dB(this.snr)}`, xs + (ctx.textAlign === 'left' ? 8 : -8), Y(mv) + (below ? 8 : -6));
    // where the mistakes go
    // (the matrix leaves room on its right for the last column's leaning name)
    const mr = wide ? { x: M.x + M.w * 0.58 + 62, y: M.y + 60, w: M.w * 0.42 - 84 } : { x: M.x + 58, y: pr.y + pr.h + 90, w: M.w - 84 };
    const side = Math.max(60, Math.min(mr.w, (wide ? M.y + M.h - mr.y : M.y + M.h - mr.y) - 40));
    this.drawConfusion(ctx, { x: mr.x, y: mr.y, w: side, h: side });
  };
  RL.matrix = function () {
    const W = NA.radarWeights;
    if (this.accSrc === 'live' && this.test) return { conf: this.test.conf, snr: this.test.snr, per: this.test.per, n: this.test.n, live: true };
    let best = 0;
    W.snrs.forEach((s, i) => { if (Math.abs(s - this.snr) < Math.abs(W.snrs[best] - this.snr)) best = i; });
    return { conf: W.conf[best], snr: W.snrs[best], per: W.per, n: W.per * NC, live: false };
  };
  RL.drawConfusion = function (ctx, r) {
    const m = this.matrix(), cw = r.w / NC, ch = r.h / NC;
    ctx.font = uf(500, 10);
    const tilt = ctx.measureText('Costas').width > cw - 4; // narrow columns: the names lean, so they do not collide
    title(ctx, { x: r.x - 54, y: r.y - (tilt ? 40 : 16) }, m.live ? `Your test at ${dB(m.snr)} · ${m.n} of ${m.per * NC}` : `Where the answers go at ${dB(m.snr)}`);
    ctx.font = uf(500, 10); ctx.fillStyle = theme.c('muted'); ctx.textAlign = 'center'; ctx.textBaseline = 'bottom';
    for (let j = 0; j < NC; j++) {
      if (!tilt) { ctx.fillText(SHORT[j], r.x + (j + 0.5) * cw, r.y - 3); continue; }
      ctx.save(); ctx.translate(r.x + (j + 0.5) * cw, r.y - 4); ctx.rotate(-Math.PI / 5); ctx.textAlign = 'left'; ctx.textBaseline = 'middle'; ctx.fillText(SHORT[j], 0, 0); ctx.restore();
    }
    ctx.textAlign = 'right'; ctx.textBaseline = 'middle';
    for (let i = 0; i < NC; i++) ctx.fillText(SHORT[i], r.x - 6, r.y + (i + 0.5) * ch);
    let right = 0, all = 0;
    for (let i = 0; i < NC; i++) {
      const tot = m.conf[i].reduce((a, b) => a + b, 0) || 1;
      for (let j = 0; j < NC; j++) {
        const v = m.conf[i][j], q = v / tot;
        if (i === j) right += v; all += v;
        ctx.fillStyle = i === j ? theme.seqColor(Math.sqrt(q)) : theme.gradColor(Math.sqrt(q));
        ctx.fillRect(r.x + j * cw + 0.5, r.y + i * ch + 0.5, cw - 1, ch - 1);
        if (cw > 22 && v) { ctx.font = mf(500, Math.min(11, cw * 0.3)); ctx.textAlign = 'center'; ctx.fillStyle = q > 0.55 ? theme.c('panel') : theme.c('ink-2'); ctx.fillText(String(v), r.x + (j + 0.5) * cw, r.y + (i + 0.5) * ch); }
      }
    }
    ctx.strokeStyle = theme.c('line-2'); ctx.lineWidth = 1; ctx.strokeRect(r.x + 0.5, r.y + 0.5, r.w - 1, r.h - 1);
    // under the matrix: how to read it and the share on the diagonal (short enough for a narrow matrix)
    ctx.font = uf(650, 12); ctx.fillStyle = theme.c('ink'); ctx.textAlign = 'left'; ctx.textBaseline = 'top';
    if (all) ctx.fillText(`${fmtPct(right / all, 1)} right`, r.x, r.y + r.h + 6);
    ctx.font = uf(500, 10.5); ctx.fillStyle = theme.c('muted');
    ctx.fillText(r.w >= 250 ? 'rows: sent · columns: answered' : 'rows: sent', r.x, r.y + r.h + 23);
  };

  // ------------------------------------------------------------------ test in the browser
  RL.runTest = function () {
    if (this.view !== 'acc') this.setView('acc');
    this.test = { snr: this.snr, per: 100, n: 0, conf: Array.from({ length: NC }, () => new Array(NC).fill(0)), rng: new NA.RNG((Math.random() * 1e9) | 0), done: false };
    this.accSrc = 'live'; this.srcSeg.set('live');
    this.dirty = true; this.infoStale = true;
  };
  RL.runTestChunk = function () {
    const T = this.test, t0 = performance.now(), total = T.per * NC;
    while (T.n < total && performance.now() - t0 < 12) {
      const c = T.n % NC, s = RD.generate(T.rng, { cls: c, snr: T.snr }), F = RD.forward(this.net, s.iq);
      T.conf[c][argmax(F.probs)]++; T.n++;
    }
    if (T.n >= total) { T.done = true; this.infoStale = true; this.inspDirty = true; }
    this.testBtn.textContent = T.done ? 'Test again' : `Testing… ${T.n} / ${total}`;
    if (this.view === 'acc') this.dirty = true;
  };

  // ------------------------------------------------------------------ explanations under the picture
  RL.showInfo = function () {
    if (this.view === 'arch') { this.arch.refreshFocus(); return; }
    this.info.set(({ signal: SIGNAL_INFO, spec: SPEC_INFO, cnn: CNN_INFO, acc: ACC_INFO })[this.view](this));
  };
  const PRE = '# generate, CLASSES, N and model as in the whole-model code (Architecture view, top level, Python tab)\n';
  const pySnr = (v) => Math.round(v * 10) / 10;
  const SIGNAL_INFO = (L) => {
    const g = L.sig, s = g.s, p = s.p, cls = g.cls, ratio = Math.pow(10, g.snr / 10), gain = 10 * Math.log10(p.L);
    return {
      kicker: 'Signal view', title: cls ? `${CL[cls]} at ${dB(g.snr)}: the raw I/Q samples` : 'Receiver noise only: the raw I/Q samples',
      about: [
        `A receiver shifts the radio signal down to frequencies around zero and samples it as I and Q, the real and imaginary parts of one complex number: together they record the signal's amplitude and phase. These are 256 samples, 50 ns apart, 12.8 µs in all. ${cls ? `This window holds a ${CL[cls]}: ${RD.describe(s)}.` : 'This window holds no pulse, only the receiver’s own noise.'}`,
        cls
          ? `At ${dB(g.snr)} the pulse has ${ratio >= 1 ? fmt(ratio, ratio < 10 ? 1 : 0) + ' times' : fmt(ratio, 2) + ' of'} the noise's power in each sample (in decibels, 10·log₁₀ of that ratio). ${g.snr < 0 ? `Every single sample is more noise than pulse, so the traces barely change where the pulse is. What gives it away is its length: added up coherently over its ${p.L} samples, the pulse grows ${p.L} times in power against the noise, a gain of ${fmt(gain, 1)} dB.` : 'The pulse stands out of the noise: the envelope steps up where it starts and drops where it ends.'} The whole window is then scaled to an average power of 1, so loudness alone tells the network nothing.`
          : 'Without a pulse every sample is Gaussian noise: I and Q are independent, the envelope wanders at random, and the points in the I/Q plane form a round cloud.',
        `Below, the instantaneous frequency: how far the phase turns from one sample to the next, smoothed over 9 samples, in MHz. Inside the pulse it follows the waveform's frequency law (dashed: flat for a plain pulse and for the phase codes, a ramp for a chirp, six steps for a Costas code); outside, the noise makes it jump at random. On the right, every sample as a point in the I/Q plane: noise is a round cloud, the pulse a ring whose radius is its amplitude.${cls === 3 || cls === 4 ? ` Switch on Remove carrier to undo the carrier offset (with the true offset, which the network is never given): ${cls === 3 ? 'the Barker code’s two phases, 180° apart, separate into two clusters' : `the Frank code’s ${p.M} phases separate into ${p.M} clusters`}.` : ''}`,
      ],
      code: `import matplotlib.pyplot as plt
${PRE}
x, y, snr = generate(1, cls=${cls}, snr=${pySnr(g.snr)})
z = torch.complex(x[0, 0], x[0, 1])             # one complex number per sample
t_us = torch.arange(N) / 20                     # 20 samples per microsecond
lag = z[1:] * z[:-1].conj()                     # the phase step between neighbours
smooth = torch.nn.functional.conv1d(torch.view_as_real(lag).T[:, None], torch.ones(1, 1, 9), padding=4)[:, 0]
freq = torch.atan2(smooth[1], smooth[0]) / (2 * math.pi) * 20     # MHz

fig = plt.figure(figsize=(14, 6))
top, low, iq = fig.add_subplot(2, 1, 1), fig.add_subplot(2, 3, (4, 5)), fig.add_subplot(2, 3, 6)
top.plot(t_us, z.real, lw=1, label='I')
top.plot(t_us, z.imag, lw=1, label='Q')
top.plot(t_us, z.abs(), 'k', lw=1, label='|x|')
top.legend()
low.plot(t_us[1:], freq, lw=1.5)
low.set_ylim(-10, 10)
low.set_ylabel('MHz')
iq.scatter(z.real, z.imag, s=4)
iq.set_aspect('equal')
plt.show()`,
      codeLabel: 'Python · PyTorch and matplotlib · one intercept, its frequency and its I/Q plane',
    };
  };
  const SPEC_INFO = (L) => {
    const g = L.sig, s = g.s, p = s.p, cls = g.cls, dur = p.L / 20;
    const fp = cls === 1 ? `A plain pulse is a single horizontal line at its carrier offset, ${mhz(p.f0)}, from ${us(p.n0)} to ${us(p.n0 + p.L)} µs.`
      : cls === 2 ? `The chirp is a sloping line: its frequency sweeps ${p.up > 0 ? 'up' : 'down'} through ${fmt(p.bw * 20, 2)} MHz in ${fmt(dur, 2)} µs, ${fmt((p.bw * 20) / dur, 2)} MHz per µs, centred on its carrier offset of ${mhz(p.f0)}.`
      : cls === 3 ? `A Barker-13 code stays on one carrier, ${mhz(p.f0)}, but its phase flips at 6 of its 12 chip boundaries. Every flip splashes energy sideways, so the line is smeared about ${fmt(20 / (p.L / 13), 1)} MHz wide (one over the chip length, ${us(p.L / 13)} µs).`
      : cls === 4 ? `A Frank code stays on one carrier, ${mhz(p.f0)}, and steps its phase through 2π·i·j/${p.M} over ${p.M * p.M} chips of ${us(p.L / (p.M * p.M))} µs. Like the Barker code it smears the line sideways, here about ${fmt(20 / (p.L / (p.M * p.M)), 1)} MHz wide, which is why the two codes look alike in a spectrogram.`
      : cls === 5 ? `A Costas code hops between 6 tones ${fmt((6 / p.L) * 20, 2)} MHz apart, each held for ${us(p.L / 6)} µs and each used once, in the order ${p.costas.join(' ')}: six short steps. The order is a Costas array, chosen so that no two pairs of hops share both their time gap and their frequency gap.`
      : 'Receiver noise is spread evenly over every frequency: speckle with no structure.';
    return {
      kicker: 'Spectrogram view', title: cls ? `Spectrogram of the ${CL[cls]} at ${dB(g.snr)}` : 'Spectrogram of receiver noise',
      about: [
        'A spectrogram cuts the window into short overlapping pieces, 32 samples (1.6 µs) long and 4 samples apart, takes the Fourier transform of each, and plots its power: time across, frequency up, brighter for more power, over the top 30 dB. A 32-sample piece separates frequencies 0.625 MHz apart; longer pieces would sharpen frequency but blur time.',
        fp,
        'The network never sees this picture: it reads the raw I/Q samples. But its first layer’s 16 filters are 7 samples long, like tiny Fourier analysers tuned by training, and each deeper layer looks at a longer stretch, so it can learn the same kind of features. Hover over the picture to read the samples at any moment.',
      ],
      code: `import matplotlib.pyplot as plt
${PRE}
x, y, snr = generate(1, cls=${cls}, snr=${pySnr(g.snr)})
z = torch.complex(x[0, 0], x[0, 1])
S = torch.stft(z, n_fft=32, hop_length=4, window=torch.hann_window(32, periodic=False),
               center=False, return_complex=True)            # (32 frequencies, 57 frames)
P = torch.fft.fftshift(10 * torch.log10(S.abs() ** 2 + 1e-9), dim=0)   # -10 MHz at the bottom
plt.imshow(P, origin='lower', aspect='auto', extent=[0.8, 12.0, -10, 10], vmin=float(P.max()) - 30)
plt.xlabel('time (us)')
plt.ylabel('frequency (MHz)')
plt.colorbar(label='dB')
plt.show()`,
      codeLabel: 'Python · PyTorch and matplotlib · the spectrogram',
    };
  };
  const CNN_INFO = (L) => {
    const g = L.sig, F = g.F, p = g.s.p, cls = g.cls, pred = g.pred, a1 = F.rec[0].act, n1 = F.rec[0].inp.n;
    let ins = 0, all = 0;
    for (let i = 0; i < a1.length; i++) { all += a1[i]; const t = i % n1; if (cls && t >= p.n0 && t < p.n0 + p.L) ins += a1[i]; }
    const share = L.camShare();
    return {
      kicker: 'What the CNN sees', title: `Inside the network: ${CL[pred]}, ${fmtPct(F.probs[pred], 0)}`,
      about: [
        `Under the input, the first layer: 16 filters, each 7 samples long across I and Q, slide along the signal; every row is one filter's response after ReLU (dark: zero). ${cls ? `The pulse covers ${fmtPct(p.L / N, 0)} of the window, and ${fmtPct(all ? ins / all : 0, 0)} of the first layer's total response comes from inside it: even the first filters mostly answer to the pulse, not the noise.` : 'With noise alone the response is weak and scattered.'}`,
        'Next, the last convolution layer: 64 channels at 32 steps, each step 8 samples apart and each number depending on 50 samples (2.5 µs) of the input. These are the features the decision is made from.',
        `At the bottom, the class activation map for the answer, ${CL[pred]}: at each step, the 64 features weighted by that class's output weights. The network averages over time before its last layer, so the class score is exactly the average of this curve plus a bias (${fmt(F.logits[pred], 2)} here): the bars show where the evidence came from. ${cls && pred !== 0 ? `${fmtPct(share, 0)} of the positive evidence lies inside the pulse, although the network was never told where the pulse is.` : pred === 0 ? 'For Noise only, the evidence is the absence of structure, spread over the whole window.' : ''}`,
      ],
      code: `import matplotlib.pyplot as plt
${PRE}
x, y, snr = generate(1, cls=${cls}, snr=${pySnr(g.snr)})
outs = layer_outputs(model, x)                  # each layer after ReLU, before pooling
with torch.no_grad():
    pred = int(model(x).argmax())
    cam = model.fc.weight[pred] @ outs[-1][0]   # (32,): the class activation map
print(CLASSES[pred], float(cam.mean() + model.fc.bias[pred]))   # the class score
fig, ax = plt.subplots(4, 1, figsize=(12, 9), sharex=False)
ax[0].plot(x[0, 0], lw=1)
ax[0].plot(x[0, 1], lw=1)
ax[1].imshow(outs[0][0], aspect='auto', cmap='magma')
ax[2].imshow(outs[-1][0], aspect='auto', cmap='magma')
ax[3].bar(torch.arange(32) * 8 + 4, cam, width=7)
plt.show()`,
      codeLabel: 'Python · PyTorch and matplotlib · feature maps and the class activation map',
    };
  };
  const ACC_INFO = (L) => {
    const W = NA.radarWeights, m = L.matrix(), A = W.acc, mean = A.map((row) => row.reduce((a, b) => a + b, 0) / NC);
    const at = (s) => mean[W.snrs.indexOf(s)], sEval = W.snrs.reduce((b, s) => (Math.abs(s - m.snr) < Math.abs(b - m.snr) ? s : b), W.snrs[0]);
    let worst = null;
    for (let i = 0; i < NC; i++) for (let j = 0; j < NC; j++) if (i !== j && m.conf[i][j] && (!worst || m.conf[i][j] > worst.v)) worst = { i, j, v: m.conf[i][j], tot: m.conf[i].reduce((a, b) => a + b, 0) };
    const tot = m.conf.reduce((a, row) => a + row.reduce((p, v) => p + v, 0), 0), right = m.conf.reduce((a, row, i) => a + row[i], 0);
    return {
      kicker: 'Accuracy view', title: 'Accuracy by signal-to-noise ratio',
      about: [
        `Each line is the share of fresh signals named correctly at each SNR: 400 of each kind at each level, ${fmtInt(W.snrs.length * NC * W.per)} signals in all, generated after training and never seen before. The thick line is the average over the six kinds; the shaded band is the range of SNRs used in training.`,
        `Above 0 dB the network is nearly perfect: ${fmtPct(at(2), 0)} at +2 dB and above. It degrades smoothly as the pulse sinks into the noise: ${fmtPct(at(-2), 0)} at −2 dB, ${fmtPct(at(-4), 0)} at −4, ${fmtPct(at(-6), 0)} at −6 and ${fmtPct(at(-8), 0)} at −8 dB. Below −10 dB nearly every signal is called Noise only: the network has learned to say "nothing there" rather than guess a waveform. The plain pulse holds out longest, since spotting it takes only energy; the chirp goes first, mistaken for a plain pulse or a Frank code.`,
        `The matrix shows where the answers go${m.live ? ` in your test at ${dB(m.snr)} (${m.n} of ${m.per * NC} signals so far)` : ` at ${dB(m.snr)}, the evaluated level nearest the slider`}: rows are what was sent, columns what the network said. ${tot ? `${fmtPct(right / tot, 1)} are on the diagonal.` : ''} ${worst ? `The most common mistake: ${CL[worst.i]} called ${CL[worst.j]}, ${fmtPct(worst.v / worst.tot, 0)} of them.` : 'There are no mistakes at all.'}`,
        'Noise only is right about 95% of the time at every SNR, so roughly 1 window of pure noise in 20 is reported as a waveform: a false-alarm rate that a real system would tune to its needs. Test now runs 100 fresh signals of each kind at the slider’s SNR through the network in your browser and fills in a matrix of its own.',
      ],
      code: `import matplotlib.pyplot as plt
${PRE}
@torch.no_grad()
def accuracy_by_snr(model, snrs=range(-16, 21, 2), per=400):
    model.eval()
    acc = torch.zeros(len(snrs), 6)
    conf = torch.zeros(len(snrs), 6, 6, dtype=torch.long)
    for i, s in enumerate(snrs):
        for c in range(6):
            x, y, _ = generate(per, cls=c, snr=s)
            p = model(x).argmax(1)
            acc[i, c] = (p == y).float().mean()
            conf[i, c] = torch.bincount(p, minlength=6)     # row: sent, column: answered
    return acc, conf

snrs = list(range(-16, 21, 2))
acc, conf = accuracy_by_snr(model)
for c, name in enumerate(CLASSES):
    plt.plot(snrs, acc[:, c], label=name)
plt.plot(snrs, acc.mean(1), 'k', lw=3, label='average')
plt.xlabel('SNR (dB)')
plt.ylabel('accuracy')
plt.legend()
plt.show()
print(conf[snrs.index(${sEval})])               # the matrix at ${sEval} dB`,
      codeLabel: 'Python · PyTorch and matplotlib · accuracy by SNR and the confusion matrices',
    };
  };

  // ------------------------------------------------------------------ right: inspector
  RL.buildTab = function (id, body) {
    this.tabId = id; const P = (this.parts = {});
    if (id === 'overview') {
      body.append(UI.modelCard(this.preset, (t) => this.runTry(t)));
      P.findings = el('div', { class: 'diag-list' });
      body.append(UI.section('This signal', P.findings));
    } else if (id === 'train') {
      const W = NA.radarWeights;
      P.tiles = el('div', { class: 'tiles' },
        UI.tile('Signals', fmtInt(W.train.signals), `${W.train.epochs} epochs`), UI.tile('Batch', `${W.train.batch}`, 'fresh signals'),
        UI.tile('Training time', `${W.train.seconds} s`, 'on a laptop GPU'), UI.tile('Parameters', fmtInt(W.params)));
      const c1 = el('canvas'), c2 = el('canvas');
      P.loss = new NA.charts.LineChart(c1, { height: 140, format: (v) => fmt(v, 3), xLabel: 'Epoch' });
      P.acc = new NA.charts.LineChart(c2, { height: 160, format: (v) => fmtPct(v, 1), yFormat: (v) => fmtPct(v, 0), xLabel: 'Epoch' });
      body.append(P.tiles,
        UI.section('Loss', el('div', { class: 'chart-box' }, c1), UI.note('Cross-entropy on the training batches, averaged over each epoch of 200 batches. It levels off near 0.46, well above zero: about a quarter of the training signals are at SNRs where nobody could name them reliably.')),
        UI.section('Accuracy during training', el('div', { class: 'chart-box' }, c2), el('div', { class: 'mini-legend' },
          ...[['s1', '+10 dB'], ['s3', '0 dB'], ['s2', '−10 dB'], ['muted', 'training batches, all SNRs']].map(([c, t]) => el('span', { class: 'lg-item' }, el('i', { class: 'key-line', style: { background: `var(--${c})` } }), t))),
        UI.note('Measured on fresh signals after every epoch. At +10 dB the network goes from 66% after the first epoch to 97% after the third; at 0 dB it climbs more slowly, past 90% by epoch 9 and to 97% by the end. At −10 dB it never gets far past guessing (about 29%): most of those pulses are too weak to read.')),
        UI.section('How it was trained', el('div', { class: 'kv-grid' }, ...[
          ['Data', 'generated afresh for every batch: no signal is seen twice'], ['Optimiser', 'AdamW, learning rate up to 0.003 (one-cycle schedule)'],
          ['Loss', 'cross-entropy over the 6 classes'], ['SNR range', '−12 to +20 dB, uniform'], ['Script', 'tools/train-radar.py (PyTorch)'],
        ].map(([k, v]) => el('div', { class: 'kv' }, el('span', { class: 'kv-k', text: k }), el('span', { class: 'kv-v', text: v, style: { fontFamily: 'var(--font-ui)', fontWeight: 500, fontSize: '12px', textAlign: 'right' } }))))));
    } else if (id === 'waves') {
      P.cards = [];
      const ABOUT = [
        'No emitter, only the receiver’s thermal noise. A detector’s first job is to say when nothing is there; this network calls about 1 noise window in 20 a waveform (a false alarm).',
        'One burst at one frequency. Simple and cheap, but its range resolution is set by its length: a 10 µs pulse cannot separate two targets closer than 1.5 km.',
        'Linear frequency modulation: the frequency sweeps steadily across the pulse. A matched filter compresses it into a spike about 1/B wide, so a long, energetic pulse still resolves targets as if it were short: with B = 5 MHz, 30 m.',
        'Thirteen chips whose phase is 0° or 180° in the Barker pattern + + + + + − − + + − + − +. Its autocorrelation peaks at 13 with sidelobes no higher than 1, the best of any binary code.',
        'A polyphase code: M × M chips whose phases step through 2π·i·j/M. It compresses the pulse like the Barker code, with more chips and more phase levels than any Barker code allows.',
        'The pulse hops between 6 tones, each used once, in an order (a Costas array) where every pair of hops differs in its time or frequency gap: the echo pins down range and speed at the same time. Hopping also makes it harder to intercept.',
      ];
      for (let k = 0; k < NC; k++) {
        const c = el('canvas');
        P.cards.push({ k, c });
        body.append(UI.section(CL[k], el('div', { class: 'chart-box' }, c), UI.note(ABOUT[k])));
      }
    } else if (id === 'net') {
      P.kv = el('div', { class: 'kv-grid' });
      body.append(UI.section('Layers', P.kv), UI.section('What runs in the browser', UI.note('Each BatchNorm is folded into its convolution (one weight and one bias per output channel), and the weights are stored as 16-bit floats: 53 KB in all. One forward pass is about 2.2 million multiply-adds, a few milliseconds in JavaScript.')));
    }
    this.updateTab();
  };
  RL.updateTab = function () {
    const id = this.tabId, P = this.parts;
    if (!P) return;
    const W = NA.radarWeights, g = this.sig;
    if (id === 'overview') {
      const items = [], F = g.F, cls = g.cls, pred = g.pred, ok = pred === cls, p = g.s.p;
      const f = (level, ttl, text) => items.push(el('div', { class: 'diag diag-' + level }, el('div', { class: 'diag-head' }, UI.pill(level, { good: 'Right', critical: 'Wrong', info: 'Note', warn: 'Unsure' }[level]), el('h4', { class: 'diag-title', text: ttl })), el('p', { class: 'diag-body', text })));
      const i0 = W.snrs.reduce((b, s, i) => (Math.abs(s - g.snr) < Math.abs(W.snrs[b] - g.snr) ? i : b), 0), expect = W.acc[i0][cls];
      f(ok ? (F.probs[pred] < 0.6 ? 'warn' : 'good') : 'critical', ok ? `${CL[pred]}, ${fmtPct(F.probs[pred], 0)}` : `Said ${CL[pred]} (${fmtPct(F.probs[pred], 0)}); it was ${cls ? 'a ' + CL[cls] : 'noise only'}`,
        `${ok ? (F.probs[pred] < 0.6 ? 'Right, but only just: the runner-up is close.' : 'Right, and confident.') : `The right answer got ${fmtPct(F.probs[cls], 0)}.`} At ${dB(W.snrs[i0])} the network names ${cls ? `a ${CL[cls]}` : 'noise'} correctly ${fmtPct(expect, 0)} of the time (400 fresh signals after training).`);
      if (cls) f('info', `SNR ${dB(g.snr)}, pulse of ${p.L} samples`, g.snr < 0 ? `Each sample holds ${fmt(Math.pow(10, g.snr / 10), 2)} of the noise's power in pulse power, but over its ${p.L} samples the pulse adds up coherently: ${fmt(10 * Math.log10(p.L), 1)} dB of gain for anything that integrates it, as the network's filters do.` : `The pulse is ${fmt(Math.pow(10, g.snr / 10), 1)} times stronger than the noise in each sample: easy to see in the traces, and the network is nearly always right at this level.`);
      if (this.live.total) f('info', `Live stream: ${this.live.right} of ${this.live.total} right`, `${fmtPct(this.live.right / this.live.total, 0)} so far, at random SNRs from −12 to +20 dB. The average over that range, measured after training, is about 83%.`);
      clear(P.findings).append(...items);
    } else if (id === 'train') {
      P.loss.frame(); P.loss.set({ series: [{ name: 'loss', color: theme.c('pos'), points: W.curve.map((r) => [r[0], r[1]]) }] });
      P.acc.frame(); P.acc.set({ yFixed: [0, 1], series: [['s1', 5, '+10 dB'], ['s3', 4, '0 dB'], ['s2', 3, '−10 dB'], ['muted', 2, 'training batches']].map(([c, i, n]) => ({ name: n, color: theme.c(c), points: W.curve.map((r) => [r[0], r[i]]) })) });
    } else if (id === 'waves') {
      for (const { k, c } of P.cards) {
        const w = Math.max(200, c.parentElement.clientWidth), h = 78, ctx = NA.fitCanvas(c, w, h);
        ctx.clearRect(0, 0, w, h);
        // a long pulse near the middle of the window shows each waveform best: pick such a draw
        let s = null;
        for (let seed = 300; seed < 400 && !s; seed++) { const q = RD.generate(new NA.RNG(seed), { cls: k, snr: 18 }); if (!k || (q.p.L >= 170 && Math.abs(q.p.n0 - (N - q.p.L) / 2) <= 12)) s = q; }
        s = s || RD.generate(new NA.RNG(300), { cls: k, snr: 18 });
        const sp = RD.spectrogram(s.iq, 32, 4), p = s.p;
        NA.archRadar.heat(ctx, sp.bins, sp.frames, (i, j) => sp.S[j * sp.bins + (sp.bins - 1 - i)], { x: 1, y: 1, w: w - 2, h: h - 16 }, sp.max - 30, sp.max, { lin: true });
        ctx.font = uf(500, 10); ctx.fillStyle = theme.c('muted'); ctx.textAlign = 'left'; ctx.textBaseline = 'bottom';
        const what = [null, 'one frequency', `sweeping ${fmt(p.bw * 20, 1)} MHz`, '13 chips', `${p.M * p.M} chips`, '6 hops'][k];
        ctx.fillText(k ? `${us(p.L)} µs pulse, ${what}, +18 dB · time across, frequency up` : 'noise only · time across, frequency up', 2, h - 1);
      }
    } else if (id === 'net') {
      const rows = [['Input', '2 × 256', 'I and Q', 0]];
      let n = N, cin = 2, rf = 1, jump = 1;
      W.widths.forEach((c, i) => {
        const k = W.ks[i];
        rf += (k - 1) * jump;
        const params = c * cin * k + 2 * c;
        rows.push([`Conv ${i + 1}`, `${c} × ${W.pools[i] ? n / 2 : n}`, `kernel ${k}, ReLU${W.pools[i] ? ', max-pool' : ''} · sees ${rf} samples`, params]);
        if (W.pools[i]) { rf += jump; jump *= 2; n /= 2; }
        cin = c;
      });
      rows.push(['Average over time', `${cin}`, 'one number per channel', 0], ['Linear', '6', 'one score per waveform', cin * 6 + 6], ['Softmax', '6', 'probabilities', 0]);
      clear(P.kv).append(...rows.map(([k, shape, note, params]) => el('div', { class: 'kv' }, el('span', { class: 'kv-k', text: k }), el('span', { class: 'kv-v', text: shape }), el('span', { class: 'kv-note', text: `${note}${params ? ` · ${fmtInt(params)} parameters` : ''}` }))),
        el('div', { class: 'kv' }, el('span', { class: 'kv-k', text: 'Total' }), el('span', { class: 'kv-v', text: fmtInt(W.params) }), el('span', { class: 'kv-note', text: 'weights, biases and BatchNorm scales' })));
    }
  };

  // ------------------------------------------------------------------ experiments, keys and the zoo card
  RL.runTry = function (t) {
    if (this.training && t.action !== 'stream') this.pause();
    if (t.action === 'lowSnr') { this.snr = -6; this.syncControls(); this.make(this.sig.cls || 1, -6, this.sig.seed); this.setView('signal'); NA.ui.toast.show('−6 dB: the noise is 4 times stronger than the pulse in every sample.'); }
    else if (t.action === 'chirp') { this.pick = 2; if (this.snr < 4) this.snr = 8; this.syncControls(); this.next(); this.setView('spec'); NA.ui.toast.show('A chirp: its frequency sweeps in a straight line across the pulse.'); }
    else if (t.action === 'codes') { this.pick = 3; if (this.snr < 6) this.snr = 12; this.syncControls(); this.next(); this.show.derot = true; NA.$$('.chip', this.chipSets.signal).forEach((b) => { if (b.textContent === 'Remove carrier') b.setAttribute('aria-pressed', 'true'); }); this.setView('signal'); NA.ui.toast.show('Barker-13 with the carrier removed: two phases, 180° apart. Pick Frank code to compare.'); }
    else if (t.action === 'stream') { this.live = { right: 0, total: 0, marks: [] }; this.setView('signal'); this.play(); NA.ui.toast.show('Streaming a new intercept every second. Press Space to pause.'); }
    if (NA.mobile && NA.mobile.mobile) NA.mobile.go('network');
    this.dirty = true; this.inspDirty = true;
  };
  RL.key = function (e) {
    if (/^[0-6]$/.test(e.key)) this.choose(e.key === '0' ? -1 : +e.key - 1);
  };
  // Zoo card: a noisy chirp.
  RL.glyph = function (ctx, W, Hh) {
    const rng = new NA.RNG(5), n = 160, a = 0.3 * W, b = 0.82 * W;
    ctx.lineWidth = 1.2; ctx.strokeStyle = theme.c('ink-2', 0.45); ctx.beginPath();
    let ph = 0;
    for (let i = 0; i < n; i++) {
      const x = 4 + (i / (n - 1)) * (W - 8), inside = x > a && x < b, u = (x - a) / (b - a);
      if (inside) ph += 0.25 + 1.0 * u;
      const y = Hh / 2 + (inside ? Math.cos(ph) * Hh * 0.32 : 0) + rng.gauss(0, Hh * 0.06);
      i ? ctx.lineTo(x, y) : ctx.moveTo(x, y);
    }
    ctx.stroke();
    ctx.strokeStyle = theme.c('s7'); ctx.lineWidth = 2; ctx.beginPath(); ctx.moveTo(a, Hh * 0.88); ctx.lineTo(b, Hh * 0.62); ctx.stroke();
  };

  H.register(RL);
  NA.radarLab = RL;
})(window.NA);
