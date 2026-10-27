/* lab-yolo.js — the YOLO26 lab: a scene of shapes you can move, resize, add and remove while the detector, running
   in your browser, redraws its boxes; YOLO26's two heads side by side (the one-to-one head used as it is, the
   one-to-many head with NMS); the three detection grids; and the whole network as a zoomable diagram. The inspector
   covers this scene, the training run (curves from PyTorch), the two heads and the layer list. */
(function (NA) {
  'use strict';
  const { el, clear, clamp, fmt, fmtPct, fmtInt, theme } = NA;
  const H = NA.labHost, UI = NA.labUI, Y = NA.yolo;
  const NAMES = Y.NAMES, GLYPH = ['●', '▲', '■', '★'];
  const CLS = (i, a = 1) => (a >= 1 ? theme.series[i] : theme.series[i].replace('rgb(', 'rgba(').replace(')', `,${a})`));
  const HEADS = { o2o: 'One-to-one (no NMS)', o2m: 'One-to-many + NMS' };

  const YL = H.transport({
    id: 'yolo26', screens: { network: 'Detect', data: 'Scene', insp: 'Inspect' },
    labels: { play: 'Play', title: 'Set the shapes moving; the detector follows them (Space)', step: 'Move the shapes one frame (S)', reset: 'New random scene (R)' },
    head: 'o2o', conf: 0.25, iouThr: 0.7, view: 'detect', show: { truth: false, raw: false, cells: true }, sel: -1,
  });

  // The scene the lab opens on: one of each shape, two of them small, on a dark gradient with two stray lines.
  const DEFAULT = () => ({
    c0: Y.hsv(0.6, 0.28, 0.3), c1: Y.hsv(0.63, 0.22, 0.52), bang: 0.9, sigma: 0.02, seed: 11,
    lines: [{ x0: 6, y0: 104, x1: 66, y1: 122, w: 1.2, col: Y.hsv(0.1, 0.3, 0.75) }, { x0: 96, y0: 4, x1: 122, y1: 56, w: 1, col: Y.hsv(0.5, 0.2, 0.15) }],
    objs: [
      { cls: 3, cx: 36, cy: 38, size: 38, ang: 0.15, col: Y.hsv(0.13, 0.85, 0.98) },
      { cls: 0, cx: 94, cy: 34, size: 26, ang: 0, col: Y.hsv(0.95, 0.65, 0.92) },
      { cls: 2, cx: 82, cy: 88, size: 32, ang: 0.4, col: Y.hsv(0.52, 0.7, 0.9) },
      { cls: 1, cx: 34, cy: 92, size: 26, ang: 0.3, col: Y.hsv(0.33, 0.65, 0.85) },
      { cls: 1, cx: 112, cy: 114, size: 11, ang: 1.2, col: Y.hsv(0.08, 0.8, 1) },
      { cls: 0, cx: 62, cy: 64, size: 10, ang: 0, col: Y.hsv(0.78, 0.45, 1) },
    ],
  });

  // ------------------------------------------------------------------ lifecycle
  YL.activate = function (preset, ui) {
    this.ui = ui; this.preset = preset;
    this.ro = UI.readouts(ui.readouts, [{ key: 'found', label: 'Found', keep: true }, { key: 'boxes', label: 'Boxes' }, { key: 'ms', label: 'Network', keep: true, title: 'Time for one forward pass of the network' }]);
    this.buildHyper(ui.hyper);
    this.buildLeft(ui.left);
    this.buildCenter(ui.center);
    this.tabs = UI.tabs(ui.right, [{ id: 'overview', label: 'Overview' }, { id: 'train', label: 'Training' }, { id: 'heads', label: 'Two heads' }, { id: 'net', label: 'Network' }], (id, body) => this.buildTab(id, body));
    this.offTheme = theme.onChange(() => { this._ic = null; this.dirty = true; this.inspDirty = true; this.leftDirty = true; if (this.arch) this.arch.dirty = true; });
    if (!this.scene) this.scene = DEFAULT();
    if (NA.yoloWeights) this.ready();
    else {
      this.loading = true;
      H.loadScript('js/yolo-weights.js').then(() => this.ready(), () => { this.loadError = true; this.dirty = true; });
    }
    this.dirty = true; this.leftDirty = true; this.inspDirty = true;
  };
  YL.ready = function () {
    this.loading = false;
    this.net = Y.load(NA.yoloWeights);
    if (!this.runner) this.runner = new Y.Runner(this.net, (r) => this.onResult(r));
    if (H.lab === this) { this.changed(); this.inspDirty = true; }
  };
  YL.deactivate = function () {
    if (this.offTheme) this.offTheme();
    if (this.resizeObs) this.resizeObs.disconnect();
    if (this.arch) this.arch.destroy();
    this.drag = null;
  };
  YL.archText = function () { return 'YOLO26 layout · 128 px image → boxes at 3 scales · 2 heads'; };
  YL.stateText = function () { return this.training ? 'Moving' : this.loading ? 'Loading' : 'Still'; };
  YL.fastForward = function () {};
  YL.trainChunk = function () { return 0; };

  // the scene changed: ask for a new detection (the runner keeps only the newest request)
  YL.changed = function () {
    this.version = (this.version || 0) + 1;
    this.dirty = true; this.leftDirty = true;
    if (this.runner) this.runner.request(this.scene);
  };
  YL.onResult = function (r) {
    this.result = r; this._ic = null;
    this.dec = { o2m: Y.decode(r.F.heads.o2m), o2o: Y.decode(r.F.heads.o2o) };
    this.redetect();
  };
  // thresholds changed (or a new result): boxes from the decoded output, then matched against the true shapes
  YL.redetect = function () {
    const D = this.dec;
    if (!D) return;
    const t0 = performance.now(), o2m = Y.oneToMany(D.o2m, this.conf, this.iouThr), t1 = performance.now(), o2o = Y.oneToOne(D.o2o, this.conf), t2 = performance.now();
    const raw = Y.oneToMany(D.o2m, this.conf, this.iouThr, true);
    this.dets = { o2m, o2o, raw, nmsMs: t1 - t0, topMs: t2 - t1 };
    this.gts = this.result.scene.objs.map((o) => ({ cls: o.cls, box: Y.boxOf(o), size: o.size }));
    this.match = { o2o: Y.match(o2o, this.gts), o2m: Y.match(o2m, this.gts), raw: Y.match(raw, this.gts) };
    this.dirty = true; this.leftDirty = true; this.inspDirty = true; this.archStale = true; this.infoStale = true;
  };
  YL.headDets = function (h) { return this.dets ? this.dets[h || this.head] : []; };

  // the network's input as a 128 x 128 canvas (what it saw, so the boxes and the picture always agree)
  YL.imageCanvas = function () {
    const r = this.result;
    if (!r) return null;
    if (this._ic && this._ic.r === r) return this._ic.c;
    const c = document.createElement('canvas'); c.width = c.height = Y.IMG;
    const g = c.getContext('2d'), id = g.createImageData(Y.IMG, Y.IMG), P = Y.IMG * Y.IMG;
    for (let p = 0; p < P; p++) id.data.set([r.img[p] * 255, r.img[P + p] * 255, r.img[2 * P + p] * 255, 255], 4 * p);
    g.putImageData(id, 0, 0);
    this._ic = { r, c };
    return c;
  };

  // ------------------------------------------------------------------ scenes and shapes
  YL.newScene = function (seed) {
    const rng = new NA.RNG(seed ?? (Math.random() * 1e9) | 0);
    this.scene = Y.randomScene(rng);
    this.sel = -1; this.savedLines = null;
    if (this.clutterChip) this.clutterChip.setAttribute('aria-pressed', 'true');
    if (this.noiseSl) this.noiseSl.set(this.scene.sigma);
    this.changed();
  };
  YL.addShape = function (cls, x, y, size) {
    const rng = new NA.RNG((Math.random() * 1e9) | 0), S = this.scene;
    let o = null;
    for (let t = 0; t < 30 && !o; t++) {
      const cand = Y.randomShape(rng, S, cls);
      if (!cand) continue;
      if (x !== undefined) { cand.cx = x; cand.cy = y; if (size) cand.size = size; Y.clampShape(cand); cand.col = Y.pickColour(rng, S, cand.cx, cand.cy) || cand.col; }
      if (x !== undefined || S.objs.every((p) => Y.iou(Y.boxOf(cand), Y.boxOf(p)) < 0.2)) o = cand;
    }
    if (!o) { NA.ui.toast.show('No free room for another shape: remove one first.'); return; }
    S.objs.push(o);
    this.sel = S.objs.length - 1;
    this.changed();
  };
  YL.removeShape = function (i) {
    if (i < 0 || i >= this.scene.objs.length) return;
    this.scene.objs.splice(i, 1);
    this.sel = -1;
    this.changed();
  };
  YL.hit = function (ix, iy) {
    const objs = this.scene.objs;
    for (let i = objs.length - 1; i >= 0; i--) if (Y.shapeDist(objs[i], ix, iy) <= 1.5) return i;
    return -1;
  };
  // animation: each shape drifts and turns, bouncing off the edges
  YL.move = function (dt) {
    for (const o of this.scene.objs) {
      if (o.vx === undefined) { const a = Math.random() * Math.PI * 2, sp = 12 + Math.random() * 18; o.vx = Math.cos(a) * sp; o.vy = Math.sin(a) * sp; o.va = (Math.random() - 0.5) * 1.2; }
      const R = o.size / 2;
      o.cx += o.vx * dt; o.cy += o.vy * dt; o.ang += o.va * dt;
      if (o.cx < R / 2 || o.cx > Y.IMG - R / 2) { o.vx = -o.vx; o.cx = clamp(o.cx, R / 2, Y.IMG - R / 2); }
      if (o.cy < R / 2 || o.cy > Y.IMG - R / 2) { o.vy = -o.vy; o.cy = clamp(o.cy, R / 2, Y.IMG - R / 2); }
    }
    this.step++;
    this.changed();
  };
  YL.stepOnce = function () { if (this.training) this.pause(); this.move(1 / 12); };
  YL.reset = function () { if (this.training) this.pause(); this.newScene(); NA.ui.toast.show('A new random scene, drawn like the training data.'); };

  YL.frame = function (now) {
    if (this.training && this.scene) {
      // the shapes move in real time, but only between detections, so the boxes keep up with them
      this.owed = Math.min(0.12, (this.owed || 0) + (this.lastMove ? (now - this.lastMove) / 1000 : 0));
      if (this.owed > 0.004 && (!this.runner || !this.runner.busy)) { this.move(this.owed); this.owed = 0; }
      this.lastMove = now;
    } else { this.lastMove = 0; this.owed = 0; }
    if (now - (this.lastRo || 0) > 150) {
      this.lastRo = now;
      const m = this.match && this.match[this.head];
      this.ro.set('found', this.gts ? `${m.found} / ${this.gts.length}` : '—');
      this.ro.set('boxes', this.dets ? String(this.headDets().length) : '—');
      this.ro.set('ms', this.runner && this.runner.ms ? `${fmt(this.runner.ms, 0)} ms` : '—');
    }
    if (this.dirty && this.canvas.offsetParent) { this.dirty = false; this.draw(); }
    if (this.view === 'arch' && this.arch) {
      if (this.archStale && (!this.training || now - (this.lastArch || 0) > 500)) { this.lastArch = now; this.archStale = false; this.arch.dirty = true; }
      this.arch.frame(now);
    }
    if (this.infoStale && !this.training && !(this.arch && this.arch.anim)) { this.infoStale = false; this.showInfo(); }
    if (this.leftDirty && this.ui.left.offsetParent && (!this.training || now - (this.lastLeft || 0) > 400)) { this.lastLeft = now; this.leftDirty = false; this.renderLeft(); }
    if (this.inspDirty && this.tabs.body.offsetParent && (!this.training || now - (this.lastInsp || 0) > 700)) { this.lastInsp = now; this.inspDirty = false; this.updateTab(); }
  };

  YL.buildHyper = function (root) {
    root.append(
      UI.field('Head', UI.select([{ value: 'o2o', label: HEADS.o2o }, { value: 'o2m', label: HEADS.o2m }], this.head, (v) => this.setHead(v))),
      UI.field('Confidence', UI.select([0.1, 0.25, 0.4, 0.6].map((v) => ({ value: v, label: '≥ ' + v })), this.conf, (v) => this.setConf(+v))),
      UI.field('NMS IoU', UI.select([0.45, 0.6, 0.7, 0.8].map((v) => ({ value: v, label: '> ' + v })), this.iouThr, (v) => this.setIou(+v)), 'NMS deletes a box that overlaps a better one of its class by more than this'),
      el('div', { class: 'sheet-actions' }, el('button', { class: 'btn', type: 'button', onclick: () => { NA.mobile && NA.mobile.sheet(false); this.reset(); } }, 'New scene')));
  };
  YL.setHead = function (h) { this.head = h; if (this.headSeg) this.headSeg.set(h); this.dirty = true; this.leftDirty = true; this.inspDirty = true; this.archStale = true; this.infoStale = true; };
  YL.setConf = function (v) { this.conf = v; if (this.confSl) this.confSl.set(v); this.redetect(); };
  YL.setIou = function (v) { this.iouThr = v; if (this.iouSl) this.iouSl.set(v); this.redetect(); };

  // ------------------------------------------------------------------ left: the scene
  YL.buildLeft = function (root) {
    const body = UI.panel(root, 'Scene');
    this.noiseSl = UI.slider({ label: 'Sensor noise σ', min: 0, max: 0.15, step: 0.005, value: this.scene ? this.scene.sigma : 0.02, format: (v) => fmt(v, 3), onInput: (v) => { this.scene.sigma = v; this.changed(); } });
    this.clutterChip = UI.chip('Clutter lines', true, (on) => {
      const S = this.scene;
      if (!on) { this.savedLines = S.lines; S.lines = []; } else { S.lines = this.savedLines || S.lines; this.savedLines = null; }
      this.changed();
    }, 'Thin lines that are not objects: the detector has to ignore them');
    const add = el('div', { class: 'yl-add' }, ...NAMES.map((n, k) => el('button', { class: 'btn yl-add-btn', type: 'button', title: `Add a ${n}`, onclick: () => this.addShape(k) }, el('span', { class: 'yl-glyph', style: { color: CLS(k) }, text: GLYPH[k] }), n)));
    this.selBox = el('div');
    this.found = el('div', { class: 'yl-found' });
    body.append(
      UI.section('Scene', el('div', { class: 'btn-row' },
        el('button', { class: 'btn btn-primary', type: 'button', onclick: () => this.reset() }, 'New random scene'),
        el('button', { class: 'btn', type: 'button', onclick: () => { this.scene.objs = []; this.sel = -1; this.changed(); } }, 'Clear')),
      UI.note('Drag a shape to move it, scroll on it to resize (Shift + scroll turns it), double-click it to remove it, double-click empty space to add one.'),
      add, this.noiseSl, el('div', { class: 'chip-row' }, this.clutterChip)),
      UI.section('Selected shape', this.selBox),
      UI.section('What the detector found', this.found),
      UI.section('How a head reads its grids',
        el('div', { class: 'fx' },
          el('div', { class: 'fx-line fx-strong', text: 'box = cell centre ± (l, t, r, b) × stride' }),
          el('div', { class: 'fx-line', text: 'score = σ(class logit), one per class' }),
          el('div', { class: 'fx-line', text: 'one-to-many: score ≥ threshold, then NMS' }),
          el('div', { class: 'fx-line', text: 'one-to-one: score ≥ threshold. Done.' })),
        UI.note('Each of 336 cells (16 × 16 + 8 × 8 + 4 × 4) predicts four distances to the edges of a box and a score for each shape. YOLO26 predicts the distances directly (no DFL) and, with its one-to-one head, needs no NMS.')));
  };
  YL.renderLeft = function () {
    // the selected shape's controls
    clear(this.selBox);
    const o = this.scene.objs[this.sel];
    if (!o) this.selBox.append(UI.note('Click a shape in the picture to select it.'));
    else {
      const seg = UI.seg(NAMES.map((n, k) => ({ value: k, label: GLYPH[k] + ' ' + n })), o.cls, (v) => { o.cls = +v; this.changed(); }, 'Kind of shape');
      this.selBox.append(seg,
        UI.slider({ label: 'Size', min: Y.SIZE[0], max: Y.SIZE[1], step: 0.5, value: o.size, format: (v) => `${fmt(v, 1)} px`, onInput: (v) => { o.size = v; Y.clampShape(o); this.changed(); } }),
        UI.slider({ label: 'Rotation', min: 0, max: 360, step: 1, value: Math.round(((o.ang * 180) / Math.PI % 360 + 360) % 360), format: (v) => `${v}°`, onInput: (v) => { o.ang = (v * Math.PI) / 180; this.changed(); } }),
        el('div', { class: 'btn-row' }, el('button', { class: 'btn', type: 'button', onclick: () => this.removeShape(this.sel) }, 'Remove it')));
    }
    // the true shapes against the current head's detections
    clear(this.found);
    if (!this.dets) { this.found.append(UI.note(this.loadError ? 'The detector’s weights could not be loaded.' : 'Running the detector…')); return; }
    const dets = this.headDets(), M = this.match[this.head], rows = [];
    this.gts.forEach((g, j) => {
      const d = dets.find((q) => q.hit === j);
      rows.push(el('div', { class: 'yl-row' + (j === this.sel ? ' is-sel' : ''), onclick: () => { this.sel = j; this.dirty = true; this.leftDirty = true; } },
        el('span', { class: 'yl-glyph', style: { color: CLS(g.cls) }, text: GLYPH[g.cls] }),
        el('span', { class: 'yl-name', text: `${NAMES[g.cls]}, ${fmt(g.size, 0)} px` }),
        d ? UI.pill('good', `${fmt(d.score, 2)} · IoU ${fmt(d.iou, 2)}`) : UI.pill('warn', 'missed')));
    });
    if (!rows.length) rows.push(UI.note('No shapes: add one, or ask for a new scene.'));
    const extra = M.dups + M.wrong;
    this.found.append(...rows, UI.note(extra ? `${extra} extra box${extra > 1 ? 'es' : ''}: ${M.dups ? `${M.dups} duplicate${M.dups > 1 ? 's' : ''}` : ''}${M.dups && M.wrong ? ', ' : ''}${M.wrong ? `${M.wrong} on nothing or the wrong kind` : ''}.` : `${HEADS[this.head]}: no extra boxes.`));
  };

  // ------------------------------------------------------------------ centre: the picture
  YL.buildCenter = function (root) {
    this.viewSeg = UI.seg([{ value: 'detect', label: 'Detect', title: 'The picture and the boxes of one head' }, { value: 'compare', label: 'Compare heads', title: 'Before NMS, after NMS, and the one-to-one head' }, { value: 'scales', label: 'Grids', title: 'The three grids each head reads: what each cell scores' }, { value: 'arch', label: 'Architecture', title: 'The network’s parts: click or zoom in to open them' }], this.view, (v) => this.setView(v), 'View');
    this.headSeg = UI.seg([{ value: 'o2o', label: 'One-to-one', title: 'YOLO26’s end-to-end head: used as it is, no NMS' }, { value: 'o2m', label: 'One-to-many + NMS', title: 'The classic head: several boxes per object, cleaned up by NMS' }], this.head, (v) => this.setHead(v), 'Head');
    this.chips = el('div', { class: 'chip-row' },
      UI.chip('Truth', this.show.truth, (on) => { this.show.truth = on; this.dirty = true; }, 'The true box of every shape (white, dashed)'),
      UI.chip('Raw boxes', this.show.raw, (on) => { this.show.raw = on; this.dirty = true; }, 'One-to-many: every box over the threshold, before NMS'),
      UI.chip('Cells', this.show.cells, (on) => { this.show.cells = on; this.dirty = true; }, 'Mark the grid cell that predicted each box'));
    root.append(el('div', { class: 'lab-toolbar' }, this.viewSeg, this.headSeg, this.chips));
    this.stage = el('div', { class: 'lab-stage yl-stage' });
    this.canvas = el('canvas', { 'aria-label': 'The scene with the detector’s boxes. Drag shapes to move them.', tabindex: '0' });
    this.stage.append(this.canvas);
    this.archStage = el('div', { class: 'lab-stage arch-stage' });
    this.arch = new NA.ArchView(this.archStage, { label: 'YOLO26 architecture: click a part or zoom in to open it', build: (size) => NA.archYolo.build(this, size), onFocus: (node, path) => { if (this.view === 'arch') this.info.set(this.arch.infoFor(node, path)); } });
    this.confSl = UI.slider({ label: 'Confidence ≥', min: 0.05, max: 0.95, step: 0.01, value: this.conf, format: (v) => fmt(v, 2), onInput: (v) => { this.conf = v; this.redetect(); } });
    this.iouSl = UI.slider({ label: 'NMS IoU >', min: 0.3, max: 0.95, step: 0.01, value: this.iouThr, format: (v) => fmt(v, 2), onInput: (v) => { this.iouThr = v; this.redetect(); } });
    this.controls = el('div', { class: 'yl-controls' }, this.confSl, this.iouSl);
    root.append(this.stage, this.archStage, this.controls);
    this.info = UI.infoPanel(root, { open: !(NA.mobile && NA.mobile.mobile) && root.clientHeight >= 680 }); // open at first only when there is room for it under the picture
    root.append(this.foot = el('div', { class: 'lab-foot' },
      ...NAMES.map((n, k) => el('span', { class: 'lg-item' }, el('i', { class: 'key-line', style: { background: CLS(k) } }), n)),
      el('span', { class: 'lg-item' }, el('i', { class: 'key-line yl-dash' }), 'removed by NMS'),
      el('span', { class: 'lg-item' }, el('i', { class: 'dot', style: { background: 'var(--focus)' } }), 'the cell that predicted it')));
    const c = this.canvas;
    c.addEventListener('pointerdown', (e) => this.pointerDown(e));
    c.addEventListener('pointermove', (e) => this.pointerMove(e));
    c.addEventListener('pointerup', (e) => this.pointerUp(e));
    c.addEventListener('pointercancel', (e) => this.pointerUp(e));
    c.addEventListener('pointerleave', () => { NA.tooltip.hide(); if (this.hoverI !== -1) { this.hoverI = -1; this.dirty = true; } });
    c.addEventListener('dblclick', (e) => this.dblclick(e));
    c.addEventListener('wheel', (e) => this.wheel(e), { passive: false });
    this.resizeObs = new ResizeObserver(() => { this.dirty = true; });
    this.resizeObs.observe(this.stage);
    this.setView(this.view);
  };
  YL.setView = function (v) {
    this.view = v;
    this.viewSeg.set(v);
    const arch = v === 'arch';
    this.stage.hidden = arch; this.chips.hidden = arch || v === 'compare'; this.headSeg.hidden = arch || v === 'compare'; this.foot.hidden = arch; this.controls.hidden = arch; this.archStage.hidden = !arch;
    this.dirty = true; this.archStale = true;
    this.showInfo();
  };
  YL.showInfo = function () {
    if (this.view === 'arch') { if (this.arch) this.arch.refreshFocus(); return; }
    this.info.set((this.view === 'detect' ? DETECT_INFO : this.view === 'compare' ? COMPARE_INFO : SCALES_INFO)(this));
  };

  // panels: the picture once (Detect) or three times (Compare heads, Grids), laid out to be as large as possible
  YL.layout = function (W, Hh, n) {
    const cap = 22, gap = 14;
    if (n === 1) { const s = Math.max(60, Math.min(W - 2 * gap, Hh - 2 * gap - cap)); return [{ x: (W - s) / 2, y: (Hh - s - cap) / 2 + cap, s }]; }
    let best = null;
    for (const [cols, rows] of [[3, 1], [1, 3], [2, 2]]) {
      const s = Math.min((W - (cols + 1) * gap) / cols, (Hh - (rows + 1) * gap - rows * cap) / rows);
      if (!best || s > best.s) best = { cols, rows, s };
    }
    const { cols, rows, s } = best, ox = (W - cols * s - (cols - 1) * gap) / 2, oy = (Hh - rows * (s + cap) - (rows - 1) * gap) / 2;
    return Array.from({ length: n }, (_, i) => ({ x: ox + (i % cols) * (s + gap) + (rows === 2 && i === 2 ? (s + gap) / 2 : 0), y: oy + Math.floor(i / cols) * (s + cap + gap) + cap, s: Math.max(40, s) }));
  };
  YL.draw = function () {
    const W = Math.max(200, this.stage.clientWidth), Hh = Math.max(200, this.stage.clientHeight), ctx = NA.fitCanvas(this.canvas, W, Hh);
    ctx.clearRect(0, 0, W, Hh);
    this.panels = [];
    if (!this.result) {
      const msg = this.loadError ? 'The detector’s weights could not be loaded.' : 'Loading the trained detector (1 MB)…';
      NA.archKit.text(ctx, msg, W / 2, Hh / 2, { align: 'center', color: theme.c('muted'), size: 13 });
      return;
    }
    const d = this.dets;
    if (this.view === 'detect') {
      const [q] = this.layout(W, Hh, 1), list = this.headDets();
      this.panel(ctx, q, { title: `${HEADS[this.head]}: ${list.length} box${list.length === 1 ? '' : 'es'}`, dets: list, raw: this.show.raw && this.head === 'o2m' ? d.raw : null, labels: true });
    } else if (this.view === 'compare') {
      const qs = this.layout(W, Hh, 3), rm = d.o2m.suppressed || [];
      this.panel(ctx, qs[0], { title: `One-to-many: ${d.raw.length} boxes`, dets: d.raw, thin: true, counts: true });
      this.panel(ctx, qs[1], { title: `+ NMS: ${d.o2m.length} kept, ${rm.length} removed`, dets: d.o2m, removed: rm, labels: true });
      this.panel(ctx, qs[2], { title: `One-to-one: ${d.o2o.length} boxes, no NMS`, dets: d.o2o, labels: true });
    } else {
      const qs = this.layout(W, Hh, 3), list = this.headDets();
      Y.STRIDES.forEach((s, si) => {
        const mine = list.filter((q) => this.dec[this.head].where[q.a].si === si), g = Y.GRIDS[si];
        this.panel(ctx, qs[si], { title: `${g} × ${g} grid (stride ${s}): ${mine.length} box${mine.length === 1 ? '' : 'es'}`, dets: mine, grid: si, labels: false });
      });
    }
  };
  // one picture with its boxes. o: { title, dets, raw, removed, grid (scale index), labels, thin }
  YL.panel = function (ctx, q, o) {
    this.panels.push(q);
    const k = q.s / Y.IMG, X = (v) => q.x + v * k, Yp = (v) => q.y + v * k, S = NA.archKit;
    S.text(ctx, o.title, q.x, q.y - 11, { size: 12, w: 600, color: theme.c('ink-2'), maxW: q.s });
    ctx.save();
    ctx.beginPath(); NA.rr(ctx, q.x, q.y, q.s, q.s, 6); ctx.clip();
    ctx.imageSmoothingEnabled = false;
    ctx.drawImage(this.imageCanvas(), q.x, q.y, q.s, q.s);
    if (o.grid !== undefined) {
      // each cell's best class score, over the picture
      const g = Y.GRIDS[o.grid], D = this.dec[this.head], off = Y.GRIDS.slice(0, o.grid).reduce((s, n) => s + n * n, 0), cs = q.s / g;
      for (let i = 0; i < g; i++) for (let j = 0; j < g; j++) {
        let m = 0, c = 0;
        for (let t = 0; t < Y.NC; t++) { const v = D.scores[(off + i * g + j) * Y.NC + t]; if (v > m) { m = v; c = t; } }
        ctx.fillStyle = m > 0.05 ? CLS(c, Math.min(0.85, 0.15 + m * 0.8)) : 'rgba(0,0,0,0.28)';
        ctx.fillRect(q.x + j * cs, q.y + i * cs, cs, cs);
      }
      ctx.strokeStyle = 'rgba(255,255,255,0.25)'; ctx.lineWidth = 1;
      ctx.beginPath();
      for (let i = 1; i < g; i++) { ctx.moveTo(q.x + i * cs, q.y); ctx.lineTo(q.x + i * cs, q.y + q.s); ctx.moveTo(q.x, q.y + i * cs); ctx.lineTo(q.x + q.s, q.y + i * cs); }
      ctx.stroke();
    }
    ctx.restore();
    ctx.strokeStyle = theme.c('line-2'); ctx.lineWidth = 1; ctx.beginPath(); NA.rr(ctx, q.x - 0.5, q.y - 0.5, q.s + 1, q.s + 1, 6); ctx.stroke();
    const box = (b, col, lw, dash) => { ctx.strokeStyle = col; ctx.lineWidth = lw; if (dash) ctx.setLineDash(dash); ctx.strokeRect(X(b[0]), Yp(b[1]), (b[2] - b[0]) * k, (b[3] - b[1]) * k); ctx.setLineDash([]); };
    if (this.show.truth && o.grid === undefined) for (const g of this.gts) { box(g.box, 'rgba(0,0,0,0.65)', 3, [5, 4]); box(g.box, 'rgba(255,255,255,0.95)', 1.4, [5, 4]); }
    if (o.raw) for (const d of o.raw) box(d.box, CLS(d.cls, 0.45), 1);
    if (o.removed) for (const d of o.removed) box(d.box, theme.c('muted', 0.85), 1.2, [3, 3]);
    for (const d of o.dets) {
      box(d.box, o.thin ? CLS(d.cls, 0.75) : CLS(d.cls), o.thin ? 1.1 : 2.2);
      if (o.counts) {
        // every raw box comes from its own cell: one dot per cell shows how many cells answered for each shape
        const w = this.dec.o2m.where[d.a];
        ctx.fillStyle = CLS(d.cls); ctx.strokeStyle = 'rgba(0,0,0,0.55)'; ctx.lineWidth = 1;
        ctx.beginPath(); ctx.arc(X((w.gx + 0.5) * w.s), Yp((w.gy + 0.5) * w.s), Math.max(1.8, q.s / 150), 0, Math.PI * 2); ctx.fill(); ctx.stroke();
      }
      if (this.show.cells && !o.thin) {
        const w = this.dec[o.dets === this.dets.o2o ? 'o2o' : 'o2m'].where[d.a], cx = X((w.gx + 0.5) * w.s), cy = Yp((w.gy + 0.5) * w.s);
        ctx.fillStyle = theme.c('focus'); ctx.strokeStyle = 'rgba(255,255,255,0.9)'; ctx.lineWidth = 1.5;
        ctx.beginPath(); ctx.arc(cx, cy, 3.2, 0, Math.PI * 2); ctx.fill(); ctx.stroke();
      }
      if (o.labels && q.s > 180) {
        const t = `${NAMES[d.cls]} ${fmt(d.score, 2)}`;
        ctx.font = `600 11px ${getComputedStyle(document.body).getPropertyValue('--font-ui') || 'sans-serif'}`;
        const tw = ctx.measureText(t).width + 8, lx = clamp(X(d.box[0]) - 1, q.x, q.x + q.s - tw), ly = Yp(d.box[1]) - 16 < q.y ? Yp(d.box[1]) + 1 : Yp(d.box[1]) - 16;
        ctx.fillStyle = CLS(d.cls); NA.rr(ctx, lx, ly, tw, 15, 3); ctx.fill();
        ctx.fillStyle = '#fff'; ctx.textAlign = 'left'; ctx.textBaseline = 'middle'; ctx.fillText(t, lx + 4, ly + 8);
      }
    }
    if (o.counts) {
      // how many raw boxes each shape got (same kind, IoU ≥ 0.5)
      ctx.font = `700 11px ${getComputedStyle(document.body).getPropertyValue('--font-ui') || 'sans-serif'}`;
      for (const g of this.gts) {
        const n = o.dets.filter((d) => d.cls === g.cls && Y.iou(d.box, g.box) >= 0.5).length;
        if (!n) continue;
        const t = `×${n}`, tw = ctx.measureText(t).width + 8, lx = clamp(X(g.box[2]) - tw + 2, q.x, q.x + q.s - tw), ly = clamp(Yp(g.box[1]) - 15, q.y, q.y + q.s - 15);
        ctx.fillStyle = 'rgba(0,0,0,0.72)'; NA.rr(ctx, lx, ly, tw, 14, 7); ctx.fill();
        ctx.fillStyle = '#fff'; ctx.textAlign = 'left'; ctx.textBaseline = 'middle'; ctx.fillText(t, lx + 4, ly + 7.5);
      }
    }
    // the selected shape, and the one being dragged at its newest position (the picture lags a frame behind)
    const S0 = this.scene.objs[this.sel];
    if (S0) {
      ctx.strokeStyle = theme.c('focus'); ctx.lineWidth = 1.5; ctx.setLineDash([4, 3]);
      ctx.beginPath();
      if (S0.cls === 0) ctx.arc(X(S0.cx), Yp(S0.cy), (S0.size / 2) * k + 3, 0, Math.PI * 2);
      else { const V = Y.vertices(S0), n = S0.cls === 1 ? 3 : S0.cls === 2 ? 4 : 10; V.slice(0, n).forEach(([x, y], i) => (i ? ctx.lineTo(X(x), Yp(y)) : ctx.moveTo(X(x), Yp(y)))); ctx.closePath(); }
      ctx.stroke(); ctx.setLineDash([]);
    }
  };
  // pointer → (panel, image coordinates)
  YL.toImage = function (e) {
    const r = this.canvas.getBoundingClientRect(), px = e.clientX - r.left, py = e.clientY - r.top;
    for (const q of this.panels || []) if (px >= q.x && py >= q.y && px <= q.x + q.s && py <= q.y + q.s) return { q, ix: ((px - q.x) / q.s) * Y.IMG, iy: ((py - q.y) / q.s) * Y.IMG };
    return null;
  };
  YL.pointerDown = function (e) {
    const p = this.toImage(e);
    if (!p || !this.result) return;
    const i = this.hit(p.ix, p.iy);
    this.sel = i; this.leftDirty = true; this.dirty = true;
    if (i < 0) return;
    const o = this.scene.objs[i];
    this.drag = { i, dx: o.cx - p.ix, dy: o.cy - p.iy, id: e.pointerId };
    try { this.canvas.setPointerCapture(e.pointerId); } catch (err) { /* a pointer that is already gone */ }
    this.canvas.style.cursor = 'grabbing';
    e.preventDefault();
  };
  YL.pointerMove = function (e) {
    const p = this.toImage(e);
    if (this.drag) {
      if (!p) return;
      const o = this.scene.objs[this.drag.i];
      o.cx = p.ix + this.drag.dx; o.cy = p.iy + this.drag.dy; Y.clampShape(o);
      this.changed();
      return;
    }
    const i = p ? this.hit(p.ix, p.iy) : -1;
    this.canvas.style.cursor = i >= 0 ? 'grab' : 'default';
    if (i !== this.hoverI) { this.hoverI = i; }
    // the box under the pointer: which cell made it and how sure it is
    const list = this.view === 'detect' ? this.headDets() : [];
    const d = p && list.find((q) => p.ix >= q.box[0] && p.ix <= q.box[2] && p.iy >= q.box[1] && p.iy <= q.box[3]);
    if (d && this.dec) {
      const w = this.dec[this.head].where[d.a];
      NA.tooltip.show([el('div', { class: 'tt-title', text: `${NAMES[d.cls]} ${fmt(d.score, 3)}` }), el('div', { class: 'tt-hint', text: `from cell (${w.gx}, ${w.gy}) of the ${Y.GRIDS[w.si]} × ${Y.GRIDS[w.si]} grid` }), el('div', { class: 'tt-hint', text: d.hit >= 0 ? `matches a ${NAMES[this.gts[d.hit].cls]}, IoU ${fmt(d.iou, 2)}` : d.dup ? 'a duplicate of a box already found' : 'on nothing, or the wrong kind' })], e.clientX, e.clientY);
    } else NA.tooltip.hide();
  };
  YL.pointerUp = function () {
    if (!this.drag) return;
    this.drag = null;
    this.canvas.style.cursor = 'grab';
    this.leftDirty = true;
  };
  YL.dblclick = function (e) {
    const p = this.toImage(e);
    if (!p || !this.result) return;
    const i = this.hit(p.ix, p.iy);
    if (i >= 0) this.removeShape(i); else this.addShape(Math.floor(Math.random() * Y.NC), p.ix, p.iy, 14 + Math.random() * 16);
  };
  YL.wheel = function (e) {
    const p = this.toImage(e);
    if (!p) return;
    const i = this.hit(p.ix, p.iy);
    if (i < 0) return;
    e.preventDefault();
    const o = this.scene.objs[i], dy = e.deltaMode === 1 ? e.deltaY * 16 : e.deltaY;
    if (e.shiftKey) o.ang += dy * 0.004;
    else { o.size = clamp(o.size * Math.exp(-dy * 0.0015), Y.SIZE[0], Y.SIZE[1]); Y.clampShape(o); }
    this.sel = i;
    this.changed();
  };
  YL.key = function (e) {
    const o = this.scene && this.scene.objs[this.sel];
    if (!o) return;
    const k = e.key, step = e.shiftKey ? 5 : 1;
    if (k === 'Delete' || k === 'Backspace') { e.preventDefault(); this.removeShape(this.sel); return; }
    const mv = { ArrowLeft: [-step, 0], ArrowRight: [step, 0], ArrowUp: [0, -step], ArrowDown: [0, step] }[k];
    if (mv) { e.preventDefault(); o.cx += mv[0]; o.cy += mv[1]; Y.clampShape(o); this.changed(); }
    else if (k === '[' || k === ']') { o.ang += (k === '[' ? -1 : 1) * 0.15; this.changed(); }
    else if (k === '-' || k === '=' || k === '+') { o.size = clamp(o.size * (k === '-' ? 0.92 : 1.08), Y.SIZE[0], Y.SIZE[1]); Y.clampShape(o); this.changed(); }
  };

  // ------------------------------------------------------------------ explanations for the three picture views
  const PRE = '# model, random_scene, decode, flat and detect: the whole-model code (Architecture view, top level, Python tab)';
  const DETECT_INFO = (L) => {
    const d = L.dets, h = L.head, M = L.match && L.match[h], n = L.gts ? L.gts.length : 0;
    return {
      kicker: 'Detect view', title: `${HEADS[h]}: ${d ? d[h].length : 0} boxes for ${n} shapes`,
      about: [
        `The network sees this 128 × 128 picture and, in one forward pass (${L.runner && L.runner.ms ? `${fmt(L.runner.ms, 0)} ms in your browser` : 'in your browser'}), predicts a box and four class scores at each of 336 cells. Boxes scoring at least ${fmt(L.conf, 2)} are drawn; the dot marks the cell that predicted each one. ${M ? `Right now it finds ${M.found} of the ${n} shapes${M.dups + M.wrong ? ` and adds ${M.dups + M.wrong} extra box${M.dups + M.wrong > 1 ? 'es' : ''}` : ''}.` : ''}`,
        h === 'o2o'
          ? 'This is YOLO26’s one-to-one head, its default. It was trained so that only one cell answers for each object, so its boxes are used exactly as they come out: no non-maximum suppression, no IoU threshold to tune. Switch to the one-to-many head and turn on Raw boxes to see what it saves.'
          : `This is the one-to-many head, the way YOLO worked up to YOLO11: several neighbouring cells answer for each object, and NMS keeps the best box and deletes the ones overlapping it by more than ${fmt(L.iouThr, 2)} IoU. Turn on Raw boxes to see everything NMS removed.`,
        'Move the shapes and the boxes follow: every change re-runs the whole network. Try a tiny shape, a big one, two of a kind touching, or noise well past the 0.05 the detector saw in training.',
      ],
      code: `import matplotlib.pyplot as plt
import matplotlib.patches as patches
${PRE}
img, true_boxes, true_classes = random_scene(np.random.default_rng(7))
boxes, scores, classes = detect(model, img, head='${h}', conf=${fmt(L.conf, 2)}${h === 'o2m' ? `, iou=${fmt(L.iouThr, 2)}` : ''})
fig, ax = plt.subplots()
ax.imshow(img.permute(1, 2, 0))
for b, s, c in zip(boxes, scores, classes):
    x1, y1, x2, y2 = b.tolist()
    ax.add_patch(patches.Rectangle((x1, y1), x2 - x1, y2 - y1, fill=False, color=f'C{int(c)}', lw=2))
    ax.text(x1, y1 - 2, f'{NAMES[c]} {s:.2f}', color=f'C{int(c)}')
plt.show()`,
      codeLabel: 'Python · PyTorch and matplotlib · detect and draw',
    };
  };
  const COMPARE_INFO = (L) => {
    const d = L.dets, rm = d ? (d.o2m.suppressed || []).length : 0;
    return {
      kicker: 'Compare heads', title: d ? `${d.raw.length} raw boxes → ${d.o2m.length} after NMS, against ${d.o2o.length} from the one-to-one head` : 'The two heads',
      about: [
        `Left: everything the one-to-many head scores at least ${fmt(L.conf, 2)}. Each shape is covered by several boxes from neighbouring cells, because in training each object was matched to its 10 best cells, and all of them learned to fire. Middle: NMS keeps the best of each overlapping group and deletes ${rm} boxes (dashed). Right: the one-to-one head, trained with exactly one cell per object, already gives one box each.`,
        `On 500 validation scenes the raw one-to-many head puts ${L.W ? fmt(L.W.final.raw.boxes_per_obj, 1) : 'about 3'} boxes on every shape it finds; after NMS that drops to ${L.W ? fmt(L.W.final.o2m.boxes_per_obj, 2) : '1.0'}, and the one-to-one head gives ${L.W ? fmt(L.W.final.o2o.boxes_per_obj, 2) : '1.0'} with no NMS at all. Its accuracy is the better of the two here: mAP@0.5 ${L.W ? fmtPct(L.W.final.o2o.map50, 1) : ''} against ${L.W ? fmtPct(L.W.final.o2m.map50, 1) : ''} with NMS.`,
        'Why it matters: NMS is a loop over boxes whose cost depends on the picture, it needs a threshold that is wrong for some scenes (two real objects that overlap a lot get merged), and it is awkward to export to phones and accelerators. Dropping it makes the network’s output final, which is a large part of why Ultralytics reports YOLO26n running up to 43% faster on CPUs than YOLO11n.',
      ],
      code: `${PRE}
img, _, _ = random_scene(np.random.default_rng(7))
with torch.no_grad():
    o2m, o2o = model.eval()(img[None])
_, logits = decode(flat(o2m))
print('one-to-many, before NMS:', (logits[0].sigmoid().max(1).values > ${fmt(L.conf, 2)}).sum().item())
print('one-to-many, after NMS: ', len(detect(model, img, 'o2m', conf=${fmt(L.conf, 2)}, iou=${fmt(L.iouThr, 2)})[0]))
print('one-to-one, no NMS:     ', len(detect(model, img, 'o2o', conf=${fmt(L.conf, 2)})[0]))`,
      codeLabel: 'Python · PyTorch · count the boxes of each head',
    };
  };
  const SCALES_INFO = (L) => ({
    kicker: 'Grids view', title: `Where ${HEADS[L.head].toLowerCase()} finds each shape`,
    about: [
      'The head reads three feature maps of the neck: 16 × 16 cells of 8 pixels, 8 × 8 cells of 16 pixels and 4 × 4 cells of 32 pixels. Each cell is coloured by its best class score (the colour of that class, stronger when surer). Every box comes from one cell of one grid.',
      'Nobody told the network which grid to use: in training, each shape was simply assigned to the cells whose boxes already fitted it best, on any grid. On these small pictures the fine grid wins almost everything. Over 250 random scenes it made every box for shapes under 24 pixels, and about half of those 32 to 40 pixels across; the rest went to the 8 × 8 grid. The 4 × 4 grid never won, because the largest shapes (44 pixels) are a third of the picture, and its 32-pixel cells place them less precisely than the finer grids. Grow a shape past about 30 pixels to see its box move to the 8 × 8 grid. On 640-pixel photos, YOLO26’s coarse grid is the one that finds objects a hundred pixels and more across.',
      'YOLO26’s small-target-aware assignment (STAL) helps the finest grid: a box smaller than a couple of cells is treated as 16 pixels wide when picking which cells may learn it, so even a 7-pixel shape always has cells to learn from.',
    ],
    code: `import matplotlib.pyplot as plt
${PRE}
img, _, _ = random_scene(np.random.default_rng(7))
with torch.no_grad():
    o2m, o2o = model.eval()(img[None])
fig, axes = plt.subplots(1, 3, figsize=(12, 4))
for ax, m, s in zip(axes, ${L.head}, STRIDES):
    best = m[0, 4:].sigmoid().max(0).values          # (cells, cells): each cell's best class score
    ax.imshow(img.permute(1, 2, 0))
    ax.imshow(best, extent=(0, IMG, IMG, 0), alpha=0.6, vmin=0, vmax=1)
    ax.set_title(f'stride {s}: {m.shape[-1]} x {m.shape[-1]} cells')
plt.show()`,
    codeLabel: 'Python · PyTorch and matplotlib · the three grids',
  });

  // ------------------------------------------------------------------ right: inspector
  YL.buildTab = function (id, body) {
    this.tabId = id; const P = (this.parts = {});
    if (id === 'overview') {
      body.append(UI.modelCard(this.preset, (t) => this.runTry(t)));
      P.findings = el('div', { class: 'diag-list' });
      body.append(UI.section('This scene', P.findings));
    } else if (id === 'train') {
      P.tiles = el('div', { class: 'tiles' });
      const c1 = el('canvas'), c2 = el('canvas'), c3 = el('canvas'), c4 = el('canvas');
      P.map = new NA.charts.LineChart(c1, { height: 150, format: (v) => fmtPct(v, 1), xLabel: 'Epoch', yFormat: (v) => fmtPct(v, 0) });
      P.prog = new NA.charts.LineChart(c2, { height: 120, format: (v) => fmt(v, 2), xLabel: 'Epoch' });
      P.loss = new NA.charts.LineChart(c3, { height: 130, format: (v) => fmt(v, 3), xLabel: 'Epoch' });
      P.small = new NA.charts.LineChart(c4, { height: 120, format: (v) => fmtPct(v, 1), xLabel: 'Epoch', yFormat: (v) => fmtPct(v, 0) });
      const key = (col, t) => el('span', { class: 'lg-item' }, el('i', { class: 'key-line', style: { background: col } }), t);
      body.append(P.tiles,
        UI.section('Accuracy on 500 validation scenes', el('div', { class: 'chart-box' }, c1), el('div', { class: 'mini-legend' }, key('var(--s7)', 'one-to-one, mAP@0.5'), key('var(--s4)', 'one-to-many + NMS, mAP@0.5'), key('var(--s1)', 'one-to-one, mAP@0.5:0.95')),
          UI.note('Mean average precision: for each kind of shape, precision averaged over every recall level, then averaged over the four kinds. @0.5 counts a box as right when it overlaps the true box by IoU 0.5 or more; @0.5:0.95 averages ten stricter thresholds, so it also rewards tight boxes.')),
        UI.section('ProgLoss: which head the training listens to', el('div', { class: 'chart-box' }, c2), el('div', { class: 'mini-legend' }, key('var(--s4)', 'one-to-many weight'), key('var(--s7)', 'one-to-one weight')),
          UI.note('Both heads are trained at once on the same features. Early on most of the loss comes from the one-to-many head, whose ten positives per object give a strong, easy signal for learning features; the weight then moves linearly to the one-to-one head, the one used at inference (0.8 → 0.1 and 0.2 → 0.9 here).')),
        UI.section('Box loss (1 − CIoU), per head', el('div', { class: 'chart-box' }, c3), el('div', { class: 'mini-legend' }, key('var(--s4)', 'one-to-many'), key('var(--s7)', 'one-to-one'))),
        UI.section('Small shapes found (under 12 px)', el('div', { class: 'chart-box' }, c4), UI.note('Recall of shapes under 12 pixels at confidence 0.25, one-to-one head. STAL widens small boxes to 16 pixels when choosing which cells learn them, so they are never left without positives.')),
        UI.section('The recipe', el('div', { class: 'kv-grid' }, ...[
          ['Optimiser', 'MuSGD: each weight matrix steps half by SGD (momentum 0.937, Nesterov) and half by Muon, its momentum orthogonalised by 5 Newton–Schulz iterations; biases and norms by SGD'],
          ['Learning rate', '0.01, one epoch of warm-up, then linear decay to 1%'],
          ['Assignment', 'task-aligned: score^0.5 × IoU^6; top 10 cells per object (one-to-many), top 1 (one-to-one); STAL widens boxes under 16 px'],
          ['Loss', '7.5 × (1 − CIoU) + 0.5 × binary cross-entropy, per head; ProgLoss weights 0.8 → 0.1 (one-to-many)'],
          ['Data', 'fresh synthetic scenes every step, never the same picture twice'],
        ].map(([k, v]) => el('div', { class: 'kv' }, el('span', { class: 'kv-k', text: k }), el('span', { class: 'kv-v', text: v }))))));
    } else if (id === 'heads') {
      P.table = el('div', { class: 'yl-table' });
      P.scene = el('div', { class: 'diag-list' });
      body.append(
        UI.section('On 500 validation scenes', P.table, UI.note('Boxes per shape: how many boxes land on each shape that is found (IoU ≥ 0.5, confidence ≥ 0.25). 1.00 means no duplicates at all.')),
        UI.section('On this scene', P.scene),
        UI.section('Why one-to-one needs no NMS', UI.note('Training decides which cells are positives for each object. With ten positives (one-to-many), the network learns that several cells around an object should all fire; that helps learning, but leaves duplicates for NMS. With one positive (one-to-one), the neighbours are explicitly trained to stay quiet, so they learn to defer to the best cell: duplicate suppression is learned inside the network instead of bolted on after it.'), UI.note('The idea comes from YOLOv10 (dual assignments). YOLO26 makes the one-to-one head the default output, removes DFL so the boxes are four plain numbers, and uses ProgLoss to move the training’s attention to that head over time.')));
    } else if (id === 'net') {
      P.kv = el('div', { class: 'kv-grid' });
      P.layers = el('div', { class: 'yl-table yl-layers' });
      body.append(UI.section('This model and YOLO26n', P.kv), UI.section('Layers (as in yolo26.yaml)', P.layers, UI.note('From: the layer each input comes from (−1 is the previous one). Shapes are channels × height × width for this scene. Open the Architecture view to see inside any of them.')));
    }
    this.updateTab();
  };
  YL.updateTab = function () {
    const id = this.tabId, P = this.parts, W = NA.yoloWeights;
    if (!P) return;
    this.W = W;
    if (id === 'overview') {
      clear(P.findings);
      if (!this.dets) { P.findings.append(UI.note(this.loadError ? 'The detector’s weights could not be loaded.' : 'Running the detector…')); return; }
      const f = (level, title, text) => P.findings.append(el('div', { class: 'diag diag-' + level }, el('div', { class: 'diag-head' }, UI.pill(level, { good: 'Good', info: 'Note', warn: 'Watch' }[level]), el('h4', { class: 'diag-title', text: title })), el('p', { class: 'diag-body', text })));
      const M = this.match[this.head], n = this.gts.length, d = this.dets;
      f(M.found === n ? 'good' : 'warn', `${HEADS[this.head]} finds ${M.found} of ${n} shapes`, M.found === n ? 'Every shape has a box of the right kind overlapping it by IoU 0.5 or more.' : `Missed: ${M.missed.map((j) => `a ${fmt(this.gts[j].size, 0)} px ${NAMES[this.gts[j].cls]}`).join(', ')}. Small, faint or heavily overlapped shapes are the hard ones; lowering the confidence threshold may bring them back, with more false alarms.`);
      f('info', `${d.raw.length} raw boxes, ${d.o2m.length} after NMS, ${d.o2o.length} one-to-one`, `NMS removed ${(d.o2m.suppressed || []).length} duplicates in ${fmt(d.nmsMs, 2)} ms; the one-to-one head needed no clean-up at all.`);
      f('info', `Network: ${this.runner && this.runner.ms ? fmt(this.runner.ms, 0) : '—'} ms per picture`, `${fmtInt(this.net.params)} parameters and about 27 million multiply-adds per picture, run ${this.runner && this.runner.worker ? 'in a background thread of your browser (a Web Worker), so dragging stays smooth' : 'on the page itself'}.`);
    } else if (id === 'train') {
      if (!W) return;
      clear(P.tiles).append(UI.tile('Epochs', String(W.train.epochs), `${W.train.iters} steps each`), UI.tile('Scenes seen', fmtInt(W.train.images), 'each new'), UI.tile('Training time', `${Math.round(W.train.seconds / 60)} min`, 'one laptop GPU'), UI.tile('Parameters', fmtInt(W.params)));
      const c = W.curve, pts = (f) => c.map((r) => [r.epoch, f(r)]);
      [P.map, P.prog, P.loss, P.small].forEach((ch) => ch.frame());
      P.map.set({ series: [{ name: 'one-to-one, mAP@0.5', color: theme.c('s7'), points: pts((r) => r.o2o.map50) }, { name: 'one-to-many + NMS, mAP@0.5', color: theme.c('s4'), points: pts((r) => r.o2m.map50) }, { name: 'one-to-one, mAP@0.5:0.95', color: theme.c('s1'), points: pts((r) => r.o2o.map) }] });
      P.prog.set({ series: [{ name: 'one-to-many weight', color: theme.c('s4'), points: pts((r) => r.w_o2m) }, { name: 'one-to-one weight', color: theme.c('s7'), points: pts((r) => 1 - r.w_o2m) }], yFixed: [0, 1] });
      P.loss.set({ series: [{ name: 'one-to-many', color: theme.c('s4'), points: pts((r) => r.o2m_box) }, { name: 'one-to-one', color: theme.c('s7'), points: pts((r) => r.o2o_box) }] });
      P.small.set({ series: [{ name: 'small shapes found', color: theme.c('s7'), points: pts((r) => r.o2o.small_recall) }] });
    } else if (id === 'heads') {
      if (!W) return;
      const F = W.final, rows = [['', 'mAP@0.5', 'mAP@0.5:0.95', 'Found', 'Boxes per shape'],
        ['One-to-one, no NMS', fmtPct(F.o2o.map50, 1), fmtPct(F.o2o.map, 1), fmtPct(F.o2o.recall, 1), fmt(F.o2o.boxes_per_obj, 2)],
        ['One-to-many + NMS', fmtPct(F.o2m.map50, 1), fmtPct(F.o2m.map, 1), fmtPct(F.o2m.recall, 1), fmt(F.o2m.boxes_per_obj, 2)],
        ['One-to-many, raw', '—', '—', fmtPct(F.raw.recall, 1), fmt(F.raw.boxes_per_obj, 2)]];
      clear(P.table).append(...rows.map((r, i) => el('div', { class: 'yl-tr' + (i ? '' : ' yl-th') }, ...r.map((v) => el('span', { text: v })))));
      clear(P.scene);
      if (this.dets) {
        const d = this.dets, m = this.match;
        P.scene.append(...[
          ['One-to-one', `${d.o2o.length} boxes, ${m.o2o.found} of ${this.gts.length} shapes found, ${m.o2o.dups} duplicates. Post-processing: a sort, ${fmt(d.topMs, 2)} ms.`],
          ['One-to-many + NMS', `${d.raw.length} boxes before NMS, ${d.o2m.length} after; ${m.o2m.found} found, ${m.o2m.dups} duplicates left. NMS: ${fmt(d.nmsMs, 2)} ms.`],
        ].map(([t, x]) => el('div', { class: 'diag diag-info' }, el('div', { class: 'diag-head' }, el('h4', { class: 'diag-title', text: t })), el('p', { class: 'diag-body', text: x }))));
      }
    } else if (id === 'net') {
      if (!this.net) return;
      clear(P.kv).append(...[
        ['Layout', 'yolo26.yaml: 11 backbone layers, 12 neck layers, a Detect head at P3, P4 and P5'],
        ['Scale', 'depth 0.50, width 0.125, max channels 512 (YOLO26n: 0.50, 0.25, 1024)'],
        ['Parameters', `${fmtInt(this.net.params)} (YOLO26n: 2.4 million)`],
        ['Input', '128 × 128 colour (YOLO26: 640 × 640)'],
        ['Classes', '4 shapes (YOLO26 on COCO: 80 kinds of object)'],
        ['Head', 'boxes as 4 distances (no DFL), sigmoid class scores; one-to-many and one-to-one copies'],
      ].map(([k, v]) => el('div', { class: 'kv' }, el('span', { class: 'kv-k', text: k }), el('span', { class: 'kv-v', text: v }))));
      const K = this.result && this.result.F.K, sh = (n) => (K && K[n] ? `${K[n].c} × ${K[n].h} × ${K[n].w}` : '—');
      const prm = (pre) => Object.values(this.net.P).filter((p) => p.name === pre || p.name.startsWith(pre + '.')).reduce((s, p) => s + p.w.length + p.b.length, 0);
      const L = [['0', 'b0', '−1', 'Conv 3×3/2'], ['1', 'b1', '−1', 'Conv 3×3/2'], ['2', 'b2', '−1', 'C3k2'], ['3', 'b3', '−1', 'Conv 3×3/2'], ['4', 'b4', '−1', 'C3k2'], ['5', 'b5', '−1', 'Conv 3×3/2'], ['6', 'b6', '−1', 'C3k2 (C3k)'], ['7', 'b7', '−1', 'Conv 3×3/2'], ['8', 'b8', '−1', 'C3k2 (C3k)'], ['9', 'b9', '−1', 'SPPF'], ['10', 'b10', '−1', 'C2PSA'],
        ['11', 'u11', '−1', 'Upsample ×2'], ['12', null, '−1, 6', 'Concat'], ['13', 'h13', '−1', 'C3k2 (C3k)'], ['14', 'u14', '−1', 'Upsample ×2'], ['15', null, '−1, 4', 'Concat'], ['16', 'h16', '−1', 'C3k2 (C3k) → P3'], ['17', 'h17', '−1', 'Conv 3×3/2'], ['18', null, '−1, 13', 'Concat'], ['19', 'h19', '−1', 'C3k2 (C3k) → P4'], ['20', 'h20', '−1', 'Conv 3×3/2'], ['21', null, '−1, 10', 'Concat'], ['22', 'h22', '−1', 'C3k2 + attention → P5'], ['23', 'detect', '16, 19, 22', 'Detect (2 heads)']];
      clear(P.layers).append(el('div', { class: 'yl-tr yl-th' }, ...['#', 'Module', 'From', 'Output', 'Params'].map((t) => el('span', { text: t }))),
        ...L.map(([i, n, from, mod]) => el('div', { class: 'yl-tr' }, el('span', { text: i }), el('span', { text: mod }), el('span', { text: from }), el('span', { text: n === 'detect' ? '3 scales × 8' : n ? sh(n) : '' }), el('span', { text: n ? fmtInt(prm(n)) : '0' }))));
    }
  };

  YL.runTry = function (t) {
    const go = () => { if (NA.mobile && NA.mobile.mobile) NA.mobile.go('network'); };
    const fresh = (S) => { this.scene = S; this.sel = -1; this.savedLines = null; this.clutterChip.setAttribute('aria-pressed', 'true'); this.noiseSl.set(S.sigma); };
    if (t.action === 'nms') {
      fresh(DEFAULT()); this.setIou(0.7); this.setView('detect'); this.setHead('o2m'); this.show.raw = true; NA.$$('.chip', this.chips)[1].setAttribute('aria-pressed', 'true'); this.changed();
      NA.ui.toast.show('Faint boxes: everything the one-to-many head proposed. NMS kept the solid ones.');
    } else if (t.action === 'overlap') {
      // the most overlap any pair had in training (true IoU 0.35); a strict NMS threshold merges them
      const S = DEFAULT();
      S.objs = [{ cls: 2, cx: 50, cy: 60, size: 36, ang: 0.1, col: Y.hsv(0.55, 0.75, 0.9) }, { cls: 2, cx: 62, cy: 64.8, size: 36, ang: 0.5, col: Y.hsv(0.12, 0.85, 0.98) }, { cls: 0, cx: 100, cy: 24, size: 20, ang: 0, col: Y.hsv(0.95, 0.65, 0.92) }];
      fresh(S); this.setIou(0.2); this.setView('compare'); this.changed();
      NA.ui.toast.show('NMS threshold 0.2: compare the middle picture (one square deleted) with the right one.');
    } else if (t.action === 'small') {
      const S = DEFAULT(), rng = new NA.RNG(5);
      S.objs = [];
      for (let k = 0; k < 7; k++) { const o = Y.randomShape(rng, S, k % 4); if (!o) continue; o.size = 7 + rng.next() * 3; if (S.objs.every((p) => Y.iou(Y.boxOf(o), Y.boxOf(p)) < 0.05)) S.objs.push(o); }
      fresh(S); this.setIou(0.7); this.setView('scales'); this.changed();
      NA.ui.toast.show('Shapes 7 to 10 pixels across: watch which grid finds them.');
    } else if (t.action === 'noise') {
      const S = DEFAULT();
      S.sigma = 0.15;
      fresh(S); this.setIou(0.7); this.setView('detect'); this.changed();
      NA.ui.toast.show('Noise σ = 0.15, three times the most the detector saw in training.');
    }
    go();
    this.dirty = true; this.inspDirty = true;
  };

  // Zoo glyph: a few shapes, each in its box.
  YL.glyph = function (ctx, W, Hh) {
    const items = [[0.24, 0.42, 0.3, 3], [0.62, 0.36, 0.24, 0], [0.5, 0.74, 0.2, 2], [0.84, 0.7, 0.14, 1]];
    for (const [fx, fy, fs, k] of items) {
      const s = fs * Hh * 1.4, x = fx * W, y = fy * Hh, col = theme.series[k];
      ctx.fillStyle = col; ctx.globalAlpha = 0.85; ctx.beginPath();
      if (k === 0) ctx.arc(x, y, s / 2, 0, Math.PI * 2);
      else Y.vertices({ cls: k, cx: x, cy: y, size: s, ang: 0.2 }).slice(0, k === 1 ? 3 : k === 2 ? 4 : 10).forEach(([px, py], i) => (i ? ctx.lineTo(px, py) : ctx.moveTo(px, py)));
      ctx.fill(); ctx.globalAlpha = 1;
      ctx.strokeStyle = col; ctx.lineWidth = 1.2; ctx.strokeRect(x - s / 2 - 2.5, y - s / 2 - 2.5, s + 5, s + 5);
    }
  };

  H.register(YL);
  NA.yoloLab = YL;
})(window.NA);
