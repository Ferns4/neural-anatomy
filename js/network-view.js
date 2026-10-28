/* network-view.js — the network diagram: neurons, connections and every lens drawn on them. */
(function (NA) {
  'use strict';
  const { theme, clamp, fmtSci: fmt, el } = NA; // fmtSci: tiny values as 4.1×10⁻⁴, not 4.1e−4
  const App = NA.app;

  const V = { dirty: true, layout: null, thumbs: new Map(), thumbsKey: '', thumbsTime: 0, lastDraw: 0, flowEdges: [], hoverEdge: null, hoverPixel: null };
  const BUCKETS = 12;

  function rr(ctx, x, y, w, h, r) {
    r = Math.max(0, Math.min(r || 0, Math.abs(w) / 2, Math.abs(h) / 2)); // a box squeezed to nothing must not throw
    ctx.beginPath();
    if (ctx.roundRect) ctx.roundRect(x, y, w, h, r);
    else { ctx.moveTo(x + r, y); ctx.arcTo(x + w, y, x + w, y + h, r); ctx.arcTo(x + w, y + h, x, y + h, r); ctx.arcTo(x, y + h, x, y, r); ctx.arcTo(x, y, x + w, y, r); ctx.closePath(); }
  }
  NA.rr = rr;

  V.init = function () {
    V.stage = document.getElementById('netStage');
    V.canvas = document.getElementById('netCanvas');
    V.overlay = document.getElementById('netOverlay');
    V.legend = document.getElementById('netLegend');
    if (NA.a11y) NA.a11y.describe(V.canvas, () => {
      const net = App.net;
      if (!net || App.mode !== 'mlp') return '';
      const hidden = net.sizes.slice(1, -1), acts = NA.range(hidden.length).map((h) => { const id = net.layerActId(h + 1); return id === 'mixed' ? 'mixed' : NA.engine.ACTIVATIONS[id].label; });
      const lens = { weight: 'weights', signal: 'the signal for the probe input', gradient: 'gradients' }[App.lenses.edges];
      return `Network diagram: ${net.sizes[0]} input${net.sizes[0] === 1 ? '' : 's'}, ${hidden.length ? `hidden layers of ${hidden.map((n, i) => `${n} ${acts[i]}`).join(', ')} neurons` : 'no hidden layer'}, and ${net.sizes[net.L]} output${net.sizes[net.L] === 1 ? '' : 's'}; connections coloured by ${lens}.${App.selected ? ` Selected: ${App.neuronName(App.selected.l, App.selected.j)}.` : ''} Step ${NA.fmtInt(App.displayStep())}.`;
    });
    new ResizeObserver(() => V.resize()).observe(V.stage);
    // zooming far into a neuron opens it (net-deep.js). A touch screen gets more room to zoom before that
    // happens: a pinch to read the labels should not fall into a neuron.
    V.touch = !!(window.matchMedia && window.matchMedia('(pointer: coarse)').matches);
    V.zoom = new NA.ui.Zoom(V.canvas, { host: V.stage, panOne: true, wheel: 'fixed', max: V.touch ? 5 : 4, onUnderflow: () => NA.scaleView && NA.scaleView.underflow(), onDouble: (px, py) => !!(NA.netDeep && NA.netDeep.openAt(px, py)), onChange: () => { V.dirty = true; positionHeaders(); NA.tooltip.hide(); if (NA.netDeep) NA.netDeep.maybeOpen(); } });
    V.canvas.addEventListener('pointermove', onMove);
    V.canvas.addEventListener('pointerleave', onLeave);
    V.canvas.addEventListener('click', onClick);
    const mark = () => { V.dirty = true; };
    ['weights', 'probe', 'selection', 'hover', 'walk', 'highlight', 'ablation', 'overlay'].forEach((t) => App.on(t, mark));
    App.on('model', () => { V.layout = null; V.thumbsKey = ''; V.dirty = true; V.zoom.reset(); buildHeaders(); renderLegend(); });
    // a layer that starts mixing functions labels its neurons, which needs room under the last one
    App.on('arch', () => { V.layout = null; V.dirty = true; buildHeaders(); renderLegend(); });
    App.on('hyper', () => { if (V.lossKey !== App.net.lossId) { V.lossKey = App.net.lossId; buildHeaders(); } });
    App.on('lens', () => { V.layout = null; V.thumbsKey = ''; V.dirty = true; renderLegend(); });
    App.on('walk', renderLegend);
    theme.onChange(() => { V.thumbsKey = ''; V.dirty = true; });
    App.register(V);
  };

  V.resize = function () {
    const r = V.stage.getBoundingClientRect();
    if (r.width < 10 || r.height < 10) return;
    V.W = r.width; V.H = r.height;
    V.ctx = NA.fitCanvas(V.canvas, V.W, V.H);
    V.layout = null; V.dirty = true;
  };

  V.frame = function (now) {
    if (!App.net || !V.ctx || !V.stage.offsetParent) return; // skip while its screen is hidden
    const animating = (App.lenses.flow && !NA.prefersReducedMotion()) || (App.highlight && now < App.highlight.until);
    if (!V.dirty && !animating) return;
    if (App.training && now - V.lastDraw < (App.isImage() ? 60 : 32)) return;
    draw(now);
  };

  // ------------------------------------------------------------------ layout
  // Where the layer headers end, in stage pixels, for a given column spacing. Measured, because they grow on a
  // phone and when narrow columns stack their controls; measured as they look unzoomed (zoomed in, they shrink
  // to their titles).
  function headerBottom(gap) {
    if (!V.headers || !V.headers.length) return 70;
    const zoomed = V.canvas.classList.contains('is-zoomed'), hidden = V.headers.map((h) => h.hidden);
    if (zoomed) V.canvas.classList.remove('is-zoomed');
    for (const h of V.headers) h.hidden = false; // headers panned off screen still count
    setHeaderMode(gap);
    let b = 0;
    for (const h of V.headers) b = Math.max(b, h.offsetTop + h.offsetHeight);
    V.headers.forEach((h, i) => { h.hidden = hidden[i]; });
    if (zoomed) V.canvas.classList.add('is-zoomed');
    return b;
  }
  // Height kept clear along the bottom for the depth and zoom controls, also measured.
  function controlsHeight() {
    let h = 46;
    for (const c of V.stage.querySelectorAll('.depth-ctl, .zoom-chip')) if (c.offsetParent) h = Math.max(h, V.H - c.offsetTop + 6);
    return h;
  }
  // Widest input label (the feature's name, or its value with the Values overlay), drawn to the left of its neuron.
  function inputLabelWidth() {
    const ds = App.ds, ctx = V.ctx;
    if (ds.kind === 'image' || !ctx) return 0;
    ctx.save();
    ctx.font = `600 13px ${getFont('ui')}`;
    let w = 0;
    for (const t of ds.featureLabels) w = Math.max(w, ctx.measureText(t).width);
    if (App.lenses.values) { ctx.font = `500 10px ${getFont('mono')}`; w = Math.max(w, ctx.measureText('−0.00').width); }
    ctx.restore();
    return w;
  }

  function computeLayout() {
    const net = App.net, ds = App.ds, S = net.sizes, L = net.L, W = V.W, H = V.H;
    const img = ds.kind === 'image';
    const narrow = W < 520;
    const imgSideGuess = img ? clamp(Math.min((H - 70) * 0.62, W * 0.2), 84, 176) : 0;
    const rightPad = narrow ? (ds.task === 'multiclass' && !img ? 66 : 58) : ds.task === 'regression' ? 84 : 124;
    const x1 = W - rightPad, nCols = L + 1;
    let maxN = 1;
    for (let l = img ? 1 : 0; l <= L; l++) maxN = Math.max(maxN, S[l]);
    const maxSize = App.lenses.maps ? 54 : 26;
    // labels under the neurons (functions, values, health) hang below the last one in each column
    let mixed = false;
    for (let l = 1; l < L; l++) if (net.layerActId(l) === 'mixed') mixed = true;
    const labelsBelow = App.lenses.acts || App.lenses.values || App.lenses.health || mixed;
    const bottom = controlsHeight() + (labelsBelow ? 14 : 0);
    // The input labels sit right-aligned to the left of their neurons, so the left margin has to hold the widest
    // one beside a neuron of the final size; the size in turn depends on the spacing the margin leaves. A wider
    // margin only ever shrinks the neurons, so this settles within a pass or two.
    const base = img ? 14 + imgSideGuess / 2 : narrow ? 40 : 58, labelW = inputLabelWidth();
    let leftPad = base, x0, colGap, top, avail, size;
    for (let pass = 0; ; pass++) {
      x0 = leftPad;
      colGap = nCols > 1 ? (x1 - x0) / (nCols - 1) : 200;
      top = headerBottom(colGap) + 12;
      avail = Math.max(60, H - top - bottom);
      size = Math.max(9, Math.min(maxSize, (avail / maxN) * (App.lenses.maps ? 0.86 : 0.7), colGap * 0.42));
      const need = img ? base : Math.max(base, size / 2 + 9 + labelW + 6);
      if (need <= leftPad + 0.5 || pass === 3) break;
      leftPad = need;
    }
    const imgSide = imgSideGuess;
    const xs = [];
    for (let l = 0; l <= L; l++) xs.push(nCols === 1 ? (x0 + x1) / 2 : x0 + ((x1 - x0) * l) / (nCols - 1));
    const spacingCap = size * 1.4 + 10;
    const cols = [], nodes = [];
    for (let l = 0; l <= L; l++) {
      const n = S[l], col = { l, x: xs[l], nodes: [] };
      if (l === 0 && img) {
        const G = ds.imgSize, cell = imgSide / G, left = xs[0] - imgSide / 2, ty = top + (avail - imgSide) / 2;
        col.image = { x: left, y: ty, side: imgSide, cell, G };
        for (let j = 0; j < n; j++) col.nodes.push({ l, j, x: left + ((j % G) + 0.5) * cell, y: ty + (Math.floor(j / G) + 0.5) * cell, s: cell, pixel: true });
      } else {
        const sp = Math.min(avail / n, spacingCap);
        for (let j = 0; j < n; j++) {
          const nd = { l, j, x: xs[l], y: top + avail / 2 + (j - (n - 1) / 2) * sp, s: size, gap: n > 1 ? sp - size : 99 };
          col.nodes.push(nd); nodes.push(nd);
        }
      }
      cols.push(col);
    }
    // connections with sampled points for hit-testing
    const edges = [];
    for (let l = 0; l < L; l++) {
      const A = cols[l].nodes, B = cols[l + 1].nodes, nIn = S[l];
      for (let j = 0; j < B.length; j++) {
        for (let i = 0; i < A.length; i++) {
          const a = A[i], b = B[j];
          const x1e = a.pixel ? a.x : a.x + a.s / 2, x2e = b.x - b.s / 2;
          edges.push({ l, i, j, w: j * nIn + i, x1: x1e, y1: a.y, x2: x2e, y2: b.y, dense: A.length * B.length > 1200 });
        }
      }
    }
    return { cols, nodes, edges, size, top, avail };
  }

  function bez(e, t) {
    const dx = (e.x2 - e.x1) * 0.5, u = 1 - t;
    const x = u * u * u * e.x1 + 3 * u * u * t * (e.x1 + dx) + 3 * u * t * t * (e.x2 - dx) + t * t * t * e.x2;
    const y = u * u * u * e.y1 + 3 * u * u * t * e.y1 + 3 * u * t * t * e.y2 + t * t * t * e.y2;
    return [x, y];
  }
  function pathEdge(p, e) {
    const dx = (e.x2 - e.x1) * 0.5;
    p.moveTo(e.x1, e.y1);
    p.bezierCurveTo(e.x1 + dx, e.y1, e.x2 - dx, e.y2, e.x2, e.y2);
  }

  /* Dropout: the hidden neurons the latest training step switched off for its last example. Shown while paused
     or training at Slow speed (faster, it would flicker), and never for history or in the walkthrough. */
  function dropShown(walk) {
    const net = App.net;
    if (walk || !(net.lastDrop > 0) || !(App.hyper.dropout > 0) || App.viewing !== null) return null;
    if (App.training && (App.speed !== 'slow' || App.ffTarget !== null || NA.prefersReducedMotion())) return null;
    return net.dropMask;
  }

  // ------------------------------------------------------------------ lens data
  function gather() {
    const net = App.net, L = net.L;
    const walk = App.walk && App.walk.active ? App.walk : null;
    const trace = walk ? walk.trace : App.probeTrace();
    const lens = App.lenses;
    const needGrad = !walk && (lens.edges === 'gradient' || lens.nodes === 'gradient');
    const st = { net, L, walk, trace, lens, grad: needGrad ? App.gradReport() : null, scale: App.layerScale(), stats: lens.health ? App.stats() : null, drop: dropShown(walk) };
    st.focus = App.hover || App.selected;
    // per weight-layer edge values: signed in [-1,1] ('div') or magnitude in [0,1] ('grad')
    st.edgeMode = [];
    st.edgeVals = [];
    let walkGMax = 1e-12;
    if (walk && walk.trace.delta) {
      for (let l = 0; l < L; l++) for (let j = 0; j < net.sizes[l + 1]; j++) for (let i = 0; i < net.sizes[l]; i++) walkGMax = Math.max(walkGMax, Math.abs(walk.trace.delta[l + 1][j] * walk.trace.a[l][i]));
    }
    for (let l = 0; l < L; l++) {
      const W = net.W[l], nIn = net.sizes[l], nOut = net.sizes[l + 1], vals = new Float32Array(W.length);
      let mode = lens.edges;
      if (walk) mode = walk.edgeMode(l);
      if (mode === 'weight') {
        let m = 1e-12; for (let i = 0; i < W.length; i++) m = Math.max(m, Math.abs(W[i]));
        for (let i = 0; i < W.length; i++) vals[i] = W[i] / m;
        st.edgeMode.push('div');
      } else if (mode === 'signal') {
        const a = trace.a[l];
        let m = 1e-12;
        for (let j = 0; j < nOut; j++) for (let i = 0; i < nIn; i++) { const v = W[j * nIn + i] * a[i]; vals[j * nIn + i] = v; m = Math.max(m, Math.abs(v)); }
        for (let i = 0; i < vals.length; i++) vals[i] /= m;
        st.edgeMode.push('div');
      } else if (mode === 'gradient') {
        const G = st.grad.gW[l], gm = st.grad.gMax || 1e-12;
        for (let i = 0; i < G.length; i++) { const g = Math.abs(G[i]); vals[i] = g > 0 ? clamp(1 + Math.log10(g / gm) / 4, 0, 1) : 0; }
        st.edgeMode.push('grad');
      } else if (mode === 'walkgrad') {
        const d = walk.trace.delta[l + 1], a = walk.trace.a[l];
        for (let j = 0; j < nOut; j++) for (let i = 0; i < nIn; i++) { const g = Math.abs(d[j] * a[i]); vals[j * nIn + i] = g > 0 ? clamp(1 + Math.log10(g / walkGMax) / 3.5, 0, 1) : 0; }
        st.edgeMode.push('grad');
      } else { st.edgeMode.push('off'); }
      st.edgeVals.push(vals);
    }
    return st;
  }

  // Colour for a neuron: {kind:'div', t} | {kind:'grad', t} | {kind:'class', k, p} | null (pending)
  function nodeValue(st, l, j) {
    const net = st.net, tr = st.trace, L = st.L, ds = App.ds;
    const mode = st.walk ? st.walk.nodeMode(l) : st.lens.nodes;
    if (mode === 'pending') return null;
    if (mode === 'gradient' || mode === 'walkdelta') {
      if (mode === 'walkdelta') { const d = tr.delta[l][j], m = st.walk.deltaMax[l] || 1e-12; return { kind: 'divd', t: d / m }; }
      const v = st.grad.dRms[l][j], m = st.grad.dMax || 1e-12;
      return { kind: 'grad', t: v > 0 ? clamp(1 + Math.log10(v / m) / 4, 0, 1) : 0 };
    }
    if (l === L) {
      const a = tr.a[L][j];
      if (ds.task === 'binary') return { kind: 'div', t: 2 * a - 1 };
      if (ds.task === 'multiclass') return { kind: 'class', k: j, p: a };
      return { kind: 'div', t: a / (st.scale[L].a || 1) };
    }
    if (mode === 'preact' && l > 0) return { kind: 'div', t: tr.z[l][j] / (st.scale[l].z || 1) };
    return { kind: 'div', t: tr.a[l][j] / (st.scale[l].a || 1) };
  }
  function valueColor(v, alpha = 1) {
    if (!v) return theme.c('panel-2', alpha);
    if (v.kind === 'grad') return theme.gradColor(v.t, alpha);
    if (v.kind === 'class') {
      const c = NA.mixLab(theme.tokens['panel-2'], theme.classes[v.k] || theme.tokens.muted, clamp(v.p, 0, 1));
      return NA.css(c, alpha);
    }
    return theme.divColor(v.t, alpha);
  }

  // ------------------------------------------------------------------ thumbnails ("maps")
  function ensureThumbs(now) {
    const key = App.version + '|' + App.probeVersion + '|' + App.lenses.maps;
    if (V.thumbsKey === key) return;
    if (App.training && now - V.thumbsTime < (App.isImage() ? 400 : 90) && V.thumbsKey) return;
    V.thumbsKey = key; V.thumbsTime = now;
    if (!App.lenses.maps) return;
    const ds = App.ds, net = App.net;
    if (ds.kind === 'plane') planeThumbs(net);
    else if (ds.kind === 'image') imageThumbs(net, ds);
  }
  function thumbCanvas(key, w, h) {
    let c = V.thumbs.get(key);
    if (!c) { c = document.createElement('canvas'); V.thumbs.set(key, c); }
    if (c.width !== w || c.height !== h) { c.width = w; c.height = h; }
    return c;
  }
  function planeThumbs(net) {
    const g = App.grid(), G = g.G, n = g.n, ds = App.ds, L = net.L, lut = theme.div;
    for (let l = 0; l <= L; l++) {
      const A = g.F.a[l];
      for (let j = 0; j < net.sizes[l]; j++) {
        const c = thumbCanvas(l + ':' + j, G, G), cx = c.getContext('2d'), im = cx.createImageData(G, G), d = im.data;
        const off = j * n;
        if (l === L && ds.task === 'multiclass') {
          const base = theme.tokens['panel-2'], cc = theme.classes[j] || theme.tokens.muted;
          for (let p = 0; p < n; p++) { const t = clamp(A[off + p], 0, 1); d[4 * p] = base[0] + (cc[0] - base[0]) * t; d[4 * p + 1] = base[1] + (cc[1] - base[1]) * t; d[4 * p + 2] = base[2] + (cc[2] - base[2]) * t; d[4 * p + 3] = 255; }
        } else {
          let m = 1e-9;
          if (l === L) m = 1;
          else for (let p = 0; p < n; p++) m = Math.max(m, Math.abs(A[off + p]));
          for (let p = 0; p < n; p++) {
            const t = l === L ? 2 * A[off + p] - 1 : A[off + p] / m;
            const k = Math.round((clamp(t, -1, 1) + 1) * 128) * 3;
            d[4 * p] = lut[k]; d[4 * p + 1] = lut[k + 1]; d[4 * p + 2] = lut[k + 2]; d[4 * p + 3] = 255;
          }
        }
        cx.putImageData(im, 0, 0);
      }
    }
  }
  function imageThumbs(net, ds) {
    const G = ds.imgSize, lut = theme.div, L = net.L;
    // first hidden layer: weight templates
    if (L >= 1) {
      const W = net.W[0], nIn = net.sizes[0];
      for (let j = 0; j < net.sizes[1]; j++) {
        const c = thumbCanvas('1:' + j, G, G), cx = c.getContext('2d'), im = cx.createImageData(G, G), d = im.data;
        let m = 1e-9; for (let i = 0; i < nIn; i++) m = Math.max(m, Math.abs(W[j * nIn + i]));
        for (let i = 0; i < nIn; i++) {
          const k = Math.round((clamp(W[j * nIn + i] / m, -1, 1) + 1) * 128) * 3;
          d[4 * i] = lut[k]; d[4 * i + 1] = lut[k + 1]; d[4 * i + 2] = lut[k + 2]; d[4 * i + 3] = 255;
        }
        cx.putImageData(im, 0, 0);
      }
    }
    // deeper layers: average of the validation images that activate the neuron most
    const F = App.dataPass(), N = F.n, idx = ds.valIdx, ink = theme.tokens.ink, base = theme.tokens['panel-2'];
    for (let l = 2; l <= L; l++) {
      const A = F.a[l];
      for (let j = 0; j < net.sizes[l]; j++) {
        const top = Array.from(idx).sort((p, q) => A[j * N + q] - A[j * N + p]).slice(0, 12);
        const avg = new Float64Array(G * G);
        for (const p of top) for (let i = 0; i < G * G; i++) avg[i] += ds.raw[p * G * G + i] / top.length;
        const c = thumbCanvas(l + ':' + j, G, G), cx = c.getContext('2d'), im = cx.createImageData(G, G), d = im.data;
        for (let i = 0; i < G * G; i++) {
          const t = clamp(avg[i], 0, 1);
          d[4 * i] = base[0] + (ink[0] - base[0]) * t; d[4 * i + 1] = base[1] + (ink[1] - base[1]) * t; d[4 * i + 2] = base[2] + (ink[2] - base[2]) * t; d[4 * i + 3] = 255;
        }
        cx.putImageData(im, 0, 0);
      }
    }
  }

  // ------------------------------------------------------------------ drawing
  function draw(now) {
    const ctx = V.ctx;
    if (!V.layout) { V.layout = computeLayout(); positionHeaders(); }
    const lay = V.layout;
    ensureThumbs(now);
    const st = gather();
    if (!!st.drop !== !!V.dropLegend) { V.dropLegend = !!st.drop; renderLegend(); }
    ctx.clearRect(0, 0, V.W, V.H);
    ctx.save();
    V.zoom.apply(ctx, V.canvas._dpr || 1); // everything below is drawn in unzoomed layout coordinates
    drawBands(ctx, lay, st);
    drawEdges(ctx, lay, st, now);
    V.labels = [];
    for (const col of lay.cols) {
      if (col.image) drawInputImage(ctx, col, st);
      else col.nodes.forEach((nd) => drawNode(ctx, nd, st, now));
    }
    drawNodeLabels(ctx, lay);
    drawLabels(ctx, lay, st);
    ctx.restore();
    V.dirty = false; V.lastDraw = now;
  }

  // Which neurons get an activation label: all of them with the Functions overlay, otherwise only
  // those in layers that mix several activation functions.
  function showsAct(l) {
    const net = App.net;
    if (l <= 0 || l >= net.L) return false;
    return !!App.lenses.acts || net.layerActId(l) === 'mixed';
  }

  function drawBands(ctx, lay, st) {
    const fl = st.walk ? st.walk.focusLayer : null;
    if (fl === null || fl === undefined) return;
    const col = lay.cols[fl];
    if (!col) return;
    let y0 = Infinity, y1 = -Infinity, half = lay.size / 2 + 12;
    if (col.image) { y0 = col.image.y; y1 = col.image.y + col.image.side; half = col.image.side / 2 + 10; }
    else for (const nd of col.nodes) { y0 = Math.min(y0, nd.y - nd.s / 2); y1 = Math.max(y1, nd.y + nd.s / 2); }
    ctx.fillStyle = theme.c('ink', theme.dark ? 0.07 : 0.045);
    rr(ctx, col.x - half, y0 - 14, half * 2, y1 - y0 + 28, 10);
    ctx.fill();
  }

  function drawEdges(ctx, lay, st, now) {
    // hovering isolates a neuron's wiring; a pinned selection only lifts its own connections
    const focus = st.focus, dimAll = !!App.hover && !st.walk;
    const buckets = new Map();
    const strong = [];
    const net = st.net, drop = st.drop;
    let dropPath = null; // wiring of dropped neurons: it carried nothing in the last training example
    for (const e of lay.edges) {
      const mode = st.edgeMode[e.l];
      if (mode === 'off') continue;
      if (drop && ((e.l > 0 && !drop[e.l][e.i]) || (e.l + 1 < net.L && !drop[e.l + 1][e.j]))) {
        if (!e.dense) pathEdge(dropPath || (dropPath = new Path2D()), e);
        continue;
      }
      const v = st.edgeVals[e.l][e.w];
      const mag = Math.abs(v);
      const level = Math.min(BUCKETS - 1, Math.floor(mag * BUCKETS));
      if (e.dense && level < 1) continue;
      const isFocus = focus && !st.walk && ((focus.l === e.l + 1 && focus.j === e.j) || (focus.l === e.l && focus.j === e.i));
      if (isFocus) { strong.push(e); continue; }
      const key = mode + (v < 0 ? '-' : '+') + level + (e.dense ? 'd' : '');
      let b = buckets.get(key);
      if (!b) { b = { mode, sign: v < 0 ? -1 : 1, level, dense: e.dense, path: new Path2D() }; buckets.set(key, b); }
      pathEdge(b.path, e);
    }
    // neutral wiring for layers whose lens is off
    ctx.lineCap = 'round';
    for (const b of buckets.values()) {
      const t = (b.level + 0.5) / BUCKETS;
      let alpha = b.mode === 'grad' ? 0.12 + 0.75 * t : 0.14 + 0.72 * t;
      if (b.dense) alpha *= 0.28;
      if (dimAll) alpha *= 0.18;
      ctx.strokeStyle = b.mode === 'grad' ? theme.gradColor(0.25 + 0.75 * t, alpha) : theme.divColor(b.sign * (0.3 + 0.7 * t), alpha);
      ctx.lineWidth = (b.dense ? 0.6 : 0.7) + (b.dense ? 1.2 : 3.6) * t;
      ctx.stroke(b.path);
    }
    if (dropPath) {
      ctx.strokeStyle = theme.c('line-2', 0.7); ctx.lineWidth = 0.7; ctx.setLineDash([2, 3]);
      ctx.stroke(dropPath); ctx.setLineDash([]);
    }
    if (st.walk) {
      // pending connections drawn as quiet wiring
      ctx.strokeStyle = theme.c('line-2', 0.5); ctx.lineWidth = 0.7;
      const p = new Path2D();
      for (const e of lay.edges) if (st.edgeMode[e.l] === 'off' && !e.dense) pathEdge(p, e);
      ctx.stroke(p);
    }
    for (const e of strong) {
      const mode = st.edgeMode[e.l], v = st.edgeVals[e.l][e.w], t = Math.abs(v);
      if (e.dense && t < 0.04) continue;
      const a = (0.3 + 0.7 * t) * (e.dense ? 0.75 : 1);
      ctx.strokeStyle = mode === 'grad' ? theme.gradColor(0.25 + 0.75 * t, a) : theme.divColor(Math.sign(v || 1) * (0.3 + 0.7 * t), a);
      ctx.lineWidth = (e.dense ? 0.7 : 0.9) + (e.dense ? 1.8 : 3.4) * t;
      const p = new Path2D(); pathEdge(p, e); ctx.stroke(p);
    }
    if (V.hoverEdge) {
      const e = V.hoverEdge;
      ctx.strokeStyle = theme.c('focus', 0.9); ctx.lineWidth = 1.5;
      const p = new Path2D(); pathEdge(p, e); ctx.stroke(p);
    }
    if (st.lens.flow && !st.walk && !NA.prefersReducedMotion()) drawFlow(ctx, lay, st, now);
  }

  function drawFlow(ctx, lay, st, now) {
    const backward = st.lens.edges === 'gradient';
    const list = [];
    for (const e of lay.edges) {
      if (st.edgeMode[e.l] === 'off') continue;
      const v = st.edgeVals[e.l][e.w], m = Math.abs(v);
      if (m > 0.22) list.push([e, v, m]);
    }
    if (list.length > 700) { list.sort((a, b) => b[2] - a[2]); list.length = 700; }
    const time = now / 1000;
    for (const [e, v, m] of list) {
      const k = m > 0.7 ? 2 : 1;
      const phase = ((e.l * 131 + e.i * 37 + e.j * 17) % 97) / 97;
      ctx.fillStyle = st.edgeMode[e.l] === 'grad' ? theme.gradColor(0.4 + 0.6 * m) : theme.divColor(Math.sign(v) * (0.45 + 0.55 * m));
      for (let q = 0; q < k; q++) {
        let u = (time * (0.22 + 0.35 * m) + phase + q / k) % 1;
        if (backward) u = 1 - u;
        const p = bez(e, u);
        ctx.beginPath(); ctx.arc(p[0], p[1], 1.3 + 1.6 * m, 0, Math.PI * 2); ctx.fill();
      }
    }
  }

  function drawInputImage(ctx, col, st) {
    const im = col.image, G = im.G, a = st.trace.a[0];
    ctx.fillStyle = theme.c('panel-2');
    rr(ctx, im.x - 3, im.y - 3, im.side + 6, im.side + 6, 6); ctx.fill();
    const ink = theme.tokens.ink, base = theme.tokens['panel-2'];
    for (let j = 0; j < G * G; j++) {
      const t = clamp(a[j], 0, 1);
      if (t < 0.02) continue;
      ctx.fillStyle = NA.css(NA.mixRgb(base, ink, t));
      ctx.fillRect(im.x + (j % G) * im.cell, im.y + Math.floor(j / G) * im.cell, im.cell + 0.4, im.cell + 0.4);
    }
    // the selected hidden neuron's weights, overlaid as a faint template outline
    ctx.strokeStyle = theme.c('line-2'); ctx.lineWidth = 1;
    rr(ctx, im.x - 3, im.y - 3, im.side + 6, im.side + 6, 6); ctx.stroke();
    if (V.hoverPixel !== null) {
      const j = V.hoverPixel;
      ctx.strokeStyle = theme.c('focus'); ctx.lineWidth = 1.5;
      ctx.strokeRect(im.x + (j % G) * im.cell, im.y + Math.floor(j / G) * im.cell, im.cell, im.cell);
    }
  }

  function drawNode(ctx, nd, st, now) {
    const l = nd.l, j = nd.j, s = nd.s, x = nd.x - s / 2, y = nd.y - s / 2, r = Math.min(6, s * 0.2);
    const net = st.net, ds = App.ds;
    const maps = App.lenses.maps && !st.walk;
    const val = nodeValue(st, l, j);
    const ablated = l < net.L && !net.mask[l][j];
    ctx.save();
    rr(ctx, x, y, s, s, r);
    ctx.fillStyle = theme.c('panel-2');
    ctx.fill();
    let drewThumb = false;
    if (maps) {
      ctx.save();
      rr(ctx, x, y, s, s, r); ctx.clip();
      if (ds.kind === 'line') { drawCurveThumb(ctx, l, j, x, y, s); drewThumb = true; }
      else {
        const c = V.thumbs.get(l + ':' + j);
        if (c && (ds.kind === 'plane' || (ds.kind === 'image' && l >= 1))) {
          ctx.imageSmoothingEnabled = ds.kind === 'plane';
          ctx.drawImage(c, x, y, s, s);
          drewThumb = true;
          if (ds.kind === 'plane') drawProbeMark(ctx, x, y, s);
        }
      }
      ctx.restore();
    }
    if (!drewThumb) {
      rr(ctx, x, y, s, s, r);
      ctx.fillStyle = valueColor(val);
      ctx.fill();
      if (!val) {
        ctx.fillStyle = theme.c('muted');
        ctx.font = `600 ${Math.max(9, s * 0.42)}px ${getFont('mono')}`;
        ctx.textAlign = 'center'; ctx.textBaseline = 'middle';
        ctx.fillText('?', nd.x, nd.y + 1);
      }
    }
    // border
    rr(ctx, x, y, s, s, r);
    ctx.lineWidth = 1;
    ctx.strokeStyle = theme.c('line-2');
    ctx.stroke();
    // value pip in the corner when a thumbnail hides the fill
    if (drewThumb && val) {
      const ps = clamp(s * 0.28, 7, 12);
      ctx.fillStyle = theme.c('panel');
      rr(ctx, x + s - ps + 2 - 1.5, y + s - ps + 2 - 1.5, ps + 3, ps + 3, 3); ctx.fill();
      ctx.fillStyle = valueColor(val);
      rr(ctx, x + s - ps + 2, y + s - ps + 2, ps, ps, 2); ctx.fill();
    }
    // walkthrough error signal ring
    if (st.walk && st.walk.showDelta(l)) {
      const d = st.trace.delta[l][j], m = st.walk.deltaMax[l] || 1e-12, t = Math.min(1, Math.abs(d) / m);
      ctx.lineWidth = 1 + 3 * t;
      ctx.strokeStyle = theme.gradColor(0.35 + 0.65 * t);
      rr(ctx, x - 3.5, y - 3.5, s + 7, s + 7, r + 3); ctx.stroke();
    }
    // health
    let healthTag = '';
    if (st.stats && l > 0 && l < net.L && !ablated) {
      const sd = st.stats[l][j], dead = NA.engine.deadKind(net, st.stats, l, j);
      if (dead) {
        ctx.save(); rr(ctx, x, y, s, s, r); ctx.clip();
        ctx.fillStyle = theme.c('panel-2', 0.75); ctx.fillRect(x, y, s, s);
        ctx.strokeStyle = theme.c('muted', 0.9); ctx.lineWidth = 1;
        for (let k = -s; k < s; k += 5) { ctx.beginPath(); ctx.moveTo(x + k, y + s); ctx.lineTo(x + k + s, y); ctx.stroke(); }
        ctx.restore();
        healthTag = dead;
      } else if (sd.satFrac > 0.9) {
        ctx.strokeStyle = theme.c('serious'); ctx.lineWidth = 2;
        rr(ctx, x - 2, y - 2, s + 4, s + 4, r + 2); ctx.stroke();
        healthTag = 'saturated';
      }
    }
    if (ablated) {
      rr(ctx, x, y, s, s, r);
      ctx.fillStyle = theme.c('panel', 0.72); ctx.fill();
      ctx.strokeStyle = theme.c('critical'); ctx.lineWidth = 1.6;
      ctx.beginPath(); ctx.moveTo(x + 4, y + 4); ctx.lineTo(x + s - 4, y + s - 4); ctx.moveTo(x + s - 4, y + 4); ctx.lineTo(x + 4, y + s - 4); ctx.stroke();
    } else if (st.drop && l > 0 && l < net.L && !st.drop[l][j]) {
      // switched off by dropout for the last training example
      rr(ctx, x, y, s, s, r);
      ctx.fillStyle = theme.c('panel', 0.7); ctx.fill();
      ctx.setLineDash([3, 2.5]); ctx.strokeStyle = theme.c('ink', 0.6); ctx.lineWidth = 1.3;
      rr(ctx, x - 2.5, y - 2.5, s + 5, s + 5, r + 2.5); ctx.stroke(); ctx.setLineDash([]);
    }
    // bias marker
    if (App.lenses.bias && l > 0) {
      const b = net.b[l - 1][j];
      let bm = 1e-9; for (const v of net.b[l - 1]) bm = Math.max(bm, Math.abs(v));
      const t = b / bm, h = Math.max(2, Math.abs(t) * s * 0.9);
      ctx.fillStyle = theme.divColor(Math.sign(t || 1) * (0.35 + 0.65 * Math.abs(t)));
      rr(ctx, x - 7, nd.y + s * 0.45 - h, 3.5, h, 1.5); ctx.fill();
    }
    // numbers (drawn later, above the wiring)
    let txt = '';
    if (App.lenses.values && l > 0 && !(l === net.L && ds.task !== 'regression')) {
      const tr = st.trace;
      const mode = st.walk ? st.walk.valueMode(l) : st.lens.nodes;
      if (mode === 'pending') txt = '';
      else if (mode === 'walkdelta' && tr.delta) txt = 'δ ' + fmt(tr.delta[l][j], 2);
      else if (mode === 'gradient' && st.grad) txt = fmt(st.grad.dRms[l][j], 3);
      else if (mode === 'preact') txt = fmt(tr.z[l][j], 2);
      else txt = fmt(tr.a[l][j], 2);
    }
    const act = showsAct(l) && !st.walk ? net.actOf(l, j) : null;
    if (txt || healthTag || act) V.labels.push({ nd, act, text: [txt, healthTag].filter(Boolean).join(' · '), health: !!healthTag });
    // selection / hover / highlight rings
    const sel = App.selected && App.selected.l === l && App.selected.j === j;
    const hov = App.hover && App.hover.l === l && App.hover.j === j;
    if (sel || hov) {
      ctx.strokeStyle = theme.c('focus', sel ? 1 : 0.6);
      ctx.lineWidth = sel ? 2 : 1.5;
      rr(ctx, x - 3, y - 3, s + 6, s + 6, r + 3); ctx.stroke();
    }
    if (App.highlight && now < App.highlight.until && App.highlight.list.some((h) => h.l === l && h.j === j)) {
      const pulse = 0.5 + 0.5 * Math.sin(now / 160);
      ctx.strokeStyle = theme.c('warn', 0.5 + 0.5 * pulse); ctx.lineWidth = 2.5;
      rr(ctx, x - 6 - pulse * 2, y - 6 - pulse * 2, s + 12 + pulse * 4, s + 12 + pulse * 4, r + 6); ctx.stroke();
    }
    ctx.restore();
  }

  // Labels under (or beside) neurons: the activation function's curve, name and formula, the value and
  // health tags. They keep a constant on-screen size, so zooming in makes room for the full formula.
  function drawNodeLabels(ctx, lay) {
    const sc = 1 / V.zoom.k, h = 13 * sc, pad = 3.5 * sc, gw = 15 * sc, gh = 9 * sc;
    const colGap = lay.cols.length > 1 ? lay.cols[1].x - lay.cols[0].x : 200;
    // the part of the diagram on screen, in layout coordinates: zoomed in, a label must not run off its edges
    const vx0 = -V.zoom.tx * sc + 4 * sc, vx1 = (V.W - V.zoom.tx) * sc - 4 * sc;
    ctx.font = `500 ${9.5 * sc}px ${getFont('mono')}`;
    ctx.textBaseline = 'middle'; ctx.textAlign = 'left';
    for (const lb of V.labels) {
      const nd = lb.nd, s = nd.s, below = nd.gap >= h + sc;
      // a function-only label beside the node would crowd the wiring: in a layer that shares one function
      // (already named on its header chip) it waits for zooming in, and anywhere it waits until it fits
      if (!below && !lb.text && ((s + nd.gap) * V.zoom.k < 15 || (V.zoom.k < 1.5 && App.net.layerActId(nd.l) !== 'mixed'))) continue;
      // the formula is spelled out only where it fits, centred under its neuron or beside it, on screen
      const sideX = nd.x + s / 2 + 4 * sc;
      const room = below ? Math.min(colGap * 0.94, 2 * Math.min(nd.x - vx0, vx1 - nd.x)) : Math.min(colGap - s - 10 * sc, vx1 - sideX);
      const tail = lb.text ? (lb.act ? ' · ' : '') + lb.text : '';
      let main = lb.act ? lb.act.short : '';
      if (lb.act) {
        const full = `${lb.act.short}  f(z) = ${lb.act.formula}`;
        if (ctx.measureText(full + tail).width + gw + 3 * pad <= room) main = full;
      }
      const text = main + tail, w = ctx.measureText(text).width + 2 * pad + (lb.act ? gw + 3 * sc : 0);
      let x, y;
      // a label under a neuron that is on screen slides sideways to stay on screen with it
      if (below) {
        x = nd.x - w / 2; y = nd.y + s / 2 + sc;
        if (nd.x >= vx0 && nd.x <= vx1) x = clamp(x, vx0, Math.max(vx0, vx1 - w));
      } else { x = sideX; y = nd.y + s / 2 - h + 3 * sc; }
      ctx.fillStyle = theme.c('panel', 0.92);
      rr(ctx, x, y, w, h, 3 * sc); ctx.fill();
      if (lb.health || lb.act) { ctx.strokeStyle = theme.c('line-2'); ctx.lineWidth = sc; rr(ctx, x + sc / 2, y + sc / 2, w - sc, h - sc, 3 * sc); ctx.stroke(); }
      let tx = x + pad;
      if (lb.act) { NA.ui.drawActGlyph(ctx, lb.act, tx, y + (h - gh) / 2, gw, gh, theme.c('ink'), theme.c('line-2'), sc); tx += gw + 3 * sc; }
      ctx.fillStyle = theme.c(lb.health || lb.act ? 'ink' : 'ink-2');
      ctx.fillText(text, tx, y + h / 2 + sc / 2);
    }
  }

  function drawProbeMark(ctx, x, y, s) {
    const p = App.probe;
    if (!p || p.raw.length !== 2) return;
    const px = x + ((p.raw[0] + 1) / 2) * s, py = y + ((1 - p.raw[1]) / 2) * s;
    ctx.beginPath(); ctx.arc(px, py, 2.6, 0, Math.PI * 2);
    ctx.fillStyle = theme.c('ink'); ctx.fill();
    ctx.lineWidth = 1.2; ctx.strokeStyle = theme.c('panel'); ctx.stroke();
  }

  function drawCurveThumb(ctx, l, j, x, y, s) {
    const g = App.grid(); if (!g) return;
    const A = g.F.a[l], n = g.n;
    let mn = Infinity, mx = -Infinity;
    for (let p = 0; p < n; p++) { const v = A[j * n + p]; if (v < mn) mn = v; if (v > mx) mx = v; }
    const lo = Math.min(mn, 0), hi = Math.max(mx, 0), span = hi - lo || 1, pad = s * 0.14;
    const Y = (v) => y + s - pad - ((v - lo) / span) * (s - 2 * pad);
    ctx.strokeStyle = theme.c('line-2'); ctx.lineWidth = 1;
    ctx.beginPath(); ctx.moveTo(x, Y(0)); ctx.lineTo(x + s, Y(0)); ctx.stroke();
    ctx.strokeStyle = theme.c('pos'); ctx.lineWidth = 1.6; ctx.lineJoin = 'round';
    ctx.beginPath();
    for (let p = 0; p < n; p += 2) { const px = x + (p / (n - 1)) * s, py = Y(A[j * n + p]); if (p === 0) ctx.moveTo(px, py); else ctx.lineTo(px, py); }
    ctx.stroke();
    const pr = App.probe;
    if (pr && pr.raw.length === 1) {
      const px = x + ((pr.raw[0] + 1) / 2) * s;
      ctx.strokeStyle = theme.c('ink', 0.55); ctx.lineWidth = 1;
      ctx.beginPath(); ctx.moveTo(px, y); ctx.lineTo(px, y + s); ctx.stroke();
    }
  }

  function drawLabels(ctx, lay, st) {
    const ds = App.ds, net = st.net, L = net.L, tr = st.trace;
    const mono = getFont('mono'), ui = getFont('ui');
    // input labels
    const c0 = lay.cols[0];
    if (!c0.image) {
      for (const nd of c0.nodes) {
        ctx.textAlign = 'right'; ctx.textBaseline = 'middle';
        ctx.fillStyle = theme.c('ink');
        ctx.font = `600 13px ${ui}`;
        ctx.fillText(ds.featureLabels[nd.j], nd.x - nd.s / 2 - 9, nd.y - (App.lenses.values ? 6 : 0));
        if (App.lenses.values) {
          // inputs live in [−1, 1]: two decimals always (no "3×10⁻⁴" for a probe a hair from 0), so the margin
          // left for these labels in computeLayout holds
          const v = tr.a[0][nd.j];
          ctx.font = `500 10px ${mono}`; ctx.fillStyle = theme.c('ink-2');
          ctx.fillText(fmt(Math.abs(v) < 0.005 ? 0 : v, 2), nd.x - nd.s / 2 - 9, nd.y + 8);
        }
      }
    } else {
      ctx.textAlign = 'center'; ctx.textBaseline = 'top';
      ctx.font = `500 10px ${mono}`; ctx.fillStyle = theme.c('muted');
      ctx.fillText(`${ds.imgSize}×${ds.imgSize} = ${net.sizes[0]} pixels`, c0.x, c0.image.y + c0.image.side + 9);
    }
    // output labels
    const cL = lay.cols[L];
    const narrow = V.W < 520;
    for (const nd of cL.nodes) {
      const lx = nd.x + nd.s / 2 + (narrow ? 6 : 10);
      const outW = Math.max(40, Math.min(84, V.W - lx - 8));
      const a = tr.a[L][nd.j];
      ctx.textBaseline = 'middle'; ctx.textAlign = 'left';
      if (narrow) {
        if (st.walk && st.walk.nodeMode(L) === 'pending') continue;
        const txt = ds.task === 'regression' ? fmt(a, 2) : ds.task === 'binary' ? NA.fmtPct(a) : `${App.isImage() ? nd.j : 'C' + nd.j} ${NA.fmtPct(a)}`;
        ctx.font = `600 10.5px ${mono}`; ctx.fillStyle = theme.c('ink');
        ctx.fillText(txt, lx, nd.y);
        continue;
      }
      if (ds.task === 'regression') {
        ctx.font = `600 12px ${mono}`; ctx.fillStyle = theme.c('ink');
        ctx.fillText('ŷ ' + fmt(a, 3), lx, nd.y);
        continue;
      }
      const p = a;
      const label = ds.task === 'binary' ? 'P(blue)' : App.isImage() ? String(nd.j) : 'C' + nd.j;
      if (st.walk && st.walk.nodeMode(L) === 'pending') continue;
      ctx.font = `600 ${App.isImage() ? 13 : 11}px ${App.isImage() ? mono : ui}`;
      ctx.fillStyle = theme.c('ink');
      const tall = nd.s >= 22;
      ctx.fillText(label, lx, nd.y - (tall ? 6 : 0));
      const lw = ctx.measureText(label).width;
      const bw = outW * clamp(p, 0, 1), by = nd.y + 4;
      if (tall) {
        ctx.fillStyle = theme.c('line');
        rr(ctx, lx, by, outW, 5, 2.5); ctx.fill();
        ctx.fillStyle = ds.task === 'binary' ? theme.divColor(2 * p - 1) : App.isImage() ? theme.c('pos') : theme.classColor(nd.j);
        if (bw > 0.5) { rr(ctx, lx, by, Math.max(2, bw), 5, 2.5); ctx.fill(); }
        ctx.font = `500 10px ${mono}`; ctx.fillStyle = theme.c('ink-2'); ctx.textAlign = 'left';
        ctx.fillText(NA.fmtPct(p, p < 0.995 && p > 0.005 ? 0 : 1), lx + lw + 6, nd.y - 6);
      } else {
        ctx.font = `500 10px ${mono}`; ctx.fillStyle = theme.c('ink-2');
        ctx.fillText(NA.fmtPct(p), lx + lw + 6, nd.y);
      }
    }
  }

  const fontCache = {};
  function getFont(kind) {
    if (!fontCache[kind]) fontCache[kind] = getComputedStyle(document.documentElement).getPropertyValue(kind === 'mono' ? '--font-mono' : '--font-ui').trim() || 'monospace';
    return fontCache[kind];
  }
  NA.getFont = getFont;

  // ------------------------------------------------------------------ headers & architecture controls
  // Chip showing a hidden layer's activation (curve + name); opens the layer menu.
  function actChip(l) {
    const id = App.net.layerActId(l), act = id !== 'mixed' ? NA.engine.ACTIVATIONS[id] : null;
    const b = el('button', { class: 'lh-chip', type: 'button', 'aria-haspopup': 'listbox',
      title: act ? `${act.label}: f(z) = ${act.formula}. Click to change it for the whole layer.` : 'Neurons in this layer use different activation functions. Click to set one for all.' },
      act ? NA.ui.actGlyph(id, 18, 12) : el('span', { class: 'lh-chip-mixed', 'aria-hidden': 'true', text: '≋' }),
      el('span', { class: 'lh-chip-text', text: act ? act.short : 'mixed' }),
      el('span', { class: 'lh-chip-count', text: String(App.net.sizes[l]) }),
      el('span', { class: 'lh-chip-caret', 'aria-hidden': 'true', text: '▾' }));
    b.setAttribute('aria-label', `Activation for ${App.layerName(l)}: ${act ? act.label : 'mixed'}. Change`);
    b.addEventListener('click', () => openLayerMenu(l, b));
    return b;
  }
  function openLayerMenu(l, anchor) {
    const net = App.net, id = net.layerActId(l), h = l - 1, n = net.sizes[l], max = App.isImage() ? 48 : 32;
    const items = NA.ui.activationItems(id).concat([
      { divider: true },
      n < max ? { id: '+n', label: `Add a neuron (${n} → ${n + 1})`, icon: '+' } : null,
      n > 1 ? { id: '-n', label: `Remove a neuron (${n} → ${n - 1})`, icon: '−' } : null,
      { id: '+after', label: 'Add a layer after this one', icon: '⊕' },
      { id: '+before', label: 'Add a layer before this one', icon: '⊕' },
      { id: 'remove', label: 'Remove this layer', icon: '×', tone: 'danger' },
    ].filter(Boolean));
    NA.ui.menu.open({
      anchor, title: `${App.layerName(l)} · ${n} neuron${n > 1 ? 's' : ''}`,
      note: id === 'mixed' ? 'Its neurons use different functions. Pick one to set them all, or tap a single neuron to change just that one.' : 'Applies to every neuron in this layer. Tap a single neuron to change just that one.',
      items,
      onSelect: (v) => {
        if (v === '+n') App.resizeLayer(h, 1);
        else if (v === '-n') App.resizeLayer(h, -1);
        else if (v === '+after') App.addLayer(h);
        else if (v === '+before') App.addLayer(h - 1);
        else if (v === 'remove') App.removeLayer(h);
        else App.setLayerAct(l, v);
      },
    });
  }
  // Activation chooser for one neuron, with a shortcut to give its whole layer the same function.
  NA.openNeuronMenu = function (l, j, anchor) {
    const act = App.net.actOf(l, j);
    NA.ui.menu.open({
      anchor, title: `Activation for ${App.neuronName(l, j)}`, note: 'Only this neuron changes; its weights are kept so you can see the effect at once.',
      items: NA.ui.activationItems(act.id).concat([{ divider: true }, { id: '@layer', label: `Use ${act.label} for all of ${App.layerName(l)}`, icon: '≡' }]),
      onSelect: (v) => { if (v === '@layer') App.setLayerAct(l, act.id); else if (v !== act.id) App.setNeuronAct(l, j, v); },
    });
  };
  // Loss chooser shared by the output header, the inspector and the settings.
  NA.openLossMenu = function (anchor) {
    const net = App.net, E = NA.engine;
    NA.ui.menu.open({
      anchor, title: 'Loss function', note: E.LOSSES[net.lossId].hint,
      items: E.LOSS_FOR[net.out].map((id) => ({ id, label: E.LOSSES[id].label, sub: E.LOSSES[id].formula[net.out], selected: id === net.lossId, icon: 'L' })),
      onSelect: (id) => { if (id !== net.lossId) App.setLoss(id); },
    });
  };
  function lossChip() {
    const net = App.net, Ls = NA.engine.LOSSES[net.lossId];
    const b = el('button', { class: 'lh-chip lh-chip-loss', type: 'button', 'aria-haspopup': 'listbox', title: `Loss: ${Ls.label}, ${Ls.formula[net.out]}. Click to change.` },
      el('span', { class: 'lh-chip-text', text: Ls.short, dataset: { full: Ls.short, tiny: 'Loss' } }), el('span', { class: 'lh-chip-caret', 'aria-hidden': 'true', text: '▾' }));
    b.setAttribute('aria-label', `Loss function: ${Ls.label}. Change`);
    b.addEventListener('click', () => NA.openLossMenu(b));
    return b;
  }
  NA.openLayerMenu = openLayerMenu;

  function buildHeaders() {
    NA.clear(V.overlay);
    V.stage.querySelectorAll('.depth-ctl').forEach((n) => n.remove());
    const net = App.net, ds = App.ds, L = net.L, max = ds.kind === 'image' ? 48 : 32;
    V.lossKey = net.lossId;
    V.headers = [];
    for (let l = 0; l <= L; l++) {
      const hidden = l > 0 && l < L;
      const h = el('div', { class: 'lh' + (hidden ? ' lh-hidden' : '') }, el('div', { class: 'lh-title', text: App.layerName(l) }));
      if (hidden) {
        h.append(
          el('div', { class: 'lh-row lh-size' },
            el('button', { class: 'mini-btn', type: 'button', 'aria-label': `Remove a neuron from hidden layer ${l}`, title: 'Remove a neuron', disabled: net.sizes[l] <= 1, onclick: () => App.resizeLayer(l - 1, -1) }, '−'),
            el('span', { class: 'lh-meta lh-count', text: String(net.sizes[l]), title: `${net.sizes[l]} neurons` }),
            el('button', { class: 'mini-btn', type: 'button', 'aria-label': `Add a neuron to hidden layer ${l}`, title: 'Add a neuron', disabled: net.sizes[l] >= max, onclick: () => App.resizeLayer(l - 1, 1) }, '+')),
          actChip(l));
      } else {
        const meta = l === 0 ? (ds.kind === 'image' ? `${ds.imgSize}×${ds.imgSize} px` : ds.kind === 'line' ? '1 input' : `${net.sizes[0]} feature${net.sizes[0] > 1 ? 's' : ''}`)
          : ds.task === 'regression' ? 'linear' : ds.task === 'binary' ? 'sigmoid' : `softmax · ${net.sizes[L]}`;
        const short = l === 0 ? (ds.kind === 'image' ? '14×14' : String(net.sizes[0])) : ds.task === 'regression' ? 'linear' : ds.task === 'binary' ? 'σ' : 'softmax';
        h.append(el('div', { class: 'lh-row' }, el('span', { class: 'lh-meta', text: meta, dataset: { full: meta, short } })));
        if (l === L) h.append(lossChip());
      }
      V.overlay.appendChild(h);
      V.headers.push(h);
    }
    // an inviting "+" between the last hidden layer and the output
    V.addBtn = el('button', { class: 'add-layer', type: 'button', title: 'Add a hidden layer here', 'aria-label': 'Add a hidden layer before the output', disabled: L - 1 >= 6, onclick: () => App.addLayer() }, '+');
    V.overlay.appendChild(V.addBtn);
    V.stage.appendChild(el('div', { class: 'depth-ctl' },
      el('span', { class: 'depth-label', text: 'Hidden layers' }),
      el('button', { class: 'mini-btn', type: 'button', 'aria-label': 'Remove the last hidden layer', disabled: L - 1 <= 0, onclick: () => App.removeLayer() }, '−'),
      el('span', { class: 'depth-n', text: String(L - 1) }),
      el('button', { class: 'mini-btn', type: 'button', 'aria-label': 'Add a hidden layer', disabled: L - 1 >= 6, onclick: () => App.addLayer() }, '+')));
    positionHeaders();
  }
  // Headers over narrow columns stack their controls (compact), then call the loss chip just "Loss", then keep
  // only a short title, the curve and the neuron count (tiny), and finally only the curve (micro). gap: the
  // on-screen distance between columns. The spacing picks a mode; the chips are then measured, and a mode whose
  // chips would touch their neighbours' (a long loss name, a phone's larger chips) gives way to the next one.
  const HEADER_MODES = ['full', 'compact', 'short loss', 'tiny', 'micro'];
  function setHeaderMode(gap) {
    let m = gap < 74 ? 3 : gap < 128 ? 1 : 0;
    applyHeaderMode(m);
    while (m < 4 && chipsCollide(gap)) applyHeaderMode(++m);
    return m >= 1;
  }
  function applyHeaderMode(m) {
    const compact = m >= 1, tiny = m >= 3, L = App.net.L;
    V.headers.forEach((h, l) => {
      h.classList.toggle('is-compact', compact);
      h.classList.toggle('is-tiny', tiny);
      h.classList.toggle('is-micro', m >= 4);
      const me = h.querySelector('.lh-meta[data-full]'), t = h.querySelector('.lh-title'), lt = h.querySelector('.lh-chip-text[data-full]');
      if (lt) lt.textContent = m >= 2 ? lt.dataset.tiny : lt.dataset.full;
      if (me) me.textContent = compact ? me.dataset.short : me.dataset.full;
      if (t) t.textContent = !tiny ? App.layerName(l) : l === 0 ? 'In' : l === L ? 'Out' : 'H' + l;
    });
    V.headerMode = HEADER_MODES[m];
  }
  function chipsCollide(gap) {
    const w = V.headers.map((h) => { const c = h.querySelector('.lh-chip'); return c ? c.offsetWidth : 0; });
    for (let i = 1; i < w.length; i++) if (w[i - 1] && w[i] && (w[i - 1] + w[i]) / 2 > gap - 3) return true;
    return false;
  }
  function positionHeaders() {
    if (!V.layout || !V.headers) return;
    // headers keep their size and stay along the top while zoomed, following their column sideways
    const z = V.zoom, k = z ? z.k : 1, tx = z ? z.tx : 0, W = V.stage.clientWidth;
    const cols = V.layout.cols, gap = (cols.length > 1 ? cols[1].x - cols[0].x : 999) * k;
    const compact = setHeaderMode(gap);
    cols.forEach((c, i) => {
      const h = V.headers[i];
      if (!h) return;
      const x = c.x * k + tx;
      h.style.left = x + 'px';
      h.hidden = x < -30 || x > W + 30;
    });
    if (V.addBtn) {
      const a = cols[cols.length - 2], b = cols[cols.length - 1];
      V.addBtn.hidden = compact || !a || !b;
      if (a && b) V.addBtn.style.left = ((a.x + b.x) / 2) * k + tx + 'px';
    }
  }

  // ------------------------------------------------------------------ legend
  function renderLegend() {
    if (!V.legend || !App.ds) return;
    const L = App.lenses, ds = App.ds;
    const bar = document.querySelector('.lensbar');
    if (bar) bar.classList.toggle('is-walking', !!(App.walk && App.walk.active));
    const ramp = (cls) => el('span', { class: 'ramp ' + cls, 'aria-hidden': 'true' });
    const items = [];
    if (App.walk && App.walk.active) {
      items.push(el('span', { class: 'lg-item walk-note' }, 'Backprop walkthrough is driving this view; lenses resume when you leave the tab.'));
      items.push(el('span', { class: 'lg-item' }, ramp('ramp-div'), 'forward: signal w·a and activations'));
      items.push(el('span', { class: 'lg-item' }, ramp('ramp-grad'), 'backward: |δ·a| and error signal δ (rings)'));
    } else {
      const edgeText = { weight: 'connection: weight w, width ∝ |w|', signal: 'connection: signal w·a for the probe', gradient: 'connection: |∂L/∂w|, log scale' }[L.edges];
      items.push(el('span', { class: 'lg-item' }, ramp(L.edges === 'gradient' ? 'ramp-grad' : 'ramp-div'), edgeText));
      const nodeText = { activation: 'neuron: activation a for the probe', preact: 'neuron: pre-activation z for the probe', gradient: 'neuron: RMS error signal δ, log scale' }[L.nodes];
      items.push(el('span', { class: 'lg-item' }, ramp(L.nodes === 'gradient' ? 'ramp-grad' : 'ramp-div'), nodeText + (L.maps ? ' (corner)' : '')));
      if (L.maps) items.push(el('span', { class: 'lg-item' }, ramp('ramp-map'), ds.kind === 'plane' ? 'thumbnail: its output over the input plane (dot: probe)' : ds.kind === 'line' ? 'thumbnail: its output for x from −1 to 1' : 'thumbnail: weights as a 14×14 image; outputs: their top images'));
      let mixed = false;
      for (let l = 1; l < App.net.L; l++) if (App.net.layerActId(l) === 'mixed') mixed = true;
      if (App.net.L > 1 && (L.acts || mixed)) {
        const a = App.net.actOf(1, 0);
        items.push(el('span', { class: 'lg-item' }, el('span', { class: 'lg-act' }, NA.ui.actGlyph(a.id, 16, 10), a.short), L.acts ? 'label: its activation function (zoom in for the formula)' : 'labelled neurons use their own activation'));
      }
    }
    if (V.dropLegend) items.push(el('span', { class: 'lg-item' }, el('i', { class: 'sw sw-drop' }), `dashed: dropped for the last training example (dropout ${NA.fmtPct(App.net.lastDrop, 0)})`));
    items.push(el('span', { class: 'lg-item lg-signs' }, el('i', { class: 'sw sw-neg' }), 'negative', el('i', { class: 'sw sw-pos' }), 'positive'));
    NA.clear(V.legend).append(...items);
  }

  // ------------------------------------------------------------------ interaction
  function hit(px, py) {
    const lay = V.layout;
    if (!lay) return null;
    for (const nd of lay.nodes) {
      const h = nd.s / 2 + 4;
      if (Math.abs(px - nd.x) <= h && Math.abs(py - nd.y) <= h) return { type: 'node', nd };
    }
    const c0 = lay.cols[0];
    if (c0.image) {
      const im = c0.image;
      if (px >= im.x && px < im.x + im.side && py >= im.y && py < im.y + im.side) {
        const gx = Math.floor((px - im.x) / im.cell), gy = Math.floor((py - im.y) / im.cell);
        return { type: 'pixel', j: gy * im.G + gx };
      }
    }
    let best = null, bd = 5;
    for (const e of lay.edges) {
      if (e.dense) continue;
      const minx = Math.min(e.x1, e.x2) - 5, maxx = Math.max(e.x1, e.x2) + 5;
      if (px < minx || px > maxx) continue;
      let prev = [e.x1, e.y1];
      for (let k = 1; k <= 12; k++) {
        const p = bez(e, k / 12);
        const d = segDist(px, py, prev[0], prev[1], p[0], p[1]);
        if (d < bd) { bd = d; best = e; }
        prev = p;
      }
    }
    return best ? { type: 'edge', e: best } : null;
  }
  function segDist(px, py, ax, ay, bx, by) {
    const dx = bx - ax, dy = by - ay, L2 = dx * dx + dy * dy;
    const t = L2 ? clamp(((px - ax) * dx + (py - ay) * dy) / L2, 0, 1) : 0;
    return Math.hypot(ax + t * dx - px, ay + t * dy - py);
  }

  function onMove(ev) {
    if (ev.pointerType === 'touch') return; // touch has no hover; a tap selects and the phone layout shows a peek card
    if (V.zoom.busy) { NA.tooltip.hide(); return; }
    const r = V.canvas.getBoundingClientRect(), [px, py] = V.zoom.toWorld(ev.clientX - r.left, ev.clientY - r.top);
    const h = hit(px, py);
    const prevEdge = V.hoverEdge, prevPix = V.hoverPixel;
    V.hoverEdge = h && h.type === 'edge' ? h.e : null;
    V.hoverPixel = h && h.type === 'pixel' ? h.j : null;
    if (h && h.type === 'node') {
      App.setHover({ l: h.nd.l, j: h.nd.j });
      V.canvas.style.cursor = 'pointer';
      NA.tooltip.show(nodeTip(h.nd.l, h.nd.j), ev.clientX, ev.clientY);
    } else {
      App.setHover(null);
      V.canvas.style.cursor = h && h.type === 'edge' ? 'pointer' : 'default';
      if (h && h.type === 'edge') NA.tooltip.show(edgeTip(h.e), ev.clientX, ev.clientY);
      else if (h && h.type === 'pixel') NA.tooltip.show(pixelTip(h.j), ev.clientX, ev.clientY);
      else NA.tooltip.hide();
    }
    if (prevEdge !== V.hoverEdge || prevPix !== V.hoverPixel) V.dirty = true;
  }
  function onLeave() { App.setHover(null); V.hoverEdge = null; V.hoverPixel = null; V.dirty = true; NA.tooltip.hide(); }
  function onClick(ev) {
    if (V.zoom.suppressClick) { V.zoom.suppressClick = false; return; } // the gesture was a pan or pinch
    const r = V.canvas.getBoundingClientRect(), [px, py] = V.zoom.toWorld(ev.clientX - r.left, ev.clientY - r.top), h = hit(px, py);
    if (h && h.type === 'node') App.select({ l: h.nd.l, j: h.nd.j }, { fromGraph: true });
    else if (h && h.type === 'edge') App.select({ l: h.e.l + 1, j: h.e.j }, { fromGraph: true, weight: h.e.i });
  }

  function row(label, value) { return el('div', { class: 'tt-row' }, el('span', { class: 'tt-k', text: label }), el('span', { class: 'tt-v', text: value })); }
  // "f(z) = …" and "a = f(<this z>) = …" for any neuron, including the output's squashing function.
  function actLines(l, j, tr) {
    const net = App.net, z = fmt(tr.z[l][j], 3), a = fmt(tr.a[l][j], 3);
    if (l === net.L) {
      if (net.out === 'softmax') return ['p = softmax(z): e^z divided by the sum over all outputs', `p = ${a}`];
      if (net.out === 'identity') return ['ŷ = z (linear output)', `ŷ = ${a}`];
    }
    const act = net.actOf(l, j);
    return [`${act.label}:  f(z) = ${act.formula}`, `${l === net.L ? 'p' : 'a'} = ${act.apply(z)} = ${a}`];
  }
  NA.actLines = actLines;
  function nodeTip(l, j) {
    const net = App.net, tr = App.walk && App.walk.active ? App.walk.trace : App.probeTrace();
    const kids = [el('div', { class: 'tt-title', text: App.neuronName(l, j) })];
    if (l === 0) kids.push(row('input value', fmt(tr.a[0][j], 3)));
    else {
      kids.push(row('z = w·a + b', fmt(tr.z[l][j], 3)));
      const [fx, ax] = actLines(l, j, tr);
      kids.push(el('div', { class: 'tt-formula', text: fx }), el('div', { class: 'tt-formula tt-strong', text: ax }));
      kids.push(row('bias b', fmt(net.b[l - 1][j], 3)));
      if (tr.delta) kids.push(row('δ for this sample', fmt(tr.delta[l][j], 4)));
      if (l < net.L && !net.mask[l][j]) kids.push(el('div', { class: 'tt-note', text: 'Ablated: forced to output 0' }));
      else if (l < net.L) {
        const drop = dropShown(App.walk && App.walk.active);
        if (drop && !drop[l][j]) kids.push(el('div', { class: 'tt-note', text: 'Dropout switched it off for the last training example: it passed nothing forward and got no update from that example. The values above have every neuron on, as when the model is used.' }));
      }
    }
    kids.push(el('div', { class: 'tt-hint', text: (l > 0 && l < net.L ? 'Click to inspect or change its activation' : 'Click to inspect') + ' · double-click or zoom in to open it' }));
    return kids;
  }
  function edgeTip(e) {
    const net = App.net, w = net.W[e.l][e.w], tr = App.probeTrace();
    const kids = [el('div', { class: 'tt-title', text: `${App.neuronName(e.l, e.i, true)} → ${App.neuronName(e.l + 1, e.j, true)}` }), row('weight w', fmt(w, 4)), row('signal w·a (probe)', fmt(w * tr.a[e.l][e.i], 4))];
    if (App.lenses.edges === 'gradient') {
      const g = App.gradReport().gW[e.l][e.w];
      kids.push(row('∂L/∂w', fmt(g, 5)));
      kids.push(el('div', { class: 'tt-note', text: g > 0 ? 'Gradient descent pushes this weight down' : g < 0 ? 'Gradient descent pushes this weight up' : 'No pressure on this weight' }));
    }
    kids.push(el('div', { class: 'tt-hint', text: 'Click to edit in the Neuron tab' }));
    return kids;
  }
  function pixelTip(j) {
    const tr = App.probeTrace(), s = App.hover || App.selected, net = App.net;
    const kids = [el('div', { class: 'tt-title', text: App.neuronName(0, j) }), row('ink', fmt(tr.a[0][j], 2))];
    if (s && s.l === 1) kids.push(row(`weight to ${App.neuronName(1, s.j, true)}`, fmt(net.W[0][s.j * net.sizes[0] + j], 3)));
    return kids;
  }

  NA.netView = V;
})(window.NA);
