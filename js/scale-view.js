/* scale-view.js — zooming out past the network. Below the network diagram's smallest zoom the view keeps going:
   your network shrinks to a dot beside the models of this site's labs, all of them fit in a corner of the
   smallest GPT-2, and GPT-2 in a corner of one GPU's memory. Each square has the area of its number of weights.
   Zoom back in (or double-click your network) to return to it. Built on the same zoomable view as the neuron's levels. */
(function (NA) {
  'use strict';
  const { el, theme, clamp, fmtInt } = NA;
  const App = NA.app;
  const SV = { open: false };
  // the labs' models, counted from their weights
  const LABS = [
    { id: 'yolo', title: 'YOLO26 detector', short: 'YOLO26', n: 385088, tone: 's1' },
    { id: 'transformer', title: 'Tiny Transformer', short: 'Transformer', n: 28452, tone: 's4' },
    { id: 'radar', title: 'Radar classifier', short: 'Radar', n: 27142, tone: 's2' },
    { id: 'diffusion', title: 'Diffusion model', short: 'Diffusion', n: 9218, tone: 's7' },
    { id: 'pinn', title: 'Physics-informed network', short: 'PINN', n: 2209, tone: 's3' },
  ];
  const GPT2 = 124439808, GPU_GB = 24, GPU_N = (GPU_GB * 1e9) / 4;
  const mine = () => (App.net ? App.net.paramCount : 0);
  const labsN = () => LABS.reduce((p, q) => p + q.n, 0) + mine();
  const bytes = (n) => { const b = n * 4; return b >= 1e9 ? `${(b / 1e9).toFixed(1)} GB` : b >= 1e6 ? `${Math.round(b / 1e6)} MB` : b >= 1e3 ? `${(b / 1e3).toFixed(1)} kB` : `${b} bytes`; };
  const pct = (a, b) => { const p = (100 * a) / b; return p >= 1 ? p.toFixed(1) : p >= 0.01 ? p.toFixed(2) : p.toPrecision(1); };
  const back = (view) => [view.tool('↩ Back to your network', () => SV.close(), { primary: true })];

  // a square of memory with a faint grid, and the square of the next level down in its top-left corner
  function memSquare(ctx, S, q, o) {
    ctx.fillStyle = theme.c(o.tone, theme.dark ? 0.13 : 0.08); ctx.fillRect(q.x, q.y, q.side, q.side);
    ctx.strokeStyle = theme.c(o.tone, 0.18); ctx.lineWidth = 1;
    const n = o.grid || 8;
    for (let i = 1; i < n; i++) { const t = (q.side * i) / n; ctx.beginPath(); ctx.moveTo(q.x + t, q.y); ctx.lineTo(q.x + t, q.y + q.side); ctx.moveTo(q.x, q.y + t); ctx.lineTo(q.x + q.side, q.y + t); ctx.stroke(); }
    ctx.strokeStyle = theme.c(o.tone, 0.75); ctx.lineWidth = 1.5; ctx.strokeRect(q.x + 0.5, q.y + 0.5, q.side - 1, q.side - 1);
  }
  // a label with a line to the small square it names
  function callout(ctx, S, from, to, lines, u) {
    ctx.strokeStyle = theme.c('ink-2', 0.7); ctx.lineWidth = 1.2; ctx.beginPath(); ctx.moveTo(from[0], from[1]); ctx.lineTo(to[0], to[1]); ctx.stroke();
    lines.forEach((l, i) => S.text(ctx, l, from[0] + 4, from[1] + u * (0.2 + i * 1.3), { size: i ? u * 0.85 : u, w: i ? 500 : 650, color: theme.c(i ? 'ink-2' : 'ink') }));
  }
  // one level: a big square on the left with a smaller one (the level inside) in its corner, words on the right
  function squareLevel(size, inside, total, o) {
    const wide = size.w / size.h >= 1.05, side = Math.min(size.h * 0.97, size.w * (wide ? 0.6 : 0.97)), small = Math.max(3, side * Math.sqrt(inside / total)), pad = side * 0.03;
    const box = { x: 0, y: 0, side }, sq = { x: pad, y: pad, side: small };
    return {
      rect: [sq.x / size.w, sq.y / size.h, small / size.w, small / size.h],
      inner: (ctx, c, S) => {
        const k = c.w / size.w, B = { x: c.x + box.x * k, y: c.y + box.y * k, side: box.side * k }, Q = { x: c.x + sq.x * k, y: c.y + sq.y * k, side: sq.side * k }, u = S.u(c, 9, 15);
        memSquare(ctx, S, B, { tone: o.tone, grid: o.grid });
        ctx.fillStyle = theme.c('panel'); ctx.fillRect(Q.x, Q.y, Q.side, Q.side);
        ctx.strokeStyle = theme.c(o.innerTone, 0.9); ctx.lineWidth = 1.5; ctx.strokeRect(Q.x, Q.y, Q.side, Q.side);
        const fx = Q.x + Q.side + Math.max(24, B.side * 0.12), fy = Q.y + Q.side + Math.max(22, B.side * 0.1);
        callout(ctx, S, [fx, fy], [Q.x + Q.side, Q.y + Q.side], o.label, u);
        const tx = wide ? B.x + B.side + c.w * 0.03 : B.x, ty = wide ? B.y + u : B.y + B.side + u * 1.6, tw = wide ? c.x + c.w - tx : c.w;
        S.lines(ctx, o.words, { x: tx, y: ty, w: tw, h: wide ? B.side : c.y + c.h - ty }, { size: u, strong: (i) => i === 0 });
      },
    };
  }
  // the labs' models (and yours) side by side, squares with the area of their weights, standing on one line
  function labsLayout(w, h) {
    const items = [...LABS.map((q) => ({ key: q.id, short: q.short, title: q.title, n: q.n, tone: q.tone })), { key: 'net', short: 'yours', title: 'Your network', n: Math.max(1, mine()), tone: 'neg' }];
    const gap = Math.max(6, w * 0.012), slotMin = clamp(w / 8, 46, 96), labelH = 34;
    let k = (h * 0.8 - labelH) / Math.sqrt(items[0].n);
    const width = (kk) => items.reduce((p, it) => p + Math.max(Math.sqrt(it.n) * kk, slotMin), 0) + gap * (items.length - 1);
    for (let i = 0; i < 60 && width(k) > w * 0.98; i++) k *= 0.93;
    const base = h * 0.1 + Math.sqrt(items[0].n) * k;
    let x = (w - width(k)) / 2;
    for (const it of items) { it.side = Math.sqrt(it.n) * k; it.slot = Math.max(it.side, slotMin); it.x = x + (it.slot - it.side) / 2; it.y = base - it.side; it.cx = x + it.slot / 2; it.sx = x; x += it.slot + gap; }
    return { items, base, labelH };
  }
  // your network as a small diagram: its layers as columns of dots, wired
  function drawNet(ctx, c, S) {
    const sizes = (App.net && App.net.sizes) || [2, 4, 1], L = sizes.length, u = S.u(c, 9, 15);
    const col = (l) => c.x + c.w * (0.12 + (0.76 * l) / Math.max(1, L - 1)), row = (l, j) => c.y + c.h * (0.12 + (0.76 * (j + 0.5)) / sizes[l]);
    ctx.strokeStyle = theme.c('line-2'); ctx.lineWidth = 1;
    for (let l = 1; l < L; l++) for (let j = 0; j < sizes[l]; j++) for (let i = 0; i < sizes[l - 1]; i++) { ctx.beginPath(); ctx.moveTo(col(l - 1), row(l - 1, i)); ctx.lineTo(col(l), row(l, j)); ctx.stroke(); }
    const rad = clamp(Math.min(c.h * 0.3 / Math.max(...sizes), c.w * 0.04), 3, 16);
    for (let l = 0; l < L; l++) for (let j = 0; j < sizes[l]; j++) { ctx.beginPath(); ctx.arc(col(l), row(l, j), rad, 0, Math.PI * 2); ctx.fillStyle = theme.c('panel'); ctx.fill(); ctx.strokeStyle = theme.c(l === 0 || l === L - 1 ? 'ink-2' : 's1'); ctx.lineWidth = 1.6; ctx.stroke(); }
    S.text(ctx, `${sizes.join(' → ')}: ${fmtInt(mine())} weights and biases`, c.x, c.y + c.h * 0.97, { size: u, w: 600, color: theme.c('ink-2') });
  }

  function build(size) {
    const m = mine(), all = labsN();
    const net = {
      id: 'net', title: 'Your network', sub: `${fmtInt(m)} weights and biases`, tone: 's1', shape: 'ghost',
      inner: (ctx, c, S) => drawNet(ctx, c, S), portal: () => SV.close(),
      about: ['Your network. Zoom in on it, or double-click it, to go back to it.'],
    };
    // the labs level: its parts are drawn by the level itself; the parts listed here only take the pointer
    const lay = labsLayout(size.w, size.h);
    const labs = {
      id: 'labs', title: 'The models in this site', sub: `and yours: ${fmtInt(all)} weights in all`, tone: 's4', shape: 'ghost',
      count: `Inside: ${fmtInt(all)} weights, ${bytes(all)} as float32.`,
      summary: (ctx, r) => { ctx.fillStyle = theme.c('s4', 0.25); ctx.fillRect(r.x, r.y, r.w, r.h); },
      inner: (ctx, c, S) => {
        const k = c.w / size.w, u = S.u(c, 8.5, 13), base = c.y + lay.base * k;
        ctx.strokeStyle = theme.c('line-2'); ctx.lineWidth = 1; ctx.beginPath(); ctx.moveTo(c.x, base + 0.5); ctx.lineTo(c.x + c.w, base + 0.5); ctx.stroke();
        for (const it of lay.items) {
          const x = c.x + it.x * k, y = c.y + it.y * k, s = Math.max(it.side * k, 2), cx = c.x + it.cx * k, slot = it.slot * k;
          if (it.key === 'net') {
            ctx.fillStyle = theme.c('neg'); ctx.fillRect(cx - s / 2, base - s, s, s);
            ctx.strokeStyle = theme.c('neg', 0.5); ctx.lineWidth = 1.5; ctx.beginPath(); ctx.arc(cx, base - s / 2, Math.max(7, s), 0, Math.PI * 2); ctx.stroke();
            callout(ctx, S, [cx - slot * 0.42, base - Math.max(56, slot * 0.6)], [cx, base - s - 4], ['your network', `${fmtInt(m)} weights`], u);
          } else {
            ctx.fillStyle = theme.c(it.tone, theme.dark ? 0.3 : 0.2); ctx.fillRect(x, y, s, s);
            ctx.strokeStyle = theme.c(it.tone, 0.85); ctx.lineWidth = 1.3; ctx.strokeRect(x + 0.5, y + 0.5, s - 1, s - 1);
          }
          S.text(ctx, it.short, cx, base + u * 1.1, { size: u, w: 650, align: 'center', maxW: slot - 2, color: theme.c(it.key === 'net' ? 'neg' : 'ink') });
          S.text(ctx, fmtInt(it.n), cx, base + u * 2.4, { size: u * 0.9, mono: true, align: 'center', maxW: slot - 2, color: theme.c('ink-2') });
        }
        S.text(ctx, 'each square has the area of a model’s number of weights', c.x, c.y + c.h * 0.98, { size: u * 0.85, color: theme.c('muted'), maxW: c.w });
      },
      children: [
        ...lay.items.filter((it) => it.key !== 'net').map((it) => ({
          id: it.key, title: it.title, sub: `${fmtInt(it.n)} weights`, shape: 'ghost', rect: [it.sx / size.w, it.y / size.h, it.slot / size.w, (it.side + lay.labelH) / size.h],
          about: [`The ${it.title} in this site’s lab has ${fmtInt(it.n)} weights: ${bytes(it.n)} as float32, ${Math.round(it.n / Math.max(1, m))} times as many as your network. Small enough to run in a browser.`],
        })),
        Object.assign(net, { rect: (() => { const it = lay.items[lay.items.length - 1]; return [it.sx / size.w, (lay.base - Math.max(40, it.slot)) / size.h, it.slot / size.w, (Math.max(40, it.slot) + lay.labelH) / size.h]; })() }),
      ],
      tools: back,
      about: [`Each square has the area of a model’s number of weights. Your network has ${fmtInt(m)} weights and biases, ${bytes(m)} as float32: beside the YOLO26 detector’s ${fmtInt(LABS[0].n)} it is a dot. The labs’ models are small on purpose, so that they run (and some of them train) right here in your browser.`, 'Zoom out again for a model people actually used, or zoom in on your network (or double-click it) to go back to it.'],
    };
    const G = squareLevel(size, all, GPT2, { tone: 's2', innerTone: 's4', grid: 12, label: ['all of this site’s models', `${fmtInt(all)} weights, ${pct(all, GPT2)}%`], words: ['124 million weights', `${bytes(GPT2)} as float32`, 'GPT-2 from 2019, its smallest size: 12 transformer blocks. All the models in this site fill the little square in its corner.'] });
    const gpt2 = {
      id: 'gpt2', title: 'GPT-2, the smallest', sub: '124 million weights', tone: 's2', shape: 'ghost',
      count: `Inside: ${fmtInt(GPT2)} weights, ${bytes(GPT2)} as float32.`,
      summary: (ctx, r) => { ctx.fillStyle = theme.c('s2', 0.22); ctx.fillRect(r.x, r.y, r.w, r.h); },
      inner: G.inner, children: [Object.assign(labs, { rect: G.rect })], tools: back,
      about: [`GPT-2 (2019) in its smallest size has ${fmtInt(GPT2)} weights, about ${bytes(GPT2)} as float32. All the models in this site, yours included, fit in the little square in its corner: ${pct(all, GPT2)}% of it. Large language models today have from a few billion to hundreds of billions of weights.`, 'Zoom out once more, or zoom in to come back.'],
    };
    const U = squareLevel(size, GPT2, GPU_N, { tone: 's7', innerTone: 's2', grid: 16, label: ['GPT-2', `${pct(GPT2, GPU_N)}% of this memory`], words: [`${GPU_GB} GB of memory`, `${fmtInt(GPU_N)} float32 numbers`, `One GPU like this holds the smallest GPT-2 ${Math.floor(GPU_N / GPT2)} times over.`] });
    return {
      id: 'gpu', title: 'One GPU’s memory', sub: `${GPU_GB} GB: 6 billion float32 numbers`, tone: 's7',
      count: `Inside: ${fmtInt(GPU_N)} float32 numbers, ${fmtInt(GPU_N * 32)} bits, each a memory cell.`,
      inner: U.inner, children: [Object.assign(gpt2, { rect: U.rect })], tools: back,
      about: [`A GPU with ${GPU_GB} GB of memory holds ${fmtInt(GPU_N)} float32 numbers. The smallest GPT-2 fills ${pct(GPT2, GPU_N)}% of it, and needs more room again for its activations while it runs, and for gradients and optimizer state while it trains. A model with more weights than one GPU holds is split across many GPUs, or stored in fewer bits per weight: open a weight in the neuron view and try bfloat16 or FP8.`, 'This is as far as the zoom goes. Zoom in to come back to your network.'],
    };
  }

  function setup() {
    const V = NA.netView;
    SV.panel = V.stage.closest('.panel') || V.stage;
    if (getComputedStyle(SV.panel).position === 'static') SV.panel.style.position = 'relative';
    SV.stage = el('div', { class: 'deep-stage scale-stage', hidden: true });
    SV.host = el('div', { class: 'deep-host' });
    SV.stage.append(SV.host);
    SV.panel.append(SV.stage);
    SV.arch = new NA.ArchView(SV.host, {
      label: 'Your network beside bigger models: zoom in to come back',
      kicker: 'Zoomed out',
      build: (size) => build(size),
      onFocus: (node, path) => SV.info.set(SV.arch.infoFor(node, path)),
    });
    SV.arch.mergeBtn.hidden = true;
    SV.info = NA.labUI.infoPanel(SV.stage, { open: !(NA.mobile && NA.mobile.mobile) && SV.panel.clientHeight >= 640 });
    App.on('model', () => { if (SV.open) SV.close(); });
    App.register(SV);
  }
  /* The network diagram reports a zoom out past its smallest zoom. The first one only says what another will do;
     a second one, as a new gesture within a few seconds, opens this view (so a long scroll does not tunnel in). */
  SV.underflow = function () {
    const ND = NA.netDeep, now = performance.now();
    if (SV.open || (ND && (ND.open || ND.flying || now - (ND.closedAt || 0) < 700)) || now - (SV.closedAt || 0) < 700) return;
    const fresh = now - (SV.lastEvent || 0) > 260;
    SV.lastEvent = now;
    if (!fresh) return;
    if (SV.armed && now - SV.armed < 4000) { SV.armed = 0; SV.show(); return; }
    SV.armed = now;
    NA.ui.toast.show('Zoom out once more to step back past your network and see it beside bigger models.');
  };
  SV.show = function () {
    if (!SV.stage) setup();
    SV.open = true;
    const A = SV.arch;
    // start inside your network and fly out of it, onto the labs' models
    A.path = ['gpu', 'gpt2', 'labs', 'net']; A.selected = null; A.focusKey = null; A.crumbKey = null; A.dirty = true;
    A.zoomSet(1, 0, 0); A.wheelHold = true; A.lastWheel = performance.now();
    SV.stage.hidden = false;
    SV.flyOut = true;
    NA.tooltip.hide();
  };
  SV.close = function () {
    if (!SV.open) return;
    SV.open = false; SV.closedAt = performance.now();
    if (SV.arch.tour) SV.arch.endFollow();
    SV.stage.hidden = true;
    NA.netView.dirty = true;
  };
  SV.frame = function (now) {
    if (!SV.open) return;
    SV.arch.frame(now);
    if (SV.flyOut && SV.arch.root && !SV.arch.anim) { SV.flyOut = false; SV.arch.exit(); }
  };

  NA.scaleView = SV;
})(window.NA);
