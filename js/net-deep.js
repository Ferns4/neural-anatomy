/* net-deep.js — zooming far into a neuron of the network diagram opens it, in place, with the same deep levels as
   the labs' architecture views: its weighted sum, its PyTorch code, what the library does inside, single float
   operations bit by bit, the adders and gates, the transistors and the electrons. Zooming back out of the neuron
   (or the Network breadcrumb) returns to the diagram. */
(function (NA) {
  'use strict';
  const { el, fmt } = NA;
  const App = NA.app;
  const ND = { open: false };

  // neuron (l, j) of the current network, for the current probe input; its inputs open the neurons behind them
  function neuron(X, l, j) {
    const net = App.net, tr = App.probeTrace(), Dp = NA.deep, name = App.neuronName(l, j);
    if (l === 0) return Dp.stored(X, tr.a[0][j], `${name} (an input)`);
    const nIn = net.sizes[l - 1], W = net.W[l - 1], out = l === net.L, act = net.actOf(l, j), soft = out && net.out === 'softmax';
    const spec = {
      title: name, sub: soft ? 'weighted sum, then softmax' : act.id === 'linear' ? 'weighted sum' : `weighted sum, then ${act.label}`,
      terms: Array.from({ length: nIn }, (_, i) => ({ a: tr.a[l - 1][i], w: W[j * nIn + i] })), bias: net.b[l - 1][j],
      act: soft || act.id === 'linear' ? null : act, outName: out ? (net.out === 'identity' ? 'y_hat' : 'p') : 'a',
      inLabel: (i) => App.neuronName(l - 1, i, true), wLabel: (i) => `weight from ${App.neuronName(l - 1, i, true)}`, inWhat: l === 1 ? 'the input features' : 'the previous layer',
      inputNode: (i) => neuron(X, l - 1, i),
      vars: { w: `layer${l}.weight[${j}]`, a: l === 1 ? 'x' : `a${l - 1}`, b: `layer${l}.bias[${j}]` },
      codeHead: [`# ${name}, for the probe input`], codeWhat: `as one row of nn.Linear(${nIn}, ${net.sizes[l]})`,
      about: [`${name} of this network, for the probe input${App.ds.kind === 'plane' ? ` (${App.ds.featureLabels.map((f, i) => `${f} = ${fmt(tr.a[0][i], 2)}`).join(', ')})` : ''}. Zoom into an input bar to open the neuron that produced it, all the way back to the inputs.`],
    };
    if (soft) {
      spec.actLine = `p = softmax(z)[${j}] = ${fmt(tr.a[l][j], 4)}   (over all ${net.sizes[l]} outputs)`;
      spec.actNode = () => Object.assign(Dp.lib(X, 'softmax', { scores: Array.from(tr.z[l]), j, label: 'p' }), { title: 'Softmax', sub: `p for ${App.neuronName(l, j, true)}` });
    }
    return Dp.neuron(X, spec);
  }

  function build() {
    const V = NA.netView;
    // the opened neuron covers the whole network panel, toolbar and legend included
    ND.panel = V.stage.closest('.panel') || V.stage;
    if (getComputedStyle(ND.panel).position === 'static') ND.panel.style.position = 'relative';
    ND.stage = el('div', { class: 'deep-stage', hidden: true });
    ND.host = el('div', { class: 'deep-host' });
    ND.stage.append(ND.host);
    ND.panel.append(ND.stage);
    ND.arch = new NA.ArchView(ND.host, {
      label: 'Inside a neuron: click a part or zoom in to open it',
      kicker: 'Neuron',
      build: (size) => Object.assign(neuron({ wide: size.w / size.h >= 1.05 }, ND.l, ND.j), { id: 'n' }),
      onFocus: (node, path) => ND.info.set(ND.arch.infoFor(node, path)),
      outer: { title: 'Network', exit: () => ND.close() },
    });
    ND.info = NA.labUI.infoPanel(ND.stage, { open: !(NA.mobile && NA.mobile.mobile) && ND.panel.clientHeight >= 640 });
    ND.stage.addEventListener('transitionend', () => { if (ND.closing) { ND.closing = false; ND.stage.hidden = true; ND.stage.classList.remove('is-closing'); } });
    const mark = () => { if (ND.open) ND.stale = true; };
    ['weights', 'probe', 'model', 'arch'].forEach((t) => App.on(t, mark));
    App.on('model', () => { if (ND.open) ND.close(true); });
    App.register(ND);
  }

  // The network view calls this whenever its zoom changes: far enough into a neuron, open it.
  ND.maybeOpen = function () {
    const V = NA.netView, z = V.zoom;
    if (ND.open || z.k < 3.4 || !z.anchor || !V.layout || !NA.ArchView) return;
    const [wx, wy] = z.toWorld(z.anchor[0], z.anchor[1]);
    let best = null, bd = Infinity;
    for (const nd of V.layout.nodes) { const d = Math.hypot(nd.x - wx, nd.y - wy); if (d < bd) { bd = d; best = nd; } }
    if (!best || bd > best.s / 2 + 14) return;
    ND.show(best.l, best.j, best.x * z.k + z.tx, best.y * z.k + z.ty);
  };
  ND.show = function (l, j, sx, sy) {
    if (!ND.stage) build();
    ND.l = l; ND.j = j; ND.open = true; ND.anchor = [sx, sy];
    ND.arch.path = []; ND.arch.selected = null; ND.arch.focusKey = null; ND.arch.crumbKey = null; ND.arch.dirty = true;
    ND.arch.zoomSet(1, 0, 0); ND.arch.wheelHold = true; ND.arch.lastWheel = performance.now();
    const s = ND.stage, a = NA.netView.stage.getBoundingClientRect(), p = ND.panel.getBoundingClientRect();
    s.style.transformOrigin = `${sx + a.left - p.left}px ${sy + a.top - p.top}px`; // grow out of the neuron
    s.hidden = false; s.classList.add('is-closing');
    void s.offsetWidth; // start from the neuron's spot, then grow to fill the stage
    s.classList.remove('is-closing');
    NA.tooltip.hide();
    if (App.selected === null || !App.selected || App.selected.l !== l || App.selected.j !== j) App.select({ l, j }, { fromGraph: true });
  };
  ND.close = function (now) {
    if (!ND.open) return;
    ND.open = false;
    const V = NA.netView, z = V.zoom, [sx, sy] = ND.anchor || [V.W / 2, V.H / 2];
    // land back on the diagram zoomed out a little, so the neuron does not reopen at once
    z.zoomAt(sx, sy, 0.6);
    if (now || NA.prefersReducedMotion()) { ND.stage.hidden = true; return; }
    ND.closing = true; ND.stage.classList.add('is-closing');
  };
  ND.frame = function (now) {
    if (!ND.open) return;
    if (ND.stale && (!App.training || now - (ND.lastStale || 0) > 400)) { ND.lastStale = now; ND.stale = false; ND.arch.dirty = true; if (!App.training) ND.arch.refreshFocus(); }
    ND.arch.frame(now);
  };

  NA.netDeep = ND;
})(window.NA);
