/* net-deep.js — zooming far into a neuron of the network diagram opens it, in place, with the same deep levels as
   the labs' architecture views: its weighted sum, its PyTorch code, what the library does inside, single float
   operations bit by bit, the adders and gates, the transistors and the electrons. Zooming back out of the neuron
   (or the Network breadcrumb) returns to the diagram. */
(function (NA) {
  'use strict';
  const { el, fmtSci: fmt } = NA; // fmtSci: tiny values as 4.1×10⁻⁴, not 4.1e−4
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
      about: [`${name} of this network, for the probe input${App.ds.kind === 'plane' ? ` (${App.ds.featureLabels.map((f, i) => `${f} = ${fmt(tr.a[0][i], 2)}`).join(', ')})` : ''}. Zoom into an input bar to open the neuron that produced it, all the way back to the inputs.`,
        out ? `This is an output: its ${soft ? 'probability comes from the softmax of all the output sums, so raising one output lowers the others' : 'value is the answer itself'}, and training compares it with the right answer to work out how every weight before it should change.`
          : `What it is for: this neuron is one ingredient of the next layer, which adds it, weighted, to the other neurons of its layer. Its square in the network shows its output at every point of the input plane; its weights and bias set where its sum crosses 0, and ${act.label} sets the shape on either side. ${tr.a[l][j] === 0 && act.id === 'relu' ? 'For this input it is silent (its sum is negative), so it passes nothing on and its weights learn nothing from this example.' : 'For this input it passes on ' + fmt(tr.a[l][j], 3) + '.'}`],
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
      label: `Inside a neuron: ${NA.tapWord(true)} a part or zoom in to open it`,
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

  /* The neuron a zoom is aimed at: the one under the cursor (hovered) if any, else the one nearest the zoom's
     centre within reach; with anyVisible (the zoom is at its limit), simply the nearest one still on screen. */
  function aimed(anchor, anyVisible) {
    const V = NA.netView, z = V.zoom, h = App.hover;
    if (h && h.l !== undefined && h.j !== undefined) { const nd = V.layout.nodes.find((q) => q.l === h.l && q.j === h.j); if (nd) return nd; }
    const [wx, wy] = z.toWorld(...(anchor || [V.W / 2, V.H / 2]));
    let best = null, bd = Infinity;
    for (const nd of V.layout.nodes) {
      const sx = nd.x * z.k + z.tx, sy = nd.y * z.k + z.ty;
      if (anyVisible && (sx < 0 || sy < 0 || sx > V.W || sy > V.H)) continue;
      const d = Math.hypot(nd.x - wx, nd.y - wy);
      if (d < bd) { bd = d; best = nd; }
    }
    if (!best) return null;
    return anyVisible || bd <= Math.max(best.s / 2 + 14, best.s / 2 + (best.gap || 0) / 2) ? best : null;
  }
  // The network view calls this whenever its zoom changes: zoomed in on a neuron, open it. With a mouse that is
  // from 2.6×; on a touch screen only at the zoom limit, since a moderate pinch is how a phone reads small labels
  // (and a tapped neuron's card has an Open button).
  ND.maybeOpen = function () {
    const V = NA.netView, z = V.zoom, atLimit = z.k >= z.max - 0.01;
    if (ND.open || ND.flying || z.k < (V.touch ? z.max - 0.01 : 2.6) || !V.layout || !NA.ArchView) return;
    const nd = aimed(z.anchor, atLimit);
    if (nd) ND.show(nd.l, nd.j, nd.x * z.k + z.tx, nd.y * z.k + z.ty);
  };
  // A double-click (or double-tap) on a neuron flies into it. Returns whether there was a neuron there.
  ND.openAt = function (px, py) {
    const V = NA.netView, z = V.zoom, [wx, wy] = z.toWorld(px, py);
    const nd = V.layout && V.layout.nodes.find((q) => Math.hypot(q.x - wx, q.y - wy) <= q.s / 2 + 8);
    if (!nd || !NA.ArchView) return false;
    ND.flyTo(nd.l, nd.j);
    return true;
  };
  // Zoom the diagram's camera into a neuron, then open it (also used by the Open buttons).
  ND.flyTo = function (l, j) {
    const V = NA.netView, z = V.zoom, nd = V.layout && V.layout.nodes.find((q) => q.l === l && q.j === j);
    if (ND.open || ND.flying) return;
    if (!nd) { ND.show(l, j, V.W / 2, V.H / 2); return; }
    const done = () => { ND.flying = false; ND.show(l, j, nd.x * z.k + z.tx, nd.y * z.k + z.ty); };
    if (NA.prefersReducedMotion()) { done(); return; }
    const k0 = z.k, tx0 = z.tx, ty0 = z.ty, k1 = Math.max(k0, 3), tx1 = V.W / 2 - nd.x * k1, ty1 = V.H / 2 - nd.y * k1, t0 = performance.now();
    ND.flying = true;
    const step = (now) => {
      const p = Math.min(1, (now - t0) / 340), e = p < 0.5 ? 4 * p * p * p : 1 - Math.pow(-2 * p + 2, 3) / 2;
      z.set(k0 + (k1 - k0) * e, tx0 + (tx1 - tx0) * e, ty0 + (ty1 - ty0) * e);
      if (p < 1) requestAnimationFrame(step); else done();
    };
    requestAnimationFrame(step);
  };
  // From the inspector or the phone's card: show the diagram if it is hidden, then fly in.
  ND.openNeuron = function (l, j) {
    if (NA.mobile && NA.mobile.mobile && NA.mobile.screen !== 'network') { NA.mobile.go('network'); setTimeout(() => ND.flyTo(l, j), 260); }
    else ND.flyTo(l, j);
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
