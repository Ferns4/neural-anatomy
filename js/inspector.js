/* inspector.js — the right-hand panel: model notes and diagnostics, neuron, layer and training views. */
(function (NA) {
  'use strict';
  const { theme, clamp, fmtSci: fmt, fmtPct, fmtInt, el, rr } = NA; // fmtSci: tiny values as 4.1×10⁻⁴, not 4.1e−4
  const App = NA.app;
  const C = NA.charts;

  const IN = { tab: 'insights', dirty: true, built: '', lastRender: 0, layer: 1 };

  IN.init = function () {
    IN.tabsEl = document.getElementById('inspTabs');
    IN.body = document.getElementById('inspBody');
    IN.tabsEl.addEventListener('click', (e) => { const b = e.target.closest('[data-tab]'); if (b) IN.setTab(b.dataset.tab); });
    IN.tabsEl.addEventListener('keydown', (e) => {
      if (e.key !== 'ArrowRight' && e.key !== 'ArrowLeft') return;
      const tabs = NA.$$('[data-tab]', IN.tabsEl), i = tabs.findIndex((t) => t.dataset.tab === IN.tab);
      const next = tabs[(i + (e.key === 'ArrowRight' ? 1 : tabs.length - 1)) % tabs.length];
      IN.setTab(next.dataset.tab); next.focus();
    });
    const mark = () => { IN.dirty = true; };
    ['weights', 'probe', 'history', 'ablation', 'training', 'hyper', 'overlay'].forEach((t) => App.on(t, mark));
    App.on('model', () => { IN.built = ''; IN.layer = clamp(IN.layer, 1, App.net.L); mark(); });
    App.on('arch', () => { IN.built = ''; mark(); });
    App.on('test', () => { IN.built = ''; mark(); });
    App.on('selection', (o) => {
      if (App.selected && App.selected.l > 0) IN.layer = App.selected.l;
      if (o && o.fromGraph && IN.tab !== 'backprop') IN.setTab('neuron');
      IN.built = ''; mark();
    });
    App.on('preset', (p) => IN.setTab(p.tab || 'insights'));
    App.on('tab', (t) => IN.setTab(t));
    theme.onChange(() => { IN.built = ''; mark(); });
    let lastW = 0;
    new ResizeObserver(() => {
      const w = IN.body.clientWidth;
      if (Math.abs(w - lastW) < 2) return;
      lastW = w;
      requestAnimationFrame(() => { IN.built = ''; mark(); });
    }).observe(IN.body);
    App.register(IN);
  };

  IN.setTab = function (t) {
    if (IN.tab === 'backprop' && t !== 'backprop') NA.walk.exit();
    IN.tab = t;
    NA.$$('[data-tab]', IN.tabsEl).forEach((b) => { const on = b.dataset.tab === t; b.setAttribute('aria-selected', String(on)); b.tabIndex = on ? 0 : -1; });
    IN.built = ''; IN.dirty = true;
    IN.body.scrollTop = 0;
    if (t === 'backprop') NA.walk.enter();
  };

  IN.frame = function (now) {
    if (!IN.dirty || !App.net || !IN.body.offsetParent) return; // a hidden screen renders when shown
    if (App.training && now - IN.lastRender < 380) return;
    IN.lastRender = now; IN.dirty = false;
    const T = TABS[IN.tab];
    const key = IN.tab + '|' + T.key();
    if (IN.built !== key) {
      NA.clear(IN.body);
      T.build(IN.body);
      IN.built = key;
    }
    T.update();
  };

  const section = (title, ...kids) => el('section', { class: 'insp-sec' }, el('h3', { class: 'sec-title', text: title }), ...kids);
  const kv = (k, v, note) => el('div', { class: 'kv' }, el('span', { class: 'kv-k', text: k }), el('span', { class: 'kv-v', text: v }), note ? el('span', { class: 'kv-note', text: note }) : null);
  const pill = (level, text) => el('span', { class: 'pill pill-' + level }, el('i', { class: 'pill-ico', 'aria-hidden': 'true' }), text);

  // ==================================================================== Insights
  const Insights = {
    key: () => (App.preset ? App.preset.id : ''),
    build(root) {
      const p = App.preset;
      this.diagEl = el('div', { class: 'diag-list' });
      this.sig = '';
      this.changedEl = el('div', { class: 'changed-note', hidden: true },
        el('span', { text: 'You have changed this model since loading it, so these notes may no longer match.' }),
        el('button', { class: 'btn btn-quiet', type: 'button', onclick: () => App.loadPreset(p.id) }, 'Reload preset'));
      const card = el('article', { class: 'model-card' },
        el('div', { class: 'eyebrow', text: `${p.group} · preset` }),
        el('h2', { class: 'mc-title', text: p.name }),
        el('p', { class: 'mc-tag', text: p.tagline }),
        el('p', { class: 'mc-about', text: p.about }),
        this.changedEl,
        el('h3', { class: 'sec-title', text: 'What to look for' }),
        el('ul', { class: 'look-list' }, ...p.look.map((t) => el('li', { text: t }))),
        el('h3', { class: 'sec-title', text: 'Experiments' }),
        NA.labUI.tryList(p.try, (t) => App.runTry(t)));
      root.append(card, section('Diagnostics', el('p', { class: 'sec-note', text: 'Read automatically from the network’s statistics and training history. They update as it trains.' }), this.diagEl));
    },
    update() {
      const p = App.preset;
      // compare layer sizes and every neuron's activation, plus the loss, against the preset
      const archSig = (a) => JSON.stringify([a.hidden, a.hidden.map((n, h) => NA.range(n).map((j) => ((a.acts && a.acts[h]) || [])[j] || a.act))]);
      const changed = archSig(App.arch) !== archSig(p.arch) || App.net.lossId !== (p.hyper.loss || NA.engine.LOSS_FOR[App.net.out][0]) ||
        App.spec.dataset !== p.data.dataset || (App.spec.features || []).join() !== (p.data.features || App.spec.features || []).join();
      this.changedEl.hidden = !changed;
      const list = NA.insights.diagnose();
      const sig = list.map((d) => d.level + d.title + d.body).join('|');
      if (sig === this.sig) return;
      this.sig = sig;
      const items = list.map((d) => el('div', { class: 'diag diag-' + d.level },
        el('div', { class: 'diag-head' }, pill(d.level, { good: 'Good', info: 'Note', warn: 'Warning', critical: 'Problem' }[d.level]), el('h4', { class: 'diag-title', text: d.title })),
        el('p', { class: 'diag-body', text: d.body }),
        d.actions.length ? el('div', { class: 'diag-actions' }, ...d.actions.map((a) => el('button', { class: 'btn btn-quiet', type: 'button', onclick: () => NA.insights.runAction(a) }, a.label))) : null));
      if (!items.length) items.push(el('p', { class: 'empty-note', text: 'Nothing unusual right now.' }));
      NA.clear(this.diagEl).append(...items);
    },
  };

  // ==================================================================== Neuron
  const Neuron = {
    key: () => { const s = App.selected; return s ? `${s.l}:${s.j}:${App.net.sizes.join('-')}:${App.ds.id}` : 'none'; },
    build(root) {
      const s = App.selected, net = App.net, L = net.L, ds = App.ds;
      if (!s) { root.append(el('p', { class: 'empty-note', text: `${NA.tapWord()} any neuron in the network to inspect it.` })); this.none = true; return; }
      this.none = false;
      const { l, j } = s;
      const kind = l === 0 ? 'Input' : l === L ? 'Output' : `Hidden ${l}`;
      const fn = l === 0 ? 'raw input' : l === L ? { sigmoid: 'sigmoid', softmax: 'softmax', identity: 'linear' }[net.out] : net.actOf(l, j).label;
      const nav = (d) => () => { const n = net.sizes[l]; App.select({ l, j: (j + d + n) % n }); };
      this.ablateBtn = l < L ? el('button', { class: 'btn', type: 'button', 'aria-pressed': 'false', onclick: () => App.toggleAblate(l, j) }, 'Ablate') : null;
      root.append(el('header', { class: 'n-head' },
        el('div', null,
          el('div', { class: 'eyebrow', text: `${kind} · ${fn}` }),
          el('h2', { class: 'n-title', text: l === 0 ? App.neuronName(0, j) : l === L ? (ds.task === 'binary' ? 'Output neuron' : ds.task === 'regression' ? 'Output ŷ' : App.neuronName(l, j)) : `Neuron ${j + 1}` })),
        el('div', { class: 'n-nav' },
          el('button', { class: 'icon-btn', type: 'button', 'aria-label': 'Previous neuron in this layer', onclick: nav(-1) }, '‹'),
          el('button', { class: 'icon-btn', type: 'button', 'aria-label': 'Next neuron in this layer', onclick: nav(1) }, '›'))));
      const openBtn = el('button', { class: 'btn btn-primary', type: 'button', title: 'Open this neuron: its sum, code, arithmetic and circuits, down to the electrons', onclick: () => NA.netDeep && NA.netDeep.openNeuron(l, j) }, 'Open neuron ⤢');
      root.append(el('div', { class: 'btn-row' }, openBtn, this.ablateBtn, this.ablateBtn ? el('span', { class: 'sec-note', text: 'Ablating forces this neuron to output 0 so you can see what breaks.' }) : null));

      this.eqEl = el('div', { class: 'eq' });
      if (l > 0) root.append(section('Computation for the probe input', this.eqEl));
      else root.append(section('Value for the probe input', this.eqEl));

      this.actCanvas = null; this.fxApplied = null;
      const plotLegend = () => el('div', { class: 'mini-legend' }, el('span', { class: 'lg-item' }, el('i', { class: 'key-line', style: { background: 'var(--ink)' } }), 'f(z)'), el('span', { class: 'lg-item' }, el('i', { class: 'key-line', style: { background: 'var(--muted)' } }), 'slope f′(z)'), el('span', { class: 'lg-item' }, el('i', { class: 'key-box' }), 'z over training data'), el('span', { class: 'lg-item' }, el('i', { class: 'dot', style: { background: 'var(--pos)' } }), 'probe'));
      if (l > 0 && l < L) {
        // this neuron's own activation function: pick it, read its formula, see where the probe sits
        const act = net.actOf(l, j);
        this.act = act;
        this.actCanvas = el('canvas');
        this.actNote = el('p', { class: 'sec-note' });
        this.fxApplied = el('div', { class: 'fx-line fx-strong' });
        const chip = el('button', { class: 'act-pick', type: 'button', 'aria-haspopup': 'listbox', 'aria-label': `Activation function: ${act.label}. Change for this neuron` },
          NA.ui.actGlyph(act.id, 34, 22), el('span', { class: 'act-pick-text' }, el('span', { class: 'act-pick-name', text: act.label }), el('span', { class: 'act-pick-hint', text: 'Change for this neuron' })), el('span', { class: 'lh-chip-caret', 'aria-hidden': 'true', text: '▾' }));
        chip.addEventListener('click', () => NA.openNeuronMenu(l, j, chip));
        root.append(section('Activation function', chip,
          el('div', { class: 'fx' }, el('div', { class: 'fx-line', text: `f(z) = ${act.formula}` }), el('div', { class: 'fx-line fx-muted', text: `f′(z) = ${act.dformula}` }), this.fxApplied, act.note ? el('div', { class: 'fx-note', text: act.note }) : null),
          el('div', { class: 'chart-box' }, this.actCanvas), plotLegend(), this.actNote));
      } else if (l === L) {
        const Ls = NA.engine.LOSSES[net.lossId], outName = { sigmoid: 'Sigmoid, a probability', softmax: 'Softmax across the outputs', identity: 'Linear (no squashing)' }[net.out];
        if (net.out !== 'softmax') {
          this.act = net.actOf(l, j);
          this.actCanvas = el('canvas');
          this.actNote = el('p', { class: 'sec-note' });
        }
        const lossBtn = el('button', { class: 'act-pick', type: 'button', 'aria-haspopup': 'listbox', 'aria-label': `Loss function: ${Ls.label}. Change` },
          el('span', { class: 'act-pick-text' }, el('span', { class: 'act-pick-name', text: Ls.label }), el('span', { class: 'act-pick-hint', text: 'Change the loss function' })), el('span', { class: 'lh-chip-caret', 'aria-hidden': 'true', text: '▾' }));
        lossBtn.addEventListener('click', () => NA.openLossMenu(lossBtn));
        this.fxApplied = el('div', { class: 'fx-line fx-strong' });
        root.append(section(`Output function: ${outName}`, el('div', { class: 'fx' }, this.fxApplied),
          this.actCanvas ? el('div', { class: 'chart-box' }, this.actCanvas) : null, this.actCanvas ? plotLegend() : null, this.actCanvas ? this.actNote : null));
        root.append(section('Loss function', lossBtn,
          el('div', { class: 'fx' }, el('div', { class: 'fx-line', text: Ls.formula[net.out] }), el('div', { class: 'fx-line fx-muted', text: Ls.delta[net.out] + '   (error signal sent back)' })),
          el('p', { class: 'sec-note', text: Ls.hint })));
      }
      this.respCanvas = el('canvas', { class: 'resp-canvas' });
      this.respExtra = el('div', { class: 'resp-extra' });
      const respTitle = ds.kind === 'plane' ? 'Response over the input plane' : ds.kind === 'line' ? 'Response as x varies' : l === 1 ? 'Weight template' : l === 0 ? 'Where this pixel sends its signal' : 'Validation images that excite it most';
      this.respNote = el('p', { class: 'sec-note' });
      root.append(section(respTitle, el('div', { class: 'chart-box' }, this.respCanvas), this.respExtra, this.respNote));

      this.sliders = [];
      if (l > 0 && net.sizes[l - 1] <= 16) {
        const box = el('div', { class: 'w-edit' });
        const nIn = net.sizes[l - 1];
        const mk = (label, get, set, id) => {
          const R = Math.max(2, Math.ceil(Math.abs(get()) * 1.5));
          const input = el('input', { type: 'range', id, min: -R, max: R, step: 0.01, value: get() });
          const out = el('output', { class: 'mono', for: id, text: fmt(get(), 2) });
          input.addEventListener('input', () => { set(+input.value); out.textContent = fmt(+input.value, 2); });
          this.sliders.push({ input, out, get });
          return el('label', { class: 'w-row', for: id }, el('span', { class: 'w-name', text: label }), input, out);
        };
        for (let i = 0; i < nIn; i++) box.appendChild(mk(App.neuronName(l - 1, i, true), () => net.W[l - 1][j * nIn + i], (v) => App.setWeight(l, j, i, v), `w-${l}-${j}-${i}`));
        box.appendChild(mk('bias b', () => net.b[l - 1][j], (v) => App.setBias(l, j, v), `b-${l}-${j}`));
        root.append(section('Incoming weights', el('p', { class: 'sec-note', text: 'Drag to change a weight by hand. Training pauses while you edit; Reset restores the initial weights.' }), box));
      }
      this.statsEl = el('div', { class: 'kv-grid' });
      root.append(section(l === 0 ? 'Statistics over the training set' : 'Statistics over the training set', this.statsEl));
    },
    update() {
      if (this.none) return;
      const s = App.selected, net = App.net, L = net.L, ds = App.ds, { l, j } = s;
      const tr = App.probeTrace();
      if (this.ablateBtn) { const off = !net.mask[l][j]; this.ablateBtn.setAttribute('aria-pressed', String(off)); this.ablateBtn.textContent = off ? 'Restore neuron' : 'Ablate'; }
      // ---- equation
      if (l === 0) {
        const outW = [], n1 = net.sizes[1];
        for (let k = 0; k < n1; k++) outW.push({ name: App.neuronName(1, k, true), w: net.W[0][k * net.sizes[0] + j] });
        NA.clear(this.eqEl).append(el('div', { class: 'eq-line' }, el('span', { class: 'mono eq-big', text: `${App.neuronName(0, j)} = ${fmt(tr.a[0][j], 3)}` })),
          el('p', { class: 'sec-note', text: `Feeds ${n1} neuron${n1 > 1 ? 's' : ''} in ${App.layerName(1)}. Its largest outgoing weights:` }),
          contribTable(outW.sort((a, b) => Math.abs(b.w) - Math.abs(a.w)).slice(0, 8).map((o) => ({ name: o.name, w: o.w, a: tr.a[0][j], c: o.w * tr.a[0][j] })), null, null, true));
      } else {
        const nIn = net.sizes[l - 1], rows = [];
        for (let i = 0; i < nIn; i++) { const w = net.W[l - 1][j * nIn + i], a = tr.a[l - 1][i]; rows.push({ name: App.neuronName(l - 1, i, true), w, a, c: w * a }); }
        rows.sort((a, b) => Math.abs(b.c) - Math.abs(a.c));
        const z = tr.z[l][j], b = net.b[l - 1][j];
        const fnText = NA.actLines(l, j, tr)[1] + (l < L && !net.mask[l][j] ? '  (ablated → 0)' : '');
        if (this.fxApplied) this.fxApplied.textContent = fnText;
        const lossLine = l === L && tr.delta ? el('p', { class: 'eq-line mono', text: `loss on this sample = ${fmt(tr.loss, 4)}   ·   δ = ∂L/∂z = ${fmt(tr.delta[l][j], 4)}` }) : null;
        NA.clear(this.eqEl).append(...[
          el('p', { class: 'eq-line mono', text: `z = Σᵢ wᵢ·aᵢ + b  over ${nIn} input${nIn > 1 ? 's' : ''}` }),
          contribTable(rows, b, z),
          el('p', { class: 'eq-line mono eq-big', text: fnText }), lossLine].filter(Boolean));
      }
      // ---- activation plot
      if (this.actCanvas) {
        const F = App.dataPass(), N = F.n, idx = App.evalTrain, zs = new Float32Array(idx.length);
        for (let q = 0; q < idx.length; q++) zs[q] = F.z[l][j * N + idx[q]];
        C.activationPlot(this.actCanvas, this.act, zs, tr.z[l][j]);
        let flat = 0, off = 0;
        for (const z of zs) { const d = Math.abs(this.act.d(z, this.act.f(z))); if (d < 0.1) flat++; if (z <= 0) off++; }
        const sorted = Array.from(zs).sort((a, b) => a - b), act = this.act;
        let note = `Over the training data z ranges from ${fmt(sorted[0], 2)} to ${fmt(sorted[sorted.length - 1], 2)} (median ${fmt(NA.quantile(sorted, 0.5), 2)}). `;
        if (act.id === 'relu') note += `It is switched off (z ≤ 0) for ${fmtPct(off / zs.length)} of inputs, where no gradient flows back through it.`;
        else if (act.id === 'leaky') note += `It is on its shallow side (z ≤ 0) for ${fmtPct(off / zs.length)} of inputs, still passing 5% of the gradient there.`;
        else if (act.id === 'linear') note += 'A linear unit passes the gradient through unchanged.';
        else if (act.id === 'sine') note += 'Sine is periodic, so very different z values can give the same output; its slope cos(z) changes sign every π.';
        else note += `For ${fmtPct(flat / zs.length)} of inputs it sits where the slope |f′(z)| is below 0.1 and learns slowly.`;
        this.actNote.textContent = note;
      }
      // ---- response
      drawResponse(this, l, j);
      // ---- sliders
      for (const sl of this.sliders) if (document.activeElement !== sl.input) { const v = sl.get(); if (Math.abs(+sl.input.value - v) > 1e-3) { sl.input.max = Math.max(+sl.input.max, Math.ceil(Math.abs(v))); sl.input.min = -sl.input.max; sl.input.value = v; } sl.out.textContent = fmt(v, 2); }
      // ---- stats
      const st = (l === 0 ? App.inputStats() : App.stats()[l])[j], items = [];
      items.push(kv('Mean output', fmt(st.mean, 3)), kv('Spread (std)', fmt(st.std, 3)));
      if (l > 0 && l < L) {
        const act = net.actOf(l, j);
        if (act.family === 'rectifier') items.push(kv('Active', fmtPct(st.activeFrac), 'share of inputs with z > 0'));
        if (act.sat) items.push(kv('Saturated', fmtPct(st.satFrac), 'share on the flat tails'));
      }
      if (l > 0 && App.displayStep() > 0) { const g = App.gradReport(); items.push(kv('RMS error signal δ', fmt(g.dRms[l][j], 4), 'how strongly the loss responds to it')); }
      if (l < L && !App.training && App.displayStep() > 0) {
        const r = App.importance().layers[l][j];
        items.push(kv('If ablated', r.off ? 'already off' : `${fmtSigned(r.dLoss)} validation loss`, r.off ? null : ds.task !== 'regression' ? `${fmtSigned(r.dAcc * 100, 1)} pts accuracy` : `${fmtSigned(r.dAcc, 3)} R²`));
      }
      NA.clear(this.statsEl).append(...items);
    },
  };
  const fmtSigned = (v, d = 3) => NA.fmtSignedSci(v, d);

  function contribTable(rows, bias, z, outgoing) {
    const MAX = 8;
    const shown = rows.slice(0, MAX), rest = rows.slice(MAX);
    let m = 1e-9;
    for (const r of shown) m = Math.max(m, Math.abs(r.c));
    if (bias !== null && bias !== undefined) m = Math.max(m, Math.abs(bias));
    const bar = (v) => {
      const w = (Math.min(1, Math.abs(v) / m) * 50).toFixed(1) + '%';
      return el('span', { class: 'cbar' }, el('span', { class: 'cbar-fill ' + (v < 0 ? 'neg' : 'pos'), style: v < 0 ? { right: '50%', width: w } : { left: '50%', width: w } }));
    };
    const head = el('div', { class: 'ct-row ct-head' }, el('span', { text: outgoing ? 'to' : 'from' }), el('span', { text: 'w' }), el('span', { text: outgoing ? '× x' : '× a' }), el('span', { text: '= w·a' }), el('span', { 'aria-hidden': 'true' }));
    const body = shown.map((r) => el('div', { class: 'ct-row' }, el('span', { class: 'ct-name', text: r.name }), el('span', { class: 'mono', text: fmt(r.w, 3) }), el('span', { class: 'mono', text: fmt(r.a, 3) }), el('span', { class: 'mono', text: fmt(r.c, 3) }), bar(r.c)));
    if (rest.length) {
      const sum = rest.reduce((s, r) => s + r.c, 0);
      body.push(el('div', { class: 'ct-row ct-more' }, el('span', { class: 'ct-name', text: `+ ${rest.length} more` }), el('span'), el('span'), el('span', { class: 'mono', text: fmt(sum, 3) }), bar(sum)));
    }
    if (bias !== null && bias !== undefined) body.push(el('div', { class: 'ct-row ct-bias' }, el('span', { class: 'ct-name', text: 'bias b' }), el('span'), el('span'), el('span', { class: 'mono', text: fmt(bias, 3) }), bar(bias)));
    if (z !== null && z !== undefined) body.push(el('div', { class: 'ct-row ct-total' }, el('span', { class: 'ct-name', text: 'z (sum)' }), el('span'), el('span'), el('span', { class: 'mono', text: fmt(z, 3) }), bar(z)));
    return el('div', { class: 'ct', role: 'table' }, head, ...body);
  }

  function drawResponse(T, l, j) {
    const ds = App.ds, net = App.net, cv = T.respCanvas;
    const W = Math.min(300, Math.max(160, cv.parentElement.clientWidth));
    NA.clear(T.respExtra);
    if (ds.kind === 'plane') {
      const ctx = NA.fitCanvas(cv, W, W), g = App.grid(), G = g.G, n = g.n, A = g.F.a[l], off = j * n;
      const off2 = document.createElement('canvas'); off2.width = G; off2.height = G;
      const oc = off2.getContext('2d'), im = oc.createImageData(G, G), lut = theme.divSoft;
      const isOutMulti = l === net.L && ds.task === 'multiclass';
      let m = 1e-9; for (let p = 0; p < n; p++) m = Math.max(m, Math.abs(A[off + p]));
      for (let p = 0; p < n; p++) {
        const t = l === net.L ? 2 * A[off + p] - 1 : A[off + p] / m;
        const k = Math.round((clamp(t, -1, 1) + 1) * 128) * 3;
        im.data[4 * p] = lut[k]; im.data[4 * p + 1] = lut[k + 1]; im.data[4 * p + 2] = lut[k + 2]; im.data[4 * p + 3] = 255;
      }
      oc.putImageData(im, 0, 0);
      ctx.save(); rr(ctx, 0, 0, W, W, 8); ctx.clip();
      ctx.imageSmoothingEnabled = true; ctx.drawImage(off2, 0, 0, W, W);
      if (l > 0) {
        const segs = NA.engine.contour(l === net.L && !isOutMulti ? A.subarray(off, off + n) : g.F.z[l].subarray(off, off + n), G, G, l === net.L && !isOutMulti ? 0.5 : 0), cell = W / G;
        ctx.strokeStyle = theme.c('ink', 0.85); ctx.lineWidth = 1.6; ctx.beginPath();
        for (let i = 0; i < segs.length; i += 4) { ctx.moveTo((segs[i] + 0.5) * cell, (segs[i + 1] + 0.5) * cell); ctx.lineTo((segs[i + 2] + 0.5) * cell, (segs[i + 3] + 0.5) * cell); }
        ctx.stroke();
      }
      for (let k = 0; k < ds.n; k++) {
        const x = ((ds.raw[2 * k] + 1) / 2) * W, y = ((1 - ds.raw[2 * k + 1]) / 2) * W;
        ctx.beginPath(); ctx.arc(x, y, 2.4, 0, Math.PI * 2); ctx.fillStyle = theme.c('panel', 0.7); ctx.fill();
        ctx.beginPath(); ctx.arc(x, y, 1.6, 0, Math.PI * 2); ctx.fillStyle = theme.classColor(ds.y[k]); ctx.fill();
      }
      const p = App.probe;
      ctx.beginPath(); ctx.arc(((p.raw[0] + 1) / 2) * W, ((1 - p.raw[1]) / 2) * W, 5, 0, Math.PI * 2); ctx.lineWidth = 2; ctx.strokeStyle = theme.c('ink'); ctx.stroke();
      ctx.restore();
      ctx.strokeStyle = theme.c('line-2'); ctx.lineWidth = 1; rr(ctx, 0.5, 0.5, W - 1, W - 1, 8); ctx.stroke();
      T.respNote.textContent = l === 0 ? 'This input’s value at every point of the plane.' : l === net.L ? (isOutMulti ? `Probability of class ${j} at every point. Blue is high, orange low.` : 'The output probability everywhere; the line is the 0.5 decision boundary.') : `Blue where the neuron is strongly positive, orange where negative, scaled to this neuron’s range (max |a| = ${fmt(m, 2)}). The line marks z = 0, where it switches.`;
    } else if (ds.kind === 'line') {
      const H = Math.round(W * 0.6), ctx = NA.fitCanvas(cv, W, H), g = App.grid(), n = g.n, A = g.F.a[l], Z = l > 0 ? g.F.z[l] : null;
      let lo = 0, hi = 0;
      for (let p = 0; p < n; p++) { lo = Math.min(lo, A[j * n + p]); hi = Math.max(hi, A[j * n + p]); if (Z) { lo = Math.min(lo, Z[j * n + p]); hi = Math.max(hi, Z[j * n + p]); } }
      const span = hi - lo || 1, pad = 10, X = (x) => pad + ((x + 1) / 2) * (W - 2 * pad), Y = (v) => H - pad - ((v - lo) / span) * (H - 2 * pad);
      ctx.clearRect(0, 0, W, H); ctx.fillStyle = theme.c('panel-2'); rr(ctx, 0, 0, W, H, 8); ctx.fill();
      ctx.strokeStyle = theme.c('line-2'); ctx.beginPath(); ctx.moveTo(pad, Y(0)); ctx.lineTo(W - pad, Y(0)); ctx.stroke();
      if (Z) { ctx.strokeStyle = theme.c('muted'); ctx.lineWidth = 1.2; ctx.beginPath(); for (let p = 0; p < n; p++) { p ? ctx.lineTo(X(g.raw[p]), Y(Z[j * n + p])) : ctx.moveTo(X(g.raw[p]), Y(Z[j * n + p])); } ctx.stroke(); }
      ctx.strokeStyle = theme.c('pos'); ctx.lineWidth = 2.2; ctx.beginPath(); for (let p = 0; p < n; p++) { p ? ctx.lineTo(X(g.raw[p]), Y(A[j * n + p])) : ctx.moveTo(X(g.raw[p]), Y(A[j * n + p])); } ctx.stroke();
      const px = X(App.probe.raw[0]); ctx.strokeStyle = theme.c('ink', 0.4); ctx.lineWidth = 1; ctx.beginPath(); ctx.moveTo(px, pad); ctx.lineTo(px, H - pad); ctx.stroke();
      T.respExtra.append(el('div', { class: 'mini-legend' }, el('span', { class: 'lg-item' }, el('i', { class: 'key-line', style: { background: 'var(--pos)' } }), 'output a(x)'), Z ? el('span', { class: 'lg-item' }, el('i', { class: 'key-line', style: { background: 'var(--muted)' } }), 'pre-activation z(x)') : null));
      T.respNote.textContent = l === 1 && l < net.L && net.actOf(l, j).family === 'rectifier' ? 'z(x) is a straight line in the first hidden layer; the rectifier bends it at z = 0, producing one kink.' : '';
    } else {
      // digits
      const G = ds.imgSize;
      if (l === 1) {
        const ctx = NA.fitCanvas(cv, Math.min(W, 196), Math.min(W, 196)), S = Math.min(W, 196), cell = S / G, nIn = net.sizes[0], Wt = net.W[0];
        let m = 1e-9; for (let i = 0; i < nIn; i++) m = Math.max(m, Math.abs(Wt[j * nIn + i]));
        for (let i = 0; i < nIn; i++) { ctx.fillStyle = theme.divColor(Wt[j * nIn + i] / m); ctx.fillRect((i % G) * cell, Math.floor(i / G) * cell, cell + 0.5, cell + 0.5); }
        T.respNote.textContent = 'Each pixel shows one incoming weight. Ink on blue pixels raises this neuron’s activation; ink on orange pixels lowers it.';
      } else if (l === 0) {
        const ctx = NA.fitCanvas(cv, 10, 10); ctx.clearRect(0, 0, 10, 10);
        T.respNote.textContent = 'See the outgoing weights above.';
      } else {
        const ctx = NA.fitCanvas(cv, 10, 10); ctx.clearRect(0, 0, 10, 10);
        T.respNote.textContent = l === net.L ? `The validation images with the highest probability for digit ${j}.` : 'The validation images that drive this neuron highest.';
      }
      if (l >= 1) {
        const F = App.dataPass(), N = F.n, A = F.a[l];
        const top = Array.from(ds.valIdx).sort((p, q) => A[j * N + q] - A[j * N + p]).slice(0, 10);
        const grid = el('div', { class: 'top-imgs' });
        for (const k of top) {
          const c = document.createElement('canvas'); c.width = G; c.height = G;
          const cx = c.getContext('2d'), im = cx.createImageData(G, G), ink = theme.tokens.ink, base = theme.tokens['panel-2'];
          for (let i = 0; i < G * G; i++) { const t = clamp(ds.raw[k * G * G + i], 0, 1); im.data[4 * i] = base[0] + (ink[0] - base[0]) * t; im.data[4 * i + 1] = base[1] + (ink[1] - base[1]) * t; im.data[4 * i + 2] = base[2] + (ink[2] - base[2]) * t; im.data[4 * i + 3] = 255; }
          cx.putImageData(im, 0, 0);
          grid.appendChild(el('button', { class: 'g-item', type: 'button', title: `Digit ${ds.y[k]}, activation ${fmt(A[j * N + k], 2)}`, onclick: () => App.setProbeSample(k) }, c, el('span', { class: 'g-lab', text: fmt(A[j * N + k], 1) })));
        }
        T.respExtra.append(el('div', { class: 'eyebrow', text: l === net.L ? 'Highest scoring validation images' : 'Top-activating validation images' }), grid);
      }
    }
  }

  // ==================================================================== Layer
  const Layer = {
    key: () => `${IN.layer}:${App.net.sizes.join('-')}:${App.ds.id}`,
    build(root) {
      const net = App.net, L = net.L, l = IN.layer = clamp(IN.layer, 1, L);
      const chips = el('div', { class: 'seg seg-sm', role: 'group', 'aria-label': 'Layer' });
      for (let k = 1; k <= L; k++) chips.appendChild(el('button', { type: 'button', 'aria-pressed': String(k === l), onclick: () => { IN.layer = k; IN.built = ''; IN.dirty = true; } }, App.layerName(k)));
      root.append(chips);
      if (l < L) {
        // the layer's activation, with a menu to set it for every neuron or add / remove layers
        const id = net.layerActId(l), act = id !== 'mixed' ? NA.engine.ACTIVATIONS[id] : null;
        const pick = el('button', { class: 'act-pick', type: 'button', 'aria-haspopup': 'listbox', 'aria-label': `Activation for ${App.layerName(l)}: ${act ? act.label : 'mixed'}. Change` },
          act ? NA.ui.actGlyph(id, 34, 22) : el('span', { class: 'lh-chip-mixed act-pick-mixed', 'aria-hidden': 'true', text: '≋' }),
          el('span', { class: 'act-pick-text' }, el('span', { class: 'act-pick-name', text: act ? act.label : 'Mixed activations' }), el('span', { class: 'act-pick-hint', text: act ? `f(z) = ${act.formula}` : 'Neurons in this layer use different functions' })),
          el('span', { class: 'lh-chip-caret', 'aria-hidden': 'true', text: '▾' }));
        pick.addEventListener('click', () => NA.openLayerMenu(l, pick));
        root.append(el('div', { class: 'layer-act' }, el('span', { class: 'eyebrow', text: 'Activation for every neuron' }), pick));
      }
      this.tiles = el('div', { class: 'tiles' });
      root.append(this.tiles);
      const nIn = net.sizes[l - 1], nOut = net.sizes[l];
      this.img = App.isImage() && l === 1;
      this.wCanvas = el('canvas', { class: 'hinton' });
      this.wBox = el('div', { class: 'chart-box chart-scroll' }, this.wCanvas);
      root.append(section(this.img ? 'Weight templates' : `Weights into ${App.layerName(l)} (${nOut} × ${nIn})`, this.wBox,
        el('p', { class: 'sec-note', text: this.img ? `One 14×14 template per hidden neuron. ${NA.tapWord()} a template to inspect that neuron.` : `Rows are this layer’s neurons, columns are their inputs. Square area ∝ |w|, colour = sign. ${NA.tapWord()} a row to inspect that neuron.` })));
      this.wCanvas.addEventListener('click', (e) => {
        const r = this.wCanvas.getBoundingClientRect(), y = e.clientY - r.top, x = e.clientX - r.left;
        if (this.img) { const k = this.tpl; const col = Math.floor(x / k.step), row = Math.floor(y / k.step), j = row * k.cols + col; if (j >= 0 && j < nOut) App.select({ l, j }, { fromGraph: true }); }
        else if (this.hin) { const j = Math.floor((y - this.hin.top) / this.hin.cell); if (j >= 0 && j < nOut) App.select({ l, j }, { fromGraph: true }); }
      });
      this.histCanvas = el('canvas');
      root.append(section('Weight distribution', el('div', { class: 'chart-box' }, this.histCanvas), el('p', { class: 'sec-note', text: 'All weights into this layer. The marks are the biases.' })));
      this.rangeCanvas = el('canvas');
      root.append(section(l === L ? 'Output range per neuron' : 'Activation range per neuron', el('div', { class: 'chart-box' }, this.rangeCanvas),
        el('div', { class: 'mini-legend' }, el('span', { class: 'lg-item', text: 'line: 5th–95th percentile' }), el('span', { class: 'lg-item', text: 'box: middle 50%' }), el('span', { class: 'lg-item', text: 'tick: median' }))));
      this.impEl = el('div', { class: 'imp-list' });
      if (l < L) root.append(section('Importance (knock-out test)', el('p', { class: 'sec-note', text: 'How much validation loss rises when each neuron is forced to 0. Tall bars mark neurons the network depends on.' }), this.impEl));
    },
    update() {
      const net = App.net, L = net.L, l = IN.layer, nIn = net.sizes[l - 1], nOut = net.sizes[l], Wt = net.W[l - 1], b = net.b[l - 1];
      const all = App.stats(), stats = all[l], last = App.metrics();
      let w2 = 0; for (const v of Wt) w2 += v * v;
      // dead (ReLU never fires) or flat neurons, and neurons stuck on a flat tail of their own activation
      let dead = 0, sat = 0, canSat = false;
      if (l < L) for (let j = 0; j < nOut; j++) {
        const act = net.actFns[l][j];
        if (act.sat && net.mask[l][j]) canSat = true;
        if (NA.engine.deadKind(net, all, l, j)) dead++;
        else if (act.sat && net.mask[l][j] && stats[j].satFrac > 0.9) sat++;
      }
      const grad = App.displayStep() > 0 ? App.gradReport().layerRms[l - 1] : 0;
      const ratio = last && last.ratio && last.ratio.length ? last.ratio[l - 1] : 0;
      const tiles = [
        tile('Parameters', fmtInt(Wt.length + b.length), `${nOut} × ${nIn} + ${nOut}`),
        tile('Weight norm ‖W‖', fmt(Math.sqrt(w2), 2), `RMS ${fmt(Math.sqrt(w2 / Wt.length), 3)}`),
        tile('Gradient RMS', fmt(grad, 4), 'full training set'),
        tile('Update / weight', ratio ? fmtPct(ratio, 2) : '—', ratio ? ratioWord(ratio) : 'per step'),
      ];
      if (l < L) {
        tiles.push(tile('Dead or flat', String(dead), `of ${nOut}`));
        if (canSat) tiles.push(tile('Saturated', String(sat), 'on a flat tail > 90% of the time'));
      }
      NA.clear(this.tiles).append(...tiles);
      const sel = App.selected && App.selected.l === l ? App.selected.j : null;
      if (this.img) {
        const G = App.ds.imgSize, avail = Math.max(160, this.wBox.clientWidth), cols = Math.max(4, Math.min(8, Math.floor(avail / 44))), step = Math.floor(avail / cols), size = step - 6, rows = Math.ceil(nOut / cols);
        const ctx = NA.fitCanvas(this.wCanvas, cols * step, rows * step);
        this.tpl = { cols, step };
        ctx.clearRect(0, 0, cols * step, rows * step);
        for (let j = 0; j < nOut; j++) {
          const ox = (j % cols) * step, oy = Math.floor(j / cols) * step, cell = size / G;
          let m = 1e-9; for (let i = 0; i < nIn; i++) m = Math.max(m, Math.abs(Wt[j * nIn + i]));
          for (let i = 0; i < nIn; i++) { ctx.fillStyle = theme.divColor(Wt[j * nIn + i] / m); ctx.fillRect(ox + (i % G) * cell, oy + Math.floor(i / G) * cell, cell + 0.4, cell + 0.4); }
          if (j === sel) { ctx.strokeStyle = theme.c('focus'); ctx.lineWidth = 2; ctx.strokeRect(ox - 1, oy - 1, size + 2, size + 2); }
          if (!net.mask[l][j]) { ctx.fillStyle = theme.c('panel', 0.7); ctx.fillRect(ox, oy, size, size); }
        }
      } else {
        // columns of a hidden layer's inputs are labelled by number only ("3" for "H1 · 3")
        this.hin = C.hinton(this.wCanvas, Wt, nOut, nIn, { selected: sel, colLabel: nIn <= 16 ? (i) => (l > 1 ? String(i + 1) : App.neuronName(0, i, true)) : null });
      }
      C.histogram(this.histCanvas, Wt, { bins: 32, colorFn: (v) => theme.divColor(v / (NA.maxAbs(Wt) || 1) * 0.8 + Math.sign(v) * 0.2), markers: Array.from(b).slice(0, 40).map((x) => ({ x, color: theme.c('ink', 0.5) })) });
      C.rangeStrips(this.rangeCanvas, stats, { selected: sel, label: (j) => (l === L ? App.neuronName(l, j, true) : String(j + 1)), isDead: (j) => l < L && !!NA.engine.deadKind(net, all, l, j) });
      if (this.impEl && l < L) {
        if (App.training || App.displayStep() === 0) { NA.clear(this.impEl).append(el('p', { class: 'empty-note', text: App.training ? 'Pause training to run the knock-out test.' : 'Train the network first.' })); }
        else {
          const imp = App.importance().layers[l];
          let m = 1e-6; for (const r of imp) m = Math.max(m, Math.abs(r.dLoss));
          const rows = imp.map((r, j) => el('button', { class: 'imp-row' + (j === sel ? ' is-sel' : ''), type: 'button', onclick: () => App.select({ l, j }, { fromGraph: true }) },
            el('span', { class: 'imp-name', text: App.neuronName(l, j, true) }),
            el('span', { class: 'imp-track' }, el('span', { class: 'imp-fill' + (r.dLoss < 0 ? ' neg' : ''), style: { width: (Math.min(1, Math.abs(r.dLoss) / m) * 100).toFixed(1) + '%' } })),
            el('span', { class: 'mono imp-val', text: r.off ? 'off' : fmtSigned(r.dLoss) })));
          NA.clear(this.impEl).append(...rows);
        }
      }
    },
  };
  function ratioWord(r) { return r < 1e-4 ? 'barely moving' : r < 3e-2 ? 'healthy range' : r < 0.1 ? 'fast' : 'unstable'; }
  function tile(label, value, sub) { return el('div', { class: 'tile' }, el('div', { class: 'tile-k', text: label }), el('div', { class: 'tile-v', text: value }), sub ? el('div', { class: 'tile-s', text: sub }) : null); }

  // ==================================================================== Training
  const Training = {
    key: () => `${App.net.sizes.join('-')}:${App.ds.id}:${App.ds.task}:${App.net.lossId}`,
    build(root) {
      const cls = App.ds.task !== 'regression', net = App.net;
      this.tiles = el('div', { class: 'tiles' });
      root.append(this.tiles);
      // the held-out test: run on request, never used for training or for choosing settings
      this.testEl = el('div', { class: 'test-card', id: 'testCard' });
      this.testSig = '';
      root.append(section('Held-out test', this.testEl));
      const Ls = NA.engine.LOSSES[net.lossId];
      const lossBtn = el('button', { class: 'act-pick', type: 'button', 'aria-haspopup': 'listbox', 'aria-label': `Loss function: ${Ls.label}. Change` },
        el('span', { class: 'act-pick-text' }, el('span', { class: 'act-pick-name', text: Ls.label }), el('span', { class: 'act-pick-hint act-pick-formula', text: Ls.formula[net.out] })),
        el('span', { class: 'lh-chip-caret', 'aria-hidden': 'true', text: '▾' }));
      lossBtn.addEventListener('click', () => NA.openLossMenu(lossBtn));
      root.append(section('Loss function', lossBtn, el('p', { class: 'sec-note', text: Ls.hint })));
      const mk = (title, opts, note, legend) => {
        const c = el('canvas');
        const chart = new C.LineChart(c, Object.assign({ label: title }, opts));
        root.append(section(title, el('div', { class: 'chart-box' }, c), legend || null, note ? el('p', { class: 'sec-note', text: note }) : null));
        return chart;
      };
      const lg = (items) => el('div', { class: 'mini-legend' }, ...items.map(([name, color]) => el('span', { class: 'lg-item' }, el('i', { class: 'key-line', style: { background: color } }), name)));
      const splits = () => lg([['train', 'var(--ink-2)'], ['validation', 'var(--pos)']]);
      this.loss = mk('Loss', { height: 150, logY: true, format: (v) => fmt(v, 4) }, 'Log scale. When the validation curve turns upward while training loss keeps falling, the model is overfitting.', splits());
      this.acc = mk(cls ? 'Accuracy' : 'R² (variance explained)', { height: 130, format: (v) => (cls ? fmtPct(v, 1) : fmt(v, 3)), yFormat: (v) => (cls ? Math.round(v * 100) + '%' : fmt(v, 1)) }, null, splits());
      const L = net.L;
      this.gradLegend = lg(NA.range(L).map((l) => [`into ${App.layerName(l + 1, true)}`, theme.series[l % 8]]));
      this.grad = mk('Gradient flow by layer', { height: 150, logY: true, format: (v) => fmt(v, 5) }, 'RMS gradient of each layer’s weights, smoothed over recent checkpoints. Lines far apart mean some layers learn much faster than others.', this.gradLegend);
      this.ratioEl = el('div', { class: 'ratio-table' });
      root.append(section('Update size per step', el('p', { class: 'sec-note', text: 'How far each layer’s weights moved in the last step, relative to their size. Roughly 0.01%–3% is healthy.' }), this.ratioEl));
      this.cmEl = el('div', { class: 'cm-wrap' });
      if (cls) root.append(section('Confusion matrix (validation set)', this.cmEl));
    },
    update() {
      const pts = App.history.points, m = App.metrics(), cls = App.ds.task !== 'regression', step = App.displayStep();
      const accLabel = cls ? 'accuracy' : 'R²', fa = (v) => (cls ? fmtPct(v, 1) : fmt(v, 3));
      // named as in the control bar ("Val." keeps "Val. accuracy" on one line in a tile)
      NA.clear(this.tiles).append(
        tile('Train loss', fmt(m.trainLoss, 4), App.hyper.reg !== 'none' && m.reg ? `+ ${fmt(m.reg, 4)} penalty` : null),
        tile('Val. loss', fmt(m.valLoss, 4), `gap ${fmtSigned(m.valLoss - m.trainLoss)}`),
        tile('Train ' + accLabel, fa(m.trainAcc)),
        tile('Val. ' + accLabel, fa(m.valAcc)),
        tile('Step', fmtInt(step), `epoch ${fmt(App.epoch(step), 1)}`));
      this.renderTest();
      const marker = App.viewing !== null ? step : null;
      const series = (key, name, color) => ({ name, color, points: pts.map((p) => [p.step, p[key]]) });
      this.loss.frame(); this.loss.set({ series: [series('trainLoss', 'train', theme.c('ink-2')), series('valLoss', 'validation', theme.c('pos'))], marker });
      this.acc.frame(); this.acc.set({ series: [series('trainAcc', 'train', theme.c('ink-2')), series('valAcc', 'validation', theme.c('pos'))], marker, yFixed: cls ? [0, 1] : undefined });
      // per-batch gradients are noisy: show an exponential moving average in log space
      const L = App.net.L, gs = [];
      for (let l = 0; l < L; l++) {
        let s = null;
        const points = pts.filter((p) => p.step > 0 && p.grad && p.grad.length === L && p.grad[l] > 0).map((p) => {
          const lg = Math.log10(p.grad[l]);
          s = s === null ? lg : 0.25 * lg + 0.75 * s;
          return [p.step, Math.pow(10, s)];
        });
        gs.push({ name: `into ${App.layerName(l + 1, true)}`, color: theme.series[l % 8], width: 1.8, points });
      }
      this.grad.frame(); this.grad.set({ series: gs, marker });
      // ratios
      const rows = [];
      for (let l = 0; l < L; l++) {
        const r = m.ratio && m.ratio.length === L ? m.ratio[l] : 0;
        const level = !r ? 'info' : r < 1e-4 ? 'info' : r < 3e-2 ? 'good' : r < 0.1 ? 'warn' : 'critical';
        rows.push(el('div', { class: 'rt-row' }, el('span', { class: 'rt-name', text: `into ${App.layerName(l + 1)}` }), el('span', { class: 'mono', text: r ? fmtPct(r, 3) : '—' }), pill(level, r ? ratioWord(r) : 'no steps yet')));
      }
      NA.clear(this.ratioEl).append(...rows);
      if (cls) {
        const ds = App.ds, preds = App.predictionsAll(), M = Array.from({ length: ds.classes }, () => new Array(ds.classes).fill(0));
        for (const k of ds.valIdx) M[ds.y[k]][preds[k]]++;
        NA.clear(this.cmEl).append(confusionTable(M));
      }
    },
    // Only redrawn when the result, its freshness or the controls around it change.
    renderTest() {
      const res = App.testResult, ds = App.ds, n = ds.testIdx.length, cls = ds.task !== 'regression';
      const stale = !!res && res.version !== App.version;
      const sig = [res ? res.version : '-', stale, App.training, App.displayStep() === 0, !!App.overlays.test].join('|');
      if (sig === this.testSig) return;
      this.testSig = sig;
      // one name for the action everywhere: "Run held-out test" (the control bar's "Held-out test · Run ›")
      const run = el('button', { class: 'btn btn-primary', type: 'button', onclick: () => { if (App.training) App.pause(); App.runTest(); } },
        res ? 'Run held-out test again' : 'Run held-out test');
      const showPts = ds.kind !== 'image' ? el('button', { class: 'chip', type: 'button', 'aria-pressed': String(!!App.overlays.test), onclick: () => App.setOverlay('test', !App.overlays.test) }, 'Show test points on the plot') : null;
      const kids = [];
      if (!res) {
        kids.push(el('p', { class: 'sec-note', text: `${fmtInt(n)} samples (${Math.round((100 * n) / ds.n)}% of the data) are held back from everything else: training never sees them and the validation numbers never use them. Tune the model against validation, then run the test once for an honest estimate of how it does on new data.` }));
        if (App.displayStep() === 0) kids.push(el('p', { class: 'sec-note', text: 'The network is untrained, so expect chance-level results.' }));
        kids.push(el('div', { class: 'btn-row' }, run, showPts));
      } else {
        kids.push(el('div', { class: 'test-head' },
          el('span', { class: 'eyebrow', text: `Tested at step ${fmtInt(res.step)} · ${fmtInt(res.n)} samples` }),
          stale ? pill('warn', 'Model changed since') : pill('good', 'Current model')));
        const tiles = cls ? [
          tile('Test accuracy', fmtPct(res.acc, 1), `validation ${fmtPct(res.valAcc, 1)}`),
          tile('Test loss', fmt(res.loss, 4), `validation ${fmt(res.valLoss, 4)}`),
          tile('Mistakes', fmtInt(res.errors.length), `of ${fmtInt(res.n)}`),
        ] : [
          tile('Test R²', fmt(res.acc, 3), `validation ${fmt(res.valAcc, 3)}`),
          tile('RMSE', fmt(res.rmse, 3), 'typical miss'),
          tile('MAE', fmt(res.mae, 3), 'mean |miss|'),
          tile('Test loss', fmt(res.loss, 4), `validation ${fmt(res.valLoss, 4)}`),
        ];
        kids.push(el('div', { class: 'tiles' }, ...tiles), testVerdict(res));
        if (cls) kids.push(el('div', { class: 'eyebrow', text: 'Confusion matrix (test set)' }), confusionTable(res.confusion));
        kids.push(testErrors(res), el('div', { class: 'btn-row' }, run, showPts));
      }
      NA.clear(this.testEl).append(...kids);
    },
  };

  function confusionTable(M) {
    const K = M.length, name = (k) => (App.isImage() ? String(k) : ['Orange', 'Blue', 'Green'][k]);
    const t = el('table', { class: 'cm' });
    t.appendChild(el('thead', null, el('tr', null, el('th', { class: 'cm-corner', text: 'true ↓ / predicted →' }), ...NA.range(K).map((k) => el('th', { scope: 'col', text: name(k) })))));
    const tb = el('tbody');
    for (let i = 0; i < K; i++) {
      const tot = M[i].reduce((a, b) => a + b, 0) || 1;
      tb.appendChild(el('tr', null, el('th', { scope: 'row', text: name(i) }), ...M[i].map((v, j) => {
        const f = v / tot, bg = i === j ? theme.seqColor(0.15 + 0.7 * f) : v ? theme.c('critical', 0.12 + 0.6 * f) : 'transparent';
        return el('td', { class: (i === j ? 'cm-diag' : v ? 'cm-err' : '') + (f > 0.55 ? ' cm-strong' : ''), style: { background: bg }, title: `True ${name(i)}, predicted ${name(j)}: ${v}` }, v ? String(v) : '·');
      })));
    }
    t.appendChild(tb);
    return el('div', { class: 'table-scroll' }, t);
  }

  // How the test score compares with validation and training, in words. The noise band is the 95%
  // range for the difference of two accuracies measured on samples of this size.
  function testVerdict(res) {
    const lines = [];
    if (res.task !== 'regression') {
      const se = Math.sqrt(Math.max(res.acc * (1 - res.acc), 0.01) / res.n) * 100, noise = Math.max(1, 2 * Math.SQRT2 * se), d = (res.acc - res.valAcc) * 100;
      if (Math.abs(d) < 0.05) lines.push(['good', `Test and validation accuracy agree (${fmtPct(res.acc, 1)}), so the validation numbers were a fair guide to how the model does on new data.`]);
      else if (Math.abs(d) <= noise) lines.push(['good', `Test accuracy is within ${fmt(Math.abs(d), 1)} points of validation, inside the ±${fmt(noise, 1)} that chance alone produces at this sample size. The validation numbers were a fair guide.`]);
      else if (d < 0) lines.push(['warn', `Test accuracy is ${fmt(-d, 1)} points below validation, more than chance (±${fmt(noise, 1)}) explains. Settings picked while watching validation may have fitted that set itself.`]);
      else lines.push(['info', `Test accuracy is ${fmt(d, 1)} points above validation. With splits this small the luck of the draw can do that; neither number is exact.`]);
      const gap = (res.trainAcc - res.acc) * 100;
      if (gap > Math.max(5, noise)) lines.push(['warn', `Training accuracy is ${fmt(gap, 1)} points higher than test accuracy: the model has learned details of the training set that do not carry over to new data.`]);
    } else {
      const d = res.acc - res.valAcc;
      if (Math.abs(d) <= 0.03) lines.push(['good', `Test R² is within ${fmt(Math.abs(d), 3)} of validation, so the validation numbers were a fair guide.`]);
      else if (d < 0) lines.push(['warn', `Test R² is ${fmt(-d, 3)} lower than validation. Settings picked while watching validation may have fitted that set itself, or the split is small.`]);
      else lines.push(['info', `Test R² is ${fmt(d, 3)} higher than validation; with this few points the split alone can do that.`]);
      const gap = res.trainAcc - res.acc;
      if (gap > 0.08) lines.push(['warn', `Training R² is ${fmt(gap, 3)} higher than test R²: the curve is following noise in the training points.`]);
    }
    return el('div', { class: 'test-verdict' }, ...lines.map(([lv, t]) => el('p', { class: 'tv-line' }, pill(lv, { good: 'Good', info: 'Note', warn: 'Warning' }[lv]), el('span', { text: t }))));
  }

  // The worst test samples; clicking one makes it the probe.
  function testErrors(res) {
    const ds = App.ds, cls = res.task !== 'regression', shown = res.errors.slice(0, 12);
    if (!shown.length) return el('p', { class: 'sec-note', text: 'No mistakes on the test set.' });
    const pick = (k) => () => { App.setProbeSample(k); if (document.body.dataset.layout === 'mobile') App.toast('Probe moved to this test sample. Open Network or Data to trace it.'); };
    const head = el('div', { class: 'eyebrow', text: (cls ? 'Most confident mistakes' : 'Largest misses') + (cls && res.errors.length > shown.length ? ` · ${shown.length} of ${res.errors.length}` : '') });
    const note = el('p', { class: 'sec-note', text: `${NA.tapWord()} one to make it the probe and trace it through the network.` });
    if (ds.kind === 'image') {
      return el('div', { class: 'test-errs' }, head, el('div', { class: 'top-imgs' }, ...shown.map((e) => el('button', { class: 'g-item is-wrong', type: 'button', title: `True ${e.truth}, predicted ${e.pred} (${fmtPct(e.conf, e.conf > 0.99 ? 1 : 0)} sure)`, onclick: pick(e.k) },
        NA.ui.digitCanvas(e.k), el('span', { class: 'g-lab', text: `${e.truth}→${e.pred}` })))), note);
    }
    const name = (k) => ['Orange', 'Blue', 'Green'][k];
    const rows = shown.map((e) => {
      const x = ds.raw[e.k * ds.rawDim], where = ds.rawDim > 1 ? `(${fmt(x, 2)}, ${fmt(ds.raw[e.k * ds.rawDim + 1], 2)})` : `x = ${fmt(x, 2)}`;
      const what = cls
        ? [el('span', { class: 'er-what' }, el('i', { class: 'dot', style: { background: theme.classColor(e.truth) } }), `${name(e.truth)}, called ${name(e.pred).toLowerCase()}`), el('span', { class: 'mono er-v', text: `${fmtPct(e.conf, e.conf > 0.99 ? 1 : 0)} sure` })]
        : [el('span', { class: 'er-what', text: `target ${fmt(e.truth, 2)}, predicted ${fmt(e.pred, 2)}` }), el('span', { class: 'mono er-v', text: `off ${fmt(e.err, 2)}` })];
      return el('button', { class: 'err-row', type: 'button', onclick: pick(e.k) }, el('span', { class: 'mono er-pos', text: where }), ...what);
    });
    return el('div', { class: 'test-errs' }, head, el('div', { class: 'err-list' }, ...rows), note);
  }

  // ==================================================================== Backprop (delegated)
  const Backprop = {
    key: () => `${App.net.sizes.join('-')}:${App.ds.id}`,
    build(root) { NA.walk.build(root); },
    update() { NA.walk.update(); },
  };

  const TABS = { insights: Insights, neuron: Neuron, layer: Layer, training: Training, backprop: Backprop };
  NA.inspector = IN;
})(window.NA);
