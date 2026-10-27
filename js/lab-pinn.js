/* lab-pinn.js — the physics-informed network lab. A plain network and a PINN train side by side on the same
   few measurements of a damped oscillator. Plots show the fit, how badly each breaks the differential
   equation, and the phase portrait; inspector tabs cover the loss terms, the derivatives behind the residual
   and the network's basis functions. */
(function (NA) {
  'use strict';
  const { el, clear, clamp, fmt, fmtPct, fmtInt, theme } = NA;
  const rr = (...a) => NA.rr(...a);
  const App = NA.app, H = NA.labHost, UI = NA.labUI, PN = NA.pinn;

  const DEFAULT = { zeta: 0.1, omega: 18, nData: 10, tMax: 0.36, noise: 0.03, nColloc: 40, seed: 3 };
  const COL = { exact: () => theme.c('ink-2'), plain: () => theme.c('neg'), pinn: () => theme.c('pos'), data: () => theme.c('ink') };

  const PL = H.transport({
    id: 'pinn', screens: { network: 'Plots', data: 'Problem', insp: 'Inspect' },
    budget: 12, lr: 5e-3, width: 32, depth: 3, lambda: 1, terms: { data: true, phys: true, ic: false }, probeT: 0.62, version: 0,
  });

  // ------------------------------------------------------------------ problem and training
  PL.setup = function (opts, pretrained) {
    this.probOpts = Object.assign({}, DEFAULT, this.probOpts, opts || {});
    this.prob = PN.makeProblem(this.probOpts);
    const sizes = [1, ...Array(this.depth).fill(this.width), 1];
    this.plain = new PN.PINN(sizes, 1);
    this.pinn = new PN.PINN(sizes, 1);
    this.history = []; this.step = 0;
    const W = NA.pinnWeights;
    if (pretrained && W && W.pinn.sizes.join() === sizes.join()) {
      this.plain.setParams(NA.tf.decodeWeights(W.plain.weights));
      this.pinn.setParams(NA.tf.decodeWeights(W.pinn.weights));
      this.step = W.pinn.steps;
      this.history = W.pinn.curve.map(([step, data, phys, ic, err], i) => ({ step, data, phys, ic, pinnErr: err, plainErr: W.plain.curve[i] ? W.plain.curve[i][4] : null }));
      this.pretrained = true;
    } else this.pretrained = false;
    this.measure();
    this.bump();
  };
  PL.weights = function () { return { data: this.terms.data ? 1 : 0, phys: this.terms.phys ? this.lambda : 0, ic: this.terms.ic ? 1 : 0 }; };
  PL.lrNow = function () { return this.lr * (this.step > 6000 ? 0.3 : 1); };
  PL.trainChunk = function (budget, maxSteps) {
    const t0 = performance.now(), P = this.prob, w = this.weights(), plainW = { data: 1, phys: 0, ic: 0 };
    let k = 0, last = null;
    do {
      const lr = this.lrNow();
      this.plain.trainStep(P, P.data, P.colloc, plainW, lr);
      last = this.pinn.trainStep(P, P.data, P.colloc, w, lr);
      this.step++; k++;
      if (this.step % 100 === 0) this.record(last);
    } while (performance.now() - t0 < budget && k < maxSteps);
    this.last = last;
    if (!isFinite(last.total)) { this.pause(); NA.ui.toast.show('Training diverged. Lower the learning rate and retrain.'); }
    this.bump();
    return k;
  };
  PL.record = function (r) {
    this.measure();
    this.history.push({ step: this.step, data: r.data, phys: r.phys, ic: r.ic, pinnErr: this.m.pinnErr, plainErr: this.m.plainErr });
    if (this.history.length > 400) this.history = this.history.filter((q, i) => i % 2 === 0 || i === this.history.length - 1);
  };
  PL.onPause = function () { this.measure(); this.bump(); };
  PL.stepOnce = function () { if (this.training) this.pause(); this.trainChunk(0, 1); this.measure(); };
  PL.reset = function () { if (this.training) this.pause(); this.setup(); NA.ui.toast.show('Both networks re-initialized. Press Train.'); };
  PL.retrain = function (msg) { this.pause(); this.setup(); this.play(); if (msg) NA.ui.toast.show(msg); };
  PL.bump = function () { this.version++; this.dirty = true; this.inspDirty = true; this.leftDirty = true; this.archStale = true; this.infoStale = true; };

  // Errors against the exact solution, overall and beyond the data, and the ODE residual of both networks.
  PL.measure = function () {
    const P = this.prob, n = 160, k = P.omega * P.omega, c = 2 * P.zeta * P.omega;
    const acc = (net) => {
      let num = 0, den = 0, numX = 0, denX = 0, res = 0;
      for (let i = 0; i <= n; i++) {
        const t = i / n, e = PN.exact(t, P.zeta, P.omega), [u, u1, u2] = net.forward(t, 2), d = u - e, r = (u2 + c * u1 + k * u) / k;
        num += d * d; den += e * e; res += r * r;
        if (t > P.tMax) { numX += d * d; denX += e * e; }
      }
      return { err: Math.sqrt(num / den), extra: denX ? Math.sqrt(numX / denX) : 0, res: Math.sqrt(res / (n + 1)) };
    };
    const a = acc(this.plain), b = acc(this.pinn);
    this.m = { plainErr: a.err, pinnErr: b.err, plainExtra: a.extra, pinnExtra: b.extra, plainRes: a.res, pinnRes: b.res };
  };

  // ------------------------------------------------------------------ lifecycle
  PL.activate = function (preset, ui) {
    this.ui = ui; this.preset = preset;
    if (!this.prob) this.setup({}, true);
    this.ro = UI.readouts(ui.readouts, [
      { key: 'step', label: 'Step', keep: true }, { key: 'pinn', label: 'PINN error', keep: true, title: 'Relative L2 error against the exact solution over the whole window' },
      { key: 'plain', label: 'Plain error', title: 'Same, for the network trained on the data alone' }, { key: 'res', label: 'PINN residual', title: 'RMS of (u″ + 2ζωu′ + ω²u) / ω² over the window' },
    ]);
    this.buildHyper(ui.hyper);
    this.buildLeft(ui.left);
    this.buildCenter(ui.center);
    this.tabs = UI.tabs(ui.right, [{ id: 'overview', label: 'Overview' }, { id: 'loss', label: 'Losses' }, { id: 'deriv', label: 'Derivatives' }, { id: 'net', label: 'Network' }], (id, body) => this.buildTab(id, body));
    this.offTheme = theme.onChange(() => this.bump());
    this.bump();
  };
  PL.deactivate = function () { if (this.offTheme) this.offTheme(); if (this.resizeObs) this.resizeObs.disconnect(); if (this.arch) this.arch.destroy(); };
  PL.archText = function () { return `t → ${this.depth} × ${this.width} tanh → u(t), with u′ and u″ by exact differentiation`; };
  PL.frame = function (now) {
    /* Three plots share the stage, so the explanation stays open at first only if all three still fit at a useful
       size with it open: the fit at least 150 px tall and a phase portrait of PH_MIN or more. Measured on the
       first frame, once the panels have their final sizes, and before anything is painted. */
    if (this.checkInfo) {
      this.checkInfo = false;
      const lay = this.stage.offsetParent && this.layout(this.stage.clientWidth, this.stage.clientHeight);
      if (lay && (!lay.ph || lay.main.h < 150)) this.info.toggle(false);
    }
    this.runFrame();
    if (now - (this.lastRo || 0) > 120) {
      this.lastRo = now;
      this.ro.set('step', fmtInt(this.step)); this.ro.set('pinn', fmtPct(this.m.pinnErr, 1)); this.ro.set('plain', fmtPct(this.m.plainErr, 1)); this.ro.set('res', fmt(this.m.pinnRes, 3));
    }
    if (this.training && now - (this.lastMeasure || 0) > 300) { this.lastMeasure = now; this.measure(); }
    if (this.dirty && this.canvas.offsetParent && (!this.training || now - (this.lastDraw || 0) > 60)) { this.lastDraw = now; this.dirty = false; this.draw(); }
    if (this.centerView === 'arch') {
      // the diagram shows live numbers: redraw it when they change
      if (this.archStale && (!this.training || now - (this.lastArch || 0) > 400)) { this.lastArch = now; this.archStale = false; this.arch.dirty = true; }
      this.arch.frame(now);
    }
    // so does the explanation under the picture; refresh it when training pauses
    if (this.infoStale && !this.training && !this.arch.anim) { this.infoStale = false; if (this.centerView === 'arch') this.arch.refreshFocus(); else this.info.set(PLOTS_INFO(this)); }
    if (this.leftDirty && this.ui.left.offsetParent && (!this.training || now - (this.lastLeft || 0) > 300)) { this.lastLeft = now; this.leftDirty = false; this.renderLeft(); }
    if (this.inspDirty && this.tabs.body.offsetParent && (!this.training || now - (this.lastInsp || 0) > 400)) { this.lastInsp = now; this.inspDirty = false; this.updateTab(); }
  };

  PL.buildHyper = function (root) {
    const rebuild = (fn) => (v) => { fn(v); this.retrain('New network shape: both networks retrain from scratch.'); App.emit('labarch'); };
    root.append(
      UI.field('Learning rate', UI.select([0.001, 0.002, 0.005, 0.01].map((v) => ({ value: v, label: String(v) })), this.lr, (v) => { this.lr = +v; }), 'Adam step size; it drops to 30% after 6,000 steps'),
      UI.field('Width', UI.select([16, 32, 64].map((v) => ({ value: v, label: v + ' neurons' })), this.width, rebuild((v) => { this.width = +v; }))),
      UI.field('Depth', UI.select([1, 2, 3, 4].map((v) => ({ value: v, label: v + ' hidden' })), this.depth, rebuild((v) => { this.depth = +v; }))),
      el('div', { class: 'sheet-actions' },
        el('button', { class: 'btn', type: 'button', onclick: () => { NA.mobile && NA.mobile.sheet(false); this.fastForward(2000); } }, 'Train 2,000 steps'),
        el('button', { class: 'btn', type: 'button', onclick: () => { NA.mobile && NA.mobile.sheet(false); this.retrain('Retraining both networks from scratch.'); } }, 'Retrain both')));
  };

  // ------------------------------------------------------------------ left: the problem
  PL.buildLeft = function (root) {
    const body = UI.panel(root, 'Problem'), o = this.probOpts;
    this.eqEl = el('div', { class: 'fx pinn-eq' });
    const change = (key) => (v) => { this.probOpts[key] = v; this.retrain('New problem: both networks retrain from scratch.'); };
    this.sliders = {
      zeta: UI.slider({ label: 'Damping ζ', min: 0.02, max: 0.4, step: 0.01, value: o.zeta, format: (v) => v.toFixed(2), onInput: () => { this.leftDirty = true; }, onChange: change('zeta') }),
      omega: UI.slider({ label: 'Natural frequency ω', min: 6, max: 30, step: 1, value: o.omega, format: (v) => v.toFixed(0), onChange: change('omega') }),
      nData: UI.slider({ label: 'Measurements', min: 2, max: 30, step: 1, value: o.nData, format: (v) => v.toFixed(0), onChange: change('nData') }),
      tMax: UI.slider({ label: 'Measured until t =', min: 0.1, max: 1, step: 0.02, value: o.tMax, format: (v) => v.toFixed(2), onChange: change('tMax') }),
      noise: UI.slider({ label: 'Measurement noise', min: 0, max: 0.1, step: 0.005, value: o.noise, format: (v) => v.toFixed(3), onChange: change('noise') }),
      nColloc: UI.slider({ label: 'Collocation points', min: 5, max: 100, step: 1, value: o.nColloc, format: (v) => v.toFixed(0), onChange: change('nColloc') }),
    };
    const terms = el('div', { class: 'chip-row' },
      UI.chip('Data loss', this.terms.data, (on) => { this.terms.data = on; this.retrainPinn(); }, 'Squared error at the measurements'),
      UI.chip('Physics loss', this.terms.phys, (on) => { this.terms.phys = on; this.retrainPinn(); }, 'Squared ODE residual at the collocation points'),
      UI.chip('Initial conditions', this.terms.ic, (on) => { this.terms.ic = on; this.retrainPinn(); }, 'u(0) = 1 and u′(0) = 0'));
    this.termChips = terms;
    const lam = UI.slider({ label: 'Physics weight λ', min: -1, max: 1, step: 0.1, value: Math.log10(this.lambda), format: (v) => fmt(Math.pow(10, v), 2), onChange: (v) => { this.lambda = Math.pow(10, v); this.retrainPinn(); } });
    this.probeEl = el('div', { class: 'kv-grid' });
    body.append(
      UI.section('The equation', this.eqEl, UI.note('A mass on a spring with friction: released from u = 1 at rest, it oscillates and slowly settles. The exact solution is known, so both networks can be scored against it.')),
      UI.section('Measurements', this.sliders.nData, this.sliders.tMax, this.sliders.noise),
      UI.section('Physics', this.sliders.zeta, this.sliders.omega, this.sliders.nColloc),
      UI.section('What the PINN minimises', UI.note('The plain network always uses the data loss alone. Switch the PINN’s terms here; it retrains when you change them.'), terms, lam),
      UI.section('At the probe', UI.note('Hover or tap the main plot to move the probe.'), this.probeEl));
  };
  PL.retrainPinn = function () {
    this.pause();
    const sizes = [1, ...Array(this.depth).fill(this.width), 1];
    this.pinn = new PN.PINN(sizes, 1);
    this.plain = new PN.PINN(sizes, 1);
    this.history = []; this.step = 0; this.pretrained = false;
    this.measure(); this.bump(); this.play();
    const w = this.weights(), parts = [w.data && 'data', w.phys && 'physics', w.ic && 'initial conditions'].filter(Boolean);
    NA.ui.toast.show(parts.length ? `PINN loss: ${parts.join(' + ')}. Retraining.` : 'With no loss terms the PINN has nothing to learn from.');
  };
  PL.renderLeft = function () {
    const P = this.prob, z = this.sliders.zeta.input ? +this.sliders.zeta.input.value : P.zeta;
    clear(this.eqEl).append(
      el('div', { class: 'fx-line fx-strong', text: `u″ + ${fmt(2 * P.zeta * P.omega, 2)}·u′ + ${fmt(P.omega * P.omega, 0)}·u = 0` }),
      el('div', { class: 'fx-line', text: `= u″ + 2ζω·u′ + ω²·u  with ζ = ${fmt(P.zeta, 2)}, ω = ${fmt(P.omega, 0)}` }),
      el('div', { class: 'fx-line fx-muted', text: 'u(0) = 1,  u′(0) = 0,  t from 0 to 1' }),
      el('div', { class: 'fx-line fx-muted', text: `exact: u = e^(−ζωt)·(cos ω_d t + (ζω/ω_d) sin ω_d t),  ω_d = ${fmt(P.omega * Math.sqrt(1 - P.zeta * P.zeta), 2)}` }));
    void z;
    const t = this.probeT, e = PN.exact(t, P.zeta, P.omega), a = this.plain.forward(t, 2), b = this.pinn.forward(t, 2), k = P.omega ** 2, c = 2 * P.zeta * P.omega;
    clear(this.probeEl).append(
      kv('t', fmt(t, 3)), kv('exact u', fmt(e, 3)),
      kv('plain network', fmt(a[0], 3), `residual ${fmt((a[2] + c * a[1] + k * a[0]) / k, 3)}`),
      kv('PINN', fmt(b[0], 3), `residual ${fmt((b[2] + c * b[1] + k * b[0]) / k, 3)}`));
  };
  const kv = (k, v, note) => el('div', { class: 'kv' }, el('span', { class: 'kv-k', text: k }), el('span', { class: 'kv-v', text: v }), note ? el('span', { class: 'kv-note', text: note }) : null);

  // ------------------------------------------------------------------ centre: plots
  PL.buildCenter = function (root) {
    const show = this.show || (this.show = { plain: true, pinn: true, exact: true });
    this.centerView = this.centerView || 'plots';
    this.viewSeg = UI.seg([{ value: 'plots', label: 'Plots', title: 'The fits, the residual and the phase portrait' }, { value: 'arch', label: 'Architecture', title: 'The network and its loss: click or zoom in to open the parts' }], this.centerView, (v) => this.setCenterView(v), 'View');
    this.plotTools = el('span', { class: 'lab-tools' }, el('span', { class: 'lens-label', text: 'Show' }),
      UI.chip('Exact solution', show.exact, (on) => { show.exact = on; this.dirty = true; }),
      UI.chip('Plain network', show.plain, (on) => { show.plain = on; this.dirty = true; }),
      UI.chip('PINN', show.pinn, (on) => { show.pinn = on; this.dirty = true; }));
    root.append(el('div', { class: 'lab-toolbar' }, this.viewSeg, this.plotTools));
    this.stage = el('div', { class: 'lab-stage' });
    this.canvas = el('canvas', { 'aria-label': 'Plots: the fitted curves against the exact solution, the equation residual, and the phase portrait' });
    this.stage.append(this.canvas);
    this.archStage = el('div', { class: 'lab-stage arch-stage' });
    this.arch = new NA.ArchView(this.archStage, { label: 'PINN architecture: click a part or zoom in to open it', build: (size) => NA.archPinn.build(this, size), onFocus: (node, path) => { if (this.centerView === 'arch') this.info.set(this.arch.infoFor(node, path)); } });
    root.append(this.stage, this.archStage);
    this.info = UI.infoPanel(root, { open: !(NA.mobile && NA.mobile.mobile) && root.clientHeight >= 680 }); // open at first only when there is room for it under the picture
    // the PINN's entry and the collocation points follow the loss terms in use (renderFoot)
    this.footPinn = el('span');
    this.footColloc = el('span', { class: 'lg-item' }, el('i', { class: 'key-tri' }), 'collocation point');
    root.append(this.foot = el('div', { class: 'lab-foot' },
      el('span', { class: 'lg-item' }, el('i', { class: 'key-line', style: { background: 'var(--ink-2)' } }), 'exact solution (dashed)'),
      el('span', { class: 'lg-item' }, el('i', { class: 'key-line', style: { background: 'var(--neg)' } }), 'plain network: data loss only'),
      el('span', { class: 'lg-item' }, el('i', { class: 'key-line', style: { background: 'var(--pos)' } }), this.footPinn),
      el('span', { class: 'lg-item' }, el('i', { class: 'dot dot-filled' }), 'measurement'),
      this.footColloc));
    this.checkInfo = this.info.open; // see frame()
    const move = (e) => { const p = this.plotAt(e); if (p !== null) { this.probeT = p; this.dirty = true; this.leftDirty = true; this.inspDirty = true; this.archStale = true; } };
    this.canvas.addEventListener('pointermove', (e) => { if (e.pointerType === 'mouse' || e.buttons) move(e); });
    this.canvas.addEventListener('pointerdown', move);
    this.resizeObs = new ResizeObserver(() => { this.dirty = true; });
    this.resizeObs.observe(this.stage);
    this.setCenterView(this.centerView);
  };
  // The key under the plots names the PINN's loss terms as they are set now, and shows the collocation points only when they are drawn
  PL.renderFoot = function () {
    const w = this.weights(), parts = [w.data && 'data', w.phys && 'physics', w.ic && 'initial conditions'].filter(Boolean);
    this.footPinn.textContent = `PINN: ${parts.length ? parts.join(' + ') : 'no loss terms'}`;
    this.footColloc.hidden = !this.terms.phys;
  };
  PL.setCenterView = function (v) {
    this.centerView = v;
    const arch = v === 'arch';
    this.stage.hidden = arch; this.foot.hidden = arch; this.plotTools.hidden = arch; this.archStage.hidden = !arch;
    this.viewSeg.set(v);
    this.dirty = true; this.arch.dirty = true;
    if (arch) this.arch.refreshFocus();
    else this.info.set(PLOTS_INFO(this));
  };
  const pct = (v) => fmtPct(v, v < 0.1 ? 1 : 0);
  const PLOTS_INFO = (L) => {
    const P = L.prob, m = L.m, num = (v, d) => NA.fmtCode(v, d);
    return {
      kicker: 'Plots view', title: 'The fit, the physics and the phase portrait',
      about: [
        `Top plot: the position u(t) of a damped spring over one second. Dashed is the exact solution, which neither network ever sees. The dots are the ${P.data.length} noisy measurements, all inside the shaded region (t ≤ ${fmt(P.tMax, 2)}); the small triangles along the bottom are the ${P.nColloc} collocation points where the PINN is checked against the equation. Orange is a plain network trained on the measurements alone, blue the PINN, trained on the measurements plus the equation.`,
        `Inside the shaded region both curves pass through the dots. Beyond it the plain network has nothing to go on: it carries on however its tanh units happen to bend, and is off by ${pct(m.plainExtra)} after the last measurement. The PINN keeps swinging at the right frequency and decaying at the right rate, because a curve that breaks the equation is penalised everywhere, not only where there is data (${pct(m.pinnExtra)} off there).`,
        `Residual plot: |u″ + 2ζω·u′ + ω²·u| / ω² along t on a log scale, where each grid line is a factor of 100. Zero means the curve obeys the physics exactly. The PINN keeps it small everywhere (root mean square ${fmt(m.pinnRes, 3)}); the plain network was never asked to and breaks it by orders of magnitude (${fmt(m.plainRes, 2)}). Its u″ comes from differentiating the network twice, so any wiggle it makes between measurements shows up here, magnified.`,
        'Phase portrait: u against u′/ω. A spring without friction would go round one circle forever; damping turns the circle into an inward spiral, one turn per oscillation, starting from the dot at (1, 0): released from 1, at rest. The PINN traces the spiral. The plain network’s portrait shows the velocity its curve implies, which nothing taught it.',
        'Hover or tap the top plot to move the probe (the dashed vertical line). The Problem panel lists the exact value and both networks’ values and residuals there, and the Architecture view follows the same t through every layer of the PINN.',
      ],
      code: `import math
import torch
import torch.nn as nn
import matplotlib.pyplot as plt

zeta, omega, t_max = ${num(P.zeta, 2)}, ${num(P.omega, 0)}.0, ${num(P.tMax, 2)}

def exact(t):
    wd = omega * math.sqrt(1 - zeta ** 2)
    return torch.exp(-zeta * omega * t) * (torch.cos(wd * t) + zeta * omega / wd * torch.sin(wd * t))

class PINN(nn.Module):
    def __init__(self, width=${L.width}, depth=${L.depth}):
        super().__init__()
        layers, d_in = [], 1
        for _ in range(depth):
            layers += [nn.Linear(d_in, width), nn.Tanh()]
            d_in = width
        self.net = nn.Sequential(*layers, nn.Linear(d_in, 1))

    def forward(self, t):
        return self.net(2 * t - 1)

def derivatives(model, t):
    """u, u' and u'' at the times t, by differentiating the network itself."""
    t = t.clone().requires_grad_(True)
    u = model(t)
    du = torch.autograd.grad(u, t, torch.ones_like(u), create_graph=True)[0]
    d2u = torch.autograd.grad(du, t, torch.ones_like(du), create_graph=True)[0]
    return u, du, d2u

def residual(model, t):
    u, du, d2u = derivatives(model, t)
    return (d2u + 2 * zeta * omega * du + omega ** 2 * u) / omega ** 2

# ${P.data.length} noisy measurements up to t_max, and ${P.nColloc} collocation points over the whole window
t_data = torch.linspace(0, t_max, ${P.data.length}).view(-1, 1)
u_data = exact(t_data) + ${num(P.noise, 3)} * torch.randn_like(t_data)
t_col = ((torch.arange(${P.nColloc}) + 0.5) / ${P.nColloc}).view(-1, 1)

def train(model, lam, steps=10_000):
    opt = torch.optim.Adam(model.parameters(), lr=${L.lr})
    for step in range(steps):
        if step == 6000:
            opt.param_groups[0]['lr'] *= 0.3
        loss = ((model(t_data) - u_data) ** 2).mean()
        if lam:
            loss = loss + lam * (residual(model, t_col) ** 2).mean()
        opt.zero_grad()
        loss.backward()
        opt.step()
    return model

plain = train(PINN(), lam=0)                       # the measurements only
pinn = train(PINN(), lam=${num(L.lambda, 2)})                    # the measurements and the equation

t = torch.linspace(0, 1, 221).view(-1, 1)
fig, (top, mid, low) = plt.subplots(3, 1, figsize=(7, 11))
top.axvspan(0, t_max, alpha=0.08)
top.plot(t, exact(t), 'k--', label='exact')
for name, model in [('plain', plain), ('PINN', pinn)]:
    u, du, _ = derivatives(model, t)
    top.plot(t, u.detach(), label=name)
    mid.semilogy(t, residual(model, t).abs().detach(), label=name)
    low.plot(u.detach(), du.detach() / omega, label=name)
top.scatter(t_data, u_data, c='k', zorder=3)
mid.set_ylabel('|residual|')
low.set_xlabel('u'); low.set_ylabel("u' / omega"); low.set_aspect('equal')
top.legend(); mid.legend(); low.legend()
plt.show()`,
      codeLabel: 'Python · PyTorch and matplotlib · train both networks and draw these plots',
    };
  };
  PL.plotAt = function (e) {
    const L = this.lay;
    if (!L) return null;
    const r = this.canvas.getBoundingClientRect(), x = e.clientX - r.left, y = e.clientY - r.top, m = L.main;
    if (x < m.x || x > m.x + m.w || y < m.y - 10 || y > m.y + m.h + 10) return null;
    return clamp((x - m.x) / m.w, 0, 1);
  };
  PL.curves = function () {
    if (this.cv && this.cv.version === this.version) return this.cv;
    const P = this.prob, n = 220, k = P.omega ** 2, c = 2 * P.zeta * P.omega, mk = () => ({ u: new Float32Array(n + 1), u1: new Float32Array(n + 1), r: new Float32Array(n + 1) });
    const out = { version: this.version, n, exact: mk(), plain: mk(), pinn: mk() };
    for (let i = 0; i <= n; i++) {
      const t = i / n;
      out.exact.u[i] = PN.exact(t, P.zeta, P.omega); out.exact.u1[i] = PN.exactD(t, P.zeta, P.omega);
      for (const key of ['plain', 'pinn']) {
        const [u, u1, u2] = this[key].forward(t, 2);
        out[key].u[i] = u; out[key].u1[i] = u1; out[key].r[i] = (u2 + c * u1 + k * u) / k;
      }
    }
    this.cv = out;
    return out;
  };
  /* Where the three plots go, for the stage's size. The fit is the main picture: full width on top, never
     shorter than MAIN_MIN. The residual goes under it. The phase portrait is a square that is unreadable under
     about PH_MIN, so it takes the first arrangement that leaves it that much: beside the residual, else in a
     column beside both plots (short, wide stages), else under the residual (tall, narrow ones). When none
     does, it is left out and a note says so, rather than drawn as a postage stamp. */
  const MAIN_MIN = 96, RES_MIN = 44, PH_MIN = 120, AX = 46, PAD = 12, TOP = 24, GAP = 40, BOT = 12, NOTE_H = 18;
  PL.layout = function (W, Hh) {
    const plotW = W - AX - PAD, avail = Hh - TOP - GAP - BOT;
    // beside the residual: the main plot keeps a little over half the height
    {
      let mainH = Math.max(MAIN_MIN, avail * 0.55), belowH = avail - mainH;
      if (belowH < PH_MIN) { belowH = PH_MIN; mainH = avail - PH_MIN; }
      const s = Math.min(belowH, plotW * 0.45), resW = plotW - s - 40;
      if (mainH >= Math.max(MAIN_MIN, avail * 0.42) && s >= PH_MIN && resW >= 130) {
        const below = TOP + mainH + GAP;
        return { kind: 'below', main: { x: AX, y: TOP, w: plotW, h: mainH }, res: { x: AX, y: below, w: resW, h: belowH }, ph: { x: W - PAD - s, y: below, w: s, h: s } };
      }
    }
    // a column of its own on the right, as tall as both plots, when the fit can stay at least 300 px wide
    {
      const s = Math.min(Hh - TOP - BOT, plotW * 0.42), mainW = plotW - s - 36, mainH = Math.max(MAIN_MIN, avail * 0.6), resH = avail - mainH;
      if (s >= PH_MIN && mainW >= 300 && resH >= RES_MIN) return { kind: 'side', main: { x: AX, y: TOP, w: mainW, h: mainH }, res: { x: AX, y: TOP + mainH + GAP, w: mainW, h: resH }, ph: { x: W - PAD - s, y: TOP, w: s, h: s } };
    }
    // one under the other
    {
      const room = avail - 34, s0 = Math.min(plotW - 40, room * 0.34), mainH = Math.max(MAIN_MIN, (room - s0) * 0.7), resH = room - s0 - mainH;
      if (s0 >= PH_MIN && resH >= RES_MIN) {
        const resY = TOP + mainH + GAP, phY = resY + resH + 34, s = Math.min(plotW - 40, Hh - BOT - phY);
        return { kind: 'stack', main: { x: AX, y: TOP, w: plotW, h: mainH }, res: { x: AX, y: resY, w: plotW, h: resH }, ph: { x: AX + (plotW - s) / 2, y: phY, w: s, h: s } };
      }
    }
    // no room for a readable phase portrait: the fit and the residual share the height, with a note under them
    const room = avail - NOTE_H, mainH = Math.max(Math.min(MAIN_MIN, room - RES_MIN), room * 0.6);
    return { kind: 'nophase', main: { x: AX, y: TOP, w: plotW, h: mainH }, res: { x: AX, y: TOP + mainH + GAP, w: plotW, h: Math.max(24, room - mainH) }, ph: null };
  };
  PL.draw = function () {
    const W = Math.max(240, this.stage.clientWidth), Hh = Math.max(220, this.stage.clientHeight), ctx = NA.fitCanvas(this.canvas, W, Hh);
    const P = this.prob, cv = this.curves(), show = this.show;
    ctx.clearRect(0, 0, W, Hh);
    const lay = this.layout(W, Hh), { main, res, ph } = lay;
    this.lay = lay;
    this.renderFoot();
    const font = (wgt, px) => `${wgt} ${px}px ${getComputedStyle(document.body).getPropertyValue('--font-ui') || 'sans-serif'}`;
    // titles hold maths (u(t), ω): upper-casing would turn them into different symbols, so they keep their case
    const title = (r, text) => { ctx.font = font(600, 11); ctx.fillStyle = theme.c('muted'); ctx.textAlign = 'left'; ctx.textBaseline = 'alphabetic'; ctx.fillText(text, r.x, r.y - 8); };
    const frame = (r) => { ctx.strokeStyle = theme.c('line'); ctx.lineWidth = 1; ctx.strokeRect(r.x + 0.5, r.y + 0.5, r.w - 1, r.h - 1); };
    // --- main plot
    const yMax = 1.25, X = (t) => main.x + t * main.w, Y = (u) => main.y + main.h / 2 - (u / yMax) * (main.h / 2);
    title(main, 'u(t): position over time');
    ctx.fillStyle = theme.c('ink', theme.dark ? 0.06 : 0.04);
    ctx.fillRect(X(0), main.y, X(P.tMax) - X(0), main.h);
    ctx.font = font(500, 10); ctx.fillStyle = theme.c('muted'); ctx.textAlign = 'center';
    // each region's label, only where it fits inside its region (measurements can end very early or very late)
    const band = (labels, a, b) => { const text = labels.find((s) => ctx.measureText(s).width + 8 <= X(b) - X(a)); if (text) ctx.fillText(text, X((a + b) / 2), main.y + 12); };
    band(['measured'], 0, P.tMax);
    band(['no data: prediction only', 'no data'], P.tMax, 1);
    frame(main);
    ctx.strokeStyle = theme.c('line-2'); ctx.beginPath(); ctx.moveTo(main.x, Y(0)); ctx.lineTo(main.x + main.w, Y(0)); ctx.stroke();
    ctx.fillStyle = theme.c('muted'); ctx.textAlign = 'right'; ctx.textBaseline = 'middle';
    for (const v of [-1, 0, 1]) ctx.fillText(String(v), main.x - 6, Y(v));
    ctx.textAlign = 'center'; ctx.textBaseline = 'top';
    for (const t of [0, 0.25, 0.5, 0.75, 1]) ctx.fillText(String(t), X(t), main.y + main.h + 4);
    const line = (arr, color, width, dash, Xf, Yf, n) => {
      ctx.save(); ctx.beginPath(); ctx.rect(main.x - 2, 0, W, Hh); ctx.clip();
      ctx.strokeStyle = color; ctx.lineWidth = width; ctx.setLineDash(dash || []); ctx.lineJoin = 'round';
      ctx.beginPath();
      for (let i = 0; i <= n; i++) { const x = Xf(i), y = Yf(arr[i]); i ? ctx.lineTo(x, y) : ctx.moveTo(x, y); }
      ctx.stroke(); ctx.restore();
    };
    const tx = (i) => X(i / cv.n), clampY = (u) => Y(clamp(u, -yMax * 1.4, yMax * 1.4));
    ctx.save(); ctx.beginPath(); ctx.rect(main.x, main.y, main.w, main.h); ctx.clip();
    if (show.exact) line(cv.exact.u, COL.exact(), 1.6, [5, 4], tx, clampY, cv.n);
    if (show.plain) line(cv.plain.u, COL.plain(), 2.2, null, tx, clampY, cv.n);
    if (show.pinn) line(cv.pinn.u, COL.pinn(), 2.4, null, tx, clampY, cv.n);
    ctx.restore();
    // measurements and collocation points
    for (const d of P.data) { ctx.beginPath(); ctx.arc(X(d.t), Y(d.u), 4, 0, Math.PI * 2); ctx.fillStyle = COL.data(); ctx.fill(); ctx.lineWidth = 1.5; ctx.strokeStyle = theme.c('panel'); ctx.stroke(); }
    if (this.terms.phys) { ctx.fillStyle = theme.c('muted'); for (const t of P.colloc) { const x = X(t), y = main.y + main.h; ctx.beginPath(); ctx.moveTo(x, y - 7); ctx.lineTo(x - 3, y - 1); ctx.lineTo(x + 3, y - 1); ctx.fill(); } }
    // probe
    const pt = this.probeT, pi = Math.round(pt * cv.n);
    ctx.strokeStyle = theme.c('ink', 0.35); ctx.setLineDash([2, 3]); ctx.beginPath(); ctx.moveTo(X(pt), main.y); ctx.lineTo(X(pt), main.y + main.h); ctx.stroke(); ctx.setLineDash([]);
    for (const [key, col] of [['plain', COL.plain()], ['pinn', COL.pinn()]]) if (show[key]) { ctx.beginPath(); ctx.arc(X(pt), clampY(cv[key].u[pi]), 4, 0, Math.PI * 2); ctx.fillStyle = col; ctx.fill(); }
    // --- residual (log of |r|)
    // its title stops short of the phase portrait's when the two sit side by side
    ctx.font = font(600, 11);
    const resRoom = lay.kind === 'below' ? ph.x - res.x - 12 : W - res.x - 4;
    title(res, ['|ODE residual| / ω²  (log scale)', '|residual| / ω²  (log scale)', '|residual| / ω²'].find((s) => ctx.measureText(s).width <= resRoom) || '|residual|');
    frame(res);
    const lo = -4, hi = 1.5, RY = (r) => res.y + res.h - ((clamp(Math.log10(Math.abs(r) + 1e-6), lo, hi) - lo) / (hi - lo)) * res.h, RX = (i) => res.x + (i / cv.n) * res.w;
    ctx.fillStyle = theme.c('muted'); ctx.font = font(500, 10); ctx.textAlign = 'right'; ctx.textBaseline = 'middle';
    // grid lines every 2 decades, or every 4 when the plot is too short for their labels to stay 14 px apart
    const decade = res.h / (hi - lo), ticks = decade * 2 >= 14 ? [-4, -2, 0] : decade * 4 >= 14 ? [-4, 0] : [0];
    for (const e of ticks) { ctx.fillText('1e' + e, res.x - 6, RY(Math.pow(10, e))); ctx.strokeStyle = theme.c('line'); ctx.beginPath(); ctx.moveTo(res.x, RY(Math.pow(10, e))); ctx.lineTo(res.x + res.w, RY(Math.pow(10, e))); ctx.stroke(); }
    ctx.fillStyle = theme.c('ink', theme.dark ? 0.06 : 0.04); ctx.fillRect(res.x, res.y, P.tMax * res.w, res.h);
    ctx.save(); ctx.beginPath(); ctx.rect(res.x, res.y, res.w, res.h); ctx.clip();
    const rline = (arr, col) => { ctx.strokeStyle = col; ctx.lineWidth = 1.8; ctx.beginPath(); for (let i = 0; i <= cv.n; i++) { const x = RX(i), y = RY(arr[i]); i ? ctx.lineTo(x, y) : ctx.moveTo(x, y); } ctx.stroke(); };
    if (show.plain) rline(cv.plain.r, COL.plain());
    if (show.pinn) rline(cv.pinn.r, COL.pinn());
    ctx.restore();
    // --- phase portrait: u against u′/ω
    if (!ph) {
      ctx.font = font(500, 10.5); ctx.fillStyle = theme.c('muted'); ctx.textAlign = 'left'; ctx.textBaseline = 'bottom';
      const notes = this.info && this.info.open ? ['The phase portrait needs more room: hide the explanation to see it.', 'Phase portrait: hide the explanation to see it.'] : ['The phase portrait needs a taller window.', 'Phase portrait: no room here.'];
      ctx.fillText(notes.find((s) => ctx.measureText(s).width <= W - 2 * PAD) || notes[1], PAD, Hh - BOT + 4);
      return;
    }
    ctx.font = font(600, 11);
    title(ph, ['Phase portrait: u against u′/ω', 'Phase: u against u′/ω', 'Phase: u vs u′/ω'].find((s) => ctx.measureText(s).width <= W - ph.x - 4) || 'Phase');
    frame(ph);
    const s = 1.25, PX = (u) => ph.x + ph.w / 2 + (u / s) * (ph.w / 2), PY = (v) => ph.y + ph.h / 2 - (v / s) * (ph.h / 2);
    ctx.strokeStyle = theme.c('line'); ctx.beginPath(); ctx.moveTo(ph.x, PY(0)); ctx.lineTo(ph.x + ph.w, PY(0)); ctx.moveTo(PX(0), ph.y); ctx.lineTo(PX(0), ph.y + ph.h); ctx.stroke();
    ctx.save(); ctx.beginPath(); ctx.rect(ph.x, ph.y, ph.w, ph.h); ctx.clip();
    const pline = (c, col, width, dash) => { ctx.strokeStyle = col; ctx.lineWidth = width; ctx.setLineDash(dash || []); ctx.beginPath(); for (let i = 0; i <= cv.n; i++) { const x = PX(c.u[i]), y = PY(c.u1[i] / P.omega); i ? ctx.lineTo(x, y) : ctx.moveTo(x, y); } ctx.stroke(); ctx.setLineDash([]); };
    if (show.exact) pline(cv.exact, COL.exact(), 1.4, [4, 3]);
    if (show.plain) pline(cv.plain, COL.plain(), 1.8);
    if (show.pinn) pline(cv.pinn, COL.pinn(), 2);
    ctx.restore();
    ctx.fillStyle = COL.data(); ctx.beginPath(); ctx.arc(PX(1), PY(0), 3.5, 0, Math.PI * 2); ctx.fill();
    ctx.font = font(500, 10); ctx.fillStyle = theme.c('muted'); ctx.textAlign = 'left'; ctx.textBaseline = 'bottom';
    ctx.fillText('start', PX(1) - 22, PY(0) - 5);
  };

  // ------------------------------------------------------------------ right: inspector
  PL.buildTab = function (id, body) {
    this.tabId = id; const P = (this.parts = {});
    if (id === 'overview') {
      body.append(UI.modelCard(this.preset, (t) => this.runTry(t)));
      P.findings = el('div', { class: 'diag-list' });
      body.append(UI.section('How the two networks compare', UI.note('Scored against the exact solution, which neither network ever sees.'), P.findings));
    } else if (id === 'loss') {
      const mk = (title, opts, legend, note) => { const c = el('canvas'), chart = new NA.charts.LineChart(c, opts); body.append(UI.section(title, el('div', { class: 'chart-box' }, c), legend, note ? UI.note(note) : null)); return chart; };
      const lg = (items) => el('div', { class: 'mini-legend' }, ...items.map(([n, col]) => el('span', { class: 'lg-item' }, el('i', { class: 'key-line', style: { background: col } }), n)));
      P.tiles = el('div', { class: 'tiles' });
      body.append(P.tiles);
      P.termsKey = el('div', { class: 'mini-legend' }); // filled in updateTab with the terms that are drawn
      P.terms = mk('PINN loss terms', { height: 150, logY: true, format: (v) => fmt(v, 5) }, P.termsKey, 'The data term wants the curve through the measurements; the physics term wants u″ + 2ζωu′ + ω²u = 0 at every collocation point. Early on they disagree; the solution that satisfies both is the true motion.');
      // in percent, like the readouts and the Overview; on a log scale, so 2% and 5% stay apart under a 300% start
      P.err = mk('Error against the exact solution', { height: 140, logY: true, format: (v) => fmtPct(v, 1), yFormat: (v) => fmtPct(v, v < 0.01 ? 1 : 0) }, lg([['plain network', 'var(--neg)'], ['PINN', 'var(--pos)']]), 'Relative L2 error over the whole window, including where there is no data.');
    } else if (id === 'deriv') {
      P.table = el('div', { class: 'kv-grid' });
      P.chain = el('div', { class: 'fx' });
      P.canvas = el('canvas');
      body.append(UI.section('The residual at the probe', P.table),
        UI.section('How the network differentiates itself', UI.note('Every layer carries three numbers per neuron: its value and its first and second derivative with respect to t. They are exact (no finite differences), and backpropagation runs through all three streams.'), P.chain),
        // section titles are set in capitals, which would turn u′ and u″ into U′ and U″: the title says it in words
        UI.section('Both derivatives across the window', el('div', { class: 'chart-box' }, P.canvas), el('div', { class: 'mini-legend' },
          el('span', { class: 'lg-item' }, el('i', { class: 'key-line', style: { background: 'var(--pos)' } }), 'PINN u′/ω'), el('span', { class: 'lg-item' }, el('i', { class: 'key-line', style: { background: 'var(--s7)' } }), 'PINN u″/ω²'), el('span', { class: 'lg-item' }, el('i', { class: 'key-line', style: { background: 'var(--ink-2)' } }), 'exact u′/ω (dashed)'))));
    } else if (id === 'net') {
      P.arch = el('div', { class: 'kv-grid' });
      P.canvas = el('canvas');
      body.append(UI.section('Architecture', P.arch), UI.section('Basis functions', UI.note('Each thin line is one neuron of the last hidden layer as t goes from 0 to 1, coloured by the sign of its output weight. The answer u(t) is their weighted sum plus a bias: tanh curves bent and shifted until they add up to a damped wave.'), el('div', { class: 'chart-box' }, P.canvas)));
    }
    this.updateTab();
  };
  PL.updateTab = function () {
    const id = this.tabId, P = this.parts, m = this.m;
    if (!P) return;
    if (id === 'overview') {
      const items = [];
      const f = (level, title, text) => items.push(el('div', { class: 'diag diag-' + level }, el('div', { class: 'diag-head' }, UI.pill(level, { good: 'Good', info: 'Note', warn: 'Watch', critical: 'Problem' }[level]), el('h4', { class: 'diag-title', text: title })), el('p', { class: 'diag-body', text })));
      const trained = this.step >= 1500;
      if (!trained) f('info', 'Still training', `Step ${fmtInt(this.step)}. Both networks usually settle within a few thousand steps.`);
      f(m.pinnErr < 0.1 ? 'good' : 'warn', `PINN: ${fmtPct(m.pinnErr, 1)} error`, `${fmtPct(m.pinnExtra, 1)} beyond the last measurement. ${m.pinnErr < 0.1 ? 'Following the equation lets it carry on correctly where there is no data.' : 'Not there yet: the physics term may still be pulling against the data.'}`);
      f(m.plainErr > 0.3 ? 'critical' : 'info', `Plain network: ${fmtPct(m.plainErr, 1)} error`, `${fmtPct(m.plainExtra, 1)} beyond the last measurement. ${m.plainErr > 0.3 ? 'It fits the measurements and then guesses: nothing tells it what happens after them.' : 'With measurements covering most of the window, data alone is enough.'}`);
      f('info', `Residual: PINN ${fmt(m.pinnRes, 3)}, plain ${fmt(m.plainRes, 2)}`, 'RMS of the equation’s left-hand side divided by ω². Zero means the curve obeys the physics exactly. The plain network was never asked to, and it shows.');
      clear(P.findings).append(...items);
    } else if (id === 'loss') {
      const last = this.last || this.history[this.history.length - 1] || {};
      clear(P.tiles).append(UI.tile('Step', fmtInt(this.step), this.pretrained ? 'trained offline, same code' : null), UI.tile('Data loss', fmt(last.data, 5)), UI.tile('Physics loss', fmt(last.phys, 5), `weight λ = ${fmt(this.lambda, 2)}`), UI.tile('IC loss', this.terms.ic ? fmt(last.ic, 5) : 'off'));
      const s = (key, name, color) => ({ name, color, points: this.history.filter((q) => q[key] > 0).map((q) => [q.step, q[key]]) });
      // a term that is switched off records nothing, so it is neither drawn nor listed in the key
      const terms = [['data', 'data', 'ink-2'], ['phys', 'physics', 'pos'], ['ic', 'initial conditions', 's4']].map(([key, name, tok]) => Object.assign(s(key, name, theme.c(tok)), { tok })).filter((x) => x.points.length);
      clear(P.termsKey).append(...terms.map((x) => el('span', { class: 'lg-item' }, el('i', { class: 'key-line', style: { background: `var(--${x.tok})` } }), x.name)));
      P.terms.frame(); P.terms.set({ series: terms });
      P.err.frame(); P.err.set({ series: [s('plainErr', 'plain network', theme.c('neg')), s('pinnErr', 'PINN', theme.c('pos'))] });
    } else if (id === 'deriv') this.renderDeriv(P);
    else if (id === 'net') this.renderNet(P);
  };
  PL.renderDeriv = function (P) {
    const Pr = this.prob, t = this.probeT, k = Pr.omega ** 2, c = 2 * Pr.zeta * Pr.omega, [u, u1, u2] = this.pinn.forward(t, 2), e = PN.exact(t, Pr.zeta, Pr.omega), e1 = PN.exactD(t, Pr.zeta, Pr.omega);
    clear(P.table).append(
      kv('t (probe)', fmt(t, 3)), kv('u', fmt(u, 4), `exact ${fmt(e, 4)}`), kv('u′', fmt(u1, 3), `exact ${fmt(e1, 3)}`), kv('u″', fmt(u2, 2)),
      kv('u″ + 2ζω·u′ + ω²·u', fmt(u2 + c * u1 + k * u, 3), `${fmt(u2, 2)} + ${fmt(c, 2)}×${fmt(u1, 3)} + ${fmt(k, 0)}×${fmt(u, 4)}; the physics loss squares this ÷ ω²`));
    clear(P.chain).append(...['input  x = 2t − 1,  x′ = 2,  x″ = 0', 'each layer  z = W·a + b,  z′ = W·a′,  z″ = W·a″', 'tanh  a = tanh z,  a′ = (1 − a²)·z′,  a″ = −2a(1 − a²)·z′² + (1 − a²)·z″', 'output  u, u′, u″ are the last layer’s z, z′, z″'].map((s, i) => el('div', { class: 'fx-line' + (i === 2 ? ' fx-strong' : '') , text: s })));
    const W = Math.max(220, P.canvas.parentElement.clientWidth), Hh = 150, ctx = NA.fitCanvas(P.canvas, W, Hh), cv = this.curves(), n = 160;
    ctx.clearRect(0, 0, W, Hh);
    ctx.strokeStyle = theme.c('line'); ctx.beginPath(); ctx.moveTo(0, Hh / 2); ctx.lineTo(W, Hh / 2); ctx.stroke();
    const Y = (v) => Hh / 2 - clamp(v, -1.4, 1.4) / 1.4 * (Hh / 2 - 6);
    const plot = (fn, col, dash) => { ctx.strokeStyle = col; ctx.lineWidth = 1.8; ctx.setLineDash(dash || []); ctx.beginPath(); for (let i = 0; i <= n; i++) { const x = (i / n) * W, y = Y(fn(i / n)); i ? ctx.lineTo(x, y) : ctx.moveTo(x, y); } ctx.stroke(); ctx.setLineDash([]); };
    plot((s) => PN.exactD(s, Pr.zeta, Pr.omega) / Pr.omega, theme.c('ink-2'), [4, 3]);
    plot((s) => this.pinn.forward(s, 2)[1] / Pr.omega, theme.c('pos'));
    plot((s) => this.pinn.forward(s, 2)[2] / k, theme.c('s7'));
    ctx.strokeStyle = theme.c('ink', 0.4); ctx.setLineDash([2, 3]); ctx.beginPath(); ctx.moveTo(t * W, 0); ctx.lineTo(t * W, Hh); ctx.stroke(); ctx.setLineDash([]);
    void cv;
  };
  PL.renderNet = function (P) {
    const net = this.pinn, sizes = net.sizes;
    clear(P.arch).append(kv('Layers', sizes.join(' → ')), kv('Activation', 'tanh (smooth, so u″ exists everywhere)'), kv('Parameters', fmtInt(net.paramCount), 'per network; the plain network has the same shape and starting weights'), kv('Optimizer', `Adam, learning rate ${this.lr}`));
    const W = Math.max(220, P.canvas.parentElement.clientWidth), Hh = 190, ctx = NA.fitCanvas(P.canvas, W, Hh), n = 120, Lh = net.L - 1, wOut = net.W[net.L - 1];
    ctx.clearRect(0, 0, W, Hh);
    ctx.strokeStyle = theme.c('line'); ctx.beginPath(); ctx.moveTo(0, Hh / 2); ctx.lineTo(W, Hh / 2); ctx.stroke();
    const vals = Array.from({ length: sizes[Lh] }, () => new Float32Array(n + 1)), out = new Float32Array(n + 1);
    for (let i = 0; i <= n; i++) { out[i] = net.forward(i / n, 0)[0]; const a = net.a[Lh][0]; for (let j = 0; j < sizes[Lh]; j++) vals[j][i] = a[j]; }
    let mw = 1e-6; for (const w of wOut) mw = Math.max(mw, Math.abs(w));
    vals.forEach((v, j) => {
      const w = wOut[j];
      ctx.strokeStyle = theme.divColor((w / mw) * 0.85 + Math.sign(w) * 0.15, 0.25 + 0.6 * Math.abs(w) / mw); ctx.lineWidth = 1;
      ctx.beginPath(); for (let i = 0; i <= n; i++) { const x = (i / n) * W, y = Hh / 2 - v[i] * (Hh / 2 - 8); i ? ctx.lineTo(x, y) : ctx.moveTo(x, y); } ctx.stroke();
    });
    ctx.strokeStyle = theme.c('ink'); ctx.lineWidth = 2.2;
    ctx.beginPath(); for (let i = 0; i <= n; i++) { const x = (i / n) * W, y = Hh / 2 - clamp(out[i], -1.2, 1.2) / 1.2 * (Hh / 2 - 8); i ? ctx.lineTo(x, y) : ctx.moveTo(x, y); } ctx.stroke();
  };

  PL.runTry = function (t) {
    if (t.action === 'physicsOnly') { this.terms = { data: false, phys: true, ic: true }; NA.$$('.chip', this.termChips).forEach((b, i) => b.setAttribute('aria-pressed', String([false, true, true][i]))); this.retrainPinn(); NA.ui.toast.show('PINN loss: physics + initial conditions only. It solves the equation without seeing a single measurement.'); }
    else if (t.action === 'lowDamping') { this.sliders.zeta.set(0.03); this.probOpts.zeta = 0.03; this.retrain('Less damping: the oscillation lasts longer, so extrapolating is harder.'); }
    else if (t.action === 'moreData') { this.sliders.tMax.set(1); this.sliders.nData.set(20); Object.assign(this.probOpts, { tMax: 1, nData: 20 }); this.retrain('Measurements across the whole window: now the plain network has data everywhere too.'); }
    else if (t.action === 'default') { this.terms = { data: true, phys: true, ic: false }; this.lambda = 1; this.probOpts = Object.assign({}, DEFAULT); this.pause(); this.ui.left.replaceChildren(); this.buildLeft(this.ui.left); this.setup({}, true); }
    this.tabs.select(this.tabId);
  };

  // Zoo glyph: a damped wave through a few dots.
  PL.glyph = function (ctx, W, Hh) {
    ctx.strokeStyle = theme.c('ink-2'); ctx.lineWidth = 1.4; ctx.beginPath();
    for (let i = 0; i <= 60; i++) { const t = i / 60, x = 4 + t * (W - 8), y = Hh / 2 - Math.exp(-2.2 * t) * Math.cos(15 * t) * (Hh / 2 - 6); i ? ctx.lineTo(x, y) : ctx.moveTo(x, y); }
    ctx.stroke();
    ctx.fillStyle = theme.c('ink');
    for (const t of [0, 0.1, 0.2]) { const x = 4 + t * (W - 8), y = Hh / 2 - Math.exp(-2.2 * t) * Math.cos(15 * t) * (Hh / 2 - 6); ctx.beginPath(); ctx.arc(x, y, 2.2, 0, Math.PI * 2); ctx.fill(); }
  };

  H.register(PL);
  NA.pinnLab = PL;
})(window.NA);
