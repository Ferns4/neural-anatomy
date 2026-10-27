/* arch-pinn.js — the physics-informed network as a zoomable architecture: input scaling → tanh layers that
   carry the value and its first and second derivatives → u, u′, u″ → the ODE residual → the loss. A hidden
   layer opens into its three streams, and a neuron into the chain rule with numbers, at the probe time. */
(function (NA) {
  'use strict';
  const { fmt, fmtInt, theme, clamp } = NA;
  const PN = NA.pinn;
  const num = (v, d) => fmt(v, d).replace('−', '-');

  // ------------------------------------------------------------------ PyTorch versions
  const py = {};
  py.model = (L) => `import torch
import torch.nn as nn

zeta, omega = ${num(L.prob.zeta, 2)}, ${num(L.prob.omega, 0)}.0      # damping and natural frequency

class PINN(nn.Module):
    def __init__(self, width=${L.width}, depth=${L.depth}):
        super().__init__()
        layers, d_in = [], 1
        for _ in range(depth):
            layers += [nn.Linear(d_in, width), nn.Tanh()]   # tanh is smooth, so u'' exists everywhere
            d_in = width
        layers.append(nn.Linear(d_in, 1))
        self.net = nn.Sequential(*layers)

    def forward(self, t):                    # t: (N, 1) times in [0, 1]
        return self.net(2 * t - 1)           # scale to [-1, 1] first


def residual(model, t):
    """How far u(t) is from obeying u'' + 2·zeta·omega·u' + omega²·u = 0, divided by omega²."""
    t = t.clone().requires_grad_(True)
    u = model(t)
    du = torch.autograd.grad(u, t, torch.ones_like(u), create_graph=True)[0]
    d2u = torch.autograd.grad(du, t, torch.ones_like(du), create_graph=True)[0]
    return (d2u + 2 * zeta * omega * du + omega ** 2 * u) / omega ** 2


def initial_conditions(model):
    """u(0) = 1 and u'(0) = 0: released from 1, at rest."""
    t0 = torch.zeros(1, 1, requires_grad=True)
    u0 = model(t0)
    du0 = torch.autograd.grad(u0, t0, torch.ones_like(u0), create_graph=True)[0]
    return ((u0 - 1) ** 2 + (du0 / omega) ** 2).sum()


model = PINN()
opt = torch.optim.Adam(model.parameters(), lr=${L.lr})
t_data = torch.tensor([${L.prob.data.map((d) => num(d.t, 3)).join(', ')}]).view(-1, 1)
u_data = torch.tensor([${L.prob.data.map((d) => num(d.u, 3)).join(', ')}]).view(-1, 1)
t_col = ((torch.arange(${L.prob.nColloc}) + 0.5) / ${L.prob.nColloc}).view(-1, 1)   # collocation points
w_data, lam, w_ic = ${num(L.weights().data, 2)}, ${num(L.weights().phys, 2)}, ${num(L.weights().ic, 2)}   # loss weights, as set on this page

for step in range(10_000):
    if step == 6000:
        opt.param_groups[0]['lr'] *= 0.3                            # the page drops the rate to 30% here
    loss_data = ((model(t_data) - u_data) ** 2).mean()              # fit the measurements
    loss_phys = (residual(model, t_col) ** 2).mean()                # obey the equation
    loss = w_data * loss_data + lam * loss_phys + w_ic * initial_conditions(model)
    opt.zero_grad()
    loss.backward()                                                 # differentiates through u' and u'' too
    opt.step()

# The plain network is the same model trained with w_data, lam, w_ic = 1, 0, 0.`;
  py.input = () => `def scale(t):
    """Map t in [0, 1] to x in [-1, 1], where tanh layers work best."""
    return 2 * t - 1        # so dx/dt = 2 and d²x/dt² = 0: the derivative streams start at (2, 0)`;
  py.layer = (L, l) => `import torch

def tanh_layer(a, da, d2a, W, b):
    """Hidden layer ${l + 1}, carrying the value and its first and second derivative in t
    (forward-mode differentiation, which is what this page computes).
    a, da, d2a: (N, ${l === 0 ? 1 : L.width}) values and derivatives coming in; W: (${L.width}, ${l === 0 ? 1 : L.width}); b: (${L.width},)"""
    z = a @ W.T + b              # the usual pre-activation
    dz = da @ W.T                # derivatives pass through the linear part unchanged
    d2z = d2a @ W.T
    y = torch.tanh(z)
    s1 = 1 - y ** 2              # tanh'(z)
    s2 = -2 * y * s1             # tanh''(z)
    return y, s1 * dz, s2 * dz ** 2 + s1 * d2z     # chain rule, then the product rule

# With PyTorch's autograd you would instead write model(t) and call torch.autograd.grad twice
# (see the whole-model code): reverse mode, same exact numbers.`;
  py.neuron = (L, l, j, n) => {
    const rows = [[`z   = W[${j}] @ a + b[${j}]`, n.z], [`dz  = W[${j}] @ da`, n.z1], [`d2z = W[${j}] @ d2a`, n.z2], ['y   = torch.tanh(z)', n.y], ['dy  = (1 - y**2) * dz', n.y1], ['d2y = -2*y*(1 - y**2) * dz**2 + (1 - y**2) * d2z', n.y2]];
    return `# Neuron ${j + 1} of hidden layer ${l + 1} at t = ${num(L.probeT, 3)}. W, b: the layer's weights and biases;
# a, da, d2a: the values and derivatives coming in from the layer below
${rows.map(([c, v]) => `${c.padEnd(50)}# = ${num(v, 4)}`).join('\n')}`;
  };
  py.output = () => `u, du, d2u = W_out @ a + b_out, W_out @ da, W_out @ d2a   # linear output: no squashing`;
  py.residual = (L) => `def ode_residual(u, du, d2u, zeta=${fmt(L.prob.zeta, 2)}, omega=${fmt(L.prob.omega, 0)}.0):
    """Left-hand side of the oscillator equation, divided by omega² so it is of order 1."""
    return (d2u + 2 * zeta * omega * du + omega ** 2 * u) / omega ** 2

# zero for any curve that obeys the physics; checked at ${L.prob.nColloc} collocation points in [0, 1]`;
  py.loss = (L) => `# model, residual, initial_conditions and the data as in the whole-model code (top level, Python tab)
loss_data = ((model(t_data) - u_data) ** 2).mean()         # ${L.prob.data.length} measurements
loss_phys = (residual(model, t_col) ** 2).mean()           # ${L.prob.nColloc} collocation points
loss_ic = initial_conditions(model)                        # (u(0) - 1)² + (u'(0) / omega)², optional
loss = w_data * loss_data + lam * loss_phys + w_ic * loss_ic

loss.backward()   # gradients flow back through u, u' and u'' to every weight`;

  // ------------------------------------------------------------------ the tree
  function build(L, size) {
    const wide = size.w / size.h >= 1.05, W2 = (w, t) => (wide ? w : t);
    const net = L.pinn, P = L.prob, t = L.probeT, sizes = net.sizes, depth = net.L - 1, k = P.omega ** 2, c = 2 * P.zeta * P.omega;
    const w = L.weights();
    let ld = 0, lp = 0;
    for (const d of P.data) { const e = net.forward(d.t, 0)[0] - d.u; ld += e * e; }
    for (const tc of P.colloc) { const [a, b1, b2] = net.forward(tc, 2), q = (b2 + c * b1 + k * a) / k; lp += q * q; }
    const [u0, u01] = net.forward(0, 2);
    const last = { data: ld / Math.max(1, P.data.length), phys: lp / Math.max(1, P.colloc.length), ic: (u0 - 1) ** 2 + (u01 / P.omega) ** 2 };
    last.total = w.data * last.data + w.phys * last.phys + w.ic * last.ic;
    // all three streams at the probe time, layer by layer
    net.forward(t, 2);
    const streams = net.a.map((s) => s.map((arr) => Float32Array.from(arr)));
    const zd = net.zd.map((s) => s.map((arr) => Float32Array.from(arr)));
    const [u, u1, u2] = [streams[net.L][0][0], streams[net.L][1][0], streams[net.L][2][0]], r = (u2 + c * u1 + k * u) / k;
    const ex = PN.exact(t, P.zeta, P.omega), ex1 = PN.exactD(t, P.zeta, P.omega);
    // below every number: NA.deep's levels, down to the transistors and electrons
    const X = { wide }, Dp = NA.deep, TANH = { id: 'tanh', label: 'tanh', f: Math.tanh }, side = [0.68, 0.03, 0.32, 0.36];
    const sumOf = (l, j, q, o = {}) => Dp.neuron(X, Object.assign({
      terms: Array.from({ length: sizes[l] }, (_, i) => ({ a: streams[l][q][i], w: net.W[l][j * sizes[l] + i] })),
      inLabel: (i) => (l === 0 ? ['x', 'dx/dt', 'd²x/dt²'][q] : `${['a', 'a′', 'a″'][q]} of neuron ${i + 1}, layer ${l}`), wLabel: (i) => `W${l + 1}[${j}, ${i}]`,
      inputNode: (i) => (l === 0 ? Dp.stored(X, streams[0][q][0], ['x = 2t − 1', 'dx/dt', 'd²x/dt²'][q]) : neuronNode(l - 1, i)),
      vars: { w: `W${l + 1}[${j}]`, a: l === 0 ? 'x' : `${['a', 'da', 'd2a'][q]}${l}`, b: `b${l + 1}[${j}]` },
    }, o));

    const inputNode = {
      id: 'input', title: 'Input t', sub: 'scaled to [−1, 1]', rect: W2([0, 0.38, 0.11, 0.24], [0.25, 0, 0.5, 0.09]),
      summary: (ctx, rr, S) => S.text(ctx, fmt(t, 3), rr.x + rr.w / 2, rr.y + rr.h * 0.4, { size: clamp(rr.w / 5, 7, 12), mono: true, align: 'center', maxW: rr.w }),
      children: [Object.assign(Dp.fmul(X, 2, t), { id: 'scale', rect: side, title: '2 × t', sub: 'the first multiplication' })],
      inner: (ctx, rr, S) => {
        const uu = S.u(rr, 9, 15);
        S.lines(ctx, [`t = ${fmt(t, 3)}   (the probe; move it on the Plots view)`, `x = 2t − 1 = ${fmt(2 * t - 1, 3)}`, 'dx/dt = 2', 'd²x/dt² = 0'], S.place(rr, [0, 0.05, 0.64, 0.4]), { mono: true, size: uu, strong: (i) => i === 1 });
        const pr = S.place(rr, [0, 0.55, 1, 0.4]);
        S.frame(ctx, pr);
        S.curve(ctx, (v) => 2 * v - 1, pr, 0, 1, -1.1, 1.1, { color: theme.c('pos'), lw: 2 });
        S.text(ctx, 'x against t', pr.x + 6, pr.y + 10, { size: uu * 0.8, color: theme.c('muted') });
      },
      about: ['The input is a single number, the time t between 0 and 1. It is first mapped to x = 2t − 1, so it lies in [−1, 1], where tanh units are most sensitive.', 'Because the network has to provide derivatives with respect to t, it carries three numbers forward instead of one: the value, its first derivative and its second derivative. For the input they are x, 2 and 0.'],
      code: py.input(),
    };

    const neuronNode = (l, j) => {
      const prev = streams[l], Wl = net.W[l], nIn = sizes[l], b = net.b[l][j];
      let z = b; for (let i = 0; i < nIn; i++) z += Wl[j * nIn + i] * prev[0][i];
      const n = { z, z1: zd[l + 1][0][j], z2: zd[l + 1][1][j], y: streams[l + 1][0][j], y1: streams[l + 1][1][j], y2: streams[l + 1][2][j] };
      const NB = W2({ value: [0.65, 0.02, 0.35, 0.3], slope: [0.65, 0.36, 0.35, 0.28], chain: [0.65, 0.68, 0.35, 0.3] }, { value: [0, 0.72, 0.32, 0.27], slope: [0.34, 0.72, 0.32, 0.27], chain: [0.68, 0.72, 0.32, 0.27] });
      return Dp.lazy({
        id: 'neuron', title: `Neuron ${j + 1}`, sub: 'the chain rule in numbers', rect: W2([0.7, 0.62, 0.3, 0.36], [0, 0.76, 1, 0.22]),
        summary: (ctx, rr, S) => S.text(ctx, `y = ${fmt(n.y, 3)}`, rr.x + rr.w / 2, rr.y + rr.h * 0.35, { size: clamp(rr.w / 9, 7, 12), mono: true, align: 'center', maxW: rr.w }),
        inner: (ctx, rr, S) => {
          const uu = S.u(rr, 9, 15);
          S.lines(ctx, [
            `z   = w·a + b                    = ${fmt(n.z, 4)}`,
            `z′  = w·a′                       = ${fmt(n.z1, 4)}`,
            `z″  = w·a″                       = ${fmt(n.z2, 4)}`,
            `y   = tanh z                     = ${fmt(n.y, 4)}`,
            `y′  = (1 − y²)·z′                = ${fmt(n.y1, 4)}`,
            `y″  = −2y(1 − y²)·z′² + (1 − y²)·z″ = ${fmt(n.y2, 4)}`,
          ], S.place(rr, W2([0, 0.02, 0.6, 0.5], [0, 0.02, 1, 0.36])), { mono: true, size: uu, strong: (i) => i === 3 });
          // the neuron's output as a function of t
          const pr = S.place(rr, W2([0, 0.6, 0.6, 0.36], [0, 0.42, 1, 0.24]));
          S.frame(ctx, pr);
          const vals = []; for (let i = 0; i <= 80; i++) { net.forward(i / 80, 0); vals.push(net.a[l + 1][0][j]); }
          net.forward(t, 2);
          S.curve(ctx, (v) => vals[Math.round(v * 80)], pr, 0, 1, -1.05, 1.05, { color: theme.c('pos'), lw: 2 });
          ctx.strokeStyle = theme.c('ink', 0.5); ctx.setLineDash([2, 3]); ctx.beginPath(); ctx.moveTo(pr.x + t * pr.w, pr.y); ctx.lineTo(pr.x + t * pr.w, pr.y + pr.h); ctx.stroke(); ctx.setLineDash([]);
          S.text(ctx, `neuron ${j + 1} over t from 0 to 1; its slope at the probe is y′, its curvature y″`, pr.x + 6, pr.y + 10, { size: uu * 0.8, color: theme.c('muted'), maxW: pr.w - 12 });
        },
        about: [`One tanh neuron of hidden layer ${l + 1} at t = ${fmt(t, 3)}, with all three streams. The pre-activation z and its derivatives are weighted sums of the layer below; the activation y = tanh z gets its derivatives from the chain rule: y′ = tanh′(z)·z′, and the product rule adds a second term to y″ because z′ itself changes with t.`, 'These are exact derivatives, not finite differences: the same numbers PyTorch would get by differentiating twice with autograd. The boxes open the value\'s weighted sum, the slope\'s weighted sum and one chain-rule multiplication, each down to the hardware.'],
        code: py.neuron(L, l, j, n),
      }, () => [
        Object.assign(sumOf(l, j, 0, { title: `Neuron ${j + 1}: value`, sub: 'z = W·a + b, then tanh', bias: net.b[l][j], act: TANH, outName: 'y', about: [`The value stream of neuron ${j + 1} in hidden layer ${l + 1}: the weighted sum of the layer below, plus a bias, through tanh.`] }), { id: 'value', rect: NB.value }),
        Object.assign(sumOf(l, j, 1, { title: `Neuron ${j + 1}: slope`, sub: 'z′ = W·a′ (no bias)', bias: null, act: null, zName: 'dz', outName: 'dz', about: [`The first-derivative stream: the same weights applied to the slopes a′ of the layer below. The bias is a constant, so it has no slope and drops out.`] }), { id: 'slope', rect: NB.slope }),
        Object.assign(Dp.fmul(X, 1 - n.y * n.y, n.z1), { id: 'chain', rect: NB.chain, title: 'Chain rule: y′ = (1 − y²) × z′', sub: `${fmt(1 - n.y * n.y, 3)} × ${fmt(n.z1, 3)}` }),
      ]);
    };

    const layerNode = (l) => {
      const s = streams[l + 1], width = sizes[l + 1];
      // the neuron that changes most at the probe opens one level deeper
      let jm = 0; for (let j = 1; j < width; j++) if (Math.abs(s[1][j]) > Math.abs(s[1][jm])) jm = j;
      const rect = W2([0.15 + l * (0.4 / depth), 0.1, 0.4 / depth - 0.03, 0.8], [0.03, 0.12 + l * (0.42 / depth), 0.94, 0.42 / depth - 0.025]);
      const rows = W2([[0, 0.08, 0.66, 0.14], [0, 0.32, 0.66, 0.14], [0, 0.56, 0.66, 0.14]], [[0, 0.06, 1, 0.11], [0, 0.25, 1, 0.11], [0, 0.44, 1, 0.11]]);
      return {
        id: 'l' + l, title: `Hidden ${l + 1}`, sub: `${width} tanh`, rect, tone: 's1',
        summary: (ctx, rr, S) => { if (wide) { const n = width, h = rr.h / n; for (let j = 0; j < n; j++) { ctx.fillStyle = theme.divColor(s[0][j]); ctx.fillRect(rr.x + rr.w * 0.3, rr.y + j * h, rr.w * 0.4, h * 0.8); } } else S.bars(ctx, s[0], rr, { max: 1 }); },
        // any bar opens its neuron; any weight opens the number stored in memory
        grids: [
          ...rows.map((f, q) => ({ id: 'abc'[q], rect: f, rows: 1, cols: width, make: (i, j) => neuronNode(l, j) })),
          ...(wide ? [{ id: 'W', rect: [0.72, 0.08, 0.28, 0.44], rows: width, cols: sizes[l], make: (i, jj) => Dp.stored(X, net.W[l][i * sizes[l] + jj], `weight W${l + 1}[${i}, ${jj}]`) }] : []),
        ],
        inner: (ctx, rr, S) => {
          const uu = S.u(rr, 8, 14), names = ['value a = tanh z', 'first derivative a′ = da/dt', 'second derivative a″ = d²a/dt²'];
          rows.forEach((f, q) => { const br = S.place(rr, f); S.caption(ctx, names[q], br, { size: uu * 0.85, minW: 200 }); S.bars(ctx, s[q], br); });
          S.lines(ctx, ['z = W·a + b,   z′ = W·a′,   z″ = W·a″', 'a = tanh z,   a′ = (1 − a²)·z′', 'a″ = −2a(1 − a²)·z′² + (1 − a²)·z″'], S.place(rr, W2([0, 0.76, 0.66, 0.2], [0, 0.6, 1, 0.14])), { mono: true, size: uu * 0.9 });
          if (wide) {
            const wr = S.place(rr, [0.72, 0.08, 0.28, 0.44]);
            S.caption(ctx, `weights W (${width} × ${sizes[l]})`, wr, { size: uu * 0.85 });
            S.heat(ctx, (i, jj) => net.W[l][i * sizes[l] + jj], width, sizes[l], wr);
          }
        },
        children: [neuronNode(l, jm)],
        about: [
          `Hidden layer ${l + 1}: ${width} tanh neurons. Each computes z = w·a + b from the layer below and outputs tanh z, like any MLP layer, but here every neuron also carries how its output changes with t (a′) and how that change changes (a″). The three bar rows show all three streams at t = ${fmt(t, 3)}.`,
          'The derivative streams pass through the weights unchanged (z′ = W·a′) and are bent by tanh through the chain rule. That is why a smooth activation is essential: with ReLU, the second derivative would be zero almost everywhere and the physics loss could not see curvature.',
          `Open the neuron box to see the chain rule with numbers for the neuron whose output changes fastest at the probe (neuron ${jm + 1}).`,
        ],
        code: py.layer(L, l),
      };
    };

    const outputNode = {
      id: 'out', title: 'Output u', sub: 'with u′ and u″', rect: W2([0.555, 0.38, 0.11, 0.24], [0.25, 0.56, 0.5, 0.09]),
      summary: (ctx, rr, S) => S.text(ctx, fmt(u, 3), rr.x + rr.w / 2, rr.y + rr.h * 0.4, { size: clamp(rr.w / 5, 7, 12), mono: true, align: 'center', maxW: rr.w }),
      children: [Object.assign(sumOf(depth, 0, 0, { title: 'Output u: weighted sum', sub: 'u = W·a + b (linear)', bias: net.b[depth][0], act: null, zName: 'u', outName: 'u', about: ['The output neuron: the last hidden layer’s values times the output weights, plus a bias, with no activation.'] }), { id: 'sum', rect: side })],
      inner: (ctx, rr, S) => {
        const uu = S.u(rr, 9, 15);
        S.lines(ctx, [`at t = ${fmt(t, 3)}:`, `u   = ${fmt(u, 4)}    exact ${fmt(ex, 4)}`, `u′  = ${fmt(u1, 3)}    exact ${fmt(ex1, 3)}`, `u″  = ${fmt(u2, 2)}`], S.place(rr, [0, 0.04, 0.64, 0.34]), { mono: true, size: uu, strong: (i) => i === 1 });
        const pr = S.place(rr, [0, 0.45, 1, 0.5]);
        S.frame(ctx, pr);
        const vals = []; for (let i = 0; i <= 120; i++) vals.push(net.forward(i / 120, 0)[0]);
        net.forward(t, 2);
        S.curve(ctx, (v) => PN.exact(v, P.zeta, P.omega), pr, 0, 1, -1.25, 1.25, { color: theme.c('ink-2'), dash: [4, 3], lw: 1.4 });
        S.curve(ctx, (v) => vals[Math.round(v * 120)], pr, 0, 1, -1.25, 1.25, { color: theme.c('pos'), lw: 2 });
      },
      about: ['The output layer is linear: u = W·a + b, and the same weights give u′ = W·a′ and u″ = W·a″. So one pass of the network yields the position, the velocity and the acceleration at once.', 'Dashed is the exact solution, which training never sees; solid is the PINN.'],
      code: py.output(),
    };

    const residualNode = {
      id: 'res', title: 'ODE residual', sub: 'how far from the physics', rect: W2([0.72, 0.04, 0.28, 0.42], [0, 0.7, 0.48, 0.3]), tone: 's7',
      summary: (ctx, rr, S) => S.text(ctx, 'u″ + 2ζω·u′ + ω²·u', rr.x + rr.w / 2, rr.y + rr.h * 0.35, { size: clamp(rr.w / 14, 6, 12), mono: true, align: 'center', maxW: rr.w }),
      inner: (ctx, rr, S) => {
        const uu = S.u(rr, 9, 15);
        S.lines(ctx, ['r = (u″ + 2ζω·u′ + ω²·u) / ω²', `  = (${fmt(u2, 2)} + ${fmt(c, 2)}×${fmt(u1, 3)} + ${fmt(k, 0)}×${fmt(u, 4)}) / ${fmt(k, 0)}`, `  = ${fmt(r, 4)}   at t = ${fmt(t, 3)}`], S.place(rr, [0, 0.04, 0.64, 0.3]), { mono: true, size: uu, strong: (i) => i === 0 });
        const pr = S.place(rr, [0, 0.45, 1, 0.5]);
        S.frame(ctx, pr);
        const lo = -4, hi = 1.5, rv = [];
        for (let i = 0; i <= 120; i++) { const [a, b1, b2] = net.forward(i / 120, 2); rv.push((b2 + c * b1 + k * a) / k); }
        net.forward(t, 2);
        S.curve(ctx, (v) => Math.log10(Math.abs(rv[Math.round(v * 120)]) + 1e-6), pr, 0, 1, lo, hi, { color: theme.c('pos'), lw: 1.6 });
        ctx.fillStyle = theme.c('muted');
        for (const tc of P.colloc) { const x = pr.x + tc * pr.w, y = pr.y + pr.h; ctx.beginPath(); ctx.moveTo(x, y - 6); ctx.lineTo(x - 3, y - 1); ctx.lineTo(x + 3, y - 1); ctx.fill(); }
        S.text(ctx, '|r| over t, log scale; ▲ collocation points', pr.x + 6, pr.y + 10, { size: uu * 0.8, color: theme.c('muted') });
      },
      about: [`The physics: a damped spring obeys u″ + 2ζω·u′ + ω²·u = 0 (here ζ = ${fmt(P.zeta, 2)}, ω = ${fmt(P.omega, 0)}). Plug in the network's own u, u′ and u″ and whatever is left over is the residual r: zero for a curve that obeys the physics, large for one that does not. Dividing by ω² keeps it of order 1.`, `It is checked at ${P.nColloc} collocation points spread over the whole window, including where there are no measurements. That is the point of a PINN: the equation supervises the network where data cannot.`],
      code: py.residual(L),
      children: [Object.assign(Dp.fmul(X, c, u1), { id: 'mul', rect: side, title: '2ζω × u′', sub: `${fmt(c, 2)} × ${fmt(u1, 3)}` })],
    };

    const lossNode = {
      id: 'loss', title: 'Loss', sub: 'data + λ·physics', rect: W2([0.72, 0.54, 0.28, 0.42], [0.52, 0.7, 0.48, 0.3]), tone: 's4',
      summary: (ctx, rr, S) => S.text(ctx, `${fmt(last.total, 5)}`, rr.x + rr.w / 2, rr.y + rr.h * 0.35, { size: clamp(rr.w / 10, 7, 12), mono: true, align: 'center', maxW: rr.w }),
      inner: (ctx, rr, S) => {
        const uu = S.u(rr, 9, 15);
        const lines = [
          `L = ${w.data}·data + ${fmt(w.phys, 2)}·physics + ${w.ic}·IC`,
          `data     = mean (u(tᵢ) − yᵢ)²   = ${fmt(last.data, 6)}   (${P.data.length} measurements)`,
          `physics  = mean r(tⱼ)²          = ${fmt(last.phys, 6)}   (${P.nColloc} points)`,
          `IC       = (u(0) − 1)² + (u′(0)/ω)² = ${fmt(last.ic, 6)}`,
          `total    = ${fmt(last.total, 6)}`,
          '',
          `plain network: data only, error ${fmt(L.m.plainErr * 100, 1)}% vs PINN ${fmt(L.m.pinnErr * 100, 1)}%`,
        ];
        S.lines(ctx, lines, S.place(rr, [0, 0.04, 0.64, 0.92]), { mono: true, size: uu * 0.95, strong: (i) => i === 0 || i === 4 });
      },
      about: [
        `Training minimises a weighted sum: the data term pulls the curve through the ${P.data.length} measurements, the physics term pushes the residual toward zero at the ${P.nColloc} collocation points, and an optional initial-condition term pins u(0) = 1 and u′(0) = 0. The weights (here λ = ${fmt(L.lambda, 2)} for physics) set how hard each one pulls.`,
        'Backpropagation runs through all three streams: a weight affects u, u′ and u″, and the gradient of the physics term includes how changing it would change the derivatives. Early in training the terms disagree; the curve that satisfies both is the true motion. The plain network on the Plots view is trained on the data term alone.',
      ],
      code: py.loss(L),
      children: [Object.assign(Dp.fmul(X, r, r), { id: 'sq', rect: [0.68, 0.3, 0.32, 0.36], title: 'r² at the probe', sub: `${fmt(r, 4)} × ${fmt(r, 4)}` })],
    };

    const layers = Array.from({ length: depth }, (_, l) => layerNode(l));
    const kids = [inputNode, ...layers, outputNode, residualNode, lossNode];
    return {
      id: 'pinn', title: 'Physics-informed network', sub: `${fmtInt(net.paramCount)} parameters`,
      inner: (ctx, rr, S) => {
        const rs = kids.map((kk) => S.place(rr, kk.rect));
        const chain = rs.slice(0, depth + 2);
        for (let i = 0; i + 1 < chain.length; i++) {
          const a = chain[i], b = chain[i + 1];
          if (wide) S.arrow(ctx, a.x + a.w + 2, a.y + a.h / 2, b.x - 3, a.y + a.h / 2, { lw: 1.2, head: 5 });
          else S.arrow(ctx, a.x + a.w / 2, a.y + a.h + 2, a.x + a.w / 2, b.y - 3, { lw: 1.2, head: 5 });
        }
        const out = rs[depth + 1], res = rs[depth + 2], loss = rs[depth + 3], uu = S.u(rr, 8, 12);
        if (wide) {
          S.arrow(ctx, out.x + out.w + 2, out.y + out.h * 0.3, res.x - 3, res.y + res.h / 2, { lw: 1.2, head: 5, bend: [0, -12] });
          S.arrow(ctx, out.x + out.w + 2, out.y + out.h * 0.7, loss.x - 3, loss.y + loss.h * 0.3, { lw: 1.2, head: 5, bend: [0, 12] });
          S.arrow(ctx, res.x + res.w / 2, res.y + res.h + 2, loss.x + loss.w / 2, loss.y - 3, { lw: 1.2, head: 5 });
          S.text(ctx, 'u, u′, u″', out.x + out.w / 2, out.y - uu, { size: uu, mono: true, color: theme.c('muted'), align: 'center' });
          S.text(ctx, '3 streams: value, d/dt, d²/dt²', chain[1].x, chain[1].y + chain[1].h + uu * 1.2, { size: uu * 0.9, color: theme.c('muted'), maxW: rr.w * 0.5 });
        } else {
          S.arrow(ctx, out.x + out.w * 0.3, out.y + out.h + 2, res.x + res.w / 2, res.y - 3, { lw: 1.2, head: 5 });
          S.arrow(ctx, out.x + out.w * 0.7, out.y + out.h + 2, loss.x + loss.w / 2, loss.y - 3, { lw: 1.2, head: 5 });
          S.arrow(ctx, res.x + res.w + 2, res.y + res.h / 2, loss.x - 3, loss.y + loss.h / 2, { lw: 1.2, head: 5 });
        }
      },
      children: kids,
      about: [
        `A small tanh network (1 → ${sizes.slice(1, -1).join(' → ')} → 1) that maps a time t to a position u(t). What makes it physics-informed is the loss: besides fitting the measurements, it is penalised wherever its own derivatives break the oscillator equation u″ + 2ζω·u′ + ω²·u = 0.`,
        'To check the equation the network needs u′ and u″ at any t. This page carries them through every layer alongside the value (forward-mode differentiation); PyTorch would usually get them by differentiating twice with autograd. Either way they are exact, and training differentiates through them too.',
        'Open a hidden layer to see its three streams, and a neuron to see the chain rule with numbers. The residual and loss boxes show the physics and the objective at the probe time.',
      ],
      code: py.model(L),
      codeLabel: 'Python · PyTorch · the whole model and training',
    };
  }

  NA.archPinn = { build, py };
})(window.NA);
