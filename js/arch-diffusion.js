/* arch-diffusion.js — the diffusion model as a zoomable architecture: the training path (data → add noise →
   denoiser → loss) and the generating path (noise → the same denoiser → update step, repeated). The denoiser
   opens into its layers. One training point is followed through every part at the slider's noise step t, so
   each box shows that point's live numbers: its noise ε, the network's guess ε_θ, its loss and its update. */
(function (NA) {
  'use strict';
  const { fmt, fmtInt, theme, clamp } = NA;

  const pair = (p, d = 3) => `(${fmt(p[0], d)}, ${fmt(p[1], d)})`;
  const NAMES = { pinwheel: 'pinwheel', spiral: 'spiral', moons: 'two moons' };

  // ------------------------------------------------------------------ PyTorch versions
  const py = {};
  // the same three shapes as the page's datasets, so the code trains on what the page shows
  py.data = (L) => `import math
import torch

def make_points(name='${L.dataset}', n=${L.D.n}, noise=${(NA.diffusionWeights && NA.diffusionWeights[L.dataset] || {}).noise ?? 0.03}):
    i, t = torch.arange(n), torch.rand(n)
    if name == 'pinwheel':                                  # three arms
        r, a = 0.08 + 0.86 * t, t * 0.85 * 2 * math.pi + (i % 3) * 2 * math.pi / 3
        xy, sd = torch.stack([r * torch.cos(a), r * torch.sin(a)], 1), noise * 0.1
    elif name == 'spiral':                                  # two arms
        r, a = 0.06 + 0.88 * t, t * 1.6 * 2 * math.pi + (i % 2) * math.pi
        xy, sd = torch.stack([r * torch.cos(a), r * torch.sin(a)], 1), noise * 0.1
    else:                                                   # two moons
        s, inner = t * math.pi, (i % 2).bool()
        x = torch.where(inner, 1 - torch.cos(s), torch.cos(s))
        y = torch.where(inner, 0.5 - torch.sin(s), torch.sin(s))
        xy, sd = torch.stack([(x - 0.5) / 1.6, (y - 0.25) / 1.6], 1), noise * 0.22
    return (xy + sd * torch.randn(n, 2)).clamp(-1, 1)

points = make_points()                          # any (N, 2) points would do
data = points - points.mean(0)
data = data / data.pow(2).mean().sqrt()          # standardise: mean 0, spread 1`;
  py.schedule = (T) => `import math
import torch

T = ${T}
def cosine_schedule(T, s=0.008):
    f = lambda t: math.cos((t / T + s) / (1 + s) * math.pi / 2) ** 2
    betas = torch.tensor([min(0.5, 1 - f(t) / f(t - 1)) for t in range(1, T + 1)])
    return torch.cumprod(1 - betas, dim=0)          # abar[t - 1] = ᾱ_t: the signal's share of the variance

abar = cosine_schedule(T)
abar_at = lambda t: abar[t - 1] if t > 0 else torch.tensor(1.0)`;
  py.forward = () => `def add_noise(x0, t):
    """Jump straight to step t of the forward process: no need to add noise t times."""
    eps = torch.randn_like(x0)                      # the noise the network must learn to predict
    a = abar[t - 1].view(-1, 1)                     # t: (N,) steps from 1 to T
    xt = a.sqrt() * x0 + (1 - a).sqrt() * eps       # x_t = √ᾱ_t·x₀ + √(1 − ᾱ_t)·ε
    return xt, eps`;
  py.time = (T) => `def time_features(t, T=${T}):
    """t (N,) integer steps → (N, 9): 2t/T − 1, then sin and cos of π·2ᵏ·t/T for k = 0..3."""
    tau = (t.float() / T).unsqueeze(1)                                       # (N, 1)
    w = math.pi * (2.0 ** torch.arange(4)) * tau                             # (N, 4)
    waves = torch.stack([torch.sin(w), torch.cos(w)], dim=-1).flatten(1)     # sin, cos, sin, cos, ...
    return torch.cat([2 * tau - 1, waves], dim=1)`;
  py.net = (hidden) => `import torch.nn as nn

class Denoiser(nn.Module):
    """(noisy point, step) → the noise it thinks was added."""
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(
${hidden.map((h, i) => `            nn.Linear(${i ? hidden[i - 1] : '2 + 9'}, ${h}), nn.SiLU(),${i ? '' : '        # x, y and 9 time features'}`).join('\n')}
            nn.Linear(${hidden[hidden.length - 1]}, 2))                        # ε_θ for x and y

    def forward(self, xt, t):
        return self.net(torch.cat([xt, time_features(t)], dim=1))`;
  py.layer = (l, nIn, nOut) => `import torch.nn.functional as F

# hidden layer ${l + 1}: ${nIn} → ${nOut}, for a batch a of shape (N, ${nIn})
z = a @ W.T + b          # W: (${nOut}, ${nIn}), b: (${nOut},)
a = F.silu(z)            # SiLU(z) = z · sigmoid(z): smooth, lets small negatives through`;
  py.loss = (batch) => `def train_step(model, opt, data, batch=${batch}):
    x0 = data[torch.randint(len(data), (batch,))]       # real points
    t = torch.randint(1, T + 1, (batch,))               # a random noise level for each
    xt, eps = add_noise(x0, t)
    loss = ((model(xt, t) - eps) ** 2).mean()           # how wrong the noise guess is
    opt.zero_grad()
    loss.backward()
    opt.step()
    return loss.item()`;
  py.sampler = () => `@torch.no_grad()
def reverse_step(model, x, t, t_prev, ddim=False):
    """One step back from noise level t to t_prev for every point in x (N, 2)."""
    a, a_prev = abar_at(t), abar_at(t_prev)
    eps = model(x, torch.full((len(x),), t))
    x0 = ((x - (1 - a).sqrt() * eps) / a.sqrt()).clamp(-4, 4)   # best guess of the clean point
    eps = (x - a.sqrt() * x0) / (1 - a).sqrt()                    # the noise that guess implies
    if ddim or t_prev == 0:                                        # deterministic step
        return a_prev.sqrt() * x0 + (1 - a_prev).sqrt() * eps, x0
    beta = 1 - a / a_prev                                          # DDPM: sample q(x_prev | x_t, x0)
    mean = (a_prev.sqrt() * beta * x0 + (1 - beta).sqrt() * (1 - a_prev) * x) / (1 - a)
    var = beta * (1 - a_prev) / (1 - a)
    return mean + var.sqrt() * torch.randn_like(x), x0

@torch.no_grad()
def sample(model, n=600, steps=100, ddim=False):
    ts = [round(k * T / steps) for k in range(steps, -1, -1)]    # T, ..., 0
    x = torch.randn(n, 2)                                         # start from pure noise
    for t, t_prev in zip(ts[:-1], ts[1:]):
        x, _ = reverse_step(model, x, t, t_prev, ddim)
    return x`;
  py.model = (L) => `import math
import torch
import torch.nn as nn

${py.schedule(L.model.T).split('\n').slice(3).join('\n')}


${py.time(L.model.T)}


${py.net(L.model.o.hidden).split('\n').slice(2).join('\n')}


${py.forward()}


${py.loss(L.batch)}


${py.sampler()}


${py.data(L).split('\n').slice(3).join('\n')}
model = Denoiser()
opt = torch.optim.Adam(model.parameters(), lr=${L.lr})
for step in range(30_000):
    train_step(model, opt, data)
samples = sample(model, steps=${L.steps}, ddim=${L.sampler === 'ddim' ? 'True' : 'False'})`;

  // A training point to follow through the diagram: one on an arm, away from the centre.
  function followIndex(L) {
    if (L.followPt && L.followPt.D === L.D) return L.followPt.i;
    const D = L.D, r = [];
    for (let i = 0; i < D.n; i++) r.push([Math.hypot(D.pts[2 * i], D.pts[2 * i + 1]), i]);
    r.sort((a, b) => a[0] - b[0]);
    L.followPt = { D, i: r[Math.floor(r.length * 0.7)][1] };
    return L.followPt.i;
  }

  // ------------------------------------------------------------------ the tree
  function build(L, size) {
    const wide = size.w / size.h >= 1.05, W2 = (w, t) => (wide ? w : t);
    const m = L.model, net = m.net, S0 = m.sched, T = m.T, D = L.D, g = L.gen;
    const t = clamp(Math.round(L.tView), 1, T), stride = Math.max(1, Math.round(T / L.steps)), tPrev = Math.max(0, t - stride);
    // the followed point: a real x₀, its fixed noise draw ε, and x_t made from them
    const i0 = followIndex(L), x0 = [D.pts[2 * i0], D.pts[2 * i0 + 1]], e = [L.eps[2 * i0], L.eps[2 * i0 + 1]];
    const ab = S0.abar[t], abPrev = S0.abar[tPrev], sa = Math.sqrt(ab), sn = Math.sqrt(1 - ab);
    const xt = [sa * x0[0] + sn * e[0], sa * x0[1] + sn * e[1]];
    const out = m.predict(Float32Array.from(xt), 1, t), eh = [out[0], out[1]];
    // predict() leaves this point's activations in the network's buffers: copy them before anything else runs
    const acts = net.sizes.map((n, l) => Float32Array.from(net.a[l].subarray(0, n)));
    const zs = net.sizes.map((n, l) => (l ? Float32Array.from(net.z[l].subarray(0, n)) : null));
    const x0h = [clamp((xt[0] - sn * eh[0]) / sa, -4, 4), clamp((xt[1] - sn * eh[1]) / sa, -4, 4)];
    const epsC = [(xt[0] - sa * x0h[0]) / sn, (xt[1] - sa * x0h[1]) / sn];
    const lossPt = ((eh[0] - e[0]) ** 2 + (eh[1] - e[1]) ** 2) / 2;
    const beta = 1 - ab / abPrev, c0 = (Math.sqrt(abPrev) * beta) / (1 - ab), ct = (Math.sqrt(1 - beta) * (1 - abPrev)) / (1 - ab), sd = Math.sqrt((beta * (1 - abPrev)) / (1 - ab));
    const mean = [c0 * x0h[0] + ct * xt[0], c0 * x0h[1] + ct * xt[1]];
    const ddimNext = [Math.sqrt(abPrev) * x0h[0] + Math.sqrt(1 - abPrev) * epsC[0], Math.sqrt(abPrev) * x0h[1] + Math.sqrt(1 - abPrev) * epsC[1]];
    const feats = Array.from(acts[0].subarray(2));
    // below every number: NA.deep's levels, down to the transistors and electrons
    const X = { wide }, Dp = NA.deep, SILU = { id: 'swish', label: 'SiLU', f: (z) => z / (1 + Math.exp(-z)) };
    const FEAT = ['x', 'y', '2t/T − 1', ...[0, 1, 2, 3].flatMap((k) => [`sin(π·${2 ** k}·t/T)`, `cos(π·${2 ** k}·t/T)`])];
    const featNode = (i) => {
      if (i < 2) return Dp.stored(X, acts[0][i], `${FEAT[i]} of the noisy point`);
      if (i === 2) return Object.assign(Dp.fmul(X, 2, t / T), { title: '2t/T − 1', sub: `2 × ${fmt(t / T, 3)}, then − 1` });
      const k = (i - 3) >> 1, arg = Math.PI * 2 ** k * (t / T), cos = (i - 3) % 2 === 1;
      return Object.assign(Dp.lib(X, 'sine', { z: cos ? arg + Math.PI / 2 : arg }), { title: FEAT[i], sub: cos ? 'cos z = sin(z + π/2)' : `z = ${fmt(arg, 3)}` });
    };
    // neuron j of weight layer l, for the followed point
    const neuronOf = (l, j) => {
      const nIn = net.sizes[l], nOut = net.sizes[l + 1], last = l === net.L - 1;
      return Dp.neuron(X, { title: last ? `Output ε_θ[${j}]` : `Neuron ${j + 1} of layer ${l + 1}`, sub: last ? 'linear' : `${nIn} inputs, SiLU`,
        terms: Array.from({ length: nIn }, (_, i) => ({ a: acts[l][i], w: net.W[l][i * nOut + j] })), bias: net.b[l][j], act: last ? null : SILU,
        inLabel: (i) => (l === 0 ? FEAT[i] : `neuron ${i + 1} of layer ${l}`), wLabel: (i) => `W${l + 1}[${i}, ${j}]`,
        inputNode: (i) => (l === 0 ? featNode(i) : neuronOf(l - 1, i)),
        vars: { w: `W${l + 1}[:, ${j}]`, a: l === 0 ? 'features' : `a${l}`, b: `b${l + 1}[${j}]` },
        about: [`${last ? `Output ${j} of the denoiser, its guess of the noise in the ${j ? 'y' : 'x'} coordinate` : `Neuron ${j + 1} of hidden layer ${l + 1} of the denoiser`}, for the followed point at t = ${t}. Zoom into an input bar to open the neuron or feature that produced it.`] });
    };
    const side = [0.74, 0.0, 0.26, 0.24];

    // plotting helpers: a ±3 square of the plane
    const plane = (r) => { const s = Math.min(r.w, r.h) / 6; return { s, X: (x) => r.x + r.w / 2 + x * s, Y: (y) => r.y + r.h / 2 - y * s }; };
    const scatter = (ctx, pts, n, r, color, step = 1, rad = 1.2) => {
      const P = plane(r);
      ctx.fillStyle = color;
      for (let i = 0; i < n; i += step) { const xx = P.X(pts[2 * i]), yy = P.Y(pts[2 * i + 1]); if (xx > r.x && xx < r.x + r.w && yy > r.y && yy < r.y + r.h) ctx.fillRect(xx - rad / 2, yy - rad / 2, rad, rad); }
      return P;
    };
    const ring = (ctx, P, p, color, rad = 5) => { ctx.strokeStyle = color; ctx.lineWidth = 2; ctx.beginPath(); ctx.arc(P.X(p[0]), P.Y(p[1]), rad, 0, Math.PI * 2); ctx.stroke(); };
    const dot = (ctx, P, p, color, rad = 3.5) => { ctx.fillStyle = color; ctx.beginPath(); ctx.arc(P.X(p[0]), P.Y(p[1]), rad, 0, Math.PI * 2); ctx.fill(); };
    const tMark = (ctx, r) => { const x = r.x + (t / T) * r.w; ctx.strokeStyle = theme.c('ink', 0.55); ctx.setLineDash([2, 3]); ctx.lineWidth = 1; ctx.beginPath(); ctx.moveTo(x, r.y); ctx.lineTo(x, r.y + r.h); ctx.stroke(); ctx.setLineDash([]); };

    const dataNode = {
      id: 'data', title: 'Data x₀', sub: size.w * W2(0.17, 0.3) < 96 ? `n = ${fmtInt(D.n)}` : `${fmtInt(D.n)} points`, rect: W2([0, 0.02, 0.17, 0.3], [0, 0, 0.3, 0.16]),
      summary: (ctx, r) => { const P = scatter(ctx, D.pts, D.n, r, theme.c('ink-2'), 4, 1); dot(ctx, P, x0, theme.c('s7'), 2.5); },
      inner: (ctx, r, S) => {
        const q = S.place(r, [0, 0.02, 0.72, 0.86]), uu = S.u(r, 8, 13);
        S.frame(ctx, q);
        const P = scatter(ctx, D.pts, D.n, q, theme.c('pos', 0.75), 1, 1.8);
        ring(ctx, P, x0, theme.c('s7'), 6);
        S.text(ctx, `ringed: the point followed through the diagram, x₀ = ${pair(x0, 2)}`, r.x, r.y + r.h * 0.95, { size: uu * 0.9, color: theme.c('muted'), maxW: r.w });
      },
      about: [
        `The training data: ${fmtInt(D.n)} points of the ${NAMES[L.dataset] || L.dataset}, standardised so they have mean 0 and spread 1. The model never stores these points; it learns the distribution they come from, so it can produce new points that look like them.`,
        `The ringed point, x₀ = ${pair(x0, 2)}, is followed through the rest of the diagram: the Add noise box makes it noisy, the network guesses its noise, the loss box scores the guess and the sampler box takes one step back with it.`,
      ],
      code: py.data(L),
      children: [Object.assign(Dp.stored(X, x0[0], 'x₀, x coordinate'), { id: 'x0', rect: [0.76, 0.3, 0.24, 0.3] })],
    };

    const fwdNode = {
      id: 'forward', title: 'Add noise', sub: 'x₀ → x_t', rect: W2([0.205, 0.02, 0.2, 0.3], [0.35, 0, 0.3, 0.16]), tone: 's2',
      summary: (ctx, r, S) => {
        S.curve(ctx, (v) => Math.sqrt(S0.abar[Math.round(v)]), r, 0, T, 0, 1, { color: theme.c('pos'), lw: 1.4 });
        S.curve(ctx, (v) => Math.sqrt(1 - S0.abar[Math.round(v)]), r, 0, T, 0, 1, { color: theme.c('neg'), lw: 1.4 });
        tMark(ctx, r);
      },
      inner: (ctx, r, S) => {
        const uu = S.u(r, 9, 14);
        S.lines(ctx, [
          'x_t = √ᾱ_t · x₀ + √(1 − ᾱ_t) · ε,     ε ~ N(0, I)',
          `t = ${t}:  √ᾱ = ${fmt(sa, 3)},  √(1 − ᾱ) = ${fmt(sn, 3)}`,
          `x_t = ${fmt(sa, 3)}·${pair(x0, 2)} + ${fmt(sn, 3)}·${pair(e, 2)} = ${pair(xt, 2)}`,
        ], S.place(r, [0, 0.02, 0.7, 0.22]), { mono: true, size: uu, strong: (i) => i === 0 });
        const pr = S.place(r, W2([0, 0.34, 0.46, 0.56], [0, 0.3, 1, 0.24])), qr = S.place(r, W2([0.52, 0.3, 0.48, 0.66], [0.1, 0.6, 0.8, 0.4]));
        S.caption(ctx, `signal kept √ᾱ_t (blue) and noise √(1 − ᾱ_t) (orange), t = 0 … ${T}`, pr, { size: uu * 0.8, minW: 200 });
        S.frame(ctx, pr);
        S.curve(ctx, (v) => Math.sqrt(S0.abar[Math.round(v)]), pr, 0, T, 0, 1, { color: theme.c('pos'), lw: 2 });
        S.curve(ctx, (v) => Math.sqrt(1 - S0.abar[Math.round(v)]), pr, 0, T, 0, 1, { color: theme.c('neg'), lw: 2 });
        tMark(ctx, pr);
        const sq = { x: qr.x + (qr.w - Math.min(qr.w, qr.h)) / 2, y: qr.y, w: Math.min(qr.w, qr.h), h: qr.h };
        S.frame(ctx, sq);
        scatter(ctx, D.pts, D.n, sq, theme.c('muted', 0.35), 2, 1.2);
        const P = scatter(ctx, m.noised(D.pts, L.eps, D.n, t), D.n, sq, theme.c('pos', 0.6), 2, 1.5);
        // the followed point's path as t goes from 0 to the slider's t: a quarter-ellipse toward its noise point
        ctx.strokeStyle = theme.c('s7'); ctx.lineWidth = 1.6; ctx.beginPath();
        for (let s = 0; s <= t; s++) { const a = Math.sqrt(S0.abar[s]), b = Math.sqrt(1 - S0.abar[s]), x = P.X(a * x0[0] + b * e[0]), y = P.Y(a * x0[1] + b * e[1]); s ? ctx.lineTo(x, y) : ctx.moveTo(x, y); }
        ctx.stroke();
        ring(ctx, P, x0, theme.c('s7'), 4); dot(ctx, P, xt, theme.c('s7'), 3.5);
        S.caption(ctx, `the data at t = ${t}; violet: x₀ → x_t`, sq, { size: uu * 0.8, minW: 160 });
      },
      about: [
        `The forward process destroys data on purpose. At step t a point is scaled by √ᾱ_t and Gaussian noise of size √(1 − ᾱ_t) is added. As t goes from 0 to ${T}, ᾱ_t falls from 1 to almost 0, so the signal fades and the noise takes over; at t = ${T} only a round Gaussian cloud is left.`,
        'Any step can be reached in one jump from the clean point, because a sum of Gaussian noises is itself Gaussian. Training relies on this: pick a random point, a random t and a random ε, make x_t in one line, and ask the network to recover ε.',
        `With its noise draw ε fixed, the followed point slides along a curve (a quarter of an ellipse) from x₀ toward ε as t grows; the violet line is that path up to t = ${t}. The data’s share of the variance (ᾱ_t) and the noise’s share (1 − ᾱ_t) always add up to 1, so the cloud’s overall spread stays the same at every t.`,
      ],
      code: `${py.schedule(T)}


${py.forward()}

xt, eps = add_noise(data, torch.full((len(data),), ${t}))   # the whole dataset at t = ${t}`,
      children: [Object.assign(Dp.fmul(X, sa, x0[0]), { id: 'mul', rect: side, title: '√ᾱ × x₀', sub: `${fmt(sa, 3)} × ${fmt(x0[0], 3)}` })],
    };

    const timeNode = {
      id: 'time', title: 'Time features', sub: 't → 9 numbers', rect: W2([0.205, 0.36, 0.2, 0.28], [0.7, 0, 0.3, 0.16]), tone: 's3',
      summary: (ctx, r, S) => S.bars(ctx, feats, r, { max: 1 }),
      grids: [{ id: 'f', rect: [0, 0.8, 1, 0.18], rows: 1, cols: 9, make: (i, j) => featNode(j + 2) }],
      inner: (ctx, r, S) => {
        const uu = S.u(r, 9, 14);
        S.lines(ctx, ['f(t) = [ 2t/T − 1,  sin(π·2ᵏ·t/T),  cos(π·2ᵏ·t/T)  for k = 0…3 ]', `t = ${t}:  [ ${feats.map((v) => fmt(v, 2)).join(', ')} ]`], S.place(r, [0, 0.02, 1, 0.14]), { mono: true, size: uu * 0.95, strong: (i) => i === 0 });
        const pr = S.place(r, [0, 0.24, 1, 0.48]);
        S.caption(ctx, `the 9 features as t goes from 0 to ${T}`, pr, { size: uu * 0.8 });
        S.frame(ctx, pr);
        for (let f = 0; f < 9; f++) {
          S.curve(ctx, (v) => { const tau = v / T; if (f === 0) return 2 * tau - 1; const q = (f - 1) >> 1, w = Math.PI * Math.pow(2, q) * tau; return (f - 1) % 2 === 0 ? Math.sin(w) : Math.cos(w); }, pr, 0, T, -1.1, 1.1, { color: theme.series[f % 8], lw: 1.3 });
        }
        tMark(ctx, pr);
        const br = S.place(r, [0, 0.8, 1, 0.18]);
        S.caption(ctx, `their values at t = ${t}`, br, { size: uu * 0.8 });
        S.bars(ctx, feats, br, { max: 1 });
      },
      about: [
        'The network has to behave very differently at high noise, where the best guess is "toward the middle", and at low noise, where it should only make small corrections. It is told the step t as nine numbers: t itself, scaled to [−1, 1], and sines and cosines of t at four frequencies.',
        'The slow waves tell it roughly how noisy the input is; the fast ones let it tell nearby steps apart. Image diffusion models use the same trick (sinusoidal embeddings, as in transformers) with hundreds of frequencies.',
      ],
      code: py.time(T),
    };

    const layerRect = (l) => { const n = net.L, w = 0.88 / n; return W2([0.12 + l * w, 0.12, w - 0.04, 0.78], [0.04, 0.11 + l * (0.84 / n), 0.92, 0.84 / n - 0.04]); };
    const layerNode = (l) => {
      const nIn = net.sizes[l], nOut = net.sizes[l + 1], last = l === net.L - 1, a = acts[l + 1], z = zs[l + 1];
      const LR = { a: W2([0, 0.08, 0.62, 0.3], [0, 0.06, 1, 0.22]), z: W2([0, 0.52, 0.62, 0.3], [0, 0.36, 1, 0.22]), w: last ? W2([0, 0.52, 0.62, 0.2], [0, 0.4, 1, 0.2]) : W2([0.7, 0.56, 0.3, 0.4], [0.52, 0.66, 0.48, 0.3]) };
      return {
        id: 'l' + l, title: last ? 'Output ε_θ' : `Layer ${l + 1}`, sub: `${nIn} → ${nOut}${last ? '' : ' SiLU'}`, tone: last ? 's7' : null, rect: layerRect(l),
        summary: (ctx, r, S) => S.bars(ctx, a, r),
        grids: [
          { id: 'a', rect: LR.a, rows: 1, cols: nOut, make: (i, j) => neuronOf(l, j) },
          ...(last ? [] : [{ id: 'z', rect: LR.z, rows: 1, cols: nOut, make: (i, j) => neuronOf(l, j) }]),
          { id: 'W', rect: LR.w, rows: nOut, cols: nIn, make: (i, j) => Dp.stored(X, net.W[l][j * nOut + i], `W${l + 1}[${j}, ${i}]`) },
        ],
        inner: (ctx, r, S) => {
          const uu = S.u(r, 9, 14);
          const ar = S.place(r, LR.a), zr = S.place(r, LR.z);
          S.caption(ctx, last ? `ε_θ = W·a + b: the predicted noise ${pair(a)}; the true ε was ${pair(e)}` : `a = SiLU(z) = z·σ(z): the ${nOut} outputs`, ar, { size: uu * 0.85, minW: 220 });
          S.bars(ctx, a, ar);
          if (!last) {
            S.caption(ctx, `z = W·a + b: the ${nOut} pre-activations`, zr, { size: uu * 0.85, minW: 200 });
            S.bars(ctx, z, zr);
            const cr = S.place(r, W2([0.7, 0.08, 0.3, 0.34], [0, 0.66, 0.45, 0.3]));
            S.caption(ctx, 'SiLU(z) for z from −5 to 5', cr, { size: uu * 0.85 });
            S.frame(ctx, cr);
            S.curve(ctx, (v) => v / (1 + Math.exp(-v)), cr, -5, 5, -0.6, 5, { color: theme.c('s1'), lw: 2 });
          }
          const wr = S.place(r, LR.w);
          S.caption(ctx, `weights W (${nOut} × ${nIn})`, wr, { size: uu * 0.85 });
          S.heat(ctx, (i, j) => net.W[l][j * nOut + i], nOut, nIn, wr);
        },
        about: last
          ? [`The last layer is linear, so its two outputs can be any size: the network’s guess ε_θ = ${pair(a)} of the noise mixed into the followed point (the true noise was ${pair(e)}). Everything else, the loss and the sampler, is built from this guess.`]
          : [
            `Hidden layer ${l + 1}: ${nOut} neurons, each computing z = w·a + b from the ${nIn} numbers below it and passing it through SiLU(z) = z·σ(z), a smooth cousin of ReLU that lets small negative values through. The bars show both for the followed point at t = ${t}.`,
            'The weight matrix W is drawn with one row per neuron and one column per input; blue weights are positive and orange ones negative.',
          ],
        code: last ? 'eps_hat = a @ W.T + b      # (N, 2): the predicted noise, no activation' : py.layer(l, nIn, nOut),
      };
    };

    const netNode = {
      id: 'net', title: 'Denoiser', sub: `${net.L - 1} × ${net.sizes[1]} SiLU`, rect: W2([0.45, 0.02, 0.21, 0.96], [0, 0.22, 1, 0.4]), tone: 's1',
      summary: (ctx, r) => {
        const cols = net.sizes.length, w = r.w / cols;
        net.sizes.forEach((n, i) => { const hh = Math.min(r.h, 6 + n * 1.1); ctx.fillStyle = theme.c(i === 0 || i === cols - 1 ? 'ink-2' : 's1', 0.7); ctx.fillRect(r.x + w * i + w * 0.3, r.y + (r.h - hh) / 2, w * 0.4, hh); });
      },
      grids: [{ id: 'in', rect: W2([0, 0.2, 0.08, 0.5], [0.25, 0.01, 0.5, 0.06]), rows: 1, cols: acts[0].length, make: (i, j) => featNode(j) }],
      inner: (ctx, r, S) => {
        const uu = S.u(r, 9, 14);
        const inR = S.place(r, W2([0, 0.2, 0.08, 0.5], [0.25, 0.01, 0.5, 0.06]));
        S.caption(ctx, 'input', inR, { size: uu * 0.8 });
        S.bars(ctx, acts[0], inR);
        S.text(ctx, `following x_t = ${pair(xt, 2)} at t = ${t}`, r.x, r.y + r.h * 0.985, { size: uu * 0.85, color: theme.c('muted'), maxW: r.w });
        const rs = net.W.map((_, l) => S.place(r, layerRect(l)));
        if (wide) S.arrow(ctx, inR.x + inR.w + 3, inR.y + inR.h / 2, rs[0].x - 3, inR.y + inR.h / 2, { lw: 1.1, head: 5 });
        else S.arrow(ctx, inR.x + inR.w / 2, inR.y + inR.h + 3, inR.x + inR.w / 2, rs[0].y - 3, { lw: 1.1, head: 5 });
        for (let i = 0; i + 1 < rs.length; i++) {
          if (wide) S.arrow(ctx, rs[i].x + rs[i].w + 2, rs[i].y + rs[i].h / 2, rs[i + 1].x - 3, rs[i].y + rs[i].h / 2, { lw: 1.1, head: 5 });
          else S.arrow(ctx, rs[i].x + rs[i].w / 2, rs[i].y + rs[i].h + 2, rs[i].x + rs[i].w / 2, rs[i + 1].y - 3, { lw: 1.1, head: 5 });
        }
      },
      children: net.W.map((_, l) => layerNode(l)),
      about: [
        `The denoiser is a plain MLP, ${net.sizes.join(' → ')}, with ${fmtInt(net.paramCount)} parameters. Its input is a noisy point (2 numbers) and the step t (as 9 time features); its output is its guess of the noise ε that was added. That one guess is all the model ever produces.`,
        `Here it sees the followed point x_t = ${pair(xt, 2)} at t = ${t} and answers ε_θ = ${pair(eh)} (the true noise was ${pair(e)}). Knowing the noise is the same as knowing which way to move a point to make it look more like the data: the arrows in the Denoiser view are this network’s answers across the whole plane.`,
        'Image generators do the same job with a U-Net or a transformer of hundreds of millions of parameters, but the training and sampling around the network are unchanged.',
      ],
      code: `${py.time(T)}


${py.net(m.o.hidden)}`,
    };

    const lossNode = {
      id: 'loss', title: 'Training loss', sub: '‖ε − ε_θ‖²', rect: W2([0.735, 0.02, 0.265, 0.44], [0, 0.68, 0.48, 0.32]), tone: 's4',
      summary: (ctx, r) => { const b = L.lossBands(), mx = Math.max(...b, 1e-6), bw = r.w / 10; b.forEach((v, i) => { const h = (v / mx) * r.h; ctx.fillStyle = theme.c('s4', 0.75); ctx.fillRect(r.x + i * bw + 1, r.y + r.h - h, bw - 2, h); }); },
      inner: (ctx, r, S) => {
        const uu = S.u(r, 9, 14), b = L.lossBands(), mx = Math.max(...b, 1e-6), band = Math.min(9, Math.floor(((t - 1) / T) * 10));
        S.lines(ctx, [
          'loss = mean of (ε − ε_θ)² over x and y, and over a batch',
          `followed point, t = ${t}:  ε = ${pair(e)}   ε_θ = ${pair(eh)}`,
          `its loss = ((${fmt(eh[0] - e[0], 3)})² + (${fmt(eh[1] - e[1], 3)})²) / 2 = ${fmt(lossPt, 4)}`,
        ], S.place(r, [0, 0.02, 0.7, 0.22]), { mono: true, size: uu * 0.92, strong: (i) => i === 0 });
        const br = S.place(r, [0.04, 0.36, 0.92, 0.5]), bw = br.w / 10;
        S.caption(ctx, `average loss in ten bands of t (fresh points and noise); violet: the band of t = ${t}`, br, { size: uu * 0.8, minW: 220 });
        b.forEach((v, i) => { const h = (v / mx) * br.h; ctx.fillStyle = theme.c(i === band ? 's7' : 'pos', 0.75); ctx.fillRect(br.x + i * bw + 2, br.y + br.h - h, bw - 4, h); S.text(ctx, fmt(v, 2), br.x + (i + 0.5) * bw, br.y + br.h - h - uu * 0.6, { size: uu * 0.7, align: 'center', color: theme.c('muted') }); });
        S.text(ctx, `t = 1 … ${T} →`, br.x, br.y + br.h + uu * 0.9, { size: uu * 0.8, color: theme.c('muted') });
      },
      about: [
        'Training is plain regression: take a real point, add a known amount of known noise, and penalise the squared difference between the true noise ε and the network’s guess ε_θ. Every step uses fresh random points, noise levels and noise, so the network sees every noise level of every part of the shape.',
        'The bars show the loss at each noise level. At high noise a point is almost all noise, so ε ≈ x_t is easy to guess and the loss is near zero. It peaks at low-to-middle noise, where data and noise are mixed about evenly. It never reaches zero: when little noise was added, nobody can tell exactly which part of a point is noise.',
      ],
      code: `${py.forward()}


${py.loss(L.batch)}`,
      children: [Object.assign(Dp.fmul(X, eh[0] - e[0], eh[0] - e[0]), { id: 'sq', rect: side, title: '(ε_θ − ε)² for x', sub: `${fmt(eh[0] - e[0], 3)} squared` })],
    };

    const sampNode = {
      id: 'sampler', title: 'Sampler step', sub: `${L.sampler === 'ddim' ? 'DDIM' : 'DDPM'}, ${L.steps} steps`, rect: W2([0.735, 0.54, 0.265, 0.44], [0.52, 0.68, 0.48, 0.32]), tone: 's7',
      summary: (ctx, r, S) => S.text(ctx, `t → t − ${stride}`, r.x + r.w / 2, r.y + r.h * 0.35, { size: clamp(r.w / 9, 7, 13), mono: true, align: 'center', maxW: r.w }),
      inner: (ctx, r, S) => {
        const uu = S.u(r, 9, 14);
        S.lines(ctx, [
          `following x_t = ${pair(xt)} from t = ${t} to ${tPrev}`,
          `ε_θ = ${pair(eh)}                 the network’s guess`,
          `x̂₀ = (x_t − √(1 − ᾱ_t)·ε_θ) / √ᾱ_t = ${pair(x0h)}   (true x₀ ${pair(x0, 2)})`,
          '',
          `DDPM:  x_${tPrev} = c₀·x̂₀ + c_t·x_t + σ·z,   z ~ N(0, I)`,
          `       c₀ = ${fmt(c0, 3)}   c_t = ${fmt(ct, 3)}   σ = ${fmt(sd, 3)}`,
          `       mean ${pair(mean)}, then add σ·z`,
          `DDIM:  x_${tPrev} = √ᾱ_${tPrev}·x̂₀ + √(1 − ᾱ_${tPrev})·ε_θ   (no new noise)`,
          `       = ${fmt(Math.sqrt(abPrev), 3)}·x̂₀ + ${fmt(Math.sqrt(1 - abPrev), 3)}·ε_θ = ${pair(ddimNext)}`,
        ], S.place(r, [0, 0.02, 0.7, 0.96]), { mono: true, size: uu * 0.92, strong: (i) => i === 2 || i === 4 || i === 7 });
      },
      about: [
        `Generating runs the process backwards. Start from pure noise; at each step ask the network for ε_θ, turn it into an estimate x̂₀ of the clean point, and move to an earlier step. With ${L.steps} steps each one goes from t to t − ${stride}.`,
        'DDPM draws the earlier point from a Gaussian whose mean is a blend of x̂₀ and the current point, then adds fresh noise σ·z; DDIM takes the deterministic step with no new noise, so the same start always lands in the same place.',
        `The numbers follow the ringed training point at t = ${t}. At high t, c₀ is small and the step mostly keeps the current point; near t = 0, c₀ grows and the point moves onto its estimate. The Generate view shows the whole journey for ${g.n} points at once.`,
      ],
      code: `${py.schedule(T).split('\n').slice(3).join('\n')}


${py.sampler()}`,
      children: [
        Object.assign(Dp.fmul(X, sn, eh[0]), { id: 'x0', rect: [0.74, 0.02, 0.26, 0.29], title: '√(1 − ᾱ) × ε_θ', sub: 'for x̂₀' }),
        Object.assign(Dp.fmul(X, c0, x0h[0]), { id: 'ddpm', rect: [0.74, 0.36, 0.26, 0.29], title: 'c₀ × x̂₀', sub: 'DDPM mean' }),
        Object.assign(Dp.fmul(X, Math.sqrt(abPrev), x0h[0]), { id: 'ddim', rect: [0.74, 0.7, 0.26, 0.28], title: '√ᾱ × x̂₀', sub: 'DDIM step' }),
      ],
    };

    const noiseNode = {
      id: 'noise', title: 'Noise x_T', sub: 'N(0, I)', rect: [0, 0.68, 0.17, 0.3],
      summary: (ctx, r) => { const rng = new NA.RNG(4), pts = new Float32Array(400); for (let i = 0; i < 400; i++) pts[i] = rng.gauss(); scatter(ctx, pts, 200, r, theme.c('ink-2'), 1, 1); },
      inner: (ctx, r) => { const rng = new NA.RNG(4), pts = new Float32Array(1200); for (let i = 0; i < 1200; i++) pts[i] = rng.gauss(); scatter(ctx, pts, 600, { x: r.x, y: r.y, w: r.w * 0.7, h: r.h }, theme.c('ink-2'), 1, 2); },
      children: [Object.assign(Dp.stored(X, e[0], 'one noise number, ε for x'), { id: 'e', rect: [0.74, 0.3, 0.26, 0.3] })],
      about: [`Generating starts here: points drawn from a standard Gaussian, the same round cloud the forward process ends in at t = ${T}. Each one is a seed; the sampler turns it into a new data point. A random-number generator makes them: uniform random bits from a counter-based generator, turned into Gaussian numbers by a formula such as Box–Muller.`],
      code: 'import torch\n\nn = 600\nx = torch.randn(n, 2)      # x_T: where every generated point starts\nprint(x.mean(0), x.std(0))  # about 0 and 1',
    };

    const kids = wide ? [dataNode, fwdNode, timeNode, noiseNode, netNode, lossNode, sampNode] : [dataNode, fwdNode, timeNode, netNode, lossNode, sampNode];
    return {
      id: 'diff', title: 'Diffusion model', sub: `${fmtInt(net.paramCount)} parameters · ${T} noise steps`,
      inner: (ctx, r, S) => {
        const at = (n) => S.place(r, n.rect), dn = at(dataNode), fw = at(fwdNode), tm = at(timeNode), nt = at(netNode), ls = at(lossNode), sp = at(sampNode), uu = S.u(r, 8, 12);
        const lab = (s, x, y, align = 'center') => S.text(ctx, s, x, y, { size: uu * 0.95, mono: true, color: theme.c('muted'), align });
        if (wide) {
          const nz = at(noiseNode), my = (q) => q.y + q.h / 2;
          S.arrow(ctx, dn.x + dn.w + 2, my(dn), fw.x - 3, my(fw), { lw: 1.2, head: 5 });
          S.arrow(ctx, fw.x + fw.w + 2, my(fw), nt.x - 3, my(fw), { lw: 1.2, head: 5 });
          lab('x_t', (fw.x + fw.w + nt.x) / 2, my(fw) - uu);
          S.arrow(ctx, tm.x + tm.w + 2, my(tm), nt.x - 3, my(tm), { lw: 1.2, head: 5 });
          lab('t', (tm.x + tm.w + nt.x) / 2, my(tm) - uu);
          S.arrow(ctx, nz.x + nz.w + 2, my(nz), nt.x - 3, my(nz), { lw: 1.2, head: 5, color: theme.c('s7') });
          lab('x_T', (nz.x + nz.w + nt.x) / 2, my(nz) - uu);
          S.arrow(ctx, nt.x + nt.w + 2, my(ls), ls.x - 3, my(ls), { lw: 1.2, head: 5 });
          lab('ε_θ', (nt.x + nt.w + ls.x) / 2, my(ls) - uu);
          S.arrow(ctx, nt.x + nt.w + 2, sp.y + sp.h * 0.3, sp.x - 3, sp.y + sp.h * 0.3, { lw: 1.2, head: 5, color: theme.c('s7') });
          lab('ε_θ', (nt.x + nt.w + sp.x) / 2, sp.y + sp.h * 0.3 - uu);
          S.arrow(ctx, sp.x - 3, sp.y + sp.h * 0.78, nt.x + nt.w + 3, sp.y + sp.h * 0.78, { lw: 1.2, head: 5, bend: [0, 14], color: theme.c('s7') });
          lab('x_t−1', (nt.x + nt.w + sp.x) / 2, sp.y + sp.h * 0.78 + uu * 2.1);
        } else {
          S.arrow(ctx, dn.x + dn.w + 2, dn.y + dn.h / 2, fw.x - 3, fw.y + fw.h / 2, { lw: 1.2, head: 5 });
          S.arrow(ctx, fw.x + fw.w / 2, fw.y + fw.h + 2, fw.x + fw.w / 2, nt.y - 3, { lw: 1.2, head: 5 });
          S.arrow(ctx, tm.x + tm.w / 2, tm.y + tm.h + 2, tm.x + tm.w / 2, nt.y - 3, { lw: 1.2, head: 5 });
          S.arrow(ctx, ls.x + ls.w / 2, nt.y + nt.h + 2, ls.x + ls.w / 2, ls.y - 3, { lw: 1.2, head: 5 });
          S.arrow(ctx, sp.x + sp.w / 2, nt.y + nt.h + 2, sp.x + sp.w / 2, sp.y - 3, { lw: 1.2, head: 5, color: theme.c('s7') });
          lab('ε_θ', ls.x + ls.w / 2 + uu, (nt.y + nt.h + ls.y) / 2, 'left'); lab('ε_θ', sp.x + sp.w / 2 + uu, (nt.y + nt.h + sp.y) / 2, 'left');
        }
      },
      children: kids,
      about: [
        `A denoising diffusion model learns to undo noise. Training takes a real point, adds a random amount of noise with the forward process, and teaches the denoiser network to predict exactly which noise was added. Generating (violet arrows) starts from pure noise and applies the same network over and over, removing a little predicted noise at each of the ${L.steps} steps, until a new point that looks like the data comes out.`,
        'The network never outputs a data point directly; it only ever answers "which noise is in this point at this noise level?". Image generators such as Stable Diffusion run the same two loops, with a far bigger network and images in place of 2-D points.',
        `One training point is followed through every box at noise step t = ${t}; move the slider below the diagram to change t. ${NA.tapWord()} or zoom into a box for its maths with that point’s numbers, and switch to the Python tab for its code.`,
      ],
      code: py.model(L),
      codeLabel: 'Python · PyTorch · the whole model, training and sampling',
    };
  }

  NA.archDiffusion = { build, py, followIndex };
})(window.NA);
