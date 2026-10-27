/* arch-radar.js — the radar classifier as a zoomable architecture: the I/Q input, five convolution layers, the
   average over time, the linear layer and the softmax, drawn with the live numbers of the signal on screen. Any
   cell of a feature map opens the neuron that computed it; its inputs open the neurons (or raw samples) below it,
   and NA.deep takes every number on down to the arithmetic, the gates and the electrons. */
(function (NA) {
  'use strict';
  const { fmt, fmtInt, fmtPct, theme, clamp } = NA;
  const RD = NA.radar;
  const dB = (v) => `${v > 0.05 ? '+' : v < -0.05 ? '−' : ''}${fmt(Math.abs(v), Math.abs(v - Math.round(v)) < 0.05 ? 0 : 1)} dB`;
  const SHORT = ['Noise', 'Pulse', 'LFM', 'Barker', 'Frank', 'Costas'];
  const CCOL = ['muted', 's3', 's7', 's4', 's5', 's8'];

  // ------------------------------------------------------------------ PyTorch versions
  const py = {};
  // the same generator as tools/train-radar.py, so the code trains on what the page shows
  py.gen = `import math
import torch

N = 256                                         # samples in the window: 12.8 µs at 20 MS/s
CLASSES = ['Noise only', 'Pulse', 'LFM chirp', 'Barker-13', 'Frank code', 'Costas FSK']
BARKER = [1, 1, 1, 1, 1, -1, -1, 1, 1, -1, 1, -1, 1]
COSTAS = [[pow(g, i + s, 7) for i in range(6)] for g in (3, 5) for s in range(6)]   # Welch Costas arrays

def generate(n, cls=None, snr=None):
    """n random intercepts: x (n, 2, N) holds I and Q; labels (n,); SNR in dB (n,)."""
    U = lambda: torch.rand(n)
    y = torch.full((n,), cls) if cls is not None else torch.randint(0, 6, (n,))
    snr_db = torch.full((n,), float(snr)) if snr is not None else -12 + 32 * U()
    L = torch.floor(96 + U() * 129)                 # pulse length: 96 to 224 samples
    n0 = torch.floor(U() * (N - L + 1))             # start: the whole pulse is inside the window
    f0 = (2 * U() - 1) * 0.12                       # carrier offset, cycles per sample (up to 2.4 MHz)
    ph0 = 2 * math.pi * U()
    bw = 0.05 + 0.2 * U()                           # chirp bandwidth (1 to 5 MHz)
    up = torch.where(U() < 0.5, -1.0, 1.0)
    M = 4 + torch.floor(3 * U())                    # Frank code of M x M chips
    arr = torch.tensor(COSTAS, dtype=torch.float32)[torch.randint(0, len(COSTAS), (n,))]
    t = torch.arange(N, dtype=torch.float32)[None]
    m = t - n0[:, None]
    inside = (m >= 0) & (m < L[:, None])
    u = (m / L[:, None]).clamp(0, 0.999999)         # position within the pulse, 0 to 1
    yy = y[:, None]
    f = f0[:, None] + torch.zeros_like(t)           # instantaneous frequency
    f = torch.where(yy == 2, f0[:, None] + up[:, None] * bw[:, None] * (u - 0.5), f)
    f = torch.where(yy == 5, f0[:, None] + (arr.gather(1, torch.floor(u * 6).long()) - 3.5) * (6 / L[:, None]), f)
    phase = ph0[:, None] + 2 * math.pi * torch.cumsum(f * inside, 1)
    barker = torch.tensor(BARKER, dtype=torch.float32)
    phase = phase + torch.where(yy == 3, math.pi * (barker[torch.floor(u * 13).long()] < 0), 0)
    chip = torch.floor(u * M[:, None] ** 2)
    phase = phase + torch.where(yy == 4, 2 * math.pi * torch.floor(chip / M[:, None]) * torch.remainder(chip, M[:, None]) / M[:, None], 0)
    on = inside * (10 ** (snr_db / 20) * (y != 0))[:, None]
    x = torch.stack((on * torch.cos(phase), on * torch.sin(phase)), 1)
    x = x + torch.randn(x.shape) * math.sqrt(0.5)   # complex noise of unit power
    return x / torch.sqrt((x ** 2).sum(1).mean(1))[:, None, None], y, snr_db   # automatic gain control`;
  py.model = `import torch.nn as nn
import torch.nn.functional as F

class RadarNet(nn.Module):
    """Conv1d, BatchNorm and ReLU five times (max-pooling after layers 2 to 4), an average over time, a linear layer."""
    def __init__(self, widths=(16, 32, 32, 64, 64), ks=(7, 5, 5, 3, 3), pools=(False, True, True, True, False)):
        super().__init__()
        self.convs, self.bns, self.pools = nn.ModuleList(), nn.ModuleList(), pools
        cin = 2                                     # I and Q
        for c, k in zip(widths, ks):
            self.convs.append(nn.Conv1d(cin, c, k, padding=k // 2, bias=False))
            self.bns.append(nn.BatchNorm1d(c))
            cin = c
        self.fc = nn.Linear(cin, len(CLASSES))

    def features(self, x):
        for conv, bn, pool in zip(self.convs, self.bns, self.pools):
            x = F.relu(bn(conv(x)))
            if pool:
                x = F.max_pool1d(x, 2)
        return x                                    # (batch, 64, 32)

    def forward(self, x):
        return self.fc(self.features(x).mean(-1))  # average over time, then 6 scores


@torch.no_grad()
def layer_outputs(model, x):
    """Every layer's output after its ReLU and before its max-pool: the feature maps the lab draws."""
    outs = []
    for conv, bn, pool in zip(model.convs, model.bns, model.pools):
        a = F.relu(bn(conv(x)))
        outs.append(a)
        x = F.max_pool1d(a, 2) if pool else a
    return outs`;
  py.train = (cls, snr) => `model = RadarNet()
opt = torch.optim.AdamW(model.parameters(), lr=3e-3, weight_decay=1e-4)
sched = torch.optim.lr_scheduler.OneCycleLR(opt, max_lr=3e-3, total_steps=6000, pct_start=0.1)
for step in range(6000):                        # 6,000 batches of 512 fresh signals: about 3 million
    x, y, _ = generate(512)
    loss = F.cross_entropy(model(x), y)
    opt.zero_grad()
    loss.backward()
    opt.step()
    sched.step()
model.eval()                                    # BatchNorm uses its running averages from here on

x, y, snr = generate(1, cls=${cls}, snr=${snr})            # one intercept: a ${RD.CLASSES[cls]}
with torch.no_grad():
    p = model(x).softmax(-1)[0]
print(CLASSES[int(p.argmax())], round(float(p.max()), 3))`;
  // the whole script: generator, model, helper, training
  py.script = (cls, snr) => `${py.gen}


${py.model}


${py.train(cls, snr)}`;
  // one convolution layer, recomputed with its BatchNorm folded in and checked against the model
  py.fold = (o) => `x, y, snr = generate(1, cls=${o.cls}, snr=${o.snr})
outs = layer_outputs(model, x)
conv, bn = model.convs[${o.l}], model.bns[${o.l}]
inp = ${o.l === 0 ? 'x' : o.prevPool ? `F.max_pool1d(outs[${o.l - 1}], 2)` : `outs[${o.l - 1}]`}    # this layer's input: (1, ${o.ci}, ${o.n})
with torch.no_grad():
    # the browser runs each layer with its BatchNorm folded into the convolution's weights and bias
    scale = bn.weight / torch.sqrt(bn.running_var + bn.eps)
    w = conv.weight * scale[:, None, None]      # (${o.co}, ${o.ci}, ${o.k})
    b = bn.bias - bn.running_mean * scale       # (${o.co},)
    a = F.relu(F.conv1d(inp, w, b, padding=${o.k >> 1}))
print(a.shape, torch.allclose(a, outs[${o.l}], atol=1e-4))     # torch.Size([1, ${o.co}, ${o.n}]) True
print('share of zeros after ReLU:', round(float((a == 0).float().mean()), 3))${o.pool ? `
pooled = F.max_pool1d(a, 2)                     # (1, ${o.co}, ${o.n / 2}): the larger of each pair of steps` : ''}`;

  /* A heatmap drawn as an image (one pixel per cell, scaled up without smoothing): crisp cells and fast, even
     for 64 x 256 maps. Sequential colours for values from lo to hi (square-root scaled, or linear with
     opt.lin, e.g. for decibels); opt.div (or true) for diverging colours from -hi to hi. */
  let OFF = null;
  function heat(ctx, rows, cols, get, r, lo, hi, opt) {
    if (!OFF) OFF = document.createElement('canvas');
    OFF.width = cols; OFF.height = rows;
    const div = opt === true || !!(opt && opt.div), lin = !!(opt && opt.lin);
    const o = OFF.getContext('2d'), img = o.createImageData(cols, rows), lut = div ? theme.div : theme.seq, span = hi - lo || 1;
    for (let i = 0; i < rows; i++) for (let j = 0; j < cols; j++) {
      const v = get(i, j), u = clamp((v - lo) / span, 0, 1), q = div ? Math.round((clamp(v / (hi || 1), -1, 1) + 1) * 128) * 3 : Math.round((lin ? u : Math.sqrt(u)) * 255) * 3, p = (i * cols + j) * 4;
      img.data[p] = lut[q]; img.data[p + 1] = lut[q + 1]; img.data[p + 2] = lut[q + 2]; img.data[p + 3] = 255;
    }
    o.putImageData(img, 0, 0);
    ctx.save(); ctx.imageSmoothingEnabled = false; ctx.drawImage(OFF, r.x, r.y, r.w, r.h); ctx.restore();
    ctx.strokeStyle = theme.c('line-2'); ctx.lineWidth = 0.8; ctx.strokeRect(r.x - 0.5, r.y - 0.5, r.w + 1, r.h + 1);
  }
  const maxOf = (a) => { let m = 1e-9; for (let i = 0; i < a.length; i++) if (Math.abs(a[i]) > m) m = Math.abs(a[i]); return m; };
  // the two input channels as thin traces
  function traces(ctx, iq, r, n0, L, lw = 1) {
    const N = RD.N, m = Math.max(1.5, maxOf(iq)) * 1.05, X = (t) => r.x + (t / (N - 1)) * r.w, Y = (v) => r.y + r.h / 2 - (v / m) * (r.h / 2);
    if (L) { ctx.fillStyle = theme.c('ink', theme.dark ? 0.07 : 0.05); ctx.fillRect(X(n0), r.y, X(n0 + L - 1) - X(n0), r.h); }
    for (const [off, col] of [[0, 's1'], [N, 's2']]) {
      ctx.strokeStyle = theme.c(col, 0.85); ctx.lineWidth = lw; ctx.beginPath();
      for (let t = 0; t < N; t++) { const x = X(t), y = Y(iq[off + t]); t ? ctx.lineTo(x, y) : ctx.moveTo(x, y); }
      ctx.stroke();
    }
  }

  // ------------------------------------------------------------------ the tree
  function build(L, size) {
    const wide = size.w / size.h >= 1.05, W2 = (w, t) => (wide ? w : t), X = { wide }, Dp = NA.deep;
    const W = NA.radarWeights, net = L.net, sig = L.sig, s = sig.s, F = sig.F, CL = RD.CLASSES, N = RD.N;
    const cls = sig.cls, snr = sig.snr, pred = sig.pred, nL = F.rec.length, p = s.p, pyS = Math.round(snr * 10) / 10;
    const RELU = { id: 'relu', label: 'ReLU', f: (z) => Math.max(0, z) };
    // how far apart (in input samples) neighbouring steps of each layer are, and how many samples each one sees
    const JUMP = [], RF = [];
    let jump = 1, rf = 1;
    W.ks.forEach((k, i) => { JUMP.push(jump); rf += (k - 1) * jump; RF.push(rf); if (W.pools[i]) { rf += jump; jump *= 2; } });
    const inPulse = (sample) => cls !== 0 && sample >= p.n0 && sample < p.n0 + p.L;
    const verdict = `${pred === cls ? 'right' : 'wrong'}: ${CL[pred]}, ${fmtPct(F.probs[pred], 0)}`;

    // ---- one number of a feature map: the neuron that computed it
    const neuronOf = (l, c, t) => {
      const r = F.rec[l], Lr = r.L, T = RD.convTerms(r, c, t), k = Lr.k, h = k >> 1;
      const inLabel = (i) => { const q = T.terms[i]; return l === 0 ? `${q.ic ? 'Q' : 'I'}[${q.s}]` : `channel ${q.ic + 1}, t = ${q.s}`; };
      const z = T.terms.reduce((a, q) => a + q.a * q.w, T.bias);
      return Dp.neuron(X, {
        title: `Conv ${l + 1} · channel ${c + 1} · t = ${t}`, sub: `${T.terms.length} inputs, ReLU`, tone: 's1',
        terms: T.terms.map((q) => ({ a: q.a, w: q.w })), bias: T.bias, act: RELU, outName: 'a',
        inLabel, wLabel: (i) => `W${l + 1}[${c}, ${T.terms[i].ic}, ${i % k}]`,
        inWhat: l === 0 ? `I then Q, samples ${t - h} to ${t + h}` : `${Lr.ci} channels of conv ${l}, steps ${t - h} to ${t + h}`,
        inputNode: (i) => inputOf(l, T.terms[i]),
        vars: { w: `w${l + 1}[${c}].flatten()`, a: 'window.flatten()', b: `b${l + 1}[${c}]` },
        codeHead: [`window = inp[0, :, ${t - h}:${t + h + 1}]    # ${Lr.ci} x ${k} inputs (zeros past the edges)`],
        codeWhat: `as one output of nn.Conv1d(${Lr.ci}, ${Lr.co}, ${k}) with its BatchNorm folded in`,
        about: [
          `One number of conv ${l + 1}'s feature map: channel ${c + 1} at time step ${t}${JUMP[l] > 1 ? ` (around input sample ${t * JUMP[l]}, since this layer's steps are ${JUMP[l]} samples apart)` : ''}. Its filter looks at ${k} neighbouring steps of all ${Lr.ci} input ${l === 0 ? 'channels, I and Q' : 'channels'}, multiplies each by a learned weight and adds them up with the bias: z = ${fmt(z, 4)}, so ReLU gives ${fmt(Math.max(0, z), 4)}${z <= 0 ? ': this filter is silent here' : ''}. Counting the layers below, it depends on ${RF[l]} input samples (${fmt(RF[l] / 20, 2)} µs)${inPulse(t * JUMP[l]) ? ', centred inside the pulse' : ', centred outside the pulse'}.`,
          `BatchNorm is folded into the weights and the bias, as the browser runs it. Zoom into an input bar to open ${l === 0 ? 'the raw I or Q sample' : 'the neuron of the layer below that produced it (after a max-pool, the larger of two neighbours)'}.`,
        ],
      });
    };
    const inputOf = (l, q) => {
      if (!q.inside) return Object.assign(Dp.stored(X, 0, 'zero padding'), { sub: 'past the edge of the window' });
      if (l === 0) return Dp.stored(X, q.a, `${q.ic ? 'Q' : 'I'}[${q.s}], a raw sample`);
      const prev = F.rec[l - 1];
      if (!prev.L.pool) return neuronOf(l - 1, q.ic, q.s);
      const n2 = prev.inp.n, a0 = prev.act[q.ic * n2 + 2 * q.s], a1 = prev.act[q.ic * n2 + 2 * q.s + 1], tw = a1 > a0 ? 2 * q.s + 1 : 2 * q.s;
      return Object.assign(neuronOf(l - 1, q.ic, tw), { sub: `max-pool: the larger of t = ${2 * q.s} and ${2 * q.s + 1}` });
    };

    // ---- layout: two rows when wide (the line wraps after conv 3), one column when tall
    const IDS = ['input', 'c0', 'c1', 'c2', 'c3', 'c4', 'gap', 'fc', 'sm'];
    const WIDE = { input: [0, 0.03, 0.19, 0.4], c0: [0.27, 0.03, 0.19, 0.4], c1: [0.54, 0.03, 0.19, 0.4], c2: [0.81, 0.03, 0.19, 0.4],
      c3: [0, 0.57, 0.15, 0.4], c4: [0.2125, 0.57, 0.15, 0.4], gap: [0.425, 0.57, 0.15, 0.4], fc: [0.6375, 0.57, 0.15, 0.4], sm: [0.85, 0.57, 0.15, 0.4] };
    const rectOf = (id) => (wide ? WIDE[id] : [0.04, 0.004 + IDS.indexOf(id) * 0.1105, 0.74, 0.094]);
    const SHAPES = ['2×256', ...F.rec.map((r) => `${r.L.co}×${r.out.n}`), '64', '6'];

    // ---- input
    const rowsIQ = W2({ i: [0, 0.2, 1, 0.26], q: [0, 0.52, 1, 0.26] }, { i: [0, 0.2, 1, 0.24], q: [0, 0.5, 1, 0.24] });
    const inputNode = {
      id: 'input', title: 'Input: I and Q', sub: `2 × ${N} samples · ${CL[cls]}, ${dB(snr)}`, rect: rectOf('input'),
      summary: (ctx, r) => traces(ctx, s.iq, r, p.n0, cls ? p.L : 0, 0.8),
      grids: [
        { id: 'I', rect: rowsIQ.i, rows: 1, cols: N, make: (i, j) => Dp.stored(X, s.iq[j], `I[${j}], a raw sample`) },
        { id: 'Q', rect: rowsIQ.q, rows: 1, cols: N, make: (i, j) => Dp.stored(X, s.iq[N + j], `Q[${j}], a raw sample`) },
      ],
      inner: (ctx, r, S) => {
        const uu = S.u(r, 8, 13), ri = S.place(r, rowsIQ.i), rq = S.place(r, rowsIQ.q), m = maxOf(s.iq);
        S.lines(ctx, [`x[t] = I[t] + j·Q[t],  t = 0 … ${N - 1}  (50 ns apart: 12.8 µs)`, cls ? `sent: ${RD.describe(s)}` : 'sent: nothing, receiver noise only'], S.place(r, [0, 0, 1, 0.14]), { mono: true, size: uu * 0.92, strong: (i) => i === 0 });
        for (const [rr, off, col, name] of [[ri, 0, 's1', 'I (in-phase)'], [rq, N, 's2', 'Q (quadrature)']]) {
          S.caption(ctx, `${name}: one bar per sample, zoom into any`, rr, { size: uu * 0.8, minW: 200 });
          if (cls) { const x0 = rr.x + (p.n0 / N) * rr.w; ctx.fillStyle = theme.c('ink', theme.dark ? 0.07 : 0.05); ctx.fillRect(x0, rr.y, (p.L / N) * rr.w, rr.h); }
          S.bars(ctx, s.iq.subarray(off, off + N), rr, { max: m, color: () => theme.c(col, 0.8) });
        }
        S.text(ctx, cls ? `shaded: the pulse, samples ${p.n0} to ${p.n0 + p.L - 1}` : 'no pulse: every sample is noise', r.x, r.y + r.h * 0.86, { size: uu * 0.85, color: theme.c('muted'), maxW: r.w });
      },
      about: [
        `What the network reads: 256 complex samples, I (the real part) and Q (the imaginary part), taken 50 ns apart by a receiver sampling at 20 MHz. ${cls ? `This window holds a ${CL[cls]}: ${RD.describe(s)}, at ${dB(snr)} per sample.` : 'This window holds receiver noise only.'}`,
        'The window is scaled to an average power of 1 before the network sees it (automatic gain control), so how loud the signal is says nothing by itself; the network has to find the pattern in the shape of the samples. Zoom into any bar to see the number as the 32 bits stored in memory.',
      ],
      code: `import matplotlib.pyplot as plt
# generate and N as in the whole-model code (the top level's Python tab)
x, y, snr = generate(1, cls=${cls}, snr=${pyS})
t_us = torch.arange(N) / 20                     # 20 samples per microsecond
plt.plot(t_us, x[0, 0], lw=1, label='I')
plt.plot(t_us, x[0, 1], lw=1, label='Q')
plt.plot(t_us, x[0].norm(dim=0), 'k', lw=1, label='|x|, the envelope')
plt.xlabel('time (us)')
plt.legend()
plt.show()
print(x.shape, float((x ** 2).sum(1).mean()))   # torch.Size([1, 2, 256]) 1.0`,
    };

    // ---- the convolution layers
    const convNode = (l) => {
      const r = F.rec[l], Lr = r.L, C = Lr.co, n = r.inp.n, k = Lr.k, h = k >> 1, act = r.act, mx = maxOf(act), wmx = maxOf(Lr.w), J = Lr.ci * k;
      let zeros = 0;
      for (let i = 0; i < act.length; i++) if (act[i] === 0) zeros++;
      // which channel answers most strongly inside the pulse compared with outside it
      let best = 0, bestGap = -Infinity;
      if (cls) for (let c = 0; c < C; c++) {
        let ins = 0, ni = 0, out = 0, no = 0;
        for (let t = 0; t < n; t++) { const v = act[c * n + t]; if (inPulse(t * JUMP[l])) { ins += v; ni++; } else { out += v; no++; } }
        const g = ins / Math.max(1, ni) - out / Math.max(1, no);
        if (g > bestGap) { bestGap = g; best = c; }
      }
      const LR = W2({ a: [0, 0.2, 0.62, 0.42], w: [0.68, 0.2, 0.32, 0.42], t: [0, 0.7, 1, 0.28] }, { a: [0, 0.16, 1, 0.3], w: [0, 0.56, 1, 0.18], t: [0, 0.8, 1, 0.2] });
      return {
        id: 'c' + l, title: `Conv ${l + 1}`, sub: `${Lr.ci} → ${C} channels · kernel ${k}${Lr.pool ? ' · max-pool' : ''}`, tone: 's1', rect: rectOf('c' + l),
        summary: (ctx, q) => heat(ctx, C, n, (i, j) => act[i * n + j], q, 0, mx),
        grids: [
          { id: 'a', rect: LR.a, rows: C, cols: n, make: (i, j) => neuronOf(l, i, j) },
          { id: 'w', rect: LR.w, rows: C, cols: J, make: (i, j) => Dp.stored(X, Lr.w[i * J + j], `W${l + 1}[${i}, ${Math.floor(j / k)}, ${j % k}], a weight`) },
        ],
        inner: (ctx, rr, S) => {
          const uu = S.u(rr, 8, 13), ar = S.place(rr, LR.a), wr = S.place(rr, LR.w);
          S.caption(ctx, `output: ${C} channels × ${n} steps, after ReLU (bar: the pulse)`, ar, { size: uu * 0.8, minW: 240 });
          heat(ctx, C, n, (i, j) => act[i * n + j], ar, 0, mx);
          if (cls) {
            const x0 = ar.x + clamp(p.n0 / JUMP[l] / n, 0, 1) * ar.w, x1 = ar.x + clamp((p.n0 + p.L) / JUMP[l] / n, 0, 1) * ar.w;
            ctx.fillStyle = theme.c(CCOL[cls]); ctx.fillRect(x0, ar.y - 4, x1 - x0, 2.5);
          }
          S.caption(ctx, `weights: ${C} × (${Lr.ci} × ${k})`, wr, { size: uu * 0.8, minW: 120 });
          heat(ctx, C, J, (i, j) => Lr.w[i * J + j], wr, 0, wmx, true);
          S.lines(ctx, [
            `a[c, t] = ReLU( b[c] + Σᵢ Σₖ W[c, i, k] · x[i, t + k − ${h}] )   ${J} products per number`,
            Lr.pool ? `then max-pool: out[c, t] = max(a[c, 2t], a[c, 2t + 1])  →  ${C} × ${n / 2}` : `no pooling: the ${C} × ${n} map goes on as it is`,
            `each number depends on ${RF[l]} input samples (${fmt(RF[l] / 20, 2)} µs); ${fmtPct(zeros / act.length, 0)} of them are 0 after ReLU`,
          ], S.place(rr, LR.t), { mono: true, size: uu * 0.88, strong: (i) => i === 0 });
        },
        about: [
          `Convolution layer ${l + 1}: ${C} filters, each ${k} steps long across all ${Lr.ci} input ${l === 0 ? 'channels (I and Q)' : 'channels'}, slide along the ${n} time steps. Every row of the map is one filter's answer at every position, after ReLU (dark: zero). ${Lr.pool ? `Max-pooling then keeps the larger of each pair of neighbouring steps, halving the length to ${n / 2}: the next layer sees twice as far for the same kernel.` : 'There is no pooling after this layer.'}`,
          `Stacked, the layers see further and further: one number here depends on ${RF[l]} input samples, ${fmt(RF[l] / 20, 2)} µs of signal${l === 0 ? ', enough to measure a phase step or a frequency' : l === nL - 1 ? ', a good part of a pulse' : ''}. ${fmtPct(zeros / act.length, 0)} of this map is exactly zero: most filters answer only to particular patterns.${cls ? ` Channel ${best + 1} responds most to the pulse compared with the noise around it.` : ''}`,
          'The weights on the right are drawn with one row per filter (blue positive, orange negative), BatchNorm already folded in. Zoom into a cell of the map for the neuron that computed it, or into a weight for its bits.',
        ],
        code: py.fold({ l, ci: Lr.ci, co: C, k, n, pool: Lr.pool, prevPool: l > 0 && F.rec[l - 1].L.pool, cls, snr: pyS }),
      };
    };

    // ---- the average over time, the linear layer and the softmax
    const last = F.rec[nL - 1], Cl = last.L.co, nl = last.out.n, gap = F.gap;
    const avgNode = (c) => Object.assign(Dp.neuron(X, {
      title: `Average of channel ${c + 1}`, sub: `${nl} values × 1/${nl}`, tone: 's3',
      terms: Array.from({ length: nl }, (_, t) => ({ a: last.out.d[c * nl + t], w: 1 / nl })), bias: null, act: null, outName: 'g',
      rowNames: [`channel ${c + 1} of conv ${nL} at its ${nl} steps`, `weights: all 1/${nl}`, 'products'],
      inLabel: (t) => `t = ${t}`, wLabel: () => `1/${nl}`, inputNode: (t) => neuronOf(nL - 1, c, t),
      vars: { w: `torch.full((${nl},), 1 / ${nl})`, a: `a${nL}[${c}]`, b: '' },
    }), { about: [`An average is a weighted sum whose weights are all 1/${nl}: g = ${fmt(gap[c], 4)} for channel ${c + 1}. Averaging over time throws away where in the window the pattern was, which is why the classifier does not care where the pulse starts. Zoom into a bar for the neuron of conv ${nL} behind it.`] });
    const logitNode = (k) => Dp.neuron(X, {
      title: `Score for ${CL[k]}`, sub: `${Cl} averages, no activation`, tone: 's7',
      terms: Array.from(gap, (g, c) => ({ a: g, w: net.fc.w[k * Cl + c] })), bias: net.fc.b[k], act: null, outName: 'score',
      inLabel: (c) => `average of channel ${c + 1}`, wLabel: (c) => `W[${k}, ${c}]`, inWhat: `the ${Cl} channel averages`,
      inputNode: (c) => avgNode(c), vars: { w: `model.fc.weight[${k}]`, a: 'g', b: `model.fc.bias[${k}]` },
      about: [`The score for ${CL[k]}: the ${Cl} channel averages, each times its own weight, plus a bias. Score ${fmt(F.logits[k], 3)}${k === pred ? ', the highest of the six, so it is the answer' : ''}. The softmax turns the six scores into probabilities.`],
    });
    const smNode = (j) => Object.assign(Dp.lib(X, 'softmax', { scores: Array.from(F.logits), j, label: `p(${SHORT[j]})` }), { title: `Probability of ${CL[j]}`, sub: fmtPct(F.probs[j], 1) });
    const kids = [inputNode, ...F.rec.map((_, l) => convNode(l)), {
      id: 'gap', title: 'Average over time', sub: `${Cl} × ${nl} → ${Cl}`, tone: 's3', rect: rectOf('gap'),
      summary: (ctx, r, S) => S.bars(ctx, gap, r, { positive: true, color: () => theme.c('s3', 0.8) }),
      grids: [{ id: 'g', rect: W2([0, 0.62, 1, 0.22], [0, 0.6, 1, 0.24]), rows: 1, cols: Cl, make: (i, c) => avgNode(c) }],
      inner: (ctx, r, S) => {
        const uu = S.u(r, 8, 13), hr = S.place(r, W2([0.2, 0.18, 0.6, 0.32], [0, 0.16, 1, 0.3])), br = S.place(r, W2([0, 0.62, 1, 0.22], [0, 0.6, 1, 0.24]));
        S.caption(ctx, `conv ${nL}'s output: ${Cl} channels × ${nl} steps`, hr, { size: uu * 0.8, minW: 200 });
        heat(ctx, Cl, nl, (i, j) => last.out.d[i * nl + j], hr, 0, maxOf(last.out.d));
        S.arrow(ctx, hr.x + hr.w / 2, hr.y + hr.h + 4, br.x + br.w / 2, br.y - uu * 1.6, { lw: 1.2, head: 5 });
        S.text(ctx, `g[c] = (1/${nl}) Σₜ a[c, t]`, hr.x + hr.w / 2 + 10, (hr.y + hr.h + br.y) / 2 - uu * 0.4, { size: uu * 0.9, mono: true });
        S.caption(ctx, `the ${Cl} averages: zoom into any`, br, { size: uu * 0.8, minW: 160 });
        S.bars(ctx, gap, br, { positive: true, color: () => theme.c('s3', 0.85) });
      },
      about: [
        `Global average pooling: each of conv ${nL}'s ${Cl} channels is averaged over its ${nl} time steps, leaving ${Cl} numbers that describe the whole window. It is what lets the classifier work wherever the pulse falls: a pattern near the start and the same pattern near the end give the same averages.`,
        'It also makes the network\'s reasoning easy to read back: because averaging comes before the last layer, each class score is the average over time of a curve, the class activation map in the "What the CNN sees" view.',
      ],
      code: `x, y, snr = generate(1, cls=${cls}, snr=${pyS})
outs = layer_outputs(model, x)
g = outs[-1].mean(-1)                           # (1, 64): one number per channel
print(g.shape, g.min().item() >= 0)             # after ReLU, every average is 0 or more`,
    }, {
      id: 'fc', title: 'Linear layer', sub: `${Cl} → 6 scores`, tone: 's7', rect: rectOf('fc'),
      summary: (ctx, r, S) => S.bars(ctx, F.logits, r, { color: (i) => theme.c(CCOL[i], 0.85) }),
      grids: [
        { id: 'z', rect: W2([0, 0.66, 1, 0.24], [0, 0.66, 1, 0.24]), rows: 1, cols: 6, make: (i, k) => logitNode(k) },
        { id: 'W', rect: W2([0, 0.18, 1, 0.3], [0, 0.18, 1, 0.3]), rows: 6, cols: Cl, make: (k, c) => Dp.stored(X, net.fc.w[k * Cl + c], `W[${k}, ${c}], a weight`) },
      ],
      inner: (ctx, r, S) => {
        const uu = S.u(r, 8, 13), wr = S.place(r, [0, 0.18, 1, 0.3]), zr = S.place(r, [0, 0.66, 1, 0.24]);
        S.caption(ctx, `weights: 6 classes × ${Cl} averages`, wr, { size: uu * 0.8, minW: 160 });
        heat(ctx, 6, Cl, (k, c) => net.fc.w[k * Cl + c], wr, 0, maxOf(net.fc.w), true);
        S.text(ctx, 'score[k] = b[k] + Σ_c W[k, c] · g[c]', r.x, (wr.y + wr.h + zr.y) / 2 - uu * 0.3, { size: uu * 0.9, mono: true });
        S.caption(ctx, 'the six scores (logits): zoom into any', zr, { size: uu * 0.8, minW: 160 });
        S.bars(ctx, F.logits, zr, { color: (i) => theme.c(CCOL[i], 0.85) });
        const bw = zr.w / 6;
        for (let k = 0; k < 6; k++) S.text(ctx, SHORT[k], zr.x + (k + 0.5) * bw, zr.y + zr.h + uu * 0.9, { size: uu * 0.75, align: 'center', color: k === pred ? theme.c('ink') : theme.c('muted'), w: k === pred ? 700 : 500, maxW: bw });
      },
      about: [`The last layer: six weighted sums of the ${Cl} averages, one score per waveform. The highest wins: here ${CL[pred]} with ${fmt(F.logits[pred], 2)}. The weight matrix is small, 6 × ${Cl} plus 6 biases; nearly all of the network's 27,142 parameters are in the convolutions.`],
      code: `x, y, snr = generate(1, cls=${cls}, snr=${pyS})
with torch.no_grad():
    g = model.features(x).mean(-1)
    scores = model.fc(g)                        # (1, 6) = g @ W.T + b
print([round(float(v), 2) for v in scores[0]])`,
    }, {
      id: 'sm', title: 'Softmax', sub: `${CL[pred]}, ${fmtPct(F.probs[pred], 0)}`, tone: 's7', rect: rectOf('sm'),
      summary: (ctx, r, S) => S.bars(ctx, F.probs, r, { positive: true, max: 1, color: (i) => theme.c(CCOL[i], i === pred ? 1 : 0.45) }),
      grids: [{ id: 'p', rect: [0, 0.2, 1, 0.5], rows: 1, cols: 6, make: (i, j) => smNode(j) }],
      inner: (ctx, r, S) => {
        const uu = S.u(r, 8, 13), pr = S.place(r, [0, 0.2, 1, 0.5]), bw = pr.w / 6;
        S.caption(ctx, 'probabilities: zoom into any', pr, { size: uu * 0.8, minW: 160 });
        S.bars(ctx, F.probs, pr, { positive: true, max: 1, color: (i) => theme.c(CCOL[i], i === pred ? 1 : 0.5) });
        for (let k = 0; k < 6; k++) {
          S.text(ctx, SHORT[k], pr.x + (k + 0.5) * bw, pr.y + pr.h + uu * 0.9, { size: uu * 0.75, align: 'center', color: k === pred ? theme.c('ink') : theme.c('muted'), w: k === pred ? 700 : 500, maxW: bw });
          S.text(ctx, fmtPct(F.probs[k], 0), pr.x + (k + 0.5) * bw, pr.y + pr.h + uu * 2.1, { size: uu * 0.75, align: 'center', mono: true, color: theme.c('ink-2'), maxW: bw });
        }
        S.lines(ctx, [`p[k] = exp(score[k]) / Σⱼ exp(score[j])`, `sent: ${CL[cls]}.  The network: ${verdict}`], S.place(r, [0, 0.84, 1, 0.15]), { mono: true, size: uu * 0.88, strong: (i) => i === 1 });
      },
      about: [
        `Softmax turns the six scores into probabilities that are positive and add up to 1: exponentiate each, divide by the total. The answer is the class with the largest one: ${CL[pred]}, ${fmtPct(F.probs[pred], 1)}. ${pred === cls ? 'That is what was sent.' : `What was sent was a ${CL[cls]}, at ${dB(snr)}.`}`,
        'Training pushed the probability of the right class up on 3 million signals (cross-entropy loss: minus the log of that probability), and the probabilities stay meaningful: at +10 dB the network is right 99% of the time, and when it is unsure the bars show it.',
      ],
      code: `x, y, snr = generate(1, cls=${cls}, snr=${pyS})
with torch.no_grad():
    p = model(x).softmax(-1)[0]
for name, v in zip(CLASSES, p):
    print(f'{name:12s} {float(v):.3f}')`,
    }];
    return {
      id: 'radar', title: 'Radar classifier', sub: `${fmtInt(net.params)} parameters · 6 classes`,
      inner: (ctx, r, S) => {
        const at = (id) => S.place(r, rectOf(id)), uu = S.u(r, 8, 12);
        const lab = (t, x, y, align = 'center', maxW) => S.text(ctx, t, x, y, { size: uu * 0.82, mono: true, color: theme.c('muted'), align, maxW });
        // each arrow carries the tensor that leaves the box it starts from
        for (let i = 0; i + 1 < IDS.length; i++) {
          const a = at(IDS[i]), b = at(IDS[i + 1]);
          if (!wide) { S.arrow(ctx, a.x + a.w / 2, a.y + a.h + 1, a.x + a.w / 2, b.y - 2, { lw: 1.1, head: 4 }); lab(SHAPES[i], a.x + a.w / 2 + 8, (a.y + a.h + b.y) / 2, 'left'); continue; }
          if (Math.abs(a.y - b.y) < 1) { S.arrow(ctx, a.x + a.w + 2, a.y + a.h / 2, b.x - 3, b.y + b.h / 2, { lw: 1.2, head: 5 }); lab(SHAPES[i], (a.x + a.w + b.x) / 2, a.y + a.h / 2 - uu * 1.05, 'center', b.x - a.x - a.w + 6); continue; }
          // the line wraps: down from conv 3, back to the left, into conv 4
          const ym = (a.y + a.h + b.y) / 2;
          ctx.strokeStyle = theme.c('ink-2'); ctx.lineWidth = 1.2; ctx.beginPath(); ctx.moveTo(a.x + a.w / 2, a.y + a.h + 2); ctx.lineTo(a.x + a.w / 2, ym); ctx.lineTo(b.x + b.w / 2, ym); ctx.stroke();
          S.arrow(ctx, b.x + b.w / 2, ym, b.x + b.w / 2, b.y - 3, { lw: 1.2, head: 5 });
          lab(SHAPES[i], (a.x + a.w / 2 + b.x + b.w / 2) / 2, ym - uu * 0.9);
        }
      },
      children: kids,
      about: [
        `A one-dimensional convolutional network with ${fmtInt(net.params)} parameters. Five convolution layers slide short filters along the 256 I/Q samples, each seeing a longer stretch of signal than the one before (7 samples in the first layer, 50 in the last); max-pooling halves the length three times. Then every channel is averaged over time, and a linear layer turns the 64 averages into six scores, one per waveform, which softmax makes into probabilities.`,
        `On screen: a ${CL[cls]} at ${dB(snr)}. The network's answer: ${CL[pred]}, ${fmtPct(F.probs[pred], 1)}${pred === cls ? ', right' : ', wrong'}.`,
        'Click or zoom into a layer, then into any number of its feature map: that opens the neuron that computed it, whose inputs open the neurons below it, all the way down to the raw samples. Below every number are the arithmetic and the circuits that compute it. The Python tab has each part, and at this top level the whole model with its training.',
      ],
      code: py.script(cls, pyS),
      codeLabel: 'Python · PyTorch · the generator, the model and its training',
    };
  }

  NA.archRadar = { build, py, heat, traces };
})(window.NA);
