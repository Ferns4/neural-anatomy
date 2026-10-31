/* builder-py-gen.js — the PyTorch of the kinds of problem in js/builder-gen.js: their data (the same generators as the
   page, in Python), the losses and measures that differ from the usual ones (pictures rebuilt pixel by pixel, a middle
   and a spread, the overlap of shapes), a few lines after training (new pictures from a VAE, the band of a spread),
   and whole scripts for the two that are not examples with answers: a GAN (your model the generator, its rival the
   discriminator) and contrastive learning (two views of each picture, NT-Xent). builder-py.js asks for them. */
(function (NA) {
  'use strict';
  const PG = {};
  const owns = (d) => !!(NA.bgen && NA.bgen.owns(d.family));
  PG.owns = owns;

  // ------------------------------------------------------------------ data
  const ASCII = [
    'def show(pictures, width=14):',
    '    """Pictures as text, side by side: the darker the character, the more ink."""',
    '    rows = [p.reshape(-1, width).clamp(0, 0.999) for p in pictures]',
    '    for y in range(rows[0].shape[0]):',
    '        print("  ".join("".join(" .:-=+*#%@"[int(10 * v)] for v in r[y]) for r in rows))',
  ];
  const DAMAGE = [
    'def damage(x, noise):',
    '    """Speckles everywhere and a scratch: a thick line through the picture."""',
    '    out = x.clone()',
    '    yy, xx = torch.meshgrid(torch.arange(G) + 0.5, torch.arange(G) + 0.5, indexing="ij")',
    '    for img in out:',
    '        a, cx, cy = random.uniform(0, math.pi), random.uniform(4, 10), random.uniform(4, 10)',
    '        line = ((xx - cx) * -math.sin(a) + (yy - cy) * math.cos(a)).abs() < 0.9',
    '        img[0] = torch.where(line, img[0].clamp(min=0.85), img[0])',
    '    return (out + noise * torch.randn_like(out)).clamp(0, 1)',
  ];
  const SEGMENT = String.raw`G = 16                                            # pictures are G x G pixels
SHAPES = ["nothing", "circle", "square", "triangle"]
py, px = torch.meshgrid(torch.arange(G, dtype=torch.float32), torch.arange(G, dtype=torch.float32), indexing="ij")

def inside(c, u, v, r):
    """Is (u, v), measured from a shape's middle along its own axes, inside shape c (1 circle, 2 square, 3 triangle)?"""
    if c == 1:
        return u ** 2 + v ** 2 <= r ** 2
    if c == 2:
        return (u.abs() <= 0.82 * r) & (v.abs() <= 0.82 * r)
    ok, R = torch.ones_like(u, dtype=torch.bool), 1.15 * r          # a triangle: inside all three edges
    for q in range(3):
        a0, a1 = q * 2 * math.pi / 3, (q + 1) * 2 * math.pi / 3
        x0, y0, x1, y1 = R * math.cos(a0), R * math.sin(a0), R * math.cos(a1), R * math.sin(a1)
        ok &= (x1 - x0) * (v - y0) - (y1 - y0) * (u - x0) >= 0
    return ok

def picture(k, noise):
    """k filled shapes of different kinds, apart from each other: the picture, and every pixel's shape (0 if none)."""
    ink, label, placed = torch.zeros(G, G), torch.zeros(G, G, dtype=torch.long), []
    for c in random.sample([1, 2, 3], k):
        for t in range(40):
            r = random.uniform(2.5, 3.6 if k == 3 else 4.3)
            x, y, a = random.uniform(r + 0.3, G - r - 0.3), random.uniform(r + 0.3, G - r - 0.3), random.uniform(0, 2 * math.pi)
            if all(math.hypot(x - q[0], y - q[1]) > (r + q[2]) * 1.1 + 1 for q in placed) or t == 39:
                break
        placed.append((x, y, r))
        def at(ox, oy):                            # the points (px + ox, py + oy) inside this shape
            dx, dy = px + ox - x, py + oy - y
            return inside(c, dx * math.cos(a) + dy * math.sin(a), -dx * math.sin(a) + dy * math.cos(a), r)
        ink = torch.maximum(ink, sum(at((i + 0.5) / 3, (j + 0.5) / 3).float() for i in range(3) for j in range(3)) / 9)
        label[at(0.5, 0.5)] = c                    # a pixel's shape: the one its middle is in
    return ink + noise * torch.randn(G, G), label`;
  const GRAPHS = String.raw`N, GROUPS, F_NODE = 12, 3, 4                       # nodes, groups, features of a node

def graph(across, hints):
    """12 people in 3 groups of 4, in a random order: linked within a group 6 times in 10 (each at least once), across
    groups with the odds given. A row per node: which group it says it is in (one-hot) and whether it says, then its
    12 links. The answers: each node's group, or -1 (no answer) for those that say."""
    grp = torch.randperm(N) // (N // GROUPS)
    same = grp[:, None] == grp[None, :]
    A = (torch.rand(N, N) < same.float() * 0.6 + (~same).float() * across).float().triu(1)
    A = A + A.T
    for i in range(N):
        mates = [j for j in range(N) if j != i and grp[j] == grp[i]]
        if A[i, mates].sum() == 0:
            j = random.choice(mates)
            A[i, j] = A[j, i] = 1
    feats, y = torch.zeros(N, F_NODE), grp.clone()
    for g in range(GROUPS):
        for i in random.sample([j for j in range(N) if grp[j] == g], hints):
            feats[i, g], feats[i, 3], y[i] = 1, 1, -1
    return torch.cat([feats, A], 1), y`;
  const SPLIT_ORDER = (frac) => [`k = round(${frac} * len(X))`, 'Xtr, Ytr, Xva, Yva = X[:k], Y[:k], X[k:], Y[k:]'];
  // the data of a problem with answers (recon, segment, graph, spread), as lines; I: builder-py's helpers
  PG.data = function (setup, d, p, s, I) {
    if (!owns(d) || d.family === 'gan' || d.family === 'contrast') return null;
    const { num } = I, out = [];
    if (d.family === 'recon') {
      const shapes = s.pictures === 'shapes';
      out.push(shapes ? I.SHAPES_PY : I.DIGITS_PY, '', `n = ${s.n}`,
        shapes ? 'X = torch.stack([draw_shape(i % 5) for i in range(n)])      # (n, 1, 14, 14), ink in [0, 1]' : 'X = torch.stack([draw_digit(i % 10) for i in range(n)])     # (n, 1, 14, 14), ink in [0, 1]',
        'Y = X.flatten(1).clone()                          # the answers: the pictures themselves, 196 numbers each');
      if (d.task === 'denoise') out.push('', ...DAMAGE, '', `X = damage(X, ${num(s.noise)})                             # what it is given: the damaged pictures`);
      out.push(...I.SPLIT(0.8));
    } else if (d.family === 'segment') {
      out.push(SEGMENT, '', `pics = [picture(${s.shapes}, ${num(s.noise)}) for _ in range(${s.n})]`,
        'X = torch.stack([q[0] for q in pics])[:, None]   # (n, 1, 16, 16)', 'Y = torch.stack([q[1] for q in pics]).flatten(1)   # (n, 256): every pixel\'s shape', ...SPLIT_ORDER(0.8));
    } else if (d.family === 'graph') {
      out.push(GRAPHS, '', `graphs = [graph(${num(s.across)}, ${s.hints}) for _ in range(${s.n})]`,
        'X = torch.stack([g[0] for g in graphs])           # (n, 12, 16): each node\'s features and links', 'Y = torch.stack([g[1] for g in graphs])           # (n, 12): its group, -1 where it says', ...SPLIT_ORDER(0.8));
    } else if (d.task === 'branches') {
      out.push(`t = torch.rand(${s.n})`, 'x0 = t + 0.3 * torch.sin(2 * math.pi * t) + 0.2 * torch.rand(len(t)) - 0.1   # x: where t lies on a wavy curve',
        'X, Y = (2 * (x0 + 0.1) / 1.2 - 1)[:, None], 2 * t - 1   # predict y from x: in the middle, three right answers', ...SPLIT_ORDER(0.75));
    } else {
      const gap = d.task === 'gap';
      out.push('def middle(x):', '    return 0.6 * torch.sin(2.5 * x) + 0.2 * x', '',
        'def spread(x):                                    # how far the points stray from the middle, at each x', gap ? '    return 0.05 + 0.1 * x.abs()' : `    return 0.03 + ${num(s.noise)} * (x + 1) / 2`, '',
        `x = 2 * torch.rand(${gap ? 3 * s.n : s.n}) - 1`, gap ? `x = x[x.abs() >= 0.3][:${s.n}]                     # no examples between -0.3 and 0.3` : null,
        'X, Y = x[:, None], middle(x) + spread(x) * torch.randn(len(x))', ...SPLIT_ORDER(0.75));
    }
    return out;
  };

  // ------------------------------------------------------------------ losses, measures, after training
  PG.loss = function (setup, d, I) {
    if (!owns(d)) return null;
    const L = setup.loss;
    if (d.family === 'recon') {
      if (L.id === 'mae') return ['def loss_fn(out, y):                          # absolute error, added up over the pixels', '    return (out - y).abs().sum(1).mean()'];
      if (L.id === 'huber') return [`def loss_fn(out, y):                          # Huber on every pixel (squared near, absolute beyond ${I.num(L.delta || 0.2)}), added up`, `    return F.huber_loss(out, y, delta=${I.num(L.delta || 0.2)}, reduction="sum") / len(y)`];
      return ['def loss_fn(out, y):                          # squared error, ½(rebuilt − picture)², added up over the pixels', '    return 0.5 * ((out - y) ** 2).sum(1).mean()'];
    }
    if (d.family === 'spread' && L.id === 'mdn') return ['def loss_fn(out, y):                          # a mixture of bell curves: −log of the likelihood of y, less ½ log 2π',
      '    m = out.shape[1] // 3                         # m weights (scores), m middles, m log spreads', '    logit, mu, s = out[:, :m], out[:, m:2 * m], out[:, 2 * m:3 * m].clamp(-7, 7)',
      '    logp = logit.log_softmax(1) - 0.5 * ((y[:, None] - mu) * torch.exp(-s)) ** 2 - s', '    return -torch.logsumexp(logp, 1).mean()'];
    if (d.family === 'spread' && L.id === 'gauss') return ['def loss_fn(out, y):                          # Gaussian: −log of the likelihood of y under N(middle, spread²), less ½ log 2π',
      '    mu, s = out[:, 0], out[:, 1].clamp(-7, 7)    # the middle, and the log of the spread squared', '    return 0.5 * (s + (y - mu) ** 2 * torch.exp(-s)).mean()'];
    return null;
  };
  PG.score = function (d, setup) {
    if (d.family === 'spread' && d.task === 'branches' && setup && setup.loss.id === 'mdn') return ['def score(out, y):                            # R² of its nearest middle: of the curves with some weight, the middle nearest each answer',
      '    m = out.shape[1] // 3', '    w, mu = out[:, :m].softmax(1), out[:, m:2 * m]', '    gap = (mu - y[:, None]).abs() + 1e9 * (w < 0.05)', '    pred = mu.gather(1, gap.argmin(1, keepdim=True))[:, 0]', '    return 1 - ((pred - y) ** 2).sum() / ((y - y.mean()) ** 2).sum()'];
    if (d.family === 'recon') return ['def score(out, y):                            # R² over every pixel: 1 is perfect, 0 no better than their mean', '    return 1 - ((out - y) ** 2).sum() / ((y - y.mean()) ** 2).sum()'];
    if (d.family === 'segment') return ['def score(out, y):                            # the overlap (IoU) of each shape with the true one, averaged over the 3 shapes',
      '    pred, ious = out.argmax(-1), []', '    for c in range(1, 4):', '        union = ((pred == c) | (y == c)).sum()', '        if union:', '            ious.append(((pred == c) & (y == c)).sum() / union)',
      '    return torch.stack(ious).mean() if ious else torch.tensor(0.0)'];
    return null;
  };
  // a VAE's Sample block: where it is in the model, so new pictures can be drawn through it
  const sampleAt = (setup) => setup.blocks.findIndex((b) => b.type === 'sample');
  PG.hasSample = (setup) => sampleAt(setup) >= 0 || setup.blocks.some((b) => b.type === 'parallel' && (b.branches || []).some((l) => l.some((q) => q.type === 'sample')));
  PG.after = function (setup, d, I) {
    if (!owns(d)) return [];
    if (d.family === 'recon') {
      const i = sampleAt(setup), out = ['', '# ---------------------------------------------------------------- look at it', 'model.eval()', ...ASCII, '', 'with torch.no_grad():', '    rebuilt = model(Xva[:6])',
        `print("${d.task === 'denoise' ? 'given, rebuilt, clean' : 'the pictures, then rebuilt'}:")`, d.task === 'denoise' ? 'show(list(Xva[:3, 0]) + list(rebuilt[:3]) + list(Yva[:3]))' : 'show(list(Xva[:6, 0]) + list(rebuilt[:6]))'];
      if (i >= 0 && I.shapes && I.shapes[i] && I.shapes[i].out) {
        const k = I.shapes[i].out[0], at = I.attrPath(setup, i, 'sample');
        out.push('', `# new pictures: codes drawn from the bell curve N(0, 1), given to the decoder in place of the encoder's`, `z = torch.randn(6, ${k})`,
          `hook = ${at}.register_forward_hook(lambda module, inputs, output: z)   # the middle gives these codes instead`,
          'with torch.no_grad():', '    new = model(Xva[:6])                         # (the pictures given only set the batch: their codes are replaced)', 'hook.remove()', 'print("new pictures from random codes:")', 'show(list(new))');
      }
      return out;
    }
    if (d.family === 'spread' && d.task === 'branches' && setup.loss.id === 'mdn') return ['', 'model.eval()', 'for x in (-0.6, 0.0, 0.6):', '    with torch.no_grad():', '        out = model(torch.tensor([[x]]))[0]',
      '    m = len(out) // 3', '    w, mu, sd = out[:m].softmax(0), out[m:2 * m], torch.exp(out[2 * m:].clamp(-7, 7))',
      '    print(f"x = {x:+.1f}: " + ", ".join(f"y ~ {mu[k]:+.2f} ± {sd[k]:.2f} (weight {w[k]:.2f})" for k in range(m) if w[k] >= 0.05))'];
    if (d.family === 'spread' && setup.loss.id === 'gauss') return ['', 'model.eval()', 'with torch.no_grad():', '    out = model(Xva)', 'inside = ((Yva - out[:, 0]).abs() <= 2 * torch.exp(0.5 * out[:, 1].clamp(-7, 7))).float().mean()',
      'print(f"{100 * inside:.0f}% of the validation points are inside its band of ±2 spreads (95% if the spread is right)")',
      'for x in (-0.9, -0.5, 0.0, 0.5, 0.9):', '    with torch.no_grad():', '        mu, s = model(torch.tensor([[x]]))[0]', '    print(f"x = {x:+.1f}: middle {mu:+.3f}, spread {torch.exp(0.5 * s.clamp(-7, 7)):.3f} (true: {middle(torch.tensor(x)):+.3f}, {spread(torch.tensor(x)):.3f})")'];
    if (d.family === 'segment') return ['', 'model.eval()', 'with torch.no_grad():', '    pred = model(Xva[:1]).argmax(-1).view(G, G)', 'print("the shapes it found in the first validation picture (. nothing, o circle, # square, ^ triangle):")',
      'for row in pred:', '    print(" ".join(".o#^"[v] for v in row))'];
    return [];
  };

  // ------------------------------------------------------------------ the GAN and contrastive learning: whole scripts
  const RING = ['def real_points(n):', '    """Points on 8 small blobs around a ring."""', '    a = torch.randint(8, (n,)) * math.pi / 4', '    return torch.stack([0.75 * torch.cos(a), 0.75 * torch.sin(a)], 1) + 0.05 * torch.randn(n, 2)'];
  const MOONS = ['def real_points(n):', '    """Points on two interlocking crescents."""', '    t, up = torch.rand(n) * math.pi, torch.rand(n) < 0.5',
    '    x = torch.where(up, torch.cos(t), 1 - torch.cos(t)) * 0.62 - 0.31', '    y = torch.where(up, torch.sin(t) - 0.25, 0.25 - torch.sin(t)) * 0.62', '    return torch.stack([x, y], 1) + 0.04 * torch.randn(n, 2)'];
  const PR = ['def precision_recall(real, fake, k=3):', '    """Each set\'s points are balls reaching to their k-th nearest neighbour in that set. Precision: the share of generated',
    '    points inside a ball of the real ones (they look real); recall: the share of real points inside a ball of the', '    generated ones (they are all covered)."""',
    '    reach = lambda a: torch.cdist(a, a).fill_diagonal_(float("inf")).kthvalue(k, 1).values',
    '    p = (torch.cdist(fake, real) <= reach(real)[None, :]).any(1).float().mean().item()', '    r = (torch.cdist(real, fake) <= reach(fake)[None, :]).any(1).float().mean().item()', '    return p, r'];
  const READER = ['def read_digits(drawings, real, labels, reach):', '    """A digit reader: each drawing\'s 10 nearest real digits vote; it is read when 7 agree and the nearest is within',
    '    reach. Precision: the share read; recall: the share of the 10 digits among them (each 2% of the drawings or more)."""',
    '    near, idx = torch.cdist(drawings, real).topk(10, 1, largest=False)', '    top, which = F.one_hot(labels[idx], 10).sum(1).max(1)', '    read = (top >= 7) & (near[:, 0] <= reach)',
    '    counts = torch.bincount(which[read], minlength=10)', '    return read.float().mean().item(), (counts >= 0.02 * len(drawings)).float().mean().item()'];
  function ganScript(setup, d, chk, I) {
    const p = NA.builderTasks.byId[setup.problem], s = setup.ps[setup.problem], T = setup.train, steps = T.steps || 3000, z = d.z, K = d.K, wide = s.disc === 'big' ? 64 : 32, pics = d.pixels, every = Math.max(50, Math.round(steps / 10 / 50) * 50);
    const opt = (params) => (T.opt === 'sgd' ? `torch.optim.SGD(${params}, lr=${I.num(T.lr)})` : T.opt === 'momentum' ? `torch.optim.SGD(${params}, lr=${I.num(T.lr)}, momentum=0.9)` : `torch.optim.${T.opt === 'adamw' ? 'AdamW' : 'Adam'}(${params}, lr=${I.num(T.lr)}, betas=(0.5, 0.999)${T.wd ? `, weight_decay=${I.num(T.wd)}` : ''})`);
    return I.lines(
      `# Built in Neural Anatomy's model builder: ${p.label} (a GAN).`,
      `# The generator (your model): ${I.summary(setup, d)}`,
      `# Its rival, the discriminator: ${K} → ${wide} → ${wide} → 1, leaky ReLU (0.05): how sure it is that an example is real.`,
      `# Training:  each step the discriminator learns from real and generated examples, then the generator learns to fool it; ${T.opt === 'sgd' || T.opt === 'momentum' ? '' : 'Adam (β₁ 0.5), '}learning rate ${I.num(T.lr)}, batch ${T.batch}, ${steps.toLocaleString('en')} steps.`,
      '# Only torch is needed.',
      'import math', 'import random', 'import torch', 'import torch.nn as nn', 'import torch.nn.functional as F', '',
      `torch.manual_seed(${T.seed})`, `random.seed(${T.seed})`, '',
      `# ---------------------------------------------------------------- data: ${p.label.toLowerCase()}`,
      pics ? [I.DIGITS_PY, '', `n = ${s.n}`, 'real = torch.stack([draw_digit(i % 10) for i in range(n)]).flatten(1)   # (n, 196): real digits, ink in [0, 1]', 'labels = torch.arange(n) % 10                     # (only for the digit reader that scores it)']
        : [...(d.task === 'ring' ? RING : MOONS), '', `real = real_points(${s.n})                       # (n, 2)`],
      `Z = ${z}                                            # random numbers the generator is given`, '', '',
      I.modelCode(setup, d, chk), '', '',
      'model = Model()                                   # the generator', I.initCode(setup, d), '',
      `D = nn.Sequential(nn.Linear(${K}, ${wide}), nn.LeakyReLU(0.05), nn.Linear(${wide}, ${wide}), nn.LeakyReLU(0.05), nn.Linear(${wide}, 1))   # the discriminator`,
      T.init === 'small' ? ['for layer in D[::2]:                              # starting weights as on the page', '    init_(layer)'] : ['for layer, gain in zip(D[::2], (2.0, 2.0, 1.0)):   # starting weights as on the page: He before the leaky ReLUs', '    init_(layer, gain)'], '',
      pics ? ['def look(out):                                    # its pictures: the sigmoid of its outputs, 0 to 1 like the real ones', '    return torch.sigmoid(out)'] : ['def look(out):                                    # its points: its outputs as they are', '    return out'], '',
      `g_opt = ${opt('model.parameters()')}`, `d_opt = ${opt('D.parameters()')}`, '',
      pics ? [...READER, '', 'with torch.no_grad():                              # the reader\'s reach: a quarter more than how far 90 in 100 real digits are from their nearest other',
        '    near = torch.cdist(real[:300], real)', '    near[torch.arange(300), torch.arange(300)] = float("inf")', '    reach = 1.25 * near.min(1).values.quantile(0.9)'] : PR, '',
      'def quality():', '    """The F1 of precision and recall, with 200 of its examples."""', '    with torch.no_grad():', '        fake = look(model(torch.randn(200, Z)))',
      pics ? '    p, r = read_digits(fake, real, labels, reach)' : '    p, r = precision_recall(real[:200], fake)', '    return 2 * p * r / max(p + r, 1e-9), p, r', '',
      '# ---------------------------------------------------------------- training', `batch = ${T.batch}`, 'ones, zeros = torch.ones(batch, 1), torch.zeros(batch, 1)',
      `for step in range(${I.big(steps)}):`, '    real_batch = real[torch.randint(len(real), (batch,))]', '    fake = look(model(torch.randn(batch, Z)))',
      '    # the discriminator: 1 for the real examples, 0 for the generated', '    d_loss = F.binary_cross_entropy_with_logits(D(torch.cat([real_batch, fake.detach()])), torch.cat([ones, zeros]))',
      '    d_opt.zero_grad()', '    d_loss.backward()', T.clip ? `    torch.nn.utils.clip_grad_norm_(D.parameters(), ${I.num(T.clip)})` : null, '    d_opt.step()',
      '    # the generator: make it answer 1 for its own, −log D(G(z)), the gradient coming back through the discriminator', '    g_loss = F.softplus(-D(fake)).mean()',
      '    g_opt.zero_grad()', '    g_loss.backward()', T.clip ? `    torch.nn.utils.clip_grad_norm_(model.parameters(), ${I.num(T.clip)})` : null, '    g_opt.step()',
      `    if step % ${every} == 0:`, '        q, p, r = quality()', '        print(f"step {step:5d}  discriminator loss {d_loss.item():.3f}  generator loss {g_loss.item():.3f}  quality {q:.2f} (precision {p:.2f}, recall {r:.2f})")', '',
      'q, p, r = quality()', 'print(f"done: quality {q:.2f} (precision {p:.2f}, recall {r:.2f})")',
      pics ? ['', ...ASCII, 'with torch.no_grad():', '    print("drawn by the generator:")', '    show(list(look(model(torch.randn(6, Z)))))']
        : ['', 'with torch.no_grad():', '    pts = model(torch.randn(8, Z))', 'print("8 of its points:", [(round(x, 2), round(y, 2)) for x, y in pts.tolist()])',
          '# try: import matplotlib.pyplot as plt; plt.scatter(*real.T, s=2); plt.scatter(*model(torch.randn(500, Z)).detach().T, s=4); plt.show()']);
  }
  const VIEWS = String.raw`def shift(img, dx, dy):
    """img (C, G, G) moved dx pixels right and dy down; what moves off the edge is lost."""
    out = torch.zeros_like(img)
    out[:, max(dy, 0):G + min(dy, 0), max(dx, 0):G + min(dx, 0)] = img[:, max(-dy, 0):G + min(-dy, 0), max(-dx, 0):G + min(-dx, 0)]
    return out

def view(img, strong=True):
    """One random view of a picture: moved up to 2 pixels (1 if gentle), its strokes sometimes thickened, its ink
    fainter or stronger, speckled, and (strong views) a 4 x 4 square sometimes rubbed out."""
    s = 2 if strong else 1
    v = shift(img, random.randint(-s, s), random.randint(-s, s))
    if random.random() < (0.4 if strong else 0.2):
        v = torch.maximum(v, 0.7 * torch.stack([shift(v, 1, 0), shift(v, -1, 0), shift(v, 0, 1), shift(v, 0, -1)]).amax(0))
    if strong and random.random() < 0.3:
        x0, y0 = random.randrange(G - 4), random.randrange(G - 4)
        v[:, y0:y0 + 4, x0:x0 + 4] = 0
    k = random.uniform(0.6, 1.3) if strong else random.uniform(0.8, 1.15)
    return (v * k).clamp(0, 1) + (0.1 if strong else 0.04) * torch.randn_like(v)

def nt_xent(codes, tau):
    """NT-Xent (SimCLR): each code should be nearer its partner, the other view of the same picture, than any other code
    in the batch. A cross-entropy over the batch, the similarities of codes made length 1 divided by tau, whose right
    answer is the partner."""
    z = F.normalize(codes, dim=1)
    sim = (z @ z.T / tau).fill_diagonal_(float("-inf"))     # never itself
    n = len(z) // 2
    return F.cross_entropy(sim, torch.cat([torch.arange(n, 2 * n), torch.arange(n)]))

def nearest_vote(ref, ref_labels, query, query_labels, k=5):
    """The share of queries whose k most similar references (by angle) mostly have their label."""
    sim = F.normalize(query.flatten(1), dim=1) @ F.normalize(ref.flatten(1), dim=1).T
    votes = F.one_hot(ref_labels[sim.topk(k, 1).indices], CLASSES).sum(1)
    return (votes.argmax(1) == query_labels).float().mean().item()`;
  function contrastScript(setup, d, chk, I) {
    const p = NA.builderTasks.byId[setup.problem], s = setup.ps[setup.problem], T = setup.train, steps = T.steps || 1500, shapes = s.pictures === 'shapes', every = Math.max(50, Math.round(steps / 10 / 50) * 50);
    return I.lines(
      `# Built in Neural Anatomy's model builder: ${p.label} (contrastive learning, no labels).`,
      `# Model:     ${I.summary(setup, d)}: a code of ${d.output.k} numbers for each picture`,
      `# Training:  NT-Xent (τ ${s.tau}) on two ${s.views} views of each picture, ${({ sgd: 'SGD', momentum: 'SGD with momentum', adam: 'Adam', adamw: 'AdamW' })[T.opt]}, learning rate ${I.num(T.lr)}, batch ${T.batch} pictures (${2 * T.batch} views), ${steps.toLocaleString('en')} steps`,
      '# Only torch is needed.',
      'import math', 'import random', 'import torch', 'import torch.nn as nn', 'import torch.nn.functional as F', '',
      `torch.manual_seed(${T.seed})`, `random.seed(${T.seed})`, '',
      '# ---------------------------------------------------------------- data: pictures, their labels kept only to score',
      shapes ? I.SHAPES_PY : I.DIGITS_PY, '', `n, CLASSES = ${s.n}, ${shapes ? 5 : 10}`,
      shapes ? 'X = torch.stack([draw_shape(i % 5) for i in range(n)])      # (n, 1, 14, 14), anywhere already' : 'X = torch.stack([draw_digit(i % 10) for i in range(n)])     # (n, 1, 14, 14)',
      `labels = torch.arange(n) % CLASSES`, '', VIEWS, '',
      shapes ? null : 'X = torch.stack([shift(x, random.randint(-3, 3), random.randint(-3, 3)) for x in X])   # digits anywhere: shared pixels no longer mean the same digit',
      'perm = torch.randperm(n)', 'k = round(0.8 * n)', 'Xtr, ytr, Xva, yva = X[perm[:k]], labels[perm[:k]], X[perm[k:]], labels[perm[k:]]', '', '',
      I.modelCode(setup, d, chk), '', '',
      'model = Model()', I.initCode(setup, d), 'print(f"{sum(p.numel() for p in model.parameters()):,} learned numbers")', '', I.optCode(T), '',
      'def codes(x):', '    model.eval()', '    with torch.no_grad():', '        out = model(x)', '    model.train()', '    return out', '',
      'print(f"the nearest-vote on the raw pixels: {nearest_vote(Xtr[:1000], ytr[:1000], Xva[:300], yva[:300]):.3f}")', '',
      '# ---------------------------------------------------------------- training: no labels, only the two views', `batch, TAU, STRONG = ${T.batch}, ${s.tau}, ${s.views !== 'gentle' ? 'True' : 'False'}`,
      `for step in range(${I.big(steps)}):`, '    pics = Xtr[torch.randint(len(Xtr), (batch,))]',
      '    v = torch.stack([view(x, STRONG) for x in pics] + [view(x, STRONG) for x in pics])   # view i and view batch + i are partners',
      '    loss = nt_xent(model(v), TAU)', '    opt.zero_grad()', '    loss.backward()',
      T.clip ? `    torch.nn.utils.clip_grad_norm_(model.parameters(), ${I.num(T.clip)})   # cap the size of the step` : null, '    opt.step()', T.sched !== 'const' ? '    sched.step()' : null,
      `    if step % ${every} == 0:`, '        vote = nearest_vote(codes(Xtr[:1000]), ytr[:1000], codes(Xva[:300]), yva[:300])', '        print(f"step {step:5d}  loss {loss.item():.4f}  nearest codes vote {vote:.3f}")', '',
      'print(f"done: the nearest codes vote right {nearest_vote(codes(Xtr[:1000]), ytr[:1000], codes(Xva[:300]), yva[:300]):.3f} of the time")');
  }
  PG.script = function (setup, d, chk, I) {
    if (d.family === 'gan') return ganScript(setup, d, chk, I);
    if (d.family === 'contrast') return contrastScript(setup, d, chk, I);
    return null;
  };

  NA.builderPyGen = PG;
})(typeof globalThis !== 'undefined' ? (globalThis.NA = globalThis.NA || {}) : (window.NA = window.NA || {}));
