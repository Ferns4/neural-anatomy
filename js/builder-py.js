/* builder-py.js — what you built in the model builder, written out as one PyTorch script you can run: the data
   (the same generators as the page, in Python), the model, the loss, the optimizer and its schedule, and the
   training loop with the same measures the page shows. Only torch is needed. */
(function (NA) {
  'use strict';
  const BT = NA.builderTasks;
  const PY = {};

  const num = (v) => String(+(+v).toPrecision(6));
  const flt = (v) => { const s = num(v); return /[.e]/.test(s) ? s : s + '.0'; };
  const big = (v) => (v >= 10000 ? String(v).replace(/\B(?=(\d{3})+(?!\d))/g, '_') : String(v));
  const lines = (...xs) => xs.flat(Infinity).filter((x) => x !== null && x !== undefined && x !== false).join('\n');

  // ------------------------------------------------------------------ data
  const PLANE_PY = {
    blobs: { noise: 0.3, body: ['    c = i % 2', '    centre = torch.where(c[:, None] == 1, torch.tensor([0.38, 0.34]), torch.tensor([-0.4, -0.32]))', '    x = centre + 0.2 * torch.randn(n, 2)'] },
    xor: { noise: 0.25, body: ['    q = i % 4', '    sx, sy = (q % 2) * 2 - 1, (q // 2) * 2 - 1           # which corner', '    x = 0.5 * torch.stack([sx, sy], 1) + 0.13 * torch.randn(n, 2)', '    c = (sx * sy > 0).long()                       # opposite corners share a class'] },
    circle: { noise: 0.2, body: ['    c = (i % 2 == 0).long()                        # 1: the disc inside, 0: the ring', '    a = 2 * math.pi * torch.rand(n)', '    r = torch.where(c == 1, 0.42 * torch.rand(n).sqrt(), 0.62 + 0.33 * torch.rand(n))', '    x = torch.stack([r * torch.cos(a), r * torch.sin(a)], 1)'] },
    moons: { noise: 0.22, body: ['    c = i % 2', '    t = math.pi * torch.rand(n)', '    u = torch.where(c == 1, 1 - torch.cos(t), torch.cos(t))', '    v = torch.where(c == 1, 0.5 - torch.sin(t), torch.sin(t))', '    x = torch.stack([(u - 0.5) / 1.6, (v - 0.25) / 1.6], 1)'] },
    spiral: { noise: 0.1, body: ['    c = i % 2', '    t = torch.rand(n)                              # how far along its arm', '    r, a = 0.06 + 0.88 * t, 1.6 * 2 * math.pi * t + math.pi * c', '    x = torch.stack([r * torch.cos(a), r * torch.sin(a)], 1)'] },
    pinwheel: { noise: 0.1, body: ['    c = i % 3', '    t = torch.rand(n)', '    r, a = 0.08 + 0.86 * t, 0.85 * 2 * math.pi * t + 2 * math.pi * c / 3', '    x = torch.stack([r * torch.cos(a), r * torch.sin(a)], 1)'] },
  };
  const CURVE_PY = {
    sine: '0.8 * torch.sin(1.6 * math.pi * x)',
    abs: '1.4 * x.abs() - 0.6',
    bumps: '0.95 * torch.exp(-(x + 0.45) ** 2 / 0.02) - 0.7 * torch.exp(-(x - 0.4) ** 2 / 0.05)',
    step: '(x > 0.1).float() * 1.1 - 0.45',
  };
  const SPLIT = (frac, n = 'len(X)') => [
    `perm = torch.randperm(${n})`,
    `k = round(${frac} * ${n})`,
    'Xtr, Ytr, Xva, Yva = X[perm[:k]], Y[perm[:k]], X[perm[k:]], Y[perm[k:]]',
  ];
  // the digit strokes of js/data.js, in Python
  const DIGITS_PY = String.raw`G = 14                                        # pictures are G x G pixels
TAU = 2 * math.pi
U = random.uniform
S = lambda pts: (pts, True)                   # a smooth stroke through these points
P = lambda pts: (pts, False)                  # straight lines between them

def ellipse(cx, cy, rx, ry, a0, sweep, k=18):
    return [(cx + rx * math.cos(a0 + sweep * i / k), cy + ry * math.sin(a0 + sweep * i / k)) for i in range(k + 1)]

# a few ways to write each digit, in a unit box with y pointing down
TEMPLATES = {
    0: [lambda: [S(ellipse(0.5, 0.5, U(0.24, 0.36), U(0.4, 0.47), -math.pi / 2 + U(-0.4, 0.3), TAU + U(-0.15, 0.35), 22))]],
    1: [lambda: [P([(0.5, 0.05), (0.5, 0.95)])],
        lambda: [P([(0.3, 0.25 + U(-0.05, 0.05)), (0.53, 0.05), (0.53, 0.95)])],
        lambda: [P([(0.32, 0.24), (0.53, 0.05), (0.53, 0.94)]), P([(0.3, 0.95), (0.76, 0.95)])]],
    2: [lambda: [S([(0.2, 0.27), (0.33, 0.09), (0.55, 0.05), (0.75, 0.15), (0.77, 0.35), (0.6, 0.56), (0.38, 0.76), (0.18, 0.95)]), P([(0.18, 0.95), (0.84, 0.93 + U(-0.03, 0.03))])],
        lambda: [S([(0.22, 0.24), (0.4, 0.06), (0.66, 0.07), (0.76, 0.26), (0.64, 0.5), (0.4, 0.74), (0.2, 0.93), (0.36, 0.88), (0.58, 0.9), (0.84, 0.94)])]],
    3: [lambda: [S([(0.2, 0.16), (0.42, 0.05), (0.68, 0.08), (0.75, 0.25), (0.62, 0.42), (0.42, 0.48)]), S([(0.42, 0.48), (0.68, 0.53), (0.8, 0.72), (0.68, 0.92), (0.42, 0.96), (0.18, 0.86)])],
        lambda: [P([(0.2, 0.06), (0.78, 0.06), (0.45, 0.42)]), S([(0.45, 0.42), (0.72, 0.52), (0.8, 0.72), (0.66, 0.92), (0.4, 0.96), (0.18, 0.86)])]],
    4: [lambda: [P([(0.6, 0.05), (0.13, 0.66), (0.88, 0.66)]), P([(0.63, 0.05 + U(0, 0.25)), (0.63, 0.97)])],
        lambda: [P([(0.24, 0.05), (0.2, 0.6), (0.86, 0.6)]), P([(0.68, 0.08), (0.68, 0.97)])]],
    5: [lambda: [P([(0.78, 0.07), (0.3, 0.07), (0.27, 0.45)]), S([(0.27, 0.45), (0.5, 0.38), (0.72, 0.48), (0.78, 0.7), (0.65, 0.9), (0.42, 0.96), (0.2, 0.88)])],
        lambda: [P([(0.3, 0.07), (0.26, 0.44)]), S([(0.26, 0.44), (0.52, 0.38), (0.74, 0.5), (0.76, 0.74), (0.6, 0.93), (0.36, 0.95), (0.2, 0.86)]), P([(0.3, 0.07), (0.8, 0.06)])]],
    6: [lambda: [S([(0.7, 0.05), (0.48, 0.15), (0.3, 0.38), (0.24, 0.62), (0.3, 0.85), (0.5, 0.95), (0.7, 0.86), (0.74, 0.68), (0.6, 0.53), (0.4, 0.53), (0.27, 0.66)])]],
    7: [lambda: [P([(0.15, 0.08), (0.85, 0.08), (0.42, 0.95)])],
        lambda: [P([(0.15, 0.08), (0.85, 0.08), (0.42, 0.95)]), P([(0.36, 0.52), (0.74, 0.52)])],
        lambda: [P([(0.15, 0.08), (0.85, 0.08)]), S([(0.85, 0.08), (0.64, 0.38), (0.5, 0.68), (0.45, 0.95)])]],
    8: [lambda: [S(ellipse(0.5, 0.27, U(0.17, 0.23), 0.21, -math.pi / 2, TAU + 0.2)), S(ellipse(0.5, 0.72, U(0.22, 0.28), 0.24, -math.pi / 2, TAU + 0.2))],
        lambda: [S([(0.68, 0.12), (0.5, 0.05), (0.32, 0.13), (0.32, 0.33), (0.5, 0.47), (0.7, 0.62), (0.72, 0.84), (0.5, 0.96), (0.28, 0.84), (0.3, 0.62), (0.5, 0.47), (0.68, 0.32), (0.68, 0.12)])]],
    9: [lambda: [S(ellipse(0.47, 0.3, 0.24, U(0.18, 0.24), 0, TAU + 0.1)), P([(0.71, 0.3), (0.68 + U(-0.06, 0.04), 0.96)])],
        lambda: [S(ellipse(0.47, 0.3, 0.24, 0.22, 0, TAU + 0.1)), S([(0.71, 0.28), (0.7, 0.6), (0.62, 0.85), (0.45, 0.96)])]],
}

def smooth(pts, k=5):
    """Catmull-Rom curve through the points, k samples per gap."""
    if len(pts) < 3:
        return pts
    out = []
    for i in range(len(pts) - 1):
        p0, p1, p2, p3 = pts[max(0, i - 1)], pts[i], pts[i + 1], pts[min(len(pts) - 1, i + 2)]
        for s in range(k):
            t = s / k
            out.append(tuple(0.5 * (2 * p1[d] + (p2[d] - p0[d]) * t + (2 * p0[d] - 5 * p1[d] + 4 * p2[d] - p3[d]) * t * t
                                    + (3 * p1[d] - p0[d] - 3 * p2[d] + p3[d]) * t ** 3) for d in (0, 1)))
    return out + [pts[-1]]

centres = torch.stack(torch.meshgrid(torch.arange(G), torch.arange(G), indexing="ij"), -1).flip(-1).reshape(-1, 2) + 0.5

def draw_digit(d):
    """One handwritten-looking digit: a random template, jittered, slanted, rotated, scaled, then drawn with a pen."""
    shear, angle, sx = random.gauss(0, 0.2), random.gauss(0, 0.13), U(0.72, 1.18)
    co, si = math.cos(angle), math.sin(angle)
    strokes = []
    for pts, is_smooth in random.choice(TEMPLATES[d])():
        pts = [(x + random.gauss(0, 0.036), y + random.gauss(0, 0.036)) for x, y in pts]
        if is_smooth:
            pts = smooth(pts)
        q = []
        for x, y in pts:
            x, y = (x - 0.5) * sx, y - 0.5
            x += shear * y
            q.append((x * co - y * si, x * si + y * co))
        strokes.append(torch.tensor(q))
    every = torch.cat(strokes)
    lo, hi = every.min(0).values, every.max(0).values
    scale = 0.74 * U(0.88, 1.08) / (hi - lo).max().clamp(min=1e-6)
    shift = torch.tensor([random.gauss(0, 0.035), random.gauss(0, 0.035)])
    r = 0.105 * U(0.7, 1.4) * G / 2                    # half the pen width, in pixels
    img = torch.zeros(G * G)
    for q in strokes:
        q = (0.5 + shift + (q - (lo + hi) / 2) * scale) * G
        a, ab = q[:-1], q[1:] - q[:-1]                 # line segments
        t = (((centres[:, None] - a) * ab).sum(-1) / (ab * ab).sum(-1).clamp(min=1e-12)).clamp(0, 1)
        dist = (a + t[..., None] * ab - centres[:, None]).norm(dim=-1)
        img = torch.maximum(img, (r + 0.5 - dist).clamp(0, 1).max(1).values)
    return img.view(1, G, G)`;
  // the tiny language of js/lm-data.js, in Python
  const LANGUAGE_PY = String.raw`WORDS = ("<bos> . alice bob carol dave emma frank tea milk juice coffee water likes drinks the "
         "big small happy sleepy cat dog bird fox cats dogs birds foxes near with sleeps runs sings sleep run sing").split()
ID = {w: i for i, w in enumerate(WORDS)}
NAMES, DRINKS, ADJ = WORDS[2:8], WORDS[8:13], WORDS[16:20]
SG, PL, PREP, VSG, VPL = WORDS[20:24], WORDS[24:28], WORDS[28:30], WORDS[30:33], WORDS[33:36]

def sentence(out, short):
    """'the [adj] cat [near the dogs] sleeps .': the verb agrees with the first animal."""
    plural = random.random() < 0.5
    out.append("the")
    if not short and random.random() < 0.35:
        out.append(random.choice(ADJ))
    out.append(random.choice(PL if plural else SG))
    if random.random() < 0.6:
        out += [random.choice(PREP), "the"]
        if not short and random.random() < 0.25:
            out.append(random.choice(ADJ))
        other = (not plural) if random.random() < 0.7 else plural
        out.append(random.choice(PL if other else SG))
    out += [random.choice(VPL if plural else VSG), "."]

def story():
    out = ["<bos>"]
    if random.random() < 0.8:                     # 'alice likes tea . bob likes milk . alice drinks tea .'
        k = 2 if random.random() < 0.55 else 3
        names, drinks = random.sample(NAMES, k), random.sample(DRINKS, k)
        two = k == 2 and random.random() < 0.5
        filler = random.randrange(2) if k == 2 and not two and random.random() < 0.45 else -1
        for p in range(k):
            out += [names[p], "likes", drinks[p], "."]
            if p == filler:
                sentence(out, True)
        for q in (random.sample([0, 1], 2) if two else [random.randrange(k)]):
            out += [names[q], "drinks", drinks[q], "."]
    else:
        sentence(out, False)
        sentence(out, False)
    return [ID[w] for w in out]`;

  function dataCode(setup, d) {
    const p = BT.byId[setup.problem], s = setup.ps[setup.problem] || BT.defaults(p), fam = p.family, out = [];
    out.push(`# ---------------------------------------------------------------- data: ${p.label.toLowerCase()}`);
    if (fam === 'plane' && !p.custom) {
      const g = PLANE_PY[p.dataset];
      out.push(`def make_points(n, noise):`, `    i = torch.arange(n)`, ...g.body,
        `    x = (x + noise * ${g.noise} * torch.randn(n, 2)).clamp(-1, 1)`, `    return x, c.long()`, '',
        `X, Y = make_points(${s.n}, ${num(s.noise)})          # X: (${s.n}, 2) points, Y: their classes`, ...SPLIT(0.75));
    } else if (fam === 'plane') {
      const pts = setup.custom.points, used = [...new Set(pts.map((q) => q.c))].sort();
      const rows = [];
      for (let i = 0; i < pts.length; i += 4) rows.push('    ' + pts.slice(i, i + 4).map((q) => `(${num(q.x)}, ${num(q.y)}, ${used.indexOf(q.c)})`).join(', ') + ',');
      out.push(`# the ${pts.length} points you placed: (x, y, class)`, 'points = torch.tensor([', ...rows, '])',
        'X, Y = points[:, :2], points[:, 2].long()', ...SPLIT(0.8));
    } else if (fam === 'curve') {
      if (p.custom) {
        const c = BT.compile(s.formula);
        out.push(`# y = ${s.formula}`, 'x = 2 * torch.rand(' + s.n + ') - 1', `y = ${c.error ? '0 * x  # (the formula has an error on the page)' : c.constant ? `(${c.py}) + 0 * x` : c.py}`);
      } else out.push(`x = -1 + 2 * (torch.arange(${s.n}) + torch.rand(${s.n})) / ${s.n}     # spread evenly over [-1, 1]`, `y = ${CURVE_PY[p.dataset]}`);
      out.push('top = y.abs().max()', 'if top > 1.5 or top < 0.25:                      # keep the values near 1 (the page does too)', '    y = y / top',
        `X, Y = x[:, None], y + ${num(s.noise)} * torch.randn_like(y)          # X: (${s.n}, 1), Y: noisy values`, ...SPLIT(0.75));
    } else if (fam === 'table') {
      const t = setup.custom.table;
      if (!t || d.error) out.push(`# ${d.error || 'Choose a CSV file on the page first.'}`);
      else {
        const used = [...d.feats, d.target], nums = d.kind === 'value' || BT.tableColumns(t).find((c) => c.index === d.target).numeric;
        const cell = (v, i) => (i < d.feats.length || nums ? num(Number(v)) : JSON.stringify(v));
        // a small table goes into the script; a big one is read from the file
        const keep = t.rows.filter((r) => d.feats.every((i) => r[i] !== '' && Number.isFinite(Number(r[i]))) && r[d.target] !== '' && (d.kind === 'classes' || Number.isFinite(Number(r[d.target]))));
        if (keep.length * used.length <= 2400) {
          out.push(`# ${t.name}: the ${keep.length} complete rows the page used, as [${used.map((i) => t.header[i]).join(', ')}]`, 'rows = [',
            ...keep.map((r) => '    [' + used.map((i, k) => cell(r[i], k)).join(', ') + '],'), ']',
            `FEATURES, TARGET = list(range(${d.feats.length})), ${d.feats.length}`);
        } else {
          out.push(`# ${t.name}: put the file next to this script`, `with open(${JSON.stringify(t.name)}, newline="") as f:`, `    rows = list(csv.reader(f${t.delim !== ',' ? `, delimiter=${JSON.stringify(t.delim)}` : ''}))`,
            t.hasHeader ? 'rows = rows[1:]                                   # the first row names the columns' : null,
            t.delim === ';' ? 'rows = [[c.replace(",", ".") for c in r] for r in rows]   # with semicolons, 1,5 means 1.5 (the page reads it so too)' : null,
            `FEATURES = [${d.feats.join(', ')}]   # ${d.featNames.join(', ')}`, `TARGET = ${d.target}   # ${d.targetName}`,
            'def is_number(s):', '    try:', '        return math.isfinite(float(s))', '    except ValueError:', '        return False',
            `rows = [r for r in rows if len(r) > max(FEATURES + [TARGET]) and all(is_number(r[i]) for i in FEATURES) and r[TARGET].strip() != ""${d.kind === 'value' ? ' and is_number(r[TARGET])' : ''}]`);
        }
        out.push('X = torch.tensor([[float(r[i]) for i in FEATURES] for r in rows])',
          'X = (X - X.mean(0)) / X.std(0, correction=0).clamp_min(1e-8)   # each input column to mean 0, spread 1 (the page does too)');
        if (d.kind === 'value') out.push('y = torch.tensor([float(r[TARGET]) for r in rows])', 'Y = (y - y.mean()) / y.std(correction=0)       # the value too; R² is the same either way');
        else if (nums) out.push(`CLASSES = [${d.names.map((v) => num(Number(v))).join(', ')}]`, 'Y = torch.tensor([CLASSES.index(float(r[TARGET])) for r in rows])');
        else out.push(`CLASSES = [${d.names.map((v) => JSON.stringify(v)).join(', ')}]`, 'Y = torch.tensor([CLASSES.index(r[TARGET].strip()) for r in rows])');
        out.push(...SPLIT(0.8));
      }
    } else if (fam === 'digits') {
      out.push(DIGITS_PY, '', `n = ${s.n}`, 'X = torch.stack([draw_digit(i % 10) for i in range(n)])     # (n, 1, 14, 14), ink in [0, 1]',
        'Y = torch.arange(n) % 10', ...SPLIT(0.8));
    } else if (p.task === 'language') {
      out.push(LANGUAGE_PY, '', `T = ${d.input.len}                                        # every story is padded or cut to T tokens`,
        `def make_stories(n):`, `    X, Y = torch.full((n, T), ID["."]), torch.full((n, T), -1)       # -1: no target (padding)`, '    for i in range(n):',
        '        ids = story()[:T + 1]', '        X[i, :len(ids[:T])] = torch.tensor(ids[:T])', '        Y[i, :len(ids) - 1] = torch.tensor(ids[1:])      # predict the next word', '    return X, Y', '',
        `X, Y = make_stories(${s.n})`, `k = ${s.n - Math.round(s.n * 0.12)}`, 'Xtr, Ytr, Xva, Yva = X[:k], Y[:k], X[k:], Y[k:]');
    } else if (p.task === 'sentences') {
      const text = String(s.text || '').split('\n');
      out.push('text = (' + text.map((t, i) => (i ? '        ' : '') + JSON.stringify(t + (i < text.length - 1 ? '\n' : ''))).join('\n') + ')',
        `words = re.sub(r"([.,!?;:])", r" \\1 ", text.lower()).split()   # words and punctuation are tokens`,
        'WORDS = list(dict.fromkeys(words))[:400]', 'ID = {w: i for i, w in enumerate(WORDS)}', 'ids = torch.tensor([ID[w] for w in words if w in ID])',
        `T = ${d.input ? d.input.len : s.window}                                        # the context window`,
        'windows = torch.stack([ids[i:i + T + 1] for i in range(len(ids) - T)])', 'X, Y = windows[:, :-1], windows[:, 1:]   # every position predicts the next word',
        `perm = torch.randperm(len(X))`, `k = len(X) - (round(0.1 * len(X)) if len(X) >= 40 else 0)`, 'Xtr, Ytr, Xva, Yva = X[perm[:k]], Y[perm[:k]], X[perm[k:]], Y[perm[k:]]',
        'if len(Xva) == 0:', '    Xva, Yva = Xtr, Ytr                       # too little text to hold some back');
    } else {
      const L = s.len, ans = p.task === 'reverse' ? 'xs.flip(1)' : p.task === 'sort' ? 'xs.sort(1).values' : 'xs';
      out.push(`L = ${L}                                         # digits per sequence; token 10 is the arrow`,
        `def make_sequences(n):`, `    xs = torch.unique(torch.randint(0, 10, (3 * n, L)), dim=0)       # distinct inputs`,
        `    xs = xs[torch.randperm(len(xs))][:min(n, 10 ** L - 50)]`, `    seq = torch.cat([xs, torch.full((len(xs), 1), 10), ${ans}], 1)   # x1..xL → answer`,
        '    X, Y = seq[:, :-1], seq[:, 1:].clone()', '    Y[:, :L] = -1                           # train only on the answer (the inputs are random)', '    return X, Y', '',
        `X, Y = make_sequences(${s.n})`, 'k = len(X) - max(50, round(0.12 * len(X)))', 'Xtr, Ytr, Xva, Yva = X[:k], Y[:k], X[k:], Y[k:]');
    }
    return out;
  }

  // ------------------------------------------------------------------ the model
  const ACT_PY = {
    relu: (z) => `F.relu(${z})`, leaky: (z) => `F.leaky_relu(${z}, 0.05)`, elu: (z) => `F.elu(${z})`, gelu: (z) => `F.gelu(${z}, approximate="tanh")`, swish: (z) => `F.silu(${z})`,
    softplus: (z) => `F.softplus(${z})`, tanh: (z) => `torch.tanh(${z})`, sigmoid: (z) => `torch.sigmoid(${z})`, sine: (z) => `torch.sin(${z})`,
    gaussian: (z) => `torch.exp(-${z} ** 2)`, linear: (z) => z,
  };
  const ACT_FN = { relu: 'F.relu', leaky: 'lambda z: F.leaky_relu(z, 0.05)', elu: 'F.elu', gelu: 'lambda z: F.gelu(z, approximate="tanh")', swish: 'F.silu', softplus: 'F.softplus', tanh: 'torch.tanh', sigmoid: 'torch.sigmoid', sine: 'torch.sin', gaussian: 'lambda z: torch.exp(-z ** 2)', linear: 'lambda z: z' };
  const shp = (s) => (s ? '(batch, ' + s.join(', ') + ')' : '');
  const MIXED_PY = String.raw`class MixedActivation(nn.Module):
    """A different activation for each neuron of a layer (the dots you changed on the page)."""
    def __init__(self, acts):
        super().__init__()
        self.groups = {}
        for j, a in enumerate(acts):
            self.groups.setdefault(a, []).append(j)

    def forward(self, z):
        out = torch.empty_like(z)
        for a, idx in self.groups.items():
            out[..., idx] = ACT[a](z[..., idx])
        return out`;
  const ATTN_PY = String.raw`class SelfAttention(nn.Module):
    """Every position makes a query, a key and a value; it takes the values of the positions whose keys match its
    query best (a softmax over scaled dot products). Heads do this in parallel on slices of the vector."""
    def __init__(self, d, heads, causal):
        super().__init__()
        self.heads, self.causal = heads, causal
        self.q, self.k, self.v, self.o = (nn.Linear(d, d) for _ in range(4))

    def forward(self, x):
        B, T, D = x.shape
        split = lambda t: t.view(B, T, self.heads, D // self.heads).transpose(1, 2)    # (B, heads, T, D / heads)
        q, k, v = split(self.q(x)), split(self.k(x)), split(self.v(x))
        scores = q @ k.transpose(-2, -1) / math.sqrt(D // self.heads)
        if self.causal:                                                                 # only look back
            scores = scores.masked_fill(torch.ones(T, T, dtype=torch.bool).triu(1), float("-inf"))
        return self.o((scores.softmax(-1) @ v).transpose(1, 2).reshape(B, T, D))`;
  const BLOCK_PY = String.raw`class TransformerBlock(nn.Module):
    """Pre-norm transformer block: x + attention(norm(x)), then x + MLP(norm(x))."""
    def __init__(self, d, heads, mlp, causal):
        super().__init__()
        self.norm1, self.norm2 = nn.LayerNorm(d), nn.LayerNorm(d)
        self.attn = SelfAttention(d, heads, causal)
        self.mlp = nn.Sequential(nn.Linear(d, round(mlp * d)), nn.GELU(approximate="tanh"), nn.Linear(round(mlp * d), d))

    def forward(self, x):
        x = x + self.attn(self.norm1(x))
        return x + self.mlp(self.norm2(x))`;

  function modelCode(setup, d, chk) {
    const blocks = setup.blocks, inp = d.input, out = d.output, init = [], fwd = [], helpers = new Set();
    if (inp.kind === 'tokens') {
      init.push(`self.tok = nn.Embedding(${inp.vocab}, ${inp.embed})           # a learned vector for each token`, `self.pos = nn.Embedding(${inp.len}, ${inp.embed})           # and one for each position`);
      fwd.push(`x = self.tok(x) + self.pos(torch.arange(x.shape[1]))`.padEnd(54) + `# ${shp([inp.len, inp.embed])}`);
    }
    blocks.forEach((b, i) => {
      const sh = chk.shapes[i] || {}, si = sh.in || [], so = sh.out || [], n = i + 1, last = si[si.length - 1];
      const put = (code) => fwd.push(code.padEnd(54) + `# ${shp(so)}`);
      if (b.type === 'dense') {
        init.push(`self.dense${n} = nn.Linear(${last}, ${b.units})`);
        let z = `self.dense${n}(x)`;
        if (b.acts) { helpers.add('mixed'); init.push(`self.act${n} = MixedActivation(${JSON.stringify(b.acts)})`); z = `self.act${n}(${z})`; }
        else z = ACT_PY[b.act](z);
        put(b.residual && last === b.units ? `x = x + ${z}` : `x = ${z}`); // the engine adds the input back only when the width is kept
      } else if (b.type === 'activation') put(`x = ${ACT_PY[b.act]('x')}`);
      else if (b.type === 'dropout') { init.push(`self.drop${n} = nn.Dropout(${num(b.p)})`); put(`x = self.drop${n}(x)`); }
      else if (b.type === 'layernorm') { init.push(`self.norm${n} = nn.LayerNorm(${last})`); put(`x = self.norm${n}(x)`); }
      else if (b.type === 'attention') {
        helpers.add('attn');
        init.push(`self.attn${n} = SelfAttention(${last}, heads=${b.heads}, causal=${b.causal !== false ? 'True' : 'False'})`);
        put(b.residual !== false ? `x = x + self.attn${n}(x)` : `x = self.attn${n}(x)`);
      } else if (b.type === 'transformer') {
        helpers.add('attn'); helpers.add('block');
        init.push(`self.block${n} = TransformerBlock(${last}, heads=${b.heads}, mlp=${b.mlp || 4}, causal=${b.causal !== false ? 'True' : 'False'})`);
        put(`x = self.block${n}(x)`);
      } else if (b.type === 'conv') {
        init.push(`self.conv${n} = nn.Conv2d(${si[0]}, ${b.filters}, ${b.k}, stride=${b.stride || 1}, padding=${(b.k - 1) >> 1})`);
        put(`x = ${ACT_PY[b.act](`self.conv${n}(x)`)}`);
      } else if (b.type === 'pool') put(`x = F.max_pool2d(x, ${b.size})                     # leftover rows and columns are dropped`);
      else if (b.type === 'flatten') put('x = x.flatten(1)');
      else if (b.type === 'gap') put(si.length === 3 ? 'x = x.mean((2, 3))' : 'x = x.mean(1)');
    });
    const fin = chk.shapes.length ? (chk.shapes[chk.shapes.length - 1] || {}).out : inp.kind === 'tokens' ? [inp.len, inp.embed] : inp.kind === 'vector' ? [inp.dim] : [inp.c, inp.h, inp.w];
    const dIn = fin ? fin[fin.length - 1] : 1, K = out.kind === 'tokens' ? out.vocab : out.k;
    init.push(`self.out = nn.Linear(${dIn}, ${K})${out.kind === 'classes' ? '                # one score per class' : out.kind === 'tokens' ? '                # a score for every possible next token' : ''}`);
    fwd.push(`return self.out(x)`.padEnd(54) + `# ${shp(out.kind === 'tokens' ? [inp.len, K] : [K])}`);
    const pre = [];
    if (helpers.has('mixed')) pre.push(`ACT = {${Object.keys(ACT_FN).map((a) => `"${a}": ${ACT_FN[a]}`).join(', ')}}`, '', MIXED_PY, '', '');
    if (helpers.has('attn')) pre.push(ATTN_PY, '', '');
    if (helpers.has('block')) pre.push(BLOCK_PY, '', '');
    return lines('# ---------------------------------------------------------------- the model', ...pre,
      'class Model(nn.Module):', '    def __init__(self):', '        super().__init__()', init.map((l) => '        ' + l), '',
      '    def forward(self, x):' + `                                   # x: ${shp(inp.kind === 'tokens' ? [inp.len] : inp.kind === 'vector' ? [inp.dim] : [inp.c, inp.h, inp.w])}`, fwd.map((l) => '        ' + l));
  }

  // ------------------------------------------------------------------ loss, optimizer, training
  function lossCode(setup, d) {
    const L = setup.loss, o = d.output, K = o.kind === 'tokens' ? o.vocab : o.k;
    if (o.kind === 'tokens') return [`def loss_fn(out, y):                          # cross-entropy at every position with a target`, `    return F.cross_entropy(out.reshape(-1, ${K}), y.reshape(-1), ignore_index=-1${L.smooth ? `, label_smoothing=${num(L.smooth)}` : ''})`];
    if (o.kind === 'value') {
      if (L.id === 'mae') return ['def loss_fn(out, y):                          # absolute error', '    return (out[:, 0] - y).abs().mean()'];
      if (L.id === 'huber') return [`def loss_fn(out, y):                          # Huber: squared near the target, absolute beyond ${num(L.delta || 0.2)}`, `    return F.huber_loss(out[:, 0], y, delta=${num(L.delta || 0.2)})`];
      return ['def loss_fn(out, y):                          # squared error, ½(prediction − target)²', '    return 0.5 * ((out[:, 0] - y) ** 2).mean()'];
    }
    if (L.id === 'mse') return ['def loss_fn(out, y):                          # squared error between the probabilities and the one-hot target', `    return 0.5 * ((out.softmax(1) - F.one_hot(y, ${K})) ** 2).sum(1).mean()`];
    if (L.id === 'hinge') return ['def loss_fn(out, y):                          # multi-class hinge: beat every other class by a margin of 1', '    right = out.gather(1, y[:, None])', '    return F.relu(1 + out - right).scatter(1, y[:, None], 0).sum(1).mean()'];
    return [`def loss_fn(out, y):                          # cross-entropy: −log of the probability of the right class`, `    return F.cross_entropy(out, y${L.smooth ? `, label_smoothing=${num(L.smooth)}` : ''})`];
  }
  function optCode(T) {
    const steps = T.steps || 3000, params = T.wd ? 'groups' : 'model.parameters()', out = [];
    if (T.wd) out.push('# weight decay on the weights only, not on biases, norms and embeddings', 'decay = [p for n, p in model.named_parameters() if p.dim() > 1 and not n.startswith(("tok.", "pos."))]', 'rest = [p for n, p in model.named_parameters() if not (p.dim() > 1 and not n.startswith(("tok.", "pos.")))]', `groups = [{"params": decay, "weight_decay": ${num(T.wd)}}, {"params": rest, "weight_decay": 0.0}]`);
    const opt = { sgd: `torch.optim.SGD(${params}, lr=${num(T.lr)})`, momentum: `torch.optim.SGD(${params}, lr=${num(T.lr)}, momentum=0.9)`, adam: `torch.optim.Adam(${params}, lr=${num(T.lr)})`, adamw: `torch.optim.AdamW(${params}, lr=${num(T.lr)}${T.wd ? '' : ', weight_decay=0.0'})` }[T.opt];
    out.push(`opt = ${opt}`);
    if (T.sched === 'cosine') out.push(`sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=${steps})        # lr falls along half a cosine`);
    else if (T.sched === 'step') out.push(`sched = torch.optim.lr_scheduler.MultiStepLR(opt, milestones=[${Math.round(steps / 2)}, ${Math.round(steps * 0.75)}], gamma=0.1)`);
    else if (T.sched === 'warmup') out.push(`warm = ${Math.max(1, Math.round(steps * 0.05))}                                    # warm up, then a cosine`, `sched = torch.optim.lr_scheduler.LambdaLR(opt, lambda s: s / warm if s < warm else 0.5 * (1 + math.cos(math.pi * (s - warm) / (${steps} - warm))))`);
    return out;
  }
  /* The page's starting weights: normal with sd √(gain / inputs), gain 2 (He) before a ReLU-like activation when
     the initialisation is automatic (always with He), 1 otherwise and for the output and attention projections;
     sd 0.02 everywhere for "tiny"; embeddings N(0, 0.1); biases 0. */
  function initCode(setup, d) {
    const mode = setup.train.init, small = mode === 'small', HE = (a) => !!(NA.engine.ACTIVATIONS[a] || {}).he;
    const gain = (he) => (mode === 'he' || (mode === 'auto' && he) ? 2 : 1);
    const call = (layer, g) => (small || g === 1 ? `init_(${layer})` : Array.isArray(g) ? `init_(${layer}, rows=[${g.join(', ')}])` : `init_(${layer}, ${g})`);
    const out = ['# ---------------------------------------------------------------- starting weights, as on the page'];
    if (small) out.push('def init_(layer):', '    """Every weight from a normal distribution with sd 0.02; zero biases."""', '    with torch.no_grad():', '        layer.weight.normal_(0.0, 0.02)', '        layer.bias.zero_()');
    else out.push('def init_(layer, gain=1.0, rows=None):', '    """Normal weights with sd √(gain / inputs): gain 2 (He) suits ReLU-like activations, 1 the rest. Zero biases."""', '    with torch.no_grad():',
      '        fan_in = layer.weight[0].numel()', '        for j, g in enumerate(rows or [gain] * layer.weight.shape[0]):', '            layer.weight[j].normal_(0.0, math.sqrt(g / fan_in))', '        layer.bias.zero_()');
    out.push('');
    if (d.input.kind === 'tokens') out.push('nn.init.normal_(model.tok.weight, 0.0, 0.1)', 'nn.init.normal_(model.pos.weight, 0.0, 0.1)');
    setup.blocks.forEach((b, i) => {
      const n = i + 1, attn = (m) => `for layer in (${['q', 'k', 'v', 'o'].map((x) => `${m}.${x}`).join(', ')}):`;
      if (b.type === 'dense') out.push(call(`model.dense${n}`, b.acts && !small ? b.acts.map((a) => gain(HE(a))) : gain(HE(b.act))));
      else if (b.type === 'conv') out.push(call(`model.conv${n}`, gain(HE(b.act))));
      else if (b.type === 'attention') out.push(attn(`model.attn${n}`), '    init_(layer)');
      else if (b.type === 'transformer') out.push(attn(`model.block${n}.attn`), '    init_(layer)', call(`model.block${n}.mlp[0]`, gain(true)), call(`model.block${n}.mlp[2]`, mode === 'he' ? 2 : 1));
    });
    out.push('init_(model.out)');
    return out;
  }
  function scoreCode(d) {
    if (d.output.kind === 'value') return ['def score(out, y):                            # R²: 1 is perfect, 0 is no better than the mean', '    return 1 - ((out[:, 0] - y) ** 2).sum() / ((y - y.mean()) ** 2).sum()'];
    if (d.output.kind === 'tokens') return ['def score(out, y):                            # share of target tokens predicted right', '    mask = y != -1', '    return (out.argmax(-1)[mask] == y[mask]).float().mean()'];
    return ['def score(out, y):                            # accuracy', '    return (out.argmax(1) == y).float().mean()'];
  }
  function afterCode(setup, d) {
    const p = BT.byId[setup.problem];
    if (d.family !== 'sequence') return [];
    if (p.task === 'language' || p.task === 'sentences') {
      const prompt = p.task === 'language' ? 'alice likes tea . bob likes milk . alice drinks' : BT.wordsOf(setup.ps[setup.problem].text).slice(0, 3).join(' ');
      return ['', '# ---------------------------------------------------------------- write with it', 'model.eval()', 'def write(prompt, count=12):', '    ids = [ID[w] for w in re.sub(r"([.,!?;:])", r" \\1 ", prompt.lower()).split() if w in ID]', '    with torch.no_grad():',
        '        for _ in range(count):', '            ctx = ids[-T:]', '            x = torch.tensor([ctx[0]] * (T - len(ctx)) + ctx)[None]', '            ids.append(model(x)[0, -1].argmax().item())     # the likeliest next word', '    return " ".join(WORDS[i] for i in ids)', '',
        `print(write(${JSON.stringify(prompt)}))`];
    }
    return ['', '# ---------------------------------------------------------------- use it: write the answer one digit at a time', 'model.eval()', 'def answer(digits):', '    x = torch.tensor(digits + [10] + [0] * (L - 1))[None]',
      '    out = []', '    with torch.no_grad():', '        for k in range(L):', '            d = model(x)[0, L + k].argmax().item()', '            out.append(d)', '            if L + k + 1 < 2 * L:', '                x[0, L + k + 1] = d                  # feed the guess back in', '    return out', '',
      `print([3, 1, 4, 1, 5, 9][:L], "->", answer([3, 1, 4, 1, 5, 9][:L]))`];
  }

  // A one-line description of the model, for the comment at the top
  function summary(setup, d) {
    const L = NA.builderLab && NA.builderLab.blockLabel ? NA.builderLab.blockLabel : (b) => b.type;
    const o = d.output, end = o.kind === 'classes' ? `${o.k} class scores` : o.kind === 'value' ? '1 number' : `${o.vocab} next-token scores`;
    return [...(d.input.kind === 'tokens' ? ['embedding'] : []), ...setup.blocks.map(L), end].join(' → ');
  }

  PY.code = function (setup, d) {
    const p = BT.byId[setup.problem], T = setup.train, steps = T.steps || 3000;
    if (!d || d.error) return `# ${d && d.error ? d.error : 'Pick a problem first.'}`;
    const chk = NA.bx ? NA.bx.check(setup.blocks, d.input, d.output) : { ok: false, shapes: [] };
    if (!chk.ok) return lines('# The model does not fit together yet, so there is nothing to write out.', `# ${((chk.problem || (chk.shapes.find((s) => s && s.problem) || {}).problem) || { reason: 'See the block marked in red.' }).reason}`);
    const needs = ['import math', p.family === 'table' && setup.custom.table && setup.custom.table.rows.length * (d.feats ? d.feats.length + 1 : 1) > 2400 ? 'import csv' : null, d.family === 'digits' || p.task === 'language' ? 'import random' : null, p.task === 'sentences' || p.task === 'language' ? 'import re' : null];
    const lossName = NA.bx.LOSSES[setup.loss.id] ? NA.bx.LOSSES[setup.loss.id].label : setup.loss.id;
    const optName = { sgd: 'SGD', momentum: 'SGD with momentum', adam: 'Adam', adamw: 'AdamW' }[T.opt];
    const evalEvery = Math.max(50, Math.round(steps / 10 / 50) * 50);
    return lines(
      `# Built in Neural Anatomy's model builder: ${p.label}.`,
      `# Model:     ${summary(setup, d)}`,
      `# Training:  ${lossName} loss, ${optName}, learning rate ${num(T.lr)}, batch ${T.batch}, ${steps.toLocaleString('en')} steps${T.steps ? '' : ' (the page trains until you pause)'}`,
      ...needs.filter(Boolean), 'import torch', 'import torch.nn as nn', 'import torch.nn.functional as F', '',
      `torch.manual_seed(${T.seed})`, d.family === 'digits' || p.task === 'language' ? `random.seed(${T.seed})` : null, '',
      dataCode(setup, d), '', '',
      modelCode(setup, d, chk), '', '',
      'model = Model()', initCode(setup, d), 'print(f"{sum(p.numel() for p in model.parameters()):,} learned numbers")', '',
      '# ---------------------------------------------------------------- loss, optimizer and measure', lossCode(setup, d), '', scoreCode(d), '', optCode(T), '',
      'def evaluate(X, Y):', '    model.eval()', '    with torch.no_grad():', '        out = model(X)', '        loss, s = loss_fn(out, Y).item(), score(out, Y).item()', '    model.train()', '    return loss, s', '',
      '# ---------------------------------------------------------------- training', `batch = ${T.batch}`, 'order, pos = torch.randperm(len(Xtr)), 0',
      `for step in range(${big(steps)}):`,
      '    if pos + batch > len(order):                  # a new shuffle every pass over the data', '        order, pos = torch.randperm(len(Xtr)), 0',
      '    i = order[pos:pos + batch]', '    pos += batch',
      '    loss = loss_fn(model(Xtr[i]), Ytr[i])', '    opt.zero_grad()', '    loss.backward()',
      T.clip ? `    torch.nn.utils.clip_grad_norm_(model.parameters(), ${num(T.clip)})   # cap the size of the step` : null,
      '    opt.step()', T.sched !== 'const' ? '    sched.step()' : null,
      `    if step % ${evalEvery} == 0:`, '        val_loss, val_score = evaluate(Xva, Yva)', '        print(f"step {step:5d}  training loss {loss.item():.4f}  validation loss {val_loss:.4f}  score {val_score:.3f}")', '',
      'val_loss, val_score = evaluate(Xva, Yva)', 'print(f"done: validation loss {val_loss:.4f}, score {val_score:.3f}")',
      afterCode(setup, d));
  };

  NA.builderPy = PY;
})(window.NA);
