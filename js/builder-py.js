/* builder-py.js — what you built in the model builder, written out as one PyTorch script you can run: the data
   (the same generators as the page, in Python), the model, the loss, the optimizer and its schedule, and the
   training loop with the same measures the page shows. Only torch is needed. A classic model (linear, k-NN,
   trees, a forest, boosting) is written with scikit-learn, or XGBoost for boosting, on the same data. Learning
   from rewards writes the world in Python and the method's own loop: DQN, REINFORCE, actor-critic, PPO or a
   genetic algorithm. */
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
  const SURFACE_PY = {
    hill: 'torch.exp(-((x - 0.35) ** 2 + (y - 0.3) ** 2) / 0.08) - 0.8 * torch.exp(-((x + 0.4) ** 2 + (y + 0.35) ** 2) / 0.12)',
    ripples: 'torch.cos(9 * torch.hypot(x, y)) * torch.exp(-1.5 * torch.hypot(x, y))',
    saddle: 'x * x - y * y',
    checker: 'torch.sin(4.5 * x) * torch.sin(4.5 * y)',
  };
  const SERIES_PY = {
    waves: 'torch.sin(2 * math.pi * t / 25) + 0.5 * torch.sin(2 * math.pi * t / 9.3 + 1)',
    seasonal: '0.0015 * t + 0.6 * torch.sin(2 * math.pi * t / 24) + 0.3 * torch.sin(2 * math.pi * t / 7 + 0.5)',
    square: 'torch.where(torch.sin(2 * math.pi * t / 30) >= 0, 0.8, -0.8)',
  };
  // the sequence classes of js/builder-tasks.js, in Python: each example() gives (token ids, class), or (None, -1) to skip
  const SEQCLASS_PY = {
    sentiment: (d) => String.raw`NOUNS, GOOD = ${JSON.stringify(d.vocab.slice(8, 14))}, ${JSON.stringify(d.vocab.slice(14, 19))}
BAD, ADV = ${JSON.stringify(d.vocab.slice(19, 24))}, ${JSON.stringify(d.vocab.slice(24, 27))}

def clause():
    pos, neg = random.random() < 0.5, random.random() < 0.4
    w = ["the", random.choice(NOUNS), "was" if random.random() < 0.5 else "is"]
    if neg:
        w.append("not")                           # "not" flips the clause
    if random.random() < 0.3:
        w.append(random.choice(ADV))
    w.append(random.choice(GOOD if pos else BAD))
    return w, int(pos != neg)

def example():
    """A tiny review: after "but" the second clause decides, which a bag of words cannot tell."""
    a, la = clause()
    if random.random() < 0.7:
        b, lb = clause()
        if random.random() < 0.85:
            words, label = a + ["but"] + b + ["."], lb
        else:
            words, label = a + ["and"] + b + ["."], la if la == lb else -1
    else:
        words, label = a + ["."], la
    if label < 0 or len(words) > T:
        return None, -1
    return [ID[w] for w in words] + [0] * (T - len(words)), label   # padded with "·"`,
    majority: () => String.raw`def example():
    """Bits; class 1 when more than half are 1."""
    ids = [random.randrange(2) for _ in range(T)]
    return ids, int(2 * sum(ids) > T)`,
    parity: () => String.raw`def example():
    """Bits; class 1 when the number of 1s is odd: every bit matters, which is hard to learn."""
    ids = [random.randrange(2) for _ in range(T)]
    return ids, sum(ids) % 2`,
    pattern: () => String.raw`def example():
    """Digits; half get "3 7" planted somewhere. Class 1: a 3 followed by a 7 anywhere."""
    ids = [random.randrange(10) for _ in range(T)]
    if random.random() < 0.5:
        at = random.randrange(T - 1)
        ids[at], ids[at + 1] = 3, 7
    return ids, int(any(a == 3 and b == 7 for a, b in zip(ids, ids[1:])))`,
  };
  // the shapes of js/builder-tasks.js, in Python
  const SHAPES_PY = String.raw`G = 14                                        # pictures are G x G pixels
SHAPES = ["circle", "square", "triangle", "cross", "line"]
centres = torch.stack(torch.meshgrid(torch.arange(G), torch.arange(G), indexing="ij"), -1).flip(-1).reshape(-1, 2) + 0.5

def seg_dist(p, a, b):
    """Distance from each point p to the line segment a-b."""
    ab = b - a
    t = (((p - a) * ab).sum(1) / (ab * ab).sum().clamp(min=1e-12)).clamp(0, 1)
    return (a + t[:, None] * ab - p).norm(dim=1)

def draw_shape(k):
    """One shape anywhere in the picture, at any size and angle, as a pen stroke about a pixel wide."""
    r = random.uniform(3, 5.2)
    cx, cy = random.uniform(r + 0.5, G - r - 1.5), random.uniform(r + 0.5, G - r - 1.5)
    a, w = random.uniform(0, math.pi), random.uniform(0.55, 0.95)
    pt = lambda ang, rad: torch.tensor([cx + rad * math.cos(ang), cy + rad * math.sin(ang)])
    if k == 0:
        d = ((centres - torch.tensor([cx, cy])).norm(dim=1) - r).abs()
    else:
        corners = {1: 4, 2: 3}.get(k)
        if corners:
            segs = [(pt(a + q * 2 * math.pi / corners, r), pt(a + (q + 1) * 2 * math.pi / corners, r)) for q in range(corners)]
        else:
            segs = [(pt(a, r), pt(a + math.pi, r))] + ([(pt(a + math.pi / 2, r), pt(a + 1.5 * math.pi, r))] if k == 3 else [])
        d = torch.stack([seg_dist(centres, p, q) for p, q in segs]).min(0).values
    img = (1 - (d - w / 2)).clamp(0, 1) + 0.04 * torch.randn(G * G)
    return img.view(1, G, G)`;
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
    } else if (fam === 'digits' && p.task === 'shapes') {
      out.push(SHAPES_PY, '', `n = ${s.n}`, 'X = torch.stack([draw_shape(i % 5) for i in range(n)])      # (n, 1, 14, 14), ink in [0, 1]',
        'Y = torch.arange(n) % 5', ...SPLIT(0.8));
    } else if (fam === 'digits') {
      out.push(DIGITS_PY, '', `n = ${s.n}`, 'X = torch.stack([draw_digit(i % 10) for i in range(n)])     # (n, 1, 14, 14), ink in [0, 1]',
        'Y = torch.arange(n) % 10', ...SPLIT(0.8));
    } else if (fam === 'surface') {
      out.push('def surface(x, y):', `    return ${SURFACE_PY[p.task]}`, '',
        `X = 2 * torch.rand(${s.n}, 2) - 1                     # points spread over the square [-1, 1]²`,
        `Y = surface(X[:, 0], X[:, 1]) + ${num(s.noise)} * torch.randn(${s.n})     # the height there, plus noise`, ...SPLIT(0.8));
    } else if (fam === 'series') {
      if (p.task === 'chaos') out.push(`v, x = torch.empty(${s.n}), 0.31 + 0.02 * random.random()`, 'for i in range(len(v)):                           # the logistic map: fully determined, yet chaotic', '    v[i] = 2 * x - 1', '    x = 3.9 * x * (1 - x)');
      else out.push(`t = torch.arange(${s.n}, dtype=torch.float32)`, `v = ${SERIES_PY[p.task]}`);
      out.push(`v = v + ${num(s.noise)} * torch.randn(len(v))`, 'v = (v - v.mean()) / (v - v.mean()).abs().max()     # centred, the largest value 1 (the page does too)', '',
        `W = ${s.window}                                          # the model sees the last W values`,
        'X = torch.stack([v[i:i + W] for i in range(len(v) - W)])[:, :, None]   # (windows, W, 1)', 'Y = v[W:].clone()                                  # the value after each window',
        'k = round(0.8 * len(X))                            # the first 80% train; the rest, the future, validates', 'Xtr, Ytr, Xva, Yva = X[:k], Y[:k], X[k:], Y[k:]');
    } else if (fam === 'seqclass') {
      out.push(`VOCAB = ${JSON.stringify(d.vocab)}`, 'ID = {w: i for i, w in enumerate(VOCAB)}', `CLASSES = ${JSON.stringify(d.names)}`, `T = ${d.input.len}                                         # tokens per sequence`, '',
        SEQCLASS_PY[p.task](d), '',
        `rows, seen, tries = [], set(), 0`, `while len(rows) < ${s.n} and tries < ${30 * s.n}:`, '    tries += 1', '    ids, label = example()',
        `    if ids is None${p.task === 'sentiment' ? '' : ' or tuple(ids) in seen'}:${p.task === 'sentiment' ? '' : '                # every sequence once'}`, '        continue', '    seen.add(tuple(ids))', '    rows.append((ids, label))',
        'X, Y = torch.tensor([r[0] for r in rows]), torch.tensor([r[1] for r in rows])', 'k = len(X) - max(40, round(0.2 * len(X)))', 'Xtr, Ytr, Xva, Yva = X[:k], Y[:k], X[k:], Y[k:]');
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
  const GATED_PY = String.raw`class Gated(nn.Module):
    """A gated linear unit: two weighted sums of the same input, out = value(x) * act(gate(x)) (with Swish: SwiGLU)."""
    def __init__(self, d_in, units, act):
        super().__init__()
        self.value, self.gate, self.act = nn.Linear(d_in, units), nn.Linear(d_in, units), act

    def forward(self, x):
        return self.value(x) * self.act(self.gate(x))`;
  const NOISE_PY = String.raw`class GaussianNoise(nn.Module):
    """Adds sigma * a standard normal draw to every number while training; nothing when the model is used."""
    def __init__(self, sigma):
        super().__init__()
        self.sigma = sigma

    def forward(self, x):
        return x + self.sigma * torch.randn_like(x) if self.training else x`;
  const SINE_PY = String.raw`def sine_positions(T, D):
    """The original transformer's positions: PE[t, 2i] = sin(t / 10000^(2i/D)), PE[t, 2i + 1] = cos(...)."""
    t, i = torch.arange(T)[:, None], torch.arange(D)[None, :]
    angle = t / 10000 ** (2 * (i // 2) / D)
    return torch.where(i % 2 == 0, torch.sin(angle), torch.cos(angle))`;
  // a block's settings with the engine's defaults filled in
  const cfg = (b) => Object.assign({}, (NA.bx.BLOCKS[b.type] || {}).defaults, ...Object.keys(b).filter((k) => b[k] !== undefined && b[k] !== null).map((k) => ({ [k]: b[k] })));
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

  /* One block: its layers (for __init__) and its lines in forward(), with n in its attribute names. Inside a part of
     your own, a size can be a name (the part's argument, or a size no layer depends on) instead of a number. */
  function emitBlock(b, si, so, n, helpers, par) {
    const init = [], fwd = [], last = si[si.length - 1];
    const put = (code) => fwd.push(/ # /.test(code) ? `${code}, ${shp(so)}` : code.padEnd(54) + `# ${shp(so)}`); // a line with its own comment gets the shape at its end
    const same = (p, q) => (typeof p === 'number' && typeof q === 'number' ? p === q : null);
    // a residual: the input is added back only when the shape is kept; inside a part whose copies differ, checked as it runs
    const res = (z, keep) => {
      if (keep === true) put(`x = x + ${z}`);
      else if (keep === false) put(`x = ${z}`);
      else { fwd.push(`z = ${z}`); put('x = x + z if z.shape == x.shape else z    # added back when the shape is kept'); }
    };
    if (b.type === 'dense') {
      init.push(`self.dense${n} = nn.Linear(${last}, ${b.units})`);
      let z = `self.dense${n}(x)`;
      if (b.acts) { helpers.add('mixed'); init.push(`self.act${n} = MixedActivation(${JSON.stringify(b.acts)})`); z = `self.act${n}(${z})`; }
      else z = ACT_PY[b.act](z);
      res(z, b.residual ? same(last, b.units) : false); // the engine adds the input back only when the width is kept
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
      res(ACT_PY[b.act](`self.conv${n}(x)`), b.residual && (b.stride || 1) === 1 ? same(si[0], b.filters) : false); // the engine adds the input back only when the shape is kept
    } else if (b.type === 'pool') put(`x = F.max_pool2d(x, ${b.size})                     # leftover rows and columns are dropped`);
    else if (b.type === 'flatten') put('x = x.flatten(1)');
    else if (b.type === 'gap') put(si.length === 3 ? 'x = x.mean((2, 3))' : 'x = x.mean(1)');
    else {
      const c = cfg(b);
      if (b.type === 'gated') { helpers.add('gated'); init.push(`self.glu${n} = Gated(${last}, ${c.units}, ${ACT_FN[c.gate] || 'F.silu'})`); put(`x = self.glu${n}(x)`); }
      else if (b.type === 'batchnorm') { init.push(`self.bn${n} = nn.BatchNorm${si.length === 3 ? '2d' : '1d'}(${si.length === 3 ? si[0] : last})`); put(si.length === 2 ? `x = self.bn${n}(x.transpose(1, 2)).transpose(1, 2)` : `x = self.bn${n}(x)`); }
      else if (b.type === 'noise') { helpers.add('noise'); init.push(`self.noise${n} = GaussianNoise(${num(c.sigma)})`); put(`x = self.noise${n}(x)`); }
      else if (b.type === 'fourier') {
        init.push(`self.register_buffer("B${n}", ${num(c.scale)} * torch.randn(${c.features}, ${last}))   # random frequencies, never trained`);
        fwd.push(`a = 2 * math.pi * x @ self.B${n}.T`); put('x = torch.cat([x, torch.sin(a), torch.cos(a)], -1)');
      } else if (b.type === 'rnn') { init.push(`self.rnn${n} = nn.${{ lstm: 'LSTM', gru: 'GRU', rnn: 'RNN' }[c.cell] || 'LSTM'}(${last}, ${c.units}, batch_first=True)`); put(`x, _ = self.rnn${n}(x)`); }
      else if (b.type === 'conv1d') {
        const reach = (c.k - 1) * (c.dilation || 1), left = c.causal ? reach : Math.floor(reach / 2);
        init.push(`self.conv${n} = nn.Conv1d(${last}, ${c.filters}, ${c.k}${c.dilation > 1 ? `, dilation=${c.dilation}` : ''})`);
        res(`${ACT_PY[c.act](`self.conv${n}(F.pad(x.transpose(1, 2), (${left}, ${reach - left})))`)}.transpose(1, 2)`, c.residual ? same(last, c.filters) : false);
      } else if (b.type === 'position') {
        if (c.mode === 'sine') { helpers.add('sine'); init.push(`self.register_buffer("pe${n}", sine_positions(${si[0]}, ${last}))   # fixed waves, one row per position`); put(`x = x + self.pe${n}`); }
        else { init.push(`self.pos${n} = nn.Parameter(0.02 * torch.randn(${si[0]}, ${last}))   # a learned vector for each position`); put(`x = x + self.pos${n}`); }
      } else if (b.type === 'seqpool') put(c.mode === 'max' ? 'x = x.max(1).values' : c.mode === 'mean' ? 'x = x.mean(1)' : 'x = x[:, -1]                                  # the last position');
      else if (b.type === 'patches') {
        init.push(`self.patch${n} = nn.Conv2d(${si[0]}, ${c.dim}, ${c.size}, stride=${c.size})   # one projection for every ${c.size}×${c.size} patch`, `self.pos${n} = nn.Parameter(0.02 * torch.randn(${so[0]}, ${c.dim}))   # each patch's learned place`);
        put(`x = self.patch${n}(x).flatten(2).transpose(1, 2) + self.pos${n}`);
      } else if (b.type === 'avgpool') put(`x = F.avg_pool2d(x, ${c.size})`);
      else if (b.type === 'parallel' && par) {
        const name = `${par.name}${n}`;
        parallelClasses(b, si, so, helpers, name, par.sink);
        init.push(`self.par${n} = ${name}()`);
        put(`x = self.par${n}(x)`);
      }
    }
    return { init, fwd };
  }


  // ------------------------------------------------------------------ parallel branches: a class per branch, and one that runs them all
  const MERGE_PY = { concat: (dim) => `torch.cat(outs, ${dim})`, add: () => 'sum(outs)', mul: () => 'torch.stack(outs).prod(0)', mean: () => 'torch.stack(outs).mean(0)' };
  const MERGE_SAY = { concat: 'joined side by side', add: 'added number by number', mul: 'multiplied number by number', mean: 'averaged number by number' };
  function parallelClasses(b, si, so, helpers, name, sink) {
    const merge = cfg(b).merge || 'concat', label = NA.builderLab && NA.builderLab.blockLabel ? NA.builderLab.blockLabel : (q) => q.type, made = [];
    b.branches.forEach((list, br) => {
      if (!list.length) { made.push('nn.Identity()'); return; } // an empty branch hands on its input
      let sh = si;
      const init = [], fwd = [];
      list.forEach((q, k) => { const o2 = symShape(q, sh), e = emitBlock(q, sh, o2, k + 1, helpers, null); init.push(...e.init); fwd.push(...e.fwd); sh = o2; });
      const cname = `${name}${'ABCD'[br]}`;
      sink.push(lines(`class ${cname}(nn.Module):`, `    """Branch ${'ABCD'[br]} of ${name}: ${list.map(label).join(' → ')}."""`, '    def __init__(self):', '        super().__init__()', init.map((l) => '        ' + l), '',
        '    def forward(self, x):'.padEnd(50) + `# x: ${shp(si)}`, fwd.map((l) => '        ' + l), '        return x'));
      made.push(`${cname}()`);
    });
    sink.push(lines(`class ${name}(nn.Module):`, `    """${b.branches.length} branches side by side on the same input, ${MERGE_SAY[merge]}."""`, '    def __init__(self):', '        super().__init__()', `        self.branches = nn.ModuleList([${made.join(', ')}])`, '',
      '    def forward(self, x):'.padEnd(50) + `# x: ${shp(si)}`, '        outs = [branch(x) for branch in self.branches]   # every branch reads the same x', `        return ${MERGE_PY[merge](so.length === 3 ? 1 : -1)}`.padEnd(58) + `# ${shp(so)}`));
  }
  // ------------------------------------------------------------------ parts of your own: a class each
  /* Blocks you boxed into a part become a class, and every copy an instance of it with its own weights. A size that
     differs between the copies (a first copy that reads 2 numbers, a second that reads 16) becomes an argument of the
     class, so it is used like a function: Bend(d_in=2), Bend(d_in=16). Names come from the part's name. */
  const PY_TAKEN = new Set(['Model', 'SelfAttention', 'TransformerBlock', 'Gated', 'GaussianNoise', 'MixedActivation', 'None', 'True', 'False']);
  const STEMS = new Set(['dense', 'act', 'drop', 'norm', 'attn', 'block', 'conv', 'glu', 'bn', 'noise', 'rnn', 'pe', 'pos', 'patch', 'b', 'tok', 'out']);
  const wordsOf = (s) => String(s).normalize('NFD').replace(/[̀-ͯ]/g, '').split(/[^A-Za-z0-9]+/).filter(Boolean);
  const specOf = (b) => { const c = Object.assign({}, b); delete c.id; delete c.grp; delete c.part; return c; };
  function partNames(setup) {
    const runs = BT.groupsOf ? BT.groupsOf(setup.blocks) : [], cls = {}, taken = new Set(), snakes = new Set(), inst = {}, count = {};
    for (const def of setup.parts || []) {
      if (!runs.some((r) => r.part === def.id)) continue;
      const w = wordsOf(def.name);
      let c = w.map((x) => x[0].toUpperCase() + x.slice(1)).join('') || 'Part';
      if (/^\d/.test(c)) c = 'Part' + c;
      if (c.length < 3 || c === c.toUpperCase() || PY_TAKEN.has(c)) c += 'Part'; // never F, X, None…: names the script uses
      let u = c, k = 2;
      while (taken.has(u)) u = c + k++;
      taken.add(u);
      let sn = w.map((x) => x.toLowerCase()).join('_') || 'part';
      if (/^\d/.test(sn)) sn = 'part_' + sn;
      if (STEMS.has(sn)) sn += '_part';
      let su = sn; k = 2;
      while (snakes.has(su)) su = `${sn}_${k++}`;
      snakes.add(su);
      cls[def.id] = { name: u, snake: su, def };
    }
    const where = setup.blocks.map((_, i) => ({ pre: 'model.', n: i + 1 }));
    for (const r of runs) {
      const c = cls[r.part];
      if (!c) continue;
      const k = (count[r.part] = (count[r.part] || 0) + 1);
      inst[r.grp] = `${c.snake}${/\d$/.test(c.snake) ? '_' : ''}${k}`;
      for (let i = r.from; i < r.to; i++) where[i] = { pre: `model.${inst[r.grp]}.`, n: i - r.from + 1 };
    }
    return { runs: runs.filter((r) => cls[r.part]), cls, inst, where };
  }
  // sizes that may be names: computed when they are numbers, written as Python when they are not
  const wrap = (v) => (typeof v === 'string' && / /.test(v) ? `(${v})` : v);
  const SX = {
    add: (p, q) => (typeof p === 'number' && typeof q === 'number' ? p + q : `${p} + ${q}`),
    mul: (...v) => (v.every((q) => typeof q === 'number') ? v.reduce((p, q) => p * q, 1) : v.map(wrap).join(' * ')),
    div: (p, q) => (typeof p === 'number' ? Math.floor(p / q) : q === 1 ? p : `${wrap(p)} // ${q}`),
    ceil: (p, q) => (typeof p === 'number' ? Math.ceil(p / q) : q === 1 ? p : `-(-${wrap(p)} // ${q})`),
  };
  // the engine's shape rules (builder-engine.js, blockShape), on sizes that may be names
  function symShape(b, s) {
    const c = cfg(b), k = s.length;
    switch (b.type) {
      case 'dense': case 'gated': return k === 1 ? [c.units] : [s[0], c.units];
      case 'conv': return [c.filters, SX.ceil(s[1], c.stride || 1), SX.ceil(s[2], c.stride || 1)];
      case 'pool': case 'avgpool': return [s[0], SX.div(s[1], c.size), SX.div(s[2], c.size)];
      case 'flatten': return [SX.mul(...s)];
      case 'gap': return k === 3 ? [s[0]] : [s[1]];
      case 'seqpool': return [s[1]];
      case 'fourier': return [SX.add(s[0], 2 * c.features)];
      case 'rnn': return [s[0], c.units];
      case 'conv1d': return [s[0], c.filters];
      case 'patches': return [SX.mul(SX.div(s[1], c.size), SX.div(s[2], c.size)), c.dim];
      case 'parallel': {
        const outs = (b.branches || []).map((list) => list.reduce((sh, q) => symShape(q, sh), s));
        if ((c.merge || 'concat') !== 'concat' || !outs.length) return (outs[0] || s).slice();
        const sum = (j) => outs.map((o) => o[j]).reduce((a, q) => SX.add(a, q));
        return k === 1 ? [sum(0)] : k === 2 ? [outs[0][0], sum(1)] : [sum(0), outs[0][1], outs[0][2]];
      }
      default: return s.slice();
    }
  }
  const ARGS = [null, ['d_in'], ['t_in', 'd_in'], ['c_in', 'h_in', 'w_in']], FREE = [null, ['D'], ['T', 'D'], ['C', 'H', 'W']];
  const ARG_TEXT = { d_in: 'the numbers it is given', t_in: 'the positions of the sequence it is given', c_in: 'the channels of the picture it is given', h_in: 'its height', w_in: 'its width' };
  const HELPER_NAME = { mixed: 'MixedActivation', attn: 'SelfAttention', block: 'TransformerBlock', gated: 'Gated', noise: 'GaussianNoise', sine: 'sine_positions' };
  // every part's class (or classes, when copies differ in kind of input), and how each copy is made and used
  function planParts(setup, chk, names) {
    const byGrp = {}, classes = [], label = NA.builderLab && NA.builderLab.blockLabel ? NA.builderLab.blockLabel : (b) => b.type;
    for (const pid of Object.keys(names.cls)) {
      const pc = names.cls[pid], groups = [];
      for (const r of names.runs.filter((q) => q.part === pid)) {
        const inS = (chk.shapes[r.from] || {}).in || [], specs = JSON.stringify(setup.blocks.slice(r.from, r.to).map(specOf));
        const exact = /"type":"parallel"/.test(specs) ? JSON.stringify(inS) : '';
        let g = groups.find((q) => q.rank === inS.length && q.specs === specs && q.exact === exact);
        if (!g) groups.push((g = { rank: inS.length, specs, exact, runs: [], ins: [] }));
        g.runs.push(r); g.ins.push(inS);
      }
      groups.forEach((g, gi) => {
        const R = g.rank, blocks = setup.blocks.slice(g.runs[0].from, g.runs[0].to), name = gi ? `${pc.name}_v${gi + 1}` : pc.name, helpers = new Set();
        const emit = (inShape) => {
          let sh = inShape;
          const init = [], fwd = [], sink = [];
          blocks.forEach((b, k) => { const so = symShape(b, sh), e = emitBlock(b, sh, so, k + 1, helpers, { name: `${name}Par`, sink }); init.push(...e.init); fwd.push(...e.fwd); sh = so; });
          return { init, fwd, out: sh, sink };
        };
        // a size the copies share is a number; one they do not is a name: an argument when a layer needs it
        const sym = g.ins[0].map((v, j) => (g.ins.every((q) => q[j] === v) ? v : (ARGS[R] || [])[j] || `s${j}`));
        const first = emit(sym), used = (nm) => first.init.some((l) => new RegExp(`\\b${nm}\\b`).test(l));
        const args = sym.filter((v) => typeof v === 'string' && used(v));
        const shown = sym.map((v, j) => (typeof v === 'string' && !args.includes(v) ? (FREE[R] || [])[j] || v : v)), e = emit(shown);
        const text = lines(
          `class ${name}(nn.Module):`,
          `    """Your part ${pc.def.name}: ${blocks.map(label).join(' → ')}. Each copy has its own weights."""`,
          `    def __init__(self${args.map((q) => ', ' + q).join('')}):`.padEnd(args.length ? 50 : 0) + (args.length ? `# ${args.map((q) => `${q}: ${ARG_TEXT[q] || 'a size'}`).join('; ')}` : ''),
          '        super().__init__()', e.init.map((l) => '        ' + l), '',
          '    def forward(self, x):'.padEnd(50) + `# x: ${shp(shown)}`, e.fwd.map((l) => '        ' + l), '        return x');
        classes.push({ name, text, helpers, before: e.sink });
        g.runs.forEach((r, ci) => {
          const vals = args.map((q) => `${q}=${g.ins[ci][sym.indexOf(q)]}`).join(', ');
          byGrp[r.grp] = { cls: name, text, helpers, before: e.sink, inst: names.inst[r.grp], make: `self.${names.inst[r.grp]} = ${name}(${vals})`, out: (chk.shapes[r.to - 1] || {}).out };
        });
      });
    }
    return { byGrp, classes };
  }

  function modelCode(setup, d, chk) {
    const blocks = setup.blocks, inp = d.input, out = d.output, init = [], fwd = [], helpers = new Set();
    if (inp.kind === 'tokens') {
      init.push(`self.tok = nn.Embedding(${inp.vocab}, ${inp.embed})           # a learned vector for each token`, `self.pos = nn.Embedding(${inp.len}, ${inp.embed})           # and one for each position`);
      fwd.push(`x = self.tok(x) + self.pos(torch.arange(x.shape[1]))`.padEnd(54) + `# ${shp([inp.len, inp.embed])}`);
    }
    const names = partNames(setup), parts = planParts(setup, chk, names), parSink = [];
    for (const c of parts.classes) for (const h of c.helpers) helpers.add(h);
    for (let i = 0; i < blocks.length;) {
      const run = names.runs.find((r) => r.from === i);
      if (run) {
        // a copy of one of your parts: one instance of its class
        const g = parts.byGrp[run.grp];
        init.push(g.make);
        fwd.push(`x = self.${g.inst}(x)`.padEnd(54) + `# ${shp(g.out)}`);
        i = run.to;
        continue;
      }
      const sh = chk.shapes[i] || {}, e = emitBlock(blocks[i], sh.in || [], sh.out || [], i + 1, helpers, { name: 'Parallel', sink: parSink });
      init.push(...e.init); fwd.push(...e.fwd);
      i++;
    }
    const fin = chk.shapes.length ? (chk.shapes[chk.shapes.length - 1] || {}).out : inp.kind === 'tokens' ? [inp.len, inp.embed] : inp.kind === 'vector' ? [inp.dim] : inp.kind === 'series' ? [inp.len, inp.dim] : [inp.c, inp.h, inp.w];
    const dIn = fin ? fin[fin.length - 1] : 1, K = out.kind === 'tokens' ? out.vocab : out.k;
    init.push(`self.out = nn.Linear(${dIn}, ${K})${out.kind === 'classes' ? '                # one score per class' : out.kind === 'tokens' ? '                # a score for every possible next token' : ''}`);
    fwd.push(`return self.out(x)`.padEnd(54) + `# ${shp(out.kind === 'tokens' ? [inp.len, K] : [K])}`);
    const pre = [];
    if (helpers.has('mixed')) pre.push(`ACT = {${Object.keys(ACT_FN).map((a) => `"${a}": ${ACT_FN[a]}`).join(', ')}}`, '', MIXED_PY, '', '');
    if (helpers.has('attn')) pre.push(ATTN_PY, '', '');
    if (helpers.has('block')) pre.push(BLOCK_PY, '', '');
    if (helpers.has('gated')) pre.push(GATED_PY, '', '');
    if (helpers.has('noise')) pre.push(NOISE_PY, '', '');
    if (helpers.has('sine')) pre.push(SINE_PY, '', '');
    for (const t of parSink) pre.push(t, '', '');
    for (const c of parts.classes) pre.push(...c.before.flatMap((t) => [t, '', '']), c.text, '', '');
    return lines('# ---------------------------------------------------------------- the model', ...pre,
      'class Model(nn.Module):', '    def __init__(self):', '        super().__init__()', init.map((l) => '        ' + l), '',
      '    def forward(self, x):' + `                                   # x: ${shp(inp.kind === 'tokens' ? [inp.len] : inp.kind === 'vector' ? [inp.dim] : inp.kind === 'series' ? [inp.len, inp.dim] : [inp.c, inp.h, inp.w])}`, fwd.map((l) => '        ' + l));
  }
  // where a block's layers live in the model: model.dense3, or model.bend2.dense1 inside a copy of one of your parts
  PY.attrPath = function (setup, i, stem) { const w = partNames(setup).where[i] || { pre: 'model.', n: i + 1 }; return `${w.pre}${stem}${w.n}`; };
  // one part's class, for the architecture view
  PY.partClass = function (setup, d, run) {
    if (!d || d.error || !NA.bx) return '';
    const chk = NA.bx.check(setup.blocks, d.input, d.output);
    if (!chk.ok) return '# This part does not fit here yet, so there is nothing to write out.';
    const names = partNames(setup), g = planParts(setup, chk, names).byGrp[run.grp];
    if (!g) return '';
    const uses = [...g.helpers].map((h) => HELPER_NAME[h]).filter(Boolean);
    return lines(uses.length ? `# uses ${uses.join(', ')}: see the whole script (the Python tab)` : null, ...g.before.flatMap((t) => [t, '']), g.text, '', `# in the model:         ${g.make}`, `# and in its forward(): x = self.${g.inst}(x)`);
  };

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
    const where = partNames(setup).where;
    const initBlock = (b, at) => {
      const attn = (m) => `for layer in (${['q', 'k', 'v', 'o'].map((x) => `${m}.${x}`).join(', ')}):`;
      if (b.type === 'parallel') (b.branches || []).forEach((list, br) => list.forEach((q, k) => initBlock(q, (stem) => `${at('par')}.branches[${br}].${stem}${k + 1}`)));
      if (b.type === 'dense') out.push(call(at('dense'), b.acts && !small ? b.acts.map((a) => gain(HE(a))) : gain(HE(b.act))));
      else if (b.type === 'conv') out.push(call(at('conv'), gain(HE(b.act))));
      else if (b.type === 'conv1d') out.push(call(at('conv'), gain(HE(cfg(b).act))));
      else if (b.type === 'gated') out.push(call(`${at('glu')}.value`, 1), call(`${at('glu')}.gate`, gain(HE(cfg(b).gate))));
      else if (b.type === 'patches') out.push(call(at('patch'), 1));
      else if (b.type === 'rnn') {
        // as on the page: normal weights with sd √(1 / inputs) (0.02 for tiny), zero biases, an LSTM's forget gates open
        out.push('with torch.no_grad():', `    r = ${at('rnn')}`, small ? '    r.weight_ih_l0.normal_(0.0, 0.02), r.weight_hh_l0.normal_(0.0, 0.02)' : '    r.weight_ih_l0.normal_(0.0, math.sqrt(1 / r.input_size)), r.weight_hh_l0.normal_(0.0, math.sqrt(1 / r.hidden_size))',
          '    r.bias_ih_l0.zero_(), r.bias_hh_l0.zero_()', ...(cfg(b).cell === 'lstm' ? ['    r.bias_ih_l0[r.hidden_size:2 * r.hidden_size] = 1.0     # forget gates start open: the memory lasts from the start'] : []));
      }
      else if (b.type === 'attention') out.push(attn(at('attn')), '    init_(layer)');
      else if (b.type === 'transformer') out.push(attn(`${at('block')}.attn`), '    init_(layer)', call(`${at('block')}.mlp[0]`, gain(true)), call(`${at('block')}.mlp[2]`, mode === 'he' ? 2 : 1));
    };
    setup.blocks.forEach((b, i) => initBlock(b, (stem) => `${where[i].pre}${stem}${where[i].n}`));
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
    const names = partNames(setup), items = [];
    for (let i = 0; i < setup.blocks.length;) {
      const r = names.runs.find((q) => q.from === i);
      if (r) { items.push(names.cls[r.part].def.name); i = r.to; } else { items.push(L(setup.blocks[i])); i++; }
    }
    return [...(d.input.kind === 'tokens' ? ['embedding'] : []), ...items, end].join(' → ');
  }
  function partsSummary(setup) {
    const L = NA.builderLab && NA.builderLab.blockLabel ? NA.builderLab.blockLabel : (b) => b.type, names = partNames(setup);
    return Object.values(names.cls).map((c) => {
      const r = names.runs.find((q) => q.part === c.def.id), n = names.runs.filter((q) => q.part === c.def.id).length;
      return `#   ${c.def.name} (${c.name}, ${n} cop${n === 1 ? 'y' : 'ies'}): ${setup.blocks.slice(r.from, r.to).map(L).join(' → ')}`;
    });
  }

  // ------------------------------------------------------------------ classic models: scikit-learn, and XGBoost for boosting
  function classicModel(b, d, seed) {
    const s = NA.cml.withDefaults(b), cls = d.output.kind === 'classes', n = d.nTr, CR = cls ? 'Classifier' : 'Regressor';
    const sk = (imp, make) => ({ imp, fit: [`model = ${make}`, 'model.fit(Xtr_, ytr_)'], predict: 'predict = model.predict', proba: 'model.predict_proba(x)' });
    if (b.type === 'linear') {
      if (cls) return sk('from sklearn.linear_model import LogisticRegression', s.lambda ? `LogisticRegression(C=${num(1 / (s.lambda * n))}, max_iter=2000)   # C = 1 / (λ × examples): the page's λ = ${num(s.lambda)}` : 'LogisticRegression(C=float("inf"), max_iter=2000)   # C = ∞: no penalty, as with the page\'s λ = 0');
      return s.lambda ? sk('from sklearn.linear_model import Ridge', `Ridge(alpha=${num(s.lambda)})                         # least squares plus λ|w|²`) : sk('from sklearn.linear_model import LinearRegression', 'LinearRegression()                                 # least squares, solved exactly');
    }
    if (b.type === 'knn') return sk(`from sklearn.neighbors import KNeighbors${CR}`, `KNeighbors${CR}(n_neighbors=${s.k}, weights="${s.weights === 'distance' ? 'distance' : 'uniform'}")`);
    if (b.type === 'tree') return sk(`from sklearn.tree import DecisionTree${CR}`, `DecisionTree${CR}(max_depth=${s.depth}, min_samples_leaf=${s.minLeaf}, random_state=${seed})`);
    if (b.type === 'forest') return sk(`from sklearn.ensemble import RandomForest${CR}`, `RandomForest${CR}(n_estimators=${s.trees}, max_depth=${s.depth}, max_features=${s.features === 'all' ? 'None' : s.features === 'third' ? '1 / 3' : '"sqrt"'}, min_samples_leaf=${s.minLeaf}, random_state=${seed})`);
    // boosting with XGBoost's own interface: its settings carry the same names as on the page (eta, lambda, gamma)
    const K = d.output.k, obj = !cls ? 'reg:squarederror' : K === 2 ? 'binary:logistic' : 'multi:softprob';
    return {
      imp: 'import xgboost as xgb',
      fit: ['params = {', `    "objective": "${obj}",${cls && K > 2 ? ` "num_class": ${K},` : ''}`,
        `    "eta": ${num(s.eta)}, "max_depth": ${s.depth}, "lambda": ${num(s.lambda)}, "gamma": ${num(s.gamma)},`,
        `    "subsample": ${num(s.subsample)}, "colsample_bytree": ${num(s.colsample)}, "tree_method": "hist", "max_bin": 48, "seed": ${seed},`, '}',
        `model = xgb.train(params, xgb.DMatrix(Xtr_, label=ytr_), num_boost_round=${s.rounds})   # one small tree per round${cls && K > 2 ? ' and class' : ''}`],
      predict: !cls ? 'predict = lambda X_: model.predict(xgb.DMatrix(X_))' : K === 2 ? 'predict = lambda X_: (model.predict(xgb.DMatrix(X_)) > 0.5).astype(int)   # p(class 1) over a half' : 'predict = lambda X_: model.predict(xgb.DMatrix(X_)).argmax(1)',
      proba: 'model.predict(xgb.DMatrix(x))',
    };
  }
  // what to print to look inside each kind of model, for the first validation example x
  function classicLook(b, d) {
    const s = NA.cml.withDefaults(b), cls = d.output.kind === 'classes';
    if (b.type === 'linear') return ['print("weights:", model.coef_.round(3), "bias:", model.intercept_.round(3))'];
    if (b.type === 'knn') return ['dist, idx = model.kneighbors(x)                    # its nearest training examples', `print("the ${s.k} nearest:", idx[0], "at distances", dist[0].round(3), "with targets", ytr_[idx[0]])`];
    if (b.type === 'tree') return ['from sklearn.tree import export_text', 'print(export_text(model, max_depth=3))             # the first questions, as text', 'print("nodes on this example\'s path:", model.decision_path(x).indices)'];
    if (b.type === 'forest') return [`print("the first trees' answers:", [${cls ? 'int(t.predict(x)[0])' : 'round(float(t.predict(x)[0]), 3)'} for t in model.estimators_[:8]])`];
    return ['print(model.get_dump()[0])                         # the first tree, as text', 'print("score:", model.predict(xgb.DMatrix(x), output_margin=True))   # before the sigmoid or softmax'];
  }
  PY.classicSnippet = function (b, d) {
    if (!d || d.error || !d.output) return '';
    const M = classicModel(b, d, 1), cls = d.output.kind === 'classes';
    return lines(M.imp, 'Xtr_, ytr_ = Xtr.flatten(1).numpy(), Ytr.numpy()     # classic models read flat rows of numbers', M.fit, M.predict,
      'x = Xva[:1].flatten(1).numpy()                     # the first validation example, as here', cls ? `print(${M.proba})                     # a probability for each class` : 'print(predict(x))                                  # the answer', classicLook(b, d));
  };
  function classicCode(setup, d) {
    const p = BT.byId[setup.problem], b = setup.blocks.find((q) => NA.cml.isClassic(q.type)), T = setup.train, M = classicModel(b, d, T.seed), cls = d.output.kind === 'classes';
    const lib = b.type === 'boost' ? 'xgboost' : 'scikit-learn', rnd = needsRandom(p, d);
    return lines(
      `# Built in Neural Anatomy's model builder: ${p.label}.`,
      `# Model:     ${NA.cml.KINDS[b.type].label}, with ${b.type === 'boost' ? 'XGBoost' : 'scikit-learn'}`,
      `# Needs torch (for the data, made as on the page) and ${lib}: pip install ${lib}`,
      'import math', rnd ? 'import random' : null, d.table && setup.custom.table && setup.custom.table.rows.length * (d.feats ? d.feats.length + 1 : 1) > 2400 ? 'import csv' : null, p.task === 'sentences' || p.task === 'language' ? 'import re' : null,
      'import torch', M.imp, '',
      `torch.manual_seed(${T.seed})`, rnd ? `random.seed(${T.seed})` : null, '',
      dataCode(setup, d), '', '',
      '# ---------------------------------------------------------------- the model',
      '# classic models read flat rows of numbers: a picture, or a window of a series, becomes one long row',
      'Xtr_, Xva_ = Xtr.flatten(1).numpy(), Xva.flatten(1).numpy()', 'ytr_, yva_ = Ytr.numpy(), Yva.numpy()',
      M.fit, M.predict, '',
      '# ---------------------------------------------------------------- how good it is',
      cls ? ['for name, X_, y_ in (("training", Xtr_, ytr_), ("validation", Xva_, yva_)):', '    print(f"{name} accuracy {(predict(X_) == y_).mean():.3f}")']
        : ['def r2(pred, y):                                  # R²: 1 is perfect, 0 is no better than the mean', '    return 1 - ((pred - y) ** 2).sum() / ((y - y.mean()) ** 2).sum()', '', 'for name, X_, y_ in (("training", Xtr_, ytr_), ("validation", Xva_, yva_)):', '    print(f"{name} R² {r2(predict(X_), y_):.3f}")'],
      '', '# ---------------------------------------------------------------- inside it, for the first validation example', 'x = Xva_[:1]',
      cls ? `print("probabilities:", ${M.proba}.round(3), "true class:", yva_[0])` : 'print("answer:", predict(x), "target:", yva_[0])', classicLook(b, d));
  }
  // ------------------------------------------------------------------ learning from rewards
  // seeded randomness for a score, so every measure plays the same starts (and training carries on where it was)
  const SEEDED = (seed, body) => ['state = random.getstate()', `random.seed(${seed})`, ...body, 'random.setstate(state)'];
  const PLAY = (pre) => [`${pre}o, done = self.reset(), False`, `${pre}while not done:`, `${pre}    o, _, done = self.step(policy(o))`];
  /* The world in Python, as the builder plays it: its physics, the inputs it gives (KEEP: those the model is
     given, NOISE: sensor noise) and what it pays, from the reward table or your own formula (written as Python on
     plain numbers by the formula language). With the defaults it is the RL lab's world, line for line. */
  function worldPy(d) {
    const w = d.world, ws = d.ws, E = NA.rlEnvs;
    const num = (v) => { const s = String(+(+v).toPrecision(12)); return /^-?\d+$/.test(s) ? s + '.0' : s; }, say = (v) => (v > 0 ? '+' : '') + String(+(+v).toPrecision(12));
    const rw = Object.fromEntries(E.REWARDS[w].map((r) => [r.key, typeof ws[r.key] === 'number' ? ws[r.key] : r.def]));
    const own = ws.rFormula ? NA.builderTasks.compile(ws.rFormula, E.REWARD_VARS[w].map((v) => v[0]), { scalar: true }) : null, f = own && !own.error ? own : null;
    // the inputs it keeps and the noise on them: obs() becomes raw_obs(), and obs() what the model is given
    const keep = d.keep, noise = ws.noise || 0;
    const consts = [...(keep ? [`    KEEP = [${keep.join(', ')}]${' '.repeat(Math.max(1, 34 - keep.join(', ').length))}# the inputs it is given (not: ${(d.hidden || []).join(', ')})`] : []),
      ...(noise ? [`    NOISE = ${num(noise)}                                # sensor noise: a random amount on every input, every step`] : [])];
    const sense = (lines) => (!keep && !noise ? lines : [lines[0].replace('def obs(self):', 'def raw_obs(self):'), ...lines.slice(1), '',
      '    def obs(self):                                # what the model is given', `        v = self.raw_obs()${keep ? '[self.KEEP]' : ''}`, noise ? '        return v + self.NOISE * torch.randn_like(v)' : '        return v']);
    const head = (lines) => [...lines, ...consts];
    const ownLine = f ? [`        r = ${f.py}`] : null;
    const pays = (parts) => (f ? `Its reward is your own: r = ${f.py}` : parts);
    // the race track, the drone arena and the rocket: builder-py-worlds.js
    const PW = NA.builderPyWorlds;
    if (PW && PW.worlds[w]) return PW.worlds[w](d, { num, say, rw, f, sense, head, ownLine, pays, SEEDED, PLAY });
    if (w === 'levers') {
      const k = ws.levers || 3, p = typeof ws.p === 'number' ? ws.p : 0.8, q = typeof ws.q === 'number' ? ws.q : 0.2, pct = (v) => `${Math.round(v * 100)}%`;
      return [...head(['class World:', `    """A light shows one of ${k} colours. The lever of that colour pays ${pct(p)} of the time, any other ${pct(q)}. One`, `    pull per episode. ${pays(`A lever that pays gives ${say(rw.rPay)}.`)}"""`,
        `    actions = [${d.actions.map((a) => JSON.stringify(a)).join(', ')}]`]), '',
        '    def reset(self):', `        self.light = random.randrange(${k})`, '        return self.obs()', '',
        ...sense(['    def obs(self):                                # one input per colour: 1 for the light that is on', `        return F.one_hot(torch.tensor(self.light), ${k}).float()`]), '',
        '    def step(self, a):                            # -> (observation, reward, done)', `        paid = random.random() < (${num(p)} if a == self.light else ${num(q)})`,
        ...(f ? ['        paid, right = float(paid), float(a == self.light)', ...ownLine, '        return self.obs(), r, True'] : [`        return self.obs(), ${rw.rPay === 1 ? 'float(paid)' : `${num(rw.rPay)} if paid else 0.0`}, True`]), '',
        '    def score(self, policy):', '        """The share of lights for which its best lever is the right one."""', '        right = 0', `        for light in range(${k}):`, '            self.light = light', '            right += policy(self.obs()) == light', `        return right / ${k}`];
    }
    if (w === 'pole') {
      const rd = ws.view === 'readings', F = ws.force > 0 ? ws.force : 10, len = ws.length > 0 ? ws.length : 1, g = ws.gravity > 0 ? ws.gravity : 9.8, steps = ws.steps > 0 ? ws.steps : 200, half = len / 2, pml = 0.1 * half;
      const reward = f ? ['        x, v, angle, spin, t = self.x, self.v, math.degrees(self.th), math.degrees(self.w), self.t', '        push, fell = (1.0 if a == 1 else -1.0), float(fell)', ...ownLine]
        : [`        r = ${num(rw.rUp)}${rw.rFell ? ` + (${num(rw.rFell)} if fell else 0.0)` : ''}`];
      return [...head(['class World:', '    """A pole hinged on a cart that can be pushed left or right (Barto, Sutton and Anderson, 1983). It falls when it',
        `    leans more than 12 degrees or the cart runs off its track. ${pays(`${say(rw.rUp)} for every step${rw.rFell ? `, ${say(rw.rFell)} when it falls` : ' it stays up'}`)}; at most ${steps} steps."""`,
        '    actions = ["push left", "push right"]', `    max_steps = ${steps}`]), '',
        '    def reset(self):', '        self.x, self.v, self.th, self.w = (random.uniform(-0.05, 0.05) for _ in range(4))', '        self.t = 0',
        rd ? '        self.hist = [(self.x / 2.4, self.th / 0.2095)] * 3     # the readings before this one' : null, '        return self.obs()', '',
        ...sense([rd ? '    def obs(self):                                # (4, 2): position and angle, the last 4 steps, oldest first; no speeds' : '    def obs(self):                                # position, speed, angle, turning speed, scaled to about ±1',
          rd ? '        return torch.tensor(self.hist + [(self.x / 2.4, self.th / 0.2095)])' : '        return torch.tensor([self.x / 2.4, self.v / 3, self.th / 0.2095, self.w / 3.5])']), '',
        `    def step(self, a):                            # one 0.02 s step: gravity ${num(g).replace(/\.0$/, '')}, cart 1 kg, pole 0.1 kg and ${num(len).replace(/\.0$/, '')} m, a push of ${num(F).replace(/\.0$/, '')} N`,
        `        force = ${num(F)} if a == 1 else -${num(F)}`, '        ct, st = math.cos(self.th), math.sin(self.th)', `        tmp = (force + ${num(pml)} * self.w * self.w * st) / 1.1`,
        `        acc = (${num(g)} * st - ct * tmp) / (${num(half)} * (4 / 3 - 0.1 * ct * ct / 1.1))`, `        xacc = tmp - ${num(pml)} * acc * ct / 1.1`,
        rd ? '        self.hist = self.hist[1:] + [(self.x / 2.4, self.th / 0.2095)]' : null,
        '        self.x, self.v, self.th, self.w = self.x + 0.02 * self.v, self.v + 0.02 * xacc, self.th + 0.02 * self.w, self.w + 0.02 * acc', '        self.t += 1',
        '        fell = abs(self.x) > 2.4 or abs(self.th) > 0.2095', ...reward, '        return self.obs(), r, bool(fell) or self.t >= self.max_steps', '',
        '    def score(self, policy):', `        """How long its best moves keep the pole up, as a share of ${steps} steps, over 5 starts."""`, '        total = 0',
        ...SEEDED(4242, ['for _ in range(5):', ...PLAY('    '), '    total += self.t']).map((l) => '        ' + l), '        return total / (5 * self.max_steps)'].filter((l) => l !== null);
    }
    if (w === 'catch') {
      const px = ws.view === 'pixels', drift = ws.drift !== false;
      const reward = f ? ['        caught, missed, moved = float(end and self.caught), float(end and not self.caught), float(moved)', '        dx, down, t = self.bx - self.px, self.by, self.t', ...ownLine]
        : [`        r = ${rw.rMove ? `(${num(rw.rMove)} if moved else 0.0) + ` : ''}((${num(rw.rCatch)} if self.caught else ${num(rw.rMiss)}) if end else 0.0)`];
      return [...head(['class World:', `    """A ball falls down a 5 x 8 board, one row a step${drift ? ', drifting sideways and bouncing off the walls' : ', straight down'}; a paddle on the`,
        `    bottom row moves left, stays or moves right. ${pays(`${say(rw.rCatch)} if the ball lands on the paddle, ${say(rw.rMiss)} if it misses${rw.rMove ? `, ${say(rw.rMove)} for every move` : ''}.`)}"""`, '    actions = ["left", "stay", "right"]']), '',
        '    def reset(self):', `        self.bx, self.by, self.vx, self.px = random.randrange(5), 0, ${drift ? 'random.randrange(3) - 1' : '0'}, 2`, '        self.prev, self.caught, self.t = None, None, 0', '        return self.obs()', '',
        ...(px ? ['    def picture(self, bx, by, px):', '        p = torch.zeros(8, 5)', '        p[by, bx] = 1.0                          # the ball', '        p[7, px] = 1.0                           # the paddle', '        return p', '',
          ...sense(['    def obs(self):                                # (2, 8, 5): the board now and a step ago', '        return torch.stack([self.picture(self.bx, self.by, self.px), self.picture(*(self.prev or (self.bx, self.by, self.px)))])'])]
          : sense(['    def obs(self):                                # where the ball is, which way it drifts, where the paddle is', '        return torch.tensor([self.bx / 4, self.by / 7, float(self.vx), self.px / 4])'])), '',
        '    def step(self, a):', '        self.prev = (self.bx, self.by, self.px)', '        self.px = min(max(self.px + a - 1, 0), 4)', '        self.by, self.bx = self.by + 1, self.bx + self.vx',
        '        if self.bx < 0:                           # a wall turns it back', '            self.bx, self.vx = 1, 1', '        elif self.bx > 4:', '            self.bx, self.vx = 3, -1',
        '        self.t += 1', '        moved, end = self.px != self.prev[2], self.by >= 7', '        if end:', '            self.caught = self.bx == self.px', ...reward, '        return self.obs(), r, end', '',
        '    def score(self, policy):', '        """The share of 40 balls its best moves catch (the same 40 every time)."""', '        caught = 0',
        ...SEEDED(4242, ['for _ in range(40):', ...PLAY('    '), '    caught += self.caught']).map((l) => '        ' + l), '        return caught / 40'];
    }
    // the maze
    const def = NA.rl.MAZE_SET.find((m) => m.id === ws.maze) || NA.rl.MAZE_SET[0], view = ws.view, rnd = ws.random, slip = ws.slip || 0, most = ws.maxSteps > 0 ? ws.maxSteps : 80;
    const obs = view === 'square' ? ['    def obs(self):                                # which square it is on: one input per square', '        return F.one_hot(torch.tensor(self.s), len(self.maze.cell)).float()']
      : view === 'picture' ? ['    def obs(self):                                # (5, 6, 8): walls, pits, the goal, where it is, where it has been', '        m = self.maze', '        p = torch.zeros(5, m.H, m.W)',
        '        for i, c in enumerate(m.cell):', '            y, x = divmod(i, m.W)', '            if c in "#XG":', '                p["#XG".index(c), y, x] = 1.0', '            if i in self.seen:', '                p[4, y, x] = 1.0', '        p[3, self.s // m.W, self.s % m.W] = 1.0', '        return p']
        : ['    def obs(self):', '        """The 24 squares around it, 2 in every direction (each: a wall or the edge, a pit, the goal, a square it has', '        been on), and which way the goal is: 104 inputs, each 0 or 1."""',
          '        m, x0, y0 = self.maze, self.s % self.maze.W, self.s // self.maze.W', '        v, k = torch.zeros(104), 0', '        for dy in range(-2, 3):', '            for dx in range(-2, 3):', '                if dx == 0 and dy == 0:', '                    continue',
          '                x, y = x0 + dx, y0 + dy', '                c = m.cell[y * m.W + x] if 0 <= x < m.W and 0 <= y < m.H else "#"', '                if c in "#XG":', '                    v["#XG".index(c) * 24 + k] = 1.0', '                elif y * m.W + x in self.seen:', '                    v[72 + k] = 1.0', '                k += 1',
          '        gx, gy = m.goal % m.W - x0, m.goal // m.W - y0', '        if gx > 0:', '            v[96 + (gx > 2)] = 1.0', '        if gx < 0:', '            v[98 + (gx < -2)] = 1.0', '        if gy > 0:', '            v[100 + (gy > 2)] = 1.0', '        if gy < 0:', '            v[102 + (gy < -2)] = 1.0', '        return v'];
    const fresh = !!f || !!rw.rNew;
    const reward = f ? ['        goal, pit, wall = float(c == "G"), float(c == "X"), float(self.s == s0)', '        new = float(fresh and c not in "GX")',
      '        closer = m.dist[s0] - m.dist[self.s] if c != "X" and m.dist[s0] >= 0 and m.dist[self.s] >= 0 else 0', '        dist, t = max(0, m.dist[self.s]), self.t', ...ownLine]
      : [`        r = ${num(rw.rGoal)} if c == "G" else ${num(rw.rPit)} if c == "X" else ${num(rw.rStep)}`,
        rw.rWall ? '        if self.s == s0:                          # walked into a wall' : null, rw.rWall ? `            r += ${num(rw.rWall)}` : null,
        rw.rNew ? '        if fresh and c not in "GX":               # a square it had not been on' : null, rw.rNew ? `            r += ${num(rw.rNew)}` : null,
        ws.closer ? '        if c != "X" and m.dist[s0] >= 0 and m.dist[self.s] >= 0:' : null, ws.closer ? `            r += ${num(rw.rCloser)} * (m.dist[s0] - m.dist[self.s])` : null];
    const paysText = f ? `Its reward is your own: r = ${f.py}` : `${say(rw.rGoal)} at the goal, ${say(rw.rPit)} in a pit, ${say(rw.rStep)} a step${rw.rWall ? `, ${say(rw.rWall)} into a wall` : ''}${rw.rNew ? `, ${say(rw.rNew)} on a new square` : ''}`;
    return [`MAZE = [${def.rows.map((r) => JSON.stringify(r)).join(', ')}]   # ${def.name}`, 'DIRS = [(0, -1), (1, 0), (0, 1), (-1, 0)]               # up, right, down, left', '', '',
      'class Maze:', '    def __init__(self, rows):', '        self.H, self.W, self.cell = len(rows), len(rows[0]), "".join(rows)', '        self.start, self.goal = self.cell.index("S"), self.cell.index("G")', '        self.dist = self.distances()', '',
      '    def move(self, s, a):                         # where move a leads from square s (into a wall or off the board: nowhere)', '        x, y = s % self.W + DIRS[a][0], s // self.W + DIRS[a][1]',
      '        if not (0 <= x < self.W and 0 <= y < self.H) or self.cell[y * self.W + x] == "#":', '            return s', '        return y * self.W + x', '',
      '    def distances(self):                          # the fewest safe steps from every square to the goal (-1: none)', '        d, queue = [-1] * len(self.cell), [self.goal]', '        d[self.goal] = 0', '        while queue:', '            s = queue.pop(0)', '            for a in range(4):',
      '                t = self.move(s, a)', '                if d[t] < 0 and self.cell[t] not in "X#":', '                    d[t] = d[s] + 1', '                    queue.append(t)', '        return d', '', '',
      ...(rnd ? ['def random_maze(rng):', '    """An 8 x 6 maze: about one square in five a wall, 1 to 3 pits, S and G on two free squares, kept when the', '    shortest safe way through takes 8 steps or more."""', '    while True:',
        '        g = [["#" if rng.random() < 0.22 else "." for _ in range(8)] for _ in range(6)]', '        for _ in range(1 + int(rng.random() * 3)):', '            g[int(rng.random() * 6)][int(rng.random() * 8)] = "X"',
        '        free = [(x, y) for y in range(6) for x in range(8) if g[y][x] == "."]', '        s, t = free[int(rng.random() * len(free))], free[int(rng.random() * len(free))]', '        if s == t:', '            continue',
        '        g[s[1]][s[0]], g[t[1]][t[0]] = "S", "G"', '        m = Maze(["".join(r) for r in g])', '        if m.dist[m.start] >= 8:', '            return m', '', ''] : []),
      ...head(['class World:', `    """${paysText}, at most ${most} steps${!f && ws.closer ? `; and a bonus of ${String(+(+rw.rCloser).toPrecision(12))} for every step that brings it` : '."""'}`,
        !f && ws.closer ? '    closer to the goal by the shortest safe way, as much off for every step away (reward shaping)."""' : null, '    actions = ["up", "right", "down", "left"]']), '',
      '    def __init__(self):', '        self.fixed = self.maze = Maze(MAZE)', rnd ? '        self.tests = [random_maze(random.Random(1234 + k)) for k in range(20)]   # 20 mazes it never trains on' : null, '',
      '    def reset(self, maze=None):', `        self.maze = maze or ${rnd ? 'random_maze(random)                      # a new random maze every episode' : 'self.fixed'}`, '        self.s, self.t, self.end = self.maze.start, 0, None', '        self.seen = {self.s}', '        return self.obs()', '',
      ...sense(obs), '',
      '    def step(self, a):', slip ? `        if random.random() < ${slip}:                 # a slippery floor: sideways instead` : null, slip ? '            a = (a + random.choice((1, 3))) % 4' : null,
      '        m, s0 = self.maze, self.s', '        self.s = m.move(self.s, a)', '        self.t += 1', fresh ? '        fresh = self.s not in self.seen' : null, '        self.seen.add(self.s)', '        c = m.cell[self.s]', `        done = c in "GX" or self.t >= ${most}`,
      '        if done:', '            self.end = "goal" if c == "G" else "pit" if c == "X" else "lost"', ...reward, '        return self.obs(), r, done', '',
      '    def score(self, policy):', rnd ? '        """The share of 20 new mazes its best moves find the goal of."""' : slip ? '        """The share of 5 tries in which its best moves reach the goal."""' : '        """1 if its best moves reach the goal, 0 if not."""',
      `        mazes, reached = ${rnd ? 'self.tests' : `[self.fixed] * ${slip ? 5 : 1}`}, 0`,
      ...SEEDED(4242, ['for m in mazes:', '    o, done = self.reset(m), False', '    while not done:', '        o, _, done = self.step(policy(o))', '    reached += self.end == "goal"']).map((l) => '        ' + l), '        return reached / len(mazes)'].filter((l) => l !== null);
  }
  // the method's training loop
  function methodPy(d, T, steps, every, setup) {
    const s = d.s, m = d.method, g = flt(s.gamma), B = T.batch;
    // evolution (a genetic algorithm, evolution strategies, MAP-Elites): builder-py-worlds.js
    if (NA.brl.isEvo(m)) return NA.builderPyWorlds.method(d, T, steps, { flt, num, big, ACT_FN, setup, every, evolvable: !!NA.bevo && NA.bevo.evolvable(setup.blocks, d.input) });
    const update = ['    opt.zero_grad()', '    loss.backward()', T.clip ? `    torch.nn.utils.clip_grad_norm_(model.parameters(), ${num(T.clip)})   # cap the size of the step` : null, '    opt.step()', T.sched !== 'const' ? '    sched.step()' : null];
    const report = (what) => [`    if step % ${every} == 0:`, `        print(f"step {step:5d}  ${what}  score {judge.score(best_move):.3f}")`];
    // moves in 8 copies of the world, sampled from the policy, the copies as one batch: actor-critic and PPO
    const roll = (n) => [`    for t in range(${n}):                         # ${n} moves in each of 8 copies of the world, the copies as one batch`, '        out = outputs(torch.stack(obs))', '        dist = torch.distributions.Categorical(logits=out[:, :n_actions])', '        acts = dist.sample()',
      '        X.append(torch.stack(obs)); A.append(acts); V.append(out[:, n_actions]); LP.append(dist.log_prob(acts))', '        rs, ds = [], []', '        for k, e in enumerate(envs):', '            o2, r, done = e.step(int(acts[k]))', '            obs[k] = e.reset() if done else o2', '            rs.append(r); ds.append(float(done))', '        R.append(torch.tensor(rs)); D.append(torch.tensor(ds))'];
    if (m === 'dqn') return [`gamma, explore, batch = ${g}, ${s.explore}, ${B}`, 'target = copy.deepcopy(model).eval()             # the target network: a copy, refreshed every 500 updates', 'memory = collections.deque(maxlen=10_000)          # the replay memory: (observation, move, reward, next observation or None)',
      'obs = env.reset()', `for step in range(${big(steps)}):`, '    eps = max(0.05, 1 - step / explore) if explore else 0.0   # the chance of a random move: from 1 down to 0.05',
      '    a = random.randrange(n_actions) if random.random() < eps else best_move(obs)', '    obs2, r, done = env.step(a)', '    memory.append((obs, a, r, None if done else obs2))', '    obs = env.reset() if done else obs2', '    if len(memory) < 64:', '        continue',
      '    sample = [memory[random.randrange(len(memory))] for _ in range(batch)]', '    X, A = torch.stack([q[0] for q in sample]), torch.tensor([q[1] for q in sample])',
      '    R, alive = torch.tensor([q[2] for q in sample]), torch.tensor([float(q[3] is not None) for q in sample])', '    X2 = torch.stack([q[0] if q[3] is None else q[3] for q in sample])',
      '    with torch.no_grad():                         # double DQN: the network picks the next move, the target network says what it is worth', '        nxt = outputs(X2)[:, :n_actions].argmax(1)', '        y = R + gamma * alive * target(X2).gather(1, nxt[:, None])[:, 0]',
      '    q = model(X).gather(1, A[:, None])[:, 0]', '    loss = F.smooth_l1_loss(q, y)                  # Huber: squared near the target, absolute beyond 1', ...update,
      '    if step % 500 == 499:', '        target.load_state_dict(model.state_dict())', ...report('loss {loss.item():.4f}')];
    if (m === 'pg') return [`gamma = ${g}`, 'obs = env.reset()', `for step in range(${big(steps)}):`, '    batch_obs, batch_a, batch_G = [], [], []',
      '    while len(batch_obs) < 64:                    # whole episodes until about 64 moves are in hand', '        rewards, done = [], False', '        while not done:', '            a = int(torch.multinomial(outputs(obs[None])[0, :n_actions].softmax(0), 1))   # a move sampled from the policy',
      '            obs2, r, done = env.step(a)', '            batch_obs.append(obs); batch_a.append(a); rewards.append(r)', '            obs = env.reset() if done else obs2', '        G, returns = 0.0, []', '        for r in reversed(rewards):                # the discounted return after each move',
      '            G = r + gamma * G', '            returns.append(G)', '        batch_G += returns[::-1]', '    G = torch.tensor(batch_G)', '    sd = G.std(unbiased=False)', '    adv = (G - G.mean()) / (sd if sd > 0 else 1)    # how much better than usual, in its spread',
      '    logp = model(torch.stack(batch_obs))[:, :n_actions].log_softmax(1)', '    entropy = -(logp.exp() * logp).sum(1)', '    loss = -(adv * logp[torch.arange(len(adv)), torch.tensor(batch_a)]).mean() - 0.01 * entropy.mean()', ...update, ...report('loss {loss.item():.4f}')];
    if (m === 'a2c') return [`gamma = ${g}`, 'envs = [World() for _ in range(8)]', 'obs = [e.reset() for e in envs]', `for step in range(${big(steps)}):`, '    X, A, R, V, LP, D = [], [], [], [], [], []', ...roll(8),
      '    ret = outputs(torch.stack(obs))[:, n_actions]   # the critic\'s value of where each copy ended up', '    returns = []', '    for t in reversed(range(8)):                 # the returns, with that value standing in for what comes after', '        ret = R[t] + gamma * ret * (1 - D[t])', '        returns.insert(0, ret)',
      '    returns, V = torch.stack(returns).flatten(), torch.stack(V).flatten()', '    adv = returns - V                              # the advantage: better or worse than the critic expected', '    adv = (adv - adv.mean()) / (adv.std(unbiased=False) + 1e-8)',
      '    out = model(torch.cat(X))', '    logp = out[:, :n_actions].log_softmax(1)', '    entropy = -(logp.exp() * logp).sum(1)', '    chosen = logp[torch.arange(len(logp)), torch.cat(A)]',
      '    loss = -(adv * chosen).mean() - 0.01 * entropy.mean() + 0.25 * ((out[:, n_actions] - returns) ** 2).mean()', ...update, ...report('loss {loss.item():.4f}')];
    if (m === 'ppo') return [`gamma, lam, batch = ${g}, 0.95, ${B}`, 'envs = [World() for _ in range(8)]', 'obs = [e.reset() for e in envs]', 'queue = []', `for step in range(${big(steps)}):`, '    if not queue:', '        X, A, R, V, LP, D = [], [], [], [], [], []', ...roll(16).map((l) => '    ' + l),
      '        nxt, gae, adv = outputs(torch.stack(obs))[:, n_actions], torch.zeros(8), []', '        for t in reversed(range(16)):            # advantages by GAE (λ = 0.95)', '            delta = R[t] + gamma * nxt * (1 - D[t]) - V[t]', '            gae = delta + gamma * lam * (1 - D[t]) * gae', '            adv.insert(0, gae)', '            nxt = V[t]',
      '        adv = torch.stack(adv).flatten()', '        ret = adv + torch.stack(V).flatten()', '        adv = (adv - adv.mean()) / (adv.std(unbiased=False) + 1e-8)', '        data = (torch.cat(X), torch.cat(A), torch.cat(LP), adv, ret)',
      '        for _ in range(4):                       # 4 passes over the 128 moves, in shuffled minibatches', '            perm = torch.randperm(len(adv))', '            queue += [perm[i:i + batch] for i in range(0, len(perm), batch)]',
      '    i = queue.pop(0)', '    x, a, old_logp, adv_i, ret_i = (q[i] for q in data)', '    out = model(x)', '    ratio = (out[:, :n_actions].log_softmax(1)[torch.arange(len(i)), a] - old_logp).exp()',
      '    clipped = ratio.clamp(0.8, 1.2)                # no further than 20% from the policy that played', '    loss = -torch.min(ratio * adv_i, clipped * adv_i).mean() + 0.25 * ((out[:, n_actions] - ret_i) ** 2).mean()', ...update, ...report('loss {loss.item():.4f}')];
  }
  function controlCode(setup, d, chk) {
    const p = BT.byId[setup.problem], T = setup.train, steps = T.steps || NA.brl.stepsFor(d.world, d.method), M = NA.brl.METHODS[d.method], evo = NA.brl.isEvo(d.method), s = d.s;
    const how = { ga: `a population of ${s.pop}${s.grow && NA.bevo && NA.bevo.evolvable(setup.blocks, d.input) ? ' whose networks grow and shrink' : ''}`, es: `${Math.max(2, Math.round((s.pop || 24) / 2))} pairs of nudged copies a generation`, me: `${s.pop} new networks a generation, kept in a 12 × 12 grid of styles` }[d.method];
    const optName = { sgd: 'SGD', momentum: 'SGD with momentum', adam: 'Adam', adamw: 'AdamW' }[T.opt], every = Math.max(1, Math.round(steps / 10));
    return lines(
      `# Built in Neural Anatomy's model builder: ${p.label}, learning from rewards.`,
      `# Model:     ${summary(setup, d)}`,
      ...(Object.keys(partNames(setup).cls).length ? ['# Your parts, each a class below:', ...partsSummary(setup)] : []),
      `# Learns by: ${M.label}${evo ? `, ${how}` : `, ${optName}, learning rate ${num(T.lr)}, batch ${T.batch}`}, ${steps.toLocaleString('en')} ${evo ? 'generations' : 'steps'}${T.steps ? '' : ' (the page trains until you pause)'}`,
      '# Only torch is needed.', 'import math', 'import random', d.method === 'dqn' ? 'import collections' : null, d.method === 'dqn' ? 'import copy' : null, 'import torch', 'import torch.nn as nn', 'import torch.nn.functional as F', '',
      `torch.manual_seed(${T.seed})`, `random.seed(${T.seed})`, '',
      '# ---------------------------------------------------------------- the world', ...worldPy(d), ...(d.method === 'me' && !NA.builderPyWorlds.worlds[d.world] ? ['', '', ...NA.builderPyWorlds.style(d.world)] : []), '', '',
      'env, judge = World(), World()                     # one to learn in, one to measure its best moves in', 'n_actions = len(World.actions)', '', '',
      '# observations from random play (there are no examples with answers here): the architecture view\'s snippets', '# use the first as "the first validation example"', 'def random_observations(n):', '    w, out = World(), []', '    o = w.reset()', '    while len(out) < n:', '        out.append(o)', '        o, _, done = w.step(random.randrange(n_actions))', '        if done:', '            o = w.reset()', '    return torch.stack(out)', '', '', 'Xva = random_observations(40)', '', '',
      modelCode(setup, d, chk), '', '',
      'model = Model()', initCode(setup, d), 'print(f"{sum(p.numel() for p in model.parameters()):,} learned numbers")', '', '',
      'def outputs(X):                                    # the outputs for a batch of observations, without learning', '    model.eval()', '    with torch.no_grad():', '        out = model(X)', '    model.train()', '    return out', '', '',
      'def best_move(obs):                                # its best move: the largest output', '    return int(outputs(obs[None])[0, :n_actions].argmax())', '', '',
      evo ? null : ['# ---------------------------------------------------------------- the optimizer', ...optCode(T), '', ''],
      `# ---------------------------------------------------------------- learning by ${M.label}`, ...methodPy(d, T, steps, every, setup), '',
      'print(f"done: its best moves score {judge.score(best_move):.3f}")', '',
      '# ---------------------------------------------------------------- watch it play one episode with its best moves', 'o, done, total = env.reset(), False, 0.0', 'while not done:', '    a = best_move(o)', '    o, r, done = env.step(a)', '    total += r', '    print(World.actions[a], end="  ")', 'print(f"\\nreturn {total:.2f}")');
  }

  // the scripts that use random (the stroke drawings, the stories, the chaotic series, the sequence classes)
  const needsRandom = (p, d) => d.family === 'digits' || p.task === 'language' || p.task === 'chaos' || d.family === 'seqclass';
  PY.code = function (setup, d) {
    const p = BT.byId[setup.problem], T = setup.train, steps = T.steps || 3000;
    if (!d || d.error) return `# ${d && d.error ? d.error : 'Pick a problem first.'}`;
    if (d.family === 'control' && NA.cml && setup.blocks.some((b) => NA.cml.isClassic(b.type))) return '# A classic model learns from examples with answers, and here there are only rewards: use a network.';
    if (NA.cml && setup.blocks.some((b) => NA.cml.isClassic(b.type))) {
      const cc = NA.cml.check(setup.blocks, d);
      if (!cc.ok) return lines('# The model does not fit together yet, so there is nothing to write out.', `# ${((cc.shapes.find((s) => s && s.problem) || {}).problem || { reason: 'See the block marked in red.' }).reason}`);
      return classicCode(setup, d);
    }
    const chk = NA.bx ? NA.bx.check(setup.blocks, d.input, d.output) : { ok: false, shapes: [] };
    if (!chk.ok) return lines('# The model does not fit together yet, so there is nothing to write out.', `# ${((chk.problem || (chk.shapes.find((s) => s && s.problem) || {}).problem) || { reason: 'See the block marked in red.' }).reason}`);
    if (d.family === 'control') return controlCode(setup, d, chk);
    const grows = !!T.grow && !!NA.bevo && NA.bevo.evolvable(setup.blocks, d.input), rnd = needsRandom(p, d) || grows; // (grow and prune picks its changes at random)
    const needs = ['import math', p.family === 'table' && setup.custom.table && setup.custom.table.rows.length * (d.feats ? d.feats.length + 1 : 1) > 2400 ? 'import csv' : null, rnd ? 'import random' : null, p.task === 'sentences' || p.task === 'language' ? 'import re' : null];
    const lossName = NA.bx.LOSSES[setup.loss.id] ? NA.bx.LOSSES[setup.loss.id].label : setup.loss.id;
    const optName = { sgd: 'SGD', momentum: 'SGD with momentum', adam: 'Adam', adamw: 'AdamW' }[T.opt];
    const evalEvery = Math.max(50, Math.round(steps / 10 / 50) * 50);
    // grow and prune while it learns (Dense blocks on a list of numbers): 6 networks whose shapes evolve, builder-py-worlds.js
    const grow = grows, HE = (a) => !!(NA.engine.ACTIVATIONS[a] || {}).he;
    const lrAt = !T.steps ? null : T.sched === 'cosine' ? ['def lr_at(step):                                 # the learning rate falls along half a cosine', `    return LR * 0.5 * (1 + math.cos(math.pi * step / ${steps}))`]
      : T.sched === 'step' ? ['def lr_at(step):                                 # a tenth of it after half the steps, a hundredth after three quarters', `    return LR * 0.1 ** ((step >= ${Math.round(steps / 2)}) + (step >= ${Math.round(steps * 0.75)}))`]
        : T.sched === 'warmup' ? ['def lr_at(step):                                 # warm up, then a cosine', `    warm = ${Math.max(1, Math.round(steps * 0.05))}`, `    return LR * (step / warm if step < warm else 0.5 * (1 + math.cos(math.pi * (step - warm) / (${steps} - warm))))`] : null;
    const growing = grow && NA.builderPyWorlds.growSupervised({ flt, num, big, ACT_FN, setup, d, evalEvery, clip: T.clip, lrAt,
      gains: T.init === 'small' ? null : [...setup.blocks.map((b) => (T.init === 'he' || (T.init === 'auto' && HE(b.act)) ? 2 : 1)), 1] });
    return lines(
      `# Built in Neural Anatomy's model builder: ${p.label}.`,
      `# Model:     ${summary(setup, d)}`,
      ...(Object.keys(partNames(setup).cls).length ? ['# Your parts, each a class below:', ...partsSummary(setup)] : []),
      `# Training:  ${lossName} loss, ${optName}, learning rate ${num(T.lr)}, batch ${T.batch}, ${steps.toLocaleString('en')} steps${T.steps ? '' : ' (the page trains until you pause)'}${grow ? '; 6 networks side by side, their shapes evolving (grow and prune)' : ''}`,
      ...needs.filter(Boolean), 'import torch', 'import torch.nn as nn', 'import torch.nn.functional as F', '',
      `torch.manual_seed(${T.seed})`, rnd ? `random.seed(${T.seed})` : null, '',
      dataCode(setup, d), '', '',
      modelCode(setup, d, chk), '', '',
      'model = Model()', initCode(setup, d), 'print(f"{sum(p.numel() for p in model.parameters()):,} learned numbers")', '',
      '# ---------------------------------------------------------------- loss, optimizer and measure', lossCode(setup, d), '', scoreCode(d), '',
      grow ? ['', growing] : [optCode(T), '',
      'def evaluate(X, Y):', '    model.eval()', '    with torch.no_grad():', '        out = model(X)', '        loss, s = loss_fn(out, Y).item(), score(out, Y).item()', '    model.train()', '    return loss, s', '',
      '# ---------------------------------------------------------------- training', `batch = ${T.batch}`, 'order, pos = torch.randperm(len(Xtr)), 0',
      `for step in range(${big(steps)}):`,
      '    if pos + batch > len(order):                  # a new shuffle every pass over the data', '        order, pos = torch.randperm(len(Xtr)), 0',
      '    i = order[pos:pos + batch]', '    pos += batch',
      '    loss = loss_fn(model(Xtr[i]), Ytr[i])', '    opt.zero_grad()', '    loss.backward()',
      T.clip ? `    torch.nn.utils.clip_grad_norm_(model.parameters(), ${num(T.clip)})   # cap the size of the step` : null,
      '    opt.step()', T.sched !== 'const' ? '    sched.step()' : null,
      `    if step % ${evalEvery} == 0:`, '        val_loss, val_score = evaluate(Xva, Yva)', '        print(f"step {step:5d}  training loss {loss.item():.4f}  validation loss {val_loss:.4f}  score {val_score:.3f}")', '',
      'val_loss, val_score = evaluate(Xva, Yva)', 'print(f"done: validation loss {val_loss:.4f}, score {val_score:.3f}")'],
      afterCode(setup, d));
  };

  NA.builderPy = PY;
})(window.NA);
