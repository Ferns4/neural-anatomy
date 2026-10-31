/* builder-py.js — what you built in the model builder, written out as one PyTorch script you can run: the data
   (the same generators as the page, in Python), the model, the loss, the optimizer and its schedule, and the
   training loop with the same measures the page shows. Only torch is needed. A classic model (linear, k-NN,
   trees, a forest, boosting) is written with scikit-learn, or XGBoost for boosting, on the same data. Learning
   from rewards writes the world in Python (its settings, reward and extra inputs, step for step as on the page)
   and the method's own loop: DQN, REINFORCE, actor-critic, PPO, or evolution (builder-py-worlds.js). */
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
    brackets: () => String.raw`def balanced(ids):
    """( is 0 and ) is 1: balanced when the count of open ones never drops below zero and ends at zero."""
    h = 0
    for v in ids:
        h += -1 if v else 1
        if h < 0:
            return False
    return h == 0

def example():
    """As many ( as ) in most (a third of the unbalanced ones have two more of one kind), always opening first and
    closing last, as balanced ones do, so the ends give nothing away: only a running count tells."""
    want = random.randrange(2)
    while True:
        opens = T // 2 if want or random.random() < 0.67 else T // 2 + random.choice((1, -1))
        ids = [0] * opens + [1] * (T - opens)
        random.shuffle(ids)
        if int(balanced(ids)) == want and ids[0] == 0 and ids[-1] == 1:
            return ids, want`,
    palindrome: () => String.raw`def example():
    """Digits. Half are palindromes; the other half differ from one in a single digit, so one pair no longer matches."""
    ids = [random.randrange(10) for _ in range(T)]
    for i in range(T // 2):
        ids[T - 1 - i] = ids[i]
    label = random.randrange(2)
    if not label:
        at = random.randrange(T)
        ids[at] = (ids[at] + 1 + random.randrange(9)) % 10
        if ids[at] == ids[T - 1 - at]:                # the middle digit of an odd length: still a palindrome
            return None, -1
    return ids, label`,
  };
  // the pictures of js/builder-tasks.js beside the digits and the shapes, in Python: dots to count, stripes, coloured
  // shapes and a dot inside or outside a ring
  const PICTURE_HEAD = String.raw`G = 14                                        # pictures are G x G pixels
py, px = torch.meshgrid(torch.arange(G) + 0.5, torch.arange(G) + 0.5, indexing="ij")   # each pixel's middle

def dot_ink(x, y, r):
    """A round dot of ink: 1 inside, fading to 0 over a pixel at its edge."""
    return (r + 0.5 - torch.hypot(px - x, py - y)).clamp(0, 1)`;
  const PICTURES_PY = {
    dots: (s) => String.raw`def draw_dots(k):
    """k dots apart from each other (${s.sizes === 'same' ? 'all one size' : 'of sizes 0.55 to 1.6'}), a little noise."""
    while True:                                  # too crowded for one more after 200 tries: start the picture again
        pts = []
        for _ in range(200):
            r = ${s.sizes === 'same' ? '1.0' : 'random.uniform(0.55, 1.6)'}
            x, y = random.uniform(r + 0.3, G - r - 0.3), random.uniform(r + 0.3, G - r - 0.3)
            if all(math.hypot(qx - x, qy - y) > qr + r + 1.2 for qx, qy, qr in pts):
                pts.append((x, y, r))
                if len(pts) == k:
                    ink = torch.zeros(G, G)
                    for qx, qy, qr in pts:
                        ink = torch.maximum(ink, dot_ink(qx, qy, qr))
                    return (ink + 0.04 * torch.randn(G, G))[None]

K = ${s.most}                                         # 1 to K dots: K classes
n = ${s.n}
X = torch.stack([draw_dots(1 + i % K) for i in range(n)])     # (n, 1, 14, 14)
Y = torch.arange(n) % K                          # class k: k + 1 dots`,
    stripes: (s) => String.raw`DIRS = [0, math.pi / 2, math.pi / 4, -math.pi / 4]   # across, up and down, rising, falling

def draw_stripes(k):
    """Stripes running one way, at any spacing and shift, faint or strong, in noise."""
    a, per, ph, c = DIRS[k], random.uniform(2.6, 5), random.uniform(0, 2 * math.pi), random.uniform(0.2, 0.5)
    nx, ny = -math.sin(a), math.cos(a)
    return (0.5 + c * torch.sin(2 * math.pi * (px * nx + py * ny) / per + ph) + ${num(s.noise)} * torch.randn(G, G))[None]

K = ${s.dirs}                                         # directions: K classes
n = ${s.n}
X = torch.stack([draw_stripes(i % K) for i in range(n)])      # (n, 1, 14, 14)
Y = torch.arange(n) % K`,
    colours: (s) => String.raw`COLOURS = [(0.95, 0.15, 0.1), (0.15, 0.85, 0.2), (0.2, 0.25, 0.75)]   # red, green, blue: equally bright on average

def draw_coloured(k):
    """Class k: colour k // 2, a circle (k even) or a cross (k odd), anywhere, at any size and angle${s.colour === 'grey' ? ', in grey' : ''}."""
    col, cross = COLOURS[k // 2], k % 2
    r = random.uniform(2.6, 4.6)
    x, y = random.uniform(r + 0.5, G - r - 0.5), random.uniform(r + 0.5, G - r - 0.5)
    a, b = random.uniform(0, math.pi / 2), random.uniform(0.75, 1)
    dx, dy = px - x, py - y
    u, v = (dx * math.cos(a) + dy * math.sin(a)).abs(), (-dx * math.sin(a) + dy * math.cos(a)).abs()
    if cross:                                    # two bars about two pixels thick, turned by a
        ink = torch.minimum(1.3 - torch.minimum(u, v), r + 0.5 - torch.maximum(u, v)).clamp(0, 1)
    else:
        ink = (r + 0.5 - torch.hypot(dx, dy)).clamp(0, 1)
${s.colour === 'grey' ? '    return (0.1 + ink * b * sum(col) / 3 + 0.04 * torch.randn(G, G))[None]      # one channel: its brightness' : '    return torch.stack([0.1 + ink * b * c + 0.04 * torch.randn(G, G) for c in col])   # three channels: red, green, blue'}

n = ${s.n}
X = torch.stack([draw_coloured(i % 6) for i in range(n)])     # (n, ${s.colour === 'grey' ? 1 : 3}, 14, 14)
Y = torch.arange(n) % 6                          # red circle, red cross, green circle, green cross, blue circle, blue cross`,
    inside: (s) => String.raw`def draw_ring_dot(k):
    """A ring anywhere, at any size, and a dot inside it (k = 1) or outside it, clear of the ring by more than a pixel."""
    r = random.uniform(2.8, 5)
    cx, cy = random.uniform(r + 0.6, G - r - 0.6), random.uniform(r + 0.6, G - r - 0.6)
    x, y = cx, cy
    for _ in range(500):
        x, y = random.uniform(1, G - 1), random.uniform(1, G - 1)
        d = math.hypot(x - cx, y - cy)
        if (d < r - 1.6) if k else (d > r + 1.6):
            break
    ring = (1.4 - (torch.hypot(px - cx, py - cy) - r).abs()).clamp(0, 1)
    return (torch.maximum(ring, dot_ink(x, y, 0.7)) + 0.04 * torch.randn(G, G))[None]

n = ${s.n}
X = torch.stack([draw_ring_dot(i % 2) for i in range(n)])     # (n, 1, 14, 14)
Y = torch.arange(n) % 2                          # 0 outside, 1 inside`,
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
    const own = NA.builderPyGen && NA.builderPyGen.data(setup, d, p, s, PY._i);
    if (own) { out.push(...own); return out; }
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
      } else if (p.task === 'wiggles') out.push(`x = 2 * torch.rand(${s.n}) - 1                    # anywhere in [-1, 1]`, `y = 0.8 * torch.sin(math.pi * ${s.waves} * x + 0.4)       # up and down ${s.waves} time${s.waves === 1 ? '' : 's'}`);
      else out.push(`x = -1 + 2 * (torch.arange(${s.n}) + torch.rand(${s.n})) / ${s.n}     # spread evenly over [-1, 1]`, `y = ${CURVE_PY[p.dataset]}`);
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
    } else if (fam === 'digits' && PICTURES_PY[p.task]) {
      out.push(PICTURE_HEAD, '', PICTURES_PY[p.task](s), ...SPLIT(0.8));
    } else if (fam === 'digits' && p.task === 'shapes') {
      out.push(SHAPES_PY, '', `n = ${s.n}`, 'X = torch.stack([draw_shape(i % 5) for i in range(n)])      # (n, 1, 14, 14), ink in [0, 1]',
        'Y = torch.arange(n) % 5', ...SPLIT(0.8));
    } else if (fam === 'digits') {
      out.push(DIGITS_PY, '', `n = ${s.n}`, 'X = torch.stack([draw_digit(i % 10) for i in range(n)])     # (n, 1, 14, 14), ink in [0, 1]',
        'Y = torch.arange(n) % 10', ...SPLIT(0.8));
    } else if (fam === 'surface') {
      // a picture as a height: a smiling face whose edges are sigmoids, sharp or soft
      if (p.task === 'picture') out.push(`SOFT = ${num((BT.FACE_SOFT || {})[s.edges] || 0.008)}                                   # how wide each edge is: ${s.edges === 'soft' ? 'soft' : 'sharp'}`, 'def surface(x, y):',
        '    """A smiling face: the face high, its eyes and mouth low; each edge a sigmoid of the distance across it."""', '    sig = lambda d: torch.sigmoid(d / SOFT)', '    face = sig(0.78 - torch.hypot(x, y))',
        '    eyes = torch.maximum(sig(0.12 - torch.hypot(x + 0.28, y - 0.26)), sig(0.12 - torch.hypot(x - 0.28, y - 0.26)))', '    mouth = sig(0.075 - (torch.hypot(x, y - 0.12) - 0.45).abs()) * sig(-0.12 - y)',
        '    return -0.6 + 1.4 * face - 1.5 * torch.maximum(eyes, mouth) * face', '',
        `X = 2 * torch.rand(${s.n}, 2) - 1                     # points spread over the square [-1, 1]²`,
        `Y = surface(X[:, 0], X[:, 1]) + ${num(s.noise)} * torch.randn(${s.n})     # the height there, plus noise`, ...SPLIT(0.8));
      else out.push('def surface(x, y):', `    return ${SURFACE_PY[p.task]}`, '',
        `X = 2 * torch.rand(${s.n}, 2) - 1                     # points spread over the square [-1, 1]²`,
        `Y = surface(X[:, 0], X[:, 1]) + ${num(s.noise)} * torch.randn(${s.n})     # the height there, plus noise`, ...SPLIT(0.8));
    } else if (fam === 'series') {
      if (p.task === 'chaos') out.push(`v, x = torch.empty(${s.n}), 0.31 + 0.02 * random.random()`, 'for i in range(len(v)):                           # the logistic map: fully determined, yet chaotic', '    v[i] = 2 * x - 1', '    x = 3.9 * x * (1 - x)');
      else if (p.task === 'walk') out.push(`v = torch.cumsum(0.05 * torch.randn(${s.n}), 0)              # a random walk: each step a nudge of random size`);
      else if (p.task === 'heartbeat') out.push('BEAT = [(-1, 0.5, -0.12), (0, 0.55, 1.0), (1, 0.6, -0.28), (6, 1.6, 0.3)]   # around each beat: (when, how wide, how high)',
        `v, t, beats, b = torch.zeros(${s.n}), torch.arange(${s.n}, dtype=torch.float32), [], random.randrange(20)`, `while b < ${s.n} + 20:                                # a beat every ${s.rhythm === 'irregular' ? '22 to 38 steps' : '30 steps'}`, '    beats.append(b)',
        `    b += ${s.rhythm === 'irregular' ? '22 + random.randrange(17)' : '30'}`, 'for b in beats:', '    near = ((t - b > -6) & (t - b < 14)).float()', '    for at, w, h in BEAT:', '        v += near * h * torch.exp(-(t - b - at) ** 2 / (2 * w * w))');
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
      if (p.task === 'add') {
        out.push(`D = ${s.digits}                                         # digits in each number; tokens 0-9 are digits, 10 the arrow (=), 11 the plus`,
          'def make_sums(n):', '    rows, seen = [], set()', '    while len(rows) < min(n, 10 ** (2 * D) - 50):', '        a, b = random.randrange(10 ** D), random.randrange(10 ** D)', '        if (a, b) in seen:', '            continue',
          '        seen.add((a, b))', '        digits = lambda v, k: [int(c) for c in str(v).zfill(k)]', '        rows.append(digits(a, D) + [11] + digits(b, D) + [10] + digits(a + b, D + 1))   # a + b → the sum, largest digit first',
          '    seq = torch.tensor(rows)', '    X, Y = seq[:, :-1], seq[:, 1:].clone()', '    Y[:, :2 * D + 1] = -1                    # train only on the answer (the question is random)', '    return X, Y', '',
          `X, Y = make_sums(${s.n})`, 'k = len(X) - max(50, round(0.12 * len(X)))', 'Xtr, Ytr, Xva, Yva = X[:k], Y[:k], X[k:], Y[k:]');
        return out;
      }
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
  // the newest parts, each a class (or a function) the model uses
  const MORE_PY = {
    sepconv: String.raw`class SeparableConv(nn.Module):
    """A k x k filter for each input channel on its own (depthwise: groups=c_in), then a 1 x 1 convolution that mixes
    the channels (pointwise): far fewer weights than one full convolution, as in MobileNet."""
    def __init__(self, c_in, filters, k, stride=1, dilation=1):
        super().__init__()
        self.depthwise = nn.Conv2d(c_in, c_in, k, stride=stride, padding=dilation * (k - 1) // 2, dilation=dilation, groups=c_in)
        self.pointwise = nn.Conv2d(c_in, filters, 1)

    def forward(self, x):
        return self.pointwise(self.depthwise(x))`,
    se: String.raw`class SqueezeExcite(nn.Module):
    """Squeeze-and-excitation: each channel's average over its positions goes through a small network whose sigmoid
    outputs, one per channel between 0 and 1, then scale the channels."""
    def __init__(self, channels, hidden):
        super().__init__()
        self.squeeze, self.excite = nn.Linear(channels, hidden), nn.Linear(hidden, channels)

    def forward(self, x):
        s = x.mean((2, 3)) if x.dim() == 4 else x.mean(1) if x.dim() == 3 else x   # a picture's pixels, a sequence's positions
        g = torch.sigmoid(self.excite(F.relu(self.squeeze(s))))
        return x * (g[:, :, None, None] if x.dim() == 4 else g[:, None, :] if x.dim() == 3 else g)`,
    coords: String.raw`def coord_channels(H, W, radius):
    """Each pixel's column and row, from -1 to 1 (with radius, also its distance from the centre, 0 to 1)."""
    y, x = torch.meshgrid(torch.linspace(-1, 1, H), torch.linspace(-1, 1, W), indexing="ij")
    return torch.stack([x, y] + ([torch.sqrt(x ** 2 + y ** 2) / math.sqrt(2)] if radius else []))`,
    augment: String.raw`class RandomShift(nn.Module):
    """While training, moves each picture by up to shift pixels each way (zeros come in) and, with flip, mirrors half
    of them left to right; when the model is used it does nothing."""
    def __init__(self, shift, flip=False):
        super().__init__()
        self.shift, self.flip = shift, flip

    def forward(self, x):
        if not self.training:
            return x
        s, (B, C, H, W) = self.shift, x.shape
        xp, out = F.pad(x, (s, s, s, s)), torch.empty_like(x)
        for i in range(B):
            oy, ox = torch.randint(0, 2 * s + 1, (2,)).tolist()
            out[i] = xp[i, :, oy:oy + H, ox:ox + W]
        if self.flip:
            mirror = torch.rand(B) < 0.5
            out[mirror] = out[mirror].flip(-1)
        return out`,
    attnpool: String.raw`class AttentionPool(nn.Module):
    """One learned query scores every position (every pixel of a picture); a softmax turns the scores into weights
    that add up to 1, and the output is the weighted average of the positions."""
    def __init__(self, d):
        super().__init__()
        self.query = nn.Linear(d, 1, bias=False)

    def forward(self, x):
        if x.dim() == 4:
            x = x.flatten(2).transpose(1, 2)               # (batch, pixels, channels)
        a = self.query(x).softmax(1)                       # (batch, positions, 1): where it looks
        return (a * x).sum(1)`,
    ssm: String.raw`class StateSpace(nn.Module):
    """A linear recurrence: state cells that each keep a moving average of a learned mix of the input, with their own
    learned memory a = sigmoid(theta): h_t = a * h_(t-1) + (1 - a) * B x_t, then out_t = C h_t + c."""
    def __init__(self, d_in, state, units):
        super().__init__()
        tau = 2 ** (1 + 4 * torch.arange(state) / max(1, state - 1))   # memories from about 2 to 32 steps
        a = torch.exp(-1 / tau)
        self.theta = nn.Parameter(torch.log(a / (1 - a)))
        self.B, self.C = nn.Linear(d_in, state, bias=False), nn.Linear(state, units)

    def forward(self, x):                                  # x: (batch, T, d_in)
        a, u = torch.sigmoid(self.theta), self.B(x)
        h, hs = torch.zeros_like(u[:, 0]), []
        for t in range(x.shape[1]):                        # one step at a time (a parallel scan would do all at once)
            h = a * h + (1 - a) * u[:, t]
            hs.append(h)
        return self.C(torch.stack(hs, 1))`,
    rbf: String.raw`class RBF(nn.Module):
    """Radial basis units: each answers how close the input is to its learned centre, exp(-|x - c|^2 / 2 sigma^2)."""
    def __init__(self, d_in, units, width):
        super().__init__()
        self.centres = nn.Parameter(torch.empty(units, d_in).uniform_(-1, 1))
        self.log_width = nn.Parameter(torch.full((units,), math.log(width)))

    def forward(self, x):
        d2 = ((x[..., None, :] - self.centres) ** 2).sum(-1)   # squared distance to every centre
        return torch.exp(-d2 / (2 * torch.exp(2 * self.log_width)))`,
    maxout: String.raw`class Maxout(nn.Module):
    """Each unit computes several weighted sums (its pieces) and keeps the largest: a bend it learns itself."""
    def __init__(self, d_in, units, pieces):
        super().__init__()
        self.units, self.pieces = units, pieces
        self.lin = nn.Linear(d_in, units * pieces)

    def forward(self, x):
        z = self.lin(x)
        return z.view(*z.shape[:-1], self.units, self.pieces).max(-1).values`,
    moe: String.raw`class MixtureOfExperts(nn.Module):
    """Experts (small dense layers) and a gate that weighs them for each example; with top, only the top experts the
    gate rates highest take part, their weights renormalised."""
    def __init__(self, d_in, units, experts, top, act):
        super().__init__()
        self.experts = nn.ModuleList([nn.Linear(d_in, units) for _ in range(experts)])
        self.gate, self.top, self.act = nn.Linear(d_in, experts), top, act

    def forward(self, x):
        logits = self.gate(x)
        if 0 < self.top < logits.shape[-1]:                # the others get no say
            kth = logits.topk(self.top, -1).values[..., -1:]
            logits = logits.masked_fill(logits < kth, float("-inf"))
        g = logits.softmax(-1)                             # (..., experts)
        outs = torch.stack([self.act(e(x)) for e in self.experts], -1)   # (..., units, experts)
        return (outs * g[..., None, :]).sum(-1)`,
    sample: String.raw`class Sample(nn.Module):
    """A VAE's middle: the first half of its input are middles mu, the second half the logs of their spreads squared.
    While training it gives mu + sigma * eps, eps drawn from N(0, 1) (the reparameterisation trick: the gradient flows
    through mu and sigma), otherwise mu. It keeps beta x the KL divergence from N(0, 1), summed over its numbers and
    averaged over the batch, in .kl for the loss."""
    def __init__(self, beta):
        super().__init__()
        self.beta, self.kl = beta, torch.tensor(0.0)

    def forward(self, x):
        mu, lv = x.chunk(2, -1)
        lv = lv.clamp(-8, 8)
        self.kl = self.beta * 0.5 * (mu ** 2 + lv.exp() - 1 - lv).sum(-1).mean()
        return mu + (0.5 * lv).exp() * torch.randn_like(mu) if self.training else mu


def kl(model):
    """The KL divergences of the model's Sample blocks: part of a VAE's loss."""
    return sum(m.kl for m in model.modules() if isinstance(m, Sample))`,
    kan: String.raw`class KAN(nn.Module):
    """Kolmogorov-Arnold layer: on every edge from an input to a unit a function of its own, learned: a weight on
    SiLU(x), plus a sum of G bumps (Gaussians spread evenly over -2...2) each with a weight of its own."""
    def __init__(self, d_in, units, grid):
        super().__init__()
        self.register_buffer("centres", torch.linspace(-2, 2, grid))
        self.h = 4 / max(1, grid - 1)                           # the bumps' width: their spacing
        self.base = nn.Linear(d_in, units)                      # the SiLU part, and the bias
        self.coef = nn.Parameter(0.1 / math.sqrt(d_in) * torch.randn(units, d_in, grid))

    def forward(self, x):
        bumps = torch.exp(-((x[..., None] - self.centres) / self.h) ** 2)   # (..., d_in, grid)
        return self.base(F.silu(x)) + torch.einsum("...ig,uig->...u", bumps, self.coef)`,
    hopfield: String.raw`class Hopfield(nn.Module):
    """A modern Hopfield memory: M learned patterns. The input is compared with each (a dot product, times beta), a
    softmax makes the similarities weights, and out comes their weighted mix of the patterns."""
    def __init__(self, d, memories, beta):
        super().__init__()
        self.patterns, self.beta = nn.Parameter(torch.randn(memories, d) / math.sqrt(d)), beta

    def forward(self, x):
        return (self.beta * x @ self.patterns.T).softmax(-1) @ self.patterns`,
    graphconv: String.raw`class GraphConv(nn.Module):
    """Message passing: each node's new features are one linear layer (the same for every node) of the mean, or the
    sum, of its own features and its neighbours'. Each row of x is a node: its features, then its links to the n
    nodes (1 where linked), passed on unchanged so another round can follow."""
    def __init__(self, f_in, n, units, act, agg="mean"):
        super().__init__()
        self.n, self.lin, self.act, self.agg = n, nn.Linear(f_in, units), act, agg

    def forward(self, x):
        h, links = x[..., :-self.n], x[..., -self.n:]
        a = ((links > 0.5).float() + torch.eye(self.n)).clamp(max=1)   # each node and the nodes linked to it
        m = a @ h
        if self.agg == "mean":
            m = m / a.sum(-1, keepdim=True)
        return torch.cat([self.act(self.lin(m)), links], -1)`,
  };
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
      else if (b.type === 'sepconv') {
        helpers.add('sepconv');
        init.push(`self.sep${n} = SeparableConv(${si[0]}, ${c.filters}, ${c.k}${(c.stride || 1) > 1 ? `, stride=${c.stride}` : ''}${(c.dilation || 1) > 1 ? `, dilation=${c.dilation}` : ''})`);
        put(`x = ${ACT_PY[c.act](`self.sep${n}(x)`)}`);
      } else if (b.type === 'se') {
        // the bottleneck: channels / ratio, rounded half up (a name inside one of your parts: the same sum in Python)
        const C = si.length === 3 ? si[0] : last, h = typeof C === 'number' ? NA.bx.seHidden(C, c.ratio) : `max(1, (2 * ${wrap(C)} + ${c.ratio}) // ${2 * c.ratio})`;
        helpers.add('se'); init.push(`self.se${n} = SqueezeExcite(${C}, ${h})`); put(`x = self.se${n}(x)`);
      } else if (b.type === 'coords') {
        helpers.add('coords');
        init.push(`self.register_buffer("xy${n}", coord_channels(${si[1]}, ${si[2]}, ${c.radius ? 'True' : 'False'}))   # never trained`);
        put(`x = torch.cat([x, self.xy${n}.expand(len(x), -1, -1, -1)], 1)`);
      } else if (b.type === 'augment') { helpers.add('augment'); init.push(`self.aug${n} = RandomShift(${c.shift}${c.flip ? ', flip=True' : ''})`); put(`x = self.aug${n}(x)`); }
      else if (b.type === 'gmp') put(si.length === 3 ? 'x = x.amax((2, 3))' : 'x = x.amax(1)');
      else if (b.type === 'attnpool') { helpers.add('attnpool'); init.push(`self.apool${n} = AttentionPool(${si.length === 3 ? si[0] : last})`); put(`x = self.apool${n}(x)`); }
      else if (b.type === 'ssm') { helpers.add('ssm'); init.push(`self.ssm${n} = StateSpace(${last}, ${c.state}, ${c.units})`); put(`x = ${ACT_PY[c.act](`self.ssm${n}(x)`)}`); }
      else if (b.type === 'rbf') { helpers.add('rbf'); init.push(`self.rbf${n} = RBF(${last}, ${c.units}, ${flt(c.width)})`); put(`x = self.rbf${n}(x)`); }
      else if (b.type === 'maxout') { helpers.add('maxout'); init.push(`self.maxout${n} = Maxout(${last}, ${c.units}, ${c.pieces})`); put(`x = self.maxout${n}(x)`); }
      else if (b.type === 'upsample') put(`x = x.repeat_interleave(${c.size}, 2).repeat_interleave(${c.size}, 3)   # each pixel ${c.size} × ${c.size}`);
      else if (b.type === 'sample') { helpers.add('sample'); init.push(`self.sample${n} = Sample(beta=${flt(c.beta)})`); put(`x = self.sample${n}(x)`); }
      else if (b.type === 'graphconv') {
        const fIn = typeof si[1] === 'number' && typeof si[0] === 'number' ? si[1] - si[0] : `${si[1]} - ${si[0]}`;
        helpers.add('graphconv'); init.push(`self.gc${n} = GraphConv(${fIn}, ${si[0]}, ${c.units}, ${ACT_FN[c.act] || 'F.relu'}${c.agg === 'sum' ? ', agg="sum"' : ''})`); put(`x = self.gc${n}(x)`);
      }
      else if (b.type === 'skip') put(c.merge === 'concat' ? `x = torch.cat([x, h${n - 1 - c.back}], ${si.length === 3 ? 1 : -1})   # joined with what was kept` : `x = x + h${n - 1 - c.back}                                  # added to what was kept`);
      else if (b.type === 'kan') { helpers.add('kan'); init.push(`self.kan${n} = KAN(${last}, ${c.units}, ${c.grid})`); put(`x = self.kan${n}(x)`); }
      else if (b.type === 'hopfield') { helpers.add('hopfield'); init.push(`self.hop${n} = Hopfield(${last}, ${c.memories}, ${flt(c.beta)})`); put(`x = self.hop${n}(x)`); }
      else if (b.type === 'moe') { helpers.add('moe'); init.push(`self.moe${n} = MixtureOfExperts(${last}, ${c.units}, ${c.experts}, top=${c.top}, act=${ACT_FN[c.act] || 'F.relu'})`); put(`x = self.moe${n}(x)`); }
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
  const PY_TAKEN = new Set(['Model', 'SelfAttention', 'TransformerBlock', 'Gated', 'GaussianNoise', 'MixedActivation', 'None', 'True', 'False',
    'SeparableConv', 'SqueezeExcite', 'RandomShift', 'AttentionPool', 'StateSpace', 'RBF', 'Maxout', 'MixtureOfExperts', 'Sample', 'GraphConv', 'KAN', 'Hopfield']);
  const STEMS = new Set(['dense', 'act', 'drop', 'norm', 'attn', 'block', 'conv', 'glu', 'bn', 'noise', 'rnn', 'pe', 'pos', 'patch', 'b', 'tok', 'out', 'sep', 'se', 'xy', 'aug', 'apool', 'ssm', 'rbf', 'maxout', 'moe', 'sample', 'gc', 'kan', 'hop']);
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
      case 'sepconv': return [c.filters, SX.ceil(s[1], c.stride || 1), SX.ceil(s[2], c.stride || 1)];
      case 'coords': return [SX.add(s[0], c.radius ? 3 : 2), s[1], s[2]];
      case 'gmp': case 'attnpool': return k === 3 ? [s[0]] : [s[1]];
      case 'ssm': return [s[0], c.units];
      case 'rbf': case 'maxout': case 'moe': return k === 1 ? [c.units] : [s[0], c.units];
      case 'upsample': return [s[0], SX.mul(s[1], c.size), SX.mul(s[2], c.size)];
      case 'sample': return [SX.div(s[0], 2)];
      case 'graphconv': return [s[0], SX.add(c.units, s[0])];
      case 'kan': return k === 1 ? [c.units] : [s[0], c.units];
      case 'hopfield': case 'skip': return s.slice();
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
  const HELPER_NAME = { mixed: 'MixedActivation', attn: 'SelfAttention', block: 'TransformerBlock', gated: 'Gated', noise: 'GaussianNoise', sine: 'sine_positions',
    sepconv: 'SeparableConv', se: 'SqueezeExcite', coords: 'coord_channels', augment: 'RandomShift', attnpool: 'AttentionPool', ssm: 'StateSpace', rbf: 'RBF', maxout: 'Maxout', moe: 'MixtureOfExperts', sample: 'Sample', graphconv: 'GraphConv', kan: 'KAN', hopfield: 'Hopfield' };
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
    const keep = new Set(blocks.flatMap((b, j) => (b.type === 'skip' ? [j - 1 - cfg(b).back] : [])));
    const kept = (j) => { if (keep.has(j)) fwd.push(`h${j + 1} = x`.padEnd(54) + '# kept for a skip connection'); };
    kept(-1);
    for (let i = 0; i < blocks.length;) {
      const run = names.runs.find((r) => r.from === i);
      if (run) {
        // a copy of one of your parts: one instance of its class
        const g = parts.byGrp[run.grp];
        init.push(g.make);
        fwd.push(`x = self.${g.inst}(x)`.padEnd(54) + `# ${shp(g.out)}`);
        kept(run.to - 1);
        i = run.to;
        continue;
      }
      const sh = chk.shapes[i] || {}, e = emitBlock(blocks[i], sh.in || [], sh.out || [], i + 1, helpers, { name: 'Parallel', sink: parSink });
      init.push(...e.init); fwd.push(...e.fwd);
      kept(i);
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
    for (const h of Object.keys(MORE_PY)) if (helpers.has(h)) pre.push(MORE_PY[h], '', '');
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
    const own = NA.builderPyGen && NA.builderPyGen.loss(setup, d, PY._i);
    if (own) return own;
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
      // the newest parts; RBF centres and widths, and the state-space memories, start as their classes set them, as on the page
      else if (b.type === 'sepconv') out.push(call(`${at('sep')}.depthwise`, 1), call(`${at('sep')}.pointwise`, gain(HE(cfg(b).act))));
      else if (b.type === 'se') out.push(call(`${at('se')}.squeeze`, gain(true)), call(`${at('se')}.excite`, 1));
      else if (b.type === 'maxout') out.push(call(`${at('maxout')}.lin`, gain(false)));
      else if (b.type === 'graphconv') out.push(call(`${at('gc')}.lin`, gain(HE(cfg(b).act))));
      else if (b.type === 'kan') out.push(call(`${at('kan')}.base`, 1));
      else if (b.type === 'moe') out.push(`for expert in ${at('moe')}.experts:`, `    ${call('expert', gain(HE(cfg(b).act)))}`, call(`${at('moe')}.gate`, 1));
      else if (b.type === 'attnpool' || b.type === 'ssm') {
        const w = b.type === 'ssm' ? `${at('ssm')}.B.weight` : `${at('apool')}.query.weight`;                 // no bias to zero
        out.push(`nn.init.normal_(${w}, 0.0, ${small ? '0.02' : `math.sqrt(1 / ${w}.shape[1])`})`);
        if (b.type === 'ssm') out.push(call(`${at('ssm')}.C`, gain(HE(cfg(b).act))));
      }
      else if (b.type === 'attention') out.push(attn(at('attn')), '    init_(layer)');
      else if (b.type === 'transformer') out.push(attn(`${at('block')}.attn`), '    init_(layer)', call(`${at('block')}.mlp[0]`, gain(true)), call(`${at('block')}.mlp[2]`, mode === 'he' ? 2 : 1));
    };
    setup.blocks.forEach((b, i) => initBlock(b, (stem) => `${where[i].pre}${stem}${where[i].n}`));
    out.push('init_(model.out)');
    return out;
  }
  function scoreCode(d, setup) {
    const own = NA.builderPyGen && NA.builderPyGen.score(d, setup);
    if (own) return own;
    if (d.output.kind === 'value') return ['def score(out, y):                            # R²: 1 is perfect, 0 is no better than the mean', '    return 1 - ((out[:, 0] - y) ** 2).sum() / ((y - y.mean()) ** 2).sum()'];
    if (d.output.kind === 'tokens') return ['def score(out, y):                            # share of target tokens predicted right', '    mask = y != -1', '    return (out.argmax(-1)[mask] == y[mask]).float().mean()'];
    return ['def score(out, y):                            # accuracy', '    return (out.argmax(1) == y).float().mean()'];
  }
  function afterCode(setup, d) {
    const p = BT.byId[setup.problem];
    if (NA.builderPyGen && NA.builderPyGen.owns(d)) return NA.builderPyGen.after(setup, d, Object.assign({ shapes: NA.bx.check(setup.blocks, d.input, d.output).shapes }, PY._i));
    if (d.family !== 'sequence') return [];
    if (p.task === 'language' || p.task === 'sentences') {
      const prompt = p.task === 'language' ? 'alice likes tea . bob likes milk . alice drinks' : BT.wordsOf(setup.ps[setup.problem].text).slice(0, 3).join(' ');
      return ['', '# ---------------------------------------------------------------- write with it', 'model.eval()', 'def write(prompt, count=12):', '    ids = [ID[w] for w in re.sub(r"([.,!?;:])", r" \\1 ", prompt.lower()).split() if w in ID]', '    with torch.no_grad():',
        '        for _ in range(count):', '            ctx = ids[-T:]', '            x = torch.tensor([ctx[0]] * (T - len(ctx)) + ctx)[None]', '            ids.append(model(x)[0, -1].argmax().item())     # the likeliest next word', '    return " ".join(WORDS[i] for i in ids)', '',
        `print(write(${JSON.stringify(prompt)}))`];
    }
    // adding: the question a + b and the arrow, then the sum written a digit at a time, each guess fed back in
    if (p.task === 'add') return ['', '# ---------------------------------------------------------------- use it: add two numbers, a digit at a time', 'model.eval()', 'def add(a, b):',
      '    q = [int(c) for c in str(a).zfill(D)] + [11] + [int(c) for c in str(b).zfill(D)] + [10]', '    x = torch.tensor(q + [0] * D)[None]          # the question, the arrow, and room for the sum',
      '    out = []', '    with torch.no_grad():', '        for k in range(D + 1):', '            d = model(x)[0, 2 * D + 1 + k].argmax().item()', '            out.append(d)', '            if k < D:', '                x[0, 2 * D + 2 + k] = d         # feed the guess back in', '    return int("".join(map(str, out)))', '',
      `print(${'4'.repeat(setup.ps[setup.problem].digits || 3)}, "+", ${'8'.repeat(setup.ps[setup.problem].digits || 3)}, "->", add(${'4'.repeat(setup.ps[setup.problem].digits || 3)}, ${'8'.repeat(setup.ps[setup.problem].digits || 3)}))`];
    return ['', '# ---------------------------------------------------------------- use it: write the answer one digit at a time', 'model.eval()', 'def answer(digits):', '    x = torch.tensor(digits + [10] + [0] * (L - 1))[None]',
      '    out = []', '    with torch.no_grad():', '        for k in range(L):', '            d = model(x)[0, L + k].argmax().item()', '            out.append(d)', '            if L + k + 1 < 2 * L:', '                x[0, L + k + 1] = d                  # feed the guess back in', '    return out', '',
      `print([3, 1, 4, 1, 5, 9][:L], "->", answer([3, 1, 4, 1, 5, 9][:L]))`];
  }

  // A one-line description of the model, for the comment at the top
  function summary(setup, d) {
    const L = NA.builderLab && NA.builderLab.blockLabel ? NA.builderLab.blockLabel : (b) => b.type;
    // (learning from rewards: a Q-value per move for DQN, move scores and a value for actor-critic and PPO, move scores otherwise)
    const o = d.output, moves = d.family === 'control' ? (d.method === 'dqn' ? `${o.k} Q-values, one per move` : o.k > d.nA ? `${d.nA} move scores and a value` : `${o.k} move scores`) : null;
    const end = o.kind === 'classes' ? `${o.k} class scores` : o.kind === 'value' ? moves || (o.k > 1 ? `${o.k} numbers` : '1 number') : `${o.vocab} next-token scores`;
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
  /* The extra inputs a world can be given (rl-envs.js E.EXTRAS), as Python: name → (the page's world, data) →
     [[expression on the World, what it is], ...], one per input, in the page's order. Its last move needs the world
     to note it (lastMoveWrap). */
  const PY_EXTRAS = {
    common: {
      'its last move': (W, d) => [[`*(float(self.last_move == k) for k in range(${d.nA}))`, 'its last move, one input per move (all 0 at the start)']],
      'time left': (W) => [[`1 - self.t / ${Number.isFinite(W.maxSteps) ? W.maxSteps : 'self.max_steps'}`, 'time left: 1 at the start, 0 at the end']],   // (the track knows its length only once it has one)
      'a random number': () => [['random.uniform(-1, 1)', 'a random number, which tells it nothing']],
    },
    thermo: {
      'the target in an hour': () => [['(self.target(((self.t + 6) / 6) % 24) - 18) / 5', 'the target in an hour']],
      'the weather in an hour': () => [['(self.outside(self.t + 6) - 10) / 10', 'the weather in an hour']],
    },
    hurdles: {
      'how far it has run': () => [['min(max(self.x / COURSE, 0.0), 1.0)', 'how much of the course it has run']],
      'the one after: a bar': () => [['float((self.obst[self.next_of(self.x) + 1:] + [(0.0, False)])[0][1])', 'the obstacle after the next: a bar?']],
    },
    memory: {
      'what the lamp said': () => [['float(self.side == 0)', 'the lamp said left'], ['float(self.side == 1)', 'the lamp said right']],
      'steps waited': () => [['self.waits / WAIT', 'the steps it has stood at the junction, of the WAIT it has']],
    },
    mountain: {
      'its height': () => [['math.sin(3 * self.x)', 'its height, -1 to 1']],
      'its energy': () => [['self.energy() / 1.7', 'its energy: 0 at rest in the bottom, about 1 at rest by the flag']],
    },
    swingup: {
      'its energy': () => [['self.energy()', 'its swing: 1 with just enough to reach the top, -1 hanging still']],
      'how far from upright': () => [['abs(self.th) / math.pi', 'its angle from upright, 0 up to 1 hanging']],
    },
    arm: {
      'the hand': () => [['self.hx / 2', 'the hand across'], ['self.hy / 2', 'the hand up']],
      'distance to the target': () => [['self.d / 2', 'how far the hand is from the target']],
    },
    trap: {
      'where it is': () => [['self.x / 200', 'its place across'], ['self.y / 200', 'its place down']],
      'its heading': () => [['math.sin(self.th)', 'facing (sine)'], ['math.cos(self.th)', 'facing (cosine)']],
      'distance to the goal': () => [['self.d / 200', 'how far the goal is in a straight line']],
    },
    chase: {
      'the nearest chaser’s distance': () => [['self.nearest()[1] / 14', 'how far the nearest chaser is, 1 at 14 m']],
      'where it is': () => [['self.me.x / SIZE', 'its place across'], ['1 - self.me.y / SIZE', 'its place up']],
    },
    traffic: {
      'cars by road': () => [['min(len(self.q[0]) + len(self.q[1]), 60) / 30', 'cars waiting north-south'], ['min(len(self.q[2]) + len(self.q[3]), 60) / 30', 'cars waiting east-west']],
    },
    walker: {
      'its feet': () => [['self.p[2][0] - self.cx()', 'the back foot from the body, m'], ['self.p[3][0] - self.cx()', 'the front foot from the body, m']],
      'how far it has walked': () => [['(self.cx() - self.x0) / 10', 'how far it has walked, a tenth for each metre']],
    },
    track: {
      'where it is across': () => [['min(max(self.track.side(self.x, self.y, self.k) / HW, -1.0), 1.0)', 'where it is across the track: -1 at the left wall, 1 at the right']],
      'its heading': () => [['self.heading() / (math.pi / 2)', 'its heading to the track: 1 is a quarter turn to the right']],
      'the bends ahead': () => [['self.bend(10) * 7', 'how sharply the track bends 10 m ahead'], ['self.bend(30) * 7', 'and 30 m ahead']],
    },
    arena: {
      'the nearest drone’s distance': () => [['min([math.hypot(d.x - self.x, d.y - self.y) for d in self.drones if d.alive] + [12.0]) / 12', 'how far the nearest drone is, 1 at 12 m or more']],
      'where it is': () => [['self.x / MW', 'its place across'], ['self.y / MH', 'its place down'], ['math.sin(self.th)', 'facing (sine)'], ['math.cos(self.th)', 'facing (cosine)']],
      'drones destroyed': () => [['self.kills / GOAL', 'the share of the drones to destroy it has destroyed']],
    },
    lander: {
      'the wind': () => [['self.wind() / 2', 'the wind\'s push now, sideways (0 when calm)']],
      'height above the ground': () => [['(self.y - self.ground(self.x)) / 30', 'its height above the ground right below it, 1 at 30 m']],
    },
    maze: {
      'where it is': () => [['(self.s % self.maze.W + 0.5) / self.maze.W', 'its square across'], ['(self.s // self.maze.W + 0.5) / self.maze.H', 'its square down']],
      'the goal from here': () => [['(self.maze.goal % self.maze.W - self.s % self.maze.W) / self.maze.W', 'the goal across from here'], ['(self.maze.goal // self.maze.W - self.s // self.maze.W) / self.maze.H', 'the goal down from here']],
      'safe steps to the goal': () => [['self.maze.dist[self.s] / 20 if self.maze.dist[self.s] >= 0 else 1.0', 'the fewest safe steps to the goal, a twentieth each']],
    },
    catch: {
      'the ball from the paddle': () => [['(self.bx - self.mid()) / (BOARD_W - 1)', 'the ball across from the paddle, of the board']],
      'where it will land': () => [['self.landing() / (BOARD_W - 1)', 'where the ball will land, across']],
    },
    herd: {
      'the nearest sheep': () => [['*self.nearest_sheep()', 'the nearest sheep still out, from the dog: across, up']],
      'the gate from the dog': () => [['(GATE[0] - self.dog.x) / 12', 'the gate across from the dog'], ['(self.dog.y - GATE[1]) / 12', 'the gate up from the dog']],
    },
  };
  PY.EXTRAS = PY_EXTRAS;
  // its last move, noted before every step (for that extra input): the World above, remembering it
  const lastMoveWrap = (d) => ((d.extra || []).includes('its last move') ? ['', '', 'class World(World):                               # the same world, noting its last move (an extra input)',
    '    def reset(self, *args, **kw):', '        self.last_move = -1', '        return super().reset(*args, **kw)', '', '    def step(self, a):', '        self.last_move = a', '        return super().step(a)'] : []);
  function worldPy(d) {
    const w = d.world, ws = d.ws, E = NA.rlEnvs;
    const num = (v) => { const s = String(+(+v).toPrecision(12)); return /^-?\d+$/.test(s) ? s + '.0' : s; }, say = (v) => (v > 0 ? '+' : '') + String(+(+v).toPrecision(12));
    const rw = Object.fromEntries(E.REWARDS[w].map((r) => [r.key, typeof ws[r.key] === 'number' ? ws[r.key] : r.def]));
    const own = ws.rFormula ? NA.builderTasks.compile(ws.rFormula, E.REWARD_VARS[w].map((v) => v[0]), { scalar: true }) : null, f = own && !own.error ? own : null;
    // the inputs it keeps and the noise on them: obs() becomes raw_obs(), and obs() what the model is given
    const keep = d.keep, noise = ws.noise || 0;
    const consts = [...(keep ? [`    KEEP = [${keep.join(', ')}]${' '.repeat(Math.max(1, 34 - keep.join(', ').length))}# the inputs it is given (not: ${(d.hidden || []).join(', ')})`] : []),
      ...(noise ? [`    NOISE = ${num(noise)}                                # sensor noise: a random amount on every input, every step`] : [])];
    // the extra inputs it is given (E.EXTRAS): after the ones it keeps, each a Python expression on the world (PY_EXTRAS)
    const extra = d.extra || [], WX = extra.length ? E.make(w, ws) : null;
    const extraLines = extra.flatMap((name) => {
      const fn = (PY_EXTRAS[w] || {})[name] || PY_EXTRAS.common[name];
      if (!fn) throw new Error(`The script cannot write the extra input "${name}" yet.`);
      return fn(WX, d).map(([code, what]) => `            ${code},${' '.repeat(Math.max(1, 44 - code.length))}# ${what || name}`);
    });
    const sense = (lines) => (!keep && !noise && !extra.length ? lines : [lines[0].replace('def obs(self):', 'def raw_obs(self):'), ...lines.slice(1), '',
      '    def obs(self):                                # what the model is given', `        v = self.raw_obs()${keep ? '[self.KEEP]' : ''}`,
      ...(extra.length ? ['        v = torch.cat([v, torch.tensor(self.extras(), dtype=torch.float32)])   # and the extra inputs'] : []),
      noise ? '        return v + self.NOISE * torch.randn_like(v)' : '        return v',
      ...(extra.length ? ['', `    def extras(self):                             # the extra inputs: ${extra.join(', ')}`, '        return [', ...extraLines, '        ]'] : [])]);
    const head = (lines) => [...lines, ...consts];
    const ownLine = f ? [`        r = ${f.py}`] : null;
    // the reward table's terms, added in the page's order (none: 0)
    const terms = (list) => list.filter(Boolean).join(' + ') || '0.0';
    // a line of code with its comment, the comments lined up
    const note = (code, text) => (code.length < 50 ? code.padEnd(51) : code + '   ') + '# ' + text;
    const pays = (parts) => (f ? `Its reward is your own: r = ${f.py}` : parts);
    // the race track, the drone arena and the rocket: builder-py-worlds.js
    const PW = NA.builderPyWorlds;
    if (PW && PW.worlds[w]) return PW.worlds[w](d, { num, say, rw, f, sense, head, ownLine, pays, SEEDED, PLAY });
    if (w === 'levers') {
      const k = ws.levers || 3, p = typeof ws.p === 'number' ? ws.p : 0.8, q = typeof ws.q === 'number' ? ws.q : 0.2, pct = (v) => `${Math.round(v * 100)}%`;
      const paysText = `A lever that pays gives ${say(rw.rPay)}${rw.rNoPay ? `, one that does not ${say(rw.rNoPay)}` : ''}${rw.rMatch ? `; the lever of the light's colour ${say(rw.rMatch)} more, paid or not` : ''}.`;
      return [...head(['class World:', `    """A light shows one of ${k} colours. The lever of that colour pays ${pct(p)} of the time, any other ${pct(q)}. One`, `    pull per episode. ${pays(paysText)}"""`,
        `    actions = [${d.actions.map((a) => JSON.stringify(a)).join(', ')}]`]), '',
        '    def reset(self):', `        self.light = random.randrange(${k})`, '        return self.obs()', '',
        ...sense(['    def obs(self):                                # one input per colour: 1 for the light that is on', `        return F.one_hot(torch.tensor(self.light), ${k}).float()`]), '',
        '    def step(self, a):                            # -> (observation, reward, done)', `        paid = random.random() < (${num(p)} if a == self.light else ${num(q)})`,
        ...(f ? ['        paid, right, unpaid, lever, light = float(paid), float(a == self.light), float(not paid), a, self.light', ...ownLine, '        return self.obs(), r, True']
          : !rw.rNoPay && !rw.rMatch ? [`        return self.obs(), ${rw.rPay === 1 ? 'float(paid)' : `${num(rw.rPay)} if paid else 0.0`}, True`]
            : [`        r = ${terms([rw.rPay ? `(${num(rw.rPay)} if paid else 0.0)` : null, rw.rNoPay ? `(${num(rw.rNoPay)} if not paid else 0.0)` : null, rw.rMatch ? `(${num(rw.rMatch)} if a == self.light else 0.0)` : null])}`, '        return self.obs(), r, True']), '',
        '    def score(self, policy):', '        """The share of lights for which its best lever is the right one."""', '        right = 0', `        for light in range(${k}):`, '            self.light = light', '            right += policy(self.obs()) == light', `        return right / ${k}`];
    }
    if (w === 'pole') {
      const W = E.make('pole', ws), rd = W.view === 'readings', steps = W.maxSteps, drop = (v) => num(v).replace(/\.0$/, '');
      const reward = f ? ['        x, v, angle, spin, t = self.x, self.v, math.degrees(self.th), math.degrees(self.w), self.t',
        '        fell, switched, lasted, left = float(fell), float(switched), float(lasted), self.max_steps - self.t', ...ownLine]
        : [`        r = ${terms([rw.rUp ? num(rw.rUp) : null, rw.rFell ? `(${num(rw.rFell)} if fell else 0.0)` : null, rw.rLean ? `${num(rw.rLean)} * abs(math.degrees(self.th))` : null,
          rw.rOff ? `${num(rw.rOff)} * abs(self.x)` : null, rw.rSwitch ? `(${num(rw.rSwitch)} if switched else 0.0)` : null, rw.rLasted ? `(${num(rw.rLasted)} if lasted else 0.0)` : null])}`];
      const paysText = `${say(rw.rUp)} for every step${rw.rFell ? `, ${say(rw.rFell)} when it falls` : ' it stays up'}${rw.rLean ? `, ${say(rw.rLean)} × the pole's lean (degrees) every step` : ''}${rw.rOff ? `, ${say(rw.rOff)} × the cart's distance from the middle (m) every step` : ''}${rw.rSwitch ? `, ${say(rw.rSwitch)} each time the push changes direction` : ''}${rw.rLasted ? `, ${say(rw.rLasted)} for lasting the whole episode` : ''}`;
      return [note(`EDGE, LIMIT = ${num(W.edge)}, ${num(W.limit)}`, `the ends of the track (±${drop(W.edge)} m) and the lean it falls past (${+(W.limit * 180 / Math.PI).toFixed(1)}°, in rad)`),
        note(`START, X_START = ${num(W.start)}, ${num(W.xStart)}`, 'how far from upright and still it may start; the cart, from the middle'), '', '',
        ...head(['class World:', '    """A pole hinged on a cart that can be pushed left or right (Barto, Sutton and Anderson, 1983). It falls when it',
        `    leans more than ${+(W.limit * 180 / Math.PI).toFixed(1)} degrees or the cart runs off its track. ${pays(paysText)}; at most ${steps} steps."""`,
        '    actions = ["push left", "push right"]', `    max_steps = ${steps}`]), '',
        '    def reset(self):', '        self.x = random.uniform(-X_START, X_START)', '        self.v, self.th, self.w = (random.uniform(-START, START) for _ in range(3))', '        self.t, self.pushed = 0, 0',
        d.s.curriculum === 'on' ? '        self.g_scale = 0.35 + 0.65 * getattr(self, "level", 1.0)   # (easy first: weaker gravity at first)' : null,
        rd ? '        self.hist = [(self.x / EDGE, self.th / LIMIT)] * 3     # the readings before this one' : null, '        return self.obs()', '',
        ...sense([rd ? '    def obs(self):                                # (4, 2): position and angle, the last 4 steps, oldest first; no speeds' : '    def obs(self):                                # position, speed, angle, turning speed, scaled to about ±1',
          rd ? '        return torch.tensor(self.hist + [(self.x / EDGE, self.th / LIMIT)])' : '        return torch.tensor([self.x / EDGE, self.v / 3, self.th / LIMIT, self.w / 3.5])']), '',
        `    def step(self, a):                            # one 0.02 s step: gravity ${drop(W.g)}, cart 1 kg, pole 0.1 kg and ${drop(W.len)} m, a push of ${drop(W.force)} N`,
        `        force = ${num(W.force)} if a == 1 else -${num(W.force)}`, '        ct, st = math.cos(self.th), math.sin(self.th)', `        tmp = (force + ${num(W.pml)} * self.w * self.w * st) / 1.1`,
        `        acc = (${num(W.g)}${d.s.curriculum === 'on' ? ' * self.g_scale' : ''} * st - ct * tmp) / (${num(W.half)} * (4 / 3 - 0.1 * ct * ct / 1.1))`, `        xacc = tmp - ${num(W.pml)} * acc * ct / 1.1`,
        rd ? '        self.hist = self.hist[1:] + [(self.x / EDGE, self.th / LIMIT)]' : null,
        '        self.x, self.v, self.th, self.w = self.x + 0.02 * self.v, self.v + 0.02 * xacc, self.th + 0.02 * self.w, self.w + 0.02 * acc', '        self.t += 1',
        '        fell = abs(self.x) > EDGE or abs(self.th) > LIMIT', '        push = 1 if a == 1 else -1',
        '        switched = self.pushed != 0 and push != self.pushed   # a push the other way from the last one', '        self.pushed = push',
        '        lasted = not fell and self.t >= self.max_steps   # the whole episode, never falling', ...reward, '        return self.obs(), r, bool(fell) or self.t >= self.max_steps', '',
        '    def score(self, policy):', `        """How long its best moves keep the pole up, as a share of ${steps} steps, over 5 starts."""`, '        total = 0',
        ...SEEDED(4242, ['for _ in range(5):', ...PLAY('    '), '    total += self.t']).map((l) => '        ' + l), '        return total / (5 * self.max_steps)'].filter((l) => l !== null);
    }
    if (w === 'catch') {
      const W = E.make('catch', ws), view = W.view, drift = W.drift, pw = W.pw, land = !!f || extra.includes('where it will land');
      const reward = f ? ['        caught, missed, moved = float(end and self.caught), float(end and not self.caught), float(moved)',
        '        dx, down, t, near, under = self.bx - self.mid(), self.by, self.t, float(near), float(under)', '        bx, px, land, drift = self.bx, self.mid(), self.landing(), self.vx', ...ownLine]
        : [`        r = ${terms([rw.rMove ? `(${num(rw.rMove)} if moved else 0.0)` : null, rw.rCatch || rw.rMiss ? `((${num(rw.rCatch)} if self.caught else ${num(rw.rMiss)}) if end else 0.0)` : null,
          rw.rNear ? `(${num(rw.rNear)} if near else 0.0)` : null, rw.rUnder ? `(${num(rw.rUnder)} if under else 0.0)` : null, rw.rStep ? num(rw.rStep) : null])}`];
      const paysText = `${say(rw.rCatch)} if the ball lands on the paddle, ${say(rw.rMiss)} if it misses${rw.rMove ? `, ${say(rw.rMove)} for every move` : ''}${rw.rNear ? `, ${say(rw.rNear)} for a near miss (right beside the paddle)` : ''}${rw.rUnder ? `, ${say(rw.rUnder)} every step the paddle is under the ball` : ''}${rw.rStep ? `, ${say(rw.rStep)} every step` : ''}.`;
      return [note(`BOARD_W, BOARD_H, PADDLE = ${W.W}, ${W.H}, ${pw}`, "the board, squares across and down; the paddle's width"), '', '',
        ...head(['class World:', `    """A ball falls down a ${W.W} x ${W.H} board, one row a step${drift ? ', drifting sideways and bouncing off the walls' : ', straight down'}; a paddle${pw > 1 ? ` ${pw} squares wide` : ''} on the`,
        `    bottom row moves left, stays or moves right. ${pays(paysText)}"""`, '    actions = ["left", "stay", "right"]']), '',
        '    def reset(self):', `        self.bx, self.by, self.vx = random.randrange(BOARD_W), 0, ${drift ? 'random.randrange(3) - 1' : '0'}`, '        self.px = (BOARD_W - PADDLE) // 2           # the paddle\'s left end',
        '        self.prev, self.caught, self.t = None, None, 0', '        return self.obs()', '',
        '    def mid(self):                                # the paddle\'s middle, across', '        return self.px + (PADDLE - 1) / 2', '',
        '    def over(self, x):                            # a ball across at x is over the paddle', '        return self.px <= x < self.px + PADDLE', '',
        ...(land ? ['    def landing(self):                            # where the ball will land, across: its drift and its bounces played forward', '        x, v = self.bx, self.vx', '        for _ in range(self.by, BOARD_H - 1):', '            x += v',
          '            if x < 0:', '                x, v = 1, 1', '            elif x > BOARD_W - 1:', '                x, v = BOARD_W - 2, -1', '        return x', ''] : []),
        ...(view !== 'numbers' ? ['    def picture(self, bx, by, px):', '        p = torch.zeros(BOARD_H, BOARD_W)', '        p[by, bx] = 1.0                          # the ball', '        p[BOARD_H - 1, px:px + PADDLE] = 1.0     # the paddle', '        return p', '',
          ...sense(view === 'pixels' ? [`    def obs(self):                                # (2, ${W.H}, ${W.W}): the board now and a step ago`, '        return torch.stack([self.picture(self.bx, self.by, self.px), self.picture(*(self.prev or (self.bx, self.by, self.px)))])']
            : [`    def obs(self):                                # (1, ${W.H}, ${W.W}): the board now (which way the ball drifts cannot be told from it)`, '        return self.picture(self.bx, self.by, self.px)[None]'])]
          : sense(['    def obs(self):                                # where the ball is, which way it drifts, where the paddle is', '        return torch.tensor([self.bx / (BOARD_W - 1), self.by / (BOARD_H - 1), float(self.vx), self.mid() / (BOARD_W - 1)])'])), '',
        '    def step(self, a):', '        self.prev = (self.bx, self.by, self.px)', '        self.px = min(max(self.px + a - 1, 0), BOARD_W - PADDLE)', '        self.by, self.bx = self.by + 1, self.bx + self.vx',
        '        if self.bx < 0:                           # a wall turns it back', '            self.bx, self.vx = 1, 1', '        elif self.bx > BOARD_W - 1:', '            self.bx, self.vx = BOARD_W - 2, -1',
        '        self.t += 1', '        moved, end = self.px != self.prev[2], self.by >= BOARD_H - 1', '        if end:', '            self.caught = self.over(self.bx)',
        '        under = self.over(self.bx)                # the paddle under the ball', '        near = end and not self.caught and self.bx in (self.px - 1, self.px + PADDLE)   # landing right beside the paddle',
        ...reward, '        return self.obs(), r, end', '',
        '    def score(self, policy):', '        """The share of 40 balls its best moves catch (the same 40 every time)."""', '        caught = 0',
        ...SEEDED(4242, ['for _ in range(40):', ...PLAY('    '), '    caught += self.caught']).map((l) => '        ' + l), '        return caught / 40'];
    }
    // the maze
    const def = ws.maze === 'own' && E.mazeCheck(ws.rows).ok ? { rows: ws.rows, name: 'your own maze' } : NA.rl.MAZE_SET.find((m) => m.id === ws.maze) || NA.rl.MAZE_SET[0], view = ws.view, rnd = ws.random, slip = ws.slip || 0, most = E.make('maze', ws).maxSteps;
    const obs = view === 'square' ? ['    def obs(self):                                # which square it is on: one input per square', '        return F.one_hot(torch.tensor(self.s), len(self.maze.cell)).float()']
      : view === 'picture' ? ['    def obs(self):                                # (5, 6, 8): walls, pits, the goal, where it is, where it has been', '        m = self.maze', '        p = torch.zeros(5, m.H, m.W)',
        '        for i, c in enumerate(m.cell):', '            y, x = divmod(i, m.W)', '            if c in "#XG":', '                p["#XG".index(c), y, x] = 1.0', '            if i in self.seen:', '                p[4, y, x] = 1.0', '        p[3, self.s // m.W, self.s % m.W] = 1.0', '        return p']
        : ['    def obs(self):', '        """The 24 squares around it, 2 in every direction (each: a wall or the edge, a pit, the goal, a square it has', '        been on), and which way the goal is: 104 inputs, each 0 or 1."""',
          '        m, x0, y0 = self.maze, self.s % self.maze.W, self.s // self.maze.W', '        v, k = torch.zeros(104), 0', '        for dy in range(-2, 3):', '            for dx in range(-2, 3):', '                if dx == 0 and dy == 0:', '                    continue',
          '                x, y = x0 + dx, y0 + dy', '                c = m.cell[y * m.W + x] if 0 <= x < m.W and 0 <= y < m.H else "#"', '                if c in "#XG":', '                    v["#XG".index(c) * 24 + k] = 1.0', '                elif y * m.W + x in self.seen:', '                    v[72 + k] = 1.0', '                k += 1',
          '        gx, gy = m.goal % m.W - x0, m.goal // m.W - y0', '        if gx > 0:', '            v[96 + (gx > 2)] = 1.0', '        if gx < 0:', '            v[98 + (gx < -2)] = 1.0', '        if gy > 0:', '            v[100 + (gy > 2)] = 1.0', '        if gy < 0:', '            v[102 + (gy < -2)] = 1.0', '        return v'];
    // what the reward needs worked out: a square it had not been on, a turn, a pit next to it
    const fresh = !!f || !!rw.rNew || !!rw.rRevisit, turn = !!f || !!rw.rTurn, nearPit = !!f || !!rw.rNearPit;
    const reward = f ? ['        goal, pit, wall = float(c == "G"), float(c == "X"), float(self.s == s0)', '        new = float(fresh and c not in "GX")',
      '        closer = m.dist[s0] - m.dist[self.s] if c != "X" and m.dist[s0] >= 0 and m.dist[self.s] >= 0 else 0',
      '        dist, t = m.dist[self.s] if m.dist[self.s] >= 0 else max(0, m.through[self.s]), self.t   # (from a pit: the way through pits)',
      '        revisit, nearpit, lost = float(not fresh and self.s != s0), float(c not in "GX" and self.pit_near(self.s)), float(self.end == "lost")',
      '        turned, slipped, x, y, visited = float(turned), float(slipped), self.s % m.W, self.s // m.W, len(self.seen)', ...ownLine]
      : [`        r = ${num(rw.rGoal)} if c == "G" else ${num(rw.rPit)} if c == "X" else ${num(rw.rStep)}`,
        rw.rWall ? '        if self.s == s0:                          # walked into a wall' : null, rw.rWall ? `            r += ${num(rw.rWall)}` : null,
        rw.rNew ? '        if fresh and c not in "GX":               # a square it had not been on' : null, rw.rNew ? `            r += ${num(rw.rNew)}` : null,
        ws.closer ? '        if c != "X" and m.dist[s0] >= 0 and m.dist[self.s] >= 0:' : null, ws.closer ? `            r += ${num(rw.rCloser)} * (m.dist[s0] - m.dist[self.s])` : null,
        rw.rRevisit ? '        if not fresh and self.s != s0:            # back on a square it had been on' : null, rw.rRevisit ? `            r += ${num(rw.rRevisit)}` : null,
        rw.rNearPit ? '        if c not in "GX" and self.pit_near(self.s):   # a pit right next to it' : null, rw.rNearPit ? `            r += ${num(rw.rNearPit)}` : null,
        rw.rTurn ? '        if turned:' : null, rw.rTurn ? `            r += ${num(rw.rTurn)}` : null,
        rw.rLost ? '        if self.end == "lost":                    # out of steps' : null, rw.rLost ? `            r += ${num(rw.rLost)}` : null];
    const paysText = f ? `Its reward is your own: r = ${f.py}` : `${say(rw.rGoal)} at the goal, ${say(rw.rPit)} in a pit, ${say(rw.rStep)} a step${rw.rWall ? `, ${say(rw.rWall)} into a wall` : ''}${rw.rNew ? `, ${say(rw.rNew)} on a new square` : ''}${rw.rRevisit ? `, ${say(rw.rRevisit)} back on one it had been on` : ''}${rw.rNearPit ? `, ${say(rw.rNearPit)} next to a pit` : ''}${rw.rTurn ? `, ${say(rw.rTurn)} for a change of direction` : ''}${rw.rLost ? `, ${say(rw.rLost)} for running out of steps` : ''}`;
    return [`MAZE = [${def.rows.map((r) => JSON.stringify(r)).join(', ')}]   # ${def.name}`, 'DIRS = [(0, -1), (1, 0), (0, 1), (-1, 0)]               # up, right, down, left', '', '',
      'class Maze:', '    def __init__(self, rows):', '        self.H, self.W, self.cell = len(rows), len(rows[0]), "".join(rows)', '        self.start, self.goal = self.cell.index("S"), self.cell.index("G")', '        self.dist = self.distances()', f ? '        self.through = self.distances(pits=True)   # (and the way through pits)' : null, '',
      '    def move(self, s, a):                         # where move a leads from square s (into a wall or off the board: nowhere)', '        x, y = s % self.W + DIRS[a][0], s // self.W + DIRS[a][1]',
      '        if not (0 <= x < self.W and 0 <= y < self.H) or self.cell[y * self.W + x] == "#":', '            return s', '        return y * self.W + x', '',
      f ? '    def distances(self, pits=False):              # the fewest safe steps from every square to the goal (-1: none); pits: through pits too'
        : '    def distances(self):                          # the fewest safe steps from every square to the goal (-1: none)', '        d, queue = [-1] * len(self.cell), [self.goal]', '        d[self.goal] = 0', '        while queue:', '            s = queue.pop(0)', '            for a in range(4):',
      '                t = self.move(s, a)', f ? '                if d[t] < 0 and self.cell[t] != "#" and (pits or self.cell[t] != "X"):' : '                if d[t] < 0 and self.cell[t] not in "X#":', '                    d[t] = d[s] + 1', '                    queue.append(t)', '        return d', '', '',
      ...(rnd ? ['def random_maze(rng):', '    """An 8 x 6 maze: about one square in five a wall, 1 to 3 pits, S and G on two free squares, kept when the', '    shortest safe way through takes 8 steps or more."""', '    while True:',
        '        g = [["#" if rng.random() < 0.22 else "." for _ in range(8)] for _ in range(6)]', '        for _ in range(1 + int(rng.random() * 3)):', '            g[int(rng.random() * 6)][int(rng.random() * 8)] = "X"',
        '        free = [(x, y) for y in range(6) for x in range(8) if g[y][x] == "."]', '        s, t = free[int(rng.random() * len(free))], free[int(rng.random() * len(free))]', '        if s == t:', '            continue',
        '        g[s[1]][s[0]], g[t[1]][t[0]] = "S", "G"', '        m = Maze(["".join(r) for r in g])', '        if m.dist[m.start] >= 8:', '            return m', '', ''] : []),
      ...head(['class World:', `    """${paysText}, at most ${most} steps${!f && ws.closer ? `; and a bonus of ${String(+(+rw.rCloser).toPrecision(12))} for every step that brings it` : '."""'}`,
        !f && ws.closer ? '    closer to the goal by the shortest safe way, as much off for every step away (reward shaping)."""' : null, '    actions = ["up", "right", "down", "left"]']), '',
      '    def __init__(self):', '        self.fixed = self.maze = Maze(MAZE)', rnd ? '        self.tests = [random_maze(random.Random(1234 + k)) for k in range(20)]   # 20 mazes it never trains on' : null, '',
      '    def reset(self, maze=None):', `        self.maze = maze or ${rnd ? 'random_maze(random)                      # a new random maze every episode' : 'self.fixed'}`, '        self.s, self.t, self.end = self.maze.start, 0, None', '        self.seen = {self.s}', turn ? '        self.dir = -1                              # the way it last went (none yet)' : null, '        return self.obs()', '',
      ...(nearPit ? ['    def pit_near(self, s):                        # a pit on one of the four squares next to s', '        return any(q != s and self.maze.cell[q] == "X" for q in (self.maze.move(s, a) for a in range(4)))', ''] : []),
      ...sense(obs), '',
      '    def step(self, a):', slip ? `        slipped = random.random() < ${slip}          # a slippery floor: sideways instead` : f ? '        slipped = False' : null, slip ? '        if slipped:' : null, slip ? '            a = (a + random.choice((1, 3))) % 4' : null,
      turn ? '        turned = self.dir >= 0 and a != self.dir   # another way than last time (the way it really went)' : null, turn ? '        self.dir = a' : null,
      '        m, s0 = self.maze, self.s', '        self.s = m.move(self.s, a)', '        self.t += 1', fresh ? '        fresh = self.s not in self.seen' : null, '        self.seen.add(self.s)', '        c = m.cell[self.s]', `        done = c in "GX" or self.t >= ${most}`,
      '        if done:', '            self.end = "goal" if c == "G" else "pit" if c == "X" else "lost"', ...reward, '        return self.obs(), r, done', '',
      '    def score(self, policy):', rnd ? '        """The share of 20 new mazes its best moves find the goal of."""' : slip ? '        """The share of 5 tries in which its best moves reach the goal."""' : '        """1 if its best moves reach the goal, 0 if not."""',
      `        mazes, reached = ${rnd ? 'self.tests' : `[self.fixed] * ${slip ? 5 : 1}`}, 0`,
      ...SEEDED(4242, ['for m in mazes:', '    o, done = self.reset(m), False', '    while not done:', '        o, _, done = self.step(policy(o))', '    reached += self.end == "goal"']).map((l) => '        ' + l), '        return reached / len(mazes)'].filter((l) => l !== null);
  }
  // the method's training loop
  // whether the script keeps the best network: the gradient methods, unless it is set off
  const keepsBest = (d) => !NA.brl.isEvo(d.method) && d.s.keepBest !== false;
  const PREF_PY = "# ---------------------------------------------------------------- Which did better? (learning from preferences)\n# The page shows you two tries and asks which did better; here the world's own reward answers (the try with the\n# larger return did better), as in the first experiments with this method (Christiano and others, 2017). A reward\n# model learns a reward per move whose totals agree with the answers, and PPO learns from its rewards, never the world's.\nreward_model = nn.Sequential(nn.Linear(Xva.shape[1] + n_actions, 32), nn.ReLU(), nn.Linear(32, 32), nn.ReLU(), nn.Linear(32, 1))\nrm_opt = torch.optim.Adam(reward_model.parameters(), lr=0.003)\ntries, pairs = [], []                               # tries: (what it saw, its moves, the world's return); pairs: (i, j, answer)\nask_every, explore = __EVERY__, 0.25                         # a new pair every so many steps; the second try's share of random moves\n\n\ndef rm_input(o, a):                                 # what the reward model sees: the observation, and the move one-hot\n    return torch.cat([o, F.one_hot(torch.as_tensor(a), n_actions).float()], -1)\n\n\ndef r_hat(o, a):                                    # the reward model's reward for move a after seeing o\n    with torch.no_grad():\n        return reward_model(rm_input(o, a)).item()\n\n\ndef one_try(seed, eps):\n    \"\"\"A try of the current policy from the start of seed, its moves sampled, a share eps of them at random.\"\"\"\n    w, saw, moves, ret = World(), [], [], 0.0\n    state = random.getstate()\n    random.seed(seed)\n    o, done = w.reset(), False\n    while not done and len(moves) < 400:\n        a = random.randrange(n_actions) if eps and random.random() < eps else int(torch.multinomial(outputs(o[None])[0, :n_actions].softmax(0), 1))\n        o2, r, done = w.step(a)\n        saw.append(o); moves.append(a); ret += r; o = o2\n    random.setstate(state)\n    return torch.stack(saw), torch.tensor(moves), ret\n\n\ndef fit_reward_model(steps):\n    \"\"\"The Bradley-Terry model: P(first did better) = sigmoid(total of its rewards - total of the other's), a\n    logistic loss, and a little pull of every reward toward 0 (the answers fix their differences, this their size).\"\"\"\n    for _ in range(steps):\n        i, j, y = random.choice(pairs)\n        (oa, ma, _), (ob, mb, _) = tries[i], tries[j]\n        ra, rb = reward_model(rm_input(oa, ma))[:, 0], reward_model(rm_input(ob, mb))[:, 0]\n        z = (ra.sum() - rb.sum()).clamp(-30, 30)\n        loss = F.binary_cross_entropy_with_logits(z, torch.tensor(y)) + 0.0005 * (ra.pow(2).sum() + rb.pow(2).sum())\n        rm_opt.zero_grad()\n        loss.backward()\n        torch.nn.utils.clip_grad_norm_(reward_model.parameters(), 1.0)\n        rm_opt.step()\n\n\ndef ask():\n    \"\"\"A new pair: two tries from the same start, the second with a quarter of its moves at random; the world answers.\"\"\"\n    a, b = one_try(5000 + len(pairs), 0.0), one_try(5000 + len(pairs), explore)\n    tries.extend([a, b])\n    pairs.append((len(tries) - 2, len(tries) - 1, 1.0 if a[2] > b[2] else 0.0 if b[2] > a[2] else 0.5))\n    fit_reward_model(min(400, 40 + 10 * len(pairs)))\n\n\nnext_ask = 0";
  function methodPy(d, T, steps, every, setup) {
    const s = d.s, m = d.method, g = flt(s.gamma), B = T.batch;
    // evolution (a genetic algorithm, evolution strategies, MAP-Elites): builder-py-worlds.js
    if (NA.brl.isEvo(m)) return NA.builderPyWorlds.method(d, T, steps, { flt, num, big, ACT_FN, setup, every, evolvable: !!NA.bevo && NA.bevo.evolvable(setup.blocks, d.input) });
    const update = ['    opt.zero_grad()', '    loss.backward()', T.clip ? `    torch.nn.utils.clip_grad_norm_(model.parameters(), ${num(T.clip)})   # cap the size of the step` : null, '    opt.step()', T.sched !== 'const' ? '    sched.step()' : null];
    // (Keep the best network: each new best score's weights, to end with)
    const keep = keepsBest(d);
    const report = (what) => (keep ? [`    if step % ${every} == 0:`, '        score = judge.score(best_move)', '        if score > best[0]:                        # its best yet: keep its weights', '            best = (score, copy.deepcopy(model.state_dict()))', `        print(f"step {step:5d}  ${what}  score {score:.3f}")`]
      : [`    if step % ${every} == 0:`, `        print(f"step {step:5d}  ${what}  score {judge.score(best_move):.3f}")`]);
    // Copy you: your moves (demos.json, from the Python tab's Download your moves), a classifier's steps on them
    if (m === 'bc') return ['# your moves, saved from the page (the Python tab\'s "Download your moves"): what it saw before each, and your move',
      'with open("demos.json") as f:', '    demos = json.load(f)', 'X_demo, A_demo = torch.tensor(demos["obs"], dtype=torch.float32), torch.tensor(demos["moves"])', 'print(f"{len(A_demo)} moves of yours")',
      `batch = ${B}`, `for step in range(${big(steps)}):`, '    i = torch.randint(len(A_demo), (batch,))',
      '    loss = F.cross_entropy(model(X_demo[i])[:, :n_actions], A_demo[i])   # your move in that moment, as a class', ...update, ...report('loss {loss.item():.4f}')];
    // moves in 8 copies of the world, sampled from the policy, the copies as one batch: actor-critic and PPO (each
    // reward normalised as the page does, unless normReward is off)
    const norm = (m === 'a2c' || m === 'ppo' || m === 'pref' || m === 'gauss') && s.normReward !== false;
    const roll = (n) => [`    for t in range(${n}):                         # ${n} moves in each of 8 copies of the world, the copies as one batch`, '        out = outputs(torch.stack(obs))', '        dist = torch.distributions.Categorical(logits=out[:, :n_actions])', '        acts = dist.sample()',
      '        X.append(torch.stack(obs)); A.append(acts); V.append(out[:, n_actions]); LP.append(dist.log_prob(acts))', '        rs, ds = [], []', '        for k, e in enumerate(envs):', '            o2, r, done = e.step(int(acts[k]))', ...(m === 'pref' ? ['            r = r_hat(obs[k], int(acts[k]))       # the reward model\'s reward, not the world\'s'] : []), '            obs[k] = e.reset() if done else o2', `            rs.append(${norm ? 'scale(k, r, done)' : 'r'}); ds.append(float(done))`, '        R.append(torch.tensor(rs)); D.append(torch.tensor(ds))'];
    const scaler = norm ? ['class ReturnScale:', '    """Each reward divided by the running spread of the discounted return (as gym\'s NormalizeReward), kept within',
      '    ±10: a world that pays in hundreds then learns like one that pays in ones. The return it reports stays the world\'s."""', '    def __init__(self, n):', '        self.disc, self.n, self.mean, self.m2 = [0.0] * n, 0, 0.0, 0.0', '    def __call__(self, k, r, done):',
      '        self.disc[k] = self.disc[k] * gamma + r     # this copy\'s discounted return so far', '        self.n += 1', '        d = self.disc[k] - self.mean', '        self.mean += d / self.n                    # its running mean and spread, over every copy',
      '        self.m2 += d * (self.disc[k] - self.mean)', '        if done:', '            self.disc[k] = 0.0', '        sd = math.sqrt(self.m2 / (self.n - 1)) if self.n > 1 else 1.0', '        return max(-10.0, min(10.0, r / (sd + 1e-8)))', 'scale = ReturnScale(8)'] : [];
    // the methods for numbers: contPy, below
    if (NA.brl.isCont(m)) return contPy(d, T, steps, every, { g, B, update, report, scaler, norm });
    if (m === 'dqn') {
      const mem = s.dqnMem || 10000, tgt = s.dqnTarget || 500, dbl = s.dqnDouble !== false, duel = !!s.dqnDueling, nst = s.dqnN || 1, Q = (x) => (duel ? `q_values(${x})` : `${x}[:, :n_actions]`);
      return [...(duel ? ['def q_values(out):                                 # dueling: the value plus each move\'s advantage, less their mean', '    adv = out[:, :n_actions]', '    return out[:, n_actions:n_actions + 1] + adv - adv.mean(1, keepdim=True)', '', ''] : []),
        `gamma, explore, batch, n_step = ${g}, ${s.explore}, ${B}, ${nst}`, `target = copy.deepcopy(model).eval()             # the target network: a copy, refreshed every ${tgt} updates`,
        `memory = collections.deque(maxlen=${big(mem)})          # the replay memory: (observation, move, ${nst > 1 ? 'its next n rewards, discounted' : 'reward'}, the observation after them or None, how many)`,
        ...(nst > 1 ? ['recent = collections.deque()                       # the last moves, waiting for their n-step return'] : []),
        'obs = env.reset()', `for step in range(${big(steps)}):`, '    eps = max(0.05, 1 - step / explore) if explore else 0.0   # the chance of a random move: from 1 down to 0.05',
        '    a = random.randrange(n_actions) if random.random() < eps else best_move(obs)', '    obs2, r, done = env.step(a)',
        ...(nst > 1 ? ['    recent.append((obs, a, r))', '    while recent and (len(recent) == n_step or done):   # each waiting move: its next n rewards (fewer at the end), then where it got to', '        o0, a0, _ = recent[0]',
          '        ret = sum(q[2] * gamma ** k for k, q in enumerate(recent))', '        memory.append((o0, a0, ret, None if done else obs2, len(recent)))', '        recent.popleft()'] : ['    memory.append((obs, a, r, None if done else obs2, 1))']),
        '    obs = env.reset() if done else obs2', '    if len(memory) < 64:', '        continue',
        '    sample = [memory[random.randrange(len(memory))] for _ in range(batch)]', '    X, A = torch.stack([q[0] for q in sample]), torch.tensor([q[1] for q in sample])',
        '    R, alive = torch.tensor([q[2] for q in sample]), torch.tensor([float(q[3] is not None) for q in sample])', '    k = torch.tensor([float(q[4]) for q in sample])', '    X2 = torch.stack([q[0] if q[3] is None else q[3] for q in sample])',
        ...(dbl ? ['    with torch.no_grad():                         # double DQN: the network picks the next move, the target network says what it is worth', `        nxt = ${Q('outputs(X2)')}.argmax(1)`, `        y = R + gamma ** k * alive * ${Q('target(X2)')}.gather(1, nxt[:, None])[:, 0]`]
          : ['    with torch.no_grad():                         # the target network picks the next move and says what it is worth', `        y = R + gamma ** k * alive * ${Q('target(X2)')}.max(1).values`]),
        `    q = ${Q('model(X)')}.gather(1, A[:, None])[:, 0]`, '    loss = F.smooth_l1_loss(q, y)                  # Huber: squared near the target, absolute beyond 1', ...update,
        `    if step % ${tgt} == ${tgt - 1}:`, '        target.load_state_dict(model.state_dict())', ...report('loss {loss.item():.4f}')];
    }
    if (m === 'pg') return [`gamma = ${g}`, 'obs = env.reset()', `for step in range(${big(steps)}):`, '    batch_obs, batch_a, batch_ret = [], [], []',
      '    while len(batch_obs) < 64:                    # whole episodes until about 64 moves are in hand', '        rewards, done = [], False', '        while not done:', '            a = int(torch.multinomial(outputs(obs[None])[0, :n_actions].softmax(0), 1))   # a move sampled from the policy',
      '            obs2, r, done = env.step(a)', '            batch_obs.append(obs); batch_a.append(a); rewards.append(r)', '            obs = env.reset() if done else obs2', '        ret, returns = 0.0, []', '        for r in reversed(rewards):                # the discounted return after each move',
      '            ret = r + gamma * ret', '            returns.append(ret)', '        batch_ret += returns[::-1]', '    ret = torch.tensor(batch_ret)', '    sd = ret.std(unbiased=False)', '    adv = (ret - ret.mean()) / (sd if sd > 0 else 1)    # how much better than usual, in its spread',
      '    logp = model(torch.stack(batch_obs))[:, :n_actions].log_softmax(1)', '    entropy = -(logp.exp() * logp).sum(1)', '    loss = -(adv * logp[torch.arange(len(adv)), torch.tensor(batch_a)]).mean() - 0.01 * entropy.mean()', ...update, ...report('loss {loss.item():.4f}')];
    if (m === 'a2c') return [`gamma = ${g}`, ...scaler, 'envs = [World() for _ in range(8)]', 'obs = [e.reset() for e in envs]', `for step in range(${big(steps)}):`, '    X, A, R, V, LP, D = [], [], [], [], [], []', ...roll(8),
      '    ret = outputs(torch.stack(obs))[:, n_actions]   # the critic\'s value of where each copy ended up', '    returns = []', '    for t in reversed(range(8)):                 # the returns, with that value standing in for what comes after', '        ret = R[t] + gamma * ret * (1 - D[t])', '        returns.insert(0, ret)',
      '    returns, V = torch.stack(returns).flatten(), torch.stack(V).flatten()', '    adv = returns - V                              # the advantage: better or worse than the critic expected', '    adv = (adv - adv.mean()) / (adv.std(unbiased=False) + 1e-8)',
      '    out = model(torch.cat(X))', '    logp = out[:, :n_actions].log_softmax(1)', '    entropy = -(logp.exp() * logp).sum(1)', '    chosen = logp[torch.arange(len(logp)), torch.cat(A)]',
      '    loss = -(adv * chosen).mean() - 0.01 * entropy.mean() + 0.25 * ((out[:, n_actions] - returns) ** 2).mean()', ...update, ...report('loss {loss.item():.4f}')];
    if (m === 'ppo' || m === 'pref') return [...(m === 'pref' ? [PREF_PY.replace('__EVERY__', String(Math.max(10, Math.round((s.prefEvery || 300) / 5)))), '', ''] : []), `gamma, lam, batch = ${g}, ${flt(s.ppoLam ?? 0.95)}, ${B}`, ...scaler, 'envs = [World() for _ in range(8)]', 'obs = [e.reset() for e in envs]', 'queue = []', `for step in range(${big(steps)}):`,
      ...(m === 'pref' ? ['    while step >= next_ask:                       # a new pair (the first four one after another); the reward model learns', '        ask()', '        next_ask = step + (0 if len(pairs) < 4 else ask_every)'] : []), '    if not queue:', '        X, A, R, V, LP, D = [], [], [], [], [], []', ...roll(16).map((l) => '    ' + l),
      '        nxt, gae, adv = outputs(torch.stack(obs))[:, n_actions], torch.zeros(8), []', '        for t in reversed(range(16)):            # advantages by GAE (λ = 0.95)', '            delta = R[t] + gamma * nxt * (1 - D[t]) - V[t]', '            gae = delta + gamma * lam * (1 - D[t]) * gae', '            adv.insert(0, gae)', '            nxt = V[t]',
      '        adv = torch.stack(adv).flatten()', '        ret = adv + torch.stack(V).flatten()', '        adv = (adv - adv.mean()) / (adv.std(unbiased=False) + 1e-8)', '        data = (torch.cat(X), torch.cat(A), torch.cat(LP), adv, ret)',
      `        for _ in range(${s.ppoEpochs ?? 4}):                       # ${s.ppoEpochs ?? 4} pass${(s.ppoEpochs ?? 4) === 1 ? '' : 'es'} over the 128 moves, in shuffled minibatches`, '            perm = torch.randperm(len(adv))', '            queue += [perm[i:i + batch] for i in range(0, len(perm), batch)]',
      '    i = queue.pop(0)', '    x, a, old_logp, adv_i, ret_i = (q[i] for q in data)', '    out = model(x)', '    logp = out[:, :n_actions].log_softmax(1)', '    ratio = (logp[torch.arange(len(i)), a] - old_logp).exp()',
      `    clipped = ratio.clamp(${flt(1 - (s.ppoClip ?? 0.2))}, ${flt(1 + (s.ppoClip ?? 0.2))})                # once a move is ${Math.round(100 * (s.ppoClip ?? 0.2))}% more or less likely than when it was played, no further push`,
      '    entropy = -(logp.exp() * logp).sum(1)           # less a little entropy, as in actor-critic, to keep it exploring',
      `    loss = -torch.min(ratio * adv_i, clipped * adv_i).mean() - ${num(s.ppoEnt ?? 0.01)} * entropy.mean() + 0.25 * ((out[:, n_actions] - ret_i) ** 2).mean()`, ...update, ...report('loss {loss.item():.4f}')];
  }
  /* The methods for numbers, in PyTorch (the page: js/builder-rl.js). PPO with a Gaussian policy: a bell curve per
     number, its middle from the network and its spread (log_std) learned beside it. SAC and TD3: two critics of their
     own (64 and 64 ReLU neurons, the observation and the numbers in), their slow copies, a replay memory, 1,000 random
     tries first; SAC's α learns to keep the curves' entropy near −(the numbers it sets). */
  function contPy(d, T, steps, every, h) {
    const m = d.method, { g, B, update, report, scaler, norm } = h;
    const W = NA.rlEnvs.make(d.world, d.ws);
    W.reset(new NA.RNG(1));
    const most = W.maxSteps || 1000;
    const s = d.s, clip = s.ppoClip ?? 0.2;
    if (m === 'gauss') return [`gamma, lam, batch = ${g}, ${flt(s.ppoLam ?? 0.95)}, ${B}`, ...scaler,
      'log_std = torch.full((n_numbers,), -0.5, requires_grad=True)   # the spread of each number\'s bell curve (its log), learned beside the network',
      `opt_std = torch.optim.Adam([log_std], lr=${num(T.lr)})`, 'envs = [World() for _ in range(8)]', 'obs = [e.reset() for e in envs]', 'queue = []', `for step in range(${big(steps)}):`, '    if not queue:',
      '        X, A, R, V, LP, D = [], [], [], [], [], []', '        for t in range(16):                      # 16 tries in each of 8 copies of the world, the copies as one batch', '            out = outputs(torch.stack(obs))',
      '            dist = torch.distributions.Normal(out[:, :n_numbers], log_std.detach().exp())', '            acts = dist.sample()',
      '            X.append(torch.stack(obs)); A.append(acts); V.append(out[:, n_numbers]); LP.append(dist.log_prob(acts).sum(1))', '            rs, ds = [], []', '            for k, e in enumerate(envs):',
      '                o2, r, done = e.step(acts[k].clamp(-1, 1).tolist())   # the try, clipped to -1 ... 1', '                obs[k] = e.reset() if done else o2', `                rs.append(${norm ? 'scale(k, r, done)' : 'r'}); ds.append(float(done))`,
      '            R.append(torch.tensor(rs)); D.append(torch.tensor(ds))', '        nxt, gae, adv = outputs(torch.stack(obs))[:, n_numbers], torch.zeros(8), []', '        for t in reversed(range(16)):            # advantages by GAE (λ = 0.95)',
      '            delta = R[t] + gamma * nxt * (1 - D[t]) - V[t]', '            gae = delta + gamma * lam * (1 - D[t]) * gae', '            adv.insert(0, gae)', '            nxt = V[t]', '        adv = torch.stack(adv).flatten()', '        ret = adv + torch.stack(V).flatten()',
      '        adv = (adv - adv.mean()) / (adv.std(unbiased=False) + 1e-8)', '        data = (torch.cat(X), torch.cat(A), torch.cat(LP), adv, ret)', `        for _ in range(${s.ppoEpochs ?? 4}):                       # passes over the 128 tries, in shuffled minibatches`,
      '            perm = torch.randperm(len(adv))', '            queue += [perm[i:i + batch] for i in range(0, len(perm), batch)]', '    i = queue.pop(0)', '    x, a, old_logp, adv_i, ret_i = (q[i] for q in data)', '    out = model(x)',
      '    logp = torch.distributions.Normal(out[:, :n_numbers], log_std.exp()).log_prob(a).sum(1)', '    ratio = (logp - old_logp).exp()', `    clipped = ratio.clamp(${flt(1 - clip)}, ${flt(1 + clip)})                # once a try is ${Math.round(100 * clip)}% more or less likely than when it was made, no further push`,
      '    entropy = (log_std + 0.5 * math.log(2 * math.pi * math.e)).sum()   # the curves\' entropy, a little of it, to keep it trying',
      `    loss = -torch.min(ratio * adv_i, clipped * adv_i).mean() - ${num((s.ppoEnt ?? 0.01) / 10)} * entropy + 0.25 * ((out[:, n_numbers] - ret_i) ** 2).mean()`,
      '    opt_std.zero_grad()', ...update, '    opt_std.step()', '    with torch.no_grad():', '        log_std.clamp_(-2.5, 0.5)', ...report('loss {loss.item():.4f}')];
    const sac = m === 'sac';
    return [
      'class Critic(nn.Module):', '    """What a state and a try are worth: the observation and the numbers in, the return it expects out."""', '    def __init__(self):', '        super().__init__()',
      '        self.net = nn.Sequential(nn.Linear(Xva[0].numel() + n_numbers, 64), nn.ReLU(), nn.Linear(64, 64), nn.ReLU(), nn.Linear(64, 1))', '',
      '    def forward(self, x, a):', '        return self.net(torch.cat([x.flatten(1), a], 1))[:, 0]', '', '',
      'q1, q2 = Critic(), Critic()                         # two critics, each judged against the smaller of their slow copies', 'q1_slow, q2_slow = copy.deepcopy(q1), copy.deepcopy(q2)',
      'critic_opt = torch.optim.Adam(list(q1.parameters()) + list(q2.parameters()), lr=1e-3)', 'memory = collections.deque(maxlen=100_000)          # the replay memory: (observation, try, reward, next observation, over)',
      `gamma, batch, tau, warm, most = ${g}, ${Math.max(64, B)}, 0.005, ${s.warm ?? 1000}, ${most}   # tau: how fast the slow copies follow; warm: random tries first; most: the longest episode`,
      ...(sac ? ['log_alpha = torch.tensor(math.log(0.2), requires_grad=True)   # α: what the curves\' entropy is worth, learned', 'alpha_opt = torch.optim.Adam([log_alpha], lr=3e-4)']
        : ['actor_slow = copy.deepcopy(model)                  # the slow copy of the network, for the critics\' targets']), '', '',
      'def slow_follow(slow, fast):                       # Polyak averaging: the slow copy moves a little toward the fast one', '    with torch.no_grad():', '        for p_slow, p in zip(slow.parameters(), fast.parameters()):', '            p_slow.mul_(1 - tau).add_(tau * p)', '', '',
      ...(sac ? ['def sac_try(out):', '    """For each row of outputs (the middles, then the logs of the spreads): tanh of a draw from the bell curve, and its log probability."""',
        '    mu, log_sd = out[:, :n_numbers], out[:, n_numbers:].clamp(-5, 2)', '    u = mu + log_sd.exp() * torch.randn_like(mu)', '    a = u.tanh()',
        '    logp = (torch.distributions.Normal(mu, log_sd.exp()).log_prob(u) - torch.log(1 - a ** 2 + 1e-6)).sum(1)', '    return a, logp', '', ''] : []),
      'obs = env.reset()', `for step in range(${big(steps)}):`, '    if step < warm:                                # random tries first, to fill the memory', '        a = [random.uniform(-1, 1) for _ in range(n_numbers)]', '    else:', '        with torch.no_grad():',
      sac ? '            a = sac_try(outputs(obs[None]))[0][0].tolist()' : '            a = (outputs(obs[None])[0, :n_numbers].tanh() + 0.1 * torch.randn(n_numbers)).clamp(-1, 1).tolist()   # with a little noise, to explore',
      '    obs2, r, done = env.step(a)', '    memory.append((obs, a, r, obs2, float(done and env.t < most)))   # (one cut short by the time limit is not over: its last state has a value)', '    obs = env.reset() if done else obs2',
      '    if step < warm or len(memory) < 256:', '        continue', '    sample = [memory[random.randrange(len(memory))] for _ in range(batch)]', '    X, X2 = torch.stack([q[0] for q in sample]), torch.stack([q[3] for q in sample])',
      '    A, R, OVER = (torch.tensor([q[k] for q in sample], dtype=torch.float32) for k in (1, 2, 4))',
      ...(sac ? ['    alpha = log_alpha.exp().detach()', '    with torch.no_grad():                         # the target: r + γ (the smaller slow critic\'s value of a fresh try at the next state − α log π of it)', '        a2, logp2 = sac_try(model(X2))',
        '        y = R + gamma * (1 - OVER) * (torch.min(q1_slow(X2, a2), q2_slow(X2, a2)) - alpha * logp2)']
        : ['    with torch.no_grad():                         # the target: the slow copy\'s try at the next state, with clipped noise, valued by the smaller slow critic', '        a2 = (actor_slow(X2)[:, :n_numbers].tanh() + (0.2 * torch.randn(batch, n_numbers)).clamp(-0.5, 0.5)).clamp(-1, 1)',
          '        y = R + gamma * (1 - OVER) * torch.min(q1_slow(X2, a2), q2_slow(X2, a2))']),
      '    critic_loss = 0.5 * ((q1(X, A) - y) ** 2).mean() + 0.5 * ((q2(X, A) - y) ** 2).mean()', '    critic_opt.zero_grad()', '    critic_loss.backward()', '    torch.nn.utils.clip_grad_norm_(list(q1.parameters()) + list(q2.parameters()), 10)', '    critic_opt.step()',
      ...(sac ? ['    a_new, logp = sac_try(model(X))                 # the policy: tries the critics value more, and wider curves', '    loss = (alpha * logp - torch.min(q1(X, a_new), q2(X, a_new))).mean()', ...update,
        '    alpha_loss = -(log_alpha * (logp.detach() - n_numbers)).mean()   # α up when the curves are narrower than wanted, down when wider', '    alpha_opt.zero_grad()', '    alpha_loss.backward()', '    alpha_opt.step()', '    slow_follow(q1_slow, q1)', '    slow_follow(q2_slow, q2)']
        : ['    if step % 2:                                   # every second step: the network toward the tries the first critic values most, and the slow copies', '        loss = -q1(X, model(X)[:, :n_numbers].tanh()).mean()', ...update.filter(Boolean).map((l) => '    ' + l),
          '        slow_follow(actor_slow, model)', '        slow_follow(q1_slow, q1)', '        slow_follow(q2_slow, q2)']),
      ...report('critic loss {critic_loss.item():.4f}')];
  }
  function controlCode(setup, d, chk) {
    const curr = d.s.curriculum === 'on' && NA.brl.hasCurriculum(d.world, d.method); // (easy first: a curriculum)
    const wm = d.s.worldModel === 'on' && NA.brl.canModel(d.input, d.method); // (a model of the world)
    const pp = !!d.s.chaserLearns && NA.brl.canChase(d.world, d.method); // (predator and prey: the chaser learns too)
    const p = BT.byId[setup.problem], T = setup.train, steps = T.steps || NA.brl.stepsFor(d.world, d.method), M = NA.brl.METHODS[d.method], evo = NA.brl.isEvo(d.method), s = d.s;
    const how = { ga: `a population of ${s.pop}${s.grow && NA.bevo && NA.bevo.evolvable(setup.blocks, d.input) ? ' whose networks grow and shrink' : ''}`, es: `${Math.max(2, Math.round((s.pop || 24) / 2))} pairs of nudged copies a generation`, me: `${s.pop} new networks a generation, kept in a 12 × 12 grid of styles` }[d.method];
    const optName = { sgd: 'SGD', momentum: 'SGD with momentum', adam: 'Adam', adamw: 'AdamW' }[T.opt], every = Math.max(1, Math.round(steps / 10));
    return lines(
      `# Built in Neural Anatomy's model builder: ${p.label}, learning from rewards.`,
      `# Model:     ${summary(setup, d)}`,
      ...(Object.keys(partNames(setup).cls).length ? ['# Your parts, each a class below:', ...partsSummary(setup)] : []),
      `# Learns by: ${M.label}${evo ? `, ${how}` : `, ${optName}, learning rate ${num(T.lr)}, batch ${T.batch}`}, ${steps.toLocaleString('en')} ${evo ? 'generations' : 'steps'}${T.steps ? '' : ' (the page trains until you pause)'}`,
      d.s.pbt > 1 && NA.brl.pbtOk(d.method) ? `# Population-based training: the page trains ${Math.min(8, d.s.pbt)} learners side by side, the worst copying the best every 500 steps with a nudged learning rate; this script trains one.` : null,
      '# Only torch is needed.', d.method === 'bc' ? 'import json' : null, 'import math', 'import random', d.method === 'dqn' || d.method === 'sac' || d.method === 'td3' || (d.s.worldModel === 'on' && NA.brl.canModel(d.input, d.method)) || (d.s.chaserLearns && NA.brl.canChase(d.world, d.method)) ? 'import collections' : null, d.method === 'dqn' || d.method === 'sac' || d.method === 'td3' || keepsBest(d) || (d.s.chaserLearns && NA.brl.canChase(d.world, d.method)) ? 'import copy' : null, 'import torch', 'import torch.nn as nn', 'import torch.nn.functional as F', '',
      `torch.manual_seed(${T.seed})`, `random.seed(${T.seed})`, '',
      '# ---------------------------------------------------------------- the world', ...worldPy(d), ...lastMoveWrap(d), ...(d.method === 'me' && !NA.builderPyWorlds.worlds[d.world] ? ['', '', ...NA.builderPyWorlds.style(d.world)] : []), '', '',
      ...(curr ? ['class Curriculum:', '    """Easy first: the worlds it learns in start easier (level 0) and get a quarter harder whenever 3 in 4 of its last', '    10 episodes or more succeed; the one that scores it stays at full difficulty."""', '    level, tally = 0.0, []', '',
        '    @classmethod', '    def ended(cls, success):', '        cls.tally = (cls.tally + [float(success)])[-20:]', '        if cls.level < 1 and len(cls.tally) >= 10 and sum(cls.tally) >= 0.75 * len(cls.tally):',
        '            cls.level, cls.tally = min(1.0, cls.level + 0.25), []', '            print(f"curriculum: level {cls.level:.2f}")', '', '',
        'class World(World):', '    full = False                                   # the one that scores it: always at full difficulty', '',
        '    def reset(self, *args, **kw):', `        if not self.full and getattr(self, "t", 0) > 0:   # the episode that just ended: did it succeed?`, `            Curriculum.ended(${({ pole: 'self.t >= self.max_steps', arm: 'self.end == "reached"', lander: 'self.end == "landed"', hurdles: 'self.end == "course"' })[d.world]})`,
        '        self.level = 1.0 if self.full else Curriculum.level', '        return super().reset(*args, **kw)', '', ''] : []),
      ...(wm ? ['moves = collections.deque(maxlen=20_000)          # every move it makes: what it saw, the move, what it saw next, the reward', '', '',
        'class World(World):', '    """The same world, keeping every move for the model of the world (the page learns it beside the network)."""', '    def reset(self, *args, **kw):', '        self.seen = super().reset(*args, **kw)', '        return self.seen', '',
        '    def step(self, a):', '        o2, r, done = super().step(a)', '        moves.append((self.seen, a, o2, max(-10.0, min(10.0, r))))', '        self.seen = o2', '        return o2, r, done', '', ''] : []),
      ...(pp ? ['class ChaserBrain:', '    """The chaser that learns too: a DQN of its own, 8 numbers in (where the runner is from it, how fast each moves, where it', '    is), 5 moves out; paid 1 for a catch, 0.1 for every metre it closes, -0.01 a step. The judge\'s chaser follows its rule."""',
        '    def __init__(self):', '        self.q = nn.Sequential(nn.Linear(8, 64), nn.ReLU(), nn.Linear(64, 64), nn.ReLU(), nn.Linear(64, 5))', '        self.slow = copy.deepcopy(self.q)', '        self.opt = torch.optim.Adam(self.q.parameters(), lr=5e-4)', '        self.memory, self.n = collections.deque(maxlen=20_000), 0', '',
        '    def act(self, o):', '        if random.random() < max(0.05, 1 - self.n / 20_000):', '            return random.randrange(5)', '        with torch.no_grad():', '            return int(self.q(o).argmax())', '',
        '    def keep(self, o, a, r, o2, over):', '        self.memory.append((o, a, r, o2, over))', '        self.n += 1', '        if self.n % 2 or len(self.memory) < 500:', '            return', '        sample = [self.memory[random.randrange(len(self.memory))] for _ in range(64)]',
        '        O, O2 = torch.stack([q[0] for q in sample]), torch.stack([q[3] for q in sample])', '        A, R, OVER = torch.tensor([q[1] for q in sample]), torch.tensor([q[2] for q in sample]), torch.tensor([q[4] for q in sample])',
        '        with torch.no_grad():', '            y = R + 0.97 * (1 - OVER) * self.slow(O2).max(1).values', '        loss = F.smooth_l1_loss(self.q(O).gather(1, A[:, None])[:, 0], y)', '        self.opt.zero_grad()', '        loss.backward()',
        '        torch.nn.utils.clip_grad_norm_(self.q.parameters(), 10)', '        self.opt.step()', '        if self.n % 1000 == 0:', '            self.slow.load_state_dict(self.q.state_dict())', '', '',
        'chaser = ChaserBrain()', '', '',
        'class World(World):', '    learning = True                                # (the judge sets this off: its chaser follows its rule)', '',
        '    def sees(self):                                # what the chaser that learns sees', '        m, c = self.me, self.chasers[0]', '        return torch.tensor([(m.x - c.x) / 10, (c.y - m.y) / 10, m.vx / 0.4, -m.vy / 0.4, c.vx / 0.35, -c.vy / 0.35, c.x / 10 - 1, 1 - c.y / 10])', '',
        '    def step(self, a):', '        if not self.learning:', '            self.chaser_move = None', '            return super().step(a)', '        o, (_, d0) = self.sees(), self.nearest()', '        self.chaser_move = chaser.act(o)', '        o2, r, done = super().step(a)',
        '        caught = self.end == "tagged"', '        chaser.keep(o, self.chaser_move, 1.0 if caught else -0.01 + 0.1 * (d0 - self.nearest()[1]), self.sees(), float(caught))', '        return o2, r, done', '', ''] : []),
      'env, judge = World(), World()                     # one to learn in, one to measure its best moves in', curr ? 'judge.full = True' : null, pp ? 'judge.learning = False' : null, 'n_actions = len(World.actions)', d.cont ? `n_numbers = ${d.dA}                                      # the numbers it sets in place of a move (${d.cActions.join(', ')}), each from -1 to 1` : null, '', '',
      '# observations from random play (there are no examples with answers here): the architecture view\'s snippets', '# use the first as "the first validation example"', 'def random_observations(n):', '    w, out = World(), []', '    o = w.reset()', '    while len(out) < n:', '        out.append(o)', '        o, _, done = w.step(random.randrange(n_actions))', '        if done:', '            o = w.reset()', '    return torch.stack(out)', '', '', 'Xva = random_observations(40)', '', '',
      modelCode(setup, d, chk), '', '',
      'model = Model()', initCode(setup, d), 'print(f"{sum(p.numel() for p in model.parameters()):,} learned numbers")', '', '',
      'def outputs(X):                                    # the outputs for a batch of observations, without learning', '    model.eval()', '    with torch.no_grad():', '        out = model(X)', '    model.train()', '    return out', '', '',
      ...(d.cont ? ['def best_move(obs):                                # its best move: the middles of its curves' + (d.method === 'gauss' ? '' : ', through tanh'), `    return outputs(obs[None])[0, :n_numbers].${d.method === 'gauss' ? 'clamp(-1, 1)' : 'tanh()'}.tolist()`]
        : ['def best_move(obs):                                # its best move: the largest output', '    return int(outputs(obs[None])[0, :n_actions].argmax())']), '', '',
      evo ? null : ['# ---------------------------------------------------------------- the optimizer', ...optCode(T), '', ''],
      `# ---------------------------------------------------------------- learning by ${M.label}`, keepsBest(d) ? 'best = (-1.0, None)                                # the best score so far, and its weights' : null, ...methodPy(d, T, steps, every, setup), '',
      ...(keepsBest(d) ? ['if best[1] is not None and judge.score(best_move) < best[0]:   # it ended below its best: the network it had then', '    model.load_state_dict(best[1])'] : []),
      'print(f"done: its best moves score {judge.score(best_move):.3f}")', '',
      ...(wm ? ['', '# ---------------------------------------------------------------- a model of the world', '# The page learns it beside the network, a batch every 4 moves; here it is fitted to the moves kept, after training.',
        `n_act = ${d.cont ? 'n_numbers' : 'n_actions'}`, 'model_world = nn.Sequential(nn.Linear(Xva[0].numel() + n_act, 64), nn.ReLU(), nn.Linear(64, 64), nn.ReLU(), nn.Linear(64, Xva[0].numel() + 1))', 'opt_world = torch.optim.Adam(model_world.parameters(), lr=1e-3)', '', '',
        'def encode(a):                                     # a move as the model sees it', d.cont ? '    return torch.tensor(a, dtype=torch.float32)' : '    return F.one_hot(torch.tensor(a), n_actions).float()', '', '',
        'def imagine(o, a):                                 # its prediction: the next observation, and the reward', '    with torch.no_grad():', '        out = model_world(torch.cat([o.flatten(), encode(a)]))', '    return o + out[:-1].view_as(o), out[-1].item()', '', '',
        `kept = [m for m in moves if isinstance(m[1], int) == ${d.cont ? 'False' : 'True'}]   # (its own kind of move: not the random moves that first showed the inputs)`, 'O = torch.stack([m[0].flatten() for m in kept])', 'A = torch.stack([encode(m[1]) for m in kept])', 'D = torch.stack([(m[2] - m[0]).flatten() for m in kept])', 'R = torch.tensor([m[3] for m in kept])',
        'for step in range(3000):', '    i = torch.randint(len(O), (64,))', '    out = model_world(torch.cat([O[i], A[i]], 1))', '    loss = ((out - torch.cat([D[i], R[i, None]], 1)) ** 2).mean()   # the change in what it sees, and the reward',
        '    opt_world.zero_grad()', '    loss.backward()', '    opt_world.step()', 'print(f"the model of the world, one move ahead: off by {loss.item() ** 0.5:.3f}")', '',
        '# a dream beside the world: its best moves from one start, in the world and in the model\'s imagination', 'w = World()', 'o = w.reset()', 'oh, done, paid, dreamt = o.clone(), False, 0.0, 0.0', 'for t in range(60):', '    o, r, done = w.step(best_move(o))', '    oh, rh = imagine(oh, best_move(oh))', '    paid, dreamt = paid + r, dreamt + rh',
        '    print(f"{t:3d}  the world {o.flatten()[0].item():+.3f}  its dream {oh.flatten()[0].item():+.3f}")', '    if done:', '        break', 'print(f"the world paid {paid:.2f}; it imagined {dreamt:.2f}")', ''] : []),
      '# ---------------------------------------------------------------- watch it play one episode with its best moves', 'o, done, total = env.reset(), False, 0.0', 'while not done:', '    a = best_move(o)', '    o, r, done = env.step(a)', '    total += r', d.cont ? '    print(" ".join(f"{v:+.2f}" for v in a), end=" | ")' : '    print(World.actions[a], end="  ")', 'print(f"\\nreturn {total:.2f}")');
  }

  // the scripts that use random (the stroke drawings, the stories, the chaotic series, the sequence classes)
  const needsRandom = (p, d) => d.family === 'digits' || p.task === 'language' || p.task === 'chaos' || p.task === 'heartbeat' || p.task === 'add' || d.family === 'seqclass' || !!(NA.bgen && NA.bgen.owns(d.family));
  PY.code = function (setup, d) {
    const p = BT.byId[setup.problem], T = setup.train, steps = T.steps || 3000;
    if (!d || d.error) return `# ${d && d.error ? d.error : 'Pick a problem first.'}`;
    if (d.family === 'control' && NA.cml && setup.blocks.some((b) => NA.cml.isClassic(b.type))) return '# A classic model learns from examples with answers, and here there are only rewards: use a network.';
    if (NA.bgen && NA.bgen.owns(d.family) && NA.cml && setup.blocks.some((b) => NA.cml.isClassic(b.type))) return '# A classic model cannot learn this kind of problem: use a network.';
    if (NA.cml && setup.blocks.some((b) => NA.cml.isClassic(b.type))) {
      const cc = NA.cml.check(setup.blocks, d);
      if (!cc.ok) return lines('# The model does not fit together yet, so there is nothing to write out.', `# ${((cc.shapes.find((s) => s && s.problem) || {}).problem || { reason: 'See the block marked in red.' }).reason}`);
      return classicCode(setup, d);
    }
    const chk = NA.bx ? NA.bx.check(setup.blocks, d.input, d.output) : { ok: false, shapes: [] };
    if (!chk.ok) return lines('# The model does not fit together yet, so there is nothing to write out.', `# ${((chk.problem || (chk.shapes.find((s) => s && s.problem) || {}).problem) || { reason: 'See the block marked in red.' }).reason}`);
    if (d.family === 'control') return controlCode(setup, d, chk);
    if (NA.builderPyGen && (d.family === 'gan' || d.family === 'contrast')) return NA.builderPyGen.script(setup, d, chk, PY._i);
    const grows = !!T.grow && !!NA.bevo && !(NA.bgen && NA.bgen.owns(d.family)) && NA.bevo.evolvable(setup.blocks, d.input), vae = !!(NA.builderPyGen && NA.builderPyGen.hasSample(setup)), rnd = needsRandom(p, d) || grows; // (grow and prune picks its changes at random)
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
      `# Training:  ${lossName} loss, ${optName}, learning rate ${num(T.lr)}, batch ${T.batch}, ${steps.toLocaleString('en')} steps${T.steps ? '' : ' (the page trains until you pause)'}${grow ? `; ${NA.bevo.growOptions(T, true).members} networks side by side, their shapes evolving (grow and prune)` : ''}`,
      ...needs.filter(Boolean), 'import torch', 'import torch.nn as nn', 'import torch.nn.functional as F', '',
      `torch.manual_seed(${T.seed})`, rnd ? `random.seed(${T.seed})` : null, '',
      dataCode(setup, d), '', '',
      modelCode(setup, d, chk), '', '',
      'model = Model()', initCode(setup, d), 'print(f"{sum(p.numel() for p in model.parameters()):,} learned numbers")', '',
      '# ---------------------------------------------------------------- loss, optimizer and measure', lossCode(setup, d), '', scoreCode(d, setup), '',
      grow ? ['', growing] : [optCode(T), '',
      'def evaluate(X, Y):', '    model.eval()', '    with torch.no_grad():', '        out = model(X)', vae ? '        loss, s = (loss_fn(out, Y) + kl(model)).item(), score(out, Y).item()' : '        loss, s = loss_fn(out, Y).item(), score(out, Y).item()', '    model.train()', '    return loss, s', '',
      '# ---------------------------------------------------------------- training', `batch = ${T.batch}`, 'order, pos = torch.randperm(len(Xtr)), 0',
      `for step in range(${big(steps)}):`,
      '    if pos + batch > len(order):                  # a new shuffle every pass over the data', '        order, pos = torch.randperm(len(Xtr)), 0',
      '    i = order[pos:pos + batch]', '    pos += batch',
      vae ? '    loss = loss_fn(model(Xtr[i]), Ytr[i]) + kl(model)   # a VAE: and its codes pulled toward one bell curve' : '    loss = loss_fn(model(Xtr[i]), Ytr[i])', '    opt.zero_grad()', '    loss.backward()',
      T.clip ? `    torch.nn.utils.clip_grad_norm_(model.parameters(), ${num(T.clip)})   # cap the size of the step` : null,
      '    opt.step()', T.sched !== 'const' ? '    sched.step()' : null,
      `    if step % ${evalEvery} == 0:`, '        val_loss, val_score = evaluate(Xva, Yva)', '        print(f"step {step:5d}  training loss {loss.item():.4f}  validation loss {val_loss:.4f}  score {val_score:.3f}")', '',
      'val_loss, val_score = evaluate(Xva, Yva)', 'print(f"done: validation loss {val_loss:.4f}, score {val_score:.3f}")'],
      afterCode(setup, d));
  };

  // the pieces builder-py-gen.js builds its scripts from
  PY._i = { lines, num, flt, big, SPLIT, DIGITS_PY, SHAPES_PY, ACT_FN, modelCode, initCode, optCode, summary, attrPath: (setup, i, stem) => PY.attrPath(setup, i, stem) };
  NA.builderPy = PY;
})(window.NA);
