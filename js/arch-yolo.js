/* arch-yolo.js — the YOLO26 detector as a zoomable architecture: the image, the backbone's blocks, the neck that
   mixes three scales, the two heads and the post-processing that turns their maps into boxes. Every block opens
   into its parts and feature maps, and any number on any map opens into whatever produced it: a convolution's
   weighted sum, a residual sum, a max-pool, an attention-weighted sum, a pixel of the image. From there NA.deep's
   levels take over: the PyTorch code, float arithmetic, circuits and electrons. All numbers are from the scene
   on screen, as the browser just computed them. */
(function (NA) {
  'use strict';
  const { fmt, fmtInt, theme, clamp } = NA;
  const Y = NA.yolo;
  const W2 = (X, w, t) => (X.wide ? w : t);
  const CLS = (i, a = 1) => (a >= 1 ? theme.series[i] : theme.series[i].replace('rgb(', 'rgba(').replace(')', `,${a})`));
  const SILU = { id: 'swish', label: 'SiLU', f: (z) => Y.silu(z) };

  // ------------------------------------------------------------------ PyTorch versions
  const py = {};
  py.layers = `import math
import copy
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

IMG, STRIDES = 128, (8, 16, 32)
NAMES = ['circle', 'triangle', 'square', 'star']
NC = len(NAMES)


class Conv(nn.Module):
    """Conv2d → BatchNorm → SiLU: the building block of every YOLO since v5."""
    def __init__(self, c1, c2, k=1, s=1, g=1, act=True):
        super().__init__()
        self.conv = nn.Conv2d(c1, c2, k, s, k // 2, groups=g, bias=False)
        self.bn = nn.BatchNorm2d(c2, eps=1e-3, momentum=0.03)
        self.act = act

    def forward(self, x):
        x = self.bn(self.conv(x))
        return F.silu(x) if self.act else x


class Bottleneck(nn.Module):
    """Two 3x3 convs and a shortcut: the block learns a correction to add to its input."""
    def __init__(self, c1, c2, shortcut=True, e=0.5):
        super().__init__()
        c_ = int(c2 * e)
        self.cv1, self.cv2 = Conv(c1, c_, 3), Conv(c_, c2, 3)
        self.add = shortcut and c1 == c2

    def forward(self, x):
        y = self.cv2(self.cv1(x))
        return x + y if self.add else y


class C3k(nn.Module):
    """CSP block: half the channels go through two bottlenecks, half skip them; a 1x1 conv mixes the two."""
    def __init__(self, c1, c2, n=2, e=0.5):
        super().__init__()
        c_ = int(c2 * e)
        self.cv1, self.cv2, self.cv3 = Conv(c1, c_), Conv(c1, c_), Conv(2 * c_, c2)
        self.m = nn.Sequential(*(Bottleneck(c_, c_, True, 1.0) for _ in range(n)))

    def forward(self, x):
        return self.cv3(torch.cat((self.m(self.cv1(x)), self.cv2(x)), 1))


class Attention(nn.Module):
    """Self-attention between the positions of a feature map, plus a 3x3 depthwise conv of the values."""
    def __init__(self, dim, num_heads, attn_ratio=0.5):
        super().__init__()
        self.nh, self.hd = num_heads, dim // num_heads
        self.kd = int(self.hd * attn_ratio)
        self.qkv = Conv(dim, dim + 2 * self.kd * num_heads, 1, act=False)
        self.proj = Conv(dim, dim, 1, act=False)
        self.pe = Conv(dim, dim, 3, g=dim, act=False)

    def forward(self, x):
        B, C, H, W = x.shape
        q, k, v = self.qkv(x).view(B, self.nh, 2 * self.kd + self.hd, H * W).split([self.kd, self.kd, self.hd], 2)
        a = ((q.transpose(-2, -1) @ k) * self.kd ** -0.5).softmax(-1)
        y = (v @ a.transpose(-2, -1)).view(B, C, H, W) + self.pe(v.reshape(B, C, H, W))
        return self.proj(y)


class PSABlock(nn.Module):
    def __init__(self, c, num_heads):
        super().__init__()
        self.attn = Attention(c, num_heads)
        self.ffn = nn.Sequential(Conv(c, 2 * c), Conv(2 * c, c, act=False))

    def forward(self, x):
        x = x + self.attn(x)
        return x + self.ffn(x)


class C3k2(nn.Module):
    """Split the channels in two, run a block on the second half, concatenate everything, mix with a 1x1 conv."""
    def __init__(self, c1, c2, n=1, c3k=False, e=0.5, attn=False):
        super().__init__()
        self.c = int(c2 * e)
        self.cv1, self.cv2 = Conv(c1, 2 * self.c), Conv((2 + n) * self.c, c2)
        block = lambda: (nn.Sequential(Bottleneck(self.c, self.c), PSABlock(self.c, max(self.c // 64, 1))) if attn
                         else C3k(self.c, self.c, 2) if c3k else Bottleneck(self.c, self.c))
        self.m = nn.ModuleList(block() for _ in range(n))

    def forward(self, x):
        y = list(self.cv1(x).chunk(2, 1))
        y.extend(m(y[-1]) for m in self.m)
        return self.cv2(torch.cat(y, 1))


class SPPF(nn.Module):
    """Three 5x5 max-pools in a row, each seeing a wider window; all four maps are concatenated."""
    def __init__(self, c1, c2, k=5, n=3, shortcut=True):
        super().__init__()
        c_ = c1 // 2
        self.cv1, self.cv2 = Conv(c1, c_, act=False), Conv(c_ * (n + 1), c2)
        self.k, self.n, self.add = k, n, shortcut and c1 == c2

    def forward(self, x):
        y = [self.cv1(x)]
        for _ in range(self.n):
            y.append(F.max_pool2d(y[-1], self.k, 1, self.k // 2))
        y = self.cv2(torch.cat(y, 1))
        return y + x if self.add else y


class C2PSA(nn.Module):
    def __init__(self, c1, c2, n=1, e=0.5):
        super().__init__()
        self.c = int(c1 * e)
        self.cv1, self.cv2 = Conv(c1, 2 * self.c), Conv(2 * self.c, c1)
        self.m = nn.Sequential(*(PSABlock(self.c, max(self.c // 64, 1)) for _ in range(n)))

    def forward(self, x):
        a, b = self.cv1(x).split((self.c, self.c), 1)
        return self.cv2(torch.cat((a, self.m(b)), 1))


class Detect(nn.Module):
    """Per scale: a box branch (4 distances: no DFL, reg_max = 1) and a class branch. The one-to-one copies
    have their own weights and are trained with one positive cell per object."""
    def __init__(self, nc, ch):
        super().__init__()
        c2, c3 = max(16, ch[0] // 4, 4), max(ch[0], min(nc, 100))
        self.cv2 = nn.ModuleList(nn.Sequential(Conv(x, c2, 3), Conv(c2, c2, 3), nn.Conv2d(c2, 4, 1)) for x in ch)
        self.cv3 = nn.ModuleList(nn.Sequential(nn.Sequential(Conv(x, x, 3, g=x), Conv(x, c3, 1)),
                                               nn.Sequential(Conv(c3, c3, 3, g=c3), Conv(c3, c3, 1)), nn.Conv2d(c3, nc, 1)) for x in ch)
        for a, b, s in zip(self.cv2, self.cv3, STRIDES):
            a[-1].bias.data[:] = 1.0                                   # start with boxes 2 cells wide
            b[-1].bias.data[:] = math.log(5 / nc / (IMG / s) ** 2)    # and a low chance of an object per cell
        self.one2one_cv2, self.one2one_cv3 = copy.deepcopy(self.cv2), copy.deepcopy(self.cv3)

    def forward(self, xs):
        o2m = [torch.cat((self.cv2[i](x), self.cv3[i](x)), 1) for i, x in enumerate(xs)]
        o2o = [torch.cat((self.one2one_cv2[i](x), self.one2one_cv3[i](x)), 1) for i, x in enumerate(xs)]
        return o2m, o2o


class YOLO26Tiny(nn.Module):
    """yolo26.yaml at depth 0.50, width 0.125, max_channels 512 (YOLO26n is 0.50, 0.25, 1024)."""
    def __init__(self, nc=NC):
        super().__init__()
        self.b0, self.b1 = Conv(3, 8, 3, 2), Conv(8, 16, 3, 2)                 # P1/2, P2/4
        self.b2 = C3k2(16, 32, 1, False, 0.25)
        self.b3, self.b4 = Conv(32, 32, 3, 2), C3k2(32, 64, 1, False, 0.25)    # P3/8
        self.b5, self.b6 = Conv(64, 64, 3, 2), C3k2(64, 64, 1, True)           # P4/16
        self.b7, self.b8 = Conv(64, 64, 3, 2), C3k2(64, 64, 1, True)           # P5/32
        self.b9, self.b10 = SPPF(64, 64, 5, 3, True), C2PSA(64, 64, 1)
        self.h13 = C3k2(128, 64, 1, True)
        self.h16 = C3k2(128, 32, 1, True)
        self.h17, self.h19 = Conv(32, 32, 3, 2), C3k2(96, 64, 1, True)
        self.h20, self.h22 = Conv(64, 64, 3, 2), C3k2(128, 64, 1, True, 0.5, attn=True)
        self.detect = Detect(nc, (32, 64, 64))

    def forward(self, x):
        up = lambda t: F.interpolate(t, scale_factor=2, mode='nearest')
        x = self.b2(self.b1(self.b0(x)))
        p3 = self.b4(self.b3(x))
        p4 = self.b6(self.b5(p3))
        p5 = self.b10(self.b9(self.b8(self.b7(p4))))
        h13 = self.h13(torch.cat((up(p5), p4), 1))
        h16 = self.h16(torch.cat((up(h13), p3), 1))
        h19 = self.h19(torch.cat((self.h17(h16), h13), 1))
        h22 = self.h22(torch.cat((self.h20(h19), p5), 1))
        return self.detect([h16, h19, h22])`;

  py.scenes = `def hsv(h, s, v):
    h = (h % 1.0) * 6
    i, f = int(h) % 6, h - int(h)
    p, q, t = v * (1 - s), v * (1 - s * f), v * (1 - s * (1 - f))
    return np.array([(v, t, p), (q, v, p), (p, v, t), (p, q, v), (t, p, v), (v, p, q)][i])


def corners(cls, cx, cy, size, ang):
    """A triangle's, square's or star's corners."""
    R = size / 2
    if cls == 1:
        a, r = ang - np.pi / 2 + 2 * np.pi * np.arange(3) / 3, np.full(3, R)
    elif cls == 2:
        a, r = ang + np.pi / 4 + np.pi / 2 * np.arange(4), np.full(4, R)
    else:
        a, r = ang - np.pi / 2 + np.pi * np.arange(10) / 5, np.where(np.arange(10) % 2, R / 2, R)
    return np.stack([cx + r * np.cos(a), cy + r * np.sin(a)], 1)


def polygon_sdf(px, py, V):
    """Signed distance from each pixel to a polygon: negative inside."""
    d2, s = (px - V[0, 0]) ** 2 + (py - V[0, 1]) ** 2, np.ones_like(px)
    for i in range(len(V)):
        ex, ey = V[i - 1] - V[i]
        wx, wy = px - V[i, 0], py - V[i, 1]
        h = np.clip((wx * ex + wy * ey) / (ex * ex + ey * ey), 0, 1)
        d2 = np.minimum(d2, (wx - ex * h) ** 2 + (wy - ey * h) ** 2)
        c1, c2, c3 = py >= V[i, 1], py < V[i - 1, 1], ex * wy > ey * wx
        s = np.where((c1 & c2 & c3) | (~c1 & ~c2 & ~c3), -s, s)
    return s * np.sqrt(d2)


def box_iou(a, b):
    """IoU of every box in a (N, 4) with every box in b (M, 4), as x1, y1, x2, y2."""
    lt, rb = torch.max(a[:, None, :2], b[None, :, :2]), torch.min(a[:, None, 2:], b[None, :, 2:])
    inter = (rb - lt).clamp(0).prod(2)
    area = lambda q: (q[:, 2:] - q[:, :2]).clamp(0).prod(1)
    return inter / (area(a)[:, None] + area(b)[None, :] - inter + 1e-9)


def random_scene(rng, n_shapes=None):
    """One image (3, 128, 128) in [0, 1] and its labels: a two-colour gradient, up to 4 thin lines, 1 to 7 shapes."""
    yy, xx = np.mgrid[0:IMG, 0:IMG] + 0.5
    h0, s0, v0 = rng.random(), rng.uniform(0, 0.35), rng.uniform(0.12, 0.95)
    c0, c1 = hsv(h0, s0, v0), hsv(h0 + rng.uniform(-0.08, 0.08), s0, np.clip(v0 + rng.uniform(-0.3, 0.3), 0.05, 1))
    a = rng.uniform(0, 2 * np.pi)
    grad = lambda x, y: np.clip(((x - IMG / 2) * np.cos(a) + (y - IMG / 2) * np.sin(a)) / IMG + 0.5, 0, 1)
    img = c0[:, None, None] + (c1 - c0)[:, None, None] * grad(xx, yy)
    paint = lambda d, col: img + (col[:, None, None] - img) * np.clip(0.5 - d, 0, 1)   # 1-pixel anti-aliasing
    for _ in range(rng.integers(0, 5)):                       # clutter: thin lines that are not objects
        x0, y0, ang, length = rng.uniform(0, IMG), rng.uniform(0, IMG), rng.uniform(0, 2 * np.pi), rng.uniform(20, 90)
        bx, by = length * np.cos(ang), length * np.sin(ang)
        h = np.clip(((xx - x0) * bx + (yy - y0) * by) / (bx * bx + by * by), 0, 1)
        img = paint(np.hypot(xx - x0 - bx * h, yy - y0 - by * h) - rng.uniform(0.4, 1.1), hsv(rng.random(), rng.uniform(0, 0.7), rng.uniform(0.15, 1)))
    boxes, classes, target = [], [], n_shapes or rng.integers(1, 8)
    for _ in range(14):
        if len(boxes) == target:
            break
        cls, size = rng.integers(NC), np.exp(rng.uniform(np.log(7), np.log(44)))
        cx, cy = rng.uniform(0, 1, 2) * (IMG - size / 2) + size / 4
        ang, col = rng.uniform(0, 2 * np.pi), hsv(rng.random(), rng.uniform(0.45, 1), rng.uniform(0.4, 1))
        bg = c0 + (c1 - c0) * grad(cx, cy)
        if abs((col - bg) @ [0.299, 0.587, 0.114]) < 0.12 and np.linalg.norm(col - bg) < 0.4:
            continue                                          # it would not stand out from the background
        V = corners(cls, cx, cy, size, ang) if cls else None
        box = np.clip([cx - size / 2, cy - size / 2, cx + size / 2, cy + size / 2] if cls == 0 else [*V.min(0), *V.max(0)], 0, IMG)
        if boxes and box_iou(torch.tensor(np.array([box])), torch.tensor(np.array(boxes))).max() >= 0.35:
            continue                                          # keep overlaps moderate
        img = paint(np.hypot(xx - cx, yy - cy) - size / 2 if cls == 0 else polygon_sdf(xx, yy, V), col)
        boxes.append(box)
        classes.append(cls)
    img = np.clip(img + rng.uniform(0, 0.05) * rng.standard_normal(img.shape), 0, 1)   # sensor noise
    return (torch.tensor(np.round(img * 255) / 255, dtype=torch.float32),
            torch.tensor(np.array(boxes), dtype=torch.float32).view(-1, 4), torch.tensor(classes, dtype=torch.long))


def make_batch(rng, n):
    """n fresh scenes, with labels padded to 7 per image (mask marks the real ones)."""
    scenes = [random_scene(rng) for _ in range(n)]
    gt_box, gt_cls, mask = torch.zeros(n, 7, 4), torch.zeros(n, 7, dtype=torch.long), torch.zeros(n, 7, dtype=torch.bool)
    for i, (_, b, c) in enumerate(scenes):
        gt_box[i, :len(b)], gt_cls[i, :len(c)], mask[i, :len(c)] = b, c, True
    return torch.stack([s[0] for s in scenes]), gt_cls, gt_box, mask`;

  py.loss = `def anchor_points():
    pts, st = [], []
    for s in STRIDES:
        n = IMG // s
        gy, gx = torch.meshgrid(torch.arange(n) + 0.5, torch.arange(n) + 0.5, indexing='ij')
        pts.append(torch.stack((gx, gy), -1).view(-1, 2))
        st.append(torch.full((n * n, 1), float(s)))
    return torch.cat(pts), torch.cat(st)            # 336 cell centres (in cells) and their strides


ANC, ST = anchor_points()


def flat(maps):
    return torch.cat([m.flatten(2) for m in maps], 2).permute(0, 2, 1)   # (B, 336, 4 + NC)


def decode(p):
    """No DFL: the first 4 numbers are the distances from the cell centre to the box's left, top, right, bottom."""
    d = p[..., :4]
    return torch.cat((ANC - d[..., :2], ANC + d[..., 2:]), -1) * ST, p[..., 4:]


def ciou(b1, b2, eps=1e-7):
    """Complete IoU: the overlap, minus penalties for centre distance and aspect ratio (1 = perfect)."""
    x1, y1, x2, y2 = b1.unbind(-1)
    X1, Y1, X2, Y2 = b2.unbind(-1)
    w1, h1, w2, h2 = x2 - x1, (y2 - y1).clamp_min(eps), X2 - X1, (Y2 - Y1).clamp_min(eps)
    inter = (torch.minimum(x2, X2) - torch.maximum(x1, X1)).clamp(0) * (torch.minimum(y2, Y2) - torch.maximum(y1, Y1)).clamp(0)
    iou = inter / (w1.clamp(0) * h1 + w2.clamp(0) * h2 - inter + eps)
    c2 = (torch.maximum(x2, X2) - torch.minimum(x1, X1)) ** 2 + (torch.maximum(y2, Y2) - torch.minimum(y1, Y1)) ** 2 + eps
    rho2 = ((X1 + X2 - x1 - x2) ** 2 + (Y1 + Y2 - y1 - y2) ** 2) / 4
    v = (4 / math.pi ** 2) * (torch.atan(w2 / h2) - torch.atan(w1 / h1)).pow(2)
    with torch.no_grad():
        alpha = v / (v - iou + (1 + eps))
    return iou - (rho2 / c2 + v * alpha)


@torch.no_grad()
def assign(scores, boxes, gt_cls, gt_box, mask, topk, stal=True, alpha=0.5, beta=6.0, eps=1e-9):
    """Task-aligned assignment: each object takes the top-k cells by score^0.5 * IoU^6 among its candidate cells.
    STAL: boxes under 16 px are widened to 16 px when choosing candidates, so tiny objects still get cells."""
    B, A, C = scores.shape
    M = gt_box.shape[1]
    cb = gt_box
    if stal:
        c, half = (gt_box[..., :2] + gt_box[..., 2:]) / 2, (gt_box[..., 2:] - gt_box[..., :2]).clamp_min(16) / 2
        cb = torch.cat((c - half, c + half), -1)
    anc = ANC * ST
    inside = torch.cat((anc[None, None] - cb[:, :, None, :2], cb[:, :, None, 2:] - anc[None, None]), -1).amin(-1) > eps
    cand = inside & mask[:, :, None]
    sc = scores.gather(2, gt_cls[:, None, :].expand(-1, A, -1)).permute(0, 2, 1)
    ious = ciou(gt_box[:, :, None, :], boxes[:, None, :, :]).clamp(0) * cand
    metric = sc.pow(alpha) * ious.pow(beta) * cand
    pos = torch.zeros_like(cand).scatter_(-1, metric.topk(topk, dim=-1).indices, True) & cand
    multi = pos.sum(1, keepdim=True) > 1                      # a cell claimed twice keeps the object it overlaps most
    pos = torch.where(multi, F.one_hot(ious.argmax(1), M).permute(0, 2, 1).bool() & cand, pos)
    fg, gi = pos.any(1), pos.float().argmax(1)
    t_cls, t_box = gt_cls.gather(1, gi), gt_box.gather(1, gi[..., None].expand(-1, -1, 4))
    metric = metric * pos
    norm = (metric * (ious * pos).amax(-1, keepdim=True) / (metric.amax(-1, keepdim=True) + eps)).amax(1)
    return t_box, F.one_hot(t_cls, C).float() * (fg.float() * norm)[..., None], fg


def det_loss(maps, gt_cls, gt_box, mask, topk):
    boxes, logits = decode(flat(maps))
    t_box, t_score, fg = assign(logits.sigmoid(), boxes.detach(), gt_cls, gt_box, mask, topk)
    norm = t_score.sum().clamp_min(1)
    l_cls = F.binary_cross_entropy_with_logits(logits, t_score, reduction='sum') / norm
    l_box = ((1 - ciou(boxes, t_box)) * t_score.sum(-1) * fg).sum() / norm
    return 7.5 * l_box + 0.5 * l_cls


def prog_weight(epoch, epochs):
    """ProgLoss: the one-to-many head's share of the loss falls from 0.8 to 0.1; the one-to-one head gets the rest."""
    return max(1 - epoch / max(epochs - 1, 1), 0) * 0.7 + 0.1`;

  py.musgd = `def newton_schulz(G, steps=5):
    """Muon's orthogonalisation: pushes every singular value of G toward 1, keeping its directions."""
    a, b, c = 3.4445, -4.7750, 2.0315
    X = G / (G.norm() + 1e-7)
    tall = X.shape[0] > X.shape[1]
    if tall:
        X = X.T
    for _ in range(steps):
        A = X @ X.T
        X = a * X + (b * A + c * A @ A) @ X
    return X.T if tall else X


class MuSGD(torch.optim.Optimizer):
    """SGD with Nesterov momentum; each weight matrix steps half by SGD, half by Muon (orthogonalised momentum)."""
    def __init__(self, params, lr=0.01, momentum=0.937, weight_decay=5e-4):
        super().__init__(params, dict(lr=lr, momentum=momentum, weight_decay=weight_decay))

    @torch.no_grad()
    def step(self):
        for g in self.param_groups:
            for p in g['params']:
                if p.grad is None:
                    continue
                d = p.grad + g['weight_decay'] * p if p.ndim > 1 else p.grad
                buf = self.state[p].setdefault('buf', torch.zeros_like(p))
                buf.mul_(g['momentum']).add_(d)
                d = d + g['momentum'] * buf                                   # Nesterov
                if p.ndim > 1 and min(p.shape[0], p[0].numel()) > 1:
                    m = d.reshape(p.shape[0], -1)
                    o = newton_schulz(m) * 0.2 * max(m.shape) ** 0.5           # the Muon step, RMS about 0.2
                    p.add_(0.5 * d + 0.5 * o.view_as(p), alpha=-g['lr'])
                else:
                    p.add_(d, alpha=-g['lr'])`;

  py.detect = `@torch.no_grad()
def detect(model, img, head='o2o', conf=0.25, iou=0.7):
    """Boxes (x1, y1, x2, y2 in pixels), scores and classes for one image."""
    o2m, o2o = model.eval()(img[None])
    boxes, logits = decode(flat(o2o if head == 'o2o' else o2m))
    boxes, scores = boxes[0], logits[0].sigmoid()
    if head == 'o2o':                                   # end to end: the highest scores are the answer, no NMS
        s, i = scores.flatten().topk(100)
        keep = s > conf
        return boxes[i // NC][keep], s[keep], (i % NC)[keep]
    s, c = scores.max(1)                                # one-to-many: best class per cell, then NMS
    b, s, c = boxes[s > conf], s[s > conf], c[s > conf]
    order, kept = s.argsort(descending=True), []
    while len(order):
        i, rest = order[0], order[1:]
        kept.append(i)
        order = rest[(box_iou(b[i][None], b[rest])[0] <= iou) | (c[rest] != c[i])]
    kept = torch.stack(kept) if kept else torch.zeros(0, dtype=torch.long)
    return b[kept], s[kept], c[kept]`;

  py.train = `def train(model, epochs=40, iters=120, batch=128, seed=0):
    """The browser's weights: 40 epochs of 120 steps of 128 fresh scenes (on a GPU, with an EMA of the weights)."""
    rng, opt = np.random.default_rng(seed), MuSGD(model.parameters())
    for step in range(4_800):
        epoch = step // iters
        opt.param_groups[0]['lr'] = 0.01 * min(1, step / iters) * (1 - 0.99 * step / (epochs * iters))
        x, gt_cls, gt_box, mask = make_batch(rng, batch)
        o2m, o2o = model.train()(x)
        w = prog_weight(epoch, epochs)
        loss = (w * det_loss(o2m, gt_cls, gt_box, mask, 10) + (1 - w) * det_loss(o2o, gt_cls, gt_box, mask, 1)) * 64
        opt.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 10.0)
        opt.step()`;

  py.model = () => `${py.layers}


${py.scenes}


${py.loss}


${py.musgd}


${py.train}


${py.detect}


model = YOLO26Tiny()
train(model)
img, true_boxes, true_classes = random_scene(np.random.default_rng(1))
for box, score, cls in zip(*detect(model, img)):
    print(NAMES[cls], round(score.item(), 2), [round(v) for v in box.tolist()])`;

  // a module name as Python: 'b2.m.0.cv1' → model.b2.m[0].cv1
  const pyPath = (n) => 'model' + n.split('.').map((p) => (/^\d+$/.test(p) ? `[${p}]` : `.${p}`)).join('');
  // the head's last layers are plain nn.Conv2d with a bias (no BatchNorm, no activation)
  const plainConv = (n) => /^detect\.(one2one_)?cv[23]\.\d\.2$/.test(n);
  py.conv = (rec) => {
    const L = rec.L, x = rec.inp, o = rec.out;
    if (plainConv(rec.name)) return `conv = ${pyPath(rec.name)}                  # nn.Conv2d(${L.ci}, ${L.co}, 1): a plain 1x1 conv with a bias
x = torch.randn(1, ${x.c}, ${x.h}, ${x.w})
y = conv(x)                                      # (1, ${o.c}, ${o.h}, ${o.w}): ${L.co === 4 ? 'left, top, right, bottom distances' : 'one logit per class'} at every cell
print(torch.allclose(y, torch.einsum('oc,bchw->bohw', conv.weight[:, :, 0, 0], x) + conv.bias[None, :, None, None], atol=1e-5))`;
    return `conv = ${pyPath(rec.name)}.eval()               # Conv(${L.ci * rec.g}, ${L.co}, k=${L.k}, s=${rec.s}${rec.g > 1 ? `, g=${rec.g}` : ''})${rec.act ? '' : ', no activation'}
x = torch.randn(1, ${x.c}, ${x.h}, ${x.w})                  # its input here: ${x.c} channels of ${x.h} x ${x.w}
y = conv(x)
print(y.shape)                                    # (1, ${o.c}, ${o.h}, ${o.w})
# what the browser runs: BatchNorm folded into the kernel and a bias
s = conv.bn.weight / (conv.bn.running_var + conv.bn.eps).sqrt()
w, b = conv.conv.weight * s[:, None, None, None], conv.bn.bias - conv.bn.running_mean * s
z = F.conv2d(x, w, b, stride=${rec.s}, padding=${L.k >> 1}, groups=${rec.g})
print(torch.allclose(${rec.act ? 'F.silu(z)' : 'z'}, y, atol=1e-5))         # True`;
  };

  // ------------------------------------------------------------------ drawing helpers
  // a channel of a tensor as a small image (one pixel per number), coloured on the diverging ramp
  const heatCache = new WeakMap();
  function heatCanvas(t, c, m) {
    let per = heatCache.get(t);
    if (!per) { per = new Map(); heatCache.set(t, per); }
    const key = c + ':' + m.toFixed(5) + ':' + (theme.dark ? 1 : 0);
    let cv = per.get(key);
    if (!cv) {
      cv = document.createElement('canvas'); cv.width = t.w; cv.height = t.h;
      const g = cv.getContext('2d'), id = g.createImageData(t.w, t.h), lut = theme.div, n = t.h * t.w;
      for (let p = 0; p < n; p++) { const i = Math.round((clamp(t.d[c * n + p] / m, -1, 1) + 1) * 128) * 3; id.data.set([lut[i], lut[i + 1], lut[i + 2], 255], 4 * p); }
      g.putImageData(id, 0, 0);
      if (per.size > 80) per.clear();
      per.set(key, cv);
    }
    return cv;
  }
  const maxAbs = (t) => { let m = 1e-6; for (let i = 0; i < t.d.length; i++) m = Math.max(m, Math.abs(t.d[i])); return m; };
  // square cells inside r, from its top-left corner (as the ArchView lays out a square grid)
  const sq = (r, rows, cols) => { const cs = Math.min(r.w / cols, r.h / rows); return { x: r.x, y: r.y, w: cs * cols, h: cs * rows, cs }; };
  function drawMap(ctx, t, c, r, m) {
    const q = sq(r, t.h, t.w);
    ctx.imageSmoothingEnabled = false;
    ctx.drawImage(heatCanvas(t, c, m || maxAbs(t)), q.x, q.y, q.w, q.h);
    ctx.strokeStyle = theme.c('line-2'); ctx.lineWidth = 0.8; ctx.strokeRect(q.x - 0.5, q.y - 0.5, q.w + 1, q.h + 1);
    return q;
  }
  // up to n channels of a tensor in a grid of square tiles: the columns that make the tiles largest in r
  function tiles(t, n, r) {
    const k = Math.min(n, t.c);
    let best = null;
    for (let cols = 1; cols <= k; cols++) {
      const rows = Math.ceil(k / cols), s = Math.min(r.w / cols, r.h / rows);
      if (!best || s > best.s) best = { cols, rows, s };
    }
    return Object.assign(best, { k });
  }
  // lay: a layout fixed in advance (so a grid of the same rows and columns lands exactly on the tiles)
  function drawTiles(ctx, t, r, n, m, lay) {
    const T = lay || tiles(t, n, r), s = Math.min(r.w / T.cols, r.h / T.rows), mm = m || maxAbs(t), gap = Math.max(1, s * 0.06);
    ctx.imageSmoothingEnabled = false;
    for (let i = 0; i < T.k; i++) ctx.drawImage(heatCanvas(t, i, mm), r.x + (i % T.cols) * s + gap / 2, r.y + Math.floor(i / T.cols) * s + gap / 2, s - gap, s - gap);
    return T;
  }
  // a weight matrix (rows × cols) as an image, cached per layer
  const matCache = new WeakMap();
  function drawMatrix(ctx, key, rows, cols, get, r) {
    let cv = matCache.get(key);
    if (!cv || cv.dark !== theme.dark) {
      cv = document.createElement('canvas'); cv.width = cols; cv.height = rows; cv.dark = theme.dark;
      let m = 1e-9;
      for (let i = 0; i < rows; i++) for (let j = 0; j < cols; j++) m = Math.max(m, Math.abs(get(i, j)));
      const g = cv.getContext('2d'), id = g.createImageData(cols, rows), lut = theme.div;
      for (let i = 0; i < rows; i++) for (let j = 0; j < cols; j++) { const q = Math.round((clamp(get(i, j) / m, -1, 1) + 1) * 128) * 3; id.data.set([lut[q], lut[q + 1], lut[q + 2], 255], 4 * (i * cols + j)); }
      g.putImageData(id, 0, 0);
      matCache.set(key, cv);
    }
    ctx.imageSmoothingEnabled = false;
    ctx.drawImage(cv, r.x, r.y, r.w, r.h);
    ctx.strokeStyle = theme.c('line-2'); ctx.lineWidth = 0.8; ctx.strokeRect(r.x - 0.5, r.y - 0.5, r.w + 1, r.h + 1);
  }
  // how strongly anything fires at each position: the mean |value| over a tensor's channels, as an image
  const energyCache = new WeakMap();
  function drawEnergy(ctx, t, r) {
    let cv = energyCache.get(t);
    if (!cv || cv.dark !== theme.dark) {
      const n = t.h * t.w, e = new Float32Array(n);
      for (let c = 0; c < t.c; c++) for (let p = 0; p < n; p++) e[p] += Math.abs(t.d[c * n + p]) / t.c;
      const m = Math.max(...e, 1e-9), lut = theme.seq;
      cv = document.createElement('canvas'); cv.width = t.w; cv.height = t.h; cv.dark = theme.dark;
      const g = cv.getContext('2d'), id = g.createImageData(t.w, t.h);
      for (let p = 0; p < n; p++) { const i = Math.round(Math.sqrt(e[p] / m) * 255) * 3; id.data.set([lut[i], lut[i + 1], lut[i + 2], 255], 4 * p); }
      g.putImageData(id, 0, 0);
      energyCache.set(t, cv);
    }
    const s = Math.min(r.w, r.h);
    ctx.imageSmoothingEnabled = false;
    ctx.drawImage(cv, r.x + (r.w - s) / 2, r.y + (r.h - s) / 2, s, s);
    ctx.strokeStyle = theme.c('line-2'); ctx.lineWidth = 0.8; ctx.strokeRect(r.x + (r.w - s) / 2 - 0.5, r.y + (r.h - s) / 2 - 0.5, s + 1, s + 1);
  }
  // the maps of a stage, one per scale, in the grid of rows and columns that makes them largest (a row, a column,
  // or two columns in a box about as wide as it is tall)
  function drawPyramid(ctx, S, ts, r, labels) {
    const n = ts.length, gap = 6, cap = 11;
    let best = null;
    for (let cols = 1; cols <= n; cols++) {
      const rows = Math.ceil(n / cols), cw = r.w / cols, ch = r.h / rows, s = Math.min(cw - gap, ch - gap - cap);
      if (!best || s > best.s + 0.5) best = { cols, rows, cw, ch, s };
    }
    const { cols, cw, ch } = best;
    for (let i = 0; i < n; i++) {
      // a last row that is not full is centred
      const row = Math.floor(i / cols), inRow = Math.min(cols, n - row * cols), shift = ((cols - inRow) * cw) / 2;
      const q = { x: r.x + shift + (i % cols) * cw + gap / 2, y: r.y + row * ch, w: cw - gap, h: ch - gap };
      const inner = { x: q.x, y: q.y + cap, w: q.w, h: q.h - cap };
      if (inner.h > 8 && inner.w > 8) { drawEnergy(ctx, ts[i], inner); S.text(ctx, labels[i], q.x + q.w / 2, q.y + cap / 2, { size: 8.5, align: 'center', color: theme.c('muted'), maxW: q.w }); }
    }
  }
  function drawImage(ctx, L, r) {
    const c = L.imageCanvas(), s = Math.min(r.w, r.h);
    if (!c) return { x: r.x, y: r.y, s };
    ctx.imageSmoothingEnabled = false;
    ctx.drawImage(c, r.x, r.y, s, s);
    ctx.strokeStyle = theme.c('line-2'); ctx.lineWidth = 1; ctx.strokeRect(r.x - 0.5, r.y - 0.5, s + 1, s + 1);
    return { x: r.x, y: r.y, s };
  }
  function drawBoxes(ctx, dets, q, o = {}) {
    const k = q.s / Y.IMG;
    for (const d of dets) {
      const [x1, y1, x2, y2] = d.box;
      ctx.strokeStyle = d.suppressedBy ? theme.c('muted', 0.55) : CLS(d.cls); ctx.lineWidth = o.lw || 1.6;
      if (d.suppressedBy) ctx.setLineDash([3, 3]);
      ctx.strokeRect(q.x + x1 * k, q.y + y1 * k, (x2 - x1) * k, (y2 - y1) * k);
      ctx.setLineDash([]);
    }
  }
  const shape = (t) => `${t.c} × ${t.h} × ${t.w}`;
  // a layer's number in yolo26.yaml from its module name: 'b9' → 9, 'h16' → 16, 'u11' → 11
  const layerNo = (name) => name.replace(/^[bhu]/, '');
  const paramsOf = (L, prefix) => Object.values(L.net.P).filter((p) => p.name === prefix || p.name.startsWith(prefix + '.')).reduce((s, p) => s + p.w.length + p.b.length, 0);

  // ------------------------------------------------------------------ the diagram
  function build(L, size) {
    const X = { wide: size.w / size.h >= 1.05 }, Dp = NA.deep;
    const res = L.result;
    if (!res) return { id: 'yolo', title: 'YOLO26 detector', sub: 'loading', inner: (ctx, r, S) => S.text(ctx, 'Running the detector…', r.x, r.y + 20, { color: theme.c('muted') }), about: ['The detector is loading.'] };
    const F = res.F, K = F.K, R = F.R;
    // which conv made a tensor: the way back from any number to its weighted sum
    const prod = new Map();
    for (const r of Object.values(R)) prod.set(r.out, r);
    const RGB = ['red', 'green', 'blue'];

    // ---- any number of any tensor → the node that explains where it came from
    function resolve(t, c, y, x, label) {
      const i = (c * t.h + y) * t.w + x, v = t.d[i];
      const rec = prod.get(t);
      if (rec) return convNeuron(rec, c, y, x);
      if (t === K.input) return Dp.stored(X, v, `${RGB[c]} value of pixel (${x}, ${y})`);
      if (t.from) return resolve(t.from, c + t.c0, y, x, label);
      if (t.chmap) return resolve(t.chmap.src, t.chmap.map[c], y, x, label);
      if (t.up) return Object.assign(resolve(t.up, c, y >> 1, x >> 1, label), { sub: `copied by the 2× upsample from (${x >> 1}, ${y >> 1})` });
      if (t.parts) { let off = 0; for (const p of t.parts) { if (c < off + p.c) return resolve(p, c - off, y, x, label); off += p.c; } }
      if (t.sum) return sumNode(t, c, y, x);
      if (t.pool) return poolNode(t, c, y, x);
      if (t.attn) return attnSum(t, c, y, x);
      return Dp.stored(X, v, label || `channel ${c} at (${x}, ${y})`);
    }
    // a convolution's output number: its weighted sum over a k x k window of every input channel
    function convNeuron(rec, oc, oy, ox) {
      const L0 = rec.L, T = Y.convTerms(rec, oc, oy, ox), k = L0.k, dw = rec.g > 1;
      const where = (q) => (q.inside ? `channel ${q.ic}, (${q.ix}, ${q.iy})` : `padding at (${q.ix}, ${q.iy})`);
      return Dp.neuron(X, {
        title: `${rec.name} · channel ${oc} at (${ox}, ${oy})`, sub: `${T.terms.length} inputs${rec.act ? ', SiLU' : ''}`,
        terms: T.terms.map((q) => ({ a: q.a, w: q.w })), bias: T.bias, act: rec.act ? SILU : null,
        inLabel: (j) => where(T.terms[j]), wLabel: (j) => `kernel[${oc}, ${T.terms[j].ic % L0.ci}, ${Math.floor((j % (k * k)) / k)}, ${j % k}]`,
        inputNode: (j) => { const q = T.terms[j]; return q.inside ? resolve(rec.inp, q.ic, q.iy, q.ix) : Dp.stored(X, 0, 'zero padding outside the map'); },
        rowNames: [`inputs: a ${k} × ${k} window of ${dw ? 'its own input channel' : `all ${L0.ci} input channels`} (${T.terms.length} numbers)`, 'the kernel’s weights (BatchNorm folded in)', 'products wᵢ·aᵢ'],
        vars: { w: 'w', a: 'patch', b: 'b' }, codeWhat: `as one output of ${rec.name}`,
        codeHead: [`w = kernel[${oc}].flatten()       # ${T.terms.length} weights`, `patch = F.pad(x, [${k >> 1}] * 4)[${dw ? `${oc}:${oc + 1}` : ':'}, ${oy * rec.s}:${oy * rec.s + k}, ${ox * rec.s}:${ox * rec.s + k}].flatten()`],
        about: [`One number of ${rec.name}'s output: channel ${oc} at column ${ox}, row ${oy}. A ${k} × ${k} convolution${rec.s > 1 ? ` with stride ${rec.s}` : ''} computes every output number the same way: lay the kernel over a ${k} × ${k} window of the input${dw ? ' (a depthwise conv looks only at its own channel)' : `, across all ${L0.ci} input channels`}, multiply and add. The same ${T.terms.length} weights slide over every position, which is why a conv layer has so few parameters for so many outputs.`, 'Training used BatchNorm after each conv; for inference its scale and shift are folded into the kernel and a bias, so this one weighted sum is all that runs. Zoom into an input bar to see where that input came from.'],
      });
    }
    function sumNode(t, c, y, x) {
      const [a, b] = t.sum, i = (c * t.h + y) * t.w + x;
      return {
        title: 'A residual sum', sub: `${fmt(a.d[i], 3)} + ${fmt(b.d[i], 3)} = ${fmt(t.d[i], 3)}`, tone: 's2',
        inner: (ctx, r, S) => S.lines(ctx, ['out = x + block(x)', `${fmt(t.d[i], 4)} = ${fmt(a.d[i], 4)} + ${fmt(b.d[i], 4)}`, 'Zoom into either side, or into the addition itself.'], S.place(r, [0, 0.02, 1, 0.3]), { mono: true, strong: (k) => k === 0 }),
        children: [
          Object.assign(resolve(a, c, y, x), { id: 'a', rect: W2(X, [0, 0.42, 0.31, 0.5], [0, 0.36, 1, 0.18]) }),
          Object.assign(resolve(b, c, y, x), { id: 'b', rect: W2(X, [0.345, 0.42, 0.31, 0.5], [0, 0.58, 1, 0.18]) }),
          Object.assign(Dp.fadd(X, a.d[i], b.d[i]), { id: 'add', rect: W2(X, [0.69, 0.42, 0.31, 0.5], [0, 0.8, 1, 0.18]), title: 'The addition' }),
        ],
        about: ['A shortcut: the block’s output is added to its input, number by number. The block only has to learn a correction, and gradients flow straight through the addition during training, which is what lets deep networks train at all (ResNet, 2015).'],
        code: 'y = x + block(x)          # same shape: every number is its own sum',
      };
    }
    function poolNode(t, c, y, x) {
      const src = t.pool, n = src.h * src.w, win = [];
      for (let dy = -2; dy <= 2; dy++) for (let dx = -2; dx <= 2; dx++) { const yy = y + dy, xx = x + dx; if (yy >= 0 && yy < src.h && xx >= 0 && xx < src.w) win.push({ v: src.d[c * n + yy * src.w + xx], yy, xx }); }
      const best = win.reduce((p, q) => (q.v > p.v ? q : p));
      return {
        title: 'A 5 × 5 max-pool', sub: `max of ${win.length} = ${fmt(best.v, 3)}`, tone: 's4',
        summary: (ctx, r, S) => S.bars(ctx, Float32Array.from(win.map((q) => q.v)), r),
        inner: (ctx, r, S) => {
          S.lines(ctx, [`out = max of the ${win.length} numbers in a 5 × 5 window = ${fmt(best.v, 4)}`, `the winner sits at (${best.xx}, ${best.yy})`], S.place(r, [0, 0.02, 1, 0.16]), { mono: true, strong: (k) => k === 0 });
          const br = S.place(r, [0, 0.3, 0.66, 0.6]);
          S.caption(ctx, 'the window, largest marked', br);
          S.bars(ctx, Float32Array.from(win.map((q) => q.v)), br, { color: (k) => (win[k] === best ? theme.c('s4') : theme.c('muted', 0.6)) });
        },
        children: [Object.assign(resolve(src, c, best.yy, best.xx), { id: 'max', rect: W2(X, [0.7, 0.3, 0.3, 0.6], [0.6, 0.3, 0.4, 0.6]) })],
        about: ['Max-pooling keeps the largest number in each window: a feature that fired anywhere nearby still counts. SPPF chains three of them, so their outputs see windows of 5, 9 and 13 cells: on this coarsest 4 × 4 grid, the whole picture.', 'A comparison is a subtraction whose sign is read: the circuit inside is the same adder as the one under an addition.'],
        code: 'y = F.max_pool2d(x, kernel_size=5, stride=1, padding=2)   # same size, every number the max of its window',
      };
    }
    function attnSum(t, c, y, x) {
      const A = K[t.attn], h = Math.floor(c / A.hd), d = c % A.hd, N = A.N, i = y * Math.round(Math.sqrt(N)) + x, base = h * (2 * A.kd + A.hd) * N;
      const terms = Array.from({ length: N }, (_, j) => ({ a: A.qkv.d[base + (2 * A.kd + d) * N + j], w: A.maps[h][i * N + j] }));
      return Dp.neuron(X, {
        kind: 'dot', title: `Attention output, channel ${c} at (${x}, ${y})`, sub: `${N} positions, weighted`, terms, bias: null, act: null,
        inLabel: (j) => `the value at position ${j}`, wLabel: (j) => `attention from (${x}, ${y}) to position ${j}`,
        inputNode: (j) => resolve(A.v, c, Math.floor(j / Math.round(Math.sqrt(N))), j % Math.round(Math.sqrt(N))),
        rowNames: ['values v at each of the 16 positions', 'attention weights (a softmax: they add up to 1)', 'products'], vars: { w: 'att[i]', a: 'v[d]', b: 'b' },
        about: [`Self-attention: position (${x}, ${y}) collects information from every position of the map, weighted by how well its query matches their keys. The weights add up to 1, so the result is a weighted average of the values.`],
      });
    }
    // a channel of a tensor, every number openable
    function channelNode(t, c, title, about) {
      const m = maxAbs(t), rect = W2(X, [0, 0.12, 0.62, 0.86], [0, 0.14, 1, 0.84]);
      return {
        title, sub: `${t.h} × ${t.w} numbers`,
        summary: (ctx, r) => drawMap(ctx, t, c, r, m),
        inner: (ctx, r, S) => {
          S.lines(ctx, [`range ${fmt(Math.min(...t.d.subarray(c * t.h * t.w, (c + 1) * t.h * t.w)), 3)} to ${fmt(Math.max(...t.d.subarray(c * t.h * t.w, (c + 1) * t.h * t.w)), 3)}`, 'Zoom into any cell to open the number.'], S.place(r, [0, 0.0, 1, 0.1]), { mono: true, size: S.u(r, 8, 12) });
          drawMap(ctx, t, c, S.place(r, rect), m);
        },
        grids: [{ id: 'px', rect, rows: t.h, cols: t.w, square: true, make: (i, j) => resolve(t, c, i, j) }],
        about: about || [`One channel of ${t.c}: one detector of some pattern, evaluated at every position of the ${t.h} × ${t.w} grid. Blue is positive, orange negative.`],
      };
    }
    /* A tensor's channels as tiles, each opening that channel. The layout is chosen once against the size of an
       open level (what build() is given), and both the drawing and the grid use it, so cells land on tiles. */
    const tileLayout = (t, n, f) => tiles(t, n, { x: 0, y: 0, w: f[2] * size.w, h: f[3] * size.h });
    const tileGrid = (t, f, name, lay) => ({ id: 'ch', index: true, rect: f, rows: lay.rows, cols: lay.cols, square: true,
      skip: (i, j) => i * lay.cols + j >= lay.k, make: (i, j) => (i * lay.cols + j < lay.k ? channelNode(t, i * lay.cols + j, `${name} · channel ${i * lay.cols + j}`) : null) });

    // a single Conv (conv + BatchNorm + SiLU)
    function convNode(name, title, rect, extra = {}) {
      const rec = R[name], L0 = rec.L, o = rec.out, m = maxAbs(o), tr = W2(X, [0.5, 0.14, 0.5, 0.84], [0, 0.42, 1, 0.56]), lay = tileLayout(o, 32, tr);
      return Object.assign({
        id: name, title: title || `Conv ${L0.k}×${L0.k}${rec.s > 1 ? ', stride ' + rec.s : ''}`, sub: `${shape(rec.inp)} → ${shape(o)}`, rect, tone: 's1',
        summary: (ctx, r) => drawTiles(ctx, o, r, 8, m),
        inner: (ctx, r, S) => {
          const u = S.u(r, 8, 13);
          S.lines(ctx, [
            `Conv(${L0.ci * rec.g}, ${L0.co}, k=${L0.k}, s=${rec.s}${rec.g > 1 ? `, groups=${rec.g}` : ''})${rec.act ? ' + SiLU' : ''}`,
            `in ${shape(rec.inp)}  →  out ${shape(o)}`,
            `${fmtInt(L0.w.length + L0.b.length)} parameters: ${L0.co} kernels of ${L0.ci} × ${L0.k} × ${L0.k}, and ${L0.co} biases`,
            `each output = ${rec.act ? 'SiLU(' : ''}Σ weights × window + bias${rec.act ? ')' : ''}`,
          ], S.place(r, W2(X, [0, 0.02, 0.47, 0.3], [0, 0, 1, 0.2])), { mono: true, size: u, strong: (k) => k === 0 });
          const kr = S.place(r, W2(X, [0, 0.42, 0.44, 0.5], [0, 0.24, 1, 0.14])), J = L0.ci * L0.k * L0.k;
          S.caption(ctx, `kernels: one row per output channel (${L0.co} × ${J})`, kr, { size: u * 0.8, minW: 160 });
          drawMatrix(ctx, L0, L0.co, J, (i, j) => L0.w[i * J + j], kr);
          const q = S.place(r, tr);
          S.caption(ctx, `output: ${Math.min(32, o.c)} of ${o.c} channels; open one`, q, { size: u * 0.8, minW: 160 });
          drawTiles(ctx, o, q, 32, m, lay);
        },
        grids: [tileGrid(o, tr, name, lay)],
        about: [
          `A convolution with ${L0.co} kernels of ${L0.k} × ${L0.k}${rec.g > 1 ? ', depthwise (each kernel sees only its own channel)' : ` over all ${L0.ci} input channels`}${rec.s > 1 ? `, moving ${rec.s} pixels at a time, so the map shrinks from ${rec.inp.h} × ${rec.inp.w} to ${o.h} × ${o.w}` : ''}. ${rec.act ? 'Then SiLU, z·σ(z), a smooth ReLU that lets small negative values through.' : 'No activation here: the output stays linear.'}`,
          'Each tile on the right is one output channel. Open a tile, then a cell, to see the weighted sum behind that number and, below it, the arithmetic, gates and electrons.',
        ],
        code: py.conv(rec), codeLabel: 'Python · PyTorch · this layer, and the folded version the browser runs',
      }, extra);
    }
    // a block's output shown as 16 tiles that open its channels
    const OUT = () => W2(X, [0.66, 0.16, 0.34, 0.8], [0, 0.66, 1, 0.33]);
    function c3k2Node(name, kind, title, rect) {
      const o = K[name], parts = K[name + '.parts'], inp = K[name + '.in'], m = maxAbs(o), c = R[name + '.cv1'].out.c / 2;
      const tr = OUT(), lay = tileLayout(o, 16, tr);
      const mid = kind === 'bott' ? 'Bottleneck' : kind === 'c3k' ? 'C3k' : 'Bottleneck + PSA';
      const kids = [
        convNode(name + '.cv1', `cv1: 1×1 conv`, W2(X, [0, 0.16, 0.18, 0.36], [0, 0.04, 0.3, 0.26])),
        kind === 'bott' ? bottNode(name + '.m.0', W2(X, [0.22, 0.16, 0.18, 0.36], [0.35, 0.04, 0.3, 0.26]))
          : kind === 'c3k' ? c3kNode(name + '.m.0', W2(X, [0.22, 0.16, 0.18, 0.36], [0.35, 0.04, 0.3, 0.26]))
            : psaPair(name, W2(X, [0.22, 0.16, 0.18, 0.36], [0.35, 0.04, 0.3, 0.26])),
        convNode(name + '.cv2', `cv2: 1×1 conv`, W2(X, [0.44, 0.16, 0.18, 0.36], [0.7, 0.04, 0.3, 0.26])),
      ];
      return {
        id: name, title: title || `${name} · C3k2`, sub: `${shape(inp)} → ${shape(o)}`, rect, tone: 's3',
        summary: (ctx, r) => drawTiles(ctx, o, r, 8, m),
        inner: (ctx, r, S) => {
          const u = S.u(r, 8, 13);
          S.lines(ctx, [`C3k2(${inp.c}, ${o.c})${kind === 'c3k' ? ', with C3k inside' : kind === 'attn' ? ', with attention' : ''}`, `${2 * c} → ${c} + ${c};  ${mid} on one half;  concat ${3 * c} → 1×1 → ${o.c}`],
            S.place(r, W2(X, [0, 0.0, 0.64, 0.12], [0, 0.32, 1, 0.1])), { mono: true, size: u, strong: (k) => k === 0 });
          const b = (k) => S.place(r, kids[k].rect);
          for (let k = 0; k < 2; k++) { const p = b(k), q = b(k + 1); S.arrow(ctx, p.x + p.w + 2, p.y + p.h / 2, q.x - 3, q.y + q.h / 2, { lw: 1.1, head: 5 }); }
          // the split and concatenation drawn as three bars of channels
          const sr = S.place(r, W2(X, [0, 0.62, 0.62, 0.3], [0, 0.44, 1, 0.18]));
          S.caption(ctx, 'the three pieces that are concatenated, mean |value| per channel', sr, { size: u * 0.8, minW: 200 });
          const means = parts.map((p) => { const n = p.h * p.w, out = new Float32Array(p.c); for (let ch = 0; ch < p.c; ch++) { let s = 0; for (let q = 0; q < n; q++) s += Math.abs(p.d[ch * n + q]); out[ch] = s / n; } return out; });
          const all = Float32Array.from(means.flatMap((v) => Array.from(v))), cols = ['s5', 's2', 's3'];
          S.bars(ctx, all, sr, { positive: true, color: (k) => theme.c(cols[k < c ? 0 : k < 2 * c ? 1 : 2], 0.85) });
          ['passed through', `into the ${mid}`, `${mid} output`].forEach((t, k) => S.text(ctx, t, sr.x + (k + 0.5) * (sr.w / 3), sr.y + sr.h + u * 0.9, { size: u * 0.78, align: 'center', color: theme.c(cols[k]), w: 600, maxW: sr.w / 3 - 4 }));
          const q = S.place(r, tr);
          S.caption(ctx, `output: ${Math.min(16, o.c)} of ${o.c} channels`, q, { size: u * 0.8, minW: 120 });
          drawTiles(ctx, o, q, 16, m, lay);
        },
        grids: [tileGrid(o, tr, name, lay)],
        children: kids,
        about: [
          `C3k2 is YOLO26's workhorse block (the "CSP" idea). A 1×1 conv makes ${2 * c} channels; the first half pass straight through, the second half go through a ${mid}${kind === 'c3k' ? ' (a smaller CSP block of two 3×3 bottlenecks)' : ''}, and its result is kept too. All three pieces are concatenated (the bars show how strongly each channel fires) and a second 1×1 conv mixes them into ${o.c} channels.`,
          'Keeping the untouched half next to the processed one gives later layers both the raw and the refined features, and costs half the computation of processing everything.',
        ],
        code: `block = ${pyPath(name)}.eval()        # C3k2(${inp.c}, ${o.c})
x = torch.randn(1, ${inp.c}, ${inp.h}, ${inp.w})
a, b = block.cv1(x).chunk(2, 1)          # ${c} + ${c} channels
m = block.m[0](b)                        # the ${mid} on the second half
y = block.cv2(torch.cat([a, b, m], 1))   # ${3 * c} → ${o.c} channels
print(torch.allclose(y, block(x)), y.shape)`,
      };
    }
    function bottNode(name, rect) {
      const out = R[name + '.cv2'].out, inp = R[name + '.cv1'].inp;
      return {
        id: name, title: 'Bottleneck', sub: 'two 3×3 convs + shortcut', rect, tone: 's2',
        summary: (ctx, r) => drawTiles(ctx, out, r, 4),
        inner: (ctx, r, S) => S.lines(ctx, ['y = x + cv2(cv1(x))', `x: ${shape(inp)}`, 'Open either conv; the addition is in every output number.'], S.place(r, [0, 0.02, 1, 0.2]), { mono: true, strong: (k) => k === 0 }),
        children: [convNode(name + '.cv1', 'cv1: 3×3 conv', W2(X, [0, 0.3, 0.46, 0.66], [0, 0.26, 1, 0.34])), convNode(name + '.cv2', 'cv2: 3×3 conv', W2(X, [0.54, 0.3, 0.46, 0.66], [0, 0.64, 1, 0.34]))],
        about: ['Two 3×3 convolutions whose result is added back to the block’s input: the block learns a correction, not a replacement.'],
        code: `bott = ${pyPath(name)}\nx = torch.randn(1, ${inp.c}, ${inp.h}, ${inp.w})\nprint(torch.allclose(bott(x), x + bott.cv2(bott.cv1(x))))`,
      };
    }
    function c3kNode(name, rect) {
      const inp = R[name + '.cv1'].inp;
      return {
        id: name, title: 'C3k', sub: 'CSP with two bottlenecks', rect, tone: 's2',
        summary: (ctx, r) => drawTiles(ctx, R[name + '.cv3'].out, r, 4),
        inner: (ctx, r, S) => S.lines(ctx, ['y = cv3(cat(m(cv1(x)), cv2(x)))', `x: ${shape(inp)}`], S.place(r, [0, 0.02, 1, 0.14]), { mono: true, strong: (k) => k === 0 }),
        children: [
          convNode(name + '.cv1', 'cv1', W2(X, [0, 0.24, 0.22, 0.34], [0, 0.2, 0.48, 0.18])),
          bottNode(name + '.m.0', W2(X, [0.26, 0.24, 0.22, 0.34], [0.52, 0.2, 0.48, 0.18])),
          bottNode(name + '.m.1', W2(X, [0.52, 0.24, 0.22, 0.34], [0, 0.42, 0.48, 0.18])),
          convNode(name + '.cv2', 'cv2 (skip)', W2(X, [0, 0.62, 0.22, 0.34], [0.52, 0.42, 0.48, 0.18])),
          convNode(name + '.cv3', 'cv3: mix', W2(X, [0.78, 0.42, 0.22, 0.34], [0.25, 0.66, 0.5, 0.2])),
        ],
        about: ['A smaller CSP block inside C3k2: one path runs two bottlenecks, the other is a single 1×1 conv, and cv3 mixes their concatenation. YOLO26 uses it in the deeper C3k2 blocks, where there are fewer positions and more channels.'],
        code: `c3k = ${pyPath(name)}\nx = torch.randn(1, ${inp.c}, ${inp.h}, ${inp.w})\nprint(c3k(x).shape)`,
      };
    }
    // C3k2 with attention: a bottleneck, then a PSA block
    function psaPair(name, rect) {
      return {
        id: name + '.m.0', title: 'Bottleneck + PSA', sub: 'convs, then attention', rect, tone: 's2',
        summary: (ctx, r) => drawTiles(ctx, R[name + '.m.0.1.ffn.1'].out, r, 4),
        inner: (ctx, r, S) => S.lines(ctx, ['y = psa(bottleneck(x))'], S.place(r, [0, 0.02, 1, 0.08]), { mono: true, strong: () => true }),
        children: [bottNode(name + '.m.0.0', W2(X, [0, 0.16, 0.46, 0.8], [0, 0.12, 1, 0.4])), psaNode(name + '.m.0.1', W2(X, [0.54, 0.16, 0.46, 0.8], [0, 0.56, 1, 0.42]))],
        about: ['YOLO26 adds attention to the last block of its neck (layer 22, the coarsest scale): a bottleneck first, then a position-wise self-attention block, so every cell of the 4 × 4 map can use what the others see.'],
        code: `pair = ${pyPath(name)}.m[0]\nprint(pair)`,
      };
    }
    function psaNode(name, rect) {
      const A = K[name + '.attn.att'], mid = K[name + '.mid'], N = A.N, side = Math.round(Math.sqrt(N));
      const ar = W2(X, [0, 0.16, 0.4, 0.8], [0, 0.18, 1, 0.42]);
      let lo = 1, hi = 0;
      for (const v of A.maps[0]) { lo = Math.min(lo, v); hi = Math.max(hi, v); }
      // the weights are close to uniform, so the map is coloured from its own lowest to highest weight
      const rel = (i, j) => (A.maps[0][i * N + j] - lo) / Math.max(hi - lo, 1e-9);
      return {
        id: name, title: 'PSA: self-attention', sub: `${N} positions, ${A.nh} head${A.nh > 1 ? 's' : ''}`, rect, tone: 's7',
        summary: (ctx, r, S) => S.heat(ctx, rel, N, N, r, { seq: true, square: true, max: 1 }),
        inner: (ctx, r, S) => {
          const u = S.u(r, 8, 13);
          S.lines(ctx, ['x = x + proj(softmax(qᵀk / √d) v + pe(v));   x = x + ffn(x)', `q, k: ${A.kd} numbers per position; v: ${A.hd}`], S.place(r, [0, 0.0, 1, 0.12]), { mono: true, size: u, strong: (k) => k === 0 });
          const q = S.place(r, ar);
          S.caption(ctx, `weights ${fmt(lo, 3)} to ${fmt(hi, 3)} (even: ${fmt(1 / N, 4)}), colours stretched`, q, { size: u * 0.78, minW: 200 });
          S.heat(ctx, rel, N, N, q, { seq: true, square: true, max: 1 });
        },
        // a cell of the attention matrix opens the softmax that made that weight from the row's raw scores
        grids: [{ id: 'att', rect: ar, rows: N, cols: N, square: true, make: (i, j) => {
          const qk = A.qkv.d, scores = Array.from({ length: N }, (_, jj) => { let s = 0; for (let d = 0; d < A.kd; d++) s += qk[d * N + i] * qk[(A.kd + d) * N + jj]; return s / Math.sqrt(A.kd); });
          return Object.assign(Dp.lib(X, 'softmax', { scores, j, label: `a[${i}, ${j}]` }), { title: `Attention weight from position ${i} to ${j}`, sub: `softmax of row ${i}'s ${N} scores` });
        } }],
        children: [
          convNode(name + '.attn.qkv', 'q, k, v: 1×1 conv', W2(X, [0.46, 0.16, 0.25, 0.24], [0, 0.64, 0.48, 0.16])),
          convNode(name + '.attn.pe', 'pe: 3×3 depthwise', W2(X, [0.75, 0.16, 0.25, 0.24], [0.52, 0.64, 0.48, 0.16])),
          convNode(name + '.attn.proj', 'proj: 1×1 conv', W2(X, [0.46, 0.46, 0.25, 0.24], [0, 0.82, 0.48, 0.16])),
          convNode(name + '.ffn.0', 'feed-forward', W2(X, [0.75, 0.46, 0.25, 0.24], [0.52, 0.82, 0.48, 0.16])),
        ],
        about: [
          `Position-wise self-attention (PSA): every one of the ${N} cells of this map computes a query and a key from its ${mid.c} features, compares its query with every key (a dot product, scaled by 1/√${A.kd}) and turns the scores into weights with a softmax. Its output is the weighted average of everyone's values: a cell can borrow evidence from the other side of the picture, which a 3×3 conv cannot do in one step.`,
          hi - lo < 0.01
            ? `What this one learned is the simplest possible use of that: every weight is between ${fmt(lo, 4)} and ${fmt(hi, 4)}, almost exactly 1/${N}. Each cell takes a plain average of all ${N}, so the block hands every cell the same summary of the whole picture (a global average pool, in effect). On a map of only 4 × 4 cells and pictures this simple, that is apparently all the network needed; the colours exaggerate tiny differences.`
            : `Here the weights range from ${fmt(lo, 3)} to ${fmt(hi, 3)} (an even spread would be ${fmt(1 / N, 4)}): each cell still takes in the whole picture, leaning a little toward some positions. The colours stretch that range to show the pattern.`,
          'A 3×3 depthwise conv of the values (pe) adds back where things are, and a two-layer feed-forward conv refines each cell. Both parts are added to their input (residual), as in a transformer block.',
        ],
        code: `attn = ${pyPath(name)}.attn
x = torch.randn(1, ${mid.c}, ${side}, ${side})
q, k, v = attn.qkv(x).view(1, attn.nh, 2 * attn.kd + attn.hd, ${N}).split([attn.kd, attn.kd, attn.hd], 2)
weights = ((q.transpose(-2, -1) @ k) * attn.kd ** -0.5).softmax(-1)    # (1, heads, ${N}, ${N}); rows add up to 1
print(weights[0, 0].sum(-1))`,
      };
    }
    function sppfNode(name, rect) {
      const o = K[name], parts = K[name + '.parts'], tr = OUT(), lay = tileLayout(o, 16, tr);
      const pr = (k) => W2(X, [k * 0.16, 0.56, 0.15, 0.36], [k * 0.25, 0.46, 0.22, 0.16]);
      return {
        id: name, title: `${layerNo(name)} · SPPF`, sub: `${shape(K.b8)} → ${shape(o)}`, rect, tone: 's4',
        summary: (ctx, r) => drawTiles(ctx, o, r, 8),
        inner: (ctx, r, S) => {
          const u = S.u(r, 8, 13);
          S.lines(ctx, ['y = x + cv2(cat(a, pool(a), pool²(a), pool³(a))),  a = cv1(x)', 'three 5×5 max-pools in a row: windows of 5, 9 and 13 cells'], S.place(r, W2(X, [0, 0, 0.64, 0.12], [0, 0.32, 1, 0.1])), { mono: true, size: u, strong: (k) => k === 0 });
          parts.forEach((p, k) => { const q = S.place(r, pr(k)); S.caption(ctx, k ? `channel 0 after pool ${k}` : 'channel 0 after cv1', q, { size: u * 0.75 }); drawMap(ctx, p, 0, q); });
          const q = S.place(r, tr);
          S.caption(ctx, 'output channels', q, { size: u * 0.8 });
          drawTiles(ctx, o, q, 16, null, lay);
        },
        grids: [tileGrid(o, tr, name, lay), ...parts.slice(1).map((p, k) => ({ id: 'pool' + (k + 1), rect: pr(k + 1), rows: p.h, cols: p.w, square: true, make: (i, j) => resolve(p, 0, i, j) }))],
        children: [convNode(name + '.cv1', 'cv1: 1×1, no activation', W2(X, [0, 0.16, 0.3, 0.32], [0, 0.04, 0.48, 0.24])), convNode(name + '.cv2', 'cv2: 1×1 mix', W2(X, [0.33, 0.16, 0.3, 0.32], [0.52, 0.04, 0.48, 0.24]))],
        about: ['Spatial pyramid pooling, fast: three 5×5 max-pools applied one after another see windows of 5, 9 and 13 cells. On this 4 × 4 map that covers the whole picture, so every cell learns whether a pattern occurs anywhere nearby, at three ranges. The four maps are concatenated, mixed by a 1×1 conv and added to the input.', 'The first map row shows channel 0 after cv1 and after each pool: the bright spots spread out as the window grows. Open a pooled cell to see the 25 numbers it took the maximum of.'],
        code: `sppf = model.${name}.eval()\nx = torch.randn(1, 64, 4, 4)\npools = [sppf.cv1(x)]\nfor _ in range(sppf.n):                    # 3 pools: windows of 5, 9 and 13 cells\n    pools.append(F.max_pool2d(pools[-1], 5, 1, 2))\nprint(torch.allclose(sppf(x), x + sppf.cv2(torch.cat(pools, 1))))`,
      };
    }
    function c2psaNode(name, rect) {
      const o = K[name], tr = OUT(), lay = tileLayout(o, 16, tr);
      return {
        id: name, title: `${layerNo(name)} · C2PSA`, sub: `${shape(K.b9)} → ${shape(o)}`, rect, tone: 's7',
        summary: (ctx, r) => drawTiles(ctx, o, r, 8),
        inner: (ctx, r, S) => {
          S.lines(ctx, ['a, b = cv1(x).split();  y = cv2(cat(a, psa(b)))', 'attention on half the channels, the other half passed through'], S.place(r, W2(X, [0, 0, 0.64, 0.12], [0, 0.32, 1, 0.1])), { mono: true, size: S.u(r, 8, 13), strong: (k) => k === 0 });
          const q = S.place(r, tr);
          S.caption(ctx, 'output channels', q, { size: S.u(r, 8, 13) * 0.8 });
          drawTiles(ctx, o, q, 16, null, lay);
        },
        grids: [tileGrid(o, tr, name, lay)],
        children: [convNode(name + '.cv1', 'cv1', W2(X, [0, 0.16, 0.2, 0.36], [0, 0.04, 0.3, 0.26])), psaNode(name + '.m.0', W2(X, [0.22, 0.16, 0.2, 0.36], [0.35, 0.04, 0.3, 0.26])), convNode(name + '.cv2', 'cv2', W2(X, [0.44, 0.16, 0.2, 0.36], [0.7, 0.04, 0.3, 0.26]))],
        about: ['The last block of the backbone: like C3k2, half the channels pass through untouched and half go through a block, here a self-attention block (PSA). Attention is expensive in the number of positions, so YOLO uses it only on the coarsest map, 4 × 4 here (20 × 20 at 640 pixels).'],
        code: `c2psa = model.${name}\nprint(c2psa)`,
      };
    }

    // ---- the backbone and the neck
    const BB = [
      ['b0', 'conv', 'Conv 3×3/2 · P1'], ['b1', 'conv', 'Conv 3×3/2 · P2'], ['b2', 'bott', 'C3k2'], ['b3', 'conv', 'Conv 3×3/2 · P3'], ['b4', 'bott', 'C3k2'],
      ['b5', 'conv', 'Conv 3×3/2 · P4'], ['b6', 'c3k', 'C3k2 (C3k)'], ['b7', 'conv', 'Conv 3×3/2 · P5'], ['b8', 'c3k', 'C3k2 (C3k)'], ['b9', 'sppf', 'SPPF'], ['b10', 'c2psa', 'C2PSA'],
    ];
    const NECK = [
      ['u11', 'up', 'Upsample ×2'], ['h13', 'c3k', 'C3k2 · cat P4'], ['u14', 'up', 'Upsample ×2'], ['h16', 'c3k', 'C3k2 · cat P3 → P3 out'],
      ['h17', 'conv', 'Conv 3×3/2'], ['h19', 'c3k', 'C3k2 · cat → P4 out'], ['h20', 'conv', 'Conv 3×3/2'], ['h22', 'attn', 'C3k2 + PSA → P5 out'],
    ];
    function layerNode([name, kind, title], rect) {
      const t = `${layerNo(name)} · ${title}`;
      if (kind === 'conv') return convNode(name, t, rect);
      if (kind === 'sppf') return sppfNode(name, rect);
      if (kind === 'c2psa') return c2psaNode(name, rect);
      if (kind === 'up') {
        const o = K[name], src = o.up, tr = W2(X, [0.5, 0.16, 0.5, 0.8], [0, 0.4, 1, 0.58]), lay = tileLayout(o, 16, tr);
        return { id: name, title: t, sub: `${shape(src)} → ${shape(o)}`, rect, tone: 's6',
          summary: (ctx, r) => drawTiles(ctx, o, r, 8),
          inner: (ctx, r, S) => { S.lines(ctx, ['every number copied into a 2 × 2 block (nearest neighbour)', `${src.h} × ${src.w} → ${o.h} × ${o.w}: the coarse map, at the next scale's size`], S.place(r, W2(X, [0, 0.02, 0.48, 0.2], [0, 0, 1, 0.16])), { mono: true, size: S.u(r, 8, 13) }); drawTiles(ctx, o, S.place(r, tr), 16, null, lay); },
          grids: [tileGrid(o, tr, name, lay)],
          about: ['The neck carries the deep, coarse features back up to finer grids so they can be combined with the earlier, sharper ones. Nearest-neighbour upsampling has no weights: each number is simply repeated in a 2 × 2 block.'],
          code: `up = torch.nn.Upsample(scale_factor=2, mode='nearest')\nprint(up(torch.arange(4.).view(1, 1, 2, 2)))`,
        };
      }
      return c3k2Node(name, kind, t, rect);
    }
    const flowRects = (n, X0) => Array.from({ length: n }, (_, i) => {
      if (X0.wide) { const cols = Math.ceil(n / 2), row = Math.floor(i / cols), col = i % cols, w = 1 / cols; return [col * w + 0.006, 0.08 + row * 0.47, w - 0.012, 0.4]; }
      const cols = 2, row = Math.floor(i / cols), col = i % cols, rows = Math.ceil(n / cols), h = 0.92 / rows; return [col * 0.5 + 0.01, 0.07 + row * h, 0.48, h - 0.02];
    });
    const stageNode = (id, title, sub, list, rect, tone, about, code, shown) => {
      const rs = flowRects(list.length, X);
      return {
        id, title, sub, rect, tone,
        // closed: where things fire at each scale (mean |activation| over the channels)
        summary: (ctx, r, S) => drawPyramid(ctx, S, shown.map((n) => K[n]), r, shown.map((n) => `${K[n].h} × ${K[n].w}`)),
        inner: (ctx, r, S) => {
          for (let i = 0; i + 1 < list.length; i++) {
            const p = S.place(r, rs[i]), q = S.place(r, rs[i + 1]);
            if (Math.abs(p.y - q.y) < 2) S.arrow(ctx, p.x + p.w + 1, p.y + p.h / 2, q.x - 2, q.y + q.h / 2, { lw: 1, head: 4 });
            else if (X.wide) {
              // end of one row to the start of the next: down, back across the gap, down again
              const my = (p.y + p.h + q.y) / 2;
              ctx.strokeStyle = theme.c('muted'); ctx.lineWidth = 1; ctx.beginPath(); ctx.moveTo(p.x + p.w / 2, p.y + p.h + 1); ctx.lineTo(p.x + p.w / 2, my); ctx.lineTo(q.x + q.w / 2, my); ctx.stroke();
              S.arrow(ctx, q.x + q.w / 2, my, q.x + q.w / 2, q.y - 2, { lw: 1, head: 4, color: theme.c('muted') });
            } else S.arrow(ctx, p.x + p.w / 2, p.y + p.h + 1, q.x + q.w / 2, q.y - 2, { lw: 1, head: 4, color: theme.c('muted') });
          }
        },
        children: list.map((d, i) => layerNode(d, rs[i])),
        about, code,
      };
    };

    // ---- the heads
    const D = L.dec, dets = L.dets;
    function anchorNode(head, si, gy, gx) {
      const maps = F.heads[head][si], dd = D[head], a = Y.GRIDS.slice(0, si).reduce((s, g) => s + g * g, 0) + gy * Y.GRIDS[si] + gx, s = Y.STRIDES[si];
      const dist = Array.from(dd.dist.subarray(a * 4, a * 4 + 4)), box = Array.from(dd.boxes.subarray(a * 4, a * 4 + 4)), sc = Array.from(dd.scores.subarray(a * Y.NC, a * Y.NC + Y.NC));
      const best = sc.indexOf(Math.max(...sc)), cx = (gx + 0.5) * s, cy = (gy + 0.5) * s, n = maps.h * maps.w, logit = (k) => maps.d[(4 + k) * n + gy * maps.w + gx];
      const names = ['left', 'top', 'right', 'bottom'];
      return {
        title: `Cell (${gx}, ${gy}) of the ${Y.GRIDS[si]} × ${Y.GRIDS[si]} grid`, sub: `${Y.NAMES[best]} ${fmt(sc[best], 2)}`, tone: 's4',
        summary: (ctx, r, S) => S.bars(ctx, Float32Array.from(sc), r, { positive: true, max: 1, color: (k) => CLS(k) }),
        inner: (ctx, r, S) => {
          const u = S.u(r, 8, 13);
          S.lines(ctx, [
            `centre = ((${gx} + 0.5) × ${s}, (${gy} + 0.5) × ${s}) = (${fmt(cx, 1)}, ${fmt(cy, 1)}) px`,
            `distances (in cells) l, t, r, b = ${dist.map((v) => fmt(v, 2)).join(', ')}`,
            `box = (cx − l·${s}, cy − t·${s}, cx + r·${s}, cy + b·${s}) = (${box.map((v) => fmt(v, 1)).join(', ')})`,
            `scores = σ(logits) = ${sc.map((v, k) => `${Y.NAMES[k]} ${fmt(v, 3)}`).join(', ')}`,
          ], S.place(r, [0, 0.0, 1, 0.3]), { mono: true, size: u, strong: (k) => k === 2 });
          const q = S.place(r, W2(X, [0, 0.36, 0.3, 0.6], [0, 0.34, 0.6, 0.3]));
          drawImage(ctx, L, q);
          const k = Math.min(q.w, q.h) / Y.IMG;
          ctx.strokeStyle = CLS(best); ctx.lineWidth = 2; ctx.strokeRect(q.x + box[0] * k, q.y + box[1] * k, (box[2] - box[0]) * k, (box[3] - box[1]) * k);
          ctx.fillStyle = theme.c('focus'); ctx.beginPath(); ctx.arc(q.x + cx * k, q.y + cy * k, 3, 0, Math.PI * 2); ctx.fill();
        },
        children: [
          ...[0, 1, 2, 3].map((k) => Object.assign(resolve(maps, k, gy, gx), { id: 'd' + k, rect: W2(X, [0.34 + k * 0.165, 0.38, 0.15, 0.26], [k * 0.25, 0.68, 0.23, 0.14]), title: `${names[k]} distance`, sub: fmt(dist[k], 3) })),
          ...[0, 1, 2, 3].map((k) => Object.assign(Dp.lib(X, 'sigmoid', { z: logit(k) }), { id: 's' + k, rect: W2(X, [0.34 + k * 0.165, 0.7, 0.15, 0.26], [k * 0.25, 0.85, 0.23, 0.14]), title: `${Y.NAMES[k]} score`, sub: `σ(${fmt(logit(k), 2)}) = ${fmt(sc[k], 3)}` })),
        ],
        about: [
          `Every cell of every grid makes a prediction. This one, on the stride-${s} grid, has its centre at (${fmt(cx, 1)}, ${fmt(cy, 1)}) pixels. Its first four outputs are distances, in cells, from that centre to the box's left, top, right and bottom edges. YOLO26 drops DFL (the earlier YOLOs predicted a probability distribution over 16 possible distances for each edge); here each distance is a single number, which is simpler to export and run.`,
          `The other four outputs are class logits; a sigmoid turns each into a score between 0 and 1 (not a softmax: a cell may score low on every class, which is how most cells say "no object here"). This cell's best guess is ${Y.NAMES[best]} at ${fmt(sc[best], 3)}.`,
        ],
        code: `# model, random_scene, flat and decode: the whole-model code (Architecture view, top level, Python tab)
img, _, _ = random_scene(np.random.default_rng(1))
with torch.no_grad():
    o2m, o2o = model.eval()(img[None])
p = flat(${head})[0]                              # (336, 8): every cell of the three grids
a = ${a}                                          # this cell: grid ${si}, row ${gy}, column ${gx}
boxes, logits = decode(p)
print(p[a, :4], boxes[a], logits[a].sigmoid())`,
      };
    }
    function scaleNode(head, si, rect) {
      const maps = F.heads[head][si], g = Y.GRIDS[si], n = g * g, pre = head === 'o2m' ? 'detect.' : 'detect.one2one_', dd = D[head];
      const off = Y.GRIDS.slice(0, si).reduce((s, q) => s + q * q, 0), best = (gy, gx) => { let m = 0; for (let k = 0; k < Y.NC; k++) m = Math.max(m, dd.scores[(off + gy * g + gx) * Y.NC + k]); return m; };
      const gr = W2(X, [0, 0.12, 0.4, 0.84], [0, 0.1, 1, 0.46]);
      return {
        id: 'p' + (si + 3), title: `P${si + 3}: ${g} × ${g} grid`, sub: `stride ${Y.STRIDES[si]}`, rect, tone: 's4',
        summary: (ctx, r, S) => S.heat(ctx, (i, j) => best(i, j), g, g, r, { seq: true, max: 1, square: true }),
        inner: (ctx, r, S) => {
          const u = S.u(r, 8, 13);
          S.lines(ctx, [`${n} cells × (4 distances + ${Y.NC} class logits)`, 'best class score per cell; open a cell for its box'], S.place(r, W2(X, [0, 0, 0.6, 0.1], [0, 0, 1, 0.08])), { mono: true, size: u });
          const q = sq(S.place(r, gr), g, g);
          drawImage(ctx, L, { x: q.x, y: q.y, w: q.w, h: q.h });
          ctx.globalAlpha = 0.72; S.heat(ctx, (i, j) => best(i, j), g, g, { x: q.x, y: q.y, w: q.w, h: q.h }, { seq: true, max: 1, square: true, frame: false }); ctx.globalAlpha = 1;
        },
        grids: [{ id: 'cell', rect: gr, rows: g, cols: g, square: true, make: (i, j) => anchorNode(head, si, i, j) }],
        children: [
          convNode(`${pre}cv2.${si}.0`, 'box: 3×3 conv', W2(X, [0.46, 0.14, 0.17, 0.36], [0, 0.6, 0.32, 0.18])),
          convNode(`${pre}cv2.${si}.1`, 'box: 3×3 conv', W2(X, [0.64, 0.14, 0.17, 0.36], [0.34, 0.6, 0.32, 0.18])),
          convNode(`${pre}cv2.${si}.2`, 'box: 1×1 → 4 distances', W2(X, [0.82, 0.14, 0.18, 0.36], [0.68, 0.6, 0.32, 0.18])),
          convNode(`${pre}cv3.${si}.0.0`, 'class: 3×3 depthwise', W2(X, [0.46, 0.56, 0.17, 0.4], [0, 0.8, 0.32, 0.18])),
          convNode(`${pre}cv3.${si}.1.1`, 'class: 1×1 conv', W2(X, [0.64, 0.56, 0.17, 0.4], [0.34, 0.8, 0.32, 0.18])),
          convNode(`${pre}cv3.${si}.2`, `class: 1×1 → ${Y.NC} logits`, W2(X, [0.82, 0.56, 0.18, 0.4], [0.68, 0.8, 0.32, 0.18])),
        ],
        about: [
          `The ${si === 0 ? 'finest' : si === 1 ? 'middle' : 'coarsest'} of the three grids: each cell covers ${Y.STRIDES[si]} × ${Y.STRIDES[si]} pixels and is responsible for objects whose centre falls nearby. ${si === 0 ? 'Small shapes are found here.' : si === 2 ? 'Large shapes are found here: each cell sees most of the picture.' : 'Medium shapes are found here.'}`,
          'Two branches read the same neck features: a box branch (two 3×3 convs and a 1×1 conv to 4 distances) and a class branch (depthwise and pointwise convs to one logit per class). The heat shows each cell’s best class score.',
        ],
        code: `head = model.detect.${head === 'o2m' ? 'cv2' : 'one2one_cv2'}[${si}], model.detect.${head === 'o2m' ? 'cv3' : 'one2one_cv3'}[${si}]\nprint(head)`,
      };
    }
    function headNode(head, rect) {
      const isO = head === 'o2o';
      return {
        id: head, title: isO ? 'One-to-one head' : 'One-to-many head', sub: isO ? 'one cell per object' : 'ten cells per object', rect, tone: isO ? 's7' : 's4',
        // closed: the fine grid's best score per cell, a square in the middle of the box
        summary: (ctx, r, S) => { const g = Y.GRIDS[0], dd = D[head], s = Math.min(r.w, r.h); S.heat(ctx, (i, j) => Math.max(...dd.scores.subarray((i * g + j) * Y.NC, (i * g + j + 1) * Y.NC)), g, g, { x: r.x + (r.w - s) / 2, y: r.y, w: s, h: s }, { seq: true, max: 1, square: true }); },
        inner: (ctx, r, S) => S.lines(ctx, [isO ? 'top-1 assignment in training → one box per object → no NMS' : 'top-10 assignment in training → several boxes per object → NMS'], S.place(r, [0, 0, 1, 0.08]), { mono: true, size: S.u(r, 8, 13), strong: () => true }),
        children: [0, 1, 2].map((si) => scaleNode(head, si, W2(X, [si * 0.335, 0.12, 0.32, 0.86], [0, 0.1 + si * 0.3, 1, 0.28]))),
        about: isO
          ? ['YOLO26’s main head. In training, each object is assigned to exactly one cell (the one whose prediction already matches it best), and every other cell is taught to say "nothing here", even cells that would have found the same object. So at inference each object gets one box, and the output can be used as it is: no NMS, no tuning of an IoU threshold, and the same speed whatever the scene. This is what Ultralytics calls end-to-end, NMS-free inference (the idea comes from YOLOv10).', 'It has its own copy of the head’s weights; the backbone and neck are shared with the one-to-many head.']
          : ['The classic YOLO head. In training each object is assigned to its 10 best cells, which gives the network plenty of positive examples and a strong learning signal, but at inference every object is found by several neighbouring cells. Non-maximum suppression (NMS) then keeps the best box and deletes the ones that overlap it.', 'YOLO26 trains both heads together (with ProgLoss shifting the weight toward the one-to-one head as training goes on) and keeps this one as an option: Ultralytics’ predict(..., end2end=False).'],
        code: `${py.detect}\n\nimg, _, _ = random_scene(np.random.default_rng(1))\nprint(detect(model, img, head='${head}'))`,
      };
    }
    // the post-processing: NMS for the one-to-many head, top-k for the one-to-one head
    function iouNode(a, b) {
      const iw = Math.max(0, Math.min(a.box[2], b.box[2]) - Math.max(a.box[0], b.box[0])), ih = Math.max(0, Math.min(a.box[3], b.box[3]) - Math.max(a.box[1], b.box[1]));
      const inter = iw * ih, area = (q) => (q.box[2] - q.box[0]) * (q.box[3] - q.box[1]), uni = area(a) + area(b) - inter;
      // the two boxes, scaled to fit, with their overlap shaded
      const pair = (ctx, r) => {
        const x0 = Math.min(a.box[0], b.box[0]), y0 = Math.min(a.box[1], b.box[1]), span = Math.max(Math.max(a.box[2], b.box[2]) - x0, Math.max(a.box[3], b.box[3]) - y0, 1);
        const k = (Math.min(r.w, r.h) * 0.9) / span, ox = r.x + (r.w - span * k) / 2, oy = r.y + (r.h - span * k) / 2, R = (q) => [ox + (q[0] - x0) * k, oy + (q[1] - y0) * k, (q[2] - q[0]) * k, (q[3] - q[1]) * k];
        ctx.fillStyle = theme.c('s5', 0.3);
        if (iw > 0 && ih > 0) ctx.fillRect(ox + (Math.max(a.box[0], b.box[0]) - x0) * k, oy + (Math.max(a.box[1], b.box[1]) - y0) * k, iw * k, ih * k);
        ctx.lineWidth = 1.6; ctx.strokeStyle = CLS(a.cls); ctx.strokeRect(...R(a.box));
        ctx.setLineDash([3, 3]); ctx.strokeStyle = theme.c('muted'); ctx.strokeRect(...R(b.box)); ctx.setLineDash([]);
      };
      return {
        title: 'IoU of two boxes', sub: `${fmt(inter, 1)} / ${fmt(uni, 1)} = ${fmt(inter / uni, 3)}`, tone: 's5',
        summary: (ctx, r) => pair(ctx, r),
        inner: (ctx, r, S) => {
          S.lines(ctx, ['IoU = intersection area / union area', `intersection = ${fmt(iw, 2)} × ${fmt(ih, 2)} = ${fmt(inter, 2)}`, `union = ${fmt(area(a), 1)} + ${fmt(area(b), 1)} − ${fmt(inter, 1)} = ${fmt(uni, 1)}`, `IoU = ${fmt(inter / uni, 4)}`], S.place(r, W2(X, [0, 0.02, 0.62, 0.34], [0, 0.02, 0.56, 0.34])), { mono: true, strong: (k) => k === 3 });
          pair(ctx, S.place(r, W2(X, [0.68, 0.0, 0.32, 0.4], [0.6, 0.02, 0.4, 0.34])));
        },
        children: [
          Object.assign(Dp.fmul(X, iw, ih), { id: 'inter', rect: W2(X, [0, 0.46, 0.31, 0.5], [0, 0.42, 1, 0.17]), title: 'Intersection: width × height' }),
          Object.assign(Dp.fadd(X, area(a) + area(b), -inter), { id: 'union', rect: W2(X, [0.345, 0.46, 0.31, 0.5], [0, 0.62, 1, 0.17]), title: 'Union: the areas minus the overlap' }),
          Object.assign(Dp.lib(X, 'div', { a: inter, b: uni }), { id: 'div', rect: W2(X, [0.69, 0.46, 0.31, 0.5], [0, 0.82, 1, 0.17]), title: 'The division' }),
        ],
        about: ['Intersection over union: how much two boxes overlap, from 0 (apart) to 1 (identical). NMS deletes a box when its IoU with a better box of the same class is above the threshold.'],
        code: 'def iou(a, b):\n    iw = max(0, min(a[2], b[2]) - max(a[0], b[0]))\n    ih = max(0, min(a[3], b[3]) - max(a[1], b[1]))\n    inter = iw * ih\n    area = lambda q: (q[2] - q[0]) * (q[3] - q[1])\n    return inter / (area(a) + area(b) - inter)',
      };
    }
    const nmsNode = (rect) => {
      const kept = dets.o2m, removed = kept.suppressed || [], pair = removed[0];
      return {
        id: 'nms', title: 'NMS', sub: `kept ${kept.length}, removed ${removed.length}`, rect, tone: 's5',
        summary: (ctx, r, S) => S.lines(ctx, [`${kept.length + removed.length} → ${kept.length}`], r, { mono: true, size: clamp(r.h * 0.3, 8, 18), strong: () => true }),
        inner: (ctx, r, S) => {
          const u = S.u(r, 8, 13);
          S.lines(ctx, [`1. sort the ${kept.length + removed.length} boxes over the threshold by score`, '2. keep the best; delete every box of its class with IoU > ' + fmt(L.iouThr, 2), '3. repeat with the next box still standing'], S.place(r, [0, 0.0, 1, 0.2]), { mono: true, size: u });
          const q = S.place(r, W2(X, [0, 0.26, 0.4, 0.7], [0, 0.24, 1, 0.42]));
          const d = drawImage(ctx, L, q), k = d.s / Y.IMG;
          drawBoxes(ctx, removed, d, { lw: 1 }); drawBoxes(ctx, kept, d, { lw: 2 });
          S.caption(ctx, 'kept boxes, each with how many it removed', q, { size: u * 0.8, minW: 200 });
          for (const b of kept) {
            const n = removed.filter((x) => x.suppressedBy === b).length;
            if (n) S.text(ctx, `−${n}`, d.x + b.box[2] * k + 2, d.y + b.box[1] * k + 6, { size: u * 0.85, w: 700, color: CLS(b.cls) });
          }
        },
        children: pair ? [Object.assign(iouNode(pair.suppressedBy, pair), { id: 'iou', rect: W2(X, [0.46, 0.3, 0.54, 0.6], [0, 0.7, 1, 0.28]), sub: `the first box removed: IoU ${fmt(Y.iou(pair.box, pair.suppressedBy.box), 2)}` })] : [],
        about: [`Non-maximum suppression, greedy: sort the boxes by score, keep the best, delete every box of the same class that overlaps it by more than the IoU threshold (${fmt(L.iouThr, 2)} here; Ultralytics' default is 0.7), and repeat. It is a sequential loop whose length depends on the scene, which makes it awkward to run on accelerators and to export; and the threshold is a trade-off: too low and it deletes real neighbours, too high and duplicates survive.`],
        code: py.detect,
      };
    };
    const topkNode = (rect) => {
      const o = dets.o2o, all = Array.from(D.o2o.scores).sort((a, b) => b - a).slice(0, 24);
      return {
        id: 'topk', title: 'Top scores', sub: `${o.length} over ${fmt(L.conf, 2)}`, rect, tone: 's7',
        summary: (ctx, r, S) => S.bars(ctx, Float32Array.from(all), r, { positive: true, max: 1, color: (k) => (all[k] > L.conf ? theme.c('s7') : theme.c('muted', 0.5)) }),
        inner: (ctx, r, S) => {
          const u = S.u(r, 8, 13);
          S.lines(ctx, ['take the highest of the 336 × 4 cell-and-class scores, keep those over the threshold. Done.'], S.place(r, [0, 0, 1, 0.08]), { mono: true, size: u });
          const br = S.place(r, [0, 0.2, 1, 0.5]);
          S.caption(ctx, `the 24 highest scores; ${o.length} are over ${fmt(L.conf, 2)}`, br, { size: u * 0.8 });
          S.bars(ctx, Float32Array.from(all), br, { positive: true, max: 1, color: (k) => (all[k] > L.conf ? theme.c('s7') : theme.c('muted', 0.5)) });
          const y = br.y + br.h - L.conf * br.h;
          ctx.strokeStyle = theme.c('ink', 0.6); ctx.setLineDash([4, 3]); ctx.beginPath(); ctx.moveTo(br.x, y); ctx.lineTo(br.x + br.w, y); ctx.stroke(); ctx.setLineDash([]);
        },
        about: ['The one-to-one head needs no clean-up: its training already made sure each object has one confident cell and every other cell stays quiet. Post-processing is a sort and a threshold, the same amount of work for every picture, and easy to export as part of the model.', `Here the scores fall off a cliff: ${o.length} are above ${fmt(L.conf, 2)}, and the next ones are far below. That gap is what makes a fixed threshold safe without NMS.`],
        code: py.detect,
      };
    };
    const outNode = (rect) => ({
      id: 'out', title: 'Detections', sub: `${(L.head === 'o2o' ? dets.o2o : dets.o2m).length} boxes`, rect, tone: 's7',
      summary: (ctx, r) => { const s = Math.min(r.w, r.h), d = drawImage(ctx, L, { x: r.x + (r.w - s) / 2, y: r.y, w: s, h: s }); drawBoxes(ctx, L.head === 'o2o' ? dets.o2o : dets.o2m, d, { lw: 1.2 }); },
      inner: (ctx, r, S) => {
        const q = S.place(r, W2(X, [0, 0, 0.62, 0.98], [0, 0, 1, 0.7])), d = drawImage(ctx, L, q), list = L.head === 'o2o' ? dets.o2o : dets.o2m;
        drawBoxes(ctx, list, d, { lw: 2 });
        S.lines(ctx, list.map((b) => `${Y.NAMES[b.cls].padEnd(8)} ${fmt(b.score, 2)}  (${b.box.map((v) => Math.round(v)).join(', ')})`), S.place(r, W2(X, [0.66, 0.02, 0.34, 0.9], [0, 0.74, 1, 0.26])), { mono: true, size: S.u(r, 8, 12) });
      },
      about: [`The final answer of the ${L.head === 'o2o' ? 'one-to-one head, as it comes' : 'one-to-many head after NMS'}: each box with its class and confidence, in pixels of the 128 × 128 image.`],
      code: `${py.detect}\n\nimg, _, _ = random_scene(np.random.default_rng(1))\nboxes, scores, classes = detect(model, img, head='${L.head}')\nfor b, s, c in zip(boxes, scores, classes):\n    print(NAMES[c], round(s.item(), 2), [round(v) for v in b.tolist()])`,
    });

    /* ---- the whole model, laid out for the stage's shape. Tall (a phone): a column, with the two heads, their
       post-processing and the detections in pairs. Most screens: two rows, the image, the backbone and the neck,
       then the heads, NMS and top scores, and the detections, every box about a third of the width so that its
       title stays readable. Very wide: one row. */
    const aspect = size.w / size.h, MODE = aspect < 1.05 ? 'tall' : aspect >= 2.3 && size.w >= 1000 ? 'wide' : 'mid';
    const LAY = {
      tall: { input: [0.25, 0, 0.5, 0.115], backbone: [0, 0.145, 1, 0.175], neck: [0, 0.35, 1, 0.155], o2m: [0, 0.535, 0.48, 0.15], o2o: [0.52, 0.535, 0.48, 0.15],
        nms: [0, 0.715, 0.48, 0.12], topk: [0.52, 0.715, 0.48, 0.12], out: [0.25, 0.865, 0.5, 0.135] },
      // the heads leave a margin on the left for the line from the neck, which wraps round like a line of text,
      // and are the widest boxes: "One-to-many head" needs about 150 px at 1280 × 720
      mid: { input: [0, 0.03, 0.355, 0.38], backbone: [0.405, 0, 0.27, 0.44], neck: [0.725, 0, 0.275, 0.44], o2m: [0.025, 0.54, 0.33, 0.21], o2o: [0.025, 0.79, 0.33, 0.21],
        nms: [0.405, 0.54, 0.27, 0.21], topk: [0.405, 0.79, 0.27, 0.21], out: [0.725, 0.57, 0.275, 0.4] },
      wide: { input: [0, 0.3, 0.1, 0.4], backbone: [0.125, 0.02, 0.15, 0.96], neck: [0.3, 0.02, 0.15, 0.96], o2m: [0.475, 0.02, 0.17, 0.47], o2o: [0.475, 0.51, 0.17, 0.47],
        nms: [0.67, 0.02, 0.14, 0.47], topk: [0.67, 0.51, 0.14, 0.47], out: [0.835, 0.28, 0.165, 0.44] },
    }[MODE];
    const inputNode = {
      id: 'input', title: 'Image', sub: '3 × 128 × 128', rect: LAY.input, tone: 's8',
      summary: (ctx, r) => drawImage(ctx, L, { x: r.x + (r.w - Math.min(r.w, r.h)) / 2, y: r.y, w: Math.min(r.w, r.h), h: Math.min(r.w, r.h) }),
      inner: (ctx, r, S) => {
        const q = S.place(r, W2(X, [0, 0.06, 0.6, 0.92], [0, 0.06, 1, 0.6])), s = Math.min(q.w, q.h);
        drawImage(ctx, L, { x: q.x, y: q.y, w: s, h: s });
        S.lines(ctx, ['128 × 128 pixels × red, green, blue', '= 49,152 numbers between 0 and 1', 'zoom into a pixel to open its red value'], S.place(r, W2(X, [0.64, 0.06, 0.36, 0.3], [0, 0.7, 1, 0.2])), { mono: true, size: S.u(r, 8, 13) });
      },
      grids: [{ id: 'px', rect: W2(X, [0, 0.06, 0.6, 0.92], [0, 0.06, 1, 0.6]), rows: Y.IMG, cols: Y.IMG, square: true, make: (i, j) => resolve(K.input, 0, i, j) }],
      about: ['The input: a 128 × 128 colour picture, 49,152 numbers between 0 and 1. YOLO26 works on 640 × 640 pictures (1.2 million numbers); this one is smaller so the browser can run it as you drag.', 'The scene is drawn with the same maths as the training data: a background gradient, thin lines that are not objects, the shapes (anti-aliased over one pixel) and a little sensor noise.'],
      code: `${py.scenes}\n\nimg, boxes, classes = random_scene(np.random.default_rng(1))\nprint(img.shape, boxes, [NAMES[c] for c in classes])`,
    };
    const bbNode = stageNode('backbone', 'Backbone', '11 layers: 128 px → 4 × 4', BB, LAY.backbone, 's1',
      ['The backbone turns pixels into features. Five convolutions with stride 2 halve the map five times, 128 → 64 → 32 → 16 → 8 → 4, while the channels grow from 3 to 64: fewer positions, each describing a larger patch in more detail. C3k2 blocks refine the features at each scale; SPPF and the attention block C2PSA let the coarsest map see the whole picture.', 'The maps at stride 8, 16 and 32 (P3, P4, P5) are handed to the neck.'],
      `${py.layers}\n\nmodel = YOLO26Tiny()\nx = torch.randn(1, 3, 128, 128)\nfor name in ['b0', 'b1', 'b2', 'b3', 'b4', 'b5', 'b6', 'b7', 'b8', 'b9', 'b10']:\n    x = getattr(model, name)(x)\n    print(name, tuple(x.shape))`, ['b0', 'b2', 'b4', 'b6', 'b10']);
    const neckNode = stageNode('neck', 'Neck', 'FPN + PAN over P3–P5', NECK, LAY.neck, 's3',
      ['The neck combines scales in both directions. Top-down (FPN), the coarse 4 × 4 map, which knows what is in the picture, is upsampled and concatenated with the sharper 8 × 8 and 16 × 16 maps from the backbone. Bottom-up (PAN), stride-2 convs carry the sharpened result back down. Each of the three outputs mixes "what" from deep layers with "where" from shallow ones.'],
      `print(model.h13, model.h16, model.h19, model.h22)`, ['h16', 'h19', 'h22']);
    const root = {
      id: 'yolo', title: 'YOLO26 detector', sub: `${fmtInt(L.net.params)} parameters`,
      inner: (ctx, r, S) => {
        const c = (id) => S.place(r, LAY[id]);
        /* an arrow from box p to box q: sideways when they share a row, straight down (or up) when they share a
           column, each time through the middle of what the two boxes have in common */
        const A = (p, q, o) => {
          const y0 = Math.max(p.y, q.y), y1 = Math.min(p.y + p.h, q.y + q.h), x0 = Math.max(p.x, q.x), x1 = Math.min(p.x + p.w, q.x + q.w);
          if (y1 - y0 > 6) { const y = (y0 + y1) / 2; return q.x >= p.x + p.w ? S.arrow(ctx, p.x + p.w + 2, y, q.x - 3, y, o) : S.arrow(ctx, p.x - 2, y, q.x + q.w + 3, y, o); }
          if (x1 - x0 > 6) { const x = (x0 + x1) / 2; return q.y >= p.y + p.h ? S.arrow(ctx, x, p.y + p.h + 2, x, q.y - 3, o) : S.arrow(ctx, x, p.y - 2, x, q.y + q.h + 3, o); }
          return S.arrow(ctx, p.x + p.w / 2, p.y + p.h + 2, q.x + q.w / 2, q.y - 3, o);
        };
        A(c('input'), c('backbone')); A(c('backbone'), c('neck'));
        if (MODE === 'mid') {
          // the neck's three maps go to both heads: down from the neck, back across under the first row, and
          // down the left margin into each head
          const n = c('neck'), a = c('o2m'), b = c('o2o'), my = (n.y + n.h + a.y) / 2, bx = r.x - 5;
          ctx.strokeStyle = theme.c('ink-2'); ctx.lineWidth = 1.3; ctx.lineJoin = 'round';
          ctx.beginPath(); ctx.moveTo(n.x + n.w / 2, n.y + n.h + 2); ctx.lineTo(n.x + n.w / 2, my); ctx.lineTo(bx, my); ctx.lineTo(bx, b.y + b.h / 2); ctx.stroke();
          S.arrow(ctx, bx, a.y + a.h / 2, a.x - 3, a.y + a.h / 2, { head: 5 }); S.arrow(ctx, bx, b.y + b.h / 2, b.x - 3, b.y + b.h / 2, { head: 5 });
        } else { A(c('neck'), c('o2m')); A(c('neck'), c('o2o')); }
        A(c('o2m'), c('nms')); A(c('o2o'), c('topk'));
        // the head in use feeds the detections (highlighted); the other one's path is dashed
        const used = { color: theme.c('focus') }, unused = { color: theme.c('muted', 0.7), dash: [3, 3] };
        A(c('nms'), c('out'), L.head === 'o2m' ? used : unused);
        A(c('topk'), c('out'), L.head === 'o2o' ? used : unused);
      },
      children: [inputNode, bbNode, neckNode, headNode('o2m', LAY.o2m), headNode('o2o', LAY.o2o), nmsNode(LAY.nms), topkNode(LAY.topk), outNode(LAY.out)],
      about: [
        `The whole detector: ${fmtInt(L.net.params)} parameters in the layer layout of Ultralytics' yolo26.yaml (YOLO26n has 2.4 million). The image goes through a backbone of convolutions, a neck that mixes three scales, and two heads that each predict, for every cell of three grids (16 × 16, 8 × 8 and 4 × 4, 336 cells in all), a box and a score for each class.`,
        `The one-to-many head needs NMS to clean up its duplicate boxes; the one-to-one head is used as it is. Everything here ran in your browser on the scene shown, in the order drawn; double-${NA.tapWord ? NA.tapWord(true) : 'click'} or zoom into any part to open it (one ${NA.tapWord ? NA.tapWord(true) : 'click'} shows what it is).`,
        'Training (in PyTorch, on a GPU) used this repository’s versions of YOLO26’s recipe: ProgLoss, STAL and the MuSGD optimiser. The Python tab has the whole thing: model, synthetic data, loss, optimiser, training loop and inference.',
      ],
      code: py.model(), codeLabel: 'Python · PyTorch · the whole detector: model, data, loss, MuSGD, training and inference',
    };
    return root;
  }

  NA.archYolo = { build, py, pyPath };
})(window.NA);
