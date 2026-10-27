/* arch-transformer.js — the Tiny Transformer as a zoomable architecture: model → blocks → attention, MLP and
   LayerNorm → individual heads and single tokens, drawn from the live trace of the current prompt, each with an
   in-depth explanation and the same component written in PyTorch. */
(function (NA) {
  'use strict';
  const { fmt, fmtPct, fmtInt, theme, clamp } = NA;
  const D = NA.lmData;
  const word = (id) => (id === D.W.BOS ? '⟨bos⟩' : D.VOCAB[id]);
  const headColor = (h) => theme.series[[0, 2, 6, 3][h % 4]];
  const row = (arr, i, w, off = 0, len = w) => arr.subarray(i * w + off, i * w + off + len);

  // ------------------------------------------------------------------ PyTorch versions
  const py = {};
  py.config = (c) => `V, T, D, H, L, FF = ${c.V}, ${c.T}, ${c.d}, ${c.H}, ${c.L}, ${c.F}   # vocabulary, context, width, heads, blocks, MLP width`;
  py.tokenizer = () => `VOCAB = [${D.VOCAB.map((w) => `'${w}'`).join(', ')}]
ID = {w: i for i, w in enumerate(VOCAB)}

def tokenize(text):
    """Split on spaces, keep '.' as a token, look each word up; unknown words are dropped."""
    words = text.lower().replace('.', ' . ').split()
    return [ID['<bos>']] + [ID[w] for w in words if w in ID]

tokenize('alice likes tea . alice drinks')   # [${[D.W.BOS, ...D.tokenize('alice likes tea . alice drinks').ids].join(', ')}]`;
  py.embedding = (c, ids) => `import torch
import torch.nn as nn

tok_emb = nn.Embedding(${c.V}, ${c.d})   # E: one learned row of ${c.d} numbers per word
pos_emb = nn.Embedding(${c.T}, ${c.d})   # P: one learned row per position

ids = torch.tensor([[${ids.join(', ')}]])     # this prompt, as token numbers
x = tok_emb(ids) + pos_emb(torch.arange(ids.shape[1]))
print(x.shape)   # (1, ${ids.length}, ${c.d}): one vector per token, the start of the residual stream`;
  py.layerNorm = (c) => `import torch
import torch.nn as nn

class LayerNorm(nn.Module):
    """What nn.LayerNorm(${c.d}) computes, written out."""
    def __init__(self, d=${c.d}, eps=1e-5):
        super().__init__()
        self.gain = nn.Parameter(torch.ones(d))    # γ, learned
        self.bias = nn.Parameter(torch.zeros(d))   # β, learned
        self.eps = eps

    def forward(self, x):                      # x: (..., d)
        mu = x.mean(-1, keepdim=True)
        var = x.var(-1, keepdim=True, unbiased=False)
        x_hat = (x - mu) / torch.sqrt(var + self.eps)   # mean 0, standard deviation 1
        return x_hat * self.gain + self.bias`;
  py.attention = (c) => `import math
import torch
import torch.nn as nn

class CausalSelfAttention(nn.Module):
    def __init__(self, d=${c.d}, n_heads=${c.H}, max_len=${c.T}):
        super().__init__()
        self.h, self.dh = n_heads, d // n_heads          # ${c.H} heads of ${c.d / c.H} dimensions
        self.qkv = nn.Linear(d, 3 * d)                    # Wq, Wk and Wv side by side
        self.out = nn.Linear(d, d)                        # Wo mixes the heads back together
        self.register_buffer('causal', torch.tril(torch.ones(max_len, max_len)).bool())

    def forward(self, x):                                 # x: (batch, n, d), already LayerNormed
        B, n, d = x.shape
        q, k, v = self.qkv(x).split(d, dim=-1)
        # (B, n, d) -> (B, heads, n, dh): each head works in its own slice
        q, k, v = (t.view(B, n, self.h, self.dh).transpose(1, 2) for t in (q, k, v))
        scores = q @ k.transpose(-2, -1) / math.sqrt(self.dh)        # (B, heads, n, n)
        scores = scores.masked_fill(~self.causal[:n, :n], float('-inf'))  # no looking ahead
        weights = scores.softmax(dim=-1)                  # each row sums to 1
        y = weights @ v                                   # weighted mix of values
        y = y.transpose(1, 2).reshape(B, n, d)            # concatenate the heads
        return self.out(y)`;
  py.head = (c, l, h) => `import math
import torch

def attention_head(x, Wq, Wk, Wv):
    """One head (block ${l + 1}, head ${h + 1}). x: (n, ${c.d}) normalised stream; W*: (${c.d}, ${c.d / c.H}) slices."""
    q, k, v = x @ Wq, x @ Wk, x @ Wv                  # (n, ${c.d / c.H}) each
    scores = q @ k.T / math.sqrt(${c.d / c.H})              # how well each query matches each key
    n = len(x)
    causal = torch.tril(torch.ones(n, n)).bool()
    scores = scores.masked_fill(~causal, float('-inf'))   # future tokens get weight 0
    weights = scores.softmax(dim=-1)                  # row i: where token i looks
    return weights @ v                                # (n, ${c.d / c.H}): what each token collects

# In the full layer the ${c.H} heads run side by side; their outputs are concatenated
# into ${c.d} numbers and multiplied by Wo.`;
  py.mlp = (c) => `import torch.nn as nn
import torch.nn.functional as F

class MLP(nn.Module):
    """Works on each token separately: expand, bend, project back."""
    def __init__(self, d=${c.d}, hidden=${c.F}):
        super().__init__()
        self.up = nn.Linear(d, hidden)       # W1, b1: ${c.d} -> ${c.F}
        self.down = nn.Linear(hidden, d)     # W2, b2: ${c.F} -> ${c.d}

    def forward(self, x):
        u = self.up(x)
        g = F.gelu(u, approximate='tanh')    # GELU(u) = u·Φ(u), smooth ReLU
        return self.down(g)`;
  py.block = (c) => `import torch.nn as nn

class Block(nn.Module):
    """Pre-norm transformer block (as in GPT-2)."""
    def __init__(self, d=${c.d}):
        super().__init__()
        self.ln1 = nn.LayerNorm(d)
        self.attn = CausalSelfAttention(d)
        self.ln2 = nn.LayerNorm(d)
        self.mlp = MLP(d)

    def forward(self, x):
        x = x + self.attn(self.ln1(x))   # mix information between positions
        x = x + self.mlp(self.ln2(x))    # process each position
        return x                          # the residual stream, updated`;
  py.residual = () => `# A residual connection: add the sublayer's output to its input.
x = x + sublayer(norm(x))
# The gradient of x + f(x) with respect to x is 1 + f'(x): there is always a direct
# path back to early layers, so deep stacks keep training.`;
  py.head_out = (c) => `import torch
import torch.nn as nn

ln_f = nn.LayerNorm(${c.d})
unembed = nn.Linear(${c.d}, ${c.V})          # one score (logit) per word

logits = unembed(ln_f(x))                # (batch, n, ${c.V})
probs = logits.softmax(dim=-1)           # next-word probabilities at every position

# training: make the word that really came next more likely
loss = nn.functional.cross_entropy(logits[:, :-1].reshape(-1, ${c.V}), ids[:, 1:].reshape(-1))`;
  py.unembed = (c) => `unembed = nn.Linear(${c.d}, ${c.V})
logits = unembed(ln_f(x))      # logit_w = x̂ · U[:, w] + b_w for every word w
# the logit lens: decode an earlier stage the same way
lens = unembed(ln_f(stream_after_block_1)).softmax(-1)`;
  py.softmax = () => `import torch

def softmax(z):
    z = z - z.max()                # same result, avoids overflow
    e = torch.exp(z)
    return e / e.sum()

def sample_next(logits, temperature=0.7):
    if temperature == 0:
        return int(logits.argmax())            # always the top word
    probs = softmax(logits / temperature)       # < 1 sharpens, > 1 flattens
    return int(torch.multinomial(probs, 1))`;
  py.model = (c) => `import math
import torch
import torch.nn as nn
import torch.nn.functional as F

${py.config(c)}
${py.tokenizer().split('\n\ntokenize(')[0]}


${py.attention(c).split('\n').slice(4).join('\n')}


${py.mlp(c).split('\n').slice(3).join('\n')}


${py.block(c).split('\n').slice(2).join('\n')}


class TinyGPT(nn.Module):
    def __init__(self):
        super().__init__()
        self.tok = nn.Embedding(V, D)                 # token embedding E
        self.pos = nn.Embedding(T, D)                 # position embedding P
        self.blocks = nn.ModuleList([Block(D) for _ in range(L)])
        self.ln_f = nn.LayerNorm(D)
        self.unembed = nn.Linear(D, V)

    def forward(self, ids):                           # ids: (batch, n) token numbers
        n = ids.shape[1]
        x = self.tok(ids) + self.pos(torch.arange(n, device=ids.device))
        for block in self.blocks:
            x = block(x)
        return self.unembed(self.ln_f(x))             # logits: (batch, n, V)


@torch.no_grad()
def generate(model, text, temperature=0.7, max_words=10):
    ids = tokenize(text)
    for _ in range(max_words):
        logits = model(torch.tensor([ids[-T:]]))[0, -1]
        probs = (logits / max(temperature, 1e-6)).softmax(-1)
        ids.append(int(torch.multinomial(probs, 1)))
        if VOCAB[ids[-1]] == '.':
            break
    return ' '.join(VOCAB[i] for i in ids[1:])


def train(model, make_story, steps=8000, batch=16):
    """make_story() returns one story as token numbers (see the lab's story generator)."""
    decay = [p for p in model.parameters() if p.dim() == 2]
    other = [p for p in model.parameters() if p.dim() < 2]
    opt = torch.optim.AdamW([{'params': decay, 'weight_decay': 0.01},
                             {'params': other, 'weight_decay': 0.0}], lr=1e-3, betas=(0.9, 0.99))
    warmup = torch.optim.lr_scheduler.LambdaLR(opt, lambda s: min(1.0, (s + 1) / 100))
    for step in range(steps):
        stories = [torch.tensor(make_story()) for _ in range(batch)]
        ids = nn.utils.rnn.pad_sequence(stories, batch_first=True, padding_value=-100)
        logits = model(ids[:, :-1].clamp(min=0))
        loss = F.cross_entropy(logits.reshape(-1, V), ids[:, 1:].reshape(-1), ignore_index=-100)
        opt.zero_grad()
        loss.backward()
        nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step(); warmup.step()


model = TinyGPT()
print(sum(p.numel() for p in model.parameters()))   # ${fmtInt(c.paramCount || 0).replace(/,/g, '_')} parameters`;

  // ------------------------------------------------------------------ drawing helpers
  function tokenColumn(ctx, S, tr, r, size, pos) {
    const ch = r.h / tr.n;
    for (let i = 0; i < tr.n; i++) S.text(ctx, word(tr.ids[i]), r.x + r.w, r.y + ch * (i + 0.5), { size: Math.min(size, ch * 0.9), mono: true, align: 'right', w: i === pos ? 700 : 500, color: i === pos ? theme.c('ink') : theme.c('ink-2'), maxW: r.w });
  }
  function lnStats(x) {
    let mu = 0; for (const v of x) mu += v; mu /= x.length;
    let v2 = 0; for (const v of x) v2 += (v - mu) ** 2; v2 /= x.length;
    return { mu, sd: Math.sqrt(v2 + 1e-5) };
  }
  const topk = (P, o, V, k) => Array.from({ length: V }, (_, c) => c).sort((a, b) => P[o + b] - P[o + a]).slice(0, k).map((id) => ({ id, p: P[o + id] }));

  // ------------------------------------------------------------------ the tree
  /* size: the level's content box. Every level fills the screen, so each one chooses a wide (landscape) or
     tall (portrait) arrangement from it. */
  function build(TL, size) {
    const wide = size.w / size.h >= 1.05, W2 = (w, t) => (wide ? w : t);
    const m = TL.model, c = Object.assign({ paramCount: m.paramCount }, m.cfg), tr = TL.trace(), pos = TL.selPos(tr), n = tr.n, d = c.d, dh = d / c.H, V = c.V, sel = `"${word(tr.ids[pos])}" (position ${pos})`;
    const A = TL.analysis && TL.analysis.version === TL.version ? TL.analysis : null;

    const tokens = {
      id: 'tokens', title: 'Tokens', sub: `${n} of ${c.T}`, rect: [0, 0, 1, 0.08],
      summary: (ctx, r, S) => {
        const w = r.w / n;
        for (let i = 0; i < n; i++) S.text(ctx, word(tr.ids[i]), r.x + w * (i + 0.5), r.y + r.h / 2, { size: Math.min(11, w / 3.2), mono: true, align: 'center', color: i === pos ? theme.c('ink') : theme.c('ink-2'), w: i === pos ? 700 : 500, maxW: w - 2 });
      },
      inner: (ctx, r, S) => {
        const u = S.u(r, 8, 14), per = wide ? n : Math.ceil(n / 2), rowsN = Math.ceil(n / per), w = r.w / per, bh = Math.min(r.h / (rowsN * 3.2), u * 4);
        S.text(ctx, `"${TL.text.trim()}"`, r.x, r.y + u, { size: u, mono: true, maxW: r.w, color: theme.c('ink-2') });
        S.text(ctx, '↓ split into words, add ⟨bos⟩, look each up', r.x, r.y + u * 2.8, { size: u * 0.85, color: theme.c('muted') });
        for (let i = 0; i < n; i++) {
          const rr = Math.floor(i / per), cc = i % per, x = r.x + w * (cc + 0.5), y = r.y + u * 5 + rr * bh * 3.1;
          S.box(ctx, { x: x - w * 0.45, y, w: w * 0.9, h: bh * 1.2 }, { fill: theme.c('panel-2'), stroke: i === pos ? theme.c('focus') : theme.c('line-2') });
          S.text(ctx, word(tr.ids[i]), x, y + bh * 0.6, { size: Math.min(u, w / 4), mono: true, align: 'center', w: 650, maxW: w * 0.86 });
          S.text(ctx, String(tr.ids[i]), x, y + bh * 1.9, { size: Math.min(u, w / 3), mono: true, align: 'center', color: theme.c('pos') });
        }
      },
      about: [
        'The tokenizer turns text into a list of integers. Here it splits on spaces, keeps the full stop as its own token and looks each word up in a 36-word vocabulary; unknown words are dropped. A ⟨bos⟩ ("beginning of sequence") token always comes first, so even the first real word has something to attend to.',
        'Large language models do the same with sub-word pieces learned by byte-pair encoding: frequent words are one token, rare ones are split into pieces, so any text maps to a vocabulary of 50,000 to 200,000 tokens. From here on the model never sees letters, only these numbers (blue).',
      ],
      code: py.tokenizer(),
    };

    const embedSel = {
      id: 'embsel', title: `Building ${sel}`, sub: 'one token, all 32 numbers', rect: W2([0.36, 0.68, 0.64, 0.3], [0, 0.72, 1, 0.26]),
      summary: (ctx, r, S) => { S.bars(ctx, row(tr.resid[0], pos, d), r); },
      inner: (ctx, r, S) => {
        const e = row(m.E.w, tr.ids[pos], d), p = row(m.Pos.w, pos, d), x = row(tr.resid[0], pos, d);
        let mx = 1e-6; for (let j = 0; j < d; j++) mx = Math.max(mx, Math.abs(e[j]), Math.abs(p[j]), Math.abs(x[j]));
        const rows = [['E[' + word(tr.ids[pos]) + ']', e, 'its word\'s row of the token table'], ['+ P[' + pos + ']', p, 'its position\'s row of the position table'], ['= x₀', x, 'what enters block 1']], lh = r.h / 3.3, u = S.u(r, 8, 14);
        rows.forEach(([lab, v, note], k) => {
          const y = r.y + lh * k * 1.08;
          S.text(ctx, lab, r.x, y + lh * 0.18, { size: u, mono: true, w: k === 2 ? 700 : 600 });
          S.text(ctx, note, r.x + r.w, y + lh * 0.18, { size: u * 0.8, align: 'right', color: theme.c('muted'), maxW: r.w * 0.6 });
          S.bars(ctx, v, { x: r.x, y: y + lh * 0.32, w: r.w, h: lh * 0.62 }, { max: mx });
        });
      },
      about: [`The vector that enters block 1 for ${sel}: its word's row of the token table plus its position's row of the position table, added number by number. Each bar is one of the ${d} numbers; tall bars dominate the dot products attention computes next.`, 'The same word at another position gets a different vector, which is the only way the model knows where a word sits.'],
      code: `x0 = tok_emb.weight[${tr.ids[pos]}] + pos_emb.weight[${pos}]   # "${word(tr.ids[pos])}" at position ${pos}\nprint(x0.shape)   # torch.Size([${d}])`,
    };
    const embed = {
      id: 'embed', title: 'Embedding', sub: `token + position → ${d} numbers each`, rect: [0, 0.1, 1, 0.12], tone: 's3',
      summary: (ctx, r, S) => { S.heat(ctx, (i, j) => tr.resid[0][i * d + j], n, d, { x: r.x + r.w * 0.15, y: r.y, w: r.w * 0.7, h: Math.max(4, r.h) }); },
      inner: (ctx, r, S) => {
        const u = S.u(r, 8, 13), ids = new Set(tr.ids);
        const er = S.place(r, W2([0, 0.06, 0.3, 0.88], [0, 0.05, 0.46, 0.4])), pr = S.place(r, W2([0.36, 0.06, 0.26, 0.5], [0.54, 0.05, 0.46, 0.26]));
        const xr = S.place(r, W2([0.74, 0.06, 0.26, 0.5], [0.22, 0.5, 0.78, 0.18]));
        S.caption(ctx, `token table E (${V} × ${d})`, er, { size: u * 0.85 });
        const g = S.heat(ctx, (i, j) => m.E.w[i * d + j], V, d, er);
        ctx.strokeStyle = theme.c('focus'); ctx.lineWidth = 1;
        for (const id of ids) ctx.strokeRect(er.x - 0.5, er.y + id * g.ch - 0.5, g.w + 1, g.ch + 1);
        S.caption(ctx, `position table P (${c.T} × ${d})`, pr, { size: u * 0.85 });
        const g2 = S.heat(ctx, (i, j) => m.Pos.w[i * d + j], c.T, d, pr);
        ctx.strokeStyle = theme.c('focus'); ctx.strokeRect(pr.x - 0.5, pr.y - 0.5, g2.w + 1, g2.ch * n + 1);
        S.caption(ctx, `x₀ = E[token] + P[position]  (${n} × ${d})`, xr, { size: u * 0.85, minW: r.w * 0.4 });
        S.heat(ctx, (i, j) => tr.resid[0][i * d + j], n, d, xr, { hiRow: pos });
        tokenColumn(ctx, S, tr, { x: xr.x - Math.min(r.w * 0.12, 70) - 4, y: xr.y, w: Math.min(r.w * 0.12, 70), h: xr.h }, u * 0.85, pos);
        S.text(ctx, 'outlined: the rows this prompt uses', er.x, er.y + g.h + u, { size: u * 0.8, color: theme.c('muted'), maxW: er.w + 40 });
      },
      children: [embedSel],
      about: [
        `The embedding layer is two lookup tables. The token table E has one learned row of ${d} numbers for each of the ${V} words; the position table P has one row for each of the ${c.T} positions. A token's starting vector is the sum of its word's row and its position's row: x₀ = E[token] + P[position].`,
        'Attention by itself treats its input as an unordered set, so the position vectors are the only thing that tells "alice likes tea" from "tea likes alice". Both tables are ordinary weights, learned by gradient descent with everything else; words that appear in the same kinds of places end up with similar rows (see the Embeddings tab).',
        'Outlined rows of E are the words in this prompt; the outlined band of P is the positions in use. Open the box below to watch one token being assembled.',
      ],
      code: py.embedding(c, Array.from(tr.ids)),
    };

    const lnNode = (id, title, l, which) => {
      const xin = which === 'f' ? row(tr.resid[tr.resid.length - 1], pos, d) : row(tr.resid[which === 1 ? 2 * l : 2 * l + 1], pos, d);
      const out = which === 'f' ? row(tr.lnf, pos, d) : row(which === 1 ? tr.ln1[l] : tr.ln2[l], pos, d);
      const st = lnStats(xin), g = which === 'f' ? m.lnfg.w : which === 1 ? m.layers[l].ln1g.w : m.layers[l].ln2g.w;
      return {
        id, title, sub: 'normalise each token', rect: [0, 0, 1, 1],
        summary: (ctx, r, S) => { S.text(ctx, '(x − μ) / σ', r.x + r.w / 2, r.y + r.h * 0.4, { size: clamp(r.w / 8, 6, 11), mono: true, align: 'center', color: theme.c('ink-2') }); },
        inner: (ctx, r, S) => {
          const u = S.u(r, 8, 14);
          const a = S.place(r, [0, 0.08, 1, 0.26]), b = S.place(r, [0, 0.55, 1, 0.26]);
          S.caption(ctx, `stream for ${sel}: ${d} numbers in`, a, { size: u * 0.85 });
          S.bars(ctx, xin, a);
          S.lines(ctx, [`μ = mean = ${fmt(st.mu, 3)}      σ = std = ${fmt(st.sd, 3)}`, 'x̂ = (x − μ) / σ · γ + β'], S.place(r, [0, 0.37, 1, 0.13]), { mono: true, size: u, strong: (i) => i === 1 });
          S.caption(ctx, 'normalised: mean 0, spread 1, then scaled by γ and shifted by β', b, { size: u * 0.85 });
          S.bars(ctx, out, b);
          S.text(ctx, `learned γ runs from ${fmt(Math.min(...g), 2)} to ${fmt(Math.max(...g), 2)}: some coordinates are amplified, others muted`, r.x, r.y + r.h * 0.9, { size: u * 0.85, color: theme.c('muted'), maxW: r.w });
        },
        about: [
          'LayerNorm rescales one token\'s vector at a time: subtract its mean μ, divide by its standard deviation σ, then multiply by a learned gain γ and add a learned bias β, one coordinate at a time: x̂ = (x − μ) / √(σ² + ε) · γ + β.',
          `It keeps the numbers in a steady range however many blocks have written into the stream, which keeps training stable. This model uses the "pre-norm" layout of GPT-2: the sublayer reads a normalised copy, and the stream itself is never normalised, so nothing written earlier is lost. For ${sel} right now: μ = ${fmt(st.mu, 3)}, σ = ${fmt(st.sd, 3)}.`,
        ],
        code: py.layerNorm(c),
      };
    };

    const headNode = (l, h) => {
      const qkv = tr.qkv[l], Aw = tr.attn[l][h], d3 = 3 * d, scale = 1 / Math.sqrt(dh), off = !m.headMask[l][h];
      const Q = (i, j) => qkv[i * d3 + h * dh + j], K = (i, j) => qkv[i * d3 + d + h * dh + j], Vv = (i, j) => qkv[i * d3 + 2 * d + h * dh + j];
      const score = (i, j) => { let s = 0; for (let t = 0; t < dh; t++) s += Q(i, t) * K(j, t); return s * scale; };
      const out = (i, t) => { let s = 0; for (let j = 0; j <= i; j++) s += Aw[i * n + j] * Vv(j, t); return s; };
      const role = A ? A.heads.find((x) => x.l === l && x.h === h) : null;
      const keys = Array.from({ length: pos + 1 }, (_, j) => j).sort((a, b) => Aw[pos * n + b] - Aw[pos * n + a]);
      return {
        id: 'h' + h, title: `Head ${h + 1}`, sub: off ? 'knocked out' : role ? role.role.label : `${dh} dims`,
        summary: (ctx, r, S) => { const s = Math.min(r.w, r.h); S.heat(ctx, (i, j) => (j <= i ? Aw[i * n + j] : 0), n, n, { x: r.x + (r.w - s) / 2, y: r.y, w: s, h: s }, { seq: true, max: 1, mask: (i, j) => j > i }); },
        inner: (ctx, r, S) => {
          const u = S.u(r, 8, 13);
          const lay = W2({ lab: [0, 0.07, 0.1, 0.38], q: [[0.11, 0.07, 0.11, 0.38], [0.24, 0.07, 0.11, 0.38], [0.37, 0.07, 0.11, 0.38]], math: [0.53, 0.02, 0.47, 0.46], s: [0.11, 0.56, 0.24, 0.42], w: [0.4, 0.56, 0.24, 0.42], o: [0.7, 0.56, 0.1, 0.42] },
            { lab: [0, 0.05, 0.16, 0.24], q: [[0.17, 0.05, 0.25, 0.24], [0.45, 0.05, 0.25, 0.24], [0.73, 0.05, 0.25, 0.24]], math: [0, 0.32, 1, 0.22], s: [0.0, 0.6, 0.38, 0.38], w: [0.42, 0.6, 0.38, 0.38], o: [0.84, 0.6, 0.16, 0.38] });
          tokenColumn(ctx, S, tr, S.place(r, lay.lab), u * 0.85, pos);
          [['q = x̂·Wq', Q], ['k = x̂·Wk', K], ['v = x̂·Wv', Vv]].forEach(([t, f], k) => { const br = S.place(r, lay.q[k]); S.caption(ctx, t, br, { size: u * 0.85 }); S.heat(ctx, f, n, dh, br, { hiRow: pos }); });
          const sr = S.place(r, lay.s), wr = S.place(r, lay.w), orr = S.place(r, lay.o);
          S.caption(ctx, 'scores q·k / √' + dh, sr, { size: u * 0.85 });
          const g = S.heat(ctx, (i, j) => (j <= i ? score(i, j) : 0), n, n, sr, { mask: (i, j) => j > i, hiRow: pos, square: true });
          S.caption(ctx, 'softmax → weights', wr, { size: u * 0.85 });
          S.heat(ctx, (i, j) => (j <= i ? Aw[i * n + j] : 0), n, n, wr, { seq: true, max: 1, mask: (i, j) => j > i, hiRow: pos, square: true });
          S.arrow(ctx, sr.x + g.w + 3, sr.y + g.h / 2, wr.x - 4, wr.y + g.h / 2, { lw: 1, head: 5 });
          S.caption(ctx, 'weights · v', orr, { size: u * 0.85, minW: 60 });
          S.heat(ctx, out, n, dh, { x: orr.x, y: orr.y, w: orr.w, h: g.h }, { hiRow: pos });
          S.text(ctx, 'grey: masked future tokens (score −∞, weight 0)', sr.x, sr.y + g.h + u * 0.9, { size: u * 0.8, color: theme.c('muted'), maxW: r.w * 0.9 });
          const lines = [`query ${sel}:`, ...keys.slice(0, 6).map((j) => `${word(tr.ids[j]).padEnd(7)} s = ${fmt(score(pos, j), 2).padStart(6)}   w = ${fmtPct(Aw[pos * n + j], 0).padStart(4)}`)];
          if (keys.length > 6) lines.push(`… ${keys.length - 6} more keys share ${fmtPct(keys.slice(6).reduce((s, j) => s + Aw[pos * n + j], 0), 0)}`);
          lines.push('w = e^s / Σ e^s');
          S.lines(ctx, lines, S.place(r, lay.math), { mono: true, size: u * 0.95, strong: (i) => i === 0, colorOf: (i) => (i === 0 ? theme.c('ink') : theme.c('ink-2')) });
        },
        about: [
          `Head ${h + 1} of block ${l + 1} works in its own ${dh}-number slice. Every token makes a query q, a key k and a value v from its normalised stream vector (the three ${n} × ${dh} grids). The score between query i and key j is their dot product divided by √${dh}: s = q·k / √${dh}. The division stops scores from growing with the dimension, so the softmax does not saturate.`,
          'Scores for later tokens are set to −∞ (grey) before the softmax, which turns each row into weights that are zero or positive and add up to 1. The output for token i is the weighted sum of the value vectors, Σⱼ wᵢⱼ vⱼ: the head copies a blend of what the tokens it attends to carry.',
          `For ${sel} this head puts ${fmtPct(Aw[pos * n + keys[0]], 0)} of its weight on "${word(tr.ids[keys[0]])}" (position ${keys[0]}). ${role ? role.role.text : 'The Attention tab measures what each head is for by switching it off.'}${off ? ' This head is knocked out: its output is replaced with zeros.' : ''}`,
        ],
        code: py.head(c, l, h),
      };
    };

    const attnNode = (l) => {
      const qkv = tr.qkv[l], d3 = 3 * d;
      const headRects = W2(Array.from({ length: c.H }, (_, h) => [0.31 + (h % 2) * 0.24, 0.04 + (h >> 1) * 0.46, 0.22, 0.4]), Array.from({ length: c.H }, (_, h) => [0.02 + (h % 2) * 0.5, 0.34 + (h >> 1) * 0.22, 0.46, 0.19]));
      const heads = Array.from({ length: c.H }, (_, h) => Object.assign(headNode(l, h), { rect: headRects[h] }));
      return {
        id: 'attn', title: 'Attention', sub: `${c.H} heads × ${dh} dims`, tone: 's1',
        summary: (ctx, r, S) => {
          const s = Math.max(4, Math.min(r.h, r.w / c.H - 3));
          for (let h = 0; h < c.H; h++) S.heat(ctx, (i, j) => (j <= i ? tr.attn[l][h][i * n + j] : 0), n, n, { x: r.x + (r.w - c.H * (s + 3)) / 2 + h * (s + 3), y: r.y, w: s, h: s }, { seq: true, max: 1, mask: (i, j) => j > i });
        },
        inner: (ctx, r, S) => {
          const u = S.u(r, 8, 13);
          const lay = W2({ x: [0, 0.1, 0.09, 0.6], qkv: [[0.13, 0.04, 0.13, 0.26], [0.13, 0.36, 0.13, 0.26], [0.13, 0.68, 0.13, 0.26]], cat: [0.82, 0.06, 0.17, 0.36], out: [0.82, 0.56, 0.17, 0.36], f: [0, 0.985] },
            { x: [0, 0.04, 0.2, 0.22], qkv: [[0.27, 0.04, 0.22, 0.22], [0.52, 0.04, 0.22, 0.22], [0.77, 0.04, 0.22, 0.22]], cat: [0.02, 0.82, 0.44, 0.13], out: [0.54, 0.82, 0.44, 0.13], f: [0, 0.995] });
          const xr = S.place(r, lay.x);
          S.caption(ctx, 'x̂ (normalised)', xr, { size: u * 0.85, minW: 80 });
          S.heat(ctx, (i, j) => tr.ln1[l][i * d + j], n, d, xr, { hiRow: pos });
          [['Q', 0], ['K', d], ['V', 2 * d]].forEach(([t, o], k) => {
            const pr = S.place(r, lay.qkv[k]);
            S.caption(ctx, `${t} = x̂·W${t.toLowerCase()}`, pr, { size: u * 0.85, minW: 70 });
            S.heat(ctx, (i, j) => qkv[i * d3 + o + j], n, d, pr, { hiRow: pos });
            if (wide) S.arrow(ctx, xr.x + xr.w + 3, xr.y + xr.h / 2, pr.x - 4, pr.y + pr.h / 2, { lw: 0.9, head: 4 });
          });
          const cr = S.place(r, lay.cat), or = S.place(r, lay.out);
          const cat = (i, j) => { const h = Math.floor(j / dh), t = j % dh; if (!m.headMask[l][h]) return 0; let s = 0; for (let q = 0; q <= i; q++) s += tr.attn[l][h][i * n + q] * qkv[q * d3 + 2 * d + h * dh + t]; return s; };
          S.caption(ctx, 'concatenate the heads', cr, { size: u * 0.85, minW: 80 });
          S.heat(ctx, cat, n, d, cr, { hiRow: pos });
          S.caption(ctx, '× Wo → added to the stream', or, { size: u * 0.85, minW: 90 });
          S.heat(ctx, (i, j) => tr.resid[2 * l + 1][i * d + j] - tr.resid[2 * l][i * d + j], n, d, or, { hiRow: pos });
          S.text(ctx, `per head: weights = softmax(QKᵀ/√${dh} + causal mask);  output = weights·V`, r.x, r.y + r.h * lay.f[1], { size: u * 0.85, mono: true, maxW: r.w, color: theme.c('ink-2') });
        },
        children: heads,
        about: [
          'Self-attention is how tokens share information. From its normalised stream vector x̂ every token makes three vectors with learned matrices: a query Q = x̂·Wq (what it is looking for), a key K = x̂·Wk (what it offers) and a value V = x̂·Wv (what it passes on). Each query is compared with every earlier key; the better the match, the more of that token\'s value is copied.',
          `The ${d} numbers are split into ${c.H} heads of ${dh}, each with its own slice of Wq, Wk and Wv, so the layer can look for ${c.H} things at once. In this model a head in block 2 finds the subject of a sentence, and two heads in block 1 gather the drinks mentioned so far. The heads' outputs are placed side by side (concatenated), mixed by the output matrix Wo, and added to the stream.`,
          'A causal mask hides later tokens, so a token can only use what came before it. That is what lets the model learn from every position of a story at once and still generate one word at a time. Open a head (the four squares) to see its queries, keys, scores and weights for this prompt.',
        ],
        code: py.attention(c),
      };
    };

    const mlpNode = (l) => {
      const pre = row(tr.pre[l], pos, c.F), g = row(tr.mlpAct[l], pos, c.F), xin = row(tr.ln2[l], pos, d), out = row(tr.mlp[l], pos, d);
      const top = Array.from({ length: c.F }, (_, j) => j).sort((a, b) => g[b] - g[a]).slice(0, 5);
      return {
        id: 'mlp', title: 'MLP', sub: `${d} → ${c.F} → ${d}`, tone: 's4',
        summary: (ctx, r, S) => { S.bars(ctx, g, { x: r.x, y: r.y, w: r.w, h: r.h * 0.85 }); },
        inner: (ctx, r, S) => {
          const u = S.u(r, 8, 13);
          const lay = W2({ a: [0, 0.07, 0.22, 0.2], b: [0.3, 0.07, 0.7, 0.2], gc: [0, 0.4, 0.22, 0.24], c2: [0.3, 0.4, 0.7, 0.2], o: [0, 0.78, 0.22, 0.18], top: [0.3, 0.7, 0.7, 0.29] },
            { a: [0, 0.05, 0.42, 0.12], b: [0, 0.24, 1, 0.12], gc: [0.55, 0.02, 0.45, 0.16], c2: [0, 0.45, 1, 0.12], o: [0, 0.65, 0.42, 0.11], top: [0.48, 0.62, 0.52, 0.36] });
          const a = S.place(r, lay.a), b = S.place(r, lay.b), gc = S.place(r, lay.gc), c2 = S.place(r, lay.c2), o = S.place(r, lay.o);
          S.caption(ctx, `x̂ (${d})`, a, { size: u * 0.85 }); S.bars(ctx, xin, a);
          S.caption(ctx, `u = W₁·x̂ + b₁   (${c.F} neurons)`, b, { size: u * 0.85, minW: 140 }); S.bars(ctx, pre, b);
          S.caption(ctx, 'GELU(u) = u·Φ(u)', gc, { size: u * 0.85, minW: 90 });
          S.frame(ctx, gc); S.curve(ctx, (z) => NA.tf.gelu(z), gc, -3, 3, -0.5, 3, { color: theme.c('s4'), lw: 1.8 });
          S.caption(ctx, 'g = GELU(u)', c2, { size: u * 0.85 }); S.bars(ctx, g, c2);
          S.caption(ctx, `out = W₂·g + b₂   (${d})`, o, { size: u * 0.85, minW: 120 }); S.bars(ctx, out, o);
          S.lines(ctx, ['most active neurons:', ...top.map((j) => `#${String(j).padEnd(4)} u = ${fmt(pre[j], 2).padStart(6)}  →  g = ${fmt(g[j], 2)}`)], S.place(r, lay.top), { mono: true, size: u * 0.9, strong: (i) => i === 0 });
        },
        about: [
          `The MLP (feed-forward network) processes each token on its own, with no mixing between positions. It expands the ${d} numbers to ${c.F} with W₁, bends them with GELU, and projects back to ${d} with W₂: MLP(x̂) = W₂·GELU(W₁x̂ + b₁) + b₂.`,
          'GELU(u) = u·Φ(u), where Φ is the normal cumulative distribution: a smooth version of ReLU that passes large positive values, damps negative ones and has a slope everywhere. Each hidden neuron reads one direction of the stream (its row of W₁) and, when it fires, writes another direction back (its column of W₂).',
          `In big models these layers hold most of the parameters and much of the stored knowledge. In this one, knocking out MLP 2 drops lookups from 100% to about 30%: it turns what attention gathered into the answer. The bars show all ${c.F} hidden values for ${sel}.`,
        ],
        code: py.mlp(c),
      };
    };

    const addNode = (id) => ({ id, title: '+', sub: 'residual add', shape: 'op', about: ['Residual connection: the sublayer\'s output is added to the stream, x ← x + f(x), instead of replacing it. The gradient of x + f(x) is 1 + f′(x), so there is always a direct path back to the early layers, and deep stacks keep training.'], code: py.residual() });

    const BLOCK_RECTS = W2(
      { ln1: [0, 0.42, 0.11, 0.42], attn: [0.15, 0.3, 0.3, 0.68], add1: [0.475, 0.0, 0.04, 0.2], ln2: [0.53, 0.42, 0.11, 0.42], mlp: [0.68, 0.3, 0.26, 0.68], add2: [0.955, 0.0, 0.04, 0.2] },
      { ln1: [0.14, 0.0, 0.5, 0.08], attn: [0.14, 0.11, 0.86, 0.33], add1: [0.0, 0.46, 0.08, 0.05], ln2: [0.14, 0.53, 0.5, 0.08], mlp: [0.14, 0.64, 0.86, 0.31], add2: [0.0, 0.95, 0.08, 0.05] });
    const blockNode = (l, rect) => ({
      id: 'block' + l, title: `Block ${l + 1}`, sub: 'attention + MLP, each normalised and added back', rect,
      summary: (ctx, r, S) => {
        const items = ['LayerNorm', 'Attention', '+', 'LayerNorm', 'MLP', '+'], w = r.w / items.length;
        items.forEach((t, i) => S.text(ctx, t, r.x + w * (i + 0.5), r.y + r.h * 0.4, { size: clamp(w / 7, 6, 11), align: 'center', color: t === '+' ? theme.c('ink') : theme.c('ink-2'), w: 600, maxW: w - 4 }));
      },
      inner: (ctx, r, S) => {
        const u = S.u(r, 8, 12), at = (k) => S.place(r, BLOCK_RECTS[k]);
        const ln1 = at('ln1'), att = at('attn'), a1 = at('add1'), ln2 = at('ln2'), mlp = at('mlp'), a2 = at('add2');
        const cen = (q) => [q.x + q.w / 2, q.y + q.h / 2];
        ctx.strokeStyle = theme.c('ink-2'); ctx.lineWidth = 2.4;
        if (wide) {
          const y = cen(a1)[1];
          ctx.beginPath(); ctx.moveTo(r.x, y); ctx.lineTo(r.x + r.w, y); ctx.stroke();
          S.text(ctx, 'residual stream (32 numbers per token) →', r.x, y - u * 1.1, { size: u * 0.85, color: theme.c('muted') });
          S.arrow(ctx, cen(ln1)[0], y, cen(ln1)[0], ln1.y - 3, { lw: 1.1, head: 5 });
          S.arrow(ctx, ln1.x + ln1.w + 2, cen(ln1)[1], att.x - 3, cen(ln1)[1], { lw: 1.1, head: 5 });
          S.arrow(ctx, att.x + att.w + 2, att.y + att.h * 0.25, cen(a1)[0], a1.y + a1.h * 0.5 + Math.min(a1.w, a1.h) / 2 + 2, { lw: 1.1, head: 5, bend: [8, 6] });
          S.arrow(ctx, cen(ln2)[0], y, cen(ln2)[0], ln2.y - 3, { lw: 1.1, head: 5 });
          S.arrow(ctx, ln2.x + ln2.w + 2, cen(ln2)[1], mlp.x - 3, cen(ln2)[1], { lw: 1.1, head: 5 });
          S.arrow(ctx, mlp.x + mlp.w + 2, mlp.y + mlp.h * 0.25, cen(a2)[0], a2.y + a2.h * 0.5 + Math.min(a2.w, a2.h) / 2 + 2, { lw: 1.1, head: 5, bend: [6, 6] });
        } else {
          const x = cen(a1)[0];
          ctx.beginPath(); ctx.moveTo(x, r.y); ctx.lineTo(x, r.y + r.h); ctx.stroke();
          S.arrow(ctx, x, cen(ln1)[1], ln1.x - 3, cen(ln1)[1], { lw: 1.1, head: 5 });
          S.arrow(ctx, cen(ln1)[0], ln1.y + ln1.h + 2, cen(ln1)[0], att.y - 3, { lw: 1.1, head: 5 });
          S.arrow(ctx, att.x + 4, att.y + att.h + 2, x + Math.min(a1.w, a1.h) / 2 + 3, cen(a1)[1], { lw: 1.1, head: 5, bend: [-4, 8] });
          S.arrow(ctx, x, cen(ln2)[1], ln2.x - 3, cen(ln2)[1], { lw: 1.1, head: 5 });
          S.arrow(ctx, cen(ln2)[0], ln2.y + ln2.h + 2, cen(ln2)[0], mlp.y - 3, { lw: 1.1, head: 5 });
          S.arrow(ctx, mlp.x + 4, mlp.y + mlp.h + 2, x + Math.min(a2.w, a2.h) / 2 + 3, cen(a2)[1], { lw: 1.1, head: 5, bend: [-4, 6] });
        }
      },
      children: [Object.assign(lnNode('ln1', 'LayerNorm', l, 1), { rect: BLOCK_RECTS.ln1 }), Object.assign(attnNode(l), { rect: BLOCK_RECTS.attn }), Object.assign(addNode('add1'), { rect: BLOCK_RECTS.add1 }),
        Object.assign(lnNode('ln2', 'LayerNorm', l, 2), { rect: BLOCK_RECTS.ln2 }), Object.assign(mlpNode(l), { rect: BLOCK_RECTS.mlp }), Object.assign(addNode('add2'), { rect: BLOCK_RECTS.add2 })],
      about: [
        `Block ${l + 1} is one round of "gather, then think". Two sublayers are wrapped the same way, normalise, transform, add back: x ← x + Attention(LayerNorm(x)), then x ← x + MLP(LayerNorm(x)).`,
        'Attention moves information between positions; it is the only place where tokens interact. The MLP then transforms each position on its own. Because both add to the stream instead of replacing it, every block makes a correction to a shared running representation, and later blocks can still read what earlier ones wrote.',
        `This model has ${c.L} block${c.L > 1 ? 's' : ''}. For lookups, block 1 collects the drinks and names and block 2 combines them; a one-layer version trained the same way cannot do the lookup at all. Open Attention or the MLP to go deeper.`,
      ],
      code: py.block(c),
    });

    const outNode = (() => {
      const o = pos * V, logits = row(tr.logits, pos, V), top = topk(tr.probs, o, V, 8);
      const R = W2({ lnf: [0, 0.2, 0.2, 0.6], un: [0.26, 0, 0.4, 1], sm: [0.72, 0, 0.28, 1] }, { lnf: [0.1, 0, 0.8, 0.16], un: [0, 0.22, 1, 0.4], sm: [0, 0.68, 1, 0.32] });
      const unemb = {
        id: 'unembed', title: 'Unembedding', sub: `${d} → ${V} scores`, rect: R.un,
        summary: (ctx, r, S) => { S.bars(ctx, logits, { x: r.x, y: r.y, w: r.w, h: r.h * 0.85 }); },
        inner: (ctx, r, S) => {
          const u = S.u(r, 8, 13), mr = S.place(r, W2([0, 0.08, 0.4, 0.86], [0, 0.05, 1, 0.4])), lr = S.place(r, W2([0.48, 0.08, 0.52, 0.56], [0, 0.55, 1, 0.24]));
          S.caption(ctx, `U (${d} × ${V}): one column per word`, mr, { size: u * 0.85, minW: 140 });
          S.heat(ctx, (i, j) => m.U.w[i * V + j], d, V, mr);
          S.caption(ctx, `logits for ${sel}`, lr, { size: u * 0.85, minW: 140 });
          S.bars(ctx, logits, lr);
          S.lines(ctx, top.slice(0, 3).map((t) => `${word(t.id).padEnd(7)} logit ${fmt(logits[t.id], 2)}`), S.place(r, W2([0.48, 0.72, 0.52, 0.26], [0, 0.84, 1, 0.16])), { mono: true, size: u * 0.9 });
        },
        about: [`The unembedding is a ${d} × ${V} matrix with one column per word. A word's score (its logit) is the dot product of the normalised stream vector with that column, plus a bias: words whose columns point the way the stream points get high scores.`, 'The logit lens in the Tokens view applies exactly this to earlier stages, to read what the model "would say" if it stopped there.'],
        code: py.unembed(c),
      };
      const soft = {
        id: 'softmax', title: 'Softmax', sub: 'scores → probabilities', rect: R.sm,
        summary: (ctx, r, S) => { S.text(ctx, `${word(top[0].id)} ${fmtPct(top[0].p)}`, r.x + r.w / 2, r.y + r.h * 0.35, { size: clamp(r.w / 9, 7, 13), mono: true, w: 650, align: 'center', maxW: r.w }); },
        inner: (ctx, r, S) => {
          const u = S.u(r, 8, 14), lh = (r.h * 0.78) / top.length;
          S.text(ctx, 'p(word) = e^logit / Σ e^logit', r.x, r.y + u * 0.7, { size: u, mono: true, maxW: r.w });
          top.forEach((t, i) => {
            const y = r.y + r.h * 0.17 + i * lh;
            S.text(ctx, word(t.id), r.x, y + lh / 2, { size: Math.min(u, lh * 0.8), mono: true, maxW: r.w * 0.3 });
            ctx.fillStyle = theme.c('panel-2'); ctx.fillRect(r.x + r.w * 0.32, y + lh * 0.25, r.w * 0.5, lh * 0.5);
            ctx.fillStyle = theme.c('pos'); ctx.fillRect(r.x + r.w * 0.32, y + lh * 0.25, r.w * 0.5 * t.p, lh * 0.5);
            S.text(ctx, fmtPct(t.p, t.p < 0.1 ? 1 : 0), r.x + r.w, y + lh / 2, { size: Math.min(u * 0.9, lh * 0.8), mono: true, align: 'right' });
          });
        },
        about: ['Softmax turns scores into probabilities: exponentiate each, divide by the total, p = e^z / Σ e^z. Larger scores win more than proportionally, and the results are positive and add up to 1.', 'When generating, the prompt panel\'s temperature divides the scores first: below 1 sharpens the choice toward the top word, above 1 spreads it out, and at 0 the top word is always taken.'],
        code: py.softmax(),
      };
      return {
        id: 'out', title: 'Output', sub: 'LayerNorm → unembedding → softmax', rect: [0, 0.83, 1, 0.17], tone: 's7',
        summary: (ctx, r, S) => { S.text(ctx, `after ${sel}: ${word(top[0].id)} (${fmtPct(top[0].p)})`, r.x + r.w / 2, r.y + r.h * 0.35, { size: clamp(r.h * 0.3, 7, 12), mono: true, align: 'center', maxW: r.w }); },
        inner: (ctx, r, S) => {
          const a = S.place(r, R.lnf), b = S.place(r, R.un), cc = S.place(r, R.sm);
          if (wide) { S.arrow(ctx, a.x + a.w + 2, a.y + a.h / 2, b.x - 3, a.y + a.h / 2, { lw: 1.1, head: 5 }); S.arrow(ctx, b.x + b.w + 2, a.y + a.h / 2, cc.x - 3, a.y + a.h / 2, { lw: 1.1, head: 5 }); }
          else { S.arrow(ctx, a.x + a.w / 2, a.y + a.h + 2, a.x + a.w / 2, b.y - 3, { lw: 1.1, head: 5 }); S.arrow(ctx, b.x + b.w / 2, b.y + b.h + 2, b.x + b.w / 2, cc.y - 3, { lw: 1.1, head: 5 }); }
        },
        children: [Object.assign(lnNode('lnf', 'Final LayerNorm', 0, 'f'), { rect: R.lnf }), unemb, soft],
        about: [
          'At every position the model reads its prediction for the next word out of the stream: a final LayerNorm, then the unembedding (one score per word), then softmax to turn the scores into probabilities.',
          'Training makes the word that really came next more probable. The loss is cross-entropy, −ln p(actual next word), averaged over every position of every story; its gradient flows back through everything above.',
        ],
        code: py.head_out(c),
      };
    })();

    const blocks = [], top0 = 0.25, span = 0.56;
    for (let l = 0; l < c.L; l++) blocks.push(blockNode(l, [0, top0 + (span / c.L) * l, 1, span / c.L - 0.025]));
    const kids = [tokens, embed, ...blocks, outNode];
    return {
      id: 'model', title: 'Tiny Transformer', sub: `${fmtInt(m.paramCount)} parameters`,
      inner: (ctx, r, S) => {
        const rs = kids.map((k) => S.place(r, k.rect));
        for (let i = 0; i + 1 < rs.length; i++) S.arrow(ctx, r.x + r.w / 2, rs[i].y + rs[i].h + 1, r.x + r.w / 2, rs[i + 1].y - 2, { lw: 1.2, head: 5 });
      },
      children: kids,
      about: [
        `A GPT-style language model: it reads a sequence of token numbers and, at every position, predicts the next token. Each token becomes a vector of ${d} numbers; every block then reads from that "residual stream" and adds its result back, so the stream accumulates everything the model has worked out about each token.`,
        `This one is tiny: ${V} words, ${d} numbers per token, ${c.L} block${c.L > 1 ? 's' : ''} of ${c.H}-head attention and ${c.F}-neuron MLPs, ${fmtInt(m.paramCount)} parameters. GPT-2 small uses the same recipe with 50,257 tokens, 768 numbers per token and 12 blocks of 12 heads (124 million parameters); the arithmetic per token is the same.`,
        'Click a part or zoom in to open it, and keep going: a block opens into LayerNorm, attention and the MLP; attention opens into its heads; a head shows its queries, keys, scores and weights for your prompt. Zoom out or use the path at the top to come back. The Python tab shows the part you are looking at in PyTorch, and here, the whole model.',
      ],
      code: py.model(c),
      codeLabel: 'Python · PyTorch · the whole model',
    };
  }

  NA.archTransformer = { build, py };

})(window.NA);
