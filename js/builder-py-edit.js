/* builder-py-edit.js — the model builder's Python tab, editable. Change the script and press Apply: the page compares
   your script with the one it wrote, line by line (comments, docstrings and spacing aside), and looks for the
   settings that would make it write your lines: training settings, each block's size, activation and settings,
   the data and the world, reward amounts and the reward formula, evolution settings, a block whose lines you took
   out. It changes them, writes the script again and keeps the change only if the new script has every change you
   made (lines that follow from it may change too: the next layer reads the new size). Otherwise nothing changes:
   it says which lines it could not follow and why, puts the script back, and keeps your edit one press away. The
   Python is never run. The writer (builder-py.js, builder-py-worlds.js) is used as it is: a setting is found by
   changing it and seeing which lines change. PE.solve(setup, text, opts) does the reading; PE.render(lab, root)
   draws the tab. */
(function (NA) {
  'use strict';
  const PE = {};
  const BT = NA.builderTasks;
  const now = () => (typeof performance !== 'undefined' ? performance.now() : Date.now());

  // ------------------------------------------------------------------ reading Python: tokens and lines
  const KW = new Set('False None True and as assert async await break class continue def del elif else except finally for from global if import in is lambda nonlocal not or pass raise return try while with yield'.split(' '));
  const OPS = ['**=', '//=', '>>=', '<<=', '->', '**', '//', '==', '!=', '<=', '>=', '+=', '-=', '*=', '/=', '%=', '&=', '|=', '^=', '@=', ':=', '<<', '>>'];
  const OPEN = { '(': ')', '[': ']', '{': '}' };
  const RE_NAME = /[\p{L}_][\p{L}\p{N}_]*/uy, RE_NUM = /(?:0[xXoObB][\da-fA-F_]+|(?:\d[\d_]*(?:\.[\d_]*)?|\.\d[\d_]*)(?:[eE][+-]?\d[\d_]*)?)[jJ]?/y;
  const numOf = (s) => Number(String(s).replace(/_/g, ''));
  const sig = (v) => String(+(+v).toPrecision(6));
  const same = (a, b) => typeof a === 'number' && typeof b === 'number' && (a === b || Math.abs(a - b) <= 1e-9 * Math.max(1, Math.abs(a), Math.abs(b)) || sig(a) === sig(b));
  /* Python source → tokens { t: 'num' | 'str' | 'name' | 'op' | 'com' | 'nl', v, line } and the first problem a
     reader can see without running it (a string or bracket left open, a bracket that does not match), { line, msg }. */
  function tokenize(src) {
    const toks = [], stack = [], n = src.length;
    let i = 0, line = 1, err = null;
    while (i < n) {
      const c = src[i];
      if (c === '\n') { toks.push({ t: 'nl', v: '\n', line, soft: stack.length > 0 }); line++; i++; continue; }
      if (c === ' ' || c === '\t' || c === '\r' || c === '\f') { i++; continue; }
      if (c === '\\' && src[i + 1] === '\n') { i += 2; line++; continue; } // a line joined to the next
      if (c === '#') { let j = src.indexOf('\n', i); if (j < 0) j = n; toks.push({ t: 'com', v: src.slice(i, j), line }); i = j; continue; }
      let pre = '';
      RE_NAME.lastIndex = i;
      const w = RE_NAME.exec(src);
      if (w && /^[rRbBuUfF]{1,2}$/.test(w[0]) && /["']/.test(src[i + w[0].length] || '')) { pre = w[0].toLowerCase(); i += w[0].length; }
      else if (w) { toks.push({ t: 'name', v: w[0], line }); i += w[0].length; continue; }
      const q0 = src[i];
      if (q0 === '"' || q0 === "'") {
        const q = src.startsWith(q0 + q0 + q0, i) ? q0 + q0 + q0 : q0, l0 = line, start = i;
        let j = i + q.length, ok = false;
        while (j < n) {
          const ch = src[j];
          if (ch === '\\') { if (src[j + 1] === '\n') line++; j += 2; continue; }
          if (ch === '\n') { if (q.length === 1) break; line++; }
          if (src.startsWith(q, j)) { j += q.length; ok = true; break; }
          j++;
        }
        if (!ok) err = err || { line: l0, msg: 'This string is never closed: it needs a quote at its end.' };
        j = Math.min(j, n);
        toks.push({ t: 'str', v: pre + src.slice(start, j), pre, body: src.slice(start + q.length, ok ? j - q.length : j), line: l0 });
        i = j;
        continue;
      }
      RE_NUM.lastIndex = i;
      const m = /[\d.]/.test(c) ? RE_NUM.exec(src) : null;
      if (m && m[0] !== '.') { toks.push({ t: 'num', v: m[0], line }); i += m[0].length; continue; }
      const op = OPS.find((o) => src.startsWith(o, i)) || c;
      if (OPEN[op]) stack.push({ ch: op, line });
      else if (op === ')' || op === ']' || op === '}') {
        const top = stack.pop();
        if (!top || OPEN[top.ch] !== op) err = err || { line, msg: top ? `This “${op}” closes the “${top.ch}” opened on line ${top.line}.` : `This “${op}” closes a bracket that was never opened.` };
      } else if (!/^[-+*/%@&|^~<>=,:.;!]/.test(op)) err = err || { line, msg: `“${op}” is not Python.` };
      toks.push({ t: 'op', v: op, line });
      i += op.length;
    }
    if (stack.length) { const top = stack[stack.length - 1]; err = err || { line: top.line, msg: `The “${top.ch}” opened here is never closed.` }; }
    return { toks, err };
  }
  // a token as it is compared: numbers by value (to 6 significant digits, as the page writes them), strings by what is inside the quotes
  const canon = (tk) => (tk.t === 'num' ? '#' + (Number.isFinite(numOf(tk.v)) ? sig(numOf(tk.v)) : tk.v) : tk.t === 'str' ? `"${tk.pre}|${tk.body}` : tk.v);
  const valueEnd = (tk) => !!tk && (tk.t === 'num' || tk.t === 'str' || (tk.t === 'name' && (!KW.has(tk.v) || /^(True|False|None)$/.test(tk.v))) || (tk.t === 'op' && /^[)\]}]$/.test(tk.v)));
  /* The statements of a script (one can run over several lines inside brackets), without comments, docstrings and
     blank lines: { toks, tk (as compared), key, line (where it starts) }. A minus sign joins the number after it when
     it is a sign (-0.5), not a subtraction (x - 0.5). */
  function statements(src) {
    const { toks, err } = tokenize(src), out = [];
    let cur = [];
    const flush = () => {
      if (cur.length && !(cur.length === 1 && cur[0].t === 'str')) { const tk = cur.map(canon); out.push({ toks: cur, tk, key: tk.join('\u0001'), line: cur[0].line }); }
      cur = [];
    };
    for (let k = 0; k < toks.length; k++) {
      const tk = toks[k];
      if (tk.t === 'com') continue;
      if (tk.t === 'nl') { if (!tk.soft) flush(); continue; }
      if (tk.t === 'op' && (tk.v === '-' || tk.v === '+') && toks[k + 1] && toks[k + 1].t === 'num' && !valueEnd(cur[cur.length - 1])) {
        const nx = toks[++k];
        cur.push({ t: 'num', v: (tk.v === '-' ? '-' : '') + nx.v, line: nx.line });
        continue;
      }
      cur.push(tk);
    }
    flush();
    return { lines: out, err };
  }

  // ------------------------------------------------------------------ comparing: Myers' diff
  // the pairs [i, j] with a[i] === b[j] that a shortest edit keeps, in order
  function match(a, b) {
    const pairs = [], tail = [];
    let s = 0, ea = a.length, eb = b.length;
    while (s < ea && s < eb && a[s] === b[s]) { pairs.push([s, s]); s++; }
    while (ea > s && eb > s && a[ea - 1] === b[eb - 1]) { ea--; eb--; tail.push([ea, eb]); }
    const N = ea - s, M = eb - s;
    if (N && M) {
      const max = N + M, off = max + 1, V = new Int32Array(2 * max + 3), trace = [];
      let done = false;
      for (let d = 0; d <= max && !done; d++) {
        trace.push(V.slice(off - d - 1, off + d + 2));
        for (let k = -d; k <= d; k += 2) {
          let x = k === -d || (k !== d && V[off + k - 1] < V[off + k + 1]) ? V[off + k + 1] : V[off + k - 1] + 1, y = x - k;
          while (x < N && y < M && a[s + x] === b[s + y]) { x++; y++; }
          V[off + k] = x;
          if (x >= N && y >= M) { done = true; break; }
        }
      }
      const mid = [];
      let x = N, y = M;
      for (let d = trace.length - 1; d >= 0 && (x > 0 || y > 0); d--) {
        const T = trace[d], at = (k) => T[k + d + 1], k = x - y;
        const pk = k === -d || (k !== d && at(k - 1) < at(k + 1)) ? k + 1 : k - 1, px = at(pk), py = px - pk;
        while (x > px && y > py) { x--; y--; mid.push([s + x, s + y]); }
        x = px; y = py;
      }
      for (let k = mid.length - 1; k >= 0; k--) pairs.push(mid[k]);
    }
    for (let k = tail.length - 1; k >= 0; k--) pairs.push(tail[k]);
    return pairs;
  }
  // the stretches between kept pairs: a[a0..a1) became b[b0..b1)
  function hunks(pairs, na, nb) {
    const out = [];
    let i = 0, j = 0;
    for (const [pi, pj] of [...pairs, [na, nb]]) {
      if (pi > i || pj > j) out.push({ a0: i, a1: pi, b0: j, b1: pj });
      i = pi + 1; j = pj + 1;
    }
    return out;
  }
  // two lines' tokens side by side: position by position when they are as long (a number changed), else by a diff
  const align = (a, b) => (a.length === b.length ? a.map((t, i) => (t === b[i] ? [i, i] : null)).filter(Boolean) : match(a, b));
  // what changed inside a line: [old token, new token, index] pairs, and tokens only one side has
  function tokenDiff(a, b) {
    const out = { pairs: [], added: [], removed: [] };
    for (const h of hunks(align(a.tk, b.tk), a.tk.length, b.tk.length)) {
      const n = Math.min(h.a1 - h.a0, h.b1 - h.b0);
      for (let t = 0; t < n; t++) out.pairs.push([a.toks[h.a0 + t], b.toks[h.b0 + t], h.a0 + t, h.b0 + t]);
      for (let t = h.a0 + n; t < h.a1; t++) out.removed.push(a.toks[t]);
      for (let t = h.b0 + n; t < h.b1; t++) out.added.push(b.toks[t]);
    }
    return out;
  }

  // ------------------------------------------------------------------ a line of Python as math
  /* A small reader for Python expressions (numbers, names, + − * / // % **, comparisons, and, or, not, a if c else b,
     calls of math, torch and the built-in functions), for two things: telling whether your line computes the same as
     the page's (the same values at many random points), and turning your reward or curve into the page's formula
     language. */
  function parseExpr(toks) {
    let i = 0;
    const is = (v) => !!toks[i] && toks[i].t !== 'str' && toks[i].v === v, take = () => toks[i++];
    const fail = (msg) => { throw new Error(msg); };
    const isOp = (list) => !!toks[i] && toks[i].t === 'op' && list.includes(toks[i].v);
    function ternary() {
      const a = orE();
      if (!is('if')) return a;
      take(); const c = orE();
      if (!is('else')) fail('An “if” here needs an “else”.');
      take();
      return { k: 'if', c, a, b: ternary() };
    }
    function orE() { let a = andE(); while (is('or')) { take(); a = { k: 'or', a, b: andE() }; } return a; }
    function andE() { let a = notE(); while (is('and')) { take(); a = { k: 'and', a, b: notE() }; } return a; }
    function notE() { if (is('not')) { take(); return { k: 'not', a: notE() }; } return cmp(); }
    const CMP = ['==', '!=', '<', '<=', '>', '>='];
    function cmp() {
      const a = arith();
      if (!isOp(CMP)) return a;
      const args = [a], ops = [];
      while (isOp(CMP)) { ops.push(take().v); args.push(arith()); }
      return { k: 'cmp', ops, args };
    }
    function arith() { let a = term(); while (isOp(['+', '-'])) { const op = take().v; a = { k: 'bin', op, a, b: term() }; } return a; }
    function term() { let a = factor(); while (isOp(['*', '/', '//', '%'])) { const op = take().v; a = { k: 'bin', op, a, b: factor() }; } return a; }
    function factor() {
      const t = toks[i];
      if (t && t.t === 'op' && (t.v === '-' || t.v === '+')) { take(); const a = factor(); return t.v === '-' ? { k: 'neg', a } : a; }
      if (t && t.t === 'num' && t.v[0] === '-' && toks[i + 1] && toks[i + 1].v === '**') { take(); return { k: 'neg', a: power({ k: 'num', v: -numOf(t.v) }) }; } // -2 ** 2 is -(2 ** 2)
      return power(primary());
    }
    function power(a) { if (is('**')) { take(); return { k: 'bin', op: '**', a, b: factor() }; } return a; }
    function primary() {
      const t = take();
      if (!t) fail('The line ends too early.');
      let a;
      if (t.t === 'num') a = { k: 'num', v: numOf(t.v), text: t.v };
      else if (t.t === 'str') a = { k: 'str', v: t.body };
      else if (t.t === 'name' && (!KW.has(t.v) || /^(True|False|None)$/.test(t.v))) a = { k: 'name', v: t.v };
      else if (t.v === '(') { a = ternary(); if (is(',')) fail('A tuple is not understood here.'); if (!is(')')) fail('A bracket is not closed.'); take(); a = { k: 'paren', a }; }
      else fail(`“${t.v}” is not understood here.`);
      for (;;) {
        if (is('.')) {
          take();
          const nm = take();
          if (!nm || nm.t !== 'name') fail('A name must follow the dot.');
          a = a.k === 'name' ? { k: 'name', v: a.v + '.' + nm.v } : { k: 'attr', obj: a, name: nm.v };
        } else if (is('(')) {
          take();
          const args = [], kw = {};
          while (!is(')')) {
            if (i >= toks.length) fail('A call is not closed.');
            if (toks[i].t === 'name' && toks[i + 1] && toks[i + 1].v === '=') { const nm = take().v; take(); kw[nm] = ternary(); } else args.push(ternary());
            if (is(',')) take(); else if (!is(')')) fail('A call is not closed.');
          }
          take();
          a = { k: 'call', f: a, args, kw };
        } else if (is('[')) {
          const from = i;
          let depth = 0;
          do { const v = toks[i].t === 'op' ? toks[i].v : ''; if (OPEN[v]) depth++; else if (/^[)\]}]$/.test(v)) depth--; i++; } while (i < toks.length && depth > 0);
          a = { k: 'opaque', v: JSON.stringify(a) + toks.slice(from, i).map(canon).join(' ') };
        } else return a;
      }
    }
    const root = ternary();
    if (i < toks.length) fail(`“${toks[i].v}” is not understood here.`);
    return root;
  }
  const CONST = { 'math.pi': Math.PI, 'math.e': Math.E, 'math.tau': 2 * Math.PI, 'math.inf': Infinity, 'torch.pi': Math.PI, True: 1, False: 0, None: 0 };
  const half = (v) => { const f = Math.floor(v), r = v - f; return r > 0.5 || (r === 0.5 && f % 2) ? f + 1 : f; }; // Python rounds halves to even
  const FN1 = { sin: Math.sin, cos: Math.cos, tan: Math.tan, exp: Math.exp, log: Math.log, ln: Math.log, sqrt: Math.sqrt, tanh: Math.tanh, floor: Math.floor, ceil: Math.ceil, fabs: Math.abs, abs: Math.abs, sign: Math.sign, round: half, sigmoid: (v) => 1 / (1 + Math.exp(-v)), relu: (v) => Math.max(0, v), float: (v) => +v, int: Math.trunc, bool: (v) => (v ? 1 : 0), log10: Math.log10, log2: Math.log2, atan: Math.atan, asin: Math.asin, acos: Math.acos };
  const LIBS = /^(math|torch|F|np|nn\.functional)\./;
  const fnName = (f) => (f && f.k === 'name' && (LIBS.test(f.v) || !f.v.includes('.')) ? f.v.replace(LIBS, '') : null);
  // a tensor's own methods, on one number: x.float(), (a == b).float().mean(), z.abs(), v.clamp(min=0)
  const SAME = new Set(['float', 'double', 'item', 'mean', 'sum', 'squeeze', 'flatten', 'detach', 'clone', 'cpu']);
  const methodOf = (f) => {
    if (f && f.k === 'attr') return { obj: f.obj, name: f.name };
    if (f && f.k === 'name' && f.v.includes('.') && !LIBS.test(f.v)) { const at = f.v.lastIndexOf('.'); return { obj: { k: 'name', v: f.v.slice(0, at) }, name: f.v.slice(at + 1) }; }
    return null;
  };
  const knownMethod = (m) => !!m && (SAME.has(m.name) || !!FN1[m.name] || m.name === 'clamp' || m.name === 'clip' || m.name === 'pow' || m.name === 'neg');
  const truthy = (v) => (typeof v === 'string' ? v.length > 0 : !!v && !Number.isNaN(v));
  function evalAst(a, env) {
    switch (a.k) {
      case 'num': case 'str': return a.v;
      case 'paren': return evalAst(a.a, env);
      case 'name': case 'opaque':
        if (a.v in CONST) return CONST[a.v];
        if (a.v in env) return env[a.v];
        throw new Error('unknown name');
      case 'neg': return -evalAst(a.a, env);
      case 'not': return truthy(evalAst(a.a, env)) ? 0 : 1;
      case 'and': { const x = evalAst(a.a, env); return truthy(x) ? evalAst(a.b, env) : x; }
      case 'or': { const x = evalAst(a.a, env); return truthy(x) ? x : evalAst(a.b, env); }
      case 'if': return truthy(evalAst(a.c, env)) ? evalAst(a.a, env) : evalAst(a.b, env);
      case 'cmp': {
        let x = evalAst(a.args[0], env);
        for (let k = 0; k < a.ops.length; k++) {
          const y = evalAst(a.args[k + 1], env), op = a.ops[k];
          const ok = op === '==' ? x === y : op === '!=' ? x !== y : typeof x === 'string' || typeof y === 'string' ? false : op === '<' ? x < y : op === '<=' ? x <= y : op === '>' ? x > y : x >= y;
          if (!ok) return 0;
          x = y;
        }
        return 1;
      }
      case 'bin': {
        const x = evalAst(a.a, env), y = evalAst(a.b, env);
        if (typeof x === 'string' || typeof y === 'string') { if (a.op === '+' && typeof x === 'string' && typeof y === 'string') return x + y; throw new Error('not a number'); }
        switch (a.op) {
          case '+': return x + y;
          case '-': return x - y;
          case '*': return x * y;
          case '/': return x / y;
          case '//': return Math.floor(x / y);
          case '%': return ((x % y) + y) % y;
          default: return Math.pow(x, y);
        }
      }
      case 'attr': { const key = JSON.stringify(a); if (key in env) return env[key]; throw new Error('unknown name'); }
      case 'call': {
        const m = methodOf(a.f);
        if (knownMethod(m)) {
          const x = evalAst(m.obj, env), args = a.args.map((q) => evalAst(q, env));
          if (SAME.has(m.name)) return x;
          if (m.name === 'neg') return -x;
          if (m.name === 'pow') return Math.pow(x, args[0]);
          if (m.name === 'clamp' || m.name === 'clip') { let v = x; const lo = a.kw.min || a.args[0], hi = a.kw.max || a.args[1]; if (lo) v = Math.max(v, evalAst(lo, env)); if (hi) v = Math.min(v, evalAst(hi, env)); return v; }
          return FN1[m.name](x);
        }
        const f = fnName(a.f), args = a.args.map((q) => evalAst(q, env));
        if (f === 'min' || f === 'minimum') return Math.min(...args);
        if (f === 'max' || f === 'maximum') return Math.max(...args);
        if (f === 'pow') return Math.pow(args[0], args[1]);
        if (f === 'clamp' || f === 'clip') { let v = args[0]; const lo = a.kw.min || a.args[1], hi = a.kw.max || a.args[2]; if (lo) v = Math.max(v, evalAst(lo, env)); if (hi) v = Math.min(v, evalAst(hi, env)); return v; }
        if (f === 'where') return truthy(args[0]) ? args[1] : args[2];
        if (f && FN1[f] && args.length === 1) return FN1[f](args[0]);
        const key = JSON.stringify(a);
        if (key in env) return env[key];
        throw new Error('unknown function');
      }
      default: throw new Error('not understood');
    }
  }
  // the names an expression reads (anything unknown counts as a name: self.s, m.dist[s0], self.dense1(x))
  function namesOf(a, out = new Set(), strs = new Set()) {
    if (!a || typeof a !== 'object') return out;
    if (a.k === 'name' || a.k === 'opaque') { if (!(a.v in CONST)) out.add(a.v); return out; }
    if (a.k === 'attr') { out.add(JSON.stringify(a)); return out; }
    if (a.k === 'str') { strs.add(a.v); return out; }
    if (a.k === 'call') {
      const m = methodOf(a.f);
      if (knownMethod(m)) { namesOf(m.obj, out, strs); a.args.forEach((q) => namesOf(q, out, strs)); Object.values(a.kw).forEach((q) => namesOf(q, out, strs)); return out; }
      const f = fnName(a.f);
      if (!(f && (FN1[f] || /^(min|max|minimum|maximum|pow|clamp|clip|where)$/.test(f)))) { out.add(JSON.stringify(a)); return out; }
      a.args.forEach((q) => namesOf(q, out, strs)); Object.values(a.kw).forEach((q) => namesOf(q, out, strs));
      return out;
    }
    for (const k of ['a', 'b', 'c']) if (a[k]) namesOf(a[k], out, strs);
    if (a.args) a.args.forEach((q) => namesOf(q, out, strs));
    return out;
  }
  const numsOf = (a, out = []) => { if (a && typeof a === 'object') { if (a.k === 'num') out.push(a.v); else for (const k of ['a', 'b', 'c', 'f', 'obj']) numsOf(a[k], out); (a.args || []).forEach((q) => numsOf(q, out)); Object.values(a.kw || {}).forEach((q) => numsOf(q, out)); } return out; };
  // a statement as target, operation and expression: x = e, r += e (r -= e is r += -(e)), return e
  function splitStatement(line) {
    const t = line.toks;
    if (t[0] && t[0].v === 'return') return { target: 'return', op: '=', expr: t.slice(1) };
    let depth = 0;
    for (let k = 0; k < t.length; k++) {
      const v = t[k].t === 'op' ? t[k].v : '';
      if (OPEN[v]) depth++; else if (/^[)\]}]$/.test(v)) depth--;
      else if (!depth && /^([-+*/]?=)$/.test(v)) return { target: t.slice(0, k).map(canon).join(' '), op: v, expr: t.slice(k + 1) };
    }
    return null;
  }
  // a fixed sequence of pseudo-random numbers, so the same lines always get the same verdict
  function rand(seed) { let s = seed >>> 0; return () => { s = (s * 1664525 + 1013904223) >>> 0; return s / 4294967296; }; }
  /* Whether two statements compute the same thing: the same target, and expressions equal at 24 random points (names
     at 0, 1, other numbers or the strings they are compared with). */
  function equivalent(x, y) {
    const a = splitStatement(x), b = splitStatement(y);
    if (!a || !b || a.target !== b.target) return false;
    let ea, eb;
    try { ea = parseExpr(a.expr); eb = parseExpr(b.expr); } catch (e) { return false; }
    const sign = (s, e) => (s.op === '-=' ? { k: 'neg', a: e } : e);
    if ((a.op === '=') !== (b.op === '=') || (a.op !== '=' && (a.op === '*=' || a.op === '/=' || b.op === '*=' || b.op === '/=') && a.op !== b.op)) return false;
    ea = sign(a, ea); eb = sign(b, eb);
    const strs = new Set(), names = [...namesOf(ea, new Set(), strs), ...namesOf(eb, new Set(), strs)].filter((v, i, l) => l.indexOf(v) === i);
    if (names.length > 16) return false;
    // every name takes every value of the pool in turn (so c == "X" is tried), then random numbers
    const pool = [0, 1, -1, 2, 0.5, ...strs, 'other'], r = rand(7), rounds = 2 * pool.length;
    let evidence = 0;
    for (let trial = 0; trial < rounds + 16; trial++) {
      const env = {};
      names.forEach((nm, k) => { env[nm] = trial < rounds ? pool[(trial + k * Math.floor(r() * pool.length)) % pool.length] : -3 + 6 * r(); });
      let p, q;
      try { p = evalAst(ea, env); } catch (e) { p = 'ERR'; }
      try { q = evalAst(eb, env); } catch (e) { q = 'ERR'; }
      if (p === 'ERR' || q === 'ERR') { if (p !== q) return false; continue; }
      if (typeof p === 'string' || typeof q === 'string') { if (p !== q) return false; evidence++; continue; }
      if (Number.isNaN(p) && Number.isNaN(q)) continue;
      if (!(p === q || Math.abs(p - q) <= 1e-9 * Math.max(1, Math.abs(p), Math.abs(q)))) return false;
      if (Number.isFinite(p)) evidence++;
    }
    return evidence >= 3;
  }
  // the page's formula language from a Python expression (sin(5x) + x^2 for torch.sin(5 * x) + x ** 2)
  function toFormula(a, vars) {
    const V = new Set(vars), fail = (msg) => { throw new Error(msg); };
    const ONE = new Set(['sin', 'cos', 'tan', 'exp', 'log', 'sqrt', 'abs', 'fabs', 'tanh', 'sign', 'floor', 'round', 'relu', 'sigmoid']);
    // { s, p }: the text and how tightly it binds (1 + −, 2 × ÷, 3 a sign, 4 a power, 5 a name, number or call), brackets only where needed
    const wrap = (x, min) => (x.p < min ? `(${x.s})` : x.s);
    const go = (a) => {
      switch (a.k) {
        case 'num': { const s = a.text ? a.text.replace(/_/g, '').replace(/^[-+]/, '') : String(+Math.abs(a.v).toPrecision(12)).replace('e+', 'e'); return a.v < 0 || /^-/.test(a.text || '') ? { s: '-' + s, p: 3 } : { s, p: 5 }; }
        case 'paren': return go(a.a);
        case 'name':
          if (V.has(a.v)) return { s: a.v, p: 5 };
          if (a.v === 'math.pi' || a.v === 'torch.pi') return { s: 'pi', p: 5 };
          if (a.v === 'math.e') return { s: 'e', p: 5 };
          return fail(`“${a.v}” is not one of the names it can use here: ${vars.join(', ')}.`);
        case 'neg': return { s: '-' + wrap(go(a.a), 3), p: 3 };
        case 'bin': {
          const x = go(a.a), y = go(a.b);
          if (a.op === '**') return { s: `${wrap(x, 5)}^${wrap(y, 3)}`, p: 4 };
          if (a.op === '//') return { s: `floor(${wrap(x, 2)} / ${wrap(y, 4)})`, p: 5 };
          if (a.op === '%') return fail('The page’s formulas have no remainder (%).');
          if (a.op === '+' || a.op === '-') return { s: `${x.s} ${a.op} ${wrap(y, 2)}`, p: 1 };
          return { s: `${wrap(x, 2)} ${a.op} ${wrap(y, a.op === '*' ? 3 : 4)}`, p: 2 };
        }
        case 'call': {
          const f = fnName(a.f), n = a.args.length, kws = Object.keys(a.kw), arg = (k) => go(a.args[k]).s;
          if (f === 'float' && n === 1 && !kws.length) return go(a.args[0]);
          if (f && ONE.has(f) && n === 1 && !kws.length) return { s: `${f === 'fabs' ? 'abs' : f}(${arg(0)})`, p: 5 };
          if (f && /^(min|max|minimum|maximum|pow)$/.test(f) && n === 2 && !kws.length) return { s: `${f.slice(0, 3)}(${arg(0)}, ${arg(1)})`, p: 5 };
          if (f === 'clamp' && n === 1 && kws.length && kws.every((k) => k === 'min' || k === 'max')) { let s = arg(0); if (a.kw.min) s = `max(${s}, ${go(a.kw.min).s})`; if (a.kw.max) s = `min(${s}, ${go(a.kw.max).s})`; return { s, p: 5 }; }
          if ((f === 'clamp' || f === 'clip') && n === 3 && !kws.length) return { s: `min(max(${arg(0)}, ${arg(1)}), ${arg(2)})`, p: 5 };
          if (f === 'where' && n === 3 && !kws.length) return { s: `if(${arg(0)}, ${arg(1)}, ${arg(2)})`, p: 5 };
          return fail(`The page’s formulas have no ${f ? `${f}()` : 'such call'}: they use + − × ÷ ^, sin, cos, tan, exp, log, sqrt, abs, tanh, sign, floor, round, relu, sigmoid, min, max and pow.`);
        }
        case 'attr': case 'opaque': return fail(`It can only use these names here: ${vars.join(', ')}.`);
        // a if c else b, comparisons (a chain such as 1 < d < 3 as two joined by and), and, or, not: as the page writes them
        case 'if': return { s: `if(${go(a.c).s}, ${go(a.a).s}, ${go(a.b).s})`, p: 5 };
        case 'cmp': {
          const parts = a.ops.map((op, k) => `${wrap(go(a.args[k]), 1)} ${op} ${wrap(go(a.args[k + 1]), 1)}`);
          return parts.length === 1 ? { s: parts[0], p: 0.5 } : { s: parts.map((q) => `(${q})`).join(' and '), p: 0.3 };
        }
        case 'not': return { s: `not ${wrap(go(a.a), 0.5)}`, p: 0.4 };
        case 'and': return { s: `${wrap(go(a.a), 0.3)} and ${wrap(go(a.b), 0.4)}`, p: 0.3 };
        case 'or': return { s: `${wrap(go(a.a), 0.2)} or ${wrap(go(a.b), 0.3)}`, p: 0.2 };
        default: return fail('The page’s formulas have no strings or lists.');
      }
    };
    return go(a).s;
  }

  // ------------------------------------------------------------------ the settings a script comes from
  /* Each setting the script can show is a knob: { id, kind: 'num' | 'enum' | 'bool' | 'formula' | 'text', where,
     what, get(S), set(S, v), opts (choices), labels, range, list (allowed numbers), int }. A setup's knobs are
     found from the setup itself, so settings added to the page later are knobs too (blocks, worlds and rewards
     come with their defaults). The ranges are those the page's controls offer. */
  const OPT_LABEL = () => Object.fromEntries(Object.entries(NA.bx.OPTIMIZERS).map(([k, o]) => [k, o.label]));
  const TRAIN = {
    lr: { what: 'learning rate', range: [1e-6, 10] }, batch: { what: 'batch size', range: [1, 1024], int: true }, steps: { what: 'steps', range: [0, 2e5], int: true },
    opt: { what: 'optimizer', opts: () => Object.keys(NA.bx.OPTIMIZERS), labels: OPT_LABEL }, sched: { what: 'learning-rate schedule', opts: ['const', 'cosine', 'step', 'warmup'], labels: () => ({ const: 'constant', cosine: 'cosine', step: 'step', warmup: 'warm-up + cosine' }) },
    seed: { what: 'random seed', range: [0, 999999], int: true }, clip: { what: 'gradient clipping', list: [0, 0.5, 1, 5] }, wd: { what: 'weight decay', list: [0, 0.0001, 0.001, 0.01, 0.05] },
    init: { what: 'starting weights', opts: ['auto', 'he', 'xavier', 'small'], labels: () => ({ auto: 'automatic', he: 'He', xavier: 'Xavier', small: 'tiny' }) },
    grow: { what: 'grow and prune', def: false }, growEvery: { what: 'steps between reshapes', range: [10, 5000], int: true, def: 150 },
    growPenalty: { what: 'size penalty', opts: ['off', 'light', 'strong'], def: 'light', labels: () => PS_LABELS.penalty }, growNew: { what: 'how new layers start', opts: ['identity', 'random'], def: 'identity', labels: () => PS_LABELS.newLayer },
    // (grow and prune's newer settings; "−1 layer" is on by default with whole layers: its default follows the others)
    growWidth: { what: 'width of new layers', opts: ['one', 'some', 'whole'], def: 'one', labels: () => PS_LABELS.layerSize }, growSize: { what: 'networks side by side', range: [2, 12], int: true, def: 6 },
    growStart: { what: 'starting network', opts: ['built', 'small'], def: 'built', labels: () => ({ built: 'the network you built', small: 'no hidden layer' }) },
    growAddNeuron: { what: 'adding neurons', def: true }, growCutNeuron: { what: 'pruning neurons', def: true }, growAddLayer: { what: 'adding layers', def: true },
    growCutLayer: { what: 'removing whole layers', def: (T) => NA.bevo.growOptions(T, true).ops['layer-'] }, growAct: { what: 'changing activations', def: false },
    growLean: { what: 'leaning', opts: ['grow', 'even', 'prune'], def: 'even', labels: () => PS_LABELS.growLean }, growWhere: { what: 'place of new layers', opts: ['any', 'first', 'last'], def: 'any', labels: () => PS_LABELS.growWhere },
    growNeuron: { what: 'start of new neurons', opts: ['silent', 'random'], def: 'silent', labels: () => PS_LABELS.growNeuron }, growPrune: { what: 'neuron pruning takes', opts: ['weak', 'random'], def: 'weak', labels: () => PS_LABELS.growPrune },
    growMaxLayers: { what: 'most hidden layers', range: [1, 12], int: true, def: 6 }, growMaxUnits: { what: 'most neurons in a layer', range: [1, 128], int: true, def: 48 },
  };
  const LOSS = { id: { what: 'loss' }, smooth: { what: 'label smoothing', range: [0, 0.3] }, delta: { what: 'Huber δ', range: [0.05, 1] } };
  const BLOCK_WORDS = { units: 'units', act: 'activation', p: 'dropout rate', filters: 'filters', k: 'filter size', stride: 'stride', size: 'patch size', heads: 'heads', mlp: 'MLP size', causal: 'causal mask', residual: 'residual',
    dim: 'token width', features: 'frequencies', scale: 'frequency spread', sigma: 'noise σ', cell: 'cell', dilation: 'dilation', mode: 'kind', merge: 'how the branches meet', gate: 'gate', lambda: 'λ', depth: 'depth',
    minLeaf: 'fewest examples in a leaf', trees: 'trees', rounds: 'rounds', eta: 'η', gamma: 'γ', subsample: 'rows per tree', colsample: 'inputs per tree', weights: 'votes', auto: 'automatic sizes', type: 'kind of block' };
  const BLOCK_RANGE = {
    'dense.units': [1, 256], 'gated.units': [1, 128], 'rnn.units': [1, 128], 'dropout.p': [0, 0.7], 'conv.filters': [1, 32], 'conv.k': [[1, 3, 5]], 'conv.stride': [[1, 2]], 'pool.size': [[2, 3]], 'avgpool.size': [[2, 3]],
    'attention.heads': [[1, 2, 4, 8]], 'transformer.heads': [[1, 2, 4, 8]], 'transformer.mlp': [[1, 2, 4]], 'noise.sigma': [0, 1], 'fourier.features': [1, 64], 'fourier.scale': [0.25, 10], 'conv1d.filters': [1, 64],
    'conv1d.k': [[2, 3, 5, 7]], 'conv1d.dilation': [[1, 2, 4, 8]], 'patches.size': [[1, 2, 3, 7]], 'patches.dim': [4, 64], 'knn.k': [1, 25], 'tree.depth': [1, 16], 'tree.minLeaf': [1, 30], 'forest.trees': [1, 200],
    'forest.depth': [1, 16], 'forest.minLeaf': [1, 30], 'boost.rounds': [5, 500], 'boost.depth': [1, 8], 'boost.eta': [[0.01, 0.03, 0.1, 0.3, 1]], 'boost.lambda': [[0, 0.1, 1, 10]], 'boost.gamma': [[0, 0.1, 1, 5]],
    'boost.subsample': [[0.5, 0.8, 1]], 'boost.colsample': [[0.5, 0.8, 1]], 'linear.lambda': [[0, 0.01, 0.1, 1, 10]],
  };
  const BLOCK_OPTS = { act: () => NA.engine.ACTIVATION_ORDER, gate: () => ['sigmoid', 'swish', 'gelu'], cell: () => ['rnn', 'lstm', 'gru'], merge: () => Object.keys(NA.bx.MERGES), weights: () => ['uniform', 'distance'],
    'position.mode': () => ['learned', 'sine'], 'seqpool.mode': () => ['last', 'max', 'mean'], 'forest.features': () => ['sqrt', 'third', 'all'] };
  // kinds of block that take the same settings, so one can become the other (a max pool an average pool)
  const SWAPS = [['pool', 'avgpool'], ['gap', 'gmp']];
  const FLOAT_KEYS = new Set(['p', 'sigma', 'scale', 'width', 'lambda', 'eta', 'gamma', 'subsample', 'colsample', 'noise', 'lr', 'clip', 'wd', 'smooth', 'delta', 'slip', 'pRight', 'pOther', 'force', 'poleLen', 'gravity', 'esSigma', 'esLr', 'explore', 'growRate']);
  const PS_WORDS = { noise: 'noise', n: 'examples', len: 'sequence length', embed: 'embedding size', window: 'window', method: 'way of learning', view: 'what it sees', closer: 'closer bonus', random: 'random mazes', gamma: 'discount γ',
    explore: 'exploration steps', pop: 'population', sigma: 'mutation size', slip: 'slippery floor', maxSteps: 'longest episode', force: 'push', poleLen: 'pole length', gravity: 'gravity', drift: 'ball drift', pRight: 'odds the matching lever pays',
    pOther: 'odds another lever pays', levers: 'levers', rFormula: 'reward formula', select: 'way of choosing parents', cross: 'way of mixing parents', mutate: 'kind of mutation', grow: 'grow and prune', penalty: 'size penalty',
    newLayer: 'how new layers start', layerSize: 'width of new layers', growStart: 'starting network', growAddNeuron: 'adding neurons', growCutNeuron: 'pruning neurons', growAddLayer: 'adding layers', growCutLayer: 'removing whole layers',
    growAct: 'changing activations', growLean: 'leaning', growRate: 'share of children that change shape', growWhere: 'place of new layers', growNeuron: 'start of new neurons', growPrune: 'neuron pruning takes',
    growMaxLayers: 'most hidden layers', growMaxUnits: 'most neurons in a layer', growProtect: 'protection of new shapes', prefAsk: 'who answers', prefEvery: 'steps between questions', esSigma: 'nudge size', esLr: 'step size', track: 'track', map: 'map', drones: 'drones', speed: 'speed', wind: 'wind', pad: 'pad', fuel: 'fuel', maze: 'maze', formula: 'formula', text: 'text', target: 'column it predicts' };
  const PS_RANGE = { noise: [0, 2], window: [1, 64], len: [2, 8], embed: [4, 128], gamma: [0, 1], explore: [0, 1e6], pop: [4, 500], sigma: [0.001, 5], slip: [0, 0.9], esSigma: [0.001, 5], esLr: [0.0001, 1], pRight: [0, 1], pOther: [0, 1], maxSteps: [0, 1e5],
    growRate: [0.01, 1], growMaxLayers: [1, 12], growMaxUnits: [1, 128] };
  const N_RANGE = { plane: [100, 1500], curve: [20, 1000], digits: [300, 4000], surface: [200, 3000], series: [400, 4000], seqclass: [500, 5000], sequence: [300, 6000] };
  const INT_KEYS = new Set(['n', 'len', 'window', 'embed', 'pop', 'maxSteps', 'growMaxLayers', 'growMaxUnits', 'levers', 'drones', 'chasers', 'sheep', 'length', 'target', 'units', 'filters', 'k', 'stride', 'size', 'heads', 'mlp', 'features', 'dim', 'dilation', 'depth', 'minLeaf', 'trees', 'rounds', 'state', 'pieces', 'experts', 'top', 'shift']);
  const PS_OPTS = { method: () => NA.brl.ORDER, select: () => ['tournament', 'truncation', 'rank'], cross: () => ['uniform', 'neuron', 'none'], mutate: () => ['plain', 'safe'], penalty: () => Object.keys((NA.bevo && NA.bevo.PENALTY) || { off: 0, light: 0, strong: 0 }), newLayer: () => ['identity', 'random'],
    layerSize: () => ['one', 'some', 'whole'], growStart: () => ['built', 'small'], growLean: () => ['grow', 'even', 'prune'], growWhere: () => ['any', 'first', 'last'], growNeuron: () => ['silent', 'random'], growPrune: () => ['weak', 'random'], prefAsk: () => ['you', 'world'] };
  // grow and prune's choices in words (for the note that says what was applied)
  const PS_LABELS = { growStart: { built: 'your network', small: 'no hidden layer' }, growLean: { grow: 'to grow', even: 'even', prune: 'to prune' }, growWhere: { any: 'anywhere', first: 'after the input', last: 'before the output' },
    growNeuron: { silent: 'silent', random: 'at random' }, growPrune: { weak: 'the weakest', random: 'one at random' }, layerSize: { one: 'one neuron', some: 'a few, at random', whole: 'a whole layer' },
    penalty: { off: 'none', light: 'light', strong: 'strong' }, newLayer: { identity: 'passing values on', random: 'at random' } };
  const callOr = (v) => (typeof v === 'function' ? v() : v);
  const partName = (type) => { const P = NA.builderLab && NA.builderLab.PARTS && NA.builderLab.PARTS.find((q) => q.type === type); return P ? P.name : (NA.bx.BLOCKS[type] || {}).label || (NA.cml && NA.cml.KINDS && NA.cml.KINDS[type] ? NA.cml.KINDS[type].label : type); };
  const blockWhat = (b, key) => (key === 'units' && b.type === 'dense' ? 'neurons' : key === 'k' && b.type === 'knn' ? 'neighbours' : key === 'k' && b.type === 'conv1d' ? 'window' : key === 'size' && b.type === 'patches' ? 'patch size'
    : key === 'features' && b.type === 'forest' ? 'inputs tried at each question' : BLOCK_WORDS[key] || key);
  const blockDefaults = (b) => Object.assign({}, (NA.bx.BLOCKS[b.type] || {}).defaults, NA.cml && NA.cml.isClassic(b.type) ? NA.cml.withDefaults(b) : null, ...Object.keys(b).filter((k) => b[k] !== undefined && b[k] !== null).map((k) => ({ [k]: b[k] })));
  // a path into a setup, read and written
  const getAt = (S, path) => path.reduce((o, k) => (o == null ? undefined : o[k]), S);
  function setAt(S, path, v) { let o = S; for (let k = 0; k < path.length - 1; k++) o = o[path[k]]; o[path[path.length - 1]] = v; }
  function knobsOf(S, d) {
    const out = [], p = BT.byId[S.problem], fam = p.family, ctl = fam === 'control';
    const num = (id, where, what, path, o) => out.push(Object.assign({ id, kind: 'num', where, what, get: (s) => getAt(s, path), set: (s, v) => setAt(s, path, v) }, o));
    // training
    for (const key of new Set([...Object.keys(TRAIN), ...Object.keys(S.train)])) {
      const T = TRAIN[key] || {}, dv = (s) => (typeof T.def === 'function' ? T.def(s.train) : T.def), v = S.train[key] !== undefined ? S.train[key] : dv(S);
      const k = { id: 'train.' + key, where: '', what: T.what || key, get: (s) => (s.train[key] !== undefined ? s.train[key] : dv(s)), set: (s, x) => { s.train[key] = x; } };
      if (key === 'steps' && ctl && d && !d.error && NA.brl.isEvo(d.method)) k.what = 'generations';
      if (typeof v === 'number') out.push(Object.assign(k, { kind: 'num', range: T.range || [-1e4, 1e4], list: T.list, int: !!T.int }));
      else if (typeof v === 'boolean') out.push(Object.assign(k, { kind: 'bool' }));
      else if (typeof v === 'string') out.push(Object.assign(k, { kind: 'enum', opts: T.opts ? callOr(T.opts) : [v], labels: T.labels ? T.labels() : null }));
    }
    // the loss (learning from rewards: the method sets it)
    if (!ctl) {
      const L = S.loss;
      out.push({ id: 'loss.id', kind: 'enum', where: '', what: 'loss', opts: d && !d.error ? BT.lossesFor(p, d) : [L.id], labels: Object.fromEntries(Object.entries(NA.bx.LOSSES).map(([k, o]) => [k, o.label])), get: (s) => s.loss.id, set: (s, v) => { s.loss.id = v; } });
      for (const key of ['smooth', 'delta']) num('loss.' + key, '', LOSS[key].what, ['loss', key], { range: LOSS[key].range, get: (s) => s.loss[key] || (key === 'delta' ? 0.2 : 0) });
    }
    // the problem: data, the world, its rewards, the way it learns
    const ps = S.ps[S.problem] || BT.defaults(p), keys = new Set([...Object.keys(BT.defaults(p)), ...Object.keys(ps)]), rw = ctl && NA.rlEnvs.REWARDS[p.task] ? NA.rlEnvs.REWARDS[p.task] : [];
    rw.forEach((r) => keys.add(r.key));
    if (ctl && NA.rlEnvs.REWARD_VARS[p.task]) keys.add('rFormula');
    // grow and prune's "−1 layer": on by default only with whole layers, so a knob with that default
    if (ctl && NA.bevo) out.push({ id: 'ps.growCutLayer', kind: 'bool', where: '', what: PS_WORDS.growCutLayer, get: (s) => NA.bevo.growOptions(s.ps[s.problem] || BT.defaults(p)).ops['layer-'], set: (s, v) => { (s.ps[s.problem] || (s.ps[s.problem] = BT.defaults(p))).growCutLayer = v; } });
    const own = (s) => s.ps[s.problem] || (s.ps[s.problem] = BT.defaults(p));
    for (const key of keys) {
      const r = rw.find((q) => q.key === key), base = { id: 'ps.' + key, where: '', what: r ? `reward for ${r.label.toLowerCase().replace(/ \(.*\)$/, '')}` : PS_WORDS[key] || key };
      const get = (s) => { const q = s.ps[s.problem] || BT.defaults(p); return q[key] !== undefined ? q[key] : r ? r.def : undefined; }, set = (s, v) => { own(s)[key] = v; };
      const v = get(S);
      if (key === 'hide' || key === 'kind' || key === 'growCutLayer' || (key === 'target' && fam === 'table')) continue;
      if (key === 'rFormula' || (key === 'formula' && p.custom && fam === 'curve')) {
        const vars = key === 'rFormula' ? (NA.rlEnvs.REWARD_VARS[p.task] || []).map((q) => q[0]) : ['x'];
        if (vars.length) out.push(Object.assign(base, { kind: 'formula', vars, scalar: key === 'rFormula', get: (s) => get(s) || '', set }));
      } else if (key === 'text') out.push(Object.assign(base, { kind: 'text', get: (s) => get(s) || '', set }));
      else if (typeof v === 'number') out.push(Object.assign(base, { kind: 'num', get, set, range: r ? [-1e4, 1e4] : key === 'n' && N_RANGE[fam] ? N_RANGE[fam] : PS_RANGE[key] || [-1e4, 1e4], int: INT_KEYS.has(key) || (!FLOAT_KEYS.has(key) && !r && Number.isInteger(v) && !PS_RANGE[key]) }));
      else if (typeof v === 'boolean') out.push(Object.assign(base, { kind: 'bool', get, set }));
      else if (typeof v === 'string') {
        const opts = PS_OPTS[key] ? PS_OPTS[key]() : key === 'view' && BT.VIEWS[p.task] ? BT.VIEWS[p.task].map((q) => q.value) : key === 'maze' && p.task === 'maze' ? NA.rl.MAZE_SET.map((m) => m.id) : [v];
        out.push(Object.assign(base, { kind: 'enum', opts, get, set, labels: key === 'method' ? Object.fromEntries(Object.entries(NA.brl.METHODS).map(([k, m]) => [k, m.label])) : key === 'view' && BT.VIEWS[p.task] ? Object.fromEntries(BT.VIEWS[p.task].map((q) => [q.value, q.label])) : PS_LABELS[key] || null }));
      }
    }
    // points you placed: read back by their value only (there can be hundreds)
    if (p.id === 'draw') (S.custom.points || []).forEach((q, j) => ['x', 'y'].forEach((c) => out.push({ id: `pt.${j}.${c}`, kind: 'num', where: `Point ${j + 1}`, what: c, range: [-1, 1], noProbe: true, get: (s) => s.custom.points[j][c],
      set: (s, v) => { s.custom = Object.assign({}, s.custom, { points: s.custom.points.map((t) => Object.assign({}, t)) }); s.custom.points[j][c] = v; } })));
    // the blocks: a part of your own once (a change changes every copy), and the blocks in each branch of a parallel block
    const runs = BT.groupsOf ? BT.groupsOf(S.blocks) : [], seenPart = new Set();
    const blockKnobs = (b, path, where, inPart) => {
      const c = blockDefaults(b);
      for (const key of Object.keys(c)) {
        if (/^(id|grp|part|branches|acts)$/.test(key)) continue;
        const v = c[key], R = BLOCK_RANGE[`${b.type}.${key}`], list = R && Array.isArray(R[0]) ? R[0] : null, k = { id: `${path.join('.')}.${key}`, where, what: blockWhat(b, key),
          get: (s) => { const q = getAt(s, path); return q[key] !== undefined && q[key] !== null ? q[key] : blockDefaults(q)[key]; }, set: (s, x) => { inPart ? inPart(s, (q) => { q[key] = x; }) : (getAt(s, path)[key] = x); } };
        if (typeof v === 'number') out.push(Object.assign(k, { kind: 'num', range: R && !list ? R : list ? [Math.min(...list), Math.max(...list)] : [-512, 512], list, int: INT_KEYS.has(key) || (!FLOAT_KEYS.has(key) && Number.isInteger(v)) }));
        else if (typeof v === 'boolean') out.push(Object.assign(k, { kind: 'bool' }));
        else if (typeof v === 'string') { const o = BLOCK_OPTS[`${b.type}.${key}`] || BLOCK_OPTS[key]; out.push(Object.assign(k, { kind: 'enum', opts: o ? o() : [v], labels: key === 'act' || key === 'gate' ? Object.fromEntries(Object.entries(NA.engine.ACTIVATIONS).map(([a, A]) => [a, A.label])) : null })); }
      }
      // each neuron's own activation, read back by value
      if (Array.isArray(b.acts)) b.acts.forEach((a, j) => out.push({ id: `${path.join('.')}.acts.${j}`, kind: 'enum', noProbe: true, where, what: `neuron ${j + 1}’s activation`, opts: NA.engine.ACTIVATION_ORDER,
        get: (s) => getAt(s, path).acts[j], set: (s, x) => { const f = (q) => { q.acts = q.acts.slice(); q.acts[j] = x; if (q.acts.every((t) => t === q.act)) delete q.acts; }; inPart ? inPart(s, f) : f(getAt(s, path)); } }));
      const swap = SWAPS.find((g) => g.includes(b.type) && g.every((t) => NA.bx.BLOCKS[t]));
      if (swap && !inPart) out.push({ id: `${path.join('.')}.type`, kind: 'enum', where, what: 'kind of block', opts: swap, labels: Object.fromEntries(swap.map((t) => [t, partName(t)])), get: (s) => getAt(s, path).type, set: (s, x) => { getAt(s, path).type = x; } });
    };
    S.blocks.forEach((b, i) => {
      const run = b.grp ? runs.find((g) => g.grp === b.grp) : null;
      if (run && (seenPart.has(run.part) && run.from === i ? false : seenPart.has(run.part))) return;
      if (run && run.from === i) seenPart.add(run.part);
      const def = run && (S.parts || []).find((q) => q.id === run.part), kk = run ? i - run.from : -1;
      // a block of a part: the definition and every copy
      const inPart = run ? (s, f) => { for (const g of BT.groupsOf(s.blocks)) if (g.part === run.part) f(s.blocks[g.from + kk]); const df = (s.parts || []).find((q) => q.id === run.part); if (df && df.blocks[kk]) f(df.blocks[kk]); } : null;
      const where = run ? `Block ${kk + 1} of your part ${def ? def.name : ''}`.trim() : `Block ${i + 1} (${partName(b.type)})`;
      blockKnobs(b, ['blocks', i], where, inPart);
      if (Array.isArray(b.branches) && !run) b.branches.forEach((list, br) => list.forEach((q, k) => blockKnobs(q, ['blocks', i, 'branches', br, k], `Branch ${'ABCD'[br]}, block ${k + 1} of block ${i + 1} (${partName(q.type)})`, null)));
    });
    return out;
  }
  const show = (k, v) => (k.kind === 'enum' ? (k.labels && k.labels[v]) || v : k.kind === 'bool' ? (v ? 'on' : 'off') : k.kind === 'formula' ? v || 'none' : k.kind === 'text' ? 'edited' : k.id === 'train.steps' && !v ? 'until paused' : sig(v));
  const cap = (s) => s.charAt(0).toUpperCase() + s.slice(1);
  const labelOf = (k) => (k.where ? `${k.what} of ${k.where.charAt(0).toLowerCase()}${k.where.slice(1)}` : `the ${k.what}`);
  // whether a value is one the page's controls offer for that setting ('' when it is)
  function outOfRange(k, v) {
    if (k.kind === 'enum') return k.opts.includes(v) ? '' : `${cap(labelOf(k))} can be ${listWords(k.opts.map((o) => (k.labels && k.labels[o]) || o))}.`;
    if (k.kind !== 'num') return '';
    if (typeof v !== 'number' || !Number.isFinite(v)) return 'That is not a number the page can use.';
    if (k.int && !Number.isInteger(v)) return `${cap(labelOf(k))} is a whole number, not ${sig(v)}.`;
    if (k.list && !k.list.some((x) => same(x, v))) return `${cap(labelOf(k))} can be ${listWords(k.list.map(sig))} on the page, not ${sig(v)}.`;
    if (k.range && (v < k.range[0] - 1e-12 || v > k.range[1] + 1e-12)) return `${cap(labelOf(k))} can be from ${sig(k.range[0])} to ${sig(k.range[1])} on the page, not ${sig(v)}.`;
    return '';
  }

  // ------------------------------------------------------------------ the search
  const clone = (S) => Object.assign({}, JSON.parse(JSON.stringify(Object.assign({}, S, { custom: null }))), { custom: S.custom }); // (a table is shared, not copied)
  const dataKey = (S) => { const p = BT.byId[S.problem]; return JSON.stringify([S.problem, S.ps[S.problem] || BT.defaults(p), S.train.seed, p.id === 'draw' ? S.custom.points : p.family === 'table' && S.custom.table ? S.custom.table.stamp || S.custom.table.name : 0]); };
  function makeData(ctx, S) {
    const key = dataKey(S);
    if (ctx.cache.has(key)) return ctx.cache.get(key);
    const p = BT.byId[S.problem];
    let d;
    try { d = BT.make(p, S.ps[S.problem] || BT.defaults(p), S.train.seed, S.custom || {}); } catch (e) { d = { error: e.message || String(e) }; }
    if (ctx.cache.size > 10) ctx.cache.delete(ctx.cache.keys().next().value);
    ctx.cache.set(key, d);
    return d;
  }
  function gen(ctx, S) { ctx.gens++; try { return NA.builderPy.code(S, makeData(ctx, S)); } catch (e) { return null; } }
  const isClassicSetup = (S) => S.blocks.some((b) => NA.cml && NA.cml.isClassic(b.type));
  function checkOf(S, d) {
    if (d.error) return { ok: false, why: d.error };
    try {
      if (isClassicSetup(S)) { if (d.family === 'control') return { ok: false, why: 'A classic model learns from examples with answers, and here there are only rewards.' }; const c = NA.cml.check(S.blocks, d); return { ok: c.ok, why: c.ok ? '' : ((c.shapes.find((s) => s && s.problem) || {}).problem || {}).reason || 'it does not fit' }; }
      const c = NA.bx.check(S.blocks, d.input, d.output);
      return { ok: c.ok, why: c.ok ? '' : ((c.problem || (c.shapes.find((s) => s && s.problem) || {}).problem) || {}).reason || 'it does not fit' };
    } catch (e) { return { ok: false, why: e.message || String(e) }; }
  }
  // a changed setup the page can train: in the controls' ranges, with data, and a model that fits together
  function validate(ctx, S1, m) {
    if (m && m.knob) {
      const r = outOfRange(m.knob, m.v);
      if (r) return { ok: false, why: r, whole: true };
      if (m.knob.kind === 'formula') {
        if (m.v.length > 200) return { ok: false, why: `${cap(labelOf(m.knob))} can be up to 200 characters long.`, whole: true };
        const c = m.knob.scalar ? BT.compile(m.v, m.knob.vars, { scalar: true }) : BT.checkFormula(m.v);
        if (c.error) return { ok: false, why: `${cap(labelOf(m.knob))}: ${c.error}`, whole: true };
      }
    }
    const d = makeData(ctx, S1);
    if (d.error && !ctx.d0.error) return { ok: false, why: d.error };
    if (ctx.ok0) { const c = checkOf(S1, d); if (!c.ok) return { ok: false, why: `the model would not fit together: ${c.why}` }; }
    return { ok: true };
  }
  /* How far a candidate script is from your edit: for every line you changed, the tokens you added or changed that
     it lacks, plus those you took out that it still has (a line that computes the same counts as reproduced); and
     every line you deleted that it still has. Lines you did not touch may differ: they follow from the change. */
  function missed(ctx, n, j) {
    const key = n.key + '\u0002' + j;
    let r = ctx.miss.get(key);
    if (r !== undefined) return r;
    const e = ctx.E[j], u = ctx.mine[j], ne = align(n.tk, e.tk), eHit = new Uint8Array(e.tk.length), nHit = new Uint8Array(n.tk.length);
    for (const [a, b] of ne) { nHit[a] = 1; eHit[b] = 1; }
    r = 0;
    for (const b of u.plus) if (!eHit[b]) r++;
    if (u.o && u.minus.length) { const on = new Map(align(u.o.tk, n.tk)); for (const a of u.minus) { const b = on.get(a); if (b !== undefined && !nHit[b]) r++; } }
    /* A name or text you kept (Adam, relu) that the candidate writes differently is another reading of your line, not
       something that follows from it: no credit for the line. Numbers may follow once the line has all your changes
       (padding from a filter size), and so may a block's number in a name (dense3 is dense2 once a block before it is
       gone); a line only part done gets credit only if nothing you kept moved. */
    const stem = (v) => v.replace(/\d+$/, ''), nNames = new Set(n.toks.filter((t) => t.t === 'name').map((t) => stem(t.v)));
    let moved = 0, names = 0;
    for (let b = 0; b < e.tk.length; b++) {
      if (eHit[b] || u.plusSet.has(b)) continue;
      moved++;
      const t = e.toks[b], q = n.toks.length === e.toks.length ? n.toks[b] : null;
      if (t.t === 'str' || (t.t === 'name' && !(/\d$/.test(t.v) && nNames.has(stem(t.v))))) names++;
      // a number that shows the same setting as one you changed, which you left (force = 20.0 if a == 1 else -10.0)
      else if (t.t === 'num' && q && q.t === 'num' && u.steps.some((s) => same(Math.abs(numOf(t.v)), s[0]) && same(Math.abs(numOf(q.v)), s[1]))) names++;
    }
    if (names || (r > 0 && (moved || u.heavy))) r = u.cost + moved; // (a line mostly rewritten is all or nothing)
    if (r > 0 && equivalent(n, e)) r = 0;
    ctx.miss.set(key, r);
    return r;
  }
  function judge(ctx, N) {
    const pairs = match(N.map((l) => l.key), ctx.ek), mN = new Int32Array(N.length).fill(-1), mE = new Int32Array(ctx.E.length).fill(-1);
    for (const [i, j] of pairs) { mN[i] = j; mE[j] = i; }
    let dist = 0, conseq = 0;
    const open = [];
    for (const h of hunks(pairs, N.length, ctx.E.length)) {
      conseq += h.a1 - h.a0;
      for (let j = h.b0; j < h.b1; j++) {
        if (!ctx.mine[j]) { conseq++; continue; } // a line you kept, which the candidate writes differently
        let best = -1, bd = ctx.mine[j].cost;
        for (let i = h.a0; i < h.a1; i++) { const dd = missed(ctx, N[i], j); if (dd < bd) { bd = dd; best = i; } }
        if (best < 0 && h.a1 > h.a0) best = h.a0;
        if (bd > 0) { dist += bd; open.push({ e: j, n: best, a0: h.a0, a1: h.a1 }); }
      }
    }
    // the lines you deleted must be gone
    const extra = new Map();
    N.forEach((l, i) => { if (mN[i] < 0) { if (!extra.has(l.key)) extra.set(l.key, []); extra.get(l.key).push(i); } });
    for (const o of ctx.deleted) {
      const left = extra.get(ctx.O[o].key);
      if (left && left.length) { const i = left.shift(); dist += ctx.O[o].tk.length; open.push({ del: o, n: i, a0: i, a1: i + 1 }); }
    }
    return { dist, conseq, open, mN, mE };
  }
  // what each setting changes in the candidate script: the lines, where it puts lines, and the numbers it moves
  function influence(ctx, st) {
    if (st.infl) return st.infl;
    const out = new Map(), nk = st.N.map((l) => l.key);
    for (const k of st.knobs) {
      if (k.noProbe || k.kind === 'formula' || k.kind === 'text' || ctx.over()) continue;
      const v0 = k.get(st.S), rec = { lines: new Set(), gaps: new Set(), nums: [], shown: new Set(), tok: new Map(), whole: new Set() };
      let probes = 0;
      for (const v1 of probesOf(k, v0)) {
        if (probes >= (k.kind === 'enum' ? 2 : 1)) break;
        const S1 = clone(st.S);
        try { k.set(S1, v1); } catch (e) { continue; }
        if (!validate(ctx, S1).ok) continue;
        const code = gen(ctx, S1);
        if (code === null) continue;
        probes++;
        const N1 = statements(code).lines;
        for (const h of hunks(match(nk, N1.map((l) => l.key)), st.N.length, N1.length)) {
          for (let i = h.a0; i < h.a1; i++) rec.lines.add(i);
          if (h.b1 > h.b0) rec.gaps.add(h.a0);
          /* line for line: which tokens it moves (and how far, for numbers); a line it rewrites in another shape is
             whole. In a small stretch that gains or loses lines (weight decay's groups), each line with its likeliest
             counterpart; a big rewrite (another way of learning) says nothing about single tokens. */
          const pairs = h.a1 - h.a0 === h.b1 - h.b0 ? NA.range(h.a1 - h.a0).map((t) => [h.a0 + t, h.b0 + t]) : h.a1 - h.a0 <= 4 && h.b1 - h.b0 <= 8 ? NA.range(h.a1 - h.a0).map((t) => {
            const a = st.N[h.a0 + t];
            let best = -1, bs = 0.5;
            for (let b = h.b0; b < h.b1; b++) { const s = align(a.tk, N1[b].tk).length / Math.max(a.tk.length, N1[b].tk.length); if (s > bs) { bs = s; best = b; } }
            return best >= 0 ? [h.a0 + t, best] : null;
          }).filter(Boolean) : [];
          for (const [i, b] of pairs) {
            const td = tokenDiff(st.N[i], N1[b]);
            if (td.added.length || td.removed.length) rec.whole.add(i);
            for (const [x, y, ti, tj] of td.pairs) {
              if (!rec.tok.has(i)) rec.tok.set(i, new Set());
              rec.tok.get(i).add(ti);
              // a number counts as moved by the setting where it keeps its place (weight_decay=0.0 is not momentum=0.9)
              if (x.t !== 'num' || y.t !== 'num' || st.N[i].tk[ti - 1] !== N1[b].tk[tj - 1] || st.N[i].tk[ti - 2] !== N1[b].tk[tj - 2]) continue;
              const q = { i, ti, v1, t0: numOf(x.v), t1: numOf(y.v) };
              rec.nums.push(q);
              if (k.kind === 'num' && same(q.t0, v0) && same(q.t1, v1)) rec.shown.add(i);
            }
          }
        }
      }
      if (rec.lines.size || rec.gaps.size) out.set(k, rec);
    }
    return (st.infl = out);
  }
  function probesOf(k, v) {
    if (k.kind === 'bool') return [!v];
    if (k.kind === 'enum') return k.opts.filter((o) => o !== v);
    if (k.kind !== 'num') return [];
    if (k.list) return k.list.filter((x) => !same(x, v));
    const c = k.int ? (Math.abs(v) >= 10 ? [Math.round(v * 1.5), Math.round(v * 0.5), v + 1] : [v + 1, v - 1, v + 2]) : v === 0 ? [0.5, -0.5, 1] : [v * 1.5, v * 0.5, v + 1, -v];
    return c.filter((x) => Number.isFinite(x) && !same(x, v) && (!k.range || (x >= k.range[0] && x <= k.range[1])));
  }
  const touches = (rec, u) => {
    if (u.n >= 0 && rec.lines.has(u.n)) return true;
    for (let i = u.a0; i < u.a1; i++) if (rec.lines.has(i)) return true;
    if (u.a1 === u.a0 || u.n < 0) { for (const g of rec.gaps) if (Math.abs(g - u.a0) <= 3) return true; for (let i = u.a0 - 2; i <= u.a0 + 1; i++) if (rec.lines.has(i)) return true; }
    return false;
  };
  const moveKey = (m) => (m.remove !== undefined ? 'rm' + m.remove : m.knob.id + '=' + JSON.stringify(m.v));
  // guesses that cost little: a setting with the number you replaced takes yours; a reward or curve you wrote becomes its formula
  function proposeDirect(ctx, st) {
    const moves = [], seen = new Set(), add = (m, j) => { const k = moveKey(m); if (!seen.has(k)) { seen.add(k); m.target = j; moves.push(m); } };
    for (const u of st.J.open) {
      if (u.del !== undefined || u.n < 0) continue;
      const e = ctx.E[u.e], n = st.N[u.n], mine = new Set(ctx.mine[u.e].plus);
      for (const [x, y, ti, tj] of tokenDiff(n, e).pairs) {
        if (!mine.has(tj)) continue;
        if (x.t === 'num' && y.t === 'num') { for (const k of st.knobs) if (k.kind === 'num' && same(k.get(st.S), numOf(x.v))) add({ knob: k, v: k.int ? numOf(y.v) : +sig(numOf(y.v)), direct: true, at: [u.n, ti, n.key] }, u.e); }
        else if (x.t === 'str' && y.t === 'str' && x.body !== y.body) {
          for (const k of st.knobs) {
            const v = k.get(st.S);
            if (k.kind === 'enum' && v === x.body) add({ knob: k, v: y.body, direct: true }, u.e);
            else if (k.kind === 'text' && x.body.length > 2) {
              let a = x.body, b = y.body;
              try { a = JSON.parse(`"${x.body}"`); b = JSON.parse(`"${y.body}"`); } catch (er) { /* kept as written */ }
              if (v.includes(a)) add({ knob: k, v: v.replace(a, b), direct: true }, u.e);
            }
          }
        }
      }
    }
    for (const u of st.J.open) {
      if (u.del !== undefined) continue;
      const s = splitStatement(ctx.E[u.e]);
      if (!s || s.op !== '=' || /[,.[]/.test(s.target)) continue;
      let ast;
      try { ast = parseExpr(s.expr); } catch (e) { continue; }
      for (const k of st.knobs) {
        if (k.kind !== 'formula') continue;
        try { const f = toFormula(ast, k.vars); if (f !== k.get(st.S)) add({ knob: k, v: f, direct: true, formula: true }, u.e); }
        catch (e) { if ([...namesOf(ast)].some((nm) => k.vars.includes(nm)) && s.target === (k.scalar ? 'r' : 'y')) ctx.why.set(u.e, ctx.why.get(u.e) || `${cap(labelOf(k))}: ${e.message}`); }
      }
    }
    return moves;
  }
  // every setting that changes the lines still to explain, tried at its values; a block whose lines you deleted, removed
  function proposeWide(ctx, st) {
    const infl = influence(ctx, st), moves = [], seen = new Set(), add = (m, j) => { const k = moveKey(m); if (!seen.has(k)) { seen.add(k); m.target = j; moves.push(m); } };
    for (const u of st.J.open) {
      const e = u.del === undefined ? ctx.E[u.e] : null, n = u.n >= 0 ? st.N[u.n] : null, td = e && n ? tokenDiff(n, e) : null;
      const fresh = e ? [...new Set(e.toks.filter((t) => t.t === 'num').flatMap((t) => [numOf(t.v), -numOf(t.v)]))] : []; // (r -= 1 is an amount of -1)
      for (const [k, rec] of infl) {
        if (!touches(rec, u)) continue;
        const v0 = k.get(st.S);
        if (k.kind === 'enum') k.opts.forEach((o) => { if (o !== v0) add({ knob: k, v: o }, u.e); });
        else if (k.kind === 'bool') add({ knob: k, v: !v0 }, u.e);
        else if (k.kind === 'num') {
          if (u.del !== undefined && !same(v0, 0)) add({ knob: k, v: 0 }, u.e);
          for (const v of fresh) if (!same(v, v0)) add({ knob: k, v: k.int ? v : +sig(v) }, u.e);
          // a number the page works out from the setting: solved as if it followed the setting in a straight line
          if (td) for (const [x, y, ti] of td.pairs) if (x.t === 'num' && y.t === 'num') for (const q of rec.nums) {
            if (q.i !== u.n || q.ti !== ti || q.t1 === q.t0) continue;
            let v = v0 + (numOf(y.v) - q.t0) * (q.v1 - v0) / (q.t1 - q.t0);
            v = k.int ? Math.round(v) : +sig(v);
            if (Number.isFinite(v) && !same(v, v0)) add({ knob: k, v, solved: true }, u.e);
          }
        }
      }
    }
    if (st.J.open.some((u) => u.del !== undefined)) st.S.blocks.forEach((b, i) => { if (!b.grp) add({ remove: i, what: `Removed block ${i + 1} (${partName(b.type)})` }, -1); });
    return moves;
  }
  /* One step of the search: every guess is applied to a copy of the setup, checked, written out and compared with
     your edit; the one that leaves least unexplained wins (ties: the fewest other lines changed). A number solved
     from another setting must not change a line that shows that setting itself (the steps, to print less often). */
  function step(ctx, st, wide) {
    let best = null;
    for (const m of wide ? proposeWide(ctx, st) : proposeDirect(ctx, st)) {
      if (ctx.over()) break;
      const S1 = clone(st.S);
      try { if (m.remove !== undefined) S1.blocks.splice(m.remove, 1); else m.knob.set(S1, m.v); } catch (e) { continue; }
      const v = validate(ctx, S1, m);
      if (ctx.trace && !v.ok) ctx.trace(moveKey(m), "invalid: " + v.why);
      // why a guess that took your number could not be used: kept for the line, if the setting is the one it shows
      if (!v.ok) { if (m.direct && m.target >= 0) { if (!ctx.fails.has(m.target)) ctx.fails.set(m.target, []); ctx.fails.get(m.target).push({ knob: m.knob, at: m.at, formula: m.formula, why: v.whole ? v.why : `With ${labelOf(m.knob)} ${show(m.knob, m.v)}, ${v.why}` }); } continue; }
      const code = gen(ctx, S1);
      if (code === null) continue;
      const N1 = statements(code).lines, J1 = judge(ctx, N1);
      if (ctx.trace) ctx.trace(moveKey(m), J1.dist, st.J.dist);
      if (J1.dist >= st.J.dist) continue;
      if (m.solved) {
        const rec = influence(ctx, st).get(m.knob), hit = rec && [...rec.shown].some((i) => st.J.mN[i] >= 0 && J1.mE[st.J.mN[i]] < 0);
        if (hit) continue;
      }
      const score = J1.dist * 1e4 + J1.conseq * 10 + (m.solved ? 5 : 0) + (m.remove !== undefined ? 3 : 0);
      if (!best || score < best.score) best = { score, S: S1, code, N: N1, J: J1, m };
    }
    if (!best) return null;
    const m = best.m, old = m.knob ? m.knob.get(st.S) : null;
    return { S: best.S, code: best.code, N: best.N, J: best.J, knobs: m.remove !== undefined ? knobsOf(best.S, makeData(ctx, best.S)) : st.knobs, moves: [...st.moves, Object.assign({}, m, { old })], infl: null };
  }

  /* Read an edited script back into a setup. Returns { ok: true, moves, changes (in words), code (the script the page
     writes now) }, { ok: true, nothing: true } when only comments, spacing or equivalent writing changed, or
     { ok: false, problems: [{ line, why }], partial } where partial (if any) is what it could follow. opts.d: the
     setup's data, when already made. */
  PE.solve = function (setup, text, o = {}) {
    const ctx = { cache: new Map(), miss: new Map(), why: new Map(), fails: new Map(), gens: 0, t0: now() };
    ctx.trace = o.traceMoves;
    ctx.over = () => ctx.gens > (o.budget || 3000) || now() - ctx.t0 > (o.ms || 6000);
    const S0 = clone(setup);
    if (o.d) ctx.cache.set(dataKey(S0), o.d);
    ctx.d0 = makeData(ctx, S0);
    ctx.ok0 = checkOf(S0, ctx.d0).ok;
    const O = gen(ctx, S0);
    if (O === null) return { ok: false, problems: [{ line: 1, why: 'The page cannot write this model as a script right now.' }] };
    const rd = statements(text);
    if (rd.err) return { ok: false, syntax: true, problems: [{ line: rd.err.line, why: rd.err.msg }] };
    ctx.O = statements(O).lines; ctx.E = rd.lines; ctx.ek = ctx.E.map((l) => l.key);
    // your changes: the lines you changed or added (and for each, the tokens you changed), and the lines you deleted
    const pOE = match(ctx.O.map((l) => l.key), ctx.ek);
    ctx.mine = new Array(ctx.E.length).fill(null); ctx.deleted = [];
    const keptO = new Uint8Array(ctx.O.length);
    for (const [i] of pOE) keptO[i] = 1;
    for (const h of hunks(pOE, ctx.O.length, ctx.E.length)) {
      for (let j = h.b0; j < h.b1; j++) {
        let o = null, bd = Infinity;
        for (let i = h.a0; i < h.a1; i++) { const p = align(ctx.O[i].tk, ctx.E[j].tk), dd = ctx.O[i].tk.length + ctx.E[j].tk.length - 2 * p.length; if (dd < bd && p.length > 0.4 * ctx.E[j].tk.length) { bd = dd; o = ctx.O[i]; } }
        // plus: your tokens (new or in place of the page's); minus: the page's tokens you only took out
        const p = o ? align(o.tk, ctx.E[j].tk) : [], inE = new Set(p.map((q) => q[1])), td = o ? tokenDiff(o, ctx.E[j]) : null;
        const plus = ctx.E[j].tk.map((_, b) => b).filter((b) => !inE.has(b)), minus = td ? td.removed.map((t) => o.toks.indexOf(t)) : [];
        const steps = td ? td.pairs.filter((q) => q[0].t === 'num' && q[1].t === 'num').map((q) => [Math.abs(numOf(q[0].v)), Math.abs(numOf(q[1].v))]) : [];
        ctx.mine[j] = { o, plus, plusSet: new Set(plus), minus, steps, cost: Math.max(1, plus.length + minus.length), heavy: plus.length + minus.length > Math.max(3, 0.5 * ctx.E[j].tk.length) };
      }
      // the page's lines here that none of yours came from were deleted; a line you rewrote whole stands in for its likeliest one
      const claimed = new Set(ctx.mine.slice(h.b0, h.b1).map((u) => u.o).filter(Boolean));
      for (let j = h.b0; j < h.b1; j++) {
        if (ctx.mine[j].o) continue;
        let best = null, bs = -1;
        for (let i = h.a0; i < h.a1; i++) { if (claimed.has(ctx.O[i])) continue; const s = align(ctx.O[i].tk, ctx.E[j].tk).length; if (s > bs) { bs = s; best = ctx.O[i]; } }
        if (best) { claimed.add(best); ctx.mine[j].replaced = best; }
      }
      for (let i = h.a0; i < h.a1; i++) if (!claimed.has(ctx.O[i])) ctx.deleted.push(i);
    }
    const changedLines = ctx.mine.filter(Boolean).length + ctx.deleted.length;
    if (!changedLines) return { ok: true, nothing: true, moves: [], changes: [], code: O };
    if (changedLines > Math.max(16, 0.4 * ctx.O.length)) {
      const j = ctx.mine.findIndex(Boolean);
      return { ok: false, problems: [{ line: j >= 0 ? ctx.E[j].line : 1, why: `${changedLines} statements changed. The page reads back small changes to the script it wrote: a few numbers or names at a time.` }] };
    }
    let st = { S: S0, code: O, N: ctx.O, knobs: knobsOf(S0, ctx.d0), moves: [], infl: null };
    st.J = judge(ctx, st.N);
    for (let iter = 0; iter < 16 && st.J.dist > 0 && !ctx.over(); iter++) {
      const next = step(ctx, st, false) || step(ctx, st, true);
      if (o.trace) o.trace(next ? `${next.moves.map(moveKey).join(", ")} -> ${next.J.dist}` : "stuck", st.J.open);
      if (!next) break;
      st = next;
    }
    const changes = summarise(st.moves);
    if (st.J.dist === 0) return st.moves.length ? { ok: true, moves: st.moves, changes, code: st.code, S: st.S } : { ok: true, nothing: true, same: true, moves: [], changes: [], code: O };
    const problems = st.J.open.map((u) => explain(ctx, st, u)).filter(Boolean).sort((a, b) => a.line - b.line).filter((p, i, l) => !i || p.line !== l[i - 1].line || p.why !== l[i - 1].why);
    return { ok: false, problems, partial: st.moves.length ? { moves: st.moves, changes } : null };
  };
  // the moves in words, a setting changed twice once (from its first value to its last)
  function summarise(moves) {
    const out = [], at = new Map();
    for (const m of moves) {
      if (m.remove !== undefined) { out.push({ text: m.what }); continue; }
      const k = m.knob.id;
      if (at.has(k)) { at.get(k).v = m.v; continue; }
      const c = { knob: m.knob, old: m.old, v: m.v };
      at.set(k, c); out.push(c);
    }
    const what = (k) => (k.where ? `${k.where}: ${k.what}` : cap(k.what));
    return out.filter((c) => c.text || JSON.stringify(c.old) !== JSON.stringify(c.v)).map((c) => c.text || (c.knob.kind === 'formula' ? `${what(c.knob)}: ${c.v || 'none (the table of amounts)'}` : c.knob.kind === 'text' ? `${what(c.knob)} edited` : `${what(c.knob)} ${show(c.knob, c.old)} → ${show(c.knob, c.v)}`));
  }

  // ------------------------------------------------------------------ saying why
  const listWords = (xs) => (xs.length < 2 ? xs.join('') : `${xs.slice(0, -1).join(', ')} and ${xs[xs.length - 1]}`);
  // the name a number has in its line: a keyword (lr=0.01), what it is assigned to (batch = 32, or a tuple's member)
  function nameOfNumber(line, ti) {
    const t = line.toks;
    if (t[ti - 1] && t[ti - 1].v === '=' && t[ti - 2] && t[ti - 2].t === 'name') return t[ti - 2].v;
    const eq = t.findIndex((q) => q.t === 'op' && q.v === '=');
    if (eq > 0 && ti > eq) {
      const lhs = t.slice(0, eq).filter((q) => q.t === 'name'), rhs = [];
      let depth = 0, item = 0;
      for (let k = eq + 1; k < t.length; k++) { const v = t[k].t === 'op' ? t[k].v : ''; if (OPEN[v]) depth++; else if (/^[)\]}]$/.test(v)) depth--; else if (v === ',' && !depth) { item++; continue; } rhs[k] = item; }
      if (lhs.length === item + 1 && lhs.every((q, k) => t[k * 2] === q)) return lhs[rhs[ti]] ? lhs[rhs[ti]].v : null;
    }
    return null;
  }
  // how the options of a setting are written in this line (F.relu, torch.tanh, …), or their names when that is long
  function optionCode(ctx, st, k, u) {
    const out = [];
    for (const o of k.opts) {
      if (ctx.gens > (ctx.budgetExplain || 4000)) break;
      const S1 = clone(st.S);
      try { k.set(S1, o); } catch (e) { continue; }
      const code = o === k.get(st.S) ? st.code : validate(ctx, S1).ok ? gen(ctx, S1) : null;
      let snip = null;
      if (code) {
        const N1 = statements(code).lines, n = st.N[u.n], p = match(st.N.map((l) => l.key), N1.map((l) => l.key)), h = hunks(p, st.N.length, N1.length).find((q) => u.n >= q.a0 && u.n < q.a1);
        if (o === k.get(st.S)) snip = null;
        else if (h && h.b1 - h.b0 === h.a1 - h.a0) { const td = tokenDiff(n, N1[h.b0 + (u.n - h.a0)]); const add = td.pairs.map((q) => q[1]).concat(td.added); if (add.length && add.length <= 4) snip = add.map((q) => q.v).join(''); }
      }
      out.push(snip || (k.labels && k.labels[o]) || o);
    }
    return out;
  }
  function explain(ctx, st, u) {
    if (u.del !== undefined) {
      const l = ctx.O[u.del];
      return { line: lineNear(ctx, u), why: `You took out “${short(l)}”: the page writes this line for every model like yours and cannot leave it out. To take a block out, delete all its lines (or remove it on the Build screen).` };
    }
    const e = ctx.E[u.e], n = u.n >= 0 ? st.N[u.n] : null, mine = new Set(ctx.mine[u.e].plus);
    const td = n ? tokenDiff(n, e) : null, first = td && (td.pairs.find((q) => mine.has(q[3])) || null);
    const line = first ? first[1].line : e.line, infl = n ? influence(ctx, st) : new Map();
    const shows = (k, ti) => { const r = infl.get(k); return !!r && (r.whole.has(u.n) || r.nums.some((q) => q.i === u.n && q.ti === ti) || (r.tok.has(u.n) && r.tok.get(u.n).has(ti))); };
    // a number of yours the setting it shows cannot take (out of range, or a model that would not fit together)
    const fail = (ctx.fails.get(u.e) || []).find((f) => f.formula || (!!n && (f.at && f.at[2] === n.key ? shows(f.knob, f.at[1]) : touches(infl.get(f.knob) || { lines: new Set(), gaps: new Set() }, u))));
    if (fail) return { line, why: fail.why };
    if (ctx.why.has(u.e)) return { line, why: ctx.why.get(u.e) };
    // on the line of the reward (or of your curve), how to write one of your own
    const fk = st.knobs.find((k) => k.kind === 'formula'), target = (splitStatement(e) || {}).target;
    const hint = fk && target === (fk.scalar ? 'r' : 'y') ? (fk.scalar ? ` A reward of your own is read as a formula of ${listWords(fk.vars.slice(0, 4))}…, such as r = 10 * ${fk.vars[0]} - ${fk.vars[2] || fk.vars[1] || fk.vars[0]}.` : ' Your curve is read as a formula in x, such as y = torch.sin(3 * x) + x ** 2.') : '';
    if (!ctx.mine[u.e].o && ctx.mine[u.e].replaced) return { line, why: `This line is too different from the page’s to read back: change one number or name at a time.${hint}` };
    if (!n || !ctx.mine[u.e].o) return { line, why: 'New lines are not read back: the page writes the script from the model. Add parts on the Build screen, and change settings in the lines the script already has.' };
    // a keyword the page does not write here (momentum=0.95)
    const kws = e.toks.map((t, i) => (t.t === 'name' && e.toks[i + 1] && e.toks[i + 1].v === '=' && mine.has(i) ? t.v : null)).filter(Boolean);
    if (kws.length) return { line, why: `The page has no setting for ${kws[0]} here.` };
    const changed = td.pairs.filter((q) => mine.has(q[3])), lineWho = [...infl].filter(([, r]) => r.whole.has(u.n)).map(([k]) => k);
    if (!changed.length && td.added.length) return { line, why: `The page has no setting that adds “${td.added.map((t) => t.v).join(' ').replace(/ ([.,:)\]])/g, '$1').replace(/([.([]) /g, '$1')}” to this line.${hint}` };
    if (changed.length && !td.added.length && changed.every((q) => q[1].t === 'str')) return { line, why: 'The page writes this text itself, the same whatever the settings: it cannot keep a change to it.' };
    const nums = changed.filter((q) => q[0].t === 'num' && q[1].t === 'num');
    const noValue = (ks) => `This line comes from ${listWords(ks.slice(0, 2).map(labelOf))}, but no value of ${ks.length > 1 ? 'them' : 'it'} gives the line you wrote.${hint}`;
    if (nums.length) {
      const [x, , ti] = nums[0], nm = nameOfNumber(n, ti), name = nm && nm.length > 1 ? nm : null, by = [...infl].filter(([, r]) => r.nums.some((q) => q.i === u.n && q.ti === ti)).map(([k]) => k);
      if (!by.length) return { line, why: lineWho.length ? noValue(lineWho) : `The page always writes ${x.v} here${name ? ` for ${name}` : ''}: it has no setting for it.` };
      const k = by[0], rec = infl.get(k), at = [...rec.shown].map((i) => st.J.mN[i]).filter((j) => j >= 0 && !rec.shown.has(u.n)).map((j) => ctx.E[j].line)[0];
      if (rec.shown.has(u.n) && rec.nums.filter((q) => q.i === u.n && q.v1 === rec.nums.find((z) => z.i === u.n).v1).length > 1) return { line, why: `This line shows ${labelOf(k)} (${show(k, k.get(st.S))}) more than once: change every one of them.` };
      return { line, why: `${name ? cap(name) : 'This number'} follows from ${listWords(by.slice(0, 2).map(labelOf))} (${show(k, k.get(st.S))}): change ${by.length > 1 ? 'those' : 'that'} instead${at ? `, on line ${at}` : ''}.` };
    }
    // the settings that change the very tokens you changed (or rewrite the line around them)
    const tis = changed.map((q) => q[2]), who = [...infl].filter(([k]) => (tis.length ? tis.some((t) => shows(k, t)) : infl.get(k).whole.has(u.n))).map(([k]) => k);
    if (!who.length) return { line, why: lineWho.length ? noValue(lineWho) : `The page writes this line the same way whatever the settings: it cannot keep a change to it.${hint}` };
    const en = who.find((k) => k.kind === 'enum' && k.opts.length > 1);
    if (en) return { line, why: `This comes from ${labelOf(en)}, which can be ${listWords(optionCode(ctx, st, en, u))}.` };
    return { line, why: noValue(who) };
  }
  const short = (l) => { const s = l.toks.map((t) => t.v).join(' ').replace(/ ([.,:)\]])/g, '$1').replace(/([.([]) /g, '$1'); return s.length > 60 ? s.slice(0, 57) + '…' : s; };
  function lineNear(ctx, u) {
    // the line of your edit where the deleted one was
    const o = u.del;
    for (let i = o - 1; i >= 0; i--) { const j = ctx.E.findIndex((l) => l.key === ctx.O[i].key); if (j >= 0) return ctx.E[j].line + 1; }
    return 1;
  }

  // ------------------------------------------------------------------ carrying an edit over to a script that changed
  /* The model changed while you were editing (on the Build screen, by Undo): your changes, line by line, are made
     again on the script as it is now. A line both changed is a conflict. */
  function rebase(base, edit, cur) {
    const B = base.split('\n'), Ed = edit.split('\n'), C = cur.split('\n'), t = (s) => s.trimEnd();
    const toC = new Int32Array(B.length).fill(-1);
    for (const [i, j] of match(B.map(t), C.map(t))) toC[i] = j;
    const reps = [], conflicts = [];
    for (const h of hunks(match(B.map(t), Ed.map(t)), B.length, Ed.length)) {
      const lines = Ed.slice(h.b0, h.b1);
      if (h.a1 > h.a0) {
        const at = [];
        for (let i = h.a0; i < h.a1; i++) at.push(toC[i]);
        if (at.some((j) => j < 0) || at.some((j, k) => k && j !== at[k - 1] + 1)) { conflicts.push(h.b0 + 1); continue; }
        reps.push({ from: at[0], to: at[at.length - 1] + 1, lines });
      } else {
        const before = h.a0 > 0 ? toC[h.a0 - 1] : -1, after = h.a0 < B.length ? toC[h.a0] : C.length;
        if (h.a0 > 0 && before < 0 && after < 0) { conflicts.push(h.b0 + 1); continue; }
        const pos = before >= 0 || h.a0 === 0 ? before + 1 : after;
        reps.push({ from: pos, to: pos, lines });
      }
    }
    if (conflicts.length) return { conflicts };
    reps.sort((a, b) => b.from - a.from);
    const out = C.slice();
    for (const r of reps) out.splice(r.from, r.to - r.from, ...r.lines);
    return { text: out.join('\n') };
  }
  PE.rebase = rebase;
  PE.statements = statements;
  PE._t = { knobsOf, judge, missed, tokenDiff, match, align };
  PE.equivalent = (a, b) => equivalent(statements(a).lines[0], statements(b).lines[0]);

  // ------------------------------------------------------------------ the tab
  const UNDERSTOOD = 'Numbers and names the page has a setting for: the learning rate, batch size, steps, optimizer, schedule, seed, clipping and weight decay; each block’s size, activation and settings (dropout rates, filters, heads…); the data; the world; reward amounts and the reward formula, as Python (r = 10 * goal - wall); evolution settings (population, mutation size, and grow and prune’s: the changes it can make, their odds, the largest network…). It also reads a block whose lines you delete as that block removed. It does not keep comments or spacing: the page writes the script again from the model.';
  const label = (lab) => (lab.isClassicSetup() ? `Python · ${lab.setup.blocks.some((b) => b.type === 'boost') ? 'XGBoost' : 'scikit-learn'}` : 'Python · PyTorch');
  // the script as lines of highlighted tokens (a docstring over several lines is split, keeping its colour)
  function lineNodes(src) {
    const el = NA.el, lines = [[]];
    for (const node of NA.labUI.pyHighlight(src)) {
      const cls = node.nodeType === 1 ? node.className : null;
      node.textContent.split('\n').forEach((t, k) => { if (k) lines.push([]); if (t) lines[lines.length - 1].push(cls ? el('span', { class: cls, text: t }) : document.createTextNode(t)); });
    }
    return lines;
  }
  function changedLines(before, after) {
    const A = before.split('\n').map((s) => s.trimEnd()), B = after.split('\n').map((s) => s.trimEnd()), kept = new Set(match(A, B).map((q) => q[1])), out = new Set();
    B.forEach((s, j) => { if (!kept.has(j) && s.trim()) out.add(j + 1); });
    return out;
  }
  function copyButton(get) {
    const b = NA.el('button', { class: 'btn btn-quiet code-copy', type: 'button' }, 'Copy');
    b.addEventListener('click', async () => {
      try { await navigator.clipboard.writeText(get()); b.textContent = 'Copied'; } catch (e) { b.textContent = 'Select the text to copy'; }
      setTimeout(() => { b.textContent = 'Copy'; }, 1800);
    });
    return b;
  }
  function downloadButton(lab, get) {
    return NA.el('button', { class: 'btn btn-quiet code-copy', type: 'button', title: 'Save the script as a .py file', onclick: () => {
      const url = URL.createObjectURL(new Blob([get()], { type: 'text/x-python' })), a = NA.el('a', { href: url, download: `model-${lab.setup.problem}.py` });
      document.body.append(a); a.click(); a.remove();
      setTimeout(() => URL.revokeObjectURL(url), 2000);
    } }, 'Download');
  }
  // your moves (Copy you), as the JSON file its script reads
  function demosButton(lab) {
    const d = lab.dataFor(), B = lab.demosFor(d);
    return NA.el('button', { class: 'btn btn-quiet code-copy', type: 'button', disabled: !B || !B.acts.length, title: B && B.acts.length ? `Save your ${B.acts.length} moves as demos.json, next to the script` : 'Jump in and play first: no moves of yours yet', onclick: () => {
      const json = JSON.stringify({ world: d.world, inputs: d.obsNames || null, obs: B.obs.map((o) => Array.from(o, (v) => +v.toFixed(5))), moves: B.acts });
      const url = URL.createObjectURL(new Blob([json], { type: 'application/json' })), a = NA.el('a', { href: url, download: 'demos.json' });
      document.body.append(a); a.click(); a.remove();
      setTimeout(() => URL.revokeObjectURL(url), 2000);
    } }, 'Download your moves');
  }
  const stateOf = (lab) => lab.pyEdit || (lab.pyEdit = { mode: 'view', note: null, changed: null, text: '', base: '', lastEdit: '' });
  // the script wide, over the page, for its long lines (the side panel is narrow); again to put it back
  function wideButton(lab, box) {
    const st = stateOf(lab), b = NA.el('button', { class: 'btn btn-quiet code-copy', type: 'button', title: 'Show the script wide, over the page', 'aria-pressed': String(!!st.wide) }, st.wide ? 'Narrow' : 'Wide');
    b.addEventListener('click', () => { st.wide = !st.wide; box.classList.toggle('is-wide', st.wide); b.textContent = st.wide ? 'Narrow' : 'Wide'; b.setAttribute('aria-pressed', String(st.wide)); });
    return b;
  }
  // the tab's contents: the script with Edit, Copy and Download, or the editor
  PE.render = function (lab, root) {
    const st = stateOf(lab), code = NA.builderPy.code(lab.setup, lab.dataFor());
    if (st.mode === 'edit') {
      if (!root.querySelector('.bd-pe-ta')) buildEditor(lab, root, st);
      else if (st.sync) st.sync();
      return;
    }
    if (st.shown !== undefined && st.shown !== code) { st.changed = null; if (st.note && st.note.kind !== 'no') st.note = null; } // the model changed since: old marks and notes no longer apply
    st.shown = code;
    const el = NA.el, lines = lineNodes(code), changed = st.changed || new Set(), tap = (window.matchMedia && matchMedia('(pointer: coarse)').matches) || (NA.mobile && NA.mobile.mobile);
    const pre = el('pre', { class: 'code bd-pe-code', tabindex: '0', 'aria-label': 'The script', title: `${tap ? 'Double-tap' : 'Double-click'} a line to edit the script there` }, el('code', { class: 'bd-pe-lines' }, ...lines.map((ns, k) => el('span', { class: 'bd-pe-line' + (changed.has(k + 1) ? ' is-new' : '') }, ...ns))));
    const edit = el('button', { class: 'btn btn-primary code-copy bd-pe-edit', type: 'button', title: 'Change the script: the page reads your changes back into the model', onclick: () => PE.startEdit(lab) }, '✎ Edit');
    const box = el('div', { class: 'code-box bd-pe-box' + (st.wide ? ' is-wide' : '') }, el('div', { class: 'code-head' }, el('span', { text: label(lab) }), el('span', { class: 'bd-pe-tools' }, edit, copyButton(() => code), downloadButton(lab, () => code))), pre);
    box.querySelector('.bd-pe-tools').append(wideButton(lab, box));
    // (Copy you: its script reads your moves from demos.json, saved here)
    if (lab.family() === 'control' && lab.ps().method === 'bc' && lab.demosFor) box.querySelector('.bd-pe-tools').prepend(demosButton(lab));
    // a double-click (or double-tap) on a line edits the script there, the word under it selected
    pre.addEventListener('dblclick', (e) => { const at = placeOf(pre, e); if (at) PE.startEdit(lab, undefined, at); });
    const intro = el('p', { class: 'sec-note bd-pe-intro' }, 'You can change this script: press ', el('strong', { text: 'Edit' }), ` or ${tap ? 'double-tap' : 'double-click'} a line, change a number or a name, then `, el('strong', { text: 'Apply' }),
      '. The model changes to match. If the page cannot follow a change, it says which line and why, and puts the script back as it was.');
    NA.clear(root).append(noteEl(lab, st), intro, box);
  };
  // the line (from 0) and column a pointer is on in the script shown
  function placeOf(pre, e) {
    const lineEl = e.target && e.target.closest ? e.target.closest('.bd-pe-line') : null;
    if (!lineEl) return null;
    const line = [...pre.querySelectorAll('.bd-pe-line')].indexOf(lineEl);
    let col = 0;
    const r = document.caretRangeFromPoint ? document.caretRangeFromPoint(e.clientX, e.clientY) : null, p = !r && document.caretPositionFromPoint ? document.caretPositionFromPoint(e.clientX, e.clientY) : null;
    const node = r ? r.startContainer : p ? p.offsetNode : null, off = r ? r.startOffset : p ? p.offset : 0;
    if (node && lineEl.contains(node)) {
      const walk = document.createTreeWalker(lineEl, NodeFilter.SHOW_TEXT);
      for (let t = walk.nextNode(); t; t = walk.nextNode()) { if (t === node) { col += off; break; } col += t.textContent.length; }
    }
    return { line: Math.max(0, line), col };
  }
  // the editor; text: an edit to show again (Show my edit); at: { line, col } to put the caret on, its word selected
  PE.startEdit = function (lab, text, at) {
    const st = stateOf(lab), code = NA.builderPy.code(lab.setup, lab.dataFor());
    Object.assign(st, { mode: 'edit', base: text !== undefined && st.base ? st.base : code, baseSnap: text !== undefined && st.baseSnap ? st.baseSnap : lab.snapshot(), text: text !== undefined ? text : code, marks: text !== undefined ? st.marks : null, note: text !== undefined ? st.note : null, focus: true, at: at || null });
    lab.pyDirty = true; lab.inspDirty = true;
    if (lab.parts && lab.parts.code) { NA.clear(lab.parts.code); PE.render(lab, lab.parts.code); }
  };
  function leave(lab) {
    const st = stateOf(lab);
    st.mode = 'view'; st.sync = null; st.marks = null;
    if (lab.parts && lab.parts.code) PE.render(lab, lab.parts.code);
  }
  // what happened, in place above the script: applied (with Undo), or not (with the reasons and your edit one press away)
  function noteEl(lab, st) {
    const el = NA.el, n = st.note;
    const box = el('div', { class: 'bd-pe-note' + (n ? ' is-' + n.kind : ''), role: 'status', 'aria-live': 'polite', tabindex: '-1' });
    if (!n) return box;
    const x = el('button', { class: 'icon-btn bd-pe-x', type: 'button', 'aria-label': 'Close this note', onclick: () => { st.note = null; box.replaceWith(noteEl(lab, st)); } }, '×');
    if (n.kind === 'ok') {
      box.append(el('p', null, el('strong', { text: 'Applied. ' }), n.changes.join('; ') + '.'),
        n.extra ? el('p', { class: 'sec-note', text: n.extra }) : null,
        el('div', { class: 'btn-row' }, el('button', { class: 'btn btn-quiet', type: 'button', onclick: () => { lab.undo(); } }, 'Undo'), x));
    } else if (n.kind === 'info') box.append(el('p', { text: n.text }), el('div', { class: 'btn-row' }, x));
    else {
      box.append(el('p', null, el('strong', { text: 'Not applied: ' }), n.lead),
        el('ul', { class: 'bd-pe-why' }, ...n.problems.slice(0, 6).map((p) => el('li', null, el('span', { class: 'bd-pe-ln mono', text: `Line ${p.line}` }), ' ', p.why))),
        n.problems.length > 6 ? el('p', { class: 'sec-note', text: `And ${n.problems.length - 6} more.` }) : null,
        el('div', { class: 'btn-row' }, el('button', { class: 'btn', type: 'button', onclick: () => { st.marks = n.problems; PE.startEdit(lab, st.lastEdit); } }, 'Show my edit'),
          n.partial ? el('button', { class: 'btn btn-quiet', type: 'button', title: n.partial.changes.join('; '), onclick: () => applyMoves(lab, n.partial.moves, n.partial.changes, st.shown || '') }, `Apply only what it could follow (${n.partial.changes.length})`) : null, x));
    }
    return box;
  }
  function applyMoves(lab, moves, changes, before) {
    const st = stateOf(lab);
    lab.edit((s) => { for (const m of moves) { if (m.remove !== undefined) { const b = s.blocks[m.remove]; if (lab.sel && b && lab.sel.id === b.id) lab.sel = null; s.blocks.splice(m.remove, 1); } else m.knob.set(s, m.v); } });
    const code = NA.builderPy.code(lab.setup, lab.dataFor()), changed = changedLines(before, code);
    st.note = { kind: 'ok', changes, extra: changed.size ? `The changed lines are marked. ${lab.training ? '' : ''}` : '' };
    st.changed = changed; st.shown = code; st.mode = 'view';
    if (lab.parts && lab.parts.code) PE.render(lab, lab.parts.code);
    lab.pyDirty = false;
    reveal(lab);
  }
  function reveal(lab) {
    const box = lab.parts && lab.parts.code && lab.parts.code.querySelector('.bd-pe-note');
    if (box && box.firstChild) { box.scrollIntoView({ block: 'nearest' }); box.focus({ preventScroll: true }); }
  }
  // Apply: read the edit back (after the page has drawn "Reading your edit…")
  function apply(lab, text) {
    const st = stateOf(lab);
    st.text = text; st.lastEdit = text;
    let edit = text;
    const cur = NA.builderPy.code(lab.setup, lab.dataFor());
    if (cur !== st.base) {
      const r = rebase(st.base, text, cur);
      if (r.conflicts) {
        st.note = { kind: 'no', lead: 'the model changed while you were editing, in lines you changed too. The script is back as the model is now.', problems: r.conflicts.map((line) => ({ line, why: 'This line changed in the model as well.' })) };
        st.base = cur; st.baseSnap = lab.snapshot();
        leave(lab); reveal(lab);
        return;
      }
      edit = r.text;
    }
    let res;
    try { res = PE.solve(lab.setup, edit, { d: lab.dataFor() }); } catch (e) { res = { ok: false, problems: [{ line: 1, why: 'Something went wrong reading the script: ' + (e.message || e) }] }; }
    if (res.ok && !res.nothing) { applyMoves(lab, res.moves, res.changes, cur); return; }
    st.note = res.ok ? { kind: 'info', text: res.same ? 'Nothing to apply: your lines compute the same as the page’s, so the model stays as it is.' : 'Nothing to apply: only comments or spacing changed, and the page writes those itself.' }
      : { kind: 'no', lead: res.syntax ? 'Python could not read the script.' : 'the page could not follow every change, so the model is as it was and so is the script.', problems: res.problems, partial: res.partial };
    leave(lab); reveal(lab);
  }
  // the editor: a text area over the highlighted script, with line numbers; Tab indents, Ctrl+Enter applies
  function buildEditor(lab, root, st) {
    const el = NA.el, hintId = 'bd-pe-hint-' + Math.random().toString(36).slice(2, 7);
    const ta = el('textarea', { class: 'bd-pe-ta', spellcheck: 'false', autocapitalize: 'off', autocomplete: 'off', autocorrect: 'off', wrap: 'off', 'aria-label': 'The script, to edit', 'aria-describedby': hintId });
    ta.value = st.text;
    const hl = el('code', { class: 'bd-pe-lines' }), gut = el('div', { class: 'bd-pe-gutter', 'aria-hidden': 'true' }), caret = el('span', { class: 'bd-pe-caret', 'aria-hidden': 'true' });
    const area = el('div', { class: 'bd-pe-area' }, el('pre', { class: 'bd-pe-hl', 'aria-hidden': 'true' }, hl), ta, caret), wrap = el('div', { class: 'bd-pe-wrap' }, gut, area);
    const applyBtn = el('button', { class: 'btn btn-primary bd-pe-apply', type: 'button', title: 'Read your changes back into the model (Ctrl+Enter)' }, 'Apply');
    const cancel = el('button', { class: 'btn btn-quiet code-copy', type: 'button', title: 'Leave the editor; the model stays as it is', onclick: () => { st.text = ta.value; if (ta.value !== st.base) st.lastEdit = ta.value; st.note = null; leave(lab); } }, 'Cancel');
    const moved = el('p', { class: 'sec-note bd-pe-moved', hidden: true }, 'The model changed since you started editing. Apply makes your changes on the model as it is now.');
    const box = el('div', { class: 'code-box bd-pe-box is-editing' + (st.wide ? ' is-wide' : '') }, el('div', { class: 'code-head' }, el('span', { text: 'Editing · ' + label(lab) }), el('span', { class: 'bd-pe-tools' }, applyBtn, cancel, copyButton(() => ta.value), downloadButton(lab, () => ta.value))), wrap,
      el('p', { class: 'sec-note bd-pe-hint', id: hintId }, 'Tab indents (Esc, then Tab, moves on). Ctrl+Enter applies.'));
    box.querySelector('.bd-pe-tools').append(wideButton(lab, box));
    NA.clear(root).append(noteEl(lab, st), box, moved, el('details', { class: 'bd-pe-help' }, el('summary', null, 'What the page reads back'), el('p', { class: 'sec-note', text: UNDERSTOOD })));
    const marks = new Map((st.marks || []).map((p) => [p.line, p.why]));
    let painted = '', raf = 0, tabMoves = false;
    function paint() {
      raf = 0;
      const text = ta.value;
      if (text === painted) return;
      painted = text;
      const lines = lineNodes(text);
      lines.push([]); // a spare line, so the text area never scrolls inside itself
      NA.clear(hl).append(...lines.map((ns, k) => el('span', { class: 'bd-pe-line' + (marks.has(k + 1) ? ' is-bad' : '') }, ...ns)));
      if (gut.childNodes.length !== lines.length) NA.clear(gut).append(...lines.map((_, k) => el('span', { class: marks.has(k + 1) ? 'is-bad' : null, title: marks.get(k + 1) || null, text: k < lines.length - 1 ? String(k + 1) : '' })));
    }
    const later = () => { if (!raf) raf = requestAnimationFrame(paint); };
    // keep the caret in view: a marker at its place, scrolled to
    function showCaret() {
      const before = ta.value.slice(0, ta.selectionEnd), row = before.split('\n').length - 1, col = before.length - before.lastIndexOf('\n') - 1, cs = getComputedStyle(ta);
      const lh = parseFloat(cs.lineHeight) || 18, cw = st.cw || (st.cw = measure(cs));
      caret.style.top = `${parseFloat(cs.paddingTop) + row * lh}px`; caret.style.left = `${parseFloat(cs.paddingLeft) + col * cw}px`; caret.style.height = `${lh}px`;
      caret.scrollIntoView({ block: 'nearest', inline: 'nearest' });
    }
    function measure(cs) { const s = el('span', { style: { font: cs.font, position: 'absolute', visibility: 'hidden', whiteSpace: 'pre' }, text: 'x'.repeat(40) }); document.body.append(s); const w = s.getBoundingClientRect().width / 40; s.remove(); return w || 7.2; }
    function insert(text) {
      ta.focus();
      if (!document.execCommand || !document.execCommand('insertText', false, text)) ta.setRangeText(text, ta.selectionStart, ta.selectionEnd, 'end');
    }
    function indent(out) {
      const v = ta.value, a = v.lastIndexOf('\n', ta.selectionStart - 1) + 1, bEnd = ta.selectionEnd, b = v.indexOf('\n', bEnd - (bEnd > ta.selectionStart && v[bEnd - 1] === '\n' ? 1 : 0));
      const end = b < 0 ? v.length : b, block = v.slice(a, end), next = block.split('\n').map((s) => (out ? s.replace(/^ {1,4}/, '') : '    ' + s)).join('\n');
      ta.setSelectionRange(a, end); insert(next); ta.setSelectionRange(a, a + next.length);
    }
    ta.addEventListener('input', () => { st.text = ta.value; if (ta.value.includes('\t')) { const p = ta.selectionStart, before = ta.value.slice(0, p); ta.value = ta.value.replace(/\t/g, '    '); const q = before.replace(/\t/g, '    ').length; ta.setSelectionRange(q, q); } later(); });
    ta.addEventListener('scroll', () => { if (ta.scrollTop || ta.scrollLeft) { ta.scrollTop = 0; ta.scrollLeft = 0; } });
    ta.addEventListener('keydown', (e) => {
      if (e.key === 'Escape') { e.stopPropagation(); tabMoves = true; return; }
      if (e.key === 'Tab' && !tabMoves && !e.ctrlKey && !e.altKey && !e.metaKey) {
        e.preventDefault();
        if (e.shiftKey || ta.value.slice(ta.selectionStart, ta.selectionEnd).includes('\n')) indent(e.shiftKey); else insert('    ');
        later(); return;
      }
      if (e.key !== 'Tab') tabMoves = false;
      if (e.key === 'Enter' && (e.ctrlKey || e.metaKey)) { e.preventDefault(); go(); return; }
      if (e.key === 'Enter' && !e.shiftKey && !e.altKey) {
        // the next line starts where this one did, one step further in after a colon
        const v = ta.value, a = v.lastIndexOf('\n', ta.selectionStart - 1) + 1, cur = v.slice(a, ta.selectionStart), ind = /^ */.exec(cur)[0] + (/:\s*$/.test(cur) ? '    ' : '');
        e.preventDefault(); insert('\n' + ind); later();
      }
    });
    ta.addEventListener('keyup', (e) => { if (/^(Arrow|Page|Home|End|Enter|Backspace|Delete)/.test(e.key) || e.key.length === 1) showCaret(); });
    ta.addEventListener('click', () => { tabMoves = false; });
    function go() {
      applyBtn.disabled = true; applyBtn.textContent = 'Reading your edit…';
      const text = ta.value;
      setTimeout(() => apply(lab, text), 30);
    }
    applyBtn.addEventListener('click', go);
    st.sync = () => { moved.hidden = lab.snapshot() === st.baseSnap; };
    st.sync();
    paint();
    if (st.focus) {
      st.focus = false;
      const first = (st.marks || [])[0], place = st.at;
      st.at = null;
      setTimeout(() => {
        ta.focus({ preventScroll: true });
        if (first) { const v = ta.value.split('\n'), at = v.slice(0, first.line - 1).join('\n').length + (first.line > 1 ? 1 : 0); ta.setSelectionRange(at, at); showCaret(); }
        else if (place) {
          // the line double-clicked, the word (a number, a name) under the pointer selected, ready to type over
          const v = ta.value.split('\n'), row = Math.min(place.line, v.length - 1), base = v.slice(0, row).join('\n').length + (row > 0 ? 1 : 0), s = v[row] || '';
          let a = Math.min(place.col, s.length), b = a;
          while (a > 0 && /[\w.]/.test(s[a - 1])) a--;
          while (b < s.length && /[\w.]/.test(s[b])) b++;
          ta.setSelectionRange(base + a, base + b); showCaret();
        } else box.scrollIntoView({ block: 'nearest' });
      }, 0);
    }
  }

  NA.builderPyEdit = PE;
})(typeof globalThis !== 'undefined' && globalThis.NA ? globalThis.NA : window.NA);
