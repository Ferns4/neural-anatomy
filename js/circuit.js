/* circuit.js — build your own full adder. A board where you place logic gates and wire them up, from the inputs
   A, B and Cin to the outputs S and Cout, with a truth table that checks all 8 input patterns as you go. Once both
   outputs are wired you can put it to work: every adder in the zoom views is then made of your full adder
   (arch-deep.js asks NA.circuit.eval), right or wrong, so a mistake shows up in the multiplications and the
   neurons above it. The circuit is kept in this browser; using it is switched off again on every visit. */
(function (NA) {
  'use strict';
  const { el, clear, clamp } = NA;
  const SVGNS = 'http://www.w3.org/2000/svg';
  const sv = (tag, attrs, ...kids) => {
    const n = document.createElementNS(SVGNS, tag);
    for (const k in attrs || {}) if (attrs[k] !== null && attrs[k] !== undefined && attrs[k] !== false) n.setAttribute(k, attrs[k]);
    for (const c of kids) if (c !== null && c !== undefined) n.append(c);
    return n;
  };
  const FN = { AND: (a, b) => a & b, OR: (a, b) => a | b, XOR: (a, b) => a ^ b, NAND: (a, b) => 1 - (a & b), NOR: (a, b) => 1 - (a | b), NOT: (a) => 1 - a };
  const TYPES = Object.keys(FN), arity = (t) => (t === 'NOT' ? 1 : 2);
  // transistors per gate, as the zoom views build them (XOR from four NANDs)
  const TR = { AND: 6, OR: 6, XOR: 16, NAND: 4, NOR: 4, NOT: 2 };
  // the board in its own units: the inputs down the left edge, the outputs down the right
  const BW = 100, BH = 62, GW = 14, GH = 10;
  const INS = [{ id: 'A', y: 14 }, { id: 'B', y: 31 }, { id: 'Cin', y: 48 }], OUTS = [{ id: 'S', y: 22 }, { id: 'Cout', y: 42 }];
  // the dots the inputs leave from and the outputs arrive at, clear of their labels
  const PIN_IN = 14, PIN_OUT = 85;
  const ANSWER = {
    gates: [{ id: 'g1', type: 'XOR', x: 30, y: 22 }, { id: 'g2', type: 'XOR', x: 56, y: 22 }, { id: 'g3', type: 'AND', x: 56, y: 39 }, { id: 'g4', type: 'AND', x: 30, y: 52 }, { id: 'g5', type: 'OR', x: 74, y: 46 }],
    wires: { 'g1.0': 'A', 'g1.1': 'B', 'g2.0': 'g1', 'g2.1': 'Cin', 'g3.0': 'g1', 'g3.1': 'Cin', 'g4.0': 'A', 'g4.1': 'B', 'g5.0': 'g3', 'g5.1': 'g4', S: 'g2', Cout: 'g5' },
  };
  const copy = (C) => JSON.parse(JSON.stringify(C));
  const blank = () => ({ gates: [], wires: {} });
  const valid = (C) => !!C && Array.isArray(C.gates) && !!C.wires && typeof C.wires === 'object' && C.gates.every((g) => g && FN[g.type] && typeof g.id === 'string' && Number.isFinite(g.x) && Number.isFinite(g.y));

  // where wires attach: a source (an input, or a gate's output on its right) and a sink (a gate's input on its
  // left, or an output)
  const outPin = (C, id) => { const i = INS.find((q) => q.id === id); if (i) return [PIN_IN, i.y]; const g = C.gates.find((q) => q.id === id); return g ? [g.x + GW / 2, g.y] : null; };
  const inPin = (C, sink) => {
    const o = OUTS.find((q) => q.id === sink);
    if (o) return [PIN_OUT, o.y];
    const [gid, k] = sink.split('.'), g = C.gates.find((q) => q.id === gid);
    return g ? [g.x - GW / 2, arity(g.type) === 1 ? g.y : g.y + (+k ? 2.6 : -2.6)] : null;
  };
  // every wire's value for one input pattern; null where a wire is missing or the wires go round in a loop
  function evaluate(C, a, b, c) {
    const val = new Map([['A', a], ['B', b], ['Cin', c]]), busy = new Set();
    const of = (id) => {
      if (val.has(id)) return val.get(id);
      const g = C.gates.find((q) => q.id === id);
      if (!g || busy.has(id)) return null;
      busy.add(id);
      const xs = [];
      for (let k = 0; k < arity(g.type); k++) {
        const s = C.wires[`${id}.${k}`], v = s == null ? null : of(s);
        if (v === null) { busy.delete(id); return null; }
        xs.push(v);
      }
      busy.delete(id);
      const v = FN[g.type](...xs);
      val.set(id, v);
      return v;
    };
    for (const g of C.gates) of(g.id);
    const out = {};
    for (const o of OUTS) out[o.id] = C.wires[o.id] == null ? null : of(C.wires[o.id]);
    return { val, out };
  }
  function hasLoop(C) {
    const state = new Map();
    const visit = (id) => {
      if (state.get(id) === 1) return true;
      if (state.get(id) === 2) return false;
      const g = C.gates.find((q) => q.id === id);
      if (!g) return false;
      state.set(id, 1);
      for (let k = 0; k < arity(g.type); k++) { const s = C.wires[`${id}.${k}`]; if (s && visit(s)) return true; }
      state.set(id, 2);
      return false;
    };
    return C.gates.some((g) => visit(g.id));
  }
  // the truth table: what the circuit gives for each of the 8 input patterns, against a full adder's answer
  function check(C) {
    const rows = [];
    for (let k = 0; k < 8; k++) {
      const a = (k >> 2) & 1, b = (k >> 1) & 1, c = k & 1, R = evaluate(C, a, b, c), want = [a ^ b ^ c, (a & b) | ((a ^ b) & c)];
      rows.push({ a, b, c, s: R.out.S, co: R.out.Cout, want, ok: R.out.S === want[0] && R.out.Cout === want[1] });
    }
    return { rows, complete: rows.every((r) => r.s !== null && r.co !== null), right: rows.filter((r) => r.ok).length, loop: hasLoop(C) };
  }

  const CX = { on: false, right: 0 };
  const saved = NA.store.get('circuit.fa', null);
  CX.circuit = valid(saved) ? saved : blank();
  let TABLE = null;
  function refresh() {
    const K = check(CX.circuit);
    TABLE = K.complete ? K.rows.map((r) => [r.s, r.co]) : null;
    CX.right = K.right;
    if (!TABLE) CX.on = false;
    NA.store.set('circuit.fa', CX.circuit);
    return K;
  }
  refresh();
  // [sum, carry] of the visitor's full adder (or the standard one while theirs is not finished)
  CX.eval = (a, b, c) => (TABLE ? TABLE[a * 4 + b * 2 + c] : [a ^ b ^ c, (a & b) | ((a ^ b) & c)]);
  CX.check = () => check(CX.circuit);
  CX.setOn = function (on, view) {
    CX.on = !!on && !!TABLE;
    if (view) { view.dirty = true; view.refreshTools(); view.refreshFocus(); }
  };
  // for tests and the answer button
  CX.load = (C) => { if (valid(C)) { CX.circuit = copy(C); refresh(); if (CX.ed) CX.ed.render(); } };
  CX.answer = () => copy(ANSWER);

  /* ------------------------------------------------------------------ the editor, over the zoom view */
  CX.open = function (view) {
    CX.close();
    const E = (CX.ed = { view, sel: null, pend: null, row: 7, prev: null, drag: null, down: null });
    const board = sv('svg', { class: 'cx-board', viewBox: `0 0 ${BW} ${BH}`, 'aria-label': 'Your circuit: inputs on the left, outputs on the right' });
    const table = el('table', { class: 'cx-table' });
    const status = el('p', { class: 'cx-status', role: 'status' });
    const use = el('input', { type: 'checkbox', id: 'cx-use' });
    const hint = el('p', { class: 'cx-hint', hidden: true, text: 'Hint: S is 1 when an odd number of A, B and Cin are 1, which two XOR gates in a row give you. Cout is 1 when at least two of them are 1: (A AND B) OR (Cin AND (A XOR B)), where A XOR B can be the output of your first XOR.' });
    const undo = el('button', { type: 'button', class: 'btn btn-quiet', hidden: true, onclick: () => { if (E.prev) { CX.circuit = E.prev; E.prev = null; E.sel = null; render(); } } }, '↶ Put mine back');
    const keep = () => { E.prev = copy(CX.circuit); };
    const head = el('div', { class: 'cx-head' },
      el('div', { class: 'cx-titles' }, el('strong', { text: 'Build your own full adder' }), el('span', { class: 'cx-sub', text: 'Add gates, then drag from a dot on the right of an input or a gate to a dot on the left of a gate or an output (or tap one dot, then the other). Tap a wired dot to unwire it; drag a gate to move it.' })),
      el('div', { class: 'cx-btns' },
        el('button', { type: 'button', class: 'btn btn-quiet', onclick: () => { hint.hidden = !hint.hidden; } }, 'Hint'),
        el('button', { type: 'button', class: 'btn btn-quiet', onclick: () => { keep(); CX.circuit = copy(ANSWER); E.sel = null; render(); } }, 'Show the answer'),
        el('button', { type: 'button', class: 'btn btn-quiet', onclick: () => { keep(); CX.circuit = blank(); E.sel = null; E.pend = null; render(); } }, 'Clear'),
        undo,
        el('button', { type: 'button', class: 'btn btn-primary', onclick: () => CX.close() }, 'Done')));
    const palette = el('div', { class: 'cx-palette', role: 'group', 'aria-label': 'Add a gate' }, ...TYPES.map((t) => el('button', { type: 'button', class: 'btn btn-quiet cx-add', title: `Add ${t === 'NOT' ? 'a' : 'an'} ${t} gate`, onclick: () => add(t) }, '+ ' + t)));
    const root = (E.root = el('section', { class: 'cx', role: 'dialog', 'aria-label': 'Build your own full adder' },
      head, hint,
      el('div', { class: 'cx-body' },
        el('div', { class: 'cx-boardwrap' }, board),
        el('div', { class: 'cx-side' }, palette, table, status, el('label', { class: 'cx-use', for: 'cx-use' }, use, el('span', { text: 'Use my full adder in every adder of these views' }))))));
    use.addEventListener('change', () => { CX.setOn(use.checked, view); render(); });
    root.addEventListener('keydown', (e) => {
      if (e.key === 'Escape') { e.preventDefault(); CX.close(); }
      else if ((e.key === 'Delete' || e.key === 'Backspace') && E.sel && e.target.tagName !== 'INPUT') { e.preventDefault(); removeGate(E.sel); }
    });
    // over the whole zoom view, its explanation included, so the board has room
    (view.stage.closest('.deep-stage') || view.stage).append(root);

    function add(type) {
      const C = CX.circuit;
      let n = 1;
      while (C.gates.some((g) => g.id === 'g' + n)) n++;
      // the first free spot of a loose grid, so new gates do not land on each other
      let spot = null;
      for (const y of [12, 26, 40, 54]) for (const x of [27, 43, 59, 74]) if (!spot && !C.gates.some((g) => Math.abs(g.x - x) < GW && Math.abs(g.y - y) < GH + 2)) spot = [x, y];
      if (!spot) spot = [30 + ((n * 7) % 50), 10 + ((n * 11) % 44)];
      C.gates.push({ id: 'g' + n, type, x: spot[0], y: spot[1] });
      E.sel = 'g' + n;
      render();
    }
    function removeGate(id) {
      const C = CX.circuit;
      C.gates = C.gates.filter((g) => g.id !== id);
      for (const k of Object.keys(C.wires)) if (k.startsWith(id + '.') || C.wires[k] === id) delete C.wires[k];
      if (E.sel === id) E.sel = null;
      if (E.pend === id) E.pend = null;
      render();
    }
    function connect(sink, src) {
      if (!sink || !src || sink.split('.')[0] === src) return; // a gate straight into itself
      CX.circuit.wires[sink] = src;
    }
    const toBoard = (e) => { const m = board.getScreenCTM(); if (!m) return [0, 0]; const p = new DOMPoint(e.clientX, e.clientY).matrixTransform(m.inverse()); return [p.x, p.y]; };
    const curve = (p, q) => { const mx = (p[0] + q[0]) / 2; return `M${p[0]} ${p[1]} C${Math.max(mx, p[0] + 6)} ${p[1]} ${Math.min(mx, q[0] - 6)} ${q[1]} ${q[0]} ${q[1]}`; };
    board.addEventListener('pointerdown', (e) => {
      const t = e.target.closest('[data-src],[data-sink],[data-del],[data-gate],[data-wire]');
      E.down = { x: e.clientX, y: e.clientY, t };
      if (!t) { if (E.sel || E.pend) { E.sel = null; E.pend = null; render(); } return; }
      if (t.dataset.del) { removeGate(t.dataset.del); E.down = null; return; }
      if (t.dataset.src) { E.drag = { kind: 'wire', from: t.dataset.src, at: toBoard(e) }; board.setPointerCapture(e.pointerId); return; }
      if (t.dataset.gate) {
        const g = CX.circuit.gates.find((q) => q.id === t.dataset.gate), p = toBoard(e);
        if (g) { E.drag = { kind: 'gate', id: g.id, dx: g.x - p[0], dy: g.y - p[1] }; board.setPointerCapture(e.pointerId); }
      }
    });
    board.addEventListener('pointermove', (e) => {
      const d = E.drag;
      if (!d) return;
      const p = toBoard(e);
      if (d.kind === 'wire') { d.at = p; const tmp = board.querySelector('.is-drawing'), s = outPin(CX.circuit, d.from); if (tmp && s) tmp.setAttribute('d', curve(s, p)); else render(); return; }
      const g = CX.circuit.gates.find((q) => q.id === d.id);
      if (g) { g.x = clamp(p[0] + d.dx, PIN_IN + 3 + GW / 2, PIN_OUT - 3 - GW / 2); g.y = clamp(p[1] + d.dy, GH / 2 + 1, BH - GH / 2 - 1); render(); }
    });
    const finish = (e) => {
      const moved = !!E.down && Math.hypot(e.clientX - E.down.x, e.clientY - E.down.y) > 5, d = E.drag, down = E.down && E.down.t;
      E.drag = null;
      if (d && d.kind === 'wire') {
        const under = document.elementFromPoint(e.clientX, e.clientY), sink = under && under.closest ? under.closest('[data-sink]') : null;
        if (moved && sink) connect(sink.dataset.sink, d.from);
        else if (!moved) E.pend = E.pend === d.from ? null : d.from; // a tap: the wire starts here, the next tap ends it
        render(); return;
      }
      if (d && d.kind === 'gate') { if (!moved) E.sel = E.sel === d.id ? null : d.id; render(); return; }
      if (down && !moved && down.dataset.sink) {
        if (E.pend) { connect(down.dataset.sink, E.pend); E.pend = null; }
        else delete CX.circuit.wires[down.dataset.sink];
        render();
      } else if (down && !moved && down.dataset.wire) { delete CX.circuit.wires[down.dataset.wire]; render(); }
      E.down = null;
    };
    board.addEventListener('pointerup', finish);
    board.addEventListener('pointercancel', () => { E.drag = null; E.down = null; render(); });

    function render() {
      const K = refresh(), C = CX.circuit, row = K.rows[E.row], R = evaluate(C, row.a, row.b, row.c), on = (id) => R.val.get(id) === 1;
      clear(board);
      board.append(sv('rect', { x: 0, y: 0, width: BW, height: BH, rx: 2, class: 'cx-bg' }));
      for (const sink of Object.keys(C.wires)) {
        const p = outPin(C, C.wires[sink]), q = inPin(C, sink);
        if (!p || !q) continue;
        const d = curve(p, q);
        board.append(sv('path', { d, class: 'cx-wire' + (on(C.wires[sink]) ? ' is-on' : '') }), sv('path', { d, class: 'cx-wire-hit', 'data-wire': sink }));
      }
      const v = { A: row.a, B: row.b, Cin: row.c };
      for (const i of INS) board.append(sv('g', { class: 'cx-term' }, sv('text', { x: 1.5, y: i.y + 1.2, class: 'cx-label' + (v[i.id] ? ' is-on' : '') }, `${i.id}=${v[i.id]}`), sv('circle', { cx: PIN_IN, cy: i.y, r: 1.9, class: 'cx-pin' + (v[i.id] ? ' is-on' : '') + (E.pend === i.id ? ' is-pending' : ''), 'data-src': i.id })));
      for (const o of OUTS) {
        const got = R.out[o.id], want = o.id === 'S' ? row.want[0] : row.want[1];
        board.append(sv('g', { class: 'cx-term' }, sv('circle', { cx: PIN_OUT, cy: o.y, r: 1.9, class: 'cx-pin' + (C.wires[o.id] ? ' is-wired' : '') + (E.pend ? ' is-target' : ''), 'data-sink': o.id }),
          sv('text', { x: 98.5, y: o.y + 1.2, 'text-anchor': 'end', class: 'cx-label' + (got === null ? '' : got !== want ? ' is-bad' : got ? ' is-on' : '') }, `${o.id}=${got === null ? '?' : got}`)));
      }
      for (const g of C.gates) {
        const sel = E.sel === g.id, G = sv('g', { class: 'cx-gate' + (sel ? ' is-sel' : ''), transform: `translate(${g.x} ${g.y})`, 'data-gate': g.id });
        G.append(sv('rect', { x: -GW / 2, y: -GH / 2, width: GW, height: GH, rx: 2.2, class: 'cx-gbody' }), sv('text', { x: 0, y: 1.2, 'text-anchor': 'middle', class: 'cx-type' }, g.type));
        for (let k = 0; k < arity(g.type); k++) { const sk = `${g.id}.${k}`, [px, py] = inPin(C, sk); G.append(sv('circle', { cx: px - g.x, cy: py - g.y, r: 1.7, class: 'cx-pin' + (C.wires[sk] ? ' is-wired' : '') + (E.pend ? ' is-target' : ''), 'data-sink': sk })); }
        G.append(sv('circle', { cx: GW / 2, cy: 0, r: 1.9, class: 'cx-pin' + (on(g.id) ? ' is-on' : '') + (E.pend === g.id ? ' is-pending' : ''), 'data-src': g.id }));
        if (sel) G.append(sv('g', { class: 'cx-del', 'data-del': g.id, transform: `translate(${GW / 2} ${-GH / 2})` }, sv('circle', { r: 2.4 }), sv('text', { y: 1.1, 'text-anchor': 'middle' }, '×')));
        board.append(G);
      }
      if (E.drag && E.drag.kind === 'wire') { const s = outPin(C, E.drag.from); if (s) board.append(sv('path', { d: curve(s, E.drag.at), class: 'cx-wire is-drawing' })); }
      // the truth table: a row shows its inputs on the board when clicked
      clear(table);
      table.append(el('thead', null, el('tr', null, ...['A', 'B', 'Cin', 'S', 'Cout', ''].map((h) => el('th', { scope: 'col', text: h })))));
      const tb = el('tbody');
      K.rows.forEach((r, k) => {
        const pick = () => { E.row = k; render(); };
        const tr = el('tr', { class: [k === E.row ? 'is-sel' : '', r.ok ? 'is-ok' : r.s !== null && r.co !== null ? 'is-bad' : ''].join(' ').trim() || null, tabindex: '0', title: `Show A = ${r.a}, B = ${r.b}, Cin = ${r.c} on the wires`, onclick: pick, onkeydown: (e) => { if (e.key === 'Enter' || e.key === ' ') { e.preventDefault(); pick(); } } },
          el('td', { text: r.a }), el('td', { text: r.b }), el('td', { text: r.c }), el('td', { text: r.s === null ? '·' : r.s }), el('td', { text: r.co === null ? '·' : r.co }),
          el('td', { class: 'cx-verdict', text: r.ok ? '✓' : r.s === null || r.co === null ? '' : `✗ should be ${r.want[0]}, ${r.want[1]}` }));
        tb.append(tr);
      });
      table.append(tb);
      const tr = C.gates.reduce((p, g) => p + TR[g.type], 0);
      status.textContent = K.loop ? 'Some wires go round in a loop (a gate feeds back into itself). Remove one of them.'
        : !C.gates.length ? 'Start with a gate from the buttons above. A full adder takes 5 of them (the standard design), but there are other ways.'
          : !K.complete ? `Not finished: wire ${C.wires.S ? '' : 'S'}${!C.wires.S && !C.wires.Cout ? ' and ' : ''}${C.wires.Cout ? '' : 'Cout'}${C.wires.S && C.wires.Cout ? 'every input of the gates that lead to S and Cout' : ''}.${K.right ? ` ${K.right} of 8 rows are right so far.` : ''}`
            : K.right === 8 ? `All 8 rows right: it works! ${C.gates.length} gates, ${tr} transistors (the standard design: 5 gates, 50).`
              : `${K.right} of 8 rows right. The rows marked ✗ show what a full adder should give.`;
      status.classList.toggle('is-done', K.right === 8);
      use.checked = CX.on; use.disabled = !K.complete;
      undo.hidden = !E.prev;
    }
    E.render = render;
    render();
    const first = palette.querySelector('button');
    if (first) first.focus({ preventScroll: true });
  };
  CX.close = function () {
    const E = CX.ed;
    if (!E) return;
    E.root.remove();
    CX.ed = null;
    E.view.dirty = true; E.view.refreshTools(); E.view.refreshFocus();
  };

  /* ------------------------------------------------------------------ the visitor's full adder as a zoom level */
  function pyCode(C) {
    const names = { A: 'a', B: 'b', Cin: 'cin' }, order = [], seen = new Set();
    const visit = (id) => {
      if (names[id] || seen.has(id)) return;
      const g = C.gates.find((q) => q.id === id);
      if (!g) return;
      seen.add(id);
      for (let k = 0; k < arity(g.type); k++) { const s = C.wires[`${id}.${k}`]; if (s) visit(s); }
      order.push(g);
    };
    for (const o of OUTS) if (C.wires[o.id]) visit(C.wires[o.id]);
    const ref = (id) => (id ? names[id] || id : '0');
    const OPS = { AND: (x, y) => `${x} & ${y}`, OR: (x, y) => `${x} | ${y}`, XOR: (x, y) => `${x} ^ ${y}`, NAND: (x, y) => `1 - (${x} & ${y})`, NOR: (x, y) => `1 - (${x} | ${y})`, NOT: (x) => `1 - ${x}` };
    const body = order.map((g) => `    ${g.id} = ${OPS[g.type](...Array.from({ length: arity(g.type) }, (_, k) => ref(C.wires[`${g.id}.${k}`])))}`.padEnd(34) + `# ${g.type}`);
    return `def full_adder(a, b, cin):\n${body.join('\n') || '    pass'}\n    return ${ref(C.wires.S)}, ${ref(C.wires.Cout)}             # sum, carry\n\nfor a in (0, 1):\n    for b in (0, 1):\n        for cin in (0, 1):\n            right = ((a + b + cin) % 2, (a + b + cin) // 2)\n            got = full_adder(a, b, cin)\n            print(a, b, cin, got, 'ok' if got == right else f'should be {right}')`;
  }
  CX.adderNode = function (X, a, b, cin) {
    const D = NA.deep, T = NA.theme, C = CX.circuit, R = evaluate(C, a, b, cin), s = R.out.S, co = R.out.Cout;
    const want = D.math.faStd(a, b, cin), ok = s === want[0] && co === want[1], tr = C.gates.reduce((p, g) => p + TR[g.type], 0);
    const P = (x, y, r) => [r.x + (x / BW) * r.w, r.y + (y / BH) * r.h];
    return D.lazy({
      title: 'Your full adder', sub: `${a} + ${b} + ${cin} → carry ${co}, sum ${s}${ok ? '' : ' (wrong)'}`, tone: 's4', animate: true,
      count: `Inside: ${C.gates.length} logic gate${C.gates.length === 1 ? '' : 's'}, ${tr} transistors (the standard design: 5 gates, 50 transistors).`,
      summary: (ctx, r, S) => S.text(ctx, `${a}+${b}+${cin} → ${s}, c${co}${ok ? '' : ' ✗'}`, r.x + r.w / 2, r.y + r.h * 0.4, { size: clamp(r.w / 8, 6, 12), mono: true, align: 'center', maxW: r.w, color: ok ? T.c('ink') : T.c('neg') }),
      inner: (ctx, r, S) => {
        const u = S.u(r, 8, 14);
        for (const sink of Object.keys(C.wires)) {
          const p = outPin(C, C.wires[sink]), q = inPin(C, sink);
          if (!p || !q) continue;
          const A1 = P(p[0], p[1], r), B1 = P(q[0], q[1], r), mx = (A1[0] + B1[0]) / 2;
          D.draw.wire(ctx, [A1, [mx, A1[1]], [mx, B1[1]], B1], R.val.get(C.wires[sink]) === 1);
        }
        const v = { A: a, B: b, Cin: cin };
        for (const i of INS) { const [x, y] = P(0.5, i.y, r); S.text(ctx, `${i.id} = ${v[i.id]}`, x, y - u * 0.9, { size: u, mono: true, w: 650, color: v[i.id] ? T.c('pos') : T.c('ink-2') }); }
        for (const o of OUTS) {
          const [x, y] = P(99.5, o.y, r), got = R.out[o.id], w = o.id === 'S' ? want[0] : want[1];
          S.text(ctx, `${o.id} = ${got}${got === w ? '' : ` ✗ (should be ${w})`}`, x, y - u * 0.9, { size: u, mono: true, w: 700, align: 'right', color: got !== w ? T.c('neg') : got ? T.c('pos') : T.c('ink-2') });
        }
        S.text(ctx, ok ? 'the full adder you built: lit wires carry a 1' : `the full adder you built gives the wrong answer here: it should be carry ${want[1]}, sum ${want[0]}`, r.x, r.y + r.h * 0.98, { size: u * 0.8, color: ok ? T.c('muted') : T.c('neg'), maxW: r.w });
      },
      tools: (view) => [view.tool('Edit your adder', () => CX.open(view)), view.tool('Use the standard one', () => CX.setOn(false, view), { title: 'Put the standard full adder back in every adder' })],
      about: [
        `This is the full adder you built, now inside every adder in these views. Here it adds ${a} + ${b} + carry ${cin} and gives sum ${s} and carry ${co}${ok ? ', which is right.' : `, but a full adder should give sum ${want[0]} and carry ${want[1]}.`} ${CX.right === 8 ? 'It is right for all 8 input patterns, so every addition and multiplication above it comes out exactly as it should.' : `It is right for ${CX.right} of the 8 input patterns, so many of the sums above it come out wrong: go up to a multiplication or the neuron and compare.`}`,
        'Zoom into one of its gates to open it, down to the transistors. Edit your adder changes it; Use the standard one puts the standard design back everywhere.',
      ],
      code: pyCode(C),
    }, () => C.gates.map((g) => {
      const ins = Array.from({ length: arity(g.type) }, (_, k) => { const x = R.val.get(C.wires[`${g.id}.${k}`]); return x === 1 ? 1 : 0; });
      return Object.assign(D.gate(X, g.type, ins), { id: g.id, rect: [(g.x - GW / 2) / BW, (g.y - GH / 2) / BH, GW / BW, GH / BH], title: g.type, sub: `${ins.join(', ')} → ${FN[g.type](...ins)}` });
    }));
  };

  NA.circuit = CX;
})(window.NA);
