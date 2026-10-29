/* lab.js — hosts the models that are not plain MLPs (transformer, PINN, diffusion). A lab owns the three
   panels, its settings and readouts in the control bar, and its own training loop; the app switches to it
   when a preset names one. Also shared building blocks for lab interfaces. */
(function (NA) {
  'use strict';
  const { el, $, clear, fmtInt } = NA;
  const App = NA.app;

  const LABS = {};
  const H = { lab: null, preset: null };

  /* A lab: { id, screens: {network, data, insp}, activate(preset, ui), deactivate(), frame(now),
     training, ffTarget, step, play(), pause(), toggle(), stepOnce(), reset(), fastForward(n),
     archText(), glyph(ctx, w, h), runTest?(), key?(e) } */
  H.register = (lab) => { LABS[lab.id] = lab; };
  H.get = (id) => LABS[id];

  H.init = function () {
    H.ui = { left: $('#labLeft'), center: $('#labCenter'), right: $('#labRight'), readouts: $('#labReadouts'), hyper: $('#labHyperBody') };
    H.stateEl = $('#labState');
    App.register(H);
  };

  H.enter = function (preset) {
    const lab = LABS[preset.lab];
    if (H.lab) H.leave();
    H.lab = lab; H.preset = preset;
    document.body.dataset.mode = 'lab';
    document.body.dataset.lab = lab.id;
    Object.values(H.ui).forEach(clear);
    lab.activate(preset, H.ui);
    H.lastState = '';
  };
  H.leave = function () {
    if (!H.lab) return;
    if (H.lab.training) H.lab.pause();
    H.lab.deactivate();
    H.lab = null; H.preset = null;
  };
  H.exit = function () {
    H.leave();
    delete document.body.dataset.mode;
    delete document.body.dataset.lab;
  };

  H.frame = function (now) {
    const lab = H.lab;
    if (!lab) return;
    lab.frame(now);
    // labs whose transport plays something other than training (a live scene, a stream) name their own states
    const s = lab.stateText ? lab.stateText() : lab.training ? (lab.ffTarget !== null && lab.ffTarget !== undefined ? 'Pre-training' : 'Training') : lab.step === 0 ? 'Untrained' : 'Paused';
    if (s !== H.lastState) {
      H.lastState = s;
      H.stateEl.textContent = s;
      H.stateEl.className = 'state-pill ' + (lab.training ? 'pill-good' : 'pill-quiet');
    }
  };

  // ------------------------------------------------------------------ shared training-loop helpers
  /* Mixes into a lab: play / pause / toggle / fastForward around lab.trainChunk(budgetMs), which trains for
     about that long and returns the number of steps taken. */
  H.transport = function (lab) {
    Object.assign(lab, {
      training: false, ffTarget: null, step: 0,
      play() { this.training = true; App.emit('training'); },
      pause() { const was = this.training; this.training = false; this.ffTarget = null; if (was && this.onPause) this.onPause(); App.emit('training'); },
      toggle() { if (this.training) this.pause(); else this.play(); },
      fastForward(n) { if (!n) return; this.ffTarget = this.step + n; this.training = true; App.emit('training'); },
      // Run training inside an animation frame: a time budget, stopping at a fast-forward target.
      runFrame() {
        if (!this.training) return false;
        const ff = this.ffTarget !== null;
        this.trainChunk(ff ? 26 : this.budget || 12, ff ? this.ffTarget - this.step : Infinity);
        if (ff && this.step >= this.ffTarget) this.pause();
        return true;
      },
    });
    return lab;
  };

  /* Loads a script once, on demand (large weight files only load when their lab opens). Resolves when it ran.
     Uses the same version stamp as the page's own scripts, so a local preview never serves a stale copy. */
  const loading = {};
  H.loadScript = function (src) {
    if (!loading[src]) {
      const own = document.querySelector('script[src*="js/lab.js"]'), stamp = own ? (own.getAttribute('src').split('?')[1] || '') : '';
      loading[src] = new Promise((resolve, reject) => {
        const s = el('script', { src: src + (stamp ? '?' + stamp : '') });
        s.onload = () => resolve(); s.onerror = () => { delete loading[src]; reject(new Error('Could not load ' + src)); };
        document.head.append(s);
      });
    }
    return loading[src];
  };

  // ------------------------------------------------------------------ UI building blocks
  const UI = {};
  UI.section = (title, ...kids) => el('section', { class: 'insp-sec' }, title ? el('h3', { class: 'sec-title', text: title }) : null, ...kids);
  UI.note = (text, cls) => el('p', { class: 'sec-note' + (cls ? ' ' + cls : ''), text });
  UI.tile = (label, value, sub) => el('div', { class: 'tile' }, el('div', { class: 'tile-k', text: label }), el('div', { class: 'tile-v', text: value }), sub ? el('div', { class: 'tile-s', text: sub }) : null);
  UI.pill = (level, text) => el('span', { class: 'pill pill-' + level }, el('i', { class: 'pill-ico', 'aria-hidden': 'true' }), text);

  /* Tab bar + body in the style of the inspector. defs: [{ id, label }]. onSelect(id, body) renders a tab. */
  UI.tabs = function (root, defs, onSelect, initial) {
    const bar = el('div', { class: 'insp-tabs', role: 'tablist' }), body = el('div', { class: 'insp-body', role: 'tabpanel' });
    const T = { current: null, body, bar };
    const btns = defs.map((d) => el('button', { class: 'tab', type: 'button', role: 'tab', 'aria-selected': 'false', tabindex: '-1', dataset: { tab: d.id }, onclick: () => T.select(d.id) }, NA.shortLabel(d.label, d.short)));
    bar.append(...btns);
    bar.addEventListener('keydown', (e) => {
      if (e.key !== 'ArrowRight' && e.key !== 'ArrowLeft') return;
      const i = defs.findIndex((d) => d.id === T.current), next = defs[(i + (e.key === 'ArrowRight' ? 1 : defs.length - 1)) % defs.length];
      T.select(next.id); btns[defs.indexOf(next)].focus();
    });
    T.select = (id) => {
      T.current = id;
      btns.forEach((b) => { const on = b.dataset.tab === id; b.setAttribute('aria-selected', String(on)); b.tabIndex = on ? 0 : -1; });
      clear(body); body.scrollTop = 0;
      onSelect(id, body);
    };
    root.append(bar, body);
    T.select(initial || defs[0].id);
    return T;
  };

  // A labelled control for the settings row (same look as the MLP settings). The label is set in capitals, so its
  // Greek letters keep their case: a capital ε reads as E, α as A, η as H.
  const keepGreek = (text) => String(text).split(/([Ͱ-Ͽ]+)/).filter(Boolean).map((s) => (/[Ͱ-Ͽ]/.test(s) ? el('span', { class: 'keep-case', text: s }) : s));
  UI.field = (label, control, title) => el('label', { class: 'field', title: title || null }, el('span', null, keepGreek(label)), control);
  UI.select = function (options, value, onChange, attrs = {}) {
    const s = el('select', attrs, ...options.map((o) => el('option', { value: o.value, selected: String(o.value) === String(value) }, o.label)));
    s.addEventListener('change', () => onChange(s.value));
    return s;
  };
  /* Range slider with a live value: { label, min, max, step, value, format, onInput, onChange } */
  UI.slider = function (o) {
    const id = 'sl-' + Math.random().toString(36).slice(2, 8);
    const out = el('output', { class: 'mono', for: id, text: (o.format || String)(o.value) });
    const input = el('input', { type: 'range', id, min: o.min, max: o.max, step: o.step, value: o.value });
    input.addEventListener('input', () => { out.textContent = (o.format || String)(+input.value); if (o.onInput) o.onInput(+input.value); });
    input.addEventListener('change', () => { if (o.onChange) o.onChange(+input.value); });
    const wrap = el('label', { class: 'slider', for: id }, el('span', { class: 'sl-top' }, el('span', { text: o.label }), out), input);
    wrap.input = input; wrap.out = out;
    wrap.set = (v) => { input.value = v; out.textContent = (o.format || String)(v); };
    return wrap;
  };
  /* Segmented buttons: options [{ value, label, title }] */
  UI.seg = function (options, value, onChange, label) {
    const seg = el('div', { class: 'seg', role: 'group', 'aria-label': label || null });
    const set = (v) => NA.$$('button', seg).forEach((b) => b.setAttribute('aria-pressed', String(b.dataset.v === String(v))));
    for (const o of options) seg.appendChild(el('button', { type: 'button', title: o.title || null, dataset: { v: String(o.value) }, onclick: () => { set(o.value); onChange(o.value); } }, o.label));
    set(value);
    seg.set = set;
    return seg;
  };
  UI.chip = function (label, pressed, onToggle, title) {
    const b = el('button', { class: 'chip', type: 'button', 'aria-pressed': String(!!pressed), title: title || null }, label);
    b.addEventListener('click', () => { const on = b.getAttribute('aria-pressed') !== 'true'; b.setAttribute('aria-pressed', String(on)); onToggle(on); });
    return b;
  };

  /* Readouts in the control bar: defs [{ key, label, short, keep }] → { set(key, text) }. A phone's bar shows only the
     kept ones, with their short label where there is one (each gets about 50 px on a 360 px screen); --lb is that
     label's width there (7.9 px a capital), so a readout makes room for its label before its neighbours grow. */
  UI.readouts = function (root, defs) {
    const map = {}, names = {};
    const fill = (k, label, short) => {
      k.replaceChildren(...NA.shortLabel(label, short));
      k.dataset.label = label + '|' + (short || '');
      k.parentNode.style.setProperty('--lb', `${(short || label).length * 7.9}px`);
    };
    for (const d of defs) {
      const v = el('span', { class: 'ro-value', text: '—' }), k = el('span', { class: 'ro-label' });
      root.append(el('div', { class: 'readout' + (d.keep ? ' ro-keep' : ''), title: d.title || null }, k, v));
      fill(k, d.label, d.short);
      map[d.key] = v; names[d.key] = { k, short: d.short };
    }
    const count = (s) => (/^-?\d{1,3}(,\d{3})+$/.test(s) ? NA.fmtCount(+s.replace(/,/g, '')) : s);
    return {
      set(key, text) { const n = map[key]; text = count(String(text)); if (n && n.textContent !== text) n.textContent = text; },
      // a readout that means something else for this setup (the builder's return, for learning from rewards)
      label(key, text, short) { const n = names[key]; if (!n) return; if (short !== undefined) n.short = short; if (n.k.dataset.label !== text + '|' + (n.short || '')) fill(n.k, text, n.short); },
    };
  };

  // The preset's notes and one-click experiments, in the style of the MLP Insights tab.
  UI.modelCard = function (p, onTry) {
    return el('article', { class: 'model-card' },
      el('div', { class: 'eyebrow', text: `${p.group} · preset` }),
      el('h2', { class: 'mc-title', text: p.name }),
      el('p', { class: 'mc-tag', text: p.tagline }),
      el('p', { class: 'mc-about', text: p.about }),
      el('h3', { class: 'sec-title', text: 'What to look for' }),
      el('ul', { class: 'look-list' }, ...p.look.map((t) => el('li', { text: t }))),
      el('h3', { class: 'sec-title', text: 'Experiments' }),
      UI.tryList(p.try, onTry));
  };
  /* A model's experiments: what to try and a button that does it. tries: preset entries { text, label, ... };
     onTry(t) runs one. Shared by the labs' cards and the main app's inspector. */
  UI.tryList = function (tries, onTry) {
    return el('ol', { class: 'try-list' }, ...tries.map((t) => el('li', null, el('p', { text: t.text }), t.predict ? predictBox(t, onTry) : null,
      el('button', { class: 'btn btn-try', type: 'button', onclick: () => onTry(t) }, (t.predict ? 'Just run it: ' : '') + t.label + ' →'))));
  };
  /* Predict, then see: an experiment with predict: { q, options, answer, why } asks first, runs the experiment
     on the pick and then reveals the answer. onTry(t) may return a promise of a line with the measured result,
     shown with it. The state lives here rather than in the page, because panels are rebuilt while the
     experiment trains; every box drawn for an experiment shows its latest state. */
  const PREDICT = new Map(), BOXES = new Map();
  function predictBox(t, onTry) {
    const box = el('div', { class: 'predict', role: 'group', 'aria-label': 'Predict first: ' + t.predict.q });
    if (!BOXES.has(t)) BOXES.set(t, new Set());
    BOXES.get(t).add(box);
    box._draw = () => drawPredict(box, t, onTry);
    box._draw();
    return box;
  }
  function drawPredict(box, t, onTry) {
    const P = t.predict, st = PREDICT.get(t);
    const choose = (i) => {
      const s = { pick: i, done: false, measured: '' };
      PREDICT.set(t, s);
      redraw(t);
      Promise.resolve(onTry(t)).then((m) => { s.measured = typeof m === 'string' ? m : ''; }, () => {}).then(() => {
        s.done = true;
        redraw(t);
        if (NA.a11y) NA.a11y.announce((i === P.answer ? 'Right. ' : 'Not quite. ') + P.why + (s.measured ? ' ' + s.measured : ''));
      });
    };
    const opts = P.options.map((o, i) => {
      const b = el('button', { class: 'predict-opt', type: 'button', 'aria-pressed': String(!!st && st.pick === i), disabled: !!st, onclick: () => choose(i) }, o);
      if (st && st.done && i === P.answer) b.classList.add('is-answer');
      else if (st && st.done && i === st.pick) b.classList.add('is-wrong');
      return b;
    });
    const out = !st ? null : !st.done ? el('p', { class: 'predict-result' }, 'Running the experiment: the answer appears when it finishes.')
      : el('p', { class: 'predict-result' }, el('strong', { text: st.pick === P.answer ? 'Right. ' : 'Not quite. ' }), P.why, st.measured ? el('span', { class: 'predict-measured', text: ' ' + st.measured }) : null, ' ',
        el('button', { class: 'link-btn', type: 'button', onclick: () => { PREDICT.delete(t); redraw(t); } }, 'Ask again'));
    NA.clear(box).append(...[el('p', { class: 'predict-q' }, el('span', { class: 'predict-tag', text: 'Predict first' }), P.q), el('div', { class: 'predict-opts' }, ...opts), out].filter(Boolean));
  }
  function redraw(t) {
    const set = BOXES.get(t);
    if (set) for (const b of [...set]) { if (b.isConnected) b._draw(); else set.delete(b); }
  }

  // A panel header with a title and optional tools, and a scrolling body.
  UI.panel = function (root, title, tools) {
    const head = el('div', { class: 'panel-head' }, el('span', { class: 'panel-title', text: title }), tools || null);
    const body = el('div', { class: 'panel-scroll' });
    root.append(head, body);
    return body;
  };

  UI.fmtSteps = (n) => fmtInt(n);

  // ------------------------------------------------------------------ explanation and code panel
  const PY_KW = new Set('def class return for in if elif else import from as with lambda None True False and or not while break continue pass yield raise try except finally assert is del global'.split(' '));
  const PY_BUILTIN = new Set('self super range len print float int min max sum zip enumerate list dict tuple abs round isinstance'.split(' '));
  // Python source → highlighted DOM nodes (no innerHTML: every token is a text node)
  UI.pyHighlight = function (src) {
    const out = [], re = /(#[^\n]*)|('''[\s\S]*?'''|"""[\s\S]*?"""|'(?:[^'\\\n]|\\.)*'|"(?:[^"\\\n]|\\.)*")|(@[A-Za-z_][\w.]*)|(\b\d+(?:\.\d+)?(?:e-?\d+)?\b)|([A-Za-z_]\w*)|(\s+)|([\s\S])/g;
    let m, lastWord = '';
    while ((m = re.exec(src))) {
      const [tok, com, str, deco, num, word] = m;
      let cls = null;
      if (com) cls = 'py-com'; else if (str) cls = 'py-str'; else if (deco) cls = 'py-deco'; else if (num) cls = 'py-num';
      else if (word) {
        if (lastWord === 'def' || lastWord === 'class') cls = 'py-def';
        else if (PY_KW.has(tok)) cls = 'py-kw';
        else if (PY_BUILTIN.has(tok)) cls = 'py-bi';
        else if (/^\s*\(/.test(src.slice(re.lastIndex, re.lastIndex + 4))) cls = 'py-fn';
        lastWord = tok;
      }
      out.push(cls ? el('span', { class: cls, text: tok }) : document.createTextNode(tok));
    }
    return out;
  };
  UI.codeBox = function (src, label) {
    const copy = el('button', { class: 'btn btn-quiet code-copy', type: 'button' }, 'Copy');
    copy.addEventListener('click', async () => {
      try { await navigator.clipboard.writeText(src); copy.textContent = 'Copied'; } catch (e) { copy.textContent = 'Select the text to copy'; }
      setTimeout(() => { copy.textContent = 'Copy'; }, 1800);
    });
    return el('div', { class: 'code-box' }, el('div', { class: 'code-head' }, el('span', { text: label || 'Python · PyTorch' }), copy), el('pre', { class: 'code', tabindex: '0' }, el('code', null, ...UI.pyHighlight(src))));
  };
  /* A panel under a lab's picture: what the current view or block is, in depth, and the same thing written as
     Python. set({ kicker, title, about: [paragraphs or nodes], code }) */
  UI.infoPanel = function (root, opts = {}) {
    const P = { tab: 'about', open: opts.open !== false, content: { title: '', about: [] } };
    const kicker = el('span', { class: 'ip-kicker' }), title = el('span', { class: 'ip-title' });
    const toggleBtn = el('button', { class: 'icon-btn ip-toggle', type: 'button', 'aria-label': 'Show or hide the explanation' }, '▾');
    const tabs = UI.seg([{ value: 'about', label: 'What it does' }, { value: 'code', label: 'Python' }], P.tab, (v) => { P.tab = v; toggle(true); render(); }, 'Explanation or code');
    const body = el('div', { class: 'ip-body' });
    const wrap = el('section', { class: 'info-panel', 'aria-label': 'Explanation of what you are looking at' }, el('div', { class: 'ip-head' }, el('div', { class: 'ip-titles' }, kicker, title), tabs, toggleBtn), body);
    toggleBtn.addEventListener('click', () => toggle(!P.open));
    root.append(wrap);
    function toggle(on) { P.open = on; wrap.classList.toggle('is-open', on); toggleBtn.setAttribute('aria-expanded', String(on)); toggleBtn.textContent = on ? '▾' : '▴'; }
    function render() {
      const scroll = body.scrollTop;
      clear(body);
      if (P.tab === 'about') body.append(...(P.content.about || []).map((t) => (typeof t === 'string' ? el('p', { text: t }) : t)));
      else body.append(P.content.code ? UI.codeBox(P.content.code, P.content.codeLabel) : el('p', { text: 'No code for this view.' }));
      body.scrollTop = P.sameAsBefore ? scroll : 0;
    }
    P.set = (content) => {
      P.sameAsBefore = content.title === P.content.title && content.kicker === P.content.kicker;
      P.content = content;
      kicker.textContent = content.kicker || ''; title.textContent = content.title || '';
      render();
    };
    P.toggle = toggle;
    toggle(P.open);
    wrap.addEventListener('na-open', () => toggle(true)); // (holding a finger on a part of a zoom view shows it here)
    return P;
  };

  NA.labHost = H;
  NA.labUI = UI;
})(window.NA);
