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
    const s = lab.training ? (lab.ffTarget !== null && lab.ffTarget !== undefined ? 'Pre-training' : 'Training') : lab.step === 0 ? 'Untrained' : 'Paused';
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
    const btns = defs.map((d) => el('button', { class: 'tab', type: 'button', role: 'tab', 'aria-selected': 'false', tabindex: '-1', dataset: { tab: d.id }, onclick: () => T.select(d.id) }, d.label));
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

  // A labelled control for the settings row (same look as the MLP settings).
  UI.field = (label, control, title) => el('label', { class: 'field', title: title || null }, el('span', { text: label }), control);
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

  /* Readouts in the control bar: defs [{ key, label, keep }] → { set(key, text) } */
  UI.readouts = function (root, defs) {
    const map = {};
    for (const d of defs) {
      const v = el('span', { class: 'ro-value', text: '—' });
      root.append(el('div', { class: 'readout' + (d.keep ? ' ro-keep' : ''), title: d.title || null }, el('span', { class: 'ro-label', text: d.label }), v));
      map[d.key] = v;
    }
    return { set(key, text) { const n = map[key]; if (n && n.textContent !== text) n.textContent = text; } };
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
      el('ol', { class: 'try-list' }, ...p.try.map((t) => el('li', null, el('p', { text: t.text }), el('button', { class: 'btn btn-try', type: 'button', onclick: () => onTry(t) }, t.label + ' →')))));
  };

  // A panel header with a title and optional tools, and a scrolling body.
  UI.panel = function (root, title, tools) {
    const head = el('div', { class: 'panel-head' }, el('span', { class: 'panel-title', text: title }), tools || null);
    const body = el('div', { class: 'panel-scroll' });
    root.append(head, body);
    return body;
  };

  UI.fmtSteps = (n) => fmtInt(n);

  NA.labHost = H;
  NA.labUI = UI;
})(window.NA);
