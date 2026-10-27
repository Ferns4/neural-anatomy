/* share.js — share links: a URL that opens the model on screen with its settings.
   The model stays in the hash (#overfit); its settings travel in ?s=, as base64url-encoded JSON. A plain network
   goes as its differences from the model's preset (data, layers, activations, training settings, lenses) plus the
   number of steps it trained, and is rebuilt and retrained when the link opens; weights are not in the link. A lab
   can add its own state through lab.shareState() and lab.loadShare(state): the model builder shares its setup. */
(function (NA) {
  'use strict';
  const { el, $ } = NA;
  const App = NA.app;
  const E = NA.engine;
  // Links made inside a frame (the page embedded in the artifact viewer) point at the public site instead.
  const PUBLIC = 'https://ferns4.github.io/neural-anatomy/';
  const MAX_STEPS = 20000;

  // ------------------------------------------------------------------ encoding
  function encode(obj) {
    const bytes = new TextEncoder().encode(JSON.stringify(obj));
    let s = '';
    for (const b of bytes) s += String.fromCharCode(b);
    return btoa(s).replace(/\+/g, '-').replace(/\//g, '_').replace(/=+$/, '');
  }
  function decode(str) {
    const s = atob(str.replace(/-/g, '+').replace(/_/g, '/'));
    return JSON.parse(new TextDecoder().decode(Uint8Array.from(s, (c) => c.charCodeAt(0))));
  }
  function base() {
    let framed = false;
    try { framed = window.top !== window.self; } catch (e) { framed = true; }
    return framed || !/^https?:$/.test(location.protocol) ? PUBLIC : location.origin + location.pathname;
  }

  // ------------------------------------------------------------------ the state of a plain network
  const same = (a, b) => JSON.stringify(a) === JSON.stringify(b);
  // the fields of cur that differ from base, or undefined when none do
  function diff(cur, base) {
    const o = {};
    let n = 0;
    for (const k of Object.keys(cur)) if (cur[k] !== undefined && !same(cur[k], base[k])) { o[k] = cur[k]; n++; }
    return n ? o : undefined;
  }
  function mlpState() {
    const p = App.preset;
    // per-neuron activations only for layers that differ from the network's default
    const acts = App.arch.hidden.map((n, h) => { const a = App.layerActs(h); return a.every((id) => id === App.arch.act) ? null : a; });
    const arch = { hidden: App.arch.hidden, act: App.arch.act, acts: acts.some(Boolean) ? acts : null };
    const s = {
      v: 1, p: p.id,
      d: diff(App.spec, Object.assign({ features: ['x1', 'x2'] }, p.data)),
      a: diff(arch, Object.assign({ acts: null }, p.arch)),
      h: diff(App.hyper, Object.assign({ loss: E.LOSS_FOR[App.net.out][0] }, p.hyper)), // the usual loss for its output goes unsaid
      l: diff(App.lenses, Object.assign({}, App.DEFAULT_LENSES, p.lenses)),
      o: diff(App.overlays, Object.assign({}, App.DEFAULT_OVERLAYS, p.overlays)),
      n: App.step,
    };
    if (App.selected) s.sel = [App.selected.l, App.selected.j];
    if (NA.inspector && NA.inspector.tab && NA.inspector.tab !== p.tab) s.t = NA.inspector.tab;
    for (const k of Object.keys(s)) if (s[k] === undefined) delete s[k];
    return s;
  }

  /* A link can come from anyone, so every field is checked against what the controls themselves allow; anything
     unknown or out of range is dropped and the preset's own value stays. */
  const isNum = (v, lo, hi) => typeof v === 'number' && isFinite(v) && v >= lo && v <= hi;
  const isInt = (v, lo, hi) => Number.isInteger(v) && v >= lo && v <= hi;
  const pick = (src, rules) => {
    const o = {};
    if (!src || typeof src !== 'object') return o;
    for (const k of Object.keys(rules)) if (src[k] !== undefined && rules[k](src[k])) o[k] = src[k];
    return o;
  };
  function cleanMlp(s) {
    const D = NA.data, p = NA.presets.byId[s.p];
    const d = pick(s.d, {
      dataset: (v) => typeof v === 'string' && Object.prototype.hasOwnProperty.call(D.DATASETS, v),
      noise: (v) => isNum(v, 0, 2), n: (v) => isInt(v, 20, 5000), trainRatio: (v) => isNum(v, 0.1, 0.95), seed: (v) => isInt(v, 1, 1e6),
      features: (v) => Array.isArray(v) && v.length > 0 && v.length <= 12 && v.every((f) => typeof f === 'string' && Object.prototype.hasOwnProperty.call(D.FEATURE_BY_ID, f)),
    });
    const kind = D.DATASETS[d.dataset || p.data.dataset].kind;
    if (kind !== 'plane') delete d.features;
    const maxN = kind === 'image' ? 48 : 32;
    const isAct = (v) => typeof v === 'string' && Object.prototype.hasOwnProperty.call(E.ACTIVATIONS, v);
    const a = pick(s.a, { hidden: (v) => Array.isArray(v) && v.length <= 6 && v.every((n) => isInt(n, 1, maxN)), act: isAct });
    const hidden = a.hidden || p.arch.hidden;
    if (s.a && Array.isArray(s.a.acts) && s.a.acts.length === hidden.length && s.a.acts.every((l, h) => l === null || (Array.isArray(l) && l.length === hidden[h] && l.every(isAct)))) a.acts = s.a.acts;
    const h = pick(s.h, {
      lr: (v) => isNum(v, 1e-6, 10), optimizer: (v) => ['sgd', 'momentum', 'adam'].includes(v), batch: (v) => isInt(v, 1, 1024),
      reg: (v) => ['none', 'l1', 'l2'].includes(v), regRate: (v) => isNum(v, 0, 1), dropout: (v) => isNum(v, 0, 0.9),
      init: (v) => ['auto', 'small', 'large'].includes(v), seed: (v) => isInt(v, 1, 999), loss: (v) => typeof v === 'string' && Object.prototype.hasOwnProperty.call(E.LOSSES, v),
    });
    const lensRules = {}, overlayRules = {};
    for (const k of Object.keys(App.DEFAULT_LENSES)) lensRules[k] = typeof App.DEFAULT_LENSES[k] === 'boolean' ? (v) => typeof v === 'boolean' : (v) => typeof v === 'string' && /^[a-z]+$/.test(v);
    lensRules.edges = (v) => ['weight', 'signal', 'gradient'].includes(v);
    lensRules.nodes = (v) => ['activation', 'preact', 'gradient'].includes(v);
    for (const k of Object.keys(App.DEFAULT_OVERLAYS)) overlayRules[k] = (v) => typeof v === 'boolean';
    return {
      d, a, h, l: pick(s.l, lensRules), o: pick(s.o, overlayRules),
      n: isInt(s.n, 0, 1e7) ? s.n : p.pretrain,
      sel: Array.isArray(s.sel) && isInt(s.sel[0], 0, 8) && isInt(s.sel[1], 0, 1000) ? { l: s.sel[0], j: s.sel[1] } : null,
      t: typeof s.t === 'string' && /^[a-z]+$/.test(s.t) ? s.t : null,
    };
  }

  // ------------------------------------------------------------------ opening a link
  // The shared state in this page's address (?s=…), or null. Removed from the address once read.
  function readLink() {
    const m = /[?&]s=([A-Za-z0-9_-]+)/.exec(location.search);
    if (!m) return null;
    try { history.replaceState(null, '', location.pathname + location.hash); } catch (e) { /* not allowed in some frames */ }
    try {
      const s = decode(m[1]);
      return s && s.v === 1 && typeof s.p === 'string' && NA.presets.byId[s.p] ? s : null;
    } catch (e) { return null; }
  }

  // Opens a shared state; returns false when there is none, so start-up loads the usual model.
  function open(s) {
    const p = NA.presets.byId[s.p];
    if (p.lab) {
      App.loadPreset(p.id);
      const lab = NA.labHost.lab;
      if (s.x !== undefined && lab && lab.loadShare) {
        if (lab.loadShare(s.x)) App.toast(`Opened a shared ${p.name} setup. ${lab.shareNote || ''}`.trim());
        else App.toast(`This link’s ${p.name} setup could not be read, so the model opened as it was.`);
      }
      return true;
    }
    const c = cleanMlp(s), steps = Math.min(c.n, MAX_STEPS);
    App.loadPreset(p.id, {
      setup(A) {
        Object.assign(A.spec, c.d);
        if (NA.data.DATASETS[A.spec.dataset].kind === 'plane' && !(A.spec.features && A.spec.features.length)) A.spec.features = ['x1', 'x2'];
        if (c.a.hidden) A.arch.hidden = c.a.hidden.slice();
        if (c.a.act) { A.arch.act = c.a.act; A.arch.acts = null; }
        if (c.a.acts) A.arch.acts = c.a.acts.map((l) => (l ? l.slice() : null));
        Object.assign(A.hyper, c.h);
        Object.assign(A.lenses, c.l);
        Object.assign(A.overlays, c.o);
      },
      steps,
    });
    if (c.sel && App.validNeuron(c.sel)) App.select(c.sel);
    if (c.t && NA.inspector && document.querySelector(`[data-tab="${c.t}"]`)) NA.inspector.setTab(c.t);
    const changed = ['d', 'a', 'h'].some((k) => Object.keys(c[k]).length);
    App.toast(`Opened a shared link: ${p.name}${changed ? ' with its own settings' : ''}${steps ? `, retraining ${NA.fmtInt(steps)} steps${steps < c.n ? ` (of the ${NA.fmtInt(c.n)} it had trained)` : ''}` : ''}.`);
    return true;
  }

  // ------------------------------------------------------------------ making a link
  function link() {
    const p = App.preset;
    let s = null;
    if (App.mode === 'mlp') s = mlpState();
    else {
      const lab = NA.labHost.lab;
      if (lab && lab.shareState) s = { v: 1, p: p.id, x: lab.shareState() };
    }
    return base() + (s ? '?s=' + encode(s) : '') + '#' + p.id;
  }
  function describe() {
    const p = App.preset;
    if (App.mode === 'mlp') return `Opens ${p.name} with your data, layers and training settings, and retrains it for the same ${NA.fmtInt(App.step)} steps from the same seed. The weights are not in the link: if you changed a setting partway through training, the retrained network can differ a little.`;
    const lab = NA.labHost.lab;
    if (lab && lab.shareState) return (typeof lab.shareAbout === 'function' ? lab.shareAbout() : lab.shareAbout) || `Opens ${p.name} with your setup.`;
    return `Opens the ${p.name} lab.`;
  }

  const SH = {};
  function build() {
    SH.url = el('input', { class: 'share-url', type: 'text', readonly: '', 'aria-label': 'Link to share', spellcheck: 'false' });
    SH.copyBtn = el('button', { class: 'btn btn-primary', type: 'button', onclick: copy }, 'Copy link');
    SH.nativeBtn = navigator.share ? el('button', { class: 'btn', type: 'button', onclick: () => navigator.share({ title: 'Neural Anatomy: ' + App.preset.name, url: SH.url.value }).catch(() => {}) }, 'Share…') : null;
    SH.about = el('p', { class: 'sheet-sub' });
    SH.note = el('p', { class: 'share-note', 'aria-live': 'polite' });
    SH.root = el('div', { class: 'modal', id: 'shareDlg', role: 'dialog', 'aria-modal': 'true', 'aria-labelledby': 'shareTitle', hidden: '' },
      el('div', { class: 'sheet share-sheet' },
        el('div', { class: 'sheet-head' },
          el('div', null, el('h2', { class: 'sheet-title', id: 'shareTitle', text: 'Share this model' }), SH.about),
          el('button', { class: 'icon-btn', type: 'button', 'data-close': '', 'aria-label': 'Close' }, '×')),
        el('div', { class: 'share-row' }, SH.url, el('div', { class: 'share-actions' }, SH.copyBtn, SH.nativeBtn)),
        SH.note));
    SH.root.addEventListener('click', (e) => { if (e.target === SH.root || e.target.closest('[data-close]')) close(); });
    SH.root.addEventListener('keydown', (e) => { if (e.key === 'Escape') { e.stopPropagation(); close(); } });
    SH.url.addEventListener('focus', () => SH.url.select());
    document.body.append(SH.root);
  }
  function show() {
    if (!SH.root) build();
    if (App.training) App.pause();
    SH.url.value = link();
    SH.about.textContent = describe();
    SH.note.textContent = '';
    SH.copyBtn.textContent = 'Copy link';
    SH.root.hidden = false;
    SH.copyBtn.focus();
  }
  function close() {
    SH.root.hidden = true;
    const b = $('#shareBtn');
    if (b) b.focus();
  }
  function copy() {
    const done = () => { SH.copyBtn.textContent = 'Copied'; SH.note.textContent = 'The link is on your clipboard.'; };
    const fallback = () => {
      SH.url.focus(); SH.url.select();
      let ok = false;
      try { ok = document.execCommand('copy'); } catch (e) { ok = false; }
      if (ok) done(); else SH.note.textContent = 'Copying is blocked here: the link is selected, so press Ctrl+C (⌘C on a Mac) to copy it.';
    };
    if (navigator.clipboard && navigator.clipboard.writeText) navigator.clipboard.writeText(SH.url.value).then(done, fallback);
    else fallback();
  }

  NA.share = { link, readLink, open, encode, decode, show, close, isOpen: () => !!SH.root && !SH.root.hidden };
  if (typeof document !== 'undefined') {
    const wire = () => { const b = $('#shareBtn'); if (b) b.addEventListener('click', show); };
    if (document.readyState === 'loading') document.addEventListener('DOMContentLoaded', wire); else wire();
  }
})(window.NA);
