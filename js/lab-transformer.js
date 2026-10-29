/* lab-transformer.js — the Tiny Transformer lab: a prompt you can edit and continue, an anatomy diagram of
   every token's residual stream with attention arcs and the logit lens, and inspector tabs for attention
   heads (with knock-outs), embeddings, the logit lens with attribution, and training. */
(function (NA) {
  'use strict';
  const { el, clear, clamp, fmt, fmtPct, fmtInt, theme } = NA;
  const rr = (...a) => NA.rr(...a); // defined by network-view.js, which loads later
  const App = NA.app, H = NA.labHost, UI = NA.labUI, D = NA.lmData, TF = NA.tf, C = NA.charts;

  /* A colour per word category, for marks only (dots, bars, edges): several of these are too light to read as
     text, so words themselves are written in ink. The eight series colours go to the content words; the three
     kinds of function word get greys of different depth: the article mid-grey, prepositions ink, markers muted. */
  const CAT_TOKEN = { special: 'muted', name: 's1', drink: 's3', verb: 's7', det: 'ink-2', adj: 's4', sg: 's2', pl: 's5', prep: 'ink', vsg: 's6', vpl: 's8' };
  const catOf = (id) => D.CATEGORIES[D.CAT_OF[id]].id;
  const tokColor = (id, a = 1) => theme.c(CAT_TOKEN[catOf(id)], a);
  // the validation stories every score on the page is measured on (readouts, Overview, Training, knock-outs)
  const VAL_N = 100;
  const word = (id) => (id === D.W.BOS ? '⟨bos⟩' : D.VOCAB[id]);
  const STAGE_NAMES = (L) => ['Embedding', ...Array.from({ length: L }, (_, l) => [`Attention ${l + 1}`, `MLP ${l + 1}`]).flat()];
  const FOOT_BLOCKS = (L) => `${L} block${L > 1 ? 's' : ''} · each row is one token; attention moves information between rows, MLPs work within a row`;

  const TL = H.transport({
    id: 'transformer', screens: { network: 'Model', data: 'Prompt', insp: 'Inspect' },
    budget: 12, lr: 1e-3, batch: 16, temp: 0.7, variant: 'twoLayer',
    text: D.PROMPTS[0].text, sel: null, stage: null, headFilter: 'all', lens: true, version: 0,
  });

  // ------------------------------------------------------------------ model and data
  TL.load = function (variant) {
    const W = NA.lmWeights[variant === 'scratch' ? 'twoLayer' : variant];
    this.variant = variant;
    this.model = new TF.TinyGPT(Object.assign({}, W.cfg, { seed: variant === 'scratch' ? 7 : W.cfg.seed }));
    this.history = [];
    if (variant !== 'scratch') {
      this.model.setParams(TF.decodeWeights(W.weights));
      this.step = W.train.steps;
      // the curve recorded while this model was trained offline
      this.history = W.curve.map(([step, train, val, lookup, agree]) => ({ step, train, val, lookup, agree }));
      this.pretrained = true;
    } else { this.step = 0; this.pretrained = false; }
    this.model.resetOptimizer();
    this.gen = new D.StoryGen(1234 + (variant === 'scratch' ? 1 : 0));
    this.val = D.makeSet(2002, 160);
    this.test = D.makeSet(3003, 400);
    this.testResult = null; this.analysis = this.lastAnalysis = null; this.runLoss = null;
    this.bumpVersion();
    if (variant === 'scratch') this.recordPoint();
    else {
      // the shipped curve was scored offline on 300 stories; what the page shows now is scored on the same
      // VAL_N validation stories as the Overview and the knock-outs, so the tabs agree with each other
      const e = TF.evaluate(this.model, this.val.slice(0, VAL_N), D.W);
      this.current = Object.assign({}, this.history[this.history.length - 1], { val: e.loss, lookup: e.lookup, agree: e.agree });
    }
    // the picker and the key under the picture follow the model, however it was loaded (a button, an experiment)
    if (this.footBlocks) this.footBlocks.textContent = FOOT_BLOCKS(this.model.cfg.L);
    const pick = document.getElementById('lmVariant'); if (pick) pick.value = variant;
    const [hl, hh] = this.headFilter === 'all' ? [0, 0] : this.headFilter.split('.').map(Number);
    if (hl >= this.model.cfg.L || hh >= this.model.cfg.H) this.headFilter = 'all';
    if (this.headSeg) this.renderHeadSeg();
    this.leftDirty = true;
    App.emit('labarch');
  };
  TL.bumpVersion = function () { this.version++; this.trCache = null; this.dirty = true; this.inspDirty = true; this.archStale = true; };

  TL.recordPoint = function () {
    const e = TF.evaluate(this.model, this.val.slice(0, VAL_N), D.W);
    const p = { step: this.step, train: this.runLoss ?? e.loss, val: e.loss, lookup: e.lookup, agree: e.agree };
    this.history.push(p);
    if (this.history.length > 400) this.history = this.history.filter((q, i) => i % 2 === 0 || i === this.history.length - 1);
    this.current = p;
  };

  TL.trainChunk = function (budget, maxSteps) {
    const t0 = performance.now();
    let k = 0;
    do {
      const batch = [];
      for (let b = 0; b < this.batch; b++) batch.push(this.gen.story().ids);
      const r = this.model.trainStep(batch, { lr: this.lr, warmup: 100 });
      if (!isFinite(r.loss)) { this.pause(); NA.ui.toast.show('Training diverged. Lower the learning rate and train from scratch.'); break; }
      this.runLoss = this.runLoss === null ? r.loss : 0.95 * this.runLoss + 0.05 * r.loss;
      this.step++; k++;
      if (this.step % 50 === 0) this.recordPoint();
    } while (performance.now() - t0 < budget && k < maxSteps);
    this.analysis = null;
    this.bumpVersion();
    return k;
  };
  TL.onPause = function () { this.recordPoint(); this.bumpVersion(); };
  TL.stepOnce = function () { if (this.training) this.pause(); this.trainChunk(0, 1); this.recordPoint(); };
  TL.reset = function () { if (this.training) this.pause(); this.load('scratch'); NA.ui.toast.show('New random weights. Press Train: grammar appears within seconds, lookups after about 2,000 steps.'); };

  TL.ids = function () {
    const t = D.tokenize(this.text);
    const ids = [D.W.BOS, ...t.ids.filter((i) => i !== D.W.BOS)].slice(0, this.model.cfg.T);
    return { ids, unknown: t.unknown };
  };
  TL.trace = function () {
    const { ids } = this.ids(), key = this.version + ':' + ids.join(',');
    if (!this.trCache || this.trCache.key !== key) this.trCache = { key, tr: this.model.trace(ids) };
    return this.trCache.tr;
  };
  TL.selPos = function (tr) { return this.sel === null || this.sel >= tr.n ? tr.n - 1 : this.sel; };

  // Ablation analysis: what each head attends to, and what breaks without it.
  // The measurement takes about half a second, so views ask for it and get it on a later frame.
  TL.analysisReady = function () {
    if (this.analysis && this.analysis.version === this.version) return this.analysis;
    if (!this.analysisTimer) this.analysisTimer = setTimeout(() => { this.analysisTimer = null; if (!this.training) { this.analyze(); this.inspDirty = true; } }, 60);
    return null;
  };
  TL.analyze = function () {
    if (this.analysis && this.analysis.version === this.version) return this.analysis;
    const m = this.model, set = this.val.slice(0, VAL_N), { L, H: NH } = m.cfg;
    const keepH = m.headMask.map((a) => Uint8Array.from(a)), keepM = Uint8Array.from(m.mlpMask);
    const base = TF.evaluate(m, set, D.W), stats = TF.headStats(m, set), profile = queryProfile(m, set);
    const heads = [];
    for (let l = 0; l < L; l++) for (let h = 0; h < NH; h++) {
      m.headMask.forEach((a, i) => a.set(keepH[i])); m.mlpMask.set(keepM);
      m.headMask[l][h] = 0;
      const e = TF.evaluate(m, set, D.W);
      heads.push({ l, h, stats: stats[l][h], profile: profile[l][h], without: e });
    }
    const mlps = [];
    for (let l = 0; l < L; l++) {
      m.headMask.forEach((a, i) => a.set(keepH[i])); m.mlpMask.set(keepM);
      m.mlpMask[l] = 0;
      mlps.push({ l, without: TF.evaluate(m, set, D.W) });
    }
    m.headMask.forEach((a, i) => a.set(keepH[i])); m.mlpMask.set(keepM);
    heads.forEach((x) => { x.role = describeHead(x, base); });
    this.analysis = this.lastAnalysis = { version: this.version, base, heads, mlps };
    return this.analysis;
  };
  // Where each head looks from the position just before a drink is asked for ("drinks").
  function queryProfile(m, stories) {
    const { L, H: NH } = m.cfg, out = Array.from({ length: L }, () => Array.from({ length: NH }, () => ({ drinks: 0, name: 0, likes: 0, dots: 0, self: 0, bos: 0, n: 0 })));
    for (const s of stories) {
      const n = m.forward(s.ids);
      for (const r of s.roles) {
        if (r.type !== 'lookup') continue;
        for (let l = 0; l < L; l++) for (let h = 0; h < NH; h++) {
          const A = m.ws.att[l][h], ro = r.q * n, P = out[l][h];
          for (let j = 0; j <= r.q; j++) {
            const t = s.ids[j], a = A[ro + j];
            if (j === r.q) P.self += a; else if (j === r.q - 1) P.name += a; else if (j === 0) P.bos += a;
            else if (D.W.DRINKLIST.includes(t)) P.drinks += a; else if (t === D.W.LIKES) P.likes += a; else if (t === D.W.DOT) P.dots += a;
          }
          P.n++;
        }
      }
    }
    out.forEach((row) => row.forEach((P) => { for (const k of ['drinks', 'name', 'likes', 'dots', 'self', 'bos']) P[k] /= Math.max(1, P.n); }));
    return out;
  }
  function describeHead(x, base) {
    const s = x.stats, p = x.profile, dl = (base.lookup - x.without.lookup) * 100, da = (base.agree - x.without.agree) * 100;
    const where = [['drinks', 'gathers the drinks mentioned so far'], ['name', 'reads the name just before "drinks"'], ['likes', 'looks at the "likes" words'], ['dots', 'looks at sentence ends'], ['bos', 'rests on ⟨bos⟩'], ['self', 'looks at itself']]
      .map(([k, t]) => ({ k, t, v: p[k] })).sort((a, b) => b.v - a.v)[0];
    if (dl >= 20) return { id: 'lookup', label: 'Lookup', level: 'critical', text: `Needed for lookups: without it accuracy falls ${Math.round(dl)} points. From "drinks" it ${where.t} (${Math.round(where.v * 100)}% of its attention).` };
    if (da >= 8 || s.subject > 0.5) return { id: 'subject', label: 'Subject', level: 'warn', text: `Before a verb it looks back past the nearer noun to the subject (${Math.round(s.subject * 100)}% of its attention)${da >= 1 ? `; without it agreement falls ${Math.round(da)} points` : ''}.` };
    if (s.prev > 0.45) return { id: 'prev', label: 'Previous token', level: 'info', text: `Mostly attends to the token just before (${Math.round(s.prev * 100)}%).` };
    if (s.bos > 0.45) return { id: 'bos', label: 'Resting', level: 'info', text: `Parks ${Math.round(s.bos * 100)}% of its attention on ⟨bos⟩, which works as "nothing to do here".` };
    if (where.v > 0.4) return { id: 'mixed', label: 'Helper', level: 'info', text: `From "drinks" it ${where.t} (${Math.round(where.v * 100)}%); knocking it out ${effect('lookups', dl)}.` };
    return { id: 'mixed', label: 'Mixed', level: 'info', text: `Spreads its attention; knocking it out ${effect('lookups', dl)} and ${effect('agreement', da)}.` };
  }
  // what a knock-out does to a score, in words: a change of d percentage points (positive d = the score falls)
  function effect(name, d) {
    const n = Math.round(Math.abs(d)), pts = `${n} point${n === 1 ? '' : 's'}`;
    return n === 0 ? `leaves ${name} unchanged` : d > 0 ? `lowers ${name} by ${pts}` : `raises ${name} by ${pts}`;
  }

  TL.runTest = function () {
    if (this.training) this.pause();
    this.testResult = Object.assign(TF.evaluate(this.model, this.test, D.W), { step: this.step, version: this.version });
    this.inspDirty = true;
    if (NA.mobile && NA.mobile.mobile) { NA.mobile.sheet(false); NA.mobile.go('insp'); }
    this.tabs.select('training');
    NA.ui.toast.show(`Held-out test: lookups ${fmtPct(this.testResult.lookup, 1)}, agreement ${fmtPct(this.testResult.agree, 1)} on ${this.test.length} new stories.`);
  };

  // ------------------------------------------------------------------ generation
  TL.sampleNext = function (tr) {
    const V = this.model.cfg.V, i = tr.n - 1, p = tr.probs.subarray(i * V, i * V + V);
    if (this.temp <= 0.01) { let b = 0; for (let c = 1; c < V; c++) if (p[c] > p[b]) b = c; return b; }
    const w = new Float64Array(V); let s = 0;
    for (let c = 0; c < V; c++) { if (c === D.W.BOS) continue; w[c] = Math.exp(Math.log(Math.max(p[c], 1e-12)) / this.temp); s += w[c]; }
    let r = Math.random() * s;
    for (let c = 0; c < V; c++) { r -= w[c]; if (r <= 0) return c; }
    return D.W.DOT;
  };
  TL.appendWord = function (id, generated) {
    const ids = this.ids().ids;
    if (ids.length >= this.model.cfg.T) { NA.ui.toast.show(`The context holds ${this.model.cfg.T} tokens; clear some words first.`); return false; }
    this.text = (this.text.trim() + ' ' + D.VOCAB[id]).trim();
    this.sel = null; this.promptChanged(!!generated);
    return true;
  };
  TL.generate = function (sentence) {
    if (this.genTimer) { clearTimeout(this.genTimer); this.genTimer = null; this.renderGenButtons(); return; }
    if (this.genStart === null || this.genStart === undefined) this.genStart = this.ids().ids.length;
    const tick = () => {
      const tr = this.trace(), id = this.sampleNext(tr);
      if (!this.appendWord(id, true) || !sentence || id === D.W.DOT) { this.genTimer = null; this.renderGenButtons(); return; }
      this.genTimer = setTimeout(tick, NA.prefersReducedMotion() ? 0 : 260);
    };
    this.genTimer = setTimeout(tick, 0);
    this.renderGenButtons();
  };
  TL.promptChanged = function (generated) {
    if (!generated) this.genStart = null;
    this.dirty = true; this.inspDirty = true; this.leftDirty = true; this.archStale = true;
    if (this.inputEl && document.activeElement !== this.inputEl) this.inputEl.value = this.text;
  };

  // ------------------------------------------------------------------ lifecycle
  TL.activate = function (preset, ui) {
    this.ui = ui; this.preset = preset;
    if (!this.model) this.load('twoLayer');
    this.buildHyper(ui.hyper);
    this.ro = UI.readouts(ui.readouts, [
      { key: 'step', label: 'Step', keep: true }, { key: 'train', label: 'Train loss' }, { key: 'val', label: 'Val. loss' },
      { key: 'lookup', label: 'Lookups', short: 'Lookup', keep: true, title: 'Validation stories: the right drink after "N drinks"' }, { key: 'agree', label: 'Agreement', title: 'Validation stories: verb number matches the subject' },
    ]);
    this.buildLeft(ui.left);
    this.buildCenter(ui.center);
    this.tabs = UI.tabs(ui.right, [{ id: 'overview', label: 'Overview' }, { id: 'attention', label: 'Attention' }, { id: 'embed', label: 'Embeddings', short: 'Embed.' }, { id: 'lens', label: 'Logit lens' }, { id: 'training', label: 'Training' }], (id, body) => this.buildTab(id, body));
    this.offTheme = theme.onChange(() => { this.dirty = true; this.inspDirty = true; this.leftDirty = true; });
    this.dirty = this.inspDirty = this.leftDirty = true;
  };
  TL.deactivate = function () {
    if (this.genTimer) { clearTimeout(this.genTimer); this.genTimer = null; }
    if (this.offTheme) this.offTheme();
    if (this.resizeObs) this.resizeObs.disconnect();
    if (this.arch) this.arch.destroy();
  };
  TL.archText = function () { const c = this.model.cfg; return `${c.V} tokens → ${c.L} × (attention ${c.H} heads + MLP) → next token`; };

  TL.frame = function (now) {
    this.runFrame();
    if (now - (this.lastRo || 0) > 120) { this.lastRo = now; this.readouts(); }
    if (this.dirty && this.canvas && this.canvas.offsetParent && (!this.training || now - (this.lastDraw || 0) > 90)) { this.lastDraw = now; this.dirty = false; this.draw(); }
    if (this.centerView === 'arch') {
      // the diagram shows live numbers: redraw it, and its explanation, when they change
      if (this.archStale && (!this.training || now - (this.lastArch || 0) > 400)) { this.lastArch = now; this.archStale = false; this.arch.dirty = true; this.archInfoStale = true; }
      this.arch.frame(now);
      if (this.archInfoStale && !this.arch.anim && !this.training) { this.archInfoStale = false; this.arch.refreshFocus(); }
    }
    if (this.leftDirty && this.ui.left.offsetParent && (!this.training || now - (this.lastLeft || 0) > 250)) { this.lastLeft = now; this.leftDirty = false; this.renderLeft(); }
    if (this.inspDirty && this.tabs.body.offsetParent && (!this.training || now - (this.lastInsp || 0) > 400)) { this.lastInsp = now; this.inspDirty = false; this.updateTab(); }
  };
  TL.readouts = function () {
    const p = this.current || {};
    this.ro.set('step', fmtInt(this.step));
    this.ro.set('train', fmt(this.runLoss ?? p.train, 3));
    this.ro.set('val', fmt(p.val, 3));
    this.ro.set('lookup', p.lookup === null || p.lookup === undefined ? '—' : fmtPct(p.lookup, 0));
    this.ro.set('agree', p.agree === null || p.agree === undefined ? '—' : fmtPct(p.agree, 0));
  };

  TL.buildHyper = function (root) {
    root.append(
      UI.field('Model', UI.select([{ value: 'twoLayer', label: '2 layers (trained)' }, { value: 'oneLayer', label: '1 layer (trained)' }, { value: 'scratch', label: '2 layers (untrained)' }], this.variant, (v) => { this.pause(); this.load(v); this.inspDirty = true; this.tabs.select(this.tabId); }, { id: 'lmVariant' }), 'Load a trained model, or fresh random weights to train yourself'),
      UI.field('Learning rate', UI.select([0.0003, 0.001, 0.003].map((v) => ({ value: v, label: String(v) })), this.lr, (v) => { this.lr = +v; }), 'Adam step size (1e-3 is what the shipped models used)'),
      UI.field('Batch', UI.select([8, 16, 32].map((v) => ({ value: v, label: v + ' stories' })), this.batch, (v) => { this.batch = +v; })),
      el('div', { class: 'sheet-actions' },
        el('button', { class: 'btn', type: 'button', onclick: () => { NA.mobile && NA.mobile.sheet(false); this.fastForward(500); } }, 'Train 500 steps'),
        el('button', { class: 'btn', type: 'button', onclick: () => this.runTest() }, 'Run held-out test')));
  };

  // ------------------------------------------------------------------ left: prompt, prediction, generation
  TL.buildLeft = function (root) {
    const body = UI.panel(root, 'Prompt');
    this.presetSel = UI.select([{ value: '', label: 'Example prompts…' }, ...D.PROMPTS.map((p) => ({ value: p.id, label: p.label }))], '', (v) => {
      const p = D.PROMPTS.find((q) => q.id === v);
      if (p) { this.text = p.text; this.sel = null; this.promptNote = p.note; this.promptChanged(); this.inputEl.value = this.text; }
      this.presetSel.value = '';
    }, { 'aria-label': 'Example prompts' });
    this.inputEl = el('input', { class: 'lm-input', type: 'text', value: this.text, spellcheck: 'false', autocomplete: 'off', autocapitalize: 'off', 'aria-label': 'Prompt text' });
    this.inputEl.addEventListener('input', () => { this.text = this.inputEl.value; this.sel = null; this.promptNote = null; this.promptChanged(); });
    this.chipsEl = el('div', { class: 'tok-row', role: 'listbox', 'aria-label': 'Tokens; choose one to inspect it' });
    this.warnEl = el('p', { class: 'lm-warn', hidden: true });
    this.noteEl = el('p', { class: 'sec-note' });
    this.predEl = el('div', { class: 'pred-list' });
    // a sentence rather than an eyebrow: eyebrows are set in capitals, and the model's words are lower-case
    this.predHead = el('p', { class: 'sec-note' });
    this.genBtns = el('div', { class: 'btn-row' });
    const temp = UI.slider({ label: 'Temperature', min: 0, max: 1.5, step: 0.05, value: this.temp, format: (v) => (v <= 0.01 ? '0 (always the top word)' : v.toFixed(2)), onInput: (v) => { this.temp = v; this.leftDirty = true; } });
    const vocab = el('div', { class: 'vocab' }, ...D.CATEGORIES.map((c, ci) => el('div', { class: 'vocab-group' },
      el('span', { class: 'vocab-k', text: c.label }),
      el('div', { class: 'vocab-words' }, ...c.words.filter((w) => w !== '<bos>').map((w) => el('button', { class: 'vocab-w', type: 'button', style: { '--tok': theme.c(CAT_TOKEN[c.id]) }, onclick: () => this.appendWord(D.ID[w]) }, w))))));
    this.vocabEl = vocab;
    body.append(
      el('div', { class: 'lm-prompt-row' }, this.inputEl, this.presetSel),
      this.warnEl,
      el('div', { class: 'eyebrow', text: 'Tokens' }), this.chipsEl, this.noteEl,
      el('div', { class: 'btn-row' },
        el('button', { class: 'btn btn-quiet', type: 'button', onclick: () => { const ids = this.ids().ids; if (ids.length > 1) { this.text = D.detok(ids.slice(1, -1)); this.sel = null; this.promptChanged(); this.inputEl.value = this.text; } } }, '⌫ Last word'),
        el('button', { class: 'btn btn-quiet', type: 'button', onclick: () => { this.text = ''; this.sel = null; this.promptChanged(); this.inputEl.value = ''; } }, 'Clear')),
      UI.section('Next word', this.predHead, this.predEl, temp, this.genBtns),
      UI.section('Vocabulary', UI.note(`Every word the model knows. ${NA.tapWord()} one to add it to the prompt.`), vocab));
    this.renderGenButtons();
  };
  TL.renderGenButtons = function () {
    if (!this.genBtns) return;
    const busy = !!this.genTimer;
    clear(this.genBtns).append(
      el('button', { class: 'btn btn-primary', type: 'button', onclick: () => this.generate(true) }, busy ? 'Stop' : 'Write to the end of the sentence'),
      el('button', { class: 'btn', type: 'button', disabled: busy, onclick: () => this.generate(false) }, '+1 word'));
  };
  TL.renderLeft = function () {
    const tr = this.trace(), { unknown } = this.ids(), V = this.model.cfg.V, pos = this.selPos(tr);
    this.warnEl.hidden = !unknown.length;
    if (unknown.length) this.warnEl.textContent = `Not in the vocabulary, so left out: ${unknown.slice(0, 5).join(', ')}${unknown.length > 5 ? '…' : ''}`;
    clear(this.chipsEl);
    tr.ids.forEach((id, i) => {
      const gen = this.genStart !== null && this.genStart !== undefined && i >= this.genStart;
      const b = el('button', { class: 'tok' + (i === pos ? ' is-sel' : '') + (gen ? ' is-gen' : ''), type: 'button', role: 'option', 'aria-selected': String(i === pos), style: { '--tok': tokColor(id) }, title: `Position ${i}: "${word(id)}" (token ${id}, ${D.CATEGORIES[D.CAT_OF[id]].label.toLowerCase()})`, onclick: () => { this.sel = i; this.dirty = true; this.inspDirty = true; this.leftDirty = true; this.archStale = true; } },
        el('span', { class: 'tok-w', text: word(id) }), el('span', { class: 'tok-i', text: String(i) }));
      this.chipsEl.append(b);
    });
    // the model's guess for what comes after the last token
    const top = topK(tr.probs, (tr.n - 1) * V, V, 1)[0];
    if (top) this.chipsEl.append(el('span', { class: 'tok tok-next', style: { '--tok': tokColor(top.id) }, title: 'The model’s most likely next word' }, el('span', { class: 'tok-w', text: word(top.id) + '?' }), el('span', { class: 'tok-i', text: fmtPct(top.p) })));
    this.noteEl.textContent = this.promptNote || (tr.n >= this.model.cfg.T ? `The context is full (${this.model.cfg.T} tokens).` : '⟨bos⟩, "beginning of sequence", is added for you. Choose a token to see what the model predicts after it, and how it got there.');
    // prediction list at the selected position
    const o = pos * V, next = pos + 1 < tr.n ? tr.ids[pos + 1] : null;
    this.predHead.textContent = `After "${word(tr.ids[pos])}" (position ${pos})${next !== null ? `; actual next word: "${word(next)}"` : ''}`;
    const list = topK(tr.probs, o, V, 8);
    if (next !== null && !list.some((r) => r.id === next)) list.push({ id: next, p: tr.probs[o + next] });
    // the word in ink, its category in the colour of its bar
    clear(this.predEl).append(...list.map((r) => el('div', { class: 'pred' + (r.id === next ? ' is-actual' : '') },
      el('span', { class: 'pred-w', text: word(r.id) }),
      el('span', { class: 'pred-bar' }, el('span', { style: { width: (r.p * 100).toFixed(1) + '%', background: tokColor(r.id, 0.85) } })),
      el('span', { class: 'pred-p mono', text: fmtPct(r.p, r.p < 0.1 ? 1 : 0) }))));
  };
  function topK(P, o, V, k) {
    const idx = Array.from({ length: V }, (_, c) => c).sort((a, b) => P[o + b] - P[o + a]);
    return idx.slice(0, k).map((id) => ({ id, p: P[o + id] }));
  }

  // ------------------------------------------------------------------ centre: anatomy diagram
  TL.buildCenter = function (root) {
    const L = this.model.cfg.L;
    this.headSeg = el('div', { class: 'seg seg-sm', role: 'group', 'aria-label': 'Attention heads shown' });
    this.renderHeadSeg();
    const lensChip = UI.chip('Logit lens', this.lens, (on) => { this.lens = on; this.dirty = true; }, 'Print what the model would predict if it stopped at each stage');
    this.centerView = this.centerView || 'tokens';
    this.viewSeg = UI.seg([{ value: 'tokens', label: 'Tokens', title: 'Every token through every stage' }, { value: 'arch', label: 'Architecture', title: 'The model’s parts: click or zoom in to open them' }], this.centerView, (v) => this.setCenterView(v), 'View');
    this.tokTools = el('span', { class: 'lab-tools' }, el('span', { class: 'lens-label', text: 'Heads' }), this.headSeg, lensChip);
    root.append(el('div', { class: 'lab-toolbar' }, this.viewSeg, this.tokTools));
    this.stage = el('div', { class: 'lab-stage' });
    this.canvas = el('canvas', { 'aria-label': 'Transformer anatomy: one row per token, one column per stage. Attention arcs show where the selected token looks.' });
    this.stage.append(this.canvas);
    this.archStage = el('div', { class: 'lab-stage arch-stage' });
    this.arch = new NA.ArchView(this.archStage, { label: `Transformer architecture: ${NA.openHint('a part', { zoom: true })}`, build: (size) => NA.archTransformer.build(this, size), onFocus: (node, path) => this.showArchInfo(node, path) });
    this.foot = el('div', { class: 'lab-foot' },
      el('span', { class: 'lg-item' }, el('span', { class: 'ramp ramp-div' }), 'cell = the token’s residual stream vector (32 numbers)'),
      el('span', { class: 'lg-item' }, el('i', { class: 'key-line', style: { background: 'var(--s1)' } }), 'arc = attention from the selected token to an earlier one; width = weight'),
      this.footBlocks = el('span', { class: 'lg-item', text: FOOT_BLOCKS(L) })); // load() keeps the count in step with the model
    root.append(this.stage, this.archStage);
    this.info = UI.infoPanel(root, { open: !(NA.mobile && NA.mobile.mobile) && root.clientHeight >= 680 }); // open at first only when there is room for it under the picture
    root.append(this.foot);
    this.zoom = new NA.ui.Zoom(this.canvas, { host: this.stage, panOne: true, max: 4, wheel: 'fixed', onChange: () => { this.dirty = true; NA.tooltip.hide(); } });
    this.canvas.addEventListener('pointermove', (e) => this.hover(e));
    this.canvas.addEventListener('pointerleave', () => NA.tooltip.hide());
    this.canvas.addEventListener('click', (e) => this.click(e));
    this.resizeObs = new ResizeObserver(() => { this.dirty = true; });
    this.resizeObs.observe(this.stage);
    this.setCenterView(this.centerView);
  };
  TL.setCenterView = function (v) {
    this.centerView = v;
    const arch = v === 'arch';
    this.stage.hidden = arch; this.foot.hidden = arch; this.tokTools.hidden = arch; this.archStage.hidden = !arch;
    this.viewSeg.set(v);
    this.dirty = true; this.arch.dirty = true;
    if (arch) this.arch.refreshFocus();
    else this.info.set(TOKENS_INFO(this));
  };
  // The info panel follows the part being looked at in the architecture.
  TL.showArchInfo = function (node, path) { if (this.centerView === 'arch') this.info.set(this.arch.infoFor(node, path)); };
  const TOKENS_INFO = (T) => ({
    kicker: 'Tokens view', title: 'Every token through every stage',
    about: [
      'Each row is one token of your prompt and each column a stage of the model, from the embedding on the left to the prediction on the right. A cell is that token’s residual-stream vector after the stage: its 32 numbers drawn as thin bars, blue for positive and orange for negative. Reading a row from left to right shows one token’s representation being rewritten as it passes through the model.',
      'The arcs in the two attention columns start at the selected token and end at the earlier tokens it attends to, one colour per head; the thicker the arc, the more attention weight. Attention is the only place where information moves between rows: the embedding, LayerNorms and MLPs all work on one row at a time. A token can never attend to a later one (the causal mask), so arcs only ever point up.',
      'With the logit lens on, every cell is decoded with the model’s own output layer as if the model stopped there, and the top word is printed in it. Try "carol likes juice . dave likes coffee . emma likes water . dave drinks": on the "drinks" row the lens says "water", a drink from the story but the wrong one, right through block 1 and block 2’s attention; only block 2’s MLP switches it to "coffee". The embedding column has seen no context at all, so it makes the same guess for every prompt.',
      `The Next column is the model’s prediction at every position, not only the last: a ✓ means it matches the word that really comes next in your prompt. Misses are expected wherever the story is random (after "likes", any unused drink is fair). ${NA.tapWord()} a row to select a token, or switch to Architecture to open the model up.`,
    ],
    code: `import torch

# Record the residual stream after every stage, as this view draws it. model, tokenize and
# VOCAB are defined in the whole-model code (Architecture, top level, Python tab).
ids = torch.tensor([tokenize(${JSON.stringify(T.text.trim())})])
stream = []
x = model.tok(ids) + model.pos(torch.arange(ids.shape[1]))
stream.append(x)                                        # embedding
for block in model.blocks:
    x = x + block.attn(block.ln1(x)); stream.append(x)  # after attention
    x = x + block.mlp(block.ln2(x)); stream.append(x)   # after the MLP

# The logit lens: decode every stage with the final LayerNorm and unembedding
names = ['embedding'] + [f'{part} {b + 1}' for b in range(len(model.blocks)) for part in ('attention', 'MLP')]
for name, s in zip(names, stream):
    probs = model.unembed(model.ln_f(s)).softmax(-1)[0]   # (n, V)
    print(name.ljust(12), [VOCAB[i] for i in probs.argmax(-1)])

# The arcs: block 1's attention weights for every head
att, n, d = model.blocks[0].attn, ids.shape[1], model.tok.embedding_dim
q, k, v = att.qkv(model.blocks[0].ln1(stream[0])).split(d, dim=-1)
q, k = (t.view(1, n, att.h, att.dh).transpose(1, 2) for t in (q, k))
scores = (q @ k.transpose(-2, -1) / att.dh ** 0.5).masked_fill(~att.causal[:n, :n], float('-inf'))
weights = scores.softmax(-1)          # (1, heads, n, n): row i = where token i looks`,
  });

  TL.renderHeadSeg = function () {
    const { L, H: NH } = this.model.cfg, opts = [{ value: 'all', label: 'All' }];
    for (let l = 0; l < L; l++) for (let h = 0; h < NH; h++) opts.push({ value: `${l}.${h}`, label: `L${l + 1}H${h + 1}` });
    clear(this.headSeg);
    for (const o of opts) {
      const [l, h] = o.value === 'all' ? [-1, -1] : o.value.split('.').map(Number);
      this.headSeg.append(el('button', { type: 'button', 'aria-pressed': String(this.headFilter === o.value), class: l >= 0 && !this.model.headMask[l][h] ? 'is-off' : null, style: l >= 0 ? { '--hc': headColor(h) } : null, onclick: () => { this.headFilter = o.value; this.renderHeadSeg(); this.dirty = true; } }, o.label));
    }
  };
  const headColor = (h) => theme.series[[0, 2, 6, 3][h % 4]];

  TL.layout = function (W, Hh, tr) {
    const L = this.model.cfg.L, nStages = 1 + 2 * L, n = tr.n, narrow = W < 520;
    /* The rows share the height between the column headers and a strip kept clear at the bottom for the zoom
       control (− 1× +), which sits over the Next column and is taller on touch screens, so it is measured. A long
       prompt on a short stage gets rows down to 9 px rather than running under the control; below 22 px each
       row is drawn on one line (compact). */
    const chip = this.zoom && this.zoom.chip, top = 30, bottom = chip && chip.offsetParent ? chip.offsetHeight + 16 : 14;
    const rowH = clamp((Hh - top - bottom) / Math.max(n, 8), 9, 34);
    const tokW = narrow ? 56 : 74, outW = narrow ? 70 : 104, gap = narrow ? 4 : 8;
    const colW = (W - tokW - outW - gap * 2) / nStages;
    const cols = [];
    for (let s = 0; s < nStages; s++) {
      const x = tokW + gap + s * colW, attn = s % 2 === 1;
      // attention columns leave room on the left for the arcs
      const cellW = attn ? colW * 0.56 : colW * 0.78;
      cols.push({ s, x, w: colW, cellX: x + (attn ? colW * 0.42 : colW * 0.11), cellW, attn, layer: attn ? (s - 1) / 2 : s > 0 ? s / 2 - 1 : -1 });
    }
    const rows = []; for (let i = 0; i < n; i++) rows.push(top + rowH * (i + 0.5));
    return { W, H: Hh, top, rowH, tokW, outW, outX: W - outW, cols, rows, narrow, compact: rowH < 22, cellH: Math.min(14, rowH * 0.56) };
  };

  TL.draw = function () {
    const st = this.stage, W = Math.max(200, st.clientWidth), Hh = Math.max(220, st.clientHeight);
    const ctx = NA.fitCanvas(this.canvas, W, Hh), tr = this.trace(), lay = this.layout(W, Hh, tr), d = this.model.cfg.d, V = this.model.cfg.V;
    this.lay = lay; this.tr = tr;
    ctx.clearRect(0, 0, W, Hh);
    ctx.save();
    this.zoom.apply(ctx, this.canvas._dpr);
    const pos = this.selPos(tr), mono = getComputedStyle(document.body).getPropertyValue('--font-mono') || 'monospace', ui = getComputedStyle(document.body).getPropertyValue('--font-ui') || 'sans-serif';
    const names = lay.narrow ? ['Emb', ...Array.from({ length: this.model.cfg.L }, (_, l) => [`Att ${l + 1}`, `MLP ${l + 1}`]).flat()] : STAGE_NAMES(this.model.cfg.L);
    // column headers
    ctx.font = `600 ${lay.narrow ? 10 : 11}px ${ui}`; ctx.textAlign = 'center'; ctx.textBaseline = 'middle'; ctx.fillStyle = theme.c('muted');
    lay.cols.forEach((c, i) => ctx.fillText(names[i].toUpperCase(), c.cellX + c.cellW / 2, 14));
    ctx.fillText('NEXT', lay.outX + lay.outW / 2, 14);
    ctx.textAlign = 'right'; ctx.fillText('TOKEN', lay.tokW - 6, 14);
    // selected row band
    const yS = lay.rows[pos];
    ctx.fillStyle = theme.c('ink', theme.dark ? 0.07 : 0.05);
    rr(ctx, 2, yS - lay.rowH / 2 + 1, lay.W - 4, lay.rowH - 2, 6); ctx.fill();
    // residual stream lines
    ctx.strokeStyle = theme.c('line-2'); ctx.lineWidth = 1;
    for (const y of lay.rows) { ctx.beginPath(); ctx.moveTo(lay.tokW, y); ctx.lineTo(lay.outX - 4, y); ctx.stroke(); }
    // attention arcs for the selected query
    for (const c of lay.cols) if (c.attn) this.drawArcs(ctx, lay, tr, c, pos);
    // cells: residual stream after each stage, as a barcode
    lay.cols.forEach((c, s) => {
      const R = tr.resid[s];
      let m = 1e-6; for (let k = 0; k < tr.n * d; k++) m = Math.max(m, Math.abs(R[k]));
      for (let i = 0; i < tr.n; i++) {
        const y = lay.rows[i] - lay.cellH / 2, bw = c.cellW / d;
        ctx.fillStyle = theme.c('panel'); ctx.fillRect(c.cellX - 1, y - 1, c.cellW + 2, lay.cellH + 2);
        for (let j = 0; j < d; j++) { ctx.fillStyle = theme.divColor(R[i * d + j] / m); ctx.fillRect(c.cellX + j * bw, y, bw + 0.3, lay.cellH); }
        const isSel = i === pos && this.stageSel === s;
        ctx.strokeStyle = isSel ? theme.c('focus') : theme.c('line-2'); ctx.lineWidth = isSel ? 1.6 : 0.6;
        ctx.strokeRect(c.cellX - 0.5, y - 0.5, c.cellW + 1, lay.cellH + 1);
        if (this.lens && lay.rowH >= 17) {
          const P = tr.lens[s], o = i * V;
          let b = 0; for (let k = 1; k < V; k++) if (P[o + k] > P[o + b]) b = k;
          const label = word(b), p = P[o + b];
          ctx.font = `600 ${lay.narrow ? 9 : 10}px ${mono}`;
          const tw = ctx.measureText(label).width + 6;
          if (tw < c.cellW + 4) {
            // a confident guess (over 50%) is written in ink on a tag outlined in its word category's colour; an unsure one in grey
            const sure = s > 0 && p > 0.5;
            ctx.fillStyle = theme.c('panel', 0.86); rr(ctx, c.cellX + c.cellW / 2 - tw / 2, lay.rows[i] - 6.5, tw, 13, 3); ctx.fill();
            if (sure) { ctx.strokeStyle = tokColor(b); ctx.lineWidth = 1.2; ctx.stroke(); }
            ctx.fillStyle = s === 0 ? theme.c('muted') : sure ? theme.c('ink') : theme.c('ink-2', 0.9); ctx.textAlign = 'center';
            ctx.fillText(label, c.cellX + c.cellW / 2, lay.rows[i] + 0.5);
          }
        }
      }
    });
    // token labels in ink; a short bar beside each carries its word category's colour, which is too light to read as text
    const fsTok = lay.compact ? clamp(lay.rowH - 2.5, 8.5, lay.narrow ? 11 : 12) : lay.narrow ? 11 : 12;
    ctx.textBaseline = 'middle';
    tr.ids.forEach((id, i) => {
      const y = lay.rows[i], mh = Math.min(10, lay.rowH - 4);
      ctx.fillStyle = tokColor(id); ctx.fillRect(lay.tokW - 6, y - mh / 2, 3, mh);
      ctx.font = `${i === pos ? 700 : 500} ${fsTok}px ${mono}`; ctx.textAlign = 'right';
      ctx.fillStyle = i === pos ? theme.c('ink') : theme.c('ink-2');
      ctx.fillText(word(id), lay.tokW - 10, y);
      if (!lay.narrow) { ctx.fillStyle = theme.c('muted'); ctx.font = `500 ${Math.min(9, fsTok)}px ${mono}`; ctx.textAlign = 'left'; ctx.fillText(String(i), 4, y); }
    });
    /* Next column: the top word in ink and its probability as a bar in the word category's colour. Rows of 22 px
       or more put the bar under the word; shorter rows run it behind the word, so nothing reaches into the row above. */
    const fsOut = lay.compact ? clamp(lay.rowH - 1.5, 8, lay.narrow ? 10 : 11) : lay.narrow ? 10 : 11, bw = lay.outW - 10;
    for (let i = 0; i < tr.n; i++) {
      const o = i * V; let b = 0; for (let k = 1; k < V; k++) if (tr.probs[o + k] > tr.probs[o + b]) b = k;
      const p = tr.probs[o + b], x = lay.outX + 4, y = lay.rows[i], actual = i + 1 < tr.n ? tr.ids[i + 1] : null, ty = lay.compact ? y + 0.5 : y - 3;
      if (lay.compact) {
        const h = lay.rowH - 3;
        ctx.fillStyle = theme.c('panel-2'); ctx.fillRect(x, y - h / 2, bw, h);
        ctx.fillStyle = tokColor(b, theme.dark ? 0.5 : 0.35); ctx.fillRect(x, y - h / 2, bw * p, h);
      } else {
        ctx.fillStyle = theme.c('panel-2'); ctx.fillRect(x, y + 5, bw, 3);
        ctx.fillStyle = tokColor(b, 0.9); ctx.fillRect(x, y + 5, bw * p, 3);
      }
      ctx.font = `600 ${fsOut}px ${mono}`; ctx.textAlign = 'left'; ctx.fillStyle = theme.c('ink');
      ctx.fillText(word(b), x + (lay.compact ? 3 : 0), ty);
      ctx.fillStyle = theme.c('muted'); ctx.font = `500 ${Math.min(9.5, fsOut)}px ${mono}`; ctx.textAlign = 'right';
      if (!lay.narrow) ctx.fillText(fmtPct(p), lay.W - 6 - (actual !== null ? 12 : 0), ty); // on phones the bar alone shows it
      if (actual !== null) { ctx.fillStyle = actual === b ? theme.c('good') : theme.c('critical'); ctx.fillText(actual === b ? '✓' : '✗', lay.W - 4, ty); }
    }
    ctx.restore();
  };
  TL.drawArcs = function (ctx, lay, tr, c, pos) {
    const l = c.layer, NH = this.model.cfg.H, x0 = c.cellX - 3, yq = lay.rows[pos], maxBulge = c.cellX - (c.x + 2);
    for (let h = 0; h < NH; h++) {
      if (this.headFilter !== 'all' && this.headFilter !== `${l}.${h}`) continue;
      const A = tr.attn[l][h], off = !this.model.headMask[l][h], solo = this.headFilter !== 'all';
      for (let j = 0; j <= pos; j++) {
        const a = A[pos * tr.n + j];
        if (a < 0.03) continue;
        const yk = lay.rows[j], bulge = Math.min(maxBulge, 6 + (pos - j) * lay.rowH * 0.35) * (0.55 + 0.45 * ((h + 1) / NH));
        ctx.strokeStyle = off ? theme.c('muted', 0.35) : solo ? theme.c('s1', 0.25 + 0.75 * a) : headColor(h);
        ctx.globalAlpha = off ? 0.5 : solo ? 1 : 0.2 + 0.8 * a;
        ctx.lineWidth = 0.6 + a * (solo ? 6 : 4.5);
        if (off) ctx.setLineDash([3, 3]);
        ctx.beginPath();
        if (j === pos) ctx.arc(x0 - 4, yq, 4, 0, Math.PI * 2);
        else { ctx.moveTo(x0, yk); ctx.bezierCurveTo(x0 - bulge, yk, x0 - bulge, yq, x0, yq); }
        ctx.stroke();
        ctx.setLineDash([]); ctx.globalAlpha = 1;
      }
    }
  };
  TL.hit = function (e) {
    if (!this.lay) return null;
    const r = this.canvas.getBoundingClientRect(), [x, y] = this.zoom.toWorld(e.clientX - r.left, e.clientY - r.top), lay = this.lay;
    let row = -1;
    lay.rows.forEach((ry, i) => { if (Math.abs(y - ry) <= lay.rowH / 2) row = i; });
    if (row < 0) return null;
    let stage = null;
    lay.cols.forEach((c) => { if (x >= c.x && x < c.x + c.w) stage = c.s; });
    return { row, stage, out: x >= lay.outX, token: x < lay.tokW };
  };
  TL.click = function (e) {
    if (this.zoom.suppressClick) return;
    const h = this.hit(e);
    if (!h) return;
    this.sel = h.row; this.stageSel = h.stage; this.archStale = true;
    this.dirty = true; this.inspDirty = true; this.leftDirty = true;
    if (h.stage !== null && this.tabId !== 'lens' && this.tabId !== 'attention') this.tabs.select('lens');
  };
  TL.hover = function (e) {
    const h = this.hit(e), tr = this.tr;
    if (!h || !tr || this.zoom.busy) { NA.tooltip.hide(); return; }
    const V = this.model.cfg.V, id = tr.ids[h.row], kids = [el('div', { class: 'tt-title', text: `"${word(id)}" · position ${h.row}` })];
    if (h.stage !== null) {
      const P = tr.lens[h.stage], o = h.row * V, top = topK(P, o, V, 3);
      kids.push(el('div', { class: 'tt-hint', text: `${STAGE_NAMES(this.model.cfg.L)[h.stage]}: residual stream after this stage. Decoded now, it says:` }));
      for (const t of top) kids.push(el('div', { class: 'tt-row' }, el('span', { class: 'tt-k', text: word(t.id) }), el('span', { class: 'tt-v', text: fmtPct(t.p, 1) })));
      const c = this.lay.cols[h.stage];
      if (c.attn) {
        const l = c.layer, parts = [];
        for (let hh = 0; hh < this.model.cfg.H; hh++) {
          const A = tr.attn[l][hh]; let b = 0; for (let j = 1; j <= h.row; j++) if (A[h.row * tr.n + j] > A[h.row * tr.n + b]) b = j;
          parts.push(el('div', { class: 'tt-row' }, el('span', { class: 'tt-k', text: `L${l + 1}H${hh + 1} looks at` }), el('span', { class: 'tt-v', text: `"${word(tr.ids[b])}" ${fmtPct(A[h.row * tr.n + b])}` })));
        }
        kids.push(el('div', { class: 'tt-hint', text: 'Strongest attention of each head:' }), ...parts);
      }
    } else if (h.out) {
      const top = topK(tr.probs, h.row * V, V, 4);
      kids.push(el('div', { class: 'tt-hint', text: 'Prediction for the next token:' }), ...top.map((t) => el('div', { class: 'tt-row' }, el('span', { class: 'tt-k', text: word(t.id) }), el('span', { class: 'tt-v', text: fmtPct(t.p, 1) }))));
    } else kids.push(el('div', { class: 'tt-hint', text: `Token ${id}, ${D.CATEGORIES[D.CAT_OF[id]].label.toLowerCase()}. Click to select.` }));
    NA.tooltip.show(kids, e.clientX, e.clientY);
  };

  // ------------------------------------------------------------------ right: inspector tabs
  TL.buildTab = function (id, body) {
    this.tabBody = body; this.tabParts = {};
    const P = this.tabParts;
    if (id === 'overview') {
      body.append(UI.modelCard(this.preset, (t) => this.runTry(t)));
      P.findings = el('div', { class: 'diag-list' });
      body.append(UI.section('What the model has learned', UI.note('Measured on validation stories it never trained on. Knock-out results come from switching each part off and re-scoring.'), P.findings));
    } else if (id === 'attention') {
      P.maps = el('div', { class: 'att-grid' });
      P.roles = el('div', { class: 'diag-list' });
      body.append(UI.section('Attention patterns for this prompt', UI.note(`Each square shows one head: row = the token doing the looking, column = the earlier token it looks at. Darker = more attention. ${NA.tapWord()} a square to show that head on the diagram.`), P.maps),
        UI.section('What each head does', UI.note('Switch a head off to see what breaks. The scores re-measure lookups and agreement without it.'), P.roles));
    } else if (id === 'embed') {
      this.embedView = this.embedView || 'tok';
      P.seg = UI.seg([{ value: 'tok', label: 'Token embeddings' }, { value: 'out', label: 'Output vectors' }, { value: 'pos', label: 'Positions' }], this.embedView, (v) => { this.embedView = v; this.inspDirty = true; }, 'Which vectors');
      P.canvas = el('canvas', { class: 'emb-canvas' });
      P.legend = el('div', { class: 'mini-legend' });
      P.nn = el('div', { class: 'nn-list' });
      P.note = UI.note('');
      // positions have no nearest words: that section is hidden in the Positions view (renderEmbed)
      P.nnSec = UI.section('Nearest words', UI.note(`${NA.tapWord()} a word in the plot. Similar vectors mean the model treats the words alike.`), P.nn);
      body.append(UI.section('The embedding layer', UI.note('The first layer is a lookup table: each of the 36 words has a learned vector of 32 numbers, and each position has another that is added to it. Here the vectors are flattened to 2-D with PCA.'), P.seg, el('div', { class: 'chart-box' }, P.canvas), P.legend, P.note), P.nnSec);
      P.canvas.addEventListener('click', (e) => this.embedClick(e));
      // crowded labels are left out of the plot, so a word can also be read by pointing at its dot
      P.canvas.addEventListener('pointermove', (e) => { if (e.pointerType === 'mouse') this.embedHover(e); });
      P.canvas.addEventListener('pointerleave', () => NA.tooltip.hide());
    } else if (id === 'lens') {
      P.head = el('p', { class: 'sec-note' });
      P.table = el('div', { class: 'lens-table' });
      P.attr = el('div', { class: 'attr-list' });
      P.attrNote = UI.note('');
      body.append(UI.section('Logit lens', P.head, P.table), UI.section('Who pushed the answer', P.attrNote, P.attr));
    } else if (id === 'training') {
      P.tiles = el('div', { class: 'tiles' });
      P.test = el('div', { class: 'test-card' });
      const mk = (title, opts, legend, note) => { const c = el('canvas'); const chart = new C.LineChart(c, opts); body.append(UI.section(title, el('div', { class: 'chart-box' }, c), legend, note ? UI.note(note) : null)); return chart; };
      const lg = (items) => el('div', { class: 'mini-legend' }, ...items.map(([n, col]) => el('span', { class: 'lg-item' }, el('i', { class: 'key-line', style: { background: col } }), n)));
      body.append(P.tiles, UI.section('Held-out test', P.test));
      P.loss = mk('Loss', { height: 150, logY: true, format: (v) => fmt(v, 3) }, lg([['train', 'var(--ink-2)'], ['validation', 'var(--pos)']]), 'Cross-entropy per token. It cannot reach 0: names and drinks are chosen at random, so many words are genuinely unpredictable.');
      P.skills = mk('Skills', { height: 140, format: (v) => fmtPct(v, 0), yFormat: (v) => Math.round(v * 100) + '%' }, lg([['lookups', 'var(--s3)'], ['agreement', 'var(--s7)']]), 'Lookups sit at chance until the circuit that combines "who" with "what" forms, then jump. Agreement is learned much earlier.');
      // the held-out test has its own button in its section above; these are the training actions
      P.actions = el('div', { class: 'btn-row' },
        el('button', { class: 'btn', type: 'button', onclick: () => this.fastForward(500) }, 'Train 500 more steps'),
        el('button', { class: 'btn', type: 'button', onclick: () => this.reset() }, 'Train from scratch'));
      body.append(UI.section('Experiment', UI.note('Training uses Adam on batches of freshly generated stories. A trained model can keep training; from scratch, lookups usually appear after about 2,000 steps.'), P.actions));
    }
    this.tabId = id;
    this.updateTab();
  };

  TL.updateTab = function () {
    const id = this.tabId, P = this.tabParts, tr = this.trace();
    if (!P) return;
    if (id === 'overview') this.renderFindings(P.findings);
    else if (id === 'attention') this.renderAttention(P, tr);
    else if (id === 'embed') this.renderEmbed(P);
    else if (id === 'lens') this.renderLens(P, tr);
    else if (id === 'training') this.renderTraining(P);
  };

  TL.renderFindings = function (root) {
    if (this.training) { clear(root).append(UI.note('Pause training to measure.')); return; }
    const A = this.analysisReady();
    if (!A) { clear(root).append(UI.note('Measuring…')); return; }
    const b = A.base, items = [];
    const f = (level, title, text, label) => items.push(el('div', { class: 'diag diag-' + level }, el('div', { class: 'diag-head' }, UI.pill(level, label || { good: 'Good', info: 'Note', warn: 'Watch' }[level]), el('h4', { class: 'diag-title', text: title })), el('p', { class: 'diag-body', text })));
    f(b.lookup > 0.9 ? 'good' : 'warn', `Lookups: ${fmtPct(b.lookup, 0)} right`, b.lookup > 0.9 ? 'After "N drinks" it names the drink that person likes, although every story pairs people and drinks at random. That can only come from reading the context.' : `Close to guessing among the drinks mentioned (about ${fmtPct(0.45, 0)}). ${this.model.cfg.L < 2 ? 'One layer cannot combine who is asking with what they like.' : 'Train longer: this skill appears suddenly.'}`);
    f(b.agree > 0.9 ? 'good' : 'warn', `Agreement: ${fmtPct(b.agree, 0)} right`, `It picks "sleeps" after one animal and "sleep" after several, ${fmtPct(b.conflict ?? 0, 0)} of the time even when a nearer noun has the other number.`);
    const key = A.heads.filter((x) => x.role.id === 'lookup');
    // a key part is news, not a fault: it gets the information badge, not the red ✕ of a problem
    if (key.length) f('info', `Lookup circuit: ${key.map((x) => `L${x.l + 1}H${x.h + 1}`).join(', ')}${A.mlps.filter((m) => (b.lookup - m.without.lookup) > 0.2).map((m) => ', MLP ' + (m.l + 1)).join('')}`, key.map((x) => `L${x.l + 1}H${x.h + 1}: ${x.role.text}`).join(' '), 'Key part');
    const subj = A.heads.filter((x) => x.role.id === 'subject');
    if (subj.length) f('info', `Agreement heads: ${subj.map((x) => `L${x.l + 1}H${x.h + 1}`).join(', ')}`, subj[0].role.text + (subj.length > 1 ? ' Several heads share the job, so knocking out one barely hurts.' : ''));
    f('info', `Perplexity ${fmt(b.ppl, 2)}`, `On average the model is as unsure as a choice between ${fmt(b.ppl, 1)} equally likely words (on the same ${VAL_N} validation stories as the Training tab). Random names and drinks keep this well above 1.`);
    clear(root).append(...items);
  };

  TL.renderAttention = function (P, tr) {
    const { L, H: NH } = this.model.cfg, n = tr.n, pos = this.selPos(tr);
    const avail = Math.max(200, P.maps.clientWidth || 300), size = Math.floor(Math.min(150, (avail - (NH - 1) * 8) / NH));
    // the list is rebuilt in place (after a knock-out, say): keep the reader where they were
    const scroller = this.tabs.body, keep = scroller.scrollTop;
    clear(P.maps);
    // with only ⟨bos⟩ every head can look at nothing but itself: eight solid squares would say nothing
    if (n < 2) P.maps.append(UI.note('The prompt is empty. Type a few words, or pick an example prompt, to see where each head looks.'));
    for (let l = 0; l < L && n >= 2; l++) {
      const row = el('div', { class: 'att-row' }, el('span', { class: 'att-l', text: `Layer ${l + 1}` }));
      for (let h = 0; h < NH; h++) {
        const c = el('canvas', { class: 'att-map' + (this.headFilter === `${l}.${h}` ? ' is-sel' : '') + (this.model.headMask[l][h] ? '' : ' is-off'), title: `L${l + 1}H${h + 1}`, role: 'button', tabindex: '0', 'aria-label': `Show head L${l + 1}H${h + 1} on the diagram` });
        const ctx = NA.fitCanvas(c, size, size), A = tr.attn[l][h], cell = size / n;
        ctx.fillStyle = theme.c('panel-2'); ctx.fillRect(0, 0, size, size);
        for (let i = 0; i < n; i++) for (let j = 0; j <= i; j++) { ctx.fillStyle = theme.seqColor(Math.sqrt(A[i * n + j])); ctx.fillRect(j * cell, i * cell, cell + 0.3, cell + 0.3); }
        ctx.strokeStyle = theme.c('focus', 0.8); ctx.lineWidth = 1; ctx.strokeRect(0.5, pos * cell + 0.5, size - 1, cell - 1);
        const pick = () => { this.headFilter = this.headFilter === `${l}.${h}` ? 'all' : `${l}.${h}`; this.renderHeadSeg(); this.dirty = true; this.inspDirty = true; };
        c.addEventListener('click', pick); c.addEventListener('keydown', (e) => { if (e.key === 'Enter' || e.key === ' ') { e.preventDefault(); pick(); } });
        // the head's colour as a dot, its name in ink (some head colours are too light to read as text)
        row.append(el('div', { class: 'att-cell' }, c, el('span', { class: 'att-cap lg-item' }, el('i', { class: 'dot', style: { background: headColor(h) } }), `H${h + 1}`)));
      }
      P.maps.append(row);
    }
    if (this.training) { clear(P.roles).append(UI.note('Pause training to measure what each head does.')); scroller.scrollTop = keep; return; }
    /* Re-measuring takes about half a second. Meanwhile the previous measurement stays on screen, with the
       buttons already showing the new state, so the list keeps its height and the page does not jump. */
    const fresh = this.analysisReady(), A = fresh || (this.lastAnalysis && this.lastAnalysis.heads.length === L * NH ? this.lastAnalysis : null);
    if (!A) { clear(P.roles).append(UI.note('Measuring each head…')); scroller.scrollTop = keep; return; }
    const b = A.base;
    const rows = A.heads.map((x) => el('div', { class: 'diag head-row' + (this.model.headMask[x.l][x.h] ? '' : ' is-off') },
      el('div', { class: 'diag-head' }, el('span', { class: 'head-dot', style: { background: headColor(x.h) } }), el('h4', { class: 'diag-title', text: `L${x.l + 1}H${x.h + 1} · ${x.role.label}` }),
        el('button', { class: 'btn btn-quiet', type: 'button', 'aria-pressed': String(!this.model.headMask[x.l][x.h]), dataset: { ko: `h${x.l}.${x.h}` }, onclick: () => { this.refocus = `h${x.l}.${x.h}`; this.toggleHead(x.l, x.h); } }, this.model.headMask[x.l][x.h] ? 'Knock out' : 'Restore')),
      el('p', { class: 'diag-body', text: x.role.text }),
      el('div', { class: 'ko' }, ko('Lookups', b.lookup, x.without.lookup), ko('Agreement', b.agree, x.without.agree))));
    const mlps = A.mlps.map((m) => el('div', { class: 'diag head-row' + (this.model.mlpMask[m.l] ? '' : ' is-off') },
      el('div', { class: 'diag-head' }, el('span', { class: 'head-dot head-dot-mlp' }), el('h4', { class: 'diag-title', text: `MLP ${m.l + 1}` }),
        el('button', { class: 'btn btn-quiet', type: 'button', 'aria-pressed': String(!this.model.mlpMask[m.l]), dataset: { ko: `m${m.l}` }, onclick: () => { this.refocus = `m${m.l}`; this.toggleMlp(m.l); } }, this.model.mlpMask[m.l] ? 'Knock out' : 'Restore')),
      el('p', { class: 'diag-body', text: `The feed-forward part of block ${m.l + 1}: 128 neurons that transform each token on its own.` }),
      el('div', { class: 'ko' }, ko('Lookups', b.lookup, m.without.lookup), ko('Agreement', b.agree, m.without.agree))));
    clear(P.roles).append(...rows, ...mlps);
    scroller.scrollTop = keep;
    // the button that was pressed has been rebuilt: give it the keyboard focus back
    if (this.refocus) { const btn = P.roles.querySelector(`[data-ko="${this.refocus}"]`); if (btn) btn.focus({ preventScroll: true }); if (fresh) this.refocus = null; }
  };
  function ko(label, base, without) {
    const drop = (base - without) * 100;
    return el('span', { class: 'ko-item' + (drop >= 20 ? ' is-big' : drop >= 5 ? ' is-mid' : '') }, el('span', { class: 'ko-k', text: label + ' without it' }), el('span', { class: 'mono', text: `${fmtPct(without, 0)}${drop >= 1 ? ` (−${Math.round(drop)})` : ''}` }));
  }
  TL.toggleHead = function (l, h) { this.model.headMask[l][h] ^= 1; this.afterAblation(); };
  TL.toggleMlp = function (l) { this.model.mlpMask[l] ^= 1; this.afterAblation(); };
  TL.afterAblation = function () {
    if (this.training) this.pause();
    this.bumpVersion(); this.leftDirty = true;
    this.current = Object.assign({}, this.current, (({ lookup, agree, loss }) => ({ lookup, agree, val: loss }))(TF.evaluate(this.model, this.val.slice(0, VAL_N), D.W)));
    this.renderHeadSeg();
    const off = this.model.headMask.reduce((s, a) => s + a.filter((v) => !v).length, 0) + this.model.mlpMask.filter((v) => !v).length;
    NA.ui.toast.show(off ? `${off} part${off > 1 ? 's' : ''} knocked out. Lookups now ${fmtPct(this.current.lookup, 0)}, agreement ${fmtPct(this.current.agree, 0)}.` : 'Everything restored.');
  };

  TL.renderEmbed = function (P) {
    const m = this.model, { d, V, T } = m.cfg, view = this.embedView;
    P.seg.set(view);
    P.nnSec.hidden = view === 'pos';
    let n, data, labels, cats;
    if (view === 'pos') {
      n = T; data = new Float32Array(d * n); labels = []; cats = null;
      for (let p = 0; p < n; p++) { labels.push(String(p)); for (let j = 0; j < d; j++) data[j * n + p] = m.Pos.w[p * d + j]; }
    } else {
      n = V - 1; data = new Float32Array(d * n); labels = []; cats = [];
      for (let k = 1; k < V; k++) { labels.push(word(k)); cats.push(k); for (let j = 0; j < d; j++) data[j * n + (k - 1)] = view === 'tok' ? m.E.w[k * d + j] : m.U.w[j * V + k]; }
    }
    const pca = NA.engine.pca2(data, n, d);
    const W = Math.max(220, P.canvas.parentElement.clientWidth), Hh = Math.min(340, Math.round(W * 0.82)), ctx = NA.fitCanvas(P.canvas, W, Hh);
    const mono = getComputedStyle(document.body).getPropertyValue('--font-mono') || 'monospace', font = (bold) => `${bold ? 700 : 500} 10.5px ${mono}`;
    ctx.font = font(true);
    // the right margin keeps the widest label free, so a point at the right edge can still be labelled beside it
    const widest = Math.max(...labels.map((t) => ctx.measureText(t).width)), padL = 14, padR = Math.min(W * 0.3, widest + 12), padT = 14, padB = 28;
    let x0 = Infinity, x1 = -Infinity, y0 = Infinity, y1 = -Infinity;
    for (let p = 0; p < n; p++) { const x = pca.proj[2 * p], y = pca.proj[2 * p + 1]; x0 = Math.min(x0, x); x1 = Math.max(x1, x); y0 = Math.min(y0, y); y1 = Math.max(y1, y); }
    const pw = W - padL - padR, ph = Hh - padT - padB, sx = pw / Math.max(1e-6, x1 - x0), sy = ph / Math.max(1e-6, y1 - y0), s = Math.min(sx, sy);
    const X = (x) => padL + pw / 2 + (x - (x0 + x1) / 2) * s, Y = (y) => padT + ph / 2 - (y - (y0 + y1) / 2) * s;
    ctx.clearRect(0, 0, W, Hh);
    ctx.fillStyle = theme.c('panel-2'); rr(ctx, 0, 0, W, Hh, 8); ctx.fill();
    this.embPts = [];
    const pts = Array.from({ length: n }, (_, p) => ({ x: X(pca.proj[2 * p]), y: Y(pca.proj[2 * p + 1]), label: labels[p], id: cats ? cats[p] : null, sel: !!cats && this.embSel === cats[p], col: cats ? tokColor(cats[p]) : theme.seqColor(0.25 + 0.75 * (p / (n - 1))) }));
    if (view === 'pos') {
      ctx.strokeStyle = theme.c('line-2'); ctx.lineWidth = 1.2; ctx.beginPath();
      pts.forEach((q, p) => (p ? ctx.lineTo(q.x, q.y) : ctx.moveTo(q.x, q.y)));
      ctx.stroke();
    }
    for (const q of pts) { ctx.fillStyle = q.col; ctx.beginPath(); ctx.arc(q.x, q.y, q.sel ? 5 : 3.5, 0, Math.PI * 2); ctx.fill(); this.embPts.push({ x: q.x, y: q.y, id: q.id, label: q.label }); }
    /* Labels, placed greedily: the selected word first, then the rest, each at the first spot around its dot
       (right, left, above, below, then shifted and diagonal spots) that stays inside the plot and covers no
       other label or dot. A word with no free spot keeps its dot unlabelled; pointing at the dot, or selecting
       it, names it. */
    const lh = 12, placed = [], dots = pts.map((q) => ({ x: q.x - 4, y: q.y - 4, w: 8, h: 8 }));
    const hits = (a, b) => a.x < b.x + b.w && b.x < a.x + a.w && a.y < b.y + b.h && b.y < a.y + a.h;
    const free = (r) => r.x >= 3 && r.y >= 3 && r.x + r.w <= W - 3 && r.y + r.h <= Hh - padB + 6 && !placed.some((q) => hits(r, q)) && !dots.some((q) => hits(r, q));
    let hidden = 0;
    for (const q of [...pts.filter((p) => p.sel), ...pts.filter((p) => !p.sel)]) {
      ctx.font = font(q.sel); const w = ctx.measureText(q.label).width + 2, x = q.x, y = q.y, my = y - lh / 2;
      const spots = [[x + 6, my], [x - 6 - w, my], [x - w / 2, y - 5 - lh], [x - w / 2, y + 5], [x + 6, my - 7], [x + 6, my + 7], [x - 6 - w, my - 7], [x - 6 - w, my + 7],
        [x - 4, y - 5 - lh], [x - w + 4, y - 5 - lh], [x - 4, y + 5], [x - w + 4, y + 5], [x + 6, my - 13], [x + 6, my + 13], [x - 6 - w, my - 13], [x - 6 - w, my + 13]];
      const at = spots.map(([sx0, sy0]) => ({ x: sx0, y: sy0, w, h: lh })).find(free) || (q.sel ? { x: x + 7, y: y - lh / 2, w, h: lh } : null);
      if (!at) { hidden++; continue; }
      placed.push(at);
      // words in ink (the selected one bold), so every category reads alike; the dot carries the category's colour
      ctx.fillStyle = q.sel ? theme.c('ink') : theme.c('ink-2'); ctx.textAlign = 'left'; ctx.textBaseline = 'middle';
      ctx.fillText(q.label, at.x + 1, at.y + lh / 2 + 0.5);
    }
    ctx.fillStyle = theme.c('muted'); ctx.font = `500 10px ${mono}`; ctx.textAlign = 'right'; ctx.textBaseline = 'middle';
    ctx.fillText(`PC1 ${fmtPct(pca.explained[0])} · PC2 ${fmtPct(pca.explained[1])} of variance`, W - 8, Hh - 12);
    clear(P.legend);
    if (cats) for (const c of D.CATEGORIES) P.legend.append(el('span', { class: 'lg-item' }, el('i', { class: 'dot', style: { background: theme.c(CAT_TOKEN[c.id]) } }), c.id === 'special' ? 'Full stop' : c.label));
    P.note.textContent = (view === 'tok' ? 'Nobody told the model which words are names or drinks. Words that can stand in the same places end up with similar vectors, so the groups appear on their own.'
      : view === 'out' ? 'The unembedding: the vector each word’s score is read from at the end. Words the model must tell apart sit apart; interchangeable words (any of the six names, say) cluster.'
        : 'Position vectors in order. Neighbouring positions get similar vectors, which helps heads find "the token just before".')
      + (hidden ? ` Where dots crowd, ${hidden} label${hidden > 1 ? 's are' : ' is'} left out${cats ? `: ${NA.tapWord(true)} a dot to name it` : ''}.` : '');
    // nearest neighbours by cosine similarity
    clear(P.nn);
    if (view !== 'pos' && this.embSel) {
      const vec = (k) => { const v = new Float32Array(d); for (let j = 0; j < d; j++) v[j] = view === 'tok' ? m.E.w[k * d + j] : m.U.w[j * V + k]; return v; };
      const a = vec(this.embSel), sims = [];
      for (let k = 1; k < V; k++) {
        if (k === this.embSel) continue;
        const b = vec(k); let ab = 0, aa = 0, bb = 0;
        for (let j = 0; j < d; j++) { ab += a[j] * b[j]; aa += a[j] * a[j]; bb += b[j] * b[j]; }
        sims.push({ k, s: ab / Math.sqrt(aa * bb) });
      }
      sims.sort((x, y) => y.s - x.s);
      P.nn.append(el('p', { class: 'sec-note', text: `Most similar to "${word(this.embSel)}":` }), ...sims.slice(0, 6).map((r) => el('div', { class: 'pred' },
        el('span', { class: 'pred-w', text: word(r.k) }),
        el('span', { class: 'pred-bar' }, el('span', { style: { width: Math.max(0, r.s * 100).toFixed(1) + '%', background: tokColor(r.k, 0.8) } })),
        el('span', { class: 'pred-p mono', text: fmt(r.s, 2) }))));
    } else if (view !== 'pos') P.nn.append(UI.note('No word selected yet.'));
  };
  TL.embedAt = function (e, words) {
    if (!this.embPts) return null;
    const r = this.tabParts.canvas.getBoundingClientRect(), x = e.clientX - r.left, y = e.clientY - r.top;
    let best = null, bd = 18 * 18;
    for (const p of this.embPts) { const dd = (p.x - x) ** 2 + (p.y - y) ** 2; if (dd < bd && (!words || p.id !== null)) { bd = dd; best = p; } }
    return best;
  };
  TL.embedClick = function (e) {
    const best = this.embedAt(e, true);
    if (best) { this.embSel = best.id; this.inspDirty = true; }
  };
  TL.embedHover = function (e) {
    const p = this.embedAt(e, false);
    if (!p) { NA.tooltip.hide(); return; }
    NA.tooltip.show([el('div', { class: 'tt-title', text: p.id === null ? `Position ${p.label}` : `"${p.label}"` }), p.id === null ? null : el('div', { class: 'tt-hint', text: `${D.CATEGORIES[D.CAT_OF[p.id]].label}. Click to list its nearest words.` })], e.clientX, e.clientY);
  };

  TL.renderLens = function (P, tr) {
    const V = this.model.cfg.V, pos = this.selPos(tr), o = pos * V, names = STAGE_NAMES(this.model.cfg.L);
    const final = topK(tr.probs, o, V, 2), a = final[0].id, b = final[1].id;
    P.head.textContent = `Position ${pos}, "${word(tr.ids[pos])}". Each row decodes the residual stream after one stage with the model’s final layer, as if the model stopped there. The embedding row has seen no context yet, so it is the same guess whatever came before; early rows are rough because those stages were never trained to be read directly.`;
    clear(P.table);
    tr.lens.forEach((Lp, s) => {
      const top = topK(Lp, o, V, 4);
      P.table.append(el('div', { class: 'lens-row' + (s === this.stageSel ? ' is-sel' : '') + (s === 0 ? ' is-nocontext' : ''), title: s === 0 ? 'Before any attention: only this word and its position, so the same guess for every context' : null }, el('span', { class: 'lens-stage', text: s === 0 ? 'Embedding (no context)' : names[s] }),
        // the category colour is set on the bar alone, so the word itself stays in ink (.lens-w takes --tok when it is inherited)
        el('div', { class: 'lens-tops' }, ...top.map((t) => el('span', { class: 'lens-tok' }, el('span', { class: 'lens-bar', style: { '--tok': tokColor(t.id), width: Math.max(4, t.p * 100).toFixed(0) + '%' } }), el('span', { class: 'lens-w', text: word(t.id) }), el('span', { class: 'lens-p mono', text: fmtPct(t.p) }))))));
    });
    const parts = this.model.attribution(tr, pos, a, b).filter((p) => Math.abs(p.value) > 0.01).sort((x, y) => Math.abs(y.value) - Math.abs(x.value)).slice(0, 10);
    const total = parts.reduce((s, p) => s + p.value, 0), m = Math.max(...parts.map((p) => Math.abs(p.value)), 1e-6);
    P.attrNote.textContent = `What made "${word(a)}" beat "${word(b)}" here: each part of the network adds to the gap between their scores (logits). Positive bars favour "${word(a)}". Total gap ${fmt(total, 2)}.`;
    clear(P.attr).append(...parts.map((p) => el('div', { class: 'attr-row' },
      el('span', { class: 'attr-name' }, p.head !== undefined ? el('i', { class: 'dot', style: { background: headColor(p.head) } }) : null, p.head !== undefined ? ' ' : null, p.label),
      el('span', { class: 'attr-track' }, el('span', { class: 'attr-fill ' + (p.value >= 0 ? 'pos' : 'neg'), style: { width: (Math.abs(p.value) / m * 50).toFixed(1) + '%', [p.value >= 0 ? 'left' : 'right']: '50%' } })),
      el('span', { class: 'mono attr-v', text: NA.fmtSigned(p.value, 2) }))));
  };

  TL.renderTraining = function (P) {
    const p = this.current || {}, hist = this.history;
    clear(P.tiles).append(UI.tile('Step', fmtInt(this.step), this.pretrained && !this.training ? 'trained offline, same code' : null), UI.tile('Train loss', fmt(this.runLoss ?? p.train, 3)), UI.tile('Val. loss', fmt(p.val, 3), `perplexity ${fmt(Math.exp(p.val ?? NaN), 2)}`),
      UI.tile('Lookups', p.lookup === undefined ? '—' : fmtPct(p.lookup, 0), 'validation'), UI.tile('Agreement', p.agree === undefined ? '—' : fmtPct(p.agree, 0), 'validation'), UI.tile('Parameters', fmtInt(this.model.paramCount), `${this.model.cfg.L} layer${this.model.cfg.L > 1 ? 's' : ''} × ${this.model.cfg.H} heads`));
    const series = (key, name, color) => ({ name, color, points: hist.filter((q) => q[key] !== null && q[key] !== undefined).map((q) => [q.step, q[key]]) });
    P.loss.frame(); P.loss.set({ series: [series('train', 'train', theme.c('ink-2')), series('val', 'validation', theme.c('pos'))] });
    P.skills.frame(); P.skills.set({ series: [series('lookup', 'lookups', theme.c('s3')), series('agree', 'agreement', theme.c('s7'))], yFixed: [0, 1] });
    const t = this.testResult;
    clear(P.test).append(...(t ? [
      el('div', { class: 'test-head' }, el('span', { class: 'eyebrow', text: `${t.n} new stories · step ${fmtInt(t.step)}` }), t.version === this.version ? UI.pill('good', 'Current model') : UI.pill('warn', 'Model changed since')),
      el('div', { class: 'tiles' }, UI.tile('Lookups', fmtPct(t.lookup, 1)), UI.tile('Agreement', fmtPct(t.agree, 1), `${fmtPct(t.conflict, 1)} with a distractor`), UI.tile('Loss', fmt(t.loss, 3), `perplexity ${fmt(t.ppl, 2)}`), UI.tile('Next-word accuracy', fmtPct(t.acc, 1), 'many words are random')),
      el('div', { class: 'btn-row' }, el('button', { class: 'btn', type: 'button', onclick: () => this.runTest() }, 'Run again')),
    ] : [UI.note(`${this.test.length} stories generated with a different random seed, never used for training or validation. Score them once you are done experimenting.`), el('div', { class: 'btn-row' }, el('button', { class: 'btn btn-primary', type: 'button', onclick: () => this.runTest() }, 'Run test'))]));
  };

  TL.runTry = function (t) {
    if (t.action === 'ablateLookup') {
      if (this.variant !== 'twoLayer') this.load('twoLayer');
      const A = this.analyze(), key = A.heads.filter((x) => x.role.id === 'lookup').sort((x, y) => x.without.lookup - y.without.lookup)[0];
      if (key) { this.model.headMask[key.l][key.h] = 0; this.headFilter = `${key.l}.${key.h}`; this.afterAblation(); }
      this.tabs.select('attention');
    } else if (t.action === 'oneLayer') { this.pause(); this.load('oneLayer'); this.renderHeadSeg(); this.tabs.select('overview'); NA.ui.toast.show('Loaded the one-layer model, trained the same way for 6,000 steps.'); }
    else if (t.action === 'scratch') { this.reset(); this.renderHeadSeg(); this.tabs.select('training'); this.play(); }
    else if (t.action === 'prompt') { this.text = t.prompt; this.sel = null; this.promptChanged(); this.inputEl.value = this.text; if (NA.mobile && NA.mobile.mobile) NA.mobile.go('data'); }
  };

  // Zoo card glyph: a column of tokens feeding stacked attention blocks.
  TL.glyph = function (ctx, W, Hh) {
    const ys = [8, 17, 26, 35];
    ctx.strokeStyle = theme.c('line-2'); ctx.lineWidth = 0.8;
    for (const x of [24, 42]) { ctx.beginPath(); ctx.moveTo(x - 3, ys[3]); ctx.bezierCurveTo(x - 12, ys[3], x - 12, ys[0], x - 3, ys[0]); ctx.stroke(); }
    ys.forEach((y, i) => {
      ctx.fillStyle = theme.series[[0, 6, 2, 3][i]]; ctx.fillRect(4, y - 2.5, 9, 5);
      ctx.fillStyle = theme.c('ink-2'); ctx.fillRect(24, y - 2.5, 12, 5); ctx.fillRect(42, y - 2.5, 12, 5);
    });
    ctx.fillStyle = theme.c('ink'); ctx.beginPath(); ctx.arc(60, ys[3], 2.6, 0, Math.PI * 2); ctx.fill();
  };

  H.register(TL);
  NA.lmLab = TL;
})(window.NA);
