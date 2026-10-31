/* builder-shelf.js — My models, and training while the page is hidden, for the model builder.
   My models: a shelf in the Runs tab of networks you keep, each a run's setup and its trained weights under a name of
   yours, in this browser (IndexedDB). Open puts the setup back in the builder and the weights into a new run, which
   starts where they were (Train carries on from them); Download saves it as a file and Load a model file brings one
   back, from this browser or another. The latest run you paused or finished is kept there by itself, so a reload
   loses nothing. In the background: a hidden page gets no animation frames, so its training would stop; a worker keeps
   the time instead, each tick trains a chunk, and the tab's title says how far it has got. */
(function (NA) {
  'use strict';
  const SH = { version: 0 };
  const { el, clear, fmtPct, fmtInt, fmt } = NA;
  const UI = () => NA.labUI;
  const toast = (t) => NA.ui && NA.ui.toast && NA.ui.toast.show(t);
  const DB = 'neural-anatomy', STORE = 'models', LATEST = 'latest', KIND = 'neural-anatomy-model', MAX = 5e6;

  // ------------------------------------------------------------------ the shelf in IndexedDB
  let dbp = null;
  function db() {
    if (!dbp) dbp = new Promise((resolve, reject) => {
      if (typeof indexedDB === 'undefined') { reject(new Error('no IndexedDB')); return; }
      const r = indexedDB.open(DB, 1);
      r.onupgradeneeded = () => r.result.createObjectStore(STORE, { keyPath: 'id' });
      r.onsuccess = () => resolve(r.result);
      r.onerror = () => reject(r.error || new Error('the shelf did not open'));
    }).catch((e) => { dbp = null; throw e; });
    return dbp;
  }
  const tx = (mode, fn) => db().then((d) => new Promise((resolve, reject) => { const t = d.transaction(STORE, mode), q = fn(t.objectStore(STORE)); t.oncomplete = () => resolve(q && q.result); t.onerror = () => reject(t.error); t.onabort = () => reject(t.error); }));
  // (the one kept by itself first, then the newest)
  SH.list = () => tx('readonly', (s) => s.getAll()).then((all) => (all || []).sort((a, b) => (b.auto ? 1 : 0) - (a.auto ? 1 : 0) || b.date - a.date));
  SH.put = (rec) => tx('readwrite', (s) => s.put(rec)).then(() => { SH.version++; return rec; });
  SH.remove = (id) => tx('readwrite', (s) => s.delete(id)).then(() => { SH.version++; });

  // ------------------------------------------------------------------ a run, as a record
  const netOf = (lab) => (lab.agent && lab.agent.net) || (lab.model && !lab.model.classic ? lab.model : null);
  SH.canKeep = (lab) => !!(lab.run && !(lab.agent && lab.agent.neat) && netOf(lab) && typeof netOf(lab).getWeights === 'function' && netOf(lab).paramCount <= MAX);
  const nameOf = (lab) => { const r = lab.run, p = NA.builderTasks.byId[r.cfg.problem], M = r.method && NA.brl.METHODS[r.method], v = r.last ? r.last.vm : null; return `${p ? p.label : r.cfg.problem}${M ? ` · ${M.short}` : ''} · run ${r.n}${v !== null && v !== undefined ? ` · ${NA.builderTasks.isValue(r.family) ? `R² ${fmt(v, 2)}` : fmtPct(v, 0)}` : ''}`; };
  /* The record of the current run: its setup as it started (a share link's, the blocks as its network has them now,
     since grow and prune may have reshaped it) and the network's weights as they are. */
  SH.recordOf = function (lab, name, auto) {
    if (!SH.canKeep(lab)) return null;
    const run = lab.run, net = netOf(lab), S = JSON.parse(JSON.stringify(run.cfg));
    S.blocks = JSON.parse(JSON.stringify(net.evoBlocks || S.blocks)); S.blocks.forEach((b) => { delete b.id; });
    const table = !!(S.custom && S.custom.table);
    if (S.custom) delete S.custom.table;
    const p = NA.builderTasks.byId[S.problem], v = run.last ? run.last.vm : null;
    return { id: auto ? LATEST : `m${Date.now().toString(36)}${Math.random().toString(36).slice(2, 6)}`, auto: !!auto, name: String(name || nameOf(lab)).slice(0, 80), date: Date.now(), problem: S.problem, label: p ? p.label : S.problem,
      family: run.family, method: run.method || null, setup: S, params: net.paramCount, weights: net.getWeights().slice(), score: v === undefined ? null : v, step: run.step, unit: lab.unitOf(run), table };
  };
  // the latest run, kept by itself when it pauses or finishes (a moment later, once: a run pauses often)
  SH.keepLatest = function (lab) {
    clearTimeout(SH._latestT);
    SH._latestT = setTimeout(() => { const rec = SH.recordOf(lab, `The latest run: ${SH.canKeep(lab) ? nameOf(lab) : ''}`, true); if (rec) SH.put(rec).then(() => { lab._runsKey = null; lab.inspDirty = true; }).catch(() => {}); }, 800);
  };
  // Open: the setup back in the builder (as a share link would put it, Undo away), the weights in a new run
  SH.open = function (lab, rec) {
    if (lab.training) lab.pause();
    if (!lab.loadShare(JSON.parse(JSON.stringify(rec.setup)))) { toast('That model’s setup could not be opened here.'); return false; }
    lab.loadedW = { w: rec.weights instanceof Float32Array ? rec.weights : Float32Array.from(rec.weights), name: rec.name };
    const ok = lab.newRun();
    const used = ok && lab.run && lab.run.from === 'kept';
    lab.loadedW = null;
    if (!ok) return false;
    if (!used) { toast(`“${rec.name}” opened, but its weights no longer fit the model, so run ${lab.run.n} starts from fresh weights.`); return true; }
    const v = lab.run.last ? lab.run.last.vm : null;
    toast(`Opened “${rec.name}”: its weights are in run ${lab.run.n}${v !== null && v !== undefined ? `, which scores ${NA.builderTasks.isValue(lab.run.family) ? `R² ${fmt(v, 3)}` : fmtPct(v, 1)}` : ''}. Train carries on from them.${rec.table ? ' It learned from a table of your own: load that table again to train it further.' : ''}`);
    if (NA.mobile && NA.mobile.mobile) NA.mobile.go('network');
    return true;
  };

  // ------------------------------------------------------------------ files
  const toB64 = (f32) => { const u8 = new Uint8Array(f32.buffer, f32.byteOffset, f32.byteLength); let s = ''; for (let i = 0; i < u8.length; i += 0x8000) s += String.fromCharCode.apply(null, u8.subarray(i, i + 0x8000)); return btoa(s); };
  const fromB64 = (s) => { const bin = atob(s), u8 = new Uint8Array(bin.length); for (let i = 0; i < bin.length; i++) u8[i] = bin.charCodeAt(i); return new Float32Array(u8.buffer, 0, Math.floor(u8.length / 4)); };
  SH.fileOf = (rec) => JSON.stringify({ kind: KIND, v: 1, name: rec.name, date: rec.date, problem: rec.problem, label: rec.label, family: rec.family, method: rec.method, params: rec.params, score: rec.score, step: rec.step, unit: rec.unit, table: rec.table, setup: rec.setup, weights: toB64(rec.weights instanceof Float32Array ? rec.weights : Float32Array.from(rec.weights)) });
  // a file read back: checked (its kind, its numbers, the weights as many as it says), else null and why
  SH.fromFile = function (text) {
    let o;
    try { o = JSON.parse(text); } catch (e) { return { why: 'That is not a model file: it is not JSON.' }; }
    if (!o || o.kind !== KIND || o.v !== 1 || !o.setup || typeof o.weights !== 'string') return { why: 'That is not a Neural Anatomy model file.' };
    if (!(o.params > 0 && o.params <= MAX) || o.weights.length > MAX * 6) return { why: 'That model is too big to open here.' };
    let w;
    try { w = fromB64(o.weights); } catch (e) { return { why: 'Its weights are damaged.' }; }
    if (w.length !== o.params || !w.every(Number.isFinite)) return { why: 'Its weights are damaged.' };
    if (!NA.builderTasks.byId[o.setup.problem]) return { why: 'It is for a problem this page does not have.' };
    const p = NA.builderTasks.byId[o.setup.problem];
    return { rec: { id: `m${Date.now().toString(36)}${Math.random().toString(36).slice(2, 6)}`, auto: false, name: String(o.name || 'A model').slice(0, 80), date: Date.now(), problem: o.setup.problem, label: p.label, family: p.family,
      method: typeof o.method === 'string' ? o.method : null, setup: o.setup, params: o.params, weights: w, score: typeof o.score === 'number' ? o.score : null, step: o.step | 0, unit: typeof o.unit === 'string' ? o.unit.slice(0, 20) : 'step', table: !!o.table } };
  };
  function download(rec) {
    const url = URL.createObjectURL(new Blob([SH.fileOf(rec)], { type: 'application/json' })), safe = rec.name.replace(/[^\w\- ]+/g, '').trim().replace(/\s+/g, '-').slice(0, 50) || 'model';
    const a = el('a', { href: url, download: `${safe}.na-model.json` });
    document.body.append(a); a.click(); a.remove();
    setTimeout(() => URL.revokeObjectURL(url), 2000);
  }

  // ------------------------------------------------------------------ the shelf in the Runs tab
  const when = (t) => { const d = new Date(t); return d.toLocaleDateString(undefined, { day: 'numeric', month: 'short' }) + ' ' + d.toLocaleTimeString(undefined, { hour: '2-digit', minute: '2-digit' }); };
  const scoreOf = (r) => (r.score === null || r.score === undefined ? null : NA.builderTasks.isValue(r.family) ? `R² ${fmt(r.score, 2)}` : fmtPct(r.score, 0));
  SH.render = function (lab, box) {
    const key = [SH.version, lab.run ? `${lab.run.n}:${lab.run.status}:${lab.run.step}` : '', SH.canKeep(lab)].join('|');
    if (box._key === key && box.firstChild) return;
    box._key = key;
    clear(box);
    const name = el('input', { type: 'text', class: 'bd-shelf-name', maxlength: '80', 'aria-label': 'A name for the network', placeholder: SH.canKeep(lab) ? nameOf(lab) : 'Train a network first' });
    const keep = el('button', { class: 'btn btn-primary', type: 'button', disabled: !SH.canKeep(lab), title: SH.canKeep(lab) ? 'Keep this run’s setup and its trained weights in My models, in this browser' : 'Train a network first (a classic model cannot be kept)', onclick: () => {
      const rec = SH.recordOf(lab, name.value.trim());
      if (!rec) return;
      SH.put(rec).then(() => { toast(`Kept “${rec.name}” in My models.`); SH.render(lab, box); }).catch(() => toast('This browser would not keep it (a private window keeps nothing): Download it instead.'));
    } }, 'Keep this network');
    const list = el('div', { class: 'bd-shelf-list', role: 'list' });
    const file = el('input', { type: 'file', accept: '.json,application/json', hidden: '' });
    file.addEventListener('change', () => {
      const f = file.files && file.files[0];
      file.value = '';
      if (!f) return;
      if (f.size > MAX * 8) { toast('That file is too big to be a model.'); return; }
      f.text().then((t) => {
        const r = SH.fromFile(t);
        if (!r.rec) { toast(r.why); return; }
        return SH.put(r.rec).then(() => { toast(`“${r.rec.name}” is in My models: Open puts it in the builder.`); SH.render(lab, box); });
      }).catch(() => toast('That file could not be read.'));
    });
    box.append(el('div', { class: 'bd-shelf-save' }, name, keep), list,
      el('div', { class: 'btn-row' }, el('button', { class: 'btn', type: 'button', onclick: () => file.click() }, 'Load a model file…'), file),
      UI().note('Kept in this browser, with the latest run you paused or finished kept by itself: a reload loses nothing. Clearing this site’s data removes them, so Download keeps a copy, which opens in any browser.'));
    SH.list().then((all) => {
      if (!box.isConnected) return;
      if (!all.length) { list.append(el('p', { class: 'note', text: 'Nothing kept yet.' })); return; }
      for (const r of all) {
        const del = el('button', { class: 'btn btn-quiet', type: 'button', 'aria-label': `Delete ${r.name}` }, 'Delete');
        del.addEventListener('click', () => {
          if (del.dataset.sure !== '1') { del.dataset.sure = '1'; del.textContent = 'Delete it?'; setTimeout(() => { if (del.isConnected) { del.dataset.sure = ''; del.textContent = 'Delete'; } }, 3000); return; }
          SH.remove(r.id).then(() => SH.render(lab, box)).catch(() => {});
        });
        const bits = [r.label, r.method && NA.brl.METHODS[r.method] ? NA.brl.METHODS[r.method].short : null, scoreOf(r), `${fmtInt(r.params)} numbers`, when(r.date)].filter(Boolean).join(' · ');
        list.append(el('div', { class: 'bd-shelf-item' + (r.auto ? ' is-auto' : ''), role: 'listitem' },
          el('div', { class: 'bd-shelf-text' }, el('b', { text: r.name }), el('span', { text: bits })),
          el('div', { class: 'btn-row' },
            el('button', { class: 'btn', type: 'button', 'aria-label': `Open ${r.name}`, onclick: () => SH.open(lab, r) }, 'Open'),
            el('button', { class: 'btn btn-quiet', type: 'button', 'aria-label': `Download ${r.name}`, onclick: () => download(r) }, 'Download'), del)));
      }
    }).catch(() => { if (box.isConnected) list.append(el('p', { class: 'note', text: 'This browser keeps no models here (a private window, or site data blocked): Download and Load a model file still work.' })); });
  };

  // ------------------------------------------------------------------ training while the page is hidden
  const TICK = 'let id = null; onmessage = (e) => { clearInterval(id); id = e.data > 0 ? setInterval(() => postMessage(0), e.data) : null; };';
  SH.backgroundOn = () => NA.store.get('builder.background', true) !== false;
  SH.backgroundSection = function () {
    return UI().section('When the page is hidden',
      UI().seg([{ value: true, label: 'Keep training' }, { value: false, label: 'Pause' }], SH.backgroundOn(), (v) => NA.store.set('builder.background', v === true || v === 'true'), 'When the page is hidden'),
      UI().note('A tab you are not looking at gets no frames, so training would stop there. With Keep training, a timer of its own trains it on in the background, a little more slowly, and the tab’s title says how far it has got.'));
  };
  SH.background = function (lab) {
    if (lab._bg) return;
    lab._bg = true;
    let w = null, title = null;
    const stop = () => { if (w) w.postMessage(0); };
    const tick = () => {
      if (!document.hidden || NA.labHost.lab !== lab) { stop(); return; }
      const r = lab.run;
      if (!lab.training) { if (title && r) document.title = `✓ Run ${r.n} done · ${title}`; stop(); return; }
      const b = lab.budget;
      lab.budget = 60;
      try { lab.runFrame(); } finally { lab.budget = b; }
      if (title && r) document.title = `${r.limit ? `${Math.min(100, Math.round((100 * r.step) / r.limit))}% · ` : ''}training run ${r.n} · ${title}`;
    };
    const start = () => {
      if (w === null) { try { w = new Worker(URL.createObjectURL(new Blob([TICK], { type: 'text/javascript' }))); w.onmessage = tick; } catch (e) { w = false; } }
      if (w) w.postMessage(100);
    };
    document.addEventListener('visibilitychange', () => {
      if (document.hidden) { if (lab.training && SH.backgroundOn() && NA.labHost.lab === lab) { title = document.title; start(); } }
      else { stop(); if (title) { document.title = title; title = null; } }
    });
  };
  NA.shelf = SH;
})(window.NA);
