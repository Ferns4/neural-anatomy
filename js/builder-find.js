/* builder-find.js — Find a setting: a search box at the top of the model builder's right panel (/ puts the cursor in
   it). Type a few letters of a setting, a section, a choice or a feature ("grow", "learning rate", "wall", "full
   screen") and pick one: its tab opens (its block selected; on a phone, its screen), and the setting is scrolled to and
   marked for a moment. The index is made from the panels themselves: each place (the Input block, every block of the
   model, the Output block, the Reward tab, the Training tab) is drawn once off the page, and its sections, fields,
   choices, buttons and notes collected, so a setting added later is found with no list to keep up to date. The
   features that are not settings in a panel (full screen, Jump in, editing the Python, the views, challenges…) are
   listed by hand. FD.build(lab) makes the box; FD.index(lab), FD.search(lab, query) and FD.go(lab, hit) are there for
   tests. */
(function (NA) {
  'use strict';
  const FD = {};
  const el = NA.el;
  // text as it is compared: lower case, accents and punctuation gone (Greek letters stay)
  const norm = (t) => String(t || '').toLowerCase().normalize('NFKD').replace(/[̀-ͯ]/g, '').replace(/[‘’'`]/g, '').replace(/[^\p{L}\p{N}]+/gu, ' ').trim();
  // words people type for what the page calls otherwise (each also finds itself)
  const SYN = {
    lr: 'learning rate', eta: 'learning rate', epsilon: 'exploration random moves', explore: 'exploration random moves', exploring: 'exploration', gamma: 'discount',
    pop: 'population', sigma: 'mutation nudge spread', prune: 'grow and prune', pruning: 'grow and prune', grow: 'grow and prune', growing: 'grow and prune',
    neurons: 'neurons units', neuron: 'neurons units', units: 'neurons units', width: 'neurons units', depth: 'hidden layers', python: 'python script', code: 'python script',
    script: 'python script', pytorch: 'python', video: 'full screen', fullscreen: 'full screen', maximise: 'full screen', maximize: 'full screen', bigger: 'full screen',
    play: 'jump in', control: 'jump in', drive: 'jump in', takeover: 'jump in', yourself: 'jump in', points: 'reward', pay: 'reward', paid: 'reward', punish: 'reward',
    epochs: 'steps', iterations: 'steps', generation: 'generations', seed: 'random seed', init: 'initialisation', initialization: 'initialisation', optimiser: 'optimizer',
    regularization: 'regularisation', normalize: 'normalise', normalization: 'normalisation', behavior: 'behaviour', color: 'colour', weights: 'weights weight',
    activation: 'activation relu tanh', relu: 'activation', tanh: 'activation', sigmoid: 'activation', loss: 'loss', error: 'loss', clip: 'clipping', clipping: 'clipping',
    decay: 'weight decay', batch: 'batch size', schedule: 'learning rate schedule', inputs: 'inputs see', sensors: 'inputs noise', noise: 'noise', hide: 'inputs',
    keep: 'keep best', best: 'keep best network', method: 'learns by', algorithm: 'learns by', dqn: 'learns by', ppo: 'learns by', evolution: 'learns by evolution',
    save: 'keep this network my models', store: 'my models', shelf: 'my models', load: 'my models load a model file', open: 'my models open', export: 'download my models', import: 'load a model file', background: 'when the page is hidden',
    hidden: 'when the page is hidden',
    neat: 'neat grows its own network species', topology: 'neat', species: 'neat species', pbt: 'learners population-based training', population: 'population learners', tree: 'family tree of shapes',
    mdn: 'mixture of bell curves', mixture: 'mixture of bell curves', kan: 'kolmogorov arnold', resnet: 'skip connection', residual: 'skip connection', shortcut: 'skip connection', hopfield: 'hopfield memory', memory: 'hopfield memory patterns',
    tau: 'temperature', kl: 'pull toward one bell curve', beta: 'pull toward one bell curve', neighbours: 'from the neighbours', neighbors: 'from the neighbours', aggregation: 'from the neighbours',
    curriculum: 'easy first curriculum', easy: 'easy first', dream: 'model of the world', predator: 'the chaser learns', prey: 'the chaser learns', selfplay: 'the chaser learns',
    dueling: 'dueling', nstep: 'looks ahead', double: 'double dqn', target: 'target refreshed', replay: 'replay memory', lambda: 'gae', gae: 'gae', endings: 'test it 100 times', success: 'test it 100 times',
    custom: 'your own', editor: 'your own', layout: 'your own', design: 'your own', walls: 'wall maze', obstacle: 'hurdles course', obstacles: 'hurdles course', sheep: 'sheep',
    world: 'world settings', gravity: 'gravity', maze: 'maze', track: 'track', formula: 'formula', wall: 'wall', template: 'templates', share: 'share link',
    imitation: 'copy you', cloning: 'copy you', copy: 'copy you', demonstration: 'copy you your moves', demos: 'copy you your moves', rlhf: 'which did better preferences',
    preference: 'which did better', preferences: 'which did better', feedback: 'which did better', compare: 'which did better race', race: 'race', versus: 'race against', vs: 'race against',
    record: 'record a video', recording: 'record a video', film: 'record a video', movie: 'record a video', slow: 'speed', speed: 'speed', pause: 'pause',
    warm: 'start from', pretrained: 'start from', transfer: 'start from', finetune: 'start from', continue: 'start from continue',
  };
  // ------------------------------------------------------------------ the index, from the panels
  // the places a setting can be: the Input block, each block of the model, the Output block, the Reward and Training tabs
  function places(lab) {
    const S = lab.setup, out = [{ id: 'input', tab: 'block', sel: { kind: 'input', id: 'input' }, name: 'Input block' }];
    S.blocks.forEach((b, i) => out.push({ id: 'b:' + b.id, tab: 'block', sel: { kind: 'block', id: b.id }, name: `Block ${i + 1} (${lab.blockName ? lab.blockName(b) : b.type})` }));
    out.push({ id: 'output', tab: 'block', sel: { kind: 'output', id: 'output' }, name: 'Output block' });
    if (lab.family() === 'control') out.push({ id: 'reward', tab: 'reward', name: 'Reward tab' });
    out.push({ id: 'train', tab: 'train', name: 'Training tab' });
    return out;
  }
  // one place drawn off the page, with the lab's state put back exactly as it was
  function drawn(lab, P) {
    const div = el('div'), keep = { sel: lab.sel, parts: lab.parts, tabId: lab.tabId, _blockKey: lab._blockKey, _trainKey: lab._trainKey, _rewardKey: lab._rewardKey };
    try {
      lab.parts = { body: div };
      if (P.tab === 'block') { lab.sel = P.sel; lab._blockKey = null; lab.renderBlockTab(div); }
      else if (P.tab === 'reward') { lab._rewardKey = null; lab.renderRewardTab(div); }
      else { lab._trainKey = null; lab.renderTrainTab(div); }
    } catch (e) { /* a place that cannot be drawn off the page is left out of the index */ }
    finally { Object.assign(lab, keep); }
    return div;
  }
  const textOf = (n) => (n ? n.textContent.replace(/\s+/g, ' ').trim() : '');
  // what a place offers: its sections, and in each its fields (with their choices), sliders, chips and buttons
  function collect(P, div) {
    const out = [], seen = new Map(), head = textOf(div.querySelector('.bd-insp-title'));
    if (head) out.push({ kind: 'place', label: head, section: null, place: P, find: { kind: 'place' } });
    [...div.querySelectorAll('.insp-sec')].forEach((sec) => {
      const title = textOf(sec.querySelector(':scope > .sec-title'));
      if (title === 'What is happening') return; // (a run's diagnosis, not a setting)
      const n = seen.get(title) || 0;
      seen.set(title, n + 1);
      const notes = [...sec.querySelectorAll('.sec-note')].map(textOf).join(' ').slice(0, 600), at = { section: title, sectionN: n };
      if (title) out.push({ kind: 'section', label: title, section: title, place: P, note: notes, find: Object.assign({ kind: 'section' }, at) });
      const count = new Map(), nth = (k) => { const c = count.get(k) || 0; count.set(k, c + 1); return c; };
      for (const f of sec.querySelectorAll('label.field')) {
        const label = textOf(f.querySelector(':scope > span'));
        if (!label) continue;
        const opts = [...f.querySelectorAll('.seg button, option, .chip')].map(textOf).filter(Boolean).join(' · ');
        out.push({ kind: 'field', label, section: title, place: P, options: opts, note: f.title || '', find: Object.assign({ kind: 'field', label, n: nth('f:' + label) }, at) });
      }
      for (const s of sec.querySelectorAll('.slider')) {
        const label = textOf(s.querySelector('.sl-top > span'));
        if (label) out.push({ kind: 'field', label, section: title, place: P, find: Object.assign({ kind: 'slider', label, n: nth('s:' + label) }, at) });
      }
      for (const c of sec.querySelectorAll('.chip')) {
        if (c.closest('label.field')) continue; // (a field's chips are its choices)
        const label = textOf(c);
        if (label) out.push({ kind: 'field', label, section: title, place: P, note: c.title || '', find: Object.assign({ kind: 'chip', label, n: nth('c:' + label) }, at) });
      }
      for (const b of sec.querySelectorAll('button.btn')) {
        const label = textOf(b);
        if (label && label.length < 60) out.push({ kind: 'button', label, section: title, place: P, note: b.title || '', find: Object.assign({ kind: 'button', label, n: nth('b:' + label) }, at) });
      }
    });
    return out;
  }
  // the features that are not settings in a panel
  function features(lab) {
    const phone = () => NA.mobile && NA.mobile.mobile, to = (screen) => { if (phone()) NA.mobile.go(screen); };
    const results = () => { to('network'); if (lab.centerView === 'model' || lab.centerView === 'arch' || lab.centerView === 'growth') { lab.centerView = 'results'; lab.layoutCenter(); } };
    const flashQ = (sel) => setTimeout(() => lab.flash(document.querySelector(sel)), 60);
    const ctl = lab.family() === 'control';
    return [
      { label: 'Full screen', where: 'the results', words: 'full screen video maximise bigger picture watch', go: () => { results(); lab.setFull(true); } },
      ctl ? { label: 'Pause, step and speed', where: 'the results', words: 'pause play step one move back on slow speed slow motion frame scrub rewind new episode', go: () => { results(); flashQ('.bd-vid'); } } : null,
      ctl ? { label: 'Record a video', where: 'the results', words: 'record a video save film movie webm mp4 clip', go: () => { results(); flashQ('.bd-vid-rec'); } } : null,
      ctl ? { label: 'Race two side by side', where: 'the results', words: 'race against side by side versus you against a run two runs', go: () => { results(); flashQ('.bd-vid-race'); } } : null,
      ctl ? { label: 'Jump in: play it yourself', where: 'the results', words: 'jump in play yourself take over controls keys drive steer', go: () => { results(); flashQ('.bd-play .btn'); } } : null,
      { label: 'Edit the Python', where: 'Python tab', words: 'edit python script code pytorch change apply', go: () => { to('insp'); lab.tabs.select('python'); if (NA.builderPyEdit) NA.builderPyEdit.startEdit(lab); } },
      { label: 'Download the Python', where: 'Python tab', words: 'download python script save file', go: () => { to('insp'); lab.tabs.select('python'); flashQ('#labRight .bd-pe-tools'); } },
      { label: 'Architecture view', where: 'the middle', words: 'architecture zoom inside neurons transistors diagram boxes network', go: () => { to('network'); lab.centerView = 'arch'; lab.layoutCenter(); } },
      { label: 'Growth view: how it learned', where: 'the middle', words: 'growth film learned history math steps curve', go: () => { to('network'); lab.centerView = 'growth'; lab.layoutCenter(); } },
      { label: 'Results', where: 'the middle', words: 'results picture plays predictions', go: () => results() },
      ctl ? { label: 'Value map', where: 'the results', words: 'value map heatmap heat map critic q-values policy map inputs across up', go: () => { results(); lab.evoView = 'map'; lab._headKey = null; lab.resultsDirty = true; } } : null,
      ctl ? { label: 'Where it went while learning', where: 'the results', words: 'visits visited where it went heatmap exploration explored places', go: () => { results(); lab.evoView = 'visits'; lab._headKey = null; lab.resultsDirty = true; } } : null,
      { label: 'My models: keep and open networks', where: 'Runs tab', words: 'my models keep save store load open download export import file weights shelf reload', go: () => { to('insp'); lab.tabs.select('runs'); flashQ('#labRight .bd-shelf'); } },
      { label: 'Compare runs', where: 'Runs tab', words: 'compare runs history earlier chart', go: () => { to('insp'); lab.tabs.select('runs'); } },
      { label: 'Challenges', where: 'left panel', words: 'challenges goals levels tasks', go: () => lab.openChallenges() },
      { label: 'Problem', where: 'left panel', words: 'problem data dataset world task choose', go: () => { to('data'); flashQ('#labLeft select'); } },
      // the palette's parts, each opening its guide (one of each)
      ...(typeof document !== 'undefined' ? [...document.querySelectorAll('.bd-part[data-type]')] : []).filter((t, i, all) => all.findIndex((u) => u.dataset.type === t.dataset.type) === i)
        .map((t) => ({ kind: 'part', label: `${(t.querySelector('.bd-part-name') || t).textContent} (a part)`, where: 'Parts', words: `${t.dataset.type} ${t.title || ''} ${((NA.partGuide && NA.partGuide.G[t.dataset.type]) || {}).what || ''}`, go: () => { to('insp'); lab.showGuide(t.dataset.type); } })),
      // the Analyse tab's sections, Guess first, the paths through the challenges
      ...[['Sweep a setting', 'sweep batch size weight decay optimizer width compare values settings'], ['Luck: several seeds', 'seeds luck variance spread repeat random chance'],
        ['How sure is it? (calibration)', 'calibration confidence sure ece temperature scaling honest'], ['Which inputs matter', 'feature importance permutation inputs matter features'],
        ['Fool it: adversarial examples', 'adversarial fgsm fool attack perturbation robust nudge'], ['Pruning and quantising', 'pruning quantise quantize compress smaller bits size weights', 'Make it smaller: prune and quantise']].map(([label, words, sec]) => ({ label, where: 'Analyse tab', words, go: () => { to('insp'); lab.tabs.select('analyse'); setTimeout(() => lab.flashSection(sec || label), 60); } })),
      { label: 'Guess first', where: 'Runs tab', words: 'guess predict before you run expect forecast intuition', go: () => { to('insp'); lab.tabs.select('runs'); setTimeout(() => lab.flashSection('Guess first'), 60); } },
      { label: 'Paths through the challenges', where: 'Challenges', words: 'paths path guided lessons course order next challenge', go: () => lab.openChallenges() },
      // the kinds of problem of version 31: each opens one of its problems
      ...[['autoenc', 'Squeeze and rebuild: autoencoders and the VAE', 'autoencoder vae variational bottleneck code latent encoder decoder denoise denoising draw new pictures'],
        ['ganring', 'Make new examples: a GAN', 'gan generative adversarial generator discriminator mode collapse fake'],
        ['contrast', 'Learn without labels: contrastive learning', 'contrastive self-supervised simclr unlabelled unlabeled views augmentation embedding nt-xent'],
        ['segment', 'Label every pixel: segmentation', 'segmentation segment u-net unet pixel mask upsample'],
        ['groups', 'Graphs: message passing', 'graph gnn message passing nodes edges links friends community'],
        ['noisy', 'Predict with uncertainty', 'uncertainty spread confidence error bars gaussian heteroscedastic calibration band']].map(([id, label, words]) => ({ label, where: 'problem', words, go: () => { lab.setProblem(id); to('data'); } })),
      { label: 'Templates: start from', where: 'left panel', words: 'templates start from mlp cnn transformer linear', go: () => { to('data'); flashQ('#labLeft .bd-tpl'); } },
      { label: 'Parts palette', where: 'left panel', words: 'parts palette add layer block dense conv', go: () => { to('data'); flashQ('#labLeft .bd-palette'); } },
      { label: 'Share a link', where: 'top bar', words: 'share link url send copy', go: () => flashQ('#shareBtn') },
      { label: 'Undo', where: 'the middle', words: 'undo back revert', go: () => { to('network'); flashQ('#labCenter .icon-btn[aria-label="Undo"]'); } },
      { label: 'The Guide', where: 'top bar', words: 'guide help explain learn chapters', go: () => NA.guide && NA.guide.show() },
    ].filter(Boolean).map((f) => Object.assign({ kind: 'feature', section: null, note: f.words, place: { name: f.where } }, f));
    // (a part: below the settings and features that match as well)
  }
  /* The whole index for the setup as it is (kept until the setup, the problem or the method changes): every place
     drawn and collected, then the features. */
  FD.index = function (lab) {
    const key = JSON.stringify([lab.snapshot(), lab.family()]);
    if (lab._findIdx && lab._findIdx.key === key) return lab._findIdx.list;
    const list = [];
    for (const P of places(lab)) list.push(...collect(P, drawn(lab, P)));
    list.push(...features(lab));
    lab._findIdx = { key, list };
    return list;
  };
  // ------------------------------------------------------------------ the search
  // how well one word (or its other names) matches an entry: its label most, then its section, place, choices, notes
  function wordScore(e, alts) {
    let best = 0;
    for (const a of alts) {
      const at = (txt, w) => { if (!txt) return 0; const t = ' ' + txt + ' '; if (a.includes(' ')) return t.includes(' ' + a + ' ') ? w : 0; return t.includes(' ' + a) ? w : t.includes(a) && a.length >= 3 ? w * 0.5 : 0; };
      best = Math.max(best, at(e._l, 10), at(e._s, 4), at(e._p, 3), at(e._o, 3), at(e._n, 1));
    }
    return best;
  }
  const RANK = { field: 0, feature: 1, section: 2, button: 3, place: 4, part: 5 };
  FD.search = function (lab, q) {
    const words = norm(q).split(' ').filter(Boolean);
    if (!words.length) return [];
    const list = FD.index(lab);
    for (const e of list) if (e._l === undefined) { e._l = norm(e.label); e._s = norm(e.section); e._p = norm(e.place && e.place.name); e._o = norm(e.options); e._n = norm(e.note); }
    const out = [];
    for (const e of list) {
      let score = 0, all = true;
      for (const w of words) { const s = wordScore(e, [w, ...(SYN[w] ? [SYN[w]] : [])]); if (!s) { all = false; break; } score += s; }
      if (!all) continue;
      if (e._l === words.join(' ')) score += 20;
      if (e.kind === 'part') score *= 0.39; // (parts of the palette after the settings and features that match)
      out.push({ e, score });
    }
    out.sort((a, b) => b.score - a.score || RANK[a.e.kind] - RANK[b.e.kind] || a.e.label.length - b.e.label.length);
    // one of each (label, where)
    const seen = new Set(), hits = [];
    for (const { e } of out) { const k = e.label + '|' + whereOf(e); if (seen.has(k)) continue; seen.add(k); hits.push(e); if (hits.length >= 8) break; }
    return hits;
  };
  const whereOf = (e) => (e.kind === 'feature' ? e.place.name : [e.place.name, e.kind === 'section' || e.kind === 'place' ? null : e.section].filter(Boolean).join(' › '));
  // ------------------------------------------------------------------ going there
  function target(body, f) {
    if (f.kind === 'place') return body.querySelector('.bd-insp-head') || body.firstElementChild;
    const secs = [...body.querySelectorAll('.insp-sec')].filter((s) => textOf(s.querySelector(':scope > .sec-title')) === f.section), sec = secs[f.sectionN || 0];
    if (!sec || f.kind === 'section') return sec || null;
    const pick = (els) => els[f.n || 0] || els[0] || sec;
    if (f.kind === 'field') return pick([...sec.querySelectorAll('label.field')].filter((x) => textOf(x.querySelector(':scope > span')) === f.label));
    if (f.kind === 'slider') return pick([...sec.querySelectorAll('.slider')].filter((x) => textOf(x.querySelector('.sl-top > span')) === f.label));
    if (f.kind === 'chip') return pick([...sec.querySelectorAll('.chip')].filter((x) => textOf(x) === f.label));
    return pick([...sec.querySelectorAll('button.btn')].filter((x) => textOf(x) === f.label));
  }
  FD.go = function (lab, e) {
    if (e.kind === 'feature') { e.go(); return true; }
    if (NA.mobile && NA.mobile.mobile) NA.mobile.go('insp');
    const P = e.place;
    if (P.tab === 'block') { lab.selectBlock(P.sel, true); if (lab.tabs.current !== 'block') lab.tabs.select('block'); }
    else if (lab.tabs.current !== P.tab) lab.tabs.select(P.tab);
    lab.updateTab(); // (drawn now, so the setting is there to scroll to)
    const t = target(lab.tabs.body, e.find);
    if (t) lab.flash(t);
    return !!t;
  };
  // ------------------------------------------------------------------ the box
  FD.build = function (lab) {
    const id = 'bd-find-' + Math.random().toString(36).slice(2, 7);
    const input = el('input', { type: 'search', class: 'bd-find-in', placeholder: 'Find a setting…', 'aria-label': 'Find a setting (press / to search)', autocomplete: 'off', spellcheck: 'false', role: 'combobox', 'aria-expanded': 'false', 'aria-controls': id, 'aria-autocomplete': 'list', enterkeyhint: 'go' });
    const list = el('ul', { class: 'bd-find-list', id, role: 'listbox', 'aria-label': 'Settings found', hidden: '' });
    const box = el('div', { class: 'bd-find', role: 'search' }, el('span', { class: 'bd-find-ico', 'aria-hidden': 'true' }, '⌕'), input, list);
    let hits = [], active = -1;
    const close = () => { list.hidden = true; input.setAttribute('aria-expanded', 'false'); input.removeAttribute('aria-activedescendant'); active = -1; };
    const choose = (k) => { const e = hits[k]; if (!e) return; close(); input.value = ''; FD.go(lab, e); };
    const mark = (k) => { active = k; [...list.children].forEach((li, i) => li.setAttribute('aria-selected', String(i === k))); if (k >= 0 && list.children[k]) { input.setAttribute('aria-activedescendant', list.children[k].id); list.children[k].scrollIntoView({ block: 'nearest' }); } };
    const show = () => {
      const q = input.value;
      hits = q.trim() ? FD.search(lab, q) : [];
      NA.clear(list);
      if (!q.trim()) { close(); return; }
      if (!hits.length) {
        list.append(el('li', { class: 'bd-find-none', role: 'presentation' }, el('span', { text: `Nothing called “${q.trim()}” here.` }),
          NA.guide ? el('button', { class: 'btn btn-quiet', type: 'button', onclick: () => { close(); NA.guide.show(null, q.trim()); } }, 'Search the Guide') : null));
      } else hits.forEach((e, k) => {
        const li = el('li', { id: `${id}-${k}`, role: 'option', 'aria-selected': 'false' }, el('span', { class: 'bd-find-label', text: e.label }), el('span', { class: 'bd-find-where', text: whereOf(e) }));
        li.addEventListener('pointerdown', (ev) => { ev.preventDefault(); choose(k); });
        list.append(li);
      });
      list.hidden = false; input.setAttribute('aria-expanded', 'true');
      mark(hits.length ? 0 : -1);
    };
    input.addEventListener('input', show);
    input.addEventListener('focus', () => { if (input.value.trim()) show(); });
    input.addEventListener('blur', () => setTimeout(close, 120));
    input.addEventListener('keydown', (e) => {
      if (e.key === 'ArrowDown' && hits.length) { e.preventDefault(); mark((active + 1) % hits.length); }
      else if (e.key === 'ArrowUp' && hits.length) { e.preventDefault(); mark((active - 1 + hits.length) % hits.length); }
      else if (e.key === 'Enter') { e.preventDefault(); choose(active >= 0 ? active : 0); }
      else if (e.key === 'Escape') { e.stopPropagation(); if (input.value) { input.value = ''; close(); } else input.blur(); }
    });
    box.focusInput = () => input.focus();
    return box;
  };
  NA.builderFind = FD;
})(window.NA);
