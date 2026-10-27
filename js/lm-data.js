/* lm-data.js — the tiny language the transformer learns: a 36-word vocabulary, a word-level tokenizer and
   a generator of short stories. Two kinds of story need attention to predict:
     lookup:    "alice likes tea . bob likes milk . alice drinks tea ."  (the last drink is only known by
                looking back at what that person likes)
     agreement: "the cat near the dogs sleeps ."  (the verb agrees with "cat", not the nearer "dogs") */
(function (NA) {
  'use strict';

  const CATEGORIES = [
    { id: 'special', label: 'Markers', words: ['<bos>', '.'] },
    { id: 'name', label: 'Names', words: ['alice', 'bob', 'carol', 'dave', 'emma', 'frank'] },
    { id: 'drink', label: 'Drinks', words: ['tea', 'milk', 'juice', 'coffee', 'water'] },
    { id: 'verb', label: 'People verbs', words: ['likes', 'drinks'] },
    { id: 'det', label: 'Article', words: ['the'] },
    { id: 'adj', label: 'Adjectives', words: ['big', 'small', 'happy', 'sleepy'] },
    { id: 'sg', label: 'One animal', words: ['cat', 'dog', 'bird', 'fox'] },
    { id: 'pl', label: 'Several animals', words: ['cats', 'dogs', 'birds', 'foxes'] },
    { id: 'prep', label: 'Prepositions', words: ['near', 'with'] },
    { id: 'vsg', label: 'Verb for one', words: ['sleeps', 'runs', 'sings'] },
    { id: 'vpl', label: 'Verb for several', words: ['sleep', 'run', 'sing'] },
  ];
  const VOCAB = [], CAT_OF = [];
  CATEGORIES.forEach((c, ci) => c.words.forEach((w) => { VOCAB.push(w); CAT_OF.push(ci); }));
  const ID = Object.fromEntries(VOCAB.map((w, i) => [w, i]));
  const ids = (cat) => CATEGORIES.find((c) => c.id === cat).words.map((w) => ID[w]);
  const W = {
    BOS: ID['<bos>'], DOT: ID['.'], LIKES: ID.likes, DRINKS: ID.drinks, THE: ID.the,
    NAMES: ids('name'), DRINKLIST: ids('drink'), ADJ: ids('adj'), SG: ids('sg'), PL: ids('pl'), PREP: ids('prep'), VSG: ids('vsg'), VPL: ids('vpl'),
  };
  const MAX_LEN = 20;

  // ------------------------------------------------------------------ tokenizer
  // Lower-cases, splits on spaces and keeps "." as its own token. Unknown words are reported, not dropped.
  function tokenize(text) {
    const words = String(text).toLowerCase().replace(/\./g, ' . ').replace(/[^a-z.<>\s]/g, ' ').split(/\s+/).filter(Boolean);
    const out = [], unknown = [];
    for (const w of words) { if (w in ID) out.push(ID[w]); else unknown.push(w); }
    return { ids: out, unknown };
  }
  const detok = (list) => list.map((i) => VOCAB[i]).join(' ');

  // ------------------------------------------------------------------ stories
  function pickDistinct(rng, list, k) { return rng.shuffle(list.slice()).slice(0, k); }

  /* Each story is { ids, roles }. A role marks a position whose next token tests a skill:
       { type: 'lookup', q, src, name, others }  q predicts the drink found at src (named at name)
       { type: 'agree', q, subj, distractor, plural, conflict }  q predicts a verb agreeing with subj */
  /* opts: lookupShare (fraction of lookup stories), twoQuestions (chance a two-person story asks about both) */
  class StoryGen {
    constructor(seed, opts = {}) { this.rng = new NA.RNG(seed); this.o = Object.assign({ lookupShare: 0.8, twoQuestions: 0.5 }, opts); }
    story() { return this.rng.next() < this.o.lookupShare ? this.lookup() : this.animals(); }

    // "N likes D ." for two or three people, sometimes an animal sentence in between, then "N drinks D ."
    lookup() {
      const r = this.rng, k = r.next() < 0.55 ? 2 : 3;
      const names = pickDistinct(r, W.NAMES, k), drinks = pickDistinct(r, W.DRINKLIST, k);
      const out = [W.BOS], roles = [], facts = [];
      const two = k === 2 && r.next() < this.o.twoQuestions;
      const fillerAfter = k === 2 && !two && r.next() < 0.45 ? r.int(2) : -1;
      for (let p = 0; p < k; p++) {
        facts.push({ name: out.length, drink: out.length + 2 });
        out.push(names[p], W.LIKES, drinks[p], W.DOT);
        if (p === fillerAfter) this.sentence(out, roles, true);
      }
      const asked = two ? r.shuffle([0, 1]) : [r.int(k)];
      for (const q of asked) {
        out.push(names[q], W.DRINKS);
        roles.push({ type: 'lookup', q: out.length - 1, src: facts[q].drink, name: facts[q].name, others: facts.filter((f, i) => i !== q).map((f) => f.drink) });
        out.push(drinks[q], W.DOT);
      }
      return { ids: out, roles };
    }

    animals() {
      const out = [W.BOS], roles = [];
      this.sentence(out, roles, false);
      this.sentence(out, roles, false);
      return { ids: out, roles };
    }

    // "the [adj] cat [near the [adj] dogs] sleeps ." The distractor noun usually has the other number.
    sentence(out, roles, short) {
      const r = this.rng, plural = r.next() < 0.5;
      out.push(W.THE);
      if (!short && r.next() < 0.35) out.push(r.pick(W.ADJ));
      const subj = out.length;
      out.push(r.pick(plural ? W.PL : W.SG));
      let distractor = -1, conflict = false;
      if (r.next() < 0.6) {
        out.push(r.pick(W.PREP), W.THE);
        if (!short && r.next() < 0.25) out.push(r.pick(W.ADJ));
        const dPlural = r.next() < 0.7 ? !plural : plural;
        conflict = dPlural !== plural;
        distractor = out.length;
        out.push(r.pick(dPlural ? W.PL : W.SG));
      }
      roles.push({ type: 'agree', q: out.length - 1, subj, distractor, plural, conflict });
      out.push(r.pick(plural ? W.VPL : W.VSG), W.DOT);
    }
  }

  function makeSet(seed, n) {
    const g = new StoryGen(seed), out = [];
    for (let i = 0; i < n; i++) out.push(g.story());
    return out;
  }

  // Prompts offered in the lab; each shows one behaviour.
  const PROMPTS = [
    { id: 'lookup', label: 'Who drinks what', text: 'alice likes tea . bob likes milk . alice drinks', note: 'Only the earlier sentences say what Alice likes. A layer-2 head looks back and copies the drink.' },
    { id: 'lookup3', label: 'Three people', text: 'carol likes juice . dave likes coffee . emma likes water . dave drinks', note: 'Three facts, one question. Change Dave to Emma and the answer follows.' },
    { id: 'agree', label: 'Agreement', text: 'the cat near the dogs', note: 'The verb must agree with "cat", not with the nearer "dogs".' },
    { id: 'agree2', label: 'Plural subject', text: 'the happy foxes with the bird', note: 'Plural subject, singular distractor: the model should prefer sleep / run / sing.' },
    { id: 'new', label: 'New or known name', text: 'bob likes tea . frank likes milk . bob', note: 'Bob was mentioned before, so "drinks" is likely next; a new name would be followed by "likes".' },
  ];

  NA.lmData = { CATEGORIES, VOCAB, CAT_OF, ID, W, MAX_LEN, tokenize, detok, StoryGen, makeSet, PROMPTS };
})(typeof globalThis !== 'undefined' ? (globalThis.NA = globalThis.NA || {}) : (window.NA = window.NA || {}));
