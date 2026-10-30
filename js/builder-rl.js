/* builder-rl.js — reinforcement learning in the model builder: the network you build, trained by reward in one of
   the worlds of rl-envs.js (and rl-worlds*.js) with one of ten methods. DQN learns a Q-value for each action from a
   replay memory, with a target network. REINFORCE raises the probability of each action by the return that
   followed it. Actor-critic (A2C) adds an output that estimates how good each state is, and learns from the
   surprise of every few steps in 8 copies of the world at once. PPO takes several small steps on each batch of
   experience, its clip stopping any step from pushing the policy far from the one that played. A2C and PPO can see
   their rewards normalised (divided by the running spread of the discounted return), so a world that pays in
   hundreds learns like one that pays in ones. Five methods use no gradients at all (builder-evo.js):
   a genetic algorithm breeds each generation from the best of the last (and can grow and prune the network
   itself), novelty search breeds from the ones that behaved most unlike anything seen before, evolution
   strategies nudge one network in many random directions and step toward those that did better, the
   cross-entropy method draws networks from a bell curve over the weights and refits it to the best, and
   MAP-Elites keeps the best network of every style of play. ERL mixes the two kinds: a genetic algorithm whose
   moves teach a DQN, which joins the population every few generations. Each trains one step at a time (step();
   the evolutionary ones a generation, spread over several calls when a deadline is given) and is scored by how
   its best moves do, from 0 to 1 (score()). Runs the same in the page and in node. */
(function (NA) {
  'use strict';
  const BRL = {};
  const E = () => NA.rlEnvs;

  BRL.METHODS = {
    dqn: { label: 'DQN (Q-learning)', short: 'DQN', extra: 0, unit: 'update', hint: 'One output per action: the reward it expects in the end if it takes that action now. It learns from a replay memory of past moves, toward reward + γ × the next state’s value, judged by a target network (a copy refreshed every 500 updates). It explores at random at first, less and less.' },
    pg: { label: 'REINFORCE (policy gradient)', short: 'REINFORCE', extra: 0, unit: 'update', hint: 'One output per action: softmax turns them into the probability of playing it. It plays whole episodes until it has at least 64 moves (one long episode, or many short ones: 64 pulls of the levers), then each action’s probability goes up by how much better than usual the return after it was, and down if worse.' },
    a2c: { label: 'Actor-critic (A2C)', short: 'Actor-critic', extra: 1, unit: 'update', hint: 'The policy’s outputs plus one more: how good the current state is (the critic). It plays 8 steps in 8 copies of the world at once, then learns from the advantage: the rewards it got plus the value where it ended up, minus what the critic expected.' },
    ppo: { label: 'PPO (proximal policy optimisation)', short: 'PPO', extra: 1, unit: 'update', hint: 'Like actor-critic, but it plays 128 moves, then takes 16 small steps on them. Once the probability of a move has changed by 20% from the policy that played it, the clip stops pushing it further (steps for other moves can still carry it a little past), so the policy changes in small, steady steps. The workhorse of modern RL, from robots to chatbots.' },
    ga: { label: 'Genetic algorithm', short: 'Genetic algorithm', extra: 0, unit: 'generation', evo: true, hint: 'No gradients and no learning rate: a population of copies of the network, each a little different, play; the best two pass on unchanged and the rest are children of the better ones, mixed and mutated. Turn on Grow and prune and children can also gain or lose neurons, and gain new layers that start with a single neuron. One step is one generation.' },
    es: { label: 'Evolution strategies', short: 'Evolution strategies', extra: 0, unit: 'generation', evo: true, hint: 'One network, nudged in many random directions at once (in pairs: plus a nudge and minus it). How each nudged copy does says which way is uphill: an estimate of the gradient made from scores alone, with no backpropagation. Adam then takes a step that way. One step is one generation.' },
    me: { label: 'MAP-Elites (many styles)', short: 'MAP-Elites', extra: 0, unit: 'generation', evo: true, hint: 'A grid of 12 × 12 styles of play, from two numbers the world measures (on the race track: how fast it drives and which side of the track it keeps). Each new network goes to the cell of its style and stays if it beats the one there: it finds the best of every style, not one best. One step is one batch of new networks.' },
    novelty: { label: 'Novelty search', short: 'Novelty search', extra: 0, unit: 'generation', evo: true, hint: 'Like the genetic algorithm, but a network is chosen for how new its behaviour is, not for its reward: where it ended up or how it played (two numbers the world measures), compared with its generation and with an archive of what earlier ones did. It never aims at the goal, so a dead end that looks like progress cannot trap it (try the deceptive maze). The network on show is the best by score found so far. One step is one generation.' },
    cem: { label: 'Cross-entropy method', short: 'Cross-entropy', extra: 0, unit: 'generation', evo: true, hint: 'A bell curve over every weight: its middle and its spread. Each generation draws networks from it; the best fifth set the new middle and spread, so the spread shrinks where good networks agree and stays wide where they do not, with a floor so it never stops looking. The middle is the network on show. One step is one generation.' },
    erl: { label: 'ERL (evolution and DQN)', short: 'ERL', extra: 0, unit: 'generation', evo: true, hint: 'A genetic algorithm and a DQN together: every move the population makes goes into a replay memory, and a DQN (a network like yours) learns Q-values from it by gradient steps between generations. Every few generations the DQN’s network joins the population in place of the worst member, and breeds if it plays well. One step is one generation.' },
  };
  BRL.ORDER = ['dqn', 'pg', 'a2c', 'ppo', 'ga', 'novelty', 'es', 'cem', 'me', 'erl'];
  // the methods that breed a population, one generation a step (builder-evo.js): no gradients, except ERL's DQN
  BRL.isEvo = (m) => !!(BRL.METHODS[m] && BRL.METHODS[m].evo);
  BRL.outputs = (method, nA) => nA + (BRL.METHODS[method] || BRL.METHODS.dqn).extra;
  /* Per world: the discount (γ: how much a reward one step later is worth), how many moves DQN explores for (its
     chance of a random move falls from 1 to 0.05 over them), how many episodes judge a member of an evolving
     population, how many steps a press of Train runs for each method (a generation is one step of the
     evolutionary methods, so they need far fewer), the return of a good episode (scale: grow and prune's size
     penalty is a share of it), the population an evolving method starts with, and its mutations' size (sigma) and
     way of mixing parents where they differ from the usual. Both measured from the page's own start (the network
     built with seed 1, runs 1 to 10): on the track, σ 0.5 with copies alone got every run round 4 laps by generation
     125, most by 75, where σ 0.2 with mixing left 2 of 7 stuck at the esses for 200; on the rocket, copies alone
     found a landing sooner, by generation 100 in 11 of 20 runs against 2 of 20 with mixing, both about 4 in 5 by 300. */
  BRL.WORLD = {
    levers: { gamma: 0, explore: 300, gaEpisodes: 8, scale: 1, pop: 24, steps: { dqn: 1000, pg: 300, a2c: 300, ppo: 1000, ga: 300, es: 300, me: 300, novelty: 300, cem: 300, erl: 300 } },
    maze: { gamma: 0.97, explore: 4000, gaEpisodes: 1, scale: 1, pop: 24, steps: { dqn: 10000, pg: 1000, a2c: 2000, ppo: 2000, ga: 300, es: 300, me: 300, novelty: 300, cem: 300, erl: 300 } },
    pole: { gamma: 0.95, explore: 3000, gaEpisodes: 2, scale: 200, pop: 24, steps: { dqn: 5000, pg: 2000, a2c: 3000, ppo: 3000, ga: 300, es: 300, me: 300, novelty: 300, cem: 300, erl: 300 } },
    catch: { gamma: 0.9, explore: 1500, gaEpisodes: 10, scale: 1, pop: 24, steps: { dqn: 5000, pg: 2000, a2c: 3000, ppo: 3000, ga: 300, es: 300, me: 300, novelty: 300, cem: 300, erl: 300 } },
    track: { gamma: 0.99, explore: 20000, gaEpisodes: 1, scale: 40, pop: 48, sigma: 0.5, cross: 'none', steps: { dqn: 30000, pg: 2000, a2c: 10000, ppo: 30000, ga: 150, es: 150, me: 150, novelty: 150, cem: 150, erl: 150 } },
    arena: { gamma: 0.97, explore: 20000, gaEpisodes: 2, scale: 10, pop: 24, steps: { dqn: 30000, pg: 2000, a2c: 6000, ppo: 6000, ga: 80, es: 100, me: 100, novelty: 100, cem: 100, erl: 80 } },
    lander: { gamma: 0.99, explore: 15000, gaEpisodes: 3, scale: 4, pop: 48, sigma: 0.5, cross: 'none', steps: { dqn: 50000, pg: 3000, a2c: 20000, ppo: 30000, ga: 300, es: 200, me: 150, novelty: 200, cem: 200, erl: 300 } },
    // the worlds of rl-worlds2.js and rl-worlds3.js
    thermo: { gamma: 0.97, explore: 3000, gaEpisodes: 2, scale: 15, pop: 24, steps: { dqn: 8000, pg: 1500, a2c: 3000, ppo: 4000, ga: 100, es: 150, me: 150, novelty: 150, cem: 150, erl: 100 } },
    hurdles: { gamma: 0.97, explore: 4000, gaEpisodes: 2, scale: 2, pop: 24, steps: { dqn: 10000, pg: 2000, a2c: 4000, ppo: 6000, ga: 80, es: 120, me: 120, novelty: 120, cem: 120, erl: 80 } },
    memory: { gamma: 0.95, explore: 2000, gaEpisodes: 4, scale: 1, pop: 24, steps: { dqn: 5000, pg: 1500, a2c: 3000, ppo: 3000, ga: 100, es: 150, me: 100, novelty: 100, cem: 150, erl: 100 } },
    mountain: { gamma: 0.99, explore: 10000, gaEpisodes: 2, scale: 100, pop: 24, steps: { dqn: 20000, pg: 2000, a2c: 6000, ppo: 10000, ga: 60, es: 100, me: 100, novelty: 60, cem: 100, erl: 60 } },
    swingup: { gamma: 0.98, explore: 8000, gaEpisodes: 2, scale: 200, pop: 24, steps: { dqn: 20000, pg: 3000, a2c: 8000, ppo: 10000, ga: 150, es: 200, me: 150, novelty: 150, cem: 200, erl: 150 } },
    arm: { gamma: 0.97, explore: 8000, gaEpisodes: 3, scale: 2, pop: 24, steps: { dqn: 15000, pg: 2000, a2c: 6000, ppo: 8000, ga: 150, es: 200, me: 150, novelty: 150, cem: 200, erl: 150 } },
    trap: { gamma: 0.99, explore: 15000, gaEpisodes: 1, scale: 2, pop: 48, steps: { dqn: 20000, pg: 2000, a2c: 6000, ppo: 10000, ga: 150, es: 150, me: 200, novelty: 150, cem: 150, erl: 150 } },
    chase: { gamma: 0.98, explore: 15000, gaEpisodes: 2, scale: 6, pop: 24, steps: { dqn: 20000, pg: 3000, a2c: 8000, ppo: 10000, ga: 300, es: 200, me: 200, novelty: 300, cem: 200, erl: 300 } },
    traffic: { gamma: 0.95, explore: 5000, gaEpisodes: 1, scale: 30, pop: 24, steps: { dqn: 10000, pg: 1500, a2c: 4000, ppo: 5000, ga: 80, es: 120, me: 120, novelty: 120, cem: 120, erl: 80 } },
    walker: { gamma: 0.98, explore: 20000, gaEpisodes: 1, scale: 5, pop: 48, steps: { dqn: 30000, pg: 3000, a2c: 10000, ppo: 20000, ga: 150, es: 200, me: 200, novelty: 200, cem: 200, erl: 150 } },
    herd: { gamma: 0.98, explore: 20000, gaEpisodes: 2, scale: 5, pop: 24, steps: { dqn: 30000, pg: 3000, a2c: 10000, ppo: 20000, ga: 150, es: 200, me: 200, novelty: 200, cem: 200, erl: 150 } },
  };
  const W = (world) => BRL.WORLD[world] || BRL.WORLD.pole;
  BRL.stepsFor = (world, method) => W(world).steps[method] || 3000;
  /* The learning settings a problem starts with. normReward: A2C and PPO see each reward divided by the running
     spread of the discounted return (DQN and REINFORCE as they are). Grow and prune: new layers start with one neuron (layerSize),
     passing on what the next layer leans on most (newLayer: 'identity') or at random; it starts from the network you
     built, may add and prune neurons and add layers (not take a whole layer out at once, growCutLayer, which is left
     out here so that whole layers bring it with them: builder-evo.js EV.growOptions; nor change activations), with
     even odds, a quarter of the children changing, new layers anywhere, new neurons silent, the weakest neuron
     pruned, at most 6 layers of 48, no protection for new shapes. Novelty search: its k
     nearest neighbours, novelty alone or mixed with the reward, behaviours kept a generation. The cross-entropy
     method: the share of each generation that refits it (the elite), its starting spread and the floor under it.
     ERL: the DQN's gradient steps a generation, how often its network joins the population, its learning rate. */
  BRL.defaults = (world) => ({ method: 'dqn', gamma: W(world).gamma, explore: W(world).explore, normReward: true, pop: W(world).pop || 24, sigma: W(world).sigma || 0.2,
    select: 'tournament', cross: W(world).cross || 'uniform', mutate: 'plain', grow: false, penalty: 'light', newLayer: 'identity', layerSize: 'one', growStart: 'built', growAddNeuron: true,
    growCutNeuron: true, growAddLayer: true, growAct: false, growLean: 'even', growRate: 0.25, growWhere: 'any', growNeuron: 'silent', growPrune: 'weak', growMaxLayers: 6, growMaxUnits: 48, growProtect: false, esSigma: 0.05, esLr: 0.03,
    novK: 10, novMix: 'novelty', novAdd: 2, cemElite: 0.2, cemSigma: 0.3, cemFloor: 0.05, erlSteps: 200, erlEvery: 5, erlLr: 0.001 });

  const argmax = (o, base, n) => { let a = 0; for (let k = 1; k < n; k++) if (o[base + k] > o[base + a]) a = k; return a; };
  const softmax = (o, base, n, p) => { let m = -Infinity; for (let k = 0; k < n; k++) m = Math.max(m, o[base + k]); let s = 0; for (let k = 0; k < n; k++) { p[k] = Math.exp(o[base + k] - m); s += p[k]; } for (let k = 0; k < n; k++) p[k] /= s; return p; };
  const pick = (p, n, rng) => { let u = rng.next(); for (let k = 0; k < n; k++) { u -= p[k]; if (u <= 0) return k; } return n - 1; };
  BRL.softmax = (o, n) => softmax(o, 0, n, new Float64Array(n));
  BRL.argmax = (o, n) => argmax(o, 0, n);

  /* A trainer: the world (by id and settings), the network built from your blocks (and the blocks, for evolution
     to reshape), a maker of more networks like it (the target network), the method, its settings (s: gamma,
     explore, batch; pop, sigma and the rest of evolution's) and a seed. */
  class Trainer {
    constructor(o) {
      Object.assign(this, { worldId: o.world, ws: o.ws || {}, net: o.net, makeNet: o.makeNet, blocks: o.blocks || null, method: BRL.METHODS[o.method] ? o.method : 'dqn', rng: new NA.RNG(o.seed || 1) });
      const s = Object.assign(BRL.defaults(this.worldId), o.s);
      this.s = s;
      this.gamma = s.gamma; this.explore = s.explore; this.popSize = Math.max(4, Math.round(s.pop)); this.sigma = s.sigma; this.B = Math.max(4, Math.round(s.batch || 32));
      // actor-critic and PPO play in 8 copies of the world at once; the others in one
      const N = this.method === 'a2c' || this.method === 'ppo' ? 8 : 1;
      this.envs = NA.range(N).map(() => ({ w: E().make(this.worldId, this.ws), obs: null, ret: 0, disc: 0 }));
      for (const e of this.envs) e.obs = e.w.reset(this.rng);
      this.world = this.envs[0].w; this.nA = this.world.nA; this.inSize = this.world.obsSize; this.K = BRL.outputs(this.method, this.nA);
      this.rl = true; this.updates = 0; this.envSteps = 0; this.episodes = 0; this.retEma = null; this.lastEp = null;
      // A2C and PPO: rewards normalised by the running spread of the discounted return (its count, mean and sum of squares)
      this.norm = N > 1 && s.normReward !== false; this.rn = { n: 0, mean: 0, m2: 0 };
      this.p = new Float64Array(this.K); this.x1 = new Float32Array(this.inSize); this.xN = new Float32Array(N * this.inSize);
      this.init();
    }
    get unit() { return BRL.METHODS[this.method].unit; }
    get obs() { return this.envs[0].obs; }
    // the network's outputs for one observation (a copy)
    outputs(obs, net = this.net) { this.x1.set(obs); return Float64Array.from(net.forward(this.x1, 1, false).subarray(0, this.K)); }
    // its best move: the largest Q-value, or the most likely action (net: another network, a member of a population)
    policy(obs, net = this.net) { this.x1.set(obs); return argmax(net.forward(this.x1, 1, false), 0, this.nA); }
    // judged in a world of its own, so the episode it is training in carries on undisturbed
    score() { this.judge = this.judge || E().make(this.worldId, this.ws); return this.judge.score((obs) => this.policy(obs), new NA.RNG(777)); }
    /* The reward the learner is given: the world's, or (normReward, A2C and PPO) divided by the running spread of the
       discounted return of every copy of the world, clipped to ±10, as gym's NormalizeReward does: a world that pays
       in hundreds then learns like one that pays in ones, and the critic's big values no longer crowd out the policy
       in a clipped gradient. The return the page shows stays the world's. */
    scaled(e, r, done) {
      if (!this.norm) return r;
      e.disc = e.disc * this.gamma + r;
      const st = this.rn, d = e.disc - st.mean;
      st.n++; st.mean += d / st.n; st.m2 += d * (e.disc - st.mean);
      if (done) e.disc = 0;
      const sd = st.n > 1 ? Math.sqrt(st.m2 / (st.n - 1)) : 1;
      return Math.max(-10, Math.min(10, r / (sd + 1e-8)));
    }
    // one move in a world, learning nothing; ends and starts episodes
    move(a, e = this.envs[0]) {
      const t = e.w.step(a, this.rng);
      this.envSteps++; e.ret += t.r;
      if (t.done) { this.episodes++; this.lastEp = e.ret; this.retEma = this.retEma === null ? e.ret : 0.95 * this.retEma + 0.05 * e.ret; e.ret = 0; e.obs = e.w.reset(this.rng); }
      else e.obs = t.obs;
      return t;
    }
    init() {
      const m = this.method;
      if (m === 'dqn') { this.cap = 10000; this.buf = []; this.next = 0; this.target = this.makeNet(); this.target.setWeights(this.net.getWeights()); }
      if (BRL.isEvo(m)) this.evo = new NA.bevo.Evolution(this, this.s, this.blocks);
      if (m === 'ppo') this.queue = [];
    }
    // one step; an evolutionary method given a deadline (in performance.now() ms) may answer { partial } until its generation is judged
    step(opt, deadline) {
      const r = this.evo ? this.evo.step(deadline) : this.method === 'dqn' ? this.dqn(opt) : this.method === 'pg' ? this.pg(opt) : this.method === 'a2c' ? this.a2c(opt) : this.ppo(opt);
      if (!r.partial) this.updates++;
      return r;
    }

    // ---------------------------------------------------------------- DQN
    eps() { return this.explore > 0 ? Math.max(0.05, 1 - this.envSteps / this.explore) : 0; }
    dqn(opt) {
      const o = Float32Array.from(this.obs), a = this.rng.next() < this.eps() ? this.rng.int(this.nA) : this.policy(o), t = this.move(a);
      const e = { o, a, r: t.r, o2: t.done ? null : Float32Array.from(t.obs) };
      if (this.buf.length < this.cap) this.buf.push(e); else { this.buf[this.next] = e; this.next = (this.next + 1) % this.cap; }
      if (this.buf.length < 64) return { loss: 0, gradNorm: 0 };
      const B = this.B, K = this.K, X = new Float32Array(B * this.inSize), X2 = new Float32Array(B * this.inSize), batch = [];
      for (let b = 0; b < B; b++) { const q = this.buf[this.rng.int(this.buf.length)]; batch.push(q); X.set(q.o, b * this.inSize); if (q.o2) X2.set(q.o2, b * this.inSize); }
      // the target: reward + γ × the next state's value. Double DQN: the network picks the next move, the target
      // network says what it is worth (one network doing both overrates moves it happens to overrate)
      const y = new Float64Array(B);
      if (this.gamma) {
        const q2 = Float64Array.from(this.target.forward(X2, B, false)), pickNext = this.net.forward(X2, B, false);
        for (let b = 0; b < B; b++) y[b] = batch[b].r + (batch[b].o2 ? this.gamma * q2[b * K + argmax(pickNext, b * K, this.nA)] : 0);
      } else for (let b = 0; b < B; b++) y[b] = batch[b].r;
      const r = this.net.stepWith(X, B, (out, d) => {
        let loss = 0;
        for (let b = 0; b < B; b++) { const i = b * K + batch[b].a, err = out[i] - y[b], ae = Math.abs(err); loss += ae <= 1 ? 0.5 * err * err : ae - 0.5; d[i] = Math.max(-1, Math.min(1, err)) / B; }
        return loss / B;
      }, opt);
      if (this.updates % 500 === 499) this.target.setWeights(this.net.getWeights());
      return r;
    }

    // ---------------------------------------------------------------- REINFORCE
    pg(opt) {
      // play whole episodes until about 64 moves are in hand, sampling each action from the policy
      const steps = [];
      do {
        let done = false;
        const ep = [];
        while (!done) {
          const o = Float32Array.from(this.obs), p = softmax(this.outputs(o), 0, this.nA, this.p), a = pick(p, this.nA, this.rng), t = this.move(a);
          ep.push({ o, a, r: t.r }); done = t.done;
        }
        let G = 0;
        for (let k = ep.length - 1; k >= 0; k--) { G = ep[k].r + this.gamma * G; ep[k].G = G; }
        steps.push(...ep);
      } while (steps.length < 64);
      // the advantage: how much better than this batch's average the return after each move was, in its spread
      let m = 0, v = 0; for (const s of steps) m += s.G; m /= steps.length; for (const s of steps) v += (s.G - m) ** 2; const sd = Math.sqrt(v / steps.length) || 1;
      return this.policyUpdate(steps.map((s) => ({ o: s.o, a: s.a, A: (s.G - m) / sd })), opt);
    }
    // −A·log π(a), less a little entropy to keep it exploring (its gradient on the logits is (π − onehot(a))·A), and
    // with a critic, ½(V − R)² for the value output
    policyUpdate(steps, opt, value) {
      const n = steps.length, X = new Float32Array(n * this.inSize), nA = this.nA, K = this.K, beta = 0.01;
      steps.forEach((s, k) => X.set(s.o, k * this.inSize));
      return this.net.stepWith(X, n, (out, d) => {
        let loss = 0;
        const p = this.p;
        for (let k = 0; k < n; k++) {
          const s = steps[k], o = k * K;
          softmax(out, o, nA, p);
          let H = 0; for (let c = 0; c < nA; c++) if (p[c] > 0) H -= p[c] * Math.log(p[c]);
          loss += -s.A * Math.log(Math.max(1e-9, p[s.a])) - beta * H;
          for (let c = 0; c < nA; c++) d[o + c] = ((p[c] - (c === s.a ? 1 : 0)) * s.A + beta * p[c] * (Math.log(Math.max(1e-9, p[c])) + H)) / n;
          if (value) { const e = out[o + nA] - s.R; loss += 0.25 * e * e; d[o + nA] = (0.5 * e) / n; }
        }
        return loss / n;
      }, opt);
    }
    // one move in each copy of the world, sampled from the policy, all copies in one batch through the network
    rollStep() {
      const N = this.envs.length, X = this.xN, K = this.K;
      this.envs.forEach((e, k) => X.set(e.obs, k * this.inSize));
      const out = Float64Array.from(this.net.forward(X, N, false).subarray(0, N * K)), res = [];
      for (let k = 0; k < N; k++) {
        const p = softmax(out, k * K, this.nA, this.p), a = pick(p, this.nA, this.rng), o = Float32Array.from(this.envs[k].obs), t = this.move(a, this.envs[k]);
        res.push({ o, a, r: this.scaled(this.envs[k], t.r, t.done), v: out[k * K + this.nA], logp: Math.log(Math.max(1e-9, p[a])), done: t.done });
      }
      return res;
    }
    // the critic's value of where each copy of the world is now
    values() { const N = this.envs.length, X = this.xN; this.envs.forEach((e, k) => X.set(e.obs, k * this.inSize)); const out = this.net.forward(X, N, false); return NA.range(N).map((k) => out[k * this.K + this.nA]); }
    // advantages scaled to spread 1 over the batch: steadier steps whatever the size of the rewards
    static normalise(list) { let m = 0, v = 0; for (const s of list) m += s.A; m /= list.length; for (const s of list) v += (s.A - m) ** 2; const sd = Math.sqrt(v / list.length) || 1; for (const s of list) s.A = (s.A - m) / sd; return list; }

    // ---------------------------------------------------------------- actor-critic
    a2c(opt) {
      const T = 8, roll = [];
      for (let t = 0; t < T; t++) roll.push(this.rollStep());
      // in each copy, the returns of its 8 steps, from the critic's value of where it ended up
      const last = this.values(), out = [];
      this.envs.forEach((e, k) => {
        let R = last[k];
        for (let t = T - 1; t >= 0; t--) { const s = roll[t][k]; R = s.r + (s.done ? 0 : this.gamma * R); out.push({ o: s.o, a: s.a, A: R - s.v, R }); }
      });
      return this.policyUpdate(Trainer.normalise(out), opt, true);
    }

    // ---------------------------------------------------------------- PPO
    /* PPO. After each batch's 16 steps, the share of its moves whose probability has moved more than 20% from the
       policy that played them (beyond the clip's band): this.clipShare, and this.clipHist, [{ update, share }] (update:
       the steps taken when that batch was done; for the depth views). */
    ppo(opt) {
      if (!this.queue.length) {
        if (this.ppoBatch) this.clipShareOf(this.ppoBatch);
        // play 16 moves in each of the 8 copies with the current policy, keeping each move's probability and the
        // critic's value; advantages by GAE (λ = 0.95), normalised over the batch
        const T = 16, roll = [];
        for (let t = 0; t < T; t++) roll.push(this.rollStep());
        const last = this.values(), all = [];
        this.envs.forEach((e, k) => {
          let next = last[k], adv = 0;
          for (let t = T - 1; t >= 0; t--) {
            const s = roll[t][k], nv = s.done ? 0 : next, delta = s.r + this.gamma * nv - s.v;
            adv = delta + (s.done ? 0 : this.gamma * 0.95 * adv); s.A = adv; s.R = adv + s.v; next = s.v;
            all.push(s);
          }
        });
        Trainer.normalise(all);
        this.ppoBatch = all;
        // 4 passes over it in shuffled minibatches (the batch size)
        const B = Math.min(this.B, all.length);
        for (let ep = 0; ep < 4; ep++) { const idx = this.rng.shuffle(NA.range(all.length)); for (let k = 0; k < idx.length; k += B) this.queue.push(idx.slice(k, k + B).map((i) => all[i])); }
      }
      const mb = this.queue.shift(), n = mb.length, X = new Float32Array(n * this.inSize), nA = this.nA, K = this.K, beta = 0.01;
      mb.forEach((s, k) => X.set(s.o, k * this.inSize));
      return this.net.stepWith(X, n, (out, d) => {
        let loss = 0;
        const p = this.p;
        for (let k = 0; k < n; k++) {
          const s = mb[k], o = k * K;
          softmax(out, o, nA, p);
          const ratio = Math.exp(Math.log(Math.max(1e-9, p[s.a])) - s.logp), un = ratio * s.A, cl = Math.min(Math.max(ratio, 0.8), 1.2) * s.A;
          loss += -Math.min(un, cl);
          // the clipped side has no gradient: a move whose probability has already moved 20% the helpful way is pushed no further
          if (un <= cl) for (let c = 0; c < nA; c++) d[o + c] = ((p[c] - (c === s.a ? 1 : 0)) * ratio * s.A) / n;
          // less a little entropy, as in actor-critic, to keep it exploring
          let H = 0; for (let c = 0; c < nA; c++) if (p[c] > 0) H -= p[c] * Math.log(p[c]);
          loss -= beta * H;
          for (let c = 0; c < nA; c++) d[o + c] += (beta * p[c] * (Math.log(Math.max(1e-9, p[c])) + H)) / n;
          const e = out[o + nA] - s.R; loss += 0.25 * e * e; d[o + nA] = (0.5 * e) / n;
        }
        return loss / n;
      }, opt);
    }
    // how many of a batch's moves now lie beyond the clip's band (their probability more than 20% from when played)
    clipShareOf(all) {
      const n = all.length, X = new Float32Array(n * this.inSize), K = this.K, nA = this.nA, p = new Float64Array(nA);
      all.forEach((s, k) => X.set(s.o, k * this.inSize));
      const out = this.net.forward(X, n, false);
      let out20 = 0;
      for (let k = 0; k < n; k++) { softmax(out, k * K, nA, p); const ratio = Math.exp(Math.log(Math.max(1e-9, p[all[k].a])) - all[k].logp); if (ratio < 0.8 || ratio > 1.2) out20++; }
      this.clipShare = out20 / n;
      (this.clipHist || (this.clipHist = [])).push({ update: this.updates, share: this.clipShare });
      if (this.clipHist.length > 2000) this.clipHist = this.clipHist.filter((h, i) => i % 2 === 0 || i === this.clipHist.length - 1);
    }
  }
  BRL.Trainer = Trainer;
  BRL.make = (o) => new Trainer(o);
  NA.brl = BRL;
})(typeof globalThis !== 'undefined' && globalThis.NA ? globalThis.NA : window.NA);
