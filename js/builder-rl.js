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
   its best moves do, from 0 to 1 (score()). Three more learn worlds whose controls are two numbers from −1 to 1
   (continuous: the race track, the rocket, the arm and the walker): PPO with a Gaussian policy, SAC and TD3. Runs the
   same in the page and in node. */
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
    bc: { label: 'Copy you (behaviour cloning)', short: 'Copy you', extra: 0, unit: 'update', teach: true, hint: 'It learns from your moves, not from rewards: Jump in and play a few episodes, and every move you make is kept with what the network would have seen. Then each step teaches it to pick your move in that moment, as a classifier learns its classes (cross-entropy). It can only be as good as you were, and it has never seen the moments you never got into: a slip takes it somewhere it has no idea what to do. Start a run of PPO from it (Start from, below) and rewards can take it further.' },
    pref: { label: 'Which did better? (learning from preferences)', short: 'Which did better?', extra: 1, unit: 'update', teach: true, hint: 'No reward table: you are shown two tries of the same start, side by side, and say which did better (or the world’s own reward answers, to compare). A second network, the reward model, learns a reward per move whose total agrees with your answers, and PPO learns to collect that reward. The same idea that taught chatbots what people prefer (RLHF), here on a world you can watch.' },
    gauss: { label: 'PPO, continuous (a Gaussian policy)', short: 'PPO (continuous)', extra: 1, unit: 'update', cont: true, hint: 'For a world whose controls are numbers (steering, the throttle, a motor’s push, from −1 to 1) rather than moves: for each, the network gives the middle of a bell curve, and a spread learned beside it says how far its tries stray; a try is drawn from the curve. Then PPO as with moves: 128 tries in 8 copies of the world, 16 small clipped steps on them, the critic as one more output.' },
    sac: { label: 'SAC (soft actor-critic)', short: 'SAC', extra: 0, unit: 'update', cont: true, hint: 'For numbers as controls: two outputs for each, the middle and the spread of a bell curve, and tanh of a draw from it is the try. Two critics of their own learn what a state and a try are worth from a replay memory, and the network learns to make the tries they value, while keeping its curves as wide as it can (entropy, weighed by α, which learns too). It learns from every move, old ones too, so it needs far fewer than PPO.' },
    td3: { label: 'TD3 (twin delayed DDPG)', short: 'TD3', extra: 0, unit: 'update', cont: true, hint: 'For numbers as controls: one output for each, tanh of it is the try, with a little noise added to explore. Two critics learn what a state and a try are worth from a replay memory, each judged against the smaller of the two (so neither can overrate a try for long); the network learns only every second step, toward the tries the first critic values most.' },
    neat: { label: 'NEAT (it grows its own network)', short: 'NEAT', extra: 0, unit: 'generation', evo: true, hint: 'NeuroEvolution of Augmenting Topologies: it does not use the network you built. Every member starts with no hidden neuron, every input wired to every output; mutations nudge weights, add a connection or add a neuron in the middle of one. Members are grouped into species by how much their genes differ and share their fitness within their species, so a new structure has time to improve before it must beat the rest. One step is one generation.' },
    erl: { label: 'ERL (evolution and DQN)', short: 'ERL', extra: 0, unit: 'generation', evo: true, hint: 'A genetic algorithm and a DQN together: every move the population makes goes into a replay memory, and a DQN (a network like yours) learns Q-values from it by gradient steps between generations. Every few generations the DQN’s network joins the population in place of the worst member, and breeds if it plays well. One step is one generation.' },
  };
  BRL.ORDER = ['dqn', 'pg', 'a2c', 'ppo', 'ga', 'neat', 'novelty', 'es', 'cem', 'me', 'erl', 'bc', 'pref', 'gauss', 'sac', 'td3'];
  // the methods for worlds whose controls are numbers, and the worlds that can be controlled that way
  BRL.isCont = (m) => !!(BRL.METHODS[m] && BRL.METHODS[m].cont);
  BRL.contWorld = (w) => !!(NA.rlEnvs && NA.rlEnvs.WORLDS[w] && NA.rlEnvs.WORLDS[w].cont);
  // the methods that breed a population, one generation a step (builder-evo.js): no gradients, except ERL's DQN
  BRL.isEvo = (m) => !!(BRL.METHODS[m] && BRL.METHODS[m].evo);
  // (nA: the moves, or for the continuous methods the numbers; SAC gives a middle and a spread for each)
  // (and DQN with dueling one more: the value)
  BRL.outputs = (method, nA, s) => (method === 'sac' ? 2 * nA : nA + (BRL.METHODS[method] || BRL.METHODS.dqn).extra + (method === 'dqn' && s && s.dqnDueling ? 1 : 0));
  // the Q-values of DQN's outputs for the page (dueling: the value plus each advantage less their mean)
  BRL.qValues = (out, nA, duel) => { const q = Float64Array.from({ length: nA }, (_, j) => out[j]); if (!duel) return q; let m = 0; for (let j = 0; j < nA; j++) m += out[j]; m /= nA; return q.map((v) => out[nA] + v - m); };
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
    track: { gamma: 0.99, explore: 20000, gaEpisodes: 1, scale: 40, pop: 48, sigma: 0.5, cross: 'none', steps: { gauss: 30000, sac: 20000, td3: 20000, dqn: 30000, pg: 2000, a2c: 10000, ppo: 30000, ga: 150, es: 150, me: 150, novelty: 150, cem: 150, erl: 150 } },
    arena: { gamma: 0.97, explore: 20000, gaEpisodes: 2, scale: 10, pop: 24, steps: { dqn: 30000, pg: 2000, a2c: 6000, ppo: 6000, ga: 80, es: 100, me: 100, novelty: 100, cem: 100, erl: 80 } },
    lander: { gamma: 0.99, explore: 15000, gaEpisodes: 3, scale: 4, pop: 48, sigma: 0.5, cross: 'none', steps: { gauss: 30000, sac: 30000, td3: 30000, dqn: 50000, pg: 3000, a2c: 20000, ppo: 30000, ga: 300, es: 200, me: 150, novelty: 200, cem: 200, erl: 300 } },
    // the worlds of rl-worlds2.js and rl-worlds3.js
    thermo: { gamma: 0.97, explore: 3000, gaEpisodes: 2, scale: 15, pop: 24, steps: { dqn: 8000, pg: 1500, a2c: 3000, ppo: 4000, ga: 100, es: 150, me: 150, novelty: 150, cem: 150, erl: 100 } },
    hurdles: { gamma: 0.97, explore: 4000, gaEpisodes: 2, scale: 2, pop: 24, steps: { dqn: 10000, pg: 2000, a2c: 4000, ppo: 6000, ga: 80, es: 120, me: 120, novelty: 120, cem: 120, erl: 80 } },
    memory: { gamma: 0.95, explore: 2000, gaEpisodes: 4, scale: 1, pop: 24, steps: { dqn: 5000, pg: 1500, a2c: 3000, ppo: 3000, ga: 100, es: 150, me: 100, novelty: 100, cem: 150, erl: 100 } },
    mountain: { gamma: 0.99, explore: 10000, gaEpisodes: 2, scale: 100, pop: 24, steps: { dqn: 20000, pg: 2000, a2c: 6000, ppo: 10000, ga: 60, es: 100, me: 100, novelty: 60, cem: 100, erl: 60 } },
    swingup: { gamma: 0.98, explore: 8000, gaEpisodes: 2, scale: 200, pop: 24, steps: { dqn: 20000, pg: 3000, a2c: 8000, ppo: 10000, ga: 150, es: 200, me: 150, novelty: 150, cem: 200, erl: 150 } },
    arm: { gamma: 0.97, explore: 8000, gaEpisodes: 3, scale: 2, pop: 24, steps: { gauss: 8000, sac: 15000, td3: 15000, dqn: 15000, pg: 2000, a2c: 6000, ppo: 8000, ga: 150, es: 200, me: 150, novelty: 150, cem: 200, erl: 150 } },
    trap: { gamma: 0.99, explore: 15000, gaEpisodes: 1, scale: 2, pop: 48, steps: { dqn: 20000, pg: 2000, a2c: 6000, ppo: 10000, ga: 150, es: 150, me: 200, novelty: 150, cem: 150, erl: 150 } },
    chase: { gamma: 0.98, explore: 15000, gaEpisodes: 2, scale: 6, pop: 24, steps: { dqn: 20000, pg: 3000, a2c: 8000, ppo: 10000, ga: 300, es: 200, me: 200, novelty: 300, cem: 200, erl: 300 } },
    traffic: { gamma: 0.95, explore: 5000, gaEpisodes: 1, scale: 30, pop: 24, steps: { dqn: 10000, pg: 1500, a2c: 4000, ppo: 5000, ga: 80, es: 120, me: 120, novelty: 120, cem: 120, erl: 80 } },
    walker: { gamma: 0.98, explore: 20000, gaEpisodes: 1, scale: 5, pop: 48, steps: { gauss: 20000, sac: 20000, td3: 20000, dqn: 30000, pg: 3000, a2c: 10000, ppo: 20000, ga: 150, es: 200, me: 200, novelty: 200, cem: 200, erl: 150 } },
    herd: { gamma: 0.98, explore: 20000, gaEpisodes: 2, scale: 5, pop: 24, steps: { dqn: 30000, pg: 3000, a2c: 10000, ppo: 20000, ga: 150, es: 200, me: 200, novelty: 200, cem: 200, erl: 150 } },
  };
  const W = (world) => BRL.WORLD[world] || BRL.WORLD.pole;
  /* Easy first: the worlds that can start easier (w.level from 0 to 1), how an episode there counts as a success,
     and what is easier at first. */
  BRL.CURRICULUM = {
    pole: { ok: (w, t) => !t.fell, say: 'gravity a third as strong at first, so the pole falls slowly' },
    arm: { ok: (w) => w.end === 'reached', say: 'the target close to the hand at first' },
    lander: { ok: (w) => w.end === 'landed', say: 'starting low, slow and nearly upright at first' },
    hurdles: { ok: (w) => w.end === 'course', say: 'obstacles only in the first quarter of the course at first' },
  };
  BRL.hasCurriculum = (world, method) => !!BRL.CURRICULUM[world] && !BRL.isEvo(method) && method !== 'bc';
  // (Copy you: a thousand steps on your moves; Which did better?: PPO's steps)
  BRL.stepsFor = (world, method) => (method === 'bc' ? 1000 : W(world).steps[method === 'pref' ? 'ppo' : method === 'neat' ? 'ga' : method] || 3000);
  // NEAT reads a list of numbers (its genomes are wired input by input)
  BRL.neatOk = (input) => !!input && input.kind === 'vector' && input.dim <= 64;
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
    novK: 10, novMix: 'novelty', novAdd: 2, cemElite: 0.2, cemSigma: 0.3, cemFloor: 0.05, erlSteps: 200, erlEvery: 5, erlLr: 0.001, prefAsk: 'you', prefEvery: 300,
    ppoClip: 0.2, ppoEpochs: 4, ppoLam: 0.95, ppoEnt: 0.01, dqnMem: 10000, dqnTarget: 500, dqnDouble: true, dqnDueling: false, dqnN: 1, warm: 1000, curriculum: 'off', worldModel: 'off', chaserLearns: false, pbt: 1 });
  // (predator and prey: the chaser learns too, by the methods that play through the trainer)
  BRL.canChase = (world, method) => world === 'chase' && !BRL.isEvo(method) && method !== 'bc' && method !== 'pref';
  // (a model of the world: for worlds seen as numbers, by the methods that play through the trainer)
  BRL.canModel = (input, method) => !!input && input.kind === 'vector' && !BRL.isEvo(method) && method !== 'bc';

  const PREF_EPS = 0.25; // Which did better?: the share of random moves in the second try of a pair
  const argmax = (o, base, n) => { let a = 0; for (let k = 1; k < n; k++) if (o[base + k] > o[base + a]) a = k; return a; };
  const softmax = (o, base, n, p) => { let m = -Infinity; for (let k = 0; k < n; k++) m = Math.max(m, o[base + k]); let s = 0; for (let k = 0; k < n; k++) { p[k] = Math.exp(o[base + k] - m); s += p[k]; } for (let k = 0; k < n; k++) p[k] /= s; return p; };
  const pick = (p, n, rng) => { let u = rng.next(); for (let k = 0; k < n; k++) { u -= p[k]; if (u <= 0) return k; } return n - 1; };
  BRL.softmax = (o, n) => softmax(o, 0, n, new Float64Array(n));
  const LOG2PI = Math.log(2 * Math.PI);
  // the numbers a continuous method's best move sets: the middles (tanh of them for SAC and TD3)
  BRL.contMove = (method, out, dA) => Float64Array.from({ length: dA }, (_, j) => (method === 'gauss' ? Math.max(-1, Math.min(1, out[j])) : Math.tanh(out[j])));
  // its best move for outputs out, as the page plays it: a move's number, or the numbers (d: the problem's data)
  BRL.moveOf = (out, d) => (d.cont ? BRL.contMove(d.method, out, d.dA) : argmax(out, 0, d.nA));
  BRL.argmax = (o, n) => argmax(o, 0, n);

  /* A small network of its own, for the critics of SAC and TD3 (what was seen and a try in, the return it expects out,
     ReLU between): written out by hand, since the actor needs what the page's networks do not give, how the output
     changes with the try (the gradient with respect to its input). Weights and biases start as PyTorch's do, uniform
     within ±1/√(inputs); Adam; a slow copy follows it by Polyak averaging. */
  class Mlp {
    constructor(sizes, seed) {
      const rng = new NA.RNG(seed);
      this.sizes = sizes; this.L = sizes.length - 1; this.params = []; this.grads = []; this.m = []; this.v = []; this.t = 0; this.acts = [];
      for (let l = 0; l < this.L; l++) {
        const nIn = sizes[l], nOut = sizes[l + 1], lim = 1 / Math.sqrt(nIn);
        this.params.push(Float32Array.from({ length: nIn * nOut }, () => rng.uniform(-lim, lim)), Float32Array.from({ length: nOut }, () => rng.uniform(-lim, lim)));
      }
      for (const p of this.params) { this.grads.push(new Float32Array(p.length)); this.m.push(new Float32Array(p.length)); this.v.push(new Float32Array(p.length)); }
    }
    // B rows of inputs → B rows of outputs (the activations are kept for backward)
    forward(X, B) {
      let A = X;
      this.acts = [X]; this.B = B;
      for (let l = 0; l < this.L; l++) {
        const nIn = this.sizes[l], nOut = this.sizes[l + 1], W = this.params[2 * l], bias = this.params[2 * l + 1], Z = new Float32Array(B * nOut), last = l === this.L - 1;
        for (let i = 0; i < B; i++) {
          const ai = i * nIn, zi = i * nOut;
          for (let o = 0; o < nOut; o++) {
            let s = bias[o];
            const wo = o * nIn;
            for (let k = 0; k < nIn; k++) s += W[wo + k] * A[ai + k];
            Z[zi + o] = last || s > 0 ? s : 0;
          }
        }
        this.acts.push(Z); A = Z;
      }
      return A;
    }
    // the gradient of the outputs (B rows) added into the parameters' gradients; the inputs' gradient returned when asked
    backward(dOut, wantDx) {
      const B = this.B;
      let D = dOut;
      for (let l = this.L - 1; l >= 0; l--) {
        const nIn = this.sizes[l], nOut = this.sizes[l + 1], W = this.params[2 * l], A = this.acts[l], Z = this.acts[l + 1], gW = this.grads[2 * l], gb = this.grads[2 * l + 1];
        if (l < this.L - 1) for (let i = 0; i < D.length; i++) if (Z[i] <= 0) D[i] = 0; // (ReLU: only where it was above 0)
        const need = l > 0 || wantDx, dA = need ? new Float32Array(B * nIn) : null;
        for (let i = 0; i < B; i++) {
          const ai = i * nIn, di = i * nOut;
          for (let o = 0; o < nOut; o++) {
            const g = D[di + o];
            if (!g) continue;
            gb[o] += g;
            const wo = o * nIn;
            if (need) for (let k = 0; k < nIn; k++) { gW[wo + k] += g * A[ai + k]; dA[ai + k] += g * W[wo + k]; }
            else for (let k = 0; k < nIn; k++) gW[wo + k] += g * A[ai + k];
          }
        }
        D = dA;
      }
      return D;
    }
    zero() { for (const g of this.grads) g.fill(0); }
    // Adam, the gradient first clipped to a total length of clip
    adam(lr, clip) {
      let sc = 1;
      if (clip > 0) { let ss = 0; for (const g of this.grads) for (let i = 0; i < g.length; i++) ss += g[i] * g[i]; const n = Math.sqrt(ss); if (n > clip) sc = clip / n; }
      this.t++;
      const c1 = 1 - 0.9 ** this.t, c2 = 1 - 0.999 ** this.t;
      this.params.forEach((p, j) => {
        const g = this.grads[j], m = this.m[j], v = this.v[j];
        for (let i = 0; i < p.length; i++) { const gi = g[i] * sc; m[i] = 0.9 * m[i] + 0.1 * gi; v[i] = 0.999 * v[i] + 0.001 * gi * gi; p[i] -= (lr * m[i] / c1) / (Math.sqrt(v[i] / c2) + 1e-8); }
      });
    }
    copy(o) { this.params.forEach((p, j) => p.set(o.params[j])); }
    polyak(o, tau) { this.params.forEach((p, j) => { const q = o.params[j]; for (let i = 0; i < p.length; i++) p[i] += tau * (q[i] - p[i]); }); }
  }
  BRL.Mlp = Mlp;

  /* A model of the world, learned beside the network (s.worldModel): from every move it makes, what it saw, the move,
     what it saw next and the reward, a small network of its own (64 and 64 ReLU neurons) learns to predict the change
     in what it sees and the reward, a batch every 4 moves. dream() then plays the network's best moves twice from one
     start: in the world, and in the model's imagination, each move fed back what the model predicted; how far apart
     they drift is how far its imagination can be trusted. */
  class WorldModel {
    constructor(inSize, aDim, cont, seed) {
      this.inSize = inSize; this.aDim = aDim; this.cont = cont; this.net = new Mlp([inSize + aDim, 64, 64, inSize + 1], seed); this.buf = []; this.cap = 20000; this.next = 0; this.n = 0; this.err = null; this.rng = new NA.RNG(seed + 5);
    }
    enc(a, X, at) { if (this.cont) for (let j = 0; j < this.aDim; j++) X[at + j] = a[j]; else X[at + a] = 1; }
    keep(o, a, o2, r) {
      const q = { o: Float32Array.from(o), a: this.cont ? Float32Array.from(a) : a, o2: Float32Array.from(o2), r: Math.max(-10, Math.min(10, r)) };
      if (this.buf.length < this.cap) this.buf.push(q); else { this.buf[this.next] = q; this.next = (this.next + 1) % this.cap; }
      if (++this.n % 4 === 0 && this.buf.length >= 256) this.learn();
    }
    learn() {
      const B = 64, S = this.inSize, D = S + this.aDim, X = new Float32Array(B * D), Y = new Float32Array(B * (S + 1));
      for (let b = 0; b < B; b++) { const q = this.buf[this.rng.int(this.buf.length)]; X.set(q.o, b * D); this.enc(q.a, X, b * D + S); for (let i = 0; i < S; i++) Y[b * (S + 1) + i] = q.o2[i] - q.o[i]; Y[b * (S + 1) + S] = q.r; }
      const out = this.net.forward(X, B), d = new Float32Array(out.length);
      let loss = 0;
      for (let i = 0; i < out.length; i++) { const e = out[i] - Y[i]; loss += e * e; d[i] = (2 * e) / out.length; }
      this.net.zero(); this.net.backward(d, false); this.net.adam(0.001, 10);
      this.err = this.err === null ? loss / out.length : 0.98 * this.err + 0.02 * (loss / out.length);
    }
    // what it predicts after move a from o: the next observation and the reward
    predict(o, a) { const X = new Float32Array(this.inSize + this.aDim); X.set(o); this.enc(a, X, this.inSize); const out = this.net.forward(X, 1); return { o: Float32Array.from(o, (v, i) => v + out[i]), r: out[this.inSize] }; }
    // H moves of the policy from the start of seed: in the world, and in its imagination
    dream(world, ws, policy, seed, H = 60) {
      const W = NA.rlEnvs.make(world, ws), rng = new NA.RNG(seed);
      let o = W.reset(rng), oh = Float32Array.from(o), done = false;
      const real = [Float32Array.from(o)], imag = [Float32Array.from(oh)], rr = [], ri = [];
      for (let t = 0; t < H && !done; t++) {
        const a = policy(o), ah = policy(oh), s = W.step(a, rng), p = this.predict(oh, ah);
        o = s.obs; done = s.done; oh = p.o;
        real.push(Float32Array.from(o)); imag.push(p.o); rr.push(s.r); ri.push(p.r);
      }
      return { real, imag, rr, ri };
    }
  }
  BRL.WorldModel = WorldModel;

  /* The chaser that learns too (Escape the chaser, s.chaserLearns): a DQN of its own, 8 numbers in (where the runner is
     from it, how fast each moves, where it is) and 5 moves out (a push up, down, left or right, or none), a replay
     memory, a slow copy for its targets. It is paid 1 for a catch, 0.1 for every metre it closes, −0.01 a step. */
  class Chaser {
    constructor(seed) {
      this.q = new Mlp([8, 64, 64, 5], seed); this.slow = new Mlp([8, 64, 64, 5], 1); this.slow.copy(this.q);
      this.buf = []; this.cap = 20000; this.next = 0; this.n = 0; this.rng = new NA.RNG(seed + 9); this.learns = 0; this.ends = [];
    }
    act(o, explore) {
      if (explore && this.rng.next() < Math.max(0.05, 1 - this.n / 20000)) return this.rng.int(5);
      const out = this.q.forward(o, 1);
      let b = 0; for (let k = 1; k < 5; k++) if (out[k] > out[b]) b = k;
      return b;
    }
    keep(o, a, r, o2, over) {
      const q = { o, a, r, o2, over };
      if (this.buf.length < this.cap) this.buf.push(q); else { this.buf[this.next] = q; this.next = (this.next + 1) % this.cap; }
      if (++this.n % 2 === 0 && this.buf.length >= 500) this.learn();
    }
    learn() {
      const B = 64, X = new Float32Array(B * 8), X2 = new Float32Array(B * 8), batch = [];
      for (let b = 0; b < B; b++) { const q = this.buf[this.rng.int(this.buf.length)]; batch.push(q); X.set(q.o, b * 8); X2.set(q.o2, b * 8); }
      const t2 = Float64Array.from(this.slow.forward(X2, B)), out = this.q.forward(X, B), d = new Float32Array(B * 5);
      for (let b = 0; b < B; b++) {
        const q = batch[b], y = q.r + (q.over ? 0 : 0.97 * Math.max(...t2.subarray(b * 5, b * 5 + 5))), e = out[b * 5 + q.a] - y;
        d[b * 5 + q.a] = Math.max(-1, Math.min(1, e)) / B;
      }
      this.q.zero(); this.q.backward(d, false); this.q.adam(0.0005, 10);
      if (++this.learns % 500 === 0) this.slow.copy(this.q);
    }
    // the last 50 games: how many it caught the runner in
    caught() { return this.ends.length ? this.ends.reduce((a, b) => a + b, 0) / this.ends.length : null; }
  }
  BRL.Chaser = Chaser;

  /* A trainer: the world (by id and settings), the network built from your blocks (and the blocks, for evolution
     to reshape), a maker of more networks like it (the target network), the method, its settings (s: gamma,
     explore, batch; pop, sigma and the rest of evolution's) and a seed. */
  class Trainer {
    constructor(o) {
      Object.assign(this, { worldId: o.world, ws: o.ws || {}, net: o.net, makeNet: o.makeNet, blocks: o.blocks || null, method: BRL.METHODS[o.method] ? (BRL.isCont(o.method) && !BRL.contWorld(o.world) ? 'ppo' : o.method) : 'dqn', rng: new NA.RNG(o.seed || 1), demos: o.demos || null, seed: o.seed || 1 });
      const s = Object.assign(BRL.defaults(this.worldId), o.s);
      this.s = s;
      this.gamma = s.gamma; this.explore = s.explore; this.popSize = Math.max(4, Math.round(s.pop)); this.sigma = s.sigma; this.B = Math.max(4, Math.round(s.batch || 32));
      // actor-critic and PPO play in 8 copies of the world at once; the others in one
      const N = this.method === 'a2c' || this.method === 'ppo' || this.method === 'pref' || this.method === 'gauss' ? 8 : 1;
      this.envs = NA.range(N).map(() => ({ w: E().make(this.worldId, this.ws), obs: null, ret: 0, disc: 0 }));
      // (easy first: the worlds it learns in start at level 0; the one that scores it stays at full difficulty)
      this.curr = s.curriculum === 'on' && BRL.hasCurriculum(this.worldId, this.method) ? { level: 0, tally: [], raised: [] } : null;
      if (this.curr) for (const e of this.envs) e.w.level = 0;
      for (const e of this.envs) e.obs = e.w.reset(this.rng);
      this.world = this.envs[0].w; this.nA = this.world.nA; this.inSize = this.world.obsSize;
      // (a continuous method sets the world's numbers, dA of them, in place of picking one of its moves)
      this.cont = BRL.isCont(this.method) && this.world.dA > 0; this.dA = this.cont ? this.world.dA : 0;
      this.K = BRL.outputs(this.method, this.cont ? this.dA : this.nA, s);
      // the knobs: PPO's clip, passes over each batch, GAE λ and entropy bonus; DQN's double and dueling, target refresh, n-step
      this.clipC = s.ppoClip ?? 0.2; this.ppoE = Math.max(1, Math.round(s.ppoEpochs ?? 4)); this.lam = s.ppoLam ?? 0.95; this.entC = s.ppoEnt ?? 0.01;
      this.dbl = s.dqnDouble !== false; this.duel = this.method === 'dqn' && !!s.dqnDueling; this.tgtEvery = Math.max(1, Math.round(s.dqnTarget || 500)); this.nStep = Math.max(1, Math.round(s.dqnN || 1));
      this.rl = true; this.updates = 0; this.envSteps = 0; this.episodes = 0; this.retEma = null; this.lastEp = null;
      // where its moves took it, 40 × 40 across the world's frame (the worlds that say where they are; not evolution, which plays elsewhere)
      this.wm = s.worldModel === 'on' && BRL.canModel(this.world.input, this.method) ? new WorldModel(this.inSize, this.cont ? this.dA : this.nA, this.cont, this.seed * 13 + 3) : null;
      this.pp = s.chaserLearns && BRL.canChase(this.worldId, this.method) ? new Chaser(this.seed * 11 + 7) : null;
      this.visits = this.world.where && this.world.where() && !BRL.isEvo(this.method) && this.method !== 'bc' ? { n: 40, m: 40, c: new Float32Array(1600), total: 0 } : null;
      // A2C and PPO: rewards normalised by the running spread of the discounted return (its count, mean and sum of squares)
      this.norm = N > 1 && s.normReward !== false; this.rn = { n: 0, mean: 0, m2: 0 };
      this.p = new Float64Array(this.K); this.x1 = new Float32Array(this.inSize); this.xN = new Float32Array(N * this.inSize);
      this.init();
    }
    get unit() { return BRL.METHODS[this.method].unit; }
    get obs() { return this.envs[0].obs; }
    // the network's outputs for one observation (a copy)
    outputs(obs, net = this.net) { if (this.neat && net === this.net) return this.neat.outputs(obs); this.x1.set(obs); return Float64Array.from(net.forward(this.x1, 1, false).subarray(0, this.K)); }
    // its best move: the largest Q-value, or the most likely action (net: another network, a member of a population)
    policy(obs, net = this.net) { if (this.neat && net === this.net) return this.neat.act(obs); this.x1.set(obs); const out = net.forward(this.x1, 1, false); return this.cont ? BRL.contMove(this.method, out, this.dA) : argmax(out, 0, this.nA); }
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
      const o0 = this.wm ? e.obs : null, P = this.pp, w = e.w, po = P ? w.chaserObs() : null, pd = P ? w.near()[1] : 0;
      if (P) w.chaserMove = P.act(po, true);
      const pa = P ? w.chaserMove : null, t = e.w.step(a, this.rng);
      // (the chaser that learns: paid for a catch, and for closing in)
      if (P) { const got = w.end === 'tagged'; P.keep(po, pa, got ? 1 : -0.01 + 0.1 * (pd - w.near()[1]), w.chaserObs(), got); if (t.done) { P.ends.push(got ? 1 : 0); if (P.ends.length > 50) P.ends.shift(); } }
      if (this.wm) this.wm.keep(o0, a, t.obs, t.r);
      this.envSteps++; e.ret += t.r;
      // (cut: the episode ended by its time limit, not by what happened in it)
      this.cut = !!(t.done && e.w.maxSteps && e.w.t >= e.w.maxSteps);
      if (this.visits) { const p = e.w.where(); if (p) { const V = this.visits, i = Math.min(V.n - 1, Math.max(0, Math.floor(p[0] * V.n))), j = Math.min(V.m - 1, Math.max(0, Math.floor(p[1] * V.m))); V.c[j * V.n + i]++; V.total++; } }
      if (t.done && this.curr) this.currEnd(e.w, t);
      if (t.done) { this.episodes++; this.lastEp = e.ret; this.retEma = this.retEma === null ? e.ret : 0.95 * this.retEma + 0.05 * e.ret; e.ret = 0; e.obs = e.w.reset(this.rng); }
      else e.obs = t.obs;
      return t;
    }
    // an episode ended at this level: a success? 3 in 4 of the last 10 or more, and every world a quarter harder
    currEnd(w, t) {
      const C = this.curr;
      C.tally.push(BRL.CURRICULUM[this.worldId].ok(w, t) ? 1 : 0);
      if (C.tally.length > 20) C.tally.shift();
      if (C.level < 1 && C.tally.length >= 10 && C.tally.reduce((a, b) => a + b, 0) >= 0.75 * C.tally.length) {
        C.level = Math.min(1, C.level + 0.25); C.tally = []; C.raised.push({ step: this.updates, level: C.level });
        for (const e of this.envs) e.w.level = C.level;
      }
    }
    init() {
      const m = this.method;
      if (m === 'dqn') { this.cap = Math.max(100, Math.round(this.s.dqnMem || 10000)); this.buf = []; this.next = 0; this.nq = []; this.target = this.makeNet(); this.target.setWeights(this.net.getWeights()); }
      if (BRL.isEvo(m)) this.evo = m === 'neat' && NA.neat ? new NA.neat.Neat(this, this.s) : new NA.bevo.Evolution(this, this.s, this.blocks);
      if (m === 'ppo' || m === 'pref') this.queue = [];
      if (m === 'pref') this.prefInit();
      if (m === 'gauss') { this.queue = []; this.logStd = new Float64Array(this.dA).fill(-0.5); }
      if (m === 'sac' || m === 'td3') this.offInit();
    }
    // one step; an evolutionary method given a deadline (in performance.now() ms) may answer { partial } until its generation is judged
    step(opt, deadline) {
      // (Which did better?: a pair waiting for an answer, or a new pair due, comes before the next step)
      if (this.method === 'pref') { const q = this.prefDue(); if (q) return q; }
      const r = this.evo ? this.evo.step(deadline) : this.method === 'dqn' ? this.dqn(opt) : this.method === 'pg' ? this.pg(opt) : this.method === 'a2c' ? this.a2c(opt) : this.method === 'bc' ? this.bc(opt) : this.method === 'gauss' ? this.gppo(opt) : this.method === 'sac' || this.method === 'td3' ? this.offStep(opt) : this.ppo(opt);
      if (!r.partial) this.updates++;
      return r;
    }

    // ---------------------------------------------------------------- DQN
    eps() { return this.explore > 0 ? Math.max(0.05, 1 - this.envSteps / this.explore) : 0; }
    // the Q-values in outputs out at base: the outputs themselves, or (dueling) the value plus each move's advantage less their mean
    qOf(out, base) {
      const nA = this.nA, q = new Float64Array(nA);
      if (!this.duel) { for (let j = 0; j < nA; j++) q[j] = out[base + j]; return q; }
      let m = 0; for (let j = 0; j < nA; j++) m += out[base + j]; m /= nA;
      for (let j = 0; j < nA; j++) q[j] = out[base + nA] + out[base + j] - m;
      return q;
    }
    /* A move, kept in the replay memory (of s.dqnMem moves); with n-step returns (s.dqnN) each move waits for the
       rewards of its next n moves (fewer when the episode ends first) and is valued from where it got to then. Then a
       batch, toward r + γⁿ × the next state's value: double DQN (s.dqnDouble) lets the network pick the next move and
       the target network say what it is worth; otherwise the target network does both, and overrates moves it happens
       to overrate. The target network is refreshed every s.dqnTarget updates. */
    dqn(opt) {
      const o = Float32Array.from(this.obs), a = this.rng.next() < this.eps() ? this.rng.int(this.nA) : this.policy(o), t = this.move(a), n = this.nStep;
      this.nq.push({ o, a, r: t.r });
      while (this.nq.length && (this.nq.length >= n || t.done)) {
        let R = 0;
        this.nq.forEach((q, k) => { R += q.r * this.gamma ** k; });
        const e = { o: this.nq[0].o, a: this.nq[0].a, r: R, k: this.nq.length, o2: t.done ? null : Float32Array.from(t.obs) };
        if (this.buf.length < this.cap) this.buf.push(e); else { this.buf[this.next] = e; this.next = (this.next + 1) % this.cap; }
        this.nq.shift();
      }
      if (this.buf.length < 64) return { loss: 0, gradNorm: 0 };
      const B = this.B, K = this.K, nA = this.nA, X = new Float32Array(B * this.inSize), X2 = new Float32Array(B * this.inSize), batch = [];
      for (let b = 0; b < B; b++) { const q = this.buf[this.rng.int(this.buf.length)]; batch.push(q); X.set(q.o, b * this.inSize); if (q.o2) X2.set(q.o2, b * this.inSize); }
      const y = new Float64Array(B);
      if (this.gamma) {
        const q2 = Float64Array.from(this.target.forward(X2, B, false)), on2 = this.dbl ? Float64Array.from(this.net.forward(X2, B, false)) : null;
        for (let b = 0; b < B; b++) {
          const e = batch[b];
          if (!e.o2) { y[b] = e.r; continue; }
          const Qt = this.qOf(q2, b * K), next = argmax(this.dbl ? this.qOf(on2, b * K) : Qt, 0, nA);
          y[b] = e.r + this.gamma ** e.k * Qt[next];
        }
      } else for (let b = 0; b < B; b++) y[b] = batch[b].r;
      const r = this.net.stepWith(X, B, (out, d) => {
        let loss = 0;
        for (let b = 0; b < B; b++) {
          const base = b * K, a = batch[b].a, q = this.duel ? this.qOf(out, base)[a] : out[base + a], err = q - y[b], ae = Math.abs(err), g = Math.max(-1, Math.min(1, err)) / B;
          loss += ae <= 1 ? 0.5 * err * err : ae - 0.5;
          // (dueling: the value takes the whole error; each advantage its share of the mean, the move's one the rest)
          if (this.duel) { d[base + nA] = g; for (let j = 0; j < nA; j++) d[base + j] = (j === a ? g : 0) - g / nA; } else d[base + a] = g;
        }
        return loss / B;
      }, opt);
      if (this.updates % this.tgtEvery === this.tgtEvery - 1) this.target.setWeights(this.net.getWeights());
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
        res.push({ o, a, r: this.scaled(this.envs[k], this.method === 'pref' ? this.rhat(o, a) : t.r, t.done), v: out[k * K + this.nA], logp: Math.log(Math.max(1e-9, p[a])), done: t.done });
      }
      return res;
    }
    // the critic's value of where each copy of the world is now
    values() { const N = this.envs.length, X = this.xN; this.envs.forEach((e, k) => X.set(e.obs, k * this.inSize)); const out = this.net.forward(X, N, false); return NA.range(N).map((k) => out[k * this.K + this.vAt]); }
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
            adv = delta + (s.done ? 0 : this.gamma * this.lam * adv); s.A = adv; s.R = adv + s.v; next = s.v;
            all.push(s);
          }
        });
        Trainer.normalise(all);
        this.ppoBatch = all;
        // 4 passes over it in shuffled minibatches (the batch size)
        const B = Math.min(this.B, all.length);
        for (let ep = 0; ep < this.ppoE; ep++) { const idx = this.rng.shuffle(NA.range(all.length)); for (let k = 0; k < idx.length; k += B) this.queue.push(idx.slice(k, k + B).map((i) => all[i])); }
      }
      const mb = this.queue.shift(), n = mb.length, X = new Float32Array(n * this.inSize), nA = this.nA, K = this.K, beta = this.entC, lo = 1 - this.clipC, hi = 1 + this.clipC;
      mb.forEach((s, k) => X.set(s.o, k * this.inSize));
      return this.net.stepWith(X, n, (out, d) => {
        let loss = 0;
        const p = this.p;
        for (let k = 0; k < n; k++) {
          const s = mb[k], o = k * K;
          softmax(out, o, nA, p);
          const ratio = Math.exp(Math.log(Math.max(1e-9, p[s.a])) - s.logp), un = ratio * s.A, cl = Math.min(Math.max(ratio, lo), hi) * s.A;
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
    // ---------------------------------------------------------------- continuous controls
    /* Three methods for worlds that take two numbers from −1 to 1 (the race track, the rocket, the robot arm and the
       walker: rl-worlds*.js, cont). PPO with a Gaussian policy: the network's first outputs are the middle of a bell
       curve for each number (μ) and a spread of its own for each (σ = e^logσ, learned beside the network, the same in
       every state) says how far its tries stray; a try is drawn from the curve, clipped to −1…1, and its probability
       density is what the clip compares; the last output is the critic. SAC (soft actor-critic): two outputs a number,
       its middle and its log spread; a try is tanh of a draw from that curve; two critics of their own learn the
       return of a state and a try, from a replay memory, each against the smaller of their two slow copies, and the
       network learns to make tries the critics value, plus the entropy of its curves times α, which is itself learned
       so that the entropy stays near −(the numbers it chooses). TD3: one output a number, tanh of it is the try (no
       curve: noise is added to explore); the same two critics, but the network learns only every second step, and the
       critics' targets use a slow copy of it with clipped noise added (target policy smoothing). Its best moves: the
       middles (tanh of them for SAC and TD3). */
    get vAt() { return this.cont ? this.dA : this.nA; }
    stepLogStd(g, opt) {
      // Adam on the spreads' logs, kept between e^−2.5 and e^0.5
      const lr = (opt && opt.lr) || 0.001, S = this.lsOpt || (this.lsOpt = { m: new Float64Array(this.dA), v: new Float64Array(this.dA), t: 0 });
      S.t++;
      for (let j = 0; j < this.dA; j++) {
        S.m[j] = 0.9 * S.m[j] + 0.1 * g[j]; S.v[j] = 0.999 * S.v[j] + 0.001 * g[j] * g[j];
        this.logStd[j] = Math.max(-2.5, Math.min(0.5, this.logStd[j] - (lr * S.m[j] / (1 - 0.9 ** S.t)) / (Math.sqrt(S.v[j] / (1 - 0.999 ** S.t)) + 1e-8)));
      }
    }
    // one try in each copy of the world, drawn from the bell curves, all copies in one batch through the network
    crollStep() {
      const N = this.envs.length, X = this.xN, K = this.K, dA = this.dA, res = [];
      this.envs.forEach((e, k) => X.set(e.obs, k * this.inSize));
      const out = Float64Array.from(this.net.forward(X, N, false).subarray(0, N * K));
      for (let k = 0; k < N; k++) {
        const a = new Float64Array(dA), act = new Float64Array(dA);
        let logp = 0;
        for (let j = 0; j < dA; j++) { const z = this.rng.gauss(), ls = this.logStd[j]; a[j] = out[k * K + j] + Math.exp(ls) * z; act[j] = Math.max(-1, Math.min(1, a[j])); logp += -0.5 * z * z - ls - 0.5 * LOG2PI; }
        const o = Float32Array.from(this.envs[k].obs), t = this.move(act, this.envs[k]);
        res.push({ o, a, r: this.scaled(this.envs[k], t.r, t.done), v: out[k * K + dA], logp, done: t.done });
      }
      return res;
    }
    gppo(opt) {
      if (!this.queue.length) {
        const T = 16, roll = [];
        for (let t = 0; t < T; t++) roll.push(this.crollStep());
        const last = this.values(), all = [];
        this.envs.forEach((e, k) => {
          let next = last[k], adv = 0;
          for (let t = T - 1; t >= 0; t--) {
            const s = roll[t][k], nv = s.done ? 0 : next, delta = s.r + this.gamma * nv - s.v;
            adv = delta + (s.done ? 0 : this.gamma * this.lam * adv); s.A = adv; s.R = adv + s.v; next = s.v;
            all.push(s);
          }
        });
        Trainer.normalise(all);
        const B = Math.min(this.B, all.length);
        for (let ep = 0; ep < this.ppoE; ep++) { const idx = this.rng.shuffle(NA.range(all.length)); for (let k = 0; k < idx.length; k += B) this.queue.push(idx.slice(k, k + B).map((i) => all[i])); }
      }
      const mb = this.queue.shift(), n = mb.length, X = new Float32Array(n * this.inSize), dA = this.dA, K = this.K, beta = this.entC / 10, g = new Float64Array(dA), lo = 1 - this.clipC, hi = 1 + this.clipC;
      mb.forEach((s, k) => X.set(s.o, k * this.inSize));
      const r = this.net.stepWith(X, n, (out, d) => {
        let loss = 0;
        for (let k = 0; k < n; k++) {
          const s = mb[k], o = k * K;
          let logp = 0;
          for (let j = 0; j < dA; j++) { const z = (s.a[j] - out[o + j]) / Math.exp(this.logStd[j]); logp += -0.5 * z * z - this.logStd[j] - 0.5 * LOG2PI; }
          const ratio = Math.exp(Math.min(20, logp - s.logp)), un = ratio * s.A, cl = Math.min(Math.max(ratio, lo), hi) * s.A;
          loss += -Math.min(un, cl);
          // (the clipped side has no gradient: a try whose density has moved 20% the helpful way is pushed no further)
          if (un <= cl) for (let j = 0; j < dA; j++) { const sg = Math.exp(this.logStd[j]), z = (s.a[j] - out[o + j]) / sg; d[o + j] = (-s.A * ratio * z / sg) / n; g[j] += (-s.A * ratio * (z * z - 1)) / n; }
          const e = out[o + dA] - s.R; loss += 0.25 * e * e; d[o + dA] = (0.5 * e) / n;
        }
        // less a little of the curves' entropy, Σ(logσ + ½ log 2πe), to keep it trying: its gradient on each logσ is 1
        let H = 0; for (let j = 0; j < dA; j++) { H += this.logStd[j] + 0.5 * (LOG2PI + 1); g[j] -= beta; }
        return loss / n - beta * H;
      }, opt);
      this.stepLogStd(g, opt);
      return r;
    }
    // SAC and TD3: the critics (and their slow copies), the replay memory, the random tries before learning starts
    offInit() {
      const D = this.inSize + this.dA, H = 64;
      this.q1 = new Mlp([D, H, H, 1], this.seed * 7 + 1); this.q2 = new Mlp([D, H, H, 1], this.seed * 7 + 2);
      this.q1t = new Mlp([D, H, H, 1], 1); this.q2t = new Mlp([D, H, H, 1], 1); this.q1t.copy(this.q1); this.q2t.copy(this.q2);
      this.cap = 100000; this.buf = []; this.next = 0; this.warm = Math.max(0, Math.round(this.s.warm ?? 1000)); this.tau = 0.005; this.criticLr = this.s.criticLr || 0.001; this.offB = this.s.offBatch || 0;
      if (this.method === 'sac') { this.logAlpha = Math.log(0.2); this.aOpt = { m: 0, v: 0, t: 0 }; this.entTarget = -this.dA; }
      if (this.method === 'td3') { this.target = this.makeNet(); this.target.setWeights(this.net.getWeights()); this.wNow = null; }
    }
    get alpha() { return Math.exp(this.logAlpha); }
    // a try from SAC's curve for outputs out (at base): { a: tanh(u), z, logp } with u = μ + σz
    sacTry(out, base, z) {
      const dA = this.dA, a = new Float64Array(dA);
      let logp = 0;
      for (let j = 0; j < dA; j++) {
        const ls = Math.max(-5, Math.min(2, out[base + dA + j])), u = out[base + j] + Math.exp(ls) * z[j], th = Math.tanh(u);
        a[j] = th; logp += -0.5 * z[j] * z[j] - ls - 0.5 * LOG2PI - Math.log(1 - th * th + 1e-6);
      }
      return { a, logp };
    }
    // one move in the world: random at first, then a try of the policy; kept in the replay memory; then a step of learning
    offStep(opt) {
      const dA = this.dA, o = Float32Array.from(this.obs), z = () => Float64Array.from({ length: dA }, () => this.rng.gauss());
      let act;
      if (this.envSteps < this.warm) act = Float64Array.from({ length: dA }, () => this.rng.uniform(-1, 1));
      else if (this.method === 'sac') act = this.sacTry(this.outputs(o), 0, z()).a;
      else { const out = this.outputs(o); act = Float64Array.from({ length: dA }, (_, j) => Math.max(-1, Math.min(1, Math.tanh(out[j]) + 0.1 * this.rng.gauss()))); }
      const t = this.move(act);
      // (an episode cut short by its time limit is no end of the world: its last state still has a value)
      const q = { o, a: act, r: t.r, o2: Float32Array.from(t.obs), end: t.done && !this.cut };
      if (this.buf.length < this.cap) this.buf.push(q); else { this.buf[this.next] = q; this.next = (this.next + 1) % this.cap; }
      if (this.envSteps < this.warm || this.buf.length < 256) return { loss: 0, gradNorm: 0 };
      return this.method === 'sac' ? this.sacUpdate(opt) : this.td3Update(opt);
    }
    // a batch from the memory: the states, the next states, what each critic sees ([state, try]) and the rest
    offBatch(B) {
      const S = this.inSize, dA = this.dA, D = S + dA, X = new Float32Array(B * S), X2 = new Float32Array(B * S), Z = new Float32Array(B * D), rew = new Float64Array(B), live = new Float64Array(B);
      for (let b = 0; b < B; b++) {
        const q = this.buf[this.rng.int(this.buf.length)];
        X.set(q.o, b * S); X2.set(q.o2, b * S); Z.set(q.o, b * D); for (let j = 0; j < dA; j++) Z[b * D + S + j] = q.a[j];
        rew[b] = q.r; live[b] = q.end ? 0 : 1;
      }
      return { X, X2, Z, rew, live };
    }
    // the critics toward y: ½(Q − y)², Adam; returns the loss
    fitCritics(Z, y, B) {
      let loss = 0;
      for (const q of [this.q1, this.q2]) {
        const out = q.forward(Z, B), d = new Float32Array(B);
        for (let b = 0; b < B; b++) { const e = out[b] - y[b]; loss += (0.5 * e * e) / B; d[b] = e / B; }
        q.zero(); q.backward(d, false); q.adam(this.criticLr, 10);
      }
      return loss / 2;
    }
    sacUpdate(opt) {
      const B = this.offB || Math.max(64, this.B), S = this.inSize, dA = this.dA, D = S + dA, K = this.K, g = this.gamma, al = this.alpha, { X, X2, Z, rew, live } = this.offBatch(B);
      // the target: r + γ (the smaller slow critic's value of the next state and a fresh try there − α log π of that try)
      const out2 = Float64Array.from(this.net.forward(X2, B, false).subarray(0, B * K)), Z2 = new Float32Array(B * D), lp2 = new Float64Array(B);
      for (let b = 0; b < B; b++) { const tr = this.sacTry(out2, b * K, Float64Array.from({ length: dA }, () => this.rng.gauss())); Z2.set(X2.subarray(b * S, b * S + S), b * D); for (let j = 0; j < dA; j++) Z2[b * D + S + j] = tr.a[j]; lp2[b] = tr.logp; }
      const t1 = Float64Array.from(this.q1t.forward(Z2, B)), t2 = this.q2t.forward(Z2, B), y = new Float64Array(B);
      for (let b = 0; b < B; b++) y[b] = rew[b] + g * live[b] * (Math.min(t1[b], t2[b]) - al * lp2[b]);
      this.criticLoss = this.fitCritics(Z, y, B);
      // the policy: tries the critics value more, and a wider curve (α log π smaller); the gradient through each try
      let lpSum = 0;
      const r = this.net.stepWith(X, B, (out, d) => {
        const Za = new Float32Array(B * D), tries = [];
        for (let b = 0; b < B; b++) { const z = Float64Array.from({ length: dA }, () => this.rng.gauss()), tr = this.sacTry(out, b * K, z); tries.push({ z, tr }); Za.set(X.subarray(b * S, b * S + S), b * D); for (let j = 0; j < dA; j++) Za[b * D + S + j] = tr.a[j]; lpSum += tr.logp; }
        const v1 = Float64Array.from(this.q1.forward(Za, B)), v2 = this.q2.forward(Za, B), d1 = new Float32Array(B), d2 = new Float32Array(B);
        let loss = 0;
        for (let b = 0; b < B; b++) { const lo = Math.min(v1[b], v2[b]); loss += (al * tries[b].tr.logp - lo) / B; if (v1[b] <= v2[b]) d1[b] = 1; else d2[b] = 1; }
        // dQ/d(try): back through the critic that gave the smaller value (their own gradients are thrown away)
        const g1 = this.q1.backward(d1, true); this.q1.zero();
        const g2 = this.q2.backward(d2, true); this.q2.zero();
        for (let b = 0; b < B; b++) {
          const { z, tr } = tries[b], o = b * K;
          for (let j = 0; j < dA; j++) {
            const a = tr.a[j], dq = g1[b * D + S + j] + g2[b * D + S + j], raw = out[o + dA + j], ls = Math.max(-5, Math.min(2, raw)), sg = Math.exp(ls), du = al * 2 * a - dq * (1 - a * a);
            d[o + j] = du / B;
            d[o + dA + j] = raw > -5 && raw < 2 ? (al * -1 + du * sg * z[j]) / B : 0;
          }
        }
        return loss;
      }, opt);
      // α: up when the curves are narrower than wanted, down when wider
      const A = this.aOpt, ga = -(lpSum / B + this.entTarget);
      A.t++; A.m = 0.9 * A.m + 0.1 * ga; A.v = 0.999 * A.v + 0.001 * ga * ga;
      this.logAlpha = Math.max(-12, Math.min(2, this.logAlpha - (0.0003 * A.m / (1 - 0.9 ** A.t)) / (Math.sqrt(A.v / (1 - 0.999 ** A.t)) + 1e-8)));
      this.q1t.polyak(this.q1, this.tau); this.q2t.polyak(this.q2, this.tau);
      return r;
    }
    td3Update(opt) {
      const B = this.offB || Math.max(64, this.B), S = this.inSize, dA = this.dA, D = S + dA, K = this.K, g = this.gamma, { X, X2, Z, rew, live } = this.offBatch(B);
      // the target: the slow copy's try at the next state, with clipped noise (smoothing), valued by the smaller slow critic
      const out2 = this.target.forward(X2, B, false), Z2 = new Float32Array(B * D);
      for (let b = 0; b < B; b++) { Z2.set(X2.subarray(b * S, b * S + S), b * D); for (let j = 0; j < dA; j++) Z2[b * D + S + j] = Math.max(-1, Math.min(1, Math.tanh(out2[b * K + j]) + Math.max(-0.5, Math.min(0.5, 0.2 * this.rng.gauss())))); }
      const t1 = Float64Array.from(this.q1t.forward(Z2, B)), t2 = this.q2t.forward(Z2, B), y = new Float64Array(B);
      for (let b = 0; b < B; b++) y[b] = rew[b] + g * live[b] * Math.min(t1[b], t2[b]);
      this.criticLoss = this.fitCritics(Z, y, B);
      this.td3n = (this.td3n || 0) + 1;
      if (this.td3n % 2) return { loss: this.criticLoss, gradNorm: 0 };
      // every second step: the policy toward tries the first critic values more, and the slow copies follow
      const r = this.net.stepWith(X, B, (out, d) => {
        const Za = new Float32Array(B * D);
        for (let b = 0; b < B; b++) { Za.set(X.subarray(b * S, b * S + S), b * D); for (let j = 0; j < dA; j++) Za[b * D + S + j] = Math.tanh(out[b * K + j]); }
        const v = this.q1.forward(Za, B), dq = new Float32Array(B).fill(1);
        let loss = 0; for (let b = 0; b < B; b++) loss -= v[b] / B;
        const gq = this.q1.backward(dq, true); this.q1.zero();
        for (let b = 0; b < B; b++) for (let j = 0; j < dA; j++) { const a = Za[b * D + S + j]; d[b * K + j] = (-gq[b * D + S + j] * (1 - a * a)) / B; }
        return loss;
      }, opt);
      const w = this.net.getWeights(), wt = this.target.getWeights();
      for (let i = 0; i < w.length; i++) wt[i] += this.tau * (w[i] - wt[i]);
      this.target.setWeights(wt);
      this.q1t.polyak(this.q1, this.tau); this.q2t.polyak(this.q2, this.tau);
      return r;
    }
    // ---------------------------------------------------------------- Copy you (behaviour cloning)
    /* A batch of your moves (this.demos: { obs, acts }, kept while you played), and a step toward picking each: the
       softmax of its move outputs against your move, cross-entropy, as a classifier learns. this.agree: how often its
       best move is yours, on the batches it learned from (a running average); it stands in for the return on the
       curve, since it plays no episodes while it learns. */
    bc(opt) {
      const D = this.demos, n = D ? D.acts.length : 0;
      if (!n) return { loss: 0, gradNorm: 0 };
      const B = Math.min(this.B, n), X = new Float32Array(B * this.inSize), pick = new Int32Array(B), K = this.K, nA = this.nA, p = this.p;
      for (let b = 0; b < B; b++) { const i = this.rng.int(n); pick[b] = i; X.set(D.obs[i], b * this.inSize); }
      let agree = 0;
      const r = this.net.stepWith(X, B, (out, d) => {
        let loss = 0;
        for (let b = 0; b < B; b++) {
          const o = b * K, a = D.acts[pick[b]];
          softmax(out, o, nA, p);
          loss -= Math.log(Math.max(1e-9, p[a]));
          if (argmax(out, o, nA) === a) agree++;
          for (let c = 0; c < nA; c++) d[o + c] = (p[c] - (c === a ? 1 : 0)) / B;
        }
        return loss / B;
      }, opt);
      this.agree = this.agree === undefined ? agree / B : 0.97 * this.agree + 0.03 * (agree / B);
      this.retEma = this.agree;
      return r;
    }

    // ---------------------------------------------------------------- Which did better? (preferences)
    /* The reward model: a small network of its own (two layers of 32) that gives each move a reward from what was seen
       and the move made. Pairs of tries are compared: two episodes of the current policy from the same start, the
       second with a quarter of its moves at random (PREF_EPS), so they differ by their moves and not by their luck (a
       policy that has made up its mind would otherwise play the same try twice, a tie that says nothing; and from two
       starts a lucky start can take the credit): by you, or by the world's own reward (s.prefAsk). The reward model learns
       so that the try with the larger total of its rewards is the one preferred: P(A better) = σ(ΣrA − ΣrB), the
       Bradley–Terry model, a logistic loss. PPO meanwhile learns from its rewards in place of the world's. The first
       pairs come one after another, then one every s.prefEvery steps. With whole episodes of different lengths the
       comparisons also fix whether a move is worth anything at all (a constant added to every reward changes the
       totals of a long and a short try differently), so lasting longer can be learned as better, or worse. */
    prefInit() {
      const s = this.s;
      this.rm = NA.bx.build([{ type: 'dense', units: 32, act: 'relu' }, { type: 'dense', units: 32, act: 'relu' }], { kind: 'vector', dim: this.inSize + this.nA }, { kind: 'value', k: 1 }, { seed: this.seed * 31 + 7 });
      this.rmX = new Float32Array(this.inSize + this.nA);
      this.prefEps = []; this.prefs = []; this.pairsMade = 0; this.pending = null; this.nextAsk = 0; this.rmLoss = null;
      this.prefAsk = s.prefAsk === 'world' ? 'world' : 'you'; this.prefEvery = Math.max(20, Math.round(s.prefEvery) || 300);
      this.prefWorld = E().make(this.worldId, this.ws);
    }
    // the reward model's reward for a move made after seeing o
    rhat(o, a) { const X = this.rmX; X.fill(0); X.set(o); X[this.inSize + a] = 1; return this.rm.forward(X, 1, false)[0]; }
    // one try of the current policy from the start of seed, its moves sampled (and a share eps at random); keeps the pictures of it when you answer
    prefEpisode(seed, pictures, eps = 0) {
      const W = this.prefWorld, rng = new NA.RNG(seed), most = Math.min(400, W.maxSteps || 400), E1 = { obs: [], acts: [], ret: 0, snaps: pictures ? [] : null, end: null };
      let o = W.reset(rng), done = false;
      if (pictures) E1.snaps.push(W.snap());
      while (!done && E1.acts.length < most) {
        const p = softmax(this.outputs(o), 0, this.nA, this.p), a = eps && this.rng.next() < eps ? this.rng.int(this.nA) : pick(p, this.nA, this.rng), t = W.step(a, rng);
        E1.obs.push(Float32Array.from(o)); E1.acts.push(a); E1.ret += t.r; o = t.obs; done = t.done;
        if (pictures) E1.snaps.push(W.snap());
      }
      E1.end = W.end || (done ? 'over' : 'time');
      this.envSteps += E1.acts.length;
      return E1;
    }
    // a pair due, or one waiting for your answer: { partial, ask } stops the steps until it is answered
    prefDue() {
      if (this.pending) return { partial: true, ask: this.pending };
      if (this.updates < this.nextAsk) return null;
      const seed = 5000 + this.pairsMade++, you = this.prefAsk === 'you', A = this.prefEpisode(seed, you, 0), B = this.prefEpisode(seed, you, PREF_EPS);
      const pair = { a: this.prefEps.push(A) - 1, b: this.prefEps.push(B) - 1, A, B, n: this.pairsMade };
      // (the first four one after another, before it learns anything from them; then one every so many steps)
      // (the world answers at once, so it is asked five times as often as you are)
      this.nextAsk = this.updates + (this.prefs.length < 3 ? 0 : this.prefAsk === 'world' ? Math.max(10, Math.round(this.prefEvery / 5)) : this.prefEvery);
      if (!you) { this.answer(A.ret > B.ret + 1e-9 ? 1 : B.ret > A.ret + 1e-9 ? 0 : 0.5, pair); return null; }
      this.pending = pair;
      return { partial: true, ask: pair };
    }
    /* An answer: 1 (the first try did better), 0 (the second), 0.5 (about the same), null (skip). The reward model
       then learns from every answer so far. The world's own verdict is kept beside yours, to show how far they agree. */
    answer(y, pair = this.pending) {
      if (!pair) return;
      this.pending = null;
      if (y !== null && y !== undefined) {
        const w = pair.A.ret > pair.B.ret + 1e-9 ? 1 : pair.B.ret > pair.A.ret + 1e-9 ? 0 : 0.5;
        this.prefs.push({ a: pair.a, b: pair.b, y, world: w });
        if (pair.A.snaps) { pair.A.snaps = null; pair.B.snaps = null; } // (the pictures are only for answering)
        this.fitRM(Math.min(400, 40 + 10 * this.prefs.length));
      }
    }
    // the reward model's steps: a pair at a time, the logistic loss of the Bradley–Terry model, Adam
    fitRM(steps) {
      if (!this.prefs.length) return;
      const D = this.inSize + this.nA, opt = { id: 'adam', lr: 0.003, clip: 1 };
      let total = 0;
      for (let s = 0; s < steps; s++) {
        const q = this.prefs[this.rng.int(this.prefs.length)], A = this.prefEps[q.a], B = this.prefEps[q.b], na = A.acts.length, n = na + B.acts.length;
        if (!n) continue;
        const X = new Float32Array(n * D);
        let k = 0;
        for (const T of [A, B]) for (let t = 0; t < T.acts.length; t++, k++) { X.set(T.obs[t], k * D); X[k * D + this.inSize + T.acts[t]] = 1; }
        const r = this.rm.stepWith(X, n, (out, d) => {
          let sa = 0, sb = 0;
          for (let i = 0; i < n; i++) if (i < na) sa += out[i]; else sb += out[i];
          const z = Math.max(-30, Math.min(30, sa - sb)), pA = 1 / (1 + Math.exp(-z)), g = pA - q.y;
          // (and a little pull of every reward toward 0: the comparisons fix their differences, this their size)
          for (let i = 0; i < n; i++) d[i] = (i < na ? g : -g) + 0.001 * out[i];
          return -(q.y * Math.log(Math.max(1e-9, pA)) + (1 - q.y) * Math.log(Math.max(1e-9, 1 - pA)));
        }, opt);
        total += r.loss;
      }
      this.rmLoss = total / steps;
    }
    /* How the answers so far line up: the reward model with the answers it learned from (does the try with the larger
       total of its rewards win?), and the answers with the world's own reward. Pairs called about the same are left out. */
    prefStats() {
      const P = (this.prefs || []).filter((q) => q.y !== 0.5), sum = (T) => { let s = 0; for (let t = 0; t < T.acts.length; t++) s += this.rhat(T.obs[t], T.acts[t]); return s; };
      let rmOk = 0, worldOk = 0, worldN = 0;
      for (const q of P) {
        const A = this.prefEps[q.a], B = this.prefEps[q.b];
        if ((sum(A) > sum(B)) === (q.y === 1)) rmOk++;
        if (q.world !== 0.5) { worldN++; if (q.world === q.y) worldOk++; }
      }
      return { answered: (this.prefs || []).length, rm: P.length ? rmOk / P.length : null, world: worldN ? worldOk / worldN : null, loss: this.rmLoss };
    }

    // how many of a batch's moves now lie beyond the clip's band (their probability more than 20% from when played)
    clipShareOf(all) {
      const n = all.length, X = new Float32Array(n * this.inSize), K = this.K, nA = this.nA, p = new Float64Array(nA);
      all.forEach((s, k) => X.set(s.o, k * this.inSize));
      const out = this.net.forward(X, n, false);
      let out20 = 0;
      for (let k = 0; k < n; k++) { softmax(out, k * K, nA, p); const ratio = Math.exp(Math.log(Math.max(1e-9, p[all[k].a])) - all[k].logp); if (ratio < 1 - this.clipC || ratio > 1 + this.clipC) out20++; }
      this.clipShare = out20 / n;
      (this.clipHist || (this.clipHist = [])).push({ update: this.updates, share: this.clipShare });
      if (this.clipHist.length > 2000) this.clipHist = this.clipHist.filter((h, i) => i % 2 === 0 || i === this.clipHist.length - 1);
    }
  }
  /* Population-based training (Jaderberg and others, 2017; s.pbt learners): several learners side by side, each its own
     network from its own seed and a learning rate of its own (the one set, times 0.3 to 3); every 500 steps each is
     scored, and the worst quarter take the weights and the learning rate of the best quarter, nudged by ×0.8 or ×1.25.
     The best learner is the one on show; whatever the page asks of the trainer goes to it. */
  class Pbt {
    constructor(o, n) {
      this.rng = new NA.RNG((o.seed || 1) * 131 + 5); this.n = n; this.every = 500; this.rounds = 0; this.hist = []; this.events = 0;
      this.learners = NA.range(n).map((k) => { const net = k ? o.makeNet() : o.net; if (k) { const r = new NA.RNG((o.seed || 1) * 977 + k), w = net.getWeights(); for (let i = 0; i < w.length; i++) w[i] *= 0.5 + r.next(); net.setWeights(w); } const T = new Trainer(Object.assign({}, o, { net, seed: (o.seed || 1) + 1000 * k })); T.lrMul = k ? Math.exp(this.rng.uniform(Math.log(0.3), Math.log(3))) : 1; T.pbtId = k + 1; return T; });
      this.best = this.learners[0]; this.pbt = true;
    }
    step(opt, deadline) {
      let loss = 0, gn = 0, n = 0;
      for (const T of this.learners) { const r = T.step(Object.assign({}, opt, { lr: (opt && opt.lr ? opt.lr : 0.001) * T.lrMul }), deadline); if (r && !r.partial) { loss += r.loss || 0; gn += r.gradNorm || 0; n++; } }
      if (++this.rounds % this.every === 0) this.exploit();
      return { loss: n ? loss / n : 0, gradNorm: n ? gn / n : 0 };
    }
    exploit() {
      const sc = this.learners.map((T) => T.score()), order = NA.range(this.n).sort((a, b) => sc[b] - sc[a]), q = Math.max(1, Math.floor(this.n / 4)), moved = [];
      for (let i = 0; i < q; i++) {
        const lo = this.learners[order[this.n - 1 - i]], hi = this.learners[order[i]];
        if (lo === hi || sc[order[this.n - 1 - i]] >= sc[order[i]]) continue;
        lo.net.setWeights(hi.net.getWeights()); if (lo.target && hi.target) lo.target.setWeights(hi.target.getWeights());
        lo.lrMul = hi.lrMul * (this.rng.next() < 0.5 ? 0.8 : 1.25); moved.push([lo.pbtId, hi.pbtId]);
      }
      this.best = this.learners[order[0]];
      this.hist.push({ round: this.rounds, scores: sc, lr: this.learners.map((T) => T.lrMul), moved });
      if (this.hist.length > 400) this.hist.shift();
    }
    score() { return this.best.score(); }
  }
  // the trainer the page talks to: the population's own fields, everything else the best learner's
  BRL.pbtOk = (method) => !BRL.isEvo(method) && method !== 'bc' && method !== 'pref';
  const makePbt = (o, n) => { const P = new Pbt(o, n); return new Proxy(P, { get: (t, k) => (k in t ? t[k] : typeof t.best[k] === 'function' ? t.best[k].bind(t.best) : t.best[k]), set: (t, k, v) => { if (k in t) t[k] = v; else t.best[k] = v; return true; } }); };
  BRL.Trainer = Trainer;
  BRL.make = (o) => (o.s && o.s.pbt > 1 && BRL.pbtOk(o.method) && o.makeNet ? makePbt(o, Math.min(8, Math.round(o.s.pbt))) : new Trainer(o));
  NA.brl = BRL;
})(typeof globalThis !== 'undefined' && globalThis.NA ? globalThis.NA : window.NA);
