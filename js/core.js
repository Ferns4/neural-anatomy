/* core.js — shared utilities: random numbers, formatting, colour, DOM and canvas helpers. */
(function (NA) {
  'use strict';

  // ---------------------------------------------------------------- random
  function mulberry32(a) {
    return function () {
      a |= 0; a = (a + 0x6d2b79f5) | 0;
      let t = Math.imul(a ^ (a >>> 15), 1 | a);
      t = (t + Math.imul(t ^ (t >>> 7), 61 | t)) ^ t;
      return ((t ^ (t >>> 14)) >>> 0) / 4294967296;
    };
  }

  class RNG {
    constructor(seed) { this.next = mulberry32((seed >>> 0) || 1); this._spare = null; }
    uniform(a = 0, b = 1) { return a + (b - a) * this.next(); }
    int(n) { return Math.floor(this.next() * n); }
    gauss(mu = 0, sd = 1) {
      if (this._spare !== null) { const s = this._spare; this._spare = null; return mu + sd * s; }
      let u, v, r;
      do { u = this.next() * 2 - 1; v = this.next() * 2 - 1; r = u * u + v * v; } while (r >= 1 || r === 0);
      const m = Math.sqrt(-2 * Math.log(r) / r);
      this._spare = v * m;
      return mu + sd * u * m;
    }
    shuffle(arr) {
      for (let i = arr.length - 1; i > 0; i--) { const j = this.int(i + 1); const t = arr[i]; arr[i] = arr[j]; arr[j] = t; }
      return arr;
    }
    pick(arr) { return arr[this.int(arr.length)]; }
  }

  // ---------------------------------------------------------------- math
  const clamp = (v, a, b) => (v < a ? a : v > b ? b : v);
  const lerp = (a, b, t) => a + (b - a) * t;
  const smoothstep = (t) => t * t * (3 - 2 * t);
  function range(n) { const a = new Array(n); for (let i = 0; i < n; i++) a[i] = i; return a; }
  function maxAbs(arr) { let m = 0; for (let i = 0; i < arr.length; i++) { const v = Math.abs(arr[i]); if (v > m) m = v; } return m; }
  function quantile(sorted, q) {
    if (!sorted.length) return 0;
    const p = (sorted.length - 1) * q, lo = Math.floor(p), hi = Math.ceil(p);
    return lo === hi ? sorted[lo] : sorted[lo] + (sorted[hi] - sorted[lo]) * (p - lo);
  }

  // ---------------------------------------------------------------- formatting
  const MINUS = '−';
  // a number in scientific notation as people write it: mantissa × 10 with a superscript exponent
  const SUPERS = '⁰¹²³⁴⁵⁶⁷⁸⁹';
  function sci(av) { const [m, e] = av.toExponential(1).split('e'), n = +e; return `${m}×10${n < 0 ? '⁻' : ''}${String(Math.abs(n)).split('').map((c) => SUPERS[c]).join('')}`; }
  function fmt(v, digits = 2) {
    if (v === null || v === undefined || Number.isNaN(v)) return '—';
    if (!isFinite(v)) return (v < 0 ? MINUS : '') + '∞';
    const av = Math.abs(v);
    let s;
    if (av !== 0 && (av < 0.5 * Math.pow(10, -digits) || av >= 1e5)) s = sci(av); // 3.5×10⁻¹⁴, not 3.5e-14
    else s = av.toFixed(digits);
    const isZero = /^0(\.0+)?$/.test(s);
    return (v < 0 && !isZero ? MINUS : '') + s;
  }
  // a number as code (Python or JS): a plain literal, ASCII minus and e for the exponent (fmt is for reading)
  function fmtCode(v, digits = 2) {
    if (v === null || v === undefined || Number.isNaN(v)) return 'float("nan")';
    if (!isFinite(v)) return v < 0 ? 'float("-inf")' : 'float("inf")';
    const av = Math.abs(v), s = av !== 0 && (av < 0.5 * Math.pow(10, -digits) || av >= 1e5) ? av.toExponential(1).replace('e+', 'e') : av.toFixed(digits);
    return (v < 0 && !/^0(\.0+)?$/.test(s) ? '-' : '') + s;
  }
  function fmtSigned(v, digits = 2) { const s = fmt(v, digits); return v > 0 && s !== fmt(0, digits) ? '+' + s : s; }
  function fmtPct(v, digits = 0) { return isFinite(v) ? (v * 100).toFixed(digits).replace('-', '\u2212') + '%' : '—'; }
  function fmtInt(n) { return Math.round(n).toLocaleString('en-US'); }
  function fmtCompact(n) {
    const a = Math.abs(n);
    if (a >= 1e6) return (n / 1e6).toFixed(a >= 1e7 ? 0 : 1) + 'M';
    if (a >= 1e3) return (n / 1e3).toFixed(a >= 1e4 ? 0 : 1) + 'K';
    return String(Math.round(n));
  }
  const SUB = '₀₁₂₃₄₅₆₇₈₉';
  function sub(n) { return String(n).split('').map((c) => SUB[+c] || c).join(''); }

  // ---------------------------------------------------------------- events
  function Emitter() {
    const map = new Map();
    return {
      on(type, fn) { if (!map.has(type)) map.set(type, new Set()); map.get(type).add(fn); return () => map.get(type).delete(fn); },
      emit(type, payload) { const s = map.get(type); if (s) s.forEach((fn) => fn(payload)); },
    };
  }

  // ---------------------------------------------------------------- colour
  function hexToRgb(hex) {
    hex = String(hex).trim().replace('#', '');
    if (hex.length === 3) hex = hex.split('').map((c) => c + c).join('');
    const n = parseInt(hex, 16);
    return [(n >> 16) & 255, (n >> 8) & 255, n & 255];
  }
  function parseColor(str) {
    str = String(str || '').trim();
    if (str.startsWith('#')) return hexToRgb(str);
    const m = str.match(/rgba?\(([^)]+)\)/);
    if (m) return m[1].split(',').slice(0, 3).map((v) => parseFloat(v));
    return [128, 128, 128];
  }
  const css = (rgb, a = 1) => (a >= 1 ? `rgb(${rgb[0] | 0},${rgb[1] | 0},${rgb[2] | 0})` : `rgba(${rgb[0] | 0},${rgb[1] | 0},${rgb[2] | 0},${a})`);

  // sRGB <-> OKLab for perceptually even ramps
  const toLin = (c) => { c /= 255; return c <= 0.04045 ? c / 12.92 : Math.pow((c + 0.055) / 1.055, 2.4); };
  const toSrgb = (c) => { const v = c <= 0.0031308 ? 12.92 * c : 1.055 * Math.pow(c, 1 / 2.4) - 0.055; return clamp(Math.round(v * 255), 0, 255); };
  function rgbToOklab(rgb) {
    const r = toLin(rgb[0]), g = toLin(rgb[1]), b = toLin(rgb[2]);
    const l = Math.cbrt(0.4122214708 * r + 0.5363325363 * g + 0.0514459929 * b);
    const m = Math.cbrt(0.2119034982 * r + 0.6806995451 * g + 0.1073969566 * b);
    const s = Math.cbrt(0.0883024619 * r + 0.2817188376 * g + 0.6299787005 * b);
    return [0.2104542553 * l + 0.793617785 * m - 0.0040720468 * s, 1.9779984951 * l - 2.428592205 * m + 0.4505937099 * s, 0.0259040371 * l + 0.7827717662 * m - 0.808675766 * s];
  }
  function oklabToRgb(lab) {
    const l = Math.pow(lab[0] + 0.3963377774 * lab[1] + 0.2158037573 * lab[2], 3);
    const m = Math.pow(lab[0] - 0.1055613458 * lab[1] - 0.0638541728 * lab[2], 3);
    const s = Math.pow(lab[0] - 0.0894841775 * lab[1] - 1.291485548 * lab[2], 3);
    return [toSrgb(4.0767416621 * l - 3.3077115913 * m + 0.2309699292 * s), toSrgb(-1.2684380046 * l + 2.6097574011 * m - 0.3413193965 * s), toSrgb(-0.0041960863 * l - 0.7034186147 * m + 1.707614701 * s)];
  }
  function mixLab(c1, c2, t) {
    const a = rgbToOklab(c1), b = rgbToOklab(c2);
    return oklabToRgb([lerp(a[0], b[0], t), lerp(a[1], b[1], t), lerp(a[2], b[2], t)]);
  }
  const mixRgb = (c1, c2, t) => [lerp(c1[0], c2[0], t), lerp(c1[1], c2[1], t), lerp(c1[2], c2[2], t)];

  // Build a lookup table of n entries going through the given colour stops (OKLab interpolation).
  function buildLUT(stops, n = 256) {
    const lut = new Uint8ClampedArray(n * 3);
    for (let i = 0; i < n; i++) {
      const t = i / (n - 1), seg = Math.min(stops.length - 2, Math.floor(t * (stops.length - 1)));
      const lt = t * (stops.length - 1) - seg;
      const c = mixLab(stops[seg], stops[seg + 1], lt);
      lut[i * 3] = c[0]; lut[i * 3 + 1] = c[1]; lut[i * 3 + 2] = c[2];
    }
    return lut;
  }

  // Theme tokens mirrored from CSS custom properties, refreshed when the theme changes.
  const theme = {
    tokens: {}, listeners: new Set(),
    read() {
      if (typeof document === 'undefined') return;
      const cs = getComputedStyle(document.documentElement);
      const names = ['bg', 'panel', 'panel-2', 'ink', 'ink-2', 'muted', 'line', 'line-2', 'pos', 'neg', 'c3', 'mid', 'grad', 'focus', 'good', 'warn', 'serious', 'critical', 's1', 's2', 's3', 's4', 's5', 's6', 's7', 's8'];
      names.forEach((n) => { this.tokens[n] = parseColor(cs.getPropertyValue('--' + n)); });
      const t = this.tokens;
      this.series = ['s1', 's2', 's3', 's4', 's5', 's6', 's7', 's8'].map((k) => css(t[k]));
      this.dark = rgbToOklab(t.panel)[0] < 0.5;
      this.div = buildLUT([t.neg, t.mid, t.pos], 257);
      this.divSoft = buildLUT([mixLab(t.neg, t.panel, 0.32), mixLab(t.mid, t.panel, 0.25), mixLab(t.pos, t.panel, 0.32)], 257);
      this.seq = buildLUT([t['panel-2'], mixLab(t.pos, t['panel-2'], 0.55), t.pos, mixLab(t.pos, t.ink, 0.35)], 256);
      this.gradLUT = buildLUT([t['panel-2'], mixLab(t.grad, t['panel-2'], 0.5), t.grad, mixLab(t.grad, t.ink, 0.3)], 256);
      this.classes = [t.neg, t.pos, t.c3];
      this.listeners.forEach((fn) => fn());
    },
    // value in [-1, 1] -> css colour from the diverging ramp
    divColor(v, alpha = 1, soft = false) {
      const lut = soft ? this.divSoft : this.div;
      const i = Math.round((clamp(v, -1, 1) + 1) * 128) * 3;
      return alpha >= 1 ? `rgb(${lut[i]},${lut[i + 1]},${lut[i + 2]})` : `rgba(${lut[i]},${lut[i + 1]},${lut[i + 2]},${alpha})`;
    },
    divRGB(v, soft = false) { const lut = soft ? this.divSoft : this.div; const i = Math.round((clamp(v, -1, 1) + 1) * 128) * 3; return [lut[i], lut[i + 1], lut[i + 2]]; },
    seqColor(v, alpha = 1, lut) { lut = lut || this.seq; const i = Math.round(clamp(v, 0, 1) * 255) * 3; return alpha >= 1 ? `rgb(${lut[i]},${lut[i + 1]},${lut[i + 2]})` : `rgba(${lut[i]},${lut[i + 1]},${lut[i + 2]},${alpha})`; },
    gradColor(v, alpha = 1) { return this.seqColor(v, alpha, this.gradLUT); },
    c(name, alpha = 1) { return css(this.tokens[name] || [128, 128, 128], alpha); },
    classColor(k, alpha = 1) { return css(this.classes[k] || this.tokens.muted, alpha); },
    onChange(fn) { this.listeners.add(fn); return () => this.listeners.delete(fn); },
  };

  // ---------------------------------------------------------------- DOM
  function el(tag, attrs, ...children) {
    const node = document.createElement(tag);
    if (attrs) {
      for (const k in attrs) {
        const v = attrs[k];
        if (v === undefined || v === null || v === false) continue;
        if (k === 'class') node.className = v;
        else if (k === 'text') node.textContent = v;
        else if (k === 'html') node.innerHTML = v; // only ever used with trusted, static markup
        else if (k === 'style' && typeof v === 'object') { for (const p in v) { if (p.startsWith('--')) node.style.setProperty(p, v[p]); else node.style[p] = v[p]; } } // custom properties need setProperty
        else if (k.startsWith('on') && typeof v === 'function') node.addEventListener(k.slice(2), v);
        else if (k === 'dataset') Object.assign(node.dataset, v);
        else node.setAttribute(k, v === true ? '' : v);
      }
    }
    for (const c of children.flat()) {
      if (c === null || c === undefined || c === false) continue;
      node.appendChild(typeof c === 'string' || typeof c === 'number' ? document.createTextNode(String(c)) : c);
    }
    return node;
  }
  const $ = (sel, root) => (root || document).querySelector(sel);
  const $$ = (sel, root) => Array.from((root || document).querySelectorAll(sel));
  function clear(node) { while (node.firstChild) node.removeChild(node.firstChild); return node; }

  // Size a canvas for the device pixel ratio; returns a context in CSS pixels.
  function fitCanvas(canvas, w, h) {
    const dpr = Math.min(2.5, (typeof window !== 'undefined' && window.devicePixelRatio) || 1);
    const W = Math.max(1, Math.round(w * dpr)), H = Math.max(1, Math.round(h * dpr));
    if (canvas.width !== W || canvas.height !== H) { canvas.width = W; canvas.height = H; }
    canvas.style.width = w + 'px'; canvas.style.height = h + 'px';
    const ctx = canvas.getContext('2d');
    ctx.setTransform(dpr, 0, 0, dpr, 0, 0);
    canvas._dpr = dpr;
    return ctx;
  }

  const prefersReducedMotion = () => typeof window !== 'undefined' && window.matchMedia && window.matchMedia('(prefers-reduced-motion: reduce)').matches;

  const store = {
    get(key, fallback) { try { const v = localStorage.getItem('na.' + key); return v === null ? fallback : JSON.parse(v); } catch (e) { return fallback; } },
    set(key, value) { try { localStorage.setItem('na.' + key, JSON.stringify(value)); } catch (e) { /* storage unavailable */ } },
  };

  Object.assign(NA, {
    RNG, clamp, lerp, smoothstep, range, maxAbs, quantile,
    MINUS, fmt, fmtCode, fmtSigned, fmtPct, fmtInt, fmtCompact, sub,
    Emitter, hexToRgb, parseColor, css, mixLab, mixRgb, buildLUT, theme,
    el, $, $$, clear, fitCanvas, prefersReducedMotion, store,
  });
})(typeof globalThis !== 'undefined' ? (globalThis.NA = globalThis.NA || {}) : (window.NA = window.NA || {}));
