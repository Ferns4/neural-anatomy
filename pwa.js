/* pwa.js — website only: offline support through a service worker, and an Install button. */
(function () {
  'use strict';
  let embedded = true;
  try { embedded = window.top !== window; } catch (e) { embedded = true; }
  if (embedded) return;

  if ('serviceWorker' in navigator) {
    window.addEventListener('load', () => { navigator.serviceWorker.register('sw.js').catch(() => { /* offline mode unavailable */ }); });
  }

  const btn = document.getElementById('installBtn');
  const help = document.getElementById('helpInstall');
  const standalone = window.matchMedia('(display-mode: standalone)').matches || window.navigator.standalone === true;
  if (help && !standalone) help.hidden = false;
  if (!btn || standalone) return;

  let deferred = null;
  const openHelp = () => {
    const h = document.getElementById('help');
    h.hidden = false;
    help.scrollIntoView({ block: 'nearest' });
    const close = h.querySelector('[data-close]');
    if (close) close.focus();
  };
  window.addEventListener('beforeinstallprompt', (e) => { e.preventDefault(); deferred = e; btn.hidden = false; });
  window.addEventListener('appinstalled', () => { deferred = null; btn.hidden = true; });
  btn.addEventListener('click', async () => {
    if (!deferred) { openHelp(); return; }
    deferred.prompt();
    try { await deferred.userChoice; } catch (e) { /* dismissed */ }
    deferred = null;
    btn.hidden = true;
  });
  // iPhone and iPad never fire beforeinstallprompt: show the button with instructions instead.
  const ios = /iphone|ipad|ipod/i.test(navigator.userAgent) || (navigator.platform === 'MacIntel' && navigator.maxTouchPoints > 1);
  if (ios) btn.hidden = false;
})();
