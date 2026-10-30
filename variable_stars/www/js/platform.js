// Abstraction de plateforme : navigateur / Electron (PC) / Capacitor (Android).
(function () {
  'use strict';
  const cap = window.Capacitor;
  const isNative = !!(cap && cap.isNativePlatform && cap.isNativePlatform());
  const plugin = (name) => (isNative && cap.Plugins ? cap.Plugins[name] : null);

  function toBase64(text) {
    const bytes = new TextEncoder().encode(text);
    let bin = '';
    for (let i = 0; i < bytes.length; i += 0x8000) bin += String.fromCharCode.apply(null, bytes.subarray(i, i + 0x8000));
    return btoa(bin);
  }

  // content : chaîne de texte, ou { base64 } pour un contenu binaire (ex. PNG)
  // Renvoie le nom du fichier enregistré, ou null si l'utilisateur a refusé.
  async function saveFile(filename, content, mime) {
    const runtime = window.claude && typeof window.claude.use === 'function' ? await window.claude.use('downloads') : null;
    if (runtime) {
      const data = typeof content === 'string'
        ? content
        : new Blob([Uint8Array.from(atob(content.base64), (c) => c.charCodeAt(0))], { type: mime || 'application/octet-stream' });
      try { await runtime.save({ filename, data }); return filename; } catch (e) { return null; }
    }
    const fs = plugin('Filesystem');
    if (fs) {
      const data = typeof content === 'string' ? toBase64(content) : content.base64;
      let res;
      try {
        res = await fs.writeFile({ path: 'AstroVar/' + filename, data, directory: 'DOCUMENTS', recursive: true });
      } catch (e) {
        res = await fs.writeFile({ path: filename, data, directory: 'CACHE', recursive: true });
      }
      const share = plugin('Share');
      if (share) {
        try { await share.share({ title: filename, url: res.uri, dialogTitle: 'Enregistrer / partager ' + filename }); } catch (e) { /* partage annulé */ }
      }
      return res.uri;
    }
    const blob = typeof content === 'string'
      ? new Blob([content], { type: mime || 'text/plain;charset=utf-8' })
      : new Blob([Uint8Array.from(atob(content.base64), (c) => c.charCodeAt(0))], { type: mime || 'application/octet-stream' });
    const a = document.createElement('a');
    a.href = URL.createObjectURL(blob);
    a.download = filename;
    document.body.appendChild(a);
    a.click();
    setTimeout(() => { URL.revokeObjectURL(a.href); a.remove(); }, 1000);
    return filename;
  }

  async function openExternal(url) {
    const browser = plugin('Browser');
    if (browser) return browser.open({ url });
    window.open(url, '_blank', 'noopener');
  }

  async function getPosition() {
    const geo = plugin('Geolocation');
    if (geo) {
      const p = await geo.getCurrentPosition({ enableHighAccuracy: false, timeout: 15000 });
      return p.coords;
    }
    return new Promise((resolve, reject) => {
      if (!navigator.geolocation) return reject(new Error('Géolocalisation non disponible'));
      navigator.geolocation.getCurrentPosition((p) => resolve(p.coords), reject, { timeout: 15000 });
    });
  }

  const store = {
    get(key, def) { try { const v = localStorage.getItem('astrovar.' + key); return v == null ? def : JSON.parse(v); } catch (e) { return def; } },
    set(key, val) { try { localStorage.setItem('astrovar.' + key, JSON.stringify(val)); } catch (e) { /* stockage indisponible */ } },
  };

  window.Platform = { isNative, isElectron: !!window.astrovarElectron, saveFile, openExternal, getPosition, store };
})();
