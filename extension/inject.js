(() => {
  document.addEventListener("lane-gestures-key", e => {
    const { type, key, code, keyCode } = JSON.parse(e.detail);
    const k = new KeyboardEvent(type, { key, code, bubbles: true, cancelable: true });
    Object.defineProperty(k, "keyCode", { get: () => keyCode });
    Object.defineProperty(k, "which", { get: () => keyCode });
    e.target.dispatchEvent(k);
  }, true);
})();
