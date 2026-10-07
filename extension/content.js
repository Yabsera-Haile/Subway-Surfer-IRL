(() => {
  const SOURCE = "lane-gestures";
  const MIN_AREA = 200 * 150;

  function gameCanvas() {
    let best = null, bestArea = 0;
    for (const c of document.querySelectorAll("canvas")) {
      const r = c.getBoundingClientRect();
      if (r.width * r.height > bestArea) {
        best = c;
        bestArea = r.width * r.height;
      }
    }
    return bestArea >= MIN_AREA ? best : null;
  }

  function fire(target, type, k) {
    target.dispatchEvent(new CustomEvent("lane-gestures-key", {
      bubbles: true,
      detail: JSON.stringify({ type, key: k.key, code: k.code, keyCode: k.keyCode }),
    }));
  }

  chrome.runtime.onMessage.addListener((msg, _sender, sendResponse) => {
    if (msg?.source !== SOURCE) return;
    const canvas = gameCanvas();
    if (canvas && msg.type === "key") {
      fire(canvas, "keydown", msg.key);
      setTimeout(() => fire(canvas, "keyup", msg.key), msg.holdMs);
    }
    if (!canvas) return sendResponse(null);
    const r = canvas.getBoundingClientRect();
    sendResponse({ host: location.host || location.href, width: Math.round(r.width), height: Math.round(r.height) });
  });
})();
