const msg = document.getElementById("msg");
navigator.mediaDevices.getUserMedia({ video: true, audio: false })
  .then(stream => {
    stream.getTracks().forEach(t => t.stop());
    msg.textContent = "Done. Close this tab and press Start camera in the side panel.";
    msg.className = "ok";
  })
  .catch(e => {
    msg.textContent = `Camera access was not granted (${e.name}). Allow it from the camera icon in the address bar, then reload this page.`;
    msg.className = "error";
  });
