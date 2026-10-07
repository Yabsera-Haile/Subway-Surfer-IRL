const Scene = (() => {
  const GROUND = -0.85;
  const FOG = 0xcfe6f5;
  const QUALITY = {
    low: { pixelRatio: 0.7, antialias: false, shadows: false, buildings: 0.6, aniso: 1, maxFps: 30 },
    standard: { pixelRatio: 1, antialias: true, shadows: false, buildings: 1, aniso: 4, maxFps: 0 },
    high: { pixelRatio: 2, antialias: true, shadows: true, buildings: 1, aniso: 8, maxFps: 0 },
  };
  const SLEEPER_SPACING = 0.4;
  const SLEEPER_PATTERN = 10;
  const PORTAL_SPACING = 8;
  const DECOR_PERIOD = 48;
  const BRIDGE_Z = -30;
  const BRIDGE_Y = GROUND + 2.3;
  const STREET = 1.4;
  const TRACK_LEN = 76;
  const TRACK_Z = -28;
  const CAM_Z = -1.15;
  const FOG_NEAR = 14, FOG_FAR = 46;
  const DECOR_END = CAM_Z - FOG_FAR - 0.35;

  let renderer, scene, camera, sun, Q;
  let time = 0, attract = 0, visX = 0, camX = 0, phase = 0, policeZ = 0.6, policeSide = 1, pending = 0;
  let runner, inspector, finishArch, blobTex, bandTex;
  const scrollers = [];
  const pools = new Map();
  const lives = {};

  function tex(w, h, draw, repeat = [1, 1]) {
    const c = document.createElement("canvas");
    c.width = w;
    c.height = h;
    draw(c.getContext("2d"), w, h);
    const t = new THREE.CanvasTexture(c);
    t.colorSpace = THREE.SRGBColorSpace;
    t.wrapS = t.wrapT = THREE.RepeatWrapping;
    t.repeat.set(repeat[0], repeat[1]);
    t.anisotropy = Math.min(Q.aniso, renderer.capabilities.getMaxAnisotropy());
    return t;
  }

  function rand(a, b) {
    return a + Math.random() * (b - a);
  }

  function pick(list) {
    return list[Math.floor(Math.random() * list.length)];
  }

  function speckle(c, w, h, n, colors, rmin, rmax) {
    for (let i = 0; i < n; i++) {
      c.fillStyle = pick(colors);
      const r = rand(rmin, rmax);
      c.fillRect(Math.random() * w, Math.random() * h, r, r);
    }
  }

  const TAGS = ["SURF", "RUN!", "ZOOM", "GO GO", "JUMP", "WOW", "DASH", "HYPE", "FLY", "BOOM"];
  const SPRAY = [["#ff4d6d", "#ffd23f"], ["#3a86ff", "#8ecae6"], ["#06d6a0", "#ffe66d"], ["#ff9f1c", "#ff4d6d"],
                 ["#c77dff", "#ff70a6"], ["#00f5d4", "#3a86ff"]];

  function graffiti(c, text, x, y, size, rot) {
    const [a, b] = pick(SPRAY);
    c.save();
    c.translate(x, y);
    c.rotate(rot);
    c.font = `900 ${size}px "Arial Black", Impact, sans-serif`;
    c.textAlign = "center";
    c.textBaseline = "middle";
    c.lineJoin = "round";
    c.lineWidth = size * 0.28;
    c.strokeStyle = "#141414";
    c.strokeText(text, 0, 0);
    c.lineWidth = size * 0.12;
    c.strokeStyle = "#ffffff";
    c.strokeText(text, 0, 0);
    const g = c.createLinearGradient(0, -size / 2, 0, size / 2);
    g.addColorStop(0, b);
    g.addColorStop(1, a);
    c.fillStyle = g;
    c.fillText(text, 0, 0);
    c.globalAlpha = 0.55;
    c.fillStyle = a;
    for (let i = 0; i < 4; i++) c.fillRect(rand(-size, size), size * 0.35, size * 0.05, rand(size * 0.2, size * 0.6));
    c.restore();
  }

  function skyTexture() {
    return tex(16, 256, (c, w, h) => {
      const g = c.createLinearGradient(0, 0, 0, h);
      g.addColorStop(0, "#3d8fe0");
      g.addColorStop(0.55, "#8cc8f2");
      g.addColorStop(1, "#d6edfb");
      c.fillStyle = g;
      c.fillRect(0, 0, w, h);
    });
  }

  function gravelTexture() {
    return tex(256, 256, (c, w, h) => {
      c.fillStyle = "#857c71";
      c.fillRect(0, 0, w, h);
      speckle(c, w, h, 9000, ["#6e665c", "#9a9184", "#a89f92", "#5c554d", "#7a6f62", "#b3aa9c"], 1, 3.5);
    }, [4, TRACK_LEN]);
  }

  function concreteTexture() {
    return tex(256, 256, (c, w, h) => {
      c.fillStyle = "#bdb6aa";
      c.fillRect(0, 0, w, h);
      speckle(c, w, h, 2500, ["#aaa397", "#c9c2b6", "#a39c90"], 1, 2.5);
      c.strokeStyle = "rgba(60,55,50,0.35)";
      c.lineWidth = 3;
      for (let y = 0; y <= h; y += 64) {
        c.beginPath();
        c.moveTo(0, y);
        c.lineTo(w, y);
        c.stroke();
      }
    }, [1, TRACK_LEN * 1.2]);
  }

  function wallTexture() {
    return tex(2048, 256, (c, w, h) => {
      c.fillStyle = "#c7c0b3";
      c.fillRect(0, 0, w, h);
      speckle(c, w, h, 6000, ["#b5aea1", "#d4cdc1", "#aea79a"], 1, 3);
      for (let i = 0; i < 18; i++) {
        c.fillStyle = `rgba(70,60,50,${rand(0.04, 0.1)})`;
        c.beginPath();
        c.ellipse(rand(0, w), rand(h * 0.4, h), rand(30, 120), rand(10, 40), 0, 0, Math.PI * 2);
        c.fill();
      }
      c.fillStyle = "rgba(0,0,0,0.25)";
      for (let x = 0; x <= w; x += 256) c.fillRect(x, 0, 4, h);
      const g = c.createLinearGradient(0, h * 0.75, 0, h);
      g.addColorStop(0, "rgba(40,35,30,0)");
      g.addColorStop(1, "rgba(40,35,30,0.45)");
      c.fillStyle = g;
      c.fillRect(0, 0, w, h);
      let x = 60;
      while (x < w - 200) {
        const text = pick(TAGS), size = rand(70, 105);
        c.font = `900 ${size}px "Arial Black", Impact, sans-serif`;
        const half = c.measureText(text).width / 2 + size * 0.2;
        if (x + 2 * half > w - 20) break;
        graffiti(c, text, x + half, h * rand(0.45, 0.6), size, rand(-0.1, 0.1));
        x += 2 * half + rand(60, 260);
      }
    }, [TRACK_LEN / 12, 1]);
  }

  function windowsTexture() {
    return tex(256, 512, (c, w, h) => {
      c.fillStyle = "#f2eee8";
      c.fillRect(0, 0, w, h);
      const cols = 4, rows = 8, cw = w / cols, rh = h / rows;
      for (let r = 0; r < rows; r++) {
        c.fillStyle = "rgba(0,0,0,0.08)";
        c.fillRect(0, r * rh, w, 5);
        for (let k = 0; k < cols; k++) {
          const x = k * cw + cw * 0.18, y = r * rh + rh * 0.22, ww = cw * 0.64, hh = rh * 0.6;
          if (r === 0 && k === 0) continue;
          c.fillStyle = "#6b6157";
          c.fillRect(x - 3, y - 3, ww + 6, hh + 6);
          const lit = Math.random() < 0.18;
          const g = c.createLinearGradient(x, y, x + ww, y + hh);
          g.addColorStop(0, lit ? "#ffe9a8" : "#7fb3d9");
          g.addColorStop(1, lit ? "#f5c66b" : "#2e5f8a");
          c.fillStyle = g;
          c.fillRect(x, y, ww, hh);
          c.fillStyle = "rgba(255,255,255,0.35)";
          c.beginPath();
          c.moveTo(x, y + hh * 0.7);
          c.lineTo(x + ww * 0.4, y);
          c.lineTo(x + ww * 0.6, y);
          c.lineTo(x, y + hh);
          c.fill();
        }
      }
    });
  }

  function trainSideTexture(base, stripe, tagged) {
    return tex(512, 128, (c, w, h) => {
      c.fillStyle = base;
      c.fillRect(0, 0, w, h);
      const top = c.createLinearGradient(0, 0, 0, h * 0.2);
      top.addColorStop(0, "rgba(255,255,255,0.35)");
      top.addColorStop(1, "rgba(255,255,255,0)");
      c.fillStyle = top;
      c.fillRect(0, 0, w, h * 0.2);
      const glass = (x, y, ww, hh) => {
        c.fillStyle = "#1d2433";
        c.fillRect(x - 3, y - 3, ww + 6, hh + 6);
        const g = c.createLinearGradient(x, y, x, y + hh);
        g.addColorStop(0, "#9fd3f7");
        g.addColorStop(1, "#24476e");
        c.fillStyle = g;
        c.fillRect(x, y, ww, hh);
        c.fillStyle = "rgba(255,255,255,0.4)";
        c.fillRect(x + ww * 0.15, y, ww * 0.12, hh);
      };
      for (const x of [20, 86, 152, 320, 386, 452]) glass(x, h * 0.16, 44, h * 0.34);
      for (const x of [214, 270]) {
        c.fillStyle = "rgba(0,0,0,0.35)";
        c.fillRect(x - 2, h * 0.1, 46, h * 0.82);
        glass(x + 6, h * 0.16, 32, h * 0.4);
      }
      c.fillStyle = stripe;
      c.fillRect(0, h * 0.62, w, h * 0.09);
      c.fillStyle = "rgba(0,0,0,0.45)";
      c.fillRect(0, h * 0.88, w, h * 0.12);
      if (tagged) graffiti(c, pick(TAGS), rand(150, 360), h * 0.76, 44, rand(-0.08, 0.08));
      speckle(c, w, h, 400, ["rgba(0,0,0,0.06)", "rgba(255,255,255,0.05)"], 1, 3);
    });
  }

  function trainFrontTexture(base, stripe, glowOnly) {
    return tex(256, 256, (c, w, h) => {
      c.fillStyle = glowOnly ? "#000" : base;
      c.fillRect(0, 0, w, h);
      if (!glowOnly) {
        c.fillStyle = "#151a24";
        c.fillRect(w * 0.12, h * 0.06, w * 0.76, h * 0.11);
        c.fillStyle = "#ff9f1c";
        c.font = "bold 22px Arial";
        c.textAlign = "center";
        c.fillText("SUBWAY 7", w / 2, h * 0.145);
        const g = c.createLinearGradient(0, h * 0.2, 0, h * 0.55);
        g.addColorStop(0, "#8ccaf2");
        g.addColorStop(1, "#1b3550");
        c.fillStyle = "#151a24";
        c.fillRect(w * 0.08, h * 0.19, w * 0.84, h * 0.38);
        c.fillStyle = g;
        c.fillRect(w * 0.11, h * 0.21, w * 0.78, h * 0.34);
        c.fillStyle = "rgba(255,255,255,0.35)";
        c.fillRect(w * 0.2, h * 0.21, w * 0.08, h * 0.34);
        c.fillStyle = stripe;
        c.fillRect(0, h * 0.62, w, h * 0.09);
        c.fillStyle = "#2a2a2a";
        c.fillRect(w * 0.06, h * 0.86, w * 0.88, h * 0.14);
      }
      for (const x of [0.22, 0.78]) {
        c.fillStyle = glowOnly ? "#fff6c8" : "#fff6c8";
        c.beginPath();
        c.arc(w * x, h * 0.77, w * 0.065, 0, Math.PI * 2);
        c.fill();
        if (!glowOnly) {
          c.strokeStyle = "#333";
          c.lineWidth = 5;
          c.stroke();
        }
      }
    });
  }

  function stripeTexture() {
    return tex(256, 32, (c, w, h) => {
      c.fillStyle = "#f4f4f4";
      c.fillRect(0, 0, w, h);
      c.fillStyle = "#e63946";
      for (let x = -h; x < w + h; x += 48) {
        c.beginPath();
        c.moveTo(x, h);
        c.lineTo(x + 24, h);
        c.lineTo(x + 24 + h, 0);
        c.lineTo(x + h, 0);
        c.fill();
      }
    });
  }

  function finishTexture() {
    return tex(1024, 96, (c, w, h) => {
      const s = h / 2;
      for (let x = 0; x < w; x += s) {
        for (let y = 0; y < h; y += s) {
          c.fillStyle = ((x + y) / s) % 2 ? "#111" : "#fff";
          c.fillRect(x, y, s, s);
        }
      }
      c.fillStyle = "#e63946";
      c.fillRect(w * 0.3, 0, w * 0.4, h);
      c.fillStyle = "#fff";
      c.font = "900 64px 'Arial Black', Impact, sans-serif";
      c.textAlign = "center";
      c.textBaseline = "middle";
      c.fillText("FINISH", w / 2, h / 2 + 4);
    });
  }

  function skylineTexture(color, lit) {
    const t = tex(1024, 256, (c, w, h) => {
      let x = 0;
      while (x < w) {
        const bw = rand(24, 70), bh = rand(h * 0.25, h * 0.95);
        c.fillStyle = color;
        c.fillRect(x, h - bh, bw, bh);
        if (Math.random() < 0.2) c.fillRect(x + bw * 0.4, h - bh - rand(8, 30), 3, 30);
        c.fillStyle = lit;
        for (let yy = h - bh + 8; yy < h - 6; yy += 10) {
          for (let xx = x + 5; xx < x + bw - 5; xx += 8) if (Math.random() < 0.25) c.fillRect(xx, yy, 3, 4);
        }
        x += bw + rand(0, 6);
      }
    });
    t.wrapS = t.wrapT = THREE.ClampToEdgeWrapping;
    return t;
  }

  function girderTexture() {
    return tex(1024, 64, (c, w, h) => {
      c.fillStyle = "#3f7d64";
      c.fillRect(0, 0, w, h);
      c.fillStyle = "rgba(255,255,255,0.2)";
      c.fillRect(0, 0, w, h * 0.14);
      c.fillStyle = "rgba(0,0,0,0.3)";
      c.fillRect(0, h * 0.84, w, h * 0.16);
      for (let x = 0; x < w; x += 64) {
        c.fillStyle = "rgba(0,0,0,0.25)";
        c.fillRect(x, h * 0.14, 6, h * 0.7);
        c.fillStyle = "rgba(255,255,255,0.12)";
        c.fillRect(x + 6, h * 0.14, 3, h * 0.7);
      }
      c.fillStyle = "rgba(0,0,0,0.4)";
      for (let x = 8; x < w; x += 16) {
        c.fillRect(x, h * 0.22, 3, 3);
        c.fillRect(x, h * 0.72, 3, 3);
      }
      const [a, b] = [...TAGS].sort(() => Math.random() - 0.5);
      graffiti(c, a, w * 0.3, h * 0.5, 30, -0.04);
      graffiti(c, b, w * 0.78, h * 0.5, 26, 0.05);
    }, [3.5, 1]);
  }

  function railingTexture() {
    return tex(64, 64, (c, w, h) => {
      c.fillStyle = "#2f5e4b";
      c.fillRect(0, 0, w, 9);
      c.fillRect(0, h - 7, w, 7);
      c.fillRect(w / 2 - 4, 0, 8, h);
    }, [52, 1]);
  }

  function coinTexture() {
    return tex(128, 128, (c, w, h) => {
      const g = c.createRadialGradient(w * 0.38, h * 0.32, 4, w / 2, h / 2, w / 2);
      g.addColorStop(0, "#fff6b8");
      g.addColorStop(0.55, "#ffcb2f");
      g.addColorStop(1, "#d49100");
      c.fillStyle = g;
      c.fillRect(0, 0, w, h);
      c.lineWidth = 8;
      c.strokeStyle = "#c27c00";
      c.beginPath();
      c.arc(w / 2, h / 2, w * 0.37, 0, Math.PI * 2);
      c.stroke();
      c.beginPath();
      for (let i = 0; i < 10; i++) {
        const r = i % 2 ? w * 0.11 : w * 0.25, a = -Math.PI / 2 + i * Math.PI / 5;
        c.lineTo(w / 2 + r * Math.cos(a), h / 2 + r * Math.sin(a));
      }
      c.closePath();
      c.fillStyle = "#e39f00";
      c.fill();
      c.lineWidth = 3;
      c.strokeStyle = "#fff1a6";
      c.stroke();
    });
  }

  function softTexture(inner, outer) {
    return tex(128, 128, (c, w, h) => {
      const g = c.createRadialGradient(w / 2, h / 2, 0, w / 2, h / 2, w / 2);
      g.addColorStop(0, inner);
      g.addColorStop(1, outer);
      c.fillStyle = g;
      c.fillRect(0, 0, w, h);
    });
  }

  const lambert = (color, extra = {}) => new THREE.MeshLambertMaterial({ color, ...extra });

  function box(w, h, d, material, x = 0, y = 0, z = 0) {
    const m = new THREE.Mesh(new THREE.BoxGeometry(w, h, d), material);
    m.position.set(x, y, z);
    return m;
  }

  function castShadows(obj) {
    if (!Q.shadows) return;
    obj.traverse(o => {
      if (o.isMesh && !o.userData.blob) o.castShadow = true;
    });
  }

  function blob(w, d, opacity = 0.5) {
    const m = new THREE.Mesh(new THREE.PlaneGeometry(w, d),
      new THREE.MeshBasicMaterial({ map: blobTex, transparent: true, depthWrite: false, opacity }));
    m.rotation.x = -Math.PI / 2;
    m.position.y = GROUND + 0.026;
    m.renderOrder = 1;
    m.userData.blob = true;
    return m;
  }

  // Merges many boxes into one geometry, with window UVs scaled to each face.
  function mergedBoxes(boxes, tileW, tileH) {
    const pos = [], nor = [], uv = [], col = [];
    const color = new THREE.Color();
    for (const b of boxes) {
      const g = new THREE.BoxGeometry(b.w, b.h, b.d).toNonIndexed();
      g.translate(b.x, b.y, b.z);
      const P = g.attributes.position.array, N = g.attributes.normal.array, U = g.attributes.uv.array;
      color.set(b.color);
      for (let i = 0; i < P.length / 3; i++) {
        pos.push(P[3 * i], P[3 * i + 1], P[3 * i + 2]);
        nor.push(N[3 * i], N[3 * i + 1], N[3 * i + 2]);
        if (Math.abs(N[3 * i + 1]) > 0.5) {
          uv.push(0.03, 0.97);
        } else {
          const width = Math.abs(N[3 * i]) > 0.5 ? b.d : b.w;
          uv.push(U[2 * i] * width / tileW, U[2 * i + 1] * b.h / tileH);
        }
        col.push(color.r, color.g, color.b);
      }
      g.dispose();
    }
    const geo = new THREE.BufferGeometry();
    geo.setAttribute("position", new THREE.Float32BufferAttribute(pos, 3));
    geo.setAttribute("normal", new THREE.Float32BufferAttribute(nor, 3));
    geo.setAttribute("uv", new THREE.Float32BufferAttribute(uv, 2));
    geo.setAttribute("color", new THREE.Float32BufferAttribute(col, 3));
    return geo;
  }

  function instanced(geometry, material, transforms, colors) {
    const m = new THREE.InstancedMesh(geometry, material, transforms.length);
    const mat = new THREE.Matrix4(), q = new THREE.Quaternion(), s = new THREE.Vector3(), p = new THREE.Vector3();
    const color = new THREE.Color();
    transforms.forEach((t, i) => {
      p.set(t.x, t.y, t.z);
      q.setFromAxisAngle(new THREE.Vector3(0, 1, 0), t.ry || 0);
      s.set(t.sx || 1, t.sy || 1, t.sz || 1);
      m.setMatrixAt(i, mat.compose(p, q, s));
      if (colors) m.setColorAt(i, color.set(colors[i]));
    });
    return m;
  }

  function buildEnvironment() {
    scene.background = skyTexture();
    scene.fog = new THREE.Fog(FOG, FOG_NEAR, FOG_FAR);

    const gravel = gravelTexture();
    const bed = new THREE.Mesh(new THREE.PlaneGeometry(3.6, TRACK_LEN), lambert(0xffffff, { map: gravel }));
    bed.rotation.x = -Math.PI / 2;
    bed.position.set(0, GROUND - 0.06, TRACK_Z);
    bed.receiveShadow = Q.shadows;
    scene.add(bed);
    scrollers.push(d => { gravel.offset.y = (d / (TRACK_LEN / gravel.repeat.y)) % 1; });

    const sleeperT = [], sleeperC = [];
    for (const lane of LANES) {
      const shades = Array.from({ length: SLEEPER_PATTERN }, () => pick(["#6d4c33", "#7a5638", "#5f422c", "#80603f"]));
      for (let i = 0, z = TRACK_Z + TRACK_LEN / 2; z > TRACK_Z - TRACK_LEN / 2; i++, z -= SLEEPER_SPACING) {
        sleeperT.push({ x: lane, y: GROUND - 0.035, z });
        sleeperC.push(shades[i % SLEEPER_PATTERN]);
      }
    }
    const sleepers = instanced(new THREE.BoxGeometry(0.86, 0.04, 0.11), lambert(0xffffff), sleeperT, sleeperC);
    sleepers.receiveShadow = Q.shadows;
    scene.add(sleepers);
    scrollers.push(d => { sleepers.position.z = d % (SLEEPER_SPACING * SLEEPER_PATTERN); });

    const railT = [];
    for (const lane of LANES) for (const side of [-0.27, 0.27]) railT.push({ x: lane + side, y: GROUND + 0.002, z: TRACK_Z });
    scene.add(instanced(new THREE.BoxGeometry(0.035, 0.035, TRACK_LEN),
      new THREE.MeshStandardMaterial({ color: 0xa8b0b8, metalness: 0.6, roughness: 0.35 }), railT));

    const concrete = concreteTexture();
    const curb = lambert(0xa7a094);
    const walkMats = [curb, curb, lambert(0xffffff, { map: concrete }), curb, curb, curb];
    for (const side of [-1, 1]) {
      const walk = box(0.8, 0.14, TRACK_LEN, walkMats, side * 2.2, GROUND + 0.01, TRACK_Z);
      walk.receiveShadow = Q.shadows;
      scene.add(walk);
    }
    scrollers.push(d => { concrete.offset.y = (d / (TRACK_LEN / concrete.repeat.y)) % 1; });

    const wallsTex = [wallTexture(), wallTexture()];
    for (const side of [-1, 1]) {
      const t = wallsTex[side < 0 ? 0 : 1];
      const wall = new THREE.Mesh(new THREE.PlaneGeometry(TRACK_LEN, 1.05), lambert(0xffffff, { map: t }));
      wall.rotation.y = side < 0 ? Math.PI / 2 : -Math.PI / 2;
      wall.position.set(side * 2.6, GROUND + 0.6, TRACK_Z);
      scene.add(wall);
      const cap = box(0.12, 0.06, TRACK_LEN, lambert(0x9c9589), side * 2.62, GROUND + 1.14, TRACK_Z);
      scene.add(cap);
      scrollers.push(d => { t.offset.x = (side < 0 ? 1 : -1) * ((d / (TRACK_LEN / t.repeat.x)) % 1); });
    }

    const decor = new THREE.Group();
    scene.add(decor);
    const clip = { clippingPlanes: [new THREE.Plane(new THREE.Vector3(0, 0, 1), -DECOR_END)] };
    scrollers.push(d => { decor.position.z = d % DECOR_PERIOD; });
    const copies = (z, add) => { for (const copy of [-1, 0, 1]) add(z + copy * DECOR_PERIOD); };

    const facades = ["#f4e1c1", "#e7b9a5", "#c9d6e3", "#d7e3c3", "#f2d2a9", "#b9c7d8", "#e8c6c6", "#d9d2c5", "#cfc1e0"];
    const buildings = [], trees = [], trunks = [], leafColors = [];
    for (const side of [-1, 1]) {
      let z = 0;
      while (z > -DECOR_PERIOD) {
        let d = rand(2.2, 4.6);
        if (z > BRIDGE_Z - STREET && z - d < BRIDGE_Z + STREET) {
          if (z - (BRIDGE_Z + STREET) < 1.2) {
            z = BRIDGE_Z - STREET;
            continue;
          }
          d = z - (BRIDGE_Z + STREET);
        }
        const w = rand(1.8, 3.6), h = rand(1.6, 6.5), inner = rand(3.5, 4.0), color = pick(facades);
        if (Math.random() < Q.buildings) {
          copies(z - d / 2, bz => buildings.push({ w, h, d, x: side * (inner + w / 2), y: GROUND + 0.12 + h / 2, z: bz, color }));
        }
        const tz = z - rand(0, d), s = rand(0.9, 1.25), sy = s * rand(0.85, 1.1), ry = rand(0, 6);
        const leaf = pick(["#5aa845", "#4e9a3c", "#6cbf4f", "#3f8a37"]);
        if (Math.random() < 0.6 && Math.abs(tz - BRIDGE_Z) > 1) {
          copies(tz, cz => {
            trees.push({ x: side * 3.15, y: GROUND + 0.12 + 0.75 * s, z: cz, sx: s, sy, sz: s, ry });
            trunks.push({ x: side * 3.15, y: GROUND + 0.12 + 0.25 * s, z: cz, sx: s, sy: s, sz: s });
            leafColors.push(leaf);
          });
        }
        z -= d + rand(0.25, 0.9);
      }
    }
    const buildingMesh = new THREE.Mesh(mergedBoxes(buildings, 1.8, 4.4),
      lambert(0xffffff, { map: windowsTexture(), vertexColors: true, ...clip }));
    decor.add(buildingMesh);
    decor.add(instanced(new THREE.IcosahedronGeometry(0.42, 1), lambert(0xffffff, { flatShading: true, ...clip }), trees, leafColors));
    decor.add(instanced(new THREE.CylinderGeometry(0.05, 0.07, 0.5, 6), lambert(0x6b4a2f, clip), trunks));

    const girder = lambert(0xffffff, { map: girderTexture(), ...clip });
    const deckMats = [girder, girder, lambert(0x8d8a84, clip), lambert(0x4a4f47, clip), girder, girder];
    const pillar = lambert(0xb8b0a3, clip);
    const railing = lambert(0xffffff, { map: railingTexture(), alphaTest: 0.5, side: THREE.DoubleSide, ...clip });
    const pillarH = BRIDGE_Y - 0.19 - GROUND;
    copies(BRIDGE_Z, z => {
      const deck = box(26, 0.38, 1.4, deckMats, 0, BRIDGE_Y, z);
      deck.castShadow = Q.shadows;
      decor.add(deck);
      for (const side of [-1, 1]) {
        decor.add(box(0.36, pillarH, 0.7, pillar, side * 2.95, GROUND + pillarH / 2, z));
        const rail = new THREE.Mesh(new THREE.PlaneGeometry(26, 0.32), railing);
        rail.position.set(0, BRIDGE_Y + 0.35, z + side * 0.66);
        decor.add(rail);
      }
      if (!Q.shadows) {
        const shade = new THREE.Mesh(new THREE.PlaneGeometry(5.2, 2.4),
          new THREE.MeshBasicMaterial({ map: bandTex, transparent: true, depthWrite: false, opacity: 0.75, ...clip }));
        shade.rotation.x = -Math.PI / 2;
        shade.position.set(0.3, GROUND + 0.09, z - 1.2);
        decor.add(shade);
      }
    });

    const poles = [], beams = [];
    for (let z = DECOR_PERIOD; z > -DECOR_PERIOD * 2; z -= PORTAL_SPACING) {
      for (const side of [-1, 1]) poles.push({ x: side * 1.98, y: GROUND + 0.9, z });
      beams.push({ x: 0, y: GROUND + 1.8, z });
    }
    const steel = lambert(0x5d6670, clip);
    decor.add(instanced(new THREE.BoxGeometry(0.07, 1.8, 0.07), steel, poles));
    decor.add(instanced(new THREE.BoxGeometry(4.1, 0.07, 0.07), steel, beams));

    // Adds an unfogged backdrop plane behind the scene.
    const far = (texture, w, h, z, y, color) => {
      const m = new THREE.Mesh(new THREE.PlaneGeometry(w, h),
        new THREE.MeshBasicMaterial({ map: texture, transparent: true, fog: false, color, depthWrite: false }));
      m.position.set(0, y, z);
      scene.add(m);
    };
    far(skylineTexture("#9db8cf", "#d7e7f3"), 260, 38, -125, GROUND + 18, 0xffffff);
    far(skylineTexture("#7f9bb4", "#c8dbe9"), 200, 26, -100, GROUND + 12, 0xffffff);
    const sunSprite = new THREE.Sprite(new THREE.SpriteMaterial({ map: softTexture("rgba(255,250,225,1)", "rgba(255,240,180,0)"), fog: false }));
    sunSprite.position.set(-30, 26, -118);
    sunSprite.scale.set(30, 30, 1);
    scene.add(sunSprite);
    const cloudTex = softTexture("rgba(255,255,255,0.95)", "rgba(255,255,255,0)");
    for (let i = 0; i < 14; i++) {
      const cloud = new THREE.Sprite(new THREE.SpriteMaterial({ map: cloudTex, fog: false, opacity: rand(0.6, 0.9) }));
      cloud.position.set(rand(-160, 160), rand(16, 30), rand(-115, -95));
      cloud.scale.set(rand(18, 34), rand(7, 11), 1);
      scene.add(cloud);
      const drift = rand(0.3, 0.8);
      scrollers.push((d, dt) => { cloud.position.x = ((cloud.position.x + 160 + drift * dt) % 320) - 160; });
    }
  }

  function buildLights() {
    scene.add(new THREE.HemisphereLight(0xdcefff, 0x9c8a6c, 2.1));
    sun = new THREE.DirectionalLight(0xfff1d8, 2.4);
    sun.position.set(-3, 7, 2);
    sun.target.position.set(0, GROUND, -6);
    scene.add(sun, sun.target);
    if (Q.shadows) {
      sun.castShadow = true;
      sun.shadow.mapSize.set(1024, 1024);
      Object.assign(sun.shadow.camera, { left: -5, right: 5, top: 7, bottom: -7, near: 0.5, far: 20 });
      sun.shadow.bias = -0.0008;
    }
  }

  // Builds a blocky character with hip and shoulder pivots for animation.
  function makeRunner(colors, scale) {
    const root = new THREE.Group();
    const body = new THREE.Group();
    root.add(body);
    const M = c => lambert(c);
    const skin = M(colors.skin), cloth = M(colors.top), legs = M(colors.pants);
    const shoeMat = M(colors.shoes);

    const limb = (radius, length, material) => {
      const m = new THREE.Mesh(new THREE.CapsuleGeometry(radius, length, 4, 8), material);
      m.position.y = -(length / 2 + radius);
      return m;
    };
    const leg = x => {
      const pivot = new THREE.Group();
      pivot.position.set(x, 0.19, 0);
      pivot.add(limb(0.034, 0.12, legs));
      pivot.add(box(0.065, 0.04, 0.11, shoeMat, 0, -0.18, -0.018));
      body.add(pivot);
      return pivot;
    };
    const arm = x => {
      const pivot = new THREE.Group();
      pivot.position.set(x, 0.335, 0);
      pivot.add(limb(0.028, 0.1, cloth));
      const hand = new THREE.Mesh(new THREE.SphereGeometry(0.03, 8, 6), skin);
      hand.position.y = -0.165;
      pivot.add(hand);
      body.add(pivot);
      return pivot;
    };
    const legL = leg(-0.046), legR = leg(0.046);
    body.add(box(0.17, 0.16, 0.1, cloth, 0, 0.268, 0));
    body.add(box(0.172, 0.03, 0.102, legs, 0, 0.2, 0));
    const armL = arm(-0.108), armR = arm(0.108);
    const head = new THREE.Mesh(new THREE.SphereGeometry(0.06, 14, 10), skin);
    head.position.y = 0.405;
    body.add(head);
    if (colors.hair) {
      const hair = new THREE.Mesh(new THREE.SphereGeometry(0.063, 14, 10, 0, Math.PI * 2, Math.PI * 0.35, Math.PI * 0.45), M(colors.hair));
      hair.position.y = 0.405;
      hair.rotation.x = -0.35;
      body.add(hair);
    }
    const capMat = M(colors.cap);
    const dome = new THREE.Mesh(new THREE.SphereGeometry(0.064, 14, 7, 0, Math.PI * 2, 0, Math.PI / 2), capMat);
    dome.position.y = 0.412;
    body.add(dome);
    body.add(box(0.09, 0.012, 0.07, capMat, 0, 0.418, colors.brimBack ? 0.07 : -0.07));
    if (colors.pack) body.add(box(0.1, 0.1, 0.04, M(colors.pack), 0, 0.29, 0.068));

    const jetpack = new THREE.Group();
    const tankMat = lambert(0xff8c42), flameMat = new THREE.MeshBasicMaterial({ color: 0xffd23f });
    const flames = [];
    for (const x of [-0.04, 0.04]) {
      const tank = new THREE.Mesh(new THREE.CylinderGeometry(0.03, 0.03, 0.13, 10), tankMat);
      tank.position.set(x, 0.29, 0.085);
      jetpack.add(tank);
      const flame = new THREE.Mesh(new THREE.ConeGeometry(0.022, 0.09, 8), flameMat);
      flame.rotation.x = Math.PI;
      flame.position.set(x, 0.18, 0.085);
      jetpack.add(flame);
      flames.push(flame);
    }
    jetpack.visible = false;
    body.add(jetpack);

    root.scale.setScalar(scale);
    castShadows(root);
    const shadow = blob(0.34 * scale, 0.26 * scale, 0.55);
    scene.add(root, shadow);
    return { root, body, legL, legR, armL, armR, jetpack, flames, shoeMat, shadow };
  }

  function animateRunner(r, state) {
    const { running, airborne, flying, lean, idleTime } = state;
    const s = Math.sin(phase);
    r.armL.rotation.z = -0.1;
    r.armR.rotation.z = 0.1;
    if (flying) {
      r.legL.rotation.x = -0.25 + s * 0.12;
      r.legR.rotation.x = -0.25 - s * 0.12;
      r.armL.rotation.x = r.armR.rotation.x = -0.45;
      r.armL.rotation.z = -0.25;
      r.armR.rotation.z = 0.25;
      r.body.position.y = Math.sin(idleTime * 3) * 0.01;
      r.body.rotation.x = -0.35;
    } else if (airborne) {
      r.legL.rotation.x = 0.85;
      r.legR.rotation.x = 0.35;
      r.armL.rotation.x = 2.5;
      r.armR.rotation.x = 2.2;
      r.body.position.y = 0;
      r.body.rotation.x = -0.05;
    } else if (running) {
      r.legL.rotation.x = s * 0.95;
      r.legR.rotation.x = -s * 0.95;
      r.armL.rotation.x = -s * 0.85;
      r.armR.rotation.x = s * 0.85;
      r.body.position.y = Math.abs(Math.cos(phase)) * 0.014;
      r.body.rotation.x = -0.14;
    } else {
      const b = Math.sin(idleTime * 2.2);
      r.legL.rotation.x = r.legR.rotation.x = 0;
      r.armL.rotation.x = 0.1 + b * 0.04;
      r.armR.rotation.x = 0.1 - b * 0.04;
      r.body.position.y = b * 0.003;
      r.body.rotation.x = 0;
    }
    r.root.rotation.z = lean;
  }

  const TRAIN_LOOKS = [["#d7263d", "#f4f4f4"], ["#1b76d1", "#ffd23f"], ["#f2a900", "#2b2d42"],
                       ["#2a9d8f", "#f4f4f4"], ["#7b2cbf", "#ffbe0b"]];
  let trainGeo, roofGeo, bogieGeo, trainLooks, roofMat, bogieMat;

  function buildTrainLooks() {
    trainGeo = new THREE.BoxGeometry(0.78, 0.6, 1.5);
    const arc = new THREE.Shape();
    arc.moveTo(-0.39, 0);
    arc.quadraticCurveTo(0, 0.2, 0.39, 0);
    arc.lineTo(-0.39, 0);
    roofGeo = new THREE.ExtrudeGeometry(arc, { depth: 1.5, bevelEnabled: false, curveSegments: 10 });
    roofGeo.translate(0, 0.7, -0.75);
    bogieGeo = new THREE.BoxGeometry(0.62, 0.1, 0.34);
    roofMat = lambert(0xb5bcc4);
    bogieMat = lambert(0x24262b);
    const under = lambert(0x2b2d33);
    trainLooks = TRAIN_LOOKS.flatMap(([base, stripe]) => [false, true].map(tagged => {
      const side = lambert(0xffffff, { map: trainSideTexture(base, stripe, tagged) });
      const front = lambert(0xffffff, { map: trainFrontTexture(base, stripe, false),
        emissiveMap: trainFrontTexture(base, stripe, true), emissive: 0xfff3c4, emissiveIntensity: 1.2 });
      const back = lambert(base);
      return [side, side, roofMat, under, front, back];
    }));
  }

  function makeTrain(kind) {
    const g = new THREE.Group();
    const bodyMesh = new THREE.Mesh(trainGeo, trainLooks[+kind.slice(5)]);
    bodyMesh.position.y = 0.4;
    g.add(bodyMesh);
    g.add(new THREE.Mesh(roofGeo, roofMat));
    for (const z of [-0.45, 0.45]) {
      const bogie = new THREE.Mesh(bogieGeo, bogieMat);
      bogie.position.set(0, 0.05, z);
      g.add(bogie);
    }
    castShadows(g);
    g.add(blob(1.1, 1.9, 0.45));
    g.children[g.children.length - 1].position.y = 0.026;
    return g;
  }

  let stripeTex;
  function makeBarrier() {
    const g = new THREE.Group();
    const postMat = lambert(0x3d3d3d);
    for (const x of [-0.32, 0.32]) {
      g.add(box(0.035, 0.2, 0.035, postMat, x, 0.1, 0));
      g.add(box(0.1, 0.02, 0.1, postMat, x, 0.01, 0));
    }
    const plain = lambert(0xf4f4f4), striped = lambert(0xffffff, { map: stripeTex });
    const board = new THREE.Mesh(new THREE.BoxGeometry(0.72, 0.08, 0.03), [plain, plain, plain, plain, striped, striped]);
    board.position.y = 0.15;
    g.add(board);
    castShadows(g);
    const shadow = blob(0.9, 0.25, 0.35);
    shadow.position.y = 0.026;
    g.add(shadow);
    return g;
  }

  let coinGeo, coinMat;
  function makeCoin() {
    const m = new THREE.Mesh(coinGeo, coinMat);
    return m;
  }

  function makeBoost(kind) {
    const g = new THREE.Group();
    if (kind === "jump") {
      const shoe = lambert(0x38d36b, { emissive: 0x0f5c2a });
      g.add(box(0.1, 0.05, 0.16, shoe, 0, 0, 0));
      g.add(box(0.11, 0.02, 0.17, lambert(0xffffff), 0, -0.035, 0));
      g.add(box(0.08, 0.05, 0.06, shoe, 0, 0.04, 0.04));
    } else {
      const tank = lambert(0xff8c42, { emissive: 0x5a2a00 });
      for (const x of [-0.04, 0.04]) {
        const t = new THREE.Mesh(new THREE.CylinderGeometry(0.035, 0.035, 0.15, 12), tank);
        t.position.x = x;
        g.add(t);
      }
      g.add(box(0.12, 0.08, 0.04, lambert(0x444444), 0, 0, -0.04));
    }
    const ring = new THREE.Mesh(new THREE.RingGeometry(0.1, 0.14, 24),
      new THREE.MeshBasicMaterial({ color: kind === "jump" ? 0x7dffaa : 0xffd27d, transparent: true, opacity: 0.6, side: THREE.DoubleSide }));
    ring.rotation.x = -Math.PI / 2;
    ring.position.y = -0.1;
    g.add(ring);
    g.scale.setScalar(1.6);
    return g;
  }

  function buildFinish() {
    const g = new THREE.Group();
    const pillar = lambert(0xe9e9e9);
    for (const x of [-2.05, 2.05]) g.add(box(0.16, 1.55, 0.16, pillar, x, 0.78, 0));
    g.add(box(4.3, 0.34, 0.12, lambert(0xffffff, { map: finishTexture() }), 0, 1.42, 0));
    g.visible = false;
    scene.add(g);
    return g;
  }

  // Keeps one mesh per logic object, reusing pooled meshes.
  function sync(name, list, kindOf, make, place) {
    const live = lives[name] || (lives[name] = new Map());
    const spares = kind => pools.get(kind) || pools.set(kind, []).get(kind);
    const current = new Set(list);
    for (const [o, m] of live) {
      if (current.has(o)) continue;
      m.visible = false;
      live.delete(o);
      spares(m.userData.kind).push(m);
    }
    for (const o of list) {
      let m = live.get(o);
      if (!m) {
        const kind = kindOf(o);
        m = spares(kind).pop() || make(kind);
        m.userData.kind = kind;
        m.visible = true;
        if (!m.parent) scene.add(m);
        live.set(o, m);
      }
      place(o, m);
    }
  }

  function init(canvas, quality) {
    Q = QUALITY[quality] || QUALITY.standard;
    renderer = new THREE.WebGLRenderer({ canvas, antialias: Q.antialias });
    renderer.setPixelRatio(Math.min(window.devicePixelRatio || 1, Q.pixelRatio));
    renderer.setSize(window.innerWidth, window.innerHeight, false);
    renderer.shadowMap.enabled = Q.shadows;
    renderer.localClippingEnabled = true;
    renderer.shadowMap.type = THREE.PCFSoftShadowMap;

    scene = new THREE.Scene();
    camera = new THREE.PerspectiveCamera(60, window.innerWidth / window.innerHeight, 0.05, 220);
    blobTex = softTexture("rgba(0,0,0,0.6)", "rgba(0,0,0,0)");
    bandTex = tex(4, 64, (c, w, h) => {
      const g = c.createLinearGradient(0, 0, 0, h);
      g.addColorStop(0, "rgba(0,0,0,0)");
      g.addColorStop(0.3, "rgba(0,0,0,0.6)");
      g.addColorStop(0.7, "rgba(0,0,0,0.6)");
      g.addColorStop(1, "rgba(0,0,0,0)");
      c.fillStyle = g;
      c.fillRect(0, 0, w, h);
    });
    stripeTex = stripeTexture();
    coinGeo = new THREE.CylinderGeometry(0.075, 0.075, 0.022, 24);
    coinGeo.rotateX(Math.PI / 2);
    coinGeo.rotateZ(Math.PI / 2);
    const coinFace = coinTexture();
    const face = new THREE.MeshStandardMaterial({ map: coinFace, metalness: 0.3, roughness: 0.35,
      emissive: 0xffffff, emissiveMap: coinFace, emissiveIntensity: 0.4 });
    coinMat = [new THREE.MeshStandardMaterial({ color: 0xe0a210, metalness: 0.5, roughness: 0.3,
      emissive: 0x7a4a00, emissiveIntensity: 0.6 }), face, face];

    buildLights();
    buildEnvironment();
    buildTrainLooks();
    runner = makeRunner({ skin: "#f0c8a0", top: "#1f8ef1", pants: "#2c3e66", shoes: "#f5f5f5",
                          cap: "#e63946", pack: "#ffb703", hair: "#4a2f1d", brimBack: true }, 1.0);
    inspector = makeRunner({ skin: "#e3b38c", top: "#25335c", pants: "#1d2742", shoes: "#151515",
                             cap: "#141b2d", hair: "#2b2b2b", brimBack: false }, 1.12);
    finishArch = buildFinish();

    window.addEventListener("resize", () => {
      renderer.setSize(window.innerWidth, window.innerHeight, false);
      camera.aspect = window.innerWidth / window.innerHeight;
      camera.updateProjectionMatrix();
    });
  }

  function frame(dt) {
    pending += dt;
    if (Q.maxFps && pending < 1 / Q.maxFps - 0.002) return;
    dt = pending;
    pending = 0;
    time += dt;
    const playing = game_start && !game_over && !finish;
    if (!game_start) attract += 1.2 * dt;
    const dist = attract + traveled;
    for (const f of scrollers) f(dist, dt);

    const p = objects[0];
    visX += (p.translate[0] - visX) * (1 - Math.exp(-dt * 16));
    const feet = p.translate[1] - 0.15;
    runner.root.position.set(visX, feet, p.translate[2]);
    if (playing) phase += dt * (4 + (speed / STEP) * 2.6);
    animateRunner(runner, { running: playing, airborne: feet > GROUND + 1e-3, flying: !!p.flyboost,
                            lean: (visX - p.translate[0]) * 0.8, idleTime: time });
    runner.jetpack.visible = !!p.flyboost;
    for (const f of runner.flames) f.scale.y = 0.8 + Math.random() * 0.6;
    runner.shoeMat.emissive.setHex(p.jumpboost ? 0x1fae55 : 0x000000);
    const lift = Math.max(0, feet - GROUND);
    runner.shadow.position.set(visX, GROUND + 0.026, p.translate[2]);
    runner.shadow.scale.setScalar(Math.max(0.4, 1 - lift * 1.4));
    runner.shadow.material.opacity = Math.max(0.15, 0.55 - lift * 0.8);

    const po = objects[1];
    const chaseZ = po.setback ? po.translate[2] - 0.2 : 0.6;
    policeZ += (chaseZ - policeZ) * (1 - Math.exp(-dt * (po.setback ? 3 : 1.2)));
    policeSide += ((visX > 0.5 ? -1 : 1) - policeSide) * (1 - Math.exp(-dt * 4));
    const policeX = visX + 0.42 * policeSide;
    inspector.root.position.set(policeX, GROUND, policeZ);
    inspector.root.visible = inspector.shadow.visible = policeZ < -1.4;
    inspector.shadow.position.set(policeX, GROUND + 0.026, policeZ);
    animateRunner(inspector, { running: true, airborne: false, lean: 0, idleTime: time });

    sync("trains", obstacles, () => "train" + Math.floor(Math.random() * trainLooks.length), makeTrain,
      (o, m) => m.position.set(o.translate[0], GROUND, o.translate[2]));
    sync("barriers", barriers, () => "barrier", makeBarrier, (o, m) => m.position.set(o.translate[0], GROUND, o.translate[2]));
    sync("coins", coins, () => "coin", makeCoin, (o, m) => {
      m.position.set(o.translate[0], o.translate[1] + 0.04 + Math.sin(time * 4 + o.translate[2]) * 0.012, o.translate[2]);
      m.rotation.y = o.rotation;
    });
    sync("boosts", boosts, o => "boost-" + o.type, kind => makeBoost(kind.slice(6)), (o, m) => {
      m.position.set(o.translate[0], o.translate[1] + Math.sin(time * 3) * 0.02, o.translate[2]);
      m.rotation.y = o.rotation;
    });

    finishArch.visible = p.score >= 100;
    finishArch.position.set(0, GROUND, finish_object.translate[2]);

    camX += (visX * 0.72 - camX) * (1 - Math.exp(-dt * 6));
    camera.position.set(camX, GROUND + 1.0 + lift * 0.3, CAM_Z);
    camera.lookAt(camX * 0.9, GROUND - 0.2 + lift * 0.15, -8.5);
    renderer.render(scene, camera);
  }

  function stats() {
    return { calls: renderer.info.render.calls, triangles: renderer.info.render.triangles };
  }

  return { init, frame, stats };
})();
