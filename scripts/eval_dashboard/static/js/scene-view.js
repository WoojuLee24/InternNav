/* Layered top-down scene viewer (from VLN-Challenge dev/jay scene-view.js), viewer 2D frame (x, -z)
   metres. Layers, bottom -> top: navmesh map · goal rings · gt path · pred path · markers · agent.
   paths = { gt:[[x,y]], pred:[[x,y]], gtGoal, predGoal, start, radii, yaw:[rad per frame] }
   The map is the habitat navmesh top-down grid (eval_recorder.py), decoded from RLE into a canvas
   and drawn as an SVG <image>. setFrame(i) moves the agent marker along the pred path. */

const NS = "http://www.w3.org/2000/svg";
function svgEl(tag, attrs) {
  const n = document.createElementNS(NS, tag);
  for (const k in attrs) if (attrs[k] != null) n.setAttribute(k, attrs[k]);
  return n;
}

const SCENE_COLORS = { gt: "#3ef08a", pred: "#ff4d5e", start: "#ffd23f", agent: "#19e3ff" };
const MAP_RGB = { 0: [6, 8, 13], 1: [38, 46, 62], 2: [95, 108, 135] };  // occupied / navigable / border

function starPoints(cx, cy, ro, ri, n) {
  const pts = [];
  for (let i = 0; i < n * 2; i++) {
    const r = i % 2 ? ri : ro;
    const a = -Math.PI / 2 + (i * Math.PI) / n;
    pts.push(`${(cx + r * Math.cos(a)).toFixed(1)},${(cy + r * Math.sin(a)).toFixed(1)}`);
  }
  return pts.join(" ");
}

// RLE grid -> PNG data URL (cached per map object)
function mapDataUrl(g) {
  if (g._url) return g._url;
  const cv = document.createElement("canvas");
  cv.width = g.cols; cv.height = g.rows;
  const ctx = cv.getContext("2d");
  const img = ctx.createImageData(g.cols, g.rows);
  let o = 0;
  for (let i = 0; i < g.rle.length; i += 2) {
    const rgb = MAP_RGB[g.rle[i]] || MAP_RGB[0];
    for (let k = 0; k < g.rle[i + 1]; k++, o += 4) {
      img.data[o] = rgb[0]; img.data[o + 1] = rgb[1]; img.data[o + 2] = rgb[2]; img.data[o + 3] = 255;
    }
  }
  ctx.putImageData(img, 0, 0);
  g._url = cv.toDataURL("image/png");
  return g._url;
}

function buildSceneView(taskId, mapKey, paths, opts) {
  opts = opts || {};
  paths = paths || {};
  const state = { map: true, gt: true, pred: true, marker: opts.marker || null, frame: null };
  const canvasWrap = el("div.scene-canvas-wrap");
  const layerBar = el("div.layer-bar");
  let geo = null;   // top-down grid (or {missing:true})

  function allBounds() {
    const xs = [], ys = [];
    const eat = (arr) => (arr || []).forEach((p) => { if (p && p.length >= 2) { xs.push(p[0]); ys.push(p[1]); } });
    eat(paths.gt); eat(paths.pred); eat([paths.gtGoal, paths.predGoal, paths.start]);
    const rmax = Math.max(0, ...(paths.radii || []));
    if (paths.gtGoal && rmax) {
      xs.push(paths.gtGoal[0] - rmax, paths.gtGoal[0] + rmax);
      ys.push(paths.gtGoal[1] - rmax, paths.gtGoal[1] + rmax);
    }
    if (!xs.length) return null;
    // frame the episode (not the whole house) with a margin, clipped to the map
    const m = 2.0;
    let b = [Math.min(...xs) - m, Math.min(...ys) - m, Math.max(...xs) + m, Math.max(...ys) + m];
    if (geo && geo.bounds) b = [Math.max(b[0], geo.bounds[0]), Math.max(b[1], geo.bounds[1]),
                                Math.min(b[2], geo.bounds[2]), Math.min(b[3], geo.bounds[3])];
    return b;
  }

  function project(b) {
    const pad = 0.3;
    const [x0, y0, x1, y1] = b;
    const bw = (x1 - x0) || 1, bh = (y1 - y0) || 1;
    const W = opts.size || 560;
    const scale = W / (bw + 2 * pad);
    return { W, H: (bh + 2 * pad) * scale, scale,
             X: (x) => (x - x0 + pad) * scale, Y: (y) => (y1 - y + pad) * scale };
  }

  let agentG = null, proj = null;
  function drawAgent() {
    if (!agentG) return;
    while (agentG.firstChild) agentG.removeChild(agentG.firstChild);
    const i = state.frame;
    if (i == null || !paths.pred || !paths.pred[i]) return;
    const [x, y] = paths.pred[i], cx = proj.X(x), cy = proj.Y(y);
    const yaw = (paths.yaw || [])[i];
    agentG.appendChild(svgEl("circle", { cx, cy, r: 7, fill: SCENE_COLORS.agent, stroke: "#fff", "stroke-width": 1.5 }));
    if (yaw != null) {  // heading arrow; yaw is CCW from +x in the (x, -z) frame, SVG y points down
      const L = 16;
      agentG.appendChild(svgEl("line", { x1: cx, y1: cy, x2: cx + L * Math.cos(yaw), y2: cy - L * Math.sin(yaw),
        stroke: "#fff", "stroke-width": 2.5, "stroke-linecap": "round" }));
    }
  }

  function render() {
    clear(canvasWrap);
    if (!geo) { canvasWrap.appendChild(el("div.pie-empty", {}, "loading map…")); return; }
    const b = allBounds();
    if (!b) { canvasWrap.appendChild(el("div.pie-empty", {}, "no trajectory")); return; }
    const p = project(b); proj = p;
    const svg = svgEl("svg", { viewBox: `0 0 ${p.W} ${p.H}`, preserveAspectRatio: "xMidYMid meet" });
    svg.appendChild(svgEl("rect", { x: 0, y: 0, width: p.W, height: p.H, fill: "#06080d" }));
    const poly = (pts, attrs) => svg.appendChild(svgEl("polygon", Object.assign({ points: pts }, attrs)));

    // --- navmesh map (pixelated so cells stay crisp when zoomed) ---
    if (state.map && geo.bounds) {
      const [gx0, gy0, gx1, gy1] = geo.bounds;
      svg.appendChild(svgEl("image", { href: mapDataUrl(geo), x: p.X(gx0), y: p.Y(gy1),
        width: (gx1 - gx0) * p.scale, height: (gy1 - gy0) * p.scale,
        preserveAspectRatio: "none", style: "image-rendering:pixelated" }));
    }

    // --- goal rings: one per metric radius, smaller = more opaque ---
    const radii = (paths.radii || []).filter((r) => r > 0).slice().sort((a, b) => b - a);
    if ((state.gt || state.pred) && paths.gtGoal && radii.length) {
      const gx = p.X(paths.gtGoal[0]), gy = p.Y(paths.gtGoal[1]);
      radii.forEach((rm, i) => {
        const t = radii.length > 1 ? i / (radii.length - 1) : 1;
        svg.appendChild(svgEl("circle", { cx: gx.toFixed(1), cy: gy.toFixed(1), r: (rm * p.scale).toFixed(1),
          fill: `rgba(255,255,255,${(0.03 + 0.10 * t).toFixed(3)})`,
          stroke: `rgba(255,255,255,${(0.28 + 0.5 * t).toFixed(3)})`, "stroke-width": 1, "stroke-dasharray": "5 4" }));
      });
    }

    const drawPath = (pp, color, dots) => {
      if (!pp || pp.length < 1) return;
      if (pp.length > 1) svg.appendChild(svgEl("polyline", {
        points: pp.map(([x, y]) => `${p.X(x).toFixed(1)},${p.Y(y).toFixed(1)}`).join(" "),
        fill: "none", stroke: color, "stroke-width": 2.3, "stroke-linejoin": "round", "stroke-linecap": "round" }));
      if (dots) pp.forEach(([x, y]) => svg.appendChild(svgEl("rect", { x: (p.X(x) - 2.2).toFixed(1),
        y: (p.Y(y) - 2.2).toFixed(1), width: 4.4, height: 4.4, fill: color, stroke: "#0a0d13", "stroke-width": 0.5 })));
    };
    if (state.gt) drawPath(paths.gt, SCENE_COLORS.gt, true);      // R2R graph nodes as squares
    if (state.pred) drawPath(paths.pred, SCENE_COLORS.pred, false);

    if ((state.gt || state.pred) && paths.start) {
      svg.appendChild(svgEl("circle", { cx: p.X(paths.start[0]).toFixed(1), cy: p.Y(paths.start[1]).toFixed(1),
        r: 5, fill: SCENE_COLORS.start, stroke: "#0a0d13", "stroke-width": 1 }));
    }
    const star = (pt, color) => { if (!pt) return; poly(starPoints(p.X(pt[0]), p.Y(pt[1]), 7, 3, 5),
      { fill: color, stroke: "#ffffff", "stroke-width": 0.9, "stroke-linejoin": "round" }); };
    if (state.gt) star(paths.gtGoal, SCENE_COLORS.gt);
    if (state.pred) star(paths.predGoal, SCENE_COLORS.pred);

    if (state.marker) {  // split / error point
      poly(starPoints(p.X(state.marker[0]), p.Y(state.marker[1]), 11, 4.6, 8),
        { fill: "#19e3ff", stroke: "#ffffff", "stroke-width": 1.8, "stroke-linejoin": "round" });
    }
    agentG = svgEl("g", {}); svg.appendChild(agentG); drawAgent();
    canvasWrap.appendChild(svg);
  }

  function buildLayers() {
    clear(layerBar);
    [{ key: "map", label: "Map", sw: "#5f6c87" }, { key: "gt", label: "GT", sw: SCENE_COLORS.gt },
     { key: "pred", label: "Pred", sw: SCENE_COLORS.pred }].forEach((r) =>
      layerBar.appendChild(el("button.layer-pill" + (state[r.key] ? ".on" : ""), {
        onclick: () => { state[r.key] = !state[r.key]; buildLayers(); render(); },
      }, [el("span.sw", { style: `background:${r.sw}` }), r.label])));
  }

  function openModal() {
    const back = el("div.modal-back", { onclick: (e) => { if (e.target === back) close(); } });
    function esc(ev) { if (ev.key === "Escape") close(); }
    const close = () => { back.remove(); document.removeEventListener("keydown", esc); };
    const big = buildSceneView(taskId, mapKey, paths, Object.assign({}, opts,
      { size: 900, _geo: geo, marker: state.marker, expanded: true, onCollapse: close }));
    big.setFrame(state.frame);
    const card = el("div.modal-card", {}, [big.node]);
    if (opts.instruction) card.appendChild(el("div.modal-instruction", {}, ["❝ ", opts.instruction]));
    back.appendChild(card);
    document.body.appendChild(back);
    document.addEventListener("keydown", esc);
  }

  const splitBtn = el("button.split-btn", { onclick: () => opts.onJumpSplit && opts.onJumpSplit() }, "⋔ split point");
  const errorBadge = el("button.error-badge", { onclick: () => opts.onJumpError && opts.onJumpError() }, "⊗ error point");
  function updateStatus(hasSplit, errorActive) {
    splitBtn.className = "split-btn" + (hasSplit && !errorActive ? "" : " disabled");
    splitBtn.title = errorActive ? "overridden by the error point"
      : hasSplit ? "jump to the auto split point" : "no split / not a failed episode";
    errorBadge.style.display = errorActive ? "" : "none";
  }
  const expandBtn = el("button.icon-btn", {
    title: opts.expanded ? "collapse" : "expand",
    onclick: opts.expanded ? () => opts.onCollapse && opts.onCollapse() : openModal,
  }, opts.expanded ? "⤡" : "⤢");

  const node = el("div.scene-view", {}, [
    el("div.scene-top", {}, [el("span.ttl", {}, `▦ ${mapKey || "no map"}`), splitBtn, errorBadge, layerBar, expandBtn]),
    canvasWrap,
  ]);
  buildLayers();
  updateStatus(!!opts.hasSplit, !!opts.errorActive);

  const api = {
    node,
    setMarker(pt) { state.marker = pt; render(); },
    setStatus(hasSplit, errorActive) { updateStatus(hasSplit, errorActive); },
    setFrame(i) { state.frame = i; drawAgent(); },
  };
  const ready = (g) => { geo = g; render(); };
  if (opts._geo) ready(opts._geo);
  else if (mapKey) API.topdown(taskId, mapKey).then(ready).catch(() => ready({ missing: true }));
  else ready({ missing: true });
  return api;
}
