/* Page 3 — episode (from VLN-Challenge dev/jay episode.js). buildEpisodeView() returns the
   reusable content node (also embedded inline in the task page). Raw-JSON episodes have no video:
   the player is a frame timer that drives the agent marker on the map + the per-frame strip. */

let _player = null;   // active DualPlayer (spacebar target)

function epMetricTable(d) {
  const m = d.metrics || {}, rm = d.result_metrics || {};
  const fa = (d.selected && d.selected.length)
    ? el("div.tag-row", {}, d.selected.map((l) => el("span.tag.fa", {}, l))) : "—";
  const rows = [
    ["Result", resultBadge(d.result)],
    ["Tags", fa],
    [srHeader(), srTriple(rm, false)],
    ["SPL (habitat)", fmt(m.spl, 3)],
    ["NDTW (ref)", fmtMetric("NDTW", rm.NDTW)],
    ["NE", fmtMetric("NE", rm.NE)],
    ["TL (agent / geodesic)", agentGt(m.TL, m.gt_tl, 2)],
    ["Steps · STOP called", `${m.steps ?? "—"} · ${m.stopped ? "yes" : "no"}`],
    ["Collisions", m.collisions != null ? String(m.collisions) : "—"],
  ];
  return el("div.tbl-wrap", {}, el("table.tbl", {}, el("tbody", {}, rows.map(([k, v]) =>
    el("tr", {}, [el("td.mono", { style: "color:var(--dim)" }, k),
                  el("td", {}, v.nodeType ? v : el("span.mono", {}, v))])))));
}

// ---- shared fail-analysis buttons (unchanged from jay) ------------------------
function interactionBlock(d) {
  const wrap = el("div");
  const summary = el("div.selected-summary");
  const buttonsRow = el("div.fa-buttons");
  const state = { buttons: d.buttons.slice(), selected: d.selected.slice() };

  function paintSummary() {
    clear(summary);
    summary.appendChild(el("div.lbl", {}, "selected reasons"));
    summary.appendChild(state.selected.length
      ? el("div.chips", {}, state.selected.map((l) => el("span.chip", {}, l)))
      : el("div.none", {}, STATIC ? "read-only static export" : "none yet — click a reason above"));
  }
  async function toggleLabel(label) {
    try {
      const r = await API.toggle(d.task_id, d.episode_id, label);
      state.selected = r.selected; state.buttons = r.buttons;
      paintButtons(); paintSummary();
    } catch (e) { alert(e.message); }
  }
  function openMenu(label, anchor) {
    document.querySelectorAll(".fa-menu").forEach((m) => m.remove());
    const menu = el("div.fa-menu", {}, [
      el("div.fa-mi", { onclick: async (ev) => {
        ev.stopPropagation(); menu.remove();
        const nn = (prompt(`Rename "${label}" to:`, label) || "").trim();
        if (!nn || nn === label) return;
        const r = await API.renameButton(label, nn);
        state.buttons = r.buttons;
        state.selected = state.selected.map((l) => (l === label ? nn : l));
        paintButtons(); paintSummary();
      } }, "✎ edit"),
      el("div.fa-mi.danger", { onclick: async (ev) => {
        ev.stopPropagation(); menu.remove();
        if (!confirm(`Delete reason "${label}" everywhere?\nThis removes it from ALL episodes in ALL tasks.`)) return;
        const r = await API.deleteButton(label);
        state.buttons = r.buttons;
        state.selected = state.selected.filter((l) => l !== label);
        paintButtons(); paintSummary();
      } }, "🗑 delete"),
    ]);
    anchor.appendChild(menu);
  }
  function paintButtons() {
    clear(buttonsRow);
    state.buttons.forEach((label) => {
      const on = state.selected.includes(label);
      const kebab = el("span.fa-kebab", { title: "edit / delete",
        onclick: (ev) => { ev.stopPropagation(); openMenu(label, btn); } }, "⋯");
      const btn = el("button.fa-btn" + (on ? ".on" : ""), { onclick: () => toggleLabel(label) },
        [el("span.dot"), label, kebab]);
      buttonsRow.appendChild(btn);
    });
    buttonsRow.appendChild(el("button.fa-btn.fa-add", {
      onclick: async () => {
        const label = (prompt("New failure reason:") || "").trim();
        if (label) await toggleLabel(label);
      },
    }, [el("span", {}, "+"), "add reason"]));
  }
  paintButtons(); paintSummary();
  wrap.appendChild(buttonsRow); wrap.appendChild(summary);
  return wrap;
}

// ---- per-frame strip: action, distance, and the S2 text in effect at this frame --------
function frameStripBlock(d) {
  const strip = el("div.frame-strip");
  const s2Pre = el("pre");
  const s2Lbl = el("span.n");
  const s2Box = el("div.vlm-box.s2-box", {}, [el("div.lbl", {}, ["S2 output ", s2Lbl]), s2Pre]);
  // last frame (<= i) that carries an S2 generation
  const lastGen = [];
  let cur = null;
  (d.frame_info || []).forEach((fi, i) => { if (fi.gen != null) cur = i; lastGen.push(cur); });

  function paint(frame) {
    const fi = (d.frame_info || [])[frame];
    clear(strip);
    if (!fi) { strip.appendChild(el("span", {}, "no per-frame info")); return; }
    const item = (k, v) => el("span.fi", {}, [el("span", {}, k), el("b", {}, v)]);
    strip.appendChild(item("frame", `${frame} / ${d.num_frames - 1}`));
    strip.appendChild(item("action", fi.label));
    if (fi.dist_to_goal != null) strip.appendChild(item("dist→goal", fmt(fi.dist_to_goal, 2) + " m"));
    if (fi.gen != null) strip.appendChild(el("span.fi.pg", {}, "S2 query"));
    if (fi.collision) strip.appendChild(el("span.fi", {}, el("b", { class: "m-bad" }, "COLLISION")));
    // decision metrics (eval_decision_metrics): exact 3 m STOP oracle, reference SPF action / pixel
    const ACT = { 0: "STOP", 1: "FORWARD", 2: "LEFT", 3: "RIGHT" };
    (fi.decision || []).forEach((dc) => {
      const stopOk = (dc.type === "stop") === dc.oracle_stop;
      strip.appendChild(el("span.fi", { title: "STOP judged against the 3 m oracle (exact)" },
        [el("span", {}, `${dc.type}${dc.oracle_stop ? " · in 3 m" : ""}`), el("b", { class: stopOk ? "m-good" : "m-bad" }, stopOk ? "stop ok" : "stop wrong")]));
      if (dc.spf_action != null) strip.appendChild(item("SPF ref", ACT[dc.spf_action]));
      if (dc.pixel_err != null) strip.appendChild(item("pixel err ref", fmt(dc.pixel_err, 0) + " px"));
    });
    const g = lastGen[frame];
    s2Lbl.textContent = g == null ? "(none yet)" : g === frame ? `@ frame ${g}` : `@ frame ${g} (${frame - g} frames ago)`;
    s2Pre.textContent = g == null ? "—" : d.frame_info[g].gen || "(empty)";
  }
  paint(0);
  return { node: el("div", {}, [strip, s2Box]), paint };
}

// ---- front-view image of the current frame (saved for failed episodes by default, raw_frames) ----
function frontViewBlock(d) {
  const imgs = d.frame_images || [];
  if (!imgs.some(Boolean)) return null;
  const img = el("img.front-view", { alt: "front view" });
  const cap = el("div.front-cap");
  function paint(f) {
    const src = imgs[f];
    if (src) { img.src = src; img.style.opacity = 1; cap.textContent = `front view · frame ${f}`; }
    else { img.style.opacity = 0.35; cap.textContent = `front view · frame ${f} (no image: camera tilted)`; }
  }
  paint(0);
  return { node: el("div.front-wrap", {}, [img, cap]), paint };
}

// ---- error point control (under the slider) — shared, overrides the auto split point ----
function errorPointBar(d, getPlayer, refresh) {
  const bar = el("div.error-bar");
  function paint() {
    clear(bar);
    if (d.error_point) {
      bar.appendChild(el("span.err-on", {}, `⊗ error point @ frame ${d.error_point.frame}`));
      bar.appendChild(el("button.err-btn.clear", {
        onclick: async () => {
          try { await API.errorClear(d.task_id, d.episode_id); } catch (e) { alert(e.message); return; }
          d.error_point = null; paint(); refresh();
        },
      }, "✕ clear"));
    } else {
      bar.appendChild(el("span.err-hint", {}, d.split ? "auto split point shown" : "no split point"));
    }
    bar.appendChild(el("button.err-btn.set", {
      title: "save the current frame as the error point (shared with everyone)",
      onclick: async () => {
        const p = getPlayer(); const f = p ? p.frame : 0;
        try { d.error_point = await API.errorSet(d.task_id, d.episode_id, f); } catch (e) { alert(e.message); return; }
        paint(); refresh();
        if (p) { p.pause(); p.seekFrame(f); }
      },
    }, "⊗ set error point @ current frame"));
  }
  paint();
  return bar;
}

// ---- main reusable view ----------------------------------------------------
function buildEpisodeView(d) {
  if (!d.has_raw) {
    return { node: el("div", {}, [epMetricTable(d), el("div.empty-note", {}, "legacy eval: no raw trajectory for this episode")]),
             player: null, destroy() {} };
  }
  const pred = d.pred_path || [];
  const predPt = (f) => pred.length ? pred[Math.max(0, Math.min(pred.length - 1, f))] : null;
  function activeMarker() {
    if (d.error_point) return { point: predPt(d.error_point.frame), frames: [d.error_point.frame] };
    if (d.split) return { point: d.split.point, frames: [d.split.before, d.split.after] };
    return null;
  }
  let player = null;
  const sceneView = buildSceneView(d.task_id, d.topdown, {
    gt: d.reference_path || [], pred, gtGoal: d.gt_goal, predGoal: d.pred_goal, start: d.start,
    radii: d.goal_radii, yaw: d.yaw,
  }, {
    instruction: d.instruction, hasSplit: !!d.split, errorActive: !!d.error_point,
    marker: (activeMarker() || {}).point || null,
    onJumpSplit: () => { if (d.split && player) { player.pause(); player.seekFrame(d.split.after); } },
    onJumpError: () => { if (d.error_point && player) { player.pause(); player.seekFrame(d.error_point.frame); } },
  });
  const strip = frameStripBlock(d);
  const front = frontViewBlock(d);
  player = new DualPlayer([], d.fps, d.num_frames, d.bookmarks,
    (f) => { strip.paint(f); sceneView.setFrame(f); if (front) front.paint(f); });

  function refresh() {
    const act = activeMarker();
    sceneView.setMarker(act ? act.point : null);
    sceneView.setStatus(!!d.split, !!d.error_point);
    player.setDivergeBookmarks(act ? act.frames : []);
  }
  const playback = el("div", {}, [player.build(), errorPointBar(d, () => player, refresh), strip.node]);
  setTimeout(() => { refresh(); player.seekFrame(0); }, 0);

  const meta = d.meta || {};
  const node = el("div", {}, [
    el("div.section-label", {}, [el("span", {}, "▣"), "Result"]),
    epMetricTable(d),
    el("div.section-label", {}, [el("span", {}, "❝"), "Instruction & trajectory"]),
    el("div.ins-scene", {}, [
      el("div.ins-col", {}, [
        el("div.instruction-box", {}, d.instruction || "—"),
        front ? front.node : null,
        el("div.mono", { style: "color:var(--faint);font-size:11px;margin-top:8px;line-height:1.6" },
          `ckpt ${meta.ckpt || "—"}\nrun ${meta.run_stamp || "—"} · rank ${meta.rank ?? "—"} · commit ${(meta.git_commit || "—").slice(0, 10)}`),
        playback,
      ]),
      sceneView.node,
    ]),
    el("div.section-label", {}, [el("span", {}, "✎"), "Fail analysis ",
      el("span", { style: "color:var(--faint);font-size:11px;text-transform:none;letter-spacing:0" }, "· shared · click to toggle")]),
    interactionBlock(d),
  ]);
  return { node, player, destroy: () => player.destroy() };
}

function renderEpisode(d) {
  if (_player) { _player.destroy(); _player = null; }
  setCrumbs([
    { text: "Leaderboard", href: "#/" },
    { text: d.task_id.split("~").slice(0, 3).join("/"), href: `#/task/${encodeURIComponent(d.task_id)}` },
    { text: d.episode_id },
  ]);
  const view = buildEpisodeView(d);
  _player = view.player;
  const run = d.task_id.split("~").slice(0, 3).join("/");
  mount(el("div", {}, [
    navBar([
      { text: "← Leaderboard", href: "#/" },
      { text: `← ${run} 목록`, href: `#/task/${encodeURIComponent(d.task_id)}`, primary: true },
    ]),
    el("div", { style: "display:flex;align-items:center;gap:12px;margin-bottom:4px;flex-wrap:wrap" }, [
      badge("group", d.group), el("h1.page-title", { style: "margin:0" }, d.episode_id),
    ]),
    el("p.page-sub", {}, d.scene),
    view.node,
  ]));

  API.neighbors(d.task_id, d.episode_id).then(({ prev, next }) => {
    const goEp = (eid) => { location.hash = `#/task/${encodeURIComponent(d.task_id)}/ep/${encodeURIComponent(eid)}`; };
    const topbarNav = el("div.ep-topbar-nav");
    // short labels: long episode ids here used to squeeze the breadcrumbs out of the top bar
    if (prev) topbarNav.appendChild(el("button.ep-nav-btn", { title: "이전 episode: " + prev, onclick: () => goEp(prev) }, "‹ 이전"));
    if (next) topbarNav.appendChild(el("button.ep-nav-btn", { title: "다음 episode: " + next, onclick: () => goEp(next) }, "다음 ›"));
    const topbarRight = document.querySelector(".topbar-right");
    if (topbarRight) topbarRight.prepend(topbarNav);
    const side = (cls, eid, txt) => el("button.ep-float-btn." + cls, {
      title: eid || "none", onclick: eid ? () => goEp(eid) : null, style: eid ? "" : "opacity:.2;pointer-events:none",
    }, txt);
    document.body.appendChild(side("ep-float-l", prev, "‹"));
    document.body.appendChild(side("ep-float-r", next, "›"));
  }).catch(() => {});
}

document.addEventListener("click", (e) => {
  if (!e.target.closest(".fa-menu") && !e.target.closest(".fa-kebab"))
    document.querySelectorAll(".fa-menu").forEach((m) => m.remove());
});

// keyboard transport: space = play/pause, ←/→ = step one frame
document.addEventListener("keydown", (e) => {
  if (!_player) return;
  const tag = document.activeElement.tagName;
  if (tag === "INPUT" || tag === "TEXTAREA") return;
  if (e.code === "Space") { e.preventDefault(); _player.toggle(); }
  else if (e.code === "ArrowLeft") { e.preventDefault(); _player.step(-1); }
  else if (e.code === "ArrowRight") { e.preventDefault(); _player.step(1); }
});
