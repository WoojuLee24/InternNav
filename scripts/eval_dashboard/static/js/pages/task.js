/* Page 2 — task (one eval of a checkpoint x config), from VLN-Challenge dev/jay task.js:
   metric cards, run info + every result_<machine>.json row, two donuts, episode table. */

const _epSort = { key: "episode_id", dir: 1 };
const _epFilter = { result: "", scene: "", analysis: "", collision: "" };

function metricCards(m) {
  const cards = metricOrder().map((k) =>
    el("div.metric-card" + (k === "SR@3" ? ".primary" : ""), {}, [
      el("div.k", {}, k), el("div.v", { class: metricClass(k, m[k]) }, fmtMetric(k, m[k])),
    ]));
  cards.push(el("div.metric-card", {}, [el("div.k", {}, "N"), el("div.v", {}, m.Count != null ? m.Count : "—")]));
  return el("div.metric-row", {}, cards);
}

// run identity + the official rows written by the evaluator (one per finished eval run)
function runInfoBlock(t) {
  const kv = [["dir", t.dir], ["ckpt", t.ckpt || "—"], ["ckpt step", t.ckpt_step ?? "—"],
              ["git commit", t.git_commit || "—"], ["raw stamps", (t.raw_stamps || []).join(", ") || "none (legacy eval)"]];
  const items = kv.map(([k, v]) => el("div.cfg-item", {}, [el("span.ck", {}, k), el("span.cv", { title: String(v) }, String(v))]));
  const rows = t.result_rows || [];
  const keys = ["timestamp", "_machine", "sucs_all", "spls_all", "oss_all", "nes_all", "tls_all", "ndtw_refs_all", "length"];
  const fmtCell = (k, v) => (v == null ? "—" : /_all$/.test(k) && k !== "nes_all" && k !== "tls_all" ? fmt(v, 4) : typeof v === "number" ? fmt(v, 3).replace(/\.000$/, "") : String(v));
  const table = rows.length ? el("div.tbl-wrap", { style: "margin-top:10px" }, el("table.tbl.result-rows", {}, [
    el("thead", {}, el("tr", {}, keys.map((k) => el("th", {}, k.replace(/^_/, ""))))),
    el("tbody", {}, rows.map((r) => el("tr", {}, keys.map((k) => el("td.mono", {}, fmtCell(k, r[k])))))),
  ])) : el("div.empty-note", {}, "no finished eval run yet (result_<machine>.json empty)");
  return el("div", {}, [el("div.cfg-grid", {}, items), table]);
}

function chartsRow(data) {
  const fc = el("div.panel.panel-pad.chart-card", {}, [
    el("div.chart-head", {}, ["Failure cases ", el("span.hint", {}, "· passed_goal = was within 3 m, ended outside")]),
    pieChart(pieData(data.failure_cases, "result"), {
      emptyText: "no failures 🎉",
      onSelect: (label) => { _epFilter.result = label || ""; rerenderEpList(data); },
    }),
  ]);
  const fa = el("div.panel.panel-pad.chart-card", {}, [
    el("div.chart-head", {}, ["Fail analysis ", el("span.hint", {}, "· manual tags")]),
    pieChart(pieData(data.fail_analysis, "fa"), {
      emptyText: "no tags yet",
      onSelect: (label) => { _epFilter.analysis = label || ""; rerenderEpList(data); },
    }),
  ]);
  return el("div.grid-2", {}, [fc, fa]);
}

const EP_SORT_COLS = [
  { key: "episode_id", label: "Episode" }, { key: "result", label: "Result" },
  { key: "NE", label: "NE" }, { key: "TL", label: "TL" }, { key: "steps", label: "Steps" },
  { key: "collisions", label: "Collisions" },
];

function epHead(data) {
  const sel = (val, opts, onchange, ph) =>
    el("select", { onchange: (e) => onchange(e.target.value) }, [
      el("option", { value: "" }, ph),
      ...opts.map((o) => el("option", { value: o, selected: o === val ? "" : null }, o)),
    ]);
  const sortSel = el("select", { onchange: (e) => { _epSort.key = e.target.value; rerenderEpList(data); } },
    EP_SORT_COLS.map((c) => el("option", { value: c.key, selected: c.key === _epSort.key ? "" : null }, "sort: " + c.label)));
  const dirBtn = el("button.ghost-btn", { onclick: () => { _epSort.dir *= -1; rerenderEpList(data); } }, _epSort.dir < 0 ? "▼" : "▲");
  return el("div.filters", {}, [
    sel(_epFilter.result, [...new Set(data.episodes.map((e) => e.result))], (v) => { _epFilter.result = v; rerenderEpList(data); }, "all results"),
    sel(_epFilter.scene, [...new Set(data.episodes.map((e) => e.scene))].sort(), (v) => { _epFilter.scene = v; rerenderEpList(data); }, "all scenes"),
    sel(_epFilter.analysis, Object.keys(data.fail_analysis), (v) => { _epFilter.analysis = v; rerenderEpList(data); }, "all tags"),
    sel(_epFilter.collision, ["collided", "no collision"], (v) => { _epFilter.collision = v; rerenderEpList(data); }, "collisions: any"),
    sortSel, dirBtn, el("span.fcount#epcount", {}, ""),
  ]);
}

function epTableHeader() {
  const cells = ["", "Episode", "Result", "Tags", srHeader(), "NDTW", "NE", "TL a/geo", "Steps", "Coll", "Error"];
  return el("div.ep-thead", {}, cells.map((c, i) =>
    el("div" + (i >= 5 && i <= 9 ? ".num" : (i === 4 || i >= 10) ? ".ctr" : ""), {}, c)));
}

const _chk = (on, cls) => on ? el("span.chk." + cls, {}, "✓") : el("span.chk.off", {}, "–");

function epCard(e, taskId) {
  const goto = () => location.hash = `#/task/${encodeURIComponent(taskId)}/ep/${encodeURIComponent(e.episode_id)}`;
  const card = el("div.ep-card");
  const body = el("div.ep-body");
  let view = null;
  async function toggleInline() {
    const opening = !card.classList.contains("open");
    card.classList.toggle("open");
    if (!opening) { if (view) view.destroy(); return; }
    if (view) { _player = view.player; return; }
    clear(body); body.appendChild(el("div.loading", {}, "loading episode…"));
    try {
      view = buildEpisodeView(await API.episode(taskId, e.episode_id));
      clear(body); body.appendChild(view.node);
      _player = view.player;
    } catch (err) { clear(body); body.appendChild(el("div.empty-note", {}, err.message)); }
  }
  const toggleBtn = el("button.row-toggle", { title: e.has_raw ? "expand inline (▶)" : "no raw output",
    onclick: (ev) => { ev.stopPropagation(); if (e.has_raw) toggleInline(); } }, e.has_raw ? "▶" : "·");
  const tags = el("div.ep-c.tag-row", {}, (e.fail_analysis && e.fail_analysis.length)
    ? e.fail_analysis.map((l) => el("span.tag.fa", {}, l)) : [el("span.tag", {}, "—")]);
  const head = el("div.ep-head", {}, [
    toggleBtn,
    el("div.ep-scene", {}, [e.scene, el("span.eid", {}, e.episode_id)]),
    el("div.ep-c", {}, badge(e.result)),
    tags,
    el("div.ep-c.ctr", {}, srTriple(e.metrics, false)),
    el("div.ep-c.num", {}, fmtMetric("NDTW", (e.metrics || {}).NDTW)),
    el("div.ep-c.num", { class: metricClass("NE", e.NE) }, fmt(e.NE, 2)),
    el("div.ep-c.num", {}, agentGt(e.TL, e.gt_tl, 1)),
    el("div.ep-c.num", {}, e.steps != null ? String(e.steps) : "—"),
    el("div.ep-c.num", { class: (e.collisions || 0) > 0 ? "m-bad" : "" }, e.collisions != null ? String(e.collisions) : "—"),
    el("div.ep-c.ctr", {}, _chk(e.has_error_point, "err")),
  ]);
  if (e.has_raw) head.addEventListener("click", goto);
  card.appendChild(head); card.appendChild(body);
  return card;
}

function filteredEpisodes(data) {
  let eps = data.episodes.slice();
  if (_epFilter.result) eps = eps.filter((e) => e.result === _epFilter.result);
  if (_epFilter.scene) eps = eps.filter((e) => e.scene === _epFilter.scene);
  if (_epFilter.analysis) eps = eps.filter((e) => (e.fail_analysis || []).includes(_epFilter.analysis));
  if (_epFilter.collision) eps = eps.filter((e) => (_epFilter.collision === "collided") === ((e.collisions || 0) > 0));
  const k = _epSort.key;
  eps.sort((a, b) => {
    const va = a[k], vb = b[k];
    if (va == null && vb == null) return 0;
    if (va == null) return 1;
    if (vb == null) return -1;
    if (typeof va === "string") return _epSort.dir * va.localeCompare(vb);
    return _epSort.dir * (va - vb);
  });
  return eps;
}

// render in chunks: 1839 episode rows at once would freeze the page
function rerenderEpList(data) {
  const host = document.getElementById("ep-list");
  if (!host) return;
  const eps = filteredEpisodes(data);
  clear(host);
  const CH = 200;
  let shown = 0;
  const more = el("button.ghost-btn", { style: "margin:10px 0", onclick: () => addChunk() }, "");
  function addChunk() {
    eps.slice(shown, shown + CH).forEach((e) => host.insertBefore(epCard(e, data.task.task_id), more));
    shown = Math.min(eps.length, shown + CH);
    more.textContent = `show more (${eps.length - shown} left)`;
    more.style.display = shown < eps.length ? "" : "none";
  }
  host.appendChild(more);
  addChunk();
  const cnt = document.getElementById("epcount");
  if (cnt) cnt.textContent = `${eps.length} / ${data.episodes.length} episodes`;
}

function renderTask(data) {
  window._taskData = data;
  const t = data.task;
  _epFilter.result = _epFilter.scene = _epFilter.analysis = _epFilter.collision = "";
  setCrumbs([{ text: "Leaderboard", href: "#/" }, { text: `${t.group} / ${t.run}` }]);
  mount(el("div", {}, [
    navBar([{ text: "← Leaderboard", href: "#/", primary: true }]),
    el("div", { style: "display:flex;align-items:center;gap:12px;margin-bottom:4px;flex-wrap:wrap" }, [
      badge("group", t.group), el("h1.page-title", { style: "margin:0" }, t.run),
      el("span.mono", { style: "color:var(--dim)" }, t.config),
    ]),
    el("p.page-sub", {}, `${t.date_str} · ${t.status} · ${t.n_episodes} episodes` +
      (t.has_raw ? "" : " · legacy eval: metrics only (no raw/ trajectories)")),
    el("div.section-label", {}, [el("span", {}, "▣"), "Results"]),
    metricCards(t.metrics || {}),
    el("div.section-label", {}, [el("span", {}, "⚙"), "Run"]),
    el("div.panel.panel-pad", {}, runInfoBlock(t)),
    el("div.section-label", {}, [el("span", {}, "◔"), "Failure breakdown"]),
    chartsRow(data),
    el("div.section-label", {}, [el("span", {}, "≣"), "Episodes"]),
    epHead(data),
    epTableHeader(),
    el("div.ep-list#ep-list"),
  ]));
  rerenderEpList(data);
}
