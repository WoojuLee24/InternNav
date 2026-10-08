/* Page 1 — leaderboard (from VLN-Challenge dev/jay): one sortable table per experiment group
   (first dir under the checkpoints root). A row = one eval of (checkpoint, config). */

function lbCols() {
  const cols = [
    { key: "run", label: "Run / checkpoint" },
    { key: "config", label: "Config" },
    { key: "ckpt_step", label: "Step", better: 1 },
    { key: "date", label: "Date", better: -1 },
  ];
  metricOrder().forEach((k) => cols.push({ key: k, label: k, metric: true, better: lowerBetter(k) ? -1 : 1 }));
  cols.push({ key: "Count", label: "N", metric: true });
  return cols;
}

const _lbSort = {};       // group -> {key, dir}
const _lbExpanded = {};   // group -> bool
const _lbFilter = { text: "" };

function _cellVal(row, key) {
  if (["run", "config", "date", "ckpt_step"].includes(key)) return row[key];
  return row.metrics ? row.metrics[key] : undefined;
}

function lbTable(rows, group) {
  const st = _lbSort[group] || (_lbSort[group] = { key: "date", dir: -1 });
  const sorted = rows.slice().sort((a, b) => {
    const ar = a.status === "running", br = b.status === "running";
    if (ar !== br) return ar ? -1 : 1;
    const va = _cellVal(a, st.key), vb = _cellVal(b, st.key);
    if (va == null && vb == null) return 0;
    if (va == null) return 1;
    if (vb == null) return -1;
    if (typeof va === "string") return st.dir * va.localeCompare(vb);
    return st.dir * (va - vb);
  });

  const cols = lbCols();
  const thead = el("tr", {}, cols.map((c) => {
    const arrow = st.key === c.key ? el("span.arrow", {}, st.dir < 0 ? "▼" : "▲") : null;
    return el("th.sortable", {
      onclick: () => {
        if (st.key === c.key) st.dir *= -1;
        else { st.key = c.key; st.dir = c.better ? -1 : 1; }
        renderLeaderboard(window._lbData);
      },
    }, [c.label, arrow]);
  }));

  const LIMIT = 12;
  const expanded = !!_lbExpanded[group];
  const shown = expanded ? sorted : sorted.slice(0, LIMIT);
  const body = shown.map((row) => {
    const m = row.metrics || {};
    const running = row.status === "running";
    const tds = cols.map((c) => {
      if (c.key === "run") {
        return el("td", { style: "display:flex;align-items:center;gap:4px;" }, [
          running ? el("span.status-dot.running", { title: "eval running" }) : null,
          el("span.task-name", {}, row.run),
          row.has_raw ? el("span.raw-dot", { title: "raw per-episode output available" }, "●raw") : null,
        ]);
      }
      if (c.key === "config") return el("td.mono", { style: "color:var(--dim)" }, row.config);
      if (c.key === "ckpt_step") return el("td.num", {}, row.ckpt_step != null ? String(row.ckpt_step) : "—");
      if (c.key === "date") return el("td.mono", {}, row.date_str);
      if (c.key === "Count") {
        const n = m.Count != null ? m.Count : "—";
        return el("td.num", {}, running ? el("span.progress-pill", {}, `${row.n_episodes}…`) : n);
      }
      return el("td.num", { class: metricClass(c.key, m[c.key]) }, fmtMetric(c.key, m[c.key]));
    });
    return el("tr.row-link" + (running ? ".row-running" : ""),
      { onclick: () => location.hash = `#/task/${encodeURIComponent(row.task_id)}` }, tds);
  });

  if (sorted.length > LIMIT) {
    body.push(el("tr.lb-more", { onclick: () => { _lbExpanded[group] = !expanded; renderLeaderboard(window._lbData); } },
      el("td", { colspan: cols.length }, expanded ? "▲  collapse" : `▾  show all (${sorted.length - LIMIT} more)`)));
  }
  return el("div.tbl-wrap", {}, el("table.tbl", {}, [el("thead", {}, thead), el("tbody", {}, body)]));
}

function renderLeaderboard(data) {
  window._lbData = data;
  setCrumbs([{ text: "Leaderboard" }]);
  const q = _lbFilter.text.toLowerCase();
  const match = (r) => !q || `${r.group} ${r.run} ${r.config}`.toLowerCase().includes(q);
  const groups = data.groups.map((g) => ({ name: g.name, rows: g.rows.filter(match) })).filter((g) => g.rows.length);
  const total = data.groups.reduce((s, g) => s + g.rows.length, 0);

  const search = el("input.lb-search", { placeholder: "filter run / config…", value: _lbFilter.text,
    style: "background:#070a10;border:1px solid var(--line);color:var(--text);padding:6px 10px;border-radius:8px;width:320px",
    oninput: (e) => { _lbFilter.text = e.target.value; clearTimeout(window._lbT);
      window._lbT = setTimeout(() => { renderLeaderboard(window._lbData); const s = document.querySelector(".lb-search");
        if (s) { s.focus(); s.setSelectionRange(s.value.length, s.value.length); } }, 200); } });

  const node = el("div", {}, [
    el("h1.page-title", {}, "InternNav Eval Leaderboard"),
    el("p.page-sub", {}, `${total} evals in ${data.groups.length} groups — R2R val_unseen (habitat). ` +
      "SR@r = STOP within r m (r=3 is the official SR); SPL uses geodesic distance; NDTW = nDTW vs the R2R reference path."),
    el("div", { style: "margin:8px 0 4px" }, search),
    ...groups.flatMap((g) => [
      el("div.section-label", {}, [el("span", {}, "◆"), `${g.name}  (${g.rows.length})`]),
      lbTable(g.rows, g.name),
    ]),
    groups.length ? null : el("div.empty-note", {}, "no evals found"),
  ]);
  mount(node);
}
