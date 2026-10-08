/* DOM + formatting helpers shared by all pages. */

// el("div.cls.cls2#id", {attr:val, onclick:fn}, [children|text])  — #id may sit anywhere
function el(sel, attrs, children) {
  let id = null;
  const idm = sel.match(/#([\w-]+)/);
  if (idm) { id = idm[1]; sel = sel.replace(/#[\w-]+/, ""); }
  const [tag, ...cls] = sel.split(".");
  const n = document.createElement(tag || "div");
  if (id) n.id = id;
  if (cls.length) n.className = cls.join(" ");
  if (attrs) for (const [k, v] of Object.entries(attrs)) {
    if (v == null) continue;
    if (k === "class") n.className += " " + v;
    else if (k.startsWith("on") && typeof v === "function") n.addEventListener(k.slice(2), v);
    else if (k === "html") n.innerHTML = v;
    else n.setAttribute(k, v);
  }
  for (const c of [].concat(children == null ? [] : children)) {
    if (c == null || c === false) continue;
    n.appendChild(c.nodeType ? c : document.createTextNode(String(c)));
  }
  return n;
}

const clear = (n) => { while (n.firstChild) n.removeChild(n.firstChild); return n; };
const mount = (node) => { const a = document.getElementById("app"); clear(a); a.appendChild(node); };

// number / metric formatting -------------------------------------------------
const isNum = (v) => typeof v === "number" && !isNaN(v);
const fmt = (v, d = 3) => (isNum(v) ? v.toFixed(d) : "—");
// SR/SPL/OS are fractions (0..1); NE/TL are metres.
const pct = (v) => (isNum(v) ? (v * 100).toFixed(1) + "%" : "—");

// "agent / gt" (gt from reference_path); shows just the agent value when no gt exists.
function agentGt(a, gt, dec = 2) {
  const av = isNum(a) ? (dec === 0 ? String(a) : a.toFixed(dec)) : "—";
  if (!isNum(gt)) return av;
  const gv = dec === 0 ? String(gt) : gt.toFixed(dec);
  return `${av} / ${gv}`;
}

// ---- result-metric set (SR@r/SPL@r/OSR@r, NDTW, NE, TL) --------------------------------
const lowerBetter = (key) => key === "NE" || key === "TL" || key === "CR";
// SR/SPL/OSR/NDTW are 0..1 fractions; NE/TL are metres
function fmtMetric(key, v) {
  if (!isNum(v)) return "—";
  if (key === "NDTW") return v.toFixed(3);
  if (key === "NE" || key === "TL") return v.toFixed(2);
  if (key === "CR") return (v * 100).toFixed(2) + "%";  // collisions per step
  return (v * 100).toFixed(1) + "%";        // SR / SPL / OSR
}
function metricClass(key, v) {
  if (!isNum(v)) return "";
  if (key === "CR") return v <= 0.01 ? "m-good" : v <= 0.05 ? "m-mid" : "m-bad";
  if (lowerBetter(key)) return v <= 3 ? "m-good" : v <= 5 ? "m-mid" : "m-bad";
  if (/^(SR|SPL|OSR)@/.test(key) || key === "NDTW" || key === "CFSR")
    return v >= 0.5 ? "m-good" : v >= 0.3 ? "m-mid" : "m-bad";
  return "";
}
const metricOrder = () => (window.CFG && window.CFG.metric_order) ||
  ["SR@3", "SPL@3", "OSR@3", "SR@1.5", "SPL@1.5", "OSR@1.5", "SR@0.5", "SPL@0.5", "OSR@0.5", "NDTW", "NE", "TL"];

// the metric radii, e.g. ["3","1.5","0.5"], and their "@3 / @1.5 / @0.5" header label
const srRadii = () => metricOrder().filter((k) => k.startsWith("SR@")).map((k) => k.slice(3));
const srHeader = () => srRadii().map((r) => "@" + r).join(" / ");
// per-episode success/fail at each radius (SR@r is 1 or 0).
// words=true -> "success / fail / fail" (episode page); words=false -> ✓ / ✗ icons (task table)
function srTriple(m, words = true) {
  const radii = srRadii();
  if (!m || radii.some((r) => !isNum(m["SR@" + r]))) return el("span.mono", {}, "—");
  const parts = [];
  radii.forEach((r, i) => {
    if (i) parts.push(el("span", { style: "color:var(--faint)" }, words ? " / " : " "));
    const ok = m["SR@" + r] >= 1;
    parts.push(el("span", { class: ok ? "m-good" : "m-bad" }, words ? (ok ? "success" : "fail") : (ok ? "✓" : "✗")));
  });
  return el("span.mono", {}, parts);
}

// failure cases from data.failure_case()
const RESULT_COLORS = {
  success: "#46d17f", passed_goal: "#ffb24d", never_reached: "#ff6b7d", no_stop: "#b58bff", unknown: "#5d6678",
};
const FA_PALETTE = ["#b58bff", "#4dd6c1", "#5aa9ff", "#ffb24d", "#ff6b7d", "#46d17f", "#e878c0", "#9aa6ff"];
const colorForLabel = (label, i) => FA_PALETTE[i % FA_PALETTE.length];

function badge(kind, text) { return el("span.badge." + kind, {}, text || kind); }
function resultBadge(r) { return badge(r || "unknown"); }

// in-page navigation buttons (always visible, unlike the topbar breadcrumbs on narrow screens)
function navBar(items) {
  return el("div.nav-bar", {}, items.map((it) =>
    el("a.nav-btn" + (it.primary ? ".primary" : ""), { href: it.href }, it.text)));
}

// breadcrumbs in the topbar ---------------------------------------------------
function setCrumbs(parts) {
  const c = clear(document.getElementById("crumbs"));
  parts.forEach((p, i) => {
    if (i) c.appendChild(el("span.sep", {}, "›"));
    if (p.href) c.appendChild(el("a", { href: p.href }, p.text));
    else c.appendChild(el("span.cur", {}, p.text));
  });
}
