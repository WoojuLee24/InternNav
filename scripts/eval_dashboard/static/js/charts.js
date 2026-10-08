/* Dependency-free SVG donut chart with a clickable legend.
   data: [{label, value, color}].  opts.onSelect(label|null) fires on slice/legend click. */
function pieChart(data, opts) {
  opts = opts || {};
  const total = data.reduce((s, d) => s + d.value, 0);
  const wrap = el("div.chart-body");
  if (!total) { wrap.appendChild(el("div.pie-empty", {}, opts.emptyText || "no data")); return wrap; }

  const R = 70, r = 42, C = 90, NS = "http://www.w3.org/2000/svg";
  const svg = document.createElementNS(NS, "svg");
  svg.setAttribute("viewBox", "0 0 180 180");
  svg.setAttribute("width", "180"); svg.setAttribute("height", "180");

  let a0 = -Math.PI / 2;
  const slices = [];
  data.forEach((d) => {
    const frac = d.value / total;
    const a1 = a0 + frac * 2 * Math.PI;
    const big = frac > 0.5 ? 1 : 0;
    const p = document.createElementNS(NS, "path");
    if (frac >= 0.999) {  // full circle -> draw a ring
      p.setAttribute("d",
        `M ${C} ${C - R} A ${R} ${R} 0 1 1 ${C - 0.01} ${C - R} Z ` +
        `M ${C} ${C - r} A ${r} ${r} 0 1 0 ${C + 0.01} ${C - r} Z`);
      p.setAttribute("fill-rule", "evenodd");
    } else {
      const x0 = C + R * Math.cos(a0), y0 = C + R * Math.sin(a0);
      const x1 = C + R * Math.cos(a1), y1 = C + R * Math.sin(a1);
      const ix1 = C + r * Math.cos(a1), iy1 = C + r * Math.sin(a1);
      const ix0 = C + r * Math.cos(a0), iy0 = C + r * Math.sin(a0);
      p.setAttribute("d",
        `M ${x0} ${y0} A ${R} ${R} 0 ${big} 1 ${x1} ${y1} ` +
        `L ${ix1} ${iy1} A ${r} ${r} 0 ${big} 0 ${ix0} ${iy0} Z`);
    }
    p.setAttribute("fill", d.color);
    p.setAttribute("class", "slice");
    p.addEventListener("click", () => select(d.label));
    svg.appendChild(p);
    slices.push({ d, p });
    a0 = a1;
  });

  const center = document.createElementNS(NS, "text");
  center.setAttribute("x", C); center.setAttribute("y", C + 5);
  center.setAttribute("text-anchor", "middle");
  center.setAttribute("fill", "#e4e9f2");
  center.setAttribute("font-size", "22");
  center.setAttribute("font-family", "JetBrains Mono, monospace");
  center.setAttribute("font-weight", "700");
  center.textContent = total;
  svg.appendChild(center);

  const legend = el("div.legend");
  const items = data.map((d) => {
    const li = el("div.li", { onclick: () => select(d.label) }, [
      el("span.sw", { style: `background:${d.color}` }),
      el("span.ln", {}, d.label),
      el("span.lv", {}, String(d.value)),
    ]);
    li._label = d.label;
    legend.appendChild(li);
    return li;
  });

  let cur = null;
  function select(label) {
    cur = cur === label ? null : label;
    items.forEach((li) => li.classList.toggle("active", li._label === cur));
    slices.forEach((s) => s.p.style.opacity = !cur || s.d.label === cur ? 1 : 0.25);
    if (opts.onSelect) opts.onSelect(cur);
  }

  wrap.appendChild(svg);
  wrap.appendChild(legend);
  return wrap;
}

/* Build [{label,value,color}] from a {label:count} map, coloring by RESULT_COLORS
   (failure cases) or a rotating palette (fail-analysis labels). */
function pieData(counts, mode) {
  const entries = Object.entries(counts || {}).sort((a, b) => b[1] - a[1]);
  return entries.map(([label, value], i) => ({
    label, value,
    color: mode === "result" ? (RESULT_COLORS[label] || FA_PALETTE[i % FA_PALETTE.length])
                             : colorForLabel(label, i),
  }));
}
