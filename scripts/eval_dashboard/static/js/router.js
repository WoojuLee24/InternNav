/* (from VLN-Challenge dev/jay) Hash router: #/  ·  #/task/<id>  ·  #/task/<id>/ep/<episode_id> */

function showError(msg) {
  mount(el("div.empty-note", { style: "margin-top:40px" }, "⚠ " + msg));
}

async function route() {
  document.querySelectorAll(".ep-topbar-nav, .ep-float-btn").forEach((n) => n.remove());
  const hash = location.hash || "#/";
  const parts = hash.replace(/^#\/?/, "").split("/").map(decodeURIComponent);
  mount(el("div.loading", {}, "loading…"));
  try {
    if (parts[0] === "task" && parts[1]) {
      if (parts[2] === "ep" && parts[3]) {
        renderEpisode(await API.episode(parts[1], parts[3]));
      } else {
        renderTask(await API.task(parts[1]));
      }
    } else {
      renderLeaderboard(await API.leaderboard());
    }
  } catch (e) {
    showError(e.message || String(e));
  }
}

async function refreshHost() {
  if (STATIC) { document.getElementById("host").textContent = "static export " + (window.STATIC_BUILT || ""); return; }
  try {
    const d = await API.leaderboard();
    document.getElementById("host").textContent = d.host || "";
  } catch (e) {}
}

async function loadConfig() {
  try {
    window.CFG = await API.get("/api/config");
  } catch (e) { window.CFG = {}; }
}

window.addEventListener("hashchange", route);
window.addEventListener("DOMContentLoaded", async () => {
  document.getElementById("refresh").addEventListener("click", async () => {
    await API.leaderboard(true);
    route();
  });
  refreshHost();
  await loadConfig();   // metric order / radii from config.json
  route();
});
