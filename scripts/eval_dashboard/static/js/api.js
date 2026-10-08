/* Fetch wrappers. Two modes:
   - server (app.py): plain fetch of /api/...
   - static (build_static.py): window.STATIC_DATA holds pre-rendered responses; missing ones are
     loaded on demand from data/<key>.js (a <script> tag, since a sandboxed page -- e.g. Jupyter's
     /files/ -- cannot fetch() sibling files). Writes are disabled in static mode. */
const STATIC = !!window.STATIC_DATA;

function staticKey(url) { return url.replace(/[^A-Za-z0-9]/g, "_"); }

// data/<key>.js calls STATIC_PUT(url, payload)
const _staticWait = {};
function STATIC_PUT(url, payload) {
  window.STATIC_DATA[url] = payload;
  (_staticWait[url] || []).forEach((res) => res(payload));
  delete _staticWait[url];
}

function staticGet(url) {
  if (url in window.STATIC_DATA) return Promise.resolve(window.STATIC_DATA[url]);
  return new Promise((res, rej) => {
    (_staticWait[url] = _staticWait[url] || []).push(res);
    const s = document.createElement("script");
    s.src = "data/" + staticKey(url) + ".js";
    s.onerror = () => rej(new Error("not in static export: " + url));
    document.head.appendChild(s);
  });
}

const API = {
  async get(url) {
    if (STATIC) return staticGet(url.replace(/\?.*$/, ""));
    const r = await fetch(url);
    if (!r.ok) throw new Error(`${r.status} ${url}`);
    return r.json();
  },
  async post(url, body) {
    if (STATIC) throw new Error("read-only static export — use the Flask dashboard to annotate");
    const r = await fetch(url, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify(body || {}),
    });
    if (!r.ok) throw new Error(`${r.status} ${url}`);
    return r.json();
  },
  leaderboard: (refresh) => API.get("/api/leaderboard" + (refresh ? "?refresh=1" : "")),
  task: (id) => API.get(`/api/task/${encodeURIComponent(id)}`),
  episode: (tid, eid) =>
    API.get(`/api/task/${encodeURIComponent(tid)}/episode/${encodeURIComponent(eid)}`),
  neighbors: (tid, eid) =>
    API.get(`/api/task/${encodeURIComponent(tid)}/neighbors/${encodeURIComponent(eid)}`),
  topdown: (tid, key) => API.get(`/api/topdown/${encodeURIComponent(tid)}/${encodeURIComponent(key)}`),
  toggle: (task_id, episode_id, label) =>
    API.post("/api/annotation/toggle", { task_id, episode_id, label }),
  addButton: (label) => API.post("/api/annotation/buttons", { label }),
  renameButton: (oldLabel, newLabel) => API.post("/api/annotation/rename", { old: oldLabel, new: newLabel }),
  deleteButton: (label) => API.post("/api/annotation/delete", { label }),
  errorSet: (task_id, episode_id, frame) => API.post("/api/error/set", { task_id, episode_id, frame }),
  errorClear: (task_id, episode_id) => API.post("/api/error/clear", { task_id, episode_id }),
};
