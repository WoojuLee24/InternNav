#!/usr/bin/env python3
"""Export the eval dashboard as static files (no server), e.g. to open through Jupyter's
/files/ link on the k8s pod, where no extra port can be exposed.

    python scripts/eval_dashboard/build_static.py --out <dir> [--root <checkpoints_root>] [--filter baseline_ablation]
    python scripts/eval_dashboard/build_static.py --single --out page.html ...   # one self-contained file

Writes <dir>/index.html + css/js + data/*.js. The leaderboard and config are inlined; every task /
episode / top-down response is a data/<key>.js file calling STATIC_PUT(url, payload), loaded on
demand by api.js via <script> (a sandboxed page cannot fetch() its sibling files). Read-only:
annotating needs the Flask app. Rebuild to pick up new evals.

--single writes ONE page with the CSS, JS and every response inlined (no <html>/<head>/<body>: the
form a claude.ai Artifact publish expects, whose sandbox only loads scripts from allowed CDNs). Keep it
to a filtered set of tasks: everything is loaded up front.
"""
import argparse
import base64
import json
import os
import re
import shutil
import sys
from datetime import datetime
from urllib.parse import quote

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import app as A  # noqa: E402
import data  # noqa: E402

STATIC = os.path.join(os.path.dirname(os.path.abspath(__file__)), "static")


def enc(s):
    return quote(s, safe="-_.!~*'()")  # == JS encodeURIComponent


def key(url):
    return re.sub(r"[^A-Za-z0-9]", "_", url)  # == api.js staticKey()


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--out", required=True)
    ap.add_argument("--root", help=f"checkpoints root (default: {data.root()})")
    ap.add_argument("--filter", default="", help="only tasks whose path contains this substring")
    ap.add_argument("--single", action="store_true", help="one self-contained page (Artifact form), --out is a file")
    args = ap.parse_args()
    if args.root:
        data.SETTINGS["checkpoints_root"] = os.path.abspath(args.root)

    tasks = {k: t for k, t in data.discover(force=True).items() if args.filter in t["dir"]}
    out = os.path.abspath(args.out)
    if args.single:
        return write_single(tasks, out)
    shutil.copytree(os.path.join(STATIC, "css"), os.path.join(out, "css"), dirs_exist_ok=True)
    shutil.copytree(os.path.join(STATIC, "js"), os.path.join(out, "js"), dirs_exist_ok=True)
    os.makedirs(os.path.join(out, "data"), exist_ok=True)

    n_files = 0

    def put(url, payload):
        nonlocal n_files
        with open(os.path.join(out, "data", key(url) + ".js"), "w") as f:
            f.write(f"STATIC_PUT({json.dumps(url)},{json.dumps(payload, ensure_ascii=False)});\n")
        n_files += 1

    for tid, t in tasks.items():
        put(f"/api/task/{enc(tid)}", A.task_payload(t))
        maps = set()
        for e in t["_episodes"]:
            if not e["has_raw"]:  # legacy episode: no episode page to open
                continue
            eid = e["episode_id"]
            put(f"/api/task/{enc(tid)}/neighbors/{enc(eid)}", A.neighbors_payload(t, eid))

            def copy_img(i, path, tid=tid, eid=eid):
                rel = f"frames/{key(tid)}/{eid}/{i:04d}.jpg"
                os.makedirs(os.path.dirname(os.path.join(out, rel)), exist_ok=True)
                shutil.copyfile(path, os.path.join(out, rel))
                return rel

            det = A.episode_payload(t, eid, image_url=copy_img)
            put(f"/api/task/{enc(tid)}/episode/{enc(eid)}", det)
            maps.add(det.get("topdown"))
        for m in filter(None, maps):
            put(f"/api/topdown/{enc(tid)}/{enc(m)}", data.topdown(t, m))
        print(f"  {tid}: {len(t['_episodes'])} episodes, {len(maps)} maps", flush=True)

    built = datetime.now().strftime("%Y-%m-%d %H:%M")
    inline = {"/api/config": A.config_payload(), "/api/leaderboard": A.leaderboard_payload(tasks)}
    html = open(os.path.join(STATIC, "index.html")).read().replace(
        "<!-- STATIC_DATA_SLOT (build_static.py injects the pre-rendered API here) -->",
        f"<script>window.STATIC_DATA={json.dumps(inline, ensure_ascii=False)};"
        f"window.STATIC_BUILT={json.dumps(built)};</script>")
    with open(os.path.join(out, "index.html"), "w") as f:
        f.write(html)
    print(f"[static] {len(tasks)} tasks, {n_files} data files -> {out}/index.html")


def data_uri(_i, path):
    with open(path, "rb") as f:
        return "data:image/jpeg;base64," + base64.b64encode(f.read()).decode()


def write_single(tasks, out_file):
    resp = {"/api/config": A.config_payload(), "/api/leaderboard": A.leaderboard_payload(tasks)}
    for tid, t in tasks.items():
        resp[f"/api/task/{enc(tid)}"] = A.task_payload(t)
        for e in t["_episodes"]:
            if not e["has_raw"]:
                continue
            eid = e["episode_id"]
            resp[f"/api/task/{enc(tid)}/neighbors/{enc(eid)}"] = A.neighbors_payload(t, eid)
            det = resp[f"/api/task/{enc(tid)}/episode/{enc(eid)}"] = A.episode_payload(t, eid, image_url=data_uri)
            if det.get("topdown"):
                url = f"/api/topdown/{enc(tid)}/{enc(det['topdown'])}"
                resp[url] = resp.get(url) or data.topdown(t, det["topdown"])
    html = open(os.path.join(STATIC, "index.html")).read()
    body = html[html.index("<body>") + len("<body>"):html.index("</body>")]
    # scripts inlined in their original order; the data block goes first (api.js reads it at load)
    order = re.findall(r'<script src="(js/[^"]+)"></script>', body)
    body = re.sub(r'\s*<script src="js/[^"]+"></script>', "", body)
    body = body.replace("<!-- STATIC_DATA_SLOT (build_static.py injects the pre-rendered API here) -->", "")
    css = open(os.path.join(STATIC, "css", "style.css")).read()
    fonts = re.findall(r'<link href="(https://fonts.googleapis.com/[^"]+)" rel="stylesheet" />', html)
    blob = json.dumps(resp, ensure_ascii=False).replace("</", "<\\/")  # never close the script tag early
    built = json.dumps(datetime.now().strftime("%Y-%m-%d %H:%M"))
    parts = (["<title>InternNav Eval Dashboard</title>"] + [f'<link rel="stylesheet" href="{u}">' for u in fonts]
             + [f"<style>{css}</style>", body, f"<script>window.STATIC_DATA={blob};window.STATIC_BUILT={built};</script>"]
             + [f"<script>{open(os.path.join(STATIC, src)).read()}</script>" for src in order])
    with open(out_file, "w") as f:
        f.write("\n".join(parts))
    print(f"[static --single] {len(tasks)} tasks, {len(resp)} responses, "
          f"{os.path.getsize(out_file) // 1024} KB -> {out_file}")


if __name__ == "__main__":
    main()
