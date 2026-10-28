"""Real-time web dashboard for the sim VLN pipeline (ROS 2 node + Flask, one process).

Open http://<host>:8090 in a browser (host network; http://<host>:8090/?snap=1 polls JPEG snapshots instead of
MJPEG streams — lower bandwidth). Shows
  * VLN overlay image (/vln/pixel_goal_image: pixel goal + projected S1 path), FPV and 30° look-down cameras
  * top-down map: scene background (scripts/sim/maps/<name>.png/.json), robot pose + driven path,
    S1 trajectory (/vln/trajectory), gdm plan (/gdm/local_plan_map), VLN goal (/vln/goal_marker)
  * S2 answers (/vln/s2 from vln_bridge_node), status (/vln/status), latency
  * instruction box -> /vln/instruction (Enter / Send, empty / STOP button = stop), teleport (x, y, yaw or map click) -> /sim/teleport
 * Reset button -> stop + teleport to the start pose (params start_x / start_y / start_yaw_deg, sim spawn = 0, 0, -90)
  * manual drive (WASD / arrow keys or buttons) -> /cmd_vel (overrides the VLN /gdq/msg/cmd_vel in the sim)
Also republishes the driven path as /vln/robot_path (nav_msgs/Path) for rviz (scripts/sim/rviz/vln.rviz).

Usage (gd_vln container): python3 scripts/sim/web_viz.py --ros-args -p use_sim_time:=true -p port:=8090
"""
import collections
import json
import math
import threading
import time
from pathlib import Path

import cv2
import numpy as np
import rclpy
from cv_bridge import CvBridge
from flask import Flask, Response, jsonify, request, send_file
from geometry_msgs.msg import Pose2D, PoseStamped, Twist
from nav_msgs.msg import Odometry, Path as PathMsg
from rclpy.executors import MultiThreadedExecutor
from rclpy.node import Node
from rclpy.qos import qos_profile_sensor_data
from sensor_msgs.msg import Image
from std_msgs.msg import String
from visualization_msgs.msg import Marker

HERE = Path(__file__).resolve().parent


class WebViz(Node):
    def __init__(self):
        super().__init__("vln_web_viz")
        p = self.declare_parameter
        p("port", 8090)
        p("map", str(HERE / "maps" / "etri_indoor"))  # prefix of <name>.png / <name>.json ("" = no background)
        p("fpv_topic", "/camera/camera/color/image_raw")
        p("lookdown_topic", "/camera_lookdown/camera/color/image_raw")
        p("stream_hz", 5.0)
        p("start_x", 0.0)  # Reset pose = sim spawn (run_rbq10_*_sim.sh: origin, GUIDEDOG_INIT_YAW=-90)
        p("start_y", 0.0)
        p("start_yaw_deg", -90.0)
        g = lambda n: self.get_parameter(n).value  # noqa: E731
        self.port = g("port")
        self.min_dt = 1.0 / g("stream_hz")
        self.start_pose = (float(g("start_x")), float(g("start_y")), float(g("start_yaw_deg")))
        self.bridge = CvBridge()
        self.lock = threading.Lock()
        self.jpeg = {}  # name -> (jpeg bytes, wall time)
        self.last_enc = {}
        self.pose = None
        self.history = collections.deque(maxlen=20000)
        self.traj, self.plan, self.goal = [], [], None
        self.status = ""
        self.s2_log = collections.deque(maxlen=40)
        self.map_meta = None
        mp = g("map")
        if mp and Path(mp + ".png").exists():
            self.map_png = mp + ".png"
            self.map_meta = json.loads(Path(mp + ".json").read_text())

        self.create_subscription(Image, "/vln/pixel_goal_image", lambda m: self.on_img("overlay", m, None), 2)
        self.create_subscription(Image, "/vln/s2_image", lambda m: self.on_img("s2", m, None, throttle=False), 2)
        self.create_subscription(Image, g("fpv_topic"), lambda m: self.on_img("fpv", m, (640, 360)),
                                 qos_profile_sensor_data)
        self.create_subscription(Image, g("lookdown_topic"), lambda m: self.on_img("lookdown", m, (640, 360)),
                                 qos_profile_sensor_data)
        self.create_subscription(Odometry, "/gdq/msg/gdq_odom", self.on_odom, 20)
        self.create_subscription(PathMsg, "/vln/trajectory", lambda m: self.on_path("traj", m), 5)
        self.create_subscription(PathMsg, "/gdm/local_plan_map", lambda m: self.on_path("plan", m), 5)
        self.create_subscription(Marker, "/vln/goal_marker", self.on_goal, 5)
        self.create_subscription(String, "/vln/status", lambda m: setattr(self, "status", m.data), 5)
        self.create_subscription(String, "/vln/s2", self.on_s2, 10)
        self.instr_pub = self.create_publisher(String, "/vln/instruction", 5)
        self.tp_pub = self.create_publisher(Pose2D, "/sim/teleport", 1)
        self.cmd_pub = self.create_publisher(Twist, "/cmd_vel", 5)
        self.path_pub = self.create_publisher(PathMsg, "/vln/robot_path", 2)
        self.create_timer(0.5, self.publish_robot_path)

    # ------------------------------------------------------------------ ROS callbacks
    def on_img(self, name, msg, size, throttle=True):
        now = time.time()
        if throttle and now - self.last_enc.get(name, 0) < self.min_dt:
            return
        self.last_enc[name] = now
        img = self.bridge.imgmsg_to_cv2(msg, "bgr8")
        if size is not None:
            img = cv2.resize(img, size, interpolation=cv2.INTER_AREA)
        ok, buf = cv2.imencode(".jpg", img, [cv2.IMWRITE_JPEG_QUALITY, 80])
        if ok:
            with self.lock:
                self.jpeg[name] = (buf.tobytes(), now)

    def on_odom(self, m):
        p, q = m.pose.pose.position, m.pose.pose.orientation
        yaw = math.atan2(2 * (q.w * q.z + q.x * q.y), 1 - 2 * (q.y * q.y + q.z * q.z))
        with self.lock:
            self.pose = (p.x, p.y, yaw)
            if self.history and math.hypot(p.x - self.history[-1][0], p.y - self.history[-1][1]) > 1.0:
                self.history.clear()  # teleport
            if not self.history or math.hypot(p.x - self.history[-1][0], p.y - self.history[-1][1]) > 0.05:
                self.history.append((p.x, p.y))

    def on_path(self, key, m):
        with self.lock:
            setattr(self, key, [(ps.pose.position.x, ps.pose.position.y) for ps in m.poses])

    def on_goal(self, m):
        with self.lock:
            self.goal = (m.pose.position.x, m.pose.position.y, "depth" if m.color.g > 0.9 else "ground")

    def on_s2(self, m):
        try:
            d = json.loads(m.data)
        except ValueError:
            return
        d["wall"] = time.strftime("%H:%M:%S")
        with self.lock:
            self.s2_log.appendleft(d)

    def publish_robot_path(self):
        with self.lock:
            hist = list(self.history)
        msg = PathMsg()
        msg.header.frame_id = "map"
        msg.header.stamp = self.get_clock().now().to_msg()
        for x, y in hist[-3000:]:
            ps = PoseStamped()
            ps.header = msg.header
            ps.pose.position.x, ps.pose.position.y = x, y
            ps.pose.orientation.w = 1.0
            msg.poses.append(ps)
        self.path_pub.publish(msg)

    def state(self):
        with self.lock:
            return {
                "pose": self.pose, "history": list(self.history)[-3000:], "traj": self.traj, "plan": self.plan,
                "goal": self.goal, "status": self.status, "s2": list(self.s2_log),
                "streams": {k: round(time.time() - t, 1) for k, (_, t) in self.jpeg.items()},
            }


def make_app(node):
    app = Flask(__name__)

    @app.route("/")
    def index():
        if "snap" in request.args:  # never start MJPEG loads in snapshot mode (the page would never finish loading)
            return PAGE.replace('src="/stream/', 'data-snap="1" src="/snap/')
        return PAGE

    @app.route("/state")
    def state():
        return jsonify(node.state())

    @app.route("/map.json")
    def map_json():
        return jsonify(node.map_meta or {})

    @app.route("/map.png")
    def map_png():
        if node.map_meta is None:
            return ("", 404)
        return send_file(node.map_png, mimetype="image/png")

    @app.route("/stream/<name>")
    def stream(name):
        def gen():
            last = 0
            while True:
                with node.lock:
                    item = node.jpeg.get(name)
                if item and item[1] != last:
                    last = item[1]
                    yield b"--frame\r\nContent-Type: image/jpeg\r\n\r\n" + item[0] + b"\r\n"
                time.sleep(0.05)

        return Response(gen(), mimetype="multipart/x-mixed-replace; boundary=frame")

    @app.route("/snap/<name>")
    def snap(name):  # single latest jpeg (page ?snap=1 polls these instead of MJPEG; low bandwidth / screenshots)
        with node.lock:
            item = node.jpeg.get(name)
        if item is None:
            return ("", 404)
        return Response(item[0], mimetype="image/jpeg", headers={"Cache-Control": "no-store"})

    @app.route("/instruction", methods=["POST"])
    def instruction():
        text = (request.get_json(force=True) or {}).get("text", "")
        node.instr_pub.publish(String(data=text))
        node.get_logger().info(f"web: instruction {text!r}")
        return jsonify({"ok": True, "text": text})

    def do_teleport(x, y, yaw_deg):
        node.instr_pub.publish(String(data=""))  # stop the bridge first
        node.tp_pub.publish(Pose2D(x=float(x), y=float(y), theta=math.radians(float(yaw_deg))))
        with node.lock:
            node.history.clear()

    @app.route("/teleport", methods=["POST"])
    def teleport():
        d = request.get_json(force=True) or {}
        do_teleport(d.get("x", 0.0), d.get("y", 0.0), d.get("yaw_deg", -90.0))
        return jsonify({"ok": True})

    @app.route("/reset", methods=["POST"])
    def reset():
        do_teleport(*node.start_pose)
        node.get_logger().info(f"web: reset to start pose {node.start_pose}")
        return jsonify({"ok": True, "pose": node.start_pose})

    @app.route("/cmd_vel", methods=["POST"])
    def cmd_vel():
        d = request.get_json(force=True) or {}
        if d.get("stop_vln"):
            node.instr_pub.publish(String(data=""))
        t = Twist()
        t.linear.x, t.angular.z = float(d.get("v", 0.0)), float(d.get("w", 0.0))
        node.cmd_pub.publish(t)
        return jsonify({"ok": True})

    return app


PAGE = r"""<!doctype html><html><head><meta charset="utf-8"><title>VLN sim monitor</title>
<meta name="viewport" content="width=device-width, initial-scale=1">
<style>
:root{--bg:#16181d;--panel:#20232a;--fg:#e6e6e6;--mut:#9aa0a6;--acc:#4ea1ff}
*{box-sizing:border-box}body{margin:0;background:var(--bg);color:var(--fg);font:14px system-ui,sans-serif}
header{padding:8px 14px;background:var(--panel);display:flex;gap:14px;align-items:center;flex-wrap:wrap}
header b{font-size:16px}.pill{padding:2px 8px;border-radius:10px;background:#333;color:var(--mut)}
main{display:grid;grid-template-columns:minmax(300px,420px) 1fr;gap:10px;padding:10px}
@media(max-width:900px){main{grid-template-columns:1fr}}
.card{background:var(--panel);border-radius:8px;padding:8px}h3{margin:2px 0 6px;font-size:13px;color:var(--mut)}
img.cam{width:100%;display:block;border-radius:4px;background:#000;min-height:60px}
.row{display:grid;grid-template-columns:1fr 1fr;gap:8px;margin-top:8px}
.row3{display:grid;grid-template-columns:1fr 1fr 1fr;gap:8px;margin-top:8px}@media(max-width:900px){.row3{grid-template-columns:1fr}}
canvas{width:100%;border-radius:4px;background:#fff}
input[type=text]{flex:1;padding:6px;border-radius:4px;border:1px solid #444;background:#111;color:var(--fg)}
button{padding:6px 12px;border:0;border-radius:4px;background:var(--acc);color:#fff;cursor:pointer}
button.stop{background:#e5484d}button.sec{background:#555}
table{width:100%;border-collapse:collapse;font-size:12px}td,th{padding:3px 4px;border-bottom:1px solid #333;text-align:left;vertical-align:top}
input[type=number]{width:62px;padding:4px;border-radius:4px;border:1px solid #444;background:#111;color:var(--fg)}
.pad{display:grid;grid-template-columns:repeat(3,56px);gap:4px}.pad button{user-select:none;touch-action:none}
.legend span{margin-right:10px;font-size:12px}.sw{display:inline-block;width:10px;height:10px;margin-right:3px;vertical-align:middle}
</style></head><body>
<header><b>VLN sim monitor</b><span class="pill" id="status">status: -</span><span class="pill" id="pose">pose: -</span>
<span class="pill" id="lat">S2: -</span></header>
<main>
<section>
 <div class="card"><h3>Top-down map</h3><canvas id="map"></canvas>
  <div class="legend"><span><i class="sw" style="background:#e5484d"></i>robot</span><span><i class="sw" style="background:#7a7aff"></i>driven</span>
  <span><i class="sw" style="background:#0078ff"></i>S1 trajectory</span><span><i class="sw" style="background:#00b400"></i>gdm plan</span>
  <span><i class="sw" style="background:#ff9900"></i>VLN goal</span></div></div>
 <div class="card" style="margin-top:10px"><h3>Instruction</h3>
  <div style="display:flex;gap:6px"><input type="text" id="instr" placeholder="e.g. Walk forward and stop in front of the printer on your right.">
  <button onclick="send()">Send</button><button class="stop" onclick="stopv()">STOP</button></div>
  <div style="margin-top:6px;display:flex;gap:4px;align-items:center">x<input type="number" id="tx" value="0" step="0.1">
  y<input type="number" id="ty" value="0" step="0.1">yaw°<input type="number" id="tyaw" value="-90" step="15">
  <button class="sec" onclick="tp()">Teleport</button><button class="stop" onclick="resetv()" title="STOP + teleport to the start pose">Reset</button></div>
  <div style="font-size:12px;color:var(--mut);margin-top:4px">Enter = Send · map click = set x, y · Reset = stop + back to start</div></div>
 <div class="card" style="margin-top:10px"><h3>Manual drive (WASD / arrows, hold) — stops VLN</h3>
  <div class="pad"><span></span><button class="sec" data-k="w">▲</button><span></span>
  <button class="sec" data-k="a">◀</button><button class="sec" data-k="s">▼</button><button class="sec" data-k="d">▶</button></div>
  <div style="margin-top:6px;font-size:12px">v <input type="number" id="vmax" value="0.5" step="0.1"> m/s · w <input type="number" id="wmax" value="0.8" step="0.1"> rad/s</div></div>
</section>
<section>
 <div class="card"><h3>Latest S2 decision — pixel goal (green) + S1 path (blue) on the frame S2 used ("LD" = 30° look-down)</h3><img class="cam" style="max-width:560px;margin:auto" src="/stream/s2"></div>
 <div class="row3"><div class="card"><h3>Live overlay (every reply)</h3><img class="cam" src="/stream/overlay"></div>
 <div class="card"><h3>FPV camera (14.8°)</h3><img class="cam" src="/stream/fpv"></div>
 <div class="card"><h3>Look-down camera (30°)</h3><img class="cam" src="/stream/lookdown"></div></div>
 <div class="card" style="margin-top:8px"><h3>S2 answers (latest first)</h3><table id="s2"><tr><th>time</th><th>S2 text</th><th>action / pixel</th><th>frame</th><th>goal (map)</th><th>ms</th></tr></table></div>
</section></main>
<script>
let meta=null, bg=new Image(), bgok=false, last_s=null;
fetch('/map.json').then(r=>r.json()).then(m=>{if(m.res){meta=m;bg.onload=()=>bgok=true;bg.src='/map.png';}});
const cv=document.getElementById('map'), cx=cv.getContext('2d');
function w2p(x,y,s){ if(meta) return [(x-meta.x0)/meta.res*s.k,(meta.height-1-(y-meta.y0)/meta.res)*s.k];
  return [s.W/2+(x-s.cx)*40, s.H/2-(y-s.cy)*40]; }
function line(pts,s,col,w){ if(!pts||pts.length<2) return; cx.strokeStyle=col; cx.lineWidth=w; cx.beginPath();
  pts.forEach((p,i)=>{const q=w2p(p[0],p[1],s); i?cx.lineTo(q[0],q[1]):cx.moveTo(q[0],q[1]);}); cx.stroke(); }
function draw(st){
  const W=cv.clientWidth, H=meta? Math.round(W*meta.height/meta.width): Math.round(W*1.2);
  cv.width=W; cv.height=H; const s={W,H,k: meta? W/meta.width:1, cx: st.pose?st.pose[0]:0, cy: st.pose?st.pose[1]:0}; last_s=s;
  cx.clearRect(0,0,W,H); if(meta&&bgok) cx.drawImage(bg,0,0,W,H);
  line(st.history,s,'#7a7aff',2); line(st.plan,s,'#00b400',3); line(st.traj,s,'#0078ff',3);
  if(st.goal){const g=w2p(st.goal[0],st.goal[1],s); cx.fillStyle='#ff9900'; cx.beginPath(); cx.arc(g[0],g[1],6,0,7); cx.fill();}
  if(st.pose){const [x,y,t]=st.pose, p=w2p(x,y,s), L=14; cx.save(); cx.translate(p[0],p[1]); cx.rotate(-t);
    cx.fillStyle='#e5484d'; cx.beginPath(); cx.moveTo(L,0); cx.lineTo(-L*0.6,L*0.5); cx.lineTo(-L*0.6,-L*0.5); cx.closePath(); cx.fill(); cx.restore();}
}
function esc(t){return (t??'').toString().replace(/[&<>]/g,c=>({'&':'&amp;','<':'&lt;','>':'&gt;'}[c]));}
async function poll(){ try{ const st=await (await fetch('/state')).json(); draw(st);
  document.getElementById('status').textContent='status: '+(st.status||'-');
  if(st.pose) document.getElementById('pose').textContent=`pose: (${st.pose[0].toFixed(2)}, ${st.pose[1].toFixed(2)}, ${(st.pose[2]*57.3).toFixed(0)}°)`;
  if(st.s2.length){ const a=st.s2[0]; document.getElementById('lat').textContent=`S2: ${Math.round((a.t_infer||0)*1000)} ms · ${esc(a.mode)} · "${esc(a.instruction).slice(0,60)}"`; }
  document.getElementById('s2').innerHTML='<tr><th>time</th><th>S2 text</th><th>action / pixel</th><th>frame</th><th>goal (map)</th><th>ms</th></tr>'+
   st.s2.slice(0,15).map(a=>`<tr><td>${a.wall}</td><td>${esc(a.llm)}</td><td>${a.action?JSON.stringify(a.action):(a.pixel?JSON.stringify(a.pixel):'-')}</td><td>${esc(a.pixel_frame)}</td><td>${a.goal_xy?a.goal_xy.map(v=>v.toFixed(2)).join(', '):'-'}</td><td>${Math.round((a.t_infer||0)*1000)}</td></tr>`).join('');
 }catch(e){} setTimeout(poll,250); }
poll();
if(new URLSearchParams(location.search).has('snap')){  // snapshot mode: poll single frames instead of MJPEG
  const imgs=[...document.querySelectorAll('img.cam')].map(im=>[im,im.getAttribute('src').split('/').pop()]);
  const tick=()=>{imgs.forEach(([im,n])=>im.src='/snap/'+n+'?t='+Date.now()); setTimeout(tick,500);}; tick();
}
function post(u,b){return fetch(u,{method:'POST',headers:{'Content-Type':'application/json'},body:JSON.stringify(b)});}
function send(){post('/instruction',{text:document.getElementById('instr').value});}
function stopv(){post('/instruction',{text:''});}
function tp(){post('/teleport',{x:+tx.value,y:+ty.value,yaw_deg:+tyaw.value});}
function resetv(){post('/reset',{});}
document.getElementById('instr').addEventListener('keydown',e=>{ if(e.key==='Enter'){ e.preventDefault(); send(); } });
cv.addEventListener('click',e=>{ if(!last_s) return; const r=cv.getBoundingClientRect(), u=e.clientX-r.left, v=e.clientY-r.top, s=last_s;
  const [x,y]= meta? [meta.x0+u/s.k*meta.res, meta.y0+(meta.height-1-v/s.k)*meta.res] : [s.cx+(u-s.W/2)/40, s.cy-(v-s.H/2)/40];
  tx.value=x.toFixed(2); ty.value=y.toFixed(2); });
const held=new Set(), KEY={w:'w',arrowup:'w',s:'s',arrowdown:'s',a:'a',arrowleft:'a',d:'d',arrowright:'d'};
let drive=null;
function vel(){ const V=+vmax.value, Wm=+wmax.value;
  return {v:(held.has('w')?V:0)-(held.has('s')?V:0), w:(held.has('a')?Wm:0)-(held.has('d')?Wm:0)}; }
function press(k){ if(held.has(k)) return; held.add(k);
  if(!drive){ post('/cmd_vel',{...vel(),stop_vln:true}); drive=setInterval(()=>post('/cmd_vel',vel()),100); } }
function release(k){ held.delete(k); if(!held.size&&drive){ clearInterval(drive); drive=null; post('/cmd_vel',{v:0,w:0}); } }
addEventListener('keydown',e=>{ const k=KEY[e.key.toLowerCase()]; if(!k||e.target.tagName==='INPUT') return; e.preventDefault(); press(k); });
addEventListener('keyup',e=>{ const k=KEY[e.key.toLowerCase()]; if(k) release(k); });
addEventListener('blur',()=>[...held].forEach(release));
document.querySelectorAll('.pad button').forEach(b=>{ const k=b.dataset.k;
  b.addEventListener('pointerdown',e=>{b.setPointerCapture(e.pointerId); press(k);});
  ['pointerup','pointercancel'].forEach(t=>b.addEventListener(t,()=>release(k))); });
</script></body></html>"""


def main():
    import os
    import signal

    # Flask's app.run blocks the main thread and rclpy only shuts down its context on SIGTERM/SIGINT,
    # which left the process (and port 8090) alive after `kill`. Exit hard instead.
    for sig in (signal.SIGTERM, signal.SIGINT):
        signal.signal(sig, lambda *_: os._exit(0))
    rclpy.init(signal_handler_options=rclpy.signals.SignalHandlerOptions.NO)
    node = WebViz()
    ex = MultiThreadedExecutor(num_threads=4)
    ex.add_node(node)
    threading.Thread(target=ex.spin, daemon=True).start()
    app = make_app(node)
    node.get_logger().info(f"web_viz on http://0.0.0.0:{node.port}")
    app.run(host="0.0.0.0", port=node.port, threaded=True)


if __name__ == "__main__":
    main()
