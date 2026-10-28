"""Shared ZMQ wire protocol for the async dual-system split.

Four unidirectional ports decouple S1 (~30 Hz) and S2 (~1-2 Hz) into independent
processes, each with its OWN input and output port (no shared mutable state, no locks):

    orchestrator                         service                       socket pattern
    ------------------------------------------------------------------------------------
    s2_in   PUSH  ->  tcp://*:5601  PULL  S2 service   (frame -> infer)   reliable, queued
    s2_out  SUB   <-  tcp://*:5602  PUB   S2 service   (pixel+latent)     async, observable
    s1_in   PUSH  ->  tcp://*:5603  PULL  S1 service   (latent+rgb)       reliable, queued
    s1_out  PULL  <-  tcp://*:5604  PUSH  S1 service   (discrete actions) reliable, fast path

(The S2 service additionally speaks the runner line protocol to the C++ llama-s2-runner
over a private ZMQ REP socket on :5610 -- see internvla_n1_policy_llamacpp.py.)

Services BIND (they are the stable servers); the orchestrator CONNECTs. PUB/SUB is used
only for the latency-tolerant S2 telemetry so a passive monitor can also subscribe; the
fast S1 action path is PUSH/PULL so no message is ever dropped.

Message framing (multipart, zero-copy for arrays):
    frame[0]      = msgpack header  {"meta": {...}, "arrays": {name:{dtype,shape}}, "order":[...]}
    frame[1..N]   = raw C-contiguous bytes of each named numpy array, in `order`
`meta` holds only JSON-able scalars (seq, instruction, action list, pixel, timings, flags).
"""
import msgpack
import numpy as np
import zmq

DEFAULT_PORTS = {
    "s2_in": 5601,
    "s2_out": 5602,
    "s1_in": 5603,
    "s1_out": 5604,
    "s2_runner": 5610,
    # --- live camera / prompt / visualization (camera_publisher.py, prompt_cli.py, visualizer.py) ---
    "cam": 5605,      # camera_publisher  PUB  -> orchestrator + visualizer SUB   (live RGB frames)
    "prompt": 5606,   # prompt_cli        PUSH -> orchestrator PULL                (live instruction updates)
    "viz": 5607,      # orchestrator      PUB  -> visualizer SUB                   (pixel goal + waypoints + action overlay)
}


def ep(port, host="127.0.0.1"):
    return f"tcp://{host}:{port}"


def bind_ep(port):
    return f"tcp://*:{port}"


# ---- (de)serialization ----------------------------------------------------
def send_msg(sock, meta, arrays=None, flags=0):
    """meta: json-able dict. arrays: {name: np.ndarray} or None."""
    arrays = arrays or {}
    order = list(arrays.keys())
    specs = {k: {"dtype": str(arrays[k].dtype), "shape": list(arrays[k].shape)} for k in order}
    header = msgpack.packb({"meta": meta, "arrays": specs, "order": order}, use_bin_type=True)
    frames = [header] + [np.ascontiguousarray(arrays[k]).tobytes() for k in order]
    sock.send_multipart(frames, flags=flags)


def recv_msg(sock, flags=0):
    frames = sock.recv_multipart(flags=flags)
    h = msgpack.unpackb(frames[0], raw=False)
    arrays = {}
    for i, name in enumerate(h["order"]):
        spec = h["arrays"][name]
        buf = frames[1 + i]
        arrays[name] = np.frombuffer(buf, dtype=spec["dtype"]).reshape(spec["shape"]).copy()
    return h["meta"], arrays


# ---- socket helpers -------------------------------------------------------
def make_socket(ctx, kind, port, *, bind, host="127.0.0.1", hwm=64, subscribe=b""):
    sock = ctx.socket(kind)
    sock.setsockopt(zmq.LINGER, 0)
    if kind in (zmq.PUSH, zmq.PULL, zmq.PUB, zmq.SUB):
        sock.set_hwm(hwm)
    if kind == zmq.SUB:
        sock.setsockopt(zmq.SUBSCRIBE, subscribe)
    (sock.bind if bind else sock.connect)(bind_ep(port) if bind else ep(port, host))
    return sock
