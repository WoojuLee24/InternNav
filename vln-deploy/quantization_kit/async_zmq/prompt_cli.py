"""Send a navigation instruction (prompt) to the running orchestrator over ZMQ, live.

One-shot:     convenv/bin/python -m quantization_kit.async_zmq.prompt_cli "go to the kitchen and stop"
Interactive:  convenv/bin/python -m quantization_kit.async_zmq.prompt_cli      # then type lines, Enter to send

The orchestrator BINDs a PULL on the prompt port; this PUSHes (reliable/queued, no slow-joiner loss).
"""
import argparse
import sys
import time
from pathlib import Path

import zmq

_QK = Path(__file__).resolve().parents[1]
_ROOT = _QK.parent
sys.path[:0] = [str(_ROOT), str(_QK)]

from quantization_kit.async_zmq.zmq_proto import DEFAULT_PORTS, ep  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("text", nargs="*", help="instruction words; omit to read lines from stdin interactively")
    ap.add_argument("--host", default="127.0.0.1")
    ap.add_argument("--port", type=int, default=DEFAULT_PORTS["prompt"])
    args = ap.parse_args()

    ctx = zmq.Context.instance()
    push = ctx.socket(zmq.PUSH)
    push.setsockopt(zmq.LINGER, 1500)
    push.connect(ep(args.port, args.host))
    time.sleep(0.2)   # let the connection settle so the first message is delivered

    def send(s):
        push.send_string(s)
        print(f"[prompt] -> {s!r}", flush=True)

    if args.text:
        send(" ".join(args.text))
    else:
        print("[prompt] type an instruction + Enter to send it live (Ctrl-D to quit):", flush=True)
        for line in sys.stdin:
            line = line.strip()
            if line:
                send(line)
    push.close()


if __name__ == "__main__":
    main()
