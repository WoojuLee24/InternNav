"""Field-deployment packaging of the InternVLA-N1 dual system.

Three deployable units, wired ONLY by ZMQ messages (no shared memory, no locks):

    s2_module/   System-2  (Qwen2.5-VL GGUF via llama-s2-runner)   ~1 Hz reasoner
    s1_module/   System-1  (lite diffusion + TensorRT)             ~30 Hz controller
    manager.py   master    lifecycle (start/stop/status) + integration tests + demo

The wire contract (ports, message schemas, clients) lives in `deploy.common.interface`.
"""
