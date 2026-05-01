#!/bin/bash
# Habitat evaluation environment diagnostic script.
# Run on both servers and compare output to identify crash root cause.

echo "=========================================="
echo "SERVER: $(hostname)"
echo "DATE:   $(date)"
echo "=========================================="

echo ""
echo "--- Python / habitat-sim ---"
python -c "import habitat_sim; print('habitat_sim:', habitat_sim.__version__)" 2>/dev/null
pip show habitat-sim 2>/dev/null | grep -E "Version|Location"

echo ""
echo "--- NVIDIA GPU / Driver ---"
nvidia-smi --query-gpu=driver_version,gpu_name,memory.total --format=csv,noheader 2>/dev/null

echo ""
echo "--- EGL Vendor Config (/usr/share/glvnd/egl_vendor.d/) ---"
ls -la /usr/share/glvnd/egl_vendor.d/ 2>/dev/null
echo ""
for f in /usr/share/glvnd/egl_vendor.d/*.json; do
    echo "  [$f]"
    cat "$f" 2>/dev/null
    echo ""
done

echo ""
echo "--- Mesa / GL library versions ---"
dpkg -l 2>/dev/null | grep -E "libgl1|libegl|mesa" | awk '{print $2, $3}'

echo ""
echo "--- libEGL files on system ---"
find /usr /lib /home/irteam/.local -name "libEGL*.so*" 2>/dev/null | sort

echo ""
echo "--- habitat_sim .so EGL/GL dependencies ---"
SO=$(find /home/irteam/.local/lib -name "habitat_sim_bindings*.so" 2>/dev/null | head -1)
echo "SO path: $SO"
ldd "$SO" 2>/dev/null | grep -E "EGL|GL|cuda|nvidia"

echo ""
echo "--- Magnum bug workarounds compiled into habitat_sim ---"
strings "$SO" 2>/dev/null | grep -E "nv-cubemap|nv-compressed|workaround|DISABLE_EXTENSIONS" | sort -u

echo ""
echo "--- Conda activate.d GL/Magnum env vars ---"
if [ -n "$CONDA_PREFIX" ]; then
    grep -rh -E "MAGNUM|GL_|EGL|NVIDIA" "$CONDA_PREFIX/etc/conda/activate.d/" 2>/dev/null | head -20
else
    echo "CONDA_PREFIX not set"
fi

echo ""
echo "--- Current GL/Magnum environment variables ---"
env | grep -E "MAGNUM|GL_|EGL|DISPLAY|__GL|NVIDIA|CUDA_VISIBLE" | sort

echo ""
echo "--- Minimal habitat-sim init test (checks for Magnum bug detection) ---"
cd /home/irteam/git/InternNav || exit 1
MAGNUM_LOG=verbose python - 2>&1 <<'PYEOF' | grep -E "nv-|OpenGL|EGL device|Error|Abort|SUCCESS|FAIL"
import sys
try:
    import habitat_sim
    cfg = habitat_sim.SimulatorConfiguration()
    cfg.scene_id = "NONE"
    cfg.enable_physics = False
    sim = habitat_sim.Simulator(cfg)
    sim.close()
    print("SUCCESS: simulator init OK")
except Exception as e:
    print(f"FAIL: {e}")
PYEOF

echo ""
echo "--- Scene crash reproduction (scene 2azQ1b91cZZ, 5 steps) ---"
cd /home/irteam/git/InternNav || exit 1
MAGNUM_LOG=verbose python - 2>&1 <<'PYEOF' | tail -20
import sys, numpy as np
scene = "data/InternData-N1-v0.5-mini/scene_data/mp3d_ce/2azQ1b91cZZ/2azQ1b91cZZ.glb"
try:
    import habitat_sim
    sim_cfg = habitat_sim.SimulatorConfiguration()
    sim_cfg.scene_id = scene
    sim_cfg.gpu_device_id = 0

    agent_cfg = habitat_sim.agent.AgentConfiguration()
    specs = []
    for name, stype in [("rgb", habitat_sim.SensorType.COLOR), ("depth", habitat_sim.SensorType.DEPTH)]:
        spec = habitat_sim.CameraSensorSpec()
        spec.uuid = name
        spec.sensor_type = stype
        spec.resolution = [224, 224]
        specs.append(spec)
    agent_cfg.sensor_specifications = specs

    sim = habitat_sim.Simulator(habitat_sim.Configuration(sim_cfg, [agent_cfg]))
    print("Init OK")
    for i in range(5):
        obs = sim.get_sensor_observations()
        print(f"Step {i}: rgb={obs['rgb'].shape} depth={obs['depth'].shape}")
    sim.close()
    print("SUCCESS: 5 steps done")
except Exception as e:
    print(f"FAIL: {e}", file=sys.stderr)
    sys.exit(1)
PYEOF

echo ""
echo "=========================================="
echo "DONE: $(hostname)"
echo "=========================================="
