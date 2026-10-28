# Shared launcher environment for the deploy module run scripts (sourced, not executed).
# Arch-aware, mirrors quantization_kit/run.sh: aarch64 (Jetson sm_110) -> build/,
# x86_64 (dev sm_89) -> build_x86/. Leaves you cd'd at the repo root (imports need it).
DEPLOY_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
ROOT="$(dirname "$DEPLOY_DIR")"
QK="$ROOT/quantization_kit"
PY="${VLN_PY:-$QK/convenv_$([ "$(uname -m)" = aarch64 ] && echo aarch64 || echo x86)/bin/python}"  # [fix] Thor/5090 자동 분기; VLN_PY overrides (HF backends: the torch image python)
BUILD="$QK/llama.cpp/$([ "$(uname -m)" = aarch64 ] && echo build || echo build_x86)"
export LD_LIBRARY_PATH="$BUILD/bin${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"
if [ "$(uname -m)" = aarch64 ]; then
    # Jetson: the system dist-packages holds the native tensorrt/libnvinfer bindings
    export PYTHONPATH="/usr/lib/python3.12/dist-packages${PYTHONPATH:+:$PYTHONPATH}"
fi
cd "$ROOT"
