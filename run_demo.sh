#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")"
export CUDA_VISIBLE_DEVICES="${GPU:-0}"

# Use cuBLASLt from the active PyTorch CUDA environment.
CUBLAS_LT=$(python - <<'PYTHON'
from pathlib import Path
import sysconfig
path = Path(sysconfig.get_paths()["purelib"]) / "nvidia/cu13/lib/libcublasLt.so.13"
print(path if path.is_file() else "")
PYTHON
)
if [[ -z "$CUBLAS_LT" ]]; then
    echo "cuBLASLt for CUDA 13 was not found in the active Python environment." >&2
    exit 1
fi
export LD_PRELOAD="$CUBLAS_LT${LD_PRELOAD:+:$LD_PRELOAD}"
exec python demo.py "$@"
