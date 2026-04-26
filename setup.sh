#!/usr/bin/env bash
# Pod setup. Run once per fresh rental.
# Idempotent: safe to re-run.

set -euo pipefail

WORKSPACE="${WORKSPACE:-/workspace}"
VENV="${WORKSPACE}/.venv"
REPO_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

echo "==> Workspace:    ${WORKSPACE}"
echo "==> Venv:         ${VENV}"
echo "==> Repo:         ${REPO_DIR}"

if [ ! -d "${WORKSPACE}" ]; then
    echo "ERROR: ${WORKSPACE} does not exist. Mount a persistent network volume there."
    exit 1
fi

mkdir -p "${WORKSPACE}/models"
mkdir -p "${WORKSPACE}/vectors"
mkdir -p "${WORKSPACE}/outputs"
mkdir -p "${WORKSPACE}/logs"

# Detect whether we're on a RunPod pytorch image (torch + CUDA already baked in).
# If so, skip the venv and install into the system Python so we keep the
# preinstalled torch wheel that matches the system CUDA libraries.
if python3 -c "import torch" >/dev/null 2>&1; then
    echo "==> Detected preinstalled torch ($(python3 -c 'import torch; print(torch.__version__)'))"
    echo "==> Using system Python (no venv); installing app-level deps only"
    PIP="python3 -m pip"
else
    if [ ! -d "${VENV}" ]; then
        echo "==> No preinstalled torch. Creating virtualenv at ${VENV}"
        python3 -m venv "${VENV}"
    fi
    # shellcheck source=/dev/null
    source "${VENV}/bin/activate"
    PIP="pip"
fi

echo "==> Upgrading pip"
$PIP install --upgrade pip wheel

echo "==> Installing requirements"
$PIP install -r "${REPO_DIR}/requirements.txt"

echo "==> Verifying CUDA"
python - <<'PY'
import torch
print(f"torch:           {torch.__version__}")
print(f"cuda available:  {torch.cuda.is_available()}")
if torch.cuda.is_available():
    print(f"device:          {torch.cuda.get_device_name(0)}")
    print(f"vram total:      {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")
else:
    raise SystemExit("CUDA not available. Aborting.")
PY

echo ""
echo "==> Setup complete."
echo "   Activate the venv in new shells with:  source ${VENV}/bin/activate"
echo "   Next:  huggingface-cli login   (then python download_model.py)"
