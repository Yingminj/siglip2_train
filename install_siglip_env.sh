#!/usr/bin/env bash
set -euo pipefail

# Local SigLIP/SigLIP2 training environment installer.
#
# Usage:
#   bash install_siglip_env.sh
#
# Optional overrides:
#   ENV_NAME=siglip2 bash install_siglip_env.sh
#   PYTHON_VERSION=3.10 bash install_siglip_env.sh
#   TORCH_BACKEND=cu121 bash install_siglip_env.sh
#   TORCH_BACKEND=cpu bash install_siglip_env.sh
#   TORCH_BACKEND=cu121 TORCH_VERSION=2.5.1 TORCHVISION_VERSION=0.20.1 TORCHAUDIO_VERSION=2.5.1 bash install_siglip_env.sh

ENV_NAME="${ENV_NAME:-siglip2}"
PYTHON_VERSION="${PYTHON_VERSION:-3.10}"
TORCH_BACKEND="${TORCH_BACKEND:-cu121}"
TORCH_VERSION="${TORCH_VERSION:-}"
TORCHVISION_VERSION="${TORCHVISION_VERSION:-}"
TORCHAUDIO_VERSION="${TORCHAUDIO_VERSION:-}"
PIP_INDEX_URL="${PIP_INDEX_URL:-https://pypi.tuna.tsinghua.edu.cn/simple}"

case "${TORCH_BACKEND}" in
  cpu)
    TORCH_CONDA_PACKAGES=(cpuonly)
    ;;
  cu118)
    TORCH_CONDA_PACKAGES=(pytorch-cuda=11.8)
    ;;
  cu121)
    TORCH_CONDA_PACKAGES=(pytorch-cuda=12.1)
    ;;
  cu124)
    TORCH_CONDA_PACKAGES=(pytorch-cuda=12.4)
    ;;
  *)
    echo "Unsupported TORCH_BACKEND='${TORCH_BACKEND}' for conda install. Use cpu, cu118, cu121, or cu124." >&2
    exit 1
    ;;
esac

if ! command -v conda >/dev/null 2>&1; then
  echo "conda was not found. Please install Miniconda/Anaconda or add conda to PATH." >&2
  exit 1
fi

echo "==> Project: $(pwd)"
echo "==> Environment: ${ENV_NAME}"
echo "==> Python: ${PYTHON_VERSION}"
echo "==> PyTorch backend: ${TORCH_BACKEND}"

if conda env list | awk '{print $1}' | grep -qx "${ENV_NAME}"; then
  echo "==> Conda env '${ENV_NAME}' already exists; reusing it."
else
  echo "==> Creating conda env '${ENV_NAME}'..."
  conda create -y -n "${ENV_NAME}" "python=${PYTHON_VERSION}" pip
fi

eval "$(conda shell.bash hook)"
conda activate "${ENV_NAME}"

echo "==> Python executable: $(command -v python)"
python -m pip install -U pip setuptools wheel -i "${PIP_INDEX_URL}"

echo "==> Installing PyTorch packages with conda via Tsinghua mirrors..."
TORCH_INSTALL_CMD=(
  conda install -y --override-channels
  -c https://mirrors.tuna.tsinghua.edu.cn/anaconda/cloud/pytorch
  -c https://mirrors.tuna.tsinghua.edu.cn/anaconda/cloud/nvidia
  -c https://mirrors.tuna.tsinghua.edu.cn/anaconda/pkgs/main
)

if [[ -n "${TORCH_VERSION}" ]]; then
  TORCH_INSTALL_CMD+=("pytorch=${TORCH_VERSION}")
else
  TORCH_INSTALL_CMD+=(pytorch)
fi

if [[ -n "${TORCHVISION_VERSION}" ]]; then
  TORCH_INSTALL_CMD+=("torchvision=${TORCHVISION_VERSION}")
else
  TORCH_INSTALL_CMD+=(torchvision)
fi

if [[ -n "${TORCHAUDIO_VERSION}" ]]; then
  TORCH_INSTALL_CMD+=("torchaudio=${TORCHAUDIO_VERSION}")
else
  TORCH_INSTALL_CMD+=(torchaudio)
fi

TORCH_INSTALL_CMD+=("${TORCH_CONDA_PACKAGES[@]}")

"${TORCH_INSTALL_CMD[@]}"

echo "==> Installing SigLIP training dependencies..."
python -m pip install -i "${PIP_INDEX_URL}" \
  transformers \
  accelerate \
  safetensors \
  sentencepiece \
  numpy \
  pillow \
  opencv-python \
  matplotlib \
  seaborn \
  scikit-learn \
  tqdm \
  pyyaml \
  pyzmq \
  torchmetrics==1.7.3

echo "==> Optional RealSense dependency..."
if python -m pip install -i "${PIP_INDEX_URL}" pyrealsense2; then
  echo "    pyrealsense2 installed."
else
  echo "    pyrealsense2 install failed; continuing because training does not require it."
fi

echo "==> Verifying imports..."
python - <<'PY'
import importlib
import sys

packages = [
    "torch",
    "torchvision",
    "transformers",
    "numpy",
    "PIL",
    "cv2",
    "matplotlib",
    "seaborn",
    "yaml",
    "zmq",
]

for package in packages:
    importlib.import_module(package)

import torch
print("Python:", sys.version.split()[0])
print("Torch:", torch.__version__)
print("CUDA available:", torch.cuda.is_available())
if torch.cuda.is_available():
    print("CUDA runtime:", torch.version.cuda)
    print("GPU:", torch.cuda.get_device_name(0))
PY

cat <<EOF

Done.

Activate this environment with:
  conda activate ${ENV_NAME}

Run a quick project import check:
  python -c "from siglip2_trainer.multiview_config import MultiViewConfig; print(MultiViewConfig.DEVICE)"

For this repo, remember to update hard-coded paths in:
  siglip2_trainer/config.py
  siglip2_trainer/multiview_config.py

If GPU is still unavailable, first check the host driver:
  nvidia-smi
EOF
