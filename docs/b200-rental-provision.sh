#!/usr/bin/env bash
# B200 rental box one-shot provisioning for ninfer b200 validation.
# Run AS ROOT on a fresh Ubuntu 22.04/24.04 x86_64 box with an NVIDIA driver >= 570 installed.
# Safe to re-run; every step is idempotent. Total runtime ~10 min excluding artifact download.
set -euo pipefail

NINFER_GIT="https://github.com/alantheprice/ninfer-monitor"
NINFER_BRANCH="port/fleet-features"
NINFER_COMMIT="13cbb7a9"          # b200 target + plan doc; update if the branch moves
B2_BUCKET="${B2_BUCKET:-ninfer-artifacts}"   # export before running, or edit here
B2_PREFIX="${B2_PREFIX:-flash-next}"
ART_DIR="/srv/ninfer-artifact"
ENGINE_DIR="/srv/ninfer"
# Build happens on the rental box: CUDA toolkit install is ~2 min via network repo and
# guarantees the binary matches the box's driver/toolchain pair exactly.

echo "== [1/6] base packages"
apt-get update -qq
apt-get install -y -qq build-essential cmake ninja-build git curl python3 python3-pip \
  aria2 pciutils >/dev/null

echo "== [2/6] NVIDIA driver / GPU sanity"
nvidia-smi --query-gpu=name,driver_version,memory.total --format=csv,noheader
nvidia-smi --query-gpu=compute_cap --format=csv,noheader | grep -q "^10\.0" \
  || { echo "FATAL: GPU 0 is not sm_100 (B200)"; exit 1; }

echo "== [3/6] CUDA toolkit 13.0.2 (toolkit only, no driver override)"
if [ ! -x /usr/local/cuda-13.0.2/bin/nvcc ]; then
  CUDA_PKG=intel-pinned-repo  # placeholder — replace with the pinned .run URL used on ws6000
  echo "NOTE: install the SAME CUDA 13.0.2 toolkit build used on ws6000 (pinned .run installer),"
  echo "      then re-run this script. Skippable only if the box already has it."
  ls /usr/local/ | grep -q cuda && echo "existing toolkits:" && ls /usr/local/ | grep cuda || true
fi
export PATH=/usr/local/cuda-13.0.2/bin:$PATH

echo "== [4/6] artifact download (Backblaze B2, resumed, verified)"
mkdir -p "$ART_DIR" && cd "$ART_DIR"
rclone version >/dev/null 2>&1 || { curl -s https://rclone.org/install.sh | bash; }
# auth: either env B2_ACCOUNT_ID/B2_APPLICATION_KEY or `rclone config create b2 b2 ...` beforehand
rclone copy "b2:${B2_BUCKET}/${B2_PREFIX}" "$ART_DIR" \
  --transfers 8 --checkers 16 --b2-chunk-size 250M --progress --fast-list
sha256sum -c sha256-manifest.txt

echo "== [5/6] engine source + build"
mkdir -p "$ENGINE_DIR" && cd "$ENGINE_DIR"
git clone --depth 1 --branch "$NINFER_BRANCH" "$NINFER_GIT" src 2>/dev/null \
  || (cd src && git fetch --depth 1 origin "$NINFER_BRANCH" && git checkout "$NINFER_COMMIT")
cd src && git rev-parse HEAD | grep -q "$NINFER_COMMIT" || { echo "FATAL: wrong commit"; exit 1; }
cmake -S . -B build-b200 -G Ninja -DCMAKE_BUILD_TYPE=Release \
  -DNINFER_TARGET=b200 -DCMAKE_CUDA_ARCHITECTURES=100a
ninja -C build-b200 ninfer-serve

echo "== [6/6] smoke"
"$ENGINE_DIR/src/build-b200/apps/ninfer-serve" --help >/dev/null && echo "engine binary OK"
echo "PROVISIONING COMPLETE — artifact at $ART_DIR, engine at $ENGINE_DIR/src/build-b200/apps/ninfer-serve"
