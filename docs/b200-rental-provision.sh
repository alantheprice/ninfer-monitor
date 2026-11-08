#!/usr/bin/env bash
# B200 rental box one-shot provisioning for ninfer b200 validation.
# Run AS ROOT on a fresh Ubuntu 22.04/24.04 x86_64 box with an NVIDIA driver >= 570 installed.
# Safe to re-run; every step is idempotent. Total runtime ~10 min excluding artifact download.
set -euo pipefail

NINFER_GIT="https://github.com/alantheprice/ninfer-monitor"
NINFER_BRANCH="port/fleet-features"
NINFER_COMMIT="13cbb7a9"          # b200 target + plan doc; update if the branch moves
B2_BUCKET="${B2_BUCKET:-t1-45-bg}"          # export before running, or edit here
B2_PREFIX="${B2_PREFIX:-models}"            # remote path prefix holding the artifact
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

echo "== [3/6] CUDA toolkit (any >= 12.8 with sm_100a; use whatever the box already has)"
# The build is toolkit-agnostic: sm_100a is supported from CUDA 12.8 onward and the engine
# compiles clean on both lines (13.0.2 verified on ws6000; 12.8.1 on runpod pytorch images).
# Prefer the box's own toolkit so nvcc stays aligned with the pod driver — never install a
# toolkit newer than the driver.
CUDA_DIR=$(ls -d /usr/local/cuda* 2>/dev/null | sort -V | tail -1)
if [ -z "$CUDA_DIR" ] || [ ! -x "$CUDA_DIR/bin/nvcc" ]; then
  echo "FATAL: no CUDA toolkit under /usr/local/cuda*; install one >= 12.8 and re-run."
  exit 1
fi
CUDA_VERSION=$("$CUDA_DIR/bin/nvcc" -V | tail -1 | grep -oE "V[0-9.]+" | tr -d V)
echo "using toolkit: $CUDA_DIR ($CUDA_VERSION)"
case "$CUDA_VERSION" in
  12.*|13.*) ;;
  *) echo "FATAL: unsupported toolkit $CUDA_VERSION (need 12.8+ or 13.x)"; exit 1 ;;
esac
export PATH="$CUDA_DIR/bin:$PATH"

echo "== [4/6] artifact download (Backblaze B2, resumed, verified)"
mkdir -p "$ART_DIR" && cd "$ART_DIR"
rclone version >/dev/null 2>&1 || { curl -s https://rclone.org/install.sh | bash; }
# auth: B2_DOWNLOAD_KEY / B2_DOWNLOAD_KEY_ID exported in the environment, or one
# positional argument "keyid:key" (parsed here, never echoed back).
if [ -n "$1" ] && [ -z "$B2_DOWNLOAD_KEY" ]; then
  B2_DOWNLOAD_KEY_ID="${1%%:*}"; B2_DOWNLOAD_KEY="${1#*:}"
fi
: "${B2_DOWNLOAD_KEY:?need B2_DOWNLOAD_KEY (env, or pass keyid:key as argument 1)}"
: "${B2_DOWNLOAD_KEY_ID:?need B2_DOWNLOAD_KEY_ID (env, or pass keyid:key as argument 1)}"
rclone config create b2b s3 provider Other \
  access_key_id "$B2_DOWNLOAD_KEY_ID" secret_access_key "$B2_DOWNLOAD_KEY" \
  endpoint "https://${B2_ENDPOINT:-s3.us-east-005.backblazeb2.com}" >/dev/null
rclone copy "b2b:${B2_BUCKET}/${B2_PREFIX}" "$ART_DIR" \
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
