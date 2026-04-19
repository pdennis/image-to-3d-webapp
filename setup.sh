#!/usr/bin/env bash
# Set up Image-to-3D web app on a new Mac.
# Clones each upstream model repo, copies custom run_inference scripts,
# applies patches, and builds a per-model venv.
#
# Usage:
#   ./setup.sh            # set up everything
#   ./setup.sh sf3d       # set up only one model (sf3d|triposg|hunyuan3d)
#
# Requires: Python 3.10+, git, and (for hunyuan3d custom ops) Xcode CLT.

set -euo pipefail

BASE_DIR="$(cd "$(dirname "$0")" && pwd)"
cd "$BASE_DIR"

# Pinned commits for reproducibility
SF3D_REPO="https://github.com/Stability-AI/stable-fast-3d.git"
SF3D_COMMIT="ff21fc491b4dc5314bf6734c7c0dabd86b5f5bb2"

TRIPOSG_REPO="https://github.com/VAST-AI-Research/TripoSG.git"
TRIPOSG_COMMIT="fc5c40990181e2a756c4e0b1c2f4d6b5202faf8c"

HUNYUAN_REPO="https://github.com/Maxim-Lanskoy/Hunyuan3D-2-Mac.git"
HUNYUAN_COMMIT="df17e93ec2db69afdb236211131e4ee195c4f8f1"

PYTHON="${PYTHON:-python3}"

log()  { printf "\n\033[1;32m==> %s\033[0m\n" "$*"; }
warn() { printf "\n\033[1;33m--- %s\033[0m\n" "$*"; }

clone_pinned() {
    local repo="$1" commit="$2" dir="$3"
    if [ -d "$dir/.git" ]; then
        log "$dir already cloned, skipping"
        return
    fi
    log "Cloning $dir"
    git clone "$repo" "$dir"
    (cd "$dir" && git checkout "$commit")
}

make_venv() {
    local dir="$1"
    if [ ! -d "$dir/venv" ]; then
        log "Creating venv in $dir"
        "$PYTHON" -m venv "$dir/venv"
    fi
    "$dir/venv/bin/pip" install --upgrade pip wheel setuptools
}

setup_top_level() {
    log "Top-level Flask app venv"
    if [ ! -d "venv" ]; then
        "$PYTHON" -m venv venv
    fi
    venv/bin/pip install --upgrade pip wheel
    venv/bin/pip install -r requirements.txt
}

setup_sf3d() {
    clone_pinned "$SF3D_REPO" "$SF3D_COMMIT" "stable-fast-3d"
    cp scripts/sf3d_run_inference.py stable-fast-3d/run_inference.py
    make_venv stable-fast-3d

    log "Installing SF3D requirements (this takes a while — builds custom ops)"
    # SF3D needs torch first, then its requirements which build local wheels
    stable-fast-3d/venv/bin/pip install torch torchvision
    (cd stable-fast-3d && venv/bin/pip install -r requirements.txt)
}

setup_triposg() {
    clone_pinned "$TRIPOSG_REPO" "$TRIPOSG_COMMIT" "triposg"

    log "Applying TripoSG Mac/CPU compatibility patch"
    (cd triposg && git apply --check ../patches/triposg-inference_utils.patch 2>/dev/null \
        && git apply ../patches/triposg-inference_utils.patch \
        || warn "Patch already applied or conflicts — skipping")

    cp scripts/triposg_run_inference.py triposg/run_inference.py
    make_venv triposg

    log "Installing TripoSG requirements"
    triposg/venv/bin/pip install torch torchvision
    (cd triposg && venv/bin/pip install -r requirements.txt)
    # rembg is needed by run_inference.py for background removal
    triposg/venv/bin/pip install rembg onnxruntime
}

setup_hunyuan3d() {
    clone_pinned "$HUNYUAN_REPO" "$HUNYUAN_COMMIT" "hunyuan3d"
    cp scripts/hunyuan3d_run_inference.py hunyuan3d/run_inference.py
    make_venv hunyuan3d

    log "Installing Hunyuan3D requirements (large)"
    hunyuan3d/venv/bin/pip install torch torchvision
    (cd hunyuan3d && venv/bin/pip install -r requirements.txt)
    (cd hunyuan3d && venv/bin/pip install -e .)
}

target="${1:-all}"
case "$target" in
    all)
        setup_top_level
        setup_sf3d
        setup_triposg
        setup_hunyuan3d
        ;;
    top)       setup_top_level ;;
    sf3d)      setup_sf3d ;;
    triposg)   setup_triposg ;;
    hunyuan3d) setup_hunyuan3d ;;
    *) echo "Unknown target: $target"; exit 1 ;;
esac

log "Done. Start the app with: venv/bin/python app.py"
log "Then open http://127.0.0.1:7860"
