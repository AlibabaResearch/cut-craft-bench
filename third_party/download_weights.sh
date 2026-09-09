#!/bin/bash
# One-click downloader for the expert-model microservice weights.
#
# Usage:
#   bash download_weights.sh            # download every weight that has a direct URL
#   MODEL_ROOT=/path bash download_weights.sh   # relocate the weights root
#
# Idempotent: files that already exist (and are non-empty) are skipped, so it is
# safe to re-run after an interrupted download. Downloads use `wget -c` (resume).
#
# This script covers the weights that can be fetched with a single command. A few
# models need an upstream repo/tool and are only PRINTED as manual steps at the end
# (MonST3R, TransNetV2, 6DRepNet, DOVER). See README.md "Downloading the weights".

set -u

BASE_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
export MODEL_ROOT="${MODEL_ROOT:-$BASE_DIR/models}"

FAILED=()

have() { command -v "$1" >/dev/null 2>&1; }

# download <dest> <url>
download() {
    local dest="$1" url="$2"
    if [ -s "$dest" ]; then
        echo "  [skip] $(basename "$dest") already present"
        return 0
    fi
    echo "  [get ] $(basename "$dest")"
    mkdir -p "$(dirname "$dest")"
    if wget -c -q --show-progress -O "$dest" "$url"; then
        if [ -s "$dest" ]; then
            return 0
        fi
    fi
    # Clean up empty/partial file so a re-run retries cleanly.
    [ -s "$dest" ] || rm -f "$dest"
    echo "  [FAIL] $(basename "$dest") <- $url"
    FAILED+=("$(basename "$dest")")
    return 1
}

if ! have wget; then
    echo "ERROR: wget is required but not found. Install it and re-run." >&2
    exit 1
fi

echo "==> Weights root: $MODEL_ROOT"
mkdir -p "$MODEL_ROOT"/{clip,demucs,dinov2,dover,e2_quality,monst3r,panns,raft,whisper} \
         "$MODEL_ROOT"/transnetv2/TransNetV2/inference-pytorch

echo ""
echo "==> OpenAI CLIP (clip 8010, e2quality 8006, transnetv2 8001)"
download "$MODEL_ROOT/clip/ViT-L-14.pt" \
  "https://openaipublic.azureedge.net/clip/models/b8cca3fd41ae0c99ba7e8951adf17d267cdb84cd88be6f7c2e0eca1737a03836/ViT-L-14.pt"
# e2quality keeps its own ViT-L/14 copy
if [ -s "$MODEL_ROOT/clip/ViT-L-14.pt" ] && [ ! -s "$MODEL_ROOT/e2_quality/ViT-L-14.pt" ]; then
    cp "$MODEL_ROOT/clip/ViT-L-14.pt" "$MODEL_ROOT/e2_quality/ViT-L-14.pt"
    echo "  [copy] e2_quality/ViT-L-14.pt"
fi
download "$MODEL_ROOT/clip/ViT-B-32.pt" \
  "https://openaipublic.azureedge.net/clip/models/40d365715913c9da98579312b702a82c18be219cc2a73407c4526f58eba950af/ViT-B-32.pt"

echo ""
echo "==> RAFT-Small (raft 8002)"
download "$MODEL_ROOT/raft/raft_small_C_T_V2-01064c6d.pth" \
  "https://download.pytorch.org/models/raft_small_C_T_V2-01064c6d.pth"

echo ""
echo "==> DINOv2 ViT-B/14 (dinov2 8003)"
download "$MODEL_ROOT/dinov2/dinov2_vitb14_pretrain.pth" \
  "https://dl.fbaipublicfiles.com/dinov2/dinov2_vitb14/dinov2_vitb14_pretrain.pth"

echo ""
echo "==> htdemucs (demucs 8005)"
download "$MODEL_ROOT/demucs/955717e8-8726e21a.th" \
  "https://dl.fbaipublicfiles.com/demucs/hybrid_transformer/955717e8-8726e21a.th"

echo ""
echo "==> PANNs CNN14 + AudioSet labels (panns 8007)"
download "$MODEL_ROOT/panns/Cnn14_mAP=0.431.pth" \
  "https://zenodo.org/record/3987831/files/Cnn14_mAP%3D0.431.pth"
download "$MODEL_ROOT/panns/class_labels_indices.csv" \
  "http://storage.googleapis.com/us_audioset/youtube_corpus/v1/csv/class_labels_indices.csv"

echo ""
echo "==> E2 quality: LAION aesthetic head + MUSIQ-SPAQ (e2quality 8006)"
download "$MODEL_ROOT/e2_quality/sa_0_4_vit_l_14_linear.pth" \
  "https://github.com/LAION-AI/aesthetic-predictor/raw/main/sa_0_4_vit_l_14_linear.pth"
download "$MODEL_ROOT/e2_quality/musiq_spaq_ckpt-358bb6af.pth" \
  "https://huggingface.co/chaofengc/IQA-PyTorch-Weights/resolve/main/musiq_spaq_ckpt-358bb6af.pth"

echo ""
echo "==> YOLOv8m (yolov8 8009) -- weights live in the service dir, not MODEL_ROOT"
download "$BASE_DIR/yolov8/yolov8m.pt" \
  "https://github.com/ultralytics/assets/releases/download/v8.3.0/yolov8m.pt"

echo ""
echo "==> Whisper small (whisper_asr 8004)"
if [ -s "$MODEL_ROOT/whisper/small.pt" ]; then
    echo "  [skip] whisper/small.pt already present"
elif have python; then
    if python -c "import whisper; whisper.load_model('small', download_root='$MODEL_ROOT/whisper')" 2>/dev/null; then
        echo "  [get ] whisper/small.pt"
    else
        echo "  [FAIL] whisper small (is the 'openai-whisper' package installed?)"
        FAILED+=("whisper/small.pt")
    fi
else
    echo "  [skip] python not found; whisper_asr will auto-download 'small' on first run"
fi

# ---------------------------------------------------------------------------
echo ""
echo "============================================================"
if [ ${#FAILED[@]} -eq 0 ]; then
    echo "Direct downloads complete."
else
    echo "Direct downloads finished with ${#FAILED[@]} failure(s):"
    printf '  - %s\n' "${FAILED[@]}"
    echo "Re-run this script to retry (existing files are skipped)."
fi

cat <<'EOF'

Manual / tool-managed weights (not covered above -- see README.md):
  * movieshots (8013)  open_clip_pytorch_model.bin is OPTIONAL; if absent the service
                       falls back to open_clip pretrained="openai" (auto-downloads to ~/.cache).
  * monst3r   (8015)   place model.safetensors + config.json under models/monst3r/
                       (MonST3R checkpoint from the upstream project).
  * transnetv2(8001)   place transnetv2-pytorch-weights.pth under
                       models/transnetv2/TransNetV2/inference-pytorch/ (clone soCzech/TransNetV2).
  * sixdrepnet(8014)   the `sixdrepnet` pip package auto-downloads 6DRepNet_300W_LP_AFLW2000.pth
                       on first use; move it to models/6DRepNet_300W_LP_AFLW2000.pth.
  * dover     (legacy) only if you swap e2quality for dover: DOVER.pth + dover.yml under models/dover/.
  * dnsmos (8008) and u2net (8011) need NO weights.
EOF

[ ${#FAILED[@]} -eq 0 ] || exit 1
