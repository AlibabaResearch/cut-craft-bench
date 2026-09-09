# Expert Model Microservices

Each subdirectory is a self-contained FastAPI service (`<service>/app.py`) that exposes one
expert model over HTTP on a fixed port. The benchmark calls them through
[`benchmark/service_client.py`](../benchmark/service_client.py).

Every service implements the same two endpoints:

- `GET /health` — readiness and which weights were loaded
- `POST /predict` — `multipart/form-data` upload (`file`), plus per-service form options

## Quick start

```bash
# Launch / stop / inspect every service
bash start_services.sh start
bash start_services.sh status
bash start_services.sh stop
```

Environment overrides:

| Variable | Default | Purpose |
|---|---|---|
| `MODEL_ROOT` | `third_party/models` | Root directory holding all model weights |
| `MODEL_DIR` | `$MODEL_ROOT/<service>` | Per-service weight directory (set individually) |
| `EXPERT_SERVICE_PYTHON` | `python` | Interpreter used to launch each service |
| `EXPERT_SERVICE_LOG_DIR` | `/tmp` | Where per-service logs are written |

## Weight layout

Weights are not distributed with this repository. Place them under `MODEL_ROOT`
(default `third_party/models/`) using the layout below:

```
models/
├── 6DRepNet_300W_LP_AFLW2000.pth
├── clip/
│   ├── ViT-L-14.pt                  # clip service
│   ├── open_clip_pytorch_model.bin  # movieshots (optional; else auto-downloads)
│   └── ViT-B-32.pt                  # transnetv2 semantic scene-change detection
├── demucs/955717e8-8726e21a.th
├── dinov2/dinov2_vitb14_pretrain.pth
├── dover/
│   ├── DOVER.pth
│   └── DOVER-master/dover.yml
├── e2_quality/
│   ├── ViT-L-14.pt
│   ├── sa_0_4_vit_l_14_linear.pth
│   └── musiq_spaq_ckpt-358bb6af.pth
├── monst3r/model.safetensors
├── panns/
│   ├── Cnn14_mAP=0.431.pth
│   └── class_labels_indices.csv
├── raft/raft_small_C_T_V2-01064c6d.pth
├── transnetv2/TransNetV2/inference-pytorch/transnetv2-pytorch-weights.pth
└── whisper/small.pt                 # filename follows WHISPER_MODEL_SIZE
```

YOLOv8 is the exception: its weights (`yolov8m.pt`, falling back to `yolov8n.pt`) are read from
the service directory itself (`third_party/yolov8/`), overridable with `MODEL_DIR`.

## Downloading the weights

Nothing is downloaded at runtime: each service loads from `MODEL_ROOT` and raises
`FileNotFoundError` when a file is missing. (The only exception is `places365` / `movieshots`,
which fall back to `open_clip`'s `pretrained="openai"` auto-download when
`open_clip_pytorch_model.bin` is absent.)

**One-click:** run the bundled script from `third_party/` — it fetches every weight that has a
direct URL, skips files already present (safe to re-run), and prints the few manual steps at the end:

```bash
bash download_weights.sh              # or: MODEL_ROOT=/path bash download_weights.sh
```

Or fetch them by hand — the equivalent commands, run from `third_party/`:

```bash
export MODEL_ROOT="$(pwd)/models"          # export to relocate the cache
mkdir -p "$MODEL_ROOT"/{clip,demucs,dinov2,dover,e2_quality,monst3r,panns,raft,whisper} \
         "$MODEL_ROOT"/transnetv2/TransNetV2/inference-pytorch

# --- OpenAI CLIP: ViT-L/14 (clip 8010, e2quality 8006) + ViT-B/32 (transnetv2 8001) ---
wget -O "$MODEL_ROOT/clip/ViT-L-14.pt" \
  https://openaipublic.azureedge.net/clip/models/b8cca3fd41ae0c99ba7e8951adf17d267cdb84cd88be6f7c2e0eca1737a03836/ViT-L-14.pt
cp "$MODEL_ROOT/clip/ViT-L-14.pt" "$MODEL_ROOT/e2_quality/ViT-L-14.pt"
wget -O "$MODEL_ROOT/clip/ViT-B-32.pt" \
  https://openaipublic.azureedge.net/clip/models/40d365715913c9da98579312b702a82c18be219cc2a73407c4526f58eba950af/ViT-B-32.pt

# --- RAFT-Small (raft 8002) ---
wget -O "$MODEL_ROOT/raft/raft_small_C_T_V2-01064c6d.pth" \
  https://download.pytorch.org/models/raft_small_C_T_V2-01064c6d.pth

# --- DINOv2 ViT-B/14 (dinov2 8003) ---
wget -O "$MODEL_ROOT/dinov2/dinov2_vitb14_pretrain.pth" \
  https://dl.fbaipublicfiles.com/dinov2/dinov2_vitb14/dinov2_vitb14_pretrain.pth

# --- htdemucs (demucs 8005) ---
wget -O "$MODEL_ROOT/demucs/955717e8-8726e21a.th" \
  https://dl.fbaipublicfiles.com/demucs/hybrid_transformer/955717e8-8726e21a.th

# --- PANNs CNN14 + AudioSet labels (panns 8007) ---
wget -O "$MODEL_ROOT/panns/Cnn14_mAP=0.431.pth" \
  https://zenodo.org/record/3987831/files/Cnn14_mAP%3D0.431.pth
wget -O "$MODEL_ROOT/panns/class_labels_indices.csv" \
  http://storage.googleapis.com/us_audioset/youtube_corpus/v1/csv/class_labels_indices.csv

# --- E2 quality: LAION aesthetic head + MUSIQ-SPAQ (e2quality 8006) ---
wget -O "$MODEL_ROOT/e2_quality/sa_0_4_vit_l_14_linear.pth" \
  https://github.com/LAION-AI/aesthetic-predictor/raw/main/sa_0_4_vit_l_14_linear.pth
wget -O "$MODEL_ROOT/e2_quality/musiq_spaq_ckpt-358bb6af.pth" \
  https://huggingface.co/chaofengc/IQA-PyTorch-Weights/resolve/main/musiq_spaq_ckpt-358bb6af.pth

# --- Whisper small (whisper_asr 8004): the package fetches into MODEL_ROOT ---
python -c "import whisper; whisper.load_model('small', download_root='$MODEL_ROOT/whisper')"

# --- YOLOv8m (yolov8 8009): weights live in the service dir, NOT MODEL_ROOT ---
wget -O yolov8/yolov8m.pt \
  https://github.com/ultralytics/assets/releases/download/v8.3.0/yolov8m.pt
```

The remaining weights need an upstream repo or tool, so they are fetched separately:

- **`open_clip_pytorch_model.bin`** (`movieshots`) — optional. Skip it and `movieshots`
  falls back to `open_clip.create_model_and_transforms("ViT-L-14", pretrained="openai")`,
  which caches under `~/.cache`. To pin it locally, save the OpenAI ViT-L/14 open_clip checkpoint
  as `models/clip/open_clip_pytorch_model.bin`. (`places365` would also read this file, but it is
  not started — see the note below — so nothing else needs it.)
- **MonST3R** (`monst3r` 8015) — download the MonST3R checkpoint from the upstream project and
  place `model.safetensors` + `config.json` under `models/monst3r/` so that
  `AsymmetricCroCo3DStereo.from_pretrained(models/monst3r)` resolves; see the vendored repo at
  `monst3r/monst3r_repo/` for its download script.
- **TransNetV2** (`transnetv2` 8001) — clone [soCzech/TransNetV2](https://github.com/soCzech/TransNetV2)
  and place the PyTorch weights at
  `models/transnetv2/TransNetV2/inference-pytorch/transnetv2-pytorch-weights.pth` (convert the
  released TF weights with `inference-pytorch/convert_weights.py` if you only have the TF export).
- **6DRepNet** (`sixdrepnet` 8014) — the `sixdrepnet` package auto-downloads
  `6DRepNet_300W_LP_AFLW2000.pth` on first use; move it to
  `models/6DRepNet_300W_LP_AFLW2000.pth`, or grab it from
  [thohemp/6DRepNet](https://github.com/thohemp/6DRepNet).
- **DOVER** (`dover`, legacy) — only needed if you swap `e2quality` for `dover`. Download
  `DOVER.pth` from [DOVER releases](https://github.com/VQAssessment/DOVER/releases) plus its
  `dover.yml` config into `models/dover/` (config under `models/dover/DOVER-master/`).
- **`dnsmos` (8008)** and **`u2net` (8011)** need no weights — both are pure signal/CV processing.
  The Haar face cascade used by `yolov8` and `movieshots` also needs no download; it ships inside
  `opencv-python` and is loaded from `cv2.data.haarcascades`.

## Services

| Port | Service | Model | Upstream source | VRAM |
|---|---|---|---|---|
| 8001 | `transnetv2` | TransNetV2 (PyTorch port, 7.6M params) + OpenCLIP ViT-B/32 | [soCzech/TransNetV2](https://github.com/soCzech/TransNetV2); weights converted TF→PyTorch via `inference-pytorch/convert_weights.py` | ~200 MB |
| 8002 | `raft` | RAFT-Small | `torchvision.models.optical_flow` (auto-download) | ~500 MB |
| 8003 | `dinov2` | DINOv2 ViT-B/14, 768-d embeddings | `torch.hub` — facebookresearch/dinov2 | ~1 GB |
| 8004 | `whisper_asr` | Whisper (`small` by default) | `pip install openai-whisper` | ~1 GB |
| 8005 | `demucs` | htdemucs (Hybrid Transformer Demucs), 4 stems | `pip install demucs` (Meta Research) | ~2 GB |
| 8006 | `e2quality` | OpenCLIP ViT-L/14 + LAION aesthetic head + MUSIQ (SPAQ) | Follows VBench `aesthetic_quality.py` / `imaging_quality.py` | ~2 GB |
| 8006 | `dover` *(legacy)* | DOVER, with a frame-metric fallback | [VQAssessment/DOVER](https://github.com/VQAssessment/DOVER) | ~500 MB |
| 8007 | `panns` | PANNs CNN14, 2048-d embeddings + audio tags | PANNs CNN14 (`Cnn14_mAP=0.431`) | ~300 MB |
| 8008 | `dnsmos` | Signal-processing audio quality analysis (librosa) | — (no weights) | none, CPU |
| 8009 | `yolov8` | YOLOv8m detection + OpenCV Haar face cascade | [ultralytics/ultralytics](https://github.com/ultralytics/ultralytics) | ~1.5 GB |
| 8010 | `clip` | OpenCLIP ViT-L/14 (OpenAI pretrained), 768-d | [mlfoundations/open_clip](https://github.com/mlfoundations/open_clip) | ~2 GB |
| 8011 | `u2net` | Spectral-residual saliency + LAB colour saliency + Laplacian sharpness | Hou & Zhang 2007 (classical CV, no network) | none, CPU |
| 8013 | `movieshots` | Zero-shot 8-level shot-scale classification via OpenCLIP + Haar face-area rule | Built on OpenCLIP ViT-L/14 | ~2 GB |
| 8014 | `sixdrepnet` | 6DRepNet head pose (yaw/pitch/roll), 300W-LP + AFLW2000 | [thohemp/6DRepNet](https://github.com/thohemp/6DRepNet) | ~500 MB |
| 8015 | `monst3r` | MonST3R / DUSt3R ViT-Large — 6DoF camera pose + focal length | `naver/DUSt3R_ViTLarge_BaseDecoder_512_dpt`; upstream repo vendored at `monst3r/monst3r_repo/` | — |

## Evaluation dimensions served

| Service | Dimensions |
|---|---|
| `transnetv2` | A1 shot count, B1 shot duration, B2 beat sync, D1 transition detection, D2 effect transitions, D4 transition timing |
| `raft` | D1 camera-move transitions, E1 camera motion (pan/tilt/dolly/zoom/static), E3 physical consistency, F1 A/V sync |
| `dinov2` | D1 similar-frame & contrast transitions, A4 style, E3 style consistency |
| `whisper_asr` | D1 setup–payoff & logical transitions, D3 transition A/V relation, F1 A/V sync |
| `demucs` | B2 beat sync, D3 transition A/V relation, F3 cross-shot audio consistency |
| `e2quality` / `dover` | E2 no-reference image quality (sharpness, lighting, contrast, detail, aesthetics) |
| `panns` | D1 similar-audio transitions, F3 cross-shot audio consistency |
| `dnsmos` | F2 audio quality |
| `yolov8` | D1 POV / enter-exit frame / empty shot, E1 shot scale & focus |
| `clip` | C1 montage, D1 setup–payoff / logical / POV transitions |
| `u2net` | D1 occlusion & empty-shot transitions, E1 composition & focus |
| `movieshots` | E1 shot scale (ECU/CU/MCU/MS/MLS/LS/ELS/XLS) |
| `sixdrepnet` | D1 POV shots, E1 camera angle |
| `monst3r` | E1 camera motion (pan/tilt/dolly/zoom/crane/static) |

## Notes

- **`e2quality` supersedes `dover`.** Both bind port 8006 and cannot run together.
  `start_services.sh` launches `e2quality`; the `dover` directory is kept only as a
  legacy reference. Pick one before starting the stack.
- **`dnsmos` is not DNSMOS P.835.** Despite the directory name, the service performs
  signal-processing audio quality analysis with librosa and returns
  `objective_quality_score`, `snr_db`, `dynamic_range_db`, `spectral_richness`,
  `onset_density`, `clip_ratio`, `silence_ratio`. No ONNX weights are required.
- **`u2net` does not use U²-Net.** It implements classical spectral-residual saliency on CPU.
- **`places365` (8012) is not started and nothing calls it.** `run_model_sequence_with_watchdog.sh`
  omits it and `mode_b_eval.py` treats its result as permanent `N/A`; the directory is kept only as a
  legacy reference (like `dover`), and its `open_clip_pytorch_model.bin` weight is not required.
- **`places365` / `movieshots` are zero-shot CLIP classifiers**, not the original Places365 or
  MovieShots models. `movieshots` loads `models/clip/open_clip_pytorch_model.bin` (optional; see
  above); `e2quality` keeps its own ViT-L/14 copy under `models/e2_quality/`.
- `monst3r/monst3r_repo/` is vendored upstream code and is excluded from version control via
  `.gitignore`; clone it from the MonST3R project and keep its upstream `LICENSE` files intact.
- `u2net` and `dnsmos` are CPU-only; the GPU IDs assigned to them in `start_services.sh` are
  placeholders.
