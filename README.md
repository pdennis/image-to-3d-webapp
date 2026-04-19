# Image to 3D — unified web app

A Flask web app that wraps three open image-to-3D models behind a single UI. Each
model runs in its own Python virtualenv via subprocess so their dependency trees
don't collide. Runs locally on macOS (Apple Silicon tested) or Linux.

| Model | Quality | Speed | Notes |
|---|---|---|---|
| [SF3D](https://github.com/Stability-AI/stable-fast-3d) | textured | ~1 min | CPU-forced on Mac |
| [TripoSG](https://github.com/VAST-AI-Research/TripoSG) | high geometry | ~8–10 min | MPS + rembg |
| [Hunyuan3D-2](https://github.com/Maxim-Lanskoy/Hunyuan3D-2-Mac) | highest quality | ~15 min | first run downloads ~10 GB |

Output is a `.glb` mesh, previewable in-browser via `<model-viewer>` and
downloadable as `.glb` or `.stl`.

## Setup

```bash
git clone <your-fork-url> 3d-webapp
cd 3d-webapp
./setup.sh              # clones the 3 upstream model repos + builds venvs
```

The three model repos (~10+ GB combined after weights download) are cloned into
`stable-fast-3d/`, `triposg/`, `hunyuan3d/` and ignored by git — they are not
part of this repo. Setup pins each to a known-good commit.

To install just one model:

```bash
./setup.sh sf3d         # or: triposg, hunyuan3d, top
```

### Requirements

- Python 3.10+
- macOS or Linux with ~40 GB free disk for all three models + weights
- Xcode Command Line Tools on macOS (needed to build some wheels)

## Running

```bash
venv/bin/python app.py
```

Open http://127.0.0.1:7860.

## Layout

```
app.py                  # Flask app — routes + job queue + polling UI
requirements.txt        # deps for app.py only
setup.sh                # one-shot setup for a fresh Mac
scripts/                # custom run_inference.py for each model
  sf3d_run_inference.py
  triposg_run_inference.py
  hunyuan3d_run_inference.py
patches/                # small patches applied on top of upstream repos
  triposg-inference_utils.patch   # CPU/MPS fallback (removes CUDA-only paths)
```

At runtime the app launches each job as:

```
<model_dir>/venv/bin/python <model_dir>/run_inference.py <input.png> <output.glb>
```

Output meshes land in `output_web/`. Uploads are deleted after each job.

## Licenses

This repo contains only the glue code (app.py, run_inference scripts, setup
script). Each upstream model keeps its own license and is cloned separately at
setup time. In particular, Hunyuan3D-2 is distributed under Tencent's
non-commercial license — review it before any non-personal use.
