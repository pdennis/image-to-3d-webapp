"""
Unified Image-to-3D web app supporting multiple models:
  - SF3D (Stable Fast 3D) - fast, textured meshes
  - TripoSG - high quality geometry
  - Hunyuan3D-2 - highest quality geometry
Each model runs in its own venv via subprocess.
Jobs run asynchronously so the browser never times out.
"""
import os
import subprocess
import threading
import time
import uuid

from flask import Flask, request, send_file, render_template_string, jsonify
from werkzeug.utils import secure_filename
from PIL import Image

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
OUTPUT_DIR = os.path.join(BASE_DIR, "output_web")
UPLOAD_DIR = os.path.join(BASE_DIR, "uploads")
os.makedirs(OUTPUT_DIR, exist_ok=True)
os.makedirs(UPLOAD_DIR, exist_ok=True)

EXAMPLE_DIR = os.path.join(BASE_DIR, "stable-fast-3d", "demo_files", "examples")

MODELS = {
    "sf3d": {
        "name": "SF3D",
        "desc": "Fast, textured meshes (~1min)",
        "venv": os.path.join(BASE_DIR, "stable-fast-3d", "venv", "bin", "python"),
        "script": os.path.join(BASE_DIR, "stable-fast-3d", "run_inference.py"),
        "cwd": os.path.join(BASE_DIR, "stable-fast-3d"),
        "env_extra": {"SF3D_USE_CPU": "1"},
    },
    "triposg": {
        "name": "TripoSG",
        "desc": "High quality geometry (~8-10min)",
        "venv": os.path.join(BASE_DIR, "triposg", "venv", "bin", "python"),
        "script": os.path.join(BASE_DIR, "triposg", "run_inference.py"),
        "cwd": os.path.join(BASE_DIR, "triposg"),
        "env_extra": {},
    },
    "hunyuan3d": {
        "name": "Hunyuan3D-2",
        "desc": "Highest quality (~15min, first run downloads model)",
        "venv": os.path.join(BASE_DIR, "hunyuan3d", "venv", "bin", "python"),
        "script": os.path.join(BASE_DIR, "hunyuan3d", "run_inference.py"),
        "cwd": os.path.join(BASE_DIR, "hunyuan3d"),
        "env_extra": {},
    },
}

# Job tracking
jobs = {}  # job_id -> {status, model, output, elapsed, error, stderr_tail}

app = Flask(__name__)


def run_job(job_id: str, model_id: str, input_path: str):
    """Run a model in a background thread."""
    cfg = MODELS[model_id]
    out_path = os.path.join(OUTPUT_DIR, f"{job_id}.glb")

    env = os.environ.copy()
    env.update(cfg["env_extra"])

    cmd = [cfg["venv"], cfg["script"], input_path, out_path]

    jobs[job_id]["status"] = "running"
    start = time.time()

    try:
        # Merge stdout and stderr so we capture everything and nothing blocks
        proc = subprocess.Popen(
            cmd,
            cwd=cfg["cwd"],
            env=env,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
        )

        # Stream combined output to capture progress
        output_lines = []
        for line in proc.stdout:
            decoded = line.decode("utf-8", errors="replace").rstrip()
            if decoded:
                output_lines.append(decoded)
            # Keep last 100 lines
            if len(output_lines) > 100:
                output_lines = output_lines[-100:]
            jobs[job_id]["stderr_tail"] = "\n".join(output_lines[-5:])

        proc.wait()
        elapsed = time.time() - start

        if proc.returncode != 0:
            jobs[job_id]["status"] = "error"
            jobs[job_id]["error"] = "\n".join(output_lines[-30:])
        elif not os.path.exists(out_path):
            jobs[job_id]["status"] = "error"
            jobs[job_id]["error"] = "No output file produced.\n" + "\n".join(output_lines[-30:])
        else:
            size = os.path.getsize(out_path)
            size_str = f"{size/1024:.0f}KB" if size < 1024*1024 else f"{size/1024/1024:.1f}MB"
            jobs[job_id]["status"] = "done"
            jobs[job_id]["output"] = f"/output/{job_id}.glb"
            jobs[job_id]["elapsed"] = f"{elapsed:.1f}"
            jobs[job_id]["size"] = size_str
    except Exception as e:
        jobs[job_id]["status"] = "error"
        jobs[job_id]["error"] = str(e)
    finally:
        # Clean up upload
        if os.path.exists(input_path):
            os.unlink(input_path)


HTML = """<!DOCTYPE html>
<html>
<head>
    <title>Image to 3D</title>
    <style>
        * { box-sizing: border-box; margin: 0; padding: 0; }
        body { font-family: -apple-system, system-ui, sans-serif; background: #111; color: #eee; padding: 2rem; }
        h1 { margin-bottom: 0.3rem; }
        .subtitle { color: #888; margin-bottom: 1.5rem; font-size: 0.95rem; }
        .container { max-width: 900px; margin: 0 auto; }

        .upload-area {
            border: 2px dashed #444; border-radius: 12px; padding: 2.5rem;
            text-align: center; cursor: pointer; transition: border-color 0.2s;
            margin-bottom: 1.2rem;
        }
        .upload-area:hover, .upload-area.dragover { border-color: #6c6; }
        .upload-area img { max-width: 280px; max-height: 280px; border-radius: 8px; }

        .examples { display: flex; gap: 0.5rem; flex-wrap: wrap; margin-bottom: 1.2rem; }
        .examples img {
            width: 72px; height: 72px; object-fit: cover; border-radius: 8px;
            cursor: pointer; border: 2px solid transparent; transition: border-color 0.2s;
        }
        .examples img:hover { border-color: #6c6; }

        .model-select { display: flex; gap: 0.8rem; margin-bottom: 1.2rem; flex-wrap: wrap; }
        .model-opt {
            flex: 1; min-width: 160px; padding: 1rem; border: 2px solid #333; border-radius: 10px;
            cursor: pointer; transition: all 0.2s; background: #1a1a1a;
        }
        .model-opt:hover { border-color: #555; }
        .model-opt.selected { border-color: #6c6; background: #1a2a1a; }
        .model-opt h3 { font-size: 1rem; margin-bottom: 0.3rem; }
        .model-opt p { font-size: 0.8rem; color: #888; }

        .btn {
            background: #2a6; color: white; border: none; padding: 0.8rem 2rem;
            border-radius: 8px; font-size: 1.1rem; cursor: pointer; transition: background 0.2s;
            width: 100%;
        }
        .btn:hover { background: #3b7; }
        .btn:disabled { background: #555; cursor: not-allowed; }

        .status {
            margin: 1rem 0; padding: 1rem; border-radius: 8px; background: #1a1a1a;
            display: none; font-size: 0.95rem; white-space: pre-wrap; font-family: monospace;
        }
        .status.error { border-left: 3px solid #c44; }

        .results-grid { display: flex; gap: 1rem; flex-wrap: wrap; margin-top: 1.5rem; }
        .result-card {
            flex: 1; min-width: 280px; border: 1px solid #333; border-radius: 10px;
            padding: 1rem; background: #1a1a1a;
        }
        .result-card h3 { margin-bottom: 0.5rem; font-size: 0.95rem; }
        .result-card model-viewer { width: 100%; height: 350px; border-radius: 8px; background: #222; }
        .result-card .meta { color: #888; font-size: 0.8rem; margin-top: 0.5rem; }
        .result-card a { color: #6c6; text-decoration: none; font-size: 0.85rem; }
    </style>
    <script type="module" src="https://ajax.googleapis.com/ajax/libs/model-viewer/3.3.0/model-viewer.min.js"></script>
</head>
<body>
<div class="container">
    <h1>Image to 3D</h1>
    <p class="subtitle">Upload an image, pick a model, and generate a 3D mesh. All models run locally.</p>

    <div class="examples" id="examples"></div>

    <div class="upload-area" id="dropzone" onclick="document.getElementById('file').click()">
        <p id="drop-text">Drop an image here or click to upload</p>
        <img id="preview" style="display:none">
        <input type="file" id="file" accept="image/*" style="display:none">
    </div>

    <div class="model-select" id="model-select"></div>

    <button class="btn" id="generate" onclick="generate()" disabled>Generate 3D Model</button>

    <div class="status" id="status"></div>

    <div class="results-grid" id="results"></div>
</div>

<script>
const MODELS = MODELS_JSON;
let selectedFile = null;
let selectedModel = 'sf3d';

// Render model options
const modelSelect = document.getElementById('model-select');
Object.entries(MODELS).forEach(([id, m]) => {
    const div = document.createElement('div');
    div.className = 'model-opt' + (id === selectedModel ? ' selected' : '');
    div.innerHTML = `<h3>${m.name}</h3><p>${m.desc}</p>`;
    div.onclick = () => {
        document.querySelectorAll('.model-opt').forEach(el => el.classList.remove('selected'));
        div.classList.add('selected');
        selectedModel = id;
    };
    modelSelect.appendChild(div);
});

// Load examples
fetch('/examples').then(r => r.json()).then(files => {
    const container = document.getElementById('examples');
    files.forEach(f => {
        const img = document.createElement('img');
        img.src = '/example/' + f;
        img.title = f;
        img.onclick = async () => {
            const resp = await fetch('/example/' + f);
            const blob = await resp.blob();
            selectedFile = new File([blob], f, {type: blob.type});
            document.getElementById('preview').src = URL.createObjectURL(blob);
            document.getElementById('preview').style.display = 'block';
            document.getElementById('drop-text').style.display = 'none';
            document.getElementById('generate').disabled = false;
        };
        container.appendChild(img);
    });
});

// Drag & drop
const dropzone = document.getElementById('dropzone');
['dragover','dragenter'].forEach(e => dropzone.addEventListener(e, ev => { ev.preventDefault(); dropzone.classList.add('dragover'); }));
['dragleave','drop'].forEach(e => dropzone.addEventListener(e, () => dropzone.classList.remove('dragover')));
dropzone.addEventListener('drop', ev => { ev.preventDefault(); handleFile(ev.dataTransfer.files[0]); });
document.getElementById('file').addEventListener('change', function() { handleFile(this.files[0]); });

function handleFile(file) {
    if (!file) return;
    selectedFile = file;
    document.getElementById('preview').src = URL.createObjectURL(file);
    document.getElementById('preview').style.display = 'block';
    document.getElementById('drop-text').style.display = 'none';
    document.getElementById('generate').disabled = false;
}

async function generate() {
    if (!selectedFile) return;
    const btn = document.getElementById('generate');
    const statusDiv = document.getElementById('status');
    const modelName = MODELS[selectedModel].name;

    btn.disabled = true;
    btn.textContent = `Starting ${modelName}...`;
    statusDiv.style.display = 'block';
    statusDiv.className = 'status';
    statusDiv.textContent = `Submitting job to ${modelName}...`;

    const form = new FormData();
    form.append('image', selectedFile);
    form.append('model', selectedModel);

    try {
        const resp = await fetch('/generate', { method: 'POST', body: form });
        if (!resp.ok) throw new Error(await resp.text());
        const data = await resp.json();
        const jobId = data.job_id;

        // Poll for completion
        const startTime = Date.now();
        const poll = async () => {
            const r = await fetch('/job/' + jobId);
            const job = await r.json();

            const elapsed = ((Date.now() - startTime) / 1000).toFixed(0);

            if (job.status === 'running') {
                let msg = `${modelName} running... ${elapsed}s elapsed`;
                if (job.stderr_tail) msg += '\\n\\n' + job.stderr_tail;
                statusDiv.textContent = msg;
                setTimeout(poll, 2000);
            } else if (job.status === 'done') {
                statusDiv.textContent = `${modelName}: Done in ${job.elapsed}s (${job.size})`;

                const results = document.getElementById('results');
                const card = document.createElement('div');
                card.className = 'result-card';
                card.innerHTML = `
                    <h3>${modelName}</h3>
                    <model-viewer src="${job.output}" auto-rotate camera-controls shadow-intensity="1" exposure="1.0"
                        style="width:100%;height:350px;border-radius:8px;background:#222;"></model-viewer>
                    <div class="meta">${job.elapsed}s &middot; ${job.size}</div>
                    <a href="${job.output}" download>Download GLB</a> &middot;
                    <a href="/convert_stl/${job.output.split('/').pop()}" download>Download STL</a>
                `;
                results.insertBefore(card, results.firstChild);
                btn.disabled = false;
                btn.textContent = 'Generate 3D Model';
            } else {
                statusDiv.className = 'status error';
                statusDiv.textContent = `${modelName} error:\\n${job.error || 'Unknown error'}`;
                btn.disabled = false;
                btn.textContent = 'Generate 3D Model';
            }
        };
        poll();
    } catch (e) {
        statusDiv.className = 'status error';
        statusDiv.textContent = 'Error: ' + e.message;
        btn.disabled = false;
        btn.textContent = 'Generate 3D Model';
    }
}
</script>
</body>
</html>"""


@app.route("/")
def index():
    import json
    models_json = json.dumps({k: {"name": v["name"], "desc": v["desc"]} for k, v in MODELS.items()})
    html = HTML.replace("MODELS_JSON", models_json)
    return render_template_string(html)


@app.route("/examples")
def examples():
    if os.path.exists(EXAMPLE_DIR):
        return jsonify(sorted(os.listdir(EXAMPLE_DIR)))
    return jsonify([])


@app.route("/example/<name>")
def example(name):
    return send_file(os.path.join(EXAMPLE_DIR, secure_filename(name)))


@app.route("/generate", methods=["POST"])
def generate():
    if "image" not in request.files:
        return "No image uploaded", 400

    model_id = request.form.get("model", "sf3d")
    if model_id not in MODELS:
        return f"Unknown model: {model_id}", 400

    # Save uploaded image
    file = request.files["image"]
    job_id = uuid.uuid4().hex
    input_path = os.path.join(UPLOAD_DIR, f"{job_id}.png")
    image = Image.open(file.stream).convert("RGBA")
    image.save(input_path)

    # Create job and start in background
    jobs[job_id] = {
        "status": "starting",
        "model": model_id,
        "output": None,
        "elapsed": None,
        "size": None,
        "error": None,
        "stderr_tail": "",
    }

    thread = threading.Thread(target=run_job, args=(job_id, model_id, input_path), daemon=True)
    thread.start()

    return jsonify({"job_id": job_id})


@app.route("/job/<job_id>")
def job_status(job_id):
    if job_id not in jobs:
        return "Job not found", 404
    return jsonify(jobs[job_id])


@app.route("/output/<name>")
def output(name):
    return send_file(os.path.join(OUTPUT_DIR, secure_filename(name)))


@app.route("/convert_stl/<name>")
def convert_stl(name):
    import trimesh
    glb_path = os.path.join(OUTPUT_DIR, secure_filename(name))
    if not os.path.exists(glb_path):
        return "File not found", 404
    stl_name = secure_filename(name).rsplit(".", 1)[0] + ".stl"
    stl_path = os.path.join(OUTPUT_DIR, stl_name)
    if not os.path.exists(stl_path):
        scene = trimesh.load(glb_path)
        if isinstance(scene, trimesh.Scene):
            mesh = scene.to_mesh()
        else:
            mesh = scene
        mesh.export(stl_path)
    return send_file(stl_path, as_attachment=True, download_name=stl_name)


if __name__ == "__main__":
    print("Starting Image-to-3D server...")
    print(f"Models: {', '.join(m['name'] for m in MODELS.values())}")
    app.run(host="127.0.0.1", port=7860, debug=False, threaded=True)
