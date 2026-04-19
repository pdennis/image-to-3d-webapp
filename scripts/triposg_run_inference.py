"""Standalone inference script for TripoSG. Called via subprocess from the web app."""
import sys
import os
import numpy as np
import torch
import trimesh
from PIL import Image
from huggingface_hub import snapshot_download

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "scripts"))

from triposg.pipelines.pipeline_triposg import TripoSGPipeline

# Determine device
if torch.backends.mps.is_available():
    device = "mps"
    dtype = torch.float32  # mps has issues with float16 on some ops
elif torch.cuda.is_available():
    device = "cuda"
    dtype = torch.float16
else:
    device = "cpu"
    dtype = torch.float32

print(f"Device: {device}, dtype: {dtype}", flush=True)


def prepare_image_simple(image: Image.Image) -> Image.Image:
    """Simple image preparation - remove background using rembg, pad to square."""
    import rembg

    # Remove background if needed
    if image.mode != "RGBA":
        image = image.convert("RGBA")

    alpha = np.array(image.getchannel("A"))
    if alpha.min() > 0:
        image = rembg.remove(image)

    # Convert to white background
    img_array = np.array(image).astype(np.float32) / 255.0
    rgb = img_array[:, :, :3]
    a = img_array[:, :, 3:4]
    white_bg = rgb * a + 1.0 * (1 - a)

    # Find bounding box of foreground
    mask = a.squeeze() > 0.1
    rows = np.any(mask, axis=1)
    cols = np.any(mask, axis=0)
    if not rows.any():
        return image
    rmin, rmax = np.where(rows)[0][[0, -1]]
    cmin, cmax = np.where(cols)[0][[0, -1]]

    # Crop and pad to square with margin
    cropped = white_bg[rmin:rmax+1, cmin:cmax+1]
    h, w = cropped.shape[:2]
    max_dim = max(h, w)
    pad = int(max_dim * 0.1)
    size = max_dim + 2 * pad

    result = np.ones((size, size, 3), dtype=np.float32)
    y_off = (size - h) // 2
    x_off = (size - w) // 2
    result[y_off:y_off+h, x_off:x_off+w] = cropped

    return Image.fromarray((result * 255).astype(np.uint8))


def main():
    input_path = sys.argv[1]
    output_path = sys.argv[2]

    # Download weights if needed
    weights_dir = os.path.join(os.path.dirname(__file__), "pretrained_weights", "TripoSG")
    if not os.path.exists(os.path.join(weights_dir, "model_index.json")):
        print("Downloading TripoSG weights...", flush=True)
        snapshot_download(repo_id="VAST-AI/TripoSG", local_dir=weights_dir)

    print("Loading model...", flush=True)
    pipe = TripoSGPipeline.from_pretrained(weights_dir).to(device, dtype)

    print("Preparing image...", flush=True)
    image = Image.open(input_path).convert("RGBA")
    img_pil = prepare_image_simple(image)

    print("Generating mesh...", flush=True)
    with torch.no_grad():
        outputs = pipe(
            image=img_pil,
            generator=torch.Generator(device="cpu").manual_seed(42),
            num_inference_steps=50,
            guidance_scale=7.0,
            use_flash_decoder=False,
        ).samples[0]

    mesh = trimesh.Trimesh(outputs[0].astype(np.float32), np.ascontiguousarray(outputs[1]))
    mesh.export(output_path)
    print(f"DONE:{output_path}", flush=True)


if __name__ == "__main__":
    main()
