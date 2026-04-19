"""Standalone inference script for SF3D. Called via subprocess from the web app."""
import sys
import os
import time
from contextlib import nullcontext

import numpy as np
import rembg
import torch
from PIL import Image

import sf3d.utils as sf3d_utils
from sf3d.system import SF3D

COND_WIDTH = 512
COND_HEIGHT = 512
COND_DISTANCE = 1.6
COND_FOVY_DEG = 40
BACKGROUND_COLOR = [0.5, 0.5, 0.5]

c2w_cond = sf3d_utils.default_cond_c2w(COND_DISTANCE)
intrinsic, intrinsic_normed_cond = sf3d_utils.create_intrinsic_from_fov_deg(
    COND_FOVY_DEG, COND_HEIGHT, COND_WIDTH
)

device = sf3d_utils.get_device()
print(f"Device: {device}", flush=True)

print("Loading SF3D model...", flush=True)
model = SF3D.from_pretrained(
    "stabilityai/stable-fast-3d",
    config_name="config.yaml",
    weight_name="model.safetensors",
)
model.eval()
model = model.to(device)

rembg_session = rembg.new_session()


def create_batch(input_image):
    img_cond = (
        torch.from_numpy(
            np.asarray(input_image.resize((COND_WIDTH, COND_HEIGHT))).astype(np.float32) / 255.0
        ).float().clip(0, 1)
    )
    mask_cond = img_cond[:, :, -1:]
    rgb_cond = torch.lerp(
        torch.tensor(BACKGROUND_COLOR)[None, None, :], img_cond[:, :, :3], mask_cond
    )
    batch_elem = {
        "rgb_cond": rgb_cond,
        "mask_cond": mask_cond,
        "c2w_cond": c2w_cond.unsqueeze(0),
        "intrinsic_cond": intrinsic.unsqueeze(0),
        "intrinsic_normed_cond": intrinsic_normed_cond.unsqueeze(0),
    }
    return {k: v.unsqueeze(0) for k, v in batch_elem.items()}


def main():
    input_path = sys.argv[1]
    output_path = sys.argv[2]

    image = Image.open(input_path).convert("RGBA")

    # Remove background if needed
    alpha = np.array(image.getchannel("A"))
    if alpha.min() > 0:
        print("Removing background...", flush=True)
        image = rembg.remove(image, session=rembg_session)

    image = sf3d_utils.resize_foreground(image, 0.85, out_size=(COND_WIDTH, COND_HEIGHT))

    print("Generating mesh...", flush=True)
    with torch.no_grad():
        with torch.autocast(device_type=device, dtype=torch.bfloat16) if "cuda" in device else nullcontext():
            batch = create_batch(image)
            batch = {k: v.to(device) for k, v in batch.items()}
            mesh, _ = model.generate_mesh(batch, 1024, "none", -1)
            mesh = mesh[0]

    mesh.export(output_path, file_type="glb", include_normals=True)
    print(f"DONE:{output_path}", flush=True)


if __name__ == "__main__":
    main()
