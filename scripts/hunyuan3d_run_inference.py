"""Standalone inference script for Hunyuan3D-2. Called via subprocess from the web app."""
import sys
import os
import traceback
import torch
from PIL import Image

from hy3dgen.shapegen import (
    Hunyuan3DDiTFlowMatchingPipeline,
    FloaterRemover,
    DegenerateFaceRemover,
    FaceReducer,
)
from hy3dgen.rembg import BackgroundRemover


def main():
    input_path = sys.argv[1]
    output_path = sys.argv[2]
    use_fast = "--fast" in sys.argv

    try:
        print("Loading image...", flush=True)
        image = Image.open(input_path)

        print("Removing background...", flush=True)
        rembg = BackgroundRemover()
        if image.mode == "RGB" or (image.mode == "RGBA" and min(image.getchannel("A").getextrema()) > 0):
            image = rembg(image)

        model_path = "tencent/Hunyuan3D-2"

        if use_fast:
            print("Loading Hunyuan3D-2 (fast variant)...", flush=True)
            pipeline = Hunyuan3DDiTFlowMatchingPipeline.from_pretrained(
                model_path,
                subfolder="hunyuan3d-dit-v2-0-fast",
                variant="fp16",
            )
        else:
            print("Loading Hunyuan3D-2...", flush=True)
            pipeline = Hunyuan3DDiTFlowMatchingPipeline.from_pretrained(model_path)

        print("Generating mesh (octree_resolution=256)...", flush=True)
        mesh = pipeline(
            image=image,
            num_inference_steps=30,
            mc_algo="mc",
            octree_resolution=256,
            generator=torch.manual_seed(2025),
        )[0]

        print(f"Mesh extracted: {type(mesh)}", flush=True)
        print("Post-processing...", flush=True)
        mesh = FloaterRemover()(mesh)
        print("FloaterRemover done", flush=True)
        mesh = DegenerateFaceRemover()(mesh)
        print("DegenerateFaceRemover done", flush=True)
        mesh = FaceReducer()(mesh)
        print("FaceReducer done", flush=True)

        mesh.export(output_path)
        print(f"DONE:{output_path}", flush=True)

    except Exception as e:
        print(f"ERROR: {e}", flush=True)
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()
