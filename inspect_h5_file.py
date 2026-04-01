import io
from pathlib import Path

import h5py
import numpy as np
from PIL import Image


H5_PATH = Path(
    "/data3/nvidia_LIBERO_regen/success_only/libero_object_regen/"
    "pick_up_the_alphabet_soup_and_place_it_in_the_basket_demo.hdf5"
)
DEMO_KEY = "demo_0"
FRAME_INDEX = 0
OUTPUT_DIR = Path("/home/jcoholich/lerobot/tmp/libero_frame_inspection")


def decode_jpeg_frame(encoded_frame):
    return np.asarray(
        Image.open(io.BytesIO(np.asarray(encoded_frame, dtype=np.uint8).tobytes())).convert("RGB"),
        dtype=np.uint8,
    )


def save_image(arr, path):
    path.parent.mkdir(parents=True, exist_ok=True)
    Image.fromarray(arr).save(path)
    print(f"Saved {path}")


def main():
    with h5py.File(H5_PATH, "r") as f:
        # breakpoint()
        demo = f["data"][DEMO_KEY]
        obs = demo["obs"]

        agentview = decode_jpeg_frame(obs["agentview_rgb_jpeg"][FRAME_INDEX])
        wrist = decode_jpeg_frame(obs["eye_in_hand_rgb_jpeg"][FRAME_INDEX])

        save_image(agentview, OUTPUT_DIR / "agentview_raw.png")
        save_image(np.flip(agentview, axis=(0, 1)).copy(), OUTPUT_DIR / "agentview_rot180.png")

        save_image(wrist, OUTPUT_DIR / "wrist_raw.png")
        save_image(np.flip(wrist, axis=(0, 1)).copy(), OUTPUT_DIR / "wrist_rot180.png")

        print(f"HDF5 file: {H5_PATH}")
        print(f"Demo: {DEMO_KEY}")
        print(f"Frame: {FRAME_INDEX}")
        print(f"Output directory: {OUTPUT_DIR}")


if __name__ == "__main__":
    main()
