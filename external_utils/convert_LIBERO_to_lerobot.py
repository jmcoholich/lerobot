import io

import h5py
import numpy as np
from pathlib import Path
from PIL import Image

from lerobot.datasets.lerobot_dataset import LeRobotDataset
from tqdm import tqdm

# Configuration
REPO_ID = "lerobot/my_libero_object_dataset"
DATASET_NAME = "my_libero_object_dataset"
ORIG_DATASET_PATH = Path("/data3/nvidia_LIBERO_regen/success_only/libero_object_regen")
FILE_GLOB = "*.hdf5"
FPS = 20
ROOT_DIR = Path(f"/data3/lerobot_data/{DATASET_NAME}")

# Match the observation/action format used by LIBERO evaluation:
# - native 7D relative actions
# - observation.state = [eef_pos(3), axis-angle(3), gripper_qpos(2)]
# - observation.images.image = agent view
# - observation.images.image2 = wrist view
#
# The regenerated dataset already has the desired camera orientation, so we
# preserve frames exactly as stored.
FEATURES = {
    "action": {
        "dtype": "float32",
        "shape": (7,),
        "names": ["dx", "dy", "dz", "dax", "day", "daz", "gripper"],
    },
    "observation.state": {
        "dtype": "float32",
        "shape": (8,),
        "names": [
            "eef_pos_x",
            "eef_pos_y",
            "eef_pos_z",
            "eef_axisangle_x",
            "eef_axisangle_y",
            "eef_axisangle_z",
            "gripper_qpos_left",
            "gripper_qpos_right",
        ],
    },
    "observation.images.image": {
        "dtype": "video",
        "shape": (256, 256, 3),
        "names": ["height", "width", "channels"],
    },
    "observation.images.image2": {
        "dtype": "video",
        "shape": (256, 256, 3),
        "names": ["height", "width", "channels"],
    },
}


def main():
    dataset = LeRobotDataset.create(
        repo_id=REPO_ID,
        fps=FPS,
        features=FEATURES,
        root=ROOT_DIR,
        use_videos=True,
    )

    h5_files = sorted(ORIG_DATASET_PATH.glob(FILE_GLOB))
    print(f"Found {len(h5_files)} LIBERO files in {ORIG_DATASET_PATH}")

    for h5_path in tqdm(h5_files):
        task_instruction = get_task_instruction(h5_path.name)
        print(f"Processing {h5_path.name}...")

        with h5py.File(h5_path, "r") as f:
            if "data" not in f:
                raise KeyError(f"Expected /data group in {h5_path}")

            demo_keys = sorted(f["data"].keys(), key=get_demo_index)
            print(f"  Found {len(demo_keys)} demos")

            for demo_key in demo_keys:
                expected_episode_index = dataset.num_episodes
                demo = f["data"][demo_key]

                actions = np.asarray(demo["actions"], dtype=np.float32)
                observation_states = build_observation_state(demo)
                obs = demo["obs"]
                image_imgs = decode_image_stream(obs, "agentview_rgb_jpeg")
                image2_imgs = decode_image_stream(obs, "eye_in_hand_rgb_jpeg")

                num_frames = observation_states.shape[0]
                if actions.shape[0] != num_frames:
                    raise RuntimeError(
                        f"Action/state length mismatch in {h5_path.name}/{demo_key}: "
                        f"{actions.shape[0]} vs {num_frames}"
                    )

                for i in range(num_frames):
                    frame = {
                        "action": actions[i],
                        "observation.state": observation_states[i],
                        "observation.images.image": image_imgs[i],
                        "observation.images.image2": image2_imgs[i],
                        "task": task_instruction,
                    }
                    dataset.add_frame(frame)

                dataset.save_episode()
                if dataset.num_episodes != expected_episode_index + 1:
                    raise RuntimeError(
                        f"Episode index did not advance after {h5_path.name}/{demo_key}: "
                        f"expected {expected_episode_index + 1}, got {dataset.num_episodes}"
                    )

                print(
                    f"  Saved episode_index={expected_episode_index}, "
                    f"demo={demo_key}, frames={num_frames}, total_saved={dataset.num_episodes}"
                )

    dataset.finalize()
    print(f"Dataset created at: {dataset.root}")
    print(
        f"Final counts: episodes={dataset.num_episodes}, frames={dataset.num_frames}. "
        "LeRobot stores many episodes inside shared parquet/mp4 files by default."
    )


def build_observation_state(demo):
    obs = demo["obs"]
    ee_pos = np.asarray(obs["ee_pos"], dtype=np.float32)
    ee_axisangle = np.asarray(obs["ee_ori"], dtype=np.float32)
    gripper_qpos = np.asarray(obs["gripper_states"], dtype=np.float32)
    return np.concatenate([ee_pos, ee_axisangle, gripper_qpos], axis=1).astype(np.float32)


def decode_image_stream(obs, jpeg_key):
    if jpeg_key not in obs:
        raise KeyError(f"Expected {jpeg_key!r} in obs keys {list(obs.keys())}")
    return np.stack(
        [decode_jpeg_frame(obs[jpeg_key][frame_idx]) for frame_idx in range(obs[jpeg_key].shape[0])],
        axis=0,
    )


def decode_jpeg_frame(encoded_frame):
    image = Image.open(io.BytesIO(np.asarray(encoded_frame, dtype=np.uint8).tobytes())).convert("RGB")
    return np.asarray(image, dtype=np.uint8)


def get_demo_index(demo_key):
    return int(demo_key.split("_")[-1])


def get_task_instruction(fname):
    stem = Path(fname).stem
    if stem.endswith("_demo"):
        stem = stem[: -len("_demo")]
    return stem.replace("_", " ")


if __name__ == "__main__":
    main()
