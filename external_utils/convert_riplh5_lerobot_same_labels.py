import h5py
import numpy as np
from pathlib import Path
from lerobot.datasets.lerobot_dataset import LeRobotDataset
from tqdm import tqdm

# Configuration
REPO_ID = "lerobot/eve_blocks_6x_same_labels"
DATASET_NAME = "eve_blocks_6x_same_labels"
ORIG_DATASET_PATH = Path("/home/jeremiah/openteach/extracted_data/eve_blocks/h5_files")
FPS = 20
ROOT_DIR = Path(f"/data3/lerobot_data/{DATASET_NAME}")  # Where the dataset will be created locally

# Define your features match your HDF5 content
# Note: 'task' is not defined here but is required in add_frame()
FEATURES = {
    "action": {
        "dtype": "float32",
        "shape": (8,),
        "names": ["x", "y", "z", "quat_x", "quat_y", "quat_z", "quat_w", "gripper"],
    },
    "observation.state": {
        "dtype": "float32",
        "shape": (8,),
        "names": ["joint1", "joint2", "joint3", "joint4", "joint5", "joint6", "joint7", "gripper_pos"],
    },
    "observation.images.camera_front": {
        "dtype": "video",
        "shape": (360, 640, 3), # (H, W, C)
        "names": ["height", "width", "channels"],
    },
    "observation.images.camera_side": {
        "dtype": "video",
        "shape": (360, 640, 3), # (H, W, C)
        "names": ["height", "width", "channels"],
    },
    "observation.images.camera_wrist": {
        "dtype": "video",
        "shape": (360, 640, 3), # (H, W, C)
        "names": ["height", "width", "channels"],
    }
}

def main():
    # 1. Create the empty LeRobotDataset
    dataset = LeRobotDataset.create(
        repo_id=REPO_ID,
        fps=FPS,
        features=FEATURES,
        root=ROOT_DIR,
        use_videos=True, # Encode images to MP4
    )

    # 2. Iterate over all HDF5 files in the directory
    h5_files = sorted(list(ORIG_DATASET_PATH.glob("*.h5")))
    print(f"Found {len(h5_files)} episodes in {ORIG_DATASET_PATH}")

    for h5_path in tqdm(h5_files):
        expected_episode_index = dataset.num_episodes
        print(f"Processing {h5_path.name}...")
        with h5py.File(h5_path, "r") as f:
            # Extract hdf5 data

            # fix quat
            fixed_cartesian_pose_cmd = fix_quat(f['cartesian_pose_cmd'][:])
            actions = np.concatenate([fixed_cartesian_pose_cmd, np.expand_dims(f['gripper_action'], axis=1)], axis=1, dtype=np.float32)
            observation_states = np.concatenate([f['joint_pos'], np.expand_dims(f['gripper_state'], axis=1)], axis=1, dtype=np.float32)
            rgb_frames = f['rgb_frames'][:]
            camera_side_imgs = rgb_frames[:, 0, :, :, ::-1]
            camera_wrist_imgs = rgb_frames[:, 1, :, :, ::-1]
            camera_front_imgs = rgb_frames[:, 2, :, :, ::-1]
            camera_front_imgs[:, :, :140] = 0
            camera_front_imgs[:, :, 500:] = 0

            num_frames = observation_states.shape[0]

            # 3. Iterate over frames
            for i in range(num_frames):
                frame = {
                    "action": actions[i],
                    "observation.state": observation_states[i],
                    "observation.images.camera_front": camera_front_imgs[i],
                    "observation.images.camera_wrist": camera_wrist_imgs[i],
                    "observation.images.camera_side": camera_side_imgs[i],
                    "task": get_task_instructions(h5_path.name),
                }

                # 4. Add frame to buffer
                dataset.add_frame(frame)

            # 5. Save the completed episode (LeRobot may pack many episodes per file)
            dataset.save_episode()
            if dataset.num_episodes != expected_episode_index + 1:
                raise RuntimeError(
                    f"Episode index did not advance after {h5_path.name}: "
                    f"expected {expected_episode_index + 1}, got {dataset.num_episodes}"
                )
            print(
                f"Saved episode_index={expected_episode_index}, "
                f"frames={num_frames}, total_saved={dataset.num_episodes}"
            )

    # 6. Finalize (closes writers and saves metadata)
    dataset.finalize()
    print(f"Dataset created at: {dataset.root}")
    print(
        f"Final counts: episodes={dataset.num_episodes}, frames={dataset.num_frames}. "
        "LeRobot stores many episodes inside shared parquet/mp4 files by default."
    )


def fix_quat(pose_action):
    idcs = pose_action[:, 3] < 0.0
    pose_action[idcs, 3:7] *= -1
    return pose_action


def get_task_instructions(fname):
    color = fname.split('_')[2]
    return f"pick up a block"


if __name__ == "__main__":
    main()
