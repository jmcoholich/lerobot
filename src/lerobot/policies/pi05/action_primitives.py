"""
These actions are in OSC_POSE space. This is assuming wrist camera observations
"""
import numpy as np
import torch
CHUNK_SIZE = 50
DES_TRANSLATION = 0.1
DES_ROTATION = np.deg2rad(15.0)
GRIPPER_ACTION = -1.0  # Hardcoded as open

BACKWARD = (0.0, 0.0, DES_TRANSLATION / CHUNK_SIZE, 0.0, 0.0, 0.0, GRIPPER_ACTION)
FORWARD = (0.0, 0.0, -DES_TRANSLATION / CHUNK_SIZE, 0.0, 0.0, 0.0, GRIPPER_ACTION)
RIGHT = (0.0, DES_TRANSLATION / CHUNK_SIZE, 0.0, 0.0, 0.0, 0.0, GRIPPER_ACTION)
LEFT = (0.0, -DES_TRANSLATION / CHUNK_SIZE, 0.0, 0.0, 0.0, 0.0, GRIPPER_ACTION)
UP = (-DES_TRANSLATION / CHUNK_SIZE, 0.0, 0.0, 0.0, 0.0, 0.0, GRIPPER_ACTION)
DOWN = (DES_TRANSLATION / CHUNK_SIZE, 0.0, 0.0, 0.0, 0.0, 0.0, GRIPPER_ACTION)
ROTATE_CCW = (0.0, 0.0, 0.0, 0.0, 0.0, DES_ROTATION / CHUNK_SIZE, GRIPPER_ACTION)
ROTATE_CW = (0.0, 0.0, 0.0, 0.0, 0.0, -DES_ROTATION / CHUNK_SIZE, GRIPPER_ACTION)

LABEL2ACTION = {
    "backward": BACKWARD,
    "forward": FORWARD,
    "right": RIGHT,
    "left": LEFT,
    "up": UP,
    "down": DOWN,
    "rotate_ccw": ROTATE_CCW,
    "rotate_cw": ROTATE_CW
}

def get_guidance_action_from_text(label, postprocessor, action_normalizer):
    if label not in LABEL2ACTION:
        raise ValueError(f"Label '{label}' not found in LABEL2ACTION mapping.")
    chunk = torch.tensor(LABEL2ACTION[label], dtype=torch.float32)
    chunk = rotate_to_robot_frame(chunk)
    normalized_chunk = action_normalizer(chunk)  # for some reason, this is None. I'm very sus of LeRobot
    # recovered_chunk = postprocessor(chunk)  # just for debugging
    return normalized_chunk


def rotate_to_robot_frame(chunk):
    return chunk
