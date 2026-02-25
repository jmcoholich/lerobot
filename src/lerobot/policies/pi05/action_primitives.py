"""
These actions are in OSC_POSE space. This is assuming wrist camera observations
"""
import numpy as np
import torch
CHUNK_SIZE = 50
DES_TRANSLATION = 0.5  # units seem arbitrary in OSC_POSE space due to scaling/normalization in controller stack, so just picked a reasonable amount for one action chunk
DES_ROTATION = 2.5  # ditto
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

def get_guidance_action_from_text(label, postprocessor, robot):
    if label not in LABEL2ACTION:
        raise ValueError(f"Label '{label}' not found in LABEL2ACTION mapping.")
    chunk = torch.tensor(LABEL2ACTION[label], dtype=torch.float32)
    chunk = rotate_to_robot_frame(chunk, robot)
    q01 = postprocessor.steps[0].stats['action']['q01']
    q99 = postprocessor.steps[0].stats['action']['q99']
    denom = q99 - q01
    chunk = 2.0 * (chunk - q01) / denom - 1.0
    return chunk


def rotate_to_robot_frame(chunk, robot):
    eef_rot_mat, _ = robot.operator.robot_interface.last_eef_rot_and_pos
    rot_180_x = torch.tensor([[1, 0, 0], [0, -1, 0], [0, 0, -1]], dtype=torch.float32)
    rot_mat = torch.from_numpy(eef_rot_mat).float() @ rot_180_x
    chunk[:3] = (rot_mat @ chunk[:3].reshape(3, 1)).reshape(3)
    chunk[3:6] = (rot_mat @ chunk[3:6].reshape(3, 1)).reshape(3)
    return chunk
