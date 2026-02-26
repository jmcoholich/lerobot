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
    "up": BACKWARD,
    "down": FORWARD,
    "right": RIGHT,
    "left": LEFT,
    "backward": UP,
    "forward": DOWN,
    "rotate_ccw": ROTATE_CCW,
    "rotate_cw": ROTATE_CW
}

def get_guidance_action_from_text(label, postprocessor, robot):
    if label not in LABEL2ACTION:
        raise ValueError(f"Label '{label}' not found in LABEL2ACTION mapping.")
    action_primitive = torch.tensor(LABEL2ACTION[label], dtype=torch.float32)
    # chunk = rotate_to_robot_frame(chunk, robot)
    chunk = create_cartesian_chunk(action_primitive, robot)
    q01 = postprocessor.steps[0].stats['action']['q01']
    q99 = postprocessor.steps[0].stats['action']['q99']
    denom = q99 - q01
    chunk = 2.0 * (chunk - q01) / denom - 1.0
    return chunk

def create_cartesian_chunk(primitive, robot):
    """Creates a chunk of absolute eef pose actions"""
    quat, pos = robot.operator.robot_interface.last_eef_quat_and_pos
    primitive = torch.tensor(primitive)
    # TODO doesn't work for ccw yet
    output = torch.zeros(1, CHUNK_SIZE, 8)
    output[0, :, 3:7] = torch.from_numpy(quat)  # unchanged
    output[0, :, :3] = torch.from_numpy(pos).squeeze()
    output[0, :, :3] += torch.arange(CHUNK_SIZE).reshape(CHUNK_SIZE, 1) * primitive[:3].reshape(1, 3)
    output[0, :, 7] = GRIPPER_ACTION
    return output

# def rotate_to_robot_frame(chunk, robot):
#     eef_rot_mat, _ = robot.operator.robot_interface.last_eef_rot_and_pos
#     rot_180_x = torch.tensor([[1, 0, 0], [0, -1, 0], [0, 0, -1]], dtype=torch.float32)
#     rot_mat = torch.from_numpy(eef_rot_mat).float() @ rot_180_xs
#     chunk[:3] = (rot_mat @ chunk[:3].reshape(3, 1)).reshape(3)
#     chunk[3:6] = (rot_mat @ chunk[3:6].reshape(3, 1)).reshape(3)
#     return chunk


if __name__ == "__main__":
    # for testing
    x = create_cartesian_chunk(
        (0.0, 0.0, DES_TRANSLATION / CHUNK_SIZE, 0.0, 0.0, 0.0, GRIPPER_ACTION),
        None)
    breakpoint()
