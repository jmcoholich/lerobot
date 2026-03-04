"""
These actions are in OSC_POSE space. This is assuming wrist camera observations
"""
import numpy as np
import torch
from deoxys.utils.transform_utils import axisangle2quat, quat_multiply
CHUNK_SIZE = 50
DES_TRANSLATION = 0.1
DES_ROTATION = np.deg2rad(22.5)
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


def _canonicalize_quat_first_positive(quat: np.ndarray) -> np.ndarray:
    """Normalize quaternion and keep it in a fixed sign hemisphere.

    Training data effectively uses a consistent sign convention. Enforce the same
    convention during guidance generation so normalization does not explode.
    """
    quat = np.asarray(quat, dtype=np.float32).reshape(4)
    norm = np.linalg.norm(quat)
    if norm < 1e-8:
        raise ValueError("Quaternion norm is too small to normalize.")
    quat = quat / norm
    # Keep the first quaternion component non-negative to match training convention.
    if quat[0] < 0.0:
        quat = -quat
    return quat


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
    if robot.debug:
        quat = np.array([0.99982, 0.00968, 0.01597, 0.00365], dtype=np.float32)
        pos = np.array([[0.45868], [0.03165], [0.26477]])
    else:
        quat, pos = robot.operator.robot_interface.last_eef_quat_and_pos
    quat = _canonicalize_quat_first_positive(quat)
    primitive = torch.as_tensor(primitive, dtype=torch.float32)
    output = torch.zeros(1, CHUNK_SIZE, 8)
    output[0, :, 3:7] = torch.from_numpy(quat)  # unchanged
    aa_chunk = torch.arange(CHUNK_SIZE).reshape(CHUNK_SIZE, 1) * primitive[3:6].reshape(1, 3)
    for i in range(CHUNK_SIZE):
        delta_quat = axisangle2quat(aa_chunk[i].cpu().numpy())
        base_quat = output[0, i, 3:7].cpu().numpy()
        step_quat = quat_multiply(delta_quat, base_quat)
        step_quat = _canonicalize_quat_first_positive(step_quat)
        output[0, i, 3:7] = torch.from_numpy(step_quat)
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
