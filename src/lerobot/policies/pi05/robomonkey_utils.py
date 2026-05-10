import requests
import math
import time
import json
import base64
import numpy as np
from PIL import Image
from pathlib import Path
from io import BytesIO
import os
import torch
import torch.nn.functional as F
import logging

log = logging.getLogger(__name__)


def _resize_nchw(image, target_size, mode="bilinear", antialias=False):
    kwargs = {
        "size": target_size,
        "mode": mode,
    }
    if mode in {"linear", "bilinear", "bicubic", "trilinear"}:
        kwargs["align_corners"] = False

    try:
        return F.interpolate(image, antialias=antialias, **kwargs)
    except TypeError:
        return F.interpolate(image, **kwargs)


def _to_batched_nchw(image):
    if isinstance(image, np.ndarray) and not image.flags.writeable:
        image = image.copy()
    image = torch.as_tensor(image)
    expanded_dims = image.ndim == 3
    if expanded_dims:
        image = image.unsqueeze(0)
    if image.ndim != 4:
        raise ValueError(f"Expected image with 3 or 4 dimensions, got {tuple(image.shape)}")

    channels_last = image.shape[-1] <= 4
    if channels_last:
        image = image.permute(0, 3, 1, 2)
    return image.to(torch.float32), expanded_dims, channels_last


def _restore_image_layout(image, expanded_dims, channels_last):
    if channels_last:
        image = image.permute(0, 2, 3, 1)
    if expanded_dims:
        image = image[0]
    return image


def _float_to_uint8_hwc(image):
    image = torch.as_tensor(image)
    if image.ndim == 4:
        image = image[0]
    if image.shape[0] <= 4:
        image = image.permute(1, 2, 0)
    return image.clamp(0, 1).mul(255).round().to(torch.uint8).cpu().numpy()


def _jpeg_roundtrip(image, quality=95):
    image = np.asarray(image, dtype=np.uint8)
    buffer = BytesIO()
    Image.fromarray(image).convert("RGB").save(buffer, format="JPEG", quality=quality)
    buffer.seek(0)
    return np.asarray(Image.open(buffer).convert("RGB"), dtype=np.uint8)


def _image_to_b64(image):
    buffer = BytesIO()
    image.convert("RGB").save(buffer, format="JPEG")
    return base64.b64encode(buffer.getvalue()).decode("ascii")


def _resize_uint8_hwc(image, target_size):
    image, _, _ = _to_batched_nchw(image)
    image = _resize_nchw(image, target_size, mode="bicubic", antialias=True)
    return image.clamp(0, 255).round().to(torch.uint8)[0].permute(1, 2, 0).cpu().numpy()


def crop_and_resize(image, crop_scale, batch_size=None, target_size=(224, 224)):
    """
    Center-crops an image to have area `crop_scale` * (original image area), and then resizes back
    to original size. We use the same logic seen in the `dlimp` RLDS datasets wrapper to avoid
    distribution shift at test time.

    Args:
        image: Tensor/array of shape (batch_size, H, W, C), (H, W, C), (batch_size, C, H, W),
               or (C, H, W) with values between [0,1].
        crop_scale: The area of the center crop with respect to the original image.
        batch_size: Batch size.
    """
    image, expanded_dims, channels_last = _to_batched_nchw(image)
    batch_size = image.shape[0]
    crop_scale = min(max(crop_scale, 0.0), 1.0)
    crop_ratio = math.sqrt(crop_scale)
    offset = (1.0 - crop_ratio) / 2.0
    y = torch.linspace(
        2.0 * offset - 1.0,
        2.0 * (offset + crop_ratio) - 1.0,
        target_size[0],
        device=image.device,
        dtype=image.dtype,
    )
    x = torch.linspace(
        2.0 * offset - 1.0,
        2.0 * (offset + crop_ratio) - 1.0,
        target_size[1],
        device=image.device,
        dtype=image.dtype,
    )
    try:
        grid_y, grid_x = torch.meshgrid(y, x, indexing="ij")
    except TypeError:
        grid_y, grid_x = torch.meshgrid(y, x)
    grid = torch.stack((grid_x, grid_y), dim=-1).expand(batch_size, -1, -1, -1)
    image = F.grid_sample(image, grid, mode="bilinear", align_corners=True)
    return _restore_image_layout(image, expanded_dims, channels_last)


def get_image_from_maniskill2_obs_dict(env, obs, camera_name=None):
    # obtain image from observation dictionary returned by ManiSkill2 environment
    if camera_name is None:
        if "google_robot" in env.robot_uid:
            camera_name = "overhead_camera"
        elif "widowx" in env.unwrapped.robot_uid:       # NOTE: we cant use from simpler_env.utils.env.observation_utils import get_image_from_maniskill2_obs_dict since we need env.unwrapped.robut_uid
            camera_name = "3rd_view_camera"
        else:
            raise NotImplementedError()
    return obs["image"][camera_name]["rgb"]


def process_image(image_path, crop_scale=0.9, target_size=(224, 224), batch_size=1):
    """
    Process an image by center-cropping and resizing using torch.
    """
    try:
        image = Image.open(image_path).convert("RGB")

        current_size = image.size  # Returns (width, height)
        if current_size == (target_size[1], target_size[0]):
            return image_path

        image = torch.as_tensor(np.asarray(image), dtype=torch.float32).div(255.0)
        image = crop_and_resize(image, crop_scale, batch_size, target_size)
        image = Image.fromarray(_float_to_uint8_hwc(image)).convert("RGB")
        image.save(image_path)

        return None

    except Exception as e:
        raise Exception(f"Error processing image: {str(e)}")


def save_reward_img(image, img_dir):
    image = preprocess_robomonkey_image(image)

    transfer_root = str(Path(img_dir).absolute())
    os.makedirs(transfer_root, exist_ok=True)
    Image.fromarray(image).save(f"{transfer_root}/reward_img.jpg")


def preprocess_robomonkey_image(image):
    # Encode/decode and resize as done in the RLDS dataset builder.
    image = _jpeg_roundtrip(image)
    image = _resize_uint8_hwc(image, (256, 256))
    image = _jpeg_roundtrip(image, quality=95)
    image = _resize_uint8_hwc(image, (256, 256))

    # Match process_image(..., crop_scale=0.9, target_size=(224, 224)) without saving.
    image = torch.as_tensor(image, dtype=torch.float32).div(255.0)
    image = crop_and_resize(image, crop_scale=0.9, batch_size=1, target_size=(224, 224))
    return _float_to_uint8_hwc(image)


def get_simpler_img(env, obs, resize_size, robomonkey_img_dir):
    """
    Takes in environment and observation and returns resized image as numpy array.

    NOTE (Moo Jin): To make input images in distribution with respect to the inputs seen at training time, we follow
                    the same resizing scheme used in the Octo dataloader, which OpenVLA uses for training.
    """
    assert isinstance(resize_size, int)
    image = get_image_from_maniskill2_obs_dict(env, obs)
    save_reward_img(image, robomonkey_img_dir)

    # Preprocess the image the exact same way that the Berkeley Bridge folks did it
    # to minimize distribution shift.
    # NOTE (Moo Jin): Yes, we resize down to 256x256 first even though the image may end up being
    # resized up to a different resolution by some models. This is just so that we're in-distribution
    # w.r.t. the original preprocessing at train time.
    IMAGE_BASE_PREPROCESS_SIZE = 128

    image = _jpeg_roundtrip(image)
    image = _resize_uint8_hwc(image, (IMAGE_BASE_PREPROCESS_SIZE, IMAGE_BASE_PREPROCESS_SIZE))
    return _resize_uint8_hwc(image, (resize_size, resize_size))


def generate_augmented_samples_from_batch(batch_actions, num_samples=32):
    """
    Generate augmented samples based on the mean and variance of a batch of actions.
    """
    # Calculate mean and variance for each dimension
    mean_values = np.mean(batch_actions, axis=0)
    var_values = np.var(batch_actions, axis=0)

    # Define valid ranges for the action dimensions
    min_values = np.array([-1., -1., -1., -1., -1., -1., 0.])
    max_values = np.array([1., 1., 1., 1., 1., 1., 1.])

    # Generate all samples at once
    augmented_array = np.random.normal(
        mean_values, np.sqrt(var_values),
        size=(num_samples, 7)
    )

    # For the 7th dimension (binary), use probability based on mean
    augmented_array[:, -1] = (mean_values[-1] >= 0.5).astype(float)

    # Clip values to valid range
    augmented_array[:, :-1] = np.clip(
        augmented_array[:, :-1],
        min_values[:-1],
        max_values[:-1]
    )

    return augmented_array


def get_rewards(instruction, image, actions, verifier_url):
    # Initialize rewards list
    all_rewards = []
    image_b64 = _image_to_b64(image)

    # Get action rewards in batches of 2, so the reward model fits in a RTX4090 with 24GB memory size
    # Change the `batch_size` accordingly if you are using a different GPU
    # NOTE: batch size 4 can fit on A40 with 48 GB memory
    batch_size = min(len(actions), 32)
    num_batches = math.ceil(len(actions) / batch_size)

    rewards_start_time = time.perf_counter()
    for i in range(num_batches):
        # Get the current batch of actions
        start_idx = i * batch_size
        end_idx = min((i + 1) * batch_size, len(actions))
        action_batch = actions[start_idx:end_idx]

        payload = {
            "instruction": instruction,
            "image_b64": image_b64,
            "action": action_batch.tolist()
        }

        request_start_time = time.perf_counter()
        response = requests.post(f"{verifier_url}/process_with_image_directly", data=json.dumps(payload))
        print(
            f"[ROBOMONKEY] verifier roundtrip batch {i + 1}/{num_batches}: "
            f"{time.perf_counter() - request_start_time:.4f}s"
        )
        print('#' * 100)
        print(response)
        print(response.text)
        print('#' * 100)
        response_data = json.loads(response.text)

        print("[ROBOMONKEY] response_data from verifier: ", response_data)
        all_rewards.extend(response_data["rewards"])

    print(f"[ROBOMONKEY] verifier roundtrip total: {time.perf_counter() - rewards_start_time:.4f}s")
    return all_rewards


def get_robomonkey_action(action_samples, instruction, robomonkey_verifier_img, verifier_url, img_dir, n_augmented):

    image = Image.fromarray(preprocess_robomonkey_image(robomonkey_verifier_img)).convert("RGB")

    actions = generate_augmented_samples_from_batch(
        batch_actions=action_samples.to(torch.float16).squeeze(1).cpu().numpy(),      # NOTE: We only score based on the first step in the current action plan
        num_samples=n_augmented
    )

    rewards = get_rewards(
        instruction,
        image,
        actions,
        verifier_url,
    )

    if len(rewards) == 0:
        log.info("[ROBOMONKEY] NO REWARDS from verifier server shifting to default [0] index action !")

    selected_index = np.argmax(rewards)
    log.info(f"[ROBOMONKEY] Selected index: {selected_index}")

    selected_action = actions[selected_index].copy()
    selected_action[-1] = (selected_action[-1] - 0.5) * 2
    return selected_action


if __name__ == "__main__":

    actions = generate_augmented_samples_from_batch(
        batch_actions=np.array([[0.006793868144893786, -0.019651793629798762, 0.003374671807119146, -0.018341272918726517, 0.047913162388673734, 0.08149821793057914, 1.0]]),
        num_samples=16
    )

    get_rewards(
        "Move the object to the target location",
        Image.open("/srv/flash1/yali30/code/symbotic/open-pi-zero/third_party/RoboMonkey/assets/banner.png"),
        actions,
        "http://127.0.0.1:3100",
    )
