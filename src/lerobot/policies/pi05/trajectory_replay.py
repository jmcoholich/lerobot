"""Replay recorded flow-matching inputs with an already-loaded PI05 policy, without a robot."""

from contextlib import nullcontext
import json
from pathlib import Path
import tempfile

import h5py
import torch


@torch.inference_mode()
def replay_sampling_call(model, record, device):
    """Return the full padded sampler output from exact recorded inputs and initial noise."""
    device = torch.device(device)
    inputs = record["inputs"]
    settings = json.loads(record.attrs["settings_json"])

    def tensor(dataset):
        return torch.from_numpy(dataset[()]).to(device)

    images = [tensor(inputs["images"][key]) for key in sorted(inputs["images"], key=int)]
    masks = [tensor(inputs["image_masks"][key]) for key in sorted(inputs["image_masks"], key=int)]
    autocast = settings.pop("autocast_enabled")
    dtype = getattr(torch, settings.pop("autocast_dtype").removeprefix("torch."))
    context = torch.autocast(device_type=device.type, dtype=dtype) if autocast else nullcontext()
    model.eval()
    with context:
        return model.sample_actions(
            images, masks, tensor(inputs["tokens"]), tensor(inputs["masks"]),
            noise=tensor(inputs["noise"]),
            guidance_actions=tensor(inputs["guidance_actions"]) if "guidance_actions" in inputs else None,
            consistency_guidance=tensor(inputs["consistency_guidance"]) if "consistency_guidance" in inputs else None,
            **settings,
        )


@torch.inference_mode()
def replay_trajectory_chunk(policy, path, chunk_index=0, *, device=None, rtol=1e-4, atol=1e-5):
    """Reproduce and verify the normalized queued actions, returning [1, horizon, action_dim].

    Load the policy with the recorded checkpoint/config first. Match the recorded runtime
    and backend settings for the closest numerical agreement. This never runs a robot or VLM.
    """
    device = torch.device(device or policy.config.device)
    with h5py.File(path, "r") as file:
        chunk = file[f"chunks/{chunk_index:06d}"]
        if not chunk.attrs["complete"] or not chunk.attrs["execution_ready"]:
            raise ValueError("This chunk was interrupted before its execution decision was recorded")
        outputs = []
        for key in sorted(chunk["sampling"]):
            call = chunk["sampling"][key]
            output = replay_sampling_call(policy.model, call, device).cpu()
            torch.testing.assert_close(output, torch.from_numpy(call["full_output"][()]), rtol=rtol, atol=atol)
            outputs.append(output)
        horizon, action_dim = chunk["original_actions"].shape[1:]
        original = outputs[0][:, :horizon, :action_dim]
        perturbed = original.clone()
        perturbed[:, 1:, :3] += torch.from_numpy(chunk["perturbation_noise"][()]) * chunk.attrs["perturb_std"]
        selected = perturbed[torch.from_numpy(chunk["selected_indices"][()])]
        for name, value in (("original_actions", original), ("perturbed_actions", perturbed), ("selected_actions", selected)):
            torch.testing.assert_close(value, torch.from_numpy(chunk[name][()]), rtol=rtol, atol=atol)
        source = chunk.attrs["execution_source"]
        if source == "original_sample_0":
            queued = original[:1]
        elif source == "selected_sample_0":
            queued = selected[:1]
        elif source == "last_sampling_call":
            queued = outputs[-1][:, :horizon, :action_dim]
        else:
            raise ValueError(f"Unknown execution source: {source}")
        # Candidate recordings may retain more steps than were queued for execution.
        queued = queued[:, :chunk["queued_actions"].shape[1]]
        torch.testing.assert_close(queued, torch.from_numpy(chunk["queued_actions"][()]), rtol=rtol, atol=atol)
        return queued


def load_recorded_processor(path, name="postprocessor"):
    """Restore captured normalization/tokenization configuration and tensor state."""
    from lerobot.processor import PolicyProcessorPipeline
    from lerobot.processor.converters import policy_action_to_transition, transition_to_policy_action

    with h5py.File(path, "r") as file, tempfile.TemporaryDirectory() as directory:
        for filename, data in file[f"artifacts/{name}"].items():
            if Path(filename).name != filename:
                raise ValueError("Invalid processor artifact filename")
            (Path(directory) / filename).write_bytes(data[()].tobytes())
        converters = {}
        if name == "postprocessor":
            converters = dict(to_transition=policy_action_to_transition, to_output=transition_to_policy_action)
        return PolicyProcessorPipeline.from_pretrained(directory, config_filename="processor.json", **converters)
