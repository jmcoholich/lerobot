import json

import numpy as np
import pytest
import torch
from PIL import Image

from lerobot.policies.pi05.robomonkey_timing import RoboMonkeyTimingRecorder
from lerobot.policies.pi05.robomonkey_utils import (
    RoboMonkeyVerifierError,
    extract_task_description,
    get_rewards,
    get_robomonkey_action,
    rollout_batch_limit_reached,
)


class _Response:
    def __init__(self, payload, status_error=None):
        self.payload = payload
        self.status_error = status_error

    def raise_for_status(self):
        if self.status_error is not None:
            raise self.status_error

    def json(self):
        return self.payload


def test_task_extraction_preserves_multi_clause_instruction():
    prompt = "Task: place the pink block, then the blue block in the bin, State: 1 2 3"
    assert extract_task_description(prompt) == "place the pink block, then the blue block in the bin"
    assert extract_task_description("place the coffee pod in the bin") == "place the coffee pod in the bin"


def test_final_sampled_batch_executes_all_cached_actions():
    assert not rollout_batch_limit_reached(8, 8, cached_actions_remaining=True)
    assert rollout_batch_limit_reached(8, 8, cached_actions_remaining=False)
    assert rollout_batch_limit_reached(9, 8, cached_actions_remaining=False)


def test_verifier_request_uses_timeout_and_validates_rewards(monkeypatch):
    call = {}

    def post(url, **kwargs):
        call.update(url=url, **kwargs)
        return _Response({"rewards": [0.2, 0.8]})

    monkeypatch.setattr("lerobot.policies.pi05.robomonkey_utils.requests.post", post)
    rewards, metrics = get_rewards(
        "task",
        Image.new("RGB", (16, 16)),
        np.zeros((2, 7)),
        "http://verifier:3100/",
        timeout_s=12.5,
    )

    assert rewards == [0.2, 0.8]
    assert call["url"] == "http://verifier:3100/process_with_image_directly"
    assert call["timeout"] == 12.5
    assert json.loads(call["data"])["instruction"] == "task"
    assert metrics["verifier_requests"][0]["status"] == "ok"


def test_invalid_verifier_rewards_are_an_explicit_failure(monkeypatch):
    monkeypatch.setattr(
        "lerobot.policies.pi05.robomonkey_utils.requests.post",
        lambda *args, **kwargs: _Response({"rewards": []}),
    )

    with pytest.raises(RoboMonkeyVerifierError) as exc_info:
        get_rewards(
            "task",
            Image.new("RGB", (16, 16)),
            np.zeros((2, 7)),
            "http://verifier:3100",
        )

    assert exc_info.value.metrics["verifier_requests"][0]["status"] == "failed"


def test_robomonkey_keeps_five_policy_samples_and_scores_eight_candidates(monkeypatch):
    scored = {}

    def rewards(instruction, image, actions, verifier_url, timeout_s):
        scored["shape"] = actions.shape
        return list(range(len(actions))), {
            "verifier_roundtrip_s": 0.1,
            "verifier_requests": [{"status": "ok", "roundtrip_s": 0.1}],
        }

    monkeypatch.setattr("lerobot.policies.pi05.robomonkey_utils.get_rewards", rewards)
    selected, metrics = get_robomonkey_action(
        torch.zeros((5, 1, 7)),
        "task",
        np.zeros((256, 256, 3), dtype=np.uint8),
        "http://verifier:3100",
        "unused",
        n_augmented=8,
    )

    assert selected.shape == (7,)
    assert scored["shape"] == (8, 7)
    assert metrics["num_policy_samples"] == 5
    assert metrics["num_augmented_candidates"] == 8
    assert metrics["selected_candidate_index"] == 7


def test_timing_recorder_writes_action_failure_and_summary(tmp_path):
    recorder = RoboMonkeyTimingRecorder("trial", process_start=0.0, output_root=tmp_path)
    recorder.begin_loop()
    recorder.write_action(
        {
            "trajectory_batch_id": 0,
            "verifier_request_id": 0,
            "fresh_policy_sampling": True,
            "policy_sampling_s": 1.0,
            "verifier_roundtrip_s": 0.5,
            "control_loop_total_s": 2.0,
            "verifier_requests": [{"status": "ok"}],
        }
    )
    recorder.write_failure(
        {
            "trajectory_batch_id": 1,
            "verifier_request_id": 1,
            "verifier_requests": [{"status": "failed"}],
            "error": "timeout",
        }
    )
    recorder.mark_rollout_end()
    summary = recorder.finalize(cleanup_s=0.25, end_reason="error", error="timeout")

    records = [json.loads(line) for line in recorder.actions_path.read_text().splitlines()]
    assert [record["record_type"] for record in records] == ["action", "failure"]
    assert summary["executed_actions"] == 1
    assert summary["failed_interventions"] == 1
    assert summary["fresh_policy_batches"] == 1
    assert summary["verifier_calls"] == 2
    assert summary["distributions"]["policy_sampling_s"]["mean_s"] == 1.0
