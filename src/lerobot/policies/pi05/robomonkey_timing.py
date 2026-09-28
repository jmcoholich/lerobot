"""Small JSON timing recorder for RoboMonkey evaluation rollouts."""

import json
import math
import os
import time
from pathlib import Path


def _distribution(values):
    values = sorted(value for value in values if isinstance(value, (int, float)) and math.isfinite(value))
    if not values:
        return None
    p95_index = math.ceil(0.95 * len(values)) - 1
    middle = len(values) // 2
    median = values[middle] if len(values) % 2 else (values[middle - 1] + values[middle]) / 2
    return {
        "count": len(values),
        "total_s": sum(values),
        "mean_s": sum(values) / len(values),
        "median_s": median,
        "p95_s": values[p95_index],
        "max_s": values[-1],
    }


class RoboMonkeyTimingRecorder:
    """Write action-level timing immediately and a compact summary at shutdown."""

    def __init__(self, record_name, process_start, output_root=None):
        if record_name in (".", "..") or Path(record_name).name != record_name:
            raise ValueError("RoboMonkey record name must be a single directory name")
        root = Path(
            output_root or os.environ.get("ROBOMONKEY_TIMING_ROOT", "/home/jeremiah/openteach/extracted_data")
        )
        self.output_dir = root / f"demonstration_{record_name}"
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.actions_path = self.output_dir / "robomonkey_timing.jsonl"
        self.summary_path = self.output_dir / "robomonkey_timing_summary.json"
        self._stream = self.actions_path.open("w", encoding="utf-8", buffering=1)
        self.process_start = process_start
        self.rollout_start = None
        self.rollout_end = None
        self.records = []
        self.failures = []

    def begin_loop(self):
        now = time.perf_counter()
        if self.rollout_start is None:
            self.rollout_start = now
        return now

    def write_action(self, record):
        record = {"record_type": "action", "action_index": len(self.records), **record}
        self.records.append(record)
        self._stream.write(json.dumps(record, sort_keys=True) + "\n")

    def write_failure(self, record):
        record = {"record_type": "failure", "failure_index": len(self.failures), **record}
        self.failures.append(record)
        self._stream.write(json.dumps(record, sort_keys=True) + "\n")

    def mark_rollout_end(self):
        if self.rollout_end is None:
            self.rollout_end = time.perf_counter()

    def finalize(self, cleanup_s, end_reason, error=None):
        if self.rollout_end is None:
            self.mark_rollout_end()
        fields = (
            "observation_acquisition_s",
            "observation_processing_s",
            "policy_inference_s",
            "policy_sampling_s",
            "image_preprocess_s",
            "candidate_augmentation_s",
            "selection_s",
            "action_conversion_s",
            "robomonkey_intervention_s",
            "verifier_roundtrip_s",
            "command_submission_s",
            "dataset_recording_s",
            "control_loop_active_s",
            "control_loop_sleep_s",
            "control_loop_total_s",
        )
        verifier_requests = [
            request
            for record in self.records + self.failures
            for request in record.get("verifier_requests", [])
        ]
        summary = {
            "record_type": "summary",
            "status": "failed" if error else "completed",
            "end_reason": end_reason,
            "error": error,
            "startup_s": (self.rollout_start - self.process_start)
            if self.rollout_start is not None
            else None,
            "overall_rollout_s": (
                self.rollout_end - self.rollout_start if self.rollout_start is not None else 0.0
            ),
            "cleanup_s": cleanup_s,
            "executed_actions": len(self.records),
            "failed_interventions": len(self.failures),
            "fresh_policy_batches": sum(bool(r.get("fresh_policy_sampling")) for r in self.records),
            "verifier_calls": len(verifier_requests),
            "successful_verifier_calls": sum(request.get("status") == "ok" for request in verifier_requests),
            "failed_verifier_calls": sum(request.get("status") == "failed" for request in verifier_requests),
            "distributions": {
                field: _distribution([record.get(field) for record in self.records]) for field in fields
            },
        }
        self.summary_path.write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8")
        self._stream.close()
        return summary
