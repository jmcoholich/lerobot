"""Client-side rollout and chunk timings sent to the policy's trajectory recorder."""

import logging
import time


class RolloutTiming:
    def __init__(self, client):
        self.client = client
        self.start = time.perf_counter()
        self.started_at_unix = time.time()
        self.chunk = None
        self.executed_actions = 0
        self.completed_chunks = 0

    def begin_step(self):
        now = time.perf_counter()
        if self.chunk is None:
            self.chunk = {
                "start": now, "started_at_unix": time.time(), "command_start": None,
                "record_index": None, "executed_actions": 0, "execution_complete": False,
                "observation_s": 0.0, "inference_rpc_s": 0.0, "command_send_s": 0.0,
            }
        return now

    def prediction_done(self, info, observation_s, inference_rpc_s):
        self.chunk["record_index"] = info["record_index"]
        self.chunk["observation_s"] += observation_s
        self.chunk["inference_rpc_s"] += inference_rpc_s

    def begin_command(self):
        now = time.perf_counter()
        if self.chunk["command_start"] is None:
            self.chunk["command_start"] = now
            self.chunk["execution_started_at_unix"] = time.time()
        return now

    def command_done(self, start):
        self.chunk["command_send_s"] += time.perf_counter() - start
        self.chunk["executed_actions"] += 1
        self.executed_actions += 1

    def end_step(self, chunk_complete):
        if chunk_complete:
            self.chunk["execution_complete"] = True
            self.completed_chunks += 1
            self.report()
            self.chunk = None

    def report(self, *, finalized=False, end_reason=""):
        now = time.perf_counter()
        rollout = {
            "started_at_unix": self.started_at_unix,
            "elapsed_s": now - self.start,
            "last_updated_at_unix": time.time(),
            "executed_actions": self.executed_actions,
            "completed_chunks": self.completed_chunks,
            "finalized": finalized,
        }
        if finalized:
            rollout.update(ended_at_unix=time.time(), end_reason=end_reason)
        chunk = self.chunk
        chunk_timing = None
        record_index = None
        if chunk is not None and chunk["record_index"] is not None:
            record_index = chunk["record_index"]
            chunk_timing = {key: value for key, value in chunk.items()
                            if key not in ("start", "command_start", "record_index")}
            chunk_timing.update(
                chunk_total_s=now - chunk["start"],
                execution_s=now - chunk["command_start"] if chunk["command_start"] is not None else float("nan"),
                ended_at_unix=time.time(),
            )
        # Never send a second request while a previous interrupted receive is pending.
        if not self.client.connection_usable:
            logging.warning("Could not finalize trajectory timing: policy connection was interrupted")
            return
        try:
            self.client.request("record_timing", record_index=record_index,
                                chunk_timing=chunk_timing, rollout_timing=rollout)
        except Exception:
            logging.exception("Could not save trajectory execution timing")
