"""Deterministic client timings persisted into a real trajectory HDF5."""

import importlib.util
from pathlib import Path
import tempfile
import unittest
from unittest.mock import Mock, patch

import h5py
import numpy as np

from lerobot.scripts import pi05_timing

spec = importlib.util.spec_from_file_location(
    'timing_recorder', Path(__file__).resolve().parents[2] / 'src/lerobot/policies/pi05/trajectory_recorder.py')
recorder_module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(recorder_module)


class TestRolloutTiming(unittest.TestCase):
    def setUp(self):
        self.directory = tempfile.TemporaryDirectory()
        self.addCleanup(self.directory.cleanup)
        self.now = 0.
        self.perf = patch.object(pi05_timing.time, 'perf_counter', side_effect=lambda: self.now)
        self.wall = patch.object(pi05_timing.time, 'time', side_effect=lambda: 1000 + self.now)
        self.perf.start()
        self.wall.start()
        self.addCleanup(self.perf.stop)
        self.addCleanup(self.wall.stop)
        self.recorder = recorder_module.TrajectoryRecorder({}, self.directory.name, demo_name='timing')
        actions = np.zeros((2, 2, 8))
        self.recorder.append(chunk_index=0, original=actions, perturbed=actions,
                             selected_indices=[0], observations={}, task='', perturb_std=0.)
        self.recorder.record_execution(0, sampling_calls=[], actions=actions[:1], source='original_sample_0',
                                       timings={'base_inference_s': .2, 'intervention_s': 0.})
        self.client = Mock(connection_usable=True)
        self.client.request.side_effect = lambda operation, **kwargs: self.recorder.record_timing(**kwargs)
        self.timer = pi05_timing.RolloutTiming(self.client)

    def first_step(self):
        self.timer.begin_step()
        self.now = 2.
        self.timer.prediction_done({'record_index': 0}, observation_s=.5, inference_rpc_s=1.5)
        start = self.timer.begin_command()
        self.now = 2.25
        self.timer.command_done(start)
        self.now = 2.5
        self.timer.end_step(False)

    def test_completed_chunk_and_rollout_have_distinct_durations(self):
        self.first_step()
        self.timer.begin_step()
        self.now = 3.
        self.timer.prediction_done({'record_index': 0}, observation_s=.4, inference_rpc_s=.1)
        start = self.timer.begin_command()
        self.now = 3.25
        self.timer.command_done(start)
        self.now = 3.5
        self.timer.end_step(True)
        with h5py.File(self.recorder.path, 'r') as f:
            self.assertFalse(f['timing/finalized'][()])
        self.now = 4.
        self.timer.report(finalized=True, end_reason='duration_limit')
        with h5py.File(self.recorder.path, 'r') as f:
            t = f['chunks/000000/timing']
            self.assertEqual(t['chunk_total_s'][()], 3.5)
            self.assertEqual(t['execution_s'][()], 1.5)
            self.assertEqual(t['command_send_s'][()], .5)
            self.assertAlmostEqual(t['inference_rpc_s'][()], 1.6)
            self.assertAlmostEqual(t['observation_s'][()], .9)
            self.assertEqual(t['base_inference_s'][()], .2)
            self.assertEqual(t['intervention_s'][()], 0.)
            self.assertEqual(t['executed_actions'][()], 2)
            self.assertTrue(t['execution_complete'][()])
            self.assertTrue(f['timing/finalized'][()])
            self.assertEqual(f['timing/elapsed_s'][()], 4.)
            self.assertEqual(f['timing/completed_chunks'][()], 1)
            self.assertEqual(f['timing'].attrs['end_reason'], 'duration_limit')

    def test_partial_chunk_is_saved_on_interrupt(self):
        self.first_step()
        self.timer.report(finalized=True, end_reason='KeyboardInterrupt')
        with h5py.File(self.recorder.path, 'r') as f:
            self.assertEqual(f['chunks/000000/timing/executed_actions'][()], 1)
            self.assertFalse(f['chunks/000000/timing/execution_complete'][()])
            self.assertEqual(f['timing/completed_chunks'][()], 0)
            self.assertEqual(f['timing'].attrs['end_reason'], 'KeyboardInterrupt')

    def test_interrupted_connection_does_not_send_another_request(self):
        self.first_step()
        self.client.connection_usable = False
        with self.assertLogs(level='WARNING'):
            self.timer.report(finalized=True, end_reason='KeyboardInterrupt')
        self.client.request.assert_not_called()


if __name__ == '__main__':
    unittest.main()
