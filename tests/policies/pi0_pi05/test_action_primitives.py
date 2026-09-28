"""Check primitive displacement without connecting to a robot."""

from types import SimpleNamespace
import unittest

import torch

from lerobot.policies.pi05 import action_primitives as primitives


class TestPrimitiveDistance(unittest.TestCase):
    def test_down_has_half_the_upward_displacement(self):
        robot = SimpleNamespace(debug=True)
        # Identity normalization lets us inspect the returned Cartesian positions.
        postprocessor = SimpleNamespace(steps=[SimpleNamespace(stats={
            "action": {"min": -torch.ones(8), "max": torch.ones(8)},
        })])
        down = primitives.get_guidance_action_from_text("down", postprocessor, robot)
        up = primitives.get_guidance_action_from_text("up", postprocessor, robot)
        self.assertEqual(down.shape, (1, 100, 8))
        down_delta = down[0, -1, :3] - down[0, 0, :3]
        up_delta = up[0, -1, :3] - up[0, 0, :3]
        # Samples run from 0 to 99: the final point is 99/100 of the setting.
        torch.testing.assert_close(down_delta, torch.tensor([0.0, 0.0, -0.0495]), atol=1e-6, rtol=0)
        torch.testing.assert_close(down_delta, -up_delta / 2, atol=1e-6, rtol=0)
        self.assertTrue(torch.all(torch.diff(down[0, :, 2]) <= 0))
        torch.testing.assert_close(down[0, :, 3:], up[0, :, 3:])

    def test_horizontal_primitives_retain_their_distance(self):
        robot = SimpleNamespace(debug=True)
        for label in ("left", "right", "forward", "backward"):
            with self.subTest(label=label):
                chunk = primitives.create_cartesian_chunk(primitives.LABEL2ACTION[label], robot)
                delta = chunk[0, -1, :3] - chunk[0, 0, :3]
                self.assertAlmostEqual(torch.linalg.vector_norm(delta).item(), 0.099, places=6)
                self.assertEqual(delta[2].item(), 0.0)
