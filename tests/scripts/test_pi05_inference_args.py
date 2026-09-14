"""Validate launcher arguments without deleting the cache or starting inference."""

from pathlib import Path
import os
import subprocess
import unittest


SCRIPT = Path(__file__).resolve().parents[2] / "pi_05_inference.bash"


class TestPi05InferenceArgs(unittest.TestCase):
    def test_invalid_request_mode_is_rejected(self):
        result = subprocess.run(
            ["/bin/bash", str(SCRIPT), "trial", "ensemble", "invalid"], capture_output=True, text=True,
        )
        self.assertNotEqual(result.returncode, 0)
        self.assertIn("parallel or serial", result.stderr)

    def test_record_name_and_default_are_forwarded(self):
        for args, expected in ((["asdf"], "asdf"), ([""], "last_recording"), (["run with spaces"], "run with spaces")):
            with self.subTest(args=args):
                result = subprocess.run(
                    [
                        "/bin/bash", "-c",
                        'rm() { :; }; python() { printf "%s\\n" "$@"; }; inference_script="$1"; shift; source "$inference_script" "$@"',
                        "test_inference", str(SCRIPT), *args,
                    ],
                    check=True, capture_output=True, text=True,
                    env=os.environ | {"LEROBOT_POLICY_SERVER": "/tmp/test-pi05.sock", "LEROBOT_MMD_GAMMA": "0.002",
                                      "LEROBOT_DIVERSITY_GAMMA": "0.05"},
                )
                self.assertIn(f"--robot.record={expected}", result.stdout.splitlines())
                self.assertIn("--policy_server=/tmp/test-pi05.sock", result.stdout.splitlines())
                self.assertIn("src/lerobot/scripts/pi05_inference.py", result.stdout.splitlines())
                self.assertIn("--interventions=none", result.stdout.splitlines())
                self.assertIn("--ensemble_request_mode=serial", result.stdout.splitlines())
                self.assertIn("--manual_guidance=false", result.stdout.splitlines())
                self.assertIn("--vis_spreads=false", result.stdout.splitlines())
                self.assertIn("--policy.mmd_gamma=0.002", result.stdout.splitlines())
                self.assertIn("--policy.diversity_gamma=0.05", result.stdout.splitlines())
                self.assertFalse(any(arg.startswith("--dataset.") for arg in result.stdout.splitlines()))

    def test_ensemble_request_mode_argument_is_forwarded(self):
        for strategy, mode in (("ensemble", "serial"), ("eve", "parallel"), ("eve", "sequential")):
            with self.subTest(strategy=strategy, mode=mode):
                result = subprocess.run(
                    ["/bin/bash", "-c",
                     'python() { printf "%s\\n" "$@"; }; inference_script="$1"; shift; source "$inference_script" "$@"',
                     "test_inference", str(SCRIPT), "trial", strategy, mode],
                    check=True, capture_output=True, text=True,
                )
                self.assertIn("--interventions=ensemble", result.stdout.splitlines())
                expected_mode = "serial" if mode == "sequential" else mode
                self.assertIn(f"--ensemble_request_mode={expected_mode}", result.stdout.splitlines())


if __name__ == "__main__":
    unittest.main()
