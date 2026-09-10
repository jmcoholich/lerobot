"""Validate launcher arguments without deleting the cache or starting inference."""

from pathlib import Path
import subprocess
import unittest


SCRIPT = Path(__file__).resolve().parents[2] / "pi_05_inference.bash"


class TestPi05InferenceArgs(unittest.TestCase):
    def test_record_name_and_default_are_forwarded(self):
        for args, expected in ((["asdf"], "asdf"), ([], "last_recording"), (["run with spaces"], "run with spaces")):
            with self.subTest(args=args):
                result = subprocess.run(
                    [
                        "/bin/bash", "-c",
                        'rm() { :; }; python() { printf "%s\\n" "$@"; }; inference_script="$1"; shift; source "$inference_script" "$@"',
                        "test_inference", str(SCRIPT), *args,
                    ],
                    check=True, capture_output=True, text=True,
                )
                self.assertIn(f"--robot.record={expected}", result.stdout.splitlines())


if __name__ == "__main__":
    unittest.main()
