"""Exercise launcher shutdown with stub children; never start inference or hardware."""

import os
import json
from pathlib import Path
import signal
import subprocess
import tempfile
import time
import unittest


LAUNCHER = Path(__file__).resolve().parents[2] / "collect_eval.bash"
STUB = '''#!/usr/bin/python3
import os
import json
from pathlib import Path
import signal
import subprocess
import sys
import time

signal.signal(signal.SIGINT, signal.default_int_handler)
role = "worker" if sys.argv[1] == "worker" else ("data" if sys.argv[1] == "-c" else "inference")
directory = Path(os.environ["STUB_DIR"])
(directory / (role + ".pid")).write_text(str(os.getpid()))
(directory / (role + ".args")).write_text(json.dumps(sys.argv[1:]))
worker = None
try:
    if role == "data":
        worker = subprocess.Popen([sys.executable, __file__, "worker"])
    (directory / (role + ".ready")).touch()
    if role == os.environ.get("EXIT_ROLE"):
        while not all((directory / (name + ".ready")).exists() for name in ("data", "inference", "worker")):
            time.sleep(0.01)
        time.sleep(0.05)
        sys.exit(int(os.environ["EXIT_STATUS"]))
    while True:
        signal.pause()
except KeyboardInterrupt:
    (directory / (role + ".sigint")).touch()
    time.sleep(0.1)  # Simulate asynchronous recorder saving.
    if worker is not None:
        worker.wait(timeout=5)
    (directory / (role + ".saved")).touch()
'''


class TestCollectEval(unittest.TestCase):
    def run_launcher(self, exit_role="", exit_status=0, interrupt=False, mode=None):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            stub = root / "bash"
            stub.write_text(STUB)
            stub.chmod(0o755)
            env = {
                **os.environ,
                "PATH": f"{root}:{os.environ['PATH']}",
                "STUB_DIR": str(root),
                "EXIT_ROLE": exit_role,
                "EXIT_STATUS": str(exit_status),
            }
            # Absolute /bin/bash runs the launcher; its child bash calls use our stub.
            process = subprocess.Popen(
                ["/bin/bash", str(LAUNCHER), "test_shutdown", "eve"] + ([mode] if mode else []),
                cwd=root,
                env=env,
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                text=True,
                start_new_session=True,
            )
            try:
                if interrupt:
                    deadline = time.monotonic() + 5
                    while not all((root / f"{role}.ready").exists() for role in ("data", "inference", "worker")):
                        self.assertLess(time.monotonic(), deadline, "Stub processes did not start")
                        time.sleep(0.01)
                    process.send_signal(signal.SIGINT)
                output, _ = process.communicate(timeout=10)
                self.assertEqual(process.returncode, 130 if interrupt else exit_status, output)
                expected = ["pi_05_inference.bash", "test_shutdown", "eve"]
                if mode:
                    expected.append("serial" if mode == "sequential" else mode)
                self.assertEqual(json.loads((root / "inference.args").read_text()), expected)
                for role in ("data", "inference", "worker"):
                    if role != exit_role:
                        self.assertTrue((root / f"{role}.sigint").exists(), f"{role} missed SIGINT: {output}")
                        self.assertTrue((root / f"{role}.saved").exists(), f"{role} did not finish saving: {output}")
            finally:
                if process.poll() is None:
                    for pid_file in root.glob("*.pid"):
                        try:
                            os.kill(int(pid_file.read_text()), signal.SIGKILL)
                        except ProcessLookupError:
                            pass
                    process.kill()
                    process.communicate(timeout=5)

    def test_inference_completion_stops_collection(self):
        self.run_launcher(exit_role="inference")

    def test_inference_failure_preserves_status(self):
        self.run_launcher(exit_role="inference", exit_status=7)

    def test_collection_failure_stops_inference(self):
        self.run_launcher(exit_role="data", exit_status=9)

    def test_ctrl_c_waits_for_both_process_groups(self):
        self.run_launcher(interrupt=True)

    def test_request_mode_is_forwarded(self):
        for mode in ("serial", "parallel", "sequential"):
            with self.subTest(mode=mode):
                self.run_launcher(exit_role="inference", mode=mode)

    def test_invalid_mode_fails_before_starting_children(self):
        result = subprocess.run(
            ["/bin/bash", str(LAUNCHER), "test", "eve", "invalid"],
            capture_output=True, text=True,
        )
        self.assertEqual(result.returncode, 2)
        self.assertIn("parallel or serial", result.stderr)
        self.assertNotIn("Starting", result.stdout)


if __name__ == "__main__":
    unittest.main()
