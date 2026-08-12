import io
import json
import tempfile
import threading
import unittest
from pathlib import Path
from urllib.request import Request, urlopen

import av
import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
from PIL import Image

from external_utils.add_annotations_to_lerobot_dataset import add_annotation_column
from external_utils.annotation_tool import AnnotationHTTPServer, AnnotationService


def make_dataset(root: Path) -> Path:
    info = {
        "fps": 10,
        "video_path": "videos/{video_key}/chunk-{chunk_index:03d}/file-{file_index:03d}.mp4",
        "features": {
            "timestamp": {"dtype": "float32", "shape": [1], "names": None},
            "frame_index": {"dtype": "int64", "shape": [1], "names": None},
            "episode_index": {"dtype": "int64", "shape": [1], "names": None},
            "observation.images.main": {"dtype": "video", "shape": [3, 32, 32], "names": None},
        },
    }
    (root / "meta").mkdir(parents=True)
    (root / "meta" / "info.json").write_text(json.dumps(info), encoding="utf-8")

    data_path = root / "data" / "chunk-000" / "file-000.parquet"
    data_path.parent.mkdir(parents=True)
    pq.write_table(
        pa.table(
            {
                "episode_index": pa.array([0, 0, 1], type=pa.int64()),
                "frame_index": pa.array([0, 1, 0], type=pa.int64()),
                "timestamp": pa.array([0.0, 0.1, 0.0], type=pa.float32()),
                "fname": ["episode_zero.h5", "episode_zero.h5", "episode_one.h5"],
                "sparse_reward": pa.array([0.0, 0.5, 1.0], type=pa.float32()),
                "action": pa.array([[0.0], [1.0], [2.0]]),
            }
        ),
        data_path,
    )
    return data_path


def make_video(root: Path) -> None:
    episodes_path = root / "meta" / "episodes" / "chunk-000" / "file-000.parquet"
    episodes_path.parent.mkdir(parents=True)
    pq.write_table(
        pa.table(
            {
                "episode_index": [0, 1],
                "videos/observation.images.main/chunk_index": [0, 0],
                "videos/observation.images.main/file_index": [0, 0],
                "videos/observation.images.main/from_timestamp": [0.0, 0.0],
            }
        ),
        episodes_path,
    )

    video_path = root / "videos" / "observation.images.main" / "chunk-000" / "file-000.mp4"
    video_path.parent.mkdir(parents=True)
    with av.open(video_path, mode="w") as container:
        stream = container.add_stream("libx264", rate=10)
        stream.width = 32
        stream.height = 24
        stream.pix_fmt = "yuv420p"
        for color in ((210, 40, 30), (20, 170, 90)):
            pixels = np.zeros((24, 32, 3), dtype=np.uint8)
            pixels[:, :] = color
            frame = av.VideoFrame.from_ndarray(pixels, format="rgb24")
            for packet in stream.encode(frame):
                container.mux(packet)
        for packet in stream.encode():
            container.mux(packet)


class FrameAnnotationToolTest(unittest.TestCase):
    def setUp(self) -> None:
        self.temporary_directory = tempfile.TemporaryDirectory()
        self.root = Path(self.temporary_directory.name)

    def tearDown(self) -> None:
        self.temporary_directory.cleanup()

    def test_add_annotation_column_is_idempotent(self) -> None:
        data_path = make_dataset(self.root)

        self.assertEqual(add_annotation_column(self.root), (1, 1))
        self.assertEqual(add_annotation_column(self.root), (0, 1))

        table = pq.read_table(data_path)
        self.assertEqual(table["annotation"].to_pylist(), ["", "", ""])
        info = json.loads((self.root / "meta" / "info.json").read_text(encoding="utf-8"))
        self.assertEqual(info["features"]["annotation"], {"dtype": "string", "shape": [1], "names": None})

    def test_drafts_commit_to_the_annotation_column(self) -> None:
        data_path = make_dataset(self.root)
        add_annotation_column(self.root)
        service = AnnotationService(self.root, "annotation", ["success", "failure"], 640)

        try:
            initial = service.state(0)
            self.assertEqual([frame["frame_index"] for frame in initial["frames"]], [0, 1])
            self.assertEqual(initial["progress"]["total_annotated"], 0)
            self.assertEqual(
                initial["episode_summary"],
                {
                    "fname": "episode_zero.h5",
                    "outcome": "partial success",
                    "outcome_value": 0.5,
                    "outcome_field": "sparse_reward",
                    "annotation_count": 0,
                },
            )
            self.assertEqual(initial["episode_summaries"]["1"]["outcome"], "success")
            self.assertEqual(initial["labels"], ["success", "failure"])

            service.annotate(0, 1, "plug picked")
            service.annotate(0, 1, "partial success")
            draft_state = service.state(0)
            self.assertEqual(draft_state["frames"][1]["annotation"], "partial success")
            self.assertTrue(draft_state["frames"][1]["pending"])
            self.assertEqual(draft_state["pending_count"], 1)
            self.assertEqual(draft_state["labels"], ["success", "failure", "plug picked", "partial success"])
            self.assertEqual(draft_state["episode_summary"]["annotation_count"], 1)

            service.annotate(0, 1, "success")
            self.assertEqual(
                service.state(0)["labels"], ["success", "failure", "plug picked", "partial success"]
            )
            reordered = ["partial success", "success", "plug picked", "failure"]
            self.assertEqual(service.reorder_labels(reordered), {"labels": reordered})
            self.assertEqual(service.state(0)["labels"], reordered)
            service.annotate(0, 1, "partial success")
            self.assertEqual(service.state(0)["labels"], reordered)

            self.assertEqual(service.commit(), {"committed": 1, "files": 1, "pending_count": 0})
            self.assertEqual(pq.read_table(data_path)["annotation"].to_pylist(), ["", "partial success", ""])
            committed_state = service.state(0)
            self.assertFalse(committed_state["frames"][1]["pending"])
            self.assertEqual(committed_state["episode_summary"]["annotation_count"], 1)

            service.annotate(0, 1, "failure")
            service.annotate(1, 0, "success")
            self.assertEqual(service.discard(), {"discarded": 2, "pending_count": 0})
            discarded_state = service.state(0)
            self.assertEqual(discarded_state["frames"][1]["annotation"], "partial success")
            self.assertFalse(discarded_state["frames"][1]["pending"])
            self.assertEqual(discarded_state["episode_summary"]["annotation_count"], 1)
            self.assertEqual(discarded_state["episode_summaries"]["1"]["annotation_count"], 0)
            self.assertEqual(pq.read_table(data_path)["annotation"].to_pylist(), ["", "partial success", ""])
        finally:
            service.close()

    def test_reward_is_used_when_sparse_reward_is_absent(self) -> None:
        data_path = make_dataset(self.root)
        table = pq.read_table(data_path)
        sparse_index = table.schema.get_field_index("sparse_reward")
        reward_values = table["sparse_reward"]
        table = table.remove_column(sparse_index).append_column("reward", reward_values)
        pq.write_table(table, data_path)
        add_annotation_column(self.root)
        service = AnnotationService(self.root, "annotation", [], 640)

        try:
            summary = service.state(0)["episode_summary"]
            self.assertEqual(summary["outcome_field"], "reward")
            self.assertEqual(summary["outcome"], "partial success")
        finally:
            service.close()

    def test_incompatible_metadata_is_rejected_before_data_changes(self) -> None:
        data_path = make_dataset(self.root)
        info_path = self.root / "meta" / "info.json"
        info = json.loads(info_path.read_text(encoding="utf-8"))
        info["features"]["annotation"] = {"dtype": "float32", "shape": [1], "names": None}
        info_path.write_text(json.dumps(info), encoding="utf-8")

        with self.assertRaisesRegex(ValueError, "expected 'string'"):
            add_annotation_column(self.root)

        self.assertNotIn("annotation", pq.read_schema(data_path).names)

    def test_frame_decoding_and_http_server(self) -> None:
        make_dataset(self.root)
        make_video(self.root)
        add_annotation_column(self.root)
        service = AnnotationService(self.root, "annotation", ["success"], 640)
        server = AnnotationHTTPServer(("127.0.0.1", 0), service)
        thread = threading.Thread(target=server.serve_forever, daemon=True)
        thread.start()

        try:
            jpeg = service.frame_jpeg(0, 1, "observation.images.main")
            with Image.open(io.BytesIO(jpeg)) as image:
                self.assertEqual(image.size, (32, 24))
            combined_jpeg = service.combined_frame_jpeg(
                0, 1, "observation.images.main", "observation.images.main"
            )
            with Image.open(io.BytesIO(combined_jpeg)) as image:
                self.assertEqual(image.size, (48, 24))

            host, port = server.server_address
            with urlopen(f"http://{host}:{port}/api/state", timeout=5) as response:
                state = json.load(response)
            self.assertEqual(state["episode_index"], 0)
            self.assertEqual(len(state["frames"]), 2)
            self.assertEqual(state["episode_summary"]["fname"], "episode_zero.h5")
            self.assertEqual(state["episode_summary"]["outcome"], "partial success")

            service.annotate(0, 0, "temporary")
            with urlopen(Request(f"http://{host}:{port}/api/discard", method="POST"), timeout=5) as response:
                discard_result = json.load(response)
            self.assertEqual(discard_result, {"discarded": 1, "pending_count": 0})

            with urlopen(f"http://{host}:{port}/", timeout=5) as response:
                html = response.read().decode("utf-8")
            self.assertIn("Frame Annotations", html)
            self.assertIn("Recent annotations", html)
            self.assertIn("Discard pending", html)
            self.assertIn("annotation-marks", html)
            self.assertIn('id="skip-ten-back"', html)
            self.assertIn('title="Advance one frame without annotating">+1', html)
            self.assertIn('id="episode-fname"', html)
            self.assertIn('id="episode-outcome"', html)
            self.assertIn('id="camera-buttons"', html)
            self.assertNotIn('id="camera-select"', html)
            self.assertNotIn("auto-advance", html)

            with urlopen(f"http://{host}:{port}/styles.css", timeout=5) as response:
                css = response.read().decode("utf-8")
            self.assertIn("[hidden]", css)
            self.assertIn("display: none !important", css)
            self.assertIn("height: 100dvh", css)
            self.assertIn("overflow-y: auto", css)

            with urlopen(f"http://{host}:{port}/app.js", timeout=5) as response:
                javascript = response.read().decode("utf-8")
            self.assertIn("loadFrameImage", javascript)
            self.assertIn("response.ok", javascript)
            self.assertIn("renderCameraButtons", javascript)
            self.assertIn("Front + Side", javascript)
            self.assertIn("/api/combined-frame", javascript)
            self.assertNotIn("autoAdvance", javascript)
            self.assertIn("annotation_count", javascript)
            self.assertIn("loadingTimer", javascript)
            self.assertIn("renderChain", javascript)
            self.assertIn("moveFrame", javascript)
            self.assertIn("requestAnimationFrame", javascript)
            self.assertNotIn("AbortController", javascript)
            self.assertNotIn("frameLoadTimer", javascript)
            self.assertNotIn("const prefetch = new Image()", javascript)
        finally:
            server.shutdown()
            thread.join(timeout=5)
            server.server_close()
            service.close()


if __name__ == "__main__":
    unittest.main()
