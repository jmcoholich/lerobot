#!/usr/bin/env python

"""Browser-based per-frame annotation tool for local or remote LeRobot datasets."""

from __future__ import annotations

import argparse
import concurrent.futures
import io
import json
import math
import signal
import sqlite3
import threading
import time
from collections import Counter, defaultdict
from dataclasses import dataclass
from functools import lru_cache
from http import HTTPStatus
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any
from urllib.parse import parse_qs, urlparse

import av
import pyarrow as pa
import pyarrow.parquet as pq
from PIL import Image

try:
    from .add_annotations_to_lerobot_dataset import add_annotation_column, load_json, write_table_atomic
except ImportError:
    from add_annotations_to_lerobot_dataset import add_annotation_column, load_json, write_table_atomic


STATIC_ROOT = Path(__file__).with_name("annotation_tool_static")
MAX_ANNOTATION_LENGTH = 200


@dataclass(frozen=True)
class FrameLocation:
    episode_index: int
    frame_index: int
    timestamp: float
    source_path: Path
    row_index: int


class DatasetCatalog:
    def __init__(self, root: Path, column: str) -> None:
        self.root = root
        self.column = column
        self.info = load_json(root / "meta" / "info.json")
        self.fps = float(self.info["fps"])
        self.video_keys = [
            key for key, feature in self.info.get("features", {}).items() if feature.get("dtype") == "video"
        ]
        if not self.video_keys:
            raise ValueError("The annotation tool currently requires at least one video feature")

        self.frames_by_episode: dict[int, list[FrameLocation]] = defaultdict(list)
        self.locations: dict[tuple[int, int], FrameLocation] = {}
        self.annotations: dict[tuple[int, int], str] = {}
        self.episode_fnames: dict[int, str] = {}
        self.episode_final_rewards: dict[int, float | None] = {}
        self.episode_final_frame_indices: dict[int, int] = {}
        self.outcome_field: str | None = None
        self.success_reward: float | None = None
        self.episode_metadata = self._load_episode_metadata()
        self._load_frames()

        self.episode_indices = sorted(self.frames_by_episode)
        self.total_frames = len(self.locations)
        self.committed_annotated_count = sum(bool(value) for value in self.annotations.values())
        self.committed_annotation_counts = Counter(
            episode for (episode, _frame), value in self.annotations.items() if value
        )
        self.label_counts = Counter(value for value in self.annotations.values() if value)

    def _load_frames(self) -> None:
        data_paths = sorted((self.root / "data").glob("**/*.parquet"))
        if not data_paths:
            raise FileNotFoundError(f"No Parquet files found under {self.root / 'data'}")

        required = {"episode_index", "frame_index", "timestamp", self.column}
        schemas = {path: set(pq.read_schema(path).names) for path in data_paths}
        if all("sparse_reward" in names for names in schemas.values()):
            self.outcome_field = "sparse_reward"
        elif all("reward" in names for names in schemas.values()):
            self.outcome_field = "reward"
        has_fname = all("fname" in names for names in schemas.values())

        for path, schema_names in schemas.items():
            missing = required - schema_names
            if missing:
                raise ValueError(f"{path} is missing required columns: {sorted(missing)}")

            columns = ["episode_index", "frame_index", "timestamp", self.column]
            if has_fname:
                columns.append("fname")
            if self.outcome_field:
                columns.append(self.outcome_field)
            table = pq.read_table(path, columns=columns)
            episode_values = table["episode_index"].to_pylist()
            frame_values = table["frame_index"].to_pylist()
            timestamps = table["timestamp"].to_pylist()
            annotations = table[self.column].to_pylist()
            fnames = table["fname"].to_pylist() if has_fname else [None] * table.num_rows
            rewards = table[self.outcome_field].to_pylist() if self.outcome_field else [None] * table.num_rows
            for row_index, (episode, frame, timestamp, annotation) in enumerate(
                zip(episode_values, frame_values, timestamps, annotations, strict=True)
            ):
                key = (int(episode), int(frame))
                if key in self.locations:
                    raise ValueError(f"Duplicate episode/frame key in dataset: {key}")
                location = FrameLocation(
                    episode_index=key[0],
                    frame_index=key[1],
                    timestamp=float(timestamp),
                    source_path=path,
                    row_index=row_index,
                )
                self.frames_by_episode[key[0]].append(location)
                self.locations[key] = location
                self.annotations[key] = "" if annotation is None else str(annotation)
                fname = fnames[row_index]
                if fname is not None and str(fname):
                    self.episode_fnames[key[0]] = str(fname)
                reward = rewards[row_index]
                previous_final_frame = self.episode_final_frame_indices.get(key[0], -1)
                if key[1] >= previous_final_frame:
                    self.episode_final_frame_indices[key[0]] = key[1]
                    self.episode_final_rewards[key[0]] = None if reward is None else float(reward)

        for frames in self.frames_by_episode.values():
            frames.sort(key=lambda item: item.frame_index)

        finite_positive_rewards = [
            value
            for value in self.episode_final_rewards.values()
            if value is not None and math.isfinite(value) and value > 0
        ]
        if finite_positive_rewards:
            self.success_reward = max(finite_positive_rewards)

    def episode_summary(self, episode_index: int) -> dict[str, Any]:
        reward = self.episode_final_rewards.get(episode_index)
        outcome = "unknown"
        if reward is not None and math.isfinite(reward):
            if reward <= 0:
                outcome = "failure"
            elif self.success_reward is not None and math.isclose(reward, self.success_reward):
                outcome = "success"
            else:
                outcome = "partial success"
        return {
            "fname": self.episode_fnames.get(episode_index),
            "outcome": outcome,
            "outcome_value": reward if reward is not None and math.isfinite(reward) else None,
            "outcome_field": self.outcome_field,
        }

    def _load_episode_metadata(self) -> dict[int, dict[str, Any]]:
        records: dict[int, dict[str, Any]] = {}
        episode_paths = sorted((self.root / "meta" / "episodes").glob("**/*.parquet"))
        for path in episode_paths:
            for row in pq.read_table(path).to_pylist():
                if "episode_index" in row:
                    records[int(row["episode_index"])] = row

        legacy_path = self.root / "meta" / "episodes.jsonl"
        if legacy_path.is_file():
            with legacy_path.open(encoding="utf-8") as file:
                for line in file:
                    if line.strip():
                        row = json.loads(line)
                        records[int(row["episode_index"])] = row
        return records

    def resolve_video(self, episode_index: int, video_key: str) -> tuple[Path, float]:
        if video_key not in self.video_keys:
            raise KeyError(f"Unknown video key: {video_key}")
        if episode_index not in self.frames_by_episode:
            raise KeyError(f"Unknown episode: {episode_index}")

        metadata = self.episode_metadata.get(episode_index, {})
        chunk_key = f"videos/{video_key}/chunk_index"
        file_key = f"videos/{video_key}/file_index"
        format_values = {
            "video_key": video_key,
            "episode_index": episode_index,
            "episode_chunk": episode_index // 1000,
            "chunk_index": int(metadata.get(chunk_key, episode_index // 1000)),
            "file_index": int(metadata.get(file_key, episode_index)),
        }
        template = self.info.get("video_path")
        if not template:
            raise ValueError("meta/info.json does not define video_path")
        relative_path = Path(template.format(**format_values))
        video_path = (self.root / relative_path).resolve()
        if not video_path.is_relative_to(self.root):
            raise ValueError("Resolved video path falls outside the dataset root")
        if not video_path.is_file():
            raise FileNotFoundError(f"Video file not found: {video_path}")

        start_time = float(metadata.get(f"videos/{video_key}/from_timestamp", 0.0))
        return video_path, start_time

    def apply_committed(self, records: list[tuple[int, int, str]]) -> None:
        for episode, frame, new_value in records:
            key = (episode, frame)
            old_value = self.annotations[key]
            if old_value == new_value:
                continue
            if old_value:
                self.label_counts[old_value] -= 1
                if self.label_counts[old_value] <= 0:
                    del self.label_counts[old_value]
                self.committed_annotated_count -= 1
                self.committed_annotation_counts[episode] -= 1
            if new_value:
                self.label_counts[new_value] += 1
                self.committed_annotated_count += 1
                self.committed_annotation_counts[episode] += 1
            self.annotations[key] = new_value


class DraftStore:
    def __init__(self, path: Path) -> None:
        self.path = path
        self.lock = threading.RLock()
        self.connection = sqlite3.connect(path, check_same_thread=False)
        self.connection.execute("PRAGMA journal_mode=WAL")
        self.connection.execute(
            """
            CREATE TABLE IF NOT EXISTS annotation_drafts (
                episode_index INTEGER NOT NULL,
                frame_index INTEGER NOT NULL,
                annotation TEXT NOT NULL,
                updated_at REAL NOT NULL,
                PRIMARY KEY (episode_index, frame_index)
            )
            """
        )
        self.connection.execute(
            """
            CREATE TABLE IF NOT EXISTS recent_labels (
                annotation TEXT PRIMARY KEY,
                last_used REAL NOT NULL,
                use_count INTEGER NOT NULL,
                position INTEGER
            )
            """
        )
        recent_columns = {
            row[1] for row in self.connection.execute("PRAGMA table_info(recent_labels)").fetchall()
        }
        if "position" not in recent_columns:
            self.connection.execute("ALTER TABLE recent_labels ADD COLUMN position INTEGER")
        unordered_labels = self.connection.execute(
            "SELECT annotation FROM recent_labels WHERE position IS NULL ORDER BY last_used DESC, use_count DESC"
        ).fetchall()
        next_position = self.connection.execute(
            "SELECT COALESCE(MAX(position), -1) + 1 FROM recent_labels"
        ).fetchone()[0]
        for (annotation,) in unordered_labels:
            self.connection.execute(
                "UPDATE recent_labels SET position = ? WHERE annotation = ?", (next_position, annotation)
            )
            next_position += 1
        self.connection.commit()

    def set(self, episode: int, frame: int, annotation: str) -> None:
        now = time.time()
        with self.lock:
            self.connection.execute(
                """
                INSERT INTO annotation_drafts (episode_index, frame_index, annotation, updated_at)
                VALUES (?, ?, ?, ?)
                ON CONFLICT (episode_index, frame_index) DO UPDATE SET
                    annotation = excluded.annotation,
                    updated_at = excluded.updated_at
                """,
                (episode, frame, annotation, now),
            )
            if annotation:
                self.connection.execute(
                    """
                    INSERT INTO recent_labels (annotation, last_used, use_count, position)
                    VALUES (?, ?, 1, (SELECT COALESCE(MAX(position), -1) + 1 FROM recent_labels))
                    ON CONFLICT (annotation) DO UPDATE SET
                        last_used = excluded.last_used,
                        use_count = recent_labels.use_count + 1
                    """,
                    (annotation, now),
                )
            self.connection.commit()

    def all_drafts(self) -> list[tuple[int, int, str]]:
        with self.lock:
            rows = self.connection.execute(
                "SELECT episode_index, frame_index, annotation FROM annotation_drafts ORDER BY updated_at"
            ).fetchall()
        return [(int(episode), int(frame), str(annotation)) for episode, frame, annotation in rows]

    def overlay(self) -> dict[tuple[int, int], str]:
        return {(episode, frame): value for episode, frame, value in self.all_drafts()}

    def ensure_labels(self, labels: list[str]) -> None:
        with self.lock:
            for annotation in labels:
                if not annotation:
                    continue
                self.connection.execute(
                    """
                    INSERT OR IGNORE INTO recent_labels (annotation, last_used, use_count, position)
                    VALUES (?, 0, 0, (SELECT COALESCE(MAX(position), -1) + 1 FROM recent_labels))
                    """,
                    (annotation,),
                )
            self.connection.commit()

    def recent_labels(self) -> list[str]:
        with self.lock:
            rows = self.connection.execute(
                "SELECT annotation FROM recent_labels ORDER BY position, annotation"
            ).fetchall()
        return [str(row[0]) for row in rows]

    def reorder_labels(self, labels: list[str]) -> None:
        with self.lock:
            existing = self.recent_labels()
            if len(labels) != len(existing) or set(labels) != set(existing):
                raise ValueError("Reordered labels must contain every existing label exactly once")
            self.connection.executemany(
                "UPDATE recent_labels SET position = ? WHERE annotation = ?",
                [(position, annotation) for position, annotation in enumerate(labels)],
            )
            self.connection.commit()

    def clear_matching(self, records: list[tuple[int, int, str]]) -> None:
        with self.lock:
            self.connection.executemany(
                "DELETE FROM annotation_drafts WHERE episode_index = ? AND frame_index = ? AND annotation = ?",
                records,
            )
            self.connection.commit()

    def clear_all(self) -> int:
        with self.lock:
            cursor = self.connection.execute("DELETE FROM annotation_drafts")
            self.connection.commit()
            return cursor.rowcount

    def close(self) -> None:
        with self.lock:
            self.connection.close()


class AnnotationService:
    def __init__(self, root: Path, column: str, default_labels: list[str], max_image_width: int) -> None:
        self.root = root
        self.column = column
        self.default_labels = list(dict.fromkeys(label for label in default_labels if label))
        self.max_image_width = max_image_width
        self.catalog = DatasetCatalog(root, column)
        self.store = DraftStore(root / "meta" / "annotation_drafts.sqlite3")
        self.commit_lock = threading.Lock()
        self.store.ensure_labels(
            [*self.default_labels, *(label for label, _count in self.catalog.label_counts.most_common())]
        )
        self.frame_executor = concurrent.futures.ThreadPoolExecutor(max_workers=2)

        invalid_drafts = [
            (episode, frame, value)
            for episode, frame, value in self.store.all_drafts()
            if (episode, frame) not in self.catalog.locations
        ]
        if invalid_drafts:
            self.store.clear_matching(invalid_drafts)

    def labels(self) -> list[str]:
        return self.store.recent_labels()

    def state(self, episode_index: int | None = None) -> dict[str, Any]:
        if episode_index is None:
            episode_index = self.catalog.episode_indices[0]
        if episode_index not in self.catalog.frames_by_episode:
            raise KeyError(f"Unknown episode: {episode_index}")

        overlay = self.store.overlay()
        frames = self.catalog.frames_by_episode[episode_index]
        frame_payload = []
        episode_annotated = 0
        for location in frames:
            key = (episode_index, location.frame_index)
            annotation = overlay.get(key, self.catalog.annotations[key])
            episode_annotated += bool(annotation)
            frame_payload.append(
                {
                    "frame_index": location.frame_index,
                    "timestamp": location.timestamp,
                    "annotation": annotation,
                    "pending": key in overlay,
                }
            )

        total_annotated = self.catalog.committed_annotated_count
        annotation_counts = self.catalog.committed_annotation_counts.copy()
        for key, value in overlay.items():
            count_delta = int(bool(value)) - int(bool(self.catalog.annotations[key]))
            total_annotated += count_delta
            annotation_counts[key[0]] += count_delta

        episode_summaries = {}
        for index in self.catalog.episode_indices:
            summary = self.catalog.episode_summary(index)
            summary["annotation_count"] = annotation_counts[index]
            episode_summaries[str(index)] = summary

        return {
            "dataset": self.root.name,
            "column": self.column,
            "episodes": self.catalog.episode_indices,
            "episode_summaries": episode_summaries,
            "episode_index": episode_index,
            "episode_summary": episode_summaries[str(episode_index)],
            "frames": frame_payload,
            "video_keys": self.catalog.video_keys,
            "labels": self.labels(),
            "pending_count": len(overlay),
            "progress": {
                "episode_annotated": episode_annotated,
                "episode_total": len(frames),
                "total_annotated": total_annotated,
                "total_frames": self.catalog.total_frames,
            },
        }

    def annotate(self, episode: int, frame: int, annotation: str) -> dict[str, Any]:
        key = (episode, frame)
        if key not in self.catalog.locations:
            raise KeyError(f"Unknown episode/frame: {key}")
        annotation = annotation.strip()
        if len(annotation) > MAX_ANNOTATION_LENGTH:
            raise ValueError(f"Annotation must be at most {MAX_ANNOTATION_LENGTH} characters")
        self.store.set(episode, frame, annotation)
        return {
            "annotation": annotation,
            "pending_count": len(self.store.all_drafts()),
            "labels": self.labels(),
        }

    def frame_jpeg(self, episode: int, frame: int, video_key: str) -> bytes:
        key = (episode, frame)
        location = self.catalog.locations.get(key)
        if location is None:
            raise KeyError(f"Unknown episode/frame: {key}")
        video_path, video_start = self.catalog.resolve_video(episode, video_key)
        target_time = video_start + location.timestamp
        return render_video_frame(str(video_path), target_time, self.catalog.fps, self.max_image_width)

    def combined_frame_jpeg(self, episode: int, frame: int, front_key: str, side_key: str) -> bytes:
        front_future = self.frame_executor.submit(self.frame_jpeg, episode, frame, front_key)
        side_future = self.frame_executor.submit(self.frame_jpeg, episode, frame, side_key)
        front = front_future.result()
        side = side_future.result()
        return combine_square_jpegs(front, side)

    def commit(self) -> dict[str, int]:
        with self.commit_lock:
            records = self.store.all_drafts()
            if not records:
                return {"committed": 0, "files": 0, "pending_count": 0}

            grouped: dict[Path, list[tuple[int, int, str]]] = defaultdict(list)
            for episode, frame, annotation in records:
                location = self.catalog.locations[(episode, frame)]
                grouped[location.source_path].append((episode, frame, annotation))

            committed: list[tuple[int, int, str]] = []
            for path, path_records in grouped.items():
                table = pq.read_table(path)
                values = table[self.column].combine_chunks().to_pylist()
                for episode, frame, annotation in path_records:
                    location = self.catalog.locations[(episode, frame)]
                    values[location.row_index] = annotation
                replacement = pa.array(values, type=pa.string())
                column_index = table.schema.get_field_index(self.column)
                table = table.set_column(column_index, self.column, replacement)
                write_table_atomic(table, path)
                committed.extend(path_records)

            self.catalog.apply_committed(committed)
            self.store.clear_matching(committed)
            return {
                "committed": len(committed),
                "files": len(grouped),
                "pending_count": len(self.store.all_drafts()),
            }

    def discard(self) -> dict[str, int]:
        with self.commit_lock:
            discarded = self.store.clear_all()
        return {"discarded": discarded, "pending_count": 0}

    def reorder_labels(self, labels: list[str]) -> dict[str, list[str]]:
        self.store.reorder_labels(labels)
        return {"labels": self.labels()}

    def close(self) -> None:
        self.frame_executor.shutdown(wait=True, cancel_futures=True)
        self.store.close()


@lru_cache(maxsize=512)
def render_video_frame(video_path: str, timestamp: float, fps: float, max_width: int) -> bytes:
    with av.open(video_path) as container:
        stream = container.streams.video[0]
        seek_target = max(0, int(timestamp / float(stream.time_base)))
        container.seek(seek_target, stream=stream, backward=True)

        selected = None
        tolerance = 0.5 / fps
        for frame in container.decode(stream):
            selected = frame
            frame_time = float(frame.pts * frame.time_base) if frame.pts is not None else timestamp
            if frame_time >= timestamp - tolerance:
                break
            if frame_time > timestamp + 2.0:
                break
        if selected is None:
            raise RuntimeError(f"Could not decode a frame at {timestamp:.3f}s from {video_path}")

        image = selected.to_image()
        if max_width > 0 and image.width > max_width:
            height = round(image.height * max_width / image.width)
            image = image.resize((max_width, height))
        output = io.BytesIO()
        image.save(output, format="JPEG", quality=88, optimize=True)
        return output.getvalue()


def combine_square_jpegs(left_jpeg: bytes, right_jpeg: bytes) -> bytes:
    def square_crop(image: Image.Image) -> Image.Image:
        size = min(image.width, image.height)
        left = (image.width - size) // 2
        top = (image.height - size) // 2
        return image.crop((left, top, left + size, top + size)).convert("RGB")

    with Image.open(io.BytesIO(left_jpeg)) as left_source, Image.open(io.BytesIO(right_jpeg)) as right_source:
        left = square_crop(left_source)
        right = square_crop(right_source)
        size = min(left.width, right.width)
        if left.size != (size, size):
            left = left.resize((size, size), Image.Resampling.LANCZOS)
        if right.size != (size, size):
            right = right.resize((size, size), Image.Resampling.LANCZOS)

        combined = Image.new("RGB", (size * 2, size))
        combined.paste(left, (0, 0))
        combined.paste(right, (size, 0))
        output = io.BytesIO()
        combined.save(output, format="JPEG", quality=88, optimize=True)
        return output.getvalue()


class AnnotationRequestHandler(BaseHTTPRequestHandler):
    server: AnnotationHTTPServer

    def do_GET(self) -> None:
        parsed = urlparse(self.path)
        try:
            if parsed.path == "/api/state":
                query = parse_qs(parsed.query)
                episode = int(query["episode"][0]) if "episode" in query else None
                self.send_json(self.server.service.state(episode))
                return
            if parsed.path == "/api/frame":
                query = parse_qs(parsed.query)
                content = self.server.service.frame_jpeg(
                    int(query["episode"][0]), int(query["frame"][0]), query["video_key"][0]
                )
                self.send_bytes(content, "image/jpeg", cache="private, max-age=31536000, immutable")
                return
            if parsed.path == "/api/combined-frame":
                query = parse_qs(parsed.query)
                content = self.server.service.combined_frame_jpeg(
                    int(query["episode"][0]),
                    int(query["frame"][0]),
                    query["front_key"][0],
                    query["side_key"][0],
                )
                self.send_bytes(content, "image/jpeg", cache="private, max-age=31536000, immutable")
                return
            self.send_static(parsed.path)
        except (KeyError, ValueError, FileNotFoundError) as error:
            self.send_json({"error": str(error)}, status=HTTPStatus.BAD_REQUEST)
        except Exception as error:
            self.log_error("Request failed: %s", error)
            self.send_json({"error": str(error)}, status=HTTPStatus.INTERNAL_SERVER_ERROR)

    def do_POST(self) -> None:
        parsed = urlparse(self.path)
        try:
            if parsed.path == "/api/annotations":
                payload = self.read_json()
                result = self.server.service.annotate(
                    int(payload["episode_index"]), int(payload["frame_index"]), str(payload["annotation"])
                )
                self.send_json(result)
                return
            if parsed.path == "/api/commit":
                self.send_json(self.server.service.commit())
                return
            if parsed.path == "/api/discard":
                self.send_json(self.server.service.discard())
                return
            if parsed.path == "/api/labels/reorder":
                payload = self.read_json()
                self.send_json(self.server.service.reorder_labels(payload["labels"]))
                return
            self.send_json({"error": "Not found"}, status=HTTPStatus.NOT_FOUND)
        except (KeyError, TypeError, ValueError) as error:
            self.send_json({"error": str(error)}, status=HTTPStatus.BAD_REQUEST)
        except Exception as error:
            self.log_error("Request failed: %s", error)
            self.send_json({"error": str(error)}, status=HTTPStatus.INTERNAL_SERVER_ERROR)

    def read_json(self) -> dict[str, Any]:
        length = int(self.headers.get("Content-Length", "0"))
        if length > 10_000:
            raise ValueError("Request body is too large")
        return json.loads(self.rfile.read(length))

    def send_static(self, request_path: str) -> None:
        paths = {
            "/": ("index.html", "text/html; charset=utf-8"),
            "/app.js": ("app.js", "text/javascript; charset=utf-8"),
            "/styles.css": ("styles.css", "text/css; charset=utf-8"),
        }
        if request_path not in paths:
            self.send_json({"error": "Not found"}, status=HTTPStatus.NOT_FOUND)
            return
        filename, content_type = paths[request_path]
        self.send_bytes((STATIC_ROOT / filename).read_bytes(), content_type, cache="no-cache")

    def send_json(self, payload: dict[str, Any], status: HTTPStatus = HTTPStatus.OK) -> None:
        content = json.dumps(payload).encode("utf-8")
        self.send_bytes(content, "application/json; charset=utf-8", status=status, cache="no-store")

    def send_bytes(
        self,
        content: bytes,
        content_type: str,
        *,
        status: HTTPStatus = HTTPStatus.OK,
        cache: str,
    ) -> None:
        self.send_response(status)
        self.send_header("Content-Type", content_type)
        self.send_header("Content-Length", str(len(content)))
        self.send_header("Cache-Control", cache)
        self.send_header("X-Content-Type-Options", "nosniff")
        self.end_headers()
        self.wfile.write(content)

    def log_message(self, format: str, *args: Any) -> None:
        print(f"{self.address_string()} - {format % args}")


class AnnotationHTTPServer(ThreadingHTTPServer):
    daemon_threads = True

    def __init__(self, address: tuple[str, int], service: AnnotationService) -> None:
        self.service = service
        super().__init__(address, AnnotationRequestHandler)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("dataset_root", type=Path, help="Path containing data/, videos/, and meta/")
    parser.add_argument("--host", default="127.0.0.1", help="Bind address (default: 127.0.0.1)")
    parser.add_argument("--port", type=int, default=7860, help="HTTP port (default: 7860)")
    parser.add_argument("--column", default="annotation", help="Annotation feature name")
    parser.add_argument(
        "--labels",
        default="plug picked,success,failure,partial success",
        help="Comma-separated quick labels",
    )
    parser.add_argument("--max-image-width", type=int, default=1280, help="Maximum decoded frame width")
    parser.add_argument(
        "--no-initialize",
        action="store_true",
        help="Fail instead of adding the annotation column when it is missing",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    root = args.dataset_root.expanduser().resolve()
    if args.no_initialize:
        paths = sorted((root / "data").glob("**/*.parquet"))
        missing = [path for path in paths if args.column not in pq.read_schema(path).names]
        if missing:
            raise ValueError(f"Feature {args.column!r} is missing; run add_annotations_to_lerobot_dataset.py")
    else:
        changed, total = add_annotation_column(root, column=args.column)
        if changed:
            print(f"Initialized {args.column!r} in {changed} of {total} Parquet files.")

    service = AnnotationService(
        root,
        args.column,
        [label.strip() for label in args.labels.split(",")],
        args.max_image_width,
    )
    server = AnnotationHTTPServer((args.host, args.port), service)

    def stop_server(_signum: int, _frame: Any) -> None:
        threading.Thread(target=server.shutdown, daemon=True).start()

    signal.signal(signal.SIGINT, stop_server)
    signal.signal(signal.SIGTERM, stop_server)
    print(f"Annotating {root}")
    print(f"Open http://{args.host}:{args.port}")
    print("Drafts are crash-safe. Use 'Write dataset' in the browser to update Parquet files.")
    try:
        server.serve_forever()
    finally:
        server.server_close()
        service.close()


if __name__ == "__main__":
    main()
