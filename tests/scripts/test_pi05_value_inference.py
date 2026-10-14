"""Concurrent inference output checks without loading a policy or using a GPU."""

import json
import multiprocessing
from concurrent.futures import ThreadPoolExecutor

import av
import pytest
import torch

from lerobot.scripts import lerobot_pi05_value_inference as inference


def _write_predictions(path, fold):
    for step in range(5):
        inference.update_prediction_json(path, {f"fold_{fold}_{step}": [fold, step]})


def test_concurrent_prediction_updates_preserve_all_models(tmp_path):
    path = tmp_path / "all_cams.json"
    inference.write_json_atomic(path, {"existing": [1]})
    context = multiprocessing.get_context("fork")
    processes = [context.Process(target=_write_predictions, args=(path, fold)) for fold in range(10)]
    for process in processes:
        process.start()
    for process in processes:
        process.join(timeout=20)
        if process.is_alive():
            process.terminate()
            process.join()
            pytest.fail("Concurrent JSON writer stalled")
        assert process.exitcode == 0
    expected = {f"fold_{fold}_{step}": [fold, step] for fold in range(10) for step in range(5)}
    assert json.loads(path.read_text()) == {"existing": [1], **expected}
    assert list(tmp_path.iterdir()) == [path]


def test_invalid_update_leaves_previous_json_intact(tmp_path):
    path = tmp_path / "all_cams.json"
    inference.write_json_atomic(path, {"existing": [1]})
    with pytest.raises(ValueError):
        inference.update_prediction_json(path, {"bad": float("nan")})
    assert json.loads(path.read_text()) == {"existing": [1]}
    assert list(tmp_path.iterdir()) == [path]


def test_concurrent_video_writers_repair_partial_video(tmp_path, monkeypatch):
    path = tmp_path / "all_cams.mp4"
    path.write_bytes(b"interrupted encoding")
    dataset = [{"camera": torch.zeros(3, 16, 16)} for _ in range(4)]

    def write():
        inference.write_all_cams_video(dataset, ["camera"], 0, 4, path, 20)

    with ThreadPoolExecutor(max_workers=3) as pool:
        list(pool.map(lambda _: write(), range(3)))
    with av.open(str(path)) as container:
        assert len(list(container.decode(video=0))) == 4
    assert list(tmp_path.iterdir()) == [path]

    def unexpected_encode(*args):
        pytest.fail("A complete video should be reused")

    monkeypatch.setattr(inference, "_write_all_cams_video", unexpected_encode)
    write()
