from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest
import yaml

from yolito.config import build_runtime_config
from yolito.runtime import run_task


class _DummyTensor:
    """Minimal tensor-like object that mimics the Ultralytics API used in ResultsParser."""

    def __init__(self, array):
        self._array = np.asarray(array)

    def cpu(self):
        return self

    def numpy(self):
        return self._array


def _make_result(source_path: str) -> SimpleNamespace:
    """Create a fake Ultralytics result object targeting the provided source file."""
    boxes = SimpleNamespace(
        xywh=_DummyTensor([[10.0, 15.0, 5.0, 5.0]]),
        conf=_DummyTensor([0.9]),
        cls=_DummyTensor([0]),
        id=None,
    )
    return SimpleNamespace(
        path=source_path,
        orig_shape=(720, 1280),
        boxes=boxes,
    )


def test_predict_flag_end_to_end(tmp_path, monkeypatch):
    """Ensure the CLI runs the predict branch end-to-end and produces a merged CSV."""
    config_path = Path("tests") / "infer_predict.yaml"
    test_config = yaml.safe_load(config_path.read_text())
    test_config["images_dir"] = str((Path(__file__).parent / "data").resolve())
    test_config["output_dir"] = str(tmp_path / "artifacts")
    test_config["change_analyze_conf"] = False

    predict_calls = []
    expected_sources = sorted(
        str(p) for p in Path(test_config["images_dir"]).iterdir() if p.is_file()
    )

    class DummyYOLO:
        def __init__(self, weights):
            self.weights = weights

        def to(self, device):
            self.device = device
            return self

        def predict(self, *args, **kwargs):
            source = kwargs.get("source")
            predict_calls.append(source)
            return [_make_result(source)]

        def track(self, *args, **kwargs):
            raise AssertionError("track should not be called for predict task")

    monkeypatch.setattr("src.inference.inferer.YOLO", DummyYOLO)

    runtime_config = build_runtime_config("infer", str(config_path))
    run_task(runtime_config, preload=test_config)

    # Ensure we invoked the Ultralytics predict branch for every source file.
    assert predict_calls, "predict() was never called"
    assert sorted(predict_calls) == expected_sources

    results_csv = Path(test_config["output_dir"]) / "results.csv"
    assert results_csv.exists(), "Merged results.csv was not created"

    df = pd.read_csv(results_csv)
    assert not df.empty, "results.csv is empty"
    for required in ("image_idx", "box_idx", "x", "y", "w", "h", "confidence", "image_name"):
        assert required in df.columns, f"Missing column '{required}' in results.csv"


def test_slice_flag_routes_through_sahi(tmp_path, monkeypatch):
    """Verify slice task uses SAHI predictions and produces merged CSV output."""
    config_path = Path("tests") / "infer_slice.yaml"
    test_config = yaml.safe_load(config_path.read_text())
    data_dir = (Path(__file__).parent / "data").resolve()
    test_config["images_dir"] = str(data_dir)
    test_config["output_dir"] = str(tmp_path / "artifacts")
    test_config["change_analyze_conf"] = False

    expected_sources = sorted(str(p) for p in data_dir.iterdir() if p.is_file())
    run_command_calls = []

    class DummySahi:
        def __init__(self, cfg):
            self.cfg = cfg

        def run_command(self, source):
            run_command_calls.append(source)
            basename = Path(source).stem
            return [
                (0, 10.0, 15.0, 5.0, 5.0, 0.85, basename, 0, 720, 1280),
            ]

    class DummyYOLO:
        def __init__(self, weights):
            self.weights = weights

        def to(self, device):
            self.device = device
            return self

        def predict(self, *args, **kwargs):
            raise AssertionError("predict should not be called for slice task")

    monkeypatch.setattr("src.inference.inferer.YOLO", DummyYOLO)
    monkeypatch.setattr("src.utils.sahi_usage.sahi_usage", DummySahi)

    runtime_config = build_runtime_config("infer", str(config_path))
    run_task(runtime_config, preload=test_config)

    assert run_command_calls == expected_sources, "SAHI was not invoked for every source file"

    results_csv = Path(test_config["output_dir"]) / "results.csv"
    assert results_csv.exists(), "Merged slice results.csv was not created"

    df = pd.read_csv(results_csv)
    assert len(df) == len(expected_sources), "Slice merge did not include every detection"
    assert set(df["image_name"]) == {Path(p).stem for p in expected_sources}


def test_predict_track_config_handles_video_source(tmp_path, monkeypatch):
    """Ensure predict task still uses YOLO.predict even when source is a video."""
    config_path = Path("tests") / "infer_predict_track.yaml"
    test_config = yaml.safe_load(config_path.read_text())
    video_path = str((Path(__file__).parent / "data" / "slow.mp4").resolve())
    test_config["images_dir"] = video_path
    test_config["output_dir"] = str(tmp_path / "artifacts")
    test_config["change_analyze_conf"] = False
    test_config["save_animations"] = False

    predict_calls = []

    class DummyYOLO:
        def __init__(self, weights):
            self.weights = weights

        def to(self, device):
            self.device = device
            return self

        def predict(self, *args, **kwargs):
            source = kwargs.get("source")
            predict_calls.append(source)
            return [_make_result(source)]

        def track(self, *args, **kwargs):
            raise AssertionError("track should not be called for predict task")

    monkeypatch.setattr("src.inference.inferer.YOLO", DummyYOLO)

    runtime_config = build_runtime_config("infer", str(config_path))
    run_task(runtime_config, preload=test_config)

    assert predict_calls == [video_path], "YOLO.predict did not receive the video source"

    results_csv = Path(test_config["output_dir"]) / "results.csv"
    assert results_csv.exists(), "Track video results.csv was not created"

    df = pd.read_csv(results_csv)
    assert not df.empty, "Track video results.csv is empty"
    assert set(df["image_name"]) == {Path(video_path).stem}


def test_slice_track_video_routes_through_sahi(tmp_path, monkeypatch):
    """Verify slice task on a video source uses SAHI and emits a merged CSV."""
    config_path = Path("tests") / "infer_slice_track.yaml"
    test_config = yaml.safe_load(config_path.read_text())
    video_path = str((Path(__file__).parent / "data" / "slow.mp4").resolve())
    test_config["images_dir"] = video_path
    test_config["output_dir"] = str(tmp_path / "artifacts")
    test_config["change_analyze_conf"] = False
    test_config["save_animations"] = False

    run_command_calls = []

    class DummySahi:
        def __init__(self, cfg):
            self.cfg = cfg

        def run_command(self, source):
            run_command_calls.append(source)
            basename = Path(source).stem
            return [
                (0, 10.0, 15.0, 5.0, 5.0, 0.9, basename, 0, 720, 1280),
                (1, 12.0, 10.0, 4.0, 4.0, 0.6, basename, 1, 720, 1280),
            ]

    class DummyYOLO:
        def __init__(self, weights):
            self.weights = weights

        def to(self, device):
            self.device = device
            return self

        def predict(self, *args, **kwargs):
            raise AssertionError("predict should not be called for slice task")

    monkeypatch.setattr("src.inference.inferer.YOLO", DummyYOLO)
    monkeypatch.setattr("src.utils.sahi_usage.sahi_usage", DummySahi)

    runtime_config = build_runtime_config("infer", str(config_path))
    run_task(runtime_config, preload=test_config)

    assert run_command_calls == [video_path], "SAHI did not receive the video source"

    results_csv = Path(test_config["output_dir"]) / "results.csv"
    assert results_csv.exists(), "Slice track results.csv was not created"

    df = pd.read_csv(results_csv)
    assert len(df) == 2, "Slice track CSV should contain both synthetic detections"
    assert set(df["image_name"]) == {Path(video_path).stem}
