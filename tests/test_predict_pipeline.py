from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest
import yaml

from mosquito_supermodel.config import build_runtime_config
from mosquito_supermodel.runtime import run_task


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
    config_path = Path("tests") / "infer.yaml"
    test_config = yaml.safe_load(config_path.read_text())
    test_config["images_dir"] = str((Path(__file__).parent / "data").resolve())
    test_config["output_dir"] = str(tmp_path / "artifacts")
    test_config["change_analyze_conf"] = False

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

    # Ensure we invoked the Ultralytics predict branch for every source file.
    assert predict_calls, "predict() was never called"
    assert all(call.endswith(".JPG") for call in predict_calls)

    results_csv = Path(test_config["output_dir"]) / "results.csv"
    assert results_csv.exists(), "Merged results.csv was not created"

    df = pd.read_csv(results_csv)
    assert not df.empty, "results.csv is empty"
    for required in ("image_idx", "box_idx", "x", "y", "w", "h", "confidence", "image_name"):
        assert required in df.columns, f"Missing column '{required}' in results.csv"
