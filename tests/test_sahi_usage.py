from pathlib import Path

from src.utils.sahi_usage import sahi_usage


class _DummyResult:
    def __init__(self):
        self.written = []

    def export_visuals(self, export_dir: str, file_name: str) -> None:
        export_path = Path(export_dir)
        export_path.mkdir(parents=True, exist_ok=True)
        target = export_path / f"{file_name}.png"
        target.write_text("visual")
        self.written.append(target)


def _base_config(tmp_path, save_animations: bool):
    return {
        "model": {
            "weights": "dummy.pt",
            "task": "slice",
        },
        "output_dir": str(tmp_path / "artifacts"),
        "images_dir": str(tmp_path / "images"),
        "save_animations": save_animations,
        "sahi": {},
    }


def test_sahi_usage_saves_visuals_when_enabled(tmp_path):
    usage = sahi_usage(_base_config(tmp_path, True))
    result = _DummyResult()

    saved_path = usage._maybe_save_visual(result, "foo")

    assert saved_path is not None
    visuals_dir = Path(usage.slice_visuals_dir)
    assert visuals_dir.exists(), "Visuals directory was not created"
    saved = list(visuals_dir.glob("foo_sliced*.png"))
    assert saved, "Expected at least one saved visualization for slice task"
    assert result.written, "Dummy result did not record any writes"


def test_sahi_usage_skips_visuals_when_disabled(tmp_path):
    usage = sahi_usage(_base_config(tmp_path, False))
    result = _DummyResult()

    saved_path = usage._maybe_save_visual(result, "bar")

    assert saved_path is None
    assert not Path(usage.slice_visuals_dir).exists(), "Visuals directory should not be created when disabled"
    assert not result.written, "No visualization export expected when save_animations=False"


def test_export_video_from_visuals_calls_writer(tmp_path, monkeypatch):
    usage = sahi_usage(_base_config(tmp_path, True))
    frames = [str(tmp_path / "f1.jpg"), str(tmp_path / "f2.jpg")]
    called = {}

    def fake_writer(frame_paths, fps, output_path):
        called["args"] = (frame_paths, fps, output_path)

    monkeypatch.setattr(usage, "_write_video_from_frames", fake_writer)
    usage._export_video_from_visuals(frames, "/tmp/video.mp4", 12.0)

    assert "args" in called
    assert called["args"][0] == frames
    assert called["args"][1] == 12.0
    assert called["args"][2].endswith("video_sahi.avi")


def test_export_video_from_visuals_noop_when_disabled(tmp_path, monkeypatch):
    usage = sahi_usage(_base_config(tmp_path, False))
    called = {}

    def fake_writer(frame_paths, fps, output_path):
        called["args"] = True

    monkeypatch.setattr(usage, "_write_video_from_frames", fake_writer)
    usage._export_video_from_visuals(["foo.jpg"], "/tmp/video.mp4", 10.0)

    assert "args" not in called


def test_sahi_predict_respects_slice_config(tmp_path, monkeypatch):
    """Ensure slice-specific config flows into the SAHI prediction helper."""
    captured = {}

    class DummyResult:
        object_prediction_list = []

    def fake_get_sliced_prediction(**kwargs):
        captured.update(kwargs)
        return DummyResult()

    monkeypatch.setattr("src.utils.sahi_usage.get_sliced_prediction", fake_get_sliced_prediction)

    cfg = _base_config(tmp_path, False)
    cfg["sahi"].update({"slice_size_h": 123, "slice_size_w": 456, "overlap_ratio": 0.33})
    usage = sahi_usage(cfg)

    usage.sahi_predict("dummy", object())

    assert captured["slice_height"] == 123
    assert captured["slice_width"] == 456
    assert captured["overlap_height_ratio"] == 0.33
    assert captured["overlap_width_ratio"] == 0.33
    assert captured["image"] == "dummy"
