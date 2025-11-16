import pandas as pd

from src.analyze.modules.sahi_tracker import track


def test_track_assigns_consistent_ids(tmp_path):
    data = [
        {"image_idx": 0, "box_idx": 0, "x": 0.0, "y": 0.0, "w": 10.0, "h": 10.0, "confidence": 0.9},
        {"image_idx": 0, "box_idx": 1, "x": 50.0, "y": 50.0, "w": 10.0, "h": 10.0, "confidence": 0.8},
        {"image_idx": 1, "box_idx": 0, "x": 1.0, "y": 0.0, "w": 10.0, "h": 10.0, "confidence": 0.95},
        {"image_idx": 1, "box_idx": 1, "x": 52.0, "y": 50.0, "w": 10.0, "h": 10.0, "confidence": 0.85},
    ]
    csv_path = tmp_path / "detections.csv"
    pd.DataFrame(data).to_csv(csv_path, index=False)

    track(csv_file=str(csv_path))

    updated = pd.read_csv(csv_path)
    assert "track_id" in updated.columns
    sorted_df = updated.sort_values(["frame", "image_idx", "box_idx"]).reset_index(drop=True)
    a_first = sorted_df.loc[0, "track_id"]
    a_second = sorted_df.loc[2, "track_id"]
    b_first = sorted_df.loc[1, "track_id"]
    b_second = sorted_df.loc[3, "track_id"]
    assert a_first == a_second
    assert b_first == b_second
    assert a_first != b_first
    assert (sorted_df["track_id"] > 0).all()
