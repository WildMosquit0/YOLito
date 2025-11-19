import os
import shutil
import pandas as pd
from typing import Dict

def sort_and_merge_outputs(config: Dict):
    output_dir = config["output_dir"]

    # Create subfolders
    videos_dir = os.path.join(output_dir, "videos")
    frames_dir = os.path.join(output_dir, "frames")
    csvs_dir = os.path.join(output_dir, "csvs")
    os.makedirs(videos_dir, exist_ok=True)
    os.makedirs(frames_dir, exist_ok=True)
    os.makedirs(csvs_dir, exist_ok=True)

    # File type sets
    video_exts = {".avi", ".mp4", ".mov", ".mkv", ".m4v"}
    frame_exts = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff"}

    # Protected files
    def is_protected(fname: str) -> bool:
        f = fname.lower()
        return f.startswith("conf")

    skip_dirs = {os.path.abspath(videos_dir), os.path.abspath(frames_dir), os.path.abspath(csvs_dir)}

    # Ensure unique path if file already exists
    def unique_path(target_dir: str, name: str) -> str:
        base, ext = os.path.splitext(name)
        candidate = os.path.join(target_dir, name)
        i = 1
        while os.path.exists(candidate):
            candidate = os.path.join(target_dir, f"{base}__{i}{ext}")
            i += 1
        return candidate

    found_results_csv = None

    # Walk all files
    for root, dirs, files in os.walk(output_dir, topdown=True):
        dirs[:] = [d for d in dirs if os.path.abspath(os.path.join(root, d)) not in skip_dirs]

        for fname in files:
            src = os.path.join(root, fname)

            # results.csv handling → move to root
            if fname.lower() == "results.csv":
                if os.path.abspath(root) != os.path.abspath(output_dir):
                    dst = os.path.join(output_dir, "results.csv")
                    shutil.move(src, dst)
                found_results_csv = os.path.join(output_dir, "results.csv")
                continue

            if is_protected(fname):
                continue

            ext = os.path.splitext(fname)[1].lower()
            if ext in video_exts:
                shutil.move(src, unique_path(videos_dir, fname))
            elif ext in frame_exts:
                shutil.move(src, unique_path(frames_dir, fname))
            elif ext == ".csv":
                shutil.move(src, unique_path(csvs_dir, fname))

    # Merge CSVs from csvs_dir
    merged_data = []
    current_max_track_id = 0

    for file in sorted(os.listdir(csvs_dir)):
        if not file.lower().endswith(".csv"):
            continue
        file_path = os.path.join(csvs_dir, file)
        try:
            df = pd.read_csv(file_path)
        except Exception as e:
            print(f"Warning: failed reading {file_path}: {e}")
            continue

        if 'track_id' in df.columns:
            df['track_id'] = pd.to_numeric(df['track_id'], errors='coerce').fillna(0).astype(int)
            df['track_id'] = df['track_id'] + current_max_track_id
            if len(df):
                current_max_track_id = df['track_id'].max() + 1
        else:
            print(f"Warning: 'track_id' not found in {file}, skipping track ID offset.")

        if 'image_name' in df.columns and 'treatment' not in df.columns:
            df['image_name'] = df['image_name'].astype(str)
            df['treatment'] = df['image_name'].apply(lambda x: x.split('_')[0] if '_' in x else x)

        merged_data.append(df)

    if not merged_data:
        print("No CSVs to merge.")
        return

    final_df = pd.concat(merged_data, ignore_index=True)
    if 'track_id' in final_df.columns:
        final_df = final_df.sort_values(by='track_id')

    output_path = os.path.join(output_dir, "results.csv")
    final_df.to_csv(output_path, index=False)


def clean_empty_folders(config: Dict):
    output_dir = config["output_dir"]

    for root, dirs, files in os.walk(output_dir, topdown=False):
        
        try:
            if not os.listdir(root):
                os.rmdir(root)
                print(f"Removed empty folder: {root}")
        except OSError as e:
            print(f"Could not remove {root}: {e}")

    return output_dir
