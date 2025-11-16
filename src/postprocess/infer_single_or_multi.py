import os
from typing import Dict, Iterable
from src.inference.inference_pipeline import run_inference
from src.postprocess.sort_multi_videos import sort_and_merge_outputs
from src.postprocess.sort_multi_videos import clean_empty_folders
from src.utils.config_ops import export_config
from src.utils.config_ops import update_analyze_config

_SUPPORTED_IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff"}
_SUPPORTED_VIDEO_EXTS = {".avi", ".mp4", ".mov", ".mkv", ".m4v", ".mpg", ".mpeg", ".wmv", ".ts", ".gif", ".webm"}
_SUPPORTED_MEDIA_EXTS = _SUPPORTED_IMAGE_EXTS | _SUPPORTED_VIDEO_EXTS


def _has_media_files(directory: str) -> bool:
    """Return True if ``directory`` (recursively) contains at least one supported media file."""
    for root, _, files in os.walk(directory):
        for fname in files:
            if os.path.splitext(fname)[1].lower() in _SUPPORTED_MEDIA_EXTS:
                return True
    return False


def _iter_candidate_sources(source_dir: str) -> Iterable[str]:
    """Yield child paths under ``source_dir`` that contain supported media."""
    for entry in sorted(os.listdir(source_dir)):
        entry_path = os.path.join(source_dir, entry)
        if os.path.isdir(entry_path):
            if _has_media_files(entry_path):
                yield entry_path
            else:
                print(f"Skipping '{entry_path}': no supported images or videos found.")
        elif os.path.isfile(entry_path):
            if os.path.splitext(entry)[1].lower() in _SUPPORTED_MEDIA_EXTS:
                yield entry_path
            else:
                print(f"Skipping '{entry_path}': unsupported file type.")


def inference_single_or_multi(config,conf_yaml_path):
    
    export_config(conf_yaml_path)
    source = config.get('images_dir', '')
    update_analyze = config.get('change_analyze_conf',False)
    flag = os.path.basename(source)
    
    if "." not in flag:  # It's likely a directory with multiple files
        candidate_sources = list(_iter_candidate_sources(source))
        if not candidate_sources:
            raise FileNotFoundError(f"No supported media files found under directory '{source}'.")

        for idx, media_path in enumerate(candidate_sources):
            export_name = os.path.splitext(os.path.basename(media_path))[0] or f"export_{idx}"
            config["images_dir"] = media_path
            config["csv_filename"] = f"{export_name}_results.csv"
            run_inference(config)
    
    else:
        run_inference(config)

    sort_and_merge_outputs(config)
    clean_empty_folders(config)
    if update_analyze:
        update_analyze_config(conf_yaml_path)
   
