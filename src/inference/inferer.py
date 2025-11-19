import cv2
import os
import subprocess
import torch
from ultralytics import YOLO
from typing import List, Dict
import yaml
from src.utils.common import create_output_dir
from sahi.predict import get_prediction, get_sliced_prediction
from sahi import AutoDetectionModel
from src.utils.common import export_middle_frame
#
#def ensure_weights():
#    try:
#        subprocess.run(["git", "lfs", "pull"], check=True)
#        print("Weights downloaded successfully.")
#    except subprocess.CalledProcessError as e:
#        print(f"Failed to download weights: {e}")
#
class Inferer:
    def __init__(self, config: Dict) -> None:
        self.config = config
        #ensure_weights()

        # Model and task parameters
        self.model_path = config['model']['weights']
        self.task = config['model']['task']
        self.conf_threshold = config['model'].get('conf_threshold', 0.5)
        self.iou_threshold = config['model'].get('iou_threshold', 0.45)
        self.vid_stride = config['model'].get('vid_stride', 1)
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'

        # Slicing parameters (used when task == 'slice')
        self.slice_height = config['model'].get('slice_height', 640)
        self.slice_width = config['model'].get('slice_width', 640)
        self.overlap_height_ratio = config['model'].get('overlap_height_ratio', 0.2)
        self.overlap_width_ratio = config['model'].get('overlap_width_ratio', 0.2)

        # Directories and output options
        self.output_dir = config['output_dir']
        self.images_dir = config['images_dir']
        self.save_animations = config.get('save_animations', False)
        self.sliced_source_dir = os.path.join(self.output_dir)
        os.makedirs(self.sliced_source_dir, exist_ok=True)
        create_output_dir(self.output_dir)

        # Load YOLO model for tracking/predict tasks.
        self.model = YOLO(self.model_path)
        self.model.to(self.device)
        export_middle_frame(self.images_dir,self.output_dir,self.task)


    def infer(self, persist: bool = True) -> List:
        results = None
        if self.task == 'track':
            results = self.model.track(
                source=self.images_dir,
                conf=self.conf_threshold,
                iou=self.iou_threshold,
                persist=persist,
                save=self.save_animations,
                vid_stride=self.vid_stride,
                project=self.output_dir,
                exist_ok=True
            )
        elif self.task == 'slice':

            from src.utils.sahi_usage import sahi_usage 
            sahi_inferer = sahi_usage(self.config)
            results = sahi_inferer.run_command(self.images_dir)
        else:
            results = self.model.predict(
                source=self.images_dir,
                conf=self.conf_threshold,
                iou=self.iou_threshold,
                save=self.save_animations,
                vid_stride=self.vid_stride,
                project=self.output_dir,
                exist_ok=True
            )
       
        print(f"Results saved to: {self.output_dir}")
        return results


