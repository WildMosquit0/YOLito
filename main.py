import yaml
import os
import argparse
import cProfile
import pstats
from src.inference.inference_pipeline import run_inference
from src.postprocess.infer_sngle_or_multi import inference_single_or_multi
from src.analyze.analysis_pipeline import run_analysis  # Ensure you have this implemented
from src.utils.config_ops import load_config


def main(task: str) -> None:

    conf_yaml_path =  os.path.abspath(f"configs/{task}.yaml")
    config = load_config(conf_yaml_path)
    
    if task == 'infer':
        inference_single_or_multi(config,conf_yaml_path)
    elif task == 'analyze':
        run_analysis(config,conf_yaml_path)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Run YOLO inference with tracking, detection, or slicing.')
    parser.add_argument('--task', choices=['infer', 'analyze'], default='infer',  help='The task to be performed')
    args = parser.parse_args()

    profiler = cProfile.Profile()
    profiler.enable()
    
    try:
        main(args.task)
    finally:
        profiler.disable()
        stats = pstats.Stats(profiler)
        stats.strip_dirs()
        stats.sort_stats('cumtime')
        stats.print_stats(10)
