import yaml
from ultralytics import YOLO

def load_yaml(filepath: str) -> dict:
    with open(filepath, 'r') as file:
        return yaml.safe_load(file)

def run_training(use_hpo: bool = False) -> None:
    # Load YAML configurations
    data_config = load_yaml('path/to/data.yaml')
    hyp_config = None # load_yaml('path/to/hyp.yaml') if use_hpo else None

    # Load YOLO model
    model = YOLO(data_config['model']['weights'])

    # Training configuration from YAML
    model.train(
        data='path/to/data.yaml',
        hyp=hyp_config if use_hpo else None,
        epochs=data_config['training']['epochs'],
        batch=data_config['training']['batch'],
        imgsz=data_config['training'].get('imgsz', 640),
        optimizer=data_config['training']['optimizer'],
        save_dir=data_config['output']['output_dir']
    )

if __name__ == "__main__":
    run_training(use_hpo=False)
