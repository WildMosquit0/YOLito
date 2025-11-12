from ultralytics import YOLO

class YOLODetectionModel:
    """
    YOLODetectionModel for Ultralytics YOLOv8-based object detection.
    Handles training, validation, and prediction.
    """

    def __init__(self, config: dict):
        self.config = config
        self.model = YOLO(config['model']['weights'])

    def train(self) -> None:
        self.model.train(
            data={
                'train': self.config['input']['train'],
                'val': self.config['input']['val']
            },
            epochs=self.config['training']['epochs'],
            batch=self.config['training']['batch'],
            imgsz=self.config['training'].get('img_size', 640),
            optimizer=self.config['training']['optimizer'],
            lr0=self.config['training']['lr'], 
            save_dir=self.config['output']['output_dir']
        )

    def validate(self) -> None:
        self.model.val()

    def predict(self, source: str) -> None:
        self.model.predict(source)
