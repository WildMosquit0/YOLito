import os
from ultralytics import YOLO
from ultralytics.utils.benchmarks import benchmark

root = '/home/bohbot/Evyatar'
model_path = 'runs/detect/most_update_with_insects_250_epocs4/weights/best.pt'
data='yaml/val.yaml'

model = os.path.join(root, model_path)
data = os.path.join(root, data)

weights = YOLO(model)

validation_results = weights.val(
    data=data, 
    imgsz=640, 
    batch=16, 
    conf=0.25, 
    iou=0.6, 
    device="0",
    save_json=False)
