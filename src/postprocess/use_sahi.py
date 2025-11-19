from sahi.predict import predict as sahi_predict
from ultralytics import YOLO
import os
# SAHI Prediction with your specific arguments

model = "/home/wildmosquit0/git/bestModel/test_different_models/sahi_l_1_more_300e_200.pt"
source = "/home/wildmosquit0/git/model_exp/peers/3-2-2023_rep2_m.mp4"


conf = 0.45


#from sahi.slicing import slice_image
#for i in os.listdir(source):
#    img_name = os.path.join(source,i)
#    slice_image_result = slice_image(
#    image=img_name,
#    output_file_name= os.path.basename(i),
#    output_dir="run",
#    slice_height=640,
#    slice_width=640,
#    overlap_height_ratio=0.2,
#    overlap_width_ratio=0.2,
#    )

new_source = "run"
sahi_predict(
    model_type="ultralytics",  # Use the model type specific to your YOLO version
    model_path= model,
    model_device="cuda:0",  # Set to 'cuda:0' if using GPU
    model_confidence_threshold=conf,
    source=source,  # Path to directory containing images
    slice_height=640,
    slice_width=640,
    overlap_height_ratio=0.2,
    overlap_width_ratio=0.2
    
)


# Load the YOLO model
model = YOLO(model)
# Perform prediction
results = model.predict(
    source=source,
    conf=conf,  # Confidence threshold for predictions
    iou = 0.8,
    save=True,
    imgsz=640,  # Save the results
    line_width = 2
)