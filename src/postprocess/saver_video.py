import csv
import os
from typing import List
from ultralytics.engine.results import Results

class ResultsParserVid:
    def __init__(self, results: List[Results], config: dict) -> None:
        self.results = results
        self.output_dir = config['output']['output_dir']
        self.csv_filename = config['output'].get('csv_filename', 'results.csv')  

        if not os.path.exists(self.output_dir):
            os.makedirs(self.output_dir)

    def parse_and_save(self):
        csv_file_path = os.path.join(self.output_dir, self.csv_filename)

        with open(csv_file_path, mode='w', newline='') as csv_file:
            writer = csv.writer(csv_file)
            writer.writerow(['image_idx', 'image_name', 'box_idx', 'x', 'y', 'w', 'h', 'confidence', 'label', 'track_id'])

            for img_idx, result in enumerate(self.results):
                image_name = os.path.basename(result.path)
                boxes = result.boxes.xywh.cpu().numpy()  # Bounding boxes
                scores = result.boxes.conf.cpu().numpy()  # Confidence scores
                labels = result.boxes.cls.cpu().numpy()  # Labels (class indices)
                track_ids = result.boxes.id.cpu().numpy() if result.boxes.id is not None else None  # Track IDs for tracking tasks

                for box_idx, (box, score, label) in enumerate(zip(boxes, scores, labels)):
                    track_id = track_ids[box_idx] if track_ids is not None else None
                    writer.writerow([img_idx, image_name, box_idx, *box, score, label, track_id])

        print(f"Results saved to {csv_file_path}")
