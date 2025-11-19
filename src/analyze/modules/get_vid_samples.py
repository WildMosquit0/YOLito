import cv2
import os
import json



class movie_slicer:
    def __init__(self, config_path="config.json"):
        # Load configuration from the specified JSON file
        self.config = self._load_config(config_path)
        self.input_dir = self.config["crop_movies"]["input_dir"]
        self.output_dir = self.config["crop_movies"].get("output_dir", os.path.join(self.input_dir,'cropped'))
        
        self.start_seconds = int(self.config["crop_movies"].get("start_seconds", 60))
        self.end_seconds = int(self.config["crop_movies"].get("end_seconds", 120))
        self.config_fps = self.config["crop_movies"].get("fps")  # May be None or invalid
    
        # Ensure output directory exists
        os.makedirs(self.output_dir, exist_ok=True)

    def _load_config(self, config_path):
        # Load the JSON configuration file
        if not os.path.exists(config_path):
            raise FileNotFoundError(f"Config file not found: {config_path}")
        with open(config_path, "r") as file:
            return json.load(file)

    def get_movies_path(self):
        # Get the list of video file paths in the input directory
        video_files = [
            os.path.join(self.input_dir, vid)
            for vid in os.listdir(self.input_dir)
            if vid.lower().endswith((".mp4", ".avi", ".mov", ".mkv"))
        ]
        return video_files

    def slice_movies(self):
        # Slice each video in the input directory
        video_files = self.get_movies_path()
        for video_path in video_files:
            self._process_and_save_video(video_path)

    def _process_and_save_video(self, video_path):
        # Open the video file
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            print(f"Error opening video file: {video_path}")
            return

        # Get video properties
        frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        video_fps = cap.get(cv2.CAP_PROP_FPS)  # FPS from the video

        # Determine FPS: use config FPS if valid, else fallback to video FPS
        fps = self.config_fps if self.config_fps and self.config_fps > 0 else video_fps
        if not fps or fps <= 0:  # Handle cases where video FPS is also invalid
            print(f"Invalid FPS for video: {video_path}. Skipping.")
            cap.release()
            return

        start_frame = int(self.start_seconds * video_fps)
        end_frame = int(self.end_seconds * video_fps)


        # Calculate valid slicing range
        start_frame = min(start_frame, frame_count - 1)
        end_frame = min(end_frame, frame_count)

        # Prepare output video writer
        output_path = os.path.join(self.output_dir, 'c_' + os.path.basename(video_path))
        fourcc = cv2.VideoWriter_fourcc(*"mp4v")  # Codec for MP4 format
        out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))

        # Process frames
        for i in range(frame_count):
            ret, frame = cap.read()
            if not ret:
                break
            if start_frame <= i < end_frame:
                out.write(frame)

        # Release resources
        cap.release()
        out.release()
        print(f"Processed and saved: {output_path}")

    def save_vids(self):
        # Wrapper for the slicing functionality
        self.slice_movies()

    def __call__(self):
        # When an instance is called, execute the slicing process
        self.save_vids()


# Main execution block
if __name__ == "__main__":
    config_path = "config.json"  # You can change this to the desired config path
    slicer = movie_slicer(config_path)
    slicer()  # Calling the instance, triggers the slicing process
