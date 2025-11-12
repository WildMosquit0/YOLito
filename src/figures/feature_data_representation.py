import yaml
import os
import cv2
import numpy as np
import matplotlib.pyplot as plt


class DataProcessor:
    def __init__(self, image_path, annotation_path):
        self.image_path = image_path
        self.annotation_path = annotation_path
        self.image_features = []
        self.annotation_features = []

    def __call__(self):
        self.calculate_image_features()
        self.calculate_annotation_features()
        return self.image_features, self.annotation_features

    def calculate_image_features(self):
        for img_file in os.listdir(self.image_path):
            img_path = os.path.join(self.image_path, img_file)
            img = cv2.imread(img_path, cv2.IMREAD_COLOR)
            if img is not None:
                brightness = np.mean(cv2.cvtColor(img, cv2.COLOR_BGR2GRAY))
                resolution = img.shape[:2]  # (height, width)
                aspect_ratio = resolution[1] / resolution[0]
                self.image_features.append({
                    "filename": img_file,
                    "brightness": brightness,
                    "resolution": resolution,
                    "aspect_ratio": aspect_ratio
                })

    def calculate_annotation_features(self):
        for annot_file in os.listdir(self.annotation_path):
            annot_path = os.path.join(self.annotation_path, annot_file)
            with open(annot_path, 'r') as file:
                for line in file:
                    components = line.strip().split()
                    if len(components) >= 5:
                        _, _, width, height = map(float, components[1:])
                        object_size_ratio = width * height  # Fraction of image area
                        aspect_ratio = width / height
                        self.annotation_features.append({
                            "filename": annot_file,
                            "object_size_ratio": object_size_ratio,
                            "aspect_ratio": aspect_ratio
                        })


class Visualizer:
    def __init__(self, save_path):
        self.save_path = save_path
        if not os.path.exists(save_path):
            os.makedirs(save_path)

    def __call__(self, image_features, annotation_features):
        # Plot and save histograms
        self.plot_histogram(
            [f['brightness'] for f in image_features],
            "Image Brightness Distribution", "Brightness", "Frequency", "brightness_histogram.png"
        )
        self.plot_histogram(
            [f['resolution'][0] * f['resolution'][1] for f in image_features],
            "Image Resolution Distribution", "Resolution (pixels)", "Frequency", "resolution_histogram.png"
        )
        self.plot_histogram(
            [f['aspect_ratio'] for f in image_features],
            "Image Aspect Ratio Distribution", "Aspect Ratio", "Frequency", "aspect_ratio_histogram.png"
        )
        self.plot_histogram(
            [f['object_size_ratio'] for f in annotation_features],
            "Object Size Ratio Distribution", "Size Ratio", "Frequency", "object_size_ratio_histogram.png"
        )
        self.plot_histogram(
            [f['aspect_ratio'] for f in annotation_features],
            "Object Aspect Ratio Distribution", "Aspect Ratio", "Frequency", "object_aspect_ratio_histogram.png"
        )

    def plot_histogram(self, data, title, xlabel, ylabel, filename, bins=20):
        plt.figure(figsize=(8, 6))
        plt.hist(data, bins=bins, edgecolor='black')
        plt.title(title)
        plt.xlabel(xlabel)
        plt.ylabel(ylabel)
        plt.grid(alpha=0.5)
        save_file_path = os.path.join(self.save_path, filename)
        plt.savefig(save_file_path)
        print(f"Plot saved to {save_file_path}")
        plt.close()


def load_config(config_path):
    with open(config_path, 'r') as file:
        return yaml.safe_load(file)


def create_figure(image_path, annotation_path, save_path):
    # Process data
    data_processor = DataProcessor(image_path, annotation_path)
    image_features, annotation_features = data_processor()

    # Visualize data
    visualizer = Visualizer(save_path)
    visualizer(image_features, annotation_features)


def main():
    # Load configuration
    config = load_config('src/figures/config.yaml')
    image_path = config['paths']['image_path']
    annotation_path = config['paths']['annotation_path']
    save_path = config['paths']['save_path']

    # Generate figures
    create_figure(image_path, annotation_path, save_path)


if __name__ == "__main__":
    main()
