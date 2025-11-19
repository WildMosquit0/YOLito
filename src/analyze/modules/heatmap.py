import os
import cv2
import pandas as pd
import numpy as np
from src.utils.common import create_output_dir
from src.utils.common import find_image_for_heat_map


class Heatmap:
    def __init__(self, config):
        self.config = config
        self.data_path = self.config["input_csv"]
        self.plot_path = os.path.join(self.config["output_dir"], "plots")
        self.frame_path = self.config["heatmap"]["image_path"]
        self.radius_increment = self.config.get("radius_increment", 0.5)
        self.clip_max = self.config.get("clip_max", 255)
        self.colormap = self.config.get("colormap", cv2.COLORMAP_JET)

    def heatmap_effect(self, heatmap, x_center, y_center, w, h):
        """Draw a circular heat effect centered on (x_center, y_center)."""
        x0 = int(x_center - w / 2)
        y0 = int(y_center - h / 2)
        x1 = int(x_center + w / 2)
        y1 = int(y_center + h / 2)

        H, W = heatmap.shape
        x0 = max(0, x0)
        y0 = max(0, y0)
        x1 = min(W, x1)
        y1 = min(H, y1)

        if x1 <= x0 or y1 <= y0:
            return

        radius_squared = (min(x1 - x0, y1 - y0) // 2) ** 2
        xv, yv = np.meshgrid(np.arange(x0, x1), np.arange(y0, y1))
        dist_squared = (xv - ((x0 + x1) // 2)) ** 2 + (yv - ((y0 + y1) // 2)) ** 2
        mask = dist_squared <= radius_squared
        heatmap[y0:y1, x0:x1][mask] += self.radius_increment

    def plot_heatmap(self, name="none"):
        if name == "none":
            name = os.path.basename(self.data_path).split('.')[0]
        create_output_dir(self.plot_path)

        image = cv2.imread(self.frame_path)
        if image is None:
            raise FileNotFoundError(f"Image not found: {self.frame_path}")

        heatmap = np.zeros(image.shape[:2], dtype=np.float32)

        for _, row in self.data.iterrows():
            self.heatmap_effect(heatmap, row["x"], row["y"], row["w"], row["h"])

        # Clip and colorize the heatmap
        heatmap_clipped = np.clip(heatmap, 0, self.clip_max).astype(np.uint8)
        color_heatmap = cv2.applyColorMap(heatmap_clipped, self.colormap)

        # Blend with original image
        overlay = cv2.addWeighted(image, 0.5, color_heatmap, 0.5, 0)

        output_path = os.path.join(self.plot_path, f"{name} Heatmap.png")
        try:
            cv2.imwrite(output_path, overlay)
            print(f"Overlay heatmap saved at: {output_path}")
        except Exception as e:
            raise RuntimeError(f"Failed to save heatmap: {e}")

    def __call__(self):
        source = self.data_path
        is_folder = not os.path.splitext(source)[-1].lower().endswith(".csv")

        if is_folder:
            csv_files = [f for f in os.listdir(source) if f.endswith(".csv")]
            for file in csv_files:
                self.data_path = os.path.join(source, file)
                self.data = pd.read_csv(self.data_path)
                img_name = self.data["image_name"].iloc[0]
                self.frame_path = find_image_for_heat_map(source, img_name)
                self.plot_heatmap(name=img_name)
        else:
            df = pd.read_csv(self.data_path)
            for img_name in df["image_name"].unique():
                self.data = df[df["image_name"] == img_name].copy()
                csv_folder = os.path.dirname(self.data_path)
                self.frame_path = find_image_for_heat_map(csv_folder, img_name)
                self.plot_heatmap(name=img_name)
