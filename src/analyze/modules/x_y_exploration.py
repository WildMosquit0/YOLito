import cv2
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import yaml

class x_y_exploration(object):
    def __init__(self, datapath="path from config"):
        self.data = pd.read_csv(datapath)

    def get_x_y_coordinates(self):
        """Extract x1, y1, x2, and y2 coordinates from the dataset."""
        x1 = self.data['x1'].tolist()
        y1 = self.data['y1'].tolist()
        x2 = self.data['x2'].tolist()
        y2 = self.data['y2'].tolist()
        return x1, x2, y1, y2

    def calculate_average(self, x1, y1, x2, y2):
        """Calculate the midpoint average of x and y coordinates."""
        x = [(x1[i] + x2[i]) / 2 for i in range(len(x1))]
        y = [(y1[i] + y2[i]) / 2 for i in range(len(y1))]
        return x, y

    def plot_and_save_coordinates(self, x, y, output_path="plot.png", xlim=None, ylim=None):
        """Plot and save coordinates as a scatter plot with optional limits."""
        plt.scatter(x, y)
        plt.xlabel("X")
        plt.ylabel("Y")

        # Apply axis limits if provided
        if xlim:
            plt.xlim(xlim)
        if ylim:
            plt.ylim(ylim)

        plt.savefig(output_path)
        plt.close()

    def save_vectors(self, x, y, output_path):
        """Save the calculated x and y vectors to a CSV file."""
        vectors_df = pd.DataFrame({'x': x, 'y': y})
        vectors_df.to_csv(output_path, index=False)

    @classmethod
    def main(cls, config_path="config.yaml"):
        """Main function to execute all steps based on the config file."""
        # Load configuration
        with open(config_path, "r") as config_file:
            config = yaml.safe_load(config_file)
        
        # Retrieve paths from config
        datapath = config['data_path']
        plot_output_path = config['plot_output_path']
        vector_output_path = config['vector_output_path']
        xlim = config.get('xlim', None)  # Get xlim from config, default to None if not provided
        ylim = config.get('ylim', None)  # Get ylim from config, default to None if not provided

        # Initialize class with the dataset path
        exploration = cls(datapath)

        # Get coordinates and calculate midpoints
        x1, x2, y1, y2 = exploration.get_x_y_coordinates()
        mid_x, mid_y = exploration.calculate_average(x1, y1, x2, y2)

        # Plot and save coordinates with axis limits if provided
        exploration.plot_and_save_coordinates(mid_x, mid_y, plot_output_path, xlim=xlim, ylim=ylim)

        # Save vectors to a CSV file
        exploration.save_vectors(mid_x, mid_y, vector_output_path)
