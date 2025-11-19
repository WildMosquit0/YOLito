
import os
import pandas as pd
from plotnine import ggplot, aes, geom_segment, theme_classic, labs, scale_color_gradient, scale_y_continuous, scale_x_continuous


def create_output_dir(dir_path):
    if not os.path.exists(dir_path):
        os.makedirs(dir_path)

class TrajectoryDarknessPlot:
    def __init__(self, config):
        self.config = config
        self.input_csv = self.config["input_csv"]
        self.output_dir = self.config["output_dir"]
        self.plot_path = f"{self.output_dir}/plots"
        # Retrieve fps from config under "trajectory_darkness", defaulting to 30 if not provided.
        self.fps = float(self.config["trajectory_darkness"].get("fps", 30))
        self.time_unit = self.config["interval_unit"]

    def _load_data(self):
        # Load CSV and ensure 'image_idx' is numeric.
        df = pd.read_csv(self.input_csv)
        df['image_idx'] = pd.to_numeric(df['image_idx'], errors='coerce')
        df = df.dropna(subset=['image_idx'])
        # Ensure required columns exist.
        if 'image_name' not in df.columns:
            raise KeyError("The column 'image_name' is missing from the input CSV.")
        if 'box_idx' not in df.columns:
            raise KeyError("The column 'box_idx' is missing from the input CSV.")
        return df

    def _process_data(self, df):
        # Group by frame (image_idx) and count unique trajectories using 'box_idx'
        traj_counts = (
            df.groupby('image_idx')['box_idx']
              .nunique()
              .reset_index(name='count')
              .sort_values('image_idx')
        )
        # Convert frame numbers to time (in seconds)
        traj_counts['time'] = traj_counts['image_idx'] / (self.fps * 60)
        traj_counts = traj_counts.reset_index(drop=True)
        return traj_counts

    def _create_segments(self, traj_counts):
        # Ensure there are at least two data points to create segments.
        if len(traj_counts) < 2:
            raise ValueError("Not enough data points to create segments for plotting.")
        # Create segments from consecutive time points.
        segments = pd.DataFrame({
            'time_start': traj_counts['time'].iloc[:-1].values,
            'time_end': traj_counts['time'].iloc[1:].values,
            # Average the counts of adjacent frames for color mapping.
            'count': (traj_counts['count'].iloc[:-1].values + traj_counts['count'].iloc[1:].values) / 2.0,
            'y': 1  # Constant y-value for all segments.
        })
        return segments

    def _plot(self, segments, global_max, image_name):
        # Build the plot: each segment's color reflects its trajectory count on a grayscale gradient.
        # The gradient is fixed with limits from 0 (white) to global_max (black).
        p = (ggplot(segments, aes(x='time_start', xend='time_end', y='y', yend='y', color='count'))
            + geom_segment(size=60)  # Very thick line segments
            + scale_color_gradient(low="white", high="black", limits=(0, global_max))
            + theme_classic()
            + labs(title=f"{image_name}_TDP",
                    x=f"Time ({self.time_unit})",
                    y="",
                    color="Trajectory Count")
            + scale_y_continuous(breaks=[], expand=(0, 0), limits=(0.9, 1.1))  # No y-axis labels and fixed y-scale
            + scale_x_continuous(breaks=range(0, 28, 3), expand=(0, 0), limits=(0, 27))
        )
        return p

    def __call__(self):
        # Load the full dataset.
        df = self._load_data()
        
        # Compute the global maximum trajectory count across all images.
        global_counts = (
            df.groupby(['image_name', 'image_idx'])["box_idx"]
              .nunique()
              .reset_index(name='count')
        )
        global_max = global_counts['count'].max()

        # Create the output plot directory if needed.
        create_output_dir(self.plot_path)
        
        # Iterate over each unique image_name.
        unique_images = df['image_name'].unique()
        for image in unique_images:
            # Filter the dataframe for the current image.
            df_image = df[df['image_name'] == image].copy()
            # Process data: count trajectories per frame and compute time.
            traj_counts = self._process_data(df_image)
            # Create segments for plotting.
            segments = self._create_segments(traj_counts)
            # Generate the plot.
            plot = self._plot(segments, global_max,image)
            # Construct output file name.
            output_file = os.path.join(self.plot_path, f"{image}_TDP.jpg")
            # Save the plot.
            plot.save(output_file)
        return "Plots generated successfully."


config = {
    "input_csv":  "/home/wildmosquit0/workspace/projects/Arad/data/results.csv",
    "output_dir": "/home/wildmosquit0/workspace/projects/Arad/test",
    "trajectory_darkness": {
        "fps": 15
    }
}
plot_generator = TrajectoryDarknessPlot(config)
result = plot_generator()
print(result)