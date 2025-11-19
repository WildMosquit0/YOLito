from abc import ABC, abstractmethod
import pandas as pd

class BaseModule(ABC):
    """
    Abstract base class for all analysis modules.
    Each module must implement compute(), summarize(), and plot().
    """
    def __init__(self, config: dict):
        self.config = config
        self.output_dir = config['output_dir']
        self.plot_path = f"{self.output_dir}/plots"

    @abstractmethod
    def compute(self, df: pd.DataFrame = None) -> pd.DataFrame:
        """
        Load and/or process the raw data into per‐interval results.
        Returns a DataFrame with at least ['time_interval', treatment_col, 'value'].
        """
        pass

    @abstractmethod
    def summarize(self, df_raw: pd.DataFrame) -> pd.DataFrame:
        """
        Aggregate df_raw into summary statistics per interval and treatment.
        Returns a DataFrame with ['time_interval', treatment_col, 'mean', 'std', 'count', 'se', 'upper', 'lower'].
        """
        pass

    @abstractmethod
    def plot(self, df_summary: pd.DataFrame) -> None:
        """
        Create and save plots based on df_summary.
        """
        pass