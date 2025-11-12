import os
import pandas as pd
import numpy as np
from plotnine import (
    ggplot, aes, geom_boxplot, geom_jitter, geom_errorbar,
    geom_point, geom_line, theme_classic, labs, scale_x_continuous
)
from .base_module import BaseModule
from src.utils.common import create_output_dir
from src.utils.common_analyze import fill_0_values, assign_intervals,save_and_rename,check_groupby_dupication, check_self_treastment_col



class Visits(BaseModule):
    """Compute average visits (unique tracks or boxes) per time interval."""
    name = 'average_visits'

    def __init__(self, config):
        super().__init__(config)
        s = config['settings']
        self.dir = config['output_dir']
        
        self.fps = float(s['fps'])
        self.interval = float(s['time_intervals'])
        self.unit = s.get('interval_unit', 'minutes')
        self.filter_max = s.get('filter_time_intervals', None)
        self.unit   = s.get('interval_unit', 'minutes')
        self.data_path = config['input_csv']
        self.treatment_col = s.get('treatment_or_image_name', 'treatment')
        self.use_track = s.get('use_track_id', True)
        self.l = check_groupby_dupication(self.treatment_col)
        self.stat =  s.get('stat','mean')
        

    def compute(self, df: pd.DataFrame = None) -> pd.DataFrame:
        if df is None:
            df = pd.read_csv(self.data_path)
            df = check_self_treastment_col(df,self)
        if self.treatment_col not in df.columns:
            raise KeyError(f"Column '{self.treatment_col}' missing in data.")
        


        # assign time intervals
        df = assign_intervals(df, 'image_idx', self.fps, self.interval, self.unit)
        # compute raw counts
        if self.use_track:
            df['track_id'] = pd.to_numeric(df['track_id'], errors='coerce')
            df = df.dropna(subset=['track_id'])
            df_raw = (
                df.groupby(self.l)
                  .agg(value=('track_id', 'nunique'))
                  .reset_index()
            )
        else:
            df['box_idx'] = pd.to_numeric(df['box_idx'], errors='coerce')
            df = df.dropna(subset=['box_idx'])
            df_raw = ( 
                df.groupby(['time_interval', self.treatment_col])
                  .agg(value=('box_idx', 'nunique'))
                  .reset_index()
            )
        df_raw = fill_0_values(df_raw)
        if self.filter_max is not None:
            df_raw = df_raw[df_raw['time_interval'] <= self.filter_max]
        
        return df_raw

    def summarize(self, df_raw: pd.DataFrame) -> pd.DataFrame:
        summary = (
            df_raw.groupby([self.treatment_col, 'time_interval'])['value']
                  .agg(['mean', 'std', 'count'])
                  .reset_index()
        )
        summary['se'] = summary['std'] / np.sqrt(summary['count'])
        summary['upper'] = summary['mean'] + summary['se']
        summary['lower'] = summary['mean'] - summary['se']
        return summary

    def plot(self, df_box, df_time) -> None:
        
        #check comman issues
        create_output_dir(self.plot_path)

        # box plot
        p1 = (
            ggplot(df_box, aes(x=self.treatment_col, y='value', fill=self.treatment_col))
            + geom_boxplot(outlier_alpha=0.4)
            + geom_jitter()
            + labs(x='', y='Visits', title='Visits by Treatment')
            + theme_classic()
        )
        p1.save(os.path.join(self.plot_path, 'visits_box.jpg'))

        # time series
        p2 = (
            ggplot(df_time, aes(x='time_interval', y='mean', color=self.treatment_col))
            + geom_point()
            + geom_line()
            + theme_classic()
            + labs(x=f'Time ({self.unit})', y='Visits', title='Visits Over Time')
            + scale_x_continuous(limits=[0, df_time['time_interval'].max()])
        )
        # only add SE errorbars if we're plotting by treatment
        if self.treatment_col == 'treatment':
            p2 += geom_errorbar(
                df_time,
                aes(x='time_interval', ymin='lower', ymax='upper', color=self.treatment_col),
                width=0.2
            )
        p2.save(os.path.join(self.plot_path, 'visits_time.jpg'))

        save_and_rename(df_box,df_time,self.dir,'visits')