# src/analyze/modules/distance.py

import os
import pandas as pd
import numpy as np
from plotnine import (
    ggplot, aes, geom_boxplot, geom_jitter, geom_errorbar, geom_point, geom_line,
    theme_classic, labs, scale_x_continuous
)
from .base_module import BaseModule
from src.utils.common_analyze import fill_0_values, check_self_treastment_col, assign_intervals, save_and_rename,check_groupby_dupication
from src.utils.common import create_output_dir

class Distance(BaseModule):
    """Compute total pixel–distance per time interval."""
    name = 'distance'

    def __init__(self, config):
        super().__init__(config)
        s = config['settings']
        self.fps = float(s['fps'])
        self.dir = config['output_dir']
        self.interval = float(s['time_intervals'])
        self.unit   = s.get('interval_unit', 'minutes')
        self.filter_max = s.get('filter_time_intervals', None)
        self.data_path     = config['input_csv']
        self.treatment_col = s.get('treatment_or_image_name', 'treatment')
        self.l = check_groupby_dupication(self.treatment_col)
        self.stat =  s.get('stat','mean')

    def compute(self, df: pd.DataFrame = None) -> pd.DataFrame:
        # 1) load
        if df is None:
            df = pd.read_csv(self.data_path)
            df = check_self_treastment_col(df,self)
        # 2) bin into time intervals
        df = assign_intervals(df, 'image_idx', self.fps, self.interval, self.unit)
        # 3) compute per-track pixel‐distance
        for col in ['track_id','image_idx','x','y']:
            df[col] = pd.to_numeric(df[col], errors='coerce')
        df = df.dropna(subset=['track_id','image_idx','x','y'])
        df = df.sort_values(['track_id','image_name','image_idx'])
        df['dx'] = df.groupby(['track_id','image_name'])['x'].diff()
        df['dy'] = df.groupby(['track_id','image_name'])['y'].diff()
        df['dist_px'] = np.sqrt(df['dx']**2 + df['dy']**2)
        # 4) collapse to per-interval
        df_raw = (
            df.groupby(self.l)
              .agg(value=('dist_px','sum'))
              .reset_index()
        )
        # 5) fill zeros & filter
        df_raw = fill_0_values(df_raw)
        if self.filter_max is not None:
            df_raw = df_raw[df_raw['time_interval'] <= self.filter_max]
        return df_raw

    def summarize(self, df_raw: pd.DataFrame) -> pd.DataFrame:
        summary = (
            df_raw
              .groupby([self.treatment_col, 'time_interval'])['value']
              .agg(['mean','std','count'])
              .reset_index()
        )
        summary['se']    = summary['std'] / np.sqrt(summary['count'])
        summary['upper'] = summary['mean'] + summary['se']
        summary['lower'] = summary['mean'] - summary['se']
        return summary

    def plot(self, df_box, df_time) -> None:
        
        #check comman issues
        create_output_dir(self.plot_path)

        # boxplot of summary means
        p1 = (
            ggplot(df_box, aes(x=self.treatment_col, y='value', fill=self.treatment_col))
            + geom_boxplot(outlier_alpha=0.4)
            + geom_jitter(width=0.2)
            + theme_classic()
            + labs(x='', y='Distance', title='Distance by Treatment')
        )
        p1.save(os.path.join(self.plot_path,'distance_box.jpg'))

        # time-series plot
        p2 = (
            ggplot(df_time, aes(x='time_interval', y='mean', color=self.treatment_col))
            + geom_point()
            + geom_line()
            + theme_classic()
            + labs(x=f'Time ({self.unit})', y='Distance', title='Distance Over Time')
            + scale_x_continuous(limits=[0, df_time['time_interval'].max()])
        )
        if self.treatment_col == 'treatment':
            p2 += geom_errorbar(
                df_time,
                aes(x='time_interval', ymin='lower', ymax='upper'),
                width=0.2
            )
        p2.save(os.path.join(self.plot_path,'distance_time.jpg'))
        
        save_and_rename(df_box,df_time,self.dir,'distance')

        