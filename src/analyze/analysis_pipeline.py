from src.analyze.modules.traj_explorer import PlotXY
from src.analyze.modules.visits import Visits
from src.analyze.modules.duration import Duration
from src.analyze.modules.distance import Distance
from src.analyze.modules.heatmap import Heatmap
from src.utils.common import data_merger
from src.utils.config_ops import update_yaml
from src.utils.config_ops import export_config

import os

def run_analysis(config,conf_yaml_path):

    export_config(conf_yaml_path)
    if os.path.isdir(config['input_csv']):
            merged_data_path = data_merger(config['input_csv'])
            config['input_csv'] = merged_data_path
            update_yaml(config, conf_yaml_path,'input_csv')
            
       # Run enabled tasks
    if config['task'].get('visits', False):
       print("Running Average Visits analysis...")
       visits = Visits(config)
       df_visits = visits.compute()                   
       sum_visits = visits.summarize(df_visits)       
       visits.plot(df_visits,sum_visits)                         
    
    if config['task']['duration']:
        print("Running Duration analysis...")
        dur = Duration(config)
        df_dur = dur.compute()
        sum_dur = dur.summarize(df_dur)
        dur.plot(df_dur,sum_dur)
    
    if config['task'].get('distance', False):
        print("Running Distance Traveled analysis...")
        dist = Distance(config)
        df_dist = dist.compute()
        sum_dist = dist.summarize(df_dist)
        dist.plot(df_dist,sum_dist)
    
    if config['task']['heatmap']:
        print("Running Heatmap analysis...")
        heatmap = Heatmap(config)
        heatmap()
    
    if config['task']['plotxy']:
        print("Running PlotXY analysis...")
        plotxy = PlotXY(config)
        plotxy()
    
