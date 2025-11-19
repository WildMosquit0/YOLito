
import pandas as pd
import numpy as np

def assign_intervals(
    df: pd.DataFrame,
    idx_col: str,
    fps: float,
    interval_length: float,
    unit: str = "minutes"
) -> pd.DataFrame:
    conversion = 60 if unit == "minutes" else 1
    df[idx_col] = pd.to_numeric(df[idx_col], errors="coerce")
    df = df.dropna(subset=[idx_col])
    step_secs = fps * interval_length * conversion
    df["time_interval"] = np.floor(df[idx_col] / step_secs) * interval_length
    return df



def fill_0_values(
    df: pd.DataFrame,
    time_col: str = 'time_interval',
    value_col: str = 'value'
) -> pd.DataFrame:
    
    # 1. Identify your grouping keys (everything except time & value)
    group_cols = [c for c in df.columns if c not in [time_col, value_col]]

    # 2. Determine the sorted list of unique intervals & step size
    intervals = sorted(df[time_col].unique())
    if len(intervals) > 1:
        step = intervals[1] - intervals[0]
    else:
        step = intervals[0] if intervals else 1
    min_int, max_int = intervals[0], intervals[-1]

    # 3. Build the full set of intervals

    try:
        all_intervals = np.arange(min_int, max_int + step, step)
    except ValueError:
        raise ValueError(
        f"Invalid interval settings: cannot compute intervals from min={min_int} to max={max_int} with step={step}. "
        f"Please check that 'fps', 'interval', and 'interval_unit' are correctly defined in your analyze config."
    )

    # 4. Get every unique combo of the other grouping columns
    combos = df[group_cols].drop_duplicates()

    # 5. Build a complete grid DataFrame
    rows = []
    for _, combo in combos.iterrows():
        for t in all_intervals:
            row = {time_col: t}
            for col in group_cols:
                row[col] = combo[col]
            rows.append(row)
    full_grid = pd.DataFrame(rows)

    # 6. Merge your original data onto that grid, filling missing with 0
    merged = full_grid.merge(df, on=group_cols + [time_col], how='left')
    merged[value_col] = merged[value_col].fillna(0)

    return merged

import os

def save_and_rename(
    df_box: pd.DataFrame,
    df_time: pd.DataFrame,
    output_dir: str,
    metric_name: str
):
    # 1) Rename the mean column to your metric
    df_box = df_box.rename(columns={'value': metric_name})
    df_time = df_time.rename(columns={'value': metric_name})

    # 2) Ensure the output directory exists
    os.makedirs(output_dir, exist_ok=True)

    # 3) Save CSVs with the metric name in the filename
    box_path  = os.path.join(output_dir, f"{metric_name}_box.csv")
    time_path = os.path.join(output_dir, f"{metric_name}_time.csv")

    df_box.to_csv(box_path, index=False)
    df_time.to_csv(time_path, index=False)

def check_groupby_dupication(col_conf):
    if "image_name" == col_conf:
        l = ['time_interval','image_name']
    else:
        l = ['time_interval','image_name', 'treatment']
    return l     

def check_groupby_dupication_duration(col_conf):
    if "image_name" == col_conf:
        l = ['time_interval','image_name','track_id']
    else:
        l = ['time_interval','image_name', 'treatment','track_id']
    return l


def check_self_treastment_col(df,self):

    if not pd.api.types.is_string_dtype(df[self.treatment_col]):
        
        df[self.treatment_col] = df[self.treatment_col].astype(str)



    return df