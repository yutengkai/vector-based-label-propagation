from __future__ import annotations
import pandas as pd

def summarize_results(df: pd.DataFrame):
    return df.groupby(["dataset","fraction","method"]).agg(time_s=("time_s","mean")).reset_index()
