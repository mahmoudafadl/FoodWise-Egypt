"""Feature Engineering module shared between model training and live app inference."""
import numpy as np
import pandas as pd


def build_features(df: pd.DataFrame, is_inference: bool = False) -> pd.DataFrame:
    """
    Generate time-series lags, rolling volatility, macro pass-through lags,
    weather lags, cyclical month encoding (sin/cos), and Hijri seasonal flags.
    """
    data = df.sort_values(["filter_id", "date"]).copy()

    # TODO (Member 1): Add MoM/YoY lags, rolling stats, macro lags, weather lags, sin/cos, Ramadan/Eid flags
    if not is_inference:
        data["target_mom_pct"] = (
            data.groupby("filter_id")["price"].pct_change().shift(-1) * 100
        )

    return data
