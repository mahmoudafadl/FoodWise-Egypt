import calendar
import numpy as np
import pandas as pd

from hijridate import Gregorian


def add_hijri_flags(data):
    """Add monthly Ramadan and Eid flags using the Hijri calendar."""

    unique_dates = data["date"].drop_duplicates()

    hijri_flags = {}

    for date in unique_dates:
        days_in_month = calendar.monthrange(
            date.year, date.month
        )[1]

        is_ramadan = False
        is_eid_fitr = False
        is_eid_adha = False

        for day in range(1, days_in_month + 1):
            hijri = Gregorian(
                date.year, date.month, day
            ).to_hijri()

            if hijri.month == 9:
                is_ramadan = True

            if hijri.month == 10 and hijri.day in [1, 2, 3]:
                is_eid_fitr = True

            if hijri.month == 12 and hijri.day in [10, 11, 12, 13]:
                is_eid_adha = True

        hijri_flags[date] = {
            "is_ramadan": int(is_ramadan),
            "is_eid_fitr": int(is_eid_fitr),
            "is_eid_adha": int(is_eid_adha)
        }

    flags_df = pd.DataFrame.from_dict(
        hijri_flags, orient="index"
    )

    flags_df.index.name = "date"

    data = data.merge(
        flags_df,
        left_on="date",
        right_index=True,
        how="left"
    )

    return data


def build_features(df: pd.DataFrame, is_inference=False):

    data = df.copy()

    data["date"] = pd.to_datetime(data["date"])

    data = data.sort_values(
        ["filter_id", "date"]
    ).reset_index(drop=True)

    grouped = data.groupby("filter_id", sort=False)

    # --------------------------------
    # 1. Price Lags
    # --------------------------------

    for lag in [1, 2, 3, 6, 12]:
        data[f"price_lag_{lag}"] = (
            grouped["price"].shift(lag)
        )

    # --------------------------------
    # 2. Monthly and Yearly Changes
    # --------------------------------

    data["price_mom_pct"] = (
        grouped["price"].pct_change() * 100
    )

    data["price_yoy_pct"] = (
        grouped["price"].pct_change(12) * 100
    )

    for lag in [1, 2, 3, 6, 12]:
        data[f"mom_lag_{lag}"] = (
            data.groupby("filter_id")["price_mom_pct"]
            .shift(lag)
        )

        data[f"yoy_lag_{lag}"] = (
            data.groupby("filter_id")["price_yoy_pct"]
            .shift(lag)
        )

    # --------------------------------
    # 3. Rolling Statistics
    # --------------------------------

    for window in [3, 6, 12]:

        grouped_mom = data.groupby(
            "filter_id"
        )["price_mom_pct"]

        data[f"rolling_mean_{window}"] = (
            grouped_mom.transform(
                lambda x: x.rolling(
                    window, min_periods=2
                ).mean()
            )
        )

        data[f"rolling_std_{window}"] = (
            grouped_mom.transform(
                lambda x: x.rolling(
                    window, min_periods=2
                ).std()
            )
        )

    # --------------------------------
    # 4. Macro Features and Lags
    # --------------------------------

    macro_columns = [
        "usd_buy_mean",
        "usd_sell_mean",
        "usd_sell_end",
        "usd_volatility",
        "usd_mom_pct",
        "headline_inflation_mom",
        "core_inflation_mom",
        "regulated_inflation_mom",
        "veg_fruit_inflation_mom",
        "cbe_deposit_rate",
        "cbe_lending_rate"
    ]

    for col in macro_columns:
        if col in data.columns:

            for lag in [1, 2, 3]:
                data[f"{col}_lag_{lag}"] = (
                    data.groupby("filter_id")[col]
                    .shift(lag)
                )

    # --------------------------------
    # 5. Weather Features and Lags
    # --------------------------------

    weather_columns = [
        "cairo_temp_mean",
        "nat_agro_temp_mean",
        "nat_heatwave_days_avg",
        "nat_cold_days_avg",
        "nat_precip_total",
        "nat_temp_anomaly"
    ]

    for col in weather_columns:
        if col in data.columns:

            for lag in [1, 2, 3]:
                data[f"{col}_lag_{lag}"] = (
                    data.groupby("filter_id")[col]
                    .shift(lag)
                )

    # --------------------------------
    # 6. Cyclical Month Encoding
    # --------------------------------

    data["month_sin"] = np.sin(
        2 * np.pi * data["month"] / 12
    )

    data["month_cos"] = np.cos(
        2 * np.pi * data["month"] / 12
    )

    # --------------------------------
    # 7. Hijri Seasonal Flags
    # --------------------------------

    data = add_hijri_flags(data)

    # --------------------------------
    # 8. Target: Next Month Price Change
    # --------------------------------

    if not is_inference:

        next_price = (
            data.groupby("filter_id")["price"]
            .shift(-1)
        )

        data["target_mom_pct"] = (
            (next_price / data["price"] - 1) * 100
        )

    return data
