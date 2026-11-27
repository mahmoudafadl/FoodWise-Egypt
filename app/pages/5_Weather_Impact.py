"""Weather Impact page for the FoodWise Egypt Streamlit dashboard."""
from pathlib import Path
import sys

import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import streamlit as st

APP_DIR = Path(__file__).resolve().parents[1]
if str(APP_DIR) not in sys.path:
    sys.path.insert(0, str(APP_DIR))

from utils import load_master_data, load_weather_regions

st.set_page_config(
    page_title="Weather Impact | FoodWise Egypt",
    page_icon="🌦️",
    layout="wide",
)

REGION_LABELS = {
    "Greater_Cairo": "Greater Cairo",
    "West_Delta_Beheira": "Beheira (West Delta)",
    "East_Delta_Sharqia": "Sharqia (East Delta)",
    "Upper_Egypt_Minya": "Minya (Upper Egypt)",
}

COMPARE_METRICS = {
    "Mean temperature (°C)": ("temp_mean", "mean", "°C"),
    "Total precipitation (mm)": ("precip_sum", "sum", "mm"),
    "Heatwave days (days)": ("heatwave_days", "sum", "days"),
    "Cold days (days)": ("cold_days", "sum", "days"),
    "Mean relative humidity (%)": ("humidity_mean", "mean", "%"),
    "Peak max temperature (°C)": ("temp_max", "max", "°C"),
}

WEATHER_VARS = {
    "Mean temperature (°C)": ("temp_mean", "°C"),
    "Max temperature (°C)": ("temp_max", "°C"),
    "Min temperature (°C)": ("temp_min", "°C"),
    "Total precipitation (mm)": ("precip_sum", "mm"),
    "Relative humidity (%)": ("humidity_mean", "%"),
    "Heatwave days": ("heatwave_days", "days"),
    "Cold days": ("cold_days", "days"),
}

PRICE_MEASURES = {
    "Price change (% MoM)": ("price_mom_pct", "% MoM"),
    "Price level (EGP)": ("price", "EGP"),
}


@st.cache_data(show_spinner=False)
def prepare_weather_data(raw_weather: pd.DataFrame) -> pd.DataFrame:
    """Add timestamp date column and display region labels to regional weather data."""
    w = raw_weather.copy()
    w["date"] = pd.to_datetime(dict(year=w["year"], month=w["month"], day=1))
    w["region_label"] = w["region"].map(REGION_LABELS).fillna(w["region"])
    return w.sort_values(["region", "date"])


@st.cache_data(show_spinner=False)
def prepare_price_data(raw_master: pd.DataFrame) -> pd.DataFrame:
    """Compute per-commodity MoM % change on national CAPMAS master prices."""
    df = raw_master.sort_values(["filter_id", "date"]).copy()
    df["price_mom_pct"] = df.groupby("filter_id")["price"].pct_change() * 100.0
    return df


def render_regional_comparison_chart(
    weather_window: pd.DataFrame,
    selected_region: str,
    compare_label: str,
) -> tuple[go.Figure, pd.DataFrame]:
    """Build a horizontal bar chart comparing all 4 regions over the selected period."""
    col, agg, unit = COMPARE_METRICS[compare_label]
    grouped = (
        weather_window.groupby(["region", "region_label"], as_index=False)
        .agg(
            temp_mean=("temp_mean", "mean"),
            precip_sum=("precip_sum", "sum"),
            heatwave_days=("heatwave_days", "sum"),
            cold_days=("cold_days", "sum"),
            humidity_mean=("humidity_mean", "mean"),
            temp_max=("temp_max", "max"),
        )
    )
    # Keep fixed, clear display order
    order_keys = list(REGION_LABELS.keys())
    grouped["order_idx"] = grouped["region"].apply(
        lambda r: order_keys.index(r) if r in order_keys else 99
    )
    grouped = grouped.sort_values("order_idx", ascending=False)

    vals = grouped[col].astype(float)
    colors = [
        "#1E4D3B" if r == selected_region else "#AFC2B5"
        for r in grouped["region"]
    ]
    text_fmt = ".0f" if unit == "days" else ".1f"

    fig = go.Figure(
        go.Bar(
            x=vals,
            y=grouped["region_label"],
            orientation="h",
            marker_color=colors,
            text=[f"{v:{text_fmt}} {unit}" for v in vals],
            textposition="outside",
            hovertemplate="<b>%{y}</b><br>" + f"{compare_label}: <b>%{{x:{text_fmt}}} {unit}</b><extra></extra>",
        )
    )
    max_x = max(float(vals.max()), 1.0)
    fig.update_layout(
        height=265,
        margin=dict(l=8, r=42, t=12, b=12),
        showlegend=False,
        xaxis=dict(title=compare_label, range=[0, max_x * 1.25], zeroline=True),
        yaxis=dict(title=None),
    )

    table_df = (
        grouped.sort_values("order_idx")[
            ["region_label", "temp_mean", "precip_sum", "heatwave_days", "cold_days", "humidity_mean"]
        ]
        .rename(
            columns={
                "region_label": "Region",
                "temp_mean": "Mean Temp (°C)",
                "precip_sum": "Total Precip (mm)",
                "heatwave_days": "Heatwave Days",
                "cold_days": "Cold Days",
                "humidity_mean": "Mean Humidity (%)",
            }
        )
    )
    return fig, table_df


def render_weather_time_chart(reg_weather: pd.DataFrame) -> go.Figure:
    """Render a dual-axis combo chart: temperature line + precipitation bars over time."""
    fig = make_subplots(specs=[[{"secondary_y": True}]])

    fig.add_trace(
        go.Bar(
            x=reg_weather["date"],
            y=reg_weather["precip_sum"],
            name="Precipitation (mm · Right axis)",
            marker_color="rgba(130, 155, 173, 0.55)",
            hovertemplate="<b>%{x|%b %Y}</b><br>Precipitation: <b>%{y:.1f} mm</b><extra></extra>",
        ),
        secondary_y=True,
    )

    fig.add_trace(
        go.Scatter(
            x=reg_weather["date"],
            y=reg_weather["temp_mean"],
            name="Mean temperature (°C · Left axis)",
            mode="lines+markers",
            line=dict(color="#1E4D3B", width=2.4),
            marker=dict(size=5, color="#1E4D3B"),
            hovertemplate="<b>%{x|%b %Y}</b><br>Mean temp: <b>%{y:.2f} °C</b><extra></extra>",
        ),
        secondary_y=False,
    )

    fig.update_layout(
        height=300,
        margin=dict(l=12, r=12, t=28, b=12),
        hovermode="x unified",
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="left", x=0),
        xaxis=dict(title=None, showgrid=False, tickformat="%b %Y"),
    )
    fig.update_yaxes(title_text="Temperature (°C)", secondary_y=False)
    fig.update_yaxes(title_text="Precipitation (mm)", secondary_y=True, showgrid=False)
    return fig


def render_weather_scatter(
    merged_df: pd.DataFrame,
    weather_label: str,
    price_label: str,
) -> tuple[go.Figure, float | None, int]:
    """Render a scatter plot of monthly weather values vs. national food price measure."""
    w_col, w_unit = WEATHER_VARS[weather_label]
    p_col, p_unit = PRICE_MEASURES[price_label]

    valid = merged_df.dropna(subset=[w_col, p_col]).copy()
    valid["month_str"] = valid["date"].dt.strftime("%b %Y")

    r_val: float | None = None
    if len(valid) >= 3 and valid[w_col].nunique() > 1 and valid[p_col].nunique() > 1:
        r_calc = float(valid[w_col].corr(valid[p_col]))
        if pd.notna(r_calc):
            r_val = r_calc

    fig = go.Figure(
        go.Scatter(
            x=valid[w_col],
            y=valid[p_col],
            mode="markers",
            marker=dict(size=8, color="#1E4D3B", opacity=0.78),
            customdata=valid["month_str"],
            hovertemplate=(
                "<b>%{customdata}</b><br>"
                f"{weather_label}: <b>%{{x:.2f}} {w_unit}</b><br>"
                f"{price_label}: <b>%{{y:.2f}} {p_unit}</b><extra></extra>"
            ),
        )
    )
    fig.update_layout(
        height=295,
        margin=dict(l=12, r=12, t=16, b=12),
        showlegend=False,
        xaxis=dict(title=weather_label),
        yaxis=dict(title=f"Food {price_label.lower()}"),
    )
    return fig, r_val, len(valid)


# ---------------------------------------------------------------------------
# Page Execution
# ---------------------------------------------------------------------------
raw_master = load_master_data()
raw_weather = load_weather_regions()

if raw_master.empty or raw_weather.empty:
    st.title("Weather Impact")
    st.warning(
        "Required datasets (`master_food_dataset.csv` and `weather_regions_monthly.csv`) were not found. "
        "Please run `python scripts/build_dataset.py` first."
    )
    st.stop()

weather_df = prepare_weather_data(raw_weather)
price_df = prepare_price_data(raw_master)

all_dates = [pd.Timestamp(d) for d in sorted(weather_df["date"].unique())]
earliest_date = all_dates[0]
latest_date = all_dates[-1]
default_1y_start = next(
    (d for d in all_dates if d >= latest_date - pd.DateOffset(months=11)),
    earliest_date,
)

region_keys = list(REGION_LABELS.keys())
all_commodities = sorted(price_df["filter_en"].unique().tolist())
default_commodity = (
    "Unpacked Wheat Flour 72%"
    if "Unpacked Wheat Flour 72%" in all_commodities
    else all_commodities[0]
)

if "wi_region" not in st.session_state or st.session_state["wi_region"] not in region_keys:
    st.session_state["wi_region"] = "Greater_Cairo"
if "wi_commodity" not in st.session_state or st.session_state["wi_commodity"] not in all_commodities:
    st.session_state["wi_commodity"] = default_commodity
if "wi_start" not in st.session_state or st.session_state["wi_start"] not in all_dates:
    st.session_state["wi_start"] = default_1y_start
if "wi_end" not in st.session_state or st.session_state["wi_end"] not in all_dates:
    st.session_state["wi_end"] = latest_date
if "wi_horizon" not in st.session_state:
    st.session_state["wi_horizon"] = "1 year"
if "wi_compare_by" not in st.session_state:
    st.session_state["wi_compare_by"] = "Mean temperature (°C)"
if "wi_scatter_weather" not in st.session_state:
    st.session_state["wi_scatter_weather"] = "Mean temperature (°C)"
if "wi_scatter_price" not in st.session_state:
    st.session_state["wi_scatter_price"] = "Price change (% MoM)"


def _reset_filters() -> None:
    st.session_state["wi_region"] = "Greater_Cairo"
    st.session_state["wi_commodity"] = default_commodity
    st.session_state["wi_start"] = default_1y_start
    st.session_state["wi_end"] = latest_date
    st.session_state["wi_horizon"] = "1 year"


def _apply_horizon() -> None:
    horizon = st.session_state.get("wi_horizon")
    end_dt = st.session_state.get("wi_end", latest_date)
    if horizon == "All":
        st.session_state["wi_start"] = earliest_date
        st.session_state["wi_end"] = latest_date
    elif horizon == "5 years":
        target_start = end_dt - pd.DateOffset(months=59)
        st.session_state["wi_start"] = next(
            (d for d in all_dates if d >= target_start), earliest_date
        )
    elif horizon == "1 year":
        target_start = end_dt - pd.DateOffset(months=11)
        st.session_state["wi_start"] = next(
            (d for d in all_dates if d >= target_start), earliest_date
        )


def _sync_manual_dates() -> None:
    if st.session_state["wi_start"] > st.session_state["wi_end"]:
        st.session_state["wi_end"] = st.session_state["wi_start"]
    st.session_state["wi_horizon"] = None


# Precompute merged dataset for export button
exp_weather = weather_df[
    (weather_df["region"] == st.session_state["wi_region"])
    & (weather_df["date"] >= st.session_state["wi_start"])
    & (weather_df["date"] <= st.session_state["wi_end"])
]
exp_price = price_df[price_df["filter_en"] == st.session_state["wi_commodity"]][
    ["year", "month", "filter_id", "filter_en", "category_en", "price", "price_mom_pct"]
]
exp_merged = pd.merge(exp_weather, exp_price, on=["year", "month"], how="left").sort_values("date")

# 1. Header
st.caption("FOOD PRICE INTELLIGENCE & EARLY WARNING SYSTEM")
hdr_l, hdr_r = st.columns([5, 1.2], vertical_alignment="center")
with hdr_l:
    st.title("Weather Impact")
    st.caption("Explore regional weather patterns and their relationship with food prices.")
with hdr_r:
    st.download_button(
        label="Export data",
        icon=":material/download:",
        data=exp_merged.to_csv(index=False).encode("utf-8"),
        file_name=(
            f"foodwise_weather_impact_{st.session_state['wi_region'].lower()}_"
            f"{st.session_state['wi_start'].strftime('%Y_%m')}_to_{st.session_state['wi_end'].strftime('%Y_%m')}.csv"
        ),
        mime="text/csv",
        width="stretch",
    )

# 2. Data Source Notice
st.info(
    f"Connected to regional Open-Meteo records (`weather_regions_monthly.csv`: 4 Egyptian regions, "
    f"{earliest_date.strftime('%b %Y')} – {latest_date.strftime('%b %Y')}) joined by `year` and `month` "
    "to national CAPMAS commodity prices (`master_food_dataset.csv`).",
    icon=":material/info:",
)

# 3. Filter Bar
with st.container(border=True):
    f1, f2, f3, f4, f5 = st.columns([1.4, 1.5, 1.1, 1.1, 0.85], vertical_alignment="bottom")
    with f1:
        selected_region: str = st.selectbox(
            "Region",
            options=region_keys,
            format_func=lambda r: REGION_LABELS.get(r, r),
            key="wi_region",
        )
    with f2:
        selected_commodity: str = st.selectbox(
            "Commodity",
            options=all_commodities,
            key="wi_commodity",
        )
    with f3:
        start_date: pd.Timestamp = st.selectbox(
            "Date range (from)",
            options=all_dates,
            format_func=lambda d: d.strftime("%B %Y"),
            key="wi_start",
            on_change=_sync_manual_dates,
        )
    with f4:
        end_date: pd.Timestamp = st.selectbox(
            "Date range (to)",
            options=all_dates,
            format_func=lambda d: d.strftime("%B %Y"),
            key="wi_end",
            on_change=_sync_manual_dates,
        )
    with f5:
        st.button("Reset filters", on_click=_reset_filters, width="stretch")

    c_l, c_r = st.columns([4.2, 1.0])
    with c_l:
        st.caption("Region filters weather only. The selected commodity uses the same national CAPMAS price series in every region.")
    with c_r:
        st.caption("Join: year + month")

selected_region_label = REGION_LABELS.get(selected_region, selected_region)

weather_window = weather_df[
    (weather_df["date"] >= start_date) & (weather_df["date"] <= end_date)
].copy()
reg_weather = weather_window[weather_window["region"] == selected_region].sort_values("date")
comm_price = price_df[price_df["filter_en"] == selected_commodity][
    ["year", "month", "filter_id", "filter_en", "category_en", "price", "price_mom_pct"]
]
merged_df = pd.merge(reg_weather, comm_price, on=["year", "month"], how="inner").sort_values("date")

n_months = len(reg_weather)

# 4. Section: <Selected Region> at a glance
gl_l, gl_r = st.columns([3.5, 1.5], vertical_alignment="bottom")
with gl_l:
    st.subheader(f"{selected_region_label} at a glance")
with gl_r:
    st.caption(f"{n_months} selected months · Real Open-Meteo & CAPMAS data")

if reg_weather.empty:
    st.warning("No weather observations found for the selected date range.")
    st.stop()

avg_temp = float(reg_weather["temp_mean"].mean())
total_precip = float(reg_weather["precip_sum"].sum())
total_heatwave = int(reg_weather["heatwave_days"].sum())

valid_price = merged_df.dropna(subset=["price"])
if not valid_price.empty and pd.notna(valid_price["price_mom_pct"].iloc[-1]):
    latest_mom = float(valid_price["price_mom_pct"].iloc[-1])
    latest_mom_str = f"{latest_mom:+.1f}%"
    latest_p_date = pd.Timestamp(valid_price["date"].iloc[-1]).strftime("%b %Y")
    mom_delta = f"{latest_p_date} · MoM · National series (not regional)"
else:
    latest_mom_str = "N/A"
    mom_delta = "No prior month in range"

k1, k2, k3, k4 = st.columns(4)
with k1:
    st.metric(
        label="Average Temperature",
        value=f"{avg_temp:.1f} °C",
        delta="Selected-period mean · Regional temp_mean",
        delta_color="off",
        delta_arrow="off",
        border=True,
        height="stretch",
    )
with k2:
    st.metric(
        label="Total Precipitation",
        value=f"{total_precip:.1f} mm",
        delta="Selected-period total · Regional precip_sum",
        delta_color="off",
        delta_arrow="off",
        border=True,
        height="stretch",
    )
with k3:
    st.metric(
        label="Heatwave Days",
        value=f"{total_heatwave} days",
        delta="Selected-period total · Regional heatwave_days",
        delta_color="off",
        delta_arrow="off",
        border=True,
        height="stretch",
    )
with k4:
    st.metric(
        label="Selected Commodity Price Change",
        value=latest_mom_str,
        delta=mom_delta,
        delta_color="off",
        delta_arrow="off",
        border=True,
        height="stretch",
    )

# 5. Middle Row: Regional weather comparison & Weather over time
mid_l, mid_r = st.columns([1.0, 1.35], gap="medium")

with mid_l:
    with st.container(border=True):
        rl, rr = st.columns([3.2, 1.3], vertical_alignment="center")
        with rl:
            st.subheader("Regional weather comparison")
            st.caption("Same period across all four regions")
        with rr:
            st.caption("FOUR REGIONS")

        compare_by = st.selectbox(
            "Compare by",
            options=list(COMPARE_METRICS.keys()),
            key="wi_compare_by",
        )
        st.caption(f"Highlighted region: **{selected_region_label}**")

        comp_fig, comp_table = render_regional_comparison_chart(
            weather_window=weather_window,
            selected_region=selected_region,
            compare_label=compare_by,
        )
        st.plotly_chart(comp_fig, width="stretch", config={"displayModeBar": False})

        with st.expander("View four-region comparison values"):
            st.dataframe(
                comp_table,
                hide_index=True,
                width="stretch",
                column_config={
                    "Mean Temp (°C)": st.column_config.NumberColumn("Mean Temp (°C)", format="%.2f"),
                    "Total Precip (mm)": st.column_config.NumberColumn("Total Precip (mm)", format="%.1f"),
                    "Mean Humidity (%)": st.column_config.NumberColumn("Mean Humidity (%)", format="%.1f"),
                },
            )
        st.caption(
            "Summary of monthly values over the selected period; weather records are not multiplied by commodity rows."
        )

with mid_r:
    with st.container(border=True):
        wt_l, wt_r = st.columns([2.6, 1.8], vertical_alignment="center")
        with wt_l:
            st.subheader("Weather over time")
            st.caption(f"{selected_region_label} · Temperature & precipitation")
        with wt_r:
            st.segmented_control(
                "Time window",
                options=["1 year", "5 years", "All"],
                selection_mode="single",
                key="wi_horizon",
                on_change=_apply_horizon,
                label_visibility="collapsed",
                width="stretch",
            )

        st.plotly_chart(
            render_weather_time_chart(reg_weather),
            width="stretch",
            config={"displayModeBar": False},
        )
        st.caption(
            "Regional source: Open-Meteo (`weather_regions_monthly.csv`). "
            "Temperature and precipitation use separate axes."
        )

# 6. Bottom Row: Weather & food price relationship + Weather-related observations
bot_l, bot_r = st.columns([1.3, 1.0], gap="medium")

with bot_l:
    with st.container(border=True):
        sl, sr = st.columns([3.5, 1.2], vertical_alignment="center")
        with sl:
            st.subheader("Weather & food price relationship")
            st.caption(f"{selected_region_label} weather · {selected_commodity} master prices")
        with sr:
            st.caption("SCATTER VIEW")

        sc1, sc2 = st.columns(2)
        with sc1:
            scatter_weather = st.selectbox(
                "Weather variable",
                options=list(WEATHER_VARS.keys()),
                key="wi_scatter_weather",
            )
        with sc2:
            scatter_price = st.selectbox(
                "Food price measure",
                options=list(PRICE_MEASURES.keys()),
                key="wi_scatter_price",
            )

        sc_fig, sc_r, sc_n = render_weather_scatter(
            merged_df=merged_df,
            weather_label=scatter_weather,
            price_label=scatter_price,
        )
        r_note = f"Pearson r = {sc_r:+.2f}" if sc_r is not None else "Pearson r = N/A"
        st.caption(f"{sc_n} monthly pairs · {r_note} · No causal trend line implied")

        if sc_n == 0:
            st.info("No matched monthly pairs available for the selected filters.")
        else:
            st.plotly_chart(sc_fig, width="stretch", config={"displayModeBar": False})

        st.caption(
            "Each point is one matched (year, month) pair. Co-occurrence is not causation; "
            "the price is a national CAPMAS series, not a regional retail quote."
        )

with bot_r:
    with st.container(border=True):
        ol, orr = st.columns([3.2, 1.3], vertical_alignment="center")
        with ol:
            st.subheader("Weather-related observations")
            st.caption("Descriptive insights computed from selected data")
        with orr:
            st.caption("DATA SUMMARY")

        hottest = comp_table.sort_values("Mean Temp (°C)", ascending=False).iloc[0]
        coolest = comp_table.sort_values("Mean Temp (°C)", ascending=True).iloc[0]
        wettest = comp_table.sort_values("Total Precip (mm)", ascending=False).iloc[0]
        driest = comp_table.sort_values("Total Precip (mm)", ascending=True).iloc[0]

        with st.container(border=True):
            st.markdown("**Regional temperature differences**")
            st.caption(
                f"In this period, **{hottest['Region']}** has the highest mean temperature at "
                f"**{hottest['Mean Temp (°C)']:.1f} °C**, compared with **{coolest['Region']}** at "
                f"**{coolest['Mean Temp (°C)']:.1f} °C**."
            )

        with st.container(border=True):
            st.markdown("**Rainfall varies across regions**")
            st.caption(
                f"Cumulative precipitation ranges from **{driest['Total Precip (mm)']:.1f} mm** "
                f"({driest['Region']}) to **{wettest['Total Precip (mm)']:.1f} mm** ({wettest['Region']}). "
                "Rainfall differences alone do not establish a food price effect."
            )

        with st.container(border=True):
            st.markdown("**Keep the wider context in view**")
            r_phrase = f"r = {sc_r:+.2f}" if sc_r is not None else "insufficient valid pairs"
            st.caption(
                f"Across {sc_n} matched months, **{selected_commodity}** ({scatter_price.lower()}) shows "
                f"**{r_phrase}** with **{selected_region_label}** {scatter_weather.lower()}. "
                "Weather co-occurs with seasonality, exchange rates, and supply cycles; no direct weather effect is estimated."
            )

        st.caption(f"{len(merged_df)} monthly matches for the selected region and commodity.")

# 7. Matched Data & Calculation Notes Expanders
with st.expander("View matched regional weather & master price data"):
    disp = merged_df[
        [
            "year",
            "month",
            "region_label",
            "temp_mean",
            "temp_max",
            "temp_min",
            "precip_sum",
            "humidity_mean",
            "heatwave_days",
            "cold_days",
            "filter_en",
            "price",
            "price_mom_pct",
        ]
    ].copy()
    disp = disp.rename(
        columns={
            "year": "Year",
            "month": "Month",
            "region_label": "Region",
            "temp_mean": "Mean Temp (°C)",
            "temp_max": "Max Temp (°C)",
            "temp_min": "Min Temp (°C)",
            "precip_sum": "Precip (mm)",
            "humidity_mean": "Humidity (%)",
            "heatwave_days": "Heatwave Days",
            "cold_days": "Cold Days",
            "filter_en": "Commodity",
            "price": "Price (EGP)",
            "price_mom_pct": "MoM Change (%)",
        }
    )
    st.dataframe(disp, hide_index=True, width="stretch")

with st.expander("Regional data & calculation notes — Join keys, aggregation & limitations"):
    st.markdown(
        """
        - **Weather Source (`data/processed/weather_regions_monthly.csv`)**: Monthly Open-Meteo climate aggregates (`temp_mean`, `temp_max`, `temp_min`, `precip_sum`, `humidity_mean`, `heatwave_days`, `cold_days`) for 4 regions: `Greater_Cairo`, `West_Delta_Beheira`, `East_Delta_Sharqia`, and `Upper_Egypt_Minya` (**Jan 2015 – Jun 2026**).
        - **Price Join (`data/processed/master_food_dataset.csv`)**: Joined on `year` and `month` to national CAPMAS retail commodity prices (`price`).
        - **Limitations**: CAPMAS prices in `master_food_dataset.csv` are national monthly averages rather than governorate-level prices; therefore changing `Region` updates regional climate metrics and charts while the commodity price series remains national. Scatter correlations are exploratory and do not isolate weather effects from seasonality or macro shocks.
        """
    )

st.caption("FoodWise Egypt · Turning market data into food security insights.")
