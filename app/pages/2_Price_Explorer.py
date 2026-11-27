"""Price Explorer page for the FoodWise Egypt Streamlit dashboard."""
from pathlib import Path
import sys

import pandas as pd
import plotly.graph_objects as go
import streamlit as st

APP_DIR = Path(__file__).resolve().parents[1]
if str(APP_DIR) not in sys.path:
    sys.path.insert(0, str(APP_DIR))

from utils import load_master_data

st.set_page_config(
    page_title="Price Explorer | FoodWise Egypt",
    page_icon="📈",
    layout="wide",
)

PALETTE = ["#1E4D3B", "#2F6FED", "#D97706", "#B42318", "#7C3AED"]


@st.cache_data(show_spinner=False)
def prepare_explorer_data(raw_df: pd.DataFrame) -> pd.DataFrame:
    """Sort observations chronologically and compute per-commodity MoM % changes."""
    df = raw_df.sort_values(["filter_id", "date"]).copy()
    df["prev_price"] = df.groupby("filter_id")["price"].shift(1)
    df["mom_pct"] = df.groupby("filter_id")["price"].pct_change() * 100.0
    return df


def render_price_chart(plot_df: pd.DataFrame, commodities: list[str]) -> go.Figure:
    """Build a Plotly line chart of historical monthly prices (EGP) for 1–5 commodities."""
    fig = go.Figure()
    for idx, item in enumerate(commodities):
        sub = plot_df[plot_df["filter_en"] == item].sort_values("date")
        if sub.empty:
            continue
        color = PALETTE[idx % len(PALETTE)]
        fig.add_trace(
            go.Scatter(
                x=sub["date"],
                y=sub["price"],
                name=f"{item} (EGP)",
                mode="lines",
                line=dict(color=color, width=2.4),
                hovertemplate=(
                    f"<b>{item}</b><br>"
                    "%{x|%b %Y}: <b>%{y:.2f} EGP</b><extra></extra>"
                ),
            )
        )

    fig.update_layout(
        height=340,
        margin=dict(l=12, r=12, t=28, b=12),
        hovermode="x unified",
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="left", x=0),
        xaxis=dict(title=None, showgrid=False, tickformat="%b %Y"),
        yaxis=dict(title="Price (EGP)", rangemode="tozero", zeroline=True),
    )
    return fig


def get_macro_highlights(
    df: pd.DataFrame,
    start_date: pd.Timestamp,
    end_date: pd.Timestamp,
) -> list[dict]:
    """
    Extract the two largest observed CBE macro shifts within [start_date, end_date].

    Uses actual dataset columns (usd_sell_end, usd_mom_pct, headline_inflation_mom,
    cbe_lending_rate); no external event-log table exists in master_food_dataset.csv.
    """
    macro = (
        df[(df["date"] >= start_date) & (df["date"] <= end_date)][
            [
                "date",
                "usd_sell_end",
                "usd_mom_pct",
                "headline_inflation_mom",
                "cbe_lending_rate",
            ]
        ]
        .drop_duplicates("date")
        .sort_values("date")
        .copy()
    )
    if macro.empty:
        return []

    macro["fx_abs"] = macro["usd_mom_pct"].abs().fillna(0.0)
    macro["cpi_abs"] = macro["headline_inflation_mom"].abs().fillna(0.0)
    macro["macro_score"] = macro["fx_abs"] + (macro["cpi_abs"] * 3.0)

    top_rows = macro.sort_values("macro_score", ascending=False).head(2).sort_values("date")
    highlights: list[dict] = []
    for _, row in top_rows.iterrows():
        dt = pd.Timestamp(row["date"])
        usd_mom = float(row["usd_mom_pct"]) if pd.notna(row["usd_mom_pct"]) else 0.0
        cpi_mom = float(row["headline_inflation_mom"]) if pd.notna(row["headline_inflation_mom"]) else 0.0
        usd_end = float(row["usd_sell_end"]) if pd.notna(row["usd_sell_end"]) else 0.0
        rate = float(row["cbe_lending_rate"]) if pd.notna(row["cbe_lending_rate"]) else 0.0

        if abs(usd_mom) >= 5.0:
            title = f"FX shift: USD/EGP {usd_mom:+.1f}% MoM"
        elif abs(cpi_mom) >= 2.0:
            title = f"Inflation surge: Headline CPI {cpi_mom:+.1f}% MoM"
        else:
            title = f"Notable macro point: USD/EGP {usd_end:.2f}"

        highlights.append(
            {
                "date_label": dt.strftime("%b %Y"),
                "title": title,
                "detail": (
                    f"Observed CBE indicators — USD/EGP sell close: {usd_end:.2f} EGP "
                    f"({usd_mom:+.1f}% MoM), Headline CPI MoM: {cpi_mom:+.2f}%, "
                    f"CBE lending rate: {rate:.2f}%."
                ),
            }
        )
    return highlights


# ---------------------------------------------------------------------------
# Page Execution
# ---------------------------------------------------------------------------
raw_df = load_master_data()
if raw_df.empty:
    st.title("Price Explorer")
    st.warning("Dataset not loaded. Please run `python scripts/build_dataset.py` first.")
    st.stop()

df = prepare_explorer_data(raw_df)
all_dates = [pd.Timestamp(d) for d in sorted(df["date"].unique())]
earliest_date = all_dates[0]
latest_date = all_dates[-1]
categories_list = ["All categories"] + sorted(df["category_en"].unique().tolist())
all_commodities = sorted(df["filter_en"].unique().tolist())
default_commodity = (
    "Unpacked Wheat Flour 72%"
    if "Unpacked Wheat Flour 72%" in all_commodities
    else all_commodities[0]
)

# Initialize session state
if "pe_category" not in st.session_state or st.session_state["pe_category"] not in categories_list:
    st.session_state["pe_category"] = "All categories"
if "pe_commodity" not in st.session_state or st.session_state["pe_commodity"] not in all_commodities:
    st.session_state["pe_commodity"] = default_commodity
if "pe_start" not in st.session_state or st.session_state["pe_start"] not in all_dates:
    st.session_state["pe_start"] = earliest_date
if "pe_end" not in st.session_state or st.session_state["pe_end"] not in all_dates:
    st.session_state["pe_end"] = latest_date
if "pe_horizon" not in st.session_state:
    st.session_state["pe_horizon"] = "Full history"
if "pe_compare" not in st.session_state:
    st.session_state["pe_compare"] = False
if "pe_compare_items" not in st.session_state:
    st.session_state["pe_compare_items"] = []


def _reset_filters() -> None:
    st.session_state["pe_category"] = "All categories"
    st.session_state["pe_commodity"] = default_commodity
    st.session_state["pe_start"] = earliest_date
    st.session_state["pe_end"] = latest_date
    st.session_state["pe_horizon"] = "Full history"
    st.session_state["pe_compare"] = False
    st.session_state["pe_compare_items"] = []


def _apply_horizon() -> None:
    horizon = st.session_state.get("pe_horizon")
    end_dt = st.session_state.get("pe_end", latest_date)
    if horizon == "Full history":
        st.session_state["pe_start"] = earliest_date
        st.session_state["pe_end"] = latest_date
    elif horizon == "5 years":
        target_start = end_dt - pd.DateOffset(months=59)
        snapped = next((d for d in all_dates if d >= target_start), earliest_date)
        st.session_state["pe_start"] = snapped
    elif horizon == "1 year":
        target_start = end_dt - pd.DateOffset(months=11)
        snapped = next((d for d in all_dates if d >= target_start), earliest_date)
        st.session_state["pe_start"] = snapped


def _sync_dates_on_manual_change() -> None:
    if st.session_state["pe_start"] > st.session_state["pe_end"]:
        st.session_state["pe_end"] = st.session_state["pe_start"]
    st.session_state["pe_horizon"] = None


# Determine active commodity options for export preview and filter bar
active_cat = st.session_state["pe_category"]
if active_cat == "All categories":
    commodity_options = all_commodities
else:
    commodity_options = sorted(
        df.loc[df["category_en"] == active_cat, "filter_en"].unique().tolist()
    )
if st.session_state["pe_commodity"] not in commodity_options:
    st.session_state["pe_commodity"] = commodity_options[0]

# Precompute export dataframe matching current selection
export_commodities = [st.session_state["pe_commodity"]]
if st.session_state["pe_compare"]:
    extra = [c for c in st.session_state["pe_compare_items"] if c != st.session_state["pe_commodity"]]
    export_commodities.extend(extra[:4])

export_df = df[
    (df["filter_en"].isin(export_commodities))
    & (df["date"] >= st.session_state["pe_start"])
    & (df["date"] <= st.session_state["pe_end"])
][
    [
        "filter_id",
        "filter_en",
        "category_en",
        "date",
        "price",
        "prev_price",
        "mom_pct",
        "usd_sell_end",
        "headline_inflation_mom",
        "cbe_lending_rate",
    ]
].sort_values(["filter_en", "date"])

# 1. Header
st.caption("FOOD PRICE INTELLIGENCE & EARLY WARNING SYSTEM")
hdr_left, hdr_right = st.columns([5, 1.2], vertical_alignment="center")
with hdr_left:
    st.title("Price Explorer")
    st.caption("Explore historical prices across food commodities.")
with hdr_right:
    st.download_button(
        label="Export data",
        icon=":material/download:",
        data=export_df.to_csv(index=False).encode("utf-8"),
        file_name=f"foodwise_price_explorer_{st.session_state['pe_start'].strftime('%Y_%m')}_to_{st.session_state['pe_end'].strftime('%Y_%m')}.csv",
        mime="text/csv",
        width="stretch",
    )

# 2. Data Coverage Notice
st.info(
    f"Connected to the CAPMAS & CBE historical dataset ({df['filter_id'].nunique()} commodities, "
    f"{earliest_date.strftime('%B %Y')} – {latest_date.strftime('%B %Y')}). "
    "Prices are in EGP (the dataset does not include a separate unit column such as kg/liter, "
    "and Economic context highlights are derived from CBE macro columns rather than an external news/event log).",
    icon=":material/info:",
)

# 3. Filter Bar
with st.container(border=True):
    f1, f2, f3, f4, f5 = st.columns([1.5, 1.3, 1.15, 1.15, 0.85], vertical_alignment="bottom")

    with f1:
        selected_commodity: str = st.selectbox(
            "Commodity",
            options=commodity_options,
            key="pe_commodity",
        )

    with f2:
        selected_category: str = st.selectbox(
            "Food category",
            options=categories_list,
            key="pe_category",
        )

    with f3:
        start_date: pd.Timestamp = st.selectbox(
            "Date range (from)",
            options=all_dates,
            format_func=lambda d: d.strftime("%B %Y"),
            key="pe_start",
            on_change=_sync_dates_on_manual_change,
        )

    with f4:
        end_date: pd.Timestamp = st.selectbox(
            "Date range (to)",
            options=all_dates,
            format_func=lambda d: d.strftime("%B %Y"),
            key="pe_end",
            on_change=_sync_dates_on_manual_change,
        )

    with f5:
        st.button("Reset filters", on_click=_reset_filters, width="stretch")

    c_left, c_right = st.columns([3.5, 1.2], vertical_alignment="center")
    with c_left:
        compare_enabled: bool = st.checkbox(
            "Compare commodities (up to 5 series, including your selected commodity)",
            key="pe_compare",
        )
    with c_right:
        st.caption(f"📅 {start_date.strftime('%b %Y')} – {end_date.strftime('%b %Y')}")

    selected_series = [selected_commodity]
    if compare_enabled:
        compare_pool = [c for c in all_commodities if c != selected_commodity]
        # Remove primary commodity if it was previously in compare_items
        st.session_state["pe_compare_items"] = [
            c for c in st.session_state["pe_compare_items"] if c in compare_pool
        ][:4]
        extra_commodities = st.multiselect(
            "Additional commodities to compare (max 4)",
            options=compare_pool,
            max_selections=4,
            key="pe_compare_items",
        )
        selected_series.extend(extra_commodities)

# Slice data for chart and KPIs
chart_df = df[
    (df["filter_en"].isin(selected_series))
    & (df["date"] >= start_date)
    & (df["date"] <= end_date)
].sort_values(["filter_en", "date"])

primary_df = chart_df[chart_df["filter_en"] == selected_commodity].sort_values("date")

# 4. Main Chart Section: Historical monthly prices
with st.container(border=True):
    ch_l, ch_r = st.columns([3.2, 1.6], vertical_alignment="center")
    with ch_l:
        st.subheader("Historical monthly prices")
        series_label = (
            selected_commodity
            if len(selected_series) == 1
            else f"{selected_commodity} + {len(selected_series) - 1} comparison(s)"
        )
        st.caption(f"{series_label} · {start_date.strftime('%b %Y')} – {end_date.strftime('%b %Y')}")
    with ch_r:
        st.segmented_control(
            "Time window",
            options=["Full history", "5 years", "1 year"],
            selection_mode="single",
            key="pe_horizon",
            on_change=_apply_horizon,
            label_visibility="collapsed",
            width="stretch",
        )

    st.caption("Hover for monthly details · Drag to zoom")
    if chart_df.empty:
        st.info("No price observations found in the selected date range.")
    else:
        st.plotly_chart(
            render_price_chart(chart_df, selected_series),
            width="stretch",
            config={"displayModeBar": False},
        )

    with st.expander("View chart data as a table"):
        table_df = chart_df[
            ["date", "filter_en", "category_en", "price", "mom_pct", "usd_sell_end"]
        ].copy()
        table_df["date"] = table_df["date"].dt.strftime("%Y-%m")
        table_df = table_df.rename(
            columns={
                "date": "Month",
                "filter_en": "Commodity",
                "category_en": "Category",
                "price": "Price (EGP)",
                "mom_pct": "MoM Change (%)",
                "usd_sell_end": "USD/EGP Sell End",
            }
        )
        st.dataframe(
            table_df,
            hide_index=True,
            width="stretch",
            column_config={
                "Price (EGP)": st.column_config.NumberColumn("Price (EGP)", format="%.2f"),
                "MoM Change (%)": st.column_config.NumberColumn("MoM Change (%)", format="%+.1f%%"),
                "USD/EGP Sell End": st.column_config.NumberColumn("USD/EGP Sell End", format="%.2f"),
            },
        )

    n_months_in_range = int(primary_df["date"].nunique())
    ft_l, ft_r = st.columns([4, 1])
    with ft_l:
        st.caption("Monthly observations in EGP from CAPMAS master dataset.")
    with ft_r:
        st.caption(f"{n_months_in_range} months")

# 5. KPI Section: <Selected Commodity> at a glance
gl_l, gl_r = st.columns([3.5, 1.5], vertical_alignment="bottom")
with gl_l:
    st.subheader(f"{selected_commodity} at a glance")
with gl_r:
    st.caption("Selected period · Real CAPMAS observations")

if primary_df.empty:
    st.info("Select a valid date range to view commodity KPIs.")
else:
    latest_row = primary_df.iloc[-1]
    latest_obs_date = pd.Timestamp(latest_row["date"])
    latest_price = float(latest_row["price"])

    prev_price = latest_row["prev_price"]
    mom_pct = latest_row["mom_pct"]

    # Determine calendar month label prior to latest observation in dataset
    commodity_full = df[df["filter_en"] == selected_commodity].sort_values("date")
    prior_rows = commodity_full[commodity_full["date"] < latest_obs_date]
    prev_obs_label = (
        pd.Timestamp(prior_rows.iloc[-1]["date"]).strftime("%b %Y")
        if not prior_rows.empty
        else "No prior month"
    )

    peak_idx = primary_df["price"].idxmax()
    peak_row = primary_df.loc[peak_idx]
    peak_price = float(peak_row["price"])
    peak_date_label = pd.Timestamp(peak_row["date"]).strftime("%b %Y")

    k1, k2, k3, k4 = st.columns(4)
    with k1:
        st.metric(
            label="Latest Price",
            value=f"{latest_price:.2f} EGP",
            delta=f"{latest_obs_date.strftime('%b %Y')} · Last observation in period",
            delta_color="off",
            delta_arrow="off",
            border=True,
            height="stretch",
        )
    with k2:
        st.metric(
            label="Previous Month Price",
            value=f"{float(prev_price):.2f} EGP" if pd.notna(prev_price) else "N/A",
            delta=f"{prev_obs_label} · Month before latest",
            delta_color="off",
            delta_arrow="off",
            border=True,
            height="stretch",
        )
    with k3:
        if pd.notna(mom_pct):
            mom_val = float(mom_pct)
            if mom_val > 0.05:
                mom_lbl = "Price Increase"
                mom_color = "inverse"
            elif mom_val < -0.05:
                mom_lbl = "Price Decrease"
                mom_color = "inverse"
            else:
                mom_lbl = "Unchanged"
                mom_color = "off"
            st.metric(
                label="Monthly Price Change (%)",
                value=f"{mom_val:+.1f}%",
                delta=mom_lbl,
                delta_color=mom_color,
                delta_arrow="auto" if mom_color != "off" else "off",
                delta_description="vs. prior month",
                border=True,
                height="stretch",
            )
        else:
            st.metric(
                label="Monthly Price Change (%)",
                value="N/A",
                delta="No prior month",
                delta_color="off",
                delta_arrow="off",
                border=True,
                height="stretch",
            )
    with k4:
        st.metric(
            label="Highest Price in Selected Period",
            value=f"{peak_price:.2f} EGP",
            delta=f"{peak_date_label} · Maximum monthly observation",
            delta_color="off",
            delta_arrow="off",
            border=True,
            height="stretch",
        )

# 6. Section: Economic context & event notes
with st.container(border=True):
    ec_l, ec_r = st.columns([4, 1.3], vertical_alignment="center")
    with ec_l:
        st.subheader("Economic context & event notes")
        st.caption(
            "Observed CBE macroeconomic highlights within the selected date range "
            "(derived from dataset indicators; no external event-log column is present)."
        )
    with ec_r:
        st.caption("CBE MACRO INDICATORS")

    highlights = get_macro_highlights(df, start_date, end_date)
    if not highlights:
        st.info("No macroeconomic observations found for the selected date range.")
    else:
        ec_cols = st.columns(len(highlights))
        for idx, (col, note) in enumerate(zip(ec_cols, highlights), start=1):
            with col:
                with st.container(border=True):
                    st.caption(f"0{idx} · {note['date_label']} · Observed CBE data")
                    st.markdown(f"**{note['title']}**")
                    st.caption(note["detail"])

    st.caption(
        "Highlights above summarize observed CBE exchange rate, inflation, and policy-rate movements "
        "in the selected period rather than curated news events."
    )

# 7. Methodology & dataset fields expander
with st.expander("About the data & calculations — Dataset fields & methodology"):
    st.markdown(
        """
        - **Data Source (`data/processed/master_food_dataset.csv`)**: Monthly CAPMAS retail food prices (`filter_id`, `filter_en`, `category_en`, `date`, `price`) merged with CBE indicators (`usd_sell_end`, `usd_mom_pct`, `headline_inflation_mom`, `cbe_lending_rate`) across **57 commodities** and **138 months** (Jan 2015 – Jun 2026).
        - **Monthly Price Change (%)**: Computed per commodity as `(price_t - price_{t-1}) / price_{t-1} * 100` for the latest observation in the selected range.
        - **Compare Commodities**: Allows overlaying up to 5 total series (`selected commodity` + up to 4 additional commodities) in EGP over the active date range.
        - **Missing Dataset Fields**: `master_food_dataset.csv` does not contain a unit column (e.g., `kg` or `liter`) or a historical event-annotation column; therefore prices are labeled in `EGP` and economic context cards report observed CBE macro shifts directly from the CSV.
        """
    )

st.caption("FoodWise Egypt · Turning market data into food security insights.")
