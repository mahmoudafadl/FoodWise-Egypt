"""Economic Impact page for the FoodWise Egypt Streamlit dashboard."""
from pathlib import Path
import sys

import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import streamlit as st

APP_DIR = Path(__file__).resolve().parents[1]
if str(APP_DIR) not in sys.path:
    sys.path.insert(0, str(APP_DIR))

from utils import load_master_data

st.set_page_config(
    page_title="Economic Impact | FoodWise Egypt",
    page_icon="💱",
    layout="wide",
)

ECONOMIC_INDICATORS = {
    "USD exchange rate (sell end)": ("usd_sell_end", "EGP/USD"),
    "USD exchange rate (sell mean)": ("usd_sell_mean", "EGP/USD"),
    "USD exchange rate (MoM %)": ("usd_mom_pct", "% MoM"),
    "Headline inflation (% MoM)": ("headline_inflation_mom", "% MoM"),
    "Core inflation (% MoM)": ("core_inflation_mom", "% MoM"),
    "Regulated inflation (% MoM)": ("regulated_inflation_mom", "% MoM"),
    "Veg & fruit inflation (% MoM)": ("veg_fruit_inflation_mom", "% MoM"),
    "CBE deposit rate (%)": ("cbe_deposit_rate", "%"),
    "CBE lending rate (%)": ("cbe_lending_rate", "%"),
}

HEATMAP_FACTORS = [
    ("price", "Food price"),
    ("usd_sell_end", "USD exchange rate"),
    ("headline_inflation_mom", "Headline infl."),
    ("core_inflation_mom", "Core infl."),
    ("cbe_deposit_rate", "CBE deposit rate"),
    ("cbe_lending_rate", "CBE lending rate"),
]


@st.cache_data(show_spinner=False)
def prepare_economic_data(raw_df: pd.DataFrame) -> pd.DataFrame:
    """Sort observations and compute per-commodity monthly percentage price changes."""
    df = raw_df.sort_values(["filter_id", "date"]).copy()
    df["price_mom_pct"] = df.groupby("filter_id")["price"].pct_change() * 100.0
    return df


def describe_correlation(r_val: float | None) -> str:
    """Return a concise plain-text strength and direction label for Pearson r."""
    if r_val is None or pd.isna(r_val):
        return "Insufficient variation"
    abs_r = abs(r_val)
    direction = "positive" if r_val >= 0 else "negative"
    if abs_r >= 0.7:
        return f"Strong {direction} relationship"
    if abs_r >= 0.4:
        return f"Moderate {direction} relationship"
    if abs_r >= 0.2:
        return f"Weak {direction} relationship"
    return "Very weak / no linear relationship"


def safe_corr(s1: pd.Series, s2: pd.Series) -> tuple[float | None, int]:
    """Compute Pearson correlation safely, returning (r, n_matched_pairs)."""
    valid = pd.concat([s1, s2], axis=1).dropna()
    n = len(valid)
    if n < 3 or valid.iloc[:, 0].nunique() < 2 or valid.iloc[:, 1].nunique() < 2:
        return None, n
    r = float(valid.iloc[:, 0].corr(valid.iloc[:, 1]))
    return (None if pd.isna(r) else r), n


def render_dual_axis_chart(
    window_df: pd.DataFrame,
    commodity: str,
    indicator_label: str,
    indicator_col: str,
    indicator_unit: str,
) -> go.Figure:
    """Render a dual-axis Plotly line chart of food price vs. economic indicator levels."""
    fig = make_subplots(specs=[[{"secondary_y": True}]])

    fig.add_trace(
        go.Scatter(
            x=window_df["date"],
            y=window_df["price"],
            name=f"{commodity} (EGP · Left axis)",
            mode="lines",
            line=dict(color="#1E4D3B", width=2.4),
            hovertemplate=f"<b>{commodity}</b>: %{{y:.2f}} EGP<extra></extra>",
        ),
        secondary_y=False,
    )

    fig.add_trace(
        go.Scatter(
            x=window_df["date"],
            y=window_df[indicator_col],
            name=f"{indicator_label} ({indicator_unit} · Right axis)",
            mode="lines",
            line=dict(color="#9C7A3C", width=2.2),
            hovertemplate=f"<b>{indicator_label}</b>: %{{y:.2f}} {indicator_unit}<extra></extra>",
        ),
        secondary_y=True,
    )

    fig.update_layout(
        height=340,
        margin=dict(l=12, r=12, t=28, b=12),
        hovermode="x unified",
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="left", x=0),
        xaxis=dict(title=None, showgrid=False, tickformat="%b %Y"),
    )
    fig.update_yaxes(title_text="Food price (EGP)", secondary_y=False)
    fig.update_yaxes(
        title_text=f"{indicator_label} ({indicator_unit})",
        secondary_y=True,
        showgrid=False,
    )
    return fig


def render_correlation_heatmap(window_df: pd.DataFrame) -> tuple[go.Figure, pd.DataFrame]:
    """Compute and plot a 6x6 Pearson correlation matrix for food price and 5 macro series."""
    cols = [c for c, _ in HEATMAP_FACTORS]
    labels = [lbl for _, lbl in HEATMAP_FACTORS]

    corr_mat = window_df[cols].corr(method="pearson")
    z_vals: list[list[float | None]] = []
    text_vals: list[list[str]] = []

    for c_row in cols:
        z_row: list[float | None] = []
        t_row: list[str] = []
        for c_col in cols:
            val = corr_mat.loc[c_row, c_col]
            if pd.isna(val):
                z_row.append(None)
                t_row.append("N/A")
            else:
                z_row.append(float(val))
                t_row.append(f"{val:+.2f}" if c_row != c_col else "1.00")
        z_vals.append(z_row)
        text_vals.append(t_row)

    # Diverging colorscale: negative (muted rose) -> neutral (light grey) -> positive (sage green)
    colorscale = [
        [0.0, "#D99B95"],
        [0.5, "#F2F1EC"],
        [1.0, "#6E9A7A"],
    ]

    fig = go.Figure(
        data=go.Heatmap(
            z=z_vals,
            x=labels,
            y=labels,
            zmin=-1.0,
            zmax=1.0,
            colorscale=colorscale,
            colorbar=dict(title="r", thickness=12, len=0.8),
            text=text_vals,
            texttemplate="%{text}",
            textfont=dict(size=11, color="#1F2D26"),
            hovertemplate="<b>%{y}</b> vs. <b>%{x}</b><br>Pearson r: <b>%{text}</b><extra></extra>",
            xgap=3,
            ygap=3,
        )
    )
    fig.update_layout(
        height=350,
        margin=dict(l=8, r=8, t=12, b=12),
        yaxis=dict(autorange="reversed", showgrid=False, tickfont=dict(size=11)),
        xaxis=dict(showgrid=False, tickfont=dict(size=11)),
    )

    table_df = corr_mat.copy()
    table_df.index = labels
    table_df.columns = labels
    return fig, table_df


def compute_lag_correlations(
    full_comm_df: pd.DataFrame,
    start_date: pd.Timestamp,
    end_date: pd.Timestamp,
    indicator_col: str,
    max_lag: int = 6,
) -> pd.DataFrame:
    """
    Compute Pearson r between indicator_{t - lag} and food price MoM % change at t.

    Uses full commodity history to shift before filtering to [start_date, end_date],
    matching indicator observations cleanly within the selected period.
    """
    work = full_comm_df.sort_values("date").copy()
    rows = []
    for lag in range(max_lag + 1):
        work["shifted_ind"] = work[indicator_col].shift(lag)
        sliced = work[(work["date"] >= start_date) & (work["date"] <= end_date)]
        r_val, n_matched = safe_corr(sliced["shifted_ind"], sliced["price_mom_pct"])
        rows.append(
            {
                "Lag (months)": lag,
                "Pearson r": r_val,
                "Matched months (n)": n_matched,
            }
        )
    return pd.DataFrame(rows)


def render_lag_chart(lag_df: pd.DataFrame, selected_lag: int) -> go.Figure:
    """Render a Plotly bar chart of lag correlations (0–6 months), highlighting selected_lag."""
    y_vals = [0.0 if pd.isna(v) else float(v) for v in lag_df["Pearson r"]]
    colors = [
        "#B4822C" if int(lag) == selected_lag else "#8FA89B"
        for lag in lag_df["Lag (months)"]
    ]
    labels = [
        f"{v:+.2f}" if pd.notna(v) else "N/A"
        for v in lag_df["Pearson r"]
    ]

    fig = go.Figure(
        go.Bar(
            x=lag_df["Lag (months)"],
            y=y_vals,
            marker_color=colors,
            text=labels,
            textposition="outside",
            customdata=lag_df["Matched months (n)"],
            hovertemplate=(
                "<b>Lag %{x} month(s)</b><br>"
                "Pearson r: <b>%{text}</b><br>"
                "Matched months: %{customdata}<extra></extra>"
            ),
        )
    )
    fig.add_hline(y=0, line_width=1, line_color="#9CA3AF")
    fig.update_layout(
        height=280,
        margin=dict(l=12, r=12, t=20, b=12),
        showlegend=False,
        xaxis=dict(
            title="Indicator lead (months)",
            tickmode="array",
            tickvals=list(range(7)),
            showgrid=False,
        ),
        yaxis=dict(title="Pearson r", range=[-1.05, 1.05], zeroline=False),
    )
    return fig


# ---------------------------------------------------------------------------
# Page Execution
# ---------------------------------------------------------------------------
raw_df = load_master_data()
if raw_df.empty:
    st.title("Economic Impact")
    st.warning("Dataset not loaded. Please run `python scripts/build_dataset.py` first.")
    st.stop()

df = prepare_economic_data(raw_df)
all_dates = [pd.Timestamp(d) for d in sorted(df["date"].unique())]
earliest_date = all_dates[0]
latest_date = all_dates[-1]
default_5y_start = next(
    (d for d in all_dates if d >= latest_date - pd.DateOffset(months=59)),
    earliest_date,
)

categories_list = ["All categories"] + sorted(df["category_en"].unique().tolist())
all_commodities = sorted(df["filter_en"].unique().tolist())
default_commodity = (
    "Unpacked Wheat Flour 72%"
    if "Unpacked Wheat Flour 72%" in all_commodities
    else all_commodities[0]
)
indicator_options = list(ECONOMIC_INDICATORS.keys())

if "ei_category" not in st.session_state or st.session_state["ei_category"] not in categories_list:
    st.session_state["ei_category"] = "All categories"
if "ei_commodity" not in st.session_state or st.session_state["ei_commodity"] not in all_commodities:
    st.session_state["ei_commodity"] = default_commodity
if "ei_indicator" not in st.session_state or st.session_state["ei_indicator"] not in indicator_options:
    st.session_state["ei_indicator"] = indicator_options[0]
if "ei_start" not in st.session_state or st.session_state["ei_start"] not in all_dates:
    st.session_state["ei_start"] = default_5y_start
if "ei_end" not in st.session_state or st.session_state["ei_end"] not in all_dates:
    st.session_state["ei_end"] = latest_date
if "ei_horizon" not in st.session_state:
    st.session_state["ei_horizon"] = "5 years"
if "ei_lag" not in st.session_state:
    st.session_state["ei_lag"] = 2


def _reset_filters() -> None:
    st.session_state["ei_category"] = "All categories"
    st.session_state["ei_commodity"] = default_commodity
    st.session_state["ei_indicator"] = indicator_options[0]
    st.session_state["ei_start"] = default_5y_start
    st.session_state["ei_end"] = latest_date
    st.session_state["ei_horizon"] = "5 years"
    st.session_state["ei_lag"] = 2


def _apply_horizon() -> None:
    horizon = st.session_state.get("ei_horizon")
    end_dt = st.session_state.get("ei_end", latest_date)
    if horizon == "Full history":
        st.session_state["ei_start"] = earliest_date
        st.session_state["ei_end"] = latest_date
    elif horizon == "5 years":
        target_start = end_dt - pd.DateOffset(months=59)
        st.session_state["ei_start"] = next(
            (d for d in all_dates if d >= target_start), earliest_date
        )
    elif horizon == "1 year":
        target_start = end_dt - pd.DateOffset(months=11)
        st.session_state["ei_start"] = next(
            (d for d in all_dates if d >= target_start), earliest_date
        )


def _sync_manual_dates() -> None:
    if st.session_state["ei_start"] > st.session_state["ei_end"]:
        st.session_state["ei_end"] = st.session_state["ei_start"]
    st.session_state["ei_horizon"] = None


active_cat = st.session_state["ei_category"]
if active_cat == "All categories":
    commodity_options = all_commodities
else:
    commodity_options = sorted(
        df.loc[df["category_en"] == active_cat, "filter_en"].unique().tolist()
    )
if st.session_state["ei_commodity"] not in commodity_options:
    st.session_state["ei_commodity"] = commodity_options[0]

# Precompute filtered export dataframe
export_df = df[
    (df["filter_en"] == st.session_state["ei_commodity"])
    & (df["date"] >= st.session_state["ei_start"])
    & (df["date"] <= st.session_state["ei_end"])
].sort_values("date")

# 1. Header
st.caption("FOOD PRICE INTELLIGENCE & EARLY WARNING SYSTEM")
hdr_l, hdr_r = st.columns([5, 1.2], vertical_alignment="center")
with hdr_l:
    st.title("Economic Impact")
    st.caption("Understand how macroeconomic conditions relate to food prices.")
with hdr_r:
    st.download_button(
        label="Export data",
        icon=":material/download:",
        data=export_df.to_csv(index=False).encode("utf-8"),
        file_name=(
            f"foodwise_economic_impact_{st.session_state['ei_commodity'].lower().replace(' ', '_')}_"
            f"{st.session_state['ei_start'].strftime('%Y_%m')}_to_{st.session_state['ei_end'].strftime('%Y_%m')}.csv"
        ),
        mime="text/csv",
        width="stretch",
    )

# 2. Data & Relationship Notice
st.info(
    f"Connected to real CAPMAS & CBE historical data ({df['filter_id'].nunique()} commodities, "
    f"{earliest_date.strftime('%b %Y')} – {latest_date.strftime('%b %Y')}). "
    "All correlations and lag statistics are computed directly from the observed monthly series in "
    "`master_food_dataset.csv`; exploratory correlations are descriptive and do not imply causation.",
    icon=":material/info:",
)

# 3. Filter Bar
with st.container(border=True):
    f1, f2, f3, f4, f5 = st.columns([1.4, 1.2, 1.5, 1.05, 1.05], vertical_alignment="bottom")
    with f1:
        selected_commodity: str = st.selectbox(
            "Commodity",
            options=commodity_options,
            key="ei_commodity",
        )
    with f2:
        selected_category: str = st.selectbox(
            "Category (optional)",
            options=categories_list,
            key="ei_category",
        )
    with f3:
        selected_ind_label: str = st.selectbox(
            "Economic indicator",
            options=indicator_options,
            key="ei_indicator",
        )
    with f4:
        start_date: pd.Timestamp = st.selectbox(
            "Date range (from)",
            options=all_dates,
            format_func=lambda d: d.strftime("%B %Y"),
            key="ei_start",
            on_change=_sync_manual_dates,
        )
    with f5:
        end_date: pd.Timestamp = st.selectbox(
            "Date range (to)",
            options=all_dates,
            format_func=lambda d: d.strftime("%B %Y"),
            key="ei_end",
            on_change=_sync_manual_dates,
        )

    sub_l, sub_r = st.columns([4.5, 1.0], vertical_alignment="center")
    with sub_l:
        st.caption("Monthly observations · Category narrows the commodity list; analysis uses one commodity.")
    with sub_r:
        st.button("Reset filters", on_click=_reset_filters, width="stretch")

ind_col, ind_unit = ECONOMIC_INDICATORS[selected_ind_label]
full_comm_df = df[df["filter_en"] == selected_commodity].sort_values("date").copy()
window_df = full_comm_df[
    (full_comm_df["date"] >= start_date) & (full_comm_df["date"] <= end_date)
].copy()
n_obs = len(window_df)

# Compute lag table up front so KPI 4 stays synced with Selected lag
lag_df = compute_lag_correlations(
    full_comm_df=full_comm_df,
    start_date=start_date,
    end_date=end_date,
    indicator_col=ind_col,
    max_lag=6,
)

# 4. Section: Relationships at a glance
gl_l, gl_r = st.columns([3.5, 1.5], vertical_alignment="bottom")
with gl_l:
    st.subheader("Relationships at a glance")
with gl_r:
    st.caption(f"{n_obs} monthly observations · Real CAPMAS & CBE data")

if window_df.empty:
    st.warning("No observations found in the selected date range.")
    st.stop()

latest_ind_val = float(window_df[ind_col].iloc[-1])
latest_obs_label = pd.Timestamp(window_df["date"].iloc[-1]).strftime("%b %Y")
sel_r, sel_n = safe_corr(window_df["price"], window_df[ind_col])

# Strongest relationship among the 5 core macro indicators (matching heatmap factors)
strongest_label = "N/A"
strongest_r: float | None = None
for col, lbl in HEATMAP_FACTORS[1:]:
    r_cand, _ = safe_corr(window_df["price"], window_df[col])
    if r_cand is not None and (strongest_r is None or abs(r_cand) > abs(strongest_r)):
        strongest_r = r_cand
        strongest_label = lbl

current_lag = int(st.session_state["ei_lag"])
lag_row = lag_df[lag_df["Lag (months)"] == current_lag].iloc[0]
lag_r = lag_row["Pearson r"]
lag_n = int(lag_row["Matched months (n)"])

k1, k2, k3, k4 = st.columns(4)
with k1:
    st.metric(
        label="Selected Indicator",
        value=f"{latest_ind_val:.2f} {ind_unit}",
        delta=f"{selected_ind_label} · Latest: {latest_obs_label}",
        delta_color="off",
        delta_arrow="off",
        border=True,
        height="stretch",
    )
with k2:
    st.metric(
        label="Correlation with Food Price",
        value=f"{sel_r:+.2f}" if sel_r is not None else "N/A",
        delta=describe_correlation(sel_r),
        delta_color="off",
        delta_arrow="off",
        delta_description=f"Pearson r (levels) · n = {sel_n}",
        border=True,
        height="stretch",
    )
with k3:
    st.metric(
        label="Strongest Relationship",
        value=f"{strongest_r:+.2f}" if strongest_r is not None else "N/A",
        delta=strongest_label,
        delta_color="off",
        delta_arrow="off",
        delta_description="Largest |r| vs. food price (5 macro indicators)",
        border=True,
        height="stretch",
    )
with k4:
    st.metric(
        label="Selected Time Lag",
        value=f"{current_lag} months",
        delta=(
            f"Lag r = {lag_r:+.2f} · n = {lag_n}"
            if pd.notna(lag_r)
            else f"Insufficient pairs · n = {lag_n}"
        ),
        delta_color="off",
        delta_arrow="off",
        delta_description="Indicator leads food price change",
        border=True,
        height="stretch",
    )

# 5. Section: Food prices & economic conditions
with st.container(border=True):
    ch_l, ch_r = st.columns([3.2, 1.6], vertical_alignment="center")
    with ch_l:
        st.subheader("Food prices & economic conditions")
        st.caption(
            f"{selected_commodity} vs. {selected_ind_label} · Monthly levels, same-month alignment"
        )
    with ch_r:
        st.segmented_control(
            "Time window",
            options=["Full history", "5 years", "1 year"],
            selection_mode="single",
            key="ei_horizon",
            on_change=_apply_horizon,
            label_visibility="collapsed",
            width="stretch",
        )

    st.plotly_chart(
        render_dual_axis_chart(
            window_df=window_df,
            commodity=selected_commodity,
            indicator_label=selected_ind_label,
            indicator_col=ind_col,
            indicator_unit=ind_unit,
        ),
        width="stretch",
        config={"displayModeBar": False},
    )
    st.caption(
        "Separate axis scales show co-movement, not equal magnitudes. "
        "Shared trends in levels can produce misleading correlations."
    )

# 6. Bottom Two-Column Row: Correlation heatmap + Lag analysis
left_col, right_col = st.columns([1.1, 1.0], gap="medium")

with left_col:
    with st.container(border=True):
        hl, hr = st.columns([3.5, 1.1], vertical_alignment="center")
        with hl:
            st.subheader("Correlation heatmap")
        with hr:
            st.caption("PEARSON r")
        st.caption(f"{selected_commodity} & five key macro indicators · Same-month levels")

        hm_fig, corr_table = render_correlation_heatmap(window_df)
        st.plotly_chart(hm_fig, width="stretch", config={"displayModeBar": False})

        with st.expander("View correlation values as a table"):
            st.dataframe(
                corr_table.round(3),
                width="stretch",
            )

        st.caption(
            "Read as a pairwise matrix. Colors show direction, not benefit or harm. "
            "N/A means fewer than 3 valid pairs or zero variation."
        )

with right_col:
    with st.container(border=True):
        ll, lr = st.columns([3.5, 1.2], vertical_alignment="center")
        with ll:
            st.subheader("Lag analysis")
        with lr:
            st.caption("EXPLORATORY")
        st.caption("Indicator at t − lag vs. food price monthly change at t")

        lag_c1, lag_c2 = st.columns([1.2, 1.8], vertical_alignment="center")
        with lag_c1:
            selected_lag = st.selectbox(
                "Selected lag",
                options=list(range(7)),
                format_func=lambda m: f"{m} month" if m == 1 else f"{m} months",
                key="ei_lag",
            )
        with lag_c2:
            st.caption(f"**{selected_ind_label}**")
            st.caption("Positive lag places the indicator earlier than the food price change.")

        st.plotly_chart(
            render_lag_chart(lag_df, selected_lag=selected_lag),
            width="stretch",
            config={"displayModeBar": False},
        )

        sel_lag_row = lag_df[lag_df["Lag (months)"] == selected_lag].iloc[0]
        sel_lag_r = sel_lag_row["Pearson r"]
        sel_lag_n = int(sel_lag_row["Matched months (n)"])
        if pd.notna(sel_lag_r):
            st.info(
                f"At a {selected_lag}-month lag: **r = {sel_lag_r:+.2f}** across {sel_lag_n} matched months.",
                icon=":material/analytics:",
            )
        else:
            st.info(
                f"At a {selected_lag}-month lag: insufficient valid pairs ({sel_lag_n} months).",
                icon=":material/analytics:",
            )

        st.caption(
            "This chart uses food price changes, unlike the level correlations above. "
            "A lagged association is not proof of a delayed cause-effect."
        )

        with st.expander("View lag values & sample counts"):
            lag_disp = lag_df.copy()
            st.dataframe(
                lag_disp,
                hide_index=True,
                width="stretch",
                column_config={
                    "Pearson r": st.column_config.NumberColumn("Pearson r", format="%+.3f"),
                },
            )

# 7. Caveat banner + Observations & Methodology expanders
st.warning(
    "**Relationships are not proof of causation.** These exploratory correlations do not isolate policy, "
    "exchange-rate, or inflation effects. Shared trends, seasonality, omitted variables, and the chosen "
    "period can all change the result. No causal model or significance test is connected.",
    icon=":material/warning:",
)

with st.expander("View filtered monthly observations & all dataset fields"):
    obs_cols = [
        "date",
        "filter_en",
        "category_en",
        "price",
        "price_mom_pct",
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
        "cbe_lending_rate",
    ]
    obs_disp = window_df[obs_cols].copy()
    obs_disp["date"] = obs_disp["date"].dt.strftime("%Y-%m")
    st.dataframe(obs_disp, hide_index=True, width="stretch")

with st.expander("About this analysis — Definitions, lag matching & source fields"):
    st.markdown(
        """
        - **Source Dataset (`data/processed/master_food_dataset.csv`)**: Monthly CAPMAS food prices (`price`) harmonized with Central Bank of Egypt (CBE) exchange rates (`usd_buy_mean`, `usd_sell_mean`, `usd_sell_end`, `usd_volatility`, `usd_mom_pct`), inflation series (`headline_inflation_mom`, `core_inflation_mom`, `regulated_inflation_mom`, `veg_fruit_inflation_mom`), and policy corridor rates (`cbe_deposit_rate`, `cbe_lending_rate`) from **January 2015 – June 2026**.
        - **Same-Month Level Correlations (`Relationships at a glance` & `Correlation heatmap`)**: Computed via Pearson `r` on same-month values within the active `[Date range (from), Date range (to)]` window.
        - **Lag Analysis (`0–6 months`)**: Shifts the selected economic indicator back by `k` months across the commodity's full history before slicing to the active window, then correlates `indicator_{t-k}` with `food_price_mom_pct_t`.
        - **Limitations**: Correlations are exploratory and unadjusted for trend/seasonality; no econometric causality model or statistical p-value filter is applied.
        """
    )

st.caption("FoodWise Egypt · Turning market data into food security insights.")
