"""Alerts & Risk page for the FoodWise Egypt Streamlit dashboard."""
from pathlib import Path
import sys

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

APP_DIR = Path(__file__).resolve().parents[1]
ROOT_DIR = APP_DIR.parent
for p in (str(APP_DIR), str(ROOT_DIR)):
    if p not in sys.path:
        sys.path.insert(0, p)

from src.predictor import load_model_artifacts
from utils import load_master_data

st.set_page_config(
    page_title="Alerts & Risk | FoodWise Egypt",
    page_icon="🚨",
    layout="wide",
)

Z_ANOMALY_THRESHOLD = 2.0
HIGH_MOM_THRESHOLD = 5.0
MEDIUM_MOM_THRESHOLD = 2.0


@st.cache_data(show_spinner=False)
def prepare_alerts_data(raw_df: pd.DataFrame) -> pd.DataFrame:
    """
    Compute MoM % changes, 12-month rolling Z-scores, and prototype rule-based risk flags.

    Note: Isolation Forest is not yet connected in models/ or src/predictor.py;
    classifications below use transparent statistical thresholds on real CAPMAS prices.
    """
    df = raw_df.sort_values(["filter_id", "date"]).copy()
    df["prev_price"] = df.groupby("filter_id")["price"].shift(1)
    df["mom_pct"] = df.groupby("filter_id")["price"].pct_change() * 100.0
    df["abs_mom_pct"] = df["mom_pct"].abs()

    # Prior 12-month rolling mean & std (shifted by 1 month to avoid current-month leakage)
    grp = df.groupby("filter_id")["mom_pct"]
    df["roll_mean_12m"] = grp.transform(lambda s: s.rolling(12, min_periods=6).mean().shift(1))
    df["roll_std_12m"] = grp.transform(lambda s: s.rolling(12, min_periods=6).std().shift(1))
    valid_std = df["roll_std_12m"].where(df["roll_std_12m"] > 1e-6, np.nan)
    df["z_score"] = (df["mom_pct"] - df["roll_mean_12m"]) / valid_std
    df["is_anomaly"] = df["z_score"].abs() >= Z_ANOMALY_THRESHOLD

    def _classify_row(row: pd.Series) -> tuple[str, int, str]:
        mom = row["mom_pct"]
        z_val = row["z_score"]
        is_anom = bool(row["is_anomaly"]) if pd.notna(row["is_anomaly"]) else False

        if pd.isna(mom):
            return "Low", 2, "Insufficient prior month history"

        direction = "increase" if mom >= 0 else "decline"
        if is_anom and abs(mom) > HIGH_MOM_THRESHOLD:
            return (
                "High",
                0,
                f"Unusual monthly {direction} under 12-mo rolling Z-score (|Z| ≥ {Z_ANOMALY_THRESHOLD:.1f}) & > {HIGH_MOM_THRESHOLD:.0f}% threshold",
            )
        if is_anom:
            return (
                "High",
                0,
                f"Unusual monthly {direction} under 12-mo rolling Z-score rule (|Z| ≥ {Z_ANOMALY_THRESHOLD:.1f})",
            )
        if abs(mom) > HIGH_MOM_THRESHOLD:
            return (
                "High",
                0,
                f"Monthly {direction} above {HIGH_MOM_THRESHOLD:.0f}% alert threshold",
            )
        if abs(mom) > MEDIUM_MOM_THRESHOLD:
            return (
                "Medium",
                1,
                f"Monthly {direction} above {MEDIUM_MOM_THRESHOLD:.0f}% watch threshold",
            )
        return "Low", 2, "Within normal monitoring range"

    classified = df.apply(_classify_row, axis=1, result_type="expand")
    df["risk_level"] = classified[0]
    df["risk_rank"] = classified[1]
    df["alert_reason"] = classified[2]
    return df


def render_risk_distribution_bar(low_n: int, med_n: int, high_n: int) -> go.Figure:
    """Render a single horizontal stacked bar showing Low / Medium / High distribution."""
    fig = go.Figure()
    for name, count, color in [
        ("Low Risk", low_n, "#6E9A7A"),
        ("Medium Risk", med_n, "#C9A45C"),
        ("High Risk", high_n, "#B86B62"),
    ]:
        fig.add_trace(
            go.Bar(
                y=["Commodities"],
                x=[count],
                name=name,
                orientation="h",
                marker_color=color,
                text=[str(count) if count > 0 else ""],
                textposition="inside",
                insidetextanchor="middle",
                textfont=dict(color="#FFFFFF", size=12),
                hovertemplate=f"<b>{name}</b>: %{{x}} commodities<extra></extra>",
            )
        )

    total = max(low_n + med_n + high_n, 1)
    fig.update_layout(
        barmode="stack",
        height=72,
        margin=dict(l=0, r=0, t=6, b=6),
        showlegend=False,
        xaxis=dict(visible=False, range=[0, total]),
        yaxis=dict(visible=False),
    )
    return fig


def render_alert_trend_chart(trend_df: pd.DataFrame) -> go.Figure:
    """Render monthly High-risk signal counts as a bar + trend line chart."""
    fig = go.Figure()
    fig.add_trace(
        go.Bar(
            x=trend_df["date"],
            y=trend_df["high_count"],
            name="High-risk signals",
            marker_color="#B86B62",
            hovertemplate="<b>%{x|%b %Y}</b><br>High-risk signals: <b>%{y}</b><extra></extra>",
        )
    )
    fig.add_trace(
        go.Scatter(
            x=trend_df["date"],
            y=trend_df["high_count"],
            name="Trend",
            mode="lines+markers",
            line=dict(color="#9C7A3C", width=1.8),
            marker=dict(size=5, color="#9C7A3C"),
            hoverinfo="skip",
        )
    )
    max_y = max(int(trend_df["high_count"].max()) if not trend_df.empty else 1, 1)
    fig.update_layout(
        height=290,
        margin=dict(l=12, r=12, t=16, b=12),
        showlegend=False,
        xaxis=dict(title=None, showgrid=False, tickformat="%b %Y"),
        yaxis=dict(
            title="Signal count",
            range=[0, max_y + max(1, int(max_y * 0.2))],
            dtick=max(1, max_y // 5),
            zeroline=True,
        ),
    )
    return fig


def render_anomaly_review_chart(comm_window: pd.DataFrame, commodity_name: str) -> go.Figure:
    """Render commodity price trajectory highlighting months flagged by |Z| >= 2.0."""
    fig = go.Figure()
    fig.add_trace(
        go.Scatter(
            x=comm_window["date"],
            y=comm_window["price"],
            name=f"{commodity_name} (EGP)",
            mode="lines",
            line=dict(color="#1E4D3B", width=2.4),
            hovertemplate="<b>%{x|%b %Y}</b><br>Price: <b>%{y:.2f} EGP</b><extra></extra>",
        )
    )

    anoms = comm_window[comm_window["is_anomaly"] == True]  # noqa: E712
    if not anoms.empty:
        fig.add_trace(
            go.Scatter(
                x=anoms["date"],
                y=anoms["price"],
                name="Z-score anomaly (|Z| ≥ 2.0)",
                mode="markers",
                marker=dict(
                    size=10,
                    color="#FFFFFF",
                    line=dict(color="#B42318", width=2.5),
                ),
                customdata=anoms[["mom_pct", "z_score"]],
                hovertemplate=(
                    "<b>%{x|%b %Y} — Anomaly</b><br>"
                    "Price: <b>%{y:.2f} EGP</b><br>"
                    "MoM: <b>%{customdata[0]:+.1f}%</b><br>"
                    "Z-score: <b>%{customdata[1]:+.2f}</b><extra></extra>"
                ),
            )
        )

    fig.update_layout(
        height=290,
        margin=dict(l=12, r=12, t=24, b=12),
        hovermode="x unified",
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="left", x=0),
        xaxis=dict(title=None, showgrid=False, tickformat="%b %Y"),
        yaxis=dict(title="Price (EGP)", zeroline=False),
    )
    return fig


# ---------------------------------------------------------------------------
# Page Execution
# ---------------------------------------------------------------------------
raw_df = load_master_data()
if raw_df.empty:
    st.title("Alerts & Risk")
    st.warning("Dataset not loaded. Please run `python scripts/build_dataset.py` first.")
    st.stop()

df = prepare_alerts_data(raw_df)
model_artifacts = load_model_artifacts()

all_dates = [pd.Timestamp(d) for d in sorted(df["date"].unique())]
earliest_date = all_dates[0]
latest_date = all_dates[-1]
default_1y_start = next(
    (d for d in all_dates if d >= latest_date - pd.DateOffset(months=11)),
    earliest_date,
)

risk_options = ["All risk levels", "High", "Medium", "Low"]
categories_list = ["All categories"] + sorted(df["category_en"].unique().tolist())
all_commodities = ["All commodities"] + sorted(df["filter_en"].unique().tolist())

if "ar_risk" not in st.session_state or st.session_state["ar_risk"] not in risk_options:
    st.session_state["ar_risk"] = "All risk levels"
if "ar_category" not in st.session_state or st.session_state["ar_category"] not in categories_list:
    st.session_state["ar_category"] = "All categories"
if "ar_commodity" not in st.session_state:
    st.session_state["ar_commodity"] = "All commodities"
if "ar_start" not in st.session_state or st.session_state["ar_start"] not in all_dates:
    st.session_state["ar_start"] = default_1y_start
if "ar_end" not in st.session_state or st.session_state["ar_end"] not in all_dates:
    st.session_state["ar_end"] = latest_date


def _reset_filters() -> None:
    st.session_state["ar_risk"] = "All risk levels"
    st.session_state["ar_category"] = "All categories"
    st.session_state["ar_commodity"] = "All commodities"
    st.session_state["ar_start"] = default_1y_start
    st.session_state["ar_end"] = latest_date


def _sync_manual_dates() -> None:
    if st.session_state["ar_start"] > st.session_state["ar_end"]:
        st.session_state["ar_end"] = st.session_state["ar_start"]


active_cat = st.session_state["ar_category"]
if active_cat == "All categories":
    commodity_options = all_commodities
else:
    commodity_options = ["All commodities"] + sorted(
        df.loc[df["category_en"] == active_cat, "filter_en"].unique().tolist()
    )
if st.session_state["ar_commodity"] not in commodity_options:
    st.session_state["ar_commodity"] = "All commodities"

# Precompute filtered latest-month alerts for export button
exp_latest = df[df["date"] == st.session_state["ar_end"]].copy()
if st.session_state["ar_category"] != "All categories":
    exp_latest = exp_latest[exp_latest["category_en"] == st.session_state["ar_category"]]
if st.session_state["ar_commodity"] != "All commodities":
    exp_latest = exp_latest[exp_latest["filter_en"] == st.session_state["ar_commodity"]]
if st.session_state["ar_risk"] != "All risk levels":
    exp_latest = exp_latest[exp_latest["risk_level"] == st.session_state["ar_risk"]]
exp_latest = exp_latest.sort_values(["risk_rank", "abs_mom_pct"], ascending=[True, False])

export_cols = [
    "filter_id",
    "filter_en",
    "category_en",
    "date",
    "price",
    "prev_price",
    "mom_pct",
    "z_score",
    "is_anomaly",
    "risk_level",
    "alert_reason",
]

# 1. Header
st.caption("FOOD PRICE INTELLIGENCE & EARLY WARNING SYSTEM")
hdr_l, hdr_r = st.columns([5, 1.2], vertical_alignment="center")
with hdr_l:
    st.title("Alerts & Risk")
    st.caption("Monitor unusual price movements and food market risk.")
with hdr_r:
    st.download_button(
        label="Export alerts",
        icon=":material/download:",
        data=exp_latest[export_cols].to_csv(index=False).encode("utf-8"),
        file_name=f"foodwise_alerts_{st.session_state['ar_end'].strftime('%Y_%m')}.csv",
        mime="text/csv",
        width="stretch",
    )

# 2. Data & Model Status Banner
st.warning(
    "**Prototype rule-based alerts (Isolation Forest not connected).** "
    "`data/processed/master_food_dataset.csv` has no precomputed risk scores, anomaly labels, or alert reasons, "
    "and `models/` does not yet contain a trained Isolation Forest artifact (`src/predictor.py` returns a stub). "
    "All classifications below are computed transparently from real CAPMAS price changes using "
    f"`|MoM %|` thresholds (> {HIGH_MOM_THRESHOLD:.0f}% High, {MEDIUM_MOM_THRESHOLD:.0f}–{HIGH_MOM_THRESHOLD:.0f}% Medium) "
    f"and a 12-month rolling Z-score rule (`|Z| ≥ {Z_ANOMALY_THRESHOLD:.1f}`).",
    icon=":material/warning:",
)

# 3. Filter Bar
with st.container(border=True):
    f1, f2, f3, f4, f5, f6 = st.columns([1.15, 1.3, 1.35, 1.1, 1.1, 0.85], vertical_alignment="bottom")
    with f1:
        selected_risk: str = st.selectbox(
            "Risk level",
            options=risk_options,
            key="ar_risk",
        )
    with f2:
        selected_category: str = st.selectbox(
            "Category",
            options=categories_list,
            key="ar_category",
        )
    with f3:
        selected_commodity: str = st.selectbox(
            "Commodity (optional)",
            options=commodity_options,
            key="ar_commodity",
        )
    with f4:
        start_date: pd.Timestamp = st.selectbox(
            "Date range (from)",
            options=all_dates,
            format_func=lambda d: d.strftime("%B %Y"),
            key="ar_start",
            on_change=_sync_manual_dates,
        )
    with f5:
        end_date: pd.Timestamp = st.selectbox(
            "Date range (to)",
            options=all_dates,
            format_func=lambda d: d.strftime("%B %Y"),
            key="ar_end",
            on_change=_sync_manual_dates,
        )
    with f6:
        st.button("Reset filters", on_click=_reset_filters, width="stretch")

    c_l, c_r = st.columns([4, 1.3])
    with c_l:
        st.caption(
            "Category, commodity, and date filters update all views; Risk level filters the Alerts table only."
        )
    with c_r:
        st.caption(f"Snapshot uses latest month in range ({end_date.strftime('%b %Y')})")

# Scope dataframe across selected category, commodity, and date window
scoped_window = df[(df["date"] >= start_date) & (df["date"] <= end_date)].copy()
if selected_category != "All categories":
    scoped_window = scoped_window[scoped_window["category_en"] == selected_category]
if selected_commodity != "All commodities":
    scoped_window = scoped_window[scoped_window["filter_en"] == selected_commodity]

snapshot_df = scoped_window[scoped_window["date"] == end_date].copy()
if snapshot_df.empty:
    st.warning("No observations match the selected filters.")
    st.stop()

# 4. Section: Current risk snapshot
sn_l, sn_r = st.columns([3.5, 1.5], vertical_alignment="bottom")
with sn_l:
    st.subheader("Current risk snapshot")
with sn_r:
    st.caption(f"{end_date.strftime('%B %Y')} · Prototype rule classifications")

high_snapshot = snapshot_df[snapshot_df["risk_level"] == "High"]
n_active_alerts = len(high_snapshot)
n_high_commodities = int(high_snapshot["filter_id"].nunique())
n_anomalies = int(snapshot_df["is_anomaly"].sum())
n_attention_cats = int(
    snapshot_df[snapshot_df["risk_level"].isin(["Medium", "High"])]["category_en"].nunique()
)
n_scope_cats = int(snapshot_df["category_en"].nunique())

k1, k2, k3, k4 = st.columns(4)
with k1:
    st.metric(
        label="Active Alerts",
        value=str(n_active_alerts),
        delta="High-risk alert signals",
        delta_color="off",
        delta_arrow="off",
        delta_description=f"Latest selected month ({end_date.strftime('%b %Y')})",
        border=True,
        height="stretch",
    )
with k2:
    st.metric(
        label="High-Risk Commodities",
        value=str(n_high_commodities),
        delta=f"of {snapshot_df['filter_id'].nunique()} monitored in scope",
        delta_color="off",
        delta_arrow="off",
        delta_description="Rule-based classification · Real CSV",
        border=True,
        height="stretch",
    )
with k3:
    st.metric(
        label="Price Anomalies",
        value=str(n_anomalies),
        delta="Rolling 12-mo Z-score flags",
        delta_color="off",
        delta_arrow="off",
        delta_description=f"Threshold |Z| ≥ {Z_ANOMALY_THRESHOLD:.1f}",
        border=True,
        height="stretch",
    )
with k4:
    st.metric(
        label="Categories Requiring Attention",
        value=str(n_attention_cats),
        delta="Medium or High risk",
        delta_color="off",
        delta_arrow="off",
        delta_description=f"Out of {n_scope_cats} selected categories",
        border=True,
        height="stretch",
    )

# 5. Section: Risk summary
with st.container(border=True):
    rs_l, rs_r = st.columns([4, 1.2], vertical_alignment="center")
    with rs_l:
        st.subheader("Risk summary")
        st.caption(f"Commodity distribution in the latest selected month ({end_date.strftime('%B %Y')})")
    with rs_r:
        st.caption("PROTOTYPE RULES")

    low_n = int((snapshot_df["risk_level"] == "Low").sum())
    med_n = int((snapshot_df["risk_level"] == "Medium").sum())
    high_n = int((snapshot_df["risk_level"] == "High").sum())
    total_n = max(len(snapshot_df), 1)

    st.plotly_chart(
        render_risk_distribution_bar(low_n, med_n, high_n),
        width="stretch",
        config={"displayModeBar": False},
    )

    r1, r2, r3 = st.columns(3)
    r1.metric(
        label="Low Risk",
        value=str(low_n),
        delta=f"{(low_n / total_n) * 100:.0f}% of scope",
        delta_color="off",
        delta_arrow="off",
        border=True,
    )
    r2.metric(
        label="Medium Risk",
        value=str(med_n),
        delta=f"{(med_n / total_n) * 100:.0f}% of scope",
        delta_color="off",
        delta_arrow="off",
        border=True,
    )
    r3.metric(
        label="High Risk",
        value=str(high_n),
        delta=f"{(high_n / total_n) * 100:.0f}% of scope",
        delta_color="off",
        delta_arrow="off",
        border=True,
    )

    st.caption(
        "Rule uses real monthly % price changes and 12-month rolling Z-score thresholds; "
        "it is not a calibrated probability or Isolation Forest risk score."
    )

# 6. Section: Alerts table
with st.container(border=True):
    at_l, at_r = st.columns([4, 1], vertical_alignment="center")
    with at_l:
        st.subheader("Alerts table")
        st.caption(f"Latest commodity observations ({end_date.strftime('%B %Y')}) and rule-based alert reasons")

    table_df = snapshot_df.copy()
    if selected_risk != "All risk levels":
        table_df = table_df[table_df["risk_level"] == selected_risk]
    table_df = table_df.sort_values(["risk_rank", "abs_mom_pct"], ascending=[True, False])

    with at_r:
        st.caption(f"{len(table_df)} rows")

    if table_df.empty:
        st.info(f"No commodities match Risk level = '{selected_risk}' in the current scope.")
    else:
        badge_map = {"High": "🔴 High", "Medium": "🟡 Medium", "Low": "🟢 Low"}
        display_alerts = pd.DataFrame(
            {
                "Commodity": table_df["filter_en"],
                "ID": table_df["filter_id"],
                "Category": table_df["category_en"],
                "Current Price (EGP)": table_df["price"],
                "Monthly Change (%)": table_df["mom_pct"],
                "Z-Score": table_df["z_score"],
                "Risk Level": table_df["risk_level"].map(badge_map),
                "Alert Reason": table_df["alert_reason"],
                "Date": table_df["date"].dt.strftime("%b %Y"),
            }
        )
        st.dataframe(
            display_alerts,
            hide_index=True,
            width="stretch",
            height=420,
            column_config={
                "Current Price (EGP)": st.column_config.NumberColumn(
                    "Current Price (EGP)", format="%.2f"
                ),
                "Monthly Change (%)": st.column_config.NumberColumn(
                    "Monthly Change (%)", format="%+.1f%%"
                ),
                "Z-Score": st.column_config.NumberColumn("Z-Score", format="%+.2f"),
            },
        )

    st.caption(
        "Risk levels and alert reasons are computed deterministically from master_food_dataset.csv price changes."
    )

# 7. Bottom Two-Column Row: Risk alert trend + Price anomaly review
bot_l, bot_r = st.columns([1.0, 1.0], gap="medium")

with bot_l:
    with st.container(border=True):
        tl, tr = st.columns([3.5, 1.3], vertical_alignment="center")
        with tl:
            st.subheader("Risk alert trend")
            st.caption("Number of High-risk signals by month")
        with tr:
            st.caption("PROTOTYPE TREND")

        trend_df = (
            scoped_window.assign(is_high=(scoped_window["risk_level"] == "High").astype(int))
            .groupby("date", as_index=False)["is_high"]
            .sum()
            .rename(columns={"is_high": "high_count"})
            .sort_values("date")
        )
        st.plotly_chart(
            render_alert_trend_chart(trend_df),
            width="stretch",
            config={"displayModeBar": False},
        )
        st.caption(
            "Category, commodity, and date filters apply. Risk-level filter does not change the High-risk trend."
        )

with bot_r:
    with st.container(border=True):
        pl, pr = st.columns([3.5, 1.3], vertical_alignment="center")
        with pl:
            st.subheader("Price anomaly review")
        with pr:
            st.caption("Z-SCORE ANOMALY")

        # Default to highest-priority commodity in current filtered table (or snapshot if empty)
        priority_source = table_df if not table_df.empty else snapshot_df.sort_values(
            ["risk_rank", "abs_mom_pct"], ascending=[True, False]
        )
        priority_options = priority_source["filter_en"].tolist()
        top_priority_comm = priority_options[0]

        review_comm = st.selectbox(
            "Commodity under review (defaults to highest-priority signal)",
            options=priority_options,
            index=0,
            key="ar_review_comm",
        )
        st.caption(f"{review_comm} · Top priority: {top_priority_comm}")

        comm_window = scoped_window[scoped_window["filter_en"] == review_comm].sort_values("date")
        st.plotly_chart(
            render_anomaly_review_chart(comm_window, review_comm),
            width="stretch",
            config={"displayModeBar": False},
        )
        st.caption(
            f"Highlighted circles mark months where |Z| ≥ {Z_ANOMALY_THRESHOLD:.1f} (12-month rolling window). "
            "An anomaly flag indicates statistical deviation, not confirmed market manipulation."
        )

# 8. Review Queue Disclaimer & Diagnostic Expanders
st.info(
    "**Use alerts as a review queue, not a conclusion.** Large or unusual price movements can reflect "
    "data-quality issues, seasonality, unit shifts, supply conditions, or macroeconomic shocks. "
    "Verify source records before escalation.",
    icon=":material/info:",
)

with st.expander("View rolling Z-score and anomaly fields"):
    z_disp = snapshot_df.sort_values(["risk_rank", "abs_mom_pct"], ascending=[True, False])[
        [
            "filter_en",
            "category_en",
            "date",
            "price",
            "prev_price",
            "mom_pct",
            "roll_mean_12m",
            "roll_std_12m",
            "z_score",
            "is_anomaly",
        ]
    ].copy()
    z_disp["date"] = z_disp["date"].dt.strftime("%Y-%m")
    z_disp = z_disp.rename(
        columns={
            "filter_en": "Commodity",
            "category_en": "Category",
            "date": "Month",
            "price": "Price (EGP)",
            "prev_price": "Prev Price (EGP)",
            "mom_pct": "MoM (%)",
            "roll_mean_12m": "12m Roll Mean (%)",
            "roll_std_12m": "12m Roll Std (%)",
            "z_score": "Z-Score",
            "is_anomaly": "Anomaly (|Z| ≥ 2.0)",
        }
    )
    st.dataframe(
        z_disp,
        hide_index=True,
        width="stretch",
        column_config={
            "Price (EGP)": st.column_config.NumberColumn("Price (EGP)", format="%.2f"),
            "Prev Price (EGP)": st.column_config.NumberColumn("Prev Price (EGP)", format="%.2f"),
            "MoM (%)": st.column_config.NumberColumn("MoM (%)", format="%+.2f%%"),
            "12m Roll Mean (%)": st.column_config.NumberColumn("12m Roll Mean (%)", format="%+.2f%%"),
            "12m Roll Std (%)": st.column_config.NumberColumn("12m Roll Std (%)", format="%.2f%%"),
            "Z-Score": st.column_config.NumberColumn("Z-Score", format="%+.2f"),
        },
    )

with st.expander("Alert logic & implementation handoff — Thresholds, gaps & Isolation Forest roadmap"):
    st.markdown(
        f"""
        - **Data Source (`data/processed/master_food_dataset.csv`)**: Monthly CAPMAS prices across **57 commodities** and **138 months** (Jan 2015 – Jun 2026). Note that CAPMAS prices are national (there is no regional price column, so a region filter is omitted).
        - **Current Statistical Prototype Rules**:
          - **Rolling 12-Month Z-Score**: `(mom_pct_t - mean(mom_pct_{{t-12..t-1}})) / std(mom_pct_{{t-12..t-1}})` (minimum 6 prior months). Months with `|Z| ≥ {Z_ANOMALY_THRESHOLD:.1f}` are flagged as price anomalies.
          - **Risk Tiers**:
            - `High`: `|MoM %| > {HIGH_MOM_THRESHOLD:.0f}%` OR `|Z| ≥ {Z_ANOMALY_THRESHOLD:.1f}`
            - `Medium`: `{MEDIUM_MOM_THRESHOLD:.0f}% < |MoM %| ≤ {HIGH_MOM_THRESHOLD:.0f}%` (and not flagged as an anomaly)
            - `Low`: `|MoM %| ≤ {MEDIUM_MOM_THRESHOLD:.0f}%` (and not flagged as an anomaly)
        - **Missing ML Integration**: `notebooks/05_anomaly_and_shap_xai.ipynb` and `src/predictor.py` specify an **Isolation Forest** anomaly detector (`models/` currently contains only `.gitkeep`). Once trained and connected, Isolation Forest anomaly scores can replace or augment the rule-based tiers above.
        """
    )

st.caption("FoodWise Egypt · Turning market data into food security insights.")
