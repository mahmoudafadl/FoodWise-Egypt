"""Market Overview page for the FoodWise Egypt Streamlit dashboard."""
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
    page_title="Market Overview | FoodWise Egypt",
    page_icon="📊",
    layout="wide",
)


def classify_risk(abs_pct: float | None) -> tuple[str, int]:
    """
    Prototype rule-based risk classifier on absolute MoM % price change.

    Returns (label, tier_code) where tier_code is 0=Low, 1=Moderate, 2=High, -1=N/A.
    """
    if abs_pct is None or pd.isna(abs_pct):
        return "N/A", -1
    if abs_pct > 5.0:
        return "High", 2
    if abs_pct > 2.0:
        return "Moderate", 1
    return "Low", 0


@st.cache_data(show_spinner=False)
def prepare_market_data(raw_df: pd.DataFrame) -> pd.DataFrame:
    """Compute per-commodity previous price, MoM % change, and absolute volatility."""
    df = raw_df.sort_values(["filter_id", "date"]).copy()
    df["prev_price"] = df.groupby("filter_id")["price"].shift(1)
    df["mom_pct"] = df.groupby("filter_id")["price"].pct_change() * 100.0
    df["abs_mom_pct"] = df["mom_pct"].abs()
    return df


def build_price_index_series(
    df: pd.DataFrame,
    selected_date: pd.Timestamp,
    category: str,
    commodity: str,
) -> tuple[pd.DataFrame, pd.Timestamp]:
    """
    Build an equal-weight normalized price index (base = 100 at window start).

    Uses a 12-month lookback window (up to 13 monthly points ending at selected_date).
    Each commodity is normalized to its own base-month price before averaging.
    """
    scoped = df[df["date"] <= selected_date].copy()
    if category != "All categories":
        scoped = scoped[scoped["category_en"] == category]
    if commodity != "All commodities":
        scoped = scoped[scoped["filter_en"] == commodity]

    available_dates = sorted(scoped["date"].unique())
    window_dates = available_dates[-13:]
    base_date = pd.Timestamp(window_dates[0])
    window_df = scoped[scoped["date"].isin(window_dates)].copy()

    base_prices = (
        window_df[window_df["date"] == base_date]
        .set_index("filter_id")["price"]
        .to_dict()
    )
    window_df["base_price"] = window_df["filter_id"].map(base_prices)
    window_df = window_df[window_df["base_price"] > 0].copy()
    window_df["index_val"] = (window_df["price"] / window_df["base_price"]) * 100.0

    series = (
        window_df.groupby("date", as_index=False)["index_val"]
        .mean()
        .sort_values("date")
    )
    return series, base_date


def render_trend_chart(series: pd.DataFrame, base_date: pd.Timestamp) -> go.Figure:
    """Create a clean Plotly area/line chart for the equal-weight food price index."""
    fig = go.Figure()
    fig.add_trace(
        go.Scatter(
            x=series["date"],
            y=series["index_val"],
            name=f"Food price index ({base_date.strftime('%b %Y')} = 100)",
            mode="lines+markers",
            line=dict(color="#1E4D3B", width=2.5),
            marker=dict(size=6, color="#1E4D3B"),
            fill="tozeroy",
            fillcolor="rgba(30, 77, 59, 0.10)",
            hovertemplate=(
                "<b>%{x|%b %Y}</b><br>"
                f"Index ({base_date.strftime('%b %Y')} = 100): "
                "<b>%{y:.1f}</b><extra></extra>"
            ),
        )
    )

    y_min = float(series["index_val"].min())
    y_max = float(series["index_val"].max())
    pad = max((y_max - y_min) * 0.25, 2.0)

    fig.update_layout(
        height=300,
        margin=dict(l=12, r=12, t=28, b=12),
        hovermode="x unified",
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="left", x=0),
        xaxis=dict(
            title=None,
            showgrid=False,
            tickformat="%b %Y",
        ),
        yaxis=dict(
            title="Normalized price index",
            range=[max(0.0, y_min - pad), y_max + pad],
            zeroline=False,
        ),
    )
    return fig


def render_risk_heatmap(
    df: pd.DataFrame,
    selected_date: pd.Timestamp,
    selected_category: str,
) -> go.Figure:
    """
    Create a Plotly annotated heatmap of category-level price pressure (avg |MoM| %).

    Displays the 6-month window ending at selected_date across all 8 food categories.
    """
    all_dates = sorted(df["date"].unique())
    idx = all_dates.index(selected_date)
    window_dates = all_dates[max(0, idx - 5) : idx + 1]

    hm_df = df[df["date"].isin(window_dates)]
    piv = hm_df.pivot_table(
        index="category_en",
        columns="date",
        values="abs_mom_pct",
        aggfunc="mean",
    )

    categories = sorted(df["category_en"].unique(), reverse=True)
    if selected_category != "All categories" and selected_category in categories:
        # Put selected category at the top of the heatmap for immediate visibility
        categories = [c for c in categories if c != selected_category] + [selected_category]

    x_labels = [pd.Timestamp(dt).strftime("%b %Y") for dt in window_dates]
    y_labels = [
        f"★ {cat}" if (selected_category != "All categories" and cat == selected_category) else cat
        for cat in categories
    ]

    z_tiers: list[list[float]] = []
    text_grid: list[list[str]] = []
    custom_hover: list[list[str]] = []

    for cat in categories:
        tier_row: list[float] = []
        text_row: list[str] = []
        hover_row: list[str] = []
        for dt in window_dates:
            val = piv.loc[cat, dt] if (cat in piv.index and dt in piv.columns) else None
            lbl, tier = classify_risk(val)
            if val is None or pd.isna(val):
                tier_row.append(-1.0)
                text_row.append("N/A")
                hover_row.append(f"<b>{cat}</b><br>{pd.Timestamp(dt):%b %Y}<br>Volatility: N/A")
            else:
                tier_row.append(float(tier))
                text_row.append(f"{val:.1f}%<br>{lbl}")
                hover_row.append(
                    f"<b>{cat}</b><br>{pd.Timestamp(dt):%b %Y}<br>"
                    f"Avg |MoM|: <b>{val:.2f}%</b><br>Risk tier (prototype): <b>{lbl}</b>"
                )
        z_tiers.append(tier_row)
        text_grid.append(text_row)
        custom_hover.append(hover_row)

    # Discrete colorscale mapped to [-1, 0, 1, 2] -> [-1..2] normalized to [0..1]
    colorscale = [
        [0.00, "#ECEAE2"], [0.30, "#ECEAE2"],  # -1: N/A
        [0.31, "#D8EEDB"], [0.58, "#D8EEDB"],  #  0: Low
        [0.59, "#FBE8B6"], [0.85, "#FBE8B6"],  #  1: Moderate
        [0.86, "#F8D0CE"], [1.00, "#F8D0CE"],  #  2: High
    ]

    fig = go.Figure(
        data=go.Heatmap(
            z=z_tiers,
            x=x_labels,
            y=y_labels,
            zmin=-1,
            zmax=2,
            colorscale=colorscale,
            showscale=False,
            text=text_grid,
            texttemplate="%{text}",
            textfont=dict(size=11, color="#1F2D26"),
            customdata=custom_hover,
            hovertemplate="%{customdata}<extra></extra>",
            xgap=4,
            ygap=4,
        )
    )

    fig.update_layout(
        height=355,
        margin=dict(l=8, r=8, t=12, b=12),
        xaxis=dict(side="top", showgrid=False, tickfont=dict(size=11)),
        yaxis=dict(showgrid=False, tickfont=dict(size=11)),
    )
    return fig


def format_movers_table(movers_df: pd.DataFrame) -> pd.DataFrame:
    """Format Top Price Movers columns cleanly for st.dataframe."""
    out = movers_df[["filter_en", "category_en", "price", "mom_pct"]].copy()

    def _direction(val: float) -> str:
        if pd.isna(val):
            return "N/A"
        if val > 0.05:
            return "↑ Rising"
        if val < -0.05:
            return "↓ Falling"
        return "• Stable"

    out["Direction"] = out["mom_pct"].apply(_direction)
    out = out.rename(
        columns={
            "filter_en": "Commodity",
            "category_en": "Category",
            "price": "Current Price (EGP)",
            "mom_pct": "Monthly Change (%)",
        }
    )
    return out


# ---------------------------------------------------------------------------
# Page Execution
# ---------------------------------------------------------------------------
raw_df = load_master_data()
if raw_df.empty:
    st.title("Market Overview")
    st.warning("Dataset not loaded. Please run `python scripts/build_dataset.py` first.")
    st.stop()

df = prepare_market_data(raw_df)
all_dates = sorted(df["date"].unique())
latest_date = pd.Timestamp(all_dates[-1])
earliest_date = pd.Timestamp(all_dates[0])
date_options = [pd.Timestamp(d) for d in reversed(all_dates)]
categories_list = ["All categories"] + sorted(df["category_en"].unique().tolist())

if "mo_month" not in st.session_state or st.session_state["mo_month"] not in date_options:
    st.session_state["mo_month"] = latest_date
if "mo_category" not in st.session_state or st.session_state["mo_category"] not in categories_list:
    st.session_state["mo_category"] = "All categories"
if "mo_commodity" not in st.session_state:
    st.session_state["mo_commodity"] = "All commodities"


def _reset_filters() -> None:
    st.session_state["mo_month"] = latest_date
    st.session_state["mo_category"] = "All categories"
    st.session_state["mo_commodity"] = "All commodities"


# Precompute export dataset matching current filter state
sel_month_preview = st.session_state["mo_month"]
sel_cat_preview = st.session_state["mo_category"]
sel_comm_preview = st.session_state["mo_commodity"]

export_df = df[df["date"] == sel_month_preview].copy()
if sel_cat_preview != "All categories":
    export_df = export_df[export_df["category_en"] == sel_cat_preview]
if sel_comm_preview != "All commodities" and sel_comm_preview in export_df["filter_en"].values:
    export_df = export_df[export_df["filter_en"] == sel_comm_preview]
export_df["rule_based_risk"] = export_df["abs_mom_pct"].apply(lambda v: classify_risk(v)[0])
export_cols = [
    "filter_id",
    "filter_en",
    "category_en",
    "date",
    "price",
    "prev_price",
    "mom_pct",
    "abs_mom_pct",
    "rule_based_risk",
]

# 1. Header
st.caption("FOOD PRICE INTELLIGENCE & EARLY WARNING SYSTEM")
hdr_left, hdr_right = st.columns([5, 1.2], vertical_alignment="center")
with hdr_left:
    st.title("Market Overview")
    st.caption("A snapshot of food prices and market conditions in Egypt.")
with hdr_right:
    st.download_button(
        label="Export data",
        icon=":material/download:",
        data=export_df[export_cols].to_csv(index=False).encode("utf-8"),
        file_name=f"foodwise_market_overview_{sel_month_preview.strftime('%Y_%m')}.csv",
        mime="text/csv",
        width="stretch",
    )

# 2. Data & Prototype Notice Banner
st.info(
    f"Live CAPMAS & CBE dataset connected ({df['filter_id'].nunique()} commodities, "
    f"{earliest_date.strftime('%b %Y')} – {latest_date.strftime('%b %Y')}). "
    "Market Risk Level and the Market Risk Heatmap use rule-based prototype thresholds "
    "(|MoM| ≤ 2% Low, 2–5% Moderate, > 5% High) until the ML risk model in `src/predictor.py` is connected.",
    icon=":material/info:",
)

# 3. Filter Bar
with st.container(border=True):
    f1, f2, f3, f4, f5 = st.columns([1.3, 1.4, 1.5, 1.1, 0.9], vertical_alignment="bottom")

    with f1:
        selected_date: pd.Timestamp = st.selectbox(
            "Month",
            options=date_options,
            format_func=lambda d: d.strftime("%B %Y"),
            key="mo_month",
        )

    with f2:
        selected_category: str = st.selectbox(
            "Food category",
            options=categories_list,
            key="mo_category",
        )

    if selected_category == "All categories":
        commodity_options = ["All commodities"] + sorted(df["filter_en"].unique().tolist())
    else:
        commodity_options = ["All commodities"] + sorted(
            df.loc[df["category_en"] == selected_category, "filter_en"].unique().tolist()
        )

    if st.session_state["mo_commodity"] not in commodity_options:
        st.session_state["mo_commodity"] = "All commodities"

    with f3:
        selected_commodity: str = st.selectbox(
            "Commodity (optional)",
            options=commodity_options,
            key="mo_commodity",
        )

    selected_idx = all_dates.index(selected_date)
    prev_date = pd.Timestamp(all_dates[selected_idx - 1]) if selected_idx > 0 else None

    with f4:
        st.caption("Compared with")
        st.markdown(f"**{prev_date.strftime('%B %Y') if prev_date is not None else 'No prior month'}**")

    with f5:
        st.button("Reset filters", on_click=_reset_filters, width="stretch")

# Filter data for selected scope
month_df = df[df["date"] == selected_date].copy()
if selected_category != "All categories":
    month_df = month_df[month_df["category_en"] == selected_category]
if selected_commodity != "All commodities":
    month_df = month_df[month_df["filter_en"] == selected_commodity]

n_commodities = int(month_df["filter_id"].nunique())
n_categories = int(month_df["category_en"].nunique())
valid_mom = month_df.dropna(subset=["mom_pct"])
n_comparable = len(valid_mom)

if n_comparable > 0:
    avg_mom = float(valid_mom["mom_pct"].mean())
    avg_abs_mom = float(valid_mom["abs_mom_pct"].mean())
    inc_count = int((valid_mom["mom_pct"] > 0).sum())
    inc_share = (inc_count / n_comparable) * 100.0
    avg_mom_str = f"{avg_mom:+.1f}%"
    inc_str = f"{inc_count} / {n_comparable}"
    inc_delta = f"{inc_share:.0f}% of matched basket"
else:
    avg_mom = None
    avg_abs_mom = None
    inc_count = 0
    inc_share = 0.0
    avg_mom_str = "N/A"
    inc_str = f"0 / {n_commodities}"
    inc_delta = "No prior month"

risk_label, _ = classify_risk(avg_abs_mom)
cat_word = "category" if n_categories == 1 else "categories"

# 4. Section: The market at a glance
glance_l, glance_r = st.columns([3, 1.5], vertical_alignment="bottom")
with glance_l:
    st.subheader("The market at a glance")
with glance_r:
    st.caption(f"{selected_date.strftime('%B %Y')} · CAPMAS master data · Prototype risk rules")

k1, k2, k3, k4 = st.columns(4)
with k1:
    st.metric(
        label="Average Food Price Change (MoM)",
        value=avg_mom_str,
        delta=f"Equal-weight mean · {n_comparable} items",
        delta_color="off",
        delta_arrow="off",
        help="Equal-weight mean of commodity-level Month-over-Month % changes in the selected scope.",
        border=True,
        height="stretch",
    )
with k2:
    st.metric(
        label="Number of Commodities Monitored",
        value=str(n_commodities),
        delta=f"{n_categories} {cat_word} in selected scope",
        delta_color="off",
        delta_arrow="off",
        help="Count of unique filter_id commodities matching the current filters.",
        border=True,
        height="stretch",
    )
with k3:
    st.metric(
        label="Commodities with Price Increases",
        value=inc_str,
        delta=inc_delta,
        delta_color="off",
        delta_arrow="off",
        delta_description=(
            f"vs. {prev_date.strftime('%b %Y')}" if prev_date is not None else "Earliest month"
        ),
        help="Number of commodities with positive MoM price change compared with the previous month.",
        border=True,
        height="stretch",
    )
with k4:
    st.metric(
        label="Market Risk Level",
        value=risk_label,
        delta=(
            f"Rule-based prototype · Avg |MoM| {avg_abs_mom:.1f}%"
            if avg_abs_mom is not None
            else "No prior month"
        ),
        delta_color="off",
        delta_arrow="off",
        help="Rule-based prototype indicator on average |MoM| volatility (Low ≤ 2%, Moderate 2–5%, High > 5%).",
        border=True,
        height="stretch",
    )

# 5. Section: Overall food price trend
index_series, base_date = build_price_index_series(
    df=df,
    selected_date=selected_date,
    category=selected_category,
    commodity=selected_commodity,
)
latest_idx_val = float(index_series["index_val"].iloc[-1]) if not index_series.empty else 100.0

with st.container(border=True):
    tr_left, tr_right = st.columns([4, 1.3], vertical_alignment="center")
    with tr_left:
        st.subheader("Overall food price trend")
        st.caption(
            f"{selected_category} · {selected_commodity} · "
            f"Equal-weight normalized price index ({base_date.strftime('%b %Y')} = 100)"
        )
    with tr_right:
        st.metric(
            label="Latest index value",
            value=f"{latest_idx_val:.1f} pts",
            delta=f"{latest_idx_val - 100.0:+.1f}% vs. {base_date.strftime('%b %Y')}",
            delta_color="inverse",
        )

    if len(index_series) <= 1:
        st.info("Only one month is available up to the selected date; choose a later month to view the multi-month trend.")
    else:
        st.plotly_chart(
            render_trend_chart(index_series, base_date),
            width="stretch",
            config={"displayModeBar": False},
        )

    st.caption(
        f"Each commodity is normalized to {base_date.strftime('%B %Y')} (= 100) before averaging. "
        "Raw prices across commodities are never averaged directly."
    )

# 6. Section: Bottom two-column row (Top Price Movers + Market Risk Heatmap)
left_col, right_col = st.columns([1.15, 1.0], gap="medium")

with left_col:
    with st.container(border=True):
        m_hdr_l, m_hdr_r = st.columns([3.5, 1], vertical_alignment="center")
        with m_hdr_l:
            st.subheader("Top Price Movers")
        with m_hdr_r:
            show_all_movers = st.toggle("Show all", value=False, key="mo_show_all_movers")

        st.caption(f"Largest absolute monthly changes · {selected_date.strftime('%B %Y')} · EGP")

        if valid_mom.empty:
            st.info(
                f"No prior month is available to compute monthly price movers for {selected_date.strftime('%B %Y')}."
            )
        else:
            sorted_movers = valid_mom.sort_values("abs_mom_pct", ascending=False)
            display_movers = sorted_movers if show_all_movers else sorted_movers.head(6)
            formatted_df = format_movers_table(display_movers)

            st.dataframe(
                formatted_df,
                hide_index=True,
                width="stretch",
                height=280 if not show_all_movers else 360,
                column_config={
                    "Commodity": st.column_config.TextColumn("Commodity"),
                    "Category": st.column_config.TextColumn("Category"),
                    "Current Price (EGP)": st.column_config.NumberColumn(
                        "Current Price (EGP)", format="%.2f"
                    ),
                    "Monthly Change (%)": st.column_config.NumberColumn(
                        "Monthly Change (%)", format="%+.1f%%"
                    ),
                    "Direction": st.column_config.TextColumn("Direction"),
                },
            )
            st.caption(
                f"Showing {len(display_movers)} of {n_comparable} comparable commodities · Prices in EGP (CAPMAS)"
            )

with right_col:
    with st.container(border=True):
        st.subheader("Market Risk Heatmap")
        hm_scope_note = (
            f"Highlighting {selected_category} (★)"
            if selected_category != "All categories"
            else "All 8 food categories"
        )
        st.caption(
            f"Category-level price pressure by month (avg |MoM| volatility · Prototype rules) · {hm_scope_note}"
        )

        st.plotly_chart(
            render_risk_heatmap(
                df=df,
                selected_date=selected_date,
                selected_category=selected_category,
            ),
            width="stretch",
            config={"displayModeBar": False},
        )
        st.caption(
            "Low ≤ 2% · Moderate 2–5% · High > 5% · Rule-based prototype price-pressure view, "
            "not a trained ML prediction model."
        )

# 7. Methodology & dataset fields
with st.expander("How these metrics are calculated — Methodology & dataset fields"):
    st.markdown(
        """
        - **Data Source (`data/processed/master_food_dataset.csv`)**: Monthly CAPMAS retail food prices (`price`, `filter_id`, `filter_en`, `category_en`, `date`) harmonized with CBE macro indicators and regional weather across **57 commodities** and **138 months** (Jan 2015 – Jun 2026).
        - **Average Food Price Change (MoM)**: Equal-weight mean of commodity-level percentage changes `(price_t - price_{t-1}) / price_{t-1} * 100` within the active filters.
        - **Overall Food Price Trend**: Each commodity in scope is indexed to `100` at the start of the 12-month lookback window (`price_t / price_base * 100`) and then averaged across commodities, preventing high-priced items from dominating low-priced items.
        - **Top Price Movers**: Commodities ranked descending by absolute monthly percentage change `|MoM %|`.
        - **Market Risk Level & Heatmap (Prototype / Mock ML Placeholder)**: Because the Isolation Forest / ML risk scoring pipeline in `src/predictor.py` is not yet connected, risk levels use transparent rule-based thresholds on average `|MoM %|` volatility (`Low ≤ 2%`, `Moderate 2–5%`, `High > 5%`).
        """
    )

st.caption("FoodWise Egypt · Turning market data into food security insights.")
