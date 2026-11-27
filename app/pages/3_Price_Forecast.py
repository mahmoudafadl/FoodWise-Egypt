"""Price Forecast page for the FoodWise Egypt Streamlit dashboard."""
from pathlib import Path
import sys

import pandas as pd
import plotly.graph_objects as go
import streamlit as st

APP_DIR = Path(__file__).resolve().parents[1]
ROOT_DIR = APP_DIR.parent
for p in (str(APP_DIR), str(ROOT_DIR)):
    if p not in sys.path:
        sys.path.insert(0, p)

from src.features import build_features
from src.predictor import load_model_artifacts, predict_next_month
from utils import load_master_data

st.set_page_config(
    page_title="Price Forecast | FoodWise Egypt",
    page_icon="🔮",
    layout="wide",
)

CORR_FEATURE_LABELS = [
    ("usd_mom_pct", "USD exchange rate (MoM %)"),
    ("headline_inflation_mom", "Headline inflation (MoM %)"),
    ("lag1_mom", "Recent commodity price (1m lag)"),
    ("core_inflation_mom", "Core inflation (MoM %)"),
    ("cbe_deposit_rate", "CBE deposit rate (%)"),
    ("nat_temp_anomaly", "Temperature anomaly (°C)"),
    ("cbe_lending_rate", "CBE lending rate (%)"),
    ("cairo_temp_mean", "Cairo temperature (°C)"),
]


@st.cache_data(show_spinner=False)
def prepare_forecast_dataset(raw_df: pd.DataFrame) -> pd.DataFrame:
    """Prepare feature-enriched dataframe and historical MoM % changes."""
    featured = build_features(raw_df, is_inference=True)
    featured = featured.sort_values(["filter_id", "date"]).copy()
    featured["mom_pct"] = featured.groupby("filter_id")["price"].pct_change() * 100.0
    featured["lag1_mom"] = featured.groupby("filter_id")["mom_pct"].shift(1)
    return featured


def compute_historical_correlations(commodity_df: pd.DataFrame) -> pd.DataFrame:
    """
    Compute real Pearson correlations between macro/weather/lag features and MoM % change.

    Used as a transparent analytical reference while SHAP TreeExplainer is unavailable.
    """
    rows = []
    for col, label in CORR_FEATURE_LABELS:
        if col in commodity_df.columns:
            corr_val = commodity_df["mom_pct"].corr(commodity_df[col])
            val = float(corr_val) if pd.notna(corr_val) else 0.0
        else:
            val = 0.0
        rows.append({"Driver": label, "Value": val})

    out = pd.DataFrame(rows)
    out["AbsValue"] = out["Value"].abs()
    out = out.sort_values("AbsValue", ascending=True).drop(columns=["AbsValue"])
    return out


def render_forecast_chart(
    hist_df: pd.DataFrame,
    forecast_date: pd.Timestamp,
    predicted_price: float,
    model_connected: bool,
) -> go.Figure:
    """Render 12-month historical prices alongside the next-month forecast point."""
    fig = go.Figure()

    fig.add_trace(
        go.Scatter(
            x=hist_df["date"],
            y=hist_df["price"],
            name="Historical price (CAPMAS)",
            mode="lines+markers",
            line=dict(color="#1E4D3B", width=2.5),
            marker=dict(size=6, color="#1E4D3B"),
            hovertemplate="<b>%{x|%b %Y}</b><br>Actual: <b>%{y:.2f} EGP</b><extra></extra>",
        )
    )

    last_hist_date = pd.Timestamp(hist_df["date"].iloc[-1])
    last_hist_price = float(hist_df["price"].iloc[-1])
    forecast_label = (
        "Next-month forecast" if model_connected else "Next-month baseline (no model)"
    )

    fig.add_trace(
        go.Scatter(
            x=[last_hist_date, forecast_date],
            y=[last_hist_price, predicted_price],
            name=forecast_label,
            mode="lines+markers",
            line=dict(color="#B4822C", width=2.5, dash="dot"),
            marker=dict(size=8, color="#B4822C"),
            hovertemplate=(
                "<b>%{x|%b %Y}</b><br>"
                f"{forecast_label}: <b>%{{y:.2f}} EGP</b><extra></extra>"
            ),
        )
    )

    # Shaded forecast window from last observation through forecast month + half-month pad
    shade_end = forecast_date + pd.DateOffset(days=15)
    fig.add_vrect(
        x0=last_hist_date,
        x1=shade_end,
        fillcolor="rgba(180, 130, 44, 0.12)",
        line_width=0,
        annotation_text="FORECAST" if model_connected else "MODEL NOT CONNECTED",
        annotation_position="top left",
        annotation_font=dict(size=10, color="#8A641E"),
    )

    all_y = list(hist_df["price"].astype(float)) + [float(predicted_price)]
    y_min, y_max = min(all_y), max(all_y)
    pad = max((y_max - y_min) * 0.25, 1.0)

    fig.update_layout(
        height=320,
        margin=dict(l=12, r=12, t=28, b=12),
        hovermode="x unified",
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="left", x=0),
        xaxis=dict(title=None, showgrid=False, tickformat="%b %Y"),
        yaxis=dict(
            title="Price (EGP)",
            range=[max(0.0, y_min - pad), y_max + pad],
            zeroline=False,
        ),
    )
    return fig


def render_drivers_chart(drivers_df: pd.DataFrame, is_shap: bool) -> go.Figure:
    """Render horizontal diverging bar chart for SHAP drivers or feature correlations."""
    colors = [
        "#B4822C" if v >= 0 else "#1E4D3B"
        for v in drivers_df["Value"]
    ]
    x_title = (
        "SHAP contribution to forecast price (EGP)"
        if is_shap
        else "Historical Pearson correlation with MoM % price change (r)"
    )

    fig = go.Figure(
        go.Bar(
            x=drivers_df["Value"],
            y=drivers_df["Driver"],
            orientation="h",
            marker_color=colors,
            text=[f"{v:+.2f}" for v in drivers_df["Value"]],
            textposition="outside",
            hovertemplate="<b>%{y}</b><br>Value: <b>%{x:+.3f}</b><extra></extra>",
        )
    )

    max_abs = max(float(drivers_df["Value"].abs().max()), 0.25)
    bound = max_abs * 1.25 if is_shap else 1.0

    fig.add_vline(x=0, line_width=1, line_color="#9CA3AF")
    fig.update_layout(
        height=320,
        margin=dict(l=10, r=24, t=12, b=12),
        showlegend=False,
        xaxis=dict(title=x_title, range=[-bound, bound], zeroline=False),
        yaxis=dict(title=None),
    )
    return fig


# ---------------------------------------------------------------------------
# Page Execution
# ---------------------------------------------------------------------------
raw_df = load_master_data()
if raw_df.empty:
    st.title("Price Forecast")
    st.warning("Dataset not loaded. Please run `python scripts/build_dataset.py` first.")
    st.stop()

df = prepare_forecast_dataset(raw_df)
model_artifacts = load_model_artifacts()
model_connected = model_artifacts is not None

all_dates = [pd.Timestamp(d) for d in sorted(df["date"].unique())]
latest_date = all_dates[-1]
next_forecast_date = latest_date + pd.DateOffset(months=1)
# Allow forward forecast month (Jul 2026) plus historical walk-forward months (Feb 2015..Jun 2026)
forecast_month_options = [next_forecast_date] + list(reversed(all_dates[1:]))

categories_list = ["All categories"] + sorted(df["category_en"].unique().tolist())
all_commodities = sorted(df["filter_en"].unique().tolist())
default_commodity = (
    "Unpacked Wheat Flour 72%"
    if "Unpacked Wheat Flour 72%" in all_commodities
    else all_commodities[0]
)

if "pf_category" not in st.session_state or st.session_state["pf_category"] not in categories_list:
    st.session_state["pf_category"] = "All categories"
if "pf_commodity" not in st.session_state or st.session_state["pf_commodity"] not in all_commodities:
    st.session_state["pf_commodity"] = default_commodity
if "pf_forecast_month" not in st.session_state or st.session_state["pf_forecast_month"] not in forecast_month_options:
    st.session_state["pf_forecast_month"] = next_forecast_date


def _reset_filters() -> None:
    st.session_state["pf_category"] = "All categories"
    st.session_state["pf_commodity"] = default_commodity
    st.session_state["pf_forecast_month"] = next_forecast_date
    st.session_state.pop("pf_scenario_ran", None)


active_cat = st.session_state["pf_category"]
if active_cat == "All categories":
    commodity_options = all_commodities
else:
    commodity_options = sorted(
        df.loc[df["category_en"] == active_cat, "filter_en"].unique().tolist()
    )
if st.session_state["pf_commodity"] not in commodity_options:
    st.session_state["pf_commodity"] = commodity_options[0]

# 1. Header
st.caption("FOOD PRICE INTELLIGENCE & EARLY WARNING SYSTEM")
hdr_left, hdr_right = st.columns([5, 1.3], vertical_alignment="center")
with hdr_left:
    st.title("Price Forecast")
    st.caption("Forecast next-month food prices and understand the main drivers.")

# Filter Bar rendered before export preparation? We can read state or render filter bar after header placeholder,
# so let's compute current selection directly from session_state for the export button, then render controls.
sel_comm_pre = st.session_state["pf_commodity"]
sel_fmonth_pre = st.session_state["pf_forecast_month"]
base_date_pre = max([d for d in all_dates if d < sel_fmonth_pre], default=all_dates[0])

comm_hist_pre = df[(df["filter_en"] == sel_comm_pre) & (df["date"] <= base_date_pre)].sort_values("date")
base_row_pre = comm_hist_pre.iloc[-1:]
base_price_pre = float(base_row_pre["price"].iloc[0])
pred_pre = predict_next_month(base_row_pre, base_price_pre)

export_rows = comm_hist_pre.tail(12)[["date", "filter_en", "category_en", "price", "mom_pct"]].copy()
export_rows["record_type"] = "Historical"
forecast_export = pd.DataFrame(
    [
        {
            "date": sel_fmonth_pre,
            "filter_en": sel_comm_pre,
            "category_en": comm_hist_pre["category_en"].iloc[-1],
            "price": pred_pre["predicted_price"],
            "mom_pct": pred_pre["predicted_pct_change"],
            "record_type": "Model Forecast" if model_connected else "Placeholder (Model Not Connected)",
        }
    ]
)
export_df = pd.concat([export_rows, forecast_export], ignore_index=True)

with hdr_right:
    st.download_button(
        label="Export forecast",
        icon=":material/download:",
        data=export_df.to_csv(index=False).encode("utf-8"),
        file_name=f"foodwise_forecast_{sel_comm_pre.lower().replace(' ', '_')}_{sel_fmonth_pre.strftime('%Y_%m')}.csv",
        mime="text/csv",
        width="stretch",
    )

# 2. Model Status Notice
if not model_connected:
    st.warning(
        "**Model not connected (`models/best_forecast_model.joblib` not found).** "
        "Historical prices and macro baselines below come from `data/processed/master_food_dataset.csv`, "
        "while `src/predictor.py` (`predict_next_month`) currently returns placeholder outputs "
        "(`0.0%` change, empty SHAP `top_drivers`) until the trained model and `shap.TreeExplainer` are saved.",
        icon=":material/warning:",
    )
else:
    st.success(
        "Trained forecasting model loaded from `models/best_forecast_model.joblib`.",
        icon=":material/check_circle:",
    )

# 3. Filter Bar
with st.container(border=True):
    f1, f2, f3, f4, f5 = st.columns([1.6, 1.3, 1.3, 1.0, 0.85], vertical_alignment="bottom")
    with f1:
        selected_commodity: str = st.selectbox(
            "Commodity",
            options=commodity_options,
            key="pf_commodity",
        )
    with f2:
        selected_forecast_month: pd.Timestamp = st.selectbox(
            "Forecast month",
            options=forecast_month_options,
            format_func=lambda d: d.strftime("%B %Y"),
            key="pf_forecast_month",
        )
    with f3:
        selected_category: str = st.selectbox(
            "Category (optional)",
            options=categories_list,
            key="pf_category",
        )
    with f4:
        st.caption("Forecast horizon")
        st.markdown("**1 month ahead**")
    with f5:
        st.button("Reset filters", on_click=_reset_filters, width="stretch")

# Slice historical observations up to the month immediately prior to selected_forecast_month
base_date = max([d for d in all_dates if d < selected_forecast_month], default=all_dates[0])
commodity_full = df[df["filter_en"] == selected_commodity].sort_values("date")
commodity_up_to_base = commodity_full[commodity_full["date"] <= base_date].copy()
hist_12m = commodity_up_to_base.tail(12).copy()

base_row_df = commodity_up_to_base.iloc[-1:]
base_row = base_row_df.iloc[0]
current_price = float(base_row["price"])

# Call project predictor
prediction = predict_next_month(base_row_df, current_price)
pred_price = float(prediction.get("predicted_price", current_price))
pred_pct = float(prediction.get("predicted_pct_change", 0.0))
risk_level = str(prediction.get("risk_level", "Low"))
top_drivers = prediction.get("top_drivers", [])

# Reset scenario outputs & input defaults whenever commodity or forecast month changes
context_token = f"{selected_commodity}|{selected_forecast_month.strftime('%Y-%m')}"
if st.session_state.get("pf_context_token") != context_token:
    st.session_state["pf_context_token"] = context_token
    st.session_state["pf_usd_input"] = round(float(base_row["usd_sell_end"]), 2)
    st.session_state["pf_inf_input"] = round(float(base_row["headline_inflation_mom"]), 2)
    st.session_state["pf_temp_input"] = round(float(base_row["nat_temp_anomaly"]), 2)
    st.session_state["pf_scenario_ran"] = False

# 4. Section: NEXT-MONTH OUTLOOK
with st.container(border=True):
    o_l, o_r = st.columns([4, 1.3], vertical_alignment="center")
    with o_l:
        st.caption("NEXT-MONTH OUTLOOK")
        st.subheader(selected_commodity)
    with o_r:
        st.caption(f"As of {base_date.strftime('%b %Y')}")
        st.caption("MODEL FORECAST" if model_connected else "MODEL NOT CONNECTED")

    k1, k2, k3, k4 = st.columns(4)
    with k1:
        st.metric(
            label="Current Price",
            value=f"{current_price:.2f} EGP",
            delta=f"{base_date.strftime('%b %Y')} · CAPMAS observation",
            delta_color="off",
            delta_arrow="off",
            border=True,
            height="stretch",
        )
    with k2:
        st.metric(
            label="Predicted Next-Month Price",
            value=f"{pred_price:.2f} EGP" if model_connected else "N/A (No model)",
            delta=(
                f"{selected_forecast_month.strftime('%b %Y')} · Model output"
                if model_connected
                else f"Baseline carry-forward: {pred_price:.2f} EGP"
            ),
            delta_color="off",
            delta_arrow="off",
            border=True,
            height="stretch",
        )
    with k3:
        if model_connected:
            dir_word = "Increase" if pred_pct > 0.05 else ("Decrease" if pred_pct < -0.05 else "Stable")
            st.metric(
                label="Predicted Monthly Change",
                value=f"{pred_pct:+.1f}%",
                delta="Compared with Current price",
                delta_color="inverse" if abs(pred_pct) > 0.05 else "off",
                border=True,
                height="stretch",
            )
        else:
            st.metric(
                label="Predicted Monthly Change",
                value="N/A",
                delta=f"Placeholder output: {pred_pct:+.1f}%",
                delta_color="off",
                delta_arrow="off",
                border=True,
                height="stretch",
            )
    with k4:
        if model_connected:
            dir_label = "↗ Increase" if pred_pct > 0.05 else ("↘ Decrease" if pred_pct < -0.05 else "→ Stable")
        else:
            dir_label = "Pending model"
        st.metric(
            label="Expected Direction",
            value=dir_label,
            delta=f"Risk: {risk_level}" if model_connected else "Requires trained model",
            delta_color="off",
            delta_arrow="off",
            border=True,
            height="stretch",
        )

# 5. Section: Historical prices & next-month forecast
with st.container(border=True):
    fc_l, fc_r = st.columns([4, 1.2], vertical_alignment="center")
    with fc_l:
        st.subheader("Historical prices & next-month forecast")
        st.caption(
            f"{selected_commodity} · EGP · {len(hist_12m)} months of historical observations"
        )
    with fc_r:
        st.caption("1-MONTH HORIZON")

    st.plotly_chart(
        render_forecast_chart(
            hist_df=hist_12m,
            forecast_date=selected_forecast_month,
            predicted_price=pred_price,
            model_connected=model_connected,
        ),
        width="stretch",
        config={"displayModeBar": False},
    )

    with st.expander("View historical & forecast values"):
        hist_tbl = hist_12m[["date", "price", "mom_pct"]].copy()
        hist_tbl["Type"] = "Historical (CAPMAS)"
        next_row = pd.DataFrame(
            [
                {
                    "date": selected_forecast_month,
                    "price": pred_price,
                    "mom_pct": pred_pct,
                    "Type": "Model Forecast" if model_connected else "Placeholder (No model)",
                }
            ]
        )
        combined_tbl = pd.concat([hist_tbl, next_row], ignore_index=True)
        # If forecast month is in the historical dataset (walk-forward selection), attach actual for reference
        actual_lookup = commodity_full.set_index("date")["price"].to_dict()
        combined_tbl["Actual (if observed)"] = combined_tbl["date"].map(actual_lookup)
        combined_tbl["Month"] = pd.to_datetime(combined_tbl["date"]).dt.strftime("%b %Y")
        combined_tbl = combined_tbl.rename(
            columns={"price": "Price (EGP)", "mom_pct": "MoM Change (%)"}
        )[["Month", "Type", "Price (EGP)", "MoM Change (%)", "Actual (if observed)"]]

        st.dataframe(
            combined_tbl,
            hide_index=True,
            width="stretch",
            column_config={
                "Price (EGP)": st.column_config.NumberColumn("Price (EGP)", format="%.2f"),
                "MoM Change (%)": st.column_config.NumberColumn("MoM Change (%)", format="%+.1f%%"),
                "Actual (if observed)": st.column_config.NumberColumn(
                    "Actual (if observed)", format="%.2f"
                ),
            },
        )

    st.caption(
        "Confidence interval unavailable until a calibrated model is connected. "
        "The shaded section identifies the forecast period, not uncertainty."
    )

# 6. Bottom Two-Column Row: Main Forecast Drivers & What-If Simulator
left_col, right_col = st.columns([1.05, 1.0], gap="medium")

with left_col:
    with st.container(border=True):
        dl, dr = st.columns([3.5, 1.2], vertical_alignment="center")
        with dl:
            st.subheader("Main Forecast Drivers")
        with dr:
            st.caption("SHAP XAI" if top_drivers else "SHAP UNAVAILABLE")

        if top_drivers:
            st.caption("Feature contributions to the predicted next-month price (SHAP TreeExplainer)")
            drivers_df = pd.DataFrame(top_drivers)
            is_shap = True
        else:
            st.caption(
                "`shap.TreeExplainer` is not connected (`top_drivers` is empty). "
                "Showing real historical Pearson correlations between candidate features and "
                f"{selected_commodity} MoM % price changes as an analytical reference."
            )
            drivers_df = compute_historical_correlations(commodity_up_to_base)
            is_shap = False

        st.plotly_chart(
            render_drivers_chart(drivers_df, is_shap=is_shap),
            width="stretch",
            config={"displayModeBar": False},
        )

        b1, b2 = st.columns(2)
        with b1:
            st.metric(
                label=f"Current observation ({base_date.strftime('%b %Y')})",
                value=f"{current_price:.2f} EGP",
                border=True,
            )
        with b2:
            st.metric(
                label=f"Forecast output ({selected_forecast_month.strftime('%b %Y')})",
                value=f"{pred_price:.2f} EGP" if model_connected else "Pending model",
                border=True,
            )

        with st.expander("View driver values"):
            display_drivers = drivers_df.sort_values("Value", key=abs, ascending=False).copy()
            col_name = "SHAP Contribution (EGP)" if is_shap else "Pearson Correlation (r)"
            display_drivers = display_drivers.rename(columns={"Value": col_name})
            st.dataframe(
                display_drivers,
                hide_index=True,
                width="stretch",
                column_config={
                    col_name: st.column_config.NumberColumn(col_name, format="%+.3f")
                },
            )

with right_col:
    with st.container(border=True):
        wl, wr = st.columns([3.5, 1.2], vertical_alignment="center")
        with wl:
            st.subheader("What-If Simulator")
            st.caption("Adjust assumptions and compare with the baseline.")
        with wr:
            st.caption("LIVE SIMULATOR" if model_connected else "MODEL REQUIRED")

        if not model_connected:
            st.info(
                "Model inference is inactive until `models/best_forecast_model.joblib` is trained. "
                "You can adjust macro/weather inputs below to inspect shock deltas against the "
                f"{base_date.strftime('%b %Y')} baseline.",
                icon=":material/info:",
            )

        base_usd = round(float(base_row["usd_sell_end"]), 2)
        base_inf = round(float(base_row["headline_inflation_mom"]), 2)
        base_temp = round(float(base_row["nat_temp_anomaly"]), 2)

        def _reset_scenario_inputs() -> None:
            st.session_state["pf_usd_input"] = base_usd
            st.session_state["pf_inf_input"] = base_inf
            st.session_state["pf_temp_input"] = base_temp
            st.session_state["pf_scenario_ran"] = False

        in1, in2, in3 = st.columns(3)
        with in1:
            usd_val = st.number_input(
                "USD Exchange Rate (EGP/USD)",
                min_value=1.0,
                max_value=200.0,
                step=0.25,
                key="pf_usd_input",
            )
            st.caption(f"Baseline ({base_date.strftime('%b %Y')}): {base_usd:.2f} EGP")
        with in2:
            inf_val = st.number_input(
                "Inflation Rate (% MoM)",
                min_value=-20.0,
                max_value=50.0,
                step=0.1,
                key="pf_inf_input",
            )
            st.caption(f"Baseline: {base_inf:+.2f}% MoM")
        with in3:
            temp_val = st.number_input(
                "Temperature Anomaly (°C)",
                min_value=-10.0,
                max_value=10.0,
                step=0.1,
                key="pf_temp_input",
            )
            st.caption(f"Baseline anomaly: {base_temp:+.2f} °C")

        btn_run, btn_rst = st.columns([1.4, 1])
        with btn_run:
            if st.button("Run Scenario", type="primary", width="stretch"):
                st.session_state["pf_scenario_ran"] = True
        with btn_rst:
            st.button("Reset inputs", on_click=_reset_scenario_inputs, width="stretch")

        st.divider()
        st.markdown("**Scenario results**")

        if st.session_state.get("pf_scenario_ran"):
            usd_delta = usd_val - base_usd
            inf_delta = inf_val - base_inf
            temp_delta = temp_val - base_temp

            scenario_row = base_row_df.copy()
            scenario_row["usd_sell_end"] = usd_val
            scenario_row["headline_inflation_mom"] = inf_val
            scenario_row["nat_temp_anomaly"] = temp_val

            if model_connected:
                scen_out = predict_next_month(scenario_row, current_price)
                scen_price = float(scen_out["predicted_price"])
                scen_pct = float(scen_out["predicted_pct_change"])
                diff_baseline = scen_price - pred_price
                s1, s2, s3 = st.columns(3)
                s1.metric("Scenario Forecast Price", f"{scen_price:.2f} EGP", border=True)
                s2.metric("Scenario Price Change", f"{scen_pct:+.1f}%", border=True)
                s3.metric("Difference from Baseline", f"{diff_baseline:+.2f} EGP", border=True)
            else:
                s1, s2, s3 = st.columns(3)
                s1.metric("Scenario Forecast Price", "N/A", delta="Model not connected", delta_color="off", delta_arrow="off", border=True)
                s2.metric("Scenario Price Change", "N/A", delta="Model not connected", delta_color="off", delta_arrow="off", border=True)
                s3.metric("Difference from Baseline", "N/A", delta="Model not connected", delta_color="off", delta_arrow="off", border=True)
                st.caption(
                    f"Applied shock vs. baseline: USD {usd_delta:+.2f} EGP · "
                    f"Inflation {inf_delta:+.2f} pp · Temp anomaly {temp_delta:+.2f} °C"
                )
        else:
            s1, s2, s3 = st.columns(3)
            s1.metric("Scenario Forecast Price", "—", border=True)
            s2.metric("Scenario Price Change", "—", border=True)
            s3.metric("Difference from Baseline", "—", border=True)
            st.caption("Set assumptions, then select Run Scenario. Results update only when submitted.")

        st.caption("One-month horizon · All other features held constant · Form resets when commodity or forecast month changes.")

# 7. Methodology & integration expander
with st.expander("Input snapshot & forecast methodology — Available fields & integration notes"):
    st.markdown(
        f"""
        - **Master Dataset (`data/processed/master_food_dataset.csv`)**: Provides monthly historical prices (`price`), macroeconomic indicators (`usd_sell_end`, `usd_mom_pct`, `headline_inflation_mom`, `core_inflation_mom`, `cbe_deposit_rate`, `cbe_lending_rate`), and climate variables (`cairo_temp_mean`, `nat_temp_anomaly`, etc.) through **{latest_date.strftime('%B %Y')}**.
        - **Forecast Pipeline Integration**: This page calls `build_features()` from `src/features.py` and `load_model_artifacts()` / `predict_next_month()` from `src/predictor.py`.
        - **Current Integration Status**:
          - `models/best_forecast_model.joblib`: **Not found** in `models/` (currently only `.gitkeep` exists).
          - `src/predictor.py`: `predict_next_month()` returns placeholder values (`predicted_pct_change: 0.0`, `top_drivers: []`) until the trained XGBoost/LightGBM model and `shap.TreeExplainer` are connected.
          - `Main Forecast Drivers`: Automatically falls back to **real historical Pearson correlations** from `master_food_dataset.csv` while SHAP attributions (`top_drivers`) are unavailable.
        """
    )

st.caption("FoodWise Egypt · Turning market data into food security insights.")
