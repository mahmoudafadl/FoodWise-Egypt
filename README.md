# FoodWise Egypt — Food Price Intelligence & Early Warning System

**FoodWise Egypt** is an end-to-end Data Science and Machine Learning decision-support system designed to monitor, analyze, and forecast food commodity price movements in Egypt. By integrating official historical food prices with macroeconomic shocks, multi-region agricultural climate data, and live API feeds, the system provides next-month price forecasts, risk classifications, anomaly alerts, and explainable AI (XAI) drivers.

---

## Problem Statement
Food price volatility in Egypt is driven by a complex interaction between currency fluctuations, inflation dynamics, seasonal agricultural cycles, religious occasions (Ramadan & Eids), and extreme weather events. Traditional single-variable forecasting fails to capture these multi-dimensional shocks. **FoodWise Egypt** bridges this gap by transforming historical and live data into actionable early-warning insights for consumers, traders, and decision-makers.

---

## System Architecture & Pipeline

```text
[Raw Data Sources: 2015–2026]
 ├── CAPMAS Food Prices (Monthly)
 ├── CBE Exchange Rates, Inflation & Policy Rates
 └── Open-Meteo Historical Weather (4 Strategic Regions)
          │
          ▼
[Data Engineering & Harmonization (`src/data_pipeline.py`)]
 ├── Unified Monthly Granularity (138 Months: Jan 2015 – Jun 2026)
 ├── ID-based Commodity Stitching & Anomaly Sanitization
 └── Multi-Region Climate Aggregation (Heatwaves, Cold Stress, Anomalies)
          │
          ▼
[Feature Engineering (`src/features.py`)]
 ├── Autoregressive Lags (1m, 3m, 12m) & Rolling Volatility
 ├── Macroeconomic Pass-Through Lags (USD MoM %, CPI, Corridor Rate)
 └── Cyclical Month Encoding (Sin/Cos) & Hijri Calendar Flags (Ramadan/Eids)
          │
          ▼
[ML Modeling & Explainability (`src/train.py`)]
 ├── Forecasting: XGBoost / LightGBM predicting MoM % Price Change (TimeSeriesSplit)
 ├── Risk & Alerts: Isolation Forest + Rolling Z-Score Anomaly Detection
 └── Explainability: SHAP Values decomposing each forecast into key drivers
          │
          ▼
[Interactive Web Application (`app/` — Streamlit + Plotly)]
 └── 6 Analytical Modules + Live Forecast APIs + What-If Macro Simulator
```

---

## Data Sources & Attribution
All datasets are aligned on a unified monthly timeline (**January 2015 – June 2026**):

1. **Food Commodity Prices:** Central Agency for Public Mobilization and Statistics (**CAPMAS**), Egypt — compiled via [Hugging Face Dataset Repository](https://huggingface.co/datasets/electricsheepafrica/africa-egypt-capmas-average-consumer-prices-for-key-food-items-577207dc).
2. **Macroeconomic Indicators:** Central Bank of Egypt (**CBE**) Official Historical Statistics:
   * Daily USD/EGP Buy & Sell Exchange Rates (aggregated to monthly mean, close, MoM %, and volatility).
   * Monthly Inflation Rates (Headline CPI, Core CPI, Regulated Items, and Fruits & Vegetables).
   * Monetary Policy Committee (MPC) Overnight Deposit & Lending Rates.
3. **Climate & Agricultural Weather Data:** [Open-Meteo Historical Archive & Forecast APIs](https://open-meteo.com/) covering 4 strategic Egyptian regions:
   * **Greater Cairo** (Consumption hub & poultry production belt)
   * **West Delta — Beheira** (Reclaimed agricultural lands & horticulture)
   * **East Delta — Sharqia** (Staple grains & poultry farming)
   * **Upper Egypt — Minya** (Continental climate & early harvest cycles)

---

## Modeling Methodology
* **Target Formulation (Inflation-Proof Forecasting):** Tree-based models cannot extrapolate beyond historical price ceilings. Instead of predicting raw prices directly, models are trained to predict **Month-over-Month Percentage Price Change ($\Delta P\%$)**, then reconstructed via:
  $$\hat{P}_{t+1} = P_t \times (1 + \widehat{\Delta P}\%)$$
* **Strict Leakage Prevention:** Evaluated using chronological **Walk-Forward Time-Series Cross-Validation** (`TimeSeriesSplit`) with zero random shuffling.
* **Explainable AI (SHAP):** Every forecast outputs human-readable drivers (e.g., *Seasonality*, *USD Exchange Rate Momentum*, *Heatwave Stress*, or *Historical Price Trend*).
* **Early Warning & Risk Scoring:** Combines **Isolation Forest** and rolling volatility thresholds to classify commodities into **Low**, **Medium**, or **High** price-surge risk.

---

## Application Modules (6 Pages)
1. **Market Overview:** KPI summary cards, top price movers, and overall market risk heatmap.
2. **Price Explorer:** Interactive historical price trajectories (2015–2026) with major economic event annotations.
3. **Price Forecast:** Next-month price & % change predictions, direction indicator (Up/Down/Stable), SHAP driver breakdown, and an interactive **What-If Simulator** (adjust USD rate, inflation, or temperature shocks).
4. **Economic Impact:** Correlation and lag-response analysis linking USD exchange rates, inflation, and interest rates to food categories.
5. **Weather Impact:** Regional climate comparison across Cairo, Beheira, Sharqia, and Minya and its effect on weather-sensitive crops and poultry.
6. **Alerts & Risk:** Automated early-warning feed highlighting abnormal price spikes and high-risk commodities.

---

## Repository Structure
```text
├── app/                    # Streamlit multi-page web application
├── data/
│   ├── raw/                # Original untouched datasets
│   └── processed/          # Cleaned & merged master datasets
├── models/                 # Serialized ML models & metadata
├── notebooks/              # Step-by-step EDA, feature engineering & modeling
├── scripts/                # Data collection & API fetching scripts
├── src/                    # Modular Python scripts (pipeline, features, train, API)
├── requirements.txt
└── README.md
```

---

## Quickstart

```bash
# 1. Clone the repository
git clone [https://github.com/mahmoudafadl/FoodWise-Egypt.git](https://github.com/mahmoudafadl/FoodWise-Egypt.git)
cd FoodWise-Egypt

# 2. Create and activate virtual environment
python -m venv venv
source venv/bin/activate        # On Windows: venv\Scripts\activate

# 3. Install dependencies
pip install -r requirements.txt

# 4. Run the Streamlit application
streamlit run app/main.py
```

---

## Project Team (DEPI Data Science Track)
* **Mahmoud Abbas**
* **Ahmed Abdelhamed**
* **Sara Elbably**
* **Basmala Sobhy**
* **Mostafa Ahmed**
* **Omar Afifi**
