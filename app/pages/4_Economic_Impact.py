import streamlit as st
from utils import load_master_data

st.set_page_config(page_title="💱 Economic Impact | FoodWise Egypt", layout="wide")
st.title("💱 Economic Impact")
st.caption("Correlation and lag-response analysis linking USD exchange rates, inflation, and CBE policy rates to food categories.")

df = load_master_data()
if df.empty:
    st.warning("Dataset not loaded. Please run `python scripts/build_dataset.py`.")
else:
    st.success(f"Loaded {len(df):,} records across {df['filter_id'].nunique()} commodities. Ready for visualization!")
    # TODO (Member 6): Implement interactive Plotly charts & KPI cards for this page
