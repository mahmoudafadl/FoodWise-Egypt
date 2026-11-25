import streamlit as st
from utils import load_master_data

st.set_page_config(page_title="🚨 Alerts & Risk | FoodWise Egypt", layout="wide")
st.title("🚨 Alerts & Risk")
st.caption("Automated early-warning feed highlighting abnormal price spikes and high-risk commodities.")

df = load_master_data()
if df.empty:
    st.warning("Dataset not loaded. Please run `python scripts/build_dataset.py`.")
else:
    st.success(f"Loaded {len(df):,} records across {df['filter_id'].nunique()} commodities. Ready for visualization!")
    # TODO (Member 6): Implement interactive Plotly charts & KPI cards for this page
