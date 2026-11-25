import streamlit as st
from utils import load_master_data

st.set_page_config(page_title="📈 Price Explorer | FoodWise Egypt", layout="wide")
st.title("📈 Price Explorer")
st.caption("Interactive historical price trajectories (2015–2026) with major economic event annotations.")

df = load_master_data()
if df.empty:
    st.warning("Dataset not loaded. Please run `python scripts/build_dataset.py`.")
else:
    st.success(f"Loaded {len(df):,} records across {df['filter_id'].nunique()} commodities. Ready for visualization!")
    # TODO (Member 6): Implement interactive Plotly charts & KPI cards for this page
