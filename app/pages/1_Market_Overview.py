import streamlit as st
from utils import load_master_data

st.set_page_config(page_title="📊 Market Overview | FoodWise Egypt", layout="wide")
st.title("📊 Market Overview")
st.caption("KPI summary cards, top price movers, and overall market risk heatmap.")

df = load_master_data()
if df.empty:
    st.warning("Dataset not loaded. Please run `python scripts/build_dataset.py`.")
else:
    st.success(f"Loaded {len(df):,} records across {df['filter_id'].nunique()} commodities. Ready for visualization!")
    # TODO (Member 6): Implement interactive Plotly charts & KPI cards for this page
