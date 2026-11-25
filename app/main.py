import streamlit as st
from utils import load_master_data

st.set_page_config(
    page_title="FoodWise Egypt | Food Price Intelligence",
    page_icon="🌾",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.title("🌾 FoodWise Egypt — Food Price Intelligence & Early Warning System")
st.markdown(
    "End-to-end Machine Learning decision-support system monitoring **57 essential food commodities** "
    "across Egypt by integrating CAPMAS prices, CBE macroeconomic indicators, and regional climate data."
)

df = load_master_data()
if df.empty:
    st.warning("Processed dataset not found yet. Run `python scripts/build_dataset.py` first.")
else:
    latest_date = df["date"].max().strftime("%B %Y")
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Tracked Commodities", f"{df['filter_id'].nunique()}")
    c2.metric("Timeline Coverage", "138 Months (2015–2026)")
    c3.metric("Latest Data Point", latest_date)
    c4.metric("Latest USD/EGP (CBE)", f"{df['usd_sell_end'].iloc[-1]:.2f} EGP")

st.info("👈 Select an analytical module from the sidebar to explore historical trends, forecasts, macro/weather drivers, and risk alerts.")
