"""Shared data loaders and UI formatting utilities for Streamlit pages."""
from pathlib import Path
import pandas as pd
import streamlit as st

PROCESSED_DIR = Path("data/processed")


@st.cache_data(show_spinner=False)
def load_master_data() -> pd.DataFrame:
    path = PROCESSED_DIR / "master_food_dataset.csv"
    if not path.exists():
        return pd.DataFrame()
    df = pd.read_csv(path)
    df["date"] = pd.to_datetime(df["date"])
    return df


@st.cache_data(show_spinner=False)
def load_weather_regions() -> pd.DataFrame:
    path = PROCESSED_DIR / "weather_regions_monthly.csv"
    if not path.exists():
        return pd.DataFrame()
    return pd.read_csv(path)
