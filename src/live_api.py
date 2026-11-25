"""Live external API connectors (Exchange Rate & Open-Meteo Forecast) with caching and fallback."""
import requests
import streamlit as st


@st.cache_data(ttl=3600, show_spinner=False)
def _fetch_live_usd_api() -> float:
    url = "https://open.er-api.com/v6/latest/USD"
    resp = requests.get(url, timeout=5)
    resp.raise_for_status()
    return float(resp.json()["rates"]["EGP"])


def get_current_usd(fallback_value: float) -> tuple[float, str]:
    """Fetch live USD/EGP rate; falls back to latest CSV value if API fails (outside cache)."""
    try:
        return _fetch_live_usd_api(), "Live API"
    except Exception:
        return float(fallback_value), "Historical Fallback (CSV)"


@st.cache_data(ttl=10800, show_spinner=False)
def _fetch_weather_forecast_api(lat: float, lon: float) -> dict:
    url = (
        f"https://api.open-meteo.com/v1/forecast?latitude={lat}&longitude={lon}"
        "&daily=temperature_2m_max,temperature_2m_min,precipitation_sum&forecast_days=14&timezone=Africa%2FCairo"
    )
    resp = requests.get(url, timeout=5)
    resp.raise_for_status()
    return resp.json()


def get_region_forecast(lat: float, lon: float, fallback_temp: float) -> tuple[float, str]:
    """Fetch 14-day forecasted mean max temperature with graceful fallback."""
    try:
        data = _fetch_weather_forecast_api(lat, lon)
        temps = data["daily"]["temperature_2m_max"]
        return round(sum(temps) / len(temps), 2), "Live Forecast API"
    except Exception:
        return float(fallback_temp), "Historical Fallback (CSV)"
