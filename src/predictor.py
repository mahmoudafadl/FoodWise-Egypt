"""Model inference, anomaly risk classification, and on-the-fly SHAP explainability."""
from pathlib import Path
import joblib
import pandas as pd

MODELS_DIR = Path("models")


def load_model_artifacts():
    """Load trained forecasting model and feature list from models/."""
    model_path = MODELS_DIR / "best_forecast_model.joblib"
    if not model_path.exists():
        return None
    return joblib.load(model_path)


def predict_next_month(row_features: pd.DataFrame, current_price: float) -> dict:
    """
    Predict next month % price change and reconstruct expected price:
    P_hat(t+1) = P(t) * (1 + delta_P_hat / 100)
    """
    # TODO (Members 3 & 4): Connect trained model + shap.TreeExplainer
    return {
        "predicted_pct_change": 0.0,
        "predicted_price": current_price,
        "risk_level": "Low",
        "top_drivers": [],
    }
