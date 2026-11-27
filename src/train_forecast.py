from pathlib import Path

import pandas as pd
import joblib
from xgboost import XGBRegressor

from features import build_features


# Project paths
ROOT = Path(__file__).resolve().parents[1]

DATA_PATH = ROOT / "data" / "processed" / "master_food_dataset.csv"
MODELS_DIR = ROOT / "models"


def main():

    # 1. Load dataset
    print("Loading dataset...")

    df = pd.read_csv(DATA_PATH, parse_dates=["date"])

    print("Original shape:", df.shape)

    # 2. Build features
    print("Building features...")

    data = build_features(df)

    print("Features shape:", data.shape)

    # 3. Prepare training data
    data = data.dropna(subset=["target_mom_pct"])

    excluded_columns = [
        "filter_en",
        "filter_ar",
        "category_ar",
        "date",
        "month",
        "target_mom_pct"
    ]

    X = data.drop(columns=excluded_columns)
    y = data["target_mom_pct"]

    # Convert categorical columns
    X["filter_id"] = X["filter_id"].astype("category")
    X["category_en"] = X["category_en"].astype("category")

    print("Training rows:", len(X))
    print("Number of features:", X.shape[1])

    # 4. Create final XGBoost model
    model = XGBRegressor(
        learning_rate=0.02,
        max_depth=4,
        n_estimators=500,
        min_child_weight=3,
        subsample=0.8,
        colsample_bytree=0.8,
        tree_method="hist",
        enable_categorical=True,
        objective="reg:squarederror",
        eval_metric="mae",
        random_state=42,
        n_jobs=-1
    )

    # 5. Train model
    print("Training final model...")

    model.fit(X, y)

    # 6. Save model files
    MODELS_DIR.mkdir(parents=True, exist_ok=True)

    joblib.dump(
        model,
        MODELS_DIR / "best_forecast_model.joblib"
    )

    joblib.dump(
        list(X.columns),
        MODELS_DIR / "feature_columns.joblib"
    )

    model.save_model(
        MODELS_DIR / "best_forecast_model.json"
    )

    print("Training completed successfully!")
    print("Model files saved in:", MODELS_DIR)


if __name__ == "__main__":
    main()

