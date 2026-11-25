from pathlib import Path
import numpy as np
import pandas as pd

RAW_DIR = Path("data/raw")
PROCESSED_DIR = Path("data/processed")
PROCESSED_DIR.mkdir(parents=True, exist_ok=True)


def clean_food_prices(filepath: Path, ym_grid: pd.DataFrame) -> pd.DataFrame:
    df_food = pd.read_csv(filepath)

    # 1. الاحتفاظ بالأسعار الفعلية بالجنيه المصري فقط
    df_prices = df_food[df_food["unit"] == "Egyptian Pounds"].copy()

    # 2. تصحيح الإشارة السالبة (الكمون -350.5) وتحويل الأصفار (2026-03) لقيم مفقودة
    df_prices["value"] = df_prices["value"].abs()
    df_prices.loc[df_prices["value"] == 0, "value"] = np.nan

    # 3. توحيد سلسلة البيض (2015-2016 كبير + 2017-2026 متوسط) في سلعة واحدة متصلة
    egg_mask = df_prices["filter_id"].isin([6985, 7040])
    df_prices.loc[egg_mask, "filter_id"] = 9999
    df_prices.loc[egg_mask, "filter_en"] = "Farm Eggs (Single)"
    df_prices.loc[egg_mask, "filter_ar"] = "بيض مزارع"

    # 4. توحيد تصنيفات المجموعات الغذائية
    category_map = {
        "Prices of Fresh Vegetables and Fruits in the Republic": ("Vegetables & Fruits", "خضر وفاكهة"),
        "Prices of grains and legumes in the governorates of the Republic": ("Grains & Legumes", "حبوب وبقول"),
        "Prices of Oils in the Republic": ("Oils & Fats", "زيوت ودهون"),
        "Prices of live and frozen poultry and eggs across the republic": ("Poultry & Eggs", "دواجن وبيض"),
        "Prices of fresh local and imported meats in the governorates of the Republic": ("Red Meats", "لحوم حمراء"),
        "Prices of Fresh and Frozen Fish in the Republic": ("Fish & Seafood", "أسماك"),
        "Milk and dairy products prices in urban areas of the republic": ("Milk & Dairy", "ألبان ومنتجاتها"),
        "Grocery and spice prices in the governorates of the Republic": ("Groceries & Spices", "بقالة وعطارة"),
    }
    df_prices["category_en"] = df_prices["indicator_name"].map(
        lambda x: category_map.get(x, ("Other", "أخرى"))[0]
    )
    df_prices["category_ar"] = df_prices["indicator_name"].map(
        lambda x: category_map.get(x, ("Other", "أخرى"))[1]
    )

    latest_cat = (
        df_prices.sort_values(["year", "month"])
        .groupby("filter_id")[["category_en", "category_ar"]]
        .last()
        .reset_index()
    )

    # 5. ربط السلاسل الزمنية عبر filter_id وإزالة التكرارات
    df_stitched = (
        df_prices.groupby(["filter_id", "filter_en", "filter_ar", "year", "month"], as_index=False)
        .agg(price=("value", "mean"))
        .merge(latest_cat, on="filter_id", how="left")
    )

    # 6. تحديد النطاق الزمني (يناير 2015 - يونيو 2026) واختيار السلع المستمرة (>= 135 شهر)
    df_2015 = df_stitched[df_stitched["year"] >= 2015].copy()
    valid_counts = df_2015.groupby("filter_id")["price"].count()
    continuous_ids = valid_counts[valid_counts >= 135].index.tolist()
    df_cont = df_2015[df_2015["filter_id"].isin(continuous_ids)].copy()

    # 7. بناء شبكة زمنية كاملة (57 سلعة × 138 شهر) وتعويض القيم المفقودة النادرة خطياً
    items_meta = df_cont[
        ["filter_id", "filter_en", "filter_ar", "category_en", "category_ar"]
    ].drop_duplicates()
    full_grid = items_meta.merge(ym_grid, how="cross")

    df_clean = full_grid.merge(
        df_cont[["filter_id", "year", "month", "price"]],
        on=["filter_id", "year", "month"],
        how="left",
    ).sort_values(["filter_id", "year", "month"])

    df_clean["price"] = (
        df_clean.groupby("filter_id")["price"]
        .transform(lambda s: s.interpolate(method="linear").bfill().ffill())
        .round(2)
    )
    return df_clean


def clean_exchange_rates(filepath: Path, ym_grid: pd.DataFrame) -> pd.DataFrame:
    df_fx = pd.read_excel(filepath, sheet_name="Exchange Rates", header=1)
    df_fx["Date"] = pd.to_datetime(df_fx["Date"])
    df_fx["Buy"] = pd.to_numeric(df_fx["Buy"], errors="coerce")
    df_fx["Sell"] = pd.to_numeric(df_fx["Sell"], errors="coerce")
    df_fx = df_fx.sort_values("Date")
    df_fx["year"] = df_fx["Date"].dt.year
    df_fx["month"] = df_fx["Date"].dt.month

    df_agg = (
        df_fx.groupby(["year", "month"], as_index=False)
        .agg(
            usd_buy_mean=("Buy", "mean"),
            usd_sell_mean=("Sell", "mean"),
            usd_sell_end=("Sell", "last"),
            usd_volatility=("Sell", "std"),
        )
    )

    df_monthly = (
        ym_grid[["year", "month"]]
        .merge(df_agg, on=["year", "month"], how="left")
        .sort_values(["year", "month"])
    )
    df_monthly[["usd_buy_mean", "usd_sell_mean", "usd_sell_end"]] = (
        df_monthly[["usd_buy_mean", "usd_sell_mean", "usd_sell_end"]].ffill().bfill()
    )
    df_monthly["usd_volatility"] = df_monthly["usd_volatility"].fillna(0)
    df_monthly["usd_mom_pct"] = (df_monthly["usd_sell_end"].pct_change() * 100).fillna(0)
    return df_monthly.round(4)


def clean_inflation(filepath: Path) -> pd.DataFrame:
    df_inf = pd.read_excel(filepath, sheet_name="Inflation Rates", header=1)
    df_inf = df_inf.dropna(subset=["Headline (m/m)"]).copy()
    df_inf["date_parsed"] = pd.to_datetime(
        df_inf["Date"].astype(str).str.strip(), format="%b %Y", errors="coerce"
    )
    df_inf = df_inf.dropna(subset=["date_parsed"]).copy()
    df_inf["year"] = df_inf["date_parsed"].dt.year
    df_inf["month"] = df_inf["date_parsed"].dt.month

    inf_cols = {
        "Headline (m/m)": "headline_inflation_mom",
        "Core (m/m)": "core_inflation_mom",
        "Regulated Items (m/m)": "regulated_inflation_mom",
        "Fruits and Vegetables (m/m)": "veg_fruit_inflation_mom",
    }
    for raw_col, clean_col in inf_cols.items():
        df_inf[clean_col] = pd.to_numeric(
            df_inf[raw_col].astype(str).str.replace("%", "", regex=False).str.strip(),
            errors="coerce",
        )

    df_clean = df_inf[["year", "month"] + list(inf_cols.values())].sort_values(["year", "month"])
    return df_clean[df_clean["year"] >= 2015].round(3)


def clean_interest_rates(filepath: Path, ym_grid: pd.DataFrame) -> pd.DataFrame:
    df_rate = pd.read_excel(
        filepath, sheet_name="Overnight deposit and lending r", header=1
    )
    df_rate["Date"] = pd.to_datetime(df_rate["Date"].astype(str).str.strip(), errors="coerce")
    df_rate = df_rate.dropna(subset=["Date"]).sort_values("Date")

    for col in ["Overnight Deposit Rate", "Overnight Lending Rate"]:
        df_rate[col] = pd.to_numeric(
            df_rate[col].astype(str).str.replace("%", "", regex=False).str.strip(),
            errors="coerce",
        )

    df_rate["year"] = df_rate["Date"].dt.year
    df_rate["month"] = df_rate["Date"].dt.month
    df_rate_m = (
        df_rate.groupby(["year", "month"], as_index=False)
        .last()[["year", "month", "Overnight Deposit Rate", "Overnight Lending Rate"]]
        .rename(
            columns={
                "Overnight Deposit Rate": "cbe_deposit_rate",
                "Overnight Lending Rate": "cbe_lending_rate",
            }
        )
    )

    df_full = (
        ym_grid[["year", "month"]]
        .merge(df_rate_m, on=["year", "month"], how="left")
        .sort_values(["year", "month"])
    )
    df_full[["cbe_deposit_rate", "cbe_lending_rate"]] = (
        df_full[["cbe_deposit_rate", "cbe_lending_rate"]].ffill().bfill()
    )
    return df_full


def clean_weather(filepath: Path) -> tuple[pd.DataFrame, pd.DataFrame]:
    df_w = pd.read_csv(filepath)
    df_w["time"] = pd.to_datetime(df_w["time"])
    df_w["year"] = df_w["time"].dt.year
    df_w["month"] = df_w["time"].dt.month

    df_w["is_heatwave_day"] = (df_w["temperature_2m_max"] >= 35.0).astype(int)
    df_w["is_cold_day"] = (df_w["temperature_2m_min"] <= 7.0).astype(int)

    w_regions_monthly = (
        df_w.groupby(["year", "month", "region"], as_index=False)
        .agg(
            temp_mean=("temperature_2m_mean", "mean"),
            temp_max=("temperature_2m_max", "max"),
            temp_min=("temperature_2m_min", "min"),
            precip_sum=("precipitation_sum", "sum"),
            humidity_mean=("relative_humidity_2m_mean", "mean"),
            heatwave_days=("is_heatwave_day", "sum"),
            cold_days=("is_cold_day", "sum"),
        )
        .round(2)
    )

    region_map = {
        "Greater_Cairo": "cairo",
        "West_Delta_Beheira": "beheira",
        "East_Delta_Sharqia": "sharqia",
        "Upper_Egypt_Minya": "minya",
    }
    w_regions_monthly["reg_short"] = w_regions_monthly["region"].map(region_map)

    w_wide = w_regions_monthly.pivot(
        index=["year", "month"],
        columns="reg_short",
        values=[
            "temp_mean",
            "temp_max",
            "temp_min",
            "precip_sum",
            "humidity_mean",
            "heatwave_days",
            "cold_days",
        ],
    )
    w_wide.columns = [f"{reg}_{feat}" for feat, reg in w_wide.columns]
    w_wide = w_wide.reset_index().sort_values(["year", "month"])

    w_wide["nat_agro_temp_mean"] = w_wide[
        ["beheira_temp_mean", "sharqia_temp_mean", "minya_temp_mean"]
    ].mean(axis=1)
    w_wide["nat_heatwave_days_avg"] = w_wide[
        [
            "cairo_heatwave_days",
            "beheira_heatwave_days",
            "sharqia_heatwave_days",
            "minya_heatwave_days",
        ]
    ].mean(axis=1)
    w_wide["nat_cold_days_avg"] = w_wide[
        ["beheira_cold_days", "sharqia_cold_days", "minya_cold_days"]
    ].mean(axis=1)
    w_wide["nat_precip_total"] = w_wide[
        [
            "cairo_precip_sum",
            "beheira_precip_sum",
            "sharqia_precip_sum",
            "minya_precip_sum",
        ]
    ].sum(axis=1)

    monthly_norm = w_wide.groupby("month")["nat_agro_temp_mean"].transform("mean")
    w_wide["nat_temp_anomaly"] = w_wide["nat_agro_temp_mean"] - monthly_norm

    weather_ml_cols = [
        "year",
        "month",
        "cairo_temp_mean",
        "nat_agro_temp_mean",
        "nat_heatwave_days_avg",
        "nat_cold_days_avg",
        "nat_precip_total",
        "nat_temp_anomaly",
    ]
    w_ml = w_wide[weather_ml_cols].round(2)
    return w_ml, w_regions_monthly.drop(columns=["reg_short"])


def run_pipeline():
    ym_grid = pd.DataFrame({"date": pd.date_range("2015-01-01", "2026-06-01", freq="MS")})
    ym_grid["year"] = ym_grid["date"].dt.year
    ym_grid["month"] = ym_grid["date"].dt.month

    df_food = clean_food_prices(RAW_DIR / "Food_Price.csv", ym_grid)
    df_fx = clean_exchange_rates(RAW_DIR / "Exchange Rates Historical.xlsx", ym_grid)
    df_inf = clean_inflation(RAW_DIR / "Inflations Historical.xlsx")
    df_rate = clean_interest_rates(RAW_DIR / "Overnight deposit and lending rate.xlsx", ym_grid)
    df_w_ml, df_w_regions = clean_weather(RAW_DIR / "egypt_weather_4regions_2015_2026.csv")

    df_master = (
        df_food.merge(df_fx, on=["year", "month"], how="left")
        .merge(df_inf, on=["year", "month"], how="left")
        .merge(df_rate, on=["year", "month"], how="left")
        .merge(df_w_ml, on=["year", "month"], how="left")
        .sort_values(["filter_id", "date"])
        .reset_index(drop=True)
    )

    master_path = PROCESSED_DIR / "master_food_dataset.csv"
    weather_path = PROCESSED_DIR / "weather_regions_monthly.csv"

    df_master.to_csv(master_path, index=False, encoding="utf-8-sig")
    df_w_regions.to_csv(weather_path, index=False, encoding="utf-8-sig")

    print("✅ Pipeline completed successfully!")
    print(f"   - Master Dataset: {master_path} -> Shape: {df_master.shape} | Nulls: {df_master.isnull().sum().sum()}")
    print(f"   - Regions Weather: {weather_path} -> Shape: {df_w_regions.shape} | Nulls: {df_w_regions.isnull().sum().sum()}")
    print(f"   - Commodities: {df_master['filter_id'].nunique()} | Date Range: {df_master['date'].min().date()} to {df_master['date'].max().date()}")


if __name__ == "__main__":
    run_pipeline()
