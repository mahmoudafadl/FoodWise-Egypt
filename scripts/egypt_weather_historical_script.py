import requests
import pandas as pd

locations = {
    "Greater_Cairo": {"lat": 30.0444, "lon": 31.2357},
    "West_Delta_Beheira": {"lat": 30.8481, "lon": 30.3436},
    "East_Delta_Sharqia": {"lat": 30.5877, "lon": 31.5020},
    "Upper_Egypt_Minya": {"lat": 28.1099, "lon": 30.7503}
}

#القاهرة الكبرى (Greater_Cairo): بتمثل مركز الاستهلاك الأكبر + حزام مزارع الدواجن والبيض.
#غرب الدلتا (West_Delta_Beheira): بتمثل الأراضي المستصلحة والنوبارية (بطاطس، طماطم، فاكهة، خضروات).
#شرق الدلتا (East_Delta_Sharqia): بتمثل الدلتا القديمة (قمح، أرز، ثروة داجنة).
#صعيد مصر (Upper_Egypt_Minya): بتمثل المناخ القاري الحار والعروات الزراعية الصيفية والشتوية المبكرة.

all_weather = []

for region, coords in locations.items():
    url = "https://archive-api.open-meteo.com/v1/archive"
    params = {
        "latitude": coords["lat"],
        "longitude": coords["lon"],
        "start_date": "2015-01-01",
        "end_date": "2026-06-30",
        "daily": [
            "temperature_2m_max",
            "temperature_2m_min",
            "temperature_2m_mean",
            "precipitation_sum",
            "relative_humidity_2m_mean"
        ],
        "timezone": "Africa/Cairo"
    }
    res = requests.get(url, params=params).json()
    df_reg = pd.DataFrame(res["daily"])
    df_reg["region"] = region
    all_weather.append(df_reg)

df_weather = pd.concat(all_weather, ignore_index=True)
df_weather.to_csv("egypt_weather_4regions_2015_2026.csv", index=False)
print("Done! Shape:", df_weather.shape)
