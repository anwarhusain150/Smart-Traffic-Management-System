"""
Smart Traffic Management – Streamlit dashboard
Run from the project folder:
    streamlit run app/app.py
"""

import sys
from datetime import date
from pathlib import Path

import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import streamlit as st

ROOT = Path(__file__).resolve().parents[1]
sys.path.append(str(ROOT / "src"))
import data_preprocessing as dp  # noqa: E402

img = ROOT / "Images"

st.set_page_config(page_title="Smart Traffic Predictor", page_icon="🚦", layout="wide")

COLORS = {"Low": "#2ca02c", "Medium": "#ff7f0e", "High": "#d62728"}
WEATHER = ["Clear", "Clouds", "Rain", "Drizzle", "Mist", "Haze", "Fog", "Snow", "Thunderstorm", "Other"]


@st.cache_resource
def load_models():
    """Load trained models once. If they don't exist yet, train them (takes ~1 minute)."""
    reg_path, clf_path = ROOT / "models" / "regressor.joblib", ROOT / "models" / "classifier.joblib"
    if not (reg_path.exists() and clf_path.exists()):
        from train_models import train_and_save
        with st.spinner("Training models for the first time (about 1 minute)..."):
            train_and_save()
    return joblib.load(reg_path), joblib.load(clf_path)


def build_input(day: date, hours, weather, temp_c, rain, snow, clouds, holiday) -> pd.DataFrame:
    """Create the same feature columns the models were trained on."""
    rows = pd.DataFrame({
        "date_time": [pd.Timestamp(day) + pd.Timedelta(hours=int(h)) for h in hours],
        "weather_main": weather, "temp_c": temp_c, "rain_1h": rain, "snow_1h": snow,
        "clouds_all": clouds, "is_holiday": int(holiday), "traffic_volume": 0,
    })
    return dp.add_features(rows)[dp.FEATURES]


def predict(reg, clf, X):
    volume = np.clip(reg.predict(X), 0, None)
    proba = clf.predict_proba(X)
    proba = pd.DataFrame(proba, columns=clf.classes_)[dp.CLASS_ORDER]
    return volume, clf.predict(X), proba


reg_model, clf_model = load_models()

# ---------------- Sidebar: user inputs ----------------
st.sidebar.header("🛣️ Enter conditions")
day = st.sidebar.date_input("Date", value=date.today())
hour = st.sidebar.slider("Hour of day", 0, 23, 8)
holiday = st.sidebar.checkbox("Public holiday?")
weather = st.sidebar.selectbox("Weather", WEATHER, index=1)
temp_c = st.sidebar.slider("Temperature (°C)", -30, 40, 15)
clouds = st.sidebar.slider("Cloud cover (%)", 0, 100, 40)
rain = st.sidebar.number_input("Rain (mm in the hour)", 0.0, 60.0, 0.0, step=0.5)
snow = st.sidebar.number_input("Snow (mm in the hour)", 0.0, 1.0, 0.0, step=0.05)

st.title("🚦 AI-Based Smart Traffic Management & Congestion Prediction")
st.caption("Trained on UCI Metro Interstate Traffic Volume data (I-94, Minneapolis–St Paul, 2012–2018). "
           "Module 1: Random Forest regression • Module 2: congestion classification.")

tab1, tab2, tab3 = st.tabs(["🔮 Prediction", "📊 Data insights", "🏆 Model results"])

# ---------------- Tab 1: prediction ----------------
with tab1:
    X_one = build_input(day, [hour], weather, temp_c, rain, snow, clouds, holiday)
    vol, cls, proba = predict(reg_model, clf_model, X_one)
    level = cls[0]

    c1, c2, c3 = st.columns(3)
    c1.metric("Predicted traffic volume", f"{vol[0]:,.0f} vehicles/hour")
    c2.markdown(f"**Congestion level**<br><span style='font-size:2rem;color:{COLORS[level]};"
                f"font-weight:700'>{level}</span>", unsafe_allow_html=True)
    c3.metric("Classifier confidence", f"{proba[level].iloc[0] * 100:.0f}%")

    advice = {"Low": "Roads are free – normal signal timing is fine.",
              "Medium": "Moderate traffic – keep an eye on main junctions.",
              "High": "Heavy congestion expected – extend green time on the main road, "
                      "suggest alternate routes and deploy traffic police."}
    st.info(f"**Suggestion:** {advice[level]}")

    st.subheader(f"24-hour forecast for {day.strftime('%A, %d %b %Y')}")
    X_day = build_input(day, range(24), weather, temp_c, rain, snow, clouds, holiday)
    vol_day, cls_day, _ = predict(reg_model, clf_model, X_day)
    fig, ax = plt.subplots(figsize=(12, 4))
    ax.bar(range(24), vol_day, color=[COLORS[c] for c in cls_day])
    ax.axvline(hour, color="black", ls="--", lw=1)
    ax.set_xticks(range(24))
    ax.set_xlabel("Hour of day"); ax.set_ylabel("Predicted vehicles/hour")
    ax.set_title("Bar colour = predicted congestion level (green Low, orange Medium, red High)")
    st.pyplot(fig)

    quiet = [h for h in range(6, 22) if cls_day[h] == "Low"]
    st.write("**Best hours to travel (6 AM–10 PM, Low congestion):** "
             + (", ".join(f"{h}:00" for h in quiet) if quiet else "none – try early morning or late night"))

# ---------------- Tab 2: insights ----------------
with tab2:
    st.image(str(img / "02_heatmap_day_hour.png"), caption="Average traffic by day of week and hour")
    col_a, col_b = st.columns(2)
    col_a.image(str(img / "02_hour_day_patterns.png"), caption="Weekday vs weekend pattern")
    col_b.image(str(img / "02_correlation.png"), caption="Correlation matrix")

# ---------------- Tab 3: model results ----------------
with tab3:
    st.subheader("Module 1 – Regression (test year Oct 2017 – Sep 2018)")
    st.dataframe(pd.read_csv(ROOT / "Reports" / "regression_results.csv"), width="stretch")
    st.subheader("Module 2 – Classification (test year)")
    st.dataframe(pd.read_csv(ROOT / "Reports" / "classification_results.csv"), width="stretch")
    st.image(str(img / "04_best_classifier_cm_roc.png"), caption="Best classifier – confusion matrix and ROC")
