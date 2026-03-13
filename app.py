# v2.0 - Updated defaults for CONFIRMED predictions
import streamlit as st
import numpy as np
import joblib

# Load models and preprocessing
@st.cache_resource
def load_models():
    classification_model = joblib.load("classification_model.pkl")
    regression_model = joblib.load("regression_model.pkl")
    scaler = joblib.load("scaler.pkl")
    return classification_model, regression_model, scaler

clf_model, reg_model, scaler = load_models()

# Streamlit App UI
st.set_page_config(
    page_title="Stellar Analytics",
    page_icon="🪐",
    layout="centered"
)

st.title("🪐 Stellar Analytics")
st.markdown("F1 = 0.833 | RMSE = 1.55 R⊕")

st.markdown("### Input Stellar Parameters")

# DEFAULT VALUES OPTIMIZED FOR CONFIRMED PREDICTION
koi_period = st.number_input(
    "Orbital Period (days)",
    min_value=0.1,
    max_value=1000.0,
    value=365.0,  # CONFIRMED: Earth-like orbit
    step=0.1
)

koi_duration = st.number_input(
    "Transit Duration (hrs)",
    min_value=0.1,
    max_value=50.0,
    value=12.0,  # CONFIRMED: Longer duration
    step=0.1
)

koi_depth = st.number_input(
    "Transit Depth (%)",
    min_value=0.0,
    max_value=5.0,
    value=1.5,  # CONFIRMED: Stronger signal
    step=0.01
)

koi_impact = st.number_input(
    "Impact Parameter",
    min_value=0.0,
    max_value=1.5,
    value=0.2,  # CONFIRMED: Low impact
    step=0.01
)

koi_model_snr = st.number_input(
    "Signal-to-Noise Ratio",
    min_value=0.1,
    max_value=10000.0,
    value=5000.0,  # CONFIRMED: High SNR
    step=1.0
)

koi_num_transits = st.number_input(
    "Number of Transits",
    min_value=1,
    max_value=50,
    value=20,  # CONFIRMED: Many observations
    step=1
)

st.markdown("### Host Star Properties")

st_teff = st.number_input(
    "Stellar Temperature (K)",
    min_value=3000,
    max_value=8000,
    value=5500,
    step=50
)

st_logg = st.number_input(
    "Stellar Gravity (log g)",
    min_value=0.0,
    max_value=5.0,
    value=3.5,  # CONFIRMED: Giant star
    step=0.1
)

st_met = st.number_input(
    "Metallicity [Fe/H]",
    min_value=-2.5,
    max_value=1.0,
    value=0.0,
    step=0.05
)

st_mass = st.number_input(
    "Stellar Mass (Solar masses)",
    min_value=0.1,
    max_value=5.0,
    value=1.0,
    step=0.05
)

st_radius = st.number_input(
    "Stellar Radius (Solar radii)",
    min_value=0.1,
    max_value=5.0,
    value=1.0,
    step=0.05
)

# Feature vector construction
def build_feature_array():
    features = np.array([[
        koi_period,
        koi_duration,
        koi_depth,
        koi_impact,
        koi_model_snr,
        koi_num_transits,
        st_teff,
        st_logg,
        st_met,
        st_mass,
        st_radius
    ]], dtype=float)
    return features

# Prediction logic
st.markdown("### Run Prediction")

if st.button("🚀 Predict"):
    X = build_feature_array()
    X_scaled = scaler.transform(X)

    # 1. Classification
    class_pred = clf_model.predict(X_scaled)[0]
    class_proba = clf_model.predict_proba(X_scaled)[0]
    pred_idx = list(clf_model.classes_).index(class_pred)
    confidence = class_proba[pred_idx] * 100

    st.markdown("### Prediction Result")

    if class_pred == "CONFIRMED":
        st.success(f"✔ Confirmed Exoplanet (Confidence: {confidence:.1f}%)")

        # 2. Regression only if confirmed
        radius_pred = reg_model.predict(X_scaled)[0]

        st.markdown("### Planet Radius Prediction")
        st.info(f"🌍 Estimated Planet Radius: **{radius_pred:.2f}** Earth Radii")

    else:
        st.error(f"❌ False Positive (Confidence: {confidence:.1f}%)")
