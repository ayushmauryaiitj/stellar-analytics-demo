import streamlit as st
import pickle
import numpy as np
import os

# -------------------------------
# PAGE CONFIG (MUST BE FIRST)
# -------------------------------
st.set_page_config(page_title="Stellar Analytics", layout="centered")

# -------------------------------
# SAFE MODEL LOADING
# -------------------------------
@st.cache_resource
def load_models():
    base_path = os.path.dirname(__file__)

    rf_path = os.path.join(base_path, "rf_classifier.pkl")
    gbr_path = os.path.join(base_path, "gbr_regressor.pkl")
    pre_path = os.path.join(base_path, "preprocessor.pkl")

    rf_clf = pickle.load(open(rf_path, "rb"))
    gbr_reg = pickle.load(open(gbr_path, "rb"))
    preprocessor = pickle.load(open(pre_path, "rb"))

    return rf_clf, gbr_reg, preprocessor

rf_clf, gbr_reg, preprocessor = load_models()

# -------------------------------
# UI
# -------------------------------

st.title("🪐 Stellar Analytics")
st.caption("F1 = 0.833 | RMSE = 1.55 R⊕")

st.subheader("Input Stellar Parameters")

col1, col2 = st.columns(2)

koi_period = col1.number_input("Orbital Period (days)", 0.1, 1000.0, 10.0)
koi_duration = col1.number_input("Transit Duration (hrs)", 0.1, 50.0, 5.0)
koi_depth = col1.number_input("Transit Depth (%)", 0.0, 5.0, 0.5)
koi_impact = col2.number_input("Impact Parameter", 0.0, 1.5, 0.8)
koi_model_snr = col2.number_input("Signal-to-Noise Ratio", 0.0, 10000.0, 1000.0)

koi_num_transits = st.number_input("Number of Transits", 1, 50, 3)

st.subheader("Host Star Properties")

st_teff = col1.number_input("Stellar Temperature (K)", 3000, 8000, 5500)
st_logg = col1.number_input("Stellar Gravity", 0.0, 5.0, 4.0)
st_met = col2.number_input("Metallicity", -2.5, 1.0, 0.0)
st_mass = col2.number_input("Stellar Mass", 0.1, 5.0, 1.0)
st_radius = st.number_input("Stellar Radius", 0.1, 5.0, 1.0)

# -------------------------------
# PREDICTION
# -------------------------------
if st.button("🚀 Predict"):

    input_data = np.array([[
        koi_period,
        koi_duration,
        koi_depth / 100,
        koi_impact,
        koi_model_snr,
        koi_num_transits,
        st_teff,
        st_logg,
        st_met,
        st_mass,
        st_radius
    ]])

    try:
        processed = preprocessor.transform(input_data)

        prob = rf_clf.predict_proba(processed)[0][1]
        pred_class = rf_clf.predict(processed)[0]

        st.subheader("Prediction Result")

        if pred_class == 1:
            st.success(f"🪐 Confirmed Candidate ({prob:.1%} confidence)")
            radius = gbr_reg.predict(processed)[0]
            st.info(f"Estimated Radius: {radius:.2f} Earth Radii")
        else:
            st.error(f"❌ False Positive ({prob:.1%} confidence)")

    except Exception as e:
        st.error("Prediction Failed ❌")
        st.exception(e)
