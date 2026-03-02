import streamlit as st
import pickle
import numpy as np

@st.cache_resource
def load_models():
    rf_clf = pickle.load(open("rf_classifier.pkl", "rb"))
    gbr_reg = pickle.load(open("gbr_regressor.pkl", "rb"))
    preprocessor = pickle.load(open("preprocessor.pkl", "rb"))
    return rf_clf, gbr_reg, preprocessor

st.title("🪐 Stellar Analytics")
st.markdown("**F1=0.833 | RMSE=1.55 R⊕ | TECHNEX '26 Live Demo**")

rf_clf, gbr_reg, preprocessor = load_models()

col1, col2 = st.columns(2)
period = col1.number_input("Period (days)", 0.1, 1000, 10.0)
depth = col1.number_input("Depth (%)", 0.0, 5.0, 0.5)
impact = col2.number_input("Impact", 0.0, 1.5, 0.8)
snr = col2.number_input("SNR", 0.0, 10000, 1000)

if st.button("🚀 PREDICT", type="primary"):
    features = np.array([[period, 1.0, depth/100, impact, snr, 3.0, 5500, 4.0, 0.0, 1.0, 1.0]])
    features_proc = preprocessor.transform(features)
    
    prob = rf_clf.predict_proba(features_proc)[0][1]
    is_planet = "🪐 CONFIRMED" if rf_clf.predict(features_proc)[0] else "❌ FALSE POSITIVE"
    
    st.metric("Classification", is_planet, f"{prob:.1%}")
    if "CONFIRMED" in is_planet:
        radius = gbr_reg.predict(features_proc)[0]
        st.metric("Radius", f"{radius:.1f} R⊕")
    
    st.balloons()
