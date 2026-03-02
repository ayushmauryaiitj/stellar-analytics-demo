import streamlit as st
import pickle
import numpy as np
import os
from pathlib import Path

# Get the directory where the script is located
BASE_DIR = Path(__file__).parent

@st.cache_resource
def load_models():
    try:
        # Try to load pickle files
        with open(BASE_DIR / "rf_classifier.pkl", "rb") as f:
            rf_clf = pickle.load(f)
        with open(BASE_DIR / "gbr_regressor.pkl", "rb") as f:
            gbr_reg = pickle.load(f)
        with open(BASE_DIR / "preprocessor.pkl", "rb") as f:
            preprocessor = pickle.load(f)
        return rf_clf, gbr_reg, preprocessor
    except Exception as e:
        st.error(f"Error loading models: {str(e)}")
        return None, None, None

st.title("🪐 Stellar Analytics")
st.markdown("**F1=0.833 | RMSE=1.55 R⊕ | TECHNEX '26 Live Demo**")

# Load models
rf_clf, gbr_reg, preprocessor = load_models()

if rf_clf is None:
    st.warning("⚠️ Models could not be loaded. Please check the repository files.")
    st.stop()

# UI
col1, col2 = st.columns(2)

with col1:
    period = col1.number_input("Period (days)", 0.1, 1000, 10.0)
    depth = col1.number_input("Depth (%)", 0.0, 5.0, 0.5)
    impact = col1.number_input("Impact", 0.0, 1.5, 0.8)

with col2:
    snr = col2.number_input("SNR", 0.0, 10000, 1000)

if st.button("🔍 PREDICT", type="primary"):
    try:
        # Create feature array
        features = np.array([[period, 1.0, depth / 100, impact, snr, 3.0, 5500, 4.0, 0.0, 1.0, 1.0]])
        
        # Transform features
        features_processed = preprocessor.transform(features)
        
        # Make predictions
        exoplanet_prob = rf_clf.predict_proba(features_processed)[0][1]
        radius = gbr_reg.predict(features_processed)[0]
        
        # Display results
        st.markdown("---")
        col1, col2 = st.columns(2)
        with col1:
            st.metric("🌍 Exoplanet Probability", f"{exoplanet_prob*100:.2f}%")
        with col2:
            st.metric("📏 Predicted Radius", f"{radius:.2f} R⊕")
        
        st.success("✅ Prediction complete!")
    except Exception as e:
        st.error(f"❌ Prediction failed: {str(e)}")

st.markdown("---")
st.info("📝 Built with Streamlit | Data: Exoplanet Hunting Dataset")
