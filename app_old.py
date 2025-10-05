import streamlit as st
import pandas as pd
import numpy as np
import pickle
import joblib
import os

# Page config
st.set_page_config(
    page_title="🚀 NASA Exoplanet Classification",
    page_icon="🚀",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for NASA theme
st.markdown("""
<style>
    .main {
        background: linear-gradient(135deg, #0B1426 0%, #1a2332 100%);
        color: white;
    }
    .stSidebar {
        background: linear-gradient(180deg, #1a2332 0%, #0B1426 100%);
    }
    .prediction-box {
        background: linear-gradient(45deg, #1e3a5f, #2d5a87);
        padding: 20px;
        border-radius: 10px;
        border: 2px solid #4CAF50;
        margin: 10px 0;
    }
    .confidence-box {
        background: rgba(30, 58, 95, 0.3);
        padding: 15px;
        border-radius: 8px;
        margin: 10px 0;
    }
</style>
""", unsafe_allow_html=True)

@st.cache_resource
def load_model():
    try:
        return joblib.load("best_lgbm_model_k2pandc.pkl")
    except Exception as e:
        st.error(f"Error loading model: {str(e)}")
        return None

model = load_model()

# Main title
st.markdown("# 🚀 Exoplanet Classification App")
st.markdown("*This app predicts the disposition of a Kepler Object of Interest (KOI) using a robustly trained LightGBM model. Enter the object's properties in the sidebar to get a prediction.*")

# Sidebar inputs
st.sidebar.markdown("## Input Features")

orbital_period = st.sidebar.number_input(
    "Orbital Period (days)", 
    min_value=0.1, max_value=1000.0, value=9.33, step=0.01
)

transit_depth = st.sidebar.number_input(
    "Transit Depth (ppm)", 
    min_value=1.0, max_value=10000.0, value=350.1, step=0.1
)

transit_duration = st.sidebar.number_input(
    "Transit Duration (hours)", 
    min_value=0.1, max_value=50.0, value=2.86, step=0.01
)

planetary_radius = st.sidebar.number_input(
    "Planetary Radius (Earth radii)", 
    min_value=0.1, max_value=50.0, value=2.30, step=0.01
)

insolation_flux = st.sidebar.number_input(
    "Insolation Flux (Earth flux)", 
    min_value=0.1, max_value=10000.0, value=93.5, step=0.1
)

equilibrium_temp = st.sidebar.number_input(
    "Equilibrium Temperature (K)", 
    min_value=100.0, max_value=3000.0, value=765.23, step=0.01
)

impact_param = st.sidebar.number_input(
    "Impact Parameter", 
    min_value=0.0, max_value=2.0, value=0.71, step=0.01
)

# Predict button
if st.sidebar.button("🔮 Predict Disposition", type="primary"):
    if model is None:
        st.error("Model not loaded properly!")
        st.stop()
    
    # Determine expected feature names from model/pipeline
    if hasattr(model, 'feature_names_in_'):
        feature_names = list(model.feature_names_in_)
    else:
        # Try to infer from pipeline steps (if any)
        try:
            last_estimator = getattr(model, 'steps', [[None, None]])[-1][1]
            feature_names = list(getattr(last_estimator, 'feature_names_in_', []))
        except Exception:
            feature_names = []

    # If we couldn't detect features, fall back to common NASA feature set
    if not feature_names:
        feature_names = [
            'pl_orbper', 'pl_rade', 'pl_insol', 'pl_eqt', 'st_teff', 'st_rad', 'st_mass', 'st_logg',
            'koi_period', 'koi_prad', 'koi_insol', 'koi_teq', 'koi_steff', 'koi_srad', 'koi_slogg',
            'koi_depth', 'koi_duration', 'koi_impact'
        ]

    # Build input row initialized with zeros (RandomForest doesn't handle NaN)
    input_row = {name: 0.0 for name in feature_names}

    # Set realistic defaults for common astrophysical features
    realistic_defaults = {
        'st_teff': 5500.0,
        'st_rad': 1.0,
        'st_mass': 1.0,
        'st_logg': 4.5,
        'sy_snum': 1.0,
        'sy_pnum': 1.0,
        'disc_year': 2020.0,
        'default_flag': 1.0,
        'pl_controv_flag': 0.0,
        'ttv_flag': 0.0
    }
    for k, v in realistic_defaults.items():
        if k in input_row:
            input_row[k] = v

    # Helper to set first existing alias
    def set_alias(value, aliases):
        for a in aliases:
            if a in input_row:
                input_row[a] = value
                return True
        return False

    # Map sidebar inputs to any matching feature aliases
    set_alias(orbital_period, ['pl_orbper', 'koi_period'])
    set_alias(planetary_radius, ['pl_rade', 'koi_prad'])
    set_alias(insolation_flux, ['pl_insol', 'koi_insol'])
    set_alias(equilibrium_temp, ['pl_eqt', 'koi_teq'])
    set_alias(transit_depth, ['koi_depth'])
    set_alias(transit_duration, ['koi_duration'])
    set_alias(impact_param, ['koi_impact'])

    # Create DataFrame with correct column order
    input_data = pd.DataFrame([input_row], columns=feature_names)

    # Make prediction
    prediction = model.predict(input_data)[0]
    probabilities = None
    if hasattr(model, 'predict_proba'):
        try:
            probabilities = model.predict_proba(input_data)[0]
        except Exception:
            probabilities = None
    
    # Display input features table
    st.markdown("## User Input Features:")
    
    feature_names = [
        "koi_period", "koi_time0bk", "koi_impact", "koi_duration", 
        "koi_depth", "koi_prad", "koi_teq", "koi_insol", 
        "koi_sma", "koi_ror", "koi_steff", "koi_slogg", "koi_srad"
    ]
    
    # Create feature display (simplified for demo)
    col1, col2, col3, col4, col5, col6, col7 = st.columns(7)
    with col1:
        st.metric("Period", f"{orbital_period:.2f}")
    with col2:
        st.metric("Depth", f"{transit_depth:.1f}")
    with col3:
        st.metric("Duration", f"{transit_duration:.2f}")
    with col4:
        st.metric("Radius", f"{planetary_radius:.2f}")
    with col5:
        st.metric("Flux", f"{insolation_flux:.1f}")
    with col6:
        st.metric("Temp", f"{equilibrium_temp:.0f}")
    with col7:
        st.metric("Impact", f"{impact_param:.2f}")
    
    # Prediction result
    st.markdown("## 🛸 Prediction Result")
    
    # Derive labels dynamically from model classes if available
    if hasattr(model, 'classes_'):
        labels = [str(c) for c in model.classes_]
    else:
        labels = ["CANDIDATE", "CONFIRMED", "FALSE POSITIVE"]

    # Safe color palette sized to number of classes
    base_palette = ["#4CAF50", "#FFA500", "#F44336", "#2196F3", "#9C27B0", "#009688"]
    colors = [base_palette[i % len(base_palette)] for i in range(len(labels))]

    # Map numeric prediction to label if needed
    try:
        predicted_label = labels[int(prediction)]
        predicted_color = colors[int(prediction)]
    except Exception:
        predicted_label = str(prediction)
        predicted_color = "#4CAF50"
    
    st.markdown(f"""
    <div class="prediction-box">
        <h3 style="color: {predicted_color};">The model predicts: {predicted_label}</h3>
    </div>
    """, unsafe_allow_html=True)
    
    # Prediction confidence
    st.markdown("## Prediction Confidence")
    
    if probabilities is not None:
        confidence_df = pd.DataFrame({
            'Classification': labels,
            'Confidence': [f"{float(p)*100:.2f}%" for p in probabilities],
            'Probability': probabilities
        })
        # Create confidence chart
        st.bar_chart(confidence_df.set_index('Classification')['Probability'])

        # Confidence table
        st.markdown("""
        <div class="confidence-box">
        """, unsafe_allow_html=True)
        st.dataframe(confidence_df[['Classification', 'Confidence']], hide_index=True, use_container_width=True)
        st.markdown("</div>", unsafe_allow_html=True)
    else:
        st.info("This model does not expose probability estimates.")

# Footer
st.markdown("---")
st.markdown("🌌 *Exploring the cosmos, one exoplanet at a time* 🌌")