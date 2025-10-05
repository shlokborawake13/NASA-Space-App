import streamlit as st
import pandas as pd
import numpy as np
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
    
    # Create input array with features
    input_features = np.array([[
        orbital_period, 0.0, impact_param, transit_duration, transit_depth,
        planetary_radius, equilibrium_temp, insolation_flux, 0.0, 0.0,
        5500.0, 4.5, 1.0
    ]])
    
    # Make prediction
    prediction = model.predict(input_features)[0]
    probabilities = model.predict_proba(input_features)[0]
    
    # Display input features table
    st.markdown("## User Input Features:")
    
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
    
    # Labels from training
    labels = ["CANDIDATE", "CONFIRMED", "FALSE POSITIVE", "REFUTED"]
    colors = ["#4CAF50", "#FFA500", "#F44336", "#2196F3"]
    
    predicted_label = labels[int(prediction)]
    predicted_color = colors[int(prediction)]
    
    st.markdown(f"""
    <div class="prediction-box">
        <h3 style="color: {predicted_color};">The model predicts: {predicted_label}</h3>
    </div>
    """, unsafe_allow_html=True)
    
    # Prediction confidence
    st.markdown("## Prediction Confidence")
    
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

# Footer
st.markdown("---")
st.markdown("🌌 *Exploring the cosmos, one exoplanet at a time* 🌌")