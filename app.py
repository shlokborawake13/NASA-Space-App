import streamlit as st
import joblib
import numpy as np
import pandas as pd
import requests
import os

# --- FUNCTION TO DOWNLOAD FILES FROM GITHUB ---
def download_file_from_github(url, local_filename):
    """Downloads a file from a raw GitHub URL if it doesn't already exist."""
    if os.path.exists(local_filename):
        return
    try:
        response = requests.get(url)
        response.raise_for_status() # Raise an exception for bad status codes
        with open(local_filename, 'wb') as f:
            f.write(response.content)
    except requests.exceptions.RequestException as e:
        st.error(f"Error downloading {local_filename}: {e}")
        st.stop()

# --- DEFINE GITHUB RAW FILE URL AND LOCAL PATH ---
# 🚨 IMPORTANT: Replace this with the RAW URL of your .pkl file on GitHub
MODEL_URL = 'https://raw.githubusercontent.com/YOUR_USERNAME/YOUR_REPOSITORY/main/best_lgbm_model_k2pandc.pkl'
MODEL_PATH = 'best_lgbm_model_k2pandc.pkl'

# --- DOWNLOAD AND LOAD THE MODEL ---
download_file_from_github(MODEL_URL, MODEL_PATH)

try:
    model = joblib.load(MODEL_PATH)
except Exception as e:
    st.error(f"Error loading model: {e}")
    st.stop()

# --- APP TITLE AND DESCRIPTION ---
st.set_page_config(page_title="K2 Planet Candidate Classifier", layout="wide")
st.title("🛰️ K2 Planet Candidate Classifier")
st.write("""
This app predicts whether a K2 space telescope object is a confirmed planet or a false positive.
Enter the object's properties in the sidebar to get a prediction.
""")

# --- DEFINE THE INPUT FEATURES IN THE SIDEBAR ---
st.sidebar.header("Object Input Features")

# Based on the notebook, these are the key features for the model
user_inputs = {
    'pl_orbper': st.sidebar.number_input('Orbital Period (days)', value=8.88, format="%.4f"),
    'pl_rade': st.sidebar.number_input('Planetary Radius (Earth radii)', value=1.99, format="%.2f"),
    'pl_bmasse': st.sidebar.number_input('Planetary Mass (Earth mass)', value=4.5, format="%.2f"),
    'st_rad': st.sidebar.number_input('Stellar Radius (Solar radii)', value=0.91, format="%.2f"),
    'st_mass': st.sidebar.number_input('Stellar Mass (Solar mass)', value=0.93, format="%.2f"),
    'sy_dist': st.sidebar.number_input('System Distance (parsecs)', value=135.2, format="%.1f")
}

# --- PREDICTION LOGIC ---
if st.sidebar.button("Classify Object"):
    try:
        # Get the feature names the model was trained on
        expected_features = model.feature_name_
        
        # Create a DataFrame with the correct features and order
        input_df = pd.DataFrame([user_inputs], columns=expected_features)

        st.write("### Input Features:")
        st.dataframe(input_df)

        # Make prediction (the model expects a DataFrame directly)
        prediction_numeric = model.predict(input_df)
        prediction_proba = model.predict_proba(input_df)
        
        # Define the class names based on the notebook's label encoding
        class_names = ['CONFIRMED', 'FALSE POSITIVE']
        result_name = class_names[prediction_numeric[0]]

        st.write("---")
        st.write("### 🤖 Prediction Result")

        if result_name == 'CONFIRMED':
            st.success(f"The model predicts: **{result_name} Planet**")
        else:
            st.error(f"The model predicts: **{result_name}**")

        st.write("### Prediction Confidence")
        proba_df = pd.DataFrame(prediction_proba, columns=class_names, index=['Confidence'])
        st.dataframe(proba_df.style.format("{:.2%}"))

    except Exception as e:
        st.error(f"An error occurred during prediction: {e}")

#### 4. Deploy the App
With all the files in your GitHub repository, you can now deploy your app.

1.  Go to [Streamlit Community Cloud](https://share.streamlit.io/).
2.  Click **"New app"** and connect your GitHub account.
3.  Select your new repository.
4.  Ensure the main file path is set to `app.py`.
5.  Click **"Deploy!"**.

Your second prototype will be live in a few minutes.
