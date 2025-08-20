

import streamlit as st
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler, LabelEncoder
from xgboost import XGBClassifier
import pickle
import datetime
import lime
import lime.lime_tabular

# Load resources (model, scaler, label encoders, and feature names)
@st.cache_resource
def load_resources():
    model = XGBClassifier()
    model.load_model('xgboost_model.json')  # Replace with your trained model path
    with open('scaler.pkl', 'rb') as f:
        scaler = pickle.load(f)
    with open('label_encoders.pkl', 'rb') as f:
        label_encoders = pickle.load(f)
    with open('feature_names.pkl', 'rb') as f:
        feature_names = pickle.load(f)
    return model, scaler, label_encoders, feature_names

model, scaler, label_encoders, feature_names = load_resources()

# App title and description
st.title("Clinical Trial Outcome Prediction")
st.write("Provide the following details to predict the study status and understand the key contributing factors.")

# User input form
def user_input_form():
    conditions = st.text_input("Conditions", "")
    sponsor = st.text_input("Sponsor", "")
    study_design = st.selectbox("Study Design", ["Single Group Assignment", "Parallel Assignment", "Crossover Assignment", "Factorial Assignment", "Not Specified"])
    funder_type = st.selectbox("Funder Type", ["Industry", "NIH", "Other", "Not Specified"])
    start_date = st.date_input("Start Date", value=datetime.date(2020, 1, 1))
    enrollment = st.number_input("Enrollment", min_value=0, step=1)
    age = st.number_input("Age", min_value=0, step=1)
    trial_duration = st.number_input("Trial Duration (Days)", min_value=0, step=1)
    phase = st.selectbox("Phase", ["Phase 1", "Phase 2", "Phase 3", "N/A"])
    
    return pd.DataFrame({
        "Conditions": [conditions],
        "Sponsor": [sponsor],
        "Study Design": [study_design],
        "Funder Type": [funder_type],
        "Start Date": [start_date],
        "Enrollment": [enrollment],
        "Age": [age],
        "Trial_Duration_Days": [trial_duration],
        "Phases": [phase]
    })

# Define a safe transform function for label encoding
def safe_transform(le, value):
    """Safely transform a value using LabelEncoder, falling back to a default."""
    if value in le.classes_:
        return le.transform([value])[0]
    else:
        # Add unseen label to classes
        le.classes_ = np.append(le.classes_, value)
        return le.transform([value])[0]

# Get user input
input_data = user_input_form()

if st.button("Predict"):
    try:
        # Label encode categorical features
        for col, le in label_encoders.items():
            if col in input_data.columns:
                input_data[col] = input_data[col].map(lambda x: safe_transform(le, x))

        # Ensure all features expected by the model are present (fill with default values if needed)
        for col in feature_names:
            if col not in input_data.columns:
                input_data[col] = 0  # Add missing features with default value 0

        # Ensure that we use exactly the same features expected by the model
        input_data = input_data[feature_names]

        # Debug: Print feature names and input data columns
        st.write("Feature Names (Expected by Model):", feature_names)
        st.write("Input Data Columns (After Matching):", input_data.columns.tolist())

        # Check if the input data shape is correct
        st.write("Input Data Shape (after matching features):", input_data.shape)
        
        if input_data.shape[1] != len(feature_names):
            st.error(f"Feature mismatch: Expected {len(feature_names)} features, got {input_data.shape[1]}")
        else:
            # Standardize the data
            input_scaled = scaler.transform(input_data)

            # Print the shape of the scaled input data
            st.write("Scaled Input Data Shape:", input_scaled.shape)

            # Make prediction
            prediction = model.predict(input_scaled)
            prediction_proba = model.predict_proba(input_scaled)

            # Map prediction to labels
            prediction_label = {0: "Not Completed", 1: "Completed"}[prediction[0]]
            st.subheader(f"Predicted Study Status: **{prediction_label}**")
            st.write("Prediction Probabilities:")
            st.write({label: f"{prob:.2%}" for label, prob in zip(["Not Completed", "Completed"], prediction_proba[0])})

            # LIME Explainability
            st.subheader("Explainability with LIME")
            explainer = lime.lime_tabular.LimeTabularExplainer(
                training_data=np.array(input_scaled),
                feature_names=feature_names,
                class_names=["Not Completed", "Completed"],
                mode="classification"
            )
            exp = explainer.explain_instance(
                data_row=input_scaled[0],
                predict_fn=model.predict_proba
            )

            # Display LIME explanation
            st.write("LIME Explanation:")
            exp.show_in_notebook(show_all=False)
            fig = exp.as_pyplot_figure()
            st.pyplot(fig)
    except ValueError as e:
        st.error(f"Prediction Error: {e}")
