import streamlit as st
import numpy as np
import pickle
import os

# -------------------------------
# Load model
# -------------------------------
MODEL_PATH = "best_RandomForest_model.pkl"

if not os.path.exists(MODEL_PATH):
    st.error("❌ Model file not found!")
    st.stop()

with open(MODEL_PATH, "rb") as f:
    model = pickle.load(f)

# -------------------------------
# Title
# -------------------------------
st.title("🧠 Parkinson’s Disease Severity Prediction")

st.write("Enter patient voice measurement values:")

# -------------------------------
# Feature names (MUST MATCH TRAINING ORDER)
# -------------------------------
feature_names = [
    'age', 'test_time', 'motor_updrs', 'jitter', 'jitter_abs', 'jitter_rap',
       'jitter_ppq5', 'jitter_ddp', 'shimmer', 'shimmer_db', 'shimmer_apq3',
       'shimmer_apq5', 'shimmer_apq11', 'shimmer_dda', 'nhr', 'hnr', 'rpde',
       'dfa', 'ppe'
]

# -------------------------------
# Input fields
# -------------------------------
features = []

for name in feature_names:
    value = st.number_input(name, value=0.0, format="%.6f")
    features.append(value)

# -------------------------------
# Predict button
# -------------------------------
if st.button("🔍 Predict"):
    try:
        # Convert input
        input_data = np.array(features).reshape(1, -1)

        # Prediction
        prediction = model.predict(input_data)[0]

        # Display result
        st.success(f"🧾 Predicted Severity Score: {prediction:.2f}")

        # Severity interpretation
        if prediction < 20:
            st.success("🟢 Mild Parkinson’s")
        elif prediction < 40:
            st.warning("🟡 Moderate Parkinson’s")
        else:
            st.error("🔴 Severe Parkinson’s")

    except Exception as e:
        st.error(f"❌ Error during prediction: {e}")