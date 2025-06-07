import pandas as pd
import streamlit as st
import numpy as np
import joblib
import matplotlib.pyplot as plt

#load the trained model
model= joblib.load(r"C:\Users\abc\OneDrive\Desktop\streamlit\socialmedia vs productivity.pkl")
scaler = joblib.load(r"C:\Users\abc\OneDrive\Desktop\streamlit\minmax_scaler.pkl")
# Print the expected feature names
print(scaler.feature_names_in_)

print("Model loaded successfully!")
print("Model type:", type(model))

st.title("📱 Social Media vs Productivity Predictor")

st.markdown("This app predicts your **Productivity Score** based on your digital and work habits.")

# Input sliders based on your dataset statistics
age = st.slider("Age", 18, 65, 41)
socialmedia_time = st.slider("Daily Social Media Time (hrs)", 0.0, 17.9, 3.1)
notifications = st.slider("Notifications per Day", 30, 90, 60)
work_hours_day = st.slider("Work Hours per Day", 0.0, 12.0, 7.0)
stress_level = st.slider("Stress Level (1–10)", 1, 10, 6)
sleep_hours = st.slider("Sleep Hours per Day", 3.0, 10.0, 6.5)
screentime_before_sleep = st.slider("Screen Time Before Sleep (hrs)", 0.0, 3.0, 1.0)
breaks = st.slider("Number of Breaks per Day", 0, 10, 5)
coffee_consumption = st.slider("Cups of Coffee per Day", 0, 10, 2)
days_feeling_burnout = st.slider("Burnout Days per Month", 0, 31, 15)
weekly_offline_hours = st.slider("Weekly Offline Hours", 0.0, 40.0, 10.3)
job_satisfaction = st.slider("Job Satisfaction (1–10)", 0.0, 10.0, 5.0)
productivity_score = st.slider("Daily Productivity Score",2,8,5)


gender = st.selectbox("Gender", ["Male", "Female"])
gender_encoded = 1 if gender == "Male" else 0


job_type = st.selectbox("Job Type", ["Unemployed", "IT", "Finance", "Student", "Education", "Health"])
job_type_encoded = {
    "Unemployed": 0,
    "IT": 1,
    "Finance": 2,
    "Student": 3,
    "Education": 4,
    "Health": 5
}[job_type]

social_platform = st.selectbox("Primary Social Platform", ["Instagram", "Facebook", "Twitter", "LinkedIn", "Other"])
social_platform_encoded = {"Instagram": 0, "Facebook": 1, "Twitter": 2, "LinkedIn": 3, "Other": 4}[social_platform]

uses_focus_apps = st.selectbox("Uses Focus Apps?", ["Yes", "No"])
uses_focus_apps_encoded = 1 if uses_focus_apps == "Yes" else 0

digital_enabled = st.selectbox("Uses Digital Wellbeing Features?", ["Yes", "No"])
digital_enabled_encoded = 1 if digital_enabled == "Yes" else 0

# ---- Feature Preparation ----
numerical_features = [
    'age', 'socialmedia_time', 'notifications', 'work_hours_day',
    'stress_level', 'sleep_hours', 'screentime_before_sleep',
    'breaks', 'coffee_consumption', 'days_feeling_burnout',
    'weekly_offline_hours', 'job_satisfaction'
]

categorical_values = [gender_encoded, job_type_encoded, social_platform_encoded, uses_focus_apps_encoded, digital_enabled_encoded]

# Only scale numerical part
numerical_input = np.array([[age, socialmedia_time, notifications, work_hours_day,
                             stress_level, sleep_hours, screentime_before_sleep,
                             breaks, coffee_consumption, days_feeling_burnout,
                             weekly_offline_hours, job_satisfaction, productivity_score]])

# Scale numerical features
scaled_input = scaler.transform(numerical_input)

# Combine with encoded categorical features
final_input = np.hstack((scaled_input, np.array(categorical_values).reshape(1, -1)))


                             
# Predict
if st.button("Predict Productivity Score"):
    prediction = model.predict(final_input)
    st.success(f"📈 Predicted Productivity Score: **{prediction[0]:.2f}**")

    # Visualize prediction on a gauge
    st.subheader("📊 Prediction Gauge")
    fig, ax = plt.subplots(figsize=(6, 1.5))
    ax.barh([0], prediction[0], color='skyblue')
    ax.set_xlim(0, 10)
    ax.set_yticks([])
    ax.set_xlabel("Predicted Productivity Score (0 - 10)")
    ax.set_title("Your Productivity Estimate")
    st.pyplot(fig)

    # Interpretation
    st.info("**Note:** Higher scores suggest higher productivity based on your habits. Consider reducing social media time or stress for improvement.")

st.markdown("---")
st.caption("🔍 Model: XGBoost | Scaler: MinMax | Dataset: Kaggle - Social Media vs Productivity")

print("Scaled input shape:", scaled_input.shape)
print("Final input shape:", final_input.shape)


#temp change to refresh Deployment
