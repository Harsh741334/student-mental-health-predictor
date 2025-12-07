import streamlit as st
import pandas as pd
import joblib
from tensorflow.keras.models import load_model
import numpy as np

MODEL_PATH = "mental_health_model.keras"
PREPROCESSOR_PATH = "preprocessor.pkl"
LABEL_ENCODER_PATH = "label_encoder.pkl"  # optional, only if you saved it


@st.cache_resource
def load_artifacts():
    model = load_model(MODEL_PATH)
    preprocessor = joblib.load(PREPROCESSOR_PATH)
    # label encoder is only needed if target was non-numeric (for inverse_transform)
    try:
        label_encoder = joblib.load(LABEL_ENCODER_PATH)
    except Exception:
        label_encoder = None
    return model, preprocessor, label_encoder


def build_input_df(
    gender, age, city, profession,
    academic_pressure, work_pressure, cgpa,
    study_satisfaction, job_satisfaction,
    sleep_duration, dietary_habits, degree,
    suicidal_thoughts, work_study_hours,
    financial_stress, family_history
):
    # Feature names MUST match training exactly
    # Numerical cols: Age, Academic Pressure, Work Pressure, CGPA, Study Satisfaction, Job Satisfaction, Work/Study Hours
    # Categorical cols: Gender, City, Profession, Sleep Duration, Dietary Habits, Degree, Have you ever had suicidal thoughts ?, Financial Stress, Family History of Mental Illness
    data = {
        "Gender": [str(gender)],
        "Age": [int(age)],
        "City": [str(city)],
        "Profession": [str(profession)],
        "Academic Pressure": [int(academic_pressure)],
        "Work Pressure": [int(work_pressure)],
        "CGPA": [float(cgpa)],
        "Study Satisfaction": [int(study_satisfaction)],
        "Job Satisfaction": [int(job_satisfaction)],
        "Sleep Duration": [str(sleep_duration)],
        "Dietary Habits": [str(dietary_habits)],
        "Degree": [str(degree)],
        "Have you ever had suicidal thoughts ?": [str(suicidal_thoughts)],
        "Work/Study Hours": [int(work_study_hours)],
        "Financial Stress": [str(financial_stress)],  # Categorical in preprocessor!
        "Family History of Mental Illness": [str(family_history)],
    }
    df = pd.DataFrame(data)
    return df


def main():
    st.set_page_config(page_title="Student Mental Health Predictor", page_icon="🧠")
    st.title("🧠 Student Mental Health Prediction")
    st.write("This app uses your trained Keras model to predict **depression risk** for students.")

    model, preprocessor, label_encoder = load_artifacts()

    st.subheader("📋 Enter Student Details")

    col1, col2 = st.columns(2)

    with col1:
        gender = st.selectbox("Gender", ["Male", "Female", "Other"])
        age = st.number_input("Age", min_value=10, max_value=80, value=25, step=1)
        city = st.text_input("City", value="Jaipur")
        profession = st.selectbox("Profession", ["Student", "Working Professional", "Unemployed", "Other"])
        academic_pressure = st.slider("Academic Pressure (1–5)", min_value=1, max_value=5, value=3)
        work_pressure = st.slider("Work Pressure (0–5)", min_value=0, max_value=5, value=0)
        cgpa = st.number_input("CGPA", min_value=0.0, max_value=10.0, value=7.5, step=0.1)
        study_satisfaction = st.slider("Study Satisfaction (1–5)", min_value=1, max_value=5, value=3)

    with col2:
        job_satisfaction = st.slider("Job Satisfaction (0–5)", min_value=0, max_value=5, value=0)
        sleep_duration = st.selectbox(
            "Sleep Duration",
            ["Less than 5 hours", "5-6 hours", "7-8 hours", "More than 8 hours"]
        )
        dietary_habits = st.selectbox("Dietary Habits", ["Healthy", "Moderate", "Unhealthy"])
        degree = st.text_input("Degree", value="B.Tech")
        suicidal_thoughts = st.selectbox("Have you ever had suicidal thoughts?", ["Yes", "No"])
        work_study_hours = st.number_input("Work/Study Hours per day", min_value=0.0, max_value=16.0, value=4.0, step=0.5)
        financial_stress = st.slider("Financial Stress (0–5)", min_value=0, max_value=5, value=2)
        family_history = st.selectbox("Family History of Mental Illness", ["Yes", "No"])

    if st.button("🔮 Predict Depression Risk"):
        # Build DataFrame
        df_input = build_input_df(
            gender=gender,
            age=age,
            city=city,
            profession=profession,
            academic_pressure=academic_pressure,
            work_pressure=work_pressure,
            cgpa=cgpa,
            study_satisfaction=study_satisfaction,
            job_satisfaction=job_satisfaction,
            sleep_duration=sleep_duration,
            dietary_habits=dietary_habits,
            degree=degree,
            suicidal_thoughts=suicidal_thoughts,
            work_study_hours=work_study_hours,
            financial_stress=financial_stress,
            family_history=family_history,
        )

        # Preprocess & predict
        try:
            X_processed = preprocessor.transform(df_input)
            prob = float(model.predict(X_processed)[0][0])
            pred = int(prob > 0.5)

            st.subheader("✅ Prediction Result")
            st.write(f"**Probability of Depression:** `{prob:.3f}`")
            if pred == 1:
                st.error("Model prediction: **High risk of depression (1)**")
            else:
                st.success("Model prediction: **Low risk of depression (0)**")
        except Exception as e:
            st.error(f"Something went wrong during prediction: {e}")


if __name__ == "__main__":
    main()
