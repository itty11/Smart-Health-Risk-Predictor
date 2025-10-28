import streamlit as st
import pandas as pd
import numpy as np
import joblib
import os

MODEL_DIR = "models"

st.set_page_config(page_title="Smart Health Risk Predictor", layout="centered")

st.title("🩺 Smart Health Risk Predictor")
st.markdown("Multi-disease prediction (Heart, Diabetes, Kidney, Stroke). Models: LogisticRegression/RandomForest/SVM/XGBoost (best saved).")

# Helper to load model
@st.cache_resource
def load_model(disease):
    path = os.path.join(MODEL_DIR, f"{disease.lower()}_best.pkl")
    if not os.path.exists(path):
        return None
    return joblib.load(path)

# Sidebar: choose disease
disease = st.sidebar.selectbox("Select disease model", ["Diabetes", "Stroke", "Heart", "Kidney"])

model = load_model(disease)
if model is None:
    st.error(f"No saved model found for {disease}. Please run train_and_save_models.py first.")
    st.stop()

st.sidebar.markdown("---")
st.sidebar.markdown("You can either fill the single input form or upload a CSV (batch). CSV must have the same feature columns as the original dataset (without target).")

# Show model info
st.write(f"Loaded model for **{disease}**.")
st.write(model)

# Input area
st.header("Single prediction")
if disease == "Diabetes":
    st.write("Features: Pregnancies, Glucose, BloodPressure, SkinThickness, Insulin, BMI, DiabetesPedigreeFunction, Age")
    pregnancies = st.number_input("Pregnancies", min_value=0, max_value=20, value=0)
    glucose = st.number_input("Glucose", min_value=0, max_value=300, value=120)
    bp = st.number_input("BloodPressure", min_value=0, max_value=200, value=70)
    skin = st.number_input("SkinThickness", min_value=0, max_value=100, value=20)
    insulin = st.number_input("Insulin", min_value=0, max_value=1000, value=80)
    bmi = st.number_input("BMI", min_value=0.0, max_value=80.0, value=25.0)
    dpf = st.number_input("DiabetesPedigreeFunction", min_value=0.0, max_value=5.0, value=0.5)
    age = st.number_input("Age", min_value=0, max_value=120, value=33)
    input_df = pd.DataFrame([{
        "Pregnancies": pregnancies, "Glucose": glucose, "BloodPressure": bp,
        "SkinThickness": skin, "Insulin": insulin, "BMI": bmi,
        "DiabetesPedigreeFunction": dpf, "Age": age
    }])
elif disease == "Stroke":
    st.write("Features: gender, age, hypertension (0/1), heart_disease (0/1), ever_married (Yes/No), work_type, Residence_type, avg_glucose_level, bmi, smoking_status")
    gender = st.selectbox("Gender", ["Male","Female","Other"])
    age = st.number_input("Age", min_value=0.0, max_value=120.0, value=45.0)
    hypertension = st.selectbox("Hypertension", [0,1])
    heart_disease = st.selectbox("Heart Disease", [0,1])
    ever_married = st.selectbox("Ever Married", ["Yes","No"])
    work_type = st.selectbox("Work Type", ["children","Govt_job","Never_worked","Private","Self-employed"])
    residence_type = st.selectbox("Residence Type", ["Urban","Rural"])
    avg_glucose_level = st.number_input("Avg Glucose Level", min_value=0.0, max_value=500.0, value=100.0)
    bmi = st.number_input("BMI", min_value=0.0, max_value=80.0, value=25.0)
    smoking_status = st.selectbox("Smoking Status", ["formerly smoked","never smoked","smokes","Unknown"])
    input_df = pd.DataFrame([{
        "gender": gender, "age": age, "hypertension": hypertension, "heart_disease": heart_disease,
        "ever_married": ever_married, "work_type": work_type, "Residence_type": residence_type,
        "avg_glucose_level": avg_glucose_level, "bmi": bmi, "smoking_status": smoking_status
    }])
elif disease == "Heart":
    st.write("Features: age, sex, cp, trestbps, chol, fbs, restecg, thalach, exang, oldpeak, slope, ca, thal")
    age = st.number_input("Age", min_value=0, max_value=120, value=55)
    sex = st.selectbox("Sex (1=male,0=female)", [1,0])
    cp = st.number_input("Chest pain type (0-3)", min_value=0, max_value=3, value=1)
    trestbps = st.number_input("Resting BP", min_value=0, max_value=300, value=130)
    chol = st.number_input("Cholesterol", min_value=0, max_value=1000, value=250)
    fbs = st.selectbox("Fasting blood sugar > 120 mg/dl (1=yes,0=no)", [1,0])
    restecg = st.selectbox("Resting ECG (0,1,2)", [0,1,2])
    thalach = st.number_input("Max heart rate achieved", min_value=0, max_value=300, value=150)
    exang = st.selectbox("Exercise induced angina (1=yes,0=no)", [1,0])
    oldpeak = st.number_input("ST depression induced by exercise", min_value=0.0, max_value=10.0, value=1.0)
    slope = st.selectbox("Slope of peak exercise ST segment (0-2)", [0,1,2])
    ca = st.number_input("Number of major vessels (0-3)", min_value=0, max_value=4, value=0)
    thal = st.selectbox("Thal (1=normal,2=fixed defect,3=reversible defect)", [1,2,3])
    input_df = pd.DataFrame([{
        "age": age, "sex": sex, "cp": cp, "trestbps": trestbps, "chol": chol,
        "fbs": fbs, "restecg": restecg, "thalach": thalach, "exang": exang,
        "oldpeak": oldpeak, "slope": slope, "ca": ca, "thal": thal
    }])
else:  # Kidney
    st.write("Features: age, bp, sg, al, su, rbc, pc, pcc, ba, bgr, bu, sc, sod, pot, hemo, pcv, wc, rc, htn, dm, cad, appet, pe, ane")
    # For brevity we include a subset; user can upload CSV for full features
    age = st.number_input("Age", min_value=0, max_value=120, value=45)
    bp = st.number_input("BP", min_value=0, max_value=300, value=80)
    bgr = st.number_input("Blood Glucose Random (bgr)", min_value=0.0, max_value=1000.0, value=120.0)
    bu = st.number_input("Blood Urea (bu)", min_value=0.0, max_value=1000.0, value=30.0)
    sc = st.number_input("Serum Creatinine (sc)", min_value=0.0, max_value=50.0, value=1.0)
    hemo = st.number_input("Hemoglobin (hemo)", min_value=0.0, max_value=30.0, value=13.0)
    htn = st.selectbox("Hypertension (htn)", ["yes","no"])
    dm = st.selectbox("Diabetes Mellitus (dm)", ["yes","no"])
    cad = st.selectbox("Coronary Artery Disease (cad)", ["yes","no"])
    input_df = pd.DataFrame([{
        "age": age, "bp": bp, "bgr": bgr, "bu": bu, "sc": sc, "hemo": hemo,
        "htn": htn, "dm": dm, "cad": cad
    }])

st.write("Input preview:")
st.dataframe(input_df)

# CSV upload
st.header("Batch prediction (upload CSV)")
uploaded = st.file_uploader("Upload CSV file with same features (no target column).", type=["csv"])
if uploaded:
    batch_df = pd.read_csv(uploaded)
    st.write("Preview:")
    st.dataframe(batch_df.head())
else:
    batch_df = None

if st.button("Predict"):
    if batch_df is not None:
        X = batch_df
    else:
        X = input_df

    try:
        preds = model.predict(X)
        if hasattr(model, "predict_proba"):
            probs = model.predict_proba(X)[:,1]
        else:
            probs_raw = model.decision_function(X)
            probs = (probs_raw - probs_raw.min())/(probs_raw.max()-probs_raw.min()+1e-9)

        out = X.copy()
        out["prediction"] = preds
        out["probability"] = probs
        st.success("Prediction finished.")
        st.dataframe(out)
        st.download_button("Download results as CSV", out.to_csv(index=False), file_name=f"{disease}_predictions.csv")
    except Exception as e:
        st.error(f"Prediction failed: {e}")
