# 🩺 Smart Health Risk Predictor using Machine Learning

A machine learning-based application that predicts the likelihood of four major health conditions — **Heart Disease**, **Diabetes**, **Kidney Disease**, and **Stroke** — using patient health parameters.

---

## 🚀 Project Overview

This project applies supervised ML algorithms to predict health risks based on medical datasets.  
It trains multiple models — Logistic Regression, Random Forest, SVM, and XGBoost — and selects the best-performing one for each disease.

---

## 🧠 Diseases Covered

| Disease | Dataset Source | Best Model | AUC Score | Accuracy |
|----------|----------------|-------------|------------|-----------|
| 🫀 Heart Disease | UCI Cleveland Dataset | **Random Forest / XGBoost** | 1.0000 | 1.0000 |
| 💉 Diabetes | PIMA Indian Diabetes Dataset | **Logistic Regression** | 0.8230 | 0.7143 |
| 🧠 Stroke | Kaggle Stroke Dataset | **Logistic Regression** | 0.8418 | 0.9521 |
| 🧫 Kidney Disease | UCI CKD Dataset | **All Models (Perfect Scores)** | 1.0000 | 0.9875 |

---

## ⚙️ Model Training Summary

### 🩸 Diabetes

AUC: 0.8230

Accuracy: 0.7143

Precision: 0.76 / 0.61

Confusion Matrix: [[82 18], [26 28]]

Saved Model → models/diabetes_best.pkl


### 🧠 Stroke

AUC: 0.8418

Accuracy: 0.9521

Precision: 0.95 / 1.00

Confusion Matrix: [[972 0], [49 1]]

Saved Model → models/stroke_best.pkl


### ❤️ Heart Disease

AUC: 1.0000

Accuracy: 1.0000

Precision: 1.00 / 1.00

Confusion Matrix: [[100 0], [0 105]]

Saved Model → models/heart_best.pkl


### 🧫 Kidney Disease

AUC: 1.0000

Accuracy: 0.9875

Precision: 0.97 / 1.00

Confusion Matrix: [[30 0], [1 49]]

Saved Model → models/kidney_best.pkl


---

## 🧰 Tech Stack

- **Language:** Python 3.x  

- **Libraries:** scikit-learn, XGBoost, pandas, numpy, matplotlib  

- **Framework:** Streamlit (for interactive UI)  

- **Containerization:** Docker + Docker Compose  

---

## 🐳 Docker Setup

### 🧾 Dockerfile

A lightweight Python image is used to containerize the Streamlit app.

### 🧩 docker-compose.yml

Brings up the app easily with one command and auto-restarts on failure.


docker-compose up


## 📊 Evaluation Summary

All models achieved strong AUC and accuracy across datasets, with Heart and Kidney achieving near-perfect results.

The Logistic Regression model was the best trade-off model for Diabetes and Stroke, showing strong generalization performance.

## 🌐 Deployment

You can deploy this containerized app on:

Azure App Service

Google Cloud Run

AWS ECS / Elastic Beanstalk

Streamlit Cloud - https://smart-health-risk-predictor-737h82qvj2trgdq4exxdjs.streamlit.app/


## ✨ Future Enhancements

Add patient data input form with validation

Integrate explainable AI (SHAP/LIME)

Add a REST API using FastAPI for external access

Store predictions in a database for analytics

## 👨‍💻 Author

Ittyavira C Abraham

MCA (AI), Amrita Vishwa Vidyapeetham (Amrita Ahead)

📧 ittyavira.c.abraham@gmail.com
