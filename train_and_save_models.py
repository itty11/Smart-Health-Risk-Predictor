import os
import warnings
warnings.filterwarnings("ignore")

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from xgboost import XGBClassifier
from sklearn.metrics import roc_auc_score, accuracy_score, classification_report, confusion_matrix
import joblib

os.makedirs("models", exist_ok=True)
RND = 42

def evaluate_and_save(best_pipeline, X_test, y_test, disease_name):
    y_pred = best_pipeline.predict(X_test)
    if hasattr(best_pipeline, "predict_proba"):
        y_proba = best_pipeline.predict_proba(X_test)[:,1]
    else:  # SVC without prob
        y_proba = best_pipeline.decision_function(X_test)
        y_proba = (y_proba - y_proba.min())/(y_proba.max()-y_proba.min()+1e-9)

    auc = roc_auc_score(y_test, y_proba)
    acc = accuracy_score(y_test, y_pred)
    print(f"\n--- {disease_name} Evaluation ---")
    print(f"AUC: {auc:.4f}  Accuracy: {acc:.4f}")
    print(classification_report(y_test, y_pred))
    print("Confusion Matrix:\n", confusion_matrix(y_test, y_pred))

    save_path = os.path.join("models", f"{disease_name.lower()}_best.pkl")
    joblib.dump(best_pipeline, save_path)
    print(f"Saved best pipeline to {save_path}")

def numeric_categorical_split(df, exclude_cols):
    num_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    num_cols = [c for c in num_cols if c not in exclude_cols]
    cat_cols = [c for c in df.columns if c not in num_cols and c not in exclude_cols]
    return num_cols, cat_cols

# Diabetes (Pima)       
def train_diabetes(path="diabetes.csv"):
    print("Training Diabetes models...")
    df = pd.read_csv(path)
    # Expected columns:
    # Pregnancies, Glucose, BloodPressure, SkinThickness, Insulin, BMI, DiabetesPedigreeFunction, Age, Outcome
    target = "Outcome"
    X = df.drop(columns=[target])
    y = df[target].astype(int)

    num_cols, cat_cols = numeric_categorical_split(X, exclude_cols=[])

    num_pipe = Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", StandardScaler())
    ])
    preproc = ColumnTransformer([
        ("num", num_pipe, num_cols),
    ], remainder="drop")

    pipelines = {
        "logreg": Pipeline([("pre", preproc), ("clf", LogisticRegression(max_iter=1000, random_state=RND))]),
        "rf": Pipeline([("pre", preproc), ("clf", RandomForestClassifier(n_estimators=200, random_state=RND))]),
        "svm": Pipeline([("pre", preproc), ("clf", SVC(probability=True, random_state=RND))]),
        "xgb": Pipeline([("pre", preproc), ("clf", XGBClassifier(use_label_encoder=False, eval_metric="logloss", random_state=RND))]),
    }

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=RND, stratify=y)

    best_auc = -1
    best_pipe = None
    for name, pipe in pipelines.items():
        pipe.fit(X_train, y_train)
        if hasattr(pipe, "predict_proba"):
            proba = pipe.predict_proba(X_test)[:,1]
        else:
            proba = pipe.decision_function(X_test)
            proba = (proba - proba.min())/(proba.max()-proba.min()+1e-9)
        auc = roc_auc_score(y_test, proba)
        print(f"{name} AUC: {auc:.4f}")
        if auc > best_auc:
            best_auc = auc
            best_pipe = pipe
    evaluate_and_save(best_pipe, X_test, y_test, "Diabetes")

# Stroke                
def preprocess_stroke(df):
    # columns: id, gender, age, hypertension, heart_disease, ever_married, work_type,
    # Residence_type, avg_glucose_level, bmi, smoking_status, stroke
    df = df.copy()
    # drop id
    if "id" in df.columns:
        df = df.drop(columns=["id"])
    # some cleaning
    df = df.dropna(subset=["age", "avg_glucose_level"])
    # fill bmi with median
    df["bmi"] = pd.to_numeric(df["bmi"], errors="coerce")
    df["bmi"].fillna(df["bmi"].median(), inplace=True)
    # convert categorical to strings
    for c in ["gender","ever_married","work_type","Residence_type","smoking_status"]:
        if c in df.columns:
            df[c] = df[c].astype(str)
    return df

def train_stroke(path="healthcare-dataset-stroke-data.csv"):
    print("Training Stroke models...")
    df = pd.read_csv(path)
    df = preprocess_stroke(df)
    target = "stroke"
    X = df.drop(columns=[target])
    y = df[target].astype(int)

    num_cols, cat_cols = numeric_categorical_split(X, exclude_cols=[])
    # ensure age and glucose and bmi included
    num_cols = [c for c in num_cols if c in X.columns]
    cat_cols = [c for c in cat_cols if c in X.columns]

    num_pipe = Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", StandardScaler())
    ])
    cat_pipe = Pipeline([
        ("imputer", SimpleImputer(strategy="constant", fill_value="missing")),
        ("ohe", OneHotEncoder(handle_unknown="ignore"))
    ])
    preproc = ColumnTransformer([
        ("num", num_pipe, num_cols),
        ("cat", cat_pipe, cat_cols)
    ], remainder="drop")

    pipelines = {
        "logreg": Pipeline([("pre", preproc), ("clf", LogisticRegression(max_iter=1000, random_state=RND))]),
        "rf": Pipeline([("pre", preproc), ("clf", RandomForestClassifier(n_estimators=200, random_state=RND))]),
        "svm": Pipeline([("pre", preproc), ("clf", SVC(probability=True, random_state=RND))]),
        "xgb": Pipeline([("pre", preproc), ("clf", XGBClassifier(use_label_encoder=False, eval_metric="logloss", random_state=RND))]),
    }

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=RND, stratify=y)
    best_auc = -1
    best_pipe = None
    for name, pipe in pipelines.items():
        pipe.fit(X_train, y_train)
        if hasattr(pipe, "predict_proba"):
            proba = pipe.predict_proba(X_test)[:,1]
        else:
            proba = pipe.decision_function(X_test)
            proba = (proba - proba.min())/(proba.max()-proba.min()+1e-9)
        auc = roc_auc_score(y_test, proba)
        print(f"{name} AUC: {auc:.4f}")
        if auc > best_auc:
            best_auc = auc
            best_pipe = pipe
    evaluate_and_save(best_pipe, X_test, y_test, "Stroke")

# Heart                 
def train_heart(path="heart.csv"):
    print("Training Heart Disease models...")
    df = pd.read_csv(path)
    target = "target"
    X = df.drop(columns=[target])
    y = df[target].astype(int)

    num_cols, cat_cols = numeric_categorical_split(X, exclude_cols=[])
    num_cols = [c for c in num_cols if c in X.columns]
    cat_cols = [c for c in cat_cols if c in X.columns]

    num_pipe = Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", StandardScaler())
    ])
    cat_pipe = Pipeline([
        ("imputer", SimpleImputer(strategy="constant", fill_value="missing")),
        ("ohe", OneHotEncoder(handle_unknown="ignore"))
    ])
    preproc = ColumnTransformer([
        ("num", num_pipe, num_cols),
        ("cat", cat_pipe, cat_cols)
    ], remainder="drop")

    pipelines = {
        "logreg": Pipeline([("pre", preproc), ("clf", LogisticRegression(max_iter=1000, random_state=RND))]),
        "rf": Pipeline([("pre", preproc), ("clf", RandomForestClassifier(n_estimators=200, random_state=RND))]),
        "svm": Pipeline([("pre", preproc), ("clf", SVC(probability=True, random_state=RND))]),
        "xgb": Pipeline([("pre", preproc), ("clf", XGBClassifier(use_label_encoder=False, eval_metric="logloss", random_state=RND))]),
    }

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=RND, stratify=y)
    best_auc = -1
    best_pipe = None
    for name, pipe in pipelines.items():
        pipe.fit(X_train, y_train)
        if hasattr(pipe, "predict_proba"):
            proba = pipe.predict_proba(X_test)[:,1]
        else:
            proba = pipe.decision_function(X_test)
            proba = (proba - proba.min())/(proba.max()-proba.min()+1e-9)
        auc = roc_auc_score(y_test, proba)
        print(f"{name} AUC: {auc:.4f}")
        if auc > best_auc:
            best_auc = auc
            best_pipe = pipe
    evaluate_and_save(best_pipe, X_test, y_test, "Heart")

# Kidney                
def preprocess_kidney(df):
    df = df.copy()
    # convert numeric-like strings to numeric
    for c in ["age","bp","bgr","bu","sc","hemo","pcv","wc","rc","sg","al","su","pot","sod"]:
        if c in df.columns:
            df[c] = pd.to_numeric(df[c], errors="coerce")
    # classification: map 'ckd'/'notckd' or numeric
    if "classification" in df.columns:
        df["classification"] = df["classification"].astype(str).str.lower()
        df["target"] = df["classification"].apply(lambda x: 1 if "ckd" in x or "yes" in x or x=="1" else 0)
    else:
        raise ValueError("kidney dataset must have 'classification' column")
    # drop id if exists
    if "id" in df.columns:
        df = df.drop(columns=["id"])
    return df

def train_kidney(path="kidney_disease.csv"):
    print("Training Kidney Disease models...")
    df = pd.read_csv(path)
    df = preprocess_kidney(df)
    target = "target"
    X = df.drop(columns=[target, "classification"], errors="ignore")
    y = df[target].astype(int)

    num_cols, cat_cols = numeric_categorical_split(X, exclude_cols=[])
    num_cols = [c for c in num_cols if c in X.columns]
    cat_cols = [c for c in cat_cols if c in X.columns]

    num_pipe = Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", StandardScaler())
    ])
    cat_pipe = Pipeline([
        ("imputer", SimpleImputer(strategy="constant", fill_value="missing")),
        ("ohe", OneHotEncoder(handle_unknown="ignore"))
    ])
    preproc = ColumnTransformer([
        ("num", num_pipe, num_cols),
        ("cat", cat_pipe, cat_cols)
    ], remainder="drop")

    pipelines = {
        "logreg": Pipeline([("pre", preproc), ("clf", LogisticRegression(max_iter=1000, random_state=RND))]),
        "rf": Pipeline([("pre", preproc), ("clf", RandomForestClassifier(n_estimators=200, random_state=RND))]),
        "svm": Pipeline([("pre", preproc), ("clf", SVC(probability=True, random_state=RND))]),
        "xgb": Pipeline([("pre", preproc), ("clf", XGBClassifier(use_label_encoder=False, eval_metric="logloss", random_state=RND))]),
    }

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=RND, stratify=y)
    best_auc = -1
    best_pipe = None
    for name, pipe in pipelines.items():
        pipe.fit(X_train, y_train)
        if hasattr(pipe, "predict_proba"):
            proba = pipe.predict_proba(X_test)[:,1]
        else:
            proba = pipe.decision_function(X_test)
            proba = (proba - proba.min())/(proba.max()-proba.min()+1e-9)
        auc = roc_auc_score(y_test, proba)
        print(f"{name} AUC: {auc:.4f}")
        if auc > best_auc:
            best_auc = auc
            best_pipe = pipe
    evaluate_and_save(best_pipe, X_test, y_test, "Kidney")

if __name__ == "__main__":
    # Update paths if your csvs are named / located differently
    train_diabetes(path="diabetes.csv")
    train_stroke(path="healthcare-dataset-stroke-data.csv")
    train_heart(path="heart.csv")
    train_kidney(path="kidney_disease_cleaned.csv")
    print("All training completed.")
