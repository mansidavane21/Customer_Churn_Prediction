import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, accuracy_score, roc_auc_score
from imblearn.over_sampling import SMOTE
import pickle

# Load dataset
df = pd.read_csv("WA_Fn-UseC_-Telco-Customer-Churn.csv")

# Clean TotalCharges
df['TotalCharges'] = pd.to_numeric(df['TotalCharges'], errors='coerce')
df['TotalCharges'] = df['TotalCharges'].fillna(df['TotalCharges'].median())

# Encode target column
df['Churn'] = df['Churn'].map({'No': 0, 'Yes': 1})

# Remove customerID if present
if 'customerID' in df.columns:
    df = df.drop(columns=['customerID'])

# Define numeric & categorical columns
numeric_cols = ["SeniorCitizen", "tenure", "MonthlyCharges", "TotalCharges"]
categorical_cols = [c for c in df.columns if c not in numeric_cols + ['Churn']]

X = df.drop("Churn", axis=1)
y = df["Churn"]

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

# Preprocessor
preprocessor = ColumnTransformer(
    transformers=[
        ("num", StandardScaler(), numeric_cols),
        ("cat", OneHotEncoder(handle_unknown="ignore", sparse_output=False), categorical_cols)
    ]
)


# SMOTE only on encoded training set
smote = SMOTE(random_state=42)
X_train_trans = preprocessor.fit_transform(X_train)
X_res, y_res = smote.fit_resample(X_train_trans, y_train)

# Random Forest Model
rf = RandomForestClassifier(n_estimators=200, random_state=42, n_jobs=-1)
rf.fit(X_res, y_res)

# Final pipeline (preprocessor + model)
pipeline = Pipeline(steps=[
    ("preprocessor", preprocessor),
    ("model", rf)
])

# Evaluate
y_pred = pipeline.predict(X_test)
y_proba = pipeline.predict_proba(X_test)[:, 1]

print("Accuracy:", accuracy_score(y_test, y_pred))
print("ROC AUC:", roc_auc_score(y_test, y_proba))
print("\nClassification Report:\n", classification_report(y_test, y_pred))

# Save pipeline
with open("pipeline.pkl", "wb") as f:
    pickle.dump(pipeline, f)

print("pipeline.pkl saved successfully!")
