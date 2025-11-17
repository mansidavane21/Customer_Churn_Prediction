# ================================================
#  Customer Churn Prediction – Premium Version (10/10 UI)
#  Author: Pranay
#  End-to-End Streamlit App with ML Pipeline
# ================================================
import streamlit as st
import pandas as pd
import numpy as np
import pickle
import seaborn as sns
import matplotlib.pyplot as plt
from io import BytesIO

# -------------------------------
# Page Config
# -------------------------------
st.set_page_config(
    page_title="Customer Churn Prediction",
    layout="wide",
    page_icon="📉",
    initial_sidebar_state="expanded"
)

# -------------------------------
# Custom CSS – (Premium Styling)
# -------------------------------
def add_css():
    st.markdown("""
        <style>
        .block-container {padding-top: 2rem; padding-bottom: 2rem;}

        h1, h2, h3 {color: #1e293b !important;}

        .kpi-card {
            background: white;
            padding: 20px;
            border-radius: 14px;
            box-shadow: 0px 4px 18px rgba(0,0,0,0.07);
            text-align: center;
        }
        .kpi-title {
            font-size: 15px;
            color: #64748b;
            margin-bottom: 4px;
        }
        .kpi-value {
            font-size: 27px;
            font-weight: bold;
            color: #0f172a;
        }

        footer {visibility: hidden;}
        </style>
    """, unsafe_allow_html=True)

add_css()

# -------------------------------
# Load ML Pipeline
# -------------------------------
@st.cache_data
def load_pipeline():
    try:
        with open("pipeline.pkl", "rb") as f:
            return pickle.load(f)
    except:
        st.error("❌ pipeline.pkl not found. Please place it in the same folder.")
        st.stop()

pipeline = load_pipeline()

# -------------------------------
# Load Baseline Dataset
# -------------------------------
@st.cache_data
def load_baseline():
    try:
        df = pd.read_csv("WA_Fn-UseC_-Telco-Customer-Churn.csv")
        df['TotalCharges'] = pd.to_numeric(df['TotalCharges'], errors='coerce')
        df['TotalCharges'].fillna(df['TotalCharges'].median(), inplace=True)
        df['Churn'] = df['Churn'].map({'Yes': 1, 'No': 0})
        return df
    except:
        return None

baseline = load_baseline()

# -------------------------------
# Prediction Helper
# -------------------------------
def predict(pipeline, df):
    df_clean = df.copy()
    if "Churn" in df_clean.columns:
        df_clean = df_clean.drop("Churn", axis=1)

    df_clean["TotalCharges"] = pd.to_numeric(df_clean["TotalCharges"], errors="coerce")
    df_clean["TotalCharges"] = df_clean["TotalCharges"].fillna(df_clean["TotalCharges"].median())

    preds = pipeline.predict(df_clean)
    probs = pipeline.predict_proba(df_clean)[:, 1]

    df_clean["Prediction"] = preds
    df_clean["Probability"] = probs
    return df_clean

# -------------------------------
# Sidebar Navigation
# -------------------------------
st.sidebar.title("📉 Customer Churn App")
menu = st.sidebar.radio(
    "Navigation",
    [
        "🏠 Home",
        "📊 Dashboard",
        "🧮 Predict (Form)",
        "📁 Predict (CSV Upload)",
        "📈 Model Metrics",
        "ℹ️ About Project"
    ]
)

# =================================================================
#  HOME PAGE
# =================================================================
if menu == "🏠 Home":
    st.title("📉 Customer Churn Prediction System")
    st.write("### Welcome to your professional Machine Learning application.")

    st.info("""
    This application predicts telecom customer churn using:
    - 💠 Raw dataset  
    - 💠 Preprocessing Pipeline  
    - 💠 RandomForest Classifier  
    - 💠 Streamlit Dashboard  
    """)

    col1, col2 = st.columns(2)
    with col1:
        st.success("Use the sidebar to navigate between Prediction and Dashboard.")
    with col2:
        st.warning("This project is perfect for college submission & deployment.")

    st.markdown("---")
    st.subheader("🚀 Features")
    st.markdown("""
    - 📊 **Interactive Dashboard**  
    - 🧮 **Single Customer Prediction**  
    - 📁 **Bulk Prediction with CSV**  
    - 📝 **Model Metrics & Evaluation**  
    - 🎨 **Modern Professional UI**  
    """)

# =================================================================
#  DASHBOARD
# =================================================================
elif menu == "📊 Dashboard":
    st.title("📊 Customer Churn Dashboard")

    if baseline is None:
        st.error("Dataset not found. Add WA_Fn-UseC_-Telco-Customer-Churn.csv")
        st.stop()

    df = baseline.copy()

    total = len(df)
    churned = df['Churn'].sum()
    rate = churned / total

    k1, k2, k3 = st.columns(3)
    with k1:
        st.markdown('<div class="kpi-card"><div class="kpi-title">Total Customers</div><div class="kpi-value">'
                    f'{total}</div></div>', unsafe_allow_html=True)
    with k2:
        st.markdown('<div class="kpi-card"><div class="kpi-title">Churned</div><div class="kpi-value">'
                    f'{churned}</div></div>', unsafe_allow_html=True)
    with k3:
        st.markdown('<div class="kpi-card"><div class="kpi-title">Churn Rate</div><div class="kpi-value">'
                    f'{rate:.2%}</div></div>', unsafe_allow_html=True)

    st.markdown("### 🔸 Churn Distribution")
    fig, ax = plt.subplots()
    sns.countplot(x=df['Churn'].map({0: "No", 1: "Yes"}), palette="Set2", ax=ax)
    st.pyplot(fig)

    st.markdown("### 🔸 Contract Type vs Churn")
    fig2, ax2 = plt.subplots(figsize=(8, 4))
    sns.countplot(data=df, x="Contract", hue="Churn", ax=ax2)
    st.pyplot(fig2)

   # Correlation Heatmap
    st.markdown("### 🔸 Correlation Heatmap")
    fig3, ax3 = plt.subplots(figsize=(10, 6))
    numeric_df = df.select_dtypes(include=np.number)  # only numeric
    sns.heatmap(numeric_df.corr(), annot=False, cmap="Blues", ax=ax3)
    st.pyplot(fig3)

# =================================================================
#  PREDICT - FORM
# =================================================================
elif menu == "🧮 Predict (Form)":
    st.title("🧮 Predict Churn (Single Customer Form)")

    if baseline is None:
        st.warning("Dataset missing — dropdown options may be limited.")

    df_sample = baseline.copy() if baseline is not None else None

    with st.form("form"):
        st.subheader("Enter Customer Details")

        col1, col2, col3 = st.columns(3)

        with col1:
            gender = st.selectbox("Gender", ["Female", "Male"])
            SeniorCitizen = st.selectbox("SeniorCitizen", [0, 1])
            Partner = st.selectbox("Partner", ["Yes", "No"])
            Dependents = st.selectbox("Dependents", ["Yes", "No"])

        with col2:
            tenure = st.number_input("Tenure (months)", 0, 100, 10)
            PhoneService = st.selectbox("PhoneService", ["Yes", "No"])
            InternetService = st.selectbox("InternetService", ["DSL", "Fiber optic", "No"])
            PaperlessBilling = st.selectbox("PaperlessBilling", ["Yes", "No"])

        with col3:
            PaymentMethod = st.selectbox("PaymentMethod", ["Electronic check", "Mailed check",
                                                           "Bank transfer (automatic)", "Credit card (automatic)"])
            MonthlyCharges = st.number_input("Monthly Charges", 0.0, 200.0, 70.0)
            TotalCharges = st.number_input("Total Charges", 0.0, 10000.0, 500.0)
            Contract = st.selectbox("Contract", ["Month-to-month", "One year", "Two year"])

        btn = st.form_submit_button("🔍 Predict")

    if btn:
        row = {
            "gender": gender,
            "SeniorCitizen": SeniorCitizen,
            "Partner": Partner,
            "Dependents": Dependents,
            "tenure": tenure,
            "PhoneService": PhoneService,
            "InternetService": InternetService,
            "PaperlessBilling": PaperlessBilling,
            "PaymentMethod": PaymentMethod,
            "MonthlyCharges": MonthlyCharges,
            "TotalCharges": TotalCharges,
            "Contract": Contract,
        }
        df_input = pd.DataFrame([row])

        result = predict(pipeline, df_input)
        pred = result["Prediction"].values[0]
        prob = result["Probability"].values[0]

        if pred == 1:
            st.error(f"⚠️ Customer MAY CHURN — Probability: **{prob:.2f}**")
        else:
            st.success(f"✅ Customer NOT likely to churn — Probability: **{prob:.2f}**")

        st.write(result)

# =================================================================
#  PREDICT - CSV UPLOAD
# =================================================================
elif menu == "📁 Predict (CSV Upload)":
    st.title("📁 Bulk Prediction Using CSV")

    file = st.file_uploader("Upload CSV File", type=["csv"])

    if file:
        df_raw = pd.read_csv(file)
        st.write("### Preview:")
        st.dataframe(df_raw.head())

        if st.button("Run Predictions"):
            output = predict(pipeline, df_raw)
            st.success("Prediction completed.")
            st.dataframe(output.head())

            csv_data = output.to_csv(index=False).encode("utf-8")
            st.download_button("📥 Download Output CSV", data=csv_data, file_name="churn_predictions.csv")

# =================================================================
#  METRICS
# =================================================================
elif menu == "📈 Model Metrics":
    st.title("📈 Model Evaluation Metrics")

    if baseline is None:
        st.error("Dataset not found to calculate metrics.")
        st.stop()

    X = baseline.drop("Churn", axis=1)
    y = baseline["Churn"]

    preds = pipeline.predict(X)
    acc = (preds == y).mean()

    st.metric("Accuracy", f"{acc:.2%}")

# =================================================================
#  ABOUT PAGE
# =================================================================
elif menu == "ℹ️ About Project":
    st.title("ℹ️ About This Project")

    st.markdown("""
    ### 📘 Customer Churn Prediction – End-to-End ML Project  
    This project predicts whether a telecom customer will *churn* based on behavior, services, and billing patterns.

    #### 🔹 Technologies Used:
    - Python, Pandas, NumPy  
    - Scikit-Learn  
    - RandomForest Classifier  
    - Imbalanced-Learn (SMOTE)  
    - Streamlit (Deployment & UI)

    #### 🔹 ML Pipeline Includes:
    - Data Cleaning  
    - Encoding  
    - Scaling  
    - SMOTE Balancing  
    - Model Training  
    - Saving as `pipeline.pkl`

    #### 🔹 Streamlit App Includes:
    - Interactive Dashboard  
    - Advanced UI (Premium Version)
    - Single + Bulk Prediction  
    - Model Metrics  

    #### 👨‍💻 Developed by Mansi Davane with ❤️  
    """)

# =================================================================
# Footer
# =================================================================
st.markdown("<br><center>Made with ❤️ by Mansi Davane</center><br>", unsafe_allow_html=True)
