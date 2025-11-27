<p align="center">
  <img src="https://img.shields.io/badge/Python-3.9%2B-blue?style=for-the-badge" />
  <img src="https://img.shields.io/badge/Streamlit-App-red?style=for-the-badge&logo=streamlit" />
  <img src="https://img.shields.io/badge/Machine%20Learning-Scikit--Learn-yellow?style=for-the-badge&logo=scikitlearn" />
  <img src="https://img.shields.io/badge/Status-Active-success?style=for-the-badge" />
  <img src="https://img.shields.io/badge/License-MIT-green?style=for-the-badge" />
</p>

# 📉 Customer Churn Prediction (Telecom) — End-to-End ML Web App

**Customer Churn Prediction** is an end-to-end machine learning web application built with **Streamlit** to predict telecom customer churn. The app supports **single and bulk predictions** via CSV uploads and provides interactive dashboards for **churn patterns, key metrics, and actionable insights**.

---

# 🚀 Project Overview

Telecom businesses face challenges in retaining customers. Early identification of high-risk customers helps in:

* Reducing customer churn
* Targeting effective retention campaigns
* Improving revenue and customer lifetime value

This project implements a **full ML pipeline**: from data ingestion and preprocessing to model training and deployment via a web application.

---

# 🎯 Key Objectives

* Predict which customers are likely to churn
* Provide **interactive dashboards** with metrics & visualizations
* Support single and bulk prediction workflows
* Deploy a **user-friendly Streamlit app** for real-time usage
* Maintain a modular and scalable ML pipeline

---

# 🛠️ Project Highlights

### **1️⃣ Data Preprocessing & Feature Engineering**

* Handles missing values, categorical encoding, and scaling
* Generates ready-to-train datasets using `train.pipline.py`
* Stores preprocessing logic in a pipeline for deployment

### **2️⃣ Machine Learning Model**

* Single pipeline (`pipeline.pkl`) integrates preprocessing and model
* Trained to classify customer churn (Yes/No)
* Modular and reusable for predictions

### **3️⃣ Streamlit Web App**

* Input form for single customer prediction
* Bulk prediction via CSV upload
* Dashboard with churn distribution, key metrics, and visual insights
* Real-time model prediction feedback

---

# 📁 Project Structure

```
Customer_Churn_Prediction/
│
├── WA_Fn-UseC_-Telco-Customer-Churn.csv   # Sample telecom churn dataset  
├── train.pipline.py                        # Data preprocessing & training script  
├── pipeline.pkl                            # Serialized trained ML pipeline  
├── App.py                                  # Streamlit app for predictions & dashboard  
├── requirements.txt                        # Python dependencies  
└── README.md                               # This README file  
```

---

# 🖥️ Setup & Installation

### **1️⃣ Clone Repository**

```bash
git clone https://github.com/mansidavane21/Customer_Churn_Prediction.git
cd Customer_Churn_Prediction
```

### **2️⃣ Install Dependencies**

```bash
pip install -r requirements.txt
```

### **3️⃣ Run the Streamlit App**

```bash
streamlit run App.py
```

Open the local URL (default: `http://localhost:8501`) in your browser.

---

# 📌 Example Prediction

### **Single Input Example**

| Feature        | Value     |
| -------------- | --------- |
| Gender         | Male      |
| SeniorCitizen  | 0         |
| Tenure         | 12 months |
| MonthlyCharges | 70.5      |
| TotalCharges   | 845       |

### **Output**

* **Churn Probability:** 23%
* **Prediction:** No Churn

---

# 📈 ML Pipeline (Workflow)

1. Load dataset (CSV)
2. Preprocess features (cleaning, encoding, scaling)
3. Split train/test sets
4. Train ML model (e.g., Logistic Regression / Random Forest)
5. Serialize model + pipeline (`pipeline.pkl`)
6. Deploy via Streamlit app
7. Support single & bulk predictions

---

# 🔮 Future Enhancements

* Add model comparison (Random Forest, XGBoost, etc.)
* Feature explainability (SHAP / feature importance)
* User authentication & database support
* Deployment scripts (Docker / Heroku / AWS)
* Advanced dashboards: cohort analysis, retention trends

---

# 🤝 Contributing

Contributions are welcome!

**Steps:**

1. Fork the repository
2. Create a new branch

```bash
git checkout -b feature-name
```

3. Commit your changes

```bash
git commit -m "Add new feature"
```

4. Push and create a Pull Request

Follow **PEP8 coding standards** for all contributions.

---

# 🙏 Acknowledgements

* **Streamlit** — Web app framework
* **Scikit-learn** — Machine Learning
* **Pandas / NumPy** — Data processing
* **Matplotlib / Seaborn** — Visualization

---

# 👤 Author

**Mansi Davane** — Data Science student & developer (B.Tech, Data Science)
This project is part of my portfolio demonstrating **real-world ML deployment in web apps**.

---
