<p align="center">
  <img src="https://img.shields.io/badge/Python-3.9%2B-blue?style=for-the-badge" />
  <img src="https://img.shields.io/badge/Streamlit-App-red?style=for-the-badge&logo=streamlit" />
  <img src="https://img.shields.io/badge/Machine%20Learning-Scikit--Learn-yellow?style=for-the-badge&logo=scikitlearn" />
  <img src="https://img.shields.io/badge/Status-Active-success?style=for-the-badge" />
  <img src="https://img.shields.io/badge/License-MIT-green?style=for-the-badge" />
</p>
📉 Customer Churn Prediction (Telecom) — End-to-End ML Web App

End-to-end machine learning web application using Streamlit to predict telecom customer churn. Supports single & bulk predictions via CSV upload, and includes interactive dashboards for churn patterns, key metrics, and insights. UI features a modern, clean design with tech badges, animated cards, and a deployment-ready layout.

GitHub

🎯 Why this Project?
Businesses — especially in telecom and subscription services — often struggle to retain customers. Predicting which customers are likely to churn helps in:
* Identifying high-risk users early
* Targeting retention campaigns
* Reducing revenue loss and improving customer lifetime value

This project demonstrates a full ML pipeline — from data ingestion, preprocessing, model training, to deployment — to predict customer churn and derive actionable insights.


🛠️ What’s Inside (Project Highlights)

✅ Data preprocessing and feature engineering (in train.pipline.py) 
✅ Model training & saving (using pipeline.pkl) 
✅ Web interface (App.py) built with Streamlit — for single and bulk prediction, with interactive dashboards and metrics view. 
✅ Sample dataset included (WA_Fn-UseC_-Telco-Customer-Churn.csv) for testing / demonstration. 
✅ Requirements file (requirements.txt) for easy setup.


📁 Repository Structure
Customer_Churn_Prediction/
│
├── WA_Fn-UseC_-Telco-Customer-Churn.csv   # Sample telecom churn dataset  
├── train.pipline.py                        # Data preprocessing + training script  
├── pipeline.pkl                            # Serialized trained model & preprocessing pipeline  
├── App.py                                  # Streamlit app for prediction & dashboard  
├── requirements.txt                        # Python dependencies  
└── README.md                               # README (this file)  


⚙️ Setup & Usage
1. Clone the repository
git clone https://github.com/mansidavane21/Customer_Churn_Prediction.git
cd Customer_Churn_Prediction

2. Install dependencies
pip install -r requirements.txt

3. Run the Streamlit App
streamlit run App.py


The app will open in your browser (usually at http://localhost:8501).
You can:
Upload a single customer’s data for immediate churn prediction
Upload a CSV file for bulk predictions
Explore dashboards showing churn distribution, key metrics, visualizations, and insights


📈 How It Works (Brief Workflow)

1. Load dataset (or user-provided data)
2. Preprocess data (cleaning, encoding, scaling, feature engineering)
3. Use trained ML pipeline (pipeline.pkl) to predict churn probability
4. Present results in a user-friendly UI — show prediction, summary metrics, and visual dashboards
5. Support both single and bulk predictions


✅ Future Enhancements (Roadmap)

* Add model comparison (e.g. Random Forest, XGBoost) and show performance metrics/ranking
* Add threshold tuning & explainability (e.g. SHAP / feature importance)
* Add user authentication & database support to store predictions/history
* Provide deployment scripts (Docker / Heroku / AWS) for one-click deployment
* Improve UI with more visualizations (cohort analysis, retention trends, etc.)


👤 About / Author

Mansi Davane — Data Science student & developer (B.Tech, Data Science)
This project is part of my portfolio showcasing real-world usage of data science + web app deployment.
