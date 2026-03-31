# 🚀 EMIPredict AI - Intelligent Financial Risk Assessment Platform

## 📌 Overview
**EMIPredict AI** is an end-to-end **FinTech machine learning platform** designed to assess financial risk and predict EMI eligibility. It combines classification and regression models with MLflow tracking and a Streamlit web application to deliver real-time, data-driven loan insights.

This platform helps individuals and financial institutions make smarter lending decisions by evaluating financial capacity and predicting safe EMI limits.

---

## 🎯 Problem Statement
Many individuals struggle to pay EMIs due to poor financial planning and lack of proper risk assessment tools.

This project solves this problem by:
- Predicting **EMI eligibility** (Eligible / High Risk / Not Eligible)
- Estimating **maximum safe EMI amount**
- Providing **real-time financial risk analysis**

---

## 💡 Key Features
- ✅ Dual ML Models:
  - Classification → EMI Eligibility
  - Regression → Maximum EMI Amount  
- 📊 Advanced Feature Engineering (22+ features)
- ⚡ Real-time predictions via Streamlit app
- 📈 MLflow integration for experiment tracking
- ☁️ Cloud deployment (Streamlit Cloud)
- 🔁 End-to-end ML pipeline

---

## 🏗️ Project Architecture
Dataset (400K Records) 
        ↓
Data Preprocessing & Cleaning
        ↓ 
Feature Engineering & EDA
        ↓ 
ML Model Training (Classification + Regression) 
        ↓
MLflow Experiment Tracking 
        ↓ 
Model Evaluation & Selection 
        ↓
Streamlit Web Application 
        ↓
Cloud Deployment

---

## 📊 Dataset Information

- **Total Records:** 400,000  
- **Features:** 22+ input variables  
- **Targets:**  
  - EMI Eligibility (Classification)  
  - Max Monthly EMI (Regression)

### 📂 EMI Scenarios
- E-commerce EMI  
- Home Appliances EMI  
- Vehicle EMI  
- Personal Loan EMI  
- Education EMI  

---

## 🔍 Features Used

### 👤 Personal Info
- Age  
- Gender  
- Marital Status  
- Education  

### 💼 Employment & Income
- Monthly Salary  
- Employment Type  
- Years of Employment  

### 🏠 Housing & Family
- House Type  
- Monthly Rent  
- Family Size  
- Dependents  

### 💸 Expenses
- School Fees  
- College Fees  
- Travel Expenses  
- Groceries & Utilities  
- Other Monthly Expenses  

### 🏦 Financial Health
- Existing Loans  
- Current EMI Amount  
- Credit Score  
- Bank Balance  
- Emergency Fund  

### 📝 Loan Details
- EMI Scenario  
- Requested Amount  
- Requested Tenure  

---

## 🤖 Machine Learning Models

### 🔹 Classification Models
- Logistic Regression  
- Random Forest Classifier  
- XGBoost Classifier  

### 🔹 Regression Models
- Linear Regression  
- Random Forest Regressor  
- XGBoost Regressor  

---

## 📈 Model Evaluation Metrics

### Classification:
- Accuracy  
- Precision  
- Recall  
- F1 Score  
- ROC-AUC  

### Regression:
- RMSE  
- MAE  
- R² Score  
- MAPE  

---

## 📊 MLflow Integration
- Experiment tracking  
- Model comparison  
- Parameter logging  
- Model registry for version control  

---

## 🌐 Streamlit Application

### Features:
- 📥 User input for financial details  
- ⚡ Real-time predictions  
- 📊 Data visualization dashboard  
- 🔍 Model performance insights  

---

## 🚀 Deployment

- Platform: **Streamlit Cloud**
- CI/CD: GitHub integration
- Live App: *(Add your link here)*

---

## 💼 Business Use Cases

### 🏦 Financial Institutions
- Automate loan approval process  
- Reduce manual underwriting time  
- Implement risk-based pricing  

### 📱 FinTech Companies
- Instant EMI eligibility checks  
- Digital lending integration  
- Automated risk scoring  

### 📊 Banks & Credit Agencies
- Loan amount recommendations  
- Portfolio risk management  
- Default prediction  

---

## 📦 Project Structure
<pre>
├── data_cleaning.ipynb
├── data-exploration.ipynb                  # (includes visualization)
├── model_training_and_evaluation 
├── data/
├     └── cleaned_dataset.txt               # (includes link to data as the dataset was large to be uploaded)
├     └── model_performance_metrics.txt     # (includes link)
├── stapp.py 
├── requirements.txt 
└── README.md
</pre>
---

## 🛠️ Tech Stack
- Python  
- Pandas, NumPy  
- Scikit-learn  
- XGBoost  
- MLflow  
- Streamlit  

---

## 📌 Results
- 🎯 Classification Accuracy: **> 90%**  
- 📉 Regression RMSE: **< 2000 INR**  
- ⚡ Real-time predictions successfully deployed  

---

## 📈 Future Improvements
- Deep Learning models  
- FastAPI backend integration  
- Mobile application support  
- Real-time banking API integration  

---
Link to MLflow UI - https://dagshub.com/armaaz.au.stats/EMI-Prediction.mlflow   <br>
Link to Streamlit Cloud Application - https://emi-prediction-6dcr5unlcugoiyeyb9awzc.streamlit.app/
## 👨‍💻 Author
**Abdul Razzaq**
