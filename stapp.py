import streamlit as st
import mlflow
import pandas as pd
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from scipy.sparse import hstack
from sklearn.preprocessing import LabelEncoder, StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.model_selection import train_test_split, cross_val_score, GridSearchCV
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.metrics import r2_score, accuracy_score, confusion_matrix, mean_squared_error, root_mean_squared_error, mean_absolute_percentage_error, precision_score, recall_score, roc_auc_score, f1_score, mean_absolute_error
from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier
from xgboost import XGBRegressor, XGBClassifier
import imblearn
from imblearn.over_sampling import SMOTE
import mlflow.xgboost
import os

os.environ["MLFLOW_TRACKING_USERNAME"] = "armaaz.au.stats"
os.environ["MLFLOW_TRACKING_PASSWORD"] = "fdddefe65a81239ff8f470c11acd7dc776fd2a00"
mlflow.set_tracking_uri("https://dagshub.com/armaaz.au.stats/EMI-Prediction.mlflow")

st.set_page_config(layout='wide')

st.sidebar.title("Select Page")
selected_page = st.sidebar.radio(
    "select page",
    ['Predict', 'Data Exploration and Visualization', 'Model Performance']
)


### ------------ FIRST PAGE -------------

if selected_page=='Predict':
    st.markdown("# Predict EMI eligibility and Monthly EMI")

    xgb_class = mlflow.xgboost.load_model("models:/XGB Classifier/1")
    xgb_reg = mlflow.xgboost.load_model("models:/XGB Regressor/1")

    # User Inputs
    age = st.number_input("Age", min_value=18, max_value=100, value=30)
    gender = st.selectbox("Gender", ["M", "F"])
    marital_status = st.selectbox("Marital Status", ["Single", "Married"])
    education = st.selectbox("Education", ["High School","Graduate", "Post Graduate", "Professional"])
    monthly_salary = st.number_input("Monthly Salary", min_value=0, value=50000)
    emp_type = st.selectbox("Employment Type", ["Private", "Government", "Self-employed"])
    yoe = st.number_input("Years of Employment", min_value=0, value=2)
    c_type = st.selectbox("Company Type", ["Startup", "MNC", "Large Indian", "Small", "Mid-size"])
    h_type = st.selectbox("House Type", ["Rented", "Own", "Family"])
    monthly_rent = st.number_input("Monthly Rent", min_value=0, value=0)
    family_size = st.number_input("Family Size", min_value=1, value=3)
    dependents = st.number_input("Dependents", min_value=0, value=1)
    school_fee = st.number_input("School Fees", min_value=0, value=0)
    college_fee = st.number_input("College Fees", min_value=0, value=0)
    travel_expense = st.number_input("Travel Expense", min_value=0, value=0)
    groc_util = st.number_input("Grocery/Utility Expense", min_value=0, value=0)
    other_monthly_expense = st.number_input("Other Monthly Expense", min_value=0, value=0)
    existing_loans = st.number_input("Existing Loans Count", min_value=0, value=0)
    current_emi = st.number_input("Current EMI", min_value=0, value=0)
    credit_score = st.number_input("Credit Score", min_value=300, max_value=900, value=700)
    bank_balance = st.number_input("Bank Balance", min_value=0, value=50000)
    e_fund = st.number_input("Emergency Fund", min_value=0, value=10000)
    emi_scenario = st.selectbox("EMI Scenario", ["Vehicle EMI", "Home Appliances EMI", "Personal Loan EMI", "E-commerce Shopping EMI", "Education EMI"])
    req_amt = st.number_input("Requested Amount", min_value=0, value=100000)
    req_ten = st.number_input("Requested Tenure (months)", min_value=1, value=12)


    inputs = [age, gender, marital_status, education, monthly_salary, emp_type, yoe, c_type,
              h_type, monthly_rent, family_size, dependents, school_fee, college_fee, travel_expense,
              groc_util, other_monthly_expense, existing_loans, current_emi, credit_score, bank_balance, e_fund,
              emi_scenario, req_amt, req_ten]
    cols =['age', 'gender', 'marital_status', 'education', 'monthly_salary',
       'employment_type', 'years_of_employment', 'company_type', 'house_type',
       'monthly_rent', 'family_size', 'dependents', 'school_fees',
       'college_fees', 'travel_expenses', 'groceries_utilities',
       'other_monthly_expenses', 'existing_loans', 'current_emi_amount',
       'credit_score', 'bank_balance', 'emergency_fund', 'emi_scenario',
       'requested_amount', 'requested_tenure']
    
    df = pd.DataFrame([inputs], columns=cols)

    # add calculated features
    df['total_expense'] = (df['monthly_rent'] + 
                        df['college_fees'] + 
                        df['school_fees'] +
                        df['travel_expenses'] +
                        df['groceries_utilities'] + 
                        df['other_monthly_expenses'])
    df['expense_ratio'] = df['total_expense'] / df['monthly_salary']
    df['emi_burden'] = df['current_emi_amount'] / df['monthly_salary']

    st.markdown("### Your Input Dataframe")
    st.dataframe(df, hide_index=True)
    # st.write(df.dtypes)

    if st.button("Confirm and Run Predictions"):
        # encode for prediction
        gender_map = {
            'F' : '0',
            'M' : '1'
        }

        marital_map = {
            'Married' : '0',
            'Single' : '1'
        }

        education_map = {
            'High School' : '0',
            'Graduate' : '1',
            'Post Graduate' : '2',
            'Professional' : '3'
        }

        employment_map = {
            'Self-employed' : '0',
            'Private' : '1',
            'Government' : '2'
        }

        company_map = {
            'Startup' : '0',
            'Small' : '1',
            'Mid-size' : '2',
            'Large Indian' : '3',
            'MNC' : '4'
        }

        house_map = {
            'Rented' : '0',
            'Family' : '1',
            'Own' : '2'
        }

        existing_loan_map = {
            'No' : '0',
            'Yes' : '1'
        }

        emi_scenario_map = {
            'Personal Loan EMI' : '0',
            'E-commerce Shopping EMI' : '1',
            'Education EMI' : '2',
            'Vehicle EMI' : '3',
            'Home Appliances EMI' : '4'
        }
        def encoder(df):
            df['gender'] = df['gender'].replace(gender_map).astype('int')
            df['marital_status'] = df['marital_status'].replace(marital_map).astype('int')
            df['education'] = df['education'].replace(education_map).astype('int')
            df['employment_type'] = df['employment_type'].replace(employment_map).astype('int')
            df['company_type'] = df['company_type'].replace(company_map).astype('int')
            df['house_type'] = df['house_type'].replace(house_map).astype('int')
            df['existing_loans'] = df['existing_loans'].replace(existing_loan_map).astype('int')
            df['emi_scenario'] = df['emi_scenario'].replace(emi_scenario_map).astype('int')

            return df

        test_row = encoder(df)
        st.session_state.show_prediction = True

        eligibility = xgb_class.predict(test_row)
        decode_map = {
                        0 : "Eligible",
                        1 : "High-Risk",
                        2 : "Not-Eligible"
                    }
        result = decode_map[int(eligibility[0])]
        result_df = pd.DataFrame([result], columns=['EMI Eligibility'])
        st.dataframe(result_df, hide_index=True)

        emi = xgb_reg.predict(test_row)
        monthly_emi = [f"₹{emi[0]:,.3f}"]
        emi_df = pd.DataFrame(monthly_emi, columns=['Max monthly EMI (INR)'])
        st.dataframe(emi_df, hide_index=True)



### ------------ SECOND PAGE -------------
    
elif selected_page == 'Data Exploration and Visualization':
    st.markdown("# Data Exploration and Visualization")

    @st.cache_data
    def prediction_data():
        file_id = "1nAeU--cSSX_jBmhl9m8ZakJYfJ7L_x4y"
        url = f"https://drive.google.com/uc?id={file_id}"
        return pd.read_csv(url)   
    
    df = prediction_data()

    st.markdown("### Data Exploration")

    st.markdown("##### Shape of the dataset")
    st.write(df.shape)

    st.markdown("##### Columns present in the dataset")

    columns = df.columns
    for col in columns:
        st.markdown(f'- {col}')
    
    st.markdown("##### First five rows")
    head = df.head()
    st.dataframe(head, hide_index=True)

    st.markdown("##### Last five rows")
    tail = df.tail()
    st.dataframe(tail, hide_index=True)

    st.markdown("##### Description of the dataset")
    des = df.describe()
    st.dataframe(des, hide_index=True)


    st.markdown("### Data Visualization")

    st.markdown("#### Average monthly salary wrt to education")

    data = df.groupby('education')['monthly_salary'].mean()

    plt.figure(figsize=(6,3))
    plt.bar(data.index,data.values)
    plt.title("Average Monthly Salary (Education)")

    st.pyplot(plt)
    plt.clf()

    st.markdown("#### Average monthly salary wrt to age")

    data = df.groupby('age')['monthly_salary'].mean()
    plt.title("Average Monthly Salary (Age)")
    sns.lineplot(data)

    st.pyplot(plt)
    plt.clf()

    sns.histplot(df['max_monthly_emi'], kde=True)
    st.pyplot(plt)
    plt.clf()

    st.markdown("#### Numerical Columns Analysis")
    num_cols = df.select_dtypes(include=['int64', 'float64']).columns
    df[num_cols].hist(figsize=(22,12), bins=30)
    st.pyplot(plt)
    plt.clf()

    for col in num_cols:
        sns.boxplot(x=df[col])
        st.pyplot(plt)
        plt.clf()

    st.markdown("#### Correlation HeatMap")
    plt.figure(figsize=(22,10))
    sns.heatmap(df[num_cols].corr(), annot=True, cmap='coolwarm')
    st.pyplot(plt)
    plt.clf()

    st.markdown("#### Categorical Column Analysis")
    cat_cols = df.select_dtypes(include=['object']).columns
    for col in cat_cols:
        sns.countplot(data=df, x=col, hue ='emi_eligibility')
        plt.xticks(rotation=45)
        st.pyplot(plt)
        plt.clf()
    
    for col in cat_cols:
        sns.boxplot(data=df, x=col, y='max_monthly_emi')
        st.pyplot(plt)
        plt.clf()

    st.subheader("Some target base analysis:")
    sns.scatterplot(data=df, x='monthly_salary', y='max_monthly_emi', color='red')
    st.pyplot(plt)
    plt.clf()
    sns.scatterplot(data=df, x='credit_score', y='max_monthly_emi', color='blue')
    st.pyplot(plt)
    plt.clf()
    sns.scatterplot(data=df, x='current_emi_amount', y='max_monthly_emi', color='green')
    st.pyplot(plt)
    plt.clf()




### ------------ THIRD PAGE -------------
elif selected_page == 'Model Performance':
    st.markdown("# Model Performance Metrics and MLflow")
    # load performance metrics

    @st.cache_data
    def reg_met_data():
        file_id = "11D30SDgHIEgIaePMwwV4B2380PG2tm02"
        url = f"https://drive.google.com/uc?id={file_id}"
        return pd.read_csv(url)
    
    reg_metrics = reg_met_data()

    @st.cache_data
    def class_met_data():
        file_id = "1HBfvgss9elXV6iLzu7oN-lgDMC_WHm8K"
        url = f"https://drive.google.com/uc?id={file_id}"
        return pd.read_csv(url)
    class_metrics = class_met_data()

    st.markdown("## Regression Models' Performance")
    st.dataframe(reg_metrics)

    st.markdown("## Classification Models' Performance")
    st.dataframe(class_metrics)

    st.markdown("### Cloud MLflow link:")
    st.markdown("#### https://dagshub.com/armaaz.au.stats/EMI-Prediction.mlflow")
