import streamlit as st
import pandas as pd
import io
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer
from joblib import load
from imblearn.over_sampling import SMOTE
from PIL import Image
import numpy as np
import matplotlib.pyplot as plt

# Load the model
model = load('/content/xgboost_model.joblib')  # Adjust the path as necessary
smote = SMOTE(random_state=42)

image2 = Image.open('/content/visual.png')
#add_selectbox = st.sidebar.selectbox (
  #'Ingin Melihat Machine Learning atau Visualisasinya?',
  #('Machine Learning','Visualisasi')
#)
#st.sidebar.info('Web ini digunakan untuk melihat prediksi churn bank berserta visualisasinya')
st.sidebar.image(image2)

def standardize_features(df):
    """Standardize the numerical features."""
    sc = StandardScaler()
    return pd.DataFrame(sc.fit_transform(df), columns=df.columns)

def preprocess(df):
    df = df.drop(columns = ['Naive_Bayes_Classifier_Attrition_Flag_Card_Category_Contacts_Count_12_mon_Dependent_count_Education_Level_Months_Inactive_12_mon_1',
                        'Naive_Bayes_Classifier_Attrition_Flag_Card_Category_Contacts_Count_12_mon_Dependent_count_Education_Level_Months_Inactive_12_mon_2'])
    df.drop(columns=['CLIENTNUM'], inplace=True)

    # Configure to replace Unknow values with missing values
    replace_un = {'Unknown': np.nan}
    df['Education_Level'].replace(replace_un, inplace=True)

    # Manage missing value in Education_Level using SimpleImputer
    imp1 = SimpleImputer(strategy="most_frequent")
    df[['Education_Level']] = imp1.fit_transform(df[['Education_Level']])

    # Configure to replace new variables
    educt_lavel = {'Uneducated':0, 'High School':1, 'College':2,
              'Graduate':3, 'Post-Graduate':4, 'Doctorate':5}
    df.replace(educt_lavel, inplace=True)

    # Configure to replace Unknow values with missing values
    replace_un = {'Unknown': np.nan}
    df['Marital_Status'].replace(replace_un, inplace=True)

    # Manage missing value in Marital status using SimpleImputer
    imp2 = SimpleImputer(strategy="most_frequent")
    df[['Marital_Status']] = imp2.fit_transform(df[['Marital_Status']])

    # Configure to replace new variables
    marital_status = {'Single':0, 'Married':1, 'Divorced':2}
    df.replace(marital_status, inplace=True)

    # Configure to replace Unknow values with missing values
    replace_un = {'Unknown': np.nan}
    df['Income_Category'].replace(replace_un, inplace=True)

    # Manage missing value in Income category using SimpleImputer
    imp3 = SimpleImputer(strategy="most_frequent")
    df[['Income_Category']] = imp3.fit_transform(df[['Income_Category']])

    # Configure to replace new variables
    income_cat = {'Less than $40K': 0, '$40K - $60K': 1, '$60K - $80K': 2,
                '$80K - $120K': 3, '$120K +': 4}
    df.replace(income_cat, inplace=True)

    # Configure to replace new variables
    att_flag = {'Existing Customer':0, 'Attrited Customer':1}
    df['Attrition_Flag'].replace(att_flag, inplace=True)

    # Configure to replace new variables
    gender = {'F':0, 'M':1}
    df['Gender'].replace(gender, inplace=True)

    # Configure to replace new variables
    card_cat = {'Blue':0, 'Silver':1, 'Gold':2, 'Platinum':3}
    df['Card_Category'].replace(card_cat, inplace=True)

    # Scale the features
    df = standardize_features(df)
    return df

st.title("Bank Customer Churn Prediction")
st.write("Upload a CSV file for prediction.")

# File uploader allows user to add their own CSV
uploaded_file = st.file_uploader("Upload your input CSV file", type=["csv"])

if uploaded_file is not None:
    # Read the CSV data into a DataFrame
    input_df = pd.read_csv(uploaded_file)

    processed_data = preprocess(input_df)

    # Preprocessing the DataFrame
    if 'Attrition_Flag' in processed_data.columns:
        processed_data.drop(columns=['Attrition_Flag'], inplace=True)  # Assuming Attrition_Flag is the target and not required for prediction

    processed_data = standardize_features(processed_data)
    # Predicting
    predictions = model.predict(processed_data)
    prediction_probs = model.predict_proba(processed_data)[:, 1]  # Probability of class 1 (churn)

    # Adding predictions to the DataFrame
    input_df['Churn Prediction'] = predictions
    input_df['Prediction Probability'] = prediction_probs

    # Displaying results
    st.write("Combined Data with Predictions:")
    st.dataframe(input_df)

    # Displaying characteristics of churn customers
    st.subheader("Characteristics of Churn Customers:")
    churn_customers = input_df[input_df['Churn Prediction'] == 1]  # Filter churn customers

    # Counting the number of churn customers
    num_churn_customers = churn_customers.shape[0]
    st.write(f"Number of Churn Customers: {num_churn_customers}")

    st.write(churn_customers.describe())  # Display statistics of churn customers

    # Commented out the Churn Prediction Bar Chart
    # Visualization of churn predictions
      # fig, ax = plt.subplots()
      # input_df['Churn Prediction'].value_counts().plot(kind='bar', ax=ax)
      # ax.set_title("Churn Prediction Bar Chart")
      # ax.set_xlabel("Churn Prediction")
      # ax.set_ylabel("Count")
      # st.pyplot(fig)
else:
    st.write("Please upload a file to get predictions.")
