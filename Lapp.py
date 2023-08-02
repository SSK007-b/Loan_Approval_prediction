import joblib
import streamlit as st
import numpy as np
import pandas as pd
from sklearn.preprocessing import LabelEncoder
from xgboost import XGBClassifier

@st.cache_resource
def load_util():
    model = joblib.load("LoanModel.pkl")
    scalar = joblib.load("LoanEncode.pkl")
    return model , scalar

st.title(" The loan Approval test ")

st.header(" Chech weather your loan will accept or reject ")

dep = st.number_input("Enter the number of Dependents")

g = st.selectbox("Enter the education" , ("Graduated" , "Not Graduated"))
if(g == "Graduated"):
    edu = 0
else:
    edu = 1

p = st.selectbox("Are you self_employed" , ("No" , "Yes"))
if(p == "No"):
    sem = 0
else:
    sem = 1 

inc = st.number_input("Enter the Annual Income")
lamt = st.number_input("Enter the loan_amount")
lterm = st.number_input("Enter the loan_term")
cib_s = st.number_input("Enter the cibil_score")
rav = st.number_input("Enter the residential_assets_value")
cav = st.number_input("Enter the commercial_assets_value")
lav = st.number_input("Enter the luxury_assets_value")
bav = st.number_input("Enter the bank_asset_value")

newdata = [[dep,edu,sem,inc,lamt,lterm,cib_s,rav,cav,lav,bav]]
data = np.asarray(newdata)
res = data.reshape(1 , -1)

button = st.button("Predict")
if(button):
    model , scalar = load_util()
    st.write(scalar.inverse_transform(model.predict(res)))
else:
    st.write("Check the Prediction: ")
