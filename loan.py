import numpy as np
import pandas as pd
from sklearn.preprocessing import LabelEncoder
from xgboost import XGBClassifier
import joblib

df = pd.read_csv(r"D:\ML\dataset\loan_approval_dataset.csv")

X = df.iloc[:, 1:-1]
y = df.iloc[: , -1]

le = LabelEncoder()
y = le.fit_transform(y)

joblib.dump(le , "LoanEncode.pkl")

classifier = XGBClassifier()
classifier.fit(X , y)

"""newdata = [[2,0,  1, 1,4100000,12200000,8,417,2700000,2200000,8800000,3300000]]
data = np.asarray(newdata)
res = data.reshape(1 , -1)
"""
#print(classifier.predict(res))

joblib.dump(classifier , "LoanModel.pkl")