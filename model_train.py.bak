import os
import pandas as pd

import numpy as np

from sklearn.model_selection import train_test_split

from sklearn.preprocessing import StandardScaler

from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score, precision_recall_fscore_support

from imblearn.over_sampling import SMOTE

from xgboost import XGBClassifier

import joblib

import matplotlib.pyplot as plt

import seaborn as sns

data_path = "D:\PROJECTS\Credit Card Fraud Detection\Credit Card Fraud\Data\creditcard.csv"
data =  pd.read_csv("D:\PROJECTS\Credit Card Fraud Detection\Credit Card Fraud\Data\creditcard.csv")
print("Dataset loaded successfully")

x= data.drop("Class", axis=1)
y= data["Class"]

scaler= StandardScaler()
x["Time"]= scaler.fit_transform(x["Time"].values.reshape(-1,1))
x["Amount"]= scaler.fit_transform(x["Amount"].values.reshape(-1,1))

x_train, x_test, y_train, y_test=  train_test_split(x, y, test_size=0.2, random_state=42, stratify=y)
print("Data Split Done")

sm= SMOTE(random_state=42)
x_train_res, y_train_res = sm.fit_resample(x_train, y_train)
print("SMOTE Applied. New Training Size:",  x_train_res.shape)

model= XGBClassifier(n_estimators=100, max_depth=6, learning_rate=0.1, subsample=0.8, random_state= 42)
model.fit(x_train_res , y_train_res)
print("Model Training Complete")

y_pred= model.predict(x_test)
print("\n Classification Report:\n", classification_report(y_test, y_pred))
print("ROC AUC Score:", roc_auc_score(y_test, y_pred))
print("Confusion Matrix:\n", confusion_matrix(y_test, y_pred))

os.makedirs("model", exist_ok = True)
joblib.dump(model, "model/fraud_detection_model.pkl")
print("Model saved successfully in 'model/fraud_detection_model.pkl'")

joblib.dump(scaler, "model/scaler.pkl")
print("Scaler Saved successfully")
