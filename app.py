import streamlit as st  
import pandas as pd
import numpy as np
import joblib
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix, classification_report

model= joblib.load("model/fraud_detection_model.pkl")
scaler= joblib.load("model/scaler.pkl")

st.set_page_config(page_title= "Credit Card Fraud Detection System ", layout= "wide")

st.title("Credit Card Fraud Detection System")
st.markdown ("### Detect fraudulent transactions using Machine Learning (XGBoost Model)")
st.write("---")

menu=["Overview Dashboard", "Predict Fraud", "Model Info"]
choice= st.sidebar.selectbox("Select Page", menu)

if choice =="Overview Dashboard":
    st.subheader("Dataset Insights")

    df= pd.read_csv("D:\PROJECTS\Credit Card Fraud Detection\Credit Card Fraud\Data\creditcard.csv")

    col1, col2, col3= st.columns(3)
    col1.metric("Total Transactions", f"{len(df):,}")
    col2.metric("Fraudulent", df['Class'].sum())


    st.write("---")
    st.markdown("Transaction Amount Distribution")
    fig2, ax2= plt.subplots(figsize=(6,3))
    sns.histplot(df["Amount"], bins= 50, kde= True, color= "teal", ax= ax2)
    ax2.set_title("Transaction Amount Distribution")
    st.pyplot(fig2)

    st.info("The Dataset is highly imbalanced, so we used SMOTE to handle class imbalance before training.")


elif  choice=="Predict Fraud":
    st.subheader("Enter values Details to Predict Fraud")

    st.markdown("Enter values for the transaction below:")

    with st.form("prediction_form"):
        time= st.number_input ("Transaction Time", min_value= 0.0)
        amount = st.number_input("Transaction Amount", min_value=0.0)
        v_features=[]

        for i in range (1,29):
            val= st.number_input(f"v{i} value", value=0.0, key=f"v{i}")
            v_features.append(val)
        submitted= st.form_submit_button("Predict Transaction")

    if submitted:

        input_data = pd.DataFrame([[time, amount] + v_features], columns=["Time", "Amount"] + [f"v{i}" for i in range (1,29)])

        input_data[["Time", "Amount"]]= scaler.transform(input_data[["Time", "Amount"]])

        prediction= model.predict(input_data)[0]
        pred_prob= model.predict_proba(input_data)[0][1]

        st.write("---")
        if prediction== 1:
            st.error(f"Fraudulent Transaction Detected! (Probability: {pred_prob:.2f})")
        
        else:
            st.success(f"Legit transaction (Probability of Fraud: {pred_prob:.2f})")

elif choice == "Model info":
    st.subheader("Model Details & Evaluation")

    st.markdown("""
    - ** Algorithm Used: ** XGBoost Classifier
    - ** Data Handling: ** SMOTE for balancing fraud samples
    - ** Scaler: ** StandardScaler
    - ** Metrics Used: ** Recall, Precision, F1-score, Roc-Auc
    """ )

    st.write("---")
    st.markdown("### Confusion Matrix(Sample From Validation Set)")
    df= pd.read_csv("D:\PROJECTS\Credit Card Fraud Detection\Credit Card Fraud\Data\creditcard.csv")
    x= df.drop("Class", axis=1)
    y= df["Class"]
    x[["Time", "Amount"]]= scaler.transform(x[["Time", "Amount"]])
    y_pred= model.predict(x)
    cm= confusion_matrix(y, y_pred)

    fig, ax= plt.subplots()
    sns.heatmap(cm, annot=True, fmt='d', camp='Blue', xticklabels=["Legit", "Fraud"], yticklabels=["Legit", "Fraud"], ax= ax)
    ax.set_xlabel("predict")
    ax.set_ylabel("Actual")
    st.pyplot(fig)

    st.write("---")
    st.markdown("### Classification Report")
    st.text(classification_report(y, y_pred))






