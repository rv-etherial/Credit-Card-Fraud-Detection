# performance_visuals.py
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import roc_curve, auc, confusion_matrix, classification_report
import joblib

# ✅ Load model and scaler
model = joblib.load("Model/fraud_detection_model.pkl")
scaler = joblib.load("Model/scaler.pkl")

# ✅ Load dataset
df = pd.read_csv("D:\PROJECTS\Credit Card Fraud Detection\Credit Card Fraud\Data\creditcard.csv")

# ✅ Features and target
X = df.drop("Class", axis=1)
y = df["Class"]

# ✅ Scale features
X_scaled = X  # skip scaling for full-dataset visualization

# ✅ Predictions
y_pred = model.predict(X_scaled)
y_prob = model.predict_proba(X_scaled)[:, 1]

# ✅ Confusion Matrix
cm = confusion_matrix(y, y_pred)
plt.figure(figsize=(6,5))
sns.heatmap(cm, annot=True, fmt='d', cmap='Greens')
plt.title("Confusion Matrix")
plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.tight_layout()
plt.savefig("images/confusion_matrix.png")
plt.close()

# ✅ ROC Curve
fpr, tpr, _ = roc_curve(y, y_prob)
roc_auc = auc(fpr, tpr)

plt.figure(figsize=(6,5))
plt.plot(fpr, tpr, color='darkorange', lw=2, label=f'ROC curve (AUC = {roc_auc:.2f})')
plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
plt.xlabel("False Positive Rate")
plt.ylabel("True Positive Rate")
plt.title("Receiver Operating Characteristic (ROC) Curve")
plt.legend(loc="lower right")
plt.tight_layout()
plt.savefig("images/roc_curve.png")
plt.close()

print("✅ Performance visuals saved successfully in 'images' folder!")
