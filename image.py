# visualize.py
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Load dataset
df = pd.read_csv("D:\PROJECTS\Credit Card Fraud Detection\Credit Card Fraud\Data\creditcard.csv")

# Class Imbalance Plot
plt.figure(figsize=(6,4))
sns.countplot(x='Class', data=df, palette='Set2')
plt.title("Class Imbalance (0 = Non-Fraud, 1 = Fraud)")
plt.savefig("images/class_imbalance.png")
plt.close()

# Correlation Heatmap
plt.figure(figsize=(12,8))
sns.heatmap(df.corr(), cmap='coolwarm', linewidths=0.5)
plt.title("Correlation Heatmap")
plt.savefig("images/heatmap.png")
plt.close()

# Amount Distribution
plt.figure(figsize=(8,5))
sns.histplot(df[df['Class'] == 0]['Amount'], bins=50, color='green', label='Non-Fraud', alpha=0.6)
sns.histplot(df[df['Class'] == 1]['Amount'], bins=50, color='red', label='Fraud', alpha=0.6)
plt.legend()
plt.title("Amount Distribution (Fraud vs Non-Fraud)")
plt.savefig("images/amount_distribution.png")
plt.close()

print("✅ All visualizations saved in the 'images' folder!")
