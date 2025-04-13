import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix
import joblib

# ======================== 1️⃣ DATA LOADING & CLEANING ======================== #

# Define dataset folder path
dataset_folder = r'C:\Users\mohdq\OneDrive\Desktop\internship projects\fraud_trans_dect.pro\fraud_detection\data.unzip'

# Load all .pkl files (one-liner summary)
df_list = [pd.read_pickle(os.path.join(dataset_folder, f)) for f in os.listdir(dataset_folder) if f.endswith('.pkl')]
print(f"Loaded {len(df_list)} files. Combined DataFrame shape: {pd.concat(df_list, ignore_index=True).shape}")
data = pd.concat(df_list, ignore_index=True)

# Convert TX_DATETIME to datetime format
data['TX_DATETIME'] = pd.to_datetime(data['TX_DATETIME'], errors='coerce')

# Drop duplicates and any rows with missing values
data = data.drop_duplicates().dropna()
print("Shape after cleaning (duplicates & missing values):", data.shape)

# ======================== 2️⃣ EDA & TX_AMOUNT CLEANING ======================== #

# Print fraud distribution
fraud_counts = data['TX_FRAUD'].value_counts()
print("Fraud distribution:\n", fraud_counts)

# Plot Fraud Distribution
plt.figure(figsize=(6, 4))
sns.countplot(x=data['TX_FRAUD'], palette=['blue', 'red'])
plt.title("Fraud vs Non-Fraud Transactions")
plt.xlabel("TX_FRAUD (0=Legit, 1=Fraud)")
plt.ylabel("Count")
plt.show()

# TX_AMOUNT cleaning: drop NaNs and replace infinities
print("Checking TX_AMOUNT before cleaning:")
print("NaN values:", data["TX_AMOUNT"].isna().sum())
print("Inf values:", (data["TX_AMOUNT"] == float("inf")).sum())
print("-Inf values:", (data["TX_AMOUNT"] == float("-inf")).sum())

data = data.dropna(subset=["TX_AMOUNT"])
max_valid_amount = data["TX_AMOUNT"][data["TX_AMOUNT"] != float("inf")].max()
data.loc[:, "TX_AMOUNT"] = data["TX_AMOUNT"].replace([float("inf"), float("-inf")], max_valid_amount)

print("Checking TX_AMOUNT after cleaning:")
print("NaN values:", data["TX_AMOUNT"].isna().sum())
print("Inf values:", (data["TX_AMOUNT"] == float("inf")).sum())
print("-Inf values:", (data["TX_AMOUNT"] == float("-inf")).sum())

# Ensure TX_AMOUNT is positive (for log-scale plotting)
data = data[data["TX_AMOUNT"] > 0]

# Plot Transaction Amount Distribution
plt.figure(figsize=(10, 5))
sns.histplot(data, x="TX_AMOUNT", hue="TX_FRAUD", bins=50, kde=True, palette={0: "blue", 1: "red"}, log_scale=True)
plt.title("Transaction Amount Distribution: Fraud vs Non-Fraud")
plt.xlabel("Transaction Amount (Log Scale)")
plt.ylabel("Count")
plt.legend(["Legit", "Fraud"])
plt.show()

# ======================== 3️⃣ FEATURE ENGINEERING ======================== #

# Extract time features from TX_DATETIME
data["TX_HOUR"] = data["TX_DATETIME"].dt.hour
data["TX_DAY_OF_WEEK"] = data["TX_DATETIME"].dt.dayofweek

# Visualize fraud by hour
plt.figure(figsize=(10, 5))
fraud_by_hour = data[data["TX_FRAUD"] == 1]["TX_HOUR"].value_counts().sort_index()
sns.barplot(x=fraud_by_hour.index, y=fraud_by_hour.values, color="red", alpha=0.7)
plt.xlabel("Hour of the Day")
plt.ylabel("Number of Fraud Transactions")
plt.title("Fraud Transactions by Hour")
plt.xticks(range(0, 24))
plt.grid(axis="y", linestyle="--", alpha=0.7)
plt.show()

# Visualize fraud by day of week
plt.figure(figsize=(8, 5))
fraud_by_day = data[data["TX_FRAUD"] == 1]["TX_DAY_OF_WEEK"].value_counts().sort_index()
sns.barplot(x=["Mon", "Tue", "Wed", "Thu", "Fri", "Sat", "Sun"], y=fraud_by_day.values, color="red", alpha=0.7)
plt.xlabel("Day of the Week")
plt.ylabel("Number of Fraud Transactions")
plt.title("Fraud Transactions by Day of the Week")
plt.grid(axis="y", linestyle="--", alpha=0.7)
plt.show()

# Optionally drop TX_DATETIME as features have been extracted
data = data.drop(columns=["TX_DATETIME"])

# ======================== 4️⃣ MODEL TRAINING & EVALUATION (Logistic Regression) ======================== #

# Define features (X) and target (y), dropping TX_FRAUD_SCENARIO to prevent data leakage
X = data.drop(columns=["TX_FRAUD", "TX_FRAUD_SCENARIO"])
y = data["TX_FRAUD"]
print("Feature columns:", X.columns.tolist())

# Train-test split (stratified for imbalanced classes)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
print("Train shape:", X_train.shape, "Test shape:", X_test.shape)

# Scale features
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Initialize and train Logistic Regression
log_reg = LogisticRegression(class_weight='balanced', random_state=42, max_iter=1000)
print("\nTraining Logistic Regression...")
log_reg.fit(X_train_scaled, y_train)
y_pred = log_reg.predict(X_test_scaled)

# Evaluate performance
accuracy = accuracy_score(y_test, y_pred)
precision = precision_score(y_test, y_pred)
recall = recall_score(y_test, y_pred)
f1 = f1_score(y_test, y_pred)
print("Logistic Regression Performance:")
print(f"Accuracy: {accuracy:.4f}")
print(f"Precision: {precision:.4f}")
print(f"Recall: {recall:.4f}")
print(f"F1-score: {f1:.4f}")

# Plot Confusion Matrix
cm = confusion_matrix(y_test, y_pred)
plt.figure(figsize=(6, 4))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
plt.title("Confusion Matrix - Logistic Regression")
plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.show()

# ======================== 5️⃣ SAVE THE MODEL & SCALER ======================== #

joblib.dump(log_reg, "fraud_detection_log_reg_model.joblib")
joblib.dump(scaler, "scaler.joblib")
print("\nLogistic Regression model and scaler saved!")

print(confusion_matrix(y_test, y_pred))
print(data.groupby('TX_FRAUD').mean())
print(data.groupby('TX_FRAUD_SCENARIO')["TX_FRAUD"].value_counts())
print(data.groupby('CUSTOMER_ID')["TX_FRAUD"].mean().sort_values(ascending=False).head(10))
