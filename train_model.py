# train_model.py

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
import pickle

print("🔄 Loading heart.csv ...")
df = pd.read_csv("heart.csv")

# Rename target column
df.rename(columns={"num": "target"}, inplace=True)

# Drop unused columns
df.drop(["id", "dataset"], axis=1, inplace=True)

# -----------------------------------------------------------
# 1️⃣ Convert all categorical text columns using one-hot encoding
# -----------------------------------------------------------
df = pd.get_dummies(df, drop_first=True)

# -----------------------------------------------------------
# 2️⃣ Fill any remaining NaN values
# -----------------------------------------------------------
df.fillna(df.mean(), inplace=True)

# -----------------------------------------------------------
# 3️⃣ Split into X and y
# -----------------------------------------------------------
X = df.drop("target", axis=1)
y = df["target"]

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# -----------------------------------------------------------
# 4️⃣ Scaling numeric values
# -----------------------------------------------------------
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)

# -----------------------------------------------------------
# 5️⃣ Train Logistic Regression
# -----------------------------------------------------------
model = LogisticRegression(max_iter=2000)
model.fit(X_train_scaled, y_train)

# -----------------------------------------------------------
# 6️⃣ Save model + scaler + columns
# -----------------------------------------------------------
with open("heart_model.pkl", "wb") as f:
    pickle.dump((model, scaler, X.columns), f)

print("🎉 SUCCESS! heart_model.pkl created with One-Hot Encoded features.")
