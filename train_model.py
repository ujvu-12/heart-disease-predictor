# train_model.py
# 💖 Heart Disease Prediction Model Trainer (Fixed Version)

import pandas as pd
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import pickle
import os

# 1️⃣ Load data
print("🔄 Loading dataset...", flush=True)
df = pd.read_csv("data/heart.csv")
print("✅ Data Loaded Successfully! Shape:", df.shape, flush=True)
print(df.head(), flush=True)

# 2️⃣ Handle missing values
print("\n🔍 Missing Values:\n", df.isnull().sum(), flush=True)
df = df.fillna(df.median(numeric_only=True))

# 3️⃣ Convert 'num' to binary target
if "num" in df.columns:
    df["target"] = df["num"].apply(lambda x: 1 if x > 0 else 0)
    df = df.drop(columns=["num"], errors="ignore")

# Drop unnecessary columns
df = df.drop(columns=["id", "dataset"], errors="ignore")

print("\n✅ Cleaned dataset shape:", df.shape, flush=True)
print("Columns:", list(df.columns), flush=True)

# 4️⃣ Encode categorical columns automatically
print("\n🔢 Encoding categorical columns...", flush=True)
cat_cols = df.select_dtypes(include=["object", "bool"]).columns
if len(cat_cols) > 0:
    print("Categorical Columns Found:", list(cat_cols), flush=True)
    for col in cat_cols:
        df[col] = LabelEncoder().fit_transform(df[col].astype(str))
else:
    print("No categorical columns to encode.", flush=True)

# 5️⃣ Split dataset
X = df.drop("target", axis=1)
y = df["target"]

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# 6️⃣ Scale features
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# 7️⃣ Train model
print("\n🚀 Training Random Forest model...", flush=True)
model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

# 8️⃣ Evaluate model
y_pred = model.predict(X_test)
acc = round(accuracy_score(y_test, y_pred) * 100, 2)
print(f"\n🎯 Model Accuracy: {acc}%", flush=True)
print("\n📋 Classification Report:\n", classification_report(y_test, y_pred), flush=True)

# 9️⃣ Confusion Matrix
cm = confusion_matrix(y_test, y_pred)
plt.figure(figsize=(5, 4))
sns.heatmap(cm, annot=True, fmt="d", cmap="Blues")
plt.title(f"Confusion Matrix (Accuracy: {acc}%)")
plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.tight_layout()
plt.savefig("confusion_matrix.png")
print("🖼️ Confusion matrix saved as 'confusion_matrix.png'", flush=True)

# 🔟 Save model
os.makedirs("models", exist_ok=True)
with open("models/heart_model.pkl", "wb") as f:
    pickle.dump(model, f)

print("💾 Model saved successfully at 'models/heart_model.pkl'", flush=True)
print("\n✅ Training Complete! 🎉", flush=True)
