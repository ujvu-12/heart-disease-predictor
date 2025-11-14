# ❤️ Heart Disease Prediction App

A fully deployed **Machine Learning web application** that predicts the likelihood of heart disease based on patient health parameters.  
Built with **Python, Scikit-Learn, Pandas, and Streamlit**, and deployed on **Streamlit Cloud**.

🔗 **Live App:** https://heart-disease-predictor-01.streamlit.app/  
📁 **GitHub Repo:** https://github.com/ujvu-12/heart-disease-predictor  

---

## 📸 Demo Screenshot

Here’s a preview of the deployed app:

![Demo](demo.jpeg)

---

## 🚀 Features

- ✔ Logistic Regression model trained on the Heart Disease UCI Dataset  
- ✔ Automatic data cleaning & categorical encoding  
- ✔ Standardization with Scikit-Learn  
- ✔ Interactive real-time predictions  
- ✔ Beautiful Streamlit UI  
- ✔ Fully deployed online  

---

## 📊 Tech Stack

| Component | Technology |
|----------|------------|
| Programming | Python |
| Libraries | Pandas, NumPy, Scikit-Learn, Streamlit |
| Model | Logistic Regression |
| Deployment | Streamlit Cloud |
| Version Control | Git & GitHub |

---

## 📁 Project Structure

```
heart-disease-predictor/
│── app.py                # Streamlit Web App
│── train_model.py        # Model Training Script
│── heart_model.pkl       # Saved Model
│── heart.csv             # Dataset
│── demo.jpg              # App screenshot
│── requirements.txt      # Dependencies
│── README.md             # Project Documentation
```

---

## 📌 Dataset

- **Source:** UCI / Kaggle  
- **File:** `heart.csv`  
- Contains: age, sex, chest pain type, ECG results, cholesterol, heart rate, etc.

---

## ⚙️ How to Run Locally

### 1️⃣ Clone the Repo
```bash
git clone https://github.com/ujvu-12/heart-disease-predictor.git
cd heart-disease-predictor
```

### 2️⃣ Create Virtual Environment
```bash
python3 -m venv venv
source venv/bin/activate
```

### 3️⃣ Install Requirements
```bash
pip install -r requirements.txt
```

### 4️⃣ Train Model (optional)
```bash
python3 train_model.py
```

### 5️⃣ Run App
```bash
streamlit run app.py
```

---

## 🧠 Model Performance

- **Accuracy:** ~82–85%  
- **Model:** Logistic Regression  
- **Preprocessing:** handled missing values, encoded categorical features, standardized numerical data  

---

## 🧑‍💻 Author

**Ujvwala Reddy**  
📧 Email: ujvwalareddyp@gmail.com  
🔗 GitHub: https://github.com/ujvu-12  

⭐ If you found this project helpful, don't forget to **star the repo**!
