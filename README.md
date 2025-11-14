# ❤️ Heart Disease Prediction App

A fully deployed **Machine Learning web application** that predicts the likelihood of heart disease based on patient health parameters.  
Built with **Python, Scikit-Learn, Pandas, and Streamlit**, and deployed on **Streamlit Cloud**.

🔗 **Live App:** https://heart-disease-predictor-01.streamlit.app/  
📁 **GitHub Repo:** https://github.com/ujvu-12/heart-disease-predictor  

---

## 🚀 Features

- ✔ Logistic Regression model trained on the Heart Disease UCI Dataset  
- ✔ Automatic data cleaning & categorical encoding  
- ✔ Standardization using Scikit-Learn's StandardScaler  
- ✔ Real-time predictions  
- ✔ Interactive and user-friendly Streamlit interface  
- ✔ Fully deployed online with a public access link  

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
│── requirements.txt      # Dependencies
│── README.md             # Project Documentation
```

---

## 📌 Dataset

- **Source:** UCI / Kaggle  
- **File:** `heart.csv`  
- **Description:** Contains medical attributes such as age, chest pain type, cholesterol, ECG results, etc.

---

## ⚙️ How to Run Locally

### 1️⃣ Clone the Repo
```bash
git clone https://github.com/ujvu-12/heart-disease-predictor.git
cd heart-disease-predictor
```

### 2️⃣ Create & Activate Virtual Environment
```bash
python3 -m venv venv
source venv/bin/activate
```

### 3️⃣ Install Dependencies
```bash
pip install -r requirements.txt
```

### 4️⃣ Train the Model (optional)
```bash
python3 train_model.py
```

### 5️⃣ Run the App
```bash
streamlit run app.py
```

---

## 🧠 Model Performance

- **Accuracy:** ~82–85%  
- **Model:** Logistic Regression  
- **Preprocessing:**  
  - Encoded categorical variables  
  - Standardized numeric features  
  - Imputed missing values  

---

## 📸 Demo Screenshot

(Add one after running locally)

---

## 🧑‍💻 Author

**Ujvwala Reddy**  
📧 Email: ujvwalareddyp@gmail.com  
🔗 GitHub: https://github.com/ujvu-12  

⭐ **If you like this project, please star the repo!**

