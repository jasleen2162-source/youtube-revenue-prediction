# 📊 YouTube Revenue Predictor

This is a Machine Learning project that predicts **YouTube ad revenue** based on video engagement metrics like views, likes, comments, and watch time.

---

## 🚀 What this project does

- Predicts YouTube ad revenue (`ad_revenue_usd`)
- Uses engagement-based features
- Trains a Linear Regression model
- Provides evaluation metrics like R² score
- Saves trained model for reuse

---




## ⚙️ Features Used

- views  
- likes  
- comments  
- watch_time_minutes  
- category  
- device  
- country  
- date  

---

## 🧠 Feature Engineering

- Engagement Rate = (likes + comments) / views  
- Watch Time per View = watch_time_minutes / views  

---

## 🧹 Data Processing

- Removed duplicates  
- Handled missing values  
- One-hot encoding for categorical features  
- Converted date column  
- Scaled numeric features  

---

## 🤖 Model Used

- Linear Regression (Scikit-learn)

---

## 📊 Evaluation Metric

- R² Score

---

## 🚀 How to run

### 1. Install dependencies

```bash
pip install pandas numpy scikit-learn