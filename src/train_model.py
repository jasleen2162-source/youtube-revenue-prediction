import pickle
import os
import pandas as pd
import numpy as np

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score


def train(df):

    # -----------------------------
    # 1. CREATE MODEL FOLDER
    # -----------------------------
    os.makedirs("models", exist_ok=True)

    # -----------------------------
    # 2. HANDLE MISSING VALUES
    # -----------------------------
    df = df.fillna(0)

    # -----------------------------
    # 3. OUTLIER REMOVAL (optional but kept minimal)
    # -----------------------------
    q1 = df['ad_revenue_usd'].quantile(0.25)
    q3 = df['ad_revenue_usd'].quantile(0.75)
    iqr = q3 - q1

    df = df[
        (df['ad_revenue_usd'] >= q1 - 1.5 * iqr) &
        (df['ad_revenue_usd'] <= q3 + 1.5 * iqr)
    ]

    # -----------------------------
    # 4. SPLIT FEATURES & TARGET
    # -----------------------------
    X = df.drop(['ad_revenue_usd', 'video_id', 'date'], axis=1, errors="ignore")
    y = df['ad_revenue_usd']

    # -----------------------------
    # 5. ENCODE CATEGORICAL VARIABLES
    # -----------------------------
    X = pd.get_dummies(X, drop_first=True)

    # Save feature columns
    pickle.dump(X.columns, open("models/columns.pkl", "wb"))

    # -----------------------------
    # 6. TRAIN-TEST SPLIT
    # -----------------------------
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    # -----------------------------
    # 7. TRAIN MODEL
    # -----------------------------
    model = LinearRegression()
    model.fit(X_train, y_train)

    # -----------------------------
    # 8. QUICK EVALUATION (minimal)
    # -----------------------------
    y_pred = model.predict(X_test)
    print("R2 Score:", r2_score(y_test, y_pred))

    # -----------------------------
    # 9. SAVE MODEL
    # -----------------------------
    pickle.dump(model, open("models/model.pkl", "wb"))

    return model