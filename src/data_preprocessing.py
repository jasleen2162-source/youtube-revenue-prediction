import pandas as pd
from sklearn.preprocessing import StandardScaler

def load_data(path):
    return pd.read_csv(path)


def clean_data(df):
    # -------------------------
    # 1. Remove duplicates (~2%)
    # -------------------------
    df = df.drop_duplicates()

    # -------------------------
    # 2. Handle missing values (~5%)
    # -------------------------

    # Numeric columns → fill with median
    numeric_cols = df.select_dtypes(include=["int64", "float64"]).columns
    df[numeric_cols] = df[numeric_cols].fillna(df[numeric_cols].median())

    # Categorical columns → fill with mode
    categorical_cols = df.select_dtypes(include=["object"]).columns
    df[categorical_cols] = df[categorical_cols].fillna(
        df[categorical_cols].mode().iloc[0]
    )

    # -------------------------
    # 3. Convert date column (if exists)
    # -------------------------
    if "date" in df.columns:
        df["date"] = pd.to_datetime(df["date"])

    return df


def encode_features(df):
    # -------------------------
    # 4. Encode categorical variables
    # -------------------------
    categorical_cols = ["category", "device", "country"]

    existing_cols = [col for col in categorical_cols if col in df.columns]

    df = pd.get_dummies(df, columns=existing_cols, drop_first=True)

    return df


def scale_features(df):
    # -------------------------
    # 5. Normalize / Scale numeric features
    # -------------------------
    scaler = StandardScaler()

    numeric_cols = df.select_dtypes(include=["int64", "float64"]).columns

    df[numeric_cols] = scaler.fit_transform(df[numeric_cols])

    return df


def preprocess(path):
    df = load_data(path)
    df = clean_data(df)
    df = encode_features(df)
    df = scale_features(df)
    return df