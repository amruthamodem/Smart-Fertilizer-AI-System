# preprocess.py

import pandas as pd
import numpy as np
import os
import joblib

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler


def load_data(path="Fertilizer Prediction.csv"):
    print("📂 Loading dataset...")
    df = pd.read_csv(path)
    print("✅ Dataset loaded successfully!")
    return df


def preprocess_data(df):
    print("🧹 Starting preprocessing...")

    # Remove duplicates
    df = df.drop_duplicates()

    # Handle missing values
    df = df.fillna(df.mode().iloc[0])

    # Separate features and target
    X = df.drop("Fertilizer Name", axis=1)
    y = df["Fertilizer Name"]

    # Identify categorical columns
    categorical_cols = X.select_dtypes(include=['object']).columns
    numerical_cols = X.select_dtypes(include=['int64', 'float64']).columns

    # Encode categorical columns
    encoders = {}
    for col in categorical_cols:
        le = LabelEncoder()
        X[col] = le.fit_transform(X[col])
        encoders[col] = le

    # Encode target variable
    target_encoder = LabelEncoder()
    y = target_encoder.fit_transform(y)

    # Scale numerical columns
    scaler = StandardScaler()
    X[numerical_cols] = scaler.fit_transform(X[numerical_cols])

    # Save encoders and scaler
    joblib.dump(encoders, "encoders.pkl")
    joblib.dump(target_encoder, "target_encoder.pkl")
    joblib.dump(scaler, "scaler.pkl")

    print("💾 Encoders and scaler saved!")

    # Train-test split
    X_train, X_test, y_train, y_test = train_test_split(
        X, y,
        test_size=0.2,
        random_state=42,
        stratify=y
    )

    print("✅ Preprocessing completed successfully!")

    return X_train, X_test, y_train, y_test


if __name__ == "__main__":
    df = load_data()
    X_train, X_test, y_train, y_test = preprocess_data(df)

    print(f"📊 Training samples: {X_train.shape}")
    print(f"📊 Testing samples: {X_test.shape}")