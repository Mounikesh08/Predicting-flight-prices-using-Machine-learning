# ------------------- model/train_model.py -------------------
from utils.db_connection import get_flight_data
from sklearn.ensemble import RandomForestRegressor
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.model_selection import train_test_split
import pandas as pd
import joblib
from datetime import datetime
import os
import numpy as np

fare_classes = ["Economy", "PremiumEconomy", "Business", "First"]

# Function to scale price increase into a forecast range
# Base increase from 10% to 45% (0.10 to 0.45), holidays push this higher

def forecast_increase(pct_change, holiday_flag):
    base = np.clip(pct_change, 0.10, 0.45)
    if holiday_flag >= 1:
        return np.clip(base + 0.10, 0.10, 0.60)  # bump up for holidays
    return base

def compute_optimal_price(base_price, forecasted_increase):
    return round(base_price * (1 + forecasted_increase), 2)

def train_model():
    df = get_flight_data()
    if df.empty:
        print(" No data available to train the model.")
        return

    df["Departure_Date"] = pd.to_datetime(df["Departure_Date"])
    df["days_until_departure"] = (df["Departure_Date"] - pd.to_datetime("today")).dt.days
    df["month"] = df["Departure_Date"].dt.month
    df["year"] = df["Departure_Date"].dt.year
    df["day_of_week"] = df["Departure_Date"].dt.dayofweek
    df["holiday_flag"] = df[["is_origin_holiday", "is_destination_holiday", "is_holiday_route"]].fillna(0).sum(axis=1)

    # Add jitter to reduce ties
    df["days_until_departure"] += np.random.randint(-3, 4, size=len(df))

    # Create new feature: % price increase compared to average for that route & month
    for fare in fare_classes:
        df[f"avg_{fare}_per_month"] = df.groupby(["Origin", "Destination", "month"])[fare].transform("mean")
        df[f"pct_change_{fare}"] = ((df[fare] - df[f"avg_{fare}_per_month"]) / df[f"avg_{fare}_per_month"]).fillna(0)
        df[f"forecasted_{fare}_increase"] = df.apply(lambda row: forecast_increase(row[f"pct_change_{fare}"], row["holiday_flag"]), axis=1)
        df[f"optimal_{fare}_price"] = df.apply(lambda row: compute_optimal_price(row[f"avg_{fare}_per_month"], row[f"forecasted_{fare}_increase"]), axis=1)

    model_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "model"))
    os.makedirs(model_dir, exist_ok=True)

    for fare in fare_classes:
        if fare not in df.columns:
            print(f" Column {fare} not found in data.")
            continue

        df_fare = df.dropna(subset=[fare, f"forecasted_{fare}_increase", f"optimal_{fare}_price"])
        if df_fare.empty:
            print(f" No data to train for fare class: {fare}")
            continue

        X = df_fare[["days_until_departure", "holiday_flag", "month", "year", "day_of_week"]]
        y = df_fare[f"forecasted_{fare}_increase"]

        if y.nunique() < 5:
            print(f" Not enough variance in {fare} forecast values to build a predictive model.")
            continue

        numeric_features = ["days_until_departure", "holiday_flag", "month", "year", "day_of_week"]
        numeric_transformer = Pipeline(steps=[
            ("imputer", SimpleImputer(strategy="mean")),
            ("scaler", StandardScaler())
        ])

        preprocessor = ColumnTransformer(transformers=[
            ("num", numeric_transformer, numeric_features)
        ])

        pipeline = Pipeline(steps=[
            ("preprocessor", preprocessor),
            ("regressor", RandomForestRegressor(n_estimators=100, random_state=42))
        ])

        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
        pipeline.fit(X_train, y_train)

        model_file = os.path.join(model_dir, f"{fare}_model.pkl")
        joblib.dump(pipeline, model_file)
        print(f" Trained and saved model for: {fare} → {model_file}")

if __name__ == "__main__":
    train_model()
