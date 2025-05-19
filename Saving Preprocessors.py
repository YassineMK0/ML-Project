# train_and_save_preprocessors.py

import pandas as pd
import numpy as np
from sklearn.preprocessing import OrdinalEncoder
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
import joblib

# Load your full training dataset
df = pd.read_csv("output.csv")

# Feature engineering (same as your code)
df["log_budget"] = np.log1p(df["budget"])

df["budget_vote_ratio"] = df["budget"] / (df["votes"] + 1)
df["budget_runtime_ratio"] = df["budget"] / (df["runtime"] + 1)
df["budget_score_ratio"] = df["log_budget"] / (df["score"] + 1)
df["vote_score_ratio"] = df["votes"] / (df["score"] + 1)
df["budget_year_ratio"] = df["log_budget"] / (df["year"] - df["year"].min() + 1)
df["vote_year_ratio"] = df["votes"] / (df["year"] - df["year"].min() + 1)
df["score_runtime_ratio"] = df["score"] / (df["runtime"] + 1)
df["budget_per_minute"] = df["budget"] / (df["runtime"] + 1)
df["votes_per_year"] = df["votes"] / (df["year"] - df["year"].min() + 1)
df["is_recent"] = (df["year"] >= df["year"].quantile(0.75)).astype(int)
df["is_high_budget"] = (df["log_budget"] >= df["log_budget"].quantile(0.75)).astype(int)
df["is_high_votes"] = (df["votes"] >= df["votes"].quantile(0.75)).astype(int)
df["is_high_score"] = (df["score"] >= df["score"].quantile(0.75)).astype(int)

categorical_features = [
    "released", "writer", "rating", "genre",
    "director", "star", "country", "company"
]

numerical_features = [
    "runtime", "score", "year", "votes", "log_budget",
    "budget_vote_ratio", "budget_runtime_ratio", "budget_score_ratio",
    "vote_score_ratio", "budget_year_ratio", "vote_year_ratio",
    "score_runtime_ratio", "budget_per_minute", "votes_per_year",
    "is_recent", "is_high_budget", "is_high_votes", "is_high_score"
]

# Fill any missing categorical values with string 'missing'
df[categorical_features] = df[categorical_features].fillna('missing').astype(str)

# Fit OrdinalEncoder on categorical features
encoder = OrdinalEncoder(handle_unknown='use_encoded_value', unknown_value=-1)
encoder.fit(df[categorical_features])

# Impute and scale numerical features
imputer = SimpleImputer(strategy='median')
imputer.fit(df[numerical_features])

scaler = StandardScaler()
scaler.fit(df[numerical_features])

# Save the preprocessors
joblib.dump(encoder, 'models/encoder.pkl')
joblib.dump(imputer, 'models/imputer.pkl')
joblib.dump(scaler, 'models/scaler.pkl')

print("Encoders and scalers saved successfully.")
