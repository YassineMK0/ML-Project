import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.impute import SimpleImputer


def preprocess_data(df):
    df = df.copy()

    # Log Transformation
    if "gross" in df.columns:
        df["log_gross"] = np.log1p(df["gross"])
    df["log_budget"] = np.log1p(df["budget"])

    # Feature engineering
    """"The code creates several derived features:

Budget-related ratios: budget_vote_ratio, budget_runtime_ratio, budget_score_ratio, etc.
Time-related features: budget_year_ratio, vote_year_ratio, votes_per_year
Per-minute metrics: budget_per_minute
Binary indicators: is_recent, is_high_budget, is_high_votes, is_high_score"""
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
    df["is_high_budget"] = (df["log_budget"] >= df["log_budget"].quantile(0.75)).astype(
        int
    )
    df["is_high_votes"] = (df["votes"] >= df["votes"].quantile(0.75)).astype(int)
    df["is_high_score"] = (df["score"] >= df["score"].quantile(0.75)).astype(int)

    categorical_features = [
        "released",
        "writer",
        "rating",
        "genre",
        "director",
        "star",
        "country",
        "company",
    ]

    for feature in categorical_features:
        df[feature] = df[feature].astype(str)
        le = LabelEncoder() #Converts categorical text data into numeric form using LabelEncoder
        df[feature] = le.fit_transform(df[feature])

    numerical_features = [
        "runtime",
        "score",
        "year",
        "votes",
        "log_budget",
        "budget_vote_ratio",
        "budget_runtime_ratio",
        "budget_score_ratio",
        "vote_score_ratio",
        "budget_year_ratio",
        "vote_year_ratio",
        "score_runtime_ratio",
        "budget_per_minute",
        "votes_per_year",
        "is_recent",
        "is_high_budget",
        "is_high_votes",
        "is_high_score",
    ]

    imputer = SimpleImputer(strategy="median")#Handling Missing Values
    df[numerical_features] = imputer.fit_transform(df[numerical_features])

    scaler = StandardScaler()
    df[numerical_features] = scaler.fit_transform(df[numerical_features])#Standardizes all numerical features to have zero mean and unit variance
                                                                        #This is important for many machine learning algorithms that are sensitive to feature scales
#Removes original monetary columns since they're replaced by their log-transformed versions
    if "gross" in df.columns:
        df = df.drop(["gross", "budget"], axis=1)
    else:
        df = df.drop(["budget"], axis=1)

    return df
"""Calls the preprocessing function to get the transformed dataframe
Separates features (X) from the target variable (y), which is "log_gross" if present
Returns these as separate objects ready for model training"""

def prepare_features(df):
    processed_df = preprocess_data(df)

    if "log_gross" in processed_df.columns:
        y = processed_df["log_gross"]
        X = processed_df.drop("log_gross", axis=1)
    else:
        y = None
        X = processed_df

    return X, y
