# app.py

import streamlit as st
import pandas as pd
import numpy as np
import joblib

# Load saved preprocessors once
encoder = joblib.load('models/encoder.pkl')
imputer = joblib.load('models/imputer.pkl')
scaler = joblib.load('models/scaler.pkl')

# Load list of unique categorical options for UI (for example)
df = pd.read_csv("output.csv")

def get_unique_values(column):
    return sorted(df[column].dropna().astype(str).unique())

genre_options = get_unique_values("genre")
director_options = get_unique_values("director")
writer_options = get_unique_values("writer")
star_options = get_unique_values("star")
country_options = get_unique_values("country")
company_options = get_unique_values("company")
rating_options = get_unique_values("rating")

# Load selected model on demand
@st.cache_resource
def load_model(model_name):
    return joblib.load(f"models/{model_name}.pkl")

def preprocess_input(input_data):
    df_input = pd.DataFrame([input_data])

    # Feature engineering & log transforms
    df_input["log_budget"] = np.log1p(df_input["budget"])

    df_input["budget_vote_ratio"] = df_input["budget"] / (df_input["votes"] + 1)
    df_input["budget_runtime_ratio"] = df_input["budget"] / (df_input["runtime"] + 1)
    df_input["budget_score_ratio"] = df_input["log_budget"] / (df_input["score"] + 1)
    df_input["vote_score_ratio"] = df_input["votes"] / (df_input["score"] + 1)
    df_input["budget_year_ratio"] = df_input["log_budget"] / (df_input["year"] - df_input["year"].min() + 1)
    df_input["vote_year_ratio"] = df_input["votes"] / (df_input["year"] - df_input["year"].min() + 1)
    df_input["score_runtime_ratio"] = df_input["score"] / (df_input["runtime"] + 1)
    df_input["budget_per_minute"] = df_input["budget"] / (df_input["runtime"] + 1)
    df_input["votes_per_year"] = df_input["votes"] / (df_input["year"] - df_input["year"].min() + 1)
    df_input["is_recent"] = (df_input["year"] >= df_input["year"].quantile(0.75)).astype(int)
    df_input["is_high_budget"] = (df_input["log_budget"] >= df_input["log_budget"].quantile(0.75)).astype(int)
    df_input["is_high_votes"] = (df_input["votes"] >= df_input["votes"].quantile(0.75)).astype(int)
    df_input["is_high_score"] = (df_input["score"] >= df_input["score"].quantile(0.75)).astype(int)

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

    # Handle missing categorical values if any
    df_input[categorical_features] = df_input[categorical_features].fillna('missing').astype(str)

    # Encode categorical
    df_input[categorical_features] = encoder.transform(df_input[categorical_features])

    # Impute + scale numerical
    df_input[numerical_features] = imputer.transform(df_input[numerical_features])
    df_input[numerical_features] = scaler.transform(df_input[numerical_features])

    # Drop original budget column
    df_input = df_input.drop(columns=["budget"])

    return df_input

def predict_revenue(input_data, model):
    processed = preprocess_input(input_data)
    # Align columns with model if necessary
    expected_features = model.feature_names_in_
    for feature in expected_features:
        if feature not in processed.columns:
            processed[feature] = 0
    processed = processed[expected_features]

    log_pred = model.predict(processed)
    prediction = np.exp(log_pred) - 1
    return prediction[0]

def revenue_range(gross):
    if gross <= 10_000_000:
        return "Low Revenue (≤ $10M)"
    elif gross <= 40_000_000:
        return "Medium-Low Revenue ($10M - $40M)"
    elif gross <= 70_000_000:
        return "Medium Revenue ($40M - $70M)"
    elif gross <= 120_000_000:
        return "Medium-High Revenue ($70M - $120M)"
    elif gross <= 200_000_000:
        return "High Revenue ($120M - $200M)"
    else:
        return "Ultra High Revenue (≥ $200M)"

# UI
st.title("🎬 Movie Revenue Prediction")

model_choice = st.selectbox(
    "Choose a Model",
    ["best_decision_tree", "best_decision_tree_bagging", "best_linear_reg",
     "best_random_forest", "best_xgb_model", "best_gradient_boost"]
)

model = load_model(model_choice)

with st.form("movie_features_form"):
    col1, col2 = st.columns(2)

    with col1:
        released = st.text_input("Release Date (e.g. June 13, 1980)", "June 13, 1980")
        writer = st.selectbox("Writer", writer_options)
        rating = st.selectbox("MPAA Rating", rating_options)
        genre = st.selectbox("Genre", genre_options)
        director = st.selectbox("Director", director_options)

    with col2:
        star = st.selectbox("Leading Star", star_options)
        country = st.selectbox("Country", country_options)
        company = st.selectbox("Production Company", company_options)
        runtime = st.number_input("Runtime (minutes)", min_value=1.0, value=100.0)
        score = st.number_input("IMDb Score", min_value=0.0, max_value=10.0, value=7.0)
        budget = st.number_input("Budget ($)", min_value=1.0, value=10_000_000.0)
        year = st.number_input("Release Year", min_value=1900, max_value=2100, value=2000)
        votes = st.number_input("Initial Votes", min_value=0, value=100000)

    submit = st.form_submit_button("Predict Revenue")

if submit:
    input_data = {
        "released": released,
        "writer": writer,
        "rating": rating,
        "genre": genre,
        "director": director,
        "star": star,
        "country": country,
        "company": company,
        "runtime": runtime,
        "score": score,
        "budget": budget,
        "year": year,
        "votes": votes
    }

    pred = predict_revenue(input_data, model)
    range_label = revenue_range(pred)

    st.success(f"Predicted Revenue: ${pred:,.0f}")
    st.info(f"Revenue Category: {range_label}")
