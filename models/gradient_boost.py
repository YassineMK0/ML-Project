import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.metrics import r2_score, mean_squared_error, mean_squared_log_error
from feature_scaling import prepare_features
import pickle
# Loading our dataset
df = pd.read_csv("output.csv")

# Getting the Preprocessed and scaled data.
X, y = prepare_features(df)
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

param_grid = {
    "n_estimators": [100, 500],
    "max_depth": [3, 6],
    "learning_rate": [0.05, 0.1],
}

grid_search = GridSearchCV(
    estimator=GradientBoostingRegressor(loss="squared_error", random_state=42),
    param_grid=param_grid,
    cv=5,
    scoring="neg_mean_squared_error",
    n_jobs=-1,
)
grid_search.fit(X, y)

best_params = grid_search.best_params_
best_score = grid_search.best_score_
print("Best Parameters:", best_params)
print("Best MSE Score:", best_score)

best_model = GradientBoostingRegressor(
    loss="squared_error", random_state=42, **best_params
)
best_model.fit(X_train, y_train)

train_predictions = best_model.predict(X_train)
test_predictions = best_model.predict(X_test)


def calculate_metrics(y_true, y_pred):
    r2 = r2_score(y_true, y_pred)
    mse = mean_squared_error(y_true, y_pred)
    msle = mean_squared_log_error(y_true, y_pred)
    mape = np.mean(np.abs((np.exp(y_true) - np.exp(y_pred)) / np.exp(y_true))) * 100
    return r2, mse, msle, mape


train_r2, train_mse, train_msle, train_mape = calculate_metrics(
    y_train, train_predictions
)
test_r2, test_mse, test_msle, test_mape = calculate_metrics(y_test, test_predictions)

print(f"\nTraining Metrics:")
print(f"R2 score: {train_r2:.4f}")
print(f"MSE: {train_mse:.4f}")
print(f"MLSE: {train_msle:.4f}")
print(f"MAPE: {train_mape:.2f}%")

print(f"\nTest Metrics:")
print(f"R2 score: {test_r2:.4f}")
print(f"MSE: {test_mse:.4f}")
print(f"MSLE: {test_msle:.4f}")
print(f"MAPE: {test_mape:.2f}%")
with open("best_gradient_boost.pkl", "wb") as file:
    pickle.dump(best_model, file)
