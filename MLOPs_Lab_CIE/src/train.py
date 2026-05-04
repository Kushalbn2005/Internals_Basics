import pandas as pd
import numpy as np
import mlflow
import mlflow.sklearn
from sklearn.model_selection import train_test_split
from sklearn.linear_model import Lasso
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import json
import os
import joblib

# Load dataset
df = pd.read_csv("data/training_data.csv")

X = df.drop("cooling_power_kw", axis=1)
y = df["cooling_power_kw"]

# REQUIRED split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# Set MLflow experiment
mlflow.set_experiment("rackcool-cooling-power-kw")

results = []

def train_and_log(name, model, params):
    with mlflow.start_run():

        # Log params
        for key, value in params.items():
            mlflow.log_param(key, value)

        # Train
        model.fit(X_train, y_train)

        # Predict
        preds = model.predict(X_test)

        # Metrics
        mae = mean_absolute_error(y_test, preds)
        rmse = np.sqrt(mean_squared_error(y_test, preds))
        r2 = r2_score(y_test, preds)

        # Log metrics
        mlflow.log_metric("mae", mae)
        mlflow.log_metric("rmse", rmse)
        mlflow.log_metric("r2", r2)

        # Tag
        mlflow.set_tag("domain", "data_center")

        # Log model
        mlflow.sklearn.log_model(model, name)

        return {
            "name": name,
            "mae": mae,
            "rmse": rmse,
            "r2": r2
        }, model


# 🔹 Train Lasso
lasso = Lasso(alpha=0.1)
lasso_res, lasso_model = train_and_log(
    "Lasso",
    lasso,
    {"alpha": 0.1}
)

# 🔹 Train RandomForest
rf = RandomForestRegressor(n_estimators=100, random_state=42)
rf_res, rf_model = train_and_log(
    "RandomForest",
    rf,
    {"n_estimators": 100, "random_state": 42}
)

results.extend([lasso_res, rf_res])

# Select best model (LOWEST MAE)
best = min(results, key=lambda x: x["mae"])

# Save best model
os.makedirs("models", exist_ok=True)
if best["name"] == "Lasso":
    joblib.dump(lasso_model, "models/model.pkl")
else:
    joblib.dump(rf_model, "models/model.pkl")

# Create JSON output
output = {
    "experiment_name": "rackcool-cooling-power-kw",
    "models": results,
    "best_model": best["name"],
    "best_metric_name": "mae",
    "best_metric_value": best["mae"]
}

os.makedirs("results", exist_ok=True)
with open("results/step1_s1.json", "w") as f:
    json.dump(output, f, indent=4)

print("✅ Task 1 completed successfully!")