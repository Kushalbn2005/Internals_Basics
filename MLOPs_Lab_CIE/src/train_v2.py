import pandas as pd
import numpy as np
import mlflow
import mlflow.sklearn
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error

df = pd.read_csv("data/training_data.csv")

X = df.drop("cooling_power_kw", axis=1)
y = df["cooling_power_kw"]

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

mlflow.set_experiment("rackcool-cooling-power-kw")

with mlflow.start_run():

    model = RandomForestRegressor(n_estimators=100, random_state=99)
    model.fit(X_train, y_train)

    preds = model.predict(X_test)
    mae = mean_absolute_error(y_test, preds)

    mlflow.log_param("random_state", 99)
    mlflow.log_metric("mae", mae)
    mlflow.set_tag("domain", "data_center")

    mlflow.sklearn.log_model(model, "RandomForest")

    print("MAE:", mae)