import mlflow
from mlflow.tracking import MlflowClient

# 🔴 IMPORTANT: replace with your actual run_id
RUN_ID = "685fa28bfa274e60bcc9fd60457cc719"

MODEL_NAME = "rackcool-cooling-power-kw-predictor"

# Model URI
model_uri = f"runs:/{RUN_ID}/Lasso"

# Register model
result = mlflow.register_model(model_uri, MODEL_NAME)

print("✅ Model registered!")
print("Version:", result.version)