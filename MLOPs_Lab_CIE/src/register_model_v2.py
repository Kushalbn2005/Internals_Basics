import mlflow

# 🔴 Replace with your NEW run ID (from train_v2.py)
RUN_ID = "9ab24ffd778248469879256510c28b6e"

MODEL_NAME = "rackcool-cooling-power-kw-predictor"

# Register as new version
result = mlflow.register_model(
    f"runs:/{RUN_ID}/RandomForest",
    MODEL_NAME
)

print("✅ Version 2 registered!")
print("New Version:", result.version)