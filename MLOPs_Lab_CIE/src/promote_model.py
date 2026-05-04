from mlflow.tracking import MlflowClient

client = MlflowClient()

model_name = "rackcool-cooling-power-kw-predictor"

# CHANGE based on comparison
best_version = 1   # or 2

client.set_registered_model_alias(
    name=model_name,
    alias="production",
    version=best_version
)

print(f"Production set to version {best_version}")