import argparse
import joblib
import numpy as np

# Argument parser
parser = argparse.ArgumentParser()

parser.add_argument("--server_rack_count", type=float, required=True)
parser.add_argument("--ambient_temp_c", type=float, required=True)
parser.add_argument("--compute_load_pct", type=float, required=True)
parser.add_argument("--is_liquid_cooled", type=float, required=True)

args = parser.parse_args()

# Load model
model = joblib.load("models/model.pkl")

# Prepare input
features = np.array([[ 
    args.server_rack_count,
    args.ambient_temp_c,
    args.compute_load_pct,
    args.is_liquid_cooled
]])

# Predict
prediction = model.predict(features)[0]

print(prediction)