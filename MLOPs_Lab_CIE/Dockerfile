FROM python:3.10-slim

WORKDIR /app

COPY . .

RUN pip install pandas numpy scikit-learn joblib

ENTRYPOINT ["python", "src/predict_cli.py"]