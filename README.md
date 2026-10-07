# Payment Risk Prediction

> End-to-end MLOps solution for predicting customer payment risk

## Overview

This project builds a simple machine learning solution to analyze customer behaviour and categorize the future payment risk of the customer using synthetic .

The workflow is:

**Synthetic Data → PySpark Data Processing → Feature Engineering → Model Training → MLflow → Model Artefacts → FastAPI**

PySpark is used for data processing and feature engineering. A Random Forest classifier is trained using scikit-learn, while MLflow tracks the experiment and model metrics.

The trained model and encoder are saved as `model.pkl` and `encoder.pkl`. These artefacts are then loaded by FastAPI to provide predictions.

## How it works

- Synthetic customer payment data is generated and saved as an csv file.
- PySpark cleans the data and performs data quality checks.
- Features such as utilisation ratio, income band and missed payment percentage are created and saved as the  final csv.
- A Random Forest classifier is trained and evaluated.
- MLflow tracks the training run and metrics.
- The trained model and encoder are saved as model artefacts.
- FastAPI loads the artefacts and exposes the model through an API.
- Docker can be used to package and run the prediction service.
- Unit tests were executed for data cleaning, feature engineering and model output.

## How to run

The project requires **Python 3.12**,  **Java/JDK 17** and **Pyspark 4.2.0** for local PySpark execution.

Configure JAVA_HOME and SPARK_HOME in the system environment variables and check 'spark-shell' to verify pyspark installation

Install the dependencies from `requirements.txt`.

Run the pipeline in this order:

**Synthetic Data → Data Processing → Feature Engineering → Model Training**

This produces `model.pkl` and `encoder.pkl`.

MLflow can then be used to view the experiment and metrics.

Start the FastAPI application to serve predictions through the `/predict` endpoint. A `/health` endpoint is also available for checking the service.

The FastAPI application will be packaged and run using Docker.


## FastAPI

### Start the API

```bash
uvicorn app:app --reload
```

### Input

Send a `POST` request to `/predict`:

```json
{
 "account_number": 1243704,
  "age": 46,
  "income": 64181,
  "employment_status": "salaried",
  "balance": 608,
  "credit_limit": 6500,
  "monthly_payment": 123,
  "num_missed_payments_6m": 6,
  "consecutive_payments_on_time": 2,
  "utilization_ratio": 0.0935,
  "monthly_income": 5348.42,
  "income_band": "high",
  "missed_payment_percentage": 100.0
}
```

### Output

```json
{
  "payment_risk": "low"
}
```

### API UI

Open the Swagger UI:

```text
http://127.0.0.1:8000/docs
```

Use `/predict` → **Try it out** → enter the input → **Execute**.

## Docker

Build the image:

```bash
docker build -t payment-risk-api .
```

Run the container:

```bash
docker run -p 8000:8000 payment-risk-api
```

Then open:

```text
http://localhost:8000/docs
```

Use the `/predict` endpoint to test a prediction.
