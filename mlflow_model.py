import pandas as pd
import joblib
import mlflow
import mlflow.sklearn
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OrdinalEncoder
from sklearn.metrics import (classification_report, confusion_matrix, accuracy_score)


mlflow.set_experiment("Payment Risk Prediction")
mlflow.sklearn.autolog()

df = pd.read_csv("final_data.csv")
print(df.head())
print()

categorical_cols = ["employment_status", "income_band"]
col_categories = [
    ["retired", "self employed", "salaried"],
    ["low", "medium", "high"]]

# Encoding categorical features using OrdinalEncoder

encoder = OrdinalEncoder(categories=col_categories)
df[categorical_cols] = encoder.fit_transform(df[categorical_cols])

X = df.drop(columns=["payment_risk"])
y = df["payment_risk"]

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.3, random_state=42, stratify=y
)

with mlflow.start_run(run_name="RandomForest_payment_risk") as run:
    model = RandomForestClassifier(
        n_estimators=250,
        max_depth=6,
        random_state=42
    )
    model.fit(X_train, y_train)

    # Registering the model with MLflow
    mlflow.sklearn.log_model(model, name="RandomForest_payment_risk_model",
    registered_model_name="RandomForest_payment_risk_model",
    skops_trusted_types=["sklearn.tree._tree.Tree"],
)

    # Saving the model and encoder for future use as pickle files
    joblib.dump(model, "model.pkl")
    joblib.dump(encoder, "encoder.pkl")

    # Predictions
    y_train_pred = model.predict(X_train)
    y_pred = model.predict(X_test)

    # Unit testing for model training and predictions

    assert model is not None

    assert len(model.estimators_) == 250

    assert len(y_pred) == len(y_test)

    assert set(y_pred).issubset({"low", "medium", "high"})


    # evaluating the model and logging metrics to MLflow
    train_acc = accuracy_score(y_train, y_train_pred)
    test_acc = accuracy_score(y_test, y_pred)
    
   
    mlflow.log_metric("train_accuracy", train_acc)
    mlflow.log_metric("test_accuracy", test_acc)

    confusion = confusion_matrix(y_test, y_pred)
    print("Confusion Matrix:")
    print(confusion)

    feature_importance = pd.DataFrame({
        "feature": X.columns,
        "importance": model.feature_importances_})

    print("Feature Importance:")
    print(feature_importance)

    print("training accuracy:", train_acc)
    print("test accuracy:", test_acc)

    print("Classification Report:")
    print(classification_report(y_test, y_pred))