
import warnings
warnings.filterwarnings("ignore")

from pyspark.sql import SparkSession
from pyspark.sql.functions import col, when

spark = SparkSession.builder.appName("CustomerPaymentProcessing").master("local[*]").getOrCreate()

df = spark.read.csv("cleaned_customers_features.csv",header=True,inferSchema=True)

print("Columns:", df.columns)
df.printSchema()
df.show(5)
print()

#Feature engineering

df = df.withColumn(
    "utilization_ratio",
    col("balance") / col("credit_limit"))

df = df.withColumn(
    "monthly_income",
    col("income") / 12)

df = df.withColumn(
    "income_band",
    when(col("income") < 30000, "low")
    .when(col("income") < 50000, "medium")
    .otherwise("high"))

df = df.withColumn(
    "missed_payment_percentage",
    (col("num_missed_payments_6m") / 6) * 100)

# Target column creation based on the number of missed payments in the last 6 months
# 0-2 missed payments = low
# 3-4 missed payments = medium
# 5-6 missed payments = high

df = df.withColumn(
    "payment_risk",
    when(
        col("num_missed_payments_6m") <= 2,
        "low")
    .when(
        col("num_missed_payments_6m") <= 4,
        "medium")
    .otherwise(
        "high"))


# Testing for the presence of expected columns and valid values in categorical columns

expected_columns = [
    "utilization_ratio",
    "monthly_income",
    "income_band",
    "missed_payment_percentage",
    "payment_risk"]

for column in expected_columns:
    assert column in df.columns


for column in ["income_band", "payment_risk"]:
    assert set(
        row[column] for row in df.select(column).distinct().collect()
    ).issubset({"low", "medium", "high"})

df.toPandas().to_csv("final_data.csv", index=False)