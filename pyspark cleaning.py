from pyspark.sql import SparkSession
from pyspark.sql.functions import col, when

spark = SparkSession.builder.appName("CustomerPaymentProcessing").master("local[*]").getOrCreate()

df = spark.read.csv(
    "customers_payments.csv",
    header=True,
    inferSchema=True
)



print("Columns:", df.columns)
df.printSchema()
df.show(5)

# Data quality checks

print("DATA QUALITY CHECKS")
print("Missing values:")

for column in df.columns:
    missing = df.filter(col(column).isNull()).count()
    print(column, ":", missing)


duplicate_rows = (df.count()- df.dropDuplicates().count())
print("Duplicate rows:", duplicate_rows)


invalid_income = df.filter(col("income") <= 0).count()
print("Invalid income rows:", invalid_income)


# Data consistency checks

print("DATA CONSISTENCY CHECKS")

invalid_balance_limit = df.filter(col("balance") > col("credit_limit")).count()

print("Balance greater than credit limit:",invalid_balance_limit)

df_clean = df.filter(
    col("balance") <= col("credit_limit")
)

#df_clean.toPandas().to_csv("cleaned_customers_features.csv", index=False)

# Unit testing
assert df.filter(col("income") <= 0).count() >= 0
assert df_clean.filter(col("balance") > col("credit_limit")).count() == 0