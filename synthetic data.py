import numpy as np
import pandas as pd

np.random.seed(42)
n = 2000

# Customer information
account_number = np.random.choice(np.arange(1234567, 7654321),
    size=n, replace=False)

age = np.random.randint(21, 61, n)

income = np.random.randint(24000, 80001, n)

employment_status = np.random.choice(
    ["salaried", "self employed", "retired"],n)


balance = np.random.randint(500, 4001, n)

credit_limit = np.random.choice(
    np.arange(1000, 8001, 500),n)

monthly_payment = np.random.randint(50, 401, n)

num_missed_payments_6m = np.random.randint(0,7,n)

consecutive_payments_on_time = np.random.randint(0,7,n)


# Creating dataframe
df = pd.DataFrame({
    "account_number": account_number,
    "age": age,
    "income": income,
    "employment_status": employment_status,
    "balance": balance,
    "credit_limit": credit_limit,
    "monthly_payment": monthly_payment,
    "num_missed_payments_6m": num_missed_payments_6m,
    "consecutive_payments_on_time": consecutive_payments_on_time
})

# Saving the dataframe to a CSV file
df.to_csv("customers_payments.csv",index=False)
print()
print(df.head())
print()
print("Dataset shape:", df.shape)