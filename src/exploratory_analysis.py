import pandas as pd
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
CLEAN = ROOT / "data" / "clean"

customers = pd.read_csv(CLEAN / "customers.csv")
products = pd.read_csv(CLEAN / "products.csv")
orders = pd.read_csv(CLEAN / "orders.csv")

completed_orders = orders[orders["status"] == "completed"].copy()
print("\nCOMPLETED ORDERS:", len(completed_orders))

# 1. Dataset information
print("\nCUSTOMERS INFO")
customers.info()

print("\nPRODUCTS INFO")
products.info()

print("\nORDERS INFO")
orders.info()

# 2. Statistical summaries
print("\nCUSTOMERS SUMMARY")
print(customers.describe(include="all"))

print("\nPRODUCTS SUMMARY")
print(products.describe(include="all"))

print("\nORDERS SUMMARY")
print(orders.describe(include="all"))
# 3. Order status distribution
print("\nORDER STATUS COUNTS")
print(orders["status"].value_counts())

# 4. Delivery time availability by order status
print("\nDELIVERY TIME BY STATUS")
print(
    orders.groupby("status")["delivery_minutes"]
    .agg(["count", "mean", "median"])
)

# 5. Revenue by city
customer_orders = completed_orders.merge(
    customers[["customer_id", "city"]],
    on="customer_id",
    how="left"
)

print("\nREVENUE BY CITY — COMPLETED ORDERS")

city_summary = customer_orders.groupby("city").agg(
    total_revenue=("order_amount", "sum"),
    average_order_value=("order_amount", "mean"),
    order_count=("order_amount", "count")
).round(2)

print(city_summary)
# 6. Completed-order revenue by city


city_revenue = (
    customer_orders.groupby("city")["order_amount"]
    .sum()
    .sort_values(ascending=False)
)


# 7. Revenue by product category


completed_product_orders = completed_orders.merge(
    products[["product_id", "category"]],
    on="product_id",
    how="left"
)

category_revenue = (
    completed_product_orders.groupby("category")["order_amount"]
    .sum()
    .sort_values(ascending=False)
)


print("\nCOMPLETED ORDER REVENUE BY CATEGORY")
print(category_revenue.round(2))

# 8. Delivery time distribution

delivery_data = orders.loc[
    orders["status"] == "completed",
    "delivery_minutes"
].dropna()

print("\nDELIVERY TIME SUMMARY")
print(delivery_data.describe())


# 9. Delivery time outlier analysis

Q1 = delivery_data.quantile(0.25)
Q3 = delivery_data.quantile(0.75)

IQR = Q3 - Q1

lower_bound = Q1 - 1.5 * IQR
upper_bound = Q3 + 1.5 * IQR

outliers = delivery_data[
    (delivery_data < lower_bound) |
    (delivery_data > upper_bound)
]

print("\nDELIVERY TIME OUTLIER ANALYSIS")
print("Q1:", Q1)
print("Q3:", Q3)
print("IQR:", IQR)
print("Lower Bound:", lower_bound)
print("Upper Bound:", upper_bound)
print("Number of Outliers:", len(outliers))
print("Percentage of outliers:", round(len(outliers) / len(delivery_data) * 100, 2), "%")
print("Minimum outlier:", outliers.min())
print("Maximum outlier:", outliers.max())


# Pandas cross-check of SQL Q2
completed_city_category = completed_orders.merge(
    customers[["customer_id", "city"]],
    on="customer_id",
    how="inner"
).merge(
    products[["product_id", "category"]],
    on="product_id",
    how="inner"
)

revenue_pivot = pd.pivot_table(
    completed_city_category,
    index="city",
    columns="category",
    values="order_amount",
    aggfunc="sum",
    margins=True,
    margins_name="Total"
)

print("\nPANDAS REVENUE PIVOT TABLE")
print(revenue_pivot.round(2))

print(
    "Pandas grand total:",
    round(revenue_pivot.loc["Total", "Total"], 2)
)
print("\nREVENUE RECONCILIATION")

print("Total completed-order revenue:",
      round(completed_orders["order_amount"].sum(), 2))

print("City revenue total:",
      round(city_revenue.sum(), 2))

print("Category revenue total:",
      round(category_revenue.sum(), 2))
# Completed-order amount statistical analysis
amounts = completed_orders["order_amount"]

q1 = amounts.quantile(0.25)
q3 = amounts.quantile(0.75)
iqr = q3 - q1

lower_fence = q1 - 1.5 * iqr
upper_fence = q3 + 1.5 * iqr

print("\nCOMPLETED ORDER AMOUNT ANALYSIS")
print("Mean:", round(amounts.mean(), 2))
print("Median:", round(amounts.median(), 2))
print("Skewness:", round(amounts.skew(), 2))
print("Q1:", round(q1, 2))
print("Q3:", round(q3, 2))
print("Lower fence:", round(lower_fence, 2))
print("Upper fence:", round(upper_fence, 2))
print("Outliers below lower fence:", (amounts < lower_fence).sum())
print("Outliers above upper fence:", (amounts > upper_fence).sum())
# Correlation analysis for completed orders
correlation_columns = [
    "order_amount",
    "discount_pct",
    "quantity",
    "delivery_minutes"
]

correlation_matrix = completed_orders[
    correlation_columns
].corr()

print("\nCOMPLETED ORDER CORRELATION MATRIX")
print(correlation_matrix.round(3))