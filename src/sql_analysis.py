import pandas as pd
import sqlite3
from pathlib import Path

# Project root directory
ROOT = Path(__file__).resolve().parents[1]

# SQLite database path
DB = ROOT / "data" / "instamart.db"

# SQL queries file path
SQL_FILE = ROOT / "sql" / "queries.sql"
# Read the SQL queries from the file
with open(SQL_FILE, "r", encoding="utf-8") as file:
    sql_script = file.read()

# Separate the seven queries
queries = [
    query.strip()
    for query in sql_script.split(";")
    if query.strip()
]

print("Total SQL queries loaded:", len(queries))
# Execute SQL Query 1
with sqlite3.connect(DB) as conn:
    q1_result = pd.read_sql_query(queries[0], conn)

print("\nQ1 — WHERE then HAVING")
print(q1_result)
print("Number of customers:", len(q1_result))
# Execute SQL Query 2
with sqlite3.connect(DB) as conn:
    q2_result = pd.read_sql_query(queries[1], conn)

print("\nQ2 — Revenue by City and Product Category")
print(q2_result)

print("Number of rows:", len(q2_result))
print("Total revenue:", round(q2_result["revenue"].sum(), 2))

print("\nHighest revenue combination:")
print(q2_result.loc[q2_result["revenue"].idxmax()])
# Execute SQL Query 3
with sqlite3.connect(DB) as conn:
    q3_result = pd.read_sql_query(queries[2], conn)

print("\nQ3 — Products Above Average Catalogue Price")
print(q3_result)
print("Number of products:", len(q3_result))
# Execute SQL Query 4
with sqlite3.connect(DB) as conn:
    q4_result = pd.read_sql_query(queries[3], conn)

print("\nQ4 — Customers Without Completed Orders")
print(q4_result)
print("Number of customers:", len(q4_result))
# Execute SQL Query 5
with sqlite3.connect(DB) as conn:
    q5_result = pd.read_sql_query(queries[4], conn)

print("\nQ5 — Customers Above Average Spending")
print(q5_result)
print("Number of customers:", len(q5_result))
# Verify Q5 average customer spending
with sqlite3.connect(DB) as conn:
    avg_spend = pd.read_sql_query(
        """
        SELECT AVG(total_spend) AS avg_spend
        FROM (
            SELECT customer_id, SUM(order_amount) AS total_spend
            FROM orders
            WHERE status = 'completed'
            GROUP BY customer_id
        )
        """,
        conn
    )

print("Average customer spending:",
      round(avg_spend["avg_spend"].iloc[0], 2))
# Execute SQL Query 6
with sqlite3.connect(DB) as conn:
    q6_result = pd.read_sql_query(queries[5], conn)

print("\nQ6 — Customer Spending Rankings by City")

# Display the highest-ranked customer in each city
top_customers = q6_result[
    q6_result["spend_rank"] == 1
]

print(top_customers[
    ["city", "customer_id", "total_spend"]
].sort_values("city"))

# Count rows where RANK differs from ROW_NUMBER
rank_differences = (
    q6_result["spend_rank"] != q6_result["row_num"]
).sum()

print("\nRows where RANK differs from ROW_NUMBER:",
      rank_differences)
# Execute SQL Query 7
with sqlite3.connect(DB) as conn:
    q7_result = pd.read_sql_query(queries[6], conn)

print("\nQ7 — Daily Revenue Running Total")
print(q7_result.tail())

print("Number of daily rows:", len(q7_result))
print(
    "Final cumulative revenue:",
    round(q7_result["cumulative_revenue"].iloc[-1], 2)
)