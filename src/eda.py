
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path

# Project paths
BASE_DIR = Path(__file__).resolve().parent.parent
DATA_PATH = BASE_DIR / "data" / "cleaned_superstore.csv"
OUTPUT_DIR = BASE_DIR / "data" / "charts"

OUTPUT_DIR.mkdir(exist_ok=True)

# Load cleaned data
df = pd.read_csv(DATA_PATH)

df["Order Date"] = pd.to_datetime(df["Order Date"])

sns.set_theme(style="whitegrid")


# 1. Monthly Sales Trend
monthly_sales = (
    df.set_index("Order Date")
    .resample("ME")["Sales"]
    .sum()
)

plt.figure(figsize=(12, 5))
monthly_sales.plot(marker="o")
plt.title("Monthly Sales Trend")
plt.xlabel("Month")
plt.ylabel("Sales")
plt.tight_layout()
plt.savefig(OUTPUT_DIR / "monthly_sales.png")
plt.close()


# 2. Sales by Category
category_sales = df.groupby("Category")["Sales"].sum()

plt.figure(figsize=(8, 5))
category_sales.plot(kind="bar")
plt.title("Sales by Category")
plt.xlabel("Category")
plt.ylabel("Total Sales")
plt.xticks(rotation=0)
plt.tight_layout()
plt.savefig(OUTPUT_DIR / "category_sales.png")
plt.close()


# 3. Profit by Category
category_profit = df.groupby("Category")["Profit"].sum()

plt.figure(figsize=(8, 5))
category_profit.plot(kind="bar")
plt.title("Profit by Category")
plt.xlabel("Category")
plt.ylabel("Total Profit")
plt.xticks(rotation=0)
plt.tight_layout()
plt.savefig(OUTPUT_DIR / "category_profit.png")
plt.close()


# 4. Sales by Region
region_sales = df.groupby("Region")["Sales"].sum().sort_values()

plt.figure(figsize=(8, 5))
region_sales.plot(kind="barh")
plt.title("Sales by Region")
plt.xlabel("Total Sales")
plt.tight_layout()
plt.savefig(OUTPUT_DIR / "region_sales.png")
plt.close()


# 5. Top 10 Products
top_products = (
    df.groupby("Product Name")["Sales"]
    .sum()
    .nlargest(10)
    .sort_values()
)

plt.figure(figsize=(12, 7))
top_products.plot(kind="barh")
plt.title("Top 10 Products by Sales")
plt.xlabel("Total Sales")
plt.tight_layout()
plt.savefig(OUTPUT_DIR / "top_products.png")
plt.close()


print("========== EDA COMPLETED ==========")
print("\nTotal Sales:", round(df["Sales"].sum(), 2))
print("Total Profit:", round(df["Profit"].sum(), 2))

print("\nCategory Sales:")
print(category_sales)

print("\nRegion Sales:")
print(region_sales)

print("\nCharts saved inside:", OUTPUT_DIR)
