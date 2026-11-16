
import pandas as pd
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent.parent

DATA_PATH = BASE_DIR / "data" / "superstore.csv"
OUTPUT_PATH = BASE_DIR / "data" / "cleaned_superstore.csv"


def clean_sales_data():

    # Load original dataset
    df = pd.read_csv(DATA_PATH)

    print("Original shape:", df.shape)

    # Remove spaces from column names
    df.columns = df.columns.str.strip()

    # Keep only rows with a numeric Row ID
    df["Row ID"] = pd.to_numeric(
        df["Row ID"], errors="coerce"
    )

    df = df.dropna(subset=["Row ID"])

    # Convert Row ID to integer
    df["Row ID"] = df["Row ID"].astype(int)

    # Convert date columns
    df["Order Date"] = pd.to_datetime(
        df["Order Date"],
        format="%m/%d/%Y",
        errors="coerce"
    )

    df["Ship Date"] = pd.to_datetime(
        df["Ship Date"],
        format="%m/%d/%Y",
        errors="coerce"
    )

    # Remove rows without valid order dates
    df = df.dropna(subset=["Order Date"])

    # Remove duplicate Row IDs
    df = df.drop_duplicates(subset=["Row ID"])

    # Convert numerical columns
    numeric_columns = [
        "Sales", "Quantity", "Discount", "Profit"
    ]

    for column in numeric_columns:
        df[column] = pd.to_numeric(
            df[column], errors="coerce"
        )

    # Remove rows missing essential sales information
    df = df.dropna(
        subset=["Sales", "Quantity", "Profit"]
    )

    # Sort records chronologically
    df = df.sort_values("Order Date")

    # Save cleaned dataset
    df.to_csv(OUTPUT_PATH, index=False)

    print("\n========== CLEANING REPORT ==========")
    print("Final rows:", len(df))
    print("Final columns:", len(df.columns))
    print("Duplicate Row IDs:", df["Row ID"].duplicated().sum())
    print("\nRemaining missing values:")
    print(df.isnull().sum())

    print("\nCleaned file saved:", OUTPUT_PATH)
    print("\nCleaning completed!")


if __name__ == "__main__":
    clean_sales_data()
