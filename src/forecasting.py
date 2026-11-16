
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from pathlib import Path
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import joblib

# Project paths
BASE_DIR = Path(__file__).resolve().parent.parent

DATA_PATH = BASE_DIR / "data" / "cleaned_superstore.csv"
MODEL_PATH = BASE_DIR / "models" / "sales_forecasting_model.pkl"
CHART_PATH = BASE_DIR / "data" / "charts" / "sales_forecast.png"

# Load dataset
df = pd.read_csv(DATA_PATH)

df["Order Date"] = pd.to_datetime(df["Order Date"])

# Aggregate daily sales
daily_sales = (
    df.groupby("Order Date")["Sales"]
    .sum()
    .sort_index()
)

# Convert daily data into monthly sales
monthly_sales = daily_sales.resample("ME").sum()

# Create forecasting features
forecast_df = pd.DataFrame({
    "Sales": monthly_sales
})

forecast_df["Month"] = forecast_df.index.month
forecast_df["Year"] = forecast_df.index.year
forecast_df["Quarter"] = forecast_df.index.quarter

forecast_df["Lag_1"] = forecast_df["Sales"].shift(1)
forecast_df["Lag_2"] = forecast_df["Sales"].shift(2)
forecast_df["Lag_3"] = forecast_df["Sales"].shift(3)

forecast_df["Rolling_Mean_3"] = (
    forecast_df["Sales"].shift(1).rolling(3).mean()
)

forecast_df = forecast_df.dropna()

# Define features and target
features = [
    "Month",
    "Year",
    "Quarter",
    "Lag_1",
    "Lag_2",
    "Lag_3",
    "Rolling_Mean_3"
]

X = forecast_df[features]
y = forecast_df["Sales"]

# Chronological train-test split
split_index = int(len(forecast_df) * 0.8)

X_train = X.iloc[:split_index]
X_test = X.iloc[split_index:]

y_train = y.iloc[:split_index]
y_test = y.iloc[split_index:]

# Train Random Forest model
model = RandomForestRegressor(
    n_estimators=300,
    max_depth=8,
    random_state=42
)

model.fit(X_train, y_train)

# Make predictions
predictions = model.predict(X_test)

# Evaluate model
mae = mean_absolute_error(y_test, predictions)
rmse = np.sqrt(mean_squared_error(y_test, predictions))
r2 = r2_score(y_test, predictions)

print("\n========== MODEL EVALUATION ==========")
print("MAE:", round(mae, 2))
print("RMSE:", round(rmse, 2))
print("R2 Score:", round(r2, 4))

# Save trained model
joblib.dump(model, MODEL_PATH)

# Plot actual vs predicted sales
plt.figure(figsize=(12, 5))

plt.plot(
    y_test.index,
    y_test.values,
    marker="o",
    label="Actual Sales"
)

plt.plot(
    y_test.index,
    predictions,
    marker="o",
    label="Predicted Sales"
)

plt.title("Actual vs Predicted Monthly Sales")
plt.xlabel("Month")
plt.ylabel("Sales")
plt.legend()
plt.xticks(rotation=45)
plt.tight_layout()

plt.savefig(CHART_PATH)
plt.close()

print("\nModel saved to:", MODEL_PATH)
print("Forecast chart saved to:", CHART_PATH)
print("\nForecasting completed!")
