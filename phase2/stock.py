# Stock Price Prediction System

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

# ------------------------------------
# STEP 1: Historical Stock Dataset
# ------------------------------------

data = {
    'Day': [1,2,3,4,5,6,7,8,9,10,
            11,12,13,14,15,16,17,18,19,20],
    
    'Volume': [1000,1200,1100,1300,1400,
               1500,1450,1600,1700,1800,
               1750,1900,2000,2100,2200,
               2300,2400,2500,2600,2700],
    
    'StockPrice': [100,102,105,107,110,
                   112,115,118,120,123,
                   125,128,130,133,136,
                   139,142,145,148,150]
}

df = pd.DataFrame(data)

print("\nHistorical Stock Data")
print(df.head())

# ------------------------------------
# STEP 2: Features and Target
# ------------------------------------

X = df[['Day', 'Volume']]
y = df['StockPrice']

# ------------------------------------
# STEP 3: Train-Test Split
# ------------------------------------

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# ------------------------------------
# STEP 4: Train Regression Model
# ------------------------------------

model = LinearRegression()
model.fit(X_train, y_train)

# ------------------------------------
# STEP 5: Predictions
# ------------------------------------

y_pred = model.predict(X_test)

# ------------------------------------
# STEP 6: Evaluate Model
# ------------------------------------

mae = mean_absolute_error(y_test, y_pred)
mse = mean_squared_error(y_test, y_pred)
rmse = np.sqrt(mse)
r2 = r2_score(y_test, y_pred)

print("\nModel Evaluation")
print("-" * 30)
print("MAE :", round(mae, 2))
print("MSE :", round(mse, 2))
print("RMSE:", round(rmse, 2))
print("R² Score:", round(r2, 2))

# ------------------------------------
# STEP 7: Market Trend Analysis
# ------------------------------------

avg_price = df['StockPrice'].mean()

print("\nMarket Trend Analysis")
print("-" * 30)
print("Average Stock Price:", round(avg_price, 2))

print("Highest Price:", df['StockPrice'].max())
print("Lowest Price :", df['StockPrice'].min())

trend = df['StockPrice'].iloc[-1] - df['StockPrice'].iloc[0]

if trend > 0:
    print("Market Trend: Upward 📈")
else:
    print("Market Trend: Downward 📉")

# ------------------------------------
# STEP 8: User Prediction
# ------------------------------------

print("\nPredict Future Stock Price")

future_day = int(input("Enter Future Day: "))
future_volume = int(input("Enter Expected Trading Volume: "))

future_data = [[future_day, future_volume]]

predicted_price = model.predict(future_data)

print("\nPredicted Stock Price:")
print(f"₹ {predicted_price[0]:.2f}")

# ------------------------------------
# STEP 9: Visualization
# ------------------------------------

plt.figure(figsize=(8,5))
plt.plot(df['Day'], df['StockPrice'], marker='o')
plt.title("Historical Stock Price Trend")
plt.xlabel("Day")
plt.ylabel("Stock Price")
plt.grid(True)
plt.show()

# ------------------------------------
# STEP 10: Actual vs Predicted
# ------------------------------------

plt.figure(figsize=(8,5))
plt.scatter(y_test, y_pred)
plt.xlabel("Actual Price")
plt.ylabel("Predicted Price")
plt.title("Actual vs Predicted Stock Prices")
plt.grid(True)
plt.show()

# ------------------------------------
# STEP 11: Feature Importance
# ------------------------------------

print("\nFeature Importance")
print("-" * 30)

for feature, coef in zip(X.columns, model.coef_):
    print(f"{feature}: {coef:.4f}")

print("\nIntercept:", round(model.intercept_, 4))