# Revenue Prediction System

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

# ------------------------------------
# STEP 1: Revenue Dataset
# ------------------------------------

data = {
    'Month': [1,2,3,4,5,6,7,8,9,10,
              11,12,13,14,15,16,17,18,19,20],

    'Advertising_Spend': [50,55,60,65,70,75,80,85,90,95,
                          100,105,110,115,120,125,130,135,140,145],

    'Customers': [500,550,600,650,700,750,800,850,900,950,
                  1000,1050,1100,1150,1200,1250,1300,1350,1400,1450],

    'Revenue': [5000,5500,6200,6800,7400,8100,8800,9500,10300,11000,
                11800,12600,13500,14400,15400,16400,17500,18600,19800,21000]
}

df = pd.DataFrame(data)

print("\nRevenue Dataset")
print(df.head())

# ------------------------------------
# STEP 2: Features and Target
# ------------------------------------

X = df[['Advertising_Spend', 'Customers']]
y = df['Revenue']

# ------------------------------------
# STEP 3: Train-Test Split
# ------------------------------------

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# ------------------------------------
# STEP 4: Train Model
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
# STEP 7: Revenue Analysis
# ------------------------------------

print("\nRevenue Analysis")
print("-" * 30)

print("Average Revenue :", round(df['Revenue'].mean(), 2))
print("Highest Revenue :", df['Revenue'].max())
print("Lowest Revenue  :", df['Revenue'].min())

growth_rate = (
    (df['Revenue'].iloc[-1] - df['Revenue'].iloc[0])
    / df['Revenue'].iloc[0]
) * 100

print(f"Revenue Growth Rate: {growth_rate:.2f}%")

# ------------------------------------
# STEP 8: Future Revenue Prediction
# ------------------------------------

print("\nPredict Future Revenue")

advertising = float(input("Enter Advertising Spend: "))
customers = int(input("Enter Expected Customers: "))

future_data = [[advertising, customers]]

predicted_revenue = model.predict(future_data)

print(f"\nPredicted Revenue: ₹{predicted_revenue[0]:,.2f}")

# ------------------------------------
# STEP 9: Revenue Trend Graph
# ------------------------------------

plt.figure(figsize=(8,5))
plt.plot(df['Month'], df['Revenue'], marker='o')
plt.title("Revenue Growth Trend")
plt.xlabel("Month")
plt.ylabel("Revenue")
plt.grid(True)
plt.show()

# ------------------------------------
# STEP 10: Actual vs Predicted
# ------------------------------------

plt.figure(figsize=(8,5))
plt.scatter(y_test, y_pred)
plt.xlabel("Actual Revenue")
plt.ylabel("Predicted Revenue")
plt.title("Actual vs Predicted Revenue")
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