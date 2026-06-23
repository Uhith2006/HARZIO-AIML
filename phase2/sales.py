# Sales Forecasting System

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

# ------------------------------------
# STEP 1: Sales Dataset
# ------------------------------------

data = {
    'Month': [1,2,3,4,5,6,7,8,9,10,11,12,
              13,14,15,16,17,18,19,20],

    'Marketing_Spend': [10,12,11,13,15,14,16,18,17,19,
                        20,22,21,23,24,25,27,28,29,30],

    'Sales': [200,220,230,250,270,280,300,320,330,350,
              370,390,400,420,440,460,480,500,520,550]
}

df = pd.DataFrame(data)

print("\nSales Dataset")
print(df.head())

# ------------------------------------
# STEP 2: Features and Target
# ------------------------------------

X = df[['Month', 'Marketing_Spend']]
y = df['Sales']

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

print("\nModel Performance")
print("-" * 30)
print("MAE :", round(mae, 2))
print("MSE :", round(mse, 2))
print("RMSE:", round(rmse, 2))
print("R² Score:", round(r2, 2))

# ------------------------------------
# STEP 7: Sales Trend Analysis
# ------------------------------------

print("\nSales Analysis")
print("-" * 30)

print("Average Sales:", round(df['Sales'].mean(), 2))
print("Maximum Sales:", df['Sales'].max())
print("Minimum Sales:", df['Sales'].min())

growth = ((df['Sales'].iloc[-1] - df['Sales'].iloc[0])
          / df['Sales'].iloc[0]) * 100

print(f"Overall Growth: {growth:.2f}%")

# ------------------------------------
# STEP 8: Future Sales Prediction
# ------------------------------------

print("\nPredict Future Sales")

month = int(input("Enter Future Month: "))
marketing = float(input("Enter Marketing Spend (in lakhs): "))

future_data = [[month, marketing]]

prediction = model.predict(future_data)

print(f"\nPredicted Sales: {prediction[0]:.2f} Units")

# ------------------------------------
# STEP 9: Sales Trend Graph
# ------------------------------------

plt.figure(figsize=(8,5))
plt.plot(df['Month'], df['Sales'], marker='o')
plt.title("Monthly Sales Trend")
plt.xlabel("Month")
plt.ylabel("Sales")
plt.grid(True)
plt.show()

# ------------------------------------
# STEP 10: Actual vs Predicted
# ------------------------------------

plt.figure(figsize=(8,5))
plt.scatter(y_test, y_pred)
plt.xlabel("Actual Sales")
plt.ylabel("Predicted Sales")
plt.title("Actual vs Predicted Sales")
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