# House Price Prediction System

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

# -----------------------------
# STEP 1: Create Dataset
# -----------------------------
data = {
    'Area': [1000, 1200, 1500, 1800, 2000, 2200, 2500, 2800, 3000, 3500,
             1100, 1400, 1600, 1900, 2100, 2400, 2600, 2900, 3200, 3600],
    
    'Bedrooms': [2, 2, 3, 3, 4, 4, 4, 5, 5, 6,
                 2, 3, 3, 4, 4, 4, 5, 5, 5, 6],
    
    'Age': [10, 8, 7, 6, 5, 4, 4, 3, 2, 1,
            9, 8, 7, 6, 5, 4, 3, 2, 2, 1],
    
    'Price': [30, 35, 45, 52, 60, 66, 75, 85, 92, 110,
              32, 42, 48, 58, 63, 72, 80, 90, 100, 115]
}

df = pd.DataFrame(data)

print("\nFirst 5 Records:")
print(df.head())

# -----------------------------
# STEP 2: Define Features & Target
# -----------------------------
X = df[['Area', 'Bedrooms', 'Age']]
y = df['Price']

# -----------------------------
# STEP 3: Split Dataset
# -----------------------------
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# -----------------------------
# STEP 4: Train Model
# -----------------------------
model = LinearRegression()
model.fit(X_train, y_train)

# -----------------------------
# STEP 5: Predictions
# -----------------------------
y_pred = model.predict(X_test)

# -----------------------------
# STEP 6: Model Evaluation
# -----------------------------
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

# -----------------------------
# STEP 7: Market Trend Analysis
# -----------------------------
average_price = df['Price'].mean()

print("\nMarket Analysis")
print("-" * 30)
print("Average House Price:", round(average_price, 2), "Lakhs")

highest_price = df['Price'].max()
lowest_price = df['Price'].min()

print("Highest House Price:", highest_price, "Lakhs")
print("Lowest House Price :", lowest_price, "Lakhs")

# -----------------------------
# STEP 8: User Prediction
# -----------------------------
print("\nEnter House Details")

area = float(input("Area (sq.ft): "))
bedrooms = int(input("Number of Bedrooms: "))
age = int(input("Age of House (years): "))

new_house = [[area, bedrooms, age]]

predicted_price = model.predict(new_house)

print("\nPredicted House Price:")
print(f"{predicted_price[0]:.2f} Lakhs")

# -----------------------------
# STEP 9: Visualization
# -----------------------------
plt.figure(figsize=(8,5))
plt.scatter(y_test, y_pred)
plt.xlabel("Actual Price")
plt.ylabel("Predicted Price")
plt.title("Actual vs Predicted House Prices")
plt.grid(True)
plt.show()

# -----------------------------
# STEP 10: Feature Importance
# -----------------------------
feature_names = X.columns

print("\nFeature Impact")
print("-" * 30)

for feature, coef in zip(feature_names, model.coef_):
    print(f"{feature}: {coef:.4f}")

print("\nModel Intercept:", round(model.intercept_, 4))