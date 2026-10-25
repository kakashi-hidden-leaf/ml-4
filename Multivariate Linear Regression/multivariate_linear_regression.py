import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

# 1. Dataset Load Karna
data = pd.read_csv("house_price_data.csv")

print("=" * 60)
print("MULTIVARIATE LINEAR REGRESSION")
print("=" * 60)
print("\nFirst 5 Records:")
print(data.head())

print("\n" + "=" * 60)
print("DATASET INFORMATION:")
print("=" * 60)
print(data.info())

print("\nStatistical Summary:")
print(data.describe())

# 2. Missing Values Check
print("\n" + "=" * 60)
print("MISSING VALUE ANALYSIS:")
print("=" * 60)
print(data.isnull().sum())

# 3. Correlation Analysis & Heatmap
print("\n" + "=" * 60)
print("CORRELATION MATRIX:")
print("=" * 60)
correlation = data.corr()
print(correlation)

plt.figure(figsize=(10, 7))
sns.heatmap(correlation, annot=True, cmap="coolwarm", fmt=".2f")
plt.title("Correlation Heatmap")
plt.tight_layout()
plt.show()

# 4. Input (X) aur Target (Y) Split
X = data[["Area_sqft", "Bedrooms", "Bathrooms", "Age_Years", "Distance_City_km"]]
Y = data["Price_Lakhs"]

# 5. Train-Test Split (80:20)
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.20, random_state=42)

print("\n" + "=" * 60)
print("TRAIN TEST SPLIT:")
print("=" * 60)
print("Total Records    :", len(data))
print("Training Records :", len(X_train))
print("Testing Records  :", len(X_test))

# 6. Model Training
model = LinearRegression()
model.fit(X_train, Y_train)

print("\n" + "=" * 60)
print("MODEL TRAINING:")
print("=" * 60)
print("Multivariate Linear Regression model trained successfully.")
print(f"Intercept: {model.intercept_:.4f}")

print("\nRegression Coefficients:")
for feature, coef in zip(X.columns, model.coef_):
    print(f"{feature}: {coef:.6f}")

# 7. Predictions on Test Data
Y_pred = model.predict(X_test)

comparison = pd.DataFrame({
    "Actual Price": Y_test.values,
    "Predicted Price": np.round(Y_pred, 2),
    "Residual": np.round(Y_test.values - Y_pred, 2)
})

print("\n" + "=" * 60)
print("ACTUAL VS PREDICTED PRICE:")
print("=" * 60)
print(comparison)

# 8. Evaluation Metrics
mae = mean_absolute_error(Y_test, Y_pred)
mse = mean_squared_error(Y_test, Y_pred)
rmse = np.sqrt(mse)
r2 = r2_score(Y_test, Y_pred)

print("\n" + "=" * 60)
print("MODEL EVALUATION:")
print("=" * 60)
print(f"Mean Absolute Error (MAE) : {mae:.4f}")
print(f"Mean Squared Error (MSE)  : {mse:.4f}")
print(f"Root Mean Squared Error   : {rmse:.4f}")
print(f"R2 Score                  : {r2:.4f}")
print(f"R2 Percentage             : {r2 * 100:.2f}%")

# 9. Actual vs Predicted Plot
plt.figure(figsize=(8, 6))
plt.scatter(Y_test, Y_pred, color="teal", label="Predicted values")
min_val = min(Y_test.min(), Y_pred.min())
max_val = max(Y_test.max(), Y_pred.max())
plt.plot([min_val, max_val], [min_val, max_val], color="red", linestyle="--", label="Perfect Prediction")
plt.xlabel("Actual House Price (Lakhs)")
plt.ylabel("Predicted House Price (Lakhs)")
plt.title("Actual vs Predicted House Prices")
plt.legend()
plt.tight_layout()
plt.show()

# 10. Residual Plot
residuals = Y_test.values - Y_pred
plt.figure(figsize=(8, 5))
plt.scatter(Y_pred, residuals, color="purple")
plt.axhline(0, color="red", linestyle="--")
plt.xlabel("Predicted Price (Lakhs)")
plt.ylabel("Residuals")
plt.title("Residual Analysis")
plt.tight_layout()
plt.show()

# 11. Predict Price for a New House
new_house = pd.DataFrame({
    "Area_sqft": [2000],
    "Bedrooms": [4],
    "Bathrooms": [3],
    "Age_Years": [5],
    "Distance_City_km": [4]
})

pred_new = model.predict(new_house)
print("\n" + "=" * 60)
print("NEW HOUSE PRICE PREDICTION:")
print("=" * 60)
print(f"Predicted House Price: {pred_new[0]:.2f} Lakhs")
print("\nEXPERIMENT COMPLETED SUCCESSFULLY!")