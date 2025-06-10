
# breakout_exercise_linear_regression.py

"""
Breakout Exercise: Linear Regression with EDA and Visualization
---------------------------------------------------------------

Objective:
Explore a real-world dataset, perform exploratory data analysis (EDA),
fit a linear regression model, and visualize key relationships.

Dataset:
We will use the California Housing dataset from sklearn.

Tasks:
1. Load and inspect the dataset.
2. Perform EDA using visualizations.
3. Fit a linear regression model to predict median house value.
4. Interpret the model and residuals.

Note: Run this script as a standalone Python file, not in a notebook.
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.datasets import fetch_california_housing
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score, mean_squared_error

# Load the dataset
# Ignore type checking for this specific line due to sklearn API
housing_data = fetch_california_housing(as_frame=True)  # type: ignore
df = housing_data.frame  # type: ignore

# Add feature names and target
X = df.drop(columns="MedHouseVal")
y = df["MedHouseVal"]

# 1. Basic EDA
print("\n--- Dataset Overview ---")
print(df.info())
print("\n--- Summary Statistics ---")
print(df.describe())

# 2. Correlation Matrix
plt.figure(figsize=(10, 8))
sns.heatmap(df.corr(), annot=True, cmap="coolwarm", fmt=".2f")
plt.title("Correlation Matrix")
plt.tight_layout()
plt.savefig("data/correlation_matrix.png")

# 3. Visualize Features vs Target
features_to_plot = ["MedInc", "AveRooms", "HouseAge", "AveOccup"]
for feature in features_to_plot:
    plt.figure()
    sns.scatterplot(data=df, x=feature, y="MedHouseVal", alpha=0.5)
    plt.title(f"{feature} vs Median House Value")
    plt.savefig(f"data/scatter_{feature}.png")

# 4. Split the data
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# 5. Fit a Linear Regression Model
model = LinearRegression()
model.fit(X_train, y_train)

print("\n--- Model Coefficients ---")
for name, coef in zip(X.columns, model.coef_):
    print(f"{name}: {coef:.4f}")
print(f"Intercept: {model.intercept_:.4f}")

# 6. Evaluate the Model
y_pred = model.predict(X_test)
mse = mean_squared_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)

print(f"\nTest MSE: {mse:.3f}")
print(f"Test R^2: {r2:.3f}")

# 7. Residual Plot
residuals = y_test - y_pred
plt.figure()
sns.histplot(residuals, kde=True)
plt.title("Distribution of Residuals")
plt.xlabel("Residual")
plt.savefig("data/residual_distribution.png")

plt.figure()
plt.scatter(y_pred, residuals, alpha=0.5)
plt.axhline(0, color="red", linestyle="--")
plt.xlabel("Predicted")
plt.ylabel("Residual")
plt.title("Residuals vs Predicted Values")
plt.savefig("data/residuals_vs_predicted.png")

print("\nVisualizations saved as PNG files.")
