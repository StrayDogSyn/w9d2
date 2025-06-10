import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression

# Generate synthetic data
np.random.seed(42)
X = np.linspace(0, 10, 20).reshape(-1, 1)
y = 3 * X.squeeze() + 5 + np.random.normal(0, 3, size=X.shape[0])

# Fit linear regression
model = LinearRegression()
model.fit(X, y)
y_pred = model.predict(X)

# Plot
plt.figure(figsize=(8, 5))
plt.scatter(X, y, label='Data Points', color='blue')
plt.plot(X, y_pred, label='Regression Line', color='red')

# Add vertical lines for residuals
for xi, yi, ypi in zip(X, y, y_pred):
    plt.vlines(xi, min(yi, ypi), max(yi, ypi), color='gray', linestyle='dotted')

plt.xlabel('X')
plt.ylabel('y')
plt.title('Linear Regression with Residuals')
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()
