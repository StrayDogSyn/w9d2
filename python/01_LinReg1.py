import numpy as np
np.random.seed(42)

n_samples = 100
X1 = np.random.normal(0, 1, n_samples)
X2 = np.random.normal(0, 1, n_samples)
noise = np.random.normal(0, 1, n_samples)

# True weights
w0, w1, w2 = 5, 3, -2
y = w0 + w1 * X1 + w2 * X2 + noise

# Design matrix with intercept
X = np.column_stack([np.ones(n_samples), X1, X2])

## Step 2: Analytical Solution


## \hat{\mathbf{w}} = (\mathbf{X}^\top \mathbf{X})^{-1}
## \mathbf{X}^\top \mathbf{y}


w_hat = np.linalg.inv(X.T @ X) @ X.T @ y
print("Estimated coefficients:", w_hat)


## Step 3: Fit with Scikit-Learn


from sklearn.linear_model import LinearRegression

model = LinearRegression(fit_intercept=True)
X_no_intercept = np.column_stack([X1, X2])  # scikit-learn adds intercept by default
model.fit(X_no_intercept, y)

print("Intercept:", model.intercept_)
print("Coefficients:", model.coef_)


## Step 4: Visualize the Fit

import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')  # type: ignore
ax.scatter(X1, X2, y, color='blue', label='Data')


x1_grid, x2_grid = np.meshgrid(np.linspace(-3, 3, 30), np.linspace(-3, 3, 30))
y_grid = model.intercept_ + model.coef_[0]*x1_grid + model.coef_[1]*x2_grid

ax.plot_surface(x1_grid, x2_grid, y_grid, color='orange', alpha=0.5)  # type: ignore
ax.set_xlabel("X1")
ax.set_ylabel("X2")
ax.set_zlabel("y")  # type: ignore
ax.set_title("Linear Regression Fit")
plt.legend()
plt.show()


## Discussion Questions

# Are the estimated coefficients close to the true values?
# What does the sign of each coefficient imply about the feature's relationship to the target?
# How does increasing the noise affect the accuracy of the estimates?

## Summary

# You learned how to simulate, fit, and visualize a multivariate linear regression.
# This is a building block for understanding real-world modeling.
