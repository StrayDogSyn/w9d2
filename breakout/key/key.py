# Answer Key: Linear Regression with the California Housing Dataset

## 1. Which features are most strongly correlated with the target variable?

Using the correlation matrix and heatmap, we find that:
- 'MedInc' (median income) has the strongest positive correlation with the target variable ('MedHouseVal').
- 'AveRooms' and 'HouseAge' have moderate correlations.
- 'AveOccup' and 'Population' are less correlated.

## 2. How does median income affect median house value?

- From the scatter plot of 'MedInc' vs 'MedHouseVal', we see a clear positive trend.
- Higher median income is associated with higher median house values.
- The relationship appears nonlinear at the high end (suggesting diminishing returns).

## 3. What does the residual plot tell us about model assumptions?

- The residuals show some heteroscedasticity: the spread of residuals increases with the predicted value.
- This violates the assumption of constant variance (homoscedasticity).
- Suggests the need for transforming the response or using robust regression methods.

## 4. Which assumptions of linear regression may be violated?

- **Linearity**: Mostly holds, especially for 'MedInc'. But some relationships appear nonlinear.
- **Heteroscedasticity**: Residual plot shows increasing variance — assumption is violated.
- **Multicollinearity**: Correlation heatmap suggests moderate multicollinearity between some predictors (e.g., AveRooms and AveOccup).
- **Normality of residuals**: The histogram shows residuals are roughly normal but with some skew.
- **Outliers**: A few points with large residuals suggest outliers.

## 5. What would improve the model?

- Feature engineering: polynomial terms or interaction terms.
- Use regularized regression (e.g., Ridge or Lasso).
- Apply log transformation to skewed predictors or response variable.
- Use tree-based models or nonlinear regression models.
