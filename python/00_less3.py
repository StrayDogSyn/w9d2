# Load libraries
from sklearn.datasets import load_wine
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score

# Load data
data = load_wine()
X = data.data
y = data.target
target_names = data.target_names
# Print out the prediction goal
print("Goal: Predict wine class (target) from chemical features (predictors).")
print(f"Target classes: {list(target_names)}")
print(f"Feature names: {list(data.feature_names)}")

# Step 2: Standardize the feature matrix
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Preprocessing: Scale features
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Split into train/test
X_train, X_test, y_train, y_test = train_test_split(
    X_scaled, y, test_size=0.2, random_state=42, stratify=y
)

# Initialize model
clf = LogisticRegression(max_iter=5000)

# Cross-validation score (macro avg)
cv_scores = cross_val_score(clf, X_train, y_train, cv=5, scoring='accuracy')
print(f"Cross-validated accuracy: {cv_scores.mean():.3f} ± {cv_scores.std():.3f}")

# Fit the model
clf.fit(X_train, y_train)

# Predict and evaluate
y_pred = clf.predict(X_test)
print("\nClassification Report:\n")
print(classification_report(y_test, y_pred, target_names=target_names))

print("Confusion Matrix:\n")
print(confusion_matrix(y_test, y_pred))

# ROC AUC (macro for multiclass)
y_prob = clf.predict_proba(X_test)
roc_auc = roc_auc_score(y_test, y_prob, multi_class='ovo', average='macro')
print(f"ROC AUC (macro-average): {roc_auc:.3f}")
