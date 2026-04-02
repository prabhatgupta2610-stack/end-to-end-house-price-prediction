# 🚀 Machine Learning Quick Reference

## IMPORTS & SETUP

```python
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split, cross_val_score, GridSearchCV
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier
from sklearn.metrics import mean_squared_error, accuracy_score, confusion_matrix
import matplotlib.pyplot as plt
```

---

## DATA PREPARATION

### Train-Test Split
```python
# Standard split (80-20)
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# Stratified split (for imbalanced data)
train_test_split(X, y, test_size=0.2, stratify=y, random_state=42)
```

### Feature Scaling
```python
# Scale features (important for: LR, KNN, SVM, Neural Networks)
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# NOT needed for tree-based models
```

### Encoding Categorical Variables
```python
# Label Encoding (for ordered categories)
encoder = LabelEncoder()
y_encoded = encoder.fit_transform(y)

# One-Hot Encoding (for unordered categories)
X_encoded = pd.get_dummies(X, columns=['category_col'])
```

---

## REGRESSION MODELS

### Linear Regression
```python
from sklearn.linear_model import LinearRegression

model = LinearRegression()
model.fit(X_train, y_train)
y_pred = model.predict(X_test)

# Coefficients & intercept
coef = model.coef_           # Slope for each feature
intercept = model.intercept_ # Y-intercept
```

### Random Forest Regressor
```python
from sklearn.ensemble import RandomForestRegressor

model = RandomForestRegressor(
    n_estimators=100,    # Number of trees
    max_depth=10,        # Tree depth
    min_samples_split=2, # Min samples to split
    random_state=42
)
model.fit(X_train, y_train)
y_pred = model.predict(X_test)
```

### Gradient Boosting
```python
from sklearn.ensemble import GradientBoostingRegressor

model = GradientBoostingRegressor(
    n_estimators=100,
    learning_rate=0.1,
    max_depth=5,
    random_state=42
)
model.fit(X_train, y_train)
y_pred = model.predict(X_test)
```

---

## CLASSIFICATION MODELS

### Logistic Regression
```python
from sklearn.linear_model import LogisticRegression

model = LogisticRegression(
    max_iter=1000,
    C=1.0,           # Regularization strength (lower = stronger)
    random_state=42
)
model.fit(X_train, y_train)
y_pred = model.predict(X_test)
y_pred_proba = model.predict_proba(X_test)  # Probabilities
```

### Random Forest Classifier
```python
from sklearn.ensemble import RandomForestClassifier

model = RandomForestClassifier(
    n_estimators=100,
    max_depth=10,
    min_samples_split=2,
    class_weight='balanced',  # Handle imbalance
    random_state=42
)
model.fit(X_train, y_train)
y_pred = model.predict(X_test)
```

### Gradient Boosting Classifier
```python
from sklearn.ensemble import GradientBoostingClassifier

model = GradientBoostingClassifier(
    n_estimators=100,
    learning_rate=0.1,
    max_depth=5,
    random_state=42
)
model.fit(X_train, y_train)
y_pred = model.predict(X_test)
```

---

## MODEL EVALUATION

### Regression Metrics
```python
from sklearn.metrics import mean_squared_error, r2_score

mae = mean_absolute_error(y_test, y_pred)           # Average error
rmse = np.sqrt(mean_squared_error(y_test, y_pred))  # Root mean squared error
r2 = r2_score(y_test, y_pred)                        # R² (0-1, higher better)
mape = np.mean(np.abs((y_test - y_pred) / y_test))  # % error
```

**Interpretation:**
- MAE/RMSE: Smaller is better (same units as y)
- R²: Closer to 1 is better (% variance explained)

### Classification Metrics
```python
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    confusion_matrix, classification_report, roc_auc_score
)

# Single metrics
accuracy = accuracy_score(y_test, y_pred)           # % correct
precision = precision_score(y_test, y_pred)        # % predicted pos that are correct
recall = recall_score(y_test, y_pred)               # % actual pos that were found
f1 = f1_score(y_test, y_pred)                       # Harmonic mean
roc_auc = roc_auc_score(y_test, y_pred_proba)      # ROC curve area

# Full report
print(classification_report(y_test, y_pred))

# Confusion matrix
cm = confusion_matrix(y_test, y_pred)
```

**Choose Metric By Use Case:**
- **Balanced Data:** Accuracy
- **Imbalanced Data:** F1-Score, Precision, Recall
- **Probabilities Needed:** ROC-AUC

---

## HYPERPARAMETER TUNING

### GridSearchCV (Exhaustive)
```python
from sklearn.model_selection import GridSearchCV

param_grid = {
    'n_estimators': [50, 100, 200],
    'max_depth': [5, 10, 20],
    'min_samples_split': [2, 5, 10]
}

grid = GridSearchCV(
    RandomForestClassifier(),
    param_grid,
    cv=5,              # 5-fold cross-validation
    n_jobs=-1,         # Use all cores
    scoring='accuracy'
)

grid.fit(X_train, y_train)
print(grid.best_params_)     # Best parameters
print(grid.best_score_)      # Best CV score
```

### RandomizedSearchCV (Random)
```python
from sklearn.model_selection import RandomizedSearchCV

param_dist = {
    'n_estimators': [50, 100, 150, 200],
    'max_depth': [3, 5, 10, 20, None],
    'min_samples_split': [2, 5, 10]
}

random = RandomizedSearchCV(
    RandomForestClassifier(),
    param_dist,
    n_iter=20,  # Test 20 random combinations
    cv=5,
    n_jobs=-1
)

random.fit(X_train, y_train)
```

---

## CROSS-VALIDATION

### K-Fold
```python
from sklearn.model_selection import cross_val_score, KFold

kfold = KFold(n_splits=5, shuffle=True, random_state=42)
scores = cross_val_score(model, X, y, cv=kfold, scoring='accuracy')

print(f"Scores: {scores}")
print(f"Mean: {scores.mean():.4f} (+/- {scores.std():.4f})")
```

### Stratified K-Fold (for imbalanced data)
```python
from sklearn.model_selection import StratifiedKFold

skf = StratifiedKFold(n_splits=5, random_state=42)
scores = cross_val_score(model, X, y, cv=skf, scoring='f1_weighted')
```

---

## FEATURE IMPORTANCE

### Tree-Based Models
```python
# Get importance
importance = model.feature_importances_

# Create dataframe
importance_df = pd.DataFrame({
    'Feature': X.columns,
    'Importance': importance
}).sort_values('Importance', ascending=False)

# Visualize
plt.barh(importance_df['Feature'][:10], importance_df['Importance'][:10])
```

### Linear Models
```python
# Get coefficients
coef = model.coef_

# Interpretation: change in y per unit change in x
coef_df = pd.DataFrame({
    'Feature': X.columns,
    'Coefficient': coef
}).sort_values('Coefficient', ascending=False, key=abs)
```

---

## VISUALIZATIONS

### Confusion Matrix
```python
from sklearn.metrics import confusion_matrix
import seaborn as sns

cm = confusion_matrix(y_test, y_pred)
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
           xticklabels=['Negative', 'Positive'],
           yticklabels=['Negative', 'Positive'])
```

### ROC Curve
```python
from sklearn.metrics import roc_curve, auc

y_pred_proba = model.predict_proba(X_test)[:, 1]
fpr, tpr, _ = roc_curve(y_test, y_pred_proba)
roc_auc = auc(fpr, tpr)

plt.plot(fpr, tpr, label=f'AUC={roc_auc:.4f}')
plt.plot([0, 1], [0, 1], 'k--', label='Random')
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.legend()
```

### Learning Curve
```python
from sklearn.model_selection import learning_curve

train_sizes, train_scores, val_scores = learning_curve(
    model, X, y, cv=5, scoring='accuracy',
    train_sizes=np.linspace(0.1, 1.0, 10)
)

plt.plot(train_sizes, train_scores.mean(axis=1), label='Train')
plt.plot(train_sizes, val_scores.mean(axis=1), label='Validation')
```

---

## COMMON PATTERNS

### Pattern 1: Complete ML Pipeline
```python
# 1. Load and split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

# 2. Scale
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# 3. Train
model = RandomForestClassifier()
model.fit(X_train, y_train)

# 4. Evaluate
y_pred = model.predict(X_test)
print(f"Accuracy: {accuracy_score(y_test, y_pred):.4f}")
```

### Pattern 2: Hyperparameter Tuning
```python
# 1. Define grid
param_grid = {'max_depth': [5, 10, 15]}

# 2. GridSearch
grid = GridSearchCV(model, param_grid, cv=5)
grid.fit(X_train, y_train)

# 3. Use best model
best_model = grid.best_estimator_
y_pred = best_model.predict(X_test)
```

### Pattern 3: Cross-Validation
```python
# Evaluate without tuning
scores = cross_val_score(model, X, y, cv=5, scoring='accuracy')
print(f"CV Score: {scores.mean():.4f} (+/- {scores.std():.4f})")
```

### Pattern 4: Feature Importance
```python
# Train model
model.fit(X_train, y_train)

# Get importance
importance = pd.DataFrame({
    'Feature': X.columns,
    'Importance': model.feature_importances_
}).sort_values('Importance', ascending=False)

print(importance.head(10))
```

---

## COMMON HYPERPARAMETERS

### Random Forest
```
n_estimators: 50-500        # More trees = better but slower
max_depth: 5-20             # Deeper = more complex
min_samples_split: 2-20     # Higher = simpler
min_samples_leaf: 1-10      # Prevent overfitting
max_features: 'sqrt', 'log2' # Feature sampling
```

### Gradient Boosting
```
n_estimators: 50-500        # More iterations = better
learning_rate: 0.001-0.1    # Lower = more gradual (need more iter)
max_depth: 3-10             # Usually shallow trees
subsample: 0.5-1.0          # Data sampling
```

### Logistic Regression
```
C: 0.001-1000               # Regularization strength (inverse)
                            # Lower C = stronger regularization
penalty: 'l2', 'l1', 'elasticnet'  # Regularization type
max_iter: 100-10000         # Iterations needed to converge
```

---

## QUICK DECISION TREE

```
Problem Type?
├─ Regression (continuous output)
│  ├─ Linear relationship? → Linear Regression
│  ├─ Non-linear? → Random Forest / Gradient Boosting
│  └─ Need interpretability? → Linear Regression
│
└─ Classification (categorical output)
   ├─ Simple/linear? → Logistic Regression
   ├─ Non-linear/complex? → Random Forest
   ├─ Best accuracy? → Gradient Boosting
   └─ Imbalanced? → Use F1, weighted loss, or resample
```

---

## TROUBLESHOOTING

### Model Too Simple (Underfitting)
- ❌ Low train accuracy, low test accuracy
- ✅ Solution: Use complex model, add features, more data

### Model Too Complex (Overfitting)
- ❌ High train accuracy, low test accuracy
- ✅ Solution: Reduce complexity, regularization, more data

### Class Imbalance
- ❌ Model biased toward majority class
- ✅ Solution: Stratified split, class_weight='balanced', F1 metric

### Scaling Issues
- ❌ Model not converging, poor performance
- ✅ Solution: StandardScaler() before training

### Data Leakage
- ❌ Unrealistically high performance
- ✅ Solution: Fit scaler/encoder on train only, not test

---

## ONE-LINERS

```python
# Quick baseline
cross_val_score(RandomForestClassifier(), X, y, cv=5).mean()

# Feature importance
pd.DataFrame({'Feature': X.columns, 'Importance': model.feature_importances_})

# Get best model from grid search
best_model = grid.best_estimator_

# Save model
import pickle; pickle.dump(model, open('model.pkl', 'wb'))

# Load model
model = pickle.load(open('model.pkl', 'rb'))
```

---

**Pro Tip:** Always start with simple models (Linear Regression, Logistic Regression). Only use complex models if performance is insufficient. 🎯
