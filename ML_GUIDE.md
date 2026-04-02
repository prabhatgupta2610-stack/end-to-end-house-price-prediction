# 🤖 Simple Machine Learning Models Toolkit

A complete guide to building ML models for **regression**, **classification**, **time series forecasting**, and **NLP sentiment analysis**.

---

## 📦 What's Included

### 1. **Main ML Models** (`ml_models_toolkit.py`)
Four essential ML projects:
- **Stock Price Prediction** - Time series regression
- **Iris Flower Classification** - Classic ML starter
- **Housing Price Prediction** - Regression with feature importance
- **Sentiment Analysis** - NLP classification

### 2. **Advanced Techniques** (`ml_advanced_techniques.py`)
Professional ML practices:
- Hyperparameter tuning (GridSearchCV, RandomizedSearchCV)
- Cross-validation strategies (K-Fold, Stratified K-Fold)
- Learning curves (bias-variance analysis)
- Validation curves (hyperparameter impact)
- ROC curves and AUC analysis

### 3. **Generated Visualizations**
- `ml_stock_prediction.png` - Predictions vs actual prices
- `ml_iris_classification.png` - Feature importance & confusion matrix
- `ml_housing_prices.png` - Price predictions & residuals
- `ml_sentiment_analysis.png` - Confusion matrix & feature importance
- `ml_learning_curves.png` - Bias-variance tradeoff
- `ml_validation_curves.png` - Hyperparameter sensitivity
- `ml_roc_curves.png` - ROC curves & AUC scores

---

## 🚀 Quick Start

### Installation
```bash
pip install scikit-learn numpy pandas matplotlib seaborn
```

### Run All Models
```bash
python ml_models_toolkit.py
```

### Run Advanced Techniques
```bash
python ml_advanced_techniques.py
```

---

## 📊 Project 1: Stock Price Prediction

### Overview
Predict stock prices using **time series regression** with multiple models.

### Model Types
- **Linear Regression** - Baseline model
- **Random Forest** - Captures non-linear patterns
- **Gradient Boosting** - Best overall performance

### Key Features
```python
# Moving averages
df['MA_5'] = df['Close'].rolling(window=5).mean()
df['MA_20'] = df['Close'].rolling(window=20).mean()

# Volatility
df['Volatility'] = df['Close'].rolling(window=10).std()

# Lag features (previous 10 days)
for lag in range(1, 11):
    df[f'Close_Lag_{lag}'] = df['Close'].shift(lag)
```

### Results Example
```
Model Performance:
  Linear Regression - MAE: $0.00, R²: 1.0000
  Random Forest     - MAE: $2.45, R²: 0.8360
  Gradient Boosting - MAE: $1.72, R²: 0.9215

Best Model: Random Forest
  Top Feature: 5-day MA (93.4%)
```

### Use Cases
- Stock market prediction
- Cryptocurrency price forecasting
- Economic indicator prediction
- Resource demand forecasting

---

## 🌸 Project 2: Iris Flower Classification

### Overview
Classify iris flowers using **supervised classification** algorithms.

### Dataset
- 150 samples
- 4 features (sepal/petal length & width)
- 3 classes (setosa, versicolor, virginica)
- Balanced distribution

### Model Types
- **Logistic Regression** - Simple, interpretable
- **Random Forest** - Handles non-linearity
- **Gradient Boosting** - Highest accuracy

### Results Example
```
Logistic Regression Accuracy: 93.33%
  Precision: 0.9333
  Recall: 0.9333
  F1-Score: 0.9333

Feature Importance (Random Forest):
  1. Petal Width:  43.72%
  2. Petal Length: 43.15%
  3. Sepal Length: 11.63%
  4. Sepal Width:  1.50%
```

### Classification Report
```
                precision  recall  f1-score  support
    setosa         1.00     1.00     1.00      10
versicolor         0.90     0.90     0.90      10
 virginica         0.90     0.90     0.90      10
```

### Use Cases
- Flower/plant species classification
- Customer segmentation
- Medical diagnosis
- Quality control

---

## 🏠 Project 3: Housing Price Prediction

### Overview
Predict house prices using **regression models** with real estate features.

### Features
- Square footage (most important)
- Number of bedrooms
- Number of bathrooms
- Property age
- Garage spaces
- Distance to city center

### Data Correlations
```
Square Feet:       0.8897 (very strong)
Bedrooms:          0.2916
Bathrooms:         0.0955
Age (negative):   -0.1670
Distance (negative): -0.1917
```

### Model Performance
```
Linear Regression - MAE: $37,817, R²: 0.9439
Random Forest     - MAE: $48,866, R²: 0.9047
Gradient Boosting - MAE: $43,390, R²: 0.9316

Best Model: Linear Regression
  R² = 0.9439 (explains 94.4% of variance)
  MAE = $37,817 (average prediction error)
```

### Feature Coefficients (Linear Model)
```
Square Feet:      +$189,586 per 1000 sq ft
Bedrooms:         +$71,566  per bedroom
Bathrooms:        +$20,251  per bathroom
Garage Spaces:    +$10,590  per space
Age (years):      -$500     per year
Distance to City: -$2,000   per mile
```

### Use Cases
- Real estate price estimation
- Property investment analysis
- Home valuation
- Market analysis

---

## 💬 Project 4: Sentiment Analysis

### Overview
Classify reviews as **positive, negative, or neutral** using NLP techniques.

### Feature Engineering
```python
# Count sentiment words
positive_words = ['excellent', 'amazing', 'great', ...]
negative_words = ['terrible', 'awful', 'bad', ...]

# Extract features
pos_count = sum(1 for word in text if word in positive_words)
neg_count = sum(1 for word in text if word in negative_words)
text_length = len(text.split())

features = [pos_count, neg_count, text_length]
```

### Model Performance
```
Logistic Regression - Accuracy: 100%
Random Forest       - Accuracy: 100%

Classification Report:
              precision  recall  f1-score  support
    Negative      1.00     1.00      1.00      36
     Neutral      1.00     1.00      1.00      31
    Positive      1.00     1.00      1.00      33
```

### Feature Importance
```
1. Negative Words:  3.0237
2. Positive Words:  0.6767
3. Text Length:     0.4202
```

### Use Cases
- Movie/product review analysis
- Customer feedback classification
- Social media sentiment tracking
- Brand reputation monitoring

---

## 🔧 Advanced Techniques

### 1. Hyperparameter Tuning

#### GridSearchCV (Exhaustive Search)
```python
from sklearn.model_selection import GridSearchCV

param_grid = {
    'n_estimators': [50, 100, 200],
    'max_depth': [3, 5, 10, None],
    'min_samples_split': [2, 5, 10]
}

grid_search = GridSearchCV(
    RandomForestClassifier(),
    param_grid,
    cv=5,
    n_jobs=-1
)
grid_search.fit(X_train, y_train)

print(f"Best params: {grid_search.best_params_}")
print(f"Best score: {grid_search.best_score_:.4f}")
```

**When to use:** Small parameter spaces (< 100 combinations)

#### RandomizedSearchCV (Random Search)
```python
from sklearn.model_selection import RandomizedSearchCV

param_dist = {
    'n_estimators': [50, 100, 150, 200, 300],
    'learning_rate': [0.01, 0.05, 0.1, 0.15],
    'max_depth': [2, 3, 4, 5, 7],
    'subsample': [0.6, 0.8, 1.0]
}

random_search = RandomizedSearchCV(
    GradientBoostingClassifier(),
    param_dist,
    n_iter=20,
    cv=5,
    n_jobs=-1
)
random_search.fit(X_train, y_train)
```

**When to use:** Large parameter spaces (> 100 combinations)

### 2. Cross-Validation Strategies

#### K-Fold Cross-Validation
```python
from sklearn.model_selection import KFold, cross_val_score

kfold = KFold(n_splits=5, shuffle=True, random_state=42)
scores = cross_val_score(model, X, y, cv=kfold, scoring='accuracy')

print(f"Scores: {scores}")
print(f"Mean: {scores.mean():.4f} (+/- {scores.std():.4f})")
```

**Use:** Balanced datasets, regular regression/classification

#### Stratified K-Fold
```python
from sklearn.model_selection import StratifiedKFold

skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
scores = cross_val_score(model, X, y, cv=skf, scoring='accuracy')
```

**Use:** Imbalanced datasets, ensure class distribution in splits

### 3. Learning Curves (Bias-Variance)

```python
from sklearn.model_selection import learning_curve

train_sizes, train_scores, val_scores = learning_curve(
    model, X, y,
    cv=5,
    train_sizes=np.linspace(0.1, 1.0, 10),
    scoring='accuracy'
)

# Analyze the gap:
# - Small gap = good fit (bias problem → need complex model)
# - Large gap = overfitting (variance problem → need regularization)
```

**Interpreting curves:**
- **Both low:** High bias (underfitting) - use complex model
- **Train high, Val low:** High variance (overfitting) - regularize
- **Both high:** Good fit

### 4. Validation Curves (Hyperparameter Sensitivity)

```python
from sklearn.model_selection import validation_curve

train_scores, val_scores = validation_curve(
    model, X, y,
    param_name='max_depth',
    param_range=range(1, 11),
    cv=5,
    scoring='accuracy'
)

# Plots show how performance changes with parameter
```

### 5. ROC Curves & AUC

```python
from sklearn.metrics import roc_curve, auc

y_pred_proba = model.predict_proba(X_test)[:, 1]
fpr, tpr, _ = roc_curve(y_test, y_pred_proba)
roc_auc = auc(fpr, tpr)

plt.plot(fpr, tpr, label=f'AUC={roc_auc:.4f}')
plt.plot([0, 1], [0, 1], 'k--', label='Random')
plt.legend()
```

---

## 📈 Model Selection Guide

### Choose Based on Problem Type

#### Regression (Continuous Output)
```
Linear Regression
├─ Best for: Linear relationships
├─ Pros: Interpretable, fast
└─ Cons: Assumes linearity

Random Forest Regressor
├─ Best for: Non-linear, complex data
├─ Pros: Handles interactions, robust
└─ Cons: Black box, slower

Gradient Boosting
├─ Best for: High-accuracy predictions
├─ Pros: Often best performance
└─ Cons: Slow, prone to overfitting
```

#### Classification (Categorical Output)
```
Logistic Regression
├─ Best for: Binary/multiclass, simple
├─ Pros: Fast, interpretable
└─ Cons: Linear decision boundaries

Random Forest Classifier
├─ Best for: Complex patterns
├─ Pros: Handles non-linearity
└─ Cons: Slow, memory-intensive

Gradient Boosting Classifier
├─ Best for: Competition-level accuracy
├─ Pros: Best performance
└─ Cons: Tuning complexity
```

---

## 🎯 Model Evaluation Metrics

### Regression
```python
from sklearn.metrics import mean_squared_error, r2_score

mae = mean_absolute_error(y_test, y_pred)        # Avg error
rmse = np.sqrt(mean_squared_error(y_test, y_pred))  # Penalizes large errors
r2 = r2_score(y_test, y_pred)                     # % variance explained (0-1)
```

### Classification
```python
from sklearn.metrics import (
    accuracy_score,    # % correct
    precision_score,   # % predicted positive that are correct
    recall_score,      # % actual positive that were found
    f1_score,         # Harmonic mean of precision & recall
    roc_auc_score     # Area under ROC curve (0-1)
)
```

---

## 💡 Best Practices

### 1. Data Preprocessing
```python
# Scale numerical features (except tree-based models)
from sklearn.preprocessing import StandardScaler

scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Encode categorical features
from sklearn.preprocessing import LabelEncoder

encoder = LabelEncoder()
y_encoded = encoder.fit_transform(y)
```

### 2. Train-Test Split
```python
from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(
    X, y,
    test_size=0.2,           # 80-20 split
    random_state=42,         # Reproducibility
    stratify=y              # For imbalanced data
)
```

### 3. Model Validation
```python
# Use cross-validation, not just train-test
from sklearn.model_selection import cross_val_score

scores = cross_val_score(model, X, y, cv=5, scoring='accuracy')
print(f"CV scores: {scores.mean():.4f} (+/- {scores.std():.4f})")
```

### 4. Avoid Common Pitfalls
```python
# ❌ DON'T: Fit scaler on entire dataset
X_all_scaled = scaler.fit_transform(X)  # Data leakage!

# ✅ DO: Fit on training data only
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# ❌ DON'T: Use training metrics for final evaluation
# ✅ DO: Use test/validation set

# ❌ DON'T: Ignore imbalanced classes
# ✅ DO: Use stratified split, weighted loss, or resampling
```

---

## 🔬 Hyperparameter Tuning Tips

### Random Forest
```python
# Number of trees: more = better (but slower)
n_estimators = [50, 100, 200, 500]

# Tree depth: lower = simpler, higher = complex
max_depth = [3, 5, 10, 20, None]

# Min samples to split: higher = simpler
min_samples_split = [2, 5, 10, 20]

# Start with: n_est=100, depth=10, min_split=2
```

### Gradient Boosting
```python
# Learning rate: lower = more iterations needed
learning_rate = [0.001, 0.01, 0.05, 0.1]

# Number of boosting stages
n_estimators = [50, 100, 200, 500]

# Subsample fraction
subsample = [0.6, 0.8, 1.0]

# Start with: lr=0.1, n_est=100, subsample=0.8
```

### Logistic Regression
```python
# Regularization strength
C = [0.001, 0.01, 0.1, 1, 10, 100]

# Penalty type
penalty = ['l2', 'l1']

# Start with: C=1.0, penalty='l2'
```

---

## 📊 Feature Importance Interpretation

### Tree-Based Models
```python
# Get importance scores
importance = model.feature_importances_

# Create dataframe
importance_df = pd.DataFrame({
    'Feature': feature_names,
    'Importance': importance
}).sort_values('Importance', ascending=False)

# Visualize
plt.barh(importance_df['Feature'], importance_df['Importance'])
```

**Interpretation:**
- Higher values = more important for predictions
- Use top 5-10 features for interpretability

### Linear Models
```python
# Get coefficients
coef = model.coef_

# Interpretation: change in y per unit change in x
# Example: $50,000 coefficient = +$50K price per bedroom
```

---

## 🚨 Common Issues & Solutions

### Issue: Model Overfitting
**Symptoms:** High train accuracy, low test accuracy

**Solutions:**
```python
# 1. Reduce model complexity
max_depth = 5  # Reduce from 20

# 2. Add regularization
C = 0.1  # Stronger for LogisticRegression

# 3. More training data
# 4. Cross-validation to catch overfitting

# 5. Early stopping (for boosting)
```

### Issue: Model Underfitting
**Symptoms:** Both train and test accuracy are low

**Solutions:**
```python
# 1. Use more complex model
max_depth = 20  # Increase from 5

# 2. Feature engineering (add interactions)
X['interaction'] = X['feature1'] * X['feature2']

# 3. More training iterations
n_estimators = 500  # Increase from 100

# 4. Reduce regularization
C = 10  # Weaker regularization
```

### Issue: Class Imbalance
```python
# Use stratified split
train_test_split(X, y, stratify=y)

# Use class weights
RandomForestClassifier(class_weight='balanced')

# Use appropriate metrics
f1_score(y_test, y_pred)  # Better than accuracy
```

---

## 📚 Resources

- **Scikit-Learn:** https://scikit-learn.org/
- **Model Selection:** https://scikit-learn.org/stable/modules/model_evaluation.html
- **Feature Engineering:** https://scikit-learn.org/stable/modules/preprocessing.html

---

## ✨ Key Takeaways

1. **Start Simple** - Linear models first, then complex
2. **Validate Properly** - Use cross-validation, not just train-test
3. **Feature Engineering** - Often beats model complexity
4. **Tune Systematically** - GridSearch or RandomSearch
5. **Evaluate Thoroughly** - Multiple metrics, not just accuracy
6. **Prevent Overfitting** - Use regularization, validation curves
7. **Interpret Results** - Feature importance, coefficients, residuals

---

Happy modeling! 🤖✨
