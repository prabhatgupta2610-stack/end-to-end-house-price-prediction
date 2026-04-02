"""
Simple Machine Learning Models Toolkit
Stock Price Prediction, Iris Classification, Housing Prices, Sentiment Analysis
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime, timedelta
import warnings
warnings.filterwarnings('ignore')

# ML Libraries
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier, GradientBoostingRegressor
from sklearn.metrics import (
    mean_squared_error, mean_absolute_error, r2_score,
    accuracy_score, precision_score, recall_score, f1_score,
    confusion_matrix, classification_report, roc_auc_score, roc_curve
)
from sklearn.datasets import load_iris
import pickle

# ============================================================================
# 1. STOCK PRICE PREDICTION (TIME SERIES REGRESSION)
# ============================================================================

class StockPricePrediction:
    """Predict stock prices using time series regression"""
    
    def __init__(self):
        self.model = None
        self.scaler = StandardScaler()
        self.history = None
        
    def generate_stock_data(self, days=500):
        """Generate synthetic stock price data"""
        np.random.seed(42)
        dates = pd.date_range(start='2023-01-01', periods=days, freq='D')
        
        # Simulate stock price with trend, seasonality, and noise
        t = np.arange(days)
        trend = 0.05 * t  # Upward trend
        seasonality = 20 * np.sin(2 * np.pi * t / 252)  # Yearly pattern
        noise = np.random.normal(0, 5, days)
        price = 100 + trend + seasonality + noise
        
        df = pd.DataFrame({
            'Date': dates,
            'Close': np.maximum(price, 50),  # Ensure positive prices
            'Volume': np.random.randint(1000000, 10000000, days)
        })
        
        return df
    
    def engineer_features(self, df, lookback=10):
        """Create lag features for time series prediction"""
        df = df.copy()
        
        # Moving averages
        df['MA_5'] = df['Close'].rolling(window=5).mean()
        df['MA_20'] = df['Close'].rolling(window=20).mean()
        
        # Volatility (standard deviation)
        df['Volatility'] = df['Close'].rolling(window=10).std()
        
        # Price change
        df['Price_Change'] = df['Close'].diff()
        df['Returns'] = df['Close'].pct_change()
        
        # Lag features
        for lag in range(1, lookback + 1):
            df[f'Close_Lag_{lag}'] = df['Close'].shift(lag)
        
        # Drop NaN rows
        df = df.dropna()
        
        return df
    
    def train_model(self, df):
        """Train stock price prediction model"""
        print("\n" + "="*60)
        print("STOCK PRICE PREDICTION MODEL")
        print("="*60)
        
        # Feature engineering
        df_features = self.engineer_features(df)
        
        # Prepare features and target
        feature_cols = [col for col in df_features.columns 
                       if col not in ['Date', 'Close', 'Volume']]
        X = df_features[feature_cols].values
        y = df_features['Close'].values
        
        # Train-test split (80-20)
        split_idx = int(len(X) * 0.8)
        X_train, X_test = X[:split_idx], X[split_idx:]
        y_train, y_test = y[:split_idx], y[split_idx:]
        
        # Scaling
        X_train_scaled = self.scaler.fit_transform(X_train)
        X_test_scaled = self.scaler.transform(X_test)
        
        # Train multiple models
        print("\n1️⃣  Training Models:")
        
        # Linear Regression
        lr_model = LinearRegression()
        lr_model.fit(X_train_scaled, y_train)
        lr_pred = lr_model.predict(X_test_scaled)
        lr_mae = mean_absolute_error(y_test, lr_pred)
        lr_r2 = r2_score(y_test, lr_pred)
        print(f"   Linear Regression - MAE: ${lr_mae:.2f}, R²: {lr_r2:.4f}")
        
        # Random Forest
        rf_model = RandomForestRegressor(n_estimators=100, random_state=42, n_jobs=-1)
        rf_model.fit(X_train_scaled, y_train)
        rf_pred = rf_model.predict(X_test_scaled)
        rf_mae = mean_absolute_error(y_test, rf_pred)
        rf_r2 = r2_score(y_test, rf_pred)
        print(f"   Random Forest - MAE: ${rf_mae:.2f}, R²: {rf_r2:.4f}")
        
        # Gradient Boosting
        gb_model = GradientBoostingRegressor(n_estimators=100, random_state=42)
        gb_model.fit(X_train_scaled, y_train)
        gb_pred = gb_model.predict(X_test_scaled)
        gb_mae = mean_absolute_error(y_test, gb_pred)
        gb_r2 = r2_score(y_test, gb_pred)
        print(f"   Gradient Boosting - MAE: ${gb_mae:.2f}, R²: {gb_r2:.4f}")
        
        # Select best model
        best_model = rf_model if rf_mae < min(lr_mae, gb_mae) else (gb_model if gb_mae < lr_mae else lr_model)
        best_pred = rf_pred if rf_mae < min(lr_mae, gb_mae) else (gb_pred if gb_mae < lr_mae else lr_pred)
        
        self.model = best_model
        
        # Model evaluation
        print("\n2️⃣  Best Model Performance (Random Forest):")
        print(f"   MAE: ${rf_mae:.2f}")
        print(f"   RMSE: ${np.sqrt(mean_squared_error(y_test, rf_pred)):.2f}")
        print(f"   R² Score: {rf_r2:.4f}")
        
        # Feature importance
        print("\n3️⃣  Top 5 Important Features:")
        importance = pd.DataFrame({
            'Feature': feature_cols,
            'Importance': rf_model.feature_importances_
        }).sort_values('Importance', ascending=False)
        
        for idx, (_, row) in enumerate(importance.head(5).iterrows(), 1):
            print(f"   {idx}. {row['Feature']}: {row['Importance']:.4f}")
        
        # Visualizations
        self._visualize_predictions(df_features['Date'].iloc[split_idx:], y_test, rf_pred)
        
        return rf_model, X_test_scaled, y_test, rf_pred
    
    def _visualize_predictions(self, dates, actual, predicted):
        """Visualize stock price predictions"""
        fig, axes = plt.subplots(2, 1, figsize=(14, 10))
        fig.suptitle('Stock Price Prediction Results', fontsize=16, fontweight='bold')
        
        # Predictions vs Actual
        axes[0].plot(dates, actual, label='Actual Price', linewidth=2, color='#378ADD', marker='o', markersize=3)
        axes[0].plot(dates, predicted, label='Predicted Price', linewidth=2, color='#D4537E', marker='s', markersize=3)
        axes[0].fill_between(dates, actual, predicted, alpha=0.2, color='gray')
        axes[0].set_title('Stock Price: Actual vs Predicted', fontweight='bold')
        axes[0].set_ylabel('Price ($)')
        axes[0].legend()
        axes[0].grid(True, alpha=0.3)
        
        # Residuals
        residuals = actual - predicted
        axes[1].scatter(dates, residuals, color='#639922', alpha=0.6, s=30)
        axes[1].axhline(y=0, color='red', linestyle='--', linewidth=2)
        axes[1].set_title('Prediction Residuals', fontweight='bold')
        axes[1].set_ylabel('Residual ($)')
        axes[1].set_xlabel('Date')
        axes[1].grid(True, alpha=0.3)
        
        plt.tight_layout()
        return fig


# ============================================================================
# 2. IRIS FLOWER CLASSIFICATION (CLASSIC ML STARTER)
# ============================================================================

class IrisClassification:
    """Classify iris flowers using multiple ML algorithms"""
    
    def __init__(self):
        self.models = {}
        self.scaler = StandardScaler()
        
    def load_data(self):
        """Load and prepare iris dataset"""
        iris = load_iris()
        X = iris.data
        y = iris.target
        
        df = pd.DataFrame(X, columns=iris.feature_names)
        df['Species'] = iris.target_names[y]
        
        return X, y, df, iris.feature_names, iris.target_names
    
    def train_models(self):
        """Train multiple classification models"""
        print("\n" + "="*60)
        print("IRIS FLOWER CLASSIFICATION")
        print("="*60)
        
        # Load data
        X, y, df, feature_names, target_names = self.load_data()
        
        # Data overview
        print("\n1️⃣  Dataset Overview:")
        print(f"   Total Samples: {len(X)}")
        print(f"   Features: {len(feature_names)}")
        print(f"   Classes: {len(target_names)}")
        print(f"   Class Distribution:")
        for target_name in target_names:
            count = (df['Species'] == target_name).sum()
            pct = count / len(df) * 100
            print(f"      {target_name}: {count} ({pct:.1f}%)")
        
        # Train-test split
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42, stratify=y
        )
        
        # Scaling
        X_train_scaled = self.scaler.fit_transform(X_train)
        X_test_scaled = self.scaler.transform(X_test)
        
        # Train models
        print("\n2️⃣  Model Performance:")
        
        results = {}
        
        # Logistic Regression
        lr_model = LogisticRegression(max_iter=1000, random_state=42)
        lr_model.fit(X_train_scaled, y_train)
        lr_pred = lr_model.predict(X_test_scaled)
        lr_acc = accuracy_score(y_test, lr_pred)
        results['Logistic Regression'] = lr_acc
        self.models['lr'] = lr_model
        print(f"   Logistic Regression - Accuracy: {lr_acc:.4f}")
        
        # Random Forest
        rf_model = RandomForestClassifier(n_estimators=100, random_state=42)
        rf_model.fit(X_train_scaled, y_train)
        rf_pred = rf_model.predict(X_test_scaled)
        rf_acc = accuracy_score(y_test, rf_pred)
        results['Random Forest'] = rf_acc
        self.models['rf'] = rf_model
        print(f"   Random Forest - Accuracy: {rf_acc:.4f}")
        
        # Get best model
        best_model_name = max(results, key=results.get)
        best_model = self.models['rf'] if best_model_name == 'Random Forest' else self.models['lr']
        best_pred = rf_pred if best_model_name == 'Random Forest' else lr_pred
        
        # Detailed metrics
        print(f"\n3️⃣  {best_model_name} Detailed Metrics:")
        print(f"   Precision: {precision_score(y_test, best_pred, average='weighted'):.4f}")
        print(f"   Recall: {recall_score(y_test, best_pred, average='weighted'):.4f}")
        print(f"   F1-Score: {f1_score(y_test, best_pred, average='weighted'):.4f}")
        
        # Classification report
        print(f"\n4️⃣  Classification Report:")
        print(classification_report(y_test, best_pred, target_names=target_names))
        
        # Feature importance
        print("\n5️⃣  Feature Importance (Random Forest):")
        importance = pd.DataFrame({
            'Feature': feature_names,
            'Importance': rf_model.feature_importances_
        }).sort_values('Importance', ascending=False)
        
        for idx, (_, row) in enumerate(importance.iterrows(), 1):
            print(f"   {idx}. {row['Feature']}: {row['Importance']:.4f}")
        
        # Visualizations
        self._visualize_iris(X_test, y_test, best_pred, feature_names, target_names)
        
        return best_model, X_test_scaled, y_test, best_pred, target_names
    
    def _visualize_iris(self, X_test, y_test, predictions, feature_names, target_names):
        """Visualize iris classification results"""
        fig, axes = plt.subplots(2, 2, figsize=(14, 10))
        fig.suptitle('Iris Flower Classification Results', fontsize=16, fontweight='bold')
        
        # Scatter plot: Sepal measurements
        colors = ['#378ADD', '#639922', '#D4537E']
        for i, target in enumerate(np.unique(y_test)):
            indices = y_test == target
            axes[0, 0].scatter(X_test[indices, 0], X_test[indices, 1], 
                             label=target_names[target], alpha=0.7, s=80, color=colors[i])
        axes[0, 0].set_xlabel(feature_names[0])
        axes[0, 0].set_ylabel(feature_names[1])
        axes[0, 0].set_title('Sepal Length vs Width', fontweight='bold')
        axes[0, 0].legend()
        axes[0, 0].grid(True, alpha=0.3)
        
        # Scatter plot: Petal measurements
        for i, target in enumerate(np.unique(y_test)):
            indices = y_test == target
            axes[0, 1].scatter(X_test[indices, 2], X_test[indices, 3], 
                             label=target_names[target], alpha=0.7, s=80, color=colors[i])
        axes[0, 1].set_xlabel(feature_names[2])
        axes[0, 1].set_ylabel(feature_names[3])
        axes[0, 1].set_title('Petal Length vs Width', fontweight='bold')
        axes[0, 1].legend()
        axes[0, 1].grid(True, alpha=0.3)
        
        # Confusion matrix
        cm = confusion_matrix(y_test, predictions)
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=axes[1, 0], 
                   xticklabels=target_names, yticklabels=target_names, cbar=False)
        axes[1, 0].set_title('Confusion Matrix', fontweight='bold')
        axes[1, 0].set_ylabel('Actual')
        axes[1, 0].set_xlabel('Predicted')
        
        # Feature importance
        feature_importance = pd.DataFrame({
            'Feature': [f.split(' ')[0] for f in feature_names],
            'Importance': self.models['rf'].feature_importances_
        }).sort_values('Importance', ascending=True)
        
        axes[1, 1].barh(feature_importance['Feature'], feature_importance['Importance'], color='#45B7D1')
        axes[1, 1].set_title('Feature Importance', fontweight='bold')
        axes[1, 1].set_xlabel('Importance Score')
        
        plt.tight_layout()
        return fig


# ============================================================================
# 3. HOUSING PRICE PREDICTION (REGRESSION)
# ============================================================================

class HousingPricePrediction:
    """Predict house prices using regression models"""
    
    def __init__(self):
        self.model = None
        self.scaler = StandardScaler()
        
    def generate_housing_data(self, samples=500):
        """Generate synthetic housing dataset"""
        np.random.seed(42)
        
        df = pd.DataFrame({
            'Square_Feet': np.random.uniform(800, 5000, samples),
            'Bedrooms': np.random.randint(1, 6, samples),
            'Bathrooms': np.random.randint(1, 4, samples),
            'Age_Years': np.random.randint(0, 100, samples),
            'Garage_Spaces': np.random.randint(0, 4, samples),
            'Distance_to_City': np.random.uniform(0, 50, samples)
        })
        
        # Generate prices based on features
        price = (
            150 * df['Square_Feet'] +
            50000 * df['Bedrooms'] +
            30000 * df['Bathrooms'] -
            500 * df['Age_Years'] +
            10000 * df['Garage_Spaces'] -
            2000 * df['Distance_to_City'] +
            np.random.normal(0, 50000, samples)
        )
        
        df['Price'] = np.maximum(price, 100000)  # Ensure positive prices
        
        return df
    
    def train_model(self, df=None):
        """Train housing price prediction model"""
        print("\n" + "="*60)
        print("HOUSING PRICE PREDICTION")
        print("="*60)
        
        if df is None:
            df = self.generate_housing_data()
        
        # Data overview
        print("\n1️⃣  Dataset Overview:")
        print(f"   Total Properties: {len(df)}")
        print(f"   Price Range: ${df['Price'].min():,.0f} - ${df['Price'].max():,.0f}")
        print(f"   Average Price: ${df['Price'].mean():,.0f}")
        print(f"\n2️⃣  Feature Correlation with Price:")
        
        correlations = df.corr()['Price'].sort_values(ascending=False)
        for feature, corr in correlations.head(7).items():
            if feature != 'Price':
                print(f"   {feature}: {corr:.4f}")
        
        # Prepare data
        X = df.drop('Price', axis=1)
        y = df['Price']
        
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42
        )
        
        # Scaling
        X_train_scaled = self.scaler.fit_transform(X_train)
        X_test_scaled = self.scaler.transform(X_test)
        
        # Train models
        print("\n3️⃣  Model Performance:")
        
        # Linear Regression
        lr_model = LinearRegression()
        lr_model.fit(X_train_scaled, y_train)
        lr_pred = lr_model.predict(X_test_scaled)
        lr_mae = mean_absolute_error(y_test, lr_pred)
        lr_r2 = r2_score(y_test, lr_pred)
        print(f"   Linear Regression - MAE: ${lr_mae:,.0f}, R²: {lr_r2:.4f}")
        
        # Random Forest
        rf_model = RandomForestRegressor(n_estimators=100, random_state=42, n_jobs=-1)
        rf_model.fit(X_train_scaled, y_train)
        rf_pred = rf_model.predict(X_test_scaled)
        rf_mae = mean_absolute_error(y_test, rf_pred)
        rf_r2 = r2_score(y_test, rf_pred)
        print(f"   Random Forest - MAE: ${rf_mae:,.0f}, R²: {rf_r2:.4f}")
        
        # Gradient Boosting
        gb_model = GradientBoostingRegressor(n_estimators=100, random_state=42)
        gb_model.fit(X_train_scaled, y_train)
        gb_pred = gb_model.predict(X_test_scaled)
        gb_mae = mean_absolute_error(y_test, gb_pred)
        gb_r2 = r2_score(y_test, gb_pred)
        print(f"   Gradient Boosting - MAE: ${gb_mae:,.0f}, R²: {gb_r2:.4f}")
        
        # Select best model
        best_model = gb_model if gb_mae < min(lr_mae, rf_mae) else (rf_model if rf_mae < lr_mae else lr_model)
        best_pred = gb_pred if gb_mae < min(lr_mae, rf_mae) else (rf_pred if rf_mae < lr_mae else lr_pred)
        best_name = 'Gradient Boosting' if gb_mae < min(lr_mae, rf_mae) else ('Random Forest' if rf_mae < lr_mae else 'Linear Regression')
        
        self.model = best_model
        
        # Feature importance
        print(f"\n4️⃣  Feature Importance ({best_name}):")
        if hasattr(best_model, 'feature_importances_'):
            importance = pd.DataFrame({
                'Feature': X.columns,
                'Importance': best_model.feature_importances_
            }).sort_values('Importance', ascending=False)
            
            for idx, (_, row) in enumerate(importance.iterrows(), 1):
                print(f"   {idx}. {row['Feature']}: {row['Importance']:.4f}")
        else:
            coef = pd.DataFrame({
                'Feature': X.columns,
                'Coefficient': best_model.coef_
            }).sort_values('Coefficient', ascending=False, key=abs)
            
            for idx, (_, row) in enumerate(coef.iterrows(), 1):
                print(f"   {idx}. {row['Feature']}: {row['Coefficient']:,.0f}")
        
        # Visualizations
        self._visualize_housing(y_test, best_pred, X_test)
        
        return best_model, X_test_scaled, y_test, best_pred
    
    def _visualize_housing(self, y_test, predictions, X_test):
        """Visualize housing price predictions"""
        fig, axes = plt.subplots(2, 2, figsize=(14, 10))
        fig.suptitle('Housing Price Prediction Results', fontsize=16, fontweight='bold')
        
        # Actual vs Predicted
        axes[0, 0].scatter(y_test, predictions, alpha=0.6, s=50, color='#378ADD')
        min_val = min(y_test.min(), predictions.min())
        max_val = max(y_test.max(), predictions.max())
        axes[0, 0].plot([min_val, max_val], [min_val, max_val], 'r--', linewidth=2, label='Perfect Prediction')
        axes[0, 0].set_xlabel('Actual Price ($)')
        axes[0, 0].set_ylabel('Predicted Price ($)')
        axes[0, 0].set_title('Actual vs Predicted Prices', fontweight='bold')
        axes[0, 0].legend()
        axes[0, 0].grid(True, alpha=0.3)
        
        # Residuals
        residuals = y_test - predictions
        axes[0, 1].scatter(predictions, residuals, alpha=0.6, s=50, color='#639922')
        axes[0, 1].axhline(y=0, color='red', linestyle='--', linewidth=2)
        axes[0, 1].set_xlabel('Predicted Price ($)')
        axes[0, 1].set_ylabel('Residual ($)')
        axes[0, 1].set_title('Residual Plot', fontweight='bold')
        axes[0, 1].grid(True, alpha=0.3)
        
        # Residual distribution
        axes[1, 0].hist(residuals, bins=30, color='#D4537E', edgecolor='black', alpha=0.7)
        axes[1, 0].set_xlabel('Residual ($)')
        axes[1, 0].set_ylabel('Frequency')
        axes[1, 0].set_title('Residual Distribution', fontweight='bold')
        axes[1, 0].axvline(residuals.mean(), color='red', linestyle='--', linewidth=2, label=f'Mean: ${residuals.mean():,.0f}')
        axes[1, 0].legend()
        
        # Square footage vs price
        axes[1, 1].scatter(X_test['Square_Feet'], y_test, alpha=0.5, s=50, color='#378ADD', label='Actual')
        axes[1, 1].scatter(X_test['Square_Feet'], predictions, alpha=0.5, s=50, color='#D4537E', label='Predicted')
        axes[1, 1].set_xlabel('Square Feet')
        axes[1, 1].set_ylabel('Price ($)')
        axes[1, 1].set_title('Price vs Square Footage', fontweight='bold')
        axes[1, 1].legend()
        axes[1, 1].grid(True, alpha=0.3)
        
        plt.tight_layout()
        return fig


# ============================================================================
# 4. SENTIMENT ANALYSIS (NLP)
# ============================================================================

class SentimentAnalysis:
    """Sentiment analysis on movie/product reviews"""
    
    def __init__(self):
        self.model = None
        self.vectorizer = None
        self.scaler = StandardScaler()
        
    def generate_review_data(self, samples=500):
        """Generate synthetic review dataset"""
        np.random.seed(42)
        
        # Sample reviews
        positive_words = ['excellent', 'amazing', 'great', 'wonderful', 'fantastic', 
                         'loved', 'best', 'outstanding', 'brilliant', 'perfect']
        negative_words = ['terrible', 'awful', 'bad', 'horrible', 'worst', 
                         'disappointed', 'waste', 'boring', 'useless', 'poor']
        neutral_words = ['okay', 'fine', 'average', 'decent', 'alright', 'normal']
        
        reviews = []
        sentiments = []
        
        for _ in range(samples):
            sentiment = np.random.choice(['Positive', 'Negative', 'Neutral'])
            sentiments.append(sentiment)
            
            if sentiment == 'Positive':
                num_words = np.random.randint(5, 15)
                review = ' '.join(np.random.choice(positive_words, num_words))
            elif sentiment == 'Negative':
                num_words = np.random.randint(5, 15)
                review = ' '.join(np.random.choice(negative_words, num_words))
            else:
                num_words = np.random.randint(5, 15)
                review = ' '.join(np.random.choice(neutral_words, num_words))
            
            reviews.append(review)
        
        return pd.DataFrame({'Review': reviews, 'Sentiment': sentiments})
    
    def extract_features(self, texts):
        """Extract simple text features"""
        features = []
        
        for text in texts:
            words = text.lower().split()
            
            positive_words = ['excellent', 'amazing', 'great', 'wonderful', 'fantastic', 
                            'loved', 'best', 'outstanding', 'brilliant', 'perfect']
            negative_words = ['terrible', 'awful', 'bad', 'horrible', 'worst', 
                            'disappointed', 'waste', 'boring', 'useless', 'poor']
            
            pos_count = sum(1 for word in words if word in positive_words)
            neg_count = sum(1 for word in words if word in negative_words)
            text_length = len(words)
            
            features.append([pos_count, neg_count, text_length])
        
        return np.array(features)
    
    def train_model(self, df=None):
        """Train sentiment analysis model"""
        print("\n" + "="*60)
        print("SENTIMENT ANALYSIS")
        print("="*60)
        
        if df is None:
            df = self.generate_review_data()
        
        # Data overview
        print("\n1️⃣  Dataset Overview:")
        print(f"   Total Reviews: {len(df)}")
        print(f"   Sentiment Distribution:")
        for sentiment in df['Sentiment'].unique():
            count = (df['Sentiment'] == sentiment).sum()
            pct = count / len(df) * 100
            print(f"      {sentiment}: {count} ({pct:.1f}%)")
        
        # Feature extraction
        X = self.extract_features(df['Review'])
        
        # Encode labels
        label_encoder = LabelEncoder()
        y = label_encoder.fit_transform(df['Sentiment'])
        
        # Train-test split
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42, stratify=y
        )
        
        # Scaling
        X_train_scaled = self.scaler.fit_transform(X_train)
        X_test_scaled = self.scaler.transform(X_test)
        
        # Train models
        print("\n2️⃣  Model Performance:")
        
        # Logistic Regression
        lr_model = LogisticRegression(max_iter=1000, random_state=42)
        lr_model.fit(X_train_scaled, y_train)
        lr_pred = lr_model.predict(X_test_scaled)
        lr_acc = accuracy_score(y_test, lr_pred)
        print(f"   Logistic Regression - Accuracy: {lr_acc:.4f}")
        
        # Random Forest
        rf_model = RandomForestClassifier(n_estimators=100, random_state=42)
        rf_model.fit(X_train_scaled, y_train)
        rf_pred = rf_model.predict(X_test_scaled)
        rf_acc = accuracy_score(y_test, rf_pred)
        print(f"   Random Forest - Accuracy: {rf_acc:.4f}")
        
        # Select best model
        best_model = rf_model if rf_acc > lr_acc else lr_model
        best_pred = rf_pred if rf_acc > lr_acc else lr_pred
        best_name = 'Random Forest' if rf_acc > lr_acc else 'Logistic Regression'
        
        self.model = best_model
        
        # Detailed metrics
        print(f"\n3️⃣  {best_name} Detailed Metrics:")
        print(f"   Accuracy: {accuracy_score(y_test, best_pred):.4f}")
        print(f"   Precision (weighted): {precision_score(y_test, best_pred, average='weighted'):.4f}")
        print(f"   Recall (weighted): {recall_score(y_test, best_pred, average='weighted'):.4f}")
        print(f"   F1-Score (weighted): {f1_score(y_test, best_pred, average='weighted'):.4f}")
        
        # Classification report
        print(f"\n4️⃣  Classification Report:")
        print(classification_report(y_test, best_pred, target_names=label_encoder.classes_))
        
        # Feature importance
        print("\n5️⃣  Feature Importance:")
        features = ['Positive Words', 'Negative Words', 'Text Length']
        if hasattr(best_model, 'feature_importances_'):
            importance = best_model.feature_importances_
        else:
            importance = np.abs(best_model.coef_[0])
        
        importance_df = pd.DataFrame({
            'Feature': features,
            'Importance': importance
        }).sort_values('Importance', ascending=False)
        
        for idx, (_, row) in enumerate(importance_df.iterrows(), 1):
            print(f"   {idx}. {row['Feature']}: {row['Importance']:.4f}")
        
        # Visualizations
        self._visualize_sentiment(y_test, best_pred, label_encoder, importance_df)
        
        return best_model, X_test_scaled, y_test, best_pred, label_encoder
    
    def _visualize_sentiment(self, y_test, predictions, label_encoder, importance_df):
        """Visualize sentiment analysis results"""
        fig, axes = plt.subplots(2, 2, figsize=(14, 10))
        fig.suptitle('Sentiment Analysis Results', fontsize=16, fontweight='bold')
        
        # Confusion matrix
        cm = confusion_matrix(y_test, predictions)
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=axes[0, 0],
                   xticklabels=label_encoder.classes_,
                   yticklabels=label_encoder.classes_, cbar=False)
        axes[0, 0].set_title('Confusion Matrix', fontweight='bold')
        axes[0, 0].set_ylabel('Actual')
        axes[0, 0].set_xlabel('Predicted')
        
        # Feature importance
        axes[0, 1].barh(importance_df['Feature'], importance_df['Importance'], color='#45B7D1')
        axes[0, 1].set_title('Feature Importance', fontweight='bold')
        axes[0, 1].set_xlabel('Importance Score')
        
        # Sentiment distribution
        unique, counts = np.unique(y_test, return_counts=True)
        colors = ['#378ADD', '#639922', '#D4537E']
        axes[1, 0].bar(label_encoder.classes_, [counts[i] for i in range(len(label_encoder.classes_))], color=colors)
        axes[1, 0].set_title('Test Set Sentiment Distribution', fontweight='bold')
        axes[1, 0].set_ylabel('Count')
        
        # Prediction distribution
        pred_unique, pred_counts = np.unique(predictions, return_counts=True)
        axes[1, 1].bar(label_encoder.classes_, [pred_counts[i] if i < len(pred_counts) else 0 for i in range(len(label_encoder.classes_))], color=colors)
        axes[1, 1].set_title('Model Prediction Distribution', fontweight='bold')
        axes[1, 1].set_ylabel('Count')
        
        plt.tight_layout()
        return fig


# ============================================================================
# MAIN EXECUTION
# ============================================================================

def run_all_models():
    """Run all machine learning models"""
    
    print("\n" + "="*70)
    print("🤖 SIMPLE MACHINE LEARNING MODELS TOOLKIT")
    print("="*70)
    
    # 1. Stock Price Prediction
    stock = StockPricePrediction()
    stock_data = stock.generate_stock_data()
    stock_model, X_stock_test, y_stock_test, stock_pred = stock.train_model(stock_data)
    plt.savefig('/home/claude/ml_stock_prediction.png', dpi=150, bbox_inches='tight')
    plt.close()
    
    # 2. Iris Classification
    iris = IrisClassification()
    iris_model, X_iris_test, y_iris_test, iris_pred, iris_targets = iris.train_models()
    plt.savefig('/home/claude/ml_iris_classification.png', dpi=150, bbox_inches='tight')
    plt.close()
    
    # 3. Housing Price Prediction
    housing = HousingPricePrediction()
    housing_model, X_housing_test, y_housing_test, housing_pred = housing.train_model()
    plt.savefig('/home/claude/ml_housing_prices.png', dpi=150, bbox_inches='tight')
    plt.close()
    
    # 4. Sentiment Analysis
    sentiment = SentimentAnalysis()
    sentiment_data = sentiment.generate_review_data()
    sentiment_model, X_sentiment_test, y_sentiment_test, sentiment_pred, sentiment_labels = sentiment.train_model(sentiment_data)
    plt.savefig('/home/claude/ml_sentiment_analysis.png', dpi=150, bbox_inches='tight')
    plt.close()
    
    print("\n" + "="*70)
    print("✅ ALL MODELS TRAINED SUCCESSFULLY!")
    print("="*70)
    print("\nGenerated Visualizations:")
    print("  📈 ml_stock_prediction.png")
    print("  🌸 ml_iris_classification.png")
    print("  🏠 ml_housing_prices.png")
    print("  💬 ml_sentiment_analysis.png")
    print("\n" + "="*70)


if __name__ == "__main__":
    run_all_models()
