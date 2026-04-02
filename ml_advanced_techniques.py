"""
Advanced Machine Learning Techniques
Hyperparameter Tuning, Cross-Validation, and Model Evaluation
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import (
    cross_val_score, GridSearchCV, RandomizedSearchCV, KFold, StratifiedKFold
)
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier, GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.datasets import load_iris, load_wine
from sklearn.metrics import (
    confusion_matrix, classification_report,
    roc_curve, auc, precision_recall_curve, f1_score, accuracy_score
)
from sklearn.model_selection import learning_curve, validation_curve
import warnings
warnings.filterwarnings('ignore')

# ============================================================================
# 1. HYPERPARAMETER TUNING WITH GRIDSEARCH
# ============================================================================

class HyperparameterTuning:
    """Demonstrate hyperparameter tuning techniques"""
    
    @staticmethod
    def grid_search_example():
        """GridSearchCV for exhaustive parameter search"""
        print("\n" + "="*70)
        print("HYPERPARAMETER TUNING - GRID SEARCH")
        print("="*70)
        
        # Load data
        iris = load_iris()
        X, y = iris.data, iris.target
        
        X_train, X_test = X[:120], X[120:]
        y_train, y_test = y[:120], y[120:]
        
        # Scale features
        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train)
        X_test_scaled = scaler.transform(X_test)
        
        print("\n1️⃣  Tuning Random Forest Classifier:")
        print("   Parameters to tune: n_estimators, max_depth, min_samples_split")
        
        # Define parameter grid
        param_grid = {
            'n_estimators': [50, 100, 200],
            'max_depth': [3, 5, 10, None],
            'min_samples_split': [2, 5, 10]
        }
        
        # GridSearchCV
        rf = RandomForestClassifier(random_state=42)
        grid_search = GridSearchCV(rf, param_grid, cv=5, n_jobs=-1, scoring='accuracy')
        grid_search.fit(X_train_scaled, y_train)
        
        print(f"\n2️⃣  Best Parameters Found:")
        for param, value in grid_search.best_params_.items():
            print(f"   {param}: {value}")
        
        print(f"\n3️⃣  Best Cross-Validation Score: {grid_search.best_score_:.4f}")
        
        # Test set performance
        y_pred = grid_search.predict(X_test_scaled)
        test_accuracy = accuracy_score(y_test, y_pred)
        print(f"\n4️⃣  Test Set Accuracy: {test_accuracy:.4f}")
        
        # Show top 5 parameter combinations
        results_df = pd.DataFrame(grid_search.cv_results_)
        top_results = results_df.nlargest(5, 'mean_test_score')[
            ['param_n_estimators', 'param_max_depth', 'param_min_samples_split', 'mean_test_score']
        ]
        
        print("\n5️⃣  Top 5 Parameter Combinations:")
        for idx, (_, row) in enumerate(top_results.iterrows(), 1):
            print(f"   {idx}. n_est={int(row['param_n_estimators'])}, "
                  f"depth={row['param_max_depth']}, "
                  f"min_split={int(row['param_min_samples_split'])} "
                  f"→ Score: {row['mean_test_score']:.4f}")
        
        return grid_search, X_test_scaled, y_test
    
    @staticmethod
    def random_search_example():
        """RandomizedSearchCV for faster parameter search"""
        print("\n" + "="*70)
        print("HYPERPARAMETER TUNING - RANDOM SEARCH")
        print("="*70)
        
        # Load data
        wine = load_wine()
        X, y = wine.data, wine.target
        
        X_train, X_test = X[:150], X[150:]
        y_train, y_test = y[:150], y[150:]
        
        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train)
        X_test_scaled = scaler.transform(X_test)
        
        print("\n1️⃣  Tuning Gradient Boosting Classifier:")
        
        # Parameter distributions
        param_dist = {
            'n_estimators': [50, 100, 150, 200, 300],
            'learning_rate': [0.01, 0.05, 0.1, 0.15],
            'max_depth': [2, 3, 4, 5, 7],
            'min_samples_split': [2, 5, 10, 15],
            'subsample': [0.6, 0.8, 1.0]
        }
        
        # RandomizedSearchCV (faster than GridSearch)
        gb = GradientBoostingClassifier(random_state=42)
        random_search = RandomizedSearchCV(
            gb, param_dist, n_iter=20, cv=5, n_jobs=-1, scoring='accuracy', random_state=42
        )
        random_search.fit(X_train_scaled, y_train)
        
        print(f"\n2️⃣  Best Parameters Found:")
        for param, value in random_search.best_params_.items():
            print(f"   {param}: {value}")
        
        print(f"\n3️⃣  Best Cross-Validation Score: {random_search.best_score_:.4f}")
        
        y_pred = random_search.predict(X_test_scaled)
        test_accuracy = accuracy_score(y_test, y_pred)
        print(f"\n4️⃣  Test Set Accuracy: {test_accuracy:.4f}")
        
        return random_search, X_test_scaled, y_test


# ============================================================================
# 2. CROSS-VALIDATION STRATEGIES
# ============================================================================

class CrossValidationAnalysis:
    """Demonstrate different cross-validation techniques"""
    
    @staticmethod
    def k_fold_analysis():
        """K-Fold Cross-Validation"""
        print("\n" + "="*70)
        print("K-FOLD CROSS-VALIDATION")
        print("="*70)
        
        iris = load_iris()
        X, y = iris.data, iris.target
        
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X)
        
        print("\n1️⃣  K-Fold with different K values:")
        
        for k in [3, 5, 10]:
            kfold = KFold(n_splits=k, shuffle=True, random_state=42)
            
            rf = RandomForestClassifier(n_estimators=100, random_state=42)
            cv_scores = cross_val_score(rf, X_scaled, y, cv=kfold, scoring='accuracy')
            
            print(f"\n   K={k}:")
            print(f"      Scores: {[f'{score:.4f}' for score in cv_scores]}")
            print(f"      Mean: {cv_scores.mean():.4f} (+/- {cv_scores.std():.4f})")
        
        return cv_scores
    
    @staticmethod
    def stratified_k_fold_analysis():
        """Stratified K-Fold for imbalanced datasets"""
        print("\n" + "="*70)
        print("STRATIFIED K-FOLD CROSS-VALIDATION")
        print("="*70)
        
        iris = load_iris()
        X, y = iris.data, iris.target
        
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X)
        
        # Create imbalanced dataset (simulate real-world scenario)
        imbalanced_idx = np.concatenate([np.where(y == 0)[0], np.where(y == 1)[0][:10], np.where(y == 2)[0][:5]])
        X_imbalanced = X_scaled[imbalanced_idx]
        y_imbalanced = y[imbalanced_idx]
        
        print("\n1️⃣  Dataset Class Distribution:")
        for cls in np.unique(y_imbalanced):
            count = np.sum(y_imbalanced == cls)
            pct = count / len(y_imbalanced) * 100
            print(f"   Class {cls}: {count} ({pct:.1f}%)")
        
        print("\n2️⃣  Stratified K-Fold (5 splits):")
        
        skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
        rf = RandomForestClassifier(n_estimators=100, random_state=42)
        cv_scores = cross_val_score(rf, X_imbalanced, y_imbalanced, cv=skf, scoring='f1_weighted')
        
        print(f"   Scores: {[f'{score:.4f}' for score in cv_scores]}")
        print(f"   Mean: {cv_scores.mean():.4f} (+/- {cv_scores.std():.4f})")
        
        return cv_scores


# ============================================================================
# 3. LEARNING CURVES
# ============================================================================

class LearningCurveAnalysis:
    """Analyze model learning with increasing training data"""
    
    @staticmethod
    def plot_learning_curves():
        """Plot learning curves to diagnose bias/variance"""
        print("\n" + "="*70)
        print("LEARNING CURVES ANALYSIS")
        print("="*70)
        
        iris = load_iris()
        X, y = iris.data, iris.target
        
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X)
        
        print("\n1️⃣  Training models with increasing dataset size...")
        
        fig, axes = plt.subplots(1, 2, figsize=(14, 5))
        fig.suptitle('Learning Curves: Bias-Variance Analysis', fontsize=14, fontweight='bold')
        
        # Simple model (high bias)
        lr = LogisticRegression(max_iter=1000, random_state=42)
        train_sizes, train_scores_lr, val_scores_lr = learning_curve(
            lr, X_scaled, y, cv=5, train_sizes=np.linspace(0.1, 1.0, 10), 
            scoring='accuracy', n_jobs=-1
        )
        
        # Complex model (high variance)
        rf = RandomForestClassifier(n_estimators=200, max_depth=10, random_state=42)
        train_sizes, train_scores_rf, val_scores_rf = learning_curve(
            rf, X_scaled, y, cv=5, train_sizes=np.linspace(0.1, 1.0, 10),
            scoring='accuracy', n_jobs=-1
        )
        
        # Plot learning curves
        train_mean_lr = np.mean(train_scores_lr, axis=1)
        train_std_lr = np.std(train_scores_lr, axis=1)
        val_mean_lr = np.mean(val_scores_lr, axis=1)
        val_std_lr = np.std(val_scores_lr, axis=1)
        
        axes[0].plot(train_sizes, train_mean_lr, 'o-', color='#378ADD', label='Training Score', linewidth=2)
        axes[0].fill_between(train_sizes, train_mean_lr - train_std_lr, 
                            train_mean_lr + train_std_lr, alpha=0.2, color='#378ADD')
        axes[0].plot(train_sizes, val_mean_lr, 'o-', color='#D4537E', label='Validation Score', linewidth=2)
        axes[0].fill_between(train_sizes, val_mean_lr - val_std_lr, 
                            val_mean_lr + val_std_lr, alpha=0.2, color='#D4537E')
        axes[0].set_xlabel('Training Set Size')
        axes[0].set_ylabel('Accuracy')
        axes[0].set_title('Logistic Regression (High Bias)', fontweight='bold')
        axes[0].legend()
        axes[0].grid(True, alpha=0.3)
        
        train_mean_rf = np.mean(train_scores_rf, axis=1)
        train_std_rf = np.std(train_scores_rf, axis=1)
        val_mean_rf = np.mean(val_scores_rf, axis=1)
        val_std_rf = np.std(val_scores_rf, axis=1)
        
        axes[1].plot(train_sizes, train_mean_rf, 'o-', color='#378ADD', label='Training Score', linewidth=2)
        axes[1].fill_between(train_sizes, train_mean_rf - train_std_rf, 
                            train_mean_rf + train_std_rf, alpha=0.2, color='#378ADD')
        axes[1].plot(train_sizes, val_mean_rf, 'o-', color='#D4537E', label='Validation Score', linewidth=2)
        axes[1].fill_between(train_sizes, val_mean_rf - val_std_rf, 
                            val_mean_rf + val_std_rf, alpha=0.2, color='#D4537E')
        axes[1].set_xlabel('Training Set Size')
        axes[1].set_ylabel('Accuracy')
        axes[1].set_title('Random Forest (High Variance)', fontweight='bold')
        axes[1].legend()
        axes[1].grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        print("\n2️⃣  Analysis:")
        print(f"   Logistic Regression - Final Val Score: {val_mean_lr[-1]:.4f}")
        print(f"   Random Forest - Final Val Score: {val_mean_rf[-1]:.4f}")
        
        return fig


# ============================================================================
# 4. VALIDATION CURVES
# ============================================================================

class ValidationCurveAnalysis:
    """Analyze model performance as hyperparameters vary"""
    
    @staticmethod
    def plot_validation_curves():
        """Plot validation curves for hyperparameter selection"""
        print("\n" + "="*70)
        print("VALIDATION CURVES ANALYSIS")
        print("="*70)
        
        iris = load_iris()
        X, y = iris.data, iris.target
        
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X)
        
        print("\n1️⃣  Analyzing impact of max_depth on Random Forest...")
        
        fig, axes = plt.subplots(1, 2, figsize=(14, 5))
        fig.suptitle('Validation Curves: Hyperparameter Impact', fontsize=14, fontweight='bold')
        
        # Validation curve for max_depth
        param_range = range(1, 11)
        rf = RandomForestClassifier(n_estimators=100, random_state=42)
        
        train_scores, val_scores = validation_curve(
            rf, X_scaled, y, param_name='max_depth', param_range=param_range,
            cv=5, scoring='accuracy', n_jobs=-1
        )
        
        train_mean = np.mean(train_scores, axis=1)
        train_std = np.std(train_scores, axis=1)
        val_mean = np.mean(val_scores, axis=1)
        val_std = np.std(val_scores, axis=1)
        
        axes[0].plot(param_range, train_mean, 'o-', color='#378ADD', label='Training Score', linewidth=2)
        axes[0].fill_between(param_range, train_mean - train_std, train_mean + train_std, alpha=0.2, color='#378ADD')
        axes[0].plot(param_range, val_mean, 'o-', color='#D4537E', label='Validation Score', linewidth=2)
        axes[0].fill_between(param_range, val_mean - val_std, val_mean + val_std, alpha=0.2, color='#D4537E')
        axes[0].set_xlabel('max_depth')
        axes[0].set_ylabel('Accuracy')
        axes[0].set_title('Impact of max_depth on Performance', fontweight='bold')
        axes[0].legend()
        axes[0].grid(True, alpha=0.3)
        
        # Validation curve for n_estimators
        print("\n2️⃣  Analyzing impact of n_estimators on Random Forest...")
        
        param_range_ne = [10, 25, 50, 75, 100, 150, 200, 250]
        train_scores_ne, val_scores_ne = validation_curve(
            rf, X_scaled, y, param_name='n_estimators', param_range=param_range_ne,
            cv=5, scoring='accuracy', n_jobs=-1
        )
        
        train_mean_ne = np.mean(train_scores_ne, axis=1)
        train_std_ne = np.std(train_scores_ne, axis=1)
        val_mean_ne = np.mean(val_scores_ne, axis=1)
        val_std_ne = np.std(val_scores_ne, axis=1)
        
        axes[1].plot(param_range_ne, train_mean_ne, 'o-', color='#378ADD', label='Training Score', linewidth=2)
        axes[1].fill_between(param_range_ne, train_mean_ne - train_std_ne, 
                            train_mean_ne + train_std_ne, alpha=0.2, color='#378ADD')
        axes[1].plot(param_range_ne, val_mean_ne, 'o-', color='#D4537E', label='Validation Score', linewidth=2)
        axes[1].fill_between(param_range_ne, val_mean_ne - val_std_ne, 
                            val_mean_ne + val_std_ne, alpha=0.2, color='#D4537E')
        axes[1].set_xlabel('n_estimators')
        axes[1].set_ylabel('Accuracy')
        axes[1].set_title('Impact of n_estimators on Performance', fontweight='bold')
        axes[1].legend()
        axes[1].grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        print(f"   Best max_depth: {param_range[np.argmax(val_mean)]}")
        print(f"   Best val_mean @ best depth: {np.max(val_mean):.4f}")
        
        return fig


# ============================================================================
# 5. ROC CURVES AND PROBABILITY CALIBRATION
# ============================================================================

class ROCAndCalibration:
    """ROC curves and probability calibration analysis"""
    
    @staticmethod
    def plot_roc_curves():
        """Plot ROC curves for multi-class classification"""
        print("\n" + "="*70)
        print("ROC CURVES AND AUC ANALYSIS")
        print("="*70)
        
        iris = load_iris()
        X, y = iris.data, iris.target
        
        # Binary classification (class 0 vs rest)
        y_binary = (y == 0).astype(int)
        
        X_train, X_test = X[:120], X[120:]
        y_train, y_test = y_binary[:120], y_binary[120:]
        
        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train)
        X_test_scaled = scaler.transform(X_test)
        
        # Train models
        lr = LogisticRegression(max_iter=1000, random_state=42)
        lr.fit(X_train_scaled, y_train)
        
        rf = RandomForestClassifier(n_estimators=100, random_state=42)
        rf.fit(X_train_scaled, y_train)
        
        # Get probabilities
        y_pred_lr = lr.predict_proba(X_test_scaled)[:, 1]
        y_pred_rf = rf.predict_proba(X_test_scaled)[:, 1]
        
        # Calculate ROC curves
        fpr_lr, tpr_lr, _ = roc_curve(y_test, y_pred_lr)
        auc_lr = auc(fpr_lr, tpr_lr)
        
        fpr_rf, tpr_rf, _ = roc_curve(y_test, y_pred_rf)
        auc_rf = auc(fpr_rf, tpr_rf)
        
        print("\n1️⃣  ROC Curve Results (Binary Classification - Class 0 vs Rest):")
        print(f"   Logistic Regression AUC: {auc_lr:.4f}")
        print(f"   Random Forest AUC: {auc_rf:.4f}")
        
        # Plot
        fig, ax = plt.subplots(figsize=(10, 8))
        
        ax.plot(fpr_lr, tpr_lr, 'o-', color='#378ADD', label=f'LR (AUC={auc_lr:.4f})', linewidth=2)
        ax.plot(fpr_rf, tpr_rf, 's-', color='#D4537E', label=f'RF (AUC={auc_rf:.4f})', linewidth=2)
        ax.plot([0, 1], [0, 1], 'k--', label='Random Classifier', linewidth=1)
        
        ax.set_xlabel('False Positive Rate')
        ax.set_ylabel('True Positive Rate')
        ax.set_title('ROC Curves: Binary Classification', fontweight='bold', fontsize=14)
        ax.legend(loc='lower right')
        ax.grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        return fig, auc_lr, auc_rf


# ============================================================================
# MAIN EXECUTION
# ============================================================================

def run_advanced_ml():
    """Run all advanced ML techniques"""
    
    print("\n" + "="*70)
    print("🚀 ADVANCED MACHINE LEARNING TECHNIQUES")
    print("="*70)
    
    # 1. Hyperparameter Tuning
    print("\n" + "▶"*35)
    gs_model, X_gs_test, y_gs_test = HyperparameterTuning.grid_search_example()
    
    print("\n" + "▶"*35)
    rs_model, X_rs_test, y_rs_test = HyperparameterTuning.random_search_example()
    
    # 2. Cross-Validation
    print("\n" + "▶"*35)
    kfold_scores = CrossValidationAnalysis.k_fold_analysis()
    
    print("\n" + "▶"*35)
    skfold_scores = CrossValidationAnalysis.stratified_k_fold_analysis()
    
    # 3. Learning Curves
    print("\n" + "▶"*35)
    lc_fig = LearningCurveAnalysis.plot_learning_curves()
    plt.savefig('/home/claude/ml_learning_curves.png', dpi=150, bbox_inches='tight')
    plt.close()
    
    # 4. Validation Curves
    print("\n" + "▶"*35)
    vc_fig = ValidationCurveAnalysis.plot_validation_curves()
    plt.savefig('/home/claude/ml_validation_curves.png', dpi=150, bbox_inches='tight')
    plt.close()
    
    # 5. ROC Curves
    print("\n" + "▶"*35)
    roc_fig, auc_lr, auc_rf = ROCAndCalibration.plot_roc_curves()
    plt.savefig('/home/claude/ml_roc_curves.png', dpi=150, bbox_inches='tight')
    plt.close()
    
    print("\n" + "="*70)
    print("✅ ADVANCED ML ANALYSIS COMPLETE!")
    print("="*70)
    print("\nGenerated Visualizations:")
    print("  📊 ml_learning_curves.png")
    print("  📈 ml_validation_curves.png")
    print("  🎯 ml_roc_curves.png")
    print("\n" + "="*70)


if __name__ == "__main__":
    run_advanced_ml()
