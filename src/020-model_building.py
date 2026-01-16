# =============================================================================
# 02-MODEL_BUILDING.PY
# =============================================================================
# This script builds and trains a regression model using HistGradientBoosting
# with custom imputers and preprocessing pipeline.
#
# Sections:
#   1. IMPORTS & SETUP
#   2. LOAD DATA
#   3. CUSTOM IMPUTERS
#   4. PREPROCESSING PIPELINE
#   5. MODEL TRAINING
#   6. HYPERPARAMETER TUNING
#   7. FEATURE IMPORTANCE ANALYSIS
# =============================================================================

# =============================================================================
# 1. IMPORTS & SETUP
# =============================================================================
import os
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler, OrdinalEncoder
from sklearn.model_selection import train_test_split, KFold, learning_curve
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.experimental import enable_halving_search_cv
from sklearn.model_selection import HalvingRandomSearchCV
from sklearn.inspection import permutation_importance

# Set project root
script_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.dirname(script_dir)
os.chdir(project_root)

# Add src directory to Python path for custom modules
src_dir = Path(project_root) / 'src'
if str(src_dir) not in sys.path:
    sys.path.insert(0, str(src_dir))

# Create output directories
models_dir = Path(project_root) / 'models'
figures_dir = Path(project_root) / 'figures'
models_dir.mkdir(exist_ok=True)
figures_dir.mkdir(exist_ok=True)

# Import custom transformers
from __fn__ActivityTypeImputer import ActivityTypeImputer
from __fn__SportImputer import SportImputer
from __fn__AAV2020Encoder import AAV2020Encoder

# =============================================================================
# 2. LOAD DATA
# =============================================================================
print("=" * 60)
print("LOADING DATA")
print("=" * 60)

learn_df = pd.read_pickle("data/processed/learn_df.pkl")
test_df = pd.read_pickle("data/processed/test_df.pkl")

# Extract column groups for imputers
job_cols_in_df = [col for col in learn_df.columns if col.endswith('_current')]
retired_cols_in_df = [col for col in learn_df.columns if col.endswith('_retired')]
pension_cols_in_df = ['RETIREMENT_INCOME']
sport_cols_in_df = ['Sports', 'Categorie']

print(f"✓ Learn dataset: {learn_df.shape}")
print(f"✓ Test dataset: {test_df.shape}")

# =============================================================================
# 3. CUSTOM IMPUTERS
# =============================================================================
print("\n" + "=" * 60)
print("CREATING CUSTOM IMPUTERS")
print("=" * 60)

activity_imputer = ActivityTypeImputer(
    retired_cols=retired_cols_in_df,
    job_cols=job_cols_in_df,
    pension_cols=pension_cols_in_df
)

sport_imputer = SportImputer(sport_cols=sport_cols_in_df)
aav_encoder = AAV2020Encoder()

print("✓ ActivityTypeImputer created")
print("✓ SportImputer created")
print("✓ AAV2020Encoder created")

# =============================================================================
# 4. PREPROCESSING PIPELINE
# =============================================================================
print("\n" + "=" * 60)
print("BUILDING PREPROCESSING PIPELINE")
print("=" * 60)

# Identify column types
numeric_features = learn_df.select_dtypes(include=['int64', 'float64']).columns.tolist()
categorical_features = learn_df.select_dtypes(include=['object', 'category']).columns.tolist()

# Remove target and primary_key from features
for col in ['target', 'primary_key']:
    if col in numeric_features:
        numeric_features.remove(col)
    if col in categorical_features:
        categorical_features.remove(col)

force_categorical = ['CATEAVV2020', 'Categorie']
for col in force_categorical:
    if col in numeric_features:
        numeric_features.remove(col)
    if col in learn_df.columns and col not in categorical_features:
        categorical_features.append(col)

# Filter high-cardinality categorical features
MAX_CATEGORIES = 150
categorical_features_filtered = []
categorical_features_dropped = []

for col in categorical_features:
    if col == 'AAV2020':  # AAV2020 encoded specially
        categorical_features_filtered.append(col)
    elif learn_df[col].nunique() <= MAX_CATEGORIES:
        categorical_features_filtered.append(col)
    else:
        categorical_features_dropped.append((col, learn_df[col].nunique()))

print(f"✓ Numeric features: {len(numeric_features)}")
print(f"✓ Categorical features kept (≤ {MAX_CATEGORIES}): {len(categorical_features_filtered)}")
print(f"✗ Categorical features dropped: {len(categorical_features_dropped)}")

# Build preprocessing transformers
numeric_transformer = Pipeline(steps=[('scaler', StandardScaler())])

categorical_transformer = OrdinalEncoder(
    handle_unknown='use_encoded_value',
    unknown_value=-1,
    encoded_missing_value=-1
)

preprocessor = ColumnTransformer(transformers=[
    ('num', numeric_transformer, numeric_features),
    ('cat', categorical_transformer, categorical_features_filtered)
])

# Indices of categorical features after preprocessing
categorical_feature_indices = list(range(
    len(numeric_features), 
    len(numeric_features) + len(categorical_features_filtered)
))

# Create full pipeline
pipeline = Pipeline(steps=[
    ('activity_imputer', activity_imputer),
    ('sport_imputer', sport_imputer),
    ('aav_encoder', aav_encoder),
    ('preprocessor', preprocessor),
    ('regressor', HistGradientBoostingRegressor(
        categorical_features=categorical_feature_indices,
        random_state=42
    ))
])

print(f"✓ Pipeline created with {len(categorical_feature_indices)} categorical feature indices")

# =============================================================================
# 5. MODEL TRAINING
# =============================================================================
print("\n" + "=" * 60)
print("TRAINING MODEL")
print("=" * 60)

# Prepare data
X = learn_df.drop(columns=['target', 'primary_key'], errors='ignore')
y = learn_df['target']

# Train-test split
X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=7)

print(f"Training set: {len(X_train):,}")
print(f"Validation set: {len(X_val):,}")

# Fit pipeline
print("\nFitting pipeline...")
pipeline.fit(X_train, y_train)
print("✓ Model trained")

# Evaluate
y_train_pred = pipeline.predict(X_train)
y_val_pred = pipeline.predict(X_val)

train_r2 = r2_score(y_train, y_train_pred)
train_rmse = np.sqrt(mean_squared_error(y_train, y_train_pred))
train_mae = mean_absolute_error(y_train, y_train_pred)

val_r2 = r2_score(y_val, y_val_pred)
val_rmse = np.sqrt(mean_squared_error(y_val, y_val_pred))
val_mae = mean_absolute_error(y_val, y_val_pred)

print("\n" + "-" * 60)
print(f"{'Metric':<25} {'Train':<15} {'Validation':<15}")
print("-" * 60)
print(f"{'R² Score':<25} {train_r2:<15.4f} {val_r2:<15.4f}")
print(f"{'RMSE':<25} {train_rmse:<15.4f} {val_rmse:<15.4f}")
print(f"{'MAE':<25} {train_mae:<15.4f} {val_mae:<15.4f}")
print("-" * 60)

if train_r2 - val_r2 > 0.1:
    print(f"⚠️  Possible overfitting (gap: {train_r2 - val_r2:.4f})")
else:
    print(f"✓ Good generalization (gap: {train_r2 - val_r2:.4f})")

# =============================================================================
# 6. HYPERPARAMETER TUNING
# =============================================================================
print("\n" + "=" * 60)
print("HYPERPARAMETER TUNING (HalvingRandomSearchCV)")
print("=" * 60)

cv_strategy = KFold(n_splits=5, shuffle=True, random_state=42)

param_dist = {
    'regressor__max_iter': [200, 300, 400],
    'regressor__learning_rate': [0.01, 0.05, 0.1, 0.2],
    'regressor__max_leaf_nodes': [10, 15, 31, 63, 127],
    'regressor__max_depth': [None, 10, 20],
    'regressor__min_samples_leaf': [5, 10, 20, 50],
    'regressor__l2_regularization': [0.0, 0.1, 1.0, 2.0, 10.0]
}

halving_search = HalvingRandomSearchCV(
    estimator=pipeline,
    param_distributions=param_dist,
    factor=2,
    resource='n_samples',
    min_resources=500,
    max_resources='auto',
    scoring='r2',
    cv=cv_strategy,
    n_jobs=-1,
    random_state=42,
    verbose=1
)

print("Starting hyperparameter search...")
halving_search.fit(X_train, y_train)

print(f"\n✓ Best Parameters: {halving_search.best_params_}")
print(f"✓ Best CV R² Score: {halving_search.best_score_:.4f}")

best_model = halving_search.best_estimator_
val_r2_tuned = best_model.score(X_val, y_val)
print(f"✓ Tuned Model Validation R²: {val_r2_tuned:.4f}")

# =============================================================================
# 7 FINAL CROSS-VALIDATION & CONFIDENCE INTERVAL
# =============================================================================
print("\n" + "=" * 60)
print("FINAL CROSS-VALIDATION (CONFIDENCE INTERVAL)")
print("=" * 60)

from sklearn.model_selection import cross_val_score

# 1. Use the best estimator found in the tuning step
final_cv_model = halving_search.best_estimator_

# 2. Define a robust CV strategy (e.g., 10-fold for better distribution)
# We use the FULL dataset (X, y) here, not just X_train
cv_strategy_final = KFold(n_splits=10, shuffle=True, random_state=42)

print(f"Running {cv_strategy_final.get_n_splits()}-Fold CV on the full dataset...")

# 3. Calculate scores
scores = cross_val_score(
    final_cv_model, 
    X, 
    y, 
    cv=cv_strategy_final, 
    scoring='r2', 
    n_jobs=-1
)

# 4. Calculate Statistics
mean_score = scores.mean()
std_dev = scores.std()
# 95% Confidence Interval ≈ Mean ± 2 * StdDev
ci_lower = mean_score - (2 * std_dev)
ci_upper = mean_score + (2 * std_dev)

print("\n" + "-" * 40)
print(f"Cross-Validation Results (R²):")
print("-" * 40)
print(f"Individual Folds: {np.round(scores, 4)}")
print(f"Mean R²:          {mean_score:.4f}")
print(f"Standard Dev:     {std_dev:.4f}")
print(f"95% Conf. Int.:   [{ci_lower:.4f}, {ci_upper:.4f}]")
print("-" * 40)

# Optional: Visualize the distribution of scores
plt.figure(figsize=(8, 4))
plt.boxplot(scores, vert=False)
plt.title(f'Cross-Validation R² Distribution (Mean: {mean_score:.3f})')
plt.xlabel('R² Score')
plt.yticks([])
plt.tight_layout()
plt.savefig(figures_dir / 'cv_score_distribution.png', dpi=150)
plt.close()
print(f"✓ CV score distribution plot saved to figures/cv_score_distribution.png")

# =============================================================================
# 8. FEATURE IMPORTANCE ANALYSIS
# =============================================================================
print("\n" + "=" * 60)
print("FEATURE IMPORTANCE ANALYSIS")
print("=" * 60)

print("Computing permutation importance...")
perm_importance = permutation_importance(
    best_model, X_val, y_val,
    n_repeats=10, random_state=42, n_jobs=-1
)

importance_df = pd.DataFrame({
    'feature': X_val.columns.tolist(),
    'importance_mean': perm_importance.importances_mean,
    'importance_std': perm_importance.importances_std
}).sort_values('importance_mean', ascending=False)

print("\nTop 20 Features:")
print(importance_df.head(20).to_string(index=False))

# Plot feature importance
top_n = 10
top_features = importance_df.head(top_n)

plt.figure(figsize=(10, 6))
plt.barh(range(len(top_features)), top_features['importance_mean'],
         xerr=top_features['importance_std'], align='center', alpha=0.8, color='steelblue')
plt.yticks(range(len(top_features)), top_features['feature'], fontsize=9)
plt.xlabel('Permutation Importance (Mean Decrease in R²)')
plt.title(f'Top {top_n} Feature Importances')
plt.gca().invert_yaxis()
plt.grid(axis='x', alpha=0.3)
plt.tight_layout()
plt.savefig(figures_dir / 'feature_importance.png', dpi=150)
plt.close()
print(f"✓ Feature importance plot saved to figures/feature_importance.png")

# =============================================================================
# 9. LEARNING CURVE
# =============================================================================
print("\n" + "=" * 60)
print("LEARNING CURVE")
print("=" * 60)

def plot_learning_curve(estimator, X, y):
    # Create KFold with shuffle=True
    cv = KFold(n_splits=5, shuffle=True, random_state=42)
    
    train_sizes, train_scores, test_scores = learning_curve(
        estimator, X, y, cv=cv, n_jobs=-1, 
        train_sizes=np.linspace(0.1, 1.0, 5), scoring="r2"
    )
    
    plt.figure(figsize=(10, 6))
    plt.plot(train_sizes, np.mean(train_scores, axis=1), 'o-', label="Training score")
    plt.plot(train_sizes, np.mean(test_scores, axis=1), 'o-', label="Cross-validation score")
    plt.xlabel("Training examples")
    plt.ylabel("R² Score")
    plt.title("Learning Curve")
    plt.legend(loc="best")
    plt.grid()
    plt.tight_layout()
    plt.savefig(figures_dir / 'learning_curve.png', dpi=150)
    plt.close()

print("Computing learning curve...")
plot_learning_curve(halving_search.best_estimator_, X, y)
print(f"✓ Learning curve saved to figures/learning_curve.png")

# =============================================================================
# 10. OUTLIER ANALYSIS - BADLY PREDICTED VALUES
# =============================================================================
print("\n" + "=" * 60)
print("OUTLIER ANALYSIS")
print("=" * 60)

# Calculate prediction errors (residuals) using the validation set
val_residuals = y_val - y_val_pred
val_abs_errors = np.abs(val_residuals)
val_pct_errors = (val_abs_errors / y_val.replace(0, np.nan)) * 100 # Avoid division by zero

# Create analysis dataframe
outlier_df = pd.DataFrame({
    'actual': y_val.values,
    'predicted': y_val_pred,
    'residual': val_residuals.values,
    'abs_error': val_abs_errors.values,
    'pct_error': val_pct_errors.values
}, index=y_val.index)

# Add original features for context analysis
outlier_df = pd.concat([outlier_df, X_val], axis=1)

# Define outlier thresholds (Method: 95th and 99th percentiles)
p95_error = np.percentile(val_abs_errors, 95)
p99_error = np.percentile(val_abs_errors, 99)

print(f"Absolute Error Statistics (Validation):")
print(f"  Mean: {val_abs_errors.mean():.2f}")
print(f"  95th percentile: {p95_error:.2f}")
print(f"  99th percentile: {p99_error:.2f}")

# Categorize
outlier_df['outlier_category'] = 'normal'
outlier_df.loc[outlier_df['abs_error'] > p95_error, 'outlier_category'] = 'moderate_outlier'
outlier_df.loc[outlier_df['abs_error'] > p99_error, 'outlier_category'] = 'extreme_outlier'

# --- VISUALIZATIONS ---
fig, axes = plt.subplots(1, 2, figsize=(16, 6))

# 1. Residuals distribution
axes[0].hist(val_residuals, bins=50, alpha=0.7, color='steelblue', edgecolor='black')
axes[0].axvline(x=0, color='red', linestyle='--', linewidth=2)
axes[0].set_xlabel('Residuals (Actual - Predicted)')
axes[0].set_ylabel('Frequency')
axes[0].set_title('Distribution of Prediction Residuals')
axes[0].grid(alpha=0.3)

# 2. Actual vs Predicted with outliers highlighted
extreme_outliers = outlier_df[outlier_df['outlier_category'] == 'extreme_outlier']
axes[1].scatter(outlier_df['actual'], outlier_df['predicted'], alpha=0.3, s=20, label='Normal', color='steelblue')
axes[1].scatter(extreme_outliers['actual'], extreme_outliers['predicted'], 
                alpha=0.8, s=50, label='Extreme Outliers (Top 1%)', color='red', edgecolor='black')
axes[1].plot([y_val.min(), y_val.max()], [y_val.min(), y_val.max()], 
               'k--', linewidth=2, label='Perfect Prediction')
axes[1].set_xlabel('Actual Target')
axes[1].set_ylabel('Predicted Target')
axes[1].set_title('Actual vs Predicted (Validation Set)')
axes[1].legend()
axes[1].grid(alpha=0.3)

plt.tight_layout()
plt.savefig(figures_dir / 'outlier_analysis.png', dpi=150)
plt.close()

print(f"✓ Outlier analysis complete. Plot saved to figures/outlier_analysis.png")

# =============================================================================
# 11. SAVE MODEL
# =============================================================================
print("\n" + "=" * 60)
print("RETRAINING ON FULL DATASET")
print("=" * 60)

# 1. Retrieve the best pipeline structure/params
final_pipeline = halving_search.best_estimator_

# 2. Fit on the FULL X and y (combining train + val)
print(f"Retraining on full dataset ({len(X)} samples)...")
final_pipeline.fit(X, y) 

# 3. Save THIS model
import joblib
joblib.dump(final_pipeline, models_dir / 'best_model.joblib')
print(f"✓ Model trained on full dataset saved to models/best_model.joblib")