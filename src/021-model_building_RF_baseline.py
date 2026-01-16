# =============================================================================
# 02BIS-MODEL_BUILDING_RF_BASELINE.PY
# =============================================================================
# This script builds and trains a baseline RandomForest regression model
# using the raw learn_dataset.csv without custom imputation or additional data.
#
# Sections:
#   1. IMPORTS & SETUP
#   2. LOAD DATA
#   3. PREPROCESSING PIPELINE
#   4. MODEL TRAINING
#   5. FEATURE IMPORTANCE ANALYSIS (GROUPED ONE-HOT FEATURES)
#   6. SAVE MODEL
# =============================================================================

# =============================================================================
# 1. IMPORTS & SETUP
# =============================================================================
import os
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import joblib

from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
from sklearn.ensemble import RandomForestRegressor

# Set project root
script_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.dirname(script_dir)
os.chdir(project_root)

# Create output directories
models_dir = Path(project_root) / 'models'
figures_dir = Path(project_root) / 'figures'
models_dir.mkdir(exist_ok=True)
figures_dir.mkdir(exist_ok=True)

# =============================================================================
# 2. LOAD DATA
# =============================================================================
print("=" * 60)
print("LOADING DATA (BASELINE - RAW CSV)")
print("=" * 60)

learn_df = pd.read_csv("data/learn_dataset.csv")
test_df = pd.read_csv("data/test_dataset.csv")

print(f"✓ Learn dataset: {learn_df.shape}")
print(f"✓ Test dataset: {test_df.shape}")

# =============================================================================
# 3. PREPROCESSING PIPELINE
# =============================================================================
print("\n" + "=" * 60)
print("BUILDING BASELINE PREPROCESSING PIPELINE")
print("=" * 60)

# Identify column types
numeric_features = learn_df.select_dtypes(include=['int64', 'float64']).columns.tolist()
categorical_features = learn_df.select_dtypes(include=['object']).columns.tolist()

# Remove target and primary_key from features
for col in ['target', 'primary_key']:
    if col in numeric_features:
        numeric_features.remove(col)
    if col in categorical_features:
        categorical_features.remove(col)

# Filter high-cardinality categorical features
MAX_CATEGORIES = 150
categorical_features_filtered = []
categorical_features_dropped = []

for col in categorical_features:
    if learn_df[col].nunique() <= MAX_CATEGORIES:
        categorical_features_filtered.append(col)
    else:
        categorical_features_dropped.append((col, learn_df[col].nunique()))

print(f"✓ Numeric features: {len(numeric_features)}")
print(f"✓ Categorical features kept (≤ {MAX_CATEGORIES}): {len(categorical_features_filtered)}")
print(f"✗ Categorical features dropped: {len(categorical_features_dropped)}")

# Build preprocessing transformers (no imputation, no scaling for baseline)
# Numeric features: pass through as-is
numeric_transformer = 'passthrough'

# Categorical features: one-hot encode
categorical_transformer = OneHotEncoder(handle_unknown='ignore', sparse_output=False)

preprocessor = ColumnTransformer(transformers=[
    ('num', numeric_transformer, numeric_features),
    ('cat', categorical_transformer, categorical_features_filtered)
])

# Create full pipeline with RandomForest
pipeline = Pipeline(steps=[
    ('preprocessor', preprocessor),
    ('regressor', RandomForestRegressor(
        n_estimators=100,
        max_depth=20,
        min_samples_split=10,
        min_samples_leaf=5,
        random_state=42,
        n_jobs=-1,
        verbose=1
    ))
])

print(f"✓ Baseline pipeline created with RandomForest (n_estimators=100)")

# =============================================================================
# 4. MODEL TRAINING
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
# 5. FEATURE IMPORTANCE ANALYSIS (GROUPED ONE-HOT FEATURES)
# =============================================================================
print("\n" + "=" * 60)
print("FEATURE IMPORTANCE ANALYSIS (GROUPED ONE-HOT FEATURES)")
print("=" * 60)

# Get feature names after preprocessing
feature_names = preprocessor.get_feature_names_out()

# Get feature importances from RandomForest
rf_importances = pipeline.named_steps['regressor'].feature_importances_

# Create mapping from encoded feature names to original feature names
# For one-hot encoded features, group them together
feature_groups = {}

# Build mapping: for each original categorical feature, find all its one-hot encoded versions
for original_cat_feat in categorical_features_filtered:
    # Find all encoded features that belong to this original categorical feature
    prefix = f'cat__{original_cat_feat}_'
    indices = [i for i, fname in enumerate(feature_names) if fname.startswith(prefix)]
    if indices:
        feature_groups[original_cat_feat] = indices

# Add numeric features (they don't get one-hot encoded)
for original_num_feat in numeric_features:
    # Find the numeric feature in the encoded names
    num_name = f'num__{original_num_feat}'
    indices = [i for i, fname in enumerate(feature_names) if fname == num_name]
    if indices:
        feature_groups[original_num_feat] = indices
# Aggregate importances for grouped features
grouped_importances = []
for feat_name, indices in feature_groups.items():
    # Sum importances for all one-hot encoded versions of the same feature
    total_importance = sum(rf_importances[idx] for idx in indices)
    grouped_importances.append({
        'feature': feat_name,
        'importance': total_importance,
        'n_encoded': len(indices)
    })

importance_df = pd.DataFrame(grouped_importances).sort_values('importance', ascending=False)

print("\nTop 20 Features (grouped one-hot encoded):")
print(importance_df.head(20).to_string(index=False))

# Plot feature importance
top_n = 20
top_features = importance_df.head(top_n)

plt.figure(figsize=(12, 10))
plt.barh(range(len(top_features)), top_features['importance'],
         align='center', alpha=0.8, color='steelblue')
plt.yticks(range(len(top_features)), top_features['feature'], fontsize=9)
plt.xlabel('Feature Importance (MDI - Grouped)')
plt.title(f'Top {top_n} Feature Importances (One-Hot Features Grouped)')
plt.gca().invert_yaxis()
plt.grid(axis='x', alpha=0.3)
plt.tight_layout()
plt.savefig(figures_dir / 'feature_importance_rf_baseline.png', dpi=150)
plt.close()
print(f"✓ Feature importance plot saved to figures/feature_importance_rf_baseline.png")

# =============================================================================
# 6. SAVE MODEL
# =============================================================================
print("\n" + "=" * 60)
print("SAVING BASELINE MODEL")
print("=" * 60)

joblib.dump(pipeline, models_dir / 'rf_model_baseline.joblib')
print(f"✓ Baseline model saved to models/rf_model_baseline.joblib")

print("\n" + "=" * 60)
print("BASELINE PIPELINE COMPLETE")
print("=" * 60)