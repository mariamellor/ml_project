# =============================================================================
# 02BIS-MODEL_BUILDING_RF.PY
# =============================================================================
# This script builds and trains a regression model using RandomForest
# with custom imputers and preprocessing pipeline.
#
# Sections:
#   1. IMPORTS & SETUP
#   2. LOAD DATA
#   3. CUSTOM IMPUTERS
#   4. PREPROCESSING PIPELINE
#   5. MODEL TRAINING
#   6. FEATURE IMPORTANCE ANALYSIS (GROUPED ONE-HOT FEATURES)
#   7. LEARNING CURVE
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
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.impute import SimpleImputer
from sklearn.model_selection import train_test_split, KFold, learning_curve
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
from sklearn.ensemble import RandomForestRegressor
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

# Force specific columns to be treated as categorical
force_categorical_cols = ['CATEAVV2020', 'Categorie']
for col in force_categorical_cols:
    if col in learn_df.columns:
        learn_df[col] = learn_df[col].astype(str)

# Identify column types
numeric_features = learn_df.select_dtypes(include=['int64', 'float64']).columns.tolist()
categorical_features = learn_df.select_dtypes(include=['object', 'category']).columns.tolist()

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
# For numeric: fill missing with -1, then scale
numeric_transformer = Pipeline(steps=[
    ('imputer', SimpleImputer(strategy='constant', fill_value=-1)),
    ('scaler', StandardScaler())
])

# For categorical: fill missing with 'n/a', then one-hot encode
categorical_transformer = Pipeline(steps=[
    ('imputer', SimpleImputer(strategy='constant', fill_value='n/a')),
    ('onehot', OneHotEncoder(handle_unknown='ignore', sparse_output=False))
])

preprocessor = ColumnTransformer(transformers=[
    ('num', numeric_transformer, numeric_features),
    ('cat', categorical_transformer, categorical_features_filtered)
])

# Create full pipeline with RandomForest
pipeline = Pipeline(steps=[
    ('activity_imputer', activity_imputer),
    ('sport_imputer', sport_imputer),
    ('aav_encoder', aav_encoder),
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

print(f"✓ Pipeline created with RandomForest (n_estimators=100)")

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
# 6. FEATURE IMPORTANCE ANALYSIS (GROUPED ONE-HOT FEATURES)
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
plt.savefig(figures_dir / 'feature_importance_rf.png', dpi=150)
plt.close()
print(f"✓ Feature importance plot saved to figures/feature_importance_rf.png")

# =============================================================================
# 7. LEARNING CURVE
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
    plt.title("Learning Curve - RandomForest")
    plt.legend(loc="best")
    plt.grid()
    plt.tight_layout()
    plt.savefig(figures_dir / 'learning_curve_rf.png', dpi=150)
    plt.close()

print("Computing learning curve...")
plot_learning_curve(pipeline, X_train, y_train)
print(f"✓ Learning curve saved to figures/learning_curve_rf.png")

# =============================================================================
# 8. SAVE MODEL
# =============================================================================
print("\n" + "=" * 60)
print("SAVING MODEL")
print("=" * 60)

import joblib

joblib.dump(pipeline, models_dir / 'rf_model.joblib')
print(f"✓ Model saved to models/rf_model.joblib")

print("\n" + "=" * 60)
print("PIPELINE COMPLETE")
print("=" * 60)