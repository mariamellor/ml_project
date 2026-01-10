import numpy as np
import pandas as pd

from sklearn.model_selection import RandomizedSearchCV
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder
from sklearn.impute import SimpleImputer
from sklearn.metrics import r2_score

from sklearn.ensemble import HistGradientBoostingRegressor

from data_integration import build_features


# ---------------------------------------------------------------------
# 1. Load training data
# ---------------------------------------------------------------------
print("Loading training data...")
df = build_features(is_train=True)

y = df["target"]
X = df.drop(columns=["target", "primary_key"], errors="ignore")

# ---------------------------------------------------------------------
# IMPORTANT: drop very high-cardinality identifiers
# ---------------------------------------------------------------------
X = X.drop(columns=["Insee_code"], errors="ignore")

print(f"Training dataset shape: {X.shape}")


# ---------------------------------------------------------------------
# 2. Identify feature types
# ---------------------------------------------------------------------
categorical_features = X.select_dtypes(
    include=["object", "category", "bool"]
).columns.tolist()

numerical_features = X.select_dtypes(
    exclude=["object", "category", "bool"]
).columns.tolist()

print(f"Categorical features: {len(categorical_features)}")
print(f"Numerical features: {len(numerical_features)}")


# ---------------------------------------------------------------------
# 3. Preprocessing
# ---------------------------------------------------------------------
numeric_transformer = Pipeline(
    steps=[
        ("imputer", SimpleImputer(strategy="median"))
    ]
)

categorical_transformer = Pipeline(
    steps=[
        ("imputer", SimpleImputer(strategy="most_frequent")),
        ("onehot", OneHotEncoder(handle_unknown="ignore", sparse_output=False))
    ]
)

preprocessor = ColumnTransformer(
    transformers=[
        ("num", numeric_transformer, numerical_features),
        ("cat", categorical_transformer, categorical_features),
    ]
)


# ---------------------------------------------------------------------
# 4. Model
# ---------------------------------------------------------------------
model = HistGradientBoostingRegressor(random_state=42)

pipeline = Pipeline(
    steps=[
        ("preprocess", preprocessor),
        ("model", model)
    ]
)


# ---------------------------------------------------------------------
# 5. Hyperparameter tuning with resampling
# ---------------------------------------------------------------------
param_distributions = {
    "model__max_depth": [3, 5, 7],
    "model__learning_rate": [0.03, 0.05],
    "model__max_iter": [300, 500],
    "model__min_samples_leaf": [20, 40],
    "model__l2_regularization": [0.0, 0.1],
}

print("Selecting a subsample for hyperparameter tuning...")
subsample_size = 15000

X_sub = X.sample(n=subsample_size, random_state=42)
y_sub = y.loc[X_sub.index]

search = RandomizedSearchCV(
    estimator=pipeline,
    param_distributions=param_distributions,
    n_iter=8,
    cv=3,
    scoring="r2",
    n_jobs=1,
    random_state=42,
    verbose=2
)

print("Starting hyperparameter search...")
search.fit(X_sub, y_sub)

print("\nBest cross-validated R²:", search.best_score_)
print("Best parameters:")
for param, value in search.best_params_.items():
    print(f"  {param}: {value}")


# ---------------------------------------------------------------------
# 6. Train final model on full dataset
# ---------------------------------------------------------------------
print("\nTraining final model on full dataset...")
final_model = search.best_estimator_
final_model.fit(X, y)

y_pred = final_model.predict(X)
r2 = r2_score(y, y_pred)

print(f"Final model in-sample R² (diagnostic): {r2:.4f}")

print("\nTraining completed successfully.")


import joblib

# ---------------------------------------------------------
# Save trained model
# ---------------------------------------------------------
joblib.dump(final_model, "final_model.joblib")
print("Model saved as final_model.joblib")
