import os
from pathlib import Path

import pandas as pd
import numpy as np
import joblib

# Set project root
script_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.dirname(script_dir)
os.chdir(project_root)

# Create output directories
models_dir = Path(project_root) / 'models'
figures_dir = Path(project_root) / 'figures'
models_dir.mkdir(exist_ok=True)
figures_dir.mkdir(exist_ok=True)

test_df = pd.read_pickle("data/processed/test_df.pkl")

# Extract column groups for imputers
job_cols_in_df = [col for col in test_df.columns if col.endswith('_current')]
retired_cols_in_df = [col for col in test_df.columns if col.endswith('_retired')]
pension_cols_in_df = ['RETIREMENT_INCOME']
sport_cols_in_df = ['Sports', 'Categorie']

final_model = joblib.load(models_dir / 'best_model.joblib')


# Make predictions on test dataset
print("=" * 60)
print("MAKING PREDICTIONS ON TEST DATASET")
print("=" * 60)

# Drop primary_key and target if they exist in test_df
X_test_final = test_df.drop(columns=['primary_key', 'target'], errors='ignore')

print(f"\nTest dataset shape: {X_test_final.shape}")
print("Making predictions...")

# Make predictions using the final trained model
test_predictions = final_model.predict(X_test_final)

# Create results dataframe
results_df = pd.DataFrame({
    'primary_key': test_df['primary_key'],
    'target': test_predictions
})

# Save to CSV
os.makedirs("data/processed", exist_ok=True)
results_df.to_csv('data/processed/predictions.csv', index=False)

print(f"✓ Predictions saved to 'data/processed/predictions.csv'")
print(f"  - Number of predictions: {len(results_df)}")
print(f"  - Prediction statistics:")
print(f"    • Mean: {test_predictions.mean():.2f}")
print(f"    • Median: {np.median(test_predictions):.2f}")
print(f"    • Min: {test_predictions.min():.2f}")
print(f"    • Max: {test_predictions.max():.2f}")
print("=" * 60)