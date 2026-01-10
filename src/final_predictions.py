import pandas as pd
import joblib

from data_integration import build_features


# ---------------------------------------------------------
# Load trained model
# ---------------------------------------------------------
print("Loading trained model...")
model = joblib.load("final_model.joblib")


# ---------------------------------------------------------
# Load TEST data
# ---------------------------------------------------------
print("Loading test data...")
df_test = build_features(is_train=False)

X_test = df_test.drop(columns=["primary_key"], errors="ignore")
X_test = X_test.drop(columns=["Insee_code"], errors="ignore")

print(f"Test dataset shape: {X_test.shape}")


# ---------------------------------------------------------
# Predict
# ---------------------------------------------------------
print("Generating predictions...")
y_pred = model.predict(X_test)


# ---------------------------------------------------------
# Save predictions
# ---------------------------------------------------------
predictions = pd.DataFrame({
    "primary_key": df_test["primary_key"],
    "target": y_pred
})

predictions.to_csv("predictions.csv", index=False)

print("predictions.csv generated successfully.")

