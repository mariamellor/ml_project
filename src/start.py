import pandas as pd

# Load main dataset
train = pd.read_csv("learn_dataset.csv")

print("MAIN DATASET")
print("Number of people:", train.shape[0])
print("Number of columns:", train.shape[1])

print("\nColumns:")
print(train.columns)

print("\nDataset info:")
print(train.info())

print("\nMissing values per column:")
print(train.isna().sum())

# Load sport dataset
print("\n--- SPORT DATASET ---")
sport = pd.read_csv("learn_dataset_sport.csv")
print("Sport dataset shape:", sport.shape)
print("Sport columns:", sport.columns)
print(sport.head())

print("\n--- JOB SECURITY DATASET ---")

job_sec = pd.read_csv("learn_dataset_JOB_SECURITY.csv")

print("Job security shape:", job_sec.shape)
print("Job security columns:")
print(job_sec.columns)
print(job_sec.head())
