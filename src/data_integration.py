import pandas as pd
from pathlib import Path

# ---------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------
BASE_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = BASE_DIR

# ---------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------
N_ROWS_TRAIN = 50044
N_ROWS_TEST = 50042
TARGET_COL = "target"


# ---------------------------------------------------------------------
# Loaders
# ---------------------------------------------------------------------
def load_csv(name):
    return pd.read_csv(DATA_DIR / name)


# ---------------------------------------------------------------------
# Feature engineering helpers
# ---------------------------------------------------------------------
def add_sport_member(df, sport_df):
    """Binary indicator of sport club membership."""
    df = df.merge(
        sport_df[["primary_key"]],
        on="primary_key",
        how="left",
        indicator=True
    )
    df["sport_member"] = (df["_merge"] == "both").astype(int)
    df.drop(columns="_merge", inplace=True)
    return df


def add_department(df):
    """Department code extracted from Insee_code."""
    df["department"] = df["Insee_code"].astype(str).str.zfill(5).str[:2]
    return df


# ---------------------------------------------------------------------
# Main builder
# ---------------------------------------------------------------------
def build_features(is_train=True):
    """
    Build the full feature dataset with robust joins.
    """

    # -----------------------------
    # Load main dataset
    # -----------------------------
    if is_train:
        df = load_csv("learn_dataset.csv")
    else:
        df = load_csv("test_dataset.csv")

    # -----------------------------
    # Sport membership
    # -----------------------------
    sport_df = load_csv(
        "learn_dataset_sport.csv" if is_train else "test_dataset_sport.csv"
    )
    df = add_sport_member(df, sport_df)

    # -----------------------------
    # Job security
    # -----------------------------
    job_sec_df = load_csv(
        "learn_dataset_JOB_SECURITY.csv" if is_train else "test_dataset_JOB_SECURITY.csv"
    )
    df = df.merge(job_sec_df, on="primary_key", how="left")

    # -----------------------------
    # Current job (employees)
    # -----------------------------
    try:
        job_df = load_csv(
            "learn_dataset_job.csv" if is_train else "test_dataset_job.csv"
        )

        job_keep = [
            "primary_key",
            "Earnings",
            "WORKING_HOURS",
            "ECONOMIC_SECTOR",
            "Work_condition",
            "Employee_count"
        ]

        job_keep = [c for c in job_keep if c in job_df.columns]
        job_df = job_df[job_keep]

        df = df.merge(job_df, on="primary_key", how="left")

    except FileNotFoundError:
        pass

    # -----------------------------
    # Retired jobs (ROBUST)
    # -----------------------------
    try:
        retired_df = load_csv(
            "learn_dataset_retired_jobs.csv"
            if is_train
            else "test_dataset_retired_jobs.csv"
        )

        retired_keep = ["primary_key"]

        for col in retired_df.columns:
            if "occupation" in col.lower():
                retired_keep.append(col)
            if "retire" in col.lower() and "age" in col.lower():
                retired_keep.append(col)

        retired_keep = list(set(retired_keep))
        retired_df = retired_df[retired_keep]

        df = df.merge(retired_df, on="primary_key", how="left")

    except FileNotFoundError:
        pass

    # -----------------------------
    # Retired pension
    # -----------------------------
    try:
        pension_df = load_csv(
            "learn_dataset_retired_pension.csv"
            if is_train
            else "test_dataset_retired_pension.csv"
        )

        pension_cols = [c for c in pension_df.columns if c != "primary_key"]
        pension_df = pension_df[["primary_key"] + pension_cols]

        df = df.merge(pension_df, on="primary_key", how="left")

    except FileNotFoundError:
        pass

    # -----------------------------
    # Geography
    # -----------------------------
    df = add_department(df)

    # -----------------------------
    # Sanity checks
    # -----------------------------
    expected_rows = N_ROWS_TRAIN if is_train else N_ROWS_TEST
    assert df.shape[0] == expected_rows, "Row count changed after merge"
    assert "sport_member" in df.columns, "sport_member missing"

    if is_train:
        assert TARGET_COL in df.columns, "Target missing"

    return df


# ---------------------------------------------------------------------
# Debug
# ---------------------------------------------------------------------
if __name__ == "__main__":
    train_df = build_features(is_train=True)
    test_df = build_features(is_train=False)

    print("Train shape:", train_df.shape)
    print("Test shape:", test_df.shape)
    print("\nColumns:")
    print(train_df.columns)
