import pandas as pd
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = BASE_DIR

N_ROWS_EXPECTED = 50044
TARGET_COL = "target"

def load_main_data():
    return pd.read_csv(DATA_DIR / "learn_dataset.csv")

def load_sport_data():
    return pd.read_csv(DATA_DIR / "learn_dataset_sport.csv")

def load_job_security_data():
    return pd.read_csv(DATA_DIR / "learn_dataset_JOB_SECURITY.csv")

def add_sport_variable(main_df, sport_df):
    """
    Add a binary variable indicating sport club membership.

    Individuals appearing in the sport dataset are considered sport members.
    Individuals not appearing are considered non-members.
    """
    df = main_df.merge(
        sport_df[["primary_key"]],
        on="primary_key",
        how="left",
        indicator=True
    )
    df["sport_member"] = (df["_merge"] == "both").astype(int)
    df.drop(columns="_merge", inplace=True)
    return df

def add_job_security(main_df, job_df):
    return main_df.merge(
        job_df,
        on="primary_key",
        how="left"
    )

def prepare_dataset():
    """
    Build the final dataset by:
    - loading the main dataset
    - merging sport information and creating a binary variable
    - merging job security information
    - applying basic sanity checks
    """
    df = load_main_data()
    df = add_sport_variable(df, load_sport_data())
    df = add_job_security(df, load_job_security_data())

    # Basic sanity checks
    assert df.shape[0] == N_ROWS_EXPECTED, "Number of rows changed after merge"
    assert "sport_member" in df.columns, "sport_member column missing"

    return df

def split_features_target(df, target_col="target"):
    """
    Split the dataset into features (X) and target (y).
    """
    X = df.drop(columns=[target_col])
    y = df[target_col]
    return X, y

def select_features(df):
    """
    Select the final set of features used for modeling.
    """
    features = [
        "AGE_2018",
        "activity_type",
        "Insee_code",
        "Studying",
        "sex",
        "Household",
        "Occupation_42",
        "HIGHEST_DIPLOMA",
        "sport_member",
        "JOB_SECURITY"
    ]

    for col in features:
        assert col in df.columns, f"Missing feature: {col}"

    return df[features]

def get_final_dataset():
    """
    Return the final features (X) and target (y), ready for modeling.
    """
    df = prepare_dataset()
    X = select_features(df)
    y = df[TARGET_COL]
    return X, y

if __name__ == "__main__":
    X, y = get_final_dataset()
    print("Final dataset ready.")
    print("X shape:", X.shape)
    print("y shape:", y.shape)

