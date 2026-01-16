import os
import pandas as pd
import numpy as np

# Set project root to ml_project directory
script_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.dirname(script_dir)
os.chdir(project_root)

# Load reference/lookup tables
job_desc_map_df = pd.read_csv("data/code_job_desc_map.csv")
departments_df = pd.read_csv("data/departments.csv")
sports_desc_df = pd.read_csv("data/code_Sports.csv")
city_pop_df = pd.read_csv("data/city_pop.csv")
city_loc_df = pd.read_csv("data/city_loc.csv")
city_loc_df = city_loc_df.drop(columns=['LAT', 'Long'], errors='ignore')
city_adm_df = pd.read_csv("data/city_adm.csv")
city_aav_df = pd.read_excel("data/additional_data/AAV2020.xlsx", sheet_name='Composition_communale', skiprows=5)
city_aav_df = city_aav_df.drop(columns=['LIBGEO', 'LIBAAV2020', 'DEP', 'REG'], errors='ignore')


def load_and_merge_datasets(dataset_type='learn'):
    """
    Load and merge all datasets for training or testing.
    
    Parameters:
    -----------
    dataset_type : str
        Either 'learn' or 'test'
    
    Returns:
    --------
    tuple: (merged_df, job_cols, retired_cols, pension_cols, sport_cols)
    """
    prefix = f"data/{dataset_type}_dataset"
    
    # Load all datasets
    main_df = pd.read_csv(f"{prefix}.csv")
    sport_df = pd.read_csv(f"{prefix}_sport.csv")
    job_df = pd.read_csv(f"{prefix}_job.csv")
    job_security_df = pd.read_csv(f"{prefix}_JOB_SECURITY.csv")
    retired_former_df = pd.read_csv(f"{prefix}_retired_former.csv")
    retired_jobs_df = pd.read_csv(f"{prefix}_retired_jobs.csv")
    retired_pension_df = pd.read_csv(f"{prefix}_retired_pension.csv")
    
    # Merge with reference tables
    job_df = job_df.merge(job_desc_map_df, left_on='job_desc', right_on='N3', how='left')
    retired_jobs_df = retired_jobs_df.merge(job_desc_map_df, left_on='job_desc', right_on='N3', how='left')
    sport_df = sport_df.merge(sports_desc_df, left_on='Sports', right_on='Code', how='left')
    sport_df = sport_df[['primary_key', 'Sports', 'Categorie']]

    # Rename columns with meaningful suffixes
    job_df_renamed = job_df.rename(columns={col: f"{col}_current" for col in job_df.columns if col != 'primary_key'})
    retired_jobs_df_renamed = retired_jobs_df.rename(columns={col: f"{col}_retired" for col in retired_jobs_df.columns if col != 'primary_key'})
    
    # Merge all datasets
    df = main_df.merge(job_df_renamed, on='primary_key', how='left')
    df = df.merge(job_security_df, on='primary_key', how='left')
    df = df.merge(retired_jobs_df_renamed, on='primary_key', how='left')
    df = df.merge(retired_pension_df, on='primary_key', how='left')
    df = df.merge(retired_former_df, on='primary_key', how='left')
    df = df.merge(sport_df, on='primary_key', how='left')
    df = df.merge(city_loc_df, on='Insee_code', how='left')
    df = df.merge(city_adm_df, on='Insee_code', how='left')
    df = df.merge(city_aav_df, left_on='Insee_code', right_on='CODGEO', how='left')
    
    # Save column lists for imputers
    job_cols = [col for col in df.columns if col.endswith('_current')]
    retired_cols = [col for col in df.columns if col.endswith('_retired')]
    pension_cols = [col for col in retired_pension_df.columns if col != 'primary_key' and col in df.columns]
    sport_cols = [col for col in sport_df.columns if col != 'primary_key' and col in df.columns]
    
    return df, job_cols, retired_cols, pension_cols, sport_cols


# Load and merge datasets
learn_df, job_cols_in_df, retired_cols_in_df, pension_cols_in_df, sport_cols_in_df = load_and_merge_datasets('learn')
test_df, _, _, _, _ = load_and_merge_datasets('test')

# =============================================================================
# CONSISTENCY CHECKS
# =============================================================================
learn_cols = set(learn_df.columns)
test_cols = set(test_df.columns)
only_in_learn = learn_cols - test_cols
only_in_test = test_cols - learn_cols

# Check columns
assert only_in_learn == {'target'} and len(only_in_test) == 0, \
    f"Column mismatch: only_in_learn={only_in_learn}, only_in_test={only_in_test}"

# Check data types
common_cols = learn_cols & test_cols
dtype_mismatches = [(col, learn_df[col].dtype, test_df[col].dtype) 
                    for col in common_cols if learn_df[col].dtype != test_df[col].dtype]
assert len(dtype_mismatches) == 0, f"Data type mismatches: {dtype_mismatches}"

# Check for duplicate primary keys
assert learn_df['primary_key'].duplicated().sum() == 0, "Duplicate primary keys in learn_df"
assert test_df['primary_key'].duplicated().sum() == 0, "Duplicate primary keys in test_df"

# Check for data leakage
overlap = set(learn_df['primary_key']) & set(test_df['primary_key'])
assert len(overlap) == 0, f"Data leakage: {len(overlap)} overlapping keys"

# =============================================================================
# SAVE TO PICKLE
# =============================================================================
os.makedirs("data/processed", exist_ok=True)
learn_df.to_pickle("data/processed/learn_df.pkl")
test_df.to_pickle("data/processed/test_df.pkl")

# Verify saved files
learn_verify = pd.read_pickle("data/processed/learn_df.pkl")
test_verify = pd.read_pickle("data/processed/test_df.pkl")
assert learn_verify.equals(learn_df) and test_verify.equals(test_df), "Pickle verification failed"

# Final summary
learn_size = os.path.getsize("data/processed/learn_df.pkl") / 1e6
test_size = os.path.getsize("data/processed/test_df.pkl") / 1e6
print(f"Data integration complete: learn({len(learn_df):,} rows, {learn_size:.1f}MB) | test({len(test_df):,} rows, {test_size:.1f}MB)")