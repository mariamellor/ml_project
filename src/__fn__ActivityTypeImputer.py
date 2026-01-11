from sklearn.base import BaseEstimator, TransformerMixin

# Custom transformer for activity_type based imputation
class ActivityTypeImputer(BaseEstimator, TransformerMixin):
    """Impute job and retired columns with 'not_applicable' based on activity_type"""
    
    def __init__(self, retired_cols=None, job_cols=None, pension_cols=None):
        self.retired_cols = retired_cols
        self.job_cols = job_cols
        self.pension_cols = pension_cols
    
    def fit(self, X, y=None):
        # Store the retired columns if not provided
        if self.retired_cols is None:
            # Detect from dataframe: columns ending with '_retired'
            self.retired_cols_ = [col for col in X.columns if col.endswith('_retired')]
        else:
            self.retired_cols_ = self.retired_cols
        
        # Store the job columns if not provided
        if self.job_cols is None:
            # Detect from dataframe: columns ending with '_current'
            self.job_cols_ = [col for col in X.columns if col.endswith('_current')]
        else:
            self.job_cols_ = self.job_cols
        
        # Store the pension columns if not provided
        if self.pension_cols is None:
            # Detect from dataframe: RETIREMENT_INCOME and similar
            self.pension_cols_ = [col for col in X.columns if 'RETIREMENT_INCOME' in col or 'pension' in col.lower()]
        else:
            self.pension_cols_ = self.pension_cols
        
        return self
    
    def transform(self, X):
        X_copy = X.copy()
        
        # Check if activity_type exists
        if 'activity_type' in X_copy.columns:
            # Mask for non-retired individuals (for retired job columns)
            non_retired_mask = X_copy['activity_type'] != 'type2_1'
            
            # Mask for non-employed individuals (for job columns)
            # Not employed = unemployed (type1_2) or any inactive (type2_X)
            non_employed_mask = (X_copy['activity_type'] == 'type1_2') | (X_copy['activity_type'].str.startswith('type2_'))
            
            # Fill retired job columns for non-retired people
            for col in self.retired_cols_:
                if col in X_copy.columns:
                    if X_copy[col].dtype == 'object' or X_copy[col].dtype.name == 'category':
                        # Categorical: fill with 'not_applicable'
                        X_copy.loc[non_retired_mask & X_copy[col].isna(), col] = 'not_applicable'
                    else:
                        # Numeric (hours, earnings): fill with 0
                        X_copy.loc[non_retired_mask & X_copy[col].isna(), col] = 0
            
            # Fill job columns for non-employed people (unemployed or inactive)
            for col in self.job_cols_:
                if col in X_copy.columns:
                    if X_copy[col].dtype == 'object' or X_copy[col].dtype.name == 'category':
                        # Categorical: fill with 'not_applicable'
                        X_copy.loc[non_employed_mask & X_copy[col].isna(), col] = 'not_applicable'
                    else:
                        # Numeric (hours, earnings): fill with 0
                        X_copy.loc[non_employed_mask & X_copy[col].isna(), col] = 0
            
            # Fill pension columns with 0 for non-retired people
            for col in self.pension_cols_:
                if col in X_copy.columns:
                    # Pension columns are numeric (income), fill with 0 for non-retired
                    X_copy.loc[non_retired_mask & X_copy[col].isna(), col] = 0
        
        return X_copy
    
