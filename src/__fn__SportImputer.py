from sklearn.base import BaseEstimator, TransformerMixin
import pandas as pd
import numpy as np

# Custom transformer for sport columns imputation
class SportImputer(BaseEstimator, TransformerMixin):
    """
    Impute sport columns with 'not_applicable' for all missing values.
    
    Improvements:
    - Forces 'Categorie' and 'Sports' to string type to prevent pipeline errors.
    - Imputes 'not_applicable' for missing values in categorical columns.
    - Imputes 0 for strictly numeric columns (if any).
    """
    
    def __init__(self, sport_cols=None):
        self.sport_cols = sport_cols
    
    def fit(self, X, y=None):
        # Store the sport columns if not provided
        if self.sport_cols is None:
            # Detect from dataframe: Sports and related columns
            self.sport_cols_ = [col for col in X.columns if col in ['Sports', 'Categorie']]
        else:
            self.sport_cols_ = self.sport_cols
        
        return self
    
    def transform(self, X):
        X_copy = X.copy()
        
        for col in self.sport_cols_:
            if col not in X_copy.columns:
                continue

            # 'Categorie' and 'Sports' are categorical codes. We must cast them to string so 'NaN' becomes 'nan'. This matches the AAV2020Encoder logic and prevents OrdinalEncoder from crashing on mixed types.
            if col in ['Categorie', 'Sports']:
                X_copy[col] = X_copy[col].astype(str)
                # Replace the string 'nan' (from casting np.nan) with 'not_applicable'
                X_copy[col] = X_copy[col].replace('nan', 'not_applicable')
            
            # --- Standard Handling for other columns ---
            else:
                if X_copy[col].dtype == 'object' or isinstance(X_copy[col].dtype, pd.CategoricalDtype):
                    X_copy[col] = X_copy[col].fillna('not_applicable')
                else:
                    # Numeric fallback (e.g. if there were count columns)
                    X_copy[col] = X_copy[col].fillna(0)
        
        return X_copy