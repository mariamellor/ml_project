from sklearn.base import BaseEstimator, TransformerMixin

# Custom transformer for sport columns imputation
class SportImputer(BaseEstimator, TransformerMixin):
    """
    Impute sport columns with 'not_applicable' for all missing values.
    
    Since sports_df is exhaustive, anyone not in sports_df will have NaN for all 
    sport-related columns after the left merge. This includes:
    - Original sport columns (Sports, etc.)
    - Categorical columns from sports_desc merge (Categorie, Nom fédération, Nom catégorie, Code)
    
    All of these will be filled with 'not_applicable' to indicate the person doesn't 
    practice any sport or the information is not available.
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
        
        # Fill sport columns based on their data type
        # Categorical columns: fill with 'not_applicable'
        # Numeric columns: fill with 0 (if any exist)
        for col in self.sport_cols_:
            if col in X_copy.columns:
                if X_copy[col].dtype == 'object' or X_copy[col].dtype.name == 'category':
                    # Categorical: fill with 'not_applicable'
                    X_copy[col] = X_copy[col].fillna('not_applicable')
                else:
                    # Numeric: fill with 0
                    X_copy[col] = X_copy[col].fillna(0)
        
        return X_copy
