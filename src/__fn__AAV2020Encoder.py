from sklearn.base import BaseEstimator, TransformerMixin
import pandas as pd

class AAV2020Encoder(BaseEstimator, TransformerMixin):
    """
    Custom encoder for AAV2020 and CATEAVV2020 columns.
    
    Improvements:
    - Handles type casting (float -> str) internally to prevent pipeline errors.
    - Groups rare AAV codes into '101'.
    """
    
    def __init__(self):
        self.kept_categories_ = None
        
    def fit(self, X, y=None):
        """Learn the top 100 AAV codes from training data"""
        if 'AAV2020' not in X.columns:
            return self
        
        # Define special categories to always keep
        special_categories = {'000', 'BAL', 'CHA', 'GEN', 'LAU', 'LUX', 'MON', 'SAR'}
        
        # Convert to string first to ensure consistent counting
        # (Avoids splitting 1.0 (float) and "1.0" (str))
        series_str = X['AAV2020'].astype(str)
        
        # Get top 100 most frequent AAV codes (excluding 'nan' string if desired, but usually we just count valid codes)
        value_counts = series_str[series_str != 'nan'].value_counts()
        self.top_100_ = set(value_counts.head(100).index)
        
        # Combine special categories and top 100
        self.kept_categories_ = special_categories.union(self.top_100_)
        
        return self
    
    def transform(self, X):
        """
        Transform AAV columns:
        1. Casts CATEAVV2020 to string (fixes OrdinalEncoder crash).
        2. Groups rare AAV2020 categories into '101'.
        """
        X_copy = X.copy()
        
        if 'CATEAVV2020' in X_copy.columns:
            # Force to string so NaN becomes "nan" and matches OrdinalEncoder expectations
            X_copy['CATEAVV2020'] = X_copy['CATEAVV2020'].astype(str)
        
        # --- Handle AAV2020 ---
        if 'AAV2020' not in X_copy.columns or self.kept_categories_ is None:
            return X_copy
            
        # Ensure AAV2020 is also string before mapping
        X_copy['AAV2020'] = X_copy['AAV2020'].astype(str)
        
        # Map values: keep if in kept_categories_, otherwise map to '101'
        # Note: We check against 'nan' string because we casted above
        X_copy['AAV2020'] = X_copy['AAV2020'].apply(
            lambda x: x if x == 'nan' or x in self.kept_categories_ else '101'
        )
        
        return X_copy