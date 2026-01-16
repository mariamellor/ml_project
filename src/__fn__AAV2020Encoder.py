from sklearn.base import BaseEstimator, TransformerMixin
import pandas as pd

class AAV2020Encoder(BaseEstimator, TransformerMixin):
    """
    Custom encoder for AAV2020 (Aire d'Attraction des Villes) column.
    Keeps:
    - '000': outside attraction area
    - Special cities: BAL, CHA, GEN, LAU, LUX, MON, SAR
    - Top 100 most frequent AAV codes
    - Groups all others into '101' category
    """
    
    def __init__(self):
        self.kept_categories_ = None
        self.top_100_ = None
        
    def fit(self, X, y=None):
        """Learn the top 100 AAV codes from training data"""
        if 'AAV2020' not in X.columns:
            return self
        
        # Define special categories to always keep
        special_categories = {'000', 'BAL', 'CHA', 'GEN', 'LAU', 'LUX', 'MON', 'SAR'}
        
        # Get top 100 most frequent AAV codes (excluding NaN)
        value_counts = X['AAV2020'].dropna().value_counts()
        self.top_100_ = set(value_counts.head(100).index)
        
        # Combine special categories and top 100
        self.kept_categories_ = special_categories.union(self.top_100_)
        
        return self
    
    def transform(self, X):
        """Transform AAV2020 column by grouping rare categories into '101'"""
        X_copy = X.copy()
        
        if 'AAV2020' not in X_copy.columns or self.kept_categories_ is None:
            return X_copy
        
        # Map values: keep if in kept_categories_, otherwise map to '101'
        X_copy['AAV2020'] = X_copy['AAV2020'].apply(
            lambda x: x if pd.isna(x) or x in self.kept_categories_ else '101'
        )
        
        return X_copy
