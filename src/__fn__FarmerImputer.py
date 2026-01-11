from sklearn.base import BaseEstimator, TransformerMixin

# Custom transformer for farmers (agriculteurs) imputation
class FarmerImputer(BaseEstimator, TransformerMixin):
    """
    Impute job columns for farmers (Occupation_42 starting with 'csp_1') with agriculture-specific values.
    
    This imputer fills missing job-related information for farmers based on:
    - Occupation code (csp_1 = agriculteurs)
    - Education level (HIGHEST_DIPLOMA) for earnings estimation
    - Standard agricultural job characteristics
    """
    
    def __init__(self):
        pass
    
    def fit(self, X, y=None):
        return self
    
    def transform(self, X):
        X_copy = X.copy()
        
        # Identify farmers: Occupation_42 starts with 'csp_1'
        if 'Occupation_42' in X_copy.columns:
            farmer_mask = X_copy['Occupation_42'].astype(str).str.startswith('csp_1')
            
            # Fill job_desc_current
            if 'job_desc_current' in X_copy.columns:
                X_copy.loc[farmer_mask & X_copy['job_desc_current'].isna(), 'job_desc_current'] = '100x'
            
            # Fill Work_condition_current
            if 'Work_condition_current' in X_copy.columns:
                X_copy.loc[farmer_mask & X_copy['Work_condition_current'].isna(), 'Work_condition_current'] = 'C'
            
            # Fill terms_of_emp_current
            if 'terms_of_emp_current' in X_copy.columns:
                X_copy.loc[farmer_mask & X_copy['terms_of_emp_current'].isna(), 'terms_of_emp_current'] = 'AUT'
            
            # Fill Earnings_current based on HIGHEST_DIPLOMA
            if 'Earnings_current' in X_copy.columns and 'HIGHEST_DIPLOMA' in X_copy.columns:
                # Create mask for missing earnings among farmers
                earnings_mask = farmer_mask & X_copy['Earnings_current'].isna()
                
                # EDU1.8 or EDU1.9 -> 28600
                high_edu_mask = earnings_mask & X_copy['HIGHEST_DIPLOMA'].isin(['EDU1.8', 'EDU1.9'])
                X_copy.loc[high_edu_mask, 'Earnings_current'] = 28600
                
                # EDU1.6 or EDU1.7 -> 24900
                mid_high_edu_mask = earnings_mask & X_copy['HIGHEST_DIPLOMA'].isin(['EDU1.6', 'EDU1.7'])
                X_copy.loc[mid_high_edu_mask, 'Earnings_current'] = 24900
                
                # EDU1.3, EDU1.4, EDU1.5 -> 22300
                mid_edu_mask = earnings_mask & X_copy['HIGHEST_DIPLOMA'].isin(['EDU1.3', 'EDU1.4', 'EDU1.5'])
                X_copy.loc[mid_edu_mask, 'Earnings_current'] = 22300
                
                # All others -> 20500
                remaining_mask = earnings_mask & ~(high_edu_mask | mid_high_edu_mask | mid_edu_mask)
                X_copy.loc[remaining_mask, 'Earnings_current'] = 20500
            
            # Fill OCCUPATIONAL_STATUS_current
            if 'OCCUPATIONAL_STATUS_current' in X_copy.columns:
                X_copy.loc[farmer_mask & X_copy['OCCUPATIONAL_STATUS_current'].isna(), 'OCCUPATIONAL_STATUS_current'] = 'O'
            
            # Fill ECONOMIC_SECTOR_current
            if 'ECONOMIC_SECTOR_current' in X_copy.columns:
                X_copy.loc[farmer_mask & X_copy['ECONOMIC_SECTOR_current'].isna(), 'ECONOMIC_SECTOR_current'] = 'AZ'
            
            # Fill EMPLOYER_TYPE_current
            if 'EMPLOYER_TYPE_current' in X_copy.columns:
                X_copy.loc[farmer_mask & X_copy['EMPLOYER_TYPE_current'].isna(), 'EMPLOYER_TYPE_current'] = 'ct_0'
            
            # Fill WORKING_HOURS_current
            if 'WORKING_HOURS_current' in X_copy.columns:
                X_copy.loc[farmer_mask & X_copy['WORKING_HOURS_current'].isna(), 'WORKING_HOURS_current'] = 2667
            
            # Fill Job_dep_current (first two digits of Insee_code)
            if 'Job_dep_current' in X_copy.columns and 'Insee_code' in X_copy.columns:
                job_dep_mask = farmer_mask & X_copy['Job_dep_current'].isna()
                X_copy.loc[job_dep_mask, 'Job_dep_current'] = X_copy.loc[job_dep_mask, 'Insee_code'].astype(str).str[:2]
            
            # Fill Employee_count_current
            if 'Employee_count_current' in X_copy.columns:
                X_copy.loc[farmer_mask & X_copy['Employee_count_current'].isna(), 'Employee_count_current'] = 'unknown'

            # Fill N3
            if 'N3_current' in X_copy.columns:
                X_copy.loc[farmer_mask & X_copy['N3_current'].isna(), 'N3_current'] = '100x'
            
            # Fill N2 with value from Occupation_42
            if 'N2_current' in X_copy.columns:
                n2_mask = farmer_mask & X_copy['N2_current'].isna()
                X_copy.loc[n2_mask, 'N2_current'] = X_copy.loc[n2_mask, 'Occupation_42']
            
            # Fill N1
            if 'N1_current' in X_copy.columns:
                X_copy.loc[farmer_mask & X_copy['N1_current'].isna(), 'N1_current'] = 'csp_1'
        
        return X_copy