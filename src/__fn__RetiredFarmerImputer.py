from sklearn.base import BaseEstimator, TransformerMixin

# Custom transformer for retired farmers imputation
class RetiredFarmerImputer(BaseEstimator, TransformerMixin):
    """
    Impute retired job columns for retired farmers (Occupation_42 starting with 'csp_1' AND activity_type == 'type2_1').
    
    This imputer fills missing retired job information for farmers based on:
    - Occupation code (csp_1 = agriculteurs)
    - Sex for retirement income estimation
    - Standard agricultural job characteristics
    """
    
    def __init__(self):
        pass
    
    def fit(self, X, y=None):
        return self
    
    def transform(self, X):
        X_copy = X.copy()
        
        # Identify retired farmers: Occupation_42 starts with 'csp_7_1' AND activity_type == 'type2_1'
        if 'Occupation_42' in X_copy.columns and 'activity_type' in X_copy.columns:
            retired_farmer_mask = (X_copy['Occupation_42'].astype(str).str.startswith('csp_7_1')) & (X_copy['activity_type'] == 'type2_1')
            
            # Fill job_desc_retired
            if 'job_desc_retired' in X_copy.columns:
                X_copy.loc[retired_farmer_mask & X_copy['job_desc_retired'].isna(), 'job_desc_retired'] = '100x'
            
            # Fill Work_condition_retired
            if 'Work_condition_retired' in X_copy.columns:
                X_copy.loc[retired_farmer_mask & X_copy['Work_condition_retired'].isna(), 'Work_condition_retired'] = 'C'
            
            # Fill terms_of_emp_retired
            if 'terms_of_emp_retired' in X_copy.columns:
                X_copy.loc[retired_farmer_mask & X_copy['terms_of_emp_retired'].isna(), 'terms_of_emp_retired'] = 'AUT'
            
            # Fill RETIREMENT_INCOME based on sex
            if 'RETIREMENT_INCOME' in X_copy.columns and 'sex' in X_copy.columns:
                retirement_income_mask = retired_farmer_mask & X_copy['RETIREMENT_INCOME'].isna()
                
                # Female -> 870
                female_mask = retirement_income_mask & (X_copy['sex'] == 'Female')
                X_copy.loc[female_mask, 'RETIREMENT_INCOME'] = 870
                
                # Male -> 1080
                male_mask = retirement_income_mask & (X_copy['sex'] == 'Male')
                X_copy.loc[male_mask, 'RETIREMENT_INCOME'] = 1080
            
            # Fill OCCUPATIONAL_STATUS_retired
            if 'OCCUPATIONAL_STATUS_retired' in X_copy.columns:
                X_copy.loc[retired_farmer_mask & X_copy['OCCUPATIONAL_STATUS_retired'].isna(), 'OCCUPATIONAL_STATUS_retired'] = 'O'
            
            # Fill ECONOMIC_SECTOR_retired
            if 'ECONOMIC_SECTOR_retired' in X_copy.columns:
                X_copy.loc[retired_farmer_mask & X_copy['ECONOMIC_SECTOR_retired'].isna(), 'ECONOMIC_SECTOR_retired'] = 'AZ'
            
            # Fill EMPLOYER_TYPE_retired
            if 'EMPLOYER_TYPE_retired' in X_copy.columns:
                X_copy.loc[retired_farmer_mask & X_copy['EMPLOYER_TYPE_retired'].isna(), 'EMPLOYER_TYPE_retired'] = 'ct_0'
            
            # Fill WORKING_HOURS_retired
            if 'WORKING_HOURS_retired' in X_copy.columns:
                X_copy.loc[retired_farmer_mask & X_copy['WORKING_HOURS_retired'].isna(), 'WORKING_HOURS_retired'] = 2667
            
            # Fill Job_dep_retired (first two digits of Insee_code)
            if 'Job_dep_retired' in X_copy.columns and 'Insee_code' in X_copy.columns:
                job_dep_mask = retired_farmer_mask & X_copy['Job_dep_retired'].isna()
                X_copy.loc[job_dep_mask, 'Job_dep_retired'] = X_copy.loc[job_dep_mask, 'Insee_code'].astype(str).str[:2]
            
            # Fill Employee_count_retired
            if 'Employee_count_retired' in X_copy.columns:
                X_copy.loc[retired_farmer_mask & X_copy['Employee_count_retired'].isna(), 'Employee_count_retired'] = 'unknown'

            # Fill N3_retired
            if 'N3_retired' in X_copy.columns:
                X_copy.loc[retired_farmer_mask & X_copy['N3_retired'].isna(), 'N3_retired'] = '100x'
            
            # Fill N2_retired with value from Occupation_42
            if 'N2_retired' in X_copy.columns:
                n2_mask = retired_farmer_mask & X_copy['N2_retired'].isna()
                X_copy.loc[n2_mask, 'N2_retired'] = X_copy.loc[n2_mask, 'Occupation_42']
            
            # Fill N1_retired
            if 'N1_retired' in X_copy.columns:
                X_copy.loc[retired_farmer_mask & X_copy['N1_retired'].isna(), 'N1_retired'] = 'csp_1'
        
        return X_copy