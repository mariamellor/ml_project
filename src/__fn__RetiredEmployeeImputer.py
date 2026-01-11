from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.ensemble import HistGradientBoostingRegressor, HistGradientBoostingClassifier
from sklearn.preprocessing import LabelEncoder
import pandas as pd

# Custom transformer for retired employee imputation using ML
class RetiredEmployeeImputer(BaseEstimator, TransformerMixin):
    """
    Impute retired job columns for retired employees (activity_type == 'type2_1') with missing values.
    
    Uses machine learning models trained on retired people data (Occupation_42 starts with 'csp_7') to impute:
    - WORKING_HOURS_retired (HistGradientBoostingRegressor)
    - RETIREMENT_INCOME (HistGradientBoostingRegressor)
    - ECONOMIC_SECTOR_retired (HistGradientBoostingClassifier)
    - terms_of_emp_retired (HistGradientBoostingClassifier)
    - EMPLOYER_TYPE_retired (HistGradientBoostingClassifier)
    
    Other fields are filled based on Occupation_42 or standard employment values.
    """
    
    def __init__(self):
        self.hours_model_ = None
        self.earnings_model_ = None
        self.sector_model_ = None
        self.sector_encoder_ = None
        self.terms_model_ = None
        self.terms_encoder_ = None
        self.employer_model_ = None
        self.employer_encoder_ = None
        self.feature_cols_ = None
    
    def fit(self, X, y=None):
        X_copy = X.copy()
        
        # Filter for retired people (Occupation_42 starts with 'csp_7')
        if 'Occupation_42' not in X_copy.columns:
            print("Warning: Occupation_42 column not found. Cannot train imputation models.")
            return self
        
        retired_mask = X_copy['Occupation_42'].astype(str).str.startswith('csp_7')
        X_retired = X_copy[retired_mask].copy()
        
        print(f"\nTraining RetiredEmployeeImputer models on {len(X_retired)} retired people...")
        
        # Select features for imputation models
        potential_features = ['age', 'HIGHEST_DIPLOMA', 'Household', 'Occupation_42', 
                            'JOB_SECURITY', 'department', 'Insee_code']
        self.feature_cols_ = [col for col in potential_features if col in X_retired.columns]
        
        if len(self.feature_cols_) == 0:
            print("Warning: No suitable features found for training imputation models.")
            return self
        
        # Prepare feature matrix with label encoding for categorical variables
        def prepare_features(X_subset):
            X_encoded = X_subset[self.feature_cols_].copy()
            for col in X_encoded.columns:
                if X_encoded[col].dtype == 'object' or X_encoded[col].dtype.name == 'category':
                    le = LabelEncoder()
                    X_encoded[col] = X_encoded[col].fillna('missing')
                    X_encoded[col] = le.fit_transform(X_encoded[col].astype(str))
            X_encoded = X_encoded.fillna(X_encoded.median())
            return X_encoded
        
        # Train WORKING_HOURS model
        if 'WORKING_HOURS_retired' in X_retired.columns:
            hours_data = X_retired[X_retired['WORKING_HOURS_retired'].notna()].copy()
            if len(hours_data) > 100:
                X_hours = prepare_features(hours_data)
                y_hours = hours_data['WORKING_HOURS_retired']
                self.hours_model_ = HistGradientBoostingRegressor(random_state=42, max_iter=100)
                self.hours_model_.fit(X_hours, y_hours)
                print(f"  ✓ WORKING_HOURS_retired model trained on {len(hours_data)} samples")
            else:
                print(f"  ✗ Not enough data for WORKING_HOURS_retired ({len(hours_data)} samples)")
        
        # Train RETIREMENT_INCOME model
        if 'RETIREMENT_INCOME' in X_retired.columns:
            earnings_data = X_retired[X_retired['RETIREMENT_INCOME'].notna()].copy()
            if len(earnings_data) > 100:
                X_earnings = prepare_features(earnings_data)
                y_earnings = earnings_data['RETIREMENT_INCOME']
                self.earnings_model_ = HistGradientBoostingRegressor(random_state=42, max_iter=100)
                self.earnings_model_.fit(X_earnings, y_earnings)
                print(f"  ✓ RETIREMENT_INCOME model trained on {len(earnings_data)} samples")
            else:
                print(f"  ✗ Not enough data for RETIREMENT_INCOME ({len(earnings_data)} samples)")
        
        # Train ECONOMIC_SECTOR model
        if 'ECONOMIC_SECTOR_retired' in X_retired.columns:
            sector_data = X_retired[X_retired['ECONOMIC_SECTOR_retired'].notna()].copy()
            if len(sector_data) > 100:
                X_sector = prepare_features(sector_data)
                y_sector = sector_data['ECONOMIC_SECTOR_retired'].astype(str)
                self.sector_encoder_ = LabelEncoder()
                y_sector_encoded = self.sector_encoder_.fit_transform(y_sector)
                self.sector_model_ = HistGradientBoostingClassifier(random_state=42, max_iter=100, early_stopping=False)
                self.sector_model_.fit(X_sector, y_sector_encoded)
                print(f"  ✓ ECONOMIC_SECTOR_retired model trained on {len(sector_data)} samples")
            else:
                print(f"  ✗ Not enough data for ECONOMIC_SECTOR_retired ({len(sector_data)} samples)")
        
        # Train terms_of_emp model
        if 'terms_of_emp_retired' in X_retired.columns:
            terms_data = X_retired[X_retired['terms_of_emp_retired'].notna()].copy()
            if len(terms_data) > 100:
                X_terms = prepare_features(terms_data)
                y_terms = terms_data['terms_of_emp_retired'].astype(str)
                self.terms_encoder_ = LabelEncoder()
                y_terms_encoded = self.terms_encoder_.fit_transform(y_terms)
                self.terms_model_ = HistGradientBoostingClassifier(random_state=42, max_iter=100, early_stopping=False)
                self.terms_model_.fit(X_terms, y_terms_encoded)
                print(f"  ✓ terms_of_emp_retired model trained on {len(terms_data)} samples")
            else:
                print(f"  ✗ Not enough data for terms_of_emp_retired ({len(terms_data)} samples)")
        
        # Train EMPLOYER_TYPE model
        if 'EMPLOYER_TYPE_retired' in X_retired.columns:
            employer_data = X_retired[X_retired['EMPLOYER_TYPE_retired'].notna()].copy()
            if len(employer_data) > 100:
                X_employer = prepare_features(employer_data)
                y_employer = employer_data['EMPLOYER_TYPE_retired'].astype(str)
                self.employer_encoder_ = LabelEncoder()
                y_employer_encoded = self.employer_encoder_.fit_transform(y_employer)
                self.employer_model_ = HistGradientBoostingClassifier(random_state=42, max_iter=100, early_stopping=False)
                self.employer_model_.fit(X_employer, y_employer_encoded)
                print(f"  ✓ EMPLOYER_TYPE_retired model trained on {len(employer_data)} samples")
            else:
                print(f"  ✗ Not enough data for EMPLOYER_TYPE_retired ({len(employer_data)} samples)")
        
        return self
    
    def transform(self, X):
        X_copy = X.copy()
        
        # Identify retired employees: activity_type == 'type2_1'
        if 'Occupation_42' not in X_copy.columns or 'activity_type' not in X_copy.columns:
            return X_copy
        
        retired_employee_mask = X_copy['activity_type'] == 'type2_1'
        
        # Prepare features for ML imputation
        def prepare_features(X_subset):
            if self.feature_cols_ is None or len(self.feature_cols_) == 0:
                return None
            X_encoded = X_subset[self.feature_cols_].copy()
            for col in X_encoded.columns:
                if X_encoded[col].dtype == 'object' or X_encoded[col].dtype.name == 'category':
                    le = LabelEncoder()
                    X_encoded[col] = X_encoded[col].fillna('missing')
                    X_encoded[col] = le.fit_transform(X_encoded[col].astype(str))
            X_encoded = X_encoded.fillna(X_encoded.median())
            return X_encoded
        
        # ML-based imputation for retired employees
        if self.hours_model_ is not None and 'WORKING_HOURS_retired' in X_copy.columns:
            hours_missing_mask = retired_employee_mask & X_copy['WORKING_HOURS_retired'].isna()
            if hours_missing_mask.sum() > 0:
                X_to_predict = prepare_features(X_copy[hours_missing_mask])
                if X_to_predict is not None:
                    predictions = self.hours_model_.predict(X_to_predict)
                    X_copy.loc[hours_missing_mask, 'WORKING_HOURS_retired'] = predictions
        
        if self.earnings_model_ is not None and 'RETIREMENT_INCOME' in X_copy.columns:
            earnings_missing_mask = retired_employee_mask & X_copy['RETIREMENT_INCOME'].isna()
            if earnings_missing_mask.sum() > 0:
                X_to_predict = prepare_features(X_copy[earnings_missing_mask])
                if X_to_predict is not None:
                    predictions = self.earnings_model_.predict(X_to_predict)
                    X_copy.loc[earnings_missing_mask, 'RETIREMENT_INCOME'] = predictions
        
        if self.sector_model_ is not None and self.sector_encoder_ is not None and 'ECONOMIC_SECTOR_retired' in X_copy.columns:
            sector_missing_mask = retired_employee_mask & X_copy['ECONOMIC_SECTOR_retired'].isna()
            if sector_missing_mask.sum() > 0:
                X_to_predict = prepare_features(X_copy[sector_missing_mask])
                if X_to_predict is not None:
                    predictions_encoded = self.sector_model_.predict(X_to_predict)
                    predictions = self.sector_encoder_.inverse_transform(predictions_encoded)
                    X_copy.loc[sector_missing_mask, 'ECONOMIC_SECTOR_retired'] = predictions
        
        # Occupation_42-based imputation for job_desc_retired, N3_retired, N1_retired
        if 'job_desc_retired' in X_copy.columns:
            job_desc_missing_mask = retired_employee_mask & X_copy['job_desc_retired'].isna()
            if job_desc_missing_mask.sum() > 0:
                def extract_job_desc(occ_42):
                    if pd.isna(occ_42):
                        return None
                    digits = ''.join([c for c in str(occ_42) if c.isdigit()])
                    if digits:
                        return digits + '0x'
                    return None
                
                job_desc_values = X_copy.loc[job_desc_missing_mask, 'Occupation_42'].apply(extract_job_desc)
                X_copy.loc[job_desc_missing_mask, 'job_desc_retired'] = job_desc_values
        
        # Fill N3_retired with same value as job_desc_retired
        if 'N3_retired' in X_copy.columns and 'job_desc_retired' in X_copy.columns:
            n3_missing_mask = retired_employee_mask & X_copy['N3_retired'].isna()
            if n3_missing_mask.sum() > 0:
                X_copy.loc[n3_missing_mask, 'N3_retired'] = X_copy.loc[n3_missing_mask, 'job_desc_retired']
        
        # Fill N1_retired with first 5 characters of Occupation_42
        if 'N1_retired' in X_copy.columns:
            n1_missing_mask = retired_employee_mask & X_copy['N1_retired'].isna()
            if n1_missing_mask.sum() > 0:
                def extract_n1(occ_42):
                    if pd.isna(occ_42):
                        return None
                    return str(occ_42)[:5]
                
                n1_values = X_copy.loc[n1_missing_mask, 'Occupation_42'].apply(extract_n1)
                X_copy.loc[n1_missing_mask, 'N1_retired'] = n1_values
        
        # ML-based imputation for terms_of_emp_retired
        if self.terms_model_ is not None and self.terms_encoder_ is not None and 'terms_of_emp_retired' in X_copy.columns:
            terms_missing_mask = retired_employee_mask & X_copy['terms_of_emp_retired'].isna()
            if terms_missing_mask.sum() > 0:
                X_to_predict = prepare_features(X_copy[terms_missing_mask])
                if X_to_predict is not None:
                    predictions_encoded = self.terms_model_.predict(X_to_predict)
                    predictions = self.terms_encoder_.inverse_transform(predictions_encoded)
                    X_copy.loc[terms_missing_mask, 'terms_of_emp_retired'] = predictions
        
        # ML-based imputation for EMPLOYER_TYPE_retired
        if self.employer_model_ is not None and self.employer_encoder_ is not None and 'EMPLOYER_TYPE_retired' in X_copy.columns:
            employer_missing_mask = retired_employee_mask & X_copy['EMPLOYER_TYPE_retired'].isna()
            if employer_missing_mask.sum() > 0:
                X_to_predict = prepare_features(X_copy[employer_missing_mask])
                if X_to_predict is not None:
                    predictions_encoded = self.employer_model_.predict(X_to_predict)
                    predictions = self.employer_encoder_.inverse_transform(predictions_encoded)
                    X_copy.loc[employer_missing_mask, 'EMPLOYER_TYPE_retired'] = predictions
        
        # Standard value imputation for other fields
        # Fill OCCUPATIONAL_STATUS_retired
        if 'OCCUPATIONAL_STATUS_retired' in X_copy.columns:
            X_copy.loc[retired_employee_mask & X_copy['OCCUPATIONAL_STATUS_retired'].isna(), 'OCCUPATIONAL_STATUS_retired'] = 'O'
        
        # Fill Job_dep_retired (first two digits of Insee_code)
        if 'Job_dep_retired' in X_copy.columns and 'Insee_code' in X_copy.columns:
            job_dep_mask = retired_employee_mask & X_copy['Job_dep_retired'].isna()
            X_copy.loc[job_dep_mask, 'Job_dep_retired'] = X_copy.loc[job_dep_mask, 'Insee_code'].astype(str).str[:2]
        
        # Fill Employee_count_retired
        if 'Employee_count_retired' in X_copy.columns:
            X_copy.loc[retired_employee_mask & X_copy['Employee_count_retired'].isna(), 'Employee_count_retired'] = 'unknown'

        # Fill N2_retired with value from Occupation_42
        if 'N2_retired' in X_copy.columns:
            n2_mask = retired_employee_mask & X_copy['N2_retired'].isna()
            X_copy.loc[n2_mask, 'N2_retired'] = X_copy.loc[n2_mask, 'Occupation_42']
        
        # Fill Work_condition_retired based on WORKING_HOURS_retired
        if 'Work_condition_retired' in X_copy.columns and 'WORKING_HOURS_retired' in X_copy.columns:
            work_condition_mask = retired_employee_mask & X_copy['Work_condition_retired'].isna()
            X_copy.loc[work_condition_mask & (X_copy['WORKING_HOURS_retired'] > 1607), 'Work_condition_retired'] = 'C'
            X_copy.loc[work_condition_mask & (X_copy['WORKING_HOURS_retired'] <= 1607), 'Work_condition_retired'] = 'P'
        
        return X_copy
