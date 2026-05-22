import pandas as pd
import numpy_financial as npf

def prepare_applicant_data(form_data, model_features):
    """
    Takes raw form data, engineers new features, and returns a 
    one-hot encoded DataFrame ready for prediction.
    """
    # Calculate installment
    installment = npf.pmt(form_data['int_rate'] / 12 / 100, form_data['term'], -form_data['loan_amnt'])
    
    applicant_data = {
        'loan_amnt': form_data['loan_amnt'], 
        'term': form_data['term'], 
        'int_rate': form_data['int_rate'],
        'installment': installment,
        'grade': form_data['grade'], 
        'emp_length': form_data['emp_length'], 
        'home_ownership': form_data['home_ownership'],
        'annual_inc': form_data['annual_inc'], 
        'verification_status': 'Verified',
        'purpose': form_data['purpose'], 
        'dti': form_data['dti'], 
        'fico_score': form_data['fico_score'],
        'revol_util': form_data['revol_util'], 
        'total_acc': form_data['total_acc'], 
        'open_acc': form_data['open_acc'], 
        'revol_bal': form_data['revol_bal'],
    }
    
    df = pd.DataFrame([applicant_data])
    
    # Feature Engineering
    df['loan_to_income_ratio'] = df['loan_amnt'] / (df['annual_inc'] + 0.01)
    df['installment_to_income_ratio'] = (df['installment'] * 12) / (df['annual_inc'] + 0.01)
    df['revol_bal_to_income_ratio'] = df['revol_bal'] / (df['annual_inc'] + 0.01)
    df['credit_history_length'] = 15 # Placeholder median
    df['avg_credit_line'] = df['revol_bal'] / (df['open_acc'] + 0.01)
    df['fico_dti_interaction'] = df['fico_score'] / (df['dti'] + 0.01)
    
    # One-hot encoding
    df_encoded = pd.get_dummies(df)
    
    # Reindex to match training features exactly
    final_df = df_encoded.reindex(columns=model_features, fill_value=0)
    
    return final_df
