import pandas as pd

def clean_data(df):
    # Convert columns to appropriate data types
    df['TotalCharges'] = pd.to_numeric(df['TotalCharges'], errors='coerce')
    df.fillna(df['TotalCharges'].mean(), inplace=True)
    df.drop(columns=['customerID'], inplace=True)
    
    # Convert "Yes"/"No" columns to boolean
    obj_columns = [
        'Churn', 'Partner', 'Dependents', 'PhoneService', 'OnlineSecurity', 
        'OnlineBackup', 'DeviceProtection', 'TechSupport', 'StreamingTV', 
        'StreamingMovies', 'PaperlessBilling'
    ]
    for col in obj_columns:
        df[col] = df[col].apply(lambda x: True if x == 'Yes' else False)
    
    # One-hot encode categorical features
    df = pd.get_dummies(df, columns=['gender', 'InternetService', 'Contract', 'PaymentMethod'])
    
    return df
