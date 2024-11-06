import pandas as pd

def engineer_features(df):
    df['MultipleLines'] = df['MultipleLines'].apply(lambda x: True if x == 'Yes' else False)
    binary_columns = ['SeniorCitizen']
    df[binary_columns] = df[binary_columns].astype(bool)
    return df



