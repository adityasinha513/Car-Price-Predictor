import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer

def clean_price(price):
    if pd.isna(price) or price == 'Ask For Price':
        return np.nan
    try:
        return float(str(price).replace(',', ''))
    except:
        return np.nan

def clean_kms(kms):
    if pd.isna(kms):
        return np.nan
    try:
        if isinstance(kms, str):
            return float(kms.replace(' kms', '').replace(',', ''))
        return float(kms)
    except:
        return np.nan

def preprocess_data(file_path):
    # Read the data
    df = pd.read_csv(file_path)
    
    # Clean the data
    df['Price'] = df['Price'].apply(clean_price)
    df['kms_driven'] = df['kms_driven'].apply(clean_kms)
    
    # Convert year to numeric, coerce errors to NaN
    df['year'] = pd.to_numeric(df['year'], errors='coerce')
    
    # Drop rows with missing values in target variable or numeric columns
    df = df.dropna(subset=['Price', 'year', 'kms_driven'])
    
    # Prepare features and target
    X = df[['name', 'company', 'year', 'kms_driven', 'fuel_type']]
    y = df['Price']
    
    # Split the data
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    # Create preprocessing pipeline
    categorical_features = ['name', 'company', 'fuel_type']
    numeric_features = ['year', 'kms_driven']
    
    categorical_transformer = Pipeline(steps=[
        ('imputer', SimpleImputer(strategy='constant', fill_value='missing')),
        ('onehot', OneHotEncoder(handle_unknown='ignore'))
    ])
    
    numeric_transformer = Pipeline(steps=[
        ('imputer', SimpleImputer(strategy='median'))
    ])
    
    preprocessor = ColumnTransformer(
        transformers=[
            ('cat', categorical_transformer, categorical_features),
            ('num', numeric_transformer, numeric_features)
        ])
    
    return X_train, X_test, y_train, y_test, preprocessor

def transform_data(preprocessor, X):
    """Transform new data using the preprocessor"""
    return preprocessor.transform(X) 