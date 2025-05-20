import pandas as pd
import numpy as np
from src.data.preprocess import transform_data

def validate_input(car_details):
    """
    Validate input car details.
    
    Args:
        car_details (dict): Dictionary containing car features
        
    Returns:
        bool: True if input is valid, False otherwise
    """
    required_fields = ['name', 'company', 'year', 'kms_driven', 'fuel_type']
    
    # Check if all required fields are present
    if not all(field in car_details for field in required_fields):
        return False
    
    # Validate year
    try:
        year = int(car_details['year'])
        if year < 1900 or year > 2024:
            return False
    except ValueError:
        return False
    
    # Validate kms_driven
    try:
        kms = float(car_details['kms_driven'])
        if kms < 0:
            return False
    except ValueError:
        return False
    
    # Validate fuel_type
    valid_fuel_types = ['Petrol', 'Diesel', 'CNG', 'Electric', 'Hybrid']
    if car_details['fuel_type'] not in valid_fuel_types:
        return False
    
    return True

def predict_price(model, preprocessor, car_details):
    """
    Predict car price using the trained model.
    
    Args:
        model: Trained model
        preprocessor: Fitted preprocessor
        car_details (dict): Dictionary containing car features
        
    Returns:
        float: Predicted price
    """
    # Validate input
    if not validate_input(car_details):
        raise ValueError("Invalid input car details")
    
    # Convert input to DataFrame
    input_df = pd.DataFrame([car_details])
    
    # Transform input
    input_transformed = transform_data(preprocessor, input_df)
    
    # Make prediction
    prediction = model.predict(input_transformed)[0]
    
    return max(0, prediction)  # Ensure non-negative price 