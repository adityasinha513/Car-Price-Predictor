import pytest
from ..models.predict import validate_input, predict_price
from ..models.train import load_model

def test_validate_input():
    # Test valid input
    valid_input = {
        'name': 'Toyota Camry',
        'company': 'Toyota',
        'year': 2020,
        'kms_driven': 50000,
        'fuel_type': 'Petrol'
    }
    assert validate_input(valid_input) is True
    
    # Test invalid year
    invalid_year = valid_input.copy()
    invalid_year['year'] = 1800
    assert validate_input(invalid_year) is False
    
    # Test invalid kms
    invalid_kms = valid_input.copy()
    invalid_kms['kms_driven'] = -1000
    assert validate_input(invalid_kms) is False
    
    # Test invalid fuel type
    invalid_fuel = valid_input.copy()
    invalid_fuel['fuel_type'] = 'Invalid'
    assert validate_input(invalid_fuel) is False
    
    # Test missing field
    missing_field = valid_input.copy()
    del missing_field['year']
    assert validate_input(missing_field) is False

def test_predict_price():
    # Load model and preprocessor
    model, preprocessor = load_model()
    
    # Test valid prediction
    valid_input = {
        'name': 'Toyota Camry',
        'company': 'Toyota',
        'year': 2020,
        'kms_driven': 50000,
        'fuel_type': 'Petrol'
    }
    
    prediction = predict_price(model, preprocessor, valid_input)
    assert isinstance(prediction, float)
    assert prediction >= 0
    
    # Test invalid input
    invalid_input = {
        'name': 'Toyota Camry',
        'company': 'Toyota',
        'year': 1800,  # Invalid year
        'kms_driven': 50000,
        'fuel_type': 'Petrol'
    }
    
    with pytest.raises(ValueError):
        predict_price(model, preprocessor, invalid_input) 