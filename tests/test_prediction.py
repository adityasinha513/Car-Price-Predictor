import requests
import json

# Test data
test_data = {
    "name": "Swift",
    "company": "Maruti",
    "year": 2020,
    "kms_driven": 50000,
    "fuel_type": "Petrol"
}

# Make prediction request
response = requests.post(
    'http://127.0.0.1:5000/predict',
    json=test_data
)

# Print response
print("Status Code:", response.status_code)
print("Response:", response.json()) 