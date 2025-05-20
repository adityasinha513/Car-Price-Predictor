from flask import Flask, request, jsonify, send_from_directory
from src.models.train import load_model
from src.models.predict import predict_price
import os

app = Flask(__name__)

# Load the model and preprocessor
model, preprocessor = load_model()

@app.route('/')
def index():
    static_dir = os.path.join(os.path.dirname(__file__), 'static')
    return send_from_directory(static_dir, 'index.html')

@app.route('/predict', methods=['POST'])
def predict():
    """
    Endpoint for making car price predictions.
    
    Expected JSON input:
    {
        "name": "Car Name",
        "company": "Company Name",
        "year": 2020,
        "kms_driven": 50000,
        "fuel_type": "Petrol"
    }
    """
    try:
        # Get input data
        car_details = request.get_json()
        
        # Make prediction
        prediction = predict_price(model, preprocessor, car_details)
        
        return jsonify({
            'status': 'success',
            'predicted_price': float(prediction)
        })
    
    except ValueError as e:
        return jsonify({
            'status': 'error',
            'message': str(e)
        }), 400
    
    except Exception as e:
        print('Exception in /predict:', e)
        return jsonify({
            'status': 'error',
            'message': 'An unexpected error occurred'
        }), 500

if __name__ == '__main__':
    app.run(debug=True) 