import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from xgboost import XGBRegressor
from sklearn.metrics import r2_score, mean_absolute_error
import joblib
from src.data.preprocess import transform_data

def train_models(X_train, X_test, y_train, y_test, preprocessor):
    """
    Train multiple models and return the best performing one.
    
    Args:
        X_train, X_test: Training and test features
        y_train, y_test: Training and test targets
        preprocessor: Fitted preprocessor
        
    Returns:
        tuple: (best_model, best_score, model_metrics)
    """
    # Fit the preprocessor on training data
    preprocessor.fit(X_train)
    # Transform the data
    X_train_transformed = preprocessor.transform(X_train)
    X_test_transformed = preprocessor.transform(X_test)
    
    # Initialize models
    models = {
        'Linear Regression': LinearRegression(),
        'Random Forest': RandomForestRegressor(n_estimators=100, random_state=42),
        'XGBoost': XGBRegressor(n_estimators=100, random_state=42)
    }
    
    # Train and evaluate models
    best_score = -np.inf
    best_model = None
    model_metrics = {}
    
    for name, model in models.items():
        # Train model
        model.fit(X_train_transformed, y_train)
        
        # Make predictions
        y_pred = model.predict(X_test_transformed)
        
        # Calculate metrics
        r2 = r2_score(y_test, y_pred)
        mae = mean_absolute_error(y_test, y_pred)
        
        model_metrics[name] = {
            'R2 Score': r2,
            'Mean Absolute Error': mae
        }
        
        # Update best model
        if r2 > best_score:
            best_score = r2
            best_model = model
    
    return best_model, best_score, model_metrics

def save_model(model, preprocessor, model_path='model.joblib', preprocessor_path='preprocessor.joblib'):
    """Save the trained model and preprocessor."""
    joblib.dump(model, model_path)
    joblib.dump(preprocessor, preprocessor_path)

def load_model(model_path='model.joblib', preprocessor_path='preprocessor.joblib'):
    """Load the trained model and preprocessor."""
    model = joblib.load(model_path)
    preprocessor = joblib.load(preprocessor_path)
    return model, preprocessor 