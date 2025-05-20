from src.data.preprocess import preprocess_data
from src.models.train import train_models, save_model

def main():
    # Preprocess data
    print("Preprocessing data...")
    X_train, X_test, y_train, y_test, preprocessor = preprocess_data('quikr_car.csv')
    
    # Train models
    print("Training models...")
    best_model, best_score, model_metrics = train_models(
        X_train, X_test, y_train, y_test, preprocessor
    )
    
    # Print model metrics
    print("\nModel Performance Metrics:")
    for model_name, metrics in model_metrics.items():
        print(f"\n{model_name}:")
        for metric_name, value in metrics.items():
            print(f"{metric_name}: {value:.4f}")
    
    # Save the best model
    print("\nSaving the best model...")
    save_model(best_model, preprocessor)
    print("Model saved successfully!")

if __name__ == '__main__':
    main() 