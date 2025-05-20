# Car Price Prediction

A machine learning project that predicts car prices based on various features using a Random Forest Regressor model.

## Project Structure

```
car-price-prediction/
├── data/                    # Data files
│   ├── raw/                # Original data
│   └── processed/          # Processed data
├── notebooks/              # Jupyter notebooks
├── src/                    # Source code
│   ├── models/            # Model training and prediction
│   ├── preprocessing/     # Data preprocessing
│   ├── static/           # Static files (HTML, CSS)
│   └── app.py            # Flask application
├── tests/                 # Test files
├── requirements.txt       # Project dependencies
└── README.md             # Project documentation
```

## Features

- Interactive web interface for car price prediction
- Dynamic form with company-specific car options
- Real-time price prediction using machine learning
- Modern and responsive UI design
- Loading spinner and error handling
- Input validation and user feedback

## Installation

1. Clone the repository:
```bash
git clone https://github.com/adityasinha513/CarPricePredictor.git
cd CarPricePredictor
```

2. Create a virtual environment:
```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

3. Install dependencies:
```bash
pip install -r requirements.txt
```

## Usage

1. Start the Flask server:
```bash
python -m src.app
```

2. Open your browser and navigate to:
```
http://127.0.0.1:5000
```

3. Fill in the car details and get the predicted price!

## Model Details

- Algorithm: Random Forest Regressor
- Features: Car name, company, year, kilometers driven, fuel type
- Evaluation Metrics: R² Score, Mean Absolute Error (MAE)

## Technologies Used

- Python
- Flask
- scikit-learn
- pandas
- HTML/CSS/JavaScript
- Bootstrap

## Contributing

1. Fork the repository
2. Create your feature branch (`git checkout -b feature/AmazingFeature`)
3. Commit your changes (`git commit -m 'Add some AmazingFeature'`)
4. Push to the branch (`git push origin feature/AmazingFeature`)
5. Open a Pull Request

## License

This project is licensed under the MIT License - see the LICENSE file for details.

## Author

Aditya Sinha
- GitHub: [@adityasinha513](https://github.com/adityasinha513) 