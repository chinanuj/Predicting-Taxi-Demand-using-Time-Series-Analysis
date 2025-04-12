# Predicting-Taxi-Demand-using-Time-Series-Analysis

## Overview
This project implements an end-to-end framework for forecasting taxi demand using time series analysis techniques. We develop and compare multiple approaches including deep learning models (LSTM and GRU) and classical statistical methods (ARIMA) to predict taxi demand patterns based on historical data from the NYC Taxi and Limousine Commission (TLC).

## Authors
- Abhimanyu Gupta (B22BB001)
- Raj Vijayvargia (B22AI064)
- Jagdish Suthar (B22AI067)
- Anuj Chincholikar (B22ES018)

## Dataset
The project utilizes the NYC TLC 2025 yellow taxi trip data in parquet format. This dataset contains rich temporal and spatial information including pickup/dropoff times, locations, fare amounts, and passenger counts.

## Methodology

### Data Preprocessing
- **Data Cleaning**: Converted timestamps, removed inconsistencies, imputed missing values, and filtered outliers
- **Feature Engineering**: 
  - Extracted temporal features (hour, day, month, weekday, weekend indicators)
  - Applied cyclical encoding using sine/cosine transformations
  - Aggregated trip records by location and time intervals
  - Generated lag features and rolling statistics

### Models Implemented
1. **LSTM Model**
   - Stacked LSTM layers followed by fully-connected output layer
   - Trained with MSE loss and Adam optimizer
   - Gradient clipping applied for training stability

2. **GRU Model**
   - Similar architecture to LSTM but using GRU layers
   - Same training approach as LSTM

3. **ARIMA Model**
   - Classical statistical approach using ARIMA(2,1,2)
   - Applied to hourly aggregated demand from top pickup locations
   - Stationarity verified using Augmented Dickey-Fuller test

## Results

### Performance Metrics

| Model | MSE | RMSE | MAE | R² | MAPE (%) |
|-------|-----|------|-----|-----|----------|
| LSTM | 0.8316 | 0.9119 | 0.6548 | -0.0179 | 30.57 |
| GRU | 0.8598 | 0.9273 | 0.6936 | -0.0524 | 34.53 |
| ARIMA | - | - | - | - | 49.23 |

### Key Findings
- Deep learning models (LSTM/GRU) effectively capture short-term fluctuations but struggle with absolute demand magnitude
- ARIMA provides a more accurate baseline with lower MAPE, excelling at overall trend capture
- Statistical methods remain competitive despite the flexibility of deep learning approaches

## Project Structure
```
├── data/
│   └── nyc_taxi_2025/                # Raw and processed taxi data
├── notebooks/
│   ├── 01_data_preprocessing.ipynb   # Data cleaning and feature engineering
│   ├── 02_exploratory_analysis.ipynb # EDA and visualizations
│   ├── 03_lstm_gru_models.ipynb      # Deep learning implementation
│   └── 04_arima_model.ipynb          # Statistical modeling
├── src/
│   ├── preprocessing/                # Data preprocessing modules
│   ├── models/                       # Model implementations
│   └── evaluation/                   # Evaluation metrics and visualization
├── results/
│   └── figures/                      # Generated plots and visualizations
└── requirements.txt                  # Project dependencies
```

## Limitations
- Data quality issues may affect model performance
- Deep learning models require significant tuning and risk overfitting
- ARIMA offers better interpretability than "black box" deep learning approaches

## Future Work
- Develop hybrid architectures combining statistical and deep learning approaches
- Incorporate exogenous variables (weather, events, traffic conditions)
- Implement systematic hyperparameter optimization
- Create real-time forecasting framework with dynamic model updates

## Setup and Installation

1. Clone the repository
```bash
git clone https://github.com/yourusername/taxi-demand-prediction.git
cd taxi-demand-prediction
```

2. Create and activate a virtual environment
```bash
python -m venv env
source env/bin/activate  # On Windows: env\Scripts\activate
```

3. Install dependencies
```bash
pip install -r requirements.txt
```

4. Run the notebooks in sequence or execute the main script
```bash
jupyter notebook notebooks/
# OR
python src/main.py
```

## Requirements
- Python 3.8+
- PyTorch
- Pandas
- NumPy
- Matplotlib
- Statsmodels
- Scikit-learn

## References
- Xu, Y., et al. (2018). Taxi Demand Prediction Using Deep Convolutional Neural Networks embedded with Recurrent Neural Networks.
- Laptev, N., et al. (2017). Time-series Extreme Event Forecasting with Neural Networks at Uber.
- Moreira-Matias, L., et al. (2013). Predicting Taxi-Passenger Demand Using Streaming Data.
- Zhang, J., et al. (2017). Deep Spatio-Temporal Residual Networks for Citywide Crowd Flows Prediction.
