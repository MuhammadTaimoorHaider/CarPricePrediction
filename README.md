# Car Price Prediction System

A supervised machine learning regression project that predicts a car's **MSRP (Manufacturer's Suggested Retail Price)** based on its specifications, using a Random Forest Regressor.

---

## Overview

This project demonstrates an end-to-end ML workflow — from data loading and preprocessing to model training, evaluation, and interactive prediction. It uses a 100-sample toy dataset sampled from a larger car pricing dataset.

**Real-world applications:** Car dealership pricing, insurance valuation, used car marketplaces, rental car pricing, loan collateral assessment.

---

## Input & Output

### Input Features

| Feature              | Type        | Description                            |
|----------------------|-------------|----------------------------------------|
| Engine HP            | Numerical   | Horsepower of the engine (55–1001)     |
| Year                 | Numerical   | Model year of the car (1990–2017)      |
| Engine Cylinders     | Numerical   | Number of cylinders (0, 3, 4, 6, 8…)  |
| City MPG             | Numerical   | City fuel efficiency (7–137)           |
| Highway MPG          | Numerical   | Highway fuel efficiency                |
| Popularity           | Numerical   | Popularity score of the car model      |
| Market Category      | Categorical | Vehicle segment (Luxury, Crossover…)   |
| Vehicle Style        | Categorical | Body style (Sedan, SUV, Coupe…)        |

### Output

| Attribute | Description                                   |
|-----------|-----------------------------------------------|
| MSRP      | Manufacturer's Suggested Retail Price (in $)  |

---

## Sample Data

| Engine HP | Year | Engine Cylinders | City MPG | Market Category        | MSRP    |
|-----------|------|------------------|----------|------------------------|---------|
| 162       | 1991 | 4                | 17       | Luxury, Performance    | $2,000  |
| 365       | 2016 | 6                | 15       | Crossover              | $42,600 |
| 620       | 2011 | 12               | 10       | Exotic, High-Perf      | $463,000|

---

## Project Structure

```
Car Price Prediction/
├── car.ipynb                          # Main notebook (v1)
├── cars_price_prediction_regression.ipynb  # Main notebook (v2)
├── cars_price_prediction_regression.html   # HTML export of notebook
├── car_price.csv                      # Original full dataset
├── car_price_toy_data.csv             # 100-sample toy dataset
├── toy_data.csv                       # Alternative toy dataset
├── car_price_encoded_dataset.csv      # Label-encoded dataset (v1)
├── car_price_encoded_dataset2.csv     # Label-encoded dataset (v2)
├── car_price_predictor.pkl            # Saved trained model (v1)
├── car_price_predictor2.pkl           # Saved trained model (v2)
└── README.md
```

---

## ML Pipeline

1. **Load Data** — Sample 100 records from `car_price.csv`
2. **Preprocess** — Handle missing values, standardize text formatting
3. **Feature Extraction** — Select 8 most relevant features
4. **Encoding** — `LabelEncoder` for categoricals + `StandardScaler` / `OneHotEncoder` via `ColumnTransformer`
5. **Train** — `RandomForestRegressor` (100 estimators) inside a sklearn `Pipeline`
6. **Evaluate** — MAE, MSE, RMSE, R²
7. **Predict** — Interactive user input → predicted MSRP

---

## Model Performance

| Metric | Value |
|--------|-------|
| MAE    | ~$10,275 |
| RMSE   | ~$31,620 |
| R²     | 0.627    |

> Model trained on 70 samples, tested on 30 samples (70/30 split).

---

## Tech Stack

| Tool | Version |
|------|---------|
| Python | 3.10.16 |
| NumPy | 1.26.4 |
| Pandas | 2.2.3 |
| scikit-learn | 1.7.0 |
| PrettyTable | 3.16.0 |
| IDE | Jupyter Notebook |

---

## How to Run

1. Clone the repository:
   ```bash
   git clone https://github.com/MuhammadTaimoorHaider/CarPricePrediction.git
   cd CarPricePrediction
   ```

2. Install dependencies:
   ```bash
   pip install numpy pandas scikit-learn prettytable matplotlib
   ```

3. Open and run the notebook:
   ```bash
   jupyter notebook cars_price_prediction_regression.ipynb
   ```

4. For interactive prediction, run **Step 8** in the notebook and enter car specifications when prompted.

