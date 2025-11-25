import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
import numpy as np

def preprocess_data(stock_data):
    
    missing_rows = stock_data[stock_data.isnull().any(axis=1)]
    if not missing_rows.empty:
        last_row_with_missing_data = missing_rows.index[-1]
        new_index = last_row_with_missing_data + pd.Timedelta(days=1)
        stock_data = stock_data.loc[new_index:]
    

    X = stock_data[['Open', 'Low', 'Volume',
                    'Price_Change', 'ATR',
                    'MACD']]
    y = stock_data['Close']

    # Smoothing using Exponential Moving Average (EMA)
    def exponential_moving_average(data, alpha):
        smoothed_data = np.zeros_like(data)
        smoothed_data[0] = data[0]  # Initialize with the first data point
        for i in range(1, len(data)):
            smoothed_data[i] = alpha * data[i] + (1 - alpha) * smoothed_data[i - 1]
        return smoothed_data

    alpha = 0.5  # Smoothing factor 

    # Smooth each feature in X separately
    X_smoothed = pd.DataFrame()
    for col in X.columns:
        X_smoothed[col] = exponential_moving_average(X[col].values, alpha)
 
    y_smoothed = exponential_moving_average(y.values, alpha)

    y_smoothed = np.array(y_smoothed)
    X = X_smoothed

    lag_steps = 8

    # Create lagged features for each original feature
    for col in X.columns:
        for i in range(1, lag_steps + 1):
            new_col_name = f'{col}_lag_{i}'
            X[new_col_name] = X[col].shift(i)

    num_original_features = len(X.columns) // (lag_steps + 1)
    X = X.drop(columns=X.columns[:num_original_features])

    X.dropna(inplace=True)

    y_smoothed = y_smoothed[X.index.to_numpy()]   
    y = y.iloc[X.index]                           

    # Scale Data
    scaler = MinMaxScaler()
    X_scaled = scaler.fit_transform(X)
    y_smoothed = np.array(y_smoothed)
    y_scaled = scaler.fit_transform(y_smoothed.reshape(-1, 1))

    # Split the data into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(
        X_scaled, y_scaled,
        test_size=0.2, random_state=42, shuffle=False
    )

    X_train_unsooth, X_test_unsmooth, y_train_unsmooth, y_test_unsmooth = train_test_split(
        X, y,
        test_size=0.2, random_state=42, shuffle=False
    )

    return X_train, X_test, y_train, y_test, scaler, y_test_unsmooth
