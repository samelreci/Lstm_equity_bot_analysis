import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
import numpy as np

def preprocess_data(stock_data):
    # Assume last_row_with_missing_data is a Timestamp index
    last_row_with_missing_data = stock_data[stock_data.isnull().any(axis=1)].index[-1]

    # Convert integer to Timedelta object and add it to Timestamp index
    new_index = last_row_with_missing_data + pd.Timedelta(days=1)

    # Delete rows from the first row to the last row with missing data
    stock_data = stock_data.loc[new_index:]

    X = stock_data[['Open', 'Close', 'Low', 'Volume', 'Price_Change', 'ATR', 'Bollinger_Bands', 'MACD', 'Stochastic_Oscillator']]
    y = stock_data['Close']

    # Smoothing using Exponential Moving Average (EMA)
    def exponential_moving_average(data, alpha):
        smoothed_data = np.zeros_like(data)
        smoothed_data[0] = data[0]  # Initialize with the first data point
        for i in range(1, len(data)):
            smoothed_data[i] = alpha * data[i] + (1 - alpha) * smoothed_data[i - 1]
        return smoothed_data

    alpha = 0.5  # Smoothing factor (adjust as needed)

    # Smooth each feature in X separately
    X_smoothed = pd.DataFrame()
    for col in X.columns:
        X_smoothed[col] = exponential_moving_average(X[col].values, alpha)

    # Smooth the target variable y (Close)
    y_smoothed = exponential_moving_average(y.values, alpha)

    # Convert y_smoothed to numpy array
    y_smoothed = np.array(y_smoothed)
    X = X_smoothed
    

    # Define the number of lagged time steps
    lag_steps = 8

    # Create lagged features for each original feature
    lagged_features = []
    for col in X.columns:
        for i in range(1, lag_steps + 1):
            new_col_name = f'{col}_lag_{i}'
            X[new_col_name] = X[col].shift(i)
            lagged_features.append(new_col_name)

    # Drop original features and columns where all lagged features are concatenated together
    X = X.drop(columns=X.columns[:len(X.columns) // lag_steps])

    # Drop rows with NaN values resulting from shifting
    X.dropna(inplace=True)
    y_smoothed = y_smoothed[X.index]  # Align y with X after dropping NaN rows
    y = y[X.index]

    # Scale Data
    scaler = MinMaxScaler()
    X_scaled = scaler.fit_transform(X)
    y_smoothed = np.array(y_smoothed)
    y_scaled = scaler.fit_transform(y_smoothed.reshape(-1, 1))

    # Split the data into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(X_scaled, y_scaled, test_size=0.2, random_state=42, shuffle=False)
    X_train_unsooth, X_test_unsmooth, y_train_unsmooth, y_test_unsmooth = train_test_split(X, y, test_size=0.2, random_state=42, shuffle=False)

    print(X_train)
    print(y_train)
    
    return X_train, X_test, y_train, y_test, scaler,y_test_unsmooth  
