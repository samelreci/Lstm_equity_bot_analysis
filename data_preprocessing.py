import pandas as pd
from sklearn.preprocessing import MinMaxScaler
import numpy as np


def _ema(data: np.ndarray, alpha: float) -> np.ndarray:
    """Simple causal exponential moving average over a 1D numpy array."""
    data = np.asarray(data, dtype=float)
    if data.ndim != 1:
        raise ValueError("EMA expects a 1D array")
    out = np.empty_like(data, dtype=float)
    out[0] = data[0]
    for i in range(1, len(data)):
        out[i] = alpha * data[i] + (1.0 - alpha) * out[i - 1]
    return out


def preprocess_data(
    stock_data: pd.DataFrame,
    lag_steps: int = 8,
    ema_alpha: float = 0.5,
    test_size: float = 0.2,
):
    """Build lagged, smoothed features for one-step-ahead forecasting.

    Each sample uses only information available up to time t to forecast
    the close price at time t+1. No same-day close is used as a feature
    for its own target, and scalers are fitted on the training data only.

    Returns
    -------
    X_train_scaled, X_test_scaled : np.ndarray
        Feature matrices for train and test.
    y_train_scaled, y_test_scaled : np.ndarray
        Smoothed, scaled targets for train and test.
    x_scaler, y_scaler : MinMaxScaler
        Scalers fitted on X_train and y_train respectively.
    y_test_unsmoothed : np.ndarray
        Unscaled close prices for the test window (t+1).
    test_index : pd.DatetimeIndex
        Index corresponding to y_test_unsmoothed / predictions.
    """

    if stock_data.empty:
        raise ValueError("stock_data is empty in preprocess_data")

    # Drop rows with any NaNs (from indicators etc.)
    stock_data = stock_data.dropna().copy()

    feature_cols = [
        "Open",
        "Close",
        "Low",
        "Volume",
        "Price_Change",
        "ATR",
        "Bollinger_Bands",
        "MACD",
        "Stochastic_Oscillator",
    ]

    # Safety check
    missing = [c for c in feature_cols if c not in stock_data.columns]
    if missing:
        raise KeyError(f"Missing required feature columns: {missing}")

    # Raw features and target (close)
    X_raw = stock_data[feature_cols]
    y_raw = stock_data["Close"]

    # One-step-ahead target: close at t+1
    y_target = y_raw.shift(-1)

    # Build feature frame aligned with y_target
    df = X_raw.copy()
    df["Target"] = y_target

    # Create lagged features for each column
    for col in feature_cols:
        for i in range(1, lag_steps + 1):
            df[f"{col}_lag_{i}"] = df[col].shift(i)

    # Drop any row that has NaNs in features or target
    df = df.dropna()

    # Separate y (unsmoothed) and X
    y_unsmoothed = df["Target"].values.astype(float)
    X = df.drop(columns=["Target"])

    # Causal EMA smoothing on y_unsmoothed
    y_smoothed = _ema(y_unsmoothed, ema_alpha)

    # Train/test split index (no shuffle)
    n_samples = len(X)
    if n_samples < 2:
        raise ValueError("Not enough samples after preprocessing")

    split_idx = int((1.0 - test_size) * n_samples)
    if split_idx <= 0 or split_idx >= n_samples:
        raise ValueError(
            "Invalid train/test split; adjust test_size or provide more data"
        )

    X_train = X.iloc[:split_idx].values
    X_test = X.iloc[split_idx:].values

    y_train_sm = y_smoothed[:split_idx]
    y_test_sm = y_smoothed[split_idx:]
    y_train_unsm = y_unsmoothed[:split_idx]
    y_test_unsm = y_unsmoothed[split_idx:]

    # Scalers – fit on training data only
    x_scaler = MinMaxScaler()
    X_train_scaled = x_scaler.fit_transform(X_train)
    X_test_scaled = x_scaler.transform(X_test)

    y_scaler = MinMaxScaler()
    y_train_scaled = y_scaler.fit_transform(y_train_sm.reshape(-1, 1))
    y_test_scaled = y_scaler.transform(y_test_sm.reshape(-1, 1))

    # Index for the samples – matches df index; test_index is last part
    full_index = df.index
    test_index = full_index[split_idx:]

    return (
        X_train_scaled,
        X_test_scaled,
        y_train_scaled,
        y_test_scaled,
        x_scaler,
        y_scaler,
        y_test_unsm,
        test_index,
    )
