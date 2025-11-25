import pandas as pd


def engineer_features(stock_data: pd.DataFrame) -> pd.DataFrame:
    """Add technical indicators used as model features.

    This function does *not* drop rows aggressively; it leaves NaNs from
    rolling calculations for the preprocessing step to handle.
    """

    stock_data = stock_data.copy()

    # Simple price change (close-to-close)
    stock_data['Price_Change'] = stock_data['Close'].diff()

    # True range approximation and ATR (14-period)
    tr = stock_data['High'] - stock_data['Low']
    stock_data['ATR'] = tr.rolling(window=14, min_periods=1).mean()

    # Bollinger %B using 20-period SMA and std
    ma20 = stock_data['Close'].rolling(window=20, min_periods=1).mean()
    std20 = stock_data['Close'].rolling(window=20, min_periods=1).std()
    # avoid division by zero
    stock_data['Bollinger_Bands'] = (stock_data['Close'] - ma20) / (
        2 * std20.replace(0, pd.NA)
    )

    # MACD (12-26 EMA difference)
    ema12 = stock_data['Close'].ewm(span=12, adjust=False).mean()
    ema26 = stock_data['Close'].ewm(span=26, adjust=False).mean()
    stock_data['MACD'] = ema12 - ema26

    # Stochastic oscillator %K (14-period)
    low14 = stock_data['Low'].rolling(window=14, min_periods=1).min()
    high14 = stock_data['High'].rolling(window=14, min_periods=1).max()
    denom = (high14 - low14).replace(0, pd.NA)
    stock_data['Stochastic_Oscillator'] = 100 * (stock_data['Close'] - low14) / denom

    return stock_data
