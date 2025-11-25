import pandas as pd

def engineer_features(stock_data):
    stock_data.dropna(inplace=True)

    # Feature Engineering
    stock_data['Price_Change'] = stock_data['Close'].diff()
    stock_data['ATR'] = stock_data[['High', 'Low', 'Close']].diff(axis=1).abs().max(axis=1)
    stock_data['MACD'] = stock_data['Close'].ewm(span=12, adjust=False).mean() - stock_data['Close'].ewm(span=26, adjust=False).mean()
    
    return stock_data
