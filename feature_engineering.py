import pandas as pd

def engineer_features(stock_data):
    stock_data.dropna(inplace=True)

    # Feature Engineering
    stock_data['Price_Change'] = stock_data['Close'].diff()
    stock_data['ATR'] = stock_data[['High', 'Low', 'Close']].diff(axis=1).abs().max(axis=1)
    stock_data['Bollinger_Bands'] = (stock_data['Close'] - stock_data['Close'].rolling(window=20).mean()) / (2 * stock_data['Close'].rolling(window=20).std())
    stock_data['MACD'] = stock_data['Close'].ewm(span=12, adjust=False).mean() - stock_data['Close'].ewm(span=26, adjust=False).mean()
    stock_data['Stochastic_Oscillator'] = (stock_data['Close'] - stock_data['Low'].rolling(window=14).min()) / (stock_data['High'].rolling(window=14).max() - stock_data['Low'].rolling(window=14).min()) * 100
    
    return stock_data


