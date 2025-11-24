import streamlit as st
import yfinance as yf
import plotly.graph_objs as go
from datetime import datetime, timedelta
import numpy as np
import feature_engineering
import data_preprocessing
import lstm_model
import signals
import pandas as pd

end_date = datetime.now() - timedelta(days=1)
end_date = end_date.strftime('%Y-%m-%d')

def plot_candlestick_chart(stock_data, signals):
    fig = go.Figure(data=[go.Candlestick(x=stock_data.index,
                                          open=stock_data['Open'],
                                          high=stock_data['High'],
                                          low=stock_data['Low'],
                                          close=stock_data['Close'])])

    # Extract entry and exit prices from signals
    entry_prices = [signal[0] for signal in signals]
    exit_prices = [signal[1] for signal in signals]

    # Find corresponding x-axis positions for entry and exit prices
    entry_dates = []
    exit_dates = []

    for entry_price in entry_prices:
        entry_date = stock_data.index[stock_data['Close'] == entry_price].tolist()
        if entry_date:
            entry_dates.append(entry_date[0])

    for exit_price in exit_prices:
        exit_date = stock_data.index[stock_data['Close'] == exit_price].tolist()
        if exit_date:
            exit_dates.append(exit_date[0])

    # Add green triangles for entry points
    fig.add_trace(go.Scatter(x=entry_dates, y=entry_prices,
                             mode='markers',
                             marker=dict(color='green', symbol='triangle-up', size=10),
                             name='Entry Points'))

    # Add red triangles for exit points
    fig.add_trace(go.Scatter(x=exit_dates, y=exit_prices,
                             mode='markers',
                             marker=dict(color='red', symbol='triangle-down', size=10),
                             name='Exit Points'))

    # Set the height of the figure
    fig.update_layout(title='Candlestick Chart with Entry and Exit Points',
                      xaxis_title='Date',
                      yaxis_title='Price',
                      height=800)  # Adjust the height as per your requirement

    return fig


# Function to plot line graph
def plot_predicted_vs_actual(predictions_unscaled, y_test_unscaled):
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=y_test_unscaled.index, y=y_test_unscaled, mode='lines', name='Actual Price'))
    fig.add_trace(go.Scatter(x=y_test_unscaled.index, y=predictions_unscaled, mode='lines', name='Predicted Price'))
    fig.update_layout(title='Predicted vs Actual Price', xaxis_title='', yaxis_title='Price', xaxis=dict(showticklabels=False))
    return fig

# Streamlit app
def main():
    st.title('Machine Learning Driven Analysis')

    # Ticker, Timeframe, Start Date input fields
    col1, col2, col3 = st.columns(3)

    with col1:
        ticker = st.text_input('Enter Ticker Symbol:', value='AAPL', max_chars=10)

    with col2:
        timeframe = st.selectbox('Select Timeframe:', ['1d', '5d', '1wk', '1h'])

    with col3:
        start_date = st.date_input('Start Date:', value=None)

    # Fetch data and plot candlestick chart
    if st.button('Start'):
        if start_date is not None:
            stock_data = yf.download(ticker, start=start_date, end=end_date, interval=timeframe)

            # Feature engineering
            stock_data = feature_engineering.engineer_features(stock_data)

            # Data preprocessing
            X_train, X_test, y_train, y_test, scaler, y_test_unsmooth = data_preprocessing.preprocess_data(stock_data)

            # Train LSTM model
            model = lstm_model.train_lstm_model(X_train, X_test, y_train, y_test)

            # Make predictions
            predictions_unscaled, y_test_unscaled, mape, mse = lstm_model.make_predictions(model, X_test, y_test, scaler)

            signals_generated, y_signal = signals.generate_signals(predictions_unscaled, y_test_unsmooth)

            candlestick_fig = plot_candlestick_chart(stock_data, signals_generated)

            # Convert predictions_unscaled to pandas Series with DateTime index
            predictions_unscaled = pd.Series(predictions_unscaled.flatten(), index=pd.date_range(start=start_date, periods=len(predictions_unscaled), freq=timeframe))

            # Convert y_test_unscaled to pandas Series with DateTime index
            y_test_unscaled = pd.Series(y_test_unscaled.flatten(), index=pd.date_range(start=start_date, periods=len(y_test_unscaled), freq=timeframe))

            overall_percentage_return = sum(percentage_return for _, _, percentage_return in signals_generated)

            # If overall_percentage_return is a numpy array, convert it to a scalar
            if isinstance(overall_percentage_return, np.ndarray):
                overall_percentage_return = overall_percentage_return.item()

            st.write('')
            total_return_slot = st.empty()  
            # Display overall percentage return at the specified location
            total_return_slot.text(f'Total Percentage Return: {overall_percentage_return:.2f}%')
            
            st.write('')
            total_return_slot = st.empty() 
            total_return_slot.text('MAPE: {:.4f}'.format(mape))
            st.write('')
            total_return_slot = st.empty() 
            total_return_slot.text('MSE: {:.2f}'.format(mse))

            # Plot predicted vs actual price
            predicted_vs_actual_fig = plot_predicted_vs_actual(predictions_unscaled, y_test_unscaled)

            # Show the graphs
            st.plotly_chart(predicted_vs_actual_fig, use_container_width=True, config={'scrollZoom': True})
            st.plotly_chart(candlestick_fig, use_container_width=True, config={'scrollZoom': True})

            # Layout for right half of the screen (trade information box)
            with st.sidebar:
                st.header('Trade Information')
                for signal in signals_generated:
                    entry_price, exit_price, percentage_return = signal
                    entry_price = "{:.2f}".format(entry_price.item() if hasattr(entry_price, 'item') else entry_price)
                    exit_price = "{:.2f}".format(exit_price.item() if hasattr(exit_price, 'item') else exit_price)
                    percentage_return = percentage_return.item() if hasattr(percentage_return, 'item') else percentage_return

                    # Apply color based on positive or negative percentage return
                    color = "green" if percentage_return >= 0 else "red"
                    st.write(f"<b>Entry Price:</b> {entry_price}, <b>Exit Price:</b> {exit_price}, <span style='color:{color}'><b>Percentage return:</b> {percentage_return:.2f}%</span>", unsafe_allow_html=True)

# Run the Streamlit app
if __name__ == '__main__':
    main()
