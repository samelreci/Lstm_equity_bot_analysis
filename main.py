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

@st.cache_data(ttl=900)   # cache for 15 minutes
def get_stock_data(ticker, start, end, timeframe):
    try:
        return yf.download(ticker, start=start, end=end, interval=timeframe)
    except Exception as e:
        print("Download error:", e)
        return pd.DataFrame()

end_date = datetime.now() - timedelta(days=1)
end_date = end_date.strftime('%Y-%m-%d')

def plot_candlestick_chart(stock_data, signals_list):
    fig = go.Figure(data=[go.Candlestick(
        x=stock_data.index,
        open=stock_data['Open'],
        high=stock_data['High'],
        low=stock_data['Low'],
        close=stock_data['Close']
    )])

    # If there are no signals just return the candlestick
    if not signals_list:
        fig.update_layout(
            title='Candlestick Chart with Entry/Exit Signals',
            xaxis_title='Date',
            yaxis_title='Price',
            height=800
        )
        return fig

    # Extract entry and exit prices from signals
    entry_prices = [signal[0] for signal in signals_list]
    exit_prices = [signal[1] for signal in signals_list]

    entry_dates = []
    exit_dates = []

    # Use np.isclose to avoid float equality issues
    closes = stock_data['Close']

    for entry_price in entry_prices:
        mask = np.isclose(closes, entry_price, rtol=1e-4, atol=1e-2)
        dates = stock_data.loc[mask].index.tolist()
        if dates:
            entry_dates.append(dates[0])

    for exit_price in exit_prices:
        mask = np.isclose(closes, exit_price, rtol=1e-4, atol=1e-2)
        dates = stock_data.loc[mask].index.tolist()
        if dates:
            exit_dates.append(dates[0])

    # Add green triangles for entry points
    if entry_dates and entry_prices:
        fig.add_trace(go.Scatter(
            x=entry_dates,
            y=entry_prices,
            mode='markers',
            marker=dict(color='green', symbol='triangle-up', size=10),
            name='Entry Points'
        ))

    # Add red triangles for exit points
    if exit_dates and exit_prices:
        fig.add_trace(go.Scatter(
            x=exit_dates,
            y=exit_prices,
            mode='markers',
            marker=dict(color='red', symbol='triangle-down', size=10),
            name='Exit Points'
        ))

    fig.update_layout(
        title='Candlestick Chart with Entry/Exit Signals',
        xaxis_title='Date',
        yaxis_title='Price',
        height=800
    )

    return fig

def plot_predicted_vs_actual(predictions_unscaled, y_test_unscaled):
    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=y_test_unscaled.index,
        y=y_test_unscaled,
        mode='lines',
        name='Actual Price'
    ))
    fig.add_trace(go.Scatter(
        x=y_test_unscaled.index,
        y=predictions_unscaled,
        mode='lines',
        name='Predicted Price'
    ))
    fig.update_layout(
        title='Predicted vs Actual Price',
        xaxis_title='',
        yaxis_title='Price',
        xaxis=dict(showticklabels=False)
    )
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
        start_date = st.date_input('Start Date:')

    # Fetch data and plot candlestick chart
    if st.button('Start'):
        if start_date is not None:
            stock_data = get_stock_data(ticker, start_date, end_date, timeframe)

            # STOP THE APP IF YF RATE-LIMITED YOU OR RETURNED NOTHING
            if stock_data is None or stock_data.empty:
                st.error("Yahoo Finance is rate limiting you or returned no data. Try again in 30–60 seconds.")
                st.stop()

            # Feature engineering
            stock_data = feature_engineering.engineer_features(stock_data)
            st.write("🔹 BEFORE preprocess_data")
            st.write("Shape:", stock_data.shape)
            st.write(stock_data.head())
            st.write(stock_data.tail())
            # Data preprocessing
            X_train, X_test, y_train, y_test, scaler, y_test_unsmooth = data_preprocessing.preprocess_data(stock_data)
            st.write("🔹 AFTER preprocess_data")
            st.write("X_train shape:", X_train.shape)
            st.write("X_test shape:", X_test.shape)
            st.write("len(y_test_unsmooth):", len(y_test_unsmooth))
            # Train LSTM model
            model = lstm_model.train_lstm_model(X_train, X_test, y_train, y_test)

            # Make predictions
            predictions_unscaled, y_test_unscaled, mape, mse = lstm_model.make_predictions(
                model, X_test, y_test, scaler
            )

            # Generate signals using unsmoothed test prices
            signals_generated, y_signal = signals.generate_signals(
                predictions_unscaled, y_test_unsmooth
            )

            # Align predictions and actual (unscaled) to real market dates
            test_index = stock_data.index[-len(y_test_unscaled):]

            predictions_unscaled = pd.Series(predictions_unscaled, index=test_index)
            y_test_unscaled = pd.Series(y_test_unscaled, index=test_index)

            # Candlestick chart
            candlestick_fig = plot_candlestick_chart(stock_data, signals_generated)

            overall_percentage_return = sum(
                percentage_return for _, _, percentage_return in signals_generated
            )

             # Sum all percentage returns as plain floats
            def _to_scalar(x):
                
                if isinstance(x, (pd.Series, np.ndarray, list, tuple)):
                    arr = np.array(x).reshape(-1)
                    if arr.size == 0:
                        return 0.0
                    return float(arr[0])
                return float(x)

            if signals_generated:
                overall_percentage_return = sum(
                    _to_scalar(percentage_return)
                    for _, _, percentage_return in signals_generated
                )
            else:
                overall_percentage_return = 0.0

            st.write('')
            total_return_slot = st.empty()
            total_return_slot.text(
                f'Total Percentage Return: {overall_percentage_return:.2f}%'
            )


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

                    entry_price_val = entry_price.item() if hasattr(entry_price, 'item') else entry_price
                    exit_price_val = exit_price.item() if hasattr(exit_price, 'item') else exit_price
                    percentage_return_val = percentage_return.item() if hasattr(percentage_return, 'item') else percentage_return

                    entry_price_str = "{:.2f}".format(entry_price_val)
                    exit_price_str = "{:.2f}".format(exit_price_val)

                    color = "green" if percentage_return_val >= 0 else "red"
                    st.write(
                        f"<b>Entry Price:</b> {entry_price_str}, "
                        f"<b>Exit Price:</b> {exit_price_str}, "
                        f"<span style='color:{color}'><b>Percentage return:</b> {percentage_return_val:.2f}%</span>",
                        unsafe_allow_html=True
                    )

# Run the Streamlit app
if __name__ == '__main__':
    main()
