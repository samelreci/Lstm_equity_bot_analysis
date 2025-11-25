import streamlit as st
import yfinance as yf
import plotly.graph_objs as go
from datetime import datetime, timedelta
import numpy as np
import pandas as pd

import feature_engineering
import data_preprocessing
import lstm_model
import signals


@st.cache_data(ttl=900)  # cache for 15 minutes
def get_stock_data(ticker, start, end, timeframe):
    """Download OHLCV data from yfinance."""
    try:
        return yf.download(ticker, start=start, end=end, interval=timeframe)
    except Exception as e:
        print("Download error:", e)
        return pd.DataFrame()


# default end date is yesterday (to avoid partial current day)
end_date = (datetime.now() - timedelta(days=1)).strftime("%Y-%m-%d")


def build_equity_and_irr(signals_generated, y_test_series: pd.Series):
    """Build equity curve and running annualised IRR over the test window.

    signals_generated: list of (entry_price, exit_price, percentage_return)
    y_test_series:     pd.Series of actual close prices for the test window,
                       indexed by dates.
    """

    # If no trades, just flat equity & zero IRR
    if not signals_generated or len(y_test_series) == 0:
        equity_series = pd.Series([1.0] * len(y_test_series), index=y_test_series.index)
        irr_series = pd.Series([0.0] * len(y_test_series), index=y_test_series.index)
        return equity_series, irr_series

    dates = y_test_series.index
    prices = y_test_series.values
    n = len(dates)

    # ---- Step 1: map each trade to an approximate exit index ----
    trade_exits = []  # list of (exit_idx, percentage_return)
    used_indices = set()

    for entry_price, exit_price, pct in signals_generated:
        # Find an entry index that hasn't been used yet
        entry_idx = None
        for i in range(n):
            if i in used_indices:
                continue
            if np.isclose(prices[i], entry_price, rtol=1e-4, atol=1e-2):
                entry_idx = i
                used_indices.add(i)
                break

        if entry_idx is None:
            # Couldn't map this trade into the test price series
            continue

        # Find an exit index AFTER the entry_idx that also hasn't been used
        exit_idx = None
        for j in range(entry_idx + 1, n):
            if j in used_indices:
                continue
            if np.isclose(prices[j], exit_price, rtol=1e-4, atol=1e-2):
                exit_idx = j
                used_indices.add(j)
                break

        if exit_idx is None:
            # No matching exit found in the test window
            continue

        trade_exits.append((exit_idx, pct))

    # If we couldn't map any trades, flat equity & zero IRR
    if not trade_exits:
        equity_series = pd.Series([1.0] * len(y_test_series), index=y_test_series.index)
        irr_series = pd.Series([0.0] * len(y_test_series), index=y_test_series.index)
        return equity_series, irr_series

    # ---- Step 2: build equity curve over time ----
    trade_exits_sorted = sorted(trade_exits, key=lambda x: x[0])

    equity = 1.0
    equity_curve = np.empty(n, dtype=float)
    k = 0  # pointer over trade_exits_sorted

    for t in range(n):
        # Carry previous equity by default
        equity_curve[t] = equity

        # Apply all trades that exit on this index
        while k < len(trade_exits_sorted) and trade_exits_sorted[k][0] == t:
            pct = trade_exits_sorted[k][1]
            equity *= (1.0 + pct / 100.0)
            equity_curve[t] = equity
            k += 1

    equity_series = pd.Series(equity_curve, index=dates).ffill()

    # ---- Step 3: compute running annualised IRR (CAGR) ----
    irr_values = []
    equity0 = equity_series.iloc[0]
    start_date = dates[0]

    for date, eq in equity_series.items():
        days = (date - start_date).days
        if days <= 0:
            irr_values.append(0.0)
        else:
            irr_values.append((eq / equity0) ** (365.0 / days) - 1.0)

    irr_series = pd.Series(irr_values, index=dates)

    return equity_series, irr_series


def plot_irr_curve(equity_series: pd.Series, irr_series: pd.Series):
    """Plot equity curve and running annualised IRR on the same chart."""
    fig = go.Figure()

    # Equity curve (left axis)
    fig.add_trace(
        go.Scatter(
            x=equity_series.index,
            y=equity_series.values,
            mode="lines",
            name="Equity (start = 1.0)",
        )
    )

    # IRR % (right axis)
    fig.add_trace(
        go.Scatter(
            x=irr_series.index,
            y=irr_series.values * 100.0,
            mode="lines",
            name="Annualised IRR (%)",
            yaxis="y2",
        )
    )

    fig.update_layout(
        title="Strategy Equity Curve and Annualised IRR",
        xaxis=dict(title="Date"),
        yaxis=dict(title="Equity"),
        yaxis2=dict(
            title="Annualised IRR (%)",
            overlaying="y",
            side="right",
        ),
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
        height=500,
    )

    return fig


def plot_signal_chart(y_test_unscaled: pd.Series, signals_list):
    """
    Signals chart: Actual test price line + entry/exit triangles (no candles).
    Uses ONLY the test window (y_test_unscaled index).
    """

    fig = go.Figure()

    # Actual close price line over the test window
    fig.add_trace(
        go.Scatter(
            x=y_test_unscaled.index,
            y=y_test_unscaled.values,
            mode="lines",
            name="Actual Price",
        )
    )

    if signals_list:
        prices = y_test_unscaled.values.astype(float)
        dates = y_test_unscaled.index

        entry_prices = [signal[0] for signal in signals_list]
        exit_prices = [signal[1] for signal in signals_list]

        entry_dates = []
        exit_dates = []

        # Match each signal price to the first date where that close occurred in the test window
        for p in entry_prices:
            mask = np.isclose(prices, float(p), rtol=1e-4, atol=1e-2)
            idxs = np.where(mask)[0]
            if idxs.size > 0:
                entry_dates.append(dates[idxs[0]])

        for p in exit_prices:
            mask = np.isclose(prices, float(p), rtol=1e-4, atol=1e-2)
            idxs = np.where(mask)[0]
            if idxs.size > 0:
                exit_dates.append(dates[idxs[0]])

        # Entry markers (green triangles up)
        if entry_dates and entry_prices:
            fig.add_trace(
                go.Scatter(
                    x=entry_dates,
                    y=entry_prices,
                    mode="markers",
                    marker=dict(color="green", symbol="triangle-up", size=10),
                    name="Entry Points",
                )
            )

        # Exit markers (red triangles down)
        if exit_dates and exit_prices:
            fig.add_trace(
                go.Scatter(
                    x=exit_dates,
                    y=exit_prices,
                    mode="markers",
                    marker=dict(color="red", symbol="triangle-down", size=10),
                    name="Exit Points",
                )
            )

    fig.update_layout(
        title="Signals Chart (Actual Price + Entries/Exits)",
        xaxis_title="Date",
        yaxis_title="Price",
        height=500,
        xaxis=dict(showgrid=True, showticklabels=True),
        yaxis=dict(showgrid=True),
    )

    return fig


def plot_predicted_vs_actual(predictions_unscaled: pd.Series, y_test_unscaled: pd.Series):
    fig = go.Figure()
    fig.add_trace(
        go.Scatter(
            x=y_test_unscaled.index,
            y=y_test_unscaled.values,
            mode="lines",
            name="Actual Price",
        )
    )
    fig.add_trace(
        go.Scatter(
            x=y_test_unscaled.index,
            y=predictions_unscaled.values,
            mode="lines",
            name="Predicted Price",
        )
    )
    fig.update_layout(
        title="Predicted vs Actual Price",
        xaxis_title="",
        yaxis_title="Price",
        height=500,
        xaxis=dict(showgrid=True, showticklabels=True),
        yaxis=dict(showgrid=True),
    )
    return fig


def main():
    st.set_page_config(layout="wide")
    st.title("LSTM Price Prediction and Trading Simulator")

    # Layout for inputs
    col1, col2, col3 = st.columns(3)

    with col1:
        ticker = st.text_input("Enter Ticker Symbol:", value="AAPL", max_chars=10)

    with col2:
        timeframe = st.selectbox("Select Timeframe:", ["1d", "5d", "1wk", "1h"])

    with col3:
        start_date = st.date_input("Start Date:")

    if st.button("Start"):
        if start_date is not None:
            stock_data = get_stock_data(ticker, start_date, end_date, timeframe)

            if stock_data.empty:
                st.error("No data found for the selected period.")
                return

            # Feature engineering
            stock_data = feature_engineering.engineer_features(stock_data)

            # Preprocess data (build lagged features, split, scale)
            (
                X_train,
                X_test,
                y_train,
                y_test,
                x_scaler,
                y_scaler,
                y_test_unsm,
                test_index,
            ) = data_preprocessing.preprocess_data(stock_data)

            # Train LSTM model
            model = lstm_model.train_lstm_model(X_train, X_test, y_train, y_test)

            # Make predictions (on scaled test set)
            predictions_unscaled, y_test_unscaled_scaled_space, mape, mse = lstm_model.make_predictions(
                model, X_test, y_test, y_scaler
            )

            # Align predictions and actual unsmoothed closes with dates from preprocess_data
            predictions_series = pd.Series(predictions_unscaled, index=test_index)
            y_test_series = pd.Series(y_test_unsm, index=test_index)

            # Generate trading signals using un-smoothed actual prices
            signals_generated, _ = signals.generate_signals(
                predictions_series.values, y_test_series
            )

            # Compute total percentage return from all trades
            def _to_scalar(x):
                if isinstance(x, (pd.Series, np.ndarray, list, tuple)):
                    arr = np.array(x).reshape(-1)
                    if arr.size == 0:
                        return 0.0
                    return float(arr[0])
                return float(x)

            if signals_generated:
                overall_percentage_return = sum(
                    _to_scalar(pct_ret) for _, _, pct_ret in signals_generated
                )
            else:
                overall_percentage_return = 0.0

            # Build equity curve & IRR
            equity_curve, irr_curve = build_equity_and_irr(
                signals_generated,
                y_test_series,   # Series with dates as index
            )
            irr_fig = plot_irr_curve(equity_curve, irr_curve)

            # Figures
            predicted_vs_actual_fig = plot_predicted_vs_actual(
                predictions_series, y_test_series
            )
            signal_chart_fig = plot_signal_chart(
                y_test_series, signals_generated
            )

            # Metrics and total return
            st.write(f"MAPE: {mape:.4f}")
            st.write(f"MSE: {mse:.2f}")
            st.write("")

            total_return_slot = st.empty()
            total_return_slot.text(
                f"Total Percentage Return: {overall_percentage_return:.2f}%"
            )

            final_irr_pct = irr_curve.iloc[-1] * 100.0
            st.write(f"Final annualised IRR over test window: {final_irr_pct:.2f}%")

            # Plots
            st.plotly_chart(
                predicted_vs_actual_fig,
                use_container_width=True,
                config={"scrollZoom": True},
            )
            st.plotly_chart(
                signal_chart_fig,
                use_container_width=True,
                config={"scrollZoom": True},
            )
            st.plotly_chart(
                irr_fig,
                use_container_width=True,
                config={"scrollZoom": True},
            )

            # Sidebar with trade information
            with st.sidebar:
                st.header("Trade Information")

                for entry_price, exit_price, percentage_return in signals_generated:
                    entry_price_val = float(np.array(entry_price).reshape(-1)[0])
                    exit_price_val = float(np.array(exit_price).reshape(-1)[0])
                    percentage_return_val = float(
                        np.array(percentage_return).reshape(-1)[0]
                    )

                    entry_price_str = f"{entry_price_val:.2f}"
                    exit_price_str = f"{exit_price_val:.2f}"

                    color = "green" if percentage_return_val >= 0 else "red"
                    st.write(
                        f"<b>Entry Price:</b> {entry_price_str}, "
                        f"<b>Exit Price:</b> {exit_price_str}, "
                        f"<span style='color:{color}'><b>Percentage return:</b> {percentage_return_val:.2f}%</span>",
                        unsafe_allow_html=True,
                    )


if __name__ == "__main__":
    main()
