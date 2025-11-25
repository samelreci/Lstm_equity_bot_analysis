import pandas as pd
import numpy as np


def generate_signals(predictions_unscaled, y_test_unscaled: pd.Series):
    """Generate simple long-only entry/exit signals from predictions.

    Parameters
    ----------
    predictions_unscaled : array-like
        1D array of predicted prices for the test window.
    y_test_unscaled : pd.Series
        Actual (unscaled) prices for the same window.

    Returns
    -------
    signals : list of tuples
        Each element is (entry_price, exit_price, percentage_return).
    y_signal : pd.Series
        Alias to y_test_unscaled (for compatibility).
    """

    predictions = np.asarray(predictions_unscaled).reshape(-1)
    if not isinstance(y_test_unscaled, pd.Series):
        # convert to Series without index if plain array is passed
        y_signal = pd.Series(y_test_unscaled)
    else:
        y_signal = y_test_unscaled

    signals = []
    entry_price = None
    exit_price = None

    for i in range(1, len(predictions)):
        prediction = predictions[i]
        prev_prediction = predictions[i - 1]

        # Prediction going up → Enter / re-enter long
        if prediction > prev_prediction:
            if entry_price is None:
                entry_price = y_signal.iloc[i]

            elif exit_price is not None:
                # Close previous trade
                percentage_return = (exit_price - entry_price) / entry_price * 100
                signals.append((entry_price, exit_price, percentage_return))

                # Re-enter
                entry_price = y_signal.iloc[i]
                exit_price = None

        # Prediction going down → Exit if in a trade
        elif prediction < prev_prediction:
            if exit_price is None and entry_price is not None:
                exit_price = y_signal.iloc[i]

        # Prediction flat → Close if we have a completed trade
        else:
            if exit_price is not None and entry_price is not None:
                percentage_return = (exit_price - entry_price) / entry_price * 100
                signals.append((entry_price, exit_price, percentage_return))
                entry_price = None
                exit_price = None

    # Close last open trade if both prices exist
    if entry_price is not None and exit_price is not None:
        percentage_return = (exit_price - entry_price) / entry_price * 100
        signals.append((entry_price, exit_price, percentage_return))

    return signals, y_signal
