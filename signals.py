def generate_signals(predictions_unscaled, y_test_unscaled):
    signals = []
    entry_price = None
    exit_price = None

    y_signal = y_test_unscaled 

    for i, prediction in enumerate(predictions_unscaled):
        if i == 0:
            continue

        prev_prediction = predictions_unscaled[i - 1]

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
