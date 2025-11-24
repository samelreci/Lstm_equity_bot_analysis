def generate_signals(predictions_unscaled, y_test_unscaled):
    signals = []
    entry_price = None
    exit_price = None
    
    y_signal = y_test_unscaled

    for i, prediction in enumerate(predictions_unscaled):
        if i == 0:
            continue
        
        prev_prediction = predictions_unscaled[i - 1]
        
        if prediction > prev_prediction:
            if entry_price is None:
                entry_price = y_signal[i]
            elif exit_price is not None:
                signals.append((entry_price, exit_price, (exit_price - entry_price) / entry_price * 100))
                entry_price = y_signal[i]
                exit_price = None
        elif prediction < prev_prediction:
            if exit_price is None and entry_price is not None:
                exit_price = y_signal[i]
        else:
            if exit_price is not None and entry_price is not None:
                signals.append((entry_price, exit_price, (exit_price - entry_price) / entry_price * 100))
                entry_price = None
                exit_price = None
    
    return signals, y_signal