from tensorflow import keras
from keras.models import Sequential
from keras.layers import LSTM, Dense
from sklearn.metrics import mean_absolute_percentage_error, mean_squared_error
from keras.optimizers import Adam

def train_lstm_model(X_train, X_test, y_train, y_test):
    """Train a simple stacked LSTM on the scaled features.

    Parameters
    ----------
    X_train, X_test : np.ndarray
        Feature matrices already scaled, shape (n_samples, n_features).
    y_train, y_test : np.ndarray
        Scaled targets, shape (n_samples, 1).
    """

    # Reshape for LSTM: (samples, timesteps, features)
    X_train = X_train.reshape((X_train.shape[0], 1, X_train.shape[1]))
    X_test = X_test.reshape((X_test.shape[0], 1, X_test.shape[1]))

    model = Sequential()
    model.add(
        LSTM(
            units=64,
            activation="relu",
            return_sequences=True,
            input_shape=(X_train.shape[1], X_train.shape[2]),
        )
    )
    model.add(LSTM(units=64, activation="relu", return_sequences=True))
    model.add(LSTM(units=64, activation="relu", return_sequences=True))
    model.add(LSTM(units=32, activation="relu"))
    model.add(Dense(units=1))  # linear output for regression

    adam_optimizer = Adam(learning_rate=0.001)
    model.compile(optimizer=adam_optimizer, loss="mean_squared_error")

    model.fit(
        X_train,
        y_train,
        epochs=20,
        batch_size=16,
        validation_data=(X_test, y_test),
        verbose=0,
    )

    return model


def make_predictions(model, X_test, y_test, y_scaler):
    """Run the model on the test set and return unscaled predictions & metrics."""

    # Reshape for LSTM
    X_test = X_test.reshape((X_test.shape[0], 1, X_test.shape[1]))

    # Predict
    predictions = model.predict(X_test, verbose=0)

    # Inverse transform predictions and y_test using the target scaler
    predictions_unscaled = y_scaler.inverse_transform(predictions).reshape(-1)
    y_test_unscaled = y_scaler.inverse_transform(y_test.reshape(-1, 1)).reshape(-1)

    # Metrics
    mape = mean_absolute_percentage_error(y_test_unscaled, predictions_unscaled)
    mse = mean_squared_error(y_test_unscaled, predictions_unscaled)

    return predictions_unscaled, y_test_unscaled, mape, mse
