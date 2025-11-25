from tensorflow import keras
from keras.models import Sequential
from keras.layers import LSTM, Dense
from sklearn.metrics import mean_absolute_percentage_error, mean_squared_error
from keras.optimizers import Adam

def train_lstm_model(X_train, X_test, y_train, y_test):
    # Reshape for LSTM: (samples, timesteps, features)
    X_train = X_train.reshape((X_train.shape[0], 1, X_train.shape[1]))
    X_test = X_test.reshape((X_test.shape[0], 1, X_test.shape[1]))

    # Define the LSTM model
    model = Sequential()
    model.add(
        LSTM(
            units=64,
            activation="tanh",
            return_sequences=True,
            input_shape=(X_train.shape[1], X_train.shape[2]),
        )
    )
    model.add(LSTM(units=64, activation="tanh", return_sequences=True))
    model.add(LSTM(units=32, activation="tanh"))
    model.add(Dense(units=1))  # linear output for regression

    adam_optimizer = Adam(learning_rate=0.001)
    model.compile(optimizer=adam_optimizer, loss="mean_squared_error")

    model.fit(
        X_train,
        y_train,
        epochs=40,
        batch_size=8,
        validation_data=(X_test, y_test),
        verbose=0,
    )

    return model


def make_predictions(model, X_test, y_test, scaler):
    # Reshape input data for LSTM
    X_test = X_test.reshape((X_test.shape[0], 1, X_test.shape[1]))

    # Make predictions
    predictions = model.predict(X_test)

    # Inverse transform predictions using the same scaler (for y)
    predictions_unscaled = scaler.inverse_transform(predictions).reshape(-1)

    # Inverse transform y_test using the same scaler
    y_test_unscaled = scaler.inverse_transform(y_test.reshape(-1, 1)).reshape(-1)

    # Metrics
    mape = mean_absolute_percentage_error(y_test_unscaled, predictions_unscaled)
    mse = mean_squared_error(y_test_unscaled, predictions_unscaled)

    return predictions_unscaled, y_test_unscaled, mape, mse
