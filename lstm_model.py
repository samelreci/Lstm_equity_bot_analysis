from tensorflow import keras
from keras.models import Sequential
from keras.layers import LSTM, Dense
from sklearn.metrics import mean_absolute_percentage_error, mean_squared_error
from keras.optimizers import Adam

def train_lstm_model(X_train, X_test, y_train, y_test):
    
    X_train = X_train.reshape((X_train.shape[0], 1, X_train.shape[1]))
    X_test = X_test.reshape((X_test.shape[0], 1, X_test.shape[1]))
    # Define the LSTM model
    model = Sequential()
    model.add(LSTM(units=64,activation='relu', return_sequences=True, input_shape=(X_train.shape[1], X_train.shape[2])))
    model.add(LSTM(units=64, activation='relu', return_sequences=True))  
    model.add(LSTM(units=32, activation='relu'))
    model.add(Dense(units=1))
    adam_optimizer = Adam(learning_rate=0.001)
    model.compile(optimizer=adam_optimizer, loss='mean_squared_error')

    # Train the LSTM model
    history = model.fit(X_train, y_train, epochs=40, batch_size=8, validation_data=(X_test, y_test))

    return model

def make_predictions(model, X_test,y_test, scaler):
    
    # Reshape input data
    X_test = X_test.reshape((X_test.shape[0], 1, X_test.shape[1]))

    # Make predictions
    predictions = model.predict(X_test)

    # Inverse transform predictions using the same scaler
    predictions_unscaled = scaler.inverse_transform(predictions)

    # Inverse transform y_test using the same scaler
    y_test_unscaled = scaler.inverse_transform(y_test.reshape(-1, 1))

    mape = mean_absolute_percentage_error(y_test_unscaled, predictions_unscaled)
    mse = mean_squared_error(y_test_unscaled, predictions_unscaled)

    return predictions_unscaled, y_test_unscaled, mape , mse
