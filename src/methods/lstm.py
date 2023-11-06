import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from tensorflow import keras
from tensorflow.python.keras.layers import LSTM, Dropout
from tensorflow.python.layers.core import Dense
import matplotlib.pyplot as plt


def lstm(data: pd.DataFrame):
    # Generate synthetic time series data
    # # Split the data into input and output sequences
    values = data.iloc[:, 1:2].values
    sc = MinMaxScaler()
    values = sc.fit_transform(values)

    sequence_length = 10  # Length of input sequences
    x = []
    y = []

    for i in range(len(data) - sequence_length):
        x.append(values[i:i+sequence_length])
        y.append(values[i+sequence_length])

    x = np.array(x)
    y = np.array(y)

    # Split the data into training and testing sets
    split_ratio = 0.8
    split_index = int(split_ratio * len(x))
    x_train, x_test = x[:split_index], x[split_index:]
    y_train, y_test = y[:split_index], y[split_index:]

    x_train = x_train.reshape(-1, sequence_length, 1)
    x_test = x_test.reshape(-1, sequence_length, 1)

    units = 150
    model = keras.Sequential()
    model.add(LSTM(units=units, return_sequences=True, input_shape=(sequence_length, 1)))
    model.add(Dropout(0.2))
    model.add(LSTM(units=units, return_sequences=True))
    model.add(Dropout(0.2))
    model.add(LSTM(units=units, return_sequences=False))
    model.add(Dropout(0.2))
    model.add(Dense(units=1))
    model.compile(optimizer='adam', loss='mean_squared_error')

    # Reshape input data for LSTM
    x_train = x_train.reshape(-1, sequence_length, 1)
    x_test = x_test.reshape(-1, sequence_length, 1)

    # Train the model
    model.fit(
        x_train,
        y_train,
        epochs=5,
        batch_size=32,
        workers=8,
        use_multiprocessing=True,
        validation_data=[x_train, y_train]
    )

    # Make predictions
    y_pred = model.predict(x_test) # noqa
    # Plot the actual vs. predicted values
    plt.plot(sc.inverse_transform(y_test), label='Actual')
    plt.plot(sc.inverse_transform(y_pred), label='Predicted')
    plt.legend()
    plt.show()
