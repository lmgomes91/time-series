import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from tensorflow import keras
from tensorflow.python.keras.layers import LSTM, Dropout
from tensorflow.python.layers.core import Dense

from src.analysis.metrics.regression_metrics import regression_metrics
from src.neural_networks_models.base_model import BaseModel
from src.neural_networks_models.multivariate.preprocess_data_multi import preprocess_data_multivariate


class LstmMultivariate(BaseModel):
    @staticmethod
    def run(data: pd.DataFrame):
        scaler = MinMaxScaler()
        data = scaler.fit_transform(data)

        sequence_length = 10
        x_train, x_test, y_train, y_test = preprocess_data_multivariate(data, sequence_length)
        x_train = x_train.reshape(x_train.shape[0], x_train.shape[1], data.shape[1])
        x_test = x_test.reshape(x_test.shape[0], x_test.shape[1], data.shape[1])

        units = 200
        model = keras.Sequential(
            [
                LSTM(units=units, return_sequences=True, input_shape=(x_train.shape[1], x_train.shape[2])),
                Dropout(0.2),
                LSTM(units=units, return_sequences=True),
                Dropout(0.2),
                LSTM(units=units, return_sequences=False),
                Dropout(0.2),
                Dense(units=1)
            ]
        )

        model.compile(optimizer='adam', loss='mean_squared_error')

        # Train the model
        model.fit(
            x_train,
            y_train,
            epochs=100,
            batch_size=128,
            workers=8,
            use_multiprocessing=True,
            validation_data=[x_train, y_train]
        )

        # Make predictions
        y_pred = model.predict(x_test)
        # Inverse transform the predicted and actual values to the original scale
        y_pred = scaler.inverse_transform(np.concatenate([x_test[:, -1, :4], y_pred.reshape(-1, 1)], axis=1))[:, 4]
        y_test = scaler.inverse_transform(np.concatenate([x_test[:, -1, :4], y_test.reshape(-1, 1)], axis=1))[:, 4]

        # metrics
        regression_metrics(y_test, y_pred, 'lstm_multi')
        # Plot the actual vs. predicted values
        # predict_plot(y_test.tolist(), y_pred.tolist())
