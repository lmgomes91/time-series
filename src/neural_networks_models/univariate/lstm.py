import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from tensorflow import keras
from tensorflow.python.keras.layers import LSTM, Dropout
from tensorflow.python.layers.core import Dense

from src.analysis.graphs.predict_plot import predict_plot
from src.analysis.metrics.regression_metrics import regression_metrics
from src.neural_networks_models.base_model import BaseModel
from src.neural_networks_models.univariate.preprocess_data import preprocess_data


class Lstm(BaseModel):
    @staticmethod
    def run(data: pd.DataFrame):
        scaler = MinMaxScaler()
        data.iloc[:, 1:2] = scaler.fit_transform(data.iloc[:, 1:2])

        sequence_length = 10
        x_train, x_test, y_train, y_test = preprocess_data(data, sequence_length)
        x_train = x_train.reshape(-1, sequence_length, 1)
        x_test = x_test.reshape(-1, sequence_length, 1)

        units = 150
        model = keras.Sequential(
            [
                LSTM(units=units, return_sequences=True, input_shape=(sequence_length, 1)),
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
            batch_size=124,
            workers=16,
            use_multiprocessing=True,
            validation_data=[x_train, y_train]
        )

        # Make predictions
        y_pred = model.predict(x_test)  # noqa
        # metrics
        y_test = scaler.inverse_transform(y_test)
        y_pred = scaler.inverse_transform(y_pred)
        regression_metrics(y_test, y_pred, 'lstm_uni')
        # Plot the actual vs. predicted values
        # predict_plot(y_test, y_pred)
