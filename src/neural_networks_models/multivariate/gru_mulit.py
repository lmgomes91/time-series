import numpy as np
import pandas as pd
from keras.src.layers import Dense, GRU
from sklearn.preprocessing import MinMaxScaler
from tensorflow import keras

from src.analysis.metrics.regression_metrics import regression_metrics
from src.neural_networks_models.base_model import BaseModel
from src.neural_networks_models.multivariate.preprocess_data_multi import preprocess_data_multivariate


class GruMultivariate(BaseModel):
    @staticmethod
    def run(data: pd.DataFrame):
        scaler = MinMaxScaler()
        data = scaler.fit_transform(data)

        sequence_length = 10
        x_train, x_test, y_train, y_test = preprocess_data_multivariate(data, sequence_length)
        x_train = x_train.reshape(x_train.shape[0], x_train.shape[1], data.shape[1])
        x_test = x_test.reshape(x_test.shape[0], x_test.shape[1], data.shape[1])
        # Build the GRU model
        model = keras.Sequential(
            [
                GRU(units=200, input_shape=(x_train.shape[1], x_train.shape[2]), activation='tanh',
                    recurrent_activation='sigmoid'),
                # GRU(units=200, activation='tanh', recurrent_activation='sigmoid'),
                Dense(units=1)
            ]
        )

        # Compile the model
        model.compile(optimizer='adam', loss='mean_squared_error')  # Adjust optimizer and loss function as needed

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
        y_pred = model.predict(x_test)  # noqa

        y_pred = scaler.inverse_transform(np.concatenate([x_test[:, -1, :4], y_pred.reshape(-1, 1)], axis=1))[:, 4]
        y_test = scaler.inverse_transform(np.concatenate([x_test[:, -1, :4], y_test.reshape(-1, 1)], axis=1))[:, 4]

        # metrics
        regression_metrics(y_test, y_pred, 'gru_multi')
        # Plot the actual vs. predicted values
        # predict_plot(scaler.inverse_transform(y_test), scaler.inverse_transform(y_pred))
