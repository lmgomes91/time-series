import numpy as np
import pandas as pd
from keras.src.layers import Conv1D, MaxPooling1D, Flatten, Dense
from sklearn.preprocessing import MinMaxScaler
from tensorflow import keras

from src.analysis.metrics.regression_metrics import regression_metrics
from src.neural_networks_models.base_model import BaseModel
from src.neural_networks_models.multivariate.preprocess_data_multi import preprocess_data_multivariate


class CnnMultivariate(BaseModel):
    @staticmethod
    def run(data: pd.DataFrame):
        scaler = MinMaxScaler()
        data = scaler.fit_transform(data)

        sequence_length = 10
        x_train, x_test, y_train, y_test = preprocess_data_multivariate(data, sequence_length)
        x_train = x_train.reshape(x_train.shape[0], x_train.shape[1], data.shape[1])
        x_test = x_test.reshape(x_test.shape[0], x_test.shape[1], data.shape[1])

        model = keras.Sequential(
            [
                Conv1D(filters=200, kernel_size=3, activation='relu', input_shape=(x_train.shape[1], x_train.shape[2])),
                MaxPooling1D(pool_size=2),
                Conv1D(filters=200, kernel_size=3, activation='relu'),
                MaxPooling1D(pool_size=2),
                Flatten(),
                Dense(50, activation='relu'),
                Dense(1)
            ]
        )
        # Compile the model
        model.compile(optimizer='adam', loss='mean_squared_error')  # Adjust optimizer and loss function as needed

        # Train the model
        model.fit(
            x_train,
            y_train,
            epochs=70,
            batch_size=10,
            workers=16,
            use_multiprocessing=True,
            validation_data=[x_train, y_train]
        )

        # Make predictions
        y_pred = model.predict(x_test)  # noqa

        y_pred = scaler.inverse_transform(np.concatenate([x_test[:, -1, :4], y_pred.reshape(-1, 1)], axis=1))[:, 4]
        y_test = scaler.inverse_transform(np.concatenate([x_test[:, -1, :4], y_test.reshape(-1, 1)], axis=1))[:, 4]
        # metrics
        regression_metrics(y_test, y_pred)
        # Plot the actual vs. predicted values
        # predict_plot(scaler.inverse_transform(y_test), scaler.inverse_transform(y_pred))
