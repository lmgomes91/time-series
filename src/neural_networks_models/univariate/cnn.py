import pandas as pd
from keras.src.layers import Conv1D, MaxPooling1D, Flatten, Dense
from sklearn.preprocessing import MinMaxScaler
from tensorflow import keras

from src.analysis.graphs.predict_plot import predict_plot
from src.analysis.metrics.regression_metrics import regression_metrics
from src.neural_networks_models.base_model import BaseModel
from src.neural_networks_models.univariate.preprocess_data import preprocess_data


class Cnn(BaseModel):
    @staticmethod
    def run(data: pd.DataFrame):
        scaler = MinMaxScaler()
        data.iloc[:, 1:2] = scaler.fit_transform(data.iloc[:, 1:2])

        sequence_length = 10
        x_train, x_test, y_train, y_test = preprocess_data(data, sequence_length)
        x_train = x_train.reshape(-1, sequence_length, 1)
        x_test = x_test.reshape(-1, sequence_length, 1)
        # Build the CNN model
        model = keras.Sequential(
            [
                Conv1D(filters=200, kernel_size=3, activation='relu', input_shape=(x_train.shape[1], 1)),
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
            epochs=200,
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
        regression_metrics(y_test, y_pred, 'cnn_uni')
        # Plot the actual vs. predicted values
        # predict_plot(y_test, y_pred)
