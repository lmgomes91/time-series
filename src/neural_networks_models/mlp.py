import pandas as pd
from sklearn.neural_network import MLPRegressor
from sklearn.preprocessing import MinMaxScaler

from src.analysis.graphs.predict_plot import predict_plot
from src.analysis.metrics.regression_metrics import regression_metrics
from src.neural_networks_models.base_model import BaseModel
from src.utils.preprocess_data import preprocess_data


class Mlp(BaseModel):

    @staticmethod
    def run(data: pd.DataFrame):
        scaler = MinMaxScaler()
        data.iloc[:, 1:2] = scaler.fit_transform(data.iloc[:, 1:2])
        sequence_length = 10
        x_train, x_test, y_train, y_test = preprocess_data(data, sequence_length)

        x_train = x_train.reshape((x_train.shape[0], -1))

        x_test = x_test.reshape((x_test.shape[0], -1))

        model = MLPRegressor(
            hidden_layer_sizes=(50, 50, 50),
            activation='relu',
            solver='adam',
            random_state=42,
            learning_rate_init=0.002,
            alpha=0.00001,
            max_iter=500,
            verbose=True
        )

        model.fit(x_train, y_train)

        # Make predictions
        y_pred = model.predict(x_test)  # noqa
        # metrics
        regression_metrics(y_test, y_pred)
        # Plot the actual vs. predicted values
        y_test = y_test.reshape(-1, 1)
        y_pred = y_pred.reshape(-1, 1)
        predict_plot(scaler.inverse_transform(y_test), scaler.inverse_transform(y_pred))
