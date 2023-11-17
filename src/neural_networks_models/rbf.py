from sklearn.preprocessing import MinMaxScaler

from src.analysis.metrics.regression_metrics import regression_metrics
from src.neural_networks_models.base_model import BaseModel
import pandas as pd
from sklearn.svm import SVR

from src.utils.preprocess_data import preprocess_data


class Rbf(BaseModel):

    @staticmethod
    def run(data: pd.DataFrame):
        scaler = MinMaxScaler()
        data.iloc[:, 1:2] = scaler.fit_transform(data.iloc[:, 1:2])
        sequence_length = 10
        x_train, x_test, y_train, y_test = preprocess_data(data, sequence_length)

        x_train = x_train.reshape((x_train.shape[0], -1))

        x_test = x_test.reshape((x_test.shape[0], -1))

        # Initialize SVR with RBF kernel
        model = SVR(kernel='rbf', verbose=True, epsilon=0.002, tol=1e-5)

        # Fit the model
        model.fit(x_train, y_train)

        # Make predictions
        y_pred = model.predict(x_test)  # noqa
        # metrics
        regression_metrics(y_test, y_pred)
        # Plot the actual vs. predicted values
        # y_test = y_test.reshape(-1, 1)
        # y_pred = y_pred.reshape(-1, 1)
        # predict_plot(scaler.inverse_transform(y_test), scaler.inverse_transform(y_pred))
