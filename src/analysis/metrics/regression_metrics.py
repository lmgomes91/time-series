import logging

import tensorflow as tf
import numpy as np

from src.utils.db import save_result_in_db


def regression_metrics(y_test: np.ndarray, y_pred: np.ndarray, model_name: str):
    # Mean Squared Error (MSE):
    mse = tf.keras.metrics.mean_squared_error(y_test, y_pred)
    # Mean Absolute Error (MAE):
    mae = tf.keras.metrics.mean_absolute_error(y_test, y_pred)
    # Root Mean Squared Error (RMSE):
    rmse = tf.sqrt(mse)
    # Mean Absolute Percentage Error (MAPE):
    mape = tf.keras.metrics.mean_absolute_percentage_error(y_test, y_pred)
    # Theil U coefficient
    theil_u_coefficient = np.mean((y_test - y_pred) ** 2) / np.mean(y_test ** 2)

    logging.info(f'Mean Squared Error (MSE): {tf.reduce_mean(mse)}')
    logging.info(f'Mean Absolute Error (MAE): {tf.reduce_mean(mae)}')
    logging.info(f'Root Mean Squared Error (RMSE): {tf.reduce_mean(rmse)}')
    logging.info(f'Mean Absolute Percentage Error (MAPE): {tf.reduce_mean(mape)}')
    logging.info(f'Theil U coefficient: {theil_u_coefficient}')

    save_result_in_db(mse, mae, rmse, mape, theil_u_coefficient, model_name)
