import logging

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split


def preprocess_data_multivariate(data: pd.DataFrame, sequence_length: int) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    # Create sequences of data for training
    x, y = [], []
    for i in range(len(data) - sequence_length):
        x.append(data[i:i + sequence_length, :])
        y.append(data[i + sequence_length, 4])

    x, y = np.array(x), np.array(y)

    # Reshape input data for LSTM
    x = x.reshape(x.shape[0], x.shape[1], data.shape[1])

    # Split data using train_test_split
    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=42)

    return x_train, x_test, y_train, y_test
