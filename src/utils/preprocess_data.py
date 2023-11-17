import numpy as np
import pandas as pd


def preprocess_data(data: pd.DataFrame, sequence_length: int) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    # # Split the data into input and output sequences
    values = data.iloc[:, 1:2].values

    x = []
    y = []

    for i in range(len(data) - sequence_length):
        x.append(values[i:i + sequence_length])
        y.append(values[i + sequence_length])

    x = np.array(x)
    y = np.array(y)

    # Split the data into training and testing sets
    split_ratio = 0.8
    split_index = int(split_ratio * len(x))
    x_train, x_test = x[:split_index], x[split_index:]
    y_train, y_test = y[:split_index], y[split_index:]

    return x_train, x_test, y_train, y_test
