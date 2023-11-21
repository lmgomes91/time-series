import logging

import pandas as pd


def correlation_matrix(path: str, file_name: str):
    data = pd.read_csv(
        f'{path}/{file_name}.csv',
        index_col='open_time',
    )

    data.head().to_csv('data_example.csv')

    data.index = pd.to_datetime(data.index, unit='ms')
    corr_matrix = (data.corr()).round()

    corr_matrix.to_csv('correlation_matrix.csv')

    logging.info(corr_matrix['close'])
