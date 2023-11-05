import pandas as pd


def open_data(path: str, file_name: str) -> pd.DataFrame:

    data = pd.read_csv(
        f'{path}/{file_name}.csv',
        index_col='open_time',
    )
    data.index = pd.to_datetime(data.index, unit='ms')

    return data
