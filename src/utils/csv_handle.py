import pandas as pd


def open_data(path: str, file_name: str, multivariate: bool = False) -> pd.DataFrame:

    data = pd.read_csv(
        f'{path}/{file_name}.csv',
        index_col='open_time',
    )
    data.index = pd.to_datetime(data.index, unit='ms')
    if multivariate:
        # selected columns by the correlation matrix
        selected_columns = ['open', 'high', 'low', 'close_time', 'close']
        data = data[selected_columns]

    return data
