import logging
import pandas as pd
from src.utils.db import get_results_by_model


def results_handle():
    methods = [
        'cnn_multi',
        'gru_multi',
        'lstm_multi',
        'cnn_uni',
        'gru_uni',
        'lstm_uni'
    ]
    for method in methods:
        results = get_results_by_model(method)
        results.columns = results.columns.str.upper()

        # Insert row with mean values
        mean_row = results.mean().to_frame().T
        mean_row.index = ['Média']

        std_row = results.std().to_frame().T
        std_row.index = ['Desvio Padrão']

        df = pd.concat([results, mean_row, std_row])

        df.to_csv(f'results/{method}.csv', index=True)
