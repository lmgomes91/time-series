from src.analysis.data_details.month_values_graph import values_line_graph
from src.utils.csv_handle import open_data


def main():
    data = open_data('./data', 'BTCBUSD-1m-2023-01') # noqa
    values_line_graph(data)


if __name__ == '__main__':
    main()
