import logging

from src.analysis.results.results_handle import results_handle
from src.neural_networks_models.multivariate.cnn_multi import CnnMultivariate
from src.neural_networks_models.multivariate.gru_mulit import GruMultivariate
from src.neural_networks_models.multivariate.lstm_multi import LstmMultivariate
from src.neural_networks_models.univariate.cnn import Cnn
from src.neural_networks_models.univariate.gru import Gru
from src.neural_networks_models.univariate.lstm import Lstm

from src.utils.csv_handle import open_data

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s [%(levelname)s] %(filename)s:%(lineno)d: %(message)s' # noqa
)


def main():
    # data = open_data('./data', 'BTCBUSD-1m-2023-01')
    # for _ in range(0, 10):
    #     Cnn().run(data)
    #     Gru.run(data)
    #     Lstm().run(data)
    #
    # data = open_data('./data', 'BTCBUSD-1m-2023-01', True)
    # for _ in range(0, 10):
    #     CnnMultivariate().run(data)
    #     GruMultivariate().run(data)
    #     LstmMultivariate().run(data)

    results_handle()


if __name__ == '__main__':
    main()
