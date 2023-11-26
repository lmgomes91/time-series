import logging

from src.neural_networks_models.multivariate.cnn_multi import CnnMultivariate
from src.neural_networks_models.univariate.cnn import Cnn
from src.neural_networks_models.univariate.gru import Gru
from src.neural_networks_models.univariate.lstm import Lstm

from src.utils.csv_handle import open_data

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s [%(levelname)s] %(filename)s:%(lineno)d: %(message)s' # noqa
)


def main():
    data = open_data('./data', 'BTCBUSD-1m-2023-01')
    # Cnn().run(data)
    # Lstm().run(data)
    # Gru.run(data)

    # data = open_data('./data', 'BTCBUSD-1m-2023-01', True)
    CnnMultivariate().run(data)


if __name__ == '__main__':
    main()
