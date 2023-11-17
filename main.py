import logging

from src.neural_networks_models.cnn import Cnn
from src.neural_networks_models.gru import Gru
from src.neural_networks_models.lstm import Lstm
from src.neural_networks_models.mlp import Mlp
from src.neural_networks_models.rbf import Rbf
from src.utils.csv_handle import open_data

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s [%(levelname)s] %(filename)s:%(lineno)d - %(message)s' # noqa
)


def main():
    data = open_data('./data', 'BTCBUSD-1m-2023-01') # noqa
    Lstm().run(data)
    # Cnn().run(data)
    # Rbf.run(data)
    # Mlp.run(data)
    # Gru.run(data)


if __name__ == '__main__':
    main()
