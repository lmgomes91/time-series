from abc import ABC, abstractmethod

import pandas as pd


class BaseModel(ABC):
    @staticmethod
    @abstractmethod
    def run(data: pd.DataFrame):
        pass
