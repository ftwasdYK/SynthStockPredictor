import numpy as np
import pandas as pd

from pathlib import Path

__all__ = ['FullDataset', 'NewDataset']

class BluePrintDataset(object):
    def __init__(self):
        pass

    def train(self) -> tuple[np.array, np.array]:
        """
        Returns all the data except the last 1000 rows
        """
        return self.X[:-1000], self.y[:-1000].squeeze()
    
    def test(self) -> tuple[np.array, np.array]:
        """
        Returns the last 1000 rows of the dataset for testing the ml algorithms
        """
        return self.X[-1000:], self.y[-1000:].squeeze()
    
    @property
    def get_name(self) -> str:
        return self.__class__.__name__

    def _preprocess_data(self):
        """
        Preprocess the data
        """
        pass

class FullDataset(BluePrintDataset):
    def __init__(self):
        self.X = pd.read_csv('processed_data/X_full_post_processed_data.csv')
        self.y = pd.read_csv('processed_data/y_post_processed_data.csv')



class NewDataset(BluePrintDataset):
    """
    This class is used to load the new data and preprocess it.
    """
    def __init__(self, dir_csv:Path):
        self.data = pd.read_csv(dir_csv)

    def _preprocess_data(self):
        pass