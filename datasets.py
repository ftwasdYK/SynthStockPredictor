import numpy as np
import pandas as pd
from pathlib import Path
import torch

from utils_training import split_time_series

__all__ = ['FullDataset', 'UncorrelatedDataset' 'NewDataset']

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
        self.X = pd.read_csv('processed_data/x_full_data.csv')
        self.y = pd.read_csv('processed_data/y_data.csv')

class UncorrelatedDataset(BluePrintDataset):
    def __init__(self):
        self.X = pd.read_csv('processed_data/x_uncorrelated_data.csv')
        self.y = pd.read_csv('processed_data/y_data.csv')


## Neural Network Datasets ##

class FullDatasetTorch(FullDataset):
    def __init__(self):
        super().__init__()
        _, self.split_indexes = split_time_series(self.train()[0], n_splits=5)
        self._preprocess_data()

    def _preprocess_data(self):
        # Convert to tensors
        self.X = torch.tensor(self.X.to_numpy(), dtype=torch.float32)
        self.y = torch.tensor(self.y.to_numpy(), dtype=torch.float32)

    def get_splits(self) -> tuple:
        """
        Returns the train and test splits
        """
        return self.split_indexes

class UncorrelatedDatasetTorch(UncorrelatedDataset):
    def __init__(self):
        super().__init__()
        _, self.split_indexes = split_time_series(self.train()[0], n_splits=5)
        self._preprocess_data()

    def _preprocess_data(self):
        # Convert to tensors
        self.X = torch.tensor(self.X.to_numpy(), dtype=torch.float32)
        self.y = torch.tensor(self.y.to_numpy(), dtype=torch.float32)
    
    def get_splits(self) -> tuple:
        """
        Returns the train and test splits
        """
        return self.split_indexes


## For unknown data (without labels), production mode ##
class NewDataset(BluePrintDataset):
    """
    This class is used to load the new data and preprocess it.
    """
    def __init__(self, dir_csv:Path):
        self.data = pd.read_csv(dir_csv)

    def _preprocess_data(self):
        """
        This method is used to add new features to the data.
        The features are:
            - returns
            - EMA_10
            - MACD
            - Signal_Line
            - RSI
            - Volume

        The data should have the following columns:
            - stock_price
            - volume
        """
        
        # Moving Averages
        self.data['EMA_10'] = self.data['stock_price'].ewm(span=10, adjust=True).mean()
        
        # MACD
        self.data['MACD'] = self.data['stock_price'].ewm(span=12, adjust=True).mean() - self.data['stock_price'].ewm(span=26, adjust=True).mean()
        self.data['Signal_Line'] = self.data['MACD'].ewm(span=9, adjust=True).mean()
        
        # RSI
        delta = self.data['stock_price'].diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
        rs = gain / loss
        self.data['RSI'] = 100 - (100 / (1 + rs))

        self.data.dropna(inplace=True)

        self.data['returns'] = self.data['stock_price'].pct_change()
        self.data = self.data.dropna()
        self.data = self.data[['returns', 'EMA_10', 'MACD', 'Signal_Line', 'RSI', 'volume']]
        

    def get_data(self) -> np.array:
        """
        Returns the data as a numpy array
        """
        self._preprocess_data()
        return self.data.to_numpy()
    
    def get_data_tensors(self) -> torch.Tensor:
        """
        Returns the data as a tensor
        """
        self._preprocess_data()
        return torch.tensor(self.data.to_numpy(), dtype=torch.float32)