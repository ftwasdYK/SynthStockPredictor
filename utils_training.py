from pathlib import Path
from sklearn.model_selection import TimeSeriesSplit
import pandas as pd
import torch
from torch import nn
import torchmetrics

def split_time_series(data, n_splits:int=5) -> tuple:
    """
    Splits time series data into train-test sets using TimeSeriesSplit.

    Parameters:
        data (array-like): The time series data to split.
        n_splits (int): Number of splits for TimeSeriesSplit.

    Returns:
        list: A list of (train_indices, test_indices) tuples.
    """
    tscv = TimeSeriesSplit(n_splits=n_splits)
    splits = []
    for train_index, test_index in tscv.split(data):
        splits.append((train_index, test_index))
    return tscv, splits


## Pytorch ##
class InferClassificationReport():
    """
    In
    """
    def __init__(self, model:nn.Module, device:str, model_ckp:Path):
        self.model = model
        self.device = device
        self.model_ckp = model_ckp
        self.acc_metric = torchmetrics.classification.BinaryAccuracy().to(self.device)
        self.prec_metric = torchmetrics.classification.BinaryPrecision().to(self.device)
        self.rec_metric = torchmetrics.classification.BinaryRecall().to(self.device)

    def eval(self, val_loader:torch.utils.data.DataLoader) -> dict:
        with torch.no_grad():
            for data, target in val_loader:
                data, target = data.to(self.device), target.to(self.device)
                output = self.model(data)
                acc = self.acc_metric(output, target)
                precision = self.prec_metric(output, target)
                recall = self.rec_metric(output, target)
                    
            acc = self.acc_metric.compute()
            precision = self.prec_metric.compute()
            recall = self.rec_metric.compute()
            return {
                "accuracy": acc,
                "precision": precision,
                "recall": recall
            }
        
class FindBestModel:
    """
    Find the best model based on the score.
    """
    def __init__(self):
        self.max = -10000
        self.best= None
    
    def __call__(self, score:float, cpk_dir:Path):
        if score > self.max:
            self.max = score
            self.cpk_dir = cpk_dir
            self.best = {
                "score": score,
                "model_weigths_dir": cpk_dir
            }
    
    
    def get_best_model(self) -> dict:
        return self.best