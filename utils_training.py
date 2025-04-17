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
    Class to infer the classification report for the model.
    """
    def __init__(self, model:nn.Module, device:str):
        self.model = model
        self.device = device
        self.acc_metric = torchmetrics.classification.BinaryAccuracy().to(self.device)
        self.prec_metric = torchmetrics.classification.BinaryPrecision().to(self.device)
        self.rec_metric = torchmetrics.classification.BinaryRecall().to(self.device)

    def eval(self, val_loader:torch.utils.data.DataLoader) -> dict:
        self.model.eval()
        with torch.no_grad():
            for data, target in val_loader:
                data, target = data.to(self.device), target.to(self.device)
                output = self.model(data)
                acc = self.acc_metric(output.squeeze(), target.squeeze())
                precision = self.prec_metric(output.squeeze(), target.squeeze())
                recall = self.rec_metric(output.squeeze(), target.squeeze())
                    
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
    
    def __call__(self, score:float, model:nn.Module):
        if score > self.max:
            self.max = score
            self.best = {
                "score": score,
                "model": model
            }
    
    
    def get_best_model(self) -> dict:
        return self.best