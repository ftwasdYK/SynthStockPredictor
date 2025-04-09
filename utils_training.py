from sklearn.model_selection import TimeSeriesSplit
import pandas as pd

def split_time_series(data, n_splits:int=5) -> list:
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

# Example usage:
# data = pd.read_csv('synthetic_data.csv')
# splits = split_time_series(data, n_splits=5)
# for train_idx, test_idx in splits:
#     print("Train:", train_idx, "Test:", test_idx)