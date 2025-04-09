import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path


from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import GridSearchCV
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import RidgeClassifier

from utils_training import split_time_series
from ml_models import RandomForest, RidgeClassifierModel

NORMALAZATION_TECHNIQUE = StandardScaler()
MODEL = RandomForest() 


if __name__ == "__main__":
    # read data
    X = pd.read_csv('processed_data/full_dataset_X.csv')
    y = pd.read_csv('processed_data/full_dataset_y.csv')
    print(y.shape)
    # Define the pipeline
    pipeline = Pipeline([
        ('norm_tech', NORMALAZATION_TECHNIQUE),
        ('model', MODEL.get_model())
    ])

    # Define the parameter grid for GridSearch
    param_grid = MODEL.param_grid

    # Initialize GridSearchCV
    tscv, _ = split_time_series(X)
    grid_search = GridSearchCV(estimator=pipeline, param_grid=param_grid, cv=tscv, scoring='accuracy', verbose=1)

    grid_search.fit(X, y.squeeze())
    print("Best parameters found: ", grid_search.best_params_)
    print("Best score: ", grid_search.best_score_)
