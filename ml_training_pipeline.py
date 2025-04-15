import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
import joblib

from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import classification_report

from utils_training import split_time_series
from ml_models import RandomForest, RidgeClassifierModel
from ml_datasets import FullDataset

NORMALAZATION_TECHNIQUE = StandardScaler()
MODELS = [RandomForest(), RidgeClassifierModel()]


if __name__ == "__main__":
    data = FullDataset()
    x_train, y_train = data.train()
    x_test, y_test = data.test()
    
    for MODEL in MODELS:

        # Define the pipeline
        pipeline = Pipeline([
            ('norm_tech', NORMALAZATION_TECHNIQUE),
            ('model', MODEL.get_model())
        ])

        # Define the parameter grid for GridSearch
        param_grid = MODEL.param_grid

        # Initialize GridSearchCV
        tscv, _ = split_time_series(x_train, n_splits=5)
        grid_search = GridSearchCV(estimator=pipeline, param_grid=param_grid, cv=tscv, scoring='accuracy', verbose=1)

        grid_search.fit(x_train, y_train)
        print("Best parameters found: ", grid_search.best_params_)
        
        # Evaluate and the save the results
        y_pred_tr = grid_search.predict(x_train)
        report_train = classification_report(y_train, y_pred_tr)

        y_pred = grid_search.predict(x_test)
        report_test = classification_report(y_test, y_pred, output_dict=True)
        report_df = pd.DataFrame(report_test).rename(columns={"0.0": "Class 0", "1.0": "Class 1"})
        report_df.to_csv(f'results/{MODEL.str_name}_Classification_Report_{data.get_name}.csv')
        
        print(report_train)
        print(report_df.transpose())

        # Save the model
        joblib.dump(grid_search, f'models_ckp/{MODEL.str_name}_Grid_Search_Model_{data.get_name}.pkl')