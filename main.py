import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
import joblib 
import argparse
import torch

from utils_training import split_time_series

NN_MODEL_DIR = 'checkpoints/TsNet.pt'
ML_MODEL_DIR = 'models_ckp/____'

def process_data(data: pd.DataFrame) -> pd.DataFrame:
    pass

def _predict_ml_model(model_ckp: Path, X: np.array | pd.DataFrame) -> np.array:
    model = joblib.load(model_ckp)
    return model.predict(X)

def _predict_nn_model(model_ckp: Path, X: np.array | pd.DataFrame) -> np.array:
    model = TsNet().load_from_checkpoint(model_ckp)
    model.eval()
    with torch.no_grad():
        preds = model(X)
    return (preds > 0.5).float().to('cpu').numpy()

def predict(dir_model:Path, data_dir:Path, mode:str) -> np.array[1, ...]:
    if mode == 'nn':
        data = NNNewDataSet(data_dir)
        X = data.eval()
        return _predict_nn_model(dir_model, X)
    else:
        data = NewDataSet(data_dir)
        X = data.eval()
        return _predict_ml_model(dir_model, X)
    
# process
def pipeline(data_dir:Path, dir_model:Path, mode:str, dir_results:Path):
    """
    Main function to process data and train models.
    """
    preds = predict(dir_model, data_dir, mode)
    # save predictions
    preds_df = pd.DataFrame(preds)
    preds_df.to_csv(dir_results, index=False)
    

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run the SynthStockPredictor pipeline.")

    parser.add_argument(
        "--data_dir", 
        type=str, 
        required=True, 
        help="Path to the data directory."
    )

    parser.add_argument(
        "--model_type", 
        type=str, 
        required=True, 
        help="Type of model to use (e.g., 'ml' for classic machine learning model or 'nn' for a neural network model)."
    )
    
    parser.add_argument(
        "--dir_results",
        type=str,
        required=True,
        help="Directory to save the results."
    )
    
    args = parser.parse_args()

    pipeline(
        data_dir=Path(args.data_dir),
        dir_model=Path(ML_MODEL_DIR if args.model_type == 'ml' else NN_MODEL_DIR),
        mode=args.model_type,
        dir_results=Path(args.dir_results)
    )


