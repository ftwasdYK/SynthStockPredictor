import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
import joblib 
import argparse
import torch

from datasets import NewDataset
from nn_models import TsNet

NN_MODEL_DIR = 'checkpoints/TsNet.pt'
ML_MODEL_DIR = 'models_ckp/SupportVectorMachine_Grid_Search_Model_UncorrelatedDataset.pkl'
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
MODEL = TsNet().to(DEVICE)

def _predict_ml_model(model_ckp: Path, X: np.array) -> np.array:
    model = joblib.load(model_ckp)
    return model.predict(X)

def _predict_nn_model(model_ckp: Path, X: np.array) -> np.array:
    model = MODEL().load_from_checkpoint(model_ckp)
    model.eval()
    with torch.no_grad():
        preds = model(X)
    return (preds > 0.5).float().to('cpu').numpy()

def predict(dir_model:Path, data_dir:Path, mode:str) -> np.array:
    if mode == 'nn':
        data = NewDataset(data_dir)
        X = data.get_data_tensor()
        return _predict_nn_model(dir_model, X)
    else:
        data = NewDataset(data_dir)
        X = data.get_data()
        return _predict_ml_model(dir_model, X)
    
# main function
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
        default="raw_data/raw_data.csv",
        help="Path to the data directory."
    )

    parser.add_argument(
        "--model_type", 
        type=str, 
        default="ml",
        help="Type of model to use (e.g., 'ml' for classic machine learning model or 'nn' for a neural network model)."
    )
    
    parser.add_argument(
        "--dir_results",
        type=str,
        default="results/results.csv",
        help="Directory to save the results."
    )
    
    args = parser.parse_args()

    pipeline(
        data_dir=Path(args.data_dir),
        dir_model=Path(ML_MODEL_DIR if args.model_type == 'ml' else NN_MODEL_DIR),
        mode=args.model_type,
        dir_results=Path(args.dir_results)
    )