import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from torch.optim.lr_scheduler import StepLR
import torchmetrics

from datasets import UncorrelatedDatasetTorch
from utils_training import split_time_series, InferClassificationReport, FindBestModel
from nn_models import TsNet, MLPNet, LSTMNet
from nn_trainers import BinaryClassificationTrainer

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
BATCH_SIZE = 64
MODELS = [TsNet, MLPNet, LSTMNet]

if __name__ == "__main__":
    store_best_model = FindBestModel()
    metric = 'accuracy'
    # Load data
    data = UncorrelatedDatasetTorch()
    x_tensor, y_tensor = data.train()
    splits_indexes = data.get_splits()

    # Create dataset
    dataset = torch.utils.data.TensorDataset(x_tensor, y_tensor.unsqueeze(1))
    
    for MODEL in MODELS:
        metric_keep = []
        for train_idx, val_idx in splits_indexes:

            train_set = torch.utils.data.Subset(dataset, train_idx)
            val_set = torch.utils.data.Subset(dataset, val_idx)
        
            train_loader = torch.utils.data.DataLoader(train_set, batch_size=BATCH_SIZE, shuffle=False)
            val_loader = torch.utils.data.DataLoader(val_set, batch_size=BATCH_SIZE, shuffle=False)

            NN_model = MODEL().to(DEVICE)

            # Training Hyperparams
            epochs = 50

            # # Optimizer
            lr = 1e-2 #
            weight_decay = 1e-5
            optimizer = torch.optim.AdamW(NN_model.parameters(), lr=lr, weight_decay=weight_decay)
            loss_fn = torch.nn.BCELoss()
            scheduler = StepLR(optimizer, step_size=5, gamma=0.5)

            model_trainer = BinaryClassificationTrainer(
                NN_model,
                train_loader,
                val_loader,
                loss_fn,
                optimizer,
                epochs,
                DEVICE,
                patience=10,
                scheduler=scheduler,
            )

            hist, ckp_dir = model_trainer.train_nn()
            metric_keep.append(InferClassificationReport(NN_model, DEVICE).eval(val_loader=val_loader)[f'{metric}'])

        avg_metric = np.mean(metric_keep)
        store_best_model(avg_metric, MODEL)
    
    best_model = store_best_model.get_best_model()

    # start retraining the best model to the whole training dataset
    nn_model = best_model['model']().to(DEVICE)
    optimizer = torch.optim.AdamW(NN_model.parameters(), lr=lr, weight_decay=weight_decay)
    loss_fn = torch.nn.BCELoss()
    scheduler = StepLR(optimizer, step_size=5, gamma=0.5)
    
    model_trainer = BinaryClassificationTrainer(
        NN_model,
        train_loader,
        val_loader,
        loss_fn,
        optimizer,
        epochs,
        DEVICE,
        patience=10,
        scheduler=scheduler,
        save_bool=True,
        plot_flag=True,
    )
    
    hist, ckp_dir = model_trainer.train_nn()
    
    # load the best model
    NN_model = model_trainer.load_checkpoint(ckp_dir)
    
    # start eval
    metric = torchmetrics.classification.BinaryAccuracy().to(DEVICE)
    x_test, y_test = data.test()
    # Create dataset
    dataset = torch.utils.data.TensorDataset(x_test, y_test)
    test_loader = torch.utils.data.DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=False)

    results = InferClassificationReport(NN_model, DEVICE).eval(val_loader=test_loader)
    # save results
    results_df = pd.DataFrame([results])
    results_df.to_csv(f'results_training/{NN_model.__class__.__name__}_Classification_Report_{data.get_name}.csv', index=False)