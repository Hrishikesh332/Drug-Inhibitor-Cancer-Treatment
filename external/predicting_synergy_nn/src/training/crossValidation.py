import torch
import numpy as np
from sklearn.model_selection import KFold
from torch.utils.data import TensorDataset, DataLoader
from src.training.trainer import train_model  
from src.utils.data_loader import load_data
import yaml


def cross_validate(cfg):
    cross_val_file = cfg['Cross_val']
    #print(f'cross_val_file : {cross_val_file}')
    with open(cross_val_file, 'r') as f:
        crs = yaml.safe_load(f)
    
    
    n_splits = crs['k_fold_split']
    data_folds = crs['data_fold']
    #print(f'K fold n_splits : {n_splits}')
    #print(f'data_folds : {data_folds}')
    X, Y, x_ts, y_ts, sc, tr_dl, ts_dl = load_data(cfg, cfg['data_dir'],data_folds, batch=cfg['batch'])

    kf = KFold(n_splits=n_splits, shuffle=True, random_state=42)
    all_results = []
    #print(f"\n---Inside Cross Validation ---")
    #print(f"kf.split(X) : {kf.split(X)}")

    for fold, (train_idx, val_idx) in enumerate(kf.split(X)):
        #print(f"\n---Inside Cross Validation for Loop ---")
        #print(f"\n--- Fold {fold + 1}/{n_splits} ---")

        x_tr = torch.tensor(X[train_idx], dtype=torch.float32)
        y_tr = torch.tensor(Y[train_idx], dtype=torch.float32)
        x_val = torch.tensor(X[val_idx], dtype=torch.float32)
        y_val = torch.tensor(Y[val_idx], dtype=torch.float32)

        tr_ds = TensorDataset(x_tr, y_tr)
        val_ds = TensorDataset(x_val, y_val)
        tr_dl = DataLoader(tr_ds, batch_size=cfg['batch'], shuffle=True)
        val_dl = DataLoader(val_ds, batch_size=cfg['batch'], shuffle=False)

        fold_cfg = cfg.copy()
        fold_cfg['fold'] = fold + 1
        fold_cfg['wb_run'] = f"{cfg.get('wb_run_base', 'run')}_fold{fold + 1}"

        #print(f"----Model Trining Started----")
        try:
            result = train_model(fold_cfg, tr_dl, val_dl,sc ,in_dim=X.shape[1]) #passed scaler parameter 
            #print(f"\n---Inside Cross Validation Training Done for Fold {fold + 1}/{n_splits}---")
            all_results.append(result)
        except Exception as e:
            print(f"Error in fold {fold + 1}: {e}")

    return all_results
