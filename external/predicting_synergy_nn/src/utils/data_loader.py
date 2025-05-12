import os
import pandas as pd
import torch
from torch.utils.data import DataLoader, TensorDataset, Dataset
from sklearn.preprocessing import StandardScaler
from typing import Tuple, Literal
import numpy as np


def load_data(dir_path, fold, target='ZIP', batch=100):
    train = pd.read_csv(os.path.join(dir_path, f'fold{fold}/fold{fold}_alltrain.csv'), header=0)
    test = pd.read_csv(os.path.join(dir_path, f'fold{fold}/fold{fold}_test.csv'), header=0)
    
    train = train.sample(frac=1)
    test = test.sample(frac=1)
    
    x_tr = train.iloc[:, 0:33].values
    y_tr = train[target].values.reshape(-1, 1)
    
    x_ts = test.iloc[:, 0:33].values
    y_ts = test[target].values.reshape(-1, 1)
    
    sc = StandardScaler()
    sc.fit(y_tr)
    y_tr = sc.transform(y_tr)
    y_ts = sc.transform(y_ts)
    
    dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    x_tr_t = torch.FloatTensor(x_tr).to(dev)
    y_tr_t = torch.FloatTensor(y_tr).to(dev)
    x_ts_t = torch.FloatTensor(x_ts).to(dev)
    y_ts_t = torch.FloatTensor(y_ts).to(dev)
    
    tr_ds = TensorDataset(x_tr_t, y_tr_t)
    ts_ds = TensorDataset(x_ts_t, y_ts_t)
    
    tr_dl = DataLoader(tr_ds, batch_size=batch, shuffle=True)
    ts_dl = DataLoader(ts_ds, batch_size=batch, shuffle=False)
    
    return x_tr, y_tr, x_ts, y_ts, sc, tr_dl, ts_dl

def load_valid_data(dir_path, fold, split, target='ZIP', batch=100):
    train = pd.read_csv(os.path.join(dir_path, f'fold{fold}/validation/fold{fold}_train{split}.csv'), header=0)
    valid = pd.read_csv(os.path.join(dir_path, f'fold{fold}/validation/fold{fold}_valid{split}.csv'), header=0)
    
    train = train.sample(frac=1)
    valid = valid.sample(frac=1)
    
    x_tr = train.iloc[:, 0:33].values
    y_tr = train[target].values.reshape(-1, 1)
    
    x_vl = valid.iloc[:, 0:33].values
    y_vl = valid[target].values.reshape(-1, 1)
    
    sc = StandardScaler()
    sc.fit(y_tr)
    y_tr = sc.transform(y_tr)
    y_vl = sc.transform(y_vl)
    
    dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    x_tr_t = torch.FloatTensor(x_tr).to(dev)
    y_tr_t = torch.FloatTensor(y_tr).to(dev)
    x_vl_t = torch.FloatTensor(x_vl).to(dev)
    y_vl_t = torch.FloatTensor(y_vl).to(dev)
    
    tr_ds = TensorDataset(x_tr_t, y_tr_t)
    vl_ds = TensorDataset(x_vl_t, y_vl_t)
    
    tr_dl = DataLoader(tr_ds, batch_size=batch, shuffle=True)
    vl_dl = DataLoader(vl_ds, batch_size=batch, shuffle=False)
    
    return x_tr, y_tr, x_vl, y_vl, sc, tr_dl, vl_dl

class CSVTabularDataset(Dataset):
    def __init__(self, df):
        self.df = df

    def __len__(self):
        return len(self.df)

    def get_dataset(self, target = "ZIP") -> Tuple[np.ndarray, np.ndarray]:
        x = self.df.iloc[:, 0:33]
        y = self.df[target]
        # cast to numpy array
        x = x.values
        y = y.values.ravel()
        return x, y

class CVDatasetHandler:
    def __init__(self, data_dir, outer_fold=1):
        self.data_dir = data_dir + '/fold' + str(outer_fold)
        self.outer_fold = outer_fold

    def _load_csvs(self, paths):
        dfs = [pd.read_csv(p) for p in paths]
        return pd.concat(dfs, ignore_index=True)

    def setup(self):
        val_dir = os.path.join(self.data_dir, 'validation')
        
        self.train_sets = []
        self.val_sets = []
        for i in range(1, 4):
            train_csv = os.path.join(val_dir, f'fold{self.outer_fold}_train{i}.csv')
            valid_csv = os.path.join(val_dir, f'fold{self.outer_fold}_valid{i}.csv')
            train_df = pd.read_csv(train_csv)
            valid_df = pd.read_csv(valid_csv)
            self.train_sets.append(CSVTabularDataset(train_df))
            self.val_sets.append(CSVTabularDataset(valid_df))

        alltrain_csv = os.path.join(self.data_dir, 'fold1_alltrain.csv')
        test_csv = os.path.join(self.data_dir, 'fold1_test.csv')
        self.all_train_dataset = CSVTabularDataset(pd.read_csv(alltrain_csv))
        self.test_dataset = CSVTabularDataset(pd.read_csv(test_csv))
        
    def get_dataset(self, type: Literal['train', 'val', 'test', 'alltrain'], fold: Literal[1, 2, 3] = 1) -> Tuple[np.ndarray, np.ndarray]:
        if type == 'train':
            return self.train_sets[fold - 1].get_dataset()
        elif type == 'val':
            return self.val_sets[fold - 1].get_dataset()
        elif type == 'test':
            return self.test_dataset.get_dataset()
        elif type == 'alltrain':
            return self.all_train_dataset.get_dataset()
        else:
            raise ValueError(f"Unknown dataset type: {type}")
