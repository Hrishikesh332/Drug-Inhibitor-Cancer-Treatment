import os
import pandas as pd
import torch
from torch.utils.data import DataLoader, TensorDataset
from sklearn.preprocessing import StandardScaler

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