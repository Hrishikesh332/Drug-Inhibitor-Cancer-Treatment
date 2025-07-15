import os
from typing import Tuple
import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader, TensorDataset, Dataset
from sklearn.preprocessing import StandardScaler
from typing import Literal

TARGET_COL = 'ZIP'
METADATA_COLS = ['Bliss', 'DRUG1', 'DRUG2', 'CELL_LINE']
CELL_LINE_COLS = ['GATA3', 'NF1', 'NF2', 'P53', 'PI3K', 'PTEN', 'RAS']

def _load_raw_train_data_for_fold(datapath: str, fold: int) -> pd.DataFrame:
    path = os.path.join(datapath, f'fold{fold}/fold{fold}_alltrain.csv')
    return pd.read_csv(path, header=0)

def _load_raw_test_data_for_fold(datapath: str, fold: int) -> pd.DataFrame:
    path = os.path.join(datapath, f'fold{fold}/fold{fold}_test.csv')
    return pd.read_csv(path, header=0)




def load_data(datapath: str, fold: Literal[1, 2, 3], cell_line_encoding: Literal['OHE', 'ordinal'] | None, batch: int = 100):
    # Load raw data
    if int(fold) in [1, 2, 3]:
        train_df = _load_raw_train_data_for_fold(datapath, fold)
        test_df = _load_raw_test_data_for_fold(datapath, fold)
    else:
        raise ValueError(f"Invalid argument for {fold=}. Please use 1, 2, or 3.")

    # Shuffle
    train_df = train_df.sample(frac=1)
    test_df = test_df.sample(frac=1)

    # Cell line encoding - Fixed in this project: previously 0: no effect 1: negative effect 2:positive effect - which is an incoreect method of encoding.
    if cell_line_encoding == 'ordinal':
        remap = {0: 0, 1: -1, 2: 1}
        train_df[CELL_LINE_COLS] = train_df[CELL_LINE_COLS].replace(remap)
        test_df[CELL_LINE_COLS] = test_df[CELL_LINE_COLS].replace(remap)
    elif cell_line_encoding == 'OHE':
        # combine train and test data to ensure consistent one-hot encoding
        combined = pd.concat([train_df, test_df], keys=["train", "test"])
        # OHE (i.e. dummy!)
        combined = pd.get_dummies(combined, columns=CELL_LINE_COLS, drop_first=True)
        # split the data back into train and test
        train_df = combined.xs("train")
        test_df = combined.xs("test")
    elif cell_line_encoding is None:
        pass
    else:
        raise ValueError(f"Invalid argument for {cell_line_encoding=}. Please use 'OHE', 'ordinal', or None.")

    # Column filtering
    non_feature_cols = METADATA_COLS + [TARGET_COL]
    feature_cols = [c for c in train_df.columns if c not in non_feature_cols]

    x_tr = train_df.loc[:, feature_cols].astype(np.float32).values
    y_tr = train_df[TARGET_COL].values.reshape(-1, 1)
    
    x_ts = test_df.loc[:, feature_cols].astype(np.float32).values
    y_ts = test_df[TARGET_COL].values.reshape(-1, 1)

    # Scaling
    sc = StandardScaler()
    sc.fit(y_tr)
    y_tr = sc.transform(y_tr)
    y_ts = sc.transform(y_ts)

    # Torch stuff
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
    
    def get_drug_names(self) -> tuple[pd.Series, pd.Series]:
        return self.df['DRUG1'], self.df['DRUG2']
    
    def get_cell_lines(self) -> np.ndarray:
        return self.df['CELL_LINE'].values
    

class CVDatasetHandler:
    def __init__(self, data_dir, outer_fold=1):
        self.data_dir = data_dir + '/fold' + str(outer_fold)
        self.outer_fold = outer_fold
        self._read_datasets()

    def _load_csvs(self, paths):
        dfs = [pd.read_csv(p) for p in paths]
        return pd.concat(dfs, ignore_index=True)

    def _read_datasets(self):
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
