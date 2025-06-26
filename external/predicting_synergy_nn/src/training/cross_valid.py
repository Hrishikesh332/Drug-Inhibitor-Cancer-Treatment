import os
import pandas as pd
import numpy as np
import torch
from torch.utils.data import DataLoader, TensorDataset
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import GroupKFold
from typing import Tuple, Union, List, Optional
import warnings

class UnifiedCrossValidator:
    
    def __init__(self, data_dir: str, cv_strategy: str = 'random', 
                 cell_line_col: str = None, drug_col: str = None):
        self.data_dir = data_dir
        self.cv_strategy = cv_strategy
        self.cell_line_col = cell_line_col
        self.drug_col = drug_col
        self._analyzed = False
        self._data_cache = None
        
    def _load_all_data(self) -> pd.DataFrame:
        if self._data_cache is not None:
            return self._data_cache
            
        all_data = []
        for fold in [1, 2, 3]:
            fold_dir = os.path.join(self.data_dir, f'fold{fold}')
            if os.path.exists(fold_dir):
                train_file = os.path.join(fold_dir, f'fold{fold}_alltrain.csv')
                if os.path.exists(train_file):
                    train_df = pd.read_csv(train_file)
                    train_df['original_fold'] = fold
                    train_df['original_split'] = 'train'
                    all_data.append(train_df)
                
                test_file = os.path.join(fold_dir, f'fold{fold}_test.csv')
                if os.path.exists(test_file):
                    test_df = pd.read_csv(test_file)
                    test_df['original_fold'] = fold
                    test_df['original_split'] = 'test'
                    all_data.append(test_df)
        
        if not all_data:
            raise FileNotFoundError("No data files found in the specified directory")
        
        combined_data = pd.concat(all_data, ignore_index=True)
        self._data_cache = combined_data
        return combined_data
    
    def _auto_detect_columns(self, df: pd.DataFrame):
        if not self._analyzed:
            columns = df.columns.tolist()
            
            if self.cell_line_col is None:
                cell_candidates = []
                for col in columns:
                    col_lower = col.lower()
                    if any(keyword in col_lower for keyword in ['cell', 'line', 'cellline']):
                        unique_count = df[col].nunique()
                        total_count = len(df)
                        if 5 <= unique_count <= total_count * 0.5:
                            cell_candidates.append((col, unique_count))
                
                if cell_candidates:
                    self.cell_line_col = max(cell_candidates, key=lambda x: x[1])[0]
                    print(f"Auto-detected cell line column: {self.cell_line_col}")
            
            if self.drug_col is None:
                drug_candidates = []
                for col in columns:
                    col_lower = col.lower()
                    if any(keyword in col_lower for keyword in ['drug', 'compound', 'agent']):
                        unique_count = df[col].nunique()
                        total_count = len(df)
                        if 5 <= unique_count <= total_count * 0.8: 
                            drug_candidates.append((col, unique_count))
                
                if drug_candidates:
                    self.drug_col = max(drug_candidates, key=lambda x: x[1])[0]
                    print(f"Auto-detected drug column: {self.drug_col}")
            
            self._analyzed = True
    
    def _create_groups(self, df: pd.DataFrame) -> np.ndarray:
        if self.cv_strategy == 'cell_line':
            if self.cell_line_col is None or self.cell_line_col not in df.columns:
                warnings.warn(f"Cell line column '{self.cell_line_col}' not found. Falling back to random CV.")
                return np.random.randint(0, 3, size=len(df))
            return df[self.cell_line_col].astype(str).values
        
        elif self.cv_strategy == 'drug':
            if self.drug_col is None or self.drug_col not in df.columns:
                warnings.warn(f"Drug column '{self.drug_col}' not found. Falling back to random CV.")
                return np.random.randint(0, 3, size=len(df))
            return df[self.drug_col].astype(str).values
        
        else:  # random
            return np.random.randint(0, 3, size=len(df))
    
    def get_fold_data(self, fold: int, feature_cols: str = '0:33', 
                     target_col: str = 'ZIP', batch_size: int = 100) -> Tuple:

        df = self._load_all_data()
        self._auto_detect_columns(df)
        
        if ':' in feature_cols:
            start, end = map(int, feature_cols.split(':'))
            feature_columns = list(range(start, end))
        else:
            feature_columns = feature_cols.split(',')
        
        groups = self._create_groups(df)
        
        if self.cv_strategy == 'random':
            train_mask = df['original_fold'] != fold
            test_mask = df['original_fold'] == fold
        else:
            unique_groups = np.unique(groups)
            n_splits = min(3, len(unique_groups))
            
            if n_splits < 3:
                warnings.warn(f"Only {n_splits} groups found. Using random splits.")
                train_mask = df['original_fold'] != fold
                test_mask = df['original_fold'] == fold
            else:
                gkf = GroupKFold(n_splits=n_splits)
                splits = list(gkf.split(df, groups=groups))
                train_idx, test_idx = splits[fold - 1]
                train_mask = df.index.isin(train_idx)
                test_mask = df.index.isin(test_idx)
        
        train_df = df[train_mask]
        test_df = df[test_mask]
        
        if isinstance(feature_columns[0], int):
            x_train = train_df.iloc[:, feature_columns].values
            x_test = test_df.iloc[:, feature_columns].values
        else:
            x_train = train_df[feature_columns].values
            x_test = test_df[feature_columns].values
        
        y_train = train_df[target_col].values.reshape(-1, 1)
        y_test = test_df[target_col].values.reshape(-1, 1)
        
        scaler = StandardScaler()
        y_train = scaler.fit_transform(y_train)
        y_test = scaler.transform(y_test)
        
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        x_train_t = torch.FloatTensor(x_train).to(device)
        y_train_t = torch.FloatTensor(y_train).to(device)
        x_test_t = torch.FloatTensor(x_test).to(device)
        y_test_t = torch.FloatTensor(y_test).to(device)
        
        train_dataset = TensorDataset(x_train_t, y_train_t)
        test_dataset = TensorDataset(x_test_t, y_test_t)
        train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
        test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)
        
        print(f"Fold {fold} ({self.cv_strategy} CV):")
        print(f"  Train: {len(train_df)} samples")
        print(f"  Test: {len(test_df)} samples")
        
        if self.cv_strategy != 'random':
            train_groups = len(np.unique(groups[train_mask]))
            test_groups = len(np.unique(groups[test_mask]))
            group_name = "cell lines" if self.cv_strategy == 'cell_line' else "drugs"
            print(f"  Train {group_name}: {train_groups}")
            print(f"  Test {group_name}: {test_groups}")
        
        return x_train, y_train, x_test, y_test, scaler, train_loader, test_loader
    
    def get_validation_data(self, fold: int, split: int, feature_cols: str = '0:33',
                           target_col: str = 'ZIP', batch_size: int = 100) -> Tuple:

        df = self._load_all_data()
        
        self._auto_detect_columns(df)
        
        if ':' in feature_cols:
            start, end = map(int, feature_cols.split(':'))
            feature_columns = list(range(start, end))
        else:
            feature_columns = feature_cols.split(',')
        
        if self.cv_strategy == 'random':
            training_df = df[df['original_fold'] != fold]
        else:
            groups = self._create_groups(df)
            unique_groups = np.unique(groups)
            n_splits = min(3, len(unique_groups))
            
            if n_splits < 3:
                training_df = df[df['original_fold'] != fold]
            else:
                gkf = GroupKFold(n_splits=n_splits)
                splits = list(gkf.split(df, groups=groups))
                train_idx, _ = splits[fold - 1]
                training_df = df.iloc[train_idx]
        
        training_groups = self._create_groups(training_df)
        unique_training_groups = np.unique(training_groups)
        n_inner_splits = min(3, len(unique_training_groups))
        
        if n_inner_splits < 3:

            indices = np.arange(len(training_df))
            np.random.shuffle(indices)
            split_size = len(indices) // 3
            if split == 1:
                val_idx = indices[:split_size]
                train_idx = indices[split_size:]
            elif split == 2:
                val_idx = indices[split_size:2*split_size]
                train_idx = np.concatenate([indices[:split_size], indices[2*split_size:]])
            else:
                val_idx = indices[2*split_size:]
                train_idx = indices[:2*split_size]
        else:
            inner_gkf = GroupKFold(n_splits=n_inner_splits)
            inner_splits = list(inner_gkf.split(training_df, groups=training_groups))
            train_idx, val_idx = inner_splits[split - 1]
        
        inner_train_df = training_df.iloc[train_idx]
        inner_val_df = training_df.iloc[val_idx]
        
        if isinstance(feature_columns[0], int):
            x_train = inner_train_df.iloc[:, feature_columns].values
            x_val = inner_val_df.iloc[:, feature_columns].values
        else:
            x_train = inner_train_df[feature_columns].values
            x_val = inner_val_df[feature_columns].values
        
        y_train = inner_train_df[target_col].values.reshape(-1, 1)
        y_val = inner_val_df[target_col].values.reshape(-1, 1)
        
        scaler = StandardScaler()
        y_train = scaler.fit_transform(y_train)
        y_val = scaler.transform(y_val)
        
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        x_train_t = torch.FloatTensor(x_train).to(device)
        y_train_t = torch.FloatTensor(y_train).to(device)
        x_val_t = torch.FloatTensor(x_val).to(device)
        y_val_t = torch.FloatTensor(y_val).to(device)
        
        train_dataset = TensorDataset(x_train_t, y_train_t)
        val_dataset = TensorDataset(x_val_t, y_val_t)
        train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
        val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
        
        return x_train, y_train, x_val, y_val, scaler, train_loader, val_loader


def load_data_unified(data_dir: str, fold: int, cv_strategy: str = 'random',
                     cell_line_col: str = None, drug_col: str = None,
                     feature_cols: str = '0:33', target: str = 'ZIP', batch: int = 100):

    cv = UnifiedCrossValidator(data_dir, cv_strategy, cell_line_col, drug_col)
    return cv.get_fold_data(fold, feature_cols, target, batch)


def load_valid_data_unified(data_dir: str, fold: int, split: int, cv_strategy: str = 'random',
                           cell_line_col: str = None, drug_col: str = None,
                           feature_cols: str = '0:33', target: str = 'ZIP', batch: int = 100):

    cv = UnifiedCrossValidator(data_dir, cv_strategy, cell_line_col, drug_col)
    return cv.get_validation_data(fold, split, feature_cols, target, batch)