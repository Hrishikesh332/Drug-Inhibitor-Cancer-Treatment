import os
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import GroupKFold, KFold
from tqdm import tqdm
import warnings
warnings.filterwarnings('ignore')

import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from models.architectures import SynergyModel
from models.metrics import calc_pearson, calc_spearman


# Cross-validation data loader for fold2 only
class CrossValidationDataLoader:

    def __init__(self, data_dir, cv_strategy='random', feature_cols='0:33', 
                 target_col='ZIP', batch_size=100, random_state=42, 
                 cell_line_col=None, drug_col=None):
        self.data_dir = data_dir
        self.cv_strategy = cv_strategy
        self.feature_cols = feature_cols
        self.target_col = target_col
        self.batch_size = batch_size
        self.random_state = random_state
        self.cell_line_col = cell_line_col
        self.drug_col = drug_col
        self.scaler = StandardScaler()
        
        if ':' in feature_cols:
            start, end = map(int, feature_cols.split(':'))
            self.feature_columns = list(range(start, end))
        else:
            self.feature_columns = feature_cols.split(',')
    
    def load_fold2_data(self):

        fold2_dir = os.path.join(self.data_dir, 'fold2')
        train_path = os.path.join(fold2_dir, 'fold2_alltrain.csv')
        test_path = os.path.join(fold2_dir, 'fold2_test.csv')
        
        if not os.path.exists(train_path):
            raise FileNotFoundError(f"Fold2 training data not found: {train_path}")
        
        train_data = pd.read_csv(train_path)
        test_data = pd.read_csv(test_path) if os.path.exists(test_path) else None
        
        print(f"Loaded fold2 data: {len(train_data)} training samples")
        if test_data is not None:
            print(f"   Test samples: {len(test_data)}")
        
        return train_data, test_data
    
    def create_cv_splits(self, data, n_splits=3):

        if self.cv_strategy == 'random':
            kf = KFold(n_splits=n_splits, shuffle=True, random_state=self.random_state)
            splits = list(kf.split(data))
            print(f"Created {n_splits} random CV splits (random_state={self.random_state})")
            
        elif self.cv_strategy == 'cell_line':
            # Use configured cell_line_col or default to 'CELL_LINE'
            cell_line_column = self.cell_line_col if self.cell_line_col else 'CELL_LINE'
            
            if cell_line_column not in data.columns:
                raise ValueError(f"Cell line column '{cell_line_column}' not found in data")
            
            cell_lines = data[cell_line_column].unique()
            if len(cell_lines) < n_splits:
                raise ValueError(f"Not enough cell lines ({len(cell_lines)}) for {n_splits} splits")
            
            group_kf = GroupKFold(n_splits=n_splits)
            splits = list(group_kf.split(data, groups=data[cell_line_column]))
            print(f"Created {n_splits} cell_line_out CV splits using column '{cell_line_column}'")
            
            # Verify no overlap between train/val sets
            for i, (train_idx, val_idx) in enumerate(splits):
                train_cell_lines = set(data.iloc[train_idx][cell_line_column])
                val_cell_lines = set(data.iloc[val_idx][cell_line_column])
                overlap = train_cell_lines.intersection(val_cell_lines)
                if not overlap:
                    print(f"   Split {i+1}: No cell line overlap")
                else:
                    print(f"   Warning: Split {i+1} has {len(overlap)} overlapping cell lines")
            
        elif self.cv_strategy == 'drug':
            # Use configured drug columns or default to 'DRUG1' and 'DRUG2'
            drug1_col = 'DRUG1' if not self.drug_col else f"{self.drug_col}1"
            drug2_col = 'DRUG2' if not self.drug_col else f"{self.drug_col}2"
            
            if drug1_col not in data.columns or drug2_col not in data.columns:
                raise ValueError(f"Drug columns '{drug1_col}' or '{drug2_col}' not found in data")
            
            # Create unique drug combination identifiers 
            data['DRUG_COMBO'] = data.apply(
                lambda row: tuple(sorted([str(row[drug1_col]).strip(), str(row[drug2_col]).strip()])), axis=1
            )
            
            unique_combos = data['DRUG_COMBO'].nunique()
            total_samples = len(data)
            print(f"   Drug combinations: {unique_combos} unique combinations from {total_samples} samples")
            print(f"   Using columns: '{drug1_col}' and '{drug2_col}'")
            
            group_kf = GroupKFold(n_splits=n_splits)
            splits = list(group_kf.split(data, groups=data['DRUG_COMBO']))
            print(f"Created {n_splits} drug_out CV splits")
            
            # Verify no overlap between train/val sets
            for i, (train_idx, val_idx) in enumerate(splits):
                train_combos = set(data.iloc[train_idx]['DRUG_COMBO'])
                val_combos = set(data.iloc[val_idx]['DRUG_COMBO'])
                overlap = train_combos.intersection(val_combos)
                if not overlap:
                    print(f"   Split {i+1}: No drug combination overlap")
                else:
                    print(f"   Warning: Split {i+1} has {len(overlap)} overlapping combinations")
            
        else:
            raise ValueError(f"Unknown cv_strategy: {self.cv_strategy}")
        
        return splits
    
    def prepare_data_split(self, data, train_idx, val_idx):

        train_data = data.iloc[train_idx]
        val_data = data.iloc[val_idx]
        
        x_train = train_data.iloc[:, self.feature_columns].values
        y_train = train_data[self.target_col].values.reshape(-1, 1)
        x_val = val_data.iloc[:, self.feature_columns].values
        y_val = val_data[self.target_col].values.reshape(-1, 1)
        
        y_train_scaled = self.scaler.fit_transform(y_train)
        y_val_scaled = self.scaler.transform(y_val)
        
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        x_train_t = torch.FloatTensor(x_train).to(device)
        y_train_t = torch.FloatTensor(y_train_scaled).to(device)
        x_val_t = torch.FloatTensor(x_val).to(device)
        y_val_t = torch.FloatTensor(y_val_scaled).to(device)
        
        train_dataset = TensorDataset(x_train_t, y_train_t)
        val_dataset = TensorDataset(x_val_t, y_val_t)
        
        train_loader = DataLoader(train_dataset, batch_size=self.batch_size, shuffle=True)
        val_loader = DataLoader(val_dataset, batch_size=self.batch_size, shuffle=False)
        
        return train_loader, val_loader, x_train.shape[1], len(train_data), len(val_data)


def train_and_validate(model, train_loader, val_loader, criterion, optimizer, 
                      device, epochs, log_every=10, use_wandb=False, run_name=""):

    best_val_spearman = -1.0
    best_epoch = 0
    best_model_state = None
    
    for epoch in tqdm(range(1, epochs + 1), desc=f"Training {run_name}"):
        model.train()
        train_loss, train_pear, train_pred, train_true = 0.0, 0.0, [], []
        
        for x, y in train_loader:
            x, y = x.to(device), y.to(device)
            optimizer.zero_grad()
            out = model(x)
            loss = criterion(out, y)
            loss.backward()
            optimizer.step()
            
            train_loss += loss.item() * x.size(0)
            train_pear += calc_pearson(out, y).item() * x.size(0)
            train_pred.append(out.detach())
            train_true.append(y.detach())
        
        train_loss /= len(train_loader.dataset)
        train_pear /= len(train_loader.dataset)
        train_pred = torch.cat(train_pred, dim=0)
        train_true = torch.cat(train_true, dim=0)
        train_spear = calc_spearman(train_pred, train_true).item()
        
        model.eval()
        val_loss, val_pear, val_pred, val_true = 0.0, 0.0, [], []
        
        with torch.no_grad():
            for x, y in val_loader:
                x, y = x.to(device), y.to(device)
                out = model(x)
                loss = criterion(out, y)
                val_loss += loss.item() * x.size(0)
                val_pear += calc_pearson(out, y).item() * x.size(0)
                val_pred.append(out)
                val_true.append(y)
        
        val_loss /= len(val_loader.dataset)
        val_pear /= len(val_loader.dataset)
        val_pred = torch.cat(val_pred, dim=0)
        val_true = torch.cat(val_true, dim=0)
        val_spearman = calc_spearman(val_pred, val_true).item()
        
        if val_spearman > best_val_spearman:
            best_val_spearman = val_spearman
            best_epoch = epoch
            best_model_state = model.state_dict().copy()
        
        if use_wandb:
            import wandb
            wandb.log({
                'epoch': epoch,
                'train_loss': train_loss,
                'train_pearson': train_pear,
                'train_spearman': train_spear,
                'val_loss': val_loss,
                'val_pearson': val_pear,
                'val_spearman': val_spearman
            })
        
        if epoch % log_every == 0 or epoch == 1 or epoch == epochs:
            print(f"{run_name} | Epoch {epoch}/{epochs} | "
                  f"Train Loss: {train_loss:.4f} | Train Spearman: {train_spear:.4f} | "
                  f"Val Loss: {val_loss:.4f} | Val Spearman: {val_spearman:.4f}")
    
    return best_val_spearman, best_epoch, best_model_state


def run_cross_validation(cfg):

    arch = str(cfg.get('arch', 'std'))
    batch_size = int(cfg.get('batch', 100))
    lr = float(cfg.get('lr', 1e-4))
    epochs = int(cfg.get('epochs', 200))
    drop = float(cfg.get('drop', 0.3))
    log_every = int(cfg.get('log_every', 10))
    use_wandb = bool(cfg.get('use_wb', False))
    wb_proj = str(cfg.get('wb_proj', 'synergy'))
    data_dir = str(cfg.get('data_dir', 'data'))
    cv_strategy = str(cfg.get('cv_strategy', 'random'))
    cell_line_col = cfg.get('cell_line_col', None)
    drug_col = cfg.get('drug_col', None)
    feature_cols = cfg.get('feature_columns', '0:33')
    target_col = cfg.get('target_column', 'ZIP')
    n_cv_splits = int(cfg.get('n_cv_splits', 3))
    random_state = int(cfg.get('random_state', 42))
    
    print(f"Cross-Validation Configuration")
    print(f"Strategy: {cv_strategy}")
    print(f"CV Splits: {n_cv_splits}")
    print(f"Architecture: {arch}")
    print(f"Batch size: {batch_size}")
    print(f"Learning rate: {lr}")
    print(f"Epochs: {epochs}")
    print(f"Feature columns: {feature_cols}")
    print(f"Target: {target_col}")
    print(f"Random state: {random_state}")
    if cell_line_col:
        print(f"Cell line column: {cell_line_col}")
    if drug_col:
        print(f"Drug column: {drug_col}")
    
    data_loader = CrossValidationDataLoader(
        data_dir=data_dir,
        cv_strategy=cv_strategy,
        feature_cols=feature_cols,
        target_col=target_col,
        batch_size=batch_size,
        random_state=random_state,
        cell_line_col=cell_line_col,
        drug_col=drug_col
    )
    
    try:
        train_data, test_data = data_loader.load_fold2_data()
        cv_splits = data_loader.create_cv_splits(train_data, n_cv_splits)
        
        all_results = []
        
        for split_idx, (train_idx, val_idx) in enumerate(cv_splits, 1):
            print(f"\n--- Fold2, Split {split_idx} ---")
            
            train_loader, val_loader, input_dim, n_train, n_val = data_loader.prepare_data_split(
                train_data, train_idx, val_idx
            )
            
            print(f"Training samples: {n_train}, Validation samples: {n_val}")
            
            device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
            model = SynergyModel(in_dim=input_dim, arch=arch, drop=drop).to(device)
            
            criterion = nn.MSELoss()
            optimizer = optim.Adam(model.parameters(), lr=lr)
            
            if use_wandb:
                import wandb
                run_name = f"fold2_split{split_idx}_{arch}_{cv_strategy}"
                wandb.init(
                    project=wb_proj, 
                    name=run_name, 
                    config={
                        'fold': 2,
                        'split': split_idx,
                        'arch': arch,
                        'cv_strategy': cv_strategy,
                        'lr': lr,
                        'epochs': epochs,
                        'batch_size': batch_size,
                        'drop': drop,
                        'n_train': n_train,
                        'n_val': n_val
                    }
                )
            
            best_spearman, best_epoch, best_model_state = train_and_validate(
                model, train_loader, val_loader, criterion, optimizer, device,
                epochs, log_every, use_wandb, f"Fold2_Split{split_idx}"
            )
            
            split_result = {
                'fold': 2,
                'split': split_idx,
                'best_spearman': best_spearman,
                'best_epoch': best_epoch,
                'n_train': n_train,
                'n_val': n_val,
                'cv_strategy': cv_strategy
            }
            
            all_results.append(split_result)
            print(f"Best validation Spearman: {best_spearman:.4f} (epoch {best_epoch})")
            
            if use_wandb:
                wandb.log({'best_val_spearman': best_spearman, 'best_epoch': best_epoch})
                wandb.finish()
        
        if all_results:
            spearman_scores = [r['best_spearman'] for r in all_results]
            overall_mean = np.mean(spearman_scores)
            overall_std = np.std(spearman_scores)
            
            print(f"\n=== FOLD2 CROSS-VALIDATION RESULTS ===")
            print(f"Total splits completed: {len(all_results)}")
            print(f"Overall Mean Spearman: {overall_mean:.4f} ± {overall_std:.4f}")
            print(f"CV Strategy: {cv_strategy}")
            print(f"Individual scores: {[f'{s:.4f}' for s in spearman_scores]}")
            
            if use_wandb:
                import wandb
                wandb.init(project=wb_proj, name=f"fold2_cv_summary_{arch}_{cv_strategy}")
                wandb.log({
                    'overall_mean_spearman': overall_mean,
                    'overall_std_spearman': overall_std,
                    'total_splits': len(all_results),
                    'cv_strategy': cv_strategy,
                    'fold': 2
                })
                wandb.finish()
            
            return overall_mean, overall_std, all_results
        else:
            print("No successful cross-validation runs completed.")
            return None, None, []
            
    except Exception as e:
        print(f"Error during cross-validation: {str(e)}")
        import traceback
        traceback.print_exc()
        return None, None, []


if __name__ == "__main__":
    import argparse
    import yaml
    
    parser = argparse.ArgumentParser(description='Run cross-validation for Biomining')
    parser.add_argument('--strategy', choices=['random', 'cell_line', 'drug'], 
                       default='random', help='CV strategy (default: random)')
    parser.add_argument('--splits', type=int, default=3, 
                       help='Number of CV splits (default: 3)')
    parser.add_argument('--config', default='configs/cv.yaml', 
                       help='Path to config file (default: configs/cv.yaml)')
    
    args = parser.parse_args()
    
    try:

        with open(args.config, 'r') as f:
            config = yaml.safe_load(f)
        
        # Update strategy and splits
        config['cv_strategy'] = args.strategy
        config['n_cv_splits'] = args.splits

        if args.strategy == 'cell_line':
            config['cell_line_col'] = 'CELL_LINE'
            config['drug_col'] = None
        elif args.strategy == 'drug':
            config['cell_line_col'] = None
            config['drug_col'] = None
        else:  # random
            config['cell_line_col'] = None
            config['drug_col'] = None
        
        print(f"Running {args.strategy} cross-validation with {args.splits} splits...")
        
        mean_score, std_score, results = run_cross_validation(config)
        
        if mean_score is not None:
            print(f"\nCross-validation completed!")
            print(f"Strategy: {args.strategy}")
            print(f"Results: {mean_score:.4f} ± {std_score:.4f}")
        else:
            print("Cross-validation failed")
            
    except FileNotFoundError:
        print(f"Config file not found: {args.config}")
        print("Make sure you're running from the correct directory")
    except Exception as e:
        print(f"Error: {str(e)}")