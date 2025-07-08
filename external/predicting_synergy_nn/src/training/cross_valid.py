import os
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import GroupKFold
from tqdm import tqdm
from src.models.architectures import SynergyModel
from src.models.metrics import calc_pearson, calc_spearman


def load_cv_split(split, data_dir, feature_cols='0:33', target_col='ZIP', batch=100, cv_strategy='random', cell_line_col=None, drug_col=None):

    import pandas as pd
    train = None
    for i in range(1, 4):
        path = os.path.join(data_dir, 'fold2', 'validation', f'fold2_train{i}.csv')
        if split == i:
            valid = os.path.join(data_dir, 'fold2', 'validation', f'fold2_valid{i}.csv')
            train = pd.read_csv(path)
            valid = pd.read_csv(valid)
            break
    if train is None:
        raise ValueError('Invalid split for fold2')
    # Feature selection
    if ':' in feature_cols:
        start, end = map(int, feature_cols.split(':'))
        feature_columns = list(range(start, end))
    else:
        feature_columns = feature_cols.split(',')
    x_tr = train.iloc[:, feature_columns].values if isinstance(feature_columns[0], int) else train[feature_columns].values
    y_tr = train[target_col].values.reshape(-1, 1)
    x_val = valid.iloc[:, feature_columns].values if isinstance(feature_columns[0], int) else valid[feature_columns].values
    y_val = valid[target_col].values.reshape(-1, 1)

    if cv_strategy == 'cell_line':
        groups = train[cell_line_col].astype(str).values
    elif cv_strategy == 'drug':


        drugs = pd.concat([train['DRUG1'], train['DRUG2']]).unique()

        train_groups = train.apply(lambda row: tuple(sorted([row['DRUG1'], row['DRUG2']])), axis=1)
        groups = train_groups.astype(str).values
    else:
        groups = None

    scaler = StandardScaler()
    y_tr = scaler.fit_transform(y_tr)
    y_val = scaler.transform(y_val)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    x_tr_t = torch.FloatTensor(x_tr).to(device)
    y_tr_t = torch.FloatTensor(y_tr).to(device)
    x_val_t = torch.FloatTensor(x_val).to(device)
    y_val_t = torch.FloatTensor(y_val).to(device)
    tr_ds = TensorDataset(x_tr_t, y_tr_t)
    val_ds = TensorDataset(x_val_t, y_val_t)
    tr_dl = DataLoader(tr_ds, batch_size=batch, shuffle=True)
    val_dl = DataLoader(val_ds, batch_size=batch, shuffle=False)
    return tr_dl, val_dl, x_tr.shape[1]


def nested_cv(cfg):
    import pandas as pd
    import wandb
    fold = 2
    arch = str(cfg.get('arch', 'std'))
    batch = int(cfg.get('batch', 100))
    lr = float(cfg.get('lr', 1e-4))
    epochs = int(cfg.get('epochs', 200))
    drop = float(cfg.get('drop', 0.3))
    log_every = int(cfg.get('log_every', 10))
    use_wb = bool(cfg.get('use_wb', False))
    wb_proj = str(cfg.get('wb_proj', 'synergy'))
    data_dir = str(cfg.get('data_dir', 'data'))
    cv_strategy = str(cfg.get('cv_strategy', 'random'))
    cell_line_col = cfg.get('cell_line_col', None)
    drug_col = cfg.get('drug_col', None)
    feature_cols = cfg.get('feature_columns', '0:33')
    target_col = cfg.get('target_column', 'ZIP')
    spearman_scores = []
    for split in [1, 2, 3]:
        tr_dl, val_dl, in_dim = load_cv_split(split, data_dir, feature_cols, target_col, batch, cv_strategy, cell_line_col, drug_col)
        model = SynergyModel(in_dim=in_dim, arch=arch, drop=drop)
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        model = model.to(device)
        crit = nn.MSELoss()
        opt = optim.Adam(model.parameters(), lr=lr)
        best_val_spearman = -1.0
        best_epoch = 0
        if use_wb:
            run_name = f"fold2_split{split}_{arch}_{cv_strategy}"
            wandb.init(project=wb_proj, name=run_name, config={
                'lr': lr,
                'epochs': epochs,
                'batch': batch,
                'arch': arch,
                'fold': fold,
                'split': split,
                'drop': drop,
                'cv_strategy': cv_strategy
            })
        for epoch in tqdm(range(1, epochs + 1), desc=f"Split {split}"):

            model.train()
            train_loss, train_pear, train_pred, train_true = 0.0, 0.0, [], []
            for x, y in tr_dl:
                x, y = x.to(device), y.to(device)
                opt.zero_grad()
                out = model(x)
                loss = crit(out, y)
                loss.backward()
                opt.step()
                train_loss += loss.item() * x.size(0)
                train_pear += calc_pearson(out, y).item() * x.size(0)
                train_pred.append(out.detach())
                train_true.append(y.detach())
            train_loss /= len(tr_dl.dataset)
            train_pear /= len(tr_dl.dataset)
            train_pred = torch.cat(train_pred, dim=0)
            train_true = torch.cat(train_true, dim=0)
            train_spear = calc_spearman(train_pred, train_true).item()
            # Validate
            model.eval()
            val_loss, val_pear, val_pred, val_true = 0.0, 0.0, [], []
            with torch.no_grad():
                for x, y in val_dl:
                    x, y = x.to(device), y.to(device)
                    out = model(x)
                    loss = crit(out, y)
                    val_loss += loss.item() * x.size(0)
                    val_pear += calc_pearson(out, y).item() * x.size(0)
                    val_pred.append(out)
                    val_true.append(y)
            val_loss /= len(val_dl.dataset)
            val_pear /= len(val_dl.dataset)
            val_pred = torch.cat(val_pred, dim=0)
            val_true = torch.cat(val_true, dim=0)
            val_spearman = calc_spearman(val_pred, val_true).item()
            if val_spearman > best_val_spearman:
                best_val_spearman = val_spearman
                best_epoch = epoch
            if use_wb:
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
                print(f"Split {split} | Epoch {epoch}/{epochs} | "
                      f"Train Loss: {train_loss:.4f} | Train Pearson: {train_pear:.4f} | Train Spearman: {train_spear:.4f} | "
                      f"Val Loss: {val_loss:.4f} | Val Pearson: {val_pear:.4f} | Val Spearman: {val_spearman:.4f}")
        spearman_scores.append(best_val_spearman)
        print(f"Best Spearman for split {split}: {best_val_spearman:.4f} (epoch {best_epoch})")
        if use_wb:
            wandb.log({'best_val_spearman': best_val_spearman, 'best_epoch': best_epoch})
            wandb.finish()
    mean_spearman = float(np.mean(spearman_scores))
    print(f"Mean Spearman across splits: {mean_spearman:.4f}")
    if use_wb:
        wandb.init(project=wb_proj, name=f"fold2_summary_{arch}_{cv_strategy}")
        wandb.log({'mean_spearman': mean_spearman})
        wandb.finish()
    return mean_spearman