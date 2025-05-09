import os
import yaml
import itertools
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from tqdm import tqdm
import wandb

from src.models.architectures import SynergyModel
from src.models.metrics import calc_pearson, calc_spearman
from src.utils.data_loader import load_valid_data
from src.utils.visualization import plot_grid_results

def train_eval(model, tr_dl, vl_dl, lr, max_ep, dev, early_stop=50):
    crit = nn.MSELoss()
    opt = optim.Adam(model.parameters(), lr=lr)
    
    best_val = -1.0
    p_cnt = 0
    
    for ep in range(1, max_ep + 1):

        model.train()
        for x, y in tr_dl:
            opt.zero_grad()
            out = model(x)
            loss = crit(out, y)
            loss.backward()
            opt.step()
        
        model.eval()
        all_pred = []
        all_true = []
        
        with torch.no_grad():
            for x, y in vl_dl:
                out = model(x)
                all_pred.append(out)
                all_true.append(y)
        
        all_pred = torch.cat(all_pred, dim=0)
        all_true = torch.cat(all_true, dim=0)
        
        val_spear = calc_spearman(all_pred, all_true).item()
        
        if val_spear > best_val:
            best_val = val_spear
            p_cnt = 0
        else:
            p_cnt += 1
        
        if p_cnt >= early_stop:
            break
        
        if ep % 20 == 0:
            print(f"  Epoch {ep}/{max_ep} - Val Spearman: {val_spear:.4f}, Best: {best_val:.4f}")
    
    return best_val, ep

def run_grid_search(cfg):
    fold = cfg.get('fold', 1)
    split = cfg.get('split', 1)
    batch = cfg.get('batch', 100)
    data_dir = cfg.get('data_dir', 'data')
    out_dir = cfg.get('out_dir', 'outputs')
    use_wb = cfg.get('use_wb', True)
    wb_proj = cfg.get('wb_proj', 'synergy-grid')
    max_ep = cfg.get('max_ep', 500)
    
    param_grid = {
        'lr': cfg.get('lr', [1e-5, 5e-5, 1e-4]),
        'arch': cfg.get('arch', ['std', 'fold2', 'fold3']),
        'drop': cfg.get('drop', [0.3, 0.5]),
    }
    
    res_dir = os.path.join(out_dir, 'grid_results')
    os.makedirs(res_dir, exist_ok=True)
    
    _, _, _, _, _, tr_dl, vl_dl = load_valid_data(data_dir, fold, split, batch=batch)
    
    dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using: {dev}")
    
    combos = list(itertools.product(
        param_grid['lr'],
        param_grid['arch'],
        param_grid['drop']
    ))
    
    print(f"Running {len(combos)} combinations")
    
    results = []
    
    for i, (lr, arch, drop) in enumerate(combos):
        print(f"\nTrying combo {i+1}/{len(combos)}:")
        print(f"  LR: {lr}, Arch: {arch}, Drop: {drop}")
        
        if use_wb:
            run_name = f"grid_f{fold}_s{split}_lr{lr}_a{arch}_d{drop}"
            wandb.init(project=wb_proj, name=run_name, 
                      config={
                          'lr': lr,
                          'arch': arch,
                          'drop': drop,
                          'fold': fold,
                          'split': split
                      },
                      reinit=True)
        
        model = SynergyModel(in_dim=33, arch=arch, drop=drop).to(dev)
        
        best_val, epochs = train_eval(model, tr_dl, vl_dl, lr, max_ep, dev)
        
        if use_wb:
            wandb.log({
                'best_val_spearman': best_val,
                'epochs_trained': epochs
            })
            wandb.finish()
        
        results.append({
            'lr': lr,
            'arch': arch,
            'drop': drop,
            'val_spearman': best_val,
            'epochs': epochs
        })
    
    res_df = pd.DataFrame(results)
    
    res_file = os.path.join(res_dir, f"grid_f{fold}_s{split}.csv")
    res_df.to_csv(res_file, index=False)
    

    best_idx = res_df['val_spearman'].idxmax()
    best = res_df.iloc[best_idx].to_dict()
    
    print("\nGrid search done!")
    print(f"Best params for fold {fold}, split {split}:")
    print(f"  LR: {best['lr']}")
    print(f"  Arch: {best['arch']}")
    print(f"  Drop: {best['drop']}")
    print(f"  Val Spearman: {best['val_spearman']:.4f}")
    
    best_file = os.path.join(res_dir, f"best_f{fold}_s{split}.yaml")
    with open(best_file, 'w') as f:
        yaml.dump({
            'fold': fold,
            'lr': float(best['lr']),
            'arch': best['arch'],
            'drop': float(best['drop']),
            'val_spearman': float(best['val_spearman'])
        }, f)
    
    plot_grid_results(
        res_df, 
        x_col='lr', 
        y_col='val_spearman', 
        group_col='arch', 
        out_path=os.path.join(res_dir, f"grid_f{fold}_s{split}.png")
    )
    
    return best

if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description='Run grid search')
    parser.add_argument('--config', type=str, required=True, help='Config YAML path')
    args = parser.parse_args()
    
    with open(args.config, 'r') as f:
        cfg = yaml.safe_load(f)
    
    run_grid_search(cfg)