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
import numpy as np

from src.models.architectures import SynergyModel
from src.models.metrics import calc_pearson, calc_spearman
from src.utils.data_loader import load_valid_data
from src.utils.visualization import plot_grid_results

def train_eval(model, training_dataloader, validation_dataloader, learning_rate: float, max_epochs: int, device: str, early_stop: int = 50, log_wandb: bool = False):
    crit = nn.MSELoss()
    lr = float(learning_rate)
    opt = optim.Adam(model.parameters(), lr=lr)
    
    best_val = np.inf
    p_cnt = 0
    
    for ep in range(1, max_epochs + 1):
        model.train()
        train_loss = 0.0
        train_pearson = 0.0
        train_samples = 0
        
        for batch_idx, (x, y) in enumerate(training_dataloader):
            x, y = x.to(device), y.to(device)
            opt.zero_grad()
            out = model(x)
            loss = crit(out, y)
            pearson = calc_pearson(out, y)
            
            loss.backward()
            opt.step()
            
            train_loss += loss.item() * x.size(0)
            train_pearson += pearson.item() * x.size(0)
            train_samples += x.size(0)
        
        avg_train_loss = train_loss / train_samples
        avg_train_pearson = train_pearson / train_samples
        
        model.eval()
        val_loss = 0.0
        val_pearson = 0.0
        val_samples = 0
        all_pred = []
        all_true = []
        
        with torch.no_grad():
            for x, y in validation_dataloader:
                x, y = x.to(device), y.to(device)
                out = model(x)
                loss = crit(out, y)
                pearson = calc_pearson(out, y)
                
                val_loss += loss.item() * x.size(0)
                val_pearson += pearson.item() * x.size(0)
                val_samples += x.size(0)
                
                all_pred.append(out)
                all_true.append(y)
        
        all_pred = torch.cat(all_pred, dim=0)
        all_true = torch.cat(all_true, dim=0)
        val_spearman = calc_spearman(all_pred, all_true).item()

        avg_val_loss = val_loss / val_samples
        avg_val_pearson = val_pearson / val_samples

        if log_wandb and wandb.run is not None:
            wandb.log({
                'epoch': ep,
                'train_loss': avg_train_loss,
                'train_pearson': avg_train_pearson,
                'val_loss': avg_val_loss,
                'val_pearson': avg_val_pearson,
                'val_spearman': val_spearman,
                'learning_rate': lr
            })
        
        if val_loss < best_val:
            best_val = val_loss
            p_cnt = 0
        else:
            p_cnt += 1
        
        if p_cnt >= early_stop:
            break
        
        if ep % 20 == 0 or ep == 1:
            print(f"      Epoch {ep:3d}: TL={avg_train_loss:.4f}, TP={avg_train_pearson:.4f}, "
                  f"VL={avg_val_loss:.4f}, VP={avg_val_pearson:.4f}, VS={val_spearman:.4f}")
    
    return best_val, ep

def run_hyperparameter_search_all_folds(config_path):

    with open(config_path, 'r') as f:
        base_cfg = yaml.safe_load(f)
    
    folds = base_cfg['folds']
    splits = base_cfg['splits']
    
    if isinstance(folds, str):
        folds = [int(f.strip()) for f in folds.split(',')]
    if isinstance(splits, str):
        splits = [int(s.strip()) for s in splits.split(',')]
    
    print(f"Running hyperparameter search for")
    print(f"  Folds: {folds}")
    print(f"  Splits: {splits}")
    print(f"  Total combinations: {len(folds) * len(splits)}")
    
    all_results = []
    
    for fold in folds:
        for split in splits:
            print(f"\n{'='*60}")
            print(f"FOLD {fold}, SPLIT {split}")
            print(f"{'='*60}")
            
            fold_split_cfg = base_cfg.copy()
            fold_split_cfg['fold'] = fold
            fold_split_cfg['split'] = split
            
            try:
                result = run_grid_search(fold_split_cfg)
                if result:
                    result['fold'] = fold
                    result['split'] = split
                    all_results.append(result)
                    print(f"Completed fold {fold}, split {split}")
                else:
                    print(f"Failed fold {fold}, split {split}")
            except Exception as e:
                print(f"Error in fold {fold}, split {split}: {e}")
    
    print(f"\n{'='*60}")
    print("HYPERPARAMETER SEARCH SUMMARY")
    print(f"{'='*60}")
    print(f"Completed {len(all_results)} out of {len(folds) * len(splits)} fold/split combinations")
    
    if all_results:
        print("\nBest results by fold/split")
        for result in all_results:
            print(f"  Fold {result['fold']}, Split {result['split']}: "
                  f"Spearman={result['val_spearman']:.4f}, "
                  f"LR={result['lr']}, Arch={result['arch']}, Drop={result['drop']}")
        
        overall_best = max(all_results, key=lambda x: x['val_spearman'])
        print(f"\nBest Result Overview")
        print(f"  Fold {overall_best['fold']}, Split {overall_best['split']}")
        print(f"  Spearman: {overall_best['val_spearman']:.4f}")
        print(f"  LR: {overall_best['lr']}, Arch: {overall_best['arch']}, Drop: {overall_best['drop']}")
    
    return all_results

def run_grid_search(cfg):
    fold = cfg.get('fold', 1)
    split = cfg.get('split', 1)
    batch = cfg.get('batch', 100)
    data_dir = cfg.get('data_dir', 'data')
    out_dir = cfg.get('out_dir', 'outputs')
    use_wb = cfg.get('use_wb', True)
    wb_proj = cfg.get('wb_proj', 'synergy-grid')
    max_ep = cfg.get('max_ep', 500)
    patience = cfg.get('patience', 50)
    
    param_grid = {
        'lr': [float(lr) for lr in cfg.get('lr', [1e-5, 5e-5, 1e-4])],
        'arch': cfg.get('arch', ['std', 'fold2', 'fold3']),
        'drop': [float(drop) for drop in cfg.get('drop', [0.3, 0.5])],
    }
    
    print(f"Fold {fold}, Split {split} - {len(param_grid['lr']) * len(param_grid['arch']) * len(param_grid['drop'])} combinations")
    
    res_dir = os.path.join(out_dir, 'grid_results')
    os.makedirs(res_dir, exist_ok=True)
    
    _, _, _, _, _, tr_dl, vl_dl = load_valid_data(data_dir, fold, split, batch=batch)
    
    dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {dev}, Training: {len(tr_dl.dataset)}, Validation: {len(vl_dl.dataset)}")

    combos = list(itertools.product(
        param_grid['lr'],
        param_grid['arch'],
        param_grid['drop']
    ))
    
    results = []
    
    for i, (lr, arch, drop) in enumerate(combos):
        print(f"\nCombination {i+1}/{len(combos)}: LR={lr}, Arch={arch}, Drop={drop}")
        
        try:
            if use_wb:
                lr_str = f"{lr:.0e}" if lr < 1e-3 else f"{lr:.6f}".rstrip('0').rstrip('.')
                run_name = f"grid_f{fold}_s{split}_lr{lr_str}_a{arch}_d{drop}"
                
                wandb.init(
                    project=wb_proj, 
                    name=run_name,
                    config={
                        'fold': fold,
                        'split': split,
                        'lr': float(lr),
                        'arch': arch,
                        'drop': float(drop),
                        'batch_size': batch,
                        'max_epochs': max_ep
                    },
                    reinit=True
                )
            
            model = SynergyModel(in_dim=33, arch=arch, drop=float(drop)).to(dev)
            
            best_val, epochs = train_eval(
                model, tr_dl, vl_dl, lr, max_ep, dev, 
                early_stop=patience, log_wandb=use_wb
            )
            
            results.append({
                'lr': float(lr),
                'arch': arch,
                'drop': float(drop),
                'val_spearman': best_val,
                'epochs': epochs
            })
            
            print(f"  Result: Spearman={best_val:.4f}, Epochs={epochs}")
            
            del model
            torch.cuda.empty_cache() if dev.type == 'cuda' else None
            
        except Exception as e:
            print(f"  ERROR: {str(e)}")
        
        finally:
            if use_wb and wandb.run is not None:
                wandb.finish()
    
    successful_results = [r for r in results if r['val_spearman'] > 0]
    
    if not successful_results:
        print("No successful runs completed!")
        return None
    
    res_df = pd.DataFrame(results)
    res_file = os.path.join(res_dir, f"grid_f{fold}_s{split}.csv")
    res_df.to_csv(res_file, index=False)

    best_df = pd.DataFrame(successful_results)
    best_idx = best_df['val_spearman'].idxmax()
    best = best_df.iloc[best_idx].to_dict()
    
    print(f"Best: LR={best['lr']}, Arch={best['arch']}, Drop={best['drop']}, Spearman={best['val_spearman']:.4f}")
    
    best_file = os.path.join(res_dir, f"best_f{fold}_s{split}.yaml")
    with open(best_file, 'w') as f:
        yaml.dump({
            'fold': fold,
            'split': split,
            'lr': float(best['lr']),
            'arch': best['arch'],
            'drop': float(best['drop']),
            'val_spearman': float(best['val_spearman']),
            'epochs': int(best['epochs'])
        }, f)
    
    try:
        plot_grid_results(
            best_df, 
            x_col='lr', 
            y_col='val_spearman', 
            group_col='arch', 
            out_path=os.path.join(res_dir, f"grid_f{fold}_s{split}.png")
        )
    except Exception as e:
        print(f"Warning: Could not create visualization: {e}")
    
    return best

if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description='Run hyperparameter search')
    parser.add_argument('--config', type=str, required=True, help='Config YAML path')
    parser.add_argument('--folds', type=str, default=None, help='Comma-separated folds (e.g., 1,2,3)')
    parser.add_argument('--splits', type=str, default=None, help='Comma-separated splits (e.g., 1,2,3)')
    args = parser.parse_args()
    
    with open(args.config, 'r') as f:
        cfg = yaml.safe_load(f)
    
    if args.folds:
        cfg['folds'] = [int(f.strip()) for f in args.folds.split(',')]
    if args.splits:
        cfg['splits'] = [int(s.strip()) for s in args.splits.split(',')]
    
    if 'folds' in cfg or 'splits' in cfg or args.folds or args.splits:
        run_hyperparameter_search_all_folds(args.config)
    else:
        result = run_grid_search(cfg)
        if result is None:
            print("Hyperparameter search failed")
            exit(1)