import os
import argparse
import yaml
import subprocess
from itertools import product

def run_grid(fold, split, base_cfg):

    cfg = base_cfg.copy()
    cfg['fold'] = fold
    cfg['split'] = split
    
    temp_cfg = f"temp_grid_f{fold}_s{split}.yaml"
    with open(temp_cfg, 'w') as f:
        yaml.dump(cfg, f)
    
    cmd = f"python -m src.training.hyperparameter --config {temp_cfg}"
    print(f"Running: {cmd}")
    subprocess.run(cmd, shell=True)

    os.remove(temp_cfg)
    
    print(f"Completed grid search for fold {fold}, split {split}")

def main():
    parser = argparse.ArgumentParser(description='Run grid search for multiple folds/splits')
    parser.add_argument('--folds', type=str, default='1,2,3', help='Comma-separated folds')
    parser.add_argument('--splits', type=str, default='1,2,3', help='Comma-separated splits')
    parser.add_argument('--config', type=str, default='configs/grid.yaml', help='Base config')
    args = parser.parse_args()
    
    with open(args.config, 'r') as f:
        base_cfg = yaml.safe_load(f)
    
    folds = [int(f) for f in args.folds.split(',')]
    splits = [int(s) for s in args.splits.split(',')]
    
    for fold, split in product(folds, splits):
        run_grid(fold, split, base_cfg)
    
    print("All grid searches completed!")

if __name__ == "__main__":
    main()