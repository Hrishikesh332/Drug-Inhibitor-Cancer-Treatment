import argparse
import yaml
from itertools import product
from training.hyperparameter import run_grid_search

def run_grid(fold, split, base_cfg):

    cfg = base_cfg.copy()
    cfg['fold'] = fold
    cfg['split'] = split
    
    run_grid_search(cfg)

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