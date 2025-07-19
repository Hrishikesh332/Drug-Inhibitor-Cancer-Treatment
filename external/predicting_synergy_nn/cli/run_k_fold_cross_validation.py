import argparse
import statistics

import yaml
from datetime import datetime

from training.trainer import train_model

def train_model_on_fold(fold, base_cfg) -> dict:

    cfg = base_cfg.copy()
    cfg['fold'] = fold
    cfg['wb_run'] = f"fold{fold}_{cfg.get('arch', 'std')}_{datetime.now().strftime('%m%d_%H%M')}"

    return train_model(cfg)


def main():
    parser = argparse.ArgumentParser(description='Run training for multiple folds')
    parser.add_argument('--folds', type=str, default='1,2,3', help='Comma-separated folds')
    parser.add_argument('--config', type=str, default='configs/base.yaml', help='Base config')
    args = parser.parse_args()
    
    with open(args.config, 'r') as f:
        base_cfg = yaml.safe_load(f)
    
    folds = [int(f) for f in args.folds.split(',')]

    per_fold_results: list[dict] = []

    for fold in folds:
        results = train_model_on_fold(fold, base_cfg)
        results = {k: v for k, v in results.items() if k in ['test_loss', 'test_pearson', 'test_spearman']}
        per_fold_results.append(results)
        print(f"Fold {fold} results: {results}")
    
    print("All folds completed!")

    # Compute means
    mean_results = {}
    for key in ['test_loss', 'test_pearson', 'test_spearman']:
        values = [res[key] for res in per_fold_results if key in res]
        mean_results[key] = statistics.mean(values)

    # Print means
    print("\nMean results across all folds:")
    for k, v in mean_results.items():
        print(f"{k}: {v:.4f}")


if __name__ == "__main__":
    main()