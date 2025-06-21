import os
import yaml
import argparse
from src.training.cross_valid import nested_cv

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Run nested cross-validation')
    parser.add_argument('--config', type=str, default='configs/cv.yaml', 
                        help='Config file path')
    args = parser.parse_args()

    with open(args.config, 'r') as f:
        cfg = yaml.safe_load(f)

    mean_performance = nested_cv(cfg)
    
    print(f"Completed nested cross validation")
    print(f"Mean Spearman correlation: {mean_performance:.4f}")