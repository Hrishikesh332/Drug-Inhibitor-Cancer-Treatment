import os
import argparse
import yaml
import subprocess
from datetime import datetime

def run_fold(fold, base_cfg):

    cfg = base_cfg.copy()
    cfg['fold'] = fold
    cfg['wb_run'] = f"fold{fold}_{cfg.get('arch', 'std')}_{datetime.now().strftime('%m%d_%H%M')}"
    
    temp_cfg = f"temp_f{fold}.yaml"
    with open(temp_cfg, 'w') as f:
        yaml.dump(cfg, f)
    
    cmd = f"python -m src.training.main --config {temp_cfg}"
    print(f"Running: {cmd}")
    subprocess.run(cmd, shell=True)
    
    #os.remove(temp_cfg)
    
    print(f"Completed fold {fold}")

def main():
    parser = argparse.ArgumentParser(description='Run training for multiple folds')
    parser.add_argument('--folds', type=str, default='1,2,3', help='Comma-separated folds')
    parser.add_argument('--config', type=str, default='configs/base.yaml', help='Base config')
    args = parser.parse_args()
    
    with open(args.config, 'r') as f:
        base_cfg = yaml.safe_load(f)
    
    folds = [int(f) for f in args.folds.split(',')]
    
    check_val = base_cfg['use_cross_val']
    if not check_val:
        for fold in folds:
            run_fold(fold, base_cfg)
    
    else:
        folds=1
        print(f"fold: {folds}")
        run_fold(folds, base_cfg)

    print("All folds completed!")

if __name__ == "__main__":
    main()