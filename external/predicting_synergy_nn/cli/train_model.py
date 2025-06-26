from training.trainer import train_model
import yaml
import argparse
import datetime

def train_final_model(cfg: dict) -> dict:
    cfg['fold'] = 'all'
    cfg['wb_run'] = f"final_model_{cfg.get('arch', 'std')}_{datetime.datetime.now().strftime('%m%d_%H%M')}"
    return train_model(cfg)


def main():
    parser = argparse.ArgumentParser(description='Run training for multiple folds')
    parser.add_argument('--config', type=str, default='configs/base.yaml', help='Base config')
    args = parser.parse_args()

    with open(args.config, 'r') as f:
        base_cfg = yaml.safe_load(f)

    results = train_final_model(base_cfg)
    results = {k: v for k, v in results.items() if k in ['test_loss', 'test_pearson', 'test_spearman']}
    print(f"Trained full model. Results: {results}")

if __name__ == "__main__":
    main()