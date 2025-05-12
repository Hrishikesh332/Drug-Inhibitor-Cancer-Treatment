from src.training.crossValidation import cross_validate
import yaml

if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description='Train synergy model')
    parser.add_argument('--config', type=str, required=True, help='Config YAML path')
    args = parser.parse_args()
    
    with open(args.config, 'r') as f:
        cfg = yaml.safe_load(f)
    
    #X, Y, _, _, _, _, _ = load_data(cfg['data_dir'], fold=1, batch=cfg['batch'])
    
    results = cross_validate(cfg)
    