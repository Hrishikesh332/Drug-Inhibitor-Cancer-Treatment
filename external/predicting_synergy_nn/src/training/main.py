from src.training.crossValidation import cross_validate
from src.training.trainer import train_model
import yaml
from src.utils.data_loader import load_data

if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description='Train synergy model')
    parser.add_argument('--config', type=str, required=True, help='Config YAML path')
    args = parser.parse_args()
    print(f"args.config: {args.config}")
    with open(args.config, 'r') as f:
        cfg = yaml.safe_load(f)
    

    check_val = cfg['use_cross_val']
    if not check_val:
        print(f"Inside check_val false")
        results = train_model(cfg,None,None,None,None)
    
    else:
        results = cross_validate(cfg)
        
    
    
    
    