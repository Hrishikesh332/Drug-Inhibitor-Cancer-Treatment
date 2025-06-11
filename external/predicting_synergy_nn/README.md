# Biomining folder structure 

```
predicting_synergy_nn/
├── data/
│   ├── fold1/
│   │   ├── fold1_alltrain.csv
│   │   ├── fold1_test.csv
│   │   └── validation/
│   │       ├── fold1_train1.csv
│   │       ├── fold1_valid1.csv
│   │       ├── fold1_train2.csv
│   │       ├── fold1_valid2.csv
│   │       ├── fold1_train3.csv
│   │       └── fold1_valid3.csv
│   ├── fold2/  # Similar structure to fold1
│   └── fold3/  # Similar structure to fold1
├── src/
│   ├── models/
│   │   ├── architectures.py  
│   │   └── metrics.py   
│   ├── utils/
│   │   ├── data_loader.py   
│   │   └── visualization.py
│   └── training/
│       ├── trainer.py        
│       └── hyperparameter.py
├── configs/
│   ├── base.yaml      
│   └── grid.yaml        
├── scripts/
│   ├── run_training.py       
│   └── run_grid_search.py    
└── outputs/                
    ├── models/               # Saved models
    ├── logs/                 
    ├── results/            
    └── grid_results/         
```
