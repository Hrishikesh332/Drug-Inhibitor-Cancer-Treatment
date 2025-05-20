# Baseline results 
Baselines are run against a stable validation set in **TranSynergy** (since we have a validation set — the same train, validation, and test sets as in the original code) and via inner cross-validation in **BioMining** (on outer fold 1).

## Process Overview 
We iterate over the parameter grid defined in the ```get_model``` method in ```baselines/traditional_ml.py```. A maximum of 15 iterations over combinations in the parameter grid is performed, with a timeout option of 60s, which means that in order for results to be valid, the model should fit the data in under a minute. 

To replicate baseline results, you need to install both `drug_combination` and `predicting_synergy_nn` projects, and run `pip install -r baselines/requirements.txt`. Then, run one of the following commands:
```
python -m baselines.main --mode transynergy --n_iter 15 --timeout 60
```
```
python -m baselines.main --mode biomining --n_iter 15 --timeout 60
```

If you want to only train one type of model, add the `--model` flag. Example:
```
-m baselines.main --mode transynergy --n_iter 100 --timeout 600 --model catboost
```
## Results

| Paper       | Model Name    |   Test Score |   Test Pearson Correlation |   Test Spearman Correlation | Test Params                                                                                                     |
|:------------|:--------------|-------------:|---------------------------:|----------------------------:|:----------------------------------------------------------------------------------------------------------------|
| transynergy | knn           |     520.798  |                   0.327595 |                    0.347093 | {'n_neighbors': 10}                                                                                             |
| transynergy | ridge         |     401.544  |                   0.489785 |                    0.533125 | {'alpha': 10.0}                                                                                                 |
| transynergy | decision_tree |     458.543  |                   0.37063  |                    0.407311 | {'min_samples_split': 2, 'min_samples_leaf': 10, 'max_features': None, 'max_depth': 5}                          |
| transynergy | svm | - | - | - | - |
| transynergy | random_forest |     324      |                   0.624637 |                    0.632665 | {'n_estimators': 100, 'min_samples_split': 2, 'min_samples_leaf': 2, 'max_features': 'log2', 'max_depth': None} |
| transynergy | catboost     |     262.073  |                   0.710145 |                    0.693605 | {'learning_rate': 0.06, 'l2_leaf_reg': 6, 'depth': 9, 'boosting_type': 'Plain'} |
| biomining   | knn           |      23.501  |                   0.416087 |                    0.416025 | {'n_neighbors': 10}                                                                                             |
| biomining   | ridge         |      26.2676 |                   0.27162  |                    0.289326 | {'alpha': 10.0}                                                                                                 |
| biomining   | decision_tree |      22.0338 |                   0.495622 |                    0.484003 | {'min_samples_split': 5, 'min_samples_leaf': 10, 'max_features': None, 'max_depth': 15}                         |
| biomining   | svm           |      23.1375 |                   0.440991 |                    0.461077 | {'kernel': 'rbf', 'gamma': 'auto', 'epsilon': 0.1, 'C': 10}                                                     |
| biomining   | random_forest |      17.8277 |                   0.609501 |                    0.58001  | {'n_estimators': 100, 'min_samples_split': 2, 'min_samples_leaf': 2, 'max_features': 'log2', 'max_depth': None} |
| biomining | catboost     |      16.1133 |                    0.65723 |                    0.640797 | {'learning_rate': 0.06, 'l2_leaf_reg': 4, 'depth': 9, 'boosting_type': 'Plain'} |


**To generate this table download Test Result tables from wandb, then process them https://colab.research.google.com/drive/1Wp1n3jo7sc9z7NIMgzSGsguRLHwk23Go?usp=sharing .**