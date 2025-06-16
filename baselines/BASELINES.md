# Baseline Results 

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

## Generate Markdown results table
To generate this table download Test Result tables from wandb, copy the resulting csv path, and run the following script:
```
python baselines/wandb_markdown_table.py --wandb_csv_path=<your path> --paper=<paper_name>
```
The script will print the markdown to console, which can be used to update the BASLINES.md file. 
Example:
```
python baselines/wandb_markdown_table.py --wandb_csv_path="./baselines/test_results.csv" --paper=transynergy
```