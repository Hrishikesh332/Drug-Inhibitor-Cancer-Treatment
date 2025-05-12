from external.drug_combination.trans_synergy.models.trans_synergy.attention_main import setup_data, prepare_splitted_datasets
from external.drug_combination.trans_synergy.data.trans_synergy_data import DataPreprocessor
from external.drug_combination.trans_synergy.models.trans_synergy.attention_main import set_seed
from baselines.traditional_ml import run_model
from tqdm import tqdm


def run_transynergy():
    
    baseline_models = ['random_forest', 'svm', 'decision_tree']

    _, X, Y, _, _ = setup_data()

    split_func = DataPreprocessor.reg_train_eval_test_split
    for fold_idx, partition in enumerate(tqdm(split_func(fold='fold', test_fold=4), desc="Folds", total=1)):
        partition_indices = {
            'train': partition[0],
            'test1': partition[1],
            'test2': partition[2],
            'eval1': partition[3],
            'eval2': partition[4]
        }

        # Dataloaders
        training_set, _, validation_set, test_set,  _ = prepare_splitted_datasets(partition_indices, Y.reshape(-1), X)
        X_train, y_train = training_set.data_cache, training_set.data_cache_y
        X_val, y_val = validation_set.data_cache, validation_set.data_cache_y
        X_test, y_test = test_set.data_cache, test_set.data_cache_y
        
        for baseline_model in baseline_models:
            run_model(
                X_train,
                y_train,
                X_val=X_val,
                y_val=y_val,
                X_test=X_test,
                y_test=y_test,
                fold_idx=fold_idx,
                model_name=baseline_model,
                paper = "transynergy",
            )   


def run_biomining():
    pass

if __name__ == "__main__":
    set_seed(42)
    run_transynergy()