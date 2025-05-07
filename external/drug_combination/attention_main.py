#!/usr/bin/env python
import concurrent.futures
import pickle
import random
from os import environ, mkdir, path

import numpy as np
import pandas as pd
import shap
import torch
import torch.nn.functional as F
import wandb
from scipy.stats import pearsonr, spearmanr
from sklearn.cluster import MiniBatchKMeans
from sklearn.metrics import mean_squared_error
from sklearn.preprocessing import StandardScaler
from torch import load, save
from torch.utils import data
from tqdm import tqdm

import trans_synergy.neural_finger_print.data_utils as data_utils
from trans_synergy import (attention_model, device2, drug_drug, logger,
                           my_data, setting)

executor = concurrent.futures.ThreadPoolExecutor(max_workers=2)

random_seed = 913

def set_seed(seed=random_seed):
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)

def get_final_index():

    ## get the index of synergy score database
    if not setting.update_final_index and path.exists(setting.final_index):
        final_index = pd.read_csv(setting.final_index, header=None)[0]
    else:
        final_index = my_data.SynergyDataReader.get_final_index()
    return final_index

def prepare_data():

    if not setting.update_xy:
        assert (path.exists(setting.old_x) and path.exists(setting.old_y)), "Data need to be downloaded from zenodo follow instruction in README"
        X = np.load(setting.old_x)
        with open(setting.old_x_lengths, 'rb') as old_x_lengths:
            drug_features_length, cellline_features_length = pickle.load(old_x_lengths)
        with open(setting.old_y, 'rb') as old_y:
            Y = pickle.load(old_y)
    else:
        X, drug_features_length, cellline_features_length = \
            my_data.SamplesDataLoader.Raw_X_features_prep(methods='flexible_attn')
        np.save(setting.old_x, X)
        with open(setting.old_x_lengths, 'wb+') as old_x_lengths:
            pickle.dump((drug_features_length,cellline_features_length), old_x_lengths)

        Y = my_data.SamplesDataLoader.Y_features_prep()
        with open(setting.old_y, 'wb+') as old_y:
            pickle.dump(Y, old_y)
    return X, Y, drug_features_length, cellline_features_length


def prepare_model(reorder_tensor, entrez_set):

    ### prepare two models
    ### drug_model: the one used for training
    ### best_drug_mode;: the one used for same the best model

    final_mask = None
    drug_model = attention_model.get_multi_models(reorder_tensor.get_reordered_slice_indices(), input_masks=final_mask,
                                                  drugs_on_the_side=False)
    best_drug_model = attention_model.get_multi_models(reorder_tensor.get_reordered_slice_indices(),
                                                       input_masks=final_mask, drugs_on_the_side=False)
    for n, m in drug_model.named_modules():
        if n == "out":
            m.register_forward_hook(drug_drug.input_hook)
    for best_n, best_m in best_drug_model.named_modules():
        if best_n == "out":
            best_m.register_forward_hook(drug_drug.input_hook)
    drug_model = drug_model.to(device2)
    best_drug_model = best_drug_model.to(device2)
    
    return drug_model, best_drug_model

def prepare_splitted_dataset(partition, labels, loaded_data):
    """
    Prepare train, validation, test, and evaluation data generators.
    """

    logger.debug("Preparing datasets ... ")

    # Training dataset and generator
    training_set = my_data.MyDataset(partition['train'], labels, loaded_data)
    train_params = {'batch_size': setting.batch_size, 'shuffle': True}
    training_generator = data.DataLoader(training_set, **train_params)

    # Evaluation training dataset and generator (train + eval1 + eval2)
    eval_train_set = my_data.MyDataset(np.concatenate([partition['train'], partition['eval1'], partition['eval2']]), labels, loaded_data)
    training_index_list = np.concatenate([partition['train'], partition['eval1'], partition['eval2']])
    logger.debug(f"Training data length: {len(training_index_list)}")
    eval_train_params = {'batch_size': setting.batch_size, 'shuffle': False}
    eval_train_generator = data.DataLoader(eval_train_set, **eval_train_params)

    # Validation dataset and generator (eval1)
    validation_set = my_data.MyDataset(partition['eval1'], labels, loaded_data)
    eval_params = {'batch_size': len(partition['test1']) // 4, 'shuffle': False}
    validation_generator = data.DataLoader(validation_set, **eval_params)

    # Test dataset and generator (test1)
    test_set = my_data.MyDataset(partition['test1'], labels, loaded_data)
    test_index_list = partition['test1']
    logger.debug(f"Test data length: {len(test_index_list)}")
    pickle.dump(test_index_list, open("test_index_list", "wb+"))
    test_params = {'batch_size': len(test_index_list) // 4, 'shuffle': False}
    test_generator = data.DataLoader(test_set, **test_params)

    # All dataset and generator (train + eval1 + test1)
    all_index_list = np.concatenate([partition['train'][:len(partition['train']) // 2], partition['eval1'], partition['test1']])
    all_set = my_data.MyDataset(all_index_list, labels, loaded_data)
    logger.debug(f"All data length: {len(set(all_index_list))}")
    pickle.dump(all_index_list, open("all_index_list", "wb+"))
    all_set_params = {'batch_size': len(all_index_list) // 8, 'shuffle': False}
    all_data_generator = data.DataLoader(all_set, **all_set_params)

    # Total all data generator (using the full batch size)
    all_set_params_total = {'batch_size': len(all_index_list), 'shuffle': False}
    all_data_generator_total = data.DataLoader(all_set, **all_set_params_total)

    return (training_generator, eval_train_generator, validation_generator,
            test_generator, all_data_generator, all_data_generator_total)

def run():

    ## get genes
    entrez_set = my_data.GenesDataReader.get_gene_entrez_set()

    std_scaler = StandardScaler()
    logger.debug("Getting features and synergy scores ...")

    X, Y, drug_features_length, cellline_features_length = prepare_data()

    logger.debug("Preparing models")
    slice_indices = drug_features_length + drug_features_length + cellline_features_length
    reorder_tensor = drug_drug.reorganize_tensor(slice_indices, setting.arrangement, 2)
    logger.debug("the layout of all features is {!r}".format(reorder_tensor.get_reordered_slice_indices()))
    set_seed()
    drug_model, best_drug_model = prepare_model(reorder_tensor, entrez_set)

    optimizer = torch.optim.Adam(drug_model.parameters(), lr=setting.start_lr, weight_decay=setting.lr_decay,
                                 betas=(0.9, 0.98), eps=1e-9)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=100, eta_min = 1e-7)

    # define variables used in testing and shap analysis
    test_generator = None
    all_data_generator_total = None
    all_data_generator = None
    test_index_list = None
    best_cv_pearson_score = 0
    partition = None

    split_func = my_data.DataPreprocessor.reg_train_eval_test_split

        
    for fold_idx, (train_index, test_index, test_index_2, evaluation_index, evaluation_index_2) in enumerate(
            tqdm(split_func(fold='fold', test_fold=4), desc="Folds", total=1) # the original code calls for this only
        ):
        
        if USE_wandb:
            wandb.init(project=f"Drug combination alpha_fold_{fold_idx}",
                    name=setting.run_dir.rsplit('/', 1)[1] + '_' + setting.data_specific[:15] + '_' + str(random_seed),  # TODO: make cross platform

                    notes=setting.data_specific)
            wandb.define_metric("Train Loss", step_metric="Epoch")
            wandb.define_metric("Validation Loss", step_metric="Epoch")
        # local_X = X[np.concatenate((train_index, test_index, test_index_2, evaluation_index, evaluation_index_2))]
       
        std_scaler.fit(Y[train_index])
        if setting.y_transform:
            Y = std_scaler.transform(Y) * 100

        partition_intigers = {
                    'train': train_index,
                    'test1': test_index, 
                    'test2': test_index_2,
                    'eval1': evaluation_index,
                    'eval2': evaluation_index_2}

        training_generator, eval_train_generator, validation_generator, test_generator, \
        all_data_generator, all_data_generator_total = prepare_splitted_dataset(partition_intigers, Y.reshape(-1), X)
        test_index_list = partition_intigers['test1']

        logger.debug("Start training")
        set_seed()

        for epoch in tqdm(range(setting.n_epochs), desc="Training Epochs"):

            drug_model.to(device2)  # Ensure the model is on the correct device
            train_total_loss = 0
            all_preds = []
            all_ys = []

            training_iter = tqdm(training_generator, desc=f"Epoch {epoch+1}", leave=False)

            for cur_local_batch, cur_local_labels in training_iter:
                
                drug_model.train()

                local_batch, local_labels = cur_local_batch.float().to(device2), cur_local_labels.float().to(device2)
                local_batch = local_batch.contiguous().view(-1, 1, sum(slice_indices) + setting.single_repsonse_feature_length)
                reorder_tensor.load_raw_tensor(local_batch)
                local_batch = reorder_tensor.get_reordered_narrow_tensor()
                preds = drug_model(*local_batch)
                preds = preds.contiguous().view(-1)
                ys = local_labels.contiguous().view(-1)
                optimizer.zero_grad()
                assert preds.size(-1) == ys.size(-1)
                loss = F.mse_loss(preds, ys)
                loss.backward()
                # added by nina
                # torch.nn.utils.clip_grad_norm_(drug_model.parameters(), max_norm=1.0)
                optimizer.step()
                
                train_total_loss += loss.item()
                
                # save preds and ys for later evaluation
                if setting.y_transform:
                    ys = std_scaler.inverse_transform(ys.cpu().reshape(-1, 1) / 100)
                    preds = std_scaler.inverse_transform(preds.detach().cpu().numpy().reshape(-1, 1) / 100)
                all_preds.append(preds)
                all_ys.append(ys)

            all_preds = np.concatenate(all_preds)
            all_ys = np.concatenate(all_ys)
            
            avg_train_loss = mean_squared_error(all_preds, all_ys)
            train_pearson = pearsonr(all_preds.reshape(-1), all_ys.reshape(-1))[0]
            train_spearman = spearmanr(all_preds.reshape(-1), all_ys.reshape(-1))[0]
            scheduler.step()

            with torch.set_grad_enabled(False):
                drug_model.eval()
                all_preds = []
                all_ys = []

                validation_iter = iter(validation_generator)

                for cur_local_batch, cur_local_labels in validation_iter:
                    # No need for reshaping on CPU
                    local_batch, local_labels = cur_local_batch.float().to(device2), cur_local_labels.float().to(device2)
                    local_batch = local_batch.contiguous().view(-1, 1, sum(slice_indices) + setting.single_repsonse_feature_length)
                    reorder_tensor.load_raw_tensor(local_batch)
                    local_batch = reorder_tensor.get_reordered_narrow_tensor()
                    preds = drug_model(*local_batch)
                    preds = preds.contiguous().view(-1)
                    ys = local_labels.contiguous().view(-1)

                    if setting.y_transform:
                        ys = std_scaler.inverse_transform(ys.cpu().reshape(-1, 1) / 100)
                        preds = std_scaler.inverse_transform(preds.detach().cpu().numpy().reshape(-1, 1) / 100)
                    all_preds.append(preds)
                    all_ys.append(ys)
                # Concatenate predictions and labels after processing all batches
                all_preds = np.concatenate(all_preds)
                all_ys = np.concatenate(all_ys)

                # Ensure that predictions and labels have the same length
                assert len(all_preds) == len(all_ys), "predictions and labels are in different lengths"

                # Compute metrics
                val_loss = mean_squared_error(all_preds, all_ys)
                val_pearson = pearsonr(all_preds.reshape(-1), all_ys.reshape(-1))[0]
                val_spearman = spearmanr(all_preds.reshape(-1), all_ys.reshape(-1))[0]
                if USE_wandb:
                    wandb.log({
                        "Train Loss": avg_train_loss,
                        "Validation Loss": val_loss,
                        "Epoch": epoch + 1,
                        "LR": scheduler.get_last_lr()[0],
                        "Val Pearson": val_pearson,
                        "Val Spearman": val_spearman,
                        "Train Pearson": train_pearson,
                        "Train Spearman": train_spearman
                    }, step=epoch + 1)

                # Save model if better performance is achieved
                if best_cv_pearson_score < val_pearson:
                    logger.debug('++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++')
                    logger.debug(f'saved a model, sample size {len(all_preds)}')
                    logger.debug('++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++')
                    best_cv_pearson_score = val_pearson
                    best_drug_model.load_state_dict(drug_model.state_dict())


            logger.info(
                "Validation mse is {0}, Validation pearson correlation is {1!r}, Validation spearman correlation is {2!r}"
                    .format(np.mean(val_loss), val_pearson, val_spearman))
        if USE_wandb:
            wandb.finish()

    wandb.init(project="Drug combination alpha_fold - Testing best model",)
    ### Testing

    if setting.load_old_model:
        best_drug_model.load_state_dict(load(setting.old_model_path).state_dict())

    with torch.set_grad_enabled(False):

        best_drug_model.eval()
        all_preds = []
        all_ys = []

        test_iter = iter(test_generator)

        for cur_local_batch, cur_local_labels in tqdm(test_iter, desc="Testing", total=len(test_iter)):
            # Transfer to GPU
            local_batch, local_labels = cur_local_batch.float().to(device2), cur_local_labels.float().to(device2)
            local_batch = local_batch.contiguous().view(-1, 1, sum(slice_indices) + setting.single_repsonse_feature_length)
            reorder_tensor.load_raw_tensor(local_batch)
            local_batch = reorder_tensor.get_reordered_narrow_tensor()
            preds = drug_model(*local_batch)
            preds = preds.contiguous().view(-1)
            ys = local_labels.contiguous().view(-1)
            assert preds.size(-1) == local_labels.size(-1)
            # Directly process predictions on GPU, avoid CPU transfer unless needed
            if setting.y_transform:
                ys = std_scaler.inverse_transform(ys.cpu().reshape(-1, 1) / 100)
                preds = std_scaler.inverse_transform(preds.detach().cpu().numpy().reshape(-1, 1) / 100)
            all_preds.append(preds)
            all_ys.append(ys)

        all_preds = np.concatenate(all_preds)
        all_ys = np.concatenate(all_ys)
        assert len(all_preds) == len(all_ys), "predictions and labels are in different length"
        sample_size = len(all_preds)
        
        test_loss = mean_squared_error(all_preds, all_ys)
        test_pearson = pearsonr(all_ys.reshape(-1), all_preds.reshape(-1))[0]
        test_spearman = spearmanr(all_ys.reshape(-1), all_preds.reshape(-1))[0]
        if not path.exists('prediction'):
            mkdir('prediction')
        save(np.concatenate((np.array(test_index_list[:sample_size]).reshape(-1,1), all_preds.reshape(-1, 1), all_ys.reshape(-1, 1)), axis=1),
             "prediction/prediction_" + setting.catoutput_output_type + "_testing")
        
        if USE_wandb:
            wandb.log({
                "Test Loss": test_loss,
                "Test Pearson": test_pearson,
                "Test Spearman": test_spearman
            })

    logger.debug("Testing mse is {0}, Testing pearson correlation is {1!r}, Testing spearman correlation is {2!r}".format(np.mean(test_loss), test_pearson, test_spearman))

    if not setting.perform_importance_study:
        return

    batch_input_importance = []
    batch_out_input_importance = []
    batch_transform_input_importance = []
    total_data, _ = next(iter(all_data_generator_total))
    logger.debug("start kmeans")
    all_index_ls = partition['train'][:len(partition['train']) // 2] + partition['eval1'] + partition['test1']
    ### same definition with the one in function generate data generator
    kmeans = MiniBatchKMeans(n_clusters=len(total_data)//40, random_state=0, batch_size=len(all_index_ls)//8).fit(total_data)
    logger.debug("fiting finished")
    total_data = kmeans.cluster_centers_
    total_data = torch.from_numpy(total_data).float().to(device2)
    total_data = total_data.contiguous().view(-1, 1, sum(slice_indices) + setting.single_repsonse_feature_length)
    reorder_tensor.load_raw_tensor(total_data)
    total_data = reorder_tensor.get_reordered_narrow_tensor()

    for (local_batch, smiles_a, smiles_b), local_labels in all_data_generator:
        # Transfer to GPU
        local_batch, local_labels = local_batch.float().to(device2), local_labels.float().to(device2)
        local_batch = local_batch.contiguous().view(-1, 1, sum(slice_indices) + setting.single_repsonse_feature_length)
        reorder_tensor.load_raw_tensor(local_batch)
        local_batch = reorder_tensor.get_reordered_narrow_tensor()

        if setting.save_feature_imp_model:
            save(best_drug_model, setting.best_model_path)
        # Model computations
        logger.debug("Start feature importances analysis")
        if setting.save_easy_input_only:
            e = shap.DeepExplainer(best_drug_model, data=list(total_data))
            input_importance = e.shap_values(list(local_batch))
        else:
            input_importance = []
            for layer in best_drug_model.linear_layers:
                cur_e = shap.GradientExplainer((best_drug_model, layer), data=list(total_data))
                cur_input_importance = cur_e.shap_values(list(local_batch))
                input_importance.append(cur_input_importance)
            input_importance = np.concatenate(tuple(input_importance), axis=1)
        batch_input_importance.append(input_importance)
        logger.debug("Finished one batch of input importance analysis")

        if setting.save_out_imp:
            e1 = shap.GradientExplainer((best_drug_model, best_drug_model.out), data=list(total_data))
            out_input_shap_value = e1.shap_values(list(local_batch))
            batch_out_input_importance.append(out_input_shap_value)
            logger.debug("Finished one batch of out input importance analysis")

        if setting.save_inter_imp:
            transform_input_importance = []
            for layer in best_drug_model.dropouts:
                cur_e = shap.GradientExplainer((best_drug_model, layer), data=list(total_data))
                cur_transform_input_shap_value = cur_e.shap_values(list(local_batch))
                transform_input_importance.append(cur_transform_input_shap_value)
            transform_input_importance = np.concatenate(tuple(transform_input_importance), axis=1)

            batch_transform_input_importance.append(transform_input_importance)
        logger.debug("Finished one batch of importance analysis")
    batch_input_importance = np.concatenate(tuple(x[0] for x in batch_input_importance), axis=0)
    pickle.dump(batch_input_importance, open(setting.input_importance_path, 'wb+'))
    wandb.log({"Input Importance": wandb.Histogram(batch_input_importance.flatten())})
    logger.debug("Finished all batches of input importance analysis")

    if setting.save_out_imp:
        batch_out_input_importance = np.concatenate(tuple(batch_out_input_importance), axis=0)
        pickle.dump(batch_out_input_importance, open(setting.out_input_importance_path, 'wb+'))

    if setting.save_inter_imp:
        batch_transform_input_importance = np.concatenate(tuple(batch_transform_input_importance), axis=0)
        pickle.dump(batch_transform_input_importance, open(setting.transform_input_importance_path, 'wb+'))



if __name__ == "__main__":

    USE_wandb = False
    if USE_wandb:
        pass
    else:
        environ["WANDB_MODE"] = "dryrun"

    try:
        run()
        logger.debug("new directory %s" % setting.run_dir)

    except:

        import shutil
        logger.debug("clean directory %s" % setting.run_dir)
        raise

