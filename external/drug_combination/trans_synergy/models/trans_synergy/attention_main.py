import logging
import pickle
import shutil
from os import environ, makedirs, path, sep

import numpy as np
import pandas as pd
import shap
import torch
import torch.nn.functional as F
import trans_synergy.data.trans_synergy_data
import trans_synergy.settings
from scipy.stats import pearsonr, spearmanr
from sklearn.cluster import MiniBatchKMeans
from sklearn.metrics import mean_squared_error
from sklearn.preprocessing import StandardScaler
from torch import load, save
from torch.utils import data
from tqdm import tqdm
from trans_synergy import device2
from trans_synergy.data.utils import TensorReorganizer
from trans_synergy.models.other import drug_drug
from trans_synergy.models.trans_synergy import attention_model
from trans_synergy.utils import set_seed

import wandb

setting = trans_synergy.settings.get()

logger = logging.getLogger(__name__)
random_seed = 913

def get_final_index():

    ## get the index of synergy score database
    if not setting.update_final_index and path.exists(setting.final_index):
        final_index = pd.read_csv(setting.final_index, header=None)[0]
    else:
        final_index = trans_synergy.data.trans_synergy_data.SynergyDataReader.get_final_index()
    return final_index

def prepare_data():

    if not setting.update_xy:
        assert (path.exists(setting.old_x) and path.exists(
            setting.old_y)), "Data need to be downloaded from zenodo follow instruction in README"
        X = np.load(setting.old_x)
        with open(setting.old_x_lengths, 'rb') as old_x_lengths:
            drug_features_length, cellline_features_length = pickle.load(old_x_lengths)
        with open(setting.old_y, 'rb') as old_y:
            Y = pickle.load(old_y)
    else:
        X, drug_features_length, cellline_features_length = \
            trans_synergy.data.trans_synergy_data.SamplesDataLoader.Raw_X_features_prep(methods='flexible_attn')
        np.save(setting.old_x, X)
        with open(setting.old_x_lengths, 'wb+') as old_x_lengths:
            pickle.dump((drug_features_length,cellline_features_length), old_x_lengths)

        Y = trans_synergy.data.trans_synergy_data.SamplesDataLoader.Y_features_prep()
        with open(setting.old_y, 'wb+') as old_y:
            pickle.dump(Y, old_y)
    return X, Y, drug_features_length, cellline_features_length


def prepare_model(reorder_tensor):

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

def prepare_splitted_dataloaders(partition, labels, loaded_data):
    """
    Prepare train, validation, test, and evaluation data generators.
    """

    logger.debug("Preparing datasets ... ")

    # Training dataset and generator
    training_set = trans_synergy.data.trans_synergy_data.TransSynergyDataset(partition['train'], labels, loaded_data)
    train_params = {'batch_size': setting.batch_size, 'shuffle': True}
    training_generator = data.DataLoader(training_set, **train_params)

    # Evaluation training dataset and generator (train + eval1 + eval2)
    eval_train_set = trans_synergy.data.trans_synergy_data.TransSynergyDataset(np.concatenate([partition['train'], partition['eval1'], partition['eval2']]), labels, loaded_data)
    training_index_list = np.concatenate([partition['train'], partition['eval1'], partition['eval2']])
    logger.debug(f"Training data length: {len(training_index_list)}")
    eval_train_params = {'batch_size': setting.batch_size, 'shuffle': False}
    eval_train_generator = data.DataLoader(eval_train_set, **eval_train_params)

    # Validation dataset and generator (eval1)
    validation_set = trans_synergy.data.trans_synergy_data.TransSynergyDataset(partition['eval1'], labels, loaded_data)
    eval_params = {'batch_size': len(partition['test1']) // 4, 'shuffle': False}
    validation_generator = data.DataLoader(validation_set, **eval_params)

    # Test dataset and generator (test1)
    test_set = trans_synergy.data.trans_synergy_data.TransSynergyDataset(partition['test1'], labels, loaded_data)
    test_index_list = partition['test1']
    logger.debug(f"Test data length: {len(test_index_list)}")
    pickle.dump(test_index_list, open("test_index_list", "wb+"))
    test_params = {'batch_size': len(test_index_list) // 4, 'shuffle': False}
    test_generator = data.DataLoader(test_set, **test_params)

    # All dataset and generator (train + eval1 + test1)
    all_index_list = np.concatenate([partition['train'][:len(partition['train']) // 2], partition['eval1'], partition['test1']])
    all_set = trans_synergy.data.trans_synergy_data.TransSynergyDataset(all_index_list, labels, loaded_data)
    logger.debug(f"All data length: {len(set(all_index_list))}")
    pickle.dump(all_index_list, open("all_index_list", "wb+"))
    all_set_params = {'batch_size': len(all_index_list) // 8, 'shuffle': False}
    all_data_generator = data.DataLoader(all_set, **all_set_params)

    # Total all data generator (using the full batch size)
    all_set_params_total = {'batch_size': len(all_index_list), 'shuffle': False}
    all_data_generator_total = data.DataLoader(all_set, **all_set_params_total)

    return (training_generator, eval_train_generator, validation_generator,
            test_generator, all_data_generator, all_data_generator_total)


def prepare_splitted_datasets(partition, labels, loaded_data):
    """
    Prepare train, validation, test, and evaluation data generators.
    """

    logger.debug("Preparing datasets ... ")

    # Training dataset and generator
    training_set = trans_synergy.data.trans_synergy_data.TransSynergyDataset(partition['train'], labels, loaded_data)
    # Evaluation training dataset and generator (train + eval1 + eval2)
    eval_train_set = trans_synergy.data.trans_synergy_data.TransSynergyDataset(np.concatenate([partition['train'], partition['eval1'], partition['eval2']]), labels, loaded_data)
    # Validation dataset and generator (eval1)
    validation_set = trans_synergy.data.trans_synergy_data.TransSynergyDataset(partition['eval1'], labels, loaded_data)
    # Test dataset and generator (test1)
    test_set = trans_synergy.data.trans_synergy_data.TransSynergyDataset(partition['test1'], labels, loaded_data)

    # All dataset and generator (train + eval1 + test1)
    all_index_list = np.concatenate([partition['train'][:len(partition['train']) // 2], partition['eval1'], partition['test1']])
    all_set = trans_synergy.data.trans_synergy_data.TransSynergyDataset(all_index_list, labels, loaded_data)

    return (training_set, eval_train_set, validation_set,
            test_set,  all_set)


def setup_data():
    logger.debug("Getting features and synergy scores ...")
    std_scaler = StandardScaler()
    X, Y, drug_features_length, cellline_features_length = prepare_data()
    return std_scaler, X, Y, drug_features_length, cellline_features_length

def setup_tensor_reorganizer(drug_features_length, cellline_features_length):
    slice_indices = drug_features_length + drug_features_length + cellline_features_length
    reorder_tensor = TensorReorganizer(slice_indices, setting.arrangement, 2)
    logger.debug("Feature layout: {!r}".format(reorder_tensor.get_reordered_slice_indices()))
    return reorder_tensor

def setup_model_and_optimizer(reorder_tensor):
    set_seed(random_seed)
    drug_model, best_drug_model = prepare_model(reorder_tensor)
    optimizer = torch.optim.Adam(
        drug_model.parameters(), lr=setting.start_lr,
        weight_decay=setting.lr_decay, betas=(0.9, 0.98), eps=1e-9
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=100, eta_min=1e-7)
    return drug_model, best_drug_model, optimizer, scheduler

def enumerate_splits(split_func):
    return tqdm(split_func(fold='fold', test_fold=4), desc="Folds", total=1)

def init_wandb(fold_idx: int = None, testing: bool = False, crossval = False, fold_col_name = 'fold'):
    suffix_cv = ' crossval' if crossval else ''
    if testing:
        wandb.init(project=f"Drug combination TRANSYNERGY Testing [{fold_col_name}] " + suffix_cv,)
    else:
        wandb.init(
            project=f"Drug combination TRANSYNERGY [{fold_col_name}] fold_{fold_idx}{suffix_cv}",
            name=path.basename(setting.run_dir).rsplit(sep, 1)[-1] + '_' + setting.data_specific[:15] + '_' + str(random_seed),
            notes=setting.data_specific
        )
        wandb.define_metric("Train Loss", step_metric="Epoch")
        wandb.define_metric("Validation Loss", step_metric="Epoch")

    return wandb


def train_loop(model, 
               best_model, 
               train_loader, 
               val_loader, 
               optimizer, 
               scheduler, 
               reorder_tensor, 
               std_scaler, 
               use_wandb, 
               slice_indices, 
               patience = 100,
               n_epochs = 800):
    best_val_score = -float("inf")
    epochs_without_improvement = 0

    for epoch in tqdm(range(n_epochs), desc="Training Epochs", leave=False):
        model.to(device2)
        model.train()
        train_loss, train_preds, train_ys = 0, [], []

        for batch, labels in tqdm(train_loader, desc=f"Epoch {epoch+1}", leave=False):
            batch = batch.float().to(device2)
            labels = labels.float().to(device2)
            batch = batch.contiguous().view(-1, 1, sum(slice_indices) + setting.single_repsonse_feature_length)
            reorder_tensor.load_raw_tensor(batch)
            batch = reorder_tensor.get_reordered_narrow_tensor()

            preds = model(*batch).view(-1)
            ys = labels.view(-1)
            loss = F.mse_loss(preds, ys)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            train_loss += loss.item()

            if setting.y_transform:
                ys = std_scaler.inverse_transform(ys.cpu().reshape(-1, 1) / 100)
                preds = std_scaler.inverse_transform(preds.detach().cpu().numpy().reshape(-1, 1) / 100)
            train_preds.append(preds)
            train_ys.append(ys)

        scheduler.step()

        train_preds = np.concatenate(train_preds)
        train_ys = np.concatenate(train_ys)
        train_mse = mean_squared_error(train_preds, train_ys)
        train_pearson = pearsonr(train_preds.ravel(), train_ys.ravel())[0]
        train_spearman = spearmanr(train_preds.ravel(), train_ys.ravel())[0]

        val_metrics = evaluate(model, val_loader, reorder_tensor, std_scaler, slice_indices)

        if use_wandb:
            wandb.log({
                "Train Loss": train_mse,
                "Validation Loss": val_metrics['mse'],
                "Epoch": epoch + 1,
                "LR": scheduler.get_last_lr()[0],
                "Val Pearson": val_metrics['pearson'],
                "Val Spearman": val_metrics['spearman'],
                "Train Pearson": train_pearson,
                "Train Spearman": train_spearman,
            }, step=epoch + 1)

        if val_metrics['pearson'] > best_val_score:
            best_val_score = val_metrics['pearson']
            best_model.load_state_dict(model.state_dict())
            logger.debug("New best model saved.")
            epochs_without_improvement = 0  # reset counter
        else:
            epochs_without_improvement += 1
            logger.debug(f"No improvement for {epochs_without_improvement} epochs.")

        logger.info(f"Epoch {epoch+1}: Val MSE={val_metrics['mse']:.4f}, Pearson={val_metrics['pearson']:.4f}")

        if epochs_without_improvement >= patience:
            logger.info(f"Early stopping triggered after {epoch+1} epochs without improvement.")
            break

    return best_model


def evaluate(model, data_loader, reorder_tensor, std_scaler, slice_indices, log_model=False):
    model.eval()
    all_preds, all_ys = [], []
    model.to(device2)
    with torch.no_grad():
        for batch, labels in data_loader:
            batch = batch.float().to(device2)
            labels = labels.float().to(device2)
            batch = batch.contiguous().view(-1, 1, sum(slice_indices) + setting.single_repsonse_feature_length)
            reorder_tensor.load_raw_tensor(batch)
            batch = reorder_tensor.get_reordered_narrow_tensor()

            preds = model(*batch).view(-1)
            ys = labels.view(-1)

            if setting.y_transform:
                ys = std_scaler.inverse_transform(ys.cpu().reshape(-1, 1) / 100)
                preds = std_scaler.inverse_transform(preds.detach().cpu().numpy().reshape(-1, 1) / 100)

            all_preds.append(preds)
            all_ys.append(ys)

    all_preds = np.concatenate(all_preds)
    all_ys = np.concatenate(all_ys)
    
    if log_model:
        save_model(model, setting.run_dir, fold="test")
    
    return {
        "mse": mean_squared_error(all_preds, all_ys),
        "pearson": pearsonr(all_preds.ravel(), all_ys.ravel())[0],
        "spearman": spearmanr(all_preds.ravel(), all_ys.ravel())[0]
    }

def test_best_model(model, test_loader, reorder_tensor, std_scaler, slice_indices, use_wandb=False, crossval=False, fold_col_name = 'fold'):
    if use_wandb:
        init_wandb(testing=True, crossval=crossval, fold_col_name = fold_col_name)

    if setting.load_old_model:
        model.load_state_dict(load(setting.old_model_path).state_dict())

    metrics = evaluate(model, test_loader, reorder_tensor, std_scaler, slice_indices, log_model=True)
    logger.info(f"Test MSE: {metrics['mse']:.4f}, Pearson: {metrics['pearson']:.4f}, Spearman: {metrics['spearman']:.4f}")
    if use_wandb:
        wandb.log({"Test MSE": metrics['mse'], "Test Pearson": metrics['pearson'], "Test Spearman": metrics['spearman']})
        wandb.finish()
    
def save_model(model, save_path, fold):
    """
    Saves the model to a given path with a fold-specific suffix.

    Args:
        model (torch.nn.Module): The model to save.
        path (str): Base directory where the model should be saved.
        fold (str or int): Fold index to append to the path.
    """
    # construct the full save path
    model_filename = f"fold_{fold}_model.pt"
    model_path = path.join(save_path, model_filename)
    makedirs(path.dirname(model_path), exist_ok=True)
    torch.save(model, model_path)
    
    try:
        wandb.save(model_path, base_path = save_path, policy="now")
    except OSError as e: # Windows throws OS errors because of symlinks https://github.com/wandb/wandb/issues/1370
        wandb_model_path = path.join(wandb.run.dir, f"fold_{fold}_model.pt")
        shutil.copy(model_path, wandb_model_path)
        wandb.save(wandb_model_path, base_path = wandb.run.dir)

def train_model_on_fold(fold_idx, partition, X, Y, std_scaler, reorder_tensor,
                        drug_model, best_drug_model, optimizer, scheduler, use_wandb, slice_indices, fold_col_name='fold'):
    if use_wandb:
        init_wandb(fold_idx, fold_col_name = fold_col_name)
    
    partition_indices = {
        'train': partition[0],
        'test1': partition[1],
        'test2': partition[2],
        'eval1': partition[3],
        'eval2': partition[4]
    }

    std_scaler.fit(Y[partition_indices['train']])
    if setting.y_transform:
        Y = std_scaler.transform(Y) * 100

    # Dataloaders
    training_generator, _, validation_generator, test_generator, \
    all_data_generator, all_data_generator_total = prepare_splitted_dataloaders(partition_indices, Y.reshape(-1), X)

    best_model = train_loop(model = drug_model,
            best_model = best_drug_model,
            train_loader = training_generator, 
            val_loader = validation_generator, 
            optimizer = optimizer,
            scheduler = scheduler, 
            reorder_tensor = reorder_tensor, 
            std_scaler = std_scaler, 
            use_wandb=  use_wandb, 
            slice_indices = slice_indices,
            n_epochs=setting.n_epochs)

    save_model(best_model, setting.run_dir, fold_idx)

    if use_wandb:
        wandb.finish()
    return training_generator, validation_generator, test_generator, all_data_generator, all_data_generator_total
        
def run_kmeans(data, n_clusters, batch_size):
    kmeans = MiniBatchKMeans(n_clusters=n_clusters, random_state=0, batch_size=batch_size)
    kmeans.fit(data)
    return kmeans.cluster_centers_

def compute_shap_importance(model, data, background, method="deep", layers=None):
    if method == "deep":
        explainer = shap.DeepExplainer(model, background)
        return explainer.shap_values(data)
    elif method == "gradient":
        importances = []
        for layer in layers:
            explainer = shap.GradientExplainer((model, layer), background)
            values = explainer.shap_values(data)
            importances.append(values)
        return np.concatenate(importances, axis=1)

def run_importance_study(
    setting,
    all_data_generator_total,
    all_data_generator,
    partition,
    reorder_tensor,
    slice_indices,
    device2,
    best_drug_model,
    logger
):

    batch_input_importance = []
    batch_out_input_importance = []
    batch_transform_input_importance = []

    total_data, _ = next(iter(all_data_generator_total))
    logger.debug("start kmeans")

    all_index_ls = partition['train'][:len(partition['train']) // 2] + partition['eval1'] + partition['test1']
    n_clusters = max(2, len(total_data) // 40)
    batch_size = max(1, len(all_index_ls) // 8)

    kmeans_centers = run_kmeans(total_data, n_clusters=n_clusters, batch_size=batch_size)
    logger.debug("fiting finished")

    total_data_tensor = torch.from_numpy(kmeans_centers).float().to(device2)
    total_data_tensor = total_data_tensor.contiguous().view(-1, 1, sum(slice_indices) + setting.single_repsonse_feature_length)
    reorder_tensor.load_raw_tensor(total_data_tensor)
    total_data_tensor = reorder_tensor.get_reordered_narrow_tensor()

    for (local_batch, smiles_a, smiles_b), local_labels in all_data_generator:
        local_batch = local_batch.float().to(device2)
        local_batch = local_batch.contiguous().view(-1, 1, sum(slice_indices) + setting.single_repsonse_feature_length)
        reorder_tensor.load_raw_tensor(local_batch)
        local_batch = reorder_tensor.get_reordered_narrow_tensor()

        if setting.save_feature_imp_model:
            save(best_drug_model, setting.best_model_path)

        logger.debug("Start feature importances analysis")
        if setting.save_easy_input_only:
            input_importance = compute_shap_importance(best_drug_model, list(local_batch), list(total_data_tensor), method="deep")
        else:
            input_importance = compute_shap_importance(best_drug_model, list(local_batch), list(total_data_tensor), method="gradient", layers=best_drug_model.linear_layers)

        batch_input_importance.append(input_importance)
        logger.debug("Finished one batch of input importance analysis")

        if setting.save_out_imp:
            out_input_importance = compute_shap_importance(best_drug_model, list(local_batch), list(total_data_tensor), method="gradient", layers=[best_drug_model.out])
            batch_out_input_importance.append(out_input_importance)
            logger.debug("Finished one batch of out input importance analysis")

        if setting.save_inter_imp:
            inter_input_importance = compute_shap_importance(best_drug_model, list(local_batch), list(total_data_tensor), method="gradient", layers=best_drug_model.dropouts)
            batch_transform_input_importance.append(inter_input_importance)
            logger.debug("Finished one batch of importance analysis")

    # Save and log results
    batch_input_importance = np.concatenate([x[0] for x in batch_input_importance], axis=0)
    pickle.dump(batch_input_importance, open(setting.input_importance_path, 'wb+'))
    wandb.log({"Input Importance": wandb.Histogram(batch_input_importance.flatten())})
    logger.debug("Finished all batches of input importance analysis")

    if setting.save_out_imp:
        batch_out_input_importance = np.concatenate(batch_out_input_importance, axis=0)
        pickle.dump(batch_out_input_importance, open(setting.out_input_importance_path, 'wb+'))

    if setting.save_inter_imp:
        batch_transform_input_importance = np.concatenate(batch_transform_input_importance, axis=0)
        pickle.dump(batch_transform_input_importance, open(setting.transform_input_importance_path, 'wb+'))


def run(use_wandb: bool = True,
        fold_col_name: str = 'fold',
        test_fold: int = 4,
        eval_fold: int = 0):
    if not use_wandb:
        environ["WANDB_MODE"] = "dryrun"

    std_scaler, X, Y, drug_features_length, cellline_features_length = setup_data()
    reorder_tensor = setup_tensor_reorganizer(drug_features_length, cellline_features_length)
    
    drug_model, best_drug_model, optimizer, scheduler = setup_model_and_optimizer(reorder_tensor)
    
    slice_indices = drug_features_length + drug_features_length + cellline_features_length
    
    split_func = trans_synergy.data.trans_synergy_data.DataPreprocessor.regular_train_eval_test_split
    fold_idx = 0
    partition = split_func(fold_col_name=fold_col_name, test_fold=test_fold, evaluation_fold=eval_fold)
    training_generator, validation_generator, test_generator, all_data_generator, all_data_generator_total = train_model_on_fold(fold_idx, partition, X, Y, std_scaler, reorder_tensor,
                        drug_model, best_drug_model, optimizer, scheduler, use_wandb, slice_indices, fold_col_name = fold_col_name)

    test_best_model(best_drug_model, test_generator, reorder_tensor, std_scaler, slice_indices, use_wandb, fold_col_name = fold_col_name)
    if setting.perform_importance_study:
        run_importance_study( setting, all_data_generator_total, all_data_generator, partition, reorder_tensor, slice_indices,
            device2, best_drug_model, logger)
        