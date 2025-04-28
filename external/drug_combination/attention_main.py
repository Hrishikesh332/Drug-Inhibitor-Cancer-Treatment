#!/usr/bin/env python

import numpy as np
import pandas as pd
from os import path, mkdir, environ
import sys
sys.path.append(path.join(path.dirname(__file__), 'src', 'NeuralFingerPrint'))
sys.path.append(path.dirname(__file__))
from time import time
import torch
from torch import save, load
from torch.utils import data
from src import attention_model, drug_drug, setting, my_data, logger, device2
import torch.nn.functional as F
from scipy.stats import pearsonr, spearmanr
from sklearn.metrics import mean_squared_error
from sklearn.preprocessing import StandardScaler
import pdb
import shap
import pickle
from sklearn.cluster import MiniBatchKMeans
import wandb
import data_utils
import concurrent.futures
import random
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
    if USE_wandb:
        wandb.watch(drug_model, log="all")
    return drug_model, best_drug_model


def persist_data_as_data_point_file(local_X, final_index_for_X):

    ### prepare files for dataloader
    for i, combin_drug_feature_array in enumerate(local_X):
        if setting.unit_test:
            if i <= 501:  # and not path.exists(path.join('datas', str(final_index_for_X.iloc[i]) + '.pt')):
                save(combin_drug_feature_array, path.join(setting.data_folder, str(final_index_for_X.iloc[i]) + '.pt'))
        else:
            if setting.update_features or not path.exists(
                    path.join(setting.data_folder, str(final_index_for_X.iloc[i]) + '.pt')):
                save(combin_drug_feature_array, path.join(setting.data_folder, str(final_index_for_X.iloc[i]) + '.pt'))


def prepare_splitted_dataset(partition, labels):

    ### prepare train, test, evaluation data generator

    logger.debug("Preparing datasets ... ")
    # training_set = my_data.MyDataset(partition['train'] + partition['eval1'] + partition['eval2'], labels)
    training_set = my_data.MyDataset(partition['train'], labels)
    train_params = {'batch_size': setting.batch_size,
                    'shuffle': True}
    training_generator = data.DataLoader(training_set, **train_params)

    eval_train_set = my_data.MyDataset(partition['train'] + partition['eval1'] + partition['eval2'], labels)
    training_index_list = partition['train'] + partition['eval1'] + partition['eval2']
    logger.debug("Training data length: {!r}".format(len(training_index_list)))
    eval_train_params = {'batch_size': setting.batch_size,
                         'shuffle': False}
    eval_train_generator = data.DataLoader(eval_train_set, **eval_train_params)

    # validation_set = my_data.MyDataset(partition['test1'], labels)
    validation_set = my_data.MyDataset(partition['eval1'], labels)
    eval_params = {'batch_size': len(partition['test1'])//4,
                   'shuffle': False}
    validation_generator = data.DataLoader(validation_set, **eval_params)

    test_set = my_data.MyDataset(partition['test1'], labels)
    test_index_list = partition['test1']
    logger.debug("Test data length: {!r}".format(len(test_index_list)))
    pickle.dump(test_index_list, open("test_index_list", "wb+"))
    test_params = {'batch_size': len(test_index_list) // 4,
                   'shuffle': False}
    test_generator = data.DataLoader(test_set, **test_params)

    all_index_list = partition['train'][:len(partition['train']) // 2] + partition['eval1'] + partition['test1']
    all_set = my_data.MyDataset(all_index_list, labels)
    logger.debug("All data length: {!r}".format(len(set(all_index_list))))
    pickle.dump(all_index_list, open("all_index_list", "wb+"))
    all_set_params = {'batch_size': len(all_index_list) // 8,
                      'shuffle': False}
    all_data_generator = data.DataLoader(all_set, **all_set_params)
    ### generate all the data in one iteration because the batch size is bigger than all_data_generator
    all_set_params_total = {'batch_size': len(all_index_list),
                            'shuffle': False}
    all_data_generator_total = data.DataLoader(all_set, **all_set_params_total)
    return training_generator, eval_train_generator, validation_generator, test_generator, all_data_generator, all_data_generator_total

def run():

    final_index = get_final_index()
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
    logger.debug("Spliting data ...")

    for train_index, test_index, test_index_2, evaluation_index, evaluation_index_2 in split_func(fold='fold', test_fold = 4):

        local_X = X[np.concatenate((train_index, test_index, test_index_2, evaluation_index, evaluation_index_2))]
        final_index_for_X = final_index.iloc[np.concatenate((train_index, test_index,
                                                             test_index_2, evaluation_index, evaluation_index_2))]

        ori_Y = Y
        std_scaler.fit(Y[train_index])
        if setting.y_transform:
            Y = std_scaler.transform(Y) * 100

        persist_data_as_data_point_file(local_X, final_index_for_X)

        partition = {'train': list(final_index.iloc[train_index]),
                     'test1': list(final_index.iloc[test_index]), 'test2': list(final_index.iloc[test_index_2]),
                     'eval1': list(final_index.iloc[evaluation_index]),
                     'eval2': list(final_index.iloc[evaluation_index_2])}

        labels = {key: value for key, value in zip(list(final_index),
                                                   list(Y.reshape(-1)))}
        ori_labels = {key: value for key, value in zip(list(final_index),
                                                   list(ori_Y.reshape(-1)))}
        save(ori_labels, setting.y_labels_file)

        training_generator, eval_train_generator, validation_generator, test_generator, \
        all_data_generator, all_data_generator_total = prepare_splitted_dataset(partition, labels)
        test_index_list = partition['test1']

        logger.debug("Start training")
        set_seed()

        for epoch in range(setting.n_epochs):

            drug_model.train()
            start = time()
            cur_epoch_train_loss = []
            train_total_loss = 0
            train_i = 0
            all_preds = []
            all_ys = []

            training_iter = iter(training_generator)
            # (pre_local_batch, pre_smiles_a, pre_smiles_b), pre_local_labels = next(training_iter)
            # drug_a_result = executor.submit(data_utils.convert_smile_to_feature,
            #                                 pre_smiles_a, device = device("cuda:0"))
            # drug_b_result = executor.submit(data_utils.convert_smile_to_feature,
            #                                 pre_smiles_b, device = device("cuda:0"))

            # Training
            for (cur_local_batch, cur_smiles_a, cur_smiles_b), cur_local_labels in training_iter:

                train_i += 1
                # Transfer to GPU
                if epoch == 0 and train_i == 1:
                    print('--------------------------------cur local labels---------------------------------')
                    print(cur_local_labels)
                    print('--------------------------------cur local labels---------------------------------')
                pre_local_batch = cur_local_batch
                pre_local_labels = cur_local_labels
                local_labels_on_cpu = np.array(pre_local_labels).reshape(-1)
                sample_size = local_labels_on_cpu.shape[-1]
                local_labels_on_cpu = local_labels_on_cpu[:sample_size]
                local_batch, local_labels = pre_local_batch.float().to(device2), pre_local_labels.float().to(device2)
                local_batch = local_batch.contiguous().view(-1, 1, sum(slice_indices) + setting.single_repsonse_feature_length)
                reorder_tensor.load_raw_tensor(local_batch)
                local_batch = reorder_tensor.get_reordered_narrow_tensor()
                # pre_drug_a, pre_drug_b = drug_a_result.result(), drug_b_result.result()
                # drugs = (pre_drug_a, pre_drug_b)
                # drug_a_result = executor.submit(data_utils.convert_smile_to_feature,
                #                                 cur_smiles_a, device=device("cuda:0"))
                # drug_b_result = executor.submit(data_utils.convert_smile_to_feature,
                #                                 cur_smiles_b, device=device("cuda:0"))
                # Model computations
                # drug_a = data_utils.convert_smile_to_feature(cur_smiles_a, device=device("cuda:0"))
                # drug_b = data_utils.convert_smile_to_feature(cur_smiles_b, device=device("cuda:0"))
                # drugs = (drug_a, drug_b)
                # drugs = (cur_smiles_a, cur_smiles_b)
                # preds = drug_model(*local_batch, drugs = drugs)
                preds = drug_model(*local_batch)
                preds = preds.contiguous().view(-1)
                ys = local_labels.contiguous().view(-1)
                optimizer.zero_grad()
                assert preds.size(-1) == ys.size(-1)
                loss = F.mse_loss(preds, ys)
                loss.backward()
                optimizer.step()
                prediction_on_cpu = preds.detach().cpu().numpy().reshape(-1)
                # mean_prediction_on_cpu = np.mean([prediction_on_cpu[:sample_size],
                #                                   prediction_on_cpu[sample_size:]], axis=0)
                mean_prediction_on_cpu = prediction_on_cpu[:sample_size]
                if setting.y_transform:
                    local_labels_on_cpu, mean_prediction_on_cpu = \
                        std_scaler.inverse_transform(local_labels_on_cpu.reshape(-1, 1) / 100), \
                        std_scaler.inverse_transform(mean_prediction_on_cpu.reshape(-1, 1) / 100)
                all_preds.append(mean_prediction_on_cpu)
                all_ys.append(local_labels_on_cpu)
                pre_local_batch = cur_local_batch
                pre_local_labels = cur_local_labels

                train_total_loss += loss.item()

                n_iter = 50
                if train_i % n_iter == 0:
                    sample_size = len(train_index) + 2* len(evaluation_index)
                    p = int(100 * train_i * setting.batch_size / sample_size)
                    avg_loss = train_total_loss / n_iter
                    if setting.y_transform:
                        avg_loss = std_scaler.inverse_transform(np.array(avg_loss/100).reshape(-1,1)).reshape(-1)[0]
                    logger.debug("   %dm: epoch %d [%s%s]  %d%%  loss = %.3f" % \
                          ((time() - start) // 60, epoch, "".join('#' * (p // 5)),
                           "".join(' ' * (20 - (p // 5))), p, avg_loss))
                    train_total_loss = 0
                    cur_epoch_train_loss.append(avg_loss)

            all_preds = np.concatenate(all_preds)
            all_ys = np.concatenate(all_ys)

            assert len(all_preds) == len(all_ys), "predictions and labels are in different length"
            val_train_loss = mean_squared_error(all_preds, all_ys)
            val_train_pearson = pearsonr(all_preds.reshape(-1), all_ys.reshape(-1))[0]
            val_train_spearman = spearmanr(all_preds.reshape(-1), all_ys.reshape(-1))[0]

            scheduler.step()

            ### Evaluation
            val_train_i = 0
            save_data_num = 0

            with torch.set_grad_enabled(False):

                drug_model.eval()
                all_preds = []
                all_ys = []

                # eval_train_iter = iter(eval_train_generator)
                # (pre_local_batch, pre_smiles_a, pre_smiles_b), pre_local_labels = next(eval_train_iter)
                # # drug_a_result = executor.submit(data_utils.convert_smile_to_feature,
                # #                                 pre_smiles_a, device=device("cuda:0"))
                # # drug_b_result = executor.submit(data_utils.convert_smile_to_feature,
                # #                                 pre_smiles_b, device=device("cuda:0"))
                #
                # for (cur_local_batch, cur_smiles_a, cur_smiles_b), cur_local_labels in eval_train_iter:
                #     val_train_i += 1
                #     local_labels_on_cpu = np.array(pre_local_labels).reshape(-1)
                #     sample_size = local_labels_on_cpu.shape[-1]
                #     local_labels_on_cpu = local_labels_on_cpu[:sample_size]
                #     # Transfer to GPU
                #     local_batch, local_labels = pre_local_batch.float().to(device2), pre_local_labels.float().to(device2)
                #     # local_batch = local_batch[:,:sum(slice_indices) + setting.single_repsonse_feature_length]
                #     reorder_tensor.load_raw_tensor(local_batch.contiguous().view(-1, 1, sum(slice_indices) + setting.single_repsonse_feature_length))
                #     local_batch = reorder_tensor.get_reordered_narrow_tensor()
                #     # pre_drug_a, pre_drug_b = drug_a_result.result(), drug_b_result.result()
                #     # drugs = (pre_drug_a, pre_drug_b)
                #     # drug_a_result = executor.submit(data_utils.convert_smile_to_feature,
                #     #                                 cur_smiles_a, device=device("cuda:0"))
                #     # drug_b_result = executor.submit(data_utils.convert_smile_to_feature,
                #     #                                 cur_smiles_b, device=device("cuda:0"))
                #
                #     if epoch == setting.n_epochs - 1:
                #         #### save intermediate steps results
                #         preds = best_drug_model(*local_batch)
                #         cur_train_start_index = setting.batch_size * (val_train_i - 1)
                #         cur_train_stop_index = min(setting.batch_size * (val_train_i), len(partition['test1']))
                #         for n, m in best_drug_model.named_modules():
                #             if n == "out":
                #                 catoutput = m._value_hook[0]
                #         for i, train_combination in enumerate(partition['test1'][cur_train_start_index: cur_train_stop_index]):
                #
                #             if not path.exists("train_" + setting.catoutput_output_type + "_datas"):
                #                 mkdir("train_" + setting.catoutput_output_type + "_datas")
                #             save(catoutput.narrow_copy(0,i,1), path.join("train_" + setting.catoutput_output_type + "_datas",
                #                                                        str(train_combination) + '.pt'))
                #             save_data_num += 1
                #     # preds = drug_model(*local_batch, drugs = drugs)
                #     preds = drug_model(*local_batch)
                #     preds = preds.contiguous().view(-1)
                #     assert preds.size(-1) == local_labels.size(-1)
                #     prediction_on_cpu = preds.cpu().numpy().reshape(-1)
                #     # mean_prediction_on_cpu = np.mean([prediction_on_cpu[:sample_size],
                #     #                                   prediction_on_cpu[sample_size:]], axis=0)
                #     mean_prediction_on_cpu = prediction_on_cpu[:sample_size]
                #     if setting.y_transform:
                #         local_labels_on_cpu, mean_prediction_on_cpu = \
                #             std_scaler.inverse_transform(local_labels_on_cpu.reshape(-1, 1) / 100), \
                #             std_scaler.inverse_transform(mean_prediction_on_cpu.reshape(-1, 1) / 100)
                #     all_preds.append(mean_prediction_on_cpu)
                #     all_ys.append(local_labels_on_cpu)
                #     pre_local_batch = cur_local_batch
                #     pre_local_labels = cur_local_labels
                #
                # logger.debug("saved {!r} data points".format(save_data_num))
                # all_preds = np.concatenate(all_preds)
                # all_ys = np.concatenate(all_ys)
                # assert len(all_preds) == len(all_ys), "predictions and labels are in different length"
                # val_train_loss = mean_squared_error(all_preds, all_ys)
                # val_train_pearson = pearsonr(all_preds.reshape(-1), all_ys.reshape(-1))[0]
                # val_train_spearman = spearmanr(all_preds.reshape(-1), all_ys.reshape(-1))[0]
                # if epoch == setting.n_epochs - 1 and setting.save_final_pred:
                #     save(np.concatenate((np.array(partition['test1']).reshape(-1,1), all_preds.reshape(-1,1), all_ys.reshape(-1,1)), axis=1), "prediction/prediction_" + setting.catoutput_output_type + "_training")

                all_preds = []
                all_ys = []
                val_i = 0

                validation_iter = iter(validation_generator)
                # (pre_local_batch, pre_smiles_a, pre_smiles_b), pre_local_labels = next(validation_iter)
                # drug_a_result = executor.submit(data_utils.convert_smile_to_feature,
                #                                 pre_smiles_a, device=device("cuda:0"))
                # drug_b_result = executor.submit(data_utils.convert_smile_to_feature,
                #                                 pre_smiles_b, device=device("cuda:0"))

                for (cur_local_batch, cur_smiles_a, cur_smiles_b), cur_local_labels in validation_iter:

                    val_i += 1
                    pre_local_batch = cur_local_batch
                    pre_local_labels = cur_local_labels
                    local_labels_on_cpu = np.array(pre_local_labels).reshape(-1)
                    sample_size = local_labels_on_cpu.shape[-1]
                    local_labels_on_cpu = local_labels_on_cpu[:sample_size]
                    # Transfer to GPU
                    local_batch, local_labels = pre_local_batch.float().to(device2), pre_local_labels.float().to(device2)
                    # local_batch = local_batch[:,:sum(slice_indices) + setting.single_repsonse_feature_length]
                    reorder_tensor.load_raw_tensor(local_batch.contiguous().view(-1, 1, sum(slice_indices) + setting.single_repsonse_feature_length))
                    local_batch = reorder_tensor.get_reordered_narrow_tensor()
                    # pre_drug_a, pre_drug_b = drug_a_result.result(), drug_b_result.result()
                    # drugs = (pre_drug_a, pre_drug_b)
                    # drug_a_result = executor.submit(data_utils.convert_smile_to_feature,
                    #                                 cur_smiles_a, device=device("cuda:0"))
                    # drug_b_result = executor.submit(data_utils.convert_smile_to_feature,
                    #                                 cur_smiles_b, device=device("cuda:0"))
                    # drug_a = data_utils.convert_smile_to_feature(cur_smiles_a, device=device("cuda:0"))
                    # drug_b = data_utils.convert_smile_to_feature(cur_smiles_b, device=device("cuda:0"))
                    # drugs = (drug_a, drug_b)
                    # drugs = (cur_smiles_a, cur_smiles_b)
                    # preds = drug_model(*local_batch, drugs=drugs)
                    preds = drug_model(*local_batch)
                    preds = preds.contiguous().view(-1)
                    assert preds.size(-1) == local_labels.size(-1)
                    prediction_on_cpu = preds.cpu().numpy().reshape(-1)
                    # mean_prediction_on_cpu = np.mean([prediction_on_cpu[:sample_size],
                    #                                   prediction_on_cpu[sample_size:]], axis=0)
                    mean_prediction_on_cpu = prediction_on_cpu[:sample_size]
                    if setting.y_transform:
                        local_labels_on_cpu, mean_prediction_on_cpu = \
                            std_scaler.inverse_transform(local_labels_on_cpu.reshape(-1,1) / 100), \
                            std_scaler.inverse_transform(mean_prediction_on_cpu.reshape(-1,1) / 100)
                    all_preds.append(mean_prediction_on_cpu)
                    all_ys.append(local_labels_on_cpu)
                    pre_local_batch = cur_local_batch
                    pre_local_labels = cur_local_labels

                all_preds = np.concatenate(all_preds)
                all_ys = np.concatenate(all_ys)

                assert len(all_preds) == len(all_ys), "predictions and labels are in different length"
                val_loss = mean_squared_error(all_preds, all_ys)
                val_pearson = pearsonr(all_preds.reshape(-1), all_ys.reshape(-1))[0]
                val_spearman = spearmanr(all_preds.reshape(-1), all_ys.reshape(-1))[0]

                if best_cv_pearson_score < val_pearson:
                    logger.debug('++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++')
                    logger.debug('saved a model, sample size {0!r}'.format(len(all_preds)))
                    logger.debug('++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++')
                    best_cv_pearson_score = val_pearson
                    best_drug_model.load_state_dict(drug_model.state_dict())

            logger.debug(
                "Training mse is {0}, Training pearson correlation is {1!r}, Training spearman correlation is {2!r}"
                    .format(np.mean(val_train_loss), val_train_pearson, val_train_spearman))

            logger.debug(
                "Validation mse is {0}, Validation pearson correlation is {1!r}, Validation spearman correlation is {2!r}"
                    .format(np.mean(val_loss), val_pearson, val_spearman))

            if USE_wandb:
                wandb.log({"Training mse": np.mean(val_train_loss), "Training pearson correlation": val_train_pearson,
                           "Training spearman correlation": val_train_spearman}, step=epoch)
                wandb.log({"Validation mse": np.mean(val_loss), "Validation pearson correlation": val_pearson,
                           "Validation spearman correlation": val_spearman}, step=epoch)

    ### Testing

    if setting.load_old_model:
        best_drug_model.load_state_dict(load(setting.old_model_path).state_dict())

    test_i = 0
    save_data_num = 0

    with torch.set_grad_enabled(False):

        best_drug_model.eval()
        all_preds = []
        all_ys = []

        test_iter = iter(test_generator)
        # (pre_local_batch, pre_smiles_a, pre_smiles_b), pre_local_labels = next(test_iter)
        # drug_a_result = executor.submit(data_utils.convert_smile_to_feature,
        #                                 pre_smiles_a, device=device("cuda:0"))
        # drug_b_result = executor.submit(data_utils.convert_smile_to_feature,
        #                                 pre_smiles_b, device=device("cuda:0"))

        for (cur_local_batch, cur_smiles_a, cur_smiles_b), cur_local_labels in test_iter:
            # Transfer to GPU
            test_i += 1
            pre_local_batch = cur_local_batch
            pre_local_labels = cur_local_labels
            local_labels_on_cpu = np.array(pre_local_labels).reshape(-1)
            sample_size = local_labels_on_cpu.shape[-1]
            local_labels_on_cpu = local_labels_on_cpu[:sample_size]
            local_batch, local_labels = pre_local_batch.float().to(device2), pre_local_labels.float().to(device2)
            # local_batch = local_batch[:,:sum(slice_indices) + setting.single_repsonse_feature_length]
            reorder_tensor.load_raw_tensor(local_batch.contiguous().view(-1, 1, sum(slice_indices) + setting.single_repsonse_feature_length))
            local_batch = reorder_tensor.get_reordered_narrow_tensor()
            # pre_drug_a, pre_drug_b = drug_a_result.result(), drug_b_result.result()
            # drugs = (pre_drug_a, pre_drug_b)
            # drug_a_result = executor.submit(data_utils.convert_smile_to_feature,
            #                                 cur_smiles_a, device=device("cuda:0"))
            # drug_b_result = executor.submit(data_utils.convert_smile_to_feature,
            #                                 cur_smiles_b, device=device("cuda:0"))
            # Model computations
            # drug_a = data_utils.convert_smile_to_feature(cur_smiles_a, device=device("cuda:0"))
            # drug_b = data_utils.convert_smile_to_feature(cur_smiles_b, device=device("cuda:0"))
            # drugs = (drug_a, drug_b)
            # drugs = (cur_smiles_a, cur_smiles_b)
            # preds = best_drug_model(*local_batch, drugs=drugs)
            preds = best_drug_model(*local_batch)
            preds = preds.contiguous().view(-1)
            cur_test_start_index = len(test_index_list) // 4 * (test_i-1)
            cur_test_stop_index = min(len(test_index_list) // 4 * (test_i), len(test_index_list))
            for n, m in best_drug_model.named_modules():
                if n == "out":
                    catoutput = m._value_hook[0]
            for i, test_combination in enumerate(test_index_list[cur_test_start_index: cur_test_stop_index]):
                if not path.exists("test_" + setting.catoutput_output_type + "_datas"):
                    mkdir("test_" + setting.catoutput_output_type + "_datas")
                save(catoutput.narrow_copy(0, i, 1), path.join("test_" + setting.catoutput_output_type + "_datas",
                                                               str(test_combination) + '.pt'))
                save_data_num += 1
            assert preds.size(-1) == local_labels.size(-1)
            prediction_on_cpu = preds.cpu().numpy().reshape(-1)
            mean_prediction_on_cpu = prediction_on_cpu
            if setting.y_transform:
                local_labels_on_cpu, mean_prediction_on_cpu = \
                    std_scaler.inverse_transform(local_labels_on_cpu.reshape(-1, 1) / 100), \
                    std_scaler.inverse_transform(prediction_on_cpu.reshape(-1, 1) / 100)
            all_preds.append(mean_prediction_on_cpu)
            all_ys.append(local_labels_on_cpu)
            pre_local_batch = cur_local_batch
            pre_local_labels = cur_local_labels


        logger.debug("saved {!r} data for testing dataset".format(save_data_num))
        all_preds = np.concatenate(all_preds)
        all_ys = np.concatenate(all_ys)
        assert len(all_preds) == len(all_ys), "predictions and labels are in different length"
        sample_size = len(all_preds)
        mean_prediction = np.mean([all_preds[:sample_size],
                                          all_preds[:sample_size]], axis=0)
        mean_y = np.mean([all_ys[:sample_size],
                          all_ys[:sample_size]], axis=0)

        test_loss = mean_squared_error(mean_prediction, mean_y)
        test_pearson = pearsonr(mean_y.reshape(-1), mean_prediction.reshape(-1))[0]
        test_spearman = spearmanr(mean_y.reshape(-1), mean_prediction.reshape(-1))[0]
        if not path.exists('prediction'):
            mkdir('prediction')
        save(np.concatenate((np.array(test_index_list[:sample_size]).reshape(-1,1), mean_prediction.reshape(-1, 1), mean_y.reshape(-1, 1)), axis=1),
             "prediction/prediction_" + setting.catoutput_output_type + "_testing")

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
    #kmeans = KMeans(n_clusters=len(total_data)//8, random_state=0, n_jobs = 20).fit(total_data)
    total_data = kmeans.cluster_centers_
    total_data = torch.from_numpy(total_data).float().to(device2)
    #total_data = total_data.float().to(device2)
    total_data = total_data.contiguous().view(-1, 1, sum(slice_indices) + setting.single_repsonse_feature_length)
    reorder_tensor.load_raw_tensor(total_data)
    total_data = reorder_tensor.get_reordered_narrow_tensor()

    for (local_batch, smiles_a, smiles_b), local_labels in all_data_generator:
        # Transfer to GPU
        local_batch, local_labels = local_batch.float().to(device2), local_labels.float().to(device2)
        local_batch = local_batch.contiguous().view(-1, 1, sum(slice_indices) + setting.single_repsonse_feature_length)
        reorder_tensor.load_raw_tensor(local_batch)
        local_batch = reorder_tensor.get_reordered_narrow_tensor()
        drug_a = data_utils.convert_smile_to_feature(smiles_a, device2)
        drug_b = data_utils.convert_smile_to_feature(smiles_b, device2)
        drugs = (drug_a, drug_b)
        if setting.save_feature_imp_model:
            save(best_drug_model, setting.best_model_path)
        # Model computations
        logger.debug("Start feature importances analysis")
        if setting.save_easy_input_only:
            e = shap.DeepExplainer(best_drug_model, data=list(total_data))
            input_importance = e.shap_values(list(local_batch))
            #pickle.dump(input_shap_values, open(setting.input_importance_path, 'wb+'))
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
    #batch_input_importance = np.concatenate(tuple(batch_input_importance), axis=0)
    pickle.dump(batch_input_importance, open(setting.input_importance_path, 'wb+'))

    if setting.save_out_imp:
        batch_out_input_importance = np.concatenate(tuple(batch_out_input_importance), axis=0)
        pickle.dump(batch_out_input_importance, open(setting.out_input_importance_path, 'wb+'))

    if setting.save_inter_imp:
        batch_transform_input_importance = np.concatenate(tuple(batch_transform_input_importance), axis=0)
        pickle.dump(batch_transform_input_importance, open(setting.transform_input_importance_path, 'wb+'))



if __name__ == "__main__":

    USE_wandb = False
    if USE_wandb:
        wandb.init(project="Drug combination alpha",
                   name=setting.run_dir.rsplit('/', 1)[1] + '_' + setting.data_specific[:15] + '_' + str(random_seed),
                   notes=setting.data_specific)
    else:
        environ["WANDB_MODE"] = "dryrun"

    try:
        run()
        logger.debug("new directory %s" % setting.run_dir)

    except:

        import shutil

        #shutil.rmtree(setting.run_dir)
        logger.debug("clean directory %s" % setting.run_dir)
        raise

