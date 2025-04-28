#!/usr/bin/env python

import numpy as np
import pandas as pd
import logging
from src import model, drug_drug, setting, my_data
from scipy.stats import pearsonr
from sklearn.metrics import mean_squared_error
from sklearn.preprocessing import StandardScaler
import os
from keras.callbacks import TensorBoard
from time import time

def create_drugs_profiles(raw_chemicals, genes):

    #drug_profile.columns = genes['symbol']
    # columns: drugs, index: genes
    if not setting.drug_profiles_renew and os.path.exists(setting.drug_profiles):
        drug_profile = pd.read_csv(setting.drug_profiles, index_col=0)
        return drug_profile

    drug_profile = np.zeros(shape=(len(raw_chemicals), len(genes)))
    drug_profile = pd.DataFrame(drug_profile, columns=genes['entrez'], index=raw_chemicals['Name'])
    entrez_set = set(genes['entrez'])
    for row in raw_chemicals.iterrows():

        if not isinstance(row[1]['combin_entrez'], str):
            continue

        chem_name, target_list = row[1]['Name'], row[1]['combin_entrez'].split(",")
        for target in target_list:
            target = int(target)
            if target in entrez_set:

                drug_profile.loc[chem_name, target] = 1
    print(setting.drug_profiles)
    drug_profile.T.to_csv(setting.drug_profiles)
    return drug_profile.T



if __name__ == "__main__":


    # print(simulated_drug_target, simulated_drug_target.shape)
    # print("synergy_score filtered data amount %s" %str(len(synergy_score)))
    # print(simulated_drug_target.shape, sel_dp.shape)
    # print(sel_dp)
    # print(sel_dp.shape)

    # setting up nvidia GPU environment
    if not setting.ml_train:
        os.environ["CUDA_VISIBLE_DEVICES"] = "3"
        # config = tf.ConfigProto()
        # config.gpu_options.allow_growth = True
        # set_session(tf.Session(config=config))

    # Setting up log file
    formatter = logging.Formatter(fmt='%(asctime)s %(levelname)s %(name)s %(message)s', datefmt='%m/%d/%Y %I:%M:%S')
    fh = logging.FileHandler(setting.logfile, mode='w+')
    fh.setFormatter(fmt=formatter)
    logger = logging.getLogger("Drug Combination")
    logger.addHandler(fh)
    logger.setLevel(logging.DEBUG)

    std_scaler = StandardScaler()
    logger.debug("Getting features and synergy scores ...")
    X, drug_features_len, cl_features_len, drug_features_name, cl_features_name = \
        my_data.SamplesDataLoader.Raw_X_features_prep(methods='mlp')
    Y = my_data.SamplesDataLoader.Y_features_prep()
    logger.debug("Spliting data ...")

    cv_pearsonr_scores = []
    cvmodels = []
    best_test_index, best_test_index_2 = None, None
    for train_index, test_index, test_index_2, evaluation_index, evaluation_index_2 in\
        my_data.DataPreprocessor.cv_train_eval_test_split_generator():

    # train_index, test_index, test_index_2, evaluation_index, evaluation_index_2 = \
    #     my_data.DataPreprocessor.reg_train_eval_test_split()

        logger.debug("Splitted data successfully")
        std_scaler.fit(Y[train_index])
        if setting.y_transform:
            Y = std_scaler.transform(Y) * 100

            best_test_index, best_test_index_2 = test_index, test_index_2

        if setting.ml_train:

            x_cols = [x + "_a" for x in drug_features_name] + [x + "_b" for x in drug_features_name] + cl_features_name
            X = pd.DataFrame(X, columns=x_cols)
            Y = pd.DataFrame(Y, columns=['synergy'])
            drug_drug.__ml_train(X, Y, train_index, test_index)

        else:

            drug_model = model.DrugsCombModel(drug_a_features_len=drug_features_len,
                                              drug_b_features_len=drug_features_len,
                                              cl_features_len=cl_features_len).get_model()
            logger.info("model information: \n %s" % drug_model.summary())
            logger.debug("Start training")
            tensorboard = TensorBoard(log_dir=setting.tensorboard_log)
            training_history = drug_model.fit(x=X[train_index], y=Y[train_index],
                                              validation_data=(X[test_index], Y[test_index]),
                                              epochs=setting.n_epochs,
                                              callbacks = [tensorboard],
                                              verbose=2)


            logger.debug("Training is done")
            train_prediction = drug_model.predict(x=X[train_index]).reshape((-1,))
            train_prediction = std_scaler.inverse_transform(train_prediction/100)
            Y = std_scaler.inverse_transform(Y/100)
            train_mse = mean_squared_error(Y[train_index], train_prediction)
            train_pearson = pearsonr(Y[train_index].reshape(-1), train_prediction.reshape(-1))[0]

            logger.info("training dataset: mse: %s, pearson: %s" % (str(train_mse), str(1-train_pearson**2)))

            eval_prediction = drug_model.predict(x=X[evaluation_index]).reshape((-1,))
            eval_prediction = std_scaler.inverse_transform(eval_prediction / 100)
            eval_prediction_2 = drug_model.predict(x=X[evaluation_index_2]).reshape((-1,))
            eval_prediction_2 = std_scaler.inverse_transform(eval_prediction_2 / 100)
            final_prediction = np.mean([eval_prediction, eval_prediction_2], axis=0)
            comparison = pd.DataFrame(
                {'ground_truth': Y[evaluation_index].reshape(-1), 'prediction': final_prediction.reshape(-1)})
            eval_mse = mean_squared_error(Y[evaluation_index], final_prediction)
            eval_pearson = pearsonr(Y[evaluation_index].reshape(-1), final_prediction.reshape(-1))[0]
            cv_pearsonr_scores.append(eval_pearson)

            logger.info("Evaluation dataset: mse: %s, pearson: %s" % (str(eval_mse), str(1 - eval_pearson ** 2)))

            cvmodels.append(drug_model)

    best_index = 0
    for i in range(len(cv_pearsonr_scores)):

        if cv_pearsonr_scores[best_index] < cv_pearsonr_scores[i]:
            best_index = i

    best_model = cvmodels[best_index]

    test_prediction = best_model.predict(x=X[best_test_index]).reshape((-1,))
    test_prediction = std_scaler.inverse_transform(test_prediction/100)
    test_prediction_2 = best_model.predict(x=X[best_test_index_2]).reshape((-1,))
    test_prediction_2 = std_scaler.inverse_transform(test_prediction_2/100)
    final_prediction = np.mean([test_prediction, test_prediction_2], axis=0)
    comparison = pd.DataFrame({'ground_truth':Y[best_test_index].reshape(-1),'prediction':final_prediction.reshape(-1)})
    comparison.to_csv("last_output_{!r}".format(int(time())) + ".csv")
    test_mse = mean_squared_error(Y[best_test_index], final_prediction)
    test_pearson = pearsonr(Y[best_test_index].reshape(-1), final_prediction.reshape(-1))[0]

    logger.info("Testing dataset: mse: %s, pearson: %s" % (str(test_mse), str(1-test_pearson**2)))
