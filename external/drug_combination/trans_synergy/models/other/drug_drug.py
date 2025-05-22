import logging


import h2o
import pandas as pd
import torch
from scipy.stats import pearsonr

import trans_synergy.settings

setting = trans_synergy.settings.get()

# Setting up log file
formatter = logging.Formatter(fmt='%(asctime)s %(levelname)s %(name)s %(message)s', datefmt='%m/%d/%Y %I:%M:%S')
fh = logging.FileHandler(setting.logfile, mode='w+')
fh.setFormatter(fmt=formatter)
logger = logging.getLogger("Drug Combination")
logger.addHandler(fh)
logger.setLevel(logging.DEBUG)


def __ml_train_model():

    if setting.estimator == 'GradientBoosting':

        from h2o.estimators import H2OGradientBoostingEstimator

        rf_drugs = H2OGradientBoostingEstimator(
            model_id="rf_drugs",
            stopping_rounds=3,
            score_each_iteration=True,
            seed=10
        )

    # setting.estimator == 'RandomForest'
    else:

        from h2o.estimators import H2ORandomForestEstimator

        rf_drugs = H2ORandomForestEstimator(
            model_id="rf_drugs",
            stopping_rounds=3,
            score_each_iteration=True,
            seed=10)

    return rf_drugs

def __ml_train(X, y, train_index, test_index):
    try:
        logger.debug("Creating h2o working environment")
        # ### Start H2O
        # Start up a 1-node H2O cloud on your local machine, and allow it to use all CPU cores and up to 2GB of memory:
        h2o.init(max_mem_size="10G", nthreads=10)
        h2o.remove_all()
        logger.debug("Created h2o working environment successfully")

        pre_h2o_df = pd.concat([X, y], axis=1)
        train_sample_size = 300 if setting.test_ml_train else len(train_index)
        h2o_drugs_train = h2o.H2OFrame(pre_h2o_df.loc[train_index[:train_sample_size], :])
        h2o_drugs_test = h2o.H2OFrame(pre_h2o_df.loc[test_index, :])

        logger.debug("Training machine learning model")
        rf_drugs = __ml_train_model()
        rf_drugs.train(x=h2o_drugs_train.col_names[:-1], y=h2o_drugs_train.col_names[-1],
                        training_frame=h2o_drugs_train)
        logger.debug("Trained successfully")

        logger.debug("Predicting training data")
        test_prediction_train = rf_drugs.predict(h2o_drugs_train[:-1])
        performance = pearsonr(test_prediction_train.as_data_frame()['predict'], h2o_drugs_train.as_data_frame()['synergy'])[0]
        logger.debug("spearman correlation coefficient for training dataset is: %f" % performance)

        logger.debug("Predicting test data")
        test_prediction = rf_drugs.predict(h2o_drugs_test[:-1])
        performance = pearsonr(test_prediction.as_data_frame()['predict'], h2o_drugs_test.as_data_frame()['synergy'])[0]
        logger.debug("spearman correlation coefficient for test dataset is: %f" % performance)

    except:
        raise

    finally:
        h2o.h2o.cluster().shutdown()


def transfer_df_to_mask(df, target_set, delete_gene = None):
    if delete_gene is not None:
        for gene in delete_gene:
            df['Entrezs'].apply(lambda x: x.discard(gene))
    mask = pd.DataFrame(columns = list(target_set))
    for i in range(len(df)):
        mask.loc[i] = pd.Series({int(x): 1 for x in df.loc[i, 'entrezs']})
    mask.fillna(0, inplace=True)
    return mask

def input_hook(module, input, _):
    setattr(module, "_value_hook", input)

