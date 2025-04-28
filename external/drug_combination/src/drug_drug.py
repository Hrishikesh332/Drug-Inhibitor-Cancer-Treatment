import pandas as pd
from src import setting
from sklearn.model_selection import ShuffleSplit, GroupKFold
from scipy.stats import pearsonr
import logging
import os
import pickle
#from pandas.io.common import EmptyDataError
import torch

# Setting up log file
formatter = logging.Formatter(fmt='%(asctime)s %(levelname)s %(name)s %(message)s', datefmt='%m/%d/%Y %I:%M:%S')
fh = logging.FileHandler(setting.logfile, mode='w+')
fh.setFormatter(fmt=formatter)
logger = logging.getLogger("Drug Combination")
logger.addHandler(fh)
logger.setLevel(logging.DEBUG)

class reorganize_tensor:

    raw_tensor = None
    def __init__(self, slice_indices, arrangement, dimension):

        self.slice_indices = slice_indices
        self.arrangement = arrangement
        self.dimension = dimension

    def load_raw_tensor(self, raw_tensor):
        self.raw_tensor = raw_tensor

    @classmethod
    def recursive_len(cls, item):
        if type(item) == list:
            return sum(cls.recursive_len(subitem) for subitem in item)
        else:
            return 1

    def get_feature_list_names(self, flatten = False):

        whole_list_names = [x +'_a' for x in setting.drug_features] + [x + '_b' for x in setting.drug_features] \
                           + [x for x in setting.cellline_features]
        result_names = []
        for ls in self.arrangement:
            cur_len = self.slice_indices[ls[0]]
            for i in ls:
                assert self.slice_indices[i] == cur_len, "concatenated tensor has different dimensions"
            result_names.append([whole_list_names[ls[-1]]])
        if flatten:
            result_names = [x for sublist in result_names for x in sublist]

        return result_names

    def get_features_names(self, flatten=False):

        whole_list_names = [x + '_a' for x in setting.drug_features] + [x + '_b' for x in setting.drug_features] \
                           + [x for x in setting.cellline_features] + [x for x in setting.single_response_feature]
        result_names = []
        for ls in self.arrangement:
            cur_len = self.slice_indices[ls[0]]
            for i in ls:
                assert self.slice_indices[i] == cur_len, "concatenated tensor has different dimensions"
            result_names.append(
                [whole_list_names[ls[-1]] + '_' + str(j) for j in range(self.slice_indices[ls[-1]])])
        result_names.append([whole_list_names[-1] + '_' + str(j) for j in range(
            setting.single_repsonse_feature_length)])
        if flatten:
            result_names = [x for sublist in result_names for x in sublist]

        return result_names

    def get_reordered_slice_indices(self):

        ### slice_indices: [2324, 400, 1200, 2324, 400, 1200, 2324]
        ### arrangement: [[0, 3, 6, 6], [1, 4], [2, 5]]
        ### return: [2324+2324+2324+2324, 400+400, 1200+1200]

        # assert len(self.slice_indices) == self.recursive_len(self.arrangement), \
        #     "slice indices length is not same with arrangement length"

        result_slice_indices = []
        for ls in self.arrangement:
            cur_len = self.slice_indices[ls[0]]
            for i in ls:
                assert self.slice_indices[i] == cur_len, "concatenated tensor has different dimensions"
            result_slice_indices.append(sum([self.slice_indices[i] for i in ls]))

        return result_slice_indices

    def __accum_slice_indices(self):

        result_slice_indices = [0]
        for i in range(1, len(self.slice_indices)):
            result_slice_indices.append(result_slice_indices[-1] + self.slice_indices[i-1])
        return result_slice_indices

    def get_reordered_narrow_tensor(self):

        ### arrangement: [[0, 3, 6], [1, 4], [2, 5]]

        # assert len(self.slice_indices) == self.recursive_len(self.arrangement), \
        #     "slice indices length is not same with arrangement length"

        assert self.raw_tensor is not None, "Raw tensor should be loaded firstly"

        result_tensors = []
        cat_tensor_list = []
        start_indices = self.__accum_slice_indices()
        for ls in self.arrangement:
            cur_len = self.slice_indices[ls[0]]
            for index in ls:
                assert self.slice_indices[index] == cur_len, "concatenated tensor has different dimensions"
                cat_tensor_list.append(self.raw_tensor.narrow_copy(dim=self.dimension, start=start_indices[index],
                                                              length=self.slice_indices[index]))
            catted_tensor = torch.cat(tuple(cat_tensor_list), dim=1)
            result_tensors.append(catted_tensor)
            cat_tensor_list = []
        if setting.single_repsonse_feature_length != 0:
            single_response_feature = self.raw_tensor.narrow_copy(dim = self.dimension,
                                                                  start=start_indices[-1] + self.slice_indices[-1],
                                                                  length=setting.single_repsonse_feature_length)
            result_tensors.append(single_response_feature)
        return result_tensors

def narrowed_tensors(raw_tensor, slice_indexs, dimension):

    result_tensors = []
    current_index = 0
    for length in slice_indexs:
        result_tensors.append(raw_tensor.narrow_copy(dimension, current_index, length))
        current_index += length
    assert current_index == list(raw_tensor.size())[dimension], "narrowed tensors didn't use all raw tensor data"
    return result_tensors

def regular_split(df, group_df = None, group_col=None, n_split = 10, rd_state = setting.split_random_seed):

    shuffle_split = ShuffleSplit(test_size=1.0/n_split, random_state = rd_state)
    return next(shuffle_split.split(df))

def split_data(df, group_df = None, group_col = None):

    logger.debug("Splitting dataset to training dataset and testing dataset based on genes")
    if not setting.index_renewal and (os.path.exists(setting.train_index) and os.path.exists(setting.test_index)):
        train_index = pickle.load(open(setting.train_index, "rb"))
        test_index = pickle.load(open(setting.test_index, "rb"))
    else:
        train_index, test_index = drugs_combo_split(df, group_df, group_col)

        with open(setting.train_index, 'wb') as train_file:
                pickle.dump(train_index, train_file)
        with open(setting.test_index, 'wb') as test_file:
                pickle.dump(test_index, test_file)

    logger.debug("Splitted data successfully")

    return train_index, test_index

def drugs_combo_split(df, group_df, group_col, n_split = 5, rd_state = setting.split_random_seed):

    if group_df is None:
        logging.debug("group df should not be empty")
        raise EmptyDataError

    logging.debug("groupkfold split based on %s" % str(group_col))
    groupkfold = GroupKFold(n_splits=n_split)

    groups = group_df.apply(lambda x: "_".join(list(x[group_col])), axis = 1)
    groupkfold_instance = groupkfold.split(group_df, groups=groups)
    for _ in range(rd_state%n_split):
        next(groupkfold_instance)

    return next(groupkfold_instance)


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

    import h2o

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

def input_hook(module, input, output):
    setattr(module, "_value_hook", input)

