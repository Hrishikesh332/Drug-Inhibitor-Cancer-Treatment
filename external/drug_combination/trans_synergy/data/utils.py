import logging
import os
import pickle

import pandas as pd
import torch
from sklearn.model_selection import GroupKFold
import numpy as np
from trans_synergy.models.other.drug_drug import setting, logger


class TensorReorganizer:

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


def train_test_split(group_df: pd.DataFrame, group_cols: list[str], n_split: int = 5, rd_state: int = setting.split_random_seed):
    logging.debug("groupkfold split based on %s" % str(group_cols))
    groupkfold = GroupKFold(n_splits=n_split)

    groups = group_df.apply(lambda x: "_".join(list(x[group_cols])), axis = 1)
    groupkfold_instance = groupkfold.split(group_df, groups=groups)
    for _ in range(rd_state%n_split):
        next(groupkfold_instance)

    return next(groupkfold_instance)
