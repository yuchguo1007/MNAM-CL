# -*- coding: utf-8 -*-

"""
@author: kyriezhao <kyriezhao@tencent.com>
@file: feature.py
@time: 2023/3/2 3:33 下午
"""

import json
import pickle
from preprocessing.inputs import SparseFeat, DenseFeat


class Feature:
    def __init__(self, cfg=None):
        self.user_sparse_features = 'user_sparse_features'
        self.user_dense_features = 'user_dense_features'

        self.item_sparse_features = 'item_sparse_features'
        self.item_dense_features = 'item_dense_features'

        self.nunique = {}
        self.user_sparse_feature_list = []
        self.user_dense_feature_list = []
        self.item_sparse_feature_list = []
        self.item_dense_feature_list = []

        self.target = 'target'
        if cfg:
            self.feature_cfg = cfg["features"]

    def get_feature_names(self):
        names = []
        for i in self.__dict__.values():
            if isinstance(i, str):
                names.extend(self.feature_cfg[i])
        return names

    def get_sparse_dense_feature_names(self):
        self.user_sparse_feature_list, self.user_dense_feature_list = self.get_user_sparse_dense_feature_names()
        self.item_sparse_feature_list, self.item_dense_feature_list = self.get_item_sparse_dense_feature_names()
        sparse_features = self.user_sparse_feature_list + self.item_sparse_feature_list
        dense_features = self.user_dense_feature_list + self.item_dense_feature_list
        return sparse_features, dense_features

    def get_user_sparse_dense_feature_names(self):
        return self.feature_cfg['user_sparse_features'], self.feature_cfg['user_dense_features']

    def get_item_sparse_dense_feature_names(self):
        return self.feature_cfg['item_sparse_features'], self.feature_cfg['item_dense_features']

    def get_user_item_feature_columns(self, sparse_feature_mapper, protected_col=None):
        user_feature_columns = [SparseFeat(feat, sparse_feature_mapper[feat], embedding_dim=4) for feat in self.user_sparse_feature_list]\
                               + [DenseFeat(feat, 1) for feat in self.user_dense_feature_list]
        if not protected_col:
            item_feature_columns = [SparseFeat(feat, sparse_feature_mapper[feat], embedding_dim=4) for feat in self.item_sparse_feature_list]\
                                   + [DenseFeat(feat, 1) for feat in self.item_dense_feature_list]
        else:
            item_feature_columns = [SparseFeat(feat, sparse_feature_mapper[feat], embedding_dim=4) for feat in self.item_sparse_feature_list]\
                                   + [DenseFeat(feat, 1) for feat in self.item_dense_feature_list if feat not in protected_col]
        return user_feature_columns, item_feature_columns

    def get_protected_feature_columns(self, protected_cols):
        protected_field = [DenseFeat(col, 1) for col in protected_cols]
        return protected_field

    def get_target_names(self):
        return self.feature_cfg['target']

    def get_feature_names_from_file(self, file_name):
        fp = open(file_name)
        my_cls = json.load(fp)
        names = []
        for i in self.__dict__.values():
            if isinstance(i, str):
                names.extend(my_cls['features'][i])
        return my_cls, names

    def get_feature_nunique(self, path, mode, df=None, names=None):
        if mode == 'save':
            nunique = {}
            for name in names:
                nunique[name] = df[name].nunique() + 1
            fp = open(path, 'wb')
            pickle.dump(nunique, fp, protocol=pickle.HIGHEST_PROTOCOL)
            self.nunique = nunique
            return nunique
        elif mode == 'load':
            print("load sparse feature mapper from file...")
            fp = open(path, 'rb')
            self.nunique = pickle.load(fp)
            return self.nunique
        else:
            raise Exception("input Error! Use 'save' or 'load' for mode")


if __name__ == '__main__':
    # feature = Feature()
    Feature().get_feature_names_from_file(file_name='../config/steam_video_game.json')
