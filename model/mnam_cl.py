# -*- coding: utf-8 -*-

import time
import torch
import torch.utils.data as Data
import torch.nn.functional as F
from torch.utils.data import DataLoader
from tqdm import tqdm
import numpy as np
from model.base_tower import BaseTower
from preprocessing.inputs import combined_dnn_input, compute_input_dim, build_input_features
from layers.core import DNN
from preprocessing.utils import slice_arrays
from preprocessing.inputs import create_embedding_matrix


class MNAM_CL(BaseTower):
    def __init__(self, user_dnn_feature_columns, item_dnn_feature_columns, fair_expert_dnn_feature_columns, protected_field,
                 use_cl_loss=False, temperature=0.1, dnn_use_bn=True, dnn_hidden_units=(300, 300, 128), fair_expert_hidden_units=(), dnn_activation='relu', l2_reg_dnn=0, l2_reg_embedding=1e-6,
                 dnn_dropout=0, init_std=0.0001, seed=1024, task='binary', device='cpu', gpus=None):
        super(MNAM_CL, self).__init__(user_dnn_feature_columns, item_dnn_feature_columns,
                                      l2_reg_embedding=l2_reg_embedding, init_std=init_std, seed=seed, task=task,
                                      device=device, gpus=gpus)

        if len(user_dnn_feature_columns) > 0:
            self.user_dnn = DNN(compute_input_dim(user_dnn_feature_columns), dnn_hidden_units,
                                activation=dnn_activation, l2_reg=l2_reg_dnn, dropout_rate=dnn_dropout,
                                use_bn=dnn_use_bn, init_std=init_std, device=device)
            self.user_dnn_embedding = None

        if len(item_dnn_feature_columns) > 0:
            self.item_dnn = DNN(compute_input_dim(item_dnn_feature_columns), dnn_hidden_units,
                                activation=dnn_activation, l2_reg=l2_reg_dnn, dropout_rate=dnn_dropout,
                                use_bn=dnn_use_bn, init_std=init_std, device=device)
            self.item_dnn_embedding = None

        if len(fair_expert_dnn_feature_columns) > 0:
            # self.fair_expert_dnn = DNN(compute_input_dim(fair_expert_dnn_feature_columns), fair_expert_hidden_units,
            #                            activation=dnn_activation, l2_reg=l2_reg_dnn, dropout_rate=dnn_dropout,
            #                            use_bn=dnn_use_bn, init_std=init_std, device=device)
            self.fair_expert_dnns = torch.nn.ModuleList(
                [DNN(compute_input_dim(fair_expert_dnn_feature_columns), fair_expert_hidden_units,
                     activation=dnn_activation, l2_reg=l2_reg_dnn, dropout_rate=dnn_dropout,
                     use_bn=dnn_use_bn, init_std=init_std, device=device) for i in range(len(protected_field))]
            )
            self.fair_dnn_feature_columns = fair_expert_dnn_feature_columns
            self.fair_embedding_dict = create_embedding_matrix(self.fair_dnn_feature_columns, init_std,
                                                               sparse=False, device=device)
            self.fair_expert_dnn_embeddings = None
            self.protected_field = protected_field
            self.add_cl_fair_loss = use_cl_loss  # for contrastive loss

        self.feature_index = build_input_features(user_dnn_feature_columns + item_dnn_feature_columns + protected_field)
        print("feature_index: ", self.feature_index)

        self.l2_reg_embedding = l2_reg_embedding
        self.seed = seed
        self.task = task
        self.device = device
        self.gpus = gpus
        self.temp = temperature

    def forward(self, inputs):
        if len(self.user_dnn_feature_columns) > 0:
            user_sparse_embedding_list, user_dense_value_list = \
                self.input_from_feature_columns(inputs, self.user_dnn_feature_columns, self.user_embedding_dict)

            user_dnn_input = combined_dnn_input(user_sparse_embedding_list, user_dense_value_list)
            self.user_dnn_embedding = self.user_dnn(user_dnn_input)

        if len(self.item_dnn_feature_columns) > 0:
            item_sparse_embedding_list, item_dense_value_list = \
                self.input_from_feature_columns(inputs, self.item_dnn_feature_columns, self.item_embedding_dict)

            item_dnn_input = combined_dnn_input(item_sparse_embedding_list, item_dense_value_list)
            self.item_dnn_embedding = self.item_dnn(item_dnn_input)

        if len(self.fair_dnn_feature_columns) > 0:
            fair_sparse_embedding_list, fair_dense_value_list = \
                self.input_from_feature_columns(inputs, self.fair_dnn_feature_columns, self.fair_embedding_dict)
            fair_expert_dnn_input = combined_dnn_input(fair_sparse_embedding_list, fair_dense_value_list)

            _, protected_dense_field = \
                self.input_from_feature_columns(inputs, self.protected_field, {})
            self.protected_field_input = [combined_dnn_input(sparse_embedding_list=[], dense_value_list=[i]) for i in protected_dense_field]
            self.fair_expert_dnn_embeddings = [self.fair_expert_dnns[i](fair_expert_dnn_input) for i in range(len(self.protected_field))]

        if len(self.user_dnn_feature_columns) > 0 and len(self.item_dnn_feature_columns) > 0 and len(self.fair_dnn_feature_columns) > 0:
            main_logits = self.user_dnn_embedding * self.item_dnn_embedding
            fair_logits = []
            for i in range(len(self.protected_field)):
                fair_logits.append(self.fair_expert_dnn_embeddings[i] * self.protected_field_input[i])
            total_fair_logits = torch.sum(torch.concat(fair_logits, dim=1), dim=1).unsqueeze(dim=1)
            score = torch.sum(torch.concat([main_logits, total_fair_logits], dim=1), dim=1)
            # add contrastive fair loss
            if self.add_cl_fair_loss:
                random_factors = self.generate_augmented_sample()
                cl_fair_score = self.calculate_augmented_logits(main_logits, fair_logits, random_factors)
                return self.out(score), self.out(torch.concat(cl_fair_score, dim=0)), torch.concat(random_factors, dim=0)

            output = self.out(score)
            return output

        elif len(self.user_dnn_feature_columns) > 0:
            return self.user_dnn_embedding

        elif len(self.item_dnn_feature_columns) > 0:
            return self.item_dnn_embedding

        else:
            raise Exception("input Error! user and item feature columns are empty.")

    def fit(self, x=None, y=None, batch_size=None, epochs=1, verbose=1, initial_epoch=0, validation_split=0.,
            validation_data=None, shuffle=True, callbacks=None):
        if isinstance(x, dict):
            x = [x[feature] for feature in self.feature_index]

        do_validation = False
        if validation_data:
            do_validation = True
            if len(validation_data) == 2:
                val_x, val_y = validation_data
            elif len(validation_data) == 3:
                val_x, val_y, val_sample_weight = validation_data
            else:
                raise ValueError(
                    'When passing a `validation_data` argument, '
                    'it must contain either 2 items (x_val, y_val), '
                    'or 3 items (x_val, y_val, val_sample_weights), '
                    'or alternatively it could be a dataset or a '
                    'dataset or a dataset iterator. '
                    'However we received `validation_data=%s`' % validation_data)

            if isinstance(val_x, dict):
                val_x = [val_x[feature] for feature in self.feature_index]

        elif validation_split and 0 < validation_split < 1.:
            do_validation = True
            if hasattr(x[0], 'shape'):
                split_at = int(x[0].shape[0] * (1. - validation_split))
            else:
                split_at = int(len(x[0]) * (1. - validation_split))

            x, val_x = (slice_arrays(x, 0, split_at),
                        slice_arrays(x, split_at))
            y, val_y = (slice_arrays(y, 0, split_at),
                        slice_arrays(y, split_at))

        else:
            val_x = []
            val_y = []

        for i in range(len(x)):
            if len(x[i].shape) == 1:
                x[i] = np.expand_dims(x[i], axis=1)

        train_tensor_data = Data.TensorDataset(torch.from_numpy(
            np.concatenate(x, axis=-1)), torch.from_numpy(y))
        if batch_size is None:
            batch_size = 256

        model = self.train()
        print(model)
        loss_func = self.loss_func
        optim = self.optim

        if self.gpus:
            print('parallel running on these gpus:', self.gpus)
            model = torch.nn.DataParallel(model, device_ids=self.gpus)
            batch_size *= len(self.gpus)  # input `batch_size` is batch_size per gpu
        else:
            print(self.device)

        train_loader = DataLoader(dataset=train_tensor_data, shuffle=shuffle, batch_size=batch_size)

        sample_num = len(train_tensor_data)
        steps_per_epoch = (sample_num - 1) // batch_size + 1

        # train
        print("Train on {0} samples, validate on {1} samples, {2} steps per epoch".format(
            len(train_tensor_data), len(val_y), steps_per_epoch))
        for epoch in range(initial_epoch, epochs):
            epoch_logs = {}
            start_time = time.time()
            loss_epoch = 0
            total_loss_epoch = 0
            train_result = {}
            main_loss_epoch = 0

            with tqdm(enumerate(train_loader), disable=verbose != 1) as t:
                for _, (x_train, y_train) in t:
                    x = x_train.to(self.device).float()
                    y = y_train.to(self.device).float()

                    if self.add_cl_fair_loss:
                        y_pred, fair_pred, factor = model(x)
                        y_pred = y_pred.squeeze()
                    else:
                        y_pred = model(x).squeeze()
                    optim.zero_grad()
                    loss = loss_func(y_pred, y.squeeze(), reduction='mean')
                    reg_loss = self.get_regularization_loss()

                    total_loss = loss + reg_loss
                    main_loss_epoch += total_loss.item()
                    if self.add_cl_fair_loss:
                        total_loss += self.add_contrastive_fair_loss(y_pred, fair_pred, factor, temp=self.temp)

                    loss_epoch += loss.item()
                    total_loss_epoch += total_loss.item()
                    total_loss.backward()
                    optim.step()

                    if verbose > 0:
                        for name, metric_fun in self.metrics.items():
                            if name not in train_result:
                                train_result[name] = []
                            train_result[name].append(metric_fun(
                                y.cpu().data.numpy(), y_pred.cpu().data.numpy().astype('float64')
                            ))

            # add epoch_logs
            epoch_logs["loss"] = total_loss_epoch / sample_num
            epoch_logs["main_loss"] = main_loss_epoch / sample_num
            for name, result in train_result.items():
                epoch_logs[name] = np.sum(result) / steps_per_epoch

            if do_validation:
                eval_result = self.evaluate(val_x, val_y, batch_size)
                for name, result in eval_result.items():
                    epoch_logs["val_" + name] = result

            if verbose > 0:
                epoch_time = int(time.time() - start_time)
                print('Epoch {0}/{1}'.format(epoch + 1, epochs))

                eval_str = "{0}s - loss: {1: .5f} - main_loss: {2: .5f}".format(epoch_time, epoch_logs["loss"], epoch_logs["main_loss"])

                for name in self.metrics:
                    eval_str += " - " + name + ": {0: .4f} ".format(epoch_logs[name]) + " - " + \
                                "val_" + name + ": {0: .4f}".format(epoch_logs["val_" + name])
                print(eval_str)
            if self.stop_training:
                break

    def predict(self, x, batch_size=256):
        model = self.eval()
        if isinstance(x, dict):
            x = [x[feature] for feature in self.feature_index]
        for i in range(len(x)):
            if len(x[i].shape) == 1:
                x[i] = np.expand_dims(x[i], axis=1)

        tensor_data = Data.TensorDataset(
            torch.from_numpy(np.concatenate(x, axis=-1))
        )
        test_loader = DataLoader(
            dataset=tensor_data, shuffle=False, batch_size=batch_size
        )
        pred_ans = []
        with torch.no_grad():
            for _, x_test in enumerate(test_loader):
                x = x_test[0].to(self.device).float()
                if self.add_cl_fair_loss:
                    y_pred = model(x)[0].cpu().data.numpy()
                else:
                    y_pred = model(x).cpu().data.numpy()
                pred_ans.append(y_pred)
        return np.concatenate(pred_ans).astype("float64")

    def add_contrastive_fair_loss(self, logits, fair_logits, factor, temp=1.0):
        cl_total_logits = logits.unsqueeze(dim=1).repeat(len(self.protected_field), 1)
        total_fair_loss = torch.zeros((1,), device=self.device)
        label = torch.gt(factor - torch.ones_like(factor), 0.0).float()
        logits_diff = torch.sigmoid(fair_logits - cl_total_logits)
        total_fair_loss += F.binary_cross_entropy(logits_diff, label)
        return total_fair_loss * temp

    def generate_augmented_sample(self, lower=0.7, upper=1.3):
        adjust_factors = []
        for i in range(len(self.protected_field)):
            field = self.protected_field[i]  # DenseFeat
            random_factor = (upper - lower) * torch.rand(self.protected_field_input[i].shape) + lower
            if field.name == 'item_mean_rating':  # limit: [1, 5]
                adjusted_field_value = self.protected_field_input[i] * random_factor
                lower_mask, upper_mask = torch.lt(adjusted_field_value, 1), torch.gt(adjusted_field_value, 5)
                adjusted_field_value[lower_mask] = 1  # lower bound
                adjusted_field_value[upper_mask] = 5  # upper bound
                adjust_factors.append(adjusted_field_value / self.protected_field_input[i])
            elif field.name == 'price':
                adjust_factors.append(random_factor)  # price has no hard lower/upper bound
        return adjust_factors

    def calculate_augmented_logits(self, main_logtis, fair_logits, random_factors):
        cl_fair_scores = []
        for i in range(len(fair_logits)):
            adjust_logit = fair_logits[i] * random_factors[i]
            cl_fair_scores.append(torch.sum(
                torch.concat([main_logtis, adjust_logit], dim=1), dim=1).unsqueeze(dim=1))
        return cl_fair_scores
