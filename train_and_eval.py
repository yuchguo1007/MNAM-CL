import torch
import argparse
import pandas as pd
import numpy as np
from sklearn.metrics import log_loss, roc_auc_score
from preprocessing.feature import Feature
from preprocessing.movielens import Movielens
from sklearn.preprocessing import LabelEncoder
from preprocessing.utils import get_configs
from model.mnam_cl import MNAM_CL


def parse_args():
    parser = argparse.ArgumentParser(description="Simple training and evaluation script for MNAM-CL.")
    parser.add_argument(
        "--config_file",
        type=str,
        default="config/fair_movielens.json",
        required=False,
        help="Training config."
    )

    parser.add_argument(
        "--nrows",
        type=int,
        default=None,
        required=False,
        help="Number of rows for training data."
    )

    parser.add_argument(
        "--sample_rate",
        type=float,
        default=1.,
        required=False,
        help="Sample rate of training data."
    )

    args = parser.parse_args()
    return args


def evaluate_monotonic_fairness(test, model, protected_fields, feature_fields):
    # generate monotonic fairness evaluation data
    for field in protected_fields:
        random_test = test.copy()
        random_test['random'] = np.random.uniform(0.7, 1.3, random_test.shape[0])

        if field == 'item_mean_rating':

            random_test[field] = random_test.apply(lambda x: random_value_with_bound(x['random'] * x[field], lower=1, upper=5), axis=1)
        elif field == 'price':
            random_test[field] = random_test['random'] * random_test[field]

        test_random_input = {name: random_test[name] for name in feature_fields}
        pred_rnd = model.predict(test_random_input, batch_size=2000)

        assert len(test["pred"]) == len(pred_rnd) == len(random_test['random'])
        irrelevance, reverse = 0, 0
        for idx, value in enumerate(random_test['random']):
            score_diff = pred_rnd[idx] - test["pred"][idx]
            if abs(score_diff) <= 1e-6:
                irrelevance += 1
            # if the correlation between protected attrbutes and the target is negative, the protected feature should be inversed
            elif score_diff * (value - 1) < 0:
                reverse += 1

        print('Unfairness rate of %s - irrelevant: %.8f' % (field, irrelevance / len(test["pred"])))
        print('Unfairness rate of %s - reverse: %.8f' % (field, reverse / len(test["pred"])))


def random_value_with_bound(x, lower, upper):
    if x < lower:
        return lower
    elif x > upper:
        return upper
    else:
        return x


def main():
    device = 'cpu'

    args = parse_args()
    print(f"model using configs from {args.config_file}")
    cfg = get_configs(args.config_file)

    # get features config
    f = Feature(cfg)
    names = f.get_feature_names()
    feature_mapper, model_mapper = cfg['features'], cfg['model']

    data_path = cfg["data"]["raw_data_path"]
    dataset = cfg["data"]["dataset_name"]

    if dataset == "movielens":
        d = Movielens(data_path)
    else:
        raise Exception(f"Unsupported dataset: {dataset}")

    sparse_features, dense_features = f.get_sparse_dense_feature_names()
    target = f.get_target_names()

    sparse_feature_mapper = f.get_feature_nunique(path=feature_mapper['nunique_path'],
                                                  mode=feature_mapper['nunique_mode'],
                                                  df=d.data, names=sparse_features)
    print("sparse_feature_mapper", sparse_feature_mapper)

    model_mapper = cfg['model']
    user_feature_columns, item_feature_columns = f.get_user_item_feature_columns(sparse_feature_mapper,
                                                                                 feature_mapper['protected_field'])
    protected_field = f.get_protected_feature_columns(feature_mapper['protected_field'])
    model = MNAM_CL(user_feature_columns, item_feature_columns, item_feature_columns, protected_field,
                    use_cl_loss=True, temperature=model_mapper["cl_temperature"],
                    dnn_use_bn=False, dnn_hidden_units=model_mapper['dnn_hidden_units'],
                    fair_expert_hidden_units=model_mapper['fair_expert_hidden_units'],
                    dnn_activation='sigmoid', task='binary',
                    device=device)

    # Train model
    train, test, data = d.data_process(data_path, nrows=args.nrows, sample_rate=1., columns=names)

    train = d.get_item_feature(d.get_user_feature(train))
    pd.set_option('display.max_columns', None)
    print(train.dtypes)
    test = d.enrich_test_feature(test, train)
    print("test shape after enriching feature: ", test.shape)

    for feat in sparse_features:
        lbe = LabelEncoder()
        lbe.fit(data[feat])
        train[feat] = lbe.transform(train[feat])
        test[feat] = lbe.transform(test[feat])

    train_model_input = {name: train[name] for name in sparse_features + dense_features}

    model.compile("adam", "binary_crossentropy", metrics=['auc', 'accuracy'])

    model.fit(train_model_input, train[target].values, batch_size=cfg["model"]["batch_size"], epochs=cfg["model"]["epochs"], verbose=1,
              validation_split=0.2)

    eval_tr = model.evaluate(train_model_input, train[target].values)
    print("Train metrics", eval_tr)

    if model_mapper['save_model']:
        torch.save(model.state_dict(), model_mapper['model_path'])

    if model_mapper['save_test']:
        test.to_csv(cfg["data"]["test_data_path"], sep='\t', header=True, index=False)

    # Evaluation
    test_model_input = {name: test[name] for name in sparse_features + dense_features}
    test["pred"] = model.predict(test_model_input, batch_size=2000)
    print("Test LogLoss", round(log_loss(test[target].values, test["pred"]), 4))
    print("Test AUC", round(roc_auc_score(test[target].values, test["pred"]), 4))

    evaluate_monotonic_fairness(test, model, feature_mapper['protected_field'], sparse_features + dense_features)


if __name__ == '__main__':
    main()
