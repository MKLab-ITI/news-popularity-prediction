__author__ = 'Georgios Rizos (georgerizos@iti.gr)'

import os

import numpy as np
from scipy.stats import kendalltau
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.cross_validation import KFold

from news_popularity_prediction.datautil.common import get_threads_number
from news_popularity_prediction.datautil.feature_rw import h5_open, h5_close, h5load_from, h5store_at,\
    load_sklearn_model, store_sklearn_model


def random_ranking(y_train, X_test):
    pass


def initialize_k_evaluation_measures(number_of_k,
                                     number_of_folds,
                                     number_of_features):
    kendall_tau_score_array = np.empty((number_of_k, number_of_folds), dtype=np.float64)
    p_value_array = np.empty((number_of_k, number_of_folds), dtype=np.float64)

    mean_square_error = np.empty((number_of_k, number_of_folds), dtype=np.float64)

    top_k_jaccard = np.empty((number_of_k, number_of_folds), dtype=np.float64)

    feature_importances_array = np.empty([number_of_k, number_of_folds, number_of_features], dtype=np.float64)

    k_evaluation_measures = [kendall_tau_score_array,
                             p_value_array,
                             mean_square_error,
                             top_k_jaccard,
                             feature_importances_array]

    return k_evaluation_measures


def update_k_evaluation_measures(k_evaluation_measures,
                                 k_index,
                                 evaluation_measure_arrays):
    k_evaluation_measures[0][k_index, :] = evaluation_measure_arrays[0]
    k_evaluation_measures[1][k_index, :] = evaluation_measure_arrays[1]

    k_evaluation_measures[2][k_index, :] = evaluation_measure_arrays[2]
    k_evaluation_measures[3][k_index, :] = evaluation_measure_arrays[3]

    try:
        k_evaluation_measures[4][k_index, :, :] = np.squeeze(evaluation_measure_arrays[4])
    except ValueError:
        k_evaluation_measures[4][k_index, :, :] = evaluation_measure_arrays[4]

    print("Kendall's tau: ", np.mean(evaluation_measure_arrays[0]))
    print("Mean MSE: ", np.mean(evaluation_measure_arrays[2]))
    print("Top-100 Jaccard: ", np.mean(evaluation_measure_arrays[3]))


def store_k_evaluation_measures(store_path,
                                k_list,
                                k_evaluation_measures,
                                feature_column_names):
    number_of_folds = k_evaluation_measures[0].shape[1]

    h5_store = h5_open(store_path + "results.h5")

    for fold_index in range(number_of_folds):
        data_frame = pd.DataFrame(k_evaluation_measures[0][:, fold_index], columns=["kendall_tau"], index=k_list)
        h5store_at(h5_store,
                   "/data/kendall_tau/fold" + str(fold_index),
                   data_frame)

        data_frame = pd.DataFrame(k_evaluation_measures[1][:, fold_index], columns=["p_value"], index=k_list)
        h5store_at(h5_store,
                   "/data/p_value/fold" + str(fold_index),
                   data_frame)

        data_frame = pd.DataFrame(k_evaluation_measures[2][:, fold_index], columns=["mse"], index=k_list)
        h5store_at(h5_store,
                   "/data/mse/fold" + str(fold_index),
                   data_frame)

        data_frame = pd.DataFrame(k_evaluation_measures[3][:, fold_index], columns=["jaccard"], index=k_list)
        h5store_at(h5_store,
                   "/data/top_k_jaccard/fold" + str(fold_index),
                   data_frame)

        data_frame = pd.DataFrame(k_evaluation_measures[4][:, fold_index, :], columns=feature_column_names, index=k_list)
        h5store_at(h5_store,
                   "/data/feature_importances/fold" + str(fold_index),
                   data_frame)

    h5_close(h5_store)


def load_k_evaluation_measures(store_path,
                               number_of_folds=10):

    h5_store = h5_open(store_path + "results.h5")

    kendall_tau_keys = ["/data/" + "kendall_tau/fold" + str(fold_index) for fold_index in range(number_of_folds)]
    p_value_keys = ["/data/" + "p_value/fold" + str(fold_index) for fold_index in range(number_of_folds)]
    mse_keys = ["/data/" + "mse/fold" + str(fold_index) for fold_index in range(number_of_folds)]
    jaccard_keys = ["/data/" + "top_k_jaccard/fold" + str(fold_index) for fold_index in range(number_of_folds)]
    feature_importances_keys = ["/data/" + "feature_importances/fold" + str(fold_index) for fold_index in range(number_of_folds)]

    if (len(kendall_tau_keys) != len(p_value_keys)) or\
            (len(kendall_tau_keys) != len(feature_importances_keys)):
        print("Fold number different for evaluation measures load.")
        raise RuntimeError

    number_of_folds = len(feature_importances_keys)
    data_frame = h5load_from(h5_store, feature_importances_keys[0])
    k_list = data_frame.index
    number_of_samples = k_list.size
    feature_names_list = data_frame.columns
    number_of_features = len(feature_names_list)

    kendall_tau_array = np.empty((number_of_samples,
                                  number_of_folds),
                                 dtype=np.float64)
    p_value_array = np.empty((number_of_samples,
                              number_of_folds),
                             dtype=np.float64)

    mean_square_error = np.empty((number_of_samples,
                                  number_of_folds), dtype=np.float64)

    top_k_jaccard = np.empty((number_of_samples,
                              number_of_folds), dtype=np.float64)

    feature_importances_array = np.empty((number_of_samples,
                                          number_of_folds,
                                          number_of_features),
                                         dtype=np.float64)

    for f in range(number_of_folds):
        kendall_tau_key = kendall_tau_keys[f]
        p_value_key = p_value_keys[f]
        mse_key = mse_keys[f]
        jaccard_key = jaccard_keys[f]
        feature_importances_key = feature_importances_keys[f]

        kendall_tau_data_frame = h5load_from(h5_store, kendall_tau_key)
        p_value_data_frame = h5load_from(h5_store, p_value_key)
        mse_data_frame = h5load_from(h5_store, mse_key)
        jaccard_data_frame = h5load_from(h5_store, jaccard_key)
        feature_importances_data_frame = h5load_from(h5_store, feature_importances_key)

        kendall_tau_array[:, f] = np.squeeze(kendall_tau_data_frame.values)
        p_value_array[:, f] = np.squeeze(p_value_data_frame.values)
        mean_square_error[:, f] = np.squeeze(mse_data_frame.values)
        top_k_jaccard[:, f] = np.squeeze(jaccard_data_frame.values)
        try:
            feature_importances_array[:, f, :] = np.squeeze(feature_importances_data_frame.values)
        except ValueError:
            feature_importances_array[:, f, :] = feature_importances_data_frame.values

    k_evaluation_measures = (kendall_tau_array,
                             p_value_array,
                             mean_square_error,
                             top_k_jaccard,
                             feature_importances_array)

    return k_list, k_evaluation_measures, feature_names_list


def form_ground_truth(y_raw):
    return y_raw, y_raw


def initialize_evaluation_measure_arrays(number_of_folds,
                                         number_of_features):
    kendall_tau_score_array = np.empty(number_of_folds,
                                       dtype=np.float64)
    p_value_array = np.empty(number_of_folds,
                             dtype=np.float64)

    mse_array = np.empty(number_of_folds,
                         dtype=np.float64)
    jaccard_array = np.empty(number_of_folds,
                             dtype=np.float64)

    feature_importances_array = np.empty((number_of_folds,
                                          number_of_features),
                                         dtype=np.float64)
    measure_arrays_list = [kendall_tau_score_array,
                           p_value_array,
                           mse_array,
                           jaccard_array,
                           feature_importances_array]
    return measure_arrays_list


def folding(y, n_folds):
    k_fold = KFold(y.size, n_folds=n_folds, random_state=0)

    return k_fold


def learning_module(file_path,
                    X_train,
                    X_test,
                    y_train,
                    y_test,
                    train_test,
                    baseline=None):
    if baseline is None:
        regressor_fitted = get_regressor_fitted(file_path,
                                                X_train,
                                                X_test,
                                                y_train,
                                                y_test)

        y_pred = regressor_fitted.predict(X_test)

        test = train_test[1]

        max_pred_index = np.argmax(y_pred)
        max_pred_index = test[max_pred_index]

        print(max_pred_index, np.max(y_pred), np.min(y_pred))

        feature_importances = regressor_fitted.feature_importances_
    else:
        if baseline == "mean":
            y_pred = np.ones_like(y_test) * baseline_mean(y_train)
        elif baseline == "median":
            y_pred = np.ones_like(y_test) * baseline_median(y_train)
        else:
            print("Invalid baseline method.")
            raise RuntimeError

        feature_importances = np.empty((1, 0))

    # Test.
    kendall_tau_score, p_value = kendalltau(y_test, y_pred)

    mse = np.mean(np.power(y_test - y_pred, 2))
    top_k_jaccard = top_k_jaccard_score(y_test, y_pred, top_k=100)

    ranking_evaluation_tuple = [kendall_tau_score, p_value,
                                mse, top_k_jaccard,
                                feature_importances]

    return ranking_evaluation_tuple


def baseline_mean(y_train, osn=None):
    if osn is None:
        return np.mean(y_train)
    else:
        raise RuntimeError


def baseline_median(y_train, osn=None):
    if osn is None:
        return np.median(y_train)
    else:
        raise RuntimeError


def top_k_jaccard_score(x, y, top_k):
    l = x.size

    x_index = np.argsort(x)
    y_index = np.argsort(y)

    jaccard = jaccard_index(x_index[l - top_k:l], y_index[l - top_k:l])
    return jaccard


def jaccard_index(x, y):
    nom = np.intersect1d(x, y).size
    denom = np.union1d(x, y).size

    return nom/denom


def get_regressor_fitted(file_path,
                         X_train,
                         X_test,
                         y_train,
                         y_test):
    if os.path.exists(file_path):
        try:
            regressor_fitted = load_sklearn_model(file_path)
        except EOFError as e:
            print(file_path)
            raise e
    else:
        regressor = RandomForestRegressor(n_estimators=50,
                                          criterion="mse",
                                          max_features="auto",
                                          n_jobs=get_threads_number())

        regressor_fitted = regressor.fit(X_train, y_train)

        store_sklearn_model(file_path, regressor_fitted)
    return regressor_fitted


def update_evaluation_measure_arrays(evaluation_measure_arrays,
                                     fold_index,
                                     evaluation_tuple):
    evaluation_measure_arrays[0][fold_index] = evaluation_tuple[0]  # Kendall's tau
    evaluation_measure_arrays[1][fold_index] = evaluation_tuple[1]  # p-value
    evaluation_measure_arrays[2][fold_index] = evaluation_tuple[2]
    evaluation_measure_arrays[3][fold_index] = evaluation_tuple[3]

    evaluation_measure_arrays[4][fold_index, :] = evaluation_tuple[4]  # Feature weights


def is_k_valid(i,
               inv_cum_dist_dict,
               number_of_folds):
    k_is_valid = True

    for r_inv_cum_dist in inv_cum_dist_dict.values():
        discrete_r_list = list()
        for r_list in r_inv_cum_dist[-i-1:]:
            discrete_r_list.extend(r_list)
        discrete_r_list = set(discrete_r_list)
        if not len(discrete_r_list) > 50:
            k_is_valid = False
            break

    return k_is_valid
