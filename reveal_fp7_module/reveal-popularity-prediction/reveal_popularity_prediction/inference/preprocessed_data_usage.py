__author__ = 'Georgios Rizos (georgerizos@iti.gr)'

import os

import numpy as np
from sklearn.ensemble import RandomForestRegressor
from sklearn.ensemble import GradientBoostingRegressor

from reveal_popularity_prediction.common.datarw import load_pickle


def decide_snapshot_for_learning(snapshots,
                                 platform_resources_path):
    features_folder = platform_resources_path + "features/"
    features_file_names = os.listdir(features_folder)

    features_lifetimes = list()
    for name in features_file_names:
        lifetime = float(name[:-4])
        features_lifetimes.append(lifetime)
    features_lifetimes = sorted(features_lifetimes)
    max_feature_lifetime = max(features_lifetimes)

    snapshot_lifetimes = list()
    for snapshot in snapshots:
        lifetime = snapshot["timestamp_list"][-1] - snapshot["timestamp_list"][0]
        snapshot_lifetimes.append(lifetime)
    snapshot_lifetimes = sorted(snapshot_lifetimes)
    max_snapshot_lifetime = max(snapshot_lifetimes)

    snapshot_for_learning = -1
    if max_snapshot_lifetime > max_feature_lifetime:
        counter = -1
        while True:
            snapshot_lifetime = snapshot_lifetimes[counter]
            if snapshot_lifetime > max_feature_lifetime:
                snapshot_for_learning = counter
            if counter <= 0:
                break
            counter -= 1
        if snapshot_for_learning < 0:  # For safety.
            snapshot_for_learning = 0

    return snapshot_for_learning


def get_appropriate_regressor_models(post_lifetime,
                                     platform_resources_path,
                                     number_of_threads):
    features_folder = platform_resources_path + "features/"
    features_file_names = os.listdir(features_folder)
    target_file_path = platform_resources_path + "targets.pkl"

    lifetime_to_path = dict()
    for name in features_file_names:
        lifetime = float(name[:-4])
        lifetime_to_path[lifetime] = features_folder + name

    lifetimes = list(lifetime_to_path.keys())
    lifetimes_diff = [abs(post_lifetime - lifetime) for lifetime in lifetimes]
    i = min(range(len(lifetimes_diff)), key=lifetimes_diff.__getitem__)

    features_path = lifetime_to_path[lifetimes[i]]
    features_matrix = load_pickle(features_path)
    target_matrix = load_pickle(target_file_path)

    regressor_models = train_regressor_models(features_matrix, target_matrix, number_of_threads)

    return regressor_models


def train_regressor_models(features_matrix, target_matrix, number_of_threads):
    # y_train = np.empty((features_matrix.shape[0], ), dtype=np.float64)

    # COMMENTS
    y_train = target_matrix[:, 0]
    regressor = RandomForestRegressor(n_estimators=20,
                                      criterion="mse",
                                      max_features="auto",
                                      n_jobs=number_of_threads)

    comments_regressor_fitted = regressor.fit(features_matrix, y_train)

    # USERS
    y_train = target_matrix[:, 1]
    regressor = RandomForestRegressor(n_estimators=20,
                                      criterion="mse",
                                      max_features="auto",
                                      n_jobs=number_of_threads)

    users_regressor_fitted = regressor.fit(features_matrix, y_train)

    # SCORE
    y_train = target_matrix[:, 2]
    regressor = RandomForestRegressor(n_estimators=20,
                                      criterion="mse",
                                      max_features="auto",
                                      n_jobs=number_of_threads)

    score_regressor_fitted = regressor.fit(features_matrix, y_train)

    # CONTROVERSIALITY
    y_train = target_matrix[:, 3]
    regressor = RandomForestRegressor(n_estimators=20,
                                      criterion="mse",
                                      max_features="auto",
                                      n_jobs=number_of_threads)

    controversiality_regressor_fitted = regressor.fit(features_matrix, y_train)

    # UPPER LIFETIME
    y_train = target_matrix[:, 4]
    alpha = 0.9
    upper_lifetime_regressor_fitted = GradientBoostingRegressor(loss='quantile', alpha=alpha,
                                                                n_estimators=25, max_depth=3,
                                                                learning_rate=.1, min_samples_leaf=9,
                                                                min_samples_split=9)
    upper_lifetime_regressor_fitted.fit(features_matrix, y_train)

    # LOWER LIFETIME
    y_train = target_matrix[:, 4]
    lower_lifetime_regressor_fitted = GradientBoostingRegressor(loss='quantile', alpha=1.0 - alpha,
                                                                n_estimators=25, max_depth=3,
                                                                learning_rate=.1, min_samples_leaf=9,
                                                                min_samples_split=9)
    lower_lifetime_regressor_fitted.fit(features_matrix, y_train)

    regressor_models = dict()
    regressor_models["comments"] = comments_regressor_fitted
    regressor_models["users"] = users_regressor_fitted
    regressor_models["score"] = score_regressor_fitted
    regressor_models["controversiality"] = controversiality_regressor_fitted
    regressor_models["prediction_window_upper"] = upper_lifetime_regressor_fitted
    regressor_models["prediction_window_lower"] = lower_lifetime_regressor_fitted

    return regressor_models


def popularity_prediction(features_vector, regressor_models, post_timestamp, last_comment_timestamp):
    comments_pred = regressor_models["comments"].predict(features_vector)
    users_pred = regressor_models["users"].predict(features_vector)
    score_pred = regressor_models["score"].predict(features_vector)
    controversiality_pred = regressor_models["controversiality"].predict(features_vector)

    predictions = dict()
    predictions["comments"] = comments_pred[0]
    predictions["users"] = users_pred[0]
    predictions["score"] = score_pred[0]
    predictions["controversiality"] = controversiality_pred[0]

    prediction_window_model_upper = regressor_models["prediction_window_upper"]
    prediction_window_model_lower = regressor_models["prediction_window_lower"]

    # post_lifetime = last_comment_timestamp - post_timestamp
    # post_lifetime = np.array(post_lifetime)

    prediction_upper_lifetime = prediction_window_model_upper.predict(features_vector)[0]
    prediction_lower_lifetime = prediction_window_model_lower.predict(features_vector)[0]

    prediction_window = dict()
    prediction_window["prediction_lower_timestamp"] = prediction_lower_lifetime + post_timestamp
    prediction_window["prediction_upper_timestamp"] = prediction_upper_lifetime + post_timestamp

    return predictions, prediction_window
