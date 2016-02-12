__author__ = 'Georgios Rizos (georgerizos@iti.gr)'

import os
try:
    import cPickle as pickle
except ImportError:
    import pickle

import numpy as np
import pandas as pd
from sklearn.externals import joblib


def h5_open(path, complevel=0, complib="bzip2"):
    store = pd.HDFStore(path, complevel=0, complib="bzip2")
    return store


def h5_close(store):
    store.close()


def h5store_at(store, name, data_frame):
    store.put("data/" + name, data_frame)
    store.get_storer("data/" + name).attrs.metadata = data_frame._metadata


def h5load_from(store, name):
    data_frame = store[name]
    data_frame._metadata = store.get_storer(name).attrs.metadata
    return data_frame


def store_features(timestamp_h5_store_file,
                   handcrafted_features_h5_store_file,
                   document,
                   target_dict,
                   comment_counter,
                   timestamp_array,
                   timestamp_column_names_list,
                   handcrafted_feature_array,
                   handcrafted_feature_names_list):
    ####################################################################################################################
    # Form a pandas data frame and store the data in h5 form along with *necessary* metadata.
    ####################################################################################################################
    # Form metadata dictionary.
    discussion_metadata = target_dict

    for v in target_dict.values():
        if np.isnan(v):
            return

    # Check whether we have the appropriate number of rows for the dataframes.
    if (comment_counter + 1) != (timestamp_array.shape[0]):
        return

    # Form data frames.
    try:
        timestamp_data_frame = pd.DataFrame(timestamp_array, columns=timestamp_column_names_list)
        handcrafted_features_data_frame = pd.DataFrame(handcrafted_feature_array, columns=handcrafted_feature_names_list)
    except ValueError:
        raise RuntimeError
    handcrafted_features_data_frame._metadata = discussion_metadata

    # Store data frame along with metadata.
    h5store_at(timestamp_h5_store_file,
               "post_" + document["post_id"],
               timestamp_data_frame)
    h5store_at(handcrafted_features_h5_store_file,
               "post_" + document["post_id"],
               handcrafted_features_data_frame)


def store_sklearn_model(file_path, model):
    joblib.dump(model, file_path, compress=9)


def load_sklearn_model(file_path):
    model = joblib.load(file_path)
    return model


def load_pickle(file_path):
    """
    Unpickle some data from a given path.

    Input:  - file_path: Target file path.

    Output: - data: The python object that was serialized and stored in disk.
    """
    pkl_file = open(file_path, 'rb')
    data = pickle.load(pkl_file)
    pkl_file.close()
    return data


def get_target_value(data_frame, target_type):
    metadata = data_frame._metadata

    target_value = metadata[target_type]

    return target_value


def get_cascade_lifetime(data_frame):
    timestamp_col = data_frame["timestamp"]
    cascade_source_timestamp = timestamp_col.iloc[0]
    last_comment_timestamp = timestamp_col.iloc[-1]

    cascade_lifetime = last_comment_timestamp - cascade_source_timestamp
    return cascade_lifetime


def get_kth_row(data_frame, k, feature_list):
    row = data_frame[feature_list]
    row = row.iloc[k]

    return row


def get_kth_col(data_frame, k, feature_list):
    col = data_frame[feature_list]
    col = col.iloc[:, k]

    return col


def make_folders(top_folder, dataset_name):
    safe_make_folder(top_folder + "/results")
    safe_make_folder(top_folder + "/results/" + dataset_name + "_focus")

    safe_make_folder(top_folder + "/features")
    safe_make_folder(top_folder + "/features/dataset_full")
    safe_make_folder(top_folder + "/features/dataset_k")
    safe_make_folder(top_folder + "/features/datasetwide")
    safe_make_folder(top_folder + "/features/k_list")
    safe_make_folder(top_folder + "/features/models")


def safe_make_folder(folder):
    if not os.path.exists(folder):
        os.makedirs(folder)
