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
    """
    Returns an h5 file store handle managed via pandas.

    :param path: The path of the h5 file store.
    :param complevel: Compression level (0-9).
    :param complib: Library used for compression.
    :return: store: The h5 file store handle.
    """
    store = pd.HDFStore(path, complevel=0, complib="bzip2")
    return store


def h5_close(store):
    """
    Safely closes an h5 file store handle.

    :param store: The h5 file store handle.
    :return: None
    """
    store.close()


def h5store_at(store, name, data_frame):
    """
    Stores a pandas data frame in an already opened h5 store file, including data frame metadata.

    :param store: The h5 file store handle.
    :param name: The key name given to the data frame in the h5 store file.
    :param data_frame: The pandas data frame to be stored.
    :return: None
    """
    store.put(name, data_frame)
    store.get_storer(name).attrs.metadata = data_frame._metadata


def h5load_from(store, name):
    """
    Loads a pandas data frame from an already opened h5 store file, including data frame metadata.

    :param store: The h5 file store handle.
    :param name: The key name given to the data frame in the h5 store file.
    :return: data_frame: The pandas data frame to be loaded.
    """
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
    """
    Forms a pandas data frame and stores the features in h5 format along with *necessary* target metadata.

    :param timestamp_h5_store_file: Contains data frames of the raw and preprocessed comment timestamps.
    :param handcrafted_features_h5_store_file: Contains data frames of the engineered features for the discussions.
    :param document: A dictionary containing the documents discussion-related fields (initial_post, comments etc).
    :param target_dict: A dictionary containing the prediction target values.
    :param comment_counter: This is the number of comments that were found when preprocessing the data.
    :param timestamp_array: The numpy array containing timestamps to be stored as a data frame in an h5 store file.
    :param timestamp_column_names_list: The column names for the timestamp data frame.
    :param handcrafted_feature_array: The numpy array containing features to be stored as a data frame in an h5 store file.
    :param handcrafted_feature_names_list: The column names for the engineered features data frame.
    :return: None
    """
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
               "/data/post_" + document["post_id"],
               timestamp_data_frame)
    h5store_at(handcrafted_features_h5_store_file,
               "/data/post_" + document["post_id"],
               handcrafted_features_data_frame)


def store_sklearn_model(file_path, model):
    """
    Stores the compressed regression sklearn model for reuse.

    :param file_path: The file path to store the model.
    :param model: The sklearn model.
    :return: None
    """
    joblib.dump(model, file_path, compress=9)


def load_sklearn_model(file_path):
    """
    Loads the compressed regression sklearn model.

    :param file_path: The file path where the model is stored.
    :return: The sklearn model.
    """
    model = joblib.load(file_path)
    return model


def get_target_value(data_frame, target_type):
    """
    Get the value of a specific target from the engineered features pandas data frame.

    :param data_frame: The engineered features pandas data frame.
    :param target_type: The name of the prediction target.
    :return: The prediction target value.
    """
    metadata = data_frame._metadata

    target_value = metadata[target_type]

    return target_value


def make_folders(top_folder, dataset_name):
    """
    Make all necessary folders for reproducing the experiments.

    :param top_folder: The top folder that needs to be defined by the user.
    :param dataset_name: The name of the news-based dataset.
    :return: None
    """
    safe_make_folder(top_folder + "/results")
    safe_make_folder(top_folder + "/results/" + dataset_name)

    safe_make_folder(top_folder + "/features")
    safe_make_folder(top_folder + "/features/dataset_full")
    safe_make_folder(top_folder + "/features/dataset_k")
    safe_make_folder(top_folder + "/features/datasetwide")
    safe_make_folder(top_folder + "/features/k_list")
    safe_make_folder(top_folder + "/features/models")
    safe_make_folder(top_folder + "/features/models/" + dataset_name)


def safe_make_folder(folder):
    """
    A utility function that sefely makes a new folder if it does not exist.

    :param folder: Folder to be made.
    :return: None
    """
    if not os.path.exists(folder):
        os.makedirs(folder)


def get_kth_row(data_frame, k, feature_list):
    row = data_frame[feature_list]
    row = row.iloc[k]

    return row


def get_kth_col(data_frame, k, feature_list):
    col = data_frame[feature_list]
    col = col.iloc[:, k]

    return col
