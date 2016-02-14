__author__ = 'Georgios Rizos (georgerizos@iti.gr)'

import os
import statistics

import numpy as np

from news_popularity_prediction.datautil.feature_rw import h5load_from, get_kth_row, h5_open


def calculate_comparison_lifetimes(features_folder,
                                   osn_focus):
    if osn_focus is None:
        osn_focus = "post"

    # This is a list of all the .h5 files as produced after preprocessing.
    h5_store_file_name_list = os.listdir(features_folder)
    h5_store_file_name_list = [h5_store_file_name for h5_store_file_name in sorted(h5_store_file_name_list) if not h5_store_file_name[-1] == "~"]

    timestamp_h5_store_file_name_list = [name for name in h5_store_file_name_list if "timestamp" in name]
    handcrafted_features_h5_store_file_name_list = [name for name in h5_store_file_name_list if "handcrafted" in name]

    timestamp_h5_store_file_path_list = [features_folder + "/" + h5_store_file_name for h5_store_file_name in timestamp_h5_store_file_name_list]
    handcrafted_features_h5_store_file_path_list = [features_folder + "/" + h5_store_file_name for h5_store_file_name in handcrafted_features_h5_store_file_name_list]

    file_path_list_zip = zip(timestamp_h5_store_file_path_list,
                             handcrafted_features_h5_store_file_path_list)

    h5_stores_and_keys = list()
    for file_paths in file_path_list_zip:
        timestamp_h5_store_file = h5_open(file_paths[0])
        handcrafted_features_h5_store_file = h5_open(file_paths[1])

        keys_dict = dict()
        keys_dict[osn_focus] = sorted((key for key in timestamp_h5_store_file.keys() if osn_focus in key))

        h5_stores_and_keys.append(((timestamp_h5_store_file,
                                    handcrafted_features_h5_store_file),
                                   keys_dict))

    t_list = get_valid_k_list(h5_stores_and_keys,
                              osn_focus)

    k_list_path = features_folder + "/k_list/focus_" + osn_focus + ".txt"

    store_valid_k_list(k_list_path,
                       t_list)


def get_valid_k_list(h5_stores_and_keys,
                     osn_focus):
    comment_lifetime_list = get_all_post_lifetimes(h5_stores_and_keys,
                                                   osn_focus)

    comment_lifetime_mean = statistics.mean(comment_lifetime_list)

    # Fit a nonnegative function.
    t_list = np.linspace(0, comment_lifetime_mean, 100)
    t_list = t_list[0:15]

    t_list = list(t_list)

    return t_list


def store_valid_k_list(k_list_path, k_list):
    with open(k_list_path, "w") as fp:
        for k in k_list:
            row = repr(k) + "\n"
            fp.write(row)


def load_valid_k_list(k_list_path):
    k_list = list()

    with open(k_list_path, "r") as fp:
        for row in fp:
            row_stripped = row.strip()
            if row_stripped == "":
                continue
            k_list.append(row_stripped)
    return k_list


def get_dataset_size(h5_store_and_keys,
                     osn_focus):
    dataset_size = 0

    for h5_store_file, h5_key_list in h5_store_and_keys:
        dataset_size += len(h5_key_list[osn_focus])

    return dataset_size


def get_all_post_lifetimes(h5_stores_and_keys, osn_focus):
    all_post_lifetimes_list = list()
    append_post_lifetime = all_post_lifetimes_list.append
    for h5_store_files, h5_keys in h5_stores_and_keys:
        for h5_key in h5_keys[osn_focus]:
            timestamps_data_frame = h5load_from(h5_store_files[0], h5_key)

            timestamps_col = timestamps_data_frame["timestamp"]

            if timestamps_col.size == 1:
                index = 0
            else:
                index = int(np.ceil(0.99 * (timestamps_col.size - 1)))

            append_post_lifetime(timestamps_col.iloc[index] - timestamps_col.iloc[0])

    return all_post_lifetimes_list


def get_all_comment_lifetimes(h5_stores_and_keys, osn_focus):
    all_comment_timestamps_list = list()
    extend_comment_timestamp = all_comment_timestamps_list.extend
    for h5_store_files, h5_keys in h5_stores_and_keys:
        for h5_key in h5_keys[osn_focus]:
            timestamps_data_frame = h5load_from(h5_store_files[0], h5_key)

            timestamps_col = timestamps_data_frame["timestamp"]

            extend_comment_timestamp(timestamps_col.iloc[1:] - timestamps_col.iloc[0])

    return all_comment_timestamps_list


def get_dataframe_row(data_frame, k, k_based_on_lifetime_old, feature_list):

    lifetime = k

    k_based_on_lifetime = get_k_based_on_lifetime(data_frame, lifetime, min_k=k_based_on_lifetime_old, max_k=-1)

    kth_row = get_kth_row(data_frame, k_based_on_lifetime, feature_list)
    return kth_row, k_based_on_lifetime


def get_k_based_on_timestamp(data_frame, timestamp, min_k=0, max_k=-1):

    timestamp_col = data_frame["timestamp"]
    timestamp_col = timestamp_col.iloc[:, min_k, max_k + 1]

    index = np.array(timestamp_col >= timestamp)

    if index.shape[1] == 0:
        k = -1
    else:
        k = index[0]

    return k


def get_k_based_on_lifetime(data_frame, lifetime, min_k, max_k):
    lifetime_col = data_frame["timestamp"] - data_frame["timestamp"].iloc[0]
    lifetime_col = lifetime_col.iloc[min_k:]

    index = np.searchsorted(lifetime_col, lifetime)

    index = max(0, index[0]-1)

    k = min_k + index

    if lifetime_col.size > (index+1):
        next_t = lifetime_col.iloc[index+1]
        if k == min_k:
            if lifetime_col.iloc[index] == lifetime_col.iloc[index+1]:
                k += 1
                if lifetime_col.size > (index+2):
                    next_t = lifetime_col.iloc[index+2]
                else:
                    next_t = np.nan
    else:
        next_t = np.nan

    return k, next_t
