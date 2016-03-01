__author__ = 'Georgios Rizos (georgerizos@iti.gr)'

import os
import statistics

import numpy as np
import pandas as pd

from news_popularity_prediction.datautil.feature_rw import h5load_from, h5store_at, h5_open, h5_close, get_kth_row,\
    get_target_value
from news_popularity_prediction.discussion.features import get_branching_feature_names, get_usergraph_feature_names,\
    get_temporal_feature_names
from news_popularity_prediction.learning import concatenate_features


def make_feature_matrices(features_folder,
                          osn_focus):
    # Read comparison lifetimes.
    k_list_file_path = features_folder + "/k_list/focus_" + "post" + ".txt"
    k_list = load_valid_k_list(k_list_file_path)

    # Get feature names.
    branching_feature_dict = dict()
    usergraph_feature_dict = dict()
    temporal_feature_dict = dict()
    branching_feature_dict[osn_focus] = get_branching_feature_names(osn_name=osn_focus)
    usergraph_feature_dict[osn_focus] = get_usergraph_feature_names(osn_name=osn_focus)
    temporal_feature_dict[osn_focus] = get_temporal_feature_names(osn_name=osn_focus)

    branching_feature_names_list_dict = dict()
    usergraph_feature_names_list_dict = dict()
    temporal_feature_names_list_dict = dict()

    branching_feature_names_list_dict[osn_focus] = sorted(branching_feature_dict[osn_focus])
    usergraph_feature_names_list_dict[osn_focus] = sorted(usergraph_feature_dict[osn_focus])
    temporal_feature_names_list_dict[osn_focus] = sorted(temporal_feature_dict[osn_focus])

    number_of_branching_features_dict = dict()
    number_of_usergraph_features_dict = dict()
    number_of_temporal_features_dict = dict()

    number_of_branching_features_dict[osn_focus] = len(branching_feature_names_list_dict[osn_focus])
    number_of_usergraph_features_dict[osn_focus] = len(usergraph_feature_names_list_dict[osn_focus])
    number_of_temporal_features_dict[osn_focus] = len(temporal_feature_names_list_dict[osn_focus])

    # Make dataset matrix at time t_{\infty}.
    dataset_full_path = features_folder + "/dataset_full/dataset_full.h5"
    h5_stores_and_keys = get_h5_stores_and_keys(features_folder,
                                                "post")

    dataset_size = get_dataset_size(h5_stores_and_keys,
                                    "post")

    dataset_full,\
    index = form_dataset_full(dataset_size,
                              h5_stores_and_keys,
                              osn_focus,
                              branching_feature_names_list_dict,
                              usergraph_feature_names_list_dict,
                              temporal_feature_names_list_dict,
                              number_of_branching_features_dict,
                              number_of_usergraph_features_dict,
                              number_of_temporal_features_dict)

    store_dataset_full(dataset_full_path,
                       dataset_full,
                       index,
                       branching_feature_names_list_dict,
                       usergraph_feature_names_list_dict,
                       temporal_feature_names_list_dict)

    X_k_min_dict = dict()
    X_t_next_dict = dict()
    X_k_min_dict[osn_focus] = np.zeros(dataset_size, dtype=int)
    X_t_next_dict[osn_focus] = np.zeros(dataset_size, dtype=float)
    for k_index, k in enumerate(k_list):
        dataset_k,\
        X_k_min_dict,\
        X_t_next_dict,\
        index = form_dataset_k(dataset_size,
                               h5_stores_and_keys,
                               float(k),
                               X_k_min_dict,
                               X_t_next_dict,
                               feature_osn_name_list=[osn_focus])

        try:
            dataset_k_path = features_folder + "/dataset_k/" + osn_focus + "_lifetime_" + k + "_dataset_k.h5"
        except TypeError:
            dataset_k_path = features_folder + "/dataset_k/" + osn_focus + "_lifetime_" + repr(k) + "_dataset_k.h5"

        store_dataset_k(dataset_k_path,
                        dataset_k,
                        X_k_min_dict,
                        X_t_next_dict,
                        index)


def form_dataset_full(dataset_size,
                      h5_stores_and_keys,
                      osn_focus,
                      branching_feature_names_list_dict,
                      usergraph_feature_names_list_dict,
                      temporal_feature_names_list_dict,
                      number_of_branching_features_dict,
                      number_of_usergraph_features_dict,
                      number_of_temporal_features_dict):
    osn_to_targetlist = dict()
    if osn_focus == "reddit":
        osn_to_targetlist["reddit"] = ["comments",
                                       "users",
                                       "score_wilson",
                                       "controversiality_wilson"]
    if osn_focus == "slashdot":
        osn_to_targetlist["slashdot"] = ["comments",
                                         "users"]

    # Initialize full feature arrays.
    dataset_full = dict()
    index = dict()

    for osn_name in osn_to_targetlist.keys():
        dataset_full[osn_name] = dict()
        index[osn_name] = list()

        X_branching_full = np.empty((dataset_size,
                                     number_of_branching_features_dict[osn_name]),
                                    dtype=np.float64)
        dataset_full[osn_name]["X_branching"] = X_branching_full

        X_usergraph_full = np.empty((dataset_size,
                                     number_of_usergraph_features_dict[osn_name]),
                                    dtype=np.float64)
        dataset_full[osn_name]["X_usergraph"] = X_usergraph_full

        X_temporal_full = np.empty((dataset_size,
                                    number_of_temporal_features_dict[osn_name]),
                                   dtype=np.float64)
        dataset_full[osn_name]["X_temporal"] = X_temporal_full

        dataset_full[osn_name]["y_raw"] = dict()
        for target_name in osn_to_targetlist[osn_name]:
            dataset_full[osn_name]["y_raw"][target_name] = np.empty(dataset_size, dtype=np.float64)

        # Fill full feature arrays.
        offset = 0
        for h5_store_files, h5_keys in h5_stores_and_keys:
            index[osn_name].extend(h5_keys)
            fill_X_handcrafted_full_and_y_raw(dataset_full,
                                              h5_store_files,
                                              h5_keys["post"],
                                              offset,
                                              osn_name,
                                              osn_to_targetlist[osn_name],
                                              branching_feature_names_list_dict,
                                              usergraph_feature_names_list_dict,
                                              temporal_feature_names_list_dict,
                                              number_of_branching_features_dict,
                                              number_of_usergraph_features_dict,
                                              number_of_temporal_features_dict)

            offset += len(h5_keys["post"])

    return dataset_full, index


def fill_X_handcrafted_full_and_y_raw(dataset_full,
                                      h5_store_files,
                                      h5_keys,
                                      offset,
                                      osn_name,
                                      target_list,
                                      branching_feature_names_list_dict,
                                      usergraph_feature_names_list_dict,
                                      temporal_feature_names_list_dict,
                                      number_of_branching_features_dict,
                                      number_of_usergraph_features_dict,
                                      number_of_temporal_features_dict):
    for d, h5_key in enumerate(h5_keys):
        handcrafted_features_data_frame = h5load_from(h5_store_files[1], h5_key)

        kth_row = get_kth_row(handcrafted_features_data_frame,
                              -1,
                              branching_feature_names_list_dict[osn_name])
        dataset_full[osn_name]["X_branching"][offset + d, :number_of_branching_features_dict[osn_name]] = kth_row

        kth_row = get_kth_row(handcrafted_features_data_frame,
                              -1,
                              usergraph_feature_names_list_dict[osn_name])
        dataset_full[osn_name]["X_usergraph"][offset + d, :number_of_usergraph_features_dict[osn_name]] = kth_row

        kth_row = get_kth_row(handcrafted_features_data_frame,
                              -1,
                              temporal_feature_names_list_dict[osn_name])
        dataset_full[osn_name]["X_temporal"][offset + d, :number_of_temporal_features_dict[osn_name]] = kth_row

        for target_name in target_list:
            dataset_full[osn_name]["y_raw"][target_name][offset + d] = get_target_value(handcrafted_features_data_frame,
                                                                                        target_name)

"""
def load_dataset_full(dataset_full_path):
    dataset_full = dict()
    dataset_full[target_osn_name] = dict()

    index = dict()

    h5_store = h5_open(dataset_full_path)

    for osn_name in feature_osn_name_list:
        df = h5load_from(h5_store, "/data/" + osn_name + "/X_branching")[branching_feature_names_list_dict[osn_name]]
        # index[osn_name] = df.index
        dataset_full[osn_name]["X_branching"] = df.values
        dataset_full[osn_name]["X_usergraph"] = h5load_from(h5_store, "/data/" + osn_name + "/X_usergraph")[usergraph_feature_names_list_dict[osn_name]].values
        dataset_full[osn_name]["X_temporal"] = h5load_from(h5_store, "/data/" + osn_name + "/X_temporal")[temporal_feature_names_list_dict[osn_name]].values

    data_frame = h5load_from(h5_store, "/data/" + target_osn_name + "/y_raw")
    dataset_full[target_osn_name]["y_raw"] = dict()
    for target_name in target_name_list:
        dataset_full[target_osn_name]["y_raw"][target_name] = data_frame[target_name].values

    h5_close(h5_store)

    return dataset_full, index
"""

def store_dataset_full(dataset_full_path,
                       dataset_full,
                       index,
                       branching_feature_names_list_dict,
                       usergraph_feature_names_list_dict,
                       temporal_feature_names_list_dict):
    h5_store = h5_open(dataset_full_path)

    for osn_name in dataset_full.keys():
        h5store_at(h5_store, osn_name + "/X_branching", pd.DataFrame(dataset_full[osn_name]["X_branching"],
                                                                     columns=branching_feature_names_list_dict[osn_name]))
        h5store_at(h5_store, osn_name + "/X_usergraph", pd.DataFrame(dataset_full[osn_name]["X_usergraph"],
                                                                     columns=usergraph_feature_names_list_dict[osn_name]))
        h5store_at(h5_store, osn_name + "/X_temporal", pd.DataFrame(dataset_full[osn_name]["X_temporal"],
                                                                    columns=temporal_feature_names_list_dict[osn_name]))

        y_raw_dict = dict()
        for target_name in dataset_full[osn_name]["y_raw"].keys():
            y_raw_dict[target_name] = dataset_full[osn_name]["y_raw"][target_name]

        h5store_at(h5_store, osn_name + "/y_raw", pd.DataFrame(y_raw_dict))

    h5_close(h5_store)

"""
def load_dataset_k(self,
                   dataset_k_path):
    dataset_k = dict()
    X_k_min_dict = dict()
    X_t_next_dict = dict()

    index = dict()

    h5_store = h5_open(dataset_k_path)

    for osn_name in self.feature_osn_name_list:
        dataset_k[osn_name] = dict()

        df = h5load_from(h5_store, "/data/" + osn_name + "/X_branching")[self.branching_feature_names_list_dict[osn_name]]
        # index[osn_name] = df.index

        dataset_k[osn_name]["X_branching"] = df.values
        dataset_k[osn_name]["X_usergraph"] = h5load_from(h5_store, "/data/" + osn_name + "/X_usergraph")[self.usergraph_feature_names_list_dict[osn_name]].values
        dataset_k[osn_name]["X_temporal"] = h5load_from(h5_store, "/data/" + osn_name + "/X_temporal")[self.temporal_feature_names_list_dict[osn_name]].values

        data_frame = h5load_from(h5_store, "/data/" + osn_name + "/utility_arrays")
        X_k_min_dict[osn_name] = data_frame["X_k_min_array"].values
        X_t_next_dict[osn_name] = data_frame["X_t_next_array"].values

    h5_close(h5_store)

    return dataset_k, X_k_min_dict, X_t_next_dict, index
"""

def store_dataset_k(dataset_k_path,
                    dataset_k,
                    X_k_min_dict,
                    X_t_next_dict,
                    index):

    h5_store = h5_open(dataset_k_path)

    for osn_name in dataset_k.keys():
        h5store_at(h5_store, osn_name + "/X_branching", pd.DataFrame(dataset_k[osn_name]["X_branching"],
                                                                     columns=sorted(list(get_branching_feature_names(osn_name)))))
        h5store_at(h5_store, osn_name + "/X_usergraph", pd.DataFrame(dataset_k[osn_name]["X_usergraph"],
                                                                     columns=sorted(list(get_usergraph_feature_names(osn_name)))))
        h5store_at(h5_store, osn_name + "/X_temporal", pd.DataFrame(dataset_k[osn_name]["X_temporal"],
                                                                    columns=sorted(list(get_temporal_feature_names(osn_name)))))

        utility_arrays = dict()
        utility_arrays["X_k_min_array"] = X_k_min_dict[osn_name]
        utility_arrays["X_t_next_array"] = X_t_next_dict[osn_name]

        h5store_at(h5_store, osn_name + "/utility_arrays", pd.DataFrame(utility_arrays))

    h5_close(h5_store)


def form_dataset_k(dataset_size,
                   h5_stores_and_keys,
                   k,
                   X_k_min_dict,
                   X_t_next_dict,
                   feature_osn_name_list):
    all_feature_osn_names = feature_osn_name_list

    dataset_k = dict()
    index = dict()

    if True:
        for osn_index, osn_name in enumerate(all_feature_osn_names):
            dataset_k[osn_name] = dict()
            index[osn_name] = list()

            X_branching_k = np.empty((dataset_size,
                                      10),
                                     dtype=np.float64)
            dataset_k[osn_name]["X_branching"] = X_branching_k

            X_usergraph_k = np.empty((dataset_size,
                                      7),
                                     dtype=np.float64)
            dataset_k[osn_name]["X_usergraph"] = X_usergraph_k

            X_temporal_k = np.empty((dataset_size,
                                     4),
                                    dtype=np.float64)
            dataset_k[osn_name]["X_temporal"] = X_temporal_k

            # Fill full feature arrays.
            offset = 0
            for h5_store_files, h5_keys in h5_stores_and_keys:
                index[osn_name].extend(h5_keys)

                calculate_k_based_on_lifetime(dataset_k, h5_store_files, h5_keys, offset, k, X_k_min_dict, X_t_next_dict, osn_name)

                fill_X_handcrafted_k(dataset_k, h5_store_files, h5_keys["post"], offset, k, X_k_min_dict, X_t_next_dict, osn_name)
                offset += len(h5_keys["post"])

    return dataset_k, X_k_min_dict, X_t_next_dict, index


def calculate_k_based_on_lifetime(dataset_k,
                                  h5_store_files,
                                  h5_keys,
                                  offset,
                                  k,
                                  X_k_min_dict,
                                  X_t_next_dict,
                                  osn_name):
    number_of_keys = len(h5_keys["post"])

    for d in range(number_of_keys):
        timestamps_data_frame = h5load_from(h5_store_files[0], h5_keys["post"][d])

        if np.isnan(X_t_next_dict[osn_name][offset + d]):
            continue

        observed_comments,\
        next_lifetime = get_k_based_on_lifetime(timestamps_data_frame,
                                                k,
                                                min_k=X_k_min_dict[osn_name][offset + d],
                                                max_k=-1)

        X_k_min_dict[osn_name][offset + d] = observed_comments
        X_t_next_dict[osn_name][offset + d] = next_lifetime


def fill_X_handcrafted_k(dataset_k,
                         h5_store_files,
                         h5_keys,
                         offset,
                         k,
                         X_k_min_dict,
                         X_t_next_dict,
                         osn_name):

        concatenate_features.fill_X_handcrafted_k_actual(dataset_k,
                                                         h5_store_files,
                                                         h5_keys,
                                                         offset,
                                                         k,
                                                         X_k_min_dict,
                                                         X_t_next_dict,
                                                         sorted(list(get_branching_feature_names(osn_name))),
                                                         sorted(list(get_usergraph_feature_names(osn_name))),
                                                         sorted(list(get_temporal_feature_names(osn_name))),
                                                         osn_name)


def calculate_comparison_lifetimes(features_folder,
                                   osn_focus):
    if osn_focus is None:
        osn_focus = "post"

    h5_stores_and_keys = get_h5_stores_and_keys(features_folder,
                                                osn_focus)

    t_list = get_valid_k_list(h5_stores_and_keys,
                              osn_focus)

    k_list_path = features_folder + "/k_list/focus_" + osn_focus + ".txt"

    store_valid_k_list(k_list_path,
                       t_list)


def get_h5_stores_and_keys(features_folder,
                           osn_focus):
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

    return h5_stores_and_keys


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
