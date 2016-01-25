__author__ = 'Georgios Rizos (georgerizos@iti.gr)'

import collections
import itertools

import numpy as np

from news_popularity_prediction.datautil.feature_rw import h5load_from, get_kth_row


def fill_X_handcrafted_k_actual(dataset_k,
                                h5_store_files,
                                h5_keys,
                                offset,
                                k,
                                X_k_min_dict,
                                X_t_next_dict,
                                branching_feature_names_list,
                                usergraph_feature_names_list,
                                temporal_feature_names_list,
                                osn_name):
    for d, h5_key in enumerate(h5_keys):
        if X_k_min_dict[osn_name][offset + d] == -1:
            dataset_k[osn_name]["X_branching"][offset + d, :] = np.nan
            dataset_k[osn_name]["X_usergraph"][offset + d, :] = np.nan
            dataset_k[osn_name]["X_temporal"][offset + d, :] = np.nan
            continue

        handcrafted_features_data_frame = h5load_from(h5_store_files[1], h5_key)

        # min_index = 0
        # max_index = len(branching_feature_names_list)

        kth_row = get_kth_row(handcrafted_features_data_frame,
                              X_k_min_dict[osn_name][offset + d],
                              branching_feature_names_list)

        dataset_k[osn_name]["X_branching"][offset + d, :] = kth_row

        # min_index = len(branching_feature_names_list)
        # max_index = len(branching_feature_names_list) + len(usergraph_feature_names_list)

        kth_row = get_kth_row(handcrafted_features_data_frame,
                              X_k_min_dict[osn_name][offset + d],
                              usergraph_feature_names_list)
        dataset_k[osn_name]["X_usergraph"][offset + d, :] = kth_row

        # min_index = len(branching_feature_names_list) + len(usergraph_feature_names_list)
        # max_index = len(branching_feature_names_list) + len(usergraph_feature_names_list) + len(temporal_feature_names_list)

        kth_row = get_kth_row(handcrafted_features_data_frame,
                              X_k_min_dict[osn_name][offset + d],
                              temporal_feature_names_list)
        dataset_k[osn_name]["X_temporal"][offset + d, :] = kth_row


def fill_X_handcrafted_k_dummy(dataset_k,
                               h5_store_files,
                               h5_keys,
                               offset,
                               k,
                               X_k_min_dict,
                               X_t_next_dict,
                               branching_feature_names_list,
                               usergraph_feature_names_list,
                               temporal_feature_names_list,
                               osn_name):
    pass


def fill_X_bipartite_graph_k_actual(dataset_k,
                                    h5_store_files,
                                    h5_keys,
                                    offset,
                                    k,
                                    X_k_min_dict,
                                    X_t_next_dict,
                                    osn_name):
    for d, h5_key in enumerate(h5_keys):
        if X_k_min_dict[osn_name][offset + d] == -1:
            continue

        bipartite_graph_features_data_frame = h5load_from(h5_store_files[3], h5_key)

        column = bipartite_graph_features_data_frame["user_id"]
        column = column.iloc[:X_k_min_dict[osn_name][offset + d]]

        # Make a multiset.
        multiset = collections.Counter(column)
        number_of_unique_users = len(multiset)

        dataset_k[osn_name]["X_bipartite_graph_row"].extend(itertools.repeat(offset + d, number_of_unique_users))
        dataset_k[osn_name]["X_bipartite_graph_col"].extend(multiset.keys())
        dataset_k[osn_name]["X_bipartite_graph_data"].extend(multiset.values())


def fill_X_bipartite_graph_k_dummy(dataset_k,
                                   h5_store_files,
                                   h5_keys,
                                   offset,
                                   k,
                                   X_k_min_dict,
                                   X_t_next_dict,
                                   osn_name):
    pass


def fill_X_handcrafted_fold_actual(X,
                                   indices,
                                   dataset_full,
                                   dataset_k,
                                   osn_name,
                                   min_column_index,
                                   max_column_index,
                                   handcrafted_features_type):
    if handcrafted_features_type == "X_author":
        X[:, min_column_index:max_column_index] = dataset_full[osn_name][handcrafted_features_type][indices, :]
    else:
        X[:, min_column_index:max_column_index] = dataset_k[osn_name][handcrafted_features_type][indices, :]


def fill_X_handcrafted_fold_dummy(X,
                                  indices,
                                  dataset_full,
                                  dataset_k,
                                  osn_name,
                                  min_column_index,
                                  max_column_index,
                                  handcrafted_features_type):
    pass
