__author__ = 'Georgios Rizos (georgerizos@iti.gr)'

import collections

import numpy as np
import scipy.sparse as spsp

from news_popularity_prediction.features.common import update_feature_value, replicate_feature_value
from news_popularity_prediction.features.intermediate import update_branching_randic_graph,\
    update_user_randic_graph, update_branching_wiener_graph
from news_popularity_prediction.features import basic_wrappers, branching_wrappers, user_graph_wrappers,\
    temporal_wrappers


def get_handcrafted_feature_names(dataset):
    """
    Returns a set of all the names of the engineered features.

    :param dataset: The name of the dataset (i.e. reddit, slashdot, barrapunto)
    :return: names: The set of feature names.
    """
    names = set()

    ####################################################################################################################
    # Add basic discussion tree features.
    ####################################################################################################################
    names.update(["basic_comment_count",
                  "basic_max_depth",
                  "basic_ave_depth",
                  "basic_max_width",
                  "basic_ave_width",
                  "basic_max_depth_max_width_ratio",
                  "basic_depth_width_ratio_ave"])

    ####################################################################################################################
    # Add branching discussion tree features.
    ####################################################################################################################
    names.update(["branching_hirsch_index",
                  "branching_wiener_index",
                  "branching_randic_index"])

    ####################################################################################################################
    # Add user graph features.
    ####################################################################################################################
    names.update(["user_graph_user_count",
                  "user_graph_hirsch_index",
                  "user_graph_randic_index",
                  "user_graph_outdegree_entropy",
                  "user_graph_outdegree_normalized_entropy",
                  "user_graph_indegree_entropy",
                  "user_graph_indegree_normalized_entropy"])

    ####################################################################################################################
    # Add temporal features.
    ####################################################################################################################
    names.update(["temporal_first_half_mean_time",
                  "temporal_last_half_mean_time",
                  "temporal_std_time",
                  "temporal_timestamp_range"])

    return names


def get_branching_feature_names(osn_name):
    """
    Returns a set of the names of the comment tree engineered features.

    :param osn_name: The name of the dataset (i.e. reddit, slashdot, barrapunto)
    :return: names: The set of feature names.
    """
    names = set()

    ####################################################################################################################
    # Add basic discussion tree features.
    ####################################################################################################################
    names.update(["basic_comment_count",
                  "basic_max_depth",
                  "basic_ave_depth",
                  "basic_max_width",
                  "basic_ave_width",
                  "basic_max_depth_max_width_ratio",
                  "basic_depth_width_ratio_ave"])

    ####################################################################################################################
    # Add branching discussion tree features.
    ####################################################################################################################
    names.update(["branching_hirsch_index",
                  "branching_wiener_index",
                  "branching_randic_index"])

    return names


def get_usergraph_feature_names(osn_name):
    """
    Returns a set of the names of the user graph engineered features.

    :param osn_name: The name of the dataset (i.e. reddit, slashdot, barrapunto)
    :return: names: The set of feature names.
    """
    names = set()

    ####################################################################################################################
    # Add user graph features.
    ####################################################################################################################
    names.update(["user_graph_user_count",
                  "user_graph_hirsch_index",
                  "user_graph_randic_index",
                  "user_graph_outdegree_entropy",
                  "user_graph_outdegree_normalized_entropy",
                  "user_graph_indegree_entropy",
                  "user_graph_indegree_normalized_entropy"])

    return names


def get_temporal_feature_names(osn_name):
    """
    Returns a set of the names of the temporal engineered features.

    :param osn_name: The name of the dataset (i.e. reddit, slashdot, barrapunto)
    :return: names: The set of feature names.
    """
    names = set()

    ####################################################################################################################
    # Add temporal features.
    ####################################################################################################################
    names.update(["temporal_first_half_mean_time",
                  "temporal_last_half_mean_time",
                  "temporal_std_time",
                  "temporal_timestamp_range"])

    return names


def initialize_timestamp_array(comment_number,
                               cascade_source_timestamp):
    timestamp_column_names_list = ["timestamp_true", "timestamp"]
    timestamp_array = np.empty((comment_number, 2),
                               dtype=np.float64)
    update_feature_value(timestamp_array, 0, 0, cascade_source_timestamp)
    update_feature_value(timestamp_array, 0, 1, cascade_source_timestamp)

    return timestamp_column_names_list, timestamp_array


def initialize_handcrafted_features(comment_number,
                                    handcrafted_feature_names_set,
                                    op_is_anonymous):
    """
    Initialize dictionary that maps a feature name to a score.

    Output: - feature_to_list: Dictionary that maps feature names to scores.
    """
    handcrafted_feature_names_list = list()
    replicate_feature_if_anonymous_set = list()
    handcrafted_function_list = list()
    handcrafted_feature_array = np.empty((comment_number,
                                          len(handcrafted_feature_names_set)),
                                         dtype=np.float64)

    ####################################################################################################################
    # Discussion tree basic features.
    ####################################################################################################################
    if "basic_comment_count" in handcrafted_feature_names_set:
        handcrafted_feature_names_list.append("basic_comment_count")
        handcrafted_function_list.append(getattr(basic_wrappers, "update_" + "basic_comment_count"))
        update_feature_value(handcrafted_feature_array, 0, len(handcrafted_function_list) - 1, 0)
    if "basic_max_depth" in handcrafted_feature_names_set:
        handcrafted_feature_names_list.append("basic_max_depth")
        handcrafted_function_list.append(getattr(basic_wrappers, "update_" + "basic_max_depth"))
        update_feature_value(handcrafted_feature_array, 0, len(handcrafted_function_list) - 1, 0)
    if "basic_ave_depth" in handcrafted_feature_names_set:
        handcrafted_feature_names_list.append("basic_ave_depth")
        handcrafted_function_list.append(getattr(basic_wrappers, "update_" + "basic_ave_depth"))
        update_feature_value(handcrafted_feature_array, 0, len(handcrafted_function_list) - 1, 0)
    if "basic_max_width" in handcrafted_feature_names_set:
        handcrafted_feature_names_list.append("basic_max_width")
        handcrafted_function_list.append(getattr(basic_wrappers, "update_" + "basic_max_width"))
        update_feature_value(handcrafted_feature_array, 0, len(handcrafted_function_list) - 1, 1)
    if "basic_ave_width" in handcrafted_feature_names_set:
        handcrafted_feature_names_list.append("basic_ave_width")
        handcrafted_function_list.append(getattr(basic_wrappers, "update_" + "basic_ave_width"))
        update_feature_value(handcrafted_feature_array, 0, len(handcrafted_function_list) - 1, 1)
    if "basic_max_depth_max_width_ratio" in handcrafted_feature_names_set:
        handcrafted_feature_names_list.append("basic_max_depth_max_width_ratio")
        handcrafted_function_list.append(getattr(basic_wrappers, "update_" + "basic_max_depth_max_width_ratio"))
        update_feature_value(handcrafted_feature_array, 0, len(handcrafted_function_list) - 1, 0)
    if "basic_depth_width_ratio_ave" in handcrafted_feature_names_set:
        handcrafted_feature_names_list.append("basic_depth_width_ratio_ave")
        handcrafted_function_list.append(getattr(basic_wrappers, "update_" + "basic_depth_width_ratio_ave"))
        update_feature_value(handcrafted_feature_array, 0, len(handcrafted_function_list) - 1, 0)

    ####################################################################################################################
    # Discussion tree branching indices.
    ####################################################################################################################
    if "branching_hirsch_index" in handcrafted_feature_names_set:
        handcrafted_feature_names_list.append("branching_hirsch_index")
        handcrafted_function_list.append(getattr(branching_wrappers, "update_" + "branching_hirsch_index"))
        update_feature_value(handcrafted_feature_array, 0, len(handcrafted_function_list) - 1, 0)
    if "branching_wiener_index" in handcrafted_feature_names_set:
        handcrafted_feature_names_list.append("branching_wiener_index")
        handcrafted_function_list.append(getattr(branching_wrappers, "update_" + "branching_wiener_index"))
        update_feature_value(handcrafted_feature_array, 0, len(handcrafted_function_list) - 1, 0)
    if "branching_randic_index" in handcrafted_feature_names_set:
        handcrafted_feature_names_list.append("branching_randic_index")
        handcrafted_function_list.append(getattr(branching_wrappers, "update_" + "branching_randic_index"))
        update_feature_value(handcrafted_feature_array, 0, len(handcrafted_function_list) - 1, 0)

    ####################################################################################################################
    # User graph features
    ####################################################################################################################
    if "user_graph_user_count" in handcrafted_feature_names_set:
        handcrafted_feature_names_list.append("user_graph_user_count")
        replicate_feature_if_anonymous_set.append(len(handcrafted_function_list) - 1)
        handcrafted_function_list.append(getattr(user_graph_wrappers, "update_" + "user_graph_user_count"))
        if op_is_anonymous:
            update_feature_value(handcrafted_feature_array, 0, len(handcrafted_function_list) - 1, 0)
        else:
            update_feature_value(handcrafted_feature_array, 0, len(handcrafted_function_list) - 1, 1)
    if "user_graph_user_count_estimated" in handcrafted_feature_names_set:
        handcrafted_feature_names_list.append("user_graph_user_count_estimated")
        handcrafted_function_list.append(getattr(user_graph_wrappers, "update_" + "user_graph_user_count_estimated"))
        update_feature_value(handcrafted_feature_array, 0, len(handcrafted_function_list) - 1, 1)
    if "user_graph_hirsch_index" in handcrafted_feature_names_set:
        handcrafted_feature_names_list.append("user_graph_hirsch_index")
        replicate_feature_if_anonymous_set.append(len(handcrafted_function_list) - 1)
        handcrafted_function_list.append(getattr(user_graph_wrappers, "update_" + "user_graph_hirsch_index"))
        update_feature_value(handcrafted_feature_array, 0, len(handcrafted_function_list) - 1, 0)
    if "user_graph_randic_index" in handcrafted_feature_names_set:
        handcrafted_feature_names_list.append("user_graph_randic_index")
        replicate_feature_if_anonymous_set.append(len(handcrafted_function_list) - 1)
        handcrafted_function_list.append(getattr(user_graph_wrappers, "update_" + "user_graph_randic_index"))
        update_feature_value(handcrafted_feature_array, 0, len(handcrafted_function_list) - 1, 0)
    if "user_graph_outdegree_entropy" in handcrafted_feature_names_set:
        handcrafted_feature_names_list.append("user_graph_outdegree_entropy")
        replicate_feature_if_anonymous_set.append(len(handcrafted_function_list) - 1)
        handcrafted_function_list.append(getattr(user_graph_wrappers, "update_" + "user_graph_outdegree_entropy"))
        update_feature_value(handcrafted_feature_array, 0, len(handcrafted_function_list) - 1, 0)
    if "user_graph_outdegree_normalized_entropy" in handcrafted_feature_names_set:
        handcrafted_feature_names_list.append("user_graph_outdegree_normalized_entropy")
        replicate_feature_if_anonymous_set.append(len(handcrafted_function_list) - 1)
        handcrafted_function_list.append(getattr(user_graph_wrappers, "update_" + "user_graph_outdegree_normalized_entropy"))
        update_feature_value(handcrafted_feature_array, 0, len(handcrafted_function_list) - 1, 1)
    if "user_graph_indegree_entropy" in handcrafted_feature_names_set:
        handcrafted_feature_names_list.append("user_graph_indegree_entropy")
        replicate_feature_if_anonymous_set.append(len(handcrafted_function_list) - 1)
        handcrafted_function_list.append(getattr(user_graph_wrappers, "update_" + "user_graph_indegree_entropy"))
        update_feature_value(handcrafted_feature_array, 0, len(handcrafted_function_list) - 1, 0)
    if "user_graph_indegree_normalized_entropy" in handcrafted_feature_names_set:
        handcrafted_feature_names_list.append("user_graph_indegree_normalized_entropy")
        replicate_feature_if_anonymous_set.append(len(handcrafted_function_list) - 1)
        handcrafted_function_list.append(getattr(user_graph_wrappers, "update_" + "user_graph_indegree_normalized_entropy"))
        update_feature_value(handcrafted_feature_array, 0, len(handcrafted_function_list) - 1, 0)

    ####################################################################################################################
    # Temporal features.
    ####################################################################################################################
    if "temporal_first_half_mean_time" in handcrafted_feature_names_set:
        handcrafted_feature_names_list.append("temporal_first_half_mean_time")
        handcrafted_function_list.append(getattr(temporal_wrappers, "update_" + "temporal_first_half_mean_time"))
        update_feature_value(handcrafted_feature_array, 0, len(handcrafted_function_list) - 1, 0)
    if "temporal_last_half_mean_time" in handcrafted_feature_names_set:
        handcrafted_feature_names_list.append("temporal_last_half_mean_time")
        handcrafted_function_list.append(getattr(temporal_wrappers, "update_" + "temporal_last_half_mean_time"))
        update_feature_value(handcrafted_feature_array, 0, len(handcrafted_function_list) - 1, 0)
    if "temporal_std_time" in handcrafted_feature_names_set:
        handcrafted_feature_names_list.append("temporal_std_time")
        handcrafted_function_list.append(getattr(temporal_wrappers, "update_" + "temporal_std_time"))
        update_feature_value(handcrafted_feature_array, 0, len(handcrafted_function_list) - 1, 0)
    if "temporal_timestamp_range" in handcrafted_feature_names_set:
        handcrafted_feature_names_list.append("temporal_timestamp_range")
        handcrafted_function_list.append(getattr(temporal_wrappers, "update_" + "temporal_timestamp_range"))
        update_feature_value(handcrafted_feature_array, 0, len(handcrafted_function_list) - 1, 0)

    return handcrafted_feature_names_list,\
           replicate_feature_if_anonymous_set,\
           handcrafted_function_list,\
           handcrafted_feature_array


def initialize_intermediate(comment_name_set,
                            user_name_set,
                            timestamp,
                            within_discussion_anonymous_coward,
                            op_is_anonymous):
    intermediate_dict = dict()

    discussion_tree_size = len(comment_name_set)
    user_graph_size = len(user_name_set)

    ####################################################################################################################
    # Initially, only the original poster has made a post.
    ####################################################################################################################
    intermediate_dict["contributor_comment_count"] = np.zeros(user_graph_size,
                                                              dtype=np.int32)
    intermediate_dict["contributor_replied_to_count"] = np.zeros(user_graph_size,
                                                                 dtype=np.int32)

    if op_is_anonymous:
        intermediate_dict["contributor_comment_count"][0] = 0
        intermediate_dict["anonymous_coward_comments_counter"] = 1
    else:
        intermediate_dict["contributor_comment_count"][0] = 1
        intermediate_dict["anonymous_coward_comments_counter"] = 0

    intermediate_dict["set_of_contributors"] = set()
    intermediate_dict["set_of_contributors"].add(0)

    intermediate_dict["within_discussion_anonymous_coward"] = within_discussion_anonymous_coward

    ####################################################################################################################
    # Initially, we only know about the depth of the original poster.
    ####################################################################################################################
    intermediate_dict["comment_depth"] = 0
    intermediate_dict["commenter_id"] = 0

    intermediate_dict["node_depth_dict"] = dict()
    intermediate_dict["node_depth_dict"][0] = 0

    intermediate_dict["depth_node_dict"] = collections.defaultdict(set)
    intermediate_dict["depth_node_dict"][0].add(0)

    ####################################################################################################################
    # We initialize the Wiener and Randic utility graphs. There are no edges initially.
    ####################################################################################################################
    intermediate_dict["subtree_size_vector"] = np.zeros(discussion_tree_size,
                                                        dtype=np.float64)
    intermediate_dict["subtree_size_vector"][0] = 1.0
    intermediate_dict["subtree_cum_size_vector"] = np.zeros(discussion_tree_size,
                                                            dtype=np.float64)
    intermediate_dict["subtree_cum_size_vector"][0] = 1.0
    intermediate_dict["subtree_cum_size_sqrt_vector"] = np.zeros(discussion_tree_size,
                                                                 dtype=np.float64)
    intermediate_dict["subtree_cum_size_sqrt_vector"][0] = 1.0

    intermediate_dict["branching_randic_graph"] = spsp.dok_matrix((discussion_tree_size,
                                                                   discussion_tree_size),
                                                                  dtype=np.float64)
    intermediate_dict["node_to_degree"] = collections.defaultdict(int)

    ####################################################################################################################
    # Initially, we only have one leaf.
    ####################################################################################################################
    intermediate_dict["set_of_leaves"] = set()
    intermediate_dict["set_of_leaves"].add(0)

    intermediate_dict["leaf_depth_sum"] = 0

    intermediate_dict["depth_width_ratio_sum"] = 0

    intermediate_dict["width_sum"] = 1

    ####################################################################################################################
    # Initially, the user graph has only one node.
    ####################################################################################################################
    intermediate_dict["comment_id_to_user_id"] = dict()
    intermediate_dict["comment_id_to_user_id"][0] = 0

    intermediate_dict["user_randic_graph"] = spsp.dok_matrix((user_graph_size,
                                                              user_graph_size),
                                                             dtype=np.float64)

    intermediate_dict["user_to_degree"] = collections.defaultdict(int)

    ####################################################################################################################
    # Initially, we only keep the original post's timestamp.
    ####################################################################################################################
    # intermediate_dict["list_of_timestamps"] = list()
    # intermediate_dict["list_of_timestamps"].append(timestamp)

    intermediate_dict["initial_timestamp"] = float(timestamp)
    intermediate_dict["latest_timestamp"] = float(timestamp)
    intermediate_dict["timestamp_differences"] = list()

    return intermediate_dict


def update_intermediate(discussion_tree,
                        user_graph,
                        intermediate_dict,
                        commenter_is_anonymous,
                        parent_commenter_is_anonymous,
                        comment_id,
                        parent_comment_id,
                        commenter_id,
                        parent_commenter_id,
                        user_graph_modified,
                        timestamp,
                        timestamp_difference):
    if not commenter_is_anonymous:
        # Change contributor-comment count vector.
        intermediate_dict["contributor_comment_count"][commenter_id] += 1

        # Add contributor to contributor set.
        intermediate_dict["set_of_contributors"].add(commenter_id)
    else:
        # If the Anonymous Coward made a comment, increase the count.
        intermediate_dict["anonymous_coward_comments_counter"] += 1

    if not parent_commenter_is_anonymous:
        intermediate_dict["contributor_replied_to_count"][parent_commenter_id] += 1

    # Update the node-depth dictionary.
    try:
        parent_comment_depth = intermediate_dict["node_depth_dict"][parent_comment_id]
    except KeyError:
        print("Parent comment not processed yet.")
        raise RuntimeError

    comment_depth = parent_comment_depth + 1
    intermediate_dict["node_depth_dict"][comment_id] = comment_depth
    intermediate_dict["comment_depth"] = comment_depth

    # Update the depth-node dictionary.
    if comment_depth not in intermediate_dict["depth_node_dict"].keys():
        # If comment has reached a new discussion tree depth.
        intermediate_dict["depth_node_dict"][comment_depth].add(comment_id)
        intermediate_dict["depth_width_ratio_sum"] += comment_depth  # Divided by current width, which is 1
    else:
        # Comment is not the deepest in the discussion.
        if comment_id in intermediate_dict["depth_node_dict"][comment_depth]:
            print("How can this comment have been already processed?")
            print(intermediate_dict["depth_node_dict"])
            raise RuntimeError
        intermediate_dict["depth_node_dict"][comment_depth].add(comment_id)
        current_width = len(intermediate_dict["depth_node_dict"][comment_depth])
        previous_width = len(intermediate_dict["depth_node_dict"][comment_depth]) - 1
        intermediate_dict["depth_width_ratio_sum"] = intermediate_dict["depth_width_ratio_sum"]\
                                                     - comment_depth/previous_width\
                                                     + comment_depth/current_width

    # Update Wiener graph.
    update_branching_wiener_graph(discussion_tree,
                                  intermediate_dict,
                                  comment_id,
                                  parent_comment_id)

    # Update Randic graph.
    update_branching_randic_graph(discussion_tree,
                                  intermediate_dict,
                                  comment_id,
                                  parent_comment_id)

    # If parent comment was a leaf, it is now no longer.
    if parent_comment_id in intermediate_dict["set_of_leaves"]:
        intermediate_dict["set_of_leaves"].remove(parent_comment_id)

    # The new comment is always a leaf.
    intermediate_dict["set_of_leaves"].add(comment_id)

    intermediate_dict["leaf_depth_sum"] = intermediate_dict["leaf_depth_sum"] - parent_comment_depth + comment_depth

    intermediate_dict["width_sum"] += 1

    intermediate_dict["commenter_id"] = commenter_id

    # Update user Randic graph.
    update_user_randic_graph(user_graph,
                             intermediate_dict,
                             commenter_id,
                             parent_commenter_id,
                             user_graph_modified)

    # # We append another date tuple timestamp.
    # intermediate_dict["list_of_timestamps"].append(timestamp)
    #
    # previous_seconds = float(intermediate_dict["list_of_timestamps"][-2])
    # current_seconds = float(intermediate_dict["list_of_timestamps"][-1])
    # timestamp_difference = current_seconds - previous_seconds
    #
    # if current_seconds < previous_seconds:
    #     print("Edited comment present.")
    #     timestamp_difference = 0.0
    #
    intermediate_dict["latest_timestamp"] = float(timestamp)
    intermediate_dict["timestamp_differences"].append(timestamp_difference)

    return intermediate_dict, comment_depth


def update_timestamp_array(timestamp_column_names,
                           timestamp_array,
                           new_timestamp,
                           comment_counter):
    previous_seconds = timestamp_array[comment_counter - 1, 1]
    current_seconds = float(new_timestamp)

    timestamp_array[comment_counter, 0] = current_seconds
    if current_seconds < previous_seconds:
        timestamp_array[comment_counter, 1] = previous_seconds
    else:
        timestamp_array[comment_counter, 1] = current_seconds


def update_handcrafted_features(handcrafted_feature_names_list,
                                replicate_feature_if_anonymous_set,
                                handcrafted_function_list,
                                handcrafted_feature_array,
                                comment_counter,
                                intermediate_dict,
                                commenter_is_anonymous):
    for j, update_function in enumerate(handcrafted_function_list):
        if commenter_is_anonymous and (j in replicate_feature_if_anonymous_set):
            replicate_feature_value(handcrafted_feature_array, comment_counter, j)
        else:
            update_function(handcrafted_feature_array, comment_counter, j, intermediate_dict)
