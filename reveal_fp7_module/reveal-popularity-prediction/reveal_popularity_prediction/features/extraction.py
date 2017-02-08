__author__ = 'Georgios Rizos (georgerizos@iti.gr)'

import numpy as np

from reveal_popularity_prediction.features import wrappers


def extract_snapshot_features(comment_tree,
                              user_graph,
                              timestamp_list,
                              tweet_timestamp,
                              initial_post,
                              author,
                              platform):
    graph_snapshot_input = dict()
    graph_snapshot_input["comment_tree"] = comment_tree
    graph_snapshot_input["user_graph"] = user_graph
    graph_snapshot_input["timestamp_list"] = timestamp_list
    graph_snapshot_input["tweet_timestamp"] = tweet_timestamp
    graph_snapshot_input["initial_post"] = initial_post
    graph_snapshot_input["author"] = author

    feature_names = sorted(get_handcrafted_feature_names(platform))

    handcrafted_function_list = [getattr(wrappers, "wrapper_" + feature_name) for feature_name in feature_names]

    features = calculate_handcrafted_features(graph_snapshot_input,
                                              feature_names,
                                              handcrafted_function_list)

    return features


def calculate_handcrafted_features(graph_snapshot_input,
                                   feature_names,
                                   handcrafted_function_list):
    features = dict()
    for feature_name, calculation_function in zip(feature_names, handcrafted_function_list):
        feature_value = calculation_function(graph_snapshot_input)
        features[feature_name] = feature_value

    return features


def get_handcrafted_feature_names(platform):
    """
    Returns a set of feature names to be calculated.

    Output: - names: A set of strings, corresponding to the features to be calculated.
    """
    names = set()

    ####################################################################################################################
    # Add basic discussion tree features.
    ####################################################################################################################
    names.update(["comment_count",
                  "max_depth",
                  "avg_depth",
                  "max_width",
                  "avg_width",
                  "max_depth_over_max_width",
                  "avg_depth_over_width"])

    ####################################################################################################################
    # Add branching discussion tree features.
    ####################################################################################################################
    names.update(["comment_tree_hirsch",
                  "comment_tree_wiener",
                  "comment_tree_randic"])

    ####################################################################################################################
    # Add user graph features.
    ####################################################################################################################
    names.update(["user_count",
                  "user_graph_hirsch",
                  "user_graph_randic",
                  "outdegree_entropy",
                  "norm_outdegree_entropy",
                  "indegree_entropy",
                  "norm_indegree_entropy"])

    ####################################################################################################################
    # Add temporal features.
    ####################################################################################################################
    names.update(["avg_time_differences_1st_half",
                  "avg_time_differences_2nd_half",
                  "time_differences_std",
                  "last_comment_lifetime"])

    ####################################################################################################################
    # Add YouTube channel features.
    ####################################################################################################################
    if platform == "YouTube":
        names.update(["author_privacy_status_youtube",
                      "author_is_linked_youtube",
                      "author_long_uploads_status_youtube",
                      "author_comment_count_youtube",
                      "author_comment_rate_youtube",
                      "author_view_count_youtube",
                      "author_view_rate_youtube",
                      "author_video_upload_count_youtube",
                      "author_video_upload_rate_youtube",
                      "author_subscriber_count_youtube",
                      "author_subscriber_rate_youtube",
                      "author_hidden_subscriber_count_youtube",
                      "author_channel_lifetime_youtube"])

    ####################################################################################################################
    # Add Reddit author features.
    ####################################################################################################################
    elif platform == "Reddit":
        names.update(["author_has_verified_mail_reddit",
                      "author_account_lifetime_reddit",
                      "author_hide_from_robots_reddit",
                      "author_is_mod_reddit",
                      "author_link_karma_reddit",
                      "author_link_karma_rate_reddit",
                      "author_comment_karma_reddit",
                      "author_comment_karma_rate_reddit",
                      "author_is_gold_reddit"])
    else:
        print("Invalid platform name.")
        raise RuntimeError

    return names


def get_comment_tree_feature_names():
    names = set()

    names.update(["comment_count",
                  "max_depth",
                  "avg_depth",
                  "max_width",
                  "avg_width",
                  "max_depth_over_max_width",
                  "avg_depth_over_width"])

    return names


def get_user_graph_feature_names():
    names = set()

    names.update(["user_count",
                  "user_graph_hirsch",
                  "user_graph_randic",
                  "outdegree_entropy",
                  "norm_outdegree_entropy",
                  "indegree_entropy",
                  "norm_indegree_entropy"])

    return names


def get_temporal_feature_names():
    names = set()

    names.update(["avg_time_differences_1st_half",
                  "avg_time_differences_2nd_half",
                  "time_differences_std",
                  "last_comment_lifetime"])

    return names


def get_author_feature_names(platform):
    names = set()

    if platform == "YouTube":
        names.update(["author_privacy_status_youtube",
                      "author_is_linked_youtube",
                      "author_long_uploads_status_youtube",
                      "author_comment_count_youtube",
                      "author_comment_rate_youtube",
                      "author_view_count_youtube",
                      "author_view_rate_youtube",
                      "author_video_upload_count_youtube",
                      "author_video_upload_rate_youtube",
                      "author_subscriber_count_youtube",
                      "author_subscriber_rate_youtube",
                      "author_hidden_subscriber_count_youtube",
                      "author_channel_lifetime_youtube"])

    ####################################################################################################################
    # Add Reddit author features.
    ####################################################################################################################
    elif platform == "Reddit":
        names.update(["author_has_verified_mail_reddit",
                      "author_account_lifetime_reddit",
                      "author_hide_from_robots_reddit",
                      "author_is_mod_reddit",
                      "author_link_karma_reddit",
                      "author_link_karma_rate_reddit",
                      "author_comment_karma_reddit",
                      "author_comment_karma_rate_reddit",
                      "author_is_gold_reddit"])
    else:
        print("Invalid platform name.")
        raise RuntimeError

    return names

# print(sorted(get_handcrafted_feature_names("YouTube")))
# print(sorted(get_handcrafted_feature_names("Reddit")))


def make_features_vector(features_dict, platform):
    feature_names = sorted(get_handcrafted_feature_names(platform))

    features_vector_list = list()
    for feature_name in feature_names:
        feature_value = features_dict[feature_name]

        features_vector_list.append(feature_value)
    features_vector = np.empty((1, len(feature_names)), dtype=np.float64)
    for i, v in enumerate(features_vector_list):
        features_vector[0, i] = v

    return features_vector
