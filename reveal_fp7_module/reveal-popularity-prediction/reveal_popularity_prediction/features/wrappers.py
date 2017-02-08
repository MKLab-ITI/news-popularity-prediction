__author__ = 'Georgios Rizos (georgerizos@iti.gr)'

from reveal_popularity_prediction.features import comment_tree
from reveal_popularity_prediction.features import user_graph
from reveal_popularity_prediction.features import temporal
from reveal_popularity_prediction.features import author


def wrapper_comment_count(graph_snapshot_input):
    comment_count = comment_tree.calculate_comment_count(graph_snapshot_input["comment_tree"])
    return comment_count


def wrapper_max_depth(graph_snapshot_input):
    basic_max_depth = comment_tree.calculate_max_depth(graph_snapshot_input["comment_tree"])
    return basic_max_depth


def wrapper_avg_depth(graph_snapshot_input):
    avg_depth = comment_tree.calculate_avg_depth(graph_snapshot_input["comment_tree"])
    return avg_depth


def wrapper_max_width(graph_snapshot_input):
    max_width = comment_tree.calculate_max_width(graph_snapshot_input["comment_tree"])
    return max_width


def wrapper_avg_width(graph_snapshot_input):
    avg_width = comment_tree.calculate_avg_width(graph_snapshot_input["comment_tree"])
    return avg_width


def wrapper_max_depth_over_max_width(graph_snapshot_input):
    max_depth_over_max_width = comment_tree.calculate_max_depth_over_max_width(graph_snapshot_input["comment_tree"])
    return max_depth_over_max_width


def wrapper_avg_depth_over_width(graph_snapshot_input):
    avg_depth_over_width = comment_tree.calculate_avg_depth_over_width(graph_snapshot_input["comment_tree"])
    return avg_depth_over_width


def wrapper_comment_tree_hirsch(graph_snapshot_input):
    comment_tree_hirsch = comment_tree.calculate_comment_tree_hirsch(graph_snapshot_input["comment_tree"])
    return comment_tree_hirsch


def wrapper_comment_tree_wiener(graph_snapshot_input):
    comment_tree_wiener = comment_tree.calculate_comment_tree_wiener(graph_snapshot_input["comment_tree"])
    return comment_tree_wiener


def wrapper_comment_tree_randic(graph_snapshot_input):
    comment_tree_randic = comment_tree.calculate_comment_tree_randic(graph_snapshot_input["comment_tree"])
    return comment_tree_randic


def wrapper_user_count(graph_snapshot_input):
    user_count = user_graph.calculate_user_count(graph_snapshot_input["user_graph"])
    return user_count


def wrapper_user_graph_hirsch(graph_snapshot_input):
    user_graph_hirsch = user_graph.calculate_user_graph_hirsch(graph_snapshot_input["user_graph"])
    return user_graph_hirsch


def wrapper_user_graph_randic(graph_snapshot_input):
    user_graph_randic = user_graph.calculate_user_graph_randic(graph_snapshot_input["user_graph"])
    return user_graph_randic


def wrapper_norm_outdegree_entropy(graph_snapshot_input):
    norm_outdegree_entropy = user_graph.calculate_norm_outdegree_entropy(graph_snapshot_input["user_graph"])
    return norm_outdegree_entropy


def wrapper_outdegree_entropy(graph_snapshot_input):
    outdegree_entropy = user_graph.calculate_outdegree_entropy(graph_snapshot_input["user_graph"])
    return outdegree_entropy


def wrapper_indegree_entropy(graph_snapshot_input):
    indegree_entropy = user_graph.calculate_indegree_entropy(graph_snapshot_input["user_graph"])
    return indegree_entropy


def wrapper_norm_indegree_entropy(graph_snapshot_input):
    norm_indegree_entropy = user_graph.calculate_norm_indegree_entropy(graph_snapshot_input["user_graph"])
    return norm_indegree_entropy


def wrapper_avg_time_differences_1st_half(graph_snapshot_input):
    avg_time_differences_1st_half = temporal.calculate_avg_time_differences_1st_half(graph_snapshot_input["timestamp_list"])
    return avg_time_differences_1st_half


def wrapper_avg_time_differences_2nd_half(graph_snapshot_input):
    avg_time_differences_2nd_half = temporal.calculate_avg_time_differences_2nd_half(graph_snapshot_input["timestamp_list"])
    return avg_time_differences_2nd_half


def wrapper_time_differences_std(graph_snapshot_input):
    time_differences_std = temporal.calculate_time_differences_std(graph_snapshot_input["timestamp_list"])
    return time_differences_std


def wrapper_last_comment_lifetime(graph_snapshot_input):
    last_comment_lifetime = temporal.calculate_last_comment_lifetime(graph_snapshot_input["timestamp_list"],
                                                                     graph_snapshot_input["tweet_timestamp"])
    return last_comment_lifetime


def wrapper_author_privacy_status_youtube(social_context_input):
    author_privacy_status_youtube = author.calculate_author_privacy_status_youtube(social_context_input["author"])

    return author_privacy_status_youtube


def wrapper_author_is_linked_youtube(social_context_input):
    author_is_linked_youtube = author.calculate_author_is_linked_youtube(social_context_input["author"])

    return author_is_linked_youtube


def wrapper_author_long_uploads_status_youtube(social_context_input):
    author_long_uploads_status_youtube = author.calculate_author_long_uploads_status_youtube(social_context_input["author"])

    return author_long_uploads_status_youtube


def wrapper_author_comment_count_youtube(social_context_input):
    author_comment_count_youtube = author.calculate_author_comment_count_youtube(social_context_input["author"])

    return author_comment_count_youtube


def wrapper_author_comment_rate_youtube(social_context_input):
    author_comment_rate_youtube = author.calculate_author_comment_rate_youtube(social_context_input["author"], social_context_input["initial_post"])

    return author_comment_rate_youtube


def wrapper_author_view_count_youtube(social_context_input):
    author_view_count_youtube = author.calculate_author_view_count_youtube(social_context_input["author"], social_context_input["initial_post"])

    return author_view_count_youtube


def wrapper_author_view_rate_youtube(social_context_input):
    author_view_rate_youtube = author.calculate_author_view_rate_youtube(social_context_input["author"], social_context_input["initial_post"])

    return author_view_rate_youtube


def wrapper_author_video_upload_count_youtube(social_context_input):
    author_video_upload_count_youtube = author.calculate_author_video_upload_count_youtube(social_context_input["author"])

    return author_video_upload_count_youtube


def wrapper_author_video_upload_rate_youtube(social_context_input):
    author_video_upload_rate_youtube = author.calculate_author_video_upload_rate_youtube(social_context_input["author"], social_context_input["initial_post"])

    return author_video_upload_rate_youtube


def wrapper_author_subscriber_count_youtube(social_context_input):
    author_subscriber_count_youtube = author.calculate_author_subscriber_count_youtube(social_context_input["author"])

    return author_subscriber_count_youtube


def wrapper_author_subscriber_rate_youtube(social_context_input):
    author_subscriber_rate_youtube = author.calculate_author_subscriber_rate_youtube(social_context_input["author"], social_context_input["initial_post"])

    return author_subscriber_rate_youtube


def wrapper_author_hidden_subscriber_count_youtube(social_context_input):
    author_hidden_subscriber_count_youtube = author.calculate_author_hidden_subscriber_count_youtube(social_context_input["author"])

    return author_hidden_subscriber_count_youtube


def wrapper_author_channel_lifetime_youtube(social_context_input):
    author_channel_lifetime_youtube = author.calculate_author_channel_lifetime_youtube(social_context_input["author"], social_context_input["initial_post"])

    return author_channel_lifetime_youtube


def wrapper_author_has_verified_mail_reddit(social_context_input):
    author_has_verified_mail_reddit = author.calculate_author_has_verified_mail_reddit(social_context_input["author"])

    return author_has_verified_mail_reddit


def wrapper_author_account_lifetime_reddit(social_context_input):
    author_account_lifetime_reddit = author.calculate_author_account_lifetime_reddit(social_context_input["author"], social_context_input["initial_post"])

    return author_account_lifetime_reddit


def wrapper_author_hide_from_robots_reddit(social_context_input):
    author_hide_from_robots_reddit = author.calculate_author_hide_from_robots_reddit(social_context_input["author"])

    return author_hide_from_robots_reddit


def wrapper_author_is_mod_reddit(social_context_input):
    author_is_mod_reddit = author.calculate_author_is_mod_reddit(social_context_input["author"])

    return author_is_mod_reddit


def wrapper_author_link_karma_reddit(social_context_input):
    author_link_karma_reddit = author.calculate_author_link_karma_reddit(social_context_input["author"])

    return author_link_karma_reddit


def wrapper_author_link_karma_rate_reddit(social_context_input):
    author_link_karma_rate_reddit = author.calculate_author_link_karma_rate_reddit(social_context_input["author"], social_context_input["initial_post"])

    return author_link_karma_rate_reddit


def wrapper_author_comment_karma_reddit(social_context_input):
    author_comment_karma_reddit = author.calculate_author_comment_karma_reddit(social_context_input["author"])

    return author_comment_karma_reddit


def wrapper_author_comment_karma_rate_reddit(social_context_input):
    author_comment_karma_rate_reddit = author.calculate_author_comment_karma_rate_reddit(social_context_input["author"], social_context_input["initial_post"])

    return author_comment_karma_rate_reddit


def wrapper_author_is_gold_reddit(social_context_input):
    author_is_gold_reddit = author.calculate_author_is_gold_reddit(social_context_input["author"])

    return author_is_gold_reddit
