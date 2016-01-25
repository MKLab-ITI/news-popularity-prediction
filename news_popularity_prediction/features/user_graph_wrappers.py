__author__ = 'Georgios Rizos (georgerizos@iti.gr)'

from news_popularity_prediction.features.common import update_feature_value
from news_popularity_prediction.features.user_graph import update_user_count_eponymous,\
    update_user_count_estimated, update_user_hirsch_eponymous,\
    update_graph_outdegree_entropy, update_normalized_graph_outdegree_entropy,\
    update_graph_indegree_entropy, update_normalized_graph_indegree_entropy
from news_popularity_prediction.features.branching import update_randic_index


def update_user_graph_user_count(feature_array, i, j, intermediate_dict):
    user_count_eponymous = update_user_count_eponymous(intermediate_dict["set_of_contributors"],
                                                       intermediate_dict["anonymous_coward_comments_counter"])
    update_feature_value(feature_array, i, j, user_count_eponymous)


def update_user_graph_user_count_estimated(feature_array, i, j, intermediate_dict):
    user_count_estimated = update_user_count_estimated(intermediate_dict["set_of_contributors"],
                                                       intermediate_dict["anonymous_coward_comments_counter"])
    update_feature_value(feature_array, i, j, user_count_estimated)

# if not commenter_is_anonymous:
#         user_hirsch = update_user_hirsch_eponymous(intermediate_dict["contributor_comment_count"],
#                                                    feature_to_list["user_graph_hirsch_index"][-1],
#                                                    intermediate_dict["contributor_comment_count"][commenter_id])
#     else:
#         user_hirsch = feature_to_list["user_hirsch"][-1]
def update_user_graph_hirsch_index(feature_array, i, j, intermediate_dict):
    commenter_id = intermediate_dict["commenter_id"]
    user_hirsch = update_user_hirsch_eponymous(intermediate_dict["contributor_comment_count"],
                                                   feature_array[i-1, j],
                                                   intermediate_dict["contributor_comment_count"][commenter_id])
    update_feature_value(feature_array, i, j, user_hirsch)


def update_user_graph_randic_index(feature_array, i, j, intermediate_dict):
    user_randic = update_randic_index(intermediate_dict["user_randic_graph"].tocoo())
    update_feature_value(feature_array, i, j, user_randic)


def update_user_graph_outdegree_entropy(feature_array, i, j, intermediate_dict):
    user_graph_outdegree_entropy = update_graph_outdegree_entropy(intermediate_dict["contributor_comment_count"])
    update_feature_value(feature_array, i, j, user_graph_outdegree_entropy)


def update_user_graph_outdegree_normalized_entropy(feature_array, i, j, intermediate_dict):
    within_discussion_anonymous_coward = intermediate_dict["within_discussion_anonymous_coward"]
    user_graph_outdegree_normalized_entropy = update_normalized_graph_outdegree_entropy(intermediate_dict["contributor_comment_count"],
                                                                                        intermediate_dict["set_of_contributors"],
                                                                                        within_discussion_anonymous_coward)
    update_feature_value(feature_array, i, j, user_graph_outdegree_normalized_entropy)


def update_user_graph_indegree_entropy(feature_array, i, j, intermediate_dict):
    user_graph_indegree_entropy = update_graph_indegree_entropy(intermediate_dict["contributor_replied_to_count"])
    update_feature_value(feature_array, i, j, user_graph_indegree_entropy)


def update_user_graph_indegree_normalized_entropy(feature_array, i, j, intermediate_dict):
    within_discussion_anonymous_coward = intermediate_dict["within_discussion_anonymous_coward"]
    user_graph_indegree_normalized_entropy = update_normalized_graph_indegree_entropy(intermediate_dict["contributor_replied_to_count"],
                                                                                      intermediate_dict["set_of_contributors"],
                                                                                      within_discussion_anonymous_coward)
    update_feature_value(feature_array, i, j, user_graph_indegree_normalized_entropy)
