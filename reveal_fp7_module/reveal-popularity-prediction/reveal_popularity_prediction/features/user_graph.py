__author__ = 'Georgios Rizos (georgerizos@iti.gr)'

import numpy as np

from reveal_popularity_prediction.features.common import get_binary_graph, get_degree_undirected, get_degree_directed


def calculate_user_count(user_graph):
    nodes_1 = list(user_graph.row)
    nodes_2 = list(user_graph.col)
    nodes = set(nodes_1 + nodes_2)
    user_count = len(nodes)
    return user_count


def calculate_user_graph_hirsch(user_graph):
    out_degree_vector,\
    out_weighted_degree_vector, \
    in_degree_vector, \
    in_weighted_degree_vector, \
    total_degree_vector,\
    total_weighted_degree_vector = get_degree_directed(user_graph)

    sorted_indices = np.argsort(out_weighted_degree_vector)

    # This is the previous hirsch index value
    minimum_hirsch_value = 1.0
    user_graph_hirsch = 1.0

    maximum_hirsch_value = out_weighted_degree_vector.size

    if maximum_hirsch_value > minimum_hirsch_value:
        comment_count = out_weighted_degree_vector[sorted_indices[-maximum_hirsch_value]]
        if comment_count >= maximum_hirsch_value:
            user_graph_hirsch = maximum_hirsch_value

    return user_graph_hirsch


def calculate_user_graph_randic(user_graph):
    user_graph_randic_binary = get_binary_graph(user_graph)

    total_degree_vector,\
    total_weighted_degree_vector = get_degree_undirected(user_graph_randic_binary)

    user_graph_randic = np.sum(
        1.0 / np.sqrt(total_degree_vector[user_graph_randic_binary.row] * total_degree_vector[user_graph_randic_binary.col]))
    return user_graph_randic


def calculate_norm_outdegree_entropy(user_graph):
    out_degree_vector,\
    out_weighted_degree_vector, \
    in_degree_vector, \
    in_weighted_degree_vector, \
    total_degree_vector,\
    total_weighted_degree_vector = get_degree_directed(user_graph)

    number_of_comments = int(np.sum(out_weighted_degree_vector))

    if number_of_comments < 2:
        norm_outdegree_entropy = 1.0
        return norm_outdegree_entropy

    # if within_discussion_anonymous_coward is not None:
    #     number_of_users = len(set_of_users) - 1
    # else:
    #     number_of_users = len(set_of_users)

    # Calculate the user-to-comment distribution entropy.
    comment_distribution = out_weighted_degree_vector/number_of_comments
    comment_distribution = comment_distribution[comment_distribution > 0]
    number_of_users = comment_distribution.size

    comment_entropy = np.abs(-np.sum(np.multiply(comment_distribution,
                                                 np.log(comment_distribution))))

    # Calculate the maximum possible user-to-comment distribution entropy given the number of comments.
    uniform_comment_count = np.zeros(number_of_users, dtype=np.float64)
    uniform_comment_count += (number_of_comments // number_of_users)
    uniform_comment_count[:(number_of_comments % number_of_users)] += 1

    uniform_comment_distribution = uniform_comment_count/number_of_comments
    uniform_comment_distribution = uniform_comment_distribution[uniform_comment_distribution > 0]

    max_comment_entropy = np.abs(-np.sum(np.multiply(uniform_comment_distribution,
                                                     np.log(uniform_comment_distribution))))

    # Calculate normalized user-to-comment entropy.
    if max_comment_entropy == 0.0:
        norm_outdegree_entropy = 1.0
    else:
        norm_outdegree_entropy = np.abs(comment_entropy/max_comment_entropy)
    return norm_outdegree_entropy


def calculate_outdegree_entropy(user_graph):
    out_degree_vector,\
    out_weighted_degree_vector, \
    in_degree_vector, \
    in_weighted_degree_vector, \
    total_degree_vector,\
    total_weighted_degree_vector = get_degree_directed(user_graph)

    number_of_comments = np.sum(out_weighted_degree_vector)

    if number_of_comments < 2:
        outdegree_entropy = 0
        return outdegree_entropy

    comment_distribution = out_weighted_degree_vector/number_of_comments
    comment_distribution = comment_distribution[comment_distribution > 0]

    outdegree_entropy = np.abs(-np.sum(np.multiply(comment_distribution,
                                                   np.log(comment_distribution))))
    return outdegree_entropy


def calculate_norm_indegree_entropy(user_graph):
    out_degree_vector,\
    out_weighted_degree_vector, \
    in_degree_vector, \
    in_weighted_degree_vector, \
    total_degree_vector,\
    total_weighted_degree_vector = get_degree_directed(user_graph)

    number_of_comments = int(np.sum(in_weighted_degree_vector))
    if number_of_comments < 2:
        norm_indegree_entropy = 1.0
        return norm_indegree_entropy

    # if within_discussion_anonymous_coward is not None:
    #     number_of_users = len(set_of_users) - 1
    # else:
    #     number_of_users = len(set_of_users)

    # Calculate the user-to-comment distribution entropy.
    replied_to_distribution = in_weighted_degree_vector/number_of_comments
    replied_to_distribution = replied_to_distribution[replied_to_distribution > 0]
    number_of_users = replied_to_distribution.size

    replied_to_entropy = np.abs(-np.sum(np.multiply(replied_to_distribution,
                                                    np.log(replied_to_distribution))))

    # Calculate the maximum possible user-to-comment distribution entropy given the number of comments.
    uniform_replied_to_count = np.zeros(number_of_users, dtype=np.float64)
    uniform_replied_to_count += (number_of_comments // number_of_users)
    uniform_replied_to_count[:(number_of_comments % number_of_users)] += 1

    uniform_replied_to_distribution = uniform_replied_to_count/number_of_comments
    uniform_replied_to_distribution = uniform_replied_to_distribution[uniform_replied_to_distribution > 0]

    max_replied_to_entropy = np.abs(-np.sum(np.multiply(uniform_replied_to_distribution,
                                                        np.log(uniform_replied_to_distribution))))

    # Calculate normalized user-to-comment entropy.
    if max_replied_to_entropy == 0.0:
        norm_indegree_entropy = 1.0
    else:
        norm_indegree_entropy = np.abs(replied_to_entropy/max_replied_to_entropy)
    return norm_indegree_entropy


def calculate_indegree_entropy(user_graph):
    out_degree_vector,\
    out_weighted_degree_vector, \
    in_degree_vector, \
    in_weighted_degree_vector, \
    total_degree_vector,\
    total_weighted_degree_vector = get_degree_directed(user_graph)

    number_of_comments = np.sum(in_weighted_degree_vector)

    if number_of_comments < 2:
        indegree_entropy = 0
        return indegree_entropy

    replied_to_distribution = in_weighted_degree_vector/number_of_comments
    replied_to_distribution = replied_to_distribution[replied_to_distribution > 0]

    indegree_entropy = np.abs(-np.sum(np.multiply(replied_to_distribution,
                                                  np.log(replied_to_distribution))))
    return indegree_entropy
