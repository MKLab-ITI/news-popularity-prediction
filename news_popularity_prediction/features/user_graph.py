__author__ = 'Georgios Rizos (georgerizos@iti.gr)'

import numpy as np


def update_user_count_eponymous(set_of_contributors, anonymous_coward_comments_counter):
    """
    Eponymous user count update.

    Input:  - set_of_contributors: A python set of user ids.
            - anonymous_coward_comments_counter: The number of comments posted by anonymous user(s).

    Output: - user_count: The number of eponymous users active in the information cascade.
    """
    user_count = len(set_of_contributors)

    return user_count


def update_user_count_estimated(set_of_contributors, anonymous_coward_comments_counter):
    """
    Total user count estimate update in the presence of anonymous users.

    Currently we use a very simplistic model for estimating the full user count.

    Inputs: - set_of_contributors: A python set of user ids.
            - anonymous_coward_comments_counter: The number of comments posted by anonymous user(s).

    Output: estimated_anonymous_contributor_count: The estimated number of users active in the information cascade.
    """
    eponymous_user_count = len(set_of_contributors)

    if anonymous_coward_comments_counter > 0:
        # TODO: Of course, I can use a much more sophisticated model.
        estimated_anonymous_user_count = (1 + anonymous_coward_comments_counter)/2
    else:
        estimated_anonymous_user_count = 0.0

    estimated_user_count = eponymous_user_count + estimated_anonymous_user_count
    return estimated_user_count


def update_user_hirsch_eponymous(contributor_comment_counts, minimum_hirsch_value, maximum_hirsch_value):
    """
    Calculates the Hirsch index for a user-comment occurrence vector.

    Inputs: - contributor_comment_counts: A map from user id to comments posted in numpy array format.
            - minimum_hirsch_value: This is the previous Hirsch value.
            - maximum_hirsch_value: This is the depth of the latest node added to the tree.

    Output: - hirsch: The Hirsch index.
    """
    sorted_indices = np.argsort(contributor_comment_counts)

    # This is the previous hirsch index value
    hirsch_index = minimum_hirsch_value

    if maximum_hirsch_value > contributor_comment_counts.size:
        maximum_hirsch_value = contributor_comment_counts.size

    if maximum_hirsch_value > minimum_hirsch_value:
        comment_count = contributor_comment_counts[sorted_indices[-maximum_hirsch_value]]
        if comment_count >= maximum_hirsch_value:
            hirsch_index = maximum_hirsch_value

    # # Check from maximum to minimum (not inclusive) possible hirsch values
    # for active_contributors in np.arange(maximum_hirsch_value, minimum_hirsch_value, -1):
    #     comment_count = contributor_comment_counts[sorted_indices[-active_contributors]]
    #     if comment_count >= active_contributors:
    #         hirsch = active_contributors
    #         break

    return hirsch_index


def update_graph_outdegree_entropy(contributor_comment_count):
    """
    Calculates the entropy of the user-to-comment distribution for eponymous users.

    Input:  - contributor_comment_count: A map from user id to comments posted in numpy array format.

    Output: - comment_entropy: The entropy of the user-to-comment distribution
    """
    # TODO: The vector also contains a position for the Anonymous user. However, the value should always remain zero.
    number_of_comments = np.sum(contributor_comment_count)

    if number_of_comments < 2:
        comment_entropy = 0
        return comment_entropy

    comment_distribution = contributor_comment_count/number_of_comments
    comment_distribution = comment_distribution[comment_distribution > 0]

    comment_entropy = np.abs(-np.sum(np.multiply(comment_distribution,
                                                 np.log(comment_distribution))))

    return comment_entropy


def update_graph_indegree_entropy(contributor_replied_to_count):
    # TODO: The vector also contains a position for the Anonymous user. However, the value should always remain zero.
    number_of_comments = np.sum(contributor_replied_to_count)

    if number_of_comments < 2:
        replied_to_entropy = 0
        return replied_to_entropy

    replied_to_distribution = contributor_replied_to_count/number_of_comments
    replied_to_distribution = replied_to_distribution[replied_to_distribution > 0]

    replied_to_entropy = np.abs(-np.sum(np.multiply(replied_to_distribution,
                                                    np.log(replied_to_distribution))))

    return replied_to_entropy


def update_normalized_graph_outdegree_entropy(contributor_comment_count,
                                              set_of_users,
                                              within_discussion_anonymous_coward):
    """
    Calculates the ratio of the entropy of the user-to-comment distribution to the maximum possible for eponymous users.

    Inputs: - contributor_comment_count: A map from user id to comments posted in numpy array format.
            - set_of_users: A python set of user ids.
            - within_discussion_anonymous_coward: The name of the Anonymous user for this dataset.

    Output: - normalized_eponymous_contributor_recurrence: The ratio of the entropy of the user-to-comment distribution
                                                           to the maximum possible for eponymous users.
    """
    number_of_comments = np.sum(contributor_comment_count)

    if number_of_comments < 2:
        normalized_eponymous_contributor_recurrence = 1.0
        return normalized_eponymous_contributor_recurrence

    if within_discussion_anonymous_coward is not None:
        number_of_users = len(set_of_users) - 1
    else:
        number_of_users = len(set_of_users)

    # Calculate the user-to-comment distribution entropy.
    comment_distribution = contributor_comment_count/number_of_comments
    comment_distribution = comment_distribution[comment_distribution > 0]

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
        normalized_eponymous_contributor_recurrence = 1.0
    else:
        normalized_eponymous_contributor_recurrence = np.abs(comment_entropy/max_comment_entropy)

    return normalized_eponymous_contributor_recurrence


def update_normalized_graph_indegree_entropy(contributor_replied_to_count,
                                             set_of_users,
                                             within_discussion_anonymous_coward):

    number_of_comments = np.sum(contributor_replied_to_count)
    if number_of_comments < 2:
        normalized_eponymous_contributor_recurrence = 1.0
        return normalized_eponymous_contributor_recurrence

    if within_discussion_anonymous_coward is not None:
        number_of_users = len(set_of_users) - 1
    else:
        number_of_users = len(set_of_users)

    # Calculate the user-to-comment distribution entropy.
    replied_to_distribution = contributor_replied_to_count/number_of_comments
    replied_to_distribution = replied_to_distribution[replied_to_distribution > 0]

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
        normalized_eponymous_contributor_recurrence = 1.0
    else:
        normalized_eponymous_contributor_recurrence = np.abs(replied_to_entropy/max_replied_to_entropy)

    return normalized_eponymous_contributor_recurrence

