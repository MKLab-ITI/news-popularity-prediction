__author__ = 'Georgios Rizos (georgerizos@iti.gr)'

import numpy as np


def update_randic_index(randic_graph):
    """
    Calculates the Randic index for a graph.

    We maintain a graph/tree that has the same edges as the original tree.
    The edge values however are the Randic values for each edge.

    Input:  - randic_graph: A tree/graph that has the Randic edge value at each edge.

    Output: - randic_index: The sum of the inverse of the edge-degree product in a graph; i.e. the Randic index.
    """
    number_of_edges = randic_graph.data.size

    if number_of_edges < 1:
        randic_index = 0
    else:
        # randic_index = np.sum(1/np.sqrt(randic_graph.data))
        randic_index = np.sum(np.sqrt(randic_graph.data))

    return randic_index


def update_wiener_index(subtree_size_vector, subtree_cum_size_vector, subtree_cum_size_sqrt_vector):
    """
    Calculates the Wiener index (average of all between-node distances) for a radial tree.

    We maintain three values for each node of the tree, i.e. {subtree size,
                                                              subtree cumulative size,
                                                              subtree cumulative size squares}.

    Inputs: - subtree_size_vector: A vector that contains subtree sizes for all tree nodes.
            - subtree_cum_size_vector: A vector that contains subtree cumulative sizes for all tree nodes.
            - subtree_cum_size_sqrt_vector: A vector that contains subtree cumulative squared sizes for all tree nodes.

    Output: wiener_index: The average of all between-node distances; i.e. the Wiener index.
    """
    size = subtree_size_vector[0]
    sum_sizes = subtree_cum_size_vector[0]
    sum_sizes_sqrt = subtree_cum_size_sqrt_vector[0]

    wiener_index = (2*size/(size-1))*(sum_sizes/size - sum_sizes_sqrt/np.power(size, 2))

    return wiener_index


def update_hirsch_index(depth_node_dict, minimum_hirsch_value, maximum_hirsch_value):
    """
    Calculates the Hirsch index for a radial tree.

    Note that we have a slightly different definition of the Hirsch index to the one found in:
    Gómez, V., Kaltenbrunner, A., & López, V. (2008, April).
    Statistical analysis of the social network and discussion threads in slashdot.
    In Proceedings of the 17th international conference on World Wide Web (pp. 645-654). ACM.

    Inputs: - depth_node_dict: A map from node depth to node ids as a python dictionary.
            - minimum_hirsch_value: This is the previous Hirsch value.
            - maximum_hirsch_value: This is the depth of the latest node added to the tree.

    Output: - hirsch: The Hirsch index.
    """
    # This is the previous hirsch index value.
    hirsch_index = minimum_hirsch_value

    if maximum_hirsch_value > minimum_hirsch_value:
        adopters = depth_node_dict[maximum_hirsch_value]
        width = len(adopters)
        if width >= maximum_hirsch_value:
            hirsch_index = maximum_hirsch_value

    return hirsch_index
