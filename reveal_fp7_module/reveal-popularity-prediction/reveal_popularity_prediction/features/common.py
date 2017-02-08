__author__ = 'Georgios Rizos (georgerizos@iti.gr)'

import numpy as np
import scipy.sparse as spsp


def get_binary_graph(graph):
    graph = spsp.coo_matrix(graph)
    binary_graph = spsp.coo_matrix((np.ones_like(graph.data,
                                                 dtype=np.float64),
                                    (graph.row, graph.col)),
                                   shape=graph.shape)
    return binary_graph


def get_degree_undirected(graph):
    graph = spsp.coo_matrix(graph)

    total_degree_vector = np.zeros(graph.shape[0], dtype=np.float64)
    total_weighted_degree_vector = np.zeros(graph.shape[0], dtype=np.float64)

    for i, j, d in zip(graph.row, graph.col, graph.data):
        total_degree_vector[i] += 1.0
        total_degree_vector[j] += 1.0

        total_weighted_degree_vector[i] += d
        total_weighted_degree_vector[j] += d

    return total_degree_vector,\
           total_weighted_degree_vector


def get_degree_directed(graph):
    graph = spsp.coo_matrix(graph)

    out_degree_vector = np.zeros(graph.shape[0], dtype=np.float64)
    out_weighted_degree_vector = np.zeros(graph.shape[0], dtype=np.float64)

    in_degree_vector = np.zeros(graph.shape[0], dtype=np.float64)
    in_weighted_degree_vector = np.zeros(graph.shape[0], dtype=np.float64)

    for i, j, d in zip(graph.row, graph.col, graph.data):
        out_degree_vector[i] += 1.0
        in_degree_vector[j] += 1.0

        out_weighted_degree_vector[i] += d
        in_weighted_degree_vector[j] += d

    total_degree_vector = out_degree_vector + in_degree_vector
    total_weighted_degree_vector = out_weighted_degree_vector + in_weighted_degree_vector

    return out_degree_vector,\
           out_weighted_degree_vector, \
           in_degree_vector, \
           in_weighted_degree_vector, \
           total_degree_vector,\
           total_weighted_degree_vector
