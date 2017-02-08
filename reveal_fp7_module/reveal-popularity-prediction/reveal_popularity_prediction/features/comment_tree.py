__author__ = 'Georgios Rizos (georgerizos@iti.gr)'

import statistics
import collections
import resource
import sys
resource.setrlimit(resource.RLIMIT_STACK, (2**29, -1))
sys.setrecursionlimit(10**6)

import networkx as nx
import numpy as np
import scipy.sparse as spsp

from reveal_popularity_prediction.features.common import get_binary_graph, get_degree_undirected


def calculate_comment_count(comment_tree):
    nodes_1 = list(comment_tree.row)
    nodes_2 = list(comment_tree.col)
    nodes = set(nodes_1 + nodes_2)
    comment_count = len(nodes)

    return comment_count


def calculate_max_depth(comment_tree):
    comment_tree_nx = nx.from_scipy_sparse_matrix(comment_tree, create_using=nx.Graph())

    if len(comment_tree_nx) == 0:
        max_depth = 0.0
    else:
        node_to_depth = nx.shortest_path_length(comment_tree_nx, 0)
        max_depth = max(node_to_depth.values())

    return max_depth


def calculate_avg_depth(comment_tree):
    comment_tree_nx = nx.from_scipy_sparse_matrix(comment_tree, create_using=nx.Graph())

    if len(comment_tree_nx) == 0:
        avg_depth = 0.0
    else:
        node_to_depth = nx.shortest_path_length(comment_tree_nx, 0)
        avg_depth = statistics.mean(node_to_depth.values())

    return avg_depth


def calculate_max_width(comment_tree):
    comment_tree_nx = nx.from_scipy_sparse_matrix(comment_tree, create_using=nx.Graph())

    if len(comment_tree_nx) == 0:
        max_width = 1.0
    else:
        node_to_depth = nx.shortest_path_length(comment_tree_nx, 0)
        depth_to_nodecount = collections.defaultdict(int)

        for k, v in node_to_depth.items():
            depth_to_nodecount[v] += 1

        max_width = max(depth_to_nodecount.values())

    return max_width


def calculate_avg_width(comment_tree):
    comment_tree_nx = nx.from_scipy_sparse_matrix(comment_tree, create_using=nx.Graph())

    if len(comment_tree_nx) == 0:
        avg_width = 1.0
    else:
        node_to_depth = nx.shortest_path_length(comment_tree_nx, 0)
        depth_to_nodecount = collections.defaultdict(int)

        for k, v in node_to_depth.items():
            depth_to_nodecount[v] += 1

        avg_width = statistics.mean(depth_to_nodecount.values())

    return avg_width


def calculate_max_depth_over_max_width(comment_tree):
    comment_tree_nx = nx.from_scipy_sparse_matrix(comment_tree, create_using=nx.Graph())

    if len(comment_tree_nx) == 0:
        max_depth_over_max_width = 0.0
    else:
        node_to_depth = nx.shortest_path_length(comment_tree_nx, 0)
        depth_to_nodecount = collections.defaultdict(int)

        for k, v in node_to_depth.items():
            depth_to_nodecount[v] += 1

        max_depth = max(node_to_depth.values())
        max_width = max(depth_to_nodecount.values())

        max_depth_over_max_width = max_depth/max_width

    return max_depth_over_max_width


def calculate_avg_depth_over_width(comment_tree):
    comment_tree_nx = nx.from_scipy_sparse_matrix(comment_tree, create_using=nx.Graph())

    if len(comment_tree_nx) == 0:
        avg_depth_over_width = 0.0
    else:
        node_to_depth = nx.shortest_path_length(comment_tree_nx, 0)
        depth_to_nodecount = collections.defaultdict(int)

        for k, v in node_to_depth.items():
            depth_to_nodecount[v] += 1

        avg_depth_over_width = 0.0
        for k, v in depth_to_nodecount.items():
            avg_depth_over_width += k/v

    return avg_depth_over_width


def calculate_comment_tree_hirsch(comment_tree):
    comment_tree_nx = nx.from_scipy_sparse_matrix(comment_tree, create_using=nx.Graph())

    if len(comment_tree_nx) == 0:
        comment_tree_hirsch = 0.0
    else:
        node_to_depth = nx.shortest_path_length(comment_tree_nx, 0)

        depth_to_nodecount = collections.defaultdict(int)

        for k, v in node_to_depth.items():
            depth_to_nodecount[v] += 1

        comment_tree_hirsch = max(node_to_depth.values())
        while True:
            if depth_to_nodecount[comment_tree_hirsch] >= comment_tree_hirsch:
                break
            else:
                comment_tree_hirsch -= 1

    return comment_tree_hirsch


def calculate_comment_tree_randic(comment_tree):
    comment_tree_binary = get_binary_graph(comment_tree)

    total_degree_vector,\
    total_weighted_degree_vector = get_degree_undirected(comment_tree_binary)

    comment_tree_randic = np.sum(
        1.0 / np.sqrt(total_degree_vector[comment_tree_binary.row] * total_degree_vector[comment_tree_binary.col]))

    return comment_tree_randic


def calculate_comment_tree_wiener(comment_tree):
    comment_tree_csr = spsp.csr_matrix(comment_tree)
    comment_tree_csc = spsp.csc_matrix(comment_tree)

    comment_tree_wiener = get_avg_node_distances_tree(comment_tree_csr, comment_tree_csc, 0)
    return comment_tree_wiener


def get_avg_node_distances_tree(tree_csr, tree_csc, root):
    if tree_csr.getnnz() == 0:
        avg_node_distances = 0.0
        return avg_node_distances

    traversed_set = set()

    # Find tree leaves.
    leaf_stack = list()

    number_of_nodes = tree_csc.shape[0]
    node_to_children = list()
    node_to_parent = list()
    for n in range(number_of_nodes):
        children = set(list(tree_csc.getcol(n).indices))
        parent = set(list(tree_csr.getrow(n).indices))

        node_to_children.append(children)
        node_to_parent.append(parent)

        if len(parent) > 0:
            if len(children) == 0:
                leaf_stack.append(n)

    # Calculate index.
    parent_set = set()

    node_to_size = np.empty(number_of_nodes, dtype=np.float64)
    node_to_sum_sizes = np.empty(number_of_nodes, dtype=np.float64)
    node_to_sum_sizes_sqrt = np.empty(number_of_nodes, dtype=np.float64)

    last_node = 0

    while True:
        while len(leaf_stack) > 0:
            current_leaf = leaf_stack.pop()
            traversed_set.add(current_leaf)
            parent_set.update(node_to_parent[current_leaf])
            last_node = current_leaf

            children = node_to_children[current_leaf]
            if len(children) == 0:
                node_to_size[current_leaf] = 1.0
                node_to_sum_sizes[current_leaf] = 1.0
                node_to_sum_sizes_sqrt[current_leaf] = 1.0
            else:
                node_to_size[current_leaf] = 0.0
                node_to_sum_sizes[current_leaf] = 0.0
                node_to_sum_sizes_sqrt[current_leaf] = 0.0
                for child in children:
                    node_to_size[current_leaf] += node_to_size[child]
                    node_to_sum_sizes[current_leaf] += node_to_sum_sizes[child]
                    node_to_sum_sizes_sqrt[current_leaf] += node_to_sum_sizes_sqrt[child]
                node_to_size[current_leaf] += 1.0
                node_to_sum_sizes[current_leaf] += node_to_size[current_leaf]
                node_to_sum_sizes_sqrt[current_leaf] += np.power(node_to_size[current_leaf], 2)

        # Refill leaf_stack.
        candidate_leaf_list = list(parent_set.difference(traversed_set))
        for candidate_leaf in candidate_leaf_list:
            is_leaf = True
            for child in node_to_children[candidate_leaf]:
                if child not in traversed_set:
                    is_leaf = False
                    break
            if is_leaf:
                leaf_stack.append(candidate_leaf)

        # If leaf_stack is empty, break.
        if len(leaf_stack) == 0:
            break

    size = node_to_size[last_node]
    sum_sizes = node_to_sum_sizes[last_node]
    sum_sizes_sqrt = node_to_sum_sizes_sqrt[last_node]

    if size <= 1.0:
        avg_node_distances = 0.0
    else:
        avg_node_distances = (2*size/(size-1))*(sum_sizes/size - sum_sizes_sqrt/np.power(size, 2))

    if avg_node_distances < 0.0:
        avg_node_distances = 0.0

    return avg_node_distances


def calculate_comment_tree_wiener_old(comment_tree):
    comment_tree = spsp.csc_matrix(comment_tree)

    comment_tree_wiener = get_avg_node_distances_tree_old(comment_tree, 0)
    return comment_tree_wiener


def get_avg_node_distances_tree_old(tree, root):
    size, sum_sizes, sum_sizes_sqrt = moments_recursion_old(tree, root)

    if size == 1:
        avg_node_distances = 0.0
    else:
        avg_node_distances = (2*size/(size-1))*(sum_sizes/size - sum_sizes_sqrt/np.power(size, 2))

    return avg_node_distances


def moments_recursion_old(tree, root):
    root_children = list(tree.getcol(root).indices)
    if len(root_children) < 1:
        size = 1.0
        sum_sizes = 1.0
        sum_sizes_sqrt = 1.0
    else:
        size = 0.0
        sum_sizes = 0.0
        sum_sizes_sqrt = 0.0
        for child in root_children:
            size_c, sum_sizes_c, sum_sizes_sqrt_c = moments_recursion_old(tree, int(child))
            size += size_c
            sum_sizes += sum_sizes_c
            sum_sizes_sqrt += sum_sizes_sqrt_c

        size += 1.0
        sum_sizes += size
        sum_sizes_sqrt += np.power(size, 2)
    return size, sum_sizes, sum_sizes_sqrt
