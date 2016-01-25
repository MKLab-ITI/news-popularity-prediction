__author__ = 'Georgios Rizos (georgerizos@iti.gr)'

import numpy as np


def update_branching_randic_graph(discussion_tree,
                                  intermediate_dict,
                                  comment_id,
                                  parent_comment_id):
    """
    Randic tree update.

    Inputs: - discussion_tree: The discussion tree is scipy sparse matrix format.
            - intermediate_dict: The python dictionary that contains all intermediate variables and structures.
            - comment_id: The latest node to the discussion tree.
            - parent_comment_id: The parent node to the latest node to the discussion tree.

    Output: - intermediate_dict: The python dictionary that contains all intermediate variables and structures.
    """
    # Get the parent of the parent node.
    compressed_graph = discussion_tree.tocsr()
    grandparent_comment_id = compressed_graph.getrow(int(parent_comment_id)).indices

    # Get the other children of the parent (only the siblings, not the self).
    compressed_graph = compressed_graph.tocsc()
    sibling_comment_id_set = set(list(compressed_graph.getcol(int(parent_comment_id)).indices))
    sibling_comment_id_set.remove(comment_id)

    # Update the node to degree map.
    intermediate_dict["node_to_degree"][parent_comment_id] += 1
    intermediate_dict["node_to_degree"][comment_id] += 1

    # Update the comment to parent comment edge.
    intermediate_dict["branching_randic_graph"][comment_id, parent_comment_id] =\
        1.0/(1.0*intermediate_dict["node_to_degree"][parent_comment_id])

    # Update the parent comment to grandparent comment edge.
    if grandparent_comment_id.size > 0:
        grandparent_comment_id = grandparent_comment_id[0]
        intermediate_dict["branching_randic_graph"][parent_comment_id, grandparent_comment_id] =\
            1.0/(intermediate_dict["node_to_degree"][parent_comment_id]*intermediate_dict["node_to_degree"][grandparent_comment_id])

    # Update all sibling comment to parent comment edges.
    if len(sibling_comment_id_set) > 0:
        for sibling_comment_id in sibling_comment_id_set:
            intermediate_dict["branching_randic_graph"][sibling_comment_id, parent_comment_id] =\
                1.0/(intermediate_dict["node_to_degree"][sibling_comment_id]*intermediate_dict["node_to_degree"][parent_comment_id])


def update_user_randic_graph(user_graph,
                             intermediate_dict,
                             source_node,
                             target_node,
                             user_graph_modified):
    """
    Randic graph update.

    Inputs: - user_graph: The user graph in scipy sparse format.
            - intermediate_dict: The python dictionary that contains all intermediate variables and structures.
            - source_node: The user that performed the latest action.
            - target_node: The user that the latest action was performed upon.
            - user_graph_modified: A boolean signifying whether the user graph actually changed after latest addition.

    Output: - intermediate_dict: The python dictionary that contains all intermediate variables and structures.
    """
    # We do this only if the user graph was actually modified.
    if user_graph_modified:
        # We get the parent nodes for the source and target nodes.
        compressed_graph = user_graph.tocsr()
        target_parent_id_set = set(list(compressed_graph.getrow(int(target_node)).indices))
        if source_node in target_parent_id_set:
            target_parent_id_set.remove(source_node)

        source_parent_id_set = set(list(compressed_graph.getrow(int(source_node)).indices))
        if target_node in source_parent_id_set:
            source_parent_id_set.remove(target_node)

        # We get the child nodes for the source and target nodes.
        compressed_graph = compressed_graph.tocsc()
        target_child_id_set = set(list(compressed_graph.getcol(int(target_node)).indices))
        if source_node in target_child_id_set:
            target_child_id_set.remove(source_node)

        source_child_id_set = set(list(compressed_graph.getcol(int(source_node)).indices))
        if target_node in source_child_id_set:
            source_child_id_set.remove(target_node)

        # Update the user to degree map.
        intermediate_dict["user_to_degree"][target_node] += 1
        intermediate_dict["user_to_degree"][source_node] += 1

        # Update the source node to target node edge. We keep only a directed edge.
        if intermediate_dict["user_randic_graph"][source_node, target_node] > 0.0:
            intermediate_dict["user_randic_graph"][source_node, target_node] =\
                1.0/(intermediate_dict["user_to_degree"][source_node]*intermediate_dict["user_to_degree"][target_node])
        elif intermediate_dict["user_randic_graph"][target_node, source_node] > 0.0:
            intermediate_dict["user_randic_graph"][target_node, source_node] =\
                1.0/(intermediate_dict["user_to_degree"][target_node]*intermediate_dict["user_to_degree"][source_node])
        else:
            intermediate_dict["user_randic_graph"][source_node, target_node] =\
                1.0/(intermediate_dict["user_to_degree"][source_node]*intermediate_dict["user_to_degree"][target_node])

        # We update all the target node to neighbor edges.
        if len(target_parent_id_set) > 0:
            for target_parent_id in target_parent_id_set:
                intermediate_dict["user_randic_graph"][target_node, target_parent_id] =\
                    1.0/(intermediate_dict["user_to_degree"][target_node]*intermediate_dict["user_to_degree"][target_parent_id])

        if len(target_child_id_set) > 0:
            for target_child_id in target_child_id_set:
                intermediate_dict["user_randic_graph"][target_child_id, target_node] =\
                    1.0/(intermediate_dict["user_to_degree"][target_child_id]*intermediate_dict["user_to_degree"][target_node])

        # We update all source node to neighbor edges.
        if len(source_parent_id_set) > 0:
            for source_parent_id in source_parent_id_set:
                intermediate_dict["user_randic_graph"][source_node, source_parent_id] =\
                    1.0/(intermediate_dict["user_to_degree"][source_node]*intermediate_dict["user_to_degree"][source_parent_id])

        if len(source_child_id_set) > 0:
            for source_child_id in source_child_id_set:
                intermediate_dict["user_randic_graph"][source_child_id, source_node] =\
                    1.0/(intermediate_dict["user_to_degree"][source_child_id]*intermediate_dict["user_to_degree"][source_node])


def update_branching_wiener_graph(discussion_tree,
                                  intermediate_dict,
                                  comment_id,
                                  parent_comment_id):
    """
    Wiener index intermediate structures update.

    Inputs: - discussion_tree: The discussion tree is scipy sparse matrix format.
            - intermediate_dict: The python dictionary that contains all intermediate variables and structures.
            - comment_id: The latest node to the discussion tree.
            - parent_comment_id: The parent node to the latest node to the discussion tree.

    Output: - intermediate_dict: The python dictionary that contains all intermediate variables and structures.
    """
    # Initialize update variables.
    compressed_discussion_tree = discussion_tree.tocsr()

    child = comment_id
    parent = parent_comment_id

    # Initialize the recursive update from the newest addition.
    child_subtree_cum_size_old = 0.0
    child_subtree_cum_size_sqrt_old = 0.0

    intermediate_dict["subtree_size_vector"][child] = 1.0
    intermediate_dict["subtree_cum_size_vector"][child] = 1.0
    intermediate_dict["subtree_cum_size_sqrt_vector"][child] = 1.0

    # Go iteratively towards the root.
    while True:
        parent_subtree_size_old = intermediate_dict["subtree_size_vector"][parent]
        parent_subtree_cum_size_old = intermediate_dict["subtree_cum_size_vector"][parent]
        parent_subtree_cum_size_sqrt_old = intermediate_dict["subtree_cum_size_sqrt_vector"][parent]

        intermediate_dict["subtree_size_vector"][parent] += 1.0
        intermediate_dict["subtree_cum_size_vector"][parent] += 1.0 +\
                                                                intermediate_dict["subtree_cum_size_vector"][child] -\
                                                                child_subtree_cum_size_old
        intermediate_dict["subtree_cum_size_sqrt_vector"][parent] += np.power(intermediate_dict["subtree_size_vector"][parent], 2) -\
                                                                     np.power(parent_subtree_size_old, 2) +\
                                                                     intermediate_dict["subtree_cum_size_sqrt_vector"][child] -\
                                                                     child_subtree_cum_size_sqrt_old

        # Get new parent.
        child = parent
        parent = compressed_discussion_tree.getrow(int(child)).indices

        if parent.size > 0:
            parent = parent[0]
        else:
            break

        # child_subtree_size_old = parent_subtree_size_old
        child_subtree_cum_size_old = parent_subtree_cum_size_old
        child_subtree_cum_size_sqrt_old = parent_subtree_cum_size_sqrt_old
