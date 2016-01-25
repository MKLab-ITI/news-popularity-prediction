__author__ = 'Georgios Rizos (georgerizos@iti.gr)'


def update_comment_count(comment_count):

    return comment_count


def update_max_depth(depth_node_dict):
    """
    Maximum tree depth update.

    Input:  - depth_node_dict: A map from node depth to node ids as a python dictionary.

    Output: - max_depth: The maximum depth of the tree.
    """
    max_depth = max(depth_node_dict.keys())
    return max_depth


def update_ave_depth(leaf_depth_sum, set_of_leaves):
    """
    Average tree depth update.

    Inputs: - leaf_depth_sum: The sum of the depth values of all leaf nodes.
            - set_of_leaves: A python set of leaf node ids.

    Output: - ave_depth: The average depth of the tree.
    """
    ave_depth = leaf_depth_sum/len(set_of_leaves)
    return ave_depth


def update_max_width(depth_node_dict, comment_depth, old_max_width):
    """
    Maximum tree width update.

    Inputs: - depth_node_dict: A map from node depth to node ids as a python dictionary.
            - comment_depth: The depth of the latest comment node.
            - old_max_width: The old value of the maximum width of the tree.

    Output: - max_width: The maximum width of the tree.
    """
    if len(depth_node_dict[comment_depth]) > old_max_width:
        max_width = len(depth_node_dict[comment_depth])
    else:
        max_width = old_max_width
    return max_width


def update_ave_width(width_sum, depth_node_dict):
    """
    Average tree width update.

    Inputs: - width_sum: The sum of all tree level widths (minus the root).
            - depth_node_dict: A map from node depth to node ids as a python dictionary.

    Output: - ave_width: The average width of the tree.
    """
    max_depth = update_max_depth(depth_node_dict)
    ave_width = width_sum/max_depth
    return ave_width


def update_max_depth_max_width_ratio(depth_node_dict):
    """
    Maximum tree depth to maximum tree width ratio update.

    Inputs: - depth_node_dict: A map from node depth to node ids as a python dictionary.
            - comment_depth: The depth of the latest comment node.
            - old_max_width: The old value of the maximum width of the tree.

    Output: - max_depth_max_width_ratio: The maximum depth to maximum width ratio of the tree.
    """
    max_depth = update_max_depth(depth_node_dict)
    max_width = max((len(level_set) for level_set in depth_node_dict.values()))
    max_depth_max_width_ratio = max_depth/max_width

    # max_depth = update_max_depth(depth_node_dict)
    # max_width = update_max_width(depth_node_dict, comment_depth, old_max_width)
    # max_depth_max_width_ratio = max_depth/max_width
    return max_depth_max_width_ratio


def update_depth_width_ratio_ave(depth_width_ratio_sum, depth_node_dict):
    """
    Average tree depth to tree width ratio update.

    Inputs: - depth_width_ratio_sum: The sum of the depth to width ratios for all tree nodes.
            - depth_node_dict: A map from node depth to node ids as a python dictionary.

    Output: - depth_width_ratio_ave: The average depth to width ratio of the tree.
    """
    max_depth = update_max_depth(depth_node_dict)
    depth_width_ratio_ave = depth_width_ratio_sum/max_depth
    return depth_width_ratio_ave

