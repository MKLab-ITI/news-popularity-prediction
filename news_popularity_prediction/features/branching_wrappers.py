__author__ = 'Georgios Rizos (georgerizos@iti.gr)'

from news_popularity_prediction.features.common import update_feature_value
from news_popularity_prediction.features.branching import update_hirsch_index, update_wiener_index, update_randic_index


def update_branching_hirsch_index(feature_array, i, j, intermediate_dict):
    comment_depth = intermediate_dict["comment_depth"]
    branching_hirsch_index = update_hirsch_index(intermediate_dict["depth_node_dict"],
                                                 feature_array[i-1, j],
                                                 comment_depth)
    update_feature_value(feature_array, i, j, branching_hirsch_index)


def update_branching_wiener_index(feature_array, i, j, intermediate_dict):
    branching_wiener_index = update_wiener_index(intermediate_dict["subtree_size_vector"],
                                                 intermediate_dict["subtree_cum_size_vector"],
                                                 intermediate_dict["subtree_cum_size_sqrt_vector"])
    update_feature_value(feature_array, i, j, branching_wiener_index)


def update_branching_randic_index(feature_array, i, j, intermediate_dict):
    branching_randic_index = update_randic_index(intermediate_dict["branching_randic_graph"].tocoo())
    update_feature_value(feature_array, i, j, branching_randic_index)


def update_branching_ifc_index():
    raise NotImplementedError
