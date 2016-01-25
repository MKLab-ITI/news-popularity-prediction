__author__ = 'Georgios Rizos (georgerizos@iti.gr)'

from news_popularity_prediction.features.common import update_feature_value
from news_popularity_prediction.features.basic import update_max_depth, update_ave_depth,\
    update_max_width, update_ave_width, update_max_depth_max_width_ratio, update_depth_width_ratio_ave,\
    update_comment_count


def update_basic_comment_count(feature_array, i, j, intermediate_dict):
    basic_comment_count = update_comment_count(feature_array[i-1, j] + 1)
    update_feature_value(feature_array, i, j, basic_comment_count)


def update_basic_max_depth(feature_array, i, j, intermediate_dict):
    basic_max_depth = update_max_depth(intermediate_dict["depth_node_dict"])
    update_feature_value(feature_array, i, j, basic_max_depth)


def update_basic_ave_depth(feature_array, i, j, intermediate_dict):
    basic_ave_depth = update_ave_depth(intermediate_dict["leaf_depth_sum"],
                                       intermediate_dict["set_of_leaves"])
    update_feature_value(feature_array, i, j, basic_ave_depth)


def update_basic_max_width(feature_array, i, j, intermediate_dict):
    comment_depth = intermediate_dict["comment_depth"]
    basic_max_width = update_max_width(intermediate_dict["depth_node_dict"],
                                       comment_depth,
                                       feature_array[i-1, j])
    update_feature_value(feature_array, i, j, basic_max_width)


def update_basic_ave_width(feature_array, i, j, intermediate_dict):
    basic_ave_width = update_ave_width(intermediate_dict["width_sum"],
                                       intermediate_dict["depth_node_dict"])
    update_feature_value(feature_array, i, j, basic_ave_width)


def update_basic_max_depth_max_width_ratio(feature_array, i, j, intermediate_dict):
    # comment_depth = intermediate_dict["comment_depth"]
    # basic_max_depth_max_width_ratio = update_max_depth_max_width_ratio(intermediate_dict["depth_node_dict"],
    #                                                                    comment_depth,
    #                                                                    feature_to_list["basic_max_width"][-1])
    basic_max_depth_max_width_ratio = update_max_depth_max_width_ratio(intermediate_dict["depth_node_dict"])
    update_feature_value(feature_array, i, j, basic_max_depth_max_width_ratio)


def update_basic_depth_width_ratio_ave(feature_array, i, j, intermediate_dict):
    basic_depth_width_ratio_ave = update_depth_width_ratio_ave(intermediate_dict["depth_width_ratio_sum"],
                                                               intermediate_dict["depth_node_dict"])
    update_feature_value(feature_array, i, j, basic_depth_width_ratio_ave)
