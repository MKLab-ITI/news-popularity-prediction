__author__ = 'Georgios Rizos (georgerizos@iti.gr)'

from news_popularity_prediction.features.common import update_feature_value
from news_popularity_prediction.features.temporal import update_first_half_time_difference_mean,\
    update_last_half_time_difference_mean, update_time_difference_std, update_timestamp_range


def update_temporal_first_half_mean_time(feature_array, i, j, intermediate_dict):
    temporal_first_half_mean_time = update_first_half_time_difference_mean(intermediate_dict["timestamp_differences"])
    update_feature_value(feature_array, i, j, temporal_first_half_mean_time)


def update_temporal_last_half_mean_time(feature_array, i, j, intermediate_dict):
    temporal_last_half_mean_time = update_last_half_time_difference_mean(intermediate_dict["timestamp_differences"])
    update_feature_value(feature_array, i, j, temporal_last_half_mean_time)


def update_temporal_std_time(feature_array, i, j, intermediate_dict):
    time_difference_std = update_time_difference_std(intermediate_dict["timestamp_differences"])
    update_feature_value(feature_array, i, j, time_difference_std)


def update_temporal_timestamp_range(feature_array, i, j, intermediate_dict):
    timestamp_range = update_timestamp_range(intermediate_dict["initial_timestamp"],
                                             intermediate_dict["latest_timestamp"])
    update_feature_value(feature_array, i, j, timestamp_range)
