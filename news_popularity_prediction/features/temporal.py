__author__ = 'Georgios Rizos (georgerizos@iti.gr)'

import statistics


def update_first_half_time_difference_mean(timestamp_differences):
    """
    First half time difference mean update.

    Input:  - timestamp_differences: The list of all action timestamp differences.

    Output: - first_half_time_difference_mean: The first half time difference mean.
    """
    half_index = len(timestamp_differences)//2
    first_half = timestamp_differences[:half_index]

    if len(first_half) == 0:
        first_half_time_difference_mean = 0.0
    else:
        first_half_time_difference_mean = statistics.mean(first_half)

    return first_half_time_difference_mean


def update_last_half_time_difference_mean(timestamp_differences):
    """
    Last half time difference mean update.

    Input:  - timestamp_differences: The list of all action timestamp differences.

    Output: - last_half_time_difference_mean: The last half time difference mean.
    """
    half_index = len(timestamp_differences)//2
    last_half = timestamp_differences[half_index:]

    if len(last_half) == 0:
        last_half_time_difference_mean = 0.0
    else:
        last_half_time_difference_mean = statistics.mean(last_half)

    return last_half_time_difference_mean


def update_time_difference_std(timestamp_differences):
    """
    Time difference standard deviation update.

    Input:  - timestamp_differences: The list of all action timestamp differences.

    Output: - time_difference_std: The time difference standard deviation.
    """
    time_difference_std = statistics.pstdev(timestamp_differences)
    return time_difference_std


def update_timestamp_range(initial_post_timestamp, latest_reply_timestamp):

    timestamp_range = latest_reply_timestamp - initial_post_timestamp
    if timestamp_range < 0.0:
        timestamp_range = 0.0
    return timestamp_range

