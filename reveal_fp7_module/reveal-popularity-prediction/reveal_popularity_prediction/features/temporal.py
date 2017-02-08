__author__ = 'Georgios Rizos (georgerizos@iti.gr)'

import statistics


def calculate_avg_time_differences_1st_half(timestamp_list):
    timestamp_differences = get_timestamp_differences(timestamp_list)

    half_index = len(timestamp_differences)//2
    first_half = timestamp_differences[:half_index]

    if len(first_half) == 0:
        avg_time_differences_1st_half = 0.0
    else:
        avg_time_differences_1st_half = statistics.mean(first_half)
    return avg_time_differences_1st_half


def calculate_avg_time_differences_2nd_half(timestamp_list):
    timestamp_differences = get_timestamp_differences(timestamp_list)

    half_index = len(timestamp_differences)//2
    last_half = timestamp_differences[half_index:]

    if len(last_half) == 0:
        avg_time_differences_2nd_half = 0.0
    else:
        avg_time_differences_2nd_half = statistics.mean(last_half)
    return avg_time_differences_2nd_half


def calculate_time_differences_std(timestamp_list):
    if len(timestamp_list) == 1:
        time_differences_std = 0.0
    else:
        timestamp_differences = get_timestamp_differences(timestamp_list)
        time_differences_std = statistics.pstdev(timestamp_differences)
    return time_differences_std


def calculate_last_comment_lifetime(timestamp_list,
                                    tweet_timestamp):
    last_comment_lifetime = timestamp_list[-1] - timestamp_list[0]
    if last_comment_lifetime < 0.0:
        last_comment_lifetime = 0.0
    return last_comment_lifetime


def get_timestamp_differences(timestamp_list):
    timestamp_differences = list()

    for counter in range(1, len(timestamp_list)):
        timestamp_differences.append(timestamp_list[counter] - timestamp_list[counter-1])

    return timestamp_differences
