# -*- coding: <UTF-8> -*-
__author__ = 'Georgios Rizos (georgerizos@iti.gr)'

import copy

import numpy as np
import scipy.sparse as spsp
from pymongo import errors as pymongo_errors

from reveal_popularity_prediction.output.rabbitmq_util import rabbitmq_server_service,\
    establish_rabbitmq_connection,\
    simple_notification, simpler_notification


def write_to_mongo(prediction_json,
                   mongo_client,
                   tweet_input_mongo_database_name):
    json_report = make_w6_json_report(prediction_json)

    mongo_database = mongo_client[tweet_input_mongo_database_name]
    mongo_collection = mongo_database["popularity_prediction_output"]

    tweet_id = int(prediction_json["tweet_id"])
    json_report["_id"] = tweet_id
    json_report["tweet_id_string"] = repr(tweet_id)
    # print("wp6", json_report)

    smaller_json = copy.copy(json_report)
    while True:
        counter = 0
        try:
            mongo_collection.replace_one({"_id": tweet_id}, smaller_json, upsert=True)
            break
        except pymongo_errors.DocumentTooLarge:
            print("It was too large.")
            if counter >= (len(json_report["graph_snapshots"]) -1):
                smaller_json = copy.copy(json_report)
                smaller_json["graph_snapshots"] = [smaller_json["graph_snapshots"][0]]
                try:
                    mongo_collection.replace_one({"_id": tweet_id}, smaller_json, upsert=True)
                except pymongo_errors.DocumentTooLarge:
                    break
            smaller_json = copy.copy(json_report)
            smaller_json["graph_snapshots"] = [smaller_json["graph_snapshots"][0:-(counter+1)]]
            counter += 1


def make_w6_json_report(item):
    json_report = dict()
    json_report["graph_snapshots"] = get_graph_snapshots(item)
    json_report["post_timestamp"] = item["snapshots"][-1]["timestamp_list"][0]
    json_report["post_title"] = item["title"]
    json_report["post_url"] = item["url"]
    json_report["platform_name"] = item["platform_name"]
    json_report["assessment_timestamp"] = item["assessment_timestamp"]

    json_report["prediction_targets"] = dict()
    json_report["prediction_targets"]["comments"] = int(np.ceil(item["predictions"]["comments"]))
    json_report["prediction_targets"]["users"] = int(np.ceil(item["predictions"]["users"]))
    json_report["prediction_targets"]["score_wilson"] = item["predictions"]["score"]
    json_report["prediction_targets"]["controversiality_wilson"] = item["predictions"]["controversiality"]

    return json_report


def get_graph_snapshots(item):
    graph_snapshots = list()

    comment_tree_cumulative = get_edge_list_compact(item["snapshots"][0]["comment_tree"])
    user_graph_cumulative = get_edge_list_compact(item["snapshots"][0]["user_graph"])
    user_graph_node_list_cumulative = get_node_list_compact(user_graph_cumulative)

    snapshot_dict = dict()
    snapshot_dict["comment_tree"] = unfold_edge_list(comment_tree_cumulative)
    snapshot_dict["user_graph"] = unfold_edge_list(user_graph_cumulative)
    snapshot_dict["user_graph_node_list"] = user_graph_node_list_cumulative
    snapshot_dict["features"] = modify_feature_values_for_visualization(item["snapshots"][0]["features"])
    snapshot_dict["lifetime_seconds"] = item["snapshots"][0]["timestamp_list"][-1] - item["snapshots"][0]["timestamp_list"][0]

    graph_snapshots.append(snapshot_dict)

    comment_tree_cumulative = set(comment_tree_cumulative)
    user_graph_cumulative = set(user_graph_cumulative)
    user_graph_node_list_cumulative = set(user_graph_node_list_cumulative)

    for snapshot in item["snapshots"][1:]:
        snapshot_dict = dict()

        comment_tree_snapshot = set(get_edge_list_compact(snapshot["comment_tree"]))
        user_graph_snapshot = set(get_edge_list_compact(snapshot["user_graph"]))
        user_graph_node_list_snapshot = set(get_node_list_compact(user_graph_snapshot))

        comment_tree_difference = comment_tree_snapshot.difference(comment_tree_cumulative)
        user_graph_difference = user_graph_snapshot.difference(user_graph_cumulative)
        user_graph_node_list_difference = user_graph_node_list_snapshot.difference(user_graph_node_list_cumulative)

        snapshot_dict["comment_tree"] = unfold_edge_list(list(comment_tree_difference))
        snapshot_dict["user_graph"] = unfold_edge_list(list(user_graph_difference))
        snapshot_dict["user_graph_node_list"] = list(user_graph_node_list_difference)

        comment_tree_cumulative = comment_tree_snapshot
        user_graph_cumulative = user_graph_snapshot
        user_graph_node_list_cumulative = user_graph_node_list_snapshot

        snapshot_dict["features"] = modify_feature_values_for_visualization(snapshot["features"])
        snapshot_dict["lifetime_seconds"] = snapshot["timestamp_list"][-1] - snapshot["timestamp_list"][0]

        graph_snapshots.append(snapshot_dict)

    # for snapshot in item["snapshots"]:
    #     snapshot_dict = dict()
    #     snapshot_dict["comment_tree"] = get_edge_list(snapshot["comment_tree"])
    #     # snapshot_dict["comment_tree_node_list"] = get_node_list(snapshot_dict["comment_tree"])
    #     snapshot_dict["user_graph"] = get_edge_list(snapshot["user_graph"])
    #     snapshot_dict["user_graph_node_list"] = get_node_list(snapshot_dict["user_graph"])
    #     snapshot_dict["features"] = modify_feature_values_for_visualization(snapshot["features"])
    #     snapshot_dict["lifetime_seconds"] = snapshot["timestamp_list"][-1] - snapshot["timestamp_list"][0]
    #
    #     graph_snapshots.append(snapshot_dict)

    return graph_snapshots


def modify_feature_values_for_visualization(features_dict):
    feature_names = set(list(features_dict.keys()))

    feature = "avg_time_differences_1st_half"
    if feature in feature_names:
        features_dict[feature] = get_human_readable_large_number(features_dict[feature])
    feature = "avg_time_differences_2nd_half"
    if feature in feature_names:
        features_dict[feature] = get_human_readable_large_number(features_dict[feature])
    feature = "time_differences_std"
    if feature in feature_names:
        features_dict[feature] = get_human_readable_large_number(features_dict[feature])
    feature = "last_comment_lifetime"
    if feature in feature_names:
        features_dict[feature] = get_human_readable_large_number(features_dict[feature])

    feature = "author_privacy_status_youtube"
    if feature in feature_names:
        if features_dict[feature] == 0:
            features_dict[feature] = "Private"
        elif features_dict[feature] == 1:
            features_dict[feature] = "Unisted"
        elif features_dict[feature] == 2:
            features_dict[feature] = "Public"
        else:
            print("Invalid feature value.")
            raise RuntimeError
    feature = "author_is_linked_youtube"
    if feature in feature_names:
        features_dict[feature] = translate_to_boolean(features_dict[feature])
    feature = "author_long_uploads_status_youtube"
    if feature in feature_names:
        if features_dict[feature] == 0:
            features_dict[feature] = "Disallowed"
        elif features_dict[feature] == 1:
            features_dict[feature] = "Long Uploads Unspecified"
        elif features_dict[feature] == 2:
            features_dict[feature] = "Eligible"
        elif features_dict[feature] == 3:
            features_dict[feature] = "Allowed"
        else:
            print("Invalid feature value.")
            raise RuntimeError
    feature = "author_comment_count_youtube"
    if feature in feature_names:
        features_dict[feature] = get_human_readable_large_number(features_dict[feature])
    # feature = "author_comment_rate_youtube"
    # if feature in feature_names:
    #     features_dict[feature] = v
    feature = "author_view_count_youtube"
    if feature in feature_names:
        features_dict[feature] = get_human_readable_large_number(features_dict[feature])
    # feature = "author_view_rate_youtube"
    # if feature in feature_names:
    #     features_dict[feature] = v
    feature = "author_video_upload_count_youtube"
    if feature in feature_names:
        features_dict[feature] = get_human_readable_large_number(features_dict[feature])
    # feature = "author_video_upload_rate_youtube"
    # if feature in feature_names:
    #     features_dict[feature] = v
    feature = "author_subscriber_count_youtube"
    if feature in feature_names:
        features_dict[feature] = get_human_readable_large_number(features_dict[feature])
    # feature = "author_subscriber_rate_youtube"
    # if feature in feature_names:
    #     features_dict[feature] = v
    feature = "author_hidden_subscriber_count_youtube"
    if feature in feature_names:
        features_dict[feature] = translate_to_boolean(features_dict[feature])
    feature = "author_channel_lifetime_youtube"
    if feature in feature_names:
        features_dict[feature] = get_human_readable_lifetime(features_dict[feature])

    feature = "author_has_verified_mail_reddit"
    if feature in feature_names:
        features_dict[feature] = translate_to_boolean(features_dict[feature])
    feature = "author_account_lifetime_reddit"
    if feature in feature_names:
        features_dict[feature] = get_human_readable_lifetime(features_dict[feature])
    feature = "author_hide_from_robots_reddit"
    if feature in feature_names:
        features_dict[feature] = translate_to_boolean(features_dict[feature])
    feature = "author_is_mod_reddit"
    if feature in feature_names:
        features_dict[feature] = translate_to_boolean(features_dict[feature])
    feature = "author_link_karma_reddit"
    if feature in feature_names:
        features_dict[feature] = get_human_readable_large_number(features_dict[feature])
    # feature = "author_link_karma_rate_reddit"
    # if feature in feature_names:
    #     features_dict[feature] = v
    feature = "author_comment_karma_reddit"
    if feature in feature_names:
        features_dict[feature] = get_human_readable_large_number(features_dict[feature])
    # feature = "author_comment_karma_rate_reddit"
    # if feature in feature_names:
    #     features_dict[feature] = v
    feature = "author_is_gold_reddit"
    if feature in feature_names:
        features_dict[feature] = translate_to_boolean(features_dict[feature])
    return features_dict


def get_human_readable_large_number(feature_value_units):
    # human_readable_large_number = "{:,}".format(feature_value)
    thousands, units = divmod(feature_value_units, 1000)
    millions, thousands = divmod(thousands, 1000)
    # billions, millions = divmod(millions, 1000)

    human_readable_names_list = ["M", "K", ""]
    human_readable_large_number_list = [int(np.floor(millions)), int(np.floor(thousands)), int(np.floor(units))]
    for counter in range(len(human_readable_large_number_list)):
        if human_readable_large_number_list[counter] > 0.0:
            human_readable_large_number_list = human_readable_large_number_list[counter:]
            human_readable_names_list = human_readable_names_list[counter:]
            break
        if counter == (len(human_readable_large_number_list) -1):
            human_readable_large_number_list = human_readable_large_number_list[counter:]
            human_readable_names_list = human_readable_names_list[counter:]
            break

    if len(human_readable_large_number_list) == 0:
        human_readable_large_number = "0"
    elif len(human_readable_large_number_list) == 1:
        large_number = human_readable_large_number_list[0]
        human_readable_large_number = "%d" % large_number
    elif len(human_readable_large_number_list) > 1:
        large_number = human_readable_large_number_list[0]
        large_number = "{:,}".format(large_number)
        large_number_name = human_readable_names_list[0]
        human_readable_large_number = ">%s%s" % (large_number, large_number_name)
    else:
        print("Invalid large number.")
        raise RuntimeError

    return human_readable_large_number


def get_human_readable_lifetime(feature_value_seconds):
    minutes, seconds = divmod(feature_value_seconds, 60)
    hours, minutes = divmod(minutes, 60)
    days, hours = divmod(hours, 24)
    months, days = divmod(days, 30)
    years, months = divmod(months, 12)

    human_readable_names_list = ["years", "months", "days", "hours", "minutes", "seconds"]
    human_readable_lifetime_list = [int(years), int(months), int(days), int(hours), int(minutes), int(seconds)]
    for counter in range(len(human_readable_lifetime_list)):
        if human_readable_lifetime_list[counter] > 0.0:
            human_readable_lifetime_list = human_readable_lifetime_list[counter:]
            human_readable_names_list = human_readable_names_list[counter:]
            break
        if counter == (len(human_readable_lifetime_list) -1):
            human_readable_lifetime_list = human_readable_lifetime_list[counter:]
            human_readable_names_list = human_readable_names_list[counter:]
            break

    if len(human_readable_lifetime_list) == 0:
        human_readable_lifetime = "0 seconds"
    elif len(human_readable_lifetime_list) == 1:
        seconds = human_readable_lifetime_list[-1]
        if seconds == 1:
            name_seconds = "second"
        else:
            name_seconds = "seconds"
        human_readable_lifetime = "%d " % seconds + name_seconds
    elif len(human_readable_lifetime_list) > 1:
        big_value = human_readable_lifetime_list[0]
        big_name = human_readable_names_list[0]
        if big_value == 1:
            big_name = big_name[:-1]
        else:
            big_name = big_name
        human_readable_lifetime = "%d %s" % (big_value, big_name)

        # small_value = human_readable_lifetime_list[1]
        # small_name = human_readable_names_list[1]
        # if small_value == 0:
        #     pass
        # elif small_value == 1:
        #     small_name = small_name[:-1]
        #     human_readable_lifetime += " and %d %s" % (small_value, small_name)
        # else:
        #     small_name = small_name
        #     human_readable_lifetime += " and %d %s" % (small_value, small_name)
    else:
        print("Invalid lifetime.")
        raise RuntimeError
    return human_readable_lifetime


def translate_to_boolean(feature_value):
    if feature_value == 0:
        boolean = "True"
    elif feature_value == 1:
        boolean = "False"
    else:
        print("Invalid feature value.")
        raise RuntimeError
    return boolean


def get_edge_list(graph):
    graph = spsp.coo_matrix(graph)
    edge_list = list()
    for i, j in zip(list(graph.row), list(graph.col)):
        edge_list.append([repr(i), repr(j)])
    return edge_list


def get_node_list(edge_list):
    node_list = list()

    for edge in edge_list:
        i = edge[0]
        j = edge[1]

        node_list.append(i)
        node_list.append(j)
    node_list = list(set(node_list))

    if len(node_list) < 1:
        node_list.append(0)

    return node_list


def get_edge_list_compact(graph):
    graph = spsp.coo_matrix(graph)
    edge_list_compact = list()
    for i, j in zip(list(graph.row), list(graph.col)):
        edge_list_compact.append(repr(i) + "_" + repr(j))
    return edge_list_compact


def unfold_edge_list(edge_list_compact):
    edge_list = list()
    for edge in edge_list_compact:
        i_j = edge.split("_")
        i = i_j[0]
        j = i_j[1]

        edge_list.append([i, j])
    return edge_list


def get_node_list_compact(edge_list_compact):
    node_list_compact = list()

    for edge in edge_list_compact:
        i_j = edge.split("_")
        i = int(i_j[0])
        j = int(i_j[1])

        node_list_compact.append(i)
        node_list_compact.append(j)
    node_list_compact = list(set(node_list_compact))

    if len(node_list_compact) < 1:
        node_list_compact.append(0)

    return node_list_compact


def notify_wp6_item(prediction_json,
                    rabbitmq_dict):
    rabbitmq_uri = rabbitmq_dict["rabbitmq_uri"]
    rabbitmq_queue = rabbitmq_dict["rabbitmq_queue"]
    rabbitmq_exchange = rabbitmq_dict["rabbitmq_exchange"]
    rabbitmq_routing_key = rabbitmq_dict["rabbitmq_routing_key"]
    rabbitmq_channel = rabbitmq_dict["channel"]

    # rabbitmq_server_service("restart")
    # rabbitmq_connection = establish_rabbitmq_connection(rabbitmq_uri)

    item_id = prediction_json["tweet_id"]

    # simple_notification(rabbitmq_connection, rabbitmq_queue, rabbitmq_exchange, rabbitmq_routing_key, "ID_" + repr(item_id) + "_SUCCESS")
    simpler_notification(rabbitmq_channel, rabbitmq_queue, rabbitmq_exchange, rabbitmq_routing_key, "ID_" + repr(item_id) + "_SUCCESS")


def notify_wp6_stream(rabbitmq_dict):
    rabbitmq_uri = rabbitmq_dict["rabbitmq_uri"]
    rabbitmq_queue = rabbitmq_dict["rabbitmq_queue"]
    rabbitmq_exchange = rabbitmq_dict["rabbitmq_exchange"]
    rabbitmq_routing_key = rabbitmq_dict["rabbitmq_routing_key"]
    rabbitmq_channel = rabbitmq_dict["channel"]

    # rabbitmq_server_service("restart")
    # rabbitmq_connection = establish_rabbitmq_connection(rabbitmq_uri)

    # simple_notification(rabbitmq_connection, rabbitmq_queue, rabbitmq_exchange, rabbitmq_routing_key, "STREAM_SUCCESS")
    simpler_notification(rabbitmq_channel, rabbitmq_queue, rabbitmq_exchange, rabbitmq_routing_key, "STREAM_SUCCESS")
    print("Tweet stream processing complete. Success message published via RabbitMQ.")
