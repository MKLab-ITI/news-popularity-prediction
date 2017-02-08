# -*- coding: <UTF-8> -*-
__author__ = 'Georgios Rizos (georgerizos@iti.gr)'

import json

from kombu.utils import uuid

from reveal_popularity_prediction.output.rabbitmq_util import rabbitmq_server_service,\
    establish_rabbitmq_connection,\
    simple_notification, simpler_notification


def publish_to_wp5(prediction_json,
                   rabbitmq_dict,
                   assessment_id):
    rabbitmq_uri = rabbitmq_dict["rabbitmq_uri"]
    rabbitmq_queue = rabbitmq_dict["rabbitmq_queue"]
    rabbitmq_exchange = rabbitmq_dict["rabbitmq_exchange"]
    rabbitmq_routing_key = rabbitmq_dict["rabbitmq_routing_key"]
    rabbitmq_channel = rabbitmq_dict["channel"]

    # rabbitmq_server_service("restart")
    # rabbitmq_connection = establish_rabbitmq_connection(rabbitmq_uri)

    # Make wp5 json report.
    json_report = make_w5_json_report(prediction_json,
                                      assessment_id)
    json_report_string = json.dumps(json_report)
    # print("wp5", json_report_string)

    # simple_notification(rabbitmq_connection, rabbitmq_queue, rabbitmq_exchange, rabbitmq_routing_key, json_report_string)
    simpler_notification(rabbitmq_channel, rabbitmq_queue, rabbitmq_exchange, rabbitmq_routing_key, json_report_string)


def make_w5_json_report(prediction_json,
                        assessment_id):
    json_report = dict()
    tweet_url = form_tweet_url(prediction_json)
    highly_controversial = is_highly_controversial(prediction_json)
    json_report["certh:tweet_url"] = tweet_url
    json_report["certh:highly_controversial"] = highly_controversial

    json_report["certh:item_url"] = prediction_json["url"]
    json_report["certh:time_posted"] = prediction_json["snapshots"][-1]["timestamp_list"][0]
    json_report["certh:assessment_timestamp"] = prediction_json["assessment_timestamp"]

    json_report["certh:assessment_id"] = assessment_id
    json_report["certh:platform"] = prediction_json["platform_name"]

    json_report["certh:current_time_stats"] = form_current_time_stats(prediction_json)
    json_report["certh:prediction_stats"] = form_prediction_stats(prediction_json)

    return json_report


def form_tweet_url(item):
    user_screen_name = item["user_screen_name"]
    tweet_id = item["tweet_id"]
    tweet_url = "https://twitter.com/" + user_screen_name + "/status/" + repr(tweet_id)
    return tweet_url


def is_highly_controversial(item):
    if item["predictions"]["controversiality"] > 0.1:
        highly_controversial = True
    else:
        highly_controversial = False
    return highly_controversial


def form_features_dict(item):
    features_dict = dict()

    for feature_name, feature_value in item["snapshots"][-1]["features"].items():
        features_dict["certh:" + feature_name] = feature_value

    return features_dict


def form_current_time_stats(item):
    current_time_stats_dict = dict()

    current_time_stats_dict["certh:time_collected"] = item["tweet_timestamp"]

    current_time_stats_dict["certh:features"] = form_features_dict(item)
    if item["platform_name"] == "YouTube":
        current_time_stats_dict["certh:user_set"] = ["https://www.youtube.com/channel/" + user_url for user_url in item["snapshots"][-1]["user_set"]]
    elif item["platform_name"] == "Reddit":
        current_time_stats_dict["certh:user_set"] = ["https://www.reddit.com/user/" + user_url for user_url in item["snapshots"][-1]["user_set"]]
    else:
        print("Invalid platform name.")
        raise RuntimeError

    current_time_stats_dict["certh:comment_count"] = item["targets"]["comment_count"]
    current_time_stats_dict["certh:user_count"] = item["targets"]["user_count"]
    current_time_stats_dict["certh:upvote_count"] = item["targets"]["upvote_count"]
    current_time_stats_dict["certh:downvote_count"] = item["targets"]["downvote_count"]
    current_time_stats_dict["certh:score"] = item["targets"]["score"]
    current_time_stats_dict["certh:controversiality"] = item["targets"]["controversiality"]

    return current_time_stats_dict


def form_prediction_stats(item):
    prediction_stats_dict = dict()

    prediction_stats_dict["certh:prediction_window"] = [item["prediction_window"]["prediction_lower_timestamp"],
                                                        item["prediction_window"]["prediction_upper_timestamp"]]

    prediction_stats_dict["certh:comment_count_prediction"] = item["predictions"]["comments"]
    prediction_stats_dict["certh:user_count_prediction"] = item["predictions"]["users"]
    prediction_stats_dict["certh:score_prediction"] = item["predictions"]["score"]
    prediction_stats_dict["certh:controversiality_prediction"] = item["predictions"]["controversiality"]

    return prediction_stats_dict


def check_wp5_rabbitmq_connection(wp5_rabbitmq_connection,
                                  wp5_rabbitmq_queue,
                                  wp5_rabbitmq_exchange,
                                  wp5_rabbitmq_routing_key,
                                  rabbitmq_connection,
                                  rabbitmq_queue,
                                  rabbitmq_exchange,
                                  rabbitmq_routing_key,
                                  assessment_id):
    wp5_rabbitmq_exchange = assessment_id + "_certh_popularity_prediction"
    wp5_rabbitmq_queue = "certh_popularity_prediction.gen-%s" % uuid()
    wp5_rabbitmq_routing_key = "reveal_routing"

    if wp5_rabbitmq_connection is None:
        wp5_rabbitmq_connection = rabbitmq_connection

    return wp5_rabbitmq_connection,\
           wp5_rabbitmq_queue,\
           wp5_rabbitmq_exchange,\
           wp5_rabbitmq_routing_key
