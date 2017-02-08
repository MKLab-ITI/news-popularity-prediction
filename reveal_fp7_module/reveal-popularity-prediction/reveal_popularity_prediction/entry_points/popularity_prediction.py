__author__ = 'Georgios Rizos (georgerizos@iti.gr)'

import argparse

from reveal_popularity_prediction.reveal import integration


def main():
    """
    Entry point.
    """
    # Parse arguments.
    parser = argparse.ArgumentParser()

    ####################################################################################################################
    # MongoDB connection for getting access to input.
    ####################################################################################################################
    parser.add_argument("-uri", "--mongo-uri", dest="mongo_uri",
                        help="A mongo client URI.",
                        type=str, required=True)
    parser.add_argument("-id", "--assessment-id", dest="assessment_id",
                        help="A mongo database-collection pair in the form: \"database_name.collection_name\".",
                        type=str, required=False, default="snow_tweet_storage.tweets")

    ####################################################################################################################
    # RabbitMQ credentials for publishing a message.
    ####################################################################################################################
    parser.add_argument("-rmquri", "--rabbitmq-uri", dest="rabbitmq_uri",
                        help="RabbitMQ connection URI.",
                        type=str, required=True)
    parser.add_argument("-rmqq", "--rabbitmq-queue", dest="rabbitmq_queue",
                        help="RabbitMQ queue to check or create for publishing a success message.",
                        type=str, required=True)
    parser.add_argument("-rmqe", "--rabbitmq-exchange", dest="rabbitmq_exchange",
                        help="RabbitMQ exchange name.",
                        type=str, required=True)
    parser.add_argument("-rmqrk", "--rabbitmq-routing-key", dest="rabbitmq_routing_key",
                        help="RabbitMQ routing key (e.g. \"amqp://guest:guest@localhost:5672/vhost\").",
                        type=str, required=True)

    ####################################################################################################################
    # WP5 RabbitMQ credentials for publishing a message.
    ####################################################################################################################
    parser.add_argument("-wp5rmquri", "--wp5-rabbitmq-uri", dest="wp5_rabbitmq_uri",
                        help="RabbitMQ connection URI for communication with WP5.",
                        type=str, required=False, default=None)
    parser.add_argument("-wp5rmqq", "--wp5-rabbitmq-queue", dest="wp5_rabbitmq_queue",
                        help="RabbitMQ queue to check or create for publishing a success message for communication with WP5.",
                        type=str, required=False, default=None)
    parser.add_argument("-wp5rmqe", "--wp5-rabbitmq-exchange", dest="wp5_rabbitmq_exchange",
                        help="RabbitMQ exchange name for communication with WP5.",
                        type=str, required=False, default=None)
    parser.add_argument("-wp5rmqrk", "--wp5-rabbitmq-routing-key", dest="wp5_rabbitmq_routing_key",
                        help="RabbitMQ routing key for communication with WP5 (e.g. \"amqp://guest:guest@localhost:5672/vhost\").",
                        type=str, required=False, default=None)

    ####################################################################################################################
    # Parameters for specifying the input document set.
    ####################################################################################################################
    parser.add_argument("-ln", "--latest-n", dest="latest_n",
                        help="Get only the N most recent documents.",
                        type=int, required=False, default=None)
    parser.add_argument("-lts", "--lower-timestamp", dest="lower_timestamp",
                        help="Get only documents created after this UNIX timestamp.",
                        type=float, required=False, default=None)
    parser.add_argument("-uts", "--upper-timestamp", dest="upper_timestamp",
                        help="Get only documents created before this UNIX timestamp.",
                        type=float, required=False, default=None)
    parser.add_argument("-tssk", "--timestamp-sort-key", dest="timestamp_sort_key",
                        help="The document timestamp sort key.",
                        type=str, required=False, default="timestamp_ms")

    ####################################################################################################################
    # Execution parameters.
    ####################################################################################################################
    parser.add_argument("-nt", "--number-of-threads", dest="number_of_threads",
                        help="The number of parallel threads for feature extraction and classification.",
                        type=int, required=False, default=None)
    parser.add_argument("-irs", "--is-replayed-stream", dest="is_replayed_stream",
                        help="Whether the tweet stream is replayed.",
                        type=bool, required=False, default=True)

    ####################################################################################################################
    # Read/write destination information.
    ####################################################################################################################
    parser.add_argument("-ppdb", "--popularity-prediction-db", dest="popularity_prediction_db",
                        help="We may store some data in a mongo database in the same client.",
                        type=str, required=False, default="popularity_prediction_db")
    parser.add_argument("-lrf", "--local-resources-folder", dest="local_resources_folder",
                        help="We may utilize resources stored locally.",
                        type=str, required=False, default="/data/popularity_prediction_resources/")
    parser.add_argument("-rocp", "--reddit-oauth-credentials-path", dest="reddit_oauth_credentials_path",
                        help="Path to a file with Reddit credentials.",
                        type=str, required=False, default="/data/popularity_prediction_resources/reddit_oauth_credentials.txt")
    parser.add_argument("-yocf", "--youtube-oauth-credentials-folder", dest="youtube_oauth_credentials_folder",
                        help="Path to a folder with YouTube credentials.",
                        type=str, required=False, default="/data/popularity_prediction_resources/YouTube_credentials/")
    parser.add_argument("-ymc", "--youtube-module-communication", dest="youtube_module_communication",
                        help="Path to a file for that helps in in-module communication.",
                        type=str, required=False, default="/data/popularity_prediction_resources/json_yt_temp.json")
    parser.add_argument("-prp", "--platform-resources-path", dest="platform_resources_path",
                        help="Feature folder.",
                        type=str, required=False, default="/data/popularity_prediction_resources/Reddit_YouTube_features/")

    # Extract arguments.
    args = parser.parse_args()

    mongo_uri = args.mongo_uri
    assessment_id = args.assessment_id

    rabbitmq_uri = args.rabbitmq_uri
    rabbitmq_queue = args.rabbitmq_queue
    rabbitmq_exchange = args.rabbitmq_exchange
    rabbitmq_routing_key = args.rabbitmq_routing_key

    wp5_rabbitmq_uri = args.wp5_rabbitmq_uri
    wp5_rabbitmq_queue = args.wp5_rabbitmq_queue
    wp5_rabbitmq_exchange = args.wp5_rabbitmq_exchange
    wp5_rabbitmq_routing_key = args.wp5_rabbitmq_routing_key

    latest_n = args.latest_n
    lower_timestamp = args.lower_timestamp
    upper_timestamp = args.upper_timestamp
    timestamp_sort_key = args.timestamp_sort_key

    number_of_threads = args.number_of_threads
    is_replayed_stream = args.is_replayed_stream

    popularity_prediction_db = args.popularity_prediction_db
    local_resources_folder = args.local_resources_folder

    reddit_oauth_credentials_path = args.reddit_oauth_credentials_path
    youtube_oauth_credentials_folder = args.youtube_oauth_credentials_folder
    youtube_module_communication = args.youtube_module_communication
    platform_resources_path = args.platform_resources_path

    integration.social_context_collector_and_modality_extractor(mongo_uri=mongo_uri,
                                                                assessment_id=assessment_id,
                                                                rabbitmq_uri=rabbitmq_uri,
                                                                rabbitmq_queue=rabbitmq_queue,
                                                                rabbitmq_exchange=rabbitmq_exchange,
                                                                rabbitmq_routing_key=rabbitmq_routing_key,
                                                                wp5_rabbitmq_uri=wp5_rabbitmq_uri,
                                                                wp5_rabbitmq_queue=wp5_rabbitmq_queue,
                                                                wp5_rabbitmq_exchange=wp5_rabbitmq_exchange,
                                                                wp5_rabbitmq_routing_key=wp5_rabbitmq_routing_key,
                                                                latest_n=latest_n,
                                                                lower_timestamp=lower_timestamp,
                                                                upper_timestamp=upper_timestamp,
                                                                timestamp_sort_key=timestamp_sort_key,
                                                                number_of_threads=number_of_threads,
                                                                is_replayed_stream=is_replayed_stream,
                                                                popularity_prediction_db=popularity_prediction_db,
                                                                local_resources_folder=local_resources_folder,
                                                                reddit_oauth_credentials_path=reddit_oauth_credentials_path,
                                                                youtube_oauth_credentials_folder=youtube_oauth_credentials_folder,
                                                                youtube_module_communication=youtube_module_communication,
                                                                platform_resources_path=platform_resources_path)
