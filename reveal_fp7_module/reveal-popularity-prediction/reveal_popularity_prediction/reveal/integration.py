__author__ = 'Georgios Rizos (georgerizos@iti.gr)'

from reveal_popularity_prediction.common.config_package import get_threads_number
from reveal_popularity_prediction.reveal.utility import make_time_window_filter, safe_establish_mongo_connection,\
    process_tweets_and_extract_urls, collect_social_context, form_graphs, extract_features, make_predictions,\
    publish_output


def social_context_collector_and_modality_extractor(mongo_uri,
                                                    assessment_id,
                                                    rabbitmq_uri,
                                                    rabbitmq_queue,
                                                    rabbitmq_exchange,
                                                    rabbitmq_routing_key,
                                                    wp5_rabbitmq_uri,
                                                    wp5_rabbitmq_queue,
                                                    wp5_rabbitmq_exchange,
                                                    wp5_rabbitmq_routing_key,
                                                    latest_n,
                                                    lower_timestamp,
                                                    upper_timestamp,
                                                    timestamp_sort_key,
                                                    number_of_threads,
                                                    is_replayed_stream,
                                                    popularity_prediction_db,
                                                    local_resources_folder,
                                                    reddit_oauth_credentials_path,
                                                    youtube_oauth_credentials_folder,
                                                    youtube_module_communication,
                                                    platform_resources_path):
    """
    Performs Online Social Network user classification.

    Specifically:
           -
    """
    ####################################################################################################################
    # Manage argument input.
    ####################################################################################################################
    spec = make_time_window_filter(lower_timestamp, upper_timestamp)

    if number_of_threads is None:
        number_of_threads = get_threads_number()

    ####################################################################################################################
    # Establish MongoDB connection.
    ####################################################################################################################
    mongo_client,\
    tweet_input_mongo_database_name,\
    tweet_input_mongo_collection_name = safe_establish_mongo_connection(mongo_uri, assessment_id)

    if is_replayed_stream:
        print("This is the replayed stream of the assessment", tweet_input_mongo_database_name)
    print("MongoDB connection established.")

    ####################################################################################################################
    # Tweet processing - URL extraction.
    ####################################################################################################################
    url_list,\
    max_tweet_timestamp = process_tweets_and_extract_urls(mongo_client,
                                                          tweet_input_mongo_database_name,
                                                          tweet_input_mongo_collection_name,
                                                          spec,
                                                          latest_n,
                                                          timestamp_sort_key,
                                                          is_replayed_stream)
    print("Tweets processing and YouTube/Reddit URLs extraction initiated.")

    ####################################################################################################################
    # Collect discussions and build graphs.
    ####################################################################################################################
    social_context_generator = collect_social_context(url_list,
                                                      max_tweet_timestamp,
                                                      reddit_oauth_credentials_path,
                                                      youtube_oauth_credentials_folder,
                                                      youtube_module_communication)
    print("Social context (discussion) collection initiated.")

    graph_generator = form_graphs(social_context_generator,
                                  max_tweet_timestamp)
    print("Social graph formation initiated.")

    ####################################################################################################################
    # Extract features and make predictions.
    ####################################################################################################################
    features_generator = extract_features(graph_generator, max_tweet_timestamp)
    print("Feature extraction initiated.")

    predictions_generator = make_predictions(features_generator,
                                             number_of_threads,
                                             platform_resources_path)
    print("Modality prediction initiated.")

    ####################################################################################################################
    # Output.
    ####################################################################################################################
    wp6_rabbitmq_dict = dict()
    wp6_rabbitmq_dict["rabbitmq_uri"] = rabbitmq_uri
    wp6_rabbitmq_dict["rabbitmq_queue"] = rabbitmq_queue
    wp6_rabbitmq_dict["rabbitmq_exchange"] = rabbitmq_exchange
    wp6_rabbitmq_dict["rabbitmq_routing_key"] = rabbitmq_routing_key

    wp5_rabbitmq_dict = dict()
    wp5_rabbitmq_dict["rabbitmq_uri"] = wp5_rabbitmq_uri
    wp5_rabbitmq_dict["rabbitmq_queue"] = wp5_rabbitmq_queue
    wp5_rabbitmq_dict["rabbitmq_exchange"] = wp5_rabbitmq_exchange
    wp5_rabbitmq_dict["rabbitmq_routing_key"] = wp5_rabbitmq_routing_key
    publish_output(predictions_generator,
                   mongo_client,
                   tweet_input_mongo_database_name,
                   wp6_rabbitmq_dict,
                   wp5_rabbitmq_dict)
    print("Module output terminated.")
