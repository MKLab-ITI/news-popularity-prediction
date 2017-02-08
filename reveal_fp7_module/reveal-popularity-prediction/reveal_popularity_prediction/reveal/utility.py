__author__ = 'Georgios Rizos (georgerizos@iti.gr)'

import time
import datetime
from queue import Queue
from threading import Thread

from reveal_popularity_prediction.input_processing.mongo import establish_mongo_connection,\
    get_safe_mongo_generator, get_collection_documents_generator
from reveal_popularity_prediction.input_processing.twitter import extract_urls_from_tweets
from reveal_popularity_prediction.builder.collect.youtube import social_context as youtube_social_context
from reveal_popularity_prediction.builder.collect.reddit import social_context as reddit_social_context
from reveal_popularity_prediction.builder.build_graphs import get_snapshot_graphs
from reveal_popularity_prediction.features.extraction import extract_snapshot_features, make_features_vector
from reveal_popularity_prediction.inference.preprocessed_data_usage import get_appropriate_regressor_models,\
    popularity_prediction, decide_snapshot_for_learning
from reveal_popularity_prediction.output import wp5_output, wp6_output, rabbitmq_util


def make_time_window_filter(lower_timestamp, upper_timestamp):
    if lower_timestamp is not None:
        # lower_datetime = datetime.datetime.strftime(datetime.datetime.utcfromtimestamp(lower_timestamp),
        #                                             "%b %d %Y %H:%M:%S")
        lower_datetime = datetime.datetime.utcfromtimestamp(lower_timestamp)
        if upper_timestamp is not None:
            # upper_datetime = datetime.datetime.strftime(datetime.datetime.utcfromtimestamp(upper_timestamp),
            #                                             "%b %d %Y %H:%M:%S")
            upper_datetime = datetime.datetime.utcfromtimestamp(upper_timestamp)
            # Both timestamps are defined.
            spec = dict()
            spec["created_at"] = {"$gte": lower_datetime, "$lt": upper_datetime}
        else:
            spec = dict()
            spec["created_at"] = {"$gte": lower_datetime}
    else:
        if upper_timestamp is not None:
            # upper_datetime = datetime.datetime.strftime(datetime.datetime.utcfromtimestamp(upper_timestamp),
            #                                             "%b %d %Y %H:%M:%S")
            upper_datetime = datetime.datetime.utcfromtimestamp(upper_timestamp)
            spec = dict()
            spec["created_at"] = {"$lt": upper_datetime}
        else:
            spec = None
    return spec


def safe_establish_mongo_connection(mongo_uri, assessment_id):
    """
           - tweet_input_database_name: The mongo database name where the input tweets are stored.
           - tweet_input_collection_name: The mongo collection name where the input tweets are stored.
    """

    client = establish_mongo_connection(mongo_uri)

    tweet_input_database_name, tweet_input_collection_name = translate_assessment_id(assessment_id)

    return client, tweet_input_database_name, tweet_input_collection_name


def translate_assessment_id(assessment_id):
    """
    The assessment id is translated to MongoDB database and collection names.

    Input:   - assessment_id: The connection details for making a connection with a MongoDB instance.

    Outputs: - database_name: The name of the Mongo database in string format.
             - collection_name: The name of the collection of tweets to read in string format.
    """
    assessment_id = assessment_id.split(".")

    database_name = assessment_id[0]
    collection_name = assessment_id[1]

    return database_name, collection_name


def process_tweets_and_extract_urls(client,
                                    tweet_input_database_name,
                                    tweet_input_collection_name,
                                    spec,
                                    latest_n,
                                    timestamp_sort_key,
                                    is_replayed_stream):
    # Form the mention and retweet graphs, as well as the attribute matrix.
    tweet_gen = get_collection_documents_generator(client=client,
                                                   database_name=tweet_input_database_name,
                                                   collection_name=tweet_input_collection_name,
                                                   spec=spec,
                                                   latest_n=latest_n,
                                                   sort_key=timestamp_sort_key)

    url_list, assessment_timestamp = extract_urls_from_tweets(tweet_gen, is_replayed_stream)

    return url_list, assessment_timestamp


def collect_social_context(url_generator,
                           assessment_timestamp,
                           reddit_oauth_credentials_path,
                           youtube_oauth_credentials_folder,
                           youtube_module_communication):
    # Fill queues.
    youtube_queue = Queue()
    reddit_queue = Queue()
    social_context_queue = Queue()
    url_counter = 0
    counter_to_dict = dict()
    for url_dict in url_generator:
        url_dict["assessment_timestamp"] = assessment_timestamp
        counter_to_dict[url_counter] = url_dict

        url = url_dict["url"]
        # upper_timestamp = float(url_dict["tweet_timestamp"])
        upper_timestamp = assessment_timestamp

        platform_name = url_dict["platform_name"]
        if platform_name == "YouTube":
            youtube_queue.put((url_counter, url, upper_timestamp))
        elif platform_name == "Reddit":
            reddit_queue.put((url_counter, url, upper_timestamp))
        else:
            print("Invalid platform name.")
            continue

        url_counter += 1

    # Summon the daemons.
    youtube_thread_batch_size_max = 16
    reddit_thread_batch_size_max = 1

    youtube_daemons = list()
    for i in range(youtube_thread_batch_size_max):
        worker = Thread(target=youtube_daemon_worker,
                        args=(i, youtube_queue, social_context_queue, youtube_module_communication, youtube_oauth_credentials_folder),
                        name="youtube-daemon-{}".format(i))
        worker.setDaemon(True)
        worker.start()
        youtube_daemons.append(worker)
    reddit_daemons = list()
    for i in range(reddit_thread_batch_size_max):
        worker = Thread(target=reddit_daemon_worker,
                        args=(i, reddit_queue, social_context_queue, reddit_oauth_credentials_path),
                        name="reddit-daemon-{}".format(i))
        worker.setDaemon(True)
        worker.start()
        reddit_daemons.append(worker)

    urls_to_process = url_counter
    urls_processed = 0
    # Yield the results.

    # fp = open("/home/georgerizos/Documents/fetch_times/fetch_time" + ".txt", "a")
    while urls_processed < urls_to_process:
        if social_context_queue.empty():
            time.sleep(0.1)
        else:
            url_counter, social_context = social_context_queue.get()
            urls_processed += 1

            # print(social_context)
            if urls_processed % 20 == 0:
                print("Items processed: %d out of %d" % (urls_processed, urls_to_process))

            if social_context is None:
                continue
            # fp.write(repr(social_context["elapsed_time"]) + "\n")

            url_dict = counter_to_dict[url_counter]

            social_context_dict = url_dict
            social_context_dict["social_context"] = social_context

            yield social_context_dict

        if urls_processed >= urls_to_process:
            break


def youtube_daemon_worker(id, youtube_queue, social_context_queue, youtube_module_communication, youtube_oauth_credentials_folder):
    while True:
        if social_context_queue.qsize() > 50:
            time.sleep(10.0)
        else:
            url_counter, url, upper_timestamp = youtube_queue.get()

            try:
                # start_time = time.perf_counter()
                social_context = youtube_social_context.collect(url, youtube_module_communication + "_" + str(id), youtube_oauth_credentials_folder)
                # elapsed_time = time.perf_counter() - start_time
                # if social_context is not None:
                #     social_context["elapsed_time"] = elapsed_time
            except KeyError:
                social_context = None

            social_context_queue.put((url_counter, social_context))
            youtube_queue.task_done()


def reddit_daemon_worker(id, reddit_queue, social_context_queue, reddit_oauth_credentials_path):
    while True:
        if social_context_queue.qsize() > 50:
            time.sleep(10.0)
        else:
            url_counter, url, upper_timestamp = reddit_queue.get()

            try:
                # start_time = time.perf_counter()
                social_context = reddit_social_context.collect(url, reddit_oauth_credentials_path)
                # elapsed_time = time.perf_counter() - start_time
                # if social_context is not None:
                #     social_context["elapsed_time"] = elapsed_time
            except KeyError:
                social_context = None

            social_context_queue.put((url_counter, social_context))
            reddit_queue.task_done()


def form_graphs(social_context_generator, assessment_timestamp):
    # fp = open("/home/georgerizos/Documents/fetch_times/build_graph_time" + ".txt", "a")
    for social_context_dict in social_context_generator:
        # start_time = time.perf_counter()
        snapshots,\
        targets,\
        title = get_snapshot_graphs(social_context_dict["social_context"],
                                    # social_context_dict["tweet_timestamp"],
                                    assessment_timestamp,
                                    social_context_dict["platform_name"])
        # elapsed_time = time.perf_counter() - start_time
        # fp.write(repr(elapsed_time) + "\n")
        if snapshots is None:
            continue

        if len(snapshots) > 1:
            graph_dict = social_context_dict
            graph_dict["snapshots"] = snapshots
            graph_dict["targets"] = targets
            graph_dict["title"] = title
            yield graph_dict


def extract_features(graph_generator, assessment_timestamp):
    # fp = open("/home/georgerizos/Documents/fetch_times/extract_features_time" + ".txt", "a")
    for graph_snapshot_dict in graph_generator:
        # start_time = time.perf_counter()
        snapshots = graph_snapshot_dict["snapshots"]

        initial_post = graph_snapshot_dict["social_context"]["initial_post"]
        author = graph_snapshot_dict["social_context"]["author"]
        platform = graph_snapshot_dict["platform_name"]

        snapshots_with_features = list()
        tweet_timestamp = graph_snapshot_dict["tweet_timestamp"]
        for snapshot_dict in snapshots:
            comment_tree = snapshot_dict["comment_tree"]
            user_graph = snapshot_dict["user_graph"]
            timestamp_list = snapshot_dict["timestamp_list"]

            features = extract_snapshot_features(comment_tree,
                                                 user_graph,
                                                 timestamp_list,
                                                 assessment_timestamp,
                                                 # tweet_timestamp,
                                                 initial_post,
                                                 author,
                                                 platform)
            snapshot_dict["features"] = features

            snapshots_with_features.append(snapshot_dict)

        features_dict = graph_snapshot_dict
        features_dict["snapshots"] = snapshots_with_features

        # elapsed_time = time.perf_counter() - start_time
        # fp.write(repr(elapsed_time) + "\n")

        yield features_dict


def make_predictions(features_generator,
                     number_of_threads,
                     platform_resources_path):
    # fp = open("/home/georgerizos/Documents/fetch_times/make_predictions_time" + ".txt", "a")
    for features_dict in features_generator:
        # start_time = time.perf_counter()
        platform = features_dict["platform_name"]
        if features_dict["platform_name"] == "YouTube":
            platform_resources_path_eff = platform_resources_path + "youtube/"
        elif features_dict["platform_name"] == "Reddit":
            platform_resources_path_eff = platform_resources_path + "reddit/"
        else:
            print("Invalid platform name.")
            raise RuntimeError

        snapshot_for_learning = decide_snapshot_for_learning(features_dict["snapshots"],
                                                             platform_resources_path_eff)

        last_comment_timestamp = features_dict["snapshots"][snapshot_for_learning]["timestamp_list"][-1]
        post_timestamp = features_dict["snapshots"][snapshot_for_learning]["timestamp_list"][0]
        post_lifetime = last_comment_timestamp - post_timestamp
        if post_lifetime < 0.0:
            post_lifetime = 0.0

        features = features_dict["snapshots"][snapshot_for_learning]["features"]
        features_vector = make_features_vector(features,
                                               platform)

        regressor_models = get_appropriate_regressor_models(post_lifetime,
                                                            platform_resources_path_eff,
                                                            number_of_threads)

        # post_timestamp = features_dict["snapshots"][-1]["timestamp_list"][0]
        # last_comment_timestamp = features_dict["snapshots"][-1]["timestamp_list"][-1]

        predictions,\
        prediction_window = popularity_prediction(features_vector, regressor_models, post_timestamp, last_comment_timestamp)

        predictions_dict = features_dict
        predictions_dict["predictions"] = predictions
        predictions_dict["prediction_window"] = prediction_window

        # elapsed_time = time.perf_counter() - start_time
        # fp.write(repr(elapsed_time) + "\n")

        yield predictions_dict


def publish_output(predictions_generator,
                   mongo_client,
                   tweet_input_mongo_database_name,
                   wp6_rabbitmq_dict,
                   wp5_rabbitmq_dict):
    rabbitmq_util.rabbitmq_server_service("restart")

    wp5_rabbitmq_uri = wp5_rabbitmq_dict["rabbitmq_uri"]
    wp5_rabbitmq_queue = wp5_rabbitmq_dict["rabbitmq_queue"]
    wp5_rabbitmq_exchange = wp5_rabbitmq_dict["rabbitmq_exchange"]
    wp5_rabbitmq_routing_key = wp5_rabbitmq_dict["rabbitmq_routing_key"]
    wp6_rabbitmq_uri = wp6_rabbitmq_dict["rabbitmq_uri"]
    wp6_rabbitmq_queue = wp6_rabbitmq_dict["rabbitmq_queue"]
    wp6_rabbitmq_exchange = wp6_rabbitmq_dict["rabbitmq_exchange"]
    wp6_rabbitmq_routing_key = wp6_rabbitmq_dict["rabbitmq_routing_key"]

    wp5_rabbitmq_uri,\
    wp5_rabbitmq_queue,\
    wp5_rabbitmq_exchange,\
    wp5_rabbitmq_routing_key = wp5_output.check_wp5_rabbitmq_connection(wp5_rabbitmq_uri,
                                                                        wp5_rabbitmq_queue,
                                                                        wp5_rabbitmq_exchange,
                                                                        wp5_rabbitmq_routing_key,
                                                                        wp6_rabbitmq_uri,
                                                                        wp6_rabbitmq_queue,
                                                                        wp6_rabbitmq_exchange,
                                                                        wp6_rabbitmq_routing_key,
                                                                        tweet_input_mongo_database_name)
    wp5_rabbitmq_dict["rabbitmq_uri"] = wp5_rabbitmq_uri
    wp5_rabbitmq_dict["rabbitmq_queue"] = wp5_rabbitmq_queue
    wp5_rabbitmq_dict["rabbitmq_exchange"] = wp5_rabbitmq_exchange
    wp5_rabbitmq_dict["rabbitmq_routing_key"] = wp5_rabbitmq_routing_key

    rabbitmq_util.rabbitmq_server_service("restart")

    wp6_connection = rabbitmq_util.establish_rabbitmq_connection(wp6_rabbitmq_uri)
    wp6_channel = rabbitmq_util.get_channel(wp6_connection, wp6_rabbitmq_queue, wp6_rabbitmq_exchange, wp6_rabbitmq_routing_key)
    wp5_connection = rabbitmq_util.establish_rabbitmq_connection(wp5_rabbitmq_uri)
    wp5_channel = rabbitmq_util.get_channel(wp5_connection, wp5_rabbitmq_queue, wp5_rabbitmq_exchange, wp5_rabbitmq_routing_key)

    wp6_rabbitmq_dict["channel"] = wp6_channel
    wp5_rabbitmq_dict["channel"] = wp5_channel

    for prediction_json in predictions_generator:
        wp6_output.write_to_mongo(prediction_json,
                                  mongo_client,
                                  tweet_input_mongo_database_name)
        wp5_output.publish_to_wp5(prediction_json,
                                  wp5_rabbitmq_dict,
                                  tweet_input_mongo_database_name)
        wp6_output.notify_wp6_item(prediction_json,
                                   wp6_rabbitmq_dict)
    wp6_output.notify_wp6_stream(wp6_rabbitmq_dict)

    wp6_connection.close()
    wp5_connection.close()
