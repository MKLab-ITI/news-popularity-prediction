__author__ = 'Georgios Rizos (georgerizos@iti.gr)'

import json
import datetime

import numpy as np
import scipy.sparse as spsp

from news_popularity_prediction.discussion.reddit import document_generator, get_post_url, calculate_targets,\
    comment_generator, extract_comment_name, extract_user_name, extract_parent_comment_name, extract_timestamp
from news_popularity_prediction.learning.cascade_lifetime import get_h5_stores_and_keys
from news_popularity_prediction.discussion.builder import within_discussion_comment_and_user_anonymization,\
    safe_comment_generator, initialize_timestamp_array, initialize_intermediate, update_discussion_and_user_graphs,\
    update_timestamp_array, update_intermediate


def make_dataset_json(output_file_path,
                      raw_data_file_path,
                      features_folder,
                      comparison_lifetimes_path,
                      anonymous_coward_name):
    # Read raw data.
    document_gen = document_generator([raw_data_file_path])

    # Read features.
    h5_stores_and_keys = get_h5_stores_and_keys(features_folder,
                                                "reddit")

    # Read comparison lifetimes.
    lifetime_list = get_comparison_lifetimes(comparison_lifetimes_path)

    with open(output_file_path, "w") as fp:
        for document in document_gen:
            timestamp_df,\
            handcrafted_df = get_features_df(document, h5_stores_and_keys)

            # if handcrafted_df is None:
            #     continue

            discussion_json = make_discussion_json(document, timestamp_df, handcrafted_df, lifetime_list, anonymous_coward_name)

            if discussion_json is None:
                continue

            json.dump(discussion_json, fp)

            fp.write("\n\n")


def get_comparison_lifetimes(file_path):
    comparison_lifetimes = list()

    with open(file_path, "r") as fp:
        for file_row in fp:
            clean_row = file_row.strip()
            if clean_row == "":
                continue
            comparison_lifetimes.append(float(clean_row))

    return comparison_lifetimes


def get_features_df(document, h5_stores_and_keys):
    post_id = document["post_id"]

    post_key = "/data/reddit_" + post_id

    timestamp_df = None
    handcrafted_df = None

    for h5_stores, keys in h5_stores_and_keys:
        if post_key in keys["reddit"]:
            timestamp_df = h5_stores[0][post_key]
            handcrafted_df = h5_stores[1][post_key]

    return timestamp_df, handcrafted_df


def make_discussion_json(document, timestamp_df, handcrafted_df, lifetime_list, anonymous_coward_name):
    discussion_json = dict()

    discussion_json["post_url"] = get_post_url(document)
    discussion_json["graph_snapshots"] = list()

    comment_gen = comment_generator(document=document)

    comment_name_set,\
    user_name_set,\
    within_discussion_comment_anonymize,\
    within_discussion_user_anonymize,\
    within_discussion_anonymous_coward = within_discussion_comment_and_user_anonymization(comment_gen=comment_gen,
                                                                                          extract_comment_name=extract_comment_name,
                                                                                          extract_user_name=extract_user_name,
                                                                                          anonymous_coward_name=anonymous_coward_name)

    try:
        discussion_json["prediction_targets"] = calculate_targets(document,
                                                                  comment_name_set,
                                                                  user_name_set,
                                                                  within_discussion_anonymous_coward)
    except KeyError as e:
        return None

    try:
        safe_comment_gen = safe_comment_generator(document=document,
                                                  comment_generator=comment_generator,
                                                  within_discussion_comment_anonymize=within_discussion_comment_anonymize,
                                                  extract_comment_name=extract_comment_name,
                                                  extract_parent_comment_name=extract_parent_comment_name,
                                                  extract_timestamp=extract_timestamp,
                                                  safe=True)
    except TypeError:
        return None

    try:
        initial_post = next(safe_comment_gen)
    except TypeError:
        return None
    try:
        timestamp = extract_timestamp(initial_post)
    except TypeError:
        return None
    op_raw_id = extract_user_name(initial_post)
    op_id = within_discussion_user_anonymize[op_raw_id]
    if op_id == within_discussion_anonymous_coward:
        op_is_anonymous = True
    else:
        op_is_anonymous = False

    comment_counter = 0

    timestamp_column_names_list,\
    timestamp_array = initialize_timestamp_array(discussion_json["prediction_targets"]["comments"] + 1,
                                                 cascade_source_timestamp=timestamp)

    intermediate_dict = initialize_intermediate(comment_name_set,
                                                user_name_set,
                                                timestamp,
                                                within_discussion_anonymous_coward,
                                                op_is_anonymous=op_is_anonymous)

    comment_tree = spsp.dok_matrix((len(comment_name_set),
                                    len(comment_name_set)),
                                   dtype=np.int8)

    user_graph = spsp.dok_matrix((len(user_name_set),
                                  len(user_name_set)),
                                 dtype=np.int32)

    current_lifetime = 0.0

    for lifetime_counter, lifetime in enumerate(lifetime_list):
        while True:
            try:
                comment = next(safe_comment_gen)
            except TypeError:
                return None
            except StopIteration:
                handcrafted_df_row = handcrafted_df.iloc[comment_counter]

                time_step_json = make_time_step_json(current_lifetime,
                                                     comment_tree,
                                                     user_graph,
                                                     timestamp_array[comment_counter, 1],
                                                     handcrafted_df_row)
                discussion_json["graph_snapshots"].append(time_step_json)
                break
            if comment is None:
                return None

            comment_counter += 1

            commenter_name = extract_user_name(comment)
            if commenter_name is None:
                commenter_is_anonymous = True
            else:
                commenter_is_anonymous = False

            try:
                discussion_tree,\
                user_graph,\
                comment_id,\
                parent_comment_id,\
                commenter_id,\
                parent_commenter_id,\
                user_graph_modified,\
                parent_commenter_is_anonymous,\
                comment_id_to_user_id = update_discussion_and_user_graphs(comment=comment,
                                                                          extract_comment_name=extract_comment_name,
                                                                          extract_parent_comment_name=extract_parent_comment_name,
                                                                          extract_user_name=extract_user_name,
                                                                          discussion_tree=comment_tree,
                                                                          user_graph=user_graph,
                                                                          within_discussion_comment_anonymize=within_discussion_comment_anonymize,
                                                                          within_discussion_user_anonymize=within_discussion_user_anonymize,
                                                                          within_discussion_anonymous_coward=within_discussion_anonymous_coward,
                                                                          comment_id_to_user_id=intermediate_dict["comment_id_to_user_id"])
                intermediate_dict["comment_id_to_user_id"] = comment_id_to_user_id
            except RuntimeError:
                return None

            try:
                timestamp = extract_timestamp(comment)
            except TypeError:
                return None

            update_timestamp_array(timestamp_column_names_list,
                                   timestamp_array,
                                   timestamp,
                                   comment_counter)
            timestamp_difference = timestamp_array[comment_counter, 1] - timestamp_array[comment_counter-1, 1]

            try:
                intermediate_dict,\
                comment_depth = update_intermediate(discussion_tree,
                                                    user_graph,
                                                    intermediate_dict,
                                                    commenter_is_anonymous,
                                                    parent_commenter_is_anonymous,
                                                    comment_id,
                                                    parent_comment_id,
                                                    commenter_id,
                                                    parent_commenter_id,
                                                    user_graph_modified,
                                                    timestamp,
                                                    timestamp_difference)
            except RuntimeError:
                return None

            current_lifetime = timestamp_array[comment_counter, 1] - timestamp_array[0, 1]
            if current_lifetime >= lifetime:
                # Read features.
                # handcrafted_df_row = handcrafted_df[feature_list]
                handcrafted_df_row = handcrafted_df.iloc[comment_counter]

                time_step_json = make_time_step_json(current_lifetime,
                                                     comment_tree,
                                                     user_graph,
                                                     timestamp_array[comment_counter, 1],
                                                     handcrafted_df_row)
                discussion_json["graph_snapshots"].append(time_step_json)
                break

    return discussion_json


def make_time_step_json(lifetime,
                        comment_tree,
                        user_graph,
                        timestamp,
                        handcrafted_df_row):
    time_step_json = dict()

    time_step_json["lifetime_seconds"] = get_human_readable_lifetime(lifetime)
    time_step_json["lifetime_date"] = get_date(timestamp)

    time_step_json["comment_tree"] = make_graph_json(comment_tree)
    time_step_json["user_graph"] = make_graph_json(user_graph)

    time_step_json["features"] = make_features_json(handcrafted_df_row)

    return time_step_json


def get_human_readable_lifetime(lifetime):

    hours = lifetime // 3600
    lifetime %= 3600
    minutes = lifetime // 60
    lifetime %= 60
    seconds = lifetime

    human_readable_lifetime = str(int(hours)) + " hours and " + str(int(minutes)) + " minutes after posting."

    return human_readable_lifetime


def get_date(timestamp):
    date = datetime.datetime.utcfromtimestamp(timestamp).strftime('%Y-%m-%d %H:%M:%S')
    return date


def make_graph_json(graph):
    graph_json = list()

    graph = spsp.csr_matrix(graph)
    graph.sum_duplicates()
    graph = spsp.coo_matrix(graph)

    for i, j, k in zip(graph.row, graph.col, graph.data):
        graph_json.append((str(int(i)), str(int(j)), str(int(k))))

    return graph_json


def make_features_json(features):
    features_json = dict()

    for column in features.index:
        features_json[column] = features.loc[column]

    return features_json


OUTPUT_FILE_PATH = "/home/georgerizos/Documents/LocalStorage/lazer.txt"
RAW_DATA_FILE_PATH = "/home/georgerizos/Documents/LocalStorage/raw_data/Rizos/Reddit_News/reddit_news_dataset.txt"
FEATURES_FOLDER = "/home/georgerizos/Documents/LocalStorage/memory/Rizos/Reddit_News/uniform"
COMPARISON_LIFETIMES_PATH = "/home/georgerizos/Documents/LocalStorage/memory/Rizos/Reddit_News/uniform/k_list/akis_focus_reddit.txt"
ANONYMOUS_COWARD_NAME = "[deleted]"

make_dataset_json(output_file_path=OUTPUT_FILE_PATH,
                  raw_data_file_path=RAW_DATA_FILE_PATH,
                  features_folder=FEATURES_FOLDER,
                  comparison_lifetimes_path=COMPARISON_LIFETIMES_PATH,
                  anonymous_coward_name=ANONYMOUS_COWARD_NAME)
