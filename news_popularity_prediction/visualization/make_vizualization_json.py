__author__ = 'Georgios Rizos (georgerizos@iti.gr)'

import json
import datetime

import numpy as np
import scipy.sparse as spsp
from scipy.stats import rankdata

from news_popularity_prediction.discussion.reddit import document_generator, get_post_url, get_post_title,\
    calculate_targets, comment_generator, extract_comment_name, extract_user_name, extract_parent_comment_name,\
    extract_timestamp
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

    post_ids_to_keep = decide_posts_to_keep(raw_data_file_path, anonymous_coward_name)

    with open(output_file_path, "w") as fp:
        fp.write("[\n")
        for document in document_gen:

            if document["post_id"] in post_ids_to_keep:
                timestamp_df,\
                handcrafted_df = get_features_df(document, h5_stores_and_keys)

                # if handcrafted_df is None:
                #     continue

                discussion_json = make_discussion_json(document, timestamp_df, handcrafted_df, lifetime_list, anonymous_coward_name)

                if discussion_json is None:
                    continue

                json.dump(discussion_json, fp)

                fp.write(",\n\n")
        fp.write("]\n")


def get_comparison_lifetimes(file_path):
    comparison_lifetimes = list()

    with open(file_path, "r") as fp:
        for file_row in fp:
            clean_row = file_row.strip()
            if clean_row == "":
                continue
            comparison_lifetimes.append(float(clean_row))

    return comparison_lifetimes


def decide_posts_to_keep(raw_data_file_path, anonymous_coward_name):
    # Read raw data.
    document_gen = document_generator([raw_data_file_path])

    post_to_targets = dict()

    for document in document_gen:
        comment_gen = comment_generator(document=document)

        ################################################################################################################
        # Within-discussion comment and user anonymization.
        ################################################################################################################
        comment_name_set,\
        user_name_set,\
        within_discussion_comment_anonymize,\
        within_discussion_user_anonymize,\
        within_discussion_anonymous_coward = within_discussion_comment_and_user_anonymization(comment_gen=comment_gen,
                                                                                              extract_comment_name=extract_comment_name,
                                                                                              extract_user_name=extract_user_name,
                                                                                              anonymous_coward_name=anonymous_coward_name)

        ################################################################################################################
        # Calculate prediction targets.
        ################################################################################################################
        try:
            targets = calculate_targets(document,
                                        comment_name_set,
                                        user_name_set,
                                        within_discussion_anonymous_coward)
        except KeyError as e:
            continue

        if targets["comments"] > 1:
            post_id = document["post_id"]
            post_to_targets[post_id] = targets

    post_id_list = list()
    comments_list = list()
    users_list = list()
    score_list = list()
    controversiality_list = list()
    for post_id, targets in post_to_targets.items():
        post_id_list.append(post_id)
        comments_list.append(targets["comments"])
        users_list.append(targets["users"])
        score_list.append(targets["score_wilson"])
        controversiality_list.append(targets["controversiality_wilson"])

    n = len(post_id_list)

    post_id_list = np.array(post_id_list)
    comments_list = np.array(comments_list)
    users_list = np.array(users_list)
    score_list = np.array(score_list)
    controversiality_list = np.array(controversiality_list)

    # Rank according to comments.
    comments_rank = rankdata(- comments_list)
    i_comments = np.argsort(comments_list)
    post_id_list_comments = post_id_list[i_comments]
    comments_list = comments_list[i_comments]

    print(np.max(comments_list))

    # Rank according to users.
    users_rank = rankdata(- users_list)
    i_users = np.argsort(users_list)
    post_id_list_users = post_id_list[i_users]
    users_list = users_list[i_users]
    print(np.max(users_list))

    # Rank according to score_wilson.
    score_rank = rankdata(- score_list)
    i_score = np.argsort(score_list)
    post_id_list_score = post_id_list[i_score]
    score_list = score_list[i_score]
    print(np.max(score_list))

    # Rank according to controversiality_wilson.
    controversiality_rank = rankdata(- controversiality_list)
    i_controversiality = np.argsort(controversiality_list)
    post_id_list_controversiality = post_id_list[i_controversiality]
    controversiality_list = controversiality_list[i_controversiality]
    print(np.max(controversiality_list))

    # Rank according to all.
    all_rank = comments_rank + users_rank + score_rank + controversiality_rank
    i = np.argsort(all_rank)
    post_id_list_new = post_id_list[i][::-1]

    # Select 500 posts.
    post_id_chunk_list = [chunk[-1] for chunk in split_list(list(post_id_list_new), 500)]

    for post_id in post_id_chunk_list:
        print(post_to_targets[post_id])

    return set(post_id_chunk_list)


def split_list(l, k):
    n = len(l)

    d = n // k
    r = n % k

    offset = 0
    for i in range(k):
        if i < r:
            size = d + 1
        else:
            size = d

        yield l[offset:offset+size]
        offset += size


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
    discussion_json["post_title"] = get_post_title(document)
    # discussion_json["snapshot_timestamps"] = [repr(float(snapshot_timestamp)) for snapshot_timestamp in lifetime_list]
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

    # lifetime_list.append(np.inf)
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

    discussion_json["post_timestamp"] = timestamp_array[0, 1]
    # discussion_json["final_comment_tree_size"] = discussion_json["prediction_targets"]["comments"] + 1
    # discussion_json["final_user_graph_size"] = discussion_json["prediction_targets"]["users"]

    return discussion_json


def make_time_step_json(lifetime,
                        comment_tree,
                        user_graph,
                        timestamp,
                        handcrafted_df_row):
    time_step_json = dict()

    # time_step_json["lifetime_seconds"] = get_human_readable_lifetime(lifetime)
    time_step_json["lifetime_seconds"] = lifetime
    # time_step_json["lifetime_date"] = get_date(timestamp)
    # time_step_json["timestamp"] = timestamp

    time_step_json["comment_tree"] = make_graph_json(comment_tree)
    time_step_json["user_graph"] = make_graph_json(user_graph)

    # time_step_json["comment_tree_node_list"] = get_graph_node_list(comment_tree)
    time_step_json["user_graph_node_list"] = get_graph_node_list(user_graph)

    # time_step_json["comment_tree_size"] = len(time_step_json["comment_tree_node_list"])
    # time_step_json["user_graph_size"] = len(time_step_json["user_graph_node_list"])

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


def get_graph_node_list(graph):

    node_list = list()

    graph = spsp.coo_matrix(graph)

    node_list.extend(list(graph.row))
    node_list.extend(list(graph.col))

    node_list = list(set(node_list))
    node_list = [str(int(n)) for n in node_list]

    return node_list


def make_graph_json(graph):
    graph_json = list()

    graph = spsp.csr_matrix(graph)
    graph.sum_duplicates()
    graph = spsp.coo_matrix(graph)

    for i, j, k in zip(graph.row, graph.col, graph.data):
        # graph_json.append((str(int(i)), str(int(j)), str(int(k))))
        graph_json.append((str(int(i)), str(int(j))))

    return graph_json


def make_features_json(features):
    features_json = dict()

    for column in features.index:
        features_json[column] = features.loc[column]

    return features_json


def make_description_texts_json(output_file_path):
    description_texts_json = dict()

    description_texts_json["target_descriptions"] = dict()
    description_texts_json["feature_descriptions"] = dict()

    # Target descriptions.

    description_texts_json["target_descriptions"]["comments"] = dict()
    description_texts_json["target_descriptions"]["comments"]["pretty_name"] = "Final Comment Count"
    description_texts_json["target_descriptions"]["comments"]["description"] = "The number of comments in the discussion at the time of the data collection."

    description_texts_json["target_descriptions"]["users"] = dict()
    description_texts_json["target_descriptions"]["users"]["pretty_name"] = "Final User Count"
    description_texts_json["target_descriptions"]["users"]["description"] = "The number of eponymous users in the discussion at the time of the data collection."

    description_texts_json["target_descriptions"]["score_wilson"] = dict()
    description_texts_json["target_descriptions"]["score_wilson"]["pretty_name"] = "Final Score"
    description_texts_json["target_descriptions"]["score_wilson"]["description"] = "The number of upvotes minus downvotes (penalized for few votes) at the time of the data collection."

    description_texts_json["target_descriptions"]["controversiality_wilson"] = dict()
    description_texts_json["target_descriptions"]["controversiality_wilson"]["pretty_name"] = "Final Controversiality"
    description_texts_json["target_descriptions"]["controversiality_wilson"]["description"] = "The number of vote disagreements (penalized for few votes) at the time of the data collection."

    # Comment tree descriptions

    description_texts_json["feature_descriptions"]["basic_comment_count"] = dict()
    description_texts_json["feature_descriptions"]["basic_comment_count"]["pretty_name"] = "Comment Count"
    description_texts_json["feature_descriptions"]["basic_comment_count"]["description"] = "The number of comments in the discussion at the time of the snapshot."

    description_texts_json["feature_descriptions"]["basic_max_depth"] = dict()
    description_texts_json["feature_descriptions"]["basic_max_depth"]["pretty_name"] = "Max Depth"
    description_texts_json["feature_descriptions"]["basic_max_depth"]["description"] = "The maximum node depth in the comment tree."

    description_texts_json["feature_descriptions"]["basic_ave_depth"] = dict()
    description_texts_json["feature_descriptions"]["basic_ave_depth"]["pretty_name"] = "Avg. Depth"
    description_texts_json["feature_descriptions"]["basic_ave_depth"]["description"] = "The average node depth in the comment tree."

    description_texts_json["feature_descriptions"]["basic_max_width"] = dict()
    description_texts_json["feature_descriptions"]["basic_max_width"]["pretty_name"] = "Max Width"
    description_texts_json["feature_descriptions"]["basic_max_width"]["description"] = "The maximum size of a depth level in the comment tree."

    description_texts_json["feature_descriptions"]["basic_ave_width"] = dict()
    description_texts_json["feature_descriptions"]["basic_ave_width"]["pretty_name"] = "Avg. Width"
    description_texts_json["feature_descriptions"]["basic_ave_width"]["description"] = "The average size of depth levels in the comment tree."

    description_texts_json["feature_descriptions"]["basic_max_depth_max_width_ratio"] = dict()
    description_texts_json["feature_descriptions"]["basic_max_depth_max_width_ratio"]["pretty_name"] = "Max Depth over Max Width"
    description_texts_json["feature_descriptions"]["basic_max_depth_max_width_ratio"]["description"] = "The maximum depth over the maximum width ratio in the comment tree."

    description_texts_json["feature_descriptions"]["basic_depth_width_ratio_ave"] = dict()
    description_texts_json["feature_descriptions"]["basic_depth_width_ratio_ave"]["pretty_name"] = "Avg. Depth over Width"
    description_texts_json["feature_descriptions"]["basic_depth_width_ratio_ave"]["description"] = "The average of depth over width ratios for every depth level in the comment tree."

    description_texts_json["feature_descriptions"]["branching_hirsch_index"] = dict()
    description_texts_json["feature_descriptions"]["branching_hirsch_index"]["pretty_name"] = "Comment Tree Hirsch"
    description_texts_json["feature_descriptions"]["branching_hirsch_index"]["description"] = "The depth/width h-index of the comment tree. A high value indicates both a deep and wide tree."

    description_texts_json["feature_descriptions"]["branching_wiener_index"] = dict()
    description_texts_json["feature_descriptions"]["branching_wiener_index"]["pretty_name"] = "Comment Tree Wiener"
    description_texts_json["feature_descriptions"]["branching_wiener_index"]["description"] = "The average of all node distances in the comment tree. A high value indicates multiple long threads."

    description_texts_json["feature_descriptions"]["branching_randic_index"] = dict()
    description_texts_json["feature_descriptions"]["branching_randic_index"]["pretty_name"] = "Comment Tree Randic"
    description_texts_json["feature_descriptions"]["branching_randic_index"]["description"] = "The Randic index of the comment tree. A high value indicates great branching complexity in the tree."

    # User graph descriptions.

    description_texts_json["feature_descriptions"]["user_graph_user_count"] = dict()
    description_texts_json["feature_descriptions"]["user_graph_user_count"]["pretty_name"] = "User Count"
    description_texts_json["feature_descriptions"]["user_graph_user_count"]["description"] = "The number of eponymous users in the discussion at the time of the snapshot."

    description_texts_json["feature_descriptions"]["user_graph_hirsch_index"] = dict()
    description_texts_json["feature_descriptions"]["user_graph_hirsch_index"]["pretty_name"] = "User Graph Hirsch"
    description_texts_json["feature_descriptions"]["user_graph_hirsch_index"]["description"] = "The node/degree h-index of the user graph. A high value indicates a large number of active users."

    description_texts_json["feature_descriptions"]["user_graph_randic_index"] = dict()
    description_texts_json["feature_descriptions"]["user_graph_randic_index"]["pretty_name"] = "User Graph Randic"
    description_texts_json["feature_descriptions"]["user_graph_randic_index"]["description"] = "The Randic index of the user graph. A high value indicates great branching complexity in the graph."

    description_texts_json["feature_descriptions"]["user_graph_outdegree_entropy"] = dict()
    description_texts_json["feature_descriptions"]["user_graph_outdegree_entropy"]["pretty_name"] = "Outdegree Entropy"
    description_texts_json["feature_descriptions"]["user_graph_outdegree_entropy"]["description"] = "The user outdegree distribution entropy in the user graph. A high value indicates a large number of actively replying users."

    description_texts_json["feature_descriptions"]["user_graph_outdegree_normalized_entropy"] = dict()
    description_texts_json["feature_descriptions"]["user_graph_outdegree_normalized_entropy"]["pretty_name"] = "Norm. Outdegree Entropy"
    description_texts_json["feature_descriptions"]["user_graph_outdegree_normalized_entropy"]["description"] = "The normalized user outdegree distribution entropy in the user graph. A high value indicates a large number of actively replying users."

    description_texts_json["feature_descriptions"]["user_graph_indegree_entropy"] = dict()
    description_texts_json["feature_descriptions"]["user_graph_indegree_entropy"]["pretty_name"] = "Indegree Entropy"
    description_texts_json["feature_descriptions"]["user_graph_indegree_entropy"]["description"] = "The user indegree distribution entropy in the user graph. A high value indicates a large number of multiply replied-to users."

    description_texts_json["feature_descriptions"]["user_graph_indegree_normalized_entropy"] = dict()
    description_texts_json["feature_descriptions"]["user_graph_indegree_normalized_entropy"]["pretty_name"] = "Norm. Indegree Entropy"
    description_texts_json["feature_descriptions"]["user_graph_indegree_normalized_entropy"]["description"] = "The normalized user indegree distribution entropy in the user graph. A high value indicates a large number of multiply replied-to users."

    # Temporal descriptions.

    description_texts_json["feature_descriptions"]["temporal_first_half_mean_time"] = dict()
    description_texts_json["feature_descriptions"]["temporal_first_half_mean_time"]["pretty_name"] = "Avg. Time Differences (1st Half)"
    description_texts_json["feature_descriptions"]["temporal_first_half_mean_time"]["description"] = "The average of the first half of between comment timestamp differences."

    description_texts_json["feature_descriptions"]["temporal_last_half_mean_time"] = dict()
    description_texts_json["feature_descriptions"]["temporal_last_half_mean_time"]["pretty_name"] = "Avg. Time Differences (2nd Half)"
    description_texts_json["feature_descriptions"]["temporal_last_half_mean_time"]["description"] = "The average of the second half of between comment timestamp differences."

    description_texts_json["feature_descriptions"]["temporal_std_time"] = dict()
    description_texts_json["feature_descriptions"]["temporal_std_time"]["pretty_name"] = "Time Differences Std."
    description_texts_json["feature_descriptions"]["temporal_std_time"]["description"] = "The standard deviation of the between comment timestamp differences."

    description_texts_json["feature_descriptions"]["temporal_timestamp_range"] = dict()
    description_texts_json["feature_descriptions"]["temporal_timestamp_range"]["pretty_name"] = "Last Comment Lifetime"
    description_texts_json["feature_descriptions"]["temporal_timestamp_range"]["description"] = "The difference between the last comment's and the initial post's timestamps."

    with open(output_file_path, "w") as fp:
        json.dump(description_texts_json, fp, indent=2)


# make_description_texts_json("/home/georgerizos/Documents/LocalStorage/description_texts.json")
#
make_dataset_json(output_file_path="/home/georgerizos/Documents/LocalStorage/reddit_news_visualization.txt",
                  raw_data_file_path="/home/georgerizos/Documents/LocalStorage/raw_data/Rizos/Reddit_News/reddit_news_dataset.txt",
                  features_folder="/home/georgerizos/Documents/LocalStorage/memory/Rizos/Reddit_News/uniform",
                  comparison_lifetimes_path="/home/georgerizos/Documents/LocalStorage/memory/Rizos/Reddit_News/uniform/k_list/akis_focus_reddit.txt",
                  anonymous_coward_name="[deleted]")
