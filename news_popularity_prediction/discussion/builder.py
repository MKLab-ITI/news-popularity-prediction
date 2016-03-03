__author__ = 'Georgios Rizos (georgerizos@iti.gr)'

import os
import heapq
import collections

import numpy as np
import scipy.sparse as spsp

# from news_popularity_prediction.discussion.anonymized import document_generator, comment_generator, extract_user_name,\
#     extract_comment_name, calculate_targets, extract_timestamp, extract_parent_comment_name
from news_popularity_prediction.discussion import anonymized, slashdot
from news_popularity_prediction.discussion.features import get_handcrafted_feature_names, initialize_timestamp_array,\
    initialize_handcrafted_features, initialize_intermediate, update_timestamp_array, update_intermediate,\
    update_handcrafted_features
from news_popularity_prediction.discussion.datasetwide import get_within_dataset_user_anonymization
from news_popularity_prediction.datautil.feature_rw import h5_open, h5_close, store_features


def get_h5_max_recommended_number_of_children():
    return 16384


def store_file_counter_generator(thread_id, number_of_threads):
    counter = thread_id
    while True:
        yield counter
        counter += number_of_threads


def extract_features_static_dataset(dataset_name,
                                    input_data_folder,
                                    output_data_folder):
    if dataset_name == "reddit":
        document_generator = anonymized.document_generator
        comment_generator = anonymized.comment_generator
        extract_user_name = anonymized.extract_user_name
        extract_comment_name = anonymized.extract_comment_name
        calculate_targets = anonymized.calculate_targets
        extract_timestamp = anonymized.extract_timestamp
        extract_parent_comment_name = anonymized.extract_parent_comment_name
    elif dataset_name in ["slashdot", "barrapunto"]:
        document_generator = slashdot.document_generator
        comment_generator = slashdot.comment_generator
        extract_user_name = slashdot.extract_user_name
        extract_comment_name = slashdot.extract_comment_name
        calculate_targets = slashdot.calculate_targets
        extract_timestamp = slashdot.extract_timestamp
        extract_parent_comment_name = slashdot.extract_parent_comment_name
    else:
        print("Invalid dataset name.")
        raise RuntimeError

    ####################################################################################################################
    # Dataset-wide user anonymization.
    ####################################################################################################################
    within_dataset_user_anonymizer_filepath = output_data_folder + "/datasetwide/user_anonymizer" + ".pkl"

    file_name_list = os.listdir(input_data_folder)
    source_file_path_list = [input_data_folder + "/" + file_name for file_name in file_name_list if not file_name[-1] == "~"]
    document_gen = document_generator(source_file_path_list)

    within_dataset_user_anonymize = get_within_dataset_user_anonymization(within_dataset_user_anonymizer_filepath,
                                                                          document_gen,
                                                                          comment_generator,
                                                                          extract_user_name)

    file_name_list = os.listdir(input_data_folder)
    source_file_path_list = sorted([input_data_folder + "/" + file_name for file_name in file_name_list if not file_name[-1] == "~"])

    ####################################################################################################################
    # Initialize the H5 store files.
    ####################################################################################################################
    total_counter = 0
    store_file_counter_gen = store_file_counter_generator(0, 1)
    store_file_counter = next(store_file_counter_gen)
    discussion_counter = 0

    timestamp_h5_store_file = h5_open(output_data_folder + "/timestamp_h5_store_file_" + str(store_file_counter) + ".h5")
    handcrafted_features_h5_store_file = h5_open(output_data_folder + "/handcrafted_features_h5_store_file_" + str(store_file_counter) + ".h5")

    ####################################################################################################################
    # Iterate over files and incrementally calculate features.
    ####################################################################################################################
    document_counter = 0
    actual_document_counter = 0
    for document in document_generator(source_file_path_list):
        document_counter += 1
        actual_document_counter += 1
        if actual_document_counter % 500 == 0:
            print("Document no: ", actual_document_counter)

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
                                                                                              extract_user_name=extract_user_name)

        ################################################################################################################
        # Calculate prediction targets.
        ################################################################################################################
        try:
            target_dict = calculate_targets(document)
        except KeyError as e:
            continue

        ################################################################################################################
        # Initiate a smart/safe iteration over all comments.
        ################################################################################################################
        safe_comment_gen = safe_comment_generator(document=document,
                                                  comment_generator=comment_generator,
                                                  within_discussion_comment_anonymize=within_discussion_comment_anonymize,
                                                  extract_comment_name=extract_comment_name,
                                                  extract_parent_comment_name=extract_parent_comment_name,
                                                  extract_timestamp=extract_timestamp,
                                                  safe=True)

        ################################################################################################################
        # Initialize features and intermediate information and structures for incrementally calculating features.
        ################################################################################################################
        # Just get the set.
        handcrafted_feature_names_set = get_handcrafted_feature_names(dataset_name)

        initial_post = next(safe_comment_gen)
        timestamp = extract_timestamp(initial_post)
        op_raw_id = extract_user_name(initial_post)
        op_id = within_discussion_user_anonymize[op_raw_id]
        if op_id == within_discussion_anonymous_coward:
            op_is_anonymous = True
        else:
            op_is_anonymous = False

        comment_counter = 0

        timestamp_column_names_list,\
        timestamp_array = initialize_timestamp_array(target_dict["comments"] + 1,
                                                     cascade_source_timestamp=timestamp)

        handcrafted_feature_names_list,\
        replicate_feature_if_anonymous_set,\
        handcrafted_function_list,\
        handcrafted_feature_array = initialize_handcrafted_features(target_dict["comments"] + 1,
                                                                    handcrafted_feature_names_set=handcrafted_feature_names_set,
                                                                    op_is_anonymous=op_is_anonymous)

        intermediate_dict = initialize_intermediate(comment_name_set,
                                                    user_name_set,
                                                    timestamp,
                                                    within_discussion_anonymous_coward,
                                                    op_is_anonymous=op_is_anonymous)

        discussion_tree = spsp.dok_matrix((len(comment_name_set),
                                           len(comment_name_set)),
                                          dtype=np.int8)

        user_graph = spsp.dok_matrix((len(user_name_set),
                                      len(user_name_set)),
                                     dtype=np.int32)

        invalid_tree = False
        for comment in safe_comment_gen:
            if comment is None:
                invalid_tree = True
                break

            comment_counter += 1

            ############################################################################################################
            # Update discussion radial tree and user graph.
            ############################################################################################################
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
                                                                          discussion_tree=discussion_tree,
                                                                          user_graph=user_graph,
                                                                          within_discussion_comment_anonymize=within_discussion_comment_anonymize,
                                                                          within_discussion_user_anonymize=within_discussion_user_anonymize,
                                                                          within_discussion_anonymous_coward=within_discussion_anonymous_coward,
                                                                          comment_id_to_user_id=intermediate_dict["comment_id_to_user_id"])
                intermediate_dict["comment_id_to_user_id"] = comment_id_to_user_id
            except RuntimeError:
                invalid_tree = True
                break

            ############################################################################################################
            # Update intermediate information and structures for incrementally calculating features.
            ############################################################################################################
            timestamp = extract_timestamp(comment)
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
                invalid_tree = True
                break

            ############################################################################################################
            # Incrementally calculate discussion features.
            ############################################################################################################
            update_handcrafted_features(handcrafted_feature_names_list,
                                        replicate_feature_if_anonymous_set,
                                        handcrafted_function_list,
                                        handcrafted_feature_array,
                                        comment_counter,
                                        intermediate_dict,
                                        commenter_is_anonymous)

        if invalid_tree:
            continue
        else:
            total_counter += 1
            if discussion_counter == get_h5_max_recommended_number_of_children():
                h5_close(timestamp_h5_store_file)
                h5_close(handcrafted_features_h5_store_file)

                store_file_counter = next(store_file_counter_gen)
                discussion_counter = 0

                timestamp_h5_store_file = h5_open(output_data_folder + "/timestamp_h5_store_file_" + str(store_file_counter) + ".h5")
                handcrafted_features_h5_store_file = h5_open(output_data_folder + "/handcrafted_features_h5_store_file_" + str(store_file_counter) + ".h5")

                # timestamp_h5_store_file_keys = set(timestamp_h5_store_file.keys())
                # handcrafted_features_h5_store_file_keys = set(handcrafted_features_h5_store_file.keys())

            store_features(timestamp_h5_store_file,
                           handcrafted_features_h5_store_file,
                           document,
                           target_dict,
                           comment_counter,
                           timestamp_array,
                           timestamp_column_names_list,
                           handcrafted_feature_array,
                           handcrafted_feature_names_list)
            discussion_counter += 1
    print(total_counter)
    h5_close(timestamp_h5_store_file)
    h5_close(handcrafted_features_h5_store_file)

    return 0


def safe_comment_generator(document,
                           comment_generator,
                           within_discussion_comment_anonymize,
                           extract_comment_name,
                           extract_parent_comment_name,
                           extract_timestamp,
                           safe=True):
    """
    We do this in order to correct for nonsensical or missing timestamps.
    """
    if not safe:
        comment_gen = comment_generator(document)

        initial_post = next(comment_gen)
        yield initial_post

        comment_list = sorted(comment_gen, key=extract_timestamp)
        for comment in comment_list:
            yield comment
    else:
        comment_id_to_comment = dict()

        comment_gen = comment_generator(document)

        initial_post = next(comment_gen)
        yield initial_post

        initial_post_id = within_discussion_comment_anonymize[extract_comment_name(initial_post)]

        comment_id_to_comment[initial_post_id] = initial_post

        if initial_post_id != 0:
            print("This cannot be.")
            raise RuntimeError

        comment_list = list(comment_gen)
        children_dict = collections.defaultdict(list)
        for comment in comment_list:
            # Anonymize comment.
            comment_name = extract_comment_name(comment)
            comment_id = within_discussion_comment_anonymize[comment_name]

            parent_comment_name = extract_parent_comment_name(comment)
            if parent_comment_name is None:
                parent_comment_id = 0
            else:
                try:
                    parent_comment_id = within_discussion_comment_anonymize[parent_comment_name]  # TODO: There is an error here for Reddit.
                except KeyError:
                    print("Parent comment does not exist. Comment name: ", comment_name)
                    yield None
                    break

            comment_id_to_comment[comment_id] = comment

            # Update discussion tree.
            children_dict[parent_comment_id].append(comment_id)

        # Starting from the root/initial post, we get the children and we put them in a priority queue.
        priority_queue = list()

        children = set(children_dict[initial_post_id])
        for child in children:
            comment = comment_id_to_comment[child]
            timestamp = extract_timestamp(comment)
            try:
                heapq.heappush(priority_queue, (timestamp, (child, comment)))
            except TypeError:
                print(timestamp)

        # We iteratively yield the topmost priority comment and add to the priority list the children of that comment.
        while True:
            # If priority list empty, we stop.
            if len(priority_queue) == 0:
                break

            t = heapq.heappop(priority_queue)
            yield t[1][1]

            children = set(children_dict[int(t[1][0])])
            for child in children:
                comment = comment_id_to_comment[child]
                timestamp = extract_timestamp(comment)
                heapq.heappush(priority_queue, (timestamp, (child, comment)))


def within_discussion_comment_and_user_anonymization(comment_gen,
                                                     extract_comment_name,
                                                     extract_user_name):
    """
    Reads all distinct users and comments in a single document and anonymizes them. Roots are 0.
    """
    comment_name_set = list()
    user_name_set = list()

    append_comment_name = comment_name_set.append
    append_user_name = user_name_set.append

    ####################################################################################################################
    # Extract comment and user name from the initial post.
    ####################################################################################################################
    initial_post = next(comment_gen)

    initial_post_name = extract_comment_name(initial_post)
    op_name = extract_user_name(initial_post)

    append_comment_name(initial_post_name)
    append_user_name(op_name)

    ####################################################################################################################
    # Iterate over all comments.
    ####################################################################################################################
    for comment in comment_gen:
        comment_name = extract_comment_name(comment)
        commenter_name = extract_user_name(comment)

        append_comment_name(comment_name)
        append_user_name(commenter_name)

    ####################################################################################################################
    # Perform anonymization.
    ####################################################################################################################
    # Remove duplicates and then remove initial post name because we want to give it id 0.
    comment_name_set = set(comment_name_set)
    comment_name_set.remove(initial_post_name)

    # Remove duplicates and then remove OP because we want to give them id 0.
    user_name_set = set(user_name_set)
    user_name_set.remove(op_name)

    # Anonymize.
    within_discussion_comment_anonymize = dict(zip(comment_name_set, range(1, len(comment_name_set) + 1)))
    within_discussion_comment_anonymize[initial_post_name] = 0  # Initial Post gets id 0.

    within_discussion_user_anonymize = dict(zip(user_name_set, range(1, len(user_name_set) + 1)))
    within_discussion_user_anonymize[op_name] = 0            # Original Poster gets id 0.

    comment_name_set.add(initial_post_name)
    user_name_set.add(op_name)

    try:
        within_discussion_anonymous_coward = within_discussion_user_anonymize[None]
    except KeyError as e:
        within_discussion_anonymous_coward = None

    return comment_name_set,\
           user_name_set,\
           within_discussion_comment_anonymize,\
           within_discussion_user_anonymize,\
           within_discussion_anonymous_coward


def update_discussion_and_user_graphs(comment,
                                      extract_comment_name,
                                      extract_parent_comment_name,
                                      extract_user_name,
                                      discussion_tree,
                                      user_graph,
                                      within_discussion_comment_anonymize,
                                      within_discussion_user_anonymize,
                                      within_discussion_anonymous_coward,
                                      comment_id_to_user_id):
    """
    Update the discussion tree and the user graph for a discussion.

    Does not handle the initial post.
    """
    # Extract comment.
    comment_name = extract_comment_name(comment)
    comment_id = within_discussion_comment_anonymize[comment_name]

    # Extract commenter.
    commenter_name = extract_user_name(comment)
    commenter_id = within_discussion_user_anonymize[commenter_name]

    # Update the comment to user map.
    comment_id_to_user_id[comment_id] = commenter_id

    # Check if this is a comment to the original post or to another comment.
    parent_comment_name = extract_parent_comment_name(comment)
    user_graph_modified = False
    parent_commenter_is_anonymous = False
    if parent_comment_name is None:
        # The parent is the original post.
        parent_comment_id = 0
        parent_commenter_id = 0
    else:
        # The parent is another comment.
        try:
            parent_comment_id = within_discussion_comment_anonymize[parent_comment_name]
        except KeyError:
            print("Parent comment does not exist. Comment name: ", comment_name)
            raise RuntimeError

        # Extract parent comment in order to update user graph.
        try:
            parent_commenter_id = comment_id_to_user_id[parent_comment_id]
        except KeyError:
            print("Parent user does not exist. Comment name: ", comment_name)
            raise RuntimeError

    try:
        if within_discussion_anonymous_coward is None:
            if user_graph[commenter_id, parent_commenter_id] == 1:
                user_graph_modified = False
            elif user_graph[parent_commenter_id, commenter_id] == 1:
                user_graph_modified = False
            else:
                user_graph_modified = True
                user_graph[commenter_id, parent_commenter_id] = 1
        else:
            if within_discussion_anonymous_coward not in (parent_commenter_id,
                                                          commenter_id):
                if user_graph[commenter_id, parent_commenter_id] == 1:
                    user_graph_modified = False
                elif user_graph[parent_commenter_id, commenter_id] == 1:
                    user_graph_modified = False
                else:
                    user_graph_modified = True
                    user_graph[commenter_id, parent_commenter_id] = 1
            if within_discussion_anonymous_coward == parent_commenter_id:
                parent_commenter_is_anonymous = True
    except IndexError:
        print("Index error: ", user_graph.shape, commenter_id, parent_commenter_id)
        raise RuntimeError

    # Update discussion radial tree.
    discussion_tree[comment_id, parent_comment_id] = 1

    return discussion_tree,\
           user_graph,\
           comment_id,\
           parent_comment_id,\
           commenter_id,\
           parent_commenter_id,\
           user_graph_modified,\
           parent_commenter_is_anonymous,\
           comment_id_to_user_id
