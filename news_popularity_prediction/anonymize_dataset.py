__author__ = 'Georgios Rizos (georgerizos@iti.gr)'

import os
import json

from news_popularity_prediction.discussion import slashdot
from news_popularity_prediction.discussion.datasetwide import calculate_within_dataset_user_anonymization
from news_popularity_prediction.discussion.builder import within_discussion_comment_and_user_anonymization, safe_comment_generator


def anonymize_static_dataset(dataset_name,
                             input_data_folder):
    document_generator = slashdot.document_generator
    comment_generator = slashdot.comment_generator
    extract_document_post_name = slashdot.extract_document_post_name
    extract_user_name = slashdot.extract_user_name
    extract_comment_name = slashdot.extract_comment_name
    calculate_targets = slashdot.calculate_targets
    extract_timestamp = slashdot.extract_timestamp
    extract_parent_comment_name = slashdot.extract_parent_comment_name
    if dataset_name == "slashdot":
        anonymous_coward_name = "Anonymous Coward"
    elif dataset_name == "barrapunto":
        anonymous_coward_name = "pobrecito hablador"  # "Pendejo Sin Nombre"
    else:
        print("Invalid dataset name.")
        raise RuntimeError

    ####################################################################################################################
    # Dataset-wide user anonymization.
    ####################################################################################################################
    file_name_list = os.listdir(input_data_folder)
    source_file_path_list = [input_data_folder + "/" + file_name for file_name in file_name_list if not file_name[-1] == "~"]
    document_gen = document_generator(source_file_path_list)

    user_name_set,\
    within_dataset_user_anonymize = calculate_within_dataset_user_anonymization(document_gen,
                                                                                comment_generator,
                                                                                extract_user_name)

    file_name_list = os.listdir(input_data_folder)
    source_file_path_list = sorted([input_data_folder + "/" + file_name for file_name in file_name_list if not file_name[-1] == "~"])

    ####################################################################################################################
    # Iterate over files and incrementally calculate features.
    ####################################################################################################################
    for document in document_generator(source_file_path_list):
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
            target_dict = calculate_targets(document,
                                            comment_name_set,
                                            user_name_set,
                                            within_discussion_anonymous_coward)
        except KeyError as e:
            continue

        ################################################################################################################
        # Initiate a smart/safe iteration over all comments.
        ################################################################################################################
        try:
            safe_comment_gen = safe_comment_generator(document=document,
                                                      comment_generator=comment_generator,
                                                      within_discussion_comment_anonymize=within_discussion_comment_anonymize,
                                                      extract_comment_name=extract_comment_name,
                                                      extract_parent_comment_name=extract_parent_comment_name,
                                                      extract_timestamp=extract_timestamp,
                                                      safe=True)
        except TypeError:
            invalid_tree = True
            continue

        ################################################################################################################
        # Make initial post json.
        ################################################################################################################
        initial_post = next(safe_comment_gen)

        uniform_json = dict()
        uniform_json["initial_post"] = dict()
        uniform_json["comments"] = list()

        uniform_json["initial_post"]["user_id"] = within_dataset_user_anonymize[extract_user_name(initial_post)]
        uniform_json["initial_post"]["comment_id"] = within_discussion_comment_anonymize[extract_comment_name(initial_post)]
        try:
            uniform_json["initial_post"]["timestamp"] = extract_timestamp(initial_post)
        except TypeError:
            continue
        uniform_json["initial_post"]["targets"] = target_dict

        ################################################################################################################
        # Make comment json list.
        ################################################################################################################
        invalid_tree = False
        while True:
            try:
                comment = next(safe_comment_gen)
            except TypeError:
                invalid_tree = True
                break
            except StopIteration:
                break
            if comment is None:
                invalid_tree = True
                break
            comment_json = dict()
            comment_json["user_id"] = within_dataset_user_anonymize[extract_user_name(comment)]
            comment_json["comment_id"] = within_discussion_comment_anonymize[extract_comment_name(comment)]
            try:
                comment_json["timestamp"] = extract_timestamp(comment)
            except TypeError:
                invalid_tree = True
                break
            try:
                parent_comment_id = within_discussion_comment_anonymize[extract_parent_comment_name(comment)]
            except KeyError:
                parent_comment_id = uniform_json["initial_post"]["comment_id"]
            comment_json["parent_comment_id"] = parent_comment_id

            uniform_json["comments"].append(comment_json)

        if invalid_tree:
            continue

        json_to_store = dict()
        json_to_store["uniform_json"] = uniform_json
        yield json_to_store


def write_anonymized_dataset(dataset_name,
                             input_data_folder,
                             output_data_folder):
    json_gen = anonymize_static_dataset(dataset_name,
                                        input_data_folder)

    file_counter = 0
    max_documents_in_file = 1000

    fp = open(output_data_folder + dataset_name + "_dataset_anonymized_" + str(file_counter) + ".txt", "w")
    document_counter = 0
    for document_json in json_gen:
        document_counter += 1
        json.dump(document_json, fp)
        fp.write("\n\n")

        if document_counter == max_documents_in_file:
            fp.close()

            document_counter = 0
            file_counter += 1
            fp = open(output_data_folder + dataset_name + "_dataset_anonymized_" + str(file_counter) + ".txt", "w")
    fp.close()
