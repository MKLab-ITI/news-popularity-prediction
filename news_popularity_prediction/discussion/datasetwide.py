__author__ = 'Georgios Rizos (georgerizos@iti.gr)'

import os

from news_popularity_prediction.datautil.common import load_pickle, store_pickle


def get_within_dataset_user_anonymization(output_file,
                                          document_gen,
                                          comment_generator,
                                          extract_user_name):
    """
    Either reads or calculates a dictionary anonymizer from user name to anonymized integer id.

    :param output_file: The file path where the map will be stored.
    :param document_gen: A generator that yields dictionaries containing online discussions.
    :param comment_generator: Returns a generator that yields the comments from an online discussion.
    :param extract_user_name: Extracts the user name from a comment.
    :return: within_dataset_user_anonymize: The map from raw data user names to anonymized integer ids.
    """
    if os.path.exists(output_file):
        within_dataset_user_anonymize = load_dataset_user_anonymizer(output_file)
    else:
        user_name_set,\
        within_dataset_user_anonymize = calculate_within_dataset_user_anonymization(document_gen,
                                                                                    comment_generator,
                                                                                    extract_user_name)

        store_dataset_user_anonymizer(output_file,
                                      within_dataset_user_anonymize)

    return within_dataset_user_anonymize


def calculate_within_dataset_user_anonymization(document_gen,
                                                comment_generator,
                                                extract_user_name):
    """
    Calculates a dictionary anonymizer from user name to anonymized integer id.

    :param document_gen: A generator that yields dictionaries containing online discussions.
    :param comment_generator: Returns a generator that yields the comments from an online discussion.
    :param extract_user_name: Extracts the user name from a comment.
    :return: user_name_set: A set of raw data user names.
             within_dataset_user_anonymize: The map from raw data user names to anonymized integer ids.
    """
    # Initialize user set.
    user_name_set = list()
    append_user_name = user_name_set.append

    # Iterate over all dataset documents.
    document_counter = 0
    for document in document_gen:
        document_counter += 1

        if document_counter % 10000 == 0:
            print(document_counter)
            user_name_set = list(set(user_name_set))
            append_user_name = user_name_set.append

        comment_gen = comment_generator(document)

        # First comment is the initial post.
        initial_post = next(comment_gen)
        op_name = extract_user_name(initial_post)
        append_user_name(op_name)

        # If others exist, they are the actual comments.
        for comment in comment_gen:
            commenter_name = extract_user_name(comment)
            append_user_name(commenter_name)

    # If anonymous users can post, the Anonymous name can be included here.
    user_name_set = set(user_name_set)
    print("Number of distinct users in dataset:", len(user_name_set))

    # Within dataset anonymization.
    within_dataset_user_anonymize = dict(zip(user_name_set, range(len(user_name_set))))

    return user_name_set, within_dataset_user_anonymize


def store_dataset_user_anonymizer(output_file,
                                  within_dataset_user_anonymize):
    """
    Stores the dictionary anonymizer from user name to anonymized integer id.
    :param output_file: The file path where the map will be stored.
    :param within_dataset_user_anonymize: The map from raw data user names to anonymized integer ids.
    :return: None
    """
    store_pickle(output_file, within_dataset_user_anonymize)


def load_dataset_user_anonymizer(output_file):
    """
    Loads the dictionary anonymizer from user name to anonymized integer id.

    :param output_file: The file path where the map is stored.
    :return: within_dataset_user_anonymize: The map from raw data user names to anonymized integer ids.
    """
    within_dataset_user_anonymize = load_pickle(output_file)
    return within_dataset_user_anonymize
