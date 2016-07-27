__author__ = 'Georgios Rizos (georgerizos@iti.gr)'

import json


def document_generator(source_file_path_list):
    document_counter = 0
    for file_path in source_file_path_list:
        with open(file_path, "r") as batch_file:

            for file_row in batch_file:
                clean_file_row = file_row.strip()
                if clean_file_row == "":
                    continue

                document_json = json.loads(clean_file_row)

                document = dict()

                document["initial_post"] = document_json["uniform_json"]["initial_post"]
                document["post_id"] = str(document_counter)
                # document["fetch_timestamp"] = document_json["fetch_timestamp"]
                document["comments"] = document_json["uniform_json"]["comments"]

                document_counter += 1

                yield document


def comment_generator(document):
    initial_post = document["initial_post"]

    yield initial_post

    comments = document["comments"]

    for comment in comments:
        yield comment


def extract_document_post_name(document):
    document_post_name = document["post_id"]

    return document_post_name


def extract_timestamp(comment):
    timestamp = comment["timestamp"]

    return timestamp


def extract_user_name(comment):
    user_name = comment["user_id"]

    return user_name


def extract_comment_name(comment):
    comment_name = comment["comment_id"]

    return comment_name


def extract_parent_comment_name(comment):
    parent_comment_name = comment["parent_comment_id"]

    return parent_comment_name


def calculate_targets(document):
    targets_dict = document["initial_post"]["targets"]

    return targets_dict
