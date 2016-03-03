__author__ = 'Georgios Rizos (georgerizos@iti.gr)'

import xml.etree.cElementTree as etree


def document_generator(source_file_path_list):
    for file_path in source_file_path_list:
        document = etree.ElementTree(file=file_path)
        yield document


def comment_generator(document):
    root = document.getroot()

    yield root

    comments = root.findall("./comments/comment")
    for comment in comments:
        yield comment


def extract_comment_name(comment):
    comment_name = comment.find("id").text

    return comment_name


def extract_parent_comment_name(comment):
    parent_comment_name = comment.find("parentid")

    if parent_comment_name is None:
        return None
    else:
        parent_comment_name = parent_comment_name.text
        return parent_comment_name


def extract_user_name(comment):
    user_name = comment.find("user").text

    return user_name


def calculate_targets(document,
                      comment_name_set,
                      user_name_set,
                      within_discussion_anonymous_coward,
                      discussion_unobserved_reply_count=0):
    target_dict = dict()

    target_dict["comments"] = len(comment_name_set) - 1
    if within_discussion_anonymous_coward is not None:
        target_dict["users"] = len(user_name_set) - 1  # -1 for Anonymous Coward.
    else:
        target_dict["users"] = len(user_name_set)

    return target_dict


def extract_timestamp(comment):
    timestamp = float(comment.find(".date/seconds").text)

    return timestamp
