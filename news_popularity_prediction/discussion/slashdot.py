__author__ = 'Georgios Rizos (georgerizos@iti.gr)'

from io import StringIO
import xml.etree.cElementTree as etree


def document_generator(source_file_path_list):
    for file_path in source_file_path_list:
        with open(file_path, "r", encoding="iso-8859-1") as f:
            # Remove .html entities.
            file_line_list = f.readlines()
            file_string = "".join(file_line_list)
            file_string = file_string.replace("&", "")
        document = etree.parse(StringIO(file_string))
        yield document


def comment_generator(document):
    root = document.getroot()

    yield root

    comments = root.findall("./comments/comment")
    for comment in comments:
        yield comment


def extract_document_post_name(document):
    document_post_name = document.find(".id").text

    return document_post_name


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
                      within_discussion_anonymous_coward):
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
