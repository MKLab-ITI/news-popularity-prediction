__author__ = 'Georgios Rizos (georgerizos@iti.gr)'

import numpy as np

from reveal_popularity_prediction.builder.targets import ci_lower_bound


def extract_author_metadata(document):
    author_metadata = document["author_metadata"]

    return author_metadata


def comment_generator(document):
    initial_post = document["initial_post"]
    comments = document["comments"]

    yield initial_post

    for comment in comments:
        yield comment


def extract_comment_name(comment):
    comment_name = comment["name"]

    return comment_name


def extract_parent_comment_name(comment):
    parent_comment_name = comment["parent_id"]

    return parent_comment_name


def extract_user_name(comment):
    user_name = comment["author"]

    return user_name


def extract_title(initial_post):
    title = initial_post["title"]
    return title


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

    target_dict["number_of_upvotes"] = int(document["initial_post"]["ups"])
    if float(document["initial_post"]["upvote_ratio"]) == 0.0:
        target_dict["number_of_downvotes"] = abs(float(document["initial_post"]["ups"]) - float(document["initial_post"]["score"]))
    else:
        target_dict["number_of_downvotes"] = float(document["initial_post"]["ups"])*((1/float(document["initial_post"]["upvote_ratio"])) - 1)
    target_dict["total_number_of_votes"] = target_dict["number_of_upvotes"] + target_dict["number_of_downvotes"]

    target_dict["score"] = target_dict["number_of_upvotes"] - target_dict["number_of_downvotes"]

    if target_dict["total_number_of_votes"] == 0:
        target_dict["score_div"] = 0

        target_dict["score_wilson"] = 0

        target_dict["controversiality_wilson"] = 0
    else:
        target_dict["score_div"] = target_dict["number_of_upvotes"] / target_dict["total_number_of_votes"]

        target_dict["score_wilson"] = ci_lower_bound(target_dict["number_of_upvotes"],
                                                     target_dict["total_number_of_votes"],
                                                     0.95)
        if np.floor(target_dict["total_number_of_votes"]/2) == 0.0:
            target_dict["controversiality_wilson"] = 0
        else:
            target_dict["controversiality_wilson"] = ci_lower_bound(min(np.floor(target_dict["number_of_upvotes"]),
                                                                        np.floor(target_dict["number_of_downvotes"])),
                                                                    np.floor(target_dict["total_number_of_votes"]/2),
                                                                    0.95)

    target_dict["controversiality"] = target_dict["total_number_of_votes"] /\
                                      max(abs(target_dict["number_of_upvotes"] -
                                              target_dict["number_of_downvotes"]), 1)

    return target_dict


def extract_timestamp(comment):
    timestamp = comment["created_utc"]

    return timestamp


def extract_text(comment):
    if "title" in comment.keys():
        text = [comment["title"], comment["description"]]
    else:
        text = [comment["body"], ]
    return text
