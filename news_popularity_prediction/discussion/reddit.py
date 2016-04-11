__author__ = 'Georgios Rizos (georgerizos@iti.gr)'

import json

import numpy as np

from news_popularity_prediction.discussion.targets import ci_lower_bound


def document_generator(source_file_path_list):

    file_counter = 0
    document_counter = 0
    for file_path in source_file_path_list:
        file_end = False

        file_counter += 1
        with open(file_path, "r") as batch_file:
            comments_end = False

            try:
                file_row = next(batch_file)
            except StopIteration:
                continue
            file_row = file_row.strip().split("\t")
            # post_id = file_row[0][2:]
            # fetch_timestamp = file_row[1]
            #
            post_id = file_row[1]
            fetch_timestamp = file_row[2]

            while True:
                document_counter += 1
                # print(file_counter, document_counter, file_path)

                document = dict()

                next(batch_file)

                initial_post = next(batch_file)
                initial_post = json.loads(initial_post)
                document["initial_post"] = initial_post
                document["post_id"] = post_id
                document["fetch_timestamp"] = fetch_timestamp

                if "redditor_metadata" in initial_post.keys():
                    author_metadata = initial_post["redditor_metadata"]
                    document["author_metadata"] = author_metadata

                comments = list()
                while True:
                    try:
                        file_row = next(batch_file)
                    except StopIteration:
                        file_end = True
                        break

                    file_row = file_row.strip()
                    if file_row == "":
                        continue
                    elif file_row[0] == "#":
                        file_row = file_row.split("\t")
                        # post_id = file_row[0][2:]
                        # fetch_timestamp = file_row[1]

                        post_id = file_row[1]
                        fetch_timestamp = file_row[2]
                        break
                    else:
                        comment = json.loads(file_row)
                        comments.append(comment)

                document["comments"] = comments

                yield document

                if file_end:
                    break


def extract_author_metadata(document):
    author_metadata = document["author_metadata"]

    return author_metadata


def comment_generator(document):
    initial_post = document["initial_post"]
    comments = document["comments"]

    yield initial_post

    for comment in comments:
        yield comment


def get_post_url(document):
    post_url = document["initial_post"]["url"]

    return post_url


def get_post_title(document):
    post_title = document["initial_post"]["title"]

    return post_title


def extract_comment_name(comment):
    comment_name = comment["name"]

    return comment_name


def extract_parent_comment_name(comment):
    parent_comment_name = comment["parent_id"]

    return parent_comment_name


def extract_user_name(comment):
    user_name = comment["author"]

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

    number_of_upvotes = int(document["initial_post"]["ups"])
    if float(document["initial_post"]["upvote_ratio"]) == 0.0:
        number_of_downvotes = abs(float(document["initial_post"]["ups"]) - float(document["initial_post"]["score"]))
    else:
        number_of_downvotes = float(document["initial_post"]["ups"])*((1/float(document["initial_post"]["upvote_ratio"])) - 1)
    total_number_of_votes = number_of_upvotes + number_of_downvotes

    if total_number_of_votes == 0:
        target_dict["score_wilson"] = 0

        target_dict["controversiality_wilson"] = 0
    else:
        target_dict["score_wilson"] = ci_lower_bound(number_of_upvotes,
                                                     total_number_of_votes,
                                                     0.95)
        if np.floor(total_number_of_votes/2) == 0.0:
            target_dict["controversiality_wilson"] = 0
        else:
            target_dict["controversiality_wilson"] = ci_lower_bound(min(np.floor(number_of_upvotes),
                                                                        np.floor(number_of_downvotes)),
                                                                    np.floor(total_number_of_votes/2),
                                                                    0.95)

    return target_dict


def extract_timestamp(comment):
    timestamp = comment["created"]

    return timestamp
