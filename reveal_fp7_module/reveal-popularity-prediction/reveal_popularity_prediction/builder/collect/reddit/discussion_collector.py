__author__ = 'Georgios Rizos (georgerizos@iti.gr)'

import datetime
from praw.helpers import flatten_tree


def fetch_discussion(reddit_handler, url):
    """
    Fetches the full discussion under a submission and stores it as a .json file.
    """
    ####################################################################################################################
    # Fetch submission and extract ALL comments - may be slow for large discussions
    ####################################################################################################################
    # Get UNIX timestamp of the fetch.
    fetch_datetime = datetime.datetime.now()
    fetch_timestamp = fetch_datetime.timestamp()

    # print(submission_id)

    submission = reddit_handler.get_submission(submission_id=url)
    submission.replace_more_comments(limit=None, threshold=0)

    redditor_name = submission.json_dict["author"]

    redditor = reddit_handler.get_redditor(redditor_name, fetch=True)

    comments = submission.comments
    comments = flatten_tree(comments)

    return submission, redditor, comments, fetch_timestamp
