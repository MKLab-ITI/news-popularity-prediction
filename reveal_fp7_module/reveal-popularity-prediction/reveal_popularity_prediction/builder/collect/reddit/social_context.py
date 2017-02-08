__author__ = 'Georgios Rizos (georgerizos@iti.gr)'

import json
import resource
import sys
resource.setrlimit(resource.RLIMIT_STACK, (2**29, -1))
sys.setrecursionlimit(10**6)

import praw

from reveal_popularity_prediction.builder.collect.reddit.reddit_util import login
from reveal_popularity_prediction.builder.collect.reddit.discussion_collector import fetch_discussion


def collect(url, reddit_oauth_credentials_path):
    # Get an authenticated YouTube service instance.
    reddit_handler = login(reddit_oauth_credentials_path)

    # parsed_url = urlparse(url)
    # post_id = parsed_url.path
    # post_id = post_id.strip().split("/")
    # post_id = post_id[-2]

    try:
        result_tuple = fetch_discussion(reddit_handler, url)

        if result_tuple is None:
            return None
        submission, redditor, comments, fetch_timestamp = result_tuple

        submission.json_dict["redditor_metadata"] = redditor.json_dict
    except praw.errors.NotFound:
        return None
    except praw.errors.Forbidden:
        return None
    except praw.errors.HTTPException:
        return None

    submission_string = json.dumps(submission.json_dict, default=set_default)
    submission = json.loads(submission_string)

    comment_list = list()
    for comment in comments:
        comment_string = json.dumps(comment.json_dict, default=set_default)
        full_comment = json.loads(comment_string)
        comment_list.append(full_comment)

    social_context_dict = dict()
    social_context_dict["initial_post"] = submission
    social_context_dict["comments"] = comment_list
    social_context_dict["author"] = submission["redditor_metadata"]
    social_context_dict["fetch_timestamp"] = fetch_timestamp

    return social_context_dict


def set_default(obj):
    if isinstance(obj, praw.objects.Comment):
        return obj.json_dict
    raise TypeError
