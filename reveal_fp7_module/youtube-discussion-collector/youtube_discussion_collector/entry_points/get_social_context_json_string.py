__author__ = 'Georgios Rizos (georgerizos@iti.gr)'

import json
import time
import resource
import sys
resource.setrlimit(resource.RLIMIT_STACK, (2**29, -1))
sys.setrecursionlimit(10**6)

from oauth2client.tools import argparser
from googleapiclient.errors import HttpError

from youtube_discussion_collector.auth_new import get_authenticated_service
from youtube_discussion_collector.collect import get_video_metadata, get_all_comment_threads, enrich_all_comment_threads


def main():
    ####################################################################################################################
    # Parse arguments.
    ####################################################################################################################
    # argparser.add_argument("-cid", "--channel-id", dest="channel_id",
    #                        help="Required; ID for channel for which the comment will be inserted.",
    #                        type=str, required=True)
    argparser.add_argument("-vid", "--video-id", dest="video_id",
                           help="Required; ID for video for which the comment will be inserted.",
                           type=str, required=True)
    argparser.add_argument("-o", "--output", dest="output",
                           help="Required; file path on which to write results.",
                           type=str, required=True)
    argparser.add_argument("-cf", "--credentials-folder", dest="credentials_folder",
                           help="Required; folder of YouTube credentials.",
                           type=str, required=True)
    # argparser.add_argument("-ut", "--upper-timestamp", dest="upper_timestamp",
    #                        help="Required; upper timestamp for which to collect comments.",
    #                        type=str, required=True)

    args = argparser.parse_args()

    ####################################################################################################################
    # Collect a YouTube discussion under a video.
    ####################################################################################################################
    collect_youtube_discussion(args)


def collect_youtube_discussion(args):
    video_id = args.video_id
    output = args.output
    credentials_folder = args.credentials_folder
    # upper_timestamp = args.upper_timestamp
    #
    # upper_timestamp = float(upper_timestamp)

    # Get an authenticated YouTube service instance.
    youtube_service = get_authenticated_service(args, credentials_folder)

    try:
        # Get UNIX timestamp of the fetch.
        fetch_timestamp = time.time()

        video_metadata = get_video_metadata(youtube_service, video_id)
        if video_metadata is None:
            return -1
        comment_thread_list = get_all_comment_threads(youtube_service, video_id)
        comment_thread_list = enrich_all_comment_threads(youtube_service, comment_thread_list)
    except HttpError:
        return -1

    if len(video_metadata["items"]) > 0:
        pass
    else:
        return -1

    social_context_dict = dict()
    social_context_dict["initial_post"] = video_metadata
    social_context_dict["comments"] = comment_thread_list
    if "channelMetaData" in video_metadata.keys():
        author_metadata = video_metadata["channelMetaData"]
        social_context_dict["author_metadata"] = author_metadata
    social_context_dict["fetch_timestamp"] = fetch_timestamp

    with open(output, "w") as fp:
        json.dump(social_context_dict, fp)
    return 0
