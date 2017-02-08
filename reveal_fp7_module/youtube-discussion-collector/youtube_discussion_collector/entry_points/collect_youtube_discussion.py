__author__ = 'Georgios Rizos (georgerizos@iti.gr)'

import datetime
import time

from oauth2client.tools import argparser
from googleapiclient.errors import HttpError

from youtube_discussion_collector.auth_new import get_authenticated_service
from youtube_discussion_collector.collect import get_video_metadata, get_all_comment_threads, enrich_all_comment_threads


def main():
    ####################################################################################################################
    # Parse arguments.
    ####################################################################################################################
    argparser.add_argument("-cid", "--channel-id", dest="channel_id",
                           help="Required; ID for channel for which the comment will be inserted.",
                           type=str, required=True)
    argparser.add_argument("-vid", "--video-id", dest="video_id",
                           help="Required; ID for video for which the comment will be inserted.",
                           type=str, required=True)
    argparser.add_argument("-o", "--output", dest="output",
                           help="Required; file path on which to write results.",
                           type=str, required=True)

    args = argparser.parse_args()

    ####################################################################################################################
    # Collect a YouTube discussion under a video.
    ####################################################################################################################
    collect_youtube_discussion(args)


def collect_youtube_discussion(args):
    video_id = args.video_id
    output = args.output
    upper_timestamp = args.upper_timestamp

    # Get an authenticated YouTube service instance.
    start_time = time.time()
    youtube_service = get_authenticated_service(args)
    elapsed_time = time.time() - start_time
    print elapsed_time

    try:
        # Get UNIX timestamp of the fetch.
        fetch_timestamp = time.time()

        video_metadata = get_video_metadata(youtube_service, video_id)
        if video_metadata is None:
            return
        comment_thread_list = get_all_comment_threads(youtube_service, video_id)
        comment_thread_list = enrich_all_comment_threads(youtube_service, comment_thread_list, upper_timestamp)
    except HttpError:
        print "Error."
        return

    if len(video_metadata["items"]) > 0:
        pass
    else:
        return

    with open(output, "w") as fp:
        fp.write("# " + video_id + "\t" + repr(fetch_timestamp) + "\n\n")

        fp.write(str(video_metadata))
        fp.write("\n\n")

        for comment in comment_thread_list:
            fp.write(str(comment))
            fp.write("\n\n")
