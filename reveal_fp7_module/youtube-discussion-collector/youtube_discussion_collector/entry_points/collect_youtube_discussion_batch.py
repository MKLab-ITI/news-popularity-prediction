__author__ = 'Georgios Rizos (georgerizos@iti.gr)'

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
    argparser.add_argument("-i", "--input", dest="input",
                           help="Required; file path of a list of YouTube video IDs.",
                           type=str, required=True)
    argparser.add_argument("-o", "--output", dest="output",
                           help="Required; file path on which to write results.",
                           type=str, required=True)

    args = argparser.parse_args()

    ####################################################################################################################
    # Collect a YouTube discussion under a video.
    ####################################################################################################################
    collect_youtube_discussion_batch(args)


def collect_youtube_discussion_batch(args):
    input = args.input
    output = args.output

    video_id_gen = video_id_generator(input)

    # Get an authenticated YouTube service instance.
    youtube_service = get_authenticated_service(args)

    collect_and_write_discussions_batch(youtube_service, video_id_gen, output)


def video_id_generator(file_path):
    print file_path
    with open(file_path, "r") as fp:
        for row in fp:
            yield row.strip()


def collect_and_write_discussions_batch(youtube_service, video_id_gen, output):
    with open(output, "w") as fp:
        counter = 1
        for video_id in video_id_gen:
            if counter % 50 == 0:
                print(counter)
            counter += 1
            try:
                # Get UNIX timestamp of the fetch.
                fetch_timestamp = time.time()

                video_metadata = get_video_metadata(youtube_service, video_id)
                if video_metadata is None:
                    continue

                comment_thread_list = get_all_comment_threads(youtube_service, video_id)
                comment_thread_list = enrich_all_comment_threads(youtube_service, comment_thread_list)
            except HttpError:
                continue

            # if len(video_metadata["items"]) > 0:
            #     pass
            # else:
            #     continue

            fp.write("# " + video_id + "\t" + repr(fetch_timestamp) + "\n\n")
            fp.write(str(video_metadata["items"][0]))
            fp.write("\n\n")

            for comment in comment_thread_list:
                fp.write(str(comment))
                fp.write("\n\n")
