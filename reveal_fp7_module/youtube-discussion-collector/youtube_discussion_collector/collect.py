__author__ = 'Georgios Rizos (georgerizos@iti.gr)'


def get_video_metadata(youtube_service, video_id):
    video_metadata = youtube_service.videos().list(
        part="snippet,statistics",
        id=video_id
    ).execute()

    if len(video_metadata["items"]) == 0:
        return None
    channel_id = video_metadata["items"][0]["snippet"]["channelId"]

    channel_metadata = get_channel_metadata(youtube_service, channel_id)
    if channel_metadata is None:
        return None

    video_metadata["items"][0]["channelMetaData"] = channel_metadata

    return video_metadata


def get_channel_metadata(youtube_service, channel_id):
    channel_metadata = youtube_service.channels().list(
        # part="auditDetails,brandingSettings,contentDetails,contentOwnerDetails,invideoPromotion,snippet,statistics,status,topicDetails",
        # part="auditDetails,snippet,statistics,status,topicDetails",
        part="snippet,statistics,status,topicDetails",
        id=channel_id
    ).execute()

    if len(channel_metadata["items"]) == 0:
        return None

    return channel_metadata["items"][0]


def get_all_comment_threads(youtube_service, video_id):
    comment_thread_list = list()

    # Gather all top video threads (initiated by a top-level comment).
    comment_threads_page,\
    next_page_token = get_comment_threads(youtube_service=youtube_service,
                                          video_id=video_id,
                                          page_token="")
    comment_thread_list.extend(comment_threads_page["items"])

    # Follow the next page link until all are collected.
    while next_page_token != "":
        # Gather all top video threads (initiated by a top-level comment).
        comment_threads_page,\
        next_page_token = get_comment_threads(youtube_service=youtube_service,
                                              video_id=video_id,
                                              page_token=next_page_token)
        comment_thread_list.extend(comment_threads_page["items"])

    return comment_thread_list


def get_comment_threads(youtube_service, video_id, page_token):
    if page_token == "":
        comment_threads_page = youtube_service.commentThreads().list(
            part="snippet",
            videoId=video_id,
            maxResults=100,
            textFormat="plainText"
        ).execute()
    else:
        comment_threads_page = youtube_service.commentThreads().list(
            part="snippet",
            videoId=video_id,
            pageToken=page_token,
            maxResults=100,
            textFormat="plainText"
        ).execute()

    if "nextPageToken" in comment_threads_page.keys():
        next_page_token = comment_threads_page["nextPageToken"]
    else:
        next_page_token = ""

    return comment_threads_page, next_page_token


def enrich_all_comment_threads(youtube_service, comment_thread_list):
    for comment_thread_id, comment_thread in enumerate(comment_thread_list):
        total_number_of_replies = int(comment_thread["snippet"]["totalReplyCount"])

        # comment_thread_timestamp = duparser.parse(timestamp=comment_thread["snippet"]["topLevelComment"]["snippet"]["publishedAt"])
        # comment_thread_timestamp = (comment_thread_timestamp - datetime.datetime(1970, 1, 1, tzinfo=datetime.timezone.utc)) / datetime.timedelta(seconds=1)

        if total_number_of_replies > 0:

            top_comment_id = comment_thread["snippet"]["topLevelComment"]["id"]

            second_level_replies = get_all_second_level_replies(youtube_service, top_comment_id)

            comment_thread_list[comment_thread_id]["replies"] = dict()
            comment_thread_list[comment_thread_id]["replies"]["comments"] = second_level_replies
    return comment_thread_list


# def enrich_all_comment_threads(youtube_service, comment_thread_list):
#     for comment_thread_id, comment_thread in enumerate(comment_thread_list):
#         total_number_of_replies = int(comment_thread["snippet"]["totalReplyCount"])
#
#         if "replies" in comment_thread.keys():
#             collected_replies = comment_thread["replies"]["comments"]
#             collected_number_of_replies = 0
#             collected_reply_ids = list()
#             for comment in collected_replies:
#                 collected_number_of_replies += 1
#                 collected_reply_ids.append(comment["id"])
#
#             if collected_number_of_replies < total_number_of_replies:
#                 top_comment_id = comment_thread["snippet"]["topLevelComment"]["id"]
#
#                 second_level_replies = get_all_second_level_replies(youtube_service, top_comment_id)
#
#                 collected_reply_ids = set(collected_reply_ids)
#
#                 for second_level_reply in second_level_replies:
#                     if second_level_reply["id"] not in collected_reply_ids:
#                         collected_replies.append(second_level_reply)
#
#                 comment_thread_list[comment_thread_id]["replies"]["comments"] = collected_replies
#
#     return comment_thread_list


def get_all_second_level_replies(youtube_service, top_comment_id):
    second_level_reply_list = list()

    # Gather all top video threads (initiated by a top-level comment).
    second_level_replies_page,\
    next_page_token = get_second_level_replies(youtube_service=youtube_service,
                                               top_comment_id=top_comment_id,
                                               page_token="")
    second_level_reply_list.extend(second_level_replies_page["items"])

    # Follow the next page link until all are collected.
    while next_page_token != "":
        # Gather all top video threads (initiated by a top-level comment).
        second_level_replies_page,\
        next_page_token = get_second_level_replies(youtube_service=youtube_service,
                                                   top_comment_id=top_comment_id,
                                                   page_token=next_page_token)
        second_level_reply_list.extend(second_level_replies_page["items"])

    return second_level_reply_list


def get_second_level_replies(youtube_service, top_comment_id, page_token):
    if page_token == "":
        second_level_replies_page = youtube_service.comments().list(
            part="snippet",
            parentId=top_comment_id,
            maxResults=100,
            textFormat="plainText"
        ).execute()
    else:
        second_level_replies_page = youtube_service.comments().list(
            part="snippet",
            parentId=top_comment_id,
            pageToken=page_token,
            maxResults=100,
            textFormat="plainText"
        ).execute()

    if "nextPageToken" in second_level_replies_page.keys():
        next_page_token = second_level_replies_page["nextPageToken"]
    else:
        next_page_token = ""

    return second_level_replies_page, next_page_token
