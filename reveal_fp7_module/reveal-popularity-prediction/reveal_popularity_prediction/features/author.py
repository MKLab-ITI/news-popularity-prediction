__author__ = 'Georgios Rizos (georgerizos@iti.gr)'

from dateutil import parser as duparser
import calendar
# import datetime


########################################################################################################################
# Reddit author features.
########################################################################################################################
def calculate_author_has_verified_mail_reddit(author_metadata):
    has_verified_mail_boolean = author_metadata["has_verified_email"]

    if has_verified_mail_boolean:
        has_verified_mail = 0
    elif not has_verified_mail_boolean:
        has_verified_mail = 1
    else:
        print(has_verified_mail_boolean)
        raise RuntimeError

    return has_verified_mail


def calculate_author_account_lifetime_reddit(author_metadata, item_metadata):
    item_publication_timestamp = get_reddit_initial_post_timestamp(item_metadata)

    account_created_at = author_metadata["created_utc"]
    account_created_at = float(account_created_at)

    lifetime = item_publication_timestamp - account_created_at

    return lifetime


def calculate_author_hide_from_robots_reddit(author_metadata):
    hide_from_robots_boolean = author_metadata["hide_from_robots"]

    if hide_from_robots_boolean:
        hide_from_robots = 0
    elif not hide_from_robots_boolean:
        hide_from_robots = 1
    else:
        print(hide_from_robots_boolean)
        raise RuntimeError

    return hide_from_robots


def calculate_author_is_mod_reddit(author_metadata):
    is_mod_boolean = author_metadata["is_mod"]

    if is_mod_boolean:
        is_mod = 0
    elif not is_mod_boolean:
        is_mod = 1
    else:
        print(is_mod_boolean)
        raise RuntimeError

    return is_mod


def calculate_author_link_karma_reddit(author_metadata):
    link_karma = author_metadata["link_karma"]

    return link_karma


def calculate_author_link_karma_rate_reddit(author_metadata, item_metadata):
    item_publication_timestamp = get_reddit_initial_post_timestamp(item_metadata)

    link_karma = author_metadata["link_karma"]

    account_created_at = author_metadata["created_utc"]
    account_created_at = float(account_created_at)

    lifetime = item_publication_timestamp - account_created_at

    link_karma_rate = link_karma/lifetime

    return link_karma_rate


def calculate_author_comment_karma_reddit(author_metadata):
    comment_karma = author_metadata["comment_karma"]

    return comment_karma


def calculate_author_comment_karma_rate_reddit(author_metadata, item_metadata):
    item_publication_timestamp = get_reddit_initial_post_timestamp(item_metadata)

    comment_karma = author_metadata["comment_karma"]

    account_created_at = author_metadata["created_utc"]
    account_created_at = float(account_created_at)

    lifetime = item_publication_timestamp - account_created_at

    comment_karma_rate = comment_karma/lifetime

    return comment_karma_rate


def calculate_author_is_gold_reddit(author_metadata):
    is_gold_boolean = author_metadata["is_gold"]

    if is_gold_boolean:
        is_gold = 0
    elif not is_gold_boolean:
        is_gold = 1
    else:
        print(is_gold_boolean)
        raise RuntimeError

    return is_gold


def get_reddit_initial_post_timestamp(item_metadata):
    item_publication_timestamp = item_metadata["created"]

    return item_publication_timestamp


########################################################################################################################
# YouTube author features.
########################################################################################################################

def calculate_author_privacy_status_youtube(channel_metadata):
    privacy_status_string = channel_metadata["status"]["privacyStatus"]

    if privacy_status_string == "private":
        privacy_status = 0
    elif privacy_status_string == "unlisted":
        privacy_status = 1
    elif privacy_status_string == "public":
        privacy_status = 2
    else:
        print(privacy_status_string)
        raise RuntimeError

    return privacy_status


def calculate_author_is_linked_youtube(channel_metadata):
    is_linked_boolean = channel_metadata["status"]["isLinked"]

    if is_linked_boolean:
        is_linked = 0
    elif not is_linked_boolean:
        is_linked = 1
    else:
        print(is_linked_boolean)
        raise RuntimeError

    return is_linked


def calculate_author_long_uploads_status_youtube(channel_metadata):
    long_uploads_status_string = channel_metadata["status"]["longUploadsStatus"]

    if long_uploads_status_string == "disallowed":
        long_uploads_status = 0
    elif long_uploads_status_string == "longUploadsUnspecified":
        long_uploads_status = 1
    elif long_uploads_status_string == "eligible":
        long_uploads_status = 2
    elif long_uploads_status_string == "allowed":
        long_uploads_status = 3
    else:
        print(long_uploads_status_string)
        raise RuntimeError

    return long_uploads_status


def calculate_author_comment_count_youtube(channel_metadata):
    author_comment_count = channel_metadata["statistics"]["commentCount"]
    author_comment_count = int(author_comment_count)

    return author_comment_count


def calculate_author_comment_rate_youtube(channel_metadata, item_metadata):
    author_comment_count = channel_metadata["statistics"]["commentCount"]
    author_comment_count = int(author_comment_count)

    channel_creation_timestamp = channel_metadata["snippet"]["publishedAt"]
    channel_creation_timestamp = datetime_to_timestamp_utc(channel_creation_timestamp)

    item_publication_timestamp = item_metadata["snippet"]["publishedAt"]
    item_publication_timestamp = datetime_to_timestamp_utc(item_publication_timestamp)

    lifetime = item_publication_timestamp - channel_creation_timestamp

    # clean_comment_rate = clean_comment_count/lifetime
    clean_comment_rate = author_comment_count/lifetime

    return clean_comment_rate


def calculate_author_view_count_youtube(channel_metadata, item_metadata):
    author_view_count = channel_metadata["statistics"]["viewCount"]
    author_view_count = int(author_view_count)

    item_view_count = item_metadata["statistics"]["viewCount"]
    item_view_count = int(item_view_count)

    clean_view_count = author_view_count - item_view_count

    return clean_view_count


def calculate_author_view_rate_youtube(channel_metadata, item_metadata):
    author_view_count = channel_metadata["statistics"]["viewCount"]
    author_view_count = int(author_view_count)

    item_view_count = item_metadata["statistics"]["viewCount"]
    item_view_count = int(item_view_count)

    clean_view_count = author_view_count - item_view_count

    channel_creation_timestamp = channel_metadata["snippet"]["publishedAt"]
    channel_creation_timestamp = datetime_to_timestamp_utc(channel_creation_timestamp)

    item_publication_timestamp = item_metadata["snippet"]["publishedAt"]
    item_publication_timestamp = datetime_to_timestamp_utc(item_publication_timestamp)

    lifetime = item_publication_timestamp - channel_creation_timestamp

    clean_view_rate = clean_view_count/lifetime

    return clean_view_rate


def calculate_author_video_upload_count_youtube(channel_metadata):
    video_upload_count = channel_metadata["statistics"]["videoCount"]
    video_upload_count = int(video_upload_count)

    return video_upload_count


def calculate_author_video_upload_rate_youtube(channel_metadata, item_metadata):
    video_upload_count = channel_metadata["statistics"]["videoCount"]
    video_upload_count = int(video_upload_count)

    channel_creation_timestamp = channel_metadata["snippet"]["publishedAt"]
    channel_creation_timestamp = datetime_to_timestamp_utc(channel_creation_timestamp)

    item_publication_timestamp = item_metadata["snippet"]["publishedAt"]
    item_publication_timestamp = datetime_to_timestamp_utc(item_publication_timestamp)

    lifetime = item_publication_timestamp - channel_creation_timestamp

    video_upload_rate = video_upload_count/lifetime

    return video_upload_rate


def calculate_author_subscriber_count_youtube(channel_metadata):
    subscriber_count = channel_metadata["statistics"]["subscriberCount"]
    subscriber_count = int(subscriber_count)

    return subscriber_count


def calculate_author_subscriber_rate_youtube(channel_metadata, item_metadata):
    subscriber_count = channel_metadata["statistics"]["subscriberCount"]
    subscriber_count = int(subscriber_count)

    channel_creation_timestamp = channel_metadata["snippet"]["publishedAt"]
    channel_creation_timestamp = datetime_to_timestamp_utc(channel_creation_timestamp)

    item_publication_timestamp = item_metadata["snippet"]["publishedAt"]
    item_publication_timestamp = datetime_to_timestamp_utc(item_publication_timestamp)

    lifetime = item_publication_timestamp - channel_creation_timestamp

    subscriber_rate = subscriber_count/lifetime

    return subscriber_rate


def calculate_author_hidden_subscriber_count_youtube(channel_metadata):
    hidden_subscriber_count_boolean = channel_metadata["statistics"]["hiddenSubscriberCount"]

    if hidden_subscriber_count_boolean:
        hidden_subscriber_count = 0
    elif not hidden_subscriber_count_boolean:
        hidden_subscriber_count = 1
    else:
        print(hidden_subscriber_count_boolean)
        raise RuntimeError

    return hidden_subscriber_count


def calculate_author_channel_lifetime_youtube(channel_metadata, item_metadata):
    channel_creation_timestamp = channel_metadata["snippet"]["publishedAt"]
    channel_creation_timestamp = datetime_to_timestamp_utc(channel_creation_timestamp)

    item_publication_timestamp = item_metadata["snippet"]["publishedAt"]
    item_publication_timestamp = datetime_to_timestamp_utc(item_publication_timestamp)

    lifetime = item_publication_timestamp - channel_creation_timestamp

    return lifetime


def datetime_to_timestamp_utc(datestring_iso8601):  # TODO: Is this correct?
    parsed_datestring = duparser.parse(datestring_iso8601)
    timestamp = calendar.timegm(parsed_datestring.timetuple())
    # utc_timestamp = datetime.datetime.utcfromtimestamp(timestamp)

    return timestamp
