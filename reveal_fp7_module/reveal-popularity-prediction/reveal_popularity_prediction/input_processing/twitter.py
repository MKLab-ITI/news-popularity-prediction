__author__ = 'Georgios Rizos (georgerizos@iti.gr)'

import time
import datetime
from datetime import timezone
from urllib.parse import urlparse, parse_qs
from urllib.request import urlopen
from urllib.error import HTTPError, URLError


def extract_urls_from_tweets(tweet_gen, is_replayed_stream):
    current_timestamp = time.perf_counter()

    json_list = list()
    append_json = json_list.append

    counter = 0

    url_set = set()

    youtube_counter = 0
    unshortened_youtube_counter = 0
    reddit_counter = 0
    unshortened_reddit_counter = 0

    youtube_netlocs = get_youtube_netlocs()
    reddit_netlocs = get_reddit_netlocs()
    url_shortener_netlocs = get_url_shortener_netlocs()

    max_tweet_timestamp = 0.0
    for tweet in tweet_gen:
        # Increment tweet counter.
        counter += 1

        # if counter % 10000 == 0:
        #     print()
        #     print(youtube_counter)
        #     print(unshortened_youtube_counter)
        #     print(reddit_counter)
        #     print(unshortened_reddit_counter)

        ################################################################################################################
        # We are dealing with an original tweet.
        ################################################################################################################
        if "retweeted_status" not in tweet.keys():
            tweet_timestamp = twitter_time_to_timestamp(tweet["created_at"])
            if tweet_timestamp > max_tweet_timestamp:
                max_tweet_timestamp = tweet_timestamp
            urls = tweet["entities"]["urls"]
            tweet_id = tweet["id"]
            user_screen_name = tweet["user"]["screen_name"]

            for url in urls:
                # 'http://www.cwi.nl:80/%7Eguido/Python.html'
                # ParseResult(scheme='http', netloc='www.cwi.nl:80', path='/%7Eguido/Python.html',
                # params='', query='', fragment='')
                parsed_url = urlparse(url["expanded_url"])

                if parsed_url.geturl() in url_set:
                    continue

                # if parsed_url.netloc in url_shortener_netlocs:
                #     unshortened_url = url_unshortener(parsed_url.geturl())
                #     parsed_unshortened_url = urlparse(unshortened_url)
                #     if parsed_unshortened_url.netloc in youtube_netlocs:
                #         unshortened_youtube_counter += 1
                #     if parsed_unshortened_url.netloc in reddit_netlocs:
                #         unshortened_reddit_counter += 1

                if parsed_url.netloc in youtube_netlocs:
                    youtube_counter += 1
                    url_set.add(parsed_url.geturl())

                    platform_name = "YouTube"

                    url_dict = dict()
                    url_dict["tweet_timestamp"] = tweet_timestamp
                    url_dict["url"] = parsed_url.geturl()
                    url_dict["tweet_id"] = tweet_id
                    url_dict["user_screen_name"] = user_screen_name
                    url_dict["platform_name"] = platform_name

                    # yield url_dict
                    append_json(url_dict)

                if parsed_url.netloc in reddit_netlocs:
                    reddit_counter += 1
                    url_set.add(parsed_url.geturl())

                    platform_name = "Reddit"

                    url_dict = dict()
                    url_dict["tweet_timestamp"] = tweet_timestamp
                    url_dict["url"] = parsed_url.geturl()
                    url_dict["tweet_id"] = tweet_id
                    url_dict["user_screen_name"] = user_screen_name
                    url_dict["platform_name"] = platform_name

                    # yield url_dict
                    append_json(url_dict)
        ################################################################################################################
        # We are dealing with a retweet.
        ################################################################################################################
        else:
            tweet_timestamp = twitter_time_to_timestamp(tweet["created_at"])
            if tweet_timestamp > max_tweet_timestamp:
                max_tweet_timestamp = tweet_timestamp
            urls = tweet["entities"]["urls"]
            tweet_id = tweet["id"]
            user_screen_name = tweet["user"]["screen_name"]

            original_tweet = tweet["retweeted_status"]
            original_tweet_timestamp = twitter_time_to_timestamp(original_tweet["created_at"])
            original_tweet_urls = original_tweet["entities"]["urls"]
            original_tweet_urls = set(original_tweet_url["expanded_url"] for original_tweet_url in original_tweet_urls)
            original_tweet_id = original_tweet["id"]
            original_tweet_user_screen_name = original_tweet["user"]["screen_name"]

            for url in urls:
                parsed_url = urlparse(url["expanded_url"])
                parse_qs(urlparse(url["expanded_url"]).query, keep_blank_values=True)

                if parsed_url.geturl() in url_set:
                    continue

                # if parsed_url.netloc in url_shortener_netlocs:
                #     unshortened_url = url_unshortener(parsed_url.geturl())
                #     parsed_unshortened_url = urlparse(unshortened_url)
                #     if parsed_unshortened_url.netloc in youtube_netlocs:
                #         unshortened_youtube_counter += 1
                #     if parsed_unshortened_url.netloc in reddit_netlocs:
                #         unshortened_reddit_counter += 1

                if parsed_url.netloc in youtube_netlocs:
                    youtube_counter += 1
                    url_set.add(parsed_url.geturl())

                    platform_name = "YouTube"

                    url_dict = dict()
                    url_dict["url"] = parsed_url.geturl()
                    url_dict["platform_name"] = platform_name

                    if url["expanded_url"] in original_tweet_urls:
                        url_dict["tweet_timestamp"] = original_tweet_timestamp
                        url_dict["tweet_id"] = original_tweet_id
                        url_dict["user_screen_name"] = original_tweet_user_screen_name
                    else:
                        url_dict["tweet_timestamp"] = tweet_timestamp
                        url_dict["tweet_id"] = tweet_id
                        url_dict["user_screen_name"] = user_screen_name

                    # yield url_dict
                    append_json(url_dict)

                if parsed_url.netloc in reddit_netlocs:
                    reddit_counter += 1
                    url_set.add(parsed_url.geturl())

                    platform_name = "Reddit"

                    url_dict = dict()
                    url_dict["url"] = parsed_url.geturl()
                    url_dict["platform_name"] = platform_name

                    if url["expanded_url"] in original_tweet_urls:
                        url_dict["tweet_timestamp"] = original_tweet_timestamp
                        url_dict["tweet_id"] = original_tweet_id
                        url_dict["user_screen_name"] = original_tweet_user_screen_name
                    else:
                        url_dict["tweet_timestamp"] = tweet_timestamp
                        url_dict["tweet_id"] = tweet_id
                        url_dict["user_screen_name"] = user_screen_name

                    # yield url_dict
                    append_json(url_dict)

    if is_replayed_stream:
        assessment_timestamp = max_tweet_timestamp
    else:
        assessment_timestamp = current_timestamp

    return json_list, assessment_timestamp


def twitter_time_to_timestamp(created_at):
    # timestamp = time.mktime(time.strptime(created_at, "%a %b %d %H:%M:%S +0000 %Y"))
    timestamp = datetime.datetime.strptime(created_at, "%a %b %d %H:%M:%S +0000 %Y")

    timestamp = timestamp.replace(tzinfo=timezone.utc).timestamp() + 1.0
    return timestamp


def url_unshortener(shortened_url):
    """
    This should be called asynchronously. A quickly spawned thread is ideal, even if there is no Python concurrency.
    """
    try:
        resp = urlopen(shortened_url)
    except HTTPError:
        return None
    except URLError:
        return None
    return resp.url


def get_url_shortener_netlocs():
    url_shortener_netlocs = set()
    url_shortener_netlocs.update(["bit.do",
                                  # "t.co",
                                  "lnkd.in",
                                  "db.tt",
                                  "qr.ae",
                                  "adf.ly",
                                  "goo.gl",
                                  "bitly.com",
                                  "cur.lv",
                                  "tinyurl.com",
                                  "ow.ly",
                                  "bit.ly",
                                  "adcrun.ch",
                                  "ity.im",
                                  "q.gs",
                                  "viralurl.com",
                                  "is.gd",
                                  "po.st",
                                  "vur.me",
                                  "bc.vc",
                                  "twitthis.com",
                                  "u.to",
                                  "j.mp",
                                  "buzurl.com",
                                  "cutt.us",
                                  "u.bb",
                                  "yourls.org",
                                  "crisco.com",
                                  "x.co",
                                  "prettylinkpro.com",
                                  "viralurl.biz",
                                  "adcraft.co",
                                  "virl.ws",
                                  "scrnch.me",
                                  "filoops.info",
                                  "vurl.bz",
                                  "vzturl.com",
                                  "lemde.fr",
                                  "qr.net",
                                  "1url.com",
                                  "tweez.me",
                                  "7vd.cn",
                                  "v.gd",
                                  "dft.ba",
                                  "aka.gr",
                                  "tr.im",
                                  "tinyarrows.com",
                                  "http://➡.ws/"])
    return url_shortener_netlocs


def get_youtube_netlocs():
    youtube_netlocs = set()
    youtube_netlocs.update(["www.youtube.com",
                            "youtu.be",
                            "m.youtube.com"])
    return youtube_netlocs


def get_reddit_netlocs():
    reddit_netlocs = set()
    reddit_netlocs.update(["www.reddit.com", ])
    return reddit_netlocs
