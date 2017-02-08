__author__ = 'Georgios Rizos (georgerizos@iti.gr)'

import subprocess
import json
from urllib.parse import urlparse, parse_qs


def collect(url, youtube_module_communication, youtube_oauth_credentials_folder):
    with open(youtube_module_communication, "w") as fp:
        file_row = "None"
        fp.write(file_row)

    video_id = get_video_id(url)
    if (video_id is None) or (video_id == ""):
        return None

    try:
        subprocess_output = subprocess.call(["get_social_context_json_string",
                                             "-vid", video_id,
                                             "-o", youtube_module_communication,
                                             "-cf", youtube_oauth_credentials_folder])
    except subprocess.CalledProcessError:
        print(url)
        print(video_id)
        return None

    with open(youtube_module_communication, "r") as fp:
        file_row = next(fp)
        if file_row == "None":
            return None
    with open(youtube_module_communication, "r") as fp:
        social_context_dict = json.load(fp)

    social_context_dict["initial_post"] = social_context_dict["initial_post"]["items"][0]
    social_context_dict["author"] = social_context_dict["initial_post"]["channelMetaData"]
    return social_context_dict


def get_video_id(url):
    parsed_url = urlparse(url)
    if parsed_url.netloc == "youtu.be":
        video_id = parsed_url.path[1:]
        if len(video_id) != 11:
            video_id = None
    else:
        parsed_query = parse_qs(parsed_url.query)
        try:
            video_id = parsed_query["v"]
            if type(video_id) == type(list()):  # TODO: Use isinstance()
                video_id = video_id[0]
        except KeyError:
            video_id = None

    return video_id
