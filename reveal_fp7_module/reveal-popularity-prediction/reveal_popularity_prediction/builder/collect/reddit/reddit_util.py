__author__ = 'Georgios Rizos (georgerizos@iti.gr)'

import praw

from reveal_popularity_prediction.common.datarw import get_file_row_generator


########################################################################################################################
# OAuth login utilities.
########################################################################################################################
def login(reddit_oauth_credentials_path):
    ####################################################################################################################
    # Log into my application.
    ####################################################################################################################
    user_agent, client_id, client_secret, redirect_uri = read_oauth_credentials(reddit_oauth_credentials_path)

    reddit = praw.Reddit(user_agent=user_agent)

    reddit.set_oauth_app_info(client_id=client_id,
                              client_secret=client_secret,
                              redirect_uri=redirect_uri)

    # We do this in order to also keep the json files for storage
    reddit.config.store_json_result = True

    return reddit


def read_oauth_credentials(reddit_oauth_credentials_path):
    file_row_gen = get_file_row_generator(reddit_oauth_credentials_path, "=")

    file_row = next(file_row_gen)
    user_agent = file_row[1]

    file_row = next(file_row_gen)
    client_id = file_row[1]

    file_row = next(file_row_gen)
    client_secret = file_row[1]

    file_row = next(file_row_gen)
    redirect_uri = file_row[1]

    return user_agent, client_id, client_secret, redirect_uri
