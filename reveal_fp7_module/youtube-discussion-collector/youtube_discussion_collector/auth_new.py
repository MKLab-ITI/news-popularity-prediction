__author__ = 'Georgios Rizos (georgerizos@iti.gr)'

import os

from googleapiclient.discovery import build_from_document
from oauth2client.client import flow_from_clientsecrets
from oauth2client.file import Storage
from oauth2client.tools import run_flow


# Authorize the request and store authorization credentials.
def get_authenticated_service(args, credentials_folder):

    client_secrets_file = credentials_folder + "client_secrets.json"

    youtube_read_write_ssl_scope = "https://www.googleapis.com/auth/youtube.force-ssl"
    youtube_api_service_name = "youtube"
    youtube_api_version = "v3"

    missing_clients_secrets_message = """
    WARNING: Please configure OAuth 2.0

    To make this sample run you will need to populate the client_secrets.json file
    found at:
       %s
    with information from the APIs Console
    https://developers.google.com/console

    For more information about the client_secrets.json file format, please visit:
    https://developers.google.com/api-client-library/python/guide/aaa_client_secrets
    """ % os.path.abspath(os.path.join(os.path.dirname(__file__),
                                       client_secrets_file))

    flow = flow_from_clientsecrets(client_secrets_file,
                                   scope=youtube_read_write_ssl_scope,
                                   message=missing_clients_secrets_message)

    storage = Storage(credentials_folder + "reveal-stored-oauth2.json")
    credentials = storage.get()

    if credentials is None or credentials.invalid:
        credentials = run_flow(flow, storage, args)

    # Trusted testers can download this discovery document from the developers page
    #  and it should be in the same directory with the code.
    with open(credentials_folder + "youtube-v3-discoverydocument.json", "r") as f:
        doc = f.read()
        return build_from_document(doc, credentials=credentials)

