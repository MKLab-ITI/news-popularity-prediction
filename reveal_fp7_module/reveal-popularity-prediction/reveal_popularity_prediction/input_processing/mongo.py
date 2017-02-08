__author__ = 'Georgios Rizos (georgerizos@iti.gr)'

import pymongo
from pymongo import ASCENDING


def establish_mongo_connection(mongo_uri):
    """
    What it says on the tin.

    Inputs: - mongo_uri: A MongoDB URI.

    Output: - A MongoDB client.
    """
    client = pymongo.MongoClient(mongo_uri)
    return client


def get_collection_documents_generator(client, database_name, collection_name, spec, latest_n, sort_key, batch_size=None):
    """
    This is a python generator that yields tweets stored in a mongodb collection.

    Tweet "created_at" field is assumed to have been stored in the format supported by MongoDB.

    Inputs: - client: A pymongo MongoClient object.
            - database_name: The name of a Mongo database as a string.
            - collection_name: The name of the tweet collection as a string.
            - spec: A python dictionary that defines higher query arguments.
            - latest_n: The number of latest results we require from the mongo document collection.
            - sort_key: A field name according to which we will sort in ascending order.

    Yields: - document: A document in python dictionary (json) format.
    """
    mongo_database = client[database_name]
    collection = mongo_database[collection_name]
    collection.create_index(sort_key)

    if latest_n is not None:
        skip_n = collection.count() - latest_n
        if collection.count() - latest_n < 0:
            skip_n = 0
        if batch_size is None:
            cursor = collection.find(filter=spec).sort([(sort_key, ASCENDING), ])
        else:
            cursor = collection.find(filter=spec).sort([(sort_key, ASCENDING), ]).batch_size(batch_size)
        cursor = cursor[skip_n:]
    else:
        if batch_size is None:
            cursor = collection.find(filter=spec).sort([(sort_key, ASCENDING), ])
        else:
            cursor = collection.find(filter=spec).sort([(sort_key, ASCENDING), ]).batch_size(batch_size)

    for document in cursor:
        yield document


def get_safe_mongo_generator(client, database_name, collection_name, spec, latest_n, sort_key, batch_size):
    doc_gen = get_collection_documents_generator(client, database_name, collection_name, spec, latest_n, sort_key, batch_size)

    collection_precessed = False

    while not collection_precessed:
        document_batch = list()

        for i in range(batch_size):
            try:
                document = next(doc_gen)
                document_batch.append(document)
            except StopIteration:
                collection_precessed = True

        for document in document_batch:
            yield document
