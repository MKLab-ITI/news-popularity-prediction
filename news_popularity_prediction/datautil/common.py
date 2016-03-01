__author__ = 'Georgios Rizos (georgerizos@iti.gr)'

import os
import inspect
import multiprocessing
try:
    import cPickle as pickle
except ImportError:
    import pickle

import news_popularity_prediction


def get_package_path():
    """
    Returns the folder path that the package lies in.

    :return: folder_path: The package folder path.
    """
    return os.path.dirname(inspect.getfile(news_popularity_prediction))


def get_file_row_generator(file_path, separator, encoding=None):
    """
    Generator that reads an separated value file row by row.

    :param file_path: The path of the separated value format file.
    :param separator: The delimiter among values (e.g. ",", "\t", " ")
    :param encoding: The encoding used in the stored text.
    :yield: words: A list of strings corresponding to each of the file's rows.
    """
    with open(file_path, encoding=encoding) as file_object:
        for line in file_object:
            words = line.strip().split(separator)
            yield words


def store_pickle(file_path, data):
    """
    Pickle some data to a given path.

    :param file_path: Target file path.
    :param data: The python object to be serialized via pickle.
    :return: None
    """
    pkl_file = open(file_path, 'wb')
    pickle.dump(data, pkl_file)
    pkl_file.close()


def load_pickle(file_path):
    """
    Unpickle some data from a given path.

    :param file_path: Target file path.
    :return: data: The python object that was serialized and stored in disk.
    """
    pkl_file = open(file_path, 'rb')
    data = pickle.load(pkl_file)
    pkl_file.close()
    return data


def get_threads_number():
    """
    Automatically determine the number of cores. If that fails, the number defaults to a manual setting.

    :return: cores_number:
    """
    try:
        cores_number = multiprocessing.cpu_count()
        return cores_number
    except NotImplementedError:
        cores_number = 2
        return cores_number


def split_list(l, k):
    """
    A generator that splits a list in k sublists of roughly equal size.

    :param l: The list to be split.
    :param k: The number of sublists.
    :yield: sublist: One of the k sublists.
    """
    n = len(l)

    d = n // k
    r = n % k

    offset = 0
    for i in range(k):
        if i < r:
            size = d + 1
        else:
            size = d

        yield l[offset:offset+size]
        offset += size
