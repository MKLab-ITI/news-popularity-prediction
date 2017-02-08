__author__ = 'Georgios Rizos (georgerizos@iti.gr)'

import os
import inspect
import multiprocessing

import youtube_discussion_collector


########################################################################################################################
# Configure path related functions.
########################################################################################################################
def get_package_path():
    return os.path.dirname(inspect.getfile(youtube_discussion_collector))


########################################################################################################################
# Configure optimization related functions.
########################################################################################################################
def get_threads_number():
    """
    Automatically determine the number of cores. If that fails, the number defaults to a manual setting.
    """
    try:
        cores_number = multiprocessing.cpu_count()
        return cores_number
    except NotImplementedError:
        cores_number = 8
        return cores_number
