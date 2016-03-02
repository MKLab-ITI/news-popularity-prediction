import os
import sys
import resource

resource.setrlimit(resource.RLIMIT_STACK, (2**29, -1))
sys.setrecursionlimit(10**6)

from news_popularity_prediction.datautil.common import get_package_path
from news_popularity_prediction.datautil.feature_rw import make_folders
from news_popularity_prediction.discussion.builder import extract_features_static_dataset
from news_popularity_prediction.learning.cascade_lifetime import calculate_comparison_lifetimes, make_feature_matrices
from news_popularity_prediction.entry_points.snow_2016_workshop import experiment_configurations


########################################################################################################################
# Configure feature extraction.
########################################################################################################################
# OUTPUT_DATA_FOLDER = "/path/to/output/data/folder"
OUTPUT_DATA_FOLDER = "/home/georgerizos/Documents/test"
########################################################################################################################

########################################################################################################################
# Check if the raw data exists.
########################################################################################################################
print("Checking whether anonymized discussions exist.")

# Check for RedditNews.
try:
    reddit_news_file_names = os.listdir(get_package_path() + "/news_post_data/reddit_news/anonymized_discussions")
except FileNotFoundError as e:
    reddit_news_file_names = list()

if len(reddit_news_file_names) != 4:
    print("RedditNews anonymized discussions not found.")
    reddit_flag = False
else:
    reddit_flag = True
    make_folders(OUTPUT_DATA_FOLDER + "/reddit",
                 dataset_name="reddit")

# Check for SlashDot.
try:
    slashdot_file_names = os.listdir(get_package_path() + "/news_post_data/slashdot/anonymized_discussions")
except FileNotFoundError as e:
    slashdot_file_names = list()

if len(slashdot_file_names) != 4:
    print("SlashDot anonymized discussions not found.")
    slashdot_flag = False
else:
    slashdot_flag = True
    make_folders(OUTPUT_DATA_FOLDER + "/slashdot",
                 dataset_name="slashdot")

# Check for BarraPunto.
try:
    barrapunto_file_names = os.listdir(get_package_path() + "/news_post_data/barrapunto/anonymized_discussions")
except FileNotFoundError as e:
    barrapunto_file_names = list()

if len(barrapunto_file_names) != 4:
    print("BarraPunto anonymized discussions not found.")
    barrapunto_flag = False
else:
    barrapunto_flag = True
    make_folders(OUTPUT_DATA_FOLDER + "/barrapunto",
                 dataset_name="barrapunto")
"""
########################################################################################################################
# Extract features for online discussions.
########################################################################################################################
print("Extracting features for online discussions ...")

# Extract features for RedditNews discussions.
if reddit_flag:
    dataset_name = "reddit"
    extract_features_static_dataset(dataset_name=dataset_name,
                                    input_data_folder=get_package_path() + "/news_post_data/" + dataset_name + "/anonymized_discussions",
                                    output_data_folder=OUTPUT_DATA_FOLDER + "/" + dataset_name + "/features")

# Extract features for SlashDot discussions.
if slashdot_flag:
    dataset_name = "slashdot"
    extract_features_static_dataset(dataset_name=dataset_name,
                                    input_data_folder=get_package_path() + "/news_post_data/" + dataset_name + "/anonymized_discussions",
                                    output_data_folder=OUTPUT_DATA_FOLDER + "/" + dataset_name + "/features")

# Extract features for BarraPunto discussions.
if barrapunto_flag:
    dataset_name = "barrapunto"
    extract_features_static_dataset(dataset_name=dataset_name,
                                    input_data_folder=get_package_path() + "/news_post_data/" + dataset_name + "/anonymized_discussions",
                                    output_data_folder=OUTPUT_DATA_FOLDER + "/" + dataset_name + "/features")

print("Features extracted.")
########################################################################################################################
# Calculate the comparison lifetimes and form/store the corresponding feature matrices.
########################################################################################################################
# Calculate the comparison lifetimes.

print("Calculating the comparison lifetimes ...")
if reddit_flag:
    dataset_name = "reddit"
    calculate_comparison_lifetimes(features_folder=OUTPUT_DATA_FOLDER + "/" + dataset_name + "/features",
                                   osn_focus=None)

if slashdot_flag:
    dataset_name = "slashdot"
    calculate_comparison_lifetimes(features_folder=OUTPUT_DATA_FOLDER + "/" + dataset_name + "/features",
                                   osn_focus=None)

if barrapunto_flag:
    dataset_name = "barrapunto"
    calculate_comparison_lifetimes(features_folder=OUTPUT_DATA_FOLDER + "/" + dataset_name + "/features",
                                   osn_focus=None)

print("Lifetimes calculated.")

# Form and store the feature matrices for the lifetimes.
print("Forming and storing the feature matrices for the lifetimes ...")

if reddit_flag:
    dataset_name = "reddit"
    make_feature_matrices(features_folder=OUTPUT_DATA_FOLDER + "/" + dataset_name + "/features",
                          osn_focus="reddit")

if slashdot_flag:
    dataset_name = "slashdot"
    make_feature_matrices(features_folder=OUTPUT_DATA_FOLDER + "/" + dataset_name + "/features",
                          osn_focus="slashdot")

if barrapunto_flag:
    dataset_name = "barrapunto"
    make_feature_matrices(features_folder=OUTPUT_DATA_FOLDER + "/" + dataset_name + "/features",
                          osn_focus="barrapunto")

print("Feature matrices stored.")
"""
########################################################################################################################
# Perform experiments.
########################################################################################################################
if reddit_flag:
    experiment_configurations.reddit_news_experiments(OUTPUT_DATA_FOLDER)

if slashdot_flag:
    experiment_configurations.slashdot_experiments(OUTPUT_DATA_FOLDER)

if barrapunto_flag:
    experiment_configurations.barrapunto_experiments(OUTPUT_DATA_FOLDER)
