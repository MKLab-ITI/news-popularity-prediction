__author__ = 'Georgios Rizos (georgerizos@iti.gr)'

from news_popularity_prediction.visualization.snow_2016_workshop.reddit_results import read_reddit_result_data,\
    make_reddit_figures
from news_popularity_prediction.visualization.snow_2016_workshop.slashdot_results import read_slashdot_result_data,\
    read_barrapunto_result_data, make_slashdot_figures, make_barrapunto_figures

########################################################################################################################
# Configure feature extraction.
########################################################################################################################
RESULTS_DATA_FOLDER = "/path/to/output/data/folder"  # Must be the same as the output of the experiments script.

REDDIT_FIGURES_FOLDER = "/path/to/figures/reddit"
SLASHDOT_FIGURES_FOLDER = "/path/to/figures/slashdot"

########################################################################################################################
# Make figures.
########################################################################################################################

METHOD_SELECTION = "_THREE_SOURCES"
method_name_list = ["Temporal",
                    "User Graph",
                    "Comment Tree",
                    "Comment Tree + User Graph",
                    "All"]
reddit_mse,\
reddit_jaccard,\
reddit_k_list = read_reddit_result_data(RESULTS_DATA_FOLDER, method_name_list)

REDDIT_PATH = REDDIT_FIGURES_FOLDER + "/engineered_features_results_reddit" + METHOD_SELECTION

make_reddit_figures(REDDIT_PATH,
                    method_name_list,
                    reddit_mse,
                    reddit_jaccard,
                    reddit_k_list)

METHOD_SELECTION = "_BASELINE"
method_name_list = ["Baseline Comments + Users",
                    "Simple Graph",
                    "Comment Tree + User Graph"]

reddit_mse,\
reddit_jaccard,\
reddit_k_list = read_reddit_result_data(RESULTS_DATA_FOLDER, method_name_list)

REDDIT_PATH = REDDIT_FIGURES_FOLDER + "/engineered_features_results_reddit" + METHOD_SELECTION

make_reddit_figures(REDDIT_PATH,
                    method_name_list,
                    reddit_mse,
                    reddit_jaccard,
                    reddit_k_list)

########################################################################################################################

METHOD_SELECTION = "_THREE_SOURCES"
method_name_list = ["Temporal",
                    "User Graph",
                    "Comment Tree",
                    "Comment Tree + User Graph",
                    "All"]
slashdot_mse,\
slashdot_jaccard,\
slashdot_k_list = read_slashdot_result_data(RESULTS_DATA_FOLDER, method_name_list)
barrapunto_mse,\
barrapunto_jaccard,\
barrapunto_k_list = read_barrapunto_result_data(RESULTS_DATA_FOLDER, method_name_list)

SLASHDOT_PATH = SLASHDOT_FIGURES_FOLDER + "/engineered_features_results_slashdot" + METHOD_SELECTION

make_slashdot_figures(SLASHDOT_PATH,
                      method_name_list,
                      slashdot_mse,
                      slashdot_jaccard,
                      slashdot_k_list)
make_barrapunto_figures(SLASHDOT_PATH,
                        method_name_list,
                        barrapunto_mse,
                        barrapunto_jaccard,
                        barrapunto_k_list)

METHOD_SELECTION = "_BASELINE"
method_name_list = ["Baseline Comments + Users",
                    "Simple Graph",
                    "Comment Tree + User Graph"]
slashdot_mse,\
slashdot_jaccard,\
slashdot_k_list = read_slashdot_result_data(RESULTS_DATA_FOLDER, method_name_list)
barrapunto_mse,\
barrapunto_jaccard,\
barrapunto_k_list = read_barrapunto_result_data(RESULTS_DATA_FOLDER, method_name_list)

SLASHDOT_PATH = SLASHDOT_FIGURES_FOLDER + "/engineered_features_results_slashdot" + METHOD_SELECTION

make_slashdot_figures(SLASHDOT_PATH,
                      method_name_list,
                      slashdot_mse,
                      slashdot_jaccard,
                      slashdot_k_list)
make_barrapunto_figures(SLASHDOT_PATH,
                        method_name_list,
                        barrapunto_mse,
                        barrapunto_jaccard,
                        barrapunto_k_list)
########################################################################################################################
