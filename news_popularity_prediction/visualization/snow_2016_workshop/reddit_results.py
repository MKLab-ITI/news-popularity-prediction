__author__ = 'Georgios Rizos (georgerizos@iti.gr)'

import matplotlib
import matplotlib.pyplot as plt
import seaborn as sns

from news_popularity_prediction.visualization.snow_2016_workshop.common import get_results_file_paths, handle_nan,\
    get_method_name_to_legend_name_dict, add_results

matplotlib.rcParams["ps.useafm"] = True
matplotlib.rcParams["pdf.use14corefonts"] = True
matplotlib.rcParams["text.usetex"] = True


def read_reddit_result_data(output_folder, method_name_list):
    reddit_mse = dict()
    reddit_jaccard = dict()
    feature_names = dict()
    reddit_k_list = None

    CONFIGURATION_DICT = dict()
    CONFIGURATION_DICT["target_name_list"] = ["comments", "users", "score_wilson", "controversiality_wilson"]
    CONFIGURATION_DICT["osn_name_focus"] = "reddit"
    CONFIGURATION_DICT["target_osn_name"] = "reddit"
    CONFIGURATION_DICT["feature_osn_name_list"] = ["reddit"]
    CONFIGURATION_DICT["add_branching_features"] = False
    CONFIGURATION_DICT["add_usergraph_features"] = False
    CONFIGURATION_DICT["add_temporal_features"] = False
    CONFIGURATION_DICT["add_author_features"] = False

    CONFIGURATION_DICT["baseline"] = "mean"

    METHOD = "Baseline Mean"

    if METHOD in method_name_list:
        reddit_k_list = add_results(mse_results=reddit_mse,
                                    jaccard_results=reddit_jaccard,
                                    feature_names=feature_names,
                                    feature_name_offset=7,
                                    method=METHOD,
                                    target_name_list=CONFIGURATION_DICT["target_name_list"],
                                    results_file_paths=get_results_file_paths(output_folder, CONFIGURATION_DICT))

    CONFIGURATION_DICT = dict()
    CONFIGURATION_DICT["target_name_list"] = ["comments", "users", "score_wilson", "controversiality_wilson"]
    CONFIGURATION_DICT["osn_name_focus"] = "reddit"
    CONFIGURATION_DICT["target_osn_name"] = "reddit"
    CONFIGURATION_DICT["feature_osn_name_list"] = ["reddit"]
    CONFIGURATION_DICT["add_branching_features"] = False
    CONFIGURATION_DICT["add_usergraph_features"] = False
    CONFIGURATION_DICT["add_temporal_features"] = False
    CONFIGURATION_DICT["add_author_features"] = False

    CONFIGURATION_DICT["baseline"] = "median"

    METHOD = "Baseline Median"

    if METHOD in method_name_list:
        reddit_k_list = add_results(mse_results=reddit_mse,
                                    jaccard_results=reddit_jaccard,
                                    feature_names=feature_names,
                                    feature_name_offset=7,
                                    method=METHOD,
                                    target_name_list=CONFIGURATION_DICT["target_name_list"],
                                    results_file_paths=get_results_file_paths(output_folder, CONFIGURATION_DICT))

    CONFIGURATION_DICT = dict()
    CONFIGURATION_DICT["target_name_list"] = ["comments", "users", "score_wilson", "controversiality_wilson"]
    CONFIGURATION_DICT["osn_name_focus"] = "reddit"
    CONFIGURATION_DICT["target_osn_name"] = "reddit"
    CONFIGURATION_DICT["feature_osn_name_list"] = ["reddit"]
    CONFIGURATION_DICT["add_branching_features"] = False
    CONFIGURATION_DICT["add_usergraph_features"] = False
    CONFIGURATION_DICT["add_temporal_features"] = False
    CONFIGURATION_DICT["add_author_features"] = False

    CONFIGURATION_DICT["baseline"] = "comments"

    METHOD = "Baseline Comments"

    if METHOD in method_name_list:
        reddit_k_list = add_results(mse_results=reddit_mse,
                                    jaccard_results=reddit_jaccard,
                                    feature_names=feature_names,
                                    feature_name_offset=7,
                                    method=METHOD,
                                    target_name_list=CONFIGURATION_DICT["target_name_list"],
                                    results_file_paths=get_results_file_paths(output_folder, CONFIGURATION_DICT))

    CONFIGURATION_DICT = dict()
    CONFIGURATION_DICT["target_name_list"] = ["comments", "users", "score_wilson", "controversiality_wilson"]
    CONFIGURATION_DICT["osn_name_focus"] = "reddit"
    CONFIGURATION_DICT["target_osn_name"] = "reddit"
    CONFIGURATION_DICT["feature_osn_name_list"] = ["reddit"]
    CONFIGURATION_DICT["add_branching_features"] = False
    CONFIGURATION_DICT["add_usergraph_features"] = False
    CONFIGURATION_DICT["add_temporal_features"] = False
    CONFIGURATION_DICT["add_author_features"] = False

    CONFIGURATION_DICT["baseline"] = "users"

    METHOD = "Baseline Users"

    if METHOD in method_name_list:
        reddit_k_list = add_results(mse_results=reddit_mse,
                                    jaccard_results=reddit_jaccard,
                                    feature_names=feature_names,
                                    feature_name_offset=7,
                                    method=METHOD,
                                    target_name_list=CONFIGURATION_DICT["target_name_list"],
                                    results_file_paths=get_results_file_paths(output_folder, CONFIGURATION_DICT))

    CONFIGURATION_DICT = dict()
    CONFIGURATION_DICT["target_name_list"] = ["comments", "users", "score_wilson", "controversiality_wilson"]
    CONFIGURATION_DICT["osn_name_focus"] = "reddit"
    CONFIGURATION_DICT["target_osn_name"] = "reddit"
    CONFIGURATION_DICT["feature_osn_name_list"] = ["reddit"]
    CONFIGURATION_DICT["add_branching_features"] = False
    CONFIGURATION_DICT["add_usergraph_features"] = False
    CONFIGURATION_DICT["add_temporal_features"] = False
    CONFIGURATION_DICT["add_author_features"] = False

    CONFIGURATION_DICT["baseline"] = "comments_users"

    METHOD = "Baseline Comments + Users"

    if METHOD in method_name_list:
        reddit_k_list = add_results(mse_results=reddit_mse,
                                    jaccard_results=reddit_jaccard,
                                    feature_names=feature_names,
                                    feature_name_offset=7,
                                    method=METHOD,
                                    target_name_list=CONFIGURATION_DICT["target_name_list"],
                                    results_file_paths=get_results_file_paths(output_folder, CONFIGURATION_DICT))

    CONFIGURATION_DICT = dict()
    CONFIGURATION_DICT["target_name_list"] = ["comments", "users", "score_wilson", "controversiality_wilson"]
    CONFIGURATION_DICT["osn_name_focus"] = "reddit"
    CONFIGURATION_DICT["target_osn_name"] = "reddit"
    CONFIGURATION_DICT["feature_osn_name_list"] = ["reddit"]
    CONFIGURATION_DICT["add_branching_features"] = False
    CONFIGURATION_DICT["add_usergraph_features"] = False
    CONFIGURATION_DICT["add_temporal_features"] = False
    CONFIGURATION_DICT["add_author_features"] = False

    CONFIGURATION_DICT["baseline"] = "simple graph"

    METHOD = "Simple Graph"

    if METHOD in method_name_list:
        reddit_k_list = add_results(mse_results=reddit_mse,
                                    jaccard_results=reddit_jaccard,
                                    feature_names=feature_names,
                                    feature_name_offset=7,
                                    method=METHOD,
                                    target_name_list=CONFIGURATION_DICT["target_name_list"],
                                    results_file_paths=get_results_file_paths(output_folder, CONFIGURATION_DICT))

    CONFIGURATION_DICT = dict()
    CONFIGURATION_DICT["target_name_list"] = ["comments", "users", "score_wilson", "controversiality_wilson"]
    CONFIGURATION_DICT["osn_name_focus"] = "reddit"
    CONFIGURATION_DICT["target_osn_name"] = "reddit"
    CONFIGURATION_DICT["feature_osn_name_list"] = ["reddit"]
    CONFIGURATION_DICT["add_branching_features"] = False
    CONFIGURATION_DICT["add_usergraph_features"] = False
    CONFIGURATION_DICT["add_temporal_features"] = True
    CONFIGURATION_DICT["add_author_features"] = False

    METHOD = "Temporal"

    if METHOD in method_name_list:
        reddit_k_list = add_results(mse_results=reddit_mse,
                                    jaccard_results=reddit_jaccard,
                                    feature_names=feature_names,
                                    feature_name_offset=7,
                                    method=METHOD,
                                    target_name_list=CONFIGURATION_DICT["target_name_list"],
                                    results_file_paths=get_results_file_paths(output_folder, CONFIGURATION_DICT))

    CONFIGURATION_DICT = dict()
    CONFIGURATION_DICT["target_name_list"] = ["comments", "users", "score_wilson", "controversiality_wilson"]
    CONFIGURATION_DICT["osn_name_focus"] = "reddit"
    CONFIGURATION_DICT["target_osn_name"] = "reddit"
    CONFIGURATION_DICT["feature_osn_name_list"] = ["reddit"]
    CONFIGURATION_DICT["add_branching_features"] = True
    CONFIGURATION_DICT["add_usergraph_features"] = True
    CONFIGURATION_DICT["add_temporal_features"] = False
    CONFIGURATION_DICT["add_author_features"] = False

    METHOD = "Comment Tree + User Graph"

    if METHOD in method_name_list:
        reddit_k_list = add_results(mse_results=reddit_mse,
                                    jaccard_results=reddit_jaccard,
                                    feature_names=feature_names,
                                    feature_name_offset=7,
                                    method=METHOD,
                                    target_name_list=CONFIGURATION_DICT["target_name_list"],
                                    results_file_paths=get_results_file_paths(output_folder, CONFIGURATION_DICT))

    CONFIGURATION_DICT = dict()
    CONFIGURATION_DICT["target_name_list"] = ["comments", "users", "score_wilson", "controversiality_wilson"]
    CONFIGURATION_DICT["osn_name_focus"] = "reddit"
    CONFIGURATION_DICT["target_osn_name"] = "reddit"
    CONFIGURATION_DICT["feature_osn_name_list"] = ["reddit"]
    CONFIGURATION_DICT["add_branching_features"] = True
    CONFIGURATION_DICT["add_usergraph_features"] = False
    CONFIGURATION_DICT["add_temporal_features"] = False
    CONFIGURATION_DICT["add_author_features"] = False

    METHOD = "Comment Tree"

    if METHOD in method_name_list:
        reddit_k_list = add_results(mse_results=reddit_mse,
                                    jaccard_results=reddit_jaccard,
                                    feature_names=feature_names,
                                    feature_name_offset=7,
                                    method=METHOD,
                                    target_name_list=CONFIGURATION_DICT["target_name_list"],
                                    results_file_paths=get_results_file_paths(output_folder, CONFIGURATION_DICT))

    CONFIGURATION_DICT = dict()
    CONFIGURATION_DICT["target_name_list"] = ["comments", "users", "score_wilson", "controversiality_wilson"]
    CONFIGURATION_DICT["osn_name_focus"] = "reddit"
    CONFIGURATION_DICT["target_osn_name"] = "reddit"
    CONFIGURATION_DICT["feature_osn_name_list"] = ["reddit"]
    CONFIGURATION_DICT["add_branching_features"] = False
    CONFIGURATION_DICT["add_usergraph_features"] = True
    CONFIGURATION_DICT["add_temporal_features"] = False
    CONFIGURATION_DICT["add_author_features"] = False

    METHOD = "User Graph"

    if METHOD in method_name_list:
        reddit_k_list = add_results(mse_results=reddit_mse,
                                    jaccard_results=reddit_jaccard,
                                    feature_names=feature_names,
                                    feature_name_offset=7,
                                    method=METHOD,
                                    target_name_list=CONFIGURATION_DICT["target_name_list"],
                                    results_file_paths=get_results_file_paths(output_folder, CONFIGURATION_DICT))

    CONFIGURATION_DICT = dict()
    CONFIGURATION_DICT["target_name_list"] = ["comments", "users", "score_wilson", "controversiality_wilson"]
    CONFIGURATION_DICT["osn_name_focus"] = "reddit"
    CONFIGURATION_DICT["target_osn_name"] = "reddit"
    CONFIGURATION_DICT["feature_osn_name_list"] = ["reddit"]
    CONFIGURATION_DICT["add_branching_features"] = True
    CONFIGURATION_DICT["add_usergraph_features"] = True
    CONFIGURATION_DICT["add_temporal_features"] = True
    CONFIGURATION_DICT["add_author_features"] = False

    METHOD = "All"

    if METHOD in method_name_list:
        reddit_k_list = add_results(mse_results=reddit_mse,
                                    jaccard_results=reddit_jaccard,
                                    feature_names=feature_names,
                                    feature_name_offset=7,
                                    method=METHOD,
                                    target_name_list=CONFIGURATION_DICT["target_name_list"],
                                    results_file_paths=get_results_file_paths(output_folder, CONFIGURATION_DICT))

    return reddit_mse, reddit_jaccard, reddit_k_list


def make_reddit_figures(output_path_prefix, method_name_list, reddit_mse, reddit_jaccard, reddit_k_list):
    sns.set_style("darkgrid")
    sns.set_context("paper")

    translator = get_method_name_to_legend_name_dict()

    reddit_k_list = list(reddit_k_list)

    fig, axes = plt.subplots(1, 2, sharex=True)

    axes[0].set_title("RedditNews Comments")
    axes[1].set_title("RedditNews Users")

    plt.locator_params(nbins=8)

    # Comments
    for m, method in enumerate(method_name_list):
        axes[0].set_ylabel("MSE")
        axes[0].set_xlabel("Lifetime (sec)")
        axes[0].plot(reddit_k_list[1:],
                            handle_nan(reddit_mse[method]["comments"].mean(axis=1))[1:],
                            label=translator[method])

    # Users
    for m, method in enumerate(method_name_list):
        # axes[1].set_ylabel("MSE")
        axes[1].set_xlabel("Lifetime (sec)")
        axes[1].plot(reddit_k_list[1:],
                            handle_nan(reddit_mse[method]["users"].mean(axis=1))[1:],
                            label=translator[method])

    axes[1].legend(loc="lower left")

    # plt.show()
    plt.savefig(output_path_prefix + "_mse_com_use_SNOW" + ".png", format="png")
    plt.savefig(output_path_prefix + "_mse_com_use_SNOW" + ".eps", format="eps")

    ####################################################################################################################

    fig, axes = plt.subplots(1, 2, sharex=True)

    axes[0].set_title("RedditNews Score")
    axes[1].set_title("RedditNews Controversiality")

    plt.locator_params(nbins=8)

    # Score-Wilson
    for m, method in enumerate(method_name_list):
        axes[0].set_ylabel("MSE")
        axes[0].set_xlabel("Lifetime (sec)")
        axes[0].plot(reddit_k_list[1:],
                            handle_nan(reddit_mse[method]["score_wilson"].mean(axis=1))[1:],
                            label=translator[method])

    # Controversiality - Wilson
    for m, method in enumerate(method_name_list):
        # axes[1].set_ylabel("MSE")
        axes[1].set_xlabel("Lifetime (sec)")
        axes[1].plot(reddit_k_list[1:],
                            handle_nan(reddit_mse[method]["controversiality_wilson"].mean(axis=1))[1:],
                            label=translator[method])

    axes[1].legend(loc="lower left")

    # plt.show()
    plt.savefig(output_path_prefix + "_mse_sco_ctr_SNOW" + ".png", format="png")
    plt.savefig(output_path_prefix + "_mse_sco_ctr_SNOW" + ".eps", format="eps")

    ####################################################################################################################

    # Controversiality - Wilson
    for m, method in enumerate(method_name_list):
        print(method, "Top-100 Jaccard index")
        print(reddit_jaccard[method]["controversiality_wilson"].mean(axis=1))
