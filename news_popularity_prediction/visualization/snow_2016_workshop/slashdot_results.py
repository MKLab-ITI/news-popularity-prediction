__author__ = 'Georgios Rizos (georgerizos@iti.gr)'

import matplotlib
import matplotlib.pyplot as plt
import seaborn as sns

from news_popularity_prediction.visualization.snow_2016_workshop.common import get_results_file_paths, handle_nan,\
    get_method_name_to_legend_name_dict, add_results

matplotlib.rcParams["ps.useafm"] = True
matplotlib.rcParams["pdf.use14corefonts"] = True
matplotlib.rcParams["text.usetex"] = True


def read_slashdot_result_data(output_folder, method_name_list):
    slashdot_mse = dict()
    slashdot_jaccard = dict()
    feature_names = dict()
    slashdot_k_list = None

    CONFIGURATION_DICT = dict()
    CONFIGURATION_DICT["target_name_list"] = ["comments", "users"]
    CONFIGURATION_DICT["osn_name_focus"] = "slashdot"
    CONFIGURATION_DICT["target_osn_name"] = "slashdot"
    CONFIGURATION_DICT["feature_osn_name_list"] = ["slashdot"]
    CONFIGURATION_DICT["add_branching_features"] = False
    CONFIGURATION_DICT["add_usergraph_features"] = False
    CONFIGURATION_DICT["add_temporal_features"] = False
    CONFIGURATION_DICT["add_author_features"] = False

    CONFIGURATION_DICT["baseline"] = "mean"

    METHOD = "Baseline Mean"

    if METHOD in method_name_list:
        slashdot_k_list = add_results(mse_results=slashdot_mse,
                                      jaccard_results=slashdot_jaccard,
                                      feature_names=feature_names,
                                      feature_name_offset=9,
                                      method=METHOD,
                                      target_name_list=CONFIGURATION_DICT["target_name_list"],
                                      results_file_paths=get_results_file_paths(output_folder, CONFIGURATION_DICT))

    CONFIGURATION_DICT = dict()
    CONFIGURATION_DICT["target_name_list"] = ["comments", "users"]
    CONFIGURATION_DICT["osn_name_focus"] = "slashdot"
    CONFIGURATION_DICT["target_osn_name"] = "slashdot"
    CONFIGURATION_DICT["feature_osn_name_list"] = ["slashdot"]
    CONFIGURATION_DICT["add_branching_features"] = False
    CONFIGURATION_DICT["add_usergraph_features"] = False
    CONFIGURATION_DICT["add_temporal_features"] = False
    CONFIGURATION_DICT["add_author_features"] = False

    CONFIGURATION_DICT["baseline"] = "median"

    METHOD = "Baseline Median"

    if METHOD in method_name_list:
        slashdot_k_list = add_results(mse_results=slashdot_mse,
                                      jaccard_results=slashdot_jaccard,
                                      feature_names=feature_names,
                                      feature_name_offset=9,
                                      method=METHOD,
                                      target_name_list=CONFIGURATION_DICT["target_name_list"],
                                      results_file_paths=get_results_file_paths(output_folder, CONFIGURATION_DICT))

    CONFIGURATION_DICT = dict()
    CONFIGURATION_DICT["target_name_list"] = ["comments", "users"]
    CONFIGURATION_DICT["osn_name_focus"] = "slashdot"
    CONFIGURATION_DICT["target_osn_name"] = "slashdot"
    CONFIGURATION_DICT["feature_osn_name_list"] = ["slashdot"]
    CONFIGURATION_DICT["add_branching_features"] = False
    CONFIGURATION_DICT["add_usergraph_features"] = False
    CONFIGURATION_DICT["add_temporal_features"] = False
    CONFIGURATION_DICT["add_author_features"] = False

    CONFIGURATION_DICT["baseline"] = "comments"

    METHOD = "Baseline Comments"

    if METHOD in method_name_list:
        slashdot_k_list = add_results(mse_results=slashdot_mse,
                                      jaccard_results=slashdot_jaccard,
                                      feature_names=feature_names,
                                      feature_name_offset=9,
                                      method=METHOD,
                                      target_name_list=CONFIGURATION_DICT["target_name_list"],
                                      results_file_paths=get_results_file_paths(output_folder, CONFIGURATION_DICT))

    CONFIGURATION_DICT = dict()
    CONFIGURATION_DICT["target_name_list"] = ["comments", "users"]
    CONFIGURATION_DICT["osn_name_focus"] = "slashdot"
    CONFIGURATION_DICT["target_osn_name"] = "slashdot"
    CONFIGURATION_DICT["feature_osn_name_list"] = ["slashdot"]
    CONFIGURATION_DICT["add_branching_features"] = False
    CONFIGURATION_DICT["add_usergraph_features"] = False
    CONFIGURATION_DICT["add_temporal_features"] = False
    CONFIGURATION_DICT["add_author_features"] = False

    CONFIGURATION_DICT["baseline"] = "users"

    METHOD = "Baseline Users"

    if METHOD in method_name_list:
        slashdot_k_list = add_results(mse_results=slashdot_mse,
                                      jaccard_results=slashdot_jaccard,
                                      feature_names=feature_names,
                                      feature_name_offset=9,
                                      method=METHOD,
                                      target_name_list=CONFIGURATION_DICT["target_name_list"],
                                      results_file_paths=get_results_file_paths(output_folder, CONFIGURATION_DICT))

    CONFIGURATION_DICT = dict()
    CONFIGURATION_DICT["target_name_list"] = ["comments", "users"]
    CONFIGURATION_DICT["osn_name_focus"] = "slashdot"
    CONFIGURATION_DICT["target_osn_name"] = "slashdot"
    CONFIGURATION_DICT["feature_osn_name_list"] = ["slashdot"]
    CONFIGURATION_DICT["add_branching_features"] = False
    CONFIGURATION_DICT["add_usergraph_features"] = False
    CONFIGURATION_DICT["add_temporal_features"] = False
    CONFIGURATION_DICT["add_author_features"] = False

    CONFIGURATION_DICT["baseline"] = "comments_users"

    METHOD = "Baseline Comments + Users"

    if METHOD in method_name_list:
        slashdot_k_list = add_results(mse_results=slashdot_mse,
                                      jaccard_results=slashdot_jaccard,
                                      feature_names=feature_names,
                                      feature_name_offset=9,
                                      method=METHOD,
                                      target_name_list=CONFIGURATION_DICT["target_name_list"],
                                      results_file_paths=get_results_file_paths(output_folder, CONFIGURATION_DICT))

    CONFIGURATION_DICT = dict()
    CONFIGURATION_DICT["target_name_list"] = ["comments", "users"]
    CONFIGURATION_DICT["osn_name_focus"] = "slashdot"
    CONFIGURATION_DICT["target_osn_name"] = "slashdot"
    CONFIGURATION_DICT["feature_osn_name_list"] = ["slashdot"]
    CONFIGURATION_DICT["add_branching_features"] = False
    CONFIGURATION_DICT["add_usergraph_features"] = False
    CONFIGURATION_DICT["add_temporal_features"] = False
    CONFIGURATION_DICT["add_author_features"] = False

    CONFIGURATION_DICT["baseline"] = "simple graph"

    METHOD = "Simple Graph"

    if METHOD in method_name_list:
        slashdot_k_list = add_results(mse_results=slashdot_mse,
                                      jaccard_results=slashdot_jaccard,
                                      feature_names=feature_names,
                                      feature_name_offset=9,
                                      method=METHOD,
                                      target_name_list=CONFIGURATION_DICT["target_name_list"],
                                      results_file_paths=get_results_file_paths(output_folder, CONFIGURATION_DICT))

    CONFIGURATION_DICT = dict()
    CONFIGURATION_DICT["target_name_list"] = ["comments", "users"]
    CONFIGURATION_DICT["osn_name_focus"] = "slashdot"
    CONFIGURATION_DICT["target_osn_name"] = "slashdot"
    CONFIGURATION_DICT["feature_osn_name_list"] = ["slashdot"]
    CONFIGURATION_DICT["add_branching_features"] = False
    CONFIGURATION_DICT["add_usergraph_features"] = False
    CONFIGURATION_DICT["add_temporal_features"] = True
    CONFIGURATION_DICT["add_author_features"] = False

    METHOD = "Temporal"

    if METHOD in method_name_list:
        slashdot_k_list = add_results(mse_results=slashdot_mse,
                                      jaccard_results=slashdot_jaccard,
                                      feature_names=feature_names,
                                      feature_name_offset=9,
                                      method=METHOD,
                                      target_name_list=CONFIGURATION_DICT["target_name_list"],
                                      results_file_paths=get_results_file_paths(output_folder, CONFIGURATION_DICT))

    CONFIGURATION_DICT = dict()
    CONFIGURATION_DICT["target_name_list"] = ["comments", "users"]
    CONFIGURATION_DICT["osn_name_focus"] = "slashdot"
    CONFIGURATION_DICT["target_osn_name"] = "slashdot"
    CONFIGURATION_DICT["feature_osn_name_list"] = ["slashdot"]
    CONFIGURATION_DICT["add_branching_features"] = True
    CONFIGURATION_DICT["add_usergraph_features"] = True
    CONFIGURATION_DICT["add_temporal_features"] = False
    CONFIGURATION_DICT["add_author_features"] = False

    METHOD = "Comment Tree + User Graph"

    if METHOD in method_name_list:
        slashdot_k_list = add_results(mse_results=slashdot_mse,
                                      jaccard_results=slashdot_jaccard,
                                      feature_names=feature_names,
                                      feature_name_offset=9,
                                      method=METHOD,
                                      target_name_list=CONFIGURATION_DICT["target_name_list"],
                                      results_file_paths=get_results_file_paths(output_folder, CONFIGURATION_DICT))

    CONFIGURATION_DICT = dict()
    CONFIGURATION_DICT["target_name_list"] = ["comments", "users"]
    CONFIGURATION_DICT["osn_name_focus"] = "slashdot"
    CONFIGURATION_DICT["target_osn_name"] = "slashdot"
    CONFIGURATION_DICT["feature_osn_name_list"] = ["slashdot"]
    CONFIGURATION_DICT["add_branching_features"] = True
    CONFIGURATION_DICT["add_usergraph_features"] = False
    CONFIGURATION_DICT["add_temporal_features"] = False
    CONFIGURATION_DICT["add_author_features"] = False

    METHOD = "Comment Tree"

    if METHOD in method_name_list:
        slashdot_k_list = add_results(mse_results=slashdot_mse,
                                      jaccard_results=slashdot_jaccard,
                                      feature_names=feature_names,
                                      feature_name_offset=9,
                                      method=METHOD,
                                      target_name_list=CONFIGURATION_DICT["target_name_list"],
                                      results_file_paths=get_results_file_paths(output_folder, CONFIGURATION_DICT))

    CONFIGURATION_DICT = dict()
    CONFIGURATION_DICT["target_name_list"] = ["comments", "users"]
    CONFIGURATION_DICT["osn_name_focus"] = "slashdot"
    CONFIGURATION_DICT["target_osn_name"] = "slashdot"
    CONFIGURATION_DICT["feature_osn_name_list"] = ["slashdot"]
    CONFIGURATION_DICT["add_branching_features"] = False
    CONFIGURATION_DICT["add_usergraph_features"] = True
    CONFIGURATION_DICT["add_temporal_features"] = False
    CONFIGURATION_DICT["add_author_features"] = False

    METHOD = "User Graph"

    if METHOD in method_name_list:
        slashdot_k_list = add_results(mse_results=slashdot_mse,
                                      jaccard_results=slashdot_jaccard,
                                      feature_names=feature_names,
                                      feature_name_offset=9,
                                      method=METHOD,
                                      target_name_list=CONFIGURATION_DICT["target_name_list"],
                                      results_file_paths=get_results_file_paths(output_folder, CONFIGURATION_DICT))

    CONFIGURATION_DICT = dict()
    CONFIGURATION_DICT["target_name_list"] = ["comments", "users"]
    CONFIGURATION_DICT["osn_name_focus"] = "slashdot"
    CONFIGURATION_DICT["target_osn_name"] = "slashdot"
    CONFIGURATION_DICT["feature_osn_name_list"] = ["slashdot"]
    CONFIGURATION_DICT["add_branching_features"] = True
    CONFIGURATION_DICT["add_usergraph_features"] = True
    CONFIGURATION_DICT["add_temporal_features"] = True
    CONFIGURATION_DICT["add_author_features"] = False

    METHOD = "All"

    if METHOD in method_name_list:
        slashdot_k_list = add_results(mse_results=slashdot_mse,
                                      jaccard_results=slashdot_jaccard,
                                      feature_names=feature_names,
                                      feature_name_offset=9,
                                      method=METHOD,
                                      target_name_list=CONFIGURATION_DICT["target_name_list"],
                                      results_file_paths=get_results_file_paths(output_folder, CONFIGURATION_DICT))

    return slashdot_mse, slashdot_jaccard, slashdot_k_list

########################################################################################################################

def read_barrapunto_result_data(output_folder, method_name_list):
    barrapunto_mse = dict()
    barrapunto_jaccard = dict()
    feature_names = dict()
    barrapunto_k_list = None

    CONFIGURATION_DICT = dict()
    CONFIGURATION_DICT["target_name_list"] = ["comments", "users"]
    CONFIGURATION_DICT["osn_name_focus"] = "barrapunto"
    CONFIGURATION_DICT["target_osn_name"] = "slashdot"
    CONFIGURATION_DICT["feature_osn_name_list"] = ["slashdot"]
    CONFIGURATION_DICT["add_branching_features"] = False
    CONFIGURATION_DICT["add_usergraph_features"] = False
    CONFIGURATION_DICT["add_temporal_features"] = False
    CONFIGURATION_DICT["add_author_features"] = False

    CONFIGURATION_DICT["baseline"] = "mean"

    METHOD = "Baseline Mean"

    if METHOD in method_name_list:
        barrapunto_k_list = add_results(mse_results=barrapunto_mse,
                                        jaccard_results=barrapunto_jaccard,
                                        feature_names=feature_names,
                                        feature_name_offset=9,
                                        method=METHOD,
                                        target_name_list=CONFIGURATION_DICT["target_name_list"],
                                        results_file_paths=get_results_file_paths(output_folder, CONFIGURATION_DICT))

    CONFIGURATION_DICT = dict()
    CONFIGURATION_DICT["target_name_list"] = ["comments", "users"]
    CONFIGURATION_DICT["osn_name_focus"] = "barrapunto"
    CONFIGURATION_DICT["target_osn_name"] = "slashdot"
    CONFIGURATION_DICT["feature_osn_name_list"] = ["slashdot"]
    CONFIGURATION_DICT["add_branching_features"] = False
    CONFIGURATION_DICT["add_usergraph_features"] = False
    CONFIGURATION_DICT["add_temporal_features"] = False
    CONFIGURATION_DICT["add_author_features"] = False

    CONFIGURATION_DICT["baseline"] = "median"

    METHOD = "Baseline Median"

    if METHOD in method_name_list:
        barrapunto_k_list = add_results(mse_results=barrapunto_mse,
                                        jaccard_results=barrapunto_jaccard,
                                        feature_names=feature_names,
                                        feature_name_offset=9,
                                        method=METHOD,
                                        target_name_list=CONFIGURATION_DICT["target_name_list"],
                                        results_file_paths=get_results_file_paths(output_folder, CONFIGURATION_DICT))

    CONFIGURATION_DICT = dict()
    CONFIGURATION_DICT["target_name_list"] = ["comments", "users"]
    CONFIGURATION_DICT["osn_name_focus"] = "barrapunto"
    CONFIGURATION_DICT["target_osn_name"] = "slashdot"
    CONFIGURATION_DICT["feature_osn_name_list"] = ["slashdot"]
    CONFIGURATION_DICT["add_branching_features"] = False
    CONFIGURATION_DICT["add_usergraph_features"] = False
    CONFIGURATION_DICT["add_temporal_features"] = False
    CONFIGURATION_DICT["add_author_features"] = False

    CONFIGURATION_DICT["baseline"] = "comments"

    METHOD = "Baseline Comments"

    if METHOD in method_name_list:
        barrapunto_k_list = add_results(mse_results=barrapunto_mse,
                                        jaccard_results=barrapunto_jaccard,
                                        feature_names=feature_names,
                                        feature_name_offset=9,
                                        method=METHOD,
                                        target_name_list=CONFIGURATION_DICT["target_name_list"],
                                        results_file_paths=get_results_file_paths(output_folder, CONFIGURATION_DICT))

    CONFIGURATION_DICT = dict()
    CONFIGURATION_DICT["target_name_list"] = ["comments", "users"]
    CONFIGURATION_DICT["osn_name_focus"] = "barrapunto"
    CONFIGURATION_DICT["target_osn_name"] = "slashdot"
    CONFIGURATION_DICT["feature_osn_name_list"] = ["slashdot"]
    CONFIGURATION_DICT["add_branching_features"] = False
    CONFIGURATION_DICT["add_usergraph_features"] = False
    CONFIGURATION_DICT["add_temporal_features"] = False
    CONFIGURATION_DICT["add_author_features"] = False

    CONFIGURATION_DICT["baseline"] = "users"

    METHOD = "Baseline Users"

    if METHOD in method_name_list:
        barrapunto_k_list = add_results(mse_results=barrapunto_mse,
                                        jaccard_results=barrapunto_jaccard,
                                        feature_names=feature_names,
                                        feature_name_offset=9,
                                        method=METHOD,
                                        target_name_list=CONFIGURATION_DICT["target_name_list"],
                                        results_file_paths=get_results_file_paths(output_folder, CONFIGURATION_DICT))

    CONFIGURATION_DICT = dict()
    CONFIGURATION_DICT["target_name_list"] = ["comments", "users"]
    CONFIGURATION_DICT["osn_name_focus"] = "barrapunto"
    CONFIGURATION_DICT["target_osn_name"] = "slashdot"
    CONFIGURATION_DICT["feature_osn_name_list"] = ["slashdot"]
    CONFIGURATION_DICT["add_branching_features"] = False
    CONFIGURATION_DICT["add_usergraph_features"] = False
    CONFIGURATION_DICT["add_temporal_features"] = False
    CONFIGURATION_DICT["add_author_features"] = False

    CONFIGURATION_DICT["baseline"] = "comments_users"

    METHOD = "Baseline Comments + Users"

    if METHOD in method_name_list:
        barrapunto_k_list = add_results(mse_results=barrapunto_mse,
                                        jaccard_results=barrapunto_jaccard,
                                        feature_names=feature_names,
                                        feature_name_offset=9,
                                        method=METHOD,
                                        target_name_list=CONFIGURATION_DICT["target_name_list"],
                                        results_file_paths=get_results_file_paths(output_folder, CONFIGURATION_DICT))

    CONFIGURATION_DICT = dict()
    CONFIGURATION_DICT["target_name_list"] = ["comments", "users"]
    CONFIGURATION_DICT["osn_name_focus"] = "barrapunto"
    CONFIGURATION_DICT["target_osn_name"] = "slashdot"
    CONFIGURATION_DICT["feature_osn_name_list"] = ["slashdot"]
    CONFIGURATION_DICT["add_branching_features"] = False
    CONFIGURATION_DICT["add_usergraph_features"] = False
    CONFIGURATION_DICT["add_temporal_features"] = False
    CONFIGURATION_DICT["add_author_features"] = False

    CONFIGURATION_DICT["baseline"] = "simple graph"

    METHOD = "Simple Graph"

    if METHOD in method_name_list:
        barrapunto_k_list = add_results(mse_results=barrapunto_mse,
                                        jaccard_results=barrapunto_jaccard,
                                        feature_names=feature_names,
                                        feature_name_offset=9,
                                        method=METHOD,
                                        target_name_list=CONFIGURATION_DICT["target_name_list"],
                                        results_file_paths=get_results_file_paths(output_folder, CONFIGURATION_DICT))

    CONFIGURATION_DICT = dict()
    CONFIGURATION_DICT["target_name_list"] = ["comments", "users"]
    CONFIGURATION_DICT["osn_name_focus"] = "barrapunto"
    CONFIGURATION_DICT["target_osn_name"] = "slashdot"
    CONFIGURATION_DICT["feature_osn_name_list"] = ["slashdot"]
    CONFIGURATION_DICT["add_branching_features"] = False
    CONFIGURATION_DICT["add_usergraph_features"] = False
    CONFIGURATION_DICT["add_temporal_features"] = True
    CONFIGURATION_DICT["add_author_features"] = False

    METHOD = "Temporal"

    if METHOD in method_name_list:
        barrapunto_k_list = add_results(mse_results=barrapunto_mse,
                                        jaccard_results=barrapunto_jaccard,
                                        feature_names=feature_names,
                                        feature_name_offset=9,
                                        method=METHOD,
                                        target_name_list=CONFIGURATION_DICT["target_name_list"],
                                        results_file_paths=get_results_file_paths(output_folder, CONFIGURATION_DICT))

    CONFIGURATION_DICT = dict()
    CONFIGURATION_DICT["target_name_list"] = ["comments", "users"]
    CONFIGURATION_DICT["osn_name_focus"] = "barrapunto"
    CONFIGURATION_DICT["target_osn_name"] = "slashdot"
    CONFIGURATION_DICT["feature_osn_name_list"] = ["slashdot"]
    CONFIGURATION_DICT["add_branching_features"] = True
    CONFIGURATION_DICT["add_usergraph_features"] = True
    CONFIGURATION_DICT["add_temporal_features"] = False
    CONFIGURATION_DICT["add_author_features"] = False

    METHOD = "Comment Tree + User Graph"

    if METHOD in method_name_list:
        barrapunto_k_list = add_results(mse_results=barrapunto_mse,
                                        jaccard_results=barrapunto_jaccard,
                                        feature_names=feature_names,
                                        feature_name_offset=9,
                                        method=METHOD,
                                        target_name_list=CONFIGURATION_DICT["target_name_list"],
                                        results_file_paths=get_results_file_paths(output_folder, CONFIGURATION_DICT))

    CONFIGURATION_DICT = dict()
    CONFIGURATION_DICT["target_name_list"] = ["comments", "users"]
    CONFIGURATION_DICT["osn_name_focus"] = "barrapunto"
    CONFIGURATION_DICT["target_osn_name"] = "slashdot"
    CONFIGURATION_DICT["feature_osn_name_list"] = ["slashdot"]
    CONFIGURATION_DICT["add_branching_features"] = True
    CONFIGURATION_DICT["add_usergraph_features"] = False
    CONFIGURATION_DICT["add_temporal_features"] = False
    CONFIGURATION_DICT["add_author_features"] = False

    METHOD = "Comment Tree"

    if METHOD in method_name_list:
        barrapunto_k_list = add_results(mse_results=barrapunto_mse,
                                        jaccard_results=barrapunto_jaccard,
                                        feature_names=feature_names,
                                        feature_name_offset=9,
                                        method=METHOD,
                                        target_name_list=CONFIGURATION_DICT["target_name_list"],
                                        results_file_paths=get_results_file_paths(output_folder, CONFIGURATION_DICT))

    CONFIGURATION_DICT = dict()
    CONFIGURATION_DICT["target_name_list"] = ["comments", "users"]
    CONFIGURATION_DICT["osn_name_focus"] = "barrapunto"
    CONFIGURATION_DICT["target_osn_name"] = "slashdot"
    CONFIGURATION_DICT["feature_osn_name_list"] = ["slashdot"]
    CONFIGURATION_DICT["add_branching_features"] = False
    CONFIGURATION_DICT["add_usergraph_features"] = True
    CONFIGURATION_DICT["add_temporal_features"] = False
    CONFIGURATION_DICT["add_author_features"] = False

    METHOD = "User Graph"

    if METHOD in method_name_list:
        barrapunto_k_list = add_results(mse_results=barrapunto_mse,
                                        jaccard_results=barrapunto_jaccard,
                                        feature_names=feature_names,
                                        feature_name_offset=9,
                                        method=METHOD,
                                        target_name_list=CONFIGURATION_DICT["target_name_list"],
                                        results_file_paths=get_results_file_paths(output_folder, CONFIGURATION_DICT))

    CONFIGURATION_DICT = dict()
    CONFIGURATION_DICT["target_name_list"] = ["comments", "users"]
    CONFIGURATION_DICT["osn_name_focus"] = "barrapunto"
    CONFIGURATION_DICT["target_osn_name"] = "slashdot"
    CONFIGURATION_DICT["feature_osn_name_list"] = ["slashdot"]
    CONFIGURATION_DICT["add_branching_features"] = True
    CONFIGURATION_DICT["add_usergraph_features"] = True
    CONFIGURATION_DICT["add_temporal_features"] = True
    CONFIGURATION_DICT["add_author_features"] = False

    METHOD = "All"

    if METHOD in method_name_list:
        barrapunto_k_list = add_results(mse_results=barrapunto_mse,
                                        jaccard_results=barrapunto_jaccard,
                                        feature_names=feature_names,
                                        feature_name_offset=9,
                                        method=METHOD,
                                        target_name_list=CONFIGURATION_DICT["target_name_list"],
                                        results_file_paths=get_results_file_paths(output_folder, CONFIGURATION_DICT))

    return barrapunto_mse, barrapunto_jaccard, barrapunto_k_list


def make_slashdot_figures(output_path_prefix, method_name_list, slashdot_mse, slashdot_jaccard, slashdot_k_list):
    sns.set_style("darkgrid")
    sns.set_context("paper")

    translator = get_method_name_to_legend_name_dict()

    slashdot_k_list = list(slashdot_k_list)

    fig, axes = plt.subplots(1, 2, sharex=True)

    axes[0].set_title("SlashDot Comments")
    axes[1].set_title("SlashDot Users")

    plt.locator_params(nbins=8)

    # Comments
    for m, method in enumerate(method_name_list):
        axes[0].set_ylabel("MSE")
        axes[0].set_xlabel("Lifetime (sec)")
        axes[0].plot(slashdot_k_list[1:],
                     handle_nan(slashdot_mse[method]["comments"].mean(axis=1))[1:],
                     label=translator[method])

    # Users
    for m, method in enumerate(method_name_list):
        # axes[1].set_ylabel("MSE")
        axes[1].set_xlabel("Lifetime (sec)")
        axes[1].plot(slashdot_k_list[1:],
                     handle_nan(slashdot_mse[method]["users"].mean(axis=1))[1:],
                     label=translator[method])


    axes[1].legend(loc="upper right")

    # plt.show()
    plt.savefig(output_path_prefix + "_mse_slashdot_SNOW" + ".png", format="png")
    plt.savefig(output_path_prefix + "_mse_slashdot_SNOW" + ".eps", format="eps")


def make_barrapunto_figures(output_path_prefix, method_name_list, barrapunto_mse, barrapunto_jaccard, barrapunto_k_list):
    sns.set_style("darkgrid")
    sns.set_context("paper")

    translator = get_method_name_to_legend_name_dict()

    barrapunto_k_list = list(barrapunto_k_list)

    fig, axes = plt.subplots(1, 2, sharex=True)

    axes[0].set_title("BarraPunto Comments")
    axes[1].set_title("BarraPunto Users")

    plt.locator_params(nbins=8)

    # Comments
    for m, method in enumerate(method_name_list):
        axes[0].set_ylabel("MSE")
        axes[0].set_xlabel("Lifetime (sec)")
        axes[0].plot(barrapunto_k_list[1:],
                        handle_nan(barrapunto_mse[method]["comments"].mean(axis=1))[1:],
                        label=translator[method])

    # Users
    for m, method in enumerate(method_name_list):
        # axes[1].set_ylabel("MSE")
        axes[1].set_xlabel("Lifetime (sec)")
        axes[1].plot(barrapunto_k_list[1:],
                        handle_nan(barrapunto_mse[method]["users"].mean(axis=1))[1:],
                        label=translator[method])


    axes[1].legend(loc="upper right")

    # plt.show()
    plt.savefig(output_path_prefix + "_mse_barrapunto_SNOW" + ".png", format="png")
    plt.savefig(output_path_prefix + "_mse_barrapunto_SNOW" + ".eps", format="eps")
