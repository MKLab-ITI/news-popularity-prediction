__author__ = 'Georgios Rizos (georgerizos@iti.gr)'

import numpy as np

from news_popularity_prediction.learning.ranking import load_k_evaluation_measures


def get_method_name_to_legend_name_dict():
    translate = dict()
    translate["Baseline Mean"] = "base\_mean"
    translate["Baseline Median"] = "base\_median"
    translate["Baseline Comments"] = "n\_c"
    translate["Baseline Users"] = "n\_u"
    translate["Baseline Comments + Users"] = "n\_c+n\_u"

    translate["Simple Graph"] = "simple\_graph"
    translate["Temporal"] = "temporal"

    translate["Comment Tree + User Graph"] = "all\_graph"
    translate["Comment Tree"] = "comment\_tree"
    translate["User Graph"] = "user\_graph"
    translate["All"] = "all"

    return translate


def handle_nan(array):
    array[np.isnan(array)] = 0.0

    return array


def get_results_reddit_focus_folder(output_folder):
    results_folder = output_folder + "/reddit/results/reddit"
    return results_folder


def get_results_slashdot_focus_folder(output_folder):
    results_folder = output_folder + "/slashdot/results/slashdot"
    return results_folder


def get_results_barrapunto_focus_folder(output_folder):
    results_folder = output_folder + "/barrapunto/results/slashdot"
    return results_folder


def get_experiment_construction_type(configuration_dict):

    feature_osn_name_list = configuration_dict["feature_osn_name_list"]

    experiment_construction_type = ""

    for osn_name in feature_osn_name_list:
        experiment_construction_type += osn_name + "_"

    if "baseline" in configuration_dict.keys():
        experiment_construction_type += "baseline_" + configuration_dict["baseline"]

        return experiment_construction_type

    if configuration_dict["add_branching_features"]:
        experiment_construction_type += "branching_"
    else:
        experiment_construction_type += ""

    if configuration_dict["add_usergraph_features"]:
        experiment_construction_type += "usergraph_"
    else:
        experiment_construction_type += ""

    if configuration_dict["add_temporal_features"]:
        experiment_construction_type += "temporal_"
    else:
        experiment_construction_type += ""

    return experiment_construction_type


def get_results_file_paths(output_folder, configuration_dict):
    target_name_list = configuration_dict["target_name_list"]
    osn_name_focus = configuration_dict["osn_name_focus"]
    if osn_name_focus == "reddit":
        results_folder = get_results_reddit_focus_folder(output_folder)
    elif osn_name_focus == "slashdot":
        results_folder = get_results_slashdot_focus_folder(output_folder)
    elif osn_name_focus == "barrapunto":
        results_folder = get_results_barrapunto_focus_folder(output_folder)
    else:
        print("Invalid OSN focus.")
        raise RuntimeError
    target_osn_name = configuration_dict["target_osn_name"]
    experiment_construction_type = get_experiment_construction_type(configuration_dict)

    results_file_path_dict = dict()
    for target_name in target_name_list:
        results_file_path = results_folder + "/" +\
                            "target_" + target_osn_name + "_" +\
                            target_name + "_" +\
                            experiment_construction_type
        results_file_path_dict[target_name] = results_file_path

    return results_file_path_dict


def add_results(mse_results,
                jaccard_results,
                feature_names,
                feature_name_offset,
                method,
                target_name_list,
                results_file_paths):
    mse_results[method] = dict()
    jaccard_results[method] = dict()
    feature_names[method] = dict()
    for target_name in target_name_list:
        k_list, k_evaluation_measures, feature_names_list = load_k_evaluation_measures(results_file_paths[target_name])
        mse_results[method][target_name] = k_evaluation_measures[2]
        jaccard_results[method][target_name] = k_evaluation_measures[3]

        feature_names[method] = [feature_name[feature_name_offset:] for feature_name in feature_names_list]

    return k_list
