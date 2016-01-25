__author__ = 'Georgios Rizos (georgerizos@iti.gr)'

import os

import pandas as pd
import numpy as np

from news_popularity_prediction.learning import ranking, cascade_lifetime
from news_popularity_prediction.discussion.features import get_branching_feature_names, get_usergraph_feature_names,\
    get_temporal_feature_names
from news_popularity_prediction.discussion.datasetwide import load_dataset_user_anonymizer
from news_popularity_prediction.datautil.feature_rw import h5_open, h5load_from, get_target_value,\
    get_kth_row, h5_close, h5store_at
from news_popularity_prediction.learning import concatenate_features


class DiscussionModellingExperiment:
    def __init__(self,
                 experiment_construction_dict,
                 data_folder,
                 feature_osn_name_list,
                 target_osn_name,
                 osn_name_focus,
                 target_name_list,
                 number_of_folds):
        ################################################################################################################
        # Initialize object fields.
        ################################################################################################################
        self.experiment_construction_dict = experiment_construction_dict
        self.data_folder = data_folder
        self.feature_osn_name_list = feature_osn_name_list
        self.target_osn_name = target_osn_name
        self.osn_name_focus = osn_name_focus
        self.target_name_list = target_name_list
        self.number_of_folds = number_of_folds

        ################################################################################################################
        # Configure experiment.
        ################################################################################################################
        # Configure according to experiment construction type.
        experiment_construction_tuple = self.parse_experiment_construction(self.experiment_construction_dict)
        self.experiment_construction_type = experiment_construction_tuple[0]
        self.add_branching_features = experiment_construction_tuple[1]
        self.add_usergraph_features = experiment_construction_tuple[2]
        self.add_temporal_features = experiment_construction_tuple[3]

        # Configure according to the dataset under study.
        folder_paths_and_features = self.select_dataset(self.feature_osn_name_list,
                                                        self.target_osn_name)
        self.uniform_folder = folder_paths_and_features[0]
        self.design_folder = folder_paths_and_features[1]
        self.results_folder = folder_paths_and_features[2]
        self.branching_feature_dict = folder_paths_and_features[3]
        self.usergraph_feature_dict = folder_paths_and_features[4]
        self.temporal_feature_dict = folder_paths_and_features[5]

        self.branching_feature_names_list_dict = dict()
        self.usergraph_feature_names_list_dict = dict()
        self.temporal_feature_names_list_dict = dict()

        self.number_of_branching_features_dict = dict()
        self.number_of_usergraph_features_dict = dict()
        self.number_of_temporal_features_dict = dict()

        self.within_dataset_user_anonymizer_dict = dict()

        for feature_osn_name in self.feature_osn_name_list:
            self.branching_feature_names_list_dict[feature_osn_name] = sorted(self.branching_feature_dict[feature_osn_name])
            self.usergraph_feature_names_list_dict[feature_osn_name] = sorted(self.usergraph_feature_dict[feature_osn_name])
            self.temporal_feature_names_list_dict[feature_osn_name] = sorted(self.temporal_feature_dict[feature_osn_name])

            self.number_of_branching_features_dict[feature_osn_name] = len(self.branching_feature_names_list_dict[feature_osn_name])
            self.number_of_usergraph_features_dict[feature_osn_name] = len(self.usergraph_feature_names_list_dict[feature_osn_name])
            self.number_of_temporal_features_dict[feature_osn_name] = len(self.temporal_feature_names_list_dict[feature_osn_name])

            # self.within_dataset_user_anonymizer_dict[feature_osn_name] = load_dataset_user_anonymizer(self.uniform_folder + "/datasetwide/" + feature_osn_name + "_user_anonymizer" + ".pkl")
            self.within_dataset_user_anonymizer_dict[feature_osn_name] = load_dataset_user_anonymizer(self.uniform_folder + "/datasetwide/user_anonymizer" + ".pkl")

        print("Focusing on OSN timeline: ", self.osn_name_focus)
        print("Targeting OSN: ", self.target_osn_name)
        print("Target variables: ", self.target_name_list)

        for feature_osn_name in self.feature_osn_name_list:
            print("Features from OSN: ", feature_osn_name)

            print("Number of branching features: ", self.number_of_branching_features_dict[feature_osn_name])
            print("Number of user-graph features: ", self.number_of_usergraph_features_dict[feature_osn_name])
            print("Number of temporal features: ", self.number_of_temporal_features_dict[feature_osn_name])

            print("Number of distinct users in dataset: ", len(self.within_dataset_user_anonymizer_dict[feature_osn_name]))

            print("")

        self.h5_stores_and_keys = None  # To be filled when needed.

        # Cascade lifetime functions.
        self.get_valid_k_list = cascade_lifetime.get_valid_k_list
        self.get_dataset_size = cascade_lifetime.get_dataset_size
        self.get_dataframe_row = cascade_lifetime.get_dataframe_row

        # Ranking functions.
        self.initialize_k_evaluation_measures = ranking.initialize_k_evaluation_measures
        self.update_k_evaluation_measures = ranking.update_k_evaluation_measures
        self.store_k_evaluation_measures = ranking.store_k_evaluation_measures
        self.form_ground_truth = ranking.form_ground_truth
        self.initialize_evaluation_measure_arrays = ranking.initialize_evaluation_measure_arrays
        self.folding = ranking.folding
        self.learning_module = ranking.learning_module
        self.update_evaluation_measure_arrays = ranking.update_evaluation_measure_arrays
        self.average_evaluation_measure_arrays_across_trials = ranking.average_evaluation_measure_arrays_across_trials
        self.is_k_valid = ranking.is_k_valid

        # Define feature matrix columns.
        feature_matrix_column_names_tuple = self.define_feature_matrix_column_names()
        self.feature_matrix_column_names = feature_matrix_column_names_tuple[0]
        self.total_number_of_features = len(self.feature_matrix_column_names)

        if self.add_branching_features:
            self.fill_X_handcrafted_k_function = concatenate_features.fill_X_handcrafted_k_actual
            self.fill_X_branching_fold_function = concatenate_features.fill_X_handcrafted_fold_actual
        else:
            self.fill_X_handcrafted_k_function = concatenate_features.fill_X_handcrafted_k_dummy
            self.fill_X_branching_fold_function = concatenate_features.fill_X_handcrafted_fold_dummy

        if self.add_usergraph_features:
            self.fill_X_handcrafted_k_function = concatenate_features.fill_X_handcrafted_k_actual
            self.fill_X_usergraph_fold_function = concatenate_features.fill_X_handcrafted_fold_actual
        else:
            self.fill_X_handcrafted_k_function = concatenate_features.fill_X_handcrafted_k_dummy
            self.fill_X_usergraph_fold_function = concatenate_features.fill_X_handcrafted_fold_dummy

        if self.add_temporal_features:
            self.fill_X_handcrafted_k_function = concatenate_features.fill_X_handcrafted_k_actual
            self.fill_X_temporal_fold_function = concatenate_features.fill_X_handcrafted_fold_actual
        else:
            self.fill_X_handcrafted_k_function = concatenate_features.fill_X_handcrafted_k_dummy
            self.fill_X_temporal_fold_function = concatenate_features.fill_X_handcrafted_fold_dummy

        if "baseline" in self.experiment_construction_dict.keys():
            if self.experiment_construction_dict["baseline"] == "comments":
                self.fill_X_handcrafted_k_function = concatenate_features.fill_X_handcrafted_k_actual
                self.fill_X_branching_fold_function = concatenate_features.fill_X_handcrafted_fold_actual
            elif self.experiment_construction_dict["baseline"] == "users":
                self.fill_X_handcrafted_k_function = concatenate_features.fill_X_handcrafted_k_actual
                self.fill_X_usergraph_fold_function = concatenate_features.fill_X_handcrafted_fold_actual
            elif self.experiment_construction_dict["baseline"] == "comments_users":
                self.fill_X_handcrafted_k_function = concatenate_features.fill_X_handcrafted_k_actual
                self.fill_X_branching_fold_function = concatenate_features.fill_X_handcrafted_fold_actual

                self.fill_X_handcrafted_k_function = concatenate_features.fill_X_handcrafted_k_actual
                self.fill_X_usergraph_fold_function = concatenate_features.fill_X_handcrafted_fold_actual
            elif self.experiment_construction_dict["baseline"] == "simple graph":
                self.fill_X_handcrafted_k_function = concatenate_features.fill_X_handcrafted_k_actual
                self.fill_X_branching_fold_function = concatenate_features.fill_X_handcrafted_fold_actual

                self.fill_X_handcrafted_k_function = concatenate_features.fill_X_handcrafted_k_actual
                self.fill_X_usergraph_fold_function = concatenate_features.fill_X_handcrafted_fold_actual

    def get_h5_stores_and_keys(self,
                               keep_dataset):
        uniform_folder = self.uniform_folder

        # This is a list of all the .h5 files as produced after preprocessing.
        h5_store_file_name_list = os.listdir(uniform_folder)
        h5_store_file_name_list = [h5_store_file_name for h5_store_file_name in sorted(h5_store_file_name_list) if not h5_store_file_name[-1] == "~"]

        timestamp_h5_store_file_name_list = [name for name in h5_store_file_name_list if "timestamp" in name]
        handcrafted_features_h5_store_file_name_list = [name for name in h5_store_file_name_list if "handcrafted" in name]
        author_features_h5_store_file_name_list = [name for name in h5_store_file_name_list if "author" in name]
        bipartite_features_h5_store_file_name_list = [name for name in h5_store_file_name_list if "bipartite" in name]

        timestamp_h5_store_file_path_list = [uniform_folder + "/" + h5_store_file_name for h5_store_file_name in timestamp_h5_store_file_name_list]
        handcrafted_features_h5_store_file_path_list = [uniform_folder + "/" + h5_store_file_name for h5_store_file_name in handcrafted_features_h5_store_file_name_list]
        author_features_h5_store_file_path_list = [uniform_folder + "/" + h5_store_file_name for h5_store_file_name in author_features_h5_store_file_name_list]
        bipartite_features_h5_store_file_path_list = [uniform_folder + "/" + h5_store_file_name for h5_store_file_name in bipartite_features_h5_store_file_name_list]

        file_path_list_zip = zip(timestamp_h5_store_file_path_list,
                                 handcrafted_features_h5_store_file_path_list,
                                 author_features_h5_store_file_path_list,
                                 bipartite_features_h5_store_file_path_list)

        h5_stores_and_keys = list()
        for file_paths in file_path_list_zip:
            timestamp_h5_store_file = h5_open(file_paths[0])
            handcrafted_features_h5_store_file = h5_open(file_paths[1])
            author_features_h5_store_file = h5_open(file_paths[2])
            bipartite_features_h5_store_file = h5_open(file_paths[3])

            keys_dict = dict()
            for osn_name in keep_dataset:
                keys_dict[osn_name] = sorted((key for key in timestamp_h5_store_file.keys() if osn_name in key))

            h5_stores_and_keys.append(((timestamp_h5_store_file,
                                        handcrafted_features_h5_store_file,
                                        author_features_h5_store_file,
                                        bipartite_features_h5_store_file),
                                       keys_dict))

        return h5_stores_and_keys

    def do_experiment(self):

        h5_stores_and_keys,\
        k_list = self.get_valid_k_list_wrapper()

        self.k_sweep(h5_stores_and_keys,
                     k_list)

    def get_valid_k_list_wrapper(self):

        k_list_path = self.uniform_folder + "/k_list/focus_" + self.osn_name_focus + ".txt"

        if os.path.exists(k_list_path):
            k_list = self.load_valid_k_list(k_list_path)
        else:
            self.h5_stores_and_keys = self.get_h5_stores_and_keys(keep_dataset=self.feature_osn_name_list)

            k_list = self.get_valid_k_list(h5_stores_and_keys=self.h5_stores_and_keys,
                                           osn_focus=self.osn_name_focus)

            self.store_valid_k_list(k_list_path,
                                    k_list)

        return self.h5_stores_and_keys, k_list

    def store_valid_k_list(self, k_list_path, k_list):
        with open(k_list_path, "w") as fp:
            for k in k_list:
                row = repr(k) + "\n"
                fp.write(row)

    def load_valid_k_list(self, k_list_path):
        k_list = list()

        with open(k_list_path, "r") as fp:
            for row in fp:
                row_stripped = row.strip()
                if row_stripped == "":
                    continue
                k_list.append(row_stripped)
        # return k_list[:1]
        return k_list

    def k_sweep(self,
                h5_stores_and_keys,
                k_list):
        number_of_discrete_k_values = len(k_list)

        target_to_evaluation_measures = dict()
        for target_name in self.target_name_list:
            k_evaluation_measures = self.initialize_k_evaluation_measures(number_of_k=number_of_discrete_k_values,
                                                                          number_of_folds=self.number_of_folds,
                                                                          number_of_features=self.total_number_of_features)

            target_to_evaluation_measures[target_name] = k_evaluation_measures

        dataset_full, dataset_size = self.get_dataset_full()

        X_k_min_dict = dict()
        X_t_next_dict = dict()
        for osn_name in self.feature_osn_name_list:
            X_k_min_dict[osn_name] = np.zeros(dataset_size, dtype=int)
            X_t_next_dict[osn_name] = np.zeros(dataset_size, dtype=float)

        for k_index, k in enumerate(k_list):
            print("k: ", k, "k_i: ", k_index + 1, " out of ", number_of_discrete_k_values)

            # We form the feature matrix and target vector for the dataset.
            print("Number of discussions in experiment: ", dataset_size)

            target_name_to_evaluation_measure_arrays,\
            X_k_min_array,\
            X_t_next_array = self.experiment(dataset_full,
                                             h5_stores_and_keys,
                                             k,
                                             X_k_min_dict,
                                             X_t_next_dict,
                                             dataset_size,
                                             target_to_evaluation_measures)

            for target_name in self.target_name_list:
                print(target_name)
                self.update_k_evaluation_measures(target_to_evaluation_measures[target_name],
                                                  k_index,
                                                  target_name_to_evaluation_measure_arrays[target_name])

        for target_name in self.target_name_list:
            store_path = self.results_folder + "/" +\
                         "target_" + self.target_osn_name + "_" +\
                         target_name + "_" +\
                         self.experiment_construction_type

            self.store_k_evaluation_measures(store_path,
                                             k_list,
                                             target_to_evaluation_measures[target_name],
                                             self.feature_matrix_column_names)

    def experiment(self,
                   dataset_full,
                   h5_stores_and_keys,
                   k,
                   X_k_min_dict,
                   X_t_next_dict,
                   dataset_size,
                   target_to_evaluation_measures):
        dataset_k,\
        X_k_min_array,\
        X_t_next_array = self.get_dataset_k(dataset_size,
                                            h5_stores_and_keys,
                                            k,
                                            X_k_min_dict,
                                            X_t_next_dict)

        # Form ground truth for given experiment parameters k, R
        target_name_to_y = dict()
        target_name_to_evaluation_measure_arrays = dict()
        for target_name in self.target_name_list:
            y, y_baseline = self.form_ground_truth(dataset_full[self.target_osn_name]["y_raw"][target_name])
            target_name_to_y[target_name] = y

            target_name_to_evaluation_measure_arrays[target_name] = self.initialize_evaluation_measure_arrays(number_of_folds=self.number_of_folds,
                                                                                                              number_of_features=self.total_number_of_features)

        k_fold = self.folding(target_name_to_y[self.target_name_list[0]], n_folds=self.number_of_folds)
        for fold_index, train_test in enumerate(k_fold):
            X_train, X_test, y_train_dict, y_test_dict = self.separate_train_test(dataset_full,
                                                                                  dataset_k,
                                                                                  target_name_to_y,
                                                                                  train_test,
                                                                                  fold_index,
                                                                                  k)

            for target_name in self.target_name_list:
                model_file_path = self.uniform_folder + "/models/" + self.osn_name_focus + "/" + self.experiment_construction_type + "target_" + self.target_osn_name + "_" + target_name + repr(k) + "_fold_" + str(fold_index) + "_factors.h5"

                baseline = None
                if "baseline" in self.experiment_construction_dict.keys():
                    if "mean" == self.experiment_construction_dict["baseline"]:
                        baseline = "mean"
                    elif "median" == self.experiment_construction_dict["baseline"]:
                        baseline = "median"

                evaluation_tuple = self.learning_module(model_file_path,
                                                        X_train,
                                                        X_test,
                                                        y_train_dict[target_name],
                                                        y_test_dict[target_name],
                                                        train_test,
                                                        baseline)

                self.update_evaluation_measure_arrays(target_name_to_evaluation_measure_arrays[target_name],
                                                      fold_index,
                                                      evaluation_tuple)

        return target_name_to_evaluation_measure_arrays, X_k_min_array, X_t_next_array

    def parse_experiment_construction(self, experiment_construction_dict):

        experiment_construction_type = ""

        for osn_name in self.feature_osn_name_list:
            experiment_construction_type += osn_name + "_"

        if "baseline" in experiment_construction_dict.keys():
            experiment_construction_type += "baseline_" + experiment_construction_dict["baseline"]
            add_branching_features = False
            add_usergraph_features = False
            add_temporal_features = False

            return experiment_construction_type,\
                   add_branching_features,\
                   add_usergraph_features,\
                   add_temporal_features

        if experiment_construction_dict["add_branching_features"]:
            if "best" in experiment_construction_dict.keys():
                experiment_construction_type += "best"
            experiment_construction_type += "branching_"
            add_branching_features = True
        else:
            experiment_construction_type += ""
            add_branching_features = False

        if experiment_construction_dict["add_usergraph_features"]:
            if "best" in experiment_construction_dict.keys():
                experiment_construction_type += "best"
            experiment_construction_type += "usergraph_"
            add_usergraph_features = True
        else:
            experiment_construction_type += ""
            add_usergraph_features = False

        if experiment_construction_dict["add_temporal_features"]:
            experiment_construction_type += "temporal_"
            add_temporal_features = True
        else:
            experiment_construction_type += ""
            add_temporal_features = False

        return experiment_construction_type,\
               add_branching_features,\
               add_usergraph_features,\
               add_temporal_features

    def parse_joint_experiment_construction(self, joint_experiment_construction_dict):
        if joint_experiment_construction_dict["cofactorize"]:
            self.experiment_construction_type += "cofactorize_"
            do_cofactorization = True
        else:
            self.experiment_construction_type += "concatenate_"
            do_cofactorization = False

        return do_cofactorization

    def define_feature_matrix_column_names(self):
        feature_matrix_column_names = list()

        for feature_osn_name in self.feature_osn_name_list:
            if self.add_branching_features:
                feature_names = self.branching_feature_names_list_dict[feature_osn_name]
                feature_matrix_column_names.extend(feature_osn_name + "_" + feature for feature in feature_names)

            if self.add_usergraph_features:
                feature_names = self.usergraph_feature_names_list_dict[feature_osn_name]
                feature_matrix_column_names.extend(feature_osn_name + "_" + feature for feature in feature_names)

            if self.add_temporal_features:
                feature_names = self.temporal_feature_names_list_dict[feature_osn_name]
                feature_matrix_column_names.extend(feature_osn_name + "_" + feature for feature in feature_names)

        if "baseline" in self.experiment_construction_dict.keys():
            if "comments" == self.experiment_construction_dict["baseline"]:
                for feature_osn_name in self.feature_osn_name_list:
                    feature_names = self.branching_feature_names_list_dict[feature_osn_name]
                    feature_matrix_column_names.extend(feature_osn_name + "_" + feature for feature in feature_names)

            elif "users" == self.experiment_construction_dict["baseline"]:
                for feature_osn_name in self.feature_osn_name_list:
                    feature_names = self.usergraph_feature_names_list_dict[feature_osn_name]
                    feature_matrix_column_names.extend(feature_osn_name + "_" + feature for feature in feature_names)

            elif "comments_users" == self.experiment_construction_dict["baseline"]:
                for feature_osn_name in self.feature_osn_name_list:
                    feature_names = self.branching_feature_names_list_dict[feature_osn_name]
                    feature_matrix_column_names.extend(feature_osn_name + "_" + feature for feature in feature_names)

                    feature_names = self.usergraph_feature_names_list_dict[feature_osn_name]
                    feature_matrix_column_names.extend(feature_osn_name + "_" + feature for feature in feature_names)

            elif "simple graph" == self.experiment_construction_dict["baseline"]:
                for feature_osn_name in self.feature_osn_name_list:
                    feature_names = self.branching_feature_names_list_dict[feature_osn_name]
                    feature_matrix_column_names.extend(feature_osn_name + "_" + feature for feature in feature_names)

                    feature_names = self.usergraph_feature_names_list_dict[feature_osn_name]
                    feature_matrix_column_names.extend(feature_osn_name + "_" + feature for feature in feature_names)

        return feature_matrix_column_names

    def select_dataset(self,
                       feature_osn_name_list,
                       target_osn_name):
        data_folder = self.data_folder

        add_branching_features = self.add_branching_features
        add_usergraph_features = self.add_usergraph_features
        add_temporal_features = self.add_temporal_features

        uniform_folder = "/home/georgerizos/Documents/LocalStorage/memory/" + data_folder + "/uniform"
        design_folder = "/home/georgerizos/Documents/LocalStorage/memory/" + data_folder + "/design"
        results_folder = "/home/georgerizos/Documents/LocalStorage/memory/" + data_folder + "/results/" + self.osn_name_focus + "_focus"

        best_features_folder = "/home/georgerizos/Documents/LocalStorage/memory/" + data_folder + "/best_features"

        branching_feature_dict = dict()
        usergraph_feature_dict = dict()
        temporal_feature_dict = dict()
        for feature_osn_name in feature_osn_name_list:
            if add_branching_features:
                branching_feature_dict[feature_osn_name] = get_branching_feature_names(osn_name=feature_osn_name)
            else:
                branching_feature_dict[feature_osn_name] = set()

            if add_usergraph_features:
                usergraph_feature_dict[feature_osn_name] = get_usergraph_feature_names(osn_name=feature_osn_name)
            else:
                usergraph_feature_dict[feature_osn_name] = set()

            if add_temporal_features:
                temporal_feature_dict[feature_osn_name] = get_temporal_feature_names(osn_name=feature_osn_name)
            else:
                temporal_feature_dict[feature_osn_name] = set()

        if "baseline" in self.experiment_construction_dict.keys():
            if "comments" == self.experiment_construction_dict["baseline"]:
                for feature_osn_name in feature_osn_name_list:
                    branching_feature_dict[feature_osn_name] = set()
                    branching_feature_dict[feature_osn_name].add("basic_comment_count")
            elif "users" == self.experiment_construction_dict["baseline"]:
                for feature_osn_name in feature_osn_name_list:
                    usergraph_feature_dict[feature_osn_name] = set()
                    usergraph_feature_dict[feature_osn_name].add("user_graph_user_count")
            elif "comments_users" == self.experiment_construction_dict["baseline"]:
                for feature_osn_name in feature_osn_name_list:
                    branching_feature_dict[feature_osn_name] = set()
                    branching_feature_dict[feature_osn_name].add("basic_comment_count")
                    usergraph_feature_dict[feature_osn_name] = set()
                    usergraph_feature_dict[feature_osn_name].add("user_graph_user_count")
            elif "simple graph" == self.experiment_construction_dict["baseline"]:
                for feature_osn_name in feature_osn_name_list:
                    branching_feature_dict[feature_osn_name] = set()
                    branching_feature_dict[feature_osn_name].add("basic_comment_count")
                    branching_feature_dict[feature_osn_name].add("basic_max_depth")
                    branching_feature_dict[feature_osn_name].add("basic_max_width")
                    branching_feature_dict[feature_osn_name].add("basic_ave_width")
                    usergraph_feature_dict[feature_osn_name] = set()
                    usergraph_feature_dict[feature_osn_name].add("user_graph_user_count")

        return uniform_folder,\
               design_folder,\
               results_folder,\
               branching_feature_dict,\
               usergraph_feature_dict,\
               temporal_feature_dict

    def get_dataset_full(self):
        dataset_full_path = self.uniform_folder + "/dataset_full/dataset_full.h5"

        if os.path.exists(dataset_full_path):
            dataset_full,\
            index = self.load_dataset_full(dataset_full_path)

            dataset_size = dataset_full[self.feature_osn_name_list[0]]["X_branching"].shape[0]
        else:
            if self.h5_stores_and_keys is None:
                self.h5_stores_and_keys = self.get_h5_stores_and_keys(keep_dataset=self.feature_osn_name_list)

            dataset_size = self.get_dataset_size(self.h5_stores_and_keys,
                                                 self.target_osn_name)

            dataset_full,\
            index = self.form_dataset_full(dataset_size,
                                           self.h5_stores_and_keys)

            self.store_dataset_full(dataset_full_path, dataset_full, index)
        return dataset_full, dataset_size

    def load_dataset_full(self,
                          dataset_full_path):
        dataset_full = dict()
        dataset_full[self.target_osn_name] = dict()

        index = dict()

        h5_store = h5_open(dataset_full_path)

        for osn_name in self.feature_osn_name_list:
            df = h5load_from(h5_store, "/data/" + osn_name + "/X_branching")[self.branching_feature_names_list_dict[osn_name]]
            # index[osn_name] = df.index
            dataset_full[osn_name]["X_branching"] = df.values
            dataset_full[osn_name]["X_usergraph"] = h5load_from(h5_store, "/data/" + osn_name + "/X_usergraph")[self.usergraph_feature_names_list_dict[osn_name]].values
            dataset_full[osn_name]["X_temporal"] = h5load_from(h5_store, "/data/" + osn_name + "/X_temporal")[self.temporal_feature_names_list_dict[osn_name]].values

        data_frame = h5load_from(h5_store, "/data/" + self.target_osn_name + "/y_raw")
        dataset_full[self.target_osn_name]["y_raw"] = dict()
        for target_name in self.target_name_list:
            dataset_full[self.target_osn_name]["y_raw"][target_name] = data_frame[target_name].values

        h5_close(h5_store)

        return dataset_full, index

    def store_dataset_full(self,
                           dataset_full_path,
                           dataset_full,
                           index):
        h5_store = h5_open(dataset_full_path)

        for osn_name in dataset_full.keys():
            h5store_at(h5_store, osn_name + "/X_branching", pd.DataFrame(dataset_full[osn_name]["X_branching"],
                                                                         columns=self.branching_feature_names_list_dict[osn_name]))
            h5store_at(h5_store, osn_name + "/X_usergraph", pd.DataFrame(dataset_full[osn_name]["X_usergraph"],
                                                                         columns=self.usergraph_feature_names_list_dict[osn_name]))
            h5store_at(h5_store, osn_name + "/X_temporal", pd.DataFrame(dataset_full[osn_name]["X_temporal"],
                                                                        columns=self.temporal_feature_names_list_dict[osn_name]))

            y_raw_dict = dict()
            for target_name in dataset_full[osn_name]["y_raw"].keys():
                y_raw_dict[target_name] = dataset_full[osn_name]["y_raw"][target_name]

            h5store_at(h5_store, osn_name + "/y_raw", pd.DataFrame(y_raw_dict))

        h5_close(h5_store)

    def form_dataset_full(self,
                         dataset_size,
                         h5_stores_and_keys):
        osn_to_targetlist = dict()
        if self.osn_name_focus == "reddit":
            osn_to_targetlist["reddit"] = ["comments",
                                           "users",
                                           "score",
                                           "score_div",
                                           "score_wilson",
                                           "controversiality",
                                           "controversiality_wilson"]
        if self.osn_name_focus == "slashdot":
            osn_to_targetlist["slashdot"] = ["comments",
                                             "users"]

        # Initialize full feature arrays.
        dataset_full = dict()
        index = dict()

        for osn_name in osn_to_targetlist.keys():
            dataset_full[osn_name] = dict()
            index[osn_name] = list()

            X_branching_full = np.empty((dataset_size,
                                         self.number_of_branching_features_dict[osn_name]),
                                        dtype=np.float64)
            dataset_full[osn_name]["X_branching"] = X_branching_full

            X_usergraph_full = np.empty((dataset_size,
                                         self.number_of_usergraph_features_dict[osn_name]),
                                        dtype=np.float64)
            dataset_full[osn_name]["X_usergraph"] = X_usergraph_full

            X_temporal_full = np.empty((dataset_size,
                                        self.number_of_temporal_features_dict[osn_name]),
                                       dtype=np.float64)
            dataset_full[osn_name]["X_temporal"] = X_temporal_full

            dataset_full[osn_name]["y_raw"] = dict()
            for target_name in osn_to_targetlist[osn_name]:
                dataset_full[osn_name]["y_raw"][target_name] = np.empty(dataset_size, dtype=np.float64)

            # Fill full feature arrays.
            offset = 0
            for h5_store_files, h5_keys in h5_stores_and_keys:
                index[osn_name].extend(h5_keys)
                self.fill_X_handcrafted_full_and_y_raw(dataset_full, h5_store_files, h5_keys[osn_name], offset, osn_name, osn_to_targetlist[osn_name])

                offset += len(h5_keys[osn_name])

        return dataset_full, index

    def fill_X_handcrafted_full_and_y_raw(self,
                                          dataset_full,
                                          h5_store_files,
                                          h5_keys,
                                          offset,
                                          osn_name,
                                          target_list):
        for d, h5_key in enumerate(h5_keys):
            handcrafted_features_data_frame = h5load_from(h5_store_files[1], h5_key)

            kth_row = get_kth_row(handcrafted_features_data_frame,
                                  -1,
                                  self.branching_feature_names_list_dict[osn_name])
            dataset_full[osn_name]["X_branching"][offset + d, :self.number_of_branching_features_dict[osn_name]] = kth_row

            kth_row = get_kth_row(handcrafted_features_data_frame,
                                  -1,
                                  self.usergraph_feature_names_list_dict[osn_name])
            dataset_full[osn_name]["X_usergraph"][offset + d, :self.number_of_usergraph_features_dict[osn_name]] = kth_row

            kth_row = get_kth_row(handcrafted_features_data_frame,
                                  -1,
                                  self.temporal_feature_names_list_dict[osn_name])
            dataset_full[osn_name]["X_temporal"][offset + d, :self.number_of_temporal_features_dict[osn_name]] = kth_row

            for target_name in target_list:
                dataset_full[osn_name]["y_raw"][target_name][offset + d] = get_target_value(handcrafted_features_data_frame,
                                                                                            target_name)

    def get_dataset_k(self,
                      dataset_size,
                      h5_stores_and_keys,
                      k,
                      X_k_min_dict,
                      X_t_next_dict):
        try:
            dataset_k_path = self.uniform_folder + "/dataset_k/" + self.osn_name_focus + "_lifetime_" + k + "_dataset_k.h5"
        except TypeError:
            dataset_k_path = self.uniform_folder + "/dataset_k/" + self.osn_name_focus + "_lifetime_" + repr(k) + "_dataset_k.h5"

        if os.path.exists(dataset_k_path):
            dataset_k,\
            X_k_min_dict,\
            X_t_next_dict,\
            index = self.load_dataset_k(dataset_k_path)
        else:
            if self.h5_stores_and_keys is None:
                self.h5_stores_and_keys = self.get_h5_stores_and_keys(self.feature_osn_name_list)

            dataset_k,\
            X_k_min_dict,\
            X_t_next_dict,\
            index = self.form_dataset_k(dataset_size,
                                        self.h5_stores_and_keys,
                                        float(k),
                                        X_k_min_dict,
                                        X_t_next_dict)

            self.store_dataset_k(dataset_k_path,
                                 dataset_k,
                                 X_k_min_dict,
                                 X_t_next_dict,
                                 index)

        return dataset_k, X_k_min_dict, X_t_next_dict

    def load_dataset_k(self,
                       dataset_k_path):
        dataset_k = dict()
        X_k_min_dict = dict()
        X_t_next_dict = dict()

        index = dict()

        h5_store = h5_open(dataset_k_path)

        for osn_name in self.feature_osn_name_list:
            dataset_k[osn_name] = dict()

            df = h5load_from(h5_store, "/data/" + osn_name + "/X_branching")[self.branching_feature_names_list_dict[osn_name]]
            # index[osn_name] = df.index

            dataset_k[osn_name]["X_branching"] = df.values
            dataset_k[osn_name]["X_usergraph"] = h5load_from(h5_store, "/data/" + osn_name + "/X_usergraph")[self.usergraph_feature_names_list_dict[osn_name]].values
            dataset_k[osn_name]["X_temporal"] = h5load_from(h5_store, "/data/" + osn_name + "/X_temporal")[self.temporal_feature_names_list_dict[osn_name]].values

            data_frame = h5load_from(h5_store, "/data/" + osn_name + "/utility_arrays")
            X_k_min_dict[osn_name] = data_frame["X_k_min_array"].values
            X_t_next_dict[osn_name] = data_frame["X_t_next_array"].values

        h5_close(h5_store)

        return dataset_k, X_k_min_dict, X_t_next_dict, index

    def store_dataset_k(self,
                        dataset_k_path,
                        dataset_k,
                        X_k_min_dict,
                        X_t_next_dict,
                        index):

        h5_store = h5_open(dataset_k_path)

        for osn_name in dataset_k.keys():
            h5store_at(h5_store, osn_name + "/X_branching", pd.DataFrame(dataset_k[osn_name]["X_branching"],
                                                                         columns=self.branching_feature_names_list_dict[osn_name]))
            h5store_at(h5_store, osn_name + "/X_usergraph", pd.DataFrame(dataset_k[osn_name]["X_usergraph"],
                                                                         columns=self.usergraph_feature_names_list_dict[osn_name]))
            h5store_at(h5_store, osn_name + "/X_temporal", pd.DataFrame(dataset_k[osn_name]["X_temporal"],
                                                                        columns=self.temporal_feature_names_list_dict[osn_name]))

            utility_arrays = dict()
            utility_arrays["X_k_min_array"] = X_k_min_dict[osn_name]
            utility_arrays["X_t_next_array"] = X_t_next_dict[osn_name]

            h5store_at(h5_store, osn_name + "/utility_arrays", pd.DataFrame(utility_arrays))

        h5_close(h5_store)

    def form_dataset_k(self,
                       dataset_size,
                       h5_stores_and_keys,
                       k,
                       X_k_min_dict,
                       X_t_next_dict):
        all_feature_osn_names = self.feature_osn_name_list

        dataset_k = dict()
        index = dict()

        if True:
            for osn_index, osn_name in enumerate(all_feature_osn_names):
                dataset_k[osn_name] = dict()
                index[osn_name] = list()

                if self.add_branching_features:
                    X_branching_k = np.empty((dataset_size,
                                              self.number_of_branching_features_dict[osn_name]),
                                             dtype=np.float64)
                    dataset_k[osn_name]["X_branching"] = X_branching_k

                if self.add_usergraph_features:
                    X_usergraph_k = np.empty((dataset_size,
                                              self.number_of_usergraph_features_dict[osn_name]),
                                             dtype=np.float64)
                    dataset_k[osn_name]["X_usergraph"] = X_usergraph_k

                if self.add_temporal_features:
                    X_temporal_k = np.empty((dataset_size,
                                             self.number_of_temporal_features_dict[osn_name]),
                                            dtype=np.float64)
                    dataset_k[osn_name]["X_temporal"] = X_temporal_k

                # Fill full feature arrays.
                offset = 0
                for h5_store_files, h5_keys in h5_stores_and_keys:
                    index[osn_name].extend(h5_keys)

                    self.calculate_k_based_on_lifetime(dataset_k, h5_store_files, h5_keys, offset, k, X_k_min_dict, X_t_next_dict, osn_name)

                    self.fill_X_handcrafted_k(dataset_k, h5_store_files, h5_keys[osn_name], offset, k, X_k_min_dict, X_t_next_dict, osn_name)
                    offset += len(h5_keys[osn_name])

        return dataset_k, X_k_min_dict, X_t_next_dict, index

    def calculate_k_based_on_lifetime(self,
                                      dataset_k,
                                      h5_store_files,
                                      h5_keys,
                                      offset,
                                      k,
                                      X_k_min_dict,
                                      X_t_next_dict,
                                      osn_name):
        number_of_keys = len(h5_keys[osn_name])

        for d in range(number_of_keys):
            timestamps_data_frame = h5load_from(h5_store_files[0], h5_keys[osn_name][d])

            if np.isnan(X_t_next_dict[osn_name][offset + d]):
                continue

            observed_comments,\
            next_lifetime = cascade_lifetime.get_k_based_on_lifetime(timestamps_data_frame,
                                                                     k,
                                                                     min_k=X_k_min_dict[osn_name][offset + d],
                                                                     max_k=-1)

            X_k_min_dict[osn_name][offset + d] = observed_comments
            X_t_next_dict[osn_name][offset + d] = next_lifetime

    def fill_X_handcrafted_k(self,
                             dataset_k,
                             h5_store_files,
                             h5_keys,
                             offset,
                             k,
                             X_k_min_dict,
                             X_t_next_dict,
                             osn_name):

        self.fill_X_handcrafted_k_function(dataset_k,
                                           h5_store_files,
                                           h5_keys,
                                           offset,
                                           k,
                                           X_k_min_dict,
                                           X_t_next_dict,
                                           self.branching_feature_names_list_dict[osn_name],
                                           self.usergraph_feature_names_list_dict[osn_name],
                                           self.temporal_feature_names_list_dict[osn_name],
                                           osn_name)

    def separate_train_test(self,
                            dataset_full,
                            dataset_k,
                            target_name_to_y,
                            train_test,
                            fold_index,
                            k):
        train, test = train_test

        X_train = np.empty((len(train),
                            self.total_number_of_features),
                           dtype=np.float64)
        X_test = np.empty((len(test),
                           self.total_number_of_features),
                          dtype=np.float64)

        y_train_dict = dict()
        y_test_dict = dict()
        for target_name in self.target_name_list:
            y_train_dict[target_name] = target_name_to_y[target_name][train]
            y_test_dict[target_name] = target_name_to_y[target_name][test]

        for osn_index, osn_name in enumerate(self.feature_osn_name_list):
            self.fill_X_handcrafted_fold(X_train, train, dataset_full, dataset_k, osn_index, osn_name)
            self.fill_X_handcrafted_fold(X_test, test, dataset_full, dataset_k, osn_index, osn_name)

        return X_train, X_test, y_train_dict, y_test_dict

    def fill_X_handcrafted_fold(self,
                                X,
                                indices,
                                dataset_full,
                                dataset_k,
                                osn_index,
                                osn_name):
        osn_offset = 0
        if osn_index > 0:
            for previous_osn_index in range(osn_index):
                previous_osn_name = self.feature_osn_name_list[previous_osn_index]
                osn_offset += self.number_of_branching_features_dict[previous_osn_name] + self.number_of_usergraph_features_dict[previous_osn_name] + self.number_of_temporal_features_dict[previous_osn_name] + self.number_of_author_features_dict[previous_osn_name] + self.bipartite_graph_features_dimensionality

        min_column_index = osn_offset
        max_column_index = osn_offset + self.number_of_branching_features_dict[osn_name]
        self.fill_X_branching_fold_function(X,
                                            indices,
                                            dataset_full,
                                            dataset_k,
                                            osn_name,
                                            min_column_index,
                                            max_column_index,
                                            "X_branching")

        min_column_index = osn_offset + self.number_of_branching_features_dict[osn_name]
        max_column_index = osn_offset + self.number_of_branching_features_dict[osn_name] + self.number_of_usergraph_features_dict[osn_name]
        self.fill_X_usergraph_fold_function(X,
                                            indices,
                                            dataset_full,
                                            dataset_k,
                                            osn_name,
                                            min_column_index,
                                            max_column_index,
                                            "X_usergraph")

        min_column_index = osn_offset + self.number_of_branching_features_dict[osn_name] + self.number_of_usergraph_features_dict[osn_name]
        max_column_index = osn_offset + self.number_of_branching_features_dict[osn_name] + self.number_of_usergraph_features_dict[osn_name] + self.number_of_temporal_features_dict[osn_name]
        self.fill_X_temporal_fold_function(X,
                                           indices,
                                           dataset_full,
                                           dataset_k,
                                           osn_name,
                                           min_column_index,
                                           max_column_index,
                                           "X_temporal")
