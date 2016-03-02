__author__ = 'Georgios Rizos (georgerizos@iti.gr)'

import os

import numpy as np

from news_popularity_prediction.learning.ranking import initialize_k_evaluation_measures, update_k_evaluation_measures,\
    store_k_evaluation_measures, form_ground_truth, initialize_evaluation_measure_arrays, folding, learning_module,\
    update_evaluation_measure_arrays
from news_popularity_prediction.learning.cascade_lifetime import load_valid_k_list, load_dataset_k, load_dataset_full
from news_popularity_prediction.discussion.features import get_branching_feature_names, get_usergraph_feature_names,\
    get_temporal_feature_names
from news_popularity_prediction.discussion.datasetwide import load_dataset_user_anonymizer
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
        self.results_folder = folder_paths_and_features[1]
        self.branching_feature_dict = folder_paths_and_features[2]
        self.usergraph_feature_dict = folder_paths_and_features[3]
        self.temporal_feature_dict = folder_paths_and_features[4]

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

        # Define feature matrix columns.
        feature_matrix_column_names_tuple = self.define_feature_matrix_column_names()
        self.feature_matrix_column_names = feature_matrix_column_names_tuple
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

    def do_experiment(self):

        h5_stores_and_keys,\
        k_list = self.get_valid_k_list_wrapper()

        self.k_sweep(h5_stores_and_keys,
                     k_list)

    def get_valid_k_list_wrapper(self):

        k_list_path = self.uniform_folder + "/k_list/focus_post" + ".txt"

        if os.path.exists(k_list_path):
            k_list = load_valid_k_list(k_list_path)
        else:
            print("Comparison lifetimes not found.")
            raise RuntimeError

        return self.h5_stores_and_keys, k_list

    def k_sweep(self,
                h5_stores_and_keys,
                k_list):
        number_of_discrete_k_values = len(k_list)

        target_to_evaluation_measures = dict()
        for target_name in self.target_name_list:
            k_evaluation_measures = initialize_k_evaluation_measures(number_of_k=number_of_discrete_k_values,
                                                                     number_of_folds=self.number_of_folds,
                                                                     number_of_features=self.total_number_of_features)

            target_to_evaluation_measures[target_name] = k_evaluation_measures

        dataset_full, dataset_size = self.get_dataset_full()

        for k_index, k in enumerate(k_list):
            print("k: ", k, "k_i: ", k_index + 1, " out of ", number_of_discrete_k_values)

            # We form the feature matrix and target vector for the dataset.
            print("Number of discussions in experiment: ", dataset_size)

            target_name_to_evaluation_measure_arrays\
                = self.experiment(dataset_full,
                                  h5_stores_and_keys,
                                  k,
                                  dataset_size,
                                  target_to_evaluation_measures)

            for target_name in self.target_name_list:
                print(target_name)
                update_k_evaluation_measures(target_to_evaluation_measures[target_name],
                                             k_index,
                                             target_name_to_evaluation_measure_arrays[target_name])

        for target_name in self.target_name_list:
            store_path = self.results_folder + "/" +\
                         "target_" + self.target_osn_name + "_" +\
                         target_name + "_" +\
                         self.experiment_construction_type

            store_k_evaluation_measures(store_path,
                                        k_list,
                                        target_to_evaluation_measures[target_name],
                                        self.feature_matrix_column_names)

    def experiment(self,
                   dataset_full,
                   h5_stores_and_keys,
                   k,
                   dataset_size,
                   target_to_evaluation_measures):
        dataset_k = self.get_dataset_k(dataset_size,
                                       h5_stores_and_keys,
                                       k)

        # Form ground truth for given experiment parameters k, R
        target_name_to_y = dict()
        target_name_to_evaluation_measure_arrays = dict()
        for target_name in self.target_name_list:
            y, y_baseline = form_ground_truth(dataset_full[self.target_osn_name]["y_raw"][target_name])
            target_name_to_y[target_name] = y

            target_name_to_evaluation_measure_arrays[target_name] = initialize_evaluation_measure_arrays(number_of_folds=self.number_of_folds,
                                                                                                         number_of_features=self.total_number_of_features)

        k_fold = folding(target_name_to_y[self.target_name_list[0]], n_folds=self.number_of_folds)
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

                evaluation_tuple = learning_module(model_file_path,
                                                   X_train,
                                                   X_test,
                                                   y_train_dict[target_name],
                                                   y_test_dict[target_name],
                                                   train_test,
                                                   baseline)

                update_evaluation_measure_arrays(target_name_to_evaluation_measure_arrays[target_name],
                                                 fold_index,
                                                 evaluation_tuple)

        return target_name_to_evaluation_measure_arrays

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

        uniform_folder = data_folder + "/features"
        results_folder = data_folder + "/results/" + self.osn_name_focus

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
               results_folder,\
               branching_feature_dict,\
               usergraph_feature_dict,\
               temporal_feature_dict

    def get_dataset_full(self):
        dataset_full_path = self.uniform_folder + "/dataset_full/dataset_full.h5"

        if os.path.exists(dataset_full_path):
            dataset_full,\
            index = load_dataset_full(dataset_full_path,
                                      self.target_osn_name,
                                      self.feature_osn_name_list,
                                      self.target_name_list)

            dataset_size = dataset_full[self.feature_osn_name_list[0]]["X_branching"].shape[0]
        else:
            print("Feature matrix at t_{\infty} not found.")
            raise RuntimeError
        return dataset_full, dataset_size

    def get_dataset_k(self,
                      dataset_size,
                      h5_stores_and_keys,
                      k):
        try:
            dataset_k_path = self.uniform_folder + "/dataset_k/" + self.osn_name_focus + "_lifetime_" + k + "_dataset_k.h5"
        except TypeError:
            dataset_k_path = self.uniform_folder + "/dataset_k/" + self.osn_name_focus + "_lifetime_" + repr(k) + "_dataset_k.h5"

        if os.path.exists(dataset_k_path):
            dataset_k,\
            X_k_min_dict,\
            X_t_next_dict,\
            index = load_dataset_k(dataset_k_path,
                                   self.feature_osn_name_list)
        else:
            print("Feature matrices not calculated.")
            raise RuntimeError

        return dataset_k

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
