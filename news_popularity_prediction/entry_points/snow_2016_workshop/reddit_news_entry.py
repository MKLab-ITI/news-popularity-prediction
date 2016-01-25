__author__ = 'Georgios Rizos (georgerizos@iti.gr)'

from news_popularity_prediction.learning.single_experiment import DiscussionModellingExperiment

EXPERIMENT_CONSTRUCTION_TYPE = dict()
EXPERIMENT_CONSTRUCTION_TYPE["add_branching_features"] = False
EXPERIMENT_CONSTRUCTION_TYPE["add_usergraph_features"] = False
EXPERIMENT_CONSTRUCTION_TYPE["add_temporal_features"] = False

EXPERIMENT_CONSTRUCTION_TYPE["baseline"] = "mean"

DATA_FOLDER = "Rizos/Reddit_News"
FEATURE_OSN_NAME_LIST = ["reddit"]  # If more than one, it is a JOINT experiment.
TARGET_OSN_NAME = "reddit"  # OSN targets.
OSN_NAME_FOCUS = "reddit"  # OSN-based timestamps.
TARGET_NAME_LIST = ["comments", "users", "score_wilson", "controversiality_wilson"]
NUMBER_OF_FOLDS = 10

experiment = DiscussionModellingExperiment(experiment_construction_dict=EXPERIMENT_CONSTRUCTION_TYPE,
                                           data_folder=DATA_FOLDER,
                                           feature_osn_name_list=FEATURE_OSN_NAME_LIST,
                                           target_osn_name=TARGET_OSN_NAME,
                                           osn_name_focus=OSN_NAME_FOCUS,
                                           target_name_list=TARGET_NAME_LIST,
                                           number_of_folds=NUMBER_OF_FOLDS)

experiment.do_experiment()

EXPERIMENT_CONSTRUCTION_TYPE = dict()
EXPERIMENT_CONSTRUCTION_TYPE["add_branching_features"] = False
EXPERIMENT_CONSTRUCTION_TYPE["add_usergraph_features"] = False
EXPERIMENT_CONSTRUCTION_TYPE["add_temporal_features"] = False

EXPERIMENT_CONSTRUCTION_TYPE["baseline"] = "median"

DATA_FOLDER = "Rizos/Reddit_News"
FEATURE_OSN_NAME_LIST = ["reddit"]  # If more than one, it is a JOINT experiment.
TARGET_OSN_NAME = "reddit"  # OSN targets.
OSN_NAME_FOCUS = "reddit"  # OSN-based timestamps.
TARGET_NAME_LIST = ["comments", "users", "score_wilson", "controversiality_wilson"]
NUMBER_OF_FOLDS = 10

experiment = DiscussionModellingExperiment(experiment_construction_dict=EXPERIMENT_CONSTRUCTION_TYPE,
                                           data_folder=DATA_FOLDER,
                                           feature_osn_name_list=FEATURE_OSN_NAME_LIST,
                                           target_osn_name=TARGET_OSN_NAME,
                                           osn_name_focus=OSN_NAME_FOCUS,
                                           target_name_list=TARGET_NAME_LIST,
                                           number_of_folds=NUMBER_OF_FOLDS)

experiment.do_experiment()

EXPERIMENT_CONSTRUCTION_TYPE = dict()
EXPERIMENT_CONSTRUCTION_TYPE["add_branching_features"] = False
EXPERIMENT_CONSTRUCTION_TYPE["add_usergraph_features"] = False
EXPERIMENT_CONSTRUCTION_TYPE["add_temporal_features"] = False

EXPERIMENT_CONSTRUCTION_TYPE["baseline"] = "comments"

DATA_FOLDER = "Rizos/Reddit_News"
FEATURE_OSN_NAME_LIST = ["reddit"]  # If more than one, it is a JOINT experiment.
TARGET_OSN_NAME = "reddit"  # OSN targets.
OSN_NAME_FOCUS = "reddit"  # OSN-based timestamps.
TARGET_NAME_LIST = ["comments", "users", "score_wilson", "controversiality_wilson"]
NUMBER_OF_FOLDS = 10

experiment = DiscussionModellingExperiment(experiment_construction_dict=EXPERIMENT_CONSTRUCTION_TYPE,
                                           data_folder=DATA_FOLDER,
                                           feature_osn_name_list=FEATURE_OSN_NAME_LIST,
                                           target_osn_name=TARGET_OSN_NAME,
                                           osn_name_focus=OSN_NAME_FOCUS,
                                           target_name_list=TARGET_NAME_LIST,
                                           number_of_folds=NUMBER_OF_FOLDS)

experiment.do_experiment()

EXPERIMENT_CONSTRUCTION_TYPE = dict()
EXPERIMENT_CONSTRUCTION_TYPE["add_branching_features"] = False
EXPERIMENT_CONSTRUCTION_TYPE["add_usergraph_features"] = False
EXPERIMENT_CONSTRUCTION_TYPE["add_temporal_features"] = False

EXPERIMENT_CONSTRUCTION_TYPE["baseline"] = "users"

DATA_FOLDER = "Rizos/Reddit_News"
FEATURE_OSN_NAME_LIST = ["reddit"]  # If more than one, it is a JOINT experiment.
TARGET_OSN_NAME = "reddit"  # OSN targets.
OSN_NAME_FOCUS = "reddit"  # OSN-based timestamps.
TARGET_NAME_LIST = ["comments", "users", "score_wilson", "controversiality_wilson"]
NUMBER_OF_FOLDS = 10

experiment = DiscussionModellingExperiment(experiment_construction_dict=EXPERIMENT_CONSTRUCTION_TYPE,
                                           data_folder=DATA_FOLDER,
                                           feature_osn_name_list=FEATURE_OSN_NAME_LIST,
                                           target_osn_name=TARGET_OSN_NAME,
                                           osn_name_focus=OSN_NAME_FOCUS,
                                           target_name_list=TARGET_NAME_LIST,
                                           number_of_folds=NUMBER_OF_FOLDS)

experiment.do_experiment()

EXPERIMENT_CONSTRUCTION_TYPE = dict()
EXPERIMENT_CONSTRUCTION_TYPE["add_branching_features"] = False
EXPERIMENT_CONSTRUCTION_TYPE["add_usergraph_features"] = False
EXPERIMENT_CONSTRUCTION_TYPE["add_temporal_features"] = False

EXPERIMENT_CONSTRUCTION_TYPE["baseline"] = "comments_users"

DATA_FOLDER = "Rizos/Reddit_News"
FEATURE_OSN_NAME_LIST = ["reddit"]  # If more than one, it is a JOINT experiment.
TARGET_OSN_NAME = "reddit"  # OSN targets.
OSN_NAME_FOCUS = "reddit"  # OSN-based timestamps.
TARGET_NAME_LIST = ["comments", "users", "score_wilson", "controversiality_wilson"]
NUMBER_OF_FOLDS = 10

experiment = DiscussionModellingExperiment(experiment_construction_dict=EXPERIMENT_CONSTRUCTION_TYPE,
                                           data_folder=DATA_FOLDER,
                                           feature_osn_name_list=FEATURE_OSN_NAME_LIST,
                                           target_osn_name=TARGET_OSN_NAME,
                                           osn_name_focus=OSN_NAME_FOCUS,
                                           target_name_list=TARGET_NAME_LIST,
                                           number_of_folds=NUMBER_OF_FOLDS)

experiment.do_experiment()

EXPERIMENT_CONSTRUCTION_TYPE = dict()
EXPERIMENT_CONSTRUCTION_TYPE["add_branching_features"] = False
EXPERIMENT_CONSTRUCTION_TYPE["add_usergraph_features"] = False
EXPERIMENT_CONSTRUCTION_TYPE["add_temporal_features"] = False

EXPERIMENT_CONSTRUCTION_TYPE["baseline"] = "simple graph"

DATA_FOLDER = "Rizos/Reddit_News"
FEATURE_OSN_NAME_LIST = ["reddit"]  # If more than one, it is a JOINT experiment.
TARGET_OSN_NAME = "reddit"  # OSN targets.
OSN_NAME_FOCUS = "reddit"  # OSN-based timestamps.
TARGET_NAME_LIST = ["comments", "users", "score_wilson", "controversiality_wilson"]
NUMBER_OF_FOLDS = 10

experiment = DiscussionModellingExperiment(experiment_construction_dict=EXPERIMENT_CONSTRUCTION_TYPE,
                                           data_folder=DATA_FOLDER,
                                           feature_osn_name_list=FEATURE_OSN_NAME_LIST,
                                           target_osn_name=TARGET_OSN_NAME,
                                           osn_name_focus=OSN_NAME_FOCUS,
                                           target_name_list=TARGET_NAME_LIST,
                                           number_of_folds=NUMBER_OF_FOLDS)

experiment.do_experiment()

EXPERIMENT_CONSTRUCTION_TYPE = dict()
EXPERIMENT_CONSTRUCTION_TYPE["add_branching_features"] = True
EXPERIMENT_CONSTRUCTION_TYPE["add_usergraph_features"] = True
EXPERIMENT_CONSTRUCTION_TYPE["add_temporal_features"] = False

DATA_FOLDER = "Rizos/Reddit_News"
FEATURE_OSN_NAME_LIST = ["reddit"]  # If more than one, it is a JOINT experiment.
TARGET_OSN_NAME = "reddit"  # OSN targets.
OSN_NAME_FOCUS = "reddit"  # OSN-based timestamps.
TARGET_NAME_LIST = ["comments", "users", "score_wilson", "controversiality_wilson"]
NUMBER_OF_FOLDS = 10

experiment = DiscussionModellingExperiment(experiment_construction_dict=EXPERIMENT_CONSTRUCTION_TYPE,
                                           data_folder=DATA_FOLDER,
                                           feature_osn_name_list=FEATURE_OSN_NAME_LIST,
                                           target_osn_name=TARGET_OSN_NAME,
                                           osn_name_focus=OSN_NAME_FOCUS,
                                           target_name_list=TARGET_NAME_LIST,
                                           number_of_folds=NUMBER_OF_FOLDS)

experiment.do_experiment()

EXPERIMENT_CONSTRUCTION_TYPE = dict()
EXPERIMENT_CONSTRUCTION_TYPE["add_branching_features"] = True
EXPERIMENT_CONSTRUCTION_TYPE["add_usergraph_features"] = False
EXPERIMENT_CONSTRUCTION_TYPE["add_temporal_features"] = False

DATA_FOLDER = "Rizos/Reddit_News"
FEATURE_OSN_NAME_LIST = ["reddit"]  # If more than one, it is a JOINT experiment.
TARGET_OSN_NAME = "reddit"  # OSN targets.
OSN_NAME_FOCUS = "reddit"  # OSN-based timestamps.
TARGET_NAME_LIST = ["comments", "users", "score_wilson", "controversiality_wilson"]
NUMBER_OF_FOLDS = 10

experiment = DiscussionModellingExperiment(experiment_construction_dict=EXPERIMENT_CONSTRUCTION_TYPE,
                                           data_folder=DATA_FOLDER,
                                           feature_osn_name_list=FEATURE_OSN_NAME_LIST,
                                           target_osn_name=TARGET_OSN_NAME,
                                           osn_name_focus=OSN_NAME_FOCUS,
                                           target_name_list=TARGET_NAME_LIST,
                                           number_of_folds=NUMBER_OF_FOLDS)

experiment.do_experiment()

EXPERIMENT_CONSTRUCTION_TYPE = dict()
EXPERIMENT_CONSTRUCTION_TYPE["add_branching_features"] = False
EXPERIMENT_CONSTRUCTION_TYPE["add_usergraph_features"] = True
EXPERIMENT_CONSTRUCTION_TYPE["add_temporal_features"] = False

DATA_FOLDER = "Rizos/Reddit_News"
FEATURE_OSN_NAME_LIST = ["reddit"]  # If more than one, it is a JOINT experiment.
TARGET_OSN_NAME = "reddit"  # OSN targets.
OSN_NAME_FOCUS = "reddit"  # OSN-based timestamps.
TARGET_NAME_LIST = ["comments", "users", "score_wilson", "controversiality_wilson"]
NUMBER_OF_FOLDS = 10

experiment = DiscussionModellingExperiment(experiment_construction_dict=EXPERIMENT_CONSTRUCTION_TYPE,
                                           data_folder=DATA_FOLDER,
                                           feature_osn_name_list=FEATURE_OSN_NAME_LIST,
                                           target_osn_name=TARGET_OSN_NAME,
                                           osn_name_focus=OSN_NAME_FOCUS,
                                           target_name_list=TARGET_NAME_LIST,
                                           number_of_folds=NUMBER_OF_FOLDS)

experiment.do_experiment()

EXPERIMENT_CONSTRUCTION_TYPE = dict()
EXPERIMENT_CONSTRUCTION_TYPE["add_branching_features"] = False
EXPERIMENT_CONSTRUCTION_TYPE["add_usergraph_features"] = False
EXPERIMENT_CONSTRUCTION_TYPE["add_temporal_features"] = True

DATA_FOLDER = "Rizos/Reddit_News"
FEATURE_OSN_NAME_LIST = ["reddit"]  # If more than one, it is a JOINT experiment.
TARGET_OSN_NAME = "reddit"  # OSN targets.
OSN_NAME_FOCUS = "reddit"  # OSN-based timestamps.
TARGET_NAME_LIST = ["comments", "users", "score_wilson", "controversiality_wilson"]
NUMBER_OF_FOLDS = 10

experiment = DiscussionModellingExperiment(experiment_construction_dict=EXPERIMENT_CONSTRUCTION_TYPE,
                                           data_folder=DATA_FOLDER,
                                           feature_osn_name_list=FEATURE_OSN_NAME_LIST,
                                           target_osn_name=TARGET_OSN_NAME,
                                           osn_name_focus=OSN_NAME_FOCUS,
                                           target_name_list=TARGET_NAME_LIST,
                                           number_of_folds=NUMBER_OF_FOLDS)

experiment.do_experiment()

EXPERIMENT_CONSTRUCTION_TYPE = dict()
EXPERIMENT_CONSTRUCTION_TYPE["add_branching_features"] = True
EXPERIMENT_CONSTRUCTION_TYPE["add_usergraph_features"] = True
EXPERIMENT_CONSTRUCTION_TYPE["add_temporal_features"] = True

DATA_FOLDER = "Rizos/Reddit_News"
FEATURE_OSN_NAME_LIST = ["reddit"]  # If more than one, it is a JOINT experiment.
TARGET_OSN_NAME = "reddit"  # OSN targets.
OSN_NAME_FOCUS = "reddit"  # OSN-based timestamps.
TARGET_NAME_LIST = ["comments", "users", "score_wilson", "controversiality_wilson"]
NUMBER_OF_FOLDS = 10

experiment = DiscussionModellingExperiment(experiment_construction_dict=EXPERIMENT_CONSTRUCTION_TYPE,
                                           data_folder=DATA_FOLDER,
                                           feature_osn_name_list=FEATURE_OSN_NAME_LIST,
                                           target_osn_name=TARGET_OSN_NAME,
                                           osn_name_focus=OSN_NAME_FOCUS,
                                           target_name_list=TARGET_NAME_LIST,
                                           number_of_folds=NUMBER_OF_FOLDS)

experiment.do_experiment()
