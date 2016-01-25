__author__ = 'Georgios Rizos (georgerizos@iti.gr)'


def append_feature_value(feature_to_list, feature_name, feature_value):
    feature_to_list[feature_name].append(feature_value)


def update_feature_value(feature_array, i, j, value):
    feature_array[i, j] = value


def replicate_feature_value(feature_array, i, j):
    feature_array[i, j] = feature_array[i-1, j]
