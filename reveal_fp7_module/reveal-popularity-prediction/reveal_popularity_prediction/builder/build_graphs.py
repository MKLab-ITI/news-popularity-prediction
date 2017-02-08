__author__ = 'Georgios Rizos (georgerizos@iti.gr)'

import heapq
import collections

import numpy as np
import scipy.sparse as spsp

from reveal_popularity_prediction.builder.collect.youtube import extract as youtube_extract
from reveal_popularity_prediction.builder.collect.reddit import extract as reddit_extract


def get_snapshot_graphs(social_context,
                        upper_timestamp,
                        platform):
    if platform == "YouTube":
        comment_generator = youtube_extract.comment_generator
        extract_comment_name = youtube_extract.extract_comment_name
        extract_parent_comment_name = youtube_extract.extract_parent_comment_name
        extract_timestamp = youtube_extract.extract_timestamp
        extract_user_name = youtube_extract.extract_user_name
        calculate_targets = youtube_extract.calculate_targets
        extract_title = youtube_extract.extract_title
        anonymous_coward_name = "REVEAL_FP7_anonymous_youtube_user"
    elif platform == "Reddit":
        comment_generator = reddit_extract.comment_generator
        extract_comment_name = reddit_extract.extract_comment_name
        extract_parent_comment_name = reddit_extract.extract_parent_comment_name
        extract_timestamp = reddit_extract.extract_timestamp
        extract_user_name = reddit_extract.extract_user_name
        calculate_targets = reddit_extract.calculate_targets
        extract_title = reddit_extract.extract_title
        anonymous_coward_name = "[deleted]"
    else:
        print("Invalid platform.")
        raise RuntimeError

    comment_gen = comment_generator(social_context)
    initial_post = next(comment_gen)
    initial_post_timestamp = extract_timestamp(initial_post)

    post_lifetime_to_assessment = upper_timestamp - initial_post_timestamp

    if post_lifetime_to_assessment < 0.0:
        print("Post timestamp smaller than assessment timestamp. Bad data. Continuing.")
    elif post_lifetime_to_assessment > 604800:
        # Post is older than a week.
        return None, None, None
    else:
        pass

    comment_gen = comment_generator(social_context)

    comment_name_set,\
    user_name_set,\
    within_discussion_comment_anonymize,\
    within_discussion_user_anonymize,\
    within_discussion_anonymous_coward = within_discussion_comment_and_user_anonymization(comment_gen,
                                                                                          extract_comment_name,
                                                                                          extract_user_name,
                                                                                          anonymous_coward_name)

    safe_comment_gen = safe_comment_generator(social_context,
                                              comment_generator=comment_generator,
                                              within_discussion_comment_anonymize=within_discussion_comment_anonymize,
                                              extract_comment_name=extract_comment_name,
                                              extract_parent_comment_name=extract_parent_comment_name,
                                              extract_timestamp=extract_timestamp,
                                              safe=True)

    snapshot_graphs = form_snapshot_graphs(safe_comment_gen,
                                           comment_name_set,
                                           user_name_set,
                                           upper_timestamp,
                                           extract_timestamp,
                                           extract_comment_name,
                                           extract_parent_comment_name,
                                           extract_user_name,
                                           within_discussion_comment_anonymize,
                                           within_discussion_user_anonymize,
                                           within_discussion_anonymous_coward)
    if snapshot_graphs is None:
        return None, None, None

    try:
        all_targets = calculate_targets(social_context,
                                        comment_name_set,
                                        user_name_set,
                                        within_discussion_anonymous_coward)
    except KeyError:
        return None, None, None

    targets = dict()
    targets["comment_count"] = all_targets["comments"]
    targets["user_count"] = all_targets["users"]
    targets["upvote_count"] = all_targets["number_of_upvotes"]
    targets["downvote_count"] = all_targets["number_of_downvotes"]
    targets["score"] = all_targets["score_wilson"]
    targets["controversiality"] = all_targets["controversiality_wilson"]

    title = extract_title(social_context["initial_post"])

    return snapshot_graphs, targets, title


def form_snapshot_graphs(safe_comment_gen,
                         comment_name_set,
                         user_name_set,
                         upper_timestamp,
                         extract_timestamp,
                         extract_comment_name,
                         extract_parent_comment_name,
                         extract_user_name,
                         within_discussion_comment_anonymize,
                         within_discussion_user_anonymize,
                         within_discussion_anonymous_coward):
    # Keep only the social context until the tweet timestamp.
    comment_list = list()
    timestamp_list = list()
    try:
        initial_post = next(safe_comment_gen)
    except StopIteration:
        return None
    initial_timestamp = extract_timestamp(initial_post)
    comment_list.append(initial_post)
    timestamp_list.append(initial_timestamp)
    for comment in safe_comment_gen:
        comment_timestamp = extract_timestamp(comment)

        # Sanitize comment timestamps.
        if comment_timestamp < timestamp_list[-1]:
            comment_timestamp = timestamp_list[-1]

        if comment_timestamp > upper_timestamp:
            break
        else:
            comment_list.append(comment)
            timestamp_list.append(comment_timestamp)

    # Decide the snapshot timestamps.
    snapshot_timestamps = decide_snapshot_timestamps(timestamp_list,
                                                     max_number_of_snapshots_including_zero=10)
    # print(snapshot_timestamps)

    snapshot_gen = snapshot_generator(comment_list,
                                      timestamp_list,
                                      snapshot_timestamps,
                                      comment_name_set,
                                      user_name_set,
                                      extract_comment_name,
                                      extract_parent_comment_name,
                                      extract_user_name,
                                      within_discussion_comment_anonymize,
                                      within_discussion_user_anonymize,
                                      within_discussion_anonymous_coward)
    snapshot_graphs = [snapshot_graph_dict for snapshot_graph_dict in snapshot_gen]

    return snapshot_graphs


def decide_snapshot_timestamps(timestamp_list,
                               max_number_of_snapshots_including_zero):
    discrete_timestamp_list = sorted(set(timestamp_list))
    discrete_timestamp_count = len(discrete_timestamp_list)
    if discrete_timestamp_count < max_number_of_snapshots_including_zero:
        max_number_of_snapshots_including_zero = discrete_timestamp_count

    snapshot_timestamps = np.linspace(0,
                                      len(discrete_timestamp_list)-1,
                                      num=max_number_of_snapshots_including_zero,
                                      endpoint=True)

    snapshot_timestamps = np.rint(snapshot_timestamps)
    snapshot_timestamps = list(snapshot_timestamps)
    snapshot_timestamps = [discrete_timestamp_list[int(t)] for t in snapshot_timestamps]

    # discrete_timestamp_list = sorted(set(timestamp_list))
    # discrete_timestamp_count = len(discrete_timestamp_list)
    # if discrete_timestamp_count < max_number_of_snapshots_including_zero:
    #     max_number_of_snapshots_including_zero = discrete_timestamp_count
    #
    # snapshot_timestamps = np.linspace(0,
    #                                   len(discrete_timestamp_list)-1,
    #                                   num=max_number_of_snapshots_including_zero,
    #                                   endpoint=True)
    #
    # snapshot_timestamps = np.rint(snapshot_timestamps)
    # snapshot_timestamps = list(snapshot_timestamps)
    # snapshot_timestamps = [discrete_timestamp_list[int(t)] for t in snapshot_timestamps]

    return snapshot_timestamps


def snapshot_generator(comment_list,
                       timestamp_list,
                       snapshot_timestamps,
                       comment_name_set,
                       user_name_set,
                       extract_comment_name,
                       extract_parent_comment_name,
                       extract_user_name,
                       within_discussion_comment_anonymize,
                       within_discussion_user_anonymize,
                       within_discussion_anonymous_coward):
    # Initialization.
    comment_tree = spsp.dok_matrix((len(comment_name_set),
                                    len(comment_name_set)),
                                   dtype=np.float64)

    user_graph = spsp.dok_matrix((len(user_name_set),
                                  len(user_name_set)),
                                 dtype=np.float64)
    comment_id_to_user_id = dict()
    comment_id_to_user_id[0] = 0

    user_name_list = list()

    initial_post = comment_list[0]
    initial_post_timestamp = timestamp_list[0]

    user_name = extract_user_name(initial_post)
    if user_name is not None:
        user_name_list.append(user_name)

    # snapshot_graph_dict = dict()
    # snapshot_graph_dict["comment_tree"] = spsp.coo_matrix(comment_tree)
    # snapshot_graph_dict["user_graph"] = spsp.coo_matrix(user_graph)
    # snapshot_graph_dict["timestamp_list"] = [initial_post_timestamp]
    # snapshot_graph_dict["user_set"] = set(user_name_list)
    # yield snapshot_graph_dict

    snapshot_counter = 0
    for counter in range(len(comment_list)):
        comment = comment_list[counter]
        comment_timestamp = timestamp_list[counter]

        user_name = extract_user_name(comment)
        if user_name is not None:
            user_name_list.append(user_name)

        if comment_timestamp > snapshot_timestamps[snapshot_counter]:
            snapshot_graph_dict = dict()
            snapshot_graph_dict["comment_tree"] = spsp.coo_matrix(comment_tree)
            snapshot_graph_dict["user_graph"] = spsp.coo_matrix(user_graph)
            snapshot_graph_dict["timestamp_list"] = timestamp_list[:counter+1]
            snapshot_graph_dict["user_set"] = set(user_name_list)
            yield snapshot_graph_dict

            snapshot_counter += 1
            if snapshot_counter >= len(snapshot_timestamps):
                raise StopIteration

        comment_tree,\
        user_graph,\
        comment_id,\
        parent_comment_id,\
        commenter_id,\
        parent_commenter_id,\
        comment_id_to_user_id = update_discussion_and_user_graphs(comment,
                                                                  extract_comment_name,
                                                                  extract_parent_comment_name,
                                                                  extract_user_name,
                                                                  comment_tree,
                                                                  user_graph,
                                                                  within_discussion_comment_anonymize,
                                                                  within_discussion_user_anonymize,
                                                                  within_discussion_anonymous_coward,
                                                                  comment_id_to_user_id)

    snapshot_graph_dict = dict()
    snapshot_graph_dict["comment_tree"] = spsp.coo_matrix(comment_tree)
    snapshot_graph_dict["user_graph"] = spsp.coo_matrix(user_graph)
    snapshot_graph_dict["timestamp_list"] = timestamp_list
    snapshot_graph_dict["user_set"] = set(user_name_list)
    yield snapshot_graph_dict


def update_discussion_and_user_graphs(comment,
                                      extract_comment_name,
                                      extract_parent_comment_name,
                                      extract_user_name,
                                      discussion_tree,
                                      user_graph,
                                      within_discussion_comment_anonymize,
                                      within_discussion_user_anonymize,
                                      within_discussion_anonymous_coward,
                                      comment_id_to_user_id):
    """
    Update the discussion tree and the user graph for a discussion.

    Does not handle the initial post.
    """
    # Extract comment.
    comment_name = extract_comment_name(comment)
    comment_id = within_discussion_comment_anonymize[comment_name]

    # Extract commenter.
    commenter_name = extract_user_name(comment)
    commenter_id = within_discussion_user_anonymize[commenter_name]

    # Update the comment to user map.
    comment_id_to_user_id[comment_id] = commenter_id

    # Check if this is a comment to the original post or to another comment.
    try:
        parent_comment_name = extract_parent_comment_name(comment)
    except KeyError:
        parent_comment_name = None
    if parent_comment_name is None:
        # The parent is the original post.
        parent_comment_id = 0
        parent_commenter_id = 0
    else:
        # The parent is another comment.
        try:
            parent_comment_id = within_discussion_comment_anonymize[parent_comment_name]
        except KeyError:
            print("Parent comment does not exist. Comment name: ", comment_name)
            raise RuntimeError

        # Extract parent comment in order to update user graph.
        try:
            parent_commenter_id = comment_id_to_user_id[parent_comment_id]
        except KeyError:
            print("Parent user does not exist. Comment name: ", comment_name)
            raise RuntimeError

    try:
        if within_discussion_anonymous_coward is None:
            if user_graph[commenter_id, parent_commenter_id] > 0.0:
                user_graph[commenter_id, parent_commenter_id] += 1.0
            elif user_graph[parent_commenter_id, commenter_id] > 0.0:
                user_graph[parent_commenter_id, commenter_id] += 1.0
            else:
                user_graph[commenter_id, parent_commenter_id] = 1.0
        else:
            if within_discussion_anonymous_coward not in (parent_commenter_id,
                                                          commenter_id):
                if user_graph[commenter_id, parent_commenter_id] > 0.0:
                    user_graph[commenter_id, parent_commenter_id] += 1.0
                elif user_graph[parent_commenter_id, commenter_id] > 0.0:
                    user_graph[parent_commenter_id, commenter_id] += 1.0
                else:
                    user_graph[commenter_id, parent_commenter_id] = 1.0
    except IndexError:
        print("Index error: ", user_graph.shape, commenter_id, parent_commenter_id)
        raise RuntimeError

    # Update discussion radial tree.
    discussion_tree[comment_id, parent_comment_id] = 1

    return discussion_tree,\
           user_graph,\
           comment_id,\
           parent_comment_id,\
           commenter_id,\
           parent_commenter_id,\
           comment_id_to_user_id


def safe_comment_generator(document,
                           comment_generator,
                           within_discussion_comment_anonymize,
                           extract_comment_name,
                           extract_parent_comment_name,
                           extract_timestamp,
                           safe):
    """
    We do this in order to correct for nonsensical or missing timestamps.
    """
    if not safe:
        comment_gen = comment_generator(document)

        initial_post = next(comment_gen)
        yield initial_post

        comment_list = sorted(comment_gen, key=extract_timestamp)
        for comment in comment_list:
            yield comment
    else:
        comment_id_to_comment = dict()

        comment_gen = comment_generator(document)

        initial_post = next(comment_gen)
        yield initial_post

        initial_post_id = within_discussion_comment_anonymize[extract_comment_name(initial_post)]

        comment_id_to_comment[initial_post_id] = initial_post

        if initial_post_id != 0:
            print("This cannot be.")
            raise RuntimeError

        comment_list = list(comment_gen)
        children_dict = collections.defaultdict(list)
        for comment in comment_list:
            # Anonymize comment.
            comment_name = extract_comment_name(comment)
            comment_id = within_discussion_comment_anonymize[comment_name]

            parent_comment_name = extract_parent_comment_name(comment)
            if parent_comment_name is None:
                parent_comment_id = 0
            else:
                parent_comment_id = within_discussion_comment_anonymize[parent_comment_name]

            comment_id_to_comment[comment_id] = comment

            # Update discussion tree.
            children_dict[parent_comment_id].append(comment_id)

        # Starting from the root/initial post, we get the children and we put them in a priority queue.
        priority_queue = list()

        children = set(children_dict[initial_post_id])
        for child in children:
            comment = comment_id_to_comment[child]
            timestamp = extract_timestamp(comment)
            heapq.heappush(priority_queue, (timestamp, (child, comment)))

        # We iteratively yield the topmost priority comment and add to the priority list the children of that comment.
        while True:
            # If priority list empty, we stop.
            if len(priority_queue) == 0:
                break

            t = heapq.heappop(priority_queue)
            comment = t[1][1]
            yield comment

            children = set(children_dict[int(t[1][0])])
            for child in children:
                comment = comment_id_to_comment[child]
                timestamp = extract_timestamp(comment)
                heapq.heappush(priority_queue, (timestamp, (child, comment)))


def within_discussion_comment_and_user_anonymization(comment_gen,
                                                     extract_comment_name,
                                                     extract_user_name,
                                                     anonymous_coward_name):
    """
    Reads all distinct users and comments in a single document and anonymizes them. Roots are 0.
    """
    comment_name_set = list()
    user_name_set = list()

    append_comment_name = comment_name_set.append
    append_user_name = user_name_set.append

    ####################################################################################################################
    # Extract comment and user name from the initial post.
    ####################################################################################################################
    initial_post = next(comment_gen)

    initial_post_name = extract_comment_name(initial_post)
    op_name = extract_user_name(initial_post)

    append_comment_name(initial_post_name)
    append_user_name(op_name)

    ####################################################################################################################
    # Iterate over all comments.
    ####################################################################################################################
    for comment in comment_gen:
        comment_name = extract_comment_name(comment)
        commenter_name = extract_user_name(comment)

        append_comment_name(comment_name)
        append_user_name(commenter_name)

    ####################################################################################################################
    # Perform anonymization.
    ####################################################################################################################
    # Remove duplicates and then remove initial post name because we want to give it id 0.
    comment_name_set = set(comment_name_set)
    comment_name_set.remove(initial_post_name)

    # Remove duplicates and then remove OP because we want to give them id 0.
    user_name_set = set(user_name_set)
    user_name_set.remove(op_name)

    # Anonymize.
    within_discussion_comment_anonymize = dict(zip(comment_name_set, range(1, len(comment_name_set) + 1)))
    within_discussion_comment_anonymize[initial_post_name] = 0  # Initial Post gets id 0.

    within_discussion_user_anonymize = dict(zip(user_name_set, range(1, len(user_name_set) + 1)))
    within_discussion_user_anonymize[op_name] = 0            # Original Poster gets id 0.

    comment_name_set.add(initial_post_name)
    user_name_set.add(op_name)

    if anonymous_coward_name is not None:
        # if op_name == anonymous_coward_name:
            # print("The Original Poster is Anonymous.")
        try:
            within_discussion_anonymous_coward = within_discussion_user_anonymize[anonymous_coward_name]
        except KeyError:
            within_discussion_anonymous_coward = None
    else:
        within_discussion_anonymous_coward = None

    return comment_name_set,\
           user_name_set,\
           within_discussion_comment_anonymize,\
           within_discussion_user_anonymize,\
           within_discussion_anonymous_coward
