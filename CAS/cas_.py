import numpy as np
import scipy.sparse as sp
from numba import njit, prange
from numba_stats import binom
from numba.typed import List


@njit
def _labels_array_to_matrix(labels):
    n_labels = np.max(labels) + 1
    n_nonoutliers = np.sum(labels >= 0)
    indptr = np.empty(n_labels + 1, dtype="int32")
    indices = np.empty(n_nonoutliers, dtype="int32")
    data = np.ones(n_nonoutliers, dtype="bool")
    labels_argsort = np.argsort(labels)
    next_index = 0
    for label in range(n_labels):
        indptr[label] = next_index
        label_nodes = np.arange(len(labels))[labels == label]
        n_label_nodes = len(label_nodes)
        indices[next_index : next_index + n_label_nodes] = label_nodes
        next_index += n_label_nodes
    indptr[-1] = next_index  # update last indptr entry
    return indptr, indices, data


@njit
def _get_degree(node, adjacency_indptr, adjacency_indices, adjacency_data):
    return np.sum(adjacency_data[adjacency_indptr[node] : adjacency_indptr[node + 1]])


@njit
def _get_community_degree(
    node,
    label,
    labels_indptr,
    labels_indices,
    adjacency_indptr,
    adjacency_indices,
    adjacency_data,
):
    neighbors = adjacency_indices[adjacency_indptr[node] : adjacency_indptr[node + 1]]
    edge_weights = adjacency_data[adjacency_indptr[node] : adjacency_indptr[node + 1]]
    community_degree = 0
    for neighbor, edge_weight in zip(neighbors, edge_weights):
        if neighbor in labels_indices[labels_indptr[label] : labels_indptr[label + 1]]:
            community_degree += edge_weight
    return community_degree


@njit
def _get_volume(nodes, adjacency_indptr, adjacency_indices, adjacency_data):
    volume = 0
    for node in nodes:
        volume += _get_degree(node, adjacency_indptr, adjacency_indices, adjacency_data)
    return volume


@njit
def _ief(
    node,
    label,
    label_volume,
    labels_indptr,
    labels_indices,
    adjacency_indptr,
    adjacency_indices,
    adjacency_data,
):
    degree = _get_degree(node, adjacency_indptr, adjacency_indices, adjacency_data)
    if degree == 0:
        return 0
    in_community_degree = _get_community_degree(
        node,
        label,
        labels_indptr,
        labels_indices,
        adjacency_indptr,
        adjacency_indices,
        adjacency_data,
    )
    return in_community_degree / degree


@njit
def _nief(
    node,
    label,
    label_volume,
    labels_indptr,
    labels_indices,
    adjacency_indptr,
    adjacency_indices,
    adjacency_data,
):
    degree = _get_degree(node, adjacency_indptr, adjacency_indices, adjacency_data)
    if degree == 0:
        return 0
    in_community_degree = _get_community_degree(
        node,
        label,
        labels_indptr,
        labels_indices,
        adjacency_indptr,
        adjacency_indices,
        adjacency_data,
    )
    return max(in_community_degree / degree - label_volume, 0)


@njit
def _p(
    node,
    label,
    label_volume,
    labels_indptr,
    labels_indices,
    adjacency_indptr,
    adjacency_indices,
    adjacency_data,
):
    degree = np.empty(1, dtype="float64")
    degree[0] = _get_degree(node, adjacency_indptr, adjacency_indices, adjacency_data)
    if degree[0] == 0:
        return 0
    in_community_degree = np.empty(1, dtype="float64")
    in_community_degree[0] = _get_community_degree(
        node,
        label,
        labels_indptr,
        labels_indices,
        adjacency_indptr,
        adjacency_indices,
        adjacency_data,
    )

    p = binom._cdf(in_community_degree, degree, label_volume)[0]
    return p


@njit
def _eliminate_zeros(indptr, indices, data):
    """Eliminate zeros from csr sparse matrix"""
    n_entries = np.sum(data)
    new_indptr = np.empty_like(indptr)
    new_indices = np.empty(n_entries, dtype=indices.dtype)
    new_data = np.ones(n_entries, dtype=data.dtype)
    next_index = 0
    for row in range(len(indptr) - 1):
        new_indptr[row] = next_index
        for data_id in range(indptr[row], indptr[row + 1]):
            if data[data_id] > 0:
                new_indices[next_index] = indices[data_id]
                next_index += 1
    new_indptr[-1] = next_index
    return new_indptr, new_indices, new_data


@njit
def _post_process_remove(
    labels_indptr,
    labels_indices,
    labels_data,
    adjacency_indptr,
    adjacency_indices,
    adjacency_data,
    threshold,
    max_rounds,
    cas,
    verbose=False,
):
    """Remove nodes from clusters if their cas score is below the provided threshold."""
    # TODO randomly select for ties
    graph_volume = np.sum(adjacency_data)
    for round_number in range(max_rounds):
        # Compute the cas score for each node to each of its labels
        cas_scores = np.empty_like(labels_data, dtype="float32")
        for label in range(len(labels_indptr) - 1):
            nodes = labels_indices[labels_indptr[label] : labels_indptr[label + 1]]
            label_volume = (
                _get_volume(nodes, adjacency_indptr, adjacency_indices, adjacency_data)
                / graph_volume
            )
            for data_offset, node in enumerate(nodes):
                cas_scores[labels_indptr[label] + data_offset] = cas(
                    node,
                    label,
                    label_volume,
                    labels_indptr,
                    labels_indices,
                    adjacency_indptr,
                    adjacency_indices,
                    adjacency_data,
                )

        # If no nodes need to be moved, break loop
        none_removed = True
        # Remove nodes (set label to 0)
        cas_order = np.argsort(cas_scores)
        for i in range(len(cas_scores)):
            remove_index = cas_order[i]
            if cas_scores[remove_index] < threshold:
                labels_data[remove_index] = 0
                none_removed = False
            else:
                break
        if verbose:
            print(f"Removed {i} in round {round_number}.")
        if none_removed:
            break
        # Update labels (eliminate_zeros)
        labels_indptr, labels_indices, labels_data = _eliminate_zeros(
            labels_indptr, labels_indices, labels_data
        )

    return labels_indptr, labels_indices, labels_data


# @njit
# def _get_nodes_to_move(
#     labels_csr_indptr,
#     labels_csr_indices,
#     labels_csc_indptr,
#     labels_csc_indices,
#     adjacency_indptr,
#     adjacency_indices,
#     adjacency_data,
#     graph_volume,
#     threshold,
#     max_per_round,
#     cas,
# ):
#     """Get the {max_per_round} node-label pairs with largest cas score distance from the threshold."""
#     cas_threshold_distance = np.zeros(max_per_round, dtype="float32")
#     nodes_to_move = np.empty(max_per_round, dtype=labels_csr_indices.dtype)
#     labels_for_nodes = np.empty_like(nodes_to_move)
#     move_action = np.empty_like(
#         nodes_to_move, dtype="bool"
#     )  # True for add, False for remove

#     number_of_moves = 0  # next empty index in the arrays
#     min_distance_to_threshold = 0  # cache the minimum distance to the threshold
#     min_distance_index = -1
#     label_volumes = np.empty(
#         len(labels_csr_indptr) - 1, dtype="float32"
#     )  # cache label volumes
#     for label in range(len(labels_csr_indptr) - 1):
#         label_nodes = labels_csr_indices[
#             labels_csr_indptr[label] : labels_csr_indptr[label + 1]
#         ]
#         label_volumes[label] = (
#             _get_volume(
#                 label_nodes, adjacency_indptr, adjacency_indices, adjacency_data
#             )
#             / graph_volume
#         )

#     # loop through every node and plausible label (a label is plausible if at least one neighbor is in that label)
#     for node in range(len(adjacency_indptr) - 1):
#         current_labels = set(
#             labels_csc_indices[labels_csc_indptr[node] : labels_csc_indptr[node + 1]]
#         )
#         plausible_labels = current_labels.copy()
#         neighbors = adjacency_indices[
#             adjacency_indptr[node] : adjacency_indptr[node + 1]
#         ]
#         for neighbor in neighbors:
#             for neighbor_label in labels_csc_indices[
#                 labels_csc_indptr[neighbor] : labels_csc_indptr[neighbor + 1]
#             ]:
#                 plausible_labels.add(neighbor_label)

#         for label in plausible_labels:
#             score = cas(
#                 node,
#                 label,
#                 label_volumes[label],
#                 labels_csr_indptr,
#                 labels_csr_indices,
#                 adjacency_indptr,
#                 adjacency_indices,
#                 adjacency_data,
#             )
#             distance_to_threshold = threshold - score if label in current_labels else score - threshold
#             if (
#                 distance_to_threshold > min_distance_to_threshold
#             ):  # move node into / out of label if score distance to theshold is sufficient
#                 if number_of_moves < max_per_round:  # empty spots
#                     cas_threshold_distance[number_of_moves] = distance_to_threshold
#                     nodes_to_move[number_of_moves] = node
#                     labels_for_nodes[number_of_moves] = label
#                     move_action[number_of_moves] = label not in current_labels
#                     number_of_moves += 1
#                     if (
#                         number_of_moves == max_per_round
#                     ):  # array is full, compute min distance
#                         min_distance_index = np.argmin(cas_threshold_distance)
#                         min_distance_to_threshold = cas_threshold_distance[
#                             min_distance_index
#                         ]

#                 else:  # replace current min distance
#                     cas_threshold_distance[min_distance_index] = distance_to_threshold
#                     nodes_to_move[min_distance_index] = node
#                     labels_for_nodes[min_distance_index] = label
#                     move_action[min_distance_index] = label not in current_labels
#                     # update min pointers
#                     min_distance_index = np.argmin(cas_threshold_distance)
#                     min_distance_to_threshold = cas_threshold_distance[
#                         min_distance_index
#                     ]

#     return number_of_moves, nodes_to_move, labels_for_nodes, move_action


@njit
def _transpose_sparse(indptr, indices, indices_dim):
    new_indptr = np.zeros(indices_dim + 1, dtype=indptr.dtype)
    new_indices = np.full(len(indices), -1, dtype=indices.dtype)
    # populate new_indptr
    for row in range(len(indptr) - 1):
        for col in indices[indptr[row] : indptr[row + 1]]:
            # increase new_indptr by one for every column after this one
            for i in range(col + 1, len(new_indptr)):
                new_indptr[i] += 1
    # populate new_indices. indptr already come in sorted order
    for row in range(len(indptr) - 1):
        for col in indices[indptr[row] : indptr[row + 1]]:
            for i in range(new_indptr[col], new_indptr[col + 1]):
                if new_indices[i] == -1:
                    new_indices[i] = row
                    break
    return new_indptr, new_indices


# @njit
# def _update_labels(
#     labels_indptr, labels_indices, nodes_to_move, labels_for_nodes, move_action
# ):
#     change_in_length = np.sum(move_action) - np.sum(~move_action)
#     new_indptr = np.empty_like(labels_indptr)
#     new_indices = np.empty(
#         len(labels_indices) + change_in_length, dtype=labels_indices.dtype
#     )
#     move_index_range = np.arange(len(move_action))

#     next_index = 0
#     for label in range(len(labels_indptr) - 1):
#         current_nodes = set(
#             labels_indices[labels_indptr[label] : labels_indptr[label + 1]]
#         )
#         move_indices = move_index_range[labels_for_nodes == label]
#         for move_index in move_indices:
#             if move_action[move_index]:
#                 current_nodes.add(nodes_to_move[move_index])
#             else:
#                 current_nodes.remove(nodes_to_move[move_index])
#         # print(current_nodes)
#         new_nodes = np.array(list(current_nodes))
#         new_nodes.sort()
#         # update
#         n_nodes = len(new_nodes)
#         new_indptr[label] = next_index
#         new_indices[next_index : next_index + n_nodes] = new_nodes
#         next_index += n_nodes
#     new_indptr[-1] = next_index
#     return new_indptr, new_indices


@njit
def _get_new_labels(
    labels_csr_indptr,
    labels_csr_indices,
    labels_csc_indptr,
    labels_csc_indices,
    adjacency_indptr,
    adjacency_indices,
    adjacency_data,
    graph_volume,
    threshold,
    cas,
):
    """Make an lil matrix of (node x label)"""
    lil_node_labels = List(
        [np.empty(0, dtype="int32") for _ in range(len(adjacency_indptr) - 1)]
    )
    label_volumes = np.empty(
        len(labels_csr_indptr) - 1, dtype="float32"
    )  # cache label volumes
    for label in range(len(labels_csr_indptr) - 1):
        label_nodes = labels_csr_indices[
            labels_csr_indptr[label] : labels_csr_indptr[label + 1]
        ]
        label_volumes[label] = (
            _get_volume(
                label_nodes, adjacency_indptr, adjacency_indices, adjacency_data
            )
            / graph_volume
        )

    # For each node, compute the assigned labels (everything above threshold)
    for node in range(len(adjacency_indptr) - 1):
        plausible_labels = set(
            labels_csc_indices[labels_csc_indptr[node] : labels_csc_indptr[node + 1]]
        )
        previous_labels = plausible_labels.copy()
        neighbors = adjacency_indices[
            adjacency_indptr[node] : adjacency_indptr[node + 1]
        ]
        for neighbor in neighbors:
            plausible_labels.update(
                labels_csc_indices[
                    labels_csc_indptr[neighbor] : labels_csc_indptr[neighbor + 1]
                ]
            )
        plausible_labels = np.array(list(plausible_labels))
        cas_scores = np.empty_like(plausible_labels, dtype="float64")
        for i in range(len(plausible_labels)):
            cas_scores[i] = cas(
                node,
                plausible_labels[i],
                label_volumes[plausible_labels[i]],
                labels_csr_indptr,
                labels_csr_indices,
                adjacency_indptr,
                adjacency_indices,
                adjacency_data,
            )
        new_labels = plausible_labels[cas_scores >= threshold]
        new_labels.sort()
        lil_node_labels[node] = new_labels
    return lil_node_labels


@njit
def _lil_to_csc(new_labels):
    """Take a list of lists (column x rows) and return csr and csc data"""
    total_labels = 0
    for i in range(len(new_labels)):
        total_labels += len(new_labels[i])
    csc_indptr = np.empty(len(new_labels) + 1, dtype="int32")
    csc_indices = np.empty(total_labels, dtype="int32")
    next_index = 0
    for col in range(len(new_labels)):
        csc_indptr[col] = next_index
        rows = new_labels[col]
        next_index = next_index + len(rows)
        csc_indices[csc_indptr[col] : next_index] = rows
    csc_indptr[-1] = next_index
    return csc_indptr, csc_indices


@njit
def _post_process(
    labels_csr_indptr,
    labels_csr_indices,
    labels_csr_data,
    adjacency_indptr,
    adjacency_indices,
    adjacency_data,
    threshold,
    max_rounds,
    cas,
    verbose=False,
):
    """Remove nodes from clusters if their cas score is below the threshold and add nodes
    to clusters if their cas is above the threshold. Nodes are added or removed in
    the order of greatest cas distance to the threshold.
    """
    graph_volume = np.sum(adjacency_data)
    n_labels = len(labels_csr_indptr) - 1
    labels_csc_indptr, labels_csc_indices = _transpose_sparse(
        labels_csr_indptr, labels_csr_indices, len(adjacency_indptr) - 1
    )
    # Break conditions. We can get stuck moving a few nodes back and forth so track two rounds.
    two_rounds_ago_indptr = np.array([-1], dtype="int32")
    two_rounds_ago_indices = np.array([-1], dtype="int32")
    last_round_indptr = np.array([-1], dtype="int32")
    last_round_indices = np.array([-1], dtype="int32")

    for round_number in range(max_rounds):
        # Get nodes to move
        new_labels = _get_new_labels(
            labels_csr_indptr,
            labels_csr_indices,
            labels_csc_indptr,
            labels_csc_indices,
            adjacency_indptr,
            adjacency_indices,
            adjacency_data,
            graph_volume,
            threshold,
            cas,
        )

        labels_csc_indptr, labels_csc_indices = _lil_to_csc(new_labels)
        labels_csr_indptr, labels_csr_indices = _transpose_sparse(
            labels_csc_indptr, labels_csc_indices, n_labels
        )

        if verbose:
            print(f"\tRound {round_number}")

        # Break if nothing changed this round, or if we are back where we were two rounds ago
        if (
            np.array_equal(labels_csr_indptr, last_round_indptr)
            and np.array_equal(labels_csr_indices, last_round_indices)
        ) or (
            np.array_equal(labels_csr_indptr, two_rounds_ago_indptr)
            and np.array_equal(labels_csr_indices, two_rounds_ago_indices)
        ):
            break
        two_rounds_ago_indptr = last_round_indptr
        two_rounds_ago_indices = last_round_indices
        last_round_indptr = labels_csr_indptr
        last_round_indices = labels_csr_indices

    labels_csr_data = np.ones(len(labels_csr_indices), dtype="bool")
    return labels_csr_indptr, labels_csr_indices, labels_csr_data


@njit
def _sparse_labels_to_numpy(labels_indptr, labels_indices, n_nodes):
    labels_csc_indptr, labels_csc_indices = _transpose_sparse(
        labels_indptr, labels_indices, n_nodes
    )
    new_labels = np.empty(len(labels_csc_indptr) - 1, dtype="int32")
    for node in range(len(labels_csc_indptr) - 1):
        if labels_csc_indptr[node] == labels_csc_indptr[node + 1]:
            new_labels[node] = -1
        elif labels_csc_indptr[node + 1] - labels_csc_indptr[node] == 1:
            new_labels[node] = labels_csc_indices[labels_csc_indptr[node]]
        else:
            raise ValueError(
                "Cannot convert a labels matrix with more than one label per node to a numpy array."
            )
    return new_labels


class CASPostProcesser:
    def __init__(
        self,
        score="nief",
        threshold=0.5,
        max_rounds=1000,
        only_remove=True,
        sparse_output=False,
        relabel_clusters=True,
        verbose=False,
    ):
        self.score = score
        self.threshold = threshold
        self.max_rounds = max_rounds
        self.only_remove = only_remove
        self.sparse_output = sparse_output
        self.relabel_clusters = relabel_clusters
        self.verbose = verbose

    def _validate_parameters(self):
        if self.score == "ief":
            self.cas = _ief
        elif self.score == "nief":
            self.cas = _nief
        elif self.score == "p":
            self.cas = _p
        else:
            raise ValueError(f"score must be in ['ief', 'nief', 'p']")

        if self.threshold < 0.0 or self.threshold > 1.0:
            raise ValueError(f"threshold must be between 0.0 and 1.0")

        if not isinstance(self.max_rounds, int):
            if self.max_rounds % 1 != 0:
                raise ValueError("max_rounds must be a whole number")
            try:
                # convert other types of int to python int
                self.max_rounds = int(self.max_rounds)
            except ValueError:
                raise ValueError("max_rounds must be an int")
        if self.max_rounds < 1:
            raise ValueError("max_rounds must be positive")

        if not isinstance(self.only_remove, bool):
            raise ValueError("only_remove must be a bool")

        if not isinstance(self.sparse_output, bool):
            raise ValueError("sparse_output must be a bool")

    def fit(self, labels, adjacency):
        self._validate_parameters()
        return_as_numpy = False
        if isinstance(labels, np.ndarray):
            if labels.ndim != 1:
                raise ValueError(f"Expected 1d numpy array. Got {labels.ndim} dims.")
            labels_indptr, labels_indices, labels_data = _labels_array_to_matrix(labels)
            return_as_numpy = (
                self.only_remove and not self.sparse_output
            )  # If passed a numpy array and only removing, return a numpy array
        elif sp.issparse(labels):
            labels = labels.tocsr()
            labels_indptr = labels.indptr
            labels_indices = labels.indices
            labels_data = labels.data
        else:
            raise ValueError(
                f"Expected labels to be a scipy sparse array of numpy array. Got {type(labels)}"
            )

        if sp.issparse(adjacency):
            adjacency = adjacency.tocsr()
        else:
            raise ValueError(
                f"Adjacnecy must be a sparse matrix. Got {type(adjacency)}."
            )

        if self.only_remove:
            labels_indptr, labels_indices, labels_data = _post_process_remove(
                labels_indptr,
                labels_indices,
                labels_data,
                adjacency.indptr,
                adjacency.indices,
                adjacency.data,
                self.threshold,
                self.max_rounds,
                self.cas,
                self.verbose,
            )
        else:
            labels_indptr, labels_indices, labels_data = _post_process(
                labels_indptr,
                labels_indices,
                labels_data,
                adjacency.indptr,
                adjacency.indices,
                adjacency.data,
                self.threshold,
                self.max_rounds,
                self.cas,
                self.verbose,
            )

        labels = sp.csr_matrix(
            (labels_data, labels_indices, labels_indptr),
            shape=(len(labels_indptr) - 1, adjacency.shape[0]),
            dtype="bool",
        )
        labels.data[:] = (
            True  # Sometime some entries are flipped to false, don't know why.
        )

        if self.relabel_clusters:
            non_empty_cluster = labels.getnnz(1) > 0
            self.old_cluster_ids = np.arange(labels.shape[0])[non_empty_cluster]
            labels = labels[non_empty_cluster]

        if return_as_numpy:
            labels = _sparse_labels_to_numpy(
                labels.indptr, labels.indices, adjacency.shape[0]
            )

        self.labels_ = labels
        return self

    def fit_predict(self, labels, adjacency):
        self.fit(labels, adjacency)
        return self.labels_
